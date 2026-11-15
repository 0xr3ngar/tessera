//! Ranks on one box, exchanging through a mapped file.

use crate::{Error, Transport};
use std::ffi::{c_int, c_void};
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
}

const PROT_READ_WRITE: c_int = 3;
const MAP_SHARED: c_int = 1;
/// Marks a file rank 0 laid out this run.
const READY: u64 = 0x7465_7373_6572_6131;
/// How long a rank waits for the others.
const PATIENCE: Duration = Duration::from_secs(10);

/// Two slots per rank: a write never races a read.
pub struct Shm {
    rank: u32,
    ranks: u32,
    /// Bytes per slot. Header words come first, then slots.
    capacity: usize,
    base: *mut u8,
    len: usize,
    /// Exchanges done so far.
    generation: u64,
    _file: File,
}

// Shared memory. Each rank writes its own slot only.
unsafe impl Send for Shm {}

impl Shm {
    fn words(ranks: u32) -> usize {
        1 + 3 * ranks as usize
    }

    fn bytes(ranks: u32, capacity: usize) -> usize {
        Self::words(ranks) * 8 + 2 * ranks as usize * capacity
    }

    /// Rank 0 lays out `path`. The others join as `rank`.
    pub fn join(path: &Path, rank: u32, ranks: u32, capacity: usize) -> Result<Self, Error> {
        let len = Self::bytes(ranks, capacity);
        let start = Instant::now();
        let file = loop {
            if rank == 0 {
                let _ = std::fs::remove_file(path);
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create_new(true)
                    .open(path)?;
                file.set_len(len as u64)?;
                break file;
            }
            match OpenOptions::new().read(true).write(true).open(path) {
                Ok(file) if file.metadata()?.len() as usize == len => break file,
                Ok(_) | Err(_) if start.elapsed() < PATIENCE => {
                    thread::sleep(Duration::from_millis(5))
                }
                Ok(_) => {
                    return Err(Error::Join(format!(
                        "{} has the wrong size",
                        path.display()
                    )));
                }
                Err(e) => return Err(e.into()),
            }
        };
        let base = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ_WRITE,
                MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if base as isize == -1 {
            return Err(Error::Io(std::io::Error::last_os_error()));
        }
        let shm = Self {
            rank,
            ranks,
            capacity,
            base: base as *mut u8,
            len,
            generation: 0,
            _file: file,
        };
        if rank == 0 {
            shm.word(0).store(READY, Ordering::Release);
        } else {
            while shm.word(0).load(Ordering::Acquire) != READY {
                if start.elapsed() > PATIENCE {
                    return Err(Error::Join("rank 0 never laid the file out".into()));
                }
                thread::sleep(Duration::from_millis(5));
            }
        }
        Ok(shm)
    }

    fn word(&self, at: usize) -> &AtomicU64 {
        // The header words are aligned and shared by design.
        unsafe { &*(self.base.add(8 * at) as *const AtomicU64) }
    }

    fn generation_of(&self, rank: u32) -> &AtomicU64 {
        self.word(1 + 3 * rank as usize)
    }

    /// The length word for `rank`'s slot of this parity.
    fn len_of(&self, rank: u32, parity: usize) -> &AtomicU64 {
        self.word(2 + 3 * rank as usize + parity)
    }

    fn slot(&self, rank: u32, parity: usize) -> *mut u8 {
        let at = Self::words(self.ranks) * 8 + (2 * rank as usize + parity) * self.capacity;
        unsafe { self.base.add(at) }
    }
}

impl Drop for Shm {
    fn drop(&mut self) {
        unsafe { munmap(self.base as *mut c_void, self.len) };
    }
}

impl Transport for Shm {
    fn rank(&self) -> u32 {
        self.rank
    }

    fn ranks(&self) -> u32 {
        self.ranks
    }

    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, Error> {
        if mine.len() > self.capacity {
            let what = format!("{} bytes over a {} byte slot", mine.len(), self.capacity);
            return Err(Error::Join(what));
        }
        let next = self.generation + 1;
        let parity = (next % 2) as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(mine.as_ptr(), self.slot(self.rank, parity), mine.len())
        };
        self.len_of(self.rank, parity)
            .store(mine.len() as u64, Ordering::Relaxed);
        self.generation_of(self.rank).store(next, Ordering::Release);
        let start = Instant::now();
        let mut out = Vec::with_capacity(self.ranks as usize);
        for r in 0..self.ranks {
            while self.generation_of(r).load(Ordering::Acquire) < next {
                if start.elapsed() > PATIENCE {
                    return Err(Error::Join(format!("rank {r} never reached round {next}")));
                }
                thread::yield_now();
            }
            let len = self.len_of(r, parity).load(Ordering::Relaxed) as usize;
            let bytes = unsafe { std::slice::from_raw_parts(self.slot(r, parity), len) };
            out.push(bytes.to_vec());
        }
        self.generation = next;
        Ok(out)
    }
}
