//! Ranks on one box, exchanging through a mapped file.

use crate::{Error, Transport};
use std::ffi::{c_int, c_void};
use std::fs::File;
use std::os::fd::AsRawFd;
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
/// How long a rank waits at a round.
const ROUND: Duration = Duration::from_secs(300);

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
        3 * ranks as usize
    }

    /// The file size `ranks` need for `capacity` byte slots.
    pub fn room(ranks: u32, capacity: usize) -> usize {
        Self::words(ranks) * 8 + 2 * ranks as usize * capacity
    }

    /// Rank `rank` over `file`, zeroed and sized by `room`.
    pub fn over(rank: u32, ranks: u32, file: File) -> Result<Self, Error> {
        let len = file.metadata()?.len() as usize;
        let header = Self::words(ranks) * 8;
        let slots = 2 * ranks as usize;
        if rank >= ranks || len < header || !(len - header).is_multiple_of(slots) {
            let what = format!("{len} bytes for rank {rank} of {ranks}");
            return Err(Error::Fit(what));
        }
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
        Ok(Self {
            rank,
            ranks,
            capacity: (len - header) / slots,
            base: base as *mut u8,
            len,
            generation: 0,
            _file: file,
        })
    }

    fn word(&self, at: usize) -> &AtomicU64 {
        // The header words are aligned and shared by design.
        unsafe { &*(self.base.add(8 * at) as *const AtomicU64) }
    }

    fn generation_of(&self, rank: u32) -> &AtomicU64 {
        self.word(3 * rank as usize)
    }

    /// The length word for `rank`'s slot of this parity.
    fn len_of(&self, rank: u32, parity: usize) -> &AtomicU64 {
        self.word(1 + 3 * rank as usize + parity)
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
            return Err(Error::Fit(what));
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
                if start.elapsed() > ROUND {
                    return Err(Error::Fit(format!("rank {r} never reached round {next}")));
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
