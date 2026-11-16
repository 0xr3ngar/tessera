//! File bytes as a buffer source: mapped, read or copied.

use crate::buffer::Backing;
use crate::{Error, mismatch};
use memmap2::{Mmap, MmapMut, MmapOptions};
use std::any::Any;
use std::fs;
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::{Arc, OnceLock};

/// An open file, mapped whole on first ask.
struct Opened {
    file: fs::File,
    whole: OnceLock<Option<Mmap>>,
}

/// Bytes `at..at + len` of a file.
#[derive(Clone)]
pub struct FileRange {
    file: Arc<Opened>,
    pub at: u64,
    pub len: usize,
}

impl FileRange {
    /// All of `file`.
    pub fn new(file: fs::File) -> io::Result<Self> {
        let len = file.metadata()?.len() as usize;
        let whole = OnceLock::new();
        Ok(Self {
            file: Arc::new(Opened { file, whole }),
            at: 0,
            len,
        })
    }

    /// `len` bytes from `at` within this range.
    pub fn sub(&self, at: usize, len: usize) -> Option<Self> {
        (at + len <= self.len).then(|| Self {
            file: self.file.clone(),
            at: self.at + at as u64,
            len,
        })
    }

    /// Reads the bytes into `out`, which is `len` long.
    pub fn read(&self, out: &mut [u8]) -> io::Result<()> {
        self.file.file.read_exact_at(out, self.at)
    }

    /// A private map. Writes stay in this process.
    pub fn map(&self) -> io::Result<Mapped> {
        let mut options = MmapOptions::new();
        options.offset(self.at).len(self.len);
        // The file is opened read-only and taken as unchanging.
        let map = unsafe { options.map_copy(&self.file.file)? };
        Ok(Mapped {
            map,
            range: self.clone(),
        })
    }

    /// The whole file mapped read-only, and what keeps it.
    pub fn whole(&self) -> Option<(&[u8], Arc<dyn Any + Send + Sync>)> {
        let map = self.file.whole.get_or_init(|| {
            // As for `map`: the file does not change under us.
            unsafe { Mmap::map(&self.file.file) }.ok()
        });
        let keep: Arc<dyn Any + Send + Sync> = self.file.clone();
        let bytes = map.as_deref()?;
        Some((bytes, keep))
    }
}

/// A file range mapped copy-on-write: host memory without a read.
pub struct Mapped {
    map: MmapMut,
    range: FileRange,
}

impl Backing for Mapped {
    fn bytes(&self) -> usize {
        self.map.len()
    }

    fn mapped(&self) -> Option<&[u8]> {
        Some(&self.map)
    }

    fn mapped_mut(&mut self) -> Option<&mut [u8]> {
        Some(&mut self.map)
    }

    fn upload(&mut self, at: usize, bytes: &[u8]) -> Result<(), Error> {
        let Some(span) = self.map.get_mut(at..at + bytes.len()) else {
            return mismatch("a write past the end of a mapped file");
        };
        span.copy_from_slice(bytes);
        Ok(())
    }

    fn download(&self, at: usize, out: &mut [u8]) -> Result<(), Error> {
        let Some(span) = self.map.get(at..at + out.len()) else {
            return mismatch("a read past the end of a mapped file");
        };
        out.copy_from_slice(span);
        Ok(())
    }

    fn source(&self) -> Option<&FileRange> {
        Some(&self.range)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
