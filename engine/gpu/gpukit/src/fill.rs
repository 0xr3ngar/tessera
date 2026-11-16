//! A device buffer filled the cheapest way that works.

use crate::Error;
use crate::device::{Device, DeviceBuffer};
use mem::FileRange;
use std::ffi::c_int;
use std::ptr::NonNull;

unsafe extern "C" {
    fn getpagesize() -> c_int;
}

/// Where a buffer's bytes come from.
#[derive(Clone, Copy)]
pub enum Source<'a> {
    /// A file range: wrapped in place where the device can.
    File(&'a FileRange),
    Host(&'a [u8]),
}

/// How a buffer got its bytes, cheapest first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fill {
    /// The device reads the mapped file in place.
    Wrap,
    /// The file read straight into memory the host maps.
    Read,
    Copy,
}

impl<D: Device> DeviceBuffer<D> {
    /// Wrapped where `align` holds, else read in, else copied.
    pub fn fill(ctx: &D, source: Source, align: usize) -> Result<(Self, Fill), Error> {
        if let Source::File(range) = source
            && let Some(buffer) = Self::wrap(ctx, range, align)
        {
            return Ok((buffer, Fill::Wrap));
        }
        let range = match source {
            Source::File(range) => range,
            Source::Host(bytes) => {
                let buffer = Self::alloc(ctx, bytes.len().max(4))?;
                buffer.write_bytes(bytes)?;
                return Ok((buffer, Fill::Copy));
            }
        };
        let buffer = Self::alloc(ctx, range.len.max(4))?;
        let io = |e: std::io::Error| D::error(format!("reading a file range: {e}"));
        if let Some(at) = ctx.mapped(buffer.ptr, range.len) {
            // This allocation alone holds the host memory.
            let out = unsafe { std::slice::from_raw_parts_mut(at.as_ptr(), range.len) };
            range.read(out).map_err(io)?;
            return Ok((buffer, Fill::Read));
        }
        let mapped = range.map().map_err(io)?;
        let bytes = mem::Backing::mapped(&mapped).expect("a map is host memory");
        buffer.write_bytes(bytes)?;
        Ok((buffer, Fill::Copy))
    }

    /// The device reading `range` where the file is mapped.
    pub fn wrap(ctx: &D, range: &FileRange, align: usize) -> Option<Self> {
        if range.len == 0 || !range.at.is_multiple_of(align as u64) {
            return None;
        }
        let page = unsafe { getpagesize() } as usize;
        let (whole, keep) = range.whole()?;
        let start = range.at as usize;
        let (low, high) = (
            start / page * page,
            (start + range.len).next_multiple_of(page),
        );
        let host = NonNull::new(whole.as_ptr().wrapping_add(low).cast_mut())?;
        // The map covers whole pages, and `keep` holds it.
        let base = unsafe { ctx.wrap(host, high - low) }?;
        Some(Self {
            ctx: ctx.clone(),
            ptr: base + (start - low) as u64,
            size: range.len,
            keep: Some(keep),
        })
    }
}
