//! What compiled bodies call: buffers, math, shuffles and tiles.

use codegen::Precision;
use host::lockstep::{self, THREADS, Tiles};
use std::sync::atomic::{AtomicU32, Ordering};

/// A buffer as the compiled code sees it.
#[repr(C)]
pub struct Buf {
    pub ptr: *mut u8,
    pub len: usize,
}

/// A launch argument: buffer index and skip, or scalar bits.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Arg {
    pub buf: usize,
    pub skip: usize,
    pub bits: u32,
}

/// What one launch hands every group.
#[repr(C)]
pub struct Ctx {
    pub bufs: *const Buf,
    pub args: *const Arg,
    pub threads: usize,
    pub lanes: usize,
    pub precision: *const Precision,
}

/// A compiled body over groups `lo..hi` of `ctx`.
pub type Entry = unsafe extern "C" fn(*const Ctx, usize, usize);

/// Set by a helper that could not do its work.
static FAILED: AtomicU32 = AtomicU32::new(0);

const PAST_END: u32 = 1;

/// Takes the failure code a launch left, if any.
pub fn failure() -> Option<&'static str> {
    match FAILED.swap(0, Ordering::Relaxed) {
        0 => None,
        PAST_END => Some("a read or write past a buffer's end"),
        _ => Some("a compiled body failed"),
    }
}

/// `len` bytes at `p`, when `end` lies within them.
unsafe fn within<'a>(p: *const u8, len: usize, end: usize) -> Option<&'a [u8]> {
    if end > len {
        FAILED.store(PAST_END, Ordering::Relaxed);
        return None;
    }
    // SAFETY: the caller's buffer holds `len` bytes at `p`.
    Some(unsafe { std::slice::from_raw_parts(p, len) })
}

macro_rules! readers {
    ($($name:ident = $read:ident : $ty:ty, $bytes:expr;)*) => {$(
        pub unsafe extern "C" fn $name(p: *const u8, len: usize, at: usize) -> $ty {
            let bytes: fn(usize) -> usize = $bytes;
            match unsafe { within(p, len, bytes(at + 1)) } {
                Some(b) => lockstep::$read(b, at),
                None => Default::default(),
            }
        }
    )*};
}

readers! {
    rd_f32 = rd_f32: f32, |n| n * 4;
    rd_bf16 = rd_bf16: f32, |n| n * 2;
    rd_u32 = rd_u32: u32, |n| n * 4;
    rd_u8 = rd_u8: u32, |n| n;
    rd_q8_0 = rd_q8_0: f32, |n| n.div_ceil(32) * 34;
    rd_q4_0 = rd_q4_0: f32, |n| n.div_ceil(32) * 18;
    rd_nvfp4 = rd_nvfp4: f32, |n| 4 + n.div_ceil(16) * 9;
}

macro_rules! dots {
    ($($name:ident = $form:expr, $bytes:expr;)*) => {$(
        #[allow(clippy::too_many_arguments, reason = "two rows and a length")]
        pub unsafe extern "C" fn $name(
            a: *const u8,
            a_len: usize,
            a_at: usize,
            b: *const u8,
            b_len: usize,
            b_at: usize,
            n: usize,
        ) -> f32 {
            let bytes: fn(usize) -> usize = $bytes;
            let (Some(a), Some(b)) = (
                unsafe { within(a, a_len, (a_at + n) * 4) },
                unsafe { within(b, b_len, bytes(b_at + n)) },
            ) else {
                return 0.0;
            };
            lockstep::dot_form($form, a, a_at, b, b_at, n)
        }
    )*};
}

dots! {
    dot_f32 = codegen::form::F32_FORM, |n| n * 4;
    dot_bf16 = codegen::form::BF16_FORM, |n| n * 2;
    dot_q8_0 = &codegen::block::Q8_0, |n| n.div_ceil(32) * 34;
    dot_q4_0 = &codegen::block::Q4_0, |n| n.div_ceil(32) * 18;
    dot_nvfp4 = &codegen::block::NVFP4, |n| 4 + n.div_ceil(16) * 9;
}

pub unsafe extern "C" fn st_f32(p: *mut u8, len: usize, at: usize, v: f32) {
    if unsafe { within(p, len, at * 4 + 4) }.is_some() {
        // SAFETY: checked above.
        lockstep::st_f32(unsafe { std::slice::from_raw_parts_mut(p, len) }, at, v);
    }
}

macro_rules! tiles {
    ($($name:ident = $form:expr, $bytes:expr;)*) => {$(
        #[allow(clippy::too_many_arguments, reason = "three buffers and their offsets")]
        pub unsafe extern "C" fn $name(
            out: *mut u8,
            out_len: usize,
            at: usize,
            stride: usize,
            x: *const u8,
            x_len: usize,
            x_at: usize,
            w: *const u8,
            w_len: usize,
            w_at: usize,
            k: usize,
            rows: usize,
        ) {
            let bytes: fn(usize) -> usize = $bytes;
            let block = lockstep::BLOCK;
            let rows = rows.min(block);
            let last = (at + rows.saturating_sub(1) * stride + block) * 4;
            let (Some(_), Some(x), Some(w)) = (
                unsafe { within(out, out_len, last) },
                unsafe { within(x, x_len, (x_at + rows * k) * 4) },
                unsafe { within(w, w_len, bytes(w_at + block * k)) },
            ) else {
                return;
            };
            // SAFETY: checked above.
            let out = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
            let d = lockstep::Dots { at, stride, x_at, w_at, k };
            lockstep::dots_form($form, out, x, w, &d, rows);
        }
    )*};
}

tiles! {
    dots_f32 = codegen::form::F32_FORM, |n| n * 4;
    dots_bf16 = codegen::form::BF16_FORM, |n| n * 2;
}

#[allow(
    clippy::too_many_arguments,
    reason = "two rows, two scales and a length"
)]
pub unsafe extern "C" fn mix_f32(
    out: *mut u8,
    out_len: usize,
    at: usize,
    keep: f32,
    p: f32,
    b: *const u8,
    b_len: usize,
    b_at: usize,
    n: usize,
) {
    let fits = unsafe { within(out, out_len, (at + n) * 4) }.is_some();
    let Some(b) = (unsafe { within(b, b_len, (b_at + n) * 4) }) else {
        return;
    };
    if fits {
        // SAFETY: checked above.
        let out = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
        lockstep::mix_f32(out, at, keep, p, b, b_at, n);
    }
}

pub unsafe extern "C" fn st_u32(p: *mut u8, len: usize, at: usize, v: u32) {
    if unsafe { within(p, len, at * 4 + 4) }.is_some() {
        // SAFETY: checked above.
        lockstep::st_u32(unsafe { std::slice::from_raw_parts_mut(p, len) }, at, v);
    }
}

pub unsafe extern "C" fn word(p: *const u8, len: usize, at: usize) -> u32 {
    match unsafe { within(p, len, at + 4) } {
        Some(b) => lockstep::word(b, at),
        None => 0,
    }
}

pub unsafe extern "C" fn half(p: *const u8, len: usize, at: usize) -> u32 {
    match unsafe { within(p, len, at + 2) } {
        Some(b) => lockstep::half(b, at),
        None => 0,
    }
}

macro_rules! unary {
    ($($name:ident = $f:expr;)*) => {$(
        pub extern "C" fn $name(x: f32) -> f32 {
            let f: fn(f32) -> f32 = $f;
            f(x)
        }
    )*};
}

unary! {
    exp = f32::exp;
    tanh = f32::tanh;
    cos = f32::cos;
    sin = f32::sin;
}

pub extern "C" fn max(a: f32, b: f32) -> f32 {
    a.max(b)
}

pub extern "C" fn pow(a: f32, b: f32) -> f32 {
    a.powf(b)
}

pub extern "C" fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    x.clamp(lo, hi)
}

pub extern "C" fn f16(w: u32, low: u32) -> f32 {
    lockstep::f16(w, low)
}

pub extern "C" fn e4m3(w: u32, low: u32) -> f32 {
    lockstep::e4m3(w, low)
}

pub extern "C" fn e2m1(w: u32, low: u32) -> f32 {
    lockstep::e2m1(w, low)
}

/// Lane arrays, as compiled code keeps them on its stack.
type Lanes<T> = [T; THREADS];

/// `out` takes each lane's value from the lane `m` away.
pub unsafe extern "C" fn shfl_xor(
    v: *const Lanes<f32>,
    m: *const Lanes<u32>,
    out: *mut Lanes<f32>,
    n: usize,
    lanes: usize,
) {
    // SAFETY: the compiled code passes its own lane arrays.
    unsafe { *out = lockstep::shfl_xor(&*v, &*m, n, lanes) };
}

pub unsafe extern "C" fn sum32(
    v: *mut Lanes<f32>,
    mask: *const Lanes<bool>,
    n: usize,
    lanes: usize,
) {
    // SAFETY: the compiled code passes its own lane arrays.
    unsafe { lockstep::sum32(&mut *v, &*mask, n, lanes) };
}

pub extern "C" fn tiles_new(rows: usize, cols: usize, n: usize, lanes: usize) -> *mut Tiles {
    Box::into_raw(Box::new(Tiles::new(rows, cols, n, lanes)))
}

pub unsafe extern "C" fn tiles_free(t: *mut Tiles) {
    // SAFETY: `tiles_new` made it. Nothing holds it past the group.
    drop(unsafe { Box::from_raw(t) });
}

pub unsafe extern "C" fn tiles_zero(t: *mut Tiles) {
    unsafe { (*t).zero() }
}

pub unsafe extern "C" fn tiles_per(t: *const Tiles) -> usize {
    unsafe { (*t).per() }
}

pub unsafe extern "C" fn tiles_row(t: *const Tiles, lane: usize, e: usize) -> u32 {
    unsafe { (*t).row(lane, e) }
}

pub unsafe extern "C" fn tiles_scale(t: *mut Tiles, lane: usize, e: usize, by: f32) {
    unsafe { (*t).scale(lane, e, by) }
}

/// Lane sum `e`: row, column, value, and whether live.
pub unsafe extern "C" fn tiles_each(
    t: *const Tiles,
    lane: usize,
    e: usize,
    live: u8,
    r: *mut u32,
    c: *mut u32,
    v: *mut f32,
) -> u8 {
    let (on, row, col, value) = unsafe { (*t).each(lane, e, live != 0) };
    unsafe {
        *r = row;
        *c = col;
        *v = value;
    }
    u8::from(on)
}

#[allow(clippy::too_many_arguments, reason = "mirrors the tile statement")]
pub unsafe extern "C" fn tiles_block(
    t: *mut Tiles,
    a: *const f32,
    a_len: usize,
    a_at: *const Lanes<u32>,
    b: *const f32,
    b_len: usize,
    b_at: *const Lanes<u32>,
    stride: *const Lanes<u32>,
    depth: usize,
    mask: *const Lanes<bool>,
    n: usize,
    precision: *const Precision,
) {
    // SAFETY: compiled code passes its own memory and lane arrays.
    unsafe {
        let (a, b) = (
            std::slice::from_raw_parts(a, a_len),
            std::slice::from_raw_parts(b, b_len),
        );
        (*t).block(a, &*a_at, b, &*b_at, &*stride, depth, &*mask, n, *precision);
    }
}
