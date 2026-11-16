//! What compiled lockstep kernels call: reads, shuffles and tiles.

use crate::exec::{ArgVal, Grid};
use codegen::block::{NVFP4, Q4_0, Q8_0};
use codegen::form::{Form, Load};
use codegen::numerics::Precision;
use graph::{E2M1, E4M3, F16};

/// The most threads one compiled group holds.
pub const THREADS: usize = 256;

/// A compiled body. False means the grid was too wide.
pub type Kernel = fn(&[ArgVal], &mut [&mut [u8]], Grid, usize, Precision) -> bool;

/// A buffer argument: launch index and elements to skip.
pub fn buf(arg: &ArgVal) -> (usize, usize) {
    match arg {
        ArgVal::Buf { at, skip, .. } => (*at, *skip),
        other => panic!("{other:?} where a buffer was due"),
    }
}

pub fn u32(arg: &ArgVal) -> u32 {
    match arg {
        ArgVal::U32(v) => *v,
        other => panic!("{other:?} where a u32 was due"),
    }
}

pub fn f32(arg: &ArgVal) -> f32 {
    match arg {
        ArgVal::F32(v) => *v,
        other => panic!("{other:?} where an f32 was due"),
    }
}

#[inline(always)]
pub fn rd_f32(bytes: &[u8], at: usize) -> f32 {
    let b = &bytes[at * 4..at * 4 + 4];
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[inline(always)]
pub fn rd_bf16(bytes: &[u8], at: usize) -> f32 {
    let b = &bytes[at * 2..at * 2 + 2];
    f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)
}

#[inline(always)]
pub fn rd_u32(bytes: &[u8], at: usize) -> u32 {
    let b = &bytes[at * 4..at * 4 + 4];
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[inline(always)]
pub fn rd_u8(bytes: &[u8], at: usize) -> u32 {
    u32::from(bytes[at])
}

#[inline(always)]
pub fn rd_q8_0(bytes: &[u8], at: usize) -> f32 {
    Q8_0.unpack(bytes, at)
}

#[inline(always)]
pub fn rd_q4_0(bytes: &[u8], at: usize) -> f32 {
    Q4_0.unpack(bytes, at)
}

#[inline(always)]
pub fn rd_nvfp4(bytes: &[u8], at: usize) -> f32 {
    NVFP4.unpack(bytes, at)
}

#[inline(always)]
pub fn st_f32(bytes: &mut [u8], at: usize, v: f32) {
    bytes[at * 4..at * 4 + 4].copy_from_slice(&v.to_le_bytes());
}

#[inline(always)]
pub fn st_u32(bytes: &mut [u8], at: usize, v: u32) {
    bytes[at * 4..at * 4 + 4].copy_from_slice(&v.to_le_bytes());
}

/// `n` floats from `at` become `out * keep + p * b`.
pub fn mix_f32(out: &mut [u8], at: usize, keep: f32, p: f32, b: &[u8], b_at: usize, n: usize) {
    let mut other = Vec::new();
    let b = f32_row(b, b_at, n, &mut other);
    let slice = &mut out[at * 4..(at + n) * 4];
    // SAFETY: f32 has no invalid bit patterns.
    let (head, body, _) = unsafe { slice.align_to_mut::<f32>() };
    if head.is_empty() && body.len() == n {
        // A zero `keep` drops old values. They may be junk.
        match keep == 0.0 {
            true => body.iter_mut().zip(b).for_each(|(o, x)| *o = p * x),
            false => body
                .iter_mut()
                .zip(b)
                .for_each(|(o, x)| *o = *o * keep + p * x),
        }
        return;
    }
    for (i, x) in b.iter().enumerate() {
        let (o, _) = slice[i * 4..].split_first_chunk::<4>().expect("four bytes");
        let old = if keep == 0.0 {
            0.0
        } else {
            f32::from_le_bytes(*o) * keep
        };
        slice[i * 4..i * 4 + 4].copy_from_slice(&(old + p * x).to_le_bytes());
    }
}

/// The u32 at byte offset `at`.
#[inline(always)]
pub fn word(bytes: &[u8], at: usize) -> u32 {
    let b = &bytes[at..at + 4];
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// The u16 at byte offset `at`, widened.
#[inline(always)]
pub fn half(bytes: &[u8], at: usize) -> u32 {
    u32::from(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

/// The `n` bits of `w` from bit `low`.
#[inline(always)]
pub fn field(w: u32, low: u32, n: u32) -> u32 {
    (w >> low) & ((1u64 << n) - 1) as u32
}

/// An `n`-bit field as a signed value.
#[inline(always)]
pub fn signed(w: u32, low: u32, n: u32) -> f32 {
    let shift = 32 - n;
    (((field(w, low, n) << shift) as i32) >> shift) as f32
}

#[inline(always)]
pub fn f16(w: u32, low: u32) -> f32 {
    f32::from(F16(field(w, low, 16) as u16))
}

#[inline(always)]
pub fn e4m3(w: u32, low: u32) -> f32 {
    f32::from(E4M3(field(w, low, 8) as u8))
}

#[inline(always)]
pub fn e2m1(w: u32, low: u32) -> f32 {
    f32::from(E2M1(field(w, low, 4) as u8))
}

macro_rules! wide {
    ($($name:ident = $op:tt),*) => {$(
        #[inline(always)]
        pub fn $name(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
            [a[0] $op b[0], a[1] $op b[1], a[2] $op b[2], a[3] $op b[3]]
        }
    )*};
}

wide!(add4 = +, sub4 = -, mul4 = *, div4 = /);

/// Each lane's value from lane `m` away in its group.
pub fn shfl_xor(v: &[f32; THREADS], m: &[u32; THREADS], n: usize, lanes: usize) -> [f32; THREADS] {
    let mut out = *v;
    for t in 0..n {
        let other = t ^ (m[t] as usize);
        if other / lanes == t / lanes && other < n {
            out[t] = v[other];
        }
    }
    out
}

/// Each live lane takes its group's sum of live lanes.
pub fn sum32(v: &mut [f32; THREADS], mask: &[bool; THREADS], n: usize, lanes: usize) {
    let mut group = 0;
    while group < n {
        let end = (group + lanes).min(n);
        let mut sum = 0.0;
        for t in group..end {
            if mask[t] {
                sum += v[t];
            }
        }
        for t in group..end {
            if mask[t] {
                v[t] = sum;
            }
        }
        group = end;
    }
}

/// `rows` by `cols` sums per SIMD group of `lanes` threads.
pub struct Tiles {
    rows: usize,
    cols: usize,
    lanes: usize,
    blocks: Vec<Vec<f32>>,
}

impl Tiles {
    pub fn new(rows: usize, cols: usize, n: usize, lanes: usize) -> Self {
        let simds = n.div_ceil(lanes);
        Self {
            rows,
            cols,
            lanes,
            blocks: vec![vec![0.0; rows * cols]; simds],
        }
    }

    /// Sums each lane holds.
    pub fn per(&self) -> usize {
        (self.rows * self.cols).div_ceil(self.lanes)
    }

    pub fn zero(&mut self) {
        for block in &mut self.blocks {
            block.fill(0.0);
        }
    }

    /// Lane `t`'s sum `e`: live, row, column and value.
    pub fn each(&self, t: usize, e: usize, live: bool) -> (bool, u32, u32, f32) {
        let at = t % self.lanes + e * self.lanes;
        let on = live && at < self.rows * self.cols;
        let (r, c) = ((at / self.cols) as u32, (at % self.cols) as u32);
        let v = if on {
            self.blocks[t / self.lanes][at]
        } else {
            0.0
        };
        (on, r, c, v)
    }

    /// The row of lane `t`'s sum `e`.
    pub fn row(&self, t: usize, e: usize) -> u32 {
        ((t % self.lanes + e * self.lanes) / self.cols) as u32
    }

    /// Scales lane `t`'s sum `e` by `by`.
    pub fn scale(&mut self, t: usize, e: usize, by: f32) {
        let at = t % self.lanes + e * self.lanes;
        if at < self.rows * self.cols {
            self.blocks[t / self.lanes][at] *= by;
        }
    }

    /// Adds `depth` columns of `a` times `b`, unit rounded.
    #[allow(clippy::too_many_arguments, reason = "mirrors the tile statement")]
    pub fn block(
        &mut self,
        a: &[f32],
        a_at: &[u32; THREADS],
        b: &[f32],
        b_at: &[u32; THREADS],
        stride: &[u32; THREADS],
        depth: usize,
        mask: &[bool; THREADS],
        n: usize,
        p: Precision,
    ) {
        for s in 0..self.blocks.len() {
            // Any live lane of the SIMD group names its slabs.
            let range = s * self.lanes..((s + 1) * self.lanes).min(n);
            let Some(t) = range.into_iter().find(|t| mask[*t]) else {
                continue;
            };
            let (a0, b0, ld) = (a_at[t] as usize, b_at[t] as usize, stride[t] as usize);
            let sa = |i: usize| a[a0 + i];
            let sb = |i: usize| b[b0 + i];
            products(
                p,
                self.rows,
                self.cols,
                ld,
                depth,
                sa,
                sb,
                &mut self.blocks[s],
            );
        }
    }
}

/// `block[i][j] += sum of a[i][k] * b[j][k]`, as `p` rounds.
#[allow(clippy::too_many_arguments, reason = "every tile operand")]
pub(crate) fn products(
    p: Precision,
    rows: usize,
    cols: usize,
    ld: usize,
    depth: usize,
    a: impl Fn(usize) -> f32,
    b: impl Fn(usize) -> f32,
    block: &mut [f32],
) {
    let chunk = p.depth.max(1);
    for i in 0..rows {
        for j in 0..cols {
            let sum = &mut block[i * cols + j];
            // The unit sums `p.depth` products, then adds them in.
            for start in (0..depth).step_by(chunk) {
                let mut part = 0.0f32;
                for k in start..(start + chunk).min(depth) {
                    let (ah, al) = p.split(a(i * ld + k));
                    let (bh, bl) = p.split(b(j * ld + k));
                    // The split's products: hi·hi, hi·lo, lo·hi.
                    part = ah.mul_add(bh, part);
                    part = ah.mul_add(bl, part);
                    part = al.mul_add(bh, part);
                }
                *sum = p.accumulate(*sum, part);
            }
        }
    }
}

/// Eight running sums, so the loop vectorizes.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (whole, rest) = (a.len() / 8 * 8, a.len() % 8);
    let (xs, _) = a[..whole].as_chunks::<8>();
    let (ys, _) = b[..whole].as_chunks::<8>();
    for (x, y) in xs.iter().zip(ys) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut sum = acc.iter().sum::<f32>();
    for i in 0..rest {
        sum += a[whole + i] * b[whole + i];
    }
    sum
}

/// `n` floats of `bytes` from element `at`, copied when unaligned.
fn f32_row<'a>(bytes: &'a [u8], at: usize, n: usize, scratch: &'a mut Vec<f32>) -> &'a [f32] {
    let slice = &bytes[at * 4..(at + n) * 4];
    // SAFETY: f32 has no invalid bit patterns.
    let (head, body, _) = unsafe { slice.align_to::<f32>() };
    if head.is_empty() && body.len() == n {
        return body;
    }
    scratch.clear();
    let (words, _) = slice.as_chunks::<4>();
    scratch.extend(words.iter().map(|c| f32::from_le_bytes(*c)));
    scratch
}

/// Depth a block sums while its rows stay in cache.
const SPAN: usize = 256;
/// Rows and columns of one block of sums.
pub const BLOCK: usize = 16;

/// `a + x * y`, lane by lane.
#[inline(always)]
fn lanes(a: [f32; 4], x: [f32; 4], y: [f32; 4]) -> [f32; 4] {
    // Fused only where one instruction does it.
    match cfg!(any(target_arch = "aarch64", target_feature = "fma")) {
        true => std::array::from_fn(|l| x[l].mul_add(y[l], a[l])),
        false => std::array::from_fn(|l| a[l] + x[l] * y[l]),
    }
}

/// `R` rows of `x` against four rows of `w`.
fn tile<const R: usize>(x: [&[f32]; R], w: [&[f32]; 4]) -> [[f32; 4]; R] {
    let len = w[0].len();
    let steps = len / 4;
    let xs = x.map(|r| &r[..steps * 4].as_chunks::<4>().0[..steps]);
    let [w0, w1, w2, w3] = w.map(|r| &r[..steps * 4].as_chunks::<4>().0[..steps]);
    // Four lanes per sum, so each load feeds four products.
    let mut acc = [[[0.0f32; 4]; 4]; R];
    for i in 0..steps {
        let (a, b, c, d) = (w0[i], w1[i], w2[i], w3[i]);
        for r in 0..R {
            let xv = xs[r][i];
            let row = &mut acc[r];
            row[0] = lanes(row[0], xv, a);
            row[1] = lanes(row[1], xv, b);
            row[2] = lanes(row[2], xv, c);
            row[3] = lanes(row[3], xv, d);
        }
    }
    let mut out = [[0.0f32; 4]; R];
    for r in 0..R {
        for c in 0..4 {
            out[r][c] = acc[r][c].iter().sum::<f32>();
            for i in steps * 4..len {
                out[r][c] += x[r][i] * w[c][i];
            }
        }
    }
    out
}

/// Rows `first..first + R` against columns `col..col + 4`.
fn quad<const R: usize>(x: &[&[f32]], w: &[&[f32]], at: (usize, usize), sums: &mut Sums) {
    let (first, col) = at;
    let got = tile::<R>(
        std::array::from_fn(|r| x[first + r]),
        std::array::from_fn(|c| w[col + c]),
    );
    for (r, line) in got.iter().enumerate() {
        for (c, v) in line.iter().enumerate() {
            sums[first + r][col + c] += v;
        }
    }
}

type Sums = [[f32; BLOCK]; BLOCK];

/// Where a block reads and writes, in elements.
pub struct Dots {
    pub at: usize,
    pub stride: usize,
    pub x_at: usize,
    pub w_at: usize,
    pub k: usize,
}

/// `rows` of `x` against `BLOCK` rows of `w`, into `out`.
pub fn dots_form(form: &dyn Load, out: &mut [u8], x: &[u8], w: &[u8], d: &Dots, rows: usize) {
    let (k, rows) = (d.k, rows.min(BLOCK));
    let mut held: [Vec<f32>; BLOCK] = std::array::from_fn(|_| Vec::new());
    let mut xr: [&[f32]; BLOCK] = [&[]; BLOCK];
    for (r, h) in held.iter_mut().enumerate().take(rows) {
        xr[r] = f32_row(x, d.x_at + r * k, k, h);
    }
    let wide = form.name() == "f32";
    let mut kept: [Vec<f32>; BLOCK] = std::array::from_fn(|_| Vec::new());
    let mut wr: [&[f32]; BLOCK] = [&[]; BLOCK];
    if wide {
        for (c, h) in kept.iter_mut().enumerate() {
            wr[c] = f32_row(w, d.w_at + c * k, k, h);
        }
    }
    // bf16 rows widen one span at a time.
    let mut narrow = vec![[0.0f32; SPAN]; if wide { 0 } else { BLOCK }];
    let mut sums: Sums = [[0.0; BLOCK]; BLOCK];
    for start in (0..k).step_by(SPAN) {
        let end = (start + SPAN).min(k);
        for (c, line) in narrow.iter_mut().enumerate() {
            let first = d.w_at + c * k + start;
            for (i, v) in line[..end - start].iter_mut().enumerate() {
                let b = &w[(first + i) * 2..(first + i) * 2 + 2];
                *v = f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16);
            }
        }
        let ws: [&[f32]; BLOCK] = match wide {
            true => wr.map(|r| &r[start..end]),
            false => std::array::from_fn(|c| &narrow[c][..end - start]),
        };
        let xs = xr.map(|r| r.get(start..end).unwrap_or(&[]));
        for first in (0..rows).step_by(4) {
            for col in (0..BLOCK).step_by(4) {
                let at = (first, col);
                match rows - first {
                    1 => quad::<1>(&xs, &ws, at, &mut sums),
                    2 => quad::<2>(&xs, &ws, at, &mut sums),
                    3 => quad::<3>(&xs, &ws, at, &mut sums),
                    _ => quad::<4>(&xs, &ws, at, &mut sums),
                }
            }
        }
    }
    for (r, line) in sums.iter().enumerate().take(rows) {
        for (c, v) in line.iter().enumerate() {
            st_f32(out, d.at + r * d.stride + c, *v);
        }
    }
}

/// `x` against bf16 `w`, widened as it is multiplied.
#[cfg(not(target_arch = "aarch64"))]
pub fn dot_bf16(x: &[f32], w: &[u8]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (xs, xr) = x.as_chunks::<8>();
    let (ws, wr) = w.as_chunks::<16>();
    for (xc, wc) in xs.iter().zip(ws) {
        for i in 0..8 {
            let bits = u16::from_le_bytes([wc[2 * i], wc[2 * i + 1]]);
            acc[i] += xc[i] * f32::from_bits(u32::from(bits) << 16);
        }
    }
    let mut sum = acc.iter().sum::<f32>();
    for (i, xi) in xr.iter().enumerate() {
        let bits = u16::from_le_bytes([wr[2 * i], wr[2 * i + 1]]);
        sum += xi * f32::from_bits(u32::from(bits) << 16);
    }
    sum
}

/// `x` against bf16 `w`: NEON widens eight halves a step.
#[cfg(target_arch = "aarch64")]
pub fn dot_bf16(x: &[f32], w: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    let (xs, xr) = x.as_chunks::<8>();
    let (ws, wr) = w.as_chunks::<16>();
    // SAFETY: aarch64 has NEON. Loads stay within the chunks.
    let mut sum = unsafe {
        let (mut lo, mut hi) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        for (xc, wc) in xs.iter().zip(ws) {
            let halves = vld1q_u16(wc.as_ptr().cast::<u16>());
            let a = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(halves)));
            let b = vreinterpretq_f32_u32(vshll_high_n_u16::<16>(halves));
            lo = vfmaq_f32(lo, vld1q_f32(xc.as_ptr()), a);
            hi = vfmaq_f32(hi, vld1q_f32(xc.as_ptr().add(4)), b);
        }
        vaddvq_f32(vaddq_f32(lo, hi))
    };
    for (i, xi) in xr.iter().enumerate() {
        let bits = u16::from_le_bytes([wr[2 * i], wr[2 * i + 1]]);
        sum += xi * f32::from_bits(u32::from(bits) << 16);
    }
    sum
}

/// `n` products of an f32 row and a `form` row.
pub fn dot_form(form: &dyn Load, a: &[u8], a_at: usize, b: &[u8], b_at: usize, n: usize) -> f32 {
    let mut scratch = Vec::new();
    let x = f32_row(a, a_at, n, &mut scratch);
    match form.name() {
        "f32" => {
            let mut other = Vec::new();
            dot(x, f32_row(b, b_at, n, &mut other))
        }
        "bf16" => dot_bf16(x, &b[b_at * 2..(b_at + n) * 2]),
        _ => form.dot(x, b, b_at),
    }
}
