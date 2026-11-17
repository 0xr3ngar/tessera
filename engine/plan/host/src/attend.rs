//! One query's softmax attention, in one call from compiled bodies.

use crate::lockstep::{f32_row, st_f32};
use crate::simd::{Level, dot_at, mix_at};

/// Where a query, its keys and its output lie.
pub struct Query {
    pub out_at: usize,
    pub q_at: usize,
    /// Key 0 in `k` and `v`, then `stride` per key.
    pub k_at: usize,
    pub v_at: usize,
    pub stride: usize,
    /// Keys the query may see, `from..to`.
    pub from: usize,
    pub to: usize,
    pub width: usize,
    pub scale: f32,
    /// The key list's start and length. Zero lists none.
    pub list_at: usize,
    pub picks: usize,
}

/// The buffers one query reads and writes, as bytes.
pub struct Rows<'a> {
    pub out: &'a mut [u8],
    pub q: &'a [u8],
    pub k: &'a [u8],
    pub v: &'a [u8],
    pub ids: &'a [u8],
}

/// The keys `a` sees: its list in `from..to`, or all.
fn seen(ids: &[u8], a: &Query) -> Vec<usize> {
    if a.picks == 0 {
        return (a.from..a.to).collect();
    }
    let mut out = Vec::with_capacity(a.picks);
    for i in 0..a.picks {
        let at = (a.list_at + i) * 4;
        let j = u32::from_le_bytes([ids[at], ids[at + 1], ids[at + 2], ids[at + 3]]) as usize;
        if (a.from..a.to).contains(&j) {
            out.push(j);
        }
    }
    out
}

/// Scores, their softmax, then the weighted value rows, into `out`.
pub fn attend_f32(b: Rows, a: &Query) {
    let level = Level::best();
    let (width, keys) = (a.width, seen(b.ids, a));
    let mut scratch = Vec::new();
    let q = f32_row(b.q, a.q_at, width, &mut scratch).to_vec();
    let mut row = Vec::new();
    let mut scores = Vec::with_capacity(keys.len());
    for &j in &keys {
        let k = f32_row(b.k, a.k_at + j * a.stride, width, &mut row);
        scores.push(dot_at(level, &q, k) * a.scale);
    }
    let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for s in &mut scores {
        *s = (*s - top).exp();
        total += *s;
    }
    let mut sum = vec![0.0f32; width];
    for (i, (&j, p)) in keys.iter().zip(&scores).enumerate() {
        let v = f32_row(b.v, a.v_at + j * a.stride, width, &mut row);
        // The first key's zero `keep` sets the row.
        mix_at(
            level,
            &mut sum,
            v,
            if i == 0 { 0.0 } else { 1.0 },
            p / total,
        );
    }
    for (c, x) in sum.iter().enumerate() {
        st_f32(b.out, a.out_at + c, *x);
    }
}
