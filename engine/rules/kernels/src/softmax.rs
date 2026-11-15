//! Softmax over rows, generated.

use crate::per_row;
use codegen::prelude::*;
use graph::Stage;
use lower::{Kernel, stage_rule};

/// One thread per row: peak, sum, then the probabilities.
#[kernel]
fn softmax(x: Buf<F32>, mut out: BufMut<F32>, rows: u32, width: u32) {
    if gid() >= rows {
        return;
    }
    let at = gid() as u64 * width as u64;
    let mut top = x[at];
    for i in 1..width {
        top = max(top, x[at + i as u64]);
    }
    let mut sum = 0.0f32;
    for i in 0..width {
        sum += exp(x[at + i as u64] - top);
    }
    for i in 0..width {
        out[at + i as u64] = exp(x[at + i as u64] - top) / sum;
    }
}

/// One thread turns each row into probabilities.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    softmax,
    threads = self.rows,
    load = Some(per_row(self.rows, self.width, body.trip(), 0.0)),
)]
pub struct Softmax {
    pub rows: usize,
    pub width: usize,
}

stage_rule!("softmax", |site| {
    (site.stage == Stage::Softmax).then(|| Softmax {
        rows: site.len / site.width,
        width: site.width,
    })
});
