//! Softmax over rows, generated.

use crate::per_row;
use sensai_codegen::prelude::*;
use sensai_graph::Stage;
use sensai_lower::{Arg, Kernel, Launch, Shape, Step, stage_rule};

/// One thread per row: peak, sum, then the probabilities.
#[kernel]
pub(crate) fn softmax(
    x: Buf<F32>,
    mut out: BufMut<F32>,
    rows: u32,
    queries: u32,
    keys: u32,
    scale: f32,
) {
    if gid() >= rows {
        return;
    }
    // Queries end at the last key. Later keys are unseen.
    let seen = keys - queries + gid() % queries + 1;
    let at = gid() as u64 * keys as u64;
    let mut top = x[at] * scale;
    for i in 1..seen {
        top = max(top, x[at + i as u64] * scale);
    }
    let mut sum = 0.0f32;
    for i in 0..seen {
        sum += exp(x[at + i as u64] * scale - top);
    }
    for i in 0..keys {
        out[at + i as u64] = if i < seen {
            exp(x[at + i as u64] * scale - top) / sum
        } else {
            0.0
        };
    }
}

/// One thread per row of `keys` scores.
pub(crate) fn launch(x: Arg, out: Arg, [rows, queries, keys]: [usize; 3], scale: f32) -> Launch {
    let sizes = [rows, queries, keys].map(Arg::count);
    let args = [x, out].into_iter().chain(sizes);
    Launch {
        args: args.chain([Arg::value(&scale)]).collect(),
        shape: Shape::Line(rows),
    }
}

/// One thread turns each row into probabilities.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    softmax,
    launches = softmax_launches,
    load = Some(per_row(self.rows, self.width, body.trip(), 0.0)),
)]
pub struct Softmax {
    pub rows: usize,
    pub width: usize,
}

/// Each row is one query that sees every key, unscaled.
fn softmax_launches(k: &Softmax, step: &Step) -> Vec<Launch> {
    vec![launch(step.read(0), step.out(), [k.rows, 1, k.width], 1.0)]
}

stage_rule!("softmax", |site| {
    (site.stage == Stage::Softmax).then(|| Softmax {
        rows: site.len / site.width,
        width: site.width,
    })
});
