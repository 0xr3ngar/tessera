//! Rows less their mean, or over their sum, generated.

use crate::per_row;
use tessera_codegen::prelude::*;
use tessera_graph::Stage;
use tessera_lower::{Kernel, stage_rule};

/// One thread per row: the sum, then each element.
#[kernel]
fn by_sum<const DIVIDE: bool>(x: Buf<F32>, mut out: BufMut<F32>, rows: u32, width: u32) {
    if gid() >= rows {
        return;
    }
    let at = gid() as u64 * width as u64;
    let mut sum = 0.0f32;
    for i in 0..width {
        sum += x[at + i as u64];
    }
    let mean = sum / width as f32;
    for i in 0..width {
        if DIVIDE {
            out[at + i as u64] = x[at + i as u64] / sum;
        } else {
            out[at + i as u64] = x[at + i as u64] - mean;
        }
    }
}

/// One thread per row: centered, or scaled to unit sum.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    by_sum,
    variant = self.divide,
    name = match self.divide {
        true => "unit_sum",
        false => "center",
    },
    threads = self.rows,
    load = Some(per_row(self.rows, self.width, body.trip(), 0.0)),
)]
pub(crate) struct BySum {
    pub rows: usize,
    pub width: usize,
    pub divide: bool,
}

stage_rule!("by_sum", |site| {
    let divide = match site.stage {
        Stage::Center => false,
        Stage::UnitSum => true,
        _ => return None,
    };
    Some(BySum {
        rows: site.len / site.width,
        width: site.width,
        divide,
    })
});
