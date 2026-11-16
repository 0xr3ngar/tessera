//! Entries past each row's position, set to -inf.

use codegen::prelude::*;
use graph::Stage;
use lower::{Kernel, stage_rule};

/// One thread per entry. Each position covers `per` rows.
#[kernel]
fn causal(x: Buf<F32>, positions: Buf<U32>, mut out: BufMut<F32>, len: u32, width: u32, per: u32) {
    if gid() >= len {
        return;
    }
    let mut v = x[gid() as u64];
    if gid() % width > positions[(gid() / width / per) as u64] {
        v = f32::NEG_INFINITY;
    }
    out[gid() as u64] = v;
}

/// Rows of `width`, `per` rows to a position.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(causal, threads = self.len)]
pub(crate) struct Causal {
    pub len: usize,
    pub width: usize,
    pub per: usize,
}

stage_rule!("causal", |site| {
    let (Stage::Causal, Some((_, positions))) = (&site.stage, site.operand) else {
        return None;
    };
    let rows = site.len / site.width;
    Some(Causal {
        len: site.len,
        width: site.width,
        per: rows / positions.max(1),
    })
});
