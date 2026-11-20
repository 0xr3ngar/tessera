//! Entries whose mask bit is clear, set to -inf.

use tessera_codegen::prelude::*;
use tessera_graph::Stage;
use tessera_lower::{Kernel, stage_rule};

/// One thread per entry. Each row has `words` mask words.
#[kernel]
fn allow(x: Buf<F32>, bits: Buf<U32>, mut out: BufMut<F32>, len: u32, width: u32, words: u32) {
    if gid() >= len {
        return;
    }
    let col = gid() % width;
    let word = bits[((gid() / width) * words + col / 32) as u64];
    let mut v = x[gid() as u64];
    if (word >> (col % 32)) & 1 == 0 {
        v = f32::NEG_INFINITY;
    }
    out[gid() as u64] = v;
}

/// Rows of `width`, a mask row each.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(allow, threads = self.len)]
pub(crate) struct Allow {
    pub len: usize,
    pub width: usize,
    pub words: usize,
}

stage_rule!("allow", |site| {
    let (Stage::Allow, Some(_)) = (&site.stage, site.operand) else {
        return None;
    };
    Some(Allow {
        len: site.len,
        width: site.width,
        words: site.width.div_ceil(32),
    })
});
