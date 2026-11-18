//! The Walsh-Hadamard transform of 128-wide blocks, a SIMD group each.

use sensai_codegen::prelude::*;
use sensai_graph::Stage;
use sensai_lower::{Kernel, Needs, stage_rule};

/// Each lane holds four values of the block.
const PER_LANE: u32 = 4;

/// The block this kernel transforms.
pub const BLOCK: u32 = LANES as u32 * PER_LANE;

/// Lower keeps the sum, upper the difference.
#[kernel]
fn butterfly(a: f32, b: f32, upper: bool) -> f32 {
    if upper { b - a } else { a + b }
}

/// One lane's four values of a block.
#[derive(Group)]
pub struct Four {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
}

/// A SIMD group's block, four values a lane, turned.
#[kernel]
pub fn spin(x: f32, y: f32, z: f32, w: f32) -> Four {
    // Distances one and two stay in the lane.
    let xy = x + y;
    let xy_minus = x - y;
    let zw = z + w;
    let zw_minus = z - w;
    let mut a = xy + zw;
    let mut b = xy_minus + zw_minus;
    let mut c = xy - zw;
    let mut d = xy_minus - zw_minus;
    // Then five exchanges across the lanes.
    for level in 0..5 {
        let mask = 1u32 << level;
        let upper = (lane() & mask) != 0;
        a = butterfly(a, shfl_xor(a, mask), upper);
        b = butterfly(b, shfl_xor(b, mask), upper);
        c = butterfly(c, shfl_xor(c, mask), upper);
        d = butterfly(d, shfl_xor(d, mask), upper);
    }
    // One over the square root of the block, 128.
    let scale = rsqrt(128.0);
    Four {
        a: a * scale,
        b: b * scale,
        c: c * scale,
        d: d * scale,
    }
}

/// One SIMD group per block.
#[kernel]
fn hadamard(x: Buf<F32>, mut out: BufMut<F32>, blocks: u32) {
    let block = gid() / LANES as u32;
    if block >= blocks {
        return;
    }
    let at = block as u64 * BLOCK as u64 + lane() as u64 * PER_LANE as u64;
    let v = x.load4(at);
    let t = spin(v.x, v.y, v.z, v.w);
    let mut o = F32x4::zero();
    o.x = t.a;
    o.y = t.b;
    o.z = t.c;
    o.w = t.d;
    out.store4(at, o);
}

/// `blocks` transforms of `BLOCK` values, one SIMD group each.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(hadamard, threads = self.blocks * LANES, needs = Needs::lanes())]
pub struct Hadamard {
    pub blocks: usize,
}

stage_rule!("hadamard", |site| {
    let Stage::Hadamard { block } = site.stage else {
        return None;
    };
    (block == BLOCK).then(|| Hadamard {
        blocks: site.len / BLOCK as usize,
    })
});
