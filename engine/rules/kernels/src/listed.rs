//! Attention over each query's listed keys, within its mask.

use codegen::prelude::*;
use lower::sites::Attend;
use lower::{Attn, Draft, Kernel, Launch, Shape, Step, T, Tree};

/// One thread per output: the peak, then the weighted sum.
#[kernel]
fn listed(
    q: Buf<F32>,
    k: Buf<F32>,
    v: Buf<F32>,
    ids: Buf<U32>,
    mut out: BufMut<F32>,
    scale: f32,
    len: u32,
    queries: u32,
    keys: u32,
    heads: u32,
    kv_heads: u32,
    width: u32,
    picks: u32,
    before: u32,
    after: u32,
) {
    if gid() >= len {
        return;
    }
    let d = gid() % width;
    let head = (gid() / width) % heads;
    let row = gid() / (width * heads);
    let seq = row / queries;
    let position = keys - queries + row % queries;
    let from = if position > before {
        position - before
    } else {
        0
    };
    let to = if position + after < keys {
        position + after + 1
    } else {
        keys
    };
    let kv_head = head / (heads / kv_heads);
    let q_at = (gid() - d) as u64;
    let list = row as u64 * picks as u64;
    let mut peak = f32::NEG_INFINITY;
    for p in 0..picks {
        let j = ids[list + p as u64];
        if j >= from && j < to {
            let k_at = ((seq * keys + j) * kv_heads + kv_head) as u64 * width as u64;
            peak = max(peak, q.dot(q_at, k, k_at, width) * scale);
        }
    }
    let mut total = 0.0f32;
    let mut sum = 0.0f32;
    for p in 0..picks {
        let j = ids[list + p as u64];
        if j >= from && j < to {
            let k_at = ((seq * keys + j) * kv_heads + kv_head) as u64 * width as u64;
            let e = exp(q.dot(q_at, k, k_at, width) * scale - peak);
            total += e;
            sum += e * v[k_at + d as u64];
        }
    }
    let mut o = 0.0f32;
    if total > 0.0 {
        o = sum / total;
    }
    out[gid() as u64] = o;
}

/// Sizes, the score scale, and keys listed per query.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    listed,
    name = "listed",
    sizes = self.sizes.sizes(),
    work = self.len() * self.picks * self.sizes.width,
    launches = listed_launches,
)]
pub(crate) struct Listed {
    pub sizes: Attn,
    pub scale: f32,
    pub picks: usize,
}

impl Listed {
    fn len(&self) -> usize {
        let a = &self.sizes;
        a.seqs * a.queries * a.heads * a.width
    }
}

fn listed_launches(k: &Listed, step: &Step) -> Vec<Launch> {
    let a = &k.sizes;
    let keys = a.keys.most();
    let (before, after) = match a.mask {
        graph::Mask::Causal => (keys, 0),
        graph::Mask::Bidirectional => (keys, keys),
        graph::Mask::Within(n) => (n as usize, n as usize),
    };
    let counts = [
        a.queries, keys, a.heads, a.kv_heads, a.width, k.picks, before, after,
    ];
    let mut launch = Launch::plain(step, &[k.len()], Shape::Line(k.len()));
    launch
        .args
        .insert(launch.args.len() - 1, lower::Arg::value(&k.scale));
    launch.args.extend(counts.map(lower::Arg::count));
    vec![launch]
}

/// Sequences whose keys are all present.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (site, ids) = Attend::listed(tree, at)?;
    if site.sizes.keys.runtime() {
        return None;
    }
    let picks = *tree.node(ids).shape.last()?;
    let kernel = Listed {
        sizes: site.sizes,
        scale: site.scale,
        picks,
    };
    let reads = [site.q, site.k, site.v, ids];
    Some(vec![Draft::new(kernel, reads, kernel.len())])
}

lower::rule!("listed", propose);
