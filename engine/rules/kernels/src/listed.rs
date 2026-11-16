//! Causal attention over listed keys, or computed keys by row.

use codegen::prelude::*;
use graph::{Mask, Op};
use lower::sites::Attend;
use lower::{Attn, Draft, Kernel, Launch, Shape, Step, T, Term, Tree};

/// One thread per output: the peak, then the weighted sum.
#[kernel]
fn listed(
    q: Buf<F32>,
    k: Buf<F32>,
    v: Buf<F32>,
    ids: Buf<U32>,
    slots: Buf<U32>,
    positions: Buf<U32>,
    mut out: BufMut<F32>,
    scale: f32,
    len: u32,
    queries: u32,
    keys: u32,
    heads: u32,
    kv_heads: u32,
    width: u32,
    picks: u32,
    ragged: u32,
    from_list: u32,
) {
    if gid() >= len {
        return;
    }
    let d = gid() % width;
    let head = (gid() / width) % heads;
    let row = gid() / (width * heads);
    // Whole sequences end together. Ragged rows carry slot and position.
    let mut first = row / queries * keys;
    let mut position = keys - queries + row % queries;
    if ragged == 1 {
        first = slots[row as u64] * keys;
        position = positions[row as u64];
    }
    let kv_head = head / (heads / kv_heads);
    let list = row as u64 * picks as u64;
    let q_at = (gid() - d) as u64;
    let mut peak = f32::NEG_INFINITY;
    for p in 0..picks {
        let mut j = p;
        if from_list == 1 {
            j = ids[list + p as u64];
        }
        if j <= position {
            let k_at = ((first + j) * kv_heads + kv_head) as u64 * width as u64;
            peak = max(peak, q.dot(q_at, k, k_at, width) * scale);
        }
    }
    let mut total = 0.0f32;
    let mut sum = 0.0f32;
    for p in 0..picks {
        let mut j = p;
        if from_list == 1 {
            j = ids[list + p as u64];
        }
        if j <= position {
            let k_at = ((first + j) * kv_heads + kv_head) as u64 * width as u64;
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

/// Query rows over `[Seqs, Keys, KvHeads, Width]` keys.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sizes {
    pub len: usize,
    /// Queries per sequence. Ragged rows have one each.
    pub queries: usize,
    pub keys: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub width: usize,
    pub picks: usize,
    pub ragged: bool,
    /// Without a list, pick `p` is key `p`.
    pub from_list: bool,
    pub scale: f32,
}

/// Whole sequences over listed keys, or ragged rows by slot.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    listed,
    name = "listed",
    sizes = [self.0.len, self.0.keys, self.0.picks],
    work = self.0.len * self.0.picks * self.0.width,
    launches = listed_launches,
)]
pub(crate) struct Listed(pub Sizes);

fn listed_launches(k: &Listed, step: &Step) -> Vec<Launch> {
    let s = &k.0;
    let flags = [usize::from(s.ragged), usize::from(s.from_list)];
    let counts = [s.queries, s.keys, s.heads, s.kv_heads, s.width, s.picks];
    let mut launch = Launch::plain(step, &[s.len], Shape::Line(s.len));
    launch
        .args
        .insert(launch.args.len() - 1, lower::Arg::value(&s.scale));
    launch
        .args
        .extend(counts.into_iter().chain(flags).map(lower::Arg::count));
    vec![launch]
}

/// Ragged rows over computed keys, not kept rows.
fn rows(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let Term::Op(Op::Attend(a)) = &node.term else {
        return None;
    };
    let (q, k, v) = (node.args[0], node.args[1], node.args[2]);
    let kept = |t: T| matches!(tree.node(t).term, Term::Op(Op::Append(_)));
    if !a.ragged || kept(k) || kept(v) || tree.node(k).extents[1].runtime() {
        return None;
    }
    let wide = |t: T| &tree.node(t).shape;
    let (&[rows, heads, width], &[_, keys, kv_heads, _]) = (&wide(q)[..], &wide(k)[..]) else {
        return None;
    };
    if wide(v).last() != Some(&width) {
        return None;
    }
    let [slots, positions] = [node.args[3], node.args[4]];
    let (ids, picks) = match node.args.get(5) {
        Some(&ids) => (ids, *wide(ids).last()?),
        None => (positions, keys),
    };
    let sizes = Sizes {
        len: rows * heads * width,
        queries: 1,
        keys,
        heads,
        kv_heads,
        width,
        picks,
        ragged: true,
        from_list: a.picked,
        scale: a.scale,
    };
    let reads = [q, k, v, ids, slots, positions];
    Some(vec![Draft::new(Listed(sizes), reads, sizes.len)])
}

/// Causal sequences whose keys are all present.
fn whole(tree: &Tree, site: Attend, ids: T) -> Option<Vec<Draft>> {
    let a: Attn = site.sizes;
    if a.mask != Mask::Causal || a.keys.runtime() {
        return None;
    }
    let sizes = Sizes {
        len: a.seqs * a.queries * a.heads * a.width,
        queries: a.queries,
        keys: a.keys.most(),
        heads: a.heads,
        kv_heads: a.kv_heads,
        width: a.width,
        picks: *tree.node(ids).shape.last()?,
        ragged: false,
        from_list: true,
        scale: site.scale,
    };
    // Slots and positions go unread, so the list stands in.
    let reads = [site.q, site.k, site.v, ids, ids, ids];
    Some(vec![Draft::new(Listed(sizes), reads, sizes.len)])
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    match Attend::listed(tree, at) {
        Some((site, ids)) => whole(tree, site, ids),
        None => rows(tree, at),
    }
}

lower::rule!("listed", propose);
