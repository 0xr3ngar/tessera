//! Causal attention over listed keys, or computed keys by row.

use codegen::prelude::*;
use graph::{Mask, Op};
use lower::sites::Attend;
use lower::{Attn, Draft, Kernel, Launch, Shape, Step, T, Term, Tree};

/// The peak, then the sum. Unlisted picks are keys.
#[kernel]
fn listed_value(
    q: Buf<F32>,
    k: Buf<F32>,
    v: Buf<F32>,
    ids: Buf<U32>,
    scale: f32,
    q_at: u64,
    d: u32,
    list: u64,
    picks: u32,
    listed: u32,
    position: u32,
    first: u32,
    kv_heads: u32,
    kv_head: u32,
    width: u32,
) -> f32 {
    let mut peak = f32::NEG_INFINITY;
    for p in 0..picks {
        let mut j = p;
        if listed == 1 {
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
        if listed == 1 {
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
    o
}

/// Whole sequences: one thread per output element.
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
    picks: u32,
    #[build] heads: u32,
    #[build] kv_heads: u32,
    #[build] width: u32,
) {
    if gid() >= len {
        return;
    }
    let d = gid() % width;
    let head = (gid() / width) % heads;
    let row = gid() / (width * heads);
    let position = keys - queries + row % queries;
    let first = row / queries * keys;
    let kv_head = head / (heads / kv_heads);
    let list = row as u64 * picks as u64;
    let q_at = (gid() - d) as u64;
    out[gid() as u64] = listed_value(
        q, k, v, ids, scale, q_at, d, list, picks, 1, position, first, kv_heads, kv_head, width,
    );
}

/// Ragged rows over computed keys: each row's slot and position.
#[kernel]
fn listed_rows(
    q: Buf<F32>,
    k: Buf<F32>,
    v: Buf<F32>,
    ids: Buf<U32>,
    slots: Buf<U32>,
    positions: Buf<U32>,
    mut out: BufMut<F32>,
    scale: f32,
    len: u32,
    keys: u32,
    picks: u32,
    listed: u32,
    #[build] heads: u32,
    #[build] kv_heads: u32,
    #[build] width: u32,
) {
    if gid() >= len {
        return;
    }
    let d = gid() % width;
    let head = (gid() / width) % heads;
    let row = gid() / (width * heads);
    let first = slots[row as u64] * keys;
    let kv_head = head / (heads / kv_heads);
    let list = row as u64 * picks as u64;
    let q_at = (gid() - d) as u64;
    out[gid() as u64] = listed_value(
        q,
        k,
        v,
        ids,
        scale,
        q_at,
        d,
        list,
        picks,
        listed,
        positions[row as u64],
        first,
        kv_heads,
        kv_head,
        width,
    );
}

/// Sizes, the score scale, and keys listed per query.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    listed(self.sizes.heads as u32, self.sizes.kv_heads as u32, self.sizes.width as u32),
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

/// The launch's buffers, the scale, then `counts`.
fn launch(step: &Step, len: usize, scale: f32, counts: &[usize]) -> Vec<Launch> {
    let mut launch = Launch::plain(step, &[len], Shape::Line(len));
    launch
        .args
        .insert(launch.args.len() - 1, lower::Arg::value(&scale));
    launch
        .args
        .extend(counts.iter().map(|n| lower::Arg::count(*n)));
    vec![launch]
}

fn listed_launches(k: &Listed, step: &Step) -> Vec<Launch> {
    let a = &k.sizes;
    let counts = [
        a.queries,
        a.keys.most(),
        a.heads,
        a.kv_heads,
        a.width,
        k.picks,
    ];
    launch(step, k.len(), k.scale, &counts)
}

/// Query rows over `[Seqs, Keys, KvHeads, Width]` keys, by slot.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    listed_rows(self.heads as u32, self.kv_heads as u32, self.width as u32),
    name = "listed_rows",
    sizes = [self.len, self.keys, self.picks],
    work = self.len * self.picks * self.width,
    launches = rows_launches,
)]
pub(crate) struct ListedRows {
    pub len: usize,
    pub keys: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub width: usize,
    pub picks: usize,
    /// Without a list, pick `p` is key `p`.
    pub listed: bool,
    pub scale: f32,
}

fn rows_launches(k: &ListedRows, step: &Step) -> Vec<Launch> {
    let listed = usize::from(k.listed);
    let counts = [k.keys, k.picks, listed];
    launch(step, k.len, k.scale, &counts)
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
    let kernel = ListedRows {
        len: rows * heads * width,
        keys,
        heads,
        kv_heads,
        width,
        picks,
        listed: a.picked,
        scale: a.scale,
    };
    let reads = [q, k, v, ids, slots, positions];
    Some(vec![Draft::new(kernel, reads, kernel.len)])
}

/// Causal sequences whose keys are all present, or ragged rows.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let Some((site, ids)) = Attend::listed(tree, at) else {
        return rows(tree, at);
    };
    if site.sizes.mask != Mask::Causal || site.sizes.keys.runtime() {
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
