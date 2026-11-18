//! A 3x3 convolution: gathered neighbours and a matmul, fused.

use codegen::prelude::*;
use graph::{Op, Select, View};
use lower::form::F32_FORM;
use lower::{Draft, Kernel, T, Term, Tree};

/// Pixels a 3x3 kernel reads.
const TAPS: u32 = 9;

/// One thread per output channel. Missing pixels read zero.
#[kernel]
fn neighbours(
    x: Buf<F32>,
    ids: Buf<U32>,
    w: Buf<F32>,
    mut out: BufMut<F32>,
    len: u32,
    pixels: u32,
    cin: u32,
    cout: u32,
) {
    if gid() >= len {
        return;
    }
    let co = gid() % cout;
    let row = gid() / cout;
    let p = row % pixels;
    let first = (row - p) * cin;
    let mut sum = 0.0f32;
    for t in 0..TAPS {
        let id = ids[(p * TAPS + t) as u64];
        if id < pixels {
            let at = first + id * cin;
            let wat = (co * TAPS + t) * cin;
            for c in 0..cin {
                sum += x[(at + c) as u64] * w[(wat + c) as u64];
            }
        }
    }
    out[gid() as u64] = sum;
}

/// `[Seqs, pixels, cin]` through `TAPS` neighbours to `cout` channels.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(neighbours, threads = self.len)]
pub(crate) struct Neighbours {
    pub len: usize,
    pub pixels: usize,
    pub cin: usize,
    pub cout: usize,
}

/// `t`'s one argument, if `t` is the view `want`.
fn view(tree: &Tree, t: T, want: &View) -> Option<T> {
    let n = tree.node(t);
    (n.term == Term::Op(Op::View(want.clone()))).then(|| n.args[0])
}

/// Matmuls of `reshape(permute(gather(permute(x), ids)))`, as models trace them.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::Matmul) {
        return None;
    }
    let (flat, w) = (node.args[0], node.args[1]);
    let Term::Op(Op::View(View::Reshape(_))) = &tree.node(flat).term else {
        return None;
    };
    let near = view(
        tree,
        tree.node(flat).args[0],
        &View::Permute(vec![2, 0, 1, 3]),
    )?;
    let g = tree.node(near);
    if g.term != Term::Op(Op::Select(Select::Gather)) {
        return None;
    }
    let (table, ids) = (g.args[0], g.args[1]);
    let x = view(tree, table, &View::Permute(vec![1, 0, 2]))?;
    let [_, pixels, cin] = tree.node(x).shape[..] else {
        return None;
    };
    let plain = |t: T| tree.node(t).form.name() == F32_FORM.name();
    let taps = tree.node(ids).shape[..] == [pixels, TAPS as usize];
    if !(taps && plain(x) && plain(w)) {
        return None;
    }
    let len = tree.elements(at);
    let cout = len * cin / tree.elements(x);
    let k = Neighbours {
        len,
        pixels,
        cin,
        cout,
    };
    Some(vec![Draft::new(k, [x, ids, w], len)])
}

lower::rule!("neighbours", propose);
