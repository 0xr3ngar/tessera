//! Slices, repeats, swaps and joins, copied into a new buffer.

use codegen::prelude::*;
use graph::{Op, View};
use lower::form::F32_FORM;
use lower::kernels::Join;
use lower::{Draft, Extent, Home, Kernel, T, Term, Tree};

/// One thread per output `[outer, mid, inner]`, read through strides.
#[kernel]
fn strided(
    x: Buf<F32>,
    mut out: BufMut<F32>,
    len: u32,
    mid: u32,
    inner: u32,
    outer_step: u32,
    mid_step: u32,
    inner_step: u32,
    start: u32,
) {
    if gid() >= len {
        return;
    }
    let i = gid() % inner;
    let m = (gid() / inner) % mid;
    let o = gid() / (inner * mid);
    out[gid() as u64] = x[(start + o * outer_step + m * mid_step + i * inner_step) as u64];
}

/// A view as three nested axes over its input.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(strided, threads = self.len)]
pub(crate) struct Strided {
    pub len: usize,
    pub mid: usize,
    pub inner: usize,
    pub outer_step: usize,
    pub mid_step: usize,
    pub inner_step: usize,
    pub start: usize,
}

/// The view as strides over `shape`, its input's shape.
fn strides(tree: &Tree, view: &View, shape: &[usize], len: usize) -> Option<Strided> {
    let product = |s: &[usize]| s.iter().product::<usize>();
    let at = |axis: u32| axis as usize;
    match view {
        View::Slice { axis, start, .. } => {
            let (a, inner) = (at(*axis), product(&shape[at(*axis) + 1..]));
            // Starts are fixed sizes.
            let start = start.value(&tree.g.dims, &|_| None)? as usize;
            let mid = len / product(&shape[..a]) / inner;
            Some(Strided {
                len,
                mid,
                inner,
                outer_step: shape[a] * inner,
                mid_step: inner,
                inner_step: 1,
                start: start * inner,
            })
        }
        View::Expand { axis, .. } => {
            let inner = product(&shape[at(*axis) + 1..]);
            Some(Strided {
                len,
                mid: len / product(&shape[..at(*axis)]) / inner,
                inner,
                outer_step: inner,
                mid_step: 0,
                inner_step: 1,
                start: 0,
            })
        }
        // Only the last two axes trade places.
        View::Permute(order) => {
            let rank = order.len();
            if rank < 2 {
                return None;
            }
            let last = (rank - 2..rank).rev().map(|a| a as u32);
            let swaps = order[..rank - 2].iter().copied().eq(0..rank as u32 - 2)
                && order[rank - 2..].iter().copied().eq(last);
            let (rows, cols) = (shape[rank - 2], shape[rank - 1]);
            swaps.then_some(Strided {
                len,
                mid: cols,
                inner: rows,
                outer_step: rows * cols,
                mid_step: 1,
                inner_step: cols,
                start: 0,
            })
        }
        View::Reshape(_) | View::Concat => None,
    }
}

/// Views of computed or fed f32 values.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let Term::Op(Op::View(view)) = &node.term else {
        return None;
    };
    // Weight views belong to the rules that read them.
    let ours = |t: &T| {
        let n = tree.node(*t);
        n.form.name() == F32_FORM.name()
            && !matches!(n.term, Term::Leaf(Home::Weight(_) | Home::Part(..)))
    };
    if !node.args.iter().all(ours) {
        return None;
    }
    let x = node.args[0];
    let len = tree.elements(at);
    if *view == View::Concat {
        let y = node.args[1];
        let side = |t: T| *tree.node(t).shape.last().unwrap_or(&1);
        let join = Join {
            left: side(x),
            right: side(y),
            rows: Extent::Fixed(tree.elements(x) / side(x).max(1)),
        };
        return Some(vec![Draft::new(join, [x, y], len)]);
    }
    let kernel = strides(tree, view, &tree.node(x).shape, len)?;
    Some(vec![Draft::new(kernel, [x], len)])
}

lower::rule!("view", propose);
