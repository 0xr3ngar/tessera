//! The mean or weighted sum over one axis.

use codegen::prelude::*;
use graph::Op;
use lower::{Draft, Extent, Kernel, T, Term, Tree};

/// Sums `count` strided values, in order. `w` weights each.
#[kernel]
fn mean<const WEIGHTED: bool>(
    x: Buf<F32>,
    w: Buf<F32>,
    mut out: BufMut<F32>,
    outer: u32,
    count: u32,
    inner: u32,
) {
    if gid() >= outer * inner {
        return;
    }
    let o = gid() / inner;
    let i = gid() % inner;
    let mut sum = 0.0f32;
    for t in 0..count {
        let mut v = x[((o * count + t) * inner + i) as u64];
        if WEIGHTED {
            v = v * w[(o * count + t) as u64];
        }
        sum += v;
    }
    if WEIGHTED {
        out[gid() as u64] = sum;
    } else {
        out[gid() as u64] = sum / count as f32;
    }
}

/// `x` is `[outer, count, inner]`. The middle axis goes.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    mean,
    variant = self.weighted,
    name = match self.weighted {
        true => "mix",
        false => "mean",
    },
    threads = self.outer.times(self.inner),
    work = self.outer.most() * self.count.most() * self.inner,
)]
pub struct Mean {
    pub outer: Extent,
    pub count: Extent,
    pub inner: usize,
    /// Weights from the second read, else the plain mean.
    pub weighted: bool,
}

/// Axes before the mean fold into one, those after too.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let (axis, weights) = match node.term {
        Term::Op(Op::Mean(axis)) => (axis, None),
        Term::Op(Op::Mix(axis)) => (axis, node.args.get(1).copied()),
        _ => return None,
    };
    let axis = axis as usize;
    let x = tree.node(node.args[0]);
    let before = &x.extents[..axis];
    let outer = match before {
        [one] => *one,
        _ => Extent::Fixed(x.shape[..axis].iter().product()),
    };
    let inner = x.shape[axis + 1..].iter().product();
    let mean = Mean {
        outer,
        count: x.extents[axis],
        inner,
        weighted: weights.is_some(),
    };
    // The plain mean reads `x` again, unused.
    let reads = [node.args[0], weights.unwrap_or(node.args[0])];
    Some(vec![Draft::new(mean, reads, outer.most() * inner)])
}

lower::rule!("mean", propose);
