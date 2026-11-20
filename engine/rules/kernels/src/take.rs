//! Entries of each row at given indices.

use tessera_codegen::prelude::*;
use tessera_graph::{Op, Select};
use tessera_lower::{Draft, Kernel, T, Term, Tree};

/// One thread per output. Ids past the row read zero.
#[kernel]
fn take(x: Buf<F32>, ids: Buf<U32>, mut out: BufMut<F32>, len: u32, width: u32, picks: u32) {
    if gid() >= len {
        return;
    }
    let i = ids[gid() as u64];
    let mut v = 0.0f32;
    if i < width {
        v = x[(gid() / picks) as u64 * width as u64 + i as u64];
    }
    out[gid() as u64] = v;
}

/// `picks` ids per row of `width`.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(take, threads = self.len)]
pub struct Take {
    pub len: usize,
    pub width: usize,
    pub picks: usize,
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::Select(Select::Take)) {
        return None;
    }
    let (x, ids) = (node.args[0], node.args[1]);
    let width = *tree.node(x).shape.last()?;
    let len = tree.elements(ids);
    let take = Take {
        len,
        width,
        picks: len / (tree.elements(x) / width).max(1),
    };
    Some(vec![Draft::new(take, [x, ids], len)])
}

tessera_lower::rule!("take", propose);
