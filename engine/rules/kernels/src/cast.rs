//! Whole floats as indices, such as fixed id tables.

use codegen::prelude::*;
use graph::{DType, Op};
use lower::form::F32_FORM;
use lower::{Draft, Kernel, T, Term, Tree};

/// One thread per element. Negatives and NaN give zero.
#[kernel]
fn cast(x: Buf<F32>, mut out: BufMut<U32>, len: u32) {
    if gid() >= len {
        return;
    }
    let v = x[gid() as u64];
    let mut id = 0u32;
    if v > 0.0f32 {
        id = v as u32;
    }
    out[gid() as u64] = id;
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(cast, threads = self.len)]
pub struct Cast {
    pub len: usize,
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::Cast(DType::U32)) {
        return None;
    }
    let x = node.args[0];
    if tree.node(x).form.name() != F32_FORM.name() {
        return None;
    }
    let len = tree.elements(at);
    Some(vec![Draft::new(Cast { len }, [x], len)])
}

lower::rule!("cast", propose);
