//! Whole floats as indices, such as fixed id tables.

use sensai_codegen::prelude::*;
use sensai_graph::{DType, Op};
use sensai_lower::form::Load as Stored;
use sensai_lower::{Draft, Kernel, T, Term, Tree};

/// One thread per element. Negatives and NaN give zero.
#[kernel]
fn cast(
    #[form(form)] x: Buf<Dyn>,
    mut out: BufMut<U32>,
    len: u32,
    #[build] form: &'static dyn Stored,
) {
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
#[launch(cast(self.form), name = self.form.suffixed("cast"), threads = self.len, accepts = cast_accepts)]
pub struct Cast {
    pub len: usize,
    /// How the input is stored.
    pub form: &'static dyn Stored,
}

fn cast_accepts(k: &Cast, input: usize, form: &dyn Stored) -> bool {
    k.form.taken(input, 0, form)
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::Cast(DType::U32)) {
        return None;
    }
    let x = node.args[0];
    let len = tree.elements(at);
    let cast = Cast {
        len,
        form: tree.node(x).form,
    };
    Some(vec![Draft::new(cast, [x], len)])
}

sensai_lower::rule!("cast", propose);
