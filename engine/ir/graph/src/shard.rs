//! Tensor parallel: columns split out, rows sum back on return.

use crate::infer::infer;
use crate::node::{Graph, Node, NodeId, Op, Row, Stage};
use crate::numerics::Numerics;
use crate::types::Placement;
use crate::verify::{VerifyError, verify};

/// Splits each residual-bound matmul the type rules allow.
pub fn shard(g: &Graph) -> Result<Graph, VerifyError> {
    let mut out = g.clone();
    let mut at = 0;
    while at < out.nodes.len() {
        let id = NodeId(at as u32);
        at += 1;
        if returns(&out, id)
            && let Ok(typed) = retype(&split(&out, id))
        {
            out = typed;
            // The sum is the new node behind this one.
            at += 1;
        }
    }
    verify(&out)?;
    Ok(out)
}

/// A matmul of a whole weight, read by an add.
fn returns(g: &Graph, id: NodeId) -> bool {
    let node = g.node(id);
    if node.op != Op::Matmul || !whole_weight(g, node.args[1]) {
        return false;
    }
    let adds = |n: &Node| matches!(&n.op, Op::Row(r) if r.0.contains(&Stage::Add));
    g.nodes.iter().any(|n| adds(n) && n.args.contains(&id))
}

/// A rank-two weight no earlier split has claimed.
fn whole_weight(g: &Graph, id: NodeId) -> bool {
    let Op::Weight(w) = g.node(id).op else {
        return false;
    };
    let ty = &g.weights[w as usize].ty;
    ty.rank() == 2 && ty.placement == Placement::Replicated
}

/// `g` with `id` split over rows, its feeders over columns.
fn split(g: &Graph, id: NodeId) -> Graph {
    let mut out = g.clone();
    let mut pending = vec![g.node(id).args[0]];
    let mut seen = vec![false; g.nodes.len()];
    while let Some(n) = pending.pop() {
        if std::mem::replace(&mut seen[n.0 as usize], true) {
            continue;
        }
        let node = g.node(n);
        if node.op == Op::Matmul && whole_weight(g, node.args[1]) {
            place(&mut out, node.args[1], 0);
            continue;
        }
        pending.extend(node.args.iter().copied());
    }
    place(&mut out, g.node(id).args[1], 1);
    let sum = Node {
        op: Op::Row(Row(vec![Stage::AllReduce])),
        args: vec![id],
        ty: g.node(id).ty.clone(),
        numerics: Some(Numerics::default()),
    };
    insert(&mut out, id, sum);
    out
}

fn place(g: &mut Graph, weight: NodeId, axis: u32) {
    let Op::Weight(w) = g.node(weight).op else {
        unreachable!("checked by whole_weight")
    };
    g.weights[w as usize].ty.placement = Placement::Split { axis };
}

/// Puts `node` after `at`. Reads of `at` move to it.
fn insert(g: &mut Graph, at: NodeId, node: Node) {
    let new = NodeId(at.0 + 1);
    let shift = |n: &mut NodeId| {
        if *n >= new {
            n.0 += 1;
        }
    };
    let redirect = |n: &mut NodeId| {
        if *n == at {
            *n = new;
        }
    };
    for n in &mut g.nodes[new.0 as usize..] {
        n.args.iter_mut().for_each(|a| {
            shift(a);
            redirect(a);
        });
    }
    g.nodes.insert(new.0 as usize, node);
    for e in &mut g.entries {
        e.outputs.iter_mut().for_each(|o| {
            shift(&mut o.node);
            redirect(&mut o.node);
        });
    }
    for r in &mut g.regions {
        shift(&mut r.start);
        // The end is past the last, so it moves.
        shift(&mut r.end);
        for n in r.inputs.iter_mut().chain(&mut r.outputs) {
            shift(n);
            redirect(n);
        }
    }
}

/// Records what every op now gives, in order.
fn retype(g: &Graph) -> Result<Graph, ()> {
    let mut out = g.clone();
    for i in 0..out.nodes.len() {
        let args: Vec<_> = out.nodes[i]
            .args
            .iter()
            .map(|a| &out.nodes[a.0 as usize].ty)
            .collect();
        let ty = infer(&out, &out.nodes[i].op, &args).map_err(|_| ())?;
        out.nodes[i].ty = ty;
    }
    Ok(out)
}
