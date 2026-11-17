//! Split nodes grouped by the sum that joins them.

use crate::model::Model;
use crate::stages::RankView;
use graph::{Op, Placement, TensorType};

/// A node's type, or its weight's type.
pub fn ty(model: &Model, i: usize) -> &TensorType {
    let g = &model.sharded;
    match g.nodes[i].op {
        Op::Weight(w) => &g.weights[w as usize].ty,
        _ => &g.nodes[i].ty,
    }
}

/// Per sharded node, the added sum its split values reach.
pub fn groups(model: &Model) -> Vec<Option<usize>> {
    let g = &model.sharded;
    let mut readers = vec![Vec::new(); g.nodes.len()];
    for (i, n) in g.nodes.iter().enumerate() {
        for a in &n.args {
            readers[a.0 as usize].push(i);
        }
    }
    let mut out = vec![None; g.nodes.len()];
    for i in (0..g.nodes.len()).rev() {
        if model.added[i] || ty(model, i).placement == Placement::Replicated {
            continue;
        }
        for &r in &readers[i] {
            let found = match model.added[r] {
                true => Some(r),
                false => out[r],
            };
            if found.is_some() {
                out[i] = found;
                break;
            }
        }
    }
    out
}

/// What node `i` holds on one rank.
pub fn local(model: &Model, rank: &RankView, i: usize) -> String {
    if let Op::Weight(w) = model.sharded.nodes[i].op {
        let held = rank.held.iter().find(|h| h.weight == w);
        return match held.and_then(|h| h.share) {
            Some([axis, first, count, whole]) => {
                format!("axis {axis}: {first}..{} of {whole}", first + count)
            }
            None => "whole".into(),
        };
    }
    match &rank.shapes[i] {
        Some(shape) => format!("{shape:?} on {}", rank.name),
        None => format!("not needed on {}", rank.name),
    }
}
