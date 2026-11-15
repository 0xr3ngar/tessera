//! Attention splits by tensor, the experts by matrix.

use super::small;
use graph::{Placement, Stage, shard};

#[test]
fn routed_experts() {
    let g = shard(&small()).unwrap();
    let placed = |p: Placement| g.weights.iter().filter(|w| w.ty.placement == p).count();
    // Per layer: q, k, v and both sets, then o.
    assert_eq!(placed(Placement::Split { axis: 0 }), 10);
    assert_eq!(placed(Placement::Split { axis: 1 }), 2);
    let sums = |n: &graph::Node| matches!(&n.op, graph::Op::Row(r) if r.0 == [Stage::AllReduce]);
    assert_eq!(g.nodes.iter().filter(|n| sums(n)).count(), 4);
    let routed = g
        .nodes
        .iter()
        .filter(|n| n.ty.placement == Placement::Routed);
    assert!(routed.count() > 0);
}
