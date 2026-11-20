//! Region details the spec does not state.

use super::release;
use tessera_trace::DimName::*;
use tessera_trace::Shape;

#[test]
fn shared_keys() {
    let g = release();
    // Indexing layers and their attention hand keys to later layers.
    let shared = g.regions.iter().filter(|r| r.outputs.len() == 2).count();
    assert_eq!(shared, 42);
    let logits = g.node(g.entries[0].outputs[0].node);
    assert_eq!(logits.ty.shape, Shape::from([Seqs, Context, Vocab]));
}
