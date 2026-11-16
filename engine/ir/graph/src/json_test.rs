use crate::verify_test::{mlp, push};
use crate::*;

fn with_scale(scale: f32) -> Graph {
    let mut g = mlp();
    let last = NodeId(g.nodes.len() as u32 - 1);
    let y = push(&mut g, Op::Row(Row(vec![Stage::Scale(scale)])), &[last]);
    g.entries[0].outputs[0].node = y;
    g
}

#[test]
fn round_trip() {
    let g = with_scale(1.0 / 3f32.sqrt());
    let back = Graph::from_json(&g.to_json()).unwrap();
    assert_eq!(back, g);
}

#[test]
fn version_checked() {
    let err = Graph::from_json(r#"{"version": 99}"#).unwrap_err();
    assert!(matches!(err, FormatError::Version { found: 99 }), "{err}");
}

#[test]
fn hash_tracks_content() {
    assert_ne!(with_scale(0.5).hash(), with_scale(0.25).hash());
}
