//! spaCy's small English tagger.

use crate::common::{Spec, suite};
use models::tagger::{Config, program};
use trace::Graph;

/// Two features of eight rows, one window layer, block-wide rows.
fn small() -> Graph {
    let c = Config {
        rows: vec![8, 8],
        width: 32,
        depth: 1,
        tags: 32,
        max_context: 16,
    };
    program(&c)
}

fn release() -> Graph {
    let c = Config {
        rows: vec![5000, 1000, 2500, 2500, 50, 50],
        width: 96,
        depth: 4,
        tags: 50,
        max_context: 512,
    };
    program(&c)
}

const SPEC: Spec = Spec {
    name: "tagger",
    release,
    parameters: 1_570_514,
    regions: &[],
    stepper: None,
    small: Some(small),
    files: &[],
    golden: None,
};

suite!(SPEC);
