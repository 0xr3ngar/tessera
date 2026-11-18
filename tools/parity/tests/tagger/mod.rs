//! spaCy's small English tagger.

use sensai_models::tagger::{Config, program};
use sensai_parity::{Spec, suite};
use sensai_trace::Graph;

/// Two features of eight rows, one window layer.
fn small() -> Graph {
    program(&sensai_models::small::tagger())
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
