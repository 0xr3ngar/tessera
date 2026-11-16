//! Granite 3.1 1B-A400M: a Llama decoder with routed experts.

mod shard_test;

use models::decoder::{Config, Moe, program, stepper};
use parity::{LOGITS, Spec, Stepper, decoder, suite};
use trace::Composite::{Attention, Moe as Routed};
use trace::Graph;

decoder!((), Moe, models::small::granite(), Config::granite);

const SPEC: Spec = Spec {
    name: "granite",
    release,
    parameters: 1_334_628_352,
    regions: &[(Attention, 24), (Routed, 24)],
    small: Some(small),
    stepper: Some(STEPPER),
    files: &[(include_str!("manifests/hf.json"), release)],
    golden: Some(LOGITS),
};

suite!(SPEC);
