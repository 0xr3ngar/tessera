//! Granite 3.1 1B-A400M: a Llama decoder with routed experts.

use crate::common::{LOGITS, Spec, Stepper, decoder, small_stack, suite};
use models::decoder::{Config, Moe, Scales, program, stepper};
use trace::Composite::{Attention, Moe as Routed};
use trace::Graph;

/// Two small layers with four experts, two per row.
fn small_config() -> Config {
    Config {
        stack: small_stack(64, 32, 128),
        kv_heads: 1,
        rope_theta: 1e4,
        eps: 1e-6,
        experts: 4,
        top_k: 2,
        scales: Scales {
            embed: 2.0,
            attention: Some(0.1),
            residual: 0.5,
            logits: 0.25,
        },
    }
}

decoder!((), Moe, small_config(), Config::granite);

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
