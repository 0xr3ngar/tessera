//! SmolLM2-360M: a Llama decoder with tied embeddings.

use crate::common::{LOGITS, Spec, Stepper, decoder, small_stack, suite};
use models::decoder::{Config, program, stepper};
use trace::Composite::{Attention, Mlp};
use trace::Graph;

/// Two small layers with no head norms.
fn small_config() -> Config {
    Config {
        stack: small_stack(64, 32, 128),
        kv_heads: 1,
        rope_theta: 1e4,
        eps: 1e-5,
    }
}

decoder!((), small_config(), Config::smollm2);

const SPEC: Spec = Spec {
    name: "smollm2",
    release,
    parameters: 361_821_120,
    regions: &[(Attention, 32), (Mlp, 32)],
    small: Some(small),
    stepper: Some(STEPPER),
    files: &[(include_str!("manifests/hf.json"), release)],
    golden: Some(LOGITS),
};

suite!(SPEC);
