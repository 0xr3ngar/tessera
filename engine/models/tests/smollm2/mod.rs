//! SmolLM2-360M: a Llama decoder with tied embeddings.

use crate::common::{LOGITS, Spec, Stepper, decoder, small_stack, suite};
use models::Stack;
use models::decoder::{Config, program, stepper};
use trace::Composite::{Attention, Mlp};
use trace::Graph;

/// The 360M checkpoint's sizes.
fn config(max_seqs: u64, max_context: u64) -> Config {
    Config {
        stack: Stack::sized([49_152, 960, 32, 15, 64, 2560], max_seqs, max_context),
        kv_heads: 5,
        rope_theta: 1e5,
        eps: 1e-5,
    }
}

/// Two small layers with no head norms.
fn small_config() -> Config {
    Config {
        stack: small_stack(64, 32, 128),
        kv_heads: 1,
        rope_theta: 1e4,
        eps: 1e-5,
    }
}

decoder!((), small_config(), config);

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
