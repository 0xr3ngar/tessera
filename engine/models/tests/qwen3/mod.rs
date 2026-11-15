//! Qwen3-0.6B.

mod shard_test;

use crate::common::{LOGITS, Spec, Stepper, decoder, small_stack, suite};
use models::decoder::{Config, Scales, Swiglu, program, stepper};
use trace::Composite::{Attention, Mlp};
use trace::{Graph, Norm};

/// Two small layers, wide enough for every block form.
fn small_config() -> Config {
    Config {
        stack: small_stack(64, 32, 128),
        kv_heads: 1,
        rope_theta: 1e4,
        eps: 1e-6,
        experts: 0,
        top_k: 0,
        scales: Scales::default(),
    }
}

decoder!(Norm, Swiglu, small_config(), Config::qwen3);

const SPEC: Spec = Spec {
    name: "qwen3",
    release,
    parameters: 596_049_920,
    regions: &[(Attention, 28), (Mlp, 28)],
    small: Some(small),
    stepper: Some(STEPPER),
    files: &[
        (include_str!("manifests/hf.json"), release),
        (include_str!("manifests/gguf.json"), release),
    ],
    golden: Some(LOGITS),
};

suite!(SPEC);
