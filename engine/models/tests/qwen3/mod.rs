//! Qwen3-0.6B.

mod shard_test;

use crate::common::{LOGITS, Spec, Stepper, decoder, suite};
use models::decoder::{Config, Swiglu, program, stepper};
use trace::Composite::{Attention, Mlp};
use trace::{Graph, Norm};

decoder!(Norm, Swiglu, models::small::qwen3(), Config::qwen3);

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
