//! SmolLM2-360M: a Llama decoder with tied embeddings.

use crate::common::{LOGITS, Spec, Stepper, decoder, suite};
use models::decoder::{Config, Swiglu, program, stepper};
use trace::Composite::{Attention, Mlp};
use trace::Graph;

decoder!((), Swiglu, models::small::smollm2(), Config::smollm2);

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
