//! DeepSeek-V3.

use models::mla::{Config, program, stepper};
use parity::{Spec, Stepper, suite};
use trace::Composite::{Attention, Mlp, Moe};
use trace::Graph;

fn small() -> Graph {
    program(&models::small::deepseek())
}

pub fn release() -> Graph {
    program(&Config::deepseek_v3(8, 163_840))
}

const STEPPER: Stepper = Stepper {
    small: || stepper(&models::small::deepseek()),
    release: || stepper(&Config::deepseek_v3(1, 1024)),
};

const SPEC: Spec = Spec {
    name: "deepseek",
    release,
    parameters: 671_026_419_200,
    regions: &[(Attention, 61), (Mlp, 3), (Moe, 58)],
    stepper: Some(STEPPER),
    small: Some(small),
    files: &[],
    golden: None,
};

suite!(SPEC);
