//! GLM-5.3.

mod regions_test;

use models::mla::{Config, program, stepper};
use parity::{Spec, Stepper, suite};
use trace::Composite::{Attention, Mlp, Moe};
use trace::Graph;

fn small() -> Graph {
    program(&models::small::glm53())
}

pub fn release() -> Graph {
    program(&Config::glm53(8, 425_984))
}

const STEPPER: Stepper = Stepper {
    small: || stepper(&models::small::glm53()),
    release: || stepper(&Config::glm53(1, 1024)),
};

const SPEC: Spec = Spec {
    name: "glm53",
    release,
    parameters: 743_377_019_904,
    regions: &[(Attention, 78), (Mlp, 3), (Moe, 75)],
    stepper: Some(STEPPER),
    small: Some(small),
    files: &[],
    golden: None,
};

suite!(SPEC);
