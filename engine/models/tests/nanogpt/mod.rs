//! A small nanogpt.

mod golden;
mod regions_test;

use crate::common::{Check, Compare, Golden, Spec, Tolerance, suite};
use lower::form::F32_FORM;
use models::nanogpt::{Config, program};
use trace::Composite::{Attention, Mlp};
use trace::Graph;

pub fn config() -> Config {
    models::small::nanogpt()
}

pub fn release() -> Graph {
    program(&config())
}

const SPEC: Spec = Spec {
    name: "nanogpt",
    release,
    parameters: 1768,
    regions: &[(Attention, 2), (Mlp, 2)],
    stepper: None,
    small: None,
    files: &[],
    golden: Some(Golden {
        record: Some(golden::read),
        form: F32_FORM,
        checks: &[Check {
            name: "logits",
            compare: Compare::Tokens,
            tolerance: Tolerance {
                max_abs: 1e-5,
                cos: 0.0,
                top: false,
            },
        }],
        anywhere: true,
    }),
};

suite!(SPEC);
