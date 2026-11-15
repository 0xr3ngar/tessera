//! DeepSeek-V3.

use crate::common::{Spec, suite};
use models::mla::{Config, program};
use trace::Composite::{Attention, Mlp, Moe};
use trace::Graph;

pub fn release() -> Graph {
    program(&Config::deepseek_v3(8, 163_840))
}

const SPEC: Spec = Spec {
    name: "deepseek",
    release,
    parameters: 671_026_419_200,
    regions: &[(Attention, 61), (Mlp, 3), (Moe, 58)],
    stepper: None,
    small: None,
    files: &[],
    golden: None,
};

suite!(SPEC);
