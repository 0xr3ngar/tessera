//! Knobs the front end sets once and the engine reads.

use std::path::PathBuf;

/// A GPU that runs kernels. gpukit has its own copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Metal,
    Cuda,
    Rocm,
}

/// Where plans run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// The host running kernel trees in lockstep.
    Host,
    /// The host running native kernels.
    Cpu,
    Gpu(Backend),
}

/// Settings for the planner, the runners and the tools.
#[derive(Clone, Debug)]
pub struct Settings {
    pub backend: Pick,
    /// Lanes per SIMD group the host executor simulates.
    pub host_lanes: usize,
    /// Where calibrate keeps cost samples. `None` stores nothing.
    pub costs: Option<PathBuf>,
    /// A JSON file of weight-name rules, applied last.
    pub names: Option<PathBuf>,
    /// The GPU whose tile rounding the host imitates.
    pub emulate: Option<Backend>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            backend: Pick::Host,
            host_lanes: 32,
            costs: None,
            names: None,
            emulate: None,
        }
    }
}

impl Settings {
    /// The GPU plans run on, or `None` for the host.
    pub fn gpu(&self) -> Option<Backend> {
        match self.backend {
            Pick::Gpu(b) => Some(b),
            Pick::Host | Pick::Cpu => None,
        }
    }
}
