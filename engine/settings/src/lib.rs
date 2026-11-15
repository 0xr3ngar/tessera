//! Knobs the front end sets once and the engine reads.

pub use gpukit::backend::Backend;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Where plans run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// This build's GPU if one is up, else the host.
    Auto,
    Host,
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
    /// Each model's reference files, by name, for the parity tests.
    pub references: BTreeMap<String, PathBuf>,
    /// A JSON file of weight-name rules, applied last.
    pub names: Option<PathBuf>,
    /// The GPU whose tile rounding the host imitates.
    pub emulate: Option<Backend>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            backend: Pick::Auto,
            host_lanes: 32,
            costs: None,
            references: BTreeMap::new(),
            names: None,
            emulate: None,
        }
    }
}

impl Settings {
    /// The reference files for `model`, where this host has them.
    pub fn reference(&self, model: &str) -> Option<PathBuf> {
        self.references.get(model).cloned()
    }

    /// The GPU plans run on, or `None` for the host.
    pub fn gpu(&self) -> Option<Backend> {
        match self.backend {
            Pick::Host => None,
            Pick::Gpu(b) => Some(b),
            Pick::Auto => gpukit::backend::available().filter(|b| gpukit::backend::ready(*b)),
        }
    }
}
