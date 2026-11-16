//! The settings tests run under: host, then a ready GPU.

use crate::{Backend, Pick, Settings, gpu};

/// Settings for plans on the host.
pub fn host() -> Settings {
    Settings::default()
}

/// The host, then this build's GPU when one is ready.
pub fn runs() -> Vec<Settings> {
    let mut runs = vec![host()];
    if let Some(b) = gpu() {
        runs.push(Settings {
            backend: Pick::Gpu(b),
            ..host()
        });
    }
    runs
}

/// `runs`, then the host imitating each GPU's rounding.
pub fn emulations() -> Vec<Settings> {
    let gpus = [Backend::Metal, Backend::Cuda, Backend::Rocm];
    let imitated = gpus.map(|b| Settings {
        emulate: Some(b),
        ..host()
    });
    runs().into_iter().chain(imitated).collect()
}
