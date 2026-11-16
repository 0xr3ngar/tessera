//! GPU readiness, and the settings tests run under.

use crate::backend::{self, Backend};
use engine_settings::{Pick, Settings};

/// Whether a CUDA driver loads on this host.
pub(crate) fn cuda_ready() -> bool {
    #[cfg(feature = "cuda")]
    return crate::cuda::driver::driver().is_ok();
    #[cfg(not(feature = "cuda"))]
    false
}

/// Whether a HIP runtime loads and sees a device.
pub(crate) fn rocm_ready() -> bool {
    #[cfg(feature = "rocm")]
    return crate::rocm::cached_context().is_ok();
    #[cfg(not(feature = "rocm"))]
    false
}

/// Whether a hosted CI runner is running this process.
#[allow(clippy::disallowed_methods, reason = "the one variable tests read")]
pub(crate) fn ci() -> bool {
    std::env::var_os("CI").is_some()
}

/// Settings for plans on the host.
pub fn host() -> Settings {
    Settings::default()
}

/// The host, then this build's GPU when one is ready.
pub fn runs() -> Vec<Settings> {
    let mut runs = vec![host()];
    if let Some(b) = backend::picked() {
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
