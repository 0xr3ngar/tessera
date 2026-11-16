//! Which GPU backend this build dispatches to.

/// A backend that can run kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, derive_more::Display)]
#[serde(rename_all = "lowercase")]
#[display(rename_all = "lowercase")]
pub enum Backend {
    Metal,
    Cuda,
    Rocm,
}

/// The backend this build uses, if any.
pub fn available() -> Option<Backend> {
    if cfg!(target_os = "macos") {
        return Some(Backend::Metal);
    }
    // Runtimes load only at dispatch, unless both are built.
    match (cfg!(feature = "cuda"), cfg!(feature = "rocm")) {
        (true, true) if !crate::testing::cuda_ready() && crate::testing::rocm_ready() => {
            Some(Backend::Rocm)
        }
        (true, _) => Some(Backend::Cuda),
        (false, true) => Some(Backend::Rocm),
        (false, false) => None,
    }
}

/// Whether `backend` can run kernels here.
pub fn ready(backend: Backend) -> bool {
    match backend {
        Backend::Metal => cfg!(target_os = "macos"),
        Backend::Cuda => crate::testing::cuda_ready(),
        Backend::Rocm => crate::testing::rocm_ready(),
    }
}
