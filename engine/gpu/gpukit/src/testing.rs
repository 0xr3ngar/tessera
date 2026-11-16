//! Whether each GPU runtime is up on this host.

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
