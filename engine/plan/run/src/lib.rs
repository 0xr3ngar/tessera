//! Runs lowered plans on the host or a GPU.

mod cpu;
#[cfg(any(target_os = "macos", feature = "cuda", feature = "rocm"))]
mod device;
mod held;
#[cfg(any(target_os = "macos", feature = "cuda", feature = "rocm"))]
mod kernels;
mod loopback;
mod rank;
mod run;

pub use held::Held;
pub use loopback::Loopback;
pub use rank::Rank;
pub use run::{Base, Run};

use codegen::Precision;
use engine_comm::Pinning;
use engine_settings::Backend;
use engine_settings::Settings;
use lower::{Error, Plan};
use std::sync::Arc;

/// How `backend`'s tiles round.
pub fn precision(backend: Backend) -> Precision {
    match backend {
        Backend::Metal => Precision::METAL,
        Backend::Cuda => Precision::CUDA,
        Backend::Rocm => Precision::HIP,
    }
}

/// Pinned memory on the GPU `settings` picks, if any.
pub fn pinning(settings: &Settings) -> Option<Arc<dyn Pinning>> {
    match settings.gpu() {
        #[cfg(feature = "cuda")]
        Some(Backend::Cuda) => {
            let ctx = gpukit::cuda::cached_context().ok()?;
            Some(Arc::new(device::Pin(ctx)))
        }
        #[cfg(feature = "rocm")]
        Some(Backend::Rocm) => {
            let ctx = gpukit::rocm::cached_context().ok()?;
            Some(Arc::new(device::Pin(ctx)))
        }
        _ => None,
    }
}

/// A runner on the GPU `settings` picks, or the host.
pub fn open(plan: Plan, settings: &Settings) -> Result<Box<dyn Run>, Error> {
    match settings.gpu() {
        #[cfg(target_os = "macos")]
        Some(Backend::Metal) => Ok(Box::new(device::Runner::<gpukit::metal::Context>::new(
            plan,
        )?)),
        #[cfg(feature = "cuda")]
        Some(Backend::Cuda) => Ok(Box::new(device::Runner::<gpukit::cuda::Context>::new(
            plan,
        )?)),
        #[cfg(feature = "rocm")]
        Some(Backend::Rocm) => Ok(Box::new(device::Runner::<gpukit::rocm::Context>::new(
            plan,
        )?)),
        _ => Ok(Box::new(cpu::Runner::new(
            plan,
            settings.host_lanes,
            settings.emulate.map_or(Precision::EXACT, precision),
        )?)),
    }
}
