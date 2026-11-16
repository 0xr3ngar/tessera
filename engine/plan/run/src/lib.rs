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

/// A failure while loading or running a plan.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Lower(#[from] lower::Error),
    #[error("gpu: {0}")]
    Gpu(#[from] gpukit::Error),
    #[error("memory: {0}")]
    Memory(#[from] mem::Error),
    #[error("comm: {0}")]
    Comm(#[from] engine_comm::Error),
    #[error("{0} is hand-written")]
    Handwritten(&'static str),
    #[error("{arg:?} as {param:?} on the host")]
    HostArg { arg: Arg, param: Param },
    #[error("jit: {0}")]
    Jit(&'static str),
    #[error("plans out of rank order")]
    RankOrder,
    #[error("ranks disagree on an output")]
    OutputsDisagree,
    #[error("rank {plan} of {plans} on rank {rank} of {ranks}")]
    RankMismatch {
        plan: u32,
        plans: u32,
        rank: u32,
        ranks: u32,
    },
    #[error("a sum of {0} floats does not fit the device")]
    SumTooLong(usize),
    #[error("a plan over several zones")]
    Zones,
    #[error("a fixed kernel on this runner")]
    FixedKernel,
    #[error("weight {weight} does not fit {form}")]
    WeightSize { weight: u32, form: &'static str },
    #[error("weight {0} has the wrong length")]
    WeightLen(u32),
    #[error("weight {weight} as {form} split on axis {axis}")]
    WeightSplit {
        weight: u32,
        form: &'static str,
        axis: u32,
    },
    #[error("the input list does not match the plan")]
    InputCount,
    #[error("input {0} does not match the plan")]
    InputShape(usize),
}

use codegen::Precision;
use codegen::ir::Param;
use engine_comm::Pinning;
use engine_settings::Backend;
use engine_settings::Settings;
use lower::{Arg, Plan};
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
