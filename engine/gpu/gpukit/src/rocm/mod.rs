//! ROCm backend. HIP and hiprtc load at runtime with dlopen.

mod context;
pub mod hip;

pub use crate::launch::{KernelArgs, Launch, launch_1d, launch_grid};
pub use context::{Context, ContextConfig, Graph, Kernel, Module};

crate::device::backend_api!(Context, Kernel);
