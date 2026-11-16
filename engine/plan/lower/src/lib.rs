//! Lowers a graph to kernel launches over device buffers.

mod abi;
mod assign;
mod cache;
mod cost;
mod cut;
mod extent;
pub mod form;
mod kernel;
pub mod kernels;
mod plan;
mod recipe;
pub mod rule;
mod schedule;
mod select;
pub mod sites;
pub mod testing;
mod tree;

pub use abi::{Arg, Launch, Shape, Source, ToArg};
pub use assign::assign;
pub use cache::Cache;
pub use codegen::kept;
pub use codegen::{Dialect, LANES, Precision, Terms, Text, float, hash, hip_tiles, terms};
pub use cost::{Cost, Device, Sample};
pub use cut::cut;
pub use engine_settings::{Backend, Pick, Settings};
pub use extent::Extent;
pub use form::{Form, Pins, Weight};
pub use gpukit::{Limits, Platform};
pub use kernel::{Kernel, KernelRef, Needs, UNPROVEN};
pub use kernel_macro::Kernel;
pub use mem::{Buffer, DType, Tensor, View, ViewMut};
pub use plan::{Loaded, Output, Plan, Share, Slot, Step};
pub use recipe::{Launching, Recipe};
pub use rule::{Attn, Draft, RULES, Read, Rule, linked, rules};
pub use schedule::{peak, schedule};
pub use select::{Home, Op, Selected, Val, Value, select};
pub use sim::{Cluster, Fit, Link, Load, Ops, Zone, ZoneKind};
pub use tree::{Node, T, Term, Tree};

use graph::{DimName, Graph};

/// This build's GPU, when it can run kernels here.
pub fn gpu() -> Option<Backend> {
    use gpukit::backend as gk;
    Some(match gk::picked()? {
        gk::Backend::Metal => Backend::Metal,
        gk::Backend::Cuda => Backend::Cuda,
        gk::Backend::Rocm => Backend::Rocm,
    })
}

/// Selects kernels, orders them, then assigns buffers.
pub fn lower(
    g: &Graph,
    entry: usize,
    sizes: &[(DimName, u64)],
    device: &Device,
) -> Result<Plan, Error> {
    lower_upto(g, entry, sizes, &[], device)
}

/// The same, with `upto` dimensions read at launch.
pub(crate) fn lower_upto(
    g: &Graph,
    entry: usize,
    sizes: &[(DimName, u64)],
    upto: &[DimName],
    device: &Device,
) -> Result<Plan, Error> {
    let picked = select::select_upto(g, entry, sizes, upto, device)?;
    Ok(assign(&picked, &schedule(&picked)))
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The graph uses something no rule covers yet.
    #[error("no rule lowers {0}")]
    Unsupported(String),
    /// A dimension has no size for this launch.
    #[error("{0} has no size in this launch")]
    Size(String),
    #[error("gpu: {0}")]
    Gpu(#[from] gpukit::Error),
    /// A view did not fit its buffer or its use.
    #[error("memory: {0}")]
    Memory(#[from] mem::Error),
    /// A runtime dimension was never bound.
    #[error("a runtime dimension has no value")]
    Unbound,
    /// A stored device file could not be read.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// The ranks lost touch.
    #[error("comm: {0}")]
    Comm(#[from] engine_comm::Error),
}
