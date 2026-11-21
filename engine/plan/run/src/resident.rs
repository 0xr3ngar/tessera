//! Host interface for persistent `tessera::table*` plan steps.

use crate::Error;
use std::collections::HashMap;
use tessera_graph::DimName;
use tessera_lower::Plan;

/// Device buffer addresses and bound sizes for one resident launch.
pub struct ResidentCall<'a> {
    pub step: usize,
    pub plan: &'a Plan,
    /// Base device address per buffer in `plan.buffers`.
    pub buffers: &'a [u64],
    /// Runtime dimension sizes bound for this launch.
    pub bound: &'a HashMap<DimName, u32>,
    /// Per-slot kept-row lengths, or empty if unset.
    pub kv_lens: &'a [u32],
}

/// Runs persistent `tessera::table*` steps on a device.
pub trait ResidentExec {
    /// Executes resident step `call.step` and waits for completion.
    fn run(&mut self, call: &ResidentCall<'_>) -> Result<(), Error>;
    /// Updates cached state when the runner adopts `plan`.
    fn replan(&mut self, _plan: &Plan) -> Result<(), Error> {
        Ok(())
    }
}
