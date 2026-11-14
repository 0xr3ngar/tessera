//! A value crossing from one zone to another.

use crate::Kernel;
use crate::kernels::Map;

/// Copies a value into the step's zone.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = crate,
    name = "move",
    sizes = [],
    // The runner copies, so nothing compiles.
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
)]
pub(crate) struct Move;
