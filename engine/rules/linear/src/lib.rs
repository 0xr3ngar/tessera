//! Generated matmul kernels, registered as lowering rules.

mod lanes;
mod product;
mod thin;
mod tiled;

pub use product::Product;
pub use tiled::{SCHEDULES, Schedule, Tiled};
