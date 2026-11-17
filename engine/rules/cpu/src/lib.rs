//! CPU kernels for matmuls: short bodies over `dot`.

mod dense;
mod norm;

pub use dense::{Dense, Picked, Tiled};
pub use norm::RmsNorm;
