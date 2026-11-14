//! Kernels and their lowering rules. Linking this crate registers them.

mod argmax;
mod attend;
mod dense;
mod flash;
mod gather;
mod hadamard;
mod join;
mod lane_norm;
mod mean;
mod rms_norm;
mod rope;
mod softmax;

pub use attend::Attend;
pub use dense::Causal;
pub use gather::Gather;
pub use hadamard::Hadamard;
pub use mean::Mean;
pub use rms_norm::RmsNorm;
pub use rope::Rope;
pub use softmax::Softmax;

use sim::{Load, Ops};

/// One thread per row of `width`, along that row alone.
pub(crate) fn per_row(rows: usize, width: usize, trip: Ops, shared: f64) -> Load {
    let all = 4.0 * (rows * width) as f64;
    Load {
        threads: rows,
        group: 256,
        trips: width as f64,
        trip,
        reads: vec![all, shared],
        written: all,
        flight: 64.0,
    }
}
