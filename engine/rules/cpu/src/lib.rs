//! CPU kernels for matmuls and attention: short bodies over `dot`.

mod attend;
mod dense;
mod norm;

pub use attend::{Attend, AttendRows};
pub use dense::{Dense, Picked, Tiled};
pub use norm::RmsNorm;

use codegen::hash;
use codegen::ir::Body;
use lower::{Shape, Source};

/// Guessed seconds per unit of work, under lockstep trees.
const GUESS: f64 = 1e-12;

/// `body` under an entry name that hashes its text.
fn source(kind: &str, mut body: Body) -> Source {
    let hashed = hash(&body.text().body);
    body.entry = format!("{kind}_{hashed:x}");
    Source::Made(body)
}

/// Small groups: workers split a launch by group.
fn small(jobs: usize) -> Shape {
    Shape::Grid {
        across: jobs.div_ceil(8),
        down: 1,
        threads: 8,
    }
}
