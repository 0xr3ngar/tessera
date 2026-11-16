//! CPU kernels for matmuls and attention: short bodies over `dot`.

mod attend;
mod dense;

pub use attend::Attend;
pub use dense::{Dense, Picked};

use codegen::hash;
use codegen::ir::Body;
use lower::Source;

/// Guessed seconds per unit of work, under lockstep trees.
const GUESS: f64 = 1e-12;

/// `body` under an entry name that hashes its text.
fn source(kind: &str, mut body: Body) -> Source {
    let hashed = hash(&body.text().body);
    body.entry = format!("{kind}_{hashed:x}");
    Source::Made(body)
}
