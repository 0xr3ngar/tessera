//! Helpers for rules that build a source from a body.

use crate::{Shape, Source};
use tessera_codegen::hash;
use tessera_codegen::ir::Body;

/// Guessed seconds per unit of work, under lockstep trees.
pub const GUESS: f64 = 1e-12;

/// `body` under an entry name that hashes its text.
pub fn source(kind: &str, mut body: Body) -> Source {
    let hashed = hash(&body.text().body);
    body.entry = format!("{kind}_{hashed:x}");
    Source::Made(body)
}

/// Small groups: workers split a launch by group.
pub fn small(jobs: usize) -> Shape {
    Shape::Grid {
        across: jobs.div_ceil(8),
        down: 1,
        threads: 8,
    }
}
