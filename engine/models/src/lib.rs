//! Model descriptions, traced into graphs.

pub mod decoder;
pub mod eg2;
pub mod mla;
pub mod nanogpt;
mod shared;
pub mod small;
pub mod tagger;

pub use shared::{Attend, Dense, FEW, Rotated, Side, Stack};
