//! Model descriptions, traced into graphs.

pub mod decoder;
pub mod eg2;
pub mod glm53;
pub mod nanogpt;
mod shared;
pub mod tagger;

pub use shared::{Attend, Dense, Rotated, Side, Stack};
