//! Buffers in zones, and typed views onto them.

mod buffer;
mod layout;
mod tensor;
mod view;

pub use buffer::{Backing, Buffer, Host};
pub use graph::DType;
pub use layout::Layout;
pub use tensor::Tensor;
pub use view::{View, ViewMut, copy};

use std::fmt;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// A view and its use disagree on type or size.
    #[error("mismatch: {0}")]
    Mismatch(String),
    /// The backend refused a copy.
    #[error("device: {0}")]
    Device(String),
}

pub(crate) fn mismatch<T>(what: impl fmt::Display) -> Result<T, Error> {
    Err(Error::Mismatch(what.to_string()))
}

#[cfg(test)]
mod mem_test;
