//! The interpreter's errors. Its tensors come from `mem`.

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// The graph needs something this interpreter lacks.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// Values disagree with the graph's types.
    #[error("mismatch: {0}")]
    Mismatch(String),
}

/// A view that did not fit is a mismatch.
impl From<tessera_mem::Error> for Error {
    fn from(e: tessera_mem::Error) -> Self {
        Error::Mismatch(e.to_string())
    }
}
