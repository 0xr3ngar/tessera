//! Pages of kept rows, shared between runs by refcount.

mod pool;

pub use pool::{Error, PageCopy, Pool, RunId, Tag};

#[cfg(test)]
mod pool_test;
