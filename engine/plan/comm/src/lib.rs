//! Moves bytes between ranks over handles the front end opened.

mod fabric;
pub mod reduce;
mod shm;
mod sockets;
mod verbs;

pub use fabric::{Fabric, Port};
pub use reduce::{Pinned, Pinning, Reduce};
pub use shm::Shm;
pub use sockets::Sockets;
pub use verbs::Verbs;

use std::fmt;
use std::io;

/// Moves bytes between ranks. One round trip per call.
pub trait Transport {
    fn rank(&self) -> u32;
    fn ranks(&self) -> u32;
    /// Every rank's bytes, in rank order, `mine` among them.
    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, Error>;
    /// A device's way to run the rounds itself, if any.
    fn device(&mut self) -> Option<Reduce> {
        None
    }
}

impl<T: Transport + ?Sized> Transport for Box<T> {
    fn rank(&self) -> u32 {
        (**self).rank()
    }

    fn ranks(&self) -> u32 {
        (**self).ranks()
    }

    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, Error> {
        (**self).exchange(mine)
    }

    fn device(&mut self) -> Option<Reduce> {
        (**self).device()
    }
}

/// The parts summed in rank order. Every rank agrees.
pub fn summed<'a>(parts: impl Iterator<Item = &'a [f32]>) -> Vec<f32> {
    let mut sum: Option<Vec<f32>> = None;
    for part in parts {
        sum = Some(match sum {
            None => part.to_vec(),
            Some(s) => s.iter().zip(part).map(|(a, b)| a + b).collect(),
        });
    }
    sum.expect("a rank")
}

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    /// A handle, slot or round the ranks do not fit.
    Fit(String),
    /// The RDMA devices or a peer failed.
    Link(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Fit(what) => write!(f, "fit: {what}"),
            Error::Link(what) => write!(f, "link: {what}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

#[cfg(test)]
mod comm_test;
#[cfg(test)]
mod fabric_test;
