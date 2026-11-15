//! Moves bytes between ranks: the transport trait and TCP sockets.

mod sockets;

pub use sockets::Sockets;

use std::fmt;
use std::io;

/// Moves bytes between ranks. One round trip per call.
pub trait Transport {
    fn rank(&self) -> u32;
    fn ranks(&self) -> u32;
    /// Every rank's bytes, in rank order, `mine` among them.
    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, Error>;
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
    /// A rank joined out of turn or twice.
    Join(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Join(what) => write!(f, "join: {what}"),
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
