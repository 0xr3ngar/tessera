//! Plain hardware description. No driver is touched here.

pub mod limits;
pub mod platform;
#[cfg(test)]
mod platform_test;

pub use limits::Limits;
pub use platform::Platform;
