//! GPU compute mechanism. Callers supply all kernel and cache policy.

mod error;
pub use error::Error;
pub use limits::Limits;
pub use platform::Platform;

pub mod backend;
pub mod ibv;
#[cfg(test)]
mod ibv_test;
pub mod limits;
pub mod platform;
#[cfg(test)]
mod platform_test;

pub mod testing;

#[cfg(target_os = "macos")]
pub mod metal;

#[cfg(any(feature = "cuda", feature = "rocm"))]
mod api;
pub mod device;
mod dl;
pub mod launch;
#[cfg(any(feature = "cuda", feature = "rocm"))]
mod rtc;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "rocm")]
pub mod rocm;
