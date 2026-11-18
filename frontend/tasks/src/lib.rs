//! Text in, answers out: tasks adapt text to a model.

pub mod api;
pub mod embed;
#[cfg(test)]
mod embed_test;
pub mod features;
#[cfg(test)]
mod features_test;
pub mod flow;
#[cfg(test)]
mod flow_test;
pub mod generate;
pub mod paint;
pub mod png;
pub mod tag;
mod text;
#[cfg(test)]
mod text_test;
pub mod worker;

pub use embed::{Embedder, Threaded};
pub use generate::{Generate, Job, Queued};
pub use tag::Tagger;
pub(crate) use text::{Spool, Stops};
pub use worker::Worker;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
