//! Text in, answers out: tasks adapt text to a model.

pub mod embed;
#[cfg(test)]
mod embed_test;
pub mod generate;
pub mod tag;
mod text;
#[cfg(test)]
mod text_test;

pub use embed::Embedder;
pub use generate::Generate;
pub use tag::Tagger;
pub use text::Spool;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
