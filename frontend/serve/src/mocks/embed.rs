//! Vectors for `/v1/embeddings`.

use sensai_tasks::Error;
use sensai_tasks::api::{Embed, Embedded};

/// One vector per text: its length in bytes.
pub(super) struct MockEmbed;

impl Embed for MockEmbed {
    fn embed(&self, texts: &[String]) -> Result<Embedded, Error> {
        Ok(Embedded {
            vectors: texts.iter().map(|t| vec![t.len() as f32]).collect(),
            tokens: texts.len(),
        })
    }
}
