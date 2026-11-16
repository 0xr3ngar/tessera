//! An EmbeddingGemma 2 exlr8 release: graph, weights, tokenizer.

use crate::names::repo;
use crate::pull;
use crate::tokenize::hf::Hf;
use checkpoint::exlr8::Release;
use engine::eg2::{Config, program};
use engine::{Graph, Linear, Weight};
use std::error::Error;
use std::path::Path;

pub struct Encoder {
    pub graph: Graph,
    pub weights: Vec<Weight>,
    pub tokenizer: Hf,
}

/// The text tower in `dir`: `seqs` texts of `context` tokens.
pub fn encoder(
    dir: &Path,
    seqs: u64,
    context: u64,
) -> Result<Encoder, Box<dyn Error + Send + Sync>> {
    let config = Config::release(seqs, context);
    let graph = program::<Linear>(&config);
    let rules = repo("eg2").unwrap_or_default();
    let release = Release::open(dir);
    let (weights, _) = pull(&graph, &release, config.stack.layers, &[&rules])?;
    let tokenizer = Hf::open(&dir.join("tokenizer.json"))?;
    Ok(Encoder {
        graph,
        weights,
        tokenizer,
    })
}
