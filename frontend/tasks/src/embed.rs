//! Texts to unit-length EmbeddingGemma 2 vectors.

use crate::Error;
use bundle::encoder::encoder;
use bundle::tokenize::hf::Hf;
use engine::{Model, Recipe, Settings, Tensor};
use std::collections::BTreeMap;
use std::path::Path;

/// Tokens per text, `<bos>` and `<eos>` included.
const MAX_TOKENS: usize = 1024;
/// The most texts of one length a launch carries.
const SEQS: usize = 16;

pub struct Embedder {
    model: Model,
    tokenizer: Hf,
}

/// Each row scaled to unit length.
fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

impl Embedder {
    /// Loads the exlr8 release in `dir` where `settings` say.
    pub fn open(dir: &Path, settings: &Settings) -> Result<Self, Error> {
        let e = encoder(dir, SEQS as u64, MAX_TOKENS as u64)?;
        let mut model = Model::open(e.graph, &Recipe::default(), settings)?;
        model.load(&e.weights)?;
        Ok(Self {
            model,
            tokenizer: e.tokenizer,
        })
    }

    /// The backend and hardware the model runs on.
    pub fn device(&self) -> String {
        self.model.device()
    }

    /// Ids of `text` between `<bos>` and `<eos>`, cut to fit.
    pub fn tokens(&self, text: &str) -> Result<Vec<u32>, Error> {
        let mut ids = self.tokenizer.encode_special(text)?;
        if ids.len() > MAX_TOKENS {
            let eos = ids[ids.len() - 1];
            ids.truncate(MAX_TOKENS - 1);
            ids.push(eos);
        }
        Ok(ids)
    }

    /// One unit vector per text, in order.
    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Error> {
        let mut ids = Vec::with_capacity(texts.len());
        for t in texts {
            ids.push(self.tokens(t)?);
        }
        self.embed_ids(&ids)
    }

    /// The same for ids. Texts of one length share launches.
    pub fn embed_ids(&mut self, ids: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, Error> {
        let mut by_len: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (at, t) in ids.iter().enumerate() {
            if t.is_empty() || t.len() > MAX_TOKENS {
                return Err(format!("text {at} has {} tokens", t.len()).into());
            }
            by_len.entry(t.len()).or_default().push(at);
        }
        let mut out = vec![Vec::new(); ids.len()];
        for (len, all) in by_len {
            for group in all.chunks(SEQS) {
                let tokens: Vec<u32> = group.iter().flat_map(|&i| ids[i].clone()).collect();
                let positions = (0..group.len()).flat_map(|_| 0..len as u32).collect();
                let shape = vec![group.len(), len];
                let inputs = [
                    Tensor::u32(shape.clone(), tokens),
                    Tensor::u32(shape, positions),
                ];
                let got = self.model.run(&inputs)?;
                let floats = got[0].floats()?;
                let width = floats.len() / group.len();
                for (&i, row) in group.iter().zip(floats.chunks(width)) {
                    out[i] = unit(row.to_vec());
                }
            }
        }
        Ok(out)
    }
}
