// SPDX-License-Identifier: MIT
// Ported from spaCy. Copyright ExplosionAI GmbH, see LICENSE.md.

//! spaCy's part-of-speech tagger over a loaded model.

mod features;
#[cfg(test)]
mod features_test;
mod symbols;
#[cfg(test)]
mod tag_test;

use crate::Error;
use bundle::spacy::{MAX_CONTEXT, Pipeline};
use bundle::tokenize::spacy::{Spacy, Word};
use engine::Model;
use features::{Features, thinc_hash};
use interp::Tensor;
use lower::{Recipe, Settings};
use models::tagger::{KEYS, PAD};
use std::path::Path;

/// The fewest rows a launch carries.
const LEAST: usize = 8;

pub struct Tagger {
    model: Model,
    tokenizer: Spacy,
    features: Features,
    tags: Vec<String>,
    /// Per feature: hash seed, rows, and first joined row.
    seeds: Vec<u32>,
    rows: Vec<u64>,
    offsets: Vec<u64>,
}

impl Tagger {
    /// Loads the pipeline in `dir` on the device `settings` pick.
    pub fn open(dir: &Path, settings: &Settings) -> Result<Self, Error> {
        let p = Pipeline::open(dir)?;
        let mut model = Model::open(p.graph, &Recipe::default(), settings)?;
        model.load(&p.weights)?;
        Ok(Self {
            model,
            tokenizer: p.tokenizer,
            features: Features::new(p.lexeme),
            tags: p.tags,
            seeds: p.seeds,
            rows: p.rows,
            offsets: p.offsets,
        })
    }

    /// Each token of `text` with its tag.
    pub fn tag(&mut self, text: &str) -> Result<Vec<(Word, &str)>, Error> {
        let words = self.tokenizer.words(text)?;
        let n = words.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n as u64 > MAX_CONTEXT {
            return Err(format!("{n} tokens, past {MAX_CONTEXT}").into());
        }
        // Lengths round up, so a few plans serve every text.
        let rows = n.next_power_of_two().max(LEAST);
        let mut keys = vec![0; rows * self.seeds.len() * KEYS as usize];
        for (w, slot) in words
            .iter()
            .zip(keys.chunks_mut(self.seeds.len() * KEYS as usize))
        {
            let feats = self.features.of(&text[w.span.clone()], w);
            let mut at = 0;
            for (c, f) in feats.iter().enumerate() {
                for k in thinc_hash(*f, self.seeds[c]) {
                    slot[at] = (u64::from(k) % self.rows[c] + self.offsets[c]) as u32;
                    at += 1;
                }
            }
        }
        let pad = PAD as usize;
        let padded = rows + 2 * pad;
        // Outside the text, rows read zero: index `rows` or `padded`.
        let mut sources = Vec::with_capacity(padded);
        let mut around = Vec::with_capacity(3 * padded);
        for j in 0..padded {
            let inside = (pad..pad + n).contains(&j);
            sources.push(if inside {
                (j - pad) as u32
            } else {
                rows as u32
            });
            let prev = if j == 0 { padded } else { j - 1 };
            let next = if j + 1 == padded { padded } else { j + 1 };
            around.extend([prev as u32, j as u32, next as u32]);
        }
        let inputs = [
            Tensor::u32(vec![1, rows, keys.len() / rows], keys),
            Tensor::u32(vec![1, padded], sources),
            Tensor::u32(vec![1, padded, 3], around),
        ];
        let out = self.model.run(&inputs)?;
        let logits = out[0].floats()?;
        let width = self.tags.len();
        let mut tagged = Vec::with_capacity(n);
        for (w, row) in words.into_iter().zip(logits.chunks(width).skip(pad)) {
            let best = (0..width)
                .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                .unwrap_or(0);
            tagged.push((w, self.tags[best].as_str()));
        }
        Ok(tagged)
    }
}
