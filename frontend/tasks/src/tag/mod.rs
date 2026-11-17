// SPDX-License-Identifier: MIT
// Ported from spaCy. Copyright ExplosionAI GmbH, see LICENSE.md.

//! spaCy's part-of-speech tagger over a loaded model.

#[cfg(test)]
mod tag_test;

use crate::Error;
use bundle::spacy::{MAX_CONTEXT, Pipeline};
use bundle::tokenize::spacy::Word;
use engine::tagger::{PAD, input};
use engine::{Model, Recipe, Settings, Tensor};
use std::path::Path;

pub struct Tagger {
    model: Model,
    pipeline: Pipeline,
}

impl Tagger {
    /// Loads the pipeline in `dir` on the device `settings` pick.
    pub fn open(dir: &Path, settings: &Settings) -> Result<Self, Error> {
        let (pipeline, graph, weights) = Pipeline::open(dir)?;
        let mut model = Model::open(graph, &Recipe::default(), settings)?;
        model.load(&weights)?;
        Ok(Self { model, pipeline })
    }

    /// Each token of `text` with its tag.
    pub fn tag(&mut self, text: &str) -> Result<Vec<(Word, &str)>, Error> {
        let words = self.pipeline.tokenizer.words(text)?;
        let n = words.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n as u64 > MAX_CONTEXT {
            return Err(format!("{n} tokens, past {MAX_CONTEXT}").into());
        }
        let ids = self.pipeline.row_ids(text, &words);
        let i = input(&ids, n, ids.len() / n);
        let inputs = [
            Tensor::u32(vec![1, i.rows, i.width], i.keys),
            Tensor::u32(vec![1, i.sources.len()], i.sources),
            Tensor::u32(vec![1, i.around.len() / 3, 3], i.around),
        ];
        let out = self.model.run(&inputs)?;
        let logits = out[0].floats()?;
        let width = self.pipeline.tags.len();
        let pad = PAD as usize;
        let mut tagged = Vec::with_capacity(n);
        for (w, row) in words.into_iter().zip(logits.chunks(width).skip(pad)) {
            let best = (0..width)
                .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                .unwrap_or(0);
            tagged.push((w, self.pipeline.tags[best].as_str()));
        }
        Ok(tagged)
    }
}
