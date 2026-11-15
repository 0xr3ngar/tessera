// SPDX-License-Identifier: MIT
// Ported from spaCy. Copyright ExplosionAI GmbH, see LICENSE.md.

//! A spaCy pipeline folder: tokenizer, tagger graph, weights and tables.

mod thinc;

use crate::tokenize::Error;
use crate::tokenize::spacy::Spacy;
use graph::{Graph, PathSeg};
use interp::{Tensor, bind};
use lower::Weight;
use models::tagger::{Config, PIECES, path, program};
use std::collections::HashMap;
use std::path::Path;
use thinc::{Array, Node};

/// Tokens past this need a longer graph.
pub const MAX_CONTEXT: u64 = 4096;

pub struct Pipeline {
    pub tokenizer: Spacy,
    pub graph: Graph,
    pub weights: Vec<Weight>,
    pub tags: Vec<String>,
    /// Per feature: hash seed, rows, and first joined row.
    pub seeds: Vec<u32>,
    pub rows: Vec<u64>,
    pub offsets: Vec<u64>,
    /// The `lexeme_norm` table, by hash of the lower case.
    pub lexeme: HashMap<u64, String>,
}

fn nodes(dir: &Path, part: &str) -> Result<Vec<Node>, Error> {
    thinc::read(&std::fs::read(dir.join(part).join("model"))?)
}

fn param<'a>(n: &'a Node, name: &str) -> Result<&'a Array, Error> {
    n.params
        .get(name)
        .ok_or_else(|| format!("{} without {name}", n.name).into())
}

/// Thinc's `[units, pieces, ..]` rows as `[pieces, units, ..]`.
fn by_piece(a: &Array) -> Vec<f32> {
    let (units, pieces) = (a.shape[0], a.shape[1]);
    let row = a.data.len() / (units * pieces);
    let mut out = Vec::with_capacity(a.data.len());
    for p in 0..pieces {
        for u in 0..units {
            let at = (u * pieces + p) * row;
            out.extend_from_slice(&a.data[at..at + row]);
        }
    }
    out
}

impl Pipeline {
    /// Reads `dir`: tok2vec, tagger, tokenizer and vocab.
    pub fn open(dir: &Path) -> Result<Self, Error> {
        let t2v = nodes(dir, "tok2vec")?;
        let mut tables: Vec<&Node> = t2v.iter().filter(|n| n.name == "hashembed").collect();
        tables.sort_by_key(|n| n.attrs.get("column").copied());
        let maxouts: Vec<&Node> = t2v.iter().filter(|n| n.name == "maxout").collect();
        let norms: Vec<&Node> = t2v.iter().filter(|n| n.name == "layernorm").collect();
        let tagger = nodes(dir, "tagger")?;
        let softmax = tagger
            .iter()
            .find(|n| n.name == "softmax")
            .ok_or("tagger without softmax")?;
        let cfg: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("tagger/cfg"))?)?;
        let tags: Vec<String> = serde_json::from_value(cfg["labels"].clone())?;

        let mut weights: Vec<(Vec<PathSeg>, Vec<f32>)> = Vec::new();
        let (mut seeds, mut rows, mut offsets, mut joined) = (vec![], vec![], vec![], vec![]);
        for t in &tables {
            let e = param(t, "E")?;
            seeds.push(*t.attrs.get("seed").ok_or("hashembed without seed")? as u32);
            offsets.push(rows.iter().sum());
            rows.push(e.shape[0] as u64);
            joined.extend_from_slice(&e.data);
        }
        weights.push((vec![PathSeg::Field("embed".into())], joined));
        for (i, (m, n)) in maxouts.iter().zip(&norms).enumerate() {
            let w = param(m, "W")?;
            if w.shape.get(1) != Some(&(PIECES as usize)) {
                return Err(format!("maxout with {:?} pieces", w.shape.get(1)).into());
            }
            let i = Some(i as u32);
            weights.push((path("maxout", i, "w"), by_piece(w)));
            weights.push((path("maxout", i, "b"), by_piece(param(m, "b")?)));
            weights.push((path("norm", i, "g"), param(n, "G")?.data.clone()));
            weights.push((path("norm", i, "b"), param(n, "b")?.data.clone()));
        }
        weights.push((path("tagger", None, "w"), param(softmax, "W")?.data.clone()));
        weights.push((path("tagger", None, "b"), param(softmax, "b")?.data.clone()));

        let config = Config {
            rows: rows.clone(),
            width: param(norms.first().ok_or("tok2vec without layers")?, "G")?
                .data
                .len() as u64,
            depth: maxouts.len() as u64 - 1,
            tags: tags.len() as u64,
            max_context: MAX_CONTEXT,
        };
        let graph = program(&config);
        let find = |p: &[PathSeg]| {
            let (_, v) = weights.iter().find(|(path, _)| path == p)?;
            let v = v.clone();
            Some(Tensor::f32(vec![v.len()], v))
        };
        let weights = bind(&graph, find)?
            .into_iter()
            .map(Weight::Floats)
            .collect();
        Ok(Self {
            tokenizer: Spacy::open(dir)?,
            graph,
            weights,
            tags,
            seeds,
            rows,
            offsets,
            lexeme: thinc::lexeme_norms(&std::fs::read(dir.join("vocab/lookups.bin"))?)?,
        })
    }
}
