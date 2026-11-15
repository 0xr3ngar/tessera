//! Decoders with grouped-query attention, RoPE and SwiGLU MLPs.

use crate::shared::{Attend, Side, Stack, step_ports, swiglu, tokens};
use trace::DimName::*;
use trace::{
    Composite, DType, Dim, DimExpr, DimName, Embed, Graph, Kept, Linear, Mask, Model, Norm, Pairs,
    Role, Size, Tensor, TensorType, Weights,
};

/// What a query or key head passes through before RoPE.
pub trait HeadNorm: Weights {
    fn apply(&self, x: Tensor, eps: f32) -> Tensor;
}

/// Qwen3 norms each head.
impl HeadNorm for Norm {
    fn apply(&self, x: Tensor, eps: f32) -> Tensor {
        x.rms_norm(self, eps)
    }
}

/// Llama does not.
impl HeadNorm for () {
    fn apply(&self, x: Tensor, _: f32) -> Tensor {
        x
    }
}

pub struct Config {
    pub stack: Stack,
    pub kv_heads: u64,
    pub rope_theta: f32,
    pub eps: f32,
}

impl Config {
    /// Qwen3-0.6B's sizes. Its heads are normed.
    pub fn qwen3(max_seqs: u64, max_context: u64) -> Self {
        Self {
            stack: Stack::sized([151_936, 1024, 28, 16, 128, 3072], max_seqs, max_context),
            kv_heads: 8,
            rope_theta: 1e6,
            eps: 1e-6,
        }
    }

    /// SmolLM2-360M's sizes. A Llama: its heads are not normed.
    pub fn smollm2(max_seqs: u64, max_context: u64) -> Self {
        Self {
            stack: Stack::sized([49_152, 960, 32, 15, 64, 2560], max_seqs, max_context),
            kv_heads: 5,
            rope_theta: 1e5,
            eps: 1e-5,
        }
    }

    /// A layout small enough to run in tests.
    pub fn toy(max_seqs: u64, max_context: u64) -> Self {
        Self {
            stack: Stack::sized([32, 16, 2, 4, 8, 32], max_seqs, max_context),
            kv_heads: 2,
            rope_theta: 1e4,
            eps: 1e-6,
        }
    }

    pub fn dims(&self) -> Vec<Dim> {
        self.stack.dims(&[(KvHeads, self.kv_heads)])
    }
}

#[derive(Weights)]
pub struct Layer<N: HeadNorm> {
    #[shape(Hidden)]
    pub ln1: Norm,
    #[shape(Hidden, Heads * HeadDim)]
    pub q: Linear,
    #[shape(Hidden, KvHeads * HeadDim)]
    pub k: Linear,
    #[shape(Hidden, KvHeads * HeadDim)]
    pub v: Linear,
    #[shape(HeadDim)]
    pub q_norm: N,
    #[shape(HeadDim)]
    pub k_norm: N,
    #[shape(Heads * HeadDim, Hidden)]
    pub o: Linear,
    #[shape(Hidden)]
    pub ln2: Norm,
    #[shape(Hidden, Ffn)]
    pub gate: Linear,
    #[shape(Hidden, Ffn)]
    pub up: Linear,
    #[shape(Ffn, Hidden)]
    pub down: Linear,
}

/// The embedding table is also the output head.
#[derive(Weights)]
pub struct Decoder<N: HeadNorm> {
    #[shape(Vocab, Hidden)]
    pub embed: Embed,
    #[shape(Layers)]
    pub layers: Vec<Layer<N>>,
    #[shape(Hidden)]
    pub norm: Norm,
}

/// One layer's kept keys and values.
type Cache = [Kept; 2];

/// What one pass over the layers shares.
struct Pass<'a> {
    m: &'a Model,
    c: &'a Config,
    positions: &'a Tensor,
    /// The axis this launch's rows lie on.
    rows: DimName,
    /// Each row's kept-row lane: the rows are ragged.
    slots: Option<&'a Tensor>,
}

pub fn forward<N: HeadNorm>(
    m: &Model,
    c: &Config,
    w: &Decoder<N>,
    tokens: &Tensor,
    positions: &Tensor,
) -> Tensor {
    let rows = Context;
    let pass = Pass {
        m,
        c,
        positions,
        rows,
        slots: None,
    };
    pass.logits(w, tokens, None)
}

impl Pass<'_> {
    /// With `kept`, each layer also reads earlier launches' rows.
    fn logits<N: HeadNorm>(
        &self,
        w: &Decoder<N>,
        tokens: &Tensor,
        kept: Option<&[Cache]>,
    ) -> Tensor {
        let (m, c) = (self.m, self.c);
        let mut x = m.embed(tokens, &w.embed);
        for (i, l) in w.layers.iter().enumerate() {
            let h = x.rms_norm(&l.ln1, c.eps);
            let attend = || [self.attention(l, &h, kept.map(|k| &k[i]))];
            let [attn] = m.composite(Composite::Attention, attend);
            x = x.add(&attn);
            let h = x.rms_norm(&l.ln2, c.eps);
            let [mlp] = m.composite(Composite::Mlp, || [swiglu(&h, [&l.gate, &l.up, &l.down])]);
            x = x.add(&mlp);
        }
        x.rms_norm(&w.norm, c.eps).unembed(&w.embed)
    }

    /// Queries and keys pass through `N` per head, then rotate.
    fn attention<N: HeadNorm>(&self, l: &Layer<N>, h: &Tensor, kept: Option<&Cache>) -> Tensor {
        let (c, rows) = (self.c, self.rows);
        let lead: Vec<DimExpr> = match self.slots {
            Some(_) => vec![rows.into()],
            None => vec![Seqs.into(), rows.into()],
        };
        let shape = [Heads.into(), KvHeads.into(), HeadDim.into()];
        let heads = Attend::new([&l.q, &l.k, &l.v, &l.o], lead, shape);
        let turn = |t: Tensor, side: Side| {
            let norm = match side {
                Side::Query => &l.q_norm,
                Side::Key => &l.k_norm,
                Side::Value => return t,
            };
            let normed = norm.apply(t, c.eps);
            normed.rope(self.positions, c.rope_theta, Pairs::Halves)
        };
        let scale = 1.0 / (c.stack.head_dim as f32).sqrt();
        heads.apply(h, turn, |[q, k, v]| match (kept, self.slots) {
            (Some([keys, values]), Some(slots)) => {
                let by = [slots, self.positions];
                let (k, v) = (k.append(keys, by), v.append(values, by));
                self.m.attend_rows(scale, [q, &k, &v], by)
            }
            _ => self.m.attend(Mask::Causal, scale, [q, k, v], None),
        })
    }
}

/// One forward entry: tokens and positions in, logits out.
pub fn program<N: HeadNorm>(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w: Decoder<N> = trace::declare(&m);
    let [tokens, positions] = tokens(&m);
    let logits = forward(&m, c, &w, &tokens, &positions);
    m.finish([(Role::Logits, &logits)])
}

/// One step entry: rows in, each row's likeliest token out.
pub fn stepper<N: HeadNorm>(c: &Config) -> Graph {
    let mut dims = c.dims();
    let size = Size::Bounded(c.stack.max_context);
    dims.push(Dim { name: Rows, size });
    let m = Model::new(dims);
    let w: Decoder<N> = trace::declare(&m);
    let [tokens, positions, slots] = step_ports(&m);
    let ty = TensorType::new([Seqs, Context, KvHeads, HeadDim], DType::F32);
    let cache = |_: &Layer<N>| [m.kept(ty.clone()), m.kept(ty.clone())];
    let kept: Vec<Cache> = w.layers.iter().map(cache).collect();
    let (positions, rows) = (&positions, Rows);
    let pass = Pass {
        m: &m,
        c,
        positions,
        rows,
        slots: Some(&slots),
    };
    let logits = pass.logits(&w, &tokens, Some(&kept));
    let ids = logits.top_k(1u64).reshape([Rows]);
    m.finish([(Role::Tokens, &ids), (Role::Logits, &logits)])
}
