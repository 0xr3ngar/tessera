//! DeepSeek-style decoders: low-rank attention, MoE, indexed keys.

use crate::shared::{Attend, Stack, rows, step_ports, swiglu, tokens};
use trace::DimName::*;
use trace::{
    Act, Composite, DType, Dim, DimExpr, DimName, Embed, Graph, Kept, Linear, Linears, Mask, Model,
    Norm, Pairs, Role, Size, Tensor, TensorType, Vector, Weights,
};

pub struct Config {
    /// Its `ffn` is an expert's width. Values use `head_dim`.
    pub stack: Stack,
    /// Leading layers with a dense MLP.
    pub dense_layers: u64,
    /// A head's unrotated key width, and its rotated width.
    pub nope: u64,
    pub rope: u64,
    pub rope_theta: f32,
    pub q_latent: u64,
    pub latent: u64,
    pub dense_ffn: u64,
    pub experts: u64,
    pub top_k: u64,
    pub route_scale: f32,
    /// Some layers score keys first. None read them all.
    pub index: Option<Index>,
    pub eps: f32,
}

/// The indexer's shape, and which layers run one.
pub struct Index {
    pub heads: u64,
    pub dim: u64,
    pub keep: u64,
    /// Past the dense layers, every n-th layer indexes.
    pub every: u64,
}

impl Config {
    /// GLM-5.3 at its released sizes.
    pub fn glm53(max_seqs: u64, max_context: u64) -> Self {
        let sizes = [154_880, 6144, 78, 64, 256, 2048];
        let index = Index {
            heads: 32,
            dim: 128,
            keep: 2048,
            every: 4,
        };
        Self {
            stack: Stack::sized(sizes, max_seqs, max_context),
            dense_layers: 3,
            nope: 192,
            rope: 64,
            rope_theta: 8e6,
            q_latent: 2048,
            latent: 512,
            dense_ffn: 12_288,
            experts: 256,
            top_k: 8,
            route_scale: 2.5,
            index: Some(index),
            eps: 1e-5,
        }
    }

    /// DeepSeek-V3 at its released sizes. Experts are picked without groups.
    pub fn deepseek_v3(max_seqs: u64, max_context: u64) -> Self {
        let sizes = [129_280, 7168, 61, 128, 128, 2048];
        Self {
            stack: Stack::sized(sizes, max_seqs, max_context),
            dense_layers: 3,
            nope: 128,
            rope: 64,
            rope_theta: 1e4,
            q_latent: 1536,
            latent: 512,
            dense_ffn: 18_432,
            experts: 256,
            top_k: 8,
            route_scale: 2.5,
            index: None,
            eps: 1e-6,
        }
    }

    /// Whether layer `l` picks keys. Others reuse the last pick.
    pub fn indexes(&self, l: u64) -> bool {
        let Some(index) = &self.index else {
            return false;
        };
        let dense = self.dense_layers;
        l < dense || (l + 1 - dense).is_multiple_of(index.every)
    }

    pub fn dims(&self) -> Vec<Dim> {
        let layers = self.stack.layers;
        let indexing = (0..layers).filter(|&l| self.indexes(l)).count();
        let mut own = vec![
            (Nope, self.nope),
            (Rope, self.rope),
            (QLatent, self.q_latent),
            (Latent, self.latent),
            (DenseFfn, self.dense_ffn),
            (Experts, self.experts),
            (TopK, self.top_k),
            (DenseLayers, self.dense_layers),
            (MoeLayers, layers - self.dense_layers),
            (IndexLayers, indexing as u64),
        ];
        if let Some(i) = &self.index {
            own.extend([
                (IndexHeads, i.heads),
                (IndexDim, i.dim),
                (IndexKeep, i.keep),
            ]);
        }
        self.stack.dims(&own)
    }
}

#[derive(Weights)]
pub struct Attention {
    #[shape(Hidden)]
    pub norm: Norm,
    #[shape(Hidden, QLatent)]
    pub q_a: Linear,
    #[shape(QLatent)]
    pub q_norm: Norm,
    #[shape(QLatent, Heads * (Nope + Rope))]
    pub q_b: Linear,
    #[shape(Hidden, Latent + Rope)]
    pub kv_a: Linear,
    #[shape(Latent)]
    pub kv_norm: Norm,
    /// Per head: an unrotated key part, then a value.
    #[shape(Latent, Heads * (Nope + HeadDim))]
    pub kv_b: Linear,
    #[shape(Heads * HeadDim, Hidden)]
    pub o: Linear,
}

/// Scores every key cheaply, so attention reads only the best.
#[derive(Weights)]
pub struct Indexer {
    #[shape(QLatent, IndexHeads * IndexDim)]
    pub wq: Linear,
    #[shape(Hidden, IndexDim)]
    pub wk: Linear,
    #[shape(IndexDim)]
    pub k_norm: Norm,
    #[shape(IndexDim)]
    pub k_bias: Vector,
    #[shape(Hidden, IndexHeads)]
    pub weights: Linear,
}

#[derive(Weights)]
pub struct DenseMlp {
    #[shape(Hidden, DenseFfn)]
    pub gate: Linear,
    #[shape(Hidden, DenseFfn)]
    pub up: Linear,
    #[shape(DenseFfn, Hidden)]
    pub down: Linear,
}

/// The expert every row uses.
#[derive(Weights)]
pub struct SharedMlp {
    #[shape(Hidden, Ffn)]
    pub gate: Linear,
    #[shape(Hidden, Ffn)]
    pub up: Linear,
    #[shape(Ffn, Hidden)]
    pub down: Linear,
}

#[derive(Weights)]
pub struct Moe {
    #[shape(Hidden, Experts)]
    pub router: Linear,
    /// Added to scores for the pick only.
    #[shape(Experts)]
    pub router_bias: Vector,
    #[shape(Experts, Hidden, Ffn)]
    pub gate: Linears,
    #[shape(Experts, Hidden, Ffn)]
    pub up: Linears,
    #[shape(Experts, Ffn, Hidden)]
    pub down: Linears,
    pub shared: SharedMlp,
}

/// Layer `l` uses `dense[l]`, or `moe[l - DenseLayers]`.
#[derive(Weights)]
pub struct Mla {
    #[shape(Vocab, Hidden)]
    pub embed: Embed,
    #[shape(Layers)]
    pub attn: Vec<Attention>,
    /// One per indexing layer, in layer order.
    #[shape(IndexLayers)]
    pub indexers: Vec<Indexer>,
    #[shape(Layers, Hidden)]
    pub mlp_norm: Vec<Norm>,
    #[shape(DenseLayers)]
    pub dense: Vec<DenseMlp>,
    #[shape(MoeLayers)]
    pub moe: Vec<Moe>,
    #[shape(Hidden)]
    pub norm: Norm,
    #[shape(Hidden, Vocab)]
    pub head: Linear,
}

/// One layer's kept latent, rotary key and index keys.
struct Cache {
    latent: Kept,
    rope: Kept,
    index: Option<Kept>,
}

/// The keys a layer reads.
enum Keys<'a> {
    All,
    /// Keys an earlier layer picked.
    Given(Tensor),
    /// Keys this indexer picks.
    Pick(&'a Indexer),
}

/// What one pass over the layers shares.
struct Pass<'a> {
    m: &'a Model,
    c: &'a Config,
    positions: &'a Tensor,
    /// When stepping: each row's slot, and every layer's kept rows.
    step: Option<(&'a Tensor, Vec<Cache>)>,
}

pub fn forward(m: &Model, c: &Config, w: &Mla, tokens: &Tensor, positions: &Tensor) -> Tensor {
    let pass = Pass {
        m,
        c,
        positions,
        step: None,
    };
    pass.logits(w, tokens)
}

impl Pass<'_> {
    fn logits(&self, w: &Mla, tokens: &Tensor) -> Tensor {
        let (m, c) = (self.m, self.c);
        let mut x = m.embed(tokens, &w.embed);
        let mut indexers = w.indexers.iter();
        let mut keys: Option<Tensor> = None;
        for (l, a) in w.attn.iter().enumerate() {
            let h = x.rms_norm(&a.norm, c.eps);
            let reused = keys.clone().filter(|_| !c.indexes(l as u64));
            let read = match (reused, &c.index) {
                (Some(k), _) => Keys::Given(k),
                (None, None) => Keys::All,
                (None, Some(_)) => {
                    Keys::Pick(indexers.next().expect("an indexer per indexing layer"))
                }
            };
            let attn = match read {
                Keys::Pick(_) => {
                    let [out, k] = m.composite(Composite::Attention, || {
                        let (out, k) = self.mla(l, a, &h, read);
                        [out, k.expect("picked keys")]
                    });
                    keys = Some(k);
                    [out]
                }
                _ => m.composite(Composite::Attention, || [self.mla(l, a, &h, read).0]),
            };
            x = x.add(&attn[0]);
            let h = x.rms_norm(&w.mlp_norm[l], c.eps);
            let mlp = match l.checked_sub(w.dense.len()) {
                None => {
                    let d = &w.dense[l];
                    m.composite(Composite::Mlp, || [swiglu(&h, [&d.gate, &d.up, &d.down])])
                }
                Some(i) => m.composite(Composite::Moe, || [moe(m, c, &w.moe[i], &h)]),
            };
            x = x.add(&mlp[0]);
        }
        x.rms_norm(&w.norm, c.eps).matmul(&w.head)
    }

    /// The axes before a row's features: `[Seqs, Context]`, or `[Rows]`.
    fn lead(&self, rest: impl IntoIterator<Item = DimExpr>) -> Vec<DimExpr> {
        let lead = match self.step {
            Some(_) => vec![Rows.into()],
            None => rows([]),
        };
        lead.into_iter().chain(rest).collect()
    }

    /// The output, and the keys it picked, if any.
    fn mla(&self, l: usize, w: &Attention, h: &Tensor, read: Keys) -> (Tensor, Option<Tensor>) {
        let (m, c) = (self.m, self.c);
        let feature = self.lead([]).len() as u32;
        let shape = [Heads.into(), Heads.into(), HeadDim.into()];
        let heads = Attend::new(&w.o, self.lead([]), shape);
        let turn = |t: Tensor| t.rope(self.positions, c.rope_theta, Pairs::Adjacent);
        let qr = h.matmul(&w.q_a).rms_norm(&w.q_norm, c.eps);
        let keys = match read {
            Keys::All => None,
            Keys::Given(k) => Some(k),
            Keys::Pick(ix) => Some(self.pick_keys(l, ix, &qr, h)),
        };
        let q = heads.split(qr.matmul(&w.q_b), Heads, Nope + Rope);
        let head = feature + 1;
        let q = q
            .slice(head, 0u64, Nope)
            .concat(&turn(q.slice(head, Nope, Rope)));
        let kv = h.matmul(&w.kv_a);
        let latent = kv.slice(feature, 0u64, Latent).rms_norm(&w.kv_norm, c.eps);
        // Every head shares one rotary key part.
        let shared = turn(kv.slice(feature, Latent, Rope));
        let scale = 1.0 / ((c.nope + c.rope) as f32).sqrt();
        let out = match &self.step {
            None => {
                let [k, v] = keys_values(w, &latent, &shared);
                m.attend(Mask::Causal, scale, [&q, &k, &v], keys.as_ref())
            }
            Some((slots, caches)) => {
                let by = [*slots, self.positions];
                let latent = latent.append(&caches[l].latent, by);
                let [k, v] = keys_values(w, &latent, &shared.append(&caches[l].rope, by));
                m.attend_rows(scale, [&q, &k, &v], by, keys.as_ref())
            }
        };
        (heads.merge(out), keys)
    }

    /// Each query's best `IndexKeep` visible keys.
    fn pick_keys(&self, l: usize, w: &Indexer, qr: &Tensor, h: &Tensor) -> Tensor {
        let c = self.c;
        let q = qr
            .matmul(&w.wq)
            .reshape(self.lead([IndexHeads.into(), IndexDim.into()]));
        let k = h.matmul(&w.wk).center().rms_norm(&w.k_norm, c.eps);
        let k = k.add_vector(&w.k_bias);
        // A step reads its slot's kept keys.
        let k = match &self.step {
            None => k,
            Some((slots, caches)) => {
                let kept = caches[l].index.as_ref().expect("an index cache");
                k.append(kept, [*slots, self.positions]).gather(slots)
            }
        };
        // Per query: one score for each head and key.
        let hits = q.matmul_rows(&k).act(Act::Relu);
        let by_head = h
            .matmul(&w.weights)
            .reshape(self.lead([DimExpr::Const(1), IndexHeads.into()]));
        let rank = self.lead([]).len() as u32 + 2;
        let swap: Vec<u32> = (0..rank - 2).chain([rank - 1, rank - 2]).collect();
        let scores = hits.permute(&swap).matmul_rows(&by_head);
        let scores = scores.reshape(self.lead([Context.into()]));
        let seen = match self.step {
            Some(_) => scores.causal(self.positions),
            None => scores.mask(Mask::Causal),
        };
        seen.top_k(IndexKeep)
    }
}

/// Per-head keys and values from latents and rotary keys.
fn keys_values(w: &Attention, latent: &Tensor, shared: &Tensor) -> [Tensor; 2] {
    let per_head = latent
        .matmul(&w.kv_b)
        .reshape(rows([Heads.into(), Nope + HeadDim]));
    let one = rows([DimExpr::Const(1), Rope.into()]);
    let shared = shared.reshape(one).expand(2, Heads);
    let k = per_head.slice(3, 0u64, Nope).concat(&shared);
    [k, per_head.slice(3, Nope, HeadDim)]
}

/// Each row's top experts, weighted, plus the shared expert.
fn moe(m: &Model, c: &Config, w: &Moe, h: &Tensor) -> Tensor {
    let scores = h.matmul(&w.router).act(Act::Sigmoid);
    let ids = scores.add_vector(&w.router_bias).top_k(TopK);
    let gates = scores.take(&ids).unit_sum().scale(c.route_scale);
    // One output row per pick, each on its own expert.
    let on = |set: &Linears, x: &Tensor| x.matmul_rows(&m.pick(set, &ids));
    let inner = on(&w.gate, h).act(Act::Silu).mul(&on(&w.up, h));
    let picks = h.ty().rank() as u32 - 1;
    let routed = on(&w.down, &inner).mix(picks, &gates);
    let s = &w.shared;
    routed.add(&swiglu(h, [&s.gate, &s.up, &s.down]))
}

/// One forward entry: tokens and positions in, logits out.
pub fn program(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w: Mla = trace::declare(&m);
    let [tokens, positions] = tokens(&m);
    let logits = forward(&m, c, &w, &tokens, &positions);
    m.finish([(Role::Logits, &logits)])
}

/// One step entry: rows in, each row's likeliest token out.
pub fn stepper(c: &Config) -> Graph {
    // Kept rows span the whole context, as buffers hold them.
    let mut dims = c.dims();
    for d in dims.iter_mut().filter(|d| d.name == Context) {
        d.size = Size::Fixed(c.stack.max_context);
    }
    let size = Size::Bounded(c.stack.max_context);
    dims.push(Dim { name: Rows, size });
    let m = Model::new(dims);
    let w: Mla = trace::declare(&m);
    let [tokens, positions, slots] = step_ports(&m);
    let kept = |width: DimName| m.kept(TensorType::new(rows([width.into()]), DType::F32));
    let cache = |l| Cache {
        latent: kept(Latent),
        rope: kept(Rope),
        index: c.indexes(l).then(|| kept(IndexDim)),
    };
    let pass = Pass {
        m: &m,
        c,
        positions: &positions,
        step: Some((&slots, (0..c.stack.layers).map(cache).collect())),
    };
    let logits = pass.logits(&w, &tokens);
    let ids = logits.top_k(1u64).reshape([Rows]);
    m.finish([(Role::Tokens, &ids), (Role::Logits, &logits)])
}
