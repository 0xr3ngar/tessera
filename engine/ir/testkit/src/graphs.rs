//! Small graphs that reach every rule, for tests and calibrate.

use graph::DimName::*;
use graph::{DType, Dim, DimExpr, DimName, Graph, Port, Role, TensorType};
use trace::{Act, Embed, EntryKind, Linear, Linears, Mask, Model, Norm, Pairs, Weights, declare};

#[derive(Weights)]
struct Layer {
    #[shape(Hidden, Ffn)]
    w: Linear,
}

#[derive(Weights)]
struct Scale {
    #[shape(Hidden)]
    norm: Norm,
}

#[derive(Weights)]
struct Table {
    #[shape(Vocab, Hidden)]
    embed: Embed,
}

#[derive(Weights)]
struct Experts {
    #[shape(Hidden, Experts)]
    router: Linear,
    /// Two matrices per expert, stacked by output row.
    #[shape(Experts, Hidden, Ffn * 2)]
    set: Linears,
}

/// An f32 input port.
pub fn floats<const N: usize>(shape: [DimName; N]) -> Port {
    let ty = TensorType::new(shape, DType::F32);
    let role = Role::Logits;
    Port { role, ty }
}

/// A u32 input port.
pub fn ids<const N: usize>(shape: [DimName; N]) -> Port {
    let ty = TensorType::new(shape, DType::U32);
    let role = Role::Tokens;
    Port { role, ty }
}

/// `bounded`, then these fixed sizes.
fn model(bounded: &[Dim], fixed: &[(DimName, u64)]) -> Model {
    let fixed = fixed.iter().map(|(name, n)| Dim::fixed(*name, *n));
    Model::new(bounded.iter().cloned().chain(fixed).collect())
}

/// Rows along every bounded axis, `width` wide.
fn rows_of(bounded: &[Dim], width: DimName, dtype: DType) -> Port {
    let lead = bounded.iter().map(|d| DimExpr::from(d.name));
    let shape: Vec<DimExpr> = lead.chain([width.into()]).collect();
    let role = Role::Logits;
    Port {
        role,
        ty: TensorType::new(shape, dtype),
    }
}

/// `k` inputs to `n` outputs: every matmul.
pub fn linear(bounded: &[Dim], k: u64, n: u64) -> Graph {
    let m = model(bounded, &[(Hidden, k), (Ffn, n)]);
    let w: Layer = declare(&m);
    let [x] = m.entry(EntryKind::Forward, [rows_of(bounded, Hidden, DType::F32)]);
    m.finish([(Role::Logits, &x.matmul(&w.w))])
}

/// Norms, maps, centering, a unit sum, softmax, argmax and mean.
pub fn rows(bounded: &[Dim], width: u64) -> Graph {
    let m = model(bounded, &[(Hidden, width)]);
    let w: Scale = declare(&m);
    let port = || rows_of(bounded, Hidden, DType::F32);
    let [x, y] = m.entry(EntryKind::Forward, [port(), port()]);
    let gated = x.rms_norm(&w.norm, 1e-6).act(Act::Silu).mul(&y.rms(1e-6));
    let sum = y.add(&gated).center();
    let ids = sum.act(Act::Sigmoid).unit_sum().softmax().top_k(1u64);
    m.finish([(Role::Tokens, &ids), (Role::Logits, &sum.mean(0))])
}

/// A rotation, then causal attention over `Seqs`, `Rows` and `Context`.
pub fn attention(bounded: &[Dim], [heads, kv_heads, width]: [u64; 3]) -> Graph {
    let fixed = [(Heads, heads), (KvHeads, kv_heads), (HeadDim, width)];
    let m = model(bounded, &fixed);
    let keys = || floats([Seqs, Context, KvHeads, HeadDim]);
    let ports = [
        floats([Seqs, Rows, Heads, HeadDim]),
        keys(),
        keys(),
        ids([Seqs, Rows]),
    ];
    let [q, k, v, at] = m.entry(EntryKind::Forward, ports);
    let q = q.rope(&at, 1e4, Pairs::Halves);
    let out = m.attend(Mask::Causal, 0.1, [&q, &k, &v], None);
    m.finish([(Role::Logits, &out)])
}

/// Rows of 256, rotated in two blocks of 128.
pub fn rotation(bounded: &[Dim]) -> Graph {
    let m = model(bounded, &[(Hidden, 256)]);
    let [x] = m.entry(EntryKind::Forward, [rows_of(bounded, Hidden, DType::F32)]);
    m.finish([(Role::Logits, &x.hadamard(128))])
}

/// Each row's top two experts, picked, applied and mixed.
pub fn routed(bounded: &[Dim], [experts, width, out]: [u64; 3]) -> Graph {
    let fixed = [(Experts, experts), (TopK, 2), (Hidden, width), (Ffn, out)];
    let m = model(bounded, &fixed);
    let w: Experts = declare(&m);
    let [x] = m.entry(EntryKind::Forward, [rows_of(bounded, Hidden, DType::F32)]);
    let scores = x.matmul(&w.router);
    let ids = scores.top_k(TopK);
    let gates = scores.take(&ids).softmax();
    let on = |set: &Linears| x.matmul_rows(&m.pick(set, &ids));
    let picks = on(&w.set.rows(0u64, Ffn))
        .act(Act::Silu)
        .mul(&on(&w.set.rows(Ffn, Ffn)));
    let axis = bounded.len() as u32;
    m.finish([(Role::Logits, &picks.mix(axis, &gates))])
}

/// Each row's top two entries weight the second input.
pub fn picks(bounded: &[Dim], width: u64) -> Graph {
    let m = model(bounded, &[(TopK, 2), (Hidden, width)]);
    let lead = bounded.iter().map(|d| DimExpr::from(d.name));
    let shape: Vec<DimExpr> = lead.chain([TopK.into(), Hidden.into()]).collect();
    let stacked = Port {
        role: Role::Logits,
        ty: TensorType::new(shape, DType::F32),
    };
    let [x, y] = m.entry(
        EntryKind::Forward,
        [rows_of(bounded, Hidden, DType::F32), stacked],
    );
    let gates = x.take(&x.top_k(TopK)).softmax();
    m.finish([(Role::Logits, &y.mix(bounded.len() as u32, &gates))])
}

/// Rows read from a table.
pub fn table(bounded: &[Dim], vocab: u64, width: u64) -> Graph {
    let m = model(bounded, &[(Vocab, vocab), (Hidden, width)]);
    let w: Table = declare(&m);
    let shape: Vec<DimExpr> = bounded.iter().map(|d| d.name.into()).collect();
    let tokens = Port {
        role: Role::Tokens,
        ty: TensorType::new(shape, DType::U32),
    };
    let [tokens] = m.entry(EntryKind::Forward, [tokens]);
    m.finish([(Role::Logits, &m.embed(&tokens, &w.embed))])
}
