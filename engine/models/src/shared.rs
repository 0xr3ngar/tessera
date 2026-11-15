//! Pieces every token model's program uses.

use trace::DimName::*;
use trace::{
    Act, DType, Dim, DimExpr, DimName, EntryKind, Linear, Model, Pairs, PathSeg, Port, Role,
    Tensor, TensorType, Vector, Weights,
};

/// A matrix a row multiplies into, however it is stored.
pub trait Dense: Weights {
    fn apply(&self, x: &Tensor) -> Tensor;
}

impl Dense for Linear {
    fn apply(&self, x: &Tensor) -> Tensor {
        x.matmul(self)
    }
}

/// exlr8 rotates in blocks of this many channels.
const ROTATION: u32 = 128;

/// A matrix between two rotations.
pub struct Rotated {
    pub w: Linear,
    pub suh: Vector,
    pub svh: Vector,
}

impl Weights for Rotated {
    fn declare(m: &Model, path: &mut Vec<PathSeg>, shape: &[DimExpr]) -> Self {
        Self {
            w: part(m, path, "w", shape),
            suh: part(m, path, "suh", &shape[..1]),
            svh: part(m, path, "svh", &shape[1..]),
        }
    }
}

/// Field `name` under `path`, declared with `shape`.
fn part<T: Weights>(m: &Model, path: &mut Vec<PathSeg>, name: &str, shape: &[DimExpr]) -> T {
    path.push(PathSeg::Field(name.into()));
    let value = T::declare(m, path, shape);
    path.pop();
    value
}

impl Dense for Rotated {
    fn apply(&self, x: &Tensor) -> Tensor {
        let turned = x.mul_vector(&self.suh).hadamard(ROTATION).matmul(&self.w);
        turned.hadamard(ROTATION).mul_vector(&self.svh)
    }
}

/// Which projection a head transform is applied to.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Query,
    Key,
    Value,
}

impl Side {
    /// The query's or the key's entry. Values have none.
    pub fn of<T>(self, [q, k]: [T; 2]) -> Option<T> {
        match self {
            Side::Query => Some(q),
            Side::Key => Some(k),
            Side::Value => None,
        }
    }
}

/// One attention block's head shape and output projection.
pub struct Attend<'a, D: Dense> {
    o: &'a D,
    /// Axes before the heads: `[Seqs, Context]`, or `[Rows]`.
    lead: Vec<DimExpr>,
    heads: DimExpr,
    kv_heads: DimExpr,
    /// A value's width. Keys match unless split by hand.
    head_dim: DimExpr,
    /// Positions, theta and pairing, if queries and keys rotate.
    rope: Option<(&'a Tensor, f32, Pairs)>,
}

impl<'a, D: Dense> Attend<'a, D> {
    /// Projection `o` over `lead`, with `[heads, kv_heads, head_dim]`.
    pub fn new(o: &'a D, lead: Vec<DimExpr>, shape: [DimExpr; 3]) -> Self {
        let [heads, kv_heads, head_dim] = shape;
        Self {
            o,
            lead,
            heads,
            kv_heads,
            head_dim,
            rope: None,
        }
    }

    /// Queries and keys rotate after their `turn`.
    pub fn rotary(mut self, positions: &'a Tensor, theta: f32, pairs: Pairs) -> Self {
        self.rope = Some((positions, theta, pairs));
        self
    }

    /// `lead`, then `rest`.
    fn shape(&self, rest: impl IntoIterator<Item = DimExpr>) -> Vec<DimExpr> {
        self.lead.iter().cloned().chain(rest).collect()
    }

    /// `t` as `heads` of `width` each, over `lead`.
    pub fn split(&self, t: Tensor, heads: impl Into<DimExpr>, width: impl Into<DimExpr>) -> Tensor {
        t.reshape(self.shape([heads.into(), width.into()]))
    }

    /// `h` through `[q, k, v]`, each side `turn`ed, then rotated.
    pub fn heads(
        &self,
        [q, k, v]: [&D; 3],
        h: &Tensor,
        turn: impl Fn(Tensor, Side) -> Tensor,
    ) -> [Tensor; 3] {
        let side = |w: &D, heads: &DimExpr, side: Side| {
            let t = turn(
                self.split(w.apply(h), heads.clone(), self.head_dim.clone()),
                side,
            );
            match (self.rope, side) {
                (Some((positions, theta, pairs)), Side::Query | Side::Key) => {
                    t.rope(positions, theta, pairs)
                }
                _ => t,
            }
        };
        [
            side(q, &self.heads, Side::Query),
            side(k, &self.kv_heads, Side::Key),
            side(v, &self.kv_heads, Side::Value),
        ]
    }

    /// Heads joined back into rows, then projected.
    pub fn merge(&self, out: Tensor) -> Tensor {
        let width = self.heads.clone() * self.head_dim.clone();
        self.o.apply(&out.reshape(self.shape([width])))
    }

    /// `heads`, then `attend`, then `merge`.
    pub fn apply(
        &self,
        w: [&D; 3],
        h: &Tensor,
        turn: impl Fn(Tensor, Side) -> Tensor,
        attend: impl FnOnce([&Tensor; 3]) -> Tensor,
    ) -> Tensor {
        let [q, k, v] = self.heads(w, h, turn);
        self.merge(attend([&q, &k, &v]))
    }
}

/// Sizes every decoder stack here declares.
pub struct Stack {
    pub vocab: u64,
    pub hidden: u64,
    pub layers: u64,
    pub heads: u64,
    pub head_dim: u64,
    pub ffn: u64,
    /// Most sequences one launch may carry.
    pub max_seqs: u64,
    pub max_context: u64,
}

impl Stack {
    /// `[vocab, hidden, layers, heads, head_dim, ffn]`, then the launch bounds.
    pub fn sized(
        [vocab, hidden, layers, heads, head_dim, ffn]: [u64; 6],
        max_seqs: u64,
        max_context: u64,
    ) -> Self {
        Self {
            vocab,
            hidden,
            layers,
            heads,
            head_dim,
            ffn,
            max_seqs,
            max_context,
        }
    }

    /// These dimensions, followed by the model's `own`.
    pub(crate) fn dims(&self, own: &[(DimName, u64)]) -> Vec<Dim> {
        let bounded = Dim::bounded;
        let fixed = |&(name, n): &(DimName, u64)| Dim::fixed(name, n);
        let shared = [
            (Vocab, self.vocab),
            (Hidden, self.hidden),
            (Layers, self.layers),
            (Heads, self.heads),
            (HeadDim, self.head_dim),
            (Ffn, self.ffn),
        ];
        let mut dims = vec![
            bounded(Seqs, self.max_seqs),
            bounded(Context, self.max_context),
        ];
        dims.extend(shared.iter().chain(own).map(fixed));
        dims
    }
}

/// A forward entry's tokens and positions, `[Seqs, Context]` each.
pub(crate) fn tokens(m: &Model) -> [Tensor; 2] {
    ports(m, EntryKind::Forward, Context)
}

/// A new entry's tokens and positions, `[Seqs, rows]` each.
pub(crate) fn ports(m: &Model, kind: EntryKind, rows: DimName) -> [Tensor; 2] {
    let ty = TensorType::new([Seqs, rows], DType::U32);
    let port = |role| Port {
        role,
        ty: ty.clone(),
    };
    m.entry(kind, [port(Role::Tokens), port(Role::Positions)])
}

/// A step entry's rows: tokens, positions and slots, `[Rows]` each.
pub(crate) fn step_ports(m: &Model) -> [Tensor; 3] {
    let ty = TensorType::new([Rows], DType::U32);
    let port = |role| Port {
        role,
        ty: ty.clone(),
    };
    let ports = [port(Role::Tokens), port(Role::Positions), port(Role::Slots)];
    m.entry(EntryKind::Step, ports)
}

/// `[Seqs, Context]` followed by `rest`.
pub(crate) fn rows<const N: usize>(rest: [DimExpr; N]) -> Vec<DimExpr> {
    let lead = [Seqs.into(), Context.into()];
    lead.into_iter().chain(rest).collect()
}

pub(crate) fn swiglu(h: &Tensor, [gate, up, down]: [&Linear; 3]) -> Tensor {
    let inner = h.matmul(gate).act(Act::Silu).mul(&h.matmul(up));
    inner.matmul(down)
}
