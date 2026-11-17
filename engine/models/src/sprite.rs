//! A masked-diffusion model of 16x16 palette sprites.

use crate::shared::{Attend, Side};
use trace::DimName::*;
use trace::{
    Act, DType, Dim, DimExpr, EntryKind, Graph, Linear, Mask, Model, Norm, PathSeg, Port, Role,
    Shape, Tensor, TensorType, Vector, Weights,
};

/// Tokens before the image: registers, then text.
pub const REGISTERS: u64 = 4;
pub const TEXT: u64 = 24;
const PREFIX: u64 = REGISTERS + TEXT;
/// The 3x3 neighbours a depthwise convolution reads.
pub const TAPS: u64 = 9;
/// Rotary tables assume this head width.
const HEAD_DIM: u64 = 64;
/// Every norm's variance floor.
const EPS: f32 = 1e-6;

pub struct Config {
    pub grid: u64,
    pub palette: u64,
    pub hidden: u64,
    pub heads: u64,
    pub ffn: u64,
    pub text_dim: u64,
    /// Blocks before, inside and after the merged middle.
    pub depth: [u64; 3],
    pub max_seqs: u64,
}

impl Config {
    /// tessera-small, from thejagstudio.
    pub fn small(max_seqs: u64) -> Self {
        Self {
            grid: 16,
            palette: 512,
            hidden: 512,
            heads: 8,
            ffn: 1536,
            text_dim: 512,
            depth: [2, 8, 2],
            max_seqs,
        }
    }

    pub fn pixels(&self) -> u64 {
        self.grid * self.grid
    }

    pub fn dims(&self) -> Vec<Dim> {
        vec![
            Dim::bounded(Seqs, self.max_seqs),
            Dim::fixed(Vocab, self.palette),
            Dim::fixed(Hidden, self.hidden),
            Dim::fixed(Heads, self.heads),
            Dim::fixed(HeadDim, self.hidden / self.heads),
            Dim::fixed(Ffn, self.ffn),
            Dim::fixed(Embedding, self.text_dim),
        ]
    }
}

/// A weight used as a plain tensor, any rank.
pub struct Table(pub Tensor);

impl Weights for Table {
    fn declare(m: &Model, path: &mut Vec<PathSeg>, shape: &[DimExpr]) -> Self {
        Table(m.weight(path, TensorType::new(Shape(shape.to_vec()), DType::F32)))
    }
}

#[derive(Weights)]
pub struct Block {
    /// Shift, scale and gate twice, scales offset by one.
    #[shape(Hidden * 6)]
    pub modulation: Vector,
    #[shape(Hidden, Hidden)]
    pub q: Linear,
    #[shape(Hidden, Hidden)]
    pub k: Linear,
    #[shape(Hidden, Hidden)]
    pub v: Linear,
    #[shape(Hidden, Hidden)]
    pub o: Linear,
    #[shape(HeadDim)]
    pub q_norm: Norm,
    #[shape(HeadDim)]
    pub k_norm: Norm,
    #[shape(Hidden, Ffn)]
    pub up: Linear,
    #[shape(Hidden, Ffn)]
    pub gate: Linear,
    #[shape(Ffn, Hidden)]
    pub down: Linear,
    /// Depthwise weights per tap, times `TAPS` for the mean.
    #[shape(TAPS, Ffn)]
    pub taps: Table,
    #[shape(1, Ffn)]
    pub conv_bias: Table,
}

/// Fixed tables for one grid size.
#[derive(Weights)]
pub struct Level {
    /// `x * cos + swap(x) * sin`, swap exchanging halves.
    #[shape(S0, HeadDim)]
    pub cos: Table,
    #[shape(S0, HeadDim)]
    pub sin: Table,
    /// Each row's neighbours. Past the end reads zero.
    #[shape(S0, TAPS)]
    pub taps: Table,
    /// Zero on image rows: they read the bias.
    #[shape(S0)]
    pub image: Table,
}

#[derive(Weights)]
pub struct Sprite {
    #[shape(Vocab + 1, Hidden)]
    pub embed: Table,
    #[shape(S0, Hidden)]
    pub pos: Table,
    /// The last step's colors, for masked pixels.
    #[shape(Vocab, Hidden)]
    pub draft: Linear,
    #[shape(Embedding)]
    pub text_norm: Norm,
    #[shape(Embedding, Hidden)]
    pub text: Linear,
    #[shape(Hidden)]
    pub text_bias: Vector,
    #[shape(TEXT, Hidden)]
    pub null_text: Table,
    #[shape(Hidden)]
    pub null_cond: Table,
    #[shape(Embedding)]
    pub pool_norm: Norm,
    #[shape(Embedding, Hidden)]
    pub pool_in: Linear,
    #[shape(Hidden)]
    pub pool_in_bias: Vector,
    #[shape(Hidden, Hidden)]
    pub pool_out: Linear,
    #[shape(Hidden)]
    pub pool_out_bias: Vector,
    /// The mask-ratio MLP, one row per masked count.
    #[shape(S0 + 1, Hidden)]
    pub ratio: Table,
    #[shape(Hidden, Hidden * 6)]
    pub modulate: Linear,
    #[shape(Hidden * 6)]
    pub modulate_bias: Vector,
    #[shape(REGISTERS, Hidden)]
    pub registers: Table,
    #[shape(Hidden * 4)]
    pub merge_norm: Norm,
    #[shape(Hidden * 4, Hidden)]
    pub merge: Linear,
    #[shape(Hidden)]
    pub merge_bias: Vector,
    #[shape(Hidden, Hidden * 4)]
    pub split: Linear,
    #[shape(Hidden * 4)]
    pub split_bias: Vector,
    #[shape(Hidden, Hidden * 2)]
    pub final_mod: Linear,
    #[shape(Hidden * 2)]
    pub final_mod_bias: Vector,
    #[shape(Hidden, Vocab)]
    pub head: Linear,
    #[shape(Vocab)]
    pub head_bias: Vector,
}

/// Blocks at each level, and each level's tables.
pub struct Stages {
    pub blocks: [Vec<Block>; 3],
    pub pixel: Level,
    pub patch: Level,
}

fn named<W: Weights>(m: &Model, path: &[&str], index: Option<u32>, shape: &[DimExpr]) -> W {
    let mut p: Vec<PathSeg> = path.iter().map(|s| PathSeg::Field((*s).into())).collect();
    p.extend(index.map(PathSeg::Index));
    W::declare(m, &mut p, shape)
}

/// Declares every weight. Paths name the converted file's tensors.
pub fn declare(m: &Model, c: &Config) -> (Sprite, Stages) {
    let pixels = c.pixels();
    let w = named(m, &[], None, &[pixels.into()]);
    let stage = |name: &str, n: u64| {
        let each = |i| named::<Block>(m, &[name], Some(i as u32), &[]);
        (0..n).map(each).collect::<Vec<_>>()
    };
    let [a, b, d] = c.depth;
    let blocks = [stage("in", a), stage("core", b), stage("out", d)];
    let rows = |n: u64| [(PREFIX + n).into()];
    let pixel = named(m, &["pixel"], None, &rows(pixels));
    let patch = named(m, &["patch"], None, &rows(pixels / 4));
    (
        w,
        Stages {
            blocks,
            pixel,
            patch,
        },
    )
}

/// The graph's inputs, one row per canvas.
pub struct Inputs {
    /// Palette ids, `Vocab` where masked.
    pub canvas: Tensor,
    /// Masked pixels per canvas.
    pub masked: Tensor,
    pub text: Tensor,
    pub pooled: Tensor,
    /// `[1, 0]` keeps the text, `[0, 1]` takes the null.
    pub guide: Tensor,
    /// Last step's probabilities, zero on committed pixels.
    pub draft: Tensor,
}

fn port(role: Role, shape: Vec<DimExpr>, dtype: DType) -> Port {
    Port {
        role,
        ty: TensorType::new(shape, dtype),
    }
}

fn inputs(m: &Model, c: &Config) -> Inputs {
    let (s, pixels): (DimExpr, DimExpr) = (Seqs.into(), c.pixels().into());
    let ports = [
        port(Role::Tokens, vec![s.clone(), pixels.clone()], DType::U32),
        port(Role::Positions, vec![s.clone()], DType::U32),
        port(
            Role::Keys,
            vec![s.clone(), TEXT.into(), Embedding.into()],
            DType::F32,
        ),
        port(Role::Keys, vec![s.clone(), Embedding.into()], DType::F32),
        port(Role::Keys, vec![s.clone(), 2u64.into()], DType::F32),
        port(Role::Keys, vec![s, pixels, Vocab.into()], DType::F32),
    ];
    let [canvas, masked, text, pooled, guide, draft] = m.entry(EntryKind::Forward, ports);
    Inputs {
        canvas,
        masked,
        text,
        pooled,
        guide,
        draft,
    }
}

/// `[Seqs, n, ..]` from `a` then `b`, joined on axis 1.
fn join(a: &Tensor, b: &Tensor) -> Tensor {
    let swap = [0, 2, 1];
    a.permute(&swap).concat(&b.permute(&swap)).permute(&swap)
}

/// `t` where the guide keeps text, else `null`.
fn choose(t: &Tensor, null: &Tensor, guide: &Tensor, rest: &[DimExpr]) -> Tensor {
    let width = rest.iter().cloned().reduce(|a, b| a * b).expect("rank");
    let flat = |x: &Tensor, lead: DimExpr| x.reshape(vec![lead, width.clone()]);
    let null = flat(null, 1u64.into()).expand(0, Seqs);
    let both = flat(t, Seqs.into()).concat(&null);
    let mut shape = vec![Seqs.into(), 2u64.into()];
    shape.extend_from_slice(rest);
    both.reshape(shape).mix(1, guide)
}

/// `[Seqs, Hidden]` as `[Seqs, rows, Hidden]`.
fn spread(v: &Tensor, rows: u64) -> Tensor {
    v.reshape(vec![Seqs.into(), 1u64.into(), Hidden.into()])
        .expand(1, rows)
}

/// Text tokens and the global condition.
fn condition(w: &Sprite, x: &Inputs) -> (Tensor, Tensor) {
    let t = x.text.rms_norm(&w.text_norm, EPS).matmul(&w.text);
    let t = t.add_vector(&w.text_bias);
    let c = x.pooled.rms_norm(&w.pool_norm, EPS).matmul(&w.pool_in);
    let c = c.add_vector(&w.pool_in_bias).act(Act::Silu);
    let c = c.matmul(&w.pool_out).add_vector(&w.pool_out_bias);
    let t = choose(&t, &w.null_text.0, &x.guide, &[TEXT.into(), Hidden.into()]);
    let c = choose(&c, &w.null_cond.0, &x.guide, &[Hidden.into()]);
    (t, c)
}

/// One stage's sequence length and grid side.
#[derive(Clone, Copy)]
struct Grid {
    side: u64,
}

impl Grid {
    fn rows(self) -> u64 {
        PREFIX + self.side * self.side
    }
}

/// 2D rotary angles from the level's tables.
fn rotate(t: Tensor, lv: &Level, rows: u64) -> Tensor {
    let table = |x: &Table| {
        x.0.reshape(vec![rows.into(), 1u64.into(), HeadDim.into()])
            .expand(1, Heads)
    };
    let half = HEAD_DIM / 2;
    let swapped = t.slice(3, half, half).concat(&t.slice(3, 0u64, half));
    t.mul(&table(&lv.cos)).add(&swapped.mul(&table(&lv.sin)))
}

fn attention(m: &Model, b: &Block, lv: &Level, h: &Tensor, g: Grid) -> Tensor {
    let lead = vec![Seqs.into(), g.rows().into()];
    let heads = Attend::new(&b.o, lead, [Heads.into(), Heads.into(), HeadDim.into()]);
    let turn = |t: Tensor, side: Side| match side.of([&b.q_norm, &b.k_norm]) {
        Some(norm) => rotate(t.rms_norm(norm, EPS), lv, g.rows()),
        None => t,
    };
    let scale = 1.0 / ((m.size(HeadDim) as f32).sqrt());
    heads.apply([&b.q, &b.k, &b.v], h, turn, |qkv| {
        m.attend(Mask::Bidirectional, scale, qkv, None)
    })
}

/// Depthwise 3x3 over image rows. Prefix rows get zero.
fn conv(b: &Block, lv: &Level, x: &Tensor) -> Tensor {
    let ids = lv.taps.0.cast(DType::U32);
    let rows = x.permute(&[1, 0, 2]).gather(&ids).permute(&[2, 0, 1, 3]);
    let summed = rows.mul(&b.taps.0).mean(2);
    let bias = b.conv_bias.0.gather(&lv.image.0.cast(DType::U32));
    summed.add(&bias)
}

fn mlp(b: &Block, lv: &Level, h: &Tensor) -> Tensor {
    let gate = h.matmul(&b.gate);
    let gate = gate.add(&conv(b, lv, &gate));
    let inner = h.matmul(&b.up).mul(&gate.act(Act::Silu));
    inner.matmul(&b.down)
}

fn block(m: &Model, b: &Block, lv: &Level, x: Tensor, cond: &Tensor, g: Grid) -> Tensor {
    let rows = g.rows();
    let hidden = m.size(Hidden);
    let mods = cond.add_vector(&b.modulation);
    let part = |i: u64| spread(&mods.slice(1, i * hidden, hidden), rows);
    let h = x.rms(EPS).mul(&part(1)).add(&part(0));
    let x = x.add(&attention(m, b, lv, &h, g).mul(&part(2)));
    let h = x.rms(EPS).mul(&part(4)).add(&part(3));
    x.add(&mlp(b, lv, &h).mul(&part(5)))
}

/// `[Seqs, side^2, Hidden]` to 2x2 patches, `[Seqs, side^2/4, 4 Hidden]`.
fn patches(img: &Tensor, side: u64, width: DimExpr) -> Tensor {
    let half = side / 2;
    let shape: Vec<DimExpr> = vec![
        Seqs.into(),
        half.into(),
        2u64.into(),
        half.into(),
        2u64.into(),
        width,
    ];
    let cells = img.reshape(shape).permute(&[0, 1, 3, 2, 4, 5]);
    cells.reshape(vec![
        Seqs.into(),
        (half * half).into(),
        DimExpr::from(Hidden) * 4u64,
    ])
}

/// The inverse of `patches` on `[Seqs, side^2/4, 4 Hidden]`.
fn pixels(core: &Tensor, side: u64) -> Tensor {
    let half = side / 2;
    let shape: Vec<DimExpr> = vec![
        Seqs.into(),
        half.into(),
        half.into(),
        2u64.into(),
        2u64.into(),
        Hidden.into(),
    ];
    let cells = core.reshape(shape).permute(&[0, 1, 3, 2, 4, 5]);
    cells.reshape(vec![Seqs.into(), (side * side).into(), Hidden.into()])
}

/// Logits over the palette for every pixel, `[Seqs, pixels, Vocab]`.
pub fn forward(m: &Model, c: &Config, w: &Sprite, s: &Stages, x: &Inputs) -> Tensor {
    let (fine, coarse) = (Grid { side: c.grid }, Grid { side: c.grid / 2 });
    let px = c.pixels();
    let img = w.embed.0.gather(&x.canvas).add(&w.pos.0);
    let img = img.add(&x.draft.matmul(&w.draft));
    let (text, cond) = condition(w, x);
    let cond = cond.add(&w.ratio.0.gather(&x.masked));
    let mods = cond.act(Act::Silu).matmul(&w.modulate);
    let mods = mods.add_vector(&w.modulate_bias);
    let shape = vec![1u64.into(), REGISTERS.into(), Hidden.into()];
    let regs = w.registers.0.reshape(shape).expand(0, Seqs);
    let mut xs = join(&join(&regs, &text), &img);
    for b in &s.blocks[0] {
        xs = block(m, b, &s.pixel, xs, &mods, fine);
    }
    let (pre, skip) = (xs.slice(1, 0u64, PREFIX), xs.slice(1, PREFIX, px));
    let merged = patches(&skip, c.grid, Hidden.into()).rms_norm(&w.merge_norm, EPS);
    let merged = merged.matmul(&w.merge).add_vector(&w.merge_bias);
    xs = join(&pre, &merged);
    for b in &s.blocks[1] {
        xs = block(m, b, &s.patch, xs, &mods, coarse);
    }
    let (pre, core) = (xs.slice(1, 0u64, PREFIX), xs.slice(1, PREFIX, px / 4));
    let up = core.matmul(&w.split).add_vector(&w.split_bias);
    xs = join(&pre, &skip.add(&pixels(&up, c.grid)));
    for b in &s.blocks[2] {
        xs = block(m, b, &s.pixel, xs, &mods, fine);
    }
    let img = xs.slice(1, PREFIX, px);
    let fin = cond
        .act(Act::Silu)
        .matmul(&w.final_mod)
        .add_vector(&w.final_mod_bias);
    let hidden = m.size(Hidden);
    let (shift, scale) = (fin.slice(1, 0u64, hidden), fin.slice(1, hidden, hidden));
    let img = img
        .rms(EPS)
        .mul(&spread(&scale, px))
        .add(&spread(&shift, px));
    img.matmul(&w.head).add_vector(&w.head_bias)
}

/// One forward entry: canvas and condition in, logits out.
pub fn program(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let (w, s) = declare(&m, c);
    let x = inputs(&m, c);
    let logits = forward(&m, c, &w, &s, &x);
    m.finish([(Role::Logits, &logits)])
}
