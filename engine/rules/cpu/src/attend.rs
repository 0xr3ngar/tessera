//! Whole-sequence attention, a thread per query and head.

use crate::{GUESS, source};
use codegen::kept;
use codegen::prelude::*;
use graph::Mask;
use lower::{Arg, Attn, Draft, Kernel, Launch, Shape, Step, T, Tree, ZoneKind, sites};

/// The widest head a thread's sums hold.
const WIDEST: usize = 256;

/// Scores, softmax and weighted values in one pass per query.
#[kernel]
fn cpu_attend(
    q: Buf<F32>,
    k: Buf<F32>,
    v: Buf<F32>,
    mut out: BufMut<F32>,
    scale: f32,
    jobs: u32,
    #[build] queries: u32,
    #[build] keys: u32,
    #[build] heads: u32,
    #[build] kv_heads: u32,
    #[build] width: u32,
    #[build] before: u32,
    #[build] after: u32,
) {
    if gid() >= jobs {
        return;
    }
    let t = gid() % queries;
    let sh = gid() / queries;
    let s = sh / heads;
    let h = sh % heads;
    let qo = ((s * queries + t) * heads + h) as u64 * width as u64;
    // Query and key ends align, as in decode.
    let position = keys - queries + t;
    let from = if position > before {
        position - before
    } else {
        0
    };
    let reach = position + 1 + after;
    let to = if reach < keys { reach } else { keys };
    let kh = h / (heads / kv_heads);
    let mut acc = [0.0f32; WIDEST];
    for d in 0..width {
        acc[d as usize] = 0.0;
    }
    let mut most = -1.0e30f32;
    let mut total = 0.0f32;
    for j in from..to {
        let ko = ((s * keys + j) * kv_heads + kh) as u64 * width as u64;
        let score = q.dot(qo, k, ko, width) * scale;
        let top = max(most, score);
        let keep = exp(most - top);
        let p = exp(score - top);
        total = total * keep + p;
        for d in 0..width {
            acc[d as usize] = acc[d as usize] * keep + p * v[ko + d as u64];
        }
        most = top;
    }
    for d in 0..width {
        out[qo + d as u64] = acc[d as usize] / total;
    }
}

/// Softmax attention over `[seqs, rows, heads, width]` tensors.
#[derive(Clone, Debug, Kernel)]
#[launch(
    crate = lower,
    name = "cpu_attend",
    sizes = self.sizes.sizes(),
    work = self.sizes.seqs * self.sizes.heads * self.sizes.queries * self.sizes.keys.most() * self.sizes.width,
    guess = GUESS,
    runs_on = ZoneKind::Cpu,
    source = source("cpu_attend", self.body()),
    launches = attend_launches,
)]
pub struct Attend {
    pub scale: f32,
    pub sizes: Attn,
}

impl Attend {
    /// Keys before and after a query's position that it sees.
    fn window(&self) -> (u32, u32) {
        let keys = self.sizes.keys.most() as u32;
        match self.sizes.mask {
            Mask::Causal => (keys, 0),
            Mask::Bidirectional => (keys, keys),
            Mask::Within(n) => (n as u32, n as u32),
        }
    }

    fn jobs(&self) -> usize {
        self.sizes.seqs * self.sizes.heads * self.sizes.queries
    }

    fn body(&self) -> codegen::ir::Body {
        let a = &self.sizes;
        let (before, after) = self.window();
        cpu_attend_ir(
            a.queries as u32,
            a.keys.most() as u32,
            a.heads as u32,
            a.kv_heads as u32,
            a.width as u32,
            before,
            after,
        )
    }
}

fn attend_launches(k: &Attend, step: &Step) -> Vec<Launch> {
    let mut launches = vec![Launch::plain(step, &[k.jobs()], Shape::Line(k.jobs()))];
    // The scale sits before the counts.
    let at = launches[0].args.len() - 1;
    launches[0].args.insert(at, lower::Arg::value(&k.scale));
    launches
}

/// One launch over every row, then the kept layout sizes.
#[kernel]
fn cpu_attend_rows(
    q: Buf<F32>,
    k: Buf<F32>,
    v: Buf<F32>,
    mut out: BufMut<F32>,
    slots: Buf<U32>,
    positions: Buf<U32>,
    rows: u32,
    keys: u32,
    heads: u32,
    kv_heads: u32,
    width: u32,
) {
    if gid() >= rows * heads {
        return;
    }
    let row = gid() / heads;
    let qh = gid() % heads;
    let slot = slots[row as u64];
    let position = positions[row as u64];
    let kvh = qh / (heads / kv_heads);
    let q_off = ((row * heads + qh) * width) as u64;
    let mut acc = [0.0f32; WIDEST];
    for d in 0..width {
        acc[d as usize] = 0.0;
    }
    let mut most = -1.0e30f32;
    let mut total = 0.0f32;
    // The row sees its slot up to its own position.
    let last = if position + 1 < keys {
        position + 1
    } else {
        keys
    };
    for t in 0..last {
        let kv_off = ((kept_row(slot, t) * kv_heads + kvh) * width) as u64;
        let score = q.dot(q_off, k, kv_off, width);
        let top = max(most, score);
        let keep = exp(most - top);
        let p = exp(score - top);
        total = total * keep + p;
        for d in 0..width {
            acc[d as usize] = acc[d as usize] * keep + p * v[kv_off + d as u64];
        }
        most = top;
    }
    for d in 0..width {
        out[q_off + d as u64] = acc[d as usize] / total;
    }
}

/// Scaled query rows over kept keys, by slot and position.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = lower,
    name = "cpu_attend_rows",
    sizes = self.0.sizes(),
    work = self.0.heads * self.0.queries * self.0.keys.most() * self.0.width,
    guess = GUESS,
    runs_on = ZoneKind::Cpu,
    source = source("cpu_attend_rows", kept::attach(cpu_attend_rows_ir(), self.0.kept)),
    launches = rows_launches,
)]
pub struct AttendRows(pub Attn);

/// One launch over every row, the kept layout's sizes last.
fn rows_launches(k: &AttendRows, step: &Step) -> Vec<Launch> {
    let a = &k.0;
    let mut args = a.rows_args(step);
    args.push(Arg::count(a.width));
    args.extend(a.kept_args());
    vec![Launch {
        args,
        shape: Shape::Line(a.queries * a.heads),
    }]
}

/// Whole sequences with fixed keys, or rows over kept keys.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Attend = tree.site(at)?;
    if site.sizes.width > WIDEST {
        return None;
    }
    if site.rows.is_some() {
        return site.ragged(AttendRows(site.sizes));
    }
    if site.sizes.keys.runtime() {
        return None;
    }
    let out = site.sizes.seqs * site.out();
    let kernel = Attend {
        scale: site.scale,
        sizes: site.sizes,
    };
    Some(vec![Draft::new(kernel, [site.q, site.k, site.v], out)])
}

lower::rule!("cpu_attend", propose);
