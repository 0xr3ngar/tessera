//! Whole-sequence attention, a thread per query and head.

use crate::{GUESS, source};
use codegen::prelude::*;
use graph::Mask;
use lower::{Attn, Draft, Kernel, Launch, Shape, Step, T, Tree, ZoneKind, sites};

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

/// Whole sequences with fixed keys. Ragged rows go to trees.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Attend = tree.site(at)?;
    if site.rows.is_some() || site.sizes.keys.runtime() || site.sizes.width > WIDEST {
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
