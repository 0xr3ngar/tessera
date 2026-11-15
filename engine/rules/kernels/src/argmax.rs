//! Each row's largest entries, best first.

use codegen::prelude::*;
use lower::sites;
use lower::{Draft, Kernel, Read, T, Tree};

/// The most threads that share one row's scan.
const PARTS: usize = 256;

/// The best index in each piece of each row.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(peaks, threads = self.rows * self.parts, work = self.rows * self.width)]
pub(crate) struct Peaks {
    pub rows: usize,
    pub width: usize,
    pub parts: usize,
}

/// The best of each row's peaks.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(argmax, threads = self.rows, work = self.rows * self.parts)]
pub(crate) struct Argmax {
    pub rows: usize,
    pub width: usize,
    pub parts: usize,
}

/// One thread per row picks `k` entries, best first.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(top_k, threads = self.rows, work = self.rows * self.width * self.k)]
pub(crate) struct TopK {
    pub rows: usize,
    pub width: usize,
    pub k: usize,
}

/// Each pick follows the last by value, then index.
#[kernel]
fn top_k(x: Buf<F32>, mut out: BufMut<U32>, rows: u32, width: u32, k: u32) {
    if gid() >= rows {
        return;
    }
    let at = gid() as u64 * width as u64;
    let mut last = f32::INFINITY;
    let mut last_i = 0u32;
    for j in 0..k {
        let mut arg = width;
        let mut best = f32::NEG_INFINITY;
        for i in 0..width {
            let v = x[at + i as u64];
            let later = v < last || (v == last && i > last_i);
            if later && v > best {
                best = v;
                arg = i;
            }
        }
        out[(gid() * k + j) as u64] = arg;
        last = best;
        last_i = arg;
    }
}

/// Strictly greater, so the lowest index wins ties.
#[kernel]
fn peaks(x: Buf<F32>, mut out: BufMut<U32>, rows: u32, width: u32, parts: u32) {
    if gid() >= rows * parts {
        return;
    }
    let span = (width + parts - 1) / parts;
    let from = (gid() % parts) * span;
    let to = if from + span < width {
        from + span
    } else {
        width
    };
    let at = (gid() / parts) as u64 * width as u64;
    let mut arg = width;
    let mut best = f32::NEG_INFINITY;
    for i in from..to {
        if x[at + i as u64] > best {
            best = x[at + i as u64];
            arg = i;
        }
    }
    out[gid() as u64] = arg;
}

/// A piece with nothing finite holds the width.
#[kernel]
fn argmax(x: Buf<F32>, peaks: Buf<U32>, mut out: BufMut<U32>, rows: u32, width: u32, parts: u32) {
    if gid() >= rows {
        return;
    }
    let at = gid() as u64 * width as u64;
    let mut arg = width;
    let mut best = f32::NEG_INFINITY;
    for p in 0..parts {
        let i = peaks[(gid() * parts + p) as u64];
        if i < width && x[at + i as u64] > best {
            best = x[at + i as u64];
            arg = i;
        }
    }
    out[gid() as u64] = arg;
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Top { x, rows, width, k } = tree.site(at)?;
    if k > 1 {
        return Some(vec![Draft::new(TopK { rows, width, k }, [x], rows * k)]);
    }
    let parts = PARTS.min(width).max(1);
    let both = [Read::Node(x), Read::Draft(0)];
    Some(vec![
        Draft::new(Peaks { rows, width, parts }, [x], rows * parts),
        Draft::new(Argmax { rows, width, parts }, both, rows),
    ])
}

lower::rule!("argmax", propose);
