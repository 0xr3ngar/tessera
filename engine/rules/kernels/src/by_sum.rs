//! Rows less their mean, or over their sum, generated.

use crate::per_row;
use codegen::prelude::*;
use graph::Stage;
use lower::sites;
use lower::{Arg, Draft, Kernel, Launch, Shape, Source, Step, T, Tree};
use sim::Load;

/// One thread per row: the sum, then each element.
#[kernel]
fn by_sum<const DIVIDE: bool>(x: Buf<F32>, mut out: BufMut<F32>, rows: u32, width: u32) {
    if gid() >= rows {
        return;
    }
    let at = gid() as u64 * width as u64;
    let mut sum = 0.0f32;
    for i in 0..width {
        sum += x[at + i as u64];
    }
    let mean = sum / width as f32;
    for i in 0..width {
        if DIVIDE {
            out[at + i as u64] = x[at + i as u64] / sum;
        } else {
            out[at + i as u64] = x[at + i as u64] - mean;
        }
    }
}

/// One thread per row: centered, or scaled to unit sum.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = sum_name(self),
    source = sum_source,
    launches = sum_launches,
    load = sum_load,
)]
pub(crate) struct BySum {
    pub rows: usize,
    pub width: usize,
    pub divide: bool,
}

fn sum_name(k: &BySum) -> &'static str {
    match k.divide {
        true => "unit_sum",
        false => "center",
    }
}

fn sum_body(k: &BySum) -> codegen::ir::Body {
    match k.divide {
        true => by_sum_ir::<true>(),
        false => by_sum_ir::<false>(),
    }
}

fn sum_source(k: &BySum) -> Source {
    Source::named(sum_body(k), format!("plan_{}", k.name()))
}

fn sum_load(k: &BySum, _: usize) -> Option<Load> {
    Some(per_row(k.rows, k.width, sum_body(k).trip(), 0.0))
}

fn sum_launches(k: &BySum, step: &Step) -> Vec<Launch> {
    let sizes = [Arg::count(k.rows), Arg::count(k.width)];
    let buffers = [step.read(0), step.out()];
    vec![Launch {
        args: buffers.into_iter().chain(sizes).collect(),
        shape: Shape::Line(k.rows),
    }]
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Stage = tree.site(at)?;
    let divide = match site.stage {
        Stage::Center => false,
        Stage::UnitSum => true,
        _ => return None,
    };
    let kernel = BySum {
        rows: site.len / site.width,
        width: site.width,
        divide,
    };
    Some(vec![Draft::new(kernel, [site.x], site.len)])
}

lower::rule!("by_sum", propose);
