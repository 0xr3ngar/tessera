//! A matmul whose rows each pick a matrix.

use crate::product::{Product, Variant};
use codegen::ir::{Body, Expr, Intrinsic, Ty};
use codegen::prelude::*;
use lower::kernels::finish;
use lower::sites::Picked as Pick;
use lower::{Arg, Draft, Form as Stored, Kernel, Launch, Load, Shape, Step, T, Tree};

/// Each row's product with the matrix `ids` names for it.
#[derive(Clone, Debug, Kernel)]
#[launch(
    name = self.product.form.suffixed("picked"),
    sizes = self.product.sizes(),
    work = self.product.work(),
    accepts = self.product.accepts(input, form),
    source = self.product.source(self.tree(), "picked"),
    launches = picked_launches,
    load = picked_load,
    unproven,
)]
pub(crate) struct Picked {
    pub product: Product,
}

/// One thread per output element, along its row's matrix.
#[kernel]
fn picked(
    x: Buf<F32>,
    #[form(form)] w: Buf<Dyn>,
    ids: Buf<U32>,
    mut out: BufMut<F32>,
    rows: u32,
    n: u32,
    k: u32,
    count: u32,
    stride: u32,
    offset: u32,
    repeat: u32,
    rank: u32,
    per_rank: u32,
    #[build] form: &'static dyn Stored,
) {
    if gid() >= rows * n {
        return;
    }
    let row = gid() / n;
    let col = gid() % n;
    // This rank's matrices start at `rank * per_rank`.
    let id = ids[row as u64] - rank * per_rank;
    let mut sums = [0.0f32; 1];
    sums[0] = 0.0;
    if ids[row as u64] >= rank * per_rank && id < count {
        let at = (id as u64 * stride as u64 + offset as u64 + col as u64) * k as u64;
        let from = (row / repeat) as u64 * k as u64;
        for i in 0..k {
            sums[0] += x[from + i as u64] * w[at + i as u64];
        }
    }
}

impl Picked {
    /// The pick this product carries.
    fn pick(&self) -> Pick {
        self.product.picked.expect("a picked product")
    }

    /// The kernel with the chain's operands and its store added.
    fn tree(&self) -> Body {
        let form = self.product.form;
        let key: Variant = ("picked", 0, form.name(), false);
        let mut body = self.product.tree(key, || picked_ir(form));
        let gid = Expr::Call(Intrinsic::Gid, Vec::new());
        let at = Expr::cast(Ty::U64, gid);
        let sum = Expr::index(Expr::var("sums"), Expr::Int(0));
        let stmts = finish(&self.product.after, sum, at);
        body.stmts.extend(stmts);
        body
    }
}

/// One thread per output element loops over all of `k`.
fn picked_load(k: &Picked, _: usize) -> Option<Load> {
    let Product { m, n, k: depth, .. } = k.product;
    let form = k.product.form;
    Some(Load {
        threads: m * n,
        group: 256,
        trips: depth as f64,
        trip: k.tree().trip(),
        reads: vec![form.bytes(m * n * depth) as f64, (4 * m * depth) as f64],
        written: 4.0 * (m * n) as f64,
        flight: 64.0,
    })
}

fn picked_launches(k: &Picked, step: &Step) -> Vec<Launch> {
    let Product { m, n, k: depth, .. } = k.product;
    let pick = k.pick();
    let counts = [
        m,
        n,
        depth,
        pick.count,
        pick.stride,
        pick.offset,
        pick.repeat,
    ];
    let mut launches = k.product.launch(step, &counts, Shape::Line(m * n));
    let per_rank = if pick.routed { pick.count } else { 0 };
    launches[0].args.extend([Arg::Rank, Arg::count(per_rank)]);
    launches
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (product, reads) = Product::find(tree, at)?;
    product.picked?;
    let len = product.outputs();
    Some(vec![Draft::new(Picked { product }, reads, len)])
}

lower::rule!("picked", propose);
