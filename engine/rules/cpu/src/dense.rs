//! `x[m, k]` times `w[n, k]` transposed, dense or picked.

use crate::{GUESS, source};
use codegen::ir::{BinOp, Body, Expr, Intrinsic, Stmt, Ty};
use codegen::prelude::*;
use lower::form::Load as Stored;
use lower::kernels::finish;
use lower::sites::{After, Picked as Pick};
use lower::{Draft, Kernel, Launch, Shape, Step, T, Tree, ZoneKind, sites};

/// One thread per output: its row dotted with its column.
#[kernel]
fn cpu_dense(
    x: Buf<F32>,
    #[form(form)] w: Buf<Dyn>,
    out: BufMut<F32>,
    m: u32,
    n: u32,
    k: u32,
    #[build] form: &'static dyn Stored,
) {
    if gid() >= m * n {
        return;
    }
    let mut sums = [0.0f32; 1];
    sums[0] = 0.0;
    let row = (gid() / n) as u64 * k as u64;
    let col = (gid() % n) as u64 * k as u64;
    sums[0] += x.dot(row, w, col, k);
}

/// One thread per block of 16 rows by 16 columns.
#[kernel]
fn cpu_tiled(
    x: Buf<F32>,
    #[form(form)] w: Buf<Dyn>,
    mut out: BufMut<F32>,
    m: u32,
    n: u32,
    k: u32,
    #[build] form: &'static dyn Stored,
) {
    // Groups of four by four blocks stay in cache.
    let across = (n / 16 + 3) / 4;
    let group = gid() / 16;
    let row = (group / across * 4 + gid() % 16 / 4) * 16;
    let col = (group % across * 4 + gid() % 4) * 16;
    if row >= m || col >= n {
        return;
    }
    let rows = if m - row < 16 { m - row } else { 16 };
    let at = row as u64 * n as u64 + col as u64;
    out.dots(
        at,
        n,
        x,
        row as u64 * k as u64,
        w,
        col as u64 * k as u64,
        k,
        rows,
    );
}

/// Which matrix each row reads, and where this rank's start.
#[derive(Group)]
struct Picking {
    count: u32,
    stride: u32,
    offset: u32,
    repeat: u32,
    first: u32,
}

/// One thread per output, along the matrix its row picks.
#[kernel]
fn cpu_picked(
    x: Buf<F32>,
    #[form(form)] w: Buf<Dyn>,
    ids: Buf<U32>,
    out: BufMut<F32>,
    rows: u32,
    n: u32,
    k: u32,
    pick: Picking,
    #[build] form: &'static dyn Stored,
) {
    if gid() >= rows * n {
        return;
    }
    let row = gid() / n;
    let col = gid() % n;
    let id = ids[row as u64];
    let mut sums = [0.0f32; 1];
    sums[0] = 0.0;
    if id >= pick.first && id - pick.first < pick.count {
        let matrix = (id - pick.first) as u64 * pick.stride as u64 + pick.offset as u64;
        let line = (matrix + col as u64) * k as u64;
        sums[0] += x.dot((row / pick.repeat) as u64 * k as u64, w, line, k);
    }
}

/// The stages after the sum, into the thread's output.
fn finished(mut body: Body, after: &After, operands_at: usize) -> Body {
    lower::kernels::add_operands(&mut body, operands_at, after.others());
    let at = Expr::cast(Ty::U64, Expr::Call(Intrinsic::Gid, Vec::new()));
    let sum = Expr::index(Expr::var("sums"), Expr::Int(0));
    body.stmts.extend(finish(after, sum, at));
    body
}

/// The stages after each sum the thread's block holds.
fn tiled_finish(mut body: Body, after: &After) -> Body {
    if after.stages.is_empty() {
        return body;
    }
    lower::kernels::add_operands(&mut body, 2, after.others());
    let e = || Expr::var("e");
    let (r, c) = (
        Expr::bin(BinOp::Div, e(), Expr::lit(16)),
        Expr::bin(BinOp::Rem, e(), Expr::lit(16)),
    );
    let offset = Expr::bin(
        BinOp::Add,
        Expr::bin(BinOp::Mul, r.clone(), Expr::var("n")),
        c,
    );
    let at = Expr::bin(BinOp::Add, Expr::var("at"), Expr::cast(Ty::U64, offset));
    let sum = Expr::index(Expr::var("out"), at.clone());
    body.stmts.push(Stmt::For {
        var: "e".into(),
        from: Expr::lit(0),
        to: Expr::lit(256),
        step: Expr::lit(1),
        unroll: 0,
        body: vec![Stmt::If {
            cond: Expr::bin(BinOp::Lt, r, Expr::var("rows")),
            then: finish(after, sum, at),
            els: Vec::new(),
        }],
    });
    body
}

/// Blocks need whole columns and f32 or bf16 weights.
fn tiles(m: usize, n: usize, form: &dyn Stored) -> bool {
    m >= 4 && n.is_multiple_of(16) && ["f32", "bf16"].contains(&form.name())
}

/// Weights read as stored. Rows and operands come as f32.
fn accepts(input: usize, form: &dyn Stored) -> bool {
    input == 1 || !form.narrow()
}

/// A matmul and its finishing stages.
#[derive(Clone, Debug, Kernel)]
#[launch(
    crate = lower,
    name = "cpu_dense",
    sizes = [self.m, self.n, self.k],
    work = self.m * self.n * self.k,
    guess = GUESS,
    runs_on = ZoneKind::Cpu,
    accepts = accepts(input, form),
    source = source("cpu_dense", finished(cpu_dense_ir(self.form), &self.after, 2)),
    launches = dense_launches,
)]
pub struct Dense {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: &'static dyn Stored,
    pub after: After,
}

fn dense_launches(k: &Dense, step: &Step) -> Vec<Launch> {
    vec![Launch::plain(
        step,
        &[k.m, k.n, k.k],
        Shape::Line(k.m * k.n),
    )]
}

/// A matmul in square blocks, and its stages.
#[derive(Clone, Debug, Kernel)]
#[launch(
    crate = lower,
    name = "cpu_tiled",
    sizes = [self.0.m, self.0.n, self.0.k],
    work = self.0.m * self.0.n * self.0.k,
    guess = GUESS / 4.0,
    runs_on = ZoneKind::Cpu,
    accepts = accepts(input, form),
    source = source("cpu_tiled", tiled_finish(cpu_tiled_ir(self.0.form), &self.0.after)),
    launches = tiled_launches,
)]
pub struct Tiled(pub Dense);

fn tiled_launches(k: &Tiled, step: &Step) -> Vec<Launch> {
    let d = &k.0;
    // Small groups, as workers split a launch by group.
    let groups = d.m.div_ceil(64) * (d.n / 16).div_ceil(4) * 4;
    let shape = Shape::Grid {
        across: groups,
        down: 1,
        threads: 4,
    };
    vec![Launch::plain(step, &[d.m, d.n, d.k], shape)]
}

/// Each row against its id's matrix, then the stages.
#[derive(Clone, Debug, Kernel)]
#[launch(
    crate = lower,
    name = "cpu_picked",
    sizes = [self.m, self.n, self.k],
    work = self.m * self.n * self.k,
    guess = GUESS,
    runs_on = ZoneKind::Cpu,
    accepts = accepts(input, form),
    source = source("cpu_picked", finished(cpu_picked_ir(self.form), &self.after, 3)),
    launches = picked_launches,
)]
pub struct Picked {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: &'static dyn Stored,
    pub after: After,
    pub pick: Pick,
}

fn picked_launches(k: &Picked, step: &Step) -> Vec<Launch> {
    let p = k.pick;
    let counts = [
        k.m,
        k.n,
        k.k,
        p.count,
        p.stride,
        p.offset,
        p.repeat,
        p.first.unwrap_or(0),
    ];
    vec![Launch::plain(step, &counts, Shape::Line(k.m * k.n))]
}

/// Any finished matmul, dense or picked.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Finished = tree.site(at)?;
    let sites::Matmul {
        m,
        n,
        k,
        form,
        picked,
        ..
    } = site.product;
    let after = site.after.clone();
    let dense = Dense {
        m,
        n,
        k,
        form,
        after: after.clone(),
    };
    let kernel = match picked {
        None if tiles(m, n, form) => Draft::new(Tiled(dense), site.reads(), m * n),
        None => Draft::new(dense, site.reads(), m * n),
        Some(pick) => Draft::new(
            Picked {
                m,
                n,
                k,
                form,
                after,
                pick,
            },
            site.reads(),
            m * n,
        ),
    };
    Some(vec![kernel])
}

lower::rule!("cpu_dense", propose);
