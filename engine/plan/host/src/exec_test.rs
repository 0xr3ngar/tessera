use crate::exec::{self, ArgVal, Grid};
use tessera_codegen::prelude::*;
use tessera_codegen::{Dialect, Precision};

/// A row's sum of squares: `n` from `at`, in order.
#[kernel]
fn squares(x: Buf<F32>, at: u64, n: u32) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += x[at + i as u64] * x[at + i as u64];
    }
    sum
}

/// Scales `n` of `x` from `at` by `by`, into `out`.
#[kernel]
fn scale<const AHEAD: u32>(x: Buf<F32>, mut out: BufMut<F32>, at: u64, n: u32, by: f32) {
    for i in (0..n).unroll(AHEAD) {
        out[at + i as u64] = x[at + i as u64] * by;
    }
}

/// An RMS norm from the two kernels above.
#[kernel]
fn calls(x: Buf<F32>, mut out: BufMut<F32>, rows: u32, width: u32) {
    if gid() >= rows {
        return;
    }
    let at = gid() as u64 * width as u64;
    let inv = rsqrt(squares(x, at, width) / width as f32);
    scale::<2>(x, out, at, width, inv);
}

#[test]
fn kernels_call() {
    let body = calls_ir();
    let text = body.text().render(Dialect::Cuda);
    // Callee locals get the call's number. Passed names stay.
    assert!(!text.contains("(u64)at;") && !text.contains("(u32)width;"));
    assert!(text.contains("for (u32 i__1 = 0; i__1 < width; i__1 += 1) {"));
    assert!(text.contains("sum__1 += (x[(at + (u64)i__1)] * x[(at + (u64)i__1)]);"));
    assert!(text.contains("float inv = f_rsqrt((sum__1 / (float)width));"));
    assert!(text.contains("#pragma unroll 2"));
    assert!(text.contains("out[(at + (u64)i__2)] = (x[(at + (u64)i__2)] * inv);"));

    let rows = [[1.0f32, 2.0, 2.0, 4.0], [3.0, 0.0, 0.0, 4.0]];
    let mut x: Vec<u8> = rows
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let mut out = vec![0u8; x.len()];
    let buf = |at| ArgVal::Buf {
        at,
        skip: 0,
        form: tessera_codegen::form::F32_FORM,
    };
    let args = [buf(0), buf(1), ArgVal::U32(2), ArgVal::U32(4)];
    let grid = Grid {
        groups: 1,
        threads: 2,
    };
    exec::run(
        &body,
        &args,
        &mut [&mut x, &mut out],
        grid,
        LANES,
        Precision::EXACT,
    );
    let got: Vec<f32> = out
        .chunks(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    // Each row's mean square is 6.25: scale 0.4.
    let want = [0.4f32, 0.8, 0.8, 1.6, 1.2, 0.0, 0.0, 1.6];
    assert_eq!(got, want);
}

/// Sums `red` into its first slot, span halving each level.
#[kernel]
fn tree_sum<const N: u32, const LEVELS: u32>(red: &mut Shared<f32>) -> f32 {
    for level in 0..LEVELS {
        let span = N >> (level + 1);
        if lid() < span {
            red[lid()] += red[lid() + span];
        }
        barrier();
    }
    red[0]
}

/// Scales each row of `acc` by its entry in `alpha`.
#[kernel]
fn scale_rows(acc: &mut Tiles, alpha: &Shared<f32>) {
    acc.scale(|r| alpha[r]);
}

/// Adds `by` to both of a thread's running sums.
#[kernel]
fn bump(sums: &mut [f32; 2], by: f32) {
    for i in 0..2 {
        sums[i as usize] += by;
    }
}

/// Eight threads sum their elements through group memory.
#[kernel]
fn group_sum(x: Buf<F32>, mut out: BufMut<F32>) {
    let mut red = shared::<f32>(8);
    red[lid()] = x[gid() as u64];
    barrier();
    let total = tree_sum::<8, 3>(&mut red);
    let mut sums = [0.0f32; 2];
    bump(&mut sums, total);
    if lid() == 0 {
        out[(gid() / 8) as u64] = sums[1];
    }
    let mut o = tiles(16, 16);
    let mut alpha = shared::<f32>(16);
    alpha[lid()] = total;
    scale_rows(&mut o, &alpha);
}

#[test]
fn callers_lend_memory() {
    let body = group_sum_ir();
    let text = body.text().render(Dialect::Metal);
    // The callee's loop is renamed. The caller's memory is not.
    assert!(text.contains("for (u32 level__1 = 0; level__1 < 3; level__1 += 1) {"));
    assert!(text.contains("u32 span__1 = (8 >> (level__1 + 1));"));
    assert!(text.contains("red[lid] += red[(lid + span__1)];"));
    assert!(text.contains("float total = red[0];"));
    assert!(text.contains("sums[(u32)i__2] += total;"));
    assert!(text.contains("o[i][j][e] *= alpha[r];"));

    let mut x: Vec<u8> = (1..=8).flat_map(|v| (v as f32).to_le_bytes()).collect();
    let mut out = vec![0u8; 4];
    let buf = |at| ArgVal::Buf {
        at,
        skip: 0,
        form: tessera_codegen::form::F32_FORM,
    };
    let grid = Grid {
        groups: 1,
        threads: 8,
    };
    let args = [buf(0), buf(1)];
    exec::run(
        &body,
        &args,
        &mut [&mut x, &mut out],
        grid,
        LANES,
        Precision::EXACT,
    );
    assert_eq!(f32::from_le_bytes(out[..4].try_into().unwrap()), 36.0);
}

/// Memory, tiles, an array and a value lent together.
#[derive(Group)]
struct Lent {
    red: Shared<f32>,
    acc: Tiles,
    sums: [f32; 2],
    b: f32,
}

/// Adds `b` to the sums, then scales the tiles.
#[kernel]
fn take_lent(lent: &mut Lent) {
    bump(&mut lent.sums, lent.b);
    lent.acc.scale(|r| lent.red[r]);
}

/// A group with sums at zero and `by` to add.
#[kernel]
fn new_lent(by: f32) -> Lent {
    Lent {
        red: shared::<f32>(16),
        acc: tiles(16, 16),
        sums: [0.0f32; 2],
        b: by,
    }
}

/// Builds a group, then lends it twice.
#[kernel]
fn lend_group(mut out: BufMut<F32>) {
    let mut lent = new_lent(2.0);
    take_lent(&mut lent);
    take_lent(&mut lent);
    out[gid() as u64] = lent.sums[1];
}

#[test]
fn groups_pass_by_field() {
    let callee = take_lent_ir();
    let names: Vec<&str> = callee.params.iter().map(|p| p.name()).collect();
    assert_eq!(names, ["lent__red", "lent__acc", "lent__sums", "lent__b"]);
    let body = lend_group_ir();
    let text = body.text().body;
    // Each field has its own name in caller and callee.
    for line in [
        "float lent__b = by__1;",
        "] += lent__b;",
        "lent__acc[i][j][e] *= lent__red[r];",
    ] {
        assert!(text.contains(line), "`{line}` in\n{text}");
    }
    let mut out = vec![0u8; 4];
    let args = [ArgVal::Buf {
        at: 0,
        skip: 0,
        form: tessera_codegen::form::F32_FORM,
    }];
    let grid = Grid {
        groups: 1,
        threads: 1,
    };
    exec::run(&body, &args, &mut [&mut out], grid, LANES, Precision::EXACT);
    assert_eq!(f32::from_le_bytes(out[..4].try_into().unwrap()), 4.0);
}
