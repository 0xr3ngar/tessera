//! Inner loops per instruction set, picked at run time.

use std::sync::OnceLock;

/// The widest instruction set this CPU runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Level {
    /// The target's own baseline.
    Base,
    /// x86 AVX2 with fused multiply-add.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// x86 AVX-512.
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl Level {
    /// Every level this CPU runs, the best last.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn all() -> Vec<Level> {
        let fma = is_x86_feature_detected!("fma");
        let mut out = vec![Level::Base];
        if fma && is_x86_feature_detected!("avx2") {
            out.push(Level::Avx2);
        }
        if fma && is_x86_feature_detected!("avx512f") {
            out.push(Level::Avx512);
        }
        out
    }

    /// Every level this CPU runs: its baseline.
    #[cfg(not(target_arch = "x86_64"))]
    pub(crate) fn all() -> Vec<Level> {
        vec![Level::Base]
    }

    /// The best level, found once.
    pub(crate) fn best() -> Level {
        static BEST: OnceLock<Level> = OnceLock::new();
        *BEST.get_or_init(|| *Level::all().last().expect("the baseline"))
    }
}

/// Fused only where the instruction set fuses in one step.
const FUSED: bool = cfg!(any(target_arch = "aarch64", target_feature = "fma"));

/// `a + x * y` lane by lane.
#[inline(always)]
fn madd<const L: usize, const F: bool>(a: [f32; L], x: [f32; L], y: [f32; L]) -> [f32; L] {
    std::array::from_fn(|l| match F {
        true => x[l].mul_add(y[l], a[l]),
        false => a[l] + x[l] * y[l],
    })
}

/// The sum of `a` times `b`, in `N` running sums.
#[inline(always)]
fn dot_in<const N: usize, const F: bool>(a: &[f32], b: &[f32]) -> f32 {
    let whole = a.len() / N * N;
    let (xs, _) = a[..whole].as_chunks::<N>();
    let (ys, _) = b[..whole].as_chunks::<N>();
    let mut acc = [0.0f32; N];
    for (x, y) in xs.iter().zip(ys) {
        acc = madd::<N, F>(acc, *x, *y);
    }
    let mut sum = acc.iter().sum::<f32>();
    for i in whole..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// `out * keep + p * b`. A zero `keep` drops `out`.
#[inline(always)]
fn mix_in<const F: bool>(out: &mut [f32], b: &[f32], keep: f32, p: f32) {
    if keep == 0.0 {
        out.iter_mut().zip(b).for_each(|(o, x)| *o = p * x);
        return;
    }
    for (o, x) in out.iter_mut().zip(b) {
        *o = match F {
            true => x.mul_add(p, *o * keep),
            false => *o * keep + p * x,
        };
    }
}

/// Rows of `x` by rows of `w`, in `L` lanes.
#[inline(always)]
fn tile_in<const R: usize, const C: usize, const L: usize, const F: bool>(
    x: [&[f32]; R],
    w: [&[f32]; C],
) -> [[f32; C]; R] {
    let len = w[0].len();
    let steps = len / L;
    let xs = x.map(|r| &r[..steps * L].as_chunks::<L>().0[..steps]);
    let ws = w.map(|r| &r[..steps * L].as_chunks::<L>().0[..steps]);
    // Each load feeds `C` or `R` products.
    let mut acc = [[[0.0f32; L]; C]; R];
    for i in 0..steps {
        let wv: [[f32; L]; C] = std::array::from_fn(|c| ws[c][i]);
        for r in 0..R {
            let xv = xs[r][i];
            for c in 0..C {
                acc[r][c] = madd::<L, F>(acc[r][c], xv, wv[c]);
            }
        }
    }
    let mut out = [[0.0f32; C]; R];
    for r in 0..R {
        for c in 0..C {
            out[r][c] = acc[r][c].iter().sum::<f32>();
            for i in steps * L..len {
                out[r][c] += x[r][i] * w[c][i];
            }
        }
    }
    out
}

/// Four columns as two passes of two, for 16 registers.
#[inline(always)]
fn halves<const R: usize, const L: usize, const F: bool>(
    x: [&[f32]; R],
    w: [&[f32]; 4],
) -> [[f32; 4]; R] {
    let a = tile_in::<R, 2, L, F>(x, [w[0], w[1]]);
    let b = tile_in::<R, 2, L, F>(x, [w[2], w[3]]);
    std::array::from_fn(|r| [a[r][0], a[r][1], b[r][0], b[r][1]])
}

/// One x86 vector width's loops, in its own intrinsics.
#[cfg(target_arch = "x86_64")]
macro_rules! level {
    (
        $name:ident, $feature:literal, $vec:ident, $lanes:literal, $cols:literal,
        $zero:ident, $load:ident, $fma:ident, $store:ident
    ) => {
        mod $name {
            use std::arch::x86_64::*;

            const L: usize = $lanes;

            #[target_feature(enable = $feature)]
            unsafe fn sum(v: $vec) -> f32 {
                let mut lanes = [0.0f32; L];
                // SAFETY: `lanes` holds one vector.
                unsafe { $store(lanes.as_mut_ptr(), v) };
                lanes.iter().sum()
            }

            /// Two running vectors, then the tail one at a time.
            #[target_feature(enable = $feature)]
            pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
                let n = a.len().min(b.len());
                let steps = n / (2 * L);
                let (mut s0, mut s1) = ($zero(), $zero());
                for i in 0..steps {
                    let o = i * 2 * L;
                    // SAFETY: `o + 2 L` is at most `n`.
                    unsafe {
                        let (pa, pb) = (a.as_ptr().add(o), b.as_ptr().add(o));
                        s0 = $fma($load(pa), $load(pb), s0);
                        s1 = $fma($load(pa.add(L)), $load(pb.add(L)), s1);
                    }
                }
                // SAFETY: the features are this function's own.
                let mut total = unsafe { sum(s0) + sum(s1) };
                for i in steps * 2 * L..n {
                    total += a[i] * b[i];
                }
                total
            }

            #[target_feature(enable = $feature)]
            pub unsafe fn mix(out: &mut [f32], b: &[f32], keep: f32, p: f32) {
                super::mix_in::<true>(out, b, keep, p)
            }

            /// `R` rows by `C` columns, every row one length.
            #[target_feature(enable = $feature)]
            unsafe fn cols<const R: usize, const C: usize>(
                x: [&[f32]; R],
                w: [&[f32]; C],
            ) -> [[f32; C]; R] {
                let len = w[0].len();
                let steps = len / L;
                let mut acc = [[$zero(); C]; R];
                let mut wv = [$zero(); C];
                for i in 0..steps {
                    let o = i * L;
                    // SAFETY: every row holds `len` floats, past `o + L`.
                    unsafe {
                        for c in 0..C {
                            wv[c] = $load(w[c].as_ptr().add(o));
                        }
                        for r in 0..R {
                            let xv = $load(x[r].as_ptr().add(o));
                            for c in 0..C {
                                acc[r][c] = $fma(xv, wv[c], acc[r][c]);
                            }
                        }
                    }
                }
                let mut out = [[0.0f32; C]; R];
                for r in 0..R {
                    for c in 0..C {
                        // SAFETY: the features are this function's own.
                        out[r][c] = unsafe { sum(acc[r][c]) };
                        for i in steps * L..len {
                            out[r][c] += x[r][i] * w[c][i];
                        }
                    }
                }
                out
            }

            #[target_feature(enable = $feature)]
            pub unsafe fn tile<const R: usize>(x: [&[f32]; R], w: [&[f32]; 4]) -> [[f32; 4]; R] {
                let len = w[0].len();
                assert!(
                    x.iter().chain(&w).all(|r| r.len() == len),
                    "rows of one length"
                );
                let mut out = [[0.0f32; 4]; R];
                for first in (0..4).step_by($cols) {
                    let mut part = [w[0]; $cols];
                    part.copy_from_slice(&w[first..first + $cols]);
                    // SAFETY: the features are this function's own.
                    let got = unsafe { cols::<R, $cols>(x, part) };
                    for r in 0..R {
                        out[r][first..first + $cols].copy_from_slice(&got[r]);
                    }
                }
                out
            }
        }
    };
}

/// All four columns at once, for 32 registers.
#[inline(always)]
fn whole<const R: usize, const L: usize, const F: bool>(
    x: [&[f32]; R],
    w: [&[f32]; 4],
) -> [[f32; 4]; R] {
    tile_in::<R, 4, L, F>(x, w)
}

// 16 AVX2 registers hold two columns, 32 hold four.
#[cfg(target_arch = "x86_64")]
level!(
    avx2,
    "avx2,fma",
    __m256,
    8,
    2,
    _mm256_setzero_ps,
    _mm256_loadu_ps,
    _mm256_fmadd_ps,
    _mm256_storeu_ps
);
#[cfg(target_arch = "x86_64")]
level!(
    avx512,
    "avx512f,fma",
    __m512,
    16,
    4,
    _mm512_setzero_ps,
    _mm512_loadu_ps,
    _mm512_fmadd_ps,
    _mm512_storeu_ps
);

/// The sum of `a` times `b` at `level`.
pub(crate) fn dot_at(level: Level, a: &[f32], b: &[f32]) -> f32 {
    match level {
        // SAFETY: `Level::all` found the features.
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 => unsafe { avx2::dot(a, b) },
        #[cfg(target_arch = "x86_64")]
        Level::Avx512 => unsafe { avx512::dot(a, b) },
        _ => dot_in::<16, FUSED>(a, b),
    }
}

/// `out * keep + p * b` into `out` at `level`.
pub(crate) fn mix_at(level: Level, out: &mut [f32], b: &[f32], keep: f32, p: f32) {
    match level {
        // SAFETY: `Level::all` found the features.
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 => unsafe { avx2::mix(out, b, keep, p) },
        #[cfg(target_arch = "x86_64")]
        Level::Avx512 => unsafe { avx512::mix(out, b, keep, p) },
        _ => mix_in::<FUSED>(out, b, keep, p),
    }
}

/// `R` rows of `x` by four rows of `w`.
pub(crate) fn tile_at<const R: usize>(
    level: Level,
    x: [&[f32]; R],
    w: [&[f32]; 4],
) -> [[f32; 4]; R] {
    match level {
        // SAFETY: `Level::all` found the features.
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 => unsafe { avx2::tile::<R>(x, w) },
        #[cfg(target_arch = "x86_64")]
        Level::Avx512 => unsafe { avx512::tile::<R>(x, w) },
        // NEON holds four columns. The x86 baseline holds two.
        _ if cfg!(target_arch = "aarch64") => whole::<R, 4, FUSED>(x, w),
        _ => halves::<R, 4, FUSED>(x, w),
    }
}
