use crate::simd::{Block, Level, block_at, dot_at, mix_at, panel_at, reduce};

/// Every level this CPU runs, against plain sums.
#[test]
fn levels_match_sums() {
    let len = 300;
    let row = |seed: usize| -> Vec<f32> {
        let value = |i: usize| ((i * seed + 3) % 13) as f32 * 0.5 - 3.0;
        (0..len).map(value).collect()
    };
    let rows: Vec<Vec<f32>> = (1..24).map(row).collect();
    let sum = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    let (x, w): (Vec<&[f32]>, Vec<&[f32]>) = (
        rows[..3].iter().map(Vec::as_slice).collect(),
        rows[7..23].iter().map(Vec::as_slice).collect(),
    );
    for level in Level::all() {
        assert_eq!(dot_at(level, x[0], x[1]), sum(x[0], x[1]));
        let mut out = rows[2].clone();
        mix_at(level, &mut out, &rows[3], 0.5, 2.0);
        let want: Vec<f32> = (0..len).map(|i| x[2][i] * 0.5 + 2.0 * rows[3][i]).collect();
        assert_eq!(out, want, "{level:?} mix");
        let mut acc = vec![[0.0f32; 16]; 3];
        // Two spans, added as a long depth adds them.
        if !panel_at(level, &x, &w, &mut acc) {
            let mut sums: Block = [[[[[0.0; 16]; 4]; 4]; 4]; 4];
            for span in [0..96, 96..len] {
                let xs: Vec<&[f32]> = x.iter().map(|r| &r[span.clone()]).collect();
                let ws: Vec<&[f32]> = w.iter().map(|r| &r[span.clone()]).collect();
                block_at(&xs, 3, &ws, &mut sums);
            }
            for (r, line) in acc.iter_mut().enumerate() {
                for (cq, quad) in sums[0].iter().enumerate() {
                    line[4 * cq..4 * cq + 4].copy_from_slice(&reduce(quad)[r]);
                }
            }
        }
        for (r, line) in acc.iter().enumerate() {
            for (c, v) in line.iter().enumerate() {
                assert_eq!(*v, sum(x[r], w[c]), "{level:?} block {r} {c}");
            }
        }
    }
}
