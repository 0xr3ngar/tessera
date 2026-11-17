use crate::simd::{Level, dot_at, mix_at, tile_at};

/// Every level this CPU runs, against plain sums.
#[test]
fn levels_match_sums() {
    let len = 300;
    let row = |seed: usize| -> Vec<f32> {
        let value = |i: usize| ((i * seed + 3) % 13) as f32 * 0.5 - 3.0;
        (0..len).map(value).collect()
    };
    let rows: Vec<Vec<f32>> = (1..8).map(row).collect();
    let sum = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    for level in Level::all() {
        assert_eq!(dot_at(level, &rows[0], &rows[1]), sum(&rows[0], &rows[1]));
        let mut out = rows[2].clone();
        mix_at(level, &mut out, &rows[3], 0.5, 2.0);
        let want: Vec<f32> = (0..len)
            .map(|i| rows[2][i] * 0.5 + 2.0 * rows[3][i])
            .collect();
        assert_eq!(out, want, "{level:?} mix");
        let w = [&rows[3][..], &rows[4], &rows[5], &rows[6]];
        let got = tile_at::<3>(level, [&rows[0][..], &rows[1], &rows[2]], w);
        for (r, line) in got.iter().enumerate() {
            for (c, v) in line.iter().enumerate() {
                assert_eq!(*v, sum(&rows[r], w[c]), "{level:?} tile {r} {c}");
            }
        }
    }
}
