use crate::lockstep::{self, BLOCK, Dots, dots_form};
use codegen::form::{BF16_FORM, F32_FORM, Load};

#[test]
fn odd_length_dots() {
    let a: Vec<f32> = (0..19).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..19).map(|i| (i % 3) as f32).collect();
    let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    assert_eq!(lockstep::dot(&a, &b), want);
    let halves = BF16_FORM.pack(&b);
    assert_eq!(lockstep::dot_bf16(&a, &halves), want);
    let bytes: Vec<u8> = a.iter().flat_map(|x| x.to_le_bytes()).collect();
    assert_eq!(
        lockstep::dot_form(BF16_FORM, &bytes, 0, &halves, 0, 19),
        want
    );
}

/// Blocks of odd rows and depths against plain sums.
#[test]
fn dots_match_sums() {
    let (k, n) = (300, BLOCK);
    // Halves of small integers: exact in bf16 too.
    let value = |i: usize| (i % 13) as f32 * 0.5 - 3.0;
    let x: Vec<f32> = (0..BLOCK * k).map(value).collect();
    let w: Vec<f32> = (0..n * k).map(|i| value(i * 7)).collect();
    let wide: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let narrow: Vec<u8> = w
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let forms: [(&dyn Load, &[u8]); 2] = [(F32_FORM, &wide), (BF16_FORM, &narrow)];
    for (form, w_bytes) in forms {
        for (rows, cols) in [(1, n), (3, 13), (13, n), (BLOCK, 13)] {
            let mut out = vec![0u8; BLOCK * n * 4];
            let d = Dots {
                at: 0,
                stride: n,
                x_at: 0,
                w_at: 0,
                k,
                cols,
            };
            dots_form(form, &mut out, &xb, w_bytes, &d, rows);
            for (r, c) in (0..rows).flat_map(|r| (0..n).map(move |c| (r, c))) {
                // Columns past `cols` stay untouched.
                let want: f32 = match c < cols {
                    true => (0..k).map(|i| x[r * k + i] * w[c * k + i]).sum(),
                    false => 0.0,
                };
                let at = (r * n + c) * 4;
                let got = f32::from_le_bytes(out[at..at + 4].try_into().unwrap());
                assert!(
                    (got - want).abs() < 1e-3,
                    "{} rows {rows}: {got} {want}",
                    form.name()
                );
            }
        }
    }
}
