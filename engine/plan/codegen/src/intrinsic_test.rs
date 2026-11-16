use crate::form::{BF16_FORM, F32_FORM, Load};
use crate::ir::{Body, Expr, Intrinsic, Param, Stmt};
use crate::lockstep::{BLOCK, Dots, dots_form};
use crate::{Dialect, kept};

/// One call of `i`, with a buffer as every argument.
fn call(i: Intrinsic) -> Body {
    let args = vec![Expr::var("x"); i.arity()];
    let stmt = match i {
        Intrinsic::Sum32 => Stmt::Assign {
            target: Expr::var("x"),
            op: None,
            value: Expr::Call(i, args),
        },
        _ => Stmt::Expr(Expr::Call(i, args)),
    };
    Body {
        entry: format!("{i:?}"),
        layout: kept::named("dense"),
        params: vec![Param::Buffer {
            name: "x".into(),
            form: F32_FORM,
            writes: true,
        }],
        stmts: vec![stmt],
        ret: None,
        helpers: Vec::new(),
    }
}

/// Each `f_` name `text` calls.
fn device_names(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut word = String::new();
    for c in text.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            word.push(c);
            continue;
        }
        if c == '(' && word.starts_with("f_") {
            names.push(word.clone());
        }
        word.clear();
    }
    names
}

#[test]
fn every_intrinsic_spelled() {
    for i in Intrinsic::ALL.iter().copied() {
        let text = call(i).text();
        for dialect in [Dialect::Metal, Dialect::Cuda] {
            let rendered = text.render(dialect);
            for name in device_names(&text.body) {
                let define = format!("#define {name}(");
                assert!(
                    rendered.contains(&define),
                    "{i:?}: no `{name}` in {dialect:?}"
                );
            }
        }
    }
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
        for rows in [1, 3, 13, BLOCK] {
            let mut out = vec![0u8; BLOCK * n * 4];
            let d = Dots {
                at: 0,
                stride: n,
                x_at: 0,
                w_at: 0,
                k,
            };
            dots_form(form, &mut out, &xb, w_bytes, &d, rows);
            for (r, c) in (0..rows).flat_map(|r| (0..n).map(move |c| (r, c))) {
                let want: f32 = (0..k).map(|i| x[r * k + i] * w[c * k + i]).sum();
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
