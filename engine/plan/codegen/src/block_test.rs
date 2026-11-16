use crate::Dialect;
use crate::block::Q4_0;
use crate::ir::{Body, Param};

/// A kernel reading a block form carries its decode functions.
#[test]
fn block_headers_render() {
    let body = Body {
        entry: "read".into(),
        params: vec![Param::Buffer {
            name: "w".into(),
            form: &Q4_0,
            writes: false,
        }],
        stmts: Vec::new(),
        ret: None,
        layout: None,
        helpers: Vec::new(),
    };
    let text = body.text();
    assert_eq!(text.headers.len(), 2, "one element, then four");
    let metal = text.render(Dialect::Metal);
    assert!(metal.contains("f_f16("), "the scale is an f16 field");
    assert!(metal.contains("f_fn float form_q4_0(f_const(u8) p, u64 i)"));
    assert!(metal.contains("f_fn float4 form_q4_0_x4(f_const(u8) p, u64 i)"));
    assert!(
        metal.contains("f_field(w, "),
        "codes come out as bit fields"
    );
    assert!(metal.contains("device const u8 *w"));
}

#[test]
fn odd_length_dots() {
    let a: Vec<f32> = (0..19).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..19).map(|i| (i % 3) as f32).collect();
    let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    assert_eq!(crate::lockstep::dot(&a, &b), want);
    let halves = crate::form::BF16_FORM.pack(&b);
    assert_eq!(crate::lockstep::dot_bf16(&a, &halves), want);
    let bytes: Vec<u8> = a.iter().flat_map(|x| x.to_le_bytes()).collect();
    assert_eq!(
        crate::lockstep::dot_form(crate::form::BF16_FORM, &bytes, 0, &halves, 0, 19),
        want
    );
}
