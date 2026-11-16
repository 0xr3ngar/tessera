use crate::Dialect;
use crate::block::{NVFP4, Q4_0, Q8_0};
use crate::form::Form;
use crate::ir::{Body, Param};

/// Each form keeps `data` within a share of its largest.
#[test]
fn blocks_round_trip() {
    let data: Vec<f32> = (0..64)
        .map(|i| ((i * 37) % 23) as f32 * 0.01 - 0.1)
        .collect();
    for (form, share) in [(&Q8_0 as &dyn Form, 0.005), (&Q4_0, 0.07), (&NVFP4, 0.17)] {
        let back = form.roundtrip(&data);
        let top = data.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        for (x, y) in data.iter().zip(&back) {
            assert!(
                (x - y).abs() <= top * share,
                "{}: {x} came back {y}",
                form.name()
            );
        }
        assert_eq!(form.pack(&data).len(), form.bytes(64));
    }
    assert_eq!(
        (Q8_0.bytes(64), Q4_0.bytes(64), NVFP4.bytes(64)),
        (68, 36, 40)
    );
    assert_eq!(
        (Q4_0.elements(36), NVFP4.offset(32), NVFP4.offset(0)),
        (64, 22, 0)
    );
}

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
