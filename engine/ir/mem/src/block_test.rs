use crate::block::{NVFP4, Q4_0, Q8_0};
use crate::form::Form;
use crate::trellis::{INPUT_SIDE, TRELLIS_K2, packed};

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

/// A part decodes as the same elements of the whole.
fn same_part(
    form: &dyn Form,
    bytes: &[u8],
    [outer, whole, inner]: [usize; 3],
    [first, count]: [usize; 2],
) {
    let part = form
        .share(bytes, [outer, whole, inner], [first, count])
        .unwrap();
    for o in 0..outer {
        for i in 0..count * inner {
            let at = (o * whole + first) * inner + i;
            let got = form.unpack(&part, o * count * inner + i);
            assert_eq!(got, form.unpack(bytes, at), "{} element {at}", form.name());
        }
    }
}

#[test]
fn shares_round_trip() {
    let data: Vec<f32> = (0..256)
        .map(|i| ((i * 37) % 23) as f32 * 0.01 - 0.1)
        .collect();
    for form in [&Q8_0 as &dyn Form, &Q4_0, &NVFP4] {
        let bytes = form.pack(&data);
        same_part(form, &bytes, [2, 4, 32], [1, 2]);
        same_part(form, &bytes, [4, 2, 32], [1, 1]);
        // Half a block is no part.
        assert!(form.share(&bytes, [16, 2, 8], [1, 1]).is_none());
    }
    // Trellis codes: noise, under each side's header.
    let mut state = 7u32;
    let mut noise = |n: usize| -> Vec<u8> {
        let byte = |_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 16) as u8
        };
        (0..n).map(byte).collect()
    };
    let outputs = packed(32, 256, 0, &noise(32 * 256 * 2 / 8));
    same_part(&TRELLIS_K2, &outputs, [1, 256, 32], [128, 128]);
    let inputs = packed(256, 128, INPUT_SIDE, &noise(128 * 256 * 2 / 8));
    same_part(&TRELLIS_K2, &inputs, [128, 256, 1], [128, 128]);
    assert!(TRELLIS_K2.share(&inputs, [1, 128, 256], [0, 64]).is_none());
    // A set: matrices behind one header, cut whole.
    let (a, b) = (noise(128 * 256 * 2 / 8), noise(128 * 256 * 2 / 8));
    let set = packed(256, 128, INPUT_SIDE, &[a, b.clone()].concat());
    let second = packed(256, 128, INPUT_SIDE, &b);
    for i in (0..128 * 256).step_by(97) {
        let got = TRELLIS_K2.unpack(&set, 128 * 256 + i);
        assert_eq!(got, TRELLIS_K2.unpack(&second, i), "element {i}");
    }
    same_part(&TRELLIS_K2, &set, [1, 2, 128 * 256], [1, 1]);
}
