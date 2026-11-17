use crate::block::{NVFP4, Q4_0, Q8_0};
use crate::form::Form;

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
pub(crate) fn same_part(
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
}
