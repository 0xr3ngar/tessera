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
