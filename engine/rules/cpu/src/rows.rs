//! Row helpers the kernels share: activations and weight rows.

use graph::Act;
use lower::Form;

pub(crate) fn act(a: Act, x: f32) -> f32 {
    match a {
        Act::Silu => x / (1.0 + (-x).exp()),
        Act::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Act::Relu => x.max(0.0),
        Act::GeluTanh => {
            let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }
    }
}

/// Weight row `row` as floats. Odd forms unpack elementwise.
pub(crate) fn weight_row(form: &dyn Form, bytes: &[u8], row: usize, k: usize, out: &mut [f32]) {
    match form.name() {
        "f32" => {
            let at = row * k * 4;
            for (i, o) in out.iter_mut().enumerate() {
                let b = &bytes[at + 4 * i..at + 4 * i + 4];
                *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        "bf16" => {
            let at = row * k * 2;
            for (i, o) in out.iter_mut().enumerate() {
                let bits = u16::from_le_bytes([bytes[at + 2 * i], bytes[at + 2 * i + 1]]);
                *o = f32::from_bits(u32::from(bits) << 16);
            }
        }
        _ => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = form.unpack(bytes, row * k + i);
            }
        }
    }
}
