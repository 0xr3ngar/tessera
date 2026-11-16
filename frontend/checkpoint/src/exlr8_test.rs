use crate::exlr8::Release;
use crate::{Offer, Stored};
use engine::{E4M3, F16};
use std::path::Path;

/// One tensor: name, safetensors dtype, shape, bytes.
type Tensor<'a> = (&'a str, &'a str, Vec<usize>, Vec<u8>);

/// `tensors` as one safetensors file at `path`.
fn write(path: &Path, tensors: &[Tensor]) {
    let (mut header, mut data): (Vec<String>, Vec<u8>) = (Vec::new(), Vec::new());
    for (name, dtype, shape, bytes) in tensors {
        let at = data.len();
        data.extend(bytes);
        let shape: Vec<String> = shape.iter().map(usize::to_string).collect();
        let end = data.len();
        header.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":[{}],"data_offsets":[{at},{end}]}}"#,
            shape.join(",")
        ));
    }
    let header = format!("{{{}}}", header.join(","));
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes.extend(data);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
}

/// A record: format, out, in, K, side, then zeros.
fn record(format: i64, out: usize, inp: usize, side: i64) -> Vec<u8> {
    let fields = [format, out as i64, inp as i64, 0, side, 0, 0, 0];
    fields.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// `n` bytes below 0x7F, the E4M3 NaN.
fn ramp(n: usize, step: usize) -> Vec<u8> {
    (0..n).map(|i| (i * step % 120) as u8).collect()
}

fn halves(values: &[f32]) -> Vec<u8> {
    let half = |v: &f32| F16::from(*v).0.to_le_bytes();
    values.iter().flat_map(half).collect()
}

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn e4m3(b: u8) -> f32 {
    f32::from(E4M3(b))
}

/// Each format decodes as the release's README says.
#[test]
fn glm_dense_formats() {
    let dir = testkit::scratch(&format!("{}-glm-exlr8", std::process::id()));
    let fp8 = ramp(2 * 128, 7);
    let (e8m0, block) = (vec![126u8, 128, 127, 129], [0.5f32]);
    let (q4, s4) = (ramp(2 * 32, 11), [0.5f32, 1.0, 2.0, 4.0]);
    let (q4_in, s4_in) = (
        ramp(2 * 64, 13),
        [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
    write(
        &dir.join("dense/exlr8-dense-000.safetensors"),
        &[
            ("mx.record", "I64", vec![8], record(3, 2, 64, 0)),
            ("mx.fp8", "U8", vec![2, 64], fp8[..128].to_vec()),
            ("mx.e8m0", "U8", vec![2, 2], e8m0.clone()),
            ("blk.record", "I64", vec![8], record(4, 2, 128, 0)),
            ("blk.fp8", "U8", vec![2, 128], fp8.clone()),
            ("blk.scale", "F32", vec![1, 1], floats(&block)),
            ("q.record", "I64", vec![8], record(2, 2, 64, 1)),
            ("q.q4", "U8", vec![2, 32], q4.clone()),
            ("q.s4", "F16", vec![2, 2], halves(&s4)),
            ("qi.record", "I64", vec![8], record(2, 2, 128, 2)),
            ("qi.q4", "U8", vec![1, 2, 64], q4_in.clone()),
            ("qi.s4", "F16", vec![1, 2, 4], halves(&s4_in)),
        ],
    );
    write(
        &dir.join("dense/exlr8-vocab.safetensors"),
        &[
            ("lm_head.record", "I64", vec![8], record(7, 2, 2, 0)),
            (
                "lm_head.fp32",
                "F32",
                vec![2, 2],
                floats(&[1.0, 2.0, 3.0, 4.0]),
            ),
            ("norm.weight", "F32", vec![2], floats(&[0.5, 0.25])),
        ],
    );
    let release = Release::open(&dir);
    let slices = release.slices();
    let shape = |name: &str| {
        let s = slices
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("{name}"));
        assert_eq!(s.stored, Stored::F32, "{name}");
        s.shape.clone()
    };
    assert_eq!(shape("model.layers.0.mx.weight"), [2, 64]);
    assert_eq!(shape("model.layers.0.qi.weight"), [2, 128]);
    assert_eq!(shape("lm_head.weight"), [2, 2]);
    assert_eq!(shape("model.norm.weight"), [2]);
    let read = |name: &str| release.floats(name).unwrap();
    let mx = read("model.layers.0.mx.weight");
    let blk = read("model.layers.0.blk.weight");
    let q = read("model.layers.0.q.weight");
    let qi = read("model.layers.0.qi.weight");
    for o in 0..2 {
        for i in 0..128 {
            let at = o * 128 + i;
            assert_eq!(blk[at], e4m3(fp8[at]) * 0.5, "block {o} {i}");
            let nibble = |b: u8| f32::from((b >> (4 * (i % 2))) & 0xF) - 8.0;
            let byte = q4_in[o * 64 + i / 2];
            assert_eq!(
                qi[at],
                nibble(byte) * s4_in[o * 4 + i / 32],
                "int4 by input {o} {i}"
            );
            if i >= 64 {
                continue;
            }
            let scale = 2f32.powi(i32::from(e8m0[o * 2 + i / 32]) - 127);
            assert_eq!(
                mx[o * 64 + i],
                e4m3(fp8[o * 64 + i]) * scale,
                "mxfp8 {o} {i}"
            );
            let byte = q4[o * 32 + i / 2];
            assert_eq!(
                q[o * 64 + i],
                nibble(byte) * s4[o * 2 + i / 32],
                "int4 {o} {i}"
            );
        }
    }
    assert_eq!(read("lm_head.weight"), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(read("model.norm.weight"), [0.5, 0.25]);
    std::fs::remove_dir_all(dir).unwrap();
}
