//! A tiny DeepSeek-shaped release over two ranks on one box.

use super::{cluster, model};
use graph::DimName::{Context, Seqs};
use graph::{Graph, PathSeg, shard};
use interp::{Tensor, run};
use lower::form::BF16_FORM;
use lower::{Device, ZoneKind, lower};
use run::Loopback;
use run::testing::runs;
use std::path::Path;
use testkit::{Noise, cosine_f32, max_abs_diff, weights};

/// Rows of one sequence.
const ROWS: usize = 8;

/// Three layers: one dense, two routed over 32 experts.
const CONFIG: &str = r#"{
  "vocab_size": 64, "hidden_size": 64, "num_hidden_layers": 3,
  "num_attention_heads": 2, "q_lora_rank": 32, "kv_lora_rank": 32,
  "qk_nope_head_dim": 16, "qk_rope_head_dim": 16, "v_head_dim": 32,
  "intermediate_size": 64, "moe_intermediate_size": 32,
  "n_routed_experts": 32, "num_experts_per_tok": 2, "first_k_dense_replace": 1,
  "routed_scaling_factor": 2.5, "rope_theta": 10000.0, "rms_norm_eps": 1e-6
}"#;

/// A weight's path as the name rules spell it.
fn path(segs: &[PathSeg]) -> String {
    let seg = |s: &PathSeg| match s {
        PathSeg::Field(f) => f.clone(),
        PathSeg::Index(i) => i.to_string(),
    };
    segs.iter().map(seg).collect::<Vec<_>>().join(".")
}

/// `weights` as HF slices in one bf16 file, experts apart.
fn write(dir: &Path, g: &Graph, weights: &[Tensor]) {
    let c = bundle::mla::config(CONFIG, 1, ROWS as u64).unwrap();
    let rules = bundle::mla::rules(&c);
    let (mut header, mut data) = (Vec::new(), Vec::new());
    let mut slice = |name: String, shape: &[usize], values: &[f32]| {
        let at = data.len();
        data.extend(BF16_FORM.pack(values));
        let shape: Vec<String> = shape.iter().map(usize::to_string).collect();
        let (shape, end) = (shape.join(","), data.len());
        header.push(format!(
            r#""{name}":{{"dtype":"BF16","shape":[{shape}],"data_offsets":[{at},{end}]}}"#
        ));
    };
    for (w, t) in g.weights.iter().zip(weights) {
        let at = path(&w.path);
        let (_, stem) = rules
            .iter()
            .find(|(p, _)| *p == at)
            .expect("a rule per weight");
        let name = |stem: &str| match stem.ends_with("bias") {
            true => stem.to_string(),
            false => format!("{stem}.weight"),
        };
        let values = t.floats().unwrap();
        if !stem.contains("{}") {
            slice(name(stem), &t.shape, values);
            continue;
        }
        let one = values.len() / t.shape[0];
        for (e, part) in values.chunks(one).enumerate() {
            slice(
                name(&stem.replace("{}", &e.to_string())),
                &t.shape[1..],
                part,
            );
        }
    }
    let header = format!("{{{}}}", header.join(","));
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes.extend(data);
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
    std::fs::write(dir.join("config.json"), CONFIG).unwrap();
}

/// Read back by name, run over two ranks, checked.
#[test]
fn mla_run() {
    let c = bundle::mla::config(CONFIG, 1, ROWS as u64).unwrap();
    let g = models::mla::program::<models::mla::Moe>(&c);
    let half = |t: Tensor| Tensor::f32(t.shape.clone(), BF16_FORM.roundtrip(t.floats().unwrap()));
    let floats: Vec<Tensor> = weights(&g, &mut Noise(9)).into_iter().map(half).collect();
    let dir = testkit::scratch(&format!("{}-mla-release", std::process::id()));
    write(&dir, &g, &floats);
    let (read, stored, pins) = model("deepseek", &dir, ROWS as u64).unwrap();
    assert_eq!(read, g, "the release reads back as the same graph");
    let tokens = Tensor::u32(
        vec![1, ROWS],
        (0..ROWS as u32).map(|t| t * 7 % 64).collect(),
    );
    let positions = Tensor::u32(vec![1, ROWS], (0..ROWS as u32).collect());
    let want = run(&g, &floats, 0, &[tokens.clone(), positions.clone()]).unwrap();
    let want = want[0].floats().unwrap();
    let sharded = shard(&g).unwrap();
    let settings = runs().pop().unwrap();
    let kind = match settings.gpu() {
        Some(_) => ZoneKind::Gpu,
        None => ZoneKind::Cpu,
    };
    let device = Device::guess(&settings)
        .pinned(pins)
        .zoned(cluster(2, kind));
    let device = device.ranked(vec![0, 1]);
    let sizes = [(Seqs, 1), (Context, ROWS as u64)];
    let plans = [0, 1].map(|r| lower(&sharded, 0, &sizes, &device.clone().as_rank(r)).unwrap());
    let mut ranks = Loopback::open(&plans, &settings).unwrap();
    for (at, w) in stored.iter().enumerate() {
        ranks.load_weight(at as u32, w).unwrap();
    }
    let got = ranks.run(&[tokens.view(), positions.view()]).unwrap();
    let got = got[0].floats().unwrap();
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let (off, cos) = (max_abs_diff(got, want) / scale, cosine_f32(got, want));
    eprintln!(
        "mla over two ranks on {}: off {off:.2e}, cos {cos:.7}",
        ranks.rank(0).device()
    );
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(off < 1e-3 && cos > 0.9999, "off {off:e} cos {cos}");
}
