use crate::mla::{config, rules};
use crate::names::resolve;
use checkpoint::{Slice, Stored};
use engine::PathSeg;
use engine::mla::{RotatedMoe, program};

/// Two layers, one routed with two experts, both indexing.
const CONFIG: &str = r#"{
  "vocab_size": 64, "hidden_size": 128, "num_hidden_layers": 2,
  "num_attention_heads": 2, "q_lora_rank": 32, "kv_lora_rank": 32,
  "qk_nope_head_dim": 16, "qk_rope_head_dim": 16, "v_head_dim": 32,
  "intermediate_size": 128, "moe_intermediate_size": 128,
  "n_routed_experts": 2, "num_experts_per_tok": 2, "first_k_dense_replace": 1,
  "routed_scaling_factor": 2.5, "rope_parameters": {"rope_theta": 8000000},
  "rms_norm_eps": 1e-5, "index_n_heads": 2, "index_head_dim": 32,
  "index_topk": 4, "index_topk_freq": 1
}"#;

/// Every weight finds the slice an exlr8 release names.
#[test]
fn rotated_names_resolve() {
    let c = config(CONFIG, 1, 8).unwrap();
    let g = program::<RotatedMoe>(&c);
    let rules = rules(&c, true);
    let mut slices = Vec::new();
    for w in &g.weights {
        let seg = |s: &PathSeg| match s {
            PathSeg::Field(f) => f.clone(),
            PathSeg::Index(i) => i.to_string(),
        };
        let mut path: Vec<String> = w.path.iter().map(seg).collect();
        let part = path
            .last()
            .filter(|p| ["w", "suh", "svh"].contains(&p.as_str()))
            .cloned();
        if part.is_some() {
            path.pop();
        }
        let (_, stem) = rules.iter().find(|(p, _)| *p == path.join(".")).unwrap();
        let bare = ["bias", ".su", ".sv", "signs_in", "signs_out"];
        let name = match part.as_deref() {
            Some("w") => format!("{stem}.codes"),
            Some(end) => format!("{stem}.{end}"),
            None if bare.iter().any(|b| stem.ends_with(b)) => stem.clone(),
            None => format!("{stem}.weight"),
        };
        let axis = |e: &engine::DimExpr| e.value(&g.dims, &|_| None).unwrap() as usize;
        let shape = w.ty.shape.0.iter().map(axis).collect();
        slices.push(Slice {
            name,
            shape,
            stored: Stored::F32,
        });
    }
    let claims = resolve(&g, &slices, &[&rules]).unwrap();
    assert_eq!(claims.len(), g.weights.len());
    let at = "model.layers.1.mlp.";
    assert!(
        slices
            .iter()
            .any(|s| s.name == format!("{at}experts.down.su"))
    );
    assert!(
        slices
            .iter()
            .any(|s| s.name == format!("{at}shared_experts.up_proj.codes"))
    );
}
