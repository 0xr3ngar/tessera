//! DeepSeek-style releases: config, weight names, weights.

use crate::names::Rule;
use crate::pull;
use checkpoint::File;
use engine::mla::{Config, Index, program};
use engine::{Graph, Pins, Stack, Weight};
use serde::Deserialize;
use std::error::Error;
use std::path::Path;

/// The `config.json` keys a DeepSeek-V3 release sets.
#[derive(Deserialize)]
struct Hf {
    vocab_size: u64,
    hidden_size: u64,
    num_hidden_layers: u64,
    num_attention_heads: u64,
    q_lora_rank: u64,
    kv_lora_rank: u64,
    qk_nope_head_dim: u64,
    qk_rope_head_dim: u64,
    v_head_dim: u64,
    intermediate_size: u64,
    moe_intermediate_size: u64,
    n_routed_experts: u64,
    num_experts_per_tok: u64,
    first_k_dense_replace: u64,
    routed_scaling_factor: f32,
    rope_theta: f32,
    rms_norm_eps: f32,
    /// DeepSeek-V3.2 keys. Every layer past the dense ones indexes.
    index_n_heads: Option<u64>,
    index_head_dim: Option<u64>,
    index_topk: Option<u64>,
}

/// The model `json` describes: `seqs` sequences of `context` rows.
pub fn config(json: &str, seqs: u64, context: u64) -> Result<Config, serde_json::Error> {
    let h: Hf = serde_json::from_str(json)?;
    let sizes = [
        h.vocab_size,
        h.hidden_size,
        h.num_hidden_layers,
        h.num_attention_heads,
        h.v_head_dim,
        h.moe_intermediate_size,
    ];
    let index = match (h.index_n_heads, h.index_head_dim, h.index_topk) {
        (Some(heads), Some(dim), Some(keep)) => Some(Index {
            heads,
            dim,
            keep,
            every: 1,
        }),
        _ => None,
    };
    Ok(Config {
        stack: Stack::sized(sizes, seqs, context),
        dense_layers: h.first_k_dense_replace,
        nope: h.qk_nope_head_dim,
        rope: h.qk_rope_head_dim,
        rope_theta: h.rope_theta,
        q_latent: h.q_lora_rank,
        latent: h.kv_lora_rank,
        dense_ffn: h.intermediate_size,
        experts: h.n_routed_experts,
        top_k: h.num_experts_per_tok,
        route_scale: h.routed_scaling_factor,
        index,
        eps: h.rms_norm_eps,
    })
}

/// Each weight path to its HF slice. `{}` is an expert.
pub fn rules(c: &Config) -> Vec<Rule> {
    let mut rules: Vec<Rule> = Vec::new();
    let mut add = |path: String, stem: String| rules.push((path, stem));
    add("embed".into(), "model.embed_tokens".into());
    add("norm".into(), "model.norm".into());
    add("head".into(), "lm_head".into());
    let attention = [
        ("norm", "input_layernorm"),
        ("q_a", "self_attn.q_a_proj"),
        ("q_norm", "self_attn.q_a_layernorm"),
        ("q_b", "self_attn.q_b_proj"),
        ("kv_a", "self_attn.kv_a_proj_with_mqa"),
        ("kv_norm", "self_attn.kv_a_layernorm"),
        ("kv_b", "self_attn.kv_b_proj"),
        ("o", "self_attn.o_proj"),
    ];
    let indexer = [
        ("wq", "wq_b"),
        ("wk", "wk"),
        ("k_norm", "k_norm"),
        ("k_bias", "k_norm.bias"),
        ("weights", "weights_proj"),
    ];
    let mlp = [
        ("gate", "gate_proj"),
        ("up", "up_proj"),
        ("down", "down_proj"),
    ];
    let mut indexers = 0;
    for l in 0..c.stack.layers {
        let at = format!("model.layers.{l}");
        for (field, stem) in attention {
            add(format!("attn.{l}.{field}"), format!("{at}.{stem}"));
        }
        add(
            format!("mlp_norm.{l}"),
            format!("{at}.post_attention_layernorm"),
        );
        if c.indexes(l) {
            for (field, stem) in indexer {
                let path = format!("indexers.{indexers}.{field}");
                add(path, format!("{at}.self_attn.indexer.{stem}"));
            }
            indexers += 1;
        }
        // Routed layers count from the first one.
        let Some(m) = l.checked_sub(c.dense_layers) else {
            for (field, stem) in mlp {
                add(format!("dense.{l}.{field}"), format!("{at}.mlp.{stem}"));
            }
            continue;
        };
        add(format!("moe.{m}.router"), format!("{at}.mlp.gate"));
        let bias = format!("{at}.mlp.gate.e_score_correction_bias");
        add(format!("moe.{m}.router_bias"), bias);
        for (field, stem) in mlp {
            let experts = format!("{at}.mlp.experts.{{}}.{stem}");
            add(format!("moe.{m}.{field}"), experts);
            let shared = format!("{at}.mlp.shared_experts.{stem}");
            add(format!("moe.{m}.shared.{field}"), shared);
        }
    }
    rules
}

pub struct Release {
    pub config: Config,
    pub graph: Graph,
    pub weights: Vec<Weight>,
    /// Forms packed in the file, when any are.
    pub pins: Pins,
}

/// The release in `dir`: `config.json` and `model.safetensors`.
pub fn release(
    dir: &Path,
    seqs: u64,
    context: u64,
) -> Result<Release, Box<dyn Error + Send + Sync>> {
    let json = std::fs::read_to_string(dir.join("config.json"))?;
    let config = config(&json, seqs, context)?;
    let graph = program(&config);
    let file = File::open(&dir.join("model.safetensors"))?;
    let rules = rules(&config);
    let (weights, pins) = pull(&graph, &file, config.stack.layers, &[&rules])?;
    Ok(Release {
        config,
        graph,
        weights,
        pins,
    })
}
