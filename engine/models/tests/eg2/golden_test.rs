//! EmbeddingGemma 2: the exlr8 release and the device.

use super::{GOLDEN, release};
use crate::common::{Case, interpreted, loaded, pulled, unit};
use checkpoint::exlr8::Release;
use interp::Tensor;
use lower::{Device, Settings, lower};
use run::Run;
use std::path::Path;
use testkit::cosine_f32;
use trace::Graph;

/// The exlr8 release: its embeddings against the reference.
fn check_exlr8(dir: &Path, g: &Graph, cases: &[Case]) {
    let release = match std::fs::read_to_string(dir.join("exlr8.repo")) {
        Ok(repo) => {
            let (repo, revision) = repo.trim().split_once('@').unwrap_or((repo.trim(), "main"));
            eprintln!("exlr8: streaming {repo} at {revision}");
            Release::remote(repo, revision)
        }
        Err(_) => Release::open(&dir.join("exlr8")),
    };
    let w = pulled("eg2", g, &release);
    for c in cases {
        let got = unit(&interpreted(g, &w, 1, &c.tokens));
        let cos = cosine_f32(&got, c.output("embedding"));
        eprintln!("exlr8 {}: cos {cos:.5} against the bf16 reference", c.name);
        assert!(cos > 0.98, "{}: cos {cos}", c.name);
    }
}

/// Every case through the forward entry on this machine's GPU.
fn check_device(settings: &Settings, g: &Graph, w: &[Tensor], cases: &[Case]) -> Vec<Vec<f32>> {
    let mut embeddings = Vec::new();
    let mut device = Device::guess(settings);
    let mut runner: Option<Box<dyn Run>> = None;
    for c in cases {
        let sizes = c.sizes();
        let feeds = c.inputs();
        let views = [feeds[0].view(), feeds[1].view()];
        let r = match runner.as_mut() {
            Some(r) => r,
            None => {
                let plan = lower(g, 0, &sizes, &device).unwrap();
                let r = loaded(plan, g, w, settings);
                device = Device::stored(r.platform(), settings).unwrap();
                runner.insert(r)
            }
        };
        let plan = lower(g, 0, &sizes, &device).unwrap();
        r.replan(&plan).unwrap();
        r.feed(&views).unwrap();
        let (slot, steps) = (r.plan().outputs[0].slot, r.plan().steps.len());
        r.submit(0..steps).unwrap();
        let out = r.read(slot).unwrap().floats().unwrap().to_vec();
        embeddings.push(out);
    }
    embeddings
}

#[test]
fn reference_checks() {
    let g = release();
    let Some(reference) = GOLDEN.reference("eg2", &g) else {
        return eprintln!("TESSERA_EG2 is not set; skipped");
    };
    let settings = settings::read();
    let dir = settings.reference("eg2").unwrap();
    let cases = &reference.cases;
    if dir.join("exlr8").is_dir() || dir.join("exlr8.repo").is_file() {
        check_exlr8(&dir, &g, &cases[..1]);
    }
    if settings.gpu().is_none() {
        return eprintln!("no GPU here; skipped");
    }
    let got = check_device(&settings, &g, &reference.weights, cases);
    for (c, got) in cases.iter().zip(&got) {
        GOLDEN.release(&format!("{} on the device", c.name), got, c);
    }
}
