use crate::flow::Flow;
use tessera_bundle::Overrides;
use tessera_checkpoint::File;
use tessera_engine::{Pick, Settings};
use tessera_testkit::cosine_f32;

#[test]
fn matches_torch() {
    let Some(dir) = tessera_testkit::reference("flux2-klein") else {
        return eprintln!("no flux2-klein reference here; skipped");
    };
    let r = File::open(&dir.join("ref.safetensors")).unwrap();
    let s = Settings {
        backend: Pick::Cpu,
        ..Settings::default()
    };
    let q = &Default::default();
    let mut flow = Flow::open(
        &dir.join("model"),
        &s,
        [1, 64, 256],
        &Overrides::default(),
        q,
    )
    .unwrap();
    let [latents, text, want] = ["latents", "text", "velocity"].map(|n| r.floats(n).unwrap());
    let got = flow.velocity(&latents, &text, 16, 0.7).unwrap();
    let cos = cosine_f32(&got, &want);
    assert!(cos > 0.9999, "cos {cos}");
}

/// Two tokens a side, three channels, four steps.
fn mock(hook: Option<&mut dyn crate::flow::Hook>) -> Vec<f32> {
    let mut x: Vec<f32> = (0..12).map(|i| i as f32 * 0.1 - 0.5).collect();
    let mut sigmas = vec![1.0, 0.75, 0.5, 0.25, 0.0];
    let velocity = |x: &[f32], t: f32| Ok(x.iter().map(|v| v * 0.5 - t).collect());
    crate::flow::euler(&mut x, &mut sigmas, 2, hook, velocity).unwrap();
    x
}

fn hold() -> crate::flow::Hold {
    crate::flow::Hold::new(vec![true, false, false, true], vec![0.25; 12])
}

#[test]
fn hold_reaches_known() {
    let (free, held) = (mock(None), mock(Some(&mut hold())));
    for (i, (f, h)) in free.iter().zip(&held).enumerate() {
        let want = if !(3..9).contains(&i) { 0.25 } else { *f };
        assert!((h - want).abs() < 1e-6, "{i}: {h} against {want}");
    }
}

#[cfg(feature = "script")]
#[test]
fn script_holds() {
    let source = "let start;
export function step(s) {
  if (s.phase == 'before') { if (s.index == 0) start = s.latents.slice(); return; }
  for (const i of [0, 1, 2, 9, 10, 11]) s.velocity[i] = start[i] - 0.25;
}";
    let script = tessera_script::Script::load("hold.js", source, Default::default());
    let mut script = crate::flow::Scripted::new(script.unwrap()).unwrap();
    assert_eq!(mock(Some(&mut script)), mock(Some(&mut hold())));
}
