//! Modules from text: one offers a matmul, others fail.

// The linked matmul rules the module's rule competes with.
use kernels as _;
use linear as _;

use graph::DimName::{Ffn, Hidden, Rows};
use interp::Tensor;
use lower::{Arg, Device, Shape, Source, lower};
use run::testing::{host, runs};
use testkit::{Noise, max_abs_diff};
use trace::{DType, Dim, EntryKind, Graph, Linear, Model, Port, Role, TensorType, Weights};

#[derive(Weights)]
struct Layer {
    #[shape(Hidden, Ffn)]
    w: Linear,
}

/// `x[rows, 64]` times a 64 by 32 weight.
fn linear() -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Rows, 16),
        Dim::fixed(Hidden, 64),
        Dim::fixed(Ffn, 32),
    ]);
    let w: Layer = trace::declare(&m);
    let ty = TensorType::new([Rows, Hidden], DType::F32);
    let role = Role::Logits;
    let [x] = m.entry(EntryKind::Forward, [Port { role, ty }]);
    m.finish([(Role::Logits, &x.matmul(&w.w))])
}

const MANIFEST: &str = r##"{"abi": 1,
  "kernels": [{"name": "wasm_gemm", "entry": "gemm", "metal": "#include <metal_stdlib>\nusing namespace metal;\nkernel void gemm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device float* y [[buffer(2)]], constant uint& m [[buffer(3)]], constant uint& n [[buffer(4)]], constant uint& k [[buffer(5)]], uint2 g [[threadgroup_position_in_grid]], uint2 t [[thread_position_in_threadgroup]]) { uint col = g.x * 32 + t.x; uint row = g.y; if (row >= m || col >= n) return; float s = 0; for (uint i = 0; i < k; i++) s += x[row * k + i] * w[col * k + i]; y[row * n + col] = s; }",
    "cuda": "extern \"C\" __global__ void gemm(const float* x, const float* w, float* y, unsigned m, unsigned n, unsigned k) { unsigned col = blockIdx.x * 32 + threadIdx.x, row = blockIdx.y; if (row >= m || col >= n) return; float s = 0; for (unsigned i = 0; i < k; i++) s += x[row * k + i] * w[col * k + i]; y[row * n + col] = s; }"}],
  "rules": [{"name": "wasm_matmul", "site": "matmul"}]}"##;

const OFFER: &str = r#"[{"kernel": "wasm_gemm", "reads": [0, 1], "len": 128,
  "counts": [4, 32, 64], "shape": {"grid": {"across": 1, "down": 4, "threads": 32}},
  "work": 8192, "guess": 1e-15}]"#;

/// A WAT string literal holding `text`.
fn quoted(text: &str) -> String {
    let text = text.replace('\\', "\\\\").replace('"', "\\\"");
    text.replace('\n', "\\n")
}

/// A module whose `manifest` and `propose` return fixed JSON.
fn module(manifest: &str, body: &str) -> Vec<u8> {
    let packed = |at: u64, text: &str| (at << 32) | text.len() as u64;
    let wat = format!(
        r#"(module
  (memory (export "memory") 1)
  (data (i32.const 0) "{}")
  (data (i32.const 2048) "{}")
  (func (export "alloc") (param i32) (result i32) (i32.const 4096))
  (func (export "manifest") (result i64) {body} (i64.const {}))
  (func (export "propose") (param i32 i32) (result i64) (i64.const {})))"#,
        quoted(manifest),
        quoted(OFFER),
        packed(0, manifest),
        packed(2048, OFFER),
    );
    wat::parse_str(wat).unwrap()
}

#[test]
fn module_rule_wins() {
    let rules = wasm::load(&module(MANIFEST, "")).unwrap();
    let names: Vec<&str> = rules.iter().map(|r| r.name).collect();
    assert_eq!(names, ["wasm_matmul"]);
    let device = Device::guess(&host()).adding(rules[0]);
    let plan = lower(&linear(), 0, &[(Rows, 4)], &device).unwrap();
    let step = &plan.steps[0];
    assert_eq!((plan.steps.len(), step.rule), (1, "wasm_matmul"));
    let Source::Fixed { entry, cuda, .. } = step.kernel.source() else {
        panic!("module kernels are fixed text");
    };
    assert!(entry == "gemm" && cuda.contains("extern \"C\""));
    let launch = &step.kernel.launches(step)[0];
    let counts: Vec<u32> = launch.args[3..].iter().filter_map(Arg::u32).collect();
    assert_eq!(counts, [4, 32, 64]);
    let grid = Shape::Grid {
        across: 1,
        down: 4,
        threads: 32,
    };
    assert_eq!(launch.shape, grid);
    // Only a GPU runs fixed text. Check it there.
    let mut draws = Noise(5);
    let w = Tensor::f32(vec![32, 64], (0..32 * 64).map(|_| draws.draw()).collect());
    let x = Tensor::f32(vec![4, 64], (0..4 * 64).map(|_| draws.draw()).collect());
    let want = interp::run(
        &linear(),
        std::slice::from_ref(&w),
        0,
        std::slice::from_ref(&x),
    );
    for settings in runs().iter().filter(|s| s.gpu().is_some()) {
        let device = Device::guess(settings).adding(rules[0]);
        let plan = lower(&linear(), 0, &[(Rows, 4)], &device).unwrap();
        let mut runner = run::open(plan, settings).unwrap();
        runner.load(0, w.view()).unwrap();
        let got = runner.run(&[x.view()]).unwrap().remove(0);
        let want = want.as_ref().unwrap()[0].floats().unwrap();
        let off = max_abs_diff(got.floats().unwrap(), want);
        assert!(off < 1e-4, "{}: off by {off:e}", runner.device());
    }
}

#[test]
fn bad_modules_fail() {
    let newer = MANIFEST.replace(r#""abi": 1"#, r#""abi": 2"#);
    assert!(matches!(
        wasm::load(&module(&newer, "")),
        Err(wasm::Error::Abi(2))
    ));
    // An endless loop runs out of fuel.
    let spin = "(loop (br 0))";
    assert!(matches!(
        wasm::load(&module(MANIFEST, spin)),
        Err(wasm::Error::Wasm(_))
    ));
    let imports = r#"(module (import "env" "f" (func)) (memory (export "memory") 1))"#;
    let imports = wat::parse_str(imports).unwrap();
    assert!(matches!(wasm::load(&imports), Err(wasm::Error::Wasm(_))));
}
