//! The tiled matmul against the CPU interpreter, on the GPU.

// The other matmul rules come from here.
use kernels as _;

use codegen::trellis;
use graph::DimName::Rows;
use interp::{Tensor, run};

mod common;
use common::{linear, noise, terms};
use linear::{SCHEDULES, Schedule};
use lower::form::FORMS;
use lower::testing::{emulations, host, runs};
use lower::{Device, Dialect, Pins, Settings, Source, lower};
use run::open;
use testkit::{Noise, cosine_f32, max_abs_diff};

/// A device that can only choose schedule `at`.
fn only(at: usize, settings: &Settings) -> Device {
    let others = SCHEDULES.iter().enumerate().filter(|(i, _)| *i != at);
    let bar = |d: Device, (_, s): (usize, &Schedule)| {
        FORMS.iter().fold(d, |d, f| d.without(f.suffixed(s.name)))
    };
    others.fold(Device::guess(settings).favoring("tiled"), bar)
}

#[test]
fn rule_registers() {
    let g = linear(72, 50);
    let device = Device::guess(&host()).favoring("tiled");
    let plan = lower(&g, 0, &[(Rows, 37)], &device).unwrap();
    let Source::Made(body) = plan.steps[0].kernel.source() else {
        panic!("tiled is generated");
    };
    for dialect in [Dialect::Metal, Dialect::Cuda] {
        let text = body.text().render(dialect);
        assert!(text.contains("TILE_MMA(acc_t[i][j], ta[i], tb[j])"));
    }
}

#[test]
fn tiles_match_cpu() {
    let mut draws = Noise(29);
    // Sizes off the block, on it, and beyond it.
    let sizes = [
        (1, 3, 4),
        (37, 50, 72),
        (64, 64, 64),
        (130, 96, 1024),
        (256, 512, 1024),
    ];
    let runs = emulations();
    for (m, n, k) in sizes {
        let g = linear(k, n);
        let x = noise(vec![m as usize, k as usize], &mut draws);
        for form in common::forms(k as usize) {
            // The reference sees the weights as the kernel stores them.
            let w = noise(vec![n as usize, k as usize], &mut draws);
            let rounded = form.roundtrip(w.floats().unwrap());
            let weights = [Tensor::f32(w.shape.clone(), rounded)];
            let want = run(&g, &weights, 0, std::slice::from_ref(&x)).unwrap();
            for ((at, schedule), settings) in SCHEDULES.iter().enumerate().zip(runs.iter().cycle())
            {
                // The host skips the largest size and the timing.
                let host = settings.gpu().is_none();
                if host && (m, n, k) == sizes[sizes.len() - 1] {
                    continue;
                }
                let device = only(at, settings).pinned(Pins::all(form));
                let plan = lower(&g, 0, &[(Rows, m)], &device).unwrap();
                let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                let name = form.suffixed(schedule.name);
                assert_eq!(names, [name]);
                let terms = terms(&plan);
                let mut runner = open(plan, settings).unwrap();
                runner.load(0, weights[0].view()).unwrap();
                let got = runner.run(&[x.view()]).unwrap().remove(0);
                let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
                let (xs, ws) = (x.floats().unwrap(), weights[0].floats().unwrap());
                let bound = terms.matmul(&runner.precision(), xs, ws, k as usize);
                let label = format!("{m}x{n}x{k} {name} on {}", runner.device());
                assert!(off <= bound, "{label}: off by {off:e}, allowed {bound:e}");
                // A bound past the outputs themselves says nothing.
                let most = want[0]
                    .floats()
                    .unwrap()
                    .iter()
                    .fold(0.0f32, |m, v| m.max(v.abs()));
                assert!(
                    bound < most / 10.0,
                    "{label}: {bound:e} against outputs of {most:e}"
                );
                if host {
                    eprintln!("{label}: off by {off:.2e} of {bound:.2e}");
                    continue;
                }
                let seconds = runner.run_timed(&[x.view()]).unwrap()[0];
                let rate = 2.0 * (m * n * k) as f64 / seconds / 1e12;
                eprintln!("{label}: {rate:.2} TFLOP/s, off by {off:.2e} of {bound:.2e}");
            }
        }
    }
}

#[test]
fn codes_match_decode() {
    let g = linear(256, 256);
    let mut draws = Noise(3);
    let (inp, out) = (256, 256);
    for k in [2, 3, 4] {
        let form = trellis::form_for(k).unwrap();
        let entries: Vec<u8> = (0..out * inp * k as usize / 8)
            .map(|_| (draws.draw() * 256.0) as u8)
            .collect();
        let bytes = trellis::packed(inp, out, 1, &entries);
        let decoded: Vec<f32> = (0..out * inp).map(|i| form.unpack(&bytes, i)).collect();
        let weights = [Tensor::f32(vec![out, inp], decoded)];
        // Few rows take the lane kernel, many the tiled one.
        for (settings, rows) in runs().iter().flat_map(|s| [3usize, 40].map(|r| (s, r))) {
            let x = noise(vec![rows, inp], &mut draws);
            let want = run(&g, &weights, 0, std::slice::from_ref(&x)).unwrap();
            let device = Device::guess(settings).pinned(Pins::all(form));
            let plan = lower(&g, 0, &[(Rows, rows as u64)], &device).unwrap();
            let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
            assert!(
                names.iter().all(|n| n.contains(form.name())),
                "{names:?} read the codes"
            );
            let mut runner = open(plan, settings).unwrap();
            runner.load_packed(0, &bytes).unwrap();
            let got = runner.run(&[x.view()]).unwrap().remove(0);
            let cos = cosine_f32(got.floats().unwrap(), want[0].floats().unwrap());
            assert!(cos > 0.99999, "K{k}, {rows} rows: cos {cos}");
        }
    }
}
