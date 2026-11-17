use crate::probe::{Kind, Probe, Reading, fit, links, read, suite};
use crate::{Grid, headroom, judge};
use lower::{Device, Pick, Settings, rules};
use sim::Chip;
use testkit::scratch;

/// This build's GPU when ready, else the host.
fn settings() -> Settings {
    run::testing::runs().pop().unwrap()
}

/// A crate's rules register once this one links it.
#[test]
fn linked_rules_register() {
    assert!(rules().iter().any(|r| r.name == "lanes"));
}

#[test]
fn grid_reaches_rules() {
    let settings = settings();
    let mut device = Device::guess(&settings);
    let timed = Grid::tiny().run(&mut device, &settings).unwrap();
    assert!(device.samples() > 0 && !device.name.is_empty());
    // CPU rules need a CPU zone.
    let skipped = |r: &&&lower::Rule| r.name.starts_with("cpu_");
    for rule in rules().iter().filter(|r| !skipped(r)) {
        let ran = timed.iter().any(|(step, _)| step.rule == rule.name);
        assert!(ran, "the grid never ran {}", rule.name);
    }
    // Everything is known now, so a second pass times nothing.
    assert!(Grid::tiny().run(&mut device, &settings).unwrap().is_empty());
    // A later run on this platform reads what this stored.
    let costs = Settings {
        costs: Some(scratch("tessera-calibrate-costs")),
        ..settings
    };
    let file = Device::path(&costs, &device.platform).unwrap();
    device.save(&file).unwrap();
    let stored = Device::stored(device.platform.clone(), &costs).unwrap();
    assert!(stored == device, "a round trip");
    std::fs::remove_file(&file).unwrap();
}

/// On a CPU zone, packed and plain matmuls are timed.
#[test]
fn cpu_matmuls_compete() {
    let cpu = Settings {
        backend: Pick::Cpu,
        ..run::testing::host()
    };
    let grid = Grid {
        rows: vec![1, 8],
        widths: vec![32],
        vocabs: vec![64],
        heads: vec![[4, 2, 16]],
        spans: vec![[4, 4]],
    };
    let timed = grid.run(&mut Device::guess(&cpu), &cpu).unwrap();
    for rule in ["cpu_dense", "cpu_tiled"] {
        assert!(timed.iter().any(|(s, _)| s.rule == rule), "{rule} untimed");
    }
}

/// A made-up chip's reading for one probe.
fn fake(probe: Probe) -> Reading {
    let (threads, trips) = (probe.threads as f64, probe.trips as f64);
    let floor = 2e-6;
    let seconds = match probe.kind {
        // 10 ns a trip, or 1e12 trips a second.
        Kind::Spin => floor + (trips * 10e-9).max(threads * trips / 1e12),
        Kind::Curve => floor + trips * 30e-9,
        // 300 ns a wait, or 100 GB/s.
        Kind::Read { depth } => (trips * 300e-9).max(4.0 * threads * trips * depth as f64 / 100e9),
        Kind::Scan => (trips * 50e-9).max(4.0 * threads * trips / 80e9),
        Kind::Write => 4.0 * threads * trips / 90e9,
        Kind::Chase => trips * 400e-9,
        // 1e13 tile flops a second.
        Kind::Tile => threads / 32.0 * trips * 65_536.0 / 1e13,
    };
    Reading { probe, seconds }
}

#[test]
fn probes_fit_pools() {
    let readings: Vec<Reading> = suite().into_iter().map(fake).collect();
    let fit = fit(&readings);
    let near = |got: f64, want: f64| (got / want - 1.0).abs() < 0.02;
    assert!(near(fit.floor, 2e-6) && near(fit.step, 10e-9));
    assert!(near(fit.curve, 30e-9) && near(fit.rate, 1e12));
    assert!(near(fit.wait, 300e-9) && near(fit.near, 40e-9));
    assert!(near(fit.read, 100e9) && near(fit.scan, 80e9));
    assert!(near(fit.write, 90e9) && near(fit.tile, 1e13), "{fit:?}");
    assert_eq!(fit.pool("a made-up chip").constants().len(), 13);
}

#[test]
fn judge_skips_unmodelled_kernels() {
    let settings = settings();
    let mut device = Device::guess(&settings);
    let timed = Grid::tiny().run(&mut device, &settings).unwrap();
    let verdicts = judge(&sim::m3_pro(), &timed, &settings);
    let names: Vec<&str> = verdicts.iter().map(|v| v.kernel).collect();
    assert!(names.contains(&"lanes") && names.contains(&"map"));
    // Attention states its load later, so judging skips it.
    assert!(!names.contains(&"attend"));
    for v in verdicts {
        let [low, median, high] = v.ratio;
        assert!(0.0 < low && low <= median && median <= high, "{v:?}");
    }
    // A kernel has a bound when it states a load.
    for h in headroom(&sim::m3_pro(), &timed) {
        assert_eq!(h.bound.is_some(), names.contains(&h.kernel), "{h:?}");
    }
}

/// A float probe, an index probe and the links run.
#[test]
fn probes_and_links_run() {
    let settings = settings();
    let probe = |kind, threads, trips| Probe {
        kind,
        threads,
        trips,
    };
    let probes = [probe(Kind::Spin, 256, 16), probe(Kind::Chase, 1, 4)];
    let readings = read(&probes, &settings).unwrap();
    assert!(readings.iter().all(|r| r.seconds > 0.0));
    let cluster = links(&settings).unwrap();
    assert_eq!(cluster.links.len(), 2);
    assert!(cluster.links.iter().all(|l| l.rate > 0.0));
}
