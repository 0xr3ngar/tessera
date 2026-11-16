//! Plans over a GPU zone and a host zone.

use graph::DimName::{Rows, Seqs, Vocab};
use graph::{DType, Dim, Graph, TensorType};
use lower::kernels::Map;
use lower::sites;
use lower::testing::host;
use lower::{
    Cluster, Device, Draft, Error, Kernel, Link, Plan, Step, T, Tree, Zone, ZoneKind, lower,
};
use run::open;
use sim::Tag;
use testkit::scratch;
use trace::{EntryKind, Model, Port, Role};

/// Picks each row's largest entry on the host.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "host_top",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.rows,
    guess = 0.0,
    runs_on = ZoneKind::Cpu,
)]
struct HostTop {
    rows: usize,
}

/// Picks each row's largest entry on the GPU, dearly.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "device_top",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.rows * 600,
    guess = 1e-8,
)]
struct DeviceTop {
    rows: usize,
}

fn host_top(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Top { x, rows, k: 1, .. } = tree.site(at)? else {
        return None;
    };
    Some(vec![Draft::new(HostTop { rows }, [x], rows)])
}

fn device_top(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Top { x, rows, k: 1, .. } = tree.site(at)? else {
        return None;
    };
    Some(vec![Draft::new(DeviceTop { rows }, [x], rows)])
}

lower::rule!("host_top", host_top);
lower::rule!("device_top", device_top);

/// Each row's likeliest entry.
fn likeliest() -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::fixed(Vocab, 600),
    ]);
    let ty = TensorType::new([Seqs, Rows, Vocab], DType::F32);
    let role = Role::Logits;
    let [logits] = m.entry(EntryKind::Forward, [Port { role, ty }]);
    m.finish([(Role::Tokens, &logits.top_k(1u64))])
}

/// A GPU and a host, linked both ways at `latency`.
fn two(latency: f64) -> Cluster {
    let mut cluster = Cluster::one("gpu");
    let host = cluster.add(Zone {
        name: "host".into(),
        kind: ZoneKind::Cpu,
        parent: None,
    });
    for (from, to) in [(0, host), (host, 0)] {
        cluster.links.push(Link {
            from,
            to,
            latency,
            rate: 1e12,
            tag: Tag::Assumed,
        });
    }
    cluster
}

fn step(device: &Device) -> Plan {
    lower(&likeliest(), 0, &[(Seqs, 1), (Rows, 2)], device).unwrap()
}

fn named<'a>(plan: &'a Plan, kernel: &'a str) -> impl Iterator<Item = &'a Step> {
    plan.steps.iter().filter(move |s| s.kernel.name() == kernel)
}

#[test]
fn cheap_links_place_work() {
    let device = Device::guess(&host()).zoned(two(1e-6));
    let plan = step(&device);
    assert!(plan.zoned());
    let [top] = &named(&plan, "host_top").collect::<Vec<_>>()[..] else {
        panic!("one top on the host");
    };
    assert_eq!((top.zone, plan.zone(top.write.buffer)), (1, 1));
    assert!(named(&plan, "device_top").next().is_none());
    // The logits move over, and the ids move back.
    let moves: Vec<&Step> = named(&plan, "move").collect();
    assert_eq!(moves.len(), 2);
    for m in &moves {
        assert_ne!(plan.zone(m.reads[0].buffer), m.zone);
        assert_eq!(plan.zone(m.write.buffer), m.zone);
        assert!(m.launches().is_empty());
    }
    assert_eq!(
        (moves[0].write, moves[1].reads[0]),
        (top.reads[0], top.write)
    );
    for o in &plan.outputs {
        assert_eq!(
            plan.zone(o.slot.buffer),
            0,
            "outputs lie where the runner reads"
        );
    }
    let cost = device.cost(&plan);
    assert!(cost.seconds.is_finite() && cost.seconds > 0.0);
    assert!(matches!(open(plan, &host()), Err(Error::Unsupported(_))));
}

#[test]
fn slow_links_stay() {
    let home = step(&Device::guess(&host()));
    assert!(!home.zoned() && named(&home, "device_top").count() == 1);
    assert_eq!(step(&Device::guess(&host()).zoned(two(1.0))), home);
    // Unlinked zones cannot be reached at all.
    let mut apart = two(0.0);
    apart.links.clear();
    assert_eq!(step(&Device::guess(&host()).zoned(apart)), home);
    assert_eq!(home.zones, vec![0; home.buffers.len()]);
}

#[test]
fn clusters_are_stored() {
    let device = Device::guess(&host()).zoned(two(2e-6));
    let file = scratch("tessera-lower-zones-test.json");
    device.save(&file).unwrap();
    let loaded = Device::load(&file).unwrap();
    std::fs::remove_file(&file).unwrap();
    assert!(loaded == device && loaded.zones() == 2);
    assert_eq!(loaded.transfer(0, 1, 1_000_000), Some(2e-6 + 1e-6));
    assert_eq!(loaded.transfer(1, 1, 1_000_000), Some(0.0));
}
