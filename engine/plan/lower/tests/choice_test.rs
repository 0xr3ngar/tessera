//! How selection picks among rules: cost, favor and limits.

use graph::DimName::{self, Hidden, Rows, Seqs};
use graph::{Act, DType, Dim, Graph, TensorType};
use lower::kernels::Map;
use lower::sites;
use lower::testing::host;
use lower::{Device, Draft, Kernel, Limits, Needs, Plan, Platform, T, Tree, lower, select};
use trace::{EntryKind, Model, Port, Role};

/// A map stand-in with its own cost.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "cheap",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.len,
    guess = 1e-9,
)]
struct Cheap {
    len: usize,
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "dear",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.len,
    guess = 1e-8,
)]
struct Dear {
    len: usize,
}

/// The cheapest, but only for 32 lanes.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "lanes",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.len,
    needs = Needs::lanes(),
    guess = 1e-12,
)]
struct Lanes {
    len: usize,
}

fn cheap(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Chain { reads, len, .. } = tree.site(at)?;
    Some(vec![Draft::new(Cheap { len }, reads, len)])
}

fn dear(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Chain { reads, len, .. } = tree.site(at)?;
    Some(vec![Draft::new(Dear { len }, reads, len)])
}

fn lanes(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Chain { reads, len, .. } = tree.site(at)?;
    Some(vec![Draft::new(Lanes { len }, reads, len)])
}

lower::rule!("cheap", cheap);
lower::rule!("dear", dear);
lower::rule!("lanes", lanes);

const SIZES: [(DimName, u64); 2] = [(Seqs, 2), (Rows, 3)];

/// One chain over rows.
fn chain() -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::fixed(Hidden, 16),
    ]);
    let ty = TensorType::new([Seqs, Rows, Hidden], DType::F32);
    let role = Role::Logits;
    let [x, y] = m.entry(
        EntryKind::Forward,
        [
            Port {
                role,
                ty: ty.clone(),
            },
            Port { role, ty },
        ],
    );
    m.finish([(Role::Logits, &x.add(&y).act(Act::Silu))])
}

fn plan(device: &Device) -> Plan {
    lower(&chain(), 0, &SIZES, device).unwrap()
}

fn names(plan: &Plan) -> Vec<&'static str> {
    plan.steps.iter().map(|s| s.kernel.name()).collect()
}

/// A device where only `cheap` and `dear` apply.
fn pair() -> Device {
    Device::guess(&host()).without("map").without("lanes")
}

#[test]
fn rules_compete() {
    let picked = select(&chain(), 0, &SIZES, &pair()).unwrap();
    assert!(picked.contested.contains("cheap") && picked.contested.contains("dear"));
}

#[test]
fn cost_picks_the_rule() {
    assert_eq!(names(&plan(&pair())), ["cheap"]);
    // Samples outrank the guesses.
    let (slow, quick) = (plan(&pair()), plan(&pair().favoring("dear")));
    let mut measured = pair();
    measured.fit([(&slow.steps[0], 1e-3), (&quick.steps[0], 1e-6)]);
    let fitted = plan(&measured);
    assert_eq!(names(&fitted), ["dear"]);
    assert!(measured.cost(&fitted).seconds < measured.cost(&slow).seconds);
    // A favored rule wins whatever it costs.
    assert_eq!(names(&plan(&measured.favoring("cheap"))), ["cheap"]);
}

#[test]
fn limits_bar_kernels() {
    let device = Device::guess(&host()).without("map");
    assert_eq!(names(&plan(&device)), ["lanes"]);
    // A 64-lane device runs none of the 32-lane kernels.
    let wide = Limits {
        lanes: 64,
        ..Limits::assumed()
    };
    let wide = device.on(Platform::assumed(wide));
    assert_eq!(names(&plan(&wide)), ["cheap"]);
}
