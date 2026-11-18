//! Two uneven ranks on hardware this machine does not have.

use super::passes;
use crate::model::{Model, named};
use crate::pump::laps;
use crate::ranks::{Wire, spec};
use crate::stages::Setup;

#[test]
fn uneven_ranks_meet() {
    let specs = [spec("strix:3").unwrap(), spec("gb10:1").unwrap()];
    let parts = sensai_graph::Parts(vec![3, 1]);
    let model = Model::new("qwen3".into(), named("qwen3", false).unwrap(), &parts).unwrap();
    let setup = Setup {
        model: &model,
        specs: &specs,
        wire: Wire {
            latency: 1e-5,
            rate: 1e9,
        },
        pins: sensai_lower::Pins::default(),
    };
    let passes = passes(&setup, 4, 1).unwrap();
    let held = |r: usize| {
        passes[0].ranks[r]
            .held
            .iter()
            .filter_map(|h| h.share)
            .next()
    };
    assert_eq!(held(0).map(|s| s[2]), Some(3 * held(1).unwrap()[2]));
    let laps = laps(&passes).unwrap();
    for e in laps.iter().flat_map(|l| &l.events).filter(|e| e.sum) {
        let other = laps.iter().flat_map(|l| &l.events);
        let met = other.filter(|o| o.sum && o.start == e.start).count();
        assert_eq!(met, specs.len());
    }
}
