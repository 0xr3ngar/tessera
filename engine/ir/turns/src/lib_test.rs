use crate::{Rules, Table, Turn};

fn table(budget: Option<u64>) -> Table {
    Table::new(Rules {
        budget,
        learn: budget.is_none(),
        quantum: 4.0,
    })
}

/// Loads seat `at` on a device of 100 bytes.
fn load(t: &mut Table, at: usize, seconds: f64, now: f64) {
    t.settle(now);
    assert_eq!(t.turn(at), Some(Turn::Load));
    t.loaded(at, 60, seconds, 100, now);
}

/// Runs one step of seat `at` that leaves `running` answers.
fn step(t: &mut Table, at: usize, admit: bool, running: usize, now: f64) {
    t.settle(now);
    assert_eq!(t.turn(at), Some(Turn::Step { admit }));
    t.stepped(at, (running > 0, running), 60, now);
}

#[test]
fn fitting_models_alternate() {
    let mut t = table(None);
    let (a, b) = (t.add("a", 0, 40, 0.0), t.add("b", 0, 40, 0.0));
    t.settle(0.0);
    // The first load runs alone and gives the budget.
    assert_eq!((t.turn(a), t.turn(b)), (Some(Turn::Load), None));
    t.loaded(a, 40, 1.0, 100, 1.0);
    t.settle(1.0);
    assert_eq!(t.turn(b), Some(Turn::Load));
    t.loaded(b, 40, 1.0, 100, 2.0);
    t.asked(a, 2.0);
    t.asked(b, 2.0);
    // The seat that ran longest ago steps next.
    step(&mut t, b, true, 1, 2.0);
    assert_eq!(t.turn(b), None);
    step(&mut t, a, true, 1, 2.0);
    step(&mut t, b, true, 1, 2.0);
}

#[test]
fn idle_model_leaves() {
    let mut t = table(Some(100));
    let (a, b) = (t.add("a", 0, 60, 0.0), t.add("b", 0, 60, 0.0));
    load(&mut t, a, 1.0, 0.0);
    t.settle(1.0);
    assert_eq!((t.turn(a), t.turn(b)), (Some(Turn::Leave), None));
    t.left(a, 1.0);
    load(&mut t, b, 1.0, 1.0);
    t.asked(a, 2.0);
    t.settle(2.0);
    assert_eq!(t.turn(b), Some(Turn::Leave));
    t.left(b, 2.0);
    load(&mut t, a, 1.0, 2.0);
    let status = t.status("a").expect("seat a");
    assert_eq!((status.resident, status.loads), (true, 2));
}

#[test]
fn busy_model_drains() {
    let mut t = table(Some(100));
    let a = t.add("a", 0, 60, 0.0);
    load(&mut t, a, 10.0, 0.0);
    t.asked(a, 0.0);
    step(&mut t, a, true, 1, 1.0);
    let b = t.add("b", 0, 60, 1.0);
    t.asked(b, 2.0);
    // Ten load seconds buy forty resident seconds.
    step(&mut t, a, true, 1, 39.0);
    step(&mut t, a, false, 1, 41.0);
    step(&mut t, a, false, 0, 42.0);
    t.settle(42.0);
    assert_eq!(t.turn(a), Some(Turn::Leave));
    t.left(a, 42.0);
    load(&mut t, b, 1.0, 42.0);
}

#[test]
fn priority_takes_memory() {
    let mut t = table(Some(100));
    let a = t.add("a", 0, 60, 0.0);
    load(&mut t, a, 10.0, 0.0);
    t.asked(a, 0.0);
    step(&mut t, a, true, 1, 1.0);
    let b = t.add("b", 1, 60, 1.0);
    // A first load does not move a busy model.
    step(&mut t, a, true, 1, 2.0);
    t.asked(b, 2.0);
    step(&mut t, a, false, 1, 2.0);
}
