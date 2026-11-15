use crate::{dot, parallel};

#[test]
fn dots_and_chunks() {
    let a: Vec<f32> = (0..19).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..19).map(|i| (i % 3) as f32).collect();
    let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    assert_eq!(dot(&a, &b), want);
    let seen = std::sync::Mutex::new(vec![false; 1000]);
    parallel(1000, |range| {
        for i in range {
            seen.lock().unwrap()[i] = true;
        }
    });
    assert!(seen.lock().unwrap().iter().all(|s| *s));
}
