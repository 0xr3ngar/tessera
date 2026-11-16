use crate::cache::{Cache, KEEP_SECS, key, now};

#[test]
fn cache_keeps_week() {
    let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/target/scratch"));
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(format!("simcheck-{}.bin", std::process::id()));
    let mut c = Cache::default();
    c.insert(key("a"), vec![1.0, 2.0]);
    c.insert(key("b"), vec![3.0, 4.0]);
    // Eight days on, only the entry used then is left.
    let later = now() + KEEP_SECS + 86_400;
    c.save_at(&path, &[key("a")], later).unwrap();
    let back = Cache::load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(back.get(key("a")), Some(&vec![1.0, 2.0]));
    assert!(back.get(key("b")).is_none() && key("a") != key("b"));
    assert!(Cache::load(&path).unwrap().get(key("a")).is_none());
}
