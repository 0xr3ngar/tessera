use crate::shm;
use engine_comm::Transport;

/// Rank 1 waits until rank 0 lays out the file.
#[test]
fn shm_joins() {
    let path = testkit::scratch(&format!("join-{}", std::process::id()));
    std::thread::scope(|s| {
        let late = s.spawn(|| shm(&path, 1, 2, 16).unwrap());
        std::thread::sleep(std::time::Duration::from_millis(20));
        let mut first = shm(&path, 0, 2, 16).unwrap();
        let mut late = late.join().unwrap();
        let a = s.spawn(move || first.exchange(&[7]).unwrap());
        assert_eq!(late.exchange(&[8]).unwrap(), [vec![7], vec![8]]);
        assert_eq!(a.join().unwrap(), [vec![7], vec![8]]);
    });
    let _ = std::fs::remove_file(&path);
}
