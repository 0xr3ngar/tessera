use crate::{Shm, Sockets, Transport, summed};
use std::net::TcpListener;

#[test]
fn rank_order() {
    let parts = [[1.0f32, 2.0], [10.0, 20.0], [100.0, 200.0]];
    assert_eq!(summed(parts.iter().map(|p| &p[..])), [111.0, 222.0]);
}

/// Three ranks on threads, each sending its rank's byte.
#[test]
fn sockets_exchange() {
    let listeners: Vec<TcpListener> = (0..3)
        .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    let addrs: Vec<_> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
    let seen: Vec<Vec<Vec<u8>>> = std::thread::scope(|s| {
        let runs: Vec<_> = (listeners.into_iter().enumerate())
            .map(|(r, listener)| {
                let addrs = &addrs;
                s.spawn(move || {
                    let mut sockets = Sockets::join(r as u32, listener, addrs).unwrap();
                    assert_eq!((sockets.rank(), sockets.ranks()), (r as u32, 3));
                    let first = sockets.exchange(&[r as u8; 3]).unwrap();
                    // A second round reuses the connections.
                    let second = sockets.exchange(&[r as u8 + 10]).unwrap();
                    [first, second].concat()
                })
            })
            .collect();
        runs.into_iter().map(|t| t.join().unwrap()).collect()
    });
    let want = vec![
        vec![0, 0, 0],
        vec![1, 1, 1],
        vec![2, 2, 2],
        vec![10],
        vec![11],
        vec![12],
    ];
    for rank in &seen {
        assert_eq!(*rank, want);
    }
}

/// Three ranks on threads over one mapped file, two rounds.
#[test]
fn shm_exchange() {
    let path = testkit::scratch(&format!("comm-{}", std::process::id()));
    let seen: Vec<Vec<Vec<u8>>> = std::thread::scope(|s| {
        let runs: Vec<_> = (0..3u32)
            .map(|r| {
                let path = &path;
                s.spawn(move || {
                    let mut shm = Shm::join(path, r, 3, 16).unwrap();
                    let first = shm.exchange(&[r as u8; 3]).unwrap();
                    let second = shm.exchange(&[r as u8 + 10]).unwrap();
                    [first, second].concat()
                })
            })
            .collect();
        runs.into_iter().map(|t| t.join().unwrap()).collect()
    });
    let _ = std::fs::remove_file(&path);
    let want = vec![
        vec![0, 0, 0],
        vec![1, 1, 1],
        vec![2, 2, 2],
        vec![10],
        vec![11],
        vec![12],
    ];
    for rank in &seen {
        assert_eq!(*rank, want);
    }
}
