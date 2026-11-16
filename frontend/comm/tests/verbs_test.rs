//! Ranks on threads over RDMA loopback, when `TESSERA_VERBS` names ports.

use engine::comm::{Fabric, Port, Transport, Verbs};
use frontend_comm::{ports, sockets};
use std::net::TcpListener;
use std::sync::Mutex;

/// The ports `TESSERA_VERBS` names, or none to skip.
fn lanes() -> Option<Vec<Port>> {
    Some(ports(&frontend_settings::read().verbs?).unwrap())
}

/// `ranks` ranks on threads over TCP, three rounds each.
fn rounds(fabric: &Fabric, ranks: u32) {
    let listeners: Vec<TcpListener> = (0..ranks)
        .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    let addrs: Vec<_> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
    let listeners: Vec<_> = listeners.into_iter().map(Mutex::new).collect();
    let seen: Vec<Vec<Vec<u8>>> = std::thread::scope(|s| {
        let runs: Vec<_> = (0..ranks)
            .map(|r| {
                let (listeners, addrs, fabric) = (&listeners, &addrs, fabric);
                s.spawn(move || {
                    let listener = listeners[r as usize].lock().unwrap().try_clone().unwrap();
                    let mut boot = sockets(r, listener, addrs).unwrap();
                    let mut t = Verbs::connect(r, ranks, fabric, 1 << 16, &mut boot, None).unwrap();
                    assert_eq!((t.rank(), t.ranks()), (r, ranks));
                    let mut out = t.exchange(&[r as u8; 3]).unwrap();
                    // Lengths differ by rank, and one is empty.
                    out.extend(t.exchange(&vec![r as u8 + 10; 5 * r as usize]).unwrap());
                    out.extend(t.exchange(&vec![r as u8 + 20; 40_000]).unwrap());
                    out
                })
            })
            .collect();
        runs.into_iter().map(|t| t.join().unwrap()).collect()
    });
    let mut want = Vec::new();
    want.extend((0..ranks).map(|r| vec![r as u8; 3]));
    want.extend((0..ranks).map(|r| vec![r as u8 + 10; 5 * r as usize]));
    want.extend((0..ranks).map(|r| vec![r as u8 + 20; 40_000]));
    for rank in &seen {
        assert_eq!(*rank, want);
    }
}

#[test]
fn mesh_loopback() {
    let Some(lanes) = lanes() else {
        return;
    };
    let fabric = Fabric::Mesh(lanes);
    rounds(&fabric, 2);
    rounds(&fabric, 3);
}

/// At four, the rank between relays for the one beyond.
#[test]
fn ring_loopback() {
    let Some(lanes) = lanes() else {
        return;
    };
    let fabric = Fabric::Ring {
        prev: lanes.clone(),
        next: lanes,
    };
    rounds(&fabric, 2);
    rounds(&fabric, 3);
    rounds(&fabric, 4);
}
