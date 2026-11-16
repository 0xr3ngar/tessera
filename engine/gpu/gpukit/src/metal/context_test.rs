use super::Context;

const SRC: &str = "\
#include <metal_stdlib>
using namespace metal;
kernel void pc_k0(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 0.0f; }
kernel void pc_k1(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 1.0f; }
kernel void pc_k2(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 2.0f; }
kernel void pc_k3(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 3.0f; }
";

/// Concurrent compiles share one set of pipelines.
#[test]
fn compile_race_free() {
    if Context::new().is_err() {
        eprintln!("skip: no Metal device");
        return;
    }
    let entries = ["pc_k0", "pc_k1", "pc_k2", "pc_k3"];
    let handles: Vec<_> = (0..8)
        .map(|_| {
            std::thread::spawn(move || {
                let ctx = Context::new().expect("device");
                for _ in 0..4 {
                    for e in entries {
                        ctx.compile_kernel(SRC, e).expect("pipeline compile");
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("worker panicked: a pipeline race?");
    }
}

/// A file range wraps where aligned, else reads in.
#[test]
fn fill_from_file() {
    use crate::device::DeviceBuffer;
    use crate::fill::{Fill, Source};
    let Ok(ctx) = Context::new() else {
        return eprintln!("skip: no Metal device");
    };
    let bytes: Vec<u8> = (0..40_000u32).map(|i| (i % 251) as u8).collect();
    let path = testkit::stage("fill", &bytes);
    let file = mem::FileRange::new(std::fs::File::open(&path).unwrap()).unwrap();
    for (at, want) in [(16, Fill::Wrap), (20, Fill::Read)] {
        let range = file.sub(at, 30_000).unwrap();
        let (buffer, fill) = DeviceBuffer::fill(&ctx, Source::File(&range), 16).unwrap();
        assert_eq!(fill, want);
        let mut out = vec![0; 30_000];
        buffer.read_bytes(&mut out).unwrap();
        assert_eq!(out, bytes[at..at + 30_000]);
    }
    std::fs::remove_file(path).unwrap();
}
