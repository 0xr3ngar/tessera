use crate::{Arg, Ctx, compile, failure};
use sensai_codegen::Precision;
use sensai_codegen::form::F32_FORM;
use sensai_codegen::prelude::*;
use sensai_host::exec::{self, ArgVal, Grid};

/// Threads past `n` read nothing: each guard decides first.
#[kernel]
fn guarded(x: Buf<F32>, mut out: BufMut<F32>, n: u32) {
    let mut v = 0.0f32;
    if gid() < n && x[gid() as u64] > 0.0 {
        v = 1.0;
    }
    if gid() >= n || x[gid() as u64] < 0.0 {
        v += 2.0;
    }
    out[gid() as u64] = v;
}

fn floats(b: &[u8]) -> Vec<f32> {
    let word = |c: &[u8]| f32::from_le_bytes(c.try_into().unwrap_or_default());
    b.chunks(4).map(word).collect()
}

#[test]
fn guards_short_circuit() {
    let body = guarded_ir();
    let x: Vec<u8> = [1.0f32, -1.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let want = [1.0f32, 2.0, 2.0, 2.0];
    // The host executor, then the compiled body.
    let (mut xs, mut out) = (x.clone(), vec![0u8; 16]);
    let buf = |at| ArgVal::Buf {
        at,
        skip: 0,
        form: F32_FORM,
    };
    let grid = Grid {
        groups: 1,
        threads: 4,
    };
    let args = [buf(0), buf(1), ArgVal::U32(2)];
    exec::run(
        &body,
        &args,
        &mut [&mut xs, &mut out],
        grid,
        LANES,
        Precision::EXACT,
    );
    assert_eq!(floats(&out), want);
    let (mut xs, mut out) = (x, vec![0u8; 16]);
    let bufs = [&mut xs, &mut out].map(|b| crate::Buf {
        ptr: b.as_mut_ptr(),
        len: b.len(),
    });
    let at = |buf| Arg {
        buf,
        ..Arg::default()
    };
    let args = [
        at(0),
        at(1),
        Arg {
            bits: 2,
            ..Arg::default()
        },
    ];
    let ctx = Ctx {
        bufs: bufs.as_ptr(),
        args: args.as_ptr(),
        threads: 4,
        lanes: LANES,
        precision: &Precision::EXACT,
    };
    let entry = compile(&body).unwrap();
    // SAFETY: the entry was compiled for this body's parameters.
    unsafe { entry(&ctx, 0, 1) };
    assert_eq!((failure(), floats(&out)), (None, want.to_vec()));
}
