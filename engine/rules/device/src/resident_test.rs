use crate::exec::{USER_ARENAS, pack_plan};
use crate::layout::{
    ARENAS, CONV_KV, CONV_LLM, CONV_STATE, Form, Held, Item, Kept, KvTables, Lane, LlmStep, NONE,
    PoolSet, Ref, StateTables, Table,
};
use crate::proxy::Op;
use crate::proxy::wired::Wired;
use crate::resident::{
    ControlPlaces, ConventionsPlan, KvPlan, LanePlan, LocalLink, Plan, PoolPlan, USER,
    layout_control,
};
use std::collections::HashMap;
use tessera_graph::DimName::Context;
use tessera_lower::{
    Collective, Conventions, Elem, Extent, FormSpec, HeldSpec, ItemData, ItemKind, ItemSpec,
    KeptSpec, LaneSpec, LlmSpec, Plan as LowerPlan, PoolSpec, ResidentSpec, Slot,
};
use tessera_run::ResidentCall;

fn build_control(plan: &Plan, rank: u32, ranks: u32) -> (Vec<u8>, ControlPlaces) {
    let pools = plan.proxy_pools(rank);
    let groups = plan.pools.iter().map(|p| p.group.clone()).collect();
    let wired = Wired::new(LocalLink, &pools, &plan.bytes(), groups).unwrap();
    let places: Vec<_> = (0..plan.pools.len())
        .map(|p| wired.place(p).unwrap())
        .collect();
    let mut arenas = [0u64; ARENAS];
    arenas[USER as usize..][..plan.arenas.len()].copy_from_slice(&plan.arenas);
    layout_control(plan, [rank, ranks], arenas, &places).unwrap()
}

fn read<T: Copy>(raw: &[u8], at: u64) -> T {
    let at = at as usize;
    assert!(at + size_of::<T>() <= raw.len());
    unsafe { std::ptr::read_unaligned(raw.as_ptr().add(at).cast()) }
}

#[test]
fn control_layouts() {
    let item = Item {
        at: Ref::at(USER, 64),
        kind: 0,
        elem: 5,
        dims: 2,
        shape: [128, 64, 1, 1],
        share: [0, 0, 128, 128],
        form: 0,
        forms: 1,
        unused: 0,
    };
    let form = Form {
        codes: Ref::at(USER, 64),
        suh: Ref::at(USER, 1024),
        svh: Ref::at(USER, 2048),
        kind: 42,
        rows: 128,
        cols: 64,
        unused: 0,
    };
    let llm = LlmStep {
        rows: 4,
        out_rows: 2,
        vocab: 32000,
        unused: 0,
        token: Ref::at(USER, 0),
        slot: Ref::at(USER, 16),
        position: Ref::at(USER, 32),
        out_row: Ref::NONE,
        temperature: Ref::NONE,
        top_k: Ref::NONE,
        seed: Ref::NONE,
        logits: Ref::NONE,
        sampled: Ref::at(USER, 48),
    };
    let kv = KvPlan {
        kept: Kept {
            base: Ref::at(USER, 4096),
            host: Ref::NONE,
            pages: Ref::NONE,
            len: Ref::NONE,
            pages_per_slot: 1,
            page_rows: 16,
            row_bytes: 256,
            elem: 0,
        },
        slots: 2,
        dense_pages: true,
    };
    let held = Held {
        base: Ref::at(USER, 8192),
        fresh: Ref::NONE,
        slot_bytes: 512,
        elem: 0,
    };
    let single = Plan {
        items: vec![item],
        forms: vec![form],
        conventions: ConventionsPlan {
            kv: vec![kv],
            llm: Some(llm),
            state: vec![held],
        },
        arenas: vec![0x1000],
        threads: 256,
        share: 1,
        shared_bytes: 49152,
        ..Plan::default()
    };
    let (raw, places) = build_control(&single, 0, 1);
    let table: Table = read(&raw, 0);
    assert_eq!(
        (table.rank, table.ranks, table.items_len, table.forms_len),
        (0, 1, 1, 1)
    );
    assert_eq!((table.lanes_len, table.pools.arena), (0, NONE));
    assert_eq!(table.arenas[USER as usize], 0x1000);
    let got_form: Form = read(&raw, table.forms.offset);
    assert_eq!(
        (got_form.kind, got_form.suh.offset, got_form.svh.offset),
        (42, 1024, 2048)
    );
    let got_llm: LlmStep = read(&raw, table.conventions[CONV_LLM].at.offset);
    assert_eq!(
        (got_llm.rows, got_llm.out_rows, got_llm.vocab),
        (4, 2, 32000)
    );
    let kv_hdr: KvTables = read(&raw, table.conventions[CONV_KV].at.offset);
    let got_kv: Kept = read(&raw, kv_hdr.tables.offset);
    let dense: [u32; 2] = read(&raw, got_kv.pages.offset);
    assert_eq!((kv_hdr.count, dense, places.kv_lens[0].1), (1, [0, 1], 2));
    let st_hdr: StateTables = read(&raw, table.conventions[CONV_STATE].at.offset);
    let got_st: Held = read(&raw, st_hdr.tables.offset);
    assert_eq!((st_hdr.count, got_st.slot_bytes), (1, 512));

    let multi = Plan {
        pools: vec![PoolPlan {
            depth: 2,
            one_way: false,
            group: vec![0, 1],
            send_bytes: 1024,
            recv_bytes: 1024,
        }],
        lanes: vec![LanePlan {
            op: Op::Sum,
            elem: 0,
            pool: 0,
            block: 0,
        }],
        items: vec![Item { kind: 1, ..item }],
        arenas: vec![0x2000],
        threads: 256,
        share: 1,
        ..Plan::default()
    };
    let (raw, _) = build_control(&multi, 1, 2);
    let t2: Table = read(&raw, 0);
    let set: PoolSet = read(&raw, t2.pools.offset);
    let lane: Lane = read(&raw, t2.lanes.offset);
    assert_eq!(
        (t2.rank, t2.ranks, set.count, lane.group_rank),
        (1, 2, 1, 1)
    );
}

#[test]
fn pack_plan_arenas() {
    let plan = LowerPlan {
        ranks: 2,
        rank: 0,
        buffers: vec![4096; 3],
        ..LowerPlan::default()
    };
    let s0 = Slot::floats(0, 64);
    let s1 = Slot { offset: 64, ..s0 };
    let spec = ResidentSpec {
        threads: 256,
        share: 1,
        shared_bytes: 8192,
        pools: vec![PoolSpec {
            depth: 2,
            one_way: false,
            group: Vec::new(),
            send_bytes: 512,
            recv_bytes: 512,
        }],
        lanes: vec![LaneSpec {
            op: Collective::Sum,
            elem: Elem::F32,
            pool: 0,
            block: 0,
        }],
        items: vec![
            ItemSpec {
                data: ItemData::Slot(s0),
                kind: ItemKind::Input,
                elem: Elem::F32,
                dims: 1,
                shape: [64, 1, 1, 1],
                share: [0; 4],
                form: NONE,
                forms: 0,
            },
            ItemSpec {
                data: ItemData::Slot(s1),
                kind: ItemKind::Output,
                elem: Elem::F32,
                dims: 1,
                shape: [64, 1, 1, 1],
                share: [0; 4],
                form: NONE,
                forms: 0,
            },
            ItemSpec {
                data: ItemData::Scratch(20),
                kind: ItemKind::Scratch,
                elem: Elem::Bytes,
                dims: 1,
                shape: [20, 1, 1, 1],
                share: [0; 4],
                form: NONE,
                forms: 0,
            },
            ItemSpec {
                data: ItemData::Scratch(32),
                kind: ItemKind::Scratch,
                elem: Elem::Bytes,
                dims: 1,
                shape: [32, 1, 1, 1],
                share: [0; 4],
                form: NONE,
                forms: 0,
            },
        ],
        forms: vec![FormSpec {
            codes: s0,
            suh: Some(s1),
            svh: None,
            kind: 7,
            rows: 8,
            cols: 8,
        }],
        values: Vec::new(),
        conventions: Conventions {
            kv: vec![KeptSpec {
                index: 0,
                base: Slot::floats(1, 256),
                pages: None,
                pages_per_slot: 1,
                page_rows: 16,
                row_bytes: 64,
                elem: Elem::F32,
                slots: 2,
            }],
            llm: Some(LlmSpec {
                rows: Extent::Fixed(4),
                out_rows: Extent::Fixed(1),
                vocab: 100,
                token: Some(s0),
                slot: None,
                position: None,
                out_row: None,
                temperature: None,
                top_k: None,
                seed: None,
                logits: None,
                sampled: Some(s1),
            }),
            state: vec![HeldSpec {
                base: Slot::floats(2, 64),
                fresh: None,
                slot_bytes: 128,
                elem: Elem::F32,
            }],
        },
    };
    let buffers = [0xA000, 0xA000, 0xB000];
    let bound = HashMap::from([(Context, 9)]);
    let call = ResidentCall {
        step: 0,
        plan: &plan,
        buffers: &buffers,
        bound: &bound,
        kv_lens: &[5],
    };
    let packed = pack_plan(&call, &spec, 0xC000).unwrap();
    // Buffers 0 and 1 share arena 0xA000.
    assert_eq!(packed.plan.arenas, [0xA000, 0xC000, 0xB000]);
    assert_eq!(
        (
            packed.plan.items[2].at.offset,
            packed.plan.items[3].at.offset
        ),
        (0, 32)
    );
    assert_eq!(packed.plan.pools[0].group, [0, 1]);
    assert_eq!(packed.kv_lens[0], [5, 9]);

    let many_items: Vec<ItemSpec> = (0..=USER_ARENAS as u32)
        .map(|b| ItemSpec {
            data: ItemData::Slot(Slot::floats(b, 4)),
            kind: ItemKind::Input,
            elem: Elem::F32,
            dims: 1,
            shape: [4, 1, 1, 1],
            share: [0; 4],
            form: NONE,
            forms: 0,
        })
        .collect();
    let too_many = ResidentSpec {
        items: many_items,
        forms: Vec::new(),
        conventions: Conventions::default(),
        ..spec
    };
    let distinct: Vec<u64> = (1..=USER_ARENAS as u64 + 1).map(|i| i << 12).collect();
    let bad = ResidentCall {
        buffers: &distinct,
        ..call
    };
    let err = pack_plan(&bad, &too_many, 0).unwrap_err().to_string();
    assert!(err.contains("unique user arenas"), "{err}");
}
