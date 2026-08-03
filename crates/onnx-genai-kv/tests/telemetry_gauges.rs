// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Telemetry gauges must agree with the pool they mirror.
//!
//! The gauges are maintained incrementally, at the sites that change a page's
//! reference count, rather than recomputed by walking the pool. That is what
//! makes them cheap enough for the decode path, and it is also the one way they
//! can be wrong: a mutation site that forgets to report leaves the gauge
//! silently drifted, and every later reading is off by that much with no
//! symptom at the point of the bug.
//!
//! Each test here therefore drives real allocate / retain / free traffic and
//! compares the incrementally-maintained gauges against
//! [`PageTable::live_page_counts`], which walks the pool and is the ground
//! truth. Asserting only against expected constants would not catch drift,
//! because a test written against the same wrong mental model passes.

use onnx_genai_kv::page_table::PageTable;
use onnx_genai_kv::{Device, KvTelemetry};
use std::sync::Arc;

const GPU: Device = Device::Gpu(0);

/// Assert the incremental gauges match a full walk of the pool.
fn assert_gauges_match_truth(table: &PageTable, telemetry: &KvTelemetry, context: &str) {
    let (in_use, shared) = table.live_page_counts();
    let snapshot = telemetry.snapshot();
    assert_eq!(
        snapshot.pages_in_use, in_use,
        "{context}: pages_in_use drifted from the pool"
    );
    assert_eq!(
        snapshot.pages_shared, shared,
        "{context}: pages_shared drifted from the pool"
    );
}

#[test]
fn gauges_track_allocate_retain_and_free() {
    let mut table = PageTable::new(16, 8);
    let telemetry = Arc::new(KvTelemetry::default());
    table.attach_telemetry(Arc::clone(&telemetry));

    assert_gauges_match_truth(&table, &telemetry, "after attach");

    let a = table.allocate(GPU).expect("pool has capacity");
    assert_gauges_match_truth(&table, &telemetry, "after first allocate");
    assert_eq!(telemetry.snapshot().pages_in_use, 1);
    assert_eq!(telemetry.snapshot().pages_shared, 0);

    let b = table.allocate(GPU).expect("pool has capacity");
    assert_gauges_match_truth(&table, &telemetry, "after second allocate");
    assert_eq!(telemetry.snapshot().pages_in_use, 2);

    // Sharing a page must move `pages_shared` without moving `pages_in_use`.
    assert!(table.retain(a));
    assert_gauges_match_truth(&table, &telemetry, "after retain");
    assert_eq!(telemetry.snapshot().pages_in_use, 2);
    assert_eq!(telemetry.snapshot().pages_shared, 1);

    // Dropping one of two references un-shares but keeps the page live.
    table.free(a);
    assert_gauges_match_truth(&table, &telemetry, "after first free of shared page");
    assert_eq!(telemetry.snapshot().pages_in_use, 2);
    assert_eq!(telemetry.snapshot().pages_shared, 0);

    table.free(a);
    table.free(b);
    assert_gauges_match_truth(&table, &telemetry, "after freeing everything");
    assert_eq!(telemetry.snapshot().pages_in_use, 0);
    assert_eq!(telemetry.snapshot().pages_shared, 0);
}

#[test]
fn gauges_survive_interleaved_traffic() {
    // A deterministic but non-trivial mix, because the edge-triggered updates
    // are exactly the kind of logic that holds for a simple sequence and breaks
    // once retains and frees interleave across several pages.
    let mut table = PageTable::new(8, 32);
    let telemetry = Arc::new(KvTelemetry::default());
    table.attach_telemetry(Arc::clone(&telemetry));

    let mut live: Vec<u32> = Vec::new();
    for step in 0..60u32 {
        match step % 4 {
            0 | 1 => {
                if let Some(id) = table.allocate(GPU) {
                    live.push(id);
                }
            }
            2 => {
                if let Some(&id) = live.get((step as usize / 3) % live.len().max(1)) {
                    table.retain(id);
                    live.push(id);
                }
            }
            _ => {
                if let Some(id) = live.pop() {
                    table.free(id);
                }
            }
        }
        assert_gauges_match_truth(&table, &telemetry, &format!("step {step}"));
    }

    // Drain, and confirm the gauges land back at zero rather than drifting.
    while let Some(id) = live.pop() {
        table.free(id);
    }
    assert_gauges_match_truth(&table, &telemetry, "after drain");
    assert_eq!(telemetry.snapshot().pages_in_use, 0);
    assert_eq!(telemetry.snapshot().pages_shared, 0);
}

#[test]
fn counters_and_geometry_reach_the_snapshot() {
    let mut table = PageTable::new(16, 4);
    let telemetry = Arc::new(KvTelemetry::default());
    table.attach_telemetry(Arc::clone(&telemetry));

    let snapshot = telemetry.snapshot();
    assert_eq!(snapshot.page_size, 16, "page size must reach the mirror");
    assert_eq!(snapshot.hot_capacity, 4, "capacity must reach the mirror");

    let mut ids = Vec::new();
    for _ in 0..4 {
        ids.push(table.allocate(GPU).expect("pool has capacity"));
    }
    assert_eq!(telemetry.snapshot().allocations, 4);

    for id in ids {
        table.free(id);
    }
    assert_eq!(telemetry.snapshot().frees, 4);
}

#[test]
fn over_capacity_pressure_publishes_evictions_not_failures() {
    // The pool copes with over-capacity demand by demoting a hot page to the
    // cold tier, not by refusing. Asserting a failure here would have encoded a
    // wrong mental model; what must be published is the eviction.
    let mut table = PageTable::new(16, 2);
    let telemetry = Arc::new(KvTelemetry::default());
    table.attach_telemetry(Arc::clone(&telemetry));

    for _ in 0..8 {
        table
            .allocate(GPU)
            .expect("pool demotes rather than refusing");
    }

    let snapshot = telemetry.snapshot();
    assert_eq!(snapshot.allocations, 8);
    assert_eq!(
        snapshot.allocation_failures, 0,
        "demotion is not a failure: {snapshot:?}"
    );
    assert!(
        snapshot.hot_evictions > 0,
        "sustained over-capacity demand must publish evictions, got {snapshot:?}"
    );
    assert_gauges_match_truth(&table, &telemetry, "after over-capacity pressure");
}

#[test]
fn a_refused_allocation_is_published_as_a_failure() {
    // Growth is a GPU-tier path, so a request for a tier with no free pages is
    // refused outright. That is the one signal a caller cannot infer from the
    // other counters, so it must reach the mirror.
    let mut table = PageTable::new(16, 2);
    let telemetry = Arc::new(KvTelemetry::default());
    table.attach_telemetry(Arc::clone(&telemetry));

    assert!(
        table.allocate(Device::Cpu).is_none(),
        "a tier with no free pages and no growth path must refuse"
    );

    let snapshot = telemetry.snapshot();
    assert_eq!(
        snapshot.allocation_failures, 1,
        "a refused allocation must be published, got {snapshot:?}"
    );
    assert_eq!(snapshot.allocations, 0);
    assert_gauges_match_truth(&table, &telemetry, "after a refused allocation");
}

#[test]
fn attaching_to_a_warm_pool_seeds_from_its_real_state() {
    // Attaching late must not publish a zero that was never true.
    let mut table = PageTable::new(16, 8);
    let a = table.allocate(GPU).expect("pool has capacity");
    let _b = table.allocate(GPU).expect("pool has capacity");
    table.retain(a);

    let telemetry = Arc::new(KvTelemetry::default());
    table.attach_telemetry(Arc::clone(&telemetry));

    assert_gauges_match_truth(&table, &telemetry, "immediately after late attach");
    let snapshot = telemetry.snapshot();
    assert_eq!(snapshot.pages_in_use, 2);
    assert_eq!(snapshot.pages_shared, 1);
    assert_eq!(
        snapshot.allocations, 2,
        "cumulative counters must be seeded too, not just the gauges"
    );
}

#[test]
fn two_pools_do_not_move_each_other_s_gauges() {
    // Two independent pools sharing one set of gauges would make every number
    // the sum of two unrelated things, which is worse than publishing nothing.
    //
    // This used to be phrased against `PageTable::clone`, which is how two
    // pools could accidentally end up attached to one telemetry handle. That
    // route is gone -- a pool is no longer `Clone`, because copying one would
    // duplicate every page's storage while leaving the memory grant behind.
    // The property still matters for pools built separately, so it is pinned
    // that way instead.
    let mut table = PageTable::new(16, 8);
    let telemetry = Arc::new(KvTelemetry::default());
    table.attach_telemetry(Arc::clone(&telemetry));
    let _ = table.allocate(GPU).expect("pool has capacity");

    let mut other = PageTable::new(16, 8);
    other.attach_telemetry(Arc::new(KvTelemetry::default()));
    let before = telemetry.snapshot();
    let _ = other.allocate(GPU).expect("pool has capacity");
    let after = telemetry.snapshot();

    assert_eq!(
        before, after,
        "allocating on an unrelated pool moved this pool's gauges"
    );
}
