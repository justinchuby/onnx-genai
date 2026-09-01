//! CPU-only half of the #1810 Slice 6 expert-route telemetry probe.
//!
//! The companion `expert_route_telemetry_probe_gpu.rs` holds the seven tests
//! that need a CUDA device. This target holds the one that does not, and it
//! lives here rather than there for a specific reason.
//!
//! `.github/scripts/verify_cuda_test_honesty.py` requires every test in a
//! `_gpu` target to be ignored on a host with no CUDA device -- a `_gpu` test
//! that passes is indistinguishable from a `_gpu` test that silently did
//! nothing, which is the whole failure mode the checker exists to catch. So the
//! CPU-only test was reported as "executed without gpu-tests" and reddened the
//! `CUDA compile (Linux x86_64)` lane.
//!
//! `#[ignore]` would have cleared the checker while destroying the test: its
//! stated purpose is to run "without a GPU so the reference cannot silently
//! rot", and an ignored test runs nowhere. Target naming is the escape hatch
//! the checker documents for exactly this case -- a name without the `_gpu`
//! suffix is not treated as a CUDA target.
//!
//! Being a non-`_gpu` target in this crate, it runs in no lane by default: the
//! honesty script skips it and `.github/scripts/workspace_test_packages.py`
//! deny-lists `onnx-runtime-ep-cuda` from the offline lanes. It therefore has
//! an explicit step in `.github/workflows/ci.yml`, alongside the ones already
//! carried by `capture_sync_contract` and `content_preserving_transition`.

use std::collections::HashSet;

mod expert_route_oracle;
use expert_route_oracle::{
    Decision, H_DEVICE, H_EPOCH, H_OVERFLOW, H_POISON, H_REQUEST, HEADER_LEN, consume_and_validate,
    cpu_bitmap, cpu_dedup, synth_routes,
};

/// CPU-only: the oracle and the boundary validator are self-consistent. Runs
/// without a GPU so the reference cannot silently rot.
#[test]
fn cpu_oracle_and_validator_self_consistent() {
    let num_experts = 256;
    let routes = synth_routes(1, 8, num_experts, 42);
    let (bits, poison) = cpu_bitmap(&routes, num_experts);
    assert!(!poison);
    let set_from_bits: HashSet<i32> = (0..num_experts)
        .filter(|&e| bits[e as usize >> 5] & (1 << (e & 31)) != 0)
        .collect();
    let (distinct, overflow, poison2) = cpu_dedup(&routes, num_experts, num_experts as usize);
    assert!(!overflow && !poison2);
    let distinct_set: HashSet<i32> = distinct.iter().copied().collect();
    assert_eq!(
        set_from_bits, distinct_set,
        "bitmap and dedup must agree on the set"
    );

    // Validator: clean record accepts; each defect fails closed.
    let mut header = vec![0u32; HEADER_LEN];
    header[H_EPOCH] = 5;
    header[H_REQUEST] = 7;
    header[H_DEVICE] = 3;
    header[5] = routes.len() as u32;
    assert!(matches!(
        consume_and_validate(&header, &bits, 5, 7, 3, num_experts as usize),
        Decision::HotSet(_)
    ));
    for (idx, val, label) in [(H_POISON, 1, "poison"), (H_OVERFLOW, 1, "overflow")] {
        let mut bad = header.clone();
        bad[idx] = val;
        assert!(
            matches!(
                consume_and_validate(&bad, &bits, 5, 7, 3, num_experts as usize),
                Decision::WholeBank(_)
            ),
            "{label} must fail closed"
        );
    }
    assert!(
        matches!(
            consume_and_validate(&header, &bits, 5, 999, 3, num_experts as usize),
            Decision::WholeBank(_)
        ),
        "request mismatch"
    );
    assert!(
        matches!(
            consume_and_validate(&header, &bits, 5, 7, 999, num_experts as usize),
            Decision::WholeBank(_)
        ),
        "device mismatch"
    );
    assert!(
        matches!(
            consume_and_validate(&header, &bits, 6, 7, 3, num_experts as usize),
            Decision::WholeBank(_)
        ),
        "stale epoch"
    );
    println!(
        "cpu_oracle_and_validator_self_consistent: OK ({} distinct experts)",
        distinct.len()
    );
}
