//! **Pure-CPU oracle self-checks** for the `com.microsoft::MatMulNBits` int4
//! parity gate.
//!
//! These tests issue **no CUDA calls** — they validate the device-free numerics
//! machinery (the `f64` dequant→GEMM oracle, an independent `f32` reference, and
//! the justified [`Envelope`]/[`ParityReport`] tolerance model) that the GPU gate
//! in `matmul_nbits_marlin_numerics.rs` measures a real int4 kernel against.
//!
//! They live in their own integration-test target, split out of the GPU
//! numerics target, precisely so the CUDA test-honesty checker
//! (`.github/scripts/verify_cuda_test_honesty.py`) can treat
//! `matmul_nbits_marlin_numerics` as purely-CUDA (every test `ignore`d without
//! `gpu-tests`) while these oracle probes legitimately run and pass on the
//! CPU-only lane. The shared machinery lives in `marlin_numerics/mod.rs`; this
//! target is listed in the checker's `ALWAYS_RUN` set as a genuine CPU-only
//! probe. See #1177.

mod marlin_numerics;

use marlin_numerics::{Envelope, GROUP_SIZES, Int4Problem, f32_dequant_reference};

#[test]
fn oracle_matches_independent_reference_symmetric() {
    // Small dims keep this a fast CPU-only check while still exercising multiple
    // blocks, several columns, and M>1.
    for &block_size in GROUP_SIZES {
        let p = Int4Problem::new(4, 256, 12, block_size, true, false, 0xA53F_0001);
        let oracle = p.f64_oracle();
        let reference = f32_dequant_reference(&p);
        for (o, r) in oracle.iter().zip(reference.iter()) {
            // f64 oracle vs f32 reference: agreement to the f32 reduction floor
            // proves the packed encoding and the decoded twins describe the same
            // weights (a wiring bug would diverge by whole quanta, not ULPs).
            assert!(
                (o - *r as f64).abs() <= o.abs().max(1.0) * 1e-4,
                "block_size={block_size}: oracle={o:e} reference={r:e}"
            );
        }
    }
}

#[test]
fn oracle_matches_independent_reference_asymmetric() {
    for &block_size in GROUP_SIZES {
        let p = Int4Problem::new(3, 128, 9, block_size, false, true, 0xA53F_0002);
        let oracle = p.f64_oracle();
        let reference = f32_dequant_reference(&p);
        for (o, r) in oracle.iter().zip(reference.iter()) {
            assert!(
                (o - *r as f64).abs() <= o.abs().max(1.0) * 1e-4,
                "asymmetric block_size={block_size}: oracle={o:e} reference={r:e}"
            );
        }
    }
}

#[test]
fn oracle_is_exact_on_a_hand_checkable_case() {
    // K = block_size = 16, N = 1, M = 1, symmetric (zp = 8). Recompute the single
    // output from the decoded codes/scale and require bit-for-bit agreement with
    // the oracle's own f64 accumulation (same association) — this pins the oracle
    // arithmetic, not just its self-consistency.
    let p = Int4Problem::new(1, 16, 1, 16, false, false, 0xA53F_0003);
    let mut expected = 0.0f64;
    for depth in 0..16 {
        let code = p.quant[depth] as i32 - 8;
        expected += p.activation_ref[depth] as f64 * code as f64 * p.scale_ref[0] as f64;
    }
    assert_eq!(p.f64_oracle()[0], expected);
}

#[test]
fn envelope_scales_with_output_magnitude_and_has_a_floor() {
    let big = Envelope::for_output(1000.0);
    assert!(
        (big.abs_bound - 4.0).abs() < 1e-9,
        "abs bound tracks max_out * 4e-3"
    );
    assert_eq!(big.rel_bound, 5e-2);
    let tiny = Envelope::for_output(0.0);
    assert!(
        (tiny.abs_bound - 4e-3).abs() < 1e-12,
        "abs bound keeps a 4e-3 floor"
    );
}

#[test]
fn parity_flags_a_perturbed_candidate() {
    let p = Int4Problem::new(2, 64, 8, 32, true, false, 0xA53F_0004);
    let oracle = p.f64_oracle();
    let good: Vec<f32> = oracle.iter().map(|&o| o as f32).collect();
    let clean = p.parity(&good);
    assert!(clean.all_finite);
    let env = Envelope::for_output(clean.max_out);
    assert!(
        clean.max_abs <= env.abs_bound,
        "an f32-cast oracle must pass its own gate"
    );

    // Inject a gross error into one element; the gate must catch it.
    let mut bad = good.clone();
    bad[0] += (clean.max_out as f32).max(1.0);
    let dirty = p.parity(&bad);
    let env = Envelope::for_output(dirty.max_out);
    assert!(
        dirty.max_abs > env.abs_bound || dirty.max_rel > env.rel_bound,
        "a corrupted candidate must fail the gate"
    );
}
