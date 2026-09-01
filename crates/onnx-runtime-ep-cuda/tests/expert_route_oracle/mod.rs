//! Pure-CPU reference oracle for the #1810 Slice 6 expert-route telemetry probe.
//!
//! This is a shared module, not a test target: Cargo auto-discovers `tests/*.rs`
//! and `tests/*/main.rs`, so `tests/expert_route_oracle/mod.rs` compiles only as
//! a `mod` of whichever target declares it.
//!
//! It exists so the oracle can be checked on a host with no GPU. Its contents
//! were extracted verbatim from `expert_route_telemetry_probe_gpu.rs`, where the
//! CPU-only self-consistency test could not run honestly: every test in a `_gpu`
//! target must be ignored without a CUDA device, so a test that passed there was
//! reported by `.github/scripts/verify_cuda_test_honesty.py` as
//! "executed without gpu-tests". Silencing it with `#[ignore]` would have
//! satisfied the checker by destroying the property the test was written for --
//! that the reference "cannot silently rot". Relocating the test to
//! `expert_route_telemetry_probe.rs` keeps it running on every CPU lane instead.

#![allow(dead_code)]

use std::collections::HashSet;

// Header word indices (u32 each), device-resident. request/device/epoch fit in
// u32 for the probe; the design specifies u64 epoch/request in production.
pub const H_EPOCH: usize = 0;
pub const H_REQUEST: usize = 1;
pub const H_DEVICE: usize = 2;
pub const H_OVERFLOW: usize = 3;
pub const H_POISON: usize = 4;
pub const H_COUNT: usize = 5;
pub const HEADER_LEN: usize = 6;

pub fn words_for(num_experts: i32) -> usize {
    (num_experts as usize).div_ceil(32)
}

/// Reference bitmap: bit `e` set iff expert `e` is routed at least once. Returns
/// `(bitmap_words, poison)` where poison is true iff any id is out of range.
pub fn cpu_bitmap(routes: &[i32], num_experts: i32) -> (Vec<u32>, bool) {
    let mut bits = vec![0u32; words_for(num_experts)];
    let mut poison = false;
    for &e in routes {
        if e < 0 || e >= num_experts {
            poison = true;
            continue;
        }
        let e = e as usize;
        bits[e >> 5] |= 1u32 << (e & 31);
    }
    (bits, poison)
}

/// Reference dedup: the distinct routed set (in first-seen order) and whether it
/// would overflow a queue of `capacity`.
pub fn cpu_dedup(routes: &[i32], num_experts: i32, capacity: usize) -> (Vec<i32>, bool, bool) {
    let mut seen = HashSet::new();
    let mut distinct = Vec::new();
    let mut poison = false;
    for &e in routes {
        if e < 0 || e >= num_experts {
            poison = true;
            continue;
        }
        if seen.insert(e) {
            distinct.push(e);
        }
    }
    let overflow = distinct.len() > capacity;
    (distinct, overflow, poison)
}

/// The §3 boundary consumer/validator. Runs on the host, at a safe boundary,
/// against a record already copied back. Any failure → whole-bank (fail closed).
#[derive(Debug, PartialEq, Eq)]
pub enum Decision {
    /// Trustworthy: the hot-set is the routed bitmap words.
    HotSet(Vec<u32>),
    /// Fail closed to the whole-bank proof, carrying the reason (design §3 /
    /// design-discipline "carry the reason").
    WholeBank(String),
}

pub fn consume_and_validate(
    header: &[u32],
    bitmap: &[u32],
    expected_epoch: u32,
    expected_request: u32,
    expected_device: u32,
    expected_num_experts: usize,
) -> Decision {
    if header.len() != HEADER_LEN {
        return Decision::WholeBank(format!(
            "header length mismatch: record={} expected {HEADER_LEN}",
            header.len()
        ));
    }
    let expected_words = expected_num_experts.div_ceil(32);
    if bitmap.len() != expected_words {
        return Decision::WholeBank(format!(
            "bitmap length mismatch: record={} expected {expected_words}",
            bitmap.len()
        ));
    }
    if header[H_POISON] != 0 {
        return Decision::WholeBank("poison: out-of-range expert id observed".into());
    }
    if header[H_OVERFLOW] != 0 {
        return Decision::WholeBank("overflow: bounded route counter saturated".into());
    }
    if header[H_DEVICE] != expected_device {
        return Decision::WholeBank(format!(
            "device mismatch: record dev={} expected {expected_device}",
            header[H_DEVICE]
        ));
    }
    if header[H_REQUEST] != expected_request {
        return Decision::WholeBank(format!(
            "request mismatch: record req={} expected {expected_request}",
            header[H_REQUEST]
        ));
    }
    if header[H_EPOCH] != expected_epoch {
        return Decision::WholeBank(format!(
            "epoch mismatch: record epoch={} expected {expected_epoch}",
            header[H_EPOCH]
        ));
    }
    if let Some(last) = bitmap.last() {
        let valid_tail_bits = expected_num_experts % 32;
        if valid_tail_bits != 0 && (*last >> valid_tail_bits) != 0 {
            return Decision::WholeBank(
                "bitmap contains experts outside the armed capacity".into(),
            );
        }
    }
    let Some(unique) = bitmap
        .iter()
        .try_fold(0u32, |total, word| total.checked_add(word.count_ones()))
    else {
        return Decision::WholeBank("unique expert count overflow".into());
    };
    let count = header[H_COUNT];
    if count < unique || (count == 0 && unique != 0) {
        return Decision::WholeBank(format!(
            "route count {count} is inconsistent with {unique} unique selected experts"
        ));
    }
    Decision::HotSet(bitmap.to_vec())
}

/// A deterministic decode-shaped route vector: `rows` tokens × `top_k` experts,
/// values drawn with a cheap LCG, per row distinct (like a real router row).
pub fn synth_routes(rows: usize, top_k: usize, num_experts: i32, seed: u64) -> Vec<i32> {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut out = Vec::with_capacity(rows * top_k);
    for _ in 0..rows {
        let mut picked = HashSet::new();
        while picked.len() < top_k.min(num_experts as usize) {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let e = ((state >> 33) as i32).rem_euclid(num_experts);
            picked.insert(e);
        }
        out.extend(picked);
    }
    out
}
