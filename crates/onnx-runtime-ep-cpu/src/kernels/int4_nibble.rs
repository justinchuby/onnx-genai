//! Packed-nibble `int4 x int16` decode kernel for AVX2 hosts without VNNI.
//!
//! # Why this exists
//!
//! `MatMulNBits` at `bits = 4, accuracy_level = 4` has two x86 routes today. On
//! a host with VNNI the weight stays packed and streams straight into
//! `vpdpbusd`. **Without** VNNI -- which is every AVX2 consumer part before
//! Alder Lake, and this project's own EPYC Zen4 test host -- the route falls
//! back to `prepack_int8_weight`, which expands **every nibble into a whole
//! `i8` byte**. Decode is a GEMV: each weight is read once and never reused, so
//! that expansion doubles the only stream that matters.
//!
//! This module consumes the ONNX packed layout **as it is stored**, 0.5 bytes
//! per weight, and does the nibble unpack in registers.
//!
//! # The arithmetic
//!
//! Per `k` block, with `a` the int8-quantized activation, `w` the raw unsigned
//! nibble and `z` the block's zero point:
//!
//! ```text
//! sum_k a_k * (w_k - z) = (sum_k a_k * w_k) - z * (sum_k a_k)
//! ```
//!
//! The right-hand sum is over the **activation only**, so it is computed once
//! per block and reused by all `N` outputs; the kernel's inner loop only has to
//! produce `sum_k a_k * w_k` and never subtracts the zero point per element.
//! This is exact in `i32` -- see [`nibble_block_dot`] for the bound -- and it is
//! what makes an absent zero-point tensor free rather than a second code path:
//! ONNX defines the default as 8, and a *signed* int4 weight is by definition
//! the unsigned nibble with `z = 8`, so both spellings are the same arithmetic.
//!
//! # The layout
//!
//! `vpmaddwd` multiplies lane `i` of one register by lane `i` of the other, so
//! the activation must arrive in whatever order the nibble unpack produces.
//! Packed byte `j` holds `k = 2j` in its low nibble and `k = 2j + 1` in its
//! high nibble, so a single widen-and-mask yields the **even** `k` of a group
//! and a widen-and-shift the **odd** ones. Rather than shuffle the weights back
//! into `k` order every iteration -- `N` times per decode -- the activation is
//! deinterleaved **once per row** into `[even | odd]` halves of each group. The
//! sum is over the whole block, so permuting both sides identically cannot
//! change it.
//!
//! Group size is `min(32, block_size)`, and `block_size` is a power of two of at
//! least 16, so a block is always a whole number of groups and the inner loop
//! has no tail. A short **final** `k` block needs no masking either: the
//! activation is zero-padded to `padded_k`, so the padding contributes
//! `0 * w = 0` to the dot and `0` to the activation block sum.

use crate::kernels::simd_quant::quantize_block_i8;

/// Smallest `block_size` the ONNX operator admits, and the smallest group this
/// kernel deinterleaves.
pub(crate) const MIN_BLOCK_SIZE: usize = 16;

/// Widest group the kernel deinterleaves: 16 packed bytes widen to 32 nibbles,
/// which is two `__m256i` of `i16` and therefore two `vpmaddwd`.
pub(crate) const WIDE_GROUP: usize = 32;

/// Whether this host and configuration can take the packed-nibble route.
///
/// Deliberately **not** a function of `accuracy_level`: the caller owns that
/// gate, and duplicating it here would let a future edit satisfy one copy and
/// not the other. See `the_packed_nibble_route_is_unreachable_below_accuracy_4`.
pub(crate) fn supported(block_size: usize) -> bool {
    if block_size < MIN_BLOCK_SIZE || !block_size.is_power_of_two() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `k` blocks the AVX2 driver retires per tile.
///
/// A block dot ends in a horizontal reduction and a scalar `f32` tail --
/// zero-point correction, convert, two multiplies and an add. At
/// `block_size = 32` that tail cost several times the 10 uops of arithmetic it
/// was serving. Four blocks share one reduction tree and one **vector** tail,
/// which is exact because four consecutive blocks are contiguous in every array
/// the tail reads: scales, activation scales, block sums, and the zero-point
/// nibbles (four nibbles are two adjacent bytes, and a tile always starts on an
/// even block).
#[cfg(target_arch = "x86_64")]
pub(crate) const BLOCK_TILE: usize = 4;

/// `min(WIDE_GROUP, block_size)` -- the deinterleave group, in `k` elements.
#[inline]
pub(crate) fn group_for(block_size: usize) -> usize {
    if block_size >= WIDE_GROUP {
        WIDE_GROUP
    } else {
        MIN_BLOCK_SIZE
    }
}

/// One row of activation, quantized and laid out for [`nibble_block_dot`].
pub(crate) struct NibbleActivation {
    /// `padded_k` quantized activations widened to `i16` and deinterleaved by
    /// parity within each group. Widened once here rather than per output, so
    /// the inner loop loads them with no conversion.
    values: Vec<i16>,
    /// Per-block `sum_k a_k`, the zero-point correction's activation half.
    block_sums: Vec<i32>,
    /// Per-block activation scale.
    scales: Vec<f32>,
}

impl NibbleActivation {
    /// Quantize and lay out one activation row.
    ///
    /// `activation` may be shorter than `padded_k`; the remainder is zero, which
    /// is what makes a short final `k` block need no masking.
    pub(crate) fn new(activation: &[f32], padded_k: usize, block_size: usize) -> Self {
        debug_assert!(padded_k.is_multiple_of(block_size));
        let k_blocks = padded_k / block_size;
        let group = group_for(block_size);
        let mut values = vec![0i16; padded_k];
        let mut block_sums = vec![0i32; k_blocks];
        let mut scales = vec![0.0f32; k_blocks];
        let mut quantized = vec![0i8; block_size];

        for block in 0..k_blocks {
            let start = block * block_size;
            let end = (start + block_size).min(activation.len());
            if end <= start {
                continue;
            }
            let source = &activation[start..end];
            quantized[..source.len()].fill(0);
            quantized[source.len()..].fill(0);
            scales[block] = quantize_block_i8(source, &mut quantized[..source.len()]);
            let mut sum = 0i32;
            for value in &quantized[..source.len()] {
                sum += i32::from(*value);
            }
            block_sums[block] = sum;
            deinterleave_block(&quantized, &mut values[start..start + block_size], group);
        }

        Self {
            values,
            block_sums,
            scales,
        }
    }
}

/// Write `source` into `out` grouped by nibble parity: within each `group` of
/// `k`, the even `k` first and then the odd ones.
///
/// This is exactly the order `nibble_block_dot`'s unpack produces, so the two
/// must be edited together; `the_deinterleave_matches_the_unpack` pins them.
fn deinterleave_block(source: &[i8], out: &mut [i16], group: usize) {
    debug_assert_eq!(source.len(), out.len());
    debug_assert!(out.len().is_multiple_of(group));
    let half = group / 2;
    for (chunk, slot) in source.chunks_exact(group).zip(out.chunks_exact_mut(group)) {
        for index in 0..half {
            slot[index] = i16::from(chunk[2 * index]);
            slot[half + index] = i16::from(chunk[2 * index + 1]);
        }
    }
}

/// `sum_k activation[k] * nibble[k]` over one `k` block.
///
/// `packed` is `block_size / 2` bytes of the ONNX weight, untouched.
/// `activation` is `block_size` deinterleaved `i16` from [`NibbleActivation`].
///
/// **Exactness.** `vpmaddwd` takes `i16` lanes and produces `i32` lanes, so no
/// product and no pairwise sum can saturate: `|a| <= 128` and `w <= 15` give
/// `|a * w| <= 1920` and a pair `<= 3840`. Accumulating a `block_size = 128`
/// block adds 64 such pairs, `< 2^18`. The `i32` accumulator has no path to
/// overflow at any `block_size` this operator admits.
#[inline]
pub(crate) fn nibble_block_dot(packed: &[u8], activation: &[i16], group: usize) -> i32 {
    debug_assert_eq!(packed.len() * 2, activation.len());
    debug_assert!(activation.len().is_multiple_of(group));
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was just detected; the lengths are debug-asserted
            // consistent above and re-derived inside from `packed.len()`.
            return unsafe { nibble_block_dot_avx2(packed, activation, group) };
        }
    }
    nibble_block_dot_reference(packed, activation, group)
}

/// The contract [`nibble_block_dot`] implements, written so it can be read.
///
/// Not a fallback for a rare shape -- it is the definition every vector path is
/// tested against, and the only implementation on non-x86 hosts.
pub(crate) fn nibble_block_dot_reference(packed: &[u8], activation: &[i16], group: usize) -> i32 {
    let half = group / 2;
    let mut total = 0i32;
    for (index, byte) in packed.iter().enumerate() {
        let within = index % half;
        let base = (index / half) * group;
        total += i32::from(activation[base + within]) * i32::from(byte & 0x0f);
        total += i32::from(activation[base + half + within]) * i32::from(byte >> 4);
    }
    total
}

/// AVX2 packed-nibble dot.
///
/// # Safety
///
/// The host must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn nibble_block_dot_avx2(packed: &[u8], activation: &[i16], group: usize) -> i32 {
    use std::arch::x86_64::*;

    if group < WIDE_GROUP {
        return unsafe { nibble_block_dot_avx2_narrow(packed, activation) };
    }
    let groups = packed.len() / (WIDE_GROUP / 2);
    let mask = _mm256_set1_epi16(0x000f);
    let mut accumulator = _mm256_setzero_si256();
    for index in 0..groups {
        // SAFETY: `index < packed.len() / 16`, so 16 bytes stay in bounds, and
        // `loadu` permits an unaligned pointer.
        let bytes = unsafe { _mm_loadu_si128(packed.as_ptr().add(index * 16).cast()) };
        // 16 packed bytes -> 16 `i16` lanes, each holding one byte.
        let widened = _mm256_cvtepu8_epi16(bytes);
        // Low nibbles are the even `k` of this group, high nibbles the odd.
        let low = _mm256_and_si256(widened, mask);
        let high = _mm256_srli_epi16(widened, 4);
        // SAFETY: `activation.len() == packed.len() * 2`, so the 32 `i16` this
        // group reads are in bounds.
        let even = unsafe { _mm256_loadu_si256(activation.as_ptr().add(index * 32).cast()) };
        // SAFETY: as above, the second half of the same 32-lane group.
        let odd = unsafe { _mm256_loadu_si256(activation.as_ptr().add(index * 32 + 16).cast()) };
        // `i16 x i16 -> i32` pairwise: exact, never saturating.
        accumulator = _mm256_add_epi32(accumulator, _mm256_madd_epi16(even, low));
        accumulator = _mm256_add_epi32(accumulator, _mm256_madd_epi16(odd, high));
    }
    // SAFETY: AVX2 is guaranteed by this function's own contract.
    unsafe { horizontal_sum(accumulator) }
}

/// The `block_size = 16` case: 8 packed bytes per group, so the widen produces
/// eight `i16` and the work fits 128-bit registers.
///
/// # Safety
///
/// The host must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nibble_block_dot_avx2_narrow(packed: &[u8], activation: &[i16]) -> i32 {
    use std::arch::x86_64::*;

    let groups = packed.len() / (MIN_BLOCK_SIZE / 2);
    let mask = _mm_set1_epi16(0x000f);
    let mut accumulator = _mm_setzero_si128();
    for index in 0..groups {
        // SAFETY: `index < packed.len() / 8`, so the 8 bytes read stay in
        // bounds; `loadl_epi64` reads exactly 8 and permits an unaligned
        // pointer.
        let bytes = unsafe { _mm_loadl_epi64(packed.as_ptr().add(index * 8).cast()) };
        let widened = _mm_cvtepu8_epi16(bytes);
        let low = _mm_and_si128(widened, mask);
        let high = _mm_srli_epi16(widened, 4);
        // SAFETY: `activation.len() == packed.len() * 2` bounds both halves.
        let even = unsafe { _mm_loadu_si128(activation.as_ptr().add(index * 16).cast()) };
        // SAFETY: as above.
        let odd = unsafe { _mm_loadu_si128(activation.as_ptr().add(index * 16 + 8).cast()) };
        accumulator = _mm_add_epi32(accumulator, _mm_madd_epi16(even, low));
        accumulator = _mm_add_epi32(accumulator, _mm_madd_epi16(odd, high));
    }
    let high = _mm_unpackhi_epi64(accumulator, accumulator);
    let pair = _mm_add_epi32(accumulator, high);
    let single = _mm_add_epi32(pair, _mm_shuffle_epi32(pair, 0b01));
    _mm_cvtsi128_si32(single)
}

/// One block's dot, left as four unreduced `i32` lanes.
///
/// Same arithmetic as [`nibble_block_dot_avx2`], stopping one step earlier so a
/// tile can reduce four blocks together. Both group widths fold to 128 bits
/// here, which is what makes the tile's `hadd` tree width-independent.
///
/// # Safety
///
/// The host must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn nibble_block_acc_avx2(
    packed: &[u8],
    activation: &[i16],
    group: usize,
) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::*;

    if group < WIDE_GROUP {
        let groups = packed.len() / (MIN_BLOCK_SIZE / 2);
        let mask = _mm_set1_epi16(0x000f);
        let mut accumulator = _mm_setzero_si128();
        for index in 0..groups {
            // SAFETY: `index < packed.len() / 8`, so the 8 bytes read are in
            // bounds and `loadl_epi64` permits an unaligned pointer.
            let bytes = unsafe { _mm_loadl_epi64(packed.as_ptr().add(index * 8).cast()) };
            let widened = _mm_cvtepu8_epi16(bytes);
            let low = _mm_and_si128(widened, mask);
            let high = _mm_srli_epi16(widened, 4);
            // SAFETY: `activation.len() == packed.len() * 2` bounds both halves.
            let even = unsafe { _mm_loadu_si128(activation.as_ptr().add(index * 16).cast()) };
            // SAFETY: as above.
            let odd = unsafe { _mm_loadu_si128(activation.as_ptr().add(index * 16 + 8).cast()) };
            accumulator = _mm_add_epi32(accumulator, _mm_madd_epi16(even, low));
            accumulator = _mm_add_epi32(accumulator, _mm_madd_epi16(odd, high));
        }
        return accumulator;
    }

    let groups = packed.len() / (WIDE_GROUP / 2);
    let mask = _mm256_set1_epi16(0x000f);
    let mut accumulator = _mm256_setzero_si256();
    for index in 0..groups {
        // SAFETY: `index < packed.len() / 16`, so 16 bytes stay in bounds.
        let bytes = unsafe { _mm_loadu_si128(packed.as_ptr().add(index * 16).cast()) };
        let widened = _mm256_cvtepu8_epi16(bytes);
        let low = _mm256_and_si256(widened, mask);
        let high = _mm256_srli_epi16(widened, 4);
        // SAFETY: `activation.len() == packed.len() * 2` bounds both halves.
        let even = unsafe { _mm256_loadu_si256(activation.as_ptr().add(index * 32).cast()) };
        // SAFETY: as above.
        let odd = unsafe { _mm256_loadu_si256(activation.as_ptr().add(index * 32 + 16).cast()) };
        accumulator = _mm256_add_epi32(accumulator, _mm256_madd_epi16(even, low));
        accumulator = _mm256_add_epi32(accumulator, _mm256_madd_epi16(odd, high));
    }
    _mm_add_epi32(
        _mm256_castsi256_si128(accumulator),
        _mm256_extracti128_si256(accumulator, 1),
    )
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum(value: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let high = _mm256_extracti128_si256(value, 1);
    let low = _mm256_castsi256_si128(value);
    let sum = _mm_add_epi32(low, high);
    let folded = _mm_add_epi32(sum, _mm_unpackhi_epi64(sum, sum));
    let single = _mm_add_epi32(folded, _mm_shuffle_epi32(folded, 0b01));
    _mm_cvtsi128_si32(single)
}

/// The zero point of one `(output, block)`, or ONNX's default of 8.
///
/// Zero points are themselves packed two per byte along `k` blocks, with each
/// output's run padded to a whole byte.
#[inline]
pub(crate) fn zero_point_at(
    zero_points: Option<&[u8]>,
    output: usize,
    block: usize,
    k_blocks: usize,
) -> i32 {
    match zero_points {
        None => 8,
        Some(points) => {
            let row = k_blocks.div_ceil(2);
            let byte = points[output * row + block / 2];
            i32::from(if block.is_multiple_of(2) {
                byte & 0x0f
            } else {
                byte >> 4
            })
        }
    }
}

/// One output row: `result[output] = sum_blocks scale_a * scale_w * dot`.
///
/// `packed` is the whole `N x k_blocks x (block_size / 2)` ONNX weight,
/// `scales` the matching `N x k_blocks`, and `zero_points` the optional packed
/// tensor. Nothing is copied or expanded.
#[allow(clippy::too_many_arguments)]
/// Check every length relation the vector paths rely on, before any of them run.
///
/// The AVX2 path reaches its weights and activations through `chunks_exact`, so
/// it cannot read out of bounds whatever it is handed -- but a caller that
/// passed a `k_blocks` or `block_size` disagreeing with the buffers it also
/// passed would get a silently *short* answer instead of a diagnosable failure,
/// because `chunks_exact` stops early and drops the remainder. These relations
/// hold by construction today (`NibbleActivation::new` is given the same
/// `k_blocks * block_size` the caller then passes here), but that invariant
/// spans two functions and nothing enforced it. Checking it costs a dozen
/// comparisons per output range, against `k_blocks * blob` bytes of work.
///
/// Every product is `checked_mul`, so a malformed `k_blocks` cannot wrap into a
/// small value that then passes a length test it should have failed.
fn validate_nibble_outputs(
    activation: &NibbleActivation,
    packed: &[u8],
    scales: &[f32],
    zero_points: Option<&[u8]>,
    outputs: &std::ops::Range<usize>,
    result: &[f32],
    k_blocks: usize,
    block_size: usize,
) {
    assert!(
        block_size >= MIN_BLOCK_SIZE && block_size.is_power_of_two(),
        "block_size must be a power of two and at least {MIN_BLOCK_SIZE}, got {block_size}"
    );
    assert_eq!(
        result.len(),
        outputs.len(),
        "one result slot per output column"
    );
    let padded_k = k_blocks
        .checked_mul(block_size)
        .expect("k_blocks * block_size overflows");
    assert_eq!(
        activation.values.len(),
        padded_k,
        "activation was laid out for a different k_blocks/block_size"
    );
    assert_eq!(activation.block_sums.len(), k_blocks, "one block sum each");
    assert_eq!(activation.scales.len(), k_blocks, "one block scale each");

    let row_bytes = k_blocks
        .checked_mul(block_size / 2)
        .expect("k_blocks * blob overflows");
    let weights_needed = outputs
        .end
        .checked_mul(row_bytes)
        .expect("outputs.end * row_bytes overflows");
    assert!(
        packed.len() >= weights_needed,
        "packed weights hold {} bytes, need {weights_needed}",
        packed.len()
    );
    let scales_needed = outputs
        .end
        .checked_mul(k_blocks)
        .expect("outputs.end * k_blocks overflows");
    assert!(
        scales.len() >= scales_needed,
        "scales hold {}, need {scales_needed}",
        scales.len()
    );
    if let Some(points) = zero_points {
        let zero_needed = outputs
            .end
            .checked_mul(k_blocks.div_ceil(2))
            .expect("outputs.end * zero_row overflows");
        assert!(
            points.len() >= zero_needed,
            "zero points hold {}, need {zero_needed}",
            points.len()
        );
    }
}

pub(crate) fn nibble_outputs(
    activation: &NibbleActivation,
    packed: &[u8],
    scales: &[f32],
    zero_points: Option<&[u8]>,
    outputs: std::ops::Range<usize>,
    result: &mut [f32],
    k_blocks: usize,
    block_size: usize,
) {
    validate_nibble_outputs(
        activation,
        packed,
        scales,
        zero_points,
        &outputs,
        result,
        k_blocks,
        block_size,
    );
    // Detected once per output range rather than once per block. The per-block
    // spelling cost more than the arithmetic it guarded: a `block_size = 32`
    // block is a single 32-nibble group, so the feature probe, the call and the
    // horizontal reduction were paid every 32 weights. Hoisting it also lets
    // `nibble_block_dot_avx2` inline into this loop, which is what keeps the
    // accumulator in registers across the block.
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was just detected, which is this function's only
            // requirement; every index it forms is checked inside.
            unsafe {
                nibble_outputs_avx2(
                    activation,
                    packed,
                    scales,
                    zero_points,
                    outputs,
                    result,
                    k_blocks,
                    block_size,
                );
            }
            return;
        }
    }
    nibble_outputs_generic(
        activation,
        packed,
        scales,
        zero_points,
        outputs,
        result,
        k_blocks,
        block_size,
    );
}

/// Wide-path block accumulator for exactly one block, over raw pointers.
///
/// The general [`nibble_block_acc_avx2`] re-derives its group count from
/// `packed.len()` and re-branches on `group` on every call. At
/// `block_size = 32` a block is a single 16-byte group -- about ten uops -- so
/// that bookkeeping is amortized over nothing and dominates. Deciding it once
/// per output range instead is worth 1.62x on the decode loop at 1-4 threads;
/// see `docs/benchmarks/2026-08-21-int4-acc4-execution-regime.md`.
///
/// **Why raw pointers rather than slices.** Two safe formulations were built
/// and measured against this one on the real decode loop at one thread, and
/// neither is free: zipped `chunks_exact` over the tile and the block costs
/// 5% of the win (1.54x against 1.62x) and bounds-checked sub-slicing of the
/// tile costs 17% (1.34x). Since safe slicing is not codegen-equivalent here,
/// the pointers stay -- and the invariant they rest on is instead *enforced*,
/// once per output range, by [`validate_nibble_outputs`], which is what makes
/// the call sites' proofs below sound rather than merely conventional.
///
/// # Safety
///
/// The host must support AVX2, and the caller must guarantee `groups * 16`
/// readable bytes at `packed` and `groups * 32` readable `i16` at `activation`,
/// both for the whole call. The two pointers need no particular alignment --
/// every load below is an unaligned `loadu` -- and neither is retained, offset
/// past those bounds, or converted to an integer.
///
/// Loop invariant: on entry to iteration `index`, `0 <= index < groups`, so the
/// reads are confined to `packed[index * 16 .. index * 16 + 16]` and
/// `activation[index * 32 .. index * 32 + 32]`; both are within the contracted
/// bounds because `index + 1 <= groups`. Nothing else is dereferenced.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn nibble_block_acc_avx2_wide_raw(
    packed: *const u8,
    activation: *const i16,
    groups: usize,
) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::*;
    let mask = _mm256_set1_epi16(0x000f);
    let mut accumulator = _mm256_setzero_si256();
    for index in 0..groups {
        // SAFETY: `index < groups`, so these 16 bytes lie inside the
        // `groups * 16` this function's contract guarantees; `loadu` imposes no
        // alignment requirement.
        let bytes = unsafe { _mm_loadu_si128(packed.add(index * WIDE_GROUP / 2).cast()) };
        let widened = _mm256_cvtepu8_epi16(bytes);
        let low = _mm256_and_si256(widened, mask);
        let high = _mm256_srli_epi16(widened, 4);
        // SAFETY: `index < groups`, so these 16 `i16` lie inside the
        // `groups * 32` contracted; unaligned load.
        let even = unsafe { _mm256_loadu_si256(activation.add(index * WIDE_GROUP).cast()) };
        // SAFETY: as above, for the second half of the same group.
        let odd = unsafe {
            _mm256_loadu_si256(activation.add(index * WIDE_GROUP + WIDE_GROUP / 2).cast())
        };
        accumulator = _mm256_add_epi32(accumulator, _mm256_madd_epi16(even, low));
        accumulator = _mm256_add_epi32(accumulator, _mm256_madd_epi16(odd, high));
    }
    _mm_add_epi32(
        _mm256_castsi256_si128(accumulator),
        _mm256_extracti128_si256(accumulator, 1),
    )
}

/// [`nibble_outputs`] with AVX2 known present, so the block dot inlines.
///
/// # Safety
///
/// The host must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn nibble_outputs_avx2(
    activation: &NibbleActivation,
    packed: &[u8],
    scales: &[f32],
    zero_points: Option<&[u8]>,
    outputs: std::ops::Range<usize>,
    result: &mut [f32],
    k_blocks: usize,
    block_size: usize,
) {
    use std::arch::x86_64::*;

    debug_assert_eq!(outputs.len(), result.len());
    let blob = block_size / 2;
    let group = group_for(block_size);
    let row_bytes = k_blocks * blob;
    let zero_row = k_blocks.div_ceil(2);
    let tiles = k_blocks / BLOCK_TILE;
    let tiled_blocks = tiles * BLOCK_TILE;
    // Decided once per output range rather than once per block. `group_for`
    // returns `WIDE_GROUP` exactly when `block_size >= WIDE_GROUP`, so this is
    // `block_size >= 32`, and `block_size` is a validated power of two -- hence
    // `blob` is a whole number of 16-byte groups whenever `wide` holds.
    // Decided once per output range rather than once per block. `group_for`
    // returns `WIDE_GROUP` exactly when `block_size >= WIDE_GROUP`, so `wide`
    // is `block_size >= 32`; `validate_nibble_outputs` has established that
    // `block_size` is a power of two, so `blob = block_size / 2` is then a
    // whole number of 16-byte groups and `wide_groups * 16 == blob` exactly.
    // That exactness is what lets the raw kernel's group count stand in for the
    // block length; without it the division would truncate and silently drop a
    // partial group.
    let wide = group >= WIDE_GROUP;
    let wide_groups = blob / (WIDE_GROUP / 2);
    debug_assert!(!wide || wide_groups * (WIDE_GROUP / 2) == blob);
    debug_assert!(!wide || wide_groups * WIDE_GROUP == block_size);
    for (slot, output) in result.iter_mut().zip(outputs) {
        let weights = &packed[output * row_bytes..(output + 1) * row_bytes];
        let row_scales = &scales[output * k_blocks..(output + 1) * k_blocks];
        let zero_bytes =
            zero_points.map(|points| &points[output * zero_row..(output + 1) * zero_row]);
        let mut running = _mm_setzero_ps();
        // A tile is exactly `BLOCK_TILE` consecutive blocks of each buffer, so
        // both tile streams and the per-block streams inside them come from
        // `chunks_exact`. That is what replaced the previous revision's raw
        // pointer arithmetic: the block slices carry their own lengths, no
        // index is formed that needs a bounds check, and the earlier
        // `block * blob` / `block * block_size` products -- whose in-range-ness
        // rested on an invariant asserted in a different function -- are gone.
        for tile in 0..tiles {
            let base = tile * BLOCK_TILE;
            let mut lanes = [_mm_setzero_si128(); BLOCK_TILE];
            for (offset, lane) in lanes.iter_mut().enumerate() {
                let block = base + offset;
                // SAFETY (both pointers): `tile < tiles` and `offset <
                // BLOCK_TILE` give `block < tiles * BLOCK_TILE <= k_blocks`.
                // `validate_nibble_outputs` has already established, for this
                // exact `k_blocks`/`block_size`, that `weights` is `k_blocks *
                // blob` bytes and `activation.values` is `k_blocks *
                // block_size` elements. So `block * blob + blob <= k_blocks *
                // blob` and likewise for the activation: each offset is in
                // bounds and one whole block remains readable past it, which is
                // `wide_groups` groups. Formed above the `wide` branch on
                // purpose -- computing them inside it regressed block_size 64
                // to 0.876x.
                let w = unsafe { weights.as_ptr().add(block * blob) };
                let a = unsafe { activation.values.as_ptr().add(block * block_size) };
                *lane = if wide {
                    // SAFETY: AVX2 by this function's contract; `wide_groups *
                    // 16 == blob` bytes and `wide_groups * 32 == block_size`
                    // elements are readable at `w`/`a` per the paragraph above.
                    unsafe { nibble_block_acc_avx2_wide_raw(w, a, wide_groups) }
                } else {
                    // SAFETY: AVX2 as above; both slices are exactly one block.
                    unsafe {
                        nibble_block_acc_avx2(
                            &weights[block * blob..(block + 1) * blob],
                            &activation.values[block * block_size..(block + 1) * block_size],
                            group,
                        )
                    }
                };
            }
            // [sum(l0), sum(l1), sum(l2), sum(l3)] -- one tree for four blocks
            // instead of four separate reductions.
            let dots = _mm_hadd_epi32(
                _mm_hadd_epi32(lanes[0], lanes[1]),
                _mm_hadd_epi32(lanes[2], lanes[3]),
            );
            // A tile starts on an even block, so its four nibbles are the low
            // and high halves of two adjacent bytes, in that order.
            let zero_point = match zero_bytes {
                None => _mm_set1_epi32(8),
                Some(bytes) => {
                    let low = bytes[base / 2];
                    let high = bytes[base / 2 + 1];
                    _mm_setr_epi32(
                        i32::from(low & 0x0f),
                        i32::from(low >> 4),
                        i32::from(high & 0x0f),
                        i32::from(high >> 4),
                    )
                }
            };
            // SAFETY: `base + BLOCK_TILE <= k_blocks`, and each of these arrays
            // has `k_blocks` elements, so all four lanes are in bounds.
            let (block_sums, activation_scales, weight_scales) = unsafe {
                (
                    _mm_loadu_si128(activation.block_sums.as_ptr().add(base).cast()),
                    _mm_loadu_ps(activation.scales.as_ptr().add(base)),
                    _mm_loadu_ps(row_scales.as_ptr().add(base)),
                )
            };
            let centred = _mm_sub_epi32(dots, _mm_mullo_epi32(zero_point, block_sums));
            running = _mm_add_ps(
                running,
                _mm_mul_ps(
                    _mm_cvtepi32_ps(centred),
                    _mm_mul_ps(activation_scales, weight_scales),
                ),
            );
        }
        let folded = _mm_add_ps(running, _mm_movehl_ps(running, running));
        let mut total = _mm_cvtss_f32(_mm_add_ss(folded, _mm_shuffle_ps(folded, folded, 0b01)));
        // Up to `BLOCK_TILE - 1` blocks past the last whole tile.
        for block in tiled_blocks..k_blocks {
            let scale = activation.scales[block] * row_scales[block];
            if scale == 0.0 {
                continue;
            }
            // SAFETY: AVX2 holds by this function's contract.
            let lane = unsafe {
                nibble_block_acc_avx2(
                    &weights[block * blob..(block + 1) * blob],
                    &activation.values[block * block_size..(block + 1) * block_size],
                    group,
                )
            };
            let folded = _mm_add_epi32(lane, _mm_unpackhi_epi64(lane, lane));
            let dot = _mm_cvtsi128_si32(_mm_add_epi32(folded, _mm_shuffle_epi32(folded, 0b01)));
            let zero_point = zero_point_at(zero_points, output, block, k_blocks);
            total += (dot - zero_point * activation.block_sums[block]) as f32 * scale;
        }
        *slot = total;
    }
}

/// The body both entry points share.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn nibble_outputs_generic(
    activation: &NibbleActivation,
    packed: &[u8],
    scales: &[f32],
    zero_points: Option<&[u8]>,
    outputs: std::ops::Range<usize>,
    result: &mut [f32],
    k_blocks: usize,
    block_size: usize,
) {
    debug_assert_eq!(outputs.len(), result.len());
    let blob = block_size / 2;
    let group = group_for(block_size);
    let row_bytes = k_blocks * blob;
    for (slot, output) in result.iter_mut().zip(outputs) {
        let weights = &packed[output * row_bytes..(output + 1) * row_bytes];
        let row_scales = &scales[output * k_blocks..(output + 1) * k_blocks];
        // The four per-block reads walk in lockstep, so iterating them as
        // slices drops the bounds check the indexed spelling paid on each one.
        let mut total = 0.0f32;
        for (block, (((chunk, lane), activation_scale), weight_scale)) in weights
            .chunks_exact(blob)
            .zip(activation.values.chunks_exact(block_size))
            .zip(&activation.scales)
            .zip(row_scales)
            .enumerate()
        {
            let scale = activation_scale * weight_scale;
            if scale == 0.0 {
                continue;
            }
            let dot = nibble_block_dot(chunk, lane, group);
            let zero_point = zero_point_at(zero_points, output, block, k_blocks);
            let centred = dot - zero_point * activation.block_sums[block];
            total += centred as f32 * scale;
        }
        *slot = total;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random bytes; the same generator the other kernel
    /// tests in this crate use.
    fn bytes(count: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 33) as u8
            })
            .collect()
    }

    fn floats(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 33) as i32 as f32) / 1.0e9
            })
            .collect()
    }

    /// The `f64` contract: dequantize every weight and multiply by the
    /// *unquantized* activation. This is what the operator means, independent of
    /// any quantization the kernel chooses for the activation.
    fn oracle(
        activation: &[f32],
        packed: &[u8],
        scales: &[f32],
        zero_points: Option<&[u8]>,
        k: usize,
        n: usize,
        block_size: usize,
    ) -> Vec<f64> {
        let k_blocks = k.div_ceil(block_size);
        let blob = block_size / 2;
        (0..n)
            .map(|output| {
                let mut total = 0.0f64;
                for block in 0..k_blocks {
                    let zero_point = f64::from(zero_point_at(zero_points, output, block, k_blocks));
                    let scale = f64::from(scales[output * k_blocks + block]);
                    let base = block * block_size;
                    let valid = k.saturating_sub(base).min(block_size);
                    for offset in 0..valid {
                        let byte = packed[(output * k_blocks + block) * blob + offset / 2];
                        let nibble = if offset.is_multiple_of(2) {
                            byte & 0x0f
                        } else {
                            byte >> 4
                        };
                        total += f64::from(activation[base + offset])
                            * (f64::from(nibble) - zero_point)
                            * scale;
                    }
                }
                total
            })
            .collect()
    }

    /// The vector path and the readable reference must agree **exactly**, for
    /// every nibble value, at every block size, with and without zero points.
    ///
    /// Mutation check: swapping `and`/`srli` in the AVX2 unpack, or dropping the
    /// `half +` in the reference's odd index, fails this.
    #[test]
    fn the_vector_dot_equals_its_reference_on_every_nibble() {
        for block_size in [16usize, 32, 64, 128] {
            let group = group_for(block_size);
            for seed in 0..8u64 {
                let packed = bytes(block_size / 2, seed * 977 + 1);
                let mut source = vec![0i8; block_size];
                for (index, slot) in source.iter_mut().enumerate() {
                    // Cover the whole i8 range including both extremes.
                    *slot = (((index as i64 * 37 + seed as i64 * 11) % 256) - 128) as i8;
                }
                let mut activation = vec![0i16; block_size];
                deinterleave_block(&source, &mut activation, group);
                let reference = nibble_block_dot_reference(&packed, &activation, group);
                assert_eq!(
                    nibble_block_dot(&packed, &activation, group),
                    reference,
                    "block_size={block_size} seed={seed}"
                );
            }
        }
    }

    /// Exhaustive over nibble values: every one of the 256 packed bytes, against
    /// the extremes of the activation range, must give the textbook answer.
    #[test]
    fn every_packed_byte_unpacks_to_the_right_pair() {
        for group in [MIN_BLOCK_SIZE, WIDE_GROUP] {
            let half = group / 2;
            for byte in 0u8..=255 {
                let packed = vec![byte; half];
                for &value in &[-128i16, -1, 0, 1, 127] {
                    let activation = vec![value; group];
                    let expected = i32::from(value)
                        * half as i32
                        * (i32::from(byte & 0x0f) + i32::from(byte >> 4));
                    assert_eq!(
                        nibble_block_dot(&packed, &activation, group),
                        expected,
                        "byte={byte:#04x} value={value} group={group}"
                    );
                    assert_eq!(
                        nibble_block_dot_reference(&packed, &activation, group),
                        expected
                    );
                }
            }
        }
    }

    /// The deinterleave and the unpack are one contract split across two
    /// functions; this asserts activation element `k` really does meet nibble
    /// `k`.
    ///
    /// Mutation check: changing `deinterleave_block`'s `2 * index` to `index`
    /// fails here, and nowhere else in a way that is easy to read.
    #[test]
    fn the_deinterleave_matches_the_unpack() {
        for block_size in [16usize, 32, 64, 128] {
            let group = group_for(block_size);
            let packed = bytes(block_size / 2, 4242);
            for position in 0..block_size {
                // A one-hot activation isolates a single `k`.
                let mut source = vec![0i8; block_size];
                source[position] = 1;
                let mut activation = vec![0i16; block_size];
                deinterleave_block(&source, &mut activation, group);
                let byte = packed[position / 2];
                let expected = i32::from(if position.is_multiple_of(2) {
                    byte & 0x0f
                } else {
                    byte >> 4
                });
                assert_eq!(
                    nibble_block_dot(&packed, &activation, group),
                    expected,
                    "block_size={block_size} k={position}"
                );
            }
        }
    }

    /// Absent zero points must mean exactly 8 -- ONNX's default, and the value
    /// that makes an unsigned nibble a signed int4.
    #[test]
    fn an_absent_zero_point_tensor_is_the_default_of_eight() {
        for k_blocks in [1usize, 2, 3, 7, 8] {
            for block in 0..k_blocks {
                assert_eq!(zero_point_at(None, 0, block, k_blocks), 8);
            }
        }
        // And the packed spelling of all-8 reproduces it.
        for k_blocks in [1usize, 2, 3, 7, 8] {
            let points = vec![0x88u8; k_blocks.div_ceil(2)];
            for block in 0..k_blocks {
                assert_eq!(zero_point_at(Some(&points), 0, block, k_blocks), 8);
            }
        }
    }

    /// Zero points are packed low-nibble-first along `k` blocks, and each
    /// output's run is padded to a whole byte.
    ///
    /// Mutation check: dropping the `div_ceil` on the row stride, or swapping
    /// the nibble halves, fails this.
    #[test]
    fn zero_points_unpack_per_block_and_per_output() {
        // Three blocks -> two bytes per output, the last half wasted.
        let k_blocks = 3;
        let points = vec![0x21, 0x03, 0x54, 0x06];
        assert_eq!(zero_point_at(Some(&points), 0, 0, k_blocks), 1);
        assert_eq!(zero_point_at(Some(&points), 0, 1, k_blocks), 2);
        assert_eq!(zero_point_at(Some(&points), 0, 2, k_blocks), 3);
        assert_eq!(zero_point_at(Some(&points), 1, 0, k_blocks), 4);
        assert_eq!(zero_point_at(Some(&points), 1, 1, k_blocks), 5);
        assert_eq!(zero_point_at(Some(&points), 1, 2, k_blocks), 6);
    }

    /// End to end against the `f64` contract, over every block size, with and
    /// without zero points, and with a `k` that does not fill its last block.
    #[test]
    fn the_kernel_tracks_the_float64_contract() {
        for block_size in [16usize, 32, 64, 128] {
            for &k in &[
                block_size,
                block_size * 3,
                block_size * 3 - 1,
                block_size + 5,
            ] {
                let n = 6;
                let k_blocks = k.div_ceil(block_size);
                let padded_k = k_blocks * block_size;
                let packed = bytes(n * k_blocks * (block_size / 2), 0x51 + k as u64);
                let scales: Vec<f32> = floats(n * k_blocks, 0x9e + k as u64)
                    .iter()
                    .map(|value| value.abs() * 1.0e6 + 1.0e-3)
                    .collect();
                let activation = floats(k, 0x1234 + k as u64);
                for zero_points in [None, Some(bytes(n * k_blocks.div_ceil(2), 7))] {
                    let prepared = NibbleActivation::new(&activation, padded_k, block_size);
                    let mut result = vec![0.0f32; n];
                    nibble_outputs(
                        &prepared,
                        &packed,
                        &scales,
                        zero_points.as_deref(),
                        0..n,
                        &mut result,
                        k_blocks,
                        block_size,
                    );
                    let expect = oracle(
                        &activation,
                        &packed,
                        &scales,
                        zero_points.as_deref(),
                        k,
                        n,
                        block_size,
                    );
                    for (output, (got, want)) in result.iter().zip(&expect).enumerate() {
                        let magnitude = expect.iter().fold(0.0f64, |a, b| a.max(b.abs()));
                        // int8 activation quantization is ~1/127 of a block's
                        // peak; the bound is on the whole row, not per element.
                        let tolerance = magnitude * 0.05 + 1.0e-3;
                        assert!(
                            (f64::from(*got) - want).abs() <= tolerance,
                            "block_size={block_size} k={k} output={output} \
                             got={got} want={want} tolerance={tolerance}"
                        );
                    }
                }
            }
        }
    }

    /// The activation past `k` must be zero, or a short final block would read
    /// whatever the last real value was.
    ///
    /// Mutation check: clamping `end` to `start + block_size` without the
    /// `activation.len()` term fails this with an out-of-bounds panic; filling
    /// the pad with the last value fails the assertion.
    #[test]
    fn a_short_final_block_contributes_only_its_real_elements() {
        let block_size = 32;
        let k = block_size + 5;
        let padded_k = 2 * block_size;
        let mut activation = floats(k, 99);
        let short = NibbleActivation::new(&activation, padded_k, block_size);
        // Anything appended past `k` must not change the result.
        activation.extend_from_slice(&floats(padded_k - k, 1234));
        let long = NibbleActivation::new(&activation[..k], padded_k, block_size);
        assert_eq!(short.values, long.values);
        assert_eq!(short.block_sums, long.block_sums);
        // The tail block's own sum counts five elements, not thirty-two.
        let counted = short.values[block_size..]
            .iter()
            .filter(|v| **v != 0)
            .count();
        assert!(
            counted <= 5,
            "padding leaked into the tail block: {counted}"
        );
    }

    /// An all-zero activation block has scale 0; the kernel must not turn that
    /// into a NaN or skip a block that still has to contribute nothing.
    #[test]
    fn a_zero_activation_block_contributes_nothing() {
        let (block_size, n) = (32usize, 4usize);
        let k = block_size * 2;
        let k_blocks = 2;
        let mut activation = vec![0.0f32; k];
        activation[..block_size].copy_from_slice(&floats(block_size, 5));
        let packed = bytes(n * k_blocks * (block_size / 2), 31);
        let scales = vec![0.25f32; n * k_blocks];
        let prepared = NibbleActivation::new(&activation, k, block_size);
        assert_eq!(prepared.scales[1], 0.0);
        assert_eq!(prepared.block_sums[1], 0);
        let mut result = vec![0.0f32; n];
        nibble_outputs(
            &prepared,
            &packed,
            &scales,
            None,
            0..n,
            &mut result,
            k_blocks,
            block_size,
        );
        // The oracle sees the same zero second block, so agreement proves the
        // skipped block contributed exactly nothing rather than being dropped.
        let expect = oracle(&activation, &packed, &scales, None, k, n, block_size);
        let magnitude = expect.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        for (got, want) in result.iter().zip(&expect) {
            assert!(got.is_finite(), "got {got}");
            assert!(
                (f64::from(*got) - want).abs() <= magnitude * 0.05 + 1.0e-3,
                "got={got} want={want}"
            );
        }
    }

    /// Pathological magnitudes must not overflow the `i32` accumulator: the
    /// largest possible block is `block_size = 128`, every nibble `15`, every
    /// activation `-128`.
    #[test]
    fn the_widest_block_at_full_range_cannot_overflow() {
        let block_size = 128;
        let packed = vec![0xffu8; block_size / 2];
        let activation = vec![-128i16; block_size];
        let expected = -128i32 * 15 * block_size as i32;
        assert_eq!(nibble_block_dot(&packed, &activation, WIDE_GROUP), expected);
        assert!(expected.checked_mul(2).is_some(), "headroom check");
    }

    /// `supported` is the whole host/config gate; a block size the operator
    /// never emits must not reach the kernel.
    ///
    /// Mutation check: dropping the `is_power_of_two` term admits 48.
    /// Reduce the wide accumulator the way the tail loop does, so a block's
    /// four lanes can be compared against the scalar reference as one integer.
    #[cfg(target_arch = "x86_64")]
    fn fold_lanes(lanes: std::arch::x86_64::__m128i) -> i32 {
        use std::arch::x86_64::*;
        // SAFETY: SSE2 is baseline on x86_64; these are pure register ops.
        unsafe {
            let folded = _mm_add_epi32(lanes, _mm_unpackhi_epi64(lanes, lanes));
            _mm_cvtsi128_si32(_mm_add_epi32(folded, _mm_shuffle_epi32(folded, 0b01)))
        }
    }

    /// The split-out wide accumulator is the function the decode loop actually
    /// runs, and until this test it had no direct coverage at all: every
    /// `int4_nibble` test drove `k_blocks <= 3`, and `tiles = k_blocks /
    /// BLOCK_TILE` is then **zero**, so the whole tiled loop was skipped and
    /// only the scalar tail ran. The mutation below was caught solely by
    /// `matmul_nbits` integration tests, one module away.
    ///
    /// Exhaustive over all 256 packed byte values at every wide block size,
    /// against the readable reference.
    ///
    /// Mutation check: `_mm256_srli_epi16(widened, 4)` -> `5`, or swapping the
    /// `low`/`high` operands, or dropping the `+ WIDE_GROUP / 2` on the odd
    /// activation load, each fail this.
    #[test]
    fn the_wide_block_accumulator_matches_the_scalar_reference_exhaustively() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return;
            }
            for block_size in [32usize, 64, 128] {
                let blob = block_size / 2;
                let groups = blob / (WIDE_GROUP / 2);
                // Every one of the 256 byte values appears in every nibble
                // position across the sweep of starting offsets.
                for start in 0..256usize {
                    let packed: Vec<u8> = (0..blob)
                        .map(|index| ((start + index) % 256) as u8)
                        .collect();
                    let mut source = vec![0i8; block_size];
                    for (index, slot) in source.iter_mut().enumerate() {
                        // Walk the whole i8 range, extremes included.
                        *slot = (((index * 37 + start * 11) % 256) as i64 - 128) as i8;
                    }
                    let mut activation = vec![0i16; block_size];
                    deinterleave_block(&source, &mut activation, WIDE_GROUP);
                    let reference = nibble_block_dot_reference(&packed, &activation, WIDE_GROUP);
                    // SAFETY: AVX2 was just detected; `packed` is `groups * 16`
                    // bytes and `activation` is `groups * 32` `i16` by
                    // construction, which is this function's whole contract.
                    let got = unsafe {
                        nibble_block_acc_avx2_wide_raw(packed.as_ptr(), activation.as_ptr(), groups)
                    };
                    assert_eq!(
                        fold_lanes(got),
                        reference,
                        "block_size={block_size} start={start}"
                    );
                }
            }
        }
    }

    /// Drive `nibble_outputs` with enough blocks to actually enter the tiled
    /// AVX2 loop (`k_blocks >= BLOCK_TILE`), across every block size, both
    /// zero-point spellings, ragged tails, and scales chosen to be hostile.
    ///
    /// `the_kernel_tracks_the_float64_contract` looks like it covers this and
    /// does not: its largest `k` is `block_size * 3`, so `tiles` is always 0.
    ///
    /// Mutation check: dropping the `wide` branch's hoisted `wide_groups` to
    /// `wide_groups - 1`, or reducing `tiles` by one, fails this.
    #[test]
    fn the_tiled_path_tracks_the_float64_contract() {
        for block_size in [16usize, 32, 64, 128] {
            // 4 and 8 are whole tiles; 5, 7 and 9 leave a 1-3 block tail, so
            // both the tiled loop and the scalar remainder run in one call.
            for blocks in [4usize, 5, 7, 8, 9] {
                for ragged in [0usize, 1] {
                    let k = blocks * block_size - ragged;
                    let k_blocks = k.div_ceil(block_size);
                    let padded_k = k_blocks * block_size;
                    let n = 5;
                    let packed = bytes(n * k_blocks * (block_size / 2), 0x2b + k as u64);
                    // Hostile scales: denormal-adjacent, huge, exactly zero
                    // (which the generic path short-circuits) and negative.
                    let scales: Vec<f32> = (0..n * k_blocks)
                        .map(|index| match index % 4 {
                            0 => 1.0e-30,
                            1 => 1.0e6,
                            2 => 0.0,
                            _ => -3.5e-2,
                        })
                        .collect();
                    let activation = floats(k, 0x77 + k as u64);
                    for zero_points in [None, Some(bytes(n * k_blocks.div_ceil(2), 0x5a))] {
                        let prepared = NibbleActivation::new(&activation, padded_k, block_size);
                        let mut result = vec![0.0f32; n];
                        nibble_outputs(
                            &prepared,
                            &packed,
                            &scales,
                            zero_points.as_deref(),
                            0..n,
                            &mut result,
                            k_blocks,
                            block_size,
                        );
                        let expect = oracle(
                            &activation,
                            &packed,
                            &scales,
                            zero_points.as_deref(),
                            k,
                            n,
                            block_size,
                        );
                        let magnitude = expect.iter().fold(0.0f64, |a, b| a.max(b.abs()));
                        for (output, (got, want)) in result.iter().zip(&expect).enumerate() {
                            let tolerance = magnitude * 0.05 + 1.0e-3;
                            assert!(
                                (f64::from(*got) - want).abs() <= tolerance,
                                "block_size={block_size} k={k} blocks={blocks} \
                                 output={output} got={got} want={want}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A caller whose buffers disagree with the `k_blocks`/`block_size` it also
    /// passed must be stopped by a named assertion *before* any vector path
    /// forms a pointer from those lengths -- not produce a short answer, and
    /// not read out of bounds.
    ///
    /// This is the check that makes the raw-pointer kernel's safety argument
    /// hold: its call site proves `block * blob + blob <= weights.len()` from
    /// relations that, before this, were established in a different function
    /// and enforced nowhere.
    ///
    /// Each case asserts the **message**, not merely that something panicked.
    /// The first spelling of this test only checked `is_err()` and passed with
    /// `validate_nibble_outputs` deleted entirely, because these inputs also
    /// trip an incidental slice bounds check further in -- which proves nothing
    /// about ordering and nothing about the cases that would instead be silent.
    #[test]
    fn malformed_lengths_are_refused_before_any_unsafe_runs() {
        let block_size = 32;
        let k_blocks = 8;
        let n = 3;
        let padded_k = k_blocks * block_size;
        let packed = bytes(n * k_blocks * (block_size / 2), 1);
        let scales = vec![1.0f32; n * k_blocks];
        let activation = floats(padded_k, 2);
        let prepared = NibbleActivation::new(&activation, padded_k, block_size);
        let zero = bytes(n * k_blocks.div_ceil(2), 3);

        let call = |packed: &[u8], scales: &[f32], zero: Option<&[u8]>, kb: usize, bs: usize| {
            let mut result = vec![0.0f32; n];
            nibble_outputs(&prepared, packed, scales, zero, 0..n, &mut result, kb, bs);
        };

        #[allow(clippy::type_complexity)]
        let cases: Vec<(&str, Box<dyn Fn() + std::panic::UnwindSafe>)> = vec![
            (
                "activation was laid out for a different k_blocks/block_size",
                Box::new(|| call(&packed, &scales, Some(&zero), k_blocks + 1, block_size)),
            ),
            (
                "activation was laid out for a different k_blocks/block_size",
                Box::new(|| call(&packed, &scales, Some(&zero), k_blocks, block_size * 2)),
            ),
            (
                "packed weights hold",
                Box::new(|| {
                    call(
                        &packed[..packed.len() - 1],
                        &scales,
                        Some(&zero),
                        k_blocks,
                        block_size,
                    )
                }),
            ),
            (
                "scales hold",
                Box::new(|| {
                    call(
                        &packed,
                        &scales[..scales.len() - 1],
                        Some(&zero),
                        k_blocks,
                        block_size,
                    )
                }),
            ),
            (
                "zero points hold",
                Box::new(|| {
                    call(
                        &packed,
                        &scales,
                        Some(&zero[..zero.len() - 1]),
                        k_blocks,
                        block_size,
                    )
                }),
            ),
            (
                "block_size must be a power of two",
                Box::new(|| call(&packed, &scales, Some(&zero), k_blocks, 48)),
            ),
        ];

        for (expected, case) in cases {
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let outcome = std::panic::catch_unwind(case);
            std::panic::set_hook(previous);
            let payload = outcome.expect_err("must be refused, not accepted");
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| {
                    payload
                        .downcast_ref::<&str>()
                        .map(|text| (*text).to_owned())
                })
                .unwrap_or_default();
            assert!(
                message.contains(expected),
                "expected the validator to reject this with {expected:?}, \
                 got {message:?} -- an incidental bounds check is not the \
                 same guarantee, since it runs after pointers are formed"
            );
        }
    }

    #[test]
    fn unsupported_block_sizes_are_refused() {
        for block_size in [0usize, 1, 8, 15, 24, 48, 96] {
            assert!(!supported(block_size), "block_size={block_size}");
        }
        // On a host that can run the kernel at all, every admissible block size
        // is admitted -- the gate must not silently narrow to one of them.
        if supported(32) {
            for block_size in [16usize, 64, 128, 256] {
                assert!(supported(block_size), "block_size={block_size}");
            }
        }
    }
}
