//! Tests for the planar block-scaled formats (block-FP8, planar-FP4).
//!
//! The [`dequantize_planar_kn`] oracle iterates **per element** and writes the
//! `[K, N]` (`in`-major) weight directly. Every parity test here cross-checks it
//! against an *independent* reference that derives the dense weight by a
//! different route:
//!
//! * `reference_logical_block_fp8` iterates **per scale block** (block-major),
//!   builds a row-major `[out, in]` matrix, then transposes;
//! * `reference_logical_fp4` decodes each 32-wide micro-block with the vetted
//!   production helper [`dequantize_fp4_e2m1_block`] (a wholly separate code
//!   path with its own nibble ordering) before transposing.
//!
//! An indexing bug in the oracle cannot survive both derivations, and both are
//! anchored to hand-computed known values (E2M1 / E4M3 / UE8M0 tables).

use super::*;
use crate::kernels::block_dequant::{
    decode_e2m1, decode_e4m3fn, decode_e8m0_scale, dequantize_fp4_e2m1_block,
};

// ---------------------------------------------------------------------------
// Deterministic byte generators (no external rng)
// ---------------------------------------------------------------------------

/// Tiny xorshift so the fixtures are reproducible without a dependency.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(1))
    }

    fn next_u8(&mut self) -> u8 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 24) as u8
    }
}

/// E4M3 codes with the two reserved NaN encodings (`0x7f`, `0xff`) mapped to a
/// benign zero, so block-FP8 fixtures never trip the fail-closed NaN guard.
fn finite_e4m3(code: u8) -> u8 {
    if code & 0x7f == 0x7f { 0x00 } else { code }
}

fn block_fp8_fixture(out: usize, in_features: usize, bs: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let mut rng = Lcg::new(seed);
    let packed: Vec<u8> = (0..out * in_features)
        .map(|_| finite_e4m3(rng.next_u8()))
        .collect();
    let scale_rows = out.div_ceil(bs);
    let scale_cols = in_features.div_ceil(bs);
    // Keep UE8M0 exponents near 127 (scale ~1) and never 0xff (reserved).
    let scale: Vec<u8> = (0..scale_rows * scale_cols)
        .map(|_| 120 + (rng.next_u8() % 15))
        .collect();
    (packed, scale)
}

fn fp4_fixture(out: usize, in_features: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let mut rng = Lcg::new(seed);
    let packed: Vec<u8> = (0..out * (in_features / FP4_PACK_FACTOR))
        .map(|_| rng.next_u8())
        .collect();
    let scale: Vec<u8> = (0..out * (in_features / FP4_MICROSCALE_BLOCK))
        .map(|_| 120 + (rng.next_u8() % 15))
        .collect();
    (packed, scale)
}

// ---------------------------------------------------------------------------
// Independent references (different derivation than the oracle)
// ---------------------------------------------------------------------------

/// Row-major `[out, in]`, built independently of the oracle's per-element loop.
///
/// Two independent derivations combine here: the packed E4M3 magnitudes are
/// decoded by **row-chunk iteration** (`chunks_exact(in)` + `enumerate`, no
/// explicit `row*in+col` multiply — so a transposition bug in the oracle's flat
/// index cannot be mirrored), and the UE8M0 scales are applied **block-major**
/// (independent of the oracle's element-major `scale_index` arithmetic). Returns
/// the logical matrix, NOT the KN transpose.
fn reference_logical_block_fp8(layout: &PlanarLayout, packed: &[u8], scale: &[u8]) -> Vec<f32> {
    let (bs0, bs1) = layout.block_shape();
    let out = layout.out_features();
    let in_features = layout.in_features();
    let scale_cols = layout.scale_cols();

    // Pass 1: decode packed magnitudes via row-chunk iteration (no flat-index
    // multiply), giving a genuinely independent packed-weight indexing.
    let mut logical = vec![0.0f32; out * in_features];
    for (row, (row_bytes, row_out)) in packed
        .chunks_exact(in_features)
        .zip(logical.chunks_exact_mut(in_features))
        .enumerate()
    {
        debug_assert!(row < out);
        for (dst, &code) in row_out.iter_mut().zip(row_bytes) {
            *dst = decode_e4m3fn(code);
        }
    }

    // Pass 2: apply the UE8M0 block scale, iterating block-major.
    for block_row in 0..layout.scale_rows() {
        for block_col in 0..scale_cols {
            let scale_value = decode_e8m0_scale(scale[block_row * scale_cols + block_col]);
            let row_start = block_row * bs0;
            let col_start = block_col * bs1;
            for row in row_start..(row_start + bs0).min(out) {
                for col in col_start..(col_start + bs1).min(in_features) {
                    logical[row * in_features + col] *= scale_value;
                }
            }
        }
    }
    logical
}

/// Row-major `[out, in]`, built with the production 32-wide block helper
/// [`dequantize_fp4_e2m1_block`] (its own nibble ordering), independent of the
/// oracle's per-element nibble math.
fn reference_logical_fp4(layout: &PlanarLayout, packed: &[u8], scale: &[u8]) -> Vec<f32> {
    let out = layout.out_features();
    let in_features = layout.in_features();
    let blocks_per_row = in_features / FP4_MICROSCALE_BLOCK;
    let packed_row = in_features / FP4_PACK_FACTOR;
    let block_bytes = FP4_MICROSCALE_BLOCK / FP4_PACK_FACTOR;
    let mut logical = vec![0.0f32; out * in_features];
    let mut block_out = vec![0.0f32; FP4_MICROSCALE_BLOCK];
    for row in 0..out {
        for block in 0..blocks_per_row {
            let scale_exp = scale[row * blocks_per_row + block];
            let byte_start = row * packed_row + block * block_bytes;
            let bytes = &packed[byte_start..byte_start + block_bytes];
            dequantize_fp4_e2m1_block(scale_exp, bytes, &mut block_out).unwrap();
            let dst = row * in_features + block * FP4_MICROSCALE_BLOCK;
            logical[dst..dst + FP4_MICROSCALE_BLOCK].copy_from_slice(&block_out);
        }
    }
    logical
}

/// Transpose a row-major `[out, in]` logical weight into `[K = in, N = out]`
/// (`in`-major), matching the oracle's output orientation.
fn transpose_to_kn(logical: &[f32], out: usize, in_features: usize) -> Vec<f32> {
    let mut kn = vec![0.0f32; out * in_features];
    for row in 0..out {
        for col in 0..in_features {
            kn[col * out + row] = logical[row * in_features + col];
        }
    }
    kn
}

fn assert_kn_eq(oracle: &[f32], reference: &[f32]) {
    assert_eq!(oracle.len(), reference.len());
    for (index, (&a, &b)) in oracle.iter().zip(reference).enumerate() {
        assert!(
            (a - b).abs() <= f32::EPSILON * a.abs().max(1.0),
            "mismatch at {index}: oracle {a} != reference {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Format string / enum surface
// ---------------------------------------------------------------------------

#[test]
fn format_parse_roundtrip_and_metadata() {
    assert_eq!(
        PlanarBlockFormat::parse("block_fp8").unwrap(),
        PlanarBlockFormat::BlockFp8
    );
    assert_eq!(
        PlanarBlockFormat::parse("fp4_planar").unwrap(),
        PlanarBlockFormat::Fp4Planar
    );
    assert!(PlanarBlockFormat::parse("mxfp4").is_err());
    assert!(PlanarBlockFormat::parse("block_mxfp4").is_err());
    assert!(PlanarBlockFormat::parse("").is_err());

    assert_eq!(PlanarBlockFormat::BlockFp8.capability_str(), "block_fp8");
    assert_eq!(PlanarBlockFormat::Fp4Planar.capability_str(), "fp4_planar");
    assert_eq!(PlanarBlockFormat::BlockFp8.weight_dtype_name(), "F8_E4M3");
    assert_eq!(PlanarBlockFormat::Fp4Planar.weight_dtype_name(), "I8");
    assert_eq!(PlanarBlockFormat::BlockFp8.scale_dtype_name(), "F8_E8M0");
    assert_eq!(PlanarBlockFormat::Fp4Planar.scale_dtype_name(), "F8_E8M0");
    assert_eq!(PlanarBlockFormat::BlockFp8.pack_factor(), 1);
    assert_eq!(PlanarBlockFormat::Fp4Planar.pack_factor(), 2);
}

// ---------------------------------------------------------------------------
// Exhaustive decode / special values
// ---------------------------------------------------------------------------

#[test]
fn e2m1_all_sixteen_nibbles_via_oracle() {
    // Two logical inputs per byte: low nibble = even column, high = odd column.
    // Pack codes 0..16 across 8 bytes (16 columns), one output row, scale = 1.
    let layout = PlanarLayout::new(PlanarBlockFormat::Fp4Planar, 1, 32, 1, 32).unwrap();
    let mut packed = vec![0u8; 16];
    for code in 0u8..16 {
        let byte_index = (code as usize) / 2;
        if code.is_multiple_of(2) {
            packed[byte_index] |= code & 0x0f;
        } else {
            packed[byte_index] |= code << 4;
        }
    }
    // Columns 16..32 stay zero. Scale exponent 127 => scale 1.0.
    let scale = vec![127u8; 1];
    let kn = dequantize_planar_kn(&layout, &packed, &scale).unwrap();
    // kn[in_col * out(1) + 0] == logical[0, in_col].
    let expected_table = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    for (col, &expected) in expected_table.iter().enumerate() {
        assert_eq!(kn[col], expected, "E2M1 code {col}");
        assert_eq!(decode_e2m1(col as u8), expected);
    }
}

#[test]
fn e4m3_known_values_and_reserved_reject() {
    // Known E4M3 codes at scale 1 (exponent 127): 0x38 => 1.0, 0x40 => 2.0,
    // 0x3c => 1.5, 0x00 => 0.0.
    let layout = PlanarLayout::new(PlanarBlockFormat::BlockFp8, 1, 4, 1, 4).unwrap();
    let packed = vec![0x38, 0x40, 0x3c, 0x00];
    let scale = vec![127u8];
    let kn = dequantize_planar_kn(&layout, &packed, &scale).unwrap();
    assert_eq!(kn, vec![1.0, 2.0, 1.5, 0.0]);

    // Both reserved E4M3 NaN encodings must be rejected, not silently dequantized.
    for reserved in [0x7fu8, 0xff] {
        let bad = vec![reserved, 0x00, 0x00, 0x00];
        assert!(dequantize_planar_kn(&layout, &bad, &scale).is_err());
    }
}

#[test]
fn e8m0_scaling_and_reserved_reject() {
    // scale exponent e => 2^(e-127). 128 => 2.0, 126 => 0.5, 127 => 1.0.
    let layout = PlanarLayout::new(PlanarBlockFormat::BlockFp8, 1, 3, 1, 3).unwrap();
    let packed = vec![0x38, 0x38, 0x38]; // all == 1.0 before scaling
    for (exp, want) in [(128u8, 2.0f32), (126, 0.5), (127, 1.0)] {
        let kn = dequantize_planar_kn(&layout, &packed, &[exp; 1]).unwrap();
        assert_eq!(kn, vec![want, want, want]);
        assert_eq!(decode_e8m0_scale(exp), want);
    }
    // Reserved 0xff exponent must fail closed.
    assert!(dequantize_planar_kn(&layout, &packed, &[0xffu8; 1]).is_err());
}

// ---------------------------------------------------------------------------
// Layout geometry (shape-faithful tiny DeepSeek dims)
// ---------------------------------------------------------------------------

#[test]
fn block_fp8_geometry_multi_block() {
    // DeepSeek attention/shared-expert projections: 128x128 blocks.
    let layout = PlanarLayout::new(PlanarBlockFormat::BlockFp8, 256, 256, 128, 128).unwrap();
    assert_eq!(layout.packed_shape(), [256, 256]); // 1:1, no sub-byte packing
    assert_eq!(layout.scale_shape(), [2, 2]);
    assert_eq!(layout.scale_rows(), 2);
    assert_eq!(layout.scale_cols(), 2);
    assert_eq!(layout.packed_bytes().unwrap(), 256 * 256);
    assert_eq!(layout.scale_bytes().unwrap(), 4);
    assert_eq!(layout.dense_elements().unwrap(), 256 * 256);

    // Ragged (non-divisible) dims => ceil scale grid.
    let ragged = PlanarLayout::new(PlanarBlockFormat::BlockFp8, 5, 3, 2, 2).unwrap();
    assert_eq!(ragged.scale_shape(), [3, 2]);
    assert_eq!(ragged.packed_bytes().unwrap(), 15);
    assert_eq!(ragged.scale_bytes().unwrap(), 6);
}

#[test]
fn fp4_geometry_planar_packing() {
    // DeepSeek routed experts: I8-packed nibbles, block-32 micro-scale.
    let layout = PlanarLayout::new(PlanarBlockFormat::Fp4Planar, 4, 64, 1, 32).unwrap();
    assert_eq!(layout.packed_shape(), [4, 32]); // in/2 (two nibbles per byte)
    assert_eq!(layout.scale_shape(), [4, 2]); // in/32 micro-scales per row
    assert_eq!(layout.packed_bytes().unwrap(), 128);
    assert_eq!(layout.scale_bytes().unwrap(), 8);
    assert_eq!(layout.dense_elements().unwrap(), 256);
    assert_eq!(layout.block_shape(), (1, 32));
}

#[test]
fn layout_new_rejections() {
    // Empty logical dims.
    assert!(PlanarLayout::new(PlanarBlockFormat::BlockFp8, 0, 4, 1, 1).is_err());
    assert!(PlanarLayout::new(PlanarBlockFormat::BlockFp8, 4, 0, 1, 1).is_err());
    // block_fp8 zero block.
    assert!(PlanarLayout::new(PlanarBlockFormat::BlockFp8, 4, 4, 0, 1).is_err());
    assert!(PlanarLayout::new(PlanarBlockFormat::BlockFp8, 4, 4, 1, 0).is_err());
    // fp4 wrong block geometry.
    assert!(PlanarLayout::new(PlanarBlockFormat::Fp4Planar, 4, 64, 2, 32).is_err());
    assert!(PlanarLayout::new(PlanarBlockFormat::Fp4Planar, 4, 64, 1, 16).is_err());
    // fp4 odd logical in (cannot pack two nibbles per byte).
    assert!(PlanarLayout::new(PlanarBlockFormat::Fp4Planar, 4, 33, 1, 32).is_err());
    // fp4 in not divisible by micro-scale block.
    assert!(PlanarLayout::new(PlanarBlockFormat::Fp4Planar, 4, 48, 1, 32).is_err());
}

#[test]
fn overflow_dims_are_typed_errors_not_panics() {
    // new() only checks non-zero + block; byte-count math must fail closed.
    let layout = PlanarLayout::new(PlanarBlockFormat::BlockFp8, usize::MAX, 2, 1, 1).unwrap();
    assert!(layout.packed_bytes().is_err());
    assert!(layout.scale_bytes().is_err());
    assert!(layout.dense_elements().is_err());
}

// ---------------------------------------------------------------------------
// validate_tensors: fail-closed on any inconsistency
// ---------------------------------------------------------------------------

#[test]
fn validate_tensors_block_fp8() {
    let layout = PlanarLayout::new(PlanarBlockFormat::BlockFp8, 256, 128, 128, 128).unwrap();
    // Correct pair.
    layout
        .validate_tensors("F8_E4M3", &[256, 128], 256 * 128, "F8_E8M0", &[2, 1], 2)
        .unwrap();
    // Wrong weight dtype.
    assert!(
        layout
            .validate_tensors("F8_E5M2", &[256, 128], 256 * 128, "F8_E8M0", &[2, 1], 2)
            .is_err()
    );
    // Wrong scale dtype.
    assert!(
        layout
            .validate_tensors("F8_E4M3", &[256, 128], 256 * 128, "F16", &[2, 1], 2)
            .is_err()
    );
    // Wrong packed shape.
    assert!(
        layout
            .validate_tensors("F8_E4M3", &[128, 256], 256 * 128, "F8_E8M0", &[2, 1], 2)
            .is_err()
    );
    // Wrong scale grid.
    assert!(
        layout
            .validate_tensors("F8_E4M3", &[256, 128], 256 * 128, "F8_E8M0", &[2, 2], 4)
            .is_err()
    );
    // Wrong packed byte count.
    assert!(
        layout
            .validate_tensors("F8_E4M3", &[256, 128], 999, "F8_E8M0", &[2, 1], 2)
            .is_err()
    );
    // Wrong scale byte count.
    assert!(
        layout
            .validate_tensors("F8_E4M3", &[256, 128], 256 * 128, "F8_E8M0", &[2, 1], 99)
            .is_err()
    );
}

#[test]
fn validate_tensors_fp4_planar() {
    let layout = PlanarLayout::new(PlanarBlockFormat::Fp4Planar, 8, 64, 1, 32).unwrap();
    // Correct: packed [8,32], scale [8,2].
    layout
        .validate_tensors("I8", &[8, 32], 8 * 32, "F8_E8M0", &[8, 2], 8 * 2)
        .unwrap();
    // Logical shape passed as packed shape must be rejected (that's the trap).
    assert!(
        layout
            .validate_tensors("I8", &[8, 64], 8 * 32, "F8_E8M0", &[8, 2], 8 * 2)
            .is_err()
    );
    // Wrong weight dtype (must be I8, not U8).
    assert!(
        layout
            .validate_tensors("U8", &[8, 32], 8 * 32, "F8_E8M0", &[8, 2], 8 * 2)
            .is_err()
    );
    // Missing/short scale bank.
    assert!(
        layout
            .validate_tensors("I8", &[8, 32], 8 * 32, "F8_E8M0", &[8, 2], 0)
            .is_err()
    );
}

// ---------------------------------------------------------------------------
// Oracle vs independent reference parity
// ---------------------------------------------------------------------------

#[test]
fn oracle_matches_reference_block_fp8() {
    for (out, in_features, bs, seed) in [
        (256usize, 256usize, 128usize, 1u64),
        (3, 5, 2, 2),
        (5, 3, 2, 3),
        (130, 129, 128, 4),
    ] {
        let layout =
            PlanarLayout::new(PlanarBlockFormat::BlockFp8, out, in_features, bs, bs).unwrap();
        let (packed, scale) = block_fp8_fixture(out, in_features, bs, seed);
        let oracle = dequantize_planar_kn(&layout, &packed, &scale).unwrap();
        let reference = transpose_to_kn(
            &reference_logical_block_fp8(&layout, &packed, &scale),
            out,
            in_features,
        );
        assert_kn_eq(&oracle, &reference);
    }
}

#[test]
fn oracle_matches_reference_fp4() {
    for (out, in_features, seed) in [(4usize, 64usize, 10u64), (7, 96, 11), (2, 32, 12)] {
        let layout =
            PlanarLayout::new(PlanarBlockFormat::Fp4Planar, out, in_features, 1, 32).unwrap();
        let (packed, scale) = fp4_fixture(out, in_features, seed);
        let oracle = dequantize_planar_kn(&layout, &packed, &scale).unwrap();
        let reference = transpose_to_kn(
            &reference_logical_fp4(&layout, &packed, &scale),
            out,
            in_features,
        );
        assert_kn_eq(&oracle, &reference);
    }
}

#[test]
fn dequantize_rejects_wrong_length_buffers() {
    let layout = PlanarLayout::new(PlanarBlockFormat::BlockFp8, 4, 4, 2, 2).unwrap();
    let (packed, scale) = block_fp8_fixture(4, 4, 2, 1);
    // Truncated packed / scale must be typed errors, never OOB reads.
    assert!(dequantize_planar_kn(&layout, &packed[..packed.len() - 1], &scale).is_err());
    assert!(dequantize_planar_kn(&layout, &packed, &scale[..scale.len() - 1]).is_err());
}

// ---------------------------------------------------------------------------
// Matmul oracle parity
// ---------------------------------------------------------------------------

fn naive_matmul(a: &[f32], m_rows: usize, weight_kn: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m_rows * n];
    for row in 0..m_rows {
        for col in 0..n {
            let mut acc = 0.0f32;
            for depth in 0..k {
                acc += a[row * k + depth] * weight_kn[depth * n + col];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

#[test]
fn matmul_oracle_parity_block_fp8() {
    let (out, in_features, bs) = (16usize, 8usize, 4usize);
    let layout = PlanarLayout::new(PlanarBlockFormat::BlockFp8, out, in_features, bs, bs).unwrap();
    let (packed, scale) = block_fp8_fixture(out, in_features, bs, 7);
    let m_rows = 3;
    let mut rng = Lcg::new(99);
    let a: Vec<f32> = (0..m_rows * in_features)
        .map(|_| (rng.next_u8() as f32 - 128.0) / 64.0)
        .collect();
    let got = planar_block_matmul(&a, m_rows, &layout, &packed, &scale).unwrap();
    let weight_kn = dequantize_planar_kn(&layout, &packed, &scale).unwrap();
    let want = naive_matmul(&a, m_rows, &weight_kn, in_features, out);
    assert_kn_eq(&got, &want);
    // Shape mismatch on A must be a typed error.
    assert!(planar_block_matmul(&a[..a.len() - 1], m_rows, &layout, &packed, &scale).is_err());
}

#[test]
fn matmul_oracle_parity_fp4() {
    let (out, in_features) = (8usize, 32usize);
    let layout = PlanarLayout::new(PlanarBlockFormat::Fp4Planar, out, in_features, 1, 32).unwrap();
    let (packed, scale) = fp4_fixture(out, in_features, 21);
    let m_rows = 2;
    let mut rng = Lcg::new(123);
    let a: Vec<f32> = (0..m_rows * in_features)
        .map(|_| (rng.next_u8() as f32 - 128.0) / 64.0)
        .collect();
    let got = planar_block_matmul(&a, m_rows, &layout, &packed, &scale).unwrap();
    let weight_kn = dequantize_planar_kn(&layout, &packed, &scale).unwrap();
    let want = naive_matmul(&a, m_rows, &weight_kn, in_features, out);
    assert_kn_eq(&got, &want);
}

// ---------------------------------------------------------------------------
// Routed-expert bank (byte-exact, expert-major)
// ---------------------------------------------------------------------------

#[test]
fn expert_bank_slices_and_dequant_are_byte_exact() {
    let layout = PlanarLayout::new(PlanarBlockFormat::Fp4Planar, 4, 64, 1, 32).unwrap();
    let experts: Vec<(Vec<u8>, Vec<u8>)> = (0..6)
        .map(|expert| fp4_fixture(4, 64, 200 + expert as u64))
        .collect();
    let refs: Vec<(&[u8], &[u8])> = experts
        .iter()
        .map(|(p, s)| (p.as_slice(), s.as_slice()))
        .collect();
    let bank = PlanarExpertBank::stack(layout, &refs).unwrap();

    assert_eq!(bank.num_experts(), 6);
    assert_eq!(bank.layout(), &layout);
    for (expert, (packed, scale)) in experts.iter().enumerate() {
        assert_eq!(bank.expert_packed(expert).unwrap(), packed.as_slice());
        assert_eq!(bank.expert_scale(expert).unwrap(), scale.as_slice());
        let via_bank = bank.dequantize_expert_kn(expert).unwrap();
        let direct = dequantize_planar_kn(&layout, packed, scale).unwrap();
        assert_eq!(via_bank, direct);
    }
    // Out-of-range expert is a typed error.
    assert!(bank.expert_packed(6).is_err());
    assert!(bank.expert_scale(6).is_err());
    assert!(bank.dequantize_expert_kn(6).is_err());
}

#[test]
fn expert_bank_rejects_ragged_and_empty() {
    let layout = PlanarLayout::new(PlanarBlockFormat::Fp4Planar, 4, 64, 1, 32).unwrap();
    let (good_packed, good_scale) = fp4_fixture(4, 64, 1);
    let short_packed = &good_packed[..good_packed.len() - 1];
    let short_scale = &good_scale[..good_scale.len() - 1];

    // Empty bank.
    assert!(PlanarExpertBank::stack(layout, &[]).is_err());
    // Ragged packed payload.
    assert!(
        PlanarExpertBank::stack(
            layout,
            &[
                (good_packed.as_slice(), good_scale.as_slice()),
                (short_packed, good_scale.as_slice())
            ],
        )
        .is_err()
    );
    // Ragged scale payload.
    assert!(
        PlanarExpertBank::stack(
            layout,
            &[
                (good_packed.as_slice(), good_scale.as_slice()),
                (good_packed.as_slice(), short_scale)
            ],
        )
        .is_err()
    );
}

// ---------------------------------------------------------------------------
// Per-projection mixed banks (fc1/fc2/fc3 differ in shape)
// ---------------------------------------------------------------------------

#[test]
fn per_projection_mixed_planar_banks() {
    // Routed expert projections in DeepSeek-V4 differ in [out, in]:
    // w1/w3 = [moe_intermediate, hidden], w2 = [hidden, moe_intermediate].
    // Use shape-faithful tiny dims (hidden=64, moe_intermediate=32).
    let w1 = PlanarLayout::new(PlanarBlockFormat::Fp4Planar, 32, 64, 1, 32).unwrap();
    let w2 = PlanarLayout::new(PlanarBlockFormat::Fp4Planar, 64, 32, 1, 32).unwrap();
    let w3 = PlanarLayout::new(PlanarBlockFormat::Fp4Planar, 32, 64, 1, 32).unwrap();
    assert_ne!(w1.packed_shape(), w2.packed_shape());
    for (index, layout) in [w1, w2, w3].into_iter().enumerate() {
        let (packed, scale) = fp4_fixture(
            layout.out_features(),
            layout.in_features(),
            500 + index as u64,
        );
        layout
            .validate_tensors(
                "I8",
                &layout.packed_shape(),
                packed.len(),
                "F8_E8M0",
                &layout.scale_shape(),
                scale.len(),
            )
            .unwrap();
        let oracle = dequantize_planar_kn(&layout, &packed, &scale).unwrap();
        let reference = transpose_to_kn(
            &reference_logical_fp4(&layout, &packed, &scale),
            layout.out_features(),
            layout.in_features(),
        );
        assert_kn_eq(&oracle, &reference);
    }
}

// ---------------------------------------------------------------------------
// Hand-computed end-to-end anchors
// ---------------------------------------------------------------------------

#[test]
fn known_value_end_to_end_block_fp8() {
    // 2x2 logical, single 2x2 block. Codes: 0x38=>1.0, 0x40=>2.0, 0x3c=>1.5,
    // 0x00=>0.0. Scale exponent 128 => 2.0. Expected logical:
    //   [[2.0, 4.0], [3.0, 0.0]]
    // KN (in-major) transpose: [2.0, 3.0, 4.0, 0.0].
    let layout = PlanarLayout::new(PlanarBlockFormat::BlockFp8, 2, 2, 2, 2).unwrap();
    let packed = vec![0x38, 0x40, 0x3c, 0x00];
    let scale = vec![128u8];
    let kn = dequantize_planar_kn(&layout, &packed, &scale).unwrap();
    assert_eq!(kn, vec![2.0, 3.0, 4.0, 0.0]);
}

#[test]
fn known_value_end_to_end_fp4() {
    // 1 row, 32 logical inputs, one micro-block. First byte 0x21 => low nibble
    // 1 (=0.5), high nibble 2 (=1.0). Scale exponent 129 => 4.0.
    // Expected logical[0,0] = 0.5*4 = 2.0, logical[0,1] = 1.0*4 = 4.0.
    let layout = PlanarLayout::new(PlanarBlockFormat::Fp4Planar, 1, 32, 1, 32).unwrap();
    let mut packed = vec![0u8; 16];
    packed[0] = 0x21;
    let scale = vec![129u8];
    let kn = dequantize_planar_kn(&layout, &packed, &scale).unwrap();
    // out=1, so kn[in_col] == logical[0, in_col].
    assert_eq!(kn[0], 2.0);
    assert_eq!(kn[1], 4.0);
    for value in kn.iter().skip(2) {
        assert_eq!(*value, 0.0);
    }
}
