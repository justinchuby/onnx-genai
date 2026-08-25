//! Host-mirror parity tests for the planar CUDA decode against the CPU oracle.
//!
//! The launched device path is exercised by the GPU integration test
//! `tests/planar_block_decode_gpu.rs` (compiles + launches the NVRTC kernel on
//! real hardware and compares to the CPU oracle). These *host* tests need no GPU:
//! exactly as [`crate::kernels::block_quant`] verifies its device quantizers
//! against the CPU `block_dequant` reference, we test the **host mirror** (a
//! bit-identical Rust transcription of the device arithmetic) against the
//! independently-vetted CPU planar oracle
//! [`onnx_runtime_ep_cpu::kernels::planar_block_quant`], plus the pure host
//! surfaces (geometry/aux length validation, dtype selection, capability
//! strings).
//!
//! * The mirror's per-element decode is compared **bit-exactly** to
//!   `dequantize_planar_kn` (same values, same indexing → no accumulation).
//! * The mirror's linear kernel is compared to `planar_block_matmul` within a
//!   tight float tolerance (identical per-output `k`-ascending accumulation
//!   order, so agreement is expected to the ULP; the tolerance only guards the
//!   `a == 0` skip the oracle takes).
//!
//! The device C and the mirror are hand-written to the same formulae and kept
//! adjacent in [`super`] so their correspondence is auditable by inspection.

use super::*;
use onnx_runtime_ep_cpu::kernels::planar_block_quant::{
    FP4_MICROSCALE_BLOCK, FP4_PACK_FACTOR, PlanarBlockFormat, PlanarLayout, dequantize_planar_kn,
    planar_block_matmul,
};
use onnx_runtime_ir::DataType;

// ---------------------------------------------------------------------------
// Deterministic fixtures (mirror of the CPU test generators)
// ---------------------------------------------------------------------------

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

/// Map the two reserved E4M3 NaN encodings to a benign zero so block-FP8
/// fixtures never trip the fail-closed reserved-code guard in the oracle.
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

fn activations(m_rows: usize, in_features: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..m_rows * in_features)
        .map(|_| {
            // Small signed values, some exact zeros to exercise the oracle's
            // `a == 0` skip vs. the mirror's unconditional `0 * w` add.
            let byte = rng.next_u8();
            if byte < 32 {
                0.0
            } else {
                (i16::from(byte) - 128) as f32 / 64.0
            }
        })
        .collect()
}

/// Build the dense `[K = in, N = out]` weight straight from the host mirror's
/// per-element decoders (no accumulation), matching the oracle's orientation.
fn mirror_dense_kn(
    format: PlanarBlockFormat,
    out_features: usize,
    in_features: usize,
    bs0: usize,
    bs1: usize,
    packed: &[u8],
    scale: &[u8],
) -> Vec<f32> {
    let mut weight_kn = vec![0.0f32; in_features * out_features];
    for out_row in 0..out_features {
        for in_col in 0..in_features {
            let value = match format {
                PlanarBlockFormat::BlockFp8 => {
                    mirror_bf8_element(packed, scale, in_features, bs0, bs1, out_row, in_col)
                }
                PlanarBlockFormat::Fp4Planar => {
                    mirror_fp4_element(packed, scale, in_features, out_row, in_col)
                }
            };
            weight_kn[in_col * out_features + out_row] = value;
        }
    }
    weight_kn
}

// ---------------------------------------------------------------------------
// Known-value anchors for the decode primitives
// ---------------------------------------------------------------------------

#[test]
fn mirror_e8m0_scale_anchors() {
    assert_eq!(mirror_e8m0_scale(127), 1.0);
    assert_eq!(mirror_e8m0_scale(126), 0.5);
    assert_eq!(mirror_e8m0_scale(128), 2.0);
    assert_eq!(mirror_e8m0_scale(0), 2.0f32.powi(-127));
    assert!(mirror_e8m0_scale(0xff).is_nan());
}

#[test]
fn mirror_e2m1_anchors() {
    assert_eq!(mirror_e2m1(0), 0.0);
    assert_eq!(mirror_e2m1(1), 0.5);
    assert_eq!(mirror_e2m1(7), 6.0);
    assert_eq!(mirror_e2m1(0x0f), -6.0);
    // Index 8 carries the sign bit of zero: it must be -0.0, not +0.0.
    assert!(mirror_e2m1(8).is_sign_negative());
    assert_eq!(mirror_e2m1(8), 0.0);
    // Only the low nibble is significant.
    assert_eq!(mirror_e2m1(0xf7), mirror_e2m1(0x07));
}

#[test]
fn mirror_e4m3_anchors() {
    assert_eq!(mirror_e4m3(0x00), 0.0);
    assert_eq!(mirror_e4m3(0x38), 1.0); // exp=7, mant=0 -> 1.0
    assert_eq!(mirror_e4m3(0x3c), 1.5); // exp=7, mant=4 -> 1.5
    assert_eq!(mirror_e4m3(0xb8), -1.0); // sign set
    assert!(mirror_e4m3(0xff).is_nan()); // reserved E4M3 NaN
}

/// The mirror primitives must equal the CPU `block_dequant` primitives on every
/// byte — this is what makes the mirror a faithful stand-in for the oracle.
#[test]
fn mirror_primitives_match_cpu_over_all_bytes() {
    use onnx_runtime_ep_cpu::kernels::block_dequant::{
        decode_e2m1, decode_e4m3fn, decode_e8m0_scale,
    };
    for code in 0u16..=255 {
        let code = code as u8;

        let m = mirror_e2m1(code);
        let c = decode_e2m1(code);
        assert_eq!(m.to_bits(), c.to_bits(), "e2m1 code 0x{code:02x}");

        let m = mirror_e4m3(code);
        let c = decode_e4m3fn(code);
        if m.is_nan() {
            assert!(c.is_nan(), "e4m3 code 0x{code:02x}");
        } else {
            assert_eq!(m.to_bits(), c.to_bits(), "e4m3 code 0x{code:02x}");
        }

        let m = mirror_e8m0_scale(code);
        let c = decode_e8m0_scale(code);
        if m.is_nan() {
            assert!(c.is_nan(), "e8m0 code 0x{code:02x}");
        } else {
            assert_eq!(m.to_bits(), c.to_bits(), "e8m0 code 0x{code:02x}");
        }
    }
}

// ---------------------------------------------------------------------------
// Dense-weight parity: mirror element decode == CPU oracle, bit-for-bit
// ---------------------------------------------------------------------------

fn assert_dense_bit_exact(
    format: PlanarBlockFormat,
    out: usize,
    in_features: usize,
    bs0: usize,
    bs1: usize,
    packed: &[u8],
    scale: &[u8],
) {
    let layout = PlanarLayout::new(format, out, in_features, bs0, bs1).unwrap();
    let oracle = dequantize_planar_kn(&layout, packed, scale).unwrap();
    let mirror = mirror_dense_kn(format, out, in_features, bs0, bs1, packed, scale);
    assert_eq!(mirror.len(), oracle.len());
    for (i, (m, o)) in mirror.iter().zip(&oracle).enumerate() {
        assert_eq!(
            m.to_bits(),
            o.to_bits(),
            "planar {format:?} dense weight element {i} differs: mirror {m} vs oracle {o}",
        );
    }
}

#[test]
fn block_fp8_dense_matches_oracle_bit_exact() {
    // Shape-faithful tiny DeepSeek-V4 dims (hidden 4096 -> 64, block 128 -> 32)
    // with a partial trailing block on both axes.
    for &(out, in_features, bs) in &[(64usize, 64usize, 32usize), (40, 96, 32), (32, 32, 128)] {
        let (packed, scale) = block_fp8_fixture(out, in_features, bs, 0x51ce);
        assert_dense_bit_exact(
            PlanarBlockFormat::BlockFp8,
            out,
            in_features,
            bs,
            bs,
            &packed,
            &scale,
        );
    }
}

#[test]
fn fp4_planar_dense_matches_oracle_bit_exact() {
    for &(out, in_features) in &[(64usize, 64usize), (32, 128), (48, 96)] {
        let (packed, scale) = fp4_fixture(out, in_features, 0xf00d);
        assert_dense_bit_exact(
            PlanarBlockFormat::Fp4Planar,
            out,
            in_features,
            1,
            FP4_MICROSCALE_BLOCK,
            &packed,
            &scale,
        );
    }
}

// ---------------------------------------------------------------------------
// Linear-kernel parity: mirror planar_linear_f32 == CPU matmul oracle
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn assert_matmul_close(
    format: PlanarBlockFormat,
    m_rows: usize,
    out: usize,
    in_features: usize,
    bs0: usize,
    bs1: usize,
    packed: &[u8],
    scale: &[u8],
    a: &[f32],
) {
    let layout = PlanarLayout::new(format, out, in_features, bs0, bs1).unwrap();
    let oracle = planar_block_matmul(a, m_rows, &layout, packed, scale).unwrap();
    let format_id = match format {
        PlanarBlockFormat::BlockFp8 => PLANAR_FORMAT_BLOCK_FP8,
        PlanarBlockFormat::Fp4Planar => PLANAR_FORMAT_FP4_PLANAR,
    };
    let mirror = mirror_planar_linear_f32(
        a,
        packed,
        scale,
        m_rows,
        in_features,
        out,
        format_id,
        bs0,
        bs1,
    );
    assert_eq!(mirror.len(), oracle.len());
    for (i, (m, o)) in mirror.iter().zip(&oracle).enumerate() {
        let diff = (m - o).abs();
        let tol = 1e-4 * o.abs().max(1.0);
        assert!(
            diff <= tol,
            "planar {format:?} matmul output {i} differs: mirror {m} vs oracle {o} (diff {diff} > tol {tol})",
        );
    }
}

#[test]
fn block_fp8_matmul_matches_oracle() {
    let (out, in_features, bs, m_rows) = (48usize, 96usize, 32usize, 5usize);
    let (packed, scale) = block_fp8_fixture(out, in_features, bs, 0xa11ce);
    let a = activations(m_rows, in_features, 0xbeef);
    assert_matmul_close(
        PlanarBlockFormat::BlockFp8,
        m_rows,
        out,
        in_features,
        bs,
        bs,
        &packed,
        &scale,
        &a,
    );
}

#[test]
fn fp4_planar_matmul_matches_oracle() {
    let (out, in_features, m_rows) = (48usize, 96usize, 5usize);
    let (packed, scale) = fp4_fixture(out, in_features, 0xc0ffee);
    let a = activations(m_rows, in_features, 0xd00d);
    assert_matmul_close(
        PlanarBlockFormat::Fp4Planar,
        m_rows,
        out,
        in_features,
        1,
        FP4_MICROSCALE_BLOCK,
        &packed,
        &scale,
        &a,
    );
}

// ---------------------------------------------------------------------------
// Device-source structural guard + claim-boundary documentation
// ---------------------------------------------------------------------------

/// The NVRTC source must define every symbol the launch path names, so a rename
/// here can never silently diverge from the host mirror or the entry constants.
#[test]
fn device_source_declares_required_symbols() {
    for needle in [
        PLANAR_LINEAR_ENTRY,
        "planar_e8m0_scale",
        "planar_e2m1",
        "planar_e4m3",
        "planar_bf8_element",
        "planar_fp4_element",
        "planar_e2m1_lut",
        "planar_to_f32",
        "planar_store",
        "planar_linear_impl",
        "#include <cuda_fp16.h>",
        "#include <cuda_bf16.h>",
    ] {
        assert!(
            PLANAR_BLOCK_DECODE_CUH.contains(needle),
            "planar device source is missing `{needle}`"
        );
    }
    for dtype in PlanarActivationDtype::all() {
        assert!(
            PLANAR_BLOCK_DECODE_CUH.contains(&format!("__global__ void {}", dtype.entry())),
            "planar device source is missing entry `{}`",
            dtype.entry()
        );
    }
}

/// The three precision entry points must be distinct and stable.
#[test]
fn planar_activation_dtype_entries_are_distinct() {
    assert_eq!(PlanarActivationDtype::F32.entry(), PLANAR_LINEAR_ENTRY);
    let entries: Vec<&str> = PlanarActivationDtype::all()
        .iter()
        .map(|dtype| dtype.entry())
        .collect();
    assert_eq!(
        entries,
        [
            "planar_linear_f32",
            "planar_linear_f16",
            "planar_linear_bf16"
        ]
    );
    assert_eq!(
        PlanarActivationDtype::from_data_type(DataType::Float32).unwrap(),
        PlanarActivationDtype::F32
    );
    assert_eq!(
        PlanarActivationDtype::from_data_type(DataType::Float16).unwrap(),
        PlanarActivationDtype::F16
    );
    assert_eq!(
        PlanarActivationDtype::from_data_type(DataType::BFloat16).unwrap(),
        PlanarActivationDtype::Bf16
    );
    assert!(PlanarActivationDtype::from_data_type(DataType::Int8).is_err());
}

/// Advertised planar matmul capability strings must exactly match the format
/// names the routed-MoE claim gate recognises (so the two never drift).
#[test]
fn planar_matmul_capability_strings_are_stable() {
    assert_eq!(planar_matmul_capable_formats(), ["block_fp8", "fp4_planar"]);
}

/// `block_fp8` geometry yields the exact packed/scale byte lengths and accepts a
/// matching set of tensors, and typed-rejects ragged banks / overflowing dims.
#[test]
fn block_fp8_length_validation() {
    let dims = PlanarLinearDims {
        format: PLANAR_FORMAT_BLOCK_FP8,
        m_rows: 3,
        in_features: 128,
        out_features: 64,
        bs0: 128,
        bs1: 128,
    };
    let lengths = dims.expected_lengths().unwrap();
    assert_eq!(lengths.packed_bytes, 64 * 128);
    // ceil(64/128) * ceil(128/128) == 1 * 1.
    assert_eq!(lengths.scale_bytes, 1);
    validate_planar_linear(
        &dims,
        3 * 128,
        lengths.packed_bytes,
        lengths.scale_bytes,
        lengths.output_elems,
    )
    .unwrap();

    // Truncated aux scale is rejected.
    assert!(
        validate_planar_linear(
            &dims,
            3 * 128,
            lengths.packed_bytes,
            0,
            lengths.output_elems
        )
        .is_err()
    );
    // Under-sized output buffer is rejected.
    assert!(
        validate_planar_linear(
            &dims,
            3 * 128,
            lengths.packed_bytes,
            lengths.scale_bytes,
            lengths.output_elems - 1
        )
        .is_err()
    );
    // Wrong packed length is rejected.
    assert!(
        validate_planar_linear(
            &dims,
            3 * 128,
            lengths.packed_bytes - 1,
            lengths.scale_bytes,
            lengths.output_elems
        )
        .is_err()
    );
    // Wrong activation length is rejected.
    assert!(
        validate_planar_linear(
            &dims,
            3 * 128 + 1,
            lengths.packed_bytes,
            lengths.scale_bytes,
            lengths.output_elems
        )
        .is_err()
    );

    // Zero block size is rejected.
    let bad = PlanarLinearDims { bs1: 0, ..dims };
    assert!(bad.expected_lengths().is_err());
}

/// A non-128-aligned `block_fp8` scale grid is still exact via ceil-division.
#[test]
fn block_fp8_ragged_scale_grid_is_exact() {
    let dims = PlanarLinearDims {
        format: PLANAR_FORMAT_BLOCK_FP8,
        m_rows: 1,
        in_features: 130,
        out_features: 130,
        bs0: 128,
        bs1: 128,
    };
    let lengths = dims.expected_lengths().unwrap();
    // ceil(130/128) == 2 in each dim.
    assert_eq!(lengths.scale_bytes, 2 * 2);
    assert_eq!(lengths.packed_bytes, 130 * 130);
}

/// `fp4_planar` geometry yields packed `[out, in/2]` and scale `[out, in/32]`,
/// and typed-rejects odd or non-block-aligned contractions.
#[test]
fn fp4_planar_length_validation() {
    let dims = PlanarLinearDims {
        format: PLANAR_FORMAT_FP4_PLANAR,
        m_rows: 2,
        in_features: 64,
        out_features: 16,
        bs0: 0,
        bs1: 0,
    };
    let lengths = dims.expected_lengths().unwrap();
    assert_eq!(lengths.packed_bytes, 16 * (64 / 2));
    assert_eq!(lengths.scale_bytes, 16 * (64 / 32));
    validate_planar_linear(
        &dims,
        2 * 64,
        lengths.packed_bytes,
        lengths.scale_bytes,
        lengths.output_elems,
    )
    .unwrap();

    // Odd contraction (not packable into nibbles) is rejected.
    let odd = PlanarLinearDims {
        in_features: 63,
        ..dims
    };
    assert!(odd.expected_lengths().is_err());
    // Even but not block-32-aligned is rejected.
    let unaligned = PlanarLinearDims {
        in_features: 48,
        ..dims
    };
    assert!(unaligned.expected_lengths().is_err());
}
