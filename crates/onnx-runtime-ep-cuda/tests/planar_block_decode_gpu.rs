//! On-device parity for the DeepSeek-V4 planar B2 weight formats
//! (`block_fp8`, `fp4_planar`).
//!
//! This is the hardware proof that the launched NVRTC planar-linear kernel
//! (`onnx_runtime_ep_cuda::launch_planar_linear`) decodes the exact on-disk byte
//! layout — packed weights + UE8M0 aux scales — and contracts it against f32 /
//! f16 / bf16 activations, matching the vetted CPU oracle
//! `onnx_runtime_ep_cpu::kernels::planar_block_quant::planar_block_matmul` to
//! within the output dtype's precision. It replaces the earlier host-mirror-only
//! proof (which could not compile or launch the kernel).
//!
//! Every test is `#[cfg_attr(not(feature = "gpu-tests"), ignore)]`d so a CPU-only
//! run leaves them ignored; enable `--features gpu-tests` on a CUDA runner. The
//! device is selected by `CUDA_VISIBLE_DEVICES` — pin an idle GPU before running
//! (the measurement probe additionally refuses a non-idle device).
//!
//! Coverage:
//! * `block_fp8` and `fp4_planar` × {f32, f16, bf16} on shape-faithful DeepSeek
//!   layer dims plus small/ragged shapes;
//! * exhaustive small codepoints (every E2M1 nibble, a sweep of E4M3 bytes);
//! * multi-request / shape-change on one device (NVRTC cache repopulation,
//!   stable buffers);
//! * invalid aux / OOB geometry → typed reject (no launch);
//! * CUDA-graph capture + ≥3 replay parity (warmed, no in-capture alloc/sync);
//! * an `#[ignore]`d measurement probe (8 s ramp, idle check, n≥3, kernel batch
//!   time and host enqueue timed separately, first-shape recheck; no tok/s).

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::uninlined_format_args
)]

use half::{bf16, f16};
use onnx_runtime_ep_api::{DeviceBuffer, DeviceGraphSlot, ExecutionProvider};
use onnx_runtime_ep_cpu::kernels::planar_block_quant::{
    FP4_MICROSCALE_BLOCK, FP4_PACK_FACTOR, PlanarBlockFormat, PlanarLayout, planar_block_matmul,
};
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ep_cuda::vmm_allocator::CudaVmmAllocator;
use onnx_runtime_ep_cuda::{
    CudaExecutionProvider, PLANAR_FORMAT_BLOCK_FP8, PLANAR_FORMAT_FP4_PLANAR,
    PlanarActivationDtype, PlanarLinearDims, admit_planar_linear, launch_planar_linear,
    planar_matmul_capable_formats, warm_planar_linear,
};
use onnx_runtime_memory_governor::{DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryRole};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn require_cuda() -> Arc<CudaExecutionProvider> {
    match std::panic::catch_unwind(|| CudaExecutionProvider::new(selected_cuda_ordinal())) {
        Ok(Ok(ep)) => Arc::new(ep),
        Ok(Err(error)) => panic!(
            "CUDA test requires CUDA device/runtime; CPU-only runs must leave this test ignored: {error}"
        ),
        Err(_) => panic!(
            "CUDA test requires CUDA runtime libraries; CPU-only runs must leave this test ignored"
        ),
    }
}

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
/// fixtures never trip the oracle's fail-closed reserved-code guard.
fn finite_e4m3(code: u8) -> u8 {
    if code & 0x7f == 0x7f { 0x00 } else { code }
}

/// A random-but-magnitude-bounded E4M3 byte: random sign + mantissa, exponent
/// clamped to `1..=7` so the decoded magnitude stays `< 2` (and never hits the
/// reserved `0x7f`/`0xff` NaN). Keeps random-fixture outputs inside f16's finite
/// range while still exercising every mantissa/sign and a spread of exponents;
/// the *full* E4M3 byte range (incl. the 448 max) is swept exactly by
/// `exhaustive_small_codepoints` and the host bit-exact tests.
fn bounded_e4m3(byte: u8) -> u8 {
    let sign = byte & 0x80;
    let mant = byte & 0x07;
    let exp = 1 + (byte >> 3) % 7; // 1..=7 -> magnitude < 2
    sign | (exp << 3) | mant
}

/// UE8M0 exponents in a tight band around 1.0 (2^-2..2^2) so decoded weights
/// stay in a sane range and never hit the reserved `0xff`.
fn benign_scale(byte: u8) -> u8 {
    125 + (byte % 5)
}

fn block_fp8_fixture(out: usize, in_features: usize, bs: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let mut rng = Lcg::new(seed);
    let packed: Vec<u8> = (0..out * in_features)
        .map(|_| bounded_e4m3(rng.next_u8()))
        .collect();
    let scale: Vec<u8> = (0..out.div_ceil(bs) * in_features.div_ceil(bs))
        .map(|_| benign_scale(rng.next_u8()))
        .collect();
    (packed, scale)
}

fn fp4_fixture(out: usize, in_features: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let mut rng = Lcg::new(seed);
    let packed: Vec<u8> = (0..out * (in_features / FP4_PACK_FACTOR))
        .map(|_| rng.next_u8())
        .collect();
    let scale: Vec<u8> = (0..out * (in_features / FP4_MICROSCALE_BLOCK))
        .map(|_| benign_scale(rng.next_u8()))
        .collect();
    (packed, scale)
}

fn activations(m_rows: usize, in_features: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..m_rows * in_features)
        .map(|_| {
            let byte = rng.next_u8();
            if byte < 32 {
                0.0
            } else {
                (i16::from(byte) - 128) as f32 / 64.0
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// dtype round-trip (quantize activations so the oracle sees exactly what the
// device kernel loads) + output decode
// ---------------------------------------------------------------------------

/// Quantize `a` to `dtype`'s device bytes and return the matching dequantized
/// f32 the CPU oracle should contract, so the only remaining difference from the
/// device result is the kernel's f32 accumulation + final store rounding.
fn quantize_activation(a: &[f32], dtype: PlanarActivationDtype) -> (Vec<u8>, Vec<f32>) {
    match dtype {
        PlanarActivationDtype::F32 => {
            let bytes = a.iter().flat_map(|v| v.to_le_bytes()).collect();
            (bytes, a.to_vec())
        }
        PlanarActivationDtype::F16 => {
            let q: Vec<f16> = a.iter().map(|&v| f16::from_f32(v)).collect();
            let bytes = q.iter().flat_map(|v| v.to_le_bytes()).collect();
            let deq = q.iter().map(|v| v.to_f32()).collect();
            (bytes, deq)
        }
        PlanarActivationDtype::Bf16 => {
            let q: Vec<bf16> = a.iter().map(|&v| bf16::from_f32(v)).collect();
            let bytes = q.iter().flat_map(|v| v.to_le_bytes()).collect();
            let deq = q.iter().map(|v| v.to_f32()).collect();
            (bytes, deq)
        }
    }
}

fn dtype_bytes(dtype: PlanarActivationDtype) -> usize {
    match dtype {
        PlanarActivationDtype::F32 => 4,
        PlanarActivationDtype::F16 | PlanarActivationDtype::Bf16 => 2,
    }
}

fn decode_output(bytes: &[u8], dtype: PlanarActivationDtype) -> Vec<f32> {
    match dtype {
        PlanarActivationDtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        PlanarActivationDtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        PlanarActivationDtype::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
    }
}

/// Relative+absolute tolerance appropriate for the output store precision.
fn tolerance(dtype: PlanarActivationDtype, magnitude: f32) -> f32 {
    let (rel, abs) = match dtype {
        PlanarActivationDtype::F32 => (2e-3, 1e-4),
        PlanarActivationDtype::F16 => (4e-3, 1e-3),
        PlanarActivationDtype::Bf16 => (3e-2, 8e-3),
    };
    rel * magnitude.abs().max(1.0) + abs
}

// ---------------------------------------------------------------------------
// Device launch + parity
// ---------------------------------------------------------------------------

fn upload(ep: &CudaExecutionProvider, bytes: &[u8]) -> DeviceBuffer {
    let buffer = ep.allocate(bytes.len().max(1), 256).unwrap();
    if !bytes.is_empty() {
        // SAFETY: `buffer` is a fresh device allocation at least `bytes.len()`
        // wide; the copy stays in bounds.
        unsafe { ep.runtime().htod(bytes, cuptr(buffer.as_ptr())).unwrap() };
    }
    buffer
}

fn download(ep: &CudaExecutionProvider, buffer: &DeviceBuffer, len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    // SAFETY: `buffer` is at least `len` bytes wide (allocated so below).
    unsafe {
        ep.runtime()
            .dtoh(&mut bytes, cuptr(buffer.as_ptr()))
            .unwrap()
    };
    bytes
}

/// Run the planar linear on device and return the decoded f32 output.
fn run_planar_gpu(
    ep: &Arc<CudaExecutionProvider>,
    dtype: PlanarActivationDtype,
    dims: &PlanarLinearDims,
    a_bytes: &[u8],
    packed: &[u8],
    scale: &[u8],
) -> Vec<f32> {
    let out_elems = dims.m_rows * dims.out_features;
    let admission = admit_planar_linear(
        ep,
        dims,
        dims.m_rows * dims.in_features,
        packed,
        scale,
        out_elems,
    )
    .unwrap();

    let out_bytes = out_elems * dtype_bytes(dtype);

    let a_buf = upload(ep, a_bytes);
    let mut out_buf = ep.allocate(out_bytes.max(1), 256).unwrap();

    // Warm-compile every entry outside any capture, then launch (no sync).
    warm_planar_linear(ep.runtime()).unwrap();
    launch_planar_linear(&admission, dtype, &a_buf, &mut out_buf).unwrap();
    ep.runtime().synchronize().unwrap();

    let out = decode_output(&download(ep, &out_buf, out_bytes), dtype);

    ep.deallocate(a_buf).unwrap();
    ep.deallocate(out_buf).unwrap();
    out
}

fn cpu_oracle(
    format: PlanarBlockFormat,
    m_rows: usize,
    out: usize,
    in_features: usize,
    bs0: usize,
    bs1: usize,
    a_deq: &[f32],
    packed: &[u8],
    scale: &[u8],
) -> Vec<f32> {
    let layout = PlanarLayout::new(format, out, in_features, bs0, bs1).unwrap();
    planar_block_matmul(a_deq, m_rows, &layout, packed, scale).unwrap()
}

fn assert_parity(label: &str, dtype: PlanarActivationDtype, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label} {dtype:?}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = tolerance(dtype, w);
        assert!(
            (g - w).abs() <= tol,
            "{label} {dtype:?} out[{i}]: got {g}, want {w}, tol {tol}"
        );
    }
}

/// One end-to-end parity assertion for a `block_fp8` shape across all dtypes.
fn check_block_fp8(
    ep: &Arc<CudaExecutionProvider>,
    m: usize,
    out: usize,
    in_features: usize,
    bs: usize,
) {
    let (packed, scale) = block_fp8_fixture(out, in_features, bs, 0xa11ce ^ (out as u64));
    let a = activations(m, in_features, 0xbeef ^ (in_features as u64));
    let dims = PlanarLinearDims {
        format: PLANAR_FORMAT_BLOCK_FP8,
        m_rows: m,
        in_features,
        out_features: out,
        bs0: bs,
        bs1: bs,
    };
    for dtype in PlanarActivationDtype::all() {
        let (a_bytes, a_deq) = quantize_activation(&a, dtype);
        let want = cpu_oracle(
            PlanarBlockFormat::BlockFp8,
            m,
            out,
            in_features,
            bs,
            bs,
            &a_deq,
            &packed,
            &scale,
        );
        let got = run_planar_gpu(ep, dtype, &dims, &a_bytes, &packed, &scale);
        assert_parity(
            &format!("block_fp8 {m}x{in_features}x{out}"),
            dtype,
            &got,
            &want,
        );
    }
}

/// One end-to-end parity assertion for an `fp4_planar` shape across all dtypes.
fn check_fp4_planar(ep: &Arc<CudaExecutionProvider>, m: usize, out: usize, in_features: usize) {
    let (packed, scale) = fp4_fixture(out, in_features, 0xc0ffee ^ (out as u64));
    let a = activations(m, in_features, 0xd00d ^ (in_features as u64));
    let dims = PlanarLinearDims {
        format: PLANAR_FORMAT_FP4_PLANAR,
        m_rows: m,
        in_features,
        out_features: out,
        bs0: 0,
        bs1: 0,
    };
    for dtype in PlanarActivationDtype::all() {
        let (a_bytes, a_deq) = quantize_activation(&a, dtype);
        let want = cpu_oracle(
            PlanarBlockFormat::Fp4Planar,
            m,
            out,
            in_features,
            1,
            FP4_MICROSCALE_BLOCK,
            &a_deq,
            &packed,
            &scale,
        );
        let got = run_planar_gpu(ep, dtype, &dims, &a_bytes, &packed, &scale);
        assert_parity(
            &format!("fp4_planar {m}x{in_features}x{out}"),
            dtype,
            &got,
            &want,
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn block_fp8_matches_cpu_oracle_all_dtypes() {
    let ep = require_cuda();
    // Shape-faithful DeepSeek block_fp8 projection (bs=128), plus small + ragged.
    check_block_fp8(&ep, 4, 64, 128, 128);
    check_block_fp8(&ep, 1, 128, 256, 128);
    check_block_fp8(&ep, 5, 48, 96, 32);
    check_block_fp8(&ep, 3, 130, 130, 128); // ragged ceil-div scale grid
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn fp4_planar_matches_cpu_oracle_all_dtypes() {
    let ep = require_cuda();
    // Shape-faithful DeepSeek expert (moe_intermediate scaled down), block-32.
    check_fp4_planar(&ep, 4, 32, 64);
    check_fp4_planar(&ep, 1, 64, 128);
    check_fp4_planar(&ep, 5, 48, 96);
    check_fp4_planar(&ep, 2, 16, 256);
}

/// Every E2M1 nibble code (0..16) and a full sweep of E4M3 bytes must decode and
/// contract exactly like the oracle. Built so weight columns cycle through all
/// codes; a single all-ones activation row makes each output the column's
/// decoded sum, isolating decode correctness.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn exhaustive_small_codepoints() {
    let ep = require_cuda();

    // fp4: out columns each pack all 16 nibble codes across in=32 (one block).
    let out = 16usize;
    let in_features = FP4_MICROSCALE_BLOCK; // 32
    let mut packed = vec![0u8; out * (in_features / FP4_PACK_FACTOR)];
    for col in 0..out {
        for pair in 0..(in_features / FP4_PACK_FACTOR) {
            let low = ((2 * pair) as u8).wrapping_add(col as u8) & 0x0f;
            let high = ((2 * pair + 1) as u8).wrapping_add(col as u8) & 0x0f;
            packed[col * (in_features / FP4_PACK_FACTOR) + pair] = (high << 4) | low;
        }
    }
    // Sweep the scale exponent per column too (still benign, != 0xff).
    let scale: Vec<u8> = (0..out).map(|c| benign_scale(c as u8)).collect();
    let a = vec![1.0f32; in_features];
    let dims = PlanarLinearDims {
        format: PLANAR_FORMAT_FP4_PLANAR,
        m_rows: 1,
        in_features,
        out_features: out,
        bs0: 0,
        bs1: 0,
    };
    for dtype in PlanarActivationDtype::all() {
        let (a_bytes, a_deq) = quantize_activation(&a, dtype);
        let want = cpu_oracle(
            PlanarBlockFormat::Fp4Planar,
            1,
            out,
            in_features,
            1,
            FP4_MICROSCALE_BLOCK,
            &a_deq,
            &packed,
            &scale,
        );
        let got = run_planar_gpu(&ep, dtype, &dims, &a_bytes, &packed, &scale);
        assert_parity("fp4 exhaustive nibbles", dtype, &got, &want);
    }

    // block_fp8: a 256-wide column sweeps every (finite) E4M3 byte.
    let out = 8usize;
    let in_features = 256usize;
    let bs = 128usize;
    let mut packed = vec![0u8; out * in_features];
    for col in 0..out {
        for k in 0..in_features {
            packed[col * in_features + k] = finite_e4m3(((k + col) & 0xff) as u8);
        }
    }
    let scale: Vec<u8> = (0..out.div_ceil(bs) * in_features.div_ceil(bs))
        .map(|i| benign_scale(i as u8))
        .collect();
    let a = vec![1.0f32; in_features];
    let dims = PlanarLinearDims {
        format: PLANAR_FORMAT_BLOCK_FP8,
        m_rows: 1,
        in_features,
        out_features: out,
        bs0: bs,
        bs1: bs,
    };
    for dtype in PlanarActivationDtype::all() {
        let (a_bytes, a_deq) = quantize_activation(&a, dtype);
        let want = cpu_oracle(
            PlanarBlockFormat::BlockFp8,
            1,
            out,
            in_features,
            bs,
            bs,
            &a_deq,
            &packed,
            &scale,
        );
        let got = run_planar_gpu(&ep, dtype, &dims, &a_bytes, &packed, &scale);
        assert_parity("block_fp8 exhaustive e4m3", dtype, &got, &want);
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn block_fp8_large_block_size_uses_overflow_free_scale_grid() {
    let ep = require_cuda();
    let dims = PlanarLinearDims {
        format: PLANAR_FORMAT_BLOCK_FP8,
        m_rows: 1,
        in_features: 32,
        out_features: 2,
        bs0: 1,
        bs1: i32::MAX as usize,
    };
    let activation = vec![1.0f32; dims.in_features];
    let (a_bytes, a_deq) = quantize_activation(&activation, PlanarActivationDtype::F32);
    let packed = vec![0x38u8; dims.out_features * dims.in_features];
    let scale = vec![127u8, 128u8];

    let got = run_planar_gpu(
        &ep,
        PlanarActivationDtype::F32,
        &dims,
        &a_bytes,
        &packed,
        &scale,
    );
    let want = cpu_oracle(
        PlanarBlockFormat::BlockFp8,
        dims.m_rows,
        dims.out_features,
        dims.in_features,
        dims.bs0,
        dims.bs1,
        &a_deq,
        &packed,
        &scale,
    );
    assert_parity(
        "block_fp8 overflow-free scale grid",
        PlanarActivationDtype::F32,
        &got,
        &want,
    );
}

/// Repeated launches with changing shapes on one device: the NVRTC cache is
/// warmed once, every shape still lands exact, and buffers are cleanly recycled.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn multi_request_shape_change_is_stable() {
    let ep = require_cuda();
    for &(m, out, in_features) in &[
        (1usize, 32usize, 64usize),
        (7, 16, 128),
        (3, 64, 96),
        (1, 32, 64),
    ] {
        check_fp4_planar(&ep, m, out, in_features);
    }
    for &(m, out, in_features, bs) in &[(2usize, 64usize, 128usize, 128usize), (1, 48, 96, 32)] {
        check_block_fp8(&ep, m, out, in_features, bs);
    }
}

/// Invalid aux / OOB geometry must typed-reject before any launch — no crash, no
/// silent dense fallback.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn invalid_geometry_is_typed_rejected() {
    let ep = require_cuda();
    // Odd fp4 contraction cannot produce an admission proof.
    let odd = PlanarLinearDims {
        format: PLANAR_FORMAT_FP4_PLANAR,
        m_rows: 1,
        in_features: 63,
        out_features: 16,
        bs0: 0,
        bs1: 0,
    };
    assert!(admit_planar_linear(&ep, &odd, 63, &[], &[], 16).is_err());

    // Truncated aux scale: length validation must refuse.
    let dims = PlanarLinearDims {
        format: PLANAR_FORMAT_BLOCK_FP8,
        m_rows: 2,
        in_features: 128,
        out_features: 64,
        bs0: 128,
        bs1: 128,
    };
    let packed = vec![0u8; 64 * 128];
    assert!(admit_planar_linear(&ep, &dims, 2 * 128, &packed, &[], 2 * 64).is_err());
    // Zero block size.
    let bad_block = PlanarLinearDims { bs1: 0, ..dims };
    assert!(admit_planar_linear(&ep, &bad_block, 2 * 128, &packed, &[], 2 * 64).is_err());
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn malformed_values_reject_before_device_activity() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let allocations = runtime.allocation_counts();
    let transfers = runtime.transfer_counts();

    let fp8 = PlanarLinearDims {
        format: PLANAR_FORMAT_BLOCK_FP8,
        m_rows: 1,
        in_features: 1,
        out_features: 1,
        bs0: 1,
        bs1: 1,
    };
    for (packed, scale, label) in [
        ([0x7fu8], [127u8], "reserved E4M3 +NaN"),
        ([0xffu8], [127u8], "reserved E4M3 -NaN"),
        ([0x38u8], [0xffu8], "reserved UE8M0"),
        ([0x7eu8], [247u8], "finite E4M3 product overflow"),
    ] {
        assert!(
            admit_planar_linear(&ep, &fp8, 1, &packed, &scale, 1).is_err(),
            "{label} must reject before upload"
        );
    }

    let fp4 = PlanarLinearDims {
        format: PLANAR_FORMAT_FP4_PLANAR,
        m_rows: 1,
        in_features: 32,
        out_features: 1,
        bs0: 0,
        bs1: 0,
    };
    let max_codes = [0x77u8; 16];
    assert!(admit_planar_linear(&ep, &fp4, 32, &max_codes, &[0xff], 1).is_err());
    assert!(admit_planar_linear(&ep, &fp4, 32, &max_codes, &[253], 1).is_err());

    assert_eq!(runtime.allocation_counts(), allocations);
    assert_eq!(runtime.transfer_counts(), transfers);
}

/// Admission owns the exact immutable device bank populated from validated
/// bytes. The supported external surface accepts no bank buffer or pointer, so
/// mutating the only caller-owned source after admission cannot affect launch.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn sealed_admission_outlives_mutated_sources() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let dims = PlanarLinearDims {
        format: PLANAR_FORMAT_BLOCK_FP8,
        m_rows: 1,
        in_features: 32,
        out_features: 2,
        bs0: 32,
        bs1: 32,
    };
    let mut packed = vec![0x38u8; dims.out_features * dims.in_features];
    let mut scale = vec![127u8; dims.out_features.div_ceil(dims.bs0)];
    let activation = vec![1.0f32; dims.in_features];
    let (activation_bytes, _) = quantize_activation(&activation, PlanarActivationDtype::F32);
    let activation_buffer = upload(&ep, &activation_bytes);
    let mut output = ep.allocate(dims.out_features * 4, 256).unwrap();
    let admission = admit_planar_linear(
        &ep,
        &dims,
        dims.in_features,
        &packed,
        &scale,
        dims.out_features,
    )
    .unwrap();
    warm_planar_linear(runtime).unwrap();
    launch_planar_linear(
        &admission,
        PlanarActivationDtype::F32,
        &activation_buffer,
        &mut output,
    )
    .unwrap();
    runtime.synchronize().unwrap();
    let admitted_output = download(&ep, &output, dims.out_features * 4);

    // Mutating the source vectors after admission cannot mutate the sealed
    // allocations used by launch or replay.
    packed.fill(0x7f);
    scale.fill(0xff);
    launch_planar_linear(
        &admission,
        PlanarActivationDtype::F32,
        &activation_buffer,
        &mut output,
    )
    .unwrap();
    runtime.synchronize().unwrap();
    assert_eq!(
        download(&ep, &output, dims.out_features * 4),
        admitted_output
    );

    drop(admission);
    for buffer in [activation_buffer, output] {
        ep.deallocate(buffer).unwrap();
    }
}

/// An externally retained allocator exposes mutation/remap capabilities outside
/// the provider's ownership domain. Sealed admission must reject it before any
/// allocation or upload rather than minting an "immutable" handle it cannot
/// enforce.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn sealed_admission_rejects_externally_controlled_allocator() {
    let provider = CudaExecutionProvider::new(selected_cuda_ordinal()).unwrap();
    let runtime = Arc::clone(provider.runtime());
    let device = DeviceKey::device(runtime.ordinal());
    let governor = LedgerGovernor::new(LeaseLedger::new(u64::MAX, 0, 0));
    let external = Arc::new(
        CudaVmmAllocator::new(
            runtime.cuda_context(),
            device,
            runtime.ordinal() as i32,
            32 * 1024 * 1024,
            &governor,
            HolderId::new(2107),
            MemoryRole::Weights,
        )
        .unwrap(),
    );
    let provider = Arc::new(
        provider
            .with_memory(Arc::clone(&external) as Arc<_>)
            .unwrap(),
    );
    let before = (
        external.committed_and_reserved().0,
        runtime.allocation_counts(),
        runtime.transfer_counts(),
    );
    let dims = PlanarLinearDims {
        format: PLANAR_FORMAT_BLOCK_FP8,
        m_rows: 1,
        in_features: 1,
        out_features: 1,
        bs0: 1,
        bs1: 1,
    };
    let error = match admit_planar_linear(&provider, &dims, 1, &[0x38], &[127], 1) {
        Ok(_) => panic!("externally controlled memory cannot back a sealed admission"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("externally controlled"),
        "{error}"
    );
    assert_eq!(
        (
            external.committed_and_reserved().0,
            runtime.allocation_counts(),
            runtime.transfer_counts(),
        ),
        before,
        "rejection must happen before allocation, mapping, or upload"
    );
}

/// A warmed fixed-shape planar linear records into a CUDA-graph capture and
/// replays ≥3× byte-identically to the eager result (no in-capture alloc/sync).
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn capture_replay_parity() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let accounting_baseline = ep.device_allocation_counts().unwrap();

    let (m, out, in_features, bs) = (4usize, 64usize, 128usize, 128usize);
    let (mut packed, mut scale) = block_fp8_fixture(out, in_features, bs, 0x5eed);
    let a = activations(m, in_features, 0x1234);
    let dims = PlanarLinearDims {
        format: PLANAR_FORMAT_BLOCK_FP8,
        m_rows: m,
        in_features,
        out_features: out,
        bs0: bs,
        bs1: bs,
    };
    let (a_bytes, _) = quantize_activation(&a, PlanarActivationDtype::F32);
    let out_bytes = m * out * 4;

    let a_buf = upload(&ep, &a_bytes);
    let mut out_buf = ep.allocate(out_bytes, 256).unwrap();
    let admission =
        admit_planar_linear(&ep, &dims, m * in_features, &packed, &scale, m * out).unwrap();

    // Warm compile BEFORE capture (compile synchronizes the device).
    warm_planar_linear(runtime).unwrap();

    // Repeated warmed launches perform no runtime allocation or transfer.
    let warmed_allocations = runtime.allocation_counts();
    let warmed_transfers = runtime.transfer_counts();
    for _ in 0..2 {
        launch_planar_linear(&admission, PlanarActivationDtype::F32, &a_buf, &mut out_buf).unwrap();
    }
    assert_eq!(runtime.allocation_counts(), warmed_allocations);
    assert_eq!(runtime.transfer_counts(), warmed_transfers);
    runtime.synchronize().unwrap();
    let eager = download(&ep, &out_buf, out_bytes);

    // Capture the warmed launch, then replay ≥3× and compare byte-for-byte.
    let capture_allocations = runtime.allocation_counts();
    let capture_transfers = runtime.transfer_counts();
    runtime.begin_graph_capture(&[]).unwrap();
    assert!(
        admit_planar_linear(&ep, &dims, m * in_features, &packed, &scale, m * out).is_err(),
        "admission must reject before allocating or uploading during capture"
    );
    // These are the only weight bytes still owned by the external caller. Make
    // them invalid after admission and before capture: the supported API has no
    // route from either source back to the sealed device allocation.
    packed.fill(0x7f);
    scale.fill(0xff);
    launch_planar_linear(&admission, PlanarActivationDtype::F32, &a_buf, &mut out_buf).unwrap();
    runtime.end_graph_capture().unwrap();
    assert_eq!(runtime.allocation_counts(), capture_allocations);
    assert_eq!(runtime.transfer_counts(), capture_transfers);

    let pinned_frees = ep.device_allocation_counts().unwrap().1;
    drop(admission);
    assert_eq!(
        ep.device_allocation_counts().unwrap().1,
        pinned_frees,
        "dropping the caller handle must not free graph-embedded banks"
    );

    let zeros = vec![0u8; out_bytes];
    for replay in 0..3 {
        packed.fill(replay as u8);
        scale.fill(255 - replay as u8);
        let before_allocations = runtime.allocation_counts();
        let before_transfers = runtime.transfer_counts();
        // SAFETY: `out_buf` is `out_bytes` wide; clear it so a stale result can't
        // masquerade as a replay.
        unsafe { runtime.htod(&zeros, cuptr(out_buf.as_ptr())).unwrap() };
        runtime.replay_graph().unwrap();
        runtime.synchronize().unwrap();
        let replayed = download(&ep, &out_buf, out_bytes);
        assert_eq!(
            replayed, eager,
            "capture replay {replay} diverged from eager"
        );
        assert_eq!(runtime.allocation_counts(), before_allocations);
        let after_transfers = runtime.transfer_counts();
        assert_eq!(
            after_transfers.host_to_device,
            before_transfers.host_to_device + 1
        );
        assert_eq!(
            after_transfers.device_to_host,
            before_transfers.device_to_host + 1
        );
        assert_eq!(
            after_transfers.async_host_to_device,
            before_transfers.async_host_to_device
        );
    }

    let before_reset_frees = ep.device_allocation_counts().unwrap().1;
    assert!(runtime.reset_graph().unwrap());
    ep.wait_for_deferred_releases().unwrap();
    assert_eq!(
        ep.device_allocation_counts().unwrap().1,
        before_reset_frees + 2,
        "graph reset must release the packed and scale banks exactly once: {:?}",
        ep.deferred_release_stats()
    );
    ep.deallocate(a_buf).unwrap();
    ep.deallocate(out_buf).unwrap();
    ep.wait_for_deferred_releases().unwrap();
    let settled = ep.device_allocation_counts().unwrap();
    assert_eq!(
        settled.0 - accounting_baseline.0,
        settled.1 - accounting_baseline.1,
        "capture/reset teardown must return planar allocations to the exact accounting baseline"
    );
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn graph_bank_pins_are_per_graph_and_abort_safe() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let (packed, scale) = block_fp8_fixture(32, 32, 32, 0x9911);
    let dims = PlanarLinearDims {
        format: PLANAR_FORMAT_BLOCK_FP8,
        m_rows: 1,
        in_features: 32,
        out_features: 32,
        bs0: 32,
        bs1: 32,
    };
    let (a_bytes, _) = quantize_activation(&[1.0; 32], PlanarActivationDtype::F32);
    let a_buf = upload(&ep, &a_bytes);
    let mut out_buf = ep.allocate(32 * 4, 256).unwrap();
    let admission = admit_planar_linear(&ep, &dims, 32, &packed, &scale, 32).unwrap();
    warm_planar_linear(runtime).unwrap();

    runtime.begin_graph_capture(&[]).unwrap();
    launch_planar_linear(&admission, PlanarActivationDtype::F32, &a_buf, &mut out_buf).unwrap();
    runtime.abort_graph_capture().unwrap();
    for slot in [DeviceGraphSlot::Primary, DeviceGraphSlot::Verify] {
        runtime.begin_graph_capture_in(slot, &[]).unwrap();
        launch_planar_linear(&admission, PlanarActivationDtype::F32, &a_buf, &mut out_buf).unwrap();
        runtime.end_graph_capture_in(slot).unwrap();
    }
    let before_drop_frees = ep.device_allocation_counts().unwrap().1;
    drop(admission);
    assert_eq!(ep.device_allocation_counts().unwrap().1, before_drop_frees);

    assert!(runtime.reset_graph_in(DeviceGraphSlot::Primary).unwrap());
    ep.wait_for_deferred_releases().unwrap();
    assert_eq!(
        ep.device_allocation_counts().unwrap().1,
        before_drop_frees,
        "resetting one graph must not release a bank pinned by its sibling"
    );
    for _ in 0..3 {
        runtime.replay_graph_in(DeviceGraphSlot::Verify).unwrap();
    }
    runtime.synchronize().unwrap();
    assert!(runtime.reset_graph_in(DeviceGraphSlot::Verify).unwrap());
    ep.wait_for_deferred_releases().unwrap();
    assert_eq!(
        ep.device_allocation_counts().unwrap().1,
        before_drop_frees + 2,
        "{:?}",
        ep.deferred_release_stats()
    );

    ep.deallocate(a_buf).unwrap();
    ep.deallocate(out_buf).unwrap();
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn unregistered_capture_and_foreign_context_are_rejected_before_launch() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let (packed, scale) = block_fp8_fixture(32, 32, 32, 0x5511);
    let dims = PlanarLinearDims {
        format: PLANAR_FORMAT_BLOCK_FP8,
        m_rows: 1,
        in_features: 32,
        out_features: 32,
        bs0: 32,
        bs1: 32,
    };
    let admission = admit_planar_linear(&ep, &dims, 32, &packed, &scale, 32).unwrap();
    let (a_bytes, _) = quantize_activation(&[1.0; 32], PlanarActivationDtype::F32);
    let a_buf = upload(&ep, &a_bytes);
    let mut out_buf = ep.allocate(32 * 4, 256).unwrap();
    warm_planar_linear(runtime).unwrap();

    runtime.test_begin_unregistered_graph_capture().unwrap();
    let error = launch_planar_linear(&admission, PlanarActivationDtype::F32, &a_buf, &mut out_buf)
        .expect_err("capture without a lifecycle ownership sink must reject");
    assert!(error.to_string().contains("no registered ownership sink"));
    runtime.test_end_unregistered_graph_capture().unwrap();

    let foreign = require_cuda();
    let foreign_a = upload(&foreign, &a_bytes);
    let mut foreign_out = foreign.allocate(32 * 4, 256).unwrap();
    let error = launch_planar_linear(
        &admission,
        PlanarActivationDtype::F32,
        &foreign_a,
        &mut foreign_out,
    )
    .expect_err("same-device buffers from a different provider context must reject");
    assert!(error.to_string().contains("provider context"));

    if let Ok(other_device) = CudaExecutionProvider::new(1) {
        let other_device = Arc::new(other_device);
        let other_a = upload(&other_device, &a_bytes);
        let mut other_out = other_device.allocate(32 * 4, 256).unwrap();
        let error = launch_planar_linear(
            &admission,
            PlanarActivationDtype::F32,
            &other_a,
            &mut other_out,
        )
        .expect_err("buffers from a different CUDA device must reject");
        assert!(error.to_string().contains("device"));
        other_device.deallocate(other_a).unwrap();
        other_device.deallocate(other_out).unwrap();
    }

    ep.deallocate(a_buf).unwrap();
    ep.deallocate(out_buf).unwrap();
    foreign.deallocate(foreign_a).unwrap();
    foreign.deallocate(foreign_out).unwrap();
}

/// The advertised capability strings the Mobius #602 / Deckard #593 planar
/// emitters probe must be exactly the two planar matmul formats — and only after
/// the parity proof above actually launches on this device.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn capability_strings_are_advertised_on_device() {
    let ep = require_cuda();
    // The kernels must actually compile on this device before we trust the claim.
    warm_planar_linear(ep.runtime()).unwrap();
    assert_eq!(planar_matmul_capable_formats(), ["block_fp8", "fp4_planar"]);
}

// ---------------------------------------------------------------------------
// Measurement probe (ignored by default; run with --ignored on an idle A100)
// ---------------------------------------------------------------------------

/// Resolve the `nvidia-smi -i <target>` argument for the CUDA device this
/// process actually pinned.
fn resolve_smi_device(visible: Option<&str>, ordinal: usize) -> Option<String> {
    match visible {
        Some(list) if !list.trim().is_empty() => {
            let mut entries = Vec::new();
            for raw in list.split(',') {
                let entry = raw.trim();
                if entry.is_empty() {
                    break;
                }
                entries.push(entry.to_string());
            }
            entries.into_iter().nth(ordinal)
        }
        _ => Some(ordinal.to_string()),
    }
}

fn pinned_smi_target() -> Option<String> {
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").ok();
    resolve_smi_device(visible.as_deref(), selected_cuda_ordinal() as usize)
}

fn selected_cuda_ordinal() -> u32 {
    std::env::var("ONNX_GENAI_CUDA_DEVICE")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(0)
}

fn gpu_is_idle() -> bool {
    let Some(target) = pinned_smi_target() else {
        return false;
    };
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
            "-i",
            &target,
        ])
        .output();
    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse::<u32>()
            .map(|util| util <= 5)
            .unwrap_or(false),
        _ => false,
    }
}

/// True if another compute process is resident on the pinned device.
fn foreign_compute_present() -> bool {
    let mine = std::process::id();
    let Some(target) = pinned_smi_target() else {
        return true;
    };
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
            "-i",
            &target,
        ])
        .output();
    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|line| line.trim().parse::<u32>().ok())
            .any(|pid| pid != mine),
        _ => true,
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn resolve_smi_device_maps_visible_ordinal_to_physical_target() {
    let _ep = require_cuda();
    assert_eq!(resolve_smi_device(None, 0).as_deref(), Some("0"));
    assert_eq!(resolve_smi_device(None, 3).as_deref(), Some("3"));
    assert_eq!(resolve_smi_device(Some(""), 2).as_deref(), Some("2"));
    assert_eq!(resolve_smi_device(Some("2,5"), 0).as_deref(), Some("2"));
    assert_eq!(resolve_smi_device(Some("2,5"), 1).as_deref(), Some("5"));
    assert_eq!(resolve_smi_device(Some(" 7 , 1 "), 0).as_deref(), Some("7"));
    assert_eq!(
        resolve_smi_device(Some("GPU-abc123,GPU-def456"), 1).as_deref(),
        Some("GPU-def456")
    );
    assert_eq!(resolve_smi_device(Some("2,5"), 2), None);
    assert_eq!(resolve_smi_device(Some("2,,5"), 1), None);
    assert_eq!(resolve_smi_device(Some("2,,5"), 0).as_deref(), Some("2"));
}

/// Warm-then-batch timing on the pinned device. Reports median + range of a
/// batched kernel enqueue-to-completion window (n≥3) and the host enqueue cost
/// separately. Not a full-model tok/s claim: a single-shape microbench of the
/// planar linear primitive. `#[ignore]`d so it never runs in the correctness
/// gate; run explicitly with `--ignored --nocapture` on a verified-idle A100.
#[test]
#[ignore = "measurement probe: run explicitly on a verified-idle A100 with --ignored --nocapture"]
fn planar_matmul_measurement() {
    use std::time::Instant;
    let ep = require_cuda();
    let runtime = ep.runtime();
    assert!(
        gpu_is_idle(),
        "measurement requires a verified-idle pinned GPU (CUDA_VISIBLE_DEVICES); it was busy"
    );

    // Shape-faithful-ish DeepSeek block_fp8 projection.
    let (m, out, in_features, bs) = (16usize, 4096usize, 4096usize, 128usize);
    let (packed, scale) = block_fp8_fixture(out, in_features, bs, 0xf00d);
    let a = activations(m, in_features, 0xcafe);
    let (a_bytes, _) = quantize_activation(&a, PlanarActivationDtype::F16);
    let dims = PlanarLinearDims {
        format: PLANAR_FORMAT_BLOCK_FP8,
        m_rows: m,
        in_features,
        out_features: out,
        bs0: bs,
        bs1: bs,
    };
    let out_bytes = m * out * dtype_bytes(PlanarActivationDtype::F16);
    let a_buf = upload(&ep, &a_bytes);
    let mut out_buf = ep.allocate(out_bytes, 256).unwrap();
    let admission =
        admit_planar_linear(&ep, &dims, m * in_features, &packed, &scale, m * out).unwrap();
    warm_planar_linear(runtime).unwrap();

    // 8 s ramp: keep the SMs busy so clocks reach steady state before timing.
    let ramp = Instant::now();
    while ramp.elapsed().as_secs_f32() < 8.0 {
        for _ in 0..32 {
            launch_planar_linear(&admission, PlanarActivationDtype::F16, &a_buf, &mut out_buf)
                .unwrap();
        }
        runtime.synchronize().unwrap();
    }

    // Host enqueue cost: time N launches without a sync in the loop.
    let enqueue_n = 200;
    let host = Instant::now();
    for _ in 0..enqueue_n {
        launch_planar_linear(&admission, PlanarActivationDtype::F16, &a_buf, &mut out_buf).unwrap();
    }
    let host_enqueue = host.elapsed();
    runtime.synchronize().unwrap();

    // Batched kernel window: median of n≥3 batches of `batch` launches each.
    let batch = 64usize;
    let mut sample_shape = |samples: &mut Vec<f64>| {
        for _ in 0..5 {
            assert!(
                !foreign_compute_present(),
                "a foreign compute process appeared on the pinned GPU mid-measurement"
            );
            let t = Instant::now();
            for _ in 0..batch {
                launch_planar_linear(&admission, PlanarActivationDtype::F16, &a_buf, &mut out_buf)
                    .unwrap();
            }
            runtime.synchronize().unwrap();
            samples.push(t.elapsed().as_secs_f64() / batch as f64);
        }
    };

    let mut samples = Vec::new();
    sample_shape(&mut samples);
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    let min = samples[0];
    let max = *samples.last().unwrap();

    let mut recheck = Vec::new();
    sample_shape(&mut recheck);
    recheck.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let recheck_median = recheck[recheck.len() / 2];
    let drift = (recheck_median - median).abs() / median;

    eprintln!(
        "planar block_fp8 f16 [{m}x{in_features}->{out}] batched kernel/launch: median {:.1} us (min {:.1}, max {:.1}); host enqueue {:.2} us/launch over {enqueue_n}; first-shape recheck median {:.1} us (drift {:.1}%)",
        median * 1e6,
        min * 1e6,
        max * 1e6,
        host_enqueue.as_secs_f64() / enqueue_n as f64 * 1e6,
        recheck_median * 1e6,
        drift * 100.0,
    );
    assert!(
        drift < 0.05,
        "first-shape drift {:.1}% exceeds 5%: device not in steady state, measurement is unreliable",
        drift * 100.0
    );

    ep.deallocate(a_buf).unwrap();
    ep.deallocate(out_buf).unwrap();
}
