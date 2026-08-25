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
use onnx_runtime_ep_api::{DeviceBuffer, ExecutionProvider};
use onnx_runtime_ep_cpu::kernels::planar_block_quant::{
    FP4_MICROSCALE_BLOCK, FP4_PACK_FACTOR, PlanarBlockFormat, PlanarLayout, planar_block_matmul,
};
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ep_cuda::{
    CudaExecutionProvider, PLANAR_FORMAT_BLOCK_FP8, PLANAR_FORMAT_FP4_PLANAR,
    PlanarActivationDtype, PlanarLinearDims, PlanarLinearPtrs, launch_planar_linear,
    planar_matmul_capable_formats, validate_planar_linear, warm_planar_linear,
};

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn require_cuda() -> CudaExecutionProvider {
    match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
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
    ep: &CudaExecutionProvider,
    dtype: PlanarActivationDtype,
    dims: &PlanarLinearDims,
    a_bytes: &[u8],
    packed: &[u8],
    scale: &[u8],
) -> Vec<f32> {
    let out_elems = dims.m_rows * dims.out_features;
    validate_planar_linear(
        dims,
        dims.m_rows * dims.in_features,
        packed.len(),
        scale.len(),
        out_elems,
    )
    .unwrap();

    let out_bytes = out_elems * dtype_bytes(dtype);

    let a_buf = upload(ep, a_bytes);
    let packed_buf = upload(ep, packed);
    let scale_buf = upload(ep, scale);
    let out_buf = ep.allocate(out_bytes.max(1), 256).unwrap();

    // Warm-compile every entry outside any capture, then launch (no sync).
    warm_planar_linear(ep.runtime()).unwrap();
    let ptrs = PlanarLinearPtrs {
        activation: cuptr(a_buf.as_ptr()),
        packed: cuptr(packed_buf.as_ptr()),
        scale: cuptr(scale_buf.as_ptr()),
        output: cuptr(out_buf.as_ptr()),
    };
    launch_planar_linear(ep.runtime(), dtype, dims, &ptrs).unwrap();
    ep.runtime().synchronize().unwrap();

    let out = decode_output(&download(ep, &out_buf, out_bytes), dtype);

    ep.deallocate(a_buf).unwrap();
    ep.deallocate(packed_buf).unwrap();
    ep.deallocate(scale_buf).unwrap();
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
    ep: &CudaExecutionProvider,
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
fn check_fp4_planar(ep: &CudaExecutionProvider, m: usize, out: usize, in_features: usize) {
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
    let runtime = ep.runtime();

    // Odd fp4 contraction: the launcher must refuse (it re-validates geometry).
    let odd = PlanarLinearDims {
        format: PLANAR_FORMAT_FP4_PLANAR,
        m_rows: 1,
        in_features: 63,
        out_features: 16,
        bs0: 0,
        bs1: 0,
    };
    let ptrs = PlanarLinearPtrs {
        activation: 0,
        packed: 0,
        scale: 0,
        output: 0,
    };
    assert!(launch_planar_linear(runtime, PlanarActivationDtype::F32, &odd, &ptrs).is_err());

    // Truncated aux scale: length validation must refuse.
    let dims = PlanarLinearDims {
        format: PLANAR_FORMAT_BLOCK_FP8,
        m_rows: 2,
        in_features: 128,
        out_features: 64,
        bs0: 128,
        bs1: 128,
    };
    assert!(validate_planar_linear(&dims, 2 * 128, 64 * 128, 0, 2 * 64).is_err());
    // Zero block size.
    let bad_block = PlanarLinearDims { bs1: 0, ..dims };
    assert!(launch_planar_linear(runtime, PlanarActivationDtype::F32, &bad_block, &ptrs).is_err());
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

    let (m, out, in_features, bs) = (4usize, 64usize, 128usize, 128usize);
    let (packed, scale) = block_fp8_fixture(out, in_features, bs, 0x5eed);
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
    let packed_buf = upload(&ep, &packed);
    let scale_buf = upload(&ep, &scale);
    let out_buf = ep.allocate(out_bytes, 256).unwrap();
    let ptrs = PlanarLinearPtrs {
        activation: cuptr(a_buf.as_ptr()),
        packed: cuptr(packed_buf.as_ptr()),
        scale: cuptr(scale_buf.as_ptr()),
        output: cuptr(out_buf.as_ptr()),
    };

    // Warm compile BEFORE capture (compile synchronizes the device).
    warm_planar_linear(runtime).unwrap();

    // Eager reference.
    launch_planar_linear(runtime, PlanarActivationDtype::F32, &dims, &ptrs).unwrap();
    runtime.synchronize().unwrap();
    let eager = download(&ep, &out_buf, out_bytes);

    // Capture the warmed launch, then replay ≥3× and compare byte-for-byte.
    runtime.begin_graph_capture(&[]).unwrap();
    launch_planar_linear(runtime, PlanarActivationDtype::F32, &dims, &ptrs).unwrap();
    runtime.end_graph_capture().unwrap();

    let zeros = vec![0u8; out_bytes];
    for replay in 0..3 {
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
    }

    runtime.reset_graph().unwrap();
    ep.deallocate(a_buf).unwrap();
    ep.deallocate(packed_buf).unwrap();
    ep.deallocate(scale_buf).unwrap();
    ep.deallocate(out_buf).unwrap();
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
/// process actually pinned. Under `CUDA_VISIBLE_DEVICES`, the process's device
/// ordinal (`ONNX_GENAI_CUDA_DEVICE`, default 0) indexes into the visible list,
/// whose entry is the *physical* index or GPU UUID that `nvidia-smi` must query
/// (`nvidia-smi -i` accepts either). Without `CUDA_VISIBLE_DEVICES` the ordinal
/// is itself the physical index. Returns `None` when the ordinal cannot be
/// resolved to a visible device (e.g. it is past the end of the list, or the
/// list is truncated by an empty token exactly as the CUDA runtime truncates),
/// so callers can refuse rather than probe the wrong GPU.
fn resolve_smi_device(visible: Option<&str>, ordinal: usize) -> Option<String> {
    match visible {
        Some(list) if !list.trim().is_empty() => {
            let mut entries = Vec::new();
            for raw in list.split(',') {
                let entry = raw.trim();
                // The CUDA runtime stops enumerating at the first empty/invalid
                // token, so anything after it is not visible to the process.
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

/// The `nvidia-smi -i` target for the device pinned by this process's
/// environment, or `None` if it cannot be resolved.
fn pinned_smi_target() -> Option<String> {
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").ok();
    let ordinal = std::env::var("ONNX_GENAI_CUDA_DEVICE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0);
    resolve_smi_device(visible.as_deref(), ordinal)
}

fn gpu_is_idle() -> bool {
    // Best-effort idle check via nvidia-smi utilization on the *pinned* device
    // (the physical GPU / UUID selected by CUDA_VISIBLE_DEVICES + ordinal), not
    // a hardcoded physical index — otherwise the probe validates the wrong GPU.
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

/// True if a compute process other than this test binary is resident on the
/// pinned device. Used as the mid-measurement tenant guard instead of
/// `utilization.gpu`: our own batched kernels legitimately drive utilization to
/// 100%, so a rolling-window utilization sample would report that self-load as
/// "busy" and trip a false positive. Foreign PIDs are the honest signal.
fn foreign_compute_present() -> bool {
    let mine = std::process::id();
    let Some(target) = pinned_smi_target() else {
        // Cannot prove exclusivity on an unresolved device — treat as foreign.
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
fn resolve_smi_device_maps_visible_ordinal_to_physical_target() {
    // No CUDA_VISIBLE_DEVICES: the ordinal is itself the physical index.
    assert_eq!(resolve_smi_device(None, 0).as_deref(), Some("0"));
    assert_eq!(resolve_smi_device(None, 3).as_deref(), Some("3"));
    assert_eq!(resolve_smi_device(Some(""), 2).as_deref(), Some("2"));

    // A visible list: ordinal indexes into it, yielding the *physical* index.
    // This is the case the old hardcoded `-i 0` got wrong: with the list "2,5"
    // and ordinal 0 the pinned GPU is physical 2, not 0.
    assert_eq!(resolve_smi_device(Some("2,5"), 0).as_deref(), Some("2"));
    assert_eq!(resolve_smi_device(Some("2,5"), 1).as_deref(), Some("5"));
    assert_eq!(resolve_smi_device(Some(" 7 , 1 "), 0).as_deref(), Some("7"));

    // UUIDs pass through verbatim (nvidia-smi -i accepts a GPU UUID).
    assert_eq!(
        resolve_smi_device(Some("GPU-abc123,GPU-def456"), 1).as_deref(),
        Some("GPU-def456")
    );

    // Ordinal past the end of the visible list is unresolvable.
    assert_eq!(resolve_smi_device(Some("2,5"), 2), None);
    // CUDA truncates the visible list at the first empty token.
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
    let packed_buf = upload(&ep, &packed);
    let scale_buf = upload(&ep, &scale);
    let out_buf = ep.allocate(out_bytes, 256).unwrap();
    let ptrs = PlanarLinearPtrs {
        activation: cuptr(a_buf.as_ptr()),
        packed: cuptr(packed_buf.as_ptr()),
        scale: cuptr(scale_buf.as_ptr()),
        output: cuptr(out_buf.as_ptr()),
    };
    warm_planar_linear(runtime).unwrap();

    // 8 s ramp: keep the SMs busy so clocks reach steady state before timing.
    let ramp = Instant::now();
    while ramp.elapsed().as_secs_f32() < 8.0 {
        for _ in 0..32 {
            launch_planar_linear(runtime, PlanarActivationDtype::F16, &dims, &ptrs).unwrap();
        }
        runtime.synchronize().unwrap();
    }

    // Host enqueue cost: time N launches without a sync in the loop.
    let enqueue_n = 200;
    let host = Instant::now();
    for _ in 0..enqueue_n {
        launch_planar_linear(runtime, PlanarActivationDtype::F16, &dims, &ptrs).unwrap();
    }
    let host_enqueue = host.elapsed();
    runtime.synchronize().unwrap();

    // Batched kernel window: median of n≥3 batches of `batch` launches each.
    // Mid-measurement exclusivity is checked by *foreign compute PIDs*, not
    // utilization: our own batched kernels drive utilization to 100% by design,
    // so a utilization sample would false-trip on self-load. A foreign resident
    // process is the honest "GPU became busy" signal.
    let batch = 64usize;
    let sample_shape = |samples: &mut Vec<f64>| {
        for _ in 0..5 {
            assert!(
                !foreign_compute_present(),
                "a foreign compute process appeared on the pinned GPU mid-measurement"
            );
            let t = Instant::now();
            for _ in 0..batch {
                launch_planar_linear(runtime, PlanarActivationDtype::F16, &dims, &ptrs).unwrap();
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

    // First-shape drift recheck: re-measure the identical shape after the run
    // and report drift vs the first median. A large drift means the device was
    // not in steady state (throttling / contention) and the number is suspect.
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
        drift < 0.25,
        "first-shape drift {:.1}% exceeds 25%: device not in steady state, measurement is unreliable",
        drift * 100.0
    );

    ep.deallocate(a_buf).unwrap();
    ep.deallocate(packed_buf).unwrap();
    ep.deallocate(scale_buf).unwrap();
    ep.deallocate(out_buf).unwrap();
}
