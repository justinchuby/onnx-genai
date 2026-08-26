//! On-device parity for the DeepSeek-V4 planar B2 **routed top-k MoE** primitive
//! (`onnx_runtime_ep_cuda::launch_planar_moe`).
//!
//! This is the hardware proof that the launched routed-MoE pipeline decodes the
//! exact on-disk byte layout of per-expert planar weights (packed bytes + a
//! *separate* UE8M0 aux-scale bank), routes each token to its top-k experts,
//! runs `fc1 (+ optional fc3 gate) → activate → fc2 → combine`, and matches a CPU
//! oracle composed from the vetted `onnx_runtime_ep_cpu` planar oracle
//! (`planar_block_matmul` / `PlanarExpertBank`) plus a faithful transcription of
//! the routing/activation/combine arithmetic. No host mirror, dequantize copy or
//! dense-expert fallback runs on the device path.
//!
//! Every test is `#[cfg_attr(not(feature = "gpu-tests"), ignore)]`d so a CPU-only
//! run leaves them ignored; enable `--features gpu-tests` on a CUDA runner. The
//! device is selected by `CUDA_VISIBLE_DEVICES` — pin an idle GPU before running.
//!
//! Coverage:
//! * uniform `block_fp8` and uniform `fp4_planar` experts on shape-faithful dims;
//! * **mixed** per-projection formats (fc1 `block_fp8`, fc2 `fp4_planar`);
//! * routing top-k (k>1), softmax weights and pre-aggregated router weights;
//! * activations: ReLU, tanh-GELU, plain SiLU, SwiGLU via a separate fc3 gate,
//!   and fused SwiGLU (`swiglu_fusion = 1`, 2*inter-wide fc1); with/without bias;
//! * multi-request / shape-change on one device (NVRTC cache reuse);
//! * invalid aux / OOB geometry → typed reject (no launch);
//! * CUDA-graph capture + ≥3 replay parity (warmed, no in-capture alloc/sync);
//! * an `#[ignore]`d measurement probe (8 s ramp, idle check, n≥3; no tok/s).

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::uninlined_format_args
)]

use onnx_runtime_ep_api::{DeviceBuffer, ExecutionProvider};
use onnx_runtime_ep_cpu::kernels::planar_block_quant::{
    FP4_MICROSCALE_BLOCK, FP4_PACK_FACTOR, PlanarBlockFormat, PlanarExpertBank, PlanarLayout,
    planar_block_matmul,
};
#[cfg(feature = "gpu-tests")]
use onnx_runtime_ep_cuda::planar_moe_source_build_count;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ep_cuda::{
    AdmittedPlanarMoe, CudaExecutionProvider, PLANAR_FORMAT_BLOCK_FP8, PLANAR_FORMAT_FP4_PLANAR,
    PlanarMoeBank, PlanarMoeBufferLengths, PlanarMoeBuffers, PlanarMoeDims, PlanarMoeProjection,
    admit_planar_moe, launch_planar_moe, planar_moe_capable_formats,
    test_planar_moe_bank_addresses, test_planar_moe_bank_owner_count,
    test_reject_planar_moe_bank_substitution, warm_planar_moe,
};
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
    fn next_f32(&mut self) -> f32 {
        (i16::from(self.next_u8()) - 128) as f32 / 96.0
    }
}

/// A random-but-magnitude-bounded E4M3 byte (sign + mantissa, exponent 1..=7 so
/// magnitude < 2), keeping f32 sums well-conditioned. The full byte range is
/// swept exactly by the matmul slice's `exhaustive_small_codepoints`.
fn bounded_e4m3(byte: u8) -> u8 {
    let sign = byte & 0x80;
    let mant = byte & 0x07;
    let exp = 1 + (byte >> 3) % 7;
    sign | (exp << 3) | mant
}

/// UE8M0 exponents in a tight band around 1.0 (never the reserved `0xff`).
fn benign_scale(byte: u8) -> u8 {
    125 + (byte % 5)
}

fn block_fp8_expert(out: usize, in_features: usize, bs: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let mut rng = Lcg::new(seed);
    let packed = (0..out * in_features)
        .map(|_| bounded_e4m3(rng.next_u8()))
        .collect();
    let scale = (0..out.div_ceil(bs) * in_features.div_ceil(bs))
        .map(|_| benign_scale(rng.next_u8()))
        .collect();
    (packed, scale)
}

fn fp4_expert(out: usize, in_features: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let mut rng = Lcg::new(seed);
    let packed = (0..out * (in_features / FP4_PACK_FACTOR))
        .map(|_| rng.next_u8())
        .collect();
    let scale = (0..out * (in_features / FP4_MICROSCALE_BLOCK))
        .map(|_| benign_scale(rng.next_u8()))
        .collect();
    (packed, scale)
}

// ---------------------------------------------------------------------------
// A projection's per-expert planar banks + oracle helpers
// ---------------------------------------------------------------------------

/// One projection (fc1/fc2/fc3) materialised for both device and oracle: the
/// expert-major concatenated packed/scale banks (exactly what the device kernel
/// slices per routed expert) and a `PlanarExpertBank` for the CPU oracle.
struct Projection {
    format: i32,
    in_features: usize,
    out_features: usize,
    bs0: usize,
    bs1: usize,
    packed_bank: Vec<u8>,
    scale_bank: Vec<u8>,
    bias: Option<Vec<f32>>,
    bank: PlanarExpertBank,
    layout: PlanarLayout,
}

impl Projection {
    fn build(
        format: i32,
        in_features: usize,
        out_features: usize,
        experts: usize,
        seed: u64,
        with_bias: bool,
    ) -> Self {
        let (cpu_format, bs0, bs1) = if format == PLANAR_FORMAT_BLOCK_FP8 {
            (PlanarBlockFormat::BlockFp8, 128usize, 128usize)
        } else {
            (PlanarBlockFormat::Fp4Planar, 1usize, FP4_MICROSCALE_BLOCK)
        };
        let layout = PlanarLayout::new(cpu_format, out_features, in_features, bs0, bs1).unwrap();
        let mut per_expert: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(experts);
        for e in 0..experts {
            let s = seed ^ (0x1000 * e as u64 + 1);
            if format == PLANAR_FORMAT_BLOCK_FP8 {
                per_expert.push(block_fp8_expert(out_features, in_features, bs0, s));
            } else {
                per_expert.push(fp4_expert(out_features, in_features, s));
            }
        }
        let mut packed_bank = Vec::new();
        let mut scale_bank = Vec::new();
        for (p, sc) in &per_expert {
            packed_bank.extend_from_slice(p);
            scale_bank.extend_from_slice(sc);
        }
        let refs: Vec<(&[u8], &[u8])> = per_expert
            .iter()
            .map(|(p, s)| (p.as_slice(), s.as_slice()))
            .collect();
        let bank = PlanarExpertBank::stack(layout, &refs).unwrap();
        let bias = with_bias.then(|| {
            let mut rng = Lcg::new(seed ^ 0xB1A5);
            (0..experts * out_features)
                .map(|_| rng.next_f32())
                .collect()
        });
        Self {
            format,
            in_features,
            out_features,
            bs0,
            bs1,
            packed_bank,
            scale_bank,
            bias,
            bank,
            layout,
        }
    }

    fn descriptor(&self) -> PlanarMoeProjection {
        PlanarMoeProjection {
            format: self.format,
            in_features: self.in_features,
            out_features: self.out_features,
            bs0: self.bs0,
            bs1: self.bs1,
        }
    }

    fn admission_bank(&self) -> PlanarMoeBank<'_> {
        PlanarMoeBank {
            packed: &self.packed_bank,
            scale: &self.scale_bank,
            bias_elems: self.bias.as_ref().map(Vec::len),
        }
    }

    /// CPU oracle: decode expert `expert`'s planar weight and contract it against
    /// one input row `[in_features]`, adding bias. Returns `[out_features]`.
    fn linear_row(&self, expert: usize, input_row: &[f32]) -> Vec<f32> {
        let packed = self.bank.expert_packed(expert).unwrap();
        let scale = self.bank.expert_scale(expert).unwrap();
        let mut out = planar_block_matmul(input_row, 1, &self.layout, packed, scale).unwrap();
        if let Some(bias) = &self.bias {
            let base = expert * self.out_features;
            for (o, v) in out.iter_mut().enumerate() {
                *v += bias[base + o];
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// CPU oracle: faithful transcription of bqmoe_route / bqmoe_activate /
// bqmoe_combine_f32 composed with the planar linear oracle above.
// ---------------------------------------------------------------------------

fn total_order_key(value: f32) -> i32 {
    let bits = value.to_bits() as i32;
    bits ^ ((bits >> 31) & 0x7fff_ffff)
}

fn route_is_better(cand: f32, cand_i: usize, best: f32, best_i: usize) -> bool {
    let ck = total_order_key(cand);
    let bk = total_order_key(best);
    ck > bk || (ck == bk && cand_i < best_i)
}

fn stable_sigmoid(v: f32) -> f32 {
    if v >= 0.0 {
        1.0 / (1.0 + (-v).exp())
    } else {
        let e = v.exp();
        e / (1.0 + e)
    }
}

fn swiglu_value(gate: f32, linear: f32, alpha: f32, beta: f32, limit: f32) -> f32 {
    let bounded_gate = gate.min(limit);
    let bounded_linear = if linear.is_nan() {
        linear
    } else {
        linear.clamp(-limit, limit)
    };
    bounded_gate * stable_sigmoid(alpha * bounded_gate) * (bounded_linear + beta)
}

struct OracleInputs<'a> {
    dims: &'a PlanarMoeDims,
    input: &'a [f32],
    router_logits: &'a [f32],
    router_weights: Option<&'a [f32]>,
    fc1: &'a Projection,
    fc2: &'a Projection,
    fc3: Option<&'a Projection>,
}

fn moe_cpu_oracle(o: &OracleInputs) -> Vec<f32> {
    let d = o.dims;
    let rows = d.rows;
    let experts = d.experts;
    let top_k = d.top_k;
    let inter = d.inter;
    let hidden = d.hidden;

    let mut output = vec![0.0f32; rows * hidden];
    for row in 0..rows {
        let logits = &o.router_logits[row * experts..][..experts];

        // Greedy total-order top-k (matches bqmoe_route slot-by-slot selection).
        let mut selected = vec![usize::MAX; top_k];
        for slot in 0..top_k {
            let mut best_i: isize = -1;
            let mut best_v = 0.0f32;
            for expert in 0..experts {
                if selected[..slot].contains(&expert) {
                    continue;
                }
                let cand = logits[expert];
                if best_i < 0 || route_is_better(cand, expert, best_v, best_i as usize) {
                    best_i = expert as isize;
                    best_v = cand;
                }
            }
            selected[slot] = best_i as usize;
        }

        // Routed weights.
        let mut weights = vec![0.0f32; top_k];
        if let Some(agg) = o.router_weights {
            let agg_row = &agg[row * experts..][..experts];
            let denom = if d.normalize_routing_weights {
                selected.iter().map(|&e| agg_row[e]).sum::<f32>()
            } else {
                1.0
            };
            for slot in 0..top_k {
                weights[slot] = if denom == 0.0 {
                    0.0
                } else {
                    agg_row[selected[slot]] / denom
                };
            }
        } else {
            let maximum = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let all_sum: f32 = logits.iter().map(|&l| (l - maximum).exp()).sum();
            let denom = if d.normalize_routing_weights {
                selected.iter().map(|&e| (logits[e] - maximum).exp()).sum()
            } else {
                all_sum
            };
            for slot in 0..top_k {
                weights[slot] = (logits[selected[slot]] - maximum).exp() / denom;
            }
        }

        // Per-route expert path, then weighted combine into this row.
        let input_row = &o.input[row * hidden..][..hidden];
        for slot in 0..top_k {
            let expert = selected[slot];
            let fc1_row = o.fc1.linear_row(expert, input_row); // [fc1_out]
            let fc3_row = o.fc3.map(|fc3| fc3.linear_row(expert, input_row)); // [inter]

            let mut activated = vec![0.0f32; inter];
            for feature in 0..inter {
                let value = fc1_row[feature];
                activated[feature] = if d.activation == 0 {
                    value.max(0.0)
                } else if d.activation == 1 {
                    let x = value as f64;
                    let inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
                    (0.5 * x * (1.0 + inner.tanh())) as f32
                } else if d.activation == 2 && fc3_row.is_none() {
                    value * stable_sigmoid(value)
                } else if d.activation == 4 {
                    value
                } else {
                    let (gate, linear) = if let Some(fc3_row) = &fc3_row {
                        (value, fc3_row[feature])
                    } else if d.swiglu_fusion == 1 {
                        (fc1_row[2 * feature], fc1_row[2 * feature + 1])
                    } else {
                        (value, fc1_row[inter + feature])
                    };
                    swiglu_value(
                        gate,
                        linear,
                        d.activation_alpha,
                        d.activation_beta,
                        d.swiglu_limit,
                    )
                };
            }

            let route_out = o.fc2.linear_row(expert, &activated); // [hidden]
            let w = weights[slot];
            for feature in 0..hidden {
                output[row * hidden + feature] += w * route_out[feature];
            }
        }
    }
    output
}

// ---------------------------------------------------------------------------
// Device upload/download + launch
// ---------------------------------------------------------------------------

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn upload(ep: &CudaExecutionProvider, bytes: &[u8]) -> DeviceBuffer {
    let buffer = ep.allocate(bytes.len().max(1), 256).unwrap();
    if !bytes.is_empty() {
        // SAFETY: `buffer` is a fresh device allocation at least `bytes.len()`
        // wide; the copy stays in bounds.
        unsafe { ep.runtime().htod(bytes, cuptr(buffer.as_ptr())).unwrap() };
    }
    buffer
}

fn upload_f32(ep: &CudaExecutionProvider, values: &[f32]) -> DeviceBuffer {
    upload(ep, &f32_bytes(values))
}

fn download_f32(ep: &CudaExecutionProvider, buffer: &DeviceBuffer, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * 4];
    // SAFETY: `buffer` is at least `len * 4` bytes wide.
    unsafe {
        ep.runtime()
            .dtoh(&mut bytes, cuptr(buffer.as_ptr()))
            .unwrap()
    };
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Device-owned non-weight inputs, optional biases, workspace, and output.
/// Sealed packed/scale banks live in `AdmittedPlanarMoe`.
struct MoeDeviceBuffers {
    input: DeviceBuffer,
    router_logits: DeviceBuffer,
    router_weights: Option<DeviceBuffer>,
    fc1_bias: Option<DeviceBuffer>,
    fc2_bias: Option<DeviceBuffer>,
    fc3_bias: Option<DeviceBuffer>,
    route_indices: DeviceBuffer,
    route_weights: DeviceBuffer,
    fc1_output: DeviceBuffer,
    fc3_output: Option<DeviceBuffer>,
    activated: DeviceBuffer,
    route_output: DeviceBuffer,
    output: DeviceBuffer,
}

impl MoeDeviceBuffers {
    fn launch_buffers(&mut self) -> PlanarMoeBuffers<'_> {
        PlanarMoeBuffers {
            input: &self.input,
            router_logits: &self.router_logits,
            router_weights: self.router_weights.as_ref(),
            fc1_bias: self.fc1_bias.as_ref(),
            fc2_bias: self.fc2_bias.as_ref(),
            fc3_bias: self.fc3_bias.as_ref(),
            route_indices: &mut self.route_indices,
            route_weights: &mut self.route_weights,
            fc1_output: &mut self.fc1_output,
            fc3_output: self.fc3_output.as_mut(),
            activated: &mut self.activated,
            route_output: &mut self.route_output,
            output: &mut self.output,
        }
    }
}

fn stage_moe(
    ep: &Arc<CudaExecutionProvider>,
    dims: &PlanarMoeDims,
    input: &[f32],
    router_logits: &[f32],
    router_weights: Option<&[f32]>,
    fc1: &Projection,
    fc2: &Projection,
    fc3: Option<&Projection>,
) -> MoeDeviceBuffers {
    let routes = dims.routes();
    let fc1_out = dims.fc1_out();
    MoeDeviceBuffers {
        input: upload_f32(ep, input),
        router_logits: upload_f32(ep, router_logits),
        router_weights: router_weights.map(|weights| upload_f32(ep, weights)),
        fc1_bias: fc1.bias.as_deref().map(|bias| upload_f32(ep, bias)),
        fc2_bias: fc2.bias.as_deref().map(|bias| upload_f32(ep, bias)),
        fc3_bias: fc3
            .and_then(|projection| projection.bias.as_deref())
            .map(|bias| upload_f32(ep, bias)),
        route_indices: ep.allocate((routes * 4).max(1), 256).unwrap(),
        route_weights: ep.allocate((routes * 4).max(1), 256).unwrap(),
        fc1_output: ep.allocate((routes * fc1_out * 4).max(1), 256).unwrap(),
        fc3_output: fc3
            .is_some()
            .then(|| ep.allocate((routes * dims.inter * 4).max(1), 256).unwrap()),
        activated: ep.allocate((routes * dims.inter * 4).max(1), 256).unwrap(),
        route_output: ep.allocate((routes * dims.hidden * 4).max(1), 256).unwrap(),
        output: ep
            .allocate((dims.rows * dims.hidden * 4).max(1), 256)
            .unwrap(),
    }
}

fn free_moe(ep: &CudaExecutionProvider, buffers: MoeDeviceBuffers) {
    let MoeDeviceBuffers {
        input,
        router_logits,
        router_weights,
        fc1_bias,
        fc2_bias,
        fc3_bias,
        route_indices,
        route_weights,
        fc1_output,
        fc3_output,
        activated,
        route_output,
        output,
    } = buffers;
    for buffer in [router_weights, fc1_bias, fc2_bias, fc3_bias, fc3_output]
        .into_iter()
        .flatten()
    {
        ep.deallocate(buffer).unwrap();
    }
    for buffer in [
        input,
        router_logits,
        route_indices,
        route_weights,
        fc1_output,
        activated,
        route_output,
        output,
    ] {
        ep.deallocate(buffer).unwrap();
    }
}

fn admit(
    ep: &Arc<CudaExecutionProvider>,
    dims: &PlanarMoeDims,
    fc1: &Projection,
    fc2: &Projection,
    fc3: Option<&Projection>,
    has_router_weights: bool,
) -> AdmittedPlanarMoe {
    let buffer_lengths = PlanarMoeBufferLengths::for_dims(dims, has_router_weights).unwrap();
    admit_planar_moe(
        ep,
        dims,
        fc1.admission_bank(),
        fc2.admission_bank(),
        fc3.map(Projection::admission_bank),
        &buffer_lengths,
    )
    .expect("planar MoE geometry/banks must admit")
}

fn assert_parity(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = 2e-3 * w.abs().max(1.0) + 3e-4;
        assert!(
            (g - w).abs() <= tol,
            "{label} out[{i}]: got {g}, want {w}, tol {tol}"
        );
    }
}

/// Full staged run: build inputs, oracle, stage, validate, warm, launch, sync,
/// download, assert parity, free.
fn run_moe_case(
    ep: &Arc<CudaExecutionProvider>,
    label: &str,
    dims: &PlanarMoeDims,
    fc1: &Projection,
    fc2: &Projection,
    fc3: Option<&Projection>,
    use_router_weights: bool,
    seed: u64,
) {
    let mut rng = Lcg::new(seed);
    let input: Vec<f32> = (0..dims.rows * dims.hidden)
        .map(|_| rng.next_f32())
        .collect();
    let router_logits: Vec<f32> = (0..dims.rows * dims.experts)
        .map(|_| rng.next_f32() * 3.0)
        .collect();
    let router_weights: Option<Vec<f32>> = use_router_weights.then(|| {
        (0..dims.rows * dims.experts)
            .map(|_| rng.next_u8() as f32 / 255.0 + 0.05)
            .collect()
    });

    let want = moe_cpu_oracle(&OracleInputs {
        dims,
        input: &input,
        router_logits: &router_logits,
        router_weights: router_weights.as_deref(),
        fc1,
        fc2,
        fc3,
    });

    let admission = admit(ep, dims, fc1, fc2, fc3, use_router_weights);
    let mut buffers = stage_moe(
        ep,
        dims,
        &input,
        &router_logits,
        router_weights.as_deref(),
        fc1,
        fc2,
        fc3,
    );
    warm_planar_moe(ep.runtime()).unwrap();
    launch_planar_moe(&admission, &mut buffers.launch_buffers()).unwrap();
    ep.runtime().synchronize().unwrap();
    let got = download_f32(ep, &buffers.output, dims.rows * dims.hidden);
    assert_parity(label, &got, &want);
    free_moe(ep, buffers);
}

fn dims_relu(
    rows: usize,
    hidden: usize,
    inter: usize,
    experts: usize,
    top_k: usize,
    fc1: &Projection,
    fc2: &Projection,
) -> PlanarMoeDims {
    PlanarMoeDims {
        rows,
        hidden,
        inter,
        experts,
        top_k,
        activation: 0,
        swiglu_fusion: 0,
        activation_alpha: 1.0,
        activation_beta: 1.0,
        swiglu_limit: f32::MAX,
        normalize_routing_weights: true,
        fc1: fc1.descriptor(),
        fc2: fc2.descriptor(),
        fc3: None,
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
fn block_fp8_routed_moe_matches_oracle() {
    let ep = require_cuda();
    let (hidden, inter, experts, top_k) = (256usize, 128usize, 4usize, 2usize);
    let fc1 = Projection::build(PLANAR_FORMAT_BLOCK_FP8, hidden, inter, experts, 0x1, false);
    let fc2 = Projection::build(PLANAR_FORMAT_BLOCK_FP8, inter, hidden, experts, 0x2, false);
    let dims = dims_relu(3, hidden, inter, experts, top_k, &fc1, &fc2);
    run_moe_case(
        &ep,
        "block_fp8 relu",
        &dims,
        &fc1,
        &fc2,
        None,
        false,
        0xABCD,
    );
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn fp4_planar_routed_moe_matches_oracle() {
    let ep = require_cuda();
    let (hidden, inter, experts, top_k) = (128usize, 64usize, 6usize, 3usize);
    let fc1 = Projection::build(PLANAR_FORMAT_FP4_PLANAR, hidden, inter, experts, 0x3, true);
    let fc2 = Projection::build(PLANAR_FORMAT_FP4_PLANAR, inter, hidden, experts, 0x4, true);
    let dims = dims_relu(4, hidden, inter, experts, top_k, &fc1, &fc2);
    run_moe_case(
        &ep,
        "fp4_planar relu bias",
        &dims,
        &fc1,
        &fc2,
        None,
        false,
        0xBEEF,
    );
}

/// Real DeepSeek-style per-projection mix: block-FP8 gate/up, planar-FP4 down.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn mixed_projection_routed_moe_matches_oracle() {
    let ep = require_cuda();
    let (hidden, inter, experts, top_k) = (256usize, 96usize, 5usize, 2usize);
    let fc1 = Projection::build(PLANAR_FORMAT_BLOCK_FP8, hidden, inter, experts, 0x5, true);
    let fc2 = Projection::build(PLANAR_FORMAT_FP4_PLANAR, inter, hidden, experts, 0x6, true);
    let dims = dims_relu(3, hidden, inter, experts, top_k, &fc1, &fc2);
    run_moe_case(&ep, "mixed fp8/fp4", &dims, &fc1, &fc2, None, false, 0xC0DE);
}

/// Pre-aggregated router weights + normalize (not softmax).
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn router_weights_path_matches_oracle() {
    let ep = require_cuda();
    let (hidden, inter, experts, top_k) = (128usize, 64usize, 4usize, 2usize);
    let fc1 = Projection::build(PLANAR_FORMAT_BLOCK_FP8, hidden, inter, experts, 0x7, false);
    let fc2 = Projection::build(PLANAR_FORMAT_BLOCK_FP8, inter, hidden, experts, 0x8, false);
    let dims = dims_relu(4, hidden, inter, experts, top_k, &fc1, &fc2);
    run_moe_case(&ep, "router_weights", &dims, &fc1, &fc2, None, true, 0xD00D);
}

/// SwiGLU via a separate fc3 gate (fc1 = gate, fc3 = linear), with bias.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn swiglu_fc3_gate_routed_moe_matches_oracle() {
    let ep = require_cuda();
    let (hidden, inter, experts, top_k) = (256usize, 128usize, 4usize, 2usize);
    let fc1 = Projection::build(PLANAR_FORMAT_BLOCK_FP8, hidden, inter, experts, 0x9, true);
    let fc3 = Projection::build(PLANAR_FORMAT_BLOCK_FP8, hidden, inter, experts, 0xA, true);
    let fc2 = Projection::build(PLANAR_FORMAT_BLOCK_FP8, inter, hidden, experts, 0xB, true);
    let dims = PlanarMoeDims {
        rows: 3,
        hidden,
        inter,
        experts,
        top_k,
        activation: 3,
        swiglu_fusion: 0,
        activation_alpha: 1.702,
        activation_beta: 1.0,
        swiglu_limit: 7.0,
        normalize_routing_weights: true,
        fc1: fc1.descriptor(),
        fc2: fc2.descriptor(),
        fc3: Some(fc3.descriptor()),
    };
    run_moe_case(
        &ep,
        "swiglu fc3",
        &dims,
        &fc1,
        &fc2,
        Some(&fc3),
        false,
        0xEE11,
    );
}

/// Fused SwiGLU: fc1 is 2*inter wide (interleaved gate/linear), no fc3.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn fused_swiglu_routed_moe_matches_oracle() {
    let ep = require_cuda();
    let (hidden, inter, experts, top_k) = (256usize, 128usize, 4usize, 2usize);
    let fc1 = Projection::build(
        PLANAR_FORMAT_BLOCK_FP8,
        hidden,
        2 * inter,
        experts,
        0xC,
        false,
    );
    let fc2 = Projection::build(PLANAR_FORMAT_BLOCK_FP8, inter, hidden, experts, 0xD, false);
    let dims = PlanarMoeDims {
        rows: 3,
        hidden,
        inter,
        experts,
        top_k,
        activation: 3,
        swiglu_fusion: 1,
        activation_alpha: 1.0,
        activation_beta: 1.0,
        swiglu_limit: f32::MAX,
        normalize_routing_weights: true,
        fc1: fc1.descriptor(),
        fc2: fc2.descriptor(),
        fc3: None,
    };
    run_moe_case(&ep, "fused swiglu", &dims, &fc1, &fc2, None, false, 0xFACE);
}

/// tanh-GELU and plain SiLU activations.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn gelu_and_silu_activations_match_oracle() {
    let ep = require_cuda();
    let (hidden, inter, experts, top_k) = (192usize, 64usize, 4usize, 2usize);
    let fc1 = Projection::build(PLANAR_FORMAT_BLOCK_FP8, hidden, inter, experts, 0x11, false);
    let fc2 = Projection::build(PLANAR_FORMAT_BLOCK_FP8, inter, hidden, experts, 0x12, false);
    for (activation, label) in [(1i32, "gelu"), (2i32, "silu")] {
        let mut dims = dims_relu(3, hidden, inter, experts, top_k, &fc1, &fc2);
        dims.activation = activation;
        run_moe_case(
            &ep,
            label,
            &dims,
            &fc1,
            &fc2,
            None,
            false,
            0x9000 + activation as u64,
        );
    }
}

/// Repeated launches with changing shapes on one device: NVRTC cache is warmed
/// once, every shape still lands exact, buffers recycle cleanly.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn multi_request_shape_change_is_stable() {
    let ep = require_cuda();
    for &(rows, hidden, inter, experts, top_k) in &[
        (1usize, 128usize, 64usize, 4usize, 1usize),
        (5, 256, 128, 8, 2),
        (2, 128, 64, 4, 3),
        (1, 128, 64, 4, 1),
    ] {
        let fc1 = Projection::build(PLANAR_FORMAT_BLOCK_FP8, hidden, inter, experts, 0x20, false);
        let fc2 = Projection::build(
            PLANAR_FORMAT_FP4_PLANAR,
            inter,
            hidden,
            experts,
            0x21,
            false,
        );
        let dims = dims_relu(rows, hidden, inter, experts, top_k, &fc1, &fc2);
        run_moe_case(
            &ep,
            "multi-shape",
            &dims,
            &fc1,
            &fc2,
            None,
            false,
            0x5000 + rows as u64,
        );
    }
}

/// Invalid aux / OOB geometry must typed-reject before any launch.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn invalid_geometry_is_typed_rejected() {
    let ep = require_cuda();
    let (hidden, inter, experts, top_k) = (128usize, 64usize, 4usize, 2usize);
    let fc1 = Projection::build(PLANAR_FORMAT_BLOCK_FP8, hidden, inter, experts, 0x30, false);
    let fc2 = Projection::build(PLANAR_FORMAT_BLOCK_FP8, inter, hidden, experts, 0x31, false);
    let dims = dims_relu(2, hidden, inter, experts, top_k, &fc1, &fc2);

    // Ragged packed bank (one byte too many) → validate rejects.
    let buffer_lengths = PlanarMoeBufferLengths::for_dims(&dims, false).unwrap();
    let mut ragged_fc1 = fc1.packed_bank.clone();
    ragged_fc1.push(0);
    assert!(
        admit_planar_moe(
            &ep,
            &dims,
            PlanarMoeBank {
                packed: &ragged_fc1,
                scale: &fc1.scale_bank,
                bias_elems: None,
            },
            fc2.admission_bank(),
            None,
            &buffer_lengths,
        )
        .is_err()
    );

    // top_k > experts → validate rejects.
    let mut bad_topk = dims;
    bad_topk.top_k = experts + 1;
    let bad_buffer_lengths = PlanarMoeBufferLengths::for_dims(&bad_topk, false).unwrap();
    assert!(
        admit_planar_moe(
            &ep,
            &bad_topk,
            fc1.admission_bank(),
            fc2.admission_bank(),
            None,
            &bad_buffer_lengths,
        )
        .is_err()
    );

    // Undersized workspace → validate rejects before launch.
    let mut short_buffers = buffer_lengths;
    short_buffers.route_output_elems -= 1;
    assert!(
        admit_planar_moe(
            &ep,
            &dims,
            fc1.admission_bank(),
            fc2.admission_bank(),
            None,
            &short_buffers,
        )
        .is_err()
    );

    // Odd fp4 contraction on fc2 cannot produce an admission proof.
    let odd_dims = PlanarMoeDims {
        rows: 1,
        hidden: 32,
        inter: 33,
        experts: 2,
        top_k: 1,
        activation: 0,
        swiglu_fusion: 0,
        activation_alpha: 1.0,
        activation_beta: 1.0,
        swiglu_limit: f32::MAX,
        normalize_routing_weights: true,
        fc1: PlanarMoeProjection {
            format: PLANAR_FORMAT_BLOCK_FP8,
            in_features: 32,
            out_features: 33,
            bs0: 128,
            bs1: 128,
        },
        fc2: PlanarMoeProjection {
            format: PLANAR_FORMAT_FP4_PLANAR,
            in_features: 33, // odd → invalid fp4 contraction
            out_features: 32,
            bs0: 1,
            bs1: FP4_MICROSCALE_BLOCK,
        },
        fc3: None,
    };
    let odd_buffers = PlanarMoeBufferLengths::for_dims(&odd_dims, false).unwrap();
    let empty = PlanarMoeBank {
        packed: &[],
        scale: &[],
        bias_elems: None,
    };
    assert!(admit_planar_moe(&ep, &odd_dims, empty, empty, None, &odd_buffers).is_err());
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn malformed_values_and_attributes_reject_before_device_activity() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let allocations = runtime.allocation_counts();
    let transfers = runtime.transfer_counts();
    let (hidden, inter, experts, top_k) = (32usize, 32usize, 2usize, 1usize);
    let fc1 = Projection::build(PLANAR_FORMAT_BLOCK_FP8, hidden, inter, experts, 0x61, false);
    let fc2 = Projection::build(PLANAR_FORMAT_BLOCK_FP8, inter, hidden, experts, 0x62, false);
    let dims = dims_relu(1, hidden, inter, experts, top_k, &fc1, &fc2);
    let buffers = PlanarMoeBufferLengths::for_dims(&dims, false).unwrap();

    let mut bad_packed = fc1.packed_bank.clone();
    bad_packed[0] = 0x7f;
    assert!(
        admit_planar_moe(
            &ep,
            &dims,
            PlanarMoeBank {
                packed: &bad_packed,
                scale: &fc1.scale_bank,
                bias_elems: None,
            },
            fc2.admission_bank(),
            None,
            &buffers,
        )
        .is_err()
    );

    bad_packed[0] = 0x7e;
    let mut bad_scale = fc1.scale_bank.clone();
    bad_scale[0] = 247;
    assert!(
        admit_planar_moe(
            &ep,
            &dims,
            PlanarMoeBank {
                packed: &bad_packed,
                scale: &bad_scale,
                bias_elems: None,
            },
            fc2.admission_bank(),
            None,
            &buffers,
        )
        .is_err()
    );

    let mut reserved_scale = fc2.scale_bank.clone();
    reserved_scale[0] = 0xff;
    assert!(
        admit_planar_moe(
            &ep,
            &dims,
            fc1.admission_bank(),
            PlanarMoeBank {
                packed: &fc2.packed_bank,
                scale: &reserved_scale,
                bias_elems: None,
            },
            None,
            &buffers,
        )
        .is_err()
    );

    let fp4_fc1 = Projection::build(
        PLANAR_FORMAT_FP4_PLANAR,
        hidden,
        inter,
        experts,
        0x63,
        false,
    );
    let fp4_fc2 = Projection::build(
        PLANAR_FORMAT_FP4_PLANAR,
        inter,
        hidden,
        experts,
        0x64,
        false,
    );
    let fp4_dims = dims_relu(1, hidden, inter, experts, top_k, &fp4_fc1, &fp4_fc2);
    let fp4_buffers = PlanarMoeBufferLengths::for_dims(&fp4_dims, false).unwrap();
    let max_codes = vec![0x77u8; fp4_fc1.packed_bank.len()];
    let overflow_scales = vec![253u8; fp4_fc1.scale_bank.len()];
    assert!(
        admit_planar_moe(
            &ep,
            &fp4_dims,
            PlanarMoeBank {
                packed: &max_codes,
                scale: &overflow_scales,
                bias_elems: None,
            },
            fp4_fc2.admission_bank(),
            None,
            &fp4_buffers,
        )
        .is_err()
    );

    for (alpha, beta, limit) in [
        (f32::NAN, 0.0, 1.0),
        (f32::INFINITY, 0.0, 1.0),
        (f32::NEG_INFINITY, 0.0, 1.0),
        (1.0, f32::NAN, 1.0),
        (1.0, f32::INFINITY, 1.0),
        (1.0, f32::NEG_INFINITY, 1.0),
        (1.0, 0.0, f32::NAN),
        (1.0, 0.0, f32::INFINITY),
        (1.0, 0.0, f32::NEG_INFINITY),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, -1.0),
    ] {
        let invalid = PlanarMoeDims {
            activation: 3,
            swiglu_fusion: 1,
            activation_alpha: alpha,
            activation_beta: beta,
            swiglu_limit: limit,
            ..dims
        };
        assert!(
            admit_planar_moe(
                &ep,
                &invalid,
                fc1.admission_bank(),
                fc2.admission_bank(),
                None,
                &buffers,
            )
            .is_err()
        );
    }

    assert_eq!(runtime.allocation_counts(), allocations);
    assert_eq!(runtime.transfer_counts(), transfers);
}

/// Every projection bank is sealed independently. Safe routed-MoE launch has no
/// raw packed/scale arguments, and the private unsafe boundary rejects all
/// external allocations regardless of matching geometry or malicious contents.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn sealed_moe_admission_rejects_projection_substitution() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let (hidden, inter, experts, top_k) = (32usize, 32usize, 2usize, 1usize);
    let mut fc1 = Projection::build(
        PLANAR_FORMAT_BLOCK_FP8,
        hidden,
        inter,
        experts,
        0x701,
        false,
    );
    let mut fc2 = Projection::build(
        PLANAR_FORMAT_FP4_PLANAR,
        inter,
        hidden,
        experts,
        0x702,
        false,
    );
    let mut fc3 = Projection::build(
        PLANAR_FORMAT_BLOCK_FP8,
        hidden,
        inter,
        experts,
        0x703,
        false,
    );
    let dims = PlanarMoeDims {
        rows: 1,
        hidden,
        inter,
        experts,
        top_k,
        activation: 3,
        swiglu_fusion: 0,
        activation_alpha: 1.0,
        activation_beta: 1.0,
        swiglu_limit: 7.0,
        normalize_routing_weights: true,
        fc1: fc1.descriptor(),
        fc2: fc2.descriptor(),
        fc3: Some(fc3.descriptor()),
    };
    let admission = admit(&ep, &dims, &fc1, &fc2, Some(&fc3), false);
    let input = vec![1.0f32; hidden];
    let logits = vec![1.0f32, -1.0];
    let mut buffers = stage_moe(&ep, &dims, &input, &logits, None, &fc1, &fc2, Some(&fc3));
    warm_planar_moe(runtime).unwrap();
    launch_planar_moe(&admission, &mut buffers.launch_buffers()).unwrap();
    runtime.synchronize().unwrap();
    let admitted_output = download_f32(&ep, &buffers.output, hidden);

    // Destroy all original host sources after admission. The second launch must
    // still consume the immutable owned copies admitted above.
    for projection in [&mut fc1, &mut fc2, &mut fc3] {
        projection.packed_bank.fill(0x7f);
        projection.scale_bank.fill(0xff);
    }
    launch_planar_moe(&admission, &mut buffers.launch_buffers()).unwrap();
    runtime.synchronize().unwrap();
    assert_eq!(download_f32(&ep, &buffers.output, hidden), admitted_output);

    let originals = [
        (
            vec![0x38; fc1.packed_bank.len()],
            vec![127; fc1.scale_bank.len()],
        ),
        (
            vec![0x11; fc2.packed_bank.len()],
            vec![127; fc2.scale_bank.len()],
        ),
        (
            vec![0x38; fc3.packed_bank.len()],
            vec![127; fc3.scale_bank.len()],
        ),
    ];
    let mut candidates = Vec::new();
    for (packed, scale) in &originals {
        candidates.push((upload(&ep, packed), upload(&ep, scale)));
    }
    let reserved = (upload(&ep, &[0x7f]), upload(&ep, &[0xff]));
    let overflow = (upload(&ep, &[0x7e]), upload(&ep, &[247]));
    let before = (runtime.allocation_counts(), runtime.transfer_counts());
    for (projection, (packed, scale)) in candidates.iter().enumerate() {
        assert!(
            test_reject_planar_moe_bank_substitution(&admission, projection, packed, scale)
                .is_err(),
            "same-geometry projection {projection} substitution must reject"
        );
        assert!(
            test_reject_planar_moe_bank_substitution(
                &admission,
                projection,
                &reserved.0,
                &reserved.1
            )
            .is_err(),
            "reserved-code projection {projection} substitution must reject"
        );
        assert!(
            test_reject_planar_moe_bank_substitution(
                &admission,
                projection,
                &overflow.0,
                &overflow.1
            )
            .is_err(),
            "non-finite-product projection {projection} substitution must reject"
        );
    }
    assert!(
        test_reject_planar_moe_bank_substitution(&admission, 0, &buffers.input, &buffers.output)
            .is_err(),
        "capture input/output storage cannot replace an admitted bank"
    );
    assert_eq!(
        (runtime.allocation_counts(), runtime.transfer_counts()),
        before,
        "substitution rejection must not allocate or transfer"
    );

    for (packed, scale) in candidates {
        ep.deallocate(packed).unwrap();
        ep.deallocate(scale).unwrap();
    }
    ep.deallocate(reserved.0).unwrap();
    ep.deallocate(reserved.1).unwrap();
    ep.deallocate(overflow.0).unwrap();
    ep.deallocate(overflow.1).unwrap();
    free_moe(&ep, buffers);
}

/// A warmed fixed-shape routed MoE records into a CUDA-graph capture and replays
/// ≥3× byte-identically to the eager result (no in-capture alloc/sync).
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn capture_replay_parity() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let accounting_baseline = ep.device_allocation_counts().unwrap();
    let (hidden, inter, experts, top_k) = (256usize, 128usize, 4usize, 2usize);
    let fc1 = Projection::build(PLANAR_FORMAT_BLOCK_FP8, hidden, inter, experts, 0x40, true);
    let fc2 = Projection::build(PLANAR_FORMAT_FP4_PLANAR, inter, hidden, experts, 0x41, true);
    let dims = dims_relu(3, hidden, inter, experts, top_k, &fc1, &fc2);

    let mut rng = Lcg::new(0x7777);
    let input: Vec<f32> = (0..dims.rows * hidden).map(|_| rng.next_f32()).collect();
    let logits: Vec<f32> = (0..dims.rows * experts)
        .map(|_| rng.next_f32() * 3.0)
        .collect();

    let admission = admit(&ep, &dims, &fc1, &fc2, None, false);
    let mut buffers = stage_moe(&ep, &dims, &input, &logits, None, &fc1, &fc2, None);

    warm_planar_moe(runtime).unwrap();
    #[cfg(feature = "gpu-tests")]
    assert_eq!(planar_moe_source_build_count(), 1);

    // Repeated warmed launches do not rebuild source, allocate, or transfer.
    let warmed_allocations = runtime.allocation_counts();
    let warmed_transfers = runtime.transfer_counts();
    for _ in 0..2 {
        launch_planar_moe(&admission, &mut buffers.launch_buffers()).unwrap();
    }
    #[cfg(feature = "gpu-tests")]
    assert_eq!(planar_moe_source_build_count(), 1);
    assert_eq!(runtime.allocation_counts(), warmed_allocations);
    assert_eq!(runtime.transfer_counts(), warmed_transfers);
    runtime.synchronize().unwrap();
    let eager = download_f32(&ep, &buffers.output, dims.rows * hidden);

    // Capture the warmed pipeline, then replay ≥3× and compare byte-for-byte.
    let capture_allocations = runtime.allocation_counts();
    let capture_transfers = runtime.transfer_counts();
    runtime.begin_graph_capture(&[]).unwrap();
    let capture_buffer_lengths = PlanarMoeBufferLengths::for_dims(&dims, false).unwrap();
    assert!(
        admit_planar_moe(
            &ep,
            &dims,
            fc1.admission_bank(),
            fc2.admission_bank(),
            None,
            &capture_buffer_lengths,
        )
        .is_err(),
        "admission must reject before allocating or uploading during capture"
    );
    launch_planar_moe(&admission, &mut buffers.launch_buffers()).unwrap();
    runtime.end_graph_capture().unwrap();
    assert_eq!(
        test_planar_moe_bank_owner_count(&admission),
        2,
        "the installed graph must hold exactly one strong MoE bank pin"
    );
    #[cfg(feature = "gpu-tests")]
    assert_eq!(planar_moe_source_build_count(), 1);
    assert_eq!(runtime.allocation_counts(), capture_allocations);
    assert_eq!(runtime.transfer_counts(), capture_transfers);

    let bank_addresses = test_planar_moe_bank_addresses(&admission);
    let pinned_frees = ep.device_allocation_counts().unwrap().1;
    drop(admission);
    assert_eq!(
        ep.device_allocation_counts().unwrap().1,
        pinned_frees,
        "dropping the caller handle must not free graph-embedded MoE banks"
    );
    let probes = [
        ep.allocate(fc1.packed_bank.len(), 256).unwrap(),
        ep.allocate(fc1.scale_bank.len(), 256).unwrap(),
        ep.allocate(fc2.packed_bank.len(), 256).unwrap(),
        ep.allocate(fc2.scale_bank.len(), 256).unwrap(),
    ];
    for probe in &probes {
        assert!(
            !bank_addresses.contains(&cuptr(probe.as_ptr())),
            "allocator reused a graph-pinned MoE bank address"
        );
    }
    for probe in probes {
        ep.deallocate(probe).unwrap();
    }
    ep.wait_for_deferred_releases().unwrap();

    let zeros = vec![0u8; dims.rows * hidden * 4];
    for replay in 0..3 {
        let before_allocations = runtime.allocation_counts();
        let before_transfers = runtime.transfer_counts();
        // SAFETY: output is rows*hidden*4 bytes wide.
        unsafe {
            runtime
                .htod(&zeros, cuptr(buffers.output.as_ptr()))
                .unwrap()
        };
        runtime.replay_graph().unwrap();
        runtime.synchronize().unwrap();
        let replayed = download_f32(&ep, &buffers.output, dims.rows * hidden);
        assert_eq!(
            replayed, eager,
            "capture replay {replay} diverged from eager"
        );
        #[cfg(feature = "gpu-tests")]
        assert_eq!(planar_moe_source_build_count(), 1);
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
        before_reset_frees + 4,
        "graph reset must release all four sealed MoE projection allocations exactly once"
    );
    free_moe(&ep, buffers);
    ep.wait_for_deferred_releases().unwrap();
    let settled = ep.device_allocation_counts().unwrap();
    assert_eq!(
        settled.0 - accounting_baseline.0,
        settled.1 - accounting_baseline.1,
        "capture/reset teardown must return MoE allocations to the exact accounting baseline"
    );
}

/// The advertised routed-MoE capability strings must be exactly the two planar
/// formats — and only after the kernels actually compile on this device.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn capability_strings_are_advertised_on_device() {
    let ep = require_cuda();
    warm_planar_moe(ep.runtime()).unwrap();
    assert_eq!(planar_moe_capable_formats(), ["block_fp8", "fp4_planar"]);
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

/// True if a compute process other than this test binary is resident on the
/// pinned device. Used as the mid-measurement tenant guard instead of
/// `utilization.gpu`: our own batched kernels legitimately drive utilization to
/// 100%, and `nvidia-smi`'s rolling-window average would report that self-load
/// as "busy" and trip a false positive. Foreign PIDs are the honest signal.
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
        // If the query fails we cannot prove exclusivity; treat as foreign.
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

/// Warm-then-batch timing of the routed MoE pipeline on the pinned device.
/// Reports median + range of a batched enqueue-to-completion window (n≥3) and
/// host enqueue cost separately. Not a full-model tok/s claim — a single-shape
/// microbench of the routed primitive. `#[ignore]`d so it never runs in the
/// correctness gate; run explicitly on a verified-idle A100.
#[test]
#[ignore = "measurement probe: run explicitly on a verified-idle A100 with --ignored --nocapture"]
fn planar_moe_measurement() {
    use std::time::Instant;
    let ep = require_cuda();
    let runtime = ep.runtime();
    assert!(
        gpu_is_idle(),
        "measurement requires a verified-idle pinned GPU (CUDA_VISIBLE_DEVICES); it was busy"
    );

    let (rows, hidden, inter, experts, top_k) = (16usize, 2048usize, 1024usize, 16usize, 4usize);
    let fc1 = Projection::build(
        PLANAR_FORMAT_BLOCK_FP8,
        hidden,
        inter,
        experts,
        0xF00D,
        false,
    );
    let fc2 = Projection::build(
        PLANAR_FORMAT_FP4_PLANAR,
        inter,
        hidden,
        experts,
        0xF00E,
        false,
    );
    let dims = dims_relu(rows, hidden, inter, experts, top_k, &fc1, &fc2);

    let mut rng = Lcg::new(0xCAFE);
    let input: Vec<f32> = (0..rows * hidden).map(|_| rng.next_f32()).collect();
    let logits: Vec<f32> = (0..rows * experts).map(|_| rng.next_f32() * 3.0).collect();
    let admission = admit(&ep, &dims, &fc1, &fc2, None, false);
    let mut buffers = stage_moe(&ep, &dims, &input, &logits, None, &fc1, &fc2, None);
    warm_planar_moe(runtime).unwrap();

    let ramp = Instant::now();
    while ramp.elapsed().as_secs_f32() < 8.0 {
        for _ in 0..16 {
            launch_planar_moe(&admission, &mut buffers.launch_buffers()).unwrap();
        }
        runtime.synchronize().unwrap();
    }

    let enqueue_n = 100;
    let host = Instant::now();
    for _ in 0..enqueue_n {
        launch_planar_moe(&admission, &mut buffers.launch_buffers()).unwrap();
    }
    let host_enqueue = host.elapsed();
    runtime.synchronize().unwrap();

    let batch = 32usize;
    let mut sample_shape = |samples: &mut Vec<f64>| {
        for _ in 0..5 {
            assert!(
                !foreign_compute_present(),
                "a foreign compute process appeared on the pinned GPU mid-measurement"
            );
            let t = Instant::now();
            for _ in 0..batch {
                launch_planar_moe(&admission, &mut buffers.launch_buffers()).unwrap();
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
        "planar routed MoE [rows={rows} hidden={hidden} inter={inter} E={experts} k={top_k}] batched pipeline/launch: median {:.1} us (min {:.1}, max {:.1}); host enqueue {:.2} us/launch over {enqueue_n}; first-shape recheck median {:.1} us (drift {:.1}%)",
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

    free_moe(&ep, buffers);
}
