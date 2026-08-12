#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::unusual_byte_groupings,
    clippy::doc_lazy_continuation,
    clippy::uninlined_format_args,
    clippy::cloned_ref_to_slice_refs,
    clippy::type_complexity,
    clippy::drop_non_drop,
    clippy::manual_repeat_n,
    clippy::manual_is_multiple_of,
    clippy::err_expect,
    clippy::clone_on_copy
)]
//! GPU correctness tests for `pkg.nxrt::VarlenAttention` v1.
//!
//! `VarlenAttention` consumes the ONNX Attention-24 `nonpad_kv_seqlen` per-batch
//! valid-KV-token count over a padded rectangular batch and runs attention over
//! only the valid keys (no compute on padding). Each case builds a ragged batch,
//! runs it once through the kernel, and compares against an independent CPU
//! reference that computes scaled-dot-product attention over each batch's valid
//! keys with the same tail-aligned causal offset (`nonpad_kv_seqlen[b] - q_seq`).
//! The reference is a from-scratch oracle — it does not call any GPU kernel — so
//! the comparison is meaningful rather than tautological.
//!
//! CPU-only CI reports these tests as ignored unless `gpu-tests` is enabled. Run with:
//! `CUDA_VISIBLE_DEVICES=5 taskset -c 1 cargo test -p onnx-runtime-ep-cuda \
//!   --test varlen_attention_gpu -- --nocapture`

use half::{bf16, f16};
use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, Result, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

const DOMAIN: &str = "pkg.nxrt";

#[derive(Clone)]
struct Tensor {
    dtype: DataType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn f32_tensor(shape: &[usize], values: &[f32]) -> Tensor {
    Tensor {
        dtype: DataType::Float32,
        shape: shape.to_vec(),
        bytes: values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
    }
}

fn f16_tensor(shape: &[usize], values: &[f32]) -> Tensor {
    Tensor {
        dtype: DataType::Float16,
        shape: shape.to_vec(),
        bytes: values
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_bits().to_ne_bytes())
            .collect(),
    }
}

fn bf16_tensor(shape: &[usize], values: &[f32]) -> Tensor {
    Tensor {
        dtype: DataType::BFloat16,
        shape: shape.to_vec(),
        bytes: values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_ne_bytes())
            .collect(),
    }
}

fn i64_tensor(shape: &[usize], values: &[i64]) -> Tensor {
    Tensor {
        dtype: DataType::Int64,
        shape: shape.to_vec(),
        bytes: values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
    }
}

/// Deterministic pseudo-random fill in [-1, 1).
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        })
        .collect()
}

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

/// Decode a device output buffer's raw bytes to `f32`, honoring its dtype.
fn decode(dtype: DataType, bytes: &[u8]) -> Vec<f32> {
    match dtype {
        DataType::Float32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect(),
        DataType::Float16 => bytes
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_ne_bytes(c.try_into().unwrap())).to_f32())
            .collect(),
        DataType::BFloat16 => bytes
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_ne_bytes(c.try_into().unwrap())).to_f32())
            .collect(),
        other => panic!("unexpected output dtype {other:?}"),
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch in comparison");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Build and run a single-node model through the CUDA EP, returning raw output
/// bytes.
fn run_node(
    ep: &CudaExecutionProvider,
    op: &str,
    domain: &str,
    opset: u64,
    inputs: &[Tensor],
    output: (DataType, Vec<usize>),
    attrs: &[(&str, Attribute)],
) -> Result<Vec<u8>> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(domain.to_string(), opset);
    let input_values = inputs
        .iter()
        .enumerate()
        .map(|(i, input)| {
            let value = graph.create_named_value(
                format!("input_{i}"),
                input.dtype,
                static_shape(input.shape.iter().copied()),
            );
            graph.add_input(value);
            value
        })
        .collect::<Vec<_>>();
    let node_inputs = input_values.iter().copied().map(Some).collect::<Vec<_>>();
    let (out_dtype, out_shape) = output;
    let output_value =
        graph.create_named_value("output", out_dtype, static_shape(out_shape.iter().copied()));
    let mut node = Node::new(NodeId(0), op, node_inputs, vec![output_value]);
    node.domain = domain.into();
    for (name, value) in attrs {
        node.attributes.insert((*name).into(), value.clone());
    }
    let node_id = graph.insert_node(node);
    graph.add_output(output_value);

    let model = Model::new(&graph);
    let kernel = ep.get_kernel(model.graph.node(node_id), &[], opset)?;

    let input_buffers = inputs
        .iter()
        .map(|input| -> Result<DeviceBuffer> {
            let buffer = ep.allocate(input.bytes.len().max(1), 256)?;
            if !input.bytes.is_empty() {
                // SAFETY: the allocation exactly covers the source tensor.
                unsafe { ep.runtime().htod(&input.bytes, cuptr(buffer.as_ptr()))? };
            }
            Ok(buffer)
        })
        .collect::<Result<Vec<_>>>()?;
    let input_strides = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect::<Vec<_>>();
    let input_views = inputs
        .iter()
        .zip(&input_buffers)
        .zip(&input_strides)
        .map(|((input, buffer), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                input.dtype,
                &input.shape,
                strides,
                ep.device_id(),
            )
        })
        .collect::<Vec<_>>();

    let out_elems: usize = out_shape.iter().product();
    let out_bytes = out_dtype.storage_bytes(out_elems);
    let mut output_buffer = ep.allocate(out_bytes.max(1), 256)?;
    let output_strides = compute_contiguous_strides(&out_shape);
    let mut output_view = TensorMut::new(
        DevicePtrMut(output_buffer.as_mut_ptr()),
        out_dtype,
        &out_shape,
        &output_strides,
        ep.device_id(),
    );

    let result = kernel.execute(&input_views, std::slice::from_mut(&mut output_view));
    let mut bytes = vec![0u8; out_bytes];
    if result.is_ok() && !bytes.is_empty() {
        // SAFETY: the destination exactly covers the output allocation.
        unsafe {
            ep.runtime()
                .dtoh(&mut bytes, cuptr(output_buffer.as_ptr()))?
        };
    }
    for buffer in input_buffers {
        ep.deallocate(buffer)?;
    }
    ep.deallocate(output_buffer)?;
    result.map(|()| bytes)
}

/// A padded ragged batch: `num_heads`/`kv_num_heads` heads, `head_size`/
/// `v_head_size` per head, a fixed padded `q_seq`, a padded `kv_seq`, and a
/// per-batch valid KV length (`nonpad`).
struct Batch {
    num_heads: usize,
    kv_num_heads: usize,
    head_size: usize,
    v_head_size: usize,
    q_seq: usize,
    kv_seq: usize,
    nonpad: Vec<i64>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl Batch {
    fn new(
        num_heads: usize,
        kv_num_heads: usize,
        head_size: usize,
        v_head_size: usize,
        q_seq: usize,
        nonpad: &[i64],
    ) -> Self {
        let batch = nonpad.len();
        let kv_seq = nonpad.iter().copied().max().unwrap_or(0).max(1) as usize;
        Self {
            num_heads,
            kv_num_heads,
            head_size,
            v_head_size,
            q_seq,
            kv_seq,
            nonpad: nonpad.to_vec(),
            q: fill(batch * q_seq * num_heads * head_size, 0x51),
            k: fill(batch * kv_seq * kv_num_heads * head_size, 0x52),
            v: fill(batch * kv_seq * kv_num_heads * v_head_size, 0x53),
        }
    }

    fn batch(&self) -> usize {
        self.nonpad.len()
    }

    /// Round Q/K/V through the storage dtype so the CPU reference sees the same
    /// operands the kernel loads.
    fn quantized_inputs(&self, dtype: DataType) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let quantize = |xs: &[f32]| match dtype {
            DataType::Float32 => xs.to_vec(),
            DataType::Float16 => xs.iter().map(|&x| f16::from_f32(x).to_f32()).collect(),
            DataType::BFloat16 => xs.iter().map(|&x| bf16::from_f32(x).to_f32()).collect(),
            other => panic!("unsupported varlen test dtype {other:?}"),
        };
        (quantize(&self.q), quantize(&self.k), quantize(&self.v))
    }

    /// Run the varlen kernel. `rank4` picks the `[batch, q_seq, heads, dim]`
    /// layout; otherwise the collapsed `[batch, seq, heads*dim]` rank-3 form
    /// (both index the same flat data).
    fn run(
        &self,
        ep: &CudaExecutionProvider,
        is_causal: bool,
        fp16: bool,
        rank4: bool,
        scale: Option<f32>,
        softcap: f32,
    ) -> Vec<f32> {
        let dtype = if fp16 {
            DataType::Float16
        } else {
            DataType::Float32
        };
        self.run_dtype(ep, is_causal, dtype, rank4, scale, softcap)
    }

    fn run_dtype(
        &self,
        ep: &CudaExecutionProvider,
        is_causal: bool,
        dtype: DataType,
        rank4: bool,
        scale: Option<f32>,
        softcap: f32,
    ) -> Vec<f32> {
        let batch = self.batch();
        let make = |shape: &[usize], data: &[f32]| match dtype {
            DataType::Float32 => f32_tensor(shape, data),
            DataType::Float16 => f16_tensor(shape, data),
            DataType::BFloat16 => bf16_tensor(shape, data),
            other => panic!("unsupported varlen test dtype {other:?}"),
        };
        let q_shape = if rank4 {
            vec![batch, self.q_seq, self.num_heads, self.head_size]
        } else {
            vec![batch, self.q_seq, self.num_heads * self.head_size]
        };
        let k_shape = if rank4 {
            vec![batch, self.kv_seq, self.kv_num_heads, self.head_size]
        } else {
            vec![batch, self.kv_seq, self.kv_num_heads * self.head_size]
        };
        let v_shape = if rank4 {
            vec![batch, self.kv_seq, self.kv_num_heads, self.v_head_size]
        } else {
            vec![batch, self.kv_seq, self.kv_num_heads * self.v_head_size]
        };
        let inputs = vec![
            make(&q_shape, &self.q),
            make(&k_shape, &self.k),
            make(&v_shape, &self.v),
            i64_tensor(&[batch], &self.nonpad),
        ];
        let mut attrs = vec![
            ("num_heads", Attribute::Int(self.num_heads as i64)),
            ("kv_num_heads", Attribute::Int(self.kv_num_heads as i64)),
            ("is_causal", Attribute::Int(is_causal as i64)),
        ];
        if let Some(s) = scale {
            attrs.push(("scale", Attribute::Float(s)));
        }
        if softcap != 0.0 {
            attrs.push(("softcap", Attribute::Float(softcap)));
        }
        let bytes = run_node(
            ep,
            "VarlenAttention",
            DOMAIN,
            1,
            &inputs,
            (
                dtype,
                vec![batch, self.q_seq, self.num_heads, self.v_head_size],
            ),
            &attrs,
        )
        .expect("varlen attention must run");
        decode(dtype, &bytes)
    }

    /// Independent CPU oracle: scaled-dot-product attention over each batch's
    /// valid keys with the tail-aligned causal offset `nonpad[b] - q_seq`.
    fn cpu_reference(
        &self,
        is_causal: bool,
        fp16: bool,
        scale: Option<f32>,
        softcap: f32,
    ) -> Vec<f32> {
        let dtype = if fp16 {
            DataType::Float16
        } else {
            DataType::Float32
        };
        self.cpu_reference_dtype(is_causal, dtype, scale, softcap)
    }

    fn cpu_reference_dtype(
        &self,
        is_causal: bool,
        dtype: DataType,
        scale: Option<f32>,
        softcap: f32,
    ) -> Vec<f32> {
        let (q, k, v) = self.quantized_inputs(dtype);
        let group = self.num_heads / self.kv_num_heads;
        let scale = scale.unwrap_or_else(|| 1.0 / (self.head_size as f32).sqrt());
        let batch = self.batch();
        let mut out = vec![0.0f32; batch * self.q_seq * self.num_heads * self.v_head_size];
        for b in 0..batch {
            let valid = self.nonpad[b].max(0) as usize;
            let causal_off = self.nonpad[b] - self.q_seq as i64;
            for i in 0..self.q_seq {
                for h in 0..self.num_heads {
                    let kvh = h / group;
                    let q_base = ((b * self.q_seq + i) * self.num_heads + h) * self.head_size;
                    let mut scores = vec![f32::NEG_INFINITY; valid];
                    for (j, score) in scores.iter_mut().enumerate() {
                        if is_causal && (j as i64) > i as i64 + causal_off {
                            continue;
                        }
                        let k_base =
                            ((b * self.kv_seq + j) * self.kv_num_heads + kvh) * self.head_size;
                        let mut dot = 0.0f32;
                        for p in 0..self.head_size {
                            dot += q[q_base + p] * k[k_base + p];
                        }
                        let mut s = dot * scale;
                        if softcap != 0.0 {
                            s = softcap * (s / softcap).tanh();
                        }
                        *score = s;
                    }
                    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let out_base = ((b * self.q_seq + i) * self.num_heads + h) * self.v_head_size;
                    if !m.is_finite() {
                        continue; // fully masked → zeros
                    }
                    let mut sum = 0.0f32;
                    let mut probs = vec![0.0f32; valid];
                    for (j, p) in probs.iter_mut().enumerate() {
                        if scores[j].is_finite() {
                            *p = (scores[j] - m).exp();
                            sum += *p;
                        }
                    }
                    let inv = 1.0 / sum;
                    for c in 0..self.v_head_size {
                        let mut acc = 0.0f32;
                        for (j, &p) in probs.iter().enumerate() {
                            let v_base = ((b * self.kv_seq + j) * self.kv_num_heads + kvh)
                                * self.v_head_size;
                            acc += p * inv * v[v_base + c];
                        }
                        out[out_base + c] = acc;
                    }
                }
            }
        }
        out
    }
}

fn check(
    batch: &Batch,
    is_causal: bool,
    fp16: bool,
    rank4: bool,
    scale: Option<f32>,
    softcap: f32,
) {
    let ep = require_cuda();
    let got = batch.run(&ep, is_causal, fp16, rank4, scale, softcap);
    let want = batch.cpu_reference(is_causal, fp16, scale, softcap);
    let tol = if fp16 { 3e-2 } else { 2e-4 };
    let diff = max_abs_diff(&got, &want);
    assert!(
        diff <= tol,
        "varlen output diverged from CPU reference: max_abs_diff={diff} (tol={tol}, fp16={fp16}, causal={is_causal}, rank4={rank4})"
    );
}

fn check_dtype(batch: &Batch, is_causal: bool, dtype: DataType, tolerance: f32) {
    let ep = require_cuda();
    let got = batch.run_dtype(&ep, is_causal, dtype, true, None, 0.0);
    let want = batch.cpu_reference_dtype(is_causal, dtype, None, 0.0);
    let diff = max_abs_diff(&got, &want);
    assert!(
        diff <= tolerance,
        "varlen output diverged from CPU reference: max_abs_diff={diff} (tol={tolerance}, dtype={dtype:?}, causal={is_causal})"
    );
}

// ---- ragged batch [3, 7, 2], f32, f16, and bf16, causal and non-causal ----

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn varlen_ragged_non_causal_f32() {
    let batch = Batch::new(3, 3, 8, 8, 4, &[3, 7, 2]);
    check(&batch, false, false, true, None, 0.0);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn varlen_ragged_causal_f32() {
    let batch = Batch::new(3, 3, 8, 8, 4, &[3, 7, 2]);
    check(&batch, true, false, true, None, 0.0);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn varlen_ragged_non_causal_f16() {
    let batch = Batch::new(3, 3, 8, 8, 4, &[3, 7, 2]);
    check(&batch, false, true, true, None, 0.0);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn varlen_ragged_causal_f16() {
    let batch = Batch::new(3, 3, 8, 8, 4, &[3, 7, 2]);
    check(&batch, true, true, true, None, 0.0);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn varlen_ragged_non_causal_bf16() {
    let batch = Batch::new(3, 3, 8, 8, 4, &[3, 7, 2]);
    check_dtype(&batch, false, DataType::BFloat16, 1e-1);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn varlen_ragged_causal_bf16() {
    let batch = Batch::new(3, 3, 8, 8, 4, &[3, 7, 2]);
    check_dtype(&batch, true, DataType::BFloat16, 1e-1);
}

// ---- rank-3 collapsed layout must match rank-4 semantics ----

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn varlen_ragged_rank3_layout_f32() {
    let batch = Batch::new(2, 2, 8, 8, 5, &[6, 3]);
    check(&batch, true, false, false, None, 0.0);
}

// ---- GQA / MQA head sharing ----

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn varlen_gqa_head_sharing_f32() {
    let batch = Batch::new(4, 2, 8, 8, 4, &[5, 2, 7]);
    check(&batch, false, false, true, None, 0.0);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn varlen_mqa_head_sharing_causal_f16() {
    let batch = Batch::new(4, 1, 8, 8, 4, &[5, 2, 7]);
    check(&batch, true, true, true, None, 0.0);
}

// ---- custom scale and softcap ----

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn varlen_scale_and_softcap_f32() {
    let batch = Batch::new(2, 2, 8, 8, 4, &[6, 3]);
    check(&batch, false, false, true, Some(0.2), 30.0);
}

// ---- edge cases: single sequence, and sequences of length 1 ----

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn varlen_single_sequence_causal_f32() {
    let batch = Batch::new(3, 3, 16, 16, 5, &[5]);
    check(&batch, true, false, true, None, 0.0);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn varlen_single_sequence_f16() {
    let batch = Batch::new(3, 3, 16, 16, 5, &[4]);
    check(&batch, false, true, true, None, 0.0);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn varlen_length_one_kv_decode_f32() {
    // Single query token attending a single valid KV token per batch (decode).
    let batch = Batch::new(4, 2, 8, 8, 1, &[1, 1, 1]);
    check(&batch, false, false, true, None, 0.0);
    check(&batch, true, false, true, None, 0.0);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn varlen_length_one_kv_decode_f16() {
    let batch = Batch::new(4, 2, 8, 8, 1, &[1, 1, 1]);
    check(&batch, true, true, true, None, 0.0);
}

// ---- mixed valid lengths incl. a fully-padded (zero valid) sequence ----

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn varlen_zero_valid_sequence_emits_zeros_f32() {
    let batch = Batch::new(2, 2, 8, 8, 3, &[0, 5]);
    check(&batch, false, false, true, None, 0.0);
    check(&batch, true, false, true, None, 0.0);
}
