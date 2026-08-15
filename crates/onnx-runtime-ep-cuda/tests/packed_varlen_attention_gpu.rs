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
//! GPU correctness tests for `pkg.nxrt::PackedVarlenAttention` v1.
//!
//! The packed (unpadded) varlen kernel must reproduce, sequence for sequence,
//! what the standard `ai.onnx::Attention` kernel computes for the SAME logical
//! batch — just without spending any compute on padding. Every case builds a
//! ragged batch, runs it once through the packed kernel, and compares against a
//! reference assembled from the padded `Attention` kernel:
//!
//! * per-sequence reference — run each sequence on its own through a batch-1
//!   `Attention` node (3D `[1, L, hidden]`, so the packed slice feeds in with no
//!   transpose) and concatenate. This is the ground truth for varlen: a packed
//!   batch is exactly independent per-sequence attention.
//! * all-equal-length — compare against a single dense batched `Attention`
//!   (`[B, L, hidden]`); with no ragged padding the packed output must match the
//!   padded output.
//! * `nonpad_kv_seqlen` — compare against a padded `Attention` fed the opset-24
//!   `nonpad_kv_seqlen` input, demonstrating the packed `cu_seqlens` is the
//!   exclusive prefix sum of the padded per-batch valid lengths.
//!
//! CPU-only CI reports these tests as ignored unless `gpu-tests` is enabled. Run with:
//! `CUDA_VISIBLE_DEVICES=5 taskset -c 1 cargo test -p onnx-runtime-ep-cuda \
//!   --test packed_varlen_attention_gpu -- --nocapture`

use half::f16;
use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, Result, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

const PACKED_DOMAIN: &str = "pkg.nxrt";

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

fn i32_tensor(shape: &[usize], values: &[i32]) -> Tensor {
    Tensor {
        dtype: DataType::Int32,
        shape: shape.to_vec(),
        bytes: values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
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

/// Build and run a single-node model through the CUDA EP, returning the raw
/// output bytes. `None` input slots model omitted optional ONNX inputs.
fn run_node(
    ep: &CudaExecutionProvider,
    op: &str,
    domain: &str,
    opset: u64,
    inputs: &[Option<Tensor>],
    output: (DataType, Vec<usize>),
    attrs: &[(&str, Attribute)],
) -> Result<Vec<u8>> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(domain.to_string(), opset);
    let input_values = inputs
        .iter()
        .enumerate()
        .map(|(i, input)| {
            input.as_ref().map(|input| {
                let value = graph.create_named_value(
                    format!("input_{i}"),
                    input.dtype,
                    static_shape(input.shape.iter().copied()),
                );
                graph.add_input(value);
                value
            })
        })
        .collect::<Vec<_>>();
    let (out_dtype, out_shape) = output;
    let output_value =
        graph.create_named_value("output", out_dtype, static_shape(out_shape.iter().copied()));
    let mut node = Node::new(NodeId(0), op, input_values, vec![output_value]);
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
        .map(|input| -> Result<Option<DeviceBuffer>> {
            let Some(input) = input else {
                return Ok(None);
            };
            let buffer = ep.allocate(input.bytes.len().max(1), 256)?;
            if !input.bytes.is_empty() {
                // SAFETY: the allocation exactly covers the source tensor.
                unsafe { ep.runtime().htod(&input.bytes, cuptr(buffer.as_ptr()))? };
            }
            Ok(Some(buffer))
        })
        .collect::<Result<Vec<_>>>()?;
    let input_strides = inputs
        .iter()
        .map(|input| {
            input
                .as_ref()
                .map(|input| compute_contiguous_strides(&input.shape))
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    let input_views = inputs
        .iter()
        .zip(&input_buffers)
        .zip(&input_strides)
        .map(|((input, buffer), strides)| match (input, buffer) {
            (Some(input), Some(buffer)) => TensorView::new(
                DevicePtr(buffer.as_ptr()),
                input.dtype,
                &input.shape,
                strides,
                ep.device_id(),
            ),
            _ => TensorView::absent(DataType::Float32),
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
    for buffer in input_buffers.into_iter().flatten() {
        ep.deallocate(buffer)?;
    }
    ep.deallocate(output_buffer)?;
    result.map(|()| bytes)
}

/// A ragged batch: `head_size`/`v_head_size` per head, `num_heads` query heads,
/// `kv_num_heads` key/value heads, and one length per sequence.
struct Batch {
    num_heads: usize,
    kv_num_heads: usize,
    head_size: usize,
    v_head_size: usize,
    lengths: Vec<usize>,
    // Packed [total, kv_or_q_heads, dim] host data.
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
        lengths: &[usize],
    ) -> Self {
        let total: usize = lengths.iter().sum();
        Self {
            num_heads,
            kv_num_heads,
            head_size,
            v_head_size,
            lengths: lengths.to_vec(),
            q: fill(total * num_heads * head_size, 0x51),
            k: fill(total * kv_num_heads * head_size, 0x52),
            v: fill(total * kv_num_heads * v_head_size, 0x53),
        }
    }

    fn total(&self) -> usize {
        self.lengths.iter().sum()
    }

    fn cu_seqlens(&self) -> Vec<i32> {
        let mut cu = Vec::with_capacity(self.lengths.len() + 1);
        let mut acc = 0i32;
        cu.push(0);
        for &l in &self.lengths {
            acc += l as i32;
            cu.push(acc);
        }
        cu
    }

    /// Run the packed varlen kernel over the whole batch.
    fn run_packed(&self, ep: &CudaExecutionProvider, is_causal: bool, fp16: bool) -> Vec<f32> {
        let total = self.total();
        let make = |shape: &[usize], data: &[f32]| {
            if fp16 {
                f16_tensor(shape, data)
            } else {
                f32_tensor(shape, data)
            }
        };
        let cu = self.cu_seqlens();
        let inputs = vec![
            Some(make(&[total, self.num_heads, self.head_size], &self.q)),
            Some(make(&[total, self.kv_num_heads, self.head_size], &self.k)),
            Some(make(&[total, self.kv_num_heads, self.v_head_size], &self.v)),
            Some(i32_tensor(&[cu.len()], &cu)),
            Some(i32_tensor(&[cu.len()], &cu)),
        ];
        let out_dtype = if fp16 {
            DataType::Float16
        } else {
            DataType::Float32
        };
        let attrs = vec![
            ("num_heads", Attribute::Int(self.num_heads as i64)),
            ("kv_num_heads", Attribute::Int(self.kv_num_heads as i64)),
            ("is_causal", Attribute::Int(is_causal as i64)),
        ];
        let bytes = run_node(
            ep,
            "PackedVarlenAttention",
            PACKED_DOMAIN,
            1,
            &inputs,
            (out_dtype, vec![total, self.num_heads, self.v_head_size]),
            &attrs,
        )
        .expect("packed varlen attention must run");
        decode(out_dtype, &bytes)
    }

    /// Ground-truth reference: run each sequence independently through the
    /// standard `Attention` kernel (batch-1, 3D `[1, L, hidden]`) and
    /// concatenate. The packed per-sequence slice feeds in with no transpose.
    fn run_per_sequence_reference(&self, ep: &CudaExecutionProvider, is_causal: bool) -> Vec<f32> {
        let q_hidden = self.num_heads * self.head_size;
        let k_hidden = self.kv_num_heads * self.head_size;
        let v_hidden = self.kv_num_heads * self.v_head_size;
        let out_hidden = self.num_heads * self.v_head_size;
        let mut out = Vec::new();
        let mut q_off = 0usize;
        let mut k_off = 0usize;
        for &l in &self.lengths {
            let q_slice = &self.q[q_off * q_hidden..(q_off + l) * q_hidden];
            let k_slice = &self.k[k_off * k_hidden..(k_off + l) * k_hidden];
            let v_slice = &self.v[k_off * v_hidden..(k_off + l) * v_hidden];
            let inputs = vec![
                Some(f32_tensor(&[1, l, q_hidden], q_slice)),
                Some(f32_tensor(&[1, l, k_hidden], k_slice)),
                Some(f32_tensor(&[1, l, v_hidden], v_slice)),
            ];
            let attrs = vec![
                ("q_num_heads", Attribute::Int(self.num_heads as i64)),
                ("kv_num_heads", Attribute::Int(self.kv_num_heads as i64)),
                ("is_causal", Attribute::Int(is_causal as i64)),
            ];
            let bytes = run_node(
                ep,
                "Attention",
                "",
                24,
                &inputs,
                (DataType::Float32, vec![1, l, out_hidden]),
                &attrs,
            )
            .expect("reference Attention must run");
            out.extend(decode(DataType::Float32, &bytes));
            q_off += l;
            k_off += l;
        }
        out
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn packed_matches_per_sequence_reference_causal() {
    let ep = require_cuda();
    // Mixed lengths including a length-1 (degenerate single-token) sequence.
    let batch = Batch::new(3, 3, 8, 8, &[3, 1, 5, 2]);
    let packed = batch.run_packed(&ep, true, false);
    let reference = batch.run_per_sequence_reference(&ep, true);
    let diff = max_abs_diff(&packed, &reference);
    eprintln!("causal MHA max_abs_diff = {diff:e}");
    assert!(
        diff < 1e-4,
        "packed causal attention diverged from per-sequence reference: {diff}"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn packed_matches_per_sequence_reference_non_causal() {
    let ep = require_cuda();
    let batch = Batch::new(2, 2, 6, 6, &[4, 2, 3]);
    let packed = batch.run_packed(&ep, false, false);
    let reference = batch.run_per_sequence_reference(&ep, false);
    let diff = max_abs_diff(&packed, &reference);
    eprintln!("non-causal MHA max_abs_diff = {diff:e}");
    assert!(
        diff < 1e-4,
        "packed non-causal attention diverged from per-sequence reference: {diff}"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn packed_single_sequence_degenerate() {
    let ep = require_cuda();
    // A single sequence must behave exactly like plain single-batch attention.
    let batch = Batch::new(4, 4, 8, 8, &[7]);
    let packed = batch.run_packed(&ep, true, false);
    let reference = batch.run_per_sequence_reference(&ep, true);
    let diff = max_abs_diff(&packed, &reference);
    eprintln!("single-sequence max_abs_diff = {diff:e}");
    assert!(
        diff < 1e-4,
        "single-sequence packed attention diverged: {diff}"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn packed_gqa_non_causal() {
    let ep = require_cuda();
    // Grouped-query attention: 4 query heads share 2 KV heads (group = 2).
    let batch = Batch::new(4, 2, 8, 8, &[5, 3, 4]);
    let packed = batch.run_packed(&ep, false, false);
    let reference = batch.run_per_sequence_reference(&ep, false);
    let diff = max_abs_diff(&packed, &reference);
    eprintln!("GQA non-causal max_abs_diff = {diff:e}");
    assert!(diff < 1e-4, "packed GQA attention diverged: {diff}");
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn packed_all_equal_lengths_match_dense_batched_padded() {
    let ep = require_cuda();
    // All sequences the same length: the packed batch equals a dense
    // (padding-free) rectangular batch, so it must match the padded kernel run
    // as one `[B, L, hidden]` batched Attention.
    let (num_heads, head_size, v_head_size, len, batch_size) = (3usize, 8, 8, 4, 3);
    let batch = Batch::new(
        num_heads,
        num_heads,
        head_size,
        v_head_size,
        &vec![len; batch_size],
    );
    let packed = batch.run_packed(&ep, true, false);

    let hidden = num_heads * head_size;
    let v_hidden = num_heads * v_head_size;
    let out_hidden = num_heads * v_head_size;
    // Reshape the packed [B*L, hidden] data to the dense batched [B, L, hidden].
    let inputs = vec![
        Some(f32_tensor(&[batch_size, len, hidden], &batch.q)),
        Some(f32_tensor(&[batch_size, len, hidden], &batch.k)),
        Some(f32_tensor(&[batch_size, len, v_hidden], &batch.v)),
    ];
    let attrs = vec![
        ("q_num_heads", Attribute::Int(num_heads as i64)),
        ("kv_num_heads", Attribute::Int(num_heads as i64)),
        ("is_causal", Attribute::Int(1)),
    ];
    let bytes = run_node(
        &ep,
        "Attention",
        "",
        24,
        &inputs,
        (DataType::Float32, vec![batch_size, len, out_hidden]),
        &attrs,
    )
    .expect("dense batched Attention must run");
    let padded = decode(DataType::Float32, &bytes);
    let diff = max_abs_diff(&packed, &padded);
    eprintln!("all-equal vs dense-padded max_abs_diff = {diff:e}");
    assert!(
        diff < 1e-5,
        "all-equal-length packed batch must match the dense padded batch: {diff}"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn packed_matches_padded_nonpad_kv_seqlen_non_causal() {
    let ep = require_cuda();
    // Demonstrate consuming the opset-24 `nonpad_kv_seqlen`: build a PADDED
    // batch whose per-sequence valid lengths are the packed cu_seqlens deltas,
    // run the padded `Attention` with `nonpad_kv_seqlen`, and compare its VALID
    // query rows against the packed kernel. Non-causal so the padded op's
    // frontier masking is the only ragged effect.
    let (num_heads, head_size, v_head_size) = (2usize, 6, 6);
    let lengths = [4usize, 2, 3];
    let batch = Batch::new(num_heads, num_heads, head_size, v_head_size, &lengths);
    let packed = batch.run_packed(&ep, false, false);

    let l_max = *lengths.iter().max().unwrap();
    let batch_size = lengths.len();
    let hidden = num_heads * head_size;
    let v_hidden = num_heads * v_head_size;
    let out_hidden = num_heads * v_head_size;

    // Scatter the packed [total, hidden] rows into a padded [B, L_max, hidden]
    // layout (padding rows left zero).
    let scatter = |packed_rows: &[f32], row_hidden: usize| -> Vec<f32> {
        let mut padded = vec![0.0f32; batch_size * l_max * row_hidden];
        let mut src = 0usize;
        for (b, &l) in lengths.iter().enumerate() {
            for t in 0..l {
                let dst = (b * l_max + t) * row_hidden;
                padded[dst..dst + row_hidden].copy_from_slice(
                    &packed_rows[(src + t) * row_hidden..(src + t + 1) * row_hidden],
                );
            }
            src += l;
        }
        padded
    };
    let q_padded = scatter(&batch.q, hidden);
    let k_padded = scatter(&batch.k, hidden);
    let v_padded = scatter(&batch.v, v_hidden);
    let nonpad: Vec<i64> = lengths.iter().map(|&l| l as i64).collect();

    let inputs = vec![
        Some(f32_tensor(&[batch_size, l_max, hidden], &q_padded)),
        Some(f32_tensor(&[batch_size, l_max, hidden], &k_padded)),
        Some(f32_tensor(&[batch_size, l_max, v_hidden], &v_padded)),
        None, // attn_mask
        None, // past_key
        None, // past_value
        Some(i64_tensor(&[batch_size], &nonpad)),
    ];
    let attrs = vec![
        ("q_num_heads", Attribute::Int(num_heads as i64)),
        ("kv_num_heads", Attribute::Int(num_heads as i64)),
        ("is_causal", Attribute::Int(0)),
    ];
    let bytes = run_node(
        &ep,
        "Attention",
        "",
        24,
        &inputs,
        (DataType::Float32, vec![batch_size, l_max, out_hidden]),
        &attrs,
    )
    .expect("padded Attention with nonpad_kv_seqlen must run");
    let padded_out = decode(DataType::Float32, &bytes);

    // Gather the valid query rows back into packed order and compare.
    let mut gathered = Vec::with_capacity(packed.len());
    for (b, &l) in lengths.iter().enumerate() {
        for t in 0..l {
            let src = (b * l_max + t) * out_hidden;
            gathered.extend_from_slice(&padded_out[src..src + out_hidden]);
        }
    }
    let diff = max_abs_diff(&packed, &gathered);
    eprintln!("packed vs padded+nonpad_kv_seqlen max_abs_diff = {diff:e}");
    assert!(
        diff < 1e-4,
        "packed kernel must match padded Attention consuming nonpad_kv_seqlen: {diff}"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn packed_fp16_matches_per_sequence_reference_causal() {
    let ep = require_cuda();
    // fp16 storage with fp32 accumulation: compare against the f32 per-sequence
    // reference within an fp16 attention tolerance.
    let batch = Batch::new(2, 2, 8, 8, &[4, 2, 3]);
    let packed = batch.run_packed(&ep, true, true);
    let reference = batch.run_per_sequence_reference(&ep, true);
    let diff = max_abs_diff(&packed, &reference);
    eprintln!("fp16 causal max_abs_diff = {diff:e}");
    assert!(
        diff < 2e-2,
        "packed fp16 attention diverged beyond fp16 tolerance: {diff}"
    );
}
