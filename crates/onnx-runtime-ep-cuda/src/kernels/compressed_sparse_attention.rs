//! `pkg.nxrt::CompressedSparseAttention` v1: correctness-first, **host-staged**
//! CUDA execution of the DeepSeek-V4-Flash / GLM-5.2 compressed sparse-attention
//! (CSA) operator.
//!
//! The fully-implemented CPU kernel in
//! `crates/onnx-runtime-ep-cpu/src/kernels/compressed_sparse_attention.rs` is the
//! authoritative numerical oracle for this op. Re-deriving its ~4.6k lines of
//! frozen-contract math (learned FP8/FP4 compression, the ratio-4 index-key
//! stream, sparse sink-softmax, and the stateful compressed KV cache/carry
//! lifecycle) on the device would be error-prone and is explicitly a later,
//! separately-tracked phase. This kernel therefore guarantees bit-parity by
//! **delegating to the CPU kernel itself**:
//!
//! 1. every device input tensor is copied host-side (D2H),
//! 2. the CPU `CompressedSparseAttention` kernel — built by the CPU factory from
//!    the same node, so it carries the identical attribute configuration — runs
//!    over host-resident views, producing every output (`Y`, the present
//!    compressed KV cache, the present compression carry, and, for ratio-4, the
//!    present index key / index carry / selected indices),
//! 3. each host output is uploaded back to its device buffer (H2D).
//!
//! ## Statefulness
//!
//! CSA is stateful, but the state is threaded through the graph as ordinary
//! `past_* → present_*` input/output tensors (the standard ONNX KV-cache
//! pattern), not held inside the kernel. A `prefill → decode → decode` sequence
//! feeds each step's `present_*` outputs back in as the next step's `past_*`
//! inputs. Because this kernel reuses the CPU kernel verbatim, the entire
//! compressed-cache / carry / index-carry lifecycle is reproduced exactly, and
//! the host-resident staging keeps state correct across steps (device-resident
//! state is the Phase-B optimization).
//!
//! ## `cuda_graph_compatible` = false
//!
//! Like the correctness-first `standard_attention` / `sparse_kv_gather`
//! kernels, execution round-trips through host memory and synchronizes the
//! stream on every D2H/H2D copy, neither of which is legal during CUDA-graph
//! capture.
//!
//! ## Claim-time gating
//!
//! [`unsupported_reason`] rejects, at claim time, any ratio / cache-layout /
//! sink-mode / dtype / arity combination the CPU oracle does not accept (by
//! dry-running the CPU factory, which validates the full frozen-v1 attribute set,
//! plus explicit checks on the dtype-fixed inputs). This upholds the doc §4.8
//! contract: "`supports_op` must reject unsupported ratio/layout/dtype/shape
//! combinations instead of claiming the node and falling back inside the kernel."
//!
// TODO(csa-cuda phase B): replace this host-staged path with a device-resident
// fused CSA kernel (device-resident compressed cache/carry, fused
// selection/score/sink-softmax/value-reduction, CUDA-graph capture, no host
// round trip). See docs/DEEPSEEK_CSA_MTP_RUNTIME.md §4.8.

use std::borrow::Cow;
use std::ffi::c_void;
use std::sync::Arc;

use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ep_cpu::kernels::compressed_sparse_attention::CompressedSparseAttentionFactory as CpuCsaFactory;
use onnx_runtime_ir::{DataType, DeviceId, Dim, Node, Shape, as_static_shape};

use crate::error::not_implemented;
use crate::runtime::{CudaRuntime, cuptr};

const OP: &str = "CompressedSparseAttention";

/// Factory for the host-staged CUDA CSA kernel. It builds the CPU CSA kernel
/// from the same node (reusing the CPU oracle's attribute validation and compute
/// core) and wraps it so execution stages tensors through the host.
pub struct CompressedSparseAttentionFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for CompressedSparseAttentionFactory {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        // Delegate construction to the CPU factory: it validates the full frozen
        // v1 attribute set (ratio, cache_format, sink_mode, index dims, arity,
        // required input names) and produces the stateful oracle kernel whose
        // compute we reuse verbatim.
        let inner = CpuCsaFactory.create(node, input_shapes)?;
        Ok(Box::new(CompressedSparseAttentionKernel {
            runtime: self.runtime.clone(),
            inner,
        }))
    }
}

/// Host-staged CUDA CSA kernel: wraps the CPU oracle kernel and moves data
/// device↔host around each `execute`.
struct CompressedSparseAttentionKernel {
    runtime: Arc<CudaRuntime>,
    inner: Box<dyn Kernel>,
}

impl std::fmt::Debug for CompressedSparseAttentionKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompressedSparseAttentionKernel").finish()
    }
}

impl Kernel for CompressedSparseAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // Stage every present input host-side. Contiguity is required because the
        // host copy is a dense byte blit; the CPU oracle then reads it densely.
        let mut staged: Vec<Vec<u8>> = Vec::with_capacity(inputs.len());
        for (index, input) in inputs.iter().enumerate() {
            if input.is_absent() {
                staged.push(Vec::new());
                continue;
            }
            if !input.is_contiguous() {
                return Err(not_implemented(format!(
                    "{OP}: non-contiguous input {index} on CUDA (host-staged path requires contiguous inputs)"
                )));
            }
            let bytes = input.byte_size();
            let mut host = vec![0u8; bytes];
            if bytes > 0 {
                // SAFETY: `input` is a live contiguous device tensor and `host`
                // is exactly its dense storage size.
                unsafe {
                    self.runtime
                        .dtoh(&mut host, cuptr(input.data_ptr::<u8>() as *const c_void))?;
                }
            }
            staged.push(host);
        }

        // Build host-resident input views over the staged buffers, reusing each
        // input's (contiguous) shape/strides. `DevicePtr` is a raw pointer, so
        // these views borrow nothing from `staged` at the type level — `staged`
        // is kept alive until after `execute`.
        let host_inputs: Vec<TensorView> = inputs
            .iter()
            .zip(&staged)
            .map(|(input, buf)| {
                if input.is_absent() {
                    TensorView::absent(input.dtype)
                } else {
                    TensorView::new(
                        onnx_runtime_ep_api::DevicePtr(buf.as_ptr() as *const c_void),
                        input.dtype,
                        input.shape,
                        input.strides,
                        DeviceId::cpu(),
                    )
                }
            })
            .collect();

        // Snapshot output metadata and allocate matching host buffers. The
        // session has already shape-inferred and allocated the device outputs, so
        // their shapes are authoritative for the oracle's own shape checks.
        for (index, output) in outputs.iter().enumerate() {
            if !output.is_contiguous() {
                return Err(not_implemented(format!(
                    "{OP}: non-contiguous output {index} on CUDA (host-staged path requires contiguous outputs)"
                )));
            }
        }
        let out_dtypes: Vec<DataType> = outputs.iter().map(|o| o.dtype).collect();
        let out_shapes: Vec<Vec<usize>> = outputs.iter().map(|o| o.shape.to_vec()).collect();
        let out_strides: Vec<Vec<i64>> = outputs.iter().map(|o| o.strides.to_vec()).collect();
        let mut out_bufs: Vec<Vec<u8>> = outputs.iter().map(|o| vec![0u8; o.byte_size()]).collect();

        let mut host_outputs: Vec<TensorMut> = out_bufs
            .iter_mut()
            .enumerate()
            .map(|(index, buf)| {
                TensorMut::new(
                    onnx_runtime_ep_api::DevicePtrMut(buf.as_mut_ptr() as *mut c_void),
                    out_dtypes[index],
                    &out_shapes[index],
                    &out_strides[index],
                    DeviceId::cpu(),
                )
            })
            .collect();

        // Run the CPU oracle over the host-staged tensors: guarantees bit-parity
        // and reproduces the full stateful cache/carry/index lifecycle.
        self.inner.execute(&host_inputs, &mut host_outputs)?;

        // Release the borrow of `out_bufs` before uploading the results.
        drop(host_outputs);
        drop(host_inputs);

        for (index, output) in outputs.iter_mut().enumerate() {
            let bytes = &out_bufs[index];
            if !bytes.is_empty() {
                // SAFETY: `output` is a live device allocation whose dense size
                // equals `bytes.len()` (built from `output.byte_size()`).
                unsafe {
                    self.runtime
                        .htod(bytes, cuptr(output.data_ptr_mut::<u8>() as *const c_void))?;
                }
            }
        }
        self.runtime.synchronize()
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        // The host-staging blit is dense; strided inputs are rejected in execute.
        false
    }

    fn cuda_graph_compatible(&self) -> bool {
        // Host round-trip (D2H inputs, H2D outputs) plus per-copy stream syncs
        // are illegal during CUDA-graph capture. Device-resident capture is a
        // Phase-B goal (docs/DEEPSEEK_CSA_MTP_RUNTIME.md §4.8).
        false
    }
}

/// Claim-time denial for `pkg.nxrt::CompressedSparseAttention`. Rejects any
/// ratio / cache-layout / sink-mode / arity combination the CPU oracle does not
/// accept (via a dry-run of the CPU factory), plus explicit dtype gating on the
/// dtype-fixed inputs, so unsupported combinations never reach `execute`
/// (docs/DEEPSEEK_CSA_MTP_RUNTIME.md §4.8).
pub(crate) fn unsupported_reason(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Option<Cow<'static, str>> {
    // Attribute/arity gating: the CPU factory validates the full frozen-v1
    // attribute set and required-input names; any rejection there is a config we
    // cannot correctly execute host-staged either.
    let concrete_shapes = shapes
        .iter()
        .map(|shape| as_static_shape(shape))
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default();
    if let Err(error) = CpuCsaFactory.create(node, &concrete_shapes) {
        return Some(Cow::Owned(format!("{OP}: {error}")));
    }

    if shapes.len() != node.inputs.len() || input_dtypes.len() != node.inputs.len() {
        return Some(Cow::Owned(format!(
            "{OP}: claim metadata must cover all {} positional inputs (got {} shapes and {} dtypes)",
            node.inputs.len(),
            shapes.len(),
            input_dtypes.len()
        )));
    }

    let ratio = usize::try_from(
        node.attr("compression_ratio")
            .and_then(|attribute| attribute.as_int())
            .expect("CPU factory accepted compression_ratio"),
    )
    .expect("CPU factory accepted positive compression_ratio");
    let cache_format = node
        .attr("cache_format")
        .and_then(|attribute| attribute.as_str())
        .unwrap_or("f32");

    let result = match ratio {
        4 => validate_ratio4_claim(node, shapes, input_dtypes, cache_format),
        128 => validate_ratio128_claim(node, shapes, input_dtypes, cache_format),
        _ => unreachable!("CPU factory rejected unsupported compression ratio"),
    }
    .and_then(|()| validate_attention_bias_claim(node, shapes, input_dtypes));

    result
        .err()
        .map(|reason| Cow::Owned(format!("{OP}: {reason}")))
}

fn validate_ratio4_claim(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
    cache_format: &str,
) -> std::result::Result<(), String> {
    if node.inputs.len() < 19 || node.inputs[11..19].iter().any(Option::is_none) {
        return Err("ratio-4 requires all eight index inputs (11..=18)".into());
    }
    if !(5..=6).contains(&node.outputs.len()) {
        return Err(format!(
            "ratio-4 requires 5 or 6 outputs, got {}",
            node.outputs.len()
        ));
    }
    if node
        .attr("index_head_dim")
        .and_then(|attribute| attribute.as_int())
        != Some(128)
    {
        return Err("ratio-4 requires index_head_dim=128".into());
    }
    if cache_format != "fp8_e4m3_block64" {
        return Err(format!(
            "ratio-4 requires cache_format='fp8_e4m3_block64', got '{cache_format}'"
        ));
    }
    require_fixed_contract(node, 4)?;

    for &(index, expected, name) in &[
        (0, DataType::Float32, "query"),
        (1, DataType::Float32, "current_kv"),
        (2, DataType::Float32, "compressor_kv"),
        (3, DataType::Float32, "compressor_gate"),
        (4, DataType::Float32, "compressor_ape"),
        (5, DataType::Float32, "compressor_norm"),
        (6, DataType::Uint8, "past_compressed_kv"),
        (7, DataType::Float32, "past_compression_carry"),
        (8, DataType::Int32, "seqlens_k"),
        (9, DataType::Int64, "total_sequence_length"),
        (10, DataType::Float32, "head_sink"),
        (11, DataType::Float32, "index_query"),
        (12, DataType::Float32, "index_weight"),
        (13, DataType::Float32, "index_compressor_kv"),
        (14, DataType::Float32, "index_compressor_gate"),
        (15, DataType::Float32, "index_compressor_ape"),
        (16, DataType::Float32, "index_compressor_norm"),
        (17, DataType::Uint8, "past_index_key"),
        (18, DataType::Float32, "past_index_carry"),
    ] {
        require_dtype(input_dtypes, index, expected, name)?;
    }
    let heads = required_attr(node, "num_heads")?;
    let index_heads = required_attr(node, "index_num_heads")?;
    for (index, name, contract) in [
        (0, "query", vec![Any, NonZero, Fixed(heads), Fixed(512)]),
        (1, "current_kv", vec![Same(0, 0), Any, Fixed(512)]),
        (
            2,
            "compressor_kv",
            vec![Same(0, 0), Same(0, 1), Fixed(1024)],
        ),
        (
            3,
            "compressor_gate",
            vec![Same(0, 0), Same(0, 1), Fixed(1024)],
        ),
        (4, "compressor_ape", vec![Fixed(4), Fixed(1024)]),
        (5, "compressor_norm", vec![Fixed(512)]),
        (6, "past_compressed_kv", vec![Same(0, 0), Any, Fixed(583)]),
        (
            7,
            "past_compression_carry",
            vec![Same(0, 0), Fixed(8), Fixed(2), Fixed(1024)],
        ),
        (8, "seqlens_k", vec![Same(0, 0)]),
        (9, "total_sequence_length", vec![]),
        (10, "head_sink", vec![Fixed(heads)]),
        (
            11,
            "index_query",
            vec![Same(0, 0), Same(0, 1), Fixed(index_heads), Fixed(128)],
        ),
        (
            12,
            "index_weight",
            vec![Same(0, 0), Same(0, 1), Fixed(index_heads)],
        ),
        (
            13,
            "index_compressor_kv",
            vec![Same(0, 0), Same(0, 1), Fixed(256)],
        ),
        (
            14,
            "index_compressor_gate",
            vec![Same(0, 0), Same(0, 1), Fixed(256)],
        ),
        (15, "index_compressor_ape", vec![Fixed(4), Fixed(256)]),
        (16, "index_compressor_norm", vec![Fixed(128)]),
        (17, "past_index_key", vec![Same(0, 0), Any, Fixed(68)]),
        (
            18,
            "past_index_carry",
            vec![Same(0, 0), Fixed(8), Fixed(2), Fixed(256)],
        ),
    ] {
        require_shape(shapes, index, name, &contract)?;
    }
    Ok(())
}

fn validate_ratio128_claim(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
    cache_format: &str,
) -> std::result::Result<(), String> {
    for index in 11..19.min(node.inputs.len()) {
        if node.inputs[index].is_some() {
            return Err(format!(
                "ratio-4-only input {index} is unsupported for ratio-128"
            ));
        }
    }
    if node.outputs.len() != 3 {
        return Err(format!(
            "ratio-128 requires exactly 3 outputs, got {}",
            node.outputs.len()
        ));
    }
    if cache_format == "fp4_e2m1_block32" {
        return Err(
            "ratio-128 attention-compressor state uses f32 or hybrid FP8/BF16 records, not FP4"
                .into(),
        );
    }
    require_fixed_contract(node, 128)?;

    let cache_dtype = if cache_format == "f32" {
        DataType::Float32
    } else {
        DataType::Uint8
    };
    for &(index, expected, name) in &[
        (0, DataType::Float32, "query"),
        (1, DataType::Float32, "current_kv"),
        (2, DataType::Float32, "compressor_kv"),
        (3, DataType::Float32, "compressor_gate"),
        (4, DataType::Float32, "compressor_ape"),
        (5, DataType::Float32, "compressor_norm"),
        (6, cache_dtype, "past_compressed_kv"),
        (7, DataType::Float32, "past_compression_carry"),
        (8, DataType::Int32, "seqlens_k"),
        (9, DataType::Int64, "total_sequence_length"),
        (10, DataType::Float32, "head_sink"),
    ] {
        require_dtype(input_dtypes, index, expected, name)?;
    }

    let heads = required_attr(node, "num_heads")?;
    let stored_width = if cache_format == "f32" { 512 } else { 583 };
    for (index, name, contract) in [
        (0, "query", vec![Any, NonZero, Fixed(heads), Fixed(512)]),
        (1, "current_kv", vec![Same(0, 0), Any, Fixed(512)]),
        (2, "compressor_kv", vec![Same(0, 0), Same(0, 1), Fixed(512)]),
        (
            3,
            "compressor_gate",
            vec![Same(0, 0), Same(0, 1), Fixed(512)],
        ),
        (4, "compressor_ape", vec![Fixed(128), Fixed(512)]),
        (5, "compressor_norm", vec![Fixed(512)]),
        (
            6,
            "past_compressed_kv",
            vec![Same(0, 0), Any, Fixed(stored_width)],
        ),
        (
            7,
            "past_compression_carry",
            vec![Same(0, 0), Fixed(128), Fixed(2), Fixed(512)],
        ),
        (8, "seqlens_k", vec![Same(0, 0)]),
        (9, "total_sequence_length", vec![]),
        (10, "head_sink", vec![Fixed(heads)]),
    ] {
        require_shape(shapes, index, name, &contract)?;
    }
    Ok(())
}

fn validate_attention_bias_claim(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> std::result::Result<(), String> {
    if !node.inputs.get(19).is_some_and(Option::is_some) {
        return Ok(());
    }

    require_dtype(input_dtypes, 19, DataType::Float32, "attention_bias")?;
    let bias_shape = &shapes[19];
    if bias_shape.len() > 4 {
        return Err(format!(
            "input 19 ('attention_bias') rank {} unsupported; expected rank <= 4",
            bias_shape.len()
        ));
    }

    if let Some(static_shape) = as_static_shape(bias_shape) {
        let elements = static_shape
            .iter()
            .try_fold(1usize, |count, &dimension| count.checked_mul(dimension));
        if elements
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(format!(
                "input 19 ('attention_bias') byte count overflow or exceeds isize::MAX for shape {static_shape:?}"
            ));
        }
    }

    let heads = required_attr(node, "num_heads")?;
    let target = [
        shapes[0][0].as_static(),
        Some(heads),
        shapes[0][1].as_static(),
        None,
    ];
    let offset = 4 - bias_shape.len();
    for (axis, dimension) in bias_shape.iter().enumerate() {
        let Some(got) = dimension.as_static() else {
            continue;
        };
        let target_axis = offset + axis;
        if got != 1 && target[target_axis].is_some_and(|expected| got != expected) {
            return Err(format!(
                "input 19 ('attention_bias') shape {bias_shape:?} is not broadcastable to attention scores [{:?}, {heads}, {:?}, ?]",
                shapes[0][0], shapes[0][1]
            ));
        }
    }
    Ok(())
}

fn require_fixed_contract(node: &Node, ratio: usize) -> std::result::Result<(), String> {
    if required_attr(node, "head_dim")? != 512 {
        return Err(format!("ratio-{ratio} requires head_dim=512"));
    }
    let rope_dim = match node.attr("qk_rope_head_dim") {
        Some(attribute) => attribute
            .as_int()
            .ok_or_else(|| "qk_rope_head_dim must be an integer".to_string())?,
        None => 0,
    };
    if rope_dim != 64 {
        return Err(format!("ratio-{ratio} requires qk_rope_head_dim=64"));
    }
    Ok(())
}

fn required_attr(node: &Node, name: &str) -> std::result::Result<usize, String> {
    node.attr(name)
        .and_then(|attribute| attribute.as_int())
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("missing or invalid integer attribute '{name}'"))
}

fn require_dtype(
    input_dtypes: &[DataType],
    index: usize,
    expected: DataType,
    name: &str,
) -> std::result::Result<(), String> {
    let got = input_dtypes[index];
    if got != expected {
        return Err(format!(
            "input {index} ('{name}') dtype {got:?} unsupported; expected {expected:?}"
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum ShapeAxis {
    Any,
    NonZero,
    Fixed(usize),
    Same(usize, usize),
}
use ShapeAxis::{Any, Fixed, NonZero, Same};

fn require_shape(
    shapes: &[Shape],
    index: usize,
    name: &str,
    contract: &[ShapeAxis],
) -> std::result::Result<(), String> {
    let shape = &shapes[index];
    if shape.len() != contract.len() {
        return Err(format!(
            "input {index} ('{name}') rank {} unsupported; expected {}",
            shape.len(),
            contract.len()
        ));
    }
    for (axis, requirement) in contract.iter().enumerate() {
        let mismatch = match requirement {
            Any => None,
            NonZero if shape[axis] == Dim::Static(0) => Some("must be nonzero".into()),
            NonZero => None,
            Fixed(expected) => shape[axis]
                .as_static()
                .filter(|got| got != expected)
                .map(|got| format!("is {got}; expected {expected}")),
            Same(other_input, other_axis) => {
                match (
                    shape[axis].as_static(),
                    shapes[*other_input][*other_axis].as_static(),
                ) {
                    (Some(got), Some(expected)) if got != expected => {
                        Some(format!("is {got}; expected {expected}"))
                    }
                    _ => None,
                }
            }
        };
        if let Some(mismatch) = mismatch {
            return Err(format!("input {index} ('{name}') axis {axis} {mismatch}"));
        }
    }
    Ok(())
}
