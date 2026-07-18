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
use onnx_runtime_ir::{DataType, DeviceId, Node};

use crate::error::not_implemented;
use crate::runtime::{CudaRuntime, cuptr};

const OP: &str = "CompressedSparseAttention";

/// Positions whose dtype is fixed regardless of ratio / cache format. These are
/// gated at claim time so an unsupported dtype is rejected rather than failing
/// inside the CPU kernel. Index 6 (`past_compressed_kv`) is gated separately
/// because its dtype depends on `cache_format`.
const FIXED_DTYPES: &[(usize, DataType, &str)] = &[
    (0, DataType::Float32, "query"),
    (8, DataType::Int32, "seqlens_k"),
    (9, DataType::Int64, "total_sequence_length"),
    (10, DataType::Float32, "head_sink"),
];

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
    input_dtypes: &[DataType],
) -> Option<Cow<'static, str>> {
    // Attribute/arity gating: the CPU factory validates the full frozen-v1
    // attribute set and required-input names; any rejection there is a config we
    // cannot correctly execute host-staged either.
    if let Err(error) = CpuCsaFactory.create(node, &[]) {
        return Some(Cow::Owned(format!("{OP}: {error}")));
    }

    // dtype gating for the positions whose dtype is fixed by the frozen contract.
    for &(index, expected, name) in FIXED_DTYPES {
        if let Some(&dtype) = input_dtypes.get(index)
            && dtype != expected
        {
            return Some(Cow::Owned(format!(
                "{OP}: input {index} ('{name}') dtype {dtype:?} unsupported; expected {expected:?}"
            )));
        }
    }

    // `past_compressed_kv` (input 6) dtype is Uint8 for the block-quantized cache
    // formats and Float32 for the plain f32 format.
    let cache_dtype = match node.attr("cache_format").and_then(|a| a.as_str()) {
        Some("f32") | None => DataType::Float32,
        Some(_) => DataType::Uint8,
    };
    if let Some(&dtype) = input_dtypes.get(6)
        && dtype != cache_dtype
    {
        return Some(Cow::Owned(format!(
            "{OP}: input 6 ('past_compressed_kv') dtype {dtype:?} unsupported for cache_format; expected {cache_dtype:?}"
        )));
    }

    None
}
