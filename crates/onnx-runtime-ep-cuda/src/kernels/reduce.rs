//! GPU **reductions** over arbitrary axes with `keepdims`
//! (`docs/CUDA_COVERAGE.md`, "Normalization & softmax" / reduce rows).
//!
//! `ReduceSum` and `ReduceMean` use `cudnnReduceTensor` for f32/f16/bf16.
//! Their previous f32 NVRTC block reduction remains the runtime fallback when
//! cuDNN is absent. `ReduceMax`/`ReduceMin` continue to use NVRTC.
//!
//! `cub::DeviceReduce` / `DeviceSegmentedReduce` are the vendor primitives for
//! reductions, and a segmented block reduction is exactly the shape they use.
//! We keep a self-contained NVRTC block-reduction kernel here (rather than
//! linking cub) so the crate stays toolkit-free (no `nvcc`), while matching the
//! cub segmented-reduce structure: **one block per output element**, cooperative
//! shared-memory tree reduction over that element's reduction group. It is
//! memory-bandwidth-bound, the same class as PyTorch's reduce kernels.
//!
//! ## Arbitrary axes via an exact base/delta split
//!
//! A row-major input offset is separable across axes:
//! `offset = Σ_axes coord·stride`. Splitting axes into **kept** and **reduced**,
//! `offset(o, r) = base(o) + delta(r)` where `base` depends only on the kept
//! coordinates (one per output element) and `delta` only on the reduced
//! coordinates. The host precomputes `base[O]` and `delta[R]` (§ [`ReductionPlan`])
//! and uploads them; the kernel walks `delta` for its output element `o`. This is
//! exact for **any** axis set and rank, mirroring the CPU EP's reduce-walk
//! (`crates/onnx-runtime-ep-cpu/src/kernels/reduce_ops.rs`).
//!
//! ## ONNX semantics
//!
//! Axes come from the `axes` **attribute** (opset < 13/18) or the optional second
//! **input** (opset ≥ 13 for `ReduceSum`, ≥ 18 for the rest); the input wins when
//! present. `keepdims` (default 1) retains reduced dims as size-1.
//! `noop_with_empty_axes` (default 0) makes an explicitly-empty axis set an
//! identity (per-element groups) instead of reduce-all. Negative axes wrap.
//! `Max`/`Min` propagate NaN (numpy semantics), matching the CPU EP.
//!
//! ## Limits (actionable errors — RULES.md #1)
//!
//! * dtype other than f32 (input/output) → deferred, naming the dtype.
//! * an axes-**input** dtype other than int32/int64 → rejected, naming it.
//! * an axis out of `[-rank, rank)` → rejected, naming the axis.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUdeviceptr;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::cudnn::{CudnnBufferPair, CudnnReduceOp, TensorDescriptorSpec};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

/// NVRTC source: one block per output element, reducing over its group of
/// `reduce_count` elements addressed by `base_off[o] + delta_off[r]`.
/// `op`: 0 = sum, 1 = max, 2 = min. `is_mean` divides a sum by the group size.
/// `Max`/`Min` propagate NaN (numpy / CPU-EP semantics).
const REDUCE_SRC: &str = r#"
extern "C" __global__ void reduce_f32(
    const float*     x,
    float*           y,
    const long long* base_off,     // [out_count]
    const long long* delta_off,    // [reduce_count]
    const int        out_count,
    const int        reduce_count,
    const int        op,           // 0 sum, 1 max, 2 min
    const int        is_mean)
{
    const int o = blockIdx.x;
    if (o >= out_count) return;

    const float NEG_INF = __int_as_float(0xff800000);
    const float POS_INF = __int_as_float(0x7f800000);
    const float QNAN    = __int_as_float(0x7fc00000);

    extern __shared__ float red[];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    const size_t base = (size_t)base_off[o];

    float acc = (op == 1) ? NEG_INF : (op == 2) ? POS_INF : 0.0f;
    for (int r = tid; r < reduce_count; r += nt) {
        const float v = x[base + (size_t)delta_off[r]];
        if (op == 1)      acc = (isnan(acc) || isnan(v)) ? QNAN : fmaxf(acc, v);
        else if (op == 2) acc = (isnan(acc) || isnan(v)) ? QNAN : fminf(acc, v);
        else              acc += v;
    }
    red[tid] = acc;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            const float a = red[tid], b = red[tid + off];
            if (op == 1)      red[tid] = (isnan(a) || isnan(b)) ? QNAN : fmaxf(a, b);
            else if (op == 2) red[tid] = (isnan(a) || isnan(b)) ? QNAN : fminf(a, b);
            else              red[tid] = a + b;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float out = red[0];
        if (is_mean) out /= (float)reduce_count;
        y[o] = out;
    }
}

extern "C" __global__ void reduce_i64_sum(
    const long long* x,
    long long*       y,
    const long long* base_off,
    const long long* delta_off,
    const int        out_count,
    const int        reduce_count)
{
    const int o = blockIdx.x;
    if (o >= out_count) return;

    extern __shared__ long long red_i64[];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    const size_t base = (size_t)base_off[o];

    long long acc = 0;
    for (int r = tid; r < reduce_count; r += nt) {
        acc += x[base + (size_t)delta_off[r]];
    }
    red_i64[tid] = acc;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red_i64[tid] += red_i64[tid + off];
        __syncthreads();
    }
    if (tid == 0) y[o] = red_i64[0];
}
"#;

const REDUCE_MODULE: &str = "reduce_f32";
const REDUCE_ENTRY: &str = "reduce_f32";
const REDUCE_I64_SUM_ENTRY: &str = "reduce_i64_sum";

/// Threads per block for the reduction (power of two → exact tree reduce).
const REDUCE_BLOCK: u32 = 256;

/// The reduction to apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
}

impl ReduceOp {
    fn name(self) -> &'static str {
        match self {
            ReduceOp::Sum => "ReduceSum",
            ReduceOp::Mean => "ReduceMean",
            ReduceOp::Max => "ReduceMax",
            ReduceOp::Min => "ReduceMin",
        }
    }

    /// (`op` tag for the kernel, `is_mean`).
    fn kernel_tags(self) -> (i32, i32) {
        match self {
            ReduceOp::Sum => (0, 0),
            ReduceOp::Mean => (0, 1),
            ReduceOp::Max => (1, 0),
            ReduceOp::Min => (2, 0),
        }
    }

    fn cudnn_op(self) -> Option<CudnnReduceOp> {
        match self {
            ReduceOp::Sum => Some(CudnnReduceOp::Add),
            ReduceOp::Mean => Some(CudnnReduceOp::Average),
            ReduceOp::Max | ReduceOp::Min => None,
        }
    }
}

/// A resolved reduction: which axes are reduced, plus the derived
/// `base`/`delta` offset tables and the expected output shape. Computed on the
/// host (GPU-free), so it is directly unit-testable.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ReductionPlan {
    /// Input base offset for each output element (`len == out_count`).
    pub base: Vec<i64>,
    /// Offset delta for each element of a reduction group (`len == reduce_count`).
    pub delta: Vec<i64>,
    /// Expected output shape (keepdims-aware).
    pub out_shape: Vec<usize>,
}

/// Row-major contiguous strides for `shape`.
fn contiguous_strides(shape: &[usize]) -> Vec<i64> {
    let mut strides = vec![0i64; shape.len()];
    let mut acc = 1i64;
    for d in (0..shape.len()).rev() {
        strides[d] = acc;
        acc *= shape[d] as i64;
    }
    strides
}

fn contiguous_strides_usize(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    let mut acc = 1usize;
    for d in (0..shape.len()).rev() {
        strides[d] = acc;
        acc *= shape[d];
    }
    strides
}

fn reduced_output_shape(in_shape: &[usize], reduce: &[bool], keepdims: bool) -> Vec<usize> {
    let mut out_shape = Vec::with_capacity(in_shape.len());
    for (dim, &is_reduced) in in_shape.iter().zip(reduce) {
        if is_reduced {
            if keepdims {
                out_shape.push(1);
            }
        } else {
            out_shape.push(*dim);
        }
    }
    out_shape
}

/// Build same-rank input/output descriptors; squeezed ONNX output dimensions
/// remain size-one in cuDNN because this preserves the same contiguous storage.
pub(crate) fn cudnn_reduce_specs(
    dtype: DataType,
    in_shape: &[usize],
    reduce: &[bool],
) -> Result<(TensorDescriptorSpec, TensorDescriptorSpec)> {
    let cudnn_out_shape: Vec<usize> = in_shape
        .iter()
        .zip(reduce)
        .map(|(&dim, &is_reduced)| if is_reduced { 1 } else { dim })
        .collect();
    let input = TensorDescriptorSpec::new(dtype, in_shape, &contiguous_strides_usize(in_shape))?;
    let output = TensorDescriptorSpec::new(
        dtype,
        &cudnn_out_shape,
        &contiguous_strides_usize(&cudnn_out_shape),
    )?;
    Ok((input, output))
}

/// Build the [`ReductionPlan`] for `in_shape`, a `reduce[d]` mask, and
/// `keepdims`. The `base`/`delta` split is exact because row-major strides are
/// independent per axis (see the module docs).
pub(crate) fn build_plan(in_shape: &[usize], reduce: &[bool], keepdims: bool) -> ReductionPlan {
    let rank = in_shape.len();
    let strides = contiguous_strides(in_shape);

    let kept_axes: Vec<usize> = (0..rank).filter(|&d| !reduce[d]).collect();
    let red_axes: Vec<usize> = (0..rank).filter(|&d| reduce[d]).collect();

    let kept_dims: Vec<usize> = kept_axes.iter().map(|&d| in_shape[d]).collect();
    let red_dims: Vec<usize> = red_axes.iter().map(|&d| in_shape[d]).collect();

    let base = enumerate_offsets(&kept_dims, &kept_axes, &strides);
    let delta = enumerate_offsets(&red_dims, &red_axes, &strides);

    // Output shape: kept dims in order; reduced dims become size-1 (keepdims) or
    // are squeezed out.
    let out_shape = reduced_output_shape(in_shape, reduce, keepdims);

    ReductionPlan {
        base,
        delta,
        out_shape,
    }
}

/// Enumerate the input offsets for every multi-index over `dims` (row-major),
/// where `axes[k]` is the input axis of `dims[k]` and `strides` are the input
/// strides. Returns `[0]` for an empty dim set (a single all-zero coordinate).
fn enumerate_offsets(dims: &[usize], axes: &[usize], strides: &[i64]) -> Vec<i64> {
    let total: usize = dims.iter().product::<usize>().max(1);
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; dims.len()];
    loop {
        let mut off = 0i64;
        for k in 0..dims.len() {
            off += idx[k] as i64 * strides[axes[k]];
        }
        out.push(off);
        if !next_index(dims, &mut idx) {
            break;
        }
    }
    out
}

/// Increment a row-major multi-index `idx` within `dims`; returns `false` on
/// wrap (end of iteration). An empty `dims` yields a single iteration.
fn next_index(dims: &[usize], idx: &mut [usize]) -> bool {
    for d in (0..dims.len()).rev() {
        idx[d] += 1;
        if idx[d] < dims[d] {
            return true;
        }
        idx[d] = 0;
    }
    false
}

macro_rules! reduce_factory {
    ($factory:ident, $variant:expr) => {
        /// Factory reading `axes` (optional attribute), `keepdims` (default 1)
        /// and `noop_with_empty_axes` (default 0), plus the shared runtime.
        pub struct $factory {
            pub runtime: Arc<CudaRuntime>,
        }
        impl KernelFactory for $factory {
            fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
                let axes_attr = node
                    .attr("axes")
                    .and_then(|a| a.as_ints())
                    .map(<[i64]>::to_vec);
                let keepdims = node.attr("keepdims").and_then(|a| a.as_int()).unwrap_or(1) != 0;
                let noop_with_empty_axes = node
                    .attr("noop_with_empty_axes")
                    .and_then(|a| a.as_int())
                    .unwrap_or(0)
                    != 0;
                Ok(Box::new(ReduceKernel {
                    op: $variant,
                    axes_attr,
                    keepdims,
                    noop_with_empty_axes,
                    runtime: self.runtime.clone(),
                }))
            }
        }
    };
}

reduce_factory!(ReduceSumFactory, ReduceOp::Sum);
reduce_factory!(ReduceMeanFactory, ReduceOp::Mean);
reduce_factory!(ReduceMaxFactory, ReduceOp::Max);
reduce_factory!(ReduceMinFactory, ReduceOp::Min);

/// f32 reduction kernel carrying the op, the attribute `axes` (opset < 13/18),
/// `keepdims`, `noop_with_empty_axes`, and the shared runtime.
#[derive(Debug)]
pub struct ReduceKernel {
    op: ReduceOp,
    axes_attr: Option<Vec<i64>>,
    keepdims: bool,
    noop_with_empty_axes: bool,
    runtime: Arc<CudaRuntime>,
}

/// Resolve the reduced-axis mask from the raw axes list (input or attribute),
/// honouring `noop_with_empty_axes`. Mirrors the CPU EP.
pub(crate) fn resolve_reduce_mask(
    op: &str,
    axes_raw: &Option<Vec<i64>>,
    rank: usize,
    noop_with_empty_axes: bool,
) -> Result<Vec<bool>> {
    let mut reduce = vec![false; rank];
    match axes_raw {
        Some(a) if a.is_empty() => {
            if !noop_with_empty_axes {
                reduce.iter_mut().for_each(|r| *r = true);
            }
        }
        Some(axes) => {
            for &a in axes {
                let ax = if a < 0 { a + rank as i64 } else { a };
                if ax < 0 || ax as usize >= rank {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep {op}: axis {a} is out of range for a rank-{rank} input; \
                         axis must lie in [-{rank}, {rank})"
                    )));
                }
                reduce[ax as usize] = true;
            }
        }
        None => {
            if !noop_with_empty_axes {
                reduce.iter_mut().for_each(|r| *r = true);
            }
        }
    }
    Ok(reduce)
}

impl ReduceKernel {
    /// Read the optional axes **input** (opset 13/18+) off the device as `i64`.
    fn read_axes_input(&self, op: &str, axes: &TensorView) -> Result<Vec<i64>> {
        if !axes.is_contiguous() {
            return Err(not_implemented(format!(
                "{op} with a non-contiguous (strided) axes input; materialise it first"
            )));
        }
        let n = axes.numel();
        let src = cuptr(axes.data_ptr::<u8>() as *const c_void);
        match axes.dtype {
            DataType::Int64 => {
                let mut bytes = vec![0u8; n * std::mem::size_of::<i64>()];
                // SAFETY: `src` is a live device allocation of `n` i64 elements
                // (contiguous, validated); `bytes` is sized to match.
                unsafe { self.runtime.dtoh(&mut bytes, src) }?;
                Ok(bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_ne_bytes(c.try_into().unwrap()))
                    .collect())
            }
            DataType::Int32 => {
                let mut bytes = vec![0u8; n * std::mem::size_of::<i32>()];
                // SAFETY: as above, for `n` i32 elements.
                unsafe { self.runtime.dtoh(&mut bytes, src) }?;
                Ok(bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_ne_bytes(c.try_into().unwrap()) as i64)
                    .collect())
            }
            other => Err(not_implemented(format!(
                "{op} with axes input dtype {other:?} (expected int32 or int64)"
            ))),
        }
    }

    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let op = self.op.name();
        if !(1..=2).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: expected 1-2 inputs (data[, axes]) and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let x = &inputs[0];
        let cudnn_op = self.op.cudnn_op();
        let supported_dtype = if self.op == ReduceOp::Sum && x.dtype == DataType::Int64 {
            true
        } else if cudnn_op.is_some() {
            matches!(
                x.dtype,
                DataType::Float32 | DataType::Float16 | DataType::BFloat16
            )
        } else {
            x.dtype == DataType::Float32
        };
        if !supported_dtype {
            return Err(not_implemented(format!(
                "{op} with input dtype {:?} (sum supports i64/f32/f16/bf16; mean supports \
                 f32/f16/bf16; max/min are f32)",
                x.dtype
            )));
        }
        if outputs[0].dtype != x.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output dtype {:?} must equal input dtype {:?}",
                outputs[0].dtype, x.dtype
            )));
        }
        if !x.is_contiguous() || !outputs[0].is_contiguous() {
            return Err(not_implemented(format!(
                "{op} with a non-contiguous (strided) input/output; materialise it first"
            )));
        }
        let rank = x.shape.len();

        // Resolve axes: input 1 (opset 13/18+) beats the attribute; both absent
        // means reduce-all (unless noop_with_empty_axes selects identity).
        let axes_raw: Option<Vec<i64>> = if inputs.len() == 2 {
            Some(self.read_axes_input(op, &inputs[1])?)
        } else {
            self.axes_attr.clone()
        };
        let reduce = resolve_reduce_mask(op, &axes_raw, rank, self.noop_with_empty_axes)?;
        let expected_shape = reduced_output_shape(x.shape, &reduce, self.keepdims);

        if outputs[0].shape != expected_shape.as_slice() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output shape {:?} does not match the reduced shape {:?} \
                 (axes {:?}, keepdims {})",
                outputs[0].shape, expected_shape, axes_raw, self.keepdims
            )));
        }

        if x.numel() == 0 || outputs[0].numel() == 0 {
            return Ok(());
        }

        if !reduce.iter().any(|&axis| axis) || rank == 0 {
            let src = cuptr(x.data_ptr::<u8>() as *const c_void);
            let dst = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
            if src != dst {
                // SAFETY: identity reduction has equal input/output storage size.
                unsafe { self.runtime.dtod(src, dst, x.byte_size()) }?;
            }
            return Ok(());
        }

        if x.dtype != DataType::Int64
            && let Some(cudnn_op) = cudnn_op
        {
            if self.runtime.cudnn().is_available() {
                let (input_spec, output_spec) = cudnn_reduce_specs(x.dtype, x.shape, &reduce)?;
                let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
                let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
                self.runtime.cudnn().with_handle(|handle| {
                    handle.reduce(
                        &input_spec,
                        &output_spec,
                        cudnn_op,
                        CudnnBufferPair {
                            input: x_ptr,
                            output: y_ptr,
                            input_numel: x.numel(),
                            output_numel: outputs[0].numel(),
                        },
                    )
                })?;
                return self.runtime.synchronize();
            }
            if x.dtype != DataType::Float32 {
                return self.runtime.cudnn().with_handle(|_| Ok(()));
            }
        }

        let plan = build_plan(x.shape, &reduce, self.keepdims);
        let out_count = plan.base.len();
        let reduce_count = plan.delta.len();
        if out_count == 0 || reduce_count == 0 {
            // Empty input (a zero dim) — nothing to compute.
            return Ok(());
        }

        // Upload the base/delta offset tables (i64).
        let base_bytes = as_i64_bytes(&plan.base);
        let delta_bytes = as_i64_bytes(&plan.delta);
        let base_buf = self.runtime.alloc_raw(base_bytes.len())?;
        let delta_buf = self.runtime.alloc_raw(delta_bytes.len())?;

        let result = self.launch(
            x,
            outputs,
            base_buf,
            delta_buf,
            &base_bytes,
            &delta_bytes,
            out_count,
            reduce_count,
        );

        // Always release the scratch tables, even on failure.
        // SAFETY: both pointers came from the `alloc_raw` calls above and are
        // each freed exactly once here.
        let free_base = unsafe { self.runtime.free_raw(base_buf) };
        let free_delta = unsafe { self.runtime.free_raw(delta_buf) };
        result.and(free_base).and(free_delta)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        x: &TensorView,
        outputs: &mut [TensorMut],
        base_buf: CUdeviceptr,
        delta_buf: CUdeviceptr,
        base_bytes: &[u8],
        delta_bytes: &[u8],
        out_count: usize,
        reduce_count: usize,
    ) -> Result<()> {
        let op = self.op.name();
        // SAFETY: `*_buf` are live device allocations sized to `*_bytes`.
        unsafe { self.runtime.htod(base_bytes, base_buf) }?;
        unsafe { self.runtime.htod(delta_bytes, delta_buf) }?;

        let out_i = i32::try_from(out_count).map_err(|_| {
            EpError::KernelFailed(format!("cuda_ep {op}: {out_count} outputs exceed i32"))
        })?;
        let red_i = i32::try_from(reduce_count).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep {op}: reduction group {reduce_count} exceeds i32"
            ))
        })?;
        let grid = u32::try_from(out_count).map_err(|_| {
            EpError::KernelFailed(format!("cuda_ep {op}: {out_count} blocks exceed u32"))
        })?;
        let (op_tag, is_mean) = self.op.kernel_tags();

        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        let entry = if x.dtype == DataType::Int64 {
            REDUCE_I64_SUM_ENTRY
        } else {
            REDUCE_ENTRY
        };
        let func = self
            .runtime
            .nvrtc_function(REDUCE_MODULE, REDUCE_SRC, entry)?;
        let bytes_per_thread = if x.dtype == DataType::Int64 {
            std::mem::size_of::<i64>() as u32
        } else {
            std::mem::size_of::<f32>() as u32
        };
        let cfg =
            self.runtime
                .reduction_launch_config(&func, grid, REDUCE_BLOCK, bytes_per_thread)?;
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&x_ptr)
            .arg(&y_ptr)
            .arg(&base_buf)
            .arg(&delta_buf)
            .arg(&out_i)
            .arg(&red_i);
        if x.dtype != DataType::Int64 {
            builder.arg(&op_tag).arg(&is_mean);
        }
        // SAFETY: `func` is the compiled reduce entry; the argument list/ABI
        // match its signature; `x_ptr`/`y_ptr` and the base/delta buffers are
        // live device allocations sized as validated above.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        self.runtime.synchronize()
    }
}

/// Reinterpret an `i64` slice as native-endian bytes for an H2D upload.
fn as_i64_bytes(v: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 8);
    for &x in v {
        out.extend_from_slice(&x.to_ne_bytes());
    }
    out
}

impl Kernel for ReduceKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn cuda_graph_compatible(&self) -> bool {
        // Per-call alloc/free of the offset tables is not capturable; a pooled
        // stream-ordered allocator (the same MatMul/Attention follow-up) makes
        // this capturable later.
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_point_present_in_source() {
        assert!(REDUCE_SRC.contains(REDUCE_ENTRY));
    }

    #[test]
    fn strides_are_row_major() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn plan_reduce_last_axis_keepdims() {
        // [2,3] reduce axis 1, keepdims → out [2,1]; 2 groups of 3.
        let reduce = [false, true];
        let plan = build_plan(&[2, 3], &reduce, true);
        assert_eq!(plan.out_shape, vec![2, 1]);
        assert_eq!(plan.base, vec![0, 3]); // row starts
        assert_eq!(plan.delta, vec![0, 1, 2]); // within-row offsets
    }

    #[test]
    fn plan_reduce_axis0_no_keepdims() {
        // [2,3] reduce axis 0, keepdims=false → out [3]; 3 groups of 2.
        let reduce = [true, false];
        let plan = build_plan(&[2, 3], &reduce, false);
        assert_eq!(plan.out_shape, vec![3]);
        assert_eq!(plan.base, vec![0, 1, 2]); // column starts
        assert_eq!(plan.delta, vec![0, 3]); // stride down the column
    }

    #[test]
    fn plan_reduce_all_axes() {
        let reduce = [true, true];
        let plan = build_plan(&[2, 3], &reduce, true);
        assert_eq!(plan.out_shape, vec![1, 1]);
        assert_eq!(plan.base, vec![0]);
        assert_eq!(plan.delta, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn resolve_mask_negative_axis_and_empty_noop() {
        let m = resolve_reduce_mask("ReduceSum", &Some(vec![-1]), 3, false).unwrap();
        assert_eq!(m, vec![false, false, true]);
        // Explicitly-empty axes with noop → reduce nothing (identity).
        let m = resolve_reduce_mask("ReduceSum", &Some(vec![]), 3, true).unwrap();
        assert_eq!(m, vec![false, false, false]);
        // Explicitly-empty axes without noop → reduce all.
        let m = resolve_reduce_mask("ReduceSum", &Some(vec![]), 3, false).unwrap();
        assert_eq!(m, vec![true, true, true]);
        // No axes given → reduce all.
        let m = resolve_reduce_mask("ReduceSum", &None, 2, false).unwrap();
        assert_eq!(m, vec![true, true]);
    }

    #[test]
    fn resolve_mask_rejects_out_of_range_axis() {
        let e = resolve_reduce_mask("ReduceMax", &Some(vec![5]), 2, false).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("out of range"), "{msg}");
        assert!(msg.contains("axis 5"), "{msg}");
    }

    #[test]
    fn kernel_tags_map_ops() {
        assert_eq!(ReduceOp::Sum.kernel_tags(), (0, 0));
        assert_eq!(ReduceOp::Mean.kernel_tags(), (0, 1));
        assert_eq!(ReduceOp::Max.kernel_tags(), (1, 0));
        assert_eq!(ReduceOp::Min.kernel_tags(), (2, 0));
    }

    #[test]
    fn cudnn_op_mapping_only_ports_sum_and_mean() {
        assert_eq!(ReduceOp::Sum.cudnn_op(), Some(CudnnReduceOp::Add));
        assert_eq!(ReduceOp::Mean.cudnn_op(), Some(CudnnReduceOp::Average));
        assert_eq!(ReduceOp::Max.cudnn_op(), None);
        assert_eq!(ReduceOp::Min.cudnn_op(), None);
    }

    #[test]
    fn cudnn_specs_keep_reduced_axes_as_size_one() {
        let (input, output) =
            cudnn_reduce_specs(DataType::BFloat16, &[2, 3, 4], &[true, false, true]).unwrap();
        assert_eq!(input.dims(), &[1, 2, 3, 4]);
        assert_eq!(input.strides(), &[24, 12, 4, 1]);
        assert_eq!(output.dims(), &[1, 1, 3, 1]);
        assert_eq!(output.strides(), &[3, 3, 1, 1]);
    }
}
