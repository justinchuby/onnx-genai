//! Unary floating-point **predicate** ops — `IsInf` and `IsNaN` — on the GPU via
//! runtime-compiled (NVRTC) `extern "C"` kernels. Each reads one f32/f16/bf16
//! input and writes a canonical 1-byte `Bool` output, matching the CPU EP
//! (`crates/onnx-runtime-ep-cpu/src/kernels/is_inf.rs` and `is_nan.rs`).
//!
//! ## Scope (all limits are actionable errors, never panics)
//!
//! * **`IsInf`**: `detect_positive`/`detect_negative` int attributes (default 1);
//!   a lane is `true` when the value is `+inf` and positive detection is on, or
//!   `-inf` and negative detection is on. f16/bf16 widen losslessly to f32 for
//!   the classification (the widening preserves inf/nan and sign).
//! * **`IsNaN`**: no attributes; a lane is `true` iff the value is NaN.
//! * **dtype:** f32/f16/bf16 input, `Bool` output; other dtypes return an
//!   actionable error naming the op/dtype. (The CPU EP also accepts f64, which
//!   the CUDA half/pointwise slice does not cover — the claim gate rejects it so
//!   an f64 node stays on the CPU EP.)

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

/// Threads per block for the 1-D pointwise grid (a full warp-multiple block).
const BLOCK: u32 = 256;

/// Grid dimension for `n` elements, capped so a huge tensor still fits the grid
/// limit (the kernel is grid-stride, so a capped grid still covers everything).
fn grid_for(n: usize) -> u32 {
    const MAX_BLOCKS: usize = 65_535;
    n.div_ceil(BLOCK as usize).clamp(1, MAX_BLOCKS) as u32
}

/// The NVRTC entry-point suffix for a supported floating input dtype.
fn float_suffix(op: &str, dtype: DataType) -> Result<&'static str> {
    match dtype {
        DataType::Float32 => Ok("f32"),
        DataType::Float16 => Ok("f16"),
        DataType::BFloat16 => Ok("bf16"),
        other => Err(not_implemented(format!(
            "{op} with input dtype {other:?} (supported: Float32, Float16, BFloat16)"
        ))),
    }
}

/// NVRTC source: dtype-templated `IsInf`/`IsNaN` kernels. Half inputs widen to
/// f32 first (the conversion preserves inf/nan and sign), then classify with the
/// device intrinsics — matching the CPU EP's `f16`/`bf16` → f32 promotion.
const PREDICATE_SRC: &str = r#"
#if __has_include(<cuda_fp16.h>) && __has_include(<cuda_bf16.h>)
#define NXRT_HAS_CUDA_HALF_HEADERS 1
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

template <typename T> __device__ float load_float(T value);
template <> __device__ float load_float<float>(float value) { return value; }
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
template <> __device__ float load_float<__half>(__half value) { return __half2float(value); }
template <> __device__ float load_float<__nv_bfloat16>(__nv_bfloat16 value) { return __bfloat162float(value); }
#endif

#define DEFINE_ISINF(TYPE, SUFFIX) \
extern "C" __global__ void is_inf_##SUFFIX(const TYPE* x, unsigned char* y, const int detect_pos, const int detect_neg, const unsigned long long n) { \
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) { \
        float v = load_float<TYPE>(x[i]); \
        bool inf = isinf(v); \
        bool pos = inf && v > 0.0f; \
        bool neg = inf && v < 0.0f; \
        y[i] = ((detect_pos && pos) || (detect_neg && neg)) ? 1 : 0; \
    } \
}
#define DEFINE_ISNAN(TYPE, SUFFIX) \
extern "C" __global__ void is_nan_##SUFFIX(const TYPE* x, unsigned char* y, const unsigned long long n) { \
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) { \
        float v = load_float<TYPE>(x[i]); \
        y[i] = (v != v) ? 1 : 0; \
    } \
}
#define DEFINE_PRED_FOR_TYPE(TYPE, SUFFIX) DEFINE_ISINF(TYPE, SUFFIX) DEFINE_ISNAN(TYPE, SUFFIX)
DEFINE_PRED_FOR_TYPE(float, f32)
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
DEFINE_PRED_FOR_TYPE(__half, f16)
DEFINE_PRED_FOR_TYPE(__nv_bfloat16, bf16)
#endif
"#;

const PREDICATE_MODULE: &str = "unary_predicate_float_v1";

/// A supported unary predicate op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredicateOp {
    IsInf,
    IsNaN,
}

impl PredicateOp {
    fn op_name(self) -> &'static str {
        match self {
            PredicateOp::IsInf => "IsInf",
            PredicateOp::IsNaN => "IsNaN",
        }
    }

    fn stem(self) -> &'static str {
        match self {
            PredicateOp::IsInf => "is_inf",
            PredicateOp::IsNaN => "is_nan",
        }
    }
}

/// Reads the `detect_positive`/`detect_negative` attributes for `IsInf`, each an
/// int flag defaulting to `1`. Any value other than `0`/`1` is a hard error,
/// matching the CPU EP's `IsInf` factory.
fn detect_flag(node: &Node, name: &str) -> Result<i32> {
    match node.attr(name) {
        None => Ok(1),
        Some(attribute) => match attribute.as_int() {
            Some(0) => Ok(0),
            Some(1) => Ok(1),
            _ => Err(EpError::KernelFailed(format!(
                "cuda_ep IsInf: `{name}` must be 0 or 1"
            ))),
        },
    }
}

/// Factory for [`PredicateKernel`]; carries the op identity and shared runtime.
pub struct PredicateFactory {
    pub op: PredicateOp,
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for PredicateFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let (detect_pos, detect_neg) = match self.op {
            PredicateOp::IsInf => (
                detect_flag(node, "detect_positive")?,
                detect_flag(node, "detect_negative")?,
            ),
            PredicateOp::IsNaN => (1, 1),
        };
        Ok(Box::new(PredicateKernel {
            op: self.op,
            detect_pos,
            detect_neg,
            runtime: self.runtime.clone(),
        }))
    }
}

/// NVRTC-backed f32/f16/bf16 unary predicate kernel with a `Bool` output.
#[derive(Debug)]
pub struct PredicateKernel {
    op: PredicateOp,
    detect_pos: i32,
    detect_neg: i32,
    runtime: Arc<CudaRuntime>,
}

impl PredicateKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let op = self.op.op_name();
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: expected 1 input and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let x = &inputs[0];
        let suffix = float_suffix(op, x.dtype)?;
        if x.dtype != DataType::Float32 {
            self.runtime.require_nvrtc_half_headers(op)?;
        }
        if outputs[0].dtype != DataType::Bool {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output dtype {:?} must be Bool",
                outputs[0].dtype
            )));
        }
        if !x.is_contiguous() || !outputs[0].is_contiguous() {
            return Err(not_implemented(format!(
                "{op} with a non-contiguous (strided) tensor; \
                 insert an explicit copy to materialise it before the op"
            )));
        }
        if outputs[0].numel() != x.numel() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output has {} elements, expected {} (same shape as input)",
                outputs[0].numel(),
                x.numel()
            )));
        }

        let n = x.numel();
        let n_u64 = u64::try_from(n)
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep {op}: {n} elements exceed u64")))?;
        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        let entry = format!("{}_{}", self.op.stem(), suffix);
        let func = self
            .runtime
            .nvrtc_function(PREDICATE_MODULE, PREDICATE_SRC, &entry)?;
        let cfg = LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        match self.op {
            PredicateOp::IsInf => {
                let detect_pos = self.detect_pos;
                let detect_neg = self.detect_neg;
                builder
                    .arg(&x_ptr)
                    .arg(&y_ptr)
                    .arg(&detect_pos)
                    .arg(&detect_neg)
                    .arg(&n_u64);
                // SAFETY: `func` is the compiled `is_inf_*` entry; its argument
                // list (const T*, unsigned char*, int, int, unsigned long long)
                // matches; `x_ptr`/`y_ptr` are live device allocations of `n`
                // elements and the count/indexing stay in bounds.
                unsafe { builder.launch(cfg) }
                    .map_err(|e| driver_err(&format!("launch {entry}"), e))?;
            }
            PredicateOp::IsNaN => {
                builder.arg(&x_ptr).arg(&y_ptr).arg(&n_u64);
                // SAFETY: `func` is the compiled `is_nan_*` entry; its argument
                // list (const T*, unsigned char*, unsigned long long) matches;
                // both pointers are live device allocations of `n` elements and
                // the count/indexing stay in bounds.
                unsafe { builder.launch(cfg) }
                    .map_err(|e| driver_err(&format!("launch {entry}"), e))?;
            }
        }
        if self.runtime.is_capturing()? {
            // A stream synchronize is illegal mid-capture. The launch is
            // recorded into the segment graph and replayed, so skip the sync
            // instead of erroring inside the captured segment.
            return Ok(());
        }
        self.runtime.synchronize()
    }
}

impl Kernel for PredicateKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::Supported
    }
}

/// Claim-time contract for `IsInf`/`IsNaN`: one f32/f16/bf16 input, one output,
/// and — for `IsInf` — integer `detect_positive`/`detect_negative` flags. Gated
/// to exactly the dtypes/attrs the kernel implements so an f64 (CPU-only) node
/// or a malformed attribute is never claimed onto CUDA.
pub(crate) fn unsupported_reason(node: &Node, input_dtypes: &[DataType]) -> Option<String> {
    let op = node.op_type.as_str();
    if node.inputs.len() != 1
        || node.outputs.len() != 1
        || node.inputs.iter().any(Option::is_none)
        || input_dtypes.len() != 1
    {
        return Some(format!(
            "{op}: requires 1 present input and 1 output, got {} inputs and {} outputs",
            node.inputs.len(),
            node.outputs.len()
        ));
    }
    if !matches!(
        input_dtypes[0],
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        return Some(format!(
            "{op}: input dtype {:?} unsupported on CUDA EP; expected Float32, Float16, or BFloat16",
            input_dtypes[0]
        ));
    }
    if op == "IsInf" {
        for name in ["detect_positive", "detect_negative"] {
            if let Some(attribute) = node.attr(name)
                && !matches!(attribute.as_int(), Some(0 | 1))
            {
                return Some(format!("IsInf: attribute '{name}' must be 0 or 1"));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, NodeId};

    #[test]
    fn predicate_entry_points_are_present_in_source() {
        for expansion in [
            "DEFINE_PRED_FOR_TYPE(float, f32)",
            "DEFINE_PRED_FOR_TYPE(__half, f16)",
            "DEFINE_PRED_FOR_TYPE(__nv_bfloat16, bf16)",
        ] {
            assert!(
                PREDICATE_SRC.contains(expansion),
                "missing NVRTC generator {expansion}"
            );
        }
    }

    fn isinf_node(attrs: &[(&str, i64)]) -> Node {
        let mut node = Node::new(NodeId(0), "IsInf", vec![], vec![]);
        for &(name, value) in attrs {
            node.attributes.insert(name.into(), Attribute::Int(value));
        }
        node
    }

    #[test]
    fn isinf_flags_default_to_both_and_honour_overrides() {
        let node = isinf_node(&[]);
        assert_eq!(detect_flag(&node, "detect_positive").unwrap(), 1);
        assert_eq!(detect_flag(&node, "detect_negative").unwrap(), 1);

        let node = isinf_node(&[("detect_positive", 0)]);
        assert_eq!(detect_flag(&node, "detect_positive").unwrap(), 0);
        assert_eq!(detect_flag(&node, "detect_negative").unwrap(), 1);
    }

    #[test]
    fn isinf_rejects_non_binary_flag() {
        let node = isinf_node(&[("detect_positive", 2)]);
        assert!(detect_flag(&node, "detect_positive").is_err());
    }
}
