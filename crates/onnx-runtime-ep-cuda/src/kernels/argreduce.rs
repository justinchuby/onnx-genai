//! Per-lane CUDA `ArgMax` / `ArgMin` reduction returning `Int64` indices,
//! matched to the CPU EP (`selection.rs`): it reduces one axis, honours the
//! `keepdims` and `select_last_index` attributes, and breaks ties on the first
//! (or, with `select_last_index`, the last) extremal element. Floating inputs
//! are widened to f32 for the comparison, exactly like the CPU oracle's
//! `to_dense_f32_widen`.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const MODULE: &str = "arg_reduce_v1";

// `op` discriminants shared with the NVRTC source below.
const OP_MAX: i32 = 0;
const OP_MIN: i32 = 1;

const SRC: &str = r#"
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

// Mirror the CPU EP's scan: keep `best`, and for each later candidate update on
// a strict improvement, or (with select_last_index) on an exact tie. NaN never
// satisfies `>`/`<`/`==`, so it never wins — identical to the CPU oracle.
#define DEFINE_ARG_REDUCE(TYPE, SUFFIX) \
extern "C" __global__ void arg_reduce_##SUFFIX( \
    const TYPE* x, long long* out, const unsigned long long lanes, \
    const unsigned long long width, const unsigned long long inner, \
    const int op, const int select_last) { \
    for (unsigned long long lane = blockIdx.x * blockDim.x + threadIdx.x; lane < lanes; \
         lane += (unsigned long long)gridDim.x * blockDim.x) { \
        const unsigned long long outer = lane / inner, i = lane % inner; \
        unsigned long long best = 0; \
        for (unsigned long long d = 1; d < width; ++d) { \
            const float candidate = load_float<TYPE>(x[(outer * width + d) * inner + i]); \
            const float value = load_float<TYPE>(x[(outer * width + best) * inner + i]); \
            const bool better = (op == 0) ? (candidate > value) : (candidate < value); \
            if (better || (select_last && candidate == value)) best = d; \
        } \
        out[lane] = (long long)best; \
    } \
}

DEFINE_ARG_REDUCE(float, f32)
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
DEFINE_ARG_REDUCE(__half, f16)
DEFINE_ARG_REDUCE(__nv_bfloat16, bf16)
#endif
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgOp {
    Max,
    Min,
}

impl ArgOp {
    fn name(self) -> &'static str {
        match self {
            Self::Max => "ArgMax",
            Self::Min => "ArgMin",
        }
    }

    fn discriminant(self) -> i32 {
        match self {
            Self::Max => OP_MAX,
            Self::Min => OP_MIN,
        }
    }
}

pub struct ArgReduceFactory {
    pub op: ArgOp,
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for ArgReduceFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ArgReduceKernel {
            op: self.op,
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(0),
            keepdims: node.attr("keepdims").and_then(|a| a.as_int()).unwrap_or(1) != 0,
            select_last_index: node
                .attr("select_last_index")
                .and_then(|a| a.as_int())
                .unwrap_or(0)
                != 0,
            runtime: self.runtime.clone(),
        }))
    }
}

struct ArgReduceKernel {
    op: ArgOp,
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
    runtime: Arc<CudaRuntime>,
}

impl Kernel for ArgReduceKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let op = self.op.name();
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: expected 1 input and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let x = &inputs[0];
        let suffix = match x.dtype {
            DataType::Float32 => "f32",
            DataType::Float16 => "f16",
            DataType::BFloat16 => "bf16",
            other => {
                return Err(not_implemented(format!(
                    "{op} with dtype {other:?} (supported: Float32, Float16, BFloat16)"
                )));
            }
        };
        if x.dtype != DataType::Float32 {
            self.runtime.require_nvrtc_half_headers(op)?;
        }
        require_contiguous(op, "input", x.is_contiguous())?;
        require_contiguous(op, "output", outputs[0].is_contiguous())?;
        if outputs[0].dtype != DataType::Int64 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output dtype {:?} must be Int64",
                outputs[0].dtype
            )));
        }

        let rank = x.shape.len();
        let axis = normalize_axis(op, self.axis, rank)?;
        let width = x.shape[axis];
        if width == 0 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: reduced axis {axis} must be non-empty"
            )));
        }
        let expected_out = arg_out_shape(x.shape, axis, self.keepdims);
        if outputs[0].shape != expected_out {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output shape {:?} must equal {expected_out:?} \
                 (input {:?} reduced on axis {axis}, keepdims={})",
                outputs[0].shape, x.shape, self.keepdims
            )));
        }

        let inner = x.shape[axis + 1..].iter().product::<usize>();
        let outer = x.shape[..axis].iter().product::<usize>();
        let lanes = outer * inner;
        if lanes == 0 {
            return Ok(());
        }

        let entry = format!("arg_reduce_{suffix}");
        let func = self.runtime.nvrtc_function(MODULE, SRC, &entry)?;
        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let out_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let lanes_u = lanes as u64;
        let width_u = width as u64;
        let inner_u = inner as u64;
        let op_d = self.op.discriminant();
        let select_last = i32::from(self.select_last_index);
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            (lanes as u64).saturating_mul(width_u)
        });
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&x_ptr)
            .arg(&out_ptr)
            .arg(&lanes_u)
            .arg(&width_u)
            .arg(&inner_u)
            .arg(&op_d)
            .arg(&select_last);
        // SAFETY: x covers outer*width*inner contiguous elements; out covers
        // `lanes` Int64 elements; the kernel only reads/writes within those.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid_for(lanes), 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        if self.runtime.is_capturing()? {
            return Ok(());
        }
        self.runtime.synchronize()
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
}

fn normalize_axis(op: &str, axis: i64, rank: usize) -> Result<usize> {
    let normalized = if axis < 0 { axis + rank as i64 } else { axis };
    if normalized < 0 || normalized as usize >= rank {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep {op}: axis {axis} out of range for rank {rank}"
        )));
    }
    Ok(normalized as usize)
}

fn arg_out_shape(in_shape: &[usize], axis: usize, keepdims: bool) -> Vec<usize> {
    let mut out = Vec::with_capacity(in_shape.len());
    for (index, &dim) in in_shape.iter().enumerate() {
        if index == axis {
            if keepdims {
                out.push(1);
            }
        } else {
            out.push(dim);
        }
    }
    out
}

fn grid_for(n: usize) -> u32 {
    const MAX_BLOCKS: usize = 65_535;
    n.div_ceil(BLOCK as usize).clamp(1, MAX_BLOCKS) as u32
}

fn require_contiguous(op: &str, name: &str, contiguous: bool) -> Result<()> {
    if !contiguous {
        return Err(not_implemented(format!(
            "{op} with a non-contiguous (strided) {name}; materialise it before the op"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_shape_honours_keepdims() {
        assert_eq!(arg_out_shape(&[2, 3, 4], 1, true), vec![2, 1, 4]);
        assert_eq!(arg_out_shape(&[2, 3, 4], 1, false), vec![2, 4]);
        assert_eq!(arg_out_shape(&[5], 0, false), Vec::<usize>::new());
        assert_eq!(arg_out_shape(&[5], 0, true), vec![1]);
    }

    #[test]
    fn axis_normalisation_matches_cpu() {
        assert_eq!(normalize_axis("ArgMax", -1, 3).unwrap(), 2);
        assert_eq!(normalize_axis("ArgMax", 0, 3).unwrap(), 0);
        assert!(normalize_axis("ArgMax", 3, 3).is_err());
        assert!(normalize_axis("ArgMax", -4, 3).is_err());
    }

    #[test]
    fn nvrtc_source_defines_every_dtype_entry() {
        assert!(SRC.contains("DEFINE_ARG_REDUCE(float, f32)"));
        assert!(SRC.contains("DEFINE_ARG_REDUCE(__half, f16)"));
        assert!(SRC.contains("DEFINE_ARG_REDUCE(__nv_bfloat16, bf16)"));
    }
}
