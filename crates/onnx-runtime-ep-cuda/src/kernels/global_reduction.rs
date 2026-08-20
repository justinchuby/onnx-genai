//! Global pooling and axis-wise `LpNormalization`.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float load_float_value(
    const void* values, int dtype, unsigned long long index) {
  if (dtype == 0) return ((const float*)values)[index];
  if (dtype == 1) return __half2float(((const __half*)values)[index]);
  return __bfloat162float(((const __nv_bfloat16*)values)[index]);
}

__device__ __forceinline__ void store_float_value(
    void* values, int dtype, unsigned long long index, float value) {
  if (dtype == 0) ((float*)values)[index] = value;
  else if (dtype == 1) ((__half*)values)[index] = __float2half_rn(value);
  else ((__nv_bfloat16*)values)[index] = __float2bfloat16_rn(value);
}

// Round an fp32 intermediate to the activation dtype and back, reproducing the
// narrow-precision round an ONNX op emits when it *stores* an intermediate
// tensor. Used by `l2_normalize_faithful` to replicate the exact rounding of the
// exported `ReduceSumSquare -> Sqrt -> Div` chain (see the kernel below).
__device__ __forceinline__ float round_to_dtype(int dtype, float value) {
  if (dtype == 0) return value;
  if (dtype == 1) return __half2float(__float2half_rn(value));
  return __bfloat162float(__float2bfloat16_rn(value));
}

extern "C" __global__ void global_pool(
    const void* x, void* y, unsigned long long groups,
    unsigned long long spatial, int dtype, int kind, int p) {
  const unsigned long long group = blockIdx.x;
  if (group >= groups) return;
  extern __shared__ float reduction[];
  const float negative_infinity = -__int_as_float(0x7f800000);
  float value = kind == 1 ? negative_infinity : 0.0f;
  for (unsigned long long i = threadIdx.x; i < spatial; i += blockDim.x) {
    const float input = load_float_value(x, dtype, group * spatial + i);
    if (kind == 0) value += input;
    else if (kind == 1) value = fmaxf(value, input);
    else value += powf(fabsf(input), (float)p);
  }
  reduction[threadIdx.x] = value;
  __syncthreads();
  for (unsigned int offset = blockDim.x >> 1; offset; offset >>= 1) {
    if (threadIdx.x < offset) {
      if (kind == 1)
        reduction[threadIdx.x] =
            fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + offset]);
      else
        reduction[threadIdx.x] += reduction[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float output = reduction[0];
    if (spatial == 0) output = kind == 1 ? negative_infinity : 0.0f;
    else if (kind == 0) output /= (float)spatial;
    else if (kind == 2) output = powf(output, 1.0f / (float)p);
    store_float_value(y, dtype, group, output);
  }
}

extern "C" __global__ void lp_normalization(
    const void* x, void* y, unsigned long long groups,
    unsigned long long axis_length, unsigned long long inner,
    int dtype, int p) {
  const unsigned long long group = blockIdx.x;
  if (group >= groups) return;
  const unsigned long long outer_index = group / inner;
  const unsigned long long inner_index = group % inner;
  const unsigned long long base = outer_index * axis_length * inner + inner_index;
  extern __shared__ float reduction[];
  float norm = 0.0f;
  for (unsigned long long axis_index = threadIdx.x;
       axis_index < axis_length; axis_index += blockDim.x) {
    const float value =
        fabsf(load_float_value(x, dtype, base + axis_index * inner));
    norm += p == 1 ? value : value * value;
  }
  reduction[threadIdx.x] = norm;
  __syncthreads();
  for (unsigned int offset = blockDim.x >> 1; offset; offset >>= 1) {
    if (threadIdx.x < offset)
      reduction[threadIdx.x] += reduction[threadIdx.x + offset];
    __syncthreads();
  }
  norm = p == 1 ? reduction[0] : sqrtf(reduction[0]);
  norm = fmaxf(norm, 1.1754943508222875e-38f);
  for (unsigned long long axis_index = threadIdx.x;
       axis_index < axis_length; axis_index += blockDim.x) {
    const unsigned long long index = base + axis_index * inner;
    store_float_value(y, dtype, index, load_float_value(x, dtype, index) / norm);
  }
}

// Byte-faithful fusion of the exported Gated-DeltaNet Q/K L2-normalize chain
//   sq   = ReduceSumSquare(x, axis)   // fp32 accumulate, store rounded to dtype
//   nrm  = Sqrt(sq)                   // fp32 sqrt, store rounded to dtype
//   y    = Div(x, nrm)                // fp32 divide, store rounded to dtype
// into a single launch, reproducing every intermediate narrow-precision round so
// the result is *bit-identical* to running the three ops separately (the CUDA EP
// lowers ReduceSumSquare to `reduce.rs::reduce_ext`, which uses this exact 256-way
// fp32 accumulate + shared-memory tree reduce). Unlike `lp_normalization`, this
// keeps the two intermediate `round_to_dtype` steps and omits the `fmaxf` norm
// clamp, matching the exported chain rather than being strictly more accurate.
extern "C" __global__ void l2_normalize_faithful(
    const void* x, void* y, unsigned long long groups,
    unsigned long long axis_length, unsigned long long inner, int dtype) {
  const unsigned long long group = blockIdx.x;
  if (group >= groups) return;
  const unsigned long long outer_index = group / inner;
  const unsigned long long inner_index = group % inner;
  const unsigned long long base = outer_index * axis_length * inner + inner_index;
  extern __shared__ float reduction[];
  float acc = 0.0f;
  for (unsigned long long axis_index = threadIdx.x;
       axis_index < axis_length; axis_index += blockDim.x) {
    const float value = load_float_value(x, dtype, base + axis_index * inner);
    acc += value * value;
  }
  reduction[threadIdx.x] = acc;
  __syncthreads();
  for (unsigned int offset = blockDim.x >> 1; offset; offset >>= 1) {
    if (threadIdx.x < offset)
      reduction[threadIdx.x] += reduction[threadIdx.x + offset];
    __syncthreads();
  }
  // ReduceSumSquare stores its fp32 sum rounded to the activation dtype; Sqrt
  // then reloads it, takes the fp32 sqrt, and stores that rounded to dtype too.
  const float sumsq = round_to_dtype(dtype, reduction[0]);
  const float norm = round_to_dtype(dtype, sqrtf(sumsq));
  for (unsigned long long axis_index = threadIdx.x;
       axis_index < axis_length; axis_index += blockDim.x) {
    const unsigned long long index = base + axis_index * inner;
    // Div: fp32 divide of the (dtype-precision) operands, rounded to dtype.
    store_float_value(y, dtype, index, load_float_value(x, dtype, index) / norm);
  }
}
"#;

fn dtype_code(dtype: DataType, op: &str) -> Result<i32> {
    match dtype {
        DataType::Float32 => Ok(0),
        DataType::Float16 => Ok(1),
        DataType::BFloat16 => Ok(2),
        other => Err(not_implemented(format!(
            "{op} dtype {other:?} (supported: Float32, Float16, BFloat16)"
        ))),
    }
}

#[derive(Clone, Copy)]
pub enum GlobalPoolKind {
    Average,
    Max,
    Lp(i32),
}

pub struct GlobalPoolFactory {
    pub kind: GlobalPoolKind,
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for GlobalPoolFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let kind = match self.kind {
            GlobalPoolKind::Lp(_) => {
                let p = node
                    .attr("p")
                    .and_then(|attribute| attribute.as_int())
                    .unwrap_or(2);
                let p = i32::try_from(p).ok().filter(|p| *p > 0).ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GlobalLpPool: p must be a positive 32-bit integer".into(),
                    )
                })?;
                GlobalPoolKind::Lp(p)
            }
            other => other,
        };
        Ok(Box::new(GlobalPoolKernel {
            kind,
            runtime: self.runtime.clone(),
        }))
    }
}

struct GlobalPoolKernel {
    kind: GlobalPoolKind,
    runtime: Arc<CudaRuntime>,
}

impl Kernel for GlobalPoolKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep GlobalPool: expected 1 input and 1 output".into(),
            ));
        }
        let input = &inputs[0];
        let output = &mut outputs[0];
        if input.shape.len() < 3 {
            return Err(EpError::KernelFailed(
                "cuda_ep GlobalPool: input must have rank at least 3".into(),
            ));
        }
        if input.dtype != output.dtype || !input.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented(
                "GlobalPool requires contiguous input/output with matching dtypes",
            ));
        }
        let dtype = dtype_code(input.dtype, "GlobalPool")?;
        if dtype != 0 {
            self.runtime.require_nvrtc_half_headers("GlobalPool")?;
        }
        let expected = [input.shape[0], input.shape[1]]
            .into_iter()
            .chain(std::iter::repeat_n(1, input.shape.len() - 2))
            .collect::<Vec<_>>();
        if output.shape != expected {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GlobalPool: output shape {:?}, expected {expected:?}",
                output.shape
            )));
        }
        let groups = input.shape[0].saturating_mul(input.shape[1]) as u64;
        if groups == 0 {
            return Ok(());
        }
        let spatial = input.shape[2..].iter().product::<usize>() as u64;
        let (kind, p) = match self.kind {
            GlobalPoolKind::Average => (0i32, 1i32),
            GlobalPoolKind::Max => (1, 1),
            GlobalPoolKind::Lp(p) => (2, p),
        };
        let function = self
            .runtime
            .nvrtc_function("global_reduction_v1", SOURCE, "global_pool")?;
        let x = cuptr(input.data_ptr::<u8>() as *const c_void);
        let y = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&x)
            .arg(&y)
            .arg(&groups)
            .arg(&spatial)
            .arg(&dtype)
            .arg(&kind)
            .arg(&p);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    u32::try_from(groups).map_err(|_| {
                        EpError::KernelFailed("cuda_ep GlobalPool: group count exceeds u32".into())
                    })?,
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * 4,
            })
        }
        .map(|_| ())
        .map_err(|error| driver_err("launch GlobalPool", error))
    }
}

pub struct LpNormalizationFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for LpNormalizationFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let p = node
            .attr("p")
            .and_then(|attribute| attribute.as_int())
            .unwrap_or(2);
        if !matches!(p, 1 | 2) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LpNormalization: p must be 1 or 2, got {p}"
            )));
        }
        Ok(Box::new(LpNormalizationKernel {
            axis: node
                .attr("axis")
                .and_then(|attribute| attribute.as_int())
                .unwrap_or(-1),
            p: p as i32,
            // Private marker set by `CudaL2NormalizeFusion` when this node
            // replaces an exported `ReduceSumSquare -> Sqrt -> Div` L2-normalize
            // chain: route to the byte-faithful kernel so decode stays
            // token-identical to the unfused graph. Absent (or p != 2) → the
            // standard, strictly-more-accurate fp32 `lp_normalization` kernel.
            faithful_chain: p == 2
                && node
                    .attr("fused_reduce_chain")
                    .and_then(|attribute| attribute.as_int())
                    .unwrap_or(0)
                    == 1,
            runtime: self.runtime.clone(),
        }))
    }
}

struct LpNormalizationKernel {
    axis: i64,
    p: i32,
    faithful_chain: bool,
    runtime: Arc<CudaRuntime>,
}

impl Kernel for LpNormalizationKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep LpNormalization: expected 1 input and 1 output".into(),
            ));
        }
        let input = &inputs[0];
        let output = &mut outputs[0];
        if input.shape != output.shape
            || input.dtype != output.dtype
            || !input.is_contiguous()
            || !output.is_contiguous()
        {
            return Err(not_implemented(
                "LpNormalization requires contiguous same-shape, same-dtype tensors",
            ));
        }
        let rank = input.shape.len();
        let axis = if self.axis < 0 {
            self.axis + rank as i64
        } else {
            self.axis
        };
        if axis < 0 || axis as usize >= rank {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LpNormalization: axis {} out of range for rank {rank}",
                self.axis
            )));
        }
        let dtype = dtype_code(input.dtype, "LpNormalization")?;
        if dtype != 0 {
            self.runtime.require_nvrtc_half_headers("LpNormalization")?;
        }
        let axis = axis as usize;
        let outer = input.shape[..axis].iter().product::<usize>();
        let axis_length = input.shape[axis];
        let inner = input.shape[axis + 1..].iter().product::<usize>();
        let groups = outer.saturating_mul(inner) as u64;
        if groups == 0 || axis_length == 0 {
            return Ok(());
        }
        let axis_length = axis_length as u64;
        let inner = inner as u64;
        let x = cuptr(input.data_ptr::<u8>() as *const c_void);
        let y = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let grid_dim = (
            u32::try_from(groups).map_err(|_| {
                EpError::KernelFailed("cuda_ep LpNormalization: group count exceeds u32".into())
            })?,
            1,
            1,
        );
        let launch = LaunchConfig {
            grid_dim,
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * 4,
        };
        if self.faithful_chain {
            // Bit-identical replica of the exported ReduceSumSquare→Sqrt→Div
            // chain (p is fixed at 2 for L2 normalization).
            let function = self.runtime.nvrtc_function(
                "global_reduction_v1",
                SOURCE,
                "l2_normalize_faithful",
            )?;
            let mut builder = self.runtime.stream().launch_builder(&function);
            builder
                .arg(&x)
                .arg(&y)
                .arg(&groups)
                .arg(&axis_length)
                .arg(&inner)
                .arg(&dtype);
            unsafe { builder.launch(launch) }
                .map(|_| ())
                .map_err(|error| driver_err("launch LpNormalization", error))
        } else {
            let function =
                self.runtime
                    .nvrtc_function("global_reduction_v1", SOURCE, "lp_normalization")?;
            let mut builder = self.runtime.stream().launch_builder(&function);
            builder
                .arg(&x)
                .arg(&y)
                .arg(&groups)
                .arg(&axis_length)
                .arg(&inner)
                .arg(&dtype)
                .arg(&self.p);
            unsafe { builder.launch(launch) }
                .map(|_| ())
                .map_err(|error| driver_err("launch LpNormalization", error))
        }
    }
}

#[cfg(test)]
mod l2_faithful_tests {
    use std::ffi::c_void;

    use half::bf16;
    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, Kernel, TensorMut, TensorView};
    use onnx_runtime_ir::{DataType, DeviceId};

    use super::LpNormalizationKernel;
    use crate::runtime::CudaRuntime;

    /// Deterministic bf16 test vector in roughly `[-4, 4]`, avoiding tiny norms.
    fn sample(len: usize, seed: u32) -> Vec<bf16> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let unit = (state >> 8) as f32 / (1u32 << 24) as f32; // [0,1)
                bf16::from_f32(unit * 8.0 - 4.0)
            })
            .collect()
    }

    /// CPU replica of the exported `ReduceSumSquare -> Sqrt -> Div` L2-normalize
    /// chain *as the CUDA EP runs it*: fp32 square + the kernel's 256-way
    /// shared-memory tree reduce, then the three intermediate narrow-precision
    /// rounds (sum, sqrt, quotient) with no norm clamp. `axis_length <= 256`, so
    /// each reduction slot holds exactly one squared element (matching one CUDA
    /// thread per element). Returns the expected bf16 bit pattern per element.
    fn chain_reference(group: &[bf16]) -> Vec<u16> {
        assert!(group.len() <= 256);
        let mut acc = [0f32; 256];
        for (slot, &v) in group.iter().enumerate() {
            let f = v.to_f32();
            acc[slot] = f * f;
        }
        let mut offset = 128usize;
        while offset > 0 {
            for i in 0..offset {
                acc[i] += acc[i + offset];
            }
            offset >>= 1;
        }
        let sumsq = bf16::from_f32(acc[0]).to_f32(); // ReduceSumSquare store round
        let norm = bf16::from_f32(sumsq.sqrt()).to_f32(); // Sqrt store round
        group
            .iter()
            .map(|&v| bf16::from_f32(v.to_f32() / norm).to_bits()) // Div store round
            .collect()
    }

    fn run_kernel(
        runtime: &std::sync::Arc<CudaRuntime>,
        faithful: bool,
        groups: usize,
        axis_length: usize,
        data: &[bf16],
    ) -> Vec<u16> {
        let bytes = std::mem::size_of_val(data);
        let in_dev = runtime.alloc_raw(bytes).unwrap();
        let out_dev = runtime.alloc_raw(bytes).unwrap();
        let as_bytes = |v: &[bf16]| unsafe {
            std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v))
        };
        unsafe { runtime.htod(as_bytes(data), in_dev).unwrap() };
        let device = DeviceId::cuda(0);
        let shape = [groups, axis_length];
        let strides = [axis_length as i64, 1];
        let inputs = [TensorView::new(
            DevicePtr(in_dev as usize as *const c_void),
            DataType::BFloat16,
            &shape,
            &strides,
            device,
        )];
        let mut outputs = [TensorMut::new(
            DevicePtrMut(out_dev as usize as *mut c_void),
            DataType::BFloat16,
            &shape,
            &strides,
            device,
        )];
        LpNormalizationKernel {
            axis: -1,
            p: 2,
            faithful_chain: faithful,
            runtime: runtime.clone(),
        }
        .execute(&inputs, &mut outputs)
        .unwrap();
        runtime.synchronize().unwrap();
        let mut host = vec![bf16::ZERO; data.len()];
        unsafe {
            runtime
                .dtoh(
                    std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), bytes),
                    out_dev,
                )
                .unwrap();
        }
        unsafe {
            runtime.free_raw(in_dev).unwrap();
            runtime.free_raw(out_dev).unwrap();
        }
        host.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn faithful_l2_normalize_matches_reduce_sqrt_div_chain_bit_exactly() {
        let Some(runtime) = crate::test_support::maybe_runtime() else {
            eprintln!("skipping faithful L2-normalize parity test: CUDA runtime unavailable");
            return;
        };
        if runtime.require_nvrtc_half_headers("LpNormalization").is_err() {
            eprintln!("skipping faithful L2-normalize parity test: bf16 headers unavailable");
            return;
        }
        // Cover the full 256-slot tree, a non-power-of-two length, and a small
        // head_dim, over several groups — the Gated-DeltaNet Q/K norm shape.
        for &axis_length in &[256usize, 170, 128, 96] {
            let groups = 5usize;
            let data = sample(groups * axis_length, axis_length as u32 + 7);
            let got = run_kernel(&runtime, true, groups, axis_length, &data);
            let mut expected = Vec::with_capacity(data.len());
            for g in 0..groups {
                expected.extend(chain_reference(&data[g * axis_length..(g + 1) * axis_length]));
            }
            assert_eq!(
                got, expected,
                "faithful L2-normalize (axis_length={axis_length}) must be bit-identical \
                 to the exported ReduceSumSquare->Sqrt->Div chain"
            );
        }
    }

    #[test]
    fn faithful_kernel_differs_from_fp32_lp_normalization() {
        let Some(runtime) = crate::test_support::maybe_runtime() else {
            eprintln!("skipping faithful-vs-fp32 divergence test: CUDA runtime unavailable");
            return;
        };
        if runtime.require_nvrtc_half_headers("LpNormalization").is_err() {
            eprintln!("skipping faithful-vs-fp32 divergence test: bf16 headers unavailable");
            return;
        }
        // The whole point of the faithful path: it reproduces the exported chain's
        // intermediate bf16 rounds, so it must diverge from the strictly-more-
        // accurate fp32-throughout `lp_normalization` on at least some inputs.
        let (groups, axis_length) = (8usize, 256usize);
        let data = sample(groups * axis_length, 99);
        let faithful = run_kernel(&runtime, true, groups, axis_length, &data);
        let fp32 = run_kernel(&runtime, false, groups, axis_length, &data);
        assert_ne!(
            faithful, fp32,
            "faithful chain-replica must differ from the fp32-throughout kernel \
             (otherwise the token-identity guarantee is vacuous)"
        );
    }
}
