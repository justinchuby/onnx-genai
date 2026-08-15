//! ONNX `InstanceNormalization` and `GroupNormalization`.

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

template <typename T> __device__ float load_norm(T value);
template <> __device__ float load_norm<float>(float value) { return value; }
template <> __device__ float load_norm<__half>(__half value) { return __half2float(value); }
template <> __device__ float load_norm<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}
template <typename T> __device__ T store_norm(float value);
template <> __device__ float store_norm<float>(float value) { return value; }
template <> __device__ __half store_norm<__half>(float value) {
  return __float2half_rn(value);
}
template <> __device__ __nv_bfloat16 store_norm<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename T>
__device__ void normalize_group(
    const T* x, const T* scale, const T* bias, T* y,
    unsigned long long group_size, unsigned long long spatial,
    unsigned long long channels_per_group, unsigned long long num_groups,
    int per_channel, float epsilon) {
  __shared__ float sums[256];
  __shared__ float squares[256];
  const unsigned long long flat_group = blockIdx.x;
  const unsigned long long base = flat_group * group_size;
  float sum = 0.0f;
  for (unsigned long long offset = threadIdx.x; offset < group_size; offset += blockDim.x) {
    sum += load_norm<T>(x[base + offset]);
  }
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sums[threadIdx.x] += sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = sums[0] / (float)group_size;
  float square = 0.0f;
  for (unsigned long long offset = threadIdx.x; offset < group_size; offset += blockDim.x) {
    const float centered = load_norm<T>(x[base + offset]) - mean;
    square += centered * centered;
  }
  squares[threadIdx.x] = square;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      squares[threadIdx.x] += squares[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float variance = squares[0] / (float)group_size;
  const float inverse_std = rsqrtf(variance + epsilon);
  const unsigned long long group = flat_group % num_groups;
  for (unsigned long long offset = threadIdx.x; offset < group_size; offset += blockDim.x) {
    const unsigned long long channel_in_group = offset / spatial;
    const unsigned long long affine =
        per_channel ? group * channels_per_group + channel_in_group : group;
    const float value = (load_norm<T>(x[base + offset]) - mean) * inverse_std;
    y[base + offset] =
        store_norm<T>(value * load_norm<T>(scale[affine]) + load_norm<T>(bias[affine]));
  }
}

#define DEFINE_NORMALIZE_GROUP(TYPE, SUFFIX) \
extern "C" __global__ void normalize_group_##SUFFIX( \
    const TYPE* x, const TYPE* scale, const TYPE* bias, TYPE* y, \
    unsigned long long group_size, unsigned long long spatial, \
    unsigned long long channels_per_group, unsigned long long num_groups, \
    int per_channel, float epsilon) { \
  normalize_group<TYPE>(x, scale, bias, y, group_size, spatial, \
                        channels_per_group, num_groups, per_channel, epsilon); \
}

DEFINE_NORMALIZE_GROUP(float, f32)
DEFINE_NORMALIZE_GROUP(__half, f16)
DEFINE_NORMALIZE_GROUP(__nv_bfloat16, bf16)
"#;

#[derive(Clone, Copy, Debug)]
enum Affine {
    PerGroup,
    PerChannel,
}

pub struct InstanceNormalizationFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for InstanceNormalizationFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(GroupNormalizationKernel {
            runtime: self.runtime.clone(),
            groups: None,
            epsilon: node
                .attr("epsilon")
                .and_then(|attribute| attribute.as_float())
                .unwrap_or(1e-5),
            affine: Affine::PerChannel,
        }))
    }
}

pub struct GroupNormalizationFactory {
    pub runtime: Arc<CudaRuntime>,
    pub since_version: u64,
}

impl KernelFactory for GroupNormalizationFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let (groups, epsilon, affine) = parse_group_attributes(node, self.since_version)?;
        Ok(Box::new(GroupNormalizationKernel {
            runtime: self.runtime.clone(),
            groups: Some(groups),
            epsilon,
            affine,
        }))
    }
}

fn parse_group_attributes(node: &Node, since_version: u64) -> Result<(usize, f32, Affine)> {
    let groups = node
        .attr("num_groups")
        .and_then(|attribute| attribute.as_int())
        .ok_or_else(|| {
            EpError::KernelFailed(
                "cuda_ep GroupNormalization: required num_groups attribute is missing".into(),
            )
        })?;
    let groups = usize::try_from(groups)
        .ok()
        .filter(|&value| value != 0)
        .ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep GroupNormalization: num_groups must be positive, got {groups}"
            ))
        })?;
    let stash_type = node
        .attr("stash_type")
        .and_then(|attribute| attribute.as_int())
        .unwrap_or(1);
    if since_version >= 21 && stash_type != 1 {
        return Err(not_implemented(format!(
            "GroupNormalization stash_type={stash_type} (only float stash_type=1 is supported)"
        )));
    }
    Ok((
        groups,
        node.attr("epsilon")
            .and_then(|attribute| attribute.as_float())
            .unwrap_or(1e-5),
        if since_version >= 21 {
            Affine::PerChannel
        } else {
            Affine::PerGroup
        },
    ))
}

struct GroupNormalizationKernel {
    runtime: Arc<CudaRuntime>,
    groups: Option<usize>,
    epsilon: f32,
    affine: Affine,
}

impl Kernel for GroupNormalizationKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let op = if self.groups.is_some() {
            "GroupNormalization"
        } else {
            "InstanceNormalization"
        };
        if inputs.len() != 3 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: expected 3 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let x = &inputs[0];
        if x.shape.len() < 3 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: X must have rank at least 3"
            )));
        }
        if !matches!(
            x.dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            return Err(not_implemented(format!(
                "{op} dtype {:?} (supported: Float32, Float16, BFloat16)",
                x.dtype
            )));
        }
        if inputs
            .iter()
            .any(|input| input.dtype != x.dtype || !input.is_contiguous())
            || outputs[0].dtype != x.dtype
            || !outputs[0].is_contiguous()
        {
            return Err(not_implemented(format!(
                "{op} requires contiguous, same-dtype tensors"
            )));
        }
        if outputs[0].shape != x.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output shape must match X"
            )));
        }
        let channels = x.shape[1];
        let spatial = x.shape[2..].iter().product::<usize>();
        let groups = self.groups.unwrap_or(channels);
        if channels == 0 || spatial == 0 || !channels.is_multiple_of(groups) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: channels ({channels}) must be non-zero and divisible by groups ({groups}); spatial dimensions must be non-empty"
            )));
        }
        let affine_length = match self.affine {
            Affine::PerGroup => groups,
            Affine::PerChannel => channels,
        };
        for (name, input) in [("scale", &inputs[1]), ("bias", &inputs[2])] {
            if input.shape != [affine_length] {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep {op}: {name} must have shape [{affine_length}], got {:?}",
                    input.shape
                )));
            }
        }

        let instances = x.shape[0];
        let flat_groups = instances
            .checked_mul(groups)
            .ok_or_else(|| EpError::KernelFailed(format!("cuda_ep {op}: group count overflow")))?;
        if flat_groups == 0 {
            return Ok(());
        }
        let grid = u32::try_from(flat_groups).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep {op}: {flat_groups} groups exceed CUDA grid"
            ))
        })?;
        let entry = match x.dtype {
            DataType::Float32 => "normalize_group_f32",
            DataType::Float16 => "normalize_group_f16",
            DataType::BFloat16 => "normalize_group_bf16",
            _ => unreachable!(),
        };
        if x.dtype != DataType::Float32 {
            self.runtime.require_nvrtc_half_headers(op)?;
        }
        let function = self
            .runtime
            .nvrtc_function("group_normalization_v1", SOURCE, entry)?;
        let x_pointer = cuptr(x.data_ptr::<u8>() as *const c_void);
        let scale_pointer = cuptr(inputs[1].data_ptr::<u8>() as *const c_void);
        let bias_pointer = cuptr(inputs[2].data_ptr::<u8>() as *const c_void);
        let output_pointer = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let channels_per_group = channels / groups;
        let group_size = channels_per_group
            .checked_mul(spatial)
            .ok_or_else(|| EpError::KernelFailed(format!("cuda_ep {op}: group size overflow")))?
            as u64;
        let spatial = spatial as u64;
        let channels_per_group = channels_per_group as u64;
        let groups = groups as u64;
        let per_channel = matches!(self.affine, Affine::PerChannel) as i32;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&x_pointer)
            .arg(&scale_pointer)
            .arg(&bias_pointer)
            .arg(&output_pointer)
            .arg(&group_size)
            .arg(&spatial)
            .arg(&channels_per_group)
            .arg(&groups)
            .arg(&per_channel)
            .arg(&self.epsilon);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| driver_err(&format!("launch {op}"), error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, Node, NodeId};

    #[test]
    fn group_normalization_requires_positive_group_count() {
        let node = Node::new(NodeId(0), "GroupNormalization", vec![], vec![]);
        assert!(parse_group_attributes(&node, 18).is_err());

        let mut node = node;
        node.attributes
            .insert("num_groups".into(), Attribute::Int(0));
        assert!(parse_group_attributes(&node, 18).is_err());
    }

    #[test]
    fn opset_21_rejects_non_float_stash_type() {
        let mut node = Node::new(NodeId(0), "GroupNormalization", vec![], vec![]);
        node.attributes
            .insert("num_groups".into(), Attribute::Int(2));
        node.attributes
            .insert("stash_type".into(), Attribute::Int(10));
        assert!(parse_group_attributes(&node, 21).is_err());
        assert!(parse_group_attributes(&node, 18).is_ok());
    }
}
