//! Common 1-D/2-D ONNX `ConvTranspose` using deterministic output-owned NVRTC work.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float load_value(
    const void* values, int dtype, unsigned long long index) {
  if (dtype == 0) return ((const float*)values)[index];
  if (dtype == 1) return __half2float(((const __half*)values)[index]);
  return __bfloat162float(((const __nv_bfloat16*)values)[index]);
}

__device__ __forceinline__ void store_value(
    void* values, int dtype, unsigned long long index, float value) {
  if (dtype == 0) ((float*)values)[index] = value;
  else if (dtype == 1) ((__half*)values)[index] = __float2half_rn(value);
  else ((__nv_bfloat16*)values)[index] = __float2bfloat16_rn(value);
}

extern "C" __global__ void conv_transpose(
    const void* x, const void* weights, const void* bias, void* output,
    unsigned long long output_elements, unsigned long long input_channels,
    unsigned long long input_height, unsigned long long input_width,
    unsigned long long output_channels, unsigned long long output_height,
    unsigned long long output_width, unsigned long long output_channels_per_group,
    unsigned long long input_channels_per_group, unsigned long long kernel_height,
    unsigned long long kernel_width, unsigned long long stride_height,
    unsigned long long stride_width, unsigned long long dilation_height,
    unsigned long long dilation_width, long long pad_height, long long pad_width,
    int dtype, int has_bias) {
  const unsigned long long output_spatial = output_height * output_width;
  const unsigned long long input_spatial = input_height * input_width;
  const unsigned long long kernel_spatial = kernel_height * kernel_width;
  for (unsigned long long output_index =
           blockIdx.x * blockDim.x + threadIdx.x;
       output_index < output_elements;
       output_index += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long batch = output_index / (output_channels * output_spatial);
    const unsigned long long within_batch = output_index % (output_channels * output_spatial);
    const unsigned long long output_channel = within_batch / output_spatial;
    const unsigned long long output_flat = within_batch % output_spatial;
    const unsigned long long output_y = output_flat / output_width;
    const unsigned long long output_x = output_flat % output_width;
    const unsigned long long group = output_channel / output_channels_per_group;
    const unsigned long long output_in_group = output_channel % output_channels_per_group;
    const unsigned long long input_begin = group * input_channels_per_group;
    const unsigned long long input_end = input_begin + input_channels_per_group;
    float accumulated = has_bias ? load_value(bias, dtype, output_channel) : 0.0f;

    // Keep the CPU reference's input-channel/input-position/kernel-position order.
    for (unsigned long long input_channel = input_begin;
         input_channel < input_end; ++input_channel) {
      for (unsigned long long input_flat = 0; input_flat < input_spatial; ++input_flat) {
        const unsigned long long input_y = input_flat / input_width;
        const unsigned long long input_x = input_flat % input_width;
        const float input_value = load_value(
            x, dtype, (batch * input_channels + input_channel) * input_spatial + input_flat);
        for (unsigned long long kernel_flat = 0; kernel_flat < kernel_spatial; ++kernel_flat) {
          const unsigned long long kernel_y = kernel_flat / kernel_width;
          const unsigned long long kernel_x = kernel_flat % kernel_width;
          const long long candidate_y =
              (long long)(input_y * stride_height + kernel_y * dilation_height) - pad_height;
          const long long candidate_x =
              (long long)(input_x * stride_width + kernel_x * dilation_width) - pad_width;
          if (candidate_y != (long long)output_y || candidate_x != (long long)output_x) continue;
          const unsigned long long weight_index =
              (input_channel * output_channels_per_group + output_in_group) * kernel_spatial
              + kernel_flat;
          accumulated += input_value * load_value(weights, dtype, weight_index);
        }
      }
    }
    store_value(output, dtype, output_index, accumulated);
  }
}
"#;

pub struct ConvTransposeFactory {
    pub runtime: Arc<CudaRuntime>,
}

#[derive(Clone)]
struct Parameters {
    dilations: Vec<usize>,
    group: usize,
    output_padding: Vec<usize>,
    pads: Vec<usize>,
    strides: Vec<usize>,
}

impl KernelFactory for ConvTransposeFactory {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        reject_deferred(node)?;
        let rank = input_shapes
            .first()
            .map(|shape| shape.len())
            .filter(|rank| matches!(rank, 3 | 4))
            .ok_or_else(|| not_implemented("ConvTranspose supports only 1-D/2-D inputs"))?
            - 2;
        if input_shapes
            .get(1)
            .is_none_or(|shape| shape.len() != rank + 2)
        {
            return Err(EpError::KernelFailed(
                "cuda_ep ConvTranspose: X and W ranks must match".into(),
            ));
        }
        let weight_shape = &input_shapes[1];
        if let Some(kernel_shape) = node.attr("kernel_shape").and_then(Attribute::as_ints)
            && (kernel_shape.len() != rank
                || kernel_shape.iter().any(|&value| value <= 0)
                || kernel_shape
                    .iter()
                    .zip(&weight_shape[2..])
                    .any(|(&attribute, &weight)| attribute as usize != weight))
        {
            return Err(EpError::KernelFailed(
                "cuda_ep ConvTranspose: kernel_shape must match W spatial dimensions".into(),
            ));
        }
        let strides = positive_attribute(node, "strides", rank)?;
        let dilations = positive_attribute(node, "dilations", rank)?;
        let output_padding = nonnegative_attribute(node, "output_padding", rank, rank)?;
        if output_padding
            .iter()
            .enumerate()
            .any(|(axis, &value)| value >= strides[axis] && value >= dilations[axis])
        {
            return Err(EpError::KernelFailed(
                "cuda_ep ConvTranspose: output_padding must be smaller than stride or dilation"
                    .into(),
            ));
        }
        let group = node.attr("group").and_then(Attribute::as_int).unwrap_or(1);
        let group = usize::try_from(group)
            .ok()
            .filter(|&value| value > 0)
            .ok_or_else(|| {
                EpError::KernelFailed("cuda_ep ConvTranspose: group must be positive".into())
            })?;
        let pads = if node
            .attr("auto_pad")
            .and_then(Attribute::as_str)
            .is_some_and(|value| value == "VALID")
        {
            vec![0; rank * 2]
        } else {
            nonnegative_attribute(node, "pads", rank, rank * 2)?
        };
        Ok(Box::new(ConvTransposeKernel {
            runtime: self.runtime.clone(),
            parameters: Parameters {
                dilations,
                group,
                output_padding,
                pads,
                strides,
            },
        }))
    }
}

pub(crate) fn reject_deferred(node: &Node) -> Result<()> {
    match node.attr("auto_pad").and_then(Attribute::as_str) {
        None | Some("" | "NOTSET" | "VALID") => {}
        Some(value) => {
            return Err(not_implemented(format!(
                "ConvTranspose auto_pad {value:?}; SAME_UPPER/SAME_LOWER are deferred"
            )));
        }
    }
    if node.attr("output_shape").is_some() {
        return Err(not_implemented(
            "ConvTranspose output_shape-driven padding is deferred",
        ));
    }
    Ok(())
}

fn positive_attribute(node: &Node, name: &str, rank: usize) -> Result<Vec<usize>> {
    let values = node
        .attr(name)
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![1; rank]);
    if values.len() != rank || values.iter().any(|&value| value <= 0) {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep ConvTranspose: {name} must contain {rank} positive values"
        )));
    }
    Ok(values.into_iter().map(|value| value as usize).collect())
}

fn nonnegative_attribute(node: &Node, name: &str, rank: usize, count: usize) -> Result<Vec<usize>> {
    let values = node
        .attr(name)
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![0; count]);
    if values.len() != count || values.iter().any(|&value| value < 0) {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep ConvTranspose: {name} must contain {count} non-negative values for rank {rank}"
        )));
    }
    Ok(values.into_iter().map(|value| value as usize).collect())
}

fn dtype_code(dtype: DataType) -> Result<i32> {
    match dtype {
        DataType::Float32 => Ok(0),
        DataType::Float16 => Ok(1),
        DataType::BFloat16 => Ok(2),
        other => Err(not_implemented(format!(
            "ConvTranspose dtype {other:?} (supported: Float32, Float16, BFloat16)"
        ))),
    }
}

struct ConvTransposeKernel {
    runtime: Arc<CudaRuntime>,
    parameters: Parameters,
}

impl ConvTransposeKernel {
    fn output_spatial(&self, input: &[usize], kernel: &[usize]) -> Result<Vec<usize>> {
        let rank = input.len();
        input
            .iter()
            .zip(kernel)
            .enumerate()
            .map(|(axis, (&input, &kernel))| {
                let effective = self.parameters.dilations[axis]
                    .checked_mul(kernel.saturating_sub(1))
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| {
                        EpError::KernelFailed(
                            "cuda_ep ConvTranspose: effective kernel overflow".into(),
                        )
                    })?;
                let base = self.parameters.strides[axis]
                    .checked_mul(input.saturating_sub(1))
                    .and_then(|value| value.checked_add(effective))
                    .and_then(|value| value.checked_add(self.parameters.output_padding[axis]))
                    .ok_or_else(|| {
                        EpError::KernelFailed(
                            "cuda_ep ConvTranspose: output dimension overflow".into(),
                        )
                    })?;
                base.checked_sub(self.parameters.pads[axis] + self.parameters.pads[axis + rank])
                    .ok_or_else(|| {
                        EpError::KernelFailed(
                            "cuda_ep ConvTranspose: pads exceed generated output".into(),
                        )
                    })
            })
            .collect()
    }
}

impl Kernel for ConvTransposeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if !(2..=3).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep ConvTranspose: expected X, W, optional B and one output".into(),
            ));
        }
        if inputs.iter().any(|input| !input.is_contiguous()) || !outputs[0].is_contiguous() {
            return Err(not_implemented("ConvTranspose with strided tensors"));
        }
        let x = &inputs[0];
        let weights = &inputs[1];
        let rank = x.shape.len();
        if !matches!(rank, 3 | 4) || weights.shape.len() != rank {
            return Err(not_implemented(
                "ConvTranspose supports only 1-D/2-D inputs",
            ));
        }
        if x.dtype != weights.dtype || outputs[0].dtype != x.dtype {
            return Err(EpError::KernelFailed(
                "cuda_ep ConvTranspose: X, W, and Y dtypes must match".into(),
            ));
        }
        let dtype = dtype_code(x.dtype)?;
        if dtype != 0 {
            self.runtime.require_nvrtc_half_headers("ConvTranspose")?;
        }
        let group = self.parameters.group;
        if weights.shape[0] != x.shape[1]
            || !x.shape[1].is_multiple_of(group)
            || weights.shape[1] == 0
        {
            return Err(EpError::KernelFailed(
                "cuda_ep ConvTranspose: invalid grouped channel geometry".into(),
            ));
        }
        let output_channels = weights.shape[1]
            .checked_mul(group)
            .ok_or_else(|| EpError::KernelFailed("output channel overflow".into()))?;
        let output_spatial = self.output_spatial(&x.shape[2..], &weights.shape[2..])?;
        let expected = [x.shape[0], output_channels]
            .into_iter()
            .chain(output_spatial.iter().copied())
            .collect::<Vec<_>>();
        if outputs[0].shape != expected {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep ConvTranspose: output shape {:?}, expected {expected:?}",
                outputs[0].shape
            )));
        }
        let bias = inputs.get(2);
        if bias.is_some_and(|value| value.dtype != x.dtype || value.shape != [output_channels]) {
            return Err(EpError::KernelFailed(
                "cuda_ep ConvTranspose: bias must match output channels and dtype".into(),
            ));
        }
        if outputs[0].numel() == 0 {
            return Ok(());
        }
        let (input_height, input_width, output_height, output_width, kernel_height, kernel_width) =
            if rank == 3 {
                (1, x.shape[2], 1, output_spatial[0], 1, weights.shape[2])
            } else {
                (
                    x.shape[2],
                    x.shape[3],
                    output_spatial[0],
                    output_spatial[1],
                    weights.shape[2],
                    weights.shape[3],
                )
            };
        let axis = |values: &[usize], height_default: usize| {
            if rank == 3 {
                (height_default, values[0])
            } else {
                (values[0], values[1])
            }
        };
        let (stride_height, stride_width) = axis(&self.parameters.strides, 1);
        let (dilation_height, dilation_width) = axis(&self.parameters.dilations, 1);
        let (pad_height, pad_width) = axis(&self.parameters.pads, 0);
        let function =
            self.runtime
                .nvrtc_function("conv_transpose_common_v1", SOURCE, "conv_transpose")?;
        let x_pointer = cuptr(x.data_ptr::<u8>() as *const c_void);
        let weights_pointer = cuptr(weights.data_ptr::<u8>() as *const c_void);
        let bias_pointer = bias
            .map(|value| cuptr(value.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_pointer = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let output_elements = outputs[0].numel() as u64;
        let dimensions = [
            x.shape[1],
            input_height,
            input_width,
            output_channels,
            output_height,
            output_width,
            weights.shape[1],
            x.shape[1] / group,
            kernel_height,
            kernel_width,
            stride_height,
            stride_width,
            dilation_height,
            dilation_width,
        ]
        .map(|value| value as u64);
        let pad_height = pad_height as i64;
        let pad_width = pad_width as i64;
        let has_bias = i32::from(bias.is_some());
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&x_pointer)
            .arg(&weights_pointer)
            .arg(&bias_pointer)
            .arg(&output_pointer)
            .arg(&output_elements);
        for dimension in &dimensions {
            builder.arg(dimension);
        }
        builder
            .arg(&pad_height)
            .arg(&pad_width)
            .arg(&dtype)
            .arg(&has_bias);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    output_elements.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32,
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err("launch ConvTranspose", error))?;
        if self.runtime.is_capturing()? {
            return Ok(());
        }
        self.runtime.synchronize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{NodeId, ValueId};

    fn node(name: &str, attribute: Attribute) -> Node {
        let mut node = Node::new(
            NodeId(0),
            "ConvTranspose",
            vec![Some(ValueId(0)), Some(ValueId(1))],
            vec![ValueId(2)],
        );
        node.attributes.insert(name.into(), attribute);
        node
    }

    #[test]
    fn deferred_geometry_is_rejected_by_claim_and_factory_contract() {
        for deferred in [
            node("auto_pad", Attribute::String(b"SAME_UPPER".to_vec())),
            node("auto_pad", Attribute::String(b"SAME_LOWER".to_vec())),
            node("output_shape", Attribute::Ints(vec![4, 4])),
        ] {
            assert!(
                crate::kernels::standard_claims::unsupported_reason(
                    &deferred,
                    &[vec![1.into(), 1.into(), 2.into(), 2.into()]],
                    &[DataType::Float32, DataType::Float32],
                )
                .is_some()
            );
            assert!(reject_deferred(&deferred).is_err());
        }
    }

    #[test]
    fn three_dimensional_input_is_rejected_by_claim_gate() {
        let node = Node::new(
            NodeId(0),
            "ConvTranspose",
            vec![Some(ValueId(0)), Some(ValueId(1))],
            vec![ValueId(2)],
        );
        let reason = crate::kernels::standard_claims::unsupported_reason(
            &node,
            &[
                vec![1.into(), 1.into(), 2.into(), 2.into(), 2.into()],
                vec![1.into(), 1.into(), 2.into(), 2.into(), 2.into()],
            ],
            &[DataType::Float32, DataType::Float32],
        )
        .expect("rank-5 ConvTranspose must be declined before CUDA placement");
        assert!(reason.contains("rank 5 unsupported"), "{reason}");
    }
}
