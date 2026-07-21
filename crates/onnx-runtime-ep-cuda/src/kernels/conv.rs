//! ONNX `Conv` via cuDNN's forward-convolution API.
//!
//! This path covers dense 2-D NCHW f32/f16/bf16 convolution, including strides,
//! dilations, groups, symmetric padding, and optional channel bias. cuDNN's
//! legacy forward API only accepts symmetric padding; asymmetric ONNX padding is
//! rejected explicitly rather than silently changing the result.

use std::ffi::c_void;
use std::sync::Arc;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::cudnn::{CudnnConvBuffers, CudnnConvSpec, CudnnTensorType};
use crate::error::not_implemented;
use crate::runtime::{CudaRuntime, cuptr};

pub struct ConvFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for ConvFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ConvKernel {
            runtime: self.runtime.clone(),
            strides: ints_attr(node, "strides", &[1, 1])?,
            pads: ints_attr(node, "pads", &[0, 0, 0, 0])?,
            dilations: ints_attr(node, "dilations", &[1, 1])?,
            group: match node.attr("group") {
                Some(value) => value.as_int().ok_or_else(|| {
                    EpError::KernelFailed("cuda_ep Conv: group must be an integer".into())
                })?,
                None => 1,
            },
            auto_pad: node
                .attr("auto_pad")
                .map(|a| {
                    a.as_str().ok_or_else(|| {
                        EpError::KernelFailed(
                            "cuda_ep Conv: auto_pad must be a UTF-8 string".into(),
                        )
                    })
                })
                .transpose()?
                .unwrap_or("NOTSET")
                .to_owned(),
        }))
    }
}

fn ints_attr(node: &Node, name: &str, default: &[i64]) -> Result<Vec<i64>> {
    match node.attr(name) {
        Some(value) => value.as_ints().map(ToOwned::to_owned).ok_or_else(|| {
            EpError::KernelFailed(format!("cuda_ep Conv: {name} must be an integer list"))
        }),
        None => Ok(default.to_vec()),
    }
}

#[derive(Debug)]
pub struct ConvKernel {
    runtime: Arc<CudaRuntime>,
    strides: Vec<i64>,
    pads: Vec<i64>,
    dilations: Vec<i64>,
    group: i64,
    auto_pad: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ConvPlan {
    output_shape: [usize; 4],
    pads: [usize; 2],
    strides: [usize; 2],
    dilations: [usize; 2],
    groups: usize,
}

impl ConvKernel {
    fn plan(&self, x: &[usize], w: &[usize]) -> Result<ConvPlan> {
        if x.len() != 4 || w.len() != 4 {
            return Err(not_implemented(format!(
                "Conv with input rank {} and filter rank {} (cuDNN path supports 2-D NCHW only)",
                x.len(),
                w.len()
            )));
        }
        let strides = pair("strides", &self.strides, false)?;
        let dilations = pair("dilations", &self.dilations, false)?;
        let groups = usize::try_from(self.group)
            .ok()
            .filter(|&v| v > 0)
            .ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "cuda_ep Conv: group must be positive, got {}",
                    self.group
                ))
            })?;
        if x[1] != w[1].saturating_mul(groups) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Conv: input channels {} must equal filter channels {} × group {groups}",
                x[1], w[1]
            )));
        }
        if w[0] % groups != 0 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Conv: output channels {} must be divisible by group {groups}",
                w[0]
            )));
        }

        let effective = [
            dilations[0]
                .checked_mul(w[2].saturating_sub(1))
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| {
                    EpError::KernelFailed("cuda_ep Conv: kernel height overflow".into())
                })?,
            dilations[1]
                .checked_mul(w[3].saturating_sub(1))
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| {
                    EpError::KernelFailed("cuda_ep Conv: kernel width overflow".into())
                })?,
        ];
        let (begin, end) = match self.auto_pad.as_str() {
            "" | "NOTSET" => {
                if self.pads.len() != 4 {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep Conv: pads must have 4 values [top,left,bottom,right], got {:?}",
                        self.pads
                    )));
                }
                let values = self
                    .pads
                    .iter()
                    .map(|&v| {
                        usize::try_from(v).map_err(|_| {
                            EpError::KernelFailed(format!(
                                "cuda_ep Conv: pads must be non-negative, got {:?}",
                                self.pads
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                ([values[0], values[1]], [values[2], values[3]])
            }
            "VALID" => ([0, 0], [0, 0]),
            "SAME_UPPER" | "SAME_LOWER" => {
                let mut begin = [0; 2];
                let mut end = [0; 2];
                for axis in 0..2 {
                    let input = x[axis + 2];
                    let output = input.div_ceil(strides[axis]);
                    let total = output
                        .saturating_sub(1)
                        .saturating_mul(strides[axis])
                        .saturating_add(effective[axis])
                        .saturating_sub(input);
                    if self.auto_pad == "SAME_LOWER" {
                        begin[axis] = total.div_ceil(2);
                    } else {
                        begin[axis] = total / 2;
                    }
                    end[axis] = total - begin[axis];
                }
                (begin, end)
            }
            other => {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Conv: unsupported auto_pad value {other:?}; expected NOTSET, VALID, \
                     SAME_UPPER, or SAME_LOWER"
                )));
            }
        };
        if begin != end {
            return Err(not_implemented(format!(
                "Conv with asymmetric pads [top={}, left={}, bottom={}, right={}] (cuDNN legacy \
                 forward API requires symmetric padding)",
                begin[0], begin[1], end[0], end[1]
            )));
        }

        let mut spatial = [0; 2];
        for axis in 0..2 {
            let padded = x[axis + 2]
                .checked_add(begin[axis].saturating_mul(2))
                .ok_or_else(|| {
                    EpError::KernelFailed("cuda_ep Conv: padded size overflow".into())
                })?;
            if padded < effective[axis] {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Conv: effective kernel {} exceeds padded input {} on spatial axis {axis}",
                    effective[axis], padded
                )));
            }
            spatial[axis] = (padded - effective[axis]) / strides[axis] + 1;
        }

        Ok(ConvPlan {
            output_shape: [x[0], w[0], spatial[0], spatial[1]],
            pads: begin,
            strides,
            dilations,
            groups,
        })
    }

    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if !(2..=3).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Conv: expected X, W, optional B and one output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
        let x = &inputs[0];
        let w = &inputs[1];
        let bias = inputs.get(2).filter(|b| !b.is_absent());
        if !matches!(
            x.dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            return Err(not_implemented(format!(
                "Conv with dtype {:?} (cuDNN path supports f32/f16/bf16)",
                x.dtype
            )));
        }
        if w.dtype != x.dtype
            || outputs[0].dtype != x.dtype
            || bias.is_some_and(|b| b.dtype != x.dtype)
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Conv: X, W, B, and Y dtypes must match (X={:?}, W={:?}, B={:?}, Y={:?})",
                x.dtype,
                w.dtype,
                bias.map(|b| b.dtype),
                outputs[0].dtype
            )));
        }
        if !x.is_contiguous()
            || !w.is_contiguous()
            || !outputs[0].is_contiguous()
            || bias.is_some_and(|b| !b.is_contiguous())
        {
            return Err(not_implemented(
                "Conv with non-contiguous X, W, B, or Y; materialise the tensor first",
            ));
        }

        let plan = self.plan(x.shape, w.shape)?;
        if outputs[0].shape != plan.output_shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Conv: output shape {:?}, expected {:?}",
                outputs[0].shape, plan.output_shape
            )));
        }
        if let Some(b) = bias
            && b.shape != [w.shape[0]]
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Conv: bias shape {:?}, expected [{}]",
                b.shape, w.shape[0]
            )));
        }
        if outputs[0].numel() == 0 {
            return Ok(());
        }

        let spec = CudnnConvSpec {
            dtype: CudnnTensorType::from_onnx(x.dtype)?,
            input_dims: dims4(x.shape, "input")?,
            input_strides: strides4(x.strides, "input")?,
            filter_dims: dims4(w.shape, "filter")?,
            output_dims: dims4(outputs[0].shape, "output")?,
            output_strides: strides4(outputs[0].strides, "output")?,
            pads: i32_pair(plan.pads, "pads")?,
            strides: i32_pair(plan.strides, "strides")?,
            dilations: i32_pair(plan.dilations, "dilations")?,
            groups: i32::try_from(plan.groups)
                .map_err(|_| EpError::KernelFailed("cuda_ep Conv: group exceeds i32".into()))?,
        };
        let buffers = CudnnConvBuffers {
            input: cuptr(x.data_ptr::<u8>() as *const c_void),
            filter: cuptr(w.data_ptr::<u8>() as *const c_void),
            bias: bias.map(|b| cuptr(b.data_ptr::<u8>() as *const c_void)),
            output: cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void),
            input_numel: x.numel(),
            filter_numel: w.numel(),
            bias_numel: bias.map_or(0, TensorView::numel),
            output_numel: outputs[0].numel(),
        };
        self.runtime
            .cudnn()
            .with_handle(|handle| handle.conv2d(&spec, buffers))?;
        self.runtime.synchronize()
    }
}

fn pair(name: &str, values: &[i64], allow_zero: bool) -> Result<[usize; 2]> {
    if values.len() != 2 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep Conv: {name} must have 2 values, got {values:?}"
        )));
    }
    let mut out = [0; 2];
    for (index, &value) in values.iter().enumerate() {
        out[index] = usize::try_from(value).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep Conv: {name} values must be non-negative, got {values:?}"
            ))
        })?;
        if !allow_zero && out[index] == 0 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Conv: {name} values must be positive, got {values:?}"
            )));
        }
    }
    Ok(out)
}

fn dims4(shape: &[usize], name: &str) -> Result<[i32; 4]> {
    shape
        .iter()
        .map(|&v| {
            i32::try_from(v).map_err(|_| {
                EpError::KernelFailed(format!("cuda_ep Conv: {name} dimension {v} exceeds i32"))
            })
        })
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| EpError::KernelFailed(format!("cuda_ep Conv: {name} must be rank 4")))
}

fn strides4(strides: &[i64], name: &str) -> Result<[i32; 4]> {
    strides
        .iter()
        .map(|&v| {
            i32::try_from(v).map_err(|_| {
                EpError::KernelFailed(format!("cuda_ep Conv: {name} stride {v} exceeds i32"))
            })
        })
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| EpError::KernelFailed(format!("cuda_ep Conv: {name} must be rank 4")))
}

fn i32_pair(values: [usize; 2], name: &str) -> Result<[i32; 2]> {
    Ok([
        i32::try_from(values[0])
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep Conv: {name} exceeds i32")))?,
        i32::try_from(values[1])
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep Conv: {name} exceeds i32")))?,
    ])
}

impl Kernel for ConvKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "convolution creates per-call cuDNN descriptors and performs a trailing host stream synchronization",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_spatial_attribute_pairs() {
        assert_eq!(pair("strides", &[2, 3], false).unwrap(), [2, 3]);
        assert!(pair("strides", &[0, 1], false).is_err());
        assert!(pair("dilations", &[1], false).is_err());
    }
}
