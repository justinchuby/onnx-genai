//! ONNX `MaxPool` and `AveragePool` via cuDNN pooling forward.

use std::ffi::c_void;
use std::sync::Arc;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::cudnn::{CudnnBufferPair, CudnnPoolingMode, CudnnPoolingSpec, CudnnTensorType};
use crate::error::not_implemented;
use crate::runtime::{cuptr, CudaRuntime};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolKind {
    Max,
    Average,
}

pub struct PoolFactory {
    pub kind: PoolKind,
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for PoolFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let op = self.kind.name();
        if self.kind == PoolKind::Max && node.outputs.len() > 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep unsupported: MaxPool Indices output; cuDNN pooling forward does not produce ONNX flat indices"
                    .into(),
            ));
        }
        let kernel_shape = ints_attr(node, op, "kernel_shape", None)?;
        let strides = ints_attr(node, op, "strides", Some(&[1, 1]))?;
        let pads = ints_attr(node, op, "pads", Some(&[0, 0, 0, 0]))?;
        let ceil_mode = int_attr(node, op, "ceil_mode", 0)?;
        if ceil_mode != 0 {
            return Err(not_implemented(format!(
                "{op} ceil_mode=1 (cuDNN pooling path supports ceil_mode=0 only)"
            )));
        }
        let dilations = ints_attr(node, op, "dilations", Some(&[1, 1]))?;
        if dilations != [1, 1] {
            return Err(not_implemented(format!(
                "{op} dilations={dilations:?} (cuDNN pooling descriptor has no dilation)"
            )));
        }
        if self.kind == PoolKind::Max {
            let storage_order = int_attr(node, op, "storage_order", 0)?;
            if storage_order != 0 {
                return Err(not_implemented(format!(
                    "MaxPool storage_order={storage_order} (only row-major storage_order=0 is supported)"
                )));
            }
        }
        let count_include_pad = match int_attr(node, op, "count_include_pad", 0)? {
            0 => false,
            1 => true,
            value => {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep {op}: count_include_pad must be 0 or 1, got {value}"
                )));
            }
        };
        let auto_pad = node
            .attr("auto_pad")
            .map(|attr| {
                attr.as_str().ok_or_else(|| {
                    EpError::KernelFailed(format!("cuda_ep {op}: auto_pad must be a UTF-8 string"))
                })
            })
            .transpose()?
            .unwrap_or("NOTSET")
            .to_owned();

        Ok(Box::new(PoolKernel {
            runtime: self.runtime.clone(),
            kind: self.kind,
            kernel_shape,
            strides,
            pads,
            auto_pad,
            count_include_pad,
        }))
    }
}

impl PoolKind {
    fn name(self) -> &'static str {
        match self {
            Self::Max => "MaxPool",
            Self::Average => "AveragePool",
        }
    }
}

fn ints_attr(node: &Node, op: &str, name: &str, default: Option<&[i64]>) -> Result<Vec<i64>> {
    match node.attr(name) {
        Some(value) => value.as_ints().map(ToOwned::to_owned).ok_or_else(|| {
            EpError::KernelFailed(format!("cuda_ep {op}: {name} must be an integer list"))
        }),
        None => default.map(ToOwned::to_owned).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep {op}: required attribute {name} is missing"
            ))
        }),
    }
}

fn int_attr(node: &Node, op: &str, name: &str, default: i64) -> Result<i64> {
    match node.attr(name) {
        Some(value) => value.as_int().ok_or_else(|| {
            EpError::KernelFailed(format!("cuda_ep {op}: {name} must be an integer"))
        }),
        None => Ok(default),
    }
}

#[derive(Debug)]
pub struct PoolKernel {
    runtime: Arc<CudaRuntime>,
    kind: PoolKind,
    kernel_shape: Vec<i64>,
    strides: Vec<i64>,
    pads: Vec<i64>,
    auto_pad: String,
    count_include_pad: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PoolPlan {
    output_shape: [usize; 4],
    window: [usize; 2],
    pads: [usize; 2],
    strides: [usize; 2],
}

impl PoolKernel {
    fn plan(&self, input: &[usize]) -> Result<PoolPlan> {
        let op = self.kind.name();
        if input.len() != 4 {
            return Err(not_implemented(format!(
                "{op} with input rank {} (cuDNN path supports 2-D NCHW only)",
                input.len()
            )));
        }
        let window = pair(op, "kernel_shape", &self.kernel_shape)?;
        let strides = pair(op, "strides", &self.strides)?;
        let (begin, end, same_output) = match self.auto_pad.as_str() {
            "" | "NOTSET" => {
                if self.pads.len() != 4 {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep {op}: pads must have 4 values [top,left,bottom,right], got {:?}",
                        self.pads
                    )));
                }
                let pads = self
                    .pads
                    .iter()
                    .map(|&value| {
                        usize::try_from(value).map_err(|_| {
                            EpError::KernelFailed(format!(
                                "cuda_ep {op}: pads must be non-negative, got {:?}",
                                self.pads
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                ([pads[0], pads[1]], [pads[2], pads[3]], None)
            }
            "VALID" => ([0, 0], [0, 0], None),
            "SAME_UPPER" | "SAME_LOWER" => {
                let mut begin = [0; 2];
                let mut end = [0; 2];
                let mut output = [0; 2];
                for axis in 0..2 {
                    output[axis] = input[axis + 2].div_ceil(strides[axis]);
                    let total = output[axis]
                        .saturating_sub(1)
                        .saturating_mul(strides[axis])
                        .saturating_add(window[axis])
                        .saturating_sub(input[axis + 2]);
                    begin[axis] = if self.auto_pad == "SAME_LOWER" {
                        total.div_ceil(2)
                    } else {
                        total / 2
                    };
                    end[axis] = total - begin[axis];
                }
                (begin, end, Some(output))
            }
            other => {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep {op}: unsupported auto_pad value {other:?}; expected NOTSET, VALID, SAME_UPPER, or SAME_LOWER"
                )));
            }
        };
        if begin != end {
            return Err(not_implemented(format!(
                "{op} with asymmetric pads [top={}, left={}, bottom={}, right={}] (cuDNN pooling requires symmetric padding)",
                begin[0], begin[1], end[0], end[1]
            )));
        }

        let spatial = if let Some(output) = same_output {
            output
        } else {
            let mut output = [0; 2];
            for axis in 0..2 {
                let padded = input[axis + 2]
                    .checked_add(begin[axis].saturating_mul(2))
                    .ok_or_else(|| {
                        EpError::KernelFailed(format!("cuda_ep {op}: padded size overflow"))
                    })?;
                if padded < window[axis] {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep {op}: kernel {} exceeds padded input {} on spatial axis {axis}",
                        window[axis], padded
                    )));
                }
                output[axis] = (padded - window[axis]) / strides[axis] + 1;
            }
            output
        };
        Ok(PoolPlan {
            output_shape: [input[0], input[1], spatial[0], spatial[1]],
            window,
            pads: begin,
            strides,
        })
    }

    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let op = self.kind.name();
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: expected one input and one output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
        let input = &inputs[0];
        let output = &mut outputs[0];
        if !matches!(
            input.dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            return Err(not_implemented(format!(
                "{op} with dtype {:?} (cuDNN path supports f32/f16/bf16)",
                input.dtype
            )));
        }
        if output.dtype != input.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: input and output dtypes must match ({:?} vs {:?})",
                input.dtype, output.dtype
            )));
        }
        if !input.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented(format!(
                "{op} with non-contiguous input or output; materialise the tensor first"
            )));
        }
        let plan = self.plan(input.shape)?;
        if output.shape != plan.output_shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output shape {:?}, expected {:?}",
                output.shape, plan.output_shape
            )));
        }
        if output.numel() == 0 {
            return Ok(());
        }
        let mode = match self.kind {
            PoolKind::Max => CudnnPoolingMode::Max,
            PoolKind::Average if self.count_include_pad => CudnnPoolingMode::AverageIncludePadding,
            PoolKind::Average => CudnnPoolingMode::AverageExcludePadding,
        };
        let spec = CudnnPoolingSpec {
            dtype: CudnnTensorType::from_onnx(input.dtype)?,
            input_dims: dims4(input.shape, op, "input")?,
            input_strides: strides4(input.strides, op, "input")?,
            output_dims: dims4(output.shape, op, "output")?,
            output_strides: strides4(output.strides, op, "output")?,
            window: i32_pair(plan.window, op, "kernel_shape")?,
            pads: i32_pair(plan.pads, op, "pads")?,
            strides: i32_pair(plan.strides, op, "strides")?,
            mode,
        };
        let buffers = CudnnBufferPair {
            input: cuptr(input.data_ptr::<u8>() as *const c_void),
            output: cuptr(output.data_ptr_mut::<u8>() as *const c_void),
            input_numel: input.numel(),
            output_numel: output.numel(),
        };
        self.runtime
            .cudnn()
            .with_handle(|handle| handle.pool2d(&spec, buffers))?;
        self.runtime.synchronize()
    }
}

fn pair(op: &str, name: &str, values: &[i64]) -> Result<[usize; 2]> {
    if values.len() != 2 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep {op}: {name} must have 2 values, got {values:?}"
        )));
    }
    let mut out = [0; 2];
    for (index, &value) in values.iter().enumerate() {
        out[index] = usize::try_from(value)
            .ok()
            .filter(|&value| value > 0)
            .ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "cuda_ep {op}: {name} values must be positive, got {values:?}"
                ))
            })?;
    }
    Ok(out)
}

fn dims4(shape: &[usize], op: &str, name: &str) -> Result<[i32; 4]> {
    shape
        .iter()
        .map(|&value| {
            i32::try_from(value).map_err(|_| {
                EpError::KernelFailed(format!(
                    "cuda_ep {op}: {name} dimension {value} exceeds i32"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| EpError::KernelFailed(format!("cuda_ep {op}: {name} must be rank 4")))
}

fn strides4(strides: &[i64], op: &str, name: &str) -> Result<[i32; 4]> {
    strides
        .iter()
        .map(|&value| {
            i32::try_from(value).map_err(|_| {
                EpError::KernelFailed(format!("cuda_ep {op}: {name} stride {value} exceeds i32"))
            })
        })
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| EpError::KernelFailed(format!("cuda_ep {op}: {name} must be rank 4")))
}

fn i32_pair(values: [usize; 2], op: &str, name: &str) -> Result<[i32; 2]> {
    Ok([
        i32::try_from(values[0])
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep {op}: {name} exceeds i32")))?,
        i32::try_from(values[1])
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep {op}: {name} exceeds i32")))?,
    ])
}

impl Kernel for PoolKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }

    fn cuda_graph_compatible(&self) -> bool {
        false
    }
}
