//! ONNX pooling: cuDNN `MaxPool`/`AveragePool` and NVRTC `LpPool`.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg, sys::CUdeviceptr};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::cudnn::{CudnnBufferPair, CudnnPoolingMode, CudnnPoolingSpec, CudnnTensorType};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const LP_SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float load_pool_value(
    const void* values, int dtype, unsigned long long index) {
  if (dtype == 0) return ((const float*)values)[index];
  if (dtype == 1) return __half2float(((const __half*)values)[index]);
  return __bfloat162float(((const __nv_bfloat16*)values)[index]);
}

__device__ __forceinline__ void store_pool_value(
    void* values, int dtype, unsigned long long index, float value) {
  if (dtype == 0) ((float*)values)[index] = value;
  else if (dtype == 1) ((__half*)values)[index] = __float2half_rn(value);
  else ((__nv_bfloat16*)values)[index] = __float2bfloat16_rn(value);
}

extern "C" __global__ void lp_pool(
    const void* input, void* output, const unsigned long long* metadata,
    int spatial_rank, int dtype, int p, unsigned long long output_elements,
    unsigned long long input_spatial, unsigned long long output_spatial,
    unsigned long long kernel_elements) {
  const unsigned long long* input_dimensions = metadata;
  const unsigned long long* output_strides = metadata + spatial_rank;
  const unsigned long long* kernel = metadata + spatial_rank * 2;
  const unsigned long long* strides = metadata + spatial_rank * 3;
  const unsigned long long* dilations = metadata + spatial_rank * 4;
  const unsigned long long* pads = metadata + spatial_rank * 5;
  for (unsigned long long output_index =
           blockIdx.x * blockDim.x + threadIdx.x;
       output_index < output_elements;
       output_index += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long group = output_index / output_spatial;
    const unsigned long long spatial_index = output_index % output_spatial;
    float sum = 0.0f;
    for (unsigned long long kernel_index = 0;
         kernel_index < kernel_elements; ++kernel_index) {
      unsigned long long output_remaining = spatial_index;
      unsigned long long input_index = 0;
      bool valid = true;
      for (int axis = 0; axis < spatial_rank; ++axis) {
        const unsigned long long output_coordinate =
            output_remaining / output_strides[axis];
        output_remaining %= output_strides[axis];
        unsigned long long kernel_stride = 1;
        for (int inner_axis = axis + 1; inner_axis < spatial_rank; ++inner_axis)
          kernel_stride *= kernel[inner_axis];
        const unsigned long long kernel_coordinate =
            (kernel_index / kernel_stride) % kernel[axis];
        const long long source =
            (long long)(output_coordinate * strides[axis])
            + (long long)(kernel_coordinate * dilations[axis])
            - (long long)pads[axis];
        if (source < 0 || source >= (long long)input_dimensions[axis]) {
          valid = false;
          break;
        }
        input_index =
            input_index * input_dimensions[axis] + (unsigned long long)source;
      }
      if (valid) {
        const float value =
            fabsf(load_pool_value(input, dtype, group * input_spatial + input_index));
        sum += p == 1 ? value : powf(value, (float)p);
      }
    }
    store_pool_value(output, dtype, output_index,
                     p == 1 ? sum : powf(sum, 1.0f / (float)p));
  }
}
"#;

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

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "pooling creates per-call cuDNN descriptors and performs a trailing host stream synchronization",
        )
    }
}

pub struct LpPoolFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for LpPoolFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let p = int_attr(node, "LpPool", "p", 2)?;
        let p = i32::try_from(p)
            .ok()
            .filter(|&value| value > 0)
            .ok_or_else(|| {
                EpError::KernelFailed("cuda_ep LpPool: p must be a positive 32-bit integer".into())
            })?;
        Ok(Box::new(LpPoolKernel {
            runtime: self.runtime.clone(),
            p,
            kernel_shape: ints_attr(node, "LpPool", "kernel_shape", None)?,
            strides: node
                .attr("strides")
                .and_then(|attribute| attribute.as_ints())
                .map(<[i64]>::to_vec),
            dilations: node
                .attr("dilations")
                .and_then(|attribute| attribute.as_ints())
                .map(<[i64]>::to_vec),
            pads: node
                .attr("pads")
                .and_then(|attribute| attribute.as_ints())
                .map(<[i64]>::to_vec),
            auto_pad: node
                .attr("auto_pad")
                .map(|attribute| {
                    attribute.as_str().ok_or_else(|| {
                        EpError::KernelFailed(
                            "cuda_ep LpPool: auto_pad must be a UTF-8 string".into(),
                        )
                    })
                })
                .transpose()?
                .unwrap_or("NOTSET")
                .to_owned(),
            ceil_mode: match int_attr(node, "LpPool", "ceil_mode", 0)? {
                0 => false,
                1 => true,
                value => {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep LpPool: ceil_mode must be 0 or 1, got {value}"
                    )));
                }
            },
        }))
    }
}

struct LpPoolKernel {
    runtime: Arc<CudaRuntime>,
    p: i32,
    kernel_shape: Vec<i64>,
    strides: Option<Vec<i64>>,
    dilations: Option<Vec<i64>>,
    pads: Option<Vec<i64>>,
    auto_pad: String,
    ceil_mode: bool,
}

fn positive_spatial(
    values: Option<&[i64]>,
    rank: usize,
    default: i64,
    name: &str,
) -> Result<Vec<u64>> {
    let values = values
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![default; rank]);
    if values.len() != rank || values.iter().any(|&value| value <= 0) {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep LpPool: {name} must contain {rank} positive values"
        )));
    }
    Ok(values.into_iter().map(|value| value as u64).collect())
}

fn upload_u64(runtime: &CudaRuntime, values: &[u64]) -> Result<CUdeviceptr> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let pointer = runtime.alloc_raw(bytes.len())?;
    if let Err(error) = unsafe { runtime.htod(&bytes, pointer) } {
        let _ = unsafe { runtime.free_raw(pointer) };
        return Err(error);
    }
    Ok(pointer)
}

fn lp_dtype(dtype: DataType) -> Result<i32> {
    match dtype {
        DataType::Float32 => Ok(0),
        DataType::Float16 => Ok(1),
        DataType::BFloat16 => Ok(2),
        other => Err(not_implemented(format!(
            "LpPool dtype {other:?} (supported: Float32, Float16, BFloat16)"
        ))),
    }
}

impl Kernel for LpPoolKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep LpPool: expected 1 input and 1 output".into(),
            ));
        }
        let input = &inputs[0];
        let output = &mut outputs[0];
        if input.shape.len() < 3 {
            return Err(EpError::KernelFailed(
                "cuda_ep LpPool: input rank must be at least 3".into(),
            ));
        }
        if input.dtype != output.dtype || !input.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented(
                "LpPool requires contiguous input/output with matching dtypes",
            ));
        }
        let dtype = lp_dtype(input.dtype)?;
        if dtype != 0 {
            self.runtime.require_nvrtc_half_headers("LpPool")?;
        }
        let rank = input.shape.len() - 2;
        let kernel = positive_spatial(Some(&self.kernel_shape), rank, 1, "kernel_shape")?;
        let strides = positive_spatial(self.strides.as_deref(), rank, 1, "strides")?;
        let dilations = positive_spatial(self.dilations.as_deref(), rank, 1, "dilations")?;
        let mut pads = match self.pads.as_deref() {
            None => vec![0_u64; rank * 2],
            Some(values) if values.len() == rank * 2 && values.iter().all(|&value| value >= 0) => {
                values.iter().map(|&value| value as u64).collect()
            }
            Some(_) => {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep LpPool: pads must contain {} non-negative values",
                    rank * 2
                )));
            }
        };
        let spatial = input.shape[2..]
            .iter()
            .map(|&value| value as u64)
            .collect::<Vec<_>>();
        let mut output_spatial = Vec::with_capacity(rank);
        match self.auto_pad.as_str() {
            "" | "NOTSET" => {
                for axis in 0..rank {
                    let effective = dilations[axis] * (kernel[axis] - 1) + 1;
                    let numerator =
                        spatial[axis] as i64 + pads[axis] as i64 + pads[axis + rank] as i64
                            - effective as i64;
                    let mut extent = if self.ceil_mode {
                        ((numerator + strides[axis] as i64 - 1) / strides[axis] as i64 + 1).max(0)
                            as u64
                    } else {
                        (numerator / strides[axis] as i64 + 1).max(0) as u64
                    };
                    if self.ceil_mode
                        && extent > 0
                        && (extent - 1) * strides[axis] >= spatial[axis] + pads[axis]
                    {
                        extent -= 1;
                    }
                    output_spatial.push(extent);
                }
            }
            "VALID" => {
                pads.fill(0);
                for axis in 0..rank {
                    let effective = dilations[axis] * (kernel[axis] - 1) + 1;
                    output_spatial.push(
                        ((spatial[axis] as i64 - effective as i64) / strides[axis] as i64 + 1)
                            .max(0) as u64,
                    );
                }
            }
            "SAME_UPPER" | "SAME_LOWER" => {
                for axis in 0..rank {
                    let extent = spatial[axis].div_ceil(strides[axis]);
                    let effective = dilations[axis] * (kernel[axis] - 1) + 1;
                    let total = (extent.saturating_sub(1) * strides[axis] + effective)
                        .saturating_sub(spatial[axis]);
                    let begin = if self.auto_pad == "SAME_LOWER" {
                        total.div_ceil(2)
                    } else {
                        total / 2
                    };
                    pads[axis] = begin;
                    pads[axis + rank] = total - begin;
                    output_spatial.push(extent);
                }
            }
            other => {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep LpPool: unsupported auto_pad {other:?}"
                )));
            }
        }
        let expected = [input.shape[0], input.shape[1]]
            .into_iter()
            .chain(output_spatial.iter().map(|&value| value as usize))
            .collect::<Vec<_>>();
        if output.shape != expected {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep LpPool: output shape {:?}, expected {expected:?}",
                output.shape
            )));
        }
        if output.numel() == 0 {
            return Ok(());
        }
        let mut output_strides = vec![1_u64; rank];
        for axis in (0..rank.saturating_sub(1)).rev() {
            output_strides[axis] = output_strides[axis + 1] * output_spatial[axis + 1];
        }
        let metadata = spatial
            .iter()
            .chain(&output_strides)
            .chain(&kernel)
            .chain(&strides)
            .chain(&dilations)
            .chain(&pads[..rank])
            .copied()
            .collect::<Vec<_>>();
        let metadata_pointer = upload_u64(&self.runtime, &metadata)?;
        let function = self
            .runtime
            .nvrtc_function("lp_pool_v1", LP_SOURCE, "lp_pool")?;
        let input_pointer = cuptr(input.data_ptr::<u8>() as *const c_void);
        let output_pointer = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let spatial_rank = i32::try_from(rank)
            .map_err(|_| EpError::KernelFailed("cuda_ep LpPool: rank exceeds i32".into()))?;
        let output_elements = output.numel() as u64;
        let input_spatial = spatial.iter().product::<u64>();
        let output_spatial_elements = output_spatial.iter().product::<u64>();
        let kernel_elements = kernel.iter().product::<u64>();
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&input_pointer)
            .arg(&output_pointer)
            .arg(&metadata_pointer)
            .arg(&spatial_rank)
            .arg(&dtype)
            .arg(&self.p)
            .arg(&output_elements)
            .arg(&input_spatial)
            .arg(&output_spatial_elements)
            .arg(&kernel_elements);
        let launch = unsafe {
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
        .map_err(|error| driver_err("launch LpPool", error));
        let sync = launch.and_then(|_| self.runtime.synchronize());
        let free = unsafe { self.runtime.free_raw(metadata_pointer) };
        sync.and(free)
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "LpPool allocates, uploads, and frees per-call shape metadata",
        )
    }
}
