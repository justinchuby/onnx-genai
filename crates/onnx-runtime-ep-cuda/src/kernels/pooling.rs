//! ONNX pooling: cuDNN for undilated 2-D pooling, NVRTC for dilated
//! AveragePool and LpPool.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, OutputMetadata, Result, TensorMetadata, TensorMut, TensorView,
    WorkspacePreparation, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Node, Shape, as_static_shape};

use crate::cudnn::{CudnnBufferPair, CudnnPoolingMode, CudnnPoolingSpec, CudnnTensorType};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

use super::metadata_workspace::{MetadataWorkspace, checked_tensor_bytes};
const METADATA_WORDS_PER_AXIS: usize = 6;

const BLOCK: u32 = 256;
const WINDOW_POOL_SOURCE: &str = r#"
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

template <typename Scalar>
__device__ __forceinline__ Scalar load_window_value(
    const void* values, int dtype, unsigned long long index);

template <>
__device__ __forceinline__ float load_window_value<float>(
    const void* values, int dtype, unsigned long long index) {
  return load_pool_value(values, dtype, index);
}

template <>
__device__ __forceinline__ double load_window_value<double>(
    const void* values, int, unsigned long long index) {
  return ((const double*)values)[index];
}

__device__ __forceinline__ void store_window_value(
    void* values, int dtype, unsigned long long index, float value) {
  store_pool_value(values, dtype, index, value);
}

__device__ __forceinline__ void store_window_value(
    void* values, int, unsigned long long index, double value) {
  ((double*)values)[index] = value;
}

__device__ __forceinline__ float lp_pow(float value, float exponent) {
  return powf(value, exponent);
}

__device__ __forceinline__ double lp_pow(double value, double exponent) {
  return pow(value, exponent);
}

template <bool Average, typename Scalar>
__device__ void window_pool(
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
    Scalar sum = 0;
    unsigned long long valid_count = 0;
    // Decode flat tap indices in reverse axis order to avoid recomputing
    // the kernel's suffix products inside each window.
    for (unsigned long long kernel_index = 0;
         kernel_index < kernel_elements; ++kernel_index) {
      unsigned long long tap = kernel_index;
      unsigned long long input_index = 0;
      unsigned long long input_stride = 1;
      bool valid = true;
      for (int axis = spatial_rank - 1; axis >= 0; --axis) {
        const unsigned long long output_dim =
            axis == 0 ? output_spatial / output_strides[0]
                      : output_strides[axis - 1] / output_strides[axis];
        const unsigned long long output_coordinate =
            (spatial_index / output_strides[axis]) % output_dim;
        const unsigned long long kernel_coordinate = tap % kernel[axis];
        tap /= kernel[axis];
        const unsigned long long position = output_coordinate * strides[axis]
            + kernel_coordinate * dilations[axis];
        const unsigned long long source = position - pads[axis];
        if (position < pads[axis] || source >= input_dimensions[axis]) {
          valid = false;
          break;
        }
        input_index += source * input_stride;
        input_stride *= input_dimensions[axis];
      }
      if (valid) {
        const Scalar value =
            load_window_value<Scalar>(input, dtype, group * input_spatial + input_index);
        if (Average) {
          sum += value;
          ++valid_count;
        } else {
          const Scalar magnitude = value == 0 ? Scalar(0) : (value < 0 ? -value : value);
          sum += p == 1 ? magnitude : lp_pow(magnitude, Scalar(p));
        }
      }
    }
    if (Average) {
      // Include-padding uses the number of taps, not the effective kernel
      // area, and treats ceil-mode overhang the same as explicit zero pads.
      const unsigned long long divisor = p != 0 ? kernel_elements : valid_count;
      store_pool_value(output, dtype, output_index, divisor > 0 ? sum / (float)divisor : 0.0f);
    } else {
      store_window_value(output, dtype, output_index,
                         p == 1 ? sum : lp_pow(sum, Scalar(1) / Scalar(p)));
    }
  }
}

extern "C" __global__ void lp_pool(
    const void* input, void* output, const unsigned long long* metadata,
    int spatial_rank, int dtype, int p, unsigned long long output_elements,
    unsigned long long input_spatial, unsigned long long output_spatial,
    unsigned long long kernel_elements) {
  if (dtype == 3) {
    window_pool<false, double>(input, output, metadata, spatial_rank, dtype, p,
                              output_elements, input_spatial, output_spatial, kernel_elements);
  } else {
    window_pool<false, float>(input, output, metadata, spatial_rank, dtype, p,
                             output_elements, input_spatial, output_spatial, kernel_elements);
  }
}

extern "C" __global__ void dilated_average_pool(
    const void* input, void* output, const unsigned long long* metadata,
    int spatial_rank, int dtype, int include_pad, unsigned long long output_elements,
    unsigned long long input_spatial, unsigned long long output_spatial,
    unsigned long long kernel_elements) {
  window_pool<true, float>(input, output, metadata, spatial_rank, dtype, include_pad,
                          output_elements, input_spatial, output_spatial, kernel_elements);
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
    pub average_supports_dilation: bool,
}

impl KernelFactory for PoolFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let op = self.kind.name();
        if self.kind == PoolKind::Average
            && let Some(rank) = dilated_average_rank(node, self.average_supports_dilation)?
        {
            return Ok(Box::new(WindowPoolKernel {
                runtime: self.runtime.clone(),
                reduction: WindowReduction::Average(count_include_pad(node, op)?),
                params: WindowParams::parse(node, op, Some(rank))?,
            }));
        }
        Ok(Box::new(PoolKernel {
            runtime: self.runtime.clone(),
            params: CudnnPoolParams::parse(node, self.kind)?,
        }))
    }
}

fn count_include_pad(node: &Node, op: &str) -> Result<bool> {
    match int_attr(node, op, "count_include_pad", 0)? {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(EpError::KernelFailed(format!(
            "cuda_ep {op}: count_include_pad must be 0 or 1, got {value}"
        ))),
    }
}

impl CudnnPoolParams {
    fn parse(node: &Node, kind: PoolKind) -> Result<Self> {
        let op = kind.name();
        if kind == PoolKind::Max && node.outputs.len() > 1 {
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
        if kind == PoolKind::Max {
            let storage_order = int_attr(node, op, "storage_order", 0)?;
            if storage_order != 0 {
                return Err(not_implemented(format!(
                    "MaxPool storage_order={storage_order} (only row-major storage_order=0 is supported)"
                )));
            }
        }
        let count_include_pad = count_include_pad(node, op)?;
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

        Ok(Self {
            kind,
            kernel_shape,
            strides,
            pads,
            auto_pad,
            count_include_pad,
        })
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
    params: CudnnPoolParams,
}

#[derive(Debug)]
struct CudnnPoolParams {
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

impl CudnnPoolParams {
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
}

impl PoolKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let op = self.params.kind.name();
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
        let plan = self.params.plan(input.shape)?;
        if output.shape != plan.output_shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output shape {:?}, expected {:?}",
                output.shape, plan.output_shape
            )));
        }
        if output.numel() == 0 {
            return Ok(());
        }
        let mode = match self.params.kind {
            PoolKind::Max => CudnnPoolingMode::Max,
            PoolKind::Average if self.params.count_include_pad => {
                CudnnPoolingMode::AverageIncludePadding
            }
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
        Ok(Box::new(WindowPoolKernel {
            runtime: self.runtime.clone(),
            reduction: WindowReduction::Lp(p),
            params: WindowParams::parse(node, "LpPool", None)?,
        }))
    }
}

fn dilated_average_rank(node: &Node, supported_opset: bool) -> Result<Option<usize>> {
    let Some(dilations) = node.attr("dilations") else {
        return Ok(None);
    };
    if !supported_opset {
        return Err(not_implemented(
            "AveragePool attribute dilations requires ONNX opset 19 or later",
        ));
    }
    let dilations = dilations.as_ints().ok_or_else(|| {
        EpError::KernelFailed("cuda_ep AveragePool: dilations must be an integer list".into())
    })?;
    if dilations.is_empty() || dilations.iter().any(|&value| value <= 0) {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep AveragePool: dilations must contain positive values, got {dilations:?}"
        )));
    }
    Ok(dilations
        .iter()
        .any(|&value| value != 1)
        .then_some(dilations.len()))
}

pub(super) fn validate_average_claim(
    node: &Node,
    opset: u64,
    input_shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Result<()> {
    let rank = dilated_average_rank(node, opset >= 19)?;
    let op = "AveragePool";
    if node.inputs.len() != 1
        || node.inputs[0].is_none()
        || node.outputs.len() != 1
        || node.outputs[0].is_none()
    {
        return Err(EpError::KernelFailed(
            "cuda_ep AveragePool: expected one present input and one present output".into(),
        ));
    }
    if let Some(&dtype) = input_dtypes.first() {
        window_dtype(op, dtype)?;
        if dtype == DataType::BFloat16 && opset < 22 {
            return Err(not_implemented(
                "AveragePool bfloat16 requires ONNX opset 22 or later",
            ));
        }
    }
    if let Some(rank) = rank {
        count_include_pad(node, op)?;
        let params = WindowParams::parse(node, op, Some(rank))?;
        if let Some(shape) = input_shapes.first() {
            if shape.len().checked_sub(2) != Some(rank) {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep AveragePool: {} dilations require rank {} input, got rank {}",
                    rank,
                    rank + 2,
                    shape.len()
                )));
            }
            if let Some(shape) = as_static_shape(shape) {
                params.geometry(op, &shape[2..])?;
            }
        }
    } else {
        let params = CudnnPoolParams::parse(node, PoolKind::Average)?;
        if let Some(shape) = input_shapes.first() {
            if shape.len() != 4 {
                return Err(not_implemented(
                    "unit-dilation AveragePool requires rank 4 (cuDNN supports 2-D NCHW only)",
                ));
            }
            if let Some(shape) = as_static_shape(shape) {
                params.plan(&shape)?;
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum WindowReduction {
    Lp(i32),
    Average(bool),
}

impl WindowReduction {
    fn op(self) -> &'static str {
        match self {
            Self::Lp(_) => "LpPool",
            Self::Average(_) => "AveragePool",
        }
    }

    fn entry(self) -> (&'static str, i32) {
        match self {
            Self::Lp(p) => ("lp_pool", p),
            Self::Average(include_pad) => ("dilated_average_pool", i32::from(include_pad)),
        }
    }
}

struct WindowPoolKernel {
    runtime: Arc<CudaRuntime>,
    reduction: WindowReduction,
    params: WindowParams,
}

impl WindowParams {
    fn parse(node: &Node, op: &str, rank: Option<usize>) -> Result<Self> {
        let list = |name| {
            node.attr(name)
                .map(|_| ints_attr(node, op, name, None))
                .transpose()
        };
        let params = Self {
            kernel_shape: ints_attr(node, op, "kernel_shape", None)?,
            strides: list("strides")?,
            dilations: list("dilations")?,
            pads: list("pads")?,
            auto_pad: node
                .attr("auto_pad")
                .map(|attribute| {
                    attribute.as_str().ok_or_else(|| {
                        EpError::KernelFailed(format!(
                            "cuda_ep {op}: auto_pad must be a UTF-8 string"
                        ))
                    })
                })
                .transpose()?
                .unwrap_or("NOTSET")
                .to_owned(),
            ceil_mode: match int_attr(node, op, "ceil_mode", 0)? {
                0 => false,
                1 => true,
                value => {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep {op}: ceil_mode must be 0 or 1, got {value}"
                    )));
                }
            },
        };
        if let Some(rank) = rank {
            if params.pads.is_some() && !matches!(params.auto_pad.as_str(), "" | "NOTSET") {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep {op}: use either explicit pads or auto_pad, not both"
                )));
            }
            params.geometry(op, &vec![0; rank])?;
        }
        Ok(params)
    }
}

struct WindowParams {
    kernel_shape: Vec<i64>,
    strides: Option<Vec<i64>>,
    dilations: Option<Vec<i64>>,
    pads: Option<Vec<i64>>,
    auto_pad: String,
    ceil_mode: bool,
}

fn positive_spatial(
    op: &str,
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
            "cuda_ep {op}: {name} must contain {rank} positive values, got {values:?}"
        )));
    }
    Ok(values.into_iter().map(|value| value as u64).collect())
}

fn window_dtype(op: &str, dtype: DataType) -> Result<i32> {
    match dtype {
        DataType::Float32 => Ok(0),
        DataType::Float16 => Ok(1),
        DataType::BFloat16 => Ok(2),
        DataType::Float64 if op == "LpPool" => Ok(3),
        other => Err(not_implemented(format!(
            "{op} dtype {other:?} (supported: Float32, Float16, BFloat16)"
        ))),
    }
}

impl WindowPoolKernel {
    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let op = self.reduction.op();
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: expected 1 input and 1 output"
            )));
        }
        if self.runtime.is_capturing()? {
            return Err(not_implemented(format!(
                "{op} during CUDA graph capture: borrowed shape metadata requires an eager host upload"
            )));
        }
        let input = &inputs[0];
        let output = &mut outputs[0];
        if input.shape.len() < 3 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: input rank must be at least 3"
            )));
        }
        if input.dtype != output.dtype || !input.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented(format!(
                "{op} requires contiguous input/output with matching dtypes"
            )));
        }
        let dtype = window_dtype(op, input.dtype)?;
        checked_tensor_bytes(op, "input", input.dtype, input.shape)?;
        checked_tensor_bytes(op, "output", output.dtype, output.shape)?;
        if matches!(input.dtype, DataType::Float16 | DataType::BFloat16) {
            self.runtime.require_nvrtc_half_headers(op)?;
        }
        let spatial = &input.shape[2..];
        let rank = spatial.len();
        let WindowGeometry {
            kernel,
            strides,
            dilations,
            pads,
            output_spatial,
            kernel_elements,
        } = self.params.geometry(op, spatial)?;
        let expected = [input.shape[0], input.shape[1]]
            .into_iter()
            .chain(output_spatial.iter().map(|&value| value as usize))
            .collect::<Vec<_>>();
        if output.shape != expected {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output shape {:?}, expected {expected:?}",
                output.shape
            )));
        }
        if output.shape.contains(&0) {
            return Ok(());
        }
        let mut output_strides = vec![1_u64; rank];
        for axis in (0..rank - 1).rev() {
            output_strides[axis] = output_strides[axis + 1] * output_spatial[axis + 1];
        }
        let metadata = spatial
            .iter()
            .map(|&dim| dim as u64)
            .chain(output_strides)
            .chain(kernel)
            .chain(strides)
            .chain(dilations)
            .chain(pads[..rank].iter().copied())
            .collect::<Vec<_>>();
        let metadata_pointer = MetadataWorkspace::for_records(
            op,
            rank,
            METADATA_WORDS_PER_AXIS,
            &OutputMetadata::from_view(output),
        )?
        .bind(workspace)?
        .upload(&self.runtime, &metadata)?;
        let (entry, parameter) = self.reduction.entry();
        let function = self
            .runtime
            .nvrtc_function("window_pool_v1", WINDOW_POOL_SOURCE, entry)?;
        let input_pointer = cuptr(input.data_ptr::<u8>() as *const c_void);
        let output_pointer = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let spatial_rank = i32::try_from(rank).map_err(|_| {
            EpError::KernelFailed(format!("cuda_ep {op}: spatial rank exceeds i32"))
        })?;
        let output_elements = output.numel() as u64;
        let input_spatial = if spatial.contains(&0) {
            0
        } else {
            spatial.iter().map(|&dim| dim as u64).product()
        };
        let output_spatial_elements = output_spatial.iter().product::<u64>();
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&input_pointer)
            .arg(&output_pointer)
            .arg(&metadata_pointer)
            .arg(&spatial_rank)
            .arg(&dtype)
            .arg(&parameter)
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
        .map_err(|error| driver_err(entry, error));
        launch.and_then(|_| self.runtime.synchronize())
    }
}

struct WindowGeometry {
    kernel: Vec<u64>,
    strides: Vec<u64>,
    dilations: Vec<u64>,
    pads: Vec<u64>,
    output_spatial: Vec<u64>,
    kernel_elements: u64,
}

impl WindowParams {
    fn geometry(&self, op: &str, spatial: &[usize]) -> Result<WindowGeometry> {
        let rank = spatial.len();
        let kernel = positive_spatial(op, Some(&self.kernel_shape), rank, 1, "kernel_shape")?;
        let strides = positive_spatial(op, self.strides.as_deref(), rank, 1, "strides")?;
        let dilations = positive_spatial(op, self.dilations.as_deref(), rank, 1, "dilations")?;
        let mut pads = match self.pads.as_deref() {
            None => vec![0_u64; rank * 2],
            Some(values) if values.len() == rank * 2 && values.iter().all(|&value| value >= 0) => {
                values.iter().map(|&value| value as u64).collect()
            }
            Some(_) => {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep {op}: pads must contain {} non-negative values",
                    rank * 2
                )));
            }
        };
        let mut output_spatial = Vec::with_capacity(rank);
        for axis in 0..rank {
            let overflow = || {
                EpError::KernelFailed(format!(
                    "cuda_ep {op}: window geometry overflows on spatial axis {axis}; reduce the extent, dilation or padding"
                ))
            };
            let input = spatial[axis] as i128;
            let stride = i128::from(strides[axis]);
            let effective = i128::from(dilations[axis]) * (i128::from(kernel[axis]) - 1) + 1;
            let extent = match self.auto_pad.as_str() {
                "SAME_UPPER" | "SAME_LOWER" => {
                    let extent = (input + stride - 1) / stride;
                    let total = ((extent - 1).max(0) * stride + effective - input).max(0);
                    let begin = (total + i128::from(self.auto_pad == "SAME_LOWER")) / 2;
                    pads[axis] = u64::try_from(begin).map_err(|_| overflow())?;
                    extent
                }
                auto @ ("" | "NOTSET" | "VALID") => {
                    // Resolve VALID to zero pads before applying floor/ceil.
                    let ceil = self.ceil_mode;
                    if auto == "VALID" {
                        pads[axis] = 0;
                        pads[axis + rank] = 0;
                    }
                    let begin = i128::from(pads[axis]);
                    let numerator = input + begin + i128::from(pads[axis + rank]) - effective;
                    let quotient = numerator.div_euclid(stride)
                        + i128::from(ceil && numerator.rem_euclid(stride) != 0);
                    let mut extent = (quotient + 1).max(0);
                    if ceil && extent > 0 && (extent - 1) * stride >= input + begin {
                        extent -= 1;
                    }
                    extent
                }
                other => {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep {op}: unsupported auto_pad {other:?}"
                    )));
                }
            };
            if extent > 0 && (extent - 1) * stride + effective - 1 > i128::from(u64::MAX) {
                return Err(overflow());
            }
            output_spatial.push(isize::try_from(extent).map_err(|_| overflow())? as u64);
        }
        let kernel_elements = kernel.iter().try_fold(1_u64, |count, &dim| {
            count.checked_mul(dim).ok_or_else(|| EpError::KernelFailed(format!(
                "cuda_ep {op}: kernel_shape {:?} exceeds addressable tap count; reduce the window size", self.kernel_shape
            )))
        })?;
        Ok(WindowGeometry {
            kernel,
            strides,
            dilations,
            pads,
            output_spatial,
            kernel_elements,
        })
    }
}

impl Kernel for WindowPoolKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }
    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.run(inputs, outputs, workspace)
    }
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        if inputs.len() != 1
            || outputs.len() != 1
            || !inputs[0].present
            || inputs[0].shape.len() < 3
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {}: workspace planning requires one present input of rank >= 3 and one output",
                self.reduction.op()
            )));
        }
        Ok(WorkspacePreparation::Ready(
            MetadataWorkspace::for_records(
                self.reduction.op(),
                inputs[0].shape.len() - 2,
                METADATA_WORDS_PER_AXIS,
                &outputs[0],
            )?
            .requirement(),
        ))
    }
    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "window pooling uploads borrowed shape metadata from the host on each eager call",
        )
    }
}
