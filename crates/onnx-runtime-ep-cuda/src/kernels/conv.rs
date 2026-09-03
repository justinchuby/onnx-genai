//! ONNX `Conv` via cuDNN's forward-convolution API.
//!
//! Rank-4 NCHW convolution uses cuDNN. Rank-3 NCL convolution uses an
//! output-owned CUDA kernel so ONNX's asymmetric (including causal) padding is
//! supported without changing the established 2-D path.

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, Result, TensorMetadata, TensorMut, TensorView,
    WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Node, compute_contiguous_strides};

use crate::cudnn::{
    CudnnConvBuffers, CudnnConvPlanCache, CudnnConvSpec, CudnnTensorType,
    governed_workspace_requirement,
};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const CONV1D_SOURCE: &str = r#"
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

extern "C" __global__ void conv1d(
    const void* x, const void* weights, const void* bias, void* output,
    unsigned long long output_elements, unsigned long long input_channels,
    unsigned long long input_length, unsigned long long output_channels,
    unsigned long long output_length, unsigned long long filter_channels,
    unsigned long long kernel_length, unsigned long long outputs_per_group,
    unsigned long long stride, unsigned long long dilation,
    unsigned long long pad_begin, int dtype, int has_bias) {
  for (unsigned long long output_index =
           blockIdx.x * blockDim.x + threadIdx.x;
       output_index < output_elements;
       output_index += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long output_position = output_index % output_length;
    const unsigned long long output_channel =
        (output_index / output_length) % output_channels;
    const unsigned long long batch =
        output_index / (output_channels * output_length);
    const unsigned long long group = output_channel / outputs_per_group;
    float accumulated =
        has_bias ? load_value(bias, dtype, output_channel) : 0.0f;

    for (unsigned long long local_channel = 0;
         local_channel < filter_channels; ++local_channel) {
      const unsigned long long input_channel =
          group * filter_channels + local_channel;
      for (unsigned long long kernel = 0; kernel < kernel_length; ++kernel) {
        const long long input_position =
            (long long)(output_position * stride + kernel * dilation)
            - (long long)pad_begin;
        if (input_position < 0 ||
            input_position >= (long long)input_length) continue;
        const unsigned long long input_index =
            (batch * input_channels + input_channel) * input_length
            + (unsigned long long)input_position;
        const unsigned long long weight_index =
            (output_channel * filter_channels + local_channel) * kernel_length
            + kernel;
        accumulated += load_value(x, dtype, input_index)
            * load_value(weights, dtype, weight_index);
      }
    }
    store_value(output, dtype, output_index, accumulated);
  }
}
"#;

pub struct ConvFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for ConvFactory {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let rank = input_shapes.first().map(Vec::len).unwrap_or(4);
        let (default_strides, default_pads, default_dilations): (&[i64], &[i64], &[i64]) =
            if rank == 3 {
                (&[1], &[0, 0], &[1])
            } else {
                (&[1, 1], &[0, 0, 0, 0], &[1, 1])
            };
        Ok(Box::new(ConvKernel {
            runtime: self.runtime.clone(),
            strides: ints_attr(node, "strides", default_strides)?,
            pads: ints_attr(node, "pads", default_pads)?,
            dilations: ints_attr(node, "dilations", default_dilations)?,
            kernel_shape: node
                .attr("kernel_shape")
                .map(|value| {
                    value.as_ints().map(ToOwned::to_owned).ok_or_else(|| {
                        EpError::KernelFailed(
                            "cuda_ep Conv: kernel_shape must be an integer list".into(),
                        )
                    })
                })
                .transpose()?,
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
            conv_plan: Mutex::new(None),
            prepared_conv_plan: Mutex::new(None),
            last_call_capture_safe: AtomicBool::new(false),
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
    kernel_shape: Option<Vec<i64>>,
    group: i64,
    auto_pad: String,
    /// Last successfully executed cuDNN plan. Workspace queries never mutate it.
    conv_plan: Mutex<Option<CudnnConvPlanCache>>,
    /// Prospective plan for the immediately following execution. Abandoning the
    /// dispatch or failing workspace allocation leaves `conv_plan` untouched.
    prepared_conv_plan: Mutex<Option<CudnnConvPlanCache>>,
    last_call_capture_safe: AtomicBool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Conv1dPlan {
    output_shape: [usize; 3],
    pad_begin: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
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
    fn plan_1d(&self, x: &[usize], w: &[usize]) -> Result<Conv1dPlan> {
        if x.len() != 3 || w.len() != 3 {
            return Err(not_implemented(format!(
                "Conv with input rank {} and filter rank {} (supported: 1-D NCL or 2-D NCHW)",
                x.len(),
                w.len()
            )));
        }
        let stride = single("strides", &self.strides, false)?;
        let dilation = single("dilations", &self.dilations, false)?;
        if let Some(kernel_shape) = &self.kernel_shape
            && (kernel_shape.len() != 1 || usize::try_from(kernel_shape[0]).ok() != Some(w[2]))
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Conv: kernel_shape must match filter spatial shape [{}], got {kernel_shape:?}",
                w[2]
            )));
        }
        let groups = self.validate_groups(x, w)?;
        let effective = dilation
            .checked_mul(w[2].saturating_sub(1))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| EpError::KernelFailed("cuda_ep Conv: kernel length overflow".into()))?;
        let (pad_begin, pad_end) = match self.auto_pad.as_str() {
            "" | "NOTSET" => {
                if self.pads.len() != 2 {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep Conv: pads must have 2 values [begin,end] for Conv1D, got {:?}",
                        self.pads
                    )));
                }
                let pads = self
                    .pads
                    .iter()
                    .map(|&value| {
                        usize::try_from(value).map_err(|_| {
                            EpError::KernelFailed(format!(
                                "cuda_ep Conv: pads must be non-negative, got {:?}",
                                self.pads
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                (pads[0], pads[1])
            }
            "VALID" => (0, 0),
            "SAME_UPPER" | "SAME_LOWER" => {
                let output = x[2].div_ceil(stride);
                let total = output
                    .saturating_sub(1)
                    .saturating_mul(stride)
                    .saturating_add(effective)
                    .saturating_sub(x[2]);
                if self.auto_pad == "SAME_LOWER" {
                    (total.div_ceil(2), total / 2)
                } else {
                    (total / 2, total.div_ceil(2))
                }
            }
            other => {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Conv: unsupported auto_pad value {other:?}; expected NOTSET, VALID, SAME_UPPER, or SAME_LOWER"
                )));
            }
        };
        let padded = x[2]
            .checked_add(pad_begin)
            .and_then(|value| value.checked_add(pad_end))
            .ok_or_else(|| EpError::KernelFailed("cuda_ep Conv: padded length overflow".into()))?;
        if padded < effective {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Conv: effective kernel {effective} exceeds padded input {padded}"
            )));
        }
        Ok(Conv1dPlan {
            output_shape: [x[0], w[0], (padded - effective) / stride + 1],
            pad_begin,
            stride,
            dilation,
            groups,
        })
    }

    fn validate_groups(&self, x: &[usize], w: &[usize]) -> Result<usize> {
        let groups = usize::try_from(self.group)
            .ok()
            .filter(|&value| value > 0)
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
        if !w[0].is_multiple_of(groups) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Conv: output channels {} must be divisible by group {groups}",
                w[0]
            )));
        }
        Ok(groups)
    }

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
        let groups = self.validate_groups(x, w)?;

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

    fn workspace_requirement_for_metadata(
        &self,
        inputs: &[TensorMetadata<'_>],
        capturing: bool,
    ) -> Result<WorkspaceRequirement> {
        *self.prepared_conv_plan.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep Conv: prepared cuDNN plan lock was poisoned".into())
        })? = None;
        let (Some(x), Some(w)) = (inputs.first(), inputs.get(1)) else {
            return Ok(WorkspaceRequirement::NONE);
        };
        if !x.present || !w.present {
            return Ok(WorkspaceRequirement::NONE);
        }
        if x.shape.len() == 3 || w.shape.len() == 3 {
            return Ok(WorkspaceRequirement::NONE);
        }
        if !matches!(
            x.dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) || w.dtype != x.dtype
        {
            return Ok(WorkspaceRequirement::NONE);
        }
        let Some(plan) = self.plan(x.shape, w.shape).ok() else {
            return Ok(WorkspaceRequirement::NONE);
        };
        let bias_present = inputs.get(2).is_some_and(|bias| bias.present);
        if bias_present && inputs.get(2).is_some_and(|bias| bias.dtype != x.dtype) {
            return Ok(WorkspaceRequirement::NONE);
        }
        if !self.runtime.cudnn().is_available() {
            return Ok(WorkspaceRequirement::NONE);
        }

        let x_strides = compute_contiguous_strides(x.shape);
        let y_strides = compute_contiguous_strides(&plan.output_shape);
        let spec = CudnnConvSpec {
            dtype: CudnnTensorType::from_onnx(x.dtype)?,
            input_dims: dims4(x.shape, "input")?,
            input_strides: strides4(&x_strides, "input")?,
            filter_dims: dims4(w.shape, "filter")?,
            output_dims: dims4(&plan.output_shape, "output")?,
            output_strides: strides4(&y_strides, "output")?,
            pads: i32_pair(plan.pads, "pads")?,
            strides: i32_pair(plan.strides, "strides")?,
            dilations: i32_pair(plan.dilations, "dilations")?,
            groups: i32::try_from(plan.groups)
                .map_err(|_| EpError::KernelFailed("cuda_ep Conv: group exceeds i32".into()))?,
        };
        let current = self
            .conv_plan
            .lock()
            .map_err(|_| {
                EpError::KernelFailed("cuda_ep Conv: cuDNN plan cache lock was poisoned".into())
            })?
            .clone();
        let prepared =
            if let Some(current) = current.filter(|current| current.matches(&spec, bias_present)) {
                current
            } else if capturing {
                return Err(EpError::KernelFailed(
                    "cuda_ep Conv: cuDNN convolution signature changed during CUDA graph capture; \
                 abort capture and warm the exact fixed shape first"
                        .into(),
                ));
            } else {
                self.runtime
                    .cudnn()
                    .with_handle(|handle| handle.prepare_conv(&spec, bias_present))?
            };
        let bytes = prepared.workspace_bytes();
        *self.prepared_conv_plan.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep Conv: prepared cuDNN plan lock was poisoned".into())
        })? = Some(prepared);
        Ok(governed_workspace_requirement(bytes))
    }

    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
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

        if x.shape.len() == 3 || w.shape.len() == 3 {
            let result = self.run_1d(x, w, bias, &mut outputs[0]);
            if result.is_ok() {
                self.last_call_capture_safe.store(true, Ordering::Relaxed);
            }
            return result;
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
        let capturing = self.runtime.is_capturing()?;
        let prepared = self
            .prepared_conv_plan
            .lock()
            .map_err(|_| {
                EpError::KernelFailed("cuda_ep Conv: prepared cuDNN plan lock was poisoned".into())
            })?
            .take();
        let current = self
            .conv_plan
            .lock()
            .map_err(|_| {
                EpError::KernelFailed("cuda_ep Conv: cuDNN plan cache lock was poisoned".into())
            })?
            .clone();
        let conv_plan = if let Some(prepared) =
            prepared.filter(|prepared| prepared.matches(&spec, bias.is_some()))
        {
            prepared
        } else if let Some(current) =
            current.filter(|current| current.matches(&spec, bias.is_some()))
        {
            current
        } else if capturing {
            return Err(EpError::KernelFailed(
                "cuda_ep Conv: cuDNN convolution signature changed during CUDA graph capture; \
                 abort capture and warm the exact fixed shape first"
                    .into(),
            ));
        } else {
            self.runtime
                .cudnn()
                .with_handle(|handle| handle.prepare_conv(&spec, bias.is_some()))?
        };
        self.runtime
            .cudnn()
            .with_handle(|handle| handle.conv2d(&conv_plan, &spec, buffers, workspace))?;
        if !capturing {
            self.runtime.synchronize()?;
            *self.conv_plan.lock().map_err(|_| {
                EpError::KernelFailed("cuda_ep Conv: cuDNN plan cache lock was poisoned".into())
            })? = Some(conv_plan);
        }
        self.last_call_capture_safe.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn run_1d(
        &self,
        x: &TensorView,
        w: &TensorView,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
    ) -> Result<()> {
        let plan = self.plan_1d(x.shape, w.shape)?;
        if output.shape != plan.output_shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Conv: output shape {:?}, expected {:?}",
                output.shape, plan.output_shape
            )));
        }
        if let Some(bias) = bias
            && bias.shape != [w.shape[0]]
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Conv: bias shape {:?}, expected [{}]",
                bias.shape, w.shape[0]
            )));
        }
        if output.numel() == 0 {
            return Ok(());
        }
        let dtype = dtype_code(x.dtype)?;
        if dtype != 0 {
            self.runtime.require_nvrtc_half_headers("Conv1D")?;
        }
        let function = self
            .runtime
            .nvrtc_function("conv1d_common_v1", CONV1D_SOURCE, "conv1d")?;
        let x_pointer = cuptr(x.data_ptr::<u8>() as *const c_void);
        let weight_pointer = cuptr(w.data_ptr::<u8>() as *const c_void);
        let bias_pointer = bias
            .map(|value| cuptr(value.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_pointer = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let output_elements = output.numel() as u64;
        let dimensions = [
            x.shape[1],
            x.shape[2],
            w.shape[0],
            plan.output_shape[2],
            w.shape[1],
            w.shape[2],
            w.shape[0] / plan.groups,
            plan.stride,
            plan.dilation,
            plan.pad_begin,
        ]
        .map(|value| value as u64);
        let has_bias = i32::from(bias.is_some());
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&x_pointer)
            .arg(&weight_pointer)
            .arg(&bias_pointer)
            .arg(&output_pointer)
            .arg(&output_elements);
        for dimension in &dimensions {
            builder.arg(dimension);
        }
        builder.arg(&dtype).arg(&has_bias);
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
        .map_err(|error| driver_err("launch Conv1D", error))?;
        Ok(())
    }
}

fn dtype_code(dtype: DataType) -> Result<i32> {
    match dtype {
        DataType::Float32 => Ok(0),
        DataType::Float16 => Ok(1),
        DataType::BFloat16 => Ok(2),
        other => Err(not_implemented(format!(
            "Conv dtype {other:?} (supported: Float32, Float16, BFloat16)"
        ))),
    }
}

fn single(name: &str, values: &[i64], allow_zero: bool) -> Result<usize> {
    if values.len() != 1 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep Conv: {name} must have 1 value for Conv1D, got {values:?}"
        )));
    }
    let value = usize::try_from(values[0]).map_err(|_| {
        EpError::KernelFailed(format!(
            "cuda_ep Conv: {name} values must be non-negative, got {values:?}"
        ))
    })?;
    if !allow_zero && value == 0 {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep Conv: {name} values must be positive, got {values:?}"
        )));
    }
    Ok(value)
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
        self.run(inputs, outputs, None)
    }

    fn workspace_requirement(&self, inputs: &[TensorMetadata<'_>]) -> Result<WorkspaceRequirement> {
        self.workspace_requirement_for_metadata(inputs, self.runtime.is_capturing()?)
    }

    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.run(inputs, outputs, workspace)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        if self.last_call_capture_safe.load(Ordering::Relaxed) {
            onnx_runtime_ep_api::CaptureSupport::Supported
        } else {
            onnx_runtime_ep_api::CaptureSupport::unsupported(
                "requires a warmed fixed-shape cuDNN convolution plan and prepared persistent workspace",
            )
        }
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
