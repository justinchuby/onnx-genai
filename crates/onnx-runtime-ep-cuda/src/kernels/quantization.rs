//! Per-tensor ONNX linear quantization kernels.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::int16_quantization as q16;
use onnx_runtime_ir::{DataType, Node, Shape};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
__device__ int quant_code(const unsigned char* data, unsigned long long i, int bits, int sign) {
  const int byte = data[bits == 4 ? i / 2 : i];
  const int q = bits == 4 ? (byte >> ((i % 2) * 4)) & 15 : byte;
  const int bias = sign ? 1 << (bits - 1) : 0;
  return (q ^ bias) - bias;
}
extern "C" __global__ void dequantize_affine(
    const unsigned char* x, const float* scale, const unsigned char* zero_point,
    float* y, unsigned long long n, unsigned long long inner,
    unsigned long long axis, unsigned long long block, int bits, int sign) {
  for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long outer = i / inner / axis;
    const unsigned long long k = i / inner % axis;
    const unsigned long long p =
        block == 0 ? 0 : (outer * (1 + (axis - 1) / block) + k / block) * inner + i % inner;
    const int zp = zero_point ? quant_code(zero_point, p, bits, sign) : 0;
    y[i] = (quant_code(x, i, bits, sign) - zp) * scale[p];
  }
}
extern "C" __global__ void quantize_u8(
    const float* x, const float* scale, const unsigned char* zero_point,
    unsigned char* y, unsigned long long n) {
  const float s = *scale;
  const int zp = zero_point ? (int)*zero_point : 0;
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
    int q = __float2int_rn(x[i] / s) + zp;
    y[i] = (unsigned char)(q < 0 ? 0 : (q > 255 ? 255 : q));
  }
}
extern "C" __global__ void quantize_i8(
    const float* x, const float* scale, const signed char* zero_point,
    signed char* y, unsigned long long n) {
  const float s = *scale;
  const int zp = zero_point ? (int)*zero_point : 0;
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
    int q = __float2int_rn(x[i] / s) + zp;
    y[i] = (signed char)(q < -128 ? -128 : (q > 127 ? 127 : q));
  }
}
extern "C" __global__ void dequantize_u8(
    const unsigned char* x, const float* scale, const unsigned char* zero_point,
    float* y, unsigned long long n) {
  const float s = *scale;
  const int zp = zero_point ? (int)*zero_point : 0;
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (unsigned long long)gridDim.x * blockDim.x)
    y[i] = ((int)x[i] - zp) * s;
}
extern "C" __global__ void dequantize_i8(
    const signed char* x, const float* scale, const signed char* zero_point,
    float* y, unsigned long long n) {
  const float s = *scale;
  const int zp = zero_point ? (int)*zero_point : 0;
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (unsigned long long)gridDim.x * blockDim.x)
    y[i] = ((int)x[i] - zp) * s;
}

extern "C" __global__ void dynamic_quantize_u8(
    const float* x, unsigned char* y, float* scale_out,
    unsigned char* zero_point_out, unsigned long long n) {
  __shared__ float minimums[256];
  __shared__ float maximums[256];
  float minimum = 0.0f;
  float maximum = 0.0f;
  for (unsigned long long i = threadIdx.x; i < n; i += blockDim.x) {
    minimum = fminf(minimum, x[i]);
    maximum = fmaxf(maximum, x[i]);
  }
  minimums[threadIdx.x] = minimum;
  maximums[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned int offset = blockDim.x >> 1; offset; offset >>= 1) {
    if (threadIdx.x < offset) {
      minimums[threadIdx.x] =
          fminf(minimums[threadIdx.x], minimums[threadIdx.x + offset]);
      maximums[threadIdx.x] =
          fmaxf(maximums[threadIdx.x], maximums[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float scale = (maximums[0] - minimums[0]) / 255.0f;
    if (scale == 0.0f) scale = 1.0f;
    const int zero_point =
        max(0, min(255, __float2int_rn(-minimums[0] / scale)));
    *scale_out = scale;
    *zero_point_out = (unsigned char)zero_point;
  }
  __syncthreads();
  const float scale = *scale_out;
  const int zero_point = (int)*zero_point_out;
  for (unsigned long long i = threadIdx.x; i < n; i += blockDim.x) {
    const int value = __float2int_rn(x[i] / scale) + zero_point;
    y[i] = (unsigned char)max(0, min(255, value));
  }
}
"#;

pub fn unsupported_reason(op: &Node, shapes: &[Shape], dtypes: &[DataType]) -> Option<String> {
    let compact_optional_zp = op.inputs.get(2) == Some(&None) && shapes.len() == 2;
    if !(2..=3).contains(&op.inputs.len())
        || (shapes.len() != op.inputs.len() && !compact_optional_zp)
    {
        return Some(format!(
            "{} requires 2 or 3 present inputs with shape metadata",
            op.op_type
        ));
    }
    if op.op_type == "DequantizeLinear" {
        if !matches!(dtypes.get(1), Some(DataType::Float32 | DataType::Undefined))
            || !matches!(
                op.attr("output_dtype"),
                None | Some(onnx_runtime_ir::Attribute::Int(0 | 1))
            )
        {
            return Some("DequantizeLinear CUDA supports Float32 scale and output".into());
        }
        let block = op.attr("block_size").and_then(|a| a.as_int()).unwrap_or(0);
        if block > 0
            && shapes[0].len() == shapes[1].len()
            && shapes
                .get(2)
                .is_none_or(|shape| op.inputs[2].is_none() || shape == &shapes[1])
        {
            return None;
        }
    }
    if !shapes[1].is_empty() {
        return Some(format!(
            "{} CUDA coverage requires a scalar scale",
            op.op_type
        ));
    }
    if shapes.get(2).is_some_and(|shape| !shape.is_empty()) {
        return Some(format!(
            "{} CUDA coverage requires a scalar zero_point",
            op.op_type
        ));
    }
    None
}

pub(super) fn claim_int16(
    node: &Node,
    opset: u64,
    dtypes: &[DataType],
) -> Option<std::result::Result<(), String>> {
    let quantize = node.op_type == "QuantizeLinear";
    let attr = q16::attribute_dtype(node, "output_dtype");
    let zp = dtypes
        .get(2)
        .copied()
        .filter(|_| node.inputs.get(2).is_some_and(Option::is_some));
    let code = if quantize {
        let attr = attr.ok().flatten();
        if !zp.is_some_and(q16::is_code) && !attr.is_some_and(q16::is_code) {
            return None;
        }
        zp.or(attr)?
    } else {
        dtypes
            .first()
            .copied()
            .filter(|&dtype| q16::is_code(dtype))?
    };
    Some((|| {
        q16::validate_opset(node, opset)?;
        if !(2..=3).contains(&node.inputs.len())
            || node.inputs[..2].iter().any(Option::is_none)
            || !matches!(node.outputs.as_slice(), [Some(_)])
        {
            return Err("requires x, scale, optional zero_point and one output".into());
        }
        let [x, scale, ..] = dtypes else {
            return Err("requires x and scale dtype metadata".into());
        };
        if node.inputs.get(2).is_some_and(Option::is_some) && zp != Some(code) {
            return Err(format!(
                "zero_point must match the quantized dtype {code:?}"
            ));
        }
        if quantize {
            if !q16::is_code(code)
                || q16::attribute_dtype(node, "output_dtype")?.is_some_and(|dtype| dtype != code)
            {
                return Err("output_dtype must match the INT16/UINT16 zero_point dtype".into());
            }
            if opset < 23 && x != scale {
                return Err(format!(
                    "opset {opset} requires x and scale to share a dtype; got {x:?} and {scale:?}"
                ));
            }
            q16::quantize_precision(q16::attribute_dtype(node, "precision")?, *x, *scale)?;
        } else {
            q16::dequantize_output(node, *scale)?;
        }
        Ok(())
    })())
}

#[derive(Clone, Copy, Debug)]
enum LinearQuantOp {
    Quantize,
    Dequantize,
}

struct LinearQuantFactory {
    op: LinearQuantOp,
    runtime: Arc<CudaRuntime>,
    int16: bool,
}

pub(crate) fn register_linear_quantization(
    registry: &mut onnx_runtime_ep_api::OpRegistry,
    runtime: &Arc<CudaRuntime>,
    native: bool,
) {
    use onnx_runtime_ep_api::{OpKey, WorkspacePlanning};
    for version in [10, 13, 19, 21, 23, 25] {
        for (name, op) in [
            ("QuantizeLinear", LinearQuantOp::Quantize),
            ("DequantizeLinear", LinearQuantOp::Dequantize),
        ] {
            registry.register(
                OpKey::new(name, "", version),
                WorkspacePlanning::None,
                Box::new(LinearQuantFactory {
                    op,
                    runtime: runtime.clone(),
                    int16: native && version >= 21,
                }),
            );
        }
    }
}

impl KernelFactory for LinearQuantFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(LinearQuantKernel {
            int16: self.int16,
            op: self.op,
            runtime: self.runtime.clone(),
            node: node.clone(),
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(1),
            block: node
                .attr("block_size")
                .and_then(|a| a.as_int())
                .unwrap_or(0),
        }))
    }
}

struct LinearQuantKernel {
    op: LinearQuantOp,
    runtime: Arc<CudaRuntime>,
    node: Node,
    axis: i64,
    block: i64,
    int16: bool,
}

impl Kernel for LinearQuantKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let name = match self.op {
            LinearQuantOp::Quantize => "QuantizeLinear",
            LinearQuantOp::Dequantize => "DequantizeLinear",
        };
        if !(2..=3).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {name}: expected 2..=3 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }

        if inputs.iter().any(|v| !v.is_absent() && !v.is_contiguous())
            || !outputs[0].is_contiguous()
        {
            return Err(not_implemented(format!("{name} with strided tensors")));
        }
        let x = &inputs[0];
        let scale = &inputs[1];
        let quant_dtype = match self.op {
            LinearQuantOp::Quantize => outputs[0].dtype,
            LinearQuantOp::Dequantize => x.dtype,
        };
        if self.int16 && q16::is_code(quant_dtype) {
            return self.execute_int16(
                name,
                quant_dtype,
                x,
                scale,
                inputs.get(2).filter(|v| !v.is_absent()),
                &mut outputs[0],
            );
        }
        if matches!(self.op, LinearQuantOp::Dequantize)
            && (self.block != 0 || matches!(x.dtype, DataType::Int4 | DataType::Uint4))
        {
            return self.dequantize_affine(
                x,
                scale,
                inputs.get(2).filter(|v| !v.is_absent()),
                &mut outputs[0],
            );
        }
        if scale.dtype != DataType::Float32 || !scale.shape.is_empty() {
            return Err(not_implemented(format!(
                "{name} requires a scalar Float32 scale"
            )));
        }
        if !matches!(quant_dtype, DataType::Int8 | DataType::Uint8) {
            return Err(not_implemented(format!(
                "{name} quantized dtype {quant_dtype:?} (supported: Int8, Uint8)"
            )));
        }
        if let Some(zero_point) = inputs.get(2).filter(|v| !v.is_absent())
            && (zero_point.dtype != quant_dtype || !zero_point.shape.is_empty())
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {name}: zero_point must be a scalar {quant_dtype:?}"
            )));
        }
        match self.op {
            LinearQuantOp::Quantize
                if x.dtype != DataType::Float32 || outputs[0].shape != x.shape =>
            {
                return Err(EpError::KernelFailed(
                    "cuda_ep QuantizeLinear: Float32 input and same-shape output required".into(),
                ));
            }
            LinearQuantOp::Dequantize
                if outputs[0].dtype != DataType::Float32 || outputs[0].shape != x.shape =>
            {
                return Err(EpError::KernelFailed(
                    "cuda_ep DequantizeLinear: Float32 same-shape output required".into(),
                ));
            }
            _ => {}
        }
        let n = x.numel() as u64;
        if n == 0 {
            return Ok(());
        }
        let stem = match (self.op, quant_dtype) {
            (LinearQuantOp::Quantize, DataType::Uint8) => "quantize_u8",
            (LinearQuantOp::Quantize, DataType::Int8) => "quantize_i8",
            (LinearQuantOp::Dequantize, DataType::Uint8) => "dequantize_u8",
            (LinearQuantOp::Dequantize, DataType::Int8) => "dequantize_i8",
            _ => unreachable!(),
        };
        let function = self
            .runtime
            .nvrtc_function("linear_quant_per_tensor_v1", SOURCE, stem)?;
        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let scale_ptr = cuptr(scale.data_ptr::<u8>() as *const c_void);
        let zero_point_ptr = inputs
            .get(2)
            .filter(|v| !v.is_absent())
            .map_or(0, |v| cuptr(v.data_ptr::<u8>() as *const c_void));
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&x_ptr)
            .arg(&scale_ptr)
            .arg(&zero_point_ptr)
            .arg(&y_ptr)
            .arg(&n);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (n.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err(&format!("launch {name}"), error))?;
        Ok(())
    }
}

impl LinearQuantKernel {
    #[allow(clippy::too_many_arguments)]
    fn execute_int16(
        &self,
        name: &str,
        code: DataType,
        x: &TensorView,
        scale: &TensorView,
        zp: Option<&TensorView>,
        y: &mut TensorMut,
    ) -> Result<()> {
        let invalid = |detail| EpError::KernelFailed(format!("cuda_ep {name}: {detail}"));
        if y.shape != x.shape || zp.is_some_and(|z| z.dtype != code || z.shape != scale.shape) {
            return Err(invalid(format!(
                "output shape {:?} must match x {:?}; zero_point must be {code:?} of scale shape {:?}",
                y.shape, x.shape, scale.shape
            )));
        }
        let (stem, precision) = match self.op {
            LinearQuantOp::Quantize => {
                let expected = q16::attribute_dtype(&self.node, "output_dtype").map_err(invalid)?;
                if expected.is_some_and(|dtype| dtype != code) || expected.is_none() && zp.is_none()
                {
                    return Err(invalid(format!(
                        "output_dtype or zero_point must select {code:?}"
                    )));
                }
                let precision = q16::attribute_dtype(&self.node, "precision").map_err(invalid)?;
                let precision =
                    q16::quantize_precision(precision, x.dtype, scale.dtype).map_err(invalid)?;
                ("quantize16", precision)
            }
            LinearQuantOp::Dequantize => {
                let expected = q16::dequantize_output(&self.node, scale.dtype).map_err(invalid)?;
                if y.dtype != expected {
                    return Err(invalid(format!(
                        "output dtype {:?} must be {expected:?}",
                        y.dtype
                    )));
                }
                ("dequantize16", DataType::Undefined)
            }
        };
        let block = u64::try_from(self.block)
            .map_err(|_| invalid(format!("block_size={} must be nonnegative", self.block)))?;
        // A nonzero inner identifies per-axis or blocked indexing. Empty data
        // never launches a kernel, even when its inner product is zero.
        let (axis, inner) = if block == 0 && scale.shape.is_empty() {
            (0, 0)
        } else {
            let rank = x.shape.len() as i64;
            if !(-rank..rank).contains(&self.axis) {
                return Err(invalid(format!(
                    "axis {} is out of range for rank {rank}",
                    self.axis
                )));
            }
            let index = self.axis.rem_euclid(rank) as usize;
            let extent = x.shape[index] as u64;
            let expected = if block == 0 {
                vec![x.shape[index]]
            } else {
                let mut shape = x.shape.to_vec();
                shape[index] = extent.div_ceil(block) as usize;
                shape
            };
            if scale.shape != expected {
                return Err(invalid(format!(
                    "scale shape {:?} must be {expected:?} for axis {} and block_size {block}",
                    scale.shape, self.axis
                )));
            }
            (
                extent,
                x.shape[index + 1..].iter().product::<usize>() as u64,
            )
        };
        let n = x.numel() as u64;
        if n == 0 {
            return Ok(());
        }
        let function = self.runtime.nvrtc_function(
            "linear_quant_int16_v1",
            include_str!("quantization_int16.cu"),
            stem,
        )?;
        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let scale_ptr = cuptr(scale.data_ptr::<u8>() as *const c_void);
        let zp_ptr = zp.map_or(0, |v| cuptr(v.data_ptr::<u8>() as *const c_void));
        let y_ptr = cuptr(y.data_ptr_mut::<u8>() as *const c_void);
        let data_type = if matches!(self.op, LinearQuantOp::Quantize) {
            x.dtype
        } else {
            y.dtype
        } as i32;
        let scale_type = scale.dtype as i32;
        let precision = precision as i32;
        let signed = i32::from(code == DataType::Int16);
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&x_ptr)
            .arg(&scale_ptr)
            .arg(&zp_ptr)
            .arg(&y_ptr)
            .arg(&n)
            .arg(&inner)
            .arg(&axis)
            .arg(&block)
            .arg(&data_type)
            .arg(&scale_type)
            .arg(&precision)
            .arg(&signed);
        // SAFETY: checked dtype/shape/contiguity bounds all kernel accesses.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (n.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| driver_err(&format!("launch {name} INT16/UINT16"), error))
    }

    fn dequantize_affine(
        &self,
        x: &TensorView,
        scale: &TensorView,
        zp: Option<&TensorView>,
        y: &mut TensorMut,
    ) -> Result<()> {
        let (bits, sign) = match x.dtype {
            DataType::Int4 => (4i32, 1i32),
            DataType::Uint4 => (4, 0),
            DataType::Int8 => (8, 1),
            DataType::Uint8 => (8, 0),
            dtype => {
                return Err(not_implemented(format!(
                    "blocked DequantizeLinear with {dtype:?}; use INT4, UINT4, INT8 or UINT8"
                )));
            }
        };
        if scale.dtype != DataType::Float32
            || y.dtype != DataType::Float32
            || y.shape != x.shape
            || zp.is_some_and(|z| z.dtype != x.dtype || z.shape != scale.shape)
        {
            return Err(EpError::KernelFailed("cuda_ep DequantizeLinear: expected FP32 scale and same-shape FP32 output; zero point must match weight dtype and scale shape".into()));
        }
        let block = u64::try_from(self.block).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep DequantizeLinear: block_size={} must be nonnegative",
                self.block
            ))
        })?;
        let (axis, inner) = if block == 0 {
            if !scale.shape.is_empty() {
                return Err(not_implemented(
                    "sub-byte DequantizeLinear with non-scalar scale requires positive block_size",
                ));
            }
            (1, 1)
        } else {
            let rank = x.shape.len() as i64;
            if !(-rank..rank).contains(&self.axis) {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep DequantizeLinear: axis {} is outside [-{rank}, {rank})",
                    self.axis
                )));
            }
            let idx = self.axis.rem_euclid(rank) as usize;
            let mut expected = x.shape.to_vec();
            expected[idx] = (x.shape[idx] as u64).div_ceil(block) as usize;
            if scale.shape != expected {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep DequantizeLinear: scale shape {:?} must be {expected:?} for axis {} and block_size {block}",
                    scale.shape, self.axis
                )));
            }
            (
                x.shape[idx] as u64,
                x.shape[idx + 1..].iter().product::<usize>() as u64,
            )
        };
        let n = x.numel() as u64;
        if n == 0 {
            return Ok(());
        }
        let function = self.runtime.nvrtc_function(
            "linear_quant_per_tensor_v1",
            SOURCE,
            "dequantize_affine",
        )?;
        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let scale_ptr = cuptr(scale.data_ptr::<u8>() as *const c_void);
        let zp_ptr = zp.map_or(0, |v| cuptr(v.data_ptr::<u8>() as *const c_void));
        let y_ptr = cuptr(y.data_ptr_mut::<u8>() as *const c_void);
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&x_ptr)
            .arg(&scale_ptr)
            .arg(&zp_ptr)
            .arg(&y_ptr)
            .arg(&n)
            .arg(&inner)
            .arg(&axis)
            .arg(&block)
            .arg(&bits)
            .arg(&sign);
        // SAFETY: validated extents cover the packed codes, metadata and output.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (n.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| driver_err("launch DequantizeLinear affine", e))
    }
}

pub struct DynamicQuantizeLinearFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for DynamicQuantizeLinearFactory {
    fn create(&self, _: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(DynamicQuantizeLinearKernel {
            runtime: self.runtime.clone(),
        }))
    }
}

struct DynamicQuantizeLinearKernel {
    runtime: Arc<CudaRuntime>,
}

impl Kernel for DynamicQuantizeLinearKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 3 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep DynamicQuantizeLinear: expected 1 input and 3 outputs, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let input = &inputs[0];
        if input.dtype != DataType::Float32
            || outputs[0].dtype != DataType::Uint8
            || outputs[1].dtype != DataType::Float32
            || outputs[2].dtype != DataType::Uint8
        {
            return Err(EpError::KernelFailed(
                "cuda_ep DynamicQuantizeLinear: dtypes must be Float32 -> (Uint8, Float32, Uint8)"
                    .into(),
            ));
        }
        if outputs[0].shape != input.shape
            || !outputs[1].shape.is_empty()
            || !outputs[2].shape.is_empty()
        {
            return Err(EpError::KernelFailed(
                "cuda_ep DynamicQuantizeLinear: output shapes must be X shape, scalar, scalar"
                    .into(),
            ));
        }
        if !input.is_contiguous() || outputs.iter().any(|output| !output.is_contiguous()) {
            return Err(not_implemented(
                "DynamicQuantizeLinear with non-contiguous tensors",
            ));
        }
        let function = self.runtime.nvrtc_function(
            "linear_quant_per_tensor_v1",
            SOURCE,
            "dynamic_quantize_u8",
        )?;
        let x = cuptr(input.data_ptr::<u8>() as *const c_void);
        let y = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let scale = cuptr(outputs[1].data_ptr_mut::<u8>() as *const c_void);
        let zero_point = cuptr(outputs[2].data_ptr_mut::<u8>() as *const c_void);
        let n = input.numel() as u64;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder.arg(&x).arg(&y).arg(&scale).arg(&zero_point).arg(&n);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| driver_err("launch DynamicQuantizeLinear", error))
    }
}
