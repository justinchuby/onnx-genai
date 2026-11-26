//! Standard ONNX linear quantization kernels.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::int16_quantization as q16;
use onnx_runtime_ir::{DataType, Node};

use super::{check_arity, to_dense_bytes, write_dense_bytes};

pub struct QuantizeLinearKernel {
    axis: i64,
    block_size: Option<usize>,
    output_dtype: Option<DataType>,
    precision: std::result::Result<Option<DataType>, String>,
    int16: Option<(u64, Node)>,
}

struct QuantizeLinearFactory {
    int16_revision: Option<u64>,
}

pub(crate) fn register_linear_quantization(
    registry: &mut onnx_runtime_ep_api::OpRegistry,
    native: bool,
) {
    use onnx_runtime_ep_api::{OpKey, WorkspacePlanning};
    for version in [10, 13, 19, 21, 23, 25] {
        let int16_revision = native.then_some(version);
        registry.register(
            OpKey::new("QuantizeLinear", "", version),
            WorkspacePlanning::None,
            Box::new(QuantizeLinearFactory { int16_revision }),
        );
        registry.register(
            OpKey::new("DequantizeLinear", "", version),
            WorkspacePlanning::None,
            Box::new(DequantizeLinearFactory { int16_revision }),
        );
    }
}

impl KernelFactory for QuantizeLinearFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(QuantizeLinearKernel {
            int16: self.int16_revision.map(|revision| (revision, node.clone())),
            output_dtype: q16::attribute_dtype(node, "output_dtype")
                .ok()
                .flatten()
                .filter(|&dtype| q16::is_code(dtype)),
            precision: q16::attribute_dtype(node, "precision"),
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(1),
            block_size: node
                .attr("block_size")
                .and_then(|a| a.as_int())
                .filter(|&n| n > 0)
                .map(|n| n as usize),
        }))
    }
}

pub struct DequantizeLinearKernel {
    axis: i64,
    block_size: Option<usize>,
    output_dtype: Option<DataType>,
    int16: Option<(u64, Node)>,
}

struct DequantizeLinearFactory {
    int16_revision: Option<u64>,
}

impl KernelFactory for DequantizeLinearFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(DequantizeLinearKernel {
            int16: self.int16_revision.map(|revision| (revision, node.clone())),
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(1),
            block_size: node
                .attr("block_size")
                .and_then(|a| a.as_int())
                .filter(|&n| n > 0)
                .map(|n| n as usize),
            output_dtype: match node
                .attr("output_dtype")
                .and_then(|a| a.as_int())
                .unwrap_or(0)
            {
                0 => None,
                1 => Some(DataType::Float32),
                10 => Some(DataType::Float16),
                16 => Some(DataType::BFloat16),
                dtype => {
                    return Err(EpError::KernelFailed(format!(
                        "DequantizeLinear: output_dtype={dtype} is unsupported on CPU; use FLOAT, FLOAT16 or BFLOAT16"
                    )));
                }
            },
        }))
    }
}

pub struct DynamicQuantizeLinearKernel;
pub struct DynamicQuantizeLinearFactory;

impl KernelFactory for DynamicQuantizeLinearFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(DynamicQuantizeLinearKernel))
    }
}

impl Kernel for QuantizeLinearKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("QuantizeLinear", inputs, outputs, 2, 3, 1)?;
        if let Some((revision, node)) = &self.int16
            && q16::is_code(outputs[0].dtype)
        {
            return self.quantize_int16(*revision, node, inputs, &mut outputs[0]);
        }
        let x = read_floats("QuantizeLinear", &inputs[0])?;
        let scale = read_floats("QuantizeLinear", &inputs[1])?;
        let zp_dtype = if inputs.len() == 3 && !inputs[2].is_absent() {
            inputs[2].dtype
        } else {
            DataType::Uint8
        };
        if outputs[0].dtype != zp_dtype {
            return Err(EpError::KernelFailed(format!(
                "QuantizeLinear: output dtype {:?} must match zero_point dtype {zp_dtype:?}",
                outputs[0].dtype
            )));
        }
        let zp = if inputs.len() == 3 && !inputs[2].is_absent() {
            read_integers("QuantizeLinear", &inputs[2], false)?
        } else {
            vec![0]
        };
        let params = Params::new(
            "QuantizeLinear",
            inputs[0].shape,
            inputs[1].shape,
            &scale,
            inputs.get(2).filter(|v| !v.is_absent()).map(|v| v.shape),
            &zp,
            self.axis,
            self.block_size,
        )?;
        let mut bytes = Vec::with_capacity(x.len() * zp_dtype.byte_size());
        for (i, value) in x.into_iter().enumerate() {
            let p = params.at(i);
            let quantized =
                (value / scale[p]).round_ties_even() as i64 + zp[if zp.len() == 1 { 0 } else { p }];
            write_integer(&mut bytes, zp_dtype, quantized)?;
        }
        write_dense_bytes(&mut outputs[0], &bytes)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

impl QuantizeLinearKernel {
    fn quantize_int16(
        &self,
        revision: u64,
        node: &Node,
        inputs: &[TensorView],
        out: &mut TensorMut,
    ) -> Result<()> {
        let op = "QuantizeLinear";
        let invalid = |detail| EpError::KernelFailed(format!("{op}: {detail}"));
        q16::validate_revision(node, revision, Some(inputs[1].shape.len())).map_err(invalid)?;
        q16::validate_quantize_types(revision, inputs[0].dtype, inputs[1].dtype)
            .map_err(invalid)?;
        let zp = inputs.get(2).filter(|v| !v.is_absent());
        let expected = zp
            .map(|z| z.dtype)
            .or(self.output_dtype)
            .unwrap_or(DataType::Uint8);
        if out.dtype != expected || self.output_dtype.is_some_and(|dtype| dtype != expected) {
            return Err(invalid(format!(
                "output dtype {:?}, zero_point dtype {:?} and output_dtype {:?} must agree",
                out.dtype,
                zp.map(|z| z.dtype),
                self.output_dtype
            )));
        }
        let precision = q16::quantize_precision(
            *self
                .precision
                .as_ref()
                .map_err(|detail| invalid(detail.clone()))?,
            inputs[0].dtype,
            inputs[1].dtype,
        )
        .map_err(invalid)?;
        let read = |view: &TensorView| -> Result<Vec<f64>> {
            if view.dtype == DataType::Int32 {
                Ok(read_integers(op, view, false)?
                    .into_iter()
                    .map(|value| value as f64)
                    .collect())
            } else {
                Ok(read_floats(op, view)?.into_iter().map(f64::from).collect())
            }
        };
        let x = read(&inputs[0])?;
        let scale = read(&inputs[1])?;
        let zeros = zp.map_or_else(|| Ok(vec![0]), |view| read_integers(op, view, true))?;
        let params = Params::new(
            op,
            inputs[0].shape,
            inputs[1].shape,
            &scale,
            zp.map(|v| v.shape),
            &zeros,
            self.axis,
            self.block_size,
        )?;
        let narrow = |value: f64| match precision {
            DataType::Float16 => half::f16::from_f64(value).to_f64(),
            DataType::BFloat16 => half::bf16::from_f64(value).to_f64(),
            _ => value,
        };
        let mut bytes = Vec::with_capacity(x.len() * out.dtype.byte_size());
        for (i, value) in x.into_iter().enumerate() {
            let p = params.at(i);
            let quotient = if precision == DataType::Float32 {
                f64::from(value as f32 / scale[p] as f32)
            } else {
                narrow(narrow(value) / narrow(scale[p]))
            };
            let code = (quotient.round_ties_even() as i64)
                .saturating_add(zeros[if zp.is_some() { p } else { 0 }]);
            write_integer(&mut bytes, out.dtype, code)?;
        }
        write_dense_bytes(out, &bytes)
    }
}

impl Kernel for DequantizeLinearKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("DequantizeLinear", inputs, outputs, 2, 3, 1)?;
        let accepts_int16 = self.int16.is_some();
        if let Some((revision, node)) = &self.int16
            && q16::is_code(inputs[0].dtype)
        {
            let invalid = |detail| EpError::KernelFailed(format!("DequantizeLinear: {detail}"));
            q16::validate_revision(node, *revision, Some(inputs[1].shape.len()))
                .map_err(invalid)?;
            q16::validate_dequantize_scale(*revision, inputs[1].dtype).map_err(invalid)?;
        }
        let x = read_integers("DequantizeLinear", &inputs[0], accepts_int16)?;
        let scale = read_floats("DequantizeLinear", &inputs[1])?;
        let output_dtype = self.output_dtype.unwrap_or(inputs[1].dtype);
        if outputs[0].dtype != output_dtype {
            return Err(EpError::KernelFailed(format!(
                "DequantizeLinear: output dtype {:?} must match resolved dtype {output_dtype:?}",
                outputs[0].dtype
            )));
        }
        let zp = if inputs.len() == 3 && !inputs[2].is_absent() {
            if inputs[2].dtype != inputs[0].dtype {
                return Err(EpError::KernelFailed(format!(
                    "DequantizeLinear: zero_point dtype {:?} must match input dtype {:?}",
                    inputs[2].dtype, inputs[0].dtype
                )));
            }
            read_integers("DequantizeLinear", &inputs[2], accepts_int16)?
        } else {
            vec![0]
        };
        let params = Params::new(
            "DequantizeLinear",
            inputs[0].shape,
            inputs[1].shape,
            &scale,
            inputs.get(2).filter(|v| !v.is_absent()).map(|v| v.shape),
            &zp,
            self.axis,
            self.block_size,
        )?;
        let out: Vec<f32> = x
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                let p = params.at(i);
                (value - zp[if zp.len() == 1 { 0 } else { p }]) as f32 * scale[p]
            })
            .collect();
        write_floats(&mut outputs[0], &out)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

impl Kernel for DynamicQuantizeLinearKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("DynamicQuantizeLinear", inputs, outputs, 1, 1, 3)?;
        if inputs[0].dtype != DataType::Float32 {
            return Err(EpError::KernelFailed(format!(
                "DynamicQuantizeLinear: input dtype must be Float32 per the ONNX schema, got {:?}; \
                 cast the input to Float32 or use QuantizeLinear with a BFloat16 scale",
                inputs[0].dtype,
            )));
        }
        if outputs[0].dtype != DataType::Uint8
            || outputs[1].dtype != DataType::Float32
            || outputs[2].dtype != DataType::Uint8
        {
            return Err(EpError::KernelFailed(
                "DynamicQuantizeLinear: output dtypes must be Uint8, Float32, Uint8".into(),
            ));
        }
        let x = read_floats("DynamicQuantizeLinear", &inputs[0])?;
        let (mut min, mut max) = (0.0f32, 0.0f32);
        for &value in &x {
            min = min.min(value);
            max = max.max(value);
        }
        let scale = (max - min) / 255.0;
        let scale = if scale == 0.0 { 1.0 } else { scale };
        let zp = (-min / scale).round_ties_even().clamp(0.0, 255.0) as u8;
        let y: Vec<u8> = x
            .iter()
            .map(|&value| ((value / scale).round_ties_even() + zp as f32).clamp(0.0, 255.0) as u8)
            .collect();
        write_dense_bytes(&mut outputs[0], &y)?;
        write_floats(&mut outputs[1], &[scale])?;
        write_dense_bytes(&mut outputs[2], &[zp])
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

struct Params {
    axis_dim: usize,
    inner: usize,
    block_size: usize,
    count: usize,
    blocked: bool,
}

impl Params {
    #[allow(clippy::too_many_arguments)]
    fn new<T: Into<f64> + Copy>(
        op: &str,
        x_shape: &[usize],
        scale_shape: &[usize],
        scale: &[T],
        zp_shape: Option<&[usize]>,
        zp: &[i64],
        axis: i64,
        block_size: Option<usize>,
    ) -> Result<Self> {
        if scale.iter().any(|&s| {
            let s = s.into();
            s <= 0.0 || !s.is_finite()
        }) {
            return Err(EpError::KernelFailed(format!(
                "{op}: scale values must be finite and positive"
            )));
        }
        if zp.len() != 1 && zp.len() != scale.len() {
            return Err(EpError::KernelFailed(format!(
                "{op}: zero_point must be scalar or have the same length as scale"
            )));
        }
        if let Some(shape) = zp_shape
            && shape != scale_shape
        {
            return Err(EpError::KernelFailed(format!(
                "{op}: zero_point shape {shape:?} must match scale shape {scale_shape:?}"
            )));
        }
        if block_size.is_none() && scale_shape.is_empty() && scale.len() == 1 {
            return Ok(Self {
                axis_dim: 1,
                inner: 1,
                block_size: 1,
                count: 1,
                blocked: false,
            });
        }
        let rank = x_shape.len();
        let normalized_axis = if axis < 0 { axis + rank as i64 } else { axis };
        if normalized_axis < 0 || normalized_axis as usize >= rank {
            return Err(EpError::KernelFailed(format!(
                "{op}: axis {axis} is out of range for rank {rank}"
            )));
        }
        let axis = normalized_axis as usize;
        let axis_dim = x_shape[axis];
        let blocked = block_size.is_some();
        let block_size = block_size.unwrap_or(1);
        let expected = axis_dim.div_ceil(block_size);
        let mut expected_shape = if blocked {
            x_shape.to_vec()
        } else {
            vec![axis_dim]
        };
        if blocked {
            expected_shape[axis] = expected;
        }
        if scale_shape != expected_shape {
            return Err(EpError::KernelFailed(format!(
                "{op}: scale shape {scale_shape:?} must be {expected_shape:?} for axis {axis} and block_size {block_size}"
            )));
        }
        Ok(Self {
            axis_dim,
            inner: x_shape[axis + 1..].iter().product(),
            block_size,
            count: scale.len(),
            blocked,
        })
    }

    fn at(&self, linear_index: usize) -> usize {
        if self.count == 1 {
            0
        } else if self.blocked {
            let outer = linear_index / self.inner / self.axis_dim;
            let axis = (linear_index / self.inner) % self.axis_dim;
            (outer * self.axis_dim.div_ceil(self.block_size) + axis / self.block_size) * self.inner
                + linear_index % self.inner
        } else {
            ((linear_index / self.inner) % self.axis_dim) / self.block_size
        }
    }
}

fn read_floats(op: &str, view: &TensorView) -> Result<Vec<f32>> {
    let bytes = to_dense_bytes(view)?;
    Ok(match view.dtype {
        DataType::Float32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        DataType::Float16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::f16::from_le_bytes(*b).to_f32())
            .collect(),
        DataType::BFloat16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::bf16::from_le_bytes(*b).to_f32())
            .collect(),
        other => {
            return Err(EpError::KernelFailed(format!(
                "{op}: only Float32, Float16, and BFloat16 inputs are supported, got {other:?}"
            )));
        }
    })
}

fn read_integers(op: &str, view: &TensorView, int16: bool) -> Result<Vec<i64>> {
    if matches!(view.dtype, DataType::Int4 | DataType::Uint4) {
        view.validate()?;
        if !view.device.is_host_accessible() || !view.is_contiguous() {
            return Err(EpError::KernelFailed(format!(
                "{op}: sub-byte inputs must be contiguous and host-accessible"
            )));
        }
        // SAFETY: validated contiguous packed storage contains ceil(numel/2) bytes.
        let bytes = unsafe { std::slice::from_raw_parts(view.data_ptr::<u8>(), view.byte_size()) };
        return Ok((0..view.numel())
            .map(|i| {
                let code = (bytes[i / 2] >> ((i % 2) * 4)) & 15;
                if view.dtype == DataType::Int4 {
                    i64::from((code ^ 8) as i8 - 8)
                } else {
                    i64::from(code)
                }
            })
            .collect());
    }
    let bytes = to_dense_bytes(view)?;
    Ok(match view.dtype {
        DataType::Int8 => bytes.iter().map(|&b| b as i8 as i64).collect(),
        DataType::Uint8 => bytes.into_iter().map(i64::from).collect(),
        DataType::Int16 if int16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| i64::from(i16::from_le_bytes(*b)))
            .collect(),
        DataType::Uint16 if int16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| i64::from(u16::from_le_bytes(*b)))
            .collect(),
        DataType::Int32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b) as i64)
            .collect(),
        other => {
            let extra = if int16 { "Int16, Uint16, " } else { "" };
            return Err(EpError::KernelFailed(format!(
                "{op}: expected Int4, Uint4, Int8, Uint8, {extra}or Int32 quantized tensors, got {other:?}"
            )));
        }
    })
}

fn write_floats(out: &mut TensorMut, data: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(data.len() * out.dtype.byte_size());
    match out.dtype {
        DataType::Float32 => {
            for &value in data {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        DataType::Float16 => {
            for &value in data {
                bytes.extend_from_slice(&half::f16::from_f32(value).to_le_bytes());
            }
        }
        DataType::BFloat16 => {
            for &value in data {
                bytes.extend_from_slice(&half::bf16::from_f32(value).to_le_bytes());
            }
        }
        other => {
            return Err(EpError::KernelFailed(format!(
                "quantization: floating output must be Float32, Float16, or BFloat16, got {other:?}"
            )));
        }
    }
    write_dense_bytes(out, &bytes)
}

fn write_integer(bytes: &mut Vec<u8>, dtype: DataType, value: i64) -> Result<()> {
    match dtype {
        DataType::Int8 => bytes.push(value.clamp(i8::MIN as i64, i8::MAX as i64) as i8 as u8),
        DataType::Uint8 => bytes.push(value.clamp(0, u8::MAX as i64) as u8),
        DataType::Int16 => bytes.extend_from_slice(
            &(value.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16).to_le_bytes(),
        ),
        DataType::Uint16 => {
            bytes.extend_from_slice(&(value.clamp(0, i64::from(u16::MAX)) as u16).to_le_bytes())
        }
        DataType::Int32 => bytes.extend_from_slice(
            &(value.clamp(i32::MIN as i64, i32::MAX as i64) as i32).to_le_bytes(),
        ),
        other => {
            return Err(EpError::KernelFailed(format!(
                "QuantizeLinear: unsupported output dtype {other:?}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::ExecutionProvider;

    use crate::kernels::testutil::Owned;

    #[test]
    fn ort_plugin_keeps_legacy_linear_quantization_factories() {
        let native = crate::CpuExecutionProvider::new();
        let plugin = crate::CpuExecutionProvider::for_ort_plugin();
        let q = Node::new(onnx_runtime_ir::NodeId(0), "QuantizeLinear", vec![], vec![]);
        let dq = Node::new(
            onnx_runtime_ir::NodeId(1),
            "DequantizeLinear",
            vec![],
            vec![],
        );
        let scale = Owned::f32(&[], &[0.5]);
        let x = Owned::f32(&[2], &[1.0, 2.0]);

        for dtype in [DataType::Uint8, DataType::Int16, DataType::Uint16] {
            assert!(super::super::supported_dtypes_for_op("QuantizeLinear", "").contains(&dtype));
            assert_eq!(
                super::super::supported_dtypes_for_op("DequantizeLinear", "").contains(&dtype),
                dtype == DataType::Uint8
            );
            let zp = Owned::zeros(dtype, &[]);
            for (provider, opset, accepts_int16) in [
                (&plugin, 21, false),
                (&native, 10, true),
                (&native, 19, true),
                (&native, 21, true),
            ] {
                let accepted = dtype == DataType::Uint8 || accepts_int16;
                let mut codes = Owned::zeros(dtype, &[2]);
                let result = provider.get_kernel(&q, &[], opset).unwrap().execute(
                    &[x.view(), scale.view(), zp.view()],
                    &mut [codes.view_mut()],
                );
                assert_eq!(
                    result.is_ok(),
                    accepted,
                    "Q {dtype:?} at opset {opset}: {result:?}"
                );
                let mut values = Owned::zeros_f32(&[2]);
                let result = provider.get_kernel(&dq, &[], opset).unwrap().execute(
                    &[codes.view(), scale.view(), zp.view()],
                    &mut [values.view_mut()],
                );
                assert_eq!(
                    result.is_ok(),
                    accepted,
                    "DQ {dtype:?} at opset {opset}: {result:?}"
                );
                if accepted {
                    assert_eq!(values.bytes, x.bytes);
                }
            }
        }
    }

    #[test]
    fn int16_factories_validate_runtime_rank_without_compile_shapes() {
        let provider = crate::CpuExecutionProvider::new();
        let x = Owned::f32(&[2, 2], &[1.; 4]);
        let scale = Owned::f32(&[2], &[0.5; 2]);
        let zp = Owned::zeros(DataType::Int16, &[2]);
        let mut codes = Owned::zeros(DataType::Int16, &[2, 2]);
        let mut values = Owned::zeros_f32(&[2, 2]);
        let q = Node::new(onnx_runtime_ir::NodeId(0), "QuantizeLinear", vec![], vec![]);
        let dq = Node::new(
            onnx_runtime_ir::NodeId(1),
            "DequantizeLinear",
            vec![],
            vec![],
        );
        let result = provider.get_kernel(&q, &[], 10).unwrap().execute(
            &[x.view(), scale.view(), zp.view()],
            &mut [codes.view_mut()],
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("requires a scalar scale")
        );
        let result = provider.get_kernel(&dq, &[], 10).unwrap().execute(
            &[codes.view(), scale.view(), zp.view()],
            &mut [values.view_mut()],
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("requires a scalar scale")
        );
    }

    fn i8(shape: &[usize], data: &[i8]) -> Owned {
        Owned {
            bytes: data.iter().map(|&v| v as u8).collect(),
            shape: shape.to_vec(),
            strides: onnx_runtime_ir::compute_contiguous_strides(shape),
            dtype: DataType::Int8,
        }
    }

    #[test]
    fn dequantize_per_tensor_and_default_zero_point() {
        let x = i8(&[3], &[-2, 0, 3]);
        let scale = Owned::f32(&[], &[0.5]);
        let mut out = Owned::zeros_f32(&[3]);
        DequantizeLinearKernel {
            int16: None,
            output_dtype: None,
            axis: 1,
            block_size: None,
        }
        .execute(&[x.view(), scale.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![-1.0, 0.0, 1.5]);
    }

    #[test]
    fn dequantize_per_axis_uint8() {
        let x = Owned::u8(&[2, 2], &[2, 4, 6, 8]);
        let scale = Owned::f32(&[2], &[0.5, 0.25]);
        let zp = Owned::u8(&[2], &[2, 4]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        DequantizeLinearKernel {
            int16: None,
            output_dtype: None,
            axis: 1,
            block_size: None,
        }
        .execute(&[x.view(), scale.view(), zp.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![0.0, 0.0, 2.0, 1.0]);
    }

    #[test]
    fn quantize_ties_saturates_and_defaults_to_uint8() {
        let x = Owned::f32(&[6], &[-1.0, 0.5, 1.5, 2.5, 300.0, -300.0]);
        let scale = Owned::f32(&[], &[1.0]);
        let mut out = Owned::zeros(DataType::Uint8, &[6]);
        QuantizeLinearKernel {
            int16: None,
            output_dtype: None,
            precision: Ok(None),
            axis: 1,
            block_size: None,
        }
        .execute(&[x.view(), scale.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_u8(), vec![0, 0, 2, 2, 255, 0]);
    }

    #[test]
    fn quantize_per_axis_and_round_trip() {
        let x = Owned::f32(&[2, 2], &[1.0, 1.0, 2.0, 2.0]);
        let scale = Owned::f32(&[2], &[0.5, 0.25]);
        let zp = Owned::u8(&[2], &[2, 4]);
        let mut q = Owned::zeros(DataType::Uint8, &[2, 2]);
        QuantizeLinearKernel {
            int16: None,
            output_dtype: None,
            precision: Ok(None),
            axis: 1,
            block_size: None,
        }
        .execute(&[x.view(), scale.view(), zp.view()], &mut [q.view_mut()])
        .unwrap();
        assert_eq!(q.to_u8(), vec![4, 8, 6, 12]);
        let mut round_trip = Owned::zeros_f32(&[2, 2]);
        DequantizeLinearKernel {
            int16: None,
            output_dtype: None,
            axis: 1,
            block_size: None,
        }
        .execute(
            &[q.view(), scale.view(), zp.view()],
            &mut [round_trip.view_mut()],
        )
        .unwrap();
        assert_eq!(round_trip.to_f32(), x.to_f32());
    }

    #[test]
    fn dequantize_blocked_axis() {
        let x = Owned::u8(&[1, 4], &[2, 4, 6, 8]);
        let scale = Owned::f32(&[1, 2], &[0.5, 0.25]);
        let zp = Owned::u8(&[1, 2], &[2, 4]);
        let mut out = Owned::zeros_f32(&[1, 4]);
        DequantizeLinearKernel {
            int16: None,
            output_dtype: None,
            axis: 1,
            block_size: Some(2),
        }
        .execute(&[x.view(), scale.view(), zp.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![0.0, 1.0, 0.5, 1.0]);
    }

    #[test]
    fn quantize_bfloat16_decode_and_prefill_match_widened_reference() {
        for rows in [1, 4] {
            let values: Vec<f32> = (0..rows * 8)
                .map(|index| ((index * 7 % 23) as f32 - 11.0) * 0.1875)
                .collect();
            let input = Owned::bf16(&[rows, 8], &values);
            let scale = Owned::bf16(&[], &[0.25]);
            let zero_point = Owned::u8(&[], &[113]);
            let mut output = Owned::zeros(DataType::Uint8, &[rows, 8]);
            QuantizeLinearKernel {
                int16: None,
                output_dtype: None,
                precision: Ok(None),
                axis: 1,
                block_size: None,
            }
            .execute(
                &[input.view(), scale.view(), zero_point.view()],
                &mut [output.view_mut()],
            )
            .unwrap();
            let scale = scale.to_bf16_as_f32()[0];
            let expected: Vec<u8> = input
                .to_bf16_as_f32()
                .into_iter()
                .map(|value| ((value / scale).round_ties_even() as i64 + 113).clamp(0, 255) as u8)
                .collect();
            assert_eq!(output.to_u8(), expected);
        }
    }

    #[test]
    fn dequantize_bfloat16_decode_and_prefill_match_widened_reference() {
        for rows in [1, 4] {
            let quantized: Vec<u8> = (0..rows * 8)
                .map(|index| 100 + (index * 5 % 29) as u8)
                .collect();
            let input = Owned::u8(&[rows, 8], &quantized);
            let scale = Owned::bf16(&[], &[0.03125]);
            let zero_point = Owned::u8(&[], &[113]);
            let mut output = Owned::zeros(DataType::BFloat16, &[rows, 8]);
            DequantizeLinearKernel {
                int16: None,
                output_dtype: None,
                axis: 1,
                block_size: None,
            }
            .execute(
                &[input.view(), scale.view(), zero_point.view()],
                &mut [output.view_mut()],
            )
            .unwrap();
            let scale = scale.to_bf16_as_f32()[0];
            let expected: Vec<f32> = quantized
                .iter()
                .map(|&value| (i64::from(value) - 113) as f32 * scale)
                .collect();
            for (actual, expected) in output.to_bf16_as_f32().into_iter().zip(expected) {
                let tolerance = 5e-4 + 1e-2 * expected.abs();
                assert!((actual - expected).abs() <= tolerance);
            }
        }
    }

    #[test]
    fn dynamic_quantize_uses_zero_in_range() {
        let x = Owned::f32(&[3], &[-2.0, 2.0, 6.0]);
        let mut y = Owned::zeros(DataType::Uint8, &[3]);
        let mut scale = Owned::zeros_f32(&[]);
        let mut zp = Owned::zeros(DataType::Uint8, &[]);
        DynamicQuantizeLinearKernel
            .execute(
                &[x.view()],
                &mut [y.view_mut(), scale.view_mut(), zp.view_mut()],
            )
            .unwrap();
        assert_eq!(scale.to_f32(), vec![8.0 / 255.0]);
        assert_eq!(zp.to_u8(), vec![64]);
        assert_eq!(y.to_u8(), vec![0, 128, 255]);
    }

    #[test]
    fn dynamic_quantize_rejects_bfloat16_per_schema() {
        let input = Owned::bf16(&[3], &[-2.0, 2.0, 6.0]);
        let mut quantized = Owned::zeros(DataType::Uint8, &[3]);
        let mut scale = Owned::zeros_f32(&[]);
        let mut zero_point = Owned::zeros(DataType::Uint8, &[]);
        let error = DynamicQuantizeLinearKernel
            .execute(
                &[input.view()],
                &mut [
                    quantized.view_mut(),
                    scale.view_mut(),
                    zero_point.view_mut(),
                ],
            )
            .unwrap_err();
        assert!(format!("{error}").contains("Float32 per the ONNX schema"));
    }
}
