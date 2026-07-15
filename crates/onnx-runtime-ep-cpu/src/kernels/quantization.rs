//! Standard ONNX linear quantization kernels.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::{check_arity, to_dense_bytes, write_dense_bytes};

pub struct QuantizeLinearKernel {
    axis: i64,
    block_size: Option<usize>,
}

pub struct QuantizeLinearFactory;

impl KernelFactory for QuantizeLinearFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(QuantizeLinearKernel {
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
}

pub struct DequantizeLinearFactory;

impl KernelFactory for DequantizeLinearFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(DequantizeLinearKernel {
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(1),
            block_size: node
                .attr("block_size")
                .and_then(|a| a.as_int())
                .filter(|&n| n > 0)
                .map(|n| n as usize),
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
            read_integers("QuantizeLinear", &inputs[2])?
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

impl Kernel for DequantizeLinearKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("DequantizeLinear", inputs, outputs, 2, 3, 1)?;
        let x = read_integers("DequantizeLinear", &inputs[0])?;
        let scale = read_floats("DequantizeLinear", &inputs[1])?;
        if outputs[0].dtype != inputs[1].dtype {
            return Err(EpError::KernelFailed(format!(
                "DequantizeLinear: output dtype {:?} must match scale dtype {:?}",
                outputs[0].dtype, inputs[1].dtype
            )));
        }
        let zp = if inputs.len() == 3 && !inputs[2].is_absent() {
            read_integers("DequantizeLinear", &inputs[2])?
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
}

impl Params {
    #[allow(clippy::too_many_arguments)]
    fn new(
        op: &str,
        x_shape: &[usize],
        scale_shape: &[usize],
        scale: &[f32],
        zp_shape: Option<&[usize]>,
        zp: &[i64],
        axis: i64,
        block_size: Option<usize>,
    ) -> Result<Self> {
        if scale_shape.len() > 1 || (scale_shape.is_empty() && scale.len() != 1) {
            return Err(EpError::KernelFailed(format!(
                "{op}: scale must be a scalar or 1-D tensor"
            )));
        }
        if scale.iter().any(|&s| s <= 0.0 || !s.is_finite()) {
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
            && shape.len() > 1
        {
            return Err(EpError::KernelFailed(format!(
                "{op}: zero_point must be a scalar or 1-D tensor"
            )));
        }
        if scale.len() == 1 {
            return Ok(Self {
                axis_dim: 1,
                inner: 1,
                block_size: 1,
                count: 1,
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
        let block_size = block_size.unwrap_or(1);
        let expected = axis_dim.div_ceil(block_size);
        if scale.len() != expected {
            return Err(EpError::KernelFailed(format!(
                "{op}: scale length {} does not match expected blocked axis length {expected}",
                scale.len()
            )));
        }
        Ok(Self {
            axis_dim,
            inner: x_shape[axis + 1..].iter().product(),
            block_size,
            count: scale.len(),
        })
    }

    fn at(&self, linear_index: usize) -> usize {
        if self.count == 1 {
            0
        } else {
            ((linear_index / self.inner) % self.axis_dim) / self.block_size
        }
    }
}

fn read_floats(op: &str, view: &TensorView) -> Result<Vec<f32>> {
    let bytes = to_dense_bytes(view)?;
    Ok(match view.dtype {
        DataType::Float32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        DataType::Float16 => bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes(b.try_into().unwrap()).to_f32())
            .collect(),
        other => {
            return Err(EpError::KernelFailed(format!(
                "{op}: only Float32 and Float16 inputs are supported, got {other:?}"
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
        other => {
            return Err(EpError::KernelFailed(format!(
                "quantization: floating output must be Float32 or Float16, got {other:?}"
            )));
        }
    }
    write_dense_bytes(out, &bytes)
}

fn read_integers(op: &str, view: &TensorView) -> Result<Vec<i64>> {
    let bytes = to_dense_bytes(view)?;
    Ok(match view.dtype {
        DataType::Int8 => bytes.iter().map(|&b| b as i8 as i64).collect(),
        DataType::Uint8 => bytes.into_iter().map(i64::from).collect(),
        DataType::Int32 => bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as i64)
            .collect(),
        other => {
            return Err(EpError::KernelFailed(format!(
                "{op}: only Int8, Uint8, and Int32 quantized tensors are supported, got {other:?}"
            )));
        }
    })
}

fn write_integer(bytes: &mut Vec<u8>, dtype: DataType, value: i64) -> Result<()> {
    match dtype {
        DataType::Int8 => bytes.push(value.clamp(i8::MIN as i64, i8::MAX as i64) as i8 as u8),
        DataType::Uint8 => bytes.push(value.clamp(0, u8::MAX as i64) as u8),
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
    use crate::kernels::testutil::Owned;

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
            axis: 1,
            block_size: None,
        }
        .execute(&[x.view(), scale.view(), zp.view()], &mut [q.view_mut()])
        .unwrap();
        assert_eq!(q.to_u8(), vec![4, 8, 6, 12]);
        let mut round_trip = Owned::zeros_f32(&[2, 2]);
        DequantizeLinearKernel {
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
        let scale = Owned::f32(&[2], &[0.5, 0.25]);
        let zp = Owned::u8(&[2], &[2, 4]);
        let mut out = Owned::zeros_f32(&[1, 4]);
        DequantizeLinearKernel {
            axis: 1,
            block_size: Some(2),
        }
        .execute(&[x.view(), scale.view(), zp.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![0.0, 1.0, 0.5, 1.0]);
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
}
