//! `Hardmax`: one-hot selection of the first maximum along an axis.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::check_arity;
use crate::dtype::{FloatElem, to_dense_float, unsupported_dtype, write_dense_float};
use crate::strided::numel;

pub struct HardmaxKernel {
    axis: i64,
}

pub struct HardmaxFactory;

impl KernelFactory for HardmaxFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(HardmaxKernel {
            axis: node
                .attr("axis")
                .and_then(|attr| attr.as_int())
                .unwrap_or(-1),
        }))
    }
}

impl Kernel for HardmaxKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Hardmax", inputs, outputs, 1, 1, 1)?;
        match inputs[0].dtype {
            DataType::Float32 => hardmax_typed::<f32>(self.axis, inputs, outputs),
            DataType::Float64 => hardmax_typed::<f64>(self.axis, inputs, outputs),
            other => Err(unsupported_dtype("Hardmax", other)),
        }
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn hardmax_typed<T: FloatElem + PartialOrd>(
    raw_axis: i64,
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
) -> Result<()> {
    if outputs[0].dtype != T::DTYPE {
        return Err(EpError::KernelFailed(format!(
            "Hardmax: output dtype {:?} must match input dtype {:?}",
            outputs[0].dtype,
            T::DTYPE
        )));
    }
    let shape = inputs[0].shape;
    let rank = shape.len();
    if rank == 0 {
        return Err(EpError::KernelFailed(
            "Hardmax: input must have rank >= 1".into(),
        ));
    }
    let axis = if raw_axis < 0 {
        raw_axis + rank as i64
    } else {
        raw_axis
    };
    if axis < 0 || axis as usize >= rank {
        return Err(EpError::KernelFailed(format!(
            "Hardmax: axis {raw_axis} out of range for rank {rank}"
        )));
    }
    let axis = axis as usize;
    let width = shape[axis];
    if width == 0 {
        return Err(EpError::KernelFailed(
            "Hardmax: selected axis must be non-empty".into(),
        ));
    }

    let values = to_dense_float::<T>(&inputs[0])?;
    let outer = numel(&shape[..axis]);
    let inner = numel(&shape[axis + 1..]);
    let mut output = vec![T::from_f32(0.0); values.len()];
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let base = outer_index * width * inner + inner_index;
            let mut best = 0;
            for index in 1..width {
                if values[base + index * inner] > values[base + best * inner] {
                    best = index;
                }
            }
            output[base + best * inner] = T::from_f32(1.0);
        }
    }
    write_dense_float::<T>(&mut outputs[0], &output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn hardmax_default_axis_selects_first_tied_maximum() {
        let input = Owned::f32(&[2, 4], &[1., 5., 5., 2., 7., 7., 6., 7.]);
        let mut output = Owned::zeros_f32(&[2, 4]);
        HardmaxKernel { axis: -1 }
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.to_f32(), vec![0., 1., 0., 0., 1., 0., 0., 0.]);
    }

    #[test]
    fn hardmax_supports_negative_interior_axis() {
        let input = Owned::f32(
            &[2, 3, 2],
            &[1., 4., 3., 2., 3., 5., 6., 1., 6., 2., 1., 3.],
        );
        let mut output = Owned::zeros_f32(&[2, 3, 2]);
        HardmaxKernel { axis: -2 }
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(
            output.to_f32(),
            vec![0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1.]
        );
    }

    #[test]
    fn hardmax_supports_float64() {
        let input = Owned::f64(&[2, 3], &[1., 3., 2., 4., 2., 4.]);
        let mut output = Owned::zeros(DataType::Float64, &[2, 3]);
        HardmaxKernel { axis: 1 }
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.to_f64(), vec![0., 1., 0., 1., 0., 0.]);
    }
}
