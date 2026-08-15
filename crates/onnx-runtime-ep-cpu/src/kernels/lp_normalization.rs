//! ONNX `LpNormalization` along one axis.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::add::require_same_dtype;
use super::check_arity;
use crate::dispatch_float;
use crate::dtype::{ComputeDomain, NumericElem, to_dense, write_dense};

trait LpAcc: ComputeDomain + PartialOrd {
    const TINY: Self;

    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
}

impl LpAcc for f32 {
    const TINY: Self = f32::MIN_POSITIVE;

    fn abs(self) -> Self {
        self.abs()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }
}

impl LpAcc for f64 {
    const TINY: Self = f64::MIN_POSITIVE;

    fn abs(self) -> Self {
        self.abs()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }
}

pub struct LpNormalizationFactory;

impl KernelFactory for LpNormalizationFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let p = node.attr("p").and_then(|a| a.as_int()).unwrap_or(2);
        if p != 1 && p != 2 {
            return Err(EpError::KernelFailed(format!(
                "LpNormalization: p must be 1 or 2, got {p}"
            )));
        }
        Ok(Box::new(LpNormalizationKernel {
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1),
            p: p as u8,
        }))
    }
}

pub struct LpNormalizationKernel {
    axis: i64,
    p: u8,
}

impl Kernel for LpNormalizationKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("LpNormalization", inputs, outputs, 1, 1, 1)?;
        dispatch_float!(inputs[0].dtype, "LpNormalization", T =>
            lp_normalization_typed::<T>(inputs, outputs, self.axis, self.p)
        )
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn lp_normalization_typed<T: NumericElem>(
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
    raw_axis: i64,
    p: u8,
) -> Result<()>
where
    T::Acc: LpAcc,
{
    let input = &inputs[0];
    let output = &mut outputs[0];
    require_same_dtype("LpNormalization", input, T::DTYPE)?;
    if output.dtype != T::DTYPE {
        return Err(EpError::KernelFailed(format!(
            "LpNormalization: output dtype {:?} must match input dtype {:?}",
            output.dtype,
            T::DTYPE
        )));
    }
    if input.shape != output.shape {
        return Err(EpError::KernelFailed(format!(
            "LpNormalization: output shape {:?} must match input shape {:?}",
            output.shape, input.shape
        )));
    }
    let rank = input.shape.len();
    let axis = if raw_axis < 0 {
        raw_axis + rank as i64
    } else {
        raw_axis
    };
    if axis < 0 || axis as usize >= rank {
        return Err(EpError::KernelFailed(format!(
            "LpNormalization: axis {raw_axis} out of range for rank {rank}"
        )));
    }
    let axis = axis as usize;
    let outer = input.shape[..axis].iter().product::<usize>();
    let axis_len = input.shape[axis];
    let inner = input.shape[axis + 1..].iter().product::<usize>();
    let x = to_dense::<T>(input)?;
    let mut y = vec![T::Acc::default(); x.len()];

    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let mut norm = T::Acc::default();
            for axis_index in 0..axis_len {
                let index = (outer_index * axis_len + axis_index) * inner + inner_index;
                let value = x[index].to_acc().abs();
                norm = norm.c_add(if p == 1 { value } else { value.c_mul(value) });
            }
            if p == 2 {
                norm = norm.sqrt();
            }
            if norm < T::Acc::TINY {
                norm = T::Acc::TINY;
            }
            for axis_index in 0..axis_len {
                let index = (outer_index * axis_len + axis_index) * inner + inner_index;
                y[index] = x[index].to_acc().c_div(norm);
            }
        }
    }

    let y = y.into_iter().map(T::from_acc).collect::<Vec<_>>();
    write_dense::<T>(output, &y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::NodeId;

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "index {index}: got {got}, want {want}"
            );
        }
    }

    fn run(shape: &[usize], values: &[f32], axis: i64, p: u8) -> Vec<f32> {
        let x = Owned::f32(shape, values);
        let mut y = Owned::zeros_f32(shape);
        LpNormalizationKernel { axis, p }
            .execute(&[x.view()], &mut [y.view_mut()])
            .unwrap();
        y.to_f32()
    }

    #[test]
    fn lp_normalization_bf16_matches_widened_f32_reference() {
        let values = [1.0f32, -2.0, 3.0, 4.0, -5.0, 6.0];
        let reference = run(&[2, 3], &values, 1, 2);
        let x = Owned::bf16(&[2, 3], &values);
        let mut y = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[2, 3]);
        LpNormalizationKernel { axis: 1, p: 2 }
            .execute(&[x.view()], &mut [y.view_mut()])
            .unwrap();
        for (&r, &g) in reference.iter().zip(y.to_bf16_as_f32().iter()) {
            assert!(
                (r - g).abs() <= 0.03 * r.abs().max(1.0),
                "lp_normalization bf16 {g} vs f32 {r}"
            );
        }
    }

    #[test]
    fn l1_normalization_axes_zero_one_and_last() {
        let values = [1.0, -2.0, 3.0, 4.0, -5.0, 6.0];
        assert_close(
            &run(&[2, 3], &values, 0, 1),
            &[0.2, -2.0 / 7.0, 1.0 / 3.0, 0.8, -5.0 / 7.0, 2.0 / 3.0],
        );
        let row_norm = 6.0;
        let row2_norm = 15.0;
        let expected = [
            1.0 / row_norm,
            -2.0 / row_norm,
            3.0 / row_norm,
            4.0 / row2_norm,
            -5.0 / row2_norm,
            6.0 / row2_norm,
        ];
        assert_close(&run(&[2, 3], &values, 1, 1), &expected);
        assert_close(&run(&[2, 3], &values, -1, 1), &expected);
    }

    #[test]
    fn l2_normalization_axes_zero_one_and_default_last() {
        let values = [3.0, 4.0, 0.0, 0.0, 0.0, 12.0];
        assert_close(
            &run(&[2, 3], &values, 0, 2),
            &[1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
        let expected = [0.6, 0.8, 0.0, 0.0, 0.0, 1.0];
        assert_close(&run(&[2, 3], &values, 1, 2), &expected);
        let x = Owned::f32(&[2, 3], &values);
        let mut y = Owned::zeros_f32(&[2, 3]);
        let node = Node::new(NodeId(0), "LpNormalization", vec![], vec![]);
        LpNormalizationFactory
            .create(&node, &[])
            .unwrap()
            .execute(&[x.view()], &mut [y.view_mut()])
            .unwrap();
        assert_close(&y.to_f32(), &expected);
    }
}
