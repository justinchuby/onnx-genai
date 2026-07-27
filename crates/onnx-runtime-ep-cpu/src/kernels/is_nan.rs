//! ONNX `IsNaN`: identify NaN elements in floating tensors.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::{check_arity, write_dense_bytes};
use crate::dtype::{NumericElem, to_dense};

pub struct IsNaNKernel;

pub struct IsNaNFactory;

impl KernelFactory for IsNaNFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(IsNaNKernel))
    }
}

trait NanElem: NumericElem {
    fn is_nan_elem(self) -> bool;
}

macro_rules! impl_nan_elem {
    ($($t:ty),* $(,)?) => {$(
        impl NanElem for $t {
            fn is_nan_elem(self) -> bool { self.is_nan() }
        }
    )*};
}
impl_nan_elem!(f32, f64, half::f16, half::bf16);

impl Kernel for IsNaNKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("IsNaN", inputs, outputs, 1, 1, 1)?;
        if outputs[0].dtype != DataType::Bool {
            return Err(EpError::KernelFailed(
                "IsNaN: output must have Bool dtype".into(),
            ));
        }
        crate::dispatch_float!(inputs[0].dtype, "IsNaN", T => {
            is_nan_typed::<T>(inputs, outputs)
        })
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn is_nan_typed<T: NanElem>(inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
    let input = to_dense::<T>(&inputs[0])?;
    let output = input
        .into_iter()
        .map(|value| u8::from(value.is_nan_elem()))
        .collect::<Vec<_>>();
    write_dense_bytes(&mut outputs[0], &output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::NodeId;

    fn run(input: Owned) -> Vec<bool> {
        let node = Node::new(NodeId(0), "IsNaN", vec![], vec![]);
        let mut output = Owned::zeros(DataType::Bool, &input.shape);
        IsNaNFactory
            .create(&node, &[])
            .unwrap()
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();
        output.to_bool()
    }

    #[test]
    fn detects_nan_in_f32() {
        let input = Owned::f32(
            &[5],
            &[f32::NAN, -1.0, f32::INFINITY, f32::NEG_INFINITY, 0.0],
        );
        assert_eq!(run(input), vec![true, false, false, false, false]);
    }

    #[test]
    fn detects_nan_in_bf16() {
        let input = Owned::bf16(&[4], &[f32::NAN, 1.0, f32::INFINITY, -2.0]);
        assert_eq!(run(input), vec![true, false, false, false]);
    }

    #[test]
    fn detects_nan_in_f16_and_f64() {
        let half = Owned::f16(&[3], &[f32::NAN, 0.0, f32::INFINITY]);
        assert_eq!(run(half), vec![true, false, false]);

        let double = Owned::f64(&[3], &[f64::NAN, 1.0, f64::NEG_INFINITY]);
        assert_eq!(run(double), vec![true, false, false]);
    }
}
