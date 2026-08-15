//! ONNX `IsInf`: identify positive and/or negative infinity in floating tensors.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::{check_arity, write_dense_bytes};
use crate::dtype::{NumericElem, to_dense};

pub struct IsInfKernel {
    detect_negative: bool,
    detect_positive: bool,
}

pub struct IsInfFactory;

impl KernelFactory for IsInfFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        fn flag(node: &Node, name: &str) -> Result<bool> {
            match node.attr(name) {
                None | Some(Attribute::Int(1)) => Ok(true),
                Some(Attribute::Int(0)) => Ok(false),
                Some(Attribute::Int(value)) => Err(EpError::KernelFailed(format!(
                    "IsInf: `{name}` must be 0 or 1, got {value}"
                ))),
                Some(_) => Err(EpError::KernelFailed(format!(
                    "IsInf: `{name}` must be an integer attribute"
                ))),
            }
        }
        Ok(Box::new(IsInfKernel {
            detect_negative: flag(node, "detect_negative")?,
            detect_positive: flag(node, "detect_positive")?,
        }))
    }
}

trait InfElem: NumericElem {
    fn is_positive_infinite(self) -> bool;
    fn is_negative_infinite(self) -> bool;
}

macro_rules! impl_inf_elem {
    ($($t:ty),* $(,)?) => {$(
        impl InfElem for $t {
            fn is_positive_infinite(self) -> bool { self.is_infinite() && self.is_sign_positive() }
            fn is_negative_infinite(self) -> bool { self.is_infinite() && self.is_sign_negative() }
        }
    )*};
}
impl_inf_elem!(f32, f64, half::f16, half::bf16);

impl Kernel for IsInfKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("IsInf", inputs, outputs, 1, 1, 1)?;
        if outputs[0].dtype != DataType::Bool {
            return Err(EpError::KernelFailed(
                "IsInf: output must have Bool dtype".into(),
            ));
        }
        crate::dispatch_float!(inputs[0].dtype, "IsInf", T => {
            is_inf_typed::<T>(self, inputs, outputs)
        })
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn is_inf_typed<T: InfElem>(
    kernel: &IsInfKernel,
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
) -> Result<()> {
    let input = to_dense::<T>(&inputs[0])?;
    let output = input
        .into_iter()
        .map(|value| {
            u8::from(
                (kernel.detect_positive && value.is_positive_infinite())
                    || (kernel.detect_negative && value.is_negative_infinite()),
            )
        })
        .collect::<Vec<_>>();
    write_dense_bytes(&mut outputs[0], &output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::NodeId;

    fn run(attrs: &[(&str, i64)]) -> Vec<bool> {
        let mut node = Node::new(NodeId(0), "IsInf", vec![], vec![]);
        for &(name, value) in attrs {
            node.attributes.insert(name.into(), Attribute::Int(value));
        }
        let input = Owned::f32(
            &[5],
            &[f32::NEG_INFINITY, -1.0, f32::INFINITY, f32::NAN, 0.0],
        );
        let mut output = Owned::zeros(DataType::Bool, &[5]);
        IsInfFactory
            .create(&node, &[])
            .unwrap()
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();
        output.to_bool()
    }

    #[test]
    fn detects_both_infinities_by_default() {
        assert_eq!(run(&[]), vec![true, false, true, false, false]);
    }
    #[test]
    fn can_disable_positive_detection() {
        assert_eq!(
            run(&[("detect_positive", 0)]),
            vec![true, false, false, false, false]
        );
    }
    #[test]
    fn can_disable_negative_detection() {
        assert_eq!(
            run(&[("detect_negative", 0)]),
            vec![false, false, true, false, false]
        );
    }

    #[test]
    fn supports_bf16_inputs() {
        let node = Node::new(NodeId(0), "IsInf", vec![], vec![]);
        let x = Owned::bf16(&[4], &[f32::INFINITY, f32::NEG_INFINITY, 1.0, f32::NAN]);
        let mut out = Owned::zeros(DataType::Bool, &[4]);
        IsInfFactory
            .create(&node, &[])
            .unwrap()
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_bool(), vec![true, true, false, false]);
    }

    #[test]
    fn supports_half_and_double_inputs() {
        let node = Node::new(NodeId(0), "IsInf", vec![], vec![]);
        let half = Owned::f16(&[2], &[f32::INFINITY, f32::NEG_INFINITY]);
        let mut half_out = Owned::zeros(DataType::Bool, &[2]);
        IsInfFactory
            .create(&node, &[])
            .unwrap()
            .execute(&[half.view()], &mut [half_out.view_mut()])
            .unwrap();
        assert_eq!(half_out.to_bool(), vec![true, true]);

        let double = Owned::f64(&[2], &[f64::INFINITY, 1.0]);
        let mut double_out = Owned::zeros(DataType::Bool, &[2]);
        IsInfFactory
            .create(&node, &[])
            .unwrap()
            .execute(&[double.view()], &mut [double_out.view_mut()])
            .unwrap();
        assert_eq!(double_out.to_bool(), vec![true, false]);
    }
}
