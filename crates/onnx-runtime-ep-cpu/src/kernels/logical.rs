//! Boolean and comparison kernels (`docs/ORT2.md` §4.4).
//!
//! ONNX `Bool` tensors store one byte per element (`0` = false, non-zero =
//! true). `Not` flips each element, emitting canonical `1`/`0` bytes. It is a
//! straight per-element map over the raw bytes.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::add::{broadcast_apply, require_same_dtype};
use super::{check_arity, to_dense_bytes, write_dense_bytes};
use crate::dispatch_arith;
use crate::dtype::{NumericElem, to_dense};
use crate::strided::numel;

/// Stateless `Not` kernel (boolean element negation).
pub struct NotKernel;

/// Factory for [`NotKernel`] (no attributes).
pub struct NotFactory;

impl KernelFactory for NotFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(NotKernel))
    }
}

/// Stateless elementwise equality comparison with NumPy broadcasting.
pub struct EqualKernel;

/// Factory for [`EqualKernel`] (no attributes).
pub struct EqualFactory;

impl KernelFactory for EqualFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(EqualKernel))
    }
}

#[derive(Clone, Copy)]
enum Comparison {
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
}

impl Comparison {
    fn name(self) -> &'static str {
        match self {
            Self::Greater => "Greater",
            Self::GreaterOrEqual => "GreaterOrEqual",
            Self::Less => "Less",
            Self::LessOrEqual => "LessOrEqual",
        }
    }

    fn apply<T: PartialOrd>(self, lhs: T, rhs: T) -> bool {
        match self {
            Self::Greater => lhs > rhs,
            Self::GreaterOrEqual => lhs >= rhs,
            Self::Less => lhs < rhs,
            Self::LessOrEqual => lhs <= rhs,
        }
    }
}

/// Elementwise ordered comparison with NumPy broadcasting.
pub struct ComparisonKernel {
    comparison: Comparison,
}

pub struct GreaterFactory;
pub struct GreaterOrEqualFactory;
pub struct LessFactory;
pub struct LessOrEqualFactory;

fn comparison_factory(comparison: Comparison) -> Box<dyn Kernel> {
    Box::new(ComparisonKernel { comparison })
}

macro_rules! comparison_factory {
    ($factory:ident, $comparison:ident) => {
        impl KernelFactory for $factory {
            fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
                Ok(comparison_factory(Comparison::$comparison))
            }
        }
    };
}

comparison_factory!(GreaterFactory, Greater);
comparison_factory!(GreaterOrEqualFactory, GreaterOrEqual);
comparison_factory!(LessFactory, Less);
comparison_factory!(LessOrEqualFactory, LessOrEqual);

impl Kernel for EqualKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Equal", inputs, outputs, 2, 2, 1)?;
        if outputs[0].dtype != DataType::Bool {
            return Err(EpError::KernelFailed(format!(
                "Equal: requires a Bool output, got {:?}. WHY: ONNX Equal produces boolean \
                 truth values. HOW: declare the output tensor as Bool.",
                outputs[0].dtype
            )));
        }
        if inputs[0].dtype == DataType::Bool {
            equal_bool(inputs, outputs)
        } else {
            dispatch_arith!(inputs[0].dtype, "Equal", T => equal_typed::<T>(inputs, outputs))
        }
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

impl Kernel for ComparisonKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let name = self.comparison.name();
        check_arity(name, inputs, outputs, 2, 2, 1)?;
        if outputs[0].dtype != DataType::Bool {
            return Err(EpError::KernelFailed(format!(
                "{name}: requires a Bool output, got {:?}. WHY: ONNX comparisons produce \
                 boolean truth values. HOW: declare the output tensor as Bool.",
                outputs[0].dtype
            )));
        }
        dispatch_arith!(inputs[0].dtype, name, T => {
            compare_typed::<T>(self.comparison, inputs, outputs)
        })
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn equal_bool(inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
    require_same_dtype("Equal", &inputs[1], DataType::Bool)?;
    let out_shape = outputs[0].shape.to_vec();
    let mut out = vec![0u8; numel(&out_shape)];
    let a = to_dense_bytes(&inputs[0])?;
    let b = to_dense_bytes(&inputs[1])?;
    broadcast_apply(&a, inputs[0].shape, &out_shape, |i, v| {
        out[i] = u8::from(v != 0)
    })?;
    broadcast_apply(&b, inputs[1].shape, &out_shape, |i, v| {
        out[i] = u8::from((out[i] != 0) == (v != 0))
    })?;
    write_dense_bytes(&mut outputs[0], &out)
}

fn equal_typed<T: NumericElem + PartialEq + Default>(
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
) -> Result<()> {
    require_same_dtype("Equal", &inputs[1], T::DTYPE)?;
    let out_shape = outputs[0].shape.to_vec();
    let mut out = vec![false; numel(&out_shape)];
    let a = to_dense::<T>(&inputs[0])?;
    let b = to_dense::<T>(&inputs[1])?;
    let mut lhs = vec![T::default(); numel(&out_shape)];
    broadcast_apply(&a, inputs[0].shape, &out_shape, |i, v| lhs[i] = v)?;
    broadcast_apply(&b, inputs[1].shape, &out_shape, |i, v| out[i] = lhs[i] == v)?;
    let bytes: Vec<u8> = out.into_iter().map(u8::from).collect();
    write_dense_bytes(&mut outputs[0], &bytes)
}

fn compare_typed<T: NumericElem + PartialOrd + Default>(
    comparison: Comparison,
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
) -> Result<()> {
    let name = comparison.name();
    require_same_dtype(name, &inputs[1], T::DTYPE)?;
    let out_shape = outputs[0].shape.to_vec();
    let a = to_dense::<T>(&inputs[0])?;
    let b = to_dense::<T>(&inputs[1])?;
    let mut lhs = vec![T::default(); numel(&out_shape)];
    let mut out = vec![0u8; numel(&out_shape)];
    broadcast_apply(&a, inputs[0].shape, &out_shape, |i, v| lhs[i] = v)?;
    broadcast_apply(&b, inputs[1].shape, &out_shape, |i, v| {
        out[i] = u8::from(comparison.apply(lhs[i], v))
    })?;
    write_dense_bytes(&mut outputs[0], &out)
}

impl Kernel for NotKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Not", inputs, outputs, 1, 1, 1)?;
        if inputs[0].dtype != DataType::Bool || outputs[0].dtype != DataType::Bool {
            return Err(EpError::KernelFailed(format!(
                "Not: requires Bool input and output, got input {:?} / output {:?}. WHY: `Not` \
                 is a logical op defined only on booleans. HOW: feed a Bool tensor.",
                inputs[0].dtype, outputs[0].dtype
            )));
        }
        let bytes = to_dense_bytes(&inputs[0])?;
        let out: Vec<u8> = bytes.iter().map(|&b| u8::from(b == 0)).collect();
        write_dense_bytes(&mut outputs[0], &out)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn not_flips_bools() {
        let a = Owned::bool_(&[4], &[true, false, true, false]);
        let mut out = Owned::zeros(DataType::Bool, &[4]);
        NotKernel
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_bool(), vec![false, true, false, true]);
    }

    #[test]
    fn not_rejects_non_bool() {
        let a = Owned::f32(&[2], &[1., 0.]);
        let mut out = Owned::zeros_f32(&[2]);
        let err = NotKernel.execute(&[a.view()], &mut [out.view_mut()]);
        assert!(err.is_err());
    }

    #[test]
    fn equal_int64_broadcasts() {
        let a = Owned::i64(&[2, 1], &[1, 2]);
        let b = Owned::i64(&[1, 3], &[1, 0, 2]);
        let mut out = Owned::zeros(DataType::Bool, &[2, 3]);
        EqualKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_bool(), vec![true, false, false, false, false, true]);
    }

    #[test]
    fn equal_float_uses_numeric_equality() {
        let a = Owned::f32(&[3], &[0., -0., f32::NAN]);
        let b = Owned::f32(&[3], &[-0., 0., f32::NAN]);
        let mut out = Owned::zeros(DataType::Bool, &[3]);
        EqualKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_bool(), vec![true, true, false]);
    }

    #[test]
    fn equal_bool_compares_truth_values() {
        let a = Owned::bool_(&[3], &[true, false, true]);
        let b = Owned::bool_(&[3], &[true, true, false]);
        let mut out = Owned::zeros(DataType::Bool, &[3]);
        EqualKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_bool(), vec![true, false, false]);
    }

    #[test]
    fn ordered_comparisons_broadcast_and_obey_equality_boundaries() {
        let a = Owned::i64(&[2, 1], &[1, 2]);
        let b = Owned::i64(&[1, 3], &[1, 2, 3]);
        let cases = [
            (
                Comparison::Greater,
                vec![false, false, false, true, false, false],
            ),
            (
                Comparison::GreaterOrEqual,
                vec![true, false, false, true, true, false],
            ),
            (
                Comparison::Less,
                vec![false, true, true, false, false, true],
            ),
            (
                Comparison::LessOrEqual,
                vec![true, true, true, false, true, true],
            ),
        ];
        for (comparison, expected) in cases {
            let mut out = Owned::zeros(DataType::Bool, &[2, 3]);
            ComparisonKernel { comparison }
                .execute(&[a.view(), b.view()], &mut [out.view_mut()])
                .unwrap();
            assert_eq!(out.to_bool(), expected, "{}", comparison.name());
        }
    }

    #[test]
    fn ordered_comparisons_follow_onnx_nan_semantics() {
        let a = Owned::f32(&[2], &[f32::NAN, 2.]);
        let b = Owned::f32(&[2], &[1., 2.]);
        let mut out = Owned::zeros(DataType::Bool, &[2]);
        ComparisonKernel {
            comparison: Comparison::LessOrEqual,
        }
        .execute(&[a.view(), b.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_bool(), vec![false, true]);
    }
}
