//! Integer bitwise kernels with NumPy-style broadcasting.

use core::ops::{BitAnd, BitOr, BitXor, Not};

use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::add::{broadcast_apply, require_same_dtype};
use super::check_arity;
use crate::dtype::{NumericElem, to_dense, unsupported_dtype, write_dense};
use crate::strided::numel;

pub struct BitwiseNotKernel;
pub struct BitwiseNotFactory;

impl KernelFactory for BitwiseNotFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(BitwiseNotKernel))
    }
}

#[derive(Clone, Copy)]
enum BitwiseOp {
    And,
    Or,
    Xor,
}

impl BitwiseOp {
    fn name(self) -> &'static str {
        match self {
            Self::And => "BitwiseAnd",
            Self::Or => "BitwiseOr",
            Self::Xor => "BitwiseXor",
        }
    }

    fn apply<T: BitAnd<Output = T> + BitOr<Output = T> + BitXor<Output = T>>(
        self,
        lhs: T,
        rhs: T,
    ) -> T {
        match self {
            Self::And => lhs & rhs,
            Self::Or => lhs | rhs,
            Self::Xor => lhs ^ rhs,
        }
    }
}

pub struct BitwiseKernel {
    op: BitwiseOp,
}

macro_rules! bitwise_factory {
    ($factory:ident, $op:ident) => {
        pub struct $factory;

        impl KernelFactory for $factory {
            fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
                Ok(Box::new(BitwiseKernel { op: BitwiseOp::$op }))
            }
        }
    };
}

bitwise_factory!(BitwiseAndFactory, And);
bitwise_factory!(BitwiseOrFactory, Or);
bitwise_factory!(BitwiseXorFactory, Xor);

macro_rules! dispatch_integer {
    ($dtype:expr, $op:expr, $T:ident => $body:expr) => {{
        match $dtype {
            DataType::Int8 => {
                type $T = i8;
                $body
            }
            DataType::Int16 => {
                type $T = i16;
                $body
            }
            DataType::Int32 => {
                type $T = i32;
                $body
            }
            DataType::Int64 => {
                type $T = i64;
                $body
            }
            DataType::Uint8 => {
                type $T = u8;
                $body
            }
            DataType::Uint16 => {
                type $T = u16;
                $body
            }
            DataType::Uint32 => {
                type $T = u32;
                $body
            }
            DataType::Uint64 => {
                type $T = u64;
                $body
            }
            other => Err(unsupported_dtype($op, other)),
        }
    }};
}

impl Kernel for BitwiseKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let name = self.op.name();
        check_arity(name, inputs, outputs, 2, 2, 1)?;
        dispatch_integer!(inputs[0].dtype, name, T => {
            bitwise_typed::<T>(self.op, inputs, outputs)
        })
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn bitwise_typed<
    T: NumericElem + Default + BitAnd<Output = T> + BitOr<Output = T> + BitXor<Output = T>,
>(
    op: BitwiseOp,
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
) -> Result<()> {
    let name = op.name();
    require_same_dtype(name, &inputs[1], T::DTYPE)?;
    if outputs[0].dtype != T::DTYPE {
        return Err(onnx_runtime_ep_api::EpError::KernelFailed(format!(
            "{name}: output dtype {:?} must match input dtype {:?}",
            outputs[0].dtype,
            T::DTYPE
        )));
    }
    let out_shape = outputs[0].shape.to_vec();
    let lhs = to_dense::<T>(&inputs[0])?;
    let rhs = to_dense::<T>(&inputs[1])?;
    let mut values = vec![T::default(); numel(&out_shape)];
    broadcast_apply(&lhs, inputs[0].shape, &out_shape, |i, value| {
        values[i] = value
    })?;
    broadcast_apply(&rhs, inputs[1].shape, &out_shape, |i, value| {
        values[i] = op.apply(values[i], value)
    })?;
    write_dense::<T>(&mut outputs[0], &values)
}

impl Kernel for BitwiseNotKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("BitwiseNot", inputs, outputs, 1, 1, 1)?;
        dispatch_integer!(inputs[0].dtype, "BitwiseNot", T => {
            bitwise_not_typed::<T>(inputs, outputs)
        })
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn bitwise_not_typed<T: NumericElem + Not<Output = T>>(
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
) -> Result<()> {
    if outputs[0].dtype != T::DTYPE {
        return Err(onnx_runtime_ep_api::EpError::KernelFailed(format!(
            "BitwiseNot: output dtype {:?} must match input dtype {:?}",
            outputs[0].dtype,
            T::DTYPE
        )));
    }
    let values: Vec<T> = to_dense::<T>(&inputs[0])?
        .into_iter()
        .map(|value| !value)
        .collect();
    write_dense::<T>(&mut outputs[0], &values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::compute_contiguous_strides;

    const INTEGER_DTYPES: [DataType; 8] = [
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::Uint8,
        DataType::Uint16,
        DataType::Uint32,
        DataType::Uint64,
    ];

    fn bytes(dtype: DataType, shape: &[usize], byte: u8) -> Owned {
        Owned {
            bytes: vec![byte; shape.iter().product::<usize>() * dtype.byte_size()],
            shape: shape.to_vec(),
            strides: compute_contiguous_strides(shape),
            dtype,
        }
    }

    #[test]
    fn binary_ops_support_every_integer_dtype() {
        let cases = [
            (BitwiseOp::And, 0xf0, 0x0f, 0x00),
            (BitwiseOp::Or, 0xf0, 0x0f, 0xff),
            (BitwiseOp::Xor, 0xf0, 0x0f, 0xff),
        ];
        for dtype in INTEGER_DTYPES {
            for (op, lhs_byte, rhs_byte, expected_byte) in cases {
                let lhs = bytes(dtype, &[2, 1], lhs_byte);
                let rhs = bytes(dtype, &[1, 3], rhs_byte);
                let mut out = bytes(dtype, &[2, 3], 0);
                BitwiseKernel { op }
                    .execute(&[lhs.view(), rhs.view()], &mut [out.view_mut()])
                    .unwrap();
                assert_eq!(out.bytes, vec![expected_byte; out.bytes.len()], "{dtype:?}");
            }
        }
    }

    #[test]
    fn bitwise_ops_broadcast_rank_four_and_rank_three() {
        let lhs = bytes(DataType::Uint64, &[2, 1, 1, 3], 0xf0);
        let rhs = bytes(DataType::Uint64, &[1, 4, 3], 0x0f);
        let mut out = bytes(DataType::Uint64, &[2, 1, 4, 3], 0);
        BitwiseKernel { op: BitwiseOp::Xor }
            .execute(&[lhs.view(), rhs.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.bytes, vec![0xff; out.bytes.len()]);
    }

    #[test]
    fn bitwise_not_supports_every_integer_dtype() {
        for dtype in INTEGER_DTYPES {
            let input = bytes(dtype, &[3], 0);
            let mut out = bytes(dtype, &[3], 0);
            BitwiseNotKernel
                .execute(&[input.view()], &mut [out.view_mut()])
                .unwrap();
            assert_eq!(out.bytes, vec![0xff; out.bytes.len()], "{dtype:?}");
        }
    }
}
