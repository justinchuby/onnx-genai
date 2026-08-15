//! Unsigned integer `BitShift` with NumPy-style broadcasting.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::add::{broadcast_apply, require_same_dtype};
use super::check_arity;
use crate::dtype::{NumericElem, to_dense, unsupported_dtype, write_dense};
use crate::strided::numel;

#[derive(Clone, Copy)]
enum Direction {
    Left,
    Right,
}

pub struct BitShiftKernel {
    direction: Direction,
}

pub struct BitShiftFactory;

impl KernelFactory for BitShiftFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let direction = match node.attr("direction").and_then(Attribute::as_str) {
            Some("LEFT") => Direction::Left,
            Some("RIGHT") => Direction::Right,
            Some(_) => {
                return Err(EpError::KernelFailed(
                    "BitShift: direction attribute must be LEFT or RIGHT".into(),
                ));
            }
            None => {
                return Err(EpError::KernelFailed(
                    "BitShift: direction attribute is required".into(),
                ));
            }
        };
        Ok(Box::new(BitShiftKernel { direction }))
    }
}

trait Shiftable: NumericElem {
    fn shift_left(self, amount: Self) -> Self;
    fn shift_right(self, amount: Self) -> Self;
}

macro_rules! impl_shiftable {
    ($($t:ty),+ $(,)?) => {
        $(
            impl Shiftable for $t {
                fn shift_left(self, amount: Self) -> Self {
                    self.checked_shl(amount as u32).unwrap_or(0)
                }

                fn shift_right(self, amount: Self) -> Self {
                    self.checked_shr(amount as u32).unwrap_or(0)
                }
            }
        )+
    };
}

impl_shiftable!(u8, u16, u32, u64);

impl Kernel for BitShiftKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("BitShift", inputs, outputs, 2, 2, 1)?;
        macro_rules! dispatch_unsigned {
            ($t:ty) => {
                bitshift_typed::<$t>(self.direction, inputs, outputs)
            };
        }
        match inputs[0].dtype {
            DataType::Uint8 => dispatch_unsigned!(u8),
            DataType::Uint16 => dispatch_unsigned!(u16),
            DataType::Uint32 => dispatch_unsigned!(u32),
            DataType::Uint64 => dispatch_unsigned!(u64),
            other => Err(unsupported_dtype("BitShift", other)),
        }
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn bitshift_typed<T: Shiftable + Default>(
    direction: Direction,
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
) -> Result<()> {
    require_same_dtype("BitShift", &inputs[1], T::DTYPE)?;
    if outputs[0].dtype != T::DTYPE {
        return Err(EpError::KernelFailed(
            "BitShift: output dtype must match input dtype".into(),
        ));
    }

    let out_shape = outputs[0].shape.to_vec();
    let lhs = to_dense::<T>(&inputs[0])?;
    let rhs = to_dense::<T>(&inputs[1])?;
    let mut values = vec![T::default(); numel(&out_shape)];
    broadcast_apply(&lhs, inputs[0].shape, &out_shape, |i, value| {
        values[i] = value;
    })?;
    broadcast_apply(&rhs, inputs[1].shape, &out_shape, |i, value| {
        values[i] = match direction {
            Direction::Left => values[i].shift_left(value),
            Direction::Right => values[i].shift_right(value),
        };
    })?;
    write_dense::<T>(&mut outputs[0], &values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::{NodeId, compute_contiguous_strides};

    fn unsigned(shape: &[usize], dtype: DataType, values: &[u64]) -> Owned {
        let mut bytes = Vec::with_capacity(values.len() * dtype.byte_size());
        for &value in values {
            bytes.extend_from_slice(&value.to_le_bytes()[..dtype.byte_size()]);
        }
        Owned {
            bytes,
            shape: shape.to_vec(),
            strides: compute_contiguous_strides(shape),
            dtype,
        }
    }

    fn values(owned: &Owned) -> Vec<u64> {
        owned
            .bytes
            .chunks_exact(owned.dtype.byte_size())
            .map(|bytes| {
                let mut padded = [0; 8];
                padded[..bytes.len()].copy_from_slice(bytes);
                u64::from_le_bytes(padded)
            })
            .collect()
    }

    #[test]
    fn shifts_with_broadcasting_for_all_unsigned_widths() {
        for dtype in [
            DataType::Uint8,
            DataType::Uint16,
            DataType::Uint32,
            DataType::Uint64,
        ] {
            let input = unsigned(&[2, 1], dtype, &[3, 8]);
            let shifts = unsigned(&[1, 3], dtype, &[0, 1, 2]);
            let mut left = Owned::zeros(dtype, &[2, 3]);
            BitShiftKernel {
                direction: Direction::Left,
            }
            .execute(&[input.view(), shifts.view()], &mut [left.view_mut()])
            .unwrap();
            assert_eq!(values(&left), vec![3, 6, 12, 8, 16, 32], "{dtype:?}");

            let mut right = Owned::zeros(dtype, &[2, 3]);
            BitShiftKernel {
                direction: Direction::Right,
            }
            .execute(&[left.view(), shifts.view()], &mut [right.view_mut()])
            .unwrap();
            assert_eq!(values(&right), vec![3, 3, 3, 8, 8, 8], "{dtype:?}");
        }
    }

    #[test]
    fn shifts_larger_than_the_element_width_yield_zero() {
        let input = Owned::u8(&[1], &[0x80]);
        let shifts = Owned::u8(&[1], &[8]);
        let mut out = Owned::zeros(DataType::Uint8, &[1]);
        BitShiftKernel {
            direction: Direction::Right,
        }
        .execute(&[input.view(), shifts.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.bytes, vec![0]);
    }

    #[test]
    fn missing_direction_attribute_errors() {
        let node = Node::new(NodeId(0), "BitShift", vec![], vec![]);
        assert!(BitShiftFactory.create(&node, &[]).is_err());
    }
}
