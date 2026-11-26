//! Native integer `BitShift` with NumPy-style broadcasting.

use onnx_runtime_ep_api::bitshift::{native_supports_dtype, unsupported_dtype_reason};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node, Shape};

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
    fn unsupported_reason(
        &self,
        _node: &Node,
        _opset: u64,
        _shapes: &[Shape],
        input_dtypes: &[DataType],
    ) -> Option<String> {
        let &dtype = input_dtypes.first()?;
        if !native_supports_dtype(dtype) {
            return Some(unsupported_dtype_reason(dtype));
        }
        if let Some(&rhs) = input_dtypes.get(1)
            && dtype != rhs
        {
            return Some(format!(
                "BitShift: operands must have the same dtype (got {dtype:?} and {rhs:?})"
            ));
        }
        None
    }

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

macro_rules! impl_signed_shiftable {
    ($($t:ty),+ $(,)?) => {
        $(
            impl Shiftable for $t {
                fn shift_left(self, amount: Self) -> Self {
                    u32::try_from(amount).ok()
                        .and_then(|amount| self.checked_shl(amount))
                        .unwrap_or(0)
                }

                fn shift_right(self, amount: Self) -> Self {
                    u32::try_from(amount).ok()
                        .and_then(|amount| self.checked_shr(amount))
                        .unwrap_or(self >> (<$t>::BITS - 1))
                }
            }
        )+
    };
}

impl_signed_shiftable!(i8, i16, i32, i64);

impl Kernel for BitShiftKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("BitShift", inputs, outputs, 2, 2, 1)?;
        macro_rules! dispatch {
            ($t:ty) => {
                bitshift_typed::<$t>(self.direction, inputs, outputs)
            };
        }
        match inputs[0].dtype {
            DataType::Uint8 => dispatch!(u8),
            DataType::Uint16 => dispatch!(u16),
            DataType::Uint32 => dispatch!(u32),
            DataType::Uint64 => dispatch!(u64),
            DataType::Int8 => dispatch!(i8),
            DataType::Int16 => dispatch!(i16),
            DataType::Int32 => dispatch!(i32),
            DataType::Int64 => dispatch!(i64),
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
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{NodeId, compute_contiguous_strides};

    fn from_bits(shape: &[usize], dtype: DataType, values: &[u64]) -> Owned {
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
            let input = from_bits(&[2, 1], dtype, &[3, 8]);
            let shifts = from_bits(&[1, 3], dtype, &[0, 1, 2]);
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

    #[test]
    fn bitshift_claim_accepts_signed_from_introduction_in_both_default_domains() {
        let cpu = crate::CpuExecutionProvider::new();
        let mut node = Node::new(onnx_runtime_ir::NodeId(0), "BitShift", vec![], vec![]);
        for domain in ["", "ai.onnx"] {
            node.domain = domain.into();
            let claim =
                |opset, dtypes: &[DataType]| cpu.supports_op(&node, opset, &[], dtypes, &[]);
            let signed = &[DataType::Int32; 2];
            assert!(!claim(10, signed).is_supported());
            assert!(claim(11, signed).is_supported());
            assert!(claim(28, signed).is_supported());
            assert!(claim(28, &[DataType::Uint32; 2]).is_supported());
            assert!(
                claim(28, &[DataType::Int32, DataType::Uint32])
                    .reason()
                    .unwrap()
                    .contains("same dtype")
            );
        }
    }

    #[test]
    fn signed_bitshift_opset_28_directions_broadcast_and_counts() {
        let cpu = crate::CpuExecutionProvider::new();
        for dtype in [
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
        ] {
            let width = (dtype.byte_size() * 8) as u64;
            let sign_bit = 1u64 << (width - 1);
            let mut counts = vec![0, 1, width - 1, width, width + 1, u64::MAX];
            if dtype == DataType::Int64 {
                // These counts alias 0 and 1 if truncated to a u32 before checking.
                counts.extend([1 << 32, -0xffff_ffffi64 as u64]);
            }
            let shapes = [vec![2, 1], vec![1, counts.len()]];
            let out_shape = [2, counts.len()];
            let x = from_bits(&shapes[0], dtype, &[1, -8i64 as u64]);
            let y = from_bits(&shapes[1], dtype, &counts);
            for (direction, positive, negative, fill) in [
                (
                    "LEFT",
                    [1, 2, sign_bit],
                    [-8i64 as u64, -16i64 as u64, 0],
                    0,
                ),
                (
                    "RIGHT",
                    [1, 0, 0],
                    [-8i64 as u64, -4i64 as u64, u64::MAX],
                    u64::MAX,
                ),
            ] {
                let expected: Vec<_> = positive
                    .into_iter()
                    .chain(std::iter::repeat_n(0, counts.len() - 3))
                    .chain(negative)
                    .chain(std::iter::repeat_n(fill, counts.len() - 3))
                    .collect();
                let mut node = Node::new(NodeId(0), "BitShift", vec![], vec![]);
                node.attributes.insert(
                    "direction".into(),
                    Attribute::String(direction.as_bytes().to_vec()),
                );
                let dtypes = [dtype; 2];
                assert!(
                    cpu.supports_op(&node, 28, &[], &dtypes, &[]).is_supported(),
                    "{dtype:?}"
                );

                let mut out = Owned::zeros(dtype, &out_shape);
                let current = cpu.get_kernel(&node, &shapes, 28).unwrap();
                current
                    .execute(&[x.view(), y.view()], &mut [out.view_mut()])
                    .unwrap();
                assert_eq!(
                    out.bytes,
                    from_bits(&out_shape, dtype, &expected).bytes,
                    "{dtype:?} {direction}"
                );
            }
        }
    }

    #[test]
    fn signed_bitshift_opset_11_executes_arithmetic_right_shift() {
        let cpu = crate::CpuExecutionProvider::new();
        let dtype = DataType::Int32;
        let x = from_bits(&[1], dtype, &[(-8i64) as u64]);
        let y = from_bits(&[4], dtype, &[1, 31, 32, u64::MAX]);
        let mut out = Owned::zeros(dtype, &[4]);
        let mut node = Node::new(NodeId(0), "BitShift", vec![], vec![]);
        node.attributes
            .insert("direction".into(), Attribute::String(b"RIGHT".to_vec()));
        cpu.get_kernel(&node, &[vec![1], vec![4]], 11)
            .unwrap()
            .execute(&[x.view(), y.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(
            values(&out),
            [0xffff_fffc, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff]
        );
    }
}
