//! ONNX sequence values and zero-copy split views.
//!
//! Sequence elements reuse the session crate's [`Tensor`] representation. A
//! [`SeqTensor`] is an immutable view over an `Arc`-owned device allocation, so
//! constructing, inserting, erasing, indexing, and splitting a sequence only
//! clone handles and metadata. Tensor storage is never deep-copied by those
//! operations.

mod concat;
mod error;
mod split;
mod tensor;
mod value;

pub use concat::concat;
pub(crate) use concat::{ConcatCopyStats, ConcatPlan, stack_new_axis};
pub use error::{SequenceError, SequenceResult};
pub use split::{SplitSpec, split, split_tensor};
pub use tensor::SeqTensor;
pub use value::SequenceValue;

pub(crate) use tensor::clone_shape;
use tensor::{
    addressable, checked_add, checked_mul, checked_product, normalize_axis, overflow,
    validate_view_bounds, zeroed_bytes,
};

#[cfg(test)]
use tensor::validate_tensor_bytes;

#[cfg(test)]
use onnx_runtime_ir::DataType;

#[cfg(test)]
use crate::Tensor;

#[cfg(test)]
mod tests {
    use super::*;

    fn elem(dtype: DataType, shape: &[usize], bytes: &[u8]) -> SeqTensor {
        SeqTensor::from_raw(dtype, shape.to_vec(), bytes).expect("valid test tensor")
    }

    #[test]
    fn value_ops_share_tensor_arcs_without_copying() {
        let original = elem(DataType::Uint8, &[1], &[7]);
        let sequence = SequenceValue::construct(vec![original.clone()]).expect("construct");
        assert!(original.shares_storage_with(&sequence.elements()[0]));

        let inserted = elem(DataType::Uint8, &[1], &[9]);
        let sequence = sequence.insert(inserted.clone(), Some(-1)).expect("insert");
        assert!(inserted.shares_storage_with(&sequence.at(0).expect("at")));
        assert!(original.shares_storage_with(&sequence.at(1).expect("at")));

        let erased = sequence.erase(Some(0)).expect("erase");
        assert!(original.shares_storage_with(&erased.at(-1).expect("negative at")));
    }

    #[test]
    fn moving_tensor_into_sequence_preserves_allocation_pointer() {
        let tensor = Tensor::from_raw(DataType::Uint8, vec![2], &[4, 5]).unwrap();
        let pointer = tensor.device_ptr();
        let element = SeqTensor::new(tensor);
        assert_eq!(element.as_ptr(), pointer);
        let sequence = SequenceValue::construct(vec![element.clone()]).unwrap();
        assert_eq!(sequence.at(0).unwrap().as_ptr(), pointer);
        assert_eq!(element.storage_strong_count(), 2);
    }

    #[test]
    fn split_produces_shared_strided_views_without_copying() {
        let input = elem(DataType::Uint8, &[2, 3], &[0, 1, 2, 3, 4, 5]);
        let sequence = split_tensor(&input, 1, SplitSpec::Sizes(&[1, 2]), true).expect("split");
        assert_eq!(sequence.length(), 2);
        assert!(input.shares_storage_with(&sequence.elements()[0]));
        assert!(input.shares_storage_with(&sequence.elements()[1]));
        assert_eq!(sequence.elements()[0].byte_offset(), 0);
        assert_eq!(sequence.elements()[1].byte_offset(), 1);
        assert_eq!(
            sequence.elements()[0].contiguous_bytes().unwrap(),
            vec![0, 3]
        );
        assert_eq!(
            sequence.elements()[1].contiguous_bytes().unwrap(),
            vec![1, 2, 4, 5]
        );
    }

    #[test]
    fn strided_split_element_byte_borrow_is_fallible_without_panicking() {
        let input = elem(DataType::Uint8, &[2, 3], &[0, 1, 2, 3, 4, 5]);
        let sequence = split_tensor(&input, 1, SplitSpec::Sizes(&[1, 2]), true).expect("split");
        let element = &sequence.elements()[0];
        assert!(matches!(
            element.as_bytes(),
            Err(SequenceError::ByteBorrowUnavailable {
                reason: "the view is strided",
                ..
            })
        ));
        assert_eq!(element.contiguous_bytes().unwrap(), vec![0, 3]);
    }

    #[test]
    fn empty_construct_insert_erase_at_and_length() {
        let empty = SequenceValue::empty(DataType::Uint8);
        assert_eq!(empty.length(), 0);
        let one = empty
            .insert(elem(DataType::Uint8, &[1], &[1]), None)
            .unwrap();
        let two = one
            .insert(elem(DataType::Uint8, &[1], &[2]), Some(-1))
            .unwrap();
        let three = two
            .insert(elem(DataType::Uint8, &[1], &[3]), Some(2))
            .unwrap();
        assert_eq!(three.length(), 3);
        assert_eq!(three.at(0).unwrap().as_bytes().unwrap(), &[2]);
        assert_eq!(three.at(-1).unwrap().as_bytes().unwrap(), &[3]);
        let erased = three.erase(Some(-2)).unwrap();
        assert_eq!(erased.length(), 2);
        assert_eq!(erased.at(0).unwrap().as_bytes().unwrap(), &[2]);
        assert_eq!(erased.at(1).unwrap().as_bytes().unwrap(), &[3]);
    }

    #[test]
    fn empty_sequence_edges_are_clean_errors() {
        let empty = SequenceValue::empty(DataType::Uint8);
        assert!(matches!(empty.erase(None), Err(SequenceError::EmptyErase)));
        assert!(matches!(
            empty.at(0),
            Err(SequenceError::IndexOutOfBounds { len: 0, .. })
        ));
        assert!(matches!(
            concat(&empty, 0, false),
            Err(SequenceError::InvalidSplit { .. })
        ));
    }

    #[test]
    fn homogeneity_violation_is_typed_error() {
        let error = SequenceValue::construct(vec![
            elem(DataType::Uint8, &[1], &[1]),
            elem(DataType::Int64, &[1], &1i64.to_le_bytes()),
        ])
        .unwrap_err();
        assert!(matches!(
            error,
            SequenceError::DtypeMismatch {
                op: "SequenceConstruct",
                index: Some(1),
                ..
            }
        ));
    }

    #[test]
    fn split_concat_roundtrip_existing_axes_and_keepdims() {
        let data: Vec<u8> = (0..12).collect();
        for (shape, axis, keepdims) in [
            (vec![3, 4], 0, true),
            (vec![3, 4], 1, true),
            (vec![3, 4], 0, false),
        ] {
            let input = elem(DataType::Uint8, &shape, &data);
            let sequence = split_tensor(&input, axis, SplitSpec::Each, keepdims).unwrap();
            let concat_axis = if keepdims { axis } else { 0 };
            let rebuilt = concat(&sequence, concat_axis, !keepdims).unwrap();
            assert_eq!(rebuilt.shape, shape);
            assert_eq!(rebuilt.as_bytes().unwrap(), data);
        }
    }

    #[test]
    fn split_concat_roundtrip_explicit_sizes() {
        let data: Vec<u8> = (0..12).collect();
        let input = elem(DataType::Uint8, &[3, 4], &data);
        let sequence = split_tensor(&input, 1, SplitSpec::Sizes(&[1, 3]), false).unwrap();
        let rebuilt = concat(&sequence, 1, false).unwrap();
        assert_eq!(rebuilt.shape, vec![3, 4]);
        assert_eq!(rebuilt.as_bytes().unwrap(), data);
    }

    #[test]
    fn stack_new_axis_variants() {
        let a = elem(DataType::Uint8, &[2], &[1, 2]);
        let b = elem(DataType::Uint8, &[2], &[3, 4]);
        let sequence = SequenceValue::construct(vec![a, b]).unwrap();
        let front = concat(&sequence, 0, true).unwrap();
        assert_eq!(front.shape, vec![2, 2]);
        assert_eq!(front.as_bytes().unwrap(), &[1, 2, 3, 4]);
        let back = concat(&sequence, 1, true).unwrap();
        assert_eq!(back.shape, vec![2, 2]);
        assert_eq!(back.as_bytes().unwrap(), &[1, 3, 2, 4]);
    }

    #[test]
    fn concat_plan_uses_one_final_destination_without_source_materialization() {
        let a = elem(DataType::Uint8, &[2], &[1, 2]);
        let b = elem(DataType::Uint8, &[2], &[3, 4]);
        let sequence = SequenceValue::construct(vec![a, b]).unwrap();
        let plan = ConcatPlan::new(&sequence, 0, false).unwrap();
        let mut destination_allocations = 0;
        let mut destination = {
            destination_allocations += 1;
            vec![0; plan.bytes]
        };
        let stats = plan
            .write(&sequence, |offset, bytes| {
                destination[offset..offset + bytes.len()].copy_from_slice(bytes);
                Ok(())
            })
            .unwrap();
        assert_eq!(destination_allocations, 1);
        assert_eq!(stats.source_materializations, 0);
        assert_eq!(stats.destination_writes, 2);
        assert_eq!(destination, vec![1, 2, 3, 4]);
    }

    #[test]
    fn byte_count_above_isize_max_is_rejected() {
        let error = validate_tensor_bytes(
            "SplitToSequence",
            &[],
            DataType::Uint8,
            &[isize::MAX as usize + 1, 1],
        )
        .unwrap_err();
        assert!(matches!(error, SequenceError::ShapeOverflow { .. }));
    }

    #[test]
    fn sequence_values_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SeqTensor>();
        assert_send_sync::<SequenceValue>();
    }
}
