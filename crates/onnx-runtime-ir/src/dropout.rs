//! Shared operand constraints for the ONNX Dropout inference kernels.

use crate::{DataType, Node, Shape};

pub fn validate_data_type(opset: u64, dtype: DataType) -> Result<(), String> {
    let supported = dtype == DataType::BFloat16 || ratio_type(opset, dtype);
    if supported {
        Ok(())
    } else {
        Err(format!(
            "data dtype {dtype:?} is not valid for Dropout at opset {opset}"
        ))
    }
}

fn ratio_type(opset: u64, dtype: DataType) -> bool {
    use DataType::*;
    matches!(dtype, Float16 | Float32 | Float64)
        || opset >= 22
            && matches!(
                dtype,
                BFloat16 | Float8E4M3FN | Float8E4M3FNUZ | Float8E5M2 | Float8E5M2FNUZ
            )
}

pub fn validate_ratio(opset: u64, dtype: DataType, rank: usize) -> Result<(), String> {
    if ratio_type(opset, dtype) && rank == 0 {
        Ok(())
    } else {
        Err(format!(
            "ratio must be a scalar of a supported floating-point dtype at opset {opset}, got rank {rank} and {dtype:?}"
        ))
    }
}

pub fn validate_training_mode(dtype: DataType, rank: usize) -> Result<(), String> {
    if dtype == DataType::Bool && rank == 0 {
        Ok(())
    } else {
        Err(format!(
            "training_mode must be a Bool scalar, got rank {rank} and {dtype:?}"
        ))
    }
}

pub fn validate_claim(
    node: &Node,
    opset: u64,
    shapes: &[Shape],
    dtypes: &[DataType],
) -> Result<(), String> {
    if !(1..=3).contains(&node.inputs.len())
        || node.inputs[0].is_none()
        || !(1..=2).contains(&node.outputs.len())
        || node.outputs[0].is_none()
    {
        return Err(
            "expected data, optional ratio and training_mode, and output with optional mask".into(),
        );
    }
    if node.inputs.len() != dtypes.len()
        || (!shapes.is_empty() && node.inputs.len() != shapes.len())
    {
        return Err("shape and dtype metadata must cover every input slot".into());
    }
    validate_data_type(opset, dtypes[0])?;
    // Load-time capability queries can omit shape metadata; execution checks
    // the actual scalar ranks in that case.
    if node.inputs.get(1).is_some_and(Option::is_some) {
        validate_ratio(opset, dtypes[1], shapes.get(1).map_or(0, Vec::len))?;
    }
    if node.inputs.get(2).is_some_and(Option::is_some) {
        validate_training_mode(dtypes[2], shapes.get(2).map_or(0, Vec::len))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NodeId, ValueId, static_shape};

    #[test]
    fn claim_tracks_opset22_operand_boundaries_and_optional_slots() {
        use DataType::*;
        let node = Node::new_with_output_slots(
            NodeId(0),
            "Dropout",
            vec![Some(ValueId(0)), Some(ValueId(1)), None],
            vec![Some(ValueId(2)), None],
        );
        let shapes = [static_shape([4]), vec![], vec![]];
        for extended in [
            BFloat16,
            Float8E4M3FN,
            Float8E4M3FNUZ,
            Float8E5M2,
            Float8E5M2FNUZ,
        ] {
            assert!(validate_claim(&node, 22, &shapes, &[Float32, extended, Undefined]).is_ok());
            assert!(
                validate_claim(&node, 21, &shapes, &[Float32, extended, Undefined])
                    .unwrap_err()
                    .contains("ratio")
            );
        }
        assert!(validate_claim(&node, 22, &[], &[Float8E5M2, Float32, Undefined]).is_ok());
        for invalid in [Int32, Float8E5M2] {
            assert!(
                validate_claim(&node, 21, &shapes, &[invalid, Float32, Undefined])
                    .unwrap_err()
                    .contains("data dtype")
            );
        }
        assert!(
            validate_claim(
                &node,
                22,
                &[static_shape([4]), static_shape([1]), vec![]],
                &[Float32, Float32, Undefined],
            )
            .unwrap_err()
            .contains("ratio must be a scalar")
        );
    }
}
