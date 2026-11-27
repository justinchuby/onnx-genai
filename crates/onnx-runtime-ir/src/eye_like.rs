//! Shared ONNX EyeLike type contract for inference and native EP admission.

use crate::{Attribute, DataType, Node, Shape};

/// Supported tensor types across all EyeLike revisions.
pub const DTYPES: &[DataType] = &[
    DataType::Bool,
    DataType::Uint8,
    DataType::Uint16,
    DataType::Uint32,
    DataType::Uint64,
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::Float16,
    DataType::Float32,
    DataType::Float64,
    DataType::BFloat16,
];

pub struct Attributes {
    pub k: i64,
    pub dtype: Option<DataType>,
}

impl Attributes {
    pub fn parse(node: &Node) -> Result<Self, String> {
        let k = match node.attr("k") {
            None => 0,
            Some(Attribute::Int(k)) => *k,
            Some(_) => return Err("attribute 'k' must be an integer".into()),
        };
        let dtype = match node.attr("dtype") {
            None => None,
            Some(Attribute::Int(raw)) => Some(
                i32::try_from(*raw)
                    .ok()
                    .and_then(DataType::from_onnx)
                    .ok_or_else(|| format!("attribute 'dtype' has invalid ONNX type code {raw}"))?,
            ),
            Some(_) => return Err("attribute 'dtype' must be an integer".into()),
        };
        Ok(Self { k, dtype })
    }

    pub fn resolve_dtype(
        &self,
        opset: u64,
        input: Option<DataType>,
    ) -> Result<Option<DataType>, String> {
        let input = input.filter(|&dtype| dtype != DataType::Undefined);
        let output = self.dtype.or(input);
        for (edge, dtype) in [("input", input), ("output", output)] {
            if let Some(dtype) = dtype {
                if dtype == DataType::BFloat16 && opset < 22 {
                    return Err(format!(
                        "{edge} dtype BFloat16 is not allowed at opset {opset}; use opset 22 or later"
                    ));
                }
                if !DTYPES.contains(&dtype) {
                    return Err(format!(
                        "{edge} dtype {dtype:?} is not allowed at opset {opset}; use a supported numeric or boolean tensor type"
                    ));
                }
            }
        }
        Ok(output)
    }
}

pub fn validate_claim(
    node: &Node,
    opset: u64,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Result<(), String> {
    if !matches!(node.inputs.as_slice(), [Some(_)]) || node.outputs.len() != 1 {
        return Err("expected one required input and one output".into());
    }
    // Admission also receives an empty shape for unknown rank; inference and
    // execution check rank 2 once the shape is known.
    if let Some(shape) = shapes.first()
        && !shape.is_empty()
        && shape.len() != 2
    {
        return Err(format!("input must be rank 2, got rank {}", shape.len()));
    }
    Attributes::parse(node)?.resolve_dtype(opset, input_dtypes.first().copied())?;
    Ok(())
}
