use crate::{DataType, Node};

/// Scatter reduction codes shared by the native implementations.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ScatterReduction {
    None = 0,
    Add = 1,
    Mul = 2,
    Max = 3,
    Min = 4,
    Mean = 5,
}

impl ScatterReduction {
    pub fn parse(node: &Node, opset: u64) -> Result<Self, String> {
        let reduction = match node.attr("reduction") {
            None => return Ok(Self::None),
            Some(attr) => attr
                .as_str()
                .ok_or_else(|| format!("{}: 'reduction' must be a string", node.op_type))?,
        };
        let (mode, since) = match reduction {
            "none" => (Self::None, 16),
            "add" => (Self::Add, 16),
            "mul" => (Self::Mul, 16),
            "max" => (Self::Max, 18),
            "min" => (Self::Min, 18),
            "mean" => (Self::Mean, 29),
            value => return Err(format!("{}: unknown reduction {value:?}", node.op_type)),
        };
        if opset < since {
            return Err(format!(
                "{}: reduction {reduction:?} requires opset {since} or later, got {opset}",
                node.op_type
            ));
        }
        Ok(mode)
    }
}

/// GatherND and ScatterND started with only Int64 tuple coordinates.
pub fn validate_nd_index_type(op: &str, opset: u64, dtype: DataType) -> Result<(), String> {
    match dtype {
        DataType::Int64 => Ok(()),
        DataType::Int32 if opset >= 29 => Ok(()),
        DataType::Int32 => Err(format!(
            "{op}: Int32 indices require opset 29 or later, got {opset}; use Int64 indices"
        )),
        _ => Err(format!(
            "{op}: indices must be Int32 or Int64, got {dtype:?}"
        )),
    }
}
