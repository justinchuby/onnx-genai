//! Claim-time contracts for CUDA's constrained standard operators.
//!
//! These kernels intentionally cover only a subset of their ONNX dtype
//! matrices. Keep their placement checks in sync with those runtime limits so a
//! node is never claimed only to fail while constructing or executing a kernel.

use onnx_runtime_ir::{Attribute, DataType, Node};

pub(crate) fn unsupported_reason(node: &Node, input_dtypes: &[DataType]) -> Option<String> {
    let result = match node.op_type.as_str() {
        "RMSNormalization" => rms_normalization(node, input_dtypes),
        "RotaryEmbedding" => rotary_embedding(node, input_dtypes),
        "TopK" => topk(node, input_dtypes),
        "CumSum" => cumsum(node, input_dtypes),
        "Gather" => gather(node, input_dtypes),
        "GatherElements" => gather_elements(node, input_dtypes),
        "ScatterElements" => scatter_elements(node, input_dtypes),
        "Where" => where_op(node, input_dtypes),
        "Expand" => expand(node, input_dtypes),
        "ConstantOfShape" => constant_of_shape(node, input_dtypes),
        "Gelu" => gelu(node, input_dtypes),
        "OneHot" => one_hot(node, input_dtypes),
        _ => return None,
    };
    result
        .err()
        .map(|reason| format!("{}: {reason}", node.op_type))
}

fn required_arity(
    node: &Node,
    input_dtypes: &[DataType],
    inputs: usize,
    min_outputs: usize,
    max_outputs: usize,
) -> Result<(), String> {
    if node.inputs.len() != inputs
        || !(min_outputs..=max_outputs).contains(&node.outputs.len())
        || node.inputs.iter().any(Option::is_none)
    {
        return Err(format!(
            "requires {inputs} present inputs and {min_outputs}..={max_outputs} outputs, got {} inputs and {} outputs",
            node.inputs.len(),
            node.outputs.len()
        ));
    }
    metadata_arity(node, input_dtypes)
}

fn metadata_arity(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    if input_dtypes.len() != node.inputs.len() {
        return Err(format!(
            "claim dtype metadata covers {} inputs, expected {}",
            input_dtypes.len(),
            node.inputs.len()
        ));
    }
    Ok(())
}

fn require_dtype(
    input_dtypes: &[DataType],
    index: usize,
    expected: DataType,
    name: &str,
) -> Result<(), String> {
    let got = input_dtypes[index];
    if got != expected {
        return Err(format!(
            "input {index} ('{name}') dtype {got:?} unsupported; expected {expected:?}"
        ));
    }
    Ok(())
}

fn require_one_of(
    input_dtypes: &[DataType],
    index: usize,
    expected: &[DataType],
    name: &str,
) -> Result<(), String> {
    let got = input_dtypes[index];
    if !expected.contains(&got) {
        return Err(format!(
            "input {index} ('{name}') dtype {got:?} unsupported; expected one of {expected:?}"
        ));
    }
    Ok(())
}

fn require_fixed_width(input_dtypes: &[DataType], index: usize, name: &str) -> Result<(), String> {
    let got = input_dtypes[index];
    if got.byte_size() == 0 {
        return Err(format!(
            "input {index} ('{name}') dtype {got:?} is packed or variable-width"
        ));
    }
    Ok(())
}

fn bool_attribute(node: &Node, name: &str) -> Result<(), String> {
    let Some(attribute) = node.attr(name) else {
        return Ok(());
    };
    match attribute.as_int() {
        Some(0 | 1) => Ok(()),
        _ => Err(format!("attribute '{name}' must be 0 or 1")),
    }
}

fn rms_normalization(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 2, 1, 2)?;
    require_one_of(
        input_dtypes,
        0,
        &[DataType::Float16, DataType::BFloat16, DataType::Float32],
        "X",
    )?;
    require_one_of(
        input_dtypes,
        1,
        &[DataType::Float16, DataType::BFloat16, DataType::Float32],
        "scale",
    )?;
    if input_dtypes[0] == DataType::Float32 && input_dtypes[1] != DataType::Float32 {
        return Err("f32 X requires f32 scale".into());
    }
    if input_dtypes[0] != DataType::Float32
        && input_dtypes[1] != DataType::Float32
        && input_dtypes[1] != input_dtypes[0]
    {
        return Err("f16/bf16 X requires matching storage dtype or f32 scale".into());
    }
    match node.attr("stash_type") {
        None => Ok(()),
        Some(attribute) if attribute.as_int() == Some(1) => Ok(()),
        Some(_) => Err("attribute 'stash_type' must be 1 (float)".into()),
    }
}

fn rotary_embedding(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    if !(3..=4).contains(&node.inputs.len())
        || node.outputs.len() != 1
        || node.inputs[..3].iter().any(Option::is_none)
    {
        return Err(format!(
            "requires 3-4 inputs with X/cos_cache/sin_cache present and 1 output, got {} inputs and {} outputs",
            node.inputs.len(),
            node.outputs.len()
        ));
    }
    metadata_arity(node, input_dtypes)?;
    require_one_of(
        input_dtypes,
        0,
        &[DataType::Float16, DataType::BFloat16, DataType::Float32],
        "X",
    )?;
    for (index, name) in [(1, "cos_cache"), (2, "sin_cache")] {
        if input_dtypes[index] != input_dtypes[0] {
            return Err(format!(
                "input {index} ('{name}') dtype {:?} must match X dtype {:?}",
                input_dtypes[index], input_dtypes[0]
            ));
        }
    }
    if node.inputs.get(3).is_some_and(Option::is_some) {
        require_dtype(input_dtypes, 3, DataType::Int64, "position_ids")?;
    }
    bool_attribute(node, "interleaved")?;
    for name in ["num_heads", "rotary_embedding_dim"] {
        if node
            .attr(name)
            .is_some_and(|attribute| !matches!(attribute.as_int(), Some(value) if value >= 0))
        {
            return Err(format!("attribute '{name}' must be a non-negative integer"));
        }
    }
    Ok(())
}

fn topk(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 2, 2, 2)?;
    require_dtype(input_dtypes, 0, DataType::Float32, "X")?;
    require_dtype(input_dtypes, 1, DataType::Int64, "K")?;
    bool_attribute(node, "largest")?;
    bool_attribute(node, "sorted")
}

fn cumsum(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 2, 1, 1)?;
    require_one_of(input_dtypes, 0, &[DataType::Float32, DataType::Int64], "X")?;
    require_dtype(input_dtypes, 1, DataType::Int64, "axis")?;
    bool_attribute(node, "exclusive")?;
    bool_attribute(node, "reverse")
}

fn gather(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 2, 1, 1)?;
    require_fixed_width(input_dtypes, 0, "data")?;
    require_one_of(
        input_dtypes,
        1,
        &[DataType::Int32, DataType::Int64],
        "indices",
    )
}

fn gather_elements(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 2, 1, 1)?;
    require_fixed_width(input_dtypes, 0, "data")?;
    require_dtype(input_dtypes, 1, DataType::Int64, "indices")
}

fn scatter_elements(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 3, 1, 1)?;
    require_one_of(
        input_dtypes,
        0,
        &[
            DataType::Float16,
            DataType::Float32,
            DataType::BFloat16,
            DataType::Int64,
        ],
        "data",
    )?;
    require_one_of(
        input_dtypes,
        1,
        &[DataType::Int32, DataType::Int64],
        "indices",
    )?;
    require_dtype(input_dtypes, 2, input_dtypes[0], "updates")?;
    match node.attr("reduction") {
        None => Ok(()),
        Some(attribute)
            if matches!(
                attribute.as_str(),
                Some("none" | "add" | "mul" | "max" | "min")
            ) =>
        {
            Ok(())
        }
        Some(_) => {
            Err("attribute 'reduction' must be one of 'none', 'add', 'mul', 'max', or 'min'".into())
        }
    }
}

fn where_op(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 3, 1, 1)?;
    require_dtype(input_dtypes, 0, DataType::Bool, "condition")?;
    require_fixed_width(input_dtypes, 1, "X")?;
    require_dtype(input_dtypes, 2, input_dtypes[1], "Y")
}

fn expand(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 2, 1, 1)?;
    require_fixed_width(input_dtypes, 0, "input")?;
    require_dtype(input_dtypes, 1, DataType::Int64, "shape")
}

fn constant_of_shape(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 1, 1, 1)?;
    require_dtype(input_dtypes, 0, DataType::Int64, "input")?;
    let Some(attribute) = node.attr("value") else {
        return Ok(());
    };
    let Attribute::Tensor(tensor) = attribute else {
        return Err("attribute 'value' must be a tensor".into());
    };
    if tensor.numel() != 1 {
        return Err("attribute 'value' must contain exactly one element".into());
    }
    if tensor.dtype.is_float() || tensor.dtype.is_int() || tensor.dtype == DataType::Bool {
        Ok(())
    } else {
        Err(format!(
            "attribute 'value' dtype {:?} unsupported; expected numeric or bool",
            tensor.dtype
        ))
    }
}

fn gelu(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 1, 1, 1)?;
    require_one_of(
        input_dtypes,
        0,
        &[DataType::Float16, DataType::Float32, DataType::BFloat16],
        "X",
    )?;
    match node.attr("approximate") {
        None => Ok(()),
        Some(attribute) if matches!(attribute.as_str(), Some("none" | "tanh")) => Ok(()),
        Some(_) => Err("attribute 'approximate' must be 'none' or 'tanh'".into()),
    }
}

fn one_hot(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 3, 1, 1)?;
    require_one_of(
        input_dtypes,
        0,
        &[DataType::Int32, DataType::Int64],
        "indices",
    )?;
    require_one_of(
        input_dtypes,
        1,
        &[DataType::Int32, DataType::Int64],
        "depth",
    )?;
    require_fixed_width(input_dtypes, 2, "values")
}
