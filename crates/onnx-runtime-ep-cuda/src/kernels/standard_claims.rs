//! Claim-time contracts for CUDA's constrained standard operators.
//!
//! These kernels intentionally cover only a subset of their ONNX dtype
//! matrices. Keep their placement checks in sync with those runtime limits so a
//! node is never claimed only to fail while constructing or executing a kernel.

use onnx_runtime_ir::{Attribute, DataType, Node, Shape};

pub(crate) fn unsupported_reason(
    node: &Node,
    input_shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Option<String> {
    let result = match node.op_type.as_str() {
        "RMSNormalization" => rms_normalization(node, input_dtypes),
        "RotaryEmbedding" => rotary_embedding(node, input_dtypes),
        "TopK" => topk(node, input_dtypes),
        "CumSum" => cumsum(node, input_dtypes),
        "Trilu" => trilu(node, input_dtypes),
        "Gather" => gather(node, input_dtypes),
        "GatherElements" => gather_elements(node, input_dtypes),
        "ScatterElements" => scatter_elements(node, input_dtypes),
        "ScatterND" => scatter_nd(node, input_dtypes),
        "HannWindow" | "HammingWindow" | "BlackmanWindow" => window(node, input_dtypes),
        "Where" => where_op(node, input_dtypes),
        "Expand" => expand(node, input_dtypes),
        "ConstantOfShape" => constant_of_shape(node, input_dtypes),
        "Gelu" => gelu(node, input_dtypes),
        "OneHot" => one_hot(node, input_dtypes),
        "GatherND" => gather_nd(node, input_dtypes),
        "SpaceToDepth" => space_to_depth(node, input_dtypes),
        "EyeLike" => eye_like(node, input_dtypes),
        "ReduceProd" | "ReduceSumSquare" | "ReduceL1" | "ReduceL2" | "ReduceLogSum"
        | "ReduceLogSumExp" => reduce_float(node, input_dtypes),
        "Swish" | "ThresholdedRelu" => float_activation(node, input_dtypes),
        "Sum" | "Mean" => variadic_float(node, input_dtypes),
        "Mod" => mod_op(node, input_dtypes),
        "QuantizeLinear" => quantize_linear(node, input_dtypes),
        "DequantizeLinear" => dequantize_linear(node, input_dtypes),
        "Dropout" => dropout(node, input_dtypes),
        "NonZero" => nonzero(node, input_dtypes),
        "AffineGrid" => affine_grid(node, input_dtypes),
        "BatchNormalization" => batch_normalization(node, input_dtypes),
        "Compress" => compress(node, input_dtypes),
        "DynamicQuantizeLinear" => dynamic_quantize_linear(node, input_dtypes),
        "GlobalAveragePool" | "GlobalMaxPool" => global_pool(node, input_dtypes),
        "GlobalLpPool" => global_lp_pool(node, input_dtypes),
        "LpNormalization" => lp_normalization(node, input_dtypes),
        "InstanceNormalization" | "GroupNormalization" => normalization(node, input_dtypes),
        "LpPool" => lp_pool(node, input_dtypes),
        "CenterCropPad" => center_crop_pad(node, input_dtypes),
        "Col2Im" => col2im(node, input_dtypes),
        "QLinearMatMul" => qlinear_matmul(node, input_dtypes),
        "Resize" => resize(node, input_dtypes),
        "ConvTranspose" => conv_transpose(node, input_shapes, input_dtypes),
        "GridSample" => grid_sample(node, input_shapes, input_dtypes),
        _ => return None,
    };
    result
        .err()
        .map(|reason| format!("{}: {reason}", node.op_type))
}

const CUDA_FLOAT_DTYPES: &[DataType] = &[DataType::Float32, DataType::Float16, DataType::BFloat16];

fn affine_grid(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 2, 1, 1)?;
    require_one_of(input_dtypes, 0, CUDA_FLOAT_DTYPES, "theta")?;
    require_dtype(input_dtypes, 1, DataType::Int64, "size")?;
    bool_attribute(node, "align_corners")
}

fn batch_normalization(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 5, 1, 1)?;
    require_one_of(input_dtypes, 0, CUDA_FLOAT_DTYPES, "X")?;
    for (index, name) in [(1, "scale"), (2, "B"), (3, "input_mean"), (4, "input_var")] {
        if input_dtypes[index] != input_dtypes[0] {
            return Err(format!(
                "input {index} ('{name}') dtype {:?} must match X dtype {:?}",
                input_dtypes[index], input_dtypes[0]
            ));
        }
    }
    match node.attr("training_mode") {
        None => Ok(()),
        Some(attribute) if attribute.as_int() == Some(0) => Ok(()),
        Some(_) => Err("attribute 'training_mode' must be 0".into()),
    }
}

fn compress(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 2, 1, 1)?;
    require_fixed_width(input_dtypes, 0, "input")?;
    require_dtype(input_dtypes, 1, DataType::Bool, "condition")
}

fn dynamic_quantize_linear(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 1, 3, 3)?;
    require_dtype(input_dtypes, 0, DataType::Float32, "X")
}

fn global_pool(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 1, 1, 1)?;
    require_one_of(input_dtypes, 0, CUDA_FLOAT_DTYPES, "X")
}

fn global_lp_pool(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    global_pool(node, input_dtypes)?;
    match node.attr("p") {
        None => Ok(()),
        Some(attribute) if matches!(attribute.as_int(), Some(value) if value > 0 && value <= i32::MAX as i64) => {
            Ok(())
        }
        Some(_) => Err("attribute 'p' must be a positive 32-bit integer".into()),
    }
}

fn lp_normalization(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    global_pool(node, input_dtypes)?;
    match node.attr("p") {
        None => Ok(()),
        Some(attribute) if matches!(attribute.as_int(), Some(1 | 2)) => Ok(()),
        Some(_) => Err("attribute 'p' must be 1 or 2".into()),
    }
}

fn normalization(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 3, 1, 1)?;
    require_one_of(input_dtypes, 0, CUDA_FLOAT_DTYPES, "X")?;
    if input_dtypes[1] != input_dtypes[0] || input_dtypes[2] != input_dtypes[0] {
        return Err("scale and bias dtypes must match X".into());
    }

    Ok(())
}

fn lp_pool(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 1, 1, 1)?;
    require_one_of(input_dtypes, 0, CUDA_FLOAT_DTYPES, "X")
}

fn center_crop_pad(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 2, 1, 1)?;
    require_fixed_width(input_dtypes, 0, "input")?;
    require_dtype(input_dtypes, 1, DataType::Int64, "shape")
}

fn col2im(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 3, 1, 1)?;
    require_one_of(input_dtypes, 0, CUDA_FLOAT_DTYPES, "input")?;
    require_dtype(input_dtypes, 1, DataType::Int64, "image_shape")?;
    require_dtype(input_dtypes, 2, DataType::Int64, "block_shape")
}

fn qlinear_matmul(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 8, 1, 1)?;
    for (index, name) in [(0, "A"), (3, "B"), (7, "y_zero_point")] {
        require_one_of(
            input_dtypes,
            index,
            &[DataType::Int8, DataType::Uint8],
            name,
        )?;
    }

    for (index, name) in [(1, "a_scale"), (4, "b_scale"), (6, "y_scale")] {
        require_dtype(input_dtypes, index, DataType::Float32, name)?;
    }
    if input_dtypes[2] != input_dtypes[0] {
        return Err("a_zero_point dtype must match A".into());
    }
    if input_dtypes[5] != input_dtypes[3] {
        return Err("b_zero_point dtype must match B".into());
    }
    Ok(())
}

fn conv_transpose(
    node: &Node,
    input_shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Result<(), String> {
    if !(2..=3).contains(&node.inputs.len()) || node.outputs.len() != 1 {
        return Err("requires X, W, optional B, and one output".into());
    }
    metadata_arity(node, input_dtypes)?;
    if node.inputs.iter().any(Option::is_none) {
        return Err("all declared inputs must be present".into());
    }
    require_one_of(input_dtypes, 0, CUDA_FLOAT_DTYPES, "X")?;
    if input_dtypes[1] != input_dtypes[0]
        || input_dtypes
            .get(2)
            .is_some_and(|dtype| *dtype != input_dtypes[0])
    {
        return Err("W and optional B must match X dtype".into());
    }
    match node.attr("auto_pad").and_then(Attribute::as_str) {
        None | Some("" | "NOTSET" | "VALID") => {}
        Some(value) => return Err(format!("auto_pad {value:?} is deferred")),
    }
    if node.attr("output_shape").is_some() {
        return Err("output_shape-driven padding is deferred".into());
    }
    require_input_rank(input_shapes, 0, &[3, 4], "X")?;
    Ok(())
}

fn grid_sample(
    node: &Node,
    input_shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Result<(), String> {
    required_arity(node, input_dtypes, 2, 1, 1)?;
    require_one_of(input_dtypes, 0, CUDA_FLOAT_DTYPES, "X")?;
    if input_dtypes[1] != input_dtypes[0] {
        return Err("grid dtype must match X".into());
    }
    match node
        .attr("mode")
        .and_then(Attribute::as_str)
        .unwrap_or("linear")
    {
        "linear" | "bilinear" | "nearest" => {}
        value => return Err(format!("mode {value:?} is deferred")),
    }
    match node
        .attr("padding_mode")
        .and_then(Attribute::as_str)
        .unwrap_or("zeros")
    {
        "zeros" | "border" | "reflection" => {}
        value => return Err(format!("padding_mode {value:?} unsupported")),
    }
    match node.attr("align_corners").and_then(Attribute::as_int) {
        None | Some(0 | 1) => {}
        Some(_) => return Err("align_corners must be 0 or 1".into()),
    }
    require_input_rank(input_shapes, 0, &[4], "X")
}

fn require_input_rank(
    input_shapes: &[Shape],
    index: usize,
    supported_ranks: &[usize],
    name: &str,
) -> Result<(), String> {
    let rank = input_shapes
        .get(index)
        .ok_or_else(|| format!("missing shape metadata for input {index} ('{name}')"))?
        .len();
    if supported_ranks.contains(&rank) {
        Ok(())
    } else {
        Err(format!(
            "input {index} ('{name}') rank {rank} unsupported; expected one of {supported_ranks:?}"
        ))
    }
}

fn resize(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    if node.outputs.len() != 1 || node.inputs.is_empty() || node.inputs.len() > 4 {
        return Err("requires 1..=4 inputs and exactly 1 output".into());
    }
    metadata_arity(node, input_dtypes)?;
    if node.inputs[0].is_none() {
        return Err("requires present data input".into());
    }
    require_one_of(input_dtypes, 0, CUDA_FLOAT_DTYPES, "X")?;
    match node.attr("mode").and_then(Attribute::as_str) {
        None | Some("nearest" | "linear") => {}
        Some(value) => {
            return Err(format!(
                "mode {value:?} unsupported; expected nearest or linear"
            ));
        }
    }
    match node
        .attr("coordinate_transformation_mode")
        .and_then(Attribute::as_str)
    {
        None | Some("half_pixel" | "align_corners" | "asymmetric") => {}
        Some(value) => {
            return Err(format!(
                "coordinate_transformation_mode {value:?} unsupported"
            ));
        }
    }
    match node.attr("nearest_mode").and_then(Attribute::as_str) {
        None | Some("round_prefer_floor" | "round_prefer_ceil" | "floor" | "ceil") => {}
        Some(value) => return Err(format!("nearest_mode {value:?} unsupported")),
    }
    match node.attr("antialias").and_then(Attribute::as_int) {
        None | Some(0) => {}
        Some(_) => return Err("antialias=1 unsupported".into()),
    }
    match node
        .attr("keep_aspect_ratio_policy")
        .and_then(Attribute::as_str)
    {
        None | Some("stretch") => {}
        Some(value) => return Err(format!("keep_aspect_ratio_policy {value:?} unsupported")),
    }
    if node.inputs.len() == 2 {
        if node.inputs[1].is_none() {
            return Err("opset-10 form requires a present scales input".into());
        }
        return require_dtype(input_dtypes, 1, DataType::Float32, "scales");
    }
    let scales_present = node.inputs.get(2).is_some_and(Option::is_some);
    let sizes_present = node.inputs.get(3).is_some_and(Option::is_some);
    if !scales_present && !sizes_present {
        return Err("requires a present scales or sizes input".into());
    }
    if scales_present {
        require_dtype(input_dtypes, 2, DataType::Float32, "scales")?;
    }
    if sizes_present {
        require_dtype(input_dtypes, 3, DataType::Int64, "sizes")?;
    }
    Ok(())
}

fn quantize_linear(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, node.inputs.len(), 1, 1)?;
    if !(2..=3).contains(&node.inputs.len()) {
        return Err("requires 2 or 3 inputs".into());
    }
    require_dtype(input_dtypes, 0, DataType::Float32, "x")?;
    require_dtype(input_dtypes, 1, DataType::Float32, "scale")?;
    if input_dtypes.len() == 3 {
        require_one_of(
            input_dtypes,
            2,
            &[DataType::Int8, DataType::Uint8],
            "zero_point",
        )?;
    }
    Ok(())
}

fn dequantize_linear(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, node.inputs.len(), 1, 1)?;
    if !(2..=3).contains(&node.inputs.len()) {
        return Err("requires 2 or 3 inputs".into());
    }
    require_one_of(input_dtypes, 0, &[DataType::Int8, DataType::Uint8], "x")?;
    require_dtype(input_dtypes, 1, DataType::Float32, "scale")?;
    if input_dtypes.len() == 3 && input_dtypes[2] != input_dtypes[0] {
        return Err("zero_point dtype must match x".into());
    }
    Ok(())
}

fn dropout(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 1, 1, 2)?;
    require_fixed_width(input_dtypes, 0, "data")
}

fn nonzero(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 1, 1, 1)?;
    require_one_of(
        input_dtypes,
        0,
        &[
            DataType::Float32,
            DataType::Float16,
            DataType::BFloat16,
            DataType::Bool,
        ],
        "X",
    )
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
    require_one_of(input_dtypes, 0, CUDA_FLOAT_DTYPES, "X")?;
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

fn trilu(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    if !(1..=2).contains(&node.inputs.len())
        || node.outputs.len() != 1
        || node.inputs.first().is_none_or(Option::is_none)
    {
        return Err(format!(
            "requires 1-2 inputs with the matrix present and 1 output, got {} inputs and {} outputs",
            node.inputs.len(),
            node.outputs.len()
        ));
    }
    metadata_arity(node, input_dtypes)?;
    require_fixed_width(input_dtypes, 0, "input")?;
    if node.inputs.get(1).is_some_and(Option::is_some) {
        require_dtype(input_dtypes, 1, DataType::Int64, "k")?;
    }
    bool_attribute(node, "upper")
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

fn gather_nd(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 2, 1, 1)?;
    require_fixed_width(input_dtypes, 0, "data")?;
    require_one_of(
        input_dtypes,
        1,
        &[DataType::Int32, DataType::Int64],
        "indices",
    )?;
    if node
        .attr("batch_dims")
        .is_some_and(|attribute| !matches!(attribute.as_int(), Some(value) if value >= 0))
    {
        return Err("attribute 'batch_dims' must be a non-negative integer".into());
    }
    Ok(())
}

fn space_to_depth(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 1, 1, 1)?;
    require_fixed_width(input_dtypes, 0, "input")?;
    if !matches!(node.attr("blocksize").and_then(Attribute::as_int), Some(value) if value > 0) {
        return Err("attribute 'blocksize' must be a positive integer".into());
    }
    Ok(())
}

fn eye_like(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 1, 1, 1)?;
    require_fixed_width(input_dtypes, 0, "input")?;
    if node
        .attr("k")
        .is_some_and(|attribute| attribute.as_int().is_none())
    {
        return Err("attribute 'k' must be an integer".into());
    }
    if let Some(attribute) = node.attr("dtype") {
        let value = attribute
            .as_int()
            .ok_or_else(|| "attribute 'dtype' must be an integer".to_string())?;
        let value = i32::try_from(value)
            .map_err(|_| format!("attribute 'dtype' value {value} is invalid"))?;
        let dtype = DataType::from_onnx(value)
            .ok_or_else(|| format!("attribute 'dtype' value {value} is invalid"))?;
        if !matches!(
            dtype,
            DataType::Bool
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::Uint8
                | DataType::Uint16
                | DataType::Uint32
                | DataType::Uint64
                | DataType::Float16
                | DataType::BFloat16
                | DataType::Float32
                | DataType::Float64
        ) {
            return Err(format!("attribute 'dtype' selects unsupported {dtype:?}"));
        }
    }
    Ok(())
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
    scatter_reduction(node)
}

fn scatter_reduction(node: &Node) -> Result<(), String> {
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

fn scatter_nd(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
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
    require_dtype(input_dtypes, 1, DataType::Int64, "indices")?;
    require_dtype(input_dtypes, 2, input_dtypes[0], "updates")?;
    scatter_reduction(node)
}

fn window(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 1, 1, 1)?;
    require_dtype(input_dtypes, 0, DataType::Int64, "size")?;
    bool_attribute(node, "periodic")?;
    if let Some(attribute) = node.attr("output_datatype") {
        let value = attribute
            .as_int()
            .ok_or_else(|| "attribute 'output_datatype' must be an integer".to_string())?;
        let value = i32::try_from(value)
            .map_err(|_| format!("attribute 'output_datatype' value {value} is invalid"))?;
        let dtype = DataType::from_onnx(value)
            .ok_or_else(|| format!("attribute 'output_datatype' value {value} is invalid"))?;
        if !matches!(
            dtype,
            DataType::Float16 | DataType::BFloat16 | DataType::Float32 | DataType::Float64
        ) {
            return Err(format!(
                "attribute 'output_datatype' selects unsupported {dtype:?}"
            ));
        }
    }
    Ok(())
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

/// `ReduceProd`/`ReduceSumSquare`/`ReduceL1`/`ReduceL2`/`ReduceLogSum`/
/// `ReduceLogSumExp`: floating-point block reductions with f32 accumulation and
/// an optional opset-18 int32/int64 `axes` input.
fn reduce_float(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    if !(1..=2).contains(&node.inputs.len())
        || node.outputs.len() != 1
        || node.inputs.first().is_none_or(Option::is_none)
    {
        return Err(format!(
            "requires 1-2 inputs with data present and 1 output, got {} inputs and {} outputs",
            node.inputs.len(),
            node.outputs.len()
        ));
    }
    metadata_arity(node, input_dtypes)?;
    require_one_of(
        input_dtypes,
        0,
        &[DataType::Float16, DataType::Float32, DataType::BFloat16],
        "data",
    )?;
    if node.inputs.get(1).is_some_and(Option::is_some) {
        require_one_of(input_dtypes, 1, &[DataType::Int32, DataType::Int64], "axes")?;
    }
    bool_attribute(node, "keepdims")?;
    bool_attribute(node, "noop_with_empty_axes")
}

/// `Swish`/`ThresholdedRelu`: single float input activations (f32/f16/bf16).
fn float_activation(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 1, 1, 1)?;
    require_one_of(
        input_dtypes,
        0,
        &[DataType::Float16, DataType::Float32, DataType::BFloat16],
        "X",
    )
}

/// `Sum`/`Mean`: variadic float inputs (f32/f16/bf16), all sharing one dtype.
fn variadic_float(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    if node.inputs.is_empty() || node.outputs.len() != 1 || node.inputs.iter().any(Option::is_none)
    {
        return Err(format!(
            "requires 1+ present inputs and 1 output, got {} inputs and {} outputs",
            node.inputs.len(),
            node.outputs.len()
        ));
    }
    metadata_arity(node, input_dtypes)?;
    for index in 0..input_dtypes.len() {
        require_one_of(
            input_dtypes,
            index,
            &[DataType::Float16, DataType::Float32, DataType::BFloat16],
            "input",
        )?;
        if input_dtypes[index] != input_dtypes[0] {
            return Err(format!(
                "input {index} dtype {:?} must match input 0 dtype {:?}",
                input_dtypes[index], input_dtypes[0]
            ));
        }
    }
    Ok(())
}

/// `Mod`: two same-dtype operands over f32/i32/i64; f32 requires `fmod=1`.
fn mod_op(node: &Node, input_dtypes: &[DataType]) -> Result<(), String> {
    required_arity(node, input_dtypes, 2, 1, 1)?;
    require_one_of(
        input_dtypes,
        0,
        &[DataType::Float32, DataType::Int32, DataType::Int64],
        "A",
    )?;
    require_dtype(input_dtypes, 1, input_dtypes[0], "B")?;
    let fmod = match node.attr("fmod") {
        None => 0,
        Some(attribute) => match attribute.as_int() {
            Some(value @ (0 | 1)) => value,
            _ => return Err("attribute 'fmod' must be 0 or 1".into()),
        },
    };
    if input_dtypes[0] == DataType::Float32 && fmod != 1 {
        return Err("Float32 Mod requires fmod=1".into());
    }
    Ok(())
}
