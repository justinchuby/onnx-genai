//! CPU parity oracle for frozen `pkg.nxrt::BlockQuantizedMoE` v1.

use std::borrow::Cow;
use std::collections::BTreeMap;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node, Shape, as_static_shape};

use super::block_quantized_matmul::{BlockFormat, dequantize_weight_kn};
use super::moe::{MoeAttributes, routing_weights, run_expert};
use super::{check_arity, to_dense_bytes, to_dense_f32, write_dense_f32};

const OP: &str = "BlockQuantizedMoE";
const LAYOUT_VERSION: i64 = 1;

const INPUT_NAMES: [&str; 9] = [
    "input",
    "router_logits",
    "fc1_experts_weights",
    "fc1_experts_bias",
    "fc2_experts_weights",
    "fc2_experts_bias",
    "fc3_experts_weights",
    "fc3_experts_bias",
    "router_weights",
];

pub struct BlockQuantizedMoEFactory;

pub struct BlockQuantizedMoEKernel {
    attributes: MoeAttributes,
    format: BlockFormat,
}

impl KernelFactory for BlockQuantizedMoEFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        validate_attributes(node)?;
        let attributes = MoeAttributes::from_block_quantized_node(node)?;
        let layout_version = optional_int_attr(node, "block_layout_version")?.unwrap_or(1);
        if layout_version != LAYOUT_VERSION {
            return Err(error(format!(
                "block_layout_version must be {LAYOUT_VERSION}, got {layout_version}"
            )));
        }
        let format = node
            .attr("format")
            .ok_or_else(|| error("missing required string attribute 'format'"))?
            .as_str()
            .ok_or_else(|| error("attribute 'format' must be a UTF-8 string"))
            .and_then(BlockFormat::parse)?;
        Ok(Box::new(BlockQuantizedMoEKernel { attributes, format }))
    }
}

pub(crate) fn unsupported_reason(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Option<Cow<'static, str>> {
    if let Err(err) = BlockQuantizedMoEFactory.create(node, &[]) {
        return Some(Cow::Owned(err.to_string()));
    }
    let result = validate_claim_metadata(node, shapes, input_dtypes);
    result
        .err()
        .map(|reason| Cow::Owned(format!("{OP}: {reason}")))
}

impl Kernel for BlockQuantizedMoEKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(OP, inputs, outputs, 5, 9, 1)?;
        for &index in &[0, 1, 2, 4] {
            if inputs[index].is_absent() {
                return Err(error(format!(
                    "required input {index} ('{}') is absent",
                    INPUT_NAMES[index]
                )));
            }
        }
        for &index in &[0, 1] {
            require_dtype(index, &inputs[index], DataType::Float32)?;
        }
        for &index in &[2, 4] {
            require_dtype(index, &inputs[index], DataType::Uint8)?;
        }
        for &index in &[3, 5, 7, 8] {
            if let Some(input) = optional_input(inputs, index) {
                require_dtype(index, input, DataType::Float32)?;
            }
        }
        if let Some(input) = optional_input(inputs, 6) {
            require_dtype(6, input, DataType::Uint8)?;
        }
        if outputs[0].dtype != DataType::Float32 {
            return Err(error(format!(
                "output dtype {:?} unsupported; expected Float32",
                outputs[0].dtype
            )));
        }

        let dimensions =
            validate_runtime_shapes(inputs, &outputs[0], &self.attributes, self.format)?;
        let Dimensions {
            rows,
            hidden,
            experts,
            inter,
            fc1_size,
        } = dimensions;

        let input = to_dense_f32(&inputs[0])?;
        let router_logits = to_dense_f32(&inputs[1])?;
        let router_weights = optional_dense(inputs, 8)?;
        let fc1_packed = to_dense_bytes(&inputs[2])?;
        let fc2_packed = to_dense_bytes(&inputs[4])?;
        let fc3_packed = optional_input(inputs, 6).map(to_dense_bytes).transpose()?;
        let fc1_bias = optional_dense(inputs, 3)?;
        let fc2_bias = optional_dense(inputs, 5)?;
        let fc3_bias = optional_dense(inputs, 7)?;

        let mut tasks = BTreeMap::<usize, Vec<(usize, f32)>>::new();
        for row in 0..rows {
            let range = row * experts..(row + 1) * experts;
            let mut route = routing_weights(
                &router_logits[range.clone()],
                router_weights
                    .as_deref()
                    .map(|weights| &weights[range.clone()]),
                self.attributes.k,
                self.attributes.normalize_routing_weights,
            );
            route.sort_unstable_by_key(|&(expert, _)| expert);
            for (expert, weight) in route {
                tasks.entry(expert).or_default().push((row, weight));
            }
        }

        let mut output = vec![0.0f32; rows * hidden];
        for (expert, expert_tasks) in tasks {
            let fc1 =
                dequantize_expert(self.format, &fc1_packed, expert, fc1_size, hidden, experts)?;
            let fc2 = dequantize_expert(self.format, &fc2_packed, expert, hidden, inter, experts)?;
            let fc3 = fc3_packed
                .as_deref()
                .map(|packed| {
                    dequantize_expert(self.format, packed, expert, inter, hidden, experts)
                })
                .transpose()?;

            for (row, route_weight) in expert_tasks {
                let expert_output = run_expert(
                    &input[row * hidden..(row + 1) * hidden],
                    &fc1,
                    fc1_bias
                        .as_deref()
                        .map(|bias| &bias[expert * fc1_size..(expert + 1) * fc1_size]),
                    &fc2,
                    fc2_bias
                        .as_deref()
                        .map(|bias| &bias[expert * hidden..(expert + 1) * hidden]),
                    fc3.as_deref(),
                    fc3_bias
                        .as_deref()
                        .map(|bias| &bias[expert * inter..(expert + 1) * inter]),
                    fc1_size,
                    hidden,
                    inter,
                    &self.attributes,
                );
                for feature in 0..hidden {
                    output[row * hidden + feature] += route_weight * expert_output[feature];
                }
            }
        }
        write_dense_f32(&mut outputs[0], &output)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[derive(Clone, Copy)]
struct Dimensions {
    rows: usize,
    hidden: usize,
    experts: usize,
    inter: usize,
    fc1_size: usize,
}

fn validate_runtime_shapes(
    inputs: &[TensorView],
    output: &TensorMut,
    attributes: &MoeAttributes,
    format: BlockFormat,
) -> Result<Dimensions> {
    let input_shape = inputs[0].shape;
    if !matches!(input_shape.len(), 2 | 3) {
        return Err(error(format!(
            "input must be [rows,H] or [B,S,H], got {input_shape:?}"
        )));
    }
    if output.shape != input_shape {
        return Err(error(format!(
            "output shape {:?} must equal input shape {input_shape:?}",
            output.shape
        )));
    }
    let hidden = *input_shape.last().expect("validated non-empty input rank");
    let rows = checked_product(&input_shape[..input_shape.len() - 1], "input rows")?;
    require_exact_rank(1, inputs[1].shape, 2)?;
    if inputs[1].shape[0] != rows {
        return Err(error(format!(
            "router_logits rows {} must equal flattened input rows {rows}",
            inputs[1].shape[0]
        )));
    }
    let experts = inputs[1].shape[1];
    if attributes.k > experts {
        return Err(error(format!(
            "requires 0 < k <= num_experts, got k={} and num_experts={experts}",
            attributes.k
        )));
    }
    require_exact_rank(2, inputs[2].shape, 4)?;
    require_exact_rank(4, inputs[4].shape, 4)?;
    if inputs[2].shape[0] != experts || inputs[4].shape[0] != experts {
        return Err(error(format!(
            "expert weight counts must equal router num_experts {experts}"
        )));
    }
    if inputs[4].shape[1] != hidden {
        return Err(error(format!(
            "fc2_experts_weights must start [experts={experts}, H={hidden}], got {:?}",
            inputs[4].shape
        )));
    }
    let fc1_size = inputs[2].shape[1];
    let inter = if attributes.swiglu_fusion == 0 {
        fc1_size
    } else {
        if fc1_size % 2 != 0 {
            return Err(error(format!(
                "fused SwiGLU fc1_out must be even, got {fc1_size}"
            )));
        }
        fc1_size / 2
    };
    if inter == 0 {
        return Err(error("inferred inter dimension must be non-zero"));
    }
    let expected_fc1 = attributes.checked_fc1_size(inter, OP)?;
    if fc1_size != expected_fc1 {
        return Err(error(format!(
            "fc1_experts_weights dimension 1 must be {expected_fc1}, got {fc1_size}"
        )));
    }
    validate_packed_shape(2, inputs[2].shape, experts, fc1_size, hidden, format)?;
    validate_packed_shape(4, inputs[4].shape, experts, hidden, inter, format)?;
    validate_bias(inputs, 3, experts, fc1_size)?;
    validate_bias(inputs, 5, experts, hidden)?;

    let has_fc3 = optional_input(inputs, 6).is_some();
    if attributes.uses_separate_gate(has_fc3) {
        let fc3 = optional_input(inputs, 6)
            .ok_or_else(|| error("unfused swiglu requires input 6 fc3_experts_weights"))?;
        validate_packed_shape(6, fc3.shape, experts, inter, hidden, format)?;
        validate_bias(inputs, 7, experts, inter)?;
    } else {
        if has_fc3 {
            return Err(error(
                "fc3_experts_weights is only valid for unfused swiglu or silu gated-GLU",
            ));
        }
        if optional_input(inputs, 7).is_some() {
            return Err(error("fc3_experts_bias requires fc3_experts_weights"));
        }
    }
    if let Some(weights) = optional_input(inputs, 8)
        && weights.shape != [rows, experts]
    {
        return Err(error(format!(
            "router_weights must have shape [{rows}, {experts}], got {:?}",
            weights.shape
        )));
    }
    Ok(Dimensions {
        rows,
        hidden,
        experts,
        inter,
        fc1_size,
    })
}

fn validate_packed_shape(
    index: usize,
    shape: &[usize],
    experts: usize,
    out_features: usize,
    in_features: usize,
    format: BlockFormat,
) -> Result<()> {
    let expected = [
        experts,
        out_features,
        in_features.div_ceil(format.qk()),
        format.block_bytes(),
    ];
    if shape != expected {
        return Err(error(format!(
            "input {index} ('{}') must have shape {expected:?}, got {shape:?}",
            INPUT_NAMES[index]
        )));
    }
    Ok(())
}

fn dequantize_expert(
    format: BlockFormat,
    packed: &[u8],
    expert: usize,
    out_features: usize,
    in_features: usize,
    experts: usize,
) -> Result<Vec<f32>> {
    let expert_bytes = out_features
        .checked_mul(in_features.div_ceil(format.qk()))
        .and_then(|count| count.checked_mul(format.block_bytes()))
        .ok_or_else(|| error("packed expert byte count overflow"))?;
    let expected = expert_bytes
        .checked_mul(experts)
        .ok_or_else(|| error("packed projection byte count overflow"))?;
    if packed.len() != expected {
        return Err(error(format!(
            "packed projection contains {} bytes, expected {expected}",
            packed.len()
        )));
    }
    let start = expert
        .checked_mul(expert_bytes)
        .ok_or_else(|| error("expert byte offset overflow"))?;
    let weight_kn = dequantize_weight_kn(
        format,
        in_features,
        out_features,
        &packed[start..start + expert_bytes],
    )?;
    let mut weight_nk = vec![0.0f32; weight_kn.len()];
    for input in 0..in_features {
        for output in 0..out_features {
            weight_nk[output * in_features + input] = weight_kn[input * out_features + output];
        }
    }
    Ok(weight_nk)
}

fn validate_attributes(node: &Node) -> Result<()> {
    for name in node.attributes.keys() {
        if !matches!(
            name.as_str(),
            "k" | "activation_type"
                | "normalize_routing_weights"
                | "swiglu_fusion"
                | "activation_alpha"
                | "activation_beta"
                | "swiglu_limit"
                | "format"
                | "block_layout_version"
        ) {
            return Err(error(format!(
                "attribute '{name}' is not part of the frozen v1 ABI"
            )));
        }
    }
    Ok(())
}

fn validate_claim_metadata(
    node: &Node,
    shapes: &[Shape],
    dtypes: &[DataType],
) -> std::result::Result<(), String> {
    if !(5..=9).contains(&node.inputs.len()) {
        return Err(format!(
            "expected 5 to 9 positional inputs, got {}",
            node.inputs.len()
        ));
    }
    if node.outputs.len() != 1 {
        return Err(format!(
            "expected exactly 1 output, got {}",
            node.outputs.len()
        ));
    }
    if shapes.len() != node.inputs.len() || dtypes.len() != node.inputs.len() {
        return Err(format!(
            "claim metadata must cover all {} positional inputs (got {} shapes and {} dtypes)",
            node.inputs.len(),
            shapes.len(),
            dtypes.len()
        ));
    }
    for &index in &[0, 1, 2, 4] {
        if node.inputs[index].is_none() {
            return Err(format!(
                "required input {index} ('{}') is omitted",
                INPUT_NAMES[index]
            ));
        }
    }
    for index in 0..node.inputs.len() {
        if node.inputs[index].is_none() {
            if dtypes[index] != DataType::Undefined {
                return Err(format!(
                    "omitted input {index} ('{}') must use dtype Undefined",
                    INPUT_NAMES[index]
                ));
            }
            continue;
        }
        let expected = if matches!(index, 2 | 4 | 6) {
            DataType::Uint8
        } else {
            DataType::Float32
        };
        if dtypes[index] != expected {
            return Err(format!(
                "input {index} ('{}') dtype {:?} unsupported; expected {expected:?}",
                INPUT_NAMES[index], dtypes[index]
            ));
        }
    }
    if !matches!(shapes[0].len(), 2 | 3) {
        return Err(format!(
            "input 0 ('input') rank {} unsupported; expected 2 or 3",
            shapes[0].len()
        ));
    }
    if shapes[1].len() != 2 {
        return Err(format!(
            "input 1 ('router_logits') rank {} unsupported; expected 2",
            shapes[1].len()
        ));
    }
    for &index in &[2, 4] {
        if shapes[index].len() != 4 {
            return Err(format!(
                "input {index} ('{}') rank {} unsupported; expected 4",
                INPUT_NAMES[index],
                shapes[index].len()
            ));
        }
    }
    for &index in &[3, 5, 7, 8] {
        if node.inputs.get(index).is_some_and(Option::is_some) && shapes[index].len() != 2 {
            return Err(format!(
                "input {index} ('{}') rank {} unsupported; expected 2",
                INPUT_NAMES[index],
                shapes[index].len()
            ));
        }
    }
    if node.inputs.get(6).is_some_and(Option::is_some) && shapes[6].len() != 4 {
        return Err(format!(
            "input 6 ('fc3_experts_weights') rank {} unsupported; expected 4",
            shapes[6].len()
        ));
    }
    validate_partial_claim_shapes(node, shapes)?;
    if let Some(static_shapes) = shapes
        .iter()
        .map(|shape| as_static_shape(shape))
        .collect::<Option<Vec<_>>>()
    {
        let views = static_shapes;
        validate_static_claim_shapes(node, &views)?;
    }
    Ok(())
}

fn validate_partial_claim_shapes(node: &Node, shapes: &[Shape]) -> std::result::Result<(), String> {
    let format = node
        .attr("format")
        .and_then(|attr| attr.as_str())
        .ok_or_else(|| "missing format".to_string())
        .and_then(|value| BlockFormat::parse(value).map_err(|err| err.to_string()))?;
    let attributes =
        MoeAttributes::from_block_quantized_node(node).map_err(|err| err.to_string())?;
    let hidden = shapes[0].last().and_then(|dim| dim.as_static());
    let experts = shapes[1][1].as_static();
    if let Some(experts) = experts
        && attributes.k > experts
    {
        return Err(format!("k={} exceeds num_experts={experts}", attributes.k));
    }
    require_same_static_axis(shapes, 2, 0, 1, 1, "fc1 expert count")?;
    require_same_static_axis(shapes, 4, 0, 1, 1, "fc2 expert count")?;
    if let (Some(fc2_hidden), Some(hidden)) = (shapes[4][1].as_static(), hidden)
        && fc2_hidden != hidden
    {
        return Err(format!(
            "fc2 output width {fc2_hidden} must equal hidden size {hidden}"
        ));
    }
    if let Some(block_bytes) = shapes[2][3].as_static()
        && block_bytes != format.block_bytes()
    {
        return Err(format!(
            "fc1 block byte width {block_bytes} must equal {}",
            format.block_bytes()
        ));
    }
    if let Some(block_bytes) = shapes[4][3].as_static()
        && block_bytes != format.block_bytes()
    {
        return Err(format!(
            "fc2 block byte width {block_bytes} must equal {}",
            format.block_bytes()
        ));
    }
    let fc1_size = shapes[2][1].as_static();
    let inter = fc1_size.and_then(|fc1_size| {
        if attributes.swiglu_fusion == 0 {
            Some(fc1_size)
        } else {
            (fc1_size % 2 == 0).then_some(fc1_size / 2)
        }
    });
    if attributes.swiglu_fusion != 0 && fc1_size.is_some() && inter.is_none() {
        return Err("fused SwiGLU fc1_out must be even".into());
    }
    if let (Some(blocks), Some(hidden)) = (shapes[2][2].as_static(), hidden) {
        let expected = hidden.div_ceil(format.qk());
        if blocks != expected {
            return Err(format!("fc1 block count {blocks} must equal {expected}"));
        }
    }
    if let (Some(blocks), Some(inter)) = (shapes[4][2].as_static(), inter) {
        let expected = inter.div_ceil(format.qk());
        if blocks != expected {
            return Err(format!("fc2 block count {blocks} must equal {expected}"));
        }
    }
    let has_fc3 = node.inputs.get(6).is_some_and(Option::is_some);
    if attributes.uses_separate_gate(has_fc3) {
        if !has_fc3 {
            return Err("unfused swiglu requires fc3_experts_weights".into());
        }
        require_same_static_axis(shapes, 6, 0, 1, 1, "fc3 expert count")?;
        if let Some(block_bytes) = shapes[6][3].as_static()
            && block_bytes != format.block_bytes()
        {
            return Err(format!(
                "fc3 block byte width {block_bytes} must equal {}",
                format.block_bytes()
            ));
        }
    } else if has_fc3 || node.inputs.get(7).is_some_and(Option::is_some) {
        return Err("fc3 inputs are only valid for unfused swiglu or silu gated-GLU".into());
    }
    Ok(())
}

fn require_same_static_axis(
    shapes: &[Shape],
    left_input: usize,
    left_axis: usize,
    right_input: usize,
    right_axis: usize,
    name: &str,
) -> std::result::Result<(), String> {
    if let (Some(left), Some(right)) = (
        shapes[left_input][left_axis].as_static(),
        shapes[right_input][right_axis].as_static(),
    ) && left != right
    {
        return Err(format!("{name} {left} must equal {right}"));
    }
    Ok(())
}

fn validate_static_claim_shapes(
    node: &Node,
    shapes: &[Vec<usize>],
) -> std::result::Result<(), String> {
    let factory = BlockQuantizedMoEFactory;
    let kernel = factory
        .create(node, shapes)
        .map_err(|err| err.to_string())?;
    let _ = kernel;
    let format = node
        .attr("format")
        .and_then(|attr| attr.as_str())
        .ok_or_else(|| "missing format".to_string())
        .and_then(|value| BlockFormat::parse(value).map_err(|err| err.to_string()))?;
    let input = &shapes[0];
    let hidden = *input
        .last()
        .ok_or_else(|| "input rank is empty".to_string())?;
    let rows = input[..input.len() - 1]
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .ok_or_else(|| "input row count overflow".to_string())?;
    if shapes[1][0] != rows {
        return Err(format!(
            "router_logits rows {} must equal flattened input rows {rows}",
            shapes[1][0]
        ));
    }
    let experts = shapes[1][1];
    if shapes[2][0] != experts || shapes[4][0] != experts {
        return Err("expert weight counts must equal router expert count".into());
    }
    if shapes[4][1] != hidden {
        return Err(format!("fc2 output width must equal hidden size {hidden}"));
    }
    let attrs = MoeAttributes::from_block_quantized_node(node).map_err(|err| err.to_string())?;
    if attrs.k > experts {
        return Err(format!("k={} exceeds num_experts={experts}", attrs.k));
    }
    let fc1_size = shapes[2][1];
    let inter = if attrs.swiglu_fusion == 0 {
        fc1_size
    } else if fc1_size % 2 == 0 {
        fc1_size / 2
    } else {
        return Err("fused SwiGLU fc1_out must be even".into());
    };
    check_claim_packed(&shapes[2], experts, fc1_size, hidden, format, 2)?;
    check_claim_packed(&shapes[4], experts, hidden, inter, format, 4)?;
    check_claim_optional_shape(node, shapes, 3, &[experts, fc1_size])?;
    check_claim_optional_shape(node, shapes, 5, &[experts, hidden])?;
    let has_fc3 = node.inputs.get(6).is_some_and(Option::is_some);
    if attrs.uses_separate_gate(has_fc3) {
        if !has_fc3 {
            return Err("unfused swiglu requires fc3_experts_weights".into());
        }
        check_claim_packed(&shapes[6], experts, inter, hidden, format, 6)?;
        check_claim_optional_shape(node, shapes, 7, &[experts, inter])?;
    } else if has_fc3 || node.inputs.get(7).is_some_and(Option::is_some) {
        return Err("fc3 inputs are only valid for unfused swiglu or silu gated-GLU".into());
    }
    check_claim_optional_shape(node, shapes, 8, &[rows, experts])?;
    Ok(())
}

fn check_claim_packed(
    shape: &[usize],
    experts: usize,
    out_features: usize,
    in_features: usize,
    format: BlockFormat,
    index: usize,
) -> std::result::Result<(), String> {
    let expected = [
        experts,
        out_features,
        in_features.div_ceil(format.qk()),
        format.block_bytes(),
    ];
    if shape != expected {
        return Err(format!(
            "input {index} ('{}') shape {shape:?} unsupported; expected {expected:?}",
            INPUT_NAMES[index]
        ));
    }
    Ok(())
}

fn check_claim_optional_shape(
    node: &Node,
    shapes: &[Vec<usize>],
    index: usize,
    expected: &[usize],
) -> std::result::Result<(), String> {
    if node.inputs.get(index).is_some_and(Option::is_some) && shapes[index] != expected {
        return Err(format!(
            "input {index} ('{}') shape {:?} unsupported; expected {expected:?}",
            INPUT_NAMES[index], shapes[index]
        ));
    }
    Ok(())
}

fn validate_bias(inputs: &[TensorView], index: usize, experts: usize, width: usize) -> Result<()> {
    if let Some(input) = optional_input(inputs, index)
        && input.shape != [experts, width]
    {
        return Err(error(format!(
            "{} must have shape [{experts}, {width}], got {:?}",
            INPUT_NAMES[index], input.shape
        )));
    }
    Ok(())
}

fn require_exact_rank(index: usize, shape: &[usize], expected: usize) -> Result<()> {
    if shape.len() != expected {
        return Err(error(format!(
            "input {index} ('{}') must have rank {expected}, got {shape:?}",
            INPUT_NAMES[index]
        )));
    }
    Ok(())
}

fn require_dtype(index: usize, input: &TensorView, expected: DataType) -> Result<()> {
    if input.dtype != expected {
        return Err(error(format!(
            "input {index} ('{}') dtype {:?} unsupported; expected {expected:?}",
            INPUT_NAMES[index], input.dtype
        )));
    }
    Ok(())
}

fn optional_input<'a>(inputs: &'a [TensorView<'a>], index: usize) -> Option<&'a TensorView<'a>> {
    inputs.get(index).filter(|input| !input.is_absent())
}

fn optional_dense(inputs: &[TensorView], index: usize) -> Result<Option<Vec<f32>>> {
    optional_input(inputs, index).map(to_dense_f32).transpose()
}

fn checked_product(shape: &[usize], name: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or_else(|| error(format!("{name} overflow")))
    })
}

fn optional_int_attr(node: &Node, name: &str) -> Result<Option<i64>> {
    node.attr(name)
        .map(|attr| {
            attr.as_int()
                .ok_or_else(|| error(format!("attribute '{name}' must be an integer")))
        })
        .transpose()
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{OP}: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuExecutionProvider;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{Attribute, Graph, NodeId, static_shape};

    const H: usize = 32;
    const E: usize = 2;

    fn attrs(
        activation: &str,
        k: usize,
        normalize: bool,
        swiglu_fusion: usize,
    ) -> Vec<(&'static str, Attribute)> {
        vec![
            ("format", Attribute::String(b"mxfp4".to_vec())),
            ("block_layout_version", Attribute::Int(1)),
            (
                "activation_type",
                Attribute::String(activation.as_bytes().to_vec()),
            ),
            ("k", Attribute::Int(k as i64)),
            (
                "normalize_routing_weights",
                Attribute::Int(i64::from(normalize)),
            ),
            ("swiglu_fusion", Attribute::Int(swiglu_fusion as i64)),
        ]
    }

    fn model_node(
        shapes: &[Option<(DataType, Vec<usize>)>],
        attrs: &[(&str, Attribute)],
    ) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert("pkg.nxrt".into(), 1);
        let inputs = shapes
            .iter()
            .enumerate()
            .map(|(index, input)| {
                input.as_ref().map(|(dtype, shape)| {
                    let value = graph.create_named_value(
                        format!("input_{index}"),
                        *dtype,
                        static_shape(shape.iter().copied()),
                    );
                    graph.add_input(value);
                    value
                })
            })
            .collect();
        let output = graph.create_named_value("output", DataType::Float32, static_shape([1, H]));
        let mut node = Node::new(NodeId(0), OP, inputs, vec![output]);
        node.domain = "pkg.nxrt".into();
        for (name, value) in attrs {
            node.attributes.insert((*name).into(), value.clone());
        }
        let node = graph.insert_node(node);
        graph.add_output(output);
        (graph, node)
    }

    fn packed_matrix(
        experts: usize,
        out_features: usize,
        mut code: impl FnMut(usize, usize, usize) -> u8,
    ) -> Vec<u8> {
        let mut packed = vec![0u8; experts * out_features * 17];
        for expert in 0..experts {
            for output in 0..out_features {
                let block = &mut packed[(expert * out_features + output) * 17..][..17];
                block[0] = 127;
                for input in 0..H {
                    let value = code(expert, output, input) & 0x0f;
                    let byte = &mut block[1 + input % 16];
                    if input < 16 {
                        *byte |= value;
                    } else {
                        *byte |= value << 4;
                    }
                }
            }
        }
        packed
    }

    fn identity_projection(scales: [u8; E]) -> Vec<u8> {
        packed_matrix(
            E,
            H,
            |expert, output, input| {
                if output == input { scales[expert] } else { 0 }
            },
        )
    }

    fn run(
        activation: &str,
        k: usize,
        normalize: bool,
        swiglu_fusion: usize,
        input: &[f32],
        logits: &[f32],
        fc1: &[u8],
        fc1_out: usize,
        fc2: &[u8],
        router_weights: Option<&[f32]>,
    ) -> Vec<f32> {
        let mut shapes = vec![
            Some((DataType::Float32, vec![1, H])),
            Some((DataType::Float32, vec![1, E])),
            Some((DataType::Uint8, vec![E, fc1_out, 1, 17])),
            None,
            Some((DataType::Uint8, vec![E, H, 1, 17])),
            None,
            None,
            None,
        ];
        if router_weights.is_some() {
            shapes.push(Some((DataType::Float32, vec![1, E])));
        }
        let attrs = attrs(activation, k, normalize, swiglu_fusion);
        let (graph, node) = model_node(&shapes, &attrs);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(
                graph.node(node),
                &shapes
                    .iter()
                    .map(|shape| {
                        shape
                            .as_ref()
                            .map_or_else(Vec::new, |(_, shape)| shape.clone())
                    })
                    .collect::<Vec<_>>(),
                1,
            )
            .expect("valid BlockQuantizedMoE kernel");
        let input = Owned::f32(&[1, H], input);
        let logits = Owned::f32(&[1, E], logits);
        let fc1 = Owned::u8(&[E, fc1_out, 1, 17], fc1);
        let fc2 = Owned::u8(&[E, H, 1, 17], fc2);
        let router = router_weights.map(|weights| Owned::f32(&[1, E], weights));
        let mut views = vec![
            input.view(),
            logits.view(),
            fc1.view(),
            TensorView::absent(DataType::Float32),
            fc2.view(),
            TensorView::absent(DataType::Float32),
            TensorView::absent(DataType::Uint8),
            TensorView::absent(DataType::Float32),
        ];
        if let Some(router) = &router {
            views.push(router.view());
        }
        let mut output = Owned::f32(&[1, H], &[0.0; H]);
        kernel
            .execute(&views, &mut [output.view_mut()])
            .expect("execute BlockQuantizedMoE");
        output.to_f32()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "index {index}: got {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn block_quantized_moe_matches_dense_reference_topk_softmax() {
        let input: Vec<f32> = (0..H).map(|i| i as f32 / 16.0 - 1.0).collect();
        let fc1 = identity_projection([2, 4]);
        let fc2 = identity_projection([2, 2]);
        let actual = run(
            "identity",
            2,
            false,
            0,
            &input,
            &[0.0, 3.0f32.ln()],
            &fc1,
            H,
            &fc2,
            None,
        );
        let expected: Vec<f32> = input.iter().map(|value| value * 1.75).collect();
        assert_close(&actual, &expected);
    }

    #[test]
    fn block_quantized_moe_router_weights_and_normalization_match_reference() {
        let input = vec![1.0f32; H];
        let fc1 = identity_projection([2, 4]);
        let fc2 = identity_projection([2, 2]);
        let unnormalized = run(
            "identity",
            2,
            false,
            0,
            &input,
            &[2.0, 1.0],
            &fc1,
            H,
            &fc2,
            Some(&[2.0, 1.0]),
        );
        let normalized = run(
            "identity",
            2,
            true,
            0,
            &input,
            &[2.0, 1.0],
            &fc1,
            H,
            &fc2,
            Some(&[2.0, 1.0]),
        );
        assert_close(&unnormalized, &[4.0; H]);
        assert_close(&normalized, &[4.0 / 3.0; H]);
    }

    #[test]
    fn block_quantized_moe_silu_matches_dense_reference() {
        let input: Vec<f32> = (0..H).map(|i| i as f32 / 32.0 - 0.5).collect();
        let fc1 = identity_projection([2, 2]);
        let fc2 = identity_projection([2, 2]);
        let actual = run(
            "silu",
            1,
            true,
            0,
            &input,
            &[2.0, -2.0],
            &fc1,
            H,
            &fc2,
            None,
        );
        let expected: Vec<f32> = input
            .iter()
            .map(|&value| value / (1.0 + (-value).exp()))
            .collect();
        assert_close(&actual, &expected);
    }

    #[test]
    fn block_quantized_moe_fused_swiglu_matches_dense_reference() {
        let input: Vec<f32> = (0..H).map(|i| i as f32 / 32.0 + 0.25).collect();
        let fc1 = packed_matrix(E, 2 * H, |_, output, input| {
            if output < H && output == input {
                2
            } else if output >= H && output - H == input {
                4
            } else {
                0
            }
        });
        let fc2 = identity_projection([2, 2]);
        let actual = run(
            "swiglu",
            1,
            true,
            2,
            &input,
            &[2.0, -2.0],
            &fc1,
            2 * H,
            &fc2,
            None,
        );
        let expected: Vec<f32> = input
            .iter()
            .map(|&value| 2.0 * value * value / (1.0 + (-value).exp()))
            .collect();
        assert_close(&actual, &expected);
    }

    #[test]
    fn block_quantized_moe_is_bit_deterministic() {
        let input: Vec<f32> = (0..H).map(|i| i as f32 * 0.03125).collect();
        let fc1 = identity_projection([2, 4]);
        let fc2 = identity_projection([2, 2]);
        let first = run(
            "identity",
            2,
            true,
            0,
            &input,
            &[1.0, 1.0],
            &fc1,
            H,
            &fc2,
            None,
        );
        let second = run(
            "identity",
            2,
            true,
            0,
            &input,
            &[1.0, 1.0],
            &fc1,
            H,
            &fc2,
            None,
        );
        assert_eq!(
            first
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    fn claim_fixture() -> (Graph, NodeId, Vec<Shape>, Vec<DataType>) {
        let shapes = vec![
            Some((DataType::Float32, vec![1, H])),
            Some((DataType::Float32, vec![1, E])),
            Some((DataType::Uint8, vec![E, H, 1, 17])),
            None,
            Some((DataType::Uint8, vec![E, H, 1, 17])),
            None,
            None,
            None,
        ];
        let (graph, node) = model_node(&shapes, &attrs("identity", 1, false, 0));
        let claim_shapes = shapes
            .iter()
            .map(|shape| {
                shape
                    .as_ref()
                    .map_or_else(Vec::new, |(_, shape)| static_shape(shape.iter().copied()))
            })
            .collect();
        let dtypes = shapes
            .iter()
            .map(|shape| {
                shape
                    .as_ref()
                    .map_or(DataType::Undefined, |(dtype, _)| *dtype)
            })
            .collect();
        (graph, node, claim_shapes, dtypes)
    }

    #[test]
    fn block_quantized_moe_claim_gate_accepts_valid_and_omitted_optionals() {
        let (graph, node, shapes, dtypes) = claim_fixture();
        let ep = CpuExecutionProvider::new();
        assert!(
            ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[])
                .is_supported()
        );
    }

    #[test]
    fn block_quantized_moe_claim_gate_rejects_bad_dtype_format_and_arity() {
        let (graph, node, shapes, mut dtypes) = claim_fixture();
        let ep = CpuExecutionProvider::new();
        dtypes[2] = DataType::Float32;
        let rejected = ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]);
        assert!(rejected.reason().unwrap().contains("dtype"));

        let (mut graph, node, shapes, dtypes) = claim_fixture();
        graph.node_mut(node).attributes.insert(
            "format".into(),
            Attribute::String(b"k3_unpublished".to_vec()),
        );
        let rejected = ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]);
        assert!(rejected.reason().unwrap().contains("unsupported format"));

        let (mut graph, node, shapes, dtypes) = claim_fixture();
        graph
            .node_mut(node)
            .attributes
            .insert("use_sparse_mixer".into(), Attribute::Int(0));
        let rejected = ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]);
        assert!(
            rejected
                .reason()
                .unwrap()
                .contains("not part of the frozen v1 ABI")
        );

        let (mut graph, node, mut shapes, mut dtypes) = claim_fixture();
        graph.node_mut(node).inputs.truncate(4);
        shapes.truncate(4);
        dtypes.truncate(4);
        let rejected = ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]);
        assert!(rejected.reason().unwrap().contains("5 to 9"));
    }
}
