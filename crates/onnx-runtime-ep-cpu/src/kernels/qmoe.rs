//! Correctness-first ORT 1.27 `com.microsoft::QMoE` CPU kernel.
//!
//! Integer expert weights use ORT's expert-major layout
//! `[experts, out_features, in_features / pack_size]`, where
//! `pack_size = 8 / expert_weight_bits`. Values are packed least-significant
//! bits first along the input-feature (K) axis. Scales use
//! `[experts, out_features, in_features / block_size]`; optional zero points
//! pack block values least-significant bits first in
//! `[experts, out_features, ceil(blocks / pack_size)]`.
//!
//! This baseline intentionally dequantizes each selected expert to f32 for each
//! routed row, then calls the float MoE's shared FFN math. Batch-union expert
//! grouping and compressed-domain GEMM are deferred optimization work.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::matmul_nbits::dequantize_nbits_row;
use super::moe::{MoeAttributes, routing_weights, run_expert};
use super::{check_arity, to_dense_bytes, to_dense_f32, write_dense_f32};

/// Factory for the ORT contrib `QMoE` operator.
pub struct QMoEFactory;

/// Per-row block-dequantizing integer QMoE reference kernel.
pub struct QMoEKernel {
    attributes: MoeAttributes,
    bits: usize,
    block_size: usize,
}

impl KernelFactory for QMoEFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let attributes = MoeAttributes::from_node(node)?;
        let bits = int_attr(node, "expert_weight_bits", 4)?;
        if !matches!(bits, 4 | 8) {
            return Err(error(format!(
                "expert_weight_bits must be 4 or 8 in the correctness-first CPU kernel, got {bits}"
            )));
        }
        let block_size = int_attr(node, "block_size", 0)?;
        if block_size < 16 || !(block_size as usize).is_power_of_two() {
            return Err(error(format!(
                "block_size must be a power of two and at least 16, got {block_size}"
            )));
        }
        let quant_type = match node.attr("quant_type") {
            Some(attr) => attr
                .as_str()
                .ok_or_else(|| error("attribute quant_type must be a string"))?,
            None => "int",
        };
        if quant_type != "int" {
            return Err(error(format!(
                "quant_type='{quant_type}' is unsupported; this kernel implements integer affine QMoE only"
            )));
        }
        Ok(Box::new(QMoEKernel {
            attributes,
            bits: bits as usize,
            block_size: block_size as usize,
        }))
    }
}

impl Kernel for QMoEKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("QMoE", inputs, outputs, 7, 21, 1)?;
        for (index, name) in [
            (0, "input"),
            (1, "router_probs"),
            (2, "fc1_experts_weights"),
            (3, "fc1_scales"),
            (5, "fc2_experts_weights"),
            (6, "fc2_scales"),
        ] {
            if inputs[index].is_absent() {
                return Err(error(format!(
                    "required input {index} ('{name}') is absent"
                )));
            }
        }
        require_dtype("input", &inputs[0], DataType::Float32)?;
        require_dtype("router_probs", &inputs[1], DataType::Float32)?;
        require_dtype("fc1_experts_weights", &inputs[2], DataType::Uint8)?;
        require_dtype("fc1_scales", &inputs[3], DataType::Float32)?;
        require_dtype("fc2_experts_weights", &inputs[5], DataType::Uint8)?;
        require_dtype("fc2_scales", &inputs[6], DataType::Float32)?;
        if outputs[0].dtype != DataType::Float32 {
            return Err(error(format!(
                "output requires Float32, got {:?}",
                outputs[0].dtype
            )));
        }
        for (index, name) in [
            (4, "fc1_experts_bias"),
            (7, "fc2_experts_bias"),
            (9, "fc3_scales"),
            (10, "fc3_experts_bias"),
            (14, "router_weights"),
        ] {
            if let Some(input) = optional_input(inputs, index) {
                require_dtype(name, input, DataType::Float32)?;
            }
        }
        for (index, name) in [
            (8, "fc3_experts_weights"),
            (11, "fc1_zero_points"),
            (12, "fc2_zero_points"),
            (13, "fc3_zero_points"),
        ] {
            if let Some(input) = optional_input(inputs, index) {
                require_dtype(name, input, DataType::Uint8)?;
            }
        }
        if let Some((index, _)) = inputs
            .iter()
            .enumerate()
            .skip(15)
            .find(|(_, input)| !input.is_absent())
        {
            return Err(error(format!(
                "input {index} is only used by FP4/FP8 QMoE modes, which are unsupported by this integer kernel"
            )));
        }

        let x_shape = inputs[0].shape;
        if !matches!(x_shape.len(), 2 | 3) {
            return Err(error(format!(
                "input must be 2-D [rows, hidden] or 3-D [batch, sequence, hidden], got {x_shape:?}"
            )));
        }
        if outputs[0].shape != x_shape {
            return Err(error(format!(
                "output shape {:?} must equal input shape {x_shape:?}",
                outputs[0].shape
            )));
        }
        let hidden = *x_shape.last().unwrap();
        let rows = checked_product(&x_shape[..x_shape.len() - 1], "flattened input row count")?;
        for (index, input) in inputs
            .iter()
            .enumerate()
            .filter(|(_, input)| !input.is_absent())
        {
            checked_tensor_layout(&format!("input {index}"), input.shape, input.dtype)?;
        }
        let output_elements = checked_tensor_layout("output", outputs[0].shape, outputs[0].dtype)?;
        require_rank("router_probs", inputs[1].shape, 2)?;
        if inputs[1].shape[0] != rows {
            return Err(error(format!(
                "router_probs rows {} must equal flattened input rows {rows}",
                inputs[1].shape[0]
            )));
        }
        let experts = inputs[1].shape[1];
        if self.attributes.k > experts {
            return Err(error(format!(
                "requires 0 < k <= num_experts, got k={} and num_experts={experts}",
                self.attributes.k
            )));
        }
        if hidden % self.block_size != 0 {
            return Err(error(format!(
                "hidden_size {hidden} must be divisible by block_size {}",
                self.block_size
            )));
        }

        require_rank("fc2_experts_weights", inputs[5].shape, 3)?;
        if inputs[5].shape[0] != experts || inputs[5].shape[1] != hidden {
            return Err(error(format!(
                "fc2_experts_weights must start with [experts={experts}, hidden={hidden}], got {:?}",
                inputs[5].shape
            )));
        }
        let pack_size = 8 / self.bits;
        let inter = inputs[5].shape[2]
            .checked_mul(pack_size)
            .ok_or_else(|| error("fc2 inter_size overflow"))?;
        if inter == 0 || inter % self.block_size != 0 {
            return Err(error(format!(
                "inferred inter_size {inter} must be non-zero and divisible by block_size {}",
                self.block_size
            )));
        }
        let fc1_size = self.attributes.checked_fc1_size(inter, "QMoE")?;

        let fc1 = QuantizedExperts::new(
            "fc1",
            &inputs[2],
            &inputs[3],
            optional_input(inputs, 11),
            experts,
            fc1_size,
            hidden,
            self.bits,
            self.block_size,
        )?;
        let fc2 = QuantizedExperts::new(
            "fc2",
            &inputs[5],
            &inputs[6],
            optional_input(inputs, 12),
            experts,
            hidden,
            inter,
            self.bits,
            self.block_size,
        )?;

        validate_bias("fc1_experts_bias", inputs, 4, experts, fc1_size)?;
        validate_bias("fc2_experts_bias", inputs, 7, experts, hidden)?;
        let fc1_bias = optional_dense(inputs, 4)?;
        let fc2_bias = optional_dense(inputs, 7)?;

        let has_fc3 = optional_input(inputs, 8).is_some();
        let uses_separate_gate = self.attributes.uses_separate_gate(has_fc3);
        let (fc3, fc3_bias) = if uses_separate_gate {
            let weights = optional_input(inputs, 8)
                .ok_or_else(|| error("unfused swiglu requires input 8 fc3_experts_weights"))?;
            let scales = optional_input(inputs, 9)
                .ok_or_else(|| error("fc3_experts_weights requires input 9 fc3_scales"))?;
            validate_bias("fc3_experts_bias", inputs, 10, experts, inter)?;
            (
                Some(QuantizedExperts::new(
                    "fc3",
                    weights,
                    scales,
                    optional_input(inputs, 13),
                    experts,
                    inter,
                    hidden,
                    self.bits,
                    self.block_size,
                )?),
                optional_dense(inputs, 10)?,
            )
        } else {
            for (index, name) in [
                (8, "fc3_experts_weights"),
                (9, "fc3_scales"),
                (10, "fc3_experts_bias"),
                (13, "fc3_zero_points"),
            ] {
                if optional_input(inputs, index).is_some() {
                    return Err(error(format!(
                        "{name} is only valid for unfused swiglu or silu gated-GLU"
                    )));
                }
            }
            (None, None)
        };

        if let Some(router_weights) = optional_input(inputs, 14) {
            require_exact_shape("router_weights", router_weights.shape, &[rows, experts])?;
        }

        let x = to_dense_f32(&inputs[0])?;
        let router = to_dense_f32(&inputs[1])?;
        let aggregation = optional_dense(inputs, 14)?;
        let expected_output_elements = checked_product(&[rows, hidden], "output element count")?;
        if output_elements != expected_output_elements {
            return Err(error(format!(
                "output has {output_elements} elements, expected {expected_output_elements}"
            )));
        }
        let mut output = vec![0.0f32; output_elements];
        for row in 0..rows {
            let router_range = checked_range(row, experts, "router row")?;
            let route = routing_weights(
                &router[router_range.clone()],
                aggregation
                    .as_deref()
                    .map(|weights| &weights[router_range.clone()]),
                self.attributes.k,
                self.attributes.normalize_routing_weights,
            );
            let input_range = checked_range(row, hidden, "input row")?;
            let input_row = &x[input_range];
            let output_range = checked_range(row, hidden, "output row")?;
            let output_row = &mut output[output_range];
            for (expert, route_weight) in route {
                let fc1_weight = fc1.dequantize_expert(expert)?;
                let fc2_weight = fc2.dequantize_expert(expert)?;
                let fc3_weight = fc3
                    .as_ref()
                    .map(|weights| weights.dequantize_expert(expert))
                    .transpose()?;
                let fc1_bias_range = checked_range(expert, fc1_size, "fc1 bias expert row")?;
                let fc2_bias_range = checked_range(expert, hidden, "fc2 bias expert row")?;
                let fc3_bias_range = checked_range(expert, inter, "fc3 bias expert row")?;
                let expert_out = run_expert(
                    input_row,
                    &fc1_weight,
                    fc1_bias.as_deref().map(|bias| &bias[fc1_bias_range]),
                    &fc2_weight,
                    fc2_bias.as_deref().map(|bias| &bias[fc2_bias_range]),
                    fc3_weight.as_deref(),
                    fc3_bias.as_deref().map(|bias| &bias[fc3_bias_range]),
                    fc1_size,
                    hidden,
                    inter,
                    &self.attributes,
                );
                for feature in 0..hidden {
                    output_row[feature] += route_weight * expert_out[feature];
                }
            }
        }
        write_dense_f32(&mut outputs[0], &output)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

struct QuantizedExperts {
    packed: Vec<u8>,
    scales: Vec<f32>,
    zero_points: Option<Vec<u8>>,
    experts: usize,
    out_features: usize,
    in_features: usize,
    packed_in: usize,
    blocks: usize,
    zero_point_bytes: usize,
    dequantized_elements: usize,
    bits: usize,
    block_size: usize,
}

impl QuantizedExperts {
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: &str,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        experts: usize,
        out_features: usize,
        in_features: usize,
        bits: usize,
        block_size: usize,
    ) -> Result<Self> {
        let pack_size = 8 / bits;
        if !in_features.is_multiple_of(pack_size) {
            return Err(error(format!(
                "{name} input features {in_features} must be divisible by pack_size {pack_size}"
            )));
        }
        let packed_in = in_features / pack_size;
        let expert_rows = checked_product(
            &[experts, out_features],
            &format!("{name} expert-row count"),
        )?;
        let packed_elements = checked_product(
            &[expert_rows, packed_in],
            &format!("{name} packed-weight element count"),
        )?;
        checked_byte_count(
            packed_elements,
            std::mem::size_of::<u8>(),
            &format!("{name} packed-weight byte count"),
        )?;
        require_exact_shape(
            &format!("{name}_experts_weights"),
            packed.shape,
            &[experts, out_features, packed_in],
        )?;
        let blocks = in_features / block_size;
        let scale_elements = checked_product(
            &[expert_rows, blocks],
            &format!("{name} scale element count"),
        )?;
        checked_byte_count(
            scale_elements,
            std::mem::size_of::<f32>(),
            &format!("{name} scale byte count"),
        )?;
        require_exact_shape(
            &format!("{name}_scales"),
            scales.shape,
            &[experts, out_features, blocks],
        )?;
        let zero_point_bytes = blocks
            .checked_add(pack_size - 1)
            .ok_or_else(|| error(format!("{name} zero-point row byte count overflow")))?
            / pack_size;
        if let Some(points) = zero_points {
            let affine_block = block_size
                .checked_mul(pack_size)
                .ok_or_else(|| error(format!("{name} affine block width overflow")))?;
            if !in_features.is_multiple_of(affine_block) {
                return Err(error(format!(
                    "{name} input features {in_features} must be divisible by block_size * \
                     pack_size ({affine_block}) when zero points are supplied"
                )));
            }
            let zero_point_elements = checked_product(
                &[expert_rows, zero_point_bytes],
                &format!("{name} zero-point element count"),
            )?;
            checked_byte_count(
                zero_point_elements,
                std::mem::size_of::<u8>(),
                &format!("{name} zero-point byte count"),
            )?;
            require_exact_shape(
                &format!("{name}_zero_points"),
                points.shape,
                &[experts, out_features, zero_point_bytes],
            )?;
        }
        preflight_row_ranges(expert_rows, packed_in, &format!("{name} packed-weight"))?;
        preflight_row_ranges(expert_rows, blocks, &format!("{name} scale"))?;
        preflight_row_ranges(expert_rows, zero_point_bytes, &format!("{name} zero-point"))?;
        let dequantized_elements = checked_product(
            &[out_features, in_features],
            &format!("{name} per-expert dequantized element count"),
        )?;
        checked_byte_count(
            dequantized_elements,
            std::mem::size_of::<f32>(),
            &format!("{name} per-expert dequantized byte count"),
        )?;
        preflight_row_ranges(
            out_features,
            in_features,
            &format!("{name} dequantized output"),
        )?;
        Ok(Self {
            packed: to_dense_bytes(packed)?,
            scales: to_dense_f32(scales)?,
            zero_points: zero_points.map(to_dense_bytes).transpose()?,
            experts,
            out_features,
            in_features,
            packed_in,
            blocks,
            zero_point_bytes,
            dequantized_elements,
            bits,
            block_size,
        })
    }

    fn dequantize_expert(&self, expert: usize) -> Result<Vec<f32>> {
        if expert >= self.experts {
            return Err(error(format!(
                "routed expert {expert} is out of range for {} experts",
                self.experts
            )));
        }
        let mut output = vec![0.0f32; self.dequantized_elements];
        for row in 0..self.out_features {
            let expert_row = expert
                .checked_mul(self.out_features)
                .and_then(|offset| offset.checked_add(row))
                .ok_or_else(|| error("expert-row offset overflow"))?;
            let packed_range =
                checked_range(expert_row, self.packed_in, "packed-weight expert row")?;
            let scale_range = checked_range(expert_row, self.blocks, "scale expert row")?;
            let zero_point_range =
                checked_range(expert_row, self.zero_point_bytes, "zero-point expert row")?;
            let output_range = checked_range(row, self.in_features, "dequantized output row")?;
            dequantize_nbits_row(
                &self.packed[packed_range],
                &self.scales[scale_range],
                self.zero_points
                    .as_ref()
                    .map(|points| &points[zero_point_range]),
                &mut output[output_range],
                self.bits,
                self.block_size,
            );
        }
        Ok(output)
    }
}

fn checked_product(factors: &[usize], context: &str) -> Result<usize> {
    let mut product = 1usize;
    let mut has_zero = false;
    for &factor in factors {
        if factor == 0 {
            has_zero = true;
        } else {
            product = product
                .checked_mul(factor)
                .ok_or_else(|| error(format!("{context} overflow")))?;
        }
    }
    Ok(if has_zero { 0 } else { product })
}

fn checked_byte_count(elements: usize, element_size: usize, context: &str) -> Result<usize> {
    elements
        .checked_mul(element_size)
        .ok_or_else(|| error(format!("{context} overflow")))
}

fn checked_tensor_layout(name: &str, shape: &[usize], dtype: DataType) -> Result<usize> {
    let elements = checked_product(shape, &format!("{name} element count"))?;
    checked_byte_count(elements, dtype.byte_size(), &format!("{name} byte count"))?;
    Ok(elements)
}

fn checked_range(index: usize, width: usize, context: &str) -> Result<std::ops::Range<usize>> {
    let start = index
        .checked_mul(width)
        .ok_or_else(|| error(format!("{context} start offset overflow")))?;
    let end = start
        .checked_add(width)
        .ok_or_else(|| error(format!("{context} end offset overflow")))?;
    Ok(start..end)
}

fn preflight_row_ranges(rows: usize, width: usize, context: &str) -> Result<()> {
    if rows != 0 {
        checked_range(rows - 1, width, context)?;
    }
    Ok(())
}

fn optional_input<'a, 'b>(
    inputs: &'a [TensorView<'b>],
    index: usize,
) -> Option<&'a TensorView<'b>> {
    inputs.get(index).filter(|input| !input.is_absent())
}

fn optional_dense(inputs: &[TensorView], index: usize) -> Result<Option<Vec<f32>>> {
    optional_input(inputs, index).map(to_dense_f32).transpose()
}

fn validate_bias(
    name: &str,
    inputs: &[TensorView],
    index: usize,
    experts: usize,
    width: usize,
) -> Result<()> {
    if let Some(bias) = optional_input(inputs, index) {
        require_exact_shape(name, bias.shape, &[experts, width])?;
    }
    Ok(())
}

fn require_dtype(name: &str, input: &TensorView, dtype: DataType) -> Result<()> {
    if input.dtype != dtype {
        return Err(error(format!(
            "{name} requires {dtype:?}, got {:?}",
            input.dtype
        )));
    }
    Ok(())
}

fn require_rank(name: &str, shape: &[usize], rank: usize) -> Result<()> {
    if shape.len() != rank {
        return Err(error(format!(
            "{name} must be {rank}-D, got shape {shape:?}"
        )));
    }
    Ok(())
}

fn require_exact_shape(name: &str, got: &[usize], expected: &[usize]) -> Result<()> {
    if got != expected {
        return Err(error(format!(
            "{name} must have shape {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

fn int_attr(node: &Node, name: &str, default: i64) -> Result<i64> {
    match node.attr(name) {
        Some(attr) => attr
            .as_int()
            .ok_or_else(|| error(format!("attribute {name} must be an integer"))),
        None => Ok(default),
    }
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("QMoE: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuExecutionProvider;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{Attribute, Graph, NodeId, static_shape};
    use onnx_runtime_loader::Model;

    struct Quantized {
        packed: Vec<u8>,
        scales: Vec<f32>,
        zero_points: Option<Vec<u8>>,
        dequantized: Vec<f32>,
    }

    fn quantize(
        experts: usize,
        out_features: usize,
        in_features: usize,
        bits: usize,
        block_size: usize,
        affine: bool,
    ) -> Quantized {
        let pack_size = 8 / bits;
        let blocks = in_features / block_size;
        let packed_in = in_features / pack_size;
        let zp_bytes = blocks.div_ceil(pack_size);
        let mask = if bits == 8 {
            u8::MAX
        } else {
            (1u8 << bits) - 1
        };
        let default_zp = 1u8 << (bits - 1);
        let mut packed = vec![0u8; experts * out_features * packed_in];
        let mut scales = vec![0.0f32; experts * out_features * blocks];
        let mut zero_points = affine.then(|| vec![0u8; experts * out_features * zp_bytes]);
        let mut dequantized = vec![0.0f32; experts * out_features * in_features];

        for expert in 0..experts {
            for row in 0..out_features {
                for block in 0..blocks {
                    let scale = 0.25 + 0.125 * ((expert + row + block) % 3) as f32;
                    scales[(expert * out_features + row) * blocks + block] = scale;
                    let zero_point = if affine {
                        default_zp.saturating_sub(((expert + row + block) % 2) as u8)
                    } else {
                        default_zp
                    };
                    if let Some(points) = &mut zero_points {
                        let index = (expert * out_features + row) * zp_bytes + block / pack_size;
                        points[index] |= zero_point << ((block % pack_size) * bits);
                    }
                    for offset in 0..block_size {
                        let depth = block * block_size + offset;
                        let centered = ((expert * 3 + row * 5 + depth * 7) % 7) as i16 - 3;
                        let quantized = (centered + zero_point as i16) as u8 & mask;
                        let packed_index =
                            (expert * out_features + row) * packed_in + depth / pack_size;
                        packed[packed_index] |= quantized << ((depth % pack_size) * bits);
                        dequantized[(expert * out_features + row) * in_features + depth] =
                            (quantized as f32 - zero_point as f32) * scale;
                    }
                }
            }
        }
        Quantized {
            packed,
            scales,
            zero_points,
            dequantized,
        }
    }

    fn model_node(
        op: &str,
        inputs: &[Option<(DataType, &[usize])>],
        output_shape: &[usize],
        attrs: &[(&str, Attribute)],
    ) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert("com.microsoft".into(), 1);
        let inputs = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                input.map(|(dtype, shape)| {
                    let value = graph.create_named_value(
                        format!("input_{index}"),
                        dtype,
                        static_shape(shape.iter().copied()),
                    );
                    graph.add_input(value);
                    value
                })
            })
            .collect();
        let output = graph.create_named_value(
            "output",
            DataType::Float32,
            static_shape(output_shape.iter().copied()),
        );
        let mut node = Node::new(NodeId(0), op, inputs, vec![output]);
        node.domain = "com.microsoft".into();
        for (name, value) in attrs {
            node.attributes.insert((*name).into(), value.clone());
        }
        let node = graph.insert_node(node);
        graph.add_output(output);
        (graph, node)
    }

    fn kernel(graph: &Graph, node: NodeId) -> Result<Box<dyn Kernel>> {
        let model = Model::new(graph);
        CpuExecutionProvider::new().get_kernel(model.graph.node(node), &[], 1)
    }

    fn attributes(
        bits: usize,
        block_size: usize,
        k: usize,
        normalize: bool,
    ) -> Vec<(&'static str, Attribute)> {
        vec![
            ("expert_weight_bits", Attribute::Int(bits as i64)),
            ("block_size", Attribute::Int(block_size as i64)),
            ("k", Attribute::Int(k as i64)),
            ("activation_type", Attribute::String(b"identity".to_vec())),
            (
                "normalize_routing_weights",
                Attribute::Int(i64::from(normalize)),
            ),
        ]
    }

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
            assert!(
                (got - want).abs() <= 1e-5,
                "index {index}: got {got}, want {want}"
            );
        }
    }

    fn tiny_owned(dtype: DataType, shape: &[usize]) -> Owned {
        Owned {
            bytes: vec![0; dtype.byte_size().max(1)],
            shape: shape.to_vec(),
            strides: vec![0; shape.len()],
            dtype,
        }
    }

    fn overflow_test_kernel() -> Box<dyn Kernel> {
        let inputs = [
            Some((DataType::Float32, &[1, 16][..])),
            Some((DataType::Float32, &[1, 1])),
            Some((DataType::Uint8, &[1, 16, 8])),
            Some((DataType::Float32, &[1, 16, 1])),
            None,
            Some((DataType::Uint8, &[1, 16, 8])),
            Some((DataType::Float32, &[1, 16, 1])),
        ];
        let attrs = attributes(4, 16, 1, false);
        let (graph, node) = model_node("QMoE", &inputs, &[1, 16], &attrs);
        kernel(&graph, node).unwrap()
    }

    fn assert_kernel_failure_contains(result: Result<()>, expected: &str) {
        match result {
            Err(EpError::KernelFailed(message)) => {
                assert!(
                    message.contains(expected),
                    "expected error containing '{expected}', got '{message}'"
                );
            }
            Err(other) => panic!("expected KernelFailed, got {other}"),
            Ok(()) => panic!("overflowing QMoE input unexpectedly succeeded"),
        }
    }

    fn run_equivalence(
        bits: usize,
        hidden: usize,
        inter: usize,
        block_size: usize,
        k: usize,
        normalize: bool,
        affine: bool,
    ) {
        let experts = 2;
        let rows = 2;
        let fc1 = quantize(experts, inter, hidden, bits, block_size, affine);
        let fc2 = quantize(experts, hidden, inter, bits, block_size, affine);
        let input: Vec<f32> = (0..rows * hidden)
            .map(|index| (index % 5) as f32 * 0.25 - 0.5)
            .collect();
        let router = vec![3.0, 1.0, 0.5, 2.5];
        let pack_size = 8 / bits;
        let hidden_blocks = hidden / block_size;
        let inter_blocks = inter / block_size;

        let float_shapes = [
            Some((DataType::Float32, &[rows, hidden][..])),
            Some((DataType::Float32, &[rows, experts])),
            Some((DataType::Float32, &[experts, inter, hidden])),
            None,
            Some((DataType::Float32, &[experts, hidden, inter])),
        ];
        let float_attrs = [
            ("k", Attribute::Int(k as i64)),
            ("activation_type", Attribute::String(b"identity".to_vec())),
            (
                "normalize_routing_weights",
                Attribute::Int(i64::from(normalize)),
            ),
        ];
        let (float_graph, float_node) =
            model_node("MoE", &float_shapes, &[rows, hidden], &float_attrs);
        let x = Owned::f32(&[rows, hidden], &input);
        let router_tensor = Owned::f32(&[rows, experts], &router);
        let fc1_float = Owned::f32(&[experts, inter, hidden], &fc1.dequantized);
        let fc2_float = Owned::f32(&[experts, hidden, inter], &fc2.dequantized);
        let mut float_output = Owned::zeros_f32(&[rows, hidden]);
        kernel(&float_graph, float_node)
            .unwrap()
            .execute(
                &[
                    x.view(),
                    router_tensor.view(),
                    fc1_float.view(),
                    TensorView::absent(DataType::Float32),
                    fc2_float.view(),
                ],
                &mut [float_output.view_mut()],
            )
            .unwrap();

        let fc1_zero_point_shape = [experts, inter, hidden_blocks.div_ceil(pack_size)];
        let fc2_zero_point_shape = [experts, hidden, inter_blocks.div_ceil(pack_size)];
        let q_shapes = [
            Some((DataType::Float32, &[rows, hidden][..])),
            Some((DataType::Float32, &[rows, experts])),
            Some((DataType::Uint8, &[experts, inter, hidden / pack_size])),
            Some((DataType::Float32, &[experts, inter, hidden_blocks])),
            None,
            Some((DataType::Uint8, &[experts, hidden, inter / pack_size])),
            Some((DataType::Float32, &[experts, hidden, inter_blocks])),
            None,
            None,
            None,
            None,
            affine.then_some((DataType::Uint8, &fc1_zero_point_shape[..])),
            affine.then_some((DataType::Uint8, &fc2_zero_point_shape[..])),
        ];
        let q_attrs = attributes(bits, block_size, k, normalize);
        let (q_graph, q_node) = model_node("QMoE", &q_shapes, &[rows, hidden], &q_attrs);
        let fc1_packed = Owned::u8(&[experts, inter, hidden / pack_size], &fc1.packed);
        let fc1_scales = Owned::f32(&[experts, inter, hidden_blocks], &fc1.scales);
        let fc2_packed = Owned::u8(&[experts, hidden, inter / pack_size], &fc2.packed);
        let fc2_scales = Owned::f32(&[experts, hidden, inter_blocks], &fc2.scales);
        let fc1_zero_points = fc1
            .zero_points
            .as_ref()
            .map(|points| Owned::u8(&[experts, inter, hidden_blocks.div_ceil(pack_size)], points));
        let fc2_zero_points = fc2
            .zero_points
            .as_ref()
            .map(|points| Owned::u8(&[experts, hidden, inter_blocks.div_ceil(pack_size)], points));
        let mut q_output = Owned::zeros_f32(&[rows, hidden]);
        kernel(&q_graph, q_node)
            .unwrap()
            .execute(
                &[
                    x.view(),
                    router_tensor.view(),
                    fc1_packed.view(),
                    fc1_scales.view(),
                    TensorView::absent(DataType::Float32),
                    fc2_packed.view(),
                    fc2_scales.view(),
                    TensorView::absent(DataType::Float32),
                    TensorView::absent(DataType::Uint8),
                    TensorView::absent(DataType::Float32),
                    TensorView::absent(DataType::Float32),
                    fc1_zero_points
                        .as_ref()
                        .map_or_else(|| TensorView::absent(DataType::Uint8), Owned::view),
                    fc2_zero_points
                        .as_ref()
                        .map_or_else(|| TensorView::absent(DataType::Uint8), Owned::view),
                ],
                &mut [q_output.view_mut()],
            )
            .unwrap();
        assert_close(&q_output.to_f32(), &float_output.to_f32());
    }

    #[test]
    fn qmoe_int4_single_block_matches_float_moe() {
        run_equivalence(4, 16, 16, 16, 1, false, false);
    }

    #[test]
    fn qmoe_int8_matches_float_moe() {
        run_equivalence(8, 16, 16, 16, 1, false, false);
    }

    #[test]
    fn qmoe_int4_multiple_blocks_affine_matches_float_moe() {
        run_equivalence(4, 32, 32, 16, 1, false, true);
    }

    #[test]
    fn qmoe_top2_normalized_matches_float_moe() {
        run_equivalence(4, 16, 16, 16, 2, true, false);
    }

    #[test]
    fn qmoe_rejects_unsupported_block_size() {
        let inputs = [
            Some((DataType::Float32, &[1, 16][..])),
            Some((DataType::Float32, &[1, 2])),
            Some((DataType::Uint8, &[2, 16, 8])),
            Some((DataType::Float32, &[2, 16, 2])),
            None,
            Some((DataType::Uint8, &[2, 16, 8])),
            Some((DataType::Float32, &[2, 16, 2])),
        ];
        let attrs = attributes(4, 8, 1, false);
        let (graph, node) = model_node("QMoE", &inputs, &[1, 16], &attrs);
        let failure = match kernel(&graph, node) {
            Ok(_) => panic!("unsupported block_size unexpectedly produced a kernel"),
            Err(error) => error.to_string(),
        };
        assert!(failure.contains("block_size must be a power of two and at least 16"));
    }

    #[test]
    fn qmoe_rejects_flattened_rows_overflow_before_allocation() {
        let x = tiny_owned(DataType::Float32, &[usize::MAX, 2, 0]);
        let router = tiny_owned(DataType::Float32, &[0, 1]);
        let fc1_packed = tiny_owned(DataType::Uint8, &[1, 16, 0]);
        let fc1_scales = tiny_owned(DataType::Float32, &[1, 16, 0]);
        let fc2_packed = tiny_owned(DataType::Uint8, &[1, 0, 8]);
        let fc2_scales = tiny_owned(DataType::Float32, &[1, 0, 1]);
        let mut output = tiny_owned(DataType::Float32, &[usize::MAX, 2, 0]);

        let result = overflow_test_kernel().execute(
            &[
                x.view(),
                router.view(),
                fc1_packed.view(),
                fc1_scales.view(),
                TensorView::absent(DataType::Float32),
                fc2_packed.view(),
                fc2_scales.view(),
            ],
            &mut [output.view_mut()],
        );

        assert_kernel_failure_contains(result, "flattened input row count overflow");
    }

    #[test]
    fn qmoe_rejects_zero_masked_tensor_overflow_before_allocation() {
        let x = tiny_owned(DataType::Float32, &[0, usize::MAX, 2]);
        let router = tiny_owned(DataType::Float32, &[0, 1]);
        let fc1_packed = tiny_owned(DataType::Uint8, &[1, 16, 1]);
        let fc1_scales = tiny_owned(DataType::Float32, &[1, 16, 0]);
        let fc2_packed = tiny_owned(DataType::Uint8, &[1, 2, 8]);
        let fc2_scales = tiny_owned(DataType::Float32, &[1, 2, 1]);
        let mut output = tiny_owned(DataType::Float32, &[0, usize::MAX, 2]);

        let result = overflow_test_kernel().execute(
            &[
                x.view(),
                router.view(),
                fc1_packed.view(),
                fc1_scales.view(),
                TensorView::absent(DataType::Float32),
                fc2_packed.view(),
                fc2_scales.view(),
            ],
            &mut [output.view_mut()],
        );

        assert_kernel_failure_contains(result, "input 0 element count overflow");
    }

    #[test]
    fn qmoe_rejects_quantized_expert_layout_overflow_before_allocation() {
        let experts = usize::MAX / 16 + 1;
        let x = tiny_owned(DataType::Float32, &[1, 16]);
        let router = tiny_owned(DataType::Float32, &[1, experts]);
        let fc1_packed = tiny_owned(DataType::Uint8, &[experts, 16, 8]);
        let fc1_scales = tiny_owned(DataType::Float32, &[experts, 16, 1]);
        let fc2_packed = tiny_owned(DataType::Uint8, &[experts, 16, 8]);
        let fc2_scales = tiny_owned(DataType::Float32, &[experts, 16, 1]);
        let mut output = tiny_owned(DataType::Float32, &[1, 16]);

        let result = overflow_test_kernel().execute(
            &[
                x.view(),
                router.view(),
                fc1_packed.view(),
                fc1_scales.view(),
                TensorView::absent(DataType::Float32),
                fc2_packed.view(),
                fc2_scales.view(),
            ],
            &mut [output.view_mut()],
        );

        assert_kernel_failure_contains(result, "input 2 element count overflow");
    }
}
