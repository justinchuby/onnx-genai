use std::path::Path;

use anyhow::Result;
use onnx_runtime_ir::{
    Attribute, DataType, Dim, Graph, Node, NodeId, Shape, TensorData, ValueId, WeightRef,
};
use onnx_runtime_loader::{Model, write_model};

pub const HIDDEN_SIZE: usize = 64;
pub const QUERY_HEADS: usize = 4;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 16;
pub const INTERMEDIATE_SIZE: usize = 128;
pub const VOCAB_SIZE: usize = 32;
const MAX_POSITIONS: usize = 512;

pub fn build_synthetic_decoder() -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 23);

    let batch = graph.intern_symbol("batch");
    let sequence = graph.intern_symbol("sequence");
    let total = graph.intern_symbol("total_sequence");
    let past = graph.intern_symbol("past_sequence");
    let shape = |dims: &[Dim]| -> Shape { dims.to_vec() };

    let input_ids = named_input(
        &mut graph,
        "input_ids",
        DataType::Int64,
        shape(&[batch.into(), sequence.into()]),
    );
    let attention_mask = named_input(
        &mut graph,
        "attention_mask",
        DataType::Int64,
        shape(&[batch.into(), total.into()]),
    );
    let position_ids = named_input(
        &mut graph,
        "position_ids",
        DataType::Int64,
        shape(&[batch.into(), sequence.into()]),
    );

    let embedding = initializer(
        &mut graph,
        "model.embed_tokens.weight",
        &[VOCAB_SIZE, HIDDEN_SIZE],
        patterned(VOCAB_SIZE * HIDDEN_SIZE, 0.035, 11),
    );
    let mut hidden = value(
        &mut graph,
        DataType::Float32,
        shape(&[batch.into(), sequence.into(), HIDDEN_SIZE.into()]),
    );
    op(
        &mut graph,
        "Gather",
        "",
        vec![Some(embedding), Some(input_ids)],
        vec![hidden],
        &[("axis", Attribute::Int(0))],
    );

    let bool_mask = value(
        &mut graph,
        DataType::Bool,
        shape(&[batch.into(), total.into()]),
    );
    op(
        &mut graph,
        "Cast",
        "",
        vec![Some(attention_mask)],
        vec![bool_mask],
        &[("to", Attribute::Int(9))],
    );

    let (cos_cache, sin_cache) = rotary_cache(&mut graph);
    for layer in 0..2 {
        let past_key = named_input(
            &mut graph,
            &format!("past_key_values.{layer}.key"),
            DataType::Float32,
            shape(&[batch.into(), KV_HEADS.into(), past.into(), HEAD_DIM.into()]),
        );
        let past_value = named_input(
            &mut graph,
            &format!("past_key_values.{layer}.value"),
            DataType::Float32,
            shape(&[batch.into(), KV_HEADS.into(), past.into(), HEAD_DIM.into()]),
        );

        let input_norm = rms_norm(&mut graph, hidden, batch, sequence, layer * 2);
        let q = linear(
            &mut graph,
            input_norm,
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            batch,
            sequence,
            layer * 10 + 1,
        );
        let k = linear(
            &mut graph,
            input_norm,
            HIDDEN_SIZE,
            KV_HEADS * HEAD_DIM,
            batch,
            sequence,
            layer * 10 + 2,
        );
        let v = linear(
            &mut graph,
            input_norm,
            HIDDEN_SIZE,
            KV_HEADS * HEAD_DIM,
            batch,
            sequence,
            layer * 10 + 3,
        );
        let q = rotary(
            &mut graph,
            q,
            cos_cache,
            sin_cache,
            position_ids,
            QUERY_HEADS,
            HIDDEN_SIZE,
            batch,
            sequence,
        );
        let k = rotary(
            &mut graph,
            k,
            cos_cache,
            sin_cache,
            position_ids,
            KV_HEADS,
            KV_HEADS * HEAD_DIM,
            batch,
            sequence,
        );

        let attention = value(
            &mut graph,
            DataType::Float32,
            shape(&[batch.into(), sequence.into(), HIDDEN_SIZE.into()]),
        );
        let present_key = graph.create_named_value(
            format!("present_key_values.{layer}.key"),
            DataType::Float32,
            shape(&[batch.into(), KV_HEADS.into(), total.into(), HEAD_DIM.into()]),
        );
        let present_value = graph.create_named_value(
            format!("present_key_values.{layer}.value"),
            DataType::Float32,
            shape(&[batch.into(), KV_HEADS.into(), total.into(), HEAD_DIM.into()]),
        );
        op(
            &mut graph,
            "Attention",
            "",
            vec![
                Some(q),
                Some(k),
                Some(v),
                Some(bool_mask),
                Some(past_key),
                Some(past_value),
            ],
            vec![attention, present_key, present_value],
            &[
                ("is_causal", Attribute::Int(1)),
                ("q_num_heads", Attribute::Int(QUERY_HEADS as i64)),
                ("kv_num_heads", Attribute::Int(KV_HEADS as i64)),
            ],
        );

        let projected = linear(
            &mut graph,
            attention,
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            batch,
            sequence,
            layer * 10 + 4,
        );
        let attention_residual = binary(
            &mut graph,
            "Add",
            hidden,
            projected,
            &[batch.into(), sequence.into(), HIDDEN_SIZE.into()],
        );
        let post_norm = rms_norm(
            &mut graph,
            attention_residual,
            batch,
            sequence,
            layer * 2 + 1,
        );
        let gate = linear(
            &mut graph,
            post_norm,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            batch,
            sequence,
            layer * 10 + 5,
        );
        let up = linear(
            &mut graph,
            post_norm,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            batch,
            sequence,
            layer * 10 + 6,
        );
        let sigmoid = value(
            &mut graph,
            DataType::Float32,
            shape(&[batch.into(), sequence.into(), INTERMEDIATE_SIZE.into()]),
        );
        op(
            &mut graph,
            "Sigmoid",
            "",
            vec![Some(gate)],
            vec![sigmoid],
            &[],
        );
        let silu = binary(
            &mut graph,
            "Mul",
            gate,
            sigmoid,
            &[batch.into(), sequence.into(), INTERMEDIATE_SIZE.into()],
        );
        let gated = binary(
            &mut graph,
            "Mul",
            silu,
            up,
            &[batch.into(), sequence.into(), INTERMEDIATE_SIZE.into()],
        );
        let down = linear(
            &mut graph,
            gated,
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            batch,
            sequence,
            layer * 10 + 7,
        );
        hidden = binary(
            &mut graph,
            "Add",
            attention_residual,
            down,
            &[batch.into(), sequence.into(), HIDDEN_SIZE.into()],
        );
        graph.add_output(present_key);
        graph.add_output(present_value);
    }

    let final_norm = rms_norm(&mut graph, hidden, batch, sequence, 4);
    let lm_head = initializer(
        &mut graph,
        "lm_head.weight",
        &[HIDDEN_SIZE, VOCAB_SIZE],
        patterned(HIDDEN_SIZE * VOCAB_SIZE, 0.03, 97),
    );
    let logits = graph.create_named_value(
        "logits",
        DataType::Float32,
        shape(&[batch.into(), sequence.into(), VOCAB_SIZE.into()]),
    );
    op(
        &mut graph,
        "MatMul",
        "",
        vec![Some(final_norm), Some(lm_head)],
        vec![logits],
        &[],
    );
    graph.insert_output(0, logits);
    graph
}

pub fn write_synthetic_decoder(path: impl AsRef<Path>) -> Result<()> {
    let graph = build_synthetic_decoder();
    write_model(&Model::new(&graph), path)?;
    Ok(())
}

fn named_input(graph: &mut Graph, name: &str, dtype: DataType, shape: Shape) -> ValueId {
    let value = graph.create_named_value(name, dtype, shape);
    graph.add_input(value);
    value
}

fn value(graph: &mut Graph, dtype: DataType, shape: Shape) -> ValueId {
    let value = graph.create_value(dtype, shape);
    graph.value_mut(value).name = Some(format!("synthetic.v{}", value.0));
    value
}

fn initializer(graph: &mut Graph, name: &str, dims: &[usize], values: Vec<f32>) -> ValueId {
    let value = graph.create_named_value(
        name,
        DataType::Float32,
        dims.iter().copied().map(Dim::from).collect(),
    );
    let bytes = values.into_iter().flat_map(f32::to_le_bytes).collect();
    graph.set_initializer(
        value,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            dims.to_vec(),
            bytes,
        )),
    );
    value
}

fn op(
    graph: &mut Graph,
    op_type: &str,
    domain: &str,
    inputs: Vec<Option<ValueId>>,
    outputs: Vec<ValueId>,
    attributes: &[(&str, Attribute)],
) {
    let mut node = Node::new(NodeId(0), op_type, inputs, outputs);
    node.domain = domain.to_string();
    for (name, attribute) in attributes {
        node.attributes
            .insert((*name).to_string(), attribute.clone());
    }
    graph.insert_node(node);
}

fn rms_norm(
    graph: &mut Graph,
    input: ValueId,
    batch: onnx_runtime_ir::SymbolId,
    sequence: onnx_runtime_ir::SymbolId,
    seed: usize,
) -> ValueId {
    let scale = initializer(
        graph,
        &format!("norm.{seed}.weight"),
        &[HIDDEN_SIZE],
        vec![1.0; HIDDEN_SIZE],
    );
    let output = value(
        graph,
        DataType::Float32,
        vec![batch.into(), sequence.into(), HIDDEN_SIZE.into()],
    );
    op(
        graph,
        "RMSNormalization",
        "",
        vec![Some(input), Some(scale)],
        vec![output],
        &[
            ("axis", Attribute::Int(-1)),
            ("epsilon", Attribute::Float(1e-5)),
        ],
    );
    output
}

fn linear(
    graph: &mut Graph,
    input: ValueId,
    in_features: usize,
    out_features: usize,
    batch: onnx_runtime_ir::SymbolId,
    sequence: onnx_runtime_ir::SymbolId,
    seed: usize,
) -> ValueId {
    let weight = initializer(
        graph,
        &format!("linear.{seed}.weight"),
        &[in_features, out_features],
        patterned(in_features * out_features, 0.025, seed),
    );
    let output = value(
        graph,
        DataType::Float32,
        vec![batch.into(), sequence.into(), out_features.into()],
    );
    op(
        graph,
        "MatMul",
        "",
        vec![Some(input), Some(weight)],
        vec![output],
        &[],
    );
    output
}

fn binary(graph: &mut Graph, op_type: &str, lhs: ValueId, rhs: ValueId, shape: &[Dim]) -> ValueId {
    let output = value(graph, DataType::Float32, shape.to_vec());
    op(
        graph,
        op_type,
        "",
        vec![Some(lhs), Some(rhs)],
        vec![output],
        &[],
    );
    output
}

#[allow(clippy::too_many_arguments)]
fn rotary(
    graph: &mut Graph,
    input: ValueId,
    cos_cache: ValueId,
    sin_cache: ValueId,
    position_ids: ValueId,
    heads: usize,
    width: usize,
    batch: onnx_runtime_ir::SymbolId,
    sequence: onnx_runtime_ir::SymbolId,
) -> ValueId {
    let output = value(
        graph,
        DataType::Float32,
        vec![batch.into(), sequence.into(), width.into()],
    );
    op(
        graph,
        "RotaryEmbedding",
        "",
        vec![
            Some(input),
            Some(cos_cache),
            Some(sin_cache),
            Some(position_ids),
        ],
        vec![output],
        &[
            ("num_heads", Attribute::Int(heads as i64)),
            ("rotary_embedding_dim", Attribute::Int(HEAD_DIM as i64)),
        ],
    );
    output
}

fn rotary_cache(graph: &mut Graph) -> (ValueId, ValueId) {
    let half = HEAD_DIM / 2;
    let mut cos = Vec::with_capacity(MAX_POSITIONS * half);
    let mut sin = Vec::with_capacity(MAX_POSITIONS * half);
    for position in 0..MAX_POSITIONS {
        for index in 0..half {
            let frequency = 1.0 / 10_000.0_f32.powf(index as f32 / half as f32);
            let angle = position as f32 * frequency;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (
        initializer(graph, "rotary.cos_cache", &[MAX_POSITIONS, half], cos),
        initializer(graph, "rotary.sin_cache", &[MAX_POSITIONS, half], sin),
    )
}

fn patterned(len: usize, scale: f32, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let value = ((index * 37 + seed * 17) % 101) as f32 - 50.0;
            value * scale / 50.0
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_decoder_has_standard_cached_decoder_contract() {
        let graph = build_synthetic_decoder();
        graph.validate().expect("synthetic graph must validate");
        let input_names = graph
            .inputs
            .iter()
            .filter_map(|&id| graph.value(id).name.as_deref())
            .collect::<Vec<_>>();
        let output_names = graph
            .outputs
            .iter()
            .filter_map(|&id| graph.value(id).name.as_deref())
            .collect::<Vec<_>>();
        assert!(input_names.contains(&"input_ids"));
        assert!(input_names.contains(&"attention_mask"));
        assert!(input_names.contains(&"position_ids"));
        assert_eq!(
            input_names
                .iter()
                .filter(|name| name.starts_with("past_key_values."))
                .count(),
            4
        );
        assert_eq!(output_names.first(), Some(&"logits"));
        assert_eq!(
            output_names
                .iter()
                .filter(|name| name.starts_with("present_key_values."))
                .count(),
            4
        );
    }
}
