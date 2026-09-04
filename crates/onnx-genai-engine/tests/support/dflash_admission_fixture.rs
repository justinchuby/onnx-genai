use std::path::Path;

use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef};
use onnx_std::Model;

pub const PROPOSER_FILE: &str = "proposer.onnx.textproto";
pub const TARGET_FILE: &str = "target.onnx.textproto";

#[derive(Debug, PartialEq, Eq)]
pub struct Documents {
    pub proposer: String,
    pub target: String,
}

fn tensor_f32(shape: Vec<usize>, values: &[f32]) -> TensorData {
    TensorData::from_raw(
        DataType::Float32,
        shape,
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
}

fn tensor_i64(shape: Vec<usize>, values: &[i64]) -> TensorData {
    TensorData::from_raw(
        DataType::Int64,
        shape,
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
}

fn insert_node(
    graph: &mut Graph,
    op_type: &str,
    inputs: Vec<ValueId>,
    outputs: Vec<ValueId>,
    attributes: &[(&str, Attribute)],
) {
    let mut node = Node::new(
        NodeId(0),
        op_type,
        inputs.into_iter().map(Some).collect(),
        outputs,
    );
    for (name, value) in attributes {
        node.attributes.insert((*name).to_string(), value.clone());
    }
    graph.insert_node(node);
}

fn initializer(graph: &mut Graph, name: &str, tensor: TensorData) -> ValueId {
    let value = graph.create_named_value(
        name,
        tensor.dtype,
        tensor.dims.iter().copied().map(Into::into).collect(),
    );
    graph.set_initializer(value, WeightRef::Inline(tensor));
    value
}

fn target_model() -> Model {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 24);
    let batch = graph.intern_symbol("batch");
    let verify = graph.intern_symbol("verify");
    let target_sequence = graph.intern_symbol("target_sequence");
    let token_sequence = graph.intern_symbol("token_sequence");
    let updated_target_sequence = graph.intern_symbol("updated_target_sequence");

    let tokens =
        graph.create_named_value("tokens", DataType::Int64, vec![batch.into(), verify.into()]);
    let past_target = graph.create_named_value(
        "past_target",
        DataType::Float32,
        vec![batch.into(), target_sequence.into(), 3.into()],
    );
    let token_history = graph.create_named_value(
        "token_history",
        DataType::Int64,
        vec![batch.into(), token_sequence.into()],
    );
    let recurrent =
        graph.create_named_value("recurrent", DataType::Float32, vec![batch.into(), 2.into()]);
    for input in [tokens, past_target, token_history, recurrent] {
        graph.add_input(input);
    }

    let token_embedding = initializer(
        &mut graph,
        "token_embedding",
        tensor_f32(
            vec![11, 3],
            &[
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                1.0, 1.0, -1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 1.0, 0.5, 0.5,
                0.5,
            ],
        ),
    );
    let lm_head = initializer(
        &mut graph,
        "lm_head",
        tensor_f32(
            vec![3, 11],
            &[
                1.0, 0.0, 0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
                1.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -1.0, -1.0,
                0.5,
            ],
        ),
    );

    let hidden_2 = graph.create_named_value(
        "hidden_2",
        DataType::Float32,
        vec![batch.into(), verify.into(), 3.into()],
    );
    insert_node(
        &mut graph,
        "Gather",
        vec![token_embedding, tokens],
        vec![hidden_2],
        &[("axis", Attribute::Int(0))],
    );
    let hidden_7 = graph.create_named_value(
        "hidden_7",
        DataType::Float32,
        vec![batch.into(), verify.into(), 3.into()],
    );
    let hidden_12 = graph.create_named_value(
        "hidden_12",
        DataType::Float32,
        vec![batch.into(), verify.into(), 3.into()],
    );
    insert_node(&mut graph, "Identity", vec![hidden_2], vec![hidden_7], &[]);
    insert_node(&mut graph, "Identity", vec![hidden_7], vec![hidden_12], &[]);

    let logits = graph.create_named_value(
        "logits",
        DataType::Float32,
        vec![batch.into(), verify.into(), 11.into()],
    );
    insert_node(
        &mut graph,
        "MatMul",
        vec![hidden_12, lm_head],
        vec![logits],
        &[],
    );

    let present_target = graph.create_named_value(
        "present_target",
        DataType::Float32,
        vec![batch.into(), updated_target_sequence.into(), 3.into()],
    );
    insert_node(
        &mut graph,
        "Concat",
        vec![past_target, hidden_12],
        vec![present_target],
        &[("axis", Attribute::Int(1))],
    );
    let present_tokens = graph.create_named_value(
        "present_tokens",
        DataType::Int64,
        vec![batch.into(), updated_target_sequence.into()],
    );
    insert_node(
        &mut graph,
        "Concat",
        vec![token_history, tokens],
        vec![present_tokens],
        &[("axis", Attribute::Int(1))],
    );
    let next_recurrent = graph.create_named_value(
        "next_recurrent",
        DataType::Float32,
        vec![batch.into(), 2.into()],
    );
    insert_node(
        &mut graph,
        "Identity",
        vec![recurrent],
        vec![next_recurrent],
        &[],
    );

    let token_shape = graph.create_named_value("token_shape", DataType::Int64, vec![2.into()]);
    insert_node(&mut graph, "Shape", vec![tokens], vec![token_shape], &[]);
    let recurrent_width = initializer(&mut graph, "recurrent_width", tensor_i64(vec![1], &[2]));
    let prefix_shape = graph.create_named_value("prefix_shape", DataType::Int64, vec![3.into()]);
    insert_node(
        &mut graph,
        "Concat",
        vec![token_shape, recurrent_width],
        vec![prefix_shape],
        &[("axis", Attribute::Int(0))],
    );
    let prefix_axis = initializer(&mut graph, "prefix_axis", tensor_i64(vec![1], &[1]));
    let recurrent_row = graph.create_named_value(
        "recurrent_row",
        DataType::Float32,
        vec![batch.into(), 1.into(), 2.into()],
    );
    insert_node(
        &mut graph,
        "Unsqueeze",
        vec![recurrent, prefix_axis],
        vec![recurrent_row],
        &[],
    );
    let recurrent_prefixes = graph.create_named_value(
        "recurrent_prefixes",
        DataType::Float32,
        vec![batch.into(), verify.into(), 2.into()],
    );
    insert_node(
        &mut graph,
        "Expand",
        vec![recurrent_row, prefix_shape],
        vec![recurrent_prefixes],
        &[],
    );

    for output in [
        hidden_2,
        hidden_7,
        hidden_12,
        logits,
        present_target,
        present_tokens,
        next_recurrent,
        recurrent_prefixes,
    ] {
        graph.add_output(output);
    }
    Model::new(graph)
}

fn proposer_model() -> Model {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 24);
    let batch = graph.intern_symbol("batch");
    let verify = graph.intern_symbol("verify");
    let total = graph.intern_symbol("total");
    let draft_sequence = graph.intern_symbol("draft_sequence");
    let updated_draft_sequence = graph.intern_symbol("updated_draft_sequence");

    let target_features = graph.create_named_value(
        "target_features",
        DataType::Float32,
        vec![batch.into(), verify.into(), 9.into()],
    );
    let noise_embeddings = graph.create_named_value(
        "noise_embeddings",
        DataType::Float32,
        vec![batch.into(), 4.into(), 3.into()],
    );
    let masked_positions = graph.create_named_value(
        "masked_positions",
        DataType::Bool,
        vec![batch.into(), 4.into()],
    );
    let position_ids = graph.create_named_value(
        "position_ids",
        DataType::Int64,
        vec![batch.into(), total.into()],
    );
    let attention_mask = graph.create_named_value(
        "attention_mask",
        DataType::Int64,
        vec![batch.into(), total.into()],
    );
    let output_projection = graph.create_named_value(
        "output_projection",
        DataType::Float32,
        vec![3.into(), 11.into()],
    );
    let past_draft = graph.create_named_value(
        "past_draft",
        DataType::Float32,
        vec![batch.into(), draft_sequence.into(), 3.into()],
    );
    for input in [
        target_features,
        noise_embeddings,
        masked_positions,
        position_ids,
        attention_mask,
        output_projection,
        past_draft,
    ] {
        graph.add_input(input);
    }

    let split_sizes = initializer(
        &mut graph,
        "conditioning_split_sizes",
        tensor_i64(vec![3], &[3, 3, 3]),
    );
    let hidden_2 = graph.create_named_value(
        "conditioning_hidden_2",
        DataType::Float32,
        vec![batch.into(), verify.into(), 3.into()],
    );
    let hidden_7 = graph.create_named_value(
        "conditioning_hidden_7",
        DataType::Float32,
        vec![batch.into(), verify.into(), 3.into()],
    );
    let hidden_12 = graph.create_named_value(
        "conditioning_hidden_12",
        DataType::Float32,
        vec![batch.into(), verify.into(), 3.into()],
    );
    insert_node(
        &mut graph,
        "Split",
        vec![target_features, split_sizes],
        vec![hidden_2, hidden_7, hidden_12],
        &[("axis", Attribute::Int(2))],
    );
    let conditioning_7 = graph.create_named_value(
        "conditioning_7",
        DataType::Float32,
        vec![batch.into(), verify.into(), 3.into()],
    );
    insert_node(
        &mut graph,
        "Add",
        vec![hidden_2, hidden_7],
        vec![conditioning_7],
        &[],
    );
    let conditioning = graph.create_named_value(
        "conditioning",
        DataType::Float32,
        vec![batch.into(), verify.into(), 3.into()],
    );
    insert_node(
        &mut graph,
        "Add",
        vec![conditioning_7, hidden_12],
        vec![conditioning],
        &[],
    );
    let projected = graph.create_named_value(
        "projected_conditioning",
        DataType::Float32,
        vec![batch.into(), verify.into(), 11.into()],
    );
    insert_node(
        &mut graph,
        "MatMul",
        vec![conditioning, output_projection],
        vec![projected],
        &[],
    );

    let masked_shape = graph.create_named_value("masked_shape", DataType::Int64, vec![2.into()]);
    insert_node(
        &mut graph,
        "Shape",
        vec![masked_positions],
        vec![masked_shape],
        &[],
    );
    let zero_index = initializer(&mut graph, "zero_index", tensor_i64(vec![], &[0]));
    let batch_size = graph.create_named_value("batch_size", DataType::Int64, vec![]);
    insert_node(
        &mut graph,
        "Gather",
        vec![masked_shape, zero_index],
        vec![batch_size],
        &[("axis", Attribute::Int(0))],
    );
    let axis_zero = initializer(&mut graph, "axis_zero", tensor_i64(vec![1], &[0]));
    let batch_vector = graph.create_named_value("batch_vector", DataType::Int64, vec![1.into()]);
    insert_node(
        &mut graph,
        "Unsqueeze",
        vec![batch_size, axis_zero],
        vec![batch_vector],
        &[],
    );

    let proposal_width = initializer(&mut graph, "proposal_width", tensor_i64(vec![1], &[3]));
    let candidate_shape =
        graph.create_named_value("candidate_shape", DataType::Int64, vec![2.into()]);
    insert_node(
        &mut graph,
        "Concat",
        vec![batch_vector, proposal_width],
        vec![candidate_shape],
        &[("axis", Attribute::Int(0))],
    );
    let candidate_seed = initializer(
        &mut graph,
        "candidate_seed",
        tensor_i64(vec![1, 3], &[1, 2, 3]),
    );
    let candidate_tokens = graph.create_named_value(
        "candidate_tokens",
        DataType::Int64,
        vec![batch.into(), 3.into()],
    );
    insert_node(
        &mut graph,
        "Expand",
        vec![candidate_seed, candidate_shape],
        vec![candidate_tokens],
        &[],
    );

    let probability_tail = initializer(
        &mut graph,
        "probability_tail",
        tensor_i64(vec![2], &[3, 11]),
    );
    let probability_shape =
        graph.create_named_value("probability_shape", DataType::Int64, vec![3.into()]);
    insert_node(
        &mut graph,
        "Concat",
        vec![batch_vector, probability_tail],
        vec![probability_shape],
        &[("axis", Attribute::Int(0))],
    );
    let mut probability_values = vec![0.0_f32; 3 * 11];
    probability_values[1] = 1.0;
    probability_values[11 + 2] = 1.0;
    probability_values[22 + 3] = 1.0;
    let probability_seed = initializer(
        &mut graph,
        "probability_seed",
        tensor_f32(vec![1, 3, 11], &probability_values),
    );
    let proposal_probabilities = graph.create_named_value(
        "proposal_probabilities",
        DataType::Float32,
        vec![batch.into(), 3.into(), 11.into()],
    );
    insert_node(
        &mut graph,
        "Expand",
        vec![probability_seed, probability_shape],
        vec![proposal_probabilities],
        &[],
    );

    let present_draft = graph.create_named_value(
        "present_draft",
        DataType::Float32,
        vec![batch.into(), updated_draft_sequence.into(), 3.into()],
    );
    insert_node(
        &mut graph,
        "Concat",
        vec![past_draft, hidden_2],
        vec![present_draft],
        &[("axis", Attribute::Int(1))],
    );

    for output in [candidate_tokens, proposal_probabilities, present_draft] {
        graph.add_output(output);
    }
    Model::new(graph)
}

pub fn documents() -> onnx_std::Result<Documents> {
    Ok(Documents {
        proposer: onnx_std::textproto::to_textproto(&proposer_model())?,
        target: onnx_std::textproto::to_textproto(&target_model())?,
    })
}

#[allow(dead_code)]
pub fn write(root: &Path) -> anyhow::Result<()> {
    let documents = documents()?;
    std::fs::create_dir_all(root)?;
    std::fs::write(root.join(PROPOSER_FILE), documents.proposer)?;
    std::fs::write(root.join(TARGET_FILE), documents.target)?;
    Ok(())
}
