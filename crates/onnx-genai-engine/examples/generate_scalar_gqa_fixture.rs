use std::path::Path;

use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, TensorData};
use onnx_std::Model;

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
    domain: &str,
    op_type: &str,
    inputs: Vec<onnx_runtime_ir::ValueId>,
    outputs: Vec<onnx_runtime_ir::ValueId>,
    attributes: &[(&str, Attribute)],
) {
    let mut node = Node::new(
        NodeId(0),
        op_type,
        inputs.into_iter().map(Some).collect(),
        outputs,
    );
    node.domain = domain.to_string();
    for (name, value) in attributes {
        node.attributes.insert((*name).to_string(), value.clone());
    }
    graph.insert_node(node);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/fixtures/tiny-native-scalar-gqa");
    std::fs::create_dir_all(&root)?;

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 11);
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let batch = graph.intern_symbol("batch");
    let sequence = graph.intern_symbol("sequence");
    let total = graph.intern_symbol("total");
    let past = graph.intern_symbol("past");

    let input_ids = graph.create_named_value(
        "input_ids",
        DataType::Int64,
        vec![batch.into(), sequence.into()],
    );
    let attention_mask = graph.create_named_value(
        "attention_mask",
        DataType::Int64,
        vec![batch.into(), total.into()],
    );
    let position_ids = graph.create_named_value(
        "position_ids",
        DataType::Int64,
        vec![batch.into(), sequence.into()],
    );
    let past_key = graph.create_named_value(
        "past_key_values.0.key",
        DataType::Float32,
        vec![batch.into(), 1.into(), past.into(), 2.into()],
    );
    let past_value = graph.create_named_value(
        "past_key_values.0.value",
        DataType::Float32,
        vec![batch.into(), 1.into(), past.into(), 2.into()],
    );
    for input in [
        input_ids,
        attention_mask,
        position_ids,
        past_key,
        past_value,
    ] {
        graph.add_input(input);
    }

    let query = graph.create_named_value(
        "query",
        DataType::Float32,
        vec![1.into(), 1.into(), 4.into()],
    );
    insert_node(
        &mut graph,
        "",
        "Constant",
        vec![],
        vec![query],
        &[(
            "value",
            Attribute::Tensor(tensor_f32(vec![1, 1, 4], &[0.0; 4])),
        )],
    );
    let key = graph.create_named_value(
        "current_key",
        DataType::Float32,
        vec![1.into(), 1.into(), 2.into()],
    );
    insert_node(
        &mut graph,
        "",
        "Constant",
        vec![],
        vec![key],
        &[(
            "value",
            Attribute::Tensor(tensor_f32(vec![1, 1, 2], &[0.0; 2])),
        )],
    );
    let value = graph.create_named_value(
        "current_value",
        DataType::Float32,
        vec![1.into(), 1.into(), 2.into()],
    );
    insert_node(
        &mut graph,
        "",
        "Constant",
        vec![],
        vec![value],
        &[(
            "value",
            Attribute::Tensor(tensor_f32(vec![1, 1, 2], &[0.0, 1.0])),
        )],
    );

    let total_i64 = graph.create_named_value("total_i64", DataType::Int64, vec![]);
    insert_node(
        &mut graph,
        "",
        "ReduceSum",
        vec![attention_mask],
        vec![total_i64],
        &[
            ("axes", Attribute::Ints(vec![0, 1])),
            ("keepdims", Attribute::Int(0)),
        ],
    );
    let one = graph.create_named_value("one", DataType::Int64, vec![]);
    insert_node(
        &mut graph,
        "",
        "Constant",
        vec![],
        vec![one],
        &[("value", Attribute::Tensor(tensor_i64(vec![], &[1])))],
    );
    let seqlens_i64 = graph.create_named_value("seqlens_i64", DataType::Int64, vec![]);
    insert_node(
        &mut graph,
        "",
        "Sub",
        vec![total_i64, one],
        vec![seqlens_i64],
        &[],
    );
    let seqlens_k = graph.create_named_value("seqlens_k", DataType::Int32, vec![]);
    insert_node(
        &mut graph,
        "",
        "Cast",
        vec![seqlens_i64],
        vec![seqlens_k],
        &[("to", Attribute::Int(DataType::Int32 as i64))],
    );
    let total_sequence_length =
        graph.create_named_value("total_sequence_length", DataType::Int32, vec![]);
    insert_node(
        &mut graph,
        "",
        "Cast",
        vec![total_i64],
        vec![total_sequence_length],
        &[("to", Attribute::Int(DataType::Int32 as i64))],
    );

    let logits = graph.create_named_value(
        "logits",
        DataType::Float32,
        vec![1.into(), 1.into(), 4.into()],
    );
    let present_key = graph.create_named_value(
        "present.0.key",
        DataType::Float32,
        vec![batch.into(), 1.into(), total.into(), 2.into()],
    );
    let present_value = graph.create_named_value(
        "present.0.value",
        DataType::Float32,
        vec![batch.into(), 1.into(), total.into(), 2.into()],
    );
    insert_node(
        &mut graph,
        "com.microsoft",
        "GroupQueryAttention",
        vec![
            query,
            key,
            value,
            past_key,
            past_value,
            seqlens_k,
            total_sequence_length,
        ],
        vec![logits, present_key, present_value],
        &[
            ("num_heads", Attribute::Int(2)),
            ("kv_num_heads", Attribute::Int(1)),
        ],
    );

    for output in [logits, present_key, present_value] {
        graph.add_output(output);
    }
    onnx_std::save_model(&Model::new(graph), root.join("model.onnx"))?;
    Ok(())
}
