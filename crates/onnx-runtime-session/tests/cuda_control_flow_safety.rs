#![cfg(feature = "cuda")]

use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef, static_shape,
};
use onnx_runtime_loader::{Model, encode_model};
use onnx_runtime_session::{DevicePreference, InferenceSession, Tensor};

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i64_bytes(data: &[i64]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn init(graph: &mut Graph, name: &str, dtype: DataType, dims: &[usize], bytes: Vec<u8>) -> ValueId {
    let value = graph.create_named_value(name, dtype, static_shape(dims.iter().copied()));
    graph.set_initializer(
        value,
        WeightRef::Inline(TensorData::from_raw(dtype, dims.to_vec(), bytes)),
    );
    value
}

fn op(graph: &mut Graph, op_type: &str, inputs: &[ValueId], name: &str, dims: &[usize]) -> ValueId {
    let output =
        graph.create_named_value(name, DataType::Float32, static_shape(dims.iter().copied()));
    graph.insert_node(Node::new(
        NodeId(0),
        op_type,
        inputs.iter().copied().map(Some).collect(),
        vec![output],
    ));
    output
}

fn encode(graph: &Graph) -> Vec<u8> {
    encode_model(&Model::new(graph)).expect("encode test model")
}

fn run_cpu(bytes: &[u8], feeds: &[(&str, &Tensor)]) -> Vec<Tensor> {
    let mut session = InferenceSession::builder()
        .model_bytes(bytes)
        .device(DevicePreference::Cpu)
        .build()
        .expect("build CPU session");
    session.run(feeds).expect("run CPU session")
}

fn run_cuda(bytes: &[u8], feeds: &[(&str, &Tensor)]) -> Vec<Tensor> {
    let mut session = InferenceSession::builder()
        .model_bytes(bytes)
        .device(DevicePreference::Gpu { index: Some(0) })
        .build()
        .expect("build CUDA session");
    session.run(feeds).expect("run CUDA session")
}

fn sequence_at_model() -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let a = init(
        &mut graph,
        "a",
        DataType::Float32,
        &[2],
        f32_bytes(&[1.0, 2.0]),
    );
    let b = init(
        &mut graph,
        "b",
        DataType::Float32,
        &[2],
        f32_bytes(&[3.0, 4.0]),
    );
    let sequence = graph.create_named_value("sequence", DataType::Float32, static_shape([]));
    graph.insert_node(Node::new(
        NodeId(0),
        "SequenceConstruct",
        vec![Some(a), Some(b)],
        vec![sequence],
    ));
    let index = init(&mut graph, "index", DataType::Int64, &[], i64_bytes(&[1]));
    let selected = op(
        &mut graph,
        "SequenceAt",
        &[sequence, index],
        "selected",
        &[2],
    );
    let bias = init(
        &mut graph,
        "bias",
        DataType::Float32,
        &[2],
        f32_bytes(&[10.0, 20.0]),
    );
    let output = op(&mut graph, "Add", &[selected, bias], "output", &[2]);
    graph.add_output(output);
    encode(&graph)
}

fn scan_body(width: usize) -> Graph {
    let mut body = Graph::new();
    body.opset_imports.insert(String::new(), 17);
    let state = body.create_named_value("state", DataType::Float32, static_shape([width]));
    let input = body.create_named_value("input", DataType::Float32, static_shape([width]));
    body.add_input(state);
    body.add_input(input);
    let next = op(&mut body, "Add", &[state, input], "next_state", &[width]);
    let zero = init(
        &mut body,
        "zero",
        DataType::Float32,
        &[width],
        f32_bytes(&vec![0.0; width]),
    );
    let output = op(&mut body, "Add", &[next, zero], "scan_output", &[width]);
    body.add_output(next);
    body.add_output(output);
    body
}

fn scan_model(steps: usize, width: usize) -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let initial = init(
        &mut graph,
        "initial",
        DataType::Float32,
        &[width],
        f32_bytes(&vec![0.0; width]),
    );
    let input = graph.create_named_value("input", DataType::Float32, static_shape([steps, width]));
    graph.add_input(input);
    let final_state =
        graph.create_named_value("final_state", DataType::Float32, static_shape([width]));
    let scan_output = graph.create_named_value(
        "scan_output",
        DataType::Float32,
        static_shape([steps, width]),
    );
    let body = scan_body(width);
    let mut scan = Node::new(
        NodeId(0),
        "Scan",
        vec![Some(initial), Some(input)],
        vec![final_state, scan_output],
    );
    scan.attributes
        .insert("num_scan_inputs".into(), Attribute::Int(1));
    scan.attributes
        .insert("body".into(), Attribute::Graph(Box::new(body.clone())));
    let scan_id = graph.insert_node(scan);
    graph.subgraphs.insert((scan_id, "body".into()), body);
    graph.add_output(final_state);
    graph.add_output(scan_output);
    encode(&graph)
}

#[test]
fn cuda_sequence_at_and_scan_match_cpu_oracles() {
    let sequence_bytes = sequence_at_model();
    let cpu_sequence = run_cpu(&sequence_bytes, &[]);
    let cuda_sequence = run_cuda(&sequence_bytes, &[]);
    assert_eq!(cpu_sequence[0].to_vec_f32(), vec![13.0, 24.0]);
    assert_eq!(cuda_sequence[0].to_vec_f32(), cpu_sequence[0].to_vec_f32());

    let scan_bytes = scan_model(3, 2);
    let input = Tensor::from_f32(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let cpu_scan = run_cpu(&scan_bytes, &[("input", &input)]);
    let cuda_scan = run_cuda(&scan_bytes, &[("input", &input)]);
    assert_eq!(cpu_scan[0].to_vec_f32(), vec![9.0, 12.0]);
    assert_eq!(
        cpu_scan[1].to_vec_f32(),
        vec![1.0, 2.0, 4.0, 6.0, 9.0, 12.0]
    );
    assert_eq!(cuda_scan[0].to_vec_f32(), cpu_scan[0].to_vec_f32());
    assert_eq!(cuda_scan[1].to_vec_f32(), cpu_scan[1].to_vec_f32());
}
