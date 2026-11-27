use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, static_shape};
use onnx_runtime_session::{InferenceSession, Tensor};

fn one_hot_graph(opset: u64, dtype: DataType) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let indices = graph.create_named_value("indices", DataType::Int64, static_shape([2, 3]));
    let depth = graph.create_named_value("depth", DataType::Int64, static_shape([]));
    let values = graph.create_named_value("values", dtype, static_shape([2]));
    let output = graph.create_named_value("output", dtype, static_shape([2, 3, 3]));
    for input in [indices, depth, values] {
        graph.add_input(input);
    }
    let mut node = Node::new(
        NodeId(0),
        "OneHot",
        vec![Some(indices), Some(depth), Some(values)],
        vec![output],
    );
    node.attributes.insert("axis".into(), Attribute::Int(-2));
    graph.insert_node(node);
    graph.add_output(output);
    graph
}

fn run_cpu(opset: u64, values: &Tensor) -> Tensor {
    let mut session = InferenceSession::from_graph(one_hot_graph(opset, values.dtype)).unwrap();
    let indices = Tensor::from_i64(&[2, 3], &[-3, -1, 3, -4, 1, 0]).unwrap();
    let depth = Tensor::from_i64(&[], &[3]).unwrap();
    session
        .run(&[("indices", &indices), ("depth", &depth), ("values", values)])
        .unwrap()
        .pop()
        .unwrap()
}

#[test]
fn one_hot_opset28_cpu_bf16_exact_reference() {
    // Signed zero and a NaN payload must be copied unchanged.
    const OFF: u16 = 0x8000;
    const ON: u16 = 0x7fc5;
    let bytes = [OFF, ON].map(u16::to_ne_bytes).concat();
    let values = Tensor::from_raw(DataType::BFloat16, vec![2], &bytes).unwrap();
    let output = run_cpu(28, &values);
    let expected = [
        ON, OFF, OFF, OFF, OFF, OFF, OFF, ON, OFF, // first indices row
        OFF, OFF, ON, OFF, ON, OFF, OFF, OFF, OFF, // second indices row
    ];
    assert_eq!(output.dtype, DataType::BFloat16);
    assert_eq!(output.shape, [2, 3, 3]);
    assert_eq!(output.as_bytes(), expected.map(u16::to_ne_bytes).concat());
}

#[test]
fn one_hot_cpu_declines_bf16_before_opset28() {
    for opset in [9, 11, 27] {
        let error = InferenceSession::from_graph(one_hot_graph(opset, DataType::BFloat16))
            .err()
            .expect("legacy OneHot must decline BF16 at session construction")
            .to_string();
        assert!(error.contains("OneHot"), "{error}");
        assert!(error.contains("'values'"), "{error}");
        assert!(error.contains("BFloat16"), "{error}");
        assert!(error.contains("OneHot-28"), "{error}");
    }
}

#[test]
fn one_hot_cpu_opset9_negative_indices_stay_off() {
    let values = Tensor::from_f32(&[2], &[0.0, 1.0]).unwrap();
    let v9 = run_cpu(9, &values);
    let v11 = run_cpu(11, &values);
    for (output, first_row) in [
        (v9, [0.0_f32; 9]),
        (v11, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    ] {
        let second_row = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let expected = [first_row, second_row].concat();
        let bytes = expected
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        assert_eq!(output.as_bytes(), bytes);
    }
}
