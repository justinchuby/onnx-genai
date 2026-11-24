//! Native CPU execution at the ONNX ReduceMax/ReduceMin BOOL version boundary.

use onnx_runtime_ir::{DataType, Graph, Node, NodeId, TensorData, WeightRef, static_shape};
use onnx_runtime_session::{InferenceSession, Tensor};

fn extrema_graph(op: &str, opset: u64, width: usize, dtype: DataType) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let data = graph.create_named_value("data", dtype, static_shape([2, width]));
    let axes = graph.create_named_value("axes", DataType::Int64, static_shape([1]));
    let output = graph.create_named_value("reduced", dtype, static_shape([2, 1]));
    graph.add_input(data);
    graph.set_initializer(
        axes,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![1],
            1i64.to_le_bytes().to_vec(),
        )),
    );
    graph.insert_node(Node::new(
        NodeId(0),
        op,
        vec![Some(data), Some(axes)],
        vec![output],
    ));
    graph.add_output(output);
    graph
}

#[test]
fn bool_extrema_decline_opset_19() {
    for op in ["ReduceMax", "ReduceMin"] {
        let Err(error) = InferenceSession::from_graph(extrema_graph(op, 19, 3, DataType::Bool))
        else {
            panic!("{op} accepted BOOL at opset 19");
        };
        let message = error.to_string();
        assert!(message.contains(op) && message.contains("19"), "{message}");
        assert!(
            message.contains("BOOL") && message.contains("20"),
            "{message}"
        );
    }
}

#[test]
fn bool_extrema_run_at_opset_20() {
    // Same axes/keepdims as the upstream BOOL and empty-set fixtures.
    for (op, mixed, empty) in [("ReduceMax", [1, 1], [0, 0]), ("ReduceMin", [1, 0], [1, 1])] {
        for (width, data, expected) in [(3, &[1, 1, 1, 0, 1, 0][..], mixed), (0, &[][..], empty)] {
            let mut session =
                InferenceSession::from_graph(extrema_graph(op, 20, width, DataType::Bool))
                    .unwrap_or_else(|error| panic!("{op} BOOL width {width}: {error}"));
            let input = Tensor::from_raw(DataType::Bool, vec![2, width], data).unwrap();
            let output = session.run(&[("data", &input)]).unwrap();
            assert_eq!(output[0].dtype, DataType::Bool);
            assert_eq!(output[0].shape, [2, 1]);
            assert_eq!(output[0].as_bytes(), &expected, "{op} BOOL width {width}");
        }
    }
}

#[test]
fn numeric_extrema_still_run_at_opset_19() {
    for (op, expected) in [("ReduceMax", [3.0, 5.0]), ("ReduceMin", [1.0, -1.0])] {
        let mut session =
            InferenceSession::from_graph(extrema_graph(op, 19, 3, DataType::Float32)).unwrap();
        let input = Tensor::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 0.0, 5.0, -1.0]).unwrap();
        let output = session.run(&[("data", &input)]).unwrap();
        assert_eq!(output[0].to_vec_f32(), expected);
    }
}
