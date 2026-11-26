use onnx_runtime_ir::{Attribute, DataType, DeviceId, Graph, Node, NodeId, static_shape};
use onnx_runtime_loader::{Model, encode_model};
use onnx_runtime_session::{DevicePreference, InferenceSession, Tensor};

// ONNX 622adf94d4e720142950aa8178f7357d52e63db0: Shape and Size
// revisions 19, 21, 23 and 24 first admit IR9, IR10, IR11 and IR12 types.
const CASES: &[(i64, u64, DataType)] = &[
    (9, 19, DataType::Float8E4M3FN),
    (9, 19, DataType::Float8E4M3FNUZ),
    (9, 19, DataType::Float8E5M2),
    (9, 19, DataType::Float8E5M2FNUZ),
    (10, 21, DataType::Int4),
    (10, 21, DataType::Uint4),
    (11, 23, DataType::Float4E2M1),
    (12, 24, DataType::Float8E8M0),
];

#[derive(Clone, Copy, Debug)]
enum MetadataOp {
    Shape,
    Size,
}

impl MetadataOp {
    fn name(self) -> &'static str {
        match self {
            Self::Shape => "Shape",
            Self::Size => "Size",
        }
    }

    fn graph(self, ir: i64, opset: u64, dtype: DataType) -> Graph {
        use onnx_runtime_ir::Dim::{Static, Symbolic};

        let mut graph = Graph::new();
        graph.ir_version = Some(ir);
        graph.opset_imports.insert(String::new(), opset);
        let extent = graph.intern_symbol("extent");
        let x = graph.create_named_value(
            "x",
            dtype,
            vec![Static(1), Static(3), Symbolic(extent), Static(5)],
        );
        graph.add_input(x);
        let shape = match self {
            Self::Shape => static_shape([2]),
            Self::Size => vec![],
        };
        let y = graph.create_named_value("y", DataType::Int64, shape);
        let mut node = Node::new(NodeId(0), self.name(), vec![Some(x)], vec![y]);
        if matches!(self, Self::Shape) {
            node.attributes.insert("start".into(), Attribute::Int(-3));
            node.attributes.insert("end".into(), Attribute::Int(-1));
        }
        graph.insert_node(node);
        graph.add_output(y);
        graph
    }

    fn model(self, ir: i64, opset: u64, dtype: DataType) -> Vec<u8> {
        encode_model(&Model::new(&self.graph(ir, opset, dtype))).expect("encode metadata model")
    }
}

fn check_cases(op: MetadataOp, preference: DevicePreference, device: DeviceId) {
    for &(ir, opset, dtype) in CASES {
        let case = format!("{op:?}-{opset} {dtype:?} on {device:?}");
        let mut session = InferenceSession::builder()
            .model_bytes(&op.model(ir, opset, dtype))
            .device(preference.clone())
            .option("optimization", "basic")
            .build()
            .unwrap_or_else(|err| panic!("{case}: build session: {err}"));
        assert_eq!(session.device_id(), device, "{case}");
        assert!(
            session.execution_provider_fallback_report().is_none()
                && session.heterogeneous_placement_report().is_none(),
            "{case}: expected single-provider placement without fallback"
        );
        assert_eq!(session.graph().num_nodes(), 1, "{case}: was the op folded?");
        assert_eq!(
            session.graph().nodes.values().next().unwrap().op_type,
            op.name(),
            "{case}: expected the original op after optimization"
        );

        // Both logical counts (45 and 75) are odd, so packed input storage
        // contains a padding nibble that is not an element.
        for (extent, size) in [(3, 45), (5, 75)] {
            let input = Tensor::from_raw(
                dtype,
                vec![1, 3, extent, 5],
                &vec![0x11; dtype.storage_bytes(size)],
            )
            .unwrap_or_else(|err| panic!("{case}: build input for extent {extent}: {err}"));
            let before = session.cache_stats().prebind_hits;
            let outputs = session
                .run(&[("x", &input)])
                .unwrap_or_else(|err| panic!("{case}: run with extent {extent}: {err}"));
            assert!(
                session.cache_stats().prebind_hits > before,
                "{case}: run did not dispatch the metadata kernel for extent {extent}"
            );
            assert_eq!(outputs.len(), 1, "{case}");
            let output = &outputs[0];
            assert_eq!(output.dtype, DataType::Int64, "{case}");
            match op {
                MetadataOp::Shape => {
                    assert_eq!(output.shape, [2], "{case}");
                    assert_eq!(output.to_vec_i64(), [3, extent as i64], "{case}");
                }
                MetadataOp::Size => {
                    assert!(output.shape.is_empty(), "{case}: expected an INT64 scalar");
                    assert_eq!(output.to_vec_i64(), [size as i64], "{case}");
                }
            }
        }
    }
}

fn check_cpu(op: MetadataOp) {
    check_cases(op, DevicePreference::Cpu, DeviceId::cpu());
}

#[test]
fn modern_shape_dynamic_cpu() {
    check_cpu(MetadataOp::Shape);
}

#[test]
fn modern_size_dynamic_cpu() {
    check_cpu(MetadataOp::Size);
}

#[test]
fn modern_shape_size_reject_pre_revision_dtypes() {
    for op in [MetadataOp::Shape, MetadataOp::Size] {
        for &(ir, opset, dtype) in CASES {
            // Keep the tensor's IR version valid so only the operator's
            // imported schema revision is under test.
            let older = opset - 1;
            let result = InferenceSession::builder()
                .model_bytes(&op.model(ir, older, dtype))
                .device(DevicePreference::Cpu)
                .build();
            let err = match result {
                Ok(_) => panic!("{}-{older} unexpectedly admitted {dtype:?}", op.name()),
                Err(err) => err.to_string(),
            };
            for expected in [
                op.name().to_owned(),
                format!("{dtype:?}"),
                format!("not admitted at opset {older}"),
                format!("export with opset {opset} or newer"),
            ] {
                assert!(err.contains(&expected), "expected {expected:?} in {err}");
            }
        }
    }
}

#[test]
fn modern_shape_size_reject_pre_revision_dtype_without_rank() {
    for op in [MetadataOp::Shape, MetadataOp::Size] {
        let mut graph = op.graph(12, 23, DataType::Float8E8M0);
        graph.mark_value_shape_unknown(graph.inputs[0]);
        let err = match InferenceSession::from_graph(graph) {
            Ok(_) => panic!("{}-23 unexpectedly admitted Float8E8M0", op.name()),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("not admitted at opset 23"), "{err}");
        assert!(err.contains("export with opset 24 or newer"), "{err}");
    }
}

#[cfg(feature = "cuda")]
fn check_cuda(op: MetadataOp) {
    check_cases(
        op,
        DevicePreference::Gpu { index: Some(0) },
        DeviceId::cuda(0),
    );
}

#[cfg(feature = "cuda")]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA device")]
#[test]
fn modern_shape_dynamic_cuda() {
    check_cuda(MetadataOp::Shape);
}

#[cfg(feature = "cuda")]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA device")]
#[test]
fn modern_size_dynamic_cuda() {
    check_cuda(MetadataOp::Size);
}
