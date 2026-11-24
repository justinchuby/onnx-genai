//! Opt-in release benchmark. Run with the shared host lock, `--release`,
//! `--test packed_fusion_perf`, and `-- --ignored --nocapture`.
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef,
    dequantize_packed::Format, static_shape,
};
use onnx_runtime_session::{InferenceSession, Tensor};

fn constant(g: &mut Graph, dtype: DataType, shape: &[usize], bytes: Vec<u8>) -> ValueId {
    let id = g.create_value(dtype, static_shape(shape.iter().copied()));
    g.set_initializer(
        id,
        WeightRef::Inline(TensorData::from_raw(dtype, shape.to_vec(), bytes)),
    );
    id
}

#[derive(Clone, Copy, Debug)]
enum Workload {
    MatMul(usize),
    Gather,
    Grouped,
}

fn fixture(
    workload: Workload,
    format: Format,
    baseline: bool,
) -> (Graph, Vec<(&'static str, Tensor)>, usize) {
    let (width, bytes) = format.block();
    let (n, k) = (2048, 4096);
    let (shape, output, inputs, op) = match workload {
        Workload::MatMul(m) => (
            vec![n, k],
            vec![m, n],
            vec![("A", Tensor::from_f32(&[m, k], &vec![0.125; m * k]).unwrap())],
            "MatMul",
        ),
        Workload::Gather => (
            vec![8192, k],
            vec![4, k],
            vec![("I", Tensor::from_i64(&[4], &[8191, 200, -2, 0]).unwrap())],
            "Gather",
        ),
        Workload::Grouped => (
            vec![8, 512, k],
            vec![4, 2, 512],
            vec![
                ("A", Tensor::from_f32(&[4, k], &vec![0.125; 4 * k]).unwrap()),
                (
                    "I",
                    Tensor::from_i64(&[4, 2], &[0, 7, 1, 6, 2, 5, 3, 4]).unwrap(),
                ),
            ],
            "GroupedMatMul",
        ),
    };
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 29);
    let ids: Vec<_> = inputs
        .iter()
        .map(|(name, tensor)| {
            let id = g.create_named_value(
                *name,
                tensor.dtype,
                static_shape(tensor.shape.iter().copied()),
            );
            g.add_input(id);
            id
        })
        .collect();
    let elements: usize = shape.iter().product();
    let mut packed = vec![0; elements / width * bytes];
    for (offset, lane) in packed.iter_mut().enumerate() {
        *lane = (offset % 61 + 17) as u8;
    }
    let packed_len = packed.len();
    let packed = constant(&mut g, DataType::Uint8, &[packed_len], packed);
    let decoded = g.create_value(DataType::Float32, static_shape(shape.iter().copied()));
    let mut dq = Node::new(
        NodeId(0),
        "DequantizePacked",
        vec![Some(packed)],
        vec![decoded],
    );
    dq.attributes.insert(
        "format".into(),
        Attribute::String(format.name().as_bytes().to_vec()),
    );
    dq.attributes.insert(
        "shape".into(),
        Attribute::Ints(shape.iter().map(|&x| x as i64).collect()),
    );
    g.insert_node(dq);
    // Break the pattern with a same-shape view, not an extra graph output
    // (which would inflate the unfused CUDA readback).
    let weight = if baseline {
        let dims = constant(
            &mut g,
            DataType::Int64,
            &[shape.len()],
            shape
                .iter()
                .flat_map(|&x| (x as i64).to_le_bytes())
                .collect(),
        );
        let view = g.create_value(DataType::Float32, static_shape(shape.iter().copied()));
        g.insert_node(Node::new(
            NodeId(0),
            "Reshape",
            vec![Some(decoded), Some(dims)],
            vec![view],
        ));
        view
    } else {
        decoded
    };
    let node_inputs = match workload {
        Workload::MatMul(_) => {
            let rhs = g.create_value(DataType::Float32, static_shape([k, n]));
            g.insert_node(Node::new(
                NodeId(0),
                "Transpose",
                vec![Some(weight)],
                vec![rhs],
            ));
            vec![Some(ids[0]), Some(rhs)]
        }
        Workload::Gather => vec![Some(weight), Some(ids[0])],
        Workload::Grouped => vec![Some(ids[0]), Some(weight), Some(ids[1])],
    };
    let y = g.create_named_value("Y", DataType::Float32, static_shape(output));
    let mut node = Node::new(NodeId(0), op, node_inputs, vec![y]);
    if matches!(workload, Workload::Grouped) {
        node.attributes.insert("transB".into(), Attribute::Int(1));
    }
    g.insert_node(node);
    g.add_output(y);
    (g, inputs, 4 * elements)
}

fn measure(device: &str, build: impl Fn(Graph) -> InferenceSession) {
    use std::{hint::black_box, time::Instant};
    onnx_runtime_session::enable_activation_memory_plan_for_process();
    for (workload, fused_op) in [
        (Workload::MatMul(1), "BlockQuantizedMatMul"),
        (Workload::MatMul(64), "BlockQuantizedMatMul"),
        (Workload::Gather, "DequantizePackedGather"),
        (Workload::Grouped, "PackedGroupedMatMul"),
    ] {
        for format in [Format::Q8_0, Format::Q4K] {
            let (g, inputs, decoded_bytes) = fixture(workload, format, false);
            let control = device == "cuda" && matches!(workload, Workload::MatMul(64));
            let baseline = if control {
                // CUDA prefill does not fuse. Compare the exact same graph in both arms
                // so a Reshape or its layout effects cannot move the A/A control.
                g.clone()
            } else {
                fixture(workload, format, true).0
            };
            let mut candidate = build(g);
            let mut baseline = build(baseline);
            let selected = candidate
                .graph()
                .nodes
                .values()
                .any(|node| node.op_type == fused_op);
            assert_eq!(selected, !control);
            assert!(
                baseline
                    .graph()
                    .nodes
                    .values()
                    .any(|node| node.op_type == "DequantizePacked")
            );
            let inputs: Vec<_> = inputs
                .iter()
                .map(|(name, tensor)| (*name, tensor))
                .collect();
            let a = candidate.run(&inputs).unwrap()[0].to_vec_f32();
            let b = baseline.run(&inputs).unwrap()[0].to_vec_f32();
            assert!(!a.is_empty());
            assert_eq!(a.len(), b.len());
            assert!(
                a.iter()
                    .zip(&b)
                    .all(|(&x, &y)| (x - y).abs() <= 1e-4 * y.abs().max(1.0))
            );
            let mut tuned =
                if selected && format == Format::Q8_0 && matches!(workload, Workload::MatMul(_)) {
                    let mut graph = candidate.graph().clone();
                    let id = graph
                        .nodes
                        .values()
                        .find(|node| node.op_type == fused_op)
                        .unwrap()
                        .id;
                    graph
                        .node_mut(id)
                        .attributes
                        .insert("format".into(), Attribute::String(b"q8_0".to_vec()));
                    let mut session = build(graph);
                    assert_eq!(a, session.run(&inputs).unwrap()[0].to_vec_f32());
                    Some(session)
                } else {
                    None
                };
            let sample = |session: &mut InferenceSession| {
                let start = Instant::now();
                for _ in 0..3 {
                    black_box(session.run(&inputs).unwrap());
                }
                // Session::run returns host outputs, including CUDA readback.
                start.elapsed().as_secs_f64() * 1e3 / 3.0
            };
            sample(&mut baseline);
            sample(&mut candidate);
            if let Some(tuned) = &mut tuned {
                sample(tuned);
            }
            let (mut original, mut fused, mut tuned_ms) = (Vec::new(), Vec::new(), Vec::new());
            for i in 0..5 {
                if i % 2 == 0 {
                    original.push(sample(&mut baseline));
                    fused.push(sample(&mut candidate));
                } else {
                    fused.push(sample(&mut candidate));
                    original.push(sample(&mut baseline));
                }
                if let Some(tuned) = &mut tuned {
                    tuned_ms.push(sample(tuned));
                }
            }
            original.sort_by(f64::total_cmp);
            fused.sort_by(f64::total_cmp);
            tuned_ms.sort_by(f64::total_cmp);
            println!(
                "{device} {workload:?} {} selected={selected} decoded_bytes={decoded_bytes} \
                peak_plan_bytes={:?}/{:?} unfused_ms={original:?} candidate_ms={fused:?} tuned_ms={tuned_ms:?}",
                format.name(),
                baseline
                    .activation_memory_plan_stats()
                    .map(|s| s.peak_bytes),
                candidate
                    .activation_memory_plan_stats()
                    .map(|s| s.peak_bytes)
            );
        }
    }
}

#[test]
#[ignore = "release performance probe; hold the shared host lock"]
#[allow(clippy::assertions_on_constants)]
fn packed_fusion_cpu_release_ab() {
    // Keep the ignored benchmark debug-compilable while rejecting debug runs.
    // A const assertion would fail every debug all-target check before selection.
    assert!(!cfg!(debug_assertions), "use a release build");
    measure("cpu", |g| InferenceSession::from_graph(g).unwrap());
}

#[cfg(feature = "gpu-tests")]
#[test]
#[ignore = "release performance probe; hold the shared host lock"]
#[allow(clippy::assertions_on_constants)]
fn packed_fusion_cuda_release_ab() {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;
    use std::{path::Path, sync::Arc};
    assert!(!cfg!(debug_assertions), "use a release build");
    let ep = Arc::new(CudaExecutionProvider::initialized(0).unwrap());
    measure("cuda", |g| {
        let session = InferenceSession::from_graph_with_provider(
            g,
            Arc::new(onnx_runtime_loader::WeightStore::new()),
            Path::new("."),
            ep.clone(),
            None,
        )
        .unwrap();
        assert!(session.execution_provider_fallback_report().is_none());
        session
    });
}
