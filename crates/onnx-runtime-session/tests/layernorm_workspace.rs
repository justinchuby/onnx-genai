use std::sync::Arc;

use onnx_runtime_ir::{DataType, Graph, Node, NodeId, static_shape};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use onnx_runtime_session::{InferenceSession, Tensor};

fn model() -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let x = graph.create_named_value("X", DataType::Float32, static_shape([2, 4]));
    let residual = graph.create_named_value("skip", DataType::Float32, static_shape([2, 4]));
    let scale = graph.create_named_value("scale", DataType::Float32, static_shape([4]));
    let beta = graph.create_named_value("beta", DataType::Float32, static_shape([4]));
    let bias = graph.create_named_value("bias", DataType::Float32, static_shape([4]));
    for input in [x, residual, scale, beta, bias] {
        graph.add_input(input);
    }
    let first = graph.create_named_value("first", DataType::Float32, static_shape([2, 4]));
    let first_mean =
        graph.create_named_value("first_mean", DataType::Float32, static_shape([2, 1]));
    let first_inv = graph.create_named_value("first_inv", DataType::Float32, static_shape([2, 1]));
    let y = graph.create_named_value("Y", DataType::Float32, static_shape([2, 4]));
    let mean = graph.create_named_value("mean", DataType::Float32, static_shape([2, 1]));
    let inv = graph.create_named_value("inv", DataType::Float32, static_shape([2, 1]));
    let sum = graph.create_named_value("sum", DataType::Float32, static_shape([2, 4]));
    graph.insert_node(Node::new(
        NodeId(0),
        "LayerNormalization",
        vec![Some(x), Some(scale), Some(beta)],
        vec![first, first_mean, first_inv],
    ));
    let mut second = Node::new(
        NodeId(1),
        "SkipLayerNormalization",
        vec![
            Some(first),
            Some(residual),
            Some(scale),
            Some(beta),
            Some(bias),
        ],
        vec![y, mean, inv, sum],
    );
    second.domain = "com.microsoft".into();
    second
        .attributes
        .insert("epsilon".into(), onnx_runtime_ir::Attribute::Float(1e-5));
    graph.insert_node(second);
    for out in [first_mean, first_inv, y, mean, inv, sum] {
        graph.add_output(out);
    }
    onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&graph)).unwrap()
}

fn reference(x: &[f64], scale: &[f32], beta: &[f32]) -> [Vec<f64>; 3] {
    let mut y = Vec::new();
    let mut means = Vec::new();
    let mut inverses = Vec::new();
    for row in x.chunks_exact(scale.len()) {
        let m = row.iter().sum::<f64>() / scale.len() as f64;
        let v = row.iter().map(|x| (x - m).powi(2)).sum::<f64>() / scale.len() as f64;
        let inv = 1. / (v + 1e-5).sqrt();
        means.push(m);
        inverses.push(inv);
        y.extend(
            row.iter()
                .zip(scale)
                .zip(beta)
                .map(|((&x, &s), &b)| (x - m) * inv * f64::from(s) + f64::from(b)),
        );
    }
    [y, means, inverses]
}

#[test]
fn layernorm_and_skip_share_one_governed_peak_and_readmit_after_reclamation() {
    let bytes = model();
    let values = [1., 2., 3., 4., 2., 3., 4., 5.];
    let skips = [0.5, -1., 1., 0., 1., 0., -1., 2.];
    let scales = [1., 2., 0.5, 1.5];
    let betas = [0., 1., -1., 0.5];
    let biases = [0.25, 0., -0.5, 1.];
    let x = Tensor::from_f32(&[2, 4], &values).unwrap();
    let skip = Tensor::from_f32(&[2, 4], &skips).unwrap();
    let scale = Tensor::from_f32(&[4], &scales).unwrap();
    let beta = Tensor::from_f32(&[4], &betas).unwrap();
    let bias = Tensor::from_f32(&[4], &biases).unwrap();
    let inputs = [
        ("X", &x),
        ("skip", &skip),
        ("scale", &scale),
        ("beta", &beta),
        ("bias", &bias),
    ];
    let [first, first_mean, first_inv] = reference(&values.map(f64::from), &scales, &betas);
    let sum = first
        .iter()
        .zip(skips)
        .enumerate()
        .map(|(i, (&x, s))| x + f64::from(s) + f64::from(biases[i % 4]))
        .collect::<Vec<_>>();
    let [expected, mean, inv] = reference(&sum, &scales, &betas);
    let expected_outputs = [first_mean, first_inv, expected, mean, inv, sum];
    for budget in [79u64, 80] {
        let books = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
            DeviceKey::HOST,
            0,
            budget,
            0,
        )));
        let manager = ProcessMemoryManager::new().unwrap();
        let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
        let resources = onnx_runtime_ep_cpu::CpuMemoryResources::new(
            onnx_runtime_ep_cpu::DerivedWeightMemory::governed(books.as_ref(), HolderId::new(13))
                .unwrap(),
            Default::default(),
        )
        .unwrap()
        .with_workspace_pool(pool.clone())
        .unwrap();
        let cpu = onnx_runtime_ep_cpu::CpuExecutionProvider::initialized_with_resources(resources)
            .unwrap();
        let mut session = InferenceSession::builder()
            .model_bytes(&bytes)
            .execution_provider(Arc::new(cpu))
            .build()
            .unwrap();
        if budget < 80 {
            assert!(
                session
                    .prepare_with_device_bindings(&inputs, &mut [])
                    .is_err()
            );
            assert!(
                session.run(&inputs).is_err(),
                "no unfunded staging fallback"
            );
            assert_eq!(books.used(Tier::Host), 0);
            assert_eq!(manager.process_used(Tier::Host), 0);
            continue;
        }
        let mut identity = None;
        for _ in 0..8 {
            let peak = session
                .prepare_with_device_bindings(&inputs, &mut [])
                .unwrap()
                .require_complete()
                .unwrap();
            assert_eq!(
                peak.step().bytes,
                80,
                "max(48-byte LayerNorm,80-byte Skip), not the sum"
            );
            let outputs = session.run(&inputs).unwrap();
            assert_eq!(outputs.len(), 6);
            for (output, expected) in outputs.iter().zip(&expected_outputs) {
                let actual = output.try_as_slice_f32().unwrap();
                assert_eq!(actual.len(), expected.len());
                for (&got, &want) in actual.iter().zip(expected) {
                    assert!((f64::from(got) - want).abs() < 1e-5, "{got} vs {want}");
                }
            }
            let snapshot = manager.snapshot().unwrap();
            assert_eq!(snapshot.allocations.len(), 1);
            let expected = identity.get_or_insert(snapshot.allocations[0].identity);
            assert_eq!(snapshot.allocations[0].identity, *expected);
            assert_eq!(books.used(Tier::Host), 80);
        }
        pool.trim_idle(80).unwrap();
        assert_eq!(books.used(Tier::Host), 0);
        session.run(&inputs).unwrap();
        assert_eq!(
            books.used(Tier::Host),
            80,
            "cached metadata re-admits reclaimed backing"
        );
        drop(session);
        pool.trim_idle(u64::MAX).unwrap();
        assert_eq!(books.used(Tier::Host), 0);
        assert_eq!(manager.process_used(Tier::Host), 0);
    }
}

#[test]
fn broadcast_affine_shapes_change_without_expanded_workspace_or_stale_coefficients() {
    use onnx_runtime_ir::{Attribute, Dim};
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x_id = graph.create_named_value("X", DataType::Float32, static_shape([2, 3, 4]));
    let scale_shape = (0..3)
        .map(|i| Dim::Symbolic(graph.create_symbol(Some(format!("scale_{i}")))))
        .collect();
    let bias_shape = (0..3)
        .map(|i| Dim::Symbolic(graph.create_symbol(Some(format!("bias_{i}")))))
        .collect();
    let s_id = graph.create_named_value("scale", DataType::Float32, scale_shape);
    let b_id = graph.create_named_value("B", DataType::Float32, bias_shape);
    let y_id = graph.create_named_value("Y", DataType::Float32, static_shape([2, 3, 4]));
    let m_id = graph.create_named_value("mean", DataType::Float32, static_shape([2, 1, 1]));
    let r_id = graph.create_named_value("inv", DataType::Float32, static_shape([2, 1, 1]));
    for value in [x_id, s_id, b_id] {
        graph.add_input(value);
    }
    for value in [y_id, m_id, r_id] {
        graph.add_output(value);
    }
    let mut node = Node::new(
        NodeId(0),
        "LayerNormalization",
        vec![Some(x_id), Some(s_id), Some(b_id)],
        vec![y_id, m_id, r_id],
    );
    node.attributes.insert("axis".into(), Attribute::Int(1));
    graph.insert_node(node);
    let bytes =
        onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&graph)).unwrap();
    let values = (0..24).map(|i| i as f32 - 12.).collect::<Vec<_>>();
    let x = Tensor::from_f32(&[2, 3, 4], &values).unwrap();
    let cases = [
        ([1, 3, 4], [1, 3, 4]),
        ([1, 1, 1], [1, 3, 1]),
        ([2, 1, 1], [1, 1, 4]),
        ([2, 3, 4], [1, 1, 1]),
        ([1, 3, 4], [1, 3, 4]),
    ];
    for budget in [111u64, 112] {
        let books = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
            DeviceKey::HOST,
            0,
            budget,
            0,
        )));
        let manager = ProcessMemoryManager::new().unwrap();
        let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
        let resources = onnx_runtime_ep_cpu::CpuMemoryResources::new(
            onnx_runtime_ep_cpu::DerivedWeightMemory::governed(books.as_ref(), HolderId::new(13))
                .unwrap(),
            Default::default(),
        )
        .unwrap()
        .with_workspace_pool(pool.clone())
        .unwrap();
        let cpu = onnx_runtime_ep_cpu::CpuExecutionProvider::initialized_with_resources(resources)
            .unwrap();
        let mut session = InferenceSession::builder()
            .model_bytes(&bytes)
            .execution_provider(Arc::new(cpu))
            .build()
            .unwrap();
        let mut identity = None;
        for (case, (scale_shape, bias_shape)) in cases.into_iter().enumerate() {
            let scale_values = (0..scale_shape.iter().product::<usize>())
                .map(|i| 0.5 + i as f32 * 0.125 + case as f32 * 0.025)
                .collect::<Vec<_>>();
            let bias_values = (0..bias_shape.iter().product::<usize>())
                .map(|i| -0.25 + i as f32 * 0.05)
                .collect::<Vec<_>>();
            let scale = Tensor::from_f32(&scale_shape, &scale_values).unwrap();
            let bias = Tensor::from_f32(&bias_shape, &bias_values).unwrap();
            let inputs = [("X", &x), ("scale", &scale), ("B", &bias)];
            if budget < 112 {
                assert!(
                    session
                        .prepare_with_device_bindings(&inputs, &mut [])
                        .is_err()
                );
                assert!(session.run(&inputs).is_err());
                assert_eq!(books.used(Tier::Host), 0);
                continue;
            }
            let peak = session
                .prepare_with_device_bindings(&inputs, &mut [])
                .unwrap()
                .require_complete()
                .unwrap();
            assert_eq!(
                peak.step().bytes,
                112,
                "only 24 Y + 2 mean + 2 inv floats, independent of affine shapes"
            );
            let outputs = session.run(&inputs).unwrap();
            assert_eq!(outputs.len(), 3);
            for (g, row) in values.chunks_exact(12).enumerate() {
                let mean = row.iter().map(|&x| f64::from(x)).sum::<f64>() / 12.;
                let variance = row
                    .iter()
                    .map(|&x| (f64::from(x) - mean).powi(2))
                    .sum::<f64>()
                    / 12.;
                let inverse = 1. / (variance + 1e-5).sqrt();
                assert!((f64::from(outputs[1].try_as_slice_f32().unwrap()[g]) - mean).abs() < 1e-5);
                assert!(
                    (f64::from(outputs[2].try_as_slice_f32().unwrap()[g]) - inverse).abs() < 1e-5
                );
                for (e, &x) in row.iter().enumerate() {
                    let flat = g * 12 + e;
                    let si = match case {
                        0 | 4 => e,
                        1 => 0,
                        2 => g,
                        3 => flat,
                        _ => unreachable!(),
                    };
                    let bi = match case {
                        0 | 4 => e,
                        1 => e / 4,
                        2 => e % 4,
                        3 => 0,
                        _ => unreachable!(),
                    };
                    let expected = (f64::from(x) - mean) * inverse * f64::from(scale_values[si])
                        + f64::from(bias_values[bi]);
                    assert!(
                        (f64::from(outputs[0].try_as_slice_f32().unwrap()[flat]) - expected).abs()
                            < 2e-5
                    );
                }
            }
            let snapshot = manager.snapshot().unwrap();
            assert_eq!(snapshot.allocations.len(), 1);
            let expected = identity.get_or_insert(snapshot.allocations[0].identity);
            assert_eq!(snapshot.allocations[0].identity, *expected);
            assert_eq!(books.used(Tier::Host), 112);
        }
        drop(session);
        pool.trim_idle(u64::MAX).unwrap();
        assert_eq!(books.used(Tier::Host), 0);
        assert_eq!(manager.process_used(Tier::Host), 0);
    }
}

#[test]
fn cpu_scoped_layernorm_fusion_runs_broadcast_coefficients_with_one_admitted_buffer() {
    use onnx_runtime_ir::{Attribute, TensorData, WeightRef};
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = graph.create_named_value("X", DataType::Float32, static_shape([2, 3, 4]));
    let scale = graph.create_named_value("scale", DataType::Float32, static_shape([4]));
    let beta = graph.create_named_value("B", DataType::Float32, static_shape([]));
    for input in [x, scale, beta] {
        graph.add_input(input);
    }
    let two = graph.create_named_value("two", DataType::Float32, static_shape([]));
    let eps = graph.create_named_value("eps", DataType::Float32, static_shape([]));
    for (id, value) in [(two, 2.0f32), (eps, 1e-5)] {
        graph.set_initializer(
            id,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![],
                value.to_le_bytes().to_vec(),
            )),
        );
    }
    let mean = graph.create_named_value("mean", DataType::Float32, static_shape([2, 1, 1]));
    let diff = graph.create_named_value("diff", DataType::Float32, static_shape([2, 3, 4]));
    let square = graph.create_named_value("square", DataType::Float32, static_shape([2, 3, 4]));
    let var = graph.create_named_value("var", DataType::Float32, static_shape([2, 1, 1]));
    let add = graph.create_named_value("add", DataType::Float32, static_shape([2, 1, 1]));
    let sqrt = graph.create_named_value("sqrt", DataType::Float32, static_shape([2, 1, 1]));
    let norm = graph.create_named_value("norm", DataType::Float32, static_shape([2, 3, 4]));
    let scaled = graph.create_named_value("scaled", DataType::Float32, static_shape([2, 3, 4]));
    let y = graph.create_named_value("Y", DataType::Float32, static_shape([2, 3, 4]));
    graph.add_output(y);
    for (id, op, inputs, output) in [
        (0, "ReduceMean", vec![Some(x)], mean),
        (1, "Sub", vec![Some(x), Some(mean)], diff),
        (2, "Pow", vec![Some(diff), Some(two)], square),
        (3, "ReduceMean", vec![Some(square)], var),
        (4, "Add", vec![Some(var), Some(eps)], add),
        (5, "Sqrt", vec![Some(add)], sqrt),
        (6, "Div", vec![Some(diff), Some(sqrt)], norm),
        (7, "Mul", vec![Some(norm), Some(scale)], scaled),
        (8, "Add", vec![Some(scaled), Some(beta)], y),
    ] {
        let mut node = Node::new(NodeId(id), op, inputs, vec![output]);
        if op == "ReduceMean" {
            node.attributes
                .insert("axes".into(), Attribute::Ints(vec![1, 2]));
            node.attributes.insert("keepdims".into(), Attribute::Int(1));
        }
        graph.insert_node(node);
    }
    assert_eq!(graph.num_nodes(), 9);
    let bytes =
        onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&graph)).unwrap();
    let values = (0..24).map(|i| i as f32 - 12.).collect::<Vec<_>>();
    let scales = [0.5, 1., 1.5, 2.];
    let x = Tensor::from_f32(&[2, 3, 4], &values).unwrap();
    let scale = Tensor::from_f32(&[4], &scales).unwrap();
    let beta = Tensor::from_f32(&[], &[0.25]).unwrap();
    let inputs = [("X", &x), ("scale", &scale), ("B", &beta)];
    for budget in [95u64, 96] {
        let books = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
            DeviceKey::HOST,
            0,
            budget,
            0,
        )));
        let manager = ProcessMemoryManager::new().unwrap();
        let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
        let resources = onnx_runtime_ep_cpu::CpuMemoryResources::new(
            onnx_runtime_ep_cpu::DerivedWeightMemory::governed(books.as_ref(), HolderId::new(13))
                .unwrap(),
            Default::default(),
        )
        .unwrap()
        .with_workspace_pool(pool.clone())
        .unwrap();
        let provider =
            onnx_runtime_ep_cpu::CpuExecutionProvider::initialized_with_resources(resources)
                .unwrap();
        let mut session = InferenceSession::builder()
            .model_bytes(&bytes)
            .execution_provider(Arc::new(provider))
            .build()
            .unwrap();
        assert_eq!(
            session.graph().num_nodes(),
            1,
            "the CPU EP must fuse the whole decomposition"
        );
        let fused = session.graph().nodes.values().next().unwrap();
        assert_eq!(fused.op_type, "LayerNormalization");
        assert_eq!(fused.domain, "com.microsoft");
        if budget < 96 {
            assert!(
                session
                    .prepare_with_device_bindings(&inputs, &mut [])
                    .is_err()
            );
            assert!(session.run(&inputs).is_err());
            assert_eq!(books.used(Tier::Host), 0);
            continue;
        }
        assert_eq!(
            session
                .prepare_with_device_bindings(&inputs, &mut [])
                .unwrap()
                .require_complete()
                .unwrap()
                .step()
                .bytes,
            96
        );
        let outputs = session.run(&inputs).unwrap();
        let actual = outputs[0].try_as_slice_f32().unwrap();
        for (g, row) in values.chunks_exact(12).enumerate() {
            let m = row.iter().map(|&x| f64::from(x)).sum::<f64>() / 12.;
            let v = row.iter().map(|&x| (f64::from(x) - m).powi(2)).sum::<f64>() / 12.;
            let inv = 1. / (v + 1e-5).sqrt();
            for (e, &x) in row.iter().enumerate() {
                let want = (f64::from(x) - m) * inv * f64::from(scales[e % 4]) + 0.25;
                assert!((f64::from(actual[g * 12 + e]) - want).abs() < 1e-5);
            }
        }
        assert_eq!(books.used(Tier::Host), 96);
        drop(session);
        pool.trim_idle(u64::MAX).unwrap();
        assert_eq!(books.used(Tier::Host), 0);
        assert_eq!(manager.process_used(Tier::Host), 0);
    }
}
