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
