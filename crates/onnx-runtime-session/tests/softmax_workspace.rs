use onnx_runtime_ir::{Attribute, DataType, Dim, Graph, Node, NodeId};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use onnx_runtime_session::{InferenceSession, Tensor};
use std::sync::Arc;

fn model(opset: u64, dtype: DataType) -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let shape = (0..3)
        .map(|i| Dim::Symbolic(graph.create_symbol(Some(format!("d{i}")))))
        .collect::<Vec<_>>();
    let x = graph.create_named_value("X", dtype, shape.clone());
    let first = graph.create_named_value("first", dtype, shape.clone());
    let y = graph.create_named_value("Y", dtype, shape);
    graph.add_input(x);
    graph.add_output(y);
    for (id, input, output, axis) in [(0, x, first, 1), (1, first, y, -1)] {
        let mut node = Node::new(NodeId(id), "Softmax", vec![Some(input)], vec![output]);
        node.attributes.insert("axis".into(), Attribute::Int(axis));
        graph.insert_node(node);
    }
    onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&graph)).unwrap()
}

fn reference(values: &[f32], shape: [usize; 3], legacy: bool, axis: usize) -> Vec<f32> {
    let count = values.len();
    (0..count)
        .map(|flat| {
            let coords = |j: usize| {
                [
                    j / (shape[1] * shape[2]),
                    (j / shape[2]) % shape[1],
                    j % shape[2],
                ]
            };
            let c = coords(flat);
            let group = |j: usize| {
                let other = coords(j);
                (0..3).all(|d| d == axis || (legacy && d > axis) || c[d] == other[d])
            };
            let max = (0..count)
                .filter(|&j| group(j))
                .map(|j| f64::from(values[j]))
                .fold(f64::NEG_INFINITY, f64::max);
            let sum = (0..count)
                .filter(|&j| group(j))
                .map(|j| (f64::from(values[j]) - max).exp())
                .sum::<f64>();
            ((f64::from(values[flat]) - max).exp() / sum) as f32
        })
        .collect()
}

#[test]
fn both_opsets_share_one_peak_refuse_unfunded_staging_and_readmit_after_trim() {
    for opset in [11, 13] {
        for (dtype, budget) in [
            (DataType::Float32, 0u64),
            (DataType::Float64, 95),
            (DataType::Float64, 96),
        ] {
            let books = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
                DeviceKey::HOST,
                0,
                budget,
                0,
            )));
            let manager = ProcessMemoryManager::new().unwrap();
            let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
            let resources = onnx_runtime_ep_cpu::CpuMemoryResources::new(
                onnx_runtime_ep_cpu::DerivedWeightMemory::governed(
                    books.as_ref(),
                    HolderId::new(13),
                )
                .unwrap(),
                Default::default(),
            )
            .unwrap()
            .with_workspace_pool(pool.clone())
            .unwrap();
            let cpu =
                onnx_runtime_ep_cpu::CpuExecutionProvider::initialized_with_resources(resources)
                    .unwrap();
            let mut session = InferenceSession::builder()
                .model_bytes(&model(opset, dtype))
                .execution_provider(Arc::new(cpu))
                .build()
                .unwrap();
            let mut identity = None;
            for (iteration, shape) in [[2, 3, 4], [1, 2, 5], [2, 3, 4]].into_iter().enumerate() {
                let values = (0..shape.iter().product())
                    .map(|i| (i as f32 * 0.175).cos() * 2. + iteration as f32 * 0.25)
                    .collect::<Vec<_>>();
                let x = if dtype == DataType::Float32 {
                    Tensor::from_f32(&shape, &values).unwrap()
                } else {
                    Tensor::from_raw(
                        dtype,
                        shape.to_vec(),
                        &values
                            .iter()
                            .flat_map(|&v| f64::from(v).to_ne_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap()
                };
                let inputs = [("X", &x)];
                if dtype != DataType::Float32 && budget < 96 {
                    assert!(
                        session
                            .prepare_with_device_bindings(&inputs, &mut [])
                            .is_err()
                    );
                    assert!(
                        session.run(&inputs).is_err(),
                        "run must not bypass the staging budget"
                    );
                    assert!(manager.snapshot().unwrap().allocations.is_empty());
                    assert_eq!(books.used(Tier::Host), 0);
                    break;
                }
                let report = session
                    .prepare_with_device_bindings(&inputs, &mut [])
                    .unwrap();
                if dtype == DataType::Float32 {
                    assert!(matches!(
                        report,
                        onnx_runtime_session::WorkspacePreparationReport::Deferred { .. }
                    ));
                } else {
                    assert_eq!(
                        report.require_complete().unwrap().step().bytes,
                        (shape.iter().product::<usize>() * 4) as u64
                    );
                }
                let expected = reference(
                    &reference(&values, shape, opset < 13, 1),
                    shape,
                    opset < 13,
                    2,
                );
                for _ in 0..2 {
                    let outputs = session.run(&inputs).unwrap();
                    let actual = if dtype == DataType::Float32 {
                        outputs[0].try_as_slice_f32().unwrap().to_vec()
                    } else {
                        outputs[0]
                            .as_bytes()
                            .chunks_exact(8)
                            .map(|b| f64::from_ne_bytes(b.try_into().unwrap()) as f32)
                            .collect()
                    };
                    for (&actual, want) in actual.iter().zip(&expected) {
                        assert!(
                            (actual - want).abs() < 3e-6,
                            "opset={opset} shape={shape:?} {actual} vs {want}"
                        );
                    }
                    let snapshot = manager.snapshot().unwrap();
                    if dtype == DataType::Float32 {
                        assert!(snapshot.allocations.is_empty());
                        assert_eq!(books.used(Tier::Host), 0);
                        continue;
                    }
                    assert_eq!(snapshot.allocations.len(), 1);
                    assert_eq!(
                        snapshot.allocations[0].identity,
                        *identity.get_or_insert(snapshot.allocations[0].identity)
                    );
                    assert_eq!(books.used(Tier::Host), 96);
                }
                if iteration == 1 {
                    pool.trim_idle(u64::MAX).unwrap();
                    assert_eq!(books.used(Tier::Host), 0);
                    identity = None;
                }
            }
            drop(session);
            pool.trim_idle(u64::MAX).unwrap();
            assert_eq!(books.used(Tier::Host), 0);
            assert_eq!(manager.process_used(Tier::Host), 0);
        }
    }
}
