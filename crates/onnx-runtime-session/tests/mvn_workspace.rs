use onnx_runtime_ir::{Attribute, DataType, Dim, Graph, Node, NodeId, TensorLayout};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use onnx_runtime_session::{InferenceSession, Tensor, WorkspacePreparationReport};
use std::sync::Arc;
fn model(dtype: DataType, two: bool) -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 29);
    let shape = (0..3)
        .map(|i| Dim::Symbolic(graph.create_symbol(Some(format!("d{i}")))))
        .collect::<Vec<_>>();
    let x = graph.create_named_value("X", dtype, shape.clone());
    let first = graph.create_named_value("first", dtype, shape.clone());
    let y = graph.create_named_value("Y", dtype, shape);
    graph.add_input(x);
    graph.add_output(y);
    let nodes = if two {
        vec![(0, x, first, vec![-1]), (1, first, y, vec![0, -1])]
    } else {
        vec![(0, x, y, vec![-1])]
    };
    for (id, input, output, axes) in nodes {
        let mut node = Node::new(
            NodeId(id),
            "MeanVarianceNormalization",
            vec![Some(input)],
            vec![output],
        );
        node.attributes.insert("axes".into(), Attribute::Ints(axes));
        node.attributes
            .insert("epsilon".into(), Attribute::Float(0.125));
        graph.insert_node(node);
    }
    onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&graph)).unwrap()
}
fn setup(
    dtype: DataType,
    budget: u64,
    two: bool,
) -> (
    InferenceSession,
    Arc<LedgerGovernor>,
    HostWorkspacePool,
    ProcessMemoryManager,
) {
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
    let cpu =
        onnx_runtime_ep_cpu::CpuExecutionProvider::initialized_with_resources(resources).unwrap();
    let session = InferenceSession::builder()
        .model_bytes(&model(dtype, two))
        .execution_provider(Arc::new(cpu))
        .build()
        .unwrap();
    (session, books, pool, manager)
}
fn encode(dtype: DataType, values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| match dtype {
            DataType::Float32 => (v as f32).to_ne_bytes().to_vec(),
            DataType::Float64 => v.to_ne_bytes().to_vec(),
            DataType::Float16 => half::f16::from_f64(v).to_bits().to_ne_bytes().to_vec(),
            DataType::BFloat16 => half::bf16::from_f64(v).to_bits().to_ne_bytes().to_vec(),
            _ => unreachable!(),
        })
        .collect()
}
fn decode(dtype: DataType, bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(dtype.byte_size())
        .map(|b| match dtype {
            DataType::Float32 => f64::from(f32::from_ne_bytes(b.try_into().unwrap())),
            DataType::Float64 => f64::from_ne_bytes(b.try_into().unwrap()),
            DataType::Float16 => {
                half::f16::from_bits(u16::from_ne_bytes(b.try_into().unwrap())).to_f64()
            }
            DataType::BFloat16 => {
                half::bf16::from_bits(u16::from_ne_bytes(b.try_into().unwrap())).to_f64()
            }
            _ => unreachable!(),
        })
        .collect()
}
fn canonical(dtype: DataType, values: &[f64]) -> Vec<f64> {
    decode(dtype, &encode(dtype, values))
}
fn reference(values: &[f64], shape: [usize; 3], axes: &[usize]) -> Vec<f64> {
    let coords = |j: usize| {
        [
            j / (shape[1] * shape[2]),
            (j / shape[2]) % shape[1],
            j % shape[2],
        ]
    };
    (0..values.len())
        .map(|i| {
            let c = coords(i);
            let group = |j: usize| {
                let other = coords(j);
                (0..3).all(|d| axes.contains(&d) || c[d] == other[d])
            };
            let width = (0..values.len()).filter(|&j| group(j)).count() as f64;
            let mean = (0..values.len())
                .filter(|&j| group(j))
                .map(|j| values[j])
                .sum::<f64>()
                / width;
            let variance = (0..values.len())
                .filter(|&j| group(j))
                .map(|j| (values[j] - mean).powi(2))
                .sum::<f64>()
                / width;
            (values[i] - mean) / (variance.sqrt() + 0.125)
        })
        .collect()
}
fn compare(dtype: DataType, actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    let tolerance = match dtype {
        DataType::Float16 => 0.005,
        DataType::BFloat16 => 0.03,
        DataType::Float64 => 1e-10,
        _ => 3e-5,
    };
    for (&a, &e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < tolerance, "{dtype:?}: {a} versus {e}");
    }
}
#[test]
fn public_dynamic_bindings_preserve_zero_scratch_and_admit_one_fallback_peak() {
    for dtype in [
        DataType::Float32,
        DataType::Float64,
        DataType::Float16,
        DataType::BFloat16,
    ] {
        let peak = (24 * dtype.byte_size()) as u64;
        for (strided, budget) in [(false, 0), (true, peak - 1), (true, peak)] {
            let (mut session, books, pool, manager) = setup(dtype, budget, true);
            let mut identity = None;
            for (iteration, shape) in [[2usize, 3, 4], [1, 2, 5], [2, 3, 4]]
                .into_iter()
                .enumerate()
            {
                let values = canonical(
                    dtype,
                    &(0..shape.iter().product())
                        .map(|i| (i as f64 * 0.175).cos() * 2. + iteration as f64 * 0.25)
                        .collect::<Vec<_>>(),
                );
                let x = Tensor::from_raw(dtype, shape.to_vec(), &encode(dtype, &values)).unwrap();
                let input_array = [("X", &x)];
                let inputs: &[(&str, &Tensor)] = if strided { &[] } else { &input_array };
                let mut physical = vec![0.; values.len() * 2];
                for b in 0..shape[0] {
                    for r in 0..shape[1] {
                        for c in 0..shape[2] {
                            physical[(b * shape[1] + r) * shape[2] * 2 + c] =
                                values[(b * shape[1] + r) * shape[2] + c];
                        }
                    }
                }
                let physical_bytes = encode(dtype, &physical);
                let mut binding = if strided {
                    Some(
                        session
                            .allocate_device_binding_fixed_strides(
                                "X",
                                "Y",
                                dtype,
                                vec![shape[0], shape[1], shape[2] * 2],
                                shape.to_vec(),
                            )
                            .unwrap(),
                    )
                } else {
                    None
                };
                if let Some(binding) = binding.as_mut() {
                    binding.write_bytes(0, &physical_bytes).unwrap();
                }
                let report = if let Some(binding) = binding.as_mut() {
                    session.prepare_with_device_bindings(inputs, std::slice::from_mut(binding))
                } else {
                    session.prepare_with_device_bindings(inputs, &mut [])
                }
                .unwrap();
                assert!(matches!(
                    report,
                    WorkspacePreparationReport::Deferred { .. }
                ));
                if strided && budget < peak {
                    let binding = binding.as_mut().unwrap();
                    assert!(
                        session
                            .run_with_device_bindings(inputs, std::slice::from_mut(binding))
                            .is_err()
                    );
                    assert_eq!(binding.read_bytes().unwrap(), physical_bytes);
                    assert!(manager.snapshot().unwrap().allocations.is_empty());
                    assert_eq!(books.used(Tier::Host), 0);
                    break;
                }
                let first = canonical(dtype, &reference(&values, shape, &[2]));
                let expected = canonical(dtype, &reference(&first, shape, &[0, 2]));
                for _ in 0..2 {
                    let actual = if let Some(binding) = binding.as_mut() {
                        binding.write_bytes(0, &physical_bytes).unwrap();
                        session
                            .run_with_device_bindings(inputs, std::slice::from_mut(binding))
                            .unwrap();
                        let output = decode(dtype, &binding.read_bytes().unwrap());
                        let mut logical = Vec::new();
                        for b in 0..shape[0] {
                            for r in 0..shape[1] {
                                for c in 0..shape[2] {
                                    logical.push(output[(b * shape[1] + r) * shape[2] * 2 + c]);
                                }
                                for c in shape[2]..shape[2] * 2 {
                                    assert_eq!(output[(b * shape[1] + r) * shape[2] * 2 + c], 0.);
                                }
                            }
                        }
                        logical
                    } else {
                        let outputs = session.run(inputs).unwrap();
                        decode(dtype, outputs[0].as_bytes())
                    };
                    compare(dtype, &actual, &expected);
                    let snapshot = manager.snapshot().unwrap();
                    if strided {
                        assert_eq!(snapshot.allocations.len(), 1);
                        assert_eq!(
                            snapshot.allocations[0].identity,
                            *identity.get_or_insert(snapshot.allocations[0].identity)
                        );
                        assert_eq!(books.used(Tier::Host), peak);
                    } else {
                        assert!(snapshot.allocations.is_empty());
                        assert_eq!(books.used(Tier::Host), 0);
                    }
                }
                if iteration == 1 {
                    pool.trim_idle(u64::MAX).unwrap();
                    assert_eq!(books.used(Tier::Host), 0);
                    identity = None;
                }
            }
            drop(session);
            pool.trim_idle(u64::MAX).unwrap();
            assert_eq!(manager.process_used(Tier::Host), 0);
        }
    }
}
#[test]
fn public_exact_inplace_binding_runs_under_zero_workspace_capacity() {
    let shape = [2, 3, 4];
    for dtype in [
        DataType::Float32,
        DataType::Float64,
        DataType::Float16,
        DataType::BFloat16,
    ] {
        let (mut session, books, pool, manager) = setup(dtype, 0, false);
        let values = canonical(
            dtype,
            &(0..24).map(|i| i as f64 * 0.125 - 1.75).collect::<Vec<_>>(),
        );
        let expected = canonical(dtype, &reference(&values, shape, &[2]));
        let mut binding = session
            .allocate_device_binding("X", Some("Y"), dtype, shape.to_vec(), shape.to_vec())
            .unwrap();
        for _ in 0..2 {
            binding.write_bytes(0, &encode(dtype, &values)).unwrap();
            assert!(matches!(
                session
                    .prepare_with_device_bindings(&[], std::slice::from_mut(&mut binding))
                    .unwrap(),
                WorkspacePreparationReport::Deferred { .. }
            ));
            session
                .run_with_device_bindings(&[], std::slice::from_mut(&mut binding))
                .unwrap();
            compare(
                dtype,
                &decode(dtype, &binding.read_bytes().unwrap()),
                &expected,
            );
            assert_eq!(books.used(Tier::Host), 0);
            assert!(manager.snapshot().unwrap().allocations.is_empty());
        }
        drop(binding);
        drop(session);
        pool.trim_idle(u64::MAX).unwrap();
        assert_eq!(manager.process_used(Tier::Host), 0);
    }
}

#[test]
fn ordinary_tensor_uploads_reject_strides_before_prepare_or_run_reads() {
    let (mut session, books, _pool, manager) = setup(DataType::Float32, 0, false);
    let mut x = Tensor::from_f32(&[2, 6, 4], &[1.; 48]).unwrap();
    x.shape = vec![2, 3, 4];
    x.layout = TensorLayout::strided(vec![24, 8, 1]);
    let inputs = [("X", &x)];
    let error = session
        .prepare_with_device_bindings(&inputs, &mut [])
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("host input 'X'")
            && error.contains("non-contiguous")
            && error.contains("allocate_device_binding_fixed_strides"),
        "{error}"
    );
    let error = session.run(&inputs).err().unwrap().to_string();
    assert!(
        error.contains("host input 'X'") && error.contains("non-contiguous"),
        "{error}"
    );
    assert_eq!(books.used(Tier::Host), 0);
    assert!(manager.snapshot().unwrap().allocations.is_empty());
    let mut dense = Tensor::from_f32(
        &[2, 3, 4],
        &(0..24).map(|i| i as f32 * 0.125).collect::<Vec<_>>(),
    )
    .unwrap();
    dense.layout = TensorLayout::strided(vec![12, 4, 1]);
    assert!(matches!(
        session
            .prepare_with_device_bindings(&[("X", &dense)], &mut [])
            .unwrap(),
        WorkspacePreparationReport::Deferred { .. }
    ));
    assert!(
        session.run(&[("X", &dense)]).is_ok(),
        "canonical contiguous strides remain valid even with a custom format hint"
    );
}
