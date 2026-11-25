//! Whole-process public-session fixed-work probe. Parent wait4 measures all CPU.
use onnx_runtime_ep_cpu::{
    CpuExecutionProvider, CpuMemoryResources, CpuWeightCacheLimits, DerivedWeightMemory,
};
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, TensorData, WeightRef, static_shape,
};
use onnx_runtime_loader::{Model, encode_model};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use onnx_runtime_session::{InferenceSession, Tensor};
use std::{hint::black_box, sync::Arc};
fn model(m: usize, k: usize, n: usize) -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 24);
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let a = graph.create_named_value("a", DataType::Float32, static_shape([m, k]));
    let y = graph.create_named_value("y", DataType::Float32, static_shape([m, k]));
    let router = graph.create_named_value("router", DataType::Float32, static_shape([m, 4]));
    graph.set_initializer(
        router,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![m, 4],
            (0..m)
                .flat_map(|row| {
                    if row % 2 == 0 {
                        [4f32, 3., -4., -5.]
                    } else {
                        [-4f32, -5., 4., 3.]
                    }
                })
                .flat_map(f32::to_le_bytes)
                .collect(),
        )),
    );
    let mut pack = |name: &str, input: usize, output: usize| {
        let w =
            graph.create_named_value(name, DataType::Uint8, static_shape([4, output, input / 2]));
        let mut state = 0x7a6351d9u32 ^ (output as u32).rotate_left(9) ^ input as u32;
        let bytes = (0..4 * output * input / 2)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 24) as u8
            })
            .collect();
        graph.set_initializer(
            w,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Uint8,
                vec![4, output, input / 2],
                bytes,
            )),
        );
        let scales = graph.create_named_value(
            format!("{name}_scales"),
            DataType::Float32,
            static_shape([4, output, input / 16]),
        );
        graph.set_initializer(
            scales,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![4, output, input / 16],
                (0..4 * output * input / 16)
                    .flat_map(|_| (1f32 / 64.).to_le_bytes())
                    .collect(),
            )),
        );
        (w, scales)
    };
    let (w1, s1) = pack("fc1", k, n);
    let (w2, s2) = pack("fc2", n, k);
    let gated = std::env::var("NXRT_BLOCK_GATED").ok().as_deref() == Some("1");
    let fc3 = gated.then(|| pack("fc3", k, n));
    let mut node = Node::new(
        NodeId(0),
        "QMoE",
        vec![
            Some(a),
            Some(router),
            Some(w1),
            Some(s1),
            None,
            Some(w2),
            Some(s2),
        ],
        vec![y],
    );
    node.domain = "com.microsoft".into();
    if let Some((w3, s3)) = fc3 {
        node.inputs.extend([None, Some(w3), Some(s3)]);
    }
    for (name, value) in [
        ("expert_weight_bits", Attribute::Int(4)),
        ("block_size", Attribute::Int(16)),
        (
            "activation_type",
            Attribute::String(if gated {
                b"swiglu".to_vec()
            } else {
                b"identity".to_vec()
            }),
        ),
        ("k", Attribute::Int(2)),
        ("normalize_routing_weights", Attribute::Int(1)),
    ] {
        node.attributes.insert(name.into(), value);
    }
    node.attributes
        .insert("swiglu_fusion".into(), Attribute::Int(0));
    graph.add_input(a);
    graph.add_output(y);
    graph.insert_node(node);
    encode_model(&Model::new(&graph)).unwrap()
}
fn execute(session: &mut InferenceSession, input: &Tensor) {
    session
        .prepare_with_device_bindings(&[("a", input)], &mut [])
        .unwrap()
        .require_complete()
        .unwrap();
    let output = session.run(black_box(&[("a", input)])).unwrap();
    black_box(&output);
    // Output owners are dropped here, before the terminal process CPU receipt.
}
fn run(m: usize, k: usize, n: usize, reps: usize, path: &str) {
    let limit = 64 * 1024 * 1024;
    let dense = std::env::var("NXRT_GATHER_DENSE").ok().as_deref() == Some("1");
    let ledger = LeaseLedger::new_for_device(DeviceKey::HOST, 0, limit, 0);
    let books = Arc::new(LedgerGovernor::new(ledger.clone()));
    let manager = ProcessMemoryManager::new().unwrap();
    let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
    let memory = DerivedWeightMemory::governed(books.as_ref(), HolderId::new(13)).unwrap();
    let resources = CpuMemoryResources::new(
        memory,
        CpuWeightCacheLimits {
            dense_bytes: if dense { limit as usize } else { 0 },
            expert_bytes: 0,
        },
    )
    .unwrap()
    .with_workspace_pool(pool.clone())
    .unwrap();
    let ep = Arc::new(CpuExecutionProvider::initialized_with_resources(resources).unwrap());
    let bytes = model(m, k, n);
    let construction_width: usize = std::env::var("NXRT_CONSTRUCTION_WIDTH").unwrap().parse().unwrap();
    let construction_pool = rayon::ThreadPoolBuilder::new().num_threads(construction_width).build().unwrap();
    let mut session = construction_pool.install(|| InferenceSession::builder()
        .model_bytes(&bytes)
        .option("optimization", "none")
        .execution_provider(ep.clone())
        .build()
        .unwrap());
    let input = Tensor::from_f32(
        &[m, k],
        &(0..m * k)
            .map(|i| (i % 31) as f32 / 1000. - 0.015)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let construction_bytes = construction_pool.install(|| session
        .prepare_with_device_bindings(&[("a", &input)], &mut [])
        .unwrap().require_complete().unwrap().step().bytes);
    let requirement = session
        .prepare_with_device_bindings(&[("a", &input)], &mut [])
        .unwrap()
        .require_complete()
        .unwrap();
    pool.trim_idle(u64::MAX).unwrap();
    let peak_workspace = construction_bytes.max(requirement.step().bytes);
    // A zero cache-retention knob does NOT force QMoE row streaming. Restrict
    // the actual shared authority to just the prepared, active workspace.
    // Any positive transient expert grant must then decline while it is lent.
    if !dense {
        ledger.set_limit(Tier::Host, peak_workspace);
    }
    execute(&mut session, &input);
    for _ in 0..4 {
        execute(&mut session, &input);
    }
    let funded = books.used(Tier::Host);
    let workspace_funded = manager.process_used(Tier::Host);
    let derived = funded - workspace_funded;
    assert_eq!(workspace_funded, peak_workspace);
    assert_eq!(requirement.persistent().bytes, 0);
    let gated = std::env::var("NXRT_BLOCK_GATED").ok().as_deref() == Some("1");
    assert_eq!(ledger.oversubscribed_bytes(Tier::Host), 0);
    if !dense {
        assert_eq!(ledger.available(Tier::Host), 0);
    }
    let host_limit = ledger.limit(Tier::Host);
    let logical_workspace = requirement.step().bytes;
    let expected_derived = 0; // Nonpageable QMoE expert sets are transient, not retained.
    assert_eq!(derived, expected_derived as u64);
    // Fixed work. This process deliberately has no short-window CPU clock.
    for _ in 0..reps {
        execute(&mut session, &input);
    }
    assert_eq!(
        books.used(Tier::Host),
        funded,
        "warm inference must not grow retained backing"
    );
    session
        .prepare_with_device_bindings(&[("a", &input)], &mut [])
        .unwrap()
        .require_complete()
        .unwrap();
    let (trace, collector) = onnx_runtime_tracer::TraceContext::in_memory();
    session.set_trace_context(trace);
    let output = session.run(&[("a", &input)]).unwrap();
    let trace_json = collector.to_chrome_json();
    let streamed_witness = trace_json.contains("qmoe.streamed_rows");
    assert_eq!(
        streamed_witness, !dense,
        "actual final dispatch must witness the intended weight path"
    );
    std::fs::write(format!("{path}.trace.json"), &trace_json).unwrap();
    std::fs::write(path, output[0].as_bytes()).unwrap();
    assert_eq!(output[0].shape, [m, k]);
    let values = output[0].to_vec_f32();
    assert!(values.iter().all(|x| x.is_finite()));
    assert!(values.iter().any(|x| *x != 0.0));
    let argmax = values
        .chunks_exact(k)
        .map(|row| {
            (0..k)
                .max_by(|&a, &b| row[a].total_cmp(&row[b]).then_with(|| b.cmp(&a)))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let observed_threads = rayon::current_num_threads();
    drop(output);
    drop(session);
    drop(ep);
    pool.trim_idle(u64::MAX).unwrap();
    assert_eq!(books.used(Tier::Host), 0);
    println!(
        "{{\"rows\":{m},\"hidden\":{k},\"inter\":{n},\"calls\":{reps},\"rayon_width\":{observed_threads},\"workspace_bytes\":{workspace_funded},\"logical_workspace_bytes\":{logical_workspace},\"construction_workspace_bytes\":{construction_bytes},\"construction_width\":{construction_width},\"host_limit_bytes\":{host_limit},\"derived_bytes\":{derived},\"dense\":{dense},\"gated\":{gated},\"final_used\":0,\"argmax\":{argmax:?}}}"
    );
}
fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    assert_eq!(args.len(), 6, "rows k n reps threads output");
    let numbers = args[..5]
        .iter()
        .map(|s| s.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    let (m, k, n, reps, threads) = (numbers[0], numbers[1], numbers[2], numbers[3], numbers[4]);
    assert!(m > 0 && k > 0 && k % 32 == 0 && n > 0 && n % 32 == 0 && threads > 0);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(|| run(m, k, n, reps, &args[5]));
}
