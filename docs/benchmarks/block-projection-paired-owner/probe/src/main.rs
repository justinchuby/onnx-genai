//! Diagnostic wall blocks only; parent wait4 records total process CPU.
use onnx_runtime_ep_api::{ExecutionProvider, WorkspaceRequirement};
use onnx_runtime_ep_cpu::kernels::block_quantized_matmul::{
    diagnostic_factory_stats, diagnostic_select_frozen,
};
use onnx_runtime_ep_cpu::{
    CpuExecutionProvider, CpuMemoryResources, CpuWeightCacheLimits, DerivedWeightMemory,
};
use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, static_shape};
use onnx_runtime_loader::{Model, encode_model};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use onnx_runtime_session::{InferenceSession, Tensor};
use std::{hint::black_box, sync::Arc, time::Instant};
const M: usize = 8;
const K: usize = 128;
const N: usize = 256;
const PEAK: usize = 16384;
fn model() -> Vec<u8> {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 24);
    g.opset_imports.insert("pkg.nxrt".into(), 1);
    let a = g.create_named_value("a", DataType::Float32, static_shape([M, K]));
    let b = g.create_named_value("b", DataType::Uint8, static_shape([N, K / 32, 17]));
    let y = g.create_named_value("y", DataType::Float32, static_shape([M, N]));
    let mut node = Node::new(
        NodeId(0),
        "BlockQuantizedMatMul",
        vec![Some(a), Some(b)],
        vec![y],
    );
    node.domain = "pkg.nxrt".into();
    for (name, value) in [
        ("K", Attribute::Int(K as i64)),
        ("N", Attribute::Int(N as i64)),
        ("format", Attribute::String(b"mxfp4".to_vec())),
        ("block_layout_version", Attribute::Int(1)),
    ] {
        node.attributes.insert(name.into(), value);
    }
    g.add_input(a);
    g.add_input(b);
    g.add_output(y);
    g.insert_node(node);
    encode_model(&Model::new(&g)).unwrap()
}
fn execute(s: &mut InferenceSession, a: &Tensor, b: &Tensor) {
    let inputs = [("a", a), ("b", b)];
    s.prepare_with_device_bindings(&inputs, &mut [])
        .unwrap()
        .require_complete()
        .unwrap();
    let result = s.run(black_box(&inputs)).unwrap();
    black_box(&result);
}
fn run(repetition: usize, calls: usize, path: &str) {
    let ledger = LeaseLedger::new_for_device(DeviceKey::HOST, 0, PEAK as u64, 0);
    let books = Arc::new(LedgerGovernor::new(ledger.clone()));
    let manager = ProcessMemoryManager::new().unwrap();
    let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
    let memory = DerivedWeightMemory::governed(books.as_ref(), HolderId::new(13)).unwrap();
    let resources = CpuMemoryResources::new(
        memory,
        CpuWeightCacheLimits {
            dense_bytes: 0,
            expert_bytes: 0,
        },
    )
    .unwrap()
    .with_workspace_pool(pool.clone())
    .unwrap();
    let ep = Arc::new(CpuExecutionProvider::initialized_with_resources(resources).unwrap());
    let loan = ep
        .allocate_workspace(PEAK, 4, WorkspaceRequirement::NONE.role())
        .unwrap();
    ep.deallocate_workspace(loan).unwrap();
    let anchor = manager
        .snapshot()
        .unwrap()
        .allocations
        .into_iter()
        .next()
        .unwrap()
        .identity;
    let a = Tensor::from_f32(
        &[M, K],
        &(0..M * K)
            .map(|i| (i % 31) as f32 / 1000. - 0.015)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut state = 0x7a6351d9u32 ^ (N as u32).rotate_left(9) ^ K as u32;
    let mut packed = vec![0u8; N * (K / 32) * 17];
    for block in packed.chunks_exact_mut(17) {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        block[0] = 121 + ((state >> 28) % 3) as u8;
        for byte in &mut block[1..] {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *byte = (state >> 24) as u8;
        }
    }
    let b = Tensor::from_raw(DataType::Uint8, vec![N, K / 32, 17], &packed).unwrap();
    let bytes = model();
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let four = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let mut expected: Option<Vec<u8>> = None;
    let mut records = Vec::new();
    let cases = if repetition % 2 == 0 { [1, 4] } else { [4, 1] };
    for construction in cases {
        let constructor = if construction == 1 { &one } else { &four };
        // ABBA BAAB: each arm has two independent recurring positions for A/A.
        let normal = if repetition % 2 == 0 {
            [true, false, false, true, false, true, true, false]
        } else {
            [false, true, true, false, true, false, false, true]
        };
        let mut windows = normal
            .into_iter()
            .map(|f| (f, calls, "work"))
            .collect::<Vec<_>>();
        windows.extend([
            (true, 0, "empty"),
            (false, 0, "empty"),
            (false, calls * 2, "double"),
            (true, calls * 2, "double"),
        ]);
        for (index, (frozen, count, kind)) in windows.into_iter().enumerate() {
            let (mut session, before, after) = constructor.install(|| {
                diagnostic_select_frozen(frozen);
                let before = diagnostic_factory_stats();
                let mut session = InferenceSession::builder()
                    .model_bytes(&bytes)
                    .option("optimization", "none")
                    .execution_provider(ep.clone())
                    .build()
                    .unwrap();
                session
                    .prepare_with_device_bindings(&[("a", &a), ("b", &b)], &mut [])
                    .unwrap()
                    .require_complete()
                    .unwrap();
                let after = diagnostic_factory_stats();
                assert!(
                    if frozen {
                        after.0 > before.0 && after.1 == before.1
                    } else {
                        after.1 > before.1 && after.0 == before.0
                    },
                    "constructor must select only the requested variant"
                );
                (session, before, after)
            });
            diagnostic_select_frozen(frozen);
            for _ in 0..5 {
                execute(&mut session, &a, &b);
            }
            let logical = session
                .prepare_with_device_bindings(&[("a", &a), ("b", &b)], &mut [])
                .unwrap()
                .require_complete()
                .unwrap()
                .step()
                .bytes;
            assert_eq!(
                logical,
                if frozen && construction == 4 {
                    16384
                } else {
                    10240
                }
            );
            let snapshot = manager.snapshot().unwrap();
            assert_eq!(snapshot.allocations.len(), 1);
            assert_eq!(snapshot.allocations[0].identity, anchor);
            assert_eq!(manager.process_used(Tier::Host), PEAK as u64);
            assert_eq!(books.used(Tier::Host), PEAK as u64);
            assert_eq!(ledger.available(Tier::Host), 0);
            let factory_before = diagnostic_factory_stats();
            let start = Instant::now();
            for _ in 0..count {
                execute(&mut session, &a, &b);
            }
            let wall = start.elapsed().as_secs_f64();
            assert_eq!(
                factory_before,
                diagnostic_factory_stats(),
                "no factory selection or kernel creation in the timed block"
            );
            let output = session.run(&[("a", &a), ("b", &b)]).unwrap();
            let output_bytes = output[0].as_bytes().to_vec();
            assert!(output[0].to_vec_f32().iter().all(|v| v.is_finite()));
            if let Some(ref value) = expected {
                assert_eq!(value, &output_bytes);
            } else {
                expected = Some(output_bytes);
            }
            drop(output);
            let (trace, collector) = onnx_runtime_tracer::TraceContext::in_memory();
            session.set_trace_context(trace);
            execute(&mut session, &a, &b);
            assert!(
                collector
                    .to_chrome_json()
                    .contains("block_matmul.streamed_rows")
            );
            drop(session);
            let snapshot = manager.snapshot().unwrap();
            assert_eq!(snapshot.allocations.len(), 1);
            assert_eq!(snapshot.allocations[0].identity, anchor);
            assert_eq!(books.used(Tier::Host), PEAK as u64);
            records.push(serde_json::json!({"construction":construction,"execution":1,"frozen":frozen,"index":index,"calls":count,"kind":kind,"wall_s":wall,"logical_bytes":logical,"physical_bytes":PEAK,"factory_before":factory_before,"constructor_stats_before":before,"constructor_stats_after":after}));
        }
    }
    std::fs::write(path, expected.unwrap()).unwrap();
    drop(ep);
    pool.trim_idle(u64::MAX).unwrap();
    assert_eq!(books.used(Tier::Host), 0);
    println!(
        "{}",
        serde_json::json!({"repetition":repetition,"anchor":format!("{anchor:?}"),"same_input_owner":true,"same_workspace_owner":true,"final_used":0,"records":records})
    );
}
fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    assert_eq!(args.len(), 3);
    let rep = args[0].parse().unwrap();
    let calls = args[1].parse().unwrap();
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| run(rep, calls, &args[2]));
}
