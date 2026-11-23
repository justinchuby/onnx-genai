//! Synthetic public InferenceSession strict prepare + run, including output drop.
//! No model throughput claim. Scoped CPU workspace/derived cap is not a process cap.
use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView};
use onnx_runtime_ep_cpu::{
    CpuExecutionProvider, CpuMemoryResources, CpuWeightCacheLimits, DerivedWeightMemory,
};
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, Graph, Node, NodeId, TensorData, WeightRef, static_shape,
};
use onnx_runtime_loader::{Model, encode_model};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use onnx_runtime_session::{InferenceSession, Tensor};
use std::{hint::black_box, sync::Arc, time::Instant};
fn usage() -> (f64, u64) {
    let mut raw = std::mem::MaybeUninit::<libc::rusage>::uninit();
    assert_eq!(
        unsafe { libc::getrusage(libc::RUSAGE_SELF, raw.as_mut_ptr()) },
        0
    );
    let u = unsafe { raw.assume_init() };
    (
        u.ru_utime.tv_sec as f64
            + u.ru_utime.tv_usec as f64 / 1e6
            + u.ru_stime.tv_sec as f64
            + u.ru_stime.tv_usec as f64 / 1e6,
        if cfg!(target_os = "macos") {
            u.ru_maxrss as u64
        } else {
            u.ru_maxrss as u64 * 1024
        },
    )
}
fn model_block(m: usize, k: usize, n: usize) -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 24);
    graph
        .opset_imports
        .insert(onnx_runtime_ir::RUNTIME_DOMAIN.into(), 1);
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
        assert_eq!(input % 32, 0);
        let shape = vec![4, output, input / 32, 17];
        let w =
            graph.create_named_value(name, DataType::Uint8, static_shape(shape.iter().copied()));
        let mut state = 0x7a6351d9u32 ^ (output as u32).rotate_left(9) ^ input as u32;
        let mut bytes = vec![0u8; 4 * output * (input / 32) * 17];
        for block in bytes.chunks_exact_mut(17) {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            block[0] = 121 + ((state >> 28) % 3) as u8;
            for byte in &mut block[1..] {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *byte = (state >> 24) as u8;
            }
        }
        graph.set_initializer(
            w,
            WeightRef::Inline(TensorData::from_raw(DataType::Uint8, shape, bytes)),
        );
        w
    };
    let gate_mode = std::env::var("NXRT_BLOCK_GATED")
        .ok()
        .and_then(|x| x.parse::<usize>().ok())
        .unwrap_or(0);
    let gated = gate_mode != 0;
    let fusion = if gate_mode >= 2 { gate_mode - 1 } else { 0 };
    let w1 = pack("fc1", k, if fusion == 0 { n } else { 2 * n });
    let w2 = pack("fc2", n, k);
    let w3 = (gated && fusion == 0).then(|| pack("fc3", k, n));
    let mut inputs = vec![Some(a), Some(router), Some(w1), None, Some(w2)];
    if let Some(w3) = w3 {
        inputs.extend([None, Some(w3)]);
    }
    let mut node = Node::new(NodeId(0), "BlockQuantizedMoE", inputs, vec![y]);
    node.domain = onnx_runtime_ir::RUNTIME_DOMAIN.into();
    for (name, value) in [
        ("fc1_format", Attribute::String(b"mxfp4".to_vec())),
        ("fc2_format", Attribute::String(b"mxfp4".to_vec())),
        ("block_layout_version", Attribute::Int(1)),
        (
            "activation_type",
            Attribute::String(if gated {
                b"swiglu".to_vec()
            } else {
                b"identity".to_vec()
            }),
        ),
        ("swiglu_fusion", Attribute::Int(fusion as i64)),
        ("k", Attribute::Int(2)),
        ("normalize_routing_weights", Attribute::Int(1)),
    ] {
        node.attributes.insert(name.into(), value);
    }
    if gated && fusion == 0 {
        node.attributes
            .insert("fc3_format".into(), Attribute::String(b"mxfp4".to_vec()));
    }
    graph.add_input(a);
    graph.add_output(y);
    graph.insert_node(node);
    encode_model(&Model::new(&graph)).unwrap()
}
fn model_float(m: usize, k: usize, n: usize, variant: &str) -> Vec<u8> {
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
    let mut weights = |name: &str, input: usize, output: usize| {
        let w = graph.create_named_value(name, DataType::Float32, static_shape([4, output, input]));
        let mut state = 0x7a6351d9u32
            ^ (output as u32).rotate_left(9)
            ^ input as u32
            ^ name
                .bytes()
                .fold(0u32, |acc, b| acc.rotate_left(5) ^ b as u32);
        let bytes = (0..4 * output * input)
            .flat_map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (((state >> 24) as f32 - 128.) / 256.).to_le_bytes()
            })
            .collect();
        graph.set_initializer(
            w,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![4, output, input],
                bytes,
            )),
        );
        w
    };
    let (activation, fusion, separate) = match variant {
        "identity" => ("identity", 0, false),
        "swiglu0" => ("swiglu", 0, true),
        "swiglu1" => ("swiglu", 1, false),
        "swiglu2" => ("swiglu", 2, false),
        "silu3" => ("silu", 0, true),
        other => panic!("unknown probe variant {other}"),
    };
    let width = if fusion == 0 { n } else { 2 * n };
    let w1 = weights("fc1", k, width);
    let w2 = weights("fc2", n, k);
    let inputs = if separate {
        let w3 = weights("fc3", k, n);
        vec![
            Some(a),
            Some(router),
            Some(w1),
            None,
            Some(w2),
            None,
            Some(w3),
        ]
    } else {
        vec![Some(a), Some(router), Some(w1), None, Some(w2)]
    };
    let mut node = Node::new(NodeId(0), "MoE", inputs, vec![y]);
    node.domain = "com.microsoft".into();
    for (name, value) in [
        (
            "activation_type",
            Attribute::String(activation.as_bytes().to_vec()),
        ),
        ("swiglu_fusion", Attribute::Int(fusion)),
        ("k", Attribute::Int(2)),
        ("normalize_routing_weights", Attribute::Int(1)),
    ] {
        node.attributes.insert(name.into(), value);
    }
    graph.add_input(a);
    graph.add_output(y);
    graph.insert_node(node);
    encode_model(&Model::new(&graph)).unwrap()
}
fn model(m: usize, k: usize, n: usize) -> Vec<u8> {
    if std::env::var("NXRT_BANK_OP").ok().as_deref() == Some("float") {
        let mode = std::env::var("NXRT_BLOCK_GATED").unwrap_or_default();
        let variant = match mode.as_str() {
            "1" => "swiglu0",
            "2" => "swiglu1",
            "3" => "swiglu2",
            _ => "identity",
        };
        model_float(m, k, n, variant)
    } else {
        model_block(m, k, n)
    }
}
fn execute(session: &mut InferenceSession, input: &Tensor) {
    session
        .prepare_with_device_bindings(&[("a", input)], &mut [])
        .unwrap();
    let output = session.run(black_box(&[("a", input)])).unwrap();
    black_box(&output);
    // Output owners are dropped here, before the batch stop time.
}


use onnx_runtime_ep_cpu::ExpertBankProbeArm;
use std::sync::atomic::{AtomicUsize,Ordering};
use std::io::Write;
fn profile(label: &str, path: &str) {
    let (m,k,n)=(8,128,256);
    let books=Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(DeviceKey::HOST,0,64*1024*1024,0)));
    let manager=ProcessMemoryManager::new().unwrap();
    let pool=HostWorkspacePool::governed(&manager,books.clone()).unwrap();
    let resources=CpuMemoryResources::new(DerivedWeightMemory::governed(books.as_ref(),HolderId::new(13)).unwrap(),CpuWeightCacheLimits{dense_bytes:0,expert_bytes:0}).unwrap().with_workspace_pool(pool.clone()).unwrap();
    let kind=match label {"before"|"null"=>ExpertBankProbeArm::Before,"banked"=>ExpertBankProbeArm::Banked,_=>panic!("unknown arm")};
    let creates=Arc::new(AtomicUsize::new(0));
    let ep=Arc::new(CpuExecutionProvider::initialized_with_expert_bank_probe(resources,kind,creates.clone()).unwrap());
    let bytes=model(m,k,n);
    let input=Tensor::from_f32(&[m,k],&(0..m*k).map(|i|(i%31) as f32/1000.-0.015).collect::<Vec<_>>()).unwrap();
    let mut session=InferenceSession::builder().model_bytes(&bytes).option("optimization","none").execution_provider(ep.clone()).build().unwrap();
    let requirement=session.prepare_with_device_bindings(&[("a",&input)],&mut[]).unwrap().bytes;
    let warm=usage().0;while usage().0-warm<0.5 {execute(&mut session,&input);}
    let funded=books.used(Tier::Host);assert_eq!(funded,requirement);assert_eq!(manager.process_used(Tier::Host),requirement);
    let count=creates.load(Ordering::Relaxed);assert!(count>0);
    session.prepare_with_device_bindings(&[("a",&input)],&mut[]).unwrap();
    let out=session.run(&[("a",&input)]).unwrap();assert_eq!(out[0].shape,[m,k]);assert!(out[0].to_vec_f32().iter().all(|x|x.is_finite()));
    let expected=out[0].as_bytes().to_vec();std::fs::write(path,&expected).unwrap();drop(out);
    println!("READY {{\"arm\":\"{label}\",\"workspace_bytes\":{requirement},\"factory_creates\":{count}}}");std::io::stdout().flush().unwrap();
    // Sampling window is wall-delimited, not a throughput measurement.
    let start=Instant::now();let mut calls=0;
    while start.elapsed().as_secs_f64()<15. {execute(&mut session,&input);calls+=1;}
    assert_eq!(creates.load(Ordering::Relaxed),count);assert_eq!(books.used(Tier::Host),funded);assert_eq!(manager.process_used(Tier::Host),requirement);
    session.prepare_with_device_bindings(&[("a",&input)],&mut[]).unwrap();let out=session.run(&[("a",&input)]).unwrap();assert_eq!(out[0].as_bytes(),expected);drop(out);
    drop(session);drop(ep);pool.trim_idle(u64::MAX).unwrap();assert_eq!(books.used(Tier::Host),0);
    println!("DONE {{\"arm\":\"{label}\",\"calls_in_sampled_window\":{calls}}}");
}
fn main(){
    let args=std::env::args().skip(1).collect::<Vec<_>>();assert_eq!(args.len(),2);
    rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap().install(||profile(&args[0],&args[1]));
}
