use super::*;
use crate::Tensor;
use onnx_runtime_ep_cpu::DerivedWeightMemory;
use onnx_runtime_ir::{DataType, Dim, Node, NodeId, TensorData, WeightRef, static_shape};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    mpsc,
};
use std::time::Duration;

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);
struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new() -> Self {
        let id = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("nxrt-cpu-worker-{}-{id}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        Self(root)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn resources(limit: u64) -> (CpuMemoryResources, Arc<LedgerGovernor>) {
    let books = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
        DeviceKey::HOST,
        0,
        limit,
        0,
    )));
    let manager = ProcessMemoryManager::new().unwrap();
    let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
    let resources = CpuMemoryResources::new(
        DerivedWeightMemory::governed(books.as_ref(), HolderId::new(912)).unwrap(),
        Default::default(),
    )
    .unwrap()
    .with_workspace_pool(pool)
    .unwrap();
    (resources, books)
}
fn model(root: &Fixture, inline: bool) -> InferenceSession {
    let path = root.0.join("weights.bin");
    let bytes = [2_f32, 0.0, 0.0, 3.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    std::fs::write(&path, &bytes).unwrap();
    let mut store = WeightStore::new();
    store.map_external(&path).unwrap();
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let sequence = graph.create_symbol(Some("sequence".into()));
    let shape = vec![Dim::Static(1), Dim::Symbolic(sequence), Dim::Static(2)];
    let input = graph.create_named_value("input", DataType::Float32, shape.clone());
    let past = graph.create_named_value("past", DataType::Float32, shape.clone());
    let weight = graph.create_named_value("weight", DataType::Float32, static_shape([2, 2]));
    let projected = graph.create_named_value("projected", DataType::Float32, shape.clone());
    let output = graph.create_named_value("output", DataType::Float32, shape);
    graph.add_input(input);
    graph.add_input(past);
    graph.add_output(output);
    graph.set_initializer(
        weight,
        if inline {
            WeightRef::Inline(TensorData::from_raw(DataType::Float32, vec![2, 2], bytes))
        } else {
            WeightRef::External {
                path,
                offset: 0,
                length: 16,
                dtype: DataType::Float32,
                dims: vec![2, 2],
            }
        },
    );
    graph.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(input), Some(weight)],
        vec![projected],
    ));
    graph.insert_node(Node::new(
        NodeId(1),
        "Add",
        vec![Some(projected), Some(past)],
        vec![output],
    ));
    let provider =
        Arc::new(CpuExecutionProvider::initialized_with_resources(Default::default()).unwrap());
    InferenceSession::from_graph_with_provider(graph, Arc::new(store), &root.0, provider, None)
        .unwrap()
}
fn run(session: &mut InferenceSession, input: &[f32], past: &[f32]) -> Tensor {
    let shape = [1, input.len() / 2, 2];
    let x = Tensor::from_f32(&shape, input).unwrap();
    let state = Tensor::from_f32(&shape, past).unwrap();
    session
        .run(&[("input", &x), ("past", &state)])
        .unwrap()
        .remove(0)
}
#[test]
fn mapped_cpu_workers_share_backing_but_not_bindings_shapes_or_outputs() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<CpuWorkerPlan>();
    let root = Fixture::new();
    let mut source = model(&root, false);
    let plan = source.cpu_worker_plan().unwrap();
    assert!(Arc::ptr_eq(
        &plan.weights,
        &source.exec.cpu_worker_parts().unwrap().1
    ));
    let source_id = source.executor_instance_id();
    let (resources, books) = resources(1024 * 1024);
    let mut first = plan.build_worker(resources.clone()).unwrap();
    let mut second = plan.build_worker(resources.clone()).unwrap();
    assert_ne!(source_id, first.executor_instance_id());
    assert_ne!(first.executor_instance_id(), second.executor_instance_id());
    assert!(Arc::ptr_eq(
        &plan.weights,
        &first.exec.cpu_worker_parts().unwrap().1
    ));
    let original = run(&mut first, &[1.0, 2.0], &[0.5, -1.0]);
    assert_eq!(original.shape, &[1, 1, 2]);
    assert_eq!(original.try_as_slice_f32().unwrap(), &[2.5, 5.0]);
    let other = run(
        &mut second,
        &[3.0, 4.0, 5.0, 6.0],
        &[10.0, 20.0, 30.0, 40.0],
    );
    assert_eq!(other.shape, &[1, 2, 2]);
    assert_eq!(other.try_as_slice_f32().unwrap(), &[16.0, 32.0, 40.0, 58.0]);
    assert_eq!(
        run(&mut first, &[1.0, 2.0], &[2.5, 5.0])
            .try_as_slice_f32()
            .unwrap(),
        &[4.5, 11.0]
    );
    assert_eq!(original.try_as_slice_f32().unwrap(), &[2.5, 5.0]);
    assert_eq!(
        run(&mut source, &[2.0, 3.0], &[0.0, 0.0])
            .try_as_slice_f32()
            .unwrap(),
        &[4.0, 9.0]
    );
    let backing = Arc::downgrade(&plan.weights);
    drop(source);
    drop(plan);
    assert!(backing.upgrade().is_some());
    assert_eq!(
        run(&mut second, &[1.0, 1.0], &[0.0, 0.0])
            .try_as_slice_f32()
            .unwrap(),
        &[2.0, 3.0]
    );
    drop(first);
    drop(second);
    drop(resources);
    assert!(backing.upgrade().is_none());
    assert_eq!(books.used(Tier::Host), 0);
}
#[test]
fn mapped_cpu_workers_construct_and_execute_on_independent_owner_threads() {
    let root = Fixture::new();
    let source = model(&root, false);
    let plan = source.cpu_worker_plan().unwrap();
    let (resources, _books) = resources(1024 * 1024);
    let (ready_tx, ready_rx) = mpsc::channel();
    let (left_tx, left_rx) = mpsc::channel();
    let (right_tx, right_rx) = mpsc::channel();
    std::thread::scope(|scope| {
        let left_plan = &plan;
        let left_resources = resources.clone();
        let left_ready = ready_tx.clone();
        let left = scope.spawn(move || {
            let mut worker = left_plan.build_worker(left_resources).unwrap();
            left_ready.send(()).unwrap();
            left_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            for _ in 0..8 {
                assert_eq!(
                    run(&mut worker, &[1.0, 2.0], &[0.0, 0.0])
                        .try_as_slice_f32()
                        .unwrap(),
                    &[2.0, 6.0]
                );
            }
        });
        let right_plan = &plan;
        let right_resources = resources.clone();
        let right_ready = ready_tx.clone();
        let right = scope.spawn(move || {
            let mut worker = right_plan.build_worker(right_resources).unwrap();
            right_ready.send(()).unwrap();
            right_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            for _ in 0..8 {
                assert_eq!(
                    run(&mut worker, &[3.0, 4.0], &[10.0, 20.0])
                        .try_as_slice_f32()
                        .unwrap(),
                    &[16.0, 32.0]
                );
            }
        });
        ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        left_tx.send(()).unwrap();
        right_tx.send(()).unwrap();
        left.join().unwrap();
        right.join().unwrap();
    });
}
#[test]
fn inline_profile_refusal_preserves_source_execution() {
    let root = Fixture::new();
    let mut source = model(&root, true);
    assert!(matches!(
        source.cpu_worker_plan(),
        Err(SessionError::UnsupportedCpuWorkerPlan { .. })
    ));
    assert_eq!(
        run(&mut source, &[1.0, 2.0], &[0.0, 0.0])
            .try_as_slice_f32()
            .unwrap(),
        &[2.0, 6.0]
    );
}
#[test]
fn ungoverned_worker_refusal_is_retryable_and_does_not_retain_backing() {
    let root = Fixture::new();
    let mut source = model(&root, false);
    let plan = source.cpu_worker_plan().unwrap();
    let before = Arc::strong_count(&plan.weights);
    let error = plan
        .build_worker(Default::default())
        .err()
        .expect("missing authority must fail");
    assert!(error.to_string().contains("ungoverned"));
    assert_eq!(Arc::strong_count(&plan.weights), before);
    assert_eq!(
        run(&mut source, &[1.0, 2.0], &[0.0, 0.0])
            .try_as_slice_f32()
            .unwrap(),
        &[2.0, 6.0]
    );
    let (resources, books) = resources(1024 * 1024);
    let mut worker = plan.build_worker(resources.clone()).unwrap();
    assert_eq!(
        run(&mut worker, &[2.0, 3.0], &[0.0, 0.0])
            .try_as_slice_f32()
            .unwrap(),
        &[4.0, 9.0]
    );
    drop(worker);
    drop(resources);
    assert_eq!(books.used(Tier::Host), 0);
}

#[test]
fn worker_workspace_denial_releases_claims_and_keeps_plan_retryable() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("input", DataType::Float16, static_shape([1, 4]));
    let output = graph.create_named_value("output", DataType::Float16, static_shape([1, 4]));
    graph.add_input(input);
    graph.add_output(output);
    graph.insert_node(Node::new(
        NodeId(0),
        "Softmax",
        vec![Some(input)],
        vec![output],
    ));
    let source = InferenceSession::from_graph(graph).unwrap();
    let plan = source.cpu_worker_plan().unwrap();
    let before = Arc::strong_count(&plan.weights);
    let (refused, books) = resources(0);
    // Half softmax requires the declared f32 conversion workspace. This tests
    // the actual workspace authority, not the ordinary unmetered allocator.
    let mut worker = plan.build_worker(refused.clone()).unwrap();
    let input = Tensor::from_raw(DataType::Float16, vec![1, 4], &[0; 8]).unwrap();
    let error = worker
        .run(&[("input", &input)])
        .expect_err("zero-byte authority must refuse real softmax workspace");
    assert!(error.to_string().contains("workspace"), "{error}");
    drop(worker);
    drop(refused);
    assert_eq!(books.used(Tier::Host), 0);
    assert_eq!(Arc::strong_count(&plan.weights), before);
    let (accepted, books) = resources(4096);
    let mut worker = plan.build_worker(accepted.clone()).unwrap();
    let output = worker.run(&[("input", &input)]).unwrap();
    assert_eq!(output[0].try_as_slice_u16().unwrap(), &[0x3400; 4]);
    drop(output);
    drop(worker);
    drop(accepted);
    assert_eq!(books.used(Tier::Host), 0);
}
