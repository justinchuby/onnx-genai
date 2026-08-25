#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use onnx_genai_ort::{
    DataType, Environment, ManagedCudaAllocatorConfig, Session, SessionOptions, Value,
    available_execution_providers, ep_selection,
};
use onnx_runtime_memory_governor::{
    DeviceKey, LeaseLedger, LedgerGovernor, ProcessMemoryLimits, ProcessMemoryManager,
};

fn tiny_llm() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm/model.onnx.textproto")
}

fn ort_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn cuda_ready() -> bool {
    available_execution_providers()
        .ok()
        .is_some_and(|providers| {
            providers
                .iter()
                .any(|provider| provider.eq_ignore_ascii_case("CUDAExecutionProvider"))
        })
        && onnx_genai_ort::cuda_rt::device_memory_info(0).is_ok()
}

fn managed_cuda_config_with_manager(
    device_id: i32,
    manager: ProcessMemoryManager,
) -> ManagedCudaAllocatorConfig {
    let governor = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
        DeviceKey::device(device_id as u32),
        1 << 33,
        0,
        0,
    )));
    ManagedCudaAllocatorConfig::new(device_id, manager, governor)
        .expect("managed CUDA allocator config")
}

fn model_inputs(session: &Session) -> (Vec<Vec<u8>>, Vec<Vec<i64>>) {
    const SEQ: i64 = 3;
    let mut buffers = Vec::new();
    let mut shapes = Vec::new();
    for input in session.inputs() {
        let is_past = input.name.contains("past");
        let shape: Vec<i64> = input
            .shape
            .iter()
            .enumerate()
            .map(|(axis, &dim)| match (dim < 0, axis, is_past) {
                (false, _, _) => dim,
                (true, 0, _) => 1,
                (true, _, true) => 0,
                (true, _, false) => SEQ,
            })
            .collect();
        let elements: usize = shape.iter().map(|&d| d as usize).product();
        let mut bytes = vec![0u8; elements * input.dtype.size_of()];
        if input.dtype == DataType::Int64 {
            for (index, chunk) in bytes.chunks_exact_mut(8).enumerate() {
                let value = if input.name == "attention_mask" {
                    1i64
                } else {
                    index as i64 % 2
                };
                chunk.copy_from_slice(&value.to_le_bytes());
            }
        }
        buffers.push(bytes);
        shapes.push(shape);
    }
    (buffers, shapes)
}

fn run_once(session: &Session) -> Result<(), onnx_genai_ort::OrtError> {
    let (buffers, shapes) = model_inputs(session);
    let mut values = Vec::new();
    for ((buffer, shape), input) in buffers
        .iter()
        .zip(shapes.iter())
        .zip(session.inputs().iter())
    {
        values.push(Value::from_raw_bytes(buffer.clone(), shape, input.dtype)?);
    }
    let inputs: Vec<(&str, &Value)> = session
        .input_names()
        .iter()
        .map(String::as_str)
        .zip(values.iter())
        .collect();
    session.run(&inputs).map(|_| ())
}

#[test]
fn a_cuda_session_can_route_internal_and_external_allocations_through_the_managed_bridge() {
    let _guard = ort_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let model = tiny_llm();
    if !model.exists() || !cuda_ready() {
        return;
    }

    let manager = ProcessMemoryManager::new().expect("process memory manager");
    let env = Environment::new("managed-cuda-allocator-session").expect("env");
    let mut options = SessionOptions::with_execution_provider(ep_selection("cuda"));
    options.use_managed_cuda_allocator(managed_cuda_config_with_manager(0, manager.clone()));
    options = options.with_intra_op_threads(1);

    let session = Session::new(&env, &model, options).expect("CUDA session");
    let build_stats = env
        .managed_cuda_allocator_stats(0)
        .expect("managed CUDA allocator stats");
    assert!(
        build_stats.total_allocations > 0 && build_stats.reserve_allocations > 0,
        "session construction did not allocate through the registered managed CUDA bridge: \
         {build_stats:?}"
    );

    let allocator = session
        .device_allocator()
        .expect("session allocator query")
        .expect("CUDA device allocator");
    let owner = Value::empty_in(&[1, 1], DataType::Float32, &allocator)
        .expect("device allocation through managed bridge");
    let after_external = env
        .managed_cuda_allocator_stats(0)
        .expect("managed CUDA allocator stats");
    assert!(
        after_external.total_allocations > build_stats.total_allocations,
        "session.device_allocator() did not allocate through the registered managed CUDA bridge: \
         before {build_stats:?}, after {after_external:?}"
    );

    run_once(&session).expect("the CUDA session must run");
    let after_run = env
        .managed_cuda_allocator_stats(0)
        .expect("managed CUDA allocator stats");
    assert!(
        after_run.total_allocations > after_external.total_allocations,
        "inference did not allocate through the registered managed CUDA bridge: \
         before {after_external:?}, after {after_run:?}"
    );

    drop(owner);
    let after_free = env
        .managed_cuda_allocator_stats(0)
        .expect("managed CUDA allocator stats");
    assert!(
        after_free.deferred_release_enqueue_failures == 0
            && after_free.deferred_release_quarantined == 0,
        "a quiescent free must reclaim without quarantine: {after_free:?}"
    );
    let plateau = manager.snapshot().expect("snapshot after first free");
    assert_eq!(plateau.mapped_bytes, 0);
    for _ in 0..16 {
        drop(
            Value::empty_in(&[1, 1], DataType::Float32, &allocator)
                .expect("repeated managed allocation"),
        );
        let snapshot = manager.snapshot().expect("snapshot after repeated free");
        assert_eq!(snapshot.mapped_bytes, 0);
        assert_eq!(
            snapshot.charged_bytes, plateau.charged_bytes,
            "repeated allocate/free must reuse or reclaim physical capacity rather than \
             monotonically charging it"
        );
    }
    drop(env);
    run_once(&session).expect("the CUDA session must run after its Environment wrapper is dropped");
    // The allocator allocates through the session's EP, so it has to be
    // released first; the borrow in `Allocator<'_>` is what makes that a
    // compile error to get wrong rather than a latent use-after-free.
    drop(allocator);
    drop(session);
}

#[test]
fn registration_requires_the_exact_process_memory_manager() {
    let _guard = ort_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let model = tiny_llm();
    if !model.exists() || !cuda_ready() {
        return;
    }

    let env = Environment::new("managed-cuda-manager-identity").expect("env");
    let first_manager = ProcessMemoryManager::new().expect("first manager");
    let first = managed_cuda_config_with_manager(0, first_manager);
    let mut options = SessionOptions::with_execution_provider(ep_selection("cuda"));
    options.use_managed_cuda_allocator(first);
    let session = Session::new(&env, &model, options).expect("first CUDA session");

    let second_manager = ProcessMemoryManager::new().expect("second manager");
    let second = managed_cuda_config_with_manager(0, second_manager);
    let mut incompatible = SessionOptions::with_execution_provider(ep_selection("cuda"));
    incompatible.use_managed_cuda_allocator(second);
    let error = match Session::new(&env, &model, incompatible) {
        Ok(_) => panic!("same device and authority shape from another manager must be rejected"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("different managed CUDA allocator")
    );
    drop(session);
}

#[test]
fn failed_registration_rolls_back_manager_records() {
    let _guard = ort_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let model = tiny_llm();
    if !model.exists() || !cuda_ready() {
        return;
    }

    let manager = ProcessMemoryManager::with_limits(ProcessMemoryLimits {
        device_bytes: 0,
        host_bytes: u64::MAX,
        disk_bytes: u64::MAX,
    })
    .expect("limited manager");
    let config = managed_cuda_config_with_manager(0, manager.clone());
    let env = Environment::new("managed-cuda-registration-rollback").expect("env");
    let mut options = SessionOptions::with_execution_provider(ep_selection("cuda"));
    options.use_managed_cuda_allocator(config);
    assert!(
        Session::new(&env, &model, options).is_err(),
        "delegating a nonzero CUDA authority into a zero-byte process limit must fail"
    );

    let snapshot = manager.snapshot().expect("manager snapshot after rollback");
    assert_eq!(snapshot.authority_count, 0);
    assert!(snapshot.mechanism_snapshots.is_empty());
}

#[test]
fn device_loss_listener_remains_registered_strongly() {
    let _guard = ort_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let model = tiny_llm();
    if !model.exists() || !cuda_ready() {
        return;
    }

    let manager = ProcessMemoryManager::new().expect("manager");
    let config = managed_cuda_config_with_manager(0, manager.clone());
    let env = Environment::new("managed-cuda-listener-retention").expect("env");
    let mut options = SessionOptions::with_execution_provider(ep_selection("cuda"));
    options.use_managed_cuda_allocator(config);
    let session = Session::new(&env, &model, options).expect("CUDA session");

    manager
        .invalidate_device(DeviceKey::device(0), "test device loss")
        .expect("invalidate device");
    assert!(
        env.managed_cuda_allocator_stats(0)
            .expect("allocator stats")
            .device_lost
    );
    drop(session);
}
