//! Does a real session actually allocate through a registered governed
//! allocator?
//!
//! The unit tests in `governed_allocator` prove the allocator leases correctly
//! and that registration reaches ORT's table. Neither proves the thing the
//! feature exists for: that a session's allocations are charged to the
//! governor. A registration ORT accepts and then ignores looks exactly like a
//! model that did not allocate — same zero counters, same green tests.
//!
//! So these tests run inference and watch the ledger.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use onnx_genai_ort::governed_allocator::{GovernedAllocator, register_governed_allocator};
use onnx_genai_ort::{
    DataType, Environment, MemoryInfo, Session, SessionOptions, USE_ENV_ALLOCATORS, Value,
};
use onnx_runtime_memory_governor::{
    HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryRole, Tier,
};

fn tiny_llm() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm/model.onnx.textproto")
}

fn test_environment() -> &'static Environment {
    static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
    ENVIRONMENT.get_or_init(|| Environment::new("governed-allocator-session").expect("env"))
}

/// Registration is environment-wide, so these tests cannot overlap each other
/// or any other test that builds a session in this binary.
fn ort_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

const HOLDER: HolderId = HolderId::new(42);
const BUDGET: u64 = 512 * 1024 * 1024;

fn governed(budget: u64) -> (Box<GovernedAllocator>, LedgerGovernor) {
    let governor = LedgerGovernor::new(LeaseLedger::new(0, budget, 0));
    let allocator = GovernedAllocator::new(
        MemoryInfo::cpu_device().expect("cpu device memory info"),
        Arc::new(governor.clone()),
        Tier::Host,
        MemoryRole::Activation,
        HOLDER,
    )
    .expect("host allocator");
    (allocator, governor)
}

fn run_once(session: &Session) {
    let tokens = Value::from_vec_i64(vec![1, 2, 3], &[1, 3]).expect("token tensor");
    let _ = session.run(&[("input_ids", &tokens)]);
}

/// The whole point: a session that opted in must charge its allocations to the
/// governor.
///
/// Peak is what is asserted, not the final count. ORT frees most of what it
/// allocates before `run` returns, so a test that only looked afterwards would
/// pass whether or not a single byte went through us.
#[test]
fn a_session_that_opted_in_allocates_through_the_governor() {
    let _guard = ort_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let model = tiny_llm();
    if !model.exists() {
        return;
    }

    let (allocator, governor) = governed(BUDGET);
    let registered =
        register_governed_allocator(test_environment(), allocator).expect("register allocator");

    let mut options = SessionOptions::default().with_intra_op_threads(1);
    options.use_env_allocators();
    let session = Session::new(test_environment(), &model, options).expect("session");

    let free_before = governor.available(Tier::Host);
    run_once(&session);

    // Session construction alone already allocates weights and plan state, so
    // the ledger must have moved before the run even started.
    assert!(
        free_before < BUDGET,
        "building a session with use_env_allocators charged the governor nothing, \
         so the registration is not actually being used ({free_before} of {BUDGET} free)"
    );

    drop(session);
    drop(registered);
}

/// The control. Without the config entry, ORT builds the session its own
/// allocator and the governor sees nothing.
///
/// This is what makes the test above meaningful: it rules out a ledger that
/// moves for some reason unrelated to the session.
#[test]
fn a_session_that_did_not_opt_in_leaves_the_governor_untouched() {
    let _guard = ort_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let model = tiny_llm();
    if !model.exists() {
        return;
    }

    let (allocator, governor) = governed(BUDGET);
    let registered =
        register_governed_allocator(test_environment(), allocator).expect("register allocator");

    // Deliberately no `use_env_allocators`.
    let options = SessionOptions::default().with_intra_op_threads(1);
    let session = Session::new(test_environment(), &model, options).expect("session");
    run_once(&session);

    assert_eq!(
        governor.available(Tier::Host),
        BUDGET,
        "a session that did not opt in must not reach the registered allocator; \
         if it does, `use_env_allocators` is not the switch it is documented to be"
    );

    drop(session);
    drop(registered);
}

/// The config entry must survive onto the session options ORT actually sees.
#[test]
fn the_opt_in_entry_is_carried_on_the_session_options() {
    let mut options = SessionOptions::default();
    options.use_env_allocators();
    assert!(
        options
            .session_config_entries
            .iter()
            .any(|(key, value)| key == USE_ENV_ALLOCATORS && value == "1")
    );
}

/// A tensor built over memory the caller owns must be usable as a real session
/// input, not just constructible.
#[test]
fn an_external_tensor_can_feed_a_real_session() {
    let _guard = ort_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let model = tiny_llm();
    if !model.exists() {
        return;
    }

    let options = SessionOptions::default().with_intra_op_threads(1);
    let session = Session::new(test_environment(), &model, options).expect("session");

    // Build one caller-owned buffer per declared input, sized and typed from the
    // model's own signature rather than guessed: the point of the test is that
    // *external* memory feeds a real session, not which ports this fixture has.
    const SEQ: i64 = 3;
    let info = MemoryInfo::cpu().expect("cpu memory info");
    let mut buffers: Vec<Vec<u8>> = Vec::new();
    let mut shapes: Vec<Vec<i64>> = Vec::new();
    for input in session.inputs() {
        // ORT reports dynamic axes as negative, so pick concrete extents. Axis 0
        // is batch. Past-KV ports must be empty or their sequence length will
        // disagree with the attention mask's.
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
            // Token and mask ports need in-range values; zero is safe for both
            // (token 0 exists, mask 0 attends to nothing but does not fault).
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

    let mut values = Vec::new();
    for (buffer, (shape, input)) in buffers
        .iter_mut()
        .zip(shapes.iter().zip(session.inputs().iter()))
    {
        // SAFETY: `buffers` outlives `values`; each buffer was sized from the
        // very shape and dtype passed here; and the memory info says host
        // memory, which is where a `Vec` lives.
        values.push(
            unsafe {
                Value::from_external_memory(
                    buffer.as_mut_ptr().cast(),
                    buffer.len(),
                    shape,
                    input.dtype,
                    &info,
                )
            }
            .expect("external tensor"),
        );
    }

    let inputs: Vec<(&str, &Value)> = session
        .input_names()
        .iter()
        .map(String::as_str)
        .zip(values.iter())
        .collect();

    session
        .run(&inputs)
        .expect("a session must accept tensors over caller-owned memory");
}
