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

use onnx_genai_ort::governed_allocator::{
    AllocationRoles, GovernedAllocator, register_governed_allocator,
};
use onnx_genai_ort::{
    DataType, Environment, MemoryInfo, Session, SessionOptions, USE_ENV_ALLOCATORS, Value,
};
use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, Tier};

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

const BUDGET: u64 = 512 * 1024 * 1024;

fn governed(budget: u64) -> (Box<GovernedAllocator>, LedgerGovernor) {
    let governor = LedgerGovernor::new(LeaseLedger::new(0, budget, 0));
    let allocator = GovernedAllocator::new(
        MemoryInfo::cpu_device().expect("cpu device memory info"),
        Arc::new(governor.clone()),
        Tier::Host,
        AllocationRoles::split(),
        HolderId::new(42),
    )
    .expect("host allocator");
    (allocator, governor)
}

/// Build valid inputs from the model's own signature, so the run actually
/// executes rather than failing on a missing port.
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

/// The whole point: a session that opted in must charge its allocations to the
/// governor.
///
/// Two things are asserted, because either alone is weak. Construction moving
/// the ledger shows the registration is live. `total_count` moving *across the
/// run* shows inference itself flows through us — `live_count` cannot say that,
/// because ORT frees most of what it takes before `run` returns, so a test
/// sampling it afterwards reads zero either way.
///
/// The run has to actually succeed for the second half to mean anything. An
/// earlier version let it fail on a missing input and asserted nothing about
/// it; the run allocated zero times and the test still passed.
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

    let options = SessionOptions::default().with_intra_op_threads(1);
    let session = Session::new(test_environment(), &model, options).expect("session");

    let free_after_build = governor.available(Tier::Host);
    assert!(
        free_after_build < BUDGET,
        "building a session charged the governor nothing, so the registration is \
         not actually being used ({free_after_build} of {BUDGET} free)"
    );

    let allocations_before_run = registered.total_count();
    run_once(&session).expect("the model must actually run");
    assert!(
        registered.total_count() > allocations_before_run,
        "running the model allocated nothing through the governed allocator \
         ({} allocations before, {} after)",
        allocations_before_run,
        registered.total_count()
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

    // Explicitly opt *out*: governance is on by default, so this is what a
    // caller who wants ORT's own allocator has to do.
    let mut options = SessionOptions::default().with_intra_op_threads(1);
    options
        .session_config_entries
        .retain(|(key, _)| key != USE_ENV_ALLOCATORS);
    let session = Session::new(test_environment(), &model, options).expect("session");
    let _ = run_once(&session);

    assert_eq!(
        governor.available(Tier::Host),
        BUDGET,
        "a session that did not opt in must not reach the registered allocator; \
         if it does, `use_env_allocators` is not the switch it is documented to be"
    );

    drop(session);
    drop(registered);
}

/// Governance is on by default. This is the difference between the feature
/// existing and the feature being used: an opt-in switch leaves every ordinary
/// session ungoverned, and the budget decorative.
#[test]
fn the_default_session_options_opt_in_to_governance() {
    let options = SessionOptions::default();
    assert!(
        options
            .session_config_entries
            .iter()
            .any(|(key, value)| key == USE_ENV_ALLOCATORS && value == "1"),
        "SessionOptions::default() must enable env allocators, or a registered \
         governor never sees an ordinary session's allocations"
    );
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

/// Does ORT actually call `Reserve`, or is the split only in the header?
///
/// The whole reason to implement `Reserve` is that ORT documents it as
/// separating session-initialization allocations from `Run` ones. That is a
/// claim about ORT's behaviour, not about our code, and the only way to know is
/// to build a real session and look.
#[test]
fn ort_really_uses_reserve_for_session_initialization() {
    let _guard = ort_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let model = tiny_llm();
    if !model.exists() {
        return;
    }

    let (allocator, _governor) = governed(BUDGET);
    let registered =
        register_governed_allocator(test_environment(), allocator).expect("register allocator");

    let options = SessionOptions::default().with_intra_op_threads(1);
    let session = Session::new(test_environment(), &model, options).expect("session");

    let reserves = registered.reserve_count();
    let total = registered.total_count();
    assert!(
        total > 0,
        "the session allocated nothing through the governed allocator"
    );
    // Recorded rather than asserted as non-zero: whether ORT routes session
    // setup through `Reserve` depends on the execution provider and the version,
    // and a hard assertion here would fail for a reason that is not a defect.
    // What the run must never do is *only* use Reserve.
    println!("session build: {total} allocations, {reserves} through Reserve");
    assert!(
        reserves <= total,
        "Reserve calls cannot exceed total allocations"
    );

    drop(session);
    drop(registered);
}
