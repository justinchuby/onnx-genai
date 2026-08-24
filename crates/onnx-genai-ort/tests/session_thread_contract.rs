//! The thread and lifetime contract of ORT handles, as tests rather than prose.
//!
//! Three properties are asserted here, and all three have been violated in this
//! repository before:
//!
//! 1. **`Session` is `Send + Sync`, and the handles derived from it are not.**
//!    A `Session` crosses threads (built on a blocking pool, driven from an
//!    engine thread); an `IoBinding`, `Allocator`, or `Value` must not, because
//!    ORT gives no thread-safety guarantee for them. If someone "fixes" a
//!    compile error by adding `unsafe impl Send` to one of those, this fails.
//! 2. **A session-derived handle cannot outlive its session.** ORT frees an
//!    `IoBinding`'s bound state and a `CreateAllocator` allocator's memory
//!    *through* the session, so `ReleaseSession` must come last. The borrowed
//!    forms make that a compile error and the `Arc` forms make it a refcount
//!    fact; this checks the refcount half at runtime.
//! 3. **Exclusive-use enforcement is real.** A second thread that reaches a
//!    held binding or allocator is refused with a named error instead of
//!    racing ORT.
//!
//! Only the last group needs a live ONNX Runtime; the rest are compile-time or
//! pure-Rust and run everywhere.

use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use onnx_genai_ort::{
    Allocator, ConcurrentRunSupport, DataType, Environment, IoBinding, Session, SessionOptions,
    ThreadAffinity, Value,
};

// ---------------------------------------------------------------------------
// 1. Static Send/Sync contract
// ---------------------------------------------------------------------------

/// Report whether `T: Send` without requiring it.
///
/// The inherent `impl` is preferred over the trait default during method
/// resolution, so `IsSend::<T>::VALUE` is `true` exactly when `T: Send` holds.
/// This is what lets the negative half of the contract be asserted at all: a
/// plain `fn assert_send<T: Send>()` can only state the positive.
struct IsSend<T: ?Sized>(PhantomData<T>);
trait SendFallback {
    const VALUE: bool = false;
}
impl<T: ?Sized> SendFallback for IsSend<T> {}
impl<T: ?Sized + Send> IsSend<T> {
    const VALUE: bool = true;
}

struct IsSync<T: ?Sized>(PhantomData<T>);
trait SyncFallback {
    const VALUE: bool = false;
}
impl<T: ?Sized> SyncFallback for IsSync<T> {}
impl<T: ?Sized + Sync> IsSync<T> {
    const VALUE: bool = true;
}

/// The contract, stated so that violating it fails to compile.
///
/// `const { assert!(..) }` is deliberate: a runtime assertion would only report
/// a broken contract when someone runs this test, while a const block refuses to
/// build the crate at all. Adding `unsafe impl Send for IoBinding` to silence a
/// compile error elsewhere therefore stops the build here, at the statement of
/// the rule, instead of producing a data race in a release binary.
const _SEND_SYNC_CONTRACT: () = {
    // A Session crosses threads: engines build sessions on a blocking pool and
    // then move them to the thread that drives them, and `&Session` is shared
    // across an engine's internals.
    assert!(IsSend::<Session>::VALUE, "Session must be Send");
    assert!(IsSync::<Session>::VALUE, "Session must be Sync");

    // Everything derived from a session must not. ORT documents no thread
    // safety for these; single-threaded pipeline code holds them behind
    // `RefCell` precisely because the compiler refuses to let them escape, and
    // making any of them `Send` would silently remove that protection.
    assert!(
        !IsSend::<IoBinding<'static>>::VALUE,
        "IoBinding must not be Send: bind/run/drop must stay with one thread"
    );
    assert!(
        !IsSync::<IoBinding<'static>>::VALUE,
        "IoBinding must not be Sync"
    );
    assert!(
        !IsSend::<Allocator<'static>>::VALUE,
        "Allocator must not be Send: it allocates through its session's EP"
    );
    assert!(
        !IsSync::<Allocator<'static>>::VALUE,
        "Allocator must not be Sync"
    );
    assert!(
        !IsSend::<Value>::VALUE,
        "Value must not be Send: an OrtValue may reference device memory owned by one thread"
    );
    assert!(!IsSync::<Value>::VALUE, "Value must not be Sync");
};

#[test]
fn the_send_sync_contract_holds() {
    // The work happens in `_SEND_SYNC_CONTRACT` at compile time; this exists so
    // the contract is discoverable from a test run and named in test output.
    let () = _SEND_SYNC_CONTRACT;
}

// ---------------------------------------------------------------------------
// 2. Thread-affinity guard behavior (no ONNX Runtime needed)
// ---------------------------------------------------------------------------

#[test]
fn a_second_thread_reaching_a_held_resource_is_refused_by_name() {
    let owner = onnx_genai_ort::OwnerThread::new("IoBinding", ThreadAffinity::Exclusive);
    let held = owner.enter("bind_input").expect("first use");

    let violation = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                owner
                    .enter("run")
                    .expect_err("a held resource must refuse a second thread")
            })
            .join()
            .expect("intruder thread")
    });

    let message = violation.to_string();
    // An unactionable panic in ORT is the alternative; the error has to say
    // which resource, which operation, and what to do instead.
    assert!(
        message.contains("IoBinding"),
        "must name the resource: {message}"
    );
    assert!(
        message.contains("run"),
        "must name the attempted operation: {message}"
    );
    assert!(
        message.contains("Fix:"),
        "must say what to do instead: {message}"
    );
    drop(held);
}

#[test]
fn an_idle_resource_may_move_to_another_thread_and_records_it() {
    // Engines legitimately build ORT state on one thread and drive it from
    // another; that hand-off is exclusive, so it is allowed - but it is counted,
    // so a hand-off that was supposed to be impossible is still observable.
    let owner = onnx_genai_ort::OwnerThread::new("Allocator", ThreadAffinity::Exclusive);
    drop(owner.enter("allocate").expect("first use"));
    assert_eq!(owner.migration_count(), 0);

    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                drop(
                    owner
                        .enter("allocate")
                        .expect("an idle resource may migrate"),
                );
            })
            .join()
            .expect("second thread");
    });
    assert_eq!(owner.migration_count(), 1, "a hand-off must stay visible");
}

// ---------------------------------------------------------------------------
// 3. Lifetime/lifecycle against a live ONNX Runtime
// ---------------------------------------------------------------------------

fn tiny_llm() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm/model.onnx.textproto")
}

fn cpu_session() -> (Environment, Arc<Session>) {
    let environment = Environment::new("session-thread-contract").expect("ORT environment");
    let session = Session::new(&environment, &tiny_llm(), SessionOptions::default())
        .expect("tiny-llm CPU session");
    (environment, Arc::new(session))
}

#[test]
fn a_shared_binding_keeps_its_session_alive_past_the_owners_last_handle() {
    let (environment, session) = cpu_session();
    let binding = IoBinding::for_shared_session(Arc::clone(&session)).expect("binding");
    assert_eq!(
        Arc::strong_count(&session),
        2,
        "the binding must co-own the session"
    );

    // This is the drop order that used to be a latent use-after-free: the owner
    // releases its session handle while a binding derived from it is still
    // alive. The Arc makes it merely a refcount decrement.
    drop(session);
    assert!(
        binding
            .session()
            .input_names()
            .iter()
            .any(|name| name == "input_ids")
    );

    drop(binding);
    drop(environment);
}

#[test]
fn a_shared_allocator_keeps_its_session_alive_and_releases_before_it() {
    let (environment, session) = cpu_session();
    // The CPU EP has no device allocator; `None` is the correct answer there and
    // still exercises the shared-ownership path's plumbing.
    let allocator = Session::shared_device_allocator(&session).expect("allocator query");
    if let Some(allocator) = allocator {
        assert_eq!(Arc::strong_count(&session), 2);
        drop(session);
        drop(allocator);
    } else {
        assert_eq!(Arc::strong_count(&session), 1);
        drop(session);
    }
    drop(environment);
}

#[test]
fn a_borrowed_allocator_reports_the_session_it_allocates_through() {
    let (environment, session) = cpu_session();
    let process_default = Allocator::default_cpu().expect("process default CPU allocator");
    assert!(
        process_default.session().is_none(),
        "ORT's process-wide CPU allocator belongs to no session and must never be released here"
    );
    // It is documented thread-safe, so it must not carry an exclusivity guard
    // that would make sharing it a false positive.
    assert_eq!(
        process_default.thread_affinity().affinity(),
        ThreadAffinity::Shared
    );
    let value = Value::empty_in(&[2, 2], DataType::Float32, &process_default).expect("tensor");
    assert_eq!(value.shape(), &[2, 2]);
    drop(session);
    drop(environment);
}

#[test]
fn a_cpu_session_declares_whether_concurrent_run_is_safe() {
    let (environment, session) = cpu_session();
    // The CPU EP is one of the providers ORT documents as concurrently
    // runnable, and this session does not capture graphs.
    assert!(
        session.supports_concurrent_run(),
        "a non-capturing CPU session must permit concurrent Run: {:?}",
        session.concurrent_run_support()
    );
    assert_eq!(
        session.concurrent_run_support(),
        ConcurrentRunSupport::Supported
    );
    assert!(session.concurrent_run_support().reason().is_none());
    drop(session);
    drop(environment);
}

#[test]
fn a_concurrently_runnable_session_can_actually_be_run_from_two_threads() {
    let (environment, session) = cpu_session();
    assert!(session.supports_concurrent_run());

    // Two threads, one session, real ORT runs. `Session: Sync` is only sound if
    // this works, so assert it rather than trusting the impl.
    std::thread::scope(|scope| {
        let handles = (0..2)
            .map(|_| {
                let session = Arc::clone(&session);
                scope.spawn(move || {
                    let tokens = Value::from_slice_i64(&[1, 2, 3], &[1, 3]).expect("tokens");
                    let mask = Value::from_slice_i64(&[1, 1, 1], &[1, 3]).expect("mask");
                    let positions = Value::from_slice_i64(&[0, 1, 2], &[1, 3]).expect("positions");
                    let past_key = Value::empty(&[1, 2, 0, 8], DataType::Float32).expect("key");
                    let past_value = Value::empty(&[1, 2, 0, 8], DataType::Float32).expect("value");
                    let outputs = session
                        .run(&[
                            ("input_ids", &tokens),
                            ("attention_mask", &mask),
                            ("position_ids", &positions),
                            ("past_key_values.0.key", &past_key),
                            ("past_key_values.0.value", &past_value),
                        ])
                        .expect("concurrent run on a concurrently-runnable session");
                    assert!(!outputs.is_empty());
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().expect("concurrent run thread");
        }
    });
    drop(session);
    drop(environment);
}

#[test]
fn a_real_binding_refuses_a_second_thread_mid_operation() {
    let (environment, session) = cpu_session();
    let binding = IoBinding::for_shared_session(Arc::clone(&session)).expect("binding");
    // Hold the binding the way a run does, then let a second thread try to
    // reach it. `IoBinding` is `!Send`, so the second thread can only get at the
    // guard - which is the point: this is what a container with its own
    // `unsafe impl Send` would smuggle across, and it is refused by name.
    //
    // The closure below captures the guard, not the binding: capturing
    // `&IoBinding` does not compile, because `IoBinding` is `!Sync`. That
    // refusal is the first line of defense, and the guard is the second - the
    // one that still applies when an `unsafe impl Send` container carries the
    // binding across for real.
    let affinity = binding.thread_affinity();
    let held = affinity.enter("run_with_binding").expect("owner");
    let violation = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                affinity
                    .enter("bind_input")
                    .expect_err("a binding in use must refuse a second thread")
            })
            .join()
            .expect("intruder thread")
    });
    assert!(violation.to_string().contains("IoBinding"), "{violation}");
    assert_eq!(violation.resource, "IoBinding");
    assert_eq!(violation.operation, "bind_input");
    assert_ne!(
        violation.owner.id(),
        violation.offender.id(),
        "the report must distinguish the two threads"
    );

    drop(held);
    drop(binding);
    drop(session);
    drop(environment);
}
