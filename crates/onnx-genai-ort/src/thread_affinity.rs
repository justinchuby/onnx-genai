//! Thread-affinity guards for ORT handles that only one thread may use at a time.
//!
//! # What this enforces, and why it is not "no cross-thread use ever"
//!
//! `OrtIoBinding`, a session-derived `OrtAllocator`, and the values bound to
//! them are *not* thread-safe: ORT documents `Run`/`RunWithBinding` as safe to
//! call concurrently on one `OrtSession`, and says nothing of the kind about the
//! binding or allocator a run mutates. Two threads inside the same binding is
//! undefined behavior, not a slow path.
//!
//! Rust already forbids that for a value the compiler can see: [`crate::IoBinding`],
//! [`crate::Allocator`], and [`crate::Value`] are `!Send + !Sync`, so a shared
//! reference cannot reach a second thread. What the compiler cannot see is a
//! resource smuggled inside a container that carries its own `unsafe impl Send`
//! (`onnx_genai_engine::Engine` does), and that is the case this module names.
//!
//! The rule is therefore *exclusive use*, not *pinning*:
//!
//! * A resource has at most one owning thread **while a guarded section is
//!   live**. A second thread entering one is a hard, reported violation.
//! * Ownership may move to another thread **while the resource is idle**. That
//!   is what an `unsafe impl Send` container legitimately does when it is moved
//!   or handed to a worker: the move itself establishes the happens-before edge,
//!   and exclusive ownership means nobody else is inside the resource.
//!
//! Pinning ("only ever the constructing thread") would reject that legitimate
//! handoff — the server builds a model on a `spawn_blocking` thread and then
//! drives it from a dedicated driver thread — so it would be a rule this
//! codebase violates by design rather than an invariant worth enforcing.
//! Migrations are counted ([`OwnerThread::migration_count`]) so the handoff
//! stays observable instead of silent.
//!
//! # Cost
//!
//! [`ThreadAffinity::Shared`] resources (ORT's process-wide default CPU
//! allocator, a session whose EPs support concurrent `Run`) take one predictable
//! branch. [`ThreadAffinity::Exclusive`] resources take one uncontended
//! `compare_exchange` on entry and one relaxed store on release — nanoseconds
//! against an ORT run, and unlike a `debug_assert!` it still holds in release
//! builds, where the concurrency bug it catches actually happens.

use std::fmt;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};

/// Which threads may use a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadAffinity {
    /// One thread at a time; the owner may change while the resource is idle.
    Exclusive,
    /// Any number of threads at once, because the underlying ORT handle
    /// documents concurrent access as safe.
    Shared,
}

/// A thread, as a violation report can name it.
///
/// One record per thread that has ever touched a guarded ORT resource is leaked
/// deliberately: a violating thread has to be able to name the *other* thread,
/// which may have exited, so the record must outlive it. Threads that never
/// touch ORT never allocate one.
#[derive(Debug)]
pub struct ThreadIdentity {
    id: std::thread::ThreadId,
    name: Option<String>,
}

impl ThreadIdentity {
    fn current() -> Self {
        let thread = std::thread::current();
        Self {
            id: thread.id(),
            name: thread.name().map(ToOwned::to_owned),
        }
    }

    /// The thread's name, when it was given one.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The thread's id.
    #[must_use]
    pub fn id(&self) -> std::thread::ThreadId {
        self.id
    }
}

impl fmt::Display for ThreadIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.name {
            Some(name) => write!(f, "'{name}' ({:?})", self.id),
            None => write!(f, "<unnamed> ({:?})", self.id),
        }
    }
}

thread_local! {
    /// This thread's leaked identity record. The address doubles as the
    /// thread's token, so an ownership check is a single pointer comparison.
    static IDENTITY: &'static ThreadIdentity = Box::leak(Box::new(ThreadIdentity::current()));
}

fn current_identity() -> &'static ThreadIdentity {
    IDENTITY.with(|identity| *identity)
}

fn identity_ptr(identity: &'static ThreadIdentity) -> *mut ThreadIdentity {
    std::ptr::from_ref(identity).cast_mut()
}

/// Use of a non-thread-safe ORT resource from a second thread.
#[derive(Debug, Clone, thiserror::Error)]
#[error(
    "{resource} is in use by thread {owner} and cannot be used by thread {offender} \
     at the same time (operation `{operation}`; the resource was created on thread {created_on}). \
     ORT io bindings, session-derived allocators, and the values bound to them are not \
     thread-safe, so one thread must own the resource for a whole run. \
     Fix: give each worker thread its own session-derived resources, or serialize the \
     workers so only one is inside this resource at a time."
)]
pub struct ThreadAffinityError {
    /// The guarded resource, e.g. `"IoBinding"`.
    pub resource: &'static str,
    /// The operation the offending thread attempted, e.g. `"bind_input"`.
    pub operation: &'static str,
    /// The thread that already holds the resource.
    ///
    /// Identity records are leaked for the process lifetime, so a violation can
    /// still name a thread that has since exited, and carrying the reference
    /// rather than a formatted `String` keeps this error small enough that
    /// wrapping it in [`crate::OrtError`] does not bloat every `Result` in the
    /// crate.
    pub owner: &'static ThreadIdentity,
    /// The thread that was refused.
    pub offender: &'static ThreadIdentity,
    /// The thread the resource was constructed on.
    pub created_on: &'static ThreadIdentity,
}

/// Records which thread owns an ORT resource and refuses overlapping use.
///
/// See the [module docs](self) for the exclusivity rule and why it permits an
/// idle resource to change threads.
#[derive(Debug)]
pub struct OwnerThread {
    resource: &'static str,
    affinity: ThreadAffinity,
    created_on: &'static ThreadIdentity,
    /// Thread inside a guarded section, or null when idle.
    owner: AtomicPtr<ThreadIdentity>,
    /// Nested guarded sections held by `owner`; only the owner mutates it.
    depth: AtomicU32,
    /// Last thread to hold a guarded section, retained so a handoff can be
    /// distinguished from re-entry by the same thread.
    previous_owner: AtomicPtr<ThreadIdentity>,
    /// How many times ownership moved to a different thread.
    migrations: AtomicU64,
}

impl OwnerThread {
    /// Guard `resource` (a static label used verbatim in violation reports).
    #[must_use]
    pub fn new(resource: &'static str, affinity: ThreadAffinity) -> Self {
        Self {
            resource,
            affinity,
            created_on: current_identity(),
            owner: AtomicPtr::new(std::ptr::null_mut()),
            depth: AtomicU32::new(0),
            previous_owner: AtomicPtr::new(std::ptr::null_mut()),
            migrations: AtomicU64::new(0),
        }
    }

    /// The label this guard reports as the resource name.
    #[must_use]
    pub fn resource(&self) -> &'static str {
        self.resource
    }

    /// Whether concurrent use is permitted at all.
    #[must_use]
    pub fn affinity(&self) -> ThreadAffinity {
        self.affinity
    }

    /// The thread the resource was constructed on.
    #[must_use]
    pub fn created_on(&self) -> &'static ThreadIdentity {
        self.created_on
    }

    /// How many times an idle resource was taken over by a different thread.
    ///
    /// Zero means the resource never left the thread that built it. A non-zero
    /// count is legal (see the [module docs](self)) but it is the number that
    /// says a handoff happened, so a test can pin the behavior it expects.
    #[must_use]
    pub fn migration_count(&self) -> u64 {
        self.migrations.load(Ordering::Relaxed)
    }

    /// Whether the calling thread is currently inside a guarded section.
    #[must_use]
    pub fn held_by_current_thread(&self) -> bool {
        std::ptr::eq(
            self.owner.load(Ordering::Acquire).cast_const(),
            std::ptr::from_ref(current_identity()),
        )
    }

    /// Take exclusive use of the resource for the duration of `operation`.
    ///
    /// Fails only when another thread is already inside a guarded section of
    /// the same resource, which is the undefined-behavior case this exists to
    /// turn into an error message.
    pub fn enter(
        &self,
        operation: &'static str,
    ) -> std::result::Result<ThreadAccess<'_>, ThreadAffinityError> {
        if self.affinity == ThreadAffinity::Shared {
            return Ok(ThreadAccess { owner: None });
        }
        let me = identity_ptr(current_identity());
        match self.owner.compare_exchange(
            std::ptr::null_mut(),
            me,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                self.depth.store(1, Ordering::Relaxed);
                let previous = self.previous_owner.swap(me, Ordering::Relaxed);
                if !previous.is_null() && !std::ptr::eq(previous, me) {
                    self.migrations.fetch_add(1, Ordering::Relaxed);
                }
                Ok(ThreadAccess { owner: Some(self) })
            }
            Err(current) if std::ptr::eq(current, me) => {
                self.depth.fetch_add(1, Ordering::Relaxed);
                Ok(ThreadAccess { owner: Some(self) })
            }
            Err(current) => Err(self.violation(operation, current)),
        }
    }

    /// Report what an [`OwnerThread::enter`] would refuse, without entering.
    ///
    /// Teardown uses this: `Drop` cannot return an error, and a resource
    /// released while another thread is still inside it is a bug worth naming
    /// even though the release itself has to proceed.
    pub fn check(&self, operation: &'static str) -> std::result::Result<(), ThreadAffinityError> {
        if self.affinity == ThreadAffinity::Shared {
            return Ok(());
        }
        let me = identity_ptr(current_identity());
        let current = self.owner.load(Ordering::Acquire);
        if current.is_null() || std::ptr::eq(current, me) {
            Ok(())
        } else {
            Err(self.violation(operation, current))
        }
    }

    fn violation(
        &self,
        operation: &'static str,
        owner: *mut ThreadIdentity,
    ) -> ThreadAffinityError {
        // SAFETY: `owner` is an identity record leaked for the whole process by
        // `IDENTITY`, so the pointer stays valid and immutable even if the
        // thread that published it has exited.
        let owner = unsafe { &*owner };
        ThreadAffinityError {
            resource: self.resource,
            operation,
            owner,
            offender: current_identity(),
            created_on: self.created_on,
        }
    }

    fn release(&self) {
        if self.depth.fetch_sub(1, Ordering::Relaxed) == 1 {
            self.owner.store(std::ptr::null_mut(), Ordering::Release);
        }
    }
}

/// Proof that the current thread holds a resource for one operation.
///
/// Dropping it releases the resource so another thread may take it over.
#[derive(Debug)]
#[must_use = "the resource is only held for as long as this guard lives"]
pub struct ThreadAccess<'a> {
    owner: Option<&'a OwnerThread>,
}

impl Drop for ThreadAccess<'_> {
    fn drop(&mut self) {
        if let Some(owner) = self.owner {
            owner.release();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exclusive_resource_allows_nested_use_by_its_owner() {
        let owner = OwnerThread::new("IoBinding", ThreadAffinity::Exclusive);
        let outer = owner.enter("bind_input").expect("owner may enter");
        let inner = owner.enter("bind_output").expect("owner may re-enter");
        assert!(owner.held_by_current_thread());
        drop(inner);
        assert!(owner.held_by_current_thread());
        drop(outer);
        assert!(!owner.held_by_current_thread());
        assert_eq!(owner.migration_count(), 0);
    }

    #[test]
    fn idle_resource_may_be_taken_over_by_another_thread() {
        let owner = OwnerThread::new("IoBinding", ThreadAffinity::Exclusive);
        drop(owner.enter("bind_input").expect("owner may enter"));
        std::thread::scope(|scope| {
            scope.spawn(|| {
                drop(owner.enter("bind_input").expect("idle resource migrates"));
            });
        });
        assert_eq!(owner.migration_count(), 1);
        // Migrating back is a second handoff, not a free re-entry.
        drop(owner.enter("bind_input").expect("owner takes it back"));
        assert_eq!(owner.migration_count(), 2);
    }

    #[test]
    fn overlapping_use_from_a_second_thread_is_refused_with_both_threads_named() {
        let owner = OwnerThread::new("IoBinding", ThreadAffinity::Exclusive);
        let held = owner.enter("run_with_binding").expect("owner may enter");
        let violation = std::thread::scope(|scope| {
            scope
                .spawn(|| owner.enter("bind_input").expect_err("must be refused"))
                .join()
                .expect("probe thread")
        });
        assert_eq!(violation.resource, "IoBinding");
        assert_eq!(violation.operation, "bind_input");
        let rendered = violation.to_string();
        assert!(
            rendered.contains("is in use by thread"),
            "violation must name the holding thread: {rendered}"
        );
        assert!(
            rendered.contains("Fix:"),
            "violation must say what to do about it: {rendered}"
        );
        drop(held);
        // Once the owner leaves, the resource is available again.
        std::thread::scope(|scope| {
            scope.spawn(|| {
                drop(
                    owner
                        .enter("bind_input")
                        .expect("released resource is free"),
                );
            });
        });
    }

    #[test]
    fn check_reports_the_same_violation_without_taking_the_resource() {
        let owner = OwnerThread::new("Allocator", ThreadAffinity::Exclusive);
        let held = owner.enter("alloc").expect("owner may enter");
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let error = owner.check("drop").expect_err("must be refused");
                assert_eq!(error.resource, "Allocator");
                assert_eq!(error.operation, "drop");
            });
        });
        drop(held);
        owner.check("drop").expect("idle resource passes the check");
    }

    #[test]
    fn shared_resources_never_refuse_concurrent_use() {
        let owner = OwnerThread::new("Session", ThreadAffinity::Shared);
        let held = owner.enter("run").expect("shared entry");
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    drop(
                        owner
                            .enter("run")
                            .expect("shared sessions run concurrently"),
                    );
                    owner.check("run").expect("shared sessions never conflict");
                });
            }
        });
        drop(held);
        assert_eq!(owner.migration_count(), 0);
    }

    #[test]
    fn identities_render_named_and_unnamed_threads() {
        let named = std::thread::Builder::new()
            .name("decode-worker".to_owned())
            .spawn(|| current_identity().to_string())
            .expect("spawn named thread")
            .join()
            .expect("named thread");
        assert!(
            named.contains("'decode-worker'"),
            "named thread must render its name: {named}"
        );
        let unnamed = std::thread::spawn(|| current_identity().to_string())
            .join()
            .expect("unnamed thread");
        assert!(
            unnamed.contains("<unnamed>"),
            "unnamed thread must still be identifiable: {unnamed}"
        );
    }
}
