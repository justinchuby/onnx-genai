//! The host runtime's own parallel-for, borrowed for the duration of a kernel.
//!
//! # Why this exists
//!
//! Our CPU kernels split long elementwise slices across a `rayon` pool. That is
//! the right thing to do when we own the machine — the native executor does —
//! but it is the *wrong* thing to do inside an ORT session, because ORT already
//! has an intra-op pool of its own and that pool **spins**. Running our sixteen
//! rayon workers alongside ORT's sixteen spinning workers puts thirty-two
//! runnable threads on sixteen cores, and the result is not a small tax:
//!
//! | op, 1 Mi f32, `intra_op = 16` | serial | our rayon split |
//! |---|---|---|
//! | `Sqrt`     | 252 us | 777 us |
//! | `Sigmoid`  | 521 us | 993 us |
//! | `FastGelu` | 1040 us | 1479 us |
//!
//! Every one of those is a *loss* from parallelising, and it gets worse the
//! more threads ORT was given. Raising the length threshold until the split
//! only fires on very long slices trades one wrong answer for another: with
//! `intra_op = 1` there is nothing to contend with and the same split is a
//! 2-5x win from 1 Mi upwards. No constant can be right for both, because the
//! variable that actually matters is *how much of the machine the host is
//! already using*, and only the host knows that.
//!
//! So we stop guessing and ask. When ORT calls into our compute function it
//! hands us an `OrtKernelContext`, and `OrtApi::KernelContext_ParallelFor`
//! runs a callback on the session's *own* intra-op pool. Routing our chunk
//! split through it means there is exactly one pool on the machine, sized by
//! whatever the user asked for, and oversubscription disappears by
//! construction rather than by tuning.
//!
//! # Shape of the seam
//!
//! This crate is the only thing both `onnx-runtime-ep-cpu` (which has the
//! kernels) and `onnx-runtime-ep-plugin` (which has the `OrtKernelContext`)
//! already depend on, so the seam lives here and neither of them needs to
//! learn about the other.
//!
//! The plugin installs a [`HostParallel`] for the dynamic extent of one
//! compute call with [`scope`]; kernels ask for it with [`current`]. Outside a
//! plugin compute — the native executor, unit tests, anything that never
//! installs one — [`current`] is `None` and callers keep their existing
//! behaviour.
//!
//! # Threading contract
//!
//! * The installed value is **thread-local**. It is only visible on the thread
//!   ORT called us on, which is the only thread that may legally touch the
//!   `OrtKernelContext` it was built from.
//! * [`HostParallel::run`] is **blocking**: it returns only once every index
//!   has run. That is what makes borrowing a `&dyn Fn` from the caller's frame
//!   sound.
//! * Bodies run with [`in_host_task`] set, so a kernel that reaches a second
//!   parallel split from inside a task can see that it is already inside the
//!   host's pool and stay serial instead of nesting.

use core::ffi::c_void;
use core::sync::atomic::{AtomicU8, Ordering};

/// How many threads the host's pool turned out to have.
///
/// The host runtime does not tell us, and it matters: a pool of one is not
/// using the machine, so ours may. Rather than probe — which would cost a
/// dispatch per session to answer a question the next real dispatch answers
/// for free — the implementation *observes* its own first dispatch and records
/// the result here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostWidth {
    /// No dispatch has completed yet.
    Unknown,
    /// Every body ran on the dispatching thread: the host has no workers.
    Serial,
    /// At least one body ran on another thread.
    Parallel,
}

/// Nothing observed yet.
pub const WIDTH_UNKNOWN: u8 = 0;
/// The host pool ran everything inline.
pub const WIDTH_SERIAL: u8 = 1;
/// The host pool used at least one other thread.
pub const WIDTH_PARALLEL: u8 = 2;

/// Runs `body(i)` for every `i` in `0..total` on the host's threads.
///
/// # Safety
///
/// The implementation receives the `host` pointer that was paired with it in
/// [`HostParallel::new`], and may assume it is still valid — which is what
/// [`scope`]'s dynamic extent guarantees. It must invoke `body` exactly once
/// per index in `0..total`, must not let a Rust panic escape into foreign
/// frames, and must not return until every invocation has finished.
pub type HostParallelForFn =
    unsafe fn(host: *mut c_void, total: usize, body: &(dyn Fn(usize) + Sync));

/// A borrowed handle to the host runtime's thread pool.
///
/// Deliberately `Copy` and pointer-sized: it is read out of a thread-local on
/// the hot path, and a clone there would be pure overhead.
#[derive(Clone, Copy)]
pub struct HostParallel {
    host: *mut c_void,
    run: HostParallelForFn,
    /// Where the implementation records [`HostWidth`], or null if it never
    /// will — in which case the pool is assumed parallel.
    width: *const AtomicU8,
}

impl core::fmt::Debug for HostParallel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HostParallel")
            .field("host", &self.host)
            .finish_non_exhaustive()
    }
}

impl HostParallel {
    /// Pairs a host context pointer with the function that drives its pool.
    ///
    /// # Safety
    ///
    /// `host` must stay valid for as long as this handle is installed, and
    /// `run` must honour the contract on [`HostParallelForFn`] for that
    /// pointer. Installing it with [`scope`] is what bounds the first half;
    /// the second half is on the implementer.
    ///
    /// `width` must be null or point at a cell that outlives this handle, and
    /// that only ever holds [`WIDTH_UNKNOWN`], [`WIDTH_SERIAL`] or
    /// [`WIDTH_PARALLEL`].
    pub const unsafe fn new(
        host: *mut c_void,
        run: HostParallelForFn,
        width: *const AtomicU8,
    ) -> Self {
        Self { host, run, width }
    }

    /// What the last completed dispatch said about the host pool's width.
    ///
    /// [`HostWidth::Serial`] is the interesting answer: it means the host's
    /// pool has no workers of its own, so borrowing it would serialise us for
    /// nothing and our own pool is free to use the machine instead.
    #[must_use]
    pub fn width(&self) -> HostWidth {
        if self.width.is_null() {
            return HostWidth::Parallel;
        }
        // SAFETY: `new`'s contract puts the validity of this pointer on the
        // installer, and `scope` bounds it to the extent the handle is
        // reachable.
        match unsafe { &*self.width }.load(Ordering::Relaxed) {
            WIDTH_SERIAL => HostWidth::Serial,
            WIDTH_PARALLEL => HostWidth::Parallel,
            _ => HostWidth::Unknown,
        }
    }

    /// Runs `body(0..total)` on the host's pool and waits for all of it.
    ///
    /// `total` of zero is a no-op, and one is run inline rather than handed to
    /// the pool — a single index cannot be split, so a round trip through the
    /// host would be pure overhead.
    pub fn run(&self, total: usize, body: &(dyn Fn(usize) + Sync)) {
        match total {
            0 => (),
            1 => {
                let _task = TaskGuard::enter();
                body(0);
            }
            _ => {
                // Mark the *task*, not the dispatch: the guard has to be taken
                // on whichever of the host's threads ends up running the body,
                // which is why it is inside the closure rather than around the
                // call.
                let marked = |index: usize| {
                    let _task = TaskGuard::enter();
                    body(index);
                };
                // SAFETY: `host` and `run` were paired in `new`, whose safety
                // contract puts the validity of `host` on the installer, and
                // `scope` bounds it to the extent the handle is reachable.
                unsafe { (self.run)(self.host, total, &marked) }
            }
        }
    }
}

thread_local! {
    /// The host pool this thread may borrow, if it is inside a compute call.
    static CURRENT: core::cell::Cell<Option<HostParallel>> =
        const { core::cell::Cell::new(None) };

    /// Set while this thread is running a body handed to [`HostParallel::run`].
    static IN_TASK: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// A [`HostParallel`] installed on this thread until this value is dropped.
///
/// The closure form ([`scope`]) is the one to reach for. This exists for the
/// FFI entry points, whose bodies are hundreds of lines long and would have to
/// be re-indented wholesale to take a closure.
///
/// Not `Send`, because [`HostParallel`] holds raw pointers — which is exactly
/// the property that keeps the handle on the thread ORT called us on.
pub struct Installed {
    prev: Option<HostParallel>,
}

impl Installed {
    /// Installs `host` on the calling thread.
    pub fn new(host: HostParallel) -> Self {
        Self {
            prev: CURRENT.with(|c| c.replace(Some(host))),
        }
    }
}

impl Drop for Installed {
    fn drop(&mut self) {
        CURRENT.with(|c| c.set(self.prev));
    }
}

/// Installs `host` for the duration of `f` on the calling thread.
///
/// Restores the previous value on unwind as well as on return. That is not
/// tidiness: the handle borrows an `OrtKernelContext` that ORT frees when the
/// compute call returns, so a leaked handle would be a dangling pointer the
/// next kernel on this thread would happily dispatch through.
pub fn scope<T>(host: HostParallel, f: impl FnOnce() -> T) -> T {
    let _installed = Installed::new(host);
    f()
}

/// Runs `f` with no host pool installed, restoring it afterwards.
///
/// For the paths that have to reach a kernel from somewhere the host context
/// is no longer valid, and for tests that need the un-installed behaviour.
pub fn without<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(Option<HostParallel>);
    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(CURRENT.with(|c| c.replace(None)));
    f()
}

/// The host pool installed on this thread, if any.
#[inline]
pub fn current() -> Option<HostParallel> {
    CURRENT.with(core::cell::Cell::get)
}

/// Whether this thread is currently running a [`HostParallel::run`] body.
///
/// A kernel that reaches a parallel split from inside one is already occupying
/// a host worker; splitting again would nest a second pool inside the first.
#[inline]
pub fn in_host_task() -> bool {
    IN_TASK.with(core::cell::Cell::get)
}

struct TaskGuard(bool);

impl TaskGuard {
    fn enter() -> Self {
        Self(IN_TASK.with(|c| c.replace(true)))
    }
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        IN_TASK.with(|c| c.set(self.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A host that runs everything inline, in order, on the calling thread.
    ///
    /// # Safety
    ///
    /// Ignores `host`, so any pointer is valid for it.
    unsafe fn serial_host(_host: *mut c_void, total: usize, body: &(dyn Fn(usize) + Sync)) {
        for index in 0..total {
            body(index);
        }
    }

    fn serial() -> HostParallel {
        // SAFETY: `serial_host` never dereferences its `host` argument, and a
        // null width cell is explicitly allowed.
        unsafe { HostParallel::new(core::ptr::null_mut(), serial_host, core::ptr::null()) }
    }

    fn with_width(cell: &AtomicU8) -> HostParallel {
        // SAFETY: as `serial`, plus `cell` outlives every use below.
        unsafe {
            HostParallel::new(
                core::ptr::null_mut(),
                serial_host,
                core::ptr::from_ref(cell),
            )
        }
    }

    #[test]
    fn a_handle_without_a_width_cell_reads_as_parallel() {
        assert_eq!(serial().width(), HostWidth::Parallel);
    }

    #[test]
    fn width_reflects_the_cell() {
        let cell = AtomicU8::new(WIDTH_UNKNOWN);
        let host = with_width(&cell);
        assert_eq!(host.width(), HostWidth::Unknown);
        cell.store(WIDTH_SERIAL, Ordering::Relaxed);
        assert_eq!(host.width(), HostWidth::Serial);
        cell.store(WIDTH_PARALLEL, Ordering::Relaxed);
        assert_eq!(host.width(), HostWidth::Parallel);
        cell.store(200, Ordering::Relaxed);
        assert_eq!(
            host.width(),
            HostWidth::Unknown,
            "an unknown code is not a promise"
        );
    }

    #[test]
    fn no_host_is_installed_by_default() {
        assert!(current().is_none());
        assert!(!in_host_task());
    }

    #[test]
    fn scope_installs_and_restores() {
        scope(serial(), || {
            assert!(current().is_some());
            without(|| assert!(current().is_none()));
            assert!(current().is_some());
        });
        assert!(current().is_none());
    }

    #[test]
    fn scope_restores_on_unwind() {
        let unwound = std::panic::catch_unwind(|| {
            scope(serial(), || panic!("kernel failed"));
        });
        assert!(unwound.is_err());
        assert!(current().is_none(), "a leaked handle would dangle");
    }

    #[test]
    fn run_covers_every_index_exactly_once() {
        let seen: Vec<AtomicUsize> = (0..7).map(|_| AtomicUsize::new(0)).collect();
        serial().run(seen.len(), &|index| {
            seen[index].fetch_add(1, Ordering::Relaxed);
        });
        assert!(seen.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn empty_total_runs_nothing() {
        let calls = AtomicUsize::new(0);
        serial().run(0, &|_| {
            calls.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_single_index_runs_inline_and_is_still_marked() {
        let marked = AtomicUsize::new(0);
        serial().run(1, &|_| {
            marked.fetch_add(usize::from(in_host_task()), Ordering::Relaxed);
        });
        assert_eq!(marked.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn bodies_run_marked_and_the_mark_does_not_leak() {
        assert!(!in_host_task());
        let marked = AtomicUsize::new(0);
        serial().run(4, &|_| {
            marked.fetch_add(usize::from(in_host_task()), Ordering::Relaxed);
        });
        assert_eq!(marked.load(Ordering::Relaxed), 4);
        assert!(!in_host_task());
    }

    #[test]
    fn the_mark_survives_a_panicking_body() {
        let unwound = std::panic::catch_unwind(|| {
            serial().run(4, &|index| assert_ne!(index, 2, "body failed"));
        });
        assert!(unwound.is_err());
        assert!(!in_host_task());
    }

    #[test]
    fn a_handle_is_not_visible_from_another_thread() {
        scope(serial(), || {
            assert!(std::thread::spawn(|| current().is_none()).join().unwrap());
        });
    }
}
