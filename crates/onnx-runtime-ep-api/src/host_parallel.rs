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
use core::sync::atomic::{AtomicU32, Ordering};

/// Sentinel for "the host's pool has been seen doing our work".
///
/// The only *permanent* state, and it takes positive evidence to reach: a body
/// ran on a thread that was not the one that dispatched it, which a pool with
/// no workers can never produce. Nothing ever clears it.
pub const HOST_HELPED: u32 = u32::MAX;

/// Dispatches a session probes back to back before it starts backing off.
///
/// A probe is deliberately hard to fail: it carries no work, and the
/// implementation holds the calling thread's first index open until another
/// thread is seen taking one, up to a deadline. Instrumented over 15
/// sixteen-thread sessions, every one latched, and the number of probes it
/// took was 1 (nine sessions), 2 (five) or 3 (one). Eight is therefore ample
/// margin. Only a pool with nobody to wake pays the deadline in full, and only
/// eight times per fused node. An earlier design probed by diverting the
/// caller's real work and needed a burst of 32, because an unstalled
/// 16-thread dispatch answers only about half the time.
pub const PROBE_BURST: u32 = 8;

/// Gap between probes when the burst has produced nothing, before backoff.
///
/// The burst is what answers the question; this is only the recovery path for
/// a session whose pool was somehow busy through all eight, so it can start
/// wide and still recover in a few hundred dispatches.
pub const PROBE_MIN: u32 = 64;

/// Longest gap between probes once the host has never been seen to help.
///
/// Bounds the steady-state cost of asking on a session whose pool really is
/// serial: one dispatch in a thousand pays for a probe.
pub const PROBE_MAX: u32 = 1024;

/// Indices in a probe dispatch.
///
/// A probe carries no work: it exists only to see whether anyone else picks up
/// an index. More than one so there is something to pick up, and small so the
/// host's queue never becomes the thing being measured.
pub const PROBE_INDICES: usize = 8;

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
    /// Where this session records whether the host's pool has ever helped,
    /// and when to ask again. Null means "never ask, always use the host".
    probe: *const AtomicU32,
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
    /// `probe` must be null or point at a cell that outlives this handle and
    /// is shared by every handle for the same host pool. Start it at zero;
    /// [`HOST_HELPED`] is the implementation's way of saying the pool has been
    /// seen running our bodies on its own threads. This crate owns every other
    /// value in it.
    pub const unsafe fn new(
        host: *mut c_void,
        run: HostParallelForFn,
        probe: *const AtomicU32,
    ) -> Self {
        Self { host, run, probe }
    }

    /// Whether this dispatch should go to the host's pool rather than ours.
    ///
    /// Answering it needs a fact the host runtime will not tell us: whether
    /// its pool has any workers. A session built with `intra_op = 1` has none,
    /// and borrowing its single thread would serialise work our own pool could
    /// spread over the machine — measured 2-9x slower over 1-4 Mi. A session
    /// with a wide pool is the exact opposite: ours would be a *second* pool
    /// on the same cores, and that measured 3-10x slower than borrowing.
    ///
    /// So ask the only question that has a trustworthy answer: *has the host's
    /// pool ever actually run one of our bodies on another thread?* Only a
    /// pool with workers can, so a "yes" is permanent and cannot be faked.
    /// Until then this returns `false` — keeping the caller on its own pool,
    /// which is what it did before this seam existed.
    ///
    /// The question is asked with a *probe*: an empty [`PROBE_INDICES`]-index
    /// dispatch, run for its scheduling behaviour alone. Probing this way
    /// rather than by sending the caller's real work to the host is what keeps
    /// the unknown state cheap. A `intra_op = 1` session never latches, and
    /// its real work would have been serialised on ORT's single thread every
    /// time it was used as the question — 800 us at 1 Mi, against a 5x win
    /// from our own pool. An empty probe costs the same on every dispatch it
    /// touches, whatever the slice length.
    ///
    /// The first [`PROBE_BURST`] dispatches all probe, because at 16 threads a
    /// single dispatch answering "nobody helped" is uninformative — it happens
    /// 35% of the time — while a burst that long is answered almost surely.
    /// After that, probes back off geometrically from [`PROBE_MIN`] to
    /// [`PROBE_MAX`], so a session that really is serial settles at one probe
    /// per thousand dispatches and still recovers if the burst was unlucky.
    ///
    /// Mutates the cell and may dispatch, so call it once per dispatch
    /// decision, and not from inside a host task.
    pub fn prefer_host(&self) -> bool {
        if self.helped() {
            return true;
        }
        if !self.probe_due() {
            return false;
        }
        self.run(PROBE_INDICES, &|_| ());
        // A probe that was answered leaves the caller's work on the host
        // straight away, rather than spending this dispatch on the wrong pool
        // to act on what was just learned.
        self.helped()
    }

    /// Whether this dispatch is the one that should carry a probe.
    ///
    /// Split out from [`prefer_host`](Self::prefer_host) so the schedule can
    /// be tested without a host to dispatch to.
    fn probe_due(&self) -> bool {
        let Some(cell) = self.probe_cell() else {
            return false;
        };
        let mut seen = cell.load(Ordering::Relaxed);
        loop {
            if seen == HOST_HELPED {
                return false;
            }
            let period = seen >> 16;
            let countdown = seen & 0xFFFF;
            let (next, probe) = if period == 0 {
                // Still in the opening burst: ask every time, and count how
                // many times asking has told us nothing.
                let asked = countdown + 1;
                let next = if asked >= PROBE_BURST {
                    (PROBE_MIN << 16) | PROBE_MIN
                } else {
                    asked
                };
                (next, true)
            } else if countdown == 0 {
                // Probe now, and put the next one twice as far out.
                let period = period.saturating_mul(2).min(PROBE_MAX);
                ((period << 16) | period, true)
            } else {
                ((period << 16) | (countdown - 1), false)
            };
            match cell.compare_exchange_weak(seen, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return probe,
                Err(current) => seen = current,
            }
        }
    }

    /// Whether the host's pool has been seen running our work.
    #[must_use]
    pub fn helped(&self) -> bool {
        self.probe_cell()
            .is_none_or(|cell| cell.load(Ordering::Relaxed) == HOST_HELPED)
    }

    fn probe_cell(&self) -> Option<&AtomicU32> {
        // SAFETY: `new`'s contract puts the validity of this pointer on the
        // installer, and `scope` bounds it to the extent the handle is
        // reachable.
        (!self.probe.is_null()).then(|| unsafe { &*self.probe })
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

    fn with_probe(cell: &AtomicU32) -> HostParallel {
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
    fn a_handle_without_a_probe_cell_always_uses_the_host() {
        assert!(serial().prefer_host());
        assert!(serial().helped());
    }

    /// The opening burst asks every time: one silent dispatch is not evidence
    /// of a serial pool, a run of 64 is.
    #[test]
    fn the_opening_burst_probes_every_dispatch() {
        let cell = AtomicU32::new(0);
        let host = with_probe(&cell);
        for step in 0..PROBE_BURST {
            assert!(host.probe_due(), "dispatch {step} of the burst");
        }
        assert!(!host.probe_due(), "the burst has to end somewhere");
    }

    /// A probe is its own dispatch, and an unhelpful one keeps the caller on
    /// its own pool.
    ///
    /// The distinction is the whole point of probing separately: an
    /// `intra_op = 1` session would otherwise answer the question by running
    /// the caller's 1 Mi slice on one thread, at 800 us a time, when our own
    /// pool does it 5x faster.
    #[test]
    fn a_probe_does_not_carry_the_callers_work() {
        let cell = AtomicU32::new(0);
        let indices = AtomicUsize::new(0);
        // SAFETY: as `with_probe`; `COUNT` outlives the handle.
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        unsafe fn counting_host(_host: *mut c_void, total: usize, body: &(dyn Fn(usize) + Sync)) {
            COUNT.fetch_add(total, Ordering::Relaxed);
            for index in 0..total {
                body(index);
            }
        }
        COUNT.store(0, Ordering::Relaxed);
        // SAFETY: `counting_host` never dereferences `host`, and `cell`
        // outlives the handle.
        let host = unsafe {
            HostParallel::new(
                core::ptr::null_mut(),
                counting_host,
                core::ptr::from_ref(&cell),
            )
        };
        assert!(!host.prefer_host(), "an inline host never proves itself");
        assert_eq!(
            COUNT.load(Ordering::Relaxed),
            PROBE_INDICES,
            "the probe dispatch is empty and fixed-size"
        );
        indices.store(0, Ordering::Relaxed);
    }

    /// Until the host has been seen helping, work stays on our pool -- which
    /// is what the caller did before this seam existed.
    #[test]
    fn an_unhelpful_host_is_asked_less_and_less_often() {
        let cell = AtomicU32::new(0);
        let host = with_probe(&cell);
        let mut gaps = Vec::new();
        let mut gap = 0u32;
        for _ in 0..8400 {
            if host.probe_due() {
                gaps.push(gap);
                gap = 0;
            } else {
                gap += 1;
            }
        }
        let burst = usize::try_from(PROBE_BURST).unwrap();
        assert!(
            gaps[..burst].iter().all(|&g| g == 0),
            "the opening burst asks on every dispatch"
        );
        assert_eq!(
            &gaps[burst..burst + 4],
            &[PROBE_MIN, PROBE_MIN * 2, PROBE_MIN * 4, PROBE_MIN * 8],
            "probes should back off geometrically once the burst is over"
        );
        assert!(
            gaps.iter().all(|&g| g <= PROBE_MAX),
            "the gap must stay bounded so a wrong guess still self-corrects"
        );
        assert_eq!(
            *gaps.last().unwrap(),
            PROBE_MAX,
            "and it should settle at the cap"
        );
    }

    /// One sighting of a worker thread is permanent: only a pool with workers
    /// can produce it, so no later evidence can argue with it.
    #[test]
    fn a_helping_host_is_used_from_then_on() {
        let cell = AtomicU32::new(0);
        let host = with_probe(&cell);
        assert!(!host.helped());
        cell.store(HOST_HELPED, Ordering::Relaxed);
        assert!(host.helped());
        for _ in 0..1000 {
            assert!(host.prefer_host());
            assert!(!host.probe_due(), "a latched host is never probed again");
        }
        assert_eq!(cell.load(Ordering::Relaxed), HOST_HELPED);
    }

    /// Two threads running the same session must not lose or double-count a
    /// probe, and must never corrupt the cell into the sentinel.
    #[test]
    fn concurrent_dispatches_keep_the_cell_sane() {
        let cell = AtomicU32::new(0);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    let host = with_probe(&cell);
                    for _ in 0..5000 {
                        host.probe_due();
                    }
                });
            }
        });
        let seen = cell.load(Ordering::Relaxed);
        assert_ne!(seen, HOST_HELPED, "no thread may invent the sentinel");
        assert!((seen >> 16) <= PROBE_MAX);
        assert!((seen & 0xFFFF) <= PROBE_MAX);
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
