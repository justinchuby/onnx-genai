//! Per-phase instrumentation for the kernel dispatch path (issue #1077).
//!
//! # Why this exists
//!
//! Small-node cost on this EP is dominated by fixed per-`Run` dispatch work,
//! not by the kernels. Hand-instrumenting `compute_execute` with `Instant`
//! probes answered that once, but it is not reproducible: the probes have to
//! be added and reverted by hand, they cannot run in CI, and `perf` is
//! unavailable on the machines this is tuned on (`perf_event_paranoid = 4`).
//!
//! This module makes the breakdown permanent and, more importantly, makes the
//! parts of it that *should not move* assertable.
//!
//! # Counters are deterministic; timings are not
//!
//! The two halves of this module are used very differently.
//!
//! * **Counters** — how many ORT FFI calls a `Run` made, how many heap
//!   allocations the dispatch path asked for, how many status objects crossed
//!   the ABI. These are exact integers, identical on every run and every
//!   machine, so a test can assert them: *"a single-node static-shape `Run`
//!   makes exactly N FFI calls"*. That turns a performance property into a
//!   correctness test, which is the only kind that survives contact with CI.
//!
//!   [`Event::OrtFfiCall`] is a genuine total: the `ffi_coverage` tests scan
//!   this crate's source and fail if any file names an ORT API member without
//!   matching instrumentation, so a new call cannot be added silently.
//!   [`Event::DispatchAlloc`], counted by hand, is a **lower bound** — see its
//!   own documentation, and use [`CountingAllocator`] when the exact figure
//!   matters.
//! * **Timings** — nanoseconds per phase. Useful for finding the next thing to
//!   fix, useless as an assertion. They are gated behind an environment
//!   variable *on top of* the build feature so that enabling counters for a
//!   test does not drag two `Instant::now()` calls into every phase.
//!
//! # Phases are not a partition
//!
//! Summing `phase_ns` does not give the time spent in `Compute`, in either
//! direction. On the success path the phases are *non-exhaustive*: work between
//! them — output allocation bookkeeping, view construction — is inside no
//! phase, so the sum is less than wall time. On an error path they *nest*:
//! `StatusCrossing` opens inside whichever phase was live when the failure was
//! detected, because a guard closes at scope exit rather than at the early
//! `return`, so the sum can exceed wall time. `DispatchLookup` is likewise
//! entered twice per node, once for shape inference and once for workspace
//! preparation, and reports their total.
//!
//! Read a phase as "time attributable to this segment", not as a slice of a
//! pie chart.
//!
//! # Cost when disabled
//!
//! Everything here is behind the `dispatch_probe` feature, which is **not** a
//! default feature. With it off, [`Phase::enter`] returns a zero-sized guard
//! whose `Drop` is empty and [`count`] is an empty `#[inline(always)]` function
//! — there is nothing left for the optimiser to remove.
//! `probe_is_compiled_out_in_production` pins that the guard is a ZST with no
//! `Drop`, and `dispatch_probe_is_not_a_default_feature` pins that a plain
//! build does not get it. Note the limit of what a test can show here: it can
//! prove the guard carries no state and runs nothing on scope exit, but it
//! cannot prove [`count`] has no side effect, because a disabled build has no
//! storage for it to observe. That claim rests on the function body being
//! empty, which is visible in the source directly below.

/// A segment of the dispatch path.
///
/// The list is the decomposition of what happens between ORT calling us and us
/// returning, in execution order. Each variant is a phase whose cost we want
/// attributed separately, because each has a different fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub enum Phase {
    /// ORT's call into `compute_execute`, up to the point where any of our own
    /// work starts: argument checks, resolving the exported state, fetching the
    /// host API table.
    CallbackEntry = 0,
    /// Asking ORT for input/output element types, ranks and dimensions.
    MetadataQuery = 1,
    /// Turning ORT's raw pointers into the `TensorView`s a kernel takes:
    /// stride computation, absent-slot placeholders, positional remapping.
    TensorBind = 2,
    /// Output allocation and materialisation — `KernelContext_GetOutput` and
    /// the scratch buffers for absent slots.
    Allocate = 3,
    /// Deciding *what* to run: shape inference and workspace/plan lookup.
    DispatchLookup = 4,
    /// The kernel itself. Everything else on this list is overhead relative to
    /// this.
    KernelInvoke = 5,
    /// Building a status object and crossing back over the C ABI.
    StatusCrossing = 6,
}

impl Phase {
    /// Number of phases; the length of the per-phase counter arrays.
    pub const COUNT: usize = 7;

    /// Short stable name, used in the dump and in test failure messages.
    pub const fn name(self) -> &'static str {
        match self {
            Phase::CallbackEntry => "callback_entry",
            Phase::MetadataQuery => "metadata_query",
            Phase::TensorBind => "tensor_bind",
            Phase::Allocate => "allocate",
            Phase::DispatchLookup => "dispatch_lookup",
            Phase::KernelInvoke => "kernel_invoke",
            Phase::StatusCrossing => "status_crossing",
        }
    }

    /// Every phase, in execution order.
    pub const ALL: [Phase; Self::COUNT] = [
        Phase::CallbackEntry,
        Phase::MetadataQuery,
        Phase::TensorBind,
        Phase::Allocate,
        Phase::DispatchLookup,
        Phase::KernelInvoke,
        Phase::StatusCrossing,
    ];
}

/// A countable event on the dispatch path.
///
/// Unlike [`Phase`], these are not time windows — they are things that happen a
/// whole number of times per `Run`, and whose count is exactly what we are
/// trying to drive down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub enum Event {
    /// One call through a function pointer in ORT's `OrtApi` table. This is the
    /// headline number for this issue: each one is a cross-library indirect
    /// call that ORT's own CPU EP does not have to make.
    ///
    /// This is a **total**, not a sample. The `ffi_coverage` tests scan every
    /// source file that reaches into `OrtApi` and fail if the set of members it
    /// names changes without the instrumentation changing with it, so a call
    /// added later cannot go uncounted without someone noticing.
    OrtFfiCall = 0,
    /// One heap allocation the dispatch path asked for, at a site we control.
    ///
    /// Unlike [`Event::OrtFfiCall`] this is a **lower bound**, and is honest
    /// about it. Hand-placed counts cannot be exhaustive: `Vec::new()` does not
    /// allocate, `Vec::with_capacity(0)` does not allocate, and whether a
    /// `collect` allocates once or twice depends on `size_hint` rather than on
    /// anything visible at the call site. It is exhaustive for `read_inputs`,
    /// where the sites are few and each is pinned by a test.
    ///
    /// When the exact whole-`Run` figure is what you need, install
    /// [`CountingAllocator`] and let the allocator count.
    DispatchAlloc = 1,
    /// One `OrtStatus` constructed and handed back across the ABI. Should be
    /// zero on a successful `Run`.
    StatusCreated = 2,
    /// One entry into `compute_execute`. Divide the other counters by this to
    /// get per-`Run` figures.
    ComputeExecute = 3,
    /// One node executed. On a fused multi-node subgraph this is larger than
    /// [`Event::ComputeExecute`].
    NodeExecuted = 4,
}

impl Event {
    /// Number of events; the length of the event counter array.
    pub const COUNT: usize = 5;

    /// Short stable name, used in the dump and in test failure messages.
    pub const fn name(self) -> &'static str {
        match self {
            Event::OrtFfiCall => "ort_ffi_call",
            Event::DispatchAlloc => "dispatch_alloc",
            Event::StatusCreated => "status_created",
            Event::ComputeExecute => "compute_execute",
            Event::NodeExecuted => "node_executed",
        }
    }

    /// Every event.
    pub const ALL: [Event; Self::COUNT] = [
        Event::OrtFfiCall,
        Event::DispatchAlloc,
        Event::StatusCreated,
        Event::ComputeExecute,
        Event::NodeExecuted,
    ];
}

/// A reading of the calling thread's counters.
///
/// Counters are per-thread and monotonic, so the way to measure a region is to
/// take a [`snapshot`] before and after and [`Counters::since`] the two. That
/// composes correctly even when something else on the thread is also counting,
/// which a reset-based API would not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Counters {
    /// Times each phase was entered.
    pub phase_calls: [u64; Phase::COUNT],
    /// Nanoseconds accumulated in each phase. Zero unless timing is enabled.
    pub phase_ns: [u64; Phase::COUNT],
    /// Event tallies.
    pub events: [u64; Event::COUNT],
}

impl Counters {
    /// This reading minus an earlier one — what happened in between.
    ///
    /// Saturating rather than wrapping: a negative delta is impossible for
    /// monotonic counters, so if one ever appears it is a bug in the caller
    /// (comparing readings from different threads, most likely) and reporting
    /// zero is a less misleading answer than a number near `u64::MAX`.
    pub fn since(&self, earlier: &Counters) -> Counters {
        let mut d = Counters::default();
        for i in 0..Phase::COUNT {
            d.phase_calls[i] = self.phase_calls[i].saturating_sub(earlier.phase_calls[i]);
            d.phase_ns[i] = self.phase_ns[i].saturating_sub(earlier.phase_ns[i]);
        }
        for i in 0..Event::COUNT {
            d.events[i] = self.events[i].saturating_sub(earlier.events[i]);
        }
        d
    }

    /// Count for one event.
    pub fn event(&self, e: Event) -> u64 {
        self.events[e as usize]
    }

    /// Times one phase was entered.
    pub fn calls(&self, p: Phase) -> u64 {
        self.phase_calls[p as usize]
    }

    /// Nanoseconds spent in one phase; zero unless timing was enabled.
    pub fn ns(&self, p: Phase) -> u64 {
        self.phase_ns[p as usize]
    }

    /// Human-readable one-line-per-row dump, used by the bench harness and by
    /// test failures.
    pub fn report(&self) -> String {
        let runs = self.event(Event::ComputeExecute).max(1);
        let mut s = String::new();
        s.push_str("[dispatch-probe] per compute_execute:\n");
        for e in Event::ALL {
            s.push_str(&format!(
                "  {:<16} {:>10}  ({:.2}/run)\n",
                e.name(),
                self.event(e),
                self.event(e) as f64 / runs as f64
            ));
        }
        for p in Phase::ALL {
            s.push_str(&format!(
                "  {:<16} {:>10} calls  {:>12} ns  ({:.0} ns/run)\n",
                p.name(),
                self.calls(p),
                self.ns(p),
                self.ns(p) as f64 / runs as f64
            ));
        }
        s
    }
}

#[cfg(feature = "dispatch_probe")]
mod imp {
    use super::{Counters, Event, Phase};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    // Counters are kept twice, because the two readers need different things.
    //
    // The thread-local copy is the precise one: a dispatch is a single-threaded
    // story, and only per-thread accumulation lets one measurement be read
    // without another thread's work bleeding into it. That isolation is what
    // makes the exact-count assertions in `kernel_ctx` meaningful.
    //
    // The global copy exists because ORT decides which thread runs `Compute`,
    // and the harness reading the numbers back over the C ABI is not
    // necessarily on it. A thread-local-only probe would report zero there and
    // read as "we made no FFI calls" rather than "you are on the wrong thread".
    //
    // Writing both costs one `Cell` store and one relaxed `fetch_add`, and the
    // whole module is compiled out of production, so the duplication is free
    // where it matters.
    thread_local! {
        static TL_PHASE_CALLS: [Cell<u64>; Phase::COUNT] =
            const { [const { Cell::new(0) }; Phase::COUNT] };
        static TL_PHASE_NS: [Cell<u64>; Phase::COUNT] =
            const { [const { Cell::new(0) }; Phase::COUNT] };
        static TL_EVENTS: [Cell<u64>; Event::COUNT] =
            const { [const { Cell::new(0) }; Event::COUNT] };
    }

    static G_PHASE_CALLS: [AtomicU64; Phase::COUNT] = [const { AtomicU64::new(0) }; Phase::COUNT];
    static G_PHASE_NS: [AtomicU64; Phase::COUNT] = [const { AtomicU64::new(0) }; Phase::COUNT];
    static G_EVENTS: [AtomicU64; Event::COUNT] = [const { AtomicU64::new(0) }; Event::COUNT];

    /// Whether to also accumulate wall time per phase.
    ///
    /// Separate from the build feature on purpose: a test that asserts FFI call
    /// counts wants the counters but emphatically does not want two
    /// `Instant::now()` calls added to every phase it is measuring.
    pub fn timing_enabled() -> bool {
        static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *E.get_or_init(|| {
            std::env::var("ONNX_GENAI_PROFILE_DISPATCH").is_ok_and(|v| v == "1" || v == "time")
        })
    }

    /// Guard returned by [`Phase::enter`]; closes the phase when dropped.
    pub struct PhaseGuard {
        phase: Phase,
        start: Option<Instant>,
    }

    impl PhaseGuard {
        /// Close this phase now rather than at end of scope.
        #[inline(always)]
        pub fn end(self) {}
    }

    impl Drop for PhaseGuard {
        fn drop(&mut self) {
            if let Some(t) = self.start {
                let ns = t.elapsed().as_nanos() as u64;
                let i = self.phase as usize;
                TL_PHASE_NS.with(|a| a[i].set(a[i].get().wrapping_add(ns)));
                G_PHASE_NS[i].fetch_add(ns, Ordering::Relaxed);
            }
        }
    }

    impl Phase {
        /// Open this phase; it closes when the returned guard drops.
        pub fn enter(self) -> PhaseGuard {
            let i = self as usize;
            TL_PHASE_CALLS.with(|a| a[i].set(a[i].get().wrapping_add(1)));
            G_PHASE_CALLS[i].fetch_add(1, Ordering::Relaxed);
            PhaseGuard {
                phase: self,
                start: timing_enabled().then(Instant::now),
            }
        }
    }

    /// Tally one event.
    pub fn count(e: Event) {
        count_n(e, 1);
    }

    /// Tally `n` occurrences of an event at once, for call sites that would
    /// otherwise invoke [`count`] in a tight cycle.
    pub fn count_n(e: Event, n: u64) {
        let i = e as usize;
        TL_EVENTS.with(|a| a[i].set(a[i].get().wrapping_add(n)));
        G_EVENTS[i].fetch_add(n, Ordering::Relaxed);
    }

    /// Read this thread's counters.
    pub fn snapshot() -> Counters {
        let mut c = Counters::default();
        TL_PHASE_CALLS.with(|a| {
            for (dst, src) in c.phase_calls.iter_mut().zip(a) {
                *dst = src.get();
            }
        });
        TL_PHASE_NS.with(|a| {
            for (dst, src) in c.phase_ns.iter_mut().zip(a) {
                *dst = src.get();
            }
        });
        TL_EVENTS.with(|a| {
            for (dst, src) in c.events.iter_mut().zip(a) {
                *dst = src.get();
            }
        });
        c
    }

    /// Read the process-wide totals, summed over every thread that has run a
    /// dispatch. This is what the C entry point reports, since its caller has
    /// no way to be on ORT's worker thread.
    pub fn snapshot_global() -> Counters {
        let mut c = Counters::default();
        for (i, (calls, ns)) in c
            .phase_calls
            .iter_mut()
            .zip(c.phase_ns.iter_mut())
            .enumerate()
        {
            *calls = G_PHASE_CALLS[i].load(Ordering::Relaxed);
            *ns = G_PHASE_NS[i].load(Ordering::Relaxed);
        }
        for (dst, src) in c.events.iter_mut().zip(&G_EVENTS) {
            *dst = src.load(Ordering::Relaxed);
        }
        c
    }

    /// Zero this thread's counters and the process-wide totals.
    ///
    /// Prefer [`snapshot`] plus [`Counters::since`] in-process; this exists for
    /// the exported C entry point, whose caller cannot hold a snapshot across
    /// the ABI. Note that it cannot reach *other* threads' local counters — it
    /// clears the globals, which is what that caller reads.
    pub fn reset() {
        TL_PHASE_CALLS.with(|a| a.iter().for_each(|c| c.set(0)));
        TL_PHASE_NS.with(|a| a.iter().for_each(|c| c.set(0)));
        TL_EVENTS.with(|a| a.iter().for_each(|c| c.set(0)));
        for c in &G_PHASE_CALLS {
            c.store(0, Ordering::Relaxed);
        }
        for c in &G_PHASE_NS {
            c.store(0, Ordering::Relaxed);
        }
        for c in &G_EVENTS {
            c.store(0, Ordering::Relaxed);
        }
    }

    /// Whether this build has the probe compiled in. Always `true` here.
    pub const fn compiled_in() -> bool {
        true
    }
}

#[cfg(not(feature = "dispatch_probe"))]
mod imp {
    use super::{Counters, Event, Phase};

    /// Zero-sized stand-in for the real guard. `Drop` is not implemented, so
    /// letting it fall out of scope is not even a call.
    pub struct PhaseGuard;

    impl PhaseGuard {
        /// No-op in a production build.
        ///
        /// Call sites use this rather than `drop(guard)` so that the
        /// production build — where the guard is a `Drop`-less ZST and `drop`
        /// would be both misleading and a clippy warning — reads correctly in
        /// both configurations.
        #[inline(always)]
        pub fn end(self) {}
    }

    impl Phase {
        /// No-op in a production build.
        #[inline(always)]
        pub fn enter(self) -> PhaseGuard {
            PhaseGuard
        }
    }

    /// No-op in a production build.
    #[inline(always)]
    pub fn count(_e: Event) {}

    /// No-op in a production build.
    #[inline(always)]
    pub fn count_n(_e: Event, _n: u64) {}

    /// All-zero in a production build.
    #[inline(always)]
    pub fn snapshot() -> Counters {
        Counters::default()
    }

    /// All-zero in a production build.
    #[inline(always)]
    pub fn snapshot_global() -> Counters {
        Counters::default()
    }

    /// No-op in a production build.
    #[inline(always)]
    pub fn reset() {}

    /// No-op in a production build.
    #[inline(always)]
    pub fn timing_enabled() -> bool {
        false
    }

    /// Whether this build has the probe compiled in. Always `false` here.
    pub const fn compiled_in() -> bool {
        false
    }
}

pub use imp::{
    PhaseGuard, compiled_in, count, count_n, reset, snapshot, snapshot_global, timing_enabled,
};

/// A `GlobalAlloc` that tallies every real heap allocation into
/// [`Event::DispatchAlloc`], for callers that need an exhaustive count rather
/// than a count of the sites someone remembered to instrument.
///
/// Hand-placed `count(DispatchAlloc)` calls cannot be exhaustive in general,
/// and pretending otherwise is the failure mode this type exists to avoid:
/// `Vec::new()` does not allocate, `Vec::with_capacity(0)` does not allocate,
/// and whether a given `collect` allocates once or twice is a property of
/// `size_hint`, not of the source text. Where the exact number matters, install
/// this and let the allocator answer.
///
/// It is deliberately *not* installed by this crate. A `#[global_allocator]`
/// may be defined only once per binary, so a library that installs one takes
/// that choice away from every dependent. Tests and benchmarks opt in:
///
/// ```ignore
/// #[global_allocator]
/// static A: dispatch_probe::CountingAllocator<std::alloc::System> =
///     dispatch_probe::CountingAllocator::new(std::alloc::System);
/// ```
///
/// Counts reallocations and the allocating half of a resize, and ignores frees,
/// because the question being asked is "how much allocator traffic does one
/// `Run` cause", not "how much memory is live".
///
/// Without the `dispatch_probe` feature this forwards to the inner allocator
/// and records nothing.
pub struct CountingAllocator<A> {
    inner: A,
}

impl<A> CountingAllocator<A> {
    /// Wrap `inner`, counting the allocations that pass through it.
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }
}

// SAFETY: every method forwards to `self.inner` with its arguments unchanged
// and returns its result unchanged, so the allocator contract is exactly the
// inner allocator's. The added work only touches counters.
unsafe impl<A: std::alloc::GlobalAlloc> std::alloc::GlobalAlloc for CountingAllocator<A> {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        count(Event::DispatchAlloc);
        unsafe { self.inner.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { self.inner.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        count(Event::DispatchAlloc);
        unsafe { self.inner.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        count(Event::DispatchAlloc);
        unsafe { self.inner.realloc(ptr, layout, new_size) }
    }
}

/// Tally one ORT FFI call. Shorthand for the single most common count.
#[inline(always)]
pub fn ort_call() {
    count(Event::OrtFfiCall);
}

/// Tally `n` ORT FFI calls at once.
#[inline(always)]
pub fn ort_calls(n: u64) {
    count_n(Event::OrtFfiCall, n);
}

/// Read the process-wide dispatch counters into a caller-provided buffer.
///
/// The e2e harness loads this EP as a cdylib through ORT's
/// `RegisterExecutionProviderLibrary`, so it cannot reach the counters above
/// directly — it resolves this symbol instead. It also has no way to know which
/// thread ORT chose to run `Compute` on, which is why this reports the
/// process-wide totals rather than the calling thread's. In-process users
/// should prefer [`snapshot`], which is isolated per thread.
///
/// Writes `phase_calls`, then `phase_ns`, then `events`, and returns the number
/// of `u64`s written, or 0 if `out` is null or `len` is too small. The required
/// length is `Phase::COUNT * 2 + Event::COUNT`.
///
/// Only exported when the `dispatch_probe` feature is on. A shipped plugin must
/// export the ORT plugin ABI and nothing else, and a `no_mangle` symbol is not
/// free just because the code behind it is: it survives `--gc-sections`, is
/// interposable, and appears in every dynamic symbol table. Absence *is* the
/// "probe not compiled in" answer, which is what a `dlsym` caller already has to
/// handle.
///
/// # Safety
///
/// `out` must be null or point to `len` writable `u64`s.
#[cfg(feature = "dispatch_probe")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nxrt_dispatch_probe_snapshot(out: *mut u64, len: usize) -> usize {
    let need = Phase::COUNT * 2 + Event::COUNT;
    if out.is_null() || len < need {
        return 0;
    }
    let c = snapshot_global();
    // SAFETY: the caller guarantees `out` addresses `len >= need` writable
    // `u64`s, and exactly `need` are written.
    unsafe {
        let mut p = out;
        for v in c.phase_calls {
            p.write(v);
            p = p.add(1);
        }
        for v in c.phase_ns {
            p.write(v);
            p = p.add(1);
        }
        for v in c.events {
            p.write(v);
            p = p.add(1);
        }
    }
    need
}

/// Zero this thread's dispatch counters, for cdylib callers.
///
/// Feature-gated for the same reason as
/// [`nxrt_dispatch_probe_snapshot`]: production exports the ORT plugin ABI only.
#[cfg(feature = "dispatch_probe")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_dispatch_probe_reset() {
    reset();
}

/// Whether the loaded library was built with the probe compiled in.
///
/// Lets a harness tell "the probe reported zero" apart from "the probe is not
/// there", which are very different answers to `did we make any FFI calls`.
///
/// This symbol only exists in a probe build, so resolving it *at all* already
/// answers the question and it always returns 1. It is kept so a caller that
/// resolved it can read a value rather than having to special-case a symbol it
/// looked up successfully, and it still reads `compiled_in()` rather than
/// hard-coding the answer so the two cannot drift apart.
#[cfg(feature = "dispatch_probe")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_dispatch_probe_available() -> i32 {
    i32::from(compiled_in())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The production build must carry no trace of the probe. A zero-sized
    /// guard with no `Drop` is the property that makes `let _g = phase.enter()`
    /// free at every call site.
    #[test]
    #[cfg(not(feature = "dispatch_probe"))]
    fn probe_is_compiled_out_in_production() {
        assert!(!compiled_in());
        assert_eq!(
            std::mem::size_of::<PhaseGuard>(),
            0,
            "the disabled guard must carry no state"
        );
        assert!(
            !std::mem::needs_drop::<PhaseGuard>(),
            "the disabled guard must not run code when it leaves scope"
        );

        let _g = Phase::KernelInvoke.enter();
        count(Event::OrtFfiCall);
        ort_calls(1000);

        // Compared against `Counters::default()`, not against a snapshot taken
        // beforehand. The relative form is vacuous: it compares two calls to a
        // function the compiler is free to fold to a constant, so it passes
        // even against a build whose "disabled" probe records on every call.
        // This was verified by mutation — the relative version did not fail.
        assert_eq!(
            snapshot(),
            Counters::default(),
            "a disabled probe must not report anything"
        );
        assert_eq!(snapshot_global(), Counters::default());
    }

    #[test]
    #[cfg(feature = "dispatch_probe")]
    fn counters_are_exact_and_composable() {
        reset();
        let a = snapshot();
        ort_calls(3);
        count(Event::DispatchAlloc);
        {
            let _g = Phase::MetadataQuery.enter();
        }
        let b = snapshot();
        let d = b.since(&a);
        assert_eq!(d.event(Event::OrtFfiCall), 3);
        assert_eq!(d.event(Event::DispatchAlloc), 1);
        assert_eq!(d.calls(Phase::MetadataQuery), 1);
        assert_eq!(d.calls(Phase::KernelInvoke), 0);

        // A second window must not see the first window's events.
        ort_calls(2);
        let c = snapshot();
        assert_eq!(c.since(&b).event(Event::OrtFfiCall), 2);
        assert_eq!(c.since(&a).event(Event::OrtFfiCall), 5);
    }

    /// Counts made on an ORT worker thread must be visible to the harness
    /// thread. This is the property that thread-local storage would break, and
    /// the reason the counters are process-global.
    #[test]
    #[cfg(feature = "dispatch_probe")]
    fn counts_from_another_thread_are_visible() {
        reset();
        let before = snapshot_global().event(Event::OrtFfiCall);
        std::thread::spawn(|| ort_calls(7)).join().unwrap();
        // `>=`, not `==`: the global mirror is process-wide by construction, so
        // a sibling test running concurrently can legitimately add to it. The
        // claim under test is that the off-thread count is *visible* here, and
        // a lower bound states exactly that without inventing a false
        // determinism the global counter does not have.
        assert!(
            snapshot_global().event(Event::OrtFfiCall) - before >= 7,
            "a count made off-thread must still be readable through the global mirror"
        );
        assert_eq!(
            snapshot().event(Event::OrtFfiCall),
            0,
            "and must not contaminate this thread's isolated counters"
        );
    }

    /// `since` must never report a negative delta as a huge positive one.
    #[test]
    fn since_saturates_rather_than_wrapping() {
        let mut low = Counters::default();
        let mut high = Counters::default();
        high.events[Event::OrtFfiCall as usize] = 10;
        low.events[Event::OrtFfiCall as usize] = 3;
        assert_eq!(high.since(&low).event(Event::OrtFfiCall), 7);
        assert_eq!(low.since(&high).event(Event::OrtFfiCall), 0);
    }

    /// Names are used in dumps and failure messages; keep them unique and
    /// stable so a grep for one finds exactly one thing.
    #[test]
    fn phase_and_event_names_are_unique() {
        let mut names: Vec<&str> = Phase::ALL.iter().map(|p| p.name()).collect();
        names.extend(Event::ALL.iter().map(|e| e.name()));
        let before = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate probe name");
    }

    /// The discriminants index into fixed-size arrays; a reordering that broke
    /// the mapping would silently attribute one phase's cost to another.
    #[test]
    fn discriminants_match_array_positions() {
        for (i, p) in Phase::ALL.iter().enumerate() {
            assert_eq!(*p as usize, i, "{} is out of position", p.name());
        }
        for (i, e) in Event::ALL.iter().enumerate() {
            assert_eq!(*e as usize, i, "{} is out of position", e.name());
        }
    }

    /// The C entry point is what the cdylib harness uses; it must refuse a
    /// buffer it would overrun rather than writing past the end.
    ///
    /// Only exists in a probe build, because the entry point only exists there.
    #[test]
    #[cfg(feature = "dispatch_probe")]
    fn c_snapshot_refuses_a_short_buffer() {
        let need = Phase::COUNT * 2 + Event::COUNT;
        let mut buf = vec![0u64; need];
        // SAFETY: `buf` has exactly `need` writable u64s.
        assert_eq!(
            unsafe { nxrt_dispatch_probe_snapshot(buf.as_mut_ptr(), need) },
            need
        );
        // SAFETY: passing a length smaller than required must be rejected.
        assert_eq!(
            unsafe { nxrt_dispatch_probe_snapshot(buf.as_mut_ptr(), need - 1) },
            0
        );
        // SAFETY: a null pointer must be rejected.
        assert_eq!(
            unsafe { nxrt_dispatch_probe_snapshot(std::ptr::null_mut(), need) },
            0
        );
    }
}

/// Guards the claim that `Event::OrtFfiCall` counts *every* call this crate
/// makes into ORT, rather than the ones somebody remembered to instrument.
///
/// An under-counting probe presented as exact is worse than no probe: it
/// invites a reader to conclude the dispatch path is cheaper than it is. But
/// completeness is not something a unit test can observe from the inside —
/// there is no hook that fires when a function pointer is called. So this
/// checks the next best thing, statically: every ORT entry point the crate
/// names, and every place it counts a call.
///
/// The mechanism is a source scan. ORT's C API members are `CamelCase`, while
/// Rust fields and methods here are `snake_case`, so `.CamelCase` picks out
/// API member accesses and essentially nothing else. Pinning both that set and
/// the instrumentation count means a newly added FFI call cannot land silently:
/// it changes one of these numbers and the author has to come here, look at
/// this comment, and decide.
#[cfg(test)]
mod ffi_coverage {
    /// Every source file that reaches into `OrtApi`, with the ORT members it
    /// names and the number of `ort_call()` sites it contains.
    ///
    /// A count may legitimately differ from the number of distinct members: a
    /// member can be called from more than one place, and an extracted
    /// function pointer can be invoked through a local binding. What must
    /// never happen is a member appearing here with no instrumentation to
    /// account for it.
    const EXPECTED: &[(&str, &str, usize, usize)] = &[
        ("compute.rs", include_str!("compute.rs"), 9, 9),
        ("kernel_ctx.rs", include_str!("kernel_ctx.rs"), 11, 11),
        ("status.rs", include_str!("status.rs"), 1, 1),
        ("host_pool.rs", include_str!("host_pool.rs"), 2, 2),
    ];

    /// Source ahead of the file's `#[cfg(test)]` block — test scaffolding
    /// names ORT members too (the fake `OrtApi` in `kernel_ctx`, the fake
    /// thread pools here), and instrumenting a fake would prove nothing.
    fn production_source(src: &str) -> &str {
        match src.find("\n#[cfg(test)]") {
            Some(i) => &src[..i],
            None => src,
        }
    }

    fn ort_members(src: &str) -> Vec<String> {
        let b = src.as_bytes();
        let mut out: Vec<String> = Vec::new();
        for (i, w) in b.windows(2).enumerate() {
            if w[0] == b'.' && w[1].is_ascii_uppercase() {
                // Skip `::Variant` — a path, not a member access.
                if i > 0 && b[i - 1] == b':' {
                    continue;
                }
                let rest = &src[i + 1..];
                let end = rest
                    .find(|c: char| !c.is_alphanumeric() && c != '_')
                    .unwrap_or(rest.len());
                out.push(rest[..end].to_string());
            }
        }
        out.sort();
        out.dedup();
        out
    }

    #[test]
    fn every_ort_entry_point_is_accounted_for() {
        for (name, src, want_members, want_calls) in EXPECTED {
            let prod = production_source(src);
            let members = ort_members(prod);
            let calls = prod.matches("dispatch_probe::ort_call()").count();
            assert_eq!(
                members.len(),
                *want_members,
                "{name} now names {} ORT API members ({members:?}), not {want_members}. \
                 If you added an FFI call, add an `ort_call()` beside it and update \
                 this table; if you removed one, just update the table.",
                members.len()
            );
            assert_eq!(
                calls, *want_calls,
                "{name} has {calls} `ort_call()` sites, not {want_calls}. Every call \
                 into ORT must be counted, or `Event::OrtFfiCall` stops being a total."
            );
        }
    }

    /// The scan has to actually find things — a heuristic that silently matched
    /// nothing would let the assertions above pass vacuously forever.
    #[test]
    fn the_source_scan_is_not_vacuous() {
        let m = ort_members(production_source(include_str!("kernel_ctx.rs")));
        assert!(m.contains(&"KernelContext_GetInput".to_string()), "{m:?}");
        assert!(m.contains(&"GetTensorData".to_string()), "{m:?}");
        assert!(
            !m.iter().any(|s| s.contains("::")),
            "path segments must not be mistaken for member accesses: {m:?}"
        );
    }
}
