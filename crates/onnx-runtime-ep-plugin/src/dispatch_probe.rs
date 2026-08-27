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
    /// Heap allocations made while each phase was the innermost open one.
    ///
    /// Length is [`ALLOC_BUCKETS`], not `Phase::COUNT`: the last slot is
    /// [`UNATTRIBUTED`], charged when an allocation happens with no phase open.
    /// Without that slot the table silently omits allocations rather than
    /// showing them, which is the opposite of what an attribution pass needs --
    /// the first live reading had four allocations per node inside phases and
    /// gave no hint whether that was all of them.
    ///
    /// Only populated when a [`CountingAllocator`] is installed as the global
    /// allocator; otherwise zero. Unlike the hand-placed
    /// [`Event::DispatchAlloc`] tally this is exhaustive -- it sees every
    /// allocation, including ones inside `Vec` growth, `format!`, and code we
    /// did not write -- so it is a total rather than a lower bound.
    pub phase_allocs: [u64; ALLOC_BUCKETS],
    /// Bytes requested by those allocations, same bucketing.
    pub phase_alloc_bytes: [u64; ALLOC_BUCKETS],
}

/// Index of the bucket for allocations made with no phase open.
pub const UNATTRIBUTED: usize = Phase::COUNT;

/// Number of allocation buckets: one per phase, plus [`UNATTRIBUTED`].
pub const ALLOC_BUCKETS: usize = Phase::COUNT + 1;

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
        for i in 0..ALLOC_BUCKETS {
            d.phase_allocs[i] = self.phase_allocs[i].saturating_sub(earlier.phase_allocs[i]);
            d.phase_alloc_bytes[i] =
                self.phase_alloc_bytes[i].saturating_sub(earlier.phase_alloc_bytes[i]);
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
        for p in Phase::ALL.into_iter() {
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
    use super::{ALLOC_BUCKETS, Counters, Event, Phase, UNATTRIBUTED};
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
        static TL_PHASE_ALLOCS: [Cell<u64>; ALLOC_BUCKETS] =
            const { [const { Cell::new(0) }; ALLOC_BUCKETS] };
        static TL_PHASE_ALLOC_BYTES: [Cell<u64>; ALLOC_BUCKETS] =
            const { [const { Cell::new(0) }; ALLOC_BUCKETS] };
    }

    // Which phase is currently open on this thread, or `NO_PHASE`.
    //
    // Read by `CountingAllocator` on every allocation, so it must not allocate
    // itself: a `Cell<u8>` with a `const` initialiser and no `Drop` compiles to
    // a plain TLS slot with no lazy initialisation.
    thread_local! {
        static TL_CURRENT_PHASE: Cell<u8> = const { Cell::new(NO_PHASE) };
    }

    /// Sentinel for "no phase is open", so allocations outside dispatch are
    /// attributed to nothing rather than to phase 0.
    pub const NO_PHASE: u8 = u8::MAX;

    static G_PHASE_CALLS: [AtomicU64; Phase::COUNT] = [const { AtomicU64::new(0) }; Phase::COUNT];
    static G_PHASE_NS: [AtomicU64; Phase::COUNT] = [const { AtomicU64::new(0) }; Phase::COUNT];
    static G_EVENTS: [AtomicU64; Event::COUNT] = [const { AtomicU64::new(0) }; Event::COUNT];
    static G_PHASE_ALLOCS: [AtomicU64; ALLOC_BUCKETS] =
        [const { AtomicU64::new(0) }; ALLOC_BUCKETS];
    static G_PHASE_ALLOC_BYTES: [AtomicU64; ALLOC_BUCKETS] =
        [const { AtomicU64::new(0) }; ALLOC_BUCKETS];

    /// Attribute one allocation of `bytes` to whichever phase is open.
    ///
    /// Called from the global allocator, so it takes the thread-local slot with
    /// `try_with`: during thread teardown the TLS may already be gone, and a
    /// panic out of `alloc` would be considerably worse than a lost count.
    pub fn record_alloc(bytes: u64) {
        let phase = TL_CURRENT_PHASE.try_with(Cell::get).unwrap_or(NO_PHASE);
        let i = if phase == NO_PHASE {
            UNATTRIBUTED
        } else {
            phase as usize
        };
        let _ = TL_PHASE_ALLOCS.try_with(|a| a[i].set(a[i].get().wrapping_add(1)));
        let _ = TL_PHASE_ALLOC_BYTES.try_with(|a| a[i].set(a[i].get().wrapping_add(bytes)));
        G_PHASE_ALLOCS[i].fetch_add(1, Ordering::Relaxed);
        G_PHASE_ALLOC_BYTES[i].fetch_add(bytes, Ordering::Relaxed);
    }

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
        /// The phase that was open when this one started, restored on close.
        ///
        /// Phases nest -- a guard closes at scope exit, not at an early
        /// `return`, so `StatusCrossing` opens inside whatever was live. Saving
        /// and restoring means an allocation inside the inner phase is
        /// attributed to it and the outer phase resumes afterwards, rather than
        /// everything after the first nested phase being lost.
        outer: u8,
        start: Option<Instant>,
    }

    impl PhaseGuard {
        /// Close this phase now rather than at end of scope.
        #[inline(always)]
        pub fn end(self) {}
    }

    impl Drop for PhaseGuard {
        fn drop(&mut self) {
            let _ = TL_CURRENT_PHASE.try_with(|c| c.set(self.outer));
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
            let outer = TL_CURRENT_PHASE.with(|c| c.replace(i as u8));
            PhaseGuard {
                phase: self,
                outer,
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
        TL_PHASE_ALLOCS.with(|a| {
            for (dst, src) in c.phase_allocs.iter_mut().zip(a) {
                *dst = src.get();
            }
        });
        TL_PHASE_ALLOC_BYTES.with(|a| {
            for (dst, src) in c.phase_alloc_bytes.iter_mut().zip(a) {
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
        for (dst, src) in c.phase_allocs.iter_mut().zip(&G_PHASE_ALLOCS) {
            *dst = src.load(Ordering::Relaxed);
        }
        for (dst, src) in c.phase_alloc_bytes.iter_mut().zip(&G_PHASE_ALLOC_BYTES) {
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
        TL_PHASE_ALLOCS.with(|a| a.iter().for_each(|c| c.set(0)));
        TL_PHASE_ALLOC_BYTES.with(|a| a.iter().for_each(|c| c.set(0)));
        for c in &G_PHASE_CALLS {
            c.store(0, Ordering::Relaxed);
        }
        for c in &G_PHASE_NS {
            c.store(0, Ordering::Relaxed);
        }
        for c in &G_EVENTS {
            c.store(0, Ordering::Relaxed);
        }
        for c in &G_PHASE_ALLOCS {
            c.store(0, Ordering::Relaxed);
        }
        for c in &G_PHASE_ALLOC_BYTES {
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

    /// No-op in a production build: nothing tracks the open phase, so there is
    /// nothing to attribute an allocation to.
    #[inline(always)]
    pub fn record_alloc(_bytes: u64) {}

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
        imp::record_alloc(layout.size() as u64);
        unsafe { self.inner.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { self.inner.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        count(Event::DispatchAlloc);
        imp::record_alloc(layout.size() as u64);
        unsafe { self.inner.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        count(Event::DispatchAlloc);
        imp::record_alloc(new_size as u64);
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
/// Writes `phase_calls`, `phase_ns`, `phase_allocs`, `phase_alloc_bytes`, then
/// `events`, and returns the number of `u64`s written, or 0 if `out` is null or
/// `len` is too small. The required length is [`SNAPSHOT_LEN`].
///
/// # Safety
///
/// `out` must be null or point to `len` writable `u64`s.
/// Name of allocation bucket `index`, or null if out of range.
///
/// Exists because the harness must not keep its own copy of the phase order.
/// It did, briefly, and the copy was wrong: the table labelled `TensorBind`'s
/// allocations "OutputMeta" and `Allocate`'s "TensorBind", which is the kind of
/// error that survives review because every number in it looks plausible.
///
/// Index `Phase::COUNT` is the unattributed bucket.
///
/// # Safety
///
/// The returned pointer is to a `'static` NUL-terminated string and must not be
/// freed. It stays valid for the lifetime of the library.
/// Only compiled under the `dispatch_probe` feature. The shipped cdylib must
/// export the ORT plugin ABI and nothing else, and a `no_mangle` symbol is not
/// free just because its body is: it survives `--gc-sections`, it is
/// interposable, and it lands in every dynamic symbol table. Absence *is* the
/// "not compiled in" answer, which `libloading`'s `Option`-returning `get`
/// already models.
#[cfg(feature = "dispatch_probe")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_dispatch_probe_phase_name(index: usize) -> *const std::os::raw::c_char {
    const NAMES: [&core::ffi::CStr; ALLOC_BUCKETS] = [
        c"callback_entry",
        c"metadata_query",
        c"tensor_bind",
        c"allocate",
        c"dispatch_lookup",
        c"kernel_invoke",
        c"status_crossing",
        c"unattributed",
    ];
    match NAMES.get(index) {
        Some(n) => n.as_ptr(),
        None => core::ptr::null(),
    }
}

/// Exported name of an event counter, or null past the end.
///
/// Exists for the same reason [`nxrt_dispatch_probe_phase_name`] does, and
/// because the lesson that export encodes was not applied to events at the
/// time. The `plugin_ort_e2e` harness carried its own hard-coded
/// `PROBE_EVENTS` list, which drifted from [`Event`] and mislabelled three of
/// the five counters: `StatusCreated` was printed as "NodeExecuted",
/// `ComputeExecute` -- the per-`Run` divisor -- as "ShapeInferred", and the
/// real `NodeExecuted` as "OutputMaterialized". Two of those names name no
/// event at all.
///
/// The drift was invisible because the harness guarded *arity* and not
/// identity: it asserted the snapshot wrote as many `u64`s as it expected, and
/// a pure reordering keeps the count at five. Its failure message claimed to
/// detect "PROBE_EVENTS is out of sync with dispatch_probe", which is the one
/// thing it could not do.
///
/// # Safety
///
/// The returned pointer is to a `'static` NUL-terminated string and must not be
/// freed. It stays valid for the lifetime of the library.
#[cfg(feature = "dispatch_probe")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_dispatch_probe_event_name(index: usize) -> *const std::os::raw::c_char {
    const NAMES: [&core::ffi::CStr; Event::COUNT] = [
        c"ort_ffi_call",
        c"dispatch_alloc",
        c"status_created",
        c"compute_execute",
        c"node_executed",
    ];
    match NAMES.get(index) {
        Some(n) => n.as_ptr(),
        None => core::ptr::null(),
    }
}

/// Number of `u64`s [`nxrt_dispatch_probe_snapshot`] writes.
///
/// Exported so the cdylib harness sizes its buffer from the same expression the
/// writer uses, rather than from a number copied into a test.
pub const SNAPSHOT_LEN: usize = Phase::COUNT * 2 + ALLOC_BUCKETS * 2 + Event::COUNT;

/// Only compiled under the `dispatch_probe` feature. The shipped cdylib must
/// export the ORT plugin ABI and nothing else, and a `no_mangle` symbol is not
/// free just because its body is: it survives `--gc-sections`, it is
/// interposable, and it lands in every dynamic symbol table. Absence *is* the
/// "not compiled in" answer, which `libloading`'s `Option`-returning `get`
/// already models.
#[cfg(feature = "dispatch_probe")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nxrt_dispatch_probe_snapshot(out: *mut u64, len: usize) -> usize {
    let need = SNAPSHOT_LEN;
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
        for v in c.phase_allocs {
            p.write(v);
            p = p.add(1);
        }
        for v in c.phase_alloc_bytes {
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
/// Only compiled under the `dispatch_probe` feature. The shipped cdylib must
/// export the ORT plugin ABI and nothing else, and a `no_mangle` symbol is not
/// free just because its body is: it survives `--gc-sections`, it is
/// interposable, and it lands in every dynamic symbol table. Absence *is* the
/// "not compiled in" answer, which `libloading`'s `Option`-returning `get`
/// already models.
#[cfg(feature = "dispatch_probe")]
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_dispatch_probe_reset() {
    reset();
}

/// Whether the loaded library was built with the probe compiled in.
///
/// Lets a harness tell "the probe reported zero" apart from "the probe is not
/// there", which are very different answers to `did we make any FFI calls`.
/// Only compiled under the `dispatch_probe` feature. The shipped cdylib must
/// export the ORT plugin ABI and nothing else, and a `no_mangle` symbol is not
/// free just because its body is: it survives `--gc-sections`, it is
/// interposable, and it lands in every dynamic symbol table. Absence *is* the
/// "not compiled in" answer, which `libloading`'s `Option`-returning `get`
/// already models.
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

    /// The module doc promises the probe is not a default feature, which is
    /// what makes every "production carries none of this" claim in this file
    /// true. That promise lives in the manifest, so it is only worth anything
    /// if something reads the manifest — a `#[cfg]` test cannot see it, because
    /// a build that wrongly defaulted the feature on would simply compile the
    /// other branch and still pass.
    ///
    /// Deliberately not `env!("CARGO_FEATURE_...")`-based for the same reason:
    /// that reports how *this* build was configured, not what the default is.
    #[test]
    fn dispatch_probe_is_not_a_default_feature() {
        let manifest = include_str!("../Cargo.toml");
        let features = manifest
            .split("\n[")
            .find(|section| section.starts_with("features]"))
            .expect("the crate must still have a [features] table");

        // Non-vacuous: if the feature were renamed or removed, the scan below
        // would pass against a section that no longer describes the probe.
        assert!(
            features.contains("dispatch_probe"),
            "the probe feature vanished from the manifest; this test is now \
             pinning nothing: {features}"
        );

        for line in features.lines() {
            let line = line.trim();
            if let Some(default) = line.strip_prefix("default") {
                let default = default.trim_start();
                if let Some(list) = default.strip_prefix('=') {
                    assert!(
                        !list.contains("dispatch_probe"),
                        "`dispatch_probe` is enabled by default, so production \
                         builds carry the counters: {line}"
                    );
                }
            }
        }
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

    /// The exported names must match `Phase::name`, in order.
    ///
    /// Two sources of truth for the same list is exactly how the harness came
    /// to mislabel two phases; this makes the duplicate a checked one.
    #[cfg(feature = "dispatch_probe")]
    #[test]
    fn exported_phase_names_match_the_enum() {
        for p in Phase::ALL.into_iter() {
            let ptr = nxrt_dispatch_probe_phase_name(p as usize);
            assert!(!ptr.is_null(), "{} has no exported name", p.name());
            // SAFETY: the export returns a 'static NUL-terminated string.
            let got = unsafe { core::ffi::CStr::from_ptr(ptr) };
            assert_eq!(got.to_str().unwrap(), p.name(), "name mismatch");
        }
        let last = nxrt_dispatch_probe_phase_name(UNATTRIBUTED);
        assert!(!last.is_null());
        // SAFETY: as above.
        assert_eq!(
            unsafe { core::ffi::CStr::from_ptr(last) }.to_str().unwrap(),
            "unattributed"
        );
        assert!(nxrt_dispatch_probe_phase_name(ALLOC_BUCKETS).is_null());
    }

    /// The exported event names must match `Event::name`, in order.
    ///
    /// The analog of `exported_phase_names_match_the_enum`, absent until the
    /// harness's hard-coded list had drifted far enough to mislabel three of
    /// five counters. Ordering is the whole assertion: the defect was never a
    /// missing name, it was correct names against the wrong indices, which an
    /// arity check cannot see.
    #[cfg(feature = "dispatch_probe")]
    #[test]
    fn exported_event_names_match_the_enum() {
        for e in Event::ALL.into_iter() {
            let ptr = nxrt_dispatch_probe_event_name(e as usize);
            assert!(!ptr.is_null(), "{} has no exported name", e.name());
            // SAFETY: the export returns a 'static NUL-terminated string.
            let got = unsafe { core::ffi::CStr::from_ptr(ptr) };
            assert_eq!(
                got.to_str().unwrap(),
                e.name(),
                "event {} exports the wrong name at index {}",
                e.name(),
                e as usize
            );
        }
        assert!(nxrt_dispatch_probe_event_name(Event::COUNT).is_null());
    }

    /// The C entry point is what the cdylib harness uses; it must refuse a
    /// buffer it would overrun rather than writing past the end.
    #[cfg(feature = "dispatch_probe")]
    #[test]
    fn c_snapshot_refuses_a_short_buffer() {
        let need = SNAPSHOT_LEN;
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
        ("kernel_ctx.rs", include_str!("kernel_ctx.rs"), 15, 16),
        ("status.rs", include_str!("status.rs"), 1, 1),
        ("host_pool.rs", include_str!("host_pool.rs"), 2, 2),
    ];

    /// Source with every `#[cfg(test)]` item removed — test scaffolding names
    /// ORT members too (the fake `OrtApi` in `kernel_ctx`, the fake thread
    /// pools here), and instrumenting a fake would prove nothing.
    ///
    /// This used to truncate at the *first* `#[cfg(test)]`, which silently
    /// dropped every real FFI call below it. A single test-only helper placed
    /// beside the function it exercises was enough to hide three instrumented
    /// ORT members from this audit while the count still matched a lowered
    /// expectation. Excluding items individually means where a test helper sits
    /// in the file cannot change what gets audited.
    ///
    /// Items are recognised by a `#[cfg(...test...)]` attribute at column 0 and
    /// skipped through to the next closing brace at column 0, which is how
    /// rustfmt lays out every top-level item in this crate.
    fn production_source(src: &str) -> String {
        let mut out = String::with_capacity(src.len());
        // `None` = emitting. `Some(depth)` = inside a `#[cfg(test)]` item, with
        // the running brace depth of that item.
        let mut skip: Option<i32> = None;
        for line in src.lines() {
            if let Some(depth) = skip.as_mut() {
                // An item is over when its braces balance. A non-brace item --
                // `#[cfg(test)] use ...;`, or a `const` -- balances on its own
                // first line and ends there. Scanning instead for the next
                // column-0 `}` swallowed every line up to some unrelated item's
                // close, which is the same silent drop this function exists to
                // prevent, just one case narrower.
                let opens = line.matches('{').count() as i32;
                let closes = line.matches('}').count() as i32;
                *depth += opens - closes;
                if *depth <= 0 && !(opens == 0 && closes == 0 && line.trim_end().is_empty()) {
                    skip = None;
                }
                continue;
            }
            if line.starts_with("#[cfg(") && line.contains("test") {
                // A `#[cfg(...)]` that does not close on its own line would
                // leave the item unrecognised, so refuse rather than guess.
                assert!(
                    line.ends_with(")]"),
                    "multi-line #[cfg(...test...)] attribute is not supported \
                     by this audit; keep it on one line: {line}"
                );
                skip = Some(0);
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
        assert!(
            skip.is_none(),
            "a #[cfg(test)] item never closed; the audit would have silently \
             dropped the rest of the file"
        );
        out
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
                // Skip `..Struct::new()` and `..Type` — struct-update and
                // range syntax. A field access is never preceded by a dot, so
                // this can only be one of those, never an `OrtApi` member.
                if i > 0 && b[i - 1] == b'.' {
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

    /// A `#[cfg(test)]` item that has no braces -- a `use`, a `const` -- must
    /// end at its own line. Skipping to the next column-0 `}` instead swallowed
    /// every production line in between, silently dropping real FFI call sites
    /// from this audit while it still reported a pass. Found in review.
    #[test]
    fn a_brace_less_cfg_test_item_does_not_swallow_the_code_after_it() {
        let src = [
            "#[cfg(test)]",
            "use crate::testkit::FakeApi;",
            "pub unsafe fn read_thing(api: &ort::OrtApi) {",
            "    dispatch_probe::ort_call();",
            "    let _ = api.GetTensorData;",
            "}",
            "fn other() {}",
        ]
        .join("\n");
        let prod = production_source(&src);
        assert!(
            !prod.contains("FakeApi"),
            "the test-only `use` was not excluded: {prod}"
        );
        assert!(
            prod.contains("read_thing") && prod.contains("fn other"),
            "production code after a brace-less #[cfg(test)] item was dropped: {prod}"
        );
        assert_eq!(
            prod.matches("dispatch_probe::ort_call()").count(),
            1,
            "the audit lost a real FFI call site: {prod}"
        );
        assert_eq!(ort_members(&prod), vec!["GetTensorData".to_string()]);
    }

    /// The ordinary case: a `#[cfg(test)] mod` and everything in it goes, and
    /// production code on the far side of it stays.
    #[test]
    fn a_cfg_test_module_is_excluded_without_truncating_the_file() {
        let src = [
            "fn before(api: &ort::OrtApi) { let _ = api.GetTensorData; }",
            "#[cfg(test)]",
            "mod tests {",
            "    fn fake(api: &ort::OrtApi) { let _ = api.KernelContext_GetInput; }",
            "}",
            "fn after(api: &ort::OrtApi) { let _ = api.GetTensorMutableData; }",
        ]
        .join("\n");
        let prod = production_source(&src);
        assert_eq!(
            ort_members(&prod),
            vec![
                "GetTensorData".to_string(),
                "GetTensorMutableData".to_string()
            ],
            "got {prod}"
        );
    }

    /// `#[cfg(all(test, feature = "..."))]` is a test item too, and must be
    /// excluded on the same terms.
    #[test]
    fn a_feature_gated_test_module_is_also_excluded() {
        let src = [
            "fn before(api: &ort::OrtApi) { let _ = api.GetTensorData; }",
            "#[cfg(all(test, feature = \"dispatch_probe\"))]",
            "mod dispatch_cost {",
            "    fn fake(api: &ort::OrtApi) { let _ = api.KernelContext_GetInput; }",
            "}",
            "fn after(api: &ort::OrtApi) { let _ = api.GetTensorMutableData; }",
        ]
        .join("\n");
        let prod = production_source(&src);
        assert!(!prod.contains("KernelContext_GetInput"), "got {prod}");
        assert!(prod.contains("fn after"), "got {prod}");
    }

    #[test]
    fn every_ort_entry_point_is_accounted_for() {
        for (name, src, want_members, want_calls) in EXPECTED {
            let prod = production_source(src);
            let prod = prod.as_str();
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
        let m = ort_members(&production_source(include_str!("kernel_ctx.rs")));
        assert!(m.contains(&"KernelContext_GetInput".to_string()), "{m:?}");
        assert!(m.contains(&"GetTensorData".to_string()), "{m:?}");
        assert!(
            !m.iter().any(|s| s.contains("::")),
            "path segments must not be mistaken for member accesses: {m:?}"
        );
    }
}
