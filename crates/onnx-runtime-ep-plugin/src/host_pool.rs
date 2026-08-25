//! ORT's intra-op thread pool, exposed to our kernels for one compute call.
//!
//! Our elementwise CPU kernels split long slices across a `rayon` pool. Under
//! an ORT session that pool is a *second* pool on the same cores, and ORT's
//! intra-op workers spin, so the two fight: at `intra_op_num_threads = 16` a
//! 1 Mi `Sqrt` went 252 us serial to 777 us split, and `FastGelu` 1040 us to
//! 1479 us. Parallelising made every op slower, and the more threads the user
//! asked for the worse it got.
//!
//! `OrtApi::KernelContext_ParallelFor` runs a callback on the session's own
//! intra-op pool. Pointing our chunk split at it leaves exactly one pool on
//! the machine, sized by whatever the user configured — so the contention is
//! gone by construction, and we stop spending threads the caller never
//! offered us. This module builds the
//! [`HostParallel`](onnx_runtime_ep_api::HostParallel) that does it, and
//! [`scope`] installs it for the dynamic extent of one compute call.
//!
//! Sessions whose ORT is older than 1.17 have a null `KernelContext_ParallelFor`
//! and get no handle at all, which leaves the kernels on their existing rayon
//! path.
//!
//! # Ownership of the `void*`
//!
//! The handle ORT is given is the address of one [`HostPool`], and that
//! address is *aliased*: ORT reads through it for the whole compute call while
//! this module still owns the allocation. Rust's ownership rules do not
//! tolerate that on a `Box` — see [`PoolSlot`] for what goes wrong, and for
//! why the allocation is owned as a raw pointer between `install` and drop.
//!
//! Covered by the Miri lane (`--lib host_pool::`) under Stacked Borrows,
//! which is the only tool here that checks the argument rather than the
//! behaviour on one machine.

use core::ffi::c_void;
use core::ptr::NonNull;
use onnx_genai_ort_sys as ort;
use onnx_runtime_ep_api::HostParallel;
use onnx_runtime_ep_api::host_parallel;
use onnx_runtime_ep_api::host_parallel::HOST_HELPED;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// What [`ort_parallel_for`] needs to reach ORT, behind one `void*`.
struct HostPool {
    parallel_for: unsafe extern "C" fn(
        *const ort::OrtKernelContext,
        Option<unsafe extern "C" fn(*mut c_void, usize)>,
        usize,
        usize,
        *mut c_void,
    ) -> *mut ort::OrtStatus,
    release_status: Option<unsafe extern "C" fn(*mut ort::OrtStatus)>,
    ctx: *mut ort::OrtKernelContext,
    /// Where this session records whether ORT's pool has ever helped, and
    /// when to ask again.
    probe: *const AtomicU32,
    /// Which `install` filled this slot. Zero-sized in release; see [`Epoch`].
    epoch: Epoch,
}

/// Serial number of one `install` on one thread, in debug builds.
///
/// The slot the `void*` points at is reused across `Run`s, so a handle held
/// past its compute call finds a *live* slot instead of a freed one. Before
/// #1526 that was a dangling pointer and any sanitiser caught it; with reuse
/// it is a well-formed read of a retired slot, and the dispatch would go to
/// whatever `OrtKernelContext` it still holds. Stamping the slot with the
/// install that filled it, and checking the stamp on every dispatch, keeps
/// that loud.
///
/// # What it catches, and what it does not
///
/// It refuses a dispatch whose slot was not filled by the install that is
/// currently live on this thread: a handle used after its compute call
/// returned, one used from inside a *different* (nested or later) install
/// whose slot differs, and one used on a thread that never installed anything.
///
/// It cannot see a handle used from inside a later `Run` **on the same thread
/// that reused the same slot**, because at that point the stale handle and the
/// live one are the same address and the same stamp. Distinguishing those
/// would mean widening [`HostParallel`] — which is `Copy`, pointer-sized and
/// read on the dispatch path — for a case no consumer can currently produce.
/// Every consumer takes the handle from `host_parallel::current()` and uses it
/// synchronously; that is the invariant, and this is a backstop for it, not a
/// replacement.
///
/// Zero-sized and every operation a no-op in release, so the dispatch path and
/// `HostPool`'s layout are byte-identical to having no guard at all — see
/// `the_staleness_guard_is_free_in_release`. Debug builds, `cargo test` and
/// Miri all have `debug_assertions`, which is where it is worth paying for.
#[cfg(debug_assertions)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Epoch(u64);

/// See the `debug_assertions` definition.
#[cfg(not(debug_assertions))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Epoch;

#[cfg(debug_assertions)]
thread_local! {
    /// Serial to stamp the next `install` on this thread with. Thread-local
    /// because the slot it identifies is, so it never needs to be an atomic.
    static NEXT_EPOCH: std::cell::Cell<u64> = const { std::cell::Cell::new(1) };

    /// Stamp of the innermost `install` currently live on this thread. Zero
    /// when there is none, which no stamp ever takes.
    static LIVE_EPOCH: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

impl Epoch {
    /// The stamp of "no install is live", which [`Epoch::next`] never issues.
    #[cfg(debug_assertions)]
    const fn none() -> Self {
        Self(0)
    }

    /// See the `debug_assertions` definition.
    #[cfg(not(debug_assertions))]
    const fn none() -> Self {
        Self
    }

    /// A stamp no live slot on this thread carries.
    #[cfg(debug_assertions)]
    fn next() -> Self {
        Self(NEXT_EPOCH.with(|n| {
            let e = n.get();
            n.set(e.wrapping_add(1).max(1));
            e
        }))
    }

    /// See the `debug_assertions` definition.
    #[cfg(not(debug_assertions))]
    const fn next() -> Self {
        Self
    }

    /// Makes this the live stamp, returning the one it displaced so a nested
    /// install can put it back.
    #[cfg(debug_assertions)]
    fn enter(self) -> Self {
        Self(LIVE_EPOCH.with(|live| live.replace(self.0)))
    }

    /// See the `debug_assertions` definition.
    #[cfg(not(debug_assertions))]
    const fn enter(self) -> Self {
        Self
    }

    /// Restores this as the live stamp.
    #[cfg(debug_assertions)]
    fn leave(self) {
        LIVE_EPOCH.with(|live| live.set(self.0));
    }

    /// See the `debug_assertions` definition.
    #[cfg(not(debug_assertions))]
    const fn leave(self) {}

    /// Panics if the slot carrying this stamp is not the installed one.
    #[cfg(debug_assertions)]
    fn assert_live(self) {
        assert_eq!(
            self.0,
            LIVE_EPOCH.with(std::cell::Cell::get),
            "a host-pool handle outlived the compute call that installed it"
        );
    }

    /// See the `debug_assertions` definition.
    #[cfg(not(debug_assertions))]
    const fn assert_live(self) {}
}

/// The closure a dispatch is running, plus somewhere to record what happened.
struct Task<'a> {
    body: &'a (dyn Fn(usize) + Sync),
    panicked: AtomicBool,
    /// The thread that called `KernelContext_ParallelFor`, while we still do
    /// not know whether ORT has any workers. `None` once the answer is known,
    /// which turns the observation below into a single relaxed load.
    caller: Option<std::thread::ThreadId>,
    /// Set when a body ran somewhere other than `caller`.
    saw_another_thread: AtomicBool,
    /// Set once the calling thread has held its first index open, to give
    /// ORT's workers a chance to claim one of the others.
    stalled: AtomicBool,
}

/// Trampoline handed to ORT: one index of one dispatch.
///
/// # Safety
///
/// `usr_data` must be the `*mut Task` that [`ort_parallel_for`] passed to
/// `KernelContext_ParallelFor`, and must outlive the call — which it does,
/// because that call is blocking.
unsafe extern "C" fn run_index(usr_data: *mut c_void, index: usize) {
    // A Rust panic must not unwind into ORT's frames: that is undefined
    // behaviour, and this callback is called from C++. Catch it here, record
    // it, and let `ort_parallel_for` re-raise on the calling thread once every
    // worker has finished.
    let task = unsafe { &*(usr_data.cast::<Task<'_>>()) };
    if let Some(caller) = task.caller {
        if std::thread::current().id() == caller {
            // A probe that the calling thread simply drains tells us nothing:
            // ORT hands its indices out dynamically, so a pool of sixteen
            // looks exactly like a pool of one when the chunks are small.
            // Hold the first index this thread takes for long enough that a
            // worker which is going to help has time to claim another. Paid
            // only on probe dispatches, and only until one of them answers.
            if !task.stalled.swap(true, Ordering::Relaxed) {
                #[cfg(test)]
                probe_stall_counter::record();
                stall_until(PROBE_STALL, &task.saw_another_thread);
            }
        } else {
            task.saw_another_thread.store(true, Ordering::Relaxed);
        }
    }
    if std::panic::catch_unwind(AssertUnwindSafe(|| (task.body)(index))).is_err() {
        task.panicked.store(true, Ordering::Relaxed);
    }
}

/// How many probe stalls this thread has actually paid.
///
/// Test-only, and thread-local rather than global: the stall runs exclusively
/// on the thread that called `KernelContext_ParallelFor` (that is what the
/// `caller` comparison above establishes), so a thread-local is exact and
/// immune to a concurrently-running test in the same binary bumping it.
///
/// This exists because the two tests below used to assert "the stall ran once,
/// not once per index" through a **wall clock**, and that instrument does not
/// survive a loaded, coverage-instrumented CI runner: the upper bound was
/// `PROBE_STALL * 4` = 1.6 ms and `Rust coverage (Windows x86_64)` measured
/// 2.0171 ms on correct code (run 32729806697, `main` @ `332077afb`). The
/// property being asserted differs by 64x -- one stall is 400 us, one per index
/// would be 25.6 ms -- so it was never a close call that needed measuring; it
/// was a count, asserted with a stopwatch. Count it. See #2018.
#[cfg(test)]
mod probe_stall_counter {
    use std::cell::Cell;

    thread_local! {
        static STALLS: Cell<u32> = const { Cell::new(0) };
    }

    pub(super) fn record() {
        STALLS.with(|stalls| stalls.set(stalls.get() + 1));
    }

    /// Stalls paid on this thread so far.
    pub(super) fn count() -> u32 {
        STALLS.with(Cell::get)
    }
}

/// Longest the calling thread holds its first index open on a probe.
///
/// A deadline, not a duration: the wait ends the moment a worker is seen, so a
/// pool with threads pays its wake-up latency and no more. The full 400 us is
/// only ever paid by a pool that has nobody to wake, and only for the
/// [`PROBE_BURST`](onnx_runtime_ep_api::host_parallel::PROBE_BURST) probes of
/// a fused node's opening burst plus one dispatch in a thousand after that.
///
/// It has to cover a *loaded* machine's wake-up, not an idle one. At 100 us,
/// probing was decisive on a quiet box (15 of 15 sessions latched within three
/// probes) but not on one running at load 5-10, where several sessions never
/// latched at all and kept their work on the wrong pool.
const PROBE_STALL: core::time::Duration = core::time::Duration::from_micros(400);

/// Busy-waits until `evidence` is set, or `how_long` has passed.
///
/// Deliberately not a sleep: this runs on ORT's calling thread, and parking it
/// would hand the core to the very workers whose presence is being tested.
fn stall_until(how_long: core::time::Duration, evidence: &AtomicBool) {
    let until = std::time::Instant::now() + how_long;
    while !evidence.load(Ordering::Relaxed) && std::time::Instant::now() < until {
        std::hint::spin_loop();
    }
}

/// Runs `body(0..total)` on the ORT session's intra-op pool.
///
/// # Safety
///
/// `host` must be the `*mut HostPool` that [`scope`] built, still valid — i.e.
/// this must be reached from inside the compute call that installed it.
unsafe fn ort_parallel_for(host: *mut c_void, total: usize, body: &(dyn Fn(usize) + Sync)) {
    // SAFETY: `install` handed ORT the address of a slot it owns for the whole
    // compute call, and the slot is only ever written through the same raw
    // pointer, so no reference to it is live here. A shared reference is all
    // this function ever needs.
    let pool = unsafe { &*(host.cast::<HostPool>()) };
    // Before anything else, and before any foreign frame is entered, so the
    // panic cannot unwind through ORT.
    pool.epoch.assert_live();
    // Watch which threads run the bodies until one of them is not ours. That
    // sighting is the whole answer -- see `HostParallel::prefer_host` -- and
    // once it is in, stop paying for `thread::current()` on every index.
    //
    // While the answer is missing, every dispatch that gets here is a probe:
    // eight empty indices sent for their scheduling behaviour alone. Real work
    // only reaches this function once the pool has proved it has workers, or
    // when there is no cell to record the answer in.
    let observing = total > 1 && !unsafe { pool.helped() };
    let task = Task {
        body,
        panicked: AtomicBool::new(false),
        caller: observing.then(|| std::thread::current().id()),
        saw_another_thread: AtomicBool::new(false),
        stalled: AtomicBool::new(false),
    };
    // `num_batch = 0` means "no limit": ORT gives every index its own task and
    // its workers claim them dynamically. That is what we want, because we cut
    // the slice by size rather than by thread count — we cannot ask ORT how
    // wide its pool is, so we hand it enough pieces to fill any pool and let
    // it decide how many to run at once.
    let status = unsafe {
        crate::dispatch_probe::ort_call();
        (pool.parallel_for)(
            pool.ctx,
            Some(run_index),
            total,
            0,
            (&raw const task).cast::<c_void>().cast_mut(),
        )
    };
    if observing
        && status.is_null()
        && task.saw_another_thread.load(Ordering::Relaxed)
        && let Some(probe) = unsafe { pool.probe() }
    {
        // Recorded per session, so a process holding both a one-thread and a
        // sixteen-thread session gets the right answer for each.
        probe.store(HOST_HELPED, Ordering::Relaxed);
    }
    if !status.is_null() {
        // The dispatch itself failed. Every index still has to run or the
        // output tensor keeps whatever was in the buffer, so fall back to
        // running them here rather than silently returning short.
        if let Some(release) = pool.release_status {
            crate::dispatch_probe::ort_call();
            unsafe { release(status) };
        }
        for index in 0..total {
            body(index);
        }
        return;
    }
    assert!(
        !task.panicked.load(Ordering::Relaxed),
        "a kernel panicked on an ORT intra-op thread"
    );
}

impl HostPool {
    /// The session's probe cell, if it has one.
    ///
    /// # Safety
    ///
    /// The pointer must still be valid, which `install`'s contract requires
    /// for as long as the guard is alive.
    unsafe fn probe(&self) -> Option<&AtomicU32> {
        (!self.probe.is_null()).then(|| unsafe { &*self.probe })
    }

    /// Whether this session has already seen ORT run a body on its own thread.
    ///
    /// # Safety
    ///
    /// As [`HostPool::probe`].
    unsafe fn helped(&self) -> bool {
        unsafe { self.probe() }.is_none_or(|cell| cell.load(Ordering::Relaxed) == HOST_HELPED)
    }
}

/// The heap slot the `void*` handle points at, owned as a raw pointer.
///
/// # Why not `Box`
///
/// A `Box` asserts that nothing else points at its contents, and Rust
/// reasserts that on every move of one: into the guard, out of `install`, out
/// of `Option::take` in `Drop`. But the whole purpose of this slot is to be
/// aliased — ORT is handed its address and reads through it for the length of
/// the compute call. The first such read invalidates the `Box`'s claim, and
/// the next move of the `Box` uses a claim that no longer exists.
///
/// That is undefined behaviour, not a technicality, and Miri's Stacked Borrows
/// reports it exactly:
///
/// ```text
/// trying to retag from <N> for Unique permission at alloc[0x0], but that tag
/// does not exist in the borrow stack for this location
///   ... <N> was created by a Unique retag           (the `Box` in `install`)
///   ... <N> was later invalidated by a SharedReadOnly retag
///                                                    (the read in `ort_parallel_for`)
/// ```
///
/// So Rust's ownership of the allocation is given up for exactly the window in
/// which it is shared: [`Box::into_raw`] on the way in, [`Box::from_raw`] in
/// `Drop` on the way out. In between there is no `Box` and no reference to
/// conflict with, and the raw pointer is never retagged however many times it
/// is moved. It is still an owning handle — this type frees the slot — so
/// teardown is unchanged, including at thread exit.
struct PoolSlot(NonNull<HostPool>);

impl PoolSlot {
    /// A fresh slot holding `pool`.
    fn new(pool: HostPool) -> Self {
        crate::dispatch_probe::count(crate::dispatch_probe::Event::DispatchAlloc);
        // SAFETY: `Box::into_raw` never returns null.
        Self(unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(pool))) })
    }

    /// Overwrites the slot for the next `Run`.
    fn refill(&mut self, pool: HostPool) {
        // `write` does not drop what it overwrites; nothing here needs it to.
        const {
            assert!(
                !core::mem::needs_drop::<HostPool>(),
                "`PoolSlot::refill` would leak a `HostPool` that owns anything"
            );
        }
        // SAFETY: the slot came from `Box::new`, so it is live, aligned and
        // initialised, and this thread's guard for it has been dropped — no
        // reference to it exists. Written through the raw pointer rather than
        // through a `&mut` because that asserts nothing about aliasing at all:
        // Miri passes either way today, since the previous `Run`'s reads are
        // finished, but the narrower operation is the one that stays correct
        // if that ever stops being true.
        unsafe { self.0.write(pool) };
    }

    /// The address handed to ORT.
    const fn as_ptr(&self) -> *mut HostPool {
        self.0.as_ptr()
    }
}

impl Drop for PoolSlot {
    fn drop(&mut self) {
        // SAFETY: pairs with the `Box::into_raw` in `new`. The slot is freed
        // only here, and `Guard`'s drop order has already removed the handle
        // that pointed at it.
        drop(unsafe { Box::from_raw(self.0.as_ptr()) });
    }
}

/// ORT's pool, installed on this thread until the guard is dropped.
///
/// Field order is the safety argument: `installed` is dropped first, so the
/// thread-local handle is gone before `pool` — the allocation it points at —
/// is recycled or freed.
pub struct Guard {
    installed: Option<host_parallel::Installed>,
    pool: Option<PoolSlot>,
    /// The stamp this install displaced, restored on drop so a nested install
    /// hands the outer one back. See [`Epoch`].
    outer_epoch: Epoch,
}

impl Guard {
    /// A guard that installs nothing, for hosts that offer no pool.
    const fn inert() -> Self {
        Self {
            installed: None,
            pool: None,
            outer_epoch: Epoch::none(),
        }
    }

    /// Whether a handle is actually installed, for tests and diagnostics.
    #[must_use]
    pub const fn is_installed(&self) -> bool {
        self.installed.is_some()
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        // Explicit, and in this order, so a later field reshuffle cannot turn
        // the handle into a dangling pointer without failing to compile.
        drop(self.installed.take());
        if let Some(pool) = self.pool.take() {
            self.outer_epoch.leave();
            recycle_pool(pool);
        }
    }
}

thread_local! {
    /// One retired `HostPool` slot per thread, kept for the next `Run`.
    ///
    /// The slot exists only to give the `void*` handle a stable address; its
    /// contents are overwritten on every install. Allocating it fresh each
    /// time was one `malloc` and one `free` per `Run` for 32 bytes, on a path
    /// where the whole fixed cost is a couple of microseconds.
    ///
    /// Thread-local because `Compute` runs on whichever thread ORT calls it
    /// from and the slot never escapes that thread. `Cell` rather than
    /// `RefCell`: the only operations are take and replace, so there is
    /// nothing to borrow and no way to panic while holding it.
    ///
    /// # The invariant reuse rests on
    ///
    /// No one may hold the `void*` handle past the compute call that installed
    /// it. That was already required, but freeing the slot made a violation
    /// loud: the stale handle pointed at freed memory, which a sanitiser
    /// catches. Reuse would make it quiet instead — a stale handle finds a
    /// live slot belonging to a *later* `Run` and would silently compute
    /// against the wrong context. [`Epoch`] is what keeps it loud: the slot
    /// carries the serial of the install that filled it, and a dispatch
    /// through a stale handle panics rather than proceeding. Every consumer
    /// today takes the handle from `host_parallel::current()` and uses it
    /// synchronously, within the compute extent; keep it that way.
    static POOL_CACHE: std::cell::Cell<Option<PoolSlot>> =
        const { std::cell::Cell::new(None) };
}

/// A slot to build a `HostPool` in, reusing this thread's retired one.
///
/// Falls back to a fresh allocation when the cache is empty, which covers both
/// the first call on a thread and a re-entrant one whose outer guard is still
/// holding the cached slot.
fn take_pool(pool: HostPool) -> PoolSlot {
    match POOL_CACHE.with(std::cell::Cell::take) {
        Some(mut slot) => {
            slot.refill(pool);
            slot
        }
        None => PoolSlot::new(pool),
    }
}

/// Hand a retired slot back for the next `Run` on this thread.
///
/// Drops it if the cache is already occupied, so a re-entrant call cannot make
/// the cache grow: at most one slot per thread is ever retained.
fn recycle_pool(pool: PoolSlot) {
    POOL_CACHE.with(|cache| {
        if let Some(existing) = cache.replace(Some(pool)) {
            drop(existing);
        }
    });
}

/// Installs ORT's pool on the calling thread until the returned guard drops.
///
/// Returns an inert guard when this ORT has no `KernelContext_ParallelFor`
/// (pre-1.17) or the context is null, which leaves the kernels on their
/// existing rayon path.
///
/// # Safety
///
/// `api` must be a valid `OrtApi`, and `ctx` a valid `OrtKernelContext` that
/// stays valid for as long as the guard is alive. Because the handle is
/// thread-local and the guard is confined to the compute call's frame, it
/// cannot be reached from a later call whose context has been freed.
#[must_use = "dropping the guard immediately uninstalls the pool"]
pub unsafe fn install(
    api: &ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
    probe: &AtomicU32,
) -> Guard {
    let (Some(parallel_for), false) = (api.KernelContext_ParallelFor, ctx.is_null()) else {
        return Guard::inert();
    };
    let epoch = Epoch::next();
    let pool = take_pool(HostPool {
        parallel_for,
        release_status: api.ReleaseStatus,
        ctx,
        probe: core::ptr::from_ref(probe),
        epoch,
    });
    // SAFETY: the slot outlives the handle (see `Guard`'s drop order), and
    // `ort_parallel_for` only ever reads the pointer back as a `*mut HostPool`.
    // The address is taken from the raw pointer `PoolSlot` holds rather than
    // through a reference, so the handle does not alias any borrow.
    let handle = unsafe {
        HostParallel::new(
            pool.as_ptr().cast::<c_void>(),
            ort_parallel_for,
            core::ptr::from_ref(probe),
        )
    };
    Guard {
        installed: Some(host_parallel::Installed::new(handle)),
        pool: Some(pool),
        outer_epoch: epoch.enter(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::host_parallel::{PROBE_BURST, PROBE_MAX, PROBE_MIN};
    use std::sync::atomic::AtomicUsize;

    /// Stands in for ORT: runs every index inline and reports success.
    ///
    /// # Safety
    ///
    /// Matches the ABI ORT expects; `usr_data` is passed straight through.
    unsafe extern "C" fn inline_parallel_for(
        _ctx: *const ort::OrtKernelContext,
        body: Option<unsafe extern "C" fn(*mut c_void, usize)>,
        total: usize,
        _num_batch: usize,
        usr_data: *mut c_void,
    ) -> *mut ort::OrtStatus {
        let body = body.expect("ORT is always given a callback");
        for index in 0..total {
            unsafe { body(usr_data, index) };
        }
        core::ptr::null_mut()
    }

    /// [`inline_parallel_for`] that counts the dispatches it is handed.
    ///
    /// # Safety
    ///
    /// As [`inline_parallel_for`].
    unsafe extern "C" fn counted_inline_parallel_for(
        ctx: *const ort::OrtKernelContext,
        body: Option<unsafe extern "C" fn(*mut c_void, usize)>,
        total: usize,
        num_batch: usize,
        usr_data: *mut c_void,
    ) -> *mut ort::OrtStatus {
        INLINE_DISPATCHES.fetch_add(1, Ordering::Relaxed);
        unsafe { inline_parallel_for(ctx, body, total, num_batch, usr_data) }
    }

    /// Dispatches [`counted_inline_parallel_for`] has seen. Only
    /// `the_probe_and_the_latch_agree` reads it, and it resets it first.
    static INLINE_DISPATCHES: AtomicUsize = AtomicUsize::new(0);

    /// Stands in for an ORT that refuses the dispatch.
    ///
    /// # Safety
    ///
    /// Returns a non-null status that is never dereferenced, because the
    /// `release_status` hook in these tests is `None`.
    unsafe extern "C" fn failing_parallel_for(
        _ctx: *const ort::OrtKernelContext,
        _body: Option<unsafe extern "C" fn(*mut c_void, usize)>,
        _total: usize,
        _num_batch: usize,
        _usr_data: *mut c_void,
    ) -> *mut ort::OrtStatus {
        core::ptr::dangling_mut()
    }

    /// Stands in for an ORT whose intra-op pool has real workers.
    ///
    /// # Safety
    ///
    /// Matches the ABI ORT expects; `usr_data` is passed straight through.
    unsafe extern "C" fn threaded_parallel_for(
        _ctx: *const ort::OrtKernelContext,
        body: Option<unsafe extern "C" fn(*mut c_void, usize)>,
        total: usize,
        _num_batch: usize,
        usr_data: *mut c_void,
    ) -> *mut ort::OrtStatus {
        /// Carries the dispatch pointer into the scoped threads.
        ///
        /// Deliberately not `usr_data as usize` and back: an integer round
        /// trip strips the pointer's provenance, and Miri says so — "this
        /// program is using integer-to-pointer casts ... Miri might miss
        /// pointer bugs". Aliasing is the entire reason this module is in the
        /// Miri lane, so the helper must not be the thing that blinds it.
        #[derive(Clone, Copy)]
        struct Dispatch(*mut c_void);
        // SAFETY: this is the `Task` the dispatch owns. ORT hands the same
        // pointer to every worker, and `thread::scope` joins them all before
        // `ort_parallel_for` returns, so it outlives every use.
        unsafe impl Send for Dispatch {}
        impl Dispatch {
            /// Reads the pointer back out. A method, not a field access, so
            /// the closures below capture the `Send` wrapper rather than
            /// destructuring straight to the `*mut c_void` inside it.
            const fn get(self) -> *mut c_void {
                self.0
            }
        }

        let body = body.expect("ORT is always given a callback");
        let usr = Dispatch(usr_data);
        std::thread::scope(|scope| {
            for index in 0..total {
                scope.spawn(move || unsafe { body(usr.get(), index) });
            }
        });
        core::ptr::null_mut()
    }

    /// A `HostPool` built by hand, for the tests that dispatch through
    /// [`ort_parallel_for`] directly rather than through [`install`].
    ///
    /// Stamped [`Epoch::none`], which is what `LIVE_EPOCH` reads as when no
    /// install is live — so the staleness guard passes, and would fire if one
    /// of these were ever dispatched from inside a real install.
    fn pool(
        parallel_for: unsafe extern "C" fn(
            *const ort::OrtKernelContext,
            Option<unsafe extern "C" fn(*mut c_void, usize)>,
            usize,
            usize,
            *mut c_void,
        ) -> *mut ort::OrtStatus,
        probe: &AtomicU32,
    ) -> HostPool {
        HostPool {
            parallel_for,
            release_status: None,
            ctx: core::ptr::null_mut(),
            probe: core::ptr::from_ref(probe),
            epoch: Epoch::none(),
        }
    }

    #[test]
    fn every_index_runs_once() {
        let width = AtomicU32::new(0);
        let mut pool = pool(inline_parallel_for, &width);
        let seen: Vec<AtomicUsize> = (0..5).map(|_| AtomicUsize::new(0)).collect();
        unsafe {
            ort_parallel_for((&raw mut pool).cast::<c_void>(), seen.len(), &|index| {
                seen[index].fetch_add(1, Ordering::Relaxed);
            });
        }
        assert!(seen.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }

    /// The sighting that matters: a body ran somewhere other than the thread
    /// that dispatched it, which only a pool with workers can do.
    #[test]
    fn a_threaded_dispatch_latches_the_host_in() {
        let probe = AtomicU32::new(0);
        let mut pool = pool(threaded_parallel_for, &probe);
        unsafe { ort_parallel_for((&raw mut pool).cast::<c_void>(), 4, &|_| {}) };
        assert_eq!(probe.load(Ordering::Relaxed), HOST_HELPED);
    }

    /// An ORT that ran everything inline has told us nothing. It happens on
    /// 16-thread sessions too -- 35% of dispatches, in runs of up to 60 --
    /// because ORT hands its indices out dynamically and the calling thread
    /// can drain the queue before a worker wakes. Concluding "no workers" from
    /// that would start our pool alongside ORT's, the 3-10x pathology.
    #[test]
    fn inline_dispatches_never_decide_anything() {
        let probe = AtomicU32::new(0);
        let mut pool = pool(inline_parallel_for, &probe);
        for _ in 0..256 {
            unsafe { ort_parallel_for((&raw mut pool).cast::<c_void>(), 4, &|_| {}) };
        }
        assert_eq!(probe.load(Ordering::Relaxed), 0, "silence is not evidence");
    }

    /// Once latched, later dispatches stop paying for the observation, and
    /// nothing can unlatch it.
    #[test]
    fn a_latched_host_is_not_revisited() {
        let probe = AtomicU32::new(HOST_HELPED);
        let mut pool = pool(inline_parallel_for, &probe);
        for _ in 0..16 {
            unsafe { ort_parallel_for((&raw mut pool).cast::<c_void>(), 4, &|_| {}) };
        }
        assert_eq!(probe.load(Ordering::Relaxed), HOST_HELPED);
    }

    /// One index cannot be split, so it proves nothing either way.
    #[test]
    fn a_single_index_dispatch_is_not_evidence() {
        let probe = AtomicU32::new(0);
        let mut pool = pool(threaded_parallel_for, &probe);
        unsafe { ort_parallel_for((&raw mut pool).cast::<c_void>(), 1, &|_| {}) };
        assert_eq!(probe.load(Ordering::Relaxed), 0);
    }

    /// The stall must not run once a session has latched: it is a probe cost,
    /// not a per-dispatch one.
    ///
    /// Asserted as a **count**, not as elapsed time. The previous version bounded
    /// eight dispatches by a single `PROBE_STALL` (400 us) -- which is a claim
    /// about the machine as much as about the code, and one that an instrumented
    /// or contended runner falsifies while the code is correct. A latched session
    /// pays *zero* stalls; that is the property, and it is exactly countable.
    ///
    /// This also makes the test meaningful under Miri, which the wall-clock
    /// version had to be `#[ignore]`d under: an interpreter three orders of
    /// magnitude slower than native cannot be timed, but it can count.
    #[test]
    fn a_latched_session_pays_no_stall() {
        let probe = AtomicU32::new(HOST_HELPED);
        let mut pool = pool(inline_parallel_for, &probe);
        let before = probe_stall_counter::count();
        for _ in 0..8 {
            unsafe { ort_parallel_for((&raw mut pool).cast::<c_void>(), 4, &|_| {}) };
        }
        assert_eq!(
            probe_stall_counter::count() - before,
            0,
            "a latched session paid a probe stall"
        );
    }

    /// And it must run at most once per probe, however many indices there are.
    ///
    /// Counted rather than timed, for the reason on
    /// [`a_latched_session_pays_no_stall`] -- and this is the test that actually
    /// went red on correct code: `PROBE_STALL * 4` = 1.6 ms against 2.0171 ms
    /// measured on `Rust coverage (Windows x86_64)`. The property is 64 stalls
    /// versus one, a 64x separation, so nothing about it needed a stopwatch.
    ///
    /// The lower bound is *not* replaced by the count, because it asserts a
    /// different thing -- that a probe actually gives workers a window rather
    /// than returning immediately -- and it is one-sided: load can only make
    /// elapsed time larger, never smaller, so it cannot flake the way the upper
    /// bound did. It stays native-only, since a virtual clock under an
    /// interpreter is not the thing it is checking.
    ///
    /// The trade is worth stating rather than leaving implicit: the old upper
    /// bound also happened to cap the wall-clock time of the *whole* dispatch,
    /// and the count does not. A timing regression elsewhere in the dispatch
    /// path would no longer trip these two tests. That ceiling was never what
    /// they were for, and it is precisely the machine-dependent half that
    /// flaked -- but it is coverage given up, not coverage that was redundant.
    #[test]
    fn a_probe_stalls_once_not_once_per_index() {
        let probe = AtomicU32::new(0);
        let mut pool = pool(inline_parallel_for, &probe);
        let before = probe_stall_counter::count();
        #[cfg(not(miri))]
        let started = std::time::Instant::now();
        unsafe { ort_parallel_for((&raw mut pool).cast::<c_void>(), 64, &|_| {}) };
        assert_eq!(
            probe_stall_counter::count() - before,
            1,
            "the stall is per dispatch, not per index"
        );
        #[cfg(not(miri))]
        assert!(
            started.elapsed() >= PROBE_STALL,
            "a probe has to give workers a chance"
        );
    }

    /// A refused dispatch ran on our fallback path, not ORT's pool.
    #[test]
    fn a_refused_dispatch_is_not_evidence() {
        let probe = AtomicU32::new(0);
        let mut pool = pool(failing_parallel_for, &probe);
        for _ in 0..8 {
            unsafe { ort_parallel_for((&raw mut pool).cast::<c_void>(), 4, &|_| {}) };
        }
        assert_eq!(probe.load(Ordering::Relaxed), 0);
    }

    /// End to end through the public seam: a session whose pool helps ends up
    /// preferring the host, and one whose pool never does keeps asking, but
    /// rarely.
    #[test]
    fn the_probe_and_the_latch_agree() {
        // Wired to the real `ort_parallel_for`, so `prefer_host` runs its own
        // probe through it exactly as a kernel would.
        let probe = AtomicU32::new(0);
        let mut threaded = pool(threaded_parallel_for, &probe);
        // SAFETY: `threaded` outlives the handle, and `ort_parallel_for` is
        // the function that pointer was built for.
        let handle = unsafe {
            HostParallel::new(
                (&raw mut threaded).cast::<c_void>(),
                ort_parallel_for,
                core::ptr::from_ref(&probe),
            )
        };
        assert!(
            handle.prefer_host(),
            "a probe the pool answers should switch this dispatch over at once"
        );
        assert!(handle.helped());
        assert!(handle.prefer_host(), "and stay switched over");

        // The same, over a pool that runs everything on the calling thread:
        // it can never prove itself, so the work stays on our own pool.
        let quiet = AtomicU32::new(0);
        let mut inline = pool(counted_inline_parallel_for, &quiet);
        INLINE_DISPATCHES.store(0, Ordering::Relaxed);
        // SAFETY: as above.
        let handle = unsafe {
            HostParallel::new(
                (&raw mut inline).cast::<c_void>(),
                ort_parallel_for,
                core::ptr::from_ref(&quiet),
            )
        };
        let mut borrowed = 0;
        for _ in 0..4096 {
            if handle.prefer_host() {
                borrowed += 1;
            }
        }
        assert!(!handle.helped());
        assert_eq!(
            borrowed, 0,
            "an inline pool never proves itself, so nothing may be handed to it"
        );
        // The cell still has to be asking often enough to recover if the
        // opening burst was unlucky, and not so often that it costs anything.
        // Counting the dispatches the pool actually saw is the only way to say
        // that: reading the settled period back out of the cell would only
        // restate the constant.
        let probes = INLINE_DISPATCHES.load(Ordering::Relaxed);
        let dispatches = 4096;
        assert!(
            probes > usize::try_from(PROBE_BURST).unwrap(),
            "probing stopped after the opening burst ({probes} probes): an \
             unlucky burst could never be recovered from"
        );
        assert!(
            probes <= dispatches / usize::try_from(PROBE_MIN).unwrap(),
            "probed {probes} times in {dispatches} dispatches, more often than \
             the shortest back-off period allows"
        );
        let settled = quiet.load(Ordering::Relaxed) >> 16;
        assert_eq!(settled, PROBE_MAX, "probes should settle at the cap");
    }

    #[test]
    fn a_refused_dispatch_still_runs_every_index() {
        let width = AtomicU32::new(0);
        let mut pool = pool(failing_parallel_for, &width);
        let seen: Vec<AtomicUsize> = (0..5).map(|_| AtomicUsize::new(0)).collect();
        unsafe {
            ort_parallel_for((&raw mut pool).cast::<c_void>(), seen.len(), &|index| {
                seen[index].fetch_add(1, Ordering::Relaxed);
            });
        }
        assert!(
            seen.iter().all(|c| c.load(Ordering::Relaxed) == 1),
            "a short write would leave the output tensor uninitialised"
        );
    }

    #[test]
    fn a_panicking_body_does_not_unwind_into_ort() {
        let width = AtomicU32::new(0);
        let mut pool = pool(inline_parallel_for, &width);
        let ran = AtomicUsize::new(0);
        let unwound = std::panic::catch_unwind(AssertUnwindSafe(|| unsafe {
            ort_parallel_for((&raw mut pool).cast::<c_void>(), 4, &|index| {
                ran.fetch_add(1, Ordering::Relaxed);
                assert_ne!(index, 1, "kernel failed");
            });
        }));
        // The panic surfaces on the calling thread, *after* the dispatch has
        // drained -- every index still ran.
        assert!(unwound.is_err());
        assert_eq!(ran.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn an_ort_without_parallel_for_installs_nothing() {
        let mut api: ort::OrtApi = unsafe { core::mem::zeroed() };
        api.KernelContext_ParallelFor = None;
        let width = AtomicU32::new(0);
        let guard = unsafe { install(&api, core::ptr::dangling_mut(), &width) };
        assert!(!guard.is_installed());
        assert!(host_parallel::current().is_none());
    }

    #[test]
    fn a_null_context_installs_nothing() {
        let mut api: ort::OrtApi = unsafe { core::mem::zeroed() };
        api.KernelContext_ParallelFor = Some(inline_parallel_for);
        let width = AtomicU32::new(0);
        let guard = unsafe { install(&api, core::ptr::null_mut(), &width) };
        assert!(!guard.is_installed());
        assert!(host_parallel::current().is_none());
    }

    #[test]
    fn install_gives_a_working_handle_and_removes_it_on_drop() {
        let mut api: ort::OrtApi = unsafe { core::mem::zeroed() };
        api.KernelContext_ParallelFor = Some(inline_parallel_for);
        let seen: Vec<AtomicUsize> = (0..6).map(|_| AtomicUsize::new(0)).collect();
        {
            let width = AtomicU32::new(0);
            let guard = unsafe { install(&api, core::ptr::dangling_mut(), &width) };
            assert!(guard.is_installed());
            let host = host_parallel::current().expect("install publishes a handle");
            host.run(seen.len(), &|index| {
                seen[index].fetch_add(1, Ordering::Relaxed);
            });
        }
        assert!(seen.iter().all(|c| c.load(Ordering::Relaxed) == 1));
        assert!(
            host_parallel::current().is_none(),
            "the handle must not outlive the compute call"
        );
    }

    #[test]
    fn a_nested_install_restores_the_outer_handle() {
        let mut api: ort::OrtApi = unsafe { core::mem::zeroed() };
        api.KernelContext_ParallelFor = Some(inline_parallel_for);
        let width = AtomicU32::new(0);
        let outer = unsafe { install(&api, core::ptr::dangling_mut(), &width) };
        assert!(outer.is_installed());
        {
            let _inner = unsafe { install(&api, core::ptr::dangling_mut(), &width) };
            assert!(host_parallel::current().is_some());
        }
        assert!(
            host_parallel::current().is_some(),
            "the inner guard must not uninstall the outer one"
        );
        drop(outer);
        assert!(host_parallel::current().is_none());
    }
    /// The box handed to ORT as a `void*` exists only for its address, and its
    /// contents are overwritten on every install — so it can be kept and
    /// reused instead of being freed and reallocated once per `Run`.
    ///
    /// Counted rather than asserted structurally: the point of the change is
    /// the allocation, so the allocation is what the test looks at.
    #[cfg(feature = "dispatch_probe")]
    #[test]
    fn repeated_installs_on_one_thread_allocate_the_box_once() {
        use crate::dispatch_probe::{self, Event};
        let mut api: ort::OrtApi = unsafe { core::mem::zeroed() };
        api.KernelContext_ParallelFor = Some(inline_parallel_for);
        let width = AtomicU32::new(0);

        // Prime the cache so the count below is not the first-call allocation.
        drop(unsafe { install(&api, core::ptr::dangling_mut(), &width) });

        let before = dispatch_probe::snapshot();
        for _ in 0..8 {
            let guard = unsafe { install(&api, core::ptr::dangling_mut(), &width) };
            assert!(guard.is_installed());
        }
        let allocs = dispatch_probe::snapshot()
            .since(&before)
            .event(Event::DispatchAlloc);
        assert_eq!(
            allocs, 0,
            "eight installs after the first must reuse this thread's box"
        );
    }

    /// A reused slot must be fully overwritten, not partly: an install that
    /// inherited a previous call's context would hand ORT a stale pointer.
    #[test]
    fn a_reused_box_carries_the_new_context_not_the_old_one() {
        let mut api: ort::OrtApi = unsafe { core::mem::zeroed() };
        api.KernelContext_ParallelFor = Some(inline_parallel_for);
        let width = AtomicU32::new(0);

        let first_ctx = core::ptr::without_provenance_mut::<ort::OrtKernelContext>(0x1000);
        let second_ctx = core::ptr::without_provenance_mut::<ort::OrtKernelContext>(0x2000);

        drop(unsafe { install(&api, first_ctx, &width) });
        let guard = unsafe { install(&api, second_ctx, &width) };
        let slot = guard.pool.as_ref().expect("a working install has a pool");
        // SAFETY: the guard is alive, so the slot is live and initialised.
        let ctx = unsafe { (*slot.as_ptr()).ctx };
        assert_eq!(
            ctx, second_ctx,
            "the recycled slot kept the previous call's context"
        );
    }

    /// Re-entrancy must not defeat the cache or make it grow: the inner call
    /// finds the cache empty and allocates, and after both guards are gone the
    /// thread still retains exactly one box.
    #[test]
    fn nesting_keeps_at_most_one_box_cached() {
        let mut api: ort::OrtApi = unsafe { core::mem::zeroed() };
        api.KernelContext_ParallelFor = Some(inline_parallel_for);
        let width = AtomicU32::new(0);

        {
            let _outer = unsafe { install(&api, core::ptr::dangling_mut(), &width) };
            let _inner = unsafe { install(&api, core::ptr::dangling_mut(), &width) };
            assert!(
                POOL_CACHE.with(|c| {
                    let taken = c.take();
                    let empty = taken.is_none();
                    c.set(taken);
                    empty
                }),
                "both boxes are checked out, so the cache must be empty"
            );
        }
        assert!(
            POOL_CACHE.with(|c| {
                let taken = c.take();
                let held = taken.is_some();
                c.set(taken);
                held
            }),
            "one box must be retained for the next call"
        );
    }

    /// The `void*` ORT is given aliases the pool slot for the whole compute
    /// call, and the slot changes owner several times while that alias is
    /// live: into the guard, out of `install`, out of `Option::take` in
    /// `Drop`, and into the thread-local cache.
    ///
    /// A `Box` cannot express that. Every move of one reasserts that nothing
    /// else points at its contents, and the handle's first read through the
    /// alias has already made that false, so the next move retags from a tag
    /// that no longer exists. Miri reports it as
    ///
    /// ```text
    /// trying to retag from <N> for Unique permission ..., but that tag does
    /// not exist in the borrow stack for this location
    /// ```
    ///
    /// pointing at `Guard::drop`. Reproduced identically on `main` before
    /// #1526 and after it, so the thread-local reuse neither caused it nor
    /// escapes it.
    ///
    /// The order below is the whole test: **dispatch first, then hand the slot
    /// on**. Running the handle after the guard is gone would be a different
    /// (and legitimately forbidden) thing; what has to be legal is aliasing a
    /// slot that Rust code still owns and moves.
    ///
    /// Run under `cargo +nightly miri test -p onnx-runtime-ep-plugin --lib
    /// host_pool::`; without Miri it is a plain liveness test and cannot fail
    /// for this reason.
    #[test]
    fn the_handle_may_alias_the_pool_slot_across_owning_moves() {
        let mut api: ort::OrtApi = unsafe { core::mem::zeroed() };
        api.KernelContext_ParallelFor = Some(inline_parallel_for);
        let width = AtomicU32::new(HOST_HELPED);
        let seen = AtomicUsize::new(0);

        // Twice, so the second install exercises the recycled slot: the alias
        // is then a read through an address a previous `Run` also aliased.
        for _ in 0..2 {
            let guard = unsafe { install(&api, core::ptr::dangling_mut(), &width) };
            assert!(guard.is_installed());
            host_parallel::current()
                .expect("install publishes a handle")
                .run(4, &|_| {
                    seen.fetch_add(1, Ordering::Relaxed);
                });
            // The moves that used to be undefined: `Option::take` in `Drop`,
            // then into the thread-local cache.
            drop(guard);
        }
        assert_eq!(seen.load(Ordering::Relaxed), 8);
    }

    /// Reuse must survive the ownership change: the second install has to land
    /// on the same address, not merely on some address.
    ///
    /// Asserted structurally rather than by allocation count so it holds
    /// without the `dispatch_probe` feature, which is off by default.
    #[test]
    fn a_recycled_slot_keeps_its_address() {
        let mut api: ort::OrtApi = unsafe { core::mem::zeroed() };
        api.KernelContext_ParallelFor = Some(inline_parallel_for);
        let width = AtomicU32::new(0);

        let first = {
            let guard = unsafe { install(&api, core::ptr::dangling_mut(), &width) };
            guard
                .pool
                .as_ref()
                .expect("a working install has a pool")
                .as_ptr()
        };
        let guard = unsafe { install(&api, core::ptr::dangling_mut(), &width) };
        let second = guard
            .pool
            .as_ref()
            .expect("a working install has a pool")
            .as_ptr();
        assert_eq!(first, second, "the second install did not reuse the slot");
    }

    /// The cost of reusing the slot instead of freeing it: a handle held past
    /// its compute call no longer points at freed memory, so no sanitiser
    /// would catch it. [`Epoch`] catches it here.
    ///
    /// This is the realistic shape of the bug — a handle stashed during one
    /// `Run` and dispatched through later, from a deferred task, a background
    /// thread, or simply after the compute call returned. The slot is alive
    /// (it is in this thread's cache), so the read is well defined and the
    /// stamp is what refuses it.
    ///
    /// Mutation-tested: deleting the `assert_live` call in `ort_parallel_for`
    /// makes this test fail with "test did not panic as expected", and the
    /// dispatch then proceeds against a retired `OrtKernelContext`.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "a host-pool handle outlived the compute call that installed it")]
    fn a_handle_from_a_finished_run_is_refused() {
        let mut api: ort::OrtApi = unsafe { core::mem::zeroed() };
        api.KernelContext_ParallelFor = Some(inline_parallel_for);
        let width = AtomicU32::new(HOST_HELPED);

        let stale = {
            let _guard = unsafe { install(&api, core::ptr::dangling_mut(), &width) };
            host_parallel::current().expect("install publishes a handle")
        };
        stale.run(4, &|_| {});
    }

    /// And a handle from a *different* live install is refused too, which is
    /// the nested case: the inner guard's slot is retired while the outer
    /// `Run` is still going.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "a host-pool handle outlived the compute call that installed it")]
    fn a_handle_from_a_retired_inner_run_is_refused() {
        let mut api: ort::OrtApi = unsafe { core::mem::zeroed() };
        api.KernelContext_ParallelFor = Some(inline_parallel_for);
        let width = AtomicU32::new(HOST_HELPED);

        let _outer = unsafe { install(&api, core::ptr::dangling_mut(), &width) };
        let stale = {
            let _inner = unsafe { install(&api, core::ptr::dangling_mut(), &width) };
            host_parallel::current().expect("install publishes a handle")
        };
        stale.run(4, &|_| {});
    }

    /// And a nested install must hand the outer stamp back, or every dispatch
    /// after an inner guard drops would be refused as stale.
    #[test]
    fn nesting_restores_the_outer_run_stamp() {
        let mut api: ort::OrtApi = unsafe { core::mem::zeroed() };
        api.KernelContext_ParallelFor = Some(inline_parallel_for);
        let width = AtomicU32::new(HOST_HELPED);
        let seen = AtomicUsize::new(0);

        let outer = unsafe { install(&api, core::ptr::dangling_mut(), &width) };
        let outer_handle = host_parallel::current().expect("install publishes a handle");
        drop(unsafe { install(&api, core::ptr::dangling_mut(), &width) });
        outer_handle.run(4, &|_| {
            seen.fetch_add(1, Ordering::Relaxed);
        });
        drop(outer);
        assert_eq!(seen.load(Ordering::Relaxed), 4);
    }

    /// The staleness guard must not be something release pays for: #1077 is a
    /// fixed-per-`Run`-cost campaign and this module is on that path.
    ///
    /// Layout is the checkable half — [`Epoch`] is a ZST in release, so
    /// `HostPool` is the same four words it was — and it is the half that
    /// would silently change if someone made the field unconditional.
    #[test]
    fn the_staleness_guard_is_free_in_release() {
        let pointer_words = 4 * size_of::<*const ()>();
        if cfg!(debug_assertions) {
            assert_eq!(size_of::<Epoch>(), size_of::<u64>());
        } else {
            assert_eq!(size_of::<Epoch>(), 0);
            assert_eq!(
                size_of::<HostPool>(),
                pointer_words,
                "the stamp grew the slot in release"
            );
        }
    }

    /// The cache is per thread, and a thread that exits must take its slot with
    /// it. Owning the allocation as a raw pointer is the part of this change
    /// that could have turned into a leak — `PoolSlot`'s `Drop` is now the only
    /// thing that frees it, and a thread-local's destructor is the only thing
    /// that runs it at thread exit.
    ///
    /// Under Miri this is the falsifier for that: removing `impl Drop for
    /// PoolSlot` reports "the evaluated program leaked memory" here. Natively
    /// it still pins that concurrent installs do not share a slot.
    #[test]
    fn each_thread_owns_its_slot_and_frees_it_on_exit() {
        let mut api: ort::OrtApi = unsafe { core::mem::zeroed() };
        api.KernelContext_ParallelFor = Some(inline_parallel_for);

        // Distinct addresses only mean distinct slots while every slot is live
        // at once. Without this rendezvous a thread may exit, its cache
        // destructor may free its slot, and the allocator may hand the same
        // address to a thread that starts later -- a legitimate reuse that
        // would read as a shared slot.
        let all_allocated = std::sync::Barrier::new(3);
        let addresses: Vec<usize> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..3)
                .map(|_| {
                    scope.spawn(|| {
                        let width = AtomicU32::new(HOST_HELPED);
                        // Twice, so the thread exercises its own cache before
                        // the thread-local destructor has to free it.
                        let mut address = 0;
                        for _ in 0..2 {
                            let guard = unsafe { install(&api, core::ptr::dangling_mut(), &width) };
                            address = guard
                                .pool
                                .as_ref()
                                .expect("a working install has a pool")
                                .as_ptr() as usize;
                            host_parallel::current()
                                .expect("install publishes a handle")
                                .run(2, &|_| {});
                        }
                        all_allocated.wait();
                        address
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut unique = addresses.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            addresses.len(),
            "two threads shared a slot: {addresses:?}"
        );
    }
}
