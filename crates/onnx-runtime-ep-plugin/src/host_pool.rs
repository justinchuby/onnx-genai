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

use core::ffi::c_void;
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
    let pool = unsafe { &*(host.cast::<HostPool>()) };
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

/// ORT's pool, installed on this thread until the guard is dropped.
///
/// Field order is the safety argument: `installed` is dropped first, so the
/// thread-local handle is gone before `pool` — the allocation it points at —
/// is freed.
pub struct Guard {
    installed: Option<host_parallel::Installed>,
    pool: Option<Box<HostPool>>,
}

impl Guard {
    /// A guard that installs nothing, for hosts that offer no pool.
    const fn inert() -> Self {
        Self {
            installed: None,
            pool: None,
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
            recycle_pool(pool);
        }
    }
}

thread_local! {
    /// One retired `HostPool` box per thread, kept for the next `Run`.
    ///
    /// The box exists only to give the `void*` handle a stable address; its
    /// contents are overwritten on every install. Allocating it fresh each
    /// time was one `malloc` and one `free` per `Run` for 32 bytes, on a path
    /// where the whole fixed cost is a couple of microseconds.
    ///
    /// Thread-local because `Compute` runs on whichever thread ORT calls it
    /// from and the box never escapes that thread. `Cell` rather than
    /// `RefCell`: the only operations are take and replace, so there is
    /// nothing to borrow and no way to panic while holding it.
    ///
    /// # The invariant reuse rests on
    ///
    /// No one may hold the `void*` handle past the compute call that installed
    /// it. That was already required, but freeing the box made a violation
    /// loud: the stale handle pointed at freed memory, which a sanitiser
    /// catches. Reuse makes it quiet instead — a stale handle would now find a
    /// live box belonging to a *later* `Run` and silently compute against the
    /// wrong context. Every consumer today takes the handle from
    /// `host_parallel::current()` and uses it synchronously, within the
    /// compute extent; keep it that way.
    static POOL_CACHE: std::cell::Cell<Option<Box<HostPool>>> =
        const { std::cell::Cell::new(None) };
}

/// A box to build a `HostPool` in, reusing this thread's retired one.
///
/// Falls back to a fresh allocation when the cache is empty, which covers both
/// the first call on a thread and a re-entrant one whose outer guard is still
/// holding the cached box.
fn take_pool(pool: HostPool) -> Box<HostPool> {
    match POOL_CACHE.with(std::cell::Cell::take) {
        Some(mut boxed) => {
            *boxed = pool;
            boxed
        }
        None => {
            crate::dispatch_probe::count(crate::dispatch_probe::Event::DispatchAlloc);
            Box::new(pool)
        }
    }
}

/// Hand a retired box back for the next `Run` on this thread.
///
/// Drops it if the cache is already occupied, so a re-entrant call cannot make
/// the cache grow: at most one box per thread is ever retained.
fn recycle_pool(pool: Box<HostPool>) {
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
    let pool = take_pool(HostPool {
        parallel_for,
        release_status: api.ReleaseStatus,
        ctx,
        probe: core::ptr::from_ref(probe),
    });
    // SAFETY: the box outlives the handle (see `Guard`'s drop order), and
    // `ort_parallel_for` only ever reads the pointer back as a `*mut HostPool`.
    let handle = unsafe {
        HostParallel::new(
            (&raw const *pool).cast::<c_void>().cast_mut(),
            ort_parallel_for,
            core::ptr::from_ref(probe),
        )
    };
    Guard {
        installed: Some(host_parallel::Installed::new(handle)),
        pool: Some(pool),
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
        let body = body.expect("ORT is always given a callback");
        let usr = usr_data as usize;
        std::thread::scope(|scope| {
            for index in 0..total {
                scope.spawn(move || unsafe { body(usr as *mut c_void, index) });
            }
        });
        core::ptr::null_mut()
    }

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
    #[test]
    fn a_latched_session_pays_no_stall() {
        let probe = AtomicU32::new(HOST_HELPED);
        let mut pool = pool(inline_parallel_for, &probe);
        let started = std::time::Instant::now();
        for _ in 0..8 {
            unsafe { ort_parallel_for((&raw mut pool).cast::<c_void>(), 4, &|_| {}) };
        }
        assert!(
            started.elapsed() < PROBE_STALL,
            "eight latched dispatches took longer than a single probe stall"
        );
    }

    /// And it must run at most once per probe, however many indices there are.
    #[test]
    fn a_probe_stalls_once_not_once_per_index() {
        let probe = AtomicU32::new(0);
        let mut pool = pool(inline_parallel_for, &probe);
        let started = std::time::Instant::now();
        unsafe { ort_parallel_for((&raw mut pool).cast::<c_void>(), 64, &|_| {}) };
        let elapsed = started.elapsed();
        assert!(
            elapsed >= PROBE_STALL,
            "a probe has to give workers a chance"
        );
        assert!(
            elapsed < PROBE_STALL * 4,
            "the stall is per dispatch, not per index: {elapsed:?}"
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

    /// A reused box must be fully overwritten, not partly: an install that
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
        let pool = guard.pool.as_ref().expect("a working install has a pool");
        assert_eq!(
            pool.ctx, second_ctx,
            "the recycled box kept the previous call's context"
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
}
