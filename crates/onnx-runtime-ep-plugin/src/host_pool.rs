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
use onnx_runtime_ep_api::host_parallel::{WIDTH_PARALLEL, WIDTH_SERIAL, WIDTH_UNKNOWN};
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

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
    /// Where this session records how wide ORT's pool turned out to be.
    width: *const AtomicU8,
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
    if let Some(caller) = task.caller
        && std::thread::current().id() != caller
    {
        task.saw_another_thread.store(true, Ordering::Relaxed);
    }
    if std::panic::catch_unwind(AssertUnwindSafe(|| (task.body)(index))).is_err() {
        task.panicked.store(true, Ordering::Relaxed);
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
    // Only look at thread identities until the answer is in. A session runs
    // this on every dispatch, and `thread::current()` is not free.
    let observing = total > 1
        && unsafe { pool.width() }.is_some_and(|w| w.load(Ordering::Relaxed) == WIDTH_UNKNOWN);
    let task = Task {
        body,
        panicked: AtomicBool::new(false),
        caller: observing.then(|| std::thread::current().id()),
        saw_another_thread: AtomicBool::new(false),
    };
    // `num_batch = 0` means "no limit": ORT gives every index its own task and
    // its workers claim them dynamically. That is what we want, because we cut
    // the slice by size rather than by thread count — we cannot ask ORT how
    // wide its pool is, so we hand it enough pieces to fill any pool and let
    // it decide how many to run at once.
    let status = unsafe {
        (pool.parallel_for)(
            pool.ctx,
            Some(run_index),
            total,
            0,
            (&raw const task).cast::<c_void>().cast_mut(),
        )
    };
    if observing && status.is_null() {
        // ORT ran every index; if none of them landed on another thread, its
        // intra-op pool has no workers. Recorded per session, so a process
        // with both a one-thread and a sixteen-thread session gets the right
        // answer for each.
        let observed = if task.saw_another_thread.load(Ordering::Relaxed) {
            WIDTH_PARALLEL
        } else {
            WIDTH_SERIAL
        };
        if let Some(width) = unsafe { pool.width() } {
            width.store(observed, Ordering::Relaxed);
        }
    }
    if !status.is_null() {
        // The dispatch itself failed. Every index still has to run or the
        // output tensor keeps whatever was in the buffer, so fall back to
        // running them here rather than silently returning short.
        if let Some(release) = pool.release_status {
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
    /// The session's width cell, if it has one.
    ///
    /// # Safety
    ///
    /// The pointer must still be valid, which `install`'s contract requires
    /// for as long as the guard is alive.
    unsafe fn width(&self) -> Option<&AtomicU8> {
        (!self.width.is_null()).then(|| unsafe { &*self.width })
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
        drop(self.pool.take());
    }
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
    width: &AtomicU8,
) -> Guard {
    let (Some(parallel_for), false) = (api.KernelContext_ParallelFor, ctx.is_null()) else {
        return Guard::inert();
    };
    let pool = Box::new(HostPool {
        parallel_for,
        release_status: api.ReleaseStatus,
        ctx,
        width: core::ptr::from_ref(width),
    });
    // SAFETY: the box outlives the handle (see `Guard`'s drop order), and
    // `ort_parallel_for` only ever reads the pointer back as a `*mut HostPool`.
    let handle = unsafe {
        HostParallel::new(
            (&raw const *pool).cast::<c_void>().cast_mut(),
            ort_parallel_for,
            core::ptr::from_ref(width),
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
        width: &AtomicU8,
    ) -> HostPool {
        HostPool {
            parallel_for,
            release_status: None,
            ctx: core::ptr::null_mut(),
            width: core::ptr::from_ref(width),
        }
    }

    #[test]
    fn every_index_runs_once() {
        let width = AtomicU8::new(WIDTH_UNKNOWN);
        let mut pool = pool(inline_parallel_for, &width);
        let seen: Vec<AtomicUsize> = (0..5).map(|_| AtomicUsize::new(0)).collect();
        unsafe {
            ort_parallel_for((&raw mut pool).cast::<c_void>(), seen.len(), &|index| {
                seen[index].fetch_add(1, Ordering::Relaxed);
            });
        }
        assert!(seen.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }

    /// An ORT that runs everything inline has no workers, and our own pool is
    /// then free to use the machine. Getting this wrong in either direction
    /// costs multiples: our pool alongside ORT's is 1.5-3.1x slower than
    /// serial, and serial where ORT has no workers is 2-9x slower than ours.
    #[test]
    fn an_inline_host_is_observed_as_serial() {
        let width = AtomicU8::new(WIDTH_UNKNOWN);
        let mut pool = pool(inline_parallel_for, &width);
        unsafe { ort_parallel_for((&raw mut pool).cast::<c_void>(), 4, &|_| {}) };
        assert_eq!(width.load(Ordering::Relaxed), WIDTH_SERIAL);
    }

    #[test]
    fn a_threaded_host_is_observed_as_parallel() {
        let width = AtomicU8::new(WIDTH_UNKNOWN);
        let mut pool = pool(threaded_parallel_for, &width);
        unsafe { ort_parallel_for((&raw mut pool).cast::<c_void>(), 4, &|_| {}) };
        assert_eq!(width.load(Ordering::Relaxed), WIDTH_PARALLEL);
    }

    /// One index proves nothing: even a sixteen-thread pool runs it inline.
    #[test]
    fn a_single_index_dispatch_does_not_decide_the_width() {
        let width = AtomicU8::new(WIDTH_UNKNOWN);
        let mut pool = pool(inline_parallel_for, &width);
        unsafe { ort_parallel_for((&raw mut pool).cast::<c_void>(), 1, &|_| {}) };
        assert_eq!(width.load(Ordering::Relaxed), WIDTH_UNKNOWN);
    }

    /// A refused dispatch says nothing about the pool either.
    #[test]
    fn a_refused_dispatch_does_not_decide_the_width() {
        let width = AtomicU8::new(WIDTH_UNKNOWN);
        let mut pool = pool(failing_parallel_for, &width);
        unsafe { ort_parallel_for((&raw mut pool).cast::<c_void>(), 4, &|_| {}) };
        assert_eq!(width.load(Ordering::Relaxed), WIDTH_UNKNOWN);
    }

    /// Once decided, later dispatches stop paying for the observation.
    #[test]
    fn an_answered_width_is_not_revisited() {
        let width = AtomicU8::new(WIDTH_PARALLEL);
        let mut pool = pool(inline_parallel_for, &width);
        unsafe { ort_parallel_for((&raw mut pool).cast::<c_void>(), 4, &|_| {}) };
        assert_eq!(width.load(Ordering::Relaxed), WIDTH_PARALLEL);
    }

    #[test]
    fn a_refused_dispatch_still_runs_every_index() {
        let width = AtomicU8::new(WIDTH_UNKNOWN);
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
        let width = AtomicU8::new(WIDTH_UNKNOWN);
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
        let width = AtomicU8::new(WIDTH_UNKNOWN);
        let guard = unsafe { install(&api, core::ptr::dangling_mut(), &width) };
        assert!(!guard.is_installed());
        assert!(host_parallel::current().is_none());
    }

    #[test]
    fn a_null_context_installs_nothing() {
        let mut api: ort::OrtApi = unsafe { core::mem::zeroed() };
        api.KernelContext_ParallelFor = Some(inline_parallel_for);
        let width = AtomicU8::new(WIDTH_UNKNOWN);
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
            let width = AtomicU8::new(WIDTH_UNKNOWN);
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
        let width = AtomicU8::new(WIDTH_UNKNOWN);
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
}
