//! Shared helpers for the crate's real-CUDA integration tests (`tests/vmm_*`).
//!
//! This module only exists for tests: it is compiled purely behind the
//! `gpu-tests` feature, so it never reaches a production build. `cfg(test)`
//! does NOT gate it (and never did usefully) — `cfg(test)` only applies when
//! this crate is compiled as its own unit-test binary, not when integration
//! tests under `tests/*.rs` link it as a regular dependency, so an integration
//! test needing this module must build with `--features gpu-tests` explicitly.
//! Its whole reason to exist is [`TestStream`], which makes the one thing
//! every device test must get right — *stream discipline* — the easy path.
//!
//! # Why single-stream discipline is not optional (#797)
//!
//! CUDA's legacy **default stream** (the one used by every "non-`Async`" driver
//! copy such as `cuMemcpyHtoD_v2` / `cuMemcpyDtoH_v2`, and by `cudaMemcpy`) and a
//! stream created with `CU_STREAM_NON_BLOCKING` are **mutually exempt from
//! implicit synchronization**. Neither waits for the other. So a test that fills
//! or reads a buffer with a plain default-stream copy while a graph / memset runs
//! on a created non-blocking stream has issued two operations that the driver is
//! free to overlap — and on a *cold* context (nothing has warmed it yet) they
//! do overlap. The readback then returns an all-zero or partially filled buffer
//! **with no CUDA error**, because nothing was actually wrong at the API level;
//! the ordering was simply never established.
//!
//! That is precisely the #797 defect: a residency GPU test failed ~50% of the
//! time in isolation and passed otherwise, because an alphabetically-earlier
//! sibling subtest happened to warm the context first. `cuCtxSynchronize` did
//! **not** fix it — synchronizing the context does not impose an order between
//! two streams that are exempt from implicit synchronization with each other;
//! only synchronizing *the stream the work was actually issued on* does.
//!
//! The remedy is to route **every** device operation a test performs — baseline
//! fills, captured-graph launches, memsets, readbacks, tail writes — through
//! **one** stream. Then a single [`TestStream::sync`] is a total order over all
//! of them, and the readback is guaranteed to observe every prior write.
//! [`TestStream`] owns that one stream and offers only stream-ordered
//! operations, so a test written against it cannot accidentally reintroduce the
//! default-stream/non-blocking-stream split.

use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::sys as cu;

/// A single owned CUDA stream that every device operation in a test flows
/// through, so one [`sync`](TestStream::sync) is a total order.
///
/// Create it once at the top of a test (holding the [`CudaContext`] alive for
/// the test's duration) and use its methods for **all** device work. Do **not**
/// mix in plain default-stream copies (`cuMemcpyHtoD_v2`, `cuMemcpyDtoH_v2`,
/// `cuMemsetD8`, …): those run on the legacy default stream, which is exempt
/// from implicit synchronization with this non-blocking stream (see the module
/// docs and #797), and reintroduce exactly the race this type exists to prevent.
///
/// The stream is created `CU_STREAM_NON_BLOCKING` — the same class the captured
/// graphs in these tests launch on — so a graph launched on it and a fill issued
/// on it share one timeline rather than racing across the default-stream barrier.
pub struct TestStream {
    /// Held so the CUDA context outlives the stream; the context must not be
    /// dropped while a stream created under it is still alive.
    _context: Arc<CudaContext>,
    stream: cu::CUstream,
}

impl TestStream {
    /// Create the CUDA context for device 0 (panicking with a clear,
    /// CPU-run-friendly message if there is no driver) and a single non-blocking
    /// stream under it.
    ///
    /// Panics rather than returns `Result` because these are `*_gpu` tests: a
    /// missing driver means the test should have been skipped via the
    /// `gpu-tests` feature gate, and any other failure here is a genuine
    /// environment fault the test must surface immediately.
    #[must_use]
    pub fn new() -> Self {
        let context = match CudaContext::new(0) {
            Ok(context) => context,
            Err(error) => panic!(
                "CUDA test requires a CUDA driver; CPU-only runs must leave this \
                 test ignored (enable the `gpu-tests` feature on a CUDA runner): \
                 {error}"
            ),
        };
        Self::with_context(context)
    }

    /// Build a [`TestStream`] on an already-created context, for tests that must
    /// construct the context themselves (e.g. to also build an allocator on it).
    #[must_use]
    pub fn with_context(context: Arc<CudaContext>) -> Self {
        let mut stream: cu::CUstream = std::ptr::null_mut();
        let result = unsafe {
            cu::cuStreamCreate(
                &mut stream,
                cu::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            )
        };
        assert_eq!(
            result,
            cu::CUresult::CUDA_SUCCESS,
            "cuStreamCreate (non-blocking test stream): {result:?}"
        );
        Self {
            _context: context,
            stream,
        }
    }

    /// The raw stream handle, for driver calls this helper does not wrap (e.g.
    /// `cuGraphLaunch`, `cuMemsetD8Async`). Pass this — never the null default
    /// stream — so the operation joins the single timeline.
    #[must_use]
    pub fn raw(&self) -> cu::CUstream {
        self.stream
    }

    /// The context this stream was created under.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self._context
    }

    fn check(call: &'static str, result: cu::CUresult) {
        assert_eq!(result, cu::CUresult::CUDA_SUCCESS, "{call}: {result:?}");
    }

    /// Block the host until every operation issued on this stream has completed.
    /// Because all device work goes through this one stream, this is a total
    /// order: after it returns, every prior fill/copy/launch is visible.
    pub fn sync(&self) {
        Self::check("cuStreamSynchronize", unsafe {
            cu::cuStreamSynchronize(self.stream)
        });
    }

    /// Set `len` bytes at `address` to `value`, ordered on this stream, and
    /// block until it completes.
    ///
    /// Uses `cuMemsetD8Async` on this stream rather than the default-stream
    /// `cuMemsetD8`, so the fill shares a timeline with everything else the test
    /// does (see the module docs on the #797 race).
    pub fn fill(&self, address: cu::CUdeviceptr, value: u8, len: usize) {
        Self::check("cuMemsetD8Async", unsafe {
            cu::cuMemsetD8Async(address, value, len, self.stream)
        });
        self.sync();
    }

    /// Copy `bytes` from the host to `address`, ordered on this stream, and
    /// block until it completes.
    pub fn write(&self, address: cu::CUdeviceptr, bytes: &[u8]) {
        Self::check("cuMemcpyHtoDAsync_v2", unsafe {
            cu::cuMemcpyHtoDAsync_v2(address, bytes.as_ptr().cast(), bytes.len(), self.stream)
        });
        self.sync();
    }

    /// Read `len` bytes from `address` to the host, ordered on this stream.
    /// Because the copy is issued on this stream and then synchronized, it is
    /// guaranteed to observe every prior write issued on the same stream — the
    /// property whose absence caused the #797 partial readbacks.
    #[must_use]
    pub fn read(&self, address: cu::CUdeviceptr, len: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; len];
        Self::check("cuMemcpyDtoHAsync_v2", unsafe {
            cu::cuMemcpyDtoHAsync_v2(bytes.as_mut_ptr().cast(), address, len, self.stream)
        });
        self.sync();
        bytes
    }
}

impl Default for TestStream {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TestStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            // Best-effort teardown: the process is a test and the context is
            // about to drop anyway, so a failure here is not worth a panic.
            let _ = unsafe { cu::cuStreamDestroy_v2(self.stream) };
        }
    }
}
