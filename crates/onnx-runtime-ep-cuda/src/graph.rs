//! Serialized ownership for the CUDA graph captured on an EP runtime stream.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::ThreadId;

use arc_swap::ArcSwapOption;
use cudarc::driver::sys::{
    CUgraph, CUgraphExec, CUgraphInstantiate_flags, CUstreamCaptureMode, CUstreamCaptureStatus,
};
use cudarc::driver::{CudaStream, result};
use onnx_runtime_cuda_memory::capture_gate::CaptureExclusion;
use onnx_runtime_ep_api::{DeviceGraphResource, EpError, Result};

use crate::error::driver_err;

/// Whether the lifecycle is currently recording a segment, and on which thread.
enum CaptureState {
    Idle,
    Capturing(ThreadId),
}

/// Owns the graph and graph-exec handles created from one runtime stream.
///
/// CUDA graph handles may cross threads only when every access is externally
/// serialized. This wrapper owns both handles and destroys each exactly once.
struct CapturedGraph {
    graph: CUgraph,
    graph_exec: CUgraphExec,
    stream: Arc<CudaStream>,
    /// Dropped only after `graph_exec` and `graph` are destroyed by `Drop`.
    resources: Vec<DeviceGraphResource>,
}

impl CapturedGraph {
    fn end_capture(
        stream: &Arc<CudaStream>,
        flags: CUgraphInstantiate_flags,
        resources: Vec<DeviceGraphResource>,
    ) -> std::result::Result<Option<Self>, cudarc::driver::DriverError> {
        stream.context().bind_to_thread()?;
        // SAFETY: this lifecycle holds the state mutex and `stream` is currently
        // capturing on the calling thread.
        let graph = unsafe { result::stream::end_capture(stream.cu_stream()) }?;
        if graph.is_null() {
            return Ok(None);
        }

        // SAFETY: `graph` is the fresh non-null handle returned by end_capture.
        let graph_exec = match unsafe { result::graph::instantiate(graph, flags) } {
            Ok(graph_exec) => graph_exec,
            Err(error) => {
                // cudarc's combined end_capture helper cannot represent ownership
                // between these calls. Destroy the intermediate graph before
                // returning an instantiate error so that path cannot leak it.
                // SAFETY: instantiation failed, so this function exclusively owns
                // the fresh graph handle and destroys it exactly once here.
                stream
                    .context()
                    .record_err(unsafe { result::graph::destroy(graph) });
                return Err(error);
            }
        };

        Ok(Some(Self {
            graph,
            graph_exec,
            stream: stream.clone(),
            resources,
        }))
    }

    fn upload(&self) -> std::result::Result<(), cudarc::driver::DriverError> {
        self.stream.context().bind_to_thread()?;
        // SAFETY: this wrapper owns `graph_exec`, which has not been published
        // for replay yet, and uploads it on its owning stream.
        unsafe { result::graph::upload(self.graph_exec, self.stream.cu_stream()) }
    }

    fn launch(&self) -> std::result::Result<(), cudarc::driver::DriverError> {
        self.stream.context().bind_to_thread()?;
        // SAFETY: the executable is immutable after publication and every
        // launch is submitted to its one owning stream.
        unsafe { result::graph::launch(self.graph_exec, self.stream.cu_stream()) }
    }
}

// SAFETY: after publication a captured graph is immutable. CUDA graph launches
// are submitted to the one owned stream, whose ordering serializes execution;
// ArcSwap keeps the handles alive across concurrent reset/invalidation.
unsafe impl Send for CapturedGraph {}
// SAFETY: same immutable-publication and stream-ordering invariant as `Send`.
unsafe impl Sync for CapturedGraph {}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        let context = self.stream.context();
        context.record_err(context.bind_to_thread());

        let graph_exec = std::mem::replace(&mut self.graph_exec, std::ptr::null_mut());
        if !graph_exec.is_null() {
            // SAFETY: this wrapper exclusively owns the non-null executable and
            // replaces it with null before destroying it.
            context.record_err(unsafe { result::graph::exec_destroy(graph_exec) });
        }

        let graph = std::mem::replace(&mut self.graph, std::ptr::null_mut());
        if !graph.is_null() {
            // SAFETY: this wrapper exclusively owns the non-null graph and
            // replaces it with null before destroying it.
            context.record_err(unsafe { result::graph::destroy(graph) });
        }

        // Graph launches already queued on `stream` may still be in flight.
        // Releasing a resource enqueues its final allocation release behind the
        // stream tail, so keeping the owners through handle destruction is the
        // required ordering boundary.
        self.resources.clear();
    }
}

/// Owns the captured graph segments installed on one EP runtime stream.
///
/// Capture mutation stays behind the lifecycle mutex. Completed executables are
/// immutable and atomically published for allocation- and mutex-free replay on
/// their single owning stream.
///
/// A whole-subgraph capture installs exactly one segment. Segmented capture —
/// used when only parts of a claimed subgraph are device-graph capturable —
/// installs one segment per maximal capturable run; the non-capturable seam
/// nodes execute eagerly between segment replays. Segments launch in capture
/// order and each is destroyed exactly once on reset/drop.
pub(crate) struct CudaGraphLifecycle {
    stream: Arc<CudaStream>,
    state: Mutex<LifecycleState>,
    replay: ArcSwapOption<ReplaySet>,
    lock_acquisitions: AtomicU64,
}

struct ReplaySet {
    segments: Vec<Arc<CapturedGraph>>,
}

/// The capture flag and the ordered list of installed segment executables.
struct LifecycleState {
    capture: CaptureState,
    /// Held for the whole capture region so no other thread performs a
    /// device-synchronizing memory operation that would invalidate it. Set in
    /// `begin`, cleared on every exit from capture (`end`, `abort`, reset).
    exclusion: Option<CaptureExclusion>,
    /// Resources provisionally retained by the active capture. On successful
    /// instantiation these move into exactly one `CapturedGraph`; abort/failure
    /// drops them after the half-recorded graph is destroyed.
    capture_resources: Vec<DeviceGraphResource>,
    segments: Vec<Arc<CapturedGraph>>,
}

// SAFETY: capture mutation is serialized through `state`; published executables
// are immutable and every segment launches on its single owning `stream`.
unsafe impl Send for CudaGraphLifecycle {}
// SAFETY: the same mutation/publication invariant covers shared references.
unsafe impl Sync for CudaGraphLifecycle {}

impl CudaGraphLifecycle {
    pub(crate) fn new(stream: Arc<CudaStream>) -> Self {
        Self {
            stream,
            state: Mutex::new(LifecycleState {
                capture: CaptureState::Idle,
                exclusion: None,
                capture_resources: Vec::new(),
                segments: Vec::new(),
            }),
            replay: ArcSwapOption::empty(),
            lock_acquisitions: AtomicU64::new(0),
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, LifecycleState>> {
        self.lock_acquisitions.fetch_add(1, Ordering::Relaxed);
        self.state.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep: CUDA graph lifecycle lock was poisoned".into())
        })
    }

    pub(crate) fn lock_acquisition_count(&self) -> u64 {
        self.lock_acquisitions.load(Ordering::Relaxed)
    }

    /// Begin recording a new segment. Additional segments may be captured while
    /// earlier ones are already installed (segmented capture); only a second
    /// concurrent capture is rejected.
    pub(crate) fn begin(&self, resources: Vec<DeviceGraphResource>) -> Result<()> {
        let mut state = self.lock()?;
        match state.capture {
            CaptureState::Idle => {}
            CaptureState::Capturing(_) => {
                return Err(EpError::KernelFailed(
                    "cuda_ep: cannot begin CUDA graph capture while capture is already active"
                        .into(),
                ));
            }
        }
        debug_assert!(
            state.capture_resources.is_empty(),
            "idle CUDA graph lifecycle retained provisional resources"
        );
        for resource in resources {
            if !state
                .capture_resources
                .iter()
                .any(|existing| existing.identity() == resource.identity())
            {
                state.capture_resources.push(resource);
            }
        }

        // Acquire *before* `cuStreamBeginCapture`: taking it afterwards leaves a
        // window in which the capture is live and unprotected. THREAD_LOCAL mode
        // relaxes CUDA's legality check on unsafe calls, not the fact that a
        // device-wide synchronization anywhere in the process invalidates this
        // capture.
        let exclusion = CaptureExclusion::acquire();
        if let Err(error) = self
            .stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        {
            state.capture_resources.clear();
            return Err(driver_err("begin CUDA graph stream capture", error));
        }
        state.capture = CaptureState::Capturing(std::thread::current().id());
        state.exclusion = Some(exclusion);
        Ok(())
    }

    /// End the active segment capture, instantiate it, and append it to the
    /// ordered segment list.
    pub(crate) fn end(&self) -> Result<()> {
        let mut state = self.lock()?;
        match state.capture {
            CaptureState::Capturing(owner) if owner == std::thread::current().id() => {}
            CaptureState::Capturing(_) => {
                return Err(EpError::KernelFailed(
                    "cuda_ep: CUDA graph capture must end on the thread that began the \
                     thread-local capture"
                        .into(),
                ));
            }
            CaptureState::Idle => {
                return Err(EpError::KernelFailed(
                    "cuda_ep: cannot end CUDA graph capture because capture is not active".into(),
                ));
            }
        }

        // Clear the capture flag even when end/instantiate fails. CUDA has ended
        // or invalidated the capture at that point, and no executable is usable.
        state.capture = CaptureState::Idle;
        // Bind rather than clear: the stream is still capturing until
        // `end_capture` returns, and `?` below must not skip the release. As a
        // local, it drops at every exit from this function and never earlier.
        let _exclusion = state.exclusion.take();
        let resources = std::mem::take(&mut state.capture_resources);
        let graph = Arc::new(
            CapturedGraph::end_capture(
                &self.stream,
                CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY,
                resources,
            )
            .map_err(|error| driver_err("end and instantiate CUDA graph capture", error))?
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep: CUDA graph capture ended without producing a graph".into(),
                )
            })?,
        );
        graph
            .upload()
            .map_err(|error| driver_err("upload CUDA graph executable", error))?;
        state.segments.push(graph);
        self.replay.store(Some(Arc::new(ReplaySet {
            segments: state.segments.clone(),
        })));
        Ok(())
    }

    /// Replay every installed segment in capture order. For a whole-subgraph
    /// capture this is the single installed graph.
    pub(crate) fn replay(&self) -> Result<()> {
        let replay = self.replay.load();
        let Some(replay) = replay.as_ref() else {
            return Err(EpError::KernelFailed(
                "cuda_ep: cannot replay CUDA graph because no executable is installed".into(),
            ));
        };
        for graph in &replay.segments {
            graph
                .launch()
                .map_err(|error| driver_err("launch CUDA graph executable", error))?;
        }
        Ok(())
    }

    /// Replay one installed segment by its zero-based capture-order index. The
    /// executor drives this per segment, running the non-capturable seam nodes
    /// eagerly between replays.
    pub(crate) fn replay_segment(&self, index: usize) -> Result<()> {
        let replay = self.replay.load();
        let graph = replay.as_ref().and_then(|set| set.segments.get(index)).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep: cannot replay CUDA graph segment {index}; only {} segment(s) installed",
                replay.as_ref().map_or(0, |set| set.segments.len())
            ))
        })?;
        graph
            .launch()
            .map_err(|error| driver_err("launch CUDA graph segment", error))
    }

    /// Abort an in-progress segment capture: terminate the stream capture,
    /// discard any half-recorded graph, and return the lifecycle to `Idle`.
    ///
    /// This is the recovery path when a node fails mid-record during segmented
    /// capture. `cuStreamEndCapture` **must** be called to take the stream out
    /// of capture mode even after the capture was invalidated — otherwise the
    /// stream stays wedged and every later launch fails with
    /// `STREAM_CAPTURE_INVALIDATED`. The invariant callers rely on is "capture
    /// is always ended before [`reset`]", so this leaves the lifecycle in a
    /// state where [`reset`] succeeds and the session can cleanly decline to an
    /// eager run.
    ///
    /// Legal only while `Capturing` on the owning thread; a no-op when idle.
    pub(crate) fn abort(&self) -> Result<()> {
        let mut state = self.lock()?;
        match state.capture {
            CaptureState::Capturing(owner) if owner == std::thread::current().id() => {}
            CaptureState::Capturing(_) => {
                return Err(EpError::KernelFailed(
                    "cuda_ep: CUDA graph capture must abort on the thread that began the \
                     thread-local capture"
                        .into(),
                ));
            }
            CaptureState::Idle => return Ok(()),
        }

        // Clear the flag unconditionally: once we call end_capture the stream is
        // no longer capturing regardless of whether a usable graph came back.
        state.capture = CaptureState::Idle;
        // Released once this function returns, i.e. after `end_capture` has
        // taken the stream out of capture mode. See `end`.
        let _exclusion = state.exclusion.take();
        let resources = std::mem::take(&mut state.capture_resources);
        // End the stream capture to drain the half-recorded graph, then drop it.
        // A mid-capture failure invalidates the capture, so end_capture may
        // report an error — but it still takes the stream out of capture mode,
        // which is the whole point, so that outcome is swallowed here.
        if let Ok(Some(graph)) = CapturedGraph::end_capture(
            &self.stream,
            CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY,
            resources,
        ) {
            drop(graph);
        }
        Ok(())
    }

    pub(crate) fn reset(&self) -> Result<bool> {
        let mut state = self.lock()?;
        if matches!(state.capture, CaptureState::Capturing(_)) {
            return Err(EpError::KernelFailed(
                "cuda_ep: cannot reset CUDA graph while stream capture is active; end capture \
                 first"
                    .into(),
            ));
        }
        let had_graph = !state.segments.is_empty();
        state.segments.clear();
        self.replay.store(None);
        Ok(had_graph)
    }

    pub(crate) fn has_executable(&self) -> Result<bool> {
        Ok(self
            .replay
            .load()
            .as_ref()
            .is_some_and(|set| !set.segments.is_empty()))
    }

    /// Number of installed segment executables (1 for a whole-subgraph capture).
    pub(crate) fn segment_count(&self) -> Result<usize> {
        Ok(self
            .replay
            .load()
            .as_ref()
            .map_or(0, |set| set.segments.len()))
    }

    /// Whether exactly one whole-subgraph segment is installed.
    ///
    /// Dormant scaffolding for the option (c) padded single-M=maxK captured
    /// verify graph (enabled in WP4). Retaining a captured graph across a
    /// contents-only `rewind` is only sound when the capture is a *single*
    /// fixed-topology whole-subgraph segment — a segmented capture interleaves
    /// eager seam nodes whose per-step effects a bare replay would not reproduce.
    /// A retained-graph verify path must gate on this invariant before reusing
    /// the capture instead of re-warming.
    // Kept for the planned WP4 retained-graph verification path.
    #[allow(dead_code)]
    pub(crate) fn holds_single_capture(&self) -> Result<bool> {
        Ok(self
            .replay
            .load()
            .as_ref()
            .is_some_and(|set| set.segments.len() == 1))
    }

    pub(crate) fn capture_status(&self) -> Result<CUstreamCaptureStatus> {
        let _state = self.lock()?;
        self.stream
            .capture_status()
            .map_err(|error| driver_err("query CUDA graph capture status", error))
    }

    pub(crate) fn test_acquire_lock(&self) -> Result<()> {
        drop(self.lock()?);
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use std::sync::Arc;

    use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
    use onnx_runtime_ep_api::{Kernel, TensorMut, TensorView};

    use super::*;
    use crate::runtime::CudaRuntime;

    const MODULE: &str = "graph_lifecycle_test";
    const SOURCE: &str = r#"
extern "C" __global__ void add_one(const float* x, float* y, unsigned long long n) {
    unsigned long long i =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + 1.0f;
}
"#;

    struct TestKernel {
        capturable: bool,
    }

    impl Kernel for TestKernel {
        fn execute(
            &self,
            _inputs: &[TensorView],
            _outputs: &mut [TensorMut],
        ) -> onnx_runtime_ep_api::Result<()> {
            Ok(())
        }

        fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
            if self.capturable {
                onnx_runtime_ep_api::CaptureSupport::Supported
            } else {
                onnx_runtime_ep_api::CaptureSupport::unsupported(
                    "test kernel is configured as non-capturable",
                )
            }
        }
    }

    fn runtime() -> Option<Arc<CudaRuntime>> {
        std::panic::catch_unwind(|| CudaRuntime::new(0).ok().map(Arc::new))
            .ok()
            .flatten()
    }

    fn bytes(values: &[f32]) -> &[u8] {
        // SAFETY: f32 has no invalid bit patterns and the returned byte slice
        // borrows the same live input slice.
        unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        }
    }

    fn read_f32(
        runtime: &CudaRuntime,
        ptr: cudarc::driver::sys::CUdeviceptr,
        n: usize,
    ) -> Vec<f32> {
        let mut values = vec![0.0f32; n];
        // SAFETY: `ptr` is a live allocation of exactly `n * size_of::<f32>()`
        // bytes and `values` provides the matching host destination.
        unsafe {
            runtime
                .dtoh(
                    std::slice::from_raw_parts_mut(
                        values.as_mut_ptr().cast::<u8>(),
                        std::mem::size_of_val(values.as_slice()),
                    ),
                    ptr,
                )
                .unwrap();
        }
        values
    }

    fn launch_add_one(
        runtime: &CudaRuntime,
        function: &CudaFunction,
        input: cudarc::driver::sys::CUdeviceptr,
        output: cudarc::driver::sys::CUdeviceptr,
        n: usize,
    ) {
        let n = n as u64;
        let mut builder = runtime.stream().launch_builder(function);
        builder.arg(&input).arg(&output).arg(&n);
        // SAFETY: the function signature is `(const float*, float*, u64)`;
        // both pointers cover `n` f32 elements and the launch bounds-checks `n`.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(n as u32))
                .unwrap();
        }
    }

    #[test]
    fn capture_replay_uses_live_buffers_without_runtime_allocations() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping CUDA graph lifecycle test: CUDA runtime unavailable");
            return;
        };
        let function = runtime.nvrtc_function(MODULE, SOURCE, "add_one").unwrap();
        let n = 64usize;
        let input_ptr = runtime.alloc_raw(n * std::mem::size_of::<f32>()).unwrap();
        let output_ptr = runtime.alloc_raw(n * std::mem::size_of::<f32>()).unwrap();
        let initial = (0..n).map(|i| i as f32).collect::<Vec<_>>();

        // SAFETY: input_ptr covers the complete host slice.
        unsafe { runtime.htod(bytes(&initial), input_ptr) }.unwrap();
        launch_add_one(&runtime, &function, input_ptr, output_ptr, n);
        runtime.synchronize().unwrap();
        let eager = read_f32(&runtime, output_ptr, n);

        let capturable = TestKernel { capturable: true };
        let allocation_counts = runtime.allocation_counts();
        runtime.begin_graph_capture(&[&capturable]).unwrap();
        assert!(runtime.is_capturing().unwrap());
        launch_add_one(&runtime, &function, input_ptr, output_ptr, n);
        runtime.end_graph_capture().unwrap();
        assert!(runtime.has_graph_executable().unwrap());

        for _ in 0..4 {
            runtime.replay_graph().unwrap();
        }
        runtime.synchronize().unwrap();
        assert_eq!(read_f32(&runtime, output_ptr, n), eager);

        let mutated = (0..n).map(|i| 1000.0 + i as f32).collect::<Vec<_>>();
        // SAFETY: input_ptr remains the same live allocation captured by the graph.
        unsafe { runtime.htod(bytes(&mutated), input_ptr) }.unwrap();
        runtime.replay_graph().unwrap();
        runtime.synchronize().unwrap();
        let mutated_output = read_f32(&runtime, output_ptr, n);
        assert_eq!(
            mutated_output,
            mutated.iter().map(|value| value + 1.0).collect::<Vec<_>>()
        );
        assert_ne!(mutated_output, eager);
        assert_eq!(runtime.allocation_counts(), allocation_counts);

        assert!(runtime.reset_graph().unwrap());
        assert!(!runtime.has_graph_executable().unwrap());
        assert!(!runtime.reset_graph().unwrap());
        // SAFETY: reset dropped graph ownership before either captured buffer is freed.
        unsafe {
            runtime.free_raw(output_ptr).unwrap();
            runtime.free_raw(input_ptr).unwrap();
        }
    }

    /// CORRECTNESS (placement invariant). A per-token excursion to another
    /// device (here the HOST) is compatible with CUDA graph capture on the
    /// native path *only* as an eager seam between captured segments, and is
    /// illegal inside an active capture. This encodes the non-obvious contract
    /// that segmented capture relies on (a host-consuming D2H needs a stream
    /// sync, which invalidates an in-flight capture) so a future refactor cannot
    /// silently break it. Models the placement scenario: GPU compute ->
    /// device->host->device round trip -> GPU compute, per token.
    #[test]
    fn host_excursion_is_capturable_as_a_seam_and_illegal_inside_capture() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping host-excursion capture test: CUDA runtime unavailable");
            return;
        };
        let function = runtime.nvrtc_function(MODULE, SOURCE, "add_one").unwrap();
        let n = 48usize;
        let size = n * std::mem::size_of::<f32>();
        let buf0 = runtime.alloc_raw(size).unwrap();
        let buf1 = runtime.alloc_raw(size).unwrap();
        let buf2 = runtime.alloc_raw(size).unwrap();
        let buf3 = runtime.alloc_raw(size).unwrap();
        let initial = (0..n).map(|i| i as f32).collect::<Vec<_>>();
        // SAFETY: buf0 covers the whole slice.
        unsafe { runtime.htod(bytes(&initial), buf0) }.unwrap();

        let host_excursion = |src: &[f32]| -> Vec<f32> { src.iter().map(|v| v + 1.0).collect() };
        let capturable = TestKernel { capturable: true };

        // --- POSITIVE: host excursion as an EAGER SEAM between two segments ---
        // Segment 0 (captured GPU): buf0 -> buf1.
        runtime.begin_graph_capture(&[&capturable]).unwrap();
        launch_add_one(&runtime, &function, buf0, buf1, n);
        runtime.end_graph_capture().unwrap();
        runtime.replay_graph_segment(0).unwrap();
        // The excursion: D2H (buf1) -> host compute -> H2D (buf2). The D2H read
        // is a blocking host sync, legal here only because no capture is active
        // between segments.
        let seam_in = read_f32(&runtime, buf1, n);
        let seam_out = host_excursion(&seam_in);
        // SAFETY: buf2 covers the whole slice.
        unsafe { runtime.htod(bytes(&seam_out), buf2) }.unwrap();
        // Segment 1 (captured GPU): buf2 -> buf3.
        runtime.begin_graph_capture(&[&capturable]).unwrap();
        launch_add_one(&runtime, &function, buf2, buf3, n);
        runtime.end_graph_capture().unwrap();
        runtime.replay_graph_segment(1).unwrap();
        runtime.synchronize().unwrap();
        let expected = initial.iter().map(|v| v + 3.0).collect::<Vec<_>>();
        assert_eq!(
            read_f32(&runtime, buf3, n),
            expected,
            "segmented host-seam capture must be token-exact"
        );
        assert_eq!(runtime.graph_segment_count().unwrap(), 2);

        // --- REPLAY: re-run both segments AND the host seam for a new token --
        let mutated = (0..n).map(|i| 500.0 + i as f32).collect::<Vec<_>>();
        // SAFETY: buf0 is the same live allocation captured by segment 0.
        unsafe { runtime.htod(bytes(&mutated), buf0) }.unwrap();
        runtime.replay_graph_segment(0).unwrap();
        let seam_in2 = read_f32(&runtime, buf1, n);
        let seam_out2 = host_excursion(&seam_in2);
        // SAFETY: buf2 covers the whole slice.
        unsafe { runtime.htod(bytes(&seam_out2), buf2) }.unwrap();
        runtime.replay_graph_segment(1).unwrap();
        runtime.synchronize().unwrap();
        assert_eq!(
            read_f32(&runtime, buf3, n),
            mutated.iter().map(|v| v + 3.0).collect::<Vec<_>>(),
            "a per-token host excursion must replay correctly"
        );
        runtime.reset_graph().unwrap();

        // --- NEGATIVE: the SAME excursion INSIDE an active capture -----------
        // A host-consuming D2H needs a stream drain; that drain is illegal while
        // capturing and invalidates the graph. This is why a monolithic capture
        // ACROSS the excursion cannot work and segmentation is mandatory.
        //
        // The drain has to be the *unconditional* one. `synchronize()` has been
        // a no-op by default since eager-sync deferral landed (#1383), so it can
        // neither invalidate a capture nor be relied on to detect one. What
        // actually makes the excursion illegal is `dtoh`'s internal
        // `force_synchronize`, which `drain_for_unmap` is the public spelling
        // of -- so that is what the negative case must exercise.
        runtime.begin_graph_capture(&[&capturable]).unwrap();
        launch_add_one(&runtime, &function, buf0, buf1, n);
        assert!(
            runtime.synchronize().is_ok(),
            "the deferred synchronize is a no-op and must not be mistaken for a capture barrier"
        );
        assert!(
            runtime.drain_for_unmap().is_err(),
            "a host-consuming drain inside active capture must invalidate it"
        );
        runtime.abort_graph_capture().unwrap();
        runtime.reset_graph().ok();

        // SAFETY: reset dropped all segment ownership before frees.
        unsafe {
            runtime.free_raw(buf3).unwrap();
            runtime.free_raw(buf2).unwrap();
            runtime.free_raw(buf1).unwrap();
            runtime.free_raw(buf0).unwrap();
        }
    }

    /// BENCHMARK + bit-identity. Prints the per-token "price of admission" for
    /// placement on the native path: the overhead of splitting a monolithic
    /// captured step into two segments around one host excursion seam (D2H
    /// ~10 KB -> host touch -> H2D ~10 KB), versus a single monolithic capture.
    /// Also asserts the segmented+seam output is BIT-IDENTICAL to the monolithic
    /// reference, so it verifies correctness rather than only timing. Keeping
    /// this lets the next person re-derive the seam price on their own hardware
    /// instead of trusting a number measured on an RTX 4060.
    #[test]
    fn bench_host_seam_price_of_admission() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping host-seam bench: CUDA runtime unavailable");
            return;
        };
        let function = runtime.nvrtc_function(MODULE, SOURCE, "add_one").unwrap();
        // ~10 KB embedding-gather output: 2560 f32 = 10240 B.
        let n = 2560usize;
        let size = n * std::mem::size_of::<f32>();
        let a = runtime.alloc_raw(size).unwrap();
        let b = runtime.alloc_raw(size).unwrap();
        let c = runtime.alloc_raw(size).unwrap();
        let logits = runtime.alloc_raw(size).unwrap();
        let init = (0..n).map(|i| i as f32).collect::<Vec<_>>();
        // SAFETY: a covers the whole slice.
        unsafe { runtime.htod(bytes(&init), a) }.unwrap();
        let capturable = TestKernel { capturable: true };
        let iters = 500u32;
        let warmup = 50u32;

        // (A) Monolithic: 4 launches captured as ONE graph; a->b->c->b->logits.
        // Reference output = input + 4.
        runtime.begin_graph_capture(&[&capturable]).unwrap();
        launch_add_one(&runtime, &function, a, b, n);
        launch_add_one(&runtime, &function, b, c, n);
        launch_add_one(&runtime, &function, c, b, n);
        launch_add_one(&runtime, &function, b, logits, n);
        runtime.end_graph_capture().unwrap();
        let mut sink = vec![0.0f32; n];
        let read_logits = |runtime: &CudaRuntime, sink: &mut Vec<f32>| {
            // Sampling-style D2H + sync that decode already pays per token.
            // SAFETY: logits is a live n-f32 allocation; sink matches.
            unsafe {
                runtime
                    .dtoh(
                        std::slice::from_raw_parts_mut(
                            sink.as_mut_ptr().cast::<u8>(),
                            std::mem::size_of_val(sink.as_slice()),
                        ),
                        logits,
                    )
                    .unwrap();
            }
            runtime.synchronize().unwrap();
        };
        for _ in 0..warmup {
            runtime.replay_graph().unwrap();
            read_logits(&runtime, &mut sink);
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            runtime.replay_graph().unwrap();
            read_logits(&runtime, &mut sink);
        }
        let mono_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let mono_ref = sink.clone();
        assert_eq!(
            mono_ref,
            init.iter().map(|v| v + 4.0).collect::<Vec<_>>(),
            "monolithic reference must be input + 4"
        );
        runtime.reset_graph().unwrap();

        // (B) Segmented / placed: the same +4 result, but the 3rd of the four
        // ops runs on the HOST as an excursion seam instead of on the device.
        // seg0: a->b->c (+2, 2 launches) ; host seam: c(+2) -> +1 -> b(+3) ;
        // seg1: b->logits (+1, 1 launch) => logits = +4. Total 3 GPU launches +
        // 1 host op = +4, so the output must be BIT-IDENTICAL to monolithic.
        runtime.begin_graph_capture(&[&capturable]).unwrap();
        launch_add_one(&runtime, &function, a, b, n);
        launch_add_one(&runtime, &function, b, c, n);
        runtime.end_graph_capture().unwrap();
        runtime.replay_graph_segment(0).unwrap();
        // materialize the seam once so seg1 capture reads real bytes.
        let seam = read_f32(&runtime, c, n);
        let seam: Vec<f32> = seam.iter().map(|v| v + 1.0).collect();
        // SAFETY: b covers the whole slice.
        unsafe { runtime.htod(bytes(&seam), b) }.unwrap();
        runtime.begin_graph_capture(&[&capturable]).unwrap();
        launch_add_one(&runtime, &function, b, logits, n);
        runtime.end_graph_capture().unwrap();
        let mut host = vec![0.0f32; n];
        let seam_step = |runtime: &CudaRuntime, host: &mut Vec<f32>, sink: &mut Vec<f32>| {
            runtime.replay_graph_segment(0).unwrap();
            // Host excursion (the placement seam): D2H ~10 KB, host op, H2D.
            // This stands in for one device op, so the total work is unchanged.
            // SAFETY: c is a live n-f32 allocation; host matches.
            unsafe {
                runtime
                    .dtoh(
                        std::slice::from_raw_parts_mut(
                            host.as_mut_ptr().cast::<u8>(),
                            std::mem::size_of_val(host.as_slice()),
                        ),
                        c,
                    )
                    .unwrap();
            }
            for v in host.iter_mut() {
                *v += 1.0;
            }
            // SAFETY: b covers the whole host slice.
            unsafe { runtime.htod(bytes(host), b) }.unwrap();
            runtime.replay_graph_segment(1).unwrap();
            read_logits(runtime, sink);
        };
        for _ in 0..warmup {
            seam_step(&runtime, &mut host, &mut sink);
        }
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            seam_step(&runtime, &mut host, &mut sink);
        }
        let seam_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
        runtime.reset_graph().unwrap();

        // Bit-identity: placement must not change the numeric result.
        assert_eq!(
            sink, mono_ref,
            "segmented + host-seam output must be bit-identical to the monolithic reference"
        );

        let delta_us = (seam_ms - mono_ms) * 1e3;
        eprintln!(
            "SEAM PRICE (this GPU): monolithic={mono_ms:.4} ms/token, \
             segmented+host-seam={seam_ms:.4} ms/token, seam_overhead={delta_us:.1} us/token"
        );

        // SAFETY: reset dropped graph ownership before frees.
        unsafe {
            runtime.free_raw(logits).unwrap();
            runtime.free_raw(c).unwrap();
            runtime.free_raw(b).unwrap();
            runtime.free_raw(a).unwrap();
        }
    }

    #[test]
    fn segmented_capture_interleaves_two_graphs_with_an_eager_seam() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping segmented CUDA graph test: CUDA runtime unavailable");
            return;
        };
        let function = runtime.nvrtc_function(MODULE, SOURCE, "add_one").unwrap();
        let n = 48usize;
        let size = n * std::mem::size_of::<f32>();
        // buf0 --seg0(captured)--> buf1 --eager seam--> buf2 --seg1(captured)--> buf3
        let buf0 = runtime.alloc_raw(size).unwrap();
        let buf1 = runtime.alloc_raw(size).unwrap();
        let buf2 = runtime.alloc_raw(size).unwrap();
        let buf3 = runtime.alloc_raw(size).unwrap();

        let initial = (0..n).map(|i| i as f32).collect::<Vec<_>>();
        // SAFETY: buf0 covers the complete host slice.
        unsafe { runtime.htod(bytes(&initial), buf0) }.unwrap();

        // Eager reference: three chained add_one launches (input + 3).
        launch_add_one(&runtime, &function, buf0, buf1, n);
        launch_add_one(&runtime, &function, buf1, buf2, n);
        launch_add_one(&runtime, &function, buf2, buf3, n);
        runtime.synchronize().unwrap();
        let eager = read_f32(&runtime, buf3, n);
        assert_eq!(eager, initial.iter().map(|v| v + 3.0).collect::<Vec<_>>());

        let capturable = TestKernel { capturable: true };
        let allocation_counts = runtime.allocation_counts();

        // --- Capture pass: record two segments around an eager seam ---------
        // Segment 0: buf0 -> buf1.
        runtime.begin_graph_capture(&[&capturable]).unwrap();
        launch_add_one(&runtime, &function, buf0, buf1, n);
        runtime.end_graph_capture().unwrap();
        // Materialize segment 0 so the eager seam reads real bytes (as the
        // executor does after ending each captured segment).
        runtime.replay_graph_segment(0).unwrap();
        // Eager seam: buf1 -> buf2 (non-capturable node runs on the stream).
        launch_add_one(&runtime, &function, buf1, buf2, n);
        // Segment 1: buf2 -> buf3.
        runtime.begin_graph_capture(&[&capturable]).unwrap();
        launch_add_one(&runtime, &function, buf2, buf3, n);
        runtime.end_graph_capture().unwrap();
        runtime.replay_graph_segment(1).unwrap();
        runtime.synchronize().unwrap();

        assert_eq!(runtime.graph_segment_count().unwrap(), 2);
        assert!(runtime.has_graph_executable().unwrap());
        // Token-exact: segmented capture pass equals the eager reference.
        assert_eq!(read_f32(&runtime, buf3, n), eager);

        // --- Replay steps: relaunch segments, re-run the eager seam ---------
        let mutated = (0..n).map(|i| 500.0 + i as f32).collect::<Vec<_>>();
        // SAFETY: buf0 remains the same live allocation captured by segment 0.
        unsafe { runtime.htod(bytes(&mutated), buf0) }.unwrap();
        runtime.replay_graph_segment(0).unwrap();
        launch_add_one(&runtime, &function, buf1, buf2, n);
        runtime.replay_graph_segment(1).unwrap();
        runtime.synchronize().unwrap();
        let replayed = read_f32(&runtime, buf3, n);
        assert_eq!(
            replayed,
            mutated.iter().map(|v| v + 3.0).collect::<Vec<_>>()
        );
        assert_ne!(replayed, eager);
        // No per-step device allocations across capture + replay.
        assert_eq!(runtime.allocation_counts(), allocation_counts);

        assert!(runtime.reset_graph().unwrap());
        assert!(!runtime.has_graph_executable().unwrap());
        assert_eq!(runtime.graph_segment_count().unwrap(), 0);
        // SAFETY: reset dropped all segment ownership before the buffers are freed.
        unsafe {
            runtime.free_raw(buf3).unwrap();
            runtime.free_raw(buf2).unwrap();
            runtime.free_raw(buf1).unwrap();
            runtime.free_raw(buf0).unwrap();
        }
    }

    #[test]
    fn mid_segment_capture_failure_is_recoverable_via_abort() {
        // Regression: a node failing mid-record during segmented capture must
        // leave the CUDA stream/lifecycle RECOVERABLE. The old cleanup called
        // reset() without ending the capture, but reset() is rejected while the
        // stream is still capturing, so the stream stayed wedged in capture mode
        // and every later launch failed with STREAM_CAPTURE_INVALIDATED. The fix
        // ends/aborts the capture before reset, restoring the invariant "capture
        // is always ended before reset".
        let Some(runtime) = runtime() else {
            eprintln!("skipping mid-capture recovery test: CUDA runtime unavailable");
            return;
        };
        let function = runtime.nvrtc_function(MODULE, SOURCE, "add_one").unwrap();
        let n = 32usize;
        let size = n * std::mem::size_of::<f32>();
        let input_ptr = runtime.alloc_raw(size).unwrap();
        let output_ptr = runtime.alloc_raw(size).unwrap();
        let initial = (0..n).map(|i| i as f32).collect::<Vec<_>>();
        // SAFETY: input_ptr covers the complete host slice.
        unsafe { runtime.htod(bytes(&initial), input_ptr) }.unwrap();
        let expected = initial.iter().map(|v| v + 1.0).collect::<Vec<_>>();

        let capturable = TestKernel { capturable: true };

        // --- Reproduce a mid-segment kernel failure during capture ----------
        // Begin recording and launch one node into the segment, then trip the
        // exact illegal operation a Supported-but-unconditionally-syncing kernel
        // would perform inside a captured segment: an unconditional stream drain
        // during capture. This invalidates the capture (CUDA_ERROR_STREAM_CAPTURE_*),
        // which is the error that reaches the executor's cleanup path.
        //
        // It has to be the unconditional drain. `synchronize()` has been a no-op
        // by default since eager-sync deferral landed (#1383), so a kernel that
        // calls it does not invalidate anything; the kernels that still force a
        // drain are the ones that go through `force_synchronize`, of which
        // `drain_for_unmap` is the public spelling.
        runtime.begin_graph_capture(&[&capturable]).unwrap();
        launch_add_one(&runtime, &function, input_ptr, output_ptr, n);
        assert!(runtime.is_capturing().unwrap());
        assert!(
            runtime.drain_for_unmap().is_err(),
            "an unconditional stream drain mid-capture is illegal and must error"
        );

        // The wedge: while the stream is still (invalidly) capturing, reset is
        // rejected. The OLD path stopped here, leaving the stream stuck.
        assert!(
            runtime.reset_graph().is_err(),
            "reset must be rejected while the stream is still capturing"
        );

        // The fix: abort ends the stream capture and returns the lifecycle to
        // idle, so a subsequent reset succeeds and the session can decline
        // cleanly to eager execution.
        runtime.abort_graph_capture().unwrap();
        assert!(
            !runtime.is_capturing().unwrap(),
            "abort must take the stream out of capture mode"
        );
        assert!(
            !runtime.reset_graph().unwrap(),
            "reset succeeds after abort; no executable was installed"
        );
        assert!(!runtime.has_graph_executable().unwrap());

        // (a)/(b) The same stream runs eager work again — no wedge.
        launch_add_one(&runtime, &function, input_ptr, output_ptr, n);
        runtime.synchronize().unwrap();
        assert_eq!(read_f32(&runtime, output_ptr, n), expected);

        // And a fresh capture/replay cycle succeeds on the recovered stream.
        runtime.begin_graph_capture(&[&capturable]).unwrap();
        launch_add_one(&runtime, &function, input_ptr, output_ptr, n);
        runtime.end_graph_capture().unwrap();
        runtime.replay_graph().unwrap();
        runtime.synchronize().unwrap();
        assert_eq!(read_f32(&runtime, output_ptr, n), expected);
        assert!(runtime.reset_graph().unwrap());

        // SAFETY: reset dropped graph ownership before either buffer is freed.
        unsafe {
            runtime.free_raw(output_ptr).unwrap();
            runtime.free_raw(input_ptr).unwrap();
        }
    }

    #[test]
    fn incompatible_sequence_is_rejected_before_stream_capture() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping CUDA graph audit test: CUDA runtime unavailable");
            return;
        };
        let incompatible = TestKernel { capturable: false };

        let error = runtime.begin_graph_capture(&[&incompatible]).unwrap_err();
        assert!(error.to_string().contains("rejected before begin_capture"));
        assert_eq!(
            runtime.graph_capture_status().unwrap(),
            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        );
        assert!(!runtime.has_graph_executable().unwrap());
    }

    #[test]
    fn holds_single_capture_tracks_whole_subgraph_segment() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping holds_single_capture test: CUDA runtime unavailable");
            return;
        };
        let function = runtime.nvrtc_function(MODULE, SOURCE, "add_one").unwrap();
        let n = 16usize;
        let size = n * std::mem::size_of::<f32>();
        let input_ptr = runtime.alloc_raw(size).unwrap();
        let output_ptr = runtime.alloc_raw(size).unwrap();

        let lifecycle = CudaGraphLifecycle::new(runtime.stream().clone());
        // No capture installed yet.
        assert!(!lifecycle.holds_single_capture().unwrap());

        // One whole-subgraph segment satisfies the option (c) retain invariant.
        lifecycle.begin(Vec::new()).unwrap();
        launch_add_one(&runtime, &function, input_ptr, output_ptr, n);
        lifecycle.end().unwrap();
        assert!(lifecycle.holds_single_capture().unwrap());
        assert_eq!(lifecycle.segment_count().unwrap(), 1);

        // A second appended segment (segmented capture) breaks the invariant.
        lifecycle.begin(Vec::new()).unwrap();
        launch_add_one(&runtime, &function, output_ptr, input_ptr, n);
        lifecycle.end().unwrap();
        assert!(!lifecycle.holds_single_capture().unwrap());
        assert_eq!(lifecycle.segment_count().unwrap(), 2);

        assert!(lifecycle.reset().unwrap());
        assert!(!lifecycle.holds_single_capture().unwrap());

        // SAFETY: reset dropped all segment ownership before the buffers are freed.
        unsafe {
            runtime.free_raw(output_ptr).unwrap();
            runtime.free_raw(input_ptr).unwrap();
        }
    }

    /// The runtime owns two independent captured-graph slots (`Primary` for the
    /// M=1 decode step, `Verify` for the MTP fixed-width verify step). This is
    /// the enabling invariant for replaying two differently-shaped decode graphs
    /// by shape key without per-step recapture: capturing/replaying/resetting one
    /// slot must never disturb the other's installed executable, even though both
    /// launch on the same compute stream.
    #[test]
    fn primary_and_verify_graph_slots_are_independent() {
        use onnx_runtime_ep_api::DeviceGraphSlot;

        let Some(runtime) = runtime() else {
            eprintln!("skipping two-slot graph test: CUDA runtime unavailable");
            return;
        };
        let function = runtime.nvrtc_function(MODULE, SOURCE, "add_one").unwrap();
        let n = 32usize;
        let size = n * std::mem::size_of::<f32>();
        let p_in = runtime.alloc_raw(size).unwrap();
        let p_out = runtime.alloc_raw(size).unwrap();
        let v_in = runtime.alloc_raw(size).unwrap();
        let v_mid = runtime.alloc_raw(size).unwrap();
        let v_out = runtime.alloc_raw(size).unwrap();
        let base = (0..n).map(|i| i as f32).collect::<Vec<_>>();
        // SAFETY: each pointer covers the whole slice.
        unsafe {
            runtime.htod(bytes(&base), p_in).unwrap();
            runtime.htod(bytes(&base), v_in).unwrap();
        }
        let capturable = TestKernel { capturable: true };

        // Primary slot: a single add_one (out = in + 1).
        runtime
            .begin_graph_capture_in(DeviceGraphSlot::Primary, &[&capturable])
            .unwrap();
        launch_add_one(&runtime, &function, p_in, p_out, n);
        runtime
            .end_graph_capture_in(DeviceGraphSlot::Primary)
            .unwrap();

        // Verify slot: a different shape/topology (two chained add_one,
        // out = in + 2) captured while Primary already holds an executable.
        runtime
            .begin_graph_capture_in(DeviceGraphSlot::Verify, &[&capturable])
            .unwrap();
        launch_add_one(&runtime, &function, v_in, v_mid, n);
        launch_add_one(&runtime, &function, v_mid, v_out, n);
        runtime
            .end_graph_capture_in(DeviceGraphSlot::Verify)
            .unwrap();

        // Both slots hold an executable simultaneously.
        assert!(
            runtime
                .has_graph_executable_in(DeviceGraphSlot::Primary)
                .unwrap()
        );
        assert!(
            runtime
                .has_graph_executable_in(DeviceGraphSlot::Verify)
                .unwrap()
        );

        // Each slot replays its own graph independently, interleaved.
        for _ in 0..3 {
            runtime.replay_graph_in(DeviceGraphSlot::Primary).unwrap();
            runtime.replay_graph_in(DeviceGraphSlot::Verify).unwrap();
        }
        runtime.synchronize().unwrap();
        assert_eq!(
            read_f32(&runtime, p_out, n),
            base.iter().map(|v| v + 1.0).collect::<Vec<_>>(),
            "Primary slot must apply +1"
        );
        assert_eq!(
            read_f32(&runtime, v_out, n),
            base.iter().map(|v| v + 2.0).collect::<Vec<_>>(),
            "Verify slot must apply +2, undisturbed by Primary replays"
        );

        // Resetting Primary leaves Verify's installed executable intact.
        assert!(runtime.reset_graph_in(DeviceGraphSlot::Primary).unwrap());
        assert!(
            !runtime
                .has_graph_executable_in(DeviceGraphSlot::Primary)
                .unwrap()
        );
        assert!(
            runtime
                .has_graph_executable_in(DeviceGraphSlot::Verify)
                .unwrap(),
            "resetting Primary must not tear down the Verify slot"
        );
        runtime.replay_graph_in(DeviceGraphSlot::Verify).unwrap();
        runtime.synchronize().unwrap();
        assert_eq!(
            read_f32(&runtime, v_out, n),
            base.iter().map(|v| v + 2.0).collect::<Vec<_>>(),
        );

        assert!(runtime.reset_graph_in(DeviceGraphSlot::Verify).unwrap());
        // SAFETY: both slots reset, dropping all graph ownership before free.
        unsafe {
            runtime.free_raw(v_out).unwrap();
            runtime.free_raw(v_mid).unwrap();
            runtime.free_raw(v_in).unwrap();
            runtime.free_raw(p_out).unwrap();
            runtime.free_raw(p_in).unwrap();
        }
    }
}
