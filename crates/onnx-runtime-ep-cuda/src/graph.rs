//! Serialized ownership for the CUDA graph captured on an EP runtime stream.

use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::ThreadId;

use cudarc::driver::sys::{
    CUgraph, CUgraphExec, CUgraphInstantiate_flags, CUstreamCaptureMode, CUstreamCaptureStatus,
};
use cudarc::driver::{CudaStream, result};
use onnx_runtime_ep_api::{EpError, Result};

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
}

impl CapturedGraph {
    fn end_capture(
        stream: &Arc<CudaStream>,
        flags: CUgraphInstantiate_flags,
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
        }))
    }

    fn upload(&self) -> std::result::Result<(), cudarc::driver::DriverError> {
        self.stream.context().bind_to_thread()?;
        // SAFETY: this wrapper owns `graph_exec`, and the lifecycle mutex
        // serializes access on its owning stream.
        unsafe { result::graph::upload(self.graph_exec, self.stream.cu_stream()) }
    }

    fn launch(&self) -> std::result::Result<(), cudarc::driver::DriverError> {
        self.stream.context().bind_to_thread()?;
        // SAFETY: this wrapper owns `graph_exec`, and the lifecycle mutex
        // serializes access on its owning stream.
        unsafe { result::graph::launch(self.graph_exec, self.stream.cu_stream()) }
    }
}

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
    }
}

/// Owns the captured graph segments installed on one EP runtime stream.
///
/// `CapturedGraph` is intentionally neither `Send` nor `Sync`. CUDA permits graph
/// objects to cross threads only when every access is externally serialized.
/// This wrapper enforces that rule with one mutex and never exposes a graph
/// handle or performs graph work without holding its guard.
///
/// A whole-subgraph capture installs exactly one segment. Segmented capture —
/// used when only parts of a claimed subgraph are device-graph capturable —
/// installs one segment per maximal capturable run; the non-capturable seam
/// nodes execute eagerly between segment replays. Segments launch in capture
/// order and each is destroyed exactly once on reset/drop.
pub(crate) struct CudaGraphLifecycle {
    stream: Arc<CudaStream>,
    state: Mutex<LifecycleState>,
}

/// The capture flag and the ordered list of installed segment executables.
struct LifecycleState {
    capture: CaptureState,
    segments: Vec<CapturedGraph>,
}

// SAFETY: all access to the non-Send/non-Sync `CapturedGraph` handles is
// confined to `state`, every method holds that mutex for the complete CUDA graph
// API call, and every segment launches on its single owning `stream`.
unsafe impl Send for CudaGraphLifecycle {}
// SAFETY: the same serialized-access invariant covers shared references.
unsafe impl Sync for CudaGraphLifecycle {}

impl CudaGraphLifecycle {
    pub(crate) fn new(stream: Arc<CudaStream>) -> Self {
        Self {
            stream,
            state: Mutex::new(LifecycleState {
                capture: CaptureState::Idle,
                segments: Vec::new(),
            }),
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, LifecycleState>> {
        self.state.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep: CUDA graph lifecycle lock was poisoned".into())
        })
    }

    /// Begin recording a new segment. Additional segments may be captured while
    /// earlier ones are already installed (segmented capture); only a second
    /// concurrent capture is rejected.
    pub(crate) fn begin(&self) -> Result<()> {
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

        self.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|error| driver_err("begin CUDA graph stream capture", error))?;
        state.capture = CaptureState::Capturing(std::thread::current().id());
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
        let graph = CapturedGraph::end_capture(
            &self.stream,
            CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY,
        )
        .map_err(|error| driver_err("end and instantiate CUDA graph capture", error))?
        .ok_or_else(|| {
            EpError::KernelFailed(
                "cuda_ep: CUDA graph capture ended without producing a graph".into(),
            )
        })?;
        graph
            .upload()
            .map_err(|error| driver_err("upload CUDA graph executable", error))?;
        state.segments.push(graph);
        Ok(())
    }

    /// Replay every installed segment in capture order. For a whole-subgraph
    /// capture this is the single installed graph.
    pub(crate) fn replay(&self) -> Result<()> {
        let state = self.lock()?;
        if state.segments.is_empty() {
            return Err(EpError::KernelFailed(
                "cuda_ep: cannot replay CUDA graph because no executable is installed".into(),
            ));
        }
        for graph in &state.segments {
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
        let state = self.lock()?;
        let graph = state.segments.get(index).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep: cannot replay CUDA graph segment {index}; only {} segment(s) installed",
                state.segments.len()
            ))
        })?;
        graph
            .launch()
            .map_err(|error| driver_err("launch CUDA graph segment", error))
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
        Ok(had_graph)
    }

    pub(crate) fn has_executable(&self) -> Result<bool> {
        Ok(!self.lock()?.segments.is_empty())
    }

    /// Number of installed segment executables (1 for a whole-subgraph capture).
    pub(crate) fn segment_count(&self) -> Result<usize> {
        Ok(self.lock()?.segments.len())
    }

    pub(crate) fn capture_status(&self) -> Result<CUstreamCaptureStatus> {
        let _state = self.lock()?;
        self.stream
            .capture_status()
            .map_err(|error| driver_err("query CUDA graph capture status", error))
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
}
