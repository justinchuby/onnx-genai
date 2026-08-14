//! Minimal dynamically-loaded CUDA runtime (`cudart`) shim.
//!
//! WHY THIS EXISTS: the shared-KV bucketing fix grows the device-resident KV
//! buffers as the sequence crosses power-of-two bucket boundaries, which
//! requires copying the already-valid KV prefix from the old (smaller) device
//! buffer into the new (larger) one. The obvious ORT primitive for this,
//! `OrtApi::CopyTensors`, only works when an env-level `IDataTransfer` is
//! registered by a *plugin* execution provider (`OrtEpDevice`). The built-in
//! CUDA EP appended via `SessionOptionsAppendExecutionProvider_V2` does NOT
//! register that transfer, so `CopyTensors` fails at runtime with
//! "Data transfer implementation between source and destination device was not
//! found. (code: 9)". We therefore bypass ORT entirely and issue the copy with
//! a direct `cudaMemcpy(..., cudaMemcpyDeviceToDevice)` on the raw device
//! pointers backing the KV tensors.
//!
//! `cudart` is loaded dynamically (via `libloading`) rather than linked at
//! build time, so a plain `--features cuda` build does not require the CUDA
//! toolkit's import libraries — only that `cudart` is discoverable at runtime,
//! which it already must be for the CUDA EP to function. The loaded library and
//! resolved symbols are cached in a process-wide `OnceLock` so growth (which
//! happens only O(log length) times per generation) never reloads it.

use std::os::raw::c_void;
use std::sync::OnceLock;

use libloading::Library;

use crate::{OrtError, Result};

/// `cudaMemcpyKind::cudaMemcpyHostToDevice`.
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;

/// `cudaMemcpyKind::cudaMemcpyDeviceToHost`.
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

/// `cudaMemcpyKind::cudaMemcpyDeviceToDevice`.
const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

type CudaMemcpyFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32;
type CudaMemsetFn = unsafe extern "C" fn(*mut c_void, i32, usize) -> i32;
type CudaDeviceSynchronizeFn = unsafe extern "C" fn() -> i32;
type CudaSetDeviceFn = unsafe extern "C" fn(i32) -> i32;
type CudaGetDeviceFn = unsafe extern "C" fn(*mut i32) -> i32;
type CudaMemGetInfoFn = unsafe extern "C" fn(*mut usize, *mut usize) -> i32;
type CudaStreamCreateFn = unsafe extern "C" fn(*mut *mut c_void) -> i32;
type CudaStreamDestroyFn = unsafe extern "C" fn(*mut c_void) -> i32;
type CudaStreamSynchronizeFn = unsafe extern "C" fn(*mut c_void) -> i32;

struct CudaRt {
    // Kept alive so the resolved function pointers remain valid; never called
    // directly after construction.
    _lib: Library,
    memcpy: CudaMemcpyFn,
    memset: CudaMemsetFn,
    device_synchronize: CudaDeviceSynchronizeFn,
    set_device: CudaSetDeviceFn,
    get_device: CudaGetDeviceFn,
    mem_get_info: CudaMemGetInfoFn,
    stream_create: CudaStreamCreateFn,
    stream_destroy: CudaStreamDestroyFn,
    stream_synchronize: CudaStreamSynchronizeFn,
}

// SAFETY: the resolved `cudart` entry points are plain C functions that are
// safe to invoke from any thread; the `Library` handle is only stored to keep
// the module mapped and is never mutated after construction.
unsafe impl Send for CudaRt {}
unsafe impl Sync for CudaRt {}

static CUDART: OnceLock<std::result::Result<CudaRt, String>> = OnceLock::new();

/// Candidate `cudart` library names, most specific first. Windows ships
/// versioned DLLs (`cudart64_12.dll` for CUDA 12.x, older `cudart64_120.dll`),
/// while the bare name lets the platform loader resolve `libcudart.so` on Linux
/// or a name already on the search path.
const CUDART_CANDIDATES: &[&str] = &[
    "cudart64_12.dll",
    "cudart64_120.dll",
    "cudart",
    "libcudart.so.12",
    "libcudart.so",
];

fn load() -> std::result::Result<CudaRt, String> {
    let mut last_err = String::from("no candidate library names were tried");
    for name in CUDART_CANDIDATES {
        // SAFETY: loading a shared library can run initializers; `cudart` is a
        // trusted NVIDIA runtime that the CUDA EP already loads in-process.
        let lib = match unsafe { Library::new(name) } {
            Ok(lib) => lib,
            Err(err) => {
                last_err = format!("{name}: {err}");
                continue;
            }
        };
        // SAFETY: the symbol signatures match the documented `cudart` ABI.
        let memcpy = unsafe { lib.get::<CudaMemcpyFn>(b"cudaMemcpy\0") };
        let memset = unsafe { lib.get::<CudaMemsetFn>(b"cudaMemset\0") };
        let device_synchronize =
            unsafe { lib.get::<CudaDeviceSynchronizeFn>(b"cudaDeviceSynchronize\0") };
        let set_device = unsafe { lib.get::<CudaSetDeviceFn>(b"cudaSetDevice\0") };
        let get_device = unsafe { lib.get::<CudaGetDeviceFn>(b"cudaGetDevice\0") };
        let mem_get_info = unsafe { lib.get::<CudaMemGetInfoFn>(b"cudaMemGetInfo\0") };
        let stream_create = unsafe { lib.get::<CudaStreamCreateFn>(b"cudaStreamCreate\0") };
        let stream_destroy = unsafe { lib.get::<CudaStreamDestroyFn>(b"cudaStreamDestroy\0") };
        let stream_synchronize =
            unsafe { lib.get::<CudaStreamSynchronizeFn>(b"cudaStreamSynchronize\0") };
        match (
            memcpy,
            memset,
            device_synchronize,
            set_device,
            get_device,
            mem_get_info,
            stream_create,
            stream_destroy,
            stream_synchronize,
        ) {
            (
                Ok(memcpy),
                Ok(memset),
                Ok(device_synchronize),
                Ok(set_device),
                Ok(get_device),
                Ok(mem_get_info),
                Ok(stream_create),
                Ok(stream_destroy),
                Ok(stream_synchronize),
            ) => {
                // Copy the function pointers out before `lib` is moved into the
                // struct; the borrows on `lib` end here.
                let memcpy = *memcpy;
                let memset = *memset;
                let device_synchronize = *device_synchronize;
                let set_device = *set_device;
                let get_device = *get_device;
                let mem_get_info = *mem_get_info;
                let stream_create = *stream_create;
                let stream_destroy = *stream_destroy;
                let stream_synchronize = *stream_synchronize;
                return Ok(CudaRt {
                    _lib: lib,
                    memcpy,
                    memset,
                    device_synchronize,
                    set_device,
                    get_device,
                    mem_get_info,
                    stream_create,
                    stream_destroy,
                    stream_synchronize,
                });
            }
            _ => {
                last_err = format!(
                    "{name}: missing cudaMemcpy/cudaMemset/cudaDeviceSynchronize/cudaSetDevice/cudaGetDevice/cudaMemGetInfo/cudaStreamCreate/cudaStreamDestroy/cudaStreamSynchronize symbol"
                );
            }
        }
    }
    Err(format!("could not load CUDA runtime (cudart): {last_err}"))
}

fn runtime() -> Result<&'static CudaRt> {
    CUDART
        .get_or_init(load)
        .as_ref()
        .map_err(|err| OrtError::InvalidArgument(err.clone()))
}

/// Block the host until all outstanding device work (on every stream) has
/// completed.
///
/// The shared-KV grow copy runs on the default stream, while the ORT CUDA EP
/// executes on its own (non-blocking) stream. Without a full-device barrier the
/// copy is unordered relative to the EP's KV writes (before) and reads (after),
/// which silently corrupts the cache. Growth is rare (O(log length)), so the
/// synchronization cost is negligible.
pub fn device_synchronize() -> Result<()> {
    let rt = runtime()?;
    // SAFETY: `cudaDeviceSynchronize` takes no arguments and matches the
    // `cudart` ABI.
    let code = unsafe { (rt.device_synchronize)() };
    if code != 0 {
        return Err(OrtError::InvalidArgument(format!(
            "cudaDeviceSynchronize failed with CUDA error code {code}"
        )));
    }
    Ok(())
}

/// Create a CUDA compute stream for one pipeline to share across its sessions.
///
/// ORT gives each session its own stream by default, and alternating between
/// per-session streams within one step costs far more than the extra sessions'
/// work: on an H200 with stock ORT 1.28 a 2499-node captured decoder island
/// replays in 15.40 ms alone and 16.11 ms with a second single-node session
/// between replays on its own stream. Sharing one stream also gives the
/// pipeline a single ordered timeline, which is what lets stream-ordered copies
/// between sessions replace device-wide barriers. Upstream ORT GenAI configures
/// its CUDA sessions the same way.
///
/// Each call returns a *distinct* stream, and callers are expected to create one
/// per pipeline rather than one per device. Stream capture is a property of the
/// stream, not the thread: ORT captures with `cudaStreamCaptureModeGlobal`, so
/// if two independently driven pipelines shared a stream, work one of them
/// enqueued could be silently recorded into the other's graph instead of
/// executing. A stream per pipeline keeps that impossible.
///
/// The stream is created with `cudaStreamCreate`, matching GenAI: a blocking
/// stream stays ordered against the legacy default stream that the engine's own
/// host-visible copies use, so those copies need no extra barrier.
///
/// The stream is owned. It is handed out as an `Arc`, every session built from
/// options carrying it keeps a clone, and `Drop` destroys it once the last of
/// them is gone — so a server that loads and unloads models returns each
/// pipeline's stream instead of accumulating one per load. The type is
/// deliberately neither `Copy` nor `Clone`, so the number of owners is always
/// known.
pub struct CudaComputeStream {
    handle: std::ptr::NonNull<c_void>,
    device_id: i32,
}

// SAFETY: a CUDA stream handle is usable from any thread; CUDA serialises the
// work issued to it. The handle is immutable after construction.
unsafe impl Send for CudaComputeStream {}
unsafe impl Sync for CudaComputeStream {}

impl CudaComputeStream {
    /// Create a stream on `device_id`, owned by the returned handle.
    pub fn new(device_id: i32) -> Result<std::sync::Arc<Self>> {
        let _guard = DeviceGuard::set(device_id)?;
        let rt = runtime()?;
        let mut stream: *mut c_void = std::ptr::null_mut();
        // SAFETY: `stream` is a valid out-parameter and the signature matches the
        // documented `cudart` ABI for `cudaStreamCreate`.
        let code = unsafe { (rt.stream_create)(&mut stream) };
        if code != 0 {
            return Err(OrtError::InvalidArgument(format!(
                "cudaStreamCreate failed with CUDA error code {code}"
            )));
        }
        let handle = std::ptr::NonNull::new(stream).ok_or(OrtError::NullPointer)?;
        Ok(std::sync::Arc::new(Self { handle, device_id }))
    }

    /// The raw stream handle, for the provider option that takes one.
    #[must_use]
    pub fn handle(&self) -> usize {
        self.handle.as_ptr() as usize
    }

    /// The device this stream belongs to. A stream is only valid on its device.
    #[must_use]
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Block until everything queued on this stream has finished.
    ///
    /// Also the cheapest way to ask the driver whether the stream is still a
    /// live handle: a destroyed stream fails here rather than succeeding
    /// silently.
    pub fn synchronize(&self) -> Result<()> {
        let _guard = DeviceGuard::set(self.device_id)?;
        let rt = runtime()?;
        // SAFETY: the handle came from `cudaStreamCreate` on this device and is
        // still owned by `self`.
        let code = unsafe { (rt.stream_synchronize)(self.handle.as_ptr()) };
        if code != 0 {
            return Err(OrtError::InvalidArgument(format!(
                "cudaStreamSynchronize failed with CUDA error code {code}"
            )));
        }
        Ok(())
    }
}

impl std::fmt::Debug for CudaComputeStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaComputeStream")
            .field("handle", &self.handle())
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl Drop for CudaComputeStream {
    fn drop(&mut self) {
        // `cudaStreamDestroy` returns immediately and releases the stream once
        // the work already queued on it finishes, so no synchronize is needed.
        // Every session that used this stream is already gone by construction,
        // because each held an `Arc` to it.
        let Ok(_guard) = DeviceGuard::set(self.device_id) else {
            return;
        };
        let Ok(rt) = runtime() else {
            return;
        };
        // SAFETY: the handle came from `cudaStreamCreate` on this device and is
        // destroyed exactly once, here, when the last owner drops.
        let code = unsafe { (rt.stream_destroy)(self.handle.as_ptr()) };
        if code != 0 {
            tracing::warn!(
                device_id = self.device_id,
                code,
                "cudaStreamDestroy failed; this stream is leaked"
            );
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaDeviceMemoryInfo {
    pub free_bytes: usize,
    pub total_bytes: usize,
}

/// Query CUDA device memory after making `device_id` current.
pub fn device_memory_info(device_id: i32) -> Result<CudaDeviceMemoryInfo> {
    let _guard = DeviceGuard::set(device_id)?;
    let rt = runtime()?;
    let mut free_bytes = 0usize;
    let mut total_bytes = 0usize;
    // SAFETY: both pointers are valid out-parameters; `cudaMemGetInfo` matches
    // the cudart ABI and reads the current device selected by `DeviceGuard`.
    let code = unsafe { (rt.mem_get_info)(&mut free_bytes, &mut total_bytes) };
    if code != 0 {
        return Err(OrtError::InvalidArgument(format!(
            "cudaMemGetInfo failed with CUDA error code {code}"
        )));
    }
    Ok(CudaDeviceMemoryInfo {
        free_bytes,
        total_bytes,
    })
}

/// RAII guard that makes `device_id` the calling thread's current CUDA device
/// for the duration of a grow copy, restoring the previous current device on
/// drop.
///
/// All of the raw `cudart` calls below (`cudaMemcpy`, `cudaMemset`,
/// `cudaDeviceSynchronize`) act on the thread's *current* device, but the KV
/// buffers live on the EP's configured device (`ONNX_GENAI_CUDA_DEVICE`, which
/// may be non-zero). Without pinning, the pre/post-copy barriers could
/// synchronize the wrong device and fail to order the copy against the EP's
/// stream — the exact race the barriers exist to prevent. Pinning is cheap and
/// growth is rare (O(log length)).
pub struct DeviceGuard {
    prev: i32,
    restore: bool,
}

impl DeviceGuard {
    /// Set the current CUDA device to `device_id`, remembering the previous one.
    pub fn set(device_id: i32) -> Result<Self> {
        let rt = runtime()?;
        let mut prev: i32 = 0;
        // SAFETY: `prev` is a valid out-parameter; `cudaGetDevice` matches the
        // `cudart` ABI.
        let code = unsafe { (rt.get_device)(&mut prev) };
        if code != 0 {
            return Err(OrtError::InvalidArgument(format!(
                "cudaGetDevice failed with CUDA error code {code}"
            )));
        }
        // Only switch (and later restore) when the target differs, so the common
        // single-GPU / device-0 path incurs no extra `cudaSetDevice` calls.
        if prev == device_id {
            return Ok(Self {
                prev,
                restore: false,
            });
        }
        // SAFETY: `cudaSetDevice` matches the `cudart` ABI.
        let code = unsafe { (rt.set_device)(device_id) };
        if code != 0 {
            return Err(OrtError::InvalidArgument(format!(
                "cudaSetDevice({device_id}) failed with CUDA error code {code}"
            )));
        }
        Ok(Self {
            prev,
            restore: true,
        })
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        if !self.restore {
            return;
        }
        if let Ok(rt) = runtime() {
            // SAFETY: `cudaSetDevice` matches the `cudart` ABI; a restore failure
            // is best-effort (the process is likely already erroring out) so the
            // return code is intentionally ignored.
            let _ = unsafe { (rt.set_device)(self.prev) };
        }
    }
}

/// Zero `bytes` of device memory at device address `dst`.
///
/// Used to define the tail of a freshly allocated (uninitialized) KV bucket so
/// positions past the valid length are deterministic zeros.
pub fn memset_zero(dst: usize, bytes: usize) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    let rt = runtime()?;
    // SAFETY: `dst` is a valid device pointer owned by a live KV tensor with at
    // least `bytes` bytes of capacity; `memset` matches the `cudart` ABI.
    let code = unsafe { (rt.memset)(dst as *mut c_void, 0, bytes) };
    if code != 0 {
        return Err(OrtError::InvalidArgument(format!(
            "cudaMemset failed with CUDA error code {code}"
        )));
    }
    Ok(())
}

/// Copy `src.len()` bytes from host memory `src` into device address `dst`
/// (`cudaMemcpyHostToDevice`).
///
/// WHY THIS EXISTS: the static-shape captured decode loop keeps its small
/// dynamic inputs (`input_ids`, `position_ids`, `attention_mask`) device-resident
/// at fixed addresses so a captured CUDA graph reads them in place on every
/// replay (see the ORT IoBinding + CUDA-graph note in `decode.rs`, issue
/// microsoft/onnxruntime#29782). Each token refreshes those buffers with this
/// host->device copy instead of clearing and re-binding the whole IoBinding set.
///
/// This issues `cudaMemcpy` on cudart's default stream. Per the CUDA API
/// synchronization contract, a copy from *pageable* host memory to device memory
/// returns once the source has been staged for DMA, but **the DMA to device
/// memory may not have completed** when the call returns. ORT replays the
/// captured graph on a stream created with `cudaStreamNonBlocking`, which does
/// not serialize against the default stream, so the caller MUST synchronize the
/// device between the input refresh and the replay to make the fresh bytes
/// globally visible (RAW ordering — done in `CaptureState::write_step_inputs_device`).
/// Ordering against the *previous* replay's read of these buffers (WAR) is
/// guaranteed separately: the device sampler fully synchronizes the device at the
/// end of every captured step before the next step overwrites the inputs.
pub fn memcpy_host_to_device(dst: usize, src: &[u8]) -> Result<()> {
    if src.is_empty() {
        return Ok(());
    }
    let rt = runtime()?;
    // SAFETY: `dst` is a valid device pointer owned by a live tensor with at
    // least `src.len()` bytes of capacity; `src` is a valid host slice of that
    // length; `memcpy` matches the `cudart` ABI and the kind constant is the
    // documented enum value.
    let code = unsafe {
        (rt.memcpy)(
            dst as *mut c_void,
            src.as_ptr().cast::<c_void>(),
            src.len(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    };
    if code != 0 {
        return Err(OrtError::InvalidArgument(format!(
            "cudaMemcpy (host-to-device) failed with CUDA error code {code}"
        )));
    }
    Ok(())
}

/// Copy `src.len()` bytes from device address `src` into host memory `dst`
/// (`cudaMemcpyDeviceToHost`).
///
/// The runtime-API `cudaMemcpy` is synchronous for a pageable host destination,
/// so `dst` holds the copied bytes once this returns. Used by tests to read back
/// device-resident tensors written through the captured-decode input helpers.
pub fn memcpy_device_to_host(dst: &mut [u8], src: usize) -> Result<()> {
    if dst.is_empty() {
        return Ok(());
    }
    let rt = runtime()?;
    // SAFETY: `src` is a valid device pointer with at least `dst.len()` bytes of
    // capacity; `dst` is a valid host slice of that length; `memcpy` matches the
    // `cudart` ABI and the kind constant is the documented enum value.
    let code = unsafe {
        (rt.memcpy)(
            dst.as_mut_ptr().cast::<c_void>(),
            src as *const c_void,
            dst.len(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    };
    if code != 0 {
        return Err(OrtError::InvalidArgument(format!(
            "cudaMemcpy (device-to-host) failed with CUDA error code {code}"
        )));
    }
    Ok(())
}

/// Copy `bytes` from device address `src` to device address `dst`
/// (`cudaMemcpyDeviceToDevice`).
pub fn memcpy_device_to_device(dst: usize, src: usize, bytes: usize) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    let rt = runtime()?;
    // SAFETY: `src`/`dst` are valid, non-overlapping device pointers with at
    // least `bytes` bytes of capacity (distinct KV buffers); `memcpy` matches
    // the `cudart` ABI and the kind constant is the documented enum value.
    let code = unsafe {
        (rt.memcpy)(
            dst as *mut c_void,
            src as *const c_void,
            bytes,
            CUDA_MEMCPY_DEVICE_TO_DEVICE,
        )
    };
    if code != 0 {
        return Err(OrtError::InvalidArgument(format!(
            "cudaMemcpy (device-to-device) failed with CUDA error code {code}"
        )));
    }
    Ok(())
}

/// Ownership and isolation of the shared compute stream.
///
/// These are the properties a serving process depends on: a pipeline that is
/// unloaded must give its stream back, and two pipelines must never end up on
/// one stream, because ONNX Runtime captures with `cudaStreamCaptureModeGlobal`
/// and would record one pipeline's work into the other's graph.
#[cfg(test)]
mod compute_stream_ownership {

    use crate::session::{SessionOptions, ep_selection};

    #[test]
    #[ignore = "requires a CUDA device"]
    fn each_pipeline_gets_its_own_stream() {
        let mut first = SessionOptions::with_execution_provider(ep_selection("cuda"));
        let mut second = SessionOptions::with_execution_provider(ep_selection("cuda"));
        first.share_cuda_compute_stream();
        second.share_cuda_compute_stream();
        let (Some(a), Some(b)) = (
            first.cuda_user_compute_stream.as_ref(),
            second.cuda_user_compute_stream.as_ref(),
        ) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        assert_ne!(
            a.handle(),
            b.handle(),
            "two independently driven pipelines must not share a capture stream"
        );
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn cloning_options_shares_the_stream_but_asking_again_does_not() {
        let mut pipeline = SessionOptions::with_execution_provider(ep_selection("cuda"));
        pipeline.share_cuda_compute_stream();
        let Some(original) = pipeline
            .cuda_user_compute_stream
            .as_ref()
            .map(|s| s.handle())
        else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        // Cloning is how one pipeline builds several sessions on one timeline.
        let sibling = pipeline.clone();
        assert_eq!(
            sibling
                .cuda_user_compute_stream
                .as_ref()
                .map(|s| s.handle()),
            Some(original),
            "sessions of one pipeline must share its stream"
        );

        // But a clone that is then made into its own pipeline must not inherit
        // the first pipeline's timeline, even though it was cloned from it.
        let mut adopted = pipeline.clone();
        adopted.share_cuda_compute_stream();
        assert_ne!(
            adopted
                .cuda_user_compute_stream
                .as_ref()
                .map(|s| s.handle()),
            Some(original),
            "asking for a shared stream must always install a fresh one"
        );
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn unloading_a_pipeline_returns_its_stream() {
        // A server that loads and unloads models must not accumulate a stream
        // per load. Two streams that are alive at the same time cannot share a
        // handle, so a batch held together gives a set of distinct handles;
        // if dropping that batch really releases them, creating the same number
        // again hands back handles from that set.
        let batch = |count: usize| -> Option<Vec<usize>> {
            let mut held = Vec::new();
            for _ in 0..count {
                let mut options = SessionOptions::with_execution_provider(ep_selection("cuda"));
                options.share_cuda_compute_stream();
                held.push(options.cuda_user_compute_stream.clone()?);
            }
            Some(held.iter().map(|stream| stream.handle()).collect())
        };
        let Some(first) = batch(8) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let live: std::collections::HashSet<_> = first.iter().copied().collect();
        assert_eq!(
            live.len(),
            first.len(),
            "streams alive together must differ"
        );
        // `first`'s owners were dropped inside `batch`, so this batch is created
        // against a driver that has had all eight returned to it.
        let second = batch(8).expect("the first batch built, so this one must too");
        let reused = second.iter().filter(|handle| live.contains(handle)).count();
        assert!(
            reused > 0,
            "no handle from the released batch came back, so releasing them did nothing: \
             {first:?} then {second:?}"
        );
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_device_change_drops_a_stream_from_the_old_device() {
        let mut options = SessionOptions::with_execution_provider(ep_selection("cuda"));
        options.share_cuda_compute_stream();
        if options.cuda_user_compute_stream.is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        // Selecting a provider with no CUDA device leaves the old stream
        // pointing at a device this session will not use.
        options.execution_providers =
            SessionOptions::with_execution_provider(ep_selection("cpu")).execution_providers;
        options.invalidate_stream_for_device_change();
        assert!(
            options.cuda_user_compute_stream.is_none(),
            "a stream from the previous device must not survive a provider change"
        );
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_stream_outlives_the_options_that_created_it() {
        // Sessions hold an `Arc`, so dropping the options must not destroy the
        // stream underneath them.
        let mut options = SessionOptions::with_execution_provider(ep_selection("cuda"));
        options.share_cuda_compute_stream();
        let Some(held) = options.cuda_user_compute_stream.clone() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        drop(options);
        // Touch the stream rather than re-reading an immutable field: a
        // destroyed handle fails here, a live one does not.
        held.synchronize()
            .expect("the stream must still be usable after its options are dropped");
        assert_eq!(std::sync::Arc::strong_count(&held), 1);
    }
}
