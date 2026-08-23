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
type CudaMemcpyAsyncFn =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32, *mut c_void) -> i32;
type CudaMemsetFn = unsafe extern "C" fn(*mut c_void, i32, usize) -> i32;
type CudaDeviceSynchronizeFn = unsafe extern "C" fn() -> i32;
type CudaSetDeviceFn = unsafe extern "C" fn(i32) -> i32;
type CudaGetDeviceFn = unsafe extern "C" fn(*mut i32) -> i32;
type CudaMemGetInfoFn = unsafe extern "C" fn(*mut usize, *mut usize) -> i32;
type CudaMallocFn = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
type CudaFreeFn = unsafe extern "C" fn(*mut c_void) -> i32;

struct CudaRt {
    // Kept alive so the resolved function pointers remain valid; never called
    // directly after construction.
    _lib: Library,
    memcpy: CudaMemcpyFn,
    memcpy_async: CudaMemcpyAsyncFn,
    memset: CudaMemsetFn,
    device_synchronize: CudaDeviceSynchronizeFn,
    set_device: CudaSetDeviceFn,
    get_device: CudaGetDeviceFn,
    mem_get_info: CudaMemGetInfoFn,
    malloc: CudaMallocFn,
    free: CudaFreeFn,
}

// SAFETY: the resolved `cudart` entry points are plain C functions that are
// safe to invoke from any thread; the `Library` handle is only stored to keep
// the module mapped and is never mutated after construction.
unsafe impl Send for CudaRt {}
unsafe impl Sync for CudaRt {}

static CUDART: OnceLock<std::result::Result<CudaRt, String>> = OnceLock::new();

fn load() -> std::result::Result<CudaRt, String> {
    let mut last_err = String::from("no candidate library names were tried");
    // Canonical `cudart` candidate names for this host, shared with the CUDA
    // EP's own loader via `onnx-genai-cuda-version-guard` so the two can never
    // drift (see issue #1180). Previously each crate kept its own list; when
    // they disagreed the failure surfaced as the *last* candidate tried (e.g. a
    // Linux `.so` name failing on Windows), reading like "this machine has no
    // CUDA" rather than "this list is stale".
    for name in onnx_genai_cuda_version_guard::cudart_candidates() {
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
        let memcpy_async = unsafe { lib.get::<CudaMemcpyAsyncFn>(b"cudaMemcpyAsync\0") };
        let memset = unsafe { lib.get::<CudaMemsetFn>(b"cudaMemset\0") };
        let device_synchronize =
            unsafe { lib.get::<CudaDeviceSynchronizeFn>(b"cudaDeviceSynchronize\0") };
        let set_device = unsafe { lib.get::<CudaSetDeviceFn>(b"cudaSetDevice\0") };
        let get_device = unsafe { lib.get::<CudaGetDeviceFn>(b"cudaGetDevice\0") };
        let mem_get_info = unsafe { lib.get::<CudaMemGetInfoFn>(b"cudaMemGetInfo\0") };
        let malloc = unsafe { lib.get::<CudaMallocFn>(b"cudaMalloc\0") };
        let free = unsafe { lib.get::<CudaFreeFn>(b"cudaFree\0") };
        match (
            memcpy,
            memcpy_async,
            memset,
            device_synchronize,
            set_device,
            get_device,
            mem_get_info,
            malloc,
            free,
        ) {
            (
                Ok(memcpy),
                Ok(memcpy_async),
                Ok(memset),
                Ok(device_synchronize),
                Ok(set_device),
                Ok(get_device),
                Ok(mem_get_info),
                Ok(malloc),
                Ok(free),
            ) => {
                // Copy the function pointers out before `lib` is moved into the
                // struct; the borrows on `lib` end here.
                let memcpy = *memcpy;
                let memcpy_async = *memcpy_async;
                let memset = *memset;
                let device_synchronize = *device_synchronize;
                let set_device = *set_device;
                let get_device = *get_device;
                let mem_get_info = *mem_get_info;
                let malloc = *malloc;
                let free = *free;
                return Ok(CudaRt {
                    _lib: lib,
                    memcpy,
                    memcpy_async,
                    memset,
                    device_synchronize,
                    set_device,
                    get_device,
                    mem_get_info,
                    malloc,
                    free,
                });
            }
            _ => {
                last_err = format!(
                    "{name}: missing cudaMemcpy/cudaMemcpyAsync/cudaMemset/cudaDeviceSynchronize/cudaSetDevice/cudaGetDevice/cudaMemGetInfo/cudaMalloc/cudaFree symbol"
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

/// A device allocation this process owns, freed when the guard drops.
///
/// Every field is a plain integer, so `Send + Sync` come from the compiler
/// rather than from an assertion — which is what
/// [`crate::Value::from_external_memory_with_owner`]'s `Box<dyn Any + Send +
/// Sync>` requires, and what a hand-written `unsafe impl` would only serve to
/// keep true by fiat if a field ever became a raw pointer.
///
/// The address is a plain `usize` so it can be handed to
/// [`crate::Value::from_external_memory_with_owner`] as a raw pointer while the
/// guard travels with the value that borrows it: the allocation outlives every
/// view derived from it, which is the property a device-resident slice depends
/// on. Dropping the guard is the *only* way the memory is released, so there is
/// no path where a live tensor points at freed device memory.
#[derive(Debug)]
pub struct CudaAllocation {
    device_ptr: usize,
    bytes: usize,
    device_id: i32,
}

/// Device allocations this process holds through [`CudaAllocation`].
///
/// An owning wrapper's whole job is that the memory goes back, and "it went
/// back" is not something a throughput number or a device-wide free-memory
/// reading can attribute: another process on the same GPU moves that number
/// under you. A process-local live count can be asserted exactly.
static LIVE_ALLOCATIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// How many [`CudaAllocation`]s this process currently holds.
pub fn live_allocations() -> usize {
    LIVE_ALLOCATIONS.load(std::sync::atomic::Ordering::Relaxed)
}

impl CudaAllocation {
    /// Allocate `bytes` of device memory on `device_id`.
    ///
    /// A zero-byte request is rejected rather than served with a null pointer:
    /// every caller here is building a tensor, and a tensor with no allocation
    /// has no address to publish.
    pub fn new(device_id: i32, bytes: usize) -> Result<Self> {
        if bytes == 0 {
            return Err(OrtError::InvalidArgument(
                "cannot allocate a zero-byte CUDA buffer; a tensor with no elements has no device \
                 address to publish"
                    .to_string(),
            ));
        }
        let _guard = DeviceGuard::set(device_id)?;
        let rt = runtime()?;
        let mut device_ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `device_ptr` is a valid out-parameter and `cudaMalloc` matches
        // the `cudart` ABI; it allocates on the device `DeviceGuard` made
        // current.
        let code = unsafe { (rt.malloc)(&mut device_ptr, bytes) };
        if code != 0 {
            return Err(OrtError::InvalidArgument(format!(
                "cudaMalloc({bytes} bytes) on CUDA device {device_id} failed with CUDA error code \
                 {code}; the device may be out of memory"
            )));
        }
        if device_ptr.is_null() {
            return Err(OrtError::InvalidArgument(format!(
                "cudaMalloc({bytes} bytes) on CUDA device {device_id} reported success but \
                 returned a null device pointer"
            )));
        }
        LIVE_ALLOCATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Self {
            device_ptr: device_ptr as usize,
            bytes,
            device_id,
        })
    }

    /// The device address of the allocation.
    pub fn device_ptr(&self) -> usize {
        self.device_ptr
    }

    /// How many bytes the allocation holds.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The CUDA device the allocation lives on.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

impl Drop for CudaAllocation {
    fn drop(&mut self) {
        LIVE_ALLOCATIONS.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // Freeing on the wrong current device is a CUDA error, so pin first; a
        // failure here is best-effort (the process is already unwinding or
        // shutting down) and would have nowhere to be reported.
        let Ok(_guard) = DeviceGuard::set(self.device_id) else {
            return;
        };
        if let Ok(rt) = runtime() {
            // SAFETY: `device_ptr` came from `cudaMalloc` on this device and has
            // not been freed; `cudaFree` matches the `cudart` ABI.
            let _ = unsafe { (rt.free)(self.device_ptr as *mut c_void) };
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

/// Enqueue a device-to-device copy on CUDA's default stream.
///
/// Callers must synchronize once after enqueueing their complete copy batch
/// and before another session consumes the destination buffers.
pub fn memcpy_device_to_device_async(dst: usize, src: usize, bytes: usize) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    let rt = runtime()?;
    // SAFETY: `src` and `dst` cover `bytes`; a null stream selects CUDA's
    // default stream; the function pointer matches the documented cudart ABI.
    let code = unsafe {
        (rt.memcpy_async)(
            dst as *mut c_void,
            src as *const c_void,
            bytes,
            CUDA_MEMCPY_DEVICE_TO_DEVICE,
            std::ptr::null_mut(),
        )
    };
    if code != 0 {
        return Err(OrtError::InvalidArgument(format!(
            "cudaMemcpyAsync (device-to-device) failed with CUDA error code {code}"
        )));
    }
    Ok(())
}
