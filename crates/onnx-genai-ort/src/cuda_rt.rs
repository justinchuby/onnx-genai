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

/// `cudaMemcpyKind::cudaMemcpyDeviceToDevice`.
const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

type CudaMemcpyFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32;
type CudaMemsetFn = unsafe extern "C" fn(*mut c_void, i32, usize) -> i32;
type CudaDeviceSynchronizeFn = unsafe extern "C" fn() -> i32;

struct CudaRt {
    // Kept alive so the resolved function pointers remain valid; never called
    // directly after construction.
    _lib: Library,
    memcpy: CudaMemcpyFn,
    memset: CudaMemsetFn,
    device_synchronize: CudaDeviceSynchronizeFn,
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
        match (memcpy, memset, device_synchronize) {
            (Ok(memcpy), Ok(memset), Ok(device_synchronize)) => {
                // Copy the function pointers out before `lib` is moved into the
                // struct; the borrows on `lib` end here.
                let memcpy = *memcpy;
                let memset = *memset;
                let device_synchronize = *device_synchronize;
                return Ok(CudaRt {
                    _lib: lib,
                    memcpy,
                    memset,
                    device_synchronize,
                });
            }
            _ => {
                last_err = format!("{name}: missing cudaMemcpy/cudaMemset/cudaDeviceSynchronize symbol");
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
pub(crate) fn device_synchronize() -> Result<()> {
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

/// Zero `bytes` of device memory at device address `dst`.
///
/// Used to define the tail of a freshly allocated (uninitialized) KV bucket so
/// positions past the valid length are deterministic zeros.
pub(crate) fn memset_zero(dst: usize, bytes: usize) -> Result<()> {
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

/// Copy `bytes` from device address `src` to device address `dst`
/// (`cudaMemcpyDeviceToDevice`).
pub(crate) fn memcpy_device_to_device(dst: usize, src: usize, bytes: usize) -> Result<()> {
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
