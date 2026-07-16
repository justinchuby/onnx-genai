//! Shared CUDA runtime state: the driver context, its default stream, and vendor
//! library backends. One [`CudaRuntime`] is created per
//! [`CudaExecutionProvider`] and shared (via `Arc`) into every kernel the
//! provider hands out, so the whole EP drives a single device + stream.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::sys::{CUdevice_attribute, CUdeviceptr};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};

use onnx_runtime_ep_api::EpError;
use onnx_runtime_ep_api::Result;

use crate::blas::CublasLt;
use crate::cudnn::CudnnBackend;
use crate::error::{driver_err, nvrtc_err};

fn nvrtc_include_paths() -> Vec<String> {
    let mut candidates = Vec::<PathBuf>::new();
    for variable in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = std::env::var_os(variable) {
            candidates.push(PathBuf::from(root).join("include"));
        }
    }
    candidates.push(PathBuf::from("/usr/local/cuda/include"));

    if let Some(paths) = std::env::var_os("LD_LIBRARY_PATH") {
        for path in std::env::split_paths(&paths) {
            if path.ends_with(Path::new("nvidia/cuda_nvrtc/lib"))
                && let Some(nvidia) = path.parent().and_then(Path::parent)
            {
                candidates.push(nvidia.join("cuda_runtime/include"));
            }
        }
    }

    candidates.sort();
    candidates.dedup();
    candidates
        .into_iter()
        .filter(|path| path.join("cuda_fp16.h").is_file())
        .map(|path| path.to_string_lossy().into_owned())
        .collect()
}

fn ptx_arch_for(major: u32, minor: u32) -> String {
    format!("compute_{major}{minor}")
}

fn cubin_arch_for(major: u32, minor: u32) -> String {
    format!("sm_{major}{minor}")
}

/// Device context, stream, and vendor-library backends shared across the EP.
pub struct CudaRuntime {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CublasLt,
    cudnn: CudnnBackend,
    ordinal: u32,
    /// Compute capability reported by the CUDA device, cached for NVRTC targets.
    compute_capability: (u32, u32),
    ptx_arch: String,
    cubin_arch: String,
    /// Cache of NVRTC-compiled modules, keyed by a stable module name, so each
    /// runtime compiles a given kernel (e.g. the fused attention softmax) at
    /// most once and reuses the loaded module for every kernel invocation.
    modules: Mutex<HashMap<&'static str, Arc<CudaModule>>>,
    /// Set after a driver rejects the toolkit's PTX ISA. Subsequent modules are
    /// compiled directly to the device's native SM CUBIN instead of repeating
    /// the failed load.
    nvrtc_cubin_fallback: AtomicBool,
}

impl std::fmt::Debug for CudaRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaRuntime")
            .field("ordinal", &self.ordinal)
            .field("compute_capability", &self.compute_capability)
            .finish()
    }
}

impl CudaRuntime {
    /// Initialise the primary context on CUDA device `ordinal`, its default
    /// stream, and a cuBLASLt handle. Returns an error (never panics) when no
    /// such device exists or the CUDA driver / cuBLASLt cannot be loaded.
    pub fn new(ordinal: u32) -> Result<Self> {
        let context =
            CudaContext::new(ordinal as usize).map_err(|e| driver_err("CudaContext::new", e))?;
        let major = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| driver_err("querying CUDA compute capability major", e))?;
        let minor = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| driver_err("querying CUDA compute capability minor", e))?;
        let major = u32::try_from(major).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep: CUDA device {ordinal} reported invalid compute capability major {major}"
            ))
        })?;
        let minor = u32::try_from(minor).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep: CUDA device {ordinal} reported invalid compute capability minor {minor}"
            ))
        })?;
        if major == 0 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: CUDA device {ordinal} reported invalid compute capability {major}.{minor}"
            )));
        }
        let compute_capability = (major, minor);
        let ptx_arch = ptx_arch_for(major, minor);
        let cubin_arch = cubin_arch_for(major, minor);
        let stream = context.default_stream();
        let blas = CublasLt::new()?;
        let cudnn = CudnnBackend::new(stream.clone());
        Ok(Self {
            context,
            stream,
            blas,
            cudnn,
            ordinal,
            compute_capability,
            ptx_arch,
            cubin_arch,
            modules: Mutex::new(HashMap::new()),
            nvrtc_cubin_fallback: AtomicBool::new(false),
        })
    }

    /// The CUDA device ordinal this runtime drives.
    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// The cuBLASLt handle.
    pub fn blas(&self) -> &CublasLt {
        &self.blas
    }

    /// The lazily initialized cuDNN backend bound to this runtime's stream.
    pub fn cudnn(&self) -> &CudnnBackend {
        &self.cudnn
    }

    /// The raw CUDA stream the EP submits work on.
    pub fn stream_ptr(&self) -> cudarc::driver::sys::CUstream {
        self.stream.cu_stream()
    }

    /// The EP's compute stream (for `launch_builder`-based kernel launches).
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Get a [`CudaFunction`] for entry point `entry` in the NVRTC module named
    /// `module_key`, compiling `src` to PTX and loading it on first use and
    /// reusing the cached module thereafter.
    ///
    /// The compile targets the device's detected virtual compute architecture.
    /// If the installed NVRTC emits a PTX ISA newer than the driver accepts,
    /// compilation is retried for the matching real SM architecture and the
    /// resulting CUBIN is loaded instead. An NVRTC failure surfaces the compiler
    /// log via [`nvrtc_err`] (RULES.md #1).
    pub fn nvrtc_function(
        &self,
        module_key: &'static str,
        src: &str,
        entry: &str,
    ) -> Result<CudaFunction> {
        self.bind()?;
        let module = {
            let mut cache = self.modules.lock().expect("cuda_ep module cache poisoned");
            if let Some(m) = cache.get(module_key) {
                m.clone()
            } else {
                let include_paths = nvrtc_include_paths();
                let m = if self.nvrtc_cubin_fallback.load(Ordering::Relaxed) {
                    self.load_nvrtc_cubin(module_key, src, &include_paths)?
                } else {
                    let opts = cudarc::nvrtc::CompileOptions {
                        include_paths: include_paths.clone(),
                        options: vec![format!("--gpu-architecture={}", self.ptx_arch)],
                        ..Default::default()
                    };
                    let ptx = cudarc::nvrtc::compile_ptx_with_opts(src, opts).map_err(|e| {
                        nvrtc_err(&format!("compiling NVRTC module '{module_key}'"), e)
                    })?;
                    match self.context.load_module(ptx) {
                        Ok(module) => module,
                        Err(error)
                            if error.0
                                == cudarc::driver::sys::CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION =>
                        {
                            self.nvrtc_cubin_fallback.store(true, Ordering::Relaxed);
                            self.load_nvrtc_cubin(module_key, src, &include_paths)?
                        }
                        Err(error) => {
                            return Err(driver_err(
                                &format!("loading NVRTC module '{module_key}'"),
                                error,
                            ));
                        }
                    }
                };
                cache.insert(module_key, m.clone());
                m
            }
        };
        module
            .load_function(entry)
            .map_err(|e| driver_err(&format!("loading NVRTC function '{entry}'"), e))
    }

    fn load_nvrtc_cubin(
        &self,
        module_key: &'static str,
        src: &str,
        include_paths: &[String],
    ) -> Result<Arc<CudaModule>> {
        let source = CString::new(src).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep: compiling NVRTC module '{module_key}': source contains a NUL byte"
            ))
        })?;
        let name = CString::new(module_key).expect("static module key cannot contain a NUL byte");
        let program =
            cudarc::nvrtc::result::create_program(source.as_c_str(), Some(name.as_c_str()))
                .map_err(|error| {
                    EpError::KernelFailed(format!(
                        "cuda_ep: creating NVRTC CUBIN module '{module_key}': {error:?}"
                    ))
                })?;
        let mut options = include_paths
            .iter()
            .map(|path| format!("--include-path={path}"))
            .collect::<Vec<_>>();
        options.push(format!("--gpu-architecture={}", self.cubin_arch));

        // SAFETY: `program` is live until the matching destroy call below.
        let compile_result = unsafe { cudarc::nvrtc::result::compile_program(program, &options) };
        if let Err(error) = compile_result {
            // SAFETY: compilation may fail, but the live program still owns its log.
            let log = unsafe { cudarc::nvrtc::result::get_program_log(program) }
                .ok()
                .map(|bytes| {
                    // SAFETY: NVRTC returns a NUL-terminated compiler log.
                    unsafe { CStr::from_ptr(bytes.as_ptr()) }
                        .to_string_lossy()
                        .into_owned()
                })
                .unwrap_or_else(|| "<compiler log unavailable>".into());
            // SAFETY: this is the single destroy for the live program.
            let _ = unsafe { cudarc::nvrtc::result::destroy_program(program) };
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: compiling NVRTC CUBIN module '{module_key}' failed ({error:?}); compiler log:\n{log}"
            )));
        }

        let cubin: Result<Vec<u8>> = (|| {
            let mut size = 0usize;
            // SAFETY: `program` compiled successfully and `size` is writable.
            unsafe { cudarc::nvrtc::sys::nvrtcGetCUBINSize(program, &mut size) }
                .result()
                .map_err(|error| {
                    EpError::KernelFailed(format!(
                        "cuda_ep: getting NVRTC CUBIN size for '{module_key}': {error:?}"
                    ))
                })?;
            let mut image = vec![0u8; size];
            // SAFETY: `image` has the exact size reported by NVRTC.
            unsafe { cudarc::nvrtc::sys::nvrtcGetCUBIN(program, image.as_mut_ptr().cast()) }
                .result()
                .map_err(|error| {
                    EpError::KernelFailed(format!(
                        "cuda_ep: getting NVRTC CUBIN for '{module_key}': {error:?}"
                    ))
                })?;
            Ok(image)
        })();
        // SAFETY: this is the single destroy for the live program.
        let destroy_result = unsafe { cudarc::nvrtc::result::destroy_program(program) };
        let image = cubin?;
        destroy_result.map_err(|error| {
            EpError::KernelFailed(format!(
                "cuda_ep: destroying NVRTC CUBIN program '{module_key}': {error:?}"
            ))
        })?;
        self.context
            .load_module(cudarc::nvrtc::Ptx::from_binary(image))
            .map_err(|error| {
                driver_err(
                    &format!("loading NVRTC CUBIN fallback module '{module_key}'"),
                    error,
                )
            })
    }

    pub fn require_nvrtc_half_headers(&self, op: &str) -> Result<()> {
        if nvrtc_include_paths().is_empty() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: f16/bf16 NVRTC kernels require cuda_fp16.h and cuda_bf16.h. \
                 Install the CUDA runtime headers (for pip CUDA 13: `pip install \
                 nvidia-cuda-runtime`; alternatively set CUDA_HOME/CUDA_PATH)."
            )));
        }
        Ok(())
    }

    /// Bind this runtime's context to the calling thread. Required before any
    /// driver call (`malloc`, `memcpy`, cuBLASLt) that targets the current
    /// context.
    pub fn bind(&self) -> Result<()> {
        self.context
            .bind_to_thread()
            .map_err(|e| driver_err("bind_to_thread", e))
    }

    /// Block until all submitted work on the default stream completes.
    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| driver_err("stream synchronize", e))
    }

    /// Allocate `bytes` (>= 1) of device memory, returning the raw device
    /// pointer. Binds the context first.
    pub fn alloc_raw(&self, bytes: usize) -> Result<CUdeviceptr> {
        self.bind()?;
        // SAFETY: `malloc_sync` returns a fresh device allocation on the current
        // (bound) context; we own it and free it exactly once via `free_raw`.
        unsafe { cudarc::driver::result::malloc_sync(bytes.max(1)) }
            .map_err(|e| driver_err("cuMemAlloc", e))
    }

    /// Free a device pointer previously returned by [`CudaRuntime::alloc_raw`].
    ///
    /// # Safety
    /// `ptr` must have come from this runtime's `alloc_raw` and not been freed.
    pub unsafe fn free_raw(&self, ptr: CUdeviceptr) -> Result<()> {
        self.bind()?;
        // SAFETY: caller upholds the single-free contract.
        unsafe { cudarc::driver::result::free_sync(ptr) }.map_err(|e| driver_err("cuMemFree", e))
    }

    /// Copy `bytes` host → device (H2D). `dst` must be large enough.
    ///
    /// # Safety
    /// `dst` is a live device allocation of at least `src.len()` bytes.
    pub unsafe fn htod(&self, src: &[u8], dst: CUdeviceptr) -> Result<()> {
        self.bind()?;
        // SAFETY: bound context; `dst` covers `src.len()` bytes per the contract.
        unsafe { cudarc::driver::result::memcpy_htod_sync(dst, src) }
            .map_err(|e| driver_err("cuMemcpyHtoD", e))
    }

    /// Copy `dst.len()` bytes device → host (D2H). `src` must be large enough.
    ///
    /// # Safety
    /// `src` is a live device allocation of at least `dst.len()` bytes.
    pub unsafe fn dtoh(&self, dst: &mut [u8], src: CUdeviceptr) -> Result<()> {
        self.bind()?;
        // SAFETY: bound context; `src` covers `dst.len()` bytes per the contract.
        unsafe { cudarc::driver::result::memcpy_dtoh_sync(dst, src) }
            .map_err(|e| driver_err("cuMemcpyDtoH", e))?;
        self.synchronize()
    }

    /// Copy `bytes` device → device (D2D).
    ///
    /// # Safety
    /// Both pointers are live allocations of at least `bytes` bytes.
    pub unsafe fn dtod(&self, src: CUdeviceptr, dst: CUdeviceptr, bytes: usize) -> Result<()> {
        self.bind()?;
        // SAFETY: bound context; both endpoints cover `bytes` per the contract.
        unsafe { cudarc::driver::result::memcpy_dtod_sync(dst, src, bytes) }
            .map_err(|e| driver_err("cuMemcpyDtoD", e))
    }
}

/// Reinterpret an EP [`onnx_runtime_ep_api::DeviceBuffer`] raw pointer (or a
/// [`onnx_runtime_ep_api::TensorView`] data pointer) as a CUDA device pointer.
/// CUDA device pointers are integer addresses; the EP stores them in the opaque
/// pointer slot, so this is a value reinterpretation, never a host deref.
#[inline]
pub fn cuptr(raw: *const c_void) -> CUdeviceptr {
    raw as usize as CUdeviceptr
}

/// Inverse of [`cuptr`]: pack a CUDA device pointer into the opaque pointer slot
/// used by [`onnx_runtime_ep_api::DeviceBuffer`].
#[inline]
pub fn raw_ptr(dptr: CUdeviceptr) -> *mut c_void {
    dptr as usize as *mut c_void
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_ptx_arch_from_compute_capability() {
        for (major, minor, expected) in [
            (6, 0, "compute_60"),
            (7, 5, "compute_75"),
            (8, 0, "compute_80"),
            (8, 6, "compute_86"),
            (8, 9, "compute_89"),
            (9, 0, "compute_90"),
            (10, 0, "compute_100"),
            (12, 0, "compute_120"),
        ] {
            assert_eq!(ptx_arch_for(major, minor), expected);
        }
    }

    #[test]
    fn derives_cubin_arch_from_compute_capability() {
        for (major, minor, expected) in [
            (6, 0, "sm_60"),
            (7, 5, "sm_75"),
            (8, 0, "sm_80"),
            (8, 6, "sm_86"),
            (8, 9, "sm_89"),
            (9, 0, "sm_90"),
            (10, 0, "sm_100"),
            (12, 0, "sm_120"),
        ] {
            assert_eq!(cubin_arch_for(major, minor), expected);
        }
    }

    #[test]
    fn nvrtc_include_paths_only_returns_cuda_header_dirs() {
        assert!(
            nvrtc_include_paths()
                .iter()
                .all(|path| Path::new(path).join("cuda_fp16.h").is_file())
        );
    }
}
