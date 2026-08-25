//! Safe ownership and bounded caching for dynamically loaded cuFFT plans.
//!
//! Plans are created with cuFFT auto-allocation disabled. The caller supplies
//! the execution work area from the execution provider's governed workspace,
//! so cuFFT never hides a per-dispatch `cudaMalloc`. Plan construction may
//! still initialize opaque cuFFT/JIT state owned by the vendor library.

use core::ffi::{c_int, c_void};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::sys::CUstream;
use onnx_runtime_ep_api::{EpError, Result};
use onnx_runtime_ir::DataType;

use crate::dynamic_library::{CudaLibrary, symbol};
use crate::runtime::CudaRuntime;

type CufftHandle = c_int;
type CufftResult = c_int;
type CufftType = c_int;

const CUFFT_SUCCESS: CufftResult = 0;
const CUFFT_C2C: CufftType = 0x29;
const CUFFT_FORWARD: c_int = -1;
const CUFFT_INVERSE: c_int = 1;

type CufftCreate = unsafe extern "C" fn(*mut CufftHandle) -> CufftResult;
type CufftDestroy = unsafe extern "C" fn(CufftHandle) -> CufftResult;
type CufftSetAutoAllocation = unsafe extern "C" fn(CufftHandle, c_int) -> CufftResult;
type CufftMakePlanMany = unsafe extern "C" fn(
    CufftHandle,
    c_int,
    *mut c_int,
    *mut c_int,
    c_int,
    c_int,
    *mut c_int,
    c_int,
    c_int,
    CufftType,
    c_int,
    *mut usize,
) -> CufftResult;
type CufftSetStream = unsafe extern "C" fn(CufftHandle, CUstream) -> CufftResult;
type CufftSetWorkArea = unsafe extern "C" fn(CufftHandle, *mut c_void) -> CufftResult;
type CufftExecC2C =
    unsafe extern "C" fn(CufftHandle, *mut CufftComplex, *mut CufftComplex, c_int) -> CufftResult;

#[repr(C)]
struct CufftComplex {
    real: f32,
    imag: f32,
}

struct CufftApi {
    create: CufftCreate,
    destroy: CufftDestroy,
    set_auto_allocation: CufftSetAutoAllocation,
    make_plan_many: CufftMakePlanMany,
    set_stream: CufftSetStream,
    set_work_area: CufftSetWorkArea,
    exec_c2c: CufftExecC2C,
}

impl CufftApi {
    fn load() -> std::result::Result<Self, String> {
        Ok(Self {
            create: symbol(CudaLibrary::Cufft, b"cufftCreate\0")?,
            destroy: symbol(CudaLibrary::Cufft, b"cufftDestroy\0")?,
            set_auto_allocation: symbol(CudaLibrary::Cufft, b"cufftSetAutoAllocation\0")?,
            make_plan_many: symbol(CudaLibrary::Cufft, b"cufftMakePlanMany\0")?,
            set_stream: symbol(CudaLibrary::Cufft, b"cufftSetStream\0")?,
            set_work_area: symbol(CudaLibrary::Cufft, b"cufftSetWorkArea\0")?,
            exec_c2c: symbol(CudaLibrary::Cufft, b"cufftExecC2C\0")?,
        })
    }
}

fn api() -> Result<&'static CufftApi> {
    static API: std::sync::OnceLock<std::result::Result<CufftApi, String>> =
        std::sync::OnceLock::new();
    API.get_or_init(CufftApi::load).as_ref().map_err(|message| {
        EpError::KernelFailed(format!(
            "cuda_ep DFT: cuFFT could not be loaded: {message}. Install the CUDA 13.1 runtime \
             with 'pip install nvidia-cufft==12.1.0.78 nvidia-nvjitlink==13.1.115', or place \
             cuFFT on the platform library search path; CPU DFT remains available"
        ))
    })
}

fn status_name(status: CufftResult) -> &'static str {
    match status {
        0 => "CUFFT_SUCCESS",
        1 => "CUFFT_INVALID_PLAN",
        2 => "CUFFT_ALLOC_FAILED",
        3 => "CUFFT_INVALID_TYPE",
        4 => "CUFFT_INVALID_VALUE",
        5 => "CUFFT_INTERNAL_ERROR",
        6 => "CUFFT_EXEC_FAILED",
        7 => "CUFFT_SETUP_FAILED",
        8 => "CUFFT_INVALID_SIZE",
        9 => "CUFFT_UNALIGNED_DATA",
        11 => "CUFFT_INVALID_DEVICE",
        13 => "CUFFT_NO_WORKSPACE",
        14 => "CUFFT_NOT_IMPLEMENTED",
        16 => "CUFFT_NOT_SUPPORTED",
        17 => "CUFFT_MISSING_DEPENDENCY",
        18 => "CUFFT_NVRTC_FAILURE",
        19 => "CUFFT_NVJITLINK_FAILURE",
        20 => "CUFFT_NVSHMEM_FAILURE",
        _ => "CUFFT_UNKNOWN_STATUS",
    }
}

fn check(operation: &str, status: CufftResult) -> Result<()> {
    if status == CUFFT_SUCCESS {
        Ok(())
    } else {
        Err(EpError::KernelFailed(format!(
            "cuda_ep DFT: {operation} failed with {} ({status})",
            status_name(status)
        )))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DftDirection {
    Forward,
    Inverse,
}

impl DftDirection {
    fn cufft(self) -> c_int {
        match self {
            Self::Forward => CUFFT_FORWARD,
            Self::Inverse => CUFFT_INVERSE,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DftInputKind {
    Real,
    Complex,
}

/// Every semantic/layout dimension that can distinguish reusable DFT plans.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CufftPlanKey {
    pub(crate) device: u32,
    pub(crate) dtype: DataType,
    pub(crate) input_kind: DftInputKind,
    pub(crate) rank: usize,
    pub(crate) axis: usize,
    pub(crate) length: usize,
    pub(crate) batch: usize,
    pub(crate) direction: DftDirection,
}

pub(crate) struct CufftPlan {
    handle: CufftHandle,
    work_bytes: usize,
    direction: DftDirection,
    runtime: Arc<CudaRuntime>,
}

// SAFETY: a plan may move between host threads. Every use is protected by the
// cache entry's mutex, and all execution is submitted to its bound EP stream.
unsafe impl Send for CufftPlan {}

impl CufftPlan {
    fn new(runtime: Arc<CudaRuntime>, key: &CufftPlanKey) -> Result<Self> {
        if key.dtype != DataType::Float32 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep DFT: cuFFT C2C plan supports Float32 only, got {:?}",
                key.dtype
            )));
        }
        let length = c_int::try_from(key.length).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep DFT: dft_length {} exceeds cuFFT's 32-bit PlanMany limit",
                key.length
            ))
        })?;
        let batch = c_int::try_from(key.batch).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep DFT: batch {} exceeds cuFFT's 32-bit PlanMany limit",
                key.batch
            ))
        })?;
        runtime.bind()?;
        let api = api()?;
        let _section = onnx_runtime_cuda_memory::capture_gate::synchronizing_section();
        let mut handle = 0;
        // SAFETY: `handle` is a valid writable out-parameter.
        check("cufftCreate", unsafe { (api.create)(&mut handle) })?;
        let mut guard = PlanGuard { handle, api };
        // SAFETY: `handle` is live and exclusively owned by `guard`.
        check("cufftSetAutoAllocation(false)", unsafe {
            (api.set_auto_allocation)(handle, 0)
        })?;
        let mut dimensions = [length];
        let mut work_bytes = 0usize;
        // SAFETY: the rank is one; both embed pointers address one live i32,
        // and `handle` remains exclusively owned while the plan is configured.
        check("cufftMakePlanMany(C2C)", unsafe {
            (api.make_plan_many)(
                handle,
                1,
                dimensions.as_mut_ptr(),
                dimensions.as_mut_ptr(),
                1,
                length,
                dimensions.as_mut_ptr(),
                1,
                length,
                CUFFT_C2C,
                batch,
                &mut work_bytes,
            )
        })?;
        // SAFETY: the runtime owns a live stream in the same bound context.
        check("cufftSetStream", unsafe {
            (api.set_stream)(handle, runtime.stream_ptr())
        })?;
        guard.handle = 0;
        PLAN_CREATIONS.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            handle,
            work_bytes,
            direction: key.direction,
            runtime,
        })
    }

    pub(crate) fn work_bytes(&self) -> usize {
        self.work_bytes
    }

    /// Execute an in-place packed C2C transform using caller-owned workspace.
    ///
    /// # Safety
    /// `data` covers `batch * length` complex f32 values from this plan's key,
    /// and `work_area` covers at least [`Self::work_bytes`] bytes when non-zero.
    pub(crate) unsafe fn execute(
        &mut self,
        data: *mut c_void,
        work_area: *mut c_void,
    ) -> Result<()> {
        self.runtime.bind()?;
        let api = api()?;
        // SAFETY: the caller supplies the governed work area sized from this
        // plan, and `self.handle` is live and mutex-exclusive.
        check("cufftSetWorkArea", unsafe {
            (api.set_work_area)(self.handle, work_area)
        })?;
        let complex = data.cast::<CufftComplex>();
        // SAFETY: caller upholds the packed-buffer extent; in-place C2C is a
        // supported cuFFT execution mode.
        check("cufftExecC2C", unsafe {
            (api.exec_c2c)(self.handle, complex, complex, self.direction.cufft())
        })
    }
}

impl Drop for CufftPlan {
    fn drop(&mut self) {
        if self.handle == 0 {
            return;
        }
        let _section = onnx_runtime_cuda_memory::capture_gate::synchronizing_section();
        let _ = self.runtime.bind();
        // An evicted plan may have just enqueued work. Keep destruction
        // explicit and conservative: retire it only after its bound stream no
        // longer references the handle or its caller-owned work area.
        let _ = self.runtime.stream().synchronize();
        if let Ok(api) = api() {
            // SAFETY: this is the single destruction of the owned plan.
            let _ = unsafe { (api.destroy)(self.handle) };
        }
        self.handle = 0;
    }
}

struct PlanGuard<'a> {
    handle: CufftHandle,
    api: &'a CufftApi,
}

impl Drop for PlanGuard<'_> {
    fn drop(&mut self) {
        if self.handle != 0 {
            // SAFETY: the guard owns the not-yet-published handle.
            let _ = unsafe { (self.api.destroy)(self.handle) };
        }
    }
}

const PLAN_CACHE_CAPACITY: usize = 16;

struct CacheEntry {
    plan: Arc<Mutex<CufftPlan>>,
    last_used: u64,
}

#[derive(Default)]
struct CacheState {
    entries: HashMap<CufftPlanKey, CacheEntry>,
    clock: u64,
}

/// Per-registry bounded LRU cache. Plans are shared only within one CUDA
/// runtime and each plan serializes host-side mutation/execution.
#[derive(Default)]
pub(crate) struct CufftPlanCache {
    state: Mutex<CacheState>,
}

impl CufftPlanCache {
    pub(crate) fn get_or_create(
        &self,
        runtime: Arc<CudaRuntime>,
        key: CufftPlanKey,
    ) -> Result<Arc<Mutex<CufftPlan>>> {
        let mut state = self.state.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep DFT: cuFFT plan cache lock was poisoned".into())
        })?;
        state.clock = state.clock.wrapping_add(1);
        let now = state.clock;
        if let Some(entry) = state.entries.get_mut(&key) {
            entry.last_used = now;
            PLAN_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            return Ok(entry.plan.clone());
        }

        let plan = Arc::new(Mutex::new(CufftPlan::new(runtime, &key)?));
        if state.entries.len() >= PLAN_CACHE_CAPACITY
            && let Some(oldest) = state
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
        {
            state.entries.remove(&oldest);
            PLAN_EVICTIONS.fetch_add(1, Ordering::Relaxed);
        }
        state.entries.insert(
            key,
            CacheEntry {
                plan: plan.clone(),
                last_used: now,
            },
        );
        Ok(plan)
    }
}

static PLAN_CREATIONS: AtomicU64 = AtomicU64::new(0);
static PLAN_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static PLAN_EVICTIONS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CufftPlanCacheStats {
    pub creations: u64,
    pub hits: u64,
    pub evictions: u64,
}

pub fn cufft_plan_cache_stats() -> CufftPlanCacheStats {
    CufftPlanCacheStats {
        creations: PLAN_CREATIONS.load(Ordering::Relaxed),
        hits: PLAN_CACHE_HITS.load(Ordering::Relaxed),
        evictions: PLAN_EVICTIONS.load(Ordering::Relaxed),
    }
}
