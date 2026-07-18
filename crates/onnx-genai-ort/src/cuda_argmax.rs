//! On-device greedy argmax for captured CUDA decode.
//!
//! In the captured greedy decode loop the model's `logits [1, 1, vocab]` output
//! is the only tensor we need to reduce to a single winning token id. When the
//! logits buffer is CPU-allocated, ORT copies the entire vocabulary (e.g.
//! 151,936 f16 = ~300 KiB) host-side every token inside `RunWithBinding`, and we
//! then argmax on the host. onnxruntime-genai avoids this by keeping logits on
//! the GPU and reducing them with a custom CUDA kernel; this module does the
//! same.
//!
//! The logits buffer is allocated on the session's CUDA device allocator (see
//! [`crate::decode`]). After each captured replay we launch a single-block
//! argmax kernel over the final vocabulary row directly on that device pointer
//! and copy back only the 4-byte token id. Both the host-side full-vocab copy
//! and the host-side argmax disappear.
//!
//! ## Context sharing
//!
//! [`cudarc::driver::CudaContext::new`] *retains the primary context* of the
//! device (`cuDevicePrimaryCtxRetain`), which is the very context ORT's built-in
//! CUDA EP drives. Device pointers are therefore valid across both, so the
//! kernel can read ORT's logits allocation directly with no cross-context copy
//! (unlike `OrtApi::CopyTensors`, which has no data-transfer path for the
//! built-in CUDA EP).
//!
//! ## Correctness
//!
//! The kernel matches the host reference argmax used elsewhere in the crate:
//! maximum value, **NaNs ignored**, **lowest index wins ties**, and index `0`
//! for an all-NaN (or empty) row. f16/bf16 are decoded to f32 with pure integer
//! bit math so the NVRTC source needs no `<cuda_fp16.h>` (keeping it
//! self-contained and header-free).

use std::sync::Mutex;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::error::{OrtError, Result};
use crate::value::DataType;

/// Threads per block. One block reduces the whole row via a grid-stride loop
/// then a shared-memory tree reduction, so this is also the shared-array width.
const BLOCK: u32 = 1024;

/// Whether to issue a device-wide `cuCtxSynchronize` before reading the logits.
/// ORT's built-in CUDA EP synchronizes its compute stream at the end of each
/// `Run`, so by the time we regain control the logits are already visible and
/// this wait returns immediately (measured: no per-token cost). It is kept on by
/// default as a correctness guard against any ORT configuration that leaves the
/// stream running asynchronously; set `ONNX_GENAI_ARGMAX_CTX_SYNC=0` to drop it.
fn ctx_sync_enabled() -> bool {
    std::env::var("ONNX_GENAI_ARGMAX_CTX_SYNC")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(true)
}

/// NVRTC source: three argmax entry points (f16 / bf16 / f32). Each launches
/// **one block per row** (`blockIdx.x` selects the row) and reduces that row's
/// `vocab` contiguous elements to the index of its maximum, ignoring NaN and
/// breaking ties toward the lowest index, writing index 0 when nothing is valid.
/// Handling `rows > 1` lets speculative decoding argmax every verified position
/// (`logits [1, N, vocab]`) in a single launch, not just the final token.
const ARGMAX_SRC: &str = r#"
#define BLOCK 1024

// binary16 -> f32 via pure integer bit math (no <cuda_fp16.h> dependency).
__device__ __forceinline__ float h2f(unsigned short h) {
    unsigned int s = ((unsigned int)(h & 0x8000u)) << 16;
    unsigned int e = (h >> 10) & 0x1Fu;
    unsigned int m = h & 0x3FFu;
    unsigned int out;
    if (e == 0u) {
        if (m == 0u) {
            out = s;
        } else {
            e = 1u;
            while ((m & 0x400u) == 0u) { m <<= 1; e--; }
            m &= 0x3FFu;
            out = s | ((e + 112u) << 23) | (m << 13);
        }
    } else if (e == 0x1Fu) {
        out = s | 0x7F800000u | (m << 13);
    } else {
        out = s | ((e + 112u) << 23) | (m << 13);
    }
    return __int_as_float((int)out);
}

// bfloat16 -> f32: high 16 bits of the f32.
__device__ __forceinline__ float bf2f(unsigned short h) {
    return __int_as_float((int)(((unsigned int)h) << 16));
}

__device__ __forceinline__ void block_reduce_argmax(float best, int bidx, int* out, int row) {
    __shared__ float sval[BLOCK];
    __shared__ int   sidx[BLOCK];
    int tid = threadIdx.x;
    sval[tid] = best;
    sidx[tid] = bidx;
    __syncthreads();
    for (int off = BLOCK >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            float ov = sval[tid + off];
            int   oi = sidx[tid + off];
            float cv = sval[tid];
            int   ci = sidx[tid];
            if (ov > cv || (ov == cv && oi < ci)) {
                sval[tid] = ov;
                sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[row] = (sidx[0] == 0x7fffffff) ? 0 : sidx[0];
    }
}

extern "C" __global__ void argmax_f16(const unsigned short* x, int rows, int vocab, int* out) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned short* r = x + (size_t)row * (size_t)vocab;
    const float NEG_INF = __int_as_float(0xff800000);
    float best = NEG_INF;
    int   bidx = 0x7fffffff;
    for (int i = threadIdx.x; i < vocab; i += BLOCK) {
        float v = h2f(r[i]);
        if (v == v && (v > best || (v == best && i < bidx))) { best = v; bidx = i; }
    }
    block_reduce_argmax(best, bidx, out, row);
}

extern "C" __global__ void argmax_bf16(const unsigned short* x, int rows, int vocab, int* out) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned short* r = x + (size_t)row * (size_t)vocab;
    const float NEG_INF = __int_as_float(0xff800000);
    float best = NEG_INF;
    int   bidx = 0x7fffffff;
    for (int i = threadIdx.x; i < vocab; i += BLOCK) {
        float v = bf2f(r[i]);
        if (v == v && (v > best || (v == best && i < bidx))) { best = v; bidx = i; }
    }
    block_reduce_argmax(best, bidx, out, row);
}

extern "C" __global__ void argmax_f32(const float* x, int rows, int vocab, int* out) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* r = x + (size_t)row * (size_t)vocab;
    const float NEG_INF = __int_as_float(0xff800000);
    float best = NEG_INF;
    int   bidx = 0x7fffffff;
    for (int i = threadIdx.x; i < vocab; i += BLOCK) {
        float v = r[i];
        if (v == v && (v > best || (v == best && i < bidx))) { best = v; bidx = i; }
    }
    block_reduce_argmax(best, bidx, out, row);
}
"#;

/// A compiled, ready-to-launch on-device argmax bound to device 0's primary
/// context. Cheap to hold; NVRTC compilation happens once at construction.
pub(crate) struct CudaArgmax {
    ctx: std::sync::Arc<CudaContext>,
    stream: std::sync::Arc<CudaStream>,
    f_f16: CudaFunction,
    f_bf16: CudaFunction,
    f_f32: CudaFunction,
    /// Reused device scratch holding one `i32` winning index per row. Guarded by
    /// `lock`; grown on demand for wider speculative-verification launches.
    out: Mutex<OutScratch>,
}

/// Growable device scratch for the per-row argmax indices.
struct OutScratch {
    ptr: CUdeviceptr,
    /// Capacity in `i32` slots.
    cap: usize,
}

// SAFETY: every device operation binds the primary context first and the shared
// `out` scratch is guarded by its `Mutex`, so the handle is safe to move/share
// across threads.
unsafe impl Send for CudaArgmax {}
unsafe impl Sync for CudaArgmax {}

impl CudaArgmax {
    /// Initialise the primary context on device `ordinal`, compile the argmax
    /// module, and allocate the initial result scratch.
    pub(crate) fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)
            .map_err(|e| OrtError::Cuda(format!("CudaContext::new({ordinal}): {e:?}")))?;
        let stream = ctx.default_stream();

        // Target the device's own compute capability so the PTX matches.
        let arch = compute_arch(&ctx)?;
        let opts = cudarc::nvrtc::CompileOptions {
            options: vec![format!("--gpu-architecture={arch}")],
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(ARGMAX_SRC, opts)
            .map_err(|e| OrtError::Cuda(format!("NVRTC compile argmax: {e:?}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| OrtError::Cuda(format!("load argmax module: {e:?}")))?;
        let f_f16 = module
            .load_function("argmax_f16")
            .map_err(|e| OrtError::Cuda(format!("load argmax_f16: {e:?}")))?;
        let f_bf16 = module
            .load_function("argmax_bf16")
            .map_err(|e| OrtError::Cuda(format!("load argmax_bf16: {e:?}")))?;
        let f_f32 = module
            .load_function("argmax_f32")
            .map_err(|e| OrtError::Cuda(format!("load argmax_f32: {e:?}")))?;

        // SAFETY: primary context is current after `CudaContext::new`; we own
        // this allocation and free it in `Drop`.
        let ptr = unsafe { cudarc::driver::result::malloc_sync(4) }
            .map_err(|e| OrtError::Cuda(format!("alloc argmax scratch: {e:?}")))?;

        Ok(Self {
            ctx,
            stream,
            f_f16,
            f_bf16,
            f_f32,
            out: Mutex::new(OutScratch { ptr, cap: 1 }),
        })
    }

    /// Argmax over the final (single) `vocab`-element device row at `ptr_addr`.
    /// Convenience wrapper over [`CudaArgmax::argmax_rows`] for the common
    /// single-token greedy decode path.
    pub(crate) fn argmax(&self, dtype: DataType, ptr_addr: usize, vocab: usize) -> Result<u32> {
        Ok(self.argmax_rows(dtype, ptr_addr, 1, vocab)?[0])
    }

    /// Argmax each of `rows` contiguous `vocab`-element rows in the device buffer
    /// at `ptr_addr` (a device pointer, e.g. from
    /// [`crate::value::Value::data_ptr_addr`]), returning one token id per row.
    /// Synchronizes the context first so all of ORT's just-issued decode work
    /// (which wrote these logits) is visible to the kernel.
    pub(crate) fn argmax_rows(
        &self,
        dtype: DataType,
        ptr_addr: usize,
        rows: usize,
        vocab: usize,
    ) -> Result<Vec<u32>> {
        if rows == 0 {
            return Ok(Vec::new());
        }
        let mut out = self.out.lock().expect("cuda argmax scratch poisoned");
        self.ctx
            .bind_to_thread()
            .map_err(|e| OrtError::Cuda(format!("bind context: {e:?}")))?;
        // ORT's CUDA EP synchronizes its compute stream at the end of each
        // `Run`, so by the time control returns here the logits are fully
        // written and visible. This device-wide wait is therefore normally a
        // no-op guard (see `ctx_sync_enabled`); it can be disabled via
        // `ONNX_GENAI_ARGMAX_CTX_SYNC=0`.
        if ctx_sync_enabled() {
            cudarc::driver::result::ctx::synchronize()
                .map_err(|e| OrtError::Cuda(format!("ctx synchronize: {e:?}")))?;
        }
        out.ensure(rows)?;

        let func = match dtype {
            DataType::Float16 => &self.f_f16,
            DataType::BFloat16 => &self.f_bf16,
            DataType::Float32 => &self.f_f32,
            other => {
                return Err(OrtError::Cuda(format!(
                    "device argmax unsupported logits dtype {other:?}"
                )));
            }
        };
        let x_ptr = ptr_addr as CUdeviceptr;
        let rows_i = i32::try_from(rows)
            .map_err(|_| OrtError::Cuda(format!("row count {rows} exceeds i32")))?;
        let vocab_i = i32::try_from(vocab)
            .map_err(|_| OrtError::Cuda(format!("vocab {vocab} exceeds i32")))?;
        let cfg = LaunchConfig {
            grid_dim: (rows_i as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = self.stream.launch_builder(func);
        builder
            .arg(&x_ptr)
            .arg(&rows_i)
            .arg(&vocab_i)
            .arg(&out.ptr);
        // SAFETY: `func` is the compiled argmax entry; the argument list matches
        // its (const T*, int, int, int*) signature; `x_ptr` is a live device
        // buffer of `rows * vocab` elements and `out.ptr` holds `rows` i32 slots.
        unsafe { builder.launch(cfg) }
            .map_err(|e| OrtError::Cuda(format!("launch argmax: {e:?}")))?;

        let mut idx = vec![0i32; rows];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(idx.as_mut_ptr().cast::<u8>(), rows * 4)
        };
        // SAFETY: `out.ptr` holds `rows` live i32 slots; `bytes` covers them.
        unsafe { cudarc::driver::result::memcpy_dtoh_sync(bytes, out.ptr) }
            .map_err(|e| OrtError::Cuda(format!("copy argmax result: {e:?}")))?;
        self.stream
            .synchronize()
            .map_err(|e| OrtError::Cuda(format!("stream synchronize: {e:?}")))?;
        Ok(idx.into_iter().map(|v| v as u32).collect())
    }

    /// Copy a `len`-element device row of `dtype` at `ptr_addr` into `dst`
    /// (host), for the non-greedy path that still needs the full vocabulary.
    /// Synchronizes the context first so ORT's writes are visible.
    pub(crate) fn copy_row_to_host(
        &self,
        dtype: DataType,
        ptr_addr: usize,
        len: usize,
        dst: &mut [u8],
    ) -> Result<()> {
        let _guard = self.out.lock().expect("cuda argmax scratch poisoned");
        self.ctx
            .bind_to_thread()
            .map_err(|e| OrtError::Cuda(format!("bind context: {e:?}")))?;
        if ctx_sync_enabled() {
            cudarc::driver::result::ctx::synchronize()
                .map_err(|e| OrtError::Cuda(format!("ctx synchronize: {e:?}")))?;
        }
        let want = len
            .checked_mul(dtype_size(dtype)?)
            .ok_or_else(|| OrtError::Cuda("logits byte size overflow".into()))?;
        if dst.len() != want {
            return Err(OrtError::Cuda(format!(
                "device logits copy expected {want} bytes, got {}",
                dst.len()
            )));
        }
        // SAFETY: `ptr_addr` is a live device row of `want` bytes; `dst` matches.
        unsafe { cudarc::driver::result::memcpy_dtoh_sync(dst, ptr_addr as CUdeviceptr) }
            .map_err(|e| OrtError::Cuda(format!("copy logits to host: {e:?}")))?;
        self.stream
            .synchronize()
            .map_err(|e| OrtError::Cuda(format!("stream synchronize: {e:?}")))
    }
}

impl OutScratch {
    /// Ensure the scratch can hold `rows` i32 indices, reallocating if needed.
    fn ensure(&mut self, rows: usize) -> Result<()> {
        if rows <= self.cap {
            return Ok(());
        }
        // SAFETY: `ptr` came from `malloc_sync`; free before replacing.
        let _ = unsafe { cudarc::driver::result::free_sync(self.ptr) };
        let bytes = rows
            .checked_mul(4)
            .ok_or_else(|| OrtError::Cuda("argmax scratch size overflow".into()))?;
        // SAFETY: primary context is current (caller bound it); we own the result.
        self.ptr = unsafe { cudarc::driver::result::malloc_sync(bytes) }
            .map_err(|e| OrtError::Cuda(format!("grow argmax scratch: {e:?}")))?;
        self.cap = rows;
        Ok(())
    }
}

impl Drop for CudaArgmax {
    fn drop(&mut self) {
        // Best-effort free of the scratch; ignore errors during teardown.
        if self.ctx.bind_to_thread().is_ok() {
            if let Ok(out) = self.out.lock() {
                // SAFETY: `out.ptr` came from `malloc_sync` and is freed once here.
                let _ = unsafe { cudarc::driver::result::free_sync(out.ptr) };
            }
        }
    }
}

/// `compute_XY` string for the device's CUDA compute capability.
fn compute_arch(ctx: &CudaContext) -> Result<String> {
    use cudarc::driver::sys::CUdevice_attribute;
    let major = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|e| OrtError::Cuda(format!("query CC major: {e:?}")))?;
    let minor = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(|e| OrtError::Cuda(format!("query CC minor: {e:?}")))?;
    Ok(format!("compute_{major}{minor}"))
}

fn dtype_size(dtype: DataType) -> Result<usize> {
    Ok(match dtype {
        DataType::Float16 | DataType::BFloat16 => 2,
        DataType::Float32 => 4,
        other => {
            return Err(OrtError::Cuda(format!(
                "device logits unsupported dtype {other:?}"
            )));
        }
    })
}

