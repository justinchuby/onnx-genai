//! `Cast` / `CastLike`: element-wise dtype conversion on the GPU via
//! runtime-compiled (NVRTC) kernels (`docs/execution/CUDA_COVERAGE.md`, "Shape /
//! data-movement" row).
//!
//! ## Backend choice — custom NVRTC (trivial pointwise), and *why*
//!
//! No vendor library owns a general ONNX `Cast`; it is a bandwidth-bound
//! element-wise conversion, so a custom NVRTC kernel is exactly right (RULES.md
//! #4 — "no library covers the op"). Numerics follow ONNX / C++ `static_cast`,
//! mirroring `crates/onnx-runtime-ep-cpu/src/kernels/cast.rs`:
//!
//! * **float → int** truncates toward zero and **saturates** to the target
//!   integer's range (NaN → 0);
//! * **int → int** is a width-narrowing/widening two's-complement reinterpret;
//! * **any numeric → bool** is `x != 0` (NaN → true);
//! * **float ↔ float** rounds to nearest representable.
//!
//! ## Two NVRTC modules — keeping f16/bf16 out of the common path
//!
//! The half-precision conversions (`Float16`, `BFloat16`) use CUDA's
//! `__half`/`__nv_bfloat16` device intrinsics, which need NVRTC's built-in
//! `cuda_fp16.h` / `cuda_bf16.h` headers. To avoid a half-only header dependency
//! breaking the common `f32 ↔ i32 ↔ i64 ↔ bool` casts, those go through a
//! **header-free core module**; a cast that touches a half type compiles the
//! separate **half module** (headers + intrinsics). If a target's CUDA lacks the
//! NVRTC fp16 headers, only half casts error (with the NVRTC log — RULES.md #1);
//! every other conversion still works.
//!
//! ## Dtype coverage & limits (actionable errors — RULES.md #1)
//!
//! Supported: `Float32`, `Float64`, `Float16`, `BFloat16`, `Int8`, `Uint8`,
//! `Int16`, `Uint16`, `Int32`, `Uint32`, `Int64`, `Uint64`, `Bool`, and FP8.
//! Packed 4-bit and 2-bit types are rejected, naming the dtype. `Int64`/`Uint64` values
//! beyond 2⁵³ lose precision when routed through the `double` conversion lane
//! (documented; the common token-id range is exact); `Uint64` above 2⁶³ is not
//! representable in the signed lane.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg, sys::CUdeviceptr};

use onnx_runtime_ep_api::float8_cast::{self, CastOptions, Float8};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

/// Threads per block for the 1-D pointwise cast grid.
const BLOCK: u32 = 256;

/// Shared device helpers (header-free): saturating float→int conversions and the
/// non-half load/store switches keyed on the ONNX dtype code. The `switch` tags
/// are the raw ONNX `DataType` discriminants (`Float32 = 1`, `Int64 = 7`, …), so
/// the Rust side passes `dtype as i32` directly.
const CAST_HELPERS: &str = r#"
__device__ __forceinline__ float decode_float6(unsigned int code, int tag) {
    const int mantissa = tag == 27 ? 3 : 2;
    const int bias = tag == 27 ? 1 : 3;
    const int exp = (code & 31) >> mantissa;
    const int fraction = code & ((1 << mantissa) - 1);
    const int significand = exp == 0 ? fraction : (1 << mantissa) | fraction;
    const unsigned int scale = 127 + (exp == 0 ? 1 : exp) - bias - mantissa;
    const float magnitude = (float)significand * __uint_as_float(scale << 23);
    return (code & 32) ? -magnitude : magnitude;
}
__device__ float load_float6(const unsigned char* x, int i, int tag) {
    const unsigned int lane = i & 3;
    const unsigned int bit = lane * 6;
    const unsigned char* p = x + (i / 4) * 3 + bit / 8;
    const unsigned int shift = bit & 7;
    unsigned int word = p[0];
    if (shift > 2) word |= (unsigned int)p[1] << 8;
    return decode_float6((word >> shift) & 63, tag);
}
__device__ long long f_to_ll_sat(double f, double lo, double hi) {
    if (isnan(f)) return 0;
    if (f < lo) return (long long)lo;
    if (f > hi) return (long long)hi;
    return (long long)f;               // truncate toward zero
}
__device__ unsigned long long f_to_ull_sat(double f) {
    if (isnan(f) || f <= 0.0) return 0ULL;
    if (f >= 18446744073709551616.0) return 0xffffffffffffffffULL; // 2^64
    return (unsigned long long)f;
}
// Load element `i` (non-half tags). `is_float` selects the exact lane: floats in
// `fv`, integers/bools in `iv`.
__device__ void load_core(const unsigned char* x, int i, int tag,
                          double* fv, long long* iv, int* is_float) {
    switch (tag) {
        case 1:  *fv = (double)((const float*)x)[i];              *is_float = 1; break; // f32
        case 11: *fv = ((const double*)x)[i];                     *is_float = 1; break; // f64
        case 3:  *iv = (long long)((const signed char*)x)[i];     *is_float = 0; break; // i8
        case 2:  *iv = (long long)((const unsigned char*)x)[i];   *is_float = 0; break; // u8
        case 5:  *iv = (long long)((const short*)x)[i];           *is_float = 0; break; // i16
        case 4:  *iv = (long long)((const unsigned short*)x)[i];  *is_float = 0; break; // u16
        case 6:  *iv = (long long)((const int*)x)[i];             *is_float = 0; break; // i32
        case 12: *iv = (long long)((const unsigned int*)x)[i];    *is_float = 0; break; // u32
        case 7:  *iv = ((const long long*)x)[i];                  *is_float = 0; break; // i64
        case 13: *iv = (long long)((const unsigned long long*)x)[i]; *is_float = 0; break; // u64
        case 9:  *iv = (((const unsigned char*)x)[i] != 0) ? 1 : 0;  *is_float = 0; break; // bool
        case 27:
        case 28: *fv = load_float6(x, i, tag); *is_float = 1; break;
        default: *fv = 0.0; *iv = 0; *is_float = 1; break;
    }
}
// Store element `i` (non-half tags), applying ONNX Cast numeric semantics.
__device__ void store_core(unsigned char* y, int i, int tag,
                           double fv, long long iv, int is_float) {
    const double dv = is_float ? fv : (double)iv;
    switch (tag) {
        case 1:  ((float*)y)[i]  = (float)dv; break;   // f32
        case 11: ((double*)y)[i] = dv;        break;   // f64
        // float source saturates to the target range; int source wraps (2's-complement).
        case 3:  ((signed char*)y)[i] =
                 (signed char)(is_float ? f_to_ll_sat(fv, -128.0, 127.0) : iv); break;      // i8
        case 2:  ((unsigned char*)y)[i] =
                 (unsigned char)(is_float ? f_to_ll_sat(fv, 0.0, 255.0) : iv); break;       // u8
        case 5:  ((short*)y)[i] =
                 (short)(is_float ? f_to_ll_sat(fv, -32768.0, 32767.0) : iv); break;        // i16
        case 4:  ((unsigned short*)y)[i] =
                 (unsigned short)(is_float ? f_to_ll_sat(fv, 0.0, 65535.0) : iv); break;    // u16
        case 6:  ((int*)y)[i] =
                 (int)(is_float ? f_to_ll_sat(fv, -2147483648.0, 2147483647.0) : iv); break;// i32
        case 12: ((unsigned int*)y)[i] =
                 (unsigned int)(is_float ? f_to_ll_sat(fv, 0.0, 4294967295.0) : iv); break; // u32
        case 7:  ((long long*)y)[i] =
                 (is_float ? (fv >= 9223372036854775808.0 ? 0x7fffffffffffffffLL
                            : f_to_ll_sat(fv, -9223372036854775808.0, 9223372036854775808.0)) : iv);
                 break;                                                                      // i64
        case 13: ((unsigned long long*)y)[i] =
                 (is_float ? f_to_ull_sat(fv)
                           : (unsigned long long)iv); break;                                // u64
        case 9:  ((unsigned char*)y)[i] =
                 ((is_float ? (fv != 0.0) : (iv != 0)) ? 1 : 0); break;                     // bool
        default: break;
    }
}
__device__ __forceinline__ bool load_cast_fp8(const unsigned char* x, int i, int fp8_src,
                               double* fv, int* is_float) {
#ifdef CAST_FLOAT8
    if (fp8_src >= 0) {
        *fv = decode_fp8(fp8_src, x[i]);
        *is_float = 1;
        return true;
    }
#endif
    return false;
}
__device__ __forceinline__ bool store_cast_fp8(unsigned char* y, int i, int fp8_dst, int src_tag,
                                double fv, long long iv, int is_float, int saturate) {
#ifdef CAST_FLOAT8
    if (fp8_dst >= 0) {
        // The E8M0 reference uses f32 for integer inputs; floating inputs use
        // their exact value. In particular, do not round binary64 to binary32.
        const double value = is_float ? fv :
            (fp8_dst >= FP8_E8_FIRST_ROW ? (double)(src_tag == 13 ? (float)(unsigned long long)iv : (float)iv)
                          : (src_tag == 13 ? (double)(unsigned long long)iv : (double)iv));
        y[i] = encode_fp8(fp8_dst, value, saturate);
        return true;
    }
#endif
    return false;
}
"#;

/// Header-free core cast entry (no half types).
const CAST_CORE_ENTRY: &str = r#"
extern "C" __global__ void cast_core(const unsigned char* x, unsigned char* y,
                                     const int n, const int src_tag, const int dst_tag,
                                     const int fp8_src, const int fp8_dst, const int saturate) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
#ifdef CAST_FLOAT8
        if (src_tag == dst_tag) { y[i] = x[i]; continue; }
#endif
        double fv = 0.0; long long iv = 0; int isf = 1;
        if (!load_cast_fp8(x, i, fp8_src, &fv, &isf)) load_core(x, i, src_tag, &fv, &iv, &isf);
        if (!store_cast_fp8(y, i, fp8_dst, src_tag, fv, iv, isf, saturate)) {
            store_core(y, i, dst_tag, fv, iv, isf);
        }
    }
}
"#;

/// Half-aware cast entry: adds `Float16` (tag 10) and `BFloat16` (tag 16) via
/// device intrinsics, delegating every other tag to the core load/store.
const CAST_HALF_ENTRY: &str = r#"
__device__ void load_any(const unsigned char* x, int i, int tag,
                         double* fv, long long* iv, int* isf) {
    if (tag == 10) { *fv = (double)__half2float(((const __half*)x)[i]); *isf = 1; return; }
    if (tag == 16) { *fv = (double)__bfloat162float(((const __nv_bfloat16*)x)[i]); *isf = 1; return; }
    load_core(x, i, tag, fv, iv, isf);
}
__device__ void store_any(unsigned char* y, int i, int tag,
                          double fv, long long iv, int isf) {
    const double dv = isf ? fv : (double)iv;
    if (tag == 10) { ((__half*)y)[i] = __float2half_rn((float)dv); return; }
    if (tag == 16) { ((__nv_bfloat16*)y)[i] = __float2bfloat16((float)dv); return; }
    store_core(y, i, tag, fv, iv, isf);
}
extern "C" __global__ void cast_half(const unsigned char* x, unsigned char* y,
                                     const int n, const int src_tag, const int dst_tag,
                                     const int fp8_src, const int fp8_dst, const int saturate) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        double fv = 0.0; long long iv = 0; int isf = 1;
        if (!load_cast_fp8(x, i, fp8_src, &fv, &isf)) load_any(x, i, src_tag, &fv, &iv, &isf);
        if (!store_cast_fp8(y, i, fp8_dst, src_tag, fv, iv, isf, saturate)) {
            store_any(y, i, dst_tag, fv, iv, isf);
        }
    }
}
"#;

const CAST_CORE_MODULE: &str = "cast_core_f32";
const CAST_HALF_MODULE: &str = "cast_half_f32";

/// The dtypes this cast kernel supports, keyed by their ONNX code.
fn is_supported(dt: DataType) -> bool {
    Float8::of(dt).is_some()
        || matches!(
            dt,
            DataType::Float32
                | DataType::Float64
                | DataType::Float16
                | DataType::BFloat16
                | DataType::Int8
                | DataType::Uint8
                | DataType::Int16
                | DataType::Uint16
                | DataType::Int32
                | DataType::Uint32
                | DataType::Int64
                | DataType::Uint64
                | DataType::Bool
        )
}

fn is_half(dt: DataType) -> bool {
    matches!(dt, DataType::Float16 | DataType::BFloat16)
}

fn cast_pair_supported(src: DataType, dst: DataType) -> bool {
    if src.is_float6() {
        matches!(
            dst,
            DataType::Float16 | DataType::BFloat16 | DataType::Float32 | DataType::Float64
        )
    } else {
        is_supported(src) && is_supported(dst)
    }
}

/// Grid dimension for `n` elements at [`BLOCK`] threads (grid-stride, capped).
fn grid_for(n: usize) -> u32 {
    const MAX_BLOCKS: usize = 65_535;
    n.div_ceil(BLOCK as usize).clamp(1, MAX_BLOCKS) as u32
}

/// Factory for [`CastKernel`], shared by `Cast` and `CastLike`. The target dtype
/// is taken from the **output** tensor (equivalently the `to` attribute for
/// `Cast`, or the "like" input's type for `CastLike`), so both ops share one
/// kernel.
pub struct CastFactory {
    pub runtime: Arc<CudaRuntime>,
    pub opset: u64,
}

impl KernelFactory for CastFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(CastKernel {
            runtime: self.runtime.clone(),
            float6: self.opset >= 28,
            options: CastOptions::from_node(node, self.opset)?,
        }))
    }
}

/// NVRTC-backed element-wise dtype conversion kernel.
#[derive(Debug)]
pub struct CastKernel {
    runtime: Arc<CudaRuntime>,
    float6: bool,
    options: CastOptions,
}

impl CastKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // `Cast` has 1 input; `CastLike` has 2 (the 2nd only supplies the target
        // dtype, already reflected in the output tensor, so it is not read).
        if !(1..=2).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Cast: expected 1-2 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let x = &inputs[0];
        let src = x.dtype;
        let dst = outputs[0].dtype;
        self.options.validate_pair(
            if inputs.len() == 1 {
                "Cast"
            } else {
                "CastLike"
            },
            src,
            dst,
        )?;
        if src.is_float6() && !self.float6 {
            return Err(not_implemented(
                "FLOAT6 widening requires standard Cast at opset 28 or newer",
            ));
        }
        if !cast_pair_supported(src, dst) {
            return Err(not_implemented(format!(
                "Cast {src:?} to {dst:?}; use f32/f64/f16/bf16/FP8/int8-64/uint8-64/bool, or widen a FLOAT6 source to f16, bf16, f32, or f64"
            )));
        }
        if !x.is_contiguous() || !outputs[0].is_contiguous() {
            return Err(not_implemented(
                "Cast with a non-contiguous (strided) input/output; \
                 insert an explicit copy to materialise it before the op",
            ));
        }
        if outputs[0].shape != x.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Cast: output shape {:?} must equal input shape {:?}",
                outputs[0].shape, x.shape
            )));
        }

        let n = x.numel();
        if n == 0 {
            return Ok(());
        }
        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        launch_cast(&self.runtime, x_ptr, src, y_ptr, dst, n, self.options)?;
        if self.runtime.is_capturing()? {
            Ok(())
        } else {
            self.runtime.synchronize()
        }
    }
}

/// Enqueue an element-wise dtype conversion between two raw device buffers of
/// `n` elements on the EP stream, reusing the shared cast kernel (and its NVRTC
/// module cache). Unlike [`CastKernel::run`] this is a low-level, stream-ordered
/// primitive with **no** trailing synchronize, so other kernels can stage a
/// tensor through a supported precision (e.g. `MatMulNBits` upcasting BFloat16
/// activations/scales to Float16) without a per-call device sync. Callers own
/// the buffer lifetimes and any required synchronization before freeing them.
///
/// # Safety
/// `src_ptr`/`dst_ptr` must be live device allocations holding at least `n`
/// elements of `src_dtype`/`dst_dtype` respectively, and both dtypes must be
/// supported by the cast kernel (see [`cast_pair_supported`]).
pub(crate) fn launch_cast_raw(
    runtime: &CudaRuntime,
    src_ptr: CUdeviceptr,
    src_dtype: DataType,
    dst_ptr: CUdeviceptr,
    dst_dtype: DataType,
    n: usize,
) -> Result<()> {
    launch_cast(
        runtime,
        src_ptr,
        src_dtype,
        dst_ptr,
        dst_dtype,
        n,
        CastOptions::defaults(28),
    )
}

fn launch_cast(
    runtime: &CudaRuntime,
    src_ptr: CUdeviceptr,
    src_dtype: DataType,
    dst_ptr: CUdeviceptr,
    dst_dtype: DataType,
    n: usize,
    options: CastOptions,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    if !cast_pair_supported(src_dtype, dst_dtype) {
        return Err(not_implemented(format!(
            "cuda_ep Cast: unsupported dtype pair {src_dtype:?} -> {dst_dtype:?}"
        )));
    }
    let n_i = i32::try_from(n)
        .map_err(|_| EpError::KernelFailed(format!("cuda_ep Cast: {n} elements exceed i32")))?;
    let src_tag = src_dtype as i32;
    let dst_tag = dst_dtype as i32;
    let fp8_src = Float8::of(src_dtype).map_or(-1, Float8::decode_row);
    let fp8_dst = Float8::of(dst_dtype).map_or(-1, |fmt| fmt.row(options.round));
    let uses_fp8 = fp8_src >= 0 || fp8_dst >= 0;
    let saturate = i32::from(options.saturate);

    let use_half = is_half(src_dtype) || is_half(dst_dtype);
    let (module, entry, includes, kernel) = if use_half {
        (
            if uses_fp8 {
                "cast_fp8_half"
            } else {
                CAST_HALF_MODULE
            },
            "cast_half",
            "#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n",
            CAST_HALF_ENTRY,
        )
    } else {
        (
            if uses_fp8 {
                "cast_fp8_core"
            } else {
                CAST_CORE_MODULE
            },
            "cast_core",
            "",
            CAST_CORE_ENTRY,
        )
    };
    let source = if uses_fp8 {
        format!(
            "{includes}#define CAST_FLOAT8\n{}{CAST_HELPERS}{kernel}",
            float8_cast::cuda_source()
        )
    } else {
        format!("{includes}{CAST_HELPERS}{kernel}")
    };

    let func = runtime.nvrtc_function(module, &source, entry)?;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(n), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = runtime.stream().launch_builder(&func);
    builder
        .arg(&src_ptr)
        .arg(&dst_ptr)
        .arg(&n_i)
        .arg(&src_tag)
        .arg(&dst_tag)
        .arg(&fp8_src)
        .arg(&fp8_dst)
        .arg(&saturate);
    // SAFETY: `func` is the compiled cast entry and the arguments match
    // its signature; the caller guarantees
    // `src_ptr`/`dst_ptr` are live device allocations of `n` elements.
    unsafe { builder.launch(cfg) }.map_err(|e| driver_err("launch cast_raw", e))?;
    Ok(())
}

impl Kernel for CastKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::Supported
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_and_half_entries_present() {
        assert!(CAST_CORE_ENTRY.contains("cast_core"));
        assert!(CAST_HALF_ENTRY.contains("cast_half"));
        assert!(CAST_HELPERS.contains("f_to_ll_sat"));
    }

    #[test]
    fn dtype_tags_match_onnx_codes() {
        // The kernel `switch` relies on the raw ONNX discriminants.
        assert_eq!(DataType::Float32 as i32, 1);
        assert_eq!(DataType::Int64 as i32, 7);
        assert_eq!(DataType::Bool as i32, 9);
        assert_eq!(DataType::Float16 as i32, 10);
        assert_eq!(DataType::BFloat16 as i32, 16);
    }

    #[test]
    fn half_routing_selects_the_half_module() {
        assert!(is_half(DataType::Float16));
        assert!(is_half(DataType::BFloat16));
        assert!(!is_half(DataType::Float32));
        // A supported non-half pair stays on the header-free core.
        assert!(!(is_half(DataType::Float32) || is_half(DataType::Int64)));
    }

    #[test]
    fn unsupported_dtypes_are_rejected() {
        assert!(!is_supported(DataType::Int4));
        assert!(!is_supported(DataType::Uint4));
        assert!(is_supported(DataType::Float32));
        assert!(is_supported(DataType::BFloat16));
    }

    #[test]
    fn grid_covers_all_elements() {
        assert_eq!(grid_for(0), 1);
        assert_eq!(grid_for(BLOCK as usize + 1), 2);
        assert_eq!(grid_for(usize::MAX / 2), 65_535);
    }
}
