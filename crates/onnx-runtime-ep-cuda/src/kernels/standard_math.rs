use std::{ffi::c_void, sync::Arc};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::standard_math::{Scan, validate};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{DataType, Node, standard_math::Math};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const SOURCE: &str = r#"
#if __has_include(<cuda_fp16.h>) && __has_include(<cuda_bf16.h>)
#define NXRT_HAS_HALF 1
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

template<typename S, typename F> __device__ S store(F v) { return v; }
template<typename F, typename S> __device__ F load(S v) { return v; }
#ifdef NXRT_HAS_HALF
template<> __device__ __half store<__half, float>(float v) { return __float2half_rn(v); }
template<> __device__ __nv_bfloat16 store<__nv_bfloat16, float>(float v) { return __float2bfloat16_rn(v); }
template<> __device__ float load<float, __half>(__half v) { return __half2float(v); }
template<> __device__ float load<float, __nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
#endif

template<typename S, typename F> __device__ F round_storage(F v) { return load<F>(store<S>(v)); }
using U = unsigned long long;
enum Op { EXPM1, LOG1P, SWIGLU, GEGLU, GEGLU_TANH, SCAN };

template<typename S, typename F, int op>
__device__ void math(const S* x, const S* b, S* y, U lanes, U length, U inner, float alpha) {
    for (U i = (U)blockIdx.x * blockDim.x + threadIdx.x; i < lanes; i += (U)gridDim.x * blockDim.x) {
        if (op == SCAN) {
            const U start = (i / inner) * length * inner + i % inner;
            const F inf = F(__int_as_float(0x7f800000));
            F max_value = -inf, rest = F(0);
            for (U k = 0; k < length; ++k) {
                const U j = start + k * inner;
                const F v = load<F>(x[j]);
                if (isnan(v)) max_value = v;
                else if (!isnan(max_value)) {
                    if (v == inf) max_value = v;
                    else if (isfinite(v) && max_value != inf) {
                        if (v > max_value) {
                            rest = (F(1) + rest) * exp(max_value - v);
                            max_value = v;
                        } else {
                            rest += exp(v - max_value);
                        }
                    }
                }
                y[j] = store<S>(isfinite(max_value) ? max_value + log1p(rest) : max_value);
            }
        } else {
            const F a = load<F>(x[i]);
            F out;
            if (op == EXPM1) out = expm1(a);
            else if (op == LOG1P) out = log1p(a);
            else if (op == SWIGLU) {
                const F z = round_storage<S>(round_storage<S>(F(alpha)) * a);
                const F e = exp(-fabs(z));
                const F sigmoid = round_storage<S>(z >= F(0) ? F(1) / (F(1) + e) : e / (F(1) + e));
                out = round_storage<S>(a * sigmoid) * load<F>(b[i]);
            } else {
                const F phi = op == GEGLU_TANH
                    ? tanh(F(0.7978845608028654) * (a + F(0.044715) * a * a * a))
                    : erf(a / F(1.4142135623730951));
                out = ((F(0.5) * a) * (F(1) + phi)) * load<F>(b[i]);
            }
            y[i] = store<S>(out);
        }
    }
}

#define ENTRY(S, F, SUFFIX, NAME, OP) \
extern "C" __global__ void NAME##_##SUFFIX(const S* x, const S* b, S* y, U lanes, U length, U inner, float alpha) { \
    math<S, F, OP>(x, b, y, lanes, length, inner, alpha); \
}
#define DTYPE(S, F, SUFFIX) \
    ENTRY(S, F, SUFFIX, expm1, EXPM1) \
    ENTRY(S, F, SUFFIX, log1p, LOG1P) \
    ENTRY(S, F, SUFFIX, swiglu, SWIGLU) \
    ENTRY(S, F, SUFFIX, geglu, GEGLU) \
    ENTRY(S, F, SUFFIX, geglu_tanh, GEGLU_TANH) \
    ENTRY(S, F, SUFFIX, scan, SCAN)
DTYPE(float, float, f32)
DTYPE(double, double, f64)
#ifdef NXRT_HAS_HALF
DTYPE(__half, float, f16)
DTYPE(__nv_bfloat16, float, bf16)
#endif
"#;

pub(super) struct Factory(pub Arc<CudaRuntime>);

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(MathKernel {
            op: Math::from_node(node).map_err(EpError::KernelFailed)?,
            runtime: Arc::clone(&self.0),
        }))
    }
}

struct MathKernel {
    op: Math,
    runtime: Arc<CudaRuntime>,
}

impl Kernel for MathKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let scan = validate(self.op, inputs, outputs)?;
        let op = self.op.name();
        let x = &inputs[0];
        let Scan {
            lanes,
            length,
            inner,
        } = scan.unwrap_or(Scan {
            lanes: x.numel(),
            length: 0,
            inner: 0,
        });
        if lanes == 0 {
            return Ok(());
        }
        if !inputs.iter().all(TensorView::is_contiguous) || !outputs[0].is_contiguous() {
            return Err(EpError::KernelFailed(format!(
                "{op}: CUDA requires contiguous inputs and output; materialize strided tensors first"
            )));
        }
        let suffix = match x.dtype {
            DataType::Float16 | DataType::BFloat16 => {
                self.runtime.require_nvrtc_half_headers(op)?;
                if x.dtype == DataType::Float16 {
                    "f16"
                } else {
                    "bf16"
                }
            }
            DataType::Float32 => "f32",
            DataType::Float64 => "f64",
            _ => unreachable!("validated math dtype"),
        };
        let (stem, alpha) = match self.op {
            Math::Expm1 => ("expm1", 0.),
            Math::Log1p => ("log1p", 0.),
            Math::SwiGlu { alpha } => ("swiglu", alpha),
            Math::GeGlu { tanh: false } => ("geglu", 0.),
            Math::GeGlu { tanh: true } => ("geglu_tanh", 0.),
            Math::LogCumSumExp { .. } => ("scan", 0.),
        };
        let entry = format!("{stem}_{suffix}");
        let func = self
            .runtime
            .nvrtc_function("standard_math", SOURCE, &entry)?;
        let x = cuptr(x.data_ptr::<u8>() as *const c_void);
        let b = inputs
            .get(1)
            .map_or(x, |b| cuptr(b.data_ptr::<u8>() as *const c_void));
        let y = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        const BLOCK: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: (lanes.div_ceil(BLOCK as usize).min(65_535) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (lanes, length, inner) = (lanes as u64, length as u64, inner as u64);
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&x)
            .arg(&b)
            .arg(&y)
            .arg(&lanes)
            .arg(&length)
            .arg(&inner)
            .arg(&alpha);
        // SAFETY: the pointers cover the validated tensor shapes and the arguments match ENTRY.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err(&format!("launch {op}"), e))?;
        if !self.runtime.is_capturing()? {
            self.runtime.synchronize()?;
        }
        Ok(())
    }

    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::Supported
    }
}
