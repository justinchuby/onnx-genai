//! Deterministic, dtype-rounded cumulative products, one CUDA thread per lane.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, ExecutionProvider, Kernel, KernelConstantInput, KernelFactory, Result,
    TensorMut, TensorView,
};
use onnx_runtime_ir::{
    DataType, Node,
    opset26::{cum_prod_axis, cum_prod_flags},
};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SCAN: &str = r#"
template<class T, class Mul, bool exclusive, bool reverse>
__device__ __forceinline__ void scan(
    const T* input, T* output, unsigned long long lanes,
    unsigned long long width, unsigned long long inner, T one) {
    for (unsigned long long lane = blockIdx.x * blockDim.x + threadIdx.x; lane < lanes;
         lane += (unsigned long long)gridDim.x * blockDim.x) {
        const unsigned long long first = ((lane / inner) * width + (reverse ? width - 1 : 0)) * inner + lane % inner;
        const long long stride = reverse ? -(long long)inner : (long long)inner;
        T total = input[first];
        output[first] = exclusive ? one : total;
        for (unsigned long long n = 1; n < width; ++n) {
            const unsigned long long offset = first + stride * (long long)n;
            T value = input[offset];
            if (exclusive) output[offset] = total;
            total = Mul::product(total, value);
            if (!exclusive) output[offset] = total;
        }
    }
}

template<class T> struct NativeMul {
    __device__ static T product(T a, T b) { return a * b; }
};

#define ENTRY(NAME, T, MUL, ONE) \
extern "C" __global__ void NAME(const T* input, T* output, unsigned long long lanes, \
    unsigned long long width, unsigned long long inner, int flags) { \
    switch (flags) { \
    case 0: scan<T, MUL, false, false>(input, output, lanes, width, inner, ONE); break; \
    case 1: scan<T, MUL, true, false>(input, output, lanes, width, inner, ONE); break; \
    case 2: scan<T, MUL, false, true>(input, output, lanes, width, inner, ONE); break; \
    case 3: scan<T, MUL, true, true>(input, output, lanes, width, inner, ONE); break; \
    } \
}
"#;

const WIDE: &str = r#"
// Signed and unsigned integer outputs share the same modulo-width bit pattern.
ENTRY(cumprod_u32, unsigned int, NativeMul<unsigned int>, 1u)
ENTRY(cumprod_u64, unsigned long long, NativeMul<unsigned long long>, 1ull)
ENTRY(cumprod_f32, float, NativeMul<float>, 1.0f)
ENTRY(cumprod_f64, double, NativeMul<double>, 1.0)
"#;

const NARROW: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
struct HalfMul {
    __device__ static __half product(__half a, __half b) {
        return __float2half_rn(__half2float(a) * __half2float(b));
    }
};
struct BFloatMul {
    __device__ static __nv_bfloat16 product(__nv_bfloat16 a, __nv_bfloat16 b) {
        return __float2bfloat16_rn(__bfloat162float(a) * __bfloat162float(b));
    }
};
ENTRY(cumprod_f16, __half, HalfMul, __float2half_rn(1.0f))
ENTRY(cumprod_bf16, __nv_bfloat16, BFloatMul, __float2bfloat16_rn(1.0f))
"#;

pub struct CumProdFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for CumProdFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let (exclusive, reverse) = cum_prod_flags(node).map_err(EpError::KernelFailed)?;
        Ok(Box::new(CumProdKernel {
            runtime: self.runtime.clone(),
            flags: i32::from(exclusive) | (i32::from(reverse) << 1),
            fixed_axis: None,
        }))
    }
}

struct CumProdKernel {
    runtime: Arc<CudaRuntime>,
    flags: i32,
    fixed_axis: Option<i64>,
}

fn decode_axis(dtype: DataType, shape: &[usize], bytes: &[u8]) -> Result<i64> {
    match (dtype, shape, bytes) {
        (DataType::Int32, [], [a, b, c, d]) => Ok(i64::from(i32::from_le_bytes([*a, *b, *c, *d]))),
        (DataType::Int64, [], [a, b, c, d, e, f, g, h]) => {
            Ok(i64::from_le_bytes([*a, *b, *c, *d, *e, *f, *g, *h]))
        }
        _ => Err(EpError::KernelFailed(format!(
            "CumProd: axis must be an Int32 or Int64 rank-0 tensor, got {dtype:?} {shape:?} with {} bytes",
            bytes.len()
        ))),
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod tests {
    use super::*;
    use cudarc::driver::sys::CUevent_flags;
    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
    use onnx_runtime_ir::{DeviceId, compute_contiguous_strides};

    #[test]
    #[ignore = "CUDA event microbenchmark; run under the shared host lock"]
    fn cumprod_cuda_microbenchmark() {
        let ep = crate::CudaExecutionProvider::new(0).expect("CUDA device 0");
        let rt = ep.runtime();
        let context = rt.cuda_context();
        let stream = rt.stream();
        let device = DeviceId::cuda(0);
        let axis_dev = rt.alloc_raw(8).unwrap();
        let axis = TensorView::new(
            DevicePtr(axis_dev as usize as *const c_void),
            DataType::Int64,
            &[],
            &[],
            device,
        );
        // Compare with the pre-opset-audit f32 implementation. Its old
        // fixed-axis capture path is represented by the same launch geometry.
        let before = rt.nvrtc_function("cumprod_bench_baseline", r#"
                    extern "C" __global__ void baseline(const float* x, float* y,
                        unsigned long long lanes, unsigned long long width,
                        unsigned long long inner, int exclusive, int reverse) {
                      for (unsigned long long lane = blockIdx.x * blockDim.x + threadIdx.x; lane < lanes;
                           lane += (unsigned long long)gridDim.x * blockDim.x) {
                        unsigned long long outer = lane / inner, i = lane % inner;
                        float total = 1.0f;
                        for (unsigned long long n = 0; n < width; ++n) {
                          unsigned long long d = reverse ? width - 1 - n : n;
                          unsigned long long offset = (outer * width + d) * inner + i;
                          float value = x[offset];
                          if (exclusive) { y[offset] = total; total *= value; }
                          else { total *= value; y[offset] = total; }
                        }
                      }
                    }
                "#, "baseline").unwrap();
        let mut first_shape_medians = None;
        const LAUNCHES: usize = 96;
        let short = ([8usize, 4096usize], 0);
        for (case, (shape, dim)) in [short, ([4096, 128], 0), ([128, 4096], 1), short]
            .into_iter()
            .enumerate()
        {
            let kernel = CumProdKernel {
                runtime: rt.clone(),
                flags: 0,
                fixed_axis: Some(dim as i64),
            };
            unsafe { rt.htod(&(dim as i64).to_le_bytes(), axis_dev).unwrap() };
            let strides = compute_contiguous_strides(&shape);
            let count = shape[0] * shape[1];
            let x = rt.alloc_raw(count * 4).unwrap();
            let y = rt.alloc_raw(count * 4).unwrap();
            let input_bytes: Vec<_> = (0..count).flat_map(|_| 1.0001_f32.to_le_bytes()).collect();
            unsafe { rt.htod(&input_bytes, x).unwrap() };
            let input = TensorView::new(
                DevicePtr(x as usize as *const c_void),
                DataType::Float32,
                &shape,
                &strides,
                device,
            );
            let out = || {
                TensorMut::new(
                    DevicePtrMut(y as usize as *mut c_void),
                    DataType::Float32,
                    &shape,
                    &strides,
                    device,
                )
            };
            let width = shape[dim] as u64;
            let lanes = count as u64 / width;
            let inner = shape[dim + 1..].iter().product::<usize>() as u64;
            let config = LaunchConfig {
                grid_dim: (lanes.div_ceil(u64::from(BLOCK)) as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let run = |arm: usize| {
                if arm == 0 {
                    let mut builder = stream.launch_builder(&before);
                    builder
                        .arg(&x)
                        .arg(&y)
                        .arg(&lanes)
                        .arg(&width)
                        .arg(&inner)
                        .arg(&0_i32)
                        .arg(&0_i32);
                    unsafe { builder.launch(config).unwrap() };
                } else {
                    kernel.execute(&[input, axis], &mut [out()]).unwrap();
                }
            };
            let mut reference = vec![0; count * 4];
            let mut actual = vec![0; count * 4];
            run(0);
            unsafe { rt.dtoh(&mut reference, y).unwrap() };
            run(1);
            unsafe { rt.dtoh(&mut actual, y).unwrap() };
            assert_eq!(actual, reference);
            let mut samples = [Vec::new(), Vec::new()];
            for rep in 0..5 {
                for arm in [rep % 2, 1 - rep % 2] {
                    rt.begin_graph_capture(&[&kernel]).unwrap();
                    for _ in 0..LAUNCHES {
                        run(arm);
                    }
                    rt.end_graph_capture().unwrap();
                    assert_eq!(rt.graph_segment_count().unwrap(), 1);
                    rt.replay_graph().unwrap();
                    rt.synchronize().unwrap();
                    if case == 0 && rep == 0 && arm == 0 {
                        let ramp = std::time::Instant::now();
                        while ramp.elapsed().as_secs() < 8 {
                            for _ in 0..16 {
                                rt.replay_graph().unwrap();
                            }
                            rt.synchronize().unwrap();
                            // The runtime otherwise appends the next arm to this capture.
                            assert!(rt.reset_graph().unwrap());
                        }
                    }
                    let start = context
                        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                        .unwrap();
                    let end = context
                        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                        .unwrap();
                    start.record(stream).unwrap();
                    rt.replay_graph().unwrap();
                    end.record(stream).unwrap();
                    samples[arm].push(
                        f64::from(start.elapsed_ms(&end).unwrap()) * 1_000.0 / LAUNCHES as f64,
                    );
                    rt.synchronize().unwrap();
                }
            }
            for (arm, times) in ["baseline", "opset-26"].into_iter().zip(&mut samples) {
                times.sort_by(f64::total_cmp);
                eprintln!(
                    "CumProd CUDA {shape:?} axis={dim} {arm}: {:.3} us [{:.3}, {:.3}]",
                    times[2], times[0], times[4]
                );
            }
            if (shape, dim) == short {
                let medians = [samples[0][2], samples[1][2]];
                if let Some(first) = first_shape_medians {
                    eprintln!("CUDA [8, 4096] start/end drift: {first:?} -> {medians:?} us");
                } else {
                    first_shape_medians = Some(medians);
                }
            }
            unsafe {
                rt.free_raw(x).unwrap();
                rt.free_raw(y).unwrap();
            }
        }
        unsafe { rt.free_raw(axis_dev).unwrap() };
    }
}
impl Kernel for CumProdKernel {
    fn prepare_constant_inputs(
        &mut self,
        constants: &[Option<KernelConstantInput<'_>>],
        _: &dyn ExecutionProvider,
    ) -> Result<()> {
        if let Some(Some(axis)) = constants.get(1) {
            self.fixed_axis = Some(decode_axis(axis.dtype, axis.shape, axis.bytes)?);
        }
        Ok(())
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let ([input, axis_input], [output]) = (inputs, outputs) else {
            return Err(EpError::KernelFailed(
                "CumProd: expected data, scalar axis and one output".into(),
            ));
        };
        if !input.is_contiguous() || !axis_input.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented(
                "CumProd requires contiguous input and output on CUDA",
            ));
        }
        if output.dtype != input.dtype || output.shape != input.shape {
            return Err(EpError::KernelFailed(format!(
                "CumProd: expected output {:?} {:?} matching the input, got {:?} {:?}",
                input.dtype, input.shape, output.dtype, output.shape
            )));
        }
        let (entry, narrow) = match input.dtype {
            DataType::Int32 | DataType::Uint32 => ("cumprod_u32", false),
            DataType::Int64 | DataType::Uint64 => ("cumprod_u64", false),
            DataType::Float32 => ("cumprod_f32", false),
            DataType::Float64 => ("cumprod_f64", false),
            DataType::Float16 => ("cumprod_f16", true),
            DataType::BFloat16 => ("cumprod_bf16", true),
            dtype => {
                return Err(not_implemented(format!(
                    "CumProd-26 does not accept {dtype:?} data"
                )));
            }
        };
        if !axis_input.shape.is_empty()
            || !matches!(axis_input.dtype, DataType::Int32 | DataType::Int64)
        {
            return Err(EpError::KernelFailed(format!(
                "CumProd: axis must be an Int32 or Int64 rank-0 tensor, got {:?} {:?}",
                axis_input.dtype, axis_input.shape
            )));
        }
        input.validate()?;
        axis_input.validate()?;
        output.validate()?;
        let raw = if let Some(axis) = self.fixed_axis {
            axis
        } else {
            if self.runtime.is_capturing()? {
                return Err(not_implemented(
                    "CumProd must read and validate its runtime axis outside CUDA graph capture",
                ));
            }
            let mut bytes = [0_u8; 8];
            let len = axis_input.dtype.byte_size();
            unsafe {
                self.runtime
                    .dtoh(&mut bytes[..len], cuptr(axis_input.data_ptr::<c_void>()))?;
            }
            decode_axis(axis_input.dtype, axis_input.shape, &bytes[..len])?
        };
        let axis = cum_prod_axis(raw, input.shape.len()).map_err(EpError::KernelFailed)?;
        if input.shape.contains(&0) {
            return Ok(());
        }
        let inner = input.shape[axis + 1..].iter().product::<usize>() as u64;
        let width = input.shape[axis] as u64;
        let lanes = input.shape[..axis].iter().product::<usize>() as u64 * inner;
        static NARROW_SOURCE: std::sync::LazyLock<String> =
            std::sync::LazyLock::new(|| format!("{SCAN}{NARROW}"));
        static WIDE_SOURCE: std::sync::LazyLock<String> =
            std::sync::LazyLock::new(|| format!("{SCAN}{WIDE}"));
        let (module, source) = if narrow {
            ("cumprod_narrow", NARROW_SOURCE.as_str())
        } else {
            ("cumprod_wide", WIDE_SOURCE.as_str())
        };
        let func = self.runtime.nvrtc_function(module, source, entry)?;
        let input_ptr = cuptr(input.data_ptr::<c_void>());
        let output_ptr = cuptr(output.data_ptr_mut::<c_void>());
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&input_ptr)
            .arg(&output_ptr)
            .arg(&lanes)
            .arg(&width)
            .arg(&inner)
            .arg(&self.flags);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    lanes.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32,
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err("launch CumProd", e))?;
        Ok(())
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }

    fn can_run_in_place(&self, input_index: usize) -> bool {
        input_index == 0
    }

    fn capture_support(&self) -> CaptureSupport {
        if self.fixed_axis.is_some() {
            CaptureSupport::Supported
        } else {
            CaptureSupport::unsupported(
                "CumProd has a runtime axis that must be read and validated on the host",
            )
        }
    }
}
