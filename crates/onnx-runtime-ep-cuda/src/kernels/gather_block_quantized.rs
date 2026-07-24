//! `com.microsoft::GatherBlockQuantized`: gather rows of a blockwise-quantized
//! table and dequantize them in one capture-safe NVRTC kernel.
//!
//! This mirrors the ONNX Runtime CUDA contrib kernel numerics exactly for the
//! `uint8` data path (bits in {2, 4, 8}): each output element resolves its source
//! element in the flattened data, unpacks the quantized code (`get_val`), unpacks
//! the optional per-block zero point, and writes `(code - zero_point) * scale`.
//! The dequantized (`scales`/output) dtype may be fp32 or fp16, and the gather
//! indices may be int32 or int64.
//!
//! ONNX Runtime restricts the `uint8` layout to `gather_axis == 0` and
//! `quantize_axis == last dim`; this kernel enforces the same and otherwise
//! declines so placement falls back to another execution provider. The kernel is
//! register-only with no host synchronization or per-call allocation on the
//! capture path, so it is legal inside the persistent decode CUDA graph (the
//! `/model/embed_tokens` embedding lookup runs once per decode step).

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;

/// Bit latched into the shared capture-error word when a device-side gather
/// index is out of range during captured replay (checked by the host before the
/// produced token is consumed).
pub const GATHER_BLOCK_QUANTIZED_CAPTURE_ERROR_INDEX: u32 = 2_048;

const SOURCE: &str = r#"
#include <cuda_fp16.h>

// Unpack one unsigned quantized code from packed uint8 storage. `elems_per_byte`
// is 8/bits, so bits=8 reads one whole byte, bits=4 two nibbles, bits=2 four
// crumbs. Unsigned only (uint8 data), matching ONNX Runtime's `sign=false` path.
__device__ __forceinline__ int gbq_get_val(
    const unsigned char* data, long long idx, int bits, int elems_per_byte, unsigned int mask) {
  const long long byte_idx = idx / elems_per_byte;
  const int bit_offset = (int)(idx % elems_per_byte) * bits;
  return (int)((data[byte_idx] >> bit_offset) & mask);
}

extern "C" __global__ void gather_block_quantized(
    const unsigned char* __restrict__ data,
    const void* __restrict__ indices,
    const void* __restrict__ scales,
    const unsigned char* __restrict__ zero_points,
    void* __restrict__ output,
    long long after_gather_dim,
    long long gather_axis_dim,
    long long ind_dim,
    int bits,
    long long block_size,
    long long n,
    int index_is_i64,
    int scales_fp16,
    unsigned int* capture_error) {
  const int elems_per_byte = 8 / bits;
  const unsigned int mask = (1u << bits) - 1u;
  for (long long out_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       out_idx < n; out_idx += (long long)gridDim.x * blockDim.x) {
    const long long gather_stride = after_gather_dim * ind_dim;
    const long long idx_before = out_idx / gather_stride;
    const long long idx_after = out_idx % after_gather_dim;
    const long long idx = (out_idx % gather_stride) / after_gather_dim;
    long long idx_at_g = index_is_i64
        ? ((const long long*)indices)[idx]
        : (long long)((const int*)indices)[idx];
    if (idx_at_g < 0) idx_at_g += gather_axis_dim;
    if (idx_at_g < 0 || idx_at_g >= gather_axis_dim) {
      if (capture_error) atomicOr(capture_error, 2048u);
      continue;
    }
    const long long in_idx =
        idx_before * gather_axis_dim * after_gather_dim + idx_at_g * after_gather_dim + idx_after;
    const long long block_id = in_idx / block_size;

    const int weight = gbq_get_val(data, in_idx, bits, elems_per_byte, mask);
    int offset = 0;
    if (zero_points) {
      offset = gbq_get_val(zero_points, block_id, bits, elems_per_byte, mask);
    }
    const int diff = weight - offset;

    if (scales_fp16) {
      const __half scale = reinterpret_cast<const __half*>(scales)[block_id];
      reinterpret_cast<__half*>(output)[out_idx] = __hmul(__float2half((float)diff), scale);
    } else {
      const float scale = reinterpret_cast<const float*>(scales)[block_id];
      reinterpret_cast<float*>(output)[out_idx] = (float)diff * scale;
    }
  }
}
"#;

pub struct GatherBlockQuantizedFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for GatherBlockQuantizedFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let bits = node
            .attr("bits")
            .and_then(|a| a.as_int())
            .ok_or_else(|| not_implemented("GatherBlockQuantized requires a 'bits' attribute"))?;
        if !matches!(bits, 2 | 4 | 8) {
            return Err(not_implemented(
                "GatherBlockQuantized CUDA supports uint8 data with bits in {2, 4, 8}",
            ));
        }
        let block_size = node
            .attr("block_size")
            .and_then(|a| a.as_int())
            .unwrap_or(128);
        if block_size < 16 || (block_size & (block_size - 1)) != 0 {
            return Err(not_implemented(
                "GatherBlockQuantized 'block_size' must be a power of two and at least 16",
            ));
        }
        Ok(Box::new(GatherBlockQuantizedKernel {
            runtime: self.runtime.clone(),
            gather_axis: node
                .attr("gather_axis")
                .and_then(|a| a.as_int())
                .unwrap_or(0),
            quantize_axis: node.attr("quantize_axis").and_then(|a| a.as_int()),
            bits,
            block_size,
            last_call_capture_safe: AtomicBool::new(false),
        }))
    }
}

#[derive(Debug)]
pub struct GatherBlockQuantizedKernel {
    runtime: Arc<CudaRuntime>,
    gather_axis: i64,
    quantize_axis: Option<i64>,
    bits: i64,
    block_size: i64,
    last_call_capture_safe: AtomicBool,
}

fn checked_product(dims: &[usize], what: &str) -> Result<usize> {
    dims.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "cuda_ep GatherBlockQuantized: {what} overflows usize"
            ))
        })
    })
}

impl Kernel for GatherBlockQuantizedKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.last_call_capture_safe.store(false, Ordering::Relaxed);
        if !(inputs.len() == 3 || inputs.len() == 4) || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GatherBlockQuantized: expected 3 or 4 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let data = &inputs[0];
        let indices = &inputs[1];
        let scales = &inputs[2];
        let zero_points = inputs.get(3).filter(|z| !z.is_absent());
        let output = &mut outputs[0];

        if data.dtype != DataType::Uint8 {
            return Err(not_implemented(
                "GatherBlockQuantized CUDA supports uint8 packed data only",
            ));
        }
        if let Some(zp) = zero_points
            && zp.dtype != DataType::Uint8
        {
            return Err(not_implemented(
                "GatherBlockQuantized CUDA requires uint8 zero_points",
            ));
        }
        if !matches!(indices.dtype, DataType::Int32 | DataType::Int64) {
            return Err(not_implemented(
                "GatherBlockQuantized indices must be Int32 or Int64",
            ));
        }
        let scales_fp16 = match scales.dtype {
            DataType::Float32 => false,
            DataType::Float16 => true,
            other => {
                return Err(not_implemented(format!(
                    "GatherBlockQuantized scales dtype {other:?} is unsupported (expected Float32 or Float16)"
                )));
            }
        };
        if output.dtype != scales.dtype {
            return Err(EpError::KernelFailed(
                "cuda_ep GatherBlockQuantized: output dtype must match scales dtype".into(),
            ));
        }
        for (name, contiguous) in [
            ("data", data.is_contiguous()),
            ("indices", indices.is_contiguous()),
            ("scales", scales.is_contiguous()),
            (
                "zero_points",
                zero_points.is_none_or(TensorView::is_contiguous),
            ),
            ("output", output.is_contiguous()),
        ] {
            if !contiguous {
                return Err(not_implemented(format!(
                    "GatherBlockQuantized requires contiguous {name}"
                )));
            }
        }

        let data_rank = data.shape.len();
        if data_rank == 0 {
            return Err(EpError::KernelFailed(
                "cuda_ep GatherBlockQuantized: data must have rank >= 1".into(),
            ));
        }
        let gather_axis = normalize_axis(self.gather_axis, data_rank, "gather_axis")?;
        let quantize_axis = match self.quantize_axis {
            Some(value) => normalize_axis(value, data_rank, "quantize_axis")?,
            None => data_rank - 1,
        };
        // ONNX Runtime's uint8 layout constraints; anything else declines so the
        // node falls back to another execution provider rather than miscomputing.
        if gather_axis != 0 {
            return Err(not_implemented(
                "GatherBlockQuantized CUDA requires gather_axis == 0 for uint8 data",
            ));
        }
        if quantize_axis != data_rank - 1 {
            return Err(not_implemented(
                "GatherBlockQuantized CUDA requires quantize_axis to be the last dimension",
            ));
        }

        let components = (8 / self.bits) as usize;
        let gather_axis_dim = data.shape[gather_axis];
        let after_gather_packed =
            checked_product(&data.shape[gather_axis + 1..], "after-gather size")?;
        let after_gather_dim = after_gather_packed.checked_mul(components).ok_or_else(|| {
            EpError::KernelFailed(
                "cuda_ep GatherBlockQuantized: unpacked after-gather size overflows usize".into(),
            )
        })?;
        let ind_dim = indices.numel();
        let block_size = self.block_size as usize;
        if after_gather_dim % block_size != 0 {
            return Err(not_implemented(
                "GatherBlockQuantized CUDA requires the quantize dimension to be a whole multiple of block_size",
            ));
        }

        // Output shape: data dims before gather_axis, then the indices dims, then
        // data dims after gather_axis with the packed last dim expanded by
        // `components` (int4/int2-in-uint8 store fewer bytes than elements).
        let mut expected_shape: Vec<usize> = data.shape[..gather_axis].to_vec();
        expected_shape.extend_from_slice(indices.shape);
        expected_shape.extend_from_slice(&data.shape[gather_axis + 1..]);
        if components > 1 {
            let last = expected_shape.len() - 1;
            expected_shape[last] =
                expected_shape[last]
                    .checked_mul(components)
                    .ok_or_else(|| {
                        EpError::KernelFailed(
                            "cuda_ep GatherBlockQuantized: unpacked output dim overflows usize"
                                .into(),
                        )
                    })?;
        }
        if output.shape != expected_shape.as_slice() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GatherBlockQuantized: output shape {:?}, expected {:?}",
                output.shape, expected_shape
            )));
        }

        let n = output.numel();
        let total_unpacked = gather_axis_dim
            .checked_mul(after_gather_dim)
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep GatherBlockQuantized: unpacked data size overflows usize".into(),
                )
            })?;
        let block_count = total_unpacked / block_size;
        if scales.numel() != block_count {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GatherBlockQuantized: scales has {} elements, expected {block_count}",
                scales.numel()
            )));
        }
        if let Some(zp) = zero_points {
            let expected_zp = block_count.div_ceil(components);
            if zp.numel() != expected_zp {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GatherBlockQuantized: zero_points has {} elements, expected {expected_zp}",
                    zp.numel()
                )));
            }
        }
        if n == 0 {
            self.last_call_capture_safe.store(true, Ordering::Relaxed);
            return Ok(());
        }
        if gather_axis_dim == 0 {
            return Err(EpError::KernelFailed(
                "cuda_ep GatherBlockQuantized: indices cannot select from an empty gather axis"
                    .into(),
            ));
        }

        let capturing = self.runtime.is_capturing()?;
        if !capturing {
            self.validate_indices_host(indices, gather_axis_dim)?;
        }

        self.runtime
            .require_nvrtc_half_headers("GatherBlockQuantized")?;
        let func = self.runtime.nvrtc_function(
            "gather_block_quantized",
            SOURCE,
            "gather_block_quantized",
        )?;

        let after_gather_dim_i64 = after_gather_dim as i64;
        let gather_axis_dim_i64 = gather_axis_dim as i64;
        let ind_dim_i64 = ind_dim as i64;
        let bits = self.bits as i32;
        let block_size_i64 = self.block_size;
        let n_i64 = n as i64;
        let index_is_i64 = i32::from(indices.dtype == DataType::Int64);
        let scales_fp16_flag = i32::from(scales_fp16);
        let capture_error = if capturing {
            self.runtime.capture_error_ptr()
        } else {
            0
        };

        let data_ptr = cuptr(data.data_ptr::<u8>() as *const c_void);
        let indices_ptr = cuptr(indices.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|zp| cuptr(zp.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);

        let grid = u32::try_from(n.div_ceil(BLOCK as usize))
            .unwrap_or(u32::MAX)
            .clamp(1, 65_535);
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&data_ptr)
            .arg(&indices_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&output_ptr)
            .arg(&after_gather_dim_i64)
            .arg(&gather_axis_dim_i64)
            .arg(&ind_dim_i64)
            .arg(&bits)
            .arg(&block_size_i64)
            .arg(&n_i64)
            .arg(&index_is_i64)
            .arg(&scales_fp16_flag)
            .arg(&capture_error);
        // SAFETY: argument types and order match `gather_block_quantized`; all
        // pointers refer to live contiguous device allocations validated above.
        // The kernel uses only registers (no shared memory, no per-call
        // allocation, no host synchronization), so it is legal to record into and
        // replay from a CUDA graph; out-of-range indices latch the shared
        // capture-error word instead of performing an out-of-bounds store.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err("launch gather_block_quantized", e))?;
        self.last_call_capture_safe.store(true, Ordering::Relaxed);
        if capturing {
            Ok(())
        } else {
            self.runtime.synchronize()
        }
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> CaptureSupport {
        if self.last_call_capture_safe.load(Ordering::Relaxed) {
            CaptureSupport::Supported
        } else {
            CaptureSupport::unsupported(
                "requires a warmed fixed-shape GatherBlockQuantized path with device-side bounds validation",
            )
        }
    }
}

impl GatherBlockQuantizedKernel {
    /// Eager-path index validation: copy the indices to the host and enforce the
    /// same bounds ONNX Runtime does, reporting a synchronous error. The captured
    /// path skips this (no host sync) and relies on the device error latch.
    fn validate_indices_host(&self, indices: &TensorView, gather_axis_dim: usize) -> Result<()> {
        let count = indices.numel();
        let index_bytes = indices.dtype.storage_bytes(count);
        if index_bytes == 0 {
            return Ok(());
        }
        let mut host_indices = vec![0u8; index_bytes];
        // SAFETY: `indices` is a live contiguous device tensor and the host buffer
        // is exactly its fixed-width storage size.
        unsafe {
            self.runtime.dtoh(
                &mut host_indices,
                cuptr(indices.data_ptr::<u8>() as *const c_void),
            )?;
        }
        for raw in host_indices.chunks_exact(indices.dtype.byte_size()) {
            let value = match indices.dtype {
                DataType::Int32 => i32::from_ne_bytes(raw.try_into().unwrap()) as i64,
                DataType::Int64 => i64::from_ne_bytes(raw.try_into().unwrap()),
                _ => unreachable!("validated above"),
            };
            let normalized = if value < 0 {
                value + gather_axis_dim as i64
            } else {
                value
            };
            if normalized < 0 || normalized >= gather_axis_dim as i64 {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GatherBlockQuantized: index {value} out of range for gather axis dimension {gather_axis_dim}"
                )));
            }
        }
        Ok(())
    }
}

fn normalize_axis(axis: i64, rank: usize, what: &str) -> Result<usize> {
    let normalized = if axis < 0 { axis + rank as i64 } else { axis };
    if !(0..rank as i64).contains(&normalized) {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep GatherBlockQuantized: {what} {axis} out of range for rank {rank}"
        )));
    }
    Ok(normalized as usize)
}

#[cfg(test)]
mod tests {
    use cudarc::driver::sys::CUdeviceptr;
    use half::f16;
    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
    use onnx_runtime_ir::DeviceId;

    use super::*;

    fn runtime() -> Option<Arc<CudaRuntime>> {
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let runtime = std::panic::catch_unwind(|| CudaRuntime::new(0).ok().map(Arc::new))
            .ok()
            .flatten();
        std::panic::set_hook(previous_hook);
        runtime
    }

    fn as_bytes<T: Copy>(values: &[T]) -> &[u8] {
        // SAFETY: reinterpreting a POD slice as raw bytes for a host->device copy.
        unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        }
    }

    fn as_bytes_mut<T: Copy>(values: &mut [T]) -> &mut [u8] {
        // SAFETY: reinterpreting a POD slice as raw bytes for a device->host copy.
        unsafe {
            std::slice::from_raw_parts_mut(
                values.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(values),
            )
        }
    }

    fn device_ptr(raw: CUdeviceptr) -> DevicePtr {
        DevicePtr(raw as usize as *const c_void)
    }

    fn device_ptr_mut(raw: CUdeviceptr) -> DevicePtrMut {
        DevicePtrMut(raw as usize as *mut c_void)
    }

    /// Dequant-and-gather parity against a host oracle that reproduces ONNX
    /// Runtime's `(code - zero_point) * scale` numerics exactly. Covers the
    /// Foundry Qwen3-0.6B embedding case (uint8 data, bits=8, block_size=128,
    /// fp32 output, explicit uint8 zero points) plus a packed int4 / fp16-output
    /// case to prove the layout math generalizes across bit widths and dtypes.
    fn run_gather_parity(bits: i64, block_size: i64, scales_fp16: bool, with_zp: bool) -> f32 {
        let Some(runtime) = runtime() else {
            eprintln!("skipping GatherBlockQuantized parity test: CUDA runtime unavailable");
            return 0.0;
        };
        if runtime
            .require_nvrtc_half_headers("gather_block_quantized")
            .is_err()
        {
            eprintln!("skipping GatherBlockQuantized parity test: fp16 NVRTC headers unavailable");
            return 0.0;
        }

        let components = (8 / bits) as usize;
        let vocab = 12usize;
        let hidden = 256usize; // unpacked element count along the quantize axis
        let packed_last = hidden / components; // stored byte count along the last dim
        let blocks_per_row = hidden / block_size as usize;
        let block_count = vocab * blocks_per_row;

        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };

        // Packed quantized codes, `components` codes per byte (low code first).
        let mask = ((1u32 << bits) - 1) as u8;
        let mut codes = vec![0u8; vocab * hidden];
        for c in codes.iter_mut() {
            *c = (next() as u8) & mask;
        }
        let mut data = vec![0u8; vocab * packed_last];
        for row in 0..vocab {
            for byte in 0..packed_last {
                let mut packed = 0u8;
                for comp in 0..components {
                    let code = codes[row * hidden + byte * components + comp] & mask;
                    packed |= code << (comp * bits as usize);
                }
                data[row * packed_last + byte] = packed;
            }
        }

        // Zero points share the packed layout (one code per block, `components`
        // per byte). Default offset 0 when absent.
        let mut zp_codes = vec![0u8; block_count];
        let zp_bytes = block_count.div_ceil(components);
        let mut zero_points = vec![0u8; zp_bytes];
        if with_zp {
            for (i, z) in zp_codes.iter_mut().enumerate() {
                *z = (next() as u8) & mask;
                let byte = &mut zero_points[i / components];
                *byte |= (*z) << ((i % components) * bits as usize);
            }
        }
        let zp_ref = |block: usize| -> i32 { if with_zp { zp_codes[block] as i32 } else { 0 } };

        let mut scale_f32 = vec![0.0f32; block_count];
        let mut scale_f16 = vec![f16::ZERO; block_count];
        for i in 0..block_count {
            let raw = 0.01 + 0.02 * ((next() % 1000) as f32 / 1000.0);
            if scales_fp16 {
                let h = f16::from_f32(raw);
                scale_f16[i] = h;
                scale_f32[i] = h.to_f32();
            } else {
                scale_f32[i] = raw;
            }
        }

        let seq = 5usize;
        let indices: Vec<i64> = (0..seq)
            .map(|_| ((next() as usize) % vocab) as i64)
            .collect();

        // Host oracle over the unpacked codes.
        let mut expected = vec![0.0f32; seq * hidden];
        for (r, &token) in indices.iter().enumerate() {
            let token = token as usize;
            for col in 0..hidden {
                let block = (token * hidden + col) / block_size as usize;
                let diff = codes[token * hidden + col] as i32 - zp_ref(block);
                expected[r * hidden + col] = if scales_fp16 {
                    // Reproduce __hmul(__float2half(diff), scale_half): a single
                    // fp16-rounded product (diff is exact in fp16).
                    f16::from_f32(diff as f32 * scale_f16[block].to_f32()).to_f32()
                } else {
                    diff as f32 * scale_f32[block]
                };
            }
        }

        let out_elems = seq * hidden;
        let scales_bytes = block_count * if scales_fp16 { 2 } else { 4 };
        let out_bytes = out_elems * if scales_fp16 { 2 } else { 4 };

        let data_dev = runtime.alloc_raw(data.len()).unwrap();
        let indices_dev = runtime.alloc_raw(indices.len() * 8).unwrap();
        let scales_dev = runtime.alloc_raw(scales_bytes).unwrap();
        let zp_dev = runtime.alloc_raw(zero_points.len().max(1)).unwrap();
        let output_dev = runtime.alloc_raw(out_bytes).unwrap();

        // SAFETY: device buffers sized to hold each source slice.
        unsafe {
            runtime.htod(&data, data_dev).unwrap();
            runtime.htod(as_bytes(&indices), indices_dev).unwrap();
            if scales_fp16 {
                runtime.htod(as_bytes(&scale_f16), scales_dev).unwrap();
            } else {
                runtime.htod(as_bytes(&scale_f32), scales_dev).unwrap();
            }
            if with_zp {
                runtime.htod(&zero_points, zp_dev).unwrap();
            }
        }

        let scales_dtype = if scales_fp16 {
            DataType::Float16
        } else {
            DataType::Float32
        };
        let device = DeviceId::cuda(0);
        let data_shape = [vocab, packed_last];
        let data_strides = [packed_last as i64, 1];
        let idx_shape = [seq];
        let idx_strides = [1i64];
        let scales_shape = [vocab, blocks_per_row];
        let scales_strides = [blocks_per_row as i64, 1];
        let zp_shape = [vocab, blocks_per_row.div_ceil(components)];
        let zp_strides = [zp_shape[1] as i64, 1];
        let out_shape = [seq, hidden];
        let out_strides = [hidden as i64, 1];

        let mut inputs = vec![
            TensorView::new(
                device_ptr(data_dev),
                DataType::Uint8,
                &data_shape,
                &data_strides,
                device,
            ),
            TensorView::new(
                device_ptr(indices_dev),
                DataType::Int64,
                &idx_shape,
                &idx_strides,
                device,
            ),
            TensorView::new(
                device_ptr(scales_dev),
                scales_dtype,
                &scales_shape,
                &scales_strides,
                device,
            ),
        ];
        if with_zp {
            inputs.push(TensorView::new(
                device_ptr(zp_dev),
                DataType::Uint8,
                &zp_shape,
                &zp_strides,
                device,
            ));
        }

        let mut outputs = [TensorMut::new(
            device_ptr_mut(output_dev),
            scales_dtype,
            &out_shape,
            &out_strides,
            device,
        )];

        let kernel = GatherBlockQuantizedKernel {
            runtime: runtime.clone(),
            gather_axis: 0,
            quantize_axis: Some(1),
            bits,
            block_size,
            last_call_capture_safe: AtomicBool::new(false),
        };
        kernel.execute(&inputs, &mut outputs).unwrap();
        runtime.synchronize().unwrap();
        assert!(
            kernel.last_call_capture_safe.load(Ordering::Relaxed),
            "GatherBlockQuantized must report capture-safe after a valid launch"
        );

        let mut worst = 0.0f32;
        if scales_fp16 {
            let mut got = vec![f16::ZERO; out_elems];
            // SAFETY: output holds `out_elems` fp16 values.
            unsafe {
                runtime.dtoh(as_bytes_mut(&mut got), output_dev).unwrap();
            }
            for (g, e) in got.iter().zip(expected.iter()) {
                worst = worst.max((g.to_f32() - e).abs());
            }
        } else {
            let mut got = vec![0.0f32; out_elems];
            // SAFETY: output holds `out_elems` fp32 values.
            unsafe {
                runtime.dtoh(as_bytes_mut(&mut got), output_dev).unwrap();
            }
            for (g, e) in got.iter().zip(expected.iter()) {
                worst = worst.max((g - e).abs());
            }
        }

        // SAFETY: each pointer came from this runtime's `alloc_raw`, freed once.
        unsafe {
            runtime.free_raw(data_dev).unwrap();
            runtime.free_raw(indices_dev).unwrap();
            runtime.free_raw(scales_dev).unwrap();
            runtime.free_raw(zp_dev).unwrap();
            runtime.free_raw(output_dev).unwrap();
        }
        worst
    }

    #[test]
    fn gather_block_quantized_matches_ort_reference() {
        // (bits, block_size, scales_fp16, with_zp)
        for (bits, block_size, scales_fp16, with_zp) in [
            (8, 128, false, true),  // Foundry Qwen3-0.6B embedding (fp32 out, uint8 zp)
            (8, 128, false, false), // symmetric (offset 0) int8
            (8, 64, true, true),    // int8, narrower block, fp16 output
            (4, 128, true, true),   // packed int4, fp16 output, asymmetric zp
            (4, 32, false, false),  // packed int4, fp32 output, symmetric
        ] {
            let worst = run_gather_parity(bits, block_size, scales_fp16, with_zp);
            eprintln!(
                "GatherBlockQuantized parity bits={bits} block_size={block_size} \
                 scales_fp16={scales_fp16} with_zp={with_zp}: worst_abs={worst:.3e}"
            );
            assert!(
                worst < 1e-3,
                "GatherBlockQuantized diverged from ORT reference (bits={bits} \
                 block_size={block_size} scales_fp16={scales_fp16} with_zp={with_zp}): \
                 worst_abs={worst:.3e}"
            );
        }
    }
}
