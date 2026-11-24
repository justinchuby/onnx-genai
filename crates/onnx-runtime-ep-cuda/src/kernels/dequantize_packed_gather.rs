//! CUDA `DequantizePacked -> Gather(axis=0)` without materializing the table.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::dequantize_packed::Metadata;
use onnx_runtime_ir::{DataType, DeviceType, Node, dequantize_packed_gather};

use super::dequantize_packed::{decoder_kind, decoder_source};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const ENTRY: &str = "dequantize_packed_gather";
const THREADS: u32 = 256;

const GATHER_SOURCE: &str = r#"
extern "C" __global__ void dequantize_packed_gather(
    const unsigned char* packed, const void* indices, unsigned char* output,
    unsigned long long lanes, unsigned long long row_elements,
    long long rows, int index_is_i64, int width, int bytes, int kind, int dtype,
    unsigned int* capture_error)
{
    unsigned long long lane =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)blockDim.x * gridDim.x;
    for (; lane < lanes; lane += stride) {
        const unsigned long long selected = lane / row_elements;
        long long row = index_is_i64
            ? ((const long long*)indices)[selected]
            : (long long)((const int*)indices)[selected];
        if (row < 0) row += rows;
        if (row < 0 || row >= rows) {
            if (capture_error) atomicOr(capture_error, 64u);
            continue;
        }
        const unsigned long long source_lane =
            (unsigned long long)row * row_elements + lane % row_elements;
        const unsigned char* block = packed + source_lane / width * bytes;
        const float value = decode_onnx_gguf(block, kind, source_lane % width);
        switch (dtype) {
        case 1: reinterpret_cast<float*>(output)[lane] = value; break;
        case 11: reinterpret_cast<double*>(output)[lane] = value; break;
        case 10: {
            unsigned short f16;
            asm("cvt.rn.f16.f32 %0, %1;" : "=h"(f16) : "f"(value));
            reinterpret_cast<unsigned short*>(output)[lane] = f16;
            break;
        }
        case 16:
            reinterpret_cast<unsigned short*>(output)[lane] = to_bfloat16(value);
            break;
        }
    }
}
"#;

pub(super) struct Factory(pub Arc<CudaRuntime>);

struct PackedGatherKernel {
    runtime: Arc<CudaRuntime>,
    metadata: Metadata,
    last_call_capture_safe: AtomicBool,
}

fn invalid(detail: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("cuda_ep DequantizePackedGather: {}", detail.into()))
}

fn source() -> &'static str {
    static SOURCE: OnceLock<String> = OnceLock::new();
    SOURCE.get_or_init(|| {
        let mut source = decoder_source().to_owned();
        source.push_str(GATHER_SOURCE);
        source
    })
}

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(PackedGatherKernel {
            runtime: self.0.clone(),
            metadata: Metadata::from_node(node).map_err(invalid)?,
            last_call_capture_safe: AtomicBool::new(false),
        }))
    }
}

impl PackedGatherKernel {
    fn validate_indices_host(&self, indices: &TensorView, rows: usize) -> Result<()> {
        let mut bytes = vec![0; indices.dtype.storage_bytes(indices.numel())];
        if !bytes.is_empty() {
            // SAFETY: the contiguous device input and host destination have the
            // same validated fixed-width byte count.
            unsafe {
                self.runtime
                    .dtoh(&mut bytes, cuptr(indices.data_ptr::<u8>().cast::<c_void>()))?
            };
        }
        for lane in bytes.chunks_exact(indices.dtype.byte_size()) {
            let raw = match indices.dtype {
                DataType::Int32 => i64::from(i32::from_ne_bytes(lane.try_into().unwrap())),
                DataType::Int64 => i64::from_ne_bytes(lane.try_into().unwrap()),
                _ => unreachable!("validated indices dtype"),
            };
            let normalized = if raw < 0 {
                i128::from(raw) + rows as i128
            } else {
                i128::from(raw)
            };
            if normalized < 0 || normalized >= rows as i128 {
                return Err(invalid(format!(
                    "index {raw} is out of range for decoded axis 0 dimension {rows}"
                )));
            }
        }
        Ok(())
    }
}

impl Kernel for PackedGatherKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.last_call_capture_safe.store(false, Ordering::Relaxed);
        let ([packed, indices], [output]) = (inputs, outputs) else {
            return Err(invalid(
                "expected packed data and indices inputs and one output",
            ));
        };
        let layout = dequantize_packed_gather::layout(
            &self.metadata,
            packed.dtype,
            packed.shape,
            indices.dtype,
            indices.shape,
            output.dtype,
            output.shape,
        )
        .map_err(invalid)?;
        if packed.device.device_type != DeviceType::Cuda
            || packed.device != indices.device
            || packed.device != output.device
        {
            return Err(invalid(
                "packed data, indices, and output must be on the same CUDA device",
            ));
        }
        if !packed.is_contiguous() || !indices.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented(
                "DequantizePackedGather requires contiguous inputs and output",
            ));
        }
        packed.validate()?;
        indices.validate()?;
        output.validate()?;
        if layout.rows == 0 && layout.indices != 0 {
            return Err(invalid(
                "indices cannot select from an empty decoded axis 0",
            ));
        }
        let capturing = self.runtime.is_capturing()?;
        if !capturing && !self.runtime.eager_sync_deferred() {
            self.validate_indices_host(indices, layout.rows)?;
        }
        if layout.lanes == 0 {
            self.last_call_capture_safe.store(true, Ordering::Relaxed);
            return Ok(());
        }

        let (width, bytes) = self.metadata.format().block();
        let width = i32::try_from(width).map_err(|_| invalid("block width exceeds i32"))?;
        let bytes = i32::try_from(bytes).map_err(|_| invalid("block byte count exceeds i32"))?;
        let rows = i64::try_from(layout.rows).map_err(|_| invalid("row count exceeds i64"))?;
        let lanes =
            u64::try_from(layout.lanes).map_err(|_| invalid("output element count exceeds u64"))?;
        let row_elements = u64::try_from(layout.row_elements)
            .map_err(|_| invalid("decoded row element count exceeds u64"))?;
        let index_is_i64 = i32::from(indices.dtype == DataType::Int64);
        let kind = decoder_kind(self.metadata.format());
        let dtype = self.metadata.dtype().to_onnx();
        let capture_error = if capturing || self.runtime.eager_sync_deferred() {
            self.runtime.capture_error_ptr()
        } else {
            0
        };
        let function = self.runtime.nvrtc_function(ENTRY, source(), ENTRY)?;
        let packed = cuptr(packed.data_ptr::<u8>().cast::<c_void>());
        let indices = cuptr(indices.data_ptr::<u8>().cast::<c_void>());
        let output = cuptr(output.data_ptr_mut::<u8>().cast::<c_void>());
        let grid = lanes.div_ceil(u64::from(THREADS)).clamp(1, 65_535) as u32;
        let stream = self.runtime.stream();
        let mut args = stream.launch_builder(&function);
        args.arg(&packed)
            .arg(&indices)
            .arg(&output)
            .arg(&lanes)
            .arg(&row_elements)
            .arg(&rows)
            .arg(&index_is_i64)
            .arg(&width)
            .arg(&bytes)
            .arg(&kind)
            .arg(&dtype)
            .arg(&capture_error);
        // SAFETY: argument order matches ENTRY; validated contiguous buffers
        // cover every selected packed block and output lane.
        unsafe {
            args.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err("launch DequantizePackedGather", error))?;
        self.last_call_capture_safe.store(true, Ordering::Relaxed);
        if capturing {
            Ok(())
        } else {
            self.runtime.synchronize()
        }
    }

    fn capture_support(&self) -> CaptureSupport {
        if self.last_call_capture_safe.load(Ordering::Relaxed) {
            CaptureSupport::Supported
        } else {
            CaptureSupport::Unsupported {
                reason: "DequantizePackedGather has not completed a capture-safe launch".into(),
            }
        }
    }
}
