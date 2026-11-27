//! Native CUDA ReverseSequence. Only the length metadata is read back for validation.

use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::reverse_sequence::Axes;
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::Node;

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
extern "C" __global__ void reverse_sequence_bytes(
    const unsigned char* input, const long long* lengths, unsigned char* output,
    unsigned long long batch_size, unsigned long long time_size, int batch_first,
    unsigned long long block_bytes, unsigned long long bytes) {
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
       i < bytes; i += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long block = i / block_bytes;
    const unsigned long long batch = batch_first ? block / time_size : block % batch_size;
    const unsigned long long time = batch_first ? block % time_size : block / batch_size;
    const unsigned long long length = lengths[batch];
    const unsigned long long source_time = time < length ? length - 1 - time : time;
    const unsigned long long source_block = batch_first
        ? batch * time_size + source_time : source_time * batch_size + batch;
    output[i] = input[source_block * block_bytes + i % block_bytes];
  }
}
"#;

pub struct ReverseSequenceFactory {
    pub runtime: Arc<CudaRuntime>,
    pub version: u64,
}

impl KernelFactory for ReverseSequenceFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ReverseSequence {
            runtime: self.runtime.clone(),
            axes: Axes::from_node(node)?,
            version: self.version,
        }))
    }
}

struct ReverseSequence {
    runtime: Arc<CudaRuntime>,
    axes: Axes,
    version: u64,
}

impl Kernel for ReverseSequence {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let ([input, lengths], [output]) = (inputs, outputs) else {
            return Err(EpError::KernelFailed(
                "ReverseSequence: expected two inputs and one output".into(),
            ));
        };
        let plan = self.axes.plan(self.version, input, lengths, output)?;
        input.validate()?;
        lengths.validate()?;
        output.validate()?;
        if !input.is_contiguous() || !lengths.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented(
                "ReverseSequence with non-contiguous tensors",
            ));
        }
        if self.runtime.is_capturing()? {
            return Err(not_implemented(
                "ReverseSequence length validation during CUDA graph capture",
            ));
        }
        let lengths_origin = lengths.data_ptr::<i64>();
        if plan.bytes != 0 && !lengths_origin.is_aligned() {
            return Err(EpError::InvalidTensorView {
                reason: "ReverseSequence: `sequence_lens` data must be aligned for Int64".into(),
            });
        }
        let lengths_ptr = cuptr(lengths_origin.cast());
        let mut lengths_bytes = vec![0; plan.lengths_bytes];
        if !lengths_bytes.is_empty() {
            // SAFETY: the validated contiguous view holds one i64 per batch.
            unsafe { self.runtime.dtoh(&mut lengths_bytes, lengths_ptr)? };
        }
        plan.validate_lengths(
            lengths_bytes
                .chunks_exact(8)
                .map(|raw| i64::from_ne_bytes(raw.try_into().unwrap())),
        )?;
        if plan.bytes == 0 {
            return Ok(());
        }

        let function =
            self.runtime
                .nvrtc_function("reverse_sequence_v1", SOURCE, "reverse_sequence_bytes")?;
        let input_ptr = cuptr(input.data_ptr::<u8>().cast());
        let output_ptr = cuptr(output.data_ptr_mut::<u8>().cast());
        let batch = plan.batch as u64;
        let time = plan.time as u64;
        let batch_first = i32::from(plan.batch_first);
        let block_bytes = plan.block_bytes as u64;
        let bytes = plan.bytes as u64;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&input_ptr)
            .arg(&lengths_ptr)
            .arg(&output_ptr)
            .arg(&batch)
            .arg(&time)
            .arg(&batch_first)
            .arg(&block_bytes)
            .arg(&bytes);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    bytes.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32,
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err("launch ReverseSequence", error))?;
        self.runtime.synchronize()
    }

    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported(
            "ReverseSequence validates dynamic sequence_lens with a synchronous readback",
        )
    }
}
