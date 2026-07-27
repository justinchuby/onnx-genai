//! Deterministic CUDA implementation of ONNX `Trilu` (opset 14): retain the
//! upper- or lower-triangular part of each of the input's trailing 2-D matrices
//! and zero the rest.
//!
//! The mask is a pure data-movement decision, so a single dtype-agnostic
//! byte-wise kernel serves every fixed-width dtype: an element is either copied
//! verbatim or written as all-zero bytes (the canonical zero for every integer,
//! float, and bool representation), matching the CPU EP's `movement_ops.rs`
//! `Trilu` semantics exactly.
//!
//! The optional `k` diagonal offset is a scalar `Int64` **device** tensor. It is
//! read once per call with a host-blocking copy outside CUDA-graph capture, and
//! a warmed `(dtype, shape, k)` signature keeps the recorded launch valid on
//! replay (mirroring [`super::cumsum`]).

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
extern "C" __global__ void trilu_bytes(
    const unsigned char* input, unsigned char* output,
    unsigned long long elements, unsigned long long rows, unsigned long long cols,
    long long k, int upper, int elem_bytes) {
  for (unsigned long long e = blockIdx.x * blockDim.x + threadIdx.x; e < elements;
       e += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long col = e % cols;
    unsigned long long row = (e / cols) % rows;
    long long diff = (long long)col - (long long)row;
    int keep = upper ? (diff >= k) : (diff <= k);
    unsigned long long base = e * (unsigned long long)elem_bytes;
    for (int b = 0; b < elem_bytes; ++b)
      output[base + b] = keep ? input[base + b] : (unsigned char)0;
  }
}
"#;

pub struct TriluFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for TriluFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let upper = match node.attr("upper") {
            None => true,
            Some(Attribute::Int(0)) => false,
            Some(Attribute::Int(1)) => true,
            Some(_) => {
                return Err(EpError::KernelFailed(
                    "cuda_ep Trilu: attribute 'upper' must be 0 or 1".into(),
                ));
            }
        };
        Ok(Box::new(TriluKernel {
            runtime: self.runtime.clone(),
            upper,
            warmed_signature: Mutex::new(None),
        }))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TriluCaptureSignature {
    dtype: DataType,
    shape: Vec<usize>,
    k: i64,
}

struct TriluKernel {
    runtime: Arc<CudaRuntime>,
    upper: bool,
    warmed_signature: Mutex<Option<TriluCaptureSignature>>,
}

impl Kernel for TriluKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if !(1..=2).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Trilu: expected 1..=2 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let input = &inputs[0];
        let output = &mut outputs[0];
        if !input.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented("Trilu with non-contiguous tensors"));
        }
        if output.dtype != input.dtype || output.shape != input.shape {
            return Err(EpError::KernelFailed(
                "cuda_ep Trilu: output must match input shape and dtype".into(),
            ));
        }
        if input.shape.len() < 2 {
            return Err(EpError::KernelFailed(
                "cuda_ep Trilu: input must have rank of at least 2".into(),
            ));
        }
        let elem_bytes = i32::try_from(input.dtype.byte_size())
            .ok()
            .filter(|&b| b > 0);
        let Some(elem_bytes) = elem_bytes else {
            return Err(not_implemented(format!(
                "Trilu for packed or variable-width dtype {:?}",
                input.dtype
            )));
        };

        let k_present = inputs
            .get(1)
            .is_some_and(|k| !k.is_absent() && k.numel() != 0);
        if k_present {
            let k_input = &inputs[1];
            if k_input.dtype != DataType::Int64 || k_input.numel() != 1 {
                return Err(EpError::KernelFailed(
                    "cuda_ep Trilu: k input must be a scalar Int64 tensor".into(),
                ));
            }
            if !k_input.is_contiguous() {
                return Err(not_implemented("Trilu with a strided k input"));
            }
        }

        let capturing = self.runtime.is_capturing()?;
        let mut warmed_signature = self.warmed_signature.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep Trilu: capture signature lock was poisoned".into())
        })?;
        let k = if capturing {
            let signature = warmed_signature.as_ref().ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep Trilu: capture started before the fixed k/shape was warmed".into(),
                )
            })?;
            if signature.dtype != input.dtype || signature.shape != input.shape {
                return Err(EpError::KernelFailed(
                    "cuda_ep Trilu: shape or dtype changed during CUDA graph capture".into(),
                ));
            }
            signature.k
        } else if k_present {
            let mut bytes = [0_u8; 8];
            // SAFETY: `k` is a live contiguous 1-element Int64 device allocation;
            // `dtoh` copies exactly 8 bytes and synchronizes before returning.
            unsafe {
                self.runtime.dtoh(
                    &mut bytes,
                    cuptr(inputs[1].data_ptr::<i64>() as *const c_void),
                )?
            };
            i64::from_ne_bytes(bytes)
        } else {
            0
        };

        let update_signature = |signature: &mut Option<TriluCaptureSignature>| {
            if !capturing {
                *signature = Some(TriluCaptureSignature {
                    dtype: input.dtype,
                    shape: input.shape.to_vec(),
                    k,
                });
            }
        };

        let elements = input.numel();
        if elements == 0 {
            update_signature(&mut warmed_signature);
            return Ok(());
        }
        let rows = input.shape[input.shape.len() - 2] as u64;
        let cols = input.shape[input.shape.len() - 1] as u64;
        let elements_u64 = elements as u64;
        let upper = i32::from(self.upper);

        let func = self
            .runtime
            .nvrtc_function("trilu", SOURCE, "trilu_bytes")?;
        let input_ptr = cuptr(input.data_ptr::<u8>() as *const c_void);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&input_ptr)
            .arg(&output_ptr)
            .arg(&elements_u64)
            .arg(&rows)
            .arg(&cols)
            .arg(&k)
            .arg(&upper)
            .arg(&elem_bytes);
        // SAFETY: `func` is the compiled `trilu_bytes` entry; its argument list
        // matches this builder, and both pointers are live device allocations of
        // `elements * elem_bytes` bytes covered by the grid-stride loop.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    elements_u64.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32,
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err("launch Trilu", e))?;
        update_signature(&mut warmed_signature);
        Ok(())
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        match self.warmed_signature.lock() {
            Ok(signature) if signature.is_some() => onnx_runtime_ep_api::CaptureSupport::Supported,
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "Trilu must warm its fixed k/shape signature before capture",
            ),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "Trilu capture signature lock was poisoned",
            ),
        }
    }
}
