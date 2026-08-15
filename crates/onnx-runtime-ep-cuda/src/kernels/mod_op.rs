//! ONNX `Mod` (`docs/execution/CUDA_COVERAGE.md`): elementwise remainder with NumPy
//! right-aligned broadcasting.
//!
//! `fmod=0` (the default) is integer floor-modulo whose result takes the sign of
//! the divisor; `fmod=1` is the C-style truncated remainder (`fmodf` for floats,
//! `%` for integers) whose result takes the sign of the dividend. Both match the
//! CPU EP (`elementwise.rs::c_mod`), including the divide-by-zero convention of
//! yielding `0`. Floating-point `Mod` requires `fmod=1`, per ONNX.

use std::ffi::c_void;
use std::sync::Arc;

use crate::error::{driver_err, not_implemented};
use cudarc::driver::{LaunchConfig, PushKernelArg};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const MODULE: &str = "mod_op_v1";

const SRC: &str = r#"
#define DEFINE_MOD_INT(TYPE, SUFFIX) \
extern "C" __global__ void mod_##SUFFIX( \
    const TYPE* a, const TYPE* b, TYPE* y, const unsigned long long* metadata, \
    const int rank, const unsigned long long n, const int fmod) { \
    const unsigned long long* shape = metadata; \
    const unsigned long long* a_strides = metadata + rank; \
    const unsigned long long* b_strides = metadata + rank * 2; \
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += (unsigned long long)gridDim.x * blockDim.x) { \
        unsigned long long linear = i, ai = 0, bi = 0; \
        for (int d = rank - 1; d >= 0; --d) { \
            unsigned long long coord = linear % shape[d]; \
            linear /= shape[d]; \
            ai += coord * a_strides[d]; \
            bi += coord * b_strides[d]; \
        } \
        const TYPE dividend = a[ai]; \
        const TYPE divisor = b[bi]; \
        TYPE r = 0; \
        if (divisor != 0) { \
            r = dividend % divisor; \
            if (!fmod && r != 0 && ((r < 0) != (divisor < 0))) r += divisor; \
        } \
        y[i] = r; \
    } \
}

DEFINE_MOD_INT(int, i32)
DEFINE_MOD_INT(long long, i64)

extern "C" __global__ void mod_f32(
    const float* a, const float* b, float* y, const unsigned long long* metadata,
    const int rank, const unsigned long long n, const int fmod) {
    const unsigned long long* shape = metadata;
    const unsigned long long* a_strides = metadata + rank;
    const unsigned long long* b_strides = metadata + rank * 2;
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned long long linear = i, ai = 0, bi = 0;
        for (int d = rank - 1; d >= 0; --d) {
            unsigned long long coord = linear % shape[d];
            linear /= shape[d];
            ai += coord * a_strides[d];
            bi += coord * b_strides[d];
        }
        y[i] = fmodf(a[ai], b[bi]);
    }
}
"#;

/// Right-aligned broadcast metadata for two operands against `out_shape`: the
/// output shape followed by each operand's strides (zero where an axis is
/// size-one or absent), as `u64` words ready for an H2D upload.
fn broadcast_metadata(a_shape: &[usize], b_shape: &[usize], out_shape: &[usize]) -> Vec<u64> {
    let rank = out_shape.len();
    let mut metadata = Vec::with_capacity(rank * 3);
    metadata.extend(out_shape.iter().map(|&dim| dim as u64));
    for shape in [a_shape, b_shape] {
        let offset = rank - shape.len();
        let mut strides = vec![0u64; shape.len()];
        let mut acc = 1u64;
        for d in (0..shape.len()).rev() {
            strides[d] = acc;
            acc *= shape[d] as u64;
        }
        for d in 0..rank {
            if d < offset {
                metadata.push(0);
            } else {
                let axis = d - offset;
                metadata.push(if shape[axis] == 1 { 0 } else { strides[axis] });
            }
        }
    }
    metadata
}

fn u64_bytes(values: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for &value in values {
        out.extend_from_slice(&value.to_ne_bytes());
    }
    out
}

fn grid_for(n: usize) -> u32 {
    const MAX_BLOCKS: usize = 65_535;
    n.div_ceil(BLOCK as usize).clamp(1, MAX_BLOCKS) as u32
}

fn entry_for(dtype: DataType) -> Option<&'static str> {
    match dtype {
        DataType::Float32 => Some("mod_f32"),
        DataType::Int32 => Some("mod_i32"),
        DataType::Int64 => Some("mod_i64"),
        _ => None,
    }
}

/// Read the `fmod` attribute (default 0), rejecting values other than 0/1.
pub(crate) fn read_fmod(node: &Node) -> Result<bool> {
    match node.attr("fmod") {
        None => Ok(false),
        Some(Attribute::Int(0)) => Ok(false),
        Some(Attribute::Int(1)) => Ok(true),
        Some(Attribute::Int(value)) => Err(EpError::KernelFailed(format!(
            "cuda_ep Mod: `fmod` must be 0 or 1, got {value}"
        ))),
        Some(_) => Err(EpError::KernelFailed(
            "cuda_ep Mod: `fmod` must be an integer attribute".into(),
        )),
    }
}

/// Factory for [`ModKernel`], carrying its `fmod` selector.
pub struct ModFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for ModFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ModKernel {
            fmod: read_fmod(node)?,
            runtime: self.runtime.clone(),
        }))
    }
}

/// NVRTC-backed `Mod` with right-aligned broadcasting for f32/i32/i64.
#[derive(Debug)]
pub struct ModKernel {
    fmod: bool,
    runtime: Arc<CudaRuntime>,
}

impl ModKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Mod: expected 2 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let a = &inputs[0];
        let b = &inputs[1];
        let Some(entry) = entry_for(a.dtype) else {
            return Err(not_implemented(format!(
                "Mod with dtype {:?} (supported: Float32, Int32, Int64)",
                a.dtype
            )));
        };
        if a.dtype == DataType::Float32 && !self.fmod {
            return Err(not_implemented(
                "Mod on Float32 requires fmod=1 (ONNX forbids floating-point floor-mod)",
            ));
        }
        if b.dtype != a.dtype || outputs[0].dtype != a.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Mod: A/B/output dtypes must match, got {:?}/{:?}/{:?}",
                a.dtype, b.dtype, outputs[0].dtype
            )));
        }
        for (name, view) in [("A", a), ("B", b)] {
            if !view.is_contiguous() {
                return Err(not_implemented(format!(
                    "Mod with a non-contiguous (strided) {name}; materialise it before the op"
                )));
            }
        }
        if !outputs[0].is_contiguous() {
            return Err(not_implemented(
                "Mod with a non-contiguous (strided) output; materialise it before the op",
            ));
        }

        let out_shape = onnx_runtime_ir::broadcast_shapes(a.shape, b.shape).map_err(EpError::Ir)?;
        if outputs[0].shape != out_shape.as_slice() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Mod: output shape {:?} must equal broadcast shape {out_shape:?}",
                outputs[0].shape
            )));
        }

        let n = outputs[0].numel();
        if n == 0 {
            return Ok(());
        }
        let n_u64 = u64::try_from(n)
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep Mod: {n} elements exceed u64")))?;
        let rank = i32::try_from(out_shape.len()).map_err(|_| {
            EpError::KernelFailed(format!("cuda_ep Mod: rank {} exceeds i32", out_shape.len()))
        })?;
        let fmod: i32 = i32::from(self.fmod);

        let metadata = broadcast_metadata(a.shape, b.shape, &out_shape);
        let metadata_bytes = u64_bytes(&metadata);
        let metadata_ptr = self.runtime.alloc_raw(metadata_bytes.len().max(1))?;

        let func = self.runtime.nvrtc_function(MODULE, SRC, entry)?;
        let cfg = LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let a_ptr = cuptr(a.data_ptr::<u8>() as *const c_void);
        let b_ptr = cuptr(b.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        let result = (|| {
            // SAFETY: the fresh allocation exactly covers the metadata slice.
            unsafe { self.runtime.htod(&metadata_bytes, metadata_ptr) }?;
            let mut builder = self.runtime.stream().launch_builder(&func);
            builder
                .arg(&a_ptr)
                .arg(&b_ptr)
                .arg(&y_ptr)
                .arg(&metadata_ptr)
                .arg(&rank)
                .arg(&n_u64)
                .arg(&fmod);
            // SAFETY: `a_ptr`/`b_ptr`/`y_ptr` cover their validated contiguous
            // elements and `metadata_ptr` holds `rank*3` u64 words per the ABI.
            unsafe { builder.launch(cfg) }
                .map_err(|error| driver_err(&format!("launch {entry}"), error))?;
            self.runtime.synchronize()
        })();

        // SAFETY: the synchronize (or a failed launch) guarantees no kernel is
        // still reading the metadata buffer; it is freed exactly once here.
        let cleanup = unsafe { self.runtime.free_raw(metadata_ptr) };
        result.and(cleanup)
    }
}

impl Kernel for ModKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "Mod allocates and uploads broadcast metadata on every call",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_present_in_source() {
        assert!(SRC.contains("DEFINE_MOD_INT(int, i32)"));
        assert!(SRC.contains("DEFINE_MOD_INT(long long, i64)"));
        assert!(SRC.contains("mod_f32"));
    }

    #[test]
    fn entry_dispatch_covers_supported_dtypes() {
        assert_eq!(entry_for(DataType::Float32), Some("mod_f32"));
        assert_eq!(entry_for(DataType::Int32), Some("mod_i32"));
        assert_eq!(entry_for(DataType::Int64), Some("mod_i64"));
        assert_eq!(entry_for(DataType::BFloat16), None);
    }

    #[test]
    fn broadcast_metadata_layout() {
        // A [2,3] and B [3] into [2,3]: A row-major, B leading axis broadcast.
        assert_eq!(
            broadcast_metadata(&[2, 3], &[3], &[2, 3]),
            vec![2, 3, /*a*/ 3, 1, /*b*/ 0, 1]
        );
    }
}
