//! ONNX `Mod` (`docs/execution/CUDA_COVERAGE.md`): elementwise remainder with NumPy
//! right-aligned broadcasting.
//!
//! `fmod=0` rounds the quotient down; `fmod=1` truncates it. Mod-28 permits
//! float floor remainder. An integer zero divisor yields 0, as on CPU.

use std::ffi::c_void;
use std::sync::Arc;

use crate::error::{driver_err, not_implemented};
use cudarc::driver::{LaunchConfig, PushKernelArg};

use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, OutputMetadata, Result, TensorMetadata, TensorMut, TensorView,
    WorkspacePreparation, WorkspaceView,
};
use onnx_runtime_ir::modulo::{FLOAT_FLOOR_SINCE, ModMode};
use onnx_runtime_ir::{DataType, Node};

use crate::runtime::{CudaRuntime, cuptr};

use super::metadata_workspace::MetadataWorkspace;
const METADATA_WORDS_PER_AXIS: usize = 3;

const BLOCK: u32 = 256;
const MODULE: &str = "mod_op_v2";

pub(super) const LEGACY_DTYPES: &[DataType] =
    &[DataType::Float32, DataType::Int32, DataType::Int64];
const SRC: &str = r#"
#if __has_include(<cuda_fp16.h>) && __has_include(<cuda_bf16.h>)
#define NXRT_HAS_HALF 1
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

template <typename T>
__device__ T mod_signed(T a, T b, int trunc) {
    // The quotient for MIN / -1 overflows, but its remainder is exactly 0.
    if (b == 0 || b == -1) return 0;
    T r = a % b;
    return !trunc && r != 0 && ((r < 0) != (b < 0)) ? r + b : r;
}

template <typename T>
__device__ T mod_unsigned(T a, T b) {
    return b == 0 ? 0 : a % b;
}

// Correct the truncated remainder rather than forming a quotient, which can
// overflow or lose precision even when the remainder is representable.
template <typename T>
__device__ T mod_floor(T r, T b) {
    if (isnan(r)) return r;
    if (r == 0) return copysign((T)0, b);
    return signbit(r) != signbit(b) ? r + b : r;
}

#define DEFINE_MOD(TYPE, SUFFIX, EXPR) \
extern "C" __global__ void mod_##SUFFIX( \
    const TYPE* a, const TYPE* b, TYPE* y, const unsigned long long* metadata, \
    const int rank, const unsigned long long n, const int trunc) { \
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
        y[i] = EXPR; \
    } \
}

DEFINE_MOD(signed char, i8, mod_signed(dividend, divisor, trunc))
DEFINE_MOD(short, i16, mod_signed(dividend, divisor, trunc))
DEFINE_MOD(int, i32, mod_signed(dividend, divisor, trunc))
DEFINE_MOD(long long, i64, mod_signed(dividend, divisor, trunc))
DEFINE_MOD(unsigned char, u8, mod_unsigned(dividend, divisor))
DEFINE_MOD(unsigned short, u16, mod_unsigned(dividend, divisor))
DEFINE_MOD(unsigned int, u32, mod_unsigned(dividend, divisor))
DEFINE_MOD(unsigned long long, u64, mod_unsigned(dividend, divisor))

DEFINE_MOD(float, f32, fmodf(dividend, divisor))
DEFINE_MOD(float, floor_f32, mod_floor(fmodf(dividend, divisor), divisor))
DEFINE_MOD(double, f64, fmod(dividend, divisor))
DEFINE_MOD(double, floor_f64, mod_floor(fmod(dividend, divisor), divisor))
#ifdef NXRT_HAS_HALF
#define DEFINE_LOW_MOD(TYPE, SUFFIX, LOAD, STORE) \
    DEFINE_MOD(TYPE, SUFFIX, STORE(fmodf(LOAD(dividend), LOAD(divisor)))) \
    DEFINE_MOD(TYPE, floor_##SUFFIX, \
        STORE(mod_floor(fmodf(LOAD(dividend), LOAD(divisor)), LOAD(divisor))))
DEFINE_LOW_MOD(__half, f16, __half2float, __float2half_rn)
DEFINE_LOW_MOD(__nv_bfloat16, bf16, __bfloat162float, __float2bfloat16_rn)
#endif
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

fn grid_for(n: usize) -> u32 {
    const MAX_BLOCKS: usize = 65_535;
    n.div_ceil(BLOCK as usize).clamp(1, MAX_BLOCKS) as u32
}

pub(super) fn entry_for(
    dtype: DataType,
    opset: u64,
    mode: ModMode,
) -> std::result::Result<&'static str, String> {
    use DataType::*;
    if opset < FLOAT_FLOOR_SINCE && !LEGACY_DTYPES.contains(&dtype) {
        return Err(format!(
            "CUDA Mod at opset {opset} supports Float32, Int32 and Int64, got {dtype:?}"
        ));
    }
    mode.validate_dtype(opset, dtype)?;
    Ok(match (dtype, mode) {
        (Float32, ModMode::Floor) => "mod_floor_f32",
        (Float32, ModMode::Trunc) => "mod_f32",
        (Float64, ModMode::Floor) => "mod_floor_f64",
        (Float64, ModMode::Trunc) => "mod_f64",
        (Float16, ModMode::Floor) => "mod_floor_f16",
        (Float16, ModMode::Trunc) => "mod_f16",
        (BFloat16, ModMode::Floor) => "mod_floor_bf16",
        (BFloat16, ModMode::Trunc) => "mod_bf16",
        (Int8, _) => "mod_i8",
        (Int16, _) => "mod_i16",
        (Int32, _) => "mod_i32",
        (Int64, _) => "mod_i64",
        (Uint8, _) => "mod_u8",
        (Uint16, _) => "mod_u16",
        (Uint32, _) => "mod_u32",
        (Uint64, _) => "mod_u64",
        _ => {
            return Err(format!(
                "CUDA Mod requires an ONNX numeric dtype, got {dtype:?}"
            ));
        }
    })
}

/// Factory for [`ModKernel`], carrying its `fmod` selector.
pub struct ModFactory {
    pub runtime: Arc<CudaRuntime>,
    pub since_version: u64,
}

impl KernelFactory for ModFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ModKernel {
            mode: ModMode::parse(node)
                .map_err(|reason| EpError::KernelFailed(format!("CUDA Mod: {reason}")))?,
            since_version: self.since_version,
            runtime: self.runtime.clone(),
        }))
    }
}

/// NVRTC-backed `Mod` with right-aligned broadcasting.
#[derive(Debug)]
pub struct ModKernel {
    mode: ModMode,
    since_version: u64,
    runtime: Arc<CudaRuntime>,
}

impl ModKernel {
    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Mod: expected 2 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        if self.runtime.is_capturing()? {
            return Err(not_implemented(
                "Mod during CUDA graph capture: borrowed broadcast metadata requires an eager host upload",
            ));
        }
        let a = &inputs[0];
        let b = &inputs[1];
        let entry = entry_for(a.dtype, self.since_version, self.mode).map_err(not_implemented)?;
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
        let trunc: i32 = i32::from(self.mode == ModMode::Trunc);
        if matches!(a.dtype, DataType::Float16 | DataType::BFloat16) {
            self.runtime.require_nvrtc_half_headers("Mod")?;
        }

        let metadata = broadcast_metadata(a.shape, b.shape, &out_shape);
        let metadata_ptr = MetadataWorkspace::for_records(
            "Mod",
            out_shape.len(),
            METADATA_WORDS_PER_AXIS,
            &OutputMetadata::from_view(&outputs[0]),
        )?
        .bind(workspace)?
        .upload(&self.runtime, &metadata)?;

        let func = self.runtime.nvrtc_function(MODULE, SRC, entry)?;
        let cfg = LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let a_ptr = cuptr(a.data_ptr::<u8>() as *const c_void);
        let b_ptr = cuptr(b.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&a_ptr)
            .arg(&b_ptr)
            .arg(&y_ptr)
            .arg(&metadata_ptr)
            .arg(&rank)
            .arg(&n_u64)
            .arg(&trunc);
        // SAFETY: `a_ptr`/`b_ptr`/`y_ptr` cover their validated contiguous
        // elements and `metadata_ptr` holds `rank*3` u64 words per the ABI.
        unsafe { builder.launch(cfg) }
            .map_err(|error| driver_err(&format!("launch {entry}"), error))?;
        self.runtime.synchronize()
    }
}

impl Kernel for ModKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }
    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.run(inputs, outputs, workspace)
    }
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        if inputs.len() != 2 || outputs.len() != 1 || inputs.iter().any(|input| !input.present) {
            return Err(EpError::KernelFailed(
                "cuda_ep Mod: workspace planning requires two present inputs and one output".into(),
            ));
        }
        let shape = onnx_runtime_ir::broadcast_shapes(inputs[0].shape, inputs[1].shape)
            .map_err(EpError::Ir)?;
        let rank = if shape.contains(&0) { 0 } else { shape.len() };
        Ok(WorkspacePreparation::Ready(
            MetadataWorkspace::for_records("Mod", rank, METADATA_WORDS_PER_AXIS, &outputs[0])?
                .requirement(),
        ))
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "Mod uploads borrowed broadcast metadata from the host on each eager call",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_metadata_layout() {
        // A [2,3] and B [3] into [2,3]: A row-major, B leading axis broadcast.
        assert_eq!(
            broadcast_metadata(&[2, 3], &[3], &[2, 3]),
            vec![2, 3, /*a*/ 3, 1, /*b*/ 0, 1]
        );
    }
}
