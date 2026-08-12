//! Integer **bitwise** ops (`BitwiseAnd`, `BitwiseOr`, `BitwiseXor`,
//! `BitwiseNot`) and the unsigned **`BitShift`** on the GPU via runtime-compiled
//! (NVRTC) `extern "C"` kernels. This mirrors the CPU EP's coverage
//! (`crates/onnx-runtime-ep-cpu/src/kernels/bitwise.rs` and `bitshift.rs`) so a
//! native CUDA model does not fall back to the CPU for these ops.
//!
//! ## Scope (all limits are actionable errors, never panics — RULES.md #1)
//!
//! * **Bitwise binary** (`BitwiseAnd`, `BitwiseOr`, `BitwiseXor`): two
//!   broadcast-compatible integer inputs of the **same** dtype
//!   (Int8/16/32/64, Uint8/16/32/64) → same-dtype output. NumPy-style
//!   right-aligned broadcasting, reusing the [`super::elementwise`] metadata.
//! * **Bitwise unary** (`BitwiseNot`): one integer input → same-dtype output,
//!   bitwise complement (`~x`).
//! * **BitShift** (`direction` = `LEFT`/`RIGHT`): two broadcast-compatible
//!   **unsigned** integer inputs (Uint8/16/32/64). The shift amount is taken
//!   modulo the exact CPU contract — `checked_shl`/`checked_shr` yields `0` when
//!   the (u32-truncated) amount is `>=` the operand bit width — so the kernel
//!   mirrors `bitshift.rs` bit-for-bit.
//!
//! Each op is one thread-per-element grid-stride kernel (bandwidth-bound).

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::elementwise::{
    BroadcastMetadataCache, BroadcastMetadataKey, capture_shape_eligible,
    require_matching_capture_signature,
};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

/// Threads per block for the 1-D grids (a full warp-multiple block).
const BLOCK: u32 = 256;

/// Grid dimension for `n` elements at [`BLOCK`] threads, capped so a huge tensor
/// still fits the grid limit (grid-stride kernels still cover every element).
fn grid_for(n: usize) -> u32 {
    const MAX_BLOCKS: usize = 65_535;
    n.div_ceil(BLOCK as usize).clamp(1, MAX_BLOCKS) as u32
}

/// `n` as `u64`, matching the kernels' `unsigned long long` count parameter.
fn count_u64(op: &str, n: usize) -> Result<u64> {
    u64::try_from(n)
        .map_err(|_| EpError::KernelFailed(format!("cuda_ep {op}: {n} elements exceed u64")))
}

/// The NVRTC type token and entry-point suffix for an integer dtype.
fn integer_type_suffix(dtype: DataType) -> Option<(&'static str, &'static str)> {
    Some(match dtype {
        DataType::Int8 => ("signed char", "i8"),
        DataType::Int16 => ("short", "i16"),
        DataType::Int32 => ("int", "i32"),
        DataType::Int64 => ("long long", "i64"),
        DataType::Uint8 => ("unsigned char", "u8"),
        DataType::Uint16 => ("unsigned short", "u16"),
        DataType::Uint32 => ("unsigned int", "u32"),
        DataType::Uint64 => ("unsigned long long", "u64"),
        _ => return None,
    })
}

/// The entry-point suffix for an unsigned integer dtype (BitShift operands).
fn unsigned_suffix(dtype: DataType) -> Option<&'static str> {
    match dtype {
        DataType::Uint8 => Some("u8"),
        DataType::Uint16 => Some("u16"),
        DataType::Uint32 => Some("u32"),
        DataType::Uint64 => Some("u64"),
        _ => None,
    }
}

// ===========================================================================
// Bitwise binary (same integer dtype in/out) — NumPy broadcasting
// ===========================================================================

/// NVRTC source: one `extern "C"` kernel per bitwise op/integer dtype. The
/// output dtype equals the operand dtype (unlike the bool-producing logical
/// ops). `&`/`|`/`^` are bit-level identical on signed and unsigned operands.
const BITWISE_BINARY_SRC: &str = r#"
__device__ __forceinline__ void broadcast_indices(unsigned long long out, const unsigned long long* m, int rank, unsigned long long* ai, unsigned long long* bi) {
    *ai = 0; *bi = 0;
    for (int axis = rank - 1; axis >= 0; --axis) {
        unsigned long long coord = out % m[axis]; out /= m[axis];
        *ai += coord * m[rank + axis]; *bi += coord * m[2 * rank + axis];
    }
}
#define DEFINE_BITWISE(name, type, suffix, expr) \
extern "C" __global__ void name##_##suffix(const type* a, const type* b, type* y, const unsigned long long* m, int rank, const unsigned long long n) { \
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) { \
        unsigned long long ai, bi; broadcast_indices(i, m, rank, &ai, &bi); y[i] = (type)(expr); \
    } \
}
#define DEFINE_BITWISE_FOR_TYPE(type, suffix) \
DEFINE_BITWISE(band, type, suffix, a[ai] & b[bi]) \
DEFINE_BITWISE(bor, type, suffix, a[ai] | b[bi]) \
DEFINE_BITWISE(bxor, type, suffix, a[ai] ^ b[bi])
DEFINE_BITWISE_FOR_TYPE(signed char, i8)
DEFINE_BITWISE_FOR_TYPE(short, i16)
DEFINE_BITWISE_FOR_TYPE(int, i32)
DEFINE_BITWISE_FOR_TYPE(long long, i64)
DEFINE_BITWISE_FOR_TYPE(unsigned char, u8)
DEFINE_BITWISE_FOR_TYPE(unsigned short, u16)
DEFINE_BITWISE_FOR_TYPE(unsigned int, u32)
DEFINE_BITWISE_FOR_TYPE(unsigned long long, u64)
"#;

const BITWISE_BINARY_MODULE: &str = "bitwise_binary_int";

/// NVRTC source: unsigned `BitShift` (LEFT/RIGHT). The shift amount is truncated
/// to `unsigned int` and compared against the operand bit width, matching the
/// CPU `checked_shl`/`checked_shr(amount as u32)` contract exactly (an amount
/// `>=` the width yields `0`). Small-type operands promote to `int` for the
/// shift, so the `(type)` store reproduces the CPU wrapping narrow.
const BITSHIFT_SRC: &str = r#"
__device__ __forceinline__ void broadcast_indices(unsigned long long out, const unsigned long long* m, int rank, unsigned long long* ai, unsigned long long* bi) {
    *ai = 0; *bi = 0;
    for (int axis = rank - 1; axis >= 0; --axis) {
        unsigned long long coord = out % m[axis]; out /= m[axis];
        *ai += coord * m[rank + axis]; *bi += coord * m[2 * rank + axis];
    }
}
#define DEFINE_SHIFT(name, type, suffix, bits, op) \
extern "C" __global__ void name##_##suffix(const type* a, const type* b, type* y, const unsigned long long* m, int rank, const unsigned long long n) { \
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) { \
        unsigned long long ai, bi; broadcast_indices(i, m, rank, &ai, &bi); \
        unsigned int amount = (unsigned int)b[bi]; \
        y[i] = (amount >= bits) ? (type)0 : (type)(a[ai] op amount); \
    } \
}
#define DEFINE_SHIFT_FOR_TYPE(type, suffix, bits) \
DEFINE_SHIFT(shl, type, suffix, bits, <<) \
DEFINE_SHIFT(shr, type, suffix, bits, >>)
DEFINE_SHIFT_FOR_TYPE(unsigned char, u8, 8u)
DEFINE_SHIFT_FOR_TYPE(unsigned short, u16, 16u)
DEFINE_SHIFT_FOR_TYPE(unsigned int, u32, 32u)
DEFINE_SHIFT_FOR_TYPE(unsigned long long, u64, 64u)
"#;

const BITSHIFT_MODULE: &str = "bitwise_shift_uint";

/// A supported bitwise binary op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitwiseBinaryOp {
    And,
    Or,
    Xor,
}

impl BitwiseBinaryOp {
    fn stem(self) -> &'static str {
        match self {
            BitwiseBinaryOp::And => "band",
            BitwiseBinaryOp::Or => "bor",
            BitwiseBinaryOp::Xor => "bxor",
        }
    }

    fn op_name(self) -> &'static str {
        match self {
            BitwiseBinaryOp::And => "BitwiseAnd",
            BitwiseBinaryOp::Or => "BitwiseOr",
            BitwiseBinaryOp::Xor => "BitwiseXor",
        }
    }
}

/// Left or right `BitShift`, selected by the `direction` attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShiftDirection {
    Left,
    Right,
}

impl ShiftDirection {
    fn stem(self) -> &'static str {
        match self {
            ShiftDirection::Left => "shl",
            ShiftDirection::Right => "shr",
        }
    }
}

/// The dtype/kernel contract for a binary integer op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BinaryKind {
    /// Bitwise `&`/`|`/`^` across every integer dtype.
    Bitwise(BitwiseBinaryOp),
    /// Unsigned-only shift with a fixed direction.
    Shift(ShiftDirection),
}

impl BinaryKind {
    fn op_name(self) -> &'static str {
        match self {
            BinaryKind::Bitwise(op) => op.op_name(),
            BinaryKind::Shift(_) => "BitShift",
        }
    }

    /// Resolve the NVRTC entry point for `dtype`, or `None` when the dtype is
    /// outside this op's supported set.
    fn entry(self, dtype: DataType) -> Option<String> {
        match self {
            BinaryKind::Bitwise(op) => {
                let (_, suffix) = integer_type_suffix(dtype)?;
                Some(format!("{}_{suffix}", op.stem()))
            }
            BinaryKind::Shift(dir) => {
                let suffix = unsigned_suffix(dtype)?;
                Some(format!("{}_{suffix}", dir.stem()))
            }
        }
    }

    fn module(self) -> &'static str {
        match self {
            BinaryKind::Bitwise(_) => BITWISE_BINARY_MODULE,
            BinaryKind::Shift(_) => BITSHIFT_MODULE,
        }
    }

    fn src(self) -> &'static str {
        match self {
            BinaryKind::Bitwise(_) => BITWISE_BINARY_SRC,
            BinaryKind::Shift(_) => BITSHIFT_SRC,
        }
    }
}

/// Factory for a bitwise `BitwiseAnd`/`BitwiseOr`/`BitwiseXor` kernel.
pub struct BitwiseBinaryFactory {
    pub op: BitwiseBinaryOp,
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for BitwiseBinaryFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(BinaryIntKernel {
            kind: BinaryKind::Bitwise(self.op),
            runtime: self.runtime.clone(),
            metadata: Mutex::new(BroadcastMetadataCache::new(self.runtime.clone())),
            last_capture_safe_signature: Mutex::new(None),
            capture_seq_independent: false,
        }))
    }
}

/// Factory for the unsigned `BitShift` kernel (reads the `direction` attribute).
pub struct BitShiftFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for BitShiftFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let direction = match node.attr("direction").and_then(Attribute::as_str) {
            Some("LEFT") => ShiftDirection::Left,
            Some("RIGHT") => ShiftDirection::Right,
            Some(other) => {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep BitShift: direction attribute must be LEFT or RIGHT, got {other:?}"
                )));
            }
            None => {
                return Err(EpError::KernelFailed(
                    "cuda_ep BitShift: direction attribute is required".into(),
                ));
            }
        };
        Ok(Box::new(BinaryIntKernel {
            kind: BinaryKind::Shift(direction),
            runtime: self.runtime.clone(),
            metadata: Mutex::new(BroadcastMetadataCache::new(self.runtime.clone())),
            last_capture_safe_signature: Mutex::new(None),
            capture_seq_independent: false,
        }))
    }
}

/// The dtype + operand/broadcast shapes a captured launch is pinned to.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BinaryCaptureSignature {
    dtype: DataType,
    shapes: BroadcastMetadataKey,
}

/// NVRTC-backed integer binary kernel producing a **same-dtype** output. Covers
/// both the bitwise family (every integer dtype) and unsigned `BitShift` via
/// [`BinaryKind`], with NumPy-style right-aligned broadcasting and the same
/// persistent-metadata capture seam as [`super::pointwise::BinaryPredKernel`].
#[derive(Debug)]
pub struct BinaryIntKernel {
    kind: BinaryKind,
    runtime: Arc<CudaRuntime>,
    /// Persistent broadcast metadata so a captured launch performs no per-step
    /// host allocation/upload/free/synchronize.
    metadata: Mutex<BroadcastMetadataCache>,
    /// The dtype/shape signature recorded by the most recent successful
    /// fixed-decode call. `Some` iff the op is currently capture-safe.
    last_capture_safe_signature: Mutex<Option<BinaryCaptureSignature>>,
    /// Metadata-derived seq-independence: `true` iff all IR output dims are
    /// statically known (no growing seq axis), making the op capture-eligible
    /// regardless of the runtime row count.
    capture_seq_independent: bool,
}

impl BinaryIntKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let mut last_signature = self.last_capture_safe_signature.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep bitwise capture signature lock was poisoned".into())
        })?;
        let warmed_signature = last_signature.take();
        let op = self.kind.op_name();
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: expected 2 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let a = &inputs[0];
        let b = &inputs[1];
        let Some(entry) = self.kind.entry(a.dtype) else {
            return Err(not_implemented(format!(
                "{op}: operand dtype {:?} not supported on CUDA EP",
                a.dtype
            )));
        };
        if b.dtype != a.dtype {
            return Err(not_implemented(format!(
                "{op}: operands must have the same dtype on CUDA EP (got {:?} and {:?})",
                a.dtype, b.dtype
            )));
        }
        if outputs[0].dtype != a.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output dtype {:?} must equal input dtype {:?}",
                outputs[0].dtype, a.dtype
            )));
        }
        if !a.is_contiguous() || !b.is_contiguous() || !outputs[0].is_contiguous() {
            return Err(not_implemented(format!(
                "{op} with a non-contiguous (strided) operand; \
                 insert an explicit copy to materialise it before the op"
            )));
        }

        let out_shape = onnx_runtime_ir::broadcast_shapes(a.shape, b.shape).map_err(EpError::Ir)?;
        if outputs[0].shape != out_shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output shape {:?} must equal broadcast shape {:?}",
                outputs[0].shape, out_shape
            )));
        }

        let n = outputs[0].numel();
        let n_u64 = count_u64(op, n)?;
        let a_ptr = cuptr(a.data_ptr::<u8>() as *const c_void);
        let b_ptr = cuptr(b.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        let capture_eligible = capture_shape_eligible(self.capture_seq_independent, &out_shape);
        let current_signature = capture_eligible.then(|| BinaryCaptureSignature {
            dtype: a.dtype,
            shapes: BroadcastMetadataKey {
                a_shape: a.shape.to_vec(),
                b_shape: b.shape.to_vec(),
                out_shape: out_shape.clone(),
            },
        });
        require_matching_capture_signature(
            &self.runtime,
            op,
            warmed_signature.as_ref(),
            current_signature.as_ref(),
        )?;

        let func = self
            .runtime
            .nvrtc_function(self.kind.module(), self.kind.src(), &entry)?;
        let mut metadata = self.metadata.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep bitwise metadata lock was poisoned".into())
        })?;
        let metadata_ptr = metadata.prepare(a.shape, b.shape, &out_shape)?;
        let rank = i32::try_from(out_shape.len())
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep {op}: rank exceeds i32")))?;
        let cfg = LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&a_ptr)
            .arg(&b_ptr)
            .arg(&y_ptr)
            .arg(&metadata_ptr)
            .arg(&rank)
            .arg(&n_u64);
        // SAFETY: `func` is the compiled bitwise/shift entry; its argument list is
        // (const T*, const T*, T*, metadata, rank, count), where T matches the
        // validated same-dtype operands. All pointers cover their allocations,
        // with matching rank/count/indexing; the metadata pointer is the
        // persistent cache buffer, valid across replays.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        *last_signature = current_signature;
        Ok(())
    }
}

impl Kernel for BinaryIntKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        match self.last_capture_safe_signature.lock() {
            Ok(signature) if signature.is_some() => onnx_runtime_ep_api::CaptureSupport::Supported,
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(format!(
                "{} broadcast shape/dtype signature does not match the warmed capture signature",
                self.kind.op_name()
            )),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(format!(
                "{} capture signature is unavailable because its state lock was poisoned",
                self.kind.op_name()
            )),
        }
    }

    fn set_capture_seq_independent(&mut self, seq_independent: bool) {
        self.capture_seq_independent = seq_independent;
    }
}

// ===========================================================================
// BitwiseNot (integer → same integer dtype)
// ===========================================================================

/// NVRTC source: bitwise complement over every integer dtype.
const BITWISE_NOT_SRC: &str = r#"
#define DEFINE_NOT(type, suffix) \
extern "C" __global__ void bnot_##suffix(const type* x, type* y, const unsigned long long n) { \
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) \
        y[i] = (type)(~x[i]); \
}
DEFINE_NOT(signed char, i8)
DEFINE_NOT(short, i16)
DEFINE_NOT(int, i32)
DEFINE_NOT(long long, i64)
DEFINE_NOT(unsigned char, u8)
DEFINE_NOT(unsigned short, u16)
DEFINE_NOT(unsigned int, u32)
DEFINE_NOT(unsigned long long, u64)
"#;

const BITWISE_NOT_MODULE: &str = "bitwise_not_int";

/// Factory for [`BitwiseNotKernel`] (no attributes).
pub struct BitwiseNotFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for BitwiseNotFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(BitwiseNotKernel {
            runtime: self.runtime.clone(),
        }))
    }
}

/// NVRTC-backed integer `BitwiseNot` kernel.
#[derive(Debug)]
pub struct BitwiseNotKernel {
    runtime: Arc<CudaRuntime>,
}

impl BitwiseNotKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep BitwiseNot: expected 1 input and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let x = &inputs[0];
        let Some((_, suffix)) = integer_type_suffix(x.dtype) else {
            return Err(not_implemented(format!(
                "BitwiseNot: operand dtype {:?} not supported on CUDA EP",
                x.dtype
            )));
        };
        if outputs[0].dtype != x.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep BitwiseNot: output dtype {:?} must equal input dtype {:?}",
                outputs[0].dtype, x.dtype
            )));
        }
        if !x.is_contiguous() || !outputs[0].is_contiguous() {
            return Err(not_implemented(
                "BitwiseNot with a non-contiguous (strided) operand; \
                 insert an explicit copy to materialise it before the op",
            ));
        }
        if outputs[0].numel() != x.numel() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep BitwiseNot: output has {} elements, expected {} (same shape as input)",
                outputs[0].numel(),
                x.numel()
            )));
        }

        let n = x.numel();
        let n_u64 = count_u64("BitwiseNot", n)?;
        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let entry = format!("bnot_{suffix}");
        let func = self
            .runtime
            .nvrtc_function(BITWISE_NOT_MODULE, BITWISE_NOT_SRC, &entry)?;
        let cfg = LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder.arg(&x_ptr).arg(&y_ptr).arg(&n_u64);
        // SAFETY: `func` is the compiled complement entry; the (const T*, T*,
        // unsigned long long) argument list matches its signature; both pointers
        // are live device allocations of `n` elements of the validated dtype.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        if self.runtime.is_capturing()? {
            return Ok(());
        }
        self.runtime.synchronize()
    }
}

impl Kernel for BitwiseNotKernel {
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

/// Claim-time rejection reason for the bitwise/shift dtype contract, so the
/// partitioner routes an unsupported dtype to another EP instead of failing at
/// kernel time.
pub(crate) fn unsupported_reason(op: &str, input_dtypes: &[DataType]) -> Option<String> {
    let shift = op == "BitShift";
    let unary = op == "BitwiseNot";
    let Some(&a) = input_dtypes.first() else {
        return Some(format!("{op}: missing operand dtype for CUDA EP"));
    };
    let supported = if shift {
        unsigned_suffix(a).is_some()
    } else {
        integer_type_suffix(a).is_some()
    };
    if !supported {
        return Some(format!(
            "{op}: operand dtype {a:?} not supported on CUDA EP"
        ));
    }
    if !unary {
        let Some(&b) = input_dtypes.get(1) else {
            return Some(format!("{op}: missing second operand dtype for CUDA EP"));
        };
        if a != b {
            return Some(format!(
                "{op}: operands must have the same dtype on CUDA EP (got {a:?} and {b:?})"
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitwise_binary_entry_points_present_for_every_integer_dtype() {
        for op in [
            BitwiseBinaryOp::And,
            BitwiseBinaryOp::Or,
            BitwiseBinaryOp::Xor,
        ] {
            assert!(
                BITWISE_BINARY_SRC
                    .contains(&format!("DEFINE_BITWISE({}, type, suffix,", op.stem())),
                "missing generator for {}",
                op.op_name()
            );
        }
        for suffix in ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"] {
            assert_eq!(
                BitwiseBinaryOp::And.entry_suffix_for_test(suffix),
                format!("band_{suffix}")
            );
        }
    }

    #[test]
    fn shift_entry_points_are_unsigned_only() {
        assert_eq!(
            BinaryKind::Shift(ShiftDirection::Left).entry(DataType::Uint32),
            Some("shl_u32".to_string())
        );
        assert_eq!(
            BinaryKind::Shift(ShiftDirection::Right).entry(DataType::Uint8),
            Some("shr_u8".to_string())
        );
        // Signed operands are unsupported for BitShift.
        assert_eq!(
            BinaryKind::Shift(ShiftDirection::Left).entry(DataType::Int32),
            None
        );
    }

    #[test]
    fn shift_guard_matches_cpu_checked_shift_contract() {
        // The kernel truncates the amount to u32 and compares against the width,
        // reproducing `checked_shl(amount as u32)` returning None (-> 0).
        assert!(BITSHIFT_SRC.contains("unsigned int amount = (unsigned int)b[bi];"));
        assert!(BITSHIFT_SRC.contains("(amount >= bits) ? (type)0"));
        for bits in ["8u", "16u", "32u", "64u"] {
            assert!(
                BITSHIFT_SRC.contains(&format!(", {bits})")),
                "missing width {bits}"
            );
        }
    }

    #[test]
    fn not_entry_points_present_for_every_integer_dtype() {
        for suffix in ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"] {
            assert!(
                BITWISE_NOT_SRC.contains(&format!(", {suffix})")),
                "missing BitwiseNot generator for {suffix}"
            );
        }
    }

    #[test]
    fn bitwise_binary_rejects_float_and_mismatched_dtypes() {
        assert_eq!(
            BinaryKind::Bitwise(BitwiseBinaryOp::And).entry(DataType::Float32),
            None
        );
        let reason = unsupported_reason("BitwiseAnd", &[DataType::Int32, DataType::Int64]).unwrap();
        assert!(reason.contains("same dtype"), "{reason}");
        let reason =
            unsupported_reason("BitwiseAnd", &[DataType::Float32, DataType::Float32]).unwrap();
        assert!(reason.contains("not supported"), "{reason}");
        assert_eq!(
            unsupported_reason("BitwiseAnd", &[DataType::Uint8, DataType::Uint8]),
            None
        );
    }

    #[test]
    fn bitshift_claim_requires_unsigned_operands() {
        assert!(unsupported_reason("BitShift", &[DataType::Int32, DataType::Int32]).is_some());
        assert_eq!(
            unsupported_reason("BitShift", &[DataType::Uint16, DataType::Uint16]),
            None
        );
    }

    #[test]
    fn bitwise_not_claim_is_unary() {
        assert_eq!(
            unsupported_reason("BitwiseNot", &[DataType::Int8]),
            None,
            "unary op must not require a second operand"
        );
        assert!(unsupported_reason("BitwiseNot", &[DataType::Float64]).is_some());
    }

    #[test]
    fn grid_covers_all_elements() {
        assert_eq!(grid_for(0), 1);
        assert_eq!(grid_for(BLOCK as usize + 1), 2);
        assert_eq!(grid_for(usize::MAX / 2), 65_535);
    }

    #[test]
    fn kernels_use_unsigned_64bit_indexing() {
        const LOOP: &str = "for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)";
        for src in [BITWISE_BINARY_SRC, BITSHIFT_SRC, BITWISE_NOT_SRC] {
            assert!(src.contains(LOOP), "kernel regressed to signed indexing");
            assert!(src.contains("const unsigned long long n)"));
        }
    }

    impl BitwiseBinaryOp {
        fn entry_suffix_for_test(self, suffix: &str) -> String {
            format!("{}_{suffix}", self.stem())
        }
    }
}
