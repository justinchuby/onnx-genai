//! `pkg.nxrt::KvCacheCapacityAppend`: the CUDA-graph-capture-safe replacement
//! for a decomposed-attention KV-cache-growth `Concat`.
//!
//! A plain `Concat(past, current) -> present` grows its declared shape and its
//! own kernel's launch geometry (grid size, per-step self-copy of the whole
//! valid prefix) every decode step, which is why CUDA-graph replay cannot
//! reuse a captured `Concat` launch across steps (see
//! `onnx-runtime-session::executor::geometry`'s S3 capacity-emission
//! analysis for the full derivation, and `is_kv_cache_growth_concat`'s doc
//! comment for why the underlying present/past device buffer is *already*
//! bound at one fixed physical-capacity address regardless of which op reads
//! it — that part was never the problem).
//!
//! This op instead writes ONLY `current`'s rows into a frozen
//! `[B, H, capacity, D]` buffer in place, at the destination row given by
//! `position_ids` — a genuine, host-refreshed-every-step graph input whose
//! *value* (not shape) carries the one thing that legitimately varies
//! decode-to-decode. Reading it from device memory at execute time, instead
//! of baking a per-step offset into the launch, is what keeps every launch
//! parameter (grid/block dims, pointers, byte counts) identical across every
//! capture replay: `past`'s exposed shape is pinned to physical capacity by
//! `geometry::kernel_input_uses_physical_capacity`'s `KvCacheCapacityAppend`
//! arm, and `current`/`position_ids` keep the same per-step shape as any
//! other ordinary decode input (`[B, H, S, D]` / `[B, S]`, S fixed within a
//! capture epoch — 1 in the steady-state decode loop).
//!
//! Inputs: `[past, current, position_ids]`.
//! Output: `present`, aliased in-place to `past` by the executor's existing
//! present==past persistent IO binding (unconditional for any KV-cache-growth
//! pair, see `is_kv_cache_growth_concat`'s doc comment) — this kernel never
//! needs to copy `past`'s already-correct existing rows, only write
//! `current`'s new ones.
//!
//! Bounds-checked: a `position_ids` value outside `[0, capacity)` is a
//! capacity overflow, checked by whichever mechanism the execution mode
//! actually allows. In plain eager execution (no capture in progress, no
//! deferred host sync), `position_ids` is downloaded and validated
//! synchronously up front — mirroring `rotary_embedding.rs`'s
//! `position_ids`-bounds guard exactly — and the call hard-errors before the
//! kernel ever launches. During capture (or overlap-driven deferred sync,
//! where a synchronous host download is either illegal or defeats the
//! overlap), that host check cannot run; the kernel instead skips the
//! out-of-range row and latches the shared device capture-error word, so the
//! executor's existing post-replay `check_device_capture_error` poll surfaces
//! the fault before the corrupted step's output is consumed.

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::Mutex;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{DataType, Node};

use crate::error::not_implemented;
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;

/// Bit latched into the shared device capture-error word when a
/// `position_ids` row falls outside `[0, capacity)`. Capture-error bits are
/// not a global registry (existing kernels reuse values freely, e.g. `256` is
/// shared by `indexing.rs` and `rotary_embedding.rs`); detection is bit-agnostic
/// (`capture_error != 0` fails the step), so this only needs to be nonzero.
pub const KV_CAPACITY_APPEND_CAPTURE_ERROR_POSITION: u32 = 16_384;

const MODULE: &str = "kv_cache_capacity_append_v1";
const SOURCE: &str = r#"
extern "C" __global__ void kv_capacity_append_bytes(
    const unsigned char* current, unsigned char* present,
    const long long* position_ids,
    unsigned long long heads, unsigned long long capacity,
    unsigned long long current_len, unsigned long long head_dim,
    int elem_bytes, unsigned long long elements,
    unsigned int* capture_error) {
  for (unsigned long long e = blockIdx.x * blockDim.x + threadIdx.x; e < elements;
       e += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long d = e % head_dim;
    unsigned long long rem = e / head_dim;
    unsigned long long s = rem % current_len;
    rem /= current_len;
    unsigned long long h = rem % heads;
    unsigned long long b = rem / heads;

    const long long pos = position_ids[b * current_len + s];
    if (pos < 0 || (unsigned long long)pos >= capacity) {
      if (capture_error) atomicOr(capture_error, 16384u);
      continue;
    }
    const unsigned long long dst =
        ((b * heads + h) * capacity + (unsigned long long)pos) * head_dim + d;
    for (int byte = 0; byte < elem_bytes; ++byte)
      present[dst * elem_bytes + byte] = current[e * elem_bytes + byte];
  }
}
"#;

fn grid(elements: usize) -> u32 {
    (elements as u64).div_ceil(BLOCK as u64).clamp(1, 65_535) as u32
}

fn elem_bytes(dtype: DataType) -> Result<usize> {
    let bytes = dtype.byte_size();
    if bytes == 0 {
        Err(not_implemented(format!(
            "cuda_ep KvCacheCapacityAppend for packed or variable-width dtype {dtype:?}"
        )))
    } else {
        Ok(bytes)
    }
}

/// Claim-time validation: declines every structural precondition
/// [`KvCacheCapacityAppendKernel::execute`] would otherwise only discover at
/// *run* time (rank, dtype, and — where statically known — the
/// batch/heads/head_dim cross-shape agreement between `past` and `current`).
/// `rewrite_kv_capacity_appends` treats `ep.supports_op` as its sole
/// per-candidate safety gate for a structurally-eligible-but-nonstandard
/// KV-cache layout (e.g. not `[B, H, S, D]`/`[B, S]`); without this check
/// such a candidate would be rewritten anyway and then hard-crash the
/// session at execution time instead of being left as an ordinary `Concat`
/// — reintroducing exactly the class of crash #1838 exists to eliminate.
/// Dynamic (symbolic) dims are not rejected here: only mismatches that are
/// staticaly *provable* at claim time are declined; a shape that turns out
/// incompatible only once resolved to concrete runtime sizes still falls
/// back to `execute`'s own hard error, unchanged from before this function
/// existed.
pub(crate) fn unsupported_reason(
    shapes: &[onnx_runtime_ir::Shape],
    input_dtypes: &[DataType],
) -> Option<String> {
    let dtype_at = |index: usize| {
        input_dtypes
            .get(index)
            .copied()
            .unwrap_or(DataType::Undefined)
    };
    let shape_at = |index: usize| shapes.get(index).map(Vec::as_slice).unwrap_or(&[]);

    let position_ids_dtype = dtype_at(2);
    if position_ids_dtype != DataType::Undefined && position_ids_dtype != DataType::Int64 {
        return Some(format!(
            "KvCacheCapacityAppend: position_ids must be Int64 on CUDA, got {position_ids_dtype:?}"
        ));
    }
    let past_dtype = dtype_at(0);
    let current_dtype = dtype_at(1);
    if past_dtype != DataType::Undefined
        && current_dtype != DataType::Undefined
        && past_dtype != current_dtype
    {
        return Some(format!(
            "KvCacheCapacityAppend: past and current dtypes must match on CUDA, got \
             {past_dtype:?} and {current_dtype:?}"
        ));
    }

    let past_shape = shape_at(0);
    let current_shape = shape_at(1);
    let position_ids_shape = shape_at(2);
    if !past_shape.is_empty() && past_shape.len() != 4 {
        return Some(format!(
            "KvCacheCapacityAppend: past must be rank 4 [batch, heads, capacity, head_dim] on \
             CUDA, got rank {}",
            past_shape.len()
        ));
    }
    if !current_shape.is_empty() && current_shape.len() != 4 {
        return Some(format!(
            "KvCacheCapacityAppend: current must be rank 4 [batch, heads, current_len, \
             head_dim] on CUDA, got rank {}",
            current_shape.len()
        ));
    }
    if !position_ids_shape.is_empty() && position_ids_shape.len() != 2 {
        return Some(format!(
            "KvCacheCapacityAppend: position_ids must be rank 2 [batch, current_len] on CUDA, \
             got rank {}",
            position_ids_shape.len()
        ));
    }
    if past_shape.len() == 4 && current_shape.len() == 4 {
        for (axis, label) in [(0, "batch"), (1, "heads")] {
            if let (Some(past_dim), Some(current_dim)) = (
                past_shape[axis].as_static(),
                current_shape[axis].as_static(),
            ) && past_dim != current_dim
            {
                return Some(format!(
                    "KvCacheCapacityAppend: past and current {label} must match on CUDA \
                     (past={past_dim}, current={current_dim})"
                ));
            }
        }
        if let (Some(past_dim), Some(current_dim)) =
            (past_shape[3].as_static(), current_shape[3].as_static())
            && past_dim != current_dim
        {
            return Some(format!(
                "KvCacheCapacityAppend: past and current head_dim must match on CUDA \
                 (past={past_dim}, current={current_dim})"
            ));
        }
    }
    None
}

pub struct KvCacheCapacityAppendFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for KvCacheCapacityAppendFactory {
    fn create(&self, _node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(KvCacheCapacityAppendKernel {
            runtime: self.runtime.clone(),
            fixed_input_shapes: (input_shapes.len() == 3
                && input_shapes
                    .iter()
                    .all(|shape| shape.len() == 4 || shape.len() == 2))
            .then(|| input_shapes.to_vec()),
            warmed_signature: Mutex::new(None),
        }))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CaptureSignature {
    past_shape: Vec<usize>,
    current_shape: Vec<usize>,
    position_ids_shape: Vec<usize>,
    dtype: DataType,
}

struct KvCacheCapacityAppendKernel {
    runtime: Arc<CudaRuntime>,
    /// Every input's shape as reported at kernel-creation time, when all three
    /// have rank 4 (past/current) or rank 2 (`position_ids`) — `None` for a
    /// malformed node the rewrite never produces, forcing capture off below.
    fixed_input_shapes: Option<Vec<Vec<usize>>>,
    /// Shape/dtype signature warmed by the most recent EAGER (non-capturing)
    /// execution; capture is only ever declared `Supported` once this exists
    /// and matches, mirroring `ConcatKernel`/`ScatterNdKernel`'s pattern: a
    /// captured launch's parameters are frozen at the point of capture, so a
    /// step whose shapes ever differ from the warmed signature must not be
    /// allowed to reuse that replay.
    warmed_signature: Mutex<Option<CaptureSignature>>,
}

impl Kernel for KvCacheCapacityAppendKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 3 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep KvCacheCapacityAppend: expected 3 inputs (past, current, position_ids) \
                 and 1 output"
                    .into(),
            ));
        }
        let past = &inputs[0];
        let current = &inputs[1];
        let position_ids = &inputs[2];
        let present = &mut outputs[0];

        if position_ids.dtype != DataType::Int64 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep KvCacheCapacityAppend: position_ids must be Int64, got {:?}",
                position_ids.dtype
            )));
        }
        if current.dtype != past.dtype || present.dtype != past.dtype {
            return Err(EpError::KernelFailed(
                "cuda_ep KvCacheCapacityAppend: past, current, and present dtypes must match"
                    .into(),
            ));
        }
        if past.shape.len() != 4 || current.shape.len() != 4 || position_ids.shape.len() != 2 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep KvCacheCapacityAppend: expected past/current rank 4 and position_ids \
                 rank 2, got {:?}/{:?}/{:?}",
                past.shape, current.shape, position_ids.shape
            )));
        }
        if !past.is_contiguous() || !current.is_contiguous() || !position_ids.is_contiguous() {
            return Err(not_implemented(
                "cuda_ep KvCacheCapacityAppend with non-contiguous inputs",
            ));
        }
        let [batch, heads, capacity, head_dim] =
            [past.shape[0], past.shape[1], past.shape[2], past.shape[3]];
        if current.shape[0] != batch || current.shape[1] != heads || current.shape[3] != head_dim {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep KvCacheCapacityAppend: current shape {:?} is incompatible with past \
                 capacity shape {:?} (batch/heads/head_dim must match)",
                current.shape, past.shape
            )));
        }
        let current_len = current.shape[2];
        if position_ids.shape[0] != batch || position_ids.shape[1] != current_len {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep KvCacheCapacityAppend: position_ids shape {:?} must be [batch, \
                 current_len] = [{batch}, {current_len}]",
                position_ids.shape
            )));
        }
        if present.shape != past.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep KvCacheCapacityAppend: present shape {:?} must equal past's physical \
                 capacity shape {:?} (present is an in-place alias of past, never a growing \
                 tensor)",
                present.shape, past.shape
            )));
        }

        let capturing = self.runtime.is_capturing()?;
        let signature = CaptureSignature {
            past_shape: past.shape.to_vec(),
            current_shape: current.shape.to_vec(),
            position_ids_shape: position_ids.shape.to_vec(),
            dtype: past.dtype,
        };
        let mut warmed_signature = self.warmed_signature.lock().map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep KvCacheCapacityAppend: capture signature lock was poisoned".into(),
            )
        })?;
        if capturing && warmed_signature.as_ref() != Some(&signature) {
            return Err(EpError::KernelFailed(
                "cuda_ep KvCacheCapacityAppend: shape or dtype changed during CUDA graph \
                 capture; warm the exact signature first"
                    .into(),
            ));
        }
        // `present` must be `past` aliased in place — the executor's persistent
        // present==past IO binding is what guarantees this identity; assert it
        // rather than silently computing into the wrong buffer if some caller
        // ever exercises this kernel outside that binding contract. Compared as
        // fully offset-resolved device addresses (not raw allocation base
        // pointers), matching how the rest of the codebase resolves aliasing
        // (e.g. `standard_attention.rs`'s present/past-key alias check) — two
        // tensors can share one allocation while being bound at different
        // `byte_offset`s, which a base-pointer-only comparison would miss.
        if !std::ptr::eq(
            present.data_ptr_mut::<u8>() as *const c_void,
            past.data_ptr::<u8>() as *const c_void,
        ) {
            return Err(EpError::KernelFailed(
                "cuda_ep KvCacheCapacityAppend: present output must alias past's device buffer \
                 in place; the executor's present==past persistent KV binding was not applied"
                    .into(),
            ));
        }

        // Eager (non-capturing) bounds check: mirrors `rotary_embedding.rs`'s
        // `position_ids`-bounds guard exactly, including its scope restriction
        // to `!capturing && !eager_sync_deferred`. During capture (or when a
        // synchronous host download is deferred for overlap), a synchronous
        // `dtoh` here would either be illegal (CUDA graph capture forbids most
        // synchronous host operations) or defeat the overlap this runtime mode
        // exists for; those paths instead rely purely on the device-side
        // `capture_error` latch below, which the executor's post-replay
        // `check_device_capture_error` poll surfaces. Without this eager check,
        // an out-of-range `position_ids` value hit only during a plain eager
        // call (e.g. the very first warm-up step, before capture ever engages)
        // would otherwise be silently skipped by the kernel with `Ok(())`
        // returned and no observable signal at all.
        if !capturing && !self.runtime.eager_sync_deferred() {
            let mut host_positions = vec![0u8; position_ids.numel() * std::mem::size_of::<i64>()];
            // SAFETY: position_ids is contiguous (checked above) and the host
            // buffer has its exact byte size.
            unsafe {
                self.runtime.dtoh(
                    &mut host_positions,
                    cuptr(position_ids.data_ptr::<u8>() as *const c_void),
                )?
            };
            if host_positions.chunks_exact(8).any(|bytes| {
                let position = i64::from_ne_bytes(bytes.try_into().unwrap());
                position < 0 || position as usize >= capacity
            }) {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep KvCacheCapacityAppend: position_ids contain a value outside the \
                     [0, {capacity}) capacity range"
                )));
            }
        }

        let elements = batch
            .checked_mul(heads)
            .and_then(|value| value.checked_mul(current_len))
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep KvCacheCapacityAppend: element count overflow".into(),
                )
            })?;
        if elements != 0 {
            let elem_bytes = elem_bytes(past.dtype)? as i32;
            let current_ptr = cuptr(current.data_ptr::<u8>() as *const c_void);
            let present_ptr = cuptr(present.data_ptr_mut::<u8>() as *const c_void);
            let position_ids_ptr = cuptr(position_ids.data_ptr::<u8>() as *const c_void);
            let heads_u64 = heads as u64;
            let capacity_u64 = capacity as u64;
            let current_len_u64 = current_len as u64;
            let head_dim_u64 = head_dim as u64;
            let elements_u64 = elements as u64;
            let capture_error = if capturing || self.runtime.eager_sync_deferred() {
                self.runtime.capture_error_ptr()
            } else {
                0
            };
            let func = self
                .runtime
                .nvrtc_function(MODULE, SOURCE, "kv_capacity_append_bytes")?;
            let mut builder = self.runtime.stream().launch_builder(&func);
            builder
                .arg(&current_ptr)
                .arg(&present_ptr)
                .arg(&position_ids_ptr)
                .arg(&heads_u64)
                .arg(&capacity_u64)
                .arg(&current_len_u64)
                .arg(&head_dim_u64)
                .arg(&elem_bytes)
                .arg(&elements_u64)
                .arg(&capture_error);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (grid(elements), 1, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|error| crate::error::driver_err("launch kv_capacity_append_bytes", error))?;
        }
        if !capturing {
            *warmed_signature = Some(signature);
            self.runtime.synchronize()?;
        }
        Ok(())
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }

    fn capture_support(&self) -> CaptureSupport {
        if self.fixed_input_shapes.is_none() {
            return CaptureSupport::unsupported(
                "KvCacheCapacityAppend requires past/current rank 4 and position_ids rank 2",
            );
        }
        match self.warmed_signature.lock() {
            Ok(signature) if signature.is_some() => CaptureSupport::Supported,
            Ok(_) => CaptureSupport::unsupported(
                "KvCacheCapacityAppend must warm its exact shape/dtype signature before capture",
            ),
            Err(_) => CaptureSupport::unsupported(
                "KvCacheCapacityAppend capture signature lock was poisoned",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_clamps_to_cuda_max_and_never_zero() {
        assert_eq!(grid(0), 1);
        assert_eq!(grid(1), 1);
        assert_eq!(grid(BLOCK as usize), 1);
        assert_eq!(grid(BLOCK as usize + 1), 2);
        assert_eq!(grid(usize::MAX), 65_535);
    }

    #[test]
    fn elem_bytes_rejects_packed_dtypes() {
        assert_eq!(elem_bytes(DataType::Float32).unwrap(), 4);
        assert_eq!(elem_bytes(DataType::Float16).unwrap(), 2);
        assert!(elem_bytes(DataType::Int4).is_err());
    }

    use onnx_runtime_ir::static_shape;

    fn shapes4x4x2(
        past: [usize; 4],
        current: [usize; 4],
        position_ids: [usize; 2],
    ) -> Vec<Vec<onnx_runtime_ir::Dim>> {
        vec![
            static_shape(past),
            static_shape(current),
            static_shape(position_ids),
        ]
    }

    #[test]
    fn unsupported_reason_accepts_matching_static_shapes_and_dtypes() {
        let shapes = shapes4x4x2([1, 2, 4, 8], [1, 2, 1, 8], [1, 1]);
        let dtypes = [DataType::Float32, DataType::Float32, DataType::Int64];
        assert_eq!(unsupported_reason(&shapes, &dtypes), None);
    }

    #[test]
    fn unsupported_reason_rejects_non_int64_position_ids() {
        let shapes = shapes4x4x2([1, 2, 4, 8], [1, 2, 1, 8], [1, 1]);
        let dtypes = [DataType::Float32, DataType::Float32, DataType::Int32];
        assert!(unsupported_reason(&shapes, &dtypes).is_some());
    }

    #[test]
    fn unsupported_reason_rejects_mismatched_past_current_dtype() {
        let shapes = shapes4x4x2([1, 2, 4, 8], [1, 2, 1, 8], [1, 1]);
        let dtypes = [DataType::Float32, DataType::Float16, DataType::Int64];
        assert!(unsupported_reason(&shapes, &dtypes).is_some());
    }

    #[test]
    fn unsupported_reason_rejects_non_rank4_past() {
        let shapes = vec![
            static_shape([2, 4, 8]),
            static_shape([1, 2, 1, 8]),
            static_shape([1, 1]),
        ];
        let dtypes = [DataType::Float32, DataType::Float32, DataType::Int64];
        assert!(unsupported_reason(&shapes, &dtypes).is_some());
    }

    #[test]
    fn unsupported_reason_rejects_non_rank2_position_ids() {
        let mut shapes = shapes4x4x2([1, 2, 4, 8], [1, 2, 1, 8], [1, 1]);
        shapes[2] = static_shape([1, 1, 1]);
        let dtypes = [DataType::Float32, DataType::Float32, DataType::Int64];
        assert!(unsupported_reason(&shapes, &dtypes).is_some());
    }

    #[test]
    fn unsupported_reason_rejects_static_batch_heads_head_dim_mismatch() {
        // heads mismatch: past has 2, current has 3.
        let shapes = shapes4x4x2([1, 2, 4, 8], [1, 3, 1, 8], [1, 1]);
        let dtypes = [DataType::Float32, DataType::Float32, DataType::Int64];
        assert!(unsupported_reason(&shapes, &dtypes).is_some());
    }

    #[test]
    fn unsupported_reason_does_not_reject_symbolic_dims_it_cannot_prove_mismatched() {
        use onnx_runtime_ir::{Dim, SymbolId};
        // batch is symbolic on both sides -- nothing statically provable, so
        // this must not be declined merely because the dims are dynamic.
        let symbolic_batch = Dim::Symbolic(SymbolId(0));
        let past = vec![
            symbolic_batch,
            Dim::Static(2),
            Dim::Static(4),
            Dim::Static(8),
        ];
        let current = vec![
            symbolic_batch,
            Dim::Static(2),
            Dim::Static(1),
            Dim::Static(8),
        ];
        let position_ids = static_shape([1, 1]);
        let shapes = vec![past, current, position_ids];
        let dtypes = [DataType::Float32, DataType::Float32, DataType::Int64];
        assert_eq!(unsupported_reason(&shapes, &dtypes), None);
    }

    #[test]
    fn unsupported_reason_tolerates_empty_shape_or_dtype_metadata() {
        // A caller with no shape/dtype metadata at all (empty slices) must not
        // be declined -- claim-time validation only rejects *provable*
        // mismatches, never absence of information.
        assert_eq!(unsupported_reason(&[], &[]), None);
    }
}
