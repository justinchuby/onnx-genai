//! CPU kernels for the Phase-1 BERT-on-CPU correctness milestone (`docs/ORT2.md`
//! §4.4). One [`Kernel`] per ONNX op, keyed purely by op type — there are **no**
//! model-specific shapes or names anywhere in this crate; BERT is only the
//! validation target.
//!
//! ## Pure-Rust reference kernels (architecture decision)
//!
//! These are straightforward, **correct** pure-Rust kernels — naive loops, no
//! FFI, no `cc`/oneDNN build dependency (oneDNN is not installed on the build
//! host, and the Phase-1 exit bar is correctness, not throughput). Every kernel
//! sits behind the [`Kernel`] trait, so a Phase-1.5 perf pass can swap in a
//! blocked/SIMD GEMM (oneDNN, or a Rust BLAS such as `matrixmultiply`/`gemm`)
//! for the hot kernels **without touching the EP contract or the session**. The
//! seam is [`Kernel`] itself; see [`matmul`] for the specific hot spot.
//!
//! ## Strided inputs
//!
//! Kernels accept non-contiguous inputs by reading through
//! [`to_dense_f32`]/[`to_dense_i64`], which materialize a view (applying its
//! strides and byte offset) into a dense row-major buffer. This keeps the
//! per-kernel `unsafe` surface to the two element accessors in this module.

use onnx_runtime_ep_api::{EpError, OpKey, OpRegistry, Result, TensorMut, TensorView};
use onnx_runtime_ir::DataType;

use crate::strided::{elem_offset, next_index, numel};

pub mod add;
pub mod gather;
pub mod layernorm;
pub mod matmul;
pub mod relu;
pub mod reshape;
pub mod transpose;

/// The set of ops the CPU EP implements for the Phase-1 BERT-on-CPU milestone.
pub const PHASE1_OPS: &[&str] = &[
    "MatMul",
    "Add",
    "Relu",
    "Reshape",
    "Transpose",
    "Gather",
    "LayerNormalization",
];

/// Whether `op_type` is one of the Phase-1 ops the CPU EP can run.
pub fn is_phase1_op(op_type: &str) -> bool {
    PHASE1_OPS.contains(&op_type)
}

/// Build an [`OpRegistry`] populated with every Phase-1 CPU kernel factory.
///
/// The provider consults this to instantiate kernels, and Track D (session) can
/// reuse the same registry for its own placement/lookup. All ops are registered
/// under the default domain (`""`) at `since_version` 1; the registry's
/// `lookup` picks the highest applicable version, so future opset-specialized
/// kernels can be added alongside these.
pub fn build_cpu_registry() -> OpRegistry {
    let mut reg = OpRegistry::new();
    reg.register(
        OpKey::new("MatMul", "", 1),
        Box::new(matmul::MatMulFactory),
    );
    reg.register(OpKey::new("Add", "", 1), Box::new(add::AddFactory));
    reg.register(OpKey::new("Relu", "", 1), Box::new(relu::ReluFactory));
    reg.register(
        OpKey::new("Reshape", "", 1),
        Box::new(reshape::ReshapeFactory),
    );
    reg.register(
        OpKey::new("Transpose", "", 1),
        Box::new(transpose::TransposeFactory),
    );
    reg.register(OpKey::new("Gather", "", 1), Box::new(gather::GatherFactory));
    reg.register(
        OpKey::new("LayerNormalization", "", 1),
        Box::new(layernorm::LayerNormFactory),
    );
    reg
}

// ---------------------------------------------------------------------------
// Shared view accessors — the only `unsafe` in the kernel layer.
// ---------------------------------------------------------------------------

/// Materialize an `f32` view into a dense, row-major `Vec<f32>`, applying the
/// view's strides and byte offset. Rejects non-`Float32` views.
pub fn to_dense_f32(view: &TensorView) -> Result<Vec<f32>> {
    view.validate()?;
    require_dtype(view.dtype, DataType::Float32, "f32 kernel input")?;
    let n = numel(view.shape);
    let origin = view.data_ptr::<f32>();
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return Ok(out);
    }
    let mut idx = vec![0usize; view.shape.len()];
    loop {
        let off = elem_offset(view.strides, &idx);
        // SAFETY: `origin` is the element origin of a validated view; `off` is
        // an in-shape element offset (each index component is `< shape[d]`), so
        // the address lies within the range the view describes. The owning EP
        // has already checked that range against the backing allocation via
        // `strided::view_in_bounds` (ep-api safety invariant #1). We never read
        // past the addressed extent, and `f32` has no invalid bit patterns.
        out.push(unsafe { *origin.offset(off) });
        if !next_index(view.shape, &mut idx) {
            break;
        }
    }
    Ok(out)
}

/// Materialize an integer index view (`Int64` or `Int32`) into a dense
/// `Vec<i64>`. Used for `Gather` indices.
pub fn to_dense_i64(view: &TensorView) -> Result<Vec<i64>> {
    view.validate()?;
    let n = numel(view.shape);
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return Ok(out);
    }
    let mut idx = vec![0usize; view.shape.len()];
    match view.dtype {
        DataType::Int64 => {
            let origin = view.data_ptr::<i64>();
            loop {
                let off = elem_offset(view.strides, &idx);
                // SAFETY: see `to_dense_f32` — in-shape offset over a validated,
                // bounds-checked view; `i64` has no invalid bit patterns.
                out.push(unsafe { *origin.offset(off) });
                if !next_index(view.shape, &mut idx) {
                    break;
                }
            }
        }
        DataType::Int32 => {
            let origin = view.data_ptr::<i32>();
            loop {
                let off = elem_offset(view.strides, &idx);
                // SAFETY: as above, for a 4-byte element type.
                out.push(unsafe { *origin.offset(off) } as i64);
                if !next_index(view.shape, &mut idx) {
                    break;
                }
            }
        }
        other => {
            return Err(EpError::InvalidTensorView {
                reason: format!("index tensor must be Int64 or Int32, got {other:?}"),
            });
        }
    }
    Ok(out)
}

/// Write a dense, row-major `f32` slice into `out`, applying the output view's
/// strides and byte offset. `data.len()` must equal the output element count.
pub fn write_dense_f32(out: &mut TensorMut, data: &[f32]) -> Result<()> {
    out.validate()?;
    require_dtype(out.dtype, DataType::Float32, "f32 kernel output")?;
    let n = numel(out.shape);
    if data.len() != n {
        return Err(EpError::KernelFailed(format!(
            "output element count {n} does not match produced {}",
            data.len()
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let origin = out.data_ptr_mut::<f32>();
    let strides = out.strides;
    let shape = out.shape;
    let mut idx = vec![0usize; shape.len()];
    let mut i = 0usize;
    loop {
        let off = elem_offset(strides, &idx);
        // SAFETY: `origin` is the element origin of a validated output view;
        // `off` is an in-shape offset, so it lies within the extent the view
        // describes (bounds-checked against the backing allocation by the EP
        // per invariant #1). Each address is written exactly once because the
        // row-major walk visits every logical index once.
        unsafe {
            *origin.offset(off) = data[i];
        }
        i += 1;
        if !next_index(shape, &mut idx) {
            break;
        }
    }
    Ok(())
}

/// Error out unless `got == want`.
fn require_dtype(got: DataType, want: DataType, ctx: &str) -> Result<()> {
    if got != want {
        return Err(EpError::InvalidTensorView {
            reason: format!("{ctx} requires {want:?}, got {got:?}"),
        });
    }
    Ok(())
}

/// Validate the arity of a kernel's input/output slices.
fn check_arity(
    op: &str,
    inputs: &[TensorView],
    outputs: &[TensorMut],
    min_inputs: usize,
    max_inputs: usize,
    outputs_wanted: usize,
) -> Result<()> {
    if inputs.len() < min_inputs || inputs.len() > max_inputs {
        return Err(EpError::KernelFailed(format!(
            "{op}: expected {min_inputs}..={max_inputs} inputs, got {}",
            inputs.len()
        )));
    }
    if outputs.len() < outputs_wanted {
        return Err(EpError::KernelFailed(format!(
            "{op}: expected at least {outputs_wanted} output(s), got {}",
            outputs.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod testutil {
    //! Helpers to build owning-buffer-backed views for kernel unit tests.

    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, TensorMut, TensorView};
    use onnx_runtime_ir::{compute_contiguous_strides, DataType, DeviceId};

    /// A dense f32 buffer plus the shape/stride metadata a view needs.
    pub struct Owned {
        pub bytes: Vec<u8>,
        pub shape: Vec<usize>,
        pub strides: Vec<i64>,
        pub dtype: DataType,
    }

    impl Owned {
        pub fn f32(shape: &[usize], data: &[f32]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Float32,
            }
        }

        pub fn i64(shape: &[usize], data: &[i64]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(data.len() * 8);
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Int64,
            }
        }

        /// A zero-filled f32 output buffer of `shape`.
        pub fn zeros_f32(shape: &[usize]) -> Self {
            let n: usize = shape.iter().product();
            Self::f32(shape, &vec![0.0; n])
        }

        /// Override strides/shape to expose the same bytes as a strided view
        /// (e.g. a transpose without copying).
        pub fn with_view(mut self, shape: &[usize], strides: &[i64]) -> Self {
            self.shape = shape.to_vec();
            self.strides = strides.to_vec();
            self
        }

        pub fn view(&self) -> TensorView<'_> {
            TensorView::new(
                DevicePtr(self.bytes.as_ptr() as *const std::ffi::c_void),
                self.dtype,
                &self.shape,
                &self.strides,
                DeviceId::cpu(),
            )
        }

        pub fn view_mut(&mut self) -> TensorMut<'_> {
            TensorMut::new(
                DevicePtrMut(self.bytes.as_mut_ptr() as *mut std::ffi::c_void),
                self.dtype,
                &self.shape,
                &self.strides,
                DeviceId::cpu(),
            )
        }

        pub fn to_f32(&self) -> Vec<f32> {
            self.bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strided::view_in_bounds;
    use testutil::Owned;

    #[test]
    fn dense_roundtrip_contiguous() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let v = a.view();
        assert_eq!(to_dense_f32(&v).unwrap(), vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn dense_reads_transposed_view() {
        // Backing [2,3] row-major; expose as transposed [3,2] with strides [1,3].
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]).with_view(&[3, 2], &[1, 3]);
        let v = a.view();
        // Transpose of [[1,2,3],[4,5,6]] is [[1,4],[2,5],[3,6]].
        assert_eq!(to_dense_f32(&v).unwrap(), vec![1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn registry_has_all_phase1_ops() {
        let reg = build_cpu_registry();
        assert_eq!(reg.len(), PHASE1_OPS.len());
        for op in PHASE1_OPS {
            assert!(reg.lookup(op, "", 21).is_some(), "missing factory for {op}");
        }
        assert!(reg.lookup("Conv", "", 21).is_none());
    }

    #[test]
    fn dense_read_stays_in_bounds() {
        let a = Owned::f32(&[3, 2], &[1., 4., 2., 5., 3., 6.]);
        let v = a.view();
        view_in_bounds(v.shape, v.strides, v.byte_offset, 4, a.bytes.len()).unwrap();
    }
}
