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
pub mod cast;
pub mod constant;
pub mod elementwise;
pub mod expand;
pub mod fused_attention;
pub mod fused_gemm;
pub mod fused_matmul_bias;
pub mod gather;
pub mod gelu;
pub mod gemm;
pub mod layernorm;
pub mod matmul;
pub mod reduce;
pub mod relu;
pub mod reshape;
pub mod shape;
pub mod slice;
pub mod softmax;
pub mod transpose;
pub mod unsqueeze;

/// The set of ops the CPU EP implements for the Phase-1 BERT-on-CPU milestone.
pub const PHASE1_OPS: &[&str] = &[
    "MatMul",
    "Add",
    "Relu",
    "Reshape",
    "Transpose",
    "Gather",
    "LayerNormalization",
    // Elementwise binary (numpy broadcasting).
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Min",
    "Max",
    // Elementwise unary.
    "Sqrt",
    "Erf",
    "Tanh",
    "Cast",
    // Reduction / normalization.
    "ReduceMean",
    "Softmax",
    // Shape / data movement.
    "Shape",
    "Unsqueeze",
    "Expand",
    "Slice",
    "Constant",
    // GEMM.
    "Gemm",
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
    // The optimizer emits fused `LayerNormalization` in the private contrib
    // domain (`com.microsoft`); bind the same kernel there so dispatch resolves
    // the fused op by (domain, op_type). The default-domain registration above
    // still serves standard ONNX `LayerNormalization`.
    reg.register(
        OpKey::new("LayerNormalization", "com.microsoft", 1),
        Box::new(layernorm::LayerNormFactory),
    );
    // The optimizer's `MatMul + Add(bias)` fusion emits `FusedMatMulBias` in the
    // contrib domain; bind its kernel there so dispatch resolves the fused op by
    // (domain, op_type). It reuses the shared MatMul GEMM + broadcast-Add.
    reg.register(
        OpKey::new("FusedMatMulBias", "com.microsoft", 1),
        Box::new(fused_matmul_bias::FusedMatMulBiasFactory),
    );
    // The optimizer's `MatMul + Add(bias) + Relu` fusion emits `FusedGemm` in
    // the contrib domain; bind its kernel there so dispatch resolves the fused
    // op by (domain, op_type). It reuses the shared MatMul GEMM + broadcast-Add
    // + elementwise Relu.
    reg.register(
        OpKey::new("FusedGemm", "com.microsoft", 1),
        Box::new(fused_gemm::FusedGemmFactory),
    );
    // The optimizer's SDPA-core fusion (MatMul(QKᵀ) → scale → [+mask] → Softmax
    // → MatMul(·V)) emits `FusedAttention` in the contrib domain; bind its
    // kernel there so dispatch resolves the fused op by (domain, op_type). It
    // reuses the shared MatMul GEMM (twice), broadcast-Add (mask) and the
    // extracted last-axis softmax helper.
    reg.register(
        OpKey::new("FusedAttention", "com.microsoft", 1),
        Box::new(fused_attention::FusedAttentionFactory),
    );
    // The optimizer's exact-GELU fusion emits `com.microsoft::Gelu`; bind its
    // CPU kernel in the same contrib domain (there is no standard-domain `Gelu`
    // op, so it is registered only under `com.microsoft`).
    reg.register(
        OpKey::new("Gelu", "com.microsoft", 1),
        Box::new(gelu::GeluFactory),
    );
    // Elementwise binary broadcasting ops.
    reg.register(OpKey::new("Sub", "", 1), Box::new(elementwise::SubFactory));
    reg.register(OpKey::new("Mul", "", 1), Box::new(elementwise::MulFactory));
    reg.register(OpKey::new("Div", "", 1), Box::new(elementwise::DivFactory));
    reg.register(OpKey::new("Pow", "", 1), Box::new(elementwise::PowFactory));
    reg.register(OpKey::new("Min", "", 1), Box::new(elementwise::MinFactory));
    reg.register(OpKey::new("Max", "", 1), Box::new(elementwise::MaxFactory));
    // Elementwise unary ops.
    reg.register(OpKey::new("Sqrt", "", 1), Box::new(elementwise::SqrtFactory));
    reg.register(OpKey::new("Erf", "", 1), Box::new(elementwise::ErfFactory));
    reg.register(OpKey::new("Tanh", "", 1), Box::new(elementwise::TanhFactory));
    reg.register(OpKey::new("Cast", "", 1), Box::new(cast::CastFactory));
    // Reduction / normalization.
    reg.register(
        OpKey::new("ReduceMean", "", 1),
        Box::new(reduce::ReduceMeanFactory),
    );
    // Softmax: legacy coerce-to-2D at opset ≤ 12, per-axis at opset ≥ 13. The
    // provider's opset-aware lookup selects the version-correct kernel.
    reg.register(
        OpKey::new("Softmax", "", 1),
        Box::new(softmax::SoftmaxLegacyFactory),
    );
    reg.register(
        OpKey::new("Softmax", "", 13),
        Box::new(softmax::SoftmaxFactory),
    );
    // Shape / data movement.
    reg.register(OpKey::new("Shape", "", 1), Box::new(shape::ShapeFactory));
    reg.register(
        OpKey::new("Unsqueeze", "", 1),
        Box::new(unsqueeze::UnsqueezeFactory),
    );
    reg.register(OpKey::new("Expand", "", 1), Box::new(expand::ExpandFactory));
    reg.register(OpKey::new("Slice", "", 1), Box::new(slice::SliceFactory));
    reg.register(
        OpKey::new("Constant", "", 1),
        Box::new(constant::ConstantFactory),
    );
    // GEMM.
    reg.register(OpKey::new("Gemm", "", 1), Box::new(gemm::GemmFactory));
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

/// The fixed element byte-width of `dtype`. Errors for variable-width
/// ([`DataType::String`]) and sub-byte-packed (`Int4`/`Uint4`) types, which the
/// dtype-generic byte movers below cannot address one-element-at-a-time.
pub fn elem_size(dtype: DataType) -> Result<usize> {
    let size = dtype.byte_size();
    if size == 0 {
        return Err(EpError::InvalidTensorView {
            reason: format!("dtype {dtype:?} has no fixed-width byte layout"),
        });
    }
    Ok(size)
}

/// Materialize any fixed-width view into a dense, row-major byte buffer,
/// applying the view's strides and byte offset. This is the dtype-agnostic
/// counterpart to [`to_dense_f32`]: it copies raw element bytes without
/// interpreting them, so it serves the pure data-movement ops (Unsqueeze,
/// Expand, Slice, Cast source read) uniformly across dtypes.
pub fn to_dense_bytes(view: &TensorView) -> Result<Vec<u8>> {
    view.validate()?;
    let esize = elem_size(view.dtype)?;
    let n = numel(view.shape);
    let mut out = vec![0u8; n * esize];
    if n == 0 {
        return Ok(out);
    }
    // Byte origin of the element at logical index 0 (applies `byte_offset`).
    let origin = view.data_ptr::<u8>();
    let mut idx = vec![0usize; view.shape.len()];
    let mut w = 0usize;
    loop {
        let elem_off = elem_offset(view.strides, &idx);
        let byte_off = elem_off * esize as isize;
        // SAFETY: `origin` is the byte origin of a validated view; `elem_off` is
        // an in-shape element offset, so `byte_off .. byte_off + esize` lies
        // within the extent the view describes (bounds-checked against the
        // backing allocation by the EP per invariant #1). `out[w..w + esize]` is
        // a fresh, uniquely-owned buffer. The regions do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(origin.offset(byte_off), out.as_mut_ptr().add(w), esize);
        }
        w += esize;
        if !next_index(view.shape, &mut idx) {
            break;
        }
    }
    Ok(out)
}

/// Write a dense, row-major byte buffer into `out`, applying the output view's
/// strides and byte offset. `data.len()` must equal `numel(out) * elem_size`.
/// The dtype-agnostic counterpart to [`write_dense_f32`].
pub fn write_dense_bytes(out: &mut TensorMut, data: &[u8]) -> Result<()> {
    out.validate()?;
    let esize = elem_size(out.dtype)?;
    let n = numel(out.shape);
    if data.len() != n * esize {
        return Err(EpError::KernelFailed(format!(
            "output byte count {} does not match produced {}",
            n * esize,
            data.len()
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let origin = out.data_ptr_mut::<u8>();
    let strides = out.strides;
    let shape = out.shape;
    let mut idx = vec![0usize; shape.len()];
    let mut r = 0usize;
    loop {
        let elem_off = elem_offset(strides, &idx);
        let byte_off = elem_off * esize as isize;
        // SAFETY: `origin` is the byte origin of a validated output view;
        // `byte_off .. byte_off + esize` is an in-shape offset lying within the
        // extent the view describes (bounds-checked by the EP per invariant #1).
        // Each destination range is written exactly once because the row-major
        // walk visits every logical index once; source and destination buffers
        // are distinct.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().add(r), origin.offset(byte_off), esize);
        }
        r += esize;
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

        pub fn i32(shape: &[usize], data: &[i32]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Int32,
            }
        }

        pub fn bool_(shape: &[usize], data: &[bool]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let bytes = data.iter().map(|&b| b as u8).collect();
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Bool,
            }
        }

        /// A zero-filled f32 output buffer of `shape`.
        pub fn zeros_f32(shape: &[usize]) -> Self {
            let n: usize = shape.iter().product();
            Self::f32(shape, &vec![0.0; n])
        }

        /// A zero-filled output buffer of `shape` with element type `dtype`.
        pub fn zeros(dtype: DataType, shape: &[usize]) -> Self {
            let n: usize = shape.iter().product();
            let strides = compute_contiguous_strides(shape);
            let esize = dtype.byte_size();
            Self {
                bytes: vec![0u8; n * esize],
                shape: shape.to_vec(),
                strides,
                dtype,
            }
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

        pub fn to_i64(&self) -> Vec<i64> {
            self.bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }

        pub fn to_i32(&self) -> Vec<i32> {
            self.bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }

        pub fn to_bool(&self) -> Vec<bool> {
            self.bytes.iter().map(|&b| b != 0).collect()
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
        // Every Phase-1 op has at least one factory, and each resolves at a
        // modern opset. `Softmax` is registered twice (legacy v1 + per-axis
        // v13), and `LayerNormalization`, `FusedMatMulBias`, `FusedGemm`,
        // `FusedAttention` and the fused exact-GELU `Gelu` add contrib
        // (`com.microsoft`) entries, so the entry count is six more than the
        // op-name count.
        assert_eq!(reg.len(), PHASE1_OPS.len() + 6);
        for op in PHASE1_OPS {
            assert!(reg.lookup(op, "", 21).is_some(), "missing factory for {op}");
        }
        // Softmax selects legacy at opset ≤ 12 and per-axis at opset ≥ 13.
        assert!(reg.lookup("Softmax", "", 12).is_some());
        assert!(reg.lookup("Softmax", "", 13).is_some());
        assert!(reg.lookup("Conv", "", 21).is_none());
        // The fused contrib-domain LayerNormalization resolves to the same
        // kernel as the standard default-domain op.
        assert!(reg.lookup("LayerNormalization", "com.microsoft", 1).is_some());
        assert!(reg.supports("LayerNormalization", "com.microsoft"));
        assert!(reg.supports("MatMul", "ai.onnx"));
        // The `MatMul + Add` fusion's contrib op now has a CPU kernel.
        assert!(reg.supports("FusedMatMulBias", "com.microsoft"));
        // The `MatMul + Add + Relu` fusion's contrib op now has a CPU kernel.
        assert!(reg.supports("FusedGemm", "com.microsoft"));
        assert!(reg.lookup("FusedGemm", "com.microsoft", 1).is_some());
        // The exact-GELU fusion's contrib op has a CPU kernel (contrib-only).
        assert!(reg.supports("Gelu", "com.microsoft"));
        assert!(reg.lookup("Gelu", "com.microsoft", 1).is_some());
        assert!(reg.lookup("Gelu", "", 21).is_none());
    }

    #[test]
    fn dense_read_stays_in_bounds() {
        let a = Owned::f32(&[3, 2], &[1., 4., 2., 5., 3., 6.]);
        let v = a.view();
        view_in_bounds(v.shape, v.strides, v.byte_offset, 4, a.bytes.len()).unwrap();
    }
}
