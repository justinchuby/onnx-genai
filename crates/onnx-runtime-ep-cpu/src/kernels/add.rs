//! `Add`: elementwise addition with numpy-style broadcasting, generic over the
//! ONNX numeric dtypes via the shared [`crate::dtype`] mechanism (`docs/architecture/ORT2.md`
//! §4.4).

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node, compute_contiguous_strides};

use super::check_arity;
use crate::dispatch_arith;
use crate::dtype::{ComputeDomain, NumericElem, to_dense, write_dense};
use crate::strided::{next_index, numel};

/// Stateless broadcasting Add kernel (dtype-generic).
pub struct AddKernel;

/// Factory for [`AddKernel`] (no attributes).
pub struct AddFactory;

impl KernelFactory for AddFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(AddKernel))
    }
}

/// Effective stride of each `out_shape` axis into a dense row-major `src_shape`
/// buffer, with 0 on every broadcast axis.
///
/// Factored out of [`broadcast_apply`] so kernels that fuse a broadcast into a
/// larger parallel pass — rather than walking the output element by element —
/// share exactly the same numpy right-alignment and compatibility checks.
pub fn broadcast_effective_strides(src_shape: &[usize], out_shape: &[usize]) -> Result<Vec<i64>> {
    let out_rank = out_shape.len();
    let src_strides = compute_contiguous_strides(src_shape);
    let mut eff = vec![0i64; out_rank];
    for axis in 0..out_rank {
        // Corresponding axis in src (right-aligned); absent => broadcast.
        let src_axis = axis as isize - (out_rank as isize - src_shape.len() as isize);
        if src_axis < 0 {
            continue;
        }
        let src_axis = src_axis as usize;
        let src_dim = src_shape[src_axis];
        if src_dim == out_shape[axis] {
            eff[axis] = src_strides[src_axis];
        } else if src_dim == 1 {
            eff[axis] = 0;
        } else {
            return Err(EpError::Ir(
                onnx_runtime_ir::IrError::BroadcastIncompatible {
                    a: src_shape.to_vec(),
                    b: out_shape.to_vec(),
                },
            ));
        }
    }
    Ok(eff)
}

/// Broadcast a dense row-major `src` of `src_shape` onto `out_shape`, calling
/// `f` with `(flat_out_index, src_value)` for every output element.
///
/// Implements numpy broadcasting: `src_shape` is right-aligned to `out_shape`
/// and any axis of extent 1 (or missing) contributes stride 0. Generic over the
/// element type `T` so every arithmetic kernel shares one broadcast walk.
pub fn broadcast_apply<T: Copy>(
    src: &[T],
    src_shape: &[usize],
    out_shape: &[usize],
    mut f: impl FnMut(usize, T),
) -> Result<()> {
    let eff = broadcast_effective_strides(src_shape, out_shape)?;
    let out_rank = out_shape.len();
    let n = numel(out_shape);
    if n == 0 {
        return Ok(());
    }
    let mut idx = vec![0usize; out_rank];
    let mut flat = 0usize;
    loop {
        let mut src_off = 0i64;
        for (e, &i) in eff.iter().zip(&idx) {
            src_off += e * i as i64;
        }
        f(flat, src[src_off as usize]);
        flat += 1;
        if !next_index(out_shape, &mut idx) {
            break;
        }
    }
    Ok(())
}

/// Dispatch counter for the vDSP contiguous f32 fast path (macOS/iOS).
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub static ADD_VDSP_TEST_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Dispatch counter for the scalar fallback path.
pub static ADD_SCALAR_TEST_HITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// The shared native dense walk must remain reachable in reference builds too.
#[cfg(test)]
static ADD_DENSE_TEST_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl Kernel for AddKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Add", inputs, outputs, 2, 2, 1)?;
        // Validate at the operator boundary before either native fast
        // path forms typed slices. Removing a reference route must not remove
        // its input-view checks or permit implicit cross-device access.
        for (label, input) in ["Add input 0", "Add input 1"].into_iter().zip(inputs) {
            if !input.device.is_host_accessible() {
                return Err(EpError::KernelFailed(format!(
                    "{label}: {:?} tensor on {:?} is not Host-accessible; explicitly transfer it to Host-accessible storage before CPU Add",
                    input.dtype, input.device,
                )));
            }
            input.validate().map_err(|source| EpError::Backend {
                operation: label,
                source: Box::new(source),
            })?;
        }
        if !outputs[0].device.is_host_accessible() {
            return Err(EpError::KernelFailed(format!(
                "Add output 0: {:?} tensor on {:?} is not Host-accessible; provide Host-accessible output storage for CPU Add",
                outputs[0].dtype, outputs[0].device,
            )));
        }
        outputs[0].validate().map_err(|source| EpError::Backend {
            operation: "Add output 0",
            source: Box::new(source),
        })?;
        crate::trace::record_kernel_metrics(inputs, outputs, || outputs[0].numel() as u64);
        // Empty results still validate dtype/broadcast semantics in the
        // general walk, without forming typed slices from zero-size backing.
        let has_elements = outputs[0].numel() != 0;
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if has_elements && add_vdsp_f32(inputs, &mut outputs[0])? {
            ADD_VDSP_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        // Shared with Sub/Mul/Div so there is exactly one dense binary walk in
        // the EP; covers same-shape and suffix-broadcast (including a scalar)
        // operands. Without it even a same-shape f32 Add fell to `add_typed`,
        // which allocates a whole-tensor accumulator and walks it three times.
        if has_elements && super::elementwise::add_dense_fast_path(inputs, &mut outputs[0]) {
            #[cfg(test)]
            ADD_DENSE_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        ADD_SCALAR_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        dispatch_arith!(inputs[0].dtype, "Add", T => add_typed::<T>(inputs, outputs))
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

/// Dtype-generic Add: widen both operands to the compute domain, broadcast-add,
/// narrow back. Both operands and the output must share `T`'s dtype.
fn add_typed<T: NumericElem>(inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
    require_same_dtype("Add", &inputs[1], T::DTYPE)?;
    let a = to_dense::<T>(&inputs[0])?;
    let b = to_dense::<T>(&inputs[1])?;
    let out_shape = outputs[0].shape.to_vec();
    let mut acc = vec![T::Acc::default(); numel(&out_shape)];
    broadcast_apply(&a, inputs[0].shape, &out_shape, |i, v| acc[i] = v.to_acc())?;
    broadcast_apply(&b, inputs[1].shape, &out_shape, |i, v| {
        acc[i] = acc[i].c_add(v.to_acc())
    })?;
    let out: Vec<T> = acc.into_iter().map(T::from_acc).collect();
    write_dense::<T>(&mut outputs[0], &out)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Apple Silicon fast path: vDSP_vadd for contiguous f32 same-shape tensors
// ═══════════════════════════════════════════════════════════════════════════════

/// Contiguous f32 same-shape elementwise Add using Accelerate's vDSP_vadd.
/// Returns `Ok(true)` when the fast path ran, `Ok(false)` when conditions
/// are not met and the caller should fall through.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn add_vdsp_f32(inputs: &[TensorView], output: &mut TensorMut) -> Result<bool> {
    use crate::dtype::to_dense_f32_widen;

    if inputs[0].dtype != DataType::Float32
        || inputs[1].dtype != DataType::Float32
        || output.dtype != DataType::Float32
        || inputs[0].shape != inputs[1].shape
        || inputs[0].shape != output.shape
        || !inputs[0].is_contiguous()
        || !inputs[1].is_contiguous()
        || !output.is_contiguous()
    {
        return Ok(false);
    }
    // Alias check: output must not overlap either input.
    let output_start = output.data_ptr_mut::<u8>() as usize;
    let output_end = output_start.saturating_add(output.byte_size());
    if inputs.iter().any(|input| {
        let start = input.data_ptr::<u8>() as usize;
        let end = start.saturating_add(input.byte_size());
        output_start < end && start < output_end
    }) {
        return Ok(false);
    }
    let left = to_dense_f32_widen("Add", &inputs[0])?;
    let right = to_dense_f32_widen("Add", &inputs[1])?;
    let n = output.numel();
    // SAFETY: validated contiguous Float32 output with numel matching both inputs.
    let out_slice = unsafe { std::slice::from_raw_parts_mut(output.data_ptr_mut::<f32>(), n) };
    // SAFETY: vDSP_vadd reads `n` floats from each source (stride 1) and writes
    // `n` floats to the output. All three pointers are valid for `n` f32s.
    unsafe {
        vDSP_vadd(
            left.as_ptr(),
            1,
            right.as_ptr(),
            1,
            out_slice.as_mut_ptr(),
            1,
            n as u64,
        );
    }
    Ok(true)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn vDSP_vadd(
        __A: *const f32,
        __IA: i64,
        __B: *const f32,
        __IB: i64,
        __C: *mut f32,
        __IC: i64,
        __N: u64,
    );
}

/// Guard that a secondary operand carries the same dtype the dispatch selected.
pub(crate) fn require_same_dtype(op: &str, view: &TensorView, want: DataType) -> Result<()> {
    if view.dtype != want {
        return Err(EpError::KernelFailed(format!(
            "{op}: all operands must share one dtype (WHAT: got a {:?} operand \
             alongside {want:?}). WHY: ONNX elementwise ops are homogeneous — \
             mixed-dtype inputs are undefined. HOW: insert a `Cast` so every \
             operand is {want:?}.",
            view.dtype
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn add_same_shape() {
        let a = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let b = Owned::f32(&[2, 2], &[10., 20., 30., 40.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        AddKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![11., 22., 33., 44.]);
    }

    #[test]
    fn add_broadcasts_row_vector() {
        // [2,3] + [3] -> [2,3]
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3], &[10., 20., 30.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        AddKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![11., 22., 33., 14., 25., 36.]);
    }

    #[test]
    fn add_broadcasts_column_vector() {
        // [2,3] + [2,1] -> [2,3]
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[2, 1], &[10., 20.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        AddKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![11., 12., 13., 24., 25., 26.]);
    }

    #[test]
    fn add_f16_broadcasts() {
        // f16 must compute in f32 and round back, NOT reinterpret the 2-byte
        // storage as f32 bits. [2,3] + [3].
        let a = Owned::f16(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f16(&[3], &[10., 20., 30.]);
        let mut out = Owned::zeros(DataType::Float16, &[2, 3]);
        AddKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f16_as_f32(), vec![11., 22., 33., 14., 25., 36.]);
    }

    #[test]
    fn add_bf16_same_shape() {
        let a = Owned::bf16(&[2, 2], &[1., 2., 3., 4.]);
        let b = Owned::bf16(&[2, 2], &[10., 20., 30., 40.]);
        let mut out = Owned::zeros(DataType::BFloat16, &[2, 2]);
        AddKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_bf16_as_f32(), vec![11., 22., 33., 44.]);
    }

    #[test]
    fn add_f16_preserves_nan_and_inf_without_bit_corruption() {
        // Adversarial: +inf (0x7C00) and NaN (0x7E00) f16 patterns. Adding 1.0
        // must keep inf==inf and NaN==NaN — a naive f32-reinterpret of the
        // 2-byte storage would silently mangle these.
        let a = Owned::f16_bits(&[3], &[0x7C00, 0xFF00 /* -NaN */, 0x3C00 /* 1.0 */]);
        let b = Owned::f16(&[3], &[1.0, 1.0, 1.0]);
        let mut out = Owned::zeros(DataType::Float16, &[3]);
        AddKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        let bits = out.to_u16_bits();
        assert_eq!(bits[0], 0x7C00, "inf + 1 must stay +inf");
        // NaN + 1 = NaN (exponent all ones, non-zero mantissa).
        assert_eq!(bits[1] & 0x7C00, 0x7C00);
        assert_ne!(bits[1] & 0x03FF, 0);
        assert_eq!(out.to_f16_as_f32()[2], 2.0);
    }

    #[test]
    fn add_int32_wraps() {
        let a = Owned::i32(&[3], &[1, 2, i32::MAX]);
        let b = Owned::i32(&[3], &[10, 20, 1]);
        let mut out = Owned::zeros(DataType::Int32, &[3]);
        AddKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i32(), vec![11, 22, i32::MIN]);
    }

    #[test]
    fn add_uint8_broadcasts_and_wraps() {
        let a = Owned::u8(&[2, 2], &[1, 200, 3, 4]);
        let b = Owned::u8(&[1], &[100]);
        let mut out = Owned::zeros(DataType::Uint8, &[2, 2]);
        AddKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_u8(), vec![101, 44, 103, 104]); // 200+100=300 wraps to 44
    }

    #[test]
    fn add_rejects_bool_with_rule1_message() {
        let a = Owned::bool_(&[2], &[true, false]);
        let b = Owned::bool_(&[2], &[false, true]);
        let mut out = Owned::zeros(DataType::Bool, &[2]);
        let err = AddKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("WHAT") && msg.contains("WHY") && msg.contains("HOW"));
    }

    #[test]
    fn add_rejects_mixed_dtype_operands() {
        let a = Owned::f16(&[2], &[1., 2.]);
        let b = Owned::f32(&[2], &[1., 2.]);
        let mut out = Owned::zeros(DataType::Float16, &[2]);
        let err = AddKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap_err();
        assert!(format!("{err}").contains("share one dtype"));
    }

    // ─── Dispatch reachability ───────────────────────────────────────────

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn add_contiguous_f32_same_shape_uses_vdsp() {
        use std::sync::atomic::Ordering;
        let read_hits = || super::ADD_VDSP_TEST_HITS.load(Ordering::Relaxed);
        let before = read_hits();
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[2, 3], &[10., 20., 30., 40., 50., 60.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        AddKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        let after = read_hits();
        assert!(
            after > before,
            "Contiguous f32 same-shape Add did not reach the native vDSP path"
        );
        assert_eq!(out.to_f32(), vec![11., 22., 33., 44., 55., 66.]);
    }

    #[test]
    fn add_suffix_broadcast_reaches_native_dense_in_every_build() {
        use std::sync::atomic::Ordering;
        let before = super::ADD_DENSE_TEST_HITS.load(Ordering::Relaxed);
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3], &[10., 20., 30.]);
        let mut output = Owned::zeros_f32(&[2, 3]);
        AddKernel
            .execute(&[a.view(), b.view()], &mut [output.view_mut()])
            .unwrap();
        assert!(super::ADD_DENSE_TEST_HITS.load(Ordering::Relaxed) > before);
        assert_eq!(output.to_f32(), vec![11., 22., 33., 14., 25., 36.]);
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn add_contiguous_f32_same_shape_uses_native_dense() {
        use std::sync::atomic::Ordering;
        let before = super::ADD_DENSE_TEST_HITS.load(Ordering::Relaxed);
        let a = Owned::f32(&[3], &[1., 2., 3.]);
        let b = Owned::f32(&[3], &[10., 20., 30.]);
        let mut output = Owned::zeros_f32(&[3]);
        AddKernel
            .execute(&[a.view(), b.view()], &mut [output.view_mut()])
            .unwrap();
        assert!(super::ADD_DENSE_TEST_HITS.load(Ordering::Relaxed) > before);
        assert_eq!(output.to_f32(), vec![11., 22., 33.]);
    }

    #[test]
    fn add_native_f32_preserves_ieee_values_and_tails_in_every_build() {
        let pairs = [
            (-0.0f32, -0.0f32),
            (-0.0, 0.0),
            (f32::MAX, f32::MAX),
            (f32::MAX, -f32::MAX),
            (f32::INFINITY, f32::NEG_INFINITY),
            (f32::INFINITY, 1.0),
            (f32::NEG_INFINITY, -1.0),
            (f32::MIN_POSITIVE, f32::from_bits(1)),
            (f32::from_bits(1), f32::from_bits(1)),
            (f32::NAN, 1.0),
            (1.0, f32::NAN),
            (1.25, -0.5),
        ];
        for n in [1, 3, 4, 7, 8, 17, 65, 1025] {
            let left: Vec<f32> = (0..n).map(|i| pairs[i % pairs.len()].0).collect();
            let right: Vec<f32> = (0..n).map(|i| pairs[i % pairs.len()].1).collect();
            let a = Owned::f32(&[n], &left);
            let b = Owned::f32(&[n], &right);
            let mut out = Owned::zeros_f32(&[n]);
            AddKernel
                .execute(&[a.view(), b.view()], &mut [out.view_mut()])
                .unwrap();
            for (i, got) in out.to_f32().into_iter().enumerate() {
                let expected = left[i] + right[i];
                if expected.is_nan() {
                    assert!(got.is_nan(), "n={n} index={i}: expected NaN, got {got}");
                } else {
                    assert_eq!(got.to_bits(), expected.to_bits(), "n={n} index={i}");
                }
            }
        }
        let a_empty = Owned::f32(&[0], &[]);
        let b_empty = Owned::f32(&[0], &[]);
        let mut out_empty = Owned::zeros_f32(&[0]);
        AddKernel
            .execute(
                &[a_empty.view(), b_empty.view()],
                &mut [out_empty.view_mut()],
            )
            .unwrap();
        assert!(out_empty.bytes.is_empty());
        // Back a zero-element view with an aligned, live f32 allocation. It
        // must leave sentinel storage untouched, including in reference builds.
        let mut a = Owned::f32(&[1], &[1.0]);
        let mut b = Owned::f32(&[1], &[2.0]);
        let mut out = Owned::f32(&[1], &[123.0]);
        a.shape[0] = 0;
        b.shape[0] = 0;
        out.shape[0] = 0;
        AddKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.bytes, 123.0f32.to_le_bytes());
    }

    #[test]
    fn add_rejects_invalid_view_metadata_before_native_pointer_access() {
        let a = Owned::f32(&[2], &[1.0, 2.0]);
        let b = Owned::f32(&[2], &[3.0, 4.0]);
        for slot in 0..3 {
            let mut out = Owned::f32(&[2], &[123.0, 456.0]);
            let mut inputs = [a.view(), b.view()];
            let mut output = out.view_mut();
            match slot {
                0 => inputs[0].byte_offset = 2,
                1 => inputs[1].strides = &[],
                _ => output.byte_offset = 2,
            }
            let error = AddKernel.execute(&inputs, &mut [output]).unwrap_err();
            let source = std::error::Error::source(&error).unwrap();
            assert!(matches!(
                source.downcast_ref::<EpError>(),
                Some(EpError::InvalidTensorView { .. })
            ));
            let label = ["Add input 0", "Add input 1", "Add output 0"][slot];
            assert!(error.to_string().contains(label), "{error}");
            assert_eq!(out.to_f32(), vec![123.0, 456.0]);
        }
    }

    #[test]
    fn add_rejects_non_host_views_without_a_cuda_device() {
        let a = Owned::f32(&[2], &[1.0, 2.0]);
        let b = Owned::f32(&[2], &[3.0, 4.0]);
        for slot in 0..3 {
            let mut out = Owned::f32(&[2], &[123.0, 456.0]);
            let mut inputs = [a.view(), b.view()];
            let mut output = out.view_mut();
            // Only the device annotation changes; no CUDA allocation or API
            // is needed to prove rejection before a CPU data access.
            match slot {
                0 | 1 => inputs[slot].device = onnx_runtime_ep_api::DeviceId::cuda(0),
                _ => output.device = onnx_runtime_ep_api::DeviceId::cuda(0),
            }
            let error = AddKernel.execute(&inputs, &mut [output]).unwrap_err();
            let label = ["Add input 0", "Add input 1", "Add output 0"][slot];
            let message = error.to_string();
            assert!(
                message.contains(label) && message.contains("Host-accessible"),
                "{message}"
            );
            assert_eq!(out.to_f32(), vec![123.0, 456.0]);
        }
    }

    /// An **interior** size-1 axis is not a right-aligned suffix, so it cannot
    /// be expressed as a repeat of a contiguous block and must still reach the
    /// general strided fallback. This is what keeps `add_dense_fast_path` from
    /// being applied unconditionally.
    #[test]
    fn add_interior_broadcast_uses_scalar_fallback() {
        use std::sync::atomic::Ordering;
        let before = super::ADD_SCALAR_TEST_HITS.load(Ordering::Relaxed);
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[2, 1], &[10., 20.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        AddKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        let after = super::ADD_SCALAR_TEST_HITS.load(Ordering::Relaxed);
        assert!(
            after > before,
            "Interior-axis broadcasting Add did not reach the scalar fallback"
        );
        assert_eq!(out.to_f32(), vec![11., 12., 13., 24., 25., 26.]);
    }

    /// A right-aligned suffix broadcast (`[2, 3] <- [2, 3] + [3]`) now takes the
    /// shared dense path instead of the strided walk, and must produce exactly
    /// the same values. The previous version of this test asserted the opposite
    /// dispatch; the values it never checked are asserted here.
    ///
    /// Dispatch is pinned by asserting the predicate itself rather than by
    /// watching `ADD_SCALAR_TEST_HITS`: that counter is process-global and other
    /// tests in the same binary increment it concurrently, so an equality
    /// assertion on it would be racy. `AddKernel::execute` reaches the counter
    /// only after `add_dense_fast_path` declines, and the earlier arm
    /// (vDSP) requires identical operand shapes, so a broadcasting
    /// input that this predicate accepts provably never reaches the fallback.
    #[test]
    fn add_suffix_broadcast_uses_the_dense_fast_path() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3], &[10., 20., 30.]);
        let want = vec![11., 22., 33., 14., 25., 36.];

        let mut direct = Owned::zeros_f32(&[2, 3]);
        assert!(
            crate::kernels::elementwise::add_dense_fast_path(
                &[a.view(), b.view()],
                &mut direct.view_mut()
            ),
            "suffix-broadcast Add was declined by the dense fast path"
        );
        assert_eq!(direct.to_f32(), want);

        let mut out = Owned::zeros_f32(&[2, 3]);
        AddKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), want);
    }

    /// Scalar operand on either side, which is what an inlined ONNX activation
    /// function emits. Dispatch is pinned the same way as the suffix case above.
    #[test]
    fn add_scalar_broadcast_uses_the_dense_fast_path() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let k = Owned::f32(&[], &[0.5]);
        let want = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
        for (l, r) in [(&a, &k), (&k, &a)] {
            let mut direct = Owned::zeros_f32(&[2, 3]);
            assert!(
                crate::kernels::elementwise::add_dense_fast_path(
                    &[l.view(), r.view()],
                    &mut direct.view_mut()
                ),
                "scalar-broadcast Add was declined by the dense fast path"
            );
            assert_eq!(direct.to_f32(), want);

            let mut out = Owned::zeros_f32(&[2, 3]);
            AddKernel
                .execute(&[l.view(), r.view()], &mut [out.view_mut()])
                .unwrap();
            assert_eq!(out.to_f32(), want);
        }
    }

    /// The decline side of the same predicate: an interior unit axis is not a
    /// right-aligned suffix, so the fast path must refuse it outright (and leave
    /// the output untouched) rather than compute a wrong answer.
    #[test]
    fn add_interior_broadcast_is_declined_by_the_dense_fast_path() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[2, 1], &[10., 20.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        assert!(
            !crate::kernels::elementwise::add_dense_fast_path(
                &[a.view(), b.view()],
                &mut out.view_mut()
            ),
            "interior-axis broadcast was accepted by the dense fast path"
        );
        assert_eq!(out.to_f32(), vec![0.; 6]);
    }
}
