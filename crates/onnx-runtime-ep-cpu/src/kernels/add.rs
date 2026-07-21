//! `Add`: elementwise addition with numpy-style broadcasting, generic over the
//! ONNX numeric dtypes via the shared [`crate::dtype`] mechanism (`docs/ORT2.md`
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
    let out_rank = out_shape.len();
    let src_strides = compute_contiguous_strides(src_shape);
    // Effective stride of each output axis into `src` (0 where broadcast).
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

impl Kernel for AddKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Add", inputs, outputs, 2, 2, 1)?;
        crate::trace::record_kernel_metrics(inputs, outputs, outputs[0].numel() as u64);
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
}
