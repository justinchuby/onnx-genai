//! `Pad`: enlarge (or crop) a tensor along each axis (`docs/ORT2.md` §4.4).
//!
//! The `pads` amounts come from the opset-11+ **input** (`int64`, laid out as
//! `[begin_0..begin_{k-1}, end_0..end_{k-1}]`) or the legacy opset-2 `pads`
//! **attribute**. Four modes are supported: `constant` (default; fill value from
//! the `constant_value` input, else the legacy `value` attribute, else zero),
//! `reflect`, `edge`, and `wrap`. The optional opset-18 `axes` input restricts
//! the pads to a subset of axes (the rest pad by zero); negative axes and
//! negative pads (which crop) follow the ONNX reference. The op copies raw
//! element bytes, so every fixed-width dtype is supported.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node, compute_contiguous_strides};

use super::{elem_size, to_dense_bytes, to_dense_i64, write_dense_bytes};
use crate::strided::{next_index, numel};

/// Padding mode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PadMode {
    Constant,
    Reflect,
    Edge,
    Wrap,
}

/// `Pad` kernel carrying the resolved `mode` and the legacy compile-time
/// `pads` / `value` attributes (used only when the corresponding inputs are
/// absent).
pub struct PadKernel {
    mode: PadMode,
    pads_attr: Option<Vec<i64>>,
    value_attr: Option<f32>,
}

/// Factory reading `mode`, the legacy `pads` attribute, and the legacy `value`
/// attribute.
pub struct PadFactory;

impl KernelFactory for PadFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let mode = match node.attr("mode").and_then(Attribute::as_str) {
            None | Some("constant") => PadMode::Constant,
            Some("reflect") => PadMode::Reflect,
            Some("edge") => PadMode::Edge,
            Some("wrap") => PadMode::Wrap,
            Some(other) => {
                return Err(EpError::KernelFailed(format!(
                    "Pad: WHAT: unknown mode {other:?}. \
                     WHY: only constant/reflect/edge/wrap are defined. \
                     HOW: use one of those four modes."
                )));
            }
        };
        let pads_attr = node
            .attr("pads")
            .and_then(Attribute::as_ints)
            .map(<[i64]>::to_vec);
        let value_attr = node.attr("value").and_then(Attribute::as_float);
        Ok(Box::new(PadKernel {
            mode,
            pads_attr,
            value_attr,
        }))
    }
}

/// numpy-style `reflect` index mapping (edge not repeated).
fn reflect_index(i: i64, n: i64) -> i64 {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut m = i % period;
    if m < 0 {
        m += period;
    }
    if m >= n { period - m } else { m }
}

/// Encode `val` into little-endian bytes for `dtype` (legacy `value` attribute
/// fill). Covers the common numeric dtypes; anything else falls back to zero.
fn scalar_bytes(dtype: DataType, val: f64) -> Vec<u8> {
    match dtype {
        DataType::Float32 => (val as f32).to_le_bytes().to_vec(),
        DataType::Float64 => val.to_le_bytes().to_vec(),
        DataType::Float16 => half::f16::from_f32(val as f32).to_le_bytes().to_vec(),
        DataType::BFloat16 => half::bf16::from_f32(val as f32).to_le_bytes().to_vec(),
        DataType::Int64 => (val as i64).to_le_bytes().to_vec(),
        DataType::Int32 => (val as i32).to_le_bytes().to_vec(),
        DataType::Int16 => (val as i16).to_le_bytes().to_vec(),
        DataType::Int8 => vec![val as i8 as u8],
        DataType::Uint64 => (val as u64).to_le_bytes().to_vec(),
        DataType::Uint32 => (val as u32).to_le_bytes().to_vec(),
        DataType::Uint16 => (val as u16).to_le_bytes().to_vec(),
        DataType::Uint8 => vec![val as u8],
        DataType::Bool => vec![u8::from(val != 0.0)],
        other => vec![0u8; other.byte_size().max(1)],
    }
}

impl Kernel for PadKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.is_empty() || inputs.len() > 4 {
            return Err(EpError::KernelFailed(format!(
                "Pad: WHAT: expected 1..=4 inputs, got {}. \
                 WHY: Pad takes data, and optional pads/constant_value/axes. \
                 HOW: connect the data (and optionally pads, constant_value, axes).",
                inputs.len()
            )));
        }
        if outputs.is_empty() {
            return Err(EpError::KernelFailed(
                "Pad: WHAT: no output view. WHY: Pad writes into the executor-provided output. \
                 HOW: allocate one Pad output."
                    .into(),
            ));
        }

        let esize = elem_size(inputs[0].dtype)?;
        let in_shape = inputs[0].shape.to_vec();
        let rank = in_shape.len();
        let src = to_dense_bytes(&inputs[0])?;

        // `pads`: input 1 (opset 11+) else the legacy attribute.
        let pads: Vec<i64> = if inputs.len() >= 2 && !inputs[1].is_absent() {
            to_dense_i64(&inputs[1])?
        } else {
            self.pads_attr.clone().ok_or_else(|| {
                EpError::KernelFailed(
                    "Pad: WHAT: no `pads` input or attribute. WHY: Pad needs the per-axis \
                     amounts. HOW: supply a `pads` input (opset 11+) or attribute (opset 2)."
                        .into(),
                )
            })?
        };

        // Optional `axes` input (opset 18+); default is all axes in order.
        let axes: Option<Vec<i64>> = if inputs.len() >= 4 && !inputs[3].is_absent() {
            Some(to_dense_i64(&inputs[3])?)
        } else {
            None
        };
        let num_axes = axes.as_ref().map_or(rank, Vec::len);
        if pads.len() != 2 * num_axes {
            return Err(EpError::KernelFailed(format!(
                "Pad: WHAT: `pads` has {} entries but {num_axes} axes need {}. \
                 WHY: `pads` is [begin.., end..], two per padded axis. \
                 HOW: provide {} pad values.",
                pads.len(),
                2 * num_axes,
                2 * num_axes
            )));
        }

        let mut begin = vec![0i64; rank];
        let mut end = vec![0i64; rank];
        for j in 0..num_axes {
            let raw = axes.as_ref().map_or(j as i64, |a| a[j]);
            let ax = if raw < 0 { raw + rank as i64 } else { raw };
            if ax < 0 || ax as usize >= rank {
                return Err(EpError::KernelFailed(format!(
                    "Pad: WHAT: axis {raw} is out of range for rank {rank}. \
                     WHY: each padded axis must exist. HOW: use axes in [-{rank}, {rank})."
                )));
            }
            begin[ax as usize] = pads[j];
            end[ax as usize] = pads[num_axes + j];
        }

        let mut out_shape = Vec::with_capacity(rank);
        for d in 0..rank {
            let dim = in_shape[d] as i64 + begin[d] + end[d];
            if dim < 0 {
                return Err(EpError::KernelFailed(format!(
                    "Pad: WHAT: axis {d} pads {begin:?}/{end:?} crop past extent {}. \
                     WHY: the padded extent {dim} is negative. \
                     HOW: use pads whose crop does not exceed the input extent.",
                    in_shape[d]
                )));
            }
            if self.mode != PadMode::Constant && dim > 0 && in_shape[d] == 0 {
                return Err(EpError::KernelFailed(format!(
                    "Pad: WHAT: {:?} mode needs a non-empty axis {d} to sample from. \
                     WHY: reflect/edge/wrap read existing elements. \
                     HOW: use constant mode for empty axes.",
                    self.mode
                )));
            }
            out_shape.push(dim as usize);
        }

        // Constant fill: constant_value input, else legacy `value` attribute,
        // else zero.
        let fill: Vec<u8> = if inputs.len() >= 3 && !inputs[2].is_absent() {
            let cv = to_dense_bytes(&inputs[2])?;
            if cv.len() < esize {
                return Err(EpError::KernelFailed(
                    "Pad: WHAT: constant_value holds fewer bytes than one element. \
                     WHY: it must be a scalar of the data dtype. \
                     HOW: provide a one-element constant_value.".into(),
                ));
            }
            cv[..esize].to_vec()
        } else if let Some(v) = self.value_attr {
            scalar_bytes(inputs[0].dtype, v as f64)
        } else {
            vec![0u8; esize]
        };

        let in_strides = compute_contiguous_strides(&in_shape);
        let n_out = numel(&out_shape);
        let mut out = vec![0u8; n_out * esize];
        if n_out == 0 {
            return write_dense_bytes(&mut outputs[0], &out);
        }

        let mut idx = vec![0usize; rank];
        let mut w = 0usize;
        loop {
            let mut in_off = 0i64;
            let mut in_range = true;
            for d in 0..rank {
                let i = idx[d] as i64 - begin[d];
                let n = in_shape[d] as i64;
                let mapped = match self.mode {
                    PadMode::Constant => {
                        if i < 0 || i >= n {
                            in_range = false;
                            0
                        } else {
                            i
                        }
                    }
                    PadMode::Reflect => reflect_index(i, n),
                    PadMode::Edge => i.clamp(0, n - 1),
                    PadMode::Wrap => ((i % n) + n) % n,
                };
                if !in_range {
                    break;
                }
                in_off += in_strides[d] * mapped;
            }
            if in_range {
                let s = in_off as usize * esize;
                out[w..w + esize].copy_from_slice(&src[s..s + esize]);
            } else {
                out[w..w + esize].copy_from_slice(&fill);
            }
            w += esize;
            if !next_index(&out_shape, &mut idx) {
                break;
            }
        }
        write_dense_bytes(&mut outputs[0], &out)
    }

    fn supports_strided_input(&self, input_idx: usize) -> bool {
        input_idx == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    fn constant_kernel(value: Option<f32>) -> PadKernel {
        PadKernel {
            mode: PadMode::Constant,
            pads_attr: None,
            value_attr: value,
        }
    }

    #[test]
    fn pad_constant_pads_as_input() {
        // [2,2] padded by 1 on each side of axis 0 → [4,2], zeros around.
        let x = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let pads = Owned::i64(&[4], &[1, 0, 1, 0]);
        let mut out = Owned::zeros_f32(&[4, 2]);
        constant_kernel(None)
            .execute(&[x.view(), pads.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(
            out.to_f32(),
            vec![0., 0., 1., 2., 3., 4., 0., 0.]
        );
    }

    #[test]
    fn pad_constant_nonzero_value_both_axes() {
        // Pad [1,2] to [3,4] with begin/end 1 on both axes, fill 9.
        let x = Owned::f32(&[1, 2], &[1., 2.]);
        let pads = Owned::i64(&[4], &[1, 1, 1, 1]);
        let cv = Owned::f32(&[], &[9.]);
        let mut out = Owned::zeros_f32(&[3, 4]);
        constant_kernel(None)
            .execute(
                &[x.view(), pads.view(), cv.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(
            out.to_f32(),
            vec![
                9., 9., 9., 9., //
                9., 1., 2., 9., //
                9., 9., 9., 9.,
            ]
        );
    }

    #[test]
    fn pad_constant_value_via_legacy_attribute() {
        let x = Owned::f32(&[2], &[5., 6.]);
        let pads = Owned::i64(&[2], &[1, 1]);
        let mut out = Owned::zeros_f32(&[4]);
        constant_kernel(Some(7.0))
            .execute(&[x.view(), pads.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![7., 5., 6., 7.]);
    }

    #[test]
    fn pad_reflect_1d() {
        // [1,2,3,4] reflect pad 2 before, 2 after → [3,2,1,2,3,4,3,2].
        let x = Owned::f32(&[4], &[1., 2., 3., 4.]);
        let pads = Owned::i64(&[2], &[2, 2]);
        let mut out = Owned::zeros_f32(&[8]);
        PadKernel {
            mode: PadMode::Reflect,
            pads_attr: None,
            value_attr: None,
        }
        .execute(&[x.view(), pads.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![3., 2., 1., 2., 3., 4., 3., 2.]);
    }

    #[test]
    fn pad_edge_1d() {
        // edge pad repeats the boundary element.
        let x = Owned::f32(&[3], &[1., 2., 3.]);
        let pads = Owned::i64(&[2], &[2, 1]);
        let mut out = Owned::zeros_f32(&[6]);
        PadKernel {
            mode: PadMode::Edge,
            pads_attr: None,
            value_attr: None,
        }
        .execute(&[x.view(), pads.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![1., 1., 1., 2., 3., 3.]);
    }

    #[test]
    fn pad_wrap_1d() {
        // wrap pad tiles the input periodically.
        let x = Owned::f32(&[3], &[1., 2., 3.]);
        let pads = Owned::i64(&[2], &[2, 2]);
        let mut out = Owned::zeros_f32(&[7]);
        PadKernel {
            mode: PadMode::Wrap,
            pads_attr: None,
            value_attr: None,
        }
        .execute(&[x.view(), pads.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![2., 3., 1., 2., 3., 1., 2.]);
    }

    #[test]
    fn pad_negative_crops() {
        // Negative pads crop: [1..4] of a length-5 axis → [2,3,4].
        let x = Owned::f32(&[5], &[1., 2., 3., 4., 5.]);
        let pads = Owned::i64(&[2], &[-1, -1]);
        let mut out = Owned::zeros_f32(&[3]);
        constant_kernel(None)
            .execute(&[x.view(), pads.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![2., 3., 4.]);
    }

    #[test]
    fn pad_with_axes_input_subset() {
        // Pad only axis 1 (of a [2,2]) by 1/1 via the axes input → [2,4].
        let x = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let pads = Owned::i64(&[2], &[1, 1]);
        let cv = Owned::f32(&[], &[0.]);
        let axes = Owned::i64(&[1], &[1]);
        let mut out = Owned::zeros_f32(&[2, 4]);
        constant_kernel(None)
            .execute(
                &[x.view(), pads.view(), cv.view(), axes.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(
            out.to_f32(),
            vec![0., 1., 2., 0., 0., 3., 4., 0.]
        );
    }

    #[test]
    fn pad_int64_dtype_agnostic() {
        let x = Owned::i64(&[2], &[10, 20]);
        let pads = Owned::i64(&[2], &[1, 0]);
        let cv = Owned::i64(&[], &[99]);
        let mut out = Owned::zeros(DataType::Int64, &[3]);
        constant_kernel(None)
            .execute(
                &[x.view(), pads.view(), cv.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(out.to_i64(), vec![99, 10, 20]);
    }
}
