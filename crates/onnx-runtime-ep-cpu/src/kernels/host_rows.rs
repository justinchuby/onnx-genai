//! Validated borrowed/snapshotted Host f32 rows, independent of an operator.
use crate::{dtype::to_dense_into, strided::validate_host_storage};
use onnx_runtime_ep_api::{DevicePtr, EpError, Result, TensorView};
fn fail(op: &str, detail: impl std::fmt::Display) -> EpError {
    EpError::KernelFailed(format!("{op}: {detail}"))
}
/// Validated fixed-width rows over rank-two or rank-three input. Borrowing
/// depends on row contiguity, not contiguity of the complete tensor.
pub(super) struct Rows<'a> {
    op: &'static str,
    view: &'a TensorView<'a>,
    origin: *const f32,
    outer: [isize; 2],
    leading_last: usize,
    count: usize,
    shape: [usize; 1],
    stride: [i64; 1],
    borrowed: bool,
    name: &'static str,
}
impl<'a> Rows<'a> {
    pub(super) fn new(
        op: &'static str,
        name: &'static str,
        view: &'a TensorView<'a>,
    ) -> Result<Self> {
        validate_host_storage::<f32>(op, name, view)?;
        if !(2..=3).contains(&view.shape.len()) || view.shape.contains(&0) {
            return Err(fail(
                op,
                format!(
                    "{name} shape {:?} must have positive rank-two/three row geometry; bind valid activation/router rows",
                    view.shape
                ),
            ));
        }
        let rank = view.shape.len();
        let width = view.shape[rank - 1];
        let count = view.shape[..rank - 1].iter().product();
        let step = |axis: usize| {
            if view.shape[axis] == 1 {
                0
            } else {
                view.strides[axis] as isize
            }
        };
        let (outer, leading_last) = if rank == 2 {
            ([step(0), 0], 1)
        } else {
            ([step(0), step(1)], view.shape[1])
        };
        let origin = view.data_ptr::<f32>();
        let borrowed = origin.is_aligned() && (width == 1 || view.strides[rank - 1] == 1);
        Ok(Self {
            op,
            view,
            origin,
            outer,
            leading_last,
            count,
            shape: [width],
            stride: [view.strides[rank - 1]],
            borrowed,
            name,
        })
    }
    pub(super) fn scratch_len(&self) -> usize {
        if self.borrowed { 0 } else { self.shape[0] }
    }
    pub(super) fn get<'r>(&'r self, row: usize, scratch: &'r mut [f32]) -> Result<&'r [f32]> {
        if row >= self.count || scratch.len() != self.scratch_len() {
            return Err(fail(
                self.op,
                format!(
                    "{} row {row} needs {} scratch elements within {} rows, received {}; refresh workspace for the actual bindings",
                    self.name,
                    self.scratch_len(),
                    self.count,
                    scratch.len()
                ),
            ));
        }
        let offset = (row / self.leading_last) as isize * self.outer[0]
            + (row % self.leading_last) as isize * self.outer[1];
        // SAFETY: complete row offset arithmetic is bounded by the Host storage
        // span proof; the owner supplies backing bounds and immutable lifetime.
        let origin = unsafe { self.origin.offset(offset) };
        if self.borrowed {
            // SAFETY: every row origin preserves f32 alignment; the last axis
            // is contiguous (or singleton), and all input reads finish before Y
            // is published. No tensor may alias the exclusive scratch backing.
            Ok(unsafe { std::slice::from_raw_parts(origin, self.shape[0]) })
        } else {
            let view = TensorView::new(
                DevicePtr(origin.cast()),
                self.view.dtype,
                &self.shape,
                &self.stride,
                self.view.device,
            );
            to_dense_into(self.op, &view, scratch)?;
            Ok(scratch)
        }
    }
}
