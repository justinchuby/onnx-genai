//! Shared validated floating projection inputs; no owner or admission policy.
use super::decoded_projection::ProjectionScratch;
use crate::dtype::{can_borrow_dense, to_dense_f32_widen_into};
use crate::strided::validate_host_storage;
use onnx_runtime_ep_api::{EpError, Result, TensorView};
use onnx_runtime_ir::DataType;

/// A validated source, with no payload read or allocation before admission.
pub(super) struct ProjectionInput<'a> {
    view: &'a TensorView<'a>,
    elements: usize,
    borrowed: bool,
    name: &'static str,
}

impl<'a> ProjectionInput<'a> {
    pub(super) fn new(
        op: &'static str,
        name: &'static str,
        view: &'a TensorView<'a>,
    ) -> Result<Self> {
        let elements = match view.dtype {
            DataType::Float32 => validate_host_storage::<f32>(op, name, view)?,
            DataType::Float16 => validate_host_storage::<half::f16>(op, name, view)?,
            DataType::BFloat16 => validate_host_storage::<half::bf16>(op, name, view)?,
            dtype => {
                return Err(EpError::KernelFailed(format!(
                    "{op}: {name} has unsupported dtype {dtype:?}; bind Float32, Float16 or BFloat16 storage"
                )));
            }
        };
        let borrowed = elements == 0 || can_borrow_dense::<f32>(view);
        Ok(Self {
            view,
            elements,
            borrowed,
            name,
        })
    }

    pub(super) fn staging_elements(&self) -> usize {
        if self.borrowed { 0 } else { self.elements }
    }

    pub(super) fn read<'b>(&'b self, lane: &'b mut ProjectionScratch<'_>) -> Result<&'b [f32]> {
        if self.elements == 0 {
            return Ok(&[]);
        }
        if self.borrowed {
            // SAFETY: construction proves Host span, Float32 dtype, contiguous
            // layout and alignment. The source lives through this short borrow.
            return Ok(unsafe {
                std::slice::from_raw_parts(self.view.data_ptr::<f32>(), self.elements)
            });
        }
        let values = lane.initialized_prefix(self.elements)?;
        to_dense_f32_widen_into(self.name, self.view, values)?;
        Ok(values)
    }
}
