//! Shared layout and binding facts for the two variable-length attention ops.
//! No allocation authority, device owner, or persistent control-value cache.

use onnx_runtime_ep_api::{
    EpError, Result, TensorMut, TensorView, WorkspaceLayout, WorkspaceLifetime,
    WorkspaceRequirement,
};
use onnx_runtime_ir::{DataType, DeviceId};
use std::alloc::Layout;

pub(super) fn product(values: &[usize], label: impl std::fmt::Display) -> Result<usize> {
    if values.contains(&0) {
        return Ok(0);
    }
    values.iter().try_fold(1usize, |n, &value| n.checked_mul(value)).ok_or_else(|| EpError::KernelFailed(format!("variable-length attention {label} overflows addressable capacity for {values:?}; reduce batch, sequence or head dimensions")))
}

pub(super) fn validate_tensor_span(
    dtype: DataType,
    shape: &[usize],
    label: impl std::fmt::Display,
) -> Result<()> {
    let elements = product(shape, &label)?;
    dtype.checked_storage_bytes(elements).filter(|&bytes| bytes <= isize::MAX as usize).ok_or_else(|| EpError::KernelFailed(format!("variable-length attention {label} dtype {dtype:?} shape {shape:?} exceeds addressable tensor byte capacity; reduce tensor dimensions")))?;
    Ok(())
}

pub(super) fn array_layout<T>(count: usize) -> Result<Layout> {
    Layout::array::<T>(count).map_err(|source| EpError::Backend {
        operation: "plan variable-length attention array; reduce control length, batch, sequence or head dimensions",
        source: Box::new(source),
    })
}

pub(super) fn scratch_layout(
    control_entries: usize,
    rows: usize,
    max_kv: usize,
) -> Result<WorkspaceLayout<2>> {
    if rows == 0 {
        return WorkspaceLayout::new([Layout::new::<()>(); 2]);
    }
    let scores = product(&[rows, max_kv.max(1)], "score elements")?;
    let align = |layout: Layout| {
        layout.align_to(256).map_err(|source| EpError::Backend {
        operation: "align variable-length attention caller regions to 256 bytes; reduce scratch dimensions",
        source: Box::new(source),
    })
    };
    WorkspaceLayout::new([
        align(array_layout::<i32>(control_entries)?)?,
        align(array_layout::<f32>(scores)?)?,
    ])
}

pub(super) fn requirement(layout: &WorkspaceLayout<2>) -> Result<WorkspaceRequirement> {
    Ok(WorkspaceRequirement {
        bytes: u64::try_from(layout.bytes()).map_err(|_| {
            EpError::KernelFailed(
                "variable-length attention workspace exceeds u64; reduce scratch dimensions".into(),
            )
        })?,
        alignment: 256,
        lifetime: WorkspaceLifetime::SessionPersistent,
    })
}

#[derive(Debug)]
struct Binding {
    base: usize,
    offset: usize,
    device: DeviceId,
    dtype: DataType,
    shape: Vec<usize>,
    strides: Vec<i64>,
}
impl Binding {
    fn input(view: &TensorView) -> Self {
        Self {
            base: view.data.0 as usize,
            offset: view.byte_offset,
            device: view.device,
            dtype: view.dtype,
            shape: view.shape.to_vec(),
            strides: view.strides.to_vec(),
        }
    }
    fn output(view: &TensorMut) -> Self {
        Self {
            base: view.data.0 as usize,
            offset: view.byte_offset,
            device: view.device,
            dtype: view.dtype,
            shape: view.shape.to_vec(),
            strides: view.strides.to_vec(),
        }
    }
    fn matches(
        &self,
        base: usize,
        offset: usize,
        device: DeviceId,
        dtype: DataType,
        shape: &[usize],
        strides: &[i64],
    ) -> bool {
        self.base == base
            && self.offset == offset
            && self.device == device
            && self.dtype == dtype
            && self.shape == shape
            && self.strides == strides
    }
}

/// Bindings, not control-value identity. A new invocation reads controls again,
/// even when pointers/geometry are unchanged. Workspace may relocate separately.
#[derive(Debug)]
pub(super) struct InvocationBindings {
    inputs: Vec<Binding>,
    outputs: Vec<Binding>,
}
impl InvocationBindings {
    pub(super) fn new(inputs: &[TensorView], outputs: &[TensorMut]) -> Self {
        Self {
            inputs: inputs.iter().map(Binding::input).collect(),
            outputs: outputs.iter().map(Binding::output).collect(),
        }
    }
    pub(super) fn validate(&self, inputs: &[TensorView], outputs: &[TensorMut]) -> Result<()> {
        let invalid = if inputs.len() != self.inputs.len() || outputs.len() != self.outputs.len() {
            Some("input/output count".to_string())
        } else if let Some(i) = self.inputs.iter().zip(inputs).position(|(saved, view)| {
            !saved.matches(
                view.data.0 as usize,
                view.byte_offset,
                view.device,
                view.dtype,
                view.shape,
                view.strides,
            )
        }) {
            Some(format!(
                "input {i} pointer/offset, device, dtype, shape or strides"
            ))
        } else {
            self.outputs
                .iter()
                .zip(outputs)
                .position(|(saved, view)| {
                    !saved.matches(
                        view.data.0 as usize,
                        view.byte_offset,
                        view.device,
                        view.dtype,
                        view.shape,
                        view.strides,
                    )
                })
                .map(|i| format!("output {i} pointer/offset, device, dtype, shape or strides"))
        };
        if let Some(reason) = invalid {
            return Err(EpError::KernelFailed(format!(
                "variable-length attention prepared {reason} changed; rerun execution workspace preparation using the current input/output views"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, WorkspaceView};
    #[test]
    fn exact_score_extent_and_complete_binding_are_checked() {
        let layout = scratch_layout(3, 12, 7).unwrap();
        assert_eq!(layout.bytes(), 256 + 12 * 7 * 4);
        assert_eq!(scratch_layout(3, 12, 0).unwrap().bytes(), 256 + 12 * 4);
        assert_eq!(
            scratch_layout(usize::MAX, 0, usize::MAX).unwrap().bytes(),
            0
        );
        assert!(scratch_layout(1, usize::MAX, 2).is_err());
        assert!(scratch_layout(usize::MAX, 1, 1).is_err());
        let ptr = DevicePtrMut(0x1000usize as *mut _);
        let regions = layout
            .bind(Some(WorkspaceView::new(ptr, layout.bytes())))
            .unwrap();
        assert_eq!(regions[1].ptr().0 as usize, 0x1100);
        for offered in [
            None,
            Some(WorkspaceView::new(ptr, layout.bytes() - 1)),
            Some(WorkspaceView::new(
                DevicePtrMut(0x1001usize as *mut _),
                layout.bytes(),
            )),
        ] {
            assert!(layout.bind(offered).is_err());
        }
        assert_eq!(
            product(&[usize::MAX, 0, usize::MAX], "empty output").unwrap(),
            0
        );
    }
    #[test]
    fn tensor_span_checks_include_dtype_bytes_and_empty_shapes() {
        assert!(validate_tensor_span(DataType::Float32, &[usize::MAX / 2], "query").is_err());
        assert!(validate_tensor_span(DataType::Float16, &[isize::MAX as usize], "output").is_err());
        validate_tensor_span(
            DataType::Float32,
            &[usize::MAX, 0, usize::MAX],
            "empty query",
        )
        .unwrap();
        validate_tensor_span(DataType::BFloat16, &[3, 7, 11], "key").unwrap();
    }

    #[test]
    fn invocation_binding_rejects_relocation_offset_and_geometry_changes() {
        let input = TensorView::new(
            DevicePtr(0x1000usize as *const _),
            DataType::Int64,
            &[2],
            &[1],
            DeviceId::cpu(),
        );
        let output = TensorMut::new(
            DevicePtrMut(0x2000usize as *mut _),
            DataType::Float32,
            &[2],
            &[1],
            DeviceId::cpu(),
        );
        let saved = InvocationBindings::new(&[input], std::slice::from_ref(&output));
        let mut changed = input;
        changed.byte_offset = 8;
        assert!(
            saved
                .validate(&[changed], std::slice::from_ref(&output))
                .is_err()
        );
        changed = input;
        changed.shape = &[3];
        assert!(
            saved
                .validate(&[changed], std::slice::from_ref(&output))
                .is_err()
        );
        let moved_output = TensorMut::new(
            DevicePtrMut(0x3000usize as *mut _),
            DataType::Float32,
            &[2],
            &[1],
            DeviceId::cpu(),
        );
        assert!(saved.validate(&[input], &[moved_output]).is_err());
        saved
            .validate(&[input], std::slice::from_ref(&output))
            .unwrap();
    }
}
