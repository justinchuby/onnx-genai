//! One admitted operation lane for packed grouped output, X rows and routes.
use super::RouteSource;
use crate::{
    kernels::{host_rows::Rows, typed_workspace::ValueIndexWorkspace},
    strided::validate_host_storage,
};
use onnx_runtime_ep_api::{
    DevicePtr, EpError, Kernel, OutputMetadata, Result, TensorMetadata, TensorMut, TensorView,
    WorkspaceExecutionPreparation, WorkspacePreparation, WorkspaceRequirement, WorkspaceView,
    packed_routed::{GroupLayout, packed_group_layout, packed_group_metadata_layout},
};
use onnx_runtime_ir::{
    dequantize_packed::Metadata, packed_routed::Op, quantized_grouped::PACKED_OP,
};
pub(super) struct Packed {
    pub(super) metadata: Metadata,
}
fn error(detail: impl std::fmt::Display) -> EpError {
    EpError::KernelFailed(format!("{PACKED_OP}: {detail}"))
}
impl Packed {
    fn plan(l: GroupLayout, row_elements: usize) -> Result<ValueIndexWorkspace<f32, 2>> {
        ValueIndexWorkspace::new(
            PACKED_OP,
            [("Y staging", l.lanes), ("activation row", row_elements)],
            ("routing indices", l.routes),
        )
    }
    fn run(
        &self,
        x: &[TensorView],
        y: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let l = packed_group_layout(&self.metadata, x, y)?;
        // Preserve Packed's read-free early return, including invalid route
        // values and device pointers that will never be accessed.
        if l.lanes == 0 {
            return Ok(());
        }
        if !x[1].is_contiguous() {
            return Err(error(
                "packed weights must be contiguous; bind a contiguous encoded bank",
            ));
        }
        validate_host_storage::<u8>(PACKED_OP, "packed weights", &x[1])?;
        let route_source = RouteSource::new(Op::GroupedMatMul { transposed: true }, &x[2])
            .map_err(|e| error(format!("group_indices input 2: {e}")))?;
        let out = &y[0];
        let view = TensorView::new(
            DevicePtr(out.data.0.cast_const()),
            out.dtype,
            out.shape,
            out.strides,
            out.device,
        )
        .with_byte_offset(out.byte_offset);
        validate_host_storage::<f32>(PACKED_OP, "Y", &view)?;
        let rows = if l.k == 0 {
            validate_host_storage::<f32>(PACKED_OP, "X", &x[0])?;
            None
        } else {
            Some(Rows::new(PACKED_OP, "X", &x[0])?)
        };
        let plan = Self::plan(l, rows.as_ref().map_or(0, Rows::scratch_len))?;
        // SAFETY: the execution owner lends exclusive Host backing disjoint
        // from tensors. Complete binding/initialization precedes value reads.
        unsafe {
            plan.with_buffers(workspace, |[output, row_scratch], routes| {
                route_source
                    .copy_into(l.groups, routes)
                    .map_err(|e| error(format!("group_indices input 2: {e}")))?;
                if let Some(rows) = rows {
                    // SAFETY: validated contiguous UINT8 bank with exact metadata
                    // extent, Host/span proof and owner-provided live backing.
                    let packed = std::slice::from_raw_parts(
                        x[1].data_ptr::<u8>(),
                        self.metadata.packed_bytes(),
                    );
                    let (width, bytes) = self.metadata.format().block();
                    let blocks = l.k / width;
                    let decode = crate::kernels::dequantize_packed::decoder(self.metadata.format());
                    let mut values = [0f32; 256];
                    for (token, outputs) in output.chunks_exact_mut(l.selections * l.n).enumerate()
                    {
                        let activation = rows.get(token, row_scratch)?;
                        for (selection, row) in outputs.chunks_exact_mut(l.n).enumerate() {
                            let expert = routes[token * l.selections + selection];
                            for (column, out) in row.iter_mut().enumerate() {
                                let packed_row = (expert * l.n + column) * blocks * bytes;
                                let mut sum = 0f32;
                                for block in 0..blocks {
                                    let source = packed_row + block * bytes;
                                    decode(&packed[source..source + bytes], &mut values[..width]);
                                    let activation =
                                        &activation[block * width..(block + 1) * width];
                                    for (&a, &weight) in activation.iter().zip(&values[..width]) {
                                        sum += a * weight;
                                    }
                                }
                                *out = sum;
                            }
                        }
                    }
                }
                // K==0 reduces to initialized +0 after route validation. No X or
                // bank values are read, and expert*N is never computed needlessly.
                crate::dtype::write_dense(&mut y[0], output)
                    .map_err(|e| error(format!("Y publication: {e}")))
            })
        }
    }
}
impl Kernel for Packed {
    fn supports_strided_input(&self, index: usize) -> bool {
        matches!(index, 0 | 2)
    }
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        let l = packed_group_metadata_layout(&self.metadata, inputs, outputs)?;
        if l.lanes == 0 {
            Ok(WorkspacePreparation::Ready(WorkspaceRequirement::NONE))
        } else if l.k == 0 {
            Ok(WorkspacePreparation::Ready(Self::plan(l, 0)?.requirement()))
        } else {
            // Shape/dtype cannot prove final-origin alignment or row strides.
            Self::plan(l, l.k)?;
            Ok(WorkspacePreparation::Deferred)
        }
    }
    fn workspace_requirement_for_execution(
        &self,
        inputs: &[TensorView],
        outputs: &[TensorMut],
        metadata: &[TensorMetadata<'_>],
        facts: &[OutputMetadata<'_>],
        _: Option<WorkspacePreparation>,
    ) -> Result<WorkspaceExecutionPreparation> {
        let l = packed_group_metadata_layout(&self.metadata, metadata, facts)?;
        if inputs.len() != metadata.len()
            || inputs.iter().zip(metadata).any(|(view, meta)| {
                view.dtype != meta.dtype
                    || view.shape != meta.shape
                    || (!view.is_absent() != meta.present && !view.shape.contains(&0))
            })
        {
            return Err(error(
                "input views differ from workspace metadata; refresh metadata for the actual bindings",
            ));
        }
        if !outputs.is_empty() {
            packed_group_layout(&self.metadata, inputs, outputs)?;
        }
        if l.lanes == 0 {
            return Ok(WorkspaceExecutionPreparation::Ready(
                WorkspaceRequirement::NONE,
            ));
        }
        let row_elements = if l.k == 0 {
            validate_host_storage::<f32>(PACKED_OP, "X", &inputs[0])?;
            0
        } else {
            Rows::new(PACKED_OP, "X", &inputs[0])?.scratch_len()
        };
        Ok(WorkspaceExecutionPreparation::Ready(
            Self::plan(l, row_elements)?.requirement(),
        ))
    }
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }
    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.run(inputs, outputs, workspace)
    }
}

#[cfg(test)]
mod tests;
