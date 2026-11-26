//! Whole-operation affine grouped workspace; no backing owner or per-route grant.
use super::RouteSource;
use crate::kernels::{
    affine_bank::{self, AffineBank},
    host_rows::Rows,
    typed_workspace::ValueIndexWorkspace,
};
use onnx_runtime_ep_api::{
    DevicePtr, EpError, Kernel, OutputMetadata, Result, TensorMetadata, TensorMut, TensorView,
    WorkspaceExecutionPreparation, WorkspacePreparation, WorkspaceView,
    packed_routed::{AffineGrouped, GroupLayout},
};
use onnx_runtime_ir::{packed_routed::Op, quantized_grouped::OP};
pub(super) struct Affine(pub(super) AffineGrouped);
fn error(detail: impl std::fmt::Display) -> EpError {
    EpError::KernelFailed(format!("{OP}: {detail}"))
}
impl Affine {
    fn plan(&self, l: GroupLayout, row_elements: usize) -> Result<ValueIndexWorkspace<f32, 2>> {
        ValueIndexWorkspace::new(
            OP,
            [("Y staging", l.lanes), ("activation row", row_elements)],
            ("routing indices", l.routes),
        )
    }
    fn geometry(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<GroupLayout> {
        let l = self.0.metadata_layout(inputs, outputs)?;
        affine_bank::validate_metadata(
            &inputs[1],
            &inputs[3],
            inputs.get(4),
            [l.groups, l.n, l.k],
            self.0.block,
            self.0.zero,
        )
        .map_err(|e| error(format!("expert bank inputs 1/3/4: {e}")))?;
        Ok(l)
    }
    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let l = self.0.layout(inputs, outputs)?;
        let op = Op::GroupedMatMul { transposed: true };
        let route_source = RouteSource::new(op, &inputs[2])
            .map_err(|e| error(format!("group_indices input 2: {e}")))?;
        let bank = AffineBank::prepare(
            &inputs[1],
            &inputs[3],
            inputs.get(4),
            [l.groups, l.n, l.k],
            self.0.block,
            self.0.zero,
        )
        .map_err(|e| error(format!("expert bank inputs 1/3/4: {e}")))?;
        // Empty affine execution still validates every route and scale, but
        // never reads X or writes Y. Their Host access is only needed otherwise.
        let rows = if l.lanes == 0 {
            None
        } else {
            let out = &outputs[0];
            let view = TensorView::new(
                DevicePtr(out.data.0.cast_const()),
                out.dtype,
                out.shape,
                out.strides,
                out.device,
            )
            .with_byte_offset(out.byte_offset);
            crate::strided::validate_host_storage::<f32>(OP, "Y", &view)?;
            Some(Rows::new(OP, "X", &inputs[0])?)
        };
        let plan = self.plan(l, rows.as_ref().map_or(0, Rows::scratch_len))?;
        // SAFETY: the owner supplies exclusive Host scratch disjoint from all
        // tensors. Complete mixed binding precedes every source value read.
        unsafe {
            plan.with_buffers(workspace, |[y, row_scratch], routes| {
                route_source
                    .copy_into(l.groups, routes)
                    .map_err(|e| error(format!("group_indices input 2: {e}")))?;
                let bank = bank
                    .validate()
                    .map_err(|e| error(format!("expert bank inputs 1/3/4: {e}")))?;
                let Some(rows) = rows else {
                    return Ok(());
                };
                for (token, outputs) in y.chunks_exact_mut(l.selections * l.n).enumerate() {
                    let input = rows.get(token, row_scratch)?;
                    for (selection, output) in outputs.chunks_exact_mut(l.n).enumerate() {
                        let expert = routes[token * l.selections + selection];
                        for (column, value) in output.iter_mut().enumerate() {
                            *value = bank.dot(expert * l.n + column, input);
                        }
                    }
                }
                crate::dtype::write_dense(&mut outputs[0], y)
                    .map_err(|e| error(format!("Y publication: {e}")))
            })
        }
    }
}
impl Kernel for Affine {
    fn supports_strided_input(&self, index: usize) -> bool {
        index < 5
    }
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        let l = self.geometry(inputs, outputs)?;
        if l.lanes == 0 {
            Ok(WorkspacePreparation::Ready(self.plan(l, 0)?.requirement()))
        } else {
            // Shape/dtype metadata cannot prove row alignment or contiguity.
            self.plan(l, l.k)?;
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
        let l = self.geometry(metadata, facts)?;
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
            self.0.layout(inputs, outputs)?;
        }
        // Descriptors only: complete arena admission still precedes all route,
        // scale and activation reads in run. Empty Y never needs X Host access.
        let row_elements = if l.lanes == 0 {
            0
        } else {
            Rows::new(OP, "X", &inputs[0])?.scratch_len()
        };
        Ok(WorkspaceExecutionPreparation::Ready(
            self.plan(l, row_elements)?.requirement(),
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
