//! Native grouped arithmetic over one admitted snapshot/output lane.
use super::RouteSource;
use crate::{dtype, kernels::typed_workspace::ValueIndexWorkspace, strided::validate_host_storage};
use onnx_runtime_ep_api::{
    DevicePtr, Kernel, OutputMetadata, Result, TensorMetadata, TensorMut, TensorView,
    WorkspaceExecutionPreparation, WorkspacePreparation, WorkspaceRequirement, WorkspaceView,
    packed_routed::{GroupLayout, fail, group_layout, group_metadata_layout, validate},
};
use onnx_runtime_ir::{DataType, packed_routed::Op};

pub(super) struct Native(pub(super) Op);

trait Accumulator:
    dtype::NumericElem + Default + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self>
{
    fn can_borrow_dtype(dtype: DataType) -> bool {
        dtype == Self::DTYPE
    }
    fn borrows(view: &TensorView) -> bool {
        // A native representation needs no conversion. The caller separately
        // establishes Host access and the full checked address span.
        !view.shape.contains(&0) && dtype::can_borrow_dense::<Self>(view)
    }
    fn copy(op: &str, view: &TensorView, target: &mut [Self]) -> Result<()>;
    fn publish(op: &str, out: &mut TensorMut, data: &[Self]) -> Result<()>;
    fn round(value: Self, dtype: DataType) -> Self;
}
impl Accumulator for f32 {
    fn copy(op: &str, view: &TensorView, target: &mut [Self]) -> Result<()> {
        dtype::to_dense_f32_widen_into(op, view, target)
    }
    fn publish(op: &str, out: &mut TensorMut, data: &[Self]) -> Result<()> {
        dtype::write_dense_f32_narrow(op, out, data)
    }
    fn round(value: Self, dtype: DataType) -> Self {
        match dtype {
            DataType::Float16 => half::f16::from_f32(value).to_f32(),
            DataType::BFloat16 => half::bf16::from_f32(value).to_f32(),
            _ => value,
        }
    }
}
impl Accumulator for f64 {
    fn copy(op: &str, view: &TensorView, target: &mut [Self]) -> Result<()> {
        dtype::to_dense_into(op, view, target)
    }
    fn publish(_: &str, out: &mut TensorMut, data: &[Self]) -> Result<()> {
        dtype::write_dense(out, data)
    }
    fn round(value: Self, _: DataType) -> Self {
        value
    }
}

fn count(op: Op, view: &TensorMetadata) -> Result<usize> {
    if view.shape.contains(&0) {
        return Ok(0);
    }
    view.shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| {
            fail(
                op,
                format!(
                    "shape {:?} exceeds snapshot capacity; reduce tensor dimensions",
                    view.shape
                ),
            )
        })
}
fn plan<F: Accumulator>(
    op: Op,
    l: GroupLayout,
    inputs: &[TensorMetadata],
    borrow: [bool; 3],
) -> Result<ValueIndexWorkspace<F, 4>> {
    let mut sizes = [0; 3];
    if l.lanes != 0 {
        for (slot, index) in [0, 1, 3].into_iter().enumerate() {
            if !borrow[slot] && (index != 3 || l.bias) {
                sizes[slot] = count(op, &inputs[index])?;
            }
        }
    }
    ValueIndexWorkspace::new(
        op.name(),
        [
            ("Y staging", l.lanes),
            ("X snapshot", sizes[0]),
            ("weight snapshot", sizes[1]),
            ("bias snapshot", sizes[2]),
        ],
        ("routing indices", l.routes),
    )
}
fn metadata<'a>(inputs: &'a [TensorView<'_>]) -> [TensorMetadata<'a>; 4] {
    let mut result = [TensorMetadata::new(DataType::Undefined, &[], false); 4];
    for (slot, view) in result.iter_mut().zip(inputs) {
        *slot = TensorMetadata::new(view.dtype, view.shape, !view.is_absent());
    }
    result
}
fn host(op: Op, name: &str, view: &TensorView) -> Result<()> {
    match view.dtype {
        DataType::Float32 => validate_host_storage::<f32>(op.name(), name, view),
        DataType::Float64 => validate_host_storage::<f64>(op.name(), name, view),
        DataType::Float16 => validate_host_storage::<half::f16>(op.name(), name, view),
        DataType::BFloat16 => validate_host_storage::<half::bf16>(op.name(), name, view),
        _ => {
            return Err(fail(
                op,
                format!(
                    "{name} dtype {:?} is unsupported; bind matching floating-point operands",
                    view.dtype
                ),
            ));
        }
    }
    .map(|_| ())
}
fn borrow_flags<F: Accumulator>(l: GroupLayout, inputs: &[TensorView]) -> [bool; 3] {
    if l.lanes == 0 {
        return [false; 3];
    }
    [
        F::borrows(&inputs[0]),
        F::borrows(&inputs[1]),
        l.bias && F::borrows(&inputs[3]),
    ]
}
fn source<'a, F: Accumulator>(
    op: Op,
    view: &'a TensorView,
    scratch: &'a mut [F],
) -> Result<&'a [F]> {
    // Empty geometry must be resolved before helpers multiply dimensions.
    if view.shape.contains(&0) {
        return Ok(&[]);
    }
    if F::borrows(view) {
        // SAFETY: borrowing requires aligned contiguous matching storage;
        // Host/span proofs precede binding, and Y is published after all reads.
        Ok(unsafe { std::slice::from_raw_parts(view.data_ptr::<F>(), view.numel()) })
    } else {
        F::copy(op.name(), view, scratch)?;
        Ok(scratch)
    }
}

// Arithmetic order is the original Native grouped implementation, writing into
// admitted output instead of returning a payload Vec.
fn grouped<F: Accumulator>(
    l: GroupLayout,
    x: &[F],
    weights: &[F],
    groups: &[usize],
    b: Option<&[F]>,
    dtype: DataType,
    y: &mut [F],
) {
    for (route, row) in y.chunks_exact_mut(l.n).enumerate() {
        let g = groups[route];
        let token = route / l.selections;
        for (n, out) in row.iter_mut().enumerate() {
            let mut sum = F::default();
            for k in 0..l.k {
                let i = if l.transposed {
                    n * l.k + k
                } else {
                    k * l.n + n
                };
                sum = sum + x[token * l.k + k] * weights[g * l.k * l.n + i];
            }
            // ONNX stores MatMul before Add; half/bf16 must round here.
            *out = if let Some(b) = b {
                F::round(sum, dtype) + b[g * l.n + n]
            } else {
                sum
            };
        }
    }
}
impl Native {
    fn validate_sources(&self, inputs: &[TensorView], l: GroupLayout) -> Result<()> {
        if l.lanes != 0 {
            for (i, name) in [(0, "X"), (1, "weights"), (3, "bias")] {
                if i != 3 || l.bias {
                    host(self.0, name, &inputs[i])?;
                }
            }
        }
        Ok(())
    }
    fn run_typed<F: Accumulator>(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        l: GroupLayout,
        routes: RouteSource,
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let op = self.0;
        let m = metadata(inputs);
        let plan = plan::<F>(op, l, &m[..inputs.len()], borrow_flags::<F>(l, inputs))?;
        // SAFETY: owner lends live exclusive Host scratch disjoint from tensors.
        // The complete mixed layout binds before any initialization/value reads.
        unsafe {
            plan.with_buffers(workspace, |[y, x, w, b], groups| {
                routes.copy_into(l.groups, groups)?;
                if l.lanes == 0 {
                    return Ok(());
                }
                let bias = if l.bias {
                    Some(source::<F>(op, &inputs[3], b)?)
                } else {
                    None
                };
                let weights = source::<F>(op, &inputs[1], w)?;
                let x = source::<F>(op, &inputs[0], x)?;
                grouped(l, x, weights, groups, bias, inputs[0].dtype, y);
                F::publish(op.name(), &mut outputs[0], y)
                    .map_err(|e| fail(op, format!("Y publication: {e}")))
            })
        }
    }
    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        validate(self.0, inputs, outputs)?;
        let l = group_layout(self.0, inputs, &outputs[0])?;
        let routes = RouteSource::new(self.0, &inputs[2])?;
        self.validate_sources(inputs, l)?;
        if l.lanes != 0 {
            let out = &outputs[0];
            let view = TensorView::new(
                DevicePtr(out.data.0.cast_const()),
                out.dtype,
                out.shape,
                out.strides,
                out.device,
            )
            .with_byte_offset(out.byte_offset);
            host(self.0, "Y", &view)?;
        }
        if inputs[0].dtype == DataType::Float64 {
            self.run_typed::<f64>(inputs, outputs, l, routes, workspace)
        } else {
            self.run_typed::<f32>(inputs, outputs, l, routes, workspace)
        }
    }
}
impl Kernel for Native {
    fn supports_strided_input(&self, index: usize) -> bool {
        index < 4
    }
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        let l = group_metadata_layout(self.0, inputs, outputs)?;
        fn prepare<F: Accumulator>(
            op: Op,
            l: GroupLayout,
            inputs: &[TensorMetadata],
        ) -> Result<WorkspacePreparation> {
            if l.lanes != 0 && F::can_borrow_dtype(inputs[0].dtype) {
                Ok(WorkspacePreparation::Deferred)
            } else {
                Ok(WorkspacePreparation::Ready(
                    plan::<F>(op, l, inputs, [false; 3])?.requirement(),
                ))
            }
        }
        if inputs[0].dtype == DataType::Float64 {
            prepare::<f64>(self.0, l, inputs)
        } else {
            prepare::<f32>(self.0, l, inputs)
        }
    }

    fn workspace_requirement_for_execution(
        &self,
        inputs: &[TensorView],
        outputs: &[TensorMut],
        facts: &[TensorMetadata<'_>],
        out_facts: &[OutputMetadata<'_>],
        _: Option<WorkspacePreparation>,
    ) -> Result<WorkspaceExecutionPreparation> {
        let l = group_metadata_layout(self.0, facts, out_facts)?;
        if inputs.len() != facts.len()
            || inputs.iter().zip(facts).enumerate().any(|(i, (v, m))| {
                v.dtype != m.dtype
                    || v.shape != m.shape
                    || (i == 3 && v.is_absent() == m.present)
                    || (i < 3 && v.is_absent() && !v.shape.contains(&0))
            })
        {
            return Err(fail(
                self.0,
                "input views differ from workspace metadata; refresh metadata for the actual bindings",
            ));
        }
        if !outputs.is_empty() {
            validate(self.0, inputs, outputs)?;
            group_layout(self.0, inputs, &outputs[0])?;
        }
        RouteSource::new(self.0, &inputs[2])?;
        self.validate_sources(inputs, l)?;
        let req: WorkspaceRequirement = if inputs[0].dtype == DataType::Float64 {
            plan::<f64>(self.0, l, facts, borrow_flags::<f64>(l, inputs))?.requirement()
        } else {
            plan::<f32>(self.0, l, facts, borrow_flags::<f32>(l, inputs))?.requirement()
        };
        Ok(WorkspaceExecutionPreparation::Ready(req))
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
