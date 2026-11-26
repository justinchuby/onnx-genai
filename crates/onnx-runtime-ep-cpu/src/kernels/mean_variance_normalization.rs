//! MeanVarianceNormalization-29 with address-refined, whole-call staging.

use super::{norm_ops::NormAcc, typed_workspace::TypedWorkspace};
use crate::{
    dispatch_float,
    dtype::{
        ComputeDomain, NumericElem, byte_ranges_overlap, to_dense_into, validate_f32_widen_input,
        validate_host_output, write_dense,
    },
};
use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, OutputMetadata, Result, TensorMetadata, TensorMut, TensorView,
    WorkspaceExecutionPreparation, WorkspacePreparation, WorkspaceRequirement, WorkspaceView,
    mean_variance_normalization as mvn,
};
use onnx_runtime_ir::{DataType, Node};

pub struct MeanVarianceNormalizationFactory;
impl KernelFactory for MeanVarianceNormalizationFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(MeanVarianceNormalization(mvn::Attributes::parse(
            node,
        )?)))
    }
}
struct MeanVarianceNormalization(mvn::Attributes);
struct Geometry {
    layout: mvn::Layout,
    dtype: DataType,
    count: usize,
    full: WorkspaceRequirement,
}
impl MeanVarianceNormalization {
    fn geometry(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<Geometry> {
        let fail = |why: String| EpError::KernelFailed(format!("{}: {why}", mvn::OP));
        let ([x], [y]) = (inputs, outputs) else {
            return Err(fail(format!(
                "expected one present X and one Y, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        };
        if !x.present || !mvn::DTYPES.contains(&x.dtype) {
            return Err(fail(format!(
                "X must be a present floating tensor, got present={} dtype {:?}",
                x.present, x.dtype
            )));
        }
        match y {
            OutputMetadata::Tensor { dtype, shape }
                if *dtype == x.dtype && shape.as_ref() == x.shape => {}
            OutputMetadata::Unresolved { dtype } if *dtype == x.dtype => {}
            _ => {
                return Err(fail(format!(
                    "Y {y:?} must match X dtype {:?} and complete shape {:?}; bind a matching output",
                    x.dtype, x.shape
                )));
            }
        }
        let count = x
            .shape
            .iter()
            .rev()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| {
                fail(format!(
                    "X shape {:?} overflows dense indexing; reduce dimensions",
                    x.shape
                ))
            })?;
        let full = dispatch_float!(x.dtype,mvn::OP,T=>Ok(TypedWorkspace::<T,1>::new(mvn::OP,[("X/Y snapshot",count)])?.requirement()))?;
        Ok(Geometry {
            layout: self.0.layout(x.shape)?,
            dtype: x.dtype,
            count,
            full,
        })
    }
    fn direct(x: &TensorView, y: &TensorMut, bytes: usize) -> Result<bool> {
        validate_f32_widen_input(mvn::OP, x).map_err(|error| {
            EpError::KernelFailed(format!(
                "{}: invalid X view: {error}; bind valid floating Host-accessible storage",
                mvn::OP
            ))
        })?;
        validate_host_output(mvn::OP, y)?;
        let range = |name: &str,
                     base: usize,
                     offset: usize,
                     contiguous: bool|
         -> Result<Option<std::ops::Range<usize>>> {
            let start = base.checked_add(offset).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "{}: {name} element origin wraps; bind valid Host storage",
                    mvn::OP
                ))
            })?;
            if !contiguous {
                return Ok(None);
            }
            let end=start.checked_add(bytes).ok_or_else(||EpError::KernelFailed(format!("{}: {name} address range wraps; bind valid Host storage for the complete tensor",mvn::OP)))?;
            Ok(Some(start..end))
        };
        let x = range("X", x.data.0 as usize, x.byte_offset, x.is_contiguous())?;
        let y = range("Y", y.data.0 as usize, y.byte_offset, y.is_contiguous())?;
        Ok(matches!((x,y),(Some(x),Some(y)) if x.start==y.start || !byte_ranges_overlap(&x,&y)))
    }
    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let metadata = inputs
            .iter()
            .map(|x| TensorMetadata::new(x.dtype, x.shape, !x.is_absent()))
            .collect::<smallvec::SmallVec<[_; 1]>>();
        let facts = outputs
            .iter()
            .map(OutputMetadata::from_view)
            .collect::<smallvec::SmallVec<[_; 1]>>();
        let geometry = self.geometry(&metadata, &facts)?;
        let direct = Self::direct(&inputs[0], &outputs[0], geometry.full.bytes as usize)?;
        if geometry.count == 0 {
            return Ok(());
        }
        dispatch_float!(geometry.dtype,mvn::OP,T=>self.run_typed::<T>(&inputs[0],&mut outputs[0],&geometry,direct,workspace))
    }
    fn run_typed<T: NumericElem + Default>(
        &self,
        x: &TensorView,
        y: &mut TensorMut,
        geometry: &Geometry,
        direct: bool,
        workspace: Option<WorkspaceView>,
    ) -> Result<()>
    where
        T::Acc: NormAcc,
    {
        if direct {
            // SAFETY: validated contiguous Host views are either disjoint or
            // exact-inplace. Every group completes its statistics before any
            // of its unique coordinates is overwritten; groups are disjoint.
            unsafe {
                normalize::<T>(
                    x.data_ptr::<T>(),
                    y.data_ptr_mut::<T>(),
                    &geometry.layout,
                    self.0.epsilon,
                )
            };
            return Ok(());
        }
        let plan = TypedWorkspace::<T, 1>::new(mvn::OP, [("X/Y snapshot", geometry.count)])?;
        // SAFETY: the synchronous executor lends exclusive live Host backing,
        // disjoint from all tensors. Complete binding validation comes first.
        unsafe {
            plan.with_buffers(workspace, |[snapshot]| {
                to_dense_into::<T>(mvn::OP, x, snapshot)?;
                let pointer = snapshot.as_mut_ptr();
                normalize::<T>(
                    pointer.cast_const(),
                    pointer,
                    &geometry.layout,
                    self.0.epsilon,
                );
                write_dense::<T>(y, snapshot)
            })
        }
    }
}
impl Kernel for MeanVarianceNormalization {
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        let geometry = self.geometry(inputs, outputs)?;
        Ok(if geometry.count == 0 {
            WorkspacePreparation::Ready(WorkspaceRequirement::NONE)
        } else {
            WorkspacePreparation::Deferred
        })
    }
    fn workspace_requirement_for_execution(
        &self,
        inputs: &[TensorView],
        outputs: &[TensorMut],
        metadata: &[TensorMetadata<'_>],
        facts: &[OutputMetadata<'_>],
        _cached: Option<WorkspacePreparation>,
    ) -> Result<WorkspaceExecutionPreparation> {
        let geometry = self.geometry(metadata, facts)?;
        if let ([x], [y]) = (inputs, outputs) {
            if x.dtype != geometry.dtype
                || y.dtype != geometry.dtype
                || x.shape != metadata[0].shape
                || y.shape != metadata[0].shape
                || x.is_absent()
                || y.is_absent()
            {
                return Err(EpError::KernelFailed(format!(
                    "{}: actual X/Y shapes {:?}/{:?} and dtypes {:?}/{:?} differ from planned dtype {:?} shape {:?}; refresh metadata for the actual bindings",
                    mvn::OP,
                    x.shape,
                    y.shape,
                    x.dtype,
                    y.dtype,
                    geometry.dtype,
                    metadata[0].shape
                )));
            }
            let direct = Self::direct(x, y, geometry.full.bytes as usize)?;
            if geometry.count == 0 || direct {
                return Ok(WorkspaceExecutionPreparation::Ready(
                    WorkspaceRequirement::NONE,
                ));
            }
        }
        // Without complete actual views, neither contiguity nor alias safety
        // is proven. Retain the complete conservative snapshot bound.
        Ok(WorkspaceExecutionPreparation::Ready(
            if geometry.count == 0 {
                WorkspaceRequirement::NONE
            } else {
                geometry.full
            },
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
    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

/// # Safety
/// `layout` describes complete valid dense T storage at both pointers; Host
/// backing remains live and writable for Y. Ranges are disjoint or origins
/// identical, and layout partitions coordinates into disjoint reduction groups.
unsafe fn normalize<T: NumericElem>(
    x_ptr: *const T,
    y_ptr: *mut T,
    layout: &mvn::Layout,
    epsilon: f32,
) where
    T::Acc: NormAcc,
{
    let epsilon = T::Acc::from_f32(epsilon);
    for group in 0..layout.groups {
        let base = mvn::offset(group, &layout.kept);
        let index = |i| base + mvn::offset(i, &layout.reduced);
        // SAFETY: each index denotes one in-bounds dense element. Statistics
        // finish before any output write; exact-inplace reads remain valid.
        let load = |i| unsafe { x_ptr.add(index(i)).read_unaligned() }.to_acc();
        let mut sum = T::Acc::default();
        for i in 0..layout.width {
            sum = sum.c_add(load(i));
        }
        let mean = sum.c_div_usize(layout.width);
        let mut squared = T::Acc::default();
        for i in 0..layout.width {
            let delta = load(i).c_sub(mean);
            squared = squared.c_add(delta.c_mul(delta));
        }
        let denominator = squared.c_div_usize(layout.width).sqrt().c_add(epsilon);
        for i in 0..layout.width {
            let offset = index(i);
            let value = T::from_acc(load(i).c_sub(mean).c_div(denominator));
            // SAFETY: same dense bounds as the load above.
            unsafe { y_ptr.add(offset).write_unaligned(value) };
        }
    }
}

#[cfg(test)]
mod tests;
