//! ONNX `LpNormalization` along one axis.

use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, OutputMetadata, Result, TensorMetadata, TensorMut, TensorView,
    WorkspacePreparation, WorkspaceView,
};
use onnx_runtime_ir::Node;

use super::typed_workspace::TypedWorkspace;
use crate::dispatch_float;
use crate::dtype::{ComputeDomain, NumericElem, to_dense, write_dense};

trait LpAcc: ComputeDomain + PartialOrd {
    const TINY: Self;

    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
}

impl LpAcc for f32 {
    const TINY: Self = f32::MIN_POSITIVE;

    fn abs(self) -> Self {
        self.abs()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }
}

impl LpAcc for f64 {
    const TINY: Self = f64::MIN_POSITIVE;

    fn abs(self) -> Self {
        self.abs()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }
}

pub struct LpNormalizationFactory;

impl KernelFactory for LpNormalizationFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let p = node.attr("p").and_then(|a| a.as_int()).unwrap_or(2);
        if p != 1 && p != 2 {
            return Err(EpError::KernelFailed(format!(
                "LpNormalization: p must be 1 or 2, got {p}"
            )));
        }
        Ok(Box::new(LpNormalizationKernel {
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1),
            p: p as u8,
        }))
    }
}

pub struct LpNormalizationKernel {
    axis: i64,
    p: u8,
}

struct Geometry {
    outer: usize,
    axis_len: usize,
    inner: usize,
    count: usize,
}
impl LpNormalizationKernel {
    fn geometry(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<Geometry> {
        let fail = |why: String| EpError::KernelFailed(format!("LpNormalization: {why}"));
        let ([x], [y]) = (inputs, outputs) else {
            return Err(fail(format!(
                "expected exactly one X and one Y, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        };
        if !x.present || !super::FLOAT_DTYPES.contains(&x.dtype) {
            return Err(fail(format!(
                "X must be present floating storage, got present={} dtype {:?}",
                x.present, x.dtype
            )));
        }
        match y {
            OutputMetadata::Tensor { dtype, shape }
                if *dtype == x.dtype && shape.as_ref() == x.shape => {}
            OutputMetadata::Unresolved { dtype } if *dtype == x.dtype => {}
            _ => {
                return Err(fail(format!(
                    "Y {y:?} must be present and match X shape {:?} and dtype {:?}; bind a matching output",
                    x.shape, x.dtype
                )));
            }
        }
        if !matches!(self.p, 1 | 2) {
            return Err(fail(format!("p={} must be 1 or 2", self.p)));
        }
        let rank = x.shape.len();
        let axis = if self.axis < 0 {
            self.axis.checked_add(rank as i64)
        } else {
            Some(self.axis)
        }
        .filter(|&a| a >= 0 && (a as usize) < rank)
        .ok_or_else(|| {
            fail(format!(
                "axis {} out of range for rank {rank}; provide rank >= 1 and axis in [-rank, rank)",
                self.axis
            ))
        })? as usize;
        let product = |dims: &[usize]| {
            dims.iter()
                .rev()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| {
                    fail(format!(
                        "X shape {:?} overflows dense indexing; reduce dimensions",
                        x.shape
                    ))
                })
        };
        Ok(Geometry {
            outer: product(&x.shape[..axis])?,
            axis_len: x.shape[axis],
            inner: product(&x.shape[axis + 1..])?,
            count: product(x.shape)?,
        })
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
        crate::dtype::validate_f32_widen_input("LpNormalization", &inputs[0]).map_err(|e| {
            EpError::KernelFailed(format!(
                "LpNormalization: invalid X view: {e}; bind valid floating Host-accessible storage"
            ))
        })?;
        crate::dtype::validate_host_output("LpNormalization", &outputs[0])?;
        dispatch_float!(inputs[0].dtype,"LpNormalization",T => lp_normalization_typed::<T>(inputs,outputs,&geometry,self.p,workspace))
    }
}
impl Kernel for LpNormalizationKernel {
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        let geometry = self.geometry(inputs, outputs)?;
        dispatch_float!(inputs[0].dtype,"LpNormalization",T => Ok(WorkspacePreparation::Ready(TypedWorkspace::<T,1>::new("LpNormalization",[("Y staging",geometry.count)])?.requirement())))
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

fn lp_normalization_typed<T: NumericElem + Default>(
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
    geometry: &Geometry,
    p: u8,
    workspace: Option<WorkspaceView>,
) -> Result<()>
where
    T::Acc: LpAcc,
{
    let input = &inputs[0];
    let output = &mut outputs[0];
    let Geometry {
        outer,
        axis_len,
        inner,
        count,
    } = *geometry;
    let plan = TypedWorkspace::<T, 1>::new("LpNormalization", [("Y staging", count)])?;
    // SAFETY: the synchronous executor lends exclusive live Host storage,
    // disjoint from all tensor storage. Validate the whole binding first.
    unsafe {
        plan.with_buffers(workspace, |[y]| {
            if count == 0 {
                return Ok(());
            }
            let x = to_dense::<T>(input)?;
            for outer_index in 0..outer {
                for inner_index in 0..inner {
                    let mut norm = T::Acc::default();
                    for axis_index in 0..axis_len {
                        let index = (outer_index * axis_len + axis_index) * inner + inner_index;
                        let value = x[index].to_acc().abs();
                        norm = norm.c_add(if p == 1 { value } else { value.c_mul(value) });
                    }
                    if p == 2 {
                        norm = norm.sqrt();
                    }
                    if norm < T::Acc::TINY {
                        norm = T::Acc::TINY;
                    }
                    for axis_index in 0..axis_len {
                        let index = (outer_index * axis_len + axis_index) * inner + inner_index;
                        y[index] = T::from_acc(x[index].to_acc().c_div(norm));
                    }
                }
            }

            write_dense::<T>(output, y)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::{Owned, PreparedTestKernel};
    use onnx_runtime_ir::NodeId;

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "index {index}: got {got}, want {want}"
            );
        }
    }

    fn run(shape: &[usize], values: &[f32], axis: i64, p: u8) -> Vec<f32> {
        let x = Owned::f32(shape, values);
        let mut y = Owned::zeros_f32(shape);
        LpNormalizationKernel { axis, p }
            .execute_prepared(&[x.view()], &mut [y.view_mut()])
            .unwrap();
        y.to_f32()
    }

    #[test]
    fn lp_normalization_bf16_matches_widened_f32_reference() {
        let values = [1.0f32, -2.0, 3.0, 4.0, -5.0, 6.0];
        let reference = run(&[2, 3], &values, 1, 2);
        let x = Owned::bf16(&[2, 3], &values);
        let mut y = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[2, 3]);
        LpNormalizationKernel { axis: 1, p: 2 }
            .execute_prepared(&[x.view()], &mut [y.view_mut()])
            .unwrap();
        for (&r, &g) in reference.iter().zip(y.to_bf16_as_f32().iter()) {
            assert!(
                (r - g).abs() <= 0.03 * r.abs().max(1.0),
                "lp_normalization bf16 {g} vs f32 {r}"
            );
        }
    }

    #[test]
    fn l1_normalization_axes_zero_one_and_last() {
        let values = [1.0, -2.0, 3.0, 4.0, -5.0, 6.0];
        assert_close(
            &run(&[2, 3], &values, 0, 1),
            &[0.2, -2.0 / 7.0, 1.0 / 3.0, 0.8, -5.0 / 7.0, 2.0 / 3.0],
        );
        let row_norm = 6.0;
        let row2_norm = 15.0;
        let expected = [
            1.0 / row_norm,
            -2.0 / row_norm,
            3.0 / row_norm,
            4.0 / row2_norm,
            -5.0 / row2_norm,
            6.0 / row2_norm,
        ];
        assert_close(&run(&[2, 3], &values, 1, 1), &expected);
        assert_close(&run(&[2, 3], &values, -1, 1), &expected);
    }

    #[test]
    fn l2_normalization_axes_zero_one_and_default_last() {
        let values = [3.0, 4.0, 0.0, 0.0, 0.0, 12.0];
        assert_close(
            &run(&[2, 3], &values, 0, 2),
            &[1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
        let expected = [0.6, 0.8, 0.0, 0.0, 0.0, 1.0];
        assert_close(&run(&[2, 3], &values, 1, 2), &expected);
        let x = Owned::f32(&[2, 3], &values);
        let mut y = Owned::zeros_f32(&[2, 3]);
        let node = Node::new(NodeId(0), "LpNormalization", vec![], vec![]);
        LpNormalizationFactory
            .create(&node, &[])
            .unwrap()
            .execute_prepared(&[x.view()], &mut [y.view_mut()])
            .unwrap();
        assert_close(&y.to_f32(), &expected);
    }
}

#[cfg(test)]
mod workspace_tests;
