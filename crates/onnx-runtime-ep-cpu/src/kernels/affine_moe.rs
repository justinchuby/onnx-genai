use onnx_runtime_ep_api::{
    DevicePtr, EpError, Kernel, KernelFactory, OutputMetadata, Result, TensorMetadata, TensorMut,
    TensorView, WorkspaceExecutionPreparation, WorkspacePreparation, WorkspaceView,
};
use onnx_runtime_ir::{Attribute, DataType, Node, Shape};
use onnx_runtime_optimizer::affine_moe::{OP, THRESHOLD};

use super::affine_bank::{AffineBank, PreparedAffineBank};
mod geometry;
use super::host_rows::Rows;
use geometry::Geometry;
struct Actual<'a> {
    x: Rows<'a>,
    router: Rows<'a>,
    first: PreparedAffineBank<'a>,
    down: PreparedAffineBank<'a>,
}

pub(crate) struct Factory;

struct BankFormat {
    block: usize,
    zero: u8,
}

struct AffineMoe {
    k: usize,
    alpha: f32,
    beta: f32,
    limit: f32,
    first: BankFormat,
    down: BankFormat,
}

impl KernelFactory for Factory {
    fn create(&self, n: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        AffineMoe::parse(n).map(|config| Box::new(config) as Box<dyn Kernel>)
    }

    fn unsupported_reason(
        &self,
        node: &Node,
        _opset: u64,
        _input_shapes: &[Shape],
        _input_dtypes: &[DataType],
    ) -> Option<String> {
        AffineMoe::parse(node).err().map(|error| error.to_string())
    }
}

impl AffineMoe {
    fn parse(n: &Node) -> Result<Self> {
        let integer = |key| n.attr(key).and_then(Attribute::as_int);
        let positive = |key| {
            integer(key)
                .and_then(|v| usize::try_from(v).ok())
                .filter(|&v| v > 0)
                .ok_or_else(|| error(format!("{key} must be a positive INT64")))
        };
        let float = |key| match n.attr(key) {
            Some(Attribute::Float(v)) if v.is_finite() => Ok(*v),
            _ => Err(error(format!("{key} must be a finite FLOAT"))),
        };
        let format = |block, zero| -> Result<BankFormat> {
            Ok(BankFormat {
                block: positive(block)?,
                zero: integer(zero)
                    .and_then(|v| u8::try_from(v).ok())
                    .filter(|&v| v <= 15)
                    .ok_or_else(|| error(format!("{zero} must be a UINT4 value")))?,
            })
        };
        let limit = float("limit")?;
        if limit <= 0.0 {
            return Err(error("limit must be positive"));
        }
        Ok(Self {
            k: positive("k")?,
            alpha: float("alpha")?,
            beta: float("beta")?,
            limit,
            first: format("first_block", "first_zero")?,
            down: format("down_block", "down_zero")?,
        })
    }
}

impl AffineMoe {
    fn actual<'a>(
        &self,
        inputs: &'a [TensorView<'a>],
        out: &TensorMut,
        g: &Geometry,
    ) -> Result<Actual<'a>> {
        let x = Rows::new(OP, "X", &inputs[0])?;
        let router = Rows::new(OP, "router", &inputs[1])?;
        let output = TensorView::new(
            DevicePtr(out.data.0.cast_const()),
            out.dtype,
            out.shape,
            out.strides,
            out.device,
        )
        .with_byte_offset(out.byte_offset);
        crate::strided::validate_host_storage::<f32>(OP, "Y", &output)?;
        let first = AffineBank::prepare(
            &inputs[2],
            &inputs[3],
            (!inputs[4].is_absent()).then_some(&inputs[4]),
            [g.experts, g.inter2, g.hidden],
            self.first.block,
            self.first.zero,
        )
        .map_err(|e| error(format!("first expert bank (inputs 2..4): {e}")))?;
        let down = AffineBank::prepare(
            &inputs[5],
            &inputs[6],
            (!inputs[7].is_absent()).then_some(&inputs[7]),
            [g.experts, g.hidden, g.intermediate],
            self.down.block,
            self.down.zero,
        )
        .map_err(|e| error(format!("down expert bank (inputs 5..7): {e}")))?;
        Ok(Actual {
            x,
            router,
            first,
            down,
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
            .collect::<smallvec::SmallVec<[_; 8]>>();
        let facts = outputs
            .iter()
            .map(OutputMetadata::from_view)
            .collect::<smallvec::SmallVec<[_; 1]>>();
        let g = Geometry::new(self, &metadata, &facts)?;
        let actual = self.actual(inputs, &outputs[0], &g)?;
        let plan = g.plan(self.k, actual.x.scratch_len(), actual.router.scratch_len())?;
        // SAFETY: execution owner supplies exclusive Host workspace disjoint
        // from tensors. Complete binding precedes every source value read.
        unsafe {
            plan.with_buffers(
                workspace,
                |[
                    y,
                    x_scratch,
                    router_scratch,
                    selected_logits,
                    probabilities,
                    activated,
                ],
                 indices| {
                    let first = actual
                        .first
                        .validate()
                        .map_err(|e| error(format!("first expert bank (inputs 2..4): {e}")))?;
                    let down = actual
                        .down
                        .validate()
                        .map_err(|e| error(format!("down expert bank (inputs 5..7): {e}")))?;
                    for (id, index) in indices.iter_mut().enumerate() {
                        *index = id;
                    }
                    let hidden = g.hidden;
                    let experts = g.experts;
                    let inter2 = g.inter2;
                    for (row, output) in y.chunks_exact_mut(hidden).enumerate() {
                        let token = actual.x.get(row, x_scratch)?;
                        let logits = actual.router.get(row, router_scratch)?;
                        // TopK on the reversed expert axis picks larger original IDs on ties.
                        let order = |a: &usize, b: &usize| {
                            logits[*b].total_cmp(&logits[*a]).then_with(|| b.cmp(a))
                        };
                        if self.k < experts {
                            indices.select_nth_unstable_by(self.k, order);
                        }
                        let selected = &mut indices[..self.k];
                        selected.sort_unstable_by(order);
                        for (p, &id) in selected_logits.iter_mut().zip(selected.iter()) {
                            *p = logits[id];
                        }
                        super::softmax::softmax_rows(selected_logits, probabilities, 1, self.k);
                        for (&expert, &p) in selected.iter().zip(probabilities.iter()) {
                            if p.partial_cmp(&THRESHOLD) != Some(std::cmp::Ordering::Greater) {
                                continue;
                            }
                            for (col, act) in activated.iter_mut().enumerate() {
                                let row = expert * inter2 + col * 2;
                                let gate = first.dot(row, token);
                                // ONNX Min propagates NaN, unlike f32::min.
                                let gate = if gate > self.limit { self.limit } else { gate };
                                let up = first.dot(row + 1, token).clamp(-self.limit, self.limit)
                                    + self.beta;
                                let z = self.alpha * gate;
                                let exp = (-z.abs()).exp();
                                let sigmoid = if z >= 0.0 {
                                    1.0 / (1.0 + exp)
                                } else {
                                    exp / (1.0 + exp)
                                };
                                *act = (gate * sigmoid) * up;
                            }
                            for (col, sum) in output.iter_mut().enumerate() {
                                *sum += down.dot(expert * hidden + col, activated) * p;
                            }
                        }
                    }
                    super::write_dense_f32(&mut outputs[0], y)
                },
            )
        }
    }
}
impl Kernel for AffineMoe {
    fn supports_strided_input(&self, index: usize) -> bool {
        index < 8
    }
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        Geometry::new(self, inputs, outputs)?.conservative(self.k)?;
        Ok(WorkspacePreparation::Deferred)
    }
    fn workspace_requirement_for_execution(
        &self,
        inputs: &[TensorView],
        outputs: &[TensorMut],
        metadata: &[TensorMetadata<'_>],
        facts: &[OutputMetadata<'_>],
        _cached: Option<WorkspacePreparation>,
    ) -> Result<WorkspaceExecutionPreparation> {
        let g = Geometry::new(self, metadata, facts)?;
        if inputs.len() == 8 && outputs.len() == 1 {
            for (index, (view, meta)) in inputs.iter().zip(metadata).enumerate() {
                if !view.is_absent() != meta.present
                    || meta.present && (view.dtype != meta.dtype || view.shape != meta.shape)
                {
                    return Err(error(format!(
                        "input {index} differs from prepared metadata; refresh workspace for the actual bindings"
                    )));
                }
            }
            if outputs[0].is_absent()
                || outputs[0].dtype != DataType::Float32
                || outputs[0].shape != metadata[0].shape
            {
                return Err(error(
                    "Y differs from prepared FLOAT X geometry; bind a matching output and refresh metadata",
                ));
            }
            let actual = self.actual(inputs, &outputs[0], &g)?;
            return Ok(WorkspaceExecutionPreparation::Ready(
                g.plan(self.k, actual.x.scratch_len(), actual.router.scratch_len())?
                    .requirement(),
            ));
        }
        Ok(WorkspaceExecutionPreparation::Ready(
            g.conservative(self.k)?.requirement(),
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

fn error(message: impl std::fmt::Display) -> EpError {
    EpError::KernelFailed(format!("{OP}: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use crate::kernels::testutil::PreparedTestKernel;
    #[test]
    fn malformed_scale_reports_the_actual_projection_before_output_writes() {
        let kernel = AffineMoe {
            k: 1,
            alpha: 1.,
            beta: 0.,
            limit: 7.,
            first: BankFormat { block: 1, zero: 7 },
            down: BankFormat { block: 1, zero: 7 },
        };
        for first_bad in [true, false] {
            let x = Owned::f32(&[1, 2], &[1., 2.]);
            let router = Owned::f32(&[1, 1], &[0.]);
            let w1 = Owned::u8(&[1, 4, 1], &[0x88; 4]);
            let z1 = Owned::u8(&[1, 4, 1], &[0x77; 4]);
            let w2 = Owned::u8(&[1, 2, 1], &[0x88; 2]);
            let z2 = Owned::u8(&[1, 2, 1], &[0x77; 2]);
            let mut scales1 = [1.; 8];
            let mut scales2 = [1.; 4];
            if first_bad {
                scales1[0] = 0.;
            } else {
                scales2[0] = 0.;
            }
            let s1 = Owned::f32(&[1, 4, 2], &scales1);
            let s2 = Owned::f32(&[1, 2, 2], &scales2);
            let mut out = Owned::f32(&[1, 2], &[123.; 2]);
            let error = kernel
                .execute_prepared(
                    &[
                        x.view(),
                        router.view(),
                        w1.view(),
                        s1.view(),
                        z1.view(),
                        w2.view(),
                        s2.view(),
                        z2.view(),
                    ],
                    &mut [out.view_mut()],
                )
                .unwrap_err();
            let message = error.to_string();
            assert!(
                message.contains(if first_bad {
                    "first expert bank (inputs 2..4)"
                } else {
                    "down expert bank (inputs 5..7)"
                }),
                "{message}"
            );
            assert!(message.contains("positive and finite"), "{message}");
            assert_eq!(out.to_f32(), [123.; 2]);
        }
    }
}

#[cfg(test)]
#[path = "affine_moe/tests.rs"]
mod workspace_tests;
