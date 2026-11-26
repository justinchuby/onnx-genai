//! `LayerNormalization`: normalize f32 `X` over the axes from `axis` onward,
//! then scale and shift (`docs/architecture/ORT2.md` §4.4).
//!
//! `Y = (X - mean) / sqrt(var + epsilon) * Scale + B`, where `mean`/`var` are
//! the population statistics over the normalized axes. The optional `Mean` and
//! `InvStdDev` outputs are filled when the caller provides those output slots.

use crate::dtype::{to_dense_f32_widen, validate_f32_widen_input, write_dense_f32_narrow};
use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, OutputMetadata, Result, TensorMetadata, TensorMut, TensorView,
    WorkspacePreparation, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Node};

use super::check_arity;
use super::typed_workspace::TypedWorkspace;

#[derive(Clone, Copy)]
enum LayerNormKind {
    Standard,
    Skip,
}

impl LayerNormKind {
    fn op(self) -> &'static str {
        match self {
            Self::Standard => "LayerNormalization",
            Self::Skip => "SkipLayerNormalization",
        }
    }
    fn is_skip(self) -> bool {
        matches!(self, Self::Skip)
    }
}

pub struct LayerNormKernel {
    axis: i64,
    epsilon: f32,
    kind: LayerNormKind,
}

pub struct LayerNormFactory;
impl KernelFactory for LayerNormFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(LayerNormKernel {
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1),
            epsilon: node
                .attr("epsilon")
                .and_then(|a| a.as_float())
                .unwrap_or(1e-5),
            kind: LayerNormKind::Standard,
        }))
    }
}

/// The dense coefficient walk follows from broadcast geometry, never values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AffineLayout {
    Normalized,
    Scalar,
    Broadcast,
}

impl AffineLayout {
    fn prepare(
        op: &str,
        slot: usize,
        parameter: &[usize],
        signal: &[usize],
        axis: usize,
    ) -> Result<Self> {
        if !onnx_runtime_ir::gemm::bias_broadcasts(parameter, signal, |&dim| Some(dim)) {
            return Err(EpError::KernelFailed(format!(
                "{op}: affine input {slot} shape {parameter:?} cannot broadcast into X {signal:?}; use a right-aligned shape with equal or singleton dimensions"
            )));
        }
        // Leading unit dimensions do not change a contiguous row's coefficients.
        // Shape equality is required: equal element counts can vary by group.
        if parameter
            .iter()
            .skip_while(|&&d| d == 1)
            .eq(signal[axis..].iter().skip_while(|&&d| d == 1))
        {
            Ok(Self::Normalized)
        } else if parameter.iter().all(|&d| d == 1) {
            Ok(Self::Scalar)
        } else {
            Ok(Self::Broadcast)
        }
    }

    #[inline]
    fn value(
        self,
        values: &[f32],
        parameter: &[usize],
        signal: &[usize],
        local: usize,
        flat: usize,
    ) -> f32 {
        match self {
            Self::Normalized => values[local],
            Self::Scalar => values[0],
            Self::Broadcast => {
                // Only called for a nonempty X: every divisor is positive.
                // Geometry already checked both dense suffix-stride products.
                let mut remaining = flat;
                let mut stride = 1usize;
                let mut index = 0usize;
                let offset = signal.len() - parameter.len();
                for d in (0..parameter.len()).rev() {
                    let coordinate = remaining % signal[offset + d];
                    remaining /= signal[offset + d];
                    if parameter[d] != 1 {
                        index += coordinate * stride;
                    }
                    stride *= parameter[d];
                }
                values[index]
            }
        }
    }
}

struct LayerNormGeometry {
    norm_size: usize,
    groups: usize,
    count: usize,
    scale_slot: usize,
    bias_slot: usize,
    writes: [bool; 4],
    affine: [AffineLayout; 2],
    workspace: TypedWorkspace<f32, 4>,
}

impl LayerNormKernel {
    pub(super) fn skip(epsilon: f32) -> Self {
        Self {
            axis: -1,
            epsilon,
            kind: LayerNormKind::Skip,
        }
    }

    fn geometry(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<LayerNormGeometry> {
        let op = self.kind.op();
        let skip = self.kind.is_skip();
        let (min_inputs, max_inputs) = if skip { (3, 5) } else { (2, 3) };
        if !(min_inputs..=max_inputs).contains(&inputs.len())
            || inputs[..min_inputs.min(inputs.len())]
                .iter()
                .any(|input| !input.present)
        {
            return Err(EpError::KernelFailed(format!(
                "{op}: expected {min_inputs} to {max_inputs} input slots with required inputs present, got {}; bind X, scale and required skip inputs",
                inputs.len()
            )));
        }
        let shape = inputs[0].shape;
        let rank = shape.len();
        let axis = if self.axis < 0 {
            self.axis.checked_add(rank as i64)
        } else {
            Some(self.axis)
        };
        let axis = axis
            .filter(|&axis| axis >= 0 && (axis as usize) < rank)
            .ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "{op}: axis {} is out of range for rank {rank}; use an axis in [-rank, rank-1]",
                    self.axis
                ))
            })? as usize;
        let product = |dims: &[usize], name: &str| {
            dims.iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "{op}: {name} dimensions {dims:?} overflow; reduce the tensor shape"
                    ))
                })
        };
        let norm_size = product(&shape[axis..], "normalization")?;
        if norm_size == 0 {
            return Err(EpError::KernelFailed(format!(
                "{op}: empty normalization axis; use a nonzero normalized width"
            )));
        }
        let groups = product(&shape[..axis], "group")?;
        let count = groups.checked_mul(norm_size).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "{op}: X shape {shape:?} overflows element count; reduce the tensor shape"
            ))
        })?;
        let scale_slot = if skip { 2 } else { 1 };
        let bias_slot = scale_slot + 1;
        let mut affine = [AffineLayout::Normalized; 2];
        for (i, input) in inputs.iter().enumerate().filter(|(_, input)| input.present) {
            if !super::FLOAT_DTYPES.contains(&input.dtype)
                || (skip
                    && (!matches!(
                        input.dtype,
                        DataType::Float16 | DataType::BFloat16 | DataType::Float32
                    ) || input.dtype != inputs[0].dtype))
            {
                return Err(EpError::KernelFailed(format!(
                    "{op}: input {i} dtype {:?} is invalid; use floating inputs, and match all Skip inputs to activation dtype {:?} (Float16/BFloat16/Float32)",
                    input.dtype, inputs[0].dtype
                )));
            }
            // Validate all suffix stride products even when an outer zero hides total size.
            let mut stride = 1usize;
            for &dim in input.shape.iter().rev() {
                stride = stride.checked_mul(dim).ok_or_else(|| EpError::KernelFailed(format!("{op}: input {i} shape {:?} overflows contiguous strides; reduce the tensor shape", input.shape)))?;
            }
            if skip && i == 1 {
                if input.shape != shape {
                    return Err(EpError::KernelFailed(format!(
                        "{op}: skip shape {:?} must have the same shape as X {shape:?}",
                        input.shape
                    )));
                }
            } else if i >= scale_slot {
                let n = product(input.shape, "parameter")?;
                if skip && n != norm_size {
                    return Err(EpError::KernelFailed(format!(
                        "{op}: input {i} parameter has {n} elements, expected {norm_size} for normalized axes {axis}..{rank}; provide a complete normalized parameter"
                    )));
                }
                if !skip {
                    affine[i - scale_slot] =
                        AffineLayout::prepare(op, i, input.shape, shape, axis)?;
                }
            }
        }
        let max_outputs = if skip { 4 } else { 3 };
        if outputs.len() > max_outputs {
            return Err(EpError::KernelFailed(format!(
                "{op}: expected at most {max_outputs} output slots, got {}",
                outputs.len()
            )));
        }
        let mut writes = [false; 4];
        for (i, output) in outputs.iter().enumerate() {
            let (dtype, output_shape) = match output {
                OutputMetadata::Absent if i > 0 => continue,
                OutputMetadata::Absent => {
                    return Err(EpError::KernelFailed(format!(
                        "{op}: required output 0 is absent; bind the normalized output"
                    )));
                }
                OutputMetadata::Unresolved { dtype } => (*dtype, None),
                OutputMetadata::Tensor { dtype, shape } => (*dtype, Some(shape.as_ref())),
            };
            let stats = i == 1 || i == 2;
            let valid_dtype = if skip {
                dtype == inputs[0].dtype || (stats && dtype == DataType::Float32)
            } else {
                super::FLOAT_DTYPES.contains(&dtype)
            };
            if !valid_dtype {
                let expected = if skip && stats {
                    format!("Float32 or {:?}", inputs[0].dtype)
                } else if skip {
                    format!("{:?}", inputs[0].dtype)
                } else {
                    "a floating dtype".into()
                };
                return Err(EpError::KernelFailed(format!(
                    "{op}: output {i} must have dtype {expected}, got {dtype:?}"
                )));
            }
            if let Some(out) = output_shape
                && (out.len() != rank
                    || out.iter().enumerate().any(|(d, &extent)| {
                        extent != if stats && d >= axis { 1 } else { shape[d] }
                    }))
            {
                return Err(EpError::KernelFailed(format!(
                    "{op}: output {i} shape {out:?} must match {} of X {shape:?} for axis {axis}",
                    if stats {
                        "the reduced shape"
                    } else {
                        "the shape"
                    }
                )));
            }

            writes[i] = true;
        }
        let workspace = TypedWorkspace::<f32, _>::new(
            op,
            [
                ("Y staging", count),
                ("mean", if writes[1] { groups } else { 0 }),
                (
                    "inverse standard deviation",
                    if writes[2] { groups } else { 0 },
                ),
                ("residual sum", if skip { count } else { 0 }),
            ],
        )?;
        Ok(LayerNormGeometry {
            norm_size,
            groups,
            count,
            scale_slot,
            bias_slot,
            writes,
            affine,
            workspace,
        })
    }

    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let op = self.kind.op();
        let skip_mode = self.kind.is_skip();
        check_arity(
            op,
            inputs,
            outputs,
            if skip_mode { 3 } else { 2 },
            if skip_mode { 5 } else { 3 },
            1,
        )?;
        if outputs.len() > 4 {
            return Err(EpError::KernelFailed(format!(
                "{op}: at most four output slots are supported; got {}",
                outputs.len()
            )));
        }
        let metadata = std::array::from_fn::<_, 5, _>(|i| {
            inputs.get(i).map_or(
                TensorMetadata::new(DataType::Undefined, &[], false),
                |input| TensorMetadata::new(input.dtype, input.shape, !input.is_absent()),
            )
        });
        let output_metadata = std::array::from_fn::<_, 4, _>(|i| {
            outputs
                .get(i)
                .map_or(OutputMetadata::Absent, OutputMetadata::from_view)
        });
        let geometry =
            self.geometry(&metadata[..inputs.len()], &output_metadata[..outputs.len()])?;
        for (index, input) in inputs
            .iter()
            .enumerate()
            .filter(|(_, input)| !input.is_absent())
        {
            validate_f32_widen_input(op,input).map_err(|error| EpError::KernelFailed(format!("{op}: input {index} view is invalid: {error}; bind a valid floating Host-accessible input")))?;
        }
        for (index, output) in outputs
            .iter()
            .enumerate()
            .filter(|(_, output)| !output.is_absent())
        {
            output.validate().map_err(|error| EpError::KernelFailed(format!("{op}: output {index} view is invalid: {error}; bind a valid floating Host-accessible output")))?;
            if !output.device.is_host_accessible() {
                return Err(EpError::KernelFailed(format!(
                    "{op}: output {index} on {:?} is not Host-accessible; explicitly transfer it before CPU execution",
                    output.device
                )));
            }
        }
        if !skip_mode {
            crate::trace::record_kernel_metrics(inputs, outputs, || {
                (geometry.count as u64)
                    .saturating_mul(7)
                    .saturating_add((geometry.groups as u64).saturating_mul(5))
                    .saturating_add(
                        if metadata
                            .get(geometry.bias_slot)
                            .is_some_and(|input| input.present)
                        {
                            geometry.count as u64
                        } else {
                            0
                        },
                    )
            });
        }
        // SAFETY: synchronous CPU execution owns exclusive live Host backing;
        // the shared descriptor validates and initializes disjoint regions.
        unsafe {
            geometry
                .workspace
                .with_buffers(workspace, |[y, means, invs, sum]| {
                    if geometry.count == 0 {
                        return Ok(());
                    }
                    let x = to_dense_f32_widen(op, &inputs[0])?;
                    let scale = to_dense_f32_widen(op, &inputs[geometry.scale_slot])?;
                    let beta = inputs
                        .get(geometry.bias_slot)
                        .filter(|input| !input.is_absent())
                        .map(|input| to_dense_f32_widen(op, input))
                        .transpose()?;
                    let skip = if skip_mode {
                        Some(to_dense_f32_widen(op, &inputs[1])?)
                    } else {
                        None
                    };
                    let bias = if skip_mode {
                        inputs
                            .get(4)
                            .filter(|input| !input.is_absent())
                            .map(|input| to_dense_f32_widen(op, input))
                            .transpose()?
                    } else {
                        None
                    };
                    let data = if let Some(skip) = skip.as_deref() {
                        for (i, ((&a, &b), value)) in
                            x.iter().zip(skip).zip(sum.iter_mut()).enumerate()
                        {
                            // Preserve the final +0.0 when bias is absent, including signed zeros.
                            *value = a
                                + b
                                + bias
                                    .as_ref()
                                    .map_or(0.0, |values| values[i % geometry.norm_size]);
                        }
                        &*sum
                    } else {
                        x.as_ref()
                    };
                    if geometry.affine == [AffineLayout::Normalized; 2] {
                        // Preserve the complete normalized-parameter inner loop:
                        // no broadcast coordinate arithmetic on this route.
                        layer_norm_into(
                            data,
                            geometry.norm_size,
                            self.epsilon,
                            [y, means, invs],
                            |e, _, norm| {
                                let mut out = norm * scale[e];
                                if let Some(b) = beta.as_deref() {
                                    out += b[e];
                                }
                                out
                            },
                        );
                    } else {
                        let shape = inputs[0].shape;
                        let scale_shape = inputs[geometry.scale_slot].shape;
                        let beta_shape = inputs
                            .get(geometry.bias_slot)
                            .map_or(&[][..], |input| input.shape);
                        layer_norm_into(
                            data,
                            geometry.norm_size,
                            self.epsilon,
                            [y, means, invs],
                            |e, flat, norm| {
                                let coefficient =
                                    geometry.affine[0].value(&scale, scale_shape, shape, e, flat);
                                let mut out = norm * coefficient;
                                if let Some(b) = beta.as_deref() {
                                    out += geometry.affine[1].value(b, beta_shape, shape, e, flat);
                                }
                                out
                            },
                        );
                    }
                    // Finish every input read and all computations before publishing outputs.
                    write_dense_f32_narrow(op, &mut outputs[0], y)?;
                    if geometry.writes[1] {
                        write_dense_f32_narrow(op, &mut outputs[1], means)?;
                    }
                    if geometry.writes[2] {
                        write_dense_f32_narrow(op, &mut outputs[2], invs)?;
                    }
                    if geometry.writes[3] {
                        write_dense_f32_narrow(op, &mut outputs[3], sum)?;
                    }
                    Ok(())
                })
        }
    }
}

impl Kernel for LayerNormKernel {
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        Ok(WorkspacePreparation::Ready(
            self.geometry(inputs, outputs)?.workspace.requirement(),
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

fn layer_norm_into(
    x: &[f32],
    norm_size: usize,
    epsilon: f32,
    [y, means, inv_stds]: [&mut [f32]; 3],
    affine: impl Fn(usize, usize, f32) -> f32,
) {
    let mut means = means.iter_mut();
    let mut inv_stds = inv_stds.iter_mut();
    for (g, slice) in x.chunks_exact(norm_size).enumerate() {
        let base = g * norm_size;
        let mean = slice.iter().sum::<f32>() / norm_size as f32;
        let var = slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / norm_size as f32;
        let inv_std = 1.0 / (var + epsilon).sqrt();
        if let Some(value) = means.next() {
            *value = mean;
        }
        if let Some(value) = inv_stds.next() {
            *value = inv_std;
        }
        for e in 0..norm_size {
            let norm = (slice[e] - mean) * inv_std;
            y[base + e] = affine(e, base + e, norm);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::{Owned, PreparedTestKernel};

    #[test]
    fn layernorm_last_axis_matches_reference() {
        // X [2,3], normalize last axis. Scale=1, Bias=0, eps=1e-5.
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 2., 4., 6.]);
        let scale = Owned::f32(&[3], &[1., 1., 1.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        let k = LayerNormKernel {
            kind: LayerNormKind::Standard,
            axis: -1,
            epsilon: 1e-5,
        };
        k.execute_prepared(&[x.view(), scale.view()], &mut [out.view_mut()])
            .unwrap();

        // Hand-computed: row0 mean=2, var=2/3, inv=1/sqrt(0.66667+1e-5)
        let expect = |row: &[f32]| -> Vec<f32> {
            let mean = row.iter().sum::<f32>() / 3.0;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 3.0;
            let inv = 1.0 / (var + 1e-5f32).sqrt();
            row.iter().map(|v| (v - mean) * inv).collect()
        };
        let mut want = expect(&[1., 2., 3.]);
        want.extend(expect(&[2., 4., 6.]));
        let got = out.to_f32();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }

    #[test]
    fn layernorm_applies_scale_and_bias() {
        let x = Owned::f32(&[1, 4], &[1., 2., 3., 4.]);
        let scale = Owned::f32(&[4], &[2., 2., 2., 2.]);
        let bias = Owned::f32(&[4], &[1., 1., 1., 1.]);
        let mut out = Owned::zeros_f32(&[1, 4]);
        let k = LayerNormKernel {
            kind: LayerNormKind::Standard,
            axis: -1,
            epsilon: 1e-5,
        };
        k.execute_prepared(
            &[x.view(), scale.view(), bias.view()],
            &mut [out.view_mut()],
        )
        .unwrap();

        let row = [1., 2., 3., 4.];
        let mean = 2.5;
        let var = row.iter().map(|v: &f32| (v - mean).powi(2)).sum::<f32>() / 4.0;
        let inv = 1.0 / (var + 1e-5f32).sqrt();
        let want: Vec<f32> = row.iter().map(|v| (v - mean) * inv * 2.0 + 1.0).collect();
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }

    #[test]
    fn layernorm_writes_optional_mean_and_invstd() {
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 2., 4., 6.]);
        let scale = Owned::f32(&[3], &[1., 1., 1.]);
        let mut y = Owned::zeros_f32(&[2, 3]);
        let mut mean = Owned::zeros_f32(&[2, 1]);
        let mut invstd = Owned::zeros_f32(&[2, 1]);
        let k = LayerNormKernel {
            kind: LayerNormKind::Standard,
            axis: -1,
            epsilon: 1e-5,
        };
        k.execute_prepared(
            &[x.view(), scale.view()],
            &mut [y.view_mut(), mean.view_mut(), invstd.view_mut()],
        )
        .unwrap();
        assert!((mean.to_f32()[0] - 2.0).abs() < 1e-6);
        assert!((mean.to_f32()[1] - 4.0).abs() < 1e-6);
    }
    #[test]
    fn layernorm_bf16_matches_widened_f32_reference() {
        let x = Owned::bf16(&[2, 3], &[-80., 0., 1., 80., -1., 2.]);
        let scale = Owned::bf16(&[3], &[1., 0.5, 2.]);
        let bias = Owned::bf16(&[3], &[0., -1., 1.]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[2, 3]);
        LayerNormKernel {
            kind: LayerNormKind::Standard,
            axis: -1,
            epsilon: 1e-5,
        }
        .execute_prepared(
            &[x.view(), scale.view(), bias.view()],
            &mut [out.view_mut()],
        )
        .unwrap();
        let reference_x = Owned::f32(&[2, 3], &x.to_bf16_as_f32());
        let reference_scale = Owned::f32(&[3], &scale.to_bf16_as_f32());
        let reference_bias = Owned::f32(&[3], &bias.to_bf16_as_f32());
        let mut reference_output = Owned::zeros_f32(&[2, 3]);
        LayerNormKernel {
            axis: -1,
            epsilon: 1e-5,
            kind: LayerNormKind::Standard,
        }
        .execute_prepared(
            &[
                reference_x.view(),
                reference_scale.view(),
                reference_bias.view(),
            ],
            &mut [reference_output.view_mut()],
        )
        .unwrap();
        let reference = reference_output.to_f32();
        let expected: Vec<_> = reference
            .into_iter()
            .map(half::bf16::from_f32)
            .map(half::bf16::to_f32)
            .collect();
        assert_eq!(out.to_bf16_as_f32(), expected);
    }
    fn standard() -> LayerNormKernel {
        LayerNormKernel {
            axis: -1,
            epsilon: 1e-5,
            kind: LayerNormKind::Standard,
        }
    }
    fn metadata<'a>(inputs: &'a [TensorView<'a>]) -> Vec<TensorMetadata<'a>> {
        inputs
            .iter()
            .map(|x| TensorMetadata::new(x.dtype, x.shape, !x.is_absent()))
            .collect()
    }

    #[test]
    fn layernorm_metadata_reserves_only_requested_statistics_and_one_residual_sum() {
        let x = TensorMetadata::new(DataType::Float32, &[2, 4], true);
        let scale = TensorMetadata::new(DataType::Float32, &[1, 4], true);
        let y = OutputMetadata::tensor(DataType::Float32, &[2, 4]);
        let stat = OutputMetadata::tensor(DataType::Float32, &[2, 1]);
        let request =
            |kernel: &LayerNormKernel, inputs: &[TensorMetadata], outputs: &[OutputMetadata]| {
                kernel
                    .workspace_requirement(inputs, outputs)
                    .unwrap()
                    .require_ready()
                    .unwrap()
                    .bytes
            };
        assert_eq!(
            request(&standard(), &[x, scale], std::slice::from_ref(&y)),
            32
        );
        assert_eq!(
            request(&standard(), &[x, scale], &[y.clone(), stat.clone()]),
            40
        );
        assert_eq!(
            request(
                &standard(),
                &[x, scale],
                &[y.clone(), OutputMetadata::Absent, stat.clone()]
            ),
            40
        );
        assert_eq!(
            request(
                &standard(),
                &[x, scale],
                &[y.clone(), stat.clone(), stat.clone()]
            ),
            48
        );
        let kernel = LayerNormKernel::skip(1e-5);
        assert_eq!(
            request(&kernel, &[x, x, scale], std::slice::from_ref(&y)),
            64
        );
        assert_eq!(
            request(
                &kernel,
                &[x, x, scale],
                &[
                    y.clone(),
                    OutputMetadata::Absent,
                    OutputMetadata::Absent,
                    y.clone()
                ]
            ),
            64
        );
        assert_eq!(
            request(&kernel, &[x, x, scale], &[y.clone(), stat.clone(), stat, y]),
            80
        );
        for axis in [2, -3] {
            assert!(
                LayerNormKernel {
                    axis,
                    epsilon: 1e-5,
                    kind: LayerNormKind::Standard
                }
                .workspace_requirement(&[x, scale], &[])
                .unwrap_err()
                .to_string()
                .contains("out of range")
            );
        }
        for shape in [
            &[0usize, usize::MAX, 2][..],
            &[1, isize::MAX as usize / 4 + 1][..],
        ] {
            let huge = TensorMetadata::new(DataType::Float32, shape, true);
            assert!(
                standard()
                    .workspace_requirement(&[huge, scale], &[])
                    .is_err()
            );
        }
        assert!(
            standard()
                .workspace_requirement(
                    &[x, scale],
                    &[OutputMetadata::tensor(DataType::Float32, &[4, 2])]
                )
                .is_err()
        );
    }

    #[test]
    fn layernorm_workspace_refuses_before_publication_and_reuses_poisoned_or_fresh_regions() {
        use onnx_runtime_ep_api::DevicePtrMut;
        use std::mem::MaybeUninit;
        let kernel = LayerNormKernel::skip(1e-5);
        let x = Owned::f32(&[2, 4], &[1., 2., 3., 4., 2., 3., 4., 5.]);
        let skip = Owned::f32(&[2, 4], &[0.5, -1., 1., 0., 1., 0., -1., 2.]);
        let scale = Owned::f32(&[4], &[1., 2., 0.5, 1.5]);
        let inputs = [x.view(), skip.view(), scale.view()];
        let mut y = Owned::f32(&[2, 4], &[-123.; 8]);
        let mut mean = Owned::f32(&[2, 1], &[-234.; 2]);
        let mut inv = Owned::f32(&[2, 1], &[-345.; 2]);
        let mut sum = Owned::f32(&[2, 4], &[-456.; 8]);
        let poison = 0x7fc0_1234u32;
        let mut backing = [poison; 24];
        let ptr = unsafe { backing.as_mut_ptr().add(1).cast::<u8>() };
        let valid = WorkspaceView::new(DevicePtrMut(ptr.cast()), 80);
        for bad in [
            None,
            Some(WorkspaceView::new(DevicePtrMut(ptr.cast()), 79)),
            Some(WorkspaceView::new(
                DevicePtrMut(unsafe { ptr.add(1).cast() }),
                80,
            )),
        ] {
            let error = kernel
                .execute_with_workspace(
                    &inputs,
                    &mut [
                        y.view_mut(),
                        mean.view_mut(),
                        inv.view_mut(),
                        sum.view_mut(),
                    ],
                    bad,
                )
                .unwrap_err();
            assert!(
                error.to_string().contains("workspace needs 80 bytes"),
                "{error}"
            );
            assert_eq!(y.to_f32(), [-123.; 8]);
            assert_eq!(mean.to_f32(), [-234.; 2]);
            assert_eq!(inv.to_f32(), [-345.; 2]);
            assert_eq!(sum.to_f32(), [-456.; 8]);
            assert!(backing.iter().all(|&v| v == poison));
        }
        kernel
            .execute_with_workspace(
                &inputs,
                &mut [
                    y.view_mut(),
                    mean.view_mut(),
                    inv.view_mut(),
                    sum.view_mut(),
                ],
                Some(valid),
            )
            .unwrap();
        let expected = [y.to_f32(), mean.to_f32(), inv.to_f32(), sum.to_f32()];
        assert_eq!(backing[0], poison);
        assert!(backing[21..].iter().all(|&v| v == poison));
        let mut fresh = [MaybeUninit::<f32>::uninit(); 20];
        kernel
            .execute_with_workspace(
                &inputs,
                &mut [
                    y.view_mut(),
                    mean.view_mut(),
                    inv.view_mut(),
                    sum.view_mut(),
                ],
                Some(WorkspaceView::new(
                    DevicePtrMut(fresh.as_mut_ptr().cast()),
                    80,
                )),
            )
            .unwrap();
        assert_eq!(
            [y.to_f32(), mean.to_f32(), inv.to_f32(), sum.to_f32()],
            expected
        );
        backing[9..].fill(poison);
        standard()
            .execute_with_workspace(
                &[
                    x.view(),
                    scale.view(),
                    TensorView::absent(DataType::Float32),
                ],
                &mut [
                    y.view_mut(),
                    mean.view_mut().mark_absent(),
                    inv.view_mut().mark_absent(),
                ],
                Some(valid),
            )
            .unwrap();
        assert!(
            backing[9..].iter().all(|&v| v == poison),
            "omitted stats and standard residual region stay untouched"
        );
    }

    #[test]
    fn skip_layernorm_finishes_all_aliased_operand_reads_before_any_output() {
        use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
        use onnx_runtime_ir::DeviceId;
        let kernel = LayerNormKernel::skip(1e-5);
        let values = [1., 2., 3., 4., 2., 3., 4., 5.];
        let skips = [0.5, -1., 1., 0., 1., 0., -1., 2.];
        let scales = [1., 2., 0.5, 1.5];
        let betas = [0., 1., -1., 0.5];
        let biases = [0.25, 0., -0.5, 1.];
        let x = Owned::f32(&[2, 4], &values);
        let skip = Owned::f32(&[2, 4], &skips);
        let scale = Owned::f32(&[4], &scales);
        let beta = Owned::f32(&[4], &betas);
        let bias = Owned::f32(&[4], &biases);
        let mut y = Owned::zeros_f32(&[2, 4]);
        let mut mean = Owned::zeros_f32(&[2, 1]);
        let mut inv = Owned::zeros_f32(&[2, 1]);
        let mut sum = Owned::zeros_f32(&[2, 4]);
        kernel
            .execute_prepared(
                &[
                    x.view(),
                    skip.view(),
                    scale.view(),
                    beta.view(),
                    bias.view(),
                ],
                &mut [
                    y.view_mut(),
                    mean.view_mut(),
                    inv.view_mut(),
                    sum.view_mut(),
                ],
            )
            .unwrap();
        let expected = [y.to_f32(), mean.to_f32(), inv.to_f32(), sum.to_f32()];
        for slot in 0..6 {
            let mut shared = values;
            let source = [
                &values[..],
                &skips[..],
                &scales[..],
                &betas[..],
                &biases[..],
            ][slot.min(4)];
            if slot < 5 {
                shared[..source.len()].copy_from_slice(source);
            }
            let ptr = shared.as_mut_ptr();
            let mut inputs = [
                x.view(),
                skip.view(),
                scale.view(),
                beta.view(),
                bias.view(),
            ];
            if slot < 5 {
                inputs[slot] = TensorView::new(
                    DevicePtr(ptr.cast()),
                    DataType::Float32,
                    if slot < 2 { &[2, 4] } else { &[4] },
                    if slot < 2 { &[4, 1] } else { &[1] },
                    DeviceId::cpu(),
                );
            } else {
                inputs[0] = TensorView::new(
                    DevicePtr(ptr.cast()),
                    DataType::Float32,
                    &[2, 4],
                    &[4, 1],
                    DeviceId::cpu(),
                );
            }
            let out = if slot < 5 {
                TensorMut::new(
                    DevicePtrMut(ptr.cast()),
                    DataType::Float32,
                    &[2, 4],
                    &[4, 1],
                    DeviceId::cpu(),
                )
            } else {
                y.view_mut()
            };
            let stat = if slot == 5 {
                TensorMut::new(
                    DevicePtrMut(ptr.cast()),
                    DataType::Float32,
                    &[2, 1],
                    &[1, 1],
                    DeviceId::cpu(),
                )
            } else {
                mean.view_mut()
            };
            kernel
                .execute_prepared(&inputs, &mut [out, stat, inv.view_mut(), sum.view_mut()])
                .unwrap();
            assert_eq!(
                if slot < 5 {
                    shared.to_vec()
                } else {
                    y.to_f32()
                },
                expected[0]
            );
            assert_eq!(
                if slot == 5 {
                    shared[..2].to_vec()
                } else {
                    mean.to_f32()
                },
                expected[1]
            );
            assert_eq!(inv.to_f32(), expected[2]);
            assert_eq!(sum.to_f32(), expected[3]);
        }
    }

    #[test]
    fn layernorm_empty_groups_preserve_parameter_view_and_output_guards() {
        let x = Owned::zeros_f32(&[0, 4]);
        let scale = Owned::f32(&[4], &[1.; 4]);
        let mut y = Owned::zeros_f32(&[0, 4]);
        standard()
            .execute(&[x.view(), scale.view()], &mut [y.view_mut()])
            .unwrap();
        LayerNormKernel::skip(1e-5)
            .execute(&[x.view(), x.view(), scale.view()], &mut [y.view_mut()])
            .unwrap();
        let mut bad = scale.view();
        bad.device = onnx_runtime_ir::DeviceId::cuda(0);
        let error = standard()
            .execute(&[x.view(), bad], &mut [y.view_mut()])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("input 1") && error.contains("Host-accessible"),
            "{error}"
        );
        let mut bad = y.view_mut();
        bad.strides = &[];
        assert!(
            standard()
                .execute(&[x.view(), scale.view()], &mut [bad])
                .unwrap_err()
                .to_string()
                .contains("output 0 view is invalid")
        );
        let wrong = Owned::f32(&[3], &[1.; 3]);
        assert!(
            standard()
                .execute(&[x.view(), wrong.view()], &mut [y.view_mut()])
                .is_err()
        );
        let empty = Owned::zeros_f32(&[2, 0]);
        let unit = Owned::f32(&[], &[1.]);
        let mut out = Owned::zeros_f32(&[2, 0]);
        assert!(
            standard()
                .execute(&[empty.view(), unit.view()], &mut [out.view_mut()])
                .unwrap_err()
                .to_string()
                .contains("empty normalization axis")
        );
    }

    #[test]
    fn skip_sum_preserves_the_unconditional_bias_add_and_optional_output_holes() {
        let x = Owned::f32(&[1, 4], &[-0.; 4]);
        let skip = Owned::f32(&[1, 4], &[-0.; 4]);
        let scale = Owned::f32(&[4], &[1.; 4]);
        let mut y = Owned::zeros_f32(&[1, 4]);
        let mut omitted = Owned::f32(&[1, 1], &[-123.]);
        let mut inv = Owned::zeros_f32(&[1, 1]);
        let mut sum = Owned::zeros_f32(&[1, 4]);
        LayerNormKernel::skip(1e-5)
            .execute_prepared(
                &[x.view(), skip.view(), scale.view()],
                &mut [
                    y.view_mut(),
                    omitted.view_mut().mark_absent(),
                    inv.view_mut(),
                    sum.view_mut(),
                ],
            )
            .unwrap();
        assert!(
            sum.to_f32().iter().all(|x| x.to_bits() == 0),
            "(-0 + -0) + absent-bias(+0) must produce +0"
        );
        assert_eq!(omitted.to_f32(), [-123.]);
        assert_eq!(
            inv.to_f32()[0].to_bits(),
            (1.0f32 / 1e-5f32.sqrt()).to_bits()
        );
    }

    #[test]
    fn layernorm_strided_half_operands_and_outputs_preserve_padding() {
        let values = [1., 2., 3., 4., 2., 3., 4., 5.];
        let scales = [1., 2., 0.5, 1.5];
        let padded = values
            .into_iter()
            .flat_map(|x| [x, 123.])
            .collect::<Vec<_>>();
        let padded_scale = scales
            .into_iter()
            .flat_map(|x| [x, 456.])
            .collect::<Vec<_>>();
        let x = Owned::f16(&[2, 8], &padded).with_view(&[2, 4], &[8, 2]);
        let scale = Owned::f16(&[8], &padded_scale).with_view(&[4], &[2]);
        let mut y = Owned::f16(&[2, 8], &[-123.; 16]).with_view(&[2, 4], &[8, 2]);
        let mut mean = Owned::zeros_f32(&[2, 1]);
        let mut inv = Owned::zeros_f32(&[2, 1]);
        standard()
            .execute_prepared(
                &[x.view(), scale.view()],
                &mut [y.view_mut(), mean.view_mut(), inv.view_mut()],
            )
            .unwrap();
        for (g, row) in values.chunks_exact(4).enumerate() {
            let m = row.iter().map(|&x| f64::from(x)).sum::<f64>() / 4.;
            let variance = row.iter().map(|&x| (f64::from(x) - m).powi(2)).sum::<f64>() / 4.;
            let inverse = 1. / (variance + 1e-5).sqrt();
            assert!((f64::from(mean.to_f32()[g]) - m).abs() < 1e-6);
            assert!((f64::from(inv.to_f32()[g]) - inverse).abs() < 1e-6);
            for e in 0..4 {
                let expected =
                    half::f16::from_f64((f64::from(row[e]) - m) * inverse * f64::from(scales[e]))
                        .to_bits();
                assert_eq!(y.to_u16_bits()[(g * 4 + e) * 2], expected);
                assert_eq!(
                    y.to_u16_bits()[(g * 4 + e) * 2 + 1],
                    half::f16::from_f32(-123.).to_bits()
                );
            }
        }
    }

    #[test]
    fn governed_layernorm_and_skip_share_one_warm_lane_without_issuer_callbacks() {
        use crate::kernels::testutil::RecordingGovernor;
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{
            HolderId, HostWorkspacePool, MemoryGovernor, ProcessMemoryManager,
        };
        use std::sync::Arc;
        let books = Arc::new(RecordingGovernor::new_host(80));
        let manager = ProcessMemoryManager::new().unwrap();
        let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
        let resources = crate::CpuMemoryResources::new(
            books.weight_memory(HolderId::new(13)),
            Default::default(),
        )
        .unwrap()
        .with_workspace_pool(pool.clone())
        .unwrap();
        let provider = crate::CpuExecutionProvider::initialized_with_resources(resources).unwrap();
        let x = Owned::f32(&[2, 4], &[1., 2., 3., 4., 2., 3., 4., 5.]);
        let skip = Owned::f32(&[2, 4], &[0.5, -1., 1., 0., 1., 0., -1., 2.]);
        let scale = Owned::f32(&[4], &[1.; 4]);
        let scalar = Owned::f32(&[], &[1.]);
        let group = Owned::f32(&[2, 1], &[1., 2.]);
        let scalar_inputs = [x.view(), scalar.view()];
        let group_inputs = [x.view(), group.view()];
        let scalar_meta = metadata(&scalar_inputs);
        let group_meta = metadata(&group_inputs);
        let ordinary = [x.view(), scale.view()];
        let residual = [x.view(), skip.view(), scale.view()];
        let ordinary_meta = metadata(&ordinary);
        let residual_meta = metadata(&residual);
        let kernels = [standard(), LayerNormKernel::skip(1e-5)];
        let outputs = [
            OutputMetadata::tensor(DataType::Float32, &[2, 4]),
            OutputMetadata::tensor(DataType::Float32, &[2, 1]),
            OutputMetadata::tensor(DataType::Float32, &[2, 1]),
            OutputMetadata::tensor(DataType::Float32, &[2, 4]),
        ];
        let request = kernels[1]
            .workspace_requirement(&residual_meta, &outputs)
            .unwrap()
            .require_ready()
            .unwrap();
        assert_eq!(request.bytes, 80);
        let first = provider
            .allocate_workspace(80, request.alignment, request.role())
            .unwrap();
        let identity = first.managed_owner().unwrap().identity();
        assert!(
            provider
                .allocate_workspace(80, request.alignment, request.role())
                .is_err()
        );
        provider.deallocate_workspace(first).unwrap();
        let before = (books.claims(), books.releases());
        drop(
            books
                .reserve(Tier::Host, 0, request.role(), HolderId::new(99))
                .unwrap(),
        );
        books.probe_zero_release();
        assert_eq!(
            (books.claims(), books.releases()),
            (before.0 + 1, before.1 + 1)
        );
        let calls = (books.claims(), books.releases());
        let mut y = Owned::zeros_f32(&[2, 4]);
        let mut mean = Owned::zeros_f32(&[2, 1]);
        let mut inv = Owned::zeros_f32(&[2, 1]);
        let mut sum = Owned::zeros_f32(&[2, 4]);
        for i in 0..4096 {
            let (which, inputs, meta, nout) = match i % 4 {
                0 => (0, &ordinary[..], &ordinary_meta[..], 3),
                1 => (1, &residual[..], &residual_meta[..], 4),
                2 => (0, &scalar_inputs[..], &scalar_meta[..], 3),
                _ => (0, &group_inputs[..], &group_meta[..], 3),
            };
            let request = kernels[which]
                .workspace_requirement_for_execution(inputs, &[], meta, &outputs[..nout], None)
                .unwrap()
                .require_ready()
                .unwrap();
            assert_eq!(request.bytes, if which == 0 { 48 } else { 80 });
            let mut allocation = provider
                .allocate_workspace(request.bytes as usize, request.alignment, request.role())
                .unwrap();
            assert_eq!(allocation.managed_owner().unwrap().identity(), identity);
            let view = WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(allocation.as_mut_ptr()),
                allocation.len(),
            );
            let mut out = [
                y.view_mut(),
                mean.view_mut(),
                inv.view_mut(),
                sum.view_mut(),
            ];
            kernels[which]
                .execute_with_workspace(inputs, &mut out[..nout], Some(view))
                .unwrap();
            provider.deallocate_workspace(allocation).unwrap();
        }
        assert_eq!(
            (books.claims(), books.releases()),
            calls,
            "including zero-byte callbacks"
        );
        assert_eq!(sum.to_f32(), [1.5, 1., 4., 4., 3., 3., 3., 7.]);
        for (g, row) in [1f32, 2., 3., 4., 2., 3., 4., 5.]
            .chunks_exact(4)
            .enumerate()
        {
            let mean = row.iter().sum::<f32>() / 4.;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.;
            let inv = 1. / (var + 1e-5).sqrt();
            for (e, &value) in row.iter().enumerate() {
                assert_eq!(
                    y.to_f32()[g * 4 + e].to_bits(),
                    ((value - mean) * inv * (g + 1) as f32).to_bits()
                );
            }
        }
        assert_eq!(books.used(Tier::Host), 80);
        assert_eq!(manager.process_used(Tier::Host), 80);
        pool.trim_idle(80).unwrap();
        assert_eq!(books.used(Tier::Host), 0);
        assert_eq!(manager.process_used(Tier::Host), 0);
    }
    fn affine_fixtures(offset: f32) -> Vec<(Vec<usize>, Vec<f32>, Vec<f32>)> {
        let shapes: [&[usize]; 9] = [
            &[],
            &[4],
            &[3, 1],
            &[2, 1, 1],
            &[1, 3, 1],
            &[3, 4],
            &[2, 3, 4],
            &[1, 1, 1],
            &[1, 3, 4],
        ];
        shapes
            .iter()
            .enumerate()
            .map(|(case, shape)| {
                let n = shape.iter().product::<usize>();
                let values = (0..n)
                    .map(|i| offset + i as f32 * 0.125)
                    .collect::<Vec<_>>();
                let mut expanded = Vec::new();
                for batch in 0..2 {
                    for row in 0..3 {
                        for column in 0..4 {
                            let index = match case {
                                0 | 7 => 0,
                                1 => column,
                                2 | 4 => row,
                                3 => batch,
                                5 | 8 => row * 4 + column,
                                6 => batch * 12 + row * 4 + column,
                                _ => unreachable!(),
                            };
                            expanded.push(values[index]);
                        }
                    }
                }
                (shape.to_vec(), values, expanded)
            })
            .collect()
    }

    #[test]
    fn layernorm_affine_broadcast_matrix_matches_independent_f64_coordinates() {
        let values = (0..24)
            .map(|i| (i as f32 * 0.375).sin() * 3. + i as f32 * 0.125)
            .collect::<Vec<_>>();
        let x = Owned::f32(&[2, 3, 4], &values);
        let scales = affine_fixtures(0.5);
        let biases = affine_fixtures(-0.25);
        let mut cells = 0;
        for raw_axis in [-3, -2, -1, 0, 1, 2] {
            let axis = if raw_axis < 0 {
                (raw_axis + 3) as usize
            } else {
                raw_axis as usize
            };
            let width = [2, 3, 4][axis..].iter().product::<usize>();
            let groups = 24 / width;
            let mut reduced = [2, 3, 4];
            reduced[axis..].fill(1);
            for (ss, sv, se) in &scales {
                for (bs, bv, be) in &biases {
                    let scale = Owned::f32(ss, sv);
                    let bias = Owned::f32(bs, bv);
                    let mut y = Owned::zeros_f32(&[2, 3, 4]);
                    let mut mean = Owned::zeros_f32(&reduced);
                    let mut inv = Owned::zeros_f32(&reduced);
                    let kernel = LayerNormKernel {
                        axis: raw_axis,
                        epsilon: 1e-5,
                        kind: LayerNormKind::Standard,
                    };
                    let inputs = [x.view(), scale.view(), bias.view()];
                    let outputs = [
                        OutputMetadata::from_view(&y.view_mut()),
                        OutputMetadata::from_view(&mean.view_mut()),
                        OutputMetadata::from_view(&inv.view_mut()),
                    ];
                    let request = kernel
                        .workspace_requirement(&metadata(&inputs), &outputs)
                        .unwrap()
                        .require_ready()
                        .unwrap();
                    assert_eq!(
                        request.bytes,
                        ((24 + 2 * groups) * 4) as u64,
                        "coefficients never expand into workspace"
                    );
                    kernel
                        .execute_prepared(
                            &inputs,
                            &mut [y.view_mut(), mean.view_mut(), inv.view_mut()],
                        )
                        .unwrap();
                    let actual = y.to_f32();
                    let means = mean.to_f32();
                    let inverses = inv.to_f32();
                    for (g, row) in values.chunks_exact(width).enumerate() {
                        let m = row.iter().map(|&x| f64::from(x)).sum::<f64>() / width as f64;
                        let v = row.iter().map(|&x| (f64::from(x) - m).powi(2)).sum::<f64>()
                            / width as f64;
                        let r = 1. / (v + 1e-5).sqrt();
                        assert!((f64::from(means[g]) - m).abs() < 1e-5);
                        assert!((f64::from(inverses[g]) - r).abs() < 1e-5);
                        for (e, &value) in row.iter().enumerate() {
                            let flat = g * width + e;
                            let expected = (f64::from(value) - m) * r * f64::from(se[flat])
                                + f64::from(be[flat]);
                            assert!(
                                (f64::from(actual[flat]) - expected).abs() < 2e-5,
                                "axis={raw_axis}, Scale={ss:?}, B={bs:?}, lane={flat}: {} vs {expected}",
                                actual[flat]
                            );
                        }
                    }
                    cells += 1;
                }
            }
        }
        assert_eq!(cells, 486);
    }

    #[test]
    fn equal_parameter_and_normalized_counts_do_not_imply_row_local_coefficients() {
        let x = Owned::f32(&[2, 2], &[-1., 1., -1., 1.]);
        let group = Owned::f32(&[2, 1], &[10., 20.]);
        let unit = Owned::f32(&[2], &[1., 1.]);
        let zero = Owned::f32(&[], &[0.]);
        let mut y = Owned::zeros_f32(&[2, 2]);
        let kernel = LayerNormKernel {
            axis: -1,
            epsilon: 0.,
            kind: LayerNormKind::Standard,
        };
        kernel
            .execute_prepared(&[x.view(), group.view(), zero.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_f32(), [-10., 10., -20., 20.]);
        kernel
            .execute_prepared(&[x.view(), unit.view(), group.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_f32(), [9., 11., 19., 21.]);
    }

    #[test]
    fn affine_shape_refusals_keep_empty_guards_and_skip_contract() {
        let x = Owned::f32(&[2, 4], &[1.; 8]);
        let scale = Owned::f32(&[4], &[1.; 4]);
        let mut y = Owned::f32(&[2, 4], &[-123.; 8]);
        for shape in [&[2usize, 2][..], &[4, 1][..], &[1, 1, 4][..]] {
            let parameter = Owned::f32(shape, &vec![1.; shape.iter().product()]);
            for slot in [1, 2] {
                let inputs = if slot == 1 {
                    [x.view(), parameter.view(), scale.view()]
                } else {
                    [x.view(), scale.view(), parameter.view()]
                };
                let error = standard()
                    .execute_prepared(&inputs, &mut [y.view_mut()])
                    .unwrap_err()
                    .to_string();
                assert!(
                    error.contains(&format!("affine input {slot}"))
                        && error.contains("cannot broadcast"),
                    "{error}"
                );
                assert_eq!(y.to_f32(), [-123.; 8]);
            }
        }
        let empty = TensorMetadata::new(DataType::Float32, &[0, 4], true);
        let incompatible = TensorMetadata::new(DataType::Float32, &[2, 1], true);
        assert!(
            standard()
                .workspace_requirement(&[empty, incompatible], &[])
                .unwrap_err()
                .to_string()
                .contains("cannot broadcast")
        );
        let huge_shape = [1, isize::MAX as usize / 4 + 1];
        let huge = TensorMetadata::new(DataType::Float32, &huge_shape, true);
        let scalar = TensorMetadata::new(DataType::Float32, &[], true);
        assert!(
            standard()
                .workspace_requirement(&[huge, scalar], &[])
                .unwrap_err()
                .to_string()
                .contains("Y staging")
        );
        let scalar = Owned::f32(&[], &[1.]);
        assert!(
            LayerNormKernel::skip(1e-5)
                .execute_prepared(&[x.view(), x.view(), scalar.view()], &mut [y.view_mut()])
                .unwrap_err()
                .to_string()
                .contains("expected 4")
        );
    }

    fn affine_owned(dtype: DataType, shape: &[usize], data: &[f32]) -> Owned {
        match dtype {
            DataType::Float16 => Owned::f16(shape, data),
            DataType::BFloat16 => Owned::bf16(shape, data),
            DataType::Float32 => Owned::f32(shape, data),
            DataType::Float64 => Owned::f64(
                shape,
                &data.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
            ),
            _ => unreachable!(),
        }
    }
    fn affine_values(value: &Owned) -> Vec<f32> {
        match value.dtype {
            DataType::Float16 => value.to_f16_as_f32(),
            DataType::BFloat16 => value.to_bf16_as_f32(),
            DataType::Float32 => value.to_f32(),
            DataType::Float64 => value.to_f64().into_iter().map(|x| x as f32).collect(),
            _ => unreachable!(),
        }
    }

    #[test]
    fn broadcast_affine_preserves_supported_storage_dtypes_and_strided_half_io() {
        let values = (0..24).map(|i| i as f32 - 12.).collect::<Vec<_>>();
        for dtype in [
            DataType::Float16,
            DataType::BFloat16,
            DataType::Float32,
            DataType::Float64,
        ] {
            let x = affine_owned(dtype, &[2, 3, 4], &values);
            let scale = affine_owned(dtype, &[2, 1, 1], &[0.5, 1.5]);
            let bias = affine_owned(dtype, &[3, 1], &[-0.125, 0.25, -0.5]);
            let mut y = Owned::zeros(dtype, &[2, 3, 4]);
            let kernel = LayerNormKernel {
                axis: 1,
                epsilon: 1e-5,
                kind: LayerNormKind::Standard,
            };
            kernel
                .execute_prepared(&[x.view(), scale.view(), bias.view()], &mut [y.view_mut()])
                .unwrap();
            for (g, row) in values.chunks_exact(12).enumerate() {
                let m = row.iter().map(|&x| f64::from(x)).sum::<f64>() / 12.;
                let v = row.iter().map(|&x| (f64::from(x) - m).powi(2)).sum::<f64>() / 12.;
                let r = 1. / (v + 1e-5).sqrt();
                for (e, &x) in row.iter().enumerate() {
                    let want = (f64::from(x) - m) * r * [0.5, 1.5][g] + [-0.125, 0.25, -0.5][e / 4];
                    let got = affine_values(&y)[g * 12 + e];
                    let tolerance = if dtype == DataType::BFloat16 {
                        0.02
                    } else if dtype == DataType::Float16 {
                        0.002
                    } else {
                        1e-5
                    };
                    assert!(
                        (f64::from(got) - want).abs() < tolerance,
                        "{dtype:?} {got} vs {want}"
                    );
                }
            }
        }
        let padded = values.iter().flat_map(|&x| [x, 123.]).collect::<Vec<_>>();
        let x = Owned::f16(&[2, 3, 8], &padded).with_view(&[2, 3, 4], &[24, 8, 2]);
        let scale = Owned::f16(&[6], &[1., 456., 2., 456., 0.5, 456.]).with_view(&[3, 1], &[2, 1]);
        let bias = Owned::f16(&[], &[-0.25]);
        let mut y = Owned::f16(&[2, 3, 8], &[-123.; 48]).with_view(&[2, 3, 4], &[24, 8, 2]);
        let kernel = LayerNormKernel {
            axis: 1,
            epsilon: 1e-5,
            kind: LayerNormKind::Standard,
        };
        kernel
            .execute_prepared(&[x.view(), scale.view(), bias.view()], &mut [y.view_mut()])
            .unwrap();
        let physical = y.to_f16_as_f32();
        for (g, row) in values.chunks_exact(12).enumerate() {
            let m = row.iter().map(|&x| f64::from(x)).sum::<f64>() / 12.;
            let v = row.iter().map(|&x| (f64::from(x) - m).powi(2)).sum::<f64>() / 12.;
            let r = 1. / (v + 1e-5).sqrt();
            for (e, &x) in row.iter().enumerate() {
                let want = (f64::from(x) - m) * r * [1., 2., 0.5][e / 4] - 0.25;
                assert!((f64::from(physical[(g * 12 + e) * 2]) - want).abs() < 0.002);
                assert_eq!(physical[(g * 12 + e) * 2 + 1], -123.);
            }
        }
    }

    #[test]
    fn broadcast_coefficients_can_alias_output_without_early_overwrite() {
        use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
        use onnx_runtime_ir::DeviceId;
        let values = (0..24).map(|i| i as f32 - 12.).collect::<Vec<_>>();
        let x = Owned::f32(&[2, 3, 4], &values);
        let scale = Owned::f32(&[2, 1, 1], &[0.5, 1.5]);
        let bias = Owned::f32(&[3, 1], &[-0.125, 0.25, -0.5]);
        let mut y = Owned::zeros_f32(&[2, 3, 4]);
        let kernel = LayerNormKernel {
            axis: 1,
            epsilon: 1e-5,
            kind: LayerNormKind::Standard,
        };
        kernel
            .execute_prepared(&[x.view(), scale.view(), bias.view()], &mut [y.view_mut()])
            .unwrap();
        let expected = y.to_f32();
        for slot in 0..3 {
            let mut shared = values.clone();
            let source = if slot == 0 {
                &values[..]
            } else if slot == 1 {
                &[0.5, 1.5][..]
            } else {
                &[-0.125, 0.25, -0.5][..]
            };
            shared[..source.len()].copy_from_slice(source);
            let ptr = shared.as_mut_ptr();
            let mut inputs = [x.view(), scale.view(), bias.view()];
            let (shape, strides): (&[usize], &[i64]) = match slot {
                0 => (&[2, 3, 4], &[12, 4, 1]),
                1 => (&[2, 1, 1], &[1, 1, 1]),
                _ => (&[3, 1], &[1, 1]),
            };
            inputs[slot] = TensorView::new(
                DevicePtr(ptr.cast()),
                DataType::Float32,
                shape,
                strides,
                DeviceId::cpu(),
            );
            let out = TensorMut::new(
                DevicePtrMut(ptr.cast()),
                DataType::Float32,
                &[2, 3, 4],
                &[12, 4, 1],
                DeviceId::cpu(),
            );
            kernel.execute_prepared(&inputs, &mut [out]).unwrap();
            assert_eq!(shared, expected);
        }
    }
}
