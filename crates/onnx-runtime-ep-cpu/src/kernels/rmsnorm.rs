//! Standard `ai.onnx::RMSNormalization` (opset 23): root-mean-square layer
//! normalization without mean subtraction or bias.
//!
//! Per the ONNX reference (`onnx/reference/ops/op_rms_normalization.py`):
//!
//! ```text
//! rms = sqrt(mean(X², axes) + epsilon)      # axes = axis..rank
//! Y   = (X / rms) * scale                    # scale broadcasts over the axes
//! ```
//!
//! Unlike [`super::layernorm::LayerNormKernel`] there is **no** mean removal and
//! **no** bias term. The default `stash_type=1` path uses the existing f32 SIMD
//! implementation; schema-legal alternate stash types use a staged scalar path
//! that rounds after every operation in the requested precision. `scale` may be
//! **any** shape unidirectionally
//! (NumPy-style, right-aligned) broadcastable to `X` — scalar, the normalized
//! axes shape (`X.shape[axis:]`), or any intermediate — and is broadcast over
//! `X` before the elementwise multiply.

use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, OutputMetadata, Result, TensorMetadata, TensorMut, TensorView,
    WorkspacePreparation, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Node};

use super::check_arity;
use super::typed_workspace::TypedWorkspace;
use crate::dtype::{to_dense_f32_widen, validate_f32_widen_input, write_dense_f32_narrow};

#[derive(Clone, Copy)]
enum RmsKind {
    Standard,
    Simplified,
}

impl RmsKind {
    fn op(self) -> &'static str {
        match self {
            Self::Standard => "RMSNormalization",
            Self::Simplified => "SimplifiedLayerNormalization",
        }
    }
}

/// Both RMS normalization schemas share geometry, arithmetic and one execution lane.
pub struct RmsNormKernel {
    axis: i64,
    epsilon: f32,
    kind: RmsKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RmsNormStash {
    Float16,
    Float32,
    Float64,
    BFloat16,
}

impl RmsNormStash {
    fn from_onnx(value: i64) -> Result<Self> {
        match value {
            10 => Ok(Self::Float16),
            1 => Ok(Self::Float32),
            11 => Ok(Self::Float64),
            16 => Ok(Self::BFloat16),
            _ => Err(EpError::KernelFailed(format!(
                "RMSNormalization: stash_type {value} is invalid; expected FLOAT (1), FLOAT16 (10), DOUBLE (11), or BFLOAT16 (16)"
            ))),
        }
    }
}

struct StagedRmsNormKernel {
    axis: i64,
    epsilon: f32,
    stash: RmsNormStash,
}

/// Factory reading `axis` (default -1), `epsilon` (default 1e-5) and
/// `stash_type` (default 1).
pub struct RmsNormFactory;

impl KernelFactory for RmsNormFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1);
        let epsilon = node
            .attr("epsilon")
            .and_then(|a| a.as_float())
            .unwrap_or(1e-5);
        let stash = RmsNormStash::from_onnx(
            node.attr("stash_type")
                .and_then(|a| a.as_int())
                .unwrap_or(1),
        )?;
        if stash == RmsNormStash::Float32 {
            return Ok(Box::new(RmsNormKernel {
                axis,
                epsilon,
                kind: RmsKind::Standard,
            }));
        }
        Ok(Box::new(StagedRmsNormKernel {
            axis,
            epsilon,
            stash,
        }))
    }
}

struct RmsGeometry {
    axis: usize,
    norm_size: usize,
    groups: usize,
    count: usize,
    stats: bool,
    workspace: TypedWorkspace<f32, 2>,
}

impl RmsNormKernel {
    pub(super) fn simplified(axis: i64, epsilon: f32) -> Self {
        Self {
            axis,
            epsilon,
            kind: RmsKind::Simplified,
        }
    }

    fn geometry(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<RmsGeometry> {
        let op = self.kind.op();
        if inputs.len() != 2 || inputs.iter().any(|input| !input.present) {
            return Err(EpError::KernelFailed(format!(
                "{op}: provide two present inputs X and scale, got {} input slots",
                inputs.len()
            )));
        }
        for (name, input) in [("X", &inputs[0]), ("scale", &inputs[1])] {
            if !super::FLOAT_DTYPES.contains(&input.dtype) {
                return Err(EpError::KernelFailed(format!(
                    "{op}: {name} dtype {:?} must be floating point",
                    input.dtype
                )));
            }
        }
        let shape = inputs[0].shape;
        let scale_shape = inputs[1].shape;
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
                    "{op}: axis {} out of range for rank {rank}",
                    self.axis
                ))
            })? as usize;
        let checked_product = |dims: &[usize], name| {
            dims.iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "{op}: {name} dimensions {dims:?} overflow; reduce the tensor shape"
                    ))
                })
        };
        let norm_size = checked_product(&shape[axis..], "normalization")?;
        if norm_size == 0 {
            return Err(EpError::KernelFailed(format!(
                "{op}: empty normalization axis; use a nonzero normalized width"
            )));
        }
        let groups = checked_product(&shape[..axis], "group")?;
        let count = groups.checked_mul(norm_size).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "{op}: X shape {shape:?} overflows; reduce the tensor shape"
            ))
        })?;
        if scale_shape.len() > rank {
            return Err(EpError::KernelFailed(format!(
                "{op}: scale rank {} exceeds X rank {rank}",
                scale_shape.len()
            )));
        }
        let offset = rank - scale_shape.len();
        if scale_shape
            .iter()
            .zip(&shape[offset..])
            .any(|(&s, &x)| s != 1 && s != x)
        {
            return Err(EpError::KernelFailed(format!(
                "{op}: scale shape {scale_shape:?} not broadcastable to X shape {shape:?}"
            )));
        }
        // Check suffix strides even when an outer zero makes total storage empty.
        let mut stride = 1usize;
        for &dim in shape.iter().rev() {
            stride = stride.checked_mul(dim).ok_or_else(|| EpError::KernelFailed(format!("{op}: X shape {shape:?} has overflowing contiguous strides; reduce the tensor shape")))?;
        }
        let mut stride = 1usize;
        for &dim in scale_shape.iter().rev() {
            stride = stride.checked_mul(dim).ok_or_else(|| EpError::KernelFailed(format!("{op}: scale shape {scale_shape:?} has overflowing contiguous strides; reduce the tensor shape")))?;
        }
        let max_outputs = if matches!(self.kind, RmsKind::Simplified) {
            2
        } else {
            1
        };
        if outputs.len() > max_outputs {
            return Err(EpError::KernelFailed(format!(
                "{op}: expected at most {max_outputs} output slots, got {}",
                outputs.len()
            )));
        }
        let mut stats = false;
        for (i, output) in outputs.iter().enumerate() {
            let (dtype, output_shape) = match output {
                OutputMetadata::Absent if i == 1 => continue,
                OutputMetadata::Absent => {
                    return Err(EpError::KernelFailed(format!(
                        "{op}: required output Y is absent; bind the normalized output"
                    )));
                }
                OutputMetadata::Unresolved { dtype } => (*dtype, None),
                OutputMetadata::Tensor { dtype, shape } => (*dtype, Some(shape.as_ref())),
            };
            if !super::FLOAT_DTYPES.contains(&dtype)
                || (i == 0 && matches!(self.kind, RmsKind::Standard) && dtype != inputs[0].dtype)
            {
                return Err(EpError::KernelFailed(format!(
                    "{op}: output {i} dtype {dtype:?} is invalid; use floating storage and match X dtype {:?} for standard RMSNormalization",
                    inputs[0].dtype
                )));
            }
            if let Some(out) = output_shape {
                let valid = out.len() == rank
                    && out.iter().enumerate().all(|(d, &extent)| {
                        extent == if i == 1 && d >= axis { 1 } else { shape[d] }
                    });
                if !valid {
                    return Err(EpError::KernelFailed(format!(
                        "{op}: output {i} shape {out:?} must match {} of X {shape:?} with axis {axis}",
                        if i == 0 {
                            "the shape"
                        } else {
                            "the reduced shape"
                        }
                    )));
                }
            }
            stats |= i == 1;
        }
        let workspace = TypedWorkspace::<f32, _>::new(
            op,
            [
                ("Y staging", count),
                ("statistics", if stats { groups } else { 0 }),
            ],
        )?;
        Ok(RmsGeometry {
            axis,
            norm_size,
            groups,
            count,
            stats,
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
        check_arity(op, inputs, outputs, 2, 2, 1)?;
        let metadata = std::array::from_fn::<_, 2, _>(|i| {
            TensorMetadata::new(inputs[i].dtype, inputs[i].shape, !inputs[i].is_absent())
        });
        // At most two slots; avoid per-execution metadata allocation.
        if outputs.len() > 2 {
            return Err(EpError::KernelFailed(format!(
                "{op}: too many output slots {}; provide only declared outputs",
                outputs.len()
            )));
        }
        let output_metadata = std::array::from_fn::<_, 2, _>(|i| {
            outputs
                .get(i)
                .map_or(OutputMetadata::Absent, OutputMetadata::from_view)
        });
        let geometry = self.geometry(&metadata, &output_metadata[..outputs.len()])?;
        for (index, input) in inputs.iter().enumerate() {
            validate_f32_widen_input(op, input).map_err(|error| EpError::KernelFailed(format!(
                "{op}: input {index} ({}) view is invalid: {error}; bind a valid floating Host-accessible view", if index == 0 { "X" } else { "scale" }
            )))?;
        }
        for (index, output) in outputs
            .iter()
            .enumerate()
            .filter(|(_, output)| !output.is_absent())
        {
            output.validate().map_err(|error| EpError::KernelFailed(format!(
                "{op}: output {index} view is invalid: {error}; bind a valid floating Host-accessible output view"
            )))?;
            if !output.device.is_host_accessible() {
                return Err(EpError::KernelFailed(format!(
                    "{op}: output {index} on {:?} must be transferred to Host-accessible storage before CPU execution",
                    output.device
                )));
            }
        }
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            (geometry.count as u64)
                .saturating_mul(4)
                .saturating_add((geometry.groups as u64).saturating_mul(4))
        });
        // SAFETY: CPU execution owns exclusive live Host scratch through this
        // synchronous borrow. The descriptor validates and initializes all regions.
        unsafe {
            geometry.workspace.with_buffers(workspace, |[y, stats]| {
                if geometry.count == 0 {
                    return Ok(());
                }

                let x = to_dense_f32_widen(op, &inputs[0])?;
                let scale = to_dense_f32_widen(op, &inputs[1])?;
                rms_norm_into(
                    &geometry,
                    self.epsilon,
                    [&x, &scale],
                    [inputs[0].shape, inputs[1].shape],
                    [y, stats],
                );
                // Compute all groups before publishing, preserving X/scale aliasing.
                write_dense_f32_narrow(op, &mut outputs[0], y)?;
                if geometry.stats {
                    write_dense_f32_narrow(op, &mut outputs[1], stats)?;
                }
                Ok(())
            })
        }
    }
}

impl Kernel for RmsNormKernel {
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

impl Kernel for StagedRmsNormKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("RMSNormalization", inputs, outputs, 2, 2, 1)?;
        if outputs[0].dtype != inputs[0].dtype {
            return Err(EpError::KernelFailed(format!(
                "RMSNormalization: output dtype {:?} must match X dtype {:?}",
                outputs[0].dtype, inputs[0].dtype
            )));
        }
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let elements = inputs[0].numel() as u64;
            let rank = inputs[0].shape.len() as i64;
            let axis = if self.axis < 0 {
                self.axis + rank
            } else {
                self.axis
            };
            let groups = if axis >= 0 && axis < rank {
                onnx_runtime_ir::checked_numel(&inputs[0].shape[..axis as usize])
                    .map_or(u64::MAX, |groups| groups as u64)
            } else {
                0
            };
            elements
                .saturating_mul(4)
                .saturating_add(groups.saturating_mul(4))
        });
        let x = to_dense_f32_widen("RMSNormalization", &inputs[0])?;
        let scale = to_dense_f32_widen("RMSNormalization", &inputs[1])?;
        let y = rms_norm_dense_staged(
            &x,
            inputs[0].shape,
            inputs[0].dtype,
            &scale,
            inputs[1].shape,
            self.axis,
            self.epsilon,
            self.stash,
        )?;
        write_dense_f32_narrow("RMSNormalization", &mut outputs[0], &y)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn round_f32(value: f32, dtype: DataType) -> f32 {
    match dtype {
        DataType::Float16 => half::f16::from_f32(value).to_f32(),
        DataType::BFloat16 => half::bf16::from_f32(value).to_f32(),
        DataType::Float32 => value,
        _ => unreachable!("RMSNormalization X dtype validated by the kernel registry"),
    }
}

fn staged_f16(value: f32) -> f32 {
    half::f16::from_f32(value).to_f32()
}

fn staged_bf16(value: f32) -> f32 {
    half::bf16::from_f32(value).to_f32()
}

fn rms_norm_dense_staged(
    x: &[f32],
    x_shape: &[usize],
    x_dtype: DataType,
    scale: &[f32],
    scale_shape: &[usize],
    axis: i64,
    epsilon: f32,
    stash: RmsNormStash,
) -> Result<Vec<f32>> {
    let rank = x_shape.len();
    let axis = if axis < 0 { axis + rank as i64 } else { axis };
    if axis < 0 || axis as usize >= rank {
        return Err(EpError::KernelFailed(format!(
            "RMSNormalization: axis {axis} out of range for rank {rank}"
        )));
    }
    let axis = axis as usize;
    let norm_size: usize = x_shape[axis..].iter().product();
    let num_groups: usize = x_shape[..axis].iter().product();
    if norm_size == 0 {
        return Err(EpError::KernelFailed(
            "RMSNormalization: empty normalization axis".into(),
        ));
    }
    if scale_shape.len() > rank {
        return Err(EpError::KernelFailed(format!(
            "RMSNormalization: scale rank {} exceeds X rank {rank}",
            scale_shape.len()
        )));
    }
    let offset = rank - scale_shape.len();
    let mut scale_strides = vec![0usize; rank];
    let mut stride = 1usize;
    for i in (0..scale_shape.len()).rev() {
        let sdim = scale_shape[i];
        let xdim = x_shape[offset + i];
        if sdim != xdim && sdim != 1 {
            return Err(EpError::KernelFailed(format!(
                "RMSNormalization: scale shape {scale_shape:?} not broadcastable to X shape {x_shape:?}"
            )));
        }
        scale_strides[offset + i] = if sdim == 1 { 0 } else { stride };
        stride *= sdim;
    }
    if scale.len() != scale_shape.iter().product::<usize>() {
        return Err(EpError::KernelFailed(format!(
            "RMSNormalization: scale has {} elements, expected {} for shape {scale_shape:?}",
            scale.len(),
            scale_shape.iter().product::<usize>()
        )));
    }
    let mut x_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        x_strides[i] = x_strides[i + 1] * x_shape[i + 1];
    }
    let scale_index = |flat: usize| -> usize {
        let mut si = 0usize;
        let mut rem = flat;
        for d in 0..rank {
            let coord = rem / x_strides[d];
            rem %= x_strides[d];
            si += coord * scale_strides[d];
        }
        si
    };

    let mut y = vec![0.0f32; x.len()];
    for group in 0..num_groups {
        let base = group * norm_size;
        match stash {
            RmsNormStash::Float16 | RmsNormStash::BFloat16 => {
                let round = if stash == RmsNormStash::Float16 {
                    staged_f16
                } else {
                    staged_bf16
                };
                let mut sum = 0.0f32;
                for &value in &x[base..base + norm_size] {
                    let xu = round(value);
                    sum += round(xu * xu);
                }
                let mean = round(sum / norm_size as f32);
                let rms = round(round(mean + round(epsilon)).sqrt());
                for element in 0..norm_size {
                    let index = base + element;
                    let normalized = round(round(x[index]) / rms);
                    y[index] = round_f32(normalized, x_dtype) * scale[scale_index(index)];
                }
            }
            RmsNormStash::Float64 => {
                let mut sum = 0.0f64;
                for &value in &x[base..base + norm_size] {
                    let xu = f64::from(value);
                    sum += xu * xu;
                }
                let rms = (sum / norm_size as f64 + f64::from(epsilon)).sqrt();
                for element in 0..norm_size {
                    let index = base + element;
                    let normalized = f64::from(x[index]) / rms;
                    y[index] = round_f32(normalized as f32, x_dtype) * scale[scale_index(index)];
                }
            }
            RmsNormStash::Float32 => {
                unreachable!("the default FLOAT stash uses the existing SIMD path")
            }
        }
    }
    Ok(y)
}

/// Shared f32 arithmetic writes only to the caller's admitted staging regions.
fn rms_norm_into(
    geometry: &RmsGeometry,
    epsilon: f32,
    [x, scale]: [&[f32]; 2],
    [x_shape, scale_shape]: [&[usize]; 2],
    [y, stats]: [&mut [f32]; 2],
) {
    let mut stats = stats.iter_mut();
    let RmsGeometry {
        norm_size,
        groups,
        axis,
        ..
    } = *geometry;
    if crate::kernels::simd_normalize::scale_shape_is_exact_identity(x_shape, axis, scale_shape) {
        for g in 0..groups {
            let base = g * norm_size;
            let slice = &x[base..base + norm_size];
            let mean_sq = crate::kernels::simd_sumsq::sum_of_squares(slice) / norm_size as f32;
            let inv_rms = 1.0 / (mean_sq + epsilon).sqrt();
            if let Some(stat) = stats.next() {
                *stat = inv_rms;
            }
            crate::kernels::simd_normalize::normalize_and_scale(
                slice,
                &mut y[base..base + norm_size],
                inv_rms,
                scale,
            );
        }
    } else {
        let rank = x_shape.len();
        let offset = rank - scale_shape.len();
        let mut scale_strides = vec![0usize; rank];
        let mut stride = 1usize;
        for i in (0..scale_shape.len()).rev() {
            scale_strides[offset + i] = if scale_shape[i] == 1 { 0 } else { stride };
            stride *= scale_shape[i];
        }
        let mut x_strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            x_strides[i] = x_strides[i + 1] * x_shape[i + 1];
        }
        let scale_index = |flat: usize| {
            let mut si = 0usize;
            let mut rem = flat;
            for d in 0..rank {
                let coord = rem / x_strides[d];
                rem %= x_strides[d];
                si += coord * scale_strides[d];
            }
            si
        };
        for g in 0..groups {
            let base = g * norm_size;
            let slice = &x[base..base + norm_size];
            let mean_sq = crate::kernels::simd_sumsq::sum_of_squares(slice) / norm_size as f32;
            let inv_rms = 1.0 / (mean_sq + epsilon).sqrt();
            if let Some(stat) = stats.next() {
                *stat = inv_rms;
            }
            for e in 0..norm_size {
                let idx = base + e;
                y[idx] = x[idx] * inv_rms * scale[scale_index(idx)];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::{Owned, PreparedTestKernel};

    /// Reference RMSNorm for a single row (scale applied elementwise).
    fn reference(row: &[f32], scale: &[f32], eps: f32) -> Vec<f32> {
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        row.iter().zip(scale).map(|(v, s)| v * inv * s).collect()
    }

    #[test]
    fn rmsnorm_rejects_non_schema_stash_type() {
        let error = RmsNormStash::from_onnx(6).unwrap_err().to_string();
        assert!(
            error.contains(
                "stash_type 6 is invalid; expected FLOAT (1), FLOAT16 (10), DOUBLE (11), or BFLOAT16 (16)"
            ),
            "{error}"
        );
    }

    #[test]
    fn rmsnorm_float16_stash_rounds_each_stage() {
        let x = [300.0f32, 300.0];
        let scale = [1.0f32, 1.0];
        let staged = rms_norm_dense_staged(
            &x,
            &[1, 2],
            DataType::Float32,
            &scale,
            &[2],
            -1,
            0.0,
            RmsNormStash::Float16,
        )
        .unwrap();
        assert_eq!(staged, [0.0, 0.0], "300² overflows FLOAT16 stage one");
        assert!(reference(&x, &scale, 0.0).iter().all(|value| *value > 0.99));
    }

    #[test]
    fn rmsnorm_bfloat16_stash_casts_x_before_statistics() {
        let x = [1.001f32, 1.002];
        let scale = [1.0f32, 1.0];
        let staged = rms_norm_dense_staged(
            &x,
            &[1, 2],
            DataType::Float32,
            &scale,
            &[2],
            -1,
            0.0,
            RmsNormStash::BFloat16,
        )
        .unwrap();
        assert_eq!(
            staged,
            [1.0, 1.0],
            "both inputs cast to the same BFLOAT16 value before normalization"
        );
        let float = reference(&x, &scale, 0.0);
        assert_ne!(float[0].to_bits(), float[1].to_bits());
    }

    #[test]
    fn rmsnorm_bfloat16_stash_reduce_mean_rounds_only_the_result() {
        let x = vec![1.0f32; 512];
        let scale = vec![1.0f32; 512];
        let staged = rms_norm_dense_staged(
            &x,
            &[1, 512],
            DataType::Float32,
            &scale,
            &[512],
            -1,
            0.0,
            RmsNormStash::BFloat16,
        )
        .unwrap();
        assert!(
            staged.iter().all(|value| *value == 1.0),
            "ReduceMean accumulates the BFLOAT16 squared inputs before rounding its result"
        );
    }

    #[test]
    fn rmsnorm_double_stash_avoids_float_square_overflow() {
        let x = [1.0e20f32, 1.0e20];
        let scale = [1.0f32, 1.0];
        let staged = rms_norm_dense_staged(
            &x,
            &[1, 2],
            DataType::Float32,
            &scale,
            &[2],
            -1,
            0.0,
            RmsNormStash::Float64,
        )
        .unwrap();
        assert_eq!(staged, [1.0, 1.0]);
        assert_eq!(reference(&x, &scale, 0.0), [0.0, 0.0]);
    }

    #[test]
    fn rmsnorm_last_axis_matches_reference() {
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 2., 4., 6.]);
        let scale = Owned::f32(&[3], &[1., 1., 1.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        RmsNormKernel {
            kind: RmsKind::Standard,
            axis: -1,
            epsilon: 1e-5,
        }
        .execute_prepared(&[x.view(), scale.view()], &mut [out.view_mut()])
        .unwrap();
        let mut want = reference(&[1., 2., 3.], &[1., 1., 1.], 1e-5);
        want.extend(reference(&[2., 4., 6.], &[1., 1., 1.], 1e-5));
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }

    #[test]
    fn rmsnorm_float16_accumulates_in_float32() {
        let x = Owned::f16(&[1, 4], &[1., 2., 3., 4.]);
        let scale = Owned::f16(&[4], &[1., 1., 1., 1.]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::Float16, &[1, 4]);
        RmsNormKernel {
            kind: RmsKind::Standard,
            axis: -1,
            epsilon: 1e-5,
        }
        .execute_prepared(&[x.view(), scale.view()], &mut [out.view_mut()])
        .unwrap();
        let want = reference(&[1., 2., 3., 4.], &[1., 1., 1., 1.], 1e-5);
        for (got, want) in out.to_f16_as_f32().iter().zip(want) {
            assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
        }
    }

    #[test]
    fn rmsnorm_bfloat16_decode_and_prefill_match_widened_reference() {
        for shape in [&[1, 8][..], &[2, 3, 8][..]] {
            let element_count: usize = shape.iter().product();
            let input_values: Vec<f32> = (0..element_count)
                .map(|index| ((index * 17 % 29) as f32 - 14.0) * 0.1875)
                .collect();
            let scale_values: Vec<f32> =
                (0..8).map(|index| 0.625 + index as f32 * 0.09375).collect();
            let input = Owned::bf16(shape, &input_values);
            let scale = Owned::bf16(&[8], &scale_values);
            let mut output = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, shape);
            RmsNormKernel {
                kind: RmsKind::Standard,
                axis: -1,
                epsilon: 1e-5,
            }
            .execute_prepared(&[input.view(), scale.view()], &mut [output.view_mut()])
            .unwrap();

            let widened_input = input.to_bf16_as_f32();
            let widened_scale = scale.to_bf16_as_f32();
            let mut expected = Vec::with_capacity(element_count);
            for row in widened_input.as_chunks::<8>().0 {
                expected.extend(reference(row, &widened_scale, 1e-5));
            }
            for (index, (actual, expected)) in output
                .to_bf16_as_f32()
                .into_iter()
                .zip(expected)
                .enumerate()
            {
                let tolerance = 2e-3 + 1e-2 * expected.abs();
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "shape {shape:?}, element {index}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn rmsnorm_applies_scale() {
        let x = Owned::f32(&[1, 4], &[1., 2., 3., 4.]);
        let scale = Owned::f32(&[4], &[2., 0.5, 1., 3.]);
        let mut out = Owned::zeros_f32(&[1, 4]);
        RmsNormKernel {
            kind: RmsKind::Standard,
            axis: -1,
            epsilon: 1e-5,
        }
        .execute_prepared(&[x.view(), scale.view()], &mut [out.view_mut()])
        .unwrap();
        let want = reference(&[1., 2., 3., 4.], &[2., 0.5, 1., 3.], 1e-5);
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }

    #[test]
    fn rmsnorm_axis_and_epsilon() {
        // axis=1 over a [2,2,2]: norm_size=4, two groups.
        let x = Owned::f32(&[2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let scale = Owned::f32(&[2, 2], &[1., 1., 1., 1.]);
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        let eps = 1e-2;
        RmsNormKernel {
            kind: RmsKind::Standard,
            axis: 1,
            epsilon: eps,
        }
        .execute_prepared(&[x.view(), scale.view()], &mut [out.view_mut()])
        .unwrap();
        let mut want = reference(&[1., 2., 3., 4.], &[1., 1., 1., 1.], eps);
        want.extend(reference(&[5., 6., 7., 8.], &[1., 1., 1., 1.], eps));
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }

    #[test]
    fn rmsnorm_no_mean_subtraction() {
        // A constant row: RMSNorm(k) = k/|k| * scale = sign(k) (eps→0), which
        // differs from LayerNorm (which would give 0 after mean removal).
        let x = Owned::f32(&[1, 3], &[5., 5., 5.]);
        let scale = Owned::f32(&[3], &[1., 1., 1.]);
        let mut out = Owned::zeros_f32(&[1, 3]);
        RmsNormKernel {
            kind: RmsKind::Standard,
            axis: -1,
            epsilon: 0.0,
        }
        .execute_prepared(&[x.view(), scale.view()], &mut [out.view_mut()])
        .unwrap();
        for &g in out.to_f32().iter() {
            assert!((g - 1.0).abs() < 1e-5, "got {g}");
        }
    }

    /// Group-wise RMS over the last `norm_size` elements, then multiply by a
    /// scale that is right-aligned broadcast against `x_shape`.
    fn reference_bcast(
        x: &[f32],
        x_shape: &[usize],
        scale: &[f32],
        scale_shape: &[usize],
        axis: usize,
        eps: f32,
    ) -> Vec<f32> {
        let rank = x_shape.len();
        let norm_size: usize = x_shape[axis..].iter().product();
        let num_groups: usize = x_shape[..axis].iter().product();
        // Row-major X strides.
        let mut xs = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            xs[i] = xs[i + 1] * x_shape[i + 1];
        }
        // Right-aligned scale strides (0 where broadcast).
        let offset = rank - scale_shape.len();
        let mut ss = vec![0usize; rank];
        let mut stride = 1usize;
        for i in (0..scale_shape.len()).rev() {
            ss[offset + i] = if scale_shape[i] == 1 { 0 } else { stride };
            stride *= scale_shape[i];
        }
        let mut out = vec![0.0f32; x.len()];
        for g in 0..num_groups {
            let base = g * norm_size;
            let slice = &x[base..base + norm_size];
            let inv =
                1.0 / (slice.iter().map(|v| v * v).sum::<f32>() / norm_size as f32 + eps).sqrt();
            for e in 0..norm_size {
                let flat = base + e;
                let mut rem = flat;
                let mut si = 0usize;
                for d in 0..rank {
                    let c = rem / xs[d];
                    rem %= xs[d];
                    si += c * ss[d];
                }
                out[flat] = x[flat] * inv * scale[si];
            }
        }
        out
    }

    #[test]
    fn rmsnorm_scalar_scale_broadcasts() {
        let x_data = [1., 2., 3., 4., 5., 6.];
        let x = Owned::f32(&[2, 3], &x_data);
        let scale = Owned::f32(&[], &[2.0]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        RmsNormKernel {
            kind: RmsKind::Standard,
            axis: -1,
            epsilon: 1e-5,
        }
        .execute_prepared(&[x.view(), scale.view()], &mut [out.view_mut()])
        .unwrap();
        let want = reference_bcast(&x_data, &[2, 3], &[2.0], &[], 1, 1e-5);
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }

    #[test]
    fn rmsnorm_scale_broadcasts_last_axis() {
        // X=[2,3,4], axis=1, Scale=[4] → broadcast over groups and the axis dim.
        let x_data: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let x = Owned::f32(&[2, 3, 4], &x_data);
        let scale_data = [1., 2., 3., 4.];
        let scale = Owned::f32(&[4], &scale_data);
        let mut out = Owned::zeros_f32(&[2, 3, 4]);
        RmsNormKernel {
            kind: RmsKind::Standard,
            axis: 1,
            epsilon: 1e-5,
        }
        .execute_prepared(&[x.view(), scale.view()], &mut [out.view_mut()])
        .unwrap();
        let want = reference_bcast(&x_data, &[2, 3, 4], &scale_data, &[4], 1, 1e-5);
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-4, "got {g}, want {w}");
        }
    }

    #[test]
    fn rmsnorm_scale_broadcasts_partial_shape() {
        // X=[2,3,4], axis=1, Scale=[3,1] → broadcast over the last dim.
        let x_data: Vec<f32> = (0..24).map(|v| (v as f32) * 0.5).collect();
        let x = Owned::f32(&[2, 3, 4], &x_data);
        let scale_data = [1., 2., 3.];
        let scale = Owned::f32(&[3, 1], &scale_data);
        let mut out = Owned::zeros_f32(&[2, 3, 4]);
        RmsNormKernel {
            kind: RmsKind::Standard,
            axis: 1,
            epsilon: 1e-5,
        }
        .execute_prepared(&[x.view(), scale.view()], &mut [out.view_mut()])
        .unwrap();
        let want = reference_bcast(&x_data, &[2, 3, 4], &scale_data, &[3, 1], 1, 1e-5);
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-4, "got {g}, want {w}");
        }
    }

    #[test]
    fn rmsnorm_scale_varying_by_group_does_not_use_identity_path() {
        let x_data = [1.0f32; 4];
        let scale_data = [10.0f32, 20.0];
        assert!(
            !crate::kernels::simd_normalize::scale_shape_is_exact_identity(&[2, 2], 1, &[2, 1],)
        );

        let x = Owned::f32(&[2, 2], &x_data);
        let scale = Owned::f32(&[2, 1], &scale_data);
        let mut y = Owned::zeros_f32(&[2, 2]);
        RmsNormKernel {
            axis: 1,
            epsilon: 1e-5,
            kind: RmsKind::Standard,
        }
        .execute_prepared(&[x.view(), scale.view()], &mut [y.view_mut()])
        .unwrap();
        let output = y.to_f32();
        let inverse_rms = 1.0 / (1.0f32 + 1e-5).sqrt();
        let expected = [
            inverse_rms * 10.0,
            inverse_rms * 10.0,
            inverse_rms * 20.0,
            inverse_rms * 20.0,
        ];
        assert_eq!(
            output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn rmsnorm_scale_full_normalized_shape() {
        // X=[2,3,4], axis=1, Scale=[3,4] → full normalized-axes shape.
        let x_data: Vec<f32> = (0..24).map(|v| (v as f32) - 12.0).collect();
        let x = Owned::f32(&[2, 3, 4], &x_data);
        let scale_data: Vec<f32> = (1..13).map(|v| v as f32 * 0.25).collect();
        assert!(
            crate::kernels::simd_normalize::scale_shape_is_exact_identity(&[2, 3, 4], 1, &[3, 4],)
        );
        let scale = Owned::f32(&[3, 4], &scale_data);
        let mut out = Owned::zeros_f32(&[2, 3, 4]);
        RmsNormKernel {
            kind: RmsKind::Standard,
            axis: 1,
            epsilon: 1e-5,
        }
        .execute_prepared(&[x.view(), scale.view()], &mut [out.view_mut()])
        .unwrap();
        let mut want = Vec::with_capacity(x_data.len());
        for row in x_data.as_chunks::<12>().0 {
            let inverse_rms =
                1.0 / (crate::kernels::simd_sumsq::sum_of_squares(row) / 12.0 + 1e-5).sqrt();
            want.extend(
                row.iter()
                    .zip(&scale_data)
                    .map(|(value, scale)| value * inverse_rms * scale),
            );
        }
        assert_eq!(
            out.to_f32()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            want.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn rmsnorm_non_broadcastable_scale_errors() {
        // Scale=[3] cannot broadcast to X's last dim of 4.
        let x = Owned::f32(&[2, 4], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let scale = Owned::f32(&[3], &[1., 1., 1.]);
        let mut out = Owned::zeros_f32(&[2, 4]);
        let err = RmsNormKernel {
            kind: RmsKind::Standard,
            axis: -1,
            epsilon: 1e-5,
        }
        .execute_prepared(&[x.view(), scale.view()], &mut [out.view_mut()]);
        assert!(err.is_err());
    }

    #[test]
    fn rmsnorm_axis_equal_rank_rejected() {
        // axis == rank is out of the valid [-rank, rank-1] range.
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let scale = Owned::f32(&[3], &[1., 1., 1.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        let err = RmsNormKernel {
            kind: RmsKind::Standard,
            axis: 2,
            epsilon: 1e-5,
        }
        .execute_prepared(&[x.view(), scale.view()], &mut [out.view_mut()]);
        assert!(err.is_err(), "axis == rank must be rejected");
    }

    #[test]
    fn rmsnorm_axis_below_negative_rank_rejected() {
        // axis == -rank-1 normalises below 0 and is out of range.
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let scale = Owned::f32(&[3], &[1., 1., 1.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        let err = RmsNormKernel {
            kind: RmsKind::Standard,
            axis: -3,
            epsilon: 1e-5,
        }
        .execute_prepared(&[x.view(), scale.view()], &mut [out.view_mut()]);
        assert!(err.is_err(), "axis == -rank-1 must be rejected");
    }
    fn metadata<'a>(inputs: &'a [TensorView<'a>]) -> Vec<TensorMetadata<'a>> {
        inputs
            .iter()
            .map(|input| TensorMetadata::new(input.dtype, input.shape, !input.is_absent()))
            .collect()
    }

    #[test]
    fn rms_workspace_counts_only_requested_outputs_and_checks_geometry() {
        use onnx_runtime_ir::DataType;
        let kernel = RmsNormKernel::simplified(-1, 1e-5);
        let inputs = [
            TensorMetadata::new(DataType::Float32, &[3, 5], true),
            TensorMetadata::new(DataType::Float32, &[5], true),
        ];
        let y = OutputMetadata::tensor(DataType::Float32, &[3, 5]);
        let request = |outputs: &[OutputMetadata<'_>]| {
            kernel
                .workspace_requirement(&inputs, outputs)
                .unwrap()
                .require_ready()
                .unwrap()
        };
        assert_eq!(request(std::slice::from_ref(&y)).bytes, 60);
        assert_eq!(request(&[y.clone(), OutputMetadata::Absent]).bytes, 60);
        let with_stats = request(&[
            y.clone(),
            OutputMetadata::Unresolved {
                dtype: DataType::Float32,
            },
        ]);
        assert_eq!(with_stats.bytes, 72);
        assert_eq!(with_stats.alignment, 4);
        assert_eq!(
            with_stats.lifetime,
            onnx_runtime_ep_api::WorkspaceLifetime::StepScoped
        );
        for bad in [
            OutputMetadata::tensor(DataType::Float32, &[5, 3]),
            OutputMetadata::tensor(DataType::Int64, &[3, 5]),
        ] {
            assert!(kernel.workspace_requirement(&inputs, &[bad]).is_err());
        }
        assert!(
            kernel
                .workspace_requirement(
                    &inputs,
                    &[
                        y.clone(),
                        OutputMetadata::tensor(DataType::Float32, &[1, 3])
                    ]
                )
                .is_err()
        );
        for shape in [
            &[0usize, usize::MAX, 2][..],
            &[1, isize::MAX as usize / 4 + 1][..],
        ] {
            let meta = [
                TensorMetadata::new(DataType::Float32, shape, true),
                TensorMetadata::new(DataType::Float32, &[1], true),
            ];
            assert!(kernel.workspace_requirement(&meta, &[]).is_err());
        }
        let standard = RmsNormKernel {
            axis: -1,
            epsilon: 1e-5,
            kind: RmsKind::Standard,
        };
        assert!(
            standard
                .workspace_requirement(
                    &inputs,
                    &[
                        y,
                        OutputMetadata::Unresolved {
                            dtype: DataType::Float32
                        }
                    ]
                )
                .is_err()
        );
    }

    #[test]
    fn rms_workspace_initializes_fresh_backing_and_refuses_before_output_writes() {
        use onnx_runtime_ep_api::DevicePtrMut;
        use std::mem::MaybeUninit;
        let kernel = RmsNormKernel::simplified(-1, 1e-5);
        let x = Owned::f32(&[2, 4], &[1., -2., 3., -4., 2., -3., 4., -5.]);
        let scale = Owned::f32(&[4], &[1., 2., 0.5, -1.]);
        let mut y = Owned::f32(&[2, 4], &[-12345.; 8]);
        let mut stats = Owned::f32(&[2, 1], &[-9876.; 2]);
        let mut backing = [0x7fc0_1234u32; 12];
        let ptr = unsafe { backing.as_mut_ptr().add(1).cast::<u8>() };
        let valid = WorkspaceView::new(DevicePtrMut(ptr.cast()), 40);
        for bad in [
            None,
            Some(WorkspaceView::new(DevicePtrMut(ptr.cast()), 39)),
            Some(WorkspaceView::new(
                DevicePtrMut(unsafe { ptr.add(1).cast() }),
                40,
            )),
        ] {
            let error = kernel
                .execute_with_workspace(
                    &[x.view(), scale.view()],
                    &mut [y.view_mut(), stats.view_mut()],
                    bad,
                )
                .unwrap_err();
            assert!(error.to_string().contains("workspace"), "{error}");
            assert_eq!(y.to_f32(), [-12345.; 8]);
            assert_eq!(stats.to_f32(), [-9876.; 2]);
        }
        kernel
            .execute_with_workspace(
                &[x.view(), scale.view()],
                &mut [y.view_mut(), stats.view_mut()],
                Some(valid),
            )
            .unwrap();
        assert_eq!(backing[0], 0x7fc0_1234);
        assert_eq!(backing[11], 0x7fc0_1234);
        let expected_y = y.to_f32();
        let expected_stats = stats.to_f32();
        let mut fresh = [MaybeUninit::<f32>::uninit(); 10];
        kernel
            .execute_with_workspace(
                &[x.view(), scale.view()],
                &mut [y.view_mut(), stats.view_mut()],
                Some(WorkspaceView::new(
                    DevicePtrMut(fresh.as_mut_ptr().cast()),
                    40,
                )),
            )
            .unwrap();
        assert_eq!(y.to_f32(), expected_y);
        assert_eq!(stats.to_f32(), expected_stats);
        // The same retained lane can omit stats without touching its old suffix.
        backing[9..].fill(0x7fc0_1234);
        kernel
            .execute_with_workspace(
                &[x.view(), scale.view()],
                &mut [y.view_mut(), stats.view_mut().mark_absent()],
                Some(valid),
            )
            .unwrap();
        assert_eq!(y.to_f32(), expected_y);
        assert!(backing[9..].iter().all(|&bits| bits == 0x7fc0_1234));
    }

    #[test]
    fn rms_outputs_may_alias_x_or_scale_without_changing_statistics() {
        use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
        use onnx_runtime_ir::{DataType, DeviceId};
        let kernel = RmsNormKernel::simplified(-1, 1e-5);
        let values = [1., 2., 3., 4., -2., -3., 4., 5.];
        let scales = [1., 2., 0.5, -1.];
        let x = Owned::f32(&[2, 4], &values);
        let scale = Owned::f32(&[4], &scales);
        let mut y = Owned::zeros_f32(&[2, 4]);
        let mut stats = Owned::zeros_f32(&[2, 1]);
        kernel
            .execute_prepared(
                &[x.view(), scale.view()],
                &mut [y.view_mut(), stats.view_mut()],
            )
            .unwrap();
        let expected_y = y.to_f32();
        let expected_stats = stats.to_f32();
        for alias in 0..3 {
            let mut shared = values;
            if alias == 1 {
                shared[..4].copy_from_slice(&scales);
            }
            let ptr = shared.as_mut_ptr();
            let input = if alias != 1 {
                TensorView::new(
                    DevicePtr(ptr.cast()),
                    DataType::Float32,
                    &[2, 4],
                    &[4, 1],
                    DeviceId::cpu(),
                )
            } else {
                x.view()
            };
            let scale_view = if alias == 1 {
                TensorView::new(
                    DevicePtr(ptr.cast()),
                    DataType::Float32,
                    &[4],
                    &[1],
                    DeviceId::cpu(),
                )
            } else {
                scale.view()
            };
            let out = if alias != 2 {
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
            let stat = if alias == 2 {
                TensorMut::new(
                    DevicePtrMut(ptr.cast()),
                    DataType::Float32,
                    &[2, 1],
                    &[1, 1],
                    DeviceId::cpu(),
                )
            } else {
                stats.view_mut()
            };
            kernel
                .execute_prepared(&[input, scale_view], &mut [out, stat])
                .unwrap();
            assert_eq!(
                if alias != 2 {
                    shared.to_vec()
                } else {
                    y.to_f32()
                },
                expected_y
            );
            assert_eq!(
                if alias == 2 {
                    shared[..2].to_vec()
                } else {
                    stats.to_f32()
                },
                expected_stats
            );
        }
    }

    #[test]
    fn rms_empty_groups_skip_scratch_but_keep_view_and_width_guards() {
        let kernel = RmsNormKernel::simplified(-1, 1e-5);
        let x = Owned::zeros_f32(&[0, 4]);
        let scale = Owned::f32(&[4], &[1.; 4]);
        let mut y = Owned::zeros_f32(&[0, 4]);
        let mut stats = Owned::zeros_f32(&[0, 1]);
        kernel
            .execute(
                &[x.view(), scale.view()],
                &mut [y.view_mut(), stats.view_mut()],
            )
            .unwrap();
        let mut invalid = scale.view();
        invalid.device = onnx_runtime_ir::DeviceId::cuda(0);
        assert!(
            kernel
                .execute(&[x.view(), invalid], &mut [y.view_mut()])
                .unwrap_err()
                .to_string()
                .contains("host-accessible")
        );
        let mut invalid = x.view();
        invalid.strides = &[];
        let error = kernel
            .execute(&[invalid, scale.view()], &mut [y.view_mut()])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("SimplifiedLayerNormalization: input 0 (X)")
                && error.contains("rank mismatch"),
            "{error}"
        );
        let mut bad_output = y.view_mut();
        bad_output.strides = &[];
        let error = kernel
            .execute(&[x.view(), scale.view()], &mut [bad_output])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("SimplifiedLayerNormalization: output 0")
                && error.contains("rank mismatch"),
            "{error}"
        );
        let zero_width = Owned::zeros_f32(&[2, 0]);
        let mut zero_out = Owned::zeros_f32(&[2, 0]);
        let unit_scale = Owned::f32(&[], &[1.]);
        assert!(
            kernel
                .execute(
                    &[zero_width.view(), unit_scale.view()],
                    &mut [zero_out.view_mut()]
                )
                .unwrap_err()
                .to_string()
                .contains("empty normalization axis")
        );
    }

    #[test]
    fn rms_statistics_share_the_exact_inverse_used_by_both_math_paths() {
        for (scale_shape, scale_values) in [
            (&[1usize, 1, 9][..], vec![0.75; 9]),
            (&[2usize, 1, 1][..], vec![0.5, 1.5]),
        ] {
            let values = (0..54)
                .map(|i| (i as f32 * 0.275).sin() * 3.)
                .collect::<Vec<_>>();
            let x = Owned::f32(&[2, 3, 9], &values);
            let scale = Owned::f32(scale_shape, &scale_values);
            let mut y = Owned::zeros_f32(&[2, 3, 9]);
            let mut stats = Owned::zeros_f32(&[2, 3, 1]);
            RmsNormKernel::simplified(-1, 1e-5)
                .execute_prepared(
                    &[x.view(), scale.view()],
                    &mut [y.view_mut(), stats.view_mut()],
                )
                .unwrap();
            for (actual, row) in stats.to_f32().iter().zip(values.chunks_exact(9)) {
                let expected =
                    1.0 / (crate::kernels::simd_sumsq::sum_of_squares(row) / 9. + 1e-5).sqrt();
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn governed_rms_staging_reuses_one_lane_without_warm_issuer_callbacks() {
        use crate::kernels::testutil::RecordingGovernor;
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{
            HolderId, HostWorkspacePool, MemoryGovernor, ProcessMemoryManager,
        };
        use std::sync::Arc;
        let books = Arc::new(RecordingGovernor::new_host(40));
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
        let kernel = RmsNormKernel::simplified(-1, 1e-5);
        let x = Owned::f32(&[2, 4], &[1., 2., 3., 4., 2., 4., 6., 8.]);
        let scale = Owned::f32(&[4], &[1.; 4]);
        let inputs = [x.view(), scale.view()];
        let meta = metadata(&inputs);
        let mut y = Owned::zeros_f32(&[2, 4]);
        let mut stats = Owned::zeros_f32(&[2, 1]);
        let outputs = [
            OutputMetadata::tensor(onnx_runtime_ir::DataType::Float32, &[2, 4]),
            OutputMetadata::tensor(onnx_runtime_ir::DataType::Float32, &[2, 1]),
        ];
        let request = kernel
            .workspace_requirement(&meta, &outputs)
            .unwrap()
            .require_ready()
            .unwrap();
        assert_eq!(request.bytes, 40);
        let first = provider
            .allocate_workspace(40, request.alignment, request.role())
            .unwrap();
        let identity = first.managed_owner().unwrap().identity();
        assert!(
            provider
                .allocate_workspace(40, request.alignment, request.role())
                .is_err(),
            "active capacity cannot fund a concurrent execution"
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
        for _ in 0..4096 {
            let request = kernel
                .workspace_requirement_for_execution(&inputs, &[], &meta, &outputs, None)
                .unwrap()
                .require_ready()
                .unwrap();
            let mut allocation = provider
                .allocate_workspace(request.bytes as usize, request.alignment, request.role())
                .unwrap();
            assert_eq!(allocation.managed_owner().unwrap().identity(), identity);
            let workspace = WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(allocation.as_mut_ptr()),
                allocation.len(),
            );
            kernel
                .execute_with_workspace(
                    &inputs,
                    &mut [y.view_mut(), stats.view_mut()],
                    Some(workspace),
                )
                .unwrap();
            provider.deallocate_workspace(allocation).unwrap();
        }
        assert_eq!(
            (books.claims(), books.releases()),
            calls,
            "including zero-byte callbacks"
        );
        assert_eq!(books.used(Tier::Host), 40);
        let expected = 1. / (7.5f32 + 1e-5).sqrt();
        assert_eq!(stats.to_f32()[0].to_bits(), expected.to_bits());
        for (&actual, value) in y.to_f32()[..4].iter().zip([1., 2., 3., 4.]) {
            assert_eq!(actual.to_bits(), (value * expected).to_bits());
        }
        pool.trim_idle(40).unwrap();
        assert_eq!(books.used(Tier::Host), 0);
        assert_eq!(manager.process_used(Tier::Host), 0);
    }
    #[test]
    fn rms_strided_half_views_and_outputs_keep_logical_order_and_padding() {
        let dense_values = [1., 2., -3., 4., 2., 4., -6., 8.];
        let dense_scale = [1., 2., 0.5, -1.];
        let values = dense_values
            .into_iter()
            .flat_map(|v| [v, 123.])
            .collect::<Vec<_>>();
        let scales = dense_scale
            .into_iter()
            .flat_map(|v| [v, 456.])
            .collect::<Vec<_>>();
        let x = Owned::f16(&[2, 8], &values).with_view(&[2, 4], &[8, 2]);
        let scale = Owned::f16(&[8], &scales).with_view(&[4], &[2]);
        let mut y = Owned::f16(&[2, 8], &[-123.; 16]).with_view(&[2, 4], &[8, 2]);
        let mut stats = Owned::f16(&[2, 2], &[-456.; 4]).with_view(&[2, 1], &[2, 1]);
        RmsNormKernel::simplified(-1, 1e-5)
            .execute_prepared(
                &[x.view(), scale.view()],
                &mut [y.view_mut(), stats.view_mut()],
            )
            .unwrap();
        let got = y.to_u16_bits();
        let got_stats = stats.to_u16_bits();
        for g in 0..2 {
            let row = &dense_values[g * 4..g * 4 + 4];
            let inverse = 1. / (crate::kernels::simd_sumsq::sum_of_squares(row) / 4. + 1e-5).sqrt();
            for e in 0..4 {
                assert_eq!(
                    got[(g * 4 + e) * 2],
                    half::f16::from_f32(row[e] * inverse * dense_scale[e]).to_bits()
                );
                assert_eq!(
                    got[(g * 4 + e) * 2 + 1],
                    half::f16::from_f32(-123.).to_bits()
                );
            }
            assert_eq!(got_stats[g * 2], half::f16::from_f32(inverse).to_bits());
            assert_eq!(got_stats[g * 2 + 1], half::f16::from_f32(-456.).to_bits());
        }
    }
}
