//! Value selection kernels: `Clip`, `ArgMax`, `ArgMin`, `TopK`, and `NonZero`.

use core::cmp::Ordering;
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::add::require_same_dtype;
use super::{check_arity, to_dense_f32, to_dense_i64, write_dense_bytes};
use crate::dispatch_arith;
use crate::dtype::{
    NumericElem, to_dense, to_dense_f32_widen, write_dense, write_dense_f32_narrow,
};
use crate::strided::numel;

pub struct ClipKernel {
    min: Option<f32>,
    max: Option<f32>,
}
pub struct ClipFactory;
impl KernelFactory for ClipFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ClipKernel {
            min: node.attr("min").and_then(|a| a.as_float()),
            max: node.attr("max").and_then(|a| a.as_float()),
        }))
    }
}
impl Kernel for ClipKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Clip", inputs, outputs, 1, 3, 1)?;
        #[cfg(feature = "mlas")]
        if clip_contiguous_f32(self, inputs, &mut outputs[0])? {
            return Ok(());
        }
        dispatch_arith!(inputs[0].dtype, "Clip", T => {
            clip_typed::<T>(self, inputs, outputs)
        })
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

#[cfg(feature = "mlas")]
fn clip_contiguous_f32(
    kernel: &ClipKernel,
    inputs: &[TensorView],
    output: &mut TensorMut,
) -> Result<bool> {
    if inputs[0].dtype != DataType::Float32
        || output.dtype != DataType::Float32
        || inputs[0].shape != output.shape
        || !inputs[0].is_contiguous()
        || !output.is_contiguous()
    {
        return Ok(false);
    }
    let minimum = if inputs.len() > 1 && !inputs[1].is_absent() {
        require_same_dtype("Clip", &inputs[1], DataType::Float32)?;
        scalar_typed::<f32>("Clip min", &inputs[1])?
    } else {
        kernel.min.unwrap_or(f32::NEG_INFINITY)
    };
    let maximum = if inputs.len() > 2 && !inputs[2].is_absent() {
        require_same_dtype("Clip", &inputs[2], DataType::Float32)?;
        scalar_typed::<f32>("Clip max", &inputs[2])?
    } else {
        kernel.max.unwrap_or(f32::INFINITY)
    };
    if minimum > maximum {
        return Err(EpError::KernelFailed(
            "Clip: min must not exceed max".into(),
        ));
    }
    let input_start = inputs[0].data_ptr::<u8>() as usize;
    let input_end = input_start.saturating_add(inputs[0].byte_size());
    let output_start = output.data_ptr_mut::<u8>() as usize;
    let output_end = output_start.saturating_add(output.byte_size());
    if output_start < input_end && input_start < output_end {
        return Ok(false);
    }
    let input = to_dense_f32_widen("Clip", &inputs[0])?;
    let output_len = output.numel();
    // SAFETY: equal contiguous Float32 shapes prove the output span, and the
    // range check proves it does not overlap the borrowed input.
    let output =
        unsafe { std::slice::from_raw_parts_mut(output.data_ptr_mut::<f32>(), output_len) };
    mlas_sys::compute_clip(&input, output, minimum, maximum);
    Ok(true)
}

fn clip_typed<T: NumericElem + PartialOrd>(
    kernel: &ClipKernel,
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
) -> Result<()> {
    if outputs[0].dtype != T::DTYPE {
        return Err(EpError::KernelFailed(format!(
            "Clip: output dtype {:?} must match input dtype {:?}",
            outputs[0].dtype,
            T::DTYPE
        )));
    }
    let min = if inputs.len() > 1 && !inputs[1].is_absent() {
        require_same_dtype("Clip", &inputs[1], T::DTYPE)?;
        Some(scalar_typed::<T>("Clip min", &inputs[1])?)
    } else {
        kernel.min.map(T::from_f32_scalar)
    };
    let max = if inputs.len() > 2 && !inputs[2].is_absent() {
        require_same_dtype("Clip", &inputs[2], T::DTYPE)?;
        Some(scalar_typed::<T>("Clip max", &inputs[2])?)
    } else {
        kernel.max.map(T::from_f32_scalar)
    };
    if let (Some(min), Some(max)) = (min, max)
        && min > max
    {
        return Err(EpError::KernelFailed(
            "Clip: min must not exceed max".into(),
        ));
    }

    let y = to_dense::<T>(&inputs[0])?
        .into_iter()
        .map(|x| {
            let x = if let Some(min) = min {
                if x < min { min } else { x }
            } else {
                x
            };
            if let Some(max) = max {
                if x > max { max } else { x }
            } else {
                x
            }
        })
        .collect::<Vec<_>>();
    write_dense::<T>(&mut outputs[0], &y)
}

fn scalar_typed<T: NumericElem>(name: &str, view: &TensorView) -> Result<T> {
    let x = to_dense::<T>(view)?;
    if x.len() == 1 {
        Ok(x[0])
    } else {
        Err(EpError::KernelFailed(format!("{name} must be a scalar")))
    }
}

#[derive(Clone, Copy)]
enum ArgOp {
    Max,
    Min,
}
pub struct ArgKernel {
    op: ArgOp,
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
}
pub struct ArgMaxFactory;
pub struct ArgMinFactory;
fn arg_factory(node: &Node, op: ArgOp) -> Box<dyn Kernel> {
    Box::new(ArgKernel {
        op,
        axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(0),
        keepdims: node.attr("keepdims").and_then(|a| a.as_int()).unwrap_or(1) != 0,
        select_last_index: node
            .attr("select_last_index")
            .and_then(|a| a.as_int())
            .unwrap_or(0)
            != 0,
    })
}
impl KernelFactory for ArgMaxFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(arg_factory(node, ArgOp::Max))
    }
}
impl KernelFactory for ArgMinFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(arg_factory(node, ArgOp::Min))
    }
}
impl Kernel for ArgKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let name = match self.op {
            ArgOp::Max => "ArgMax",
            ArgOp::Min => "ArgMin",
        };
        check_arity(name, inputs, outputs, 1, 1, 1)?;
        if outputs[0].dtype != DataType::Int64 {
            return Err(EpError::KernelFailed(format!(
                "{name}: output must be Int64"
            )));
        }
        let x = to_dense_f32_widen(name, &inputs[0])?;
        let axis = axis(name, self.axis, inputs[0].shape.len())?;
        let width = inputs[0].shape[axis];
        if width == 0 {
            return Err(EpError::KernelFailed(format!(
                "{name}: reduced axis must be non-empty"
            )));
        }
        let inner = numel(&inputs[0].shape[axis + 1..]);
        let mut out = Vec::with_capacity(numel(outputs[0].shape));
        for outer in 0..numel(&inputs[0].shape[..axis]) {
            for i in 0..inner {
                let mut best = 0;
                for d in 1..width {
                    let candidate = x[(outer * width + d) * inner + i];
                    let value = x[(outer * width + best) * inner + i];
                    let better = match self.op {
                        ArgOp::Max => candidate > value,
                        ArgOp::Min => candidate < value,
                    };
                    if better || (self.select_last_index && candidate == value) {
                        best = d;
                    }
                }
                out.push(best as i64);
            }
        }
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = self.keepdims;
        write_dense_bytes(&mut outputs[0], &bytes)
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

pub struct TopKKernel {
    axis: i64,
    largest: bool,
    sorted: bool,
}

/// ONNX `NonMaxSuppression` for the standard `[batch, boxes, 4]` /
/// `[batch, classes, boxes]` representation.
pub struct NonMaxSuppressionKernel {
    center_point_box: i64,
}

pub struct NonMaxSuppressionFactory;

impl KernelFactory for NonMaxSuppressionFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let center_point_box = node
            .attr("center_point_box")
            .and_then(|attribute| attribute.as_int())
            .unwrap_or(0);
        if !matches!(center_point_box, 0 | 1) {
            return Err(EpError::KernelFailed(format!(
                "NonMaxSuppression: center_point_box must be 0 or 1, got {center_point_box}"
            )));
        }
        Ok(Box::new(NonMaxSuppressionKernel { center_point_box }))
    }
}

impl Kernel for NonMaxSuppressionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("NonMaxSuppression", inputs, outputs, 2, 5, 1)?;
        if outputs[0].dtype != DataType::Int64 {
            return Err(EpError::KernelFailed(
                "NonMaxSuppression: selected_indices output must be Int64".into(),
            ));
        }
        let boxes = to_dense_f32(&inputs[0])?;
        let scores = to_dense_f32(&inputs[1])?;
        let max_output_boxes_per_class = optional_i64_scalar(
            "NonMaxSuppression max_output_boxes_per_class",
            inputs.get(2),
        )?
        .unwrap_or(0);
        let iou_threshold =
            optional_f32_scalar("NonMaxSuppression iou_threshold", inputs.get(3))?.unwrap_or(0.0);
        let score_threshold =
            optional_f32_scalar("NonMaxSuppression score_threshold", inputs.get(4))?
                .unwrap_or(f32::NEG_INFINITY);
        let selected = non_max_suppression(
            &boxes,
            inputs[0].shape,
            &scores,
            inputs[1].shape,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            self.center_point_box,
        )?;
        if outputs[0].shape != [selected.len(), 3] {
            return Err(EpError::KernelFailed(format!(
                "NonMaxSuppression: output shape {:?} does not match selected index count {}. \
                 HOW: size selected_indices as [num_selected_indices, 3].",
                outputs[0].shape,
                selected.len()
            )));
        }
        let bytes = selected
            .iter()
            .flat_map(|indices| indices.iter().flat_map(|index| index.to_le_bytes()))
            .collect::<Vec<_>>();
        write_dense_bytes(&mut outputs[0], &bytes)
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

/// Calculate the selected `[batch_index, class_index, box_index]` rows for
/// ONNX `NonMaxSuppression`. The session uses this same reference routine to
/// resolve the data-dependent output extent before allocating its output.
pub fn non_max_suppression(
    boxes: &[f32],
    boxes_shape: &[usize],
    scores: &[f32],
    scores_shape: &[usize],
    max_output_boxes_per_class: i64,
    iou_threshold: f32,
    score_threshold: f32,
    center_point_box: i64,
) -> Result<Vec<[i64; 3]>> {
    if boxes_shape.len() != 3 || boxes_shape[2] != 4 {
        return Err(EpError::KernelFailed(format!(
            "NonMaxSuppression: boxes must have shape [batch, spatial_dimension, 4], got {boxes_shape:?}"
        )));
    }
    if scores_shape.len() != 3
        || scores_shape[0] != boxes_shape[0]
        || scores_shape[2] != boxes_shape[1]
    {
        return Err(EpError::KernelFailed(format!(
            "NonMaxSuppression: scores shape {scores_shape:?} must be [batch, classes, spatial_dimension] matching boxes shape {boxes_shape:?}"
        )));
    }
    if max_output_boxes_per_class < 0 {
        return Err(EpError::KernelFailed(
            "NonMaxSuppression: max_output_boxes_per_class must be non-negative".into(),
        ));
    }
    let batch = boxes_shape[0];
    let box_count = boxes_shape[1];
    let classes = scores_shape[1];
    if boxes.len() != batch * box_count * 4 || scores.len() != batch * classes * box_count {
        return Err(EpError::KernelFailed(
            "NonMaxSuppression: input data length does not match its declared shape".into(),
        ));
    }
    let limit = usize::try_from(max_output_boxes_per_class).map_err(|_| {
        EpError::KernelFailed(
            "NonMaxSuppression: max_output_boxes_per_class exceeds this platform's address space"
                .into(),
        )
    })?;
    let mut selected = Vec::new();
    for batch_index in 0..batch {
        for class_index in 0..classes {
            let score_offset = (batch_index * classes + class_index) * box_count;
            let mut candidates = (0..box_count)
                .filter(|&box_index| scores[score_offset + box_index] > score_threshold)
                .collect::<Vec<_>>();
            candidates.sort_by(|&left, &right| {
                scores[score_offset + right]
                    .total_cmp(&scores[score_offset + left])
                    .then_with(|| left.cmp(&right))
            });
            let mut kept = Vec::new();
            for candidate in candidates {
                if kept.len() == limit {
                    break;
                }
                let candidate_box = box_coordinates(
                    &boxes[(batch_index * box_count + candidate) * 4..][..4],
                    center_point_box,
                );
                if kept.iter().all(|&kept_index| {
                    let kept_box = box_coordinates(
                        &boxes[(batch_index * box_count + kept_index) * 4..][..4],
                        center_point_box,
                    );
                    iou(candidate_box, kept_box) <= iou_threshold
                }) {
                    kept.push(candidate);
                    selected.push([batch_index as i64, class_index as i64, candidate as i64]);
                }
            }
        }
    }
    Ok(selected)
}

fn optional_i64_scalar(name: &str, input: Option<&TensorView>) -> Result<Option<i64>> {
    let Some(input) = input.filter(|input| !input.is_absent()) else {
        return Ok(None);
    };
    let values = to_dense_i64(input)?;
    if values.len() != 1 {
        return Err(EpError::KernelFailed(format!(
            "{name}: expected an Int64 scalar"
        )));
    }
    Ok(Some(values[0]))
}

fn optional_f32_scalar(name: &str, input: Option<&TensorView>) -> Result<Option<f32>> {
    let Some(input) = input.filter(|input| !input.is_absent()) else {
        return Ok(None);
    };
    let values = to_dense_f32(input)?;
    if values.len() != 1 {
        return Err(EpError::KernelFailed(format!(
            "{name}: expected a Float32 scalar"
        )));
    }
    Ok(Some(values[0]))
}

fn box_coordinates(values: &[f32], center_point_box: i64) -> (f32, f32, f32, f32) {
    if center_point_box == 1 {
        let (x_center, y_center, width, height) = (values[0], values[1], values[2], values[3]);
        (
            y_center - height * 0.5,
            x_center - width * 0.5,
            y_center + height * 0.5,
            x_center + width * 0.5,
        )
    } else {
        (
            values[0].min(values[2]),
            values[1].min(values[3]),
            values[0].max(values[2]),
            values[1].max(values[3]),
        )
    }
}

fn iou(left: (f32, f32, f32, f32), right: (f32, f32, f32, f32)) -> f32 {
    let intersection_height = (left.2.min(right.2) - left.0.max(right.0)).max(0.0);
    let intersection_width = (left.3.min(right.3) - left.1.max(right.1)).max(0.0);
    let intersection = intersection_height * intersection_width;
    let left_area = (left.2 - left.0).max(0.0) * (left.3 - left.1).max(0.0);
    let right_area = (right.2 - right.0).max(0.0) * (right.3 - right.1).max(0.0);
    let union = left_area + right_area - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}
pub struct TopKFactory;
impl KernelFactory for TopKFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(TopKKernel {
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1),
            largest: node.attr("largest").and_then(|a| a.as_int()).unwrap_or(1) != 0,
            sorted: node.attr("sorted").and_then(|a| a.as_int()).unwrap_or(1) != 0,
        }))
    }
}
impl Kernel for TopKKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("TopK", inputs, outputs, 2, 2, 2)?;
        if outputs[1].dtype != DataType::Int64 {
            return Err(EpError::KernelFailed(
                "TopK: indices output must be Int64".into(),
            ));
        }
        if outputs[0].dtype != inputs[0].dtype {
            return Err(EpError::KernelFailed(format!(
                "TopK: values output dtype {:?} must match input dtype {:?}",
                outputs[0].dtype, inputs[0].dtype
            )));
        }
        let x = to_dense_f32_widen("TopK", &inputs[0])?;
        let k_values = to_dense_i64(&inputs[1])?;
        if k_values.len() != 1 || k_values[0] < 0 {
            return Err(EpError::KernelFailed(
                "TopK: K must be a non-negative scalar".into(),
            ));
        }
        let axis = axis("TopK", self.axis, inputs[0].shape.len())?;
        let width = inputs[0].shape[axis];
        let k = k_values[0] as usize;
        if k > width {
            return Err(EpError::KernelFailed(
                "TopK: K exceeds selected axis".into(),
            ));
        }
        let inner = numel(&inputs[0].shape[axis + 1..]);
        let mut values = Vec::with_capacity(numel(outputs[0].shape));
        let mut indices = Vec::with_capacity(numel(outputs[1].shape));
        for outer in 0..numel(&inputs[0].shape[..axis]) {
            for i in 0..inner {
                let mut candidates: Vec<usize> = (0..width).collect();
                candidates.sort_by(|&a, &b| {
                    topk_order(
                        x[(outer * width + a) * inner + i],
                        x[(outer * width + b) * inner + i],
                        a,
                        b,
                        self.largest,
                    )
                });
                if !self.sorted {
                    candidates.truncate(k);
                }
                for d in candidates.into_iter().take(k) {
                    values.push(x[(outer * width + d) * inner + i]);
                    indices.push(d as i64);
                }
            }
        }
        write_dense_f32_narrow("TopK", &mut outputs[0], &values)?;
        write_dense_bytes(
            &mut outputs[1],
            &indices
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}
fn topk_order(a: f32, b: f32, ia: usize, ib: usize, largest: bool) -> Ordering {
    let order = if largest {
        b.total_cmp(&a)
    } else {
        a.total_cmp(&b)
    };
    if order == Ordering::Equal {
        ia.cmp(&ib)
    } else {
        order
    }
}

pub struct NonZeroKernel;
pub struct NonZeroFactory;
impl KernelFactory for NonZeroFactory {
    fn create(&self, _: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(NonZeroKernel))
    }
}
impl Kernel for NonZeroKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("NonZero", inputs, outputs, 1, 1, 1)?;
        if outputs[0].dtype != DataType::Int64 {
            return Err(EpError::KernelFailed(
                "NonZero: output must be Int64".into(),
            ));
        }
        let x = to_dense_f32_widen("NonZero", &inputs[0])?;
        let rank = inputs[0].shape.len();
        let strides = contiguous(inputs[0].shape);
        let mut coordinates = vec![Vec::new(); rank];
        for (linear, &v) in x.iter().enumerate() {
            if v != 0. {
                let mut rem = linear;
                for d in 0..rank {
                    coordinates[d].push((rem / strides[d]) as i64);
                    rem %= strides[d];
                }
            }
        }
        let bytes: Vec<u8> = coordinates
            .into_iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        write_dense_bytes(&mut outputs[0], &bytes)
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}
fn axis(name: &str, raw: i64, rank: usize) -> Result<usize> {
    let a = if raw < 0 { raw + rank as i64 } else { raw };
    if a < 0 || a as usize >= rank {
        Err(EpError::KernelFailed(format!("{name}: axis out of range")))
    } else {
        Ok(a as usize)
    }
}
fn contiguous(shape: &[usize]) -> Vec<usize> {
    let mut r = vec![1; shape.len()];
    for d in (0..shape.len()).rev().skip(1) {
        r[d] = r[d + 1] * shape[d + 1];
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    #[test]
    fn clip_tensor_bounds() {
        let x = Owned::f32(&[3], &[-2., 0.5, 3.]);
        let lo = Owned::f32(&[], &[0.]);
        let hi = Owned::f32(&[], &[1.]);
        let mut y = Owned::zeros_f32(&[3]);
        ClipKernel {
            min: None,
            max: None,
        }
        .execute(&[x.view(), lo.view(), hi.view()], &mut [y.view_mut()])
        .unwrap();
        assert_eq!(y.to_f32(), vec![0., 0.5, 1.]);
    }

    #[test]
    fn clip_supports_int8_defaults_and_f16_tensor_bounds() {
        let x = Owned {
            bytes: vec![(-3i8) as u8, 0, 4],
            shape: vec![3],
            strides: vec![1],
            dtype: DataType::Int8,
        };
        let mut int_out = Owned::zeros(DataType::Int8, &[3]);
        ClipKernel {
            min: None,
            max: None,
        }
        .execute(&[x.view()], &mut [int_out.view_mut()])
        .unwrap();
        assert_eq!(int_out.bytes, x.bytes);

        let f16 = Owned::f16(&[3], &[-2., 0.5, 3.]);
        let lo = Owned::f16(&[], &[0.]);
        let hi = Owned::f16(&[], &[1.]);
        let mut f16_out = Owned::zeros(DataType::Float16, &[3]);
        ClipKernel {
            min: None,
            max: None,
        }
        .execute(
            &[f16.view(), lo.view(), hi.view()],
            &mut [f16_out.view_mut()],
        )
        .unwrap();
        assert_eq!(f16_out.to_f16_as_f32(), vec![0., 0.5, 1.]);
    }

    #[test]
    fn clip_bfloat16_matches_widened_reference() {
        let input = Owned::bf16(&[2, 4], &[-3.25, -0.5, 0.25, 1.5, 4.0, 0.75, -1.0, 2.0]);
        let minimum = Owned::bf16(&[], &[-0.75]);
        let maximum = Owned::bf16(&[], &[1.25]);
        let mut output = Owned::zeros(DataType::BFloat16, &[2, 4]);
        ClipKernel {
            min: None,
            max: None,
        }
        .execute(
            &[input.view(), minimum.view(), maximum.view()],
            &mut [output.view_mut()],
        )
        .unwrap();
        let minimum = minimum.to_bf16_as_f32()[0];
        let maximum = maximum.to_bf16_as_f32()[0];
        let expected: Vec<f32> = input
            .to_bf16_as_f32()
            .into_iter()
            .map(|value| value.clamp(minimum, maximum))
            .collect();
        assert_eq!(output.to_bf16_as_f32(), expected);
    }

    #[test]
    fn clip_clamps_negative_i32_values() {
        let x = Owned::i32(&[5], &[-5, -1, 0, 3, 9]);
        let lo = Owned::i32(&[], &[-2]);
        let hi = Owned::i32(&[], &[4]);
        let mut y = Owned::zeros(DataType::Int32, &[5]);
        ClipKernel {
            min: None,
            max: None,
        }
        .execute(&[x.view(), lo.view(), hi.view()], &mut [y.view_mut()])
        .unwrap();
        assert_eq!(y.to_i32(), vec![-2, -1, 0, 3, 4]);
    }

    #[test]
    fn clip_honors_absent_integer_bound_slots() {
        let x = Owned::i32(&[5], &[-5, -1, 0, 3, 9]);
        let lo = Owned::i32(&[], &[-2]);
        let hi = Owned::i32(&[], &[4]);

        let mut lower_only = Owned::zeros(DataType::Int32, &[5]);
        ClipKernel {
            min: None,
            max: None,
        }
        .execute(
            &[x.view(), lo.view(), TensorView::absent(DataType::Int32)],
            &mut [lower_only.view_mut()],
        )
        .unwrap();
        assert_eq!(lower_only.to_i32(), vec![-2, -1, 0, 3, 9]);

        let mut upper_only = Owned::zeros(DataType::Int32, &[5]);
        ClipKernel {
            min: None,
            max: None,
        }
        .execute(
            &[x.view(), TensorView::absent(DataType::Int32), hi.view()],
            &mut [upper_only.view_mut()],
        )
        .unwrap();
        assert_eq!(upper_only.to_i32(), vec![-5, -1, 0, 3, 4]);
    }

    #[test]
    fn clip_supports_i64_tensor_bounds() {
        let x = Owned::i64(&[5], &[-9, -2, 0, 4, 12]);
        let lo = Owned::i64(&[], &[-3]);
        let hi = Owned::i64(&[], &[5]);
        let mut y = Owned::zeros(DataType::Int64, &[5]);
        ClipKernel {
            min: None,
            max: None,
        }
        .execute(&[x.view(), lo.view(), hi.view()], &mut [y.view_mut()])
        .unwrap();
        assert_eq!(y.to_i64(), vec![-3, -2, 0, 4, 5]);
    }

    #[test]
    fn clip_supports_f64_tensor_bounds() {
        let f64_owned = |shape: &[usize], data: &[f64]| Owned {
            bytes: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            shape: shape.to_vec(),
            strides: onnx_runtime_ir::compute_contiguous_strides(shape),
            dtype: DataType::Float64,
        };
        let x = f64_owned(&[5], &[-3.5, -1.0, 0.5, 2.0, 8.5]);
        let lo = f64_owned(&[], &[-2.0]);
        let hi = f64_owned(&[], &[4.0]);
        let mut y = Owned::zeros(DataType::Float64, &[5]);
        ClipKernel {
            min: None,
            max: None,
        }
        .execute(&[x.view(), lo.view(), hi.view()], &mut [y.view_mut()])
        .unwrap();
        let values: Vec<f64> = y
            .bytes
            .chunks_exact(8)
            .map(|bytes| f64::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![-2.0, -1.0, 0.5, 2.0, 4.0]);
    }

    #[test]
    fn argmax_last_tie_negative_axis() {
        let x = Owned::f32(&[2, 3], &[1., 4., 4., 3., 2., 1.]);
        let mut y = Owned::zeros(DataType::Int64, &[2]);
        ArgKernel {
            op: ArgOp::Max,
            axis: -1,
            keepdims: false,
            select_last_index: true,
        }
        .execute(&[x.view()], &mut [y.view_mut()])
        .unwrap();
        assert_eq!(y.to_i64(), vec![2, 0]);
    }
    #[test]
    fn argmin_keepdims_selects_last_tie() {
        let x = Owned::f32(&[2, 3], &[3., 1., 1., 2., 0., 0.]);
        let mut y = Owned::zeros(DataType::Int64, &[2, 1]);
        ArgKernel {
            op: ArgOp::Min,
            axis: 1,
            keepdims: true,
            select_last_index: true,
        }
        .execute(&[x.view()], &mut [y.view_mut()])
        .unwrap();
        assert_eq!(y.to_i64(), vec![2, 2]);
    }

    #[test]
    fn argmax_and_argmin_accept_bfloat16() {
        let input = Owned::bf16(&[2, 4], &[1.25, 4.5, 4.0, -2.0, -3.0, -1.0, -2.0, -1.5]);
        let mut maximum = Owned::zeros(DataType::Int64, &[2]);
        ArgKernel {
            op: ArgOp::Max,
            axis: -1,
            keepdims: false,
            select_last_index: false,
        }
        .execute(&[input.view()], &mut [maximum.view_mut()])
        .unwrap();
        assert_eq!(maximum.to_i64(), vec![1, 1]);

        let mut minimum = Owned::zeros(DataType::Int64, &[2]);
        ArgKernel {
            op: ArgOp::Min,
            axis: -1,
            keepdims: false,
            select_last_index: false,
        }
        .execute(&[input.view()], &mut [minimum.view_mut()])
        .unwrap();
        assert_eq!(minimum.to_i64(), vec![3, 0]);
    }

    #[test]
    fn topk_bfloat16_values_match_widened_reference() {
        let input = Owned::bf16(
            &[2, 5],
            &[2.25, 5.5, 1.0, 4.25, -1.0, -3.0, 0.5, 7.0, 6.5, 2.0],
        );
        let count = Owned::i64(&[], &[3]);
        let mut values = Owned::zeros(DataType::BFloat16, &[2, 3]);
        let mut indices = Owned::zeros(DataType::Int64, &[2, 3]);
        TopKKernel {
            axis: -1,
            largest: true,
            sorted: true,
        }
        .execute(
            &[input.view(), count.view()],
            &mut [values.view_mut(), indices.view_mut()],
        )
        .unwrap();
        assert_eq!(
            values.to_bf16_as_f32(),
            vec![5.5, 4.25, 2.25, 7.0, 6.5, 2.0]
        );
        assert_eq!(indices.to_i64(), vec![1, 3, 0, 2, 3, 4]);
    }

    #[test]
    fn nonzero_accepts_bfloat16() {
        let input = Owned::bf16(&[2, 3], &[0.0, -0.0, 2.5, -1.0, 0.0, 3.0]);
        let mut output = Owned::zeros(DataType::Int64, &[2, 3]);
        NonZeroKernel
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.to_i64(), vec![0, 1, 1, 2, 0, 2]);
    }

    #[test]
    fn topk_and_nonzero() {
        let x = Owned::f32(&[4], &[2., 5., 1., 4.]);
        let k = Owned::i64(&[], &[2]);
        let mut v = Owned::zeros_f32(&[2]);
        let mut i = Owned::zeros(DataType::Int64, &[2]);
        TopKKernel {
            axis: -1,
            largest: true,
            sorted: true,
        }
        .execute(&[x.view(), k.view()], &mut [v.view_mut(), i.view_mut()])
        .unwrap();
        assert_eq!(v.to_f32(), vec![5., 4.]);
        assert_eq!(i.to_i64(), vec![1, 3]);
        let z = Owned::f32(&[2, 2], &[0., 3., 4., 0.]);
        let mut o = Owned::zeros(DataType::Int64, &[2, 2]);
        NonZeroKernel
            .execute(&[z.view()], &mut [o.view_mut()])
            .unwrap();
        assert_eq!(o.to_i64(), vec![0, 1, 1, 0]);
    }

    #[test]
    fn non_max_suppression_filters_overlaps_per_class() {
        let boxes = Owned::f32(
            &[1, 3, 4],
            &[
                0., 0., 1., 1., // box 0
                0., 0., 0.9, 0.9, // box 1 overlaps box 0
                2., 2., 3., 3., // box 2 is disjoint
            ],
        );
        let scores = Owned::f32(&[1, 2, 3], &[0.9, 0.8, 0.7, 0.1, 0.95, 0.2]);
        let max_output = Owned::i64(&[], &[2]);
        let iou = Owned::f32(&[], &[0.5]);
        let score = Owned::f32(&[], &[0.15]);
        let mut output = Owned::zeros(DataType::Int64, &[4, 3]);
        NonMaxSuppressionKernel {
            center_point_box: 0,
        }
        .execute(
            &[
                boxes.view(),
                scores.view(),
                max_output.view(),
                iou.view(),
                score.view(),
            ],
            &mut [output.view_mut()],
        )
        .unwrap();
        assert_eq!(output.to_i64(), vec![0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 1, 2]);
    }
}
