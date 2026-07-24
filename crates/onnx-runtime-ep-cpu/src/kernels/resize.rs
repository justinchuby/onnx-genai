//! ONNX `Resize` (opset 10+): N-D nearest, linear, and cubic interpolation.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, Node};
use rayon::prelude::*;

use super::{check_arity, to_dense_i64};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use crate::strided::next_index;

const OP: &str = "Resize";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Nearest,
    Linear,
    Cubic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CoordinateMode {
    HalfPixel,
    PytorchHalfPixel,
    AlignCorners,
    Asymmetric,
    TfCropAndResize,
    HalfPixelSymmetric,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NearestMode {
    RoundPreferFloor,
    RoundPreferCeil,
    Floor,
    Ceil,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AspectPolicy {
    Stretch,
    NotLarger,
    NotSmaller,
}

pub struct ResizeFactory {
    pub since_version: u32,
}

pub struct ResizeKernel {
    since_version: u32,
    mode: Mode,
    coordinate_mode: CoordinateMode,
    nearest_mode: NearestMode,
    cubic_coeff_a: f64,
    exclude_outside: bool,
    extrapolation_value: f32,
    antialias: bool,
    axes: Option<Vec<i64>>,
    aspect_policy: AspectPolicy,
}

fn unsupported_attribute(name: &str, value: &str, expected: &str) -> EpError {
    EpError::KernelFailed(format!(
        "{OP}: WHAT: unsupported {name} {value:?}. WHY: this value is not implemented by the \
         native CPU kernel. HOW: use one of {expected}, or route this node to another EP."
    ))
}

impl KernelFactory for ResizeFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let mode = match node
            .attr("mode")
            .and_then(Attribute::as_str)
            .unwrap_or("nearest")
        {
            "nearest" => Mode::Nearest,
            "linear" => Mode::Linear,
            "cubic" => Mode::Cubic,
            other => {
                return Err(unsupported_attribute(
                    "mode",
                    other,
                    "nearest, linear, or cubic",
                ));
            }
        };
        let coordinate_default = if self.since_version == 10 {
            "asymmetric"
        } else {
            "half_pixel"
        };
        let coordinate_mode = match node
            .attr("coordinate_transformation_mode")
            .and_then(Attribute::as_str)
            .unwrap_or(coordinate_default)
        {
            "half_pixel" => CoordinateMode::HalfPixel,
            "pytorch_half_pixel" => CoordinateMode::PytorchHalfPixel,
            "align_corners" => CoordinateMode::AlignCorners,
            "asymmetric" => CoordinateMode::Asymmetric,
            "tf_crop_and_resize" => CoordinateMode::TfCropAndResize,
            "half_pixel_symmetric" => CoordinateMode::HalfPixelSymmetric,
            other => {
                return Err(unsupported_attribute(
                    "coordinate_transformation_mode",
                    other,
                    "half_pixel, pytorch_half_pixel, align_corners, asymmetric, \
                     tf_crop_and_resize, or half_pixel_symmetric",
                ));
            }
        };
        let nearest_default = if self.since_version == 10 {
            "floor"
        } else {
            "round_prefer_floor"
        };
        let nearest_mode = match node
            .attr("nearest_mode")
            .and_then(Attribute::as_str)
            .unwrap_or(nearest_default)
        {
            "round_prefer_floor" => NearestMode::RoundPreferFloor,
            "round_prefer_ceil" => NearestMode::RoundPreferCeil,
            "floor" => NearestMode::Floor,
            "ceil" => NearestMode::Ceil,
            other => {
                return Err(unsupported_attribute(
                    "nearest_mode",
                    other,
                    "round_prefer_floor, round_prefer_ceil, floor, or ceil",
                ));
            }
        };
        let aspect_policy = match node
            .attr("keep_aspect_ratio_policy")
            .and_then(Attribute::as_str)
            .unwrap_or("stretch")
        {
            "stretch" => AspectPolicy::Stretch,
            "not_larger" => AspectPolicy::NotLarger,
            "not_smaller" => AspectPolicy::NotSmaller,
            other => {
                return Err(unsupported_attribute(
                    "keep_aspect_ratio_policy",
                    other,
                    "stretch, not_larger, or not_smaller",
                ));
            }
        };

        Ok(Box::new(ResizeKernel {
            since_version: self.since_version,
            mode,
            coordinate_mode,
            nearest_mode,
            cubic_coeff_a: node
                .attr("cubic_coeff_a")
                .and_then(Attribute::as_float)
                .unwrap_or(-0.75) as f64,
            exclude_outside: node
                .attr("exclude_outside")
                .and_then(Attribute::as_int)
                .unwrap_or(0)
                != 0,
            extrapolation_value: node
                .attr("extrapolation_value")
                .and_then(Attribute::as_float)
                .unwrap_or(0.0),
            antialias: node
                .attr("antialias")
                .and_then(Attribute::as_int)
                .unwrap_or(0)
                != 0,
            axes: node
                .attr("axes")
                .and_then(Attribute::as_ints)
                .map(<[i64]>::to_vec),
            aspect_policy,
        }))
    }
}

fn normalize_axes(raw: Option<&[i64]>, rank: usize) -> Result<Vec<usize>> {
    let Some(raw) = raw.filter(|axes| !axes.is_empty()) else {
        return Ok((0..rank).collect());
    };
    let mut axes = Vec::with_capacity(raw.len());
    for &axis in raw {
        let normalized = if axis < 0 { axis + rank as i64 } else { axis };
        if normalized < 0 || normalized as usize >= rank {
            return Err(EpError::KernelFailed(format!(
                "{OP}: WHAT: axis {axis} is out of range for rank {rank}. \
                 WHY: every resize axis must identify an input dimension. \
                 HOW: use axes in [-{rank}, {rank})."
            )));
        }
        let normalized = normalized as usize;
        if axes.contains(&normalized) {
            return Err(EpError::KernelFailed(format!(
                "{OP}: WHAT: axis {axis} appears more than once. \
                 WHY: Resize axes must be unique. HOW: remove the duplicate axis."
            )));
        }
        axes.push(normalized);
    }
    Ok(axes)
}

fn checked_numel(shape: &[usize], what: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |n, &dim| {
        n.checked_mul(dim).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "{OP}: WHAT: {what} shape {shape:?} overflows usize. \
                 WHY: its element count is not addressable. HOW: use smaller extents."
            ))
        })
    })
}

fn contiguous_strides(shape: &[usize]) -> Result<Vec<usize>> {
    let mut strides = vec![1usize; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1]
            .checked_mul(shape[axis + 1])
            .ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "{OP}: shape {shape:?} overflows while computing strides"
                ))
            })?;
    }
    Ok(strides)
}

fn round_aspect_extent(input: usize, numerator: usize, denominator: usize) -> Result<usize> {
    let product = (input as u128)
        .checked_mul(numerator as u128)
        .ok_or_else(|| EpError::KernelFailed(format!("{OP}: aspect-ratio extent overflow")))?;
    let denominator = denominator as u128;
    usize::try_from((product + denominator / 2) / denominator).map_err(|_| {
        EpError::KernelFailed(format!("{OP}: aspect-ratio output extent exceeds usize"))
    })
}

impl ResizeKernel {
    fn validate_output_shape(
        &self,
        input_shape: &[usize],
        output_shape: &[usize],
        axes: &[usize],
        scales: Option<&[f32]>,
        sizes: Option<&[i64]>,
    ) -> Result<Vec<f64>> {
        if output_shape.len() != input_shape.len() {
            return Err(EpError::KernelFailed(format!(
                "{OP}: WHAT: output rank {} does not match input rank {}. \
                 WHY: Resize preserves rank. HOW: allocate an output with rank {}.",
                output_shape.len(),
                input_shape.len(),
                input_shape.len()
            )));
        }
        let mut expected = input_shape.to_vec();
        let mut effective_scales = vec![1.0f64; input_shape.len()];
        if let Some(scales) = scales {
            if self.aspect_policy != AspectPolicy::Stretch {
                return Err(EpError::KernelFailed(format!(
                    "{OP}: WHAT: scales were supplied with keep_aspect_ratio_policy other \
                         than \"stretch\". WHY: ONNX permits aspect-ratio policies only with \
                         sizes. HOW: use sizes or set keep_aspect_ratio_policy=\"stretch\"."
                )));
            }
            for (&axis, &scale) in axes.iter().zip(scales) {
                if !scale.is_finite() || scale <= 0.0 {
                    return Err(EpError::KernelFailed(format!(
                        "{OP}: WHAT: scale {scale} for axis {axis} is not positive and finite. \
                         WHY: ONNX Resize scales must be > 0. HOW: provide a positive finite scale."
                    )));
                }
                let extent = (input_shape[axis] as f64) * f64::from(scale);
                if extent > usize::MAX as f64 {
                    return Err(EpError::KernelFailed(format!(
                        "{OP}: WHAT: scale {scale} makes axis {axis} exceed usize. \
                         WHY: the output extent is not addressable. HOW: use a smaller scale."
                    )));
                }
                expected[axis] = extent.floor() as usize;
                effective_scales[axis] = f64::from(scale);
            }
        } else if let Some(sizes) = sizes {
            let requested = sizes
                .iter()
                .enumerate()
                .map(|(index, &size)| {
                    usize::try_from(size)
                        .ok()
                        .filter(|&size| size > 0)
                        .ok_or_else(|| {
                            EpError::KernelFailed(format!(
                                "{OP}: WHAT: size {size} at position {index} is not positive. \
                             WHY: ONNX Resize sizes must be > 0. HOW: provide positive sizes."
                            ))
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            match self.aspect_policy {
                AspectPolicy::Stretch => {
                    for (&axis, &size) in axes.iter().zip(&requested) {
                        expected[axis] = size;
                    }
                }
                policy => {
                    if axes.is_empty() {
                        return Err(EpError::KernelFailed(format!(
                            "{OP}: aspect-ratio policy requires at least one resize axis"
                        )));
                    }
                    if axes.iter().any(|&axis| input_shape[axis] == 0) {
                        return Err(EpError::KernelFailed(format!(
                            "{OP}: WHAT: aspect-ratio resize has a zero input extent. \
                             WHY: the common scale would divide by zero. \
                             HOW: use non-empty resized dimensions."
                        )));
                    }
                    let (numerator, denominator) = axes
                        .iter()
                        .zip(&requested)
                        .map(|(&axis, &size)| (size, input_shape[axis]))
                        .reduce(|left, right| {
                            let order = (left.0 as u128 * right.1 as u128)
                                .cmp(&(right.0 as u128 * left.1 as u128));
                            if (policy == AspectPolicy::NotLarger && order.is_le())
                                || (policy == AspectPolicy::NotSmaller && order.is_ge())
                            {
                                left
                            } else {
                                right
                            }
                        })
                        .expect("axes is non-empty");
                    for &axis in axes {
                        expected[axis] =
                            round_aspect_extent(input_shape[axis], numerator, denominator)?;
                    }
                }
            }
            for &axis in axes {
                effective_scales[axis] = if input_shape[axis] == 0 {
                    1.0
                } else {
                    expected[axis] as f64 / input_shape[axis] as f64
                };
            }
        }
        if expected != output_shape {
            return Err(EpError::KernelFailed(format!(
                "{OP}: WHAT: output shape {output_shape:?} does not match the shape {expected:?} \
                 requested by scales/sizes. WHY: the executor buffer and Resize geometry disagree. \
                 HOW: run ONNX shape inference or allocate the requested output shape."
            )));
        }
        Ok(effective_scales)
    }

    fn coordinate(
        &self,
        output_index: usize,
        input_size: usize,
        output_size: usize,
        scale: f64,
        roi_start: f64,
        roi_end: f64,
    ) -> f64 {
        let x = output_index as f64;
        match self.coordinate_mode {
            CoordinateMode::HalfPixel => (x + 0.5) / scale - 0.5,
            CoordinateMode::PytorchHalfPixel => {
                if output_size > 1 {
                    (x + 0.5) / scale - 0.5
                } else {
                    0.0
                }
            }
            CoordinateMode::AlignCorners => {
                if output_size <= 1 {
                    0.0
                } else {
                    x * (input_size.saturating_sub(1)) as f64
                        / (output_size.saturating_sub(1)) as f64
                }
            }
            CoordinateMode::Asymmetric => x / scale,
            CoordinateMode::TfCropAndResize => {
                let input_span = input_size.saturating_sub(1) as f64;
                if output_size > 1 {
                    roi_start * input_span
                        + x * (roi_end - roi_start) * input_span / (output_size - 1) as f64
                } else {
                    0.5 * (roi_start + roi_end) * input_span
                }
            }
            CoordinateMode::HalfPixelSymmetric => {
                let exact_output = input_size as f64 * scale;
                let adjustment = if exact_output == 0.0 {
                    0.0
                } else {
                    output_size as f64 / exact_output
                };
                let offset = input_size as f64 * 0.5 * (1.0 - adjustment);
                offset + (x + 0.5) / scale - 0.5
            }
        }
    }

    fn nearest_index(&self, coordinate: f64, size: usize) -> usize {
        let floor = coordinate.floor();
        let fraction = coordinate - floor;
        let index = match self.nearest_mode {
            NearestMode::RoundPreferFloor => {
                if fraction <= 0.5 {
                    floor
                } else {
                    floor + 1.0
                }
            }
            NearestMode::RoundPreferCeil => {
                if fraction < 0.5 {
                    floor
                } else {
                    floor + 1.0
                }
            }
            NearestMode::Floor => floor,
            NearestMode::Ceil => coordinate.ceil(),
        };
        index.clamp(0.0, size.saturating_sub(1) as f64) as usize
    }

    fn cubic_weight(&self, distance: f64) -> f64 {
        let x = distance.abs();
        if x <= 1.0 {
            (self.cubic_coeff_a + 2.0) * x.powi(3) - (self.cubic_coeff_a + 3.0) * x.powi(2) + 1.0
        } else if x < 2.0 {
            self.cubic_coeff_a * x.powi(3) - 5.0 * self.cubic_coeff_a * x.powi(2)
                + 8.0 * self.cubic_coeff_a * x
                - 4.0 * self.cubic_coeff_a
        } else {
            0.0
        }
    }

    fn axis_samples(&self, coordinate: f64, size: usize) -> Vec<(usize, f64)> {
        match self.mode {
            Mode::Nearest => vec![(self.nearest_index(coordinate, size), 1.0)],
            Mode::Linear => {
                let low = coordinate.floor() as i64;
                let fraction = coordinate - low as f64;
                if fraction == 0.0 {
                    vec![(low.clamp(0, size as i64 - 1) as usize, 1.0)]
                } else {
                    vec![
                        (low.clamp(0, size as i64 - 1) as usize, 1.0 - fraction),
                        ((low + 1).clamp(0, size as i64 - 1) as usize, fraction),
                    ]
                }
            }
            Mode::Cubic => {
                let base = coordinate.floor() as i64;
                let mut samples = Vec::with_capacity(4);
                let mut sum = 0.0;
                for offset in -1..=2 {
                    let index = base + offset;
                    let mut weight = self.cubic_weight(coordinate - index as f64);
                    if self.exclude_outside && (index < 0 || index >= size as i64) {
                        weight = 0.0;
                    }
                    sum += weight;
                    samples.push((index.clamp(0, size as i64 - 1) as usize, weight));
                }
                if self.exclude_outside && sum != 0.0 {
                    for (_, weight) in &mut samples {
                        *weight /= sum;
                    }
                }
                samples
            }
        }
    }

    fn weighted_sample(
        input: &[f32],
        strides: &[usize],
        samples: &[&[(usize, f64)]],
        axis: usize,
        offset: usize,
        weight: f64,
    ) -> f64 {
        if axis == samples.len() {
            return f64::from(input[offset]) * weight;
        }
        samples[axis]
            .iter()
            .map(|&(index, axis_weight)| {
                Self::weighted_sample(
                    input,
                    strides,
                    samples,
                    axis + 1,
                    offset + index * strides[axis],
                    weight * axis_weight,
                )
            })
            .sum()
    }
}

impl Kernel for ResizeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if self.since_version == 10 {
            check_arity(OP, inputs, outputs, 2, 2, 1)?;
        } else {
            check_arity(OP, inputs, outputs, 1, 4, 1)?;
        }
        if self.antialias {
            return Err(EpError::KernelFailed(format!(
                "{OP}: WHAT: antialias=1 is not implemented by the native CPU kernel. \
                     WHY: ONNX antialiasing uses a widened reconstruction filter, not ordinary \
                     point interpolation. HOW: set antialias=0 or route this node to another EP."
            )));
        }
        if inputs[0].dtype != outputs[0].dtype {
            return Err(EpError::KernelFailed(format!(
                "{OP}: WHAT: input dtype {:?} differs from output dtype {:?}. \
                 WHY: Resize preserves the element type. HOW: allocate Y with X's dtype.",
                inputs[0].dtype, outputs[0].dtype
            )));
        }
        if inputs[0].shape.iter().any(|&extent| extent == 0) {
            return Err(EpError::KernelFailed(format!(
                "{OP}: WHAT: input shape {:?} contains a zero extent. \
                 WHY: interpolation cannot sample an empty axis. \
                 HOW: avoid resizing empty tensors or route this case to another EP.",
                inputs[0].shape
            )));
        }

        let axes = normalize_axes(self.axes.as_deref(), inputs[0].shape.len())?;
        let (roi_input, scales_input, sizes_input) = if self.since_version == 10 {
            (None, inputs.get(1).filter(|input| !input.is_absent()), None)
        } else {
            (
                inputs.get(1).filter(|input| !input.is_absent()),
                inputs
                    .get(2)
                    .filter(|input| !input.is_absent() && input.numel() != 0),
                inputs
                    .get(3)
                    .filter(|input| !input.is_absent() && input.numel() != 0),
            )
        };
        if scales_input.is_some() == sizes_input.is_some() {
            return Err(EpError::KernelFailed(format!(
                "{OP}: WHAT: expected exactly one non-empty scales or sizes input. \
                 WHY: Resize geometry would otherwise be missing or ambiguous. \
                 HOW: provide scales in slot 2 or sizes in slot 3, but not both."
            )));
        }

        let scales_data = scales_input
            .map(|input| to_dense_f32_widen(OP, input).map(|values| values.into_owned()))
            .transpose()?;
        let sizes_data = sizes_input.map(to_dense_i64).transpose()?;
        let control_len = scales_data
            .as_ref()
            .map_or_else(|| sizes_data.as_ref().map_or(0, Vec::len), Vec::len);
        if control_len != axes.len() {
            return Err(EpError::KernelFailed(format!(
                "{OP}: WHAT: scales/sizes has {control_len} values for {} selected axes {axes:?}. \
                 WHY: ONNX requires one value per resize axis. \
                 HOW: provide exactly {} values.",
                axes.len(),
                axes.len()
            )));
        }

        let effective_scales = self.validate_output_shape(
            inputs[0].shape,
            outputs[0].shape,
            &axes,
            scales_data.as_deref(),
            sizes_data.as_deref(),
        )?;
        let roi = if self.coordinate_mode == CoordinateMode::TfCropAndResize {
            let roi_input = roi_input.ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "{OP}: WHAT: tf_crop_and_resize has no ROI input. \
                     WHY: this coordinate mode needs start/end coordinates for each resize axis. \
                     HOW: provide a ROI vector with {} values.",
                    2 * axes.len()
                ))
            })?;
            let roi = to_dense_f32_widen(OP, roi_input)?.into_owned();
            if roi.len() != 2 * axes.len() {
                return Err(EpError::KernelFailed(format!(
                    "{OP}: WHAT: ROI has {} values, expected {} for axes {axes:?}. \
                     WHY: ROI is [starts..., ends...]. HOW: provide two values per resize axis.",
                    roi.len(),
                    2 * axes.len()
                )));
            }
            roi.into_iter().map(f64::from).collect::<Vec<_>>()
        } else {
            vec![0.0; 2 * axes.len()]
        };

        let input = to_dense_f32_widen(OP, &inputs[0])?;
        let input_strides = contiguous_strides(inputs[0].shape)?;
        let output_numel = checked_numel(outputs[0].shape, "output")?;
        let mut output = Vec::with_capacity(output_numel);
        let mut output_index = vec![0usize; outputs[0].shape.len()];
        let mut axis_positions = vec![None; output_index.len()];
        for (position, &axis) in axes.iter().enumerate() {
            axis_positions[axis] = Some(position);
        }
        let mut sample_maps = Vec::with_capacity(output_index.len());
        let mut outside_maps = Vec::with_capacity(output_index.len());
        for axis in 0..output_index.len() {
            let mut axis_samples = Vec::with_capacity(outputs[0].shape[axis]);
            let mut axis_outside = Vec::with_capacity(outputs[0].shape[axis]);
            for output_axis_index in 0..outputs[0].shape[axis] {
                if let Some(position) = axis_positions[axis] {
                    let coordinate = self.coordinate(
                        output_axis_index,
                        inputs[0].shape[axis],
                        outputs[0].shape[axis],
                        effective_scales[axis],
                        roi[position],
                        roi[axes.len() + position],
                    );
                    axis_outside.push(
                        self.coordinate_mode == CoordinateMode::TfCropAndResize
                            && (coordinate < 0.0
                                || coordinate > inputs[0].shape[axis].saturating_sub(1) as f64),
                    );
                    axis_samples.push(self.axis_samples(coordinate, inputs[0].shape[axis]));
                } else {
                    axis_outside.push(false);
                    axis_samples.push(vec![(output_axis_index, 1.0)]);
                }
            }
            sample_maps.push(axis_samples);
            outside_maps.push(axis_outside);
        }

        if self.mode == Mode::Nearest {
            let output_shape = outputs[0].shape.to_vec();
            let sample = |linear_index: usize| {
                let mut remaining = linear_index;
                let mut offset = 0usize;
                for axis in (0..output_shape.len()).rev() {
                    let output_axis_index = remaining % output_shape[axis];
                    remaining /= output_shape[axis];
                    if outside_maps[axis][output_axis_index] {
                        return self.extrapolation_value;
                    }
                    offset += sample_maps[axis][output_axis_index][0].0 * input_strides[axis];
                }
                input[offset]
            };
            let parallel_threshold = rayon::current_num_threads().saturating_mul(32 * 1024);
            let output: Vec<f32> = if output_numel >= parallel_threshold {
                (0..output_numel).into_par_iter().map(sample).collect()
            } else {
                (0..output_numel).map(sample).collect()
            };
            return write_dense_f32_narrow(OP, &mut outputs[0], &output);
        }

        let mut selected_samples: Vec<&[(usize, f64)]> = vec![&[]; output_index.len()];
        for linear_index in 0..output_numel {
            let outside_roi =
                (0..output_index.len()).any(|axis| outside_maps[axis][output_index[axis]]);
            output.push(if outside_roi {
                self.extrapolation_value
            } else {
                for axis in 0..output_index.len() {
                    selected_samples[axis] = &sample_maps[axis][output_index[axis]];
                }
                Self::weighted_sample(&input, &input_strides, &selected_samples, 0, 0, 1.0) as f32
            });
            if linear_index + 1 < output_numel {
                next_index(outputs[0].shape, &mut output_index);
            }
        }
        write_dense_f32_narrow(OP, &mut outputs[0], &output)
    }

    fn supports_strided_input(&self, input_idx: usize) -> bool {
        input_idx < 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{build_cpu_registry, testutil::Owned};
    use onnx_runtime_ep_api::TensorView;
    use onnx_runtime_ir::{DataType, Node, NodeId};

    fn node(attrs: &[(&str, Attribute)]) -> Node {
        let mut node = Node::new(NodeId(0), OP, vec![], vec![]);
        for (name, value) in attrs {
            node.attributes.insert((*name).into(), value.clone());
        }
        node
    }

    fn kernel(opset: u32, attrs: &[(&str, Attribute)]) -> Box<dyn Kernel> {
        ResizeFactory {
            since_version: opset,
        }
        .create(&node(attrs), &[])
        .unwrap()
    }

    fn absent(dtype: DataType) -> TensorView<'static> {
        TensorView::absent(dtype)
    }

    fn run_scales(
        opset: u32,
        attrs: &[(&str, Attribute)],
        input: &Owned,
        scales: &[f32],
        output_shape: &[usize],
    ) -> Vec<f32> {
        let scales = Owned::f32(&[scales.len()], scales);
        let mut output = Owned::zeros(input.dtype, output_shape);
        let inputs = if opset == 10 {
            vec![input.view(), scales.view()]
        } else {
            vec![
                input.view(),
                absent(DataType::Float32),
                scales.view(),
                absent(DataType::Int64),
            ]
        };
        kernel(opset, attrs)
            .execute(&inputs, &mut [output.view_mut()])
            .unwrap();
        match input.dtype {
            DataType::Float16 => output.to_f16_as_f32(),
            DataType::BFloat16 => output.to_bf16_as_f32(),
            _ => output.to_f32(),
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "element {index}: actual {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn resize_nearest_asymmetric_opset10_upscale() {
        let input = Owned::f32(&[1, 1, 2, 2], &[1., 2., 3., 4.]);
        let actual = run_scales(10, &[], &input, &[1., 1., 2., 2.], &[1, 1, 4, 4]);
        assert_eq!(
            actual,
            vec![
                1., 1., 2., 2., 1., 1., 2., 2., 3., 3., 4., 4., 3., 3., 4., 4.
            ]
        );
    }

    #[test]
    fn resize_linear_half_pixel_bilinear_upscale() {
        let input = Owned::f32(&[1, 1, 2, 2], &[1., 2., 3., 4.]);
        let actual = run_scales(
            11,
            &[("mode", Attribute::String("linear".into()))],
            &input,
            &[1., 1., 2., 2.],
            &[1, 1, 4, 4],
        );
        assert_close(
            &actual,
            &[
                1., 1.25, 1.75, 2., 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3., 3.25, 3.75, 4.,
            ],
        );
    }

    #[test]
    fn resize_linear_align_corners_bilinear() {
        let input = Owned::f32(&[1, 1, 2, 2], &[1., 2., 3., 4.]);
        let actual = run_scales(
            11,
            &[
                ("mode", Attribute::String("linear".into())),
                (
                    "coordinate_transformation_mode",
                    Attribute::String("align_corners".into()),
                ),
            ],
            &input,
            &[1., 1., 1.5, 1.5],
            &[1, 1, 3, 3],
        );
        assert_close(&actual, &[1., 1.5, 2., 2., 2.5, 3., 3., 3.5, 4.]);
    }

    #[test]
    fn resize_linear_asymmetric_and_sizes_downscale() {
        let input = Owned::f32(&[1, 1, 1, 4], &[0., 10., 20., 30.]);
        let actual = run_scales(
            11,
            &[
                ("mode", Attribute::String("linear".into())),
                (
                    "coordinate_transformation_mode",
                    Attribute::String("asymmetric".into()),
                ),
            ],
            &input,
            &[1., 1., 1., 0.5],
            &[1, 1, 1, 2],
        );
        assert_close(&actual, &[0., 20.]);

        let sizes = Owned::i64(&[4], &[1, 1, 1, 2]);
        let mut output = Owned::zeros_f32(&[1, 1, 1, 2]);
        kernel(
            11,
            &[
                ("mode", Attribute::String("linear".into())),
                (
                    "coordinate_transformation_mode",
                    Attribute::String("asymmetric".into()),
                ),
            ],
        )
        .execute(
            &[
                input.view(),
                absent(DataType::Float32),
                absent(DataType::Float32),
                sizes.view(),
            ],
            &mut [output.view_mut()],
        )
        .unwrap();
        assert_close(&output.to_f32(), &[0., 20.]);
    }

    #[test]
    fn resize_nearest_modes_choose_ties_as_specified() {
        let input = Owned::f32(&[1, 4], &[0., 10., 20., 30.]);
        let attrs = [
            (
                "coordinate_transformation_mode",
                Attribute::String("half_pixel".into()),
            ),
            (
                "nearest_mode",
                Attribute::String("round_prefer_floor".into()),
            ),
        ];
        assert_eq!(
            run_scales(11, &attrs, &input, &[1., 0.5], &[1, 2]),
            vec![0., 20.]
        );
        let attrs = [
            (
                "coordinate_transformation_mode",
                Attribute::String("half_pixel".into()),
            ),
            (
                "nearest_mode",
                Attribute::String("round_prefer_ceil".into()),
            ),
        ];
        assert_eq!(
            run_scales(11, &attrs, &input, &[1., 0.5], &[1, 2]),
            vec![10., 30.]
        );
    }

    #[test]
    fn resize_axes_and_half_pixel_symmetric() {
        let input = Owned::f32(&[1, 1, 4], &[0., 10., 20., 30.]);
        let actual = run_scales(
            18,
            &[
                ("mode", Attribute::String("linear".into())),
                (
                    "coordinate_transformation_mode",
                    Attribute::String("half_pixel_symmetric".into()),
                ),
                ("axes", Attribute::Ints(vec![-1])),
            ],
            &input,
            &[0.6],
            &[1, 1, 2],
        );
        assert_close(&actual, &[6.666_666_5, 23.333_334]);
    }

    #[test]
    fn resize_tf_crop_and_resize_uses_roi_and_extrapolation() {
        let input = Owned::f32(&[1, 1, 1, 3], &[10., 20., 30.]);
        let roi = Owned::f32(&[8], &[0., 0., 0., -0.5, 1., 1., 1., 1.5]);
        let sizes = Owned::i64(&[4], &[1, 1, 1, 3]);
        let mut output = Owned::zeros_f32(&[1, 1, 1, 3]);
        kernel(
            11,
            &[
                ("mode", Attribute::String("linear".into())),
                (
                    "coordinate_transformation_mode",
                    Attribute::String("tf_crop_and_resize".into()),
                ),
                ("extrapolation_value", Attribute::Float(-1.)),
            ],
        )
        .execute(
            &[
                input.view(),
                roi.view(),
                absent(DataType::Float32),
                sizes.view(),
            ],
            &mut [output.view_mut()],
        )
        .unwrap();
        assert_eq!(output.to_f32(), vec![-1., 20., -1.]);
    }

    #[test]
    fn resize_cubic_exclude_outside_and_float16_bfloat16() {
        let input = Owned::f32(&[1, 1, 1, 4], &[0., 10., 20., 30.]);
        let actual = run_scales(
            11,
            &[
                ("mode", Attribute::String("cubic".into())),
                ("exclude_outside", Attribute::Int(1)),
            ],
            &input,
            &[1., 1., 1., 2.],
            &[1, 1, 1, 8],
        );
        assert_eq!(actual.len(), 8);
        assert!(actual.iter().all(|value| value.is_finite()));

        for input in [
            Owned::f16(&[1, 1, 2], &[0., 10.]),
            Owned::bf16(&[1, 1, 2], &[0., 10.]),
        ] {
            let actual = run_scales(
                11,
                &[("mode", Attribute::String("linear".into()))],
                &input,
                &[1., 1., 2.],
                &[1, 1, 4],
            );
            assert_close(&actual, &[0., 2.5, 7.5, 10.]);
        }
    }

    #[test]
    fn resize_rejects_antialias_with_actionable_error() {
        let input = Owned::f32(&[1], &[1.]);
        let scales = Owned::f32(&[1], &[1.]);
        let mut output = Owned::zeros_f32(&[1]);
        let error = kernel(18, &[("antialias", Attribute::Int(1))])
            .execute(
                &[
                    input.view(),
                    absent(DataType::Float32),
                    scales.view(),
                    absent(DataType::Int64),
                ],
                &mut [output.view_mut()],
            )
            .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("antialias=1"));
        assert!(message.contains("route this node to another EP"));
    }

    #[test]
    fn resize_registry_covers_opset10_through_latest() {
        let registry = build_cpu_registry();
        assert!(registry.lookup(OP, "", 9).is_none());
        assert!(registry.lookup(OP, "", 10).is_some());
        assert!(registry.lookup(OP, "", 11).is_some());
        assert!(registry.lookup(OP, "", 25).is_some());
    }
}
