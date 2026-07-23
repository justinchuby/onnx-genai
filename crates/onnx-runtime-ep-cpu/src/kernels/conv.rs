//! MLAS-backed 2-D NCHW `Conv` for Float32 tensors.
//!
//! Two execution strategies are selected at kernel-construction time from the
//! static shape/group attributes:
//!
//! * **NCHWc blocked path** (fast; mirrors ONNX Runtime's `nchwc_transformer`).
//!   The filter is reordered ("pre-packed") once into MLAS's channels-blocked
//!   layout and reused across inferences; each call reorders the activation to
//!   NCHWc, runs [`mlas_sys::nchwc_conv`], and reorders the output back to NCHW.
//!   Used for the shapes that dominate CNNs: pointwise/3x3 group-1 convs, the
//!   3-channel first layer, and depthwise convs.
//! * **im2col GEMM fallback** ([`mlas_sys::ConvPlan`]) for every other shape
//!   (e.g. general grouped convs), preserving full generality and parity.

use std::borrow::Cow;
use std::sync::{Mutex, OnceLock};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::check_arity;
use crate::dtype::to_dense_f32_widen;
use crate::strided::numel;

#[derive(Clone, Copy)]
enum AutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

pub struct ConvFactory;

/// Blocked-convolution strategy and the pre-packed weights it reuses.
struct NchwcConv {
    /// MLAS channel-block width (8 for AVX2, 16 for AVX-512).
    block: usize,
    /// `GroupCount` passed to MLAS: 1 for group-1 convs, the blocked channel
    /// count for depthwise (matching `nchwc_transformer`).
    group_count: usize,
    /// Whether the activation must be reordered NCHW -> NCHWc before the conv.
    /// False only for the 3-channel first-layer algorithm.
    reorder_input: bool,
    /// True selects the `OIHWBiBo` filter layout, false selects `OIHWBo`.
    filter_bibo: bool,
    batch: usize,
    input_channels: usize,
    input_h: usize,
    input_w: usize,
    output_h: usize,
    output_w: usize,
    kernel: [i64; 2],
    dilation: [i64; 2],
    pads: [i64; 4],
    stride: [i64; 2],
    /// Original filter shape `[O, I/group, H, W]`.
    weight_shape: [i64; 4],
    /// Input channel count fed to MLAS (blocked+aligned when `reorder_input`).
    conv_input_channels: usize,
    /// Output channels rounded up to `block`.
    nchwc_out_channels: usize,
    activation: mlas_sys::NchwcActivation,
    weight_constant: bool,
    bias_constant: bool,
    packed_filter: OnceLock<Vec<f32>>,
    packed_bias: OnceLock<Vec<f32>>,
    input_scratch: Mutex<Vec<f32>>,
    output_scratch: Mutex<Vec<f32>>,
}

/// im2col + GEMM fallback for shapes the NCHWc path does not cover.
struct FallbackConv {
    plan: mlas_sys::ConvPlan,
    scratch: Mutex<Vec<f32>>,
}

enum ConvImpl {
    Nchwc(Box<NchwcConv>),
    Fallback(FallbackConv),
}

pub struct ConvKernel {
    imp: ConvImpl,
    expected_input_shape: Vec<usize>,
    expected_weight_shape: Vec<usize>,
    expected_output_shape: Vec<usize>,
    output_channels: usize,
    /// Fused activation applied in the convolution epilogue. Set by the
    /// `CpuConvBatchNormActivationFusion` graph pass via the `activation`
    /// attribute; `IDENTITY` when the Conv stands alone.
    activation: mlas_sys::NchwcActivation,
}

/// Parse the optional fused-`activation` attribute the CPU EP's Conv+BN+Relu
/// fusion writes onto a `Conv` node. Only `Relu` is currently emitted; anything
/// else (including a missing attribute) means no fused activation.
fn parse_activation(node: &Node) -> mlas_sys::NchwcActivation {
    match node.attr("activation").and_then(Attribute::as_str) {
        Some("Relu") => mlas_sys::NchwcActivation::RELU,
        _ => mlas_sys::NchwcActivation::IDENTITY,
    }
}

#[inline]
fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

fn auto_pad(node: &Node) -> Result<AutoPad> {
    match node.attr("auto_pad").and_then(Attribute::as_str) {
        None | Some("NOTSET") => Ok(AutoPad::NotSet),
        Some("SAME_UPPER") => Ok(AutoPad::SameUpper),
        Some("SAME_LOWER") => Ok(AutoPad::SameLower),
        Some("VALID") => Ok(AutoPad::Valid),
        Some(value) => Err(EpError::KernelFailed(format!(
            "Conv: unsupported auto_pad {value:?}; expected NOTSET, SAME_UPPER, SAME_LOWER, or VALID"
        ))),
    }
}

fn positive_values(node: &Node, name: &str, default: usize) -> Result<[usize; 2]> {
    let values = node
        .attr(name)
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![default as i64; 2]);
    if values.len() != 2 || values.iter().any(|&value| value <= 0) {
        return Err(EpError::KernelFailed(format!(
            "Conv: {name} must contain two positive values, got {values:?}"
        )));
    }
    Ok([values[0] as usize, values[1] as usize])
}

fn explicit_pads(node: &Node) -> Result<[usize; 4]> {
    let values = node
        .attr("pads")
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![0; 4]);
    if values.len() != 4 || values.iter().any(|&value| value < 0) {
        return Err(EpError::KernelFailed(format!(
            "Conv: pads must contain four non-negative values, got {values:?}"
        )));
    }
    Ok([
        values[0] as usize,
        values[1] as usize,
        values[2] as usize,
        values[3] as usize,
    ])
}

fn output_geometry(
    input: [usize; 2],
    kernel: [usize; 2],
    dilations: [usize; 2],
    strides: [usize; 2],
    mut pads: [usize; 4],
    auto_pad: AutoPad,
) -> Result<([usize; 2], [usize; 4])> {
    let mut output = [0; 2];
    for axis in 0..2 {
        let effective = dilations[axis]
            .checked_mul(kernel[axis] - 1)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| EpError::KernelFailed("Conv: effective kernel size overflow".into()))?;
        match auto_pad {
            AutoPad::SameUpper | AutoPad::SameLower => {
                output[axis] = input[axis].div_ceil(strides[axis]);
                let total = output[axis]
                    .saturating_sub(1)
                    .checked_mul(strides[axis])
                    .and_then(|value| value.checked_add(effective))
                    .map(|value| value.saturating_sub(input[axis]))
                    .ok_or_else(|| EpError::KernelFailed("Conv: padding size overflow".into()))?;
                let begin = if matches!(auto_pad, AutoPad::SameUpper) {
                    total / 2
                } else {
                    total - total / 2
                };
                pads[axis] = begin;
                pads[axis + 2] = total - begin;
            }
            AutoPad::Valid => {
                pads[axis] = 0;
                pads[axis + 2] = 0;
                output[axis] = if input[axis] < effective {
                    0
                } else {
                    (input[axis] - effective) / strides[axis] + 1
                };
            }
            AutoPad::NotSet => {
                let padded = input[axis]
                    .checked_add(pads[axis])
                    .and_then(|value| value.checked_add(pads[axis + 2]))
                    .ok_or_else(|| {
                        EpError::KernelFailed("Conv: padded input size overflow".into())
                    })?;
                output[axis] = if padded < effective {
                    0
                } else {
                    (padded - effective) / strides[axis] + 1
                };
            }
        }
    }
    Ok((output, pads))
}

fn to_i64(values: impl IntoIterator<Item = usize>, what: &str) -> Result<Vec<i64>> {
    values
        .into_iter()
        .map(|value| {
            i64::try_from(value)
                .map_err(|_| EpError::KernelFailed(format!("Conv: {what} exceeds i64")))
        })
        .collect()
}

impl KernelFactory for ConvFactory {
    fn create(&self, node: &Node, shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let x_shape = shapes
            .first()
            .ok_or_else(|| EpError::KernelFailed("Conv: missing X shape".into()))?;
        let w_shape = shapes
            .get(1)
            .ok_or_else(|| EpError::KernelFailed("Conv: missing W shape".into()))?;
        if x_shape.len() != 4 || w_shape.len() != 4 {
            return Err(EpError::KernelFailed(format!(
                "Conv: MLAS kernel currently supports 2-D NCHW tensors; got X={x_shape:?}, W={w_shape:?}"
            )));
        }

        let group = node.attr("group").and_then(Attribute::as_int).unwrap_or(1);
        if group <= 0 {
            return Err(EpError::KernelFailed(format!(
                "Conv: group must be positive, got {group}"
            )));
        }
        let group = group as usize;
        let input_channels = x_shape[1];
        let output_channels = w_shape[0];
        if !input_channels.is_multiple_of(group)
            || !output_channels.is_multiple_of(group)
            || w_shape[1] != input_channels / group
        {
            return Err(EpError::KernelFailed(format!(
                "Conv: incompatible channels/group: X channels={input_channels}, W={w_shape:?}, group={group}"
            )));
        }

        let inferred_kernel = [w_shape[2], w_shape[3]];
        let kernel = match node.attr("kernel_shape").and_then(Attribute::as_ints) {
            None => inferred_kernel,
            Some(values)
                if values.len() == 2
                    && values.iter().all(|&value| value > 0)
                    && values[0] as usize == inferred_kernel[0]
                    && values[1] as usize == inferred_kernel[1] =>
            {
                inferred_kernel
            }
            Some(values) => {
                return Err(EpError::KernelFailed(format!(
                    "Conv: kernel_shape must match W spatial shape {inferred_kernel:?}, got {values:?}"
                )));
            }
        };
        let dilations = positive_values(node, "dilations", 1)?;
        let strides = positive_values(node, "strides", 1)?;
        let (output_spatial, pads) = output_geometry(
            [x_shape[2], x_shape[3]],
            kernel,
            dilations,
            strides,
            explicit_pads(node)?,
            auto_pad(node)?,
        )?;
        let expected_output_shape = vec![
            x_shape[0],
            output_channels,
            output_spatial[0],
            output_spatial[1],
        ];

        let activation = parse_activation(node);
        let imp = Self::select_impl(
            x_shape,
            w_shape,
            group,
            input_channels,
            output_channels,
            kernel,
            dilations,
            strides,
            pads,
            output_spatial,
            activation,
        )?;

        Ok(Box::new(ConvKernel {
            imp,
            expected_input_shape: x_shape.clone(),
            expected_weight_shape: w_shape.clone(),
            expected_output_shape,
            output_channels,
            activation,
        }))
    }
}

impl ConvFactory {
    /// Choose the NCHWc blocked path when the shape matches one of the
    /// algorithms MLAS supports with a pre-reordered filter, otherwise fall back
    /// to the im2col GEMM path. Mirrors ONNX Runtime's `nchwc_transformer`
    /// eligibility rules.
    #[allow(clippy::too_many_arguments)]
    fn select_impl(
        x_shape: &[usize],
        w_shape: &[usize],
        group: usize,
        input_channels: usize,
        output_channels: usize,
        kernel: [usize; 2],
        dilations: [usize; 2],
        strides: [usize; 2],
        pads: [usize; 4],
        output_spatial: [usize; 2],
        activation: mlas_sys::NchwcActivation,
    ) -> Result<ConvImpl> {
        let block = mlas_sys::nchwc_block_size();
        // `MlasReorderInputNchw` reads channels four at a time, so any reordered
        // activation must have a channel count that is a multiple of four.
        const CHANNEL_ALIGNMENT: usize = 4;

        // Decide filter layout and whether the input needs NCHWc reordering.
        let nchwc = (block >= 8).then(|| {
            if group == 1 {
                if input_channels < block {
                    // First-layer algorithm: keep the input in dense NCHW.
                    Some((false, false))
                } else if input_channels.is_multiple_of(CHANNEL_ALIGNMENT) {
                    // Blocked NCHWc / pointwise algorithm.
                    Some((true, true))
                } else {
                    None
                }
            } else if w_shape[1] == 1
                && output_channels == group
                && output_channels.is_multiple_of(CHANNEL_ALIGNMENT)
            {
                // Depthwise separable convolution.
                Some((true, false))
            } else {
                // General grouped convolution: use the fallback path.
                None
            }
        });

        if let Some(Some((reorder_input, filter_bibo))) = nchwc {
            let nchwc_out_channels = round_up(output_channels, block);
            let group_count = if group == 1 { 1 } else { nchwc_out_channels };
            let conv_input_channels = if reorder_input {
                round_up(input_channels, block)
            } else {
                input_channels
            };
            return Ok(ConvImpl::Nchwc(Box::new(NchwcConv {
                block,
                group_count,
                reorder_input,
                filter_bibo,
                batch: x_shape[0],
                input_channels,
                input_h: x_shape[2],
                input_w: x_shape[3],
                output_h: output_spatial[0],
                output_w: output_spatial[1],
                kernel: [kernel[0] as i64, kernel[1] as i64],
                dilation: [dilations[0] as i64, dilations[1] as i64],
                pads: [
                    pads[0] as i64,
                    pads[1] as i64,
                    pads[2] as i64,
                    pads[3] as i64,
                ],
                stride: [strides[0] as i64, strides[1] as i64],
                weight_shape: [
                    w_shape[0] as i64,
                    w_shape[1] as i64,
                    w_shape[2] as i64,
                    w_shape[3] as i64,
                ],
                conv_input_channels,
                nchwc_out_channels,
                activation,
                weight_constant: false,
                bias_constant: false,
                packed_filter: OnceLock::new(),
                packed_bias: OnceLock::new(),
                input_scratch: Mutex::new(Vec::new()),
                output_scratch: Mutex::new(Vec::new()),
            })));
        }

        let plan = mlas_sys::ConvPlan::new(
            x_shape[0],
            group,
            input_channels / group,
            &to_i64([x_shape[2], x_shape[3]], "input shape")?,
            &to_i64(kernel, "kernel shape")?,
            &to_i64(dilations, "dilations")?,
            &to_i64(pads, "pads")?,
            &to_i64(strides, "strides")?,
            &to_i64(output_spatial, "output shape")?,
            output_channels / group,
        )
        .ok_or_else(|| EpError::KernelFailed("Conv: MLAS failed to prepare convolution".into()))?;
        let scratch = vec![0.0; plan.working_buffer_elements()];
        Ok(ConvImpl::Fallback(FallbackConv {
            plan,
            scratch: Mutex::new(scratch),
        }))
    }
}

fn byte_ranges_overlap(input: &TensorView<'_>, output: &mut TensorMut<'_>) -> bool {
    let input_start = input.data_ptr::<u8>() as usize;
    let input_end = input_start.saturating_add(input.byte_size());
    let output_start = output.data_ptr_mut::<u8>() as usize;
    let output_end = output_start.saturating_add(output.byte_size());
    output_start < input_end && input_start < output_end
}

impl NchwcConv {
    fn packed_filter_len(&self) -> usize {
        let out = self.nchwc_out_channels;
        let kernel = (self.kernel[0] * self.kernel[1]) as usize;
        let filter_in = if self.filter_bibo {
            round_up(self.weight_shape[1] as usize, self.block)
        } else {
            self.weight_shape[1] as usize
        };
        out * filter_in * kernel
    }

    fn reorder_filter(&self, weights: &[f32]) -> Vec<f32> {
        let mut packed = vec![0.0f32; self.packed_filter_len()];
        if self.filter_bibo {
            mlas_sys::nchwc_reorder_filter_bibo(&self.weight_shape, weights, &mut packed);
        } else {
            mlas_sys::nchwc_reorder_filter_bo(&self.weight_shape, weights, &mut packed);
        }
        packed
    }

    /// Pre-packed filter, cached across inferences when the weight is a constant
    /// initializer (the amortized-packing win); reordered per call otherwise.
    fn filter<'a>(&'a self, weight: &'a TensorView<'_>) -> Result<Cow<'a, [f32]>> {
        if self.weight_constant {
            if let Some(packed) = self.packed_filter.get() {
                return Ok(Cow::Borrowed(packed));
            }
            let weights = to_dense_f32_widen("Conv", weight)?;
            let packed = self.reorder_filter(&weights);
            let _ = self.packed_filter.set(packed);
            return Ok(Cow::Borrowed(
                self.packed_filter
                    .get()
                    .expect("Conv: packed filter just initialized"),
            ));
        }
        let weights = to_dense_f32_widen("Conv", weight)?;
        Ok(Cow::Owned(self.reorder_filter(&weights)))
    }

    fn pad_bias(&self, bias: &[f32]) -> Vec<f32> {
        let mut padded = vec![0.0f32; self.nchwc_out_channels];
        padded[..bias.len()].copy_from_slice(bias);
        padded
    }

    /// Bias padded up to the blocked output-channel count (MLAS reads one bias
    /// per NCHWc output plane). Cached when the bias is a constant initializer.
    fn bias<'a>(&'a self, bias: Option<&'a TensorView<'_>>) -> Result<Option<Cow<'a, [f32]>>> {
        let Some(bias) = bias else {
            return Ok(None);
        };
        if self.bias_constant {
            if let Some(padded) = self.packed_bias.get() {
                return Ok(Some(Cow::Borrowed(padded)));
            }
            let dense = to_dense_f32_widen("Conv", bias)?;
            let _ = self.packed_bias.set(self.pad_bias(&dense));
            return Ok(Some(Cow::Borrowed(
                self.packed_bias
                    .get()
                    .expect("Conv: padded bias just initialized"),
            )));
        }
        let dense = to_dense_f32_widen("Conv", bias)?;
        Ok(Some(Cow::Owned(self.pad_bias(&dense))))
    }

    fn run(
        &self,
        input: &TensorView<'_>,
        weight: &TensorView<'_>,
        bias: Option<&TensorView<'_>>,
        output: &mut [f32],
    ) -> Result<()> {
        let x = to_dense_f32_widen("Conv", input)?;
        let filter = self.filter(weight)?;
        let bias = self.bias(bias)?;

        let input_size = self.input_h * self.input_w;
        let mut input_guard = self
            .input_scratch
            .lock()
            .map_err(|_| EpError::KernelFailed("Conv: input scratch lock poisoned".into()))?;
        let conv_input: &[f32] = if self.reorder_input {
            let blocked_channels = self.conv_input_channels;
            input_guard.clear();
            input_guard.resize(self.batch * blocked_channels * input_size, 0.0);
            for n in 0..self.batch {
                let src = &x[n * self.input_channels * input_size
                    ..(n + 1) * self.input_channels * input_size];
                let dst = &mut input_guard
                    [n * blocked_channels * input_size..(n + 1) * blocked_channels * input_size];
                mlas_sys::nchwc_reorder_input_nchw(src, dst, self.input_channels, input_size);
            }
            &input_guard
        } else {
            &x
        };

        let mut output_guard = self
            .output_scratch
            .lock()
            .map_err(|_| EpError::KernelFailed("Conv: output scratch lock poisoned".into()))?;
        let output_size = self.output_h * self.output_w;
        output_guard.clear();
        output_guard.resize(self.batch * self.nchwc_out_channels * output_size, 0.0);

        mlas_sys::nchwc_conv(
            &[
                self.batch as i64,
                self.conv_input_channels as i64,
                self.input_h as i64,
                self.input_w as i64,
            ],
            &self.kernel,
            &self.dilation,
            &self.pads,
            &self.stride,
            &[
                self.batch as i64,
                self.nchwc_out_channels as i64,
                self.output_h as i64,
                self.output_w as i64,
            ],
            self.group_count,
            conv_input,
            &filter,
            bias.as_deref(),
            &mut output_guard,
            self.activation,
            true,
        );

        mlas_sys::nchwc_reorder_output_nchw(
            &[
                self.batch as i64,
                (self.output_channels()) as i64,
                self.output_h as i64,
                self.output_w as i64,
            ],
            &output_guard,
            output,
        );
        Ok(())
    }

    fn output_channels(&self) -> usize {
        self.weight_shape[0] as usize
    }
}

impl ConvKernel {
    /// Apply the fused activation to a dense NCHW output in place. The NCHWc path
    /// applies the activation inside the MLAS convolution epilogue; the im2col
    /// fallback has no epilogue hook, so we run the equivalent SIMD activation
    /// over the finished output here to keep the two paths numerically identical.
    fn apply_fallback_activation(&self, output: &mut [f32]) {
        let (minimum, maximum) = match self.activation.kind {
            1 => (0.0, f32::INFINITY),                                   // Relu
            5 => (self.activation.values[0], self.activation.values[1]), // Clip
            _ => return,                                                 // Identity / unhandled
        };
        let input = output.to_vec();
        mlas_sys::compute_clip(&input, output, minimum, maximum);
    }
}

impl Kernel for ConvKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Conv", inputs, outputs, 2, 3, 1)?;
        if inputs[0].dtype != DataType::Float32
            || inputs[1].dtype != DataType::Float32
            || inputs
                .get(2)
                .is_some_and(|bias| bias.dtype != DataType::Float32)
            || outputs[0].dtype != DataType::Float32
        {
            return Err(EpError::KernelFailed(
                "Conv: MLAS kernel requires Float32 X, W, optional B, and Y".into(),
            ));
        }
        if inputs[0].shape != self.expected_input_shape
            || inputs[1].shape != self.expected_weight_shape
            || outputs[0].shape != self.expected_output_shape
        {
            return Err(EpError::KernelFailed(format!(
                "Conv: runtime shapes X={:?}, W={:?}, Y={:?}; expected X={:?}, W={:?}, Y={:?}",
                inputs[0].shape,
                inputs[1].shape,
                outputs[0].shape,
                self.expected_input_shape,
                self.expected_weight_shape,
                self.expected_output_shape
            )));
        }
        if let Some(bias) = inputs.get(2)
            && bias.shape != [self.output_channels]
        {
            return Err(EpError::KernelFailed(format!(
                "Conv: bias must have shape [{}], got {:?}",
                self.output_channels, bias.shape
            )));
        }
        if !outputs[0].is_contiguous()
            || inputs
                .iter()
                .any(|input| byte_ranges_overlap(input, &mut outputs[0]))
        {
            return Err(EpError::KernelFailed(
                "Conv: output must be contiguous and must not alias an input".into(),
            ));
        }

        let output_elements = numel(&self.expected_output_shape);
        // SAFETY: the executor validated this contiguous Float32 output view,
        // and `output_elements` is exactly the product of its checked shape.
        let output = unsafe {
            std::slice::from_raw_parts_mut(outputs[0].data_ptr_mut::<f32>(), output_elements)
        };

        match &self.imp {
            ConvImpl::Nchwc(nchwc) => nchwc.run(&inputs[0], &inputs[1], inputs.get(2), output)?,
            ConvImpl::Fallback(fallback) => {
                let x = to_dense_f32_widen("Conv", &inputs[0])?;
                let weights = to_dense_f32_widen("Conv", &inputs[1])?;
                let bias = inputs
                    .get(2)
                    .map(|value| to_dense_f32_widen("Conv", value))
                    .transpose()?;
                let mut scratch = fallback
                    .scratch
                    .lock()
                    .map_err(|_| EpError::KernelFailed("Conv: scratch lock poisoned".into()))?;
                fallback
                    .plan
                    .run(&x, &weights, bias.as_deref(), &mut scratch, output);
                self.apply_fallback_activation(output);
            }
        }

        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let output_spatial =
                self.expected_output_shape[2].saturating_mul(self.expected_output_shape[3]);
            let kernel_elements = self.expected_weight_shape[1]
                .saturating_mul(self.expected_weight_shape[2])
                .saturating_mul(self.expected_weight_shape[3]);
            (self.expected_output_shape[0] as u64)
                .saturating_mul(self.output_channels as u64)
                .saturating_mul(output_spatial as u64)
                .saturating_mul(kernel_elements as u64)
                .saturating_mul(2)
        });
        Ok(())
    }

    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        if let ConvImpl::Nchwc(nchwc) = &mut self.imp {
            nchwc.weight_constant = constant_inputs.get(1).copied().unwrap_or(false);
            nchwc.bias_constant = constant_inputs.get(2).copied().unwrap_or(false);
        }
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::{Attribute, NodeId};

    fn run_conv(
        x_shape: &[usize],
        x: &[f32],
        w_shape: &[usize],
        w: &[f32],
        bias: Option<&[f32]>,
        output_shape: &[usize],
        attributes: &[(&str, Attribute)],
    ) -> Vec<f32> {
        let mut node = Node::new(NodeId(0), "Conv", vec![], vec![]);
        for (name, value) in attributes {
            node.attributes.insert((*name).into(), value.clone());
        }
        let kernel = ConvFactory
            .create(&node, &[x_shape.to_vec(), w_shape.to_vec()])
            .unwrap();
        let x = Owned::f32(x_shape, x);
        let w = Owned::f32(w_shape, w);
        let bias = bias.map(|values| Owned::f32(&[values.len()], values));
        let mut output = Owned::zeros_f32(output_shape);
        let mut inputs = vec![x.view(), w.view()];
        if let Some(bias) = &bias {
            inputs.push(bias.view());
        }
        kernel.execute(&inputs, &mut [output.view_mut()]).unwrap();
        output.to_f32()
    }

    #[test]
    fn conv_bias_stride_and_explicit_padding() {
        let output = run_conv(
            &[1, 1, 3, 3],
            &[1., 2., 3., 4., 5., 6., 7., 8., 9.],
            &[1, 1, 2, 2],
            &[1., 0., 0., 1.],
            Some(&[1.]),
            &[1, 1, 2, 2],
            &[
                ("strides", Attribute::Ints(vec![2, 2])),
                ("pads", Attribute::Ints(vec![1, 1, 0, 0])),
            ],
        );
        assert_eq!(output, vec![2., 4., 8., 15.]);
    }

    #[test]
    fn conv_dilation() {
        let output = run_conv(
            &[1, 1, 3, 3],
            &[1., 2., 3., 4., 5., 6., 7., 8., 9.],
            &[1, 1, 2, 2],
            &[1., 1., 1., 1.],
            None,
            &[1, 1, 1, 1],
            &[("dilations", Attribute::Ints(vec![2, 2]))],
        );
        assert_eq!(output, vec![20.]);
    }

    #[test]
    fn conv_grouped_and_depthwise() {
        let grouped = run_conv(
            &[1, 2, 2, 2],
            &[1., 2., 3., 4., 10., 20., 30., 40.],
            &[2, 1, 1, 1],
            &[2., 3.],
            None,
            &[1, 2, 2, 2],
            &[("group", Attribute::Int(2))],
        );
        assert_eq!(grouped, vec![2., 4., 6., 8., 30., 60., 90., 120.]);

        let depthwise = run_conv(
            &[1, 2, 2, 2],
            &[1., 2., 3., 4., 10., 20., 30., 40.],
            &[4, 1, 1, 1],
            &[1., 2., 3., 4.],
            Some(&[0., 1., 2., 3.]),
            &[1, 4, 2, 2],
            &[("group", Attribute::Int(2))],
        );
        assert_eq!(
            depthwise,
            vec![1., 2., 3., 4., 3., 5., 7., 9., 32., 62., 92., 122., 43., 83., 123., 163.]
        );
    }

    #[test]
    fn same_upper_same_lower_and_valid_geometry() {
        let input = [4, 4];
        let kernel = [3, 3];
        let dilation = [1, 1];
        let stride = [2, 2];
        let (upper_output, upper_pads) =
            output_geometry(input, kernel, dilation, stride, [0; 4], AutoPad::SameUpper).unwrap();
        let (lower_output, lower_pads) =
            output_geometry(input, kernel, dilation, stride, [0; 4], AutoPad::SameLower).unwrap();
        let (valid_output, valid_pads) =
            output_geometry(input, kernel, dilation, stride, [9; 4], AutoPad::Valid).unwrap();
        assert_eq!(upper_output, [2, 2]);
        assert_eq!(lower_output, [2, 2]);
        assert_eq!(upper_pads, [0, 0, 1, 1]);
        assert_eq!(lower_pads, [1, 1, 0, 0]);
        assert_eq!(valid_output, [1, 1]);
        assert_eq!(valid_pads, [0; 4]);
    }
}
