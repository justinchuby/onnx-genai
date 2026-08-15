//! Three-tier ONNX `Conv` kernel with BNNS, im2col+GEMM, and scalar reference dispatch.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::check_arity;
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use crate::strided::numel;

const OP: &str = "Conv";

// ─── Dispatch reachability counters ─────────────────────────────────────────

#[cfg(any(target_os = "macos", target_os = "ios"))]
static CONV_BNNS_TEST_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

static CONV_IM2COL_GEMM_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Counts pointwise 1×1 convolutions that deliberately bypassed BNNS. Only
/// meaningful where BNNS exists, so it is gated to the same platforms as the
/// bypass itself — an ungated counter would compile-warn elsewhere and, worse,
/// would count a "bypass" on platforms that never had BNNS to bypass.
#[cfg(any(target_os = "macos", target_os = "ios"))]
static CONV_POINTWISE_GEMM_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(target_arch = "aarch64")]
static CONV_NEON_DEPTHWISE_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

static CONV_SCALAR_REF_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[derive(Clone, Copy)]
enum AutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

pub struct ConvFactory;

pub struct ConvKernel {
    x_shape: Vec<usize>,
    w_shape: Vec<usize>,
    output_shape: Vec<usize>,
    group: usize,
    strides: Vec<usize>,
    dilations: Vec<usize>,
    pads: Vec<usize>,
    relu: bool,
}

fn positive_attribute(node: &Node, name: &str, rank: usize, default: usize) -> Result<Vec<usize>> {
    let values = node
        .attr(name)
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![default as i64; rank]);
    if values.len() != rank || values.iter().any(|&value| value <= 0) {
        return Err(EpError::KernelFailed(format!(
            "{OP}: {name} must contain {rank} positive values, got {values:?}"
        )));
    }
    Ok(values.into_iter().map(|value| value as usize).collect())
}

fn explicit_pads(node: &Node, rank: usize) -> Result<Vec<usize>> {
    let values = node
        .attr("pads")
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![0; rank * 2]);
    if values.len() != rank * 2 || values.iter().any(|&value| value < 0) {
        return Err(EpError::KernelFailed(format!(
            "{OP}: pads must contain {} non-negative values, got {values:?}",
            rank * 2
        )));
    }
    Ok(values.into_iter().map(|value| value as usize).collect())
}

fn auto_pad(node: &Node) -> Result<AutoPad> {
    match node.attr("auto_pad").and_then(Attribute::as_str) {
        None | Some("NOTSET") => Ok(AutoPad::NotSet),
        Some("SAME_UPPER") => Ok(AutoPad::SameUpper),
        Some("SAME_LOWER") => Ok(AutoPad::SameLower),
        Some("VALID") => Ok(AutoPad::Valid),
        Some(value) => Err(EpError::KernelFailed(format!(
            "{OP}: unsupported auto_pad {value:?}"
        ))),
    }
}

fn output_geometry(
    input: &[usize],
    kernel: &[usize],
    dilations: &[usize],
    strides: &[usize],
    mut pads: Vec<usize>,
    auto_pad: AutoPad,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let rank = input.len();
    let mut output = vec![0; rank];
    for axis in 0..rank {
        let effective = dilations[axis]
            .checked_mul(kernel[axis].saturating_sub(1))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| EpError::KernelFailed(format!("{OP}: kernel size overflow")))?;
        match auto_pad {
            AutoPad::SameUpper | AutoPad::SameLower => {
                output[axis] = input[axis].div_ceil(strides[axis]);
                let total = output[axis]
                    .saturating_sub(1)
                    .checked_mul(strides[axis])
                    .and_then(|value| value.checked_add(effective))
                    .map(|value| value.saturating_sub(input[axis]))
                    .ok_or_else(|| EpError::KernelFailed(format!("{OP}: padding size overflow")))?;
                let begin = if matches!(auto_pad, AutoPad::SameUpper) {
                    total / 2
                } else {
                    total - total / 2
                };
                pads[axis] = begin;
                pads[axis + rank] = total - begin;
            }
            AutoPad::Valid => {
                pads[axis] = 0;
                pads[axis + rank] = 0;
                output[axis] = input[axis]
                    .checked_sub(effective)
                    .map_or(0, |value| value / strides[axis] + 1);
            }
            AutoPad::NotSet => {
                let padded = input[axis]
                    .checked_add(pads[axis])
                    .and_then(|value| value.checked_add(pads[axis + rank]))
                    .ok_or_else(|| EpError::KernelFailed(format!("{OP}: padded size overflow")))?;
                output[axis] = padded
                    .checked_sub(effective)
                    .map_or(0, |value| value / strides[axis] + 1);
            }
        }
    }
    Ok((output, pads))
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

impl KernelFactory for ConvFactory {
    fn create(&self, node: &Node, shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let x_shape = shapes
            .first()
            .ok_or_else(|| EpError::KernelFailed(format!("{OP}: missing X shape")))?;
        let w_shape = shapes
            .get(1)
            .ok_or_else(|| EpError::KernelFailed(format!("{OP}: missing W shape")))?;
        if x_shape.len() != w_shape.len() || !matches!(x_shape.len(), 3 | 4) {
            return Err(EpError::KernelFailed(format!(
                "{OP}: requires matching rank-3 NCL or rank-4 NCHW tensors, got X={x_shape:?}, W={w_shape:?}"
            )));
        }
        let spatial_rank = x_shape.len() - 2;
        let group = node.attr("group").and_then(Attribute::as_int).unwrap_or(1);
        if group <= 0 {
            return Err(EpError::KernelFailed(format!(
                "{OP}: group must be positive, got {group}"
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
                "{OP}: incompatible channels/group: X channels={input_channels}, W={w_shape:?}, group={group}"
            )));
        }
        let kernel = w_shape[2..].to_vec();
        if kernel.contains(&0) {
            return Err(EpError::KernelFailed(format!(
                "{OP}: kernel dimensions must be positive, got {kernel:?}"
            )));
        }
        if let Some(declared) = node.attr("kernel_shape").and_then(Attribute::as_ints)
            && (declared.len() != spatial_rank
                || declared
                    .iter()
                    .zip(&kernel)
                    .any(|(&value, &actual)| value <= 0 || value as usize != actual))
        {
            return Err(EpError::KernelFailed(format!(
                "{OP}: kernel_shape must match W spatial shape {kernel:?}, got {declared:?}"
            )));
        }
        let strides = positive_attribute(node, "strides", spatial_rank, 1)?;
        let dilations = positive_attribute(node, "dilations", spatial_rank, 1)?;
        let (output_spatial, pads) = output_geometry(
            &x_shape[2..],
            &kernel,
            &dilations,
            &strides,
            explicit_pads(node, spatial_rank)?,
            auto_pad(node)?,
        )?;
        let mut output_shape = vec![x_shape[0], output_channels];
        output_shape.extend(output_spatial);
        let relu = matches!(
            node.attr("activation").and_then(Attribute::as_str),
            Some("Relu")
        );
        Ok(Box::new(ConvKernel {
            x_shape: x_shape.clone(),
            w_shape: w_shape.clone(),
            output_shape,
            group,
            strides,
            dilations,
            pads,
            relu,
        }))
    }
}

impl Kernel for ConvKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(OP, inputs, outputs, 2, 3, 1)?;
        let dtype = outputs[0].dtype;
        if !matches!(
            dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) || inputs[0].dtype != dtype
            || inputs[1].dtype != dtype
            || inputs.get(2).is_some_and(|bias| bias.dtype != dtype)
        {
            return Err(EpError::KernelFailed(format!(
                "{OP}: X, W, optional B, and Y must share f32, f16, or bf16 dtype"
            )));
        }
        if inputs[0].shape != self.x_shape
            || inputs[1].shape != self.w_shape
            || outputs[0].shape != self.output_shape
        {
            return Err(EpError::KernelFailed(format!(
                "{OP}: runtime shapes X={:?}, W={:?}, Y={:?}; expected X={:?}, W={:?}, Y={:?}",
                inputs[0].shape,
                inputs[1].shape,
                outputs[0].shape,
                self.x_shape,
                self.w_shape,
                self.output_shape
            )));
        }
        let output_channels = self.w_shape[0];
        if let Some(bias) = inputs.get(2)
            && bias.shape != [output_channels]
        {
            return Err(EpError::KernelFailed(format!(
                "{OP}: bias must have shape [{output_channels}], got {:?}",
                bias.shape
            )));
        }

        let x = to_dense_f32_widen(OP, &inputs[0])?;
        let weights = to_dense_f32_widen(OP, &inputs[1])?;
        let bias = inputs
            .get(2)
            .map(|value| to_dense_f32_widen(OP, value))
            .transpose()?;

        // Promote rank-3 (1D conv) shapes to rank-4 by inserting H=1 so the
        // BNNS and im2col+GEMM tiers handle them without a separate code path.
        // Data layout is unchanged: [N, C, W] → [N, C, 1, W] is a logical
        // reshape with no memory movement.
        let (x_shape_4d, w_shape_4d, out_shape_4d, strides_2d, dilations_2d, pads_2d);
        let (eff_x_shape, eff_w_shape, eff_out_shape, eff_strides, eff_dilations, eff_pads);
        if self.x_shape.len() == 3 {
            // 1D → 2D: insert H=1 at spatial index 0
            x_shape_4d = [self.x_shape[0], self.x_shape[1], 1, self.x_shape[2]];
            w_shape_4d = [self.w_shape[0], self.w_shape[1], 1, self.w_shape[2]];
            out_shape_4d = [
                self.output_shape[0],
                self.output_shape[1],
                1,
                self.output_shape[2],
            ];
            strides_2d = [1, self.strides[0]];
            dilations_2d = [1, self.dilations[0]];
            pads_2d = [0, self.pads[0], 0, self.pads[1]];
            eff_x_shape = x_shape_4d.as_slice();
            eff_w_shape = w_shape_4d.as_slice();
            eff_out_shape = out_shape_4d.as_slice();
            eff_strides = strides_2d.as_slice();
            eff_dilations = dilations_2d.as_slice();
            eff_pads = pads_2d.as_slice();
        } else {
            eff_x_shape = &self.x_shape;
            eff_w_shape = &self.w_shape;
            eff_out_shape = &self.output_shape;
            eff_strides = &self.strides;
            eff_dilations = &self.dilations;
            eff_pads = &self.pads;
        }

        let is_rank4 = eff_x_shape.len() == 4;

        let output = if is_rank4 {
            // Try Tier 1 (BNNS) for undilated convolutions, unless the pointwise
            // guard below says BNNS would lose. See skip_bnns_for_pointwise_1x1.
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                let skip_bnns = skip_bnns_for_pointwise_1x1(eff_w_shape, eff_out_shape);

                if self.group == 1
                    && eff_dilations.iter().all(|&d| d == 1)
                    && !skip_bnns
                    && let Some(result) = bnns::bnns_conv_execute(
                        &x,
                        &weights,
                        bias.as_deref(),
                        eff_x_shape,
                        eff_w_shape,
                        eff_out_shape,
                        eff_strides,
                        eff_pads,
                        1,
                        self.relu,
                    )
                {
                    CONV_BNNS_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    write_dense_f32_narrow(OP, &mut outputs[0], &result)?;
                    record_conv_metrics(inputs, outputs, self);
                    return Ok(());
                }
            }

            // Tier 2a: Direct NEON depthwise convolution (aarch64 only).
            // Depthwise proper: groups == in_channels == out_channels, ic_per_group == 1.
            // This eliminates the im2col buffer entirely — depthwise is memory-bound
            // (M=1, K=kernel_size per group), so the im2col expansion costs more in
            // memory traffic than it saves in arithmetic density.
            #[cfg(target_arch = "aarch64")]
            if self.group > 1
                && eff_w_shape[1] == 1
                && self.group == eff_x_shape[1]
                && eff_w_shape[0] == eff_x_shape[1]
            {
                CONV_NEON_DEPTHWISE_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let result = neon_depthwise::depthwise_conv_execute(
                    &x,
                    &weights,
                    bias.as_deref(),
                    eff_x_shape,
                    eff_w_shape,
                    eff_out_shape,
                    eff_strides,
                    eff_dilations,
                    eff_pads,
                    self.relu,
                );
                write_dense_f32_narrow(OP, &mut outputs[0], &result)?;
                record_conv_metrics(inputs, outputs, self);
                return Ok(());
            }

            // Tier 2b: im2col + GEMM.
            CONV_IM2COL_GEMM_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Track pointwise 1×1 convolutions that bypassed BNNS and reached
            // the direct GEMM fast path due to low spatial reuse. The predicate
            // is the *same function* the dispatch used above — re-deriving it
            // here let the counter and the dispatch drift apart silently, which
            // would make the manifest claim prove the wrong thing.
            //
            // Gated to the same platforms as the BNNS attempt: where BNNS does
            // not exist there is nothing to bypass, and counting there would
            // report a bypass that never happened.
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            if skip_bnns_for_pointwise_1x1(eff_w_shape, eff_out_shape) {
                CONV_POINTWISE_GEMM_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if self.group == 1 {
                im2col_gemm_execute(
                    &x,
                    &weights,
                    bias.as_deref(),
                    eff_x_shape,
                    eff_w_shape,
                    eff_out_shape,
                    eff_strides,
                    eff_dilations,
                    eff_pads,
                    self.relu,
                )?
            } else {
                grouped_im2col_gemm_execute(
                    &x,
                    &weights,
                    bias.as_deref(),
                    eff_x_shape,
                    eff_w_shape,
                    eff_out_shape,
                    self.group,
                    eff_strides,
                    eff_dilations,
                    eff_pads,
                    self.relu,
                )?
            }
        } else {
            // Tier 3: scalar reference for non-spatial convolutions (should not reach here
            // after the rank-3→rank-4 promotion above, but kept as a safety net).
            CONV_SCALAR_REF_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            scalar_ref_execute(
                &x,
                &weights,
                bias.as_deref(),
                &self.x_shape,
                &self.w_shape,
                &self.output_shape,
                self.group,
                &self.strides,
                &self.dilations,
                &self.pads,
                self.relu,
            )
        };

        write_dense_f32_narrow(OP, &mut outputs[0], &output)?;
        record_conv_metrics(inputs, outputs, self);
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

/// Should a 1×1 pointwise Conv bypass BNNS and go straight to im2col+GEMM?
///
/// This is the single source of truth for the decision. It exists as a function
/// because the dispatch site and the instrumentation counter previously
/// re-derived it independently, with the counter using bare literals (`16_384`,
/// `6`) against the dispatch's named constants. Two copies of a predicate drift,
/// and when they drift the counter proves a claim the code no longer makes.
///
/// Mechanism: BNNS copies the full weight tensor (OC × IC × 4 bytes) and
/// creates/destroys a filter on every call. Each weight element is reused N
/// times (once per spatial position). When N/IC < BNNS_REUSE_MIN, that copy
/// overhead exceeds BNNS's compute advantage.
///
/// L1 threshold: Apple Silicon E-cores have 64 KB L1 data cache; P-cores have
/// 128 KB (M1–M4). We deliberately size to the smaller E-core L1 so the guard is
/// conservative and portable across core types and parts — the work may land on
/// either. Below this size the weight copy is cache-local and ~free regardless
/// of which core runs it, so BNNS overhead is negligible.
///
/// Reuse threshold (BNNS_REUSE_MIN = 6) is **fitted, not derived**. It is the
/// observed minimum N/IC ratio at which BNNS's AMX kernel recovers its
/// copy/pack overhead on M1 Max, measured over shapes from 24→144 @ 14×14 to
/// 2048→512 @ 7×7 — 15 shapes spanning MobileNetV2 and ResNet, interleaved A/B
/// at loads 5–53, corroborated 3×. The *mechanism* generalizes across Apple
/// Silicon; the *coefficient* is an M1 Max measurement. Do not "correct" it to
/// a value implied by a datasheet: it is empirical.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn skip_bnns_for_pointwise_1x1(w_shape: &[usize], out_shape: &[usize]) -> bool {
    const APPLE_SILICON_L1_F32: usize = 16_384; // E-core 64 KB / 4 bytes per f32
    const BNNS_REUSE_MIN: usize = 6; // fitted on M1 Max; see doc comment

    let is_1x1 = w_shape[2..].iter().all(|&k| k == 1);
    let weight_elems = w_shape[0] * w_shape[1];
    let spatial_size = out_shape[2..].iter().product::<usize>();

    is_1x1 && weight_elems > APPLE_SILICON_L1_F32 && spatial_size <= w_shape[1] * BNNS_REUSE_MIN
}

fn record_conv_metrics(inputs: &[TensorView], outputs: &mut [TensorMut], kernel: &ConvKernel) {
    let output_channels = kernel.w_shape[0];
    let output_spatial_size = numel(&kernel.output_shape[2..]);
    let channels_per_group = kernel.x_shape[1] / kernel.group;
    let kernel_size = numel(&kernel.w_shape[2..]);
    crate::trace::record_kernel_metrics(inputs, outputs, || {
        (kernel.x_shape[0] as u64)
            .saturating_mul(output_channels as u64)
            .saturating_mul(output_spatial_size as u64)
            .saturating_mul(channels_per_group as u64)
            .saturating_mul(kernel_size as u64)
            .saturating_mul(2)
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tier 1: BNNS Convolution (macOS/iOS)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod bnns {
    use std::ffi::c_void;
    const BNNS_DATA_TYPE_FLOAT32: u32 = 0x10020;
    const BNNS_DATA_LAYOUT_IMAGE_CHW: u32 = 0x30000;
    const BNNS_DATA_LAYOUT_CONV_WEIGHTS_OIHW: u32 = 0x40000;

    #[repr(C)]
    struct BNNSNDArrayDescriptor {
        flags: u32,
        layout: u32,
        size: [usize; 8],
        stride: [usize; 8],
        data: *mut c_void,
        data_type: u32,
        _pad0: u32,
        table_data: *mut c_void,
        table_data_type: u32,
        data_scale: f32,
        data_bias: f32,
        _pad1: u32,
    }
    #[repr(C)]
    struct BNNSActivation {
        function: u32,
        alpha: f32,
        beta: f32,
        iscale: i32,
        ioffset: i32,
        ishift: i32,
        iscale_per_channel: *const i32,
        ioffset_per_channel: *const i32,
        ishift_per_channel: *const i32,
    }
    #[repr(C)]
    struct BNNSLayerParametersConvolution {
        i_desc: BNNSNDArrayDescriptor,
        w_desc: BNNSNDArrayDescriptor,
        o_desc: BNNSNDArrayDescriptor,
        bias: BNNSNDArrayDescriptor,
        activation: BNNSActivation,
        x_stride: usize,
        y_stride: usize,
        x_dilation_stride: usize,
        y_dilation_stride: usize,
        x_padding: usize,
        y_padding: usize,
        groups: usize,
        pad: [usize; 4],
    }
    type BNNSFilter = *mut c_void;
    #[repr(C)]
    struct BNNSFilterParameters {
        _opaque: [u8; 0],
    }
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn BNNSFilterCreateLayerConvolution(
            params: *const BNNSLayerParametersConvolution,
            filter_params: *const BNNSFilterParameters,
        ) -> BNNSFilter;
        fn BNNSFilterApply(filter: BNNSFilter, input: *const c_void, output: *mut c_void) -> i32;
        fn BNNSFilterDestroy(filter: BNNSFilter);
    }

    fn make_nd(
        layout: u32,
        sz: &[usize],
        st: &[usize],
        data: *mut c_void,
    ) -> BNNSNDArrayDescriptor {
        let mut size = [0usize; 8];
        let mut stride = [0usize; 8];
        for (i, (&s, &t)) in sz.iter().zip(st.iter()).enumerate() {
            size[i] = s;
            stride[i] = t;
        }
        BNNSNDArrayDescriptor {
            flags: 0,
            layout,
            size,
            stride,
            data,
            data_type: BNNS_DATA_TYPE_FLOAT32,
            _pad0: 0,
            table_data: std::ptr::null_mut(),
            table_data_type: 0,
            data_scale: 0.0,
            data_bias: 0.0,
            _pad1: 0,
        }
    }

    pub fn bnns_conv_execute(
        x: &[f32],
        weights: &[f32],
        bias: Option<&[f32]>,
        x_shape: &[usize],
        w_shape: &[usize],
        out_shape: &[usize],
        strides: &[usize],
        pads: &[usize],
        groups: usize,
        relu: bool,
    ) -> Option<Vec<f32>> {
        // Require symmetric padding.
        if pads[0] != pads[2] || pads[1] != pads[3] {
            return None;
        }
        let (batch, ic, ih, iw) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
        let (oc, kh, kw) = (w_shape[0], w_shape[2], w_shape[3]);
        // w_shape[1] = input_channels_per_group (ONNX Conv weight layout: [O, I/groups, kH, kW])
        let ic_per_group = w_shape[1];
        let oc_per_group = oc / groups;
        let (oh, ow) = (out_shape[2], out_shape[3]);
        let in_stride = ic * ih * iw;
        let out_stride = oc * oh * ow;
        let mut w_copy = weights.to_vec();
        let mut bias_vec = bias.map_or(vec![0.0f32; oc], |b| b.to_vec());
        let params = BNNSLayerParametersConvolution {
            i_desc: make_nd(
                BNNS_DATA_LAYOUT_IMAGE_CHW,
                &[iw, ih, ic],
                &[1, iw, iw * ih],
                std::ptr::null_mut(),
            ),
            w_desc: make_nd(
                BNNS_DATA_LAYOUT_CONV_WEIGHTS_OIHW,
                &[kw, kh, ic_per_group, oc_per_group],
                &[1, kw, kw * kh, kw * kh * ic_per_group],
                w_copy.as_mut_ptr() as *mut c_void,
            ),
            o_desc: make_nd(
                BNNS_DATA_LAYOUT_IMAGE_CHW,
                &[ow, oh, oc],
                &[1, ow, ow * oh],
                std::ptr::null_mut(),
            ),
            bias: make_nd(0x10000, &[oc], &[1], bias_vec.as_mut_ptr() as *mut c_void),
            activation: BNNSActivation {
                function: if relu { 1 } else { 0 },
                alpha: 0.0,
                beta: 0.0,
                iscale: 0,
                ioffset: 0,
                ishift: 0,
                iscale_per_channel: std::ptr::null(),
                ioffset_per_channel: std::ptr::null(),
                ishift_per_channel: std::ptr::null(),
            },
            x_stride: strides[1],
            y_stride: strides[0],
            x_dilation_stride: 1,
            y_dilation_stride: 1,
            x_padding: pads[1],
            y_padding: pads[0],
            groups,
            pad: [0; 4],
        };
        let filter = unsafe { BNNSFilterCreateLayerConvolution(&params, std::ptr::null()) };
        if filter.is_null() {
            return None;
        }
        let mut output = vec![0.0f32; batch * out_stride];
        // Apply the filter per-image. BNNSFilterApplyBatch with batch>1
        // crashes (SIGSEGV inside libBNNS) for convolution filters on macOS —
        // the single-image BNNSFilterApply is safe and BNNS still uses its
        // internal thread pool per call, so the compute advantage is preserved.
        let mut ok = true;
        for b in 0..batch {
            let rc = unsafe {
                BNNSFilterApply(
                    filter,
                    x.as_ptr().add(b * in_stride) as *const c_void,
                    output.as_mut_ptr().add(b * out_stride) as *mut c_void,
                )
            };
            if rc != 0 {
                ok = false;
                break;
            }
        }
        unsafe {
            BNNSFilterDestroy(filter);
        }
        if ok { Some(output) } else { None }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tier 2: im2col + GEMM
// ═══════════════════════════════════════════════════════════════════════════════

fn im2col_gemm_execute(
    x: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    x_shape: &[usize],
    w_shape: &[usize],
    out_shape: &[usize],
    strides: &[usize],
    dilations: &[usize],
    pads: &[usize],
    relu: bool,
) -> Result<Vec<f32>> {
    let (batch, ic, ih, iw) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    let (oc, kh, kw) = (w_shape[0], w_shape[2], w_shape[3]);
    let (oh, ow) = (out_shape[2], out_shape[3]);
    let m = oc;
    let k = ic * kh * kw;
    let n = oh * ow;
    if n == 0 || m == 0 {
        return Ok(vec![]);
    }
    let is_1x1 = kh == 1
        && kw == 1
        && strides.iter().all(|&s| s == 1)
        && dilations.iter().all(|&d| d == 1)
        && pads.iter().all(|&p| p == 0);
    let backend = crate::backend::CpuBackend::auto_detect();
    let mut output = vec![0.0f32; batch * m * n];
    for b in 0..batch {
        let x_b = &x[b * ic * ih * iw..][..ic * ih * iw];
        let o_b = &mut output[b * m * n..][..m * n];
        if is_1x1 {
            super::matmul::gemm_with_backend(backend, weights, x_b, o_b, m, k, n)?;
        } else {
            let mut col = vec![0.0f32; k * n];
            im2col(
                x_b, ic, ih, iw, kh, kw, strides, dilations, pads, oh, ow, &mut col,
            );
            super::matmul::gemm_with_backend(backend, weights, &col, o_b, m, k, n)?;
        }
        if let Some(bias) = bias {
            for oc_idx in 0..m {
                let bv = bias[oc_idx];
                for s in 0..n {
                    o_b[oc_idx * n + s] += bv;
                }
            }
        }
        if relu {
            for v in o_b.iter_mut() {
                *v = v.max(0.0);
            }
        }
    }
    Ok(output)
}

fn im2col(
    input: &[f32],
    c: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    strides: &[usize],
    dilations: &[usize],
    pads: &[usize],
    oh: usize,
    ow: usize,
    col: &mut [f32],
) {
    let (pt, pl) = (pads[0], pads[1]);
    let (sh, sw, dh, dw) = (strides[0], strides[1], dilations[0], dilations[1]);
    let mut idx = 0;
    for ch in 0..c {
        let co = ch * ih * iw;
        for kh_i in 0..kh {
            for kw_i in 0..kw {
                for o_h in 0..oh {
                    let i_h = (o_h * sh + kh_i * dh).wrapping_sub(pt);
                    for o_w in 0..ow {
                        let i_w = (o_w * sw + kw_i * dw).wrapping_sub(pl);
                        col[idx] = if i_h < ih && i_w < iw {
                            input[co + i_h * iw + i_w]
                        } else {
                            0.0
                        };
                        idx += 1;
                    }
                }
            }
        }
    }
}

/// Grouped im2col + GEMM: for each group, performs im2col on the group's input
/// channels and applies GEMM with the group's weight slice.
fn grouped_im2col_gemm_execute(
    x: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    x_shape: &[usize],
    w_shape: &[usize],
    out_shape: &[usize],
    group: usize,
    strides: &[usize],
    dilations: &[usize],
    pads: &[usize],
    relu: bool,
) -> Result<Vec<f32>> {
    let (batch, ic, ih, iw) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    let (oc, kh, kw) = (w_shape[0], w_shape[2], w_shape[3]);
    let (oh, ow) = (out_shape[2], out_shape[3]);
    let ic_per_group = ic / group;
    let oc_per_group = oc / group;
    // Per-group GEMM: M = oc_per_group, K = ic_per_group * kH * kW, N = oH * oW
    let m = oc_per_group;
    let k = ic_per_group * kh * kw;
    let n = oh * ow;
    if n == 0 || m == 0 {
        return Ok(vec![]);
    }
    let backend = crate::backend::CpuBackend::auto_detect();
    let mut output = vec![0.0f32; batch * oc * n];
    let mut col = vec![0.0f32; k * n];
    for b in 0..batch {
        for g in 0..group {
            let x_offset = b * ic * ih * iw + g * ic_per_group * ih * iw;
            let x_group = &x[x_offset..][..ic_per_group * ih * iw];
            let w_offset = g * oc_per_group * k;
            let w_group = &weights[w_offset..][..oc_per_group * k];
            let o_offset = b * oc * n + g * oc_per_group * n;
            let o_group = &mut output[o_offset..][..oc_per_group * n];
            let is_1x1 = kh == 1
                && kw == 1
                && strides.iter().all(|&s| s == 1)
                && dilations.iter().all(|&d| d == 1)
                && pads.iter().all(|&p| p == 0);
            if is_1x1 {
                super::matmul::gemm_with_backend(backend, w_group, x_group, o_group, m, k, n)?;
            } else {
                im2col(
                    x_group,
                    ic_per_group,
                    ih,
                    iw,
                    kh,
                    kw,
                    strides,
                    dilations,
                    pads,
                    oh,
                    ow,
                    &mut col,
                );
                super::matmul::gemm_with_backend(backend, w_group, &col, o_group, m, k, n)?;
            }
            if let Some(bias) = bias {
                for oc_idx in 0..m {
                    let bv = bias[g * oc_per_group + oc_idx];
                    for s in 0..n {
                        o_group[oc_idx * n + s] += bv;
                    }
                }
            }
            if relu {
                for v in o_group.iter_mut() {
                    *v = v.max(0.0);
                }
            }
        }
    }
    Ok(output)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tier 2a: Direct NEON depthwise convolution (aarch64)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Depthwise convolution is memory-bound (M=1, K=kernel_size per group).
// im2col expands the input by ~kernel_size× in memory with almost no
// arithmetic density to show for it. A direct kernel reads input once,
// accumulates with NEON, and writes output — roughly 18× less memory
// traffic for 3×3.
//
// Specialized inner loops for 3×3 stride-1 and 3×3 stride-2 cover
// the overwhelming majority of MobileNet/EfficientNet depthwise layers.
// A general fallback handles all other shapes.

#[cfg(target_arch = "aarch64")]
mod neon_depthwise {
    use std::arch::aarch64::*;

    /// Entry point: direct depthwise convolution.
    /// Requires: groups == in_channels == out_channels, w_shape[1] == 1.
    pub fn depthwise_conv_execute(
        x: &[f32],
        weights: &[f32],
        bias: Option<&[f32]>,
        x_shape: &[usize],
        w_shape: &[usize],
        out_shape: &[usize],
        strides: &[usize],
        dilations: &[usize],
        pads: &[usize],
        relu: bool,
    ) -> Vec<f32> {
        let (batch, c, ih, iw) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
        let (kh, kw) = (w_shape[2], w_shape[3]);
        let (oh, ow) = (out_shape[2], out_shape[3]);
        let (sh, sw) = (strides[0], strides[1]);
        let (dh, dw) = (dilations[0], dilations[1]);
        let (pt, pl, pb, pr) = (pads[0], pads[1], pads[2], pads[3]);
        let _ = (pb, pr); // used implicitly through output geometry

        let mut output = vec![0.0f32; batch * c * oh * ow];

        for b in 0..batch {
            let x_batch = &x[b * c * ih * iw..][..c * ih * iw];
            let o_batch = &mut output[b * c * oh * ow..][..c * oh * ow];

            // Dispatch to specialized kernels when possible.
            if kh == 3 && kw == 3 && dh == 1 && dw == 1 {
                if sh == 1 && sw == 1 {
                    depthwise_3x3_s1(x_batch, weights, bias, c, ih, iw, oh, ow, pt, pl, o_batch);
                } else if sh == 2 && sw == 2 {
                    depthwise_3x3_s2(x_batch, weights, bias, c, ih, iw, oh, ow, pt, pl, o_batch);
                } else {
                    depthwise_general(
                        x_batch, weights, bias, c, ih, iw, kh, kw, oh, ow, sh, sw, dh, dw, pt, pl,
                        o_batch,
                    );
                }
            } else {
                depthwise_general(
                    x_batch, weights, bias, c, ih, iw, kh, kw, oh, ow, sh, sw, dh, dw, pt, pl,
                    o_batch,
                );
            }

            if relu {
                apply_relu_neon(o_batch);
            }
        }

        output
    }

    /// Apply ReLU in-place using NEON.
    fn apply_relu_neon(data: &mut [f32]) {
        let len = data.len();
        let chunks = len / 4;
        let remainder = len % 4;
        unsafe {
            let zero = vdupq_n_f32(0.0);
            let ptr = data.as_mut_ptr();
            for i in 0..chunks {
                let v = vld1q_f32(ptr.add(i * 4));
                vst1q_f32(ptr.add(i * 4), vmaxq_f32(v, zero));
            }
            for i in (chunks * 4)..(chunks * 4 + remainder) {
                let p = ptr.add(i);
                *p = (*p).max(0.0);
            }
        }
    }

    /// Specialized 3×3 depthwise, stride 1, undilated.
    ///
    /// Splits each output row into three regions:
    /// - Left boundary (scalar): output cols where the leftmost kernel column
    ///   would access a negative (padded) input column.
    /// - Interior (NEON, chunks of 4): output cols where all 9 kernel positions
    ///   have guaranteed in-bounds column accesses, enabling raw pointer loads.
    /// - Right boundary (scalar): output cols where the rightmost kernel column
    ///   exceeds the input width, plus any NEON tail.
    fn depthwise_3x3_s1(
        x: &[f32],
        weights: &[f32],
        bias: Option<&[f32]>,
        c: usize,
        ih: usize,
        iw: usize,
        oh: usize,
        ow: usize,
        pt: usize,
        pl: usize,
        output: &mut [f32],
    ) {
        // For a NEON chunk of 4 outputs starting at o_w, the input accesses are:
        //   kc=0: col = o_w - pl       .. o_w+3 - pl
        //   kc=2: col = o_w - pl + 2   .. o_w+3 - pl + 2 = o_w + 5 - pl
        // All valid when: o_w >= pl AND o_w + 5 - pl < iw, i.e. o_w < iw + pl - 5
        // Also need chunk to fit: o_w + 4 <= ow, i.e. o_w < ow - 3
        let neon_start = pl;
        let neon_limit = (iw + pl).saturating_sub(5).min(ow.saturating_sub(3));

        for ch in 0..c {
            let x_ch = &x[ch * ih * iw..][..ih * iw];
            let w_ch = &weights[ch * 9..][..9];
            let o_ch = &mut output[ch * oh * ow..][..oh * ow];
            let bv = bias.map_or(0.0, |b| b[ch]);

            let w: [f32; 9] = [
                w_ch[0], w_ch[1], w_ch[2], w_ch[3], w_ch[4], w_ch[5], w_ch[6], w_ch[7], w_ch[8],
            ];

            for o_h in 0..oh {
                let i_h_base = o_h.wrapping_sub(pt);
                let row_out = o_h * ow;

                // Left boundary (scalar).
                for o_w in 0..neon_start.min(ow) {
                    o_ch[row_out + o_w] = scalar_3x3_pixel(x_ch, &w, bv, ih, iw, i_h_base, o_w, pl);
                }

                // Interior: NEON chunks of 4 with guaranteed in-bounds loads.
                let mut o_w = neon_start;
                if neon_start < neon_limit {
                    unsafe {
                        let bias_v = vdupq_n_f32(bv);
                        while o_w < neon_limit {
                            let i_w_base = o_w - pl; // safe: o_w >= pl
                            let mut acc = bias_v;

                            for kr in 0..3usize {
                                let i_h = i_h_base.wrapping_add(kr);
                                if i_h >= ih {
                                    continue;
                                }
                                let row_ptr = x_ch.as_ptr().add(i_h * iw + i_w_base);
                                let v0 = vld1q_f32(row_ptr);
                                let v1 = vld1q_f32(row_ptr.add(1));
                                let v2 = vld1q_f32(row_ptr.add(2));
                                acc = vfmaq_f32(acc, v0, vdupq_n_f32(w[kr * 3]));
                                acc = vfmaq_f32(acc, v1, vdupq_n_f32(w[kr * 3 + 1]));
                                acc = vfmaq_f32(acc, v2, vdupq_n_f32(w[kr * 3 + 2]));
                            }

                            vst1q_f32(o_ch.as_mut_ptr().add(row_out + o_w), acc);
                            o_w += 4;
                        }
                    }
                }

                // Right boundary + NEON tail (scalar).
                for o_w in o_w..ow {
                    o_ch[row_out + o_w] = scalar_3x3_pixel(x_ch, &w, bv, ih, iw, i_h_base, o_w, pl);
                }
            }
        }
    }

    /// Scalar computation for a single output pixel of a 3×3 depthwise conv.
    #[inline(always)]
    fn scalar_3x3_pixel(
        x_ch: &[f32],
        w: &[f32; 9],
        bv: f32,
        ih: usize,
        iw: usize,
        i_h_base: usize,
        o_w: usize,
        pl: usize,
    ) -> f32 {
        let mut sum = bv;
        for kr in 0..3usize {
            let i_h = i_h_base.wrapping_add(kr);
            if i_h >= ih {
                continue;
            }
            for kc in 0..3usize {
                let i_w = o_w.wrapping_sub(pl).wrapping_add(kc);
                if i_w < iw {
                    sum += x_ch[i_h * iw + i_w] * w[kr * 3 + kc];
                }
            }
        }
        sum
    }

    /// Specialized 3×3 depthwise, stride 2, undilated.
    fn depthwise_3x3_s2(
        x: &[f32],
        weights: &[f32],
        bias: Option<&[f32]>,
        c: usize,
        ih: usize,
        iw: usize,
        oh: usize,
        ow: usize,
        pt: usize,
        pl: usize,
        output: &mut [f32],
    ) {
        for ch in 0..c {
            let x_ch = &x[ch * ih * iw..][..ih * iw];
            let w_ch = &weights[ch * 9..][..9];
            let o_ch = &mut output[ch * oh * ow..][..oh * ow];
            let bv = bias.map_or(0.0, |b| b[ch]);

            let w: [f32; 9] = [
                w_ch[0], w_ch[1], w_ch[2], w_ch[3], w_ch[4], w_ch[5], w_ch[6], w_ch[7], w_ch[8],
            ];

            for o_h in 0..oh {
                let i_h_base = (o_h * 2).wrapping_sub(pt);

                // For stride-2, consecutive output pixels read from input
                // positions 2 apart, so we can't trivially use contiguous
                // NEON loads. Use scalar with per-element accumulation.
                // This is still faster than im2col because there is no
                // buffer allocation.
                for o_w in 0..ow {
                    let i_w_base = (o_w * 2).wrapping_sub(pl);
                    let mut sum = bv;
                    for kr in 0..3usize {
                        let i_h = i_h_base.wrapping_add(kr);
                        if i_h >= ih {
                            continue;
                        }
                        for kc in 0..3usize {
                            let i_w = i_w_base.wrapping_add(kc);
                            if i_w < iw {
                                sum += x_ch[i_h * iw + i_w] * w[kr * 3 + kc];
                            }
                        }
                    }
                    o_ch[o_h * ow + o_w] = sum;
                }
            }
        }
    }

    /// General depthwise convolution — handles arbitrary kernel sizes, strides,
    /// and dilations. No im2col buffer.
    fn depthwise_general(
        x: &[f32],
        weights: &[f32],
        bias: Option<&[f32]>,
        c: usize,
        ih: usize,
        iw: usize,
        kh: usize,
        kw: usize,
        oh: usize,
        ow: usize,
        sh: usize,
        sw: usize,
        dh: usize,
        dw: usize,
        pt: usize,
        pl: usize,
        output: &mut [f32],
    ) {
        let ks = kh * kw;
        for ch in 0..c {
            let x_ch = &x[ch * ih * iw..][..ih * iw];
            let w_ch = &weights[ch * ks..][..ks];
            let o_ch = &mut output[ch * oh * ow..][..oh * ow];
            let bv = bias.map_or(0.0, |b| b[ch]);

            for o_h in 0..oh {
                let i_h_base = (o_h * sh).wrapping_sub(pt);
                for o_w in 0..ow {
                    let i_w_base = (o_w * sw).wrapping_sub(pl);
                    let mut sum = bv;
                    for kr in 0..kh {
                        let i_h = i_h_base.wrapping_add(kr * dh);
                        if i_h >= ih {
                            continue;
                        }
                        for kc in 0..kw {
                            let i_w = i_w_base.wrapping_add(kc * dw);
                            if i_w < iw {
                                sum += x_ch[i_h * iw + i_w] * w_ch[kr * kw + kc];
                            }
                        }
                    }
                    o_ch[o_h * ow + o_w] = sum;
                }
            }
        }
    }
}

fn scalar_ref_execute(
    x: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    x_shape: &[usize],
    w_shape: &[usize],
    output_shape: &[usize],
    group: usize,
    strides: &[usize],
    dilations: &[usize],
    pads: &[usize],
    relu: bool,
) -> Vec<f32> {
    let spatial_rank = x_shape.len() - 2;
    let input_spatial = &x_shape[2..];
    let kernel_shape = &w_shape[2..];
    let output_spatial = &output_shape[2..];
    let input_spatial_size = numel(input_spatial);
    let kernel_size = numel(kernel_shape);
    let output_spatial_size = numel(output_spatial);
    let input_spatial_strides = contiguous_strides(input_spatial);
    let kernel_strides = contiguous_strides(kernel_shape);
    let output_strides = contiguous_strides(output_spatial);
    let input_channels = x_shape[1];
    let output_channels = w_shape[0];
    let channels_per_group = input_channels / group;
    let outputs_per_group = output_channels / group;
    let mut output = vec![0.0f32; numel(output_shape)];
    for batch in 0..x_shape[0] {
        for oc in 0..output_channels {
            let grp = oc / outputs_per_group;
            for ol in 0..output_spatial_size {
                let mut rem = ol;
                let mut coords = vec![0; spatial_rank];
                for axis in 0..spatial_rank {
                    coords[axis] = rem / output_strides[axis];
                    rem %= output_strides[axis];
                }
                let mut sum = bias.map_or(0.0, |b| b[oc]);
                for ic in 0..channels_per_group {
                    let abs_c = grp * channels_per_group + ic;
                    for kl in 0..kernel_size {
                        let mut kr = kl;
                        let mut io = 0usize;
                        let mut ok = true;
                        for axis in 0..spatial_rank {
                            let kc = kr / kernel_strides[axis];
                            kr %= kernel_strides[axis];
                            let coord = coords[axis]
                                .saturating_mul(strides[axis])
                                .saturating_add(kc.saturating_mul(dilations[axis]));
                            let Some(coord) = coord.checked_sub(pads[axis]) else {
                                ok = false;
                                break;
                            };
                            if coord >= input_spatial[axis] {
                                ok = false;
                                break;
                            }
                            io += coord * input_spatial_strides[axis];
                        }
                        if ok {
                            let xi = (batch * input_channels + abs_c) * input_spatial_size + io;
                            let wi = (oc * channels_per_group + ic) * kernel_size + kl;
                            sum += x[xi] * weights[wi];
                        }
                    }
                }
                if relu {
                    sum = sum.max(0.0);
                }
                output[(batch * output_channels + oc) * output_spatial_size + ol] = sum;
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::NodeId;
    use std::sync::atomic::Ordering;

    fn run(
        x_shape: &[usize],
        x: &[f32],
        w_shape: &[usize],
        w: &[f32],
        bias: Option<&[f32]>,
        output_shape: &[usize],
        attributes: &[(&str, Attribute)],
    ) -> Vec<f32> {
        let mut node = Node::new(NodeId(0), OP, vec![], vec![]);
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
    fn conv_2d_bias_stride_and_explicit_padding() {
        assert_eq!(
            run(
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
            ),
            vec![2., 4., 8., 15.]
        );
    }

    #[test]
    fn conv_2d_dilation_and_non_square_kernel() {
        assert_eq!(
            run(
                &[1, 1, 3, 5],
                &(1..=15).map(|value| value as f32).collect::<Vec<_>>(),
                &[1, 1, 2, 3],
                &[1.; 6],
                None,
                &[1, 1, 1, 3],
                &[("dilations", Attribute::Ints(vec![2, 1]))],
            ),
            vec![42., 48., 54.]
        );
    }

    #[test]
    fn conv_2d_groups_and_depthwise_multiplier() {
        assert_eq!(
            run(
                &[1, 2, 2, 2],
                &[1., 2., 3., 4., 10., 20., 30., 40.],
                &[4, 1, 1, 1],
                &[1., 2., 3., 4.],
                Some(&[0., 1., 2., 3.]),
                &[1, 4, 2, 2],
                &[("group", Attribute::Int(2))],
            ),
            vec![
                1., 2., 3., 4., 3., 5., 7., 9., 32., 62., 92., 122., 43., 83., 123., 163.
            ]
        );
    }

    #[test]
    fn conv_1d_matches_onnxruntime_reference() {
        let x = (1..=16).map(|value| value as f32).collect::<Vec<_>>();
        let w = (1..=18).map(|value| value as f32 * 0.1).collect::<Vec<_>>();
        let actual = run(
            &[1, 2, 8],
            &x,
            &[3, 2, 3],
            &w,
            Some(&[0.5, -0.5, 1.0]),
            &[1, 3, 4],
            &[
                ("strides", Attribute::Ints(vec![2])),
                ("pads", Attribute::Ints(vec![1, 1])),
            ],
        );
        let expected = [
            11.8, 19.2, 23.4, 27.6, 24.0, 43.4, 54.8, 66.2, 38.7, 70.1, 88.7, 107.3,
        ];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-5 * expected.abs().max(1.0));
        }
    }

    #[test]
    fn conv_same_upper_and_same_lower_split_odd_padding() {
        let x = [1., 2., 3., 4.];
        let upper = run(
            &[1, 1, 4],
            &x,
            &[1, 1, 3],
            &[1.; 3],
            None,
            &[1, 1, 2],
            &[
                ("strides", Attribute::Ints(vec![2])),
                ("auto_pad", Attribute::String(b"SAME_UPPER".to_vec())),
            ],
        );
        let lower = run(
            &[1, 1, 4],
            &x,
            &[1, 1, 3],
            &[1.; 3],
            None,
            &[1, 1, 2],
            &[
                ("strides", Attribute::Ints(vec![2])),
                ("auto_pad", Attribute::String(b"SAME_LOWER".to_vec())),
            ],
        );
        assert_eq!(upper, vec![6., 7.]);
        assert_eq!(lower, vec![3., 9.]);
    }

    #[test]
    fn conv_empty_output_when_kernel_exceeds_unpadded_input() {
        assert!(
            run(
                &[1, 1, 2],
                &[1., 2.],
                &[1, 1, 3],
                &[1.; 3],
                None,
                &[1, 1, 0],
                &[],
            )
            .is_empty()
        );
    }

    #[test]
    fn conv_bfloat16_widens_and_narrows() {
        let node = Node::new(NodeId(0), OP, vec![], vec![]);
        let kernel = ConvFactory
            .create(&node, &[vec![1, 1, 2, 2], vec![1, 1, 1, 1]])
            .unwrap();
        let x = Owned::bf16(&[1, 1, 2, 2], &[1., 2., 3., 4.]);
        let w = Owned::bf16(&[1, 1, 1, 1], &[2.]);
        let bias = Owned::bf16(&[1], &[1.]);
        let mut output = Owned::zeros(DataType::BFloat16, &[1, 1, 2, 2]);
        kernel
            .execute(&[x.view(), w.view(), bias.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.to_bf16_as_f32(), vec![3., 5., 7., 9.]);
    }

    // ─── Tier dispatch reachability ─────────────────────────────────────

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn fast_path_2d_group1_uses_conv_bnns() {
        let before = super::CONV_BNNS_TEST_HITS.load(Ordering::Relaxed);
        let input = vec![1.0; 3 * 8 * 8];
        let weight = vec![0.01; 16 * 3 * 3 * 3];
        let bias = vec![0.0; 16];
        let _ = run(
            &[1, 3, 8, 8],
            &input,
            &[16, 3, 3, 3],
            &weight,
            Some(&bias),
            &[1, 16, 6, 6],
            &[],
        );
        let after = super::CONV_BNNS_TEST_HITS.load(Ordering::Relaxed);
        assert!(
            after > before,
            "Standard 2D group=1 undilated conv did not reach BNNS path"
        );
    }

    #[test]
    fn im2col_gemm_handles_dilated_conv() {
        let before = super::CONV_IM2COL_GEMM_TEST_HITS.load(Ordering::Relaxed);
        let input = vec![1.0; 25];
        let weight = vec![1.0; 9];
        let _ = run(
            &[1, 1, 5, 5],
            &input,
            &[1, 1, 3, 3],
            &weight,
            None,
            &[1, 1, 1, 1],
            &[("dilations", Attribute::Ints(vec![2, 2]))],
        );
        let after = super::CONV_IM2COL_GEMM_TEST_HITS.load(Ordering::Relaxed);
        assert!(
            after > before,
            "Dilated conv did not reach im2col+GEMM path"
        );
    }

    #[test]
    fn grouped_conv_reaches_im2col_gemm() {
        // True depthwise: groups == in_channels == out_channels, 3x3 kernel.
        // On aarch64, this now reaches the direct NEON path instead of im2col.
        #[cfg(target_arch = "aarch64")]
        let before = super::CONV_NEON_DEPTHWISE_TEST_HITS.load(Ordering::Relaxed);
        #[cfg(not(target_arch = "aarch64"))]
        let before = super::CONV_IM2COL_GEMM_TEST_HITS.load(Ordering::Relaxed);
        let ic = 32usize;
        let h = 14usize;
        let w = 14usize;
        let input: Vec<f32> = (0..ic * h * w)
            .map(|i: usize| ((i.wrapping_mul(37) % 257) as f32 - 128.0) / 128.0)
            .collect();
        let weight: Vec<f32> = (0..ic * 9)
            .map(|i: usize| ((i.wrapping_mul(53) % 131) as f32 - 65.0) / 65.0)
            .collect();
        let bias: Vec<f32> = (0..ic).map(|i| i as f32 * 0.01).collect();
        let _ = run(
            &[1, ic, h, w],
            &input,
            &[ic, 1, 3, 3],
            &weight,
            Some(&bias),
            &[1, ic, h, w],
            &[
                ("group", Attribute::Int(ic as i64)),
                ("pads", Attribute::Ints(vec![1, 1, 1, 1])),
            ],
        );
        #[cfg(target_arch = "aarch64")]
        {
            let after = super::CONV_NEON_DEPTHWISE_TEST_HITS.load(Ordering::Relaxed);
            assert!(
                after > before,
                "Depthwise conv (groups={ic}) did not reach NEON depthwise path"
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let after = super::CONV_IM2COL_GEMM_TEST_HITS.load(Ordering::Relaxed);
            assert!(
                after > before,
                "Depthwise conv (groups={ic}) did not reach im2col+GEMM path"
            );
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn grouped_conv_reaches_im2col_gemm_non_apple() {
        let before = super::CONV_IM2COL_GEMM_TEST_HITS.load(Ordering::Relaxed);
        let _ = run(
            &[1, 2, 2, 2],
            &[1., 2., 3., 4., 10., 20., 30., 40.],
            &[4, 1, 1, 1],
            &[1., 2., 3., 4.],
            Some(&[0., 1., 2., 3.]),
            &[1, 4, 2, 2],
            &[("group", Attribute::Int(2))],
        );
        let after = super::CONV_IM2COL_GEMM_TEST_HITS.load(Ordering::Relaxed);
        assert!(
            after > before,
            "Grouped conv did not reach im2col+GEMM path"
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn depthwise_conv_reaches_neon_direct() {
        let before = super::CONV_NEON_DEPTHWISE_TEST_HITS.load(Ordering::Relaxed);
        let ic = 16usize;
        let input: Vec<f32> = (0..ic * 7 * 7)
            .map(|i| ((i.wrapping_mul(37) % 257) as f32 - 128.0) / 128.0)
            .collect();
        let weight: Vec<f32> = (0..ic * 9)
            .map(|i| ((i.wrapping_mul(53) % 131) as f32 - 65.0) / 65.0)
            .collect();
        let bias: Vec<f32> = (0..ic).map(|i| i as f32 * 0.01).collect();
        let _ = run(
            &[1, ic, 7, 7],
            &input,
            &[ic, 1, 3, 3],
            &weight,
            Some(&bias),
            &[1, ic, 7, 7],
            &[
                ("group", Attribute::Int(ic as i64)),
                ("pads", Attribute::Ints(vec![1, 1, 1, 1])),
            ],
        );
        let after = super::CONV_NEON_DEPTHWISE_TEST_HITS.load(Ordering::Relaxed);
        assert!(
            after > before,
            "Depthwise conv did not reach NEON direct path on aarch64"
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn depthwise_conv_5x5_reaches_neon_direct() {
        // Non-3×3 depthwise should also hit the NEON general path.
        let before = super::CONV_NEON_DEPTHWISE_TEST_HITS.load(Ordering::Relaxed);
        let ic = 24usize;
        let input: Vec<f32> = (0..ic * 14 * 14)
            .map(|i| ((i.wrapping_mul(37) % 257) as f32 - 128.0) / 128.0)
            .collect();
        let weight: Vec<f32> = (0..ic * 25)
            .map(|i| ((i.wrapping_mul(53) % 131) as f32 - 65.0) / 65.0)
            .collect();
        let bias: Vec<f32> = (0..ic).map(|i| i as f32 * 0.01).collect();
        let _ = run(
            &[1, ic, 14, 14],
            &input,
            &[ic, 1, 5, 5],
            &weight,
            Some(&bias),
            &[1, ic, 10, 10],
            &[("group", Attribute::Int(ic as i64))],
        );
        let after = super::CONV_NEON_DEPTHWISE_TEST_HITS.load(Ordering::Relaxed);
        assert!(
            after > before,
            "5×5 depthwise conv did not reach NEON direct path on aarch64"
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn depthwise_1d_promoted_reaches_neon_direct() {
        // 1D depthwise should be promoted to 2D and then hit the NEON path.
        let before = super::CONV_NEON_DEPTHWISE_TEST_HITS.load(Ordering::Relaxed);
        let ic = 32usize;
        let input: Vec<f32> = (0..ic * 64)
            .map(|i| ((i.wrapping_mul(37) % 257) as f32 - 128.0) / 128.0)
            .collect();
        let weight: Vec<f32> = (0..ic * 3)
            .map(|i| ((i.wrapping_mul(53) % 131) as f32 - 65.0) / 65.0)
            .collect();
        let _ = run(
            &[1, ic, 64],
            &input,
            &[ic, 1, 3],
            &weight,
            None,
            &[1, ic, 64],
            &[
                ("group", Attribute::Int(ic as i64)),
                ("pads", Attribute::Ints(vec![1, 1])),
            ],
        );
        let after = super::CONV_NEON_DEPTHWISE_TEST_HITS.load(Ordering::Relaxed);
        assert!(
            after > before,
            "1D depthwise conv did not reach NEON direct path on aarch64"
        );
    }

    #[test]
    fn conv_1d_reaches_im2col_or_bnns() {
        // 1D convs are promoted to 2D and dispatch through the accelerated path
        let before_im2col = super::CONV_IM2COL_GEMM_TEST_HITS.load(Ordering::Relaxed);
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let before_bnns = super::CONV_BNNS_TEST_HITS.load(Ordering::Relaxed);
        let x = (1..=16).map(|v| v as f32).collect::<Vec<_>>();
        let w = (1..=18).map(|v| v as f32 * 0.1).collect::<Vec<_>>();
        let _ = run(
            &[1, 2, 8],
            &x,
            &[3, 2, 3],
            &w,
            Some(&[0.5, -0.5, 1.0]),
            &[1, 3, 4],
            &[
                ("strides", Attribute::Ints(vec![2])),
                ("pads", Attribute::Ints(vec![1, 1])),
            ],
        );
        let after_im2col = super::CONV_IM2COL_GEMM_TEST_HITS.load(Ordering::Relaxed);
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let after_bnns = super::CONV_BNNS_TEST_HITS.load(Ordering::Relaxed);
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let reached = after_im2col > before_im2col || after_bnns > before_bnns;
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let reached = after_im2col > before_im2col;
        assert!(
            reached,
            "1D conv did not reach accelerated path (im2col or BNNS)"
        );
    }

    // ─── Scalar reference reachability and exclusion ────────────────────

    #[test]
    fn scalar_ref_fires_for_non_rank4_safety_net() {
        // Construct a ConvKernel directly with a 5D shape to force the scalar
        // reference path. This bypasses the factory (which rejects rank != 3|4)
        // and proves the counter is live. The scalar reference handles arbitrary
        // spatial rank correctly.
        let before = super::CONV_SCALAR_REF_TEST_HITS.load(Ordering::Relaxed);

        // 3D convolution: [N=1, C=1, D=2, H=2, W=2], kernel [1, 1, 1, 1, 1]
        let kernel = super::ConvKernel {
            x_shape: vec![1, 1, 2, 2, 2],
            w_shape: vec![1, 1, 1, 1, 1],
            output_shape: vec![1, 1, 2, 2, 2],
            group: 1,
            strides: vec![1, 1, 1],
            dilations: vec![1, 1, 1],
            pads: vec![0, 0, 0, 0, 0, 0],
            relu: false,
        };
        let x = Owned::f32(&[1, 1, 2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let w = Owned::f32(&[1, 1, 1, 1, 1], &[2.0]);
        let mut output = Owned::zeros_f32(&[1, 1, 2, 2, 2]);
        kernel
            .execute(&[x.view(), w.view()], &mut [output.view_mut()])
            .unwrap();

        let after = super::CONV_SCALAR_REF_TEST_HITS.load(Ordering::Relaxed);
        assert!(
            after > before,
            "5D conv did not reach scalar reference path"
        );
        // Verify correctness: 1x1x1 kernel with weight=2 is a pointwise multiply
        assert_eq!(output.to_f32(), vec![2., 4., 6., 8., 10., 12., 14., 16.]);
    }

    #[test]
    fn scalar_ref_excluded_for_standard_2d_conv() {
        // A standard rank-4 conv MUST NOT fall to the scalar reference —
        // it should be served by BNNS (macOS) or im2col+GEMM. We verify by
        // checking that im2col or BNNS incremented their counter instead.
        let before_im2col = super::CONV_IM2COL_GEMM_TEST_HITS.load(Ordering::Relaxed);
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let before_bnns = super::CONV_BNNS_TEST_HITS.load(Ordering::Relaxed);
        let input = vec![1.0; 3 * 8 * 8];
        let weight = vec![0.01; 16 * 3 * 3 * 3];
        let bias = vec![0.0; 16];
        let _ = run(
            &[1, 3, 8, 8],
            &input,
            &[16, 3, 3, 3],
            &weight,
            Some(&bias),
            &[1, 16, 6, 6],
            &[],
        );
        let after_im2col = super::CONV_IM2COL_GEMM_TEST_HITS.load(Ordering::Relaxed);
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let after_bnns = super::CONV_BNNS_TEST_HITS.load(Ordering::Relaxed);
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let reached_fast = after_im2col > before_im2col || after_bnns > before_bnns;
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let reached_fast = after_im2col > before_im2col;
        assert!(
            reached_fast,
            "Standard 2D conv fell to scalar reference — a faster tier should have handled it"
        );
    }

    #[test]
    fn scalar_ref_excluded_for_1d_conv() {
        // Rank-3 (1D) convs are promoted to rank-4 and must NOT fall to scalar ref.
        // This is the exact bug that made Whisper's Conv path 643× slower.
        // Verify by checking that a fast tier fired.
        let before_im2col = super::CONV_IM2COL_GEMM_TEST_HITS.load(Ordering::Relaxed);
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let before_bnns = super::CONV_BNNS_TEST_HITS.load(Ordering::Relaxed);
        let x = (1..=16).map(|v| v as f32).collect::<Vec<_>>();
        let w = (1..=18).map(|v| v as f32 * 0.1).collect::<Vec<_>>();
        let _ = run(
            &[1, 2, 8],
            &x,
            &[3, 2, 3],
            &w,
            Some(&[0.5, -0.5, 1.0]),
            &[1, 3, 4],
            &[
                ("strides", Attribute::Ints(vec![2])),
                ("pads", Attribute::Ints(vec![1, 1])),
            ],
        );
        let after_im2col = super::CONV_IM2COL_GEMM_TEST_HITS.load(Ordering::Relaxed);
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let after_bnns = super::CONV_BNNS_TEST_HITS.load(Ordering::Relaxed);
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let reached_fast = after_im2col > before_im2col || after_bnns > before_bnns;
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let reached_fast = after_im2col > before_im2col;
        assert!(
            reached_fast,
            "1D conv fell to scalar reference — rank-3 promotion should route to im2col/BNNS"
        );
    }

    // ─── Pointwise 1×1 dispatch and parity ────────────────────────────

    #[test]
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn pointwise_1x1_reaches_gemm_not_bnns() {
        // 1×1 pointwise Conv at small spatial sizes with large weights must route
        // to im2col+GEMM (direct GEMM for 1×1), NOT to BNNS.
        // Guard: weight_elems > E-core L1 (16384 f32) AND spatial ≤ IC × 6.
        // Apple-only: on other platforms there is no BNNS to avoid, so the
        // assertion would be vacuous.
        let before = super::CONV_POINTWISE_GEMM_TEST_HITS.load(Ordering::Relaxed);
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let before_bnns = super::CONV_BNNS_TEST_HITS.load(Ordering::Relaxed);

        // Shape: [1, 256, 7, 7] × [128, 256, 1, 1]
        // weight_elems = 256*128 = 32768 > 16384 (L1)
        // spatial = 49, IC*6 = 1536, 49 ≤ 1536 → skip BNNS
        let ic = 256usize;
        let oc = 128usize;
        let h = 7usize;
        let w = 7usize;
        let x: Vec<f32> = (0..ic * h * w)
            .map(|i| ((i.wrapping_mul(37) % 257) as f32 - 128.0) / 128.0)
            .collect();
        let weights: Vec<f32> = (0..oc * ic)
            .map(|i| ((i.wrapping_mul(53) % 131) as f32 - 65.0) / 65.0)
            .collect();
        let bias: Vec<f32> = (0..oc).map(|i| i as f32 * 0.01).collect();
        let _ = run(
            &[1, ic, h, w],
            &x,
            &[oc, ic, 1, 1],
            &weights,
            Some(&bias),
            &[1, oc, h, w],
            &[],
        );
        let after = super::CONV_POINTWISE_GEMM_TEST_HITS.load(Ordering::Relaxed);
        assert!(
            after > before,
            "1×1 pointwise conv at small spatial did not reach the direct GEMM path"
        );
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let after_bnns = super::CONV_BNNS_TEST_HITS.load(Ordering::Relaxed);
            assert_eq!(
                after_bnns, before_bnns,
                "1×1 pointwise conv at small spatial incorrectly reached BNNS (should use GEMM)"
            );
        }
    }

    #[test]
    fn pointwise_1x1_matches_scalar_reference() {
        // Verify numerics parity between the tiered dispatch (GEMM path)
        // and the scalar reference for shapes that trigger the BNNS bypass
        // (weight > L1 and spatial ≤ IC × 6).
        for &(ic, oc, h, w) in &[(96usize, 128usize, 14usize, 14usize), (128, 256, 7, 7)] {
            parity_check(
                &[1, ic, h, w],
                &[oc, ic, 1, 1],
                &[1, 1],
                &[1, 1],
                &[0, 0, 0, 0],
            );
        }
    }

    // ─── Numerics parity ────────────────────────────────────────────────

    fn parity_check(
        x_shape: &[usize],
        w_shape: &[usize],
        strides: &[usize],
        dilations: &[usize],
        pads: &[usize],
    ) {
        let x_count: usize = x_shape.iter().product();
        let w_count: usize = w_shape.iter().product();
        let x: Vec<f32> = (0..x_count)
            .map(|i| ((i.wrapping_mul(37) % 257) as f32 - 128.0) / 128.0)
            .collect();
        let w: Vec<f32> = (0..w_count)
            .map(|i| ((i.wrapping_mul(53) % 131) as f32 - 65.0) / 65.0)
            .collect();
        let bias: Vec<f32> = (0..w_shape[0])
            .map(|i| (i as f32 - w_shape[0] as f32 / 2.0) * 0.1)
            .collect();
        let spatial_rank = x_shape.len() - 2;
        let mut output_shape = vec![x_shape[0], w_shape[0]];
        for axis in 0..spatial_rank {
            let effective = dilations[axis] * (w_shape[2 + axis] - 1) + 1;
            let padded = x_shape[2 + axis] + pads[axis] + pads[axis + spatial_rank];
            let out_dim = if padded >= effective {
                (padded - effective) / strides[axis] + 1
            } else {
                0
            };
            output_shape.push(out_dim);
        }
        let reference = super::scalar_ref_execute(
            &x,
            &w,
            Some(&bias),
            x_shape,
            w_shape,
            &output_shape,
            1,
            strides,
            dilations,
            pads,
            false,
        );
        let mut attrs: Vec<(&str, Attribute)> = vec![];
        if strides.iter().any(|&s| s != 1) {
            attrs.push((
                "strides",
                Attribute::Ints(strides.iter().map(|&s| s as i64).collect()),
            ));
        }
        if dilations.iter().any(|&d| d != 1) {
            attrs.push((
                "dilations",
                Attribute::Ints(dilations.iter().map(|&d| d as i64).collect()),
            ));
        }
        if pads.iter().any(|&p| p != 0) {
            attrs.push((
                "pads",
                Attribute::Ints(pads.iter().map(|&p| p as i64).collect()),
            ));
        }
        let tiered = run(x_shape, &x, w_shape, &w, Some(&bias), &output_shape, &attrs);
        assert_eq!(reference.len(), tiered.len());
        let max_diff = reference
            .iter()
            .zip(tiered.iter())
            .map(|(r, t)| (r - t).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "parity: max_diff={max_diff} (x={x_shape:?}, w={w_shape:?})"
        );
    }

    fn parity_check_grouped(
        x_shape: &[usize],
        w_shape: &[usize],
        group: usize,
        strides: &[usize],
        dilations: &[usize],
        pads: &[usize],
    ) {
        let x_count: usize = x_shape.iter().product();
        let w_count: usize = w_shape.iter().product();
        let x: Vec<f32> = (0..x_count)
            .map(|i| ((i.wrapping_mul(37) % 257) as f32 - 128.0) / 128.0)
            .collect();
        let w: Vec<f32> = (0..w_count)
            .map(|i| ((i.wrapping_mul(53) % 131) as f32 - 65.0) / 65.0)
            .collect();
        let bias: Vec<f32> = (0..w_shape[0])
            .map(|i| (i as f32 - w_shape[0] as f32 / 2.0) * 0.1)
            .collect();
        let spatial_rank = x_shape.len() - 2;
        let mut output_shape = vec![x_shape[0], w_shape[0]];
        for axis in 0..spatial_rank {
            let effective = dilations[axis] * (w_shape[2 + axis] - 1) + 1;
            let padded = x_shape[2 + axis] + pads[axis] + pads[axis + spatial_rank];
            let out_dim = if padded >= effective {
                (padded - effective) / strides[axis] + 1
            } else {
                0
            };
            output_shape.push(out_dim);
        }
        let reference = super::scalar_ref_execute(
            &x,
            &w,
            Some(&bias),
            x_shape,
            w_shape,
            &output_shape,
            group,
            strides,
            dilations,
            pads,
            false,
        );
        let mut attrs: Vec<(&str, Attribute)> = vec![("group", Attribute::Int(group as i64))];
        if strides.iter().any(|&s| s != 1) {
            attrs.push((
                "strides",
                Attribute::Ints(strides.iter().map(|&s| s as i64).collect()),
            ));
        }
        if dilations.iter().any(|&d| d != 1) {
            attrs.push((
                "dilations",
                Attribute::Ints(dilations.iter().map(|&d| d as i64).collect()),
            ));
        }
        if pads.iter().any(|&p| p != 0) {
            attrs.push((
                "pads",
                Attribute::Ints(pads.iter().map(|&p| p as i64).collect()),
            ));
        }
        let tiered = run(x_shape, &x, w_shape, &w, Some(&bias), &output_shape, &attrs);
        assert_eq!(reference.len(), tiered.len());
        let max_diff = reference
            .iter()
            .zip(tiered.iter())
            .map(|(r, t)| (r - t).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "grouped parity: max_diff={max_diff} (x={x_shape:?}, w={w_shape:?}, group={group})"
        );
    }

    #[test]
    fn conv_parity_3x3_stride1_pad1() {
        parity_check(
            &[1, 64, 56, 56],
            &[64, 64, 3, 3],
            &[1, 1],
            &[1, 1],
            &[1, 1, 1, 1],
        );
    }

    #[test]
    fn conv_parity_3x3_stride2() {
        parity_check(
            &[1, 64, 56, 56],
            &[128, 64, 3, 3],
            &[2, 2],
            &[1, 1],
            &[1, 1, 1, 1],
        );
    }

    #[test]
    fn conv_parity_1x1_pointwise() {
        parity_check(
            &[1, 64, 56, 56],
            &[256, 64, 1, 1],
            &[1, 1],
            &[1, 1],
            &[0, 0, 0, 0],
        );
    }

    #[test]
    fn conv_parity_dilated() {
        parity_check(
            &[1, 32, 28, 28],
            &[32, 32, 3, 3],
            &[1, 1],
            &[2, 2],
            &[2, 2, 2, 2],
        );
    }

    #[test]
    fn conv_parity_asymmetric_padding() {
        parity_check(
            &[1, 16, 7, 7],
            &[32, 16, 3, 3],
            &[2, 2],
            &[1, 1],
            &[1, 1, 0, 0],
        );
    }

    #[test]
    fn conv_parity_non_multiple_channels() {
        parity_check(
            &[1, 13, 11, 11],
            &[17, 13, 3, 3],
            &[1, 1],
            &[1, 1],
            &[1, 1, 1, 1],
        );
    }

    // ─── Depthwise / grouped conv parity tests ──────────────────────────

    #[test]
    fn conv_parity_depthwise_3x3_stride1() {
        // True depthwise: groups == in_channels == out_channels
        parity_check_grouped(
            &[1, 32, 28, 28],
            &[32, 1, 3, 3],
            32,
            &[1, 1],
            &[1, 1],
            &[1, 1, 1, 1],
        );
    }

    #[test]
    fn conv_parity_depthwise_3x3_stride2() {
        parity_check_grouped(
            &[1, 64, 56, 56],
            &[64, 1, 3, 3],
            64,
            &[2, 2],
            &[1, 1],
            &[1, 1, 1, 1],
        );
    }

    #[test]
    fn conv_parity_depthwise_asymmetric_padding() {
        parity_check_grouped(
            &[1, 32, 7, 7],
            &[32, 1, 3, 3],
            32,
            &[2, 2],
            &[1, 1],
            &[1, 1, 0, 0],
        );
    }

    #[test]
    fn conv_parity_depthwise_non_vector_width_channels() {
        // 13 channels — not a multiple of any SIMD vector width
        parity_check_grouped(
            &[1, 13, 14, 14],
            &[13, 1, 3, 3],
            13,
            &[1, 1],
            &[1, 1],
            &[1, 1, 1, 1],
        );
    }

    #[test]
    fn conv_parity_depthwise_dilation() {
        parity_check_grouped(
            &[1, 32, 28, 28],
            &[32, 1, 3, 3],
            32,
            &[1, 1],
            &[2, 2],
            &[2, 2, 2, 2],
        );
    }

    #[test]
    fn conv_parity_grouped_not_depthwise() {
        // groups < in_channels (grouped but not depthwise): groups=4, in=16, out=32
        parity_check_grouped(
            &[1, 16, 14, 14],
            &[32, 4, 3, 3],
            4,
            &[1, 1],
            &[1, 1],
            &[1, 1, 1, 1],
        );
    }

    #[test]
    fn conv_parity_depthwise_channel_multiplier() {
        // Channel multiplier: groups=16, in=16, out=32 (multiplier=2)
        parity_check_grouped(
            &[1, 16, 14, 14],
            &[32, 1, 3, 3],
            16,
            &[1, 1],
            &[1, 1],
            &[1, 1, 1, 1],
        );
    }

    #[test]
    fn conv_parity_depthwise_5x5_mobilenet_style() {
        // EfficientNet uses 5x5 depthwise
        parity_check_grouped(
            &[1, 48, 28, 28],
            &[48, 1, 5, 5],
            48,
            &[1, 1],
            &[1, 1],
            &[2, 2, 2, 2],
        );
    }

    #[test]
    fn conv_parity_depthwise_1pixel_spatial() {
        // 1-pixel spatial dim — edge case for NEON tail handling
        parity_check_grouped(
            &[1, 8, 1, 1],
            &[8, 1, 1, 1],
            8,
            &[1, 1],
            &[1, 1],
            &[0, 0, 0, 0],
        );
    }

    #[test]
    fn conv_parity_depthwise_3x3_stride1_single_channel() {
        // Single channel — minimum case for depthwise
        parity_check_grouped(
            &[1, 1, 7, 7],
            &[1, 1, 3, 3],
            1,
            &[1, 1],
            &[1, 1],
            &[1, 1, 1, 1],
        );
    }

    #[test]
    fn conv_parity_depthwise_3x3_stride1_odd_channels() {
        // 7 channels — not a multiple of 4 (NEON vector width)
        parity_check_grouped(
            &[1, 7, 11, 11],
            &[7, 1, 3, 3],
            7,
            &[1, 1],
            &[1, 1],
            &[1, 1, 1, 1],
        );
    }

    #[test]
    fn conv_parity_depthwise_3x3_stride2_narrow() {
        // 3-pixel width input with stride 2 — boundary case
        parity_check_grouped(
            &[1, 16, 3, 3],
            &[16, 1, 3, 3],
            16,
            &[2, 2],
            &[1, 1],
            &[1, 1, 1, 1],
        );
    }

    #[test]
    fn conv_parity_depthwise_batch2() {
        // Multi-batch
        parity_check_grouped(
            &[2, 32, 14, 14],
            &[32, 1, 3, 3],
            32,
            &[1, 1],
            &[1, 1],
            &[1, 1, 1, 1],
        );
    }

    // ─── Rank-3 (1D) Conv parity tests ─────────────────────────────────
    // The rank-3→rank-4 promotion is brand-new surface; a dimension-indexing
    // bug there yields plausible output, not a crash.

    fn parity_check_1d(
        x_shape: &[usize],
        w_shape: &[usize],
        group: usize,
        strides: &[usize],
        dilations: &[usize],
        pads: &[usize],
    ) {
        assert_eq!(x_shape.len(), 3);
        assert_eq!(w_shape.len(), 3);
        let x_count: usize = x_shape.iter().product();
        let w_count: usize = w_shape.iter().product();
        let x: Vec<f32> = (0..x_count)
            .map(|i| ((i.wrapping_mul(37) % 257) as f32 - 128.0) / 128.0)
            .collect();
        let w: Vec<f32> = (0..w_count)
            .map(|i| ((i.wrapping_mul(53) % 131) as f32 - 65.0) / 65.0)
            .collect();
        let bias: Vec<f32> = (0..w_shape[0])
            .map(|i| (i as f32 - w_shape[0] as f32 / 2.0) * 0.1)
            .collect();
        let effective = dilations[0] * (w_shape[2] - 1) + 1;
        let padded = x_shape[2] + pads[0] + pads[1];
        let out_w = if padded >= effective {
            (padded - effective) / strides[0] + 1
        } else {
            0
        };
        let output_shape = vec![x_shape[0], w_shape[0], out_w];
        let reference = super::scalar_ref_execute(
            &x,
            &w,
            Some(&bias),
            x_shape,
            w_shape,
            &output_shape,
            group,
            strides,
            dilations,
            pads,
            false,
        );
        let mut attrs: Vec<(&str, Attribute)> = vec![];
        if group > 1 {
            attrs.push(("group", Attribute::Int(group as i64)));
        }
        if strides.iter().any(|&s| s != 1) {
            attrs.push((
                "strides",
                Attribute::Ints(strides.iter().map(|&s| s as i64).collect()),
            ));
        }
        if dilations.iter().any(|&d| d != 1) {
            attrs.push((
                "dilations",
                Attribute::Ints(dilations.iter().map(|&d| d as i64).collect()),
            ));
        }
        if pads.iter().any(|&p| p != 0) {
            attrs.push((
                "pads",
                Attribute::Ints(pads.iter().map(|&p| p as i64).collect()),
            ));
        }
        let tiered = run(x_shape, &x, w_shape, &w, Some(&bias), &output_shape, &attrs);
        assert_eq!(reference.len(), tiered.len());
        let max_diff = reference
            .iter()
            .zip(tiered.iter())
            .map(|(r, t)| (r - t).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "rank-3 parity: max_diff={max_diff} (x={x_shape:?}, w={w_shape:?}, group={group})"
        );
    }

    #[test]
    fn conv_parity_rank3_group1_stride1() {
        // Whisper-style: 80 mel channels, kernel 3, stride 1
        parity_check_1d(&[1, 80, 3000], &[512, 80, 3], 1, &[1], &[1], &[1, 1]);
    }

    #[test]
    fn conv_parity_rank3_group1_stride2() {
        // Whisper first conv: stride 2, pad 1
        parity_check_1d(&[1, 80, 3000], &[512, 80, 3], 1, &[2], &[1], &[1, 1]);
    }

    #[test]
    fn conv_parity_rank3_depthwise() {
        // 1D depthwise: groups == in_channels
        parity_check_1d(&[1, 32, 64], &[32, 1, 3], 32, &[1], &[1], &[1, 1]);
    }

    #[test]
    fn conv_parity_rank3_grouped() {
        // 1D grouped: groups=4, ic=16, oc=32
        parity_check_1d(&[1, 16, 32], &[32, 4, 3], 4, &[1], &[1], &[1, 1]);
    }

    #[test]
    fn conv_parity_rank3_dilated() {
        // 1D dilated, no groups
        parity_check_1d(&[1, 16, 64], &[32, 16, 3], 1, &[1], &[2], &[2, 2]);
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn conv_parity_bnns_vs_scalar_ref() {
        let x_shape = [1usize, 64, 56, 56];
        let w_shape = [64usize, 64, 3, 3];
        let strides = [1usize, 1];
        let pads = [1usize, 1, 1, 1];
        let x_count: usize = x_shape.iter().product();
        let w_count: usize = w_shape.iter().product();
        let x: Vec<f32> = (0..x_count)
            .map(|i| ((i.wrapping_mul(37) % 257) as f32 - 128.0) / 128.0)
            .collect();
        let w: Vec<f32> = (0..w_count)
            .map(|i| ((i.wrapping_mul(53) % 131) as f32 - 65.0) / 65.0)
            .collect();
        let bias: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let output_shape = [1usize, 64, 56, 56];
        let reference = super::scalar_ref_execute(
            &x,
            &w,
            Some(&bias),
            &x_shape,
            &w_shape,
            &output_shape,
            1,
            &strides,
            &[1, 1],
            &pads,
            false,
        );
        let bnns_result = super::bnns::bnns_conv_execute(
            &x,
            &w,
            Some(&bias),
            &x_shape,
            &w_shape,
            &output_shape,
            &strides,
            &pads,
            1,
            false,
        );
        let bnns_out = bnns_result.expect("BNNS should accept this config");
        assert_eq!(reference.len(), bnns_out.len());
        let max_diff = reference
            .iter()
            .zip(bnns_out.iter())
            .map(|(r, b)| (r - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "BNNS vs scalar ref max_diff={max_diff}");
    }

    /// A/B micro-benchmark: BNNS Conv vs im2col+GEMM for 1×1 pointwise.
    /// Run with: `cargo test -p onnx-runtime-ep-cpu --release -- bench_pointwise_bnns_vs_gemm --ignored --nocapture`
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    #[ignore]
    fn bench_pointwise_bnns_vs_gemm() {
        use std::time::Instant;

        let shapes: &[(usize, usize, usize, usize, &str)] = &[
            // Bracket the crossover: 14×14=196, 16×16=256, 20×20=400,
            // 24×24=576, 28×28=784, 32×32=1024, 56×56=3136
            (96, 576, 14, 14, "late_expand_14"),
            (96, 576, 16, 16, "late_expand_16"),
            (96, 576, 20, 20, "late_expand_20"),
            (96, 576, 24, 24, "late_expand_24"),
            (96, 576, 28, 28, "late_expand_28"),
            (96, 576, 32, 32, "late_expand_32"),
            // Vary channel dims at key spatial sizes to check generality
            (24, 144, 14, 14, "small_ch_14"),
            (24, 144, 20, 20, "small_ch_20"),
            (24, 144, 28, 28, "small_ch_28"),
            (320, 1280, 7, 7, "final_conv_7"),
            (1024, 256, 14, 14, "resnet_late_14"),
            (512, 128, 20, 20, "resnet_mid_20"),
            (512, 128, 28, 28, "resnet_mid_28"),
            (64, 256, 56, 56, "resnet_bottleneck_56"),
            (2048, 512, 7, 7, "resnet_final_7"),
        ];
        let warmup = 10;
        let iters = 100;

        eprintln!(
            "\n{:<25} {:>10} {:>10} {:>10} {:>12} {:>7} {:>6}",
            "shape", "BNNS_µs", "GEMM_µs", "N=h*w", "FLOPs", "ratio", "win"
        );
        eprintln!("{}", "-".repeat(85));
        for &(ic, oc, h, w, name) in shapes {
            let x_shape = [1usize, ic, h, w];
            let w_shape = [oc, ic, 1, 1];
            let out_shape = [1usize, oc, h, w];
            let strides = [1usize, 1];
            let pads = [0usize, 0, 0, 0];
            let x: Vec<f32> = (0..ic * h * w)
                .map(|i| ((i.wrapping_mul(37) % 257) as f32 - 128.0) / 128.0)
                .collect();
            let weights: Vec<f32> = (0..oc * ic)
                .map(|i| ((i.wrapping_mul(53) % 131) as f32 - 65.0) / 65.0)
                .collect();
            let bias: Vec<f32> = (0..oc).map(|i| i as f32 * 0.01).collect();

            // Warmup both
            for _ in 0..warmup {
                let _ = super::bnns::bnns_conv_execute(
                    &x,
                    &weights,
                    Some(&bias),
                    &x_shape,
                    &w_shape,
                    &out_shape,
                    &strides,
                    &pads,
                    1,
                    false,
                );
                let _ = super::im2col_gemm_execute(
                    &x,
                    &weights,
                    Some(&bias),
                    &x_shape,
                    &w_shape,
                    &out_shape,
                    &strides,
                    &[1, 1],
                    &pads,
                    false,
                );
            }

            // Interleaved measurement
            let mut bnns_us = Vec::with_capacity(iters);
            let mut gemm_us = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let _ = super::bnns::bnns_conv_execute(
                    &x,
                    &weights,
                    Some(&bias),
                    &x_shape,
                    &w_shape,
                    &out_shape,
                    &strides,
                    &pads,
                    1,
                    false,
                );
                bnns_us.push(t0.elapsed().as_micros() as f64);

                let t1 = Instant::now();
                let _ = super::im2col_gemm_execute(
                    &x,
                    &weights,
                    Some(&bias),
                    &x_shape,
                    &w_shape,
                    &out_shape,
                    &strides,
                    &[1, 1],
                    &pads,
                    false,
                );
                gemm_us.push(t1.elapsed().as_micros() as f64);
            }
            bnns_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            gemm_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let bnns_med = bnns_us[iters / 2];
            let gemm_med = gemm_us[iters / 2];
            let ratio = bnns_med / gemm_med;
            let flops = 2.0 * oc as f64 * ic as f64 * (h * w) as f64;
            let winner = if ratio > 1.0 { "GEMM" } else { "BNNS" };
            eprintln!(
                "{name:<25} {bnns_med:>10.0} {gemm_med:>10.0} {:>10} {:>12.0} {ratio:>7.2} {winner:>6}",
                h * w,
                flops,
            );
        }

        // Measure BNNS per-call setup overhead in isolation:
        // Compare a tiny-compute shape (1×1 @ 1×1, N=1) where compute ≈ 0
        // to estimate the fixed cost of filter create/destroy.
        {
            let x_shape = [1usize, 4, 2, 2];
            let w_shape = [4usize, 4, 1, 1];
            let out_shape = [1usize, 4, 2, 2];
            let strides = [1usize, 1];
            let pads = [0usize, 0, 0, 0];
            let x = vec![1.0f32; 16];
            let w = vec![0.01f32; 16];
            let bias = vec![0.0f32; 4];
            for _ in 0..warmup {
                let _ = super::bnns::bnns_conv_execute(
                    &x,
                    &w,
                    Some(&bias),
                    &x_shape,
                    &w_shape,
                    &out_shape,
                    &strides,
                    &pads,
                    1,
                    false,
                );
                let _ = super::im2col_gemm_execute(
                    &x,
                    &w,
                    Some(&bias),
                    &x_shape,
                    &w_shape,
                    &out_shape,
                    &strides,
                    &[1, 1],
                    &pads,
                    false,
                );
            }
            let mut bnns_us = Vec::with_capacity(iters);
            let mut gemm_us = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let _ = super::bnns::bnns_conv_execute(
                    &x,
                    &w,
                    Some(&bias),
                    &x_shape,
                    &w_shape,
                    &out_shape,
                    &strides,
                    &pads,
                    1,
                    false,
                );
                bnns_us.push(t0.elapsed().as_micros() as f64);
                let t1 = Instant::now();
                let _ = super::im2col_gemm_execute(
                    &x,
                    &w,
                    Some(&bias),
                    &x_shape,
                    &w_shape,
                    &out_shape,
                    &strides,
                    &[1, 1],
                    &pads,
                    false,
                );
                gemm_us.push(t1.elapsed().as_micros() as f64);
            }
            bnns_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            gemm_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            eprintln!(
                "\nBNNS fixed setup cost (tiny 4×4@2×2 shape, compute≈0): median={:.0}µs",
                bnns_us[iters / 2],
            );
            eprintln!(
                "GEMM fixed cost (same tiny shape): median={:.0}µs",
                gemm_us[iters / 2],
            );
            eprintln!(
                "BNNS overhead delta ≈ {:.0}µs per call",
                bnns_us[iters / 2] - gemm_us[iters / 2],
            );
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn conv_bnns_relu_fusion() {
        let x_shape = [1, 3, 4, 4];
        let w_shape = [2, 3, 3, 3];
        let x: Vec<f32> = (0..48).map(|i| (i as f32 - 24.0) / 12.0).collect();
        let w: Vec<f32> = (0..54).map(|i| ((i % 7) as f32 - 3.0) / 3.0).collect();
        let bias = vec![-5.0, 5.0];
        let mut node = Node::new(NodeId(0), OP, vec![], vec![]);
        node.attributes
            .insert("activation".into(), Attribute::String(b"Relu".to_vec()));
        let kernel = ConvFactory
            .create(&node, &[x_shape.to_vec(), w_shape.to_vec()])
            .unwrap();
        let x_owned = Owned::f32(&x_shape, &x);
        let w_owned = Owned::f32(&w_shape, &w);
        let bias_owned = Owned::f32(&[2], &bias);
        let mut output = Owned::zeros_f32(&[1, 2, 2, 2]);
        kernel
            .execute(
                &[x_owned.view(), w_owned.view(), bias_owned.view()],
                &mut [output.view_mut()],
            )
            .unwrap();
        let out = output.to_f32();
        assert!(
            out.iter().all(|&v| v >= 0.0),
            "fused ReLU produced negative values: {out:?}"
        );
    }
}
