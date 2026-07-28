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

        let is_rank4 = self.x_shape.len() == 4;
        let is_group1 = self.group == 1;
        let is_undilated = self.dilations.iter().all(|&d| d == 1);

        let output = if is_rank4 && is_group1 {
            // Try Tier 1 (BNNS) for undilated, symmetric-padding convolutions.
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            if is_undilated
                && let Some(result) = bnns::bnns_conv_execute(
                    &x,
                    &weights,
                    bias.as_deref(),
                    &self.x_shape,
                    &self.w_shape,
                    &self.output_shape,
                    &self.strides,
                    &self.pads,
                    self.relu,
                )
            {
                CONV_BNNS_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                write_dense_f32_narrow(OP, &mut outputs[0], &result)?;
                record_conv_metrics(inputs, outputs, self);
                return Ok(());
            }

            // Tier 2: im2col + GEMM.
            CONV_IM2COL_GEMM_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            im2col_gemm_execute(
                &x,
                &weights,
                bias.as_deref(),
                &self.x_shape,
                &self.w_shape,
                &self.output_shape,
                &self.strides,
                &self.dilations,
                &self.pads,
                self.relu,
            )?
        } else {
            // Tier 3: scalar reference for rank-3, grouped, or other cases.
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
        fn BNNSFilterApplyBatch(
            filter: BNNSFilter,
            batch_size: usize,
            input: *const c_void,
            input_stride: usize,
            output: *mut c_void,
            output_stride: usize,
        ) -> i32;
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
        relu: bool,
    ) -> Option<Vec<f32>> {
        // Require symmetric padding.
        if pads[0] != pads[2] || pads[1] != pads[3] {
            return None;
        }
        let (batch, ic, ih, iw) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
        let (oc, kh, kw) = (w_shape[0], w_shape[2], w_shape[3]);
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
                &[kw, kh, ic, oc],
                &[1, kw, kw * kh, kw * kh * ic],
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
            groups: 1,
            pad: [0; 4],
        };
        let filter = unsafe { BNNSFilterCreateLayerConvolution(&params, std::ptr::null()) };
        if filter.is_null() {
            return None;
        }
        let mut output = vec![0.0f32; batch * out_stride];
        let rc = unsafe {
            BNNSFilterApplyBatch(
                filter,
                batch,
                x.as_ptr() as *const c_void,
                in_stride * 4,
                output.as_mut_ptr() as *mut c_void,
                out_stride * 4,
            )
        };
        unsafe {
            BNNSFilterDestroy(filter);
        }
        if rc == 0 { Some(output) } else { None }
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

// ═══════════════════════════════════════════════════════════════════════════════
// Tier 3: Scalar reference
// ═══════════════════════════════════════════════════════════════════════════════

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
    fn scalar_ref_handles_grouped_conv() {
        let before = super::CONV_SCALAR_REF_TEST_HITS.load(Ordering::Relaxed);
        let _ = run(
            &[1, 2, 2, 2],
            &[1., 2., 3., 4., 10., 20., 30., 40.],
            &[4, 1, 1, 1],
            &[1., 2., 3., 4.],
            Some(&[0., 1., 2., 3.]),
            &[1, 4, 2, 2],
            &[("group", Attribute::Int(2))],
        );
        let after = super::CONV_SCALAR_REF_TEST_HITS.load(Ordering::Relaxed);
        assert!(
            after > before,
            "Grouped conv did not reach scalar reference path"
        );
    }

    #[test]
    fn scalar_ref_handles_1d_conv() {
        let before = super::CONV_SCALAR_REF_TEST_HITS.load(Ordering::Relaxed);
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
        let after = super::CONV_SCALAR_REF_TEST_HITS.load(Ordering::Relaxed);
        assert!(after > before, "1D conv did not reach scalar ref path");
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
