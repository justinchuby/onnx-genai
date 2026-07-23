//! Graph-level NCHWc (channels-blocked) kernels used by the CPU EP's
//! `CpuNchwcLayoutPropagation` pass.
//!
//! The per-op `Conv` kernel (`kernels::conv`) reorders NCHW -> NCHWc on its
//! input and NCHWc -> NCHW on its output for *every* convolution. That per-op
//! reorder is the dominant residual overhead versus ONNX Runtime, which
//! transforms the activation to NCHWc *once* at graph entry and keeps the whole
//! backbone blocked (Conv, Pool, Add, Relu, Clip all consume/produce NCHWc),
//! reordering back only at the exit.
//!
//! The layout-propagation pass mirrors ORT's `NchwcTransformer`: it inserts a
//! single [`NchwcReorderToBlocked`] at each region entry and
//! [`NchwcReorderToNchw`] at each exit, and rewrites the interior ops to these
//! blocked kernels. Element-wise interior ops (`Add`, `Relu`, `Clip`) are left
//! as their standard kernels because they operate identically on the blocked
//! buffer (the channel axis is merely reblocked); only `Conv` and pooling need
//! dedicated blocked kernels, which live here.
//!
//! All ops are registered in the private `pkg.nxrt` domain and are only ever
//! produced by the CPU EP's own pass, never parsed from a model.

use std::borrow::Cow;
use std::sync::OnceLock;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::check_arity;
use crate::dtype::to_dense_f32_widen;
use crate::strided::numel;

/// The private domain the layout pass emits its blocked ops under.
pub const NCHWC_DOMAIN: &str = "pkg.nxrt";

pub const REORDER_TO_BLOCKED_OP: &str = "NchwcReorderToBlocked";
pub const REORDER_TO_NCHW_OP: &str = "NchwcReorderToNchw";
pub const NCHWC_CONV_OP: &str = "NchwcConv";
pub const NCHWC_MAX_POOL_OP: &str = "NchwcMaxPool";
pub const NCHWC_AVERAGE_POOL_OP: &str = "NchwcAveragePool";
pub const NCHWC_GLOBAL_AVERAGE_POOL_OP: &str = "NchwcGlobalAveragePool";

#[inline]
pub fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

fn attr_int(node: &Node, name: &str) -> Result<i64> {
    node.attr(name)
        .and_then(Attribute::as_int)
        .ok_or_else(|| EpError::KernelFailed(format!("{}: missing attribute {name}", node.op_type)))
}

fn attr_ints(node: &Node, name: &str, len: usize) -> Result<Vec<i64>> {
    let values = node
        .attr(name)
        .and_then(Attribute::as_ints)
        .ok_or_else(|| {
            EpError::KernelFailed(format!("{}: missing attribute {name}", node.op_type))
        })?;
    if values.len() != len {
        return Err(EpError::KernelFailed(format!(
            "{}: attribute {name} must have {len} values, got {values:?}",
            node.op_type
        )));
    }
    Ok(values.to_vec())
}

fn check_f32_io(op: &str, inputs: &[TensorView], outputs: &[TensorMut]) -> Result<()> {
    if inputs.iter().any(|t| t.dtype != DataType::Float32)
        || outputs.iter().any(|t| t.dtype != DataType::Float32)
    {
        return Err(EpError::KernelFailed(format!(
            "{op}: blocked kernels require Float32 tensors"
        )));
    }
    if !outputs[0].is_contiguous() {
        return Err(EpError::KernelFailed(format!(
            "{op}: output must be contiguous"
        )));
    }
    Ok(())
}

/// Borrow the contiguous Float32 output as a mutable slice sized by its shape.
///
/// # Safety
/// The executor guarantees the output view is a contiguous Float32 tensor whose
/// element count is the product of its checked shape.
fn output_slice<'a>(op: &str, output: &'a mut TensorMut<'_>) -> Result<&'a mut [f32]> {
    let elements = numel(output.shape);
    if !output.is_contiguous() {
        return Err(EpError::KernelFailed(format!(
            "{op}: output must be contiguous"
        )));
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(output.data_ptr_mut::<f32>(), elements) })
}

// ---- Reorder kernels ---------------------------------------------------------

/// `NchwcReorderToBlocked`: dense NCHW `[N, C, H, W]` -> blocked NCHWc
/// `[N, round_up(C, block), H, W]`. Inserted at every region entry boundary.
pub struct NchwcReorderToBlockedFactory;

struct ReorderToBlockedKernel {
    block: usize,
}

impl KernelFactory for NchwcReorderToBlockedFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ReorderToBlockedKernel {
            block: mlas_sys::nchwc_block_size(),
        }))
    }
}

impl Kernel for ReorderToBlockedKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(REORDER_TO_BLOCKED_OP, inputs, outputs, 1, 1, 1)?;
        check_f32_io(REORDER_TO_BLOCKED_OP, inputs, outputs)?;
        let in_shape = &inputs[0].shape;
        if in_shape.len() != 4 {
            return Err(EpError::KernelFailed(format!(
                "{REORDER_TO_BLOCKED_OP}: expected a 4-D NCHW input, got {in_shape:?}"
            )));
        }
        let (n, channels, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        if !channels.is_multiple_of(4) {
            return Err(EpError::KernelFailed(format!(
                "{REORDER_TO_BLOCKED_OP}: channel count {channels} must be a multiple of 4"
            )));
        }
        let blocked_channels = round_up(channels, self.block);
        let plane = h * w;
        let x = to_dense_f32_widen(REORDER_TO_BLOCKED_OP, &inputs[0])?;
        let out = output_slice(REORDER_TO_BLOCKED_OP, &mut outputs[0])?;
        if out.len() != n * blocked_channels * plane {
            return Err(EpError::KernelFailed(format!(
                "{REORDER_TO_BLOCKED_OP}: output holds {} elements, expected {}",
                out.len(),
                n * blocked_channels * plane
            )));
        }
        out.fill(0.0);
        for image in 0..n {
            let src = &x[image * channels * plane..(image + 1) * channels * plane];
            let dst =
                &mut out[image * blocked_channels * plane..(image + 1) * blocked_channels * plane];
            mlas_sys::nchwc_reorder_input_nchw(src, dst, channels, plane);
        }
        Ok(())
    }
}

/// `NchwcReorderToNchw`: blocked NCHWc `[N, round_up(C, block), H, W]` -> dense
/// NCHW `[N, C, H, W]`, keeping only the `C` logical channels. Inserted at every
/// region exit boundary.
pub struct NchwcReorderToNchwFactory;

struct ReorderToNchwKernel;

impl KernelFactory for NchwcReorderToNchwFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ReorderToNchwKernel))
    }
}

impl Kernel for ReorderToNchwKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(REORDER_TO_NCHW_OP, inputs, outputs, 1, 1, 1)?;
        check_f32_io(REORDER_TO_NCHW_OP, inputs, outputs)?;
        let out_shape = outputs[0].shape.to_vec();
        if out_shape.len() != 4 {
            return Err(EpError::KernelFailed(format!(
                "{REORDER_TO_NCHW_OP}: expected a 4-D NCHW output, got {out_shape:?}"
            )));
        }
        let (n, channels, h, w) = (out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
        let x = to_dense_f32_widen(REORDER_TO_NCHW_OP, &inputs[0])?;
        let out = output_slice(REORDER_TO_NCHW_OP, &mut outputs[0])?;
        mlas_sys::nchwc_reorder_output_nchw(
            &[n as i64, channels as i64, h as i64, w as i64],
            &x,
            out,
        );
        Ok(())
    }
}

// ---- Blocked convolution -----------------------------------------------------

fn parse_activation(node: &Node) -> mlas_sys::NchwcActivation {
    match node.attr("activation").and_then(Attribute::as_str) {
        Some("Relu") => mlas_sys::NchwcActivation::RELU,
        _ => mlas_sys::NchwcActivation::IDENTITY,
    }
}

/// `NchwcConv`: blocked-in / blocked-out convolution with no per-op activation
/// reorder. Consumes NCHWc input (or plain NCHW for the 3-channel first layer)
/// and writes NCHWc output directly. Emitted only by the layout pass, which
/// already validated the shape is one of the blocked-eligible algorithms.
pub struct NchwcConvFactory;

struct NchwcConvKernel {
    block: usize,
    group_count: usize,
    /// True when the input is already NCHWc-blocked; false for the first-layer
    /// NCHW-input algorithm.
    input_blocked: bool,
    /// True selects the `OIHWBiBo` filter layout, false selects `OIHWBo`.
    filter_bibo: bool,
    batch: usize,
    in_channels: usize,
    input_h: usize,
    input_w: usize,
    output_h: usize,
    output_w: usize,
    kernel: [i64; 2],
    dilation: [i64; 2],
    pads: [i64; 4],
    stride: [i64; 2],
    weight_shape: [i64; 4],
    conv_input_channels: usize,
    nchwc_out_channels: usize,
    out_channels: usize,
    activation: mlas_sys::NchwcActivation,
    weight_constant: bool,
    bias_constant: bool,
    packed_filter: OnceLock<Vec<f32>>,
    packed_bias: OnceLock<Vec<f32>>,
}

impl KernelFactory for NchwcConvFactory {
    fn create(&self, node: &Node, shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let block = mlas_sys::nchwc_block_size();
        if block < 8 {
            return Err(EpError::KernelFailed(
                "NchwcConv: host has no NCHWc blocked-convolution kernel".into(),
            ));
        }
        let x_shape = shapes
            .first()
            .ok_or_else(|| EpError::KernelFailed("NchwcConv: missing X shape".into()))?;
        if x_shape.len() != 4 {
            return Err(EpError::KernelFailed(format!(
                "NchwcConv: expected 4-D input, got {x_shape:?}"
            )));
        }
        let kernel = attr_ints(node, "kernel_shape", 2)?;
        let strides = attr_ints(node, "strides", 2)?;
        let dilations = attr_ints(node, "dilations", 2)?;
        let pads = attr_ints(node, "pads", 4)?;
        let group_count = attr_int(node, "group_count")? as usize;
        let in_channels = attr_int(node, "in_channels")? as usize;
        let out_channels = attr_int(node, "out_channels")? as usize;
        let output_h = attr_int(node, "output_h")? as usize;
        let output_w = attr_int(node, "output_w")? as usize;
        let input_blocked = attr_int(node, "input_blocked")? != 0;
        let filter_bibo = attr_int(node, "filter_bibo")? != 0;
        let w_shape = node
            .attr("weight_shape")
            .and_then(Attribute::as_ints)
            .ok_or_else(|| EpError::KernelFailed("NchwcConv: missing weight_shape".into()))?;
        if w_shape.len() != 4 {
            return Err(EpError::KernelFailed(format!(
                "NchwcConv: weight_shape must be 4-D, got {w_shape:?}"
            )));
        }

        let nchwc_out_channels = round_up(out_channels, block);
        let conv_input_channels = if input_blocked {
            round_up(in_channels, block)
        } else {
            in_channels
        };

        Ok(Box::new(NchwcConvKernel {
            block,
            group_count,
            input_blocked,
            filter_bibo,
            batch: x_shape[0],
            in_channels,
            input_h: x_shape[2],
            input_w: x_shape[3],
            output_h,
            output_w,
            kernel: [kernel[0], kernel[1]],
            dilation: [dilations[0], dilations[1]],
            pads: [pads[0], pads[1], pads[2], pads[3]],
            stride: [strides[0], strides[1]],
            weight_shape: [w_shape[0], w_shape[1], w_shape[2], w_shape[3]],
            conv_input_channels,
            nchwc_out_channels,
            out_channels,
            activation: parse_activation(node),
            weight_constant: false,
            bias_constant: false,
            packed_filter: OnceLock::new(),
            packed_bias: OnceLock::new(),
        }))
    }
}

impl NchwcConvKernel {
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

    fn filter<'a>(&'a self, weight: &'a TensorView<'_>) -> Result<Cow<'a, [f32]>> {
        if self.weight_constant {
            if let Some(packed) = self.packed_filter.get() {
                return Ok(Cow::Borrowed(packed));
            }
            let weights = to_dense_f32_widen("NchwcConv", weight)?;
            let packed = self.reorder_filter(&weights);
            let _ = self.packed_filter.set(packed);
            return Ok(Cow::Borrowed(
                self.packed_filter
                    .get()
                    .expect("NchwcConv: packed filter just initialized"),
            ));
        }
        let weights = to_dense_f32_widen("NchwcConv", weight)?;
        Ok(Cow::Owned(self.reorder_filter(&weights)))
    }

    fn pad_bias(&self, bias: &[f32]) -> Vec<f32> {
        let mut padded = vec![0.0f32; self.nchwc_out_channels];
        padded[..bias.len()].copy_from_slice(bias);
        padded
    }

    fn bias<'a>(&'a self, bias: Option<&'a TensorView<'_>>) -> Result<Option<Cow<'a, [f32]>>> {
        let Some(bias) = bias else {
            return Ok(None);
        };
        if self.bias_constant {
            if let Some(padded) = self.packed_bias.get() {
                return Ok(Some(Cow::Borrowed(padded)));
            }
            let dense = to_dense_f32_widen("NchwcConv", bias)?;
            let _ = self.packed_bias.set(self.pad_bias(&dense));
            return Ok(Some(Cow::Borrowed(
                self.packed_bias
                    .get()
                    .expect("NchwcConv: padded bias just initialized"),
            )));
        }
        let dense = to_dense_f32_widen("NchwcConv", bias)?;
        Ok(Some(Cow::Owned(self.pad_bias(&dense))))
    }
}

impl Kernel for NchwcConvKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(NCHWC_CONV_OP, inputs, outputs, 2, 3, 1)?;
        check_f32_io(NCHWC_CONV_OP, inputs, outputs)?;

        let x = to_dense_f32_widen(NCHWC_CONV_OP, &inputs[0])?;
        let filter = self.filter(&inputs[1])?;
        let bias = self.bias(inputs.get(2))?;

        let expected_out = self.batch * self.nchwc_out_channels * self.output_h * self.output_w;
        let out = output_slice(NCHWC_CONV_OP, &mut outputs[0])?;
        if out.len() != expected_out {
            return Err(EpError::KernelFailed(format!(
                "{NCHWC_CONV_OP}: output holds {} elements, expected blocked {expected_out}",
                out.len()
            )));
        }
        out.fill(0.0);

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
            &x,
            &filter,
            bias.as_deref(),
            out,
            self.activation,
            true,
        );

        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let output_spatial = self.output_h.saturating_mul(self.output_w);
            let kernel_elements =
                (self.weight_shape[1] * self.weight_shape[2] * self.weight_shape[3]) as usize;
            (self.batch as u64)
                .saturating_mul(self.out_channels as u64)
                .saturating_mul(output_spatial as u64)
                .saturating_mul(kernel_elements as u64)
                .saturating_mul(2)
        });
        // Silence unused-field lint on the debug-only helper.
        let _ = (self.in_channels, self.input_blocked);
        Ok(())
    }

    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        self.weight_constant = constant_inputs.get(1).copied().unwrap_or(false);
        self.bias_constant = constant_inputs.get(2).copied().unwrap_or(false);
    }
}

// ---- Blocked pooling ---------------------------------------------------------

#[derive(Clone, Copy)]
enum NchwcPoolKind {
    Max,
    Average { include_pad: bool },
    GlobalAverage,
}

/// `NchwcMaxPool` / `NchwcAveragePool` / `NchwcGlobalAveragePool`: 2-D pooling on
/// blocked NCHWc buffers via `MlasNchwcPool` (channel-independent, so the
/// activation stays blocked across the pool with no reorder).
pub struct NchwcPoolFactory {
    kind: NchwcPoolKind,
}

impl NchwcPoolFactory {
    pub fn max() -> Self {
        Self {
            kind: NchwcPoolKind::Max,
        }
    }
    pub fn average() -> Self {
        Self {
            kind: NchwcPoolKind::Average { include_pad: false },
        }
    }
    pub fn global_average() -> Self {
        Self {
            kind: NchwcPoolKind::GlobalAverage,
        }
    }
}

struct NchwcPoolKernel {
    kind: NchwcPoolKind,
    op: &'static str,
    kernel: [i64; 2],
    strides: [i64; 2],
    pads: [i64; 4],
    dilations: [i64; 2],
}

impl KernelFactory for NchwcPoolFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let (op, kernel, strides, pads, dilations) = match self.kind {
            NchwcPoolKind::GlobalAverage => (
                NCHWC_GLOBAL_AVERAGE_POOL_OP,
                [0, 0],
                [1, 1],
                [0, 0, 0, 0],
                [1, 1],
            ),
            NchwcPoolKind::Max | NchwcPoolKind::Average { .. } => {
                let op = if matches!(self.kind, NchwcPoolKind::Max) {
                    NCHWC_MAX_POOL_OP
                } else {
                    NCHWC_AVERAGE_POOL_OP
                };
                let kernel = attr_ints(node, "kernel_shape", 2)?;
                let strides = attr_ints(node, "strides", 2)?;
                let pads = attr_ints(node, "pads", 4)?;
                let dilations = node
                    .attr("dilations")
                    .and_then(Attribute::as_ints)
                    .map(<[i64]>::to_vec)
                    .unwrap_or_else(|| vec![1, 1]);
                if dilations.len() != 2 {
                    return Err(EpError::KernelFailed(format!(
                        "{op}: dilations must have 2 values"
                    )));
                }
                (
                    op,
                    [kernel[0], kernel[1]],
                    [strides[0], strides[1]],
                    [pads[0], pads[1], pads[2], pads[3]],
                    [dilations[0], dilations[1]],
                )
            }
        };
        let kind = match self.kind {
            NchwcPoolKind::Average { .. } => NchwcPoolKind::Average {
                include_pad: node
                    .attr("count_include_pad")
                    .and_then(Attribute::as_int)
                    .unwrap_or(0)
                    != 0,
            },
            other => other,
        };
        Ok(Box::new(NchwcPoolKernel {
            kind,
            op,
            kernel,
            strides,
            pads,
            dilations,
        }))
    }
}

impl Kernel for NchwcPoolKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(self.op, inputs, outputs, 1, 1, 1)?;
        check_f32_io(self.op, inputs, outputs)?;
        let in_shape = inputs[0].shape.to_vec();
        let out_shape = outputs[0].shape.to_vec();
        if in_shape.len() != 4 || out_shape.len() != 4 {
            return Err(EpError::KernelFailed(format!(
                "{}: expected 4-D blocked tensors, got {in_shape:?} -> {out_shape:?}",
                self.op
            )));
        }
        let (n, cb, hin, win) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let (hout, wout) = (out_shape[2], out_shape[3]);

        let (mlas_kind, kernel, strides, pads) = match self.kind {
            NchwcPoolKind::Max => (
                mlas_sys::PoolKind::Maximum,
                self.kernel,
                self.strides,
                self.pads,
            ),
            NchwcPoolKind::Average { include_pad } => {
                let kind = if include_pad {
                    mlas_sys::PoolKind::AverageIncludePad
                } else {
                    mlas_sys::PoolKind::AverageExcludePad
                };
                (kind, self.kernel, self.strides, self.pads)
            }
            NchwcPoolKind::GlobalAverage => (
                mlas_sys::PoolKind::AverageIncludePad,
                [hin as i64, win as i64],
                [1, 1],
                [0, 0, 0, 0],
            ),
        };

        let x = to_dense_f32_widen(self.op, &inputs[0])?;
        let out = output_slice(self.op, &mut outputs[0])?;
        if out.len() != n * cb * hout * wout {
            return Err(EpError::KernelFailed(format!(
                "{}: output holds {} elements, expected {}",
                self.op,
                out.len(),
                n * cb * hout * wout
            )));
        }
        mlas_sys::nchwc_pool(
            mlas_kind,
            &[n as i64, cb as i64, hin as i64, win as i64],
            &kernel,
            &self.dilations,
            &pads,
            &strides,
            &[n as i64, cb as i64, hout as i64, wout as i64],
            &x,
            out,
        );
        Ok(())
    }
}
