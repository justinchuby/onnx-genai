//! CPU parity oracle for `pkg.nxrt::BlockQuantizedMoE` (mixed-projection ABI).
//!
//! Each routed expert carries an independent native block-quantized format for
//! every projection: `fc1_format` (gate/up), `fc2_format` (down), and
//! `fc3_format` (the separate gate of an unfused SwiGLU / gated-GLU). This is a
//! hard requirement of real GLM-5.2 GGUF checkpoints, whose routed experts pack
//! `ffn_gate/up_exps` and `ffn_down_exps` at *different* qtypes and block widths
//! (e.g. `IQ1_S` gate/up with `IQ3_XXS` down). A single uniform format cannot
//! represent them.
//!
//! Every consumer — the schema/claim validator, this CPU oracle, and the CUDA
//! claim gate — derives its byte offsets, strides and decode parameters from one
//! property-typed [`ProjectionLayout`] contract (qtype, elements/block,
//! bytes/block, logical K/N, per-row and per-expert bank strides, expert count).
//! The GLM-5.2 IQ/K-quant/Q8 layouts are decoded by the CPU parity oracle.
//!
//! This CPU kernel is memory-format-only: it keeps expert weights in the native
//! block-quantized wire layout at the operator boundary, then dequantizes each
//! routed expert projection to dense f32 with that projection's own decoder and
//! runs the dense grouped MoE path. It does not perform quantized-domain compute.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node, Shape};

use super::block_quantized_matmul::{
    BlockFormat, DenseWeightCache, DenseWeightCacheStatus, DenseWeightIdentity,
    dequantize_weight_kn,
};
use super::moe::{MoeAttributes, routing_weights, run_expert_grouped};
use super::{check_arity, to_dense_f32, write_dense_f32};

const OP: &str = "BlockQuantizedMoE";
const LAYOUT_VERSION: i64 = 1;

pub static BLOCK_QUANT_MOE_CACHED_DENSE_TEST_HITS: AtomicUsize = AtomicUsize::new(0);
pub static BLOCK_QUANT_MOE_DENSE_EXPANSIONS: AtomicUsize = AtomicUsize::new(0);

const INPUT_NAMES: [&str; 9] = [
    "input",
    "router_logits",
    "fc1_experts_weights",
    "fc1_experts_bias",
    "fc2_experts_weights",
    "fc2_experts_bias",
    "fc3_experts_weights",
    "fc3_experts_bias",
    "router_weights",
];

#[cfg(test)]
static BLOCK_QUANTIZED_MOE_DENSE_F32_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Per-projection native block formats for one `BlockQuantizedMoE` node.
///
/// `fc1` (gate/up) and `fc2` (down) are always present; `fc3` (the separate gate
/// of an unfused SwiGLU / gated-GLU) is present exactly when the `fc3` weights
/// input is wired. The three formats are independent — this is the whole point
/// of the mixed-projection ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProjectionFormats {
    fc1: BlockFormat,
    fc2: BlockFormat,
    fc3: Option<BlockFormat>,
}

/// One property-typed packed-projection layout contract, shared by the claim
/// validator and the CPU oracle (and mirrored by the CUDA claim gate).
///
/// A packed projection tensor has shape `[experts, out_features, blocks_per_row,
/// block_bytes]` where `blocks_per_row = in_features / qk`. Partial native
/// blocks are rejected.
/// embedded per block (the IQ/MXFP4 formats are self-describing), so there is no
/// external scale or codebook tensor. Every byte offset the kernel touches is
/// derived from this contract, never recomputed ad hoc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProjectionLayout {
    format: BlockFormat,
    in_features: usize,
    out_features: usize,
    experts: usize,
}

impl ProjectionLayout {
    fn new(format: BlockFormat, out_features: usize, in_features: usize, experts: usize) -> Self {
        Self {
            format,
            in_features,
            out_features,
            experts,
        }
    }

    fn qk(&self) -> usize {
        self.format.qk()
    }

    fn block_bytes(&self) -> usize {
        self.format.block_bytes()
    }

    fn blocks_per_row(&self) -> usize {
        self.in_features.div_ceil(self.qk())
    }

    /// Packed tensor shape `[E, N, blocks_per_row, block_bytes]`.
    fn packed_shape(&self) -> [usize; 4] {
        [
            self.experts,
            self.out_features,
            self.blocks_per_row(),
            self.block_bytes(),
        ]
    }

    /// Bank stride of one packed output row, in bytes.
    fn row_stride_bytes(&self) -> Result<usize> {
        self.blocks_per_row()
            .checked_mul(self.block_bytes())
            .ok_or_else(|| error("packed row byte stride overflow"))
    }

    /// Bank stride of one expert (all its output rows), in bytes.
    fn expert_stride_bytes(&self) -> Result<usize> {
        self.out_features
            .checked_mul(self.row_stride_bytes()?)
            .ok_or_else(|| error("packed expert byte stride overflow"))
    }

    /// Total packed bytes across every expert.
    fn total_bytes(&self) -> Result<usize> {
        self.experts
            .checked_mul(self.expert_stride_bytes()?)
            .ok_or_else(|| error("packed projection byte count overflow"))
    }

    /// Byte range `start..end` of one expert's bank within the packed tensor.
    fn expert_byte_range(&self, expert: usize) -> Result<std::ops::Range<usize>> {
        let stride = self.expert_stride_bytes()?;
        let start = expert
            .checked_mul(stride)
            .ok_or_else(|| error("expert byte offset overflow"))?;
        let end = start
            .checked_add(stride)
            .ok_or_else(|| error("expert byte range overflow"))?;
        Ok(start..end)
    }
}

pub struct BlockQuantizedMoEFactory;

pub struct BlockQuantizedMoEKernel {
    attributes: MoeAttributes,
    formats: ProjectionFormats,
    constant_inputs: [bool; 9],
    weight_identities: [DenseWeightIdentity; 3],
    weight_cache: DenseWeightCache,
}

#[derive(Debug)]
struct ValidatedMetadata {
    attributes: MoeAttributes,
    formats: ProjectionFormats,
}

impl KernelFactory for BlockQuantizedMoEFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let ValidatedMetadata {
            attributes,
            formats,
        } = validate_metadata(node, None)?;
        Ok(Box::new(BlockQuantizedMoEKernel {
            attributes,
            formats,
            constant_inputs: [false; 9],
            weight_identities: std::array::from_fn(|_| DenseWeightIdentity::default()),
            weight_cache: DenseWeightCache::new(),
        }))
    }
}

pub(crate) fn unsupported_reason(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Option<Cow<'static, str>> {
    validate_metadata(node, Some((shapes, input_dtypes)))
        .err()
        .map(|error| Cow::Owned(error.to_string()))
}

impl Kernel for BlockQuantizedMoEKernel {
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        for (index, is_constant) in constant_inputs.iter().copied().enumerate().take(9) {
            self.constant_inputs[index] = is_constant;
        }
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(OP, inputs, outputs, 5, 9, 1)?;
        for &index in &[0, 1, 2, 4] {
            if inputs[index].is_absent() {
                return Err(error(format!(
                    "required input {index} ('{}') is absent",
                    INPUT_NAMES[index]
                )));
            }
        }
        for &index in &[0, 1] {
            require_dtype(index, &inputs[index], DataType::Float32)?;
        }
        for &index in &[2, 4] {
            require_dtype(index, &inputs[index], DataType::Uint8)?;
        }
        for &index in &[3, 5, 7, 8] {
            if let Some(input) = optional_input(inputs, index) {
                require_dtype(index, input, DataType::Float32)?;
            }
        }
        if let Some(input) = optional_input(inputs, 6) {
            require_dtype(6, input, DataType::Uint8)?;
        }
        if outputs[0].dtype != DataType::Float32 {
            return Err(error(format!(
                "output dtype {:?} unsupported; expected Float32",
                outputs[0].dtype
            )));
        }

        let dimensions =
            validate_runtime_shapes(inputs, &outputs[0], &self.attributes, self.formats)?;
        let Dimensions {
            rows,
            hidden,
            experts,
            inter,
            fc1_size,
        } = dimensions;

        let input = to_dense_f32(&inputs[0])?;
        let router_logits = to_dense_f32(&inputs[1])?;
        let router_weights = optional_dense(inputs, 8)?;
        let fc1_bias = optional_dense(inputs, 3)?;
        let fc2_bias = optional_dense(inputs, 5)?;
        let fc3_bias = optional_dense(inputs, 7)?;

        let mut tasks = BTreeMap::<usize, Vec<(usize, f32)>>::new();
        for row in 0..rows {
            let range = row * experts..(row + 1) * experts;
            let mut route = routing_weights(
                &router_logits[range.clone()],
                router_weights
                    .as_deref()
                    .map(|weights| &weights[range.clone()]),
                self.attributes.k,
                self.attributes.normalize_routing_weights,
            );
            route.sort_unstable_by_key(|&(expert, _)| expert);
            for (expert, weight) in route {
                tasks.entry(expert).or_default().push((row, weight));
            }
        }

        let mut output = vec![0.0f32; rows * hidden];
        #[cfg(test)]
        BLOCK_QUANTIZED_MOE_DENSE_F32_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        for (expert, expert_tasks) in tasks {
            let fc1 = self.dequantize_expert_cached(
                self.constant_inputs[2].then_some(&self.weight_identities[0]),
                1,
                &inputs[2],
                expert,
                fc1_size,
                hidden,
                experts,
                self.formats.fc1,
            )?;
            let fc2 = self.dequantize_expert_cached(
                self.constant_inputs[4].then_some(&self.weight_identities[1]),
                2,
                &inputs[4],
                expert,
                hidden,
                inter,
                experts,
                self.formats.fc2,
            )?;
            let fc3 = optional_input(inputs, 6)
                .map(|packed| {
                    let fc3_format = self.formats.fc3.ok_or_else(|| {
                        error("fc3_experts_weights present but fc3_format is missing")
                    })?;
                    self.dequantize_expert_cached(
                        self.constant_inputs[6].then_some(&self.weight_identities[2]),
                        3,
                        packed,
                        expert,
                        inter,
                        hidden,
                        experts,
                        fc3_format,
                    )
                })
                .transpose()?;

            let mut grouped_input = Vec::with_capacity(expert_tasks.len() * hidden);
            for &(row, _) in &expert_tasks {
                grouped_input.extend_from_slice(&input[row * hidden..(row + 1) * hidden]);
            }
            let expert_output = run_expert_grouped(
                &grouped_input,
                expert_tasks.len(),
                fc1.as_slice(),
                fc1_bias
                    .as_deref()
                    .map(|bias| &bias[expert * fc1_size..(expert + 1) * fc1_size]),
                fc2.as_slice(),
                fc2_bias
                    .as_deref()
                    .map(|bias| &bias[expert * hidden..(expert + 1) * hidden]),
                fc3.as_ref().map(|weight| weight.as_slice()),
                fc3_bias
                    .as_deref()
                    .map(|bias| &bias[expert * inter..(expert + 1) * inter]),
                fc1_size,
                hidden,
                inter,
                &self.attributes,
            )?;
            for (grouped_row, (row, route_weight)) in expert_tasks.into_iter().enumerate() {
                for feature in 0..hidden {
                    output[row * hidden + feature] +=
                        route_weight * expert_output[grouped_row * hidden + feature];
                }
            }
        }
        write_dense_f32(&mut outputs[0], &output)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

impl BlockQuantizedMoEKernel {
    #[allow(clippy::too_many_arguments)]
    fn dequantize_expert_cached(
        &self,
        identity: Option<&DenseWeightIdentity>,
        role: u8,
        packed: &TensorView,
        expert: usize,
        out_features: usize,
        in_features: usize,
        experts: usize,
        format: BlockFormat,
    ) -> Result<Arc<Vec<f32>>> {
        let layout = ProjectionLayout::new(format, out_features, in_features, experts);
        // Validate the packed projection's total byte footprint against the
        // shared layout contract before slicing any expert bank.
        expert_byte_count(packed, &layout)?;
        if let Some(identity) = identity {
            let resolved = identity.resolve(
                packed,
                format,
                in_features,
                out_features,
                role,
                Some(expert),
                || packed_expert_bytes(packed, &layout, expert),
            )?;
            let mut resolved_payload = resolved.payload;
            let (weight, status) =
                self.weight_cache
                    .get_or_insert_with(resolved.key.as_ref(), || {
                        BLOCK_QUANT_MOE_DENSE_EXPANSIONS.fetch_add(1, Ordering::Relaxed);
                        let packed = match resolved_payload.take() {
                            Some(packed) => packed,
                            None => packed_expert_bytes(packed, &layout, expert)?,
                        };
                        dequantize_expert_slice(format, &packed, out_features, in_features)
                    })?;
            if matches!(status, DenseWeightCacheStatus::Hit) {
                BLOCK_QUANT_MOE_CACHED_DENSE_TEST_HITS.fetch_add(1, Ordering::Relaxed);
            }
            Ok(weight)
        } else {
            BLOCK_QUANT_MOE_DENSE_EXPANSIONS.fetch_add(1, Ordering::Relaxed);
            let packed = packed_expert_bytes(packed, &layout, expert)?;
            Ok(Arc::new(dequantize_expert_slice(
                format,
                &packed,
                out_features,
                in_features,
            )?))
        }
    }
}

#[derive(Clone, Copy)]
struct Dimensions {
    rows: usize,
    hidden: usize,
    experts: usize,
    inter: usize,
    fc1_size: usize,
}

fn validate_runtime_shapes(
    inputs: &[TensorView],
    output: &TensorMut,
    attributes: &MoeAttributes,
    formats: ProjectionFormats,
) -> Result<Dimensions> {
    let input_shape = inputs[0].shape;
    if !matches!(input_shape.len(), 2 | 3) {
        return Err(error(format!(
            "input must be [rows,H] or [B,S,H], got {input_shape:?}"
        )));
    }
    if output.shape != input_shape {
        return Err(error(format!(
            "output shape {:?} must equal input shape {input_shape:?}",
            output.shape
        )));
    }
    let hidden = *input_shape.last().expect("validated non-empty input rank");
    let rows = checked_product(&input_shape[..input_shape.len() - 1], "input rows")?;
    require_exact_rank(1, inputs[1].shape, 2)?;
    if inputs[1].shape[0] != rows {
        return Err(error(format!(
            "router_logits rows {} must equal flattened input rows {rows}",
            inputs[1].shape[0]
        )));
    }
    let experts = inputs[1].shape[1];
    if attributes.k > experts {
        return Err(error(format!(
            "requires 0 < k <= num_experts, got k={} and num_experts={experts}",
            attributes.k
        )));
    }
    require_exact_rank(2, inputs[2].shape, 4)?;
    require_exact_rank(4, inputs[4].shape, 4)?;
    if inputs[2].shape[0] != experts || inputs[4].shape[0] != experts {
        return Err(error(format!(
            "expert weight counts must equal router num_experts {experts}"
        )));
    }
    if inputs[4].shape[1] != hidden {
        return Err(error(format!(
            "fc2_experts_weights must start [experts={experts}, H={hidden}], got {:?}",
            inputs[4].shape
        )));
    }
    let fc1_size = inputs[2].shape[1];
    let inter = if attributes.swiglu_fusion == 0 {
        fc1_size
    } else {
        if !fc1_size.is_multiple_of(2) {
            return Err(error(format!(
                "fused SwiGLU fc1_out must be even, got {fc1_size}"
            )));
        }
        fc1_size / 2
    };
    if inter == 0 {
        return Err(error("inferred inter dimension must be non-zero"));
    }
    let expected_fc1 = attributes.checked_fc1_size(inter, OP)?;
    if fc1_size != expected_fc1 {
        return Err(error(format!(
            "fc1_experts_weights dimension 1 must be {expected_fc1}, got {fc1_size}"
        )));
    }
    validate_packed_shape(
        2,
        inputs[2].shape,
        ProjectionLayout::new(formats.fc1, fc1_size, hidden, experts),
    )?;
    validate_packed_shape(
        4,
        inputs[4].shape,
        ProjectionLayout::new(formats.fc2, hidden, inter, experts),
    )?;
    validate_bias(inputs, 3, experts, fc1_size)?;
    validate_bias(inputs, 5, experts, hidden)?;

    let has_fc3 = optional_input(inputs, 6).is_some();
    if has_fc3 != formats.fc3.is_some() {
        return Err(error(
            "fc3_format must be present exactly when fc3_experts_weights is wired",
        ));
    }
    if attributes.uses_separate_gate(has_fc3) {
        let fc3 = optional_input(inputs, 6)
            .ok_or_else(|| error("unfused swiglu requires input 6 fc3_experts_weights"))?;
        let fc3_format = formats
            .fc3
            .ok_or_else(|| error("fc3_experts_weights requires the fc3_format attribute"))?;
        validate_packed_shape(
            6,
            fc3.shape,
            ProjectionLayout::new(fc3_format, inter, hidden, experts),
        )?;
        validate_bias(inputs, 7, experts, inter)?;
    } else {
        if has_fc3 {
            return Err(error(
                "fc3_experts_weights is only valid for unfused swiglu or silu gated-GLU",
            ));
        }
        if optional_input(inputs, 7).is_some() {
            return Err(error("fc3_experts_bias requires fc3_experts_weights"));
        }
    }
    if let Some(weights) = optional_input(inputs, 8)
        && weights.shape != [rows, experts]
    {
        return Err(error(format!(
            "router_weights must have shape [{rows}, {experts}], got {:?}",
            weights.shape
        )));
    }
    Ok(Dimensions {
        rows,
        hidden,
        experts,
        inter,
        fc1_size,
    })
}

fn validate_packed_shape(index: usize, shape: &[usize], layout: ProjectionLayout) -> Result<()> {
    if !layout.in_features.is_multiple_of(layout.qk()) {
        return Err(error(format!(
            "input {index} ('{}') has input width {} with a partial {:?} block tail; \
             full blocks of {} elements are required",
            INPUT_NAMES[index],
            layout.in_features,
            layout.format,
            layout.qk()
        )));
    }
    let expected = layout.packed_shape();
    if shape != expected {
        return Err(error(format!(
            "input {index} ('{}') must have shape {expected:?}, got {shape:?}",
            INPUT_NAMES[index]
        )));
    }
    Ok(())
}

fn expert_byte_count(packed: &TensorView, layout: &ProjectionLayout) -> Result<usize> {
    let expert_bytes = layout.expert_stride_bytes()?;
    let expected = layout.total_bytes()?;
    if packed.byte_size() != expected {
        return Err(error(format!(
            "packed projection contains {} bytes, expected {expected}",
            packed.byte_size()
        )));
    }
    Ok(expert_bytes)
}

fn packed_expert_bytes<'a>(
    packed: &'a TensorView<'_>,
    layout: &ProjectionLayout,
    expert: usize,
) -> Result<Cow<'a, [u8]>> {
    packed.validate()?;
    if packed.dtype != DataType::Uint8 {
        return Err(error(format!(
            "packed expert dtype {:?} unsupported; expected Uint8",
            packed.dtype
        )));
    }
    if expert >= packed.shape[0] {
        return Err(error(format!(
            "expert index {expert} is out of range for {} experts",
            packed.shape[0]
        )));
    }
    // The bank offset/stride is derived from the one shared layout contract.
    let range = layout.expert_byte_range(expert)?;
    let expert_bytes = range.len();
    if packed.is_contiguous() {
        let len = packed.byte_size();
        // SAFETY: the validated contiguous Uint8 view has `len` logical bytes,
        // and `range` was derived from the already-validated expert-major shape.
        // The returned borrow cannot outlive the input view.
        let bytes = unsafe { std::slice::from_raw_parts(packed.data_ptr::<u8>(), len) };
        return bytes
            .get(range)
            .map(Cow::Borrowed)
            .ok_or_else(|| error("expert byte range exceeds packed projection"));
    }

    if packed.shape.len() != 4 {
        return Err(error(format!(
            "packed expert tensor must have rank 4, got {:?}",
            packed.shape
        )));
    }
    let mut dense = Vec::with_capacity(expert_bytes);
    let origin = packed.data_ptr::<u8>();
    for output in 0..packed.shape[1] {
        for block in 0..packed.shape[2] {
            for byte in 0..packed.shape[3] {
                let index = [expert, output, block, byte];
                let offset = crate::strided::elem_offset(packed.strides, &index);
                // SAFETY: every component of `index` is within the validated
                // view shape, so the executor's backing-bounds gate guarantees
                // this element address is readable.
                dense.push(unsafe { *origin.offset(offset) });
            }
        }
    }
    if dense.len() != expert_bytes {
        return Err(error(format!(
            "materialized expert contains {} bytes, expected {expert_bytes}",
            dense.len()
        )));
    }
    Ok(Cow::Owned(dense))
}

fn dequantize_expert_slice(
    format: BlockFormat,
    packed: &[u8],
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>> {
    let weight_kn = dequantize_weight_kn(format, in_features, out_features, packed)?;
    let mut weight_nk = vec![0.0f32; weight_kn.len()];
    for input in 0..in_features {
        for output in 0..out_features {
            weight_nk[output * in_features + input] = weight_kn[input * out_features + output];
        }
    }
    Ok(weight_nk)
}

/// Decode one expert projection with the authoritative CPU GGUF decoder and
/// return row-major `[out_features, in_features]` values widened to f64.
///
/// This is an oracle/testing seam; production CPU execution retains its cached
/// f32 path.
#[doc(hidden)]
pub fn decode_expert_projection_f64(
    format: &str,
    packed: &[u8],
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f64>> {
    let format = BlockFormat::parse(format)?;
    dequantize_expert_slice(format, packed, out_features, in_features)
        .map(|values| values.into_iter().map(f64::from).collect())
}

fn validate_attributes(node: &Node) -> Result<()> {
    for name in node.attributes.keys() {
        if !matches!(
            name.as_str(),
            "k" | "activation_type"
                | "normalize_routing_weights"
                | "swiglu_fusion"
                | "activation_alpha"
                | "activation_beta"
                | "swiglu_limit"
                | "fc1_format"
                | "fc2_format"
                | "fc3_format"
                | "block_layout_version"
        ) {
            return Err(error(format!(
                "attribute '{name}' is not part of the BlockQuantizedMoE ABI"
            )));
        }
    }
    Ok(())
}

fn parse_format_attr(node: &Node, name: &str) -> Result<BlockFormat> {
    node.attr(name)
        .ok_or_else(|| error(format!("missing required string attribute '{name}'")))?
        .as_str()
        .ok_or_else(|| error(format!("attribute '{name}' must be a UTF-8 string")))
        .and_then(BlockFormat::parse)
}

fn optional_format_attr(node: &Node, name: &str) -> Result<Option<BlockFormat>> {
    match node.attr(name) {
        None => Ok(None),
        Some(attr) => attr
            .as_str()
            .ok_or_else(|| error(format!("attribute '{name}' must be a UTF-8 string")))
            .and_then(BlockFormat::parse)
            .map(Some),
    }
}

/// Parse the per-projection formats. `fc3_format` must be present exactly when
/// the `fc3_experts_weights` input (index 6) is wired on the node.
fn parse_projection_formats(node: &Node) -> Result<ProjectionFormats> {
    let fc1 = parse_format_attr(node, "fc1_format")?;
    let fc2 = parse_format_attr(node, "fc2_format")?;
    let fc3_wired = node.inputs.get(6).is_some_and(Option::is_some);
    let fc3_attr = optional_format_attr(node, "fc3_format")?;
    match (fc3_wired, fc3_attr) {
        (true, Some(fc3)) => Ok(ProjectionFormats {
            fc1,
            fc2,
            fc3: Some(fc3),
        }),
        (true, None) => Err(error(
            "fc3_experts_weights is wired but the required fc3_format attribute is missing",
        )),
        (false, Some(_)) => Err(error(
            "fc3_format is only valid when fc3_experts_weights is wired",
        )),
        (false, None) => Ok(ProjectionFormats {
            fc1,
            fc2,
            fc3: None,
        }),
    }
}

fn validate_metadata(
    node: &Node,
    claim_metadata: Option<(&[Shape], &[DataType])>,
) -> Result<ValidatedMetadata> {
    validate_attributes(node)?;
    let attributes = MoeAttributes::from_block_quantized_node(node)?;
    let layout_version = optional_int_attr(node, "block_layout_version")?.unwrap_or(1);
    if layout_version != LAYOUT_VERSION {
        return Err(error(format!(
            "block_layout_version must be {LAYOUT_VERSION}, got {layout_version}"
        )));
    }
    let formats = parse_projection_formats(node)?;
    if let Some((shapes, dtypes)) = claim_metadata {
        validate_claim_metadata(node, shapes, dtypes, &attributes, formats).map_err(error)?;
    }
    Ok(ValidatedMetadata {
        attributes,
        formats,
    })
}

fn validate_claim_metadata(
    node: &Node,
    shapes: &[Shape],
    dtypes: &[DataType],
    attributes: &MoeAttributes,
    formats: ProjectionFormats,
) -> std::result::Result<(), String> {
    if !(5..=9).contains(&node.inputs.len()) {
        return Err(format!(
            "expected 5 to 9 positional inputs, got {}",
            node.inputs.len()
        ));
    }
    if node.outputs.len() != 1 {
        return Err(format!(
            "expected exactly 1 output, got {}",
            node.outputs.len()
        ));
    }
    if shapes.len() != node.inputs.len() || dtypes.len() != node.inputs.len() {
        return Err(format!(
            "claim metadata must cover all {} positional inputs (got {} shapes and {} dtypes)",
            node.inputs.len(),
            shapes.len(),
            dtypes.len()
        ));
    }
    for &index in &[0, 1, 2, 4] {
        if node.inputs[index].is_none() {
            return Err(format!(
                "required input {index} ('{}') is omitted",
                INPUT_NAMES[index]
            ));
        }
    }
    for index in 0..node.inputs.len() {
        if node.inputs[index].is_none() {
            if dtypes[index] != DataType::Undefined {
                return Err(format!(
                    "omitted input {index} ('{}') must use dtype Undefined",
                    INPUT_NAMES[index]
                ));
            }
            continue;
        }
        let expected = if matches!(index, 2 | 4 | 6) {
            DataType::Uint8
        } else {
            DataType::Float32
        };
        if dtypes[index] != expected {
            return Err(format!(
                "input {index} ('{}') dtype {:?} unsupported; expected {expected:?}",
                INPUT_NAMES[index], dtypes[index]
            ));
        }
    }
    if !matches!(shapes[0].len(), 2 | 3) {
        return Err(format!(
            "input 0 ('input') rank {} unsupported; expected 2 or 3",
            shapes[0].len()
        ));
    }
    if shapes[1].len() != 2 {
        return Err(format!(
            "input 1 ('router_logits') rank {} unsupported; expected 2",
            shapes[1].len()
        ));
    }
    for &index in &[2, 4] {
        if shapes[index].len() != 4 {
            return Err(format!(
                "input {index} ('{}') rank {} unsupported; expected 4",
                INPUT_NAMES[index],
                shapes[index].len()
            ));
        }
    }
    for &index in &[3, 5, 7, 8] {
        if node.inputs.get(index).is_some_and(Option::is_some) && shapes[index].len() != 2 {
            return Err(format!(
                "input {index} ('{}') rank {} unsupported; expected 2",
                INPUT_NAMES[index],
                shapes[index].len()
            ));
        }
    }
    if node.inputs.get(6).is_some_and(Option::is_some) && shapes[6].len() != 4 {
        return Err(format!(
            "input 6 ('fc3_experts_weights') rank {} unsupported; expected 4",
            shapes[6].len()
        ));
    }
    validate_partial_claim_shapes(node, shapes, attributes, formats)?;
    Ok(())
}

fn validate_partial_claim_shapes(
    node: &Node,
    shapes: &[Shape],
    attributes: &MoeAttributes,
    formats: ProjectionFormats,
) -> std::result::Result<(), String> {
    let hidden = shapes[0].last().and_then(|dim| dim.as_static());
    let experts = shapes[1][1].as_static();
    let rows = shapes[0][..shapes[0].len() - 1]
        .iter()
        .map(|dim| dim.as_static())
        .try_fold(Some(1usize), |rows, dim| match (rows, dim) {
            (Some(rows), Some(dim)) => rows
                .checked_mul(dim)
                .map(Some)
                .ok_or_else(|| "input row count overflow".to_string()),
            _ => Ok(None),
        })?;
    if let Some(experts) = experts
        && attributes.k > experts
    {
        return Err(format!("k={} exceeds num_experts={experts}", attributes.k));
    }
    check_static_axis(shapes, 1, 0, rows, "router_logits rows")?;
    require_same_static_axis(shapes, 2, 0, 1, 1, "fc1 expert count")?;
    require_same_static_axis(shapes, 4, 0, 1, 1, "fc2 expert count")?;
    if let (Some(fc2_hidden), Some(hidden)) = (shapes[4][1].as_static(), hidden)
        && fc2_hidden != hidden
    {
        return Err(format!(
            "fc2 output width {fc2_hidden} must equal hidden size {hidden}"
        ));
    }
    if let Some(block_bytes) = shapes[2][3].as_static()
        && block_bytes != formats.fc1.block_bytes()
    {
        return Err(format!(
            "fc1 block byte width {block_bytes} must equal {}",
            formats.fc1.block_bytes()
        ));
    }
    if let Some(block_bytes) = shapes[4][3].as_static()
        && block_bytes != formats.fc2.block_bytes()
    {
        return Err(format!(
            "fc2 block byte width {block_bytes} must equal {}",
            formats.fc2.block_bytes()
        ));
    }
    let fc1_size = shapes[2][1].as_static();
    let inter = fc1_size.and_then(|fc1_size| {
        if attributes.swiglu_fusion == 0 {
            Some(fc1_size)
        } else {
            (fc1_size % 2 == 0).then_some(fc1_size / 2)
        }
    });
    if attributes.swiglu_fusion != 0 && fc1_size.is_some() && inter.is_none() {
        return Err("fused SwiGLU fc1_out must be even".into());
    }
    if inter == Some(0) {
        return Err("inferred inter dimension must be non-zero".into());
    }
    check_static_packed_shape(shapes, 2, experts, fc1_size, hidden, formats.fc1)?;
    check_static_packed_shape(shapes, 4, experts, hidden, inter, formats.fc2)?;
    check_static_optional_shape(node, shapes, 3, experts, fc1_size)?;
    check_static_optional_shape(node, shapes, 5, experts, hidden)?;

    let has_fc3 = node.inputs.get(6).is_some_and(Option::is_some);
    if attributes.uses_separate_gate(has_fc3) {
        if !has_fc3 {
            return Err("unfused swiglu requires fc3_experts_weights".into());
        }
        let fc3_format = formats
            .fc3
            .ok_or_else(|| "fc3_experts_weights requires the fc3_format attribute".to_string())?;
        check_static_packed_shape(shapes, 6, experts, inter, hidden, fc3_format)?;
        check_static_optional_shape(node, shapes, 7, experts, inter)?;
    } else if has_fc3 || node.inputs.get(7).is_some_and(Option::is_some) {
        return Err("fc3 inputs are only valid for unfused swiglu or silu gated-GLU".into());
    }
    check_static_optional_shape(node, shapes, 8, rows, experts)?;
    Ok(())
}

fn check_static_packed_shape(
    shapes: &[Shape],
    index: usize,
    experts: Option<usize>,
    out_features: Option<usize>,
    in_features: Option<usize>,
    format: BlockFormat,
) -> std::result::Result<(), String> {
    if let Some(width) = in_features
        && !width.is_multiple_of(format.qk())
    {
        return Err(format!(
            "input {index} ('{}') input width {width} has a partial {:?} block tail; \
             full blocks of {} elements are required",
            INPUT_NAMES[index],
            format,
            format.qk()
        ));
    }
    check_static_axis(shapes, index, 0, experts, "expert count")?;
    check_static_axis(shapes, index, 1, out_features, "output width")?;
    check_static_axis(
        shapes,
        index,
        2,
        in_features.map(|width| width / format.qk()),
        "block count",
    )?;
    check_static_axis(
        shapes,
        index,
        3,
        Some(format.block_bytes()),
        "block byte width",
    )
}

fn check_static_optional_shape(
    node: &Node,
    shapes: &[Shape],
    index: usize,
    rows: Option<usize>,
    width: Option<usize>,
) -> std::result::Result<(), String> {
    if node.inputs.get(index).is_some_and(Option::is_some) {
        check_static_axis(shapes, index, 0, rows, "dimension 0")?;
        check_static_axis(shapes, index, 1, width, "dimension 1")?;
    }
    Ok(())
}

fn check_static_axis(
    shapes: &[Shape],
    index: usize,
    axis: usize,
    expected: Option<usize>,
    name: &str,
) -> std::result::Result<(), String> {
    if let (Some(actual), Some(expected)) = (shapes[index][axis].as_static(), expected)
        && actual != expected
    {
        return Err(format!(
            "input {index} ('{}') {name} {actual} must equal {expected}",
            INPUT_NAMES[index]
        ));
    }
    Ok(())
}

fn require_same_static_axis(
    shapes: &[Shape],
    left_input: usize,
    left_axis: usize,
    right_input: usize,
    right_axis: usize,
    name: &str,
) -> std::result::Result<(), String> {
    if let (Some(left), Some(right)) = (
        shapes[left_input][left_axis].as_static(),
        shapes[right_input][right_axis].as_static(),
    ) && left != right
    {
        return Err(format!("{name} {left} must equal {right}"));
    }
    Ok(())
}

fn validate_bias(inputs: &[TensorView], index: usize, experts: usize, width: usize) -> Result<()> {
    if let Some(input) = optional_input(inputs, index)
        && input.shape != [experts, width]
    {
        return Err(error(format!(
            "{} must have shape [{experts}, {width}], got {:?}",
            INPUT_NAMES[index], input.shape
        )));
    }
    Ok(())
}

fn require_exact_rank(index: usize, shape: &[usize], expected: usize) -> Result<()> {
    if shape.len() != expected {
        return Err(error(format!(
            "input {index} ('{}') must have rank {expected}, got {shape:?}",
            INPUT_NAMES[index]
        )));
    }
    Ok(())
}

fn require_dtype(index: usize, input: &TensorView, expected: DataType) -> Result<()> {
    if input.dtype != expected {
        return Err(error(format!(
            "input {index} ('{}') dtype {:?} unsupported; expected {expected:?}",
            INPUT_NAMES[index], input.dtype
        )));
    }
    Ok(())
}

fn optional_input<'a>(inputs: &'a [TensorView<'a>], index: usize) -> Option<&'a TensorView<'a>> {
    inputs.get(index).filter(|input| !input.is_absent())
}

fn optional_dense(inputs: &[TensorView], index: usize) -> Result<Option<Vec<f32>>> {
    optional_input(inputs, index).map(to_dense_f32).transpose()
}

fn checked_product(shape: &[usize], name: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or_else(|| error(format!("{name} overflow")))
    })
}

fn optional_int_attr(node: &Node, name: &str) -> Result<Option<i64>> {
    node.attr(name)
        .map(|attr| {
            attr.as_int()
                .ok_or_else(|| error(format!("attribute '{name}' must be an integer")))
        })
        .transpose()
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{OP}: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuExecutionProvider;
    use crate::kernels::block_quantized_matmul::DEFAULT_DENSE_WEIGHT_CACHE_BYTES;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{Attribute, Dim, Graph, NodeId, SymbolId, static_shape};

    const H: usize = 32;
    const E: usize = 2;

    fn attrs(
        activation: &str,
        k: usize,
        normalize: bool,
        swiglu_fusion: usize,
    ) -> Vec<(&'static str, Attribute)> {
        vec![
            ("fc1_format", Attribute::String(b"mxfp4".to_vec())),
            ("fc2_format", Attribute::String(b"mxfp4".to_vec())),
            ("block_layout_version", Attribute::Int(1)),
            (
                "activation_type",
                Attribute::String(activation.as_bytes().to_vec()),
            ),
            ("k", Attribute::Int(k as i64)),
            (
                "normalize_routing_weights",
                Attribute::Int(i64::from(normalize)),
            ),
            ("swiglu_fusion", Attribute::Int(swiglu_fusion as i64)),
        ]
    }

    /// Extend a uniform-format attribute set with `fc3_format` for the unfused
    /// gated (fc3-wired) paths.
    fn with_fc3_format(
        mut spec: Vec<(&'static str, Attribute)>,
        fc3_format: &str,
    ) -> Vec<(&'static str, Attribute)> {
        spec.push((
            "fc3_format",
            Attribute::String(fc3_format.as_bytes().to_vec()),
        ));
        spec
    }

    fn model_node(
        shapes: &[Option<(DataType, Vec<usize>)>],
        attrs: &[(&str, Attribute)],
    ) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert("pkg.nxrt".into(), 1);
        let inputs = shapes
            .iter()
            .enumerate()
            .map(|(index, input)| {
                input.as_ref().map(|(dtype, shape)| {
                    let value = graph.create_named_value(
                        format!("input_{index}"),
                        *dtype,
                        static_shape(shape.iter().copied()),
                    );
                    graph.add_input(value);
                    value
                })
            })
            .collect();
        let output = graph.create_named_value("output", DataType::Float32, static_shape([1, H]));
        let mut node = Node::new(NodeId(0), OP, inputs, vec![output]);
        node.domain = "pkg.nxrt".into();
        for (name, value) in attrs {
            node.attributes.insert((*name).into(), value.clone());
        }
        let node = graph.insert_node(node);
        graph.add_output(output);
        (graph, node)
    }

    fn packed_matrix(
        experts: usize,
        out_features: usize,
        mut code: impl FnMut(usize, usize, usize) -> u8,
    ) -> Vec<u8> {
        let mut packed = vec![0u8; experts * out_features * 17];
        for expert in 0..experts {
            for output in 0..out_features {
                let block = &mut packed[(expert * out_features + output) * 17..][..17];
                block[0] = 127;
                for input in 0..H {
                    let value = code(expert, output, input) & 0x0f;
                    let byte = &mut block[1 + input % 16];
                    if input < 16 {
                        *byte |= value;
                    } else {
                        *byte |= value << 4;
                    }
                }
            }
        }
        packed
    }

    fn identity_projection(scales: [u8; E]) -> Vec<u8> {
        packed_matrix(
            E,
            H,
            |expert, output, input| {
                if output == input { scales[expert] } else { 0 }
            },
        )
    }

    /// Build a decodable packed projection tensor `[experts, out_features,
    /// ceil(in_features/qk), block_bytes]` for an arbitrary native format. Each
    /// block is filled with a deterministic byte pattern plus the format's own
    /// scale header so every decoder accepts it. This is the fixture that lets a
    /// single projection carry its own qtype/block width independently of the
    /// other projections (the whole point of the mixed-projection ABI).
    fn packed_for_format(
        format: BlockFormat,
        experts: usize,
        out_features: usize,
        in_features: usize,
    ) -> Vec<u8> {
        let block_bytes = format.block_bytes();
        let blocks_per_row = in_features.div_ceil(format.qk());
        let mut packed = vec![0u8; experts * out_features * blocks_per_row * block_bytes];
        for (block_index, block) in packed.chunks_exact_mut(block_bytes).enumerate() {
            for (index, byte) in block.iter_mut().enumerate() {
                *byte = block_index.wrapping_mul(29).wrapping_add(index * 17) as u8;
            }
            match format {
                BlockFormat::Mxfp4 => block[0] = 127,
                BlockFormat::Iq1M => block[48..56].fill(0),
                _ => block[..2].copy_from_slice(&half::f16::from_f32(0.125).to_le_bytes()),
            }
        }
        packed
    }

    fn run(
        activation: &str,
        k: usize,
        normalize: bool,
        swiglu_fusion: usize,
        input: &[f32],
        logits: &[f32],
        fc1: &[u8],
        fc1_out: usize,
        fc2: &[u8],
        router_weights: Option<&[f32]>,
    ) -> Vec<f32> {
        run_with_attrs(
            &attrs(activation, k, normalize, swiglu_fusion),
            input,
            logits,
            fc1,
            fc1_out,
            fc2,
            router_weights,
        )
    }

    fn run_with_attrs(
        attrs: &[(&str, Attribute)],
        input: &[f32],
        logits: &[f32],
        fc1: &[u8],
        fc1_out: usize,
        fc2: &[u8],
        router_weights: Option<&[f32]>,
    ) -> Vec<f32> {
        run_with_attrs_and_experts(attrs, E, input, logits, fc1, fc1_out, fc2, router_weights)
    }

    fn run_with_attrs_and_experts(
        attrs: &[(&str, Attribute)],
        experts: usize,
        input: &[f32],
        logits: &[f32],
        fc1: &[u8],
        fc1_out: usize,
        fc2: &[u8],
        router_weights: Option<&[f32]>,
    ) -> Vec<f32> {
        let mut shapes = vec![
            Some((DataType::Float32, vec![1, H])),
            Some((DataType::Float32, vec![1, experts])),
            Some((DataType::Uint8, vec![experts, fc1_out, 1, 17])),
            None,
            Some((DataType::Uint8, vec![experts, H, 1, 17])),
            None,
            None,
            None,
        ];
        if router_weights.is_some() {
            shapes.push(Some((DataType::Float32, vec![1, experts])));
        }
        let (graph, node) = model_node(&shapes, attrs);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(
                graph.node(node),
                &shapes
                    .iter()
                    .map(|shape| {
                        shape
                            .as_ref()
                            .map_or_else(Vec::new, |(_, shape)| shape.clone())
                    })
                    .collect::<Vec<_>>(),
                1,
            )
            .expect("valid BlockQuantizedMoE kernel");
        let input = Owned::f32(&[1, H], input);
        let logits = Owned::f32(&[1, experts], logits);
        let fc1 = Owned::u8(&[experts, fc1_out, 1, 17], fc1);
        let fc2 = Owned::u8(&[experts, H, 1, 17], fc2);
        let router = router_weights.map(|weights| Owned::f32(&[1, experts], weights));
        let mut views = vec![
            input.view(),
            logits.view(),
            fc1.view(),
            TensorView::absent(DataType::Float32),
            fc2.view(),
            TensorView::absent(DataType::Float32),
            TensorView::absent(DataType::Uint8),
            TensorView::absent(DataType::Float32),
        ];
        if let Some(router) = &router {
            views.push(router.view());
        }
        let mut output = Owned::f32(&[1, H], &[0.0; H]);
        kernel
            .execute(&views, &mut [output.view_mut()])
            .expect("execute BlockQuantizedMoE");
        output.to_f32()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "index {index}: got {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn constant_moe_expert_weights_reuse_bounded_cached_dense_path() {
        let experts = E;
        let fc1_out = H;
        let input_values: Vec<f32> = (0..H).map(|i| i as f32 / 16.0 - 1.0).collect();
        let logits_values = [4.0, -4.0];
        let fc1_values = identity_projection([2, 4]);
        let fc2_values = identity_projection([2, 2]);
        let shapes = vec![
            Some((DataType::Float32, vec![1, H])),
            Some((DataType::Float32, vec![1, experts])),
            Some((DataType::Uint8, vec![experts, fc1_out, 1, 17])),
            None,
            Some((DataType::Uint8, vec![experts, H, 1, 17])),
            None,
            None,
            None,
        ];
        let (graph, node) = model_node(&shapes, &attrs("identity", 1, false, 0));
        let ValidatedMetadata {
            attributes,
            formats,
        } = validate_metadata(graph.node(node), None).expect("valid BlockQuantizedMoE metadata");
        let mut kernel = BlockQuantizedMoEKernel {
            attributes,
            formats,
            constant_inputs: [false; 9],
            weight_identities: std::array::from_fn(|_| DenseWeightIdentity::default()),
            weight_cache: DenseWeightCache::new(),
        };
        kernel.set_constant_inputs(&[false, false, true, false, true, false, false, false]);

        let input = Owned::f32(&[1, H], &input_values);
        let logits = Owned::f32(&[1, experts], &logits_values);
        let fc1 = Owned::u8(&[experts, fc1_out, 1, 17], &fc1_values);
        let fc2 = Owned::u8(&[experts, H, 1, 17], &fc2_values);
        let views = [
            input.view(),
            logits.view(),
            fc1.view(),
            TensorView::absent(DataType::Float32),
            fc2.view(),
            TensorView::absent(DataType::Float32),
            TensorView::absent(DataType::Uint8),
            TensorView::absent(DataType::Float32),
        ];
        let mut output = Owned::f32(&[1, H], &[0.0; H]);

        let hits_before = BLOCK_QUANT_MOE_CACHED_DENSE_TEST_HITS.load(Ordering::Relaxed);
        kernel
            .execute(&views, &mut [output.view_mut()])
            .expect("first MoE execution");
        let identity_after_first = [
            kernel.weight_identities[0].stats(),
            kernel.weight_identities[1].stats(),
        ];
        let activity_after_first = kernel.weight_cache.activity();
        kernel
            .execute(&views, &mut [output.view_mut()])
            .expect("second MoE execution");
        assert_eq!(
            kernel.weight_cache.stats().0,
            2,
            "one routed expert should cache exactly fc1 and fc2 dense projections across repeated calls"
        );
        assert_eq!(
            [
                kernel.weight_identities[0].stats(),
                kernel.weight_identities[1].stats(),
            ],
            identity_after_first,
            "repeat routing must not rehash or rematerialize either multi-expert tensor"
        );
        let fc1_expert_bytes = fc1_values.len() / experts;
        let fc2_expert_bytes = fc2_values.len() / experts;
        assert_eq!(
            identity_after_first,
            [(1, fc1_expert_bytes, 0), (1, fc2_expert_bytes, 0)],
            "only the routed expert slices should be hashed, never the full multi-expert tensors"
        );
        assert_eq!(
            kernel.weight_cache.activity(),
            (activity_after_first.0 + 2, activity_after_first.1),
            "the second MoE call must hit both cached projections without rebuilding"
        );
        assert!(
            BLOCK_QUANT_MOE_CACHED_DENSE_TEST_HITS.load(Ordering::Relaxed) >= hits_before + 2,
            "second MoE execution must hit cached-dense fc1 and fc2 weights"
        );
    }

    /// One expert's dense expansion is built once and handed to every row
    /// routed to it -- not re-resolved per row, and not rebuilt per row block.
    ///
    /// The sharing is what makes cached-dense affordable at prefill widths: a
    /// 48-row batch routed to one expert must pay for one fc1 and one fc2
    /// expansion, the same as a single decode token. Counting *builds* alone
    /// would not see a regression that moved resolution into the row loop --
    /// the cache would absorb it and turn 46 rows into 46 hits -- so this
    /// pins the whole activity pair. `(hits, builds) == (0, 2)` says the
    /// resolve happened once per projection for the entire batch.
    ///
    /// Kernel-local counters deliberately, not the process-wide
    /// `BLOCK_QUANT_MOE_DENSE_EXPANSIONS`: tests in this binary run
    /// concurrently, so a delta on a global would be a race dressed as an
    /// assertion.
    #[test]
    fn one_expert_expansion_is_shared_by_every_row_routed_to_it() {
        const ROWS: usize = 48;
        let experts = E;
        let fc1_out = H;
        let input_values: Vec<f32> = (0..ROWS * H)
            .map(|i| (i % 64) as f32 / 16.0 - 1.0)
            .collect();
        // Every row prefers expert 0, so the batch is one expert group of 48.
        let logits_values: Vec<f32> = (0..ROWS).flat_map(|_| [4.0f32, -4.0]).collect();
        let fc1_values = identity_projection([2, 4]);
        let fc2_values = identity_projection([2, 2]);
        let shapes = vec![
            Some((DataType::Float32, vec![ROWS, H])),
            Some((DataType::Float32, vec![ROWS, experts])),
            Some((DataType::Uint8, vec![experts, fc1_out, 1, 17])),
            None,
            Some((DataType::Uint8, vec![experts, H, 1, 17])),
            None,
            None,
            None,
        ];
        let (graph, node) = model_node(&shapes, &attrs("identity", 1, false, 0));
        let ValidatedMetadata {
            attributes,
            formats,
        } = validate_metadata(graph.node(node), None).expect("valid BlockQuantizedMoE metadata");
        let mut kernel = BlockQuantizedMoEKernel {
            attributes,
            formats,
            constant_inputs: [false; 9],
            weight_identities: std::array::from_fn(|_| DenseWeightIdentity::default()),
            // The default ceiling, pinned rather than inherited: `new()` reads
            // `ONNX_GENAI_CPU_BLOCK_QUANT_CACHE_BYTES` through a process-wide
            // `OnceLock`, so an ambient `=0` turns every resolve into
            // `MissNotStored` and this cell would be reporting the environment
            // instead of the kernel. It fails loudly there rather than
            // silently, but a cell about cache accounting should not depend on
            // the box it runs on.
            weight_cache: DenseWeightCache::with_limit(DEFAULT_DENSE_WEIGHT_CACHE_BYTES),
        };
        kernel.set_constant_inputs(&[false, false, true, false, true, false, false, false]);

        let input = Owned::f32(&[ROWS, H], &input_values);
        let logits = Owned::f32(&[ROWS, experts], &logits_values);
        let fc1 = Owned::u8(&[experts, fc1_out, 1, 17], &fc1_values);
        let fc2 = Owned::u8(&[experts, H, 1, 17], &fc2_values);
        let views = [
            input.view(),
            logits.view(),
            fc1.view(),
            TensorView::absent(DataType::Float32),
            fc2.view(),
            TensorView::absent(DataType::Float32),
            TensorView::absent(DataType::Uint8),
            TensorView::absent(DataType::Float32),
        ];
        let mut output = Owned::f32(&[ROWS, H], &vec![0.0; ROWS * H]);

        kernel
            .execute(&views, &mut [output.view_mut()])
            .expect("48-row MoE execution");

        assert_eq!(
            kernel.weight_cache.activity(),
            (0, 2),
            "48 rows through one expert must resolve fc1 and fc2 once each, \
             not once per row"
        );
        assert_eq!(
            kernel.weight_cache.stats().0,
            2,
            "exactly the routed expert's two projections are resident"
        );

        // Non-vacuity: the batch really was 48 rows of real work. Without this
        // the assertions above are satisfied by a kernel that computed nothing,
        // which is the cheapest way to expand a weight zero times.
        let produced = output.to_f32();
        assert_eq!(produced.len(), ROWS * H);
        assert!(
            produced
                .chunks_exact(H)
                .all(|row| row.iter().any(|v| *v != 0.0)),
            "every routed row must carry expert output"
        );

        // The shared buffer itself, rather than a count that stands in for it:
        // two resolutions of the same expert projection hand back one
        // allocation, so every row block reads the same immutable pack.
        let first = kernel
            .dequantize_expert_cached(
                Some(&kernel.weight_identities[0]),
                1,
                &fc1.view(),
                0,
                fc1_out,
                H,
                experts,
                kernel.formats.fc1,
            )
            .expect("resolve expert 0 fc1");
        let again = kernel
            .dequantize_expert_cached(
                Some(&kernel.weight_identities[0]),
                1,
                &fc1.view(),
                0,
                fc1_out,
                H,
                experts,
                kernel.formats.fc1,
            )
            .expect("resolve expert 0 fc1 again");
        assert!(
            Arc::ptr_eq(&first, &again),
            "row blocks must share one immutable expansion, not copies of it"
        );

        // Negative control: the counter is not simply stuck. A different
        // expert is a different pack, and says so both ways.
        let builds_before = kernel.weight_cache.activity().1;
        let other = kernel
            .dequantize_expert_cached(
                Some(&kernel.weight_identities[0]),
                1,
                &fc1.view(),
                1,
                fc1_out,
                H,
                experts,
                kernel.formats.fc1,
            )
            .expect("resolve expert 1 fc1");
        assert_eq!(
            kernel.weight_cache.activity().1,
            builds_before + 1,
            "an unseen expert must build"
        );
        assert!(
            !Arc::ptr_eq(&first, &other),
            "two experts must not share one expansion"
        );
    }

    #[test]
    fn nonconstant_fc3_matches_dense_reference_for_unfused_gated_activations() {
        let experts = E;
        let input_values: Vec<f32> = (0..H).map(|i| i as f32 / 32.0 + 0.25).collect();
        let logits_values = [4.0, -4.0];
        let fc1_values = identity_projection([2, 4]);
        let fc2_values = identity_projection([2, 2]);
        let fc3_values = identity_projection([4, 6]);

        for activation in ["swiglu", "silu"] {
            let shapes = vec![
                Some((DataType::Float32, vec![1, H])),
                Some((DataType::Float32, vec![1, experts])),
                Some((DataType::Uint8, vec![experts, H, 1, 17])),
                None,
                Some((DataType::Uint8, vec![experts, H, 1, 17])),
                None,
                Some((DataType::Uint8, vec![experts, H, 1, 17])),
                None,
            ];
            let attributes_spec = with_fc3_format(attrs(activation, 1, true, 0), "mxfp4");
            let (graph, node) = model_node(&shapes, &attributes_spec);
            let ValidatedMetadata {
                attributes,
                formats,
            } = validate_metadata(graph.node(node), None).expect("valid gated MoE metadata");
            let mut kernel = BlockQuantizedMoEKernel {
                attributes,
                formats,
                constant_inputs: [false; 9],
                weight_identities: std::array::from_fn(|_| DenseWeightIdentity::default()),
                weight_cache: DenseWeightCache::new(),
            };
            kernel.set_constant_inputs(&[false, false, true, false, true, false, false, false]);
            assert!(
                !kernel.constant_inputs[6],
                "fc3 must remain a runtime input"
            );

            let input = Owned::f32(&[1, H], &input_values);
            let logits = Owned::f32(&[1, experts], &logits_values);
            let fc1 = Owned::u8(&[experts, H, 1, 17], &fc1_values);
            let fc2 = Owned::u8(&[experts, H, 1, 17], &fc2_values);
            let fc3 = Owned::u8(&[experts, H, 1, 17], &fc3_values);
            let views = [
                input.view(),
                logits.view(),
                fc1.view(),
                TensorView::absent(DataType::Float32),
                fc2.view(),
                TensorView::absent(DataType::Float32),
                fc3.view(),
                TensorView::absent(DataType::Float32),
            ];
            let mut output = Owned::f32(&[1, H], &[0.0; H]);
            kernel
                .execute(&views, &mut [output.view_mut()])
                .expect("execute gated BlockQuantizedMoE with runtime fc3");

            let expert_bytes = H * 17;
            let dense_fc1 =
                dequantize_expert_slice(formats.fc1, &fc1_values[..expert_bytes], H, H).unwrap();
            let dense_fc2 =
                dequantize_expert_slice(formats.fc2, &fc2_values[..expert_bytes], H, H).unwrap();
            let dense_fc3 = dequantize_expert_slice(
                formats.fc3.expect("fc3_format present for gated path"),
                &fc3_values[..expert_bytes],
                H,
                H,
            )
            .unwrap();
            let expected = run_expert_grouped(
                &input_values,
                1,
                &dense_fc1,
                None,
                &dense_fc2,
                None,
                Some(&dense_fc3),
                None,
                H,
                H,
                H,
                &attributes,
            )
            .expect("dense gated MoE reference");
            assert_close(&output.to_f32(), &expected);
        }
    }

    #[test]
    fn cached_moe_matches_uncached_for_all_supported_block_formats() {
        for (format_name, expected_format) in [
            ("mxfp4", BlockFormat::Mxfp4),
            ("iq4_nl", BlockFormat::Iq4Nl),
            ("iq4_xs", BlockFormat::Iq4Xs),
            ("iq3_s", BlockFormat::Iq3S),
            ("iq3_xxs", BlockFormat::Iq3Xxs),
            ("iq2_s", BlockFormat::Iq2S),
            ("iq2_xs", BlockFormat::Iq2Xs),
            ("iq2_xxs", BlockFormat::Iq2Xxs),
            ("iq1_s", BlockFormat::Iq1S),
            ("iq1_m", BlockFormat::Iq1M),
        ] {
            let hidden = expected_format.qk();
            let experts = 2usize;
            let block_bytes = expected_format.block_bytes();
            let mut packed = vec![0u8; experts * hidden * block_bytes];
            for (block_index, block) in packed.chunks_exact_mut(block_bytes).enumerate() {
                for (index, byte) in block.iter_mut().enumerate() {
                    *byte = block_index.wrapping_mul(29).wrapping_add(index * 17) as u8;
                }
                match expected_format {
                    BlockFormat::Mxfp4 => block[0] = 127,
                    BlockFormat::Iq1M => block[48..56].fill(0),
                    _ => block[..2].copy_from_slice(&half::f16::from_f32(0.125).to_le_bytes()),
                }
            }
            let mut attributes_spec = attrs("identity", 1, true, 0);
            attributes_spec[0] = (
                "fc1_format",
                Attribute::String(format_name.as_bytes().to_vec()),
            );
            attributes_spec[1] = (
                "fc2_format",
                Attribute::String(format_name.as_bytes().to_vec()),
            );
            let shapes = vec![
                Some((DataType::Float32, vec![1, hidden])),
                Some((DataType::Float32, vec![1, experts])),
                Some((DataType::Uint8, vec![experts, hidden, 1, block_bytes])),
                None,
                Some((DataType::Uint8, vec![experts, hidden, 1, block_bytes])),
                None,
                None,
                None,
            ];
            let (graph, node) = model_node(&shapes, &attributes_spec);
            let ValidatedMetadata {
                attributes,
                formats,
            } = validate_metadata(graph.node(node), None).expect("valid block format");
            assert_eq!(formats.fc1, expected_format);
            assert_eq!(formats.fc2, expected_format);
            let input_values: Vec<f32> = (0..hidden)
                .map(|index| ((index * 7 % 23) as f32 - 11.0) / 16.0)
                .collect();
            let input = Owned::f32(&[1, hidden], &input_values);
            let logits = Owned::f32(&[1, experts], &[4.0, -4.0]);
            let fc1 = Owned::u8(&[experts, hidden, 1, block_bytes], &packed);
            let fc2 = Owned::u8(&[experts, hidden, 1, block_bytes], &packed);
            let views = [
                input.view(),
                logits.view(),
                fc1.view(),
                TensorView::absent(DataType::Float32),
                fc2.view(),
                TensorView::absent(DataType::Float32),
                TensorView::absent(DataType::Uint8),
                TensorView::absent(DataType::Float32),
            ];

            let uncached = BlockQuantizedMoEKernel {
                attributes,
                formats,
                constant_inputs: [false; 9],
                weight_identities: std::array::from_fn(|_| DenseWeightIdentity::default()),
                weight_cache: DenseWeightCache::new(),
            };
            let mut expected = Owned::f32(&[1, hidden], &vec![0.0; hidden]);
            uncached
                .execute(&views, &mut [expected.view_mut()])
                .expect("uncached MoE reference");

            let mut cached = BlockQuantizedMoEKernel {
                attributes,
                formats,
                constant_inputs: [false; 9],
                weight_identities: std::array::from_fn(|_| DenseWeightIdentity::default()),
                weight_cache: DenseWeightCache::new(),
            };
            cached.set_constant_inputs(&[false, false, true, false, true]);
            let mut actual = Owned::f32(&[1, hidden], &vec![0.0; hidden]);
            cached
                .execute(&views, &mut [actual.view_mut()])
                .expect("cold cached MoE");
            cached
                .execute(&views, &mut [actual.view_mut()])
                .expect("warm cached MoE");
            assert_close(&actual.to_f32(), &expected.to_f32());
        }
    }

    #[test]
    fn block_quantized_moe_matches_dense_reference_topk_softmax() {
        let input: Vec<f32> = (0..H).map(|i| i as f32 / 16.0 - 1.0).collect();
        let fc1 = identity_projection([2, 4]);
        let fc2 = identity_projection([2, 2]);
        let actual = run(
            "identity",
            2,
            false,
            0,
            &input,
            &[0.0, 3.0f32.ln()],
            &fc1,
            H,
            &fc2,
            None,
        );
        let expected: Vec<f32> = input.iter().map(|value| value * 1.75).collect();
        assert_close(&actual, &expected);
    }

    #[test]
    fn block_quantized_moe_topk_selects_the_highest_scoring_expert() {
        const EXPERTS: usize = 3;
        let input: Vec<f32> = (0..H).map(|i| i as f32 / 8.0 - 1.0).collect();
        let fc1 = packed_matrix(EXPERTS, H, |expert, output, input| {
            if output == input {
                [2, 4, 6][expert]
            } else {
                0
            }
        });
        let fc2 = packed_matrix(
            EXPERTS,
            H,
            |_, output, input| {
                if output == input { 2 } else { 0 }
            },
        );
        let actual = run_with_attrs_and_experts(
            &attrs("identity", 2, true, 0),
            EXPERTS,
            &input,
            &[4.0, -5.0, 3.0],
            &fc1,
            H,
            &fc2,
            None,
        );
        let expert_zero_weight = 1.0 / (1.0 + (-1.0f32).exp());
        let expert_two_weight = 1.0 - expert_zero_weight;
        let expected: Vec<f32> = input
            .iter()
            .map(|value| value * (expert_zero_weight + 4.0 * expert_two_weight))
            .collect();
        assert_close(&actual, &expected);
    }

    #[test]
    fn block_quantized_moe_router_weights_and_normalization_match_reference() {
        let input = vec![1.0f32; H];
        let fc1 = identity_projection([2, 4]);
        let fc2 = identity_projection([2, 2]);
        let unnormalized = run(
            "identity",
            2,
            false,
            0,
            &input,
            &[2.0, 1.0],
            &fc1,
            H,
            &fc2,
            Some(&[2.0, 1.0]),
        );
        let normalized = run(
            "identity",
            2,
            true,
            0,
            &input,
            &[2.0, 1.0],
            &fc1,
            H,
            &fc2,
            Some(&[2.0, 1.0]),
        );
        assert_close(&unnormalized, &[4.0; H]);
        assert_close(&normalized, &[4.0 / 3.0; H]);
    }

    #[test]
    fn block_quantized_moe_silu_matches_dense_reference() {
        let input: Vec<f32> = (0..H).map(|i| i as f32 / 32.0 - 0.5).collect();
        let fc1 = identity_projection([2, 2]);
        let fc2 = identity_projection([2, 2]);
        let actual = run(
            "silu",
            1,
            true,
            0,
            &input,
            &[2.0, -2.0],
            &fc1,
            H,
            &fc2,
            None,
        );
        let expected: Vec<f32> = input
            .iter()
            .map(|&value| value / (1.0 + (-value).exp()))
            .collect();
        assert_close(&actual, &expected);
    }

    #[test]
    fn block_quantized_moe_relu_and_gelu_match_dense_reference() {
        let input: Vec<f32> = (0..H).map(|i| i as f32 / 8.0 - 2.0).collect();
        let fc1 = identity_projection([2, 2]);
        let fc2 = identity_projection([2, 2]);
        let relu = run(
            "relu",
            1,
            true,
            0,
            &input,
            &[3.0, -3.0],
            &fc1,
            H,
            &fc2,
            None,
        );
        assert_close(
            &relu,
            &input.iter().map(|value| value.max(0.0)).collect::<Vec<_>>(),
        );

        let gelu = run(
            "gelu",
            1,
            true,
            0,
            &input,
            &[3.0, -3.0],
            &fc1,
            H,
            &fc2,
            None,
        );
        let expected: Vec<f32> = input
            .iter()
            .map(|&value| {
                0.5 * value
                    * (1.0 + (0.797_884_6_f32 * (value + 0.044_715 * value * value * value)).tanh())
            })
            .collect();
        assert_close(&gelu, &expected);
    }

    #[test]
    fn block_quantized_moe_fused_swiglu_matches_dense_reference() {
        let input: Vec<f32> = (0..H).map(|i| i as f32 / 32.0 + 0.25).collect();
        let fc1 = packed_matrix(E, 2 * H, |_, output, input| {
            if output < H && output == input {
                2
            } else if output >= H && output - H == input {
                4
            } else {
                0
            }
        });
        let fc2 = identity_projection([2, 2]);
        let actual = run(
            "swiglu",
            1,
            true,
            2,
            &input,
            &[2.0, -2.0],
            &fc1,
            2 * H,
            &fc2,
            None,
        );
        let expected: Vec<f32> = input
            .iter()
            .map(|&value| 2.0 * value * value / (1.0 + (-value).exp()))
            .collect();
        assert_close(&actual, &expected);
    }

    #[test]
    fn block_quantized_moe_swiglu_attributes_affect_dense_reference() {
        let input = vec![1.0f32; H];
        let fc1 = packed_matrix(E, 2 * H, |_, output, input| {
            if output < H && output == input {
                2
            } else if output >= H && output - H == input {
                4
            } else {
                0
            }
        });
        let fc2 = identity_projection([2, 2]);
        let default = run(
            "swiglu",
            1,
            true,
            2,
            &input,
            &[2.0, -2.0],
            &fc1,
            2 * H,
            &fc2,
            None,
        );
        let mut custom_attrs = attrs("swiglu", 1, true, 2);
        custom_attrs.extend([
            ("activation_alpha", Attribute::Float(2.0)),
            ("activation_beta", Attribute::Float(1.0)),
            ("swiglu_limit", Attribute::Float(0.5)),
        ]);
        let actual = run_with_attrs(&custom_attrs, &input, &[2.0, -2.0], &fc1, 2 * H, &fc2, None);
        let expected = 0.5 * (1.0 / (1.0 + (-1.0f32).exp())) * 1.5;
        assert_close(&actual, &[expected; H]);
        assert!(
            actual
                .iter()
                .zip(default)
                .any(|(&actual, default)| actual != default)
        );
    }

    #[test]
    fn block_quantized_moe_is_bit_deterministic() {
        let input: Vec<f32> = (0..H).map(|i| i as f32 * 0.03125).collect();
        let fc1 = identity_projection([2, 4]);
        let fc2 = identity_projection([2, 2]);
        let first = run(
            "identity",
            2,
            true,
            0,
            &input,
            &[1.0, 1.0],
            &fc1,
            H,
            &fc2,
            None,
        );
        let second = run(
            "identity",
            2,
            true,
            0,
            &input,
            &[1.0, 1.0],
            &fc1,
            H,
            &fc2,
            None,
        );
        assert_eq!(
            first
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn block_quantized_moe_manifest_counter_proves_dense_f32_expert_dispatch() {
        let before =
            BLOCK_QUANTIZED_MOE_DENSE_F32_TEST_HITS.load(std::sync::atomic::Ordering::Relaxed);
        let input = vec![1.0f32; H];
        let fc1 = identity_projection([2, 4]);
        let fc2 = identity_projection([2, 2]);

        let actual = run(
            "identity",
            1,
            true,
            0,
            &input,
            &[3.0, -3.0],
            &fc1,
            H,
            &fc2,
            None,
        );

        let after =
            BLOCK_QUANTIZED_MOE_DENSE_F32_TEST_HITS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            after > before,
            "dispatch manifest counter must prove BlockQuantizedMoE ran the dense-f32 expert path"
        );
        assert_close(&actual, &[1.0; H]);
    }

    fn claim_fixture() -> (Graph, NodeId, Vec<Shape>, Vec<DataType>) {
        let shapes = vec![
            Some((DataType::Float32, vec![1, H])),
            Some((DataType::Float32, vec![1, E])),
            Some((DataType::Uint8, vec![E, H, 1, 17])),
            None,
            Some((DataType::Uint8, vec![E, H, 1, 17])),
            None,
            None,
            None,
        ];
        let (graph, node) = model_node(&shapes, &attrs("identity", 1, false, 0));
        let claim_shapes = shapes
            .iter()
            .map(|shape| {
                shape
                    .as_ref()
                    .map_or_else(Vec::new, |(_, shape)| static_shape(shape.iter().copied()))
            })
            .collect();
        let dtypes = shapes
            .iter()
            .map(|shape| {
                shape
                    .as_ref()
                    .map_or(DataType::Undefined, |(dtype, _)| *dtype)
            })
            .collect();
        (graph, node, claim_shapes, dtypes)
    }

    #[test]
    fn block_quantized_moe_claim_gate_accepts_valid_and_omitted_optionals() {
        let (graph, node, shapes, dtypes) = claim_fixture();
        let ep = CpuExecutionProvider::new();
        assert!(
            ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[])
                .is_supported()
        );
    }

    #[test]
    fn block_quantized_moe_claim_gate_rejects_bad_dtype_format_and_arity() {
        let (graph, node, shapes, mut dtypes) = claim_fixture();
        let ep = CpuExecutionProvider::new();
        dtypes[2] = DataType::Float32;
        let rejected = ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]);
        assert!(rejected.reason().unwrap().contains("dtype"));

        let (mut graph, node, shapes, dtypes) = claim_fixture();
        graph.node_mut(node).attributes.insert(
            "fc1_format".into(),
            Attribute::String(b"k3_unpublished".to_vec()),
        );
        let rejected = ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]);
        assert!(rejected.reason().unwrap().contains("unsupported format"));

        let (mut graph, node, shapes, dtypes) = claim_fixture();
        graph
            .node_mut(node)
            .attributes
            .insert("use_sparse_mixer".into(), Attribute::Int(0));
        let rejected = ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]);
        assert!(
            rejected
                .reason()
                .unwrap()
                .contains("not part of the BlockQuantizedMoE ABI")
        );

        let (mut graph, node, mut shapes, mut dtypes) = claim_fixture();
        graph.node_mut(node).inputs.truncate(4);
        shapes.truncate(4);
        dtypes.truncate(4);
        let rejected = ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]);
        assert!(rejected.reason().unwrap().contains("5 to 9"));
    }

    #[test]
    fn block_quantized_moe_claim_gate_rejects_static_optional_and_fc3_errors_with_symbolic_dims() {
        let inputs = vec![
            Some((DataType::Float32, vec![1, H])),
            Some((DataType::Float32, vec![1, E])),
            Some((DataType::Uint8, vec![E, H, 1, 17])),
            Some((DataType::Float32, vec![E, H])),
            Some((DataType::Uint8, vec![E, H, 1, 17])),
            None,
            Some((DataType::Uint8, vec![E, H, 1, 17])),
            Some((DataType::Float32, vec![E, H])),
            Some((DataType::Float32, vec![1, E])),
        ];
        let (graph, node) = model_node(
            &inputs,
            &with_fc3_format(attrs("swiglu", 1, false, 0), "mxfp4"),
        );
        let mut shapes: Vec<Shape> = inputs
            .iter()
            .map(|input| {
                input
                    .as_ref()
                    .map_or_else(Vec::new, |(_, shape)| static_shape(shape.iter().copied()))
            })
            .collect();
        shapes[0][0] = Dim::Symbolic(SymbolId(0));
        let dtypes = inputs
            .iter()
            .map(|input| {
                input
                    .as_ref()
                    .map_or(DataType::Undefined, |(dtype, _)| *dtype)
            })
            .collect::<Vec<_>>();
        let ep = CpuExecutionProvider::new();

        shapes[3][1] = H.saturating_add(1).into();
        let rejected = ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]);
        assert!(rejected.reason().unwrap().contains("fc1_experts_bias"));

        shapes[3][1] = H.into();
        shapes[6][1] = H.saturating_add(1).into();
        let rejected = ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]);
        assert!(rejected.reason().unwrap().contains("fc3_experts_weights"));

        shapes[6][1] = H.into();
        shapes[8][1] = E.saturating_add(1).into();
        let rejected = ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]);
        assert!(rejected.reason().unwrap().contains("router_weights"));
    }

    #[test]
    fn mixed_fc1_fc2_formats_match_per_projection_dense_reference() {
        // Real GLM-5.2 UD-IQ1_S routed experts pack gate/up (fc1) at IQ1_S and
        // down (fc2) at IQ3_XXS — different qtypes AND block byte widths on the
        // same node. The frozen single-format v1 ABI could not represent this;
        // each projection must decode with its own format.
        const HID: usize = 256;
        let experts = 2usize;
        let fc1_format = BlockFormat::Iq1S;
        let fc2_format = BlockFormat::Iq3Xxs;
        assert_ne!(
            fc1_format.block_bytes(),
            fc2_format.block_bytes(),
            "fixture must exercise different block widths per projection"
        );

        let fc1 = packed_for_format(fc1_format, experts, HID, HID);
        let fc2 = packed_for_format(fc2_format, experts, HID, HID);

        let mut spec = attrs("identity", 1, true, 0);
        spec[0] = ("fc1_format", Attribute::String(b"iq1_s".to_vec()));
        spec[1] = ("fc2_format", Attribute::String(b"iq3_xxs".to_vec()));
        let shapes = vec![
            Some((DataType::Float32, vec![1, HID])),
            Some((DataType::Float32, vec![1, experts])),
            Some((
                DataType::Uint8,
                vec![experts, HID, 1, fc1_format.block_bytes()],
            )),
            None,
            Some((
                DataType::Uint8,
                vec![experts, HID, 1, fc2_format.block_bytes()],
            )),
            None,
            None,
            None,
        ];
        let (graph, node) = model_node(&shapes, &spec);
        let ValidatedMetadata {
            attributes,
            formats,
        } = validate_metadata(graph.node(node), None).expect("valid mixed-format metadata");
        assert_eq!(formats.fc1, fc1_format);
        assert_eq!(formats.fc2, fc2_format);
        assert!(formats.fc3.is_none());

        let kernel = BlockQuantizedMoEKernel {
            attributes,
            formats,
            constant_inputs: [false; 9],
            weight_identities: std::array::from_fn(|_| DenseWeightIdentity::default()),
            weight_cache: DenseWeightCache::new(),
        };

        let input_values: Vec<f32> = (0..HID)
            .map(|index| ((index * 7 % 23) as f32 - 11.0) / 16.0)
            .collect();
        let input = Owned::f32(&[1, HID], &input_values);
        let logits = Owned::f32(&[1, experts], &[4.0, -4.0]);
        let fc1_view = Owned::u8(&[experts, HID, 1, fc1_format.block_bytes()], &fc1);
        let fc2_view = Owned::u8(&[experts, HID, 1, fc2_format.block_bytes()], &fc2);
        let views = [
            input.view(),
            logits.view(),
            fc1_view.view(),
            TensorView::absent(DataType::Float32),
            fc2_view.view(),
            TensorView::absent(DataType::Float32),
            TensorView::absent(DataType::Uint8),
            TensorView::absent(DataType::Float32),
        ];
        let mut output = Owned::f32(&[1, HID], &vec![0.0; HID]);
        kernel
            .execute(&views, &mut [output.view_mut()])
            .expect("execute mixed-format MoE");

        // Reference: expert 0 is the top-1 routed expert; decode each projection
        // with ITS OWN format from its own byte-exact expert bank derived from
        // the shared layout contract.
        let fc1_layout = ProjectionLayout::new(fc1_format, HID, HID, experts);
        let fc2_layout = ProjectionLayout::new(fc2_format, HID, HID, experts);
        let dense_fc1 = dequantize_expert_slice(
            fc1_format,
            &fc1[fc1_layout.expert_byte_range(0).unwrap()],
            HID,
            HID,
        )
        .unwrap();
        let dense_fc2 = dequantize_expert_slice(
            fc2_format,
            &fc2[fc2_layout.expert_byte_range(0).unwrap()],
            HID,
            HID,
        )
        .unwrap();
        let expected = run_expert_grouped(
            &input_values,
            1,
            &dense_fc1,
            None,
            &dense_fc2,
            None,
            None,
            None,
            HID,
            HID,
            HID,
            &attributes,
        )
        .expect("dense mixed-format reference");
        assert_close(&output.to_f32(), &expected);
    }

    #[test]
    fn mixed_unfused_gate_uses_independent_fc3_format() {
        // Unfused gated GLU: up (fc1), down (fc2) and the separate gate (fc3)
        // each carry an independent native qtype. Prove fc3 decodes with its
        // own format, not fc1's.
        const HID: usize = 256;
        let experts = 2usize;
        let fc1_format = BlockFormat::Iq1S; // up
        let fc2_format = BlockFormat::Iq3Xxs; // down
        let fc3_format = BlockFormat::Iq2Xxs; // gate — independent qtype
        let fc1 = packed_for_format(fc1_format, experts, HID, HID);
        let fc2 = packed_for_format(fc2_format, experts, HID, HID);
        let fc3 = packed_for_format(fc3_format, experts, HID, HID);

        let mut spec = attrs("swiglu", 1, true, 0);
        spec[0] = ("fc1_format", Attribute::String(b"iq1_s".to_vec()));
        spec[1] = ("fc2_format", Attribute::String(b"iq3_xxs".to_vec()));
        let spec = with_fc3_format(spec, "iq2_xxs");
        let shapes = vec![
            Some((DataType::Float32, vec![1, HID])),
            Some((DataType::Float32, vec![1, experts])),
            Some((
                DataType::Uint8,
                vec![experts, HID, 1, fc1_format.block_bytes()],
            )),
            None,
            Some((
                DataType::Uint8,
                vec![experts, HID, 1, fc2_format.block_bytes()],
            )),
            None,
            Some((
                DataType::Uint8,
                vec![experts, HID, 1, fc3_format.block_bytes()],
            )),
            None,
        ];
        let (graph, node) = model_node(&shapes, &spec);
        let ValidatedMetadata {
            attributes,
            formats,
        } = validate_metadata(graph.node(node), None).expect("valid mixed gated metadata");
        assert_eq!(formats.fc1, fc1_format);
        assert_eq!(formats.fc2, fc2_format);
        assert_eq!(formats.fc3, Some(fc3_format));

        let kernel = BlockQuantizedMoEKernel {
            attributes,
            formats,
            constant_inputs: [false; 9],
            weight_identities: std::array::from_fn(|_| DenseWeightIdentity::default()),
            weight_cache: DenseWeightCache::new(),
        };
        let input_values: Vec<f32> = (0..HID).map(|index| (index as f32) / 256.0 + 0.1).collect();
        let input = Owned::f32(&[1, HID], &input_values);
        let logits = Owned::f32(&[1, experts], &[4.0, -4.0]);
        let fc1_view = Owned::u8(&[experts, HID, 1, fc1_format.block_bytes()], &fc1);
        let fc2_view = Owned::u8(&[experts, HID, 1, fc2_format.block_bytes()], &fc2);
        let fc3_view = Owned::u8(&[experts, HID, 1, fc3_format.block_bytes()], &fc3);
        let views = [
            input.view(),
            logits.view(),
            fc1_view.view(),
            TensorView::absent(DataType::Float32),
            fc2_view.view(),
            TensorView::absent(DataType::Float32),
            fc3_view.view(),
            TensorView::absent(DataType::Float32),
        ];
        let mut output = Owned::f32(&[1, HID], &vec![0.0; HID]);
        kernel
            .execute(&views, &mut [output.view_mut()])
            .expect("execute mixed gated MoE");

        let fc1_layout = ProjectionLayout::new(fc1_format, HID, HID, experts);
        let fc2_layout = ProjectionLayout::new(fc2_format, HID, HID, experts);
        let fc3_layout = ProjectionLayout::new(fc3_format, HID, HID, experts);
        let dense_fc1 = dequantize_expert_slice(
            fc1_format,
            &fc1[fc1_layout.expert_byte_range(0).unwrap()],
            HID,
            HID,
        )
        .unwrap();
        let dense_fc2 = dequantize_expert_slice(
            fc2_format,
            &fc2[fc2_layout.expert_byte_range(0).unwrap()],
            HID,
            HID,
        )
        .unwrap();
        let dense_fc3 = dequantize_expert_slice(
            fc3_format,
            &fc3[fc3_layout.expert_byte_range(0).unwrap()],
            HID,
            HID,
        )
        .unwrap();
        let expected = run_expert_grouped(
            &input_values,
            1,
            &dense_fc1,
            None,
            &dense_fc2,
            None,
            Some(&dense_fc3),
            None,
            HID,
            HID,
            HID,
            &attributes,
        )
        .expect("dense mixed gated reference");
        assert_close(&output.to_f32(), &expected);
    }

    #[test]
    fn projection_layout_contract_is_byte_exact_and_rejects_tails() {
        // Byte offsets, per-row and per-expert strides all derive from one
        // contract, and expert banks tile the tensor with no gap/overlap.
        for (format, out, in_features) in [
            (BlockFormat::Iq1S, 3usize, 256usize),
            (BlockFormat::Iq3Xxs, 5, 512),
            (BlockFormat::Mxfp4, 4, 64),
            (BlockFormat::Iq4Nl, 2, 96),
        ] {
            let experts = 4usize;
            let layout = ProjectionLayout::new(format, out, in_features, experts);
            let blocks = in_features.div_ceil(format.qk());
            assert_eq!(
                layout.packed_shape(),
                [experts, out, blocks, format.block_bytes()]
            );
            assert_eq!(
                layout.row_stride_bytes().unwrap(),
                blocks * format.block_bytes()
            );
            assert_eq!(
                layout.expert_stride_bytes().unwrap(),
                out * blocks * format.block_bytes()
            );
            assert_eq!(
                layout.total_bytes().unwrap(),
                experts * out * blocks * format.block_bytes()
            );

            let stride = layout.expert_stride_bytes().unwrap();
            let mut prev_end = 0usize;
            for expert in 0..experts {
                let range = layout.expert_byte_range(expert).unwrap();
                assert_eq!(range.start, expert * stride);
                assert_eq!(range.end - range.start, stride);
                assert_eq!(range.start, prev_end, "expert banks must be contiguous");
                prev_end = range.end;
            }
            assert_eq!(
                prev_end,
                layout.total_bytes().unwrap(),
                "expert banks must cover the whole packed projection"
            );
        }

        for (format, width) in [(BlockFormat::Iq3Xxs, 257), (BlockFormat::Mxfp4, 33)] {
            let layout = ProjectionLayout::new(format, 2, width, 1);
            let error = validate_packed_shape(2, &layout.packed_shape(), layout).unwrap_err();
            assert!(error.to_string().contains("partial"));
        }
    }

    #[test]
    fn projection_layout_rejects_overflowing_dimensions() {
        // Offset/stride arithmetic is checked; degenerate dimensions must error
        // rather than wrap and read out of bounds.
        let layout = ProjectionLayout::new(BlockFormat::Iq1S, usize::MAX, usize::MAX, usize::MAX);
        assert!(layout.expert_stride_bytes().is_err());
        assert!(layout.total_bytes().is_err());
        assert!(layout.expert_byte_range(0).is_err());
    }

    #[test]
    fn glm52_native_formats_are_accepted_per_projection() {
        let shapes = vec![
            Some((DataType::Float32, vec![1, H])),
            Some((DataType::Float32, vec![1, E])),
            Some((DataType::Uint8, vec![E, H, 1, 17])),
            None,
            Some((DataType::Uint8, vec![E, H, 1, 17])),
            None,
            None,
            None,
        ];
        for native in ["q2_k", "q3_k", "q5_k", "q6_k", "q8_0"] {
            let mut fc1_spec = attrs("identity", 1, false, 0);
            fc1_spec[0] = ("fc1_format", Attribute::String(native.as_bytes().to_vec()));
            let (graph, node) = model_node(&shapes, &fc1_spec);
            validate_metadata(graph.node(node), None)
                .unwrap_or_else(|error| panic!("fc1 {native}: {error}"));

            let mut fc2_spec = attrs("identity", 1, false, 0);
            fc2_spec[1] = ("fc2_format", Attribute::String(native.as_bytes().to_vec()));
            let (graph, node) = model_node(&shapes, &fc2_spec);
            validate_metadata(graph.node(node), None)
                .unwrap_or_else(|error| panic!("fc2 {native}: {error}"));
        }
    }

    #[test]
    fn fc3_format_wiring_mismatch_is_typed_rejected() {
        // fc3 weights wired but fc3_format attribute missing.
        let wired = vec![
            Some((DataType::Float32, vec![1, H])),
            Some((DataType::Float32, vec![1, E])),
            Some((DataType::Uint8, vec![E, H, 1, 17])),
            None,
            Some((DataType::Uint8, vec![E, H, 1, 17])),
            None,
            Some((DataType::Uint8, vec![E, H, 1, 17])),
            None,
        ];
        let (graph, node) = model_node(&wired, &attrs("swiglu", 1, false, 0));
        let err = validate_metadata(graph.node(node), None)
            .expect_err("wired fc3 without fc3_format must be rejected");
        assert!(err.to_string().contains("fc3_format"), "{err}");

        // fc3_format attribute present but fc3 weights not wired.
        let unwired = vec![
            Some((DataType::Float32, vec![1, H])),
            Some((DataType::Float32, vec![1, E])),
            Some((DataType::Uint8, vec![E, H, 1, 17])),
            None,
            Some((DataType::Uint8, vec![E, H, 1, 17])),
            None,
            None,
            None,
        ];
        let (graph, node) = model_node(
            &unwired,
            &with_fc3_format(attrs("identity", 1, false, 0), "mxfp4"),
        );
        let err = validate_metadata(graph.node(node), None)
            .expect_err("fc3_format without wired fc3 must be rejected");
        assert!(
            err.to_string().contains("fc3_format is only valid"),
            "{err}"
        );
    }
}
