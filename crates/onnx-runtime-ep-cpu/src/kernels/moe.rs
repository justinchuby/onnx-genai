//! Float32 reference kernel for ORT 1.27 `com.microsoft::MoE`.
//!
//! The positional inputs are:
//! `input`, `router_probs`, `fc1_experts_weights`, `fc1_experts_bias?`,
//! `fc2_experts_weights`, `fc2_experts_bias?`, `fc3_experts_weights?`,
//! `fc3_experts_bias?`. Weights use ORT's expert-major canonical layout.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};
// Only the MLAS bucketing and the `#[cfg(test)]` reference algorithm still
// need an ordered map; the driver itself orders by expert through `RoutingPlan`.
#[cfg(any(feature = "mlas", test))]
use std::collections::BTreeMap;

use super::check_arity;
use super::gelu::tanh_gelu;
#[cfg(not(feature = "mlas"))]
use super::matmul::gemm;
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Activation {
    Relu,
    Gelu,
    Silu,
    Swiglu,
    Identity,
}

/// Factory for the ORT contrib `MoE` operator.
pub struct MoEFactory;

/// Float MoE kernel that groups routed rows by expert before expert GEMMs.
pub struct MoEKernel {
    attributes: MoeAttributes,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct MoeAttributes {
    pub k: usize,
    pub activation: Activation,
    pub normalize_routing_weights: bool,
    pub swiglu_fusion: usize,
    activation_alpha: f32,
    activation_beta: f32,
    swiglu_limit: f32,
}

impl KernelFactory for MoEFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let attributes = MoeAttributes::from_node(node)?;
        if int_attr(node, "block_size", 0)? != 0 {
            return Err(error(
                "block_size is a QMoE quantization attribute and is unsupported by float MoE",
            ));
        }
        Ok(Box::new(MoEKernel { attributes }))
    }
}

impl MoeAttributes {
    pub(super) fn from_node(node: &Node) -> Result<Self> {
        Self::from_node_impl(node, true)
    }

    pub(super) fn from_block_quantized_node(node: &Node) -> Result<Self> {
        Self::from_node_impl(node, false)
    }

    fn from_node_impl(node: &Node, parse_sparse_mixer: bool) -> Result<Self> {
        let k = int_attr(node, "k", 1)?;
        if k <= 0 {
            return Err(error(format!("k must be > 0, got {k}")));
        }
        let activation_name = match node.attr("activation_type") {
            Some(attr) => attr
                .as_str()
                .ok_or_else(|| error("attribute activation_type must be a string"))?,
            None => "relu",
        };
        let activation = match activation_name {
            "relu" => Activation::Relu,
            "gelu" => Activation::Gelu,
            "silu" => Activation::Silu,
            "swiglu" => Activation::Swiglu,
            "identity" => Activation::Identity,
            other => {
                return Err(error(format!(
                    "unsupported activation_type '{other}' (supported: relu, gelu, silu, swiglu, identity)"
                )));
            }
        };
        let normalize = bool_attr(node, "normalize_routing_weights", false)?;
        if parse_sparse_mixer && bool_attr(node, "use_sparse_mixer", false)? {
            return Err(error(
                "use_sparse_mixer=1 is unsupported by the Phase-1 CPU reference kernel",
            ));
        }
        let swiglu_fusion = int_attr(node, "swiglu_fusion", 0)?;
        if !(0..=2).contains(&swiglu_fusion) {
            return Err(error(format!(
                "swiglu_fusion must be 0, 1, or 2, got {swiglu_fusion}"
            )));
        }
        if activation != Activation::Swiglu && swiglu_fusion != 0 {
            return Err(error(
                "swiglu_fusion is only valid when activation_type='swiglu'",
            ));
        }
        Ok(Self {
            k: k as usize,
            activation,
            normalize_routing_weights: normalize,
            swiglu_fusion: swiglu_fusion as usize,
            activation_alpha: float_attr(node, "activation_alpha", 1.0)?,
            activation_beta: float_attr(node, "activation_beta", 0.0)?,
            swiglu_limit: float_attr(node, "swiglu_limit", f32::INFINITY)?,
        })
    }

    fn swiglu(&self, gate: f32, linear: f32) -> f32 {
        let g = gate.min(self.swiglu_limit);
        let l = linear.clamp(-self.swiglu_limit, self.swiglu_limit);
        g * sigmoid(self.activation_alpha * g) * (l + self.activation_beta)
    }
}

impl Kernel for MoEKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("MoE", inputs, outputs, 5, 8, 1)?;
        if outputs.len() != 1 {
            return Err(error(format!(
                "expected exactly 1 output, got {}",
                outputs.len()
            )));
        }
        for (index, name) in [
            (0, "input"),
            (1, "router_probs"),
            (2, "fc1_experts_weights"),
            (4, "fc2_experts_weights"),
        ] {
            if inputs[index].is_absent() {
                return Err(error(format!(
                    "required input {index} ('{name}') is absent"
                )));
            }
        }
        for (index, input) in inputs.iter().enumerate().filter(|(_, v)| !v.is_absent()) {
            if !matches!(
                input.dtype,
                DataType::Float32 | DataType::Float16 | DataType::BFloat16
            ) {
                return Err(error(format!(
                    "input {index} requires Float32, Float16, or BFloat16, got {:?}",
                    input.dtype
                )));
            }
            if input.dtype != inputs[0].dtype {
                return Err(error(format!(
                    "input {index} dtype {:?} must match input 0 dtype {:?}",
                    input.dtype, inputs[0].dtype
                )));
            }
        }
        if !matches!(
            outputs[0].dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            return Err(error(format!(
                "output requires Float32, Float16, or BFloat16, got {:?}",
                outputs[0].dtype
            )));
        }

        let x_shape = inputs[0].shape;
        if !matches!(x_shape.len(), 2 | 3) {
            return Err(error(format!(
                "input must be 2-D [rows, hidden] or 3-D [batch, sequence, hidden], got {x_shape:?}"
            )));
        }
        if outputs[0].shape != x_shape {
            return Err(error(format!(
                "output shape {:?} must equal input shape {x_shape:?}",
                outputs[0].shape
            )));
        }
        let hidden = *x_shape.last().unwrap();
        let rows = x_shape[..x_shape.len() - 1].iter().product::<usize>();
        require_shape("router_probs", inputs[1].shape, 2)?;
        if inputs[1].shape[0] != rows {
            return Err(error(format!(
                "router_probs rows {} must equal flattened input rows {rows}",
                inputs[1].shape[0]
            )));
        }
        let experts = inputs[1].shape[1];
        if self.attributes.k > experts {
            return Err(error(format!(
                "requires 0 < k <= num_experts, got k={} and num_experts={experts}",
                self.attributes.k
            )));
        }

        require_shape("fc1_experts_weights", inputs[2].shape, 3)?;
        require_shape("fc2_experts_weights", inputs[4].shape, 3)?;
        if inputs[2].shape[0] != experts || inputs[4].shape[0] != experts {
            return Err(error(format!(
                "expert weight counts must equal router num_experts {experts}"
            )));
        }
        if inputs[2].shape[2] != hidden {
            return Err(error(format!(
                "fc1_experts_weights must have canonical shape [experts, fc1_size, hidden={hidden}], got {:?}",
                inputs[2].shape
            )));
        }
        if inputs[4].shape[1] != hidden {
            return Err(error(format!(
                "fc2_experts_weights must have canonical shape [experts, hidden={hidden}, inter_size], got {:?}",
                inputs[4].shape
            )));
        }
        let inter = inputs[4].shape[2];
        let expected_fc1 = self.attributes.checked_fc1_size(inter, "MoE")?;
        if inputs[2].shape[1] != expected_fc1 {
            return Err(error(format!(
                "fc1_experts_weights dimension 1 must be {expected_fc1}, got {}",
                inputs[2].shape[1]
            )));
        }

        let fc1_bias = optional_dense(inputs, 3)?;
        let fc2_bias = optional_dense(inputs, 5)?;
        validate_bias("fc1_experts_bias", inputs, 3, experts, expected_fc1)?;
        validate_bias("fc2_experts_bias", inputs, 5, experts, hidden)?;

        let has_fc3 = inputs.get(6).is_some_and(|v| !v.is_absent());
        let uses_separate_gate = self.attributes.uses_separate_gate(has_fc3);
        let (fc3, fc3_bias) = if uses_separate_gate {
            let view = inputs
                .get(6)
                .filter(|v| !v.is_absent())
                .ok_or_else(|| error("unfused swiglu requires input 6 fc3_experts_weights"))?;
            require_exact_shape("fc3_experts_weights", view.shape, &[experts, inter, hidden])?;
            validate_bias("fc3_experts_bias", inputs, 7, experts, inter)?;
            (
                Some(to_dense_f32_widen("MoE", view)?),
                optional_dense(inputs, 7)?,
            )
        } else {
            if has_fc3 {
                return Err(error(
                    "fc3_experts_weights is only valid for unfused swiglu or silu gated-GLU",
                ));
            }
            if inputs.get(7).is_some_and(|v| !v.is_absent()) {
                return Err(error(
                    "fc3_experts_bias requires fc3_experts_weights in unfused swiglu or silu gated-GLU",
                ));
            }
            (None, None)
        };

        // Borrow rather than `into_owned()`. Expert weights are the largest
        // tensors in the graph - a Mixtral-shaped `fc1` + `fc2` pair is 352 MiB
        // - and they are contiguous f32 initializers in every real model, so
        // owning them copied a third of a gigabyte on every single forward.
        // `to_dense_f32_widen` still allocates for half-precision or strided
        // sources, which is the only case that actually needs a copy.
        let x = to_dense_f32_widen("MoE", &inputs[0])?;
        let router = to_dense_f32_widen("MoE", &inputs[1])?;
        let fc1 = to_dense_f32_widen("MoE", &inputs[2])?;
        let fc2 = to_dense_f32_widen("MoE", &inputs[4])?;
        let mut plan = RoutingPlan::build(
            &router,
            rows,
            experts,
            self.attributes.k,
            self.attributes.normalize_routing_weights,
        );
        let driver = if use_grouped_driver(&plan, expected_fc1, hidden, inter) {
            plan.build_row_slots();
            run_moe_grouped
        } else {
            run_moe_per_expert
        };
        let output = driver(
            &plan,
            &x,
            &fc1,
            fc1_bias.as_deref(),
            &fc2,
            fc2_bias.as_deref(),
            fc3.as_deref(),
            fc3_bias.as_deref(),
            expected_fc1,
            hidden,
            inter,
            &self.attributes,
        )?;
        write_dense_f32_narrow("MoE", &mut outputs[0], &output)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

impl MoeAttributes {
    pub(super) fn checked_fc1_size(&self, inter: usize, op: &str) -> Result<usize> {
        if self.activation == Activation::Swiglu && self.swiglu_fusion != 0 {
            inter.checked_mul(2).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "{op}: fused SwiGLU FC1 width overflow for inter_size {inter}"
                ))
            })
        } else {
            Ok(inter)
        }
    }

    pub(super) fn uses_separate_gate(&self, has_fc3: bool) -> bool {
        (self.activation == Activation::Swiglu && self.swiglu_fusion == 0)
            || (self.activation == Activation::Silu && has_fc3)
    }
}

/// Rows below this many elements stay on the calling thread: a `rayon`
/// fan-out costs more than a few thousand floats of memcpy or activation.
const MIN_PARALLEL_MOE_ELEMENTS: usize = 16 * 1024;

/// Apply `f` to every `width`-wide row of `data`, in parallel once the pass is
/// large enough to pay for the fan-out.
///
/// Every stage between the two expert GEMMs - the gather, the bias adds, the
/// activation and the weighted scatter - is a row-wise map, and each was a
/// serial scalar loop before. On a 512-token Mixtral-shaped forward those
/// stages touch more bytes than the GEMMs do, so leaving them serial capped
/// the whole operator's thread scaling regardless of how well MLAS threaded
/// the GEMMs themselves.
fn for_each_row<F>(data: &mut [f32], width: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Send + Sync,
{
    if width == 0 {
        return;
    }
    let rows = data.len() / width;
    match parallel_rows_per_task(rows, width) {
        Some(rows_per_task) => {
            use rayon::prelude::*;
            data.par_chunks_mut(rows_per_task * width)
                .enumerate()
                .for_each(|(chunk, block)| {
                    let base = chunk * rows_per_task;
                    for (offset, row) in block.chunks_mut(width).enumerate() {
                        f(base + offset, row);
                    }
                });
        }
        None => {
            for (index, row) in data.chunks_mut(width).enumerate() {
                f(index, row);
            }
        }
    }
}

/// Whole rows per `rayon` task, or `None` to stay on the calling thread.
///
/// The task count is capped at the width the GEMMs themselves will use, not at
/// `rayon`'s global pool size. `ONNX_GENAI_CPU_DECODE_THREADS` bounds MLAS but
/// not `rayon`, so fanning these passes out to every logical CPU would run the
/// operator wider than the caller asked for - and, measured against a real ORT
/// session configured with the same width, would not be a like-for-like
/// comparison.
fn parallel_rows_per_task(rows: usize, width: usize) -> Option<usize> {
    if rows < 2 || rows.saturating_mul(width) < MIN_PARALLEL_MOE_ELEMENTS {
        return None;
    }
    let workers = moe_worker_budget().min(rows);
    if workers < 2 {
        return None;
    }
    Some(rows.div_ceil(workers).max(1))
}

fn moe_worker_budget() -> usize {
    let rayon_workers = rayon::current_num_threads().max(1);
    #[cfg(feature = "mlas")]
    {
        rayon_workers.min(mlas_sys::configured_pool_threads())
    }
    #[cfg(not(feature = "mlas"))]
    {
        rayon_workers
    }
}

/// One contiguous run of routed rows that share an expert.
#[derive(Clone, Copy, Debug)]
struct ExpertGroup {
    expert: usize,
    /// First slot of the run in the routed-row ordering.
    start: usize,
    /// Number of routed rows in the run.
    len: usize,
}

/// Top-k routing flattened into a slot ordering: all rows routed to the lowest
/// active expert first, then the next, and so on.
///
/// Laying the routed rows out this way is what lets every later stage be a flat
/// row-wise map over one buffer, and it makes each expert's GEMM operand a
/// contiguous window - the precondition for handing several experts to MLAS as
/// a single batched GEMM.
struct RoutingPlan {
    rows: usize,
    k: usize,
    /// Source row for each slot.
    slot_row: Vec<u32>,
    /// Expert serving each slot, for the per-expert bias lookup.
    slot_expert: Vec<u32>,
    /// Routing weight for each slot.
    slot_weight: Vec<f32>,
    /// Slots serving each row, `k` entries per row, ascending by expert.
    /// `u32::MAX` marks an unused entry.
    row_slots: Vec<u32>,
    groups: Vec<ExpertGroup>,
}

impl RoutingPlan {
    fn build(router: &[f32], rows: usize, experts: usize, k: usize, normalize: bool) -> Self {
        let mut per_expert: Vec<Vec<(u32, f32)>> = vec![Vec::new(); experts];
        for row in 0..rows {
            for (expert, weight) in routing_weights(
                &router[row * experts..(row + 1) * experts],
                None,
                k,
                normalize,
            ) {
                per_expert[expert].push((row as u32, weight));
            }
        }
        let total = per_expert.iter().map(Vec::len).sum::<usize>();
        let mut slot_row = Vec::with_capacity(total);
        let mut slot_expert = Vec::with_capacity(total);
        let mut slot_weight = Vec::with_capacity(total);
        let mut groups = Vec::new();
        for (expert, assigned) in per_expert.iter().enumerate() {
            if assigned.is_empty() {
                continue;
            }
            groups.push(ExpertGroup {
                expert,
                start: slot_row.len(),
                len: assigned.len(),
            });
            for &(row, weight) in assigned {
                slot_row.push(row);
                slot_expert.push(expert as u32);
                slot_weight.push(weight);
            }
        }
        Self {
            rows,
            k,
            slot_row,
            slot_expert,
            slot_weight,
            row_slots: Vec::new(),
            groups,
        }
    }

    /// Invert the layout so the final scatter can be a parallel map over
    /// *output* rows: two experts share an output row, so scattering by expert
    /// would race.
    ///
    /// Only the grouped driver scatters that way, and this is an `rows * k`
    /// allocation, so the per-expert driver does not pay for it.
    fn build_row_slots(&mut self) {
        self.row_slots = vec![u32::MAX; self.rows * self.k];
        let mut filled = vec![0usize; self.rows];
        for (slot, &row) in self.slot_row.iter().enumerate() {
            let row = row as usize;
            self.row_slots[row * self.k + filled[row]] = slot as u32;
            filled[row] += 1;
        }
    }

    fn slots(&self) -> usize {
        self.slot_row.len()
    }
}

/// `output[slot, out_features] = input[slot, in_features] * weights[expert]ᵀ`
/// for every routed slot.
///
/// Experts whose groups hold the same number of rows are issued as one batched
/// MLAS GEMM. A decode step is the extreme case: `k` groups of exactly one row
/// each, which individually look far too small for MLAS to thread and pay a
/// full dispatch each.
///
/// The per-expert bias is deliberately **not** applied here. Both biases fold
/// into a pass that already reads the result - FC1's into the activation,
/// FC2's into the weighted scatter - which removes two full read-modify-write
/// passes over the largest intermediates in the operator.
fn grouped_linear(
    input: &[f32],
    plan: &RoutingPlan,
    weights: &[f32],
    out_features: usize,
    in_features: usize,
    output: &mut [f32],
) -> Result<()> {
    let expert_stride = out_features * in_features;
    #[cfg(feature = "mlas")]
    {
        let mut by_len: BTreeMap<usize, Vec<&ExpertGroup>> = BTreeMap::new();
        for group in &plan.groups {
            by_len.entry(group.len).or_default().push(group);
        }
        for (len, groups) in by_len {
            if len == 0 || out_features == 0 || in_features == 0 {
                continue;
            }
            let items: Vec<mlas_sys::SgemmBatchItem<'_>> = groups
                .iter()
                .map(|group| mlas_sys::SgemmBatchItem {
                    a: &input[group.start * in_features..],
                    b: &weights[group.expert * expert_stride..(group.expert + 1) * expert_stride],
                    c_offset: group.start * out_features,
                })
                .collect();
            mlas_sys::sgemm_batch(
                false,
                true,
                len,
                out_features,
                in_features,
                1.0,
                &items,
                in_features,
                in_features,
                0.0,
                output,
                out_features,
            );
        }
    }
    #[cfg(not(feature = "mlas"))]
    {
        for group in &plan.groups {
            let mut weights_kn = vec![0.0f32; expert_stride];
            let expert = &weights[group.expert * expert_stride..(group.expert + 1) * expert_stride];
            for output_feature in 0..out_features {
                for input_feature in 0..in_features {
                    weights_kn[input_feature * out_features + output_feature] =
                        expert[output_feature * in_features + input_feature];
                }
            }
            gemm(
                &input[group.start * in_features..(group.start + group.len) * in_features],
                &weights_kn,
                &mut output[group.start * out_features..(group.start + group.len) * out_features],
                group.len,
                in_features,
                out_features,
            )?;
        }
    }
    Ok(())
}

/// Average GEMM work per expert group, in multiply-accumulates, below which
/// the grouped driver is *not* used.
///
/// Measured, not derived. Against a real ORT CPU session on 9 synthetic
/// production-shaped MoE graphs x 3 thread counts, the grouped driver wins
/// decisively once each expert's own GEMMs are large (1.35x-1.57x less native
/// time at 512 tokens) and loses by 5-16% when they are small: at that size
/// MLAS's per-GEMM threading is already the limit, and the extra gather and
/// scatter buffers cost more than parallelising the elementwise stages saves.
/// The measured separation was 7.6e7 (losing) to 6.3e8 (winning) work units,
/// so the floor sits near the geometric mean of that band.
///
/// This is a one-host calibration on an AMD EPYC 9V74 (16C/32T, 32 MiB L3).
/// `ONNX_GENAI_MOE_GROUPED_MIN_WORK` overrides it; `0` forces the grouped
/// driver on for every shape.
const MOE_GROUPED_MIN_WORK: u64 = 300_000_000;

/// Whether the grouped/batched driver is expected to beat the per-expert loop.
///
/// The discriminator is the work in *one* expert's GEMMs, not the total: the
/// grouped driver's win comes from parallelising the gather, activation and
/// scatter, which only matters once MLAS is already saturating the machine
/// inside each expert's GEMM. `slots / groups` is the average rows per expert,
/// and `fc1_size + inter` counts both the FC1(+FC3) and FC2 passes.
fn use_grouped_driver(plan: &RoutingPlan, fc1_size: usize, hidden: usize, inter: usize) -> bool {
    let groups = plan.groups.len();
    if groups == 0 {
        return true;
    }
    if moe_worker_budget() < 2 {
        // Single-threaded: the two drivers measured identically, so prefer the
        // one with fewer intermediate buffers.
        return false;
    }
    grouped_work_units(plan, fc1_size, hidden, inter) >= grouped_min_work()
}

/// The gate's work estimate, split out so it can be pinned by a test on a
/// machine of any width.
///
/// The largest group rather than the mean: a collapsed router that sends most
/// of its rows to one expert still has one large GEMM to hide the elementwise
/// stages behind, and the mean would hide it.
fn grouped_work_units(plan: &RoutingPlan, fc1_size: usize, hidden: usize, inter: usize) -> u64 {
    let rows_per_group = plan.groups.iter().map(|group| group.len).max().unwrap_or(0) as u64;
    rows_per_group
        .saturating_mul(hidden as u64)
        .saturating_mul((fc1_size + inter) as u64)
}

fn grouped_min_work() -> u64 {
    // Deliberately not cached: the read is once per operator call, next to a
    // multi-millisecond GEMM, and caching it in a `OnceLock` would let whichever
    // test ran first freeze the value for the whole process.
    std::env::var("ONNX_GENAI_MOE_GROUPED_MIN_WORK")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(MOE_GROUPED_MIN_WORK)
}

/// The per-expert driver: gather each expert's rows into a fresh buffer, run
/// the pair of expert GEMMs, then accumulate the weighted result.
///
/// Kept as the small-shape path. It allocates per expert and leaves the
/// gather, activation and scatter serial, which is exactly the trade that wins
/// when each expert only has a handful of rows.
///
/// Iterating `plan.groups` reproduces the ascending-expert order, and slots
/// inside a group are in ascending row order, so this is **bit-identical** to
/// the grouped driver rather than merely close.
#[allow(clippy::too_many_arguments)]
fn run_moe_per_expert(
    plan: &RoutingPlan,
    x: &[f32],
    fc1_weights: &[f32],
    fc1_bias: Option<&[f32]>,
    fc2_weights: &[f32],
    fc2_bias: Option<&[f32]>,
    fc3_weights: Option<&[f32]>,
    fc3_bias: Option<&[f32]>,
    fc1_size: usize,
    hidden: usize,
    inter: usize,
    attributes: &MoeAttributes,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; plan.rows * hidden];
    if plan.slots() == 0 || hidden == 0 {
        return Ok(output);
    }
    for group in &plan.groups {
        let expert = group.expert;
        let slots = group.start..group.start + group.len;
        let mut grouped_input = Vec::with_capacity(group.len * hidden);
        for slot in slots.clone() {
            let source = plan.slot_row[slot] as usize * hidden;
            grouped_input.extend_from_slice(&x[source..source + hidden]);
        }
        let expert_out = run_expert_grouped(
            &grouped_input,
            group.len,
            &fc1_weights[expert * fc1_size * hidden..(expert + 1) * fc1_size * hidden],
            fc1_bias.map(|b| &b[expert * fc1_size..(expert + 1) * fc1_size]),
            &fc2_weights[expert * hidden * inter..(expert + 1) * hidden * inter],
            fc2_bias.map(|b| &b[expert * hidden..(expert + 1) * hidden]),
            fc3_weights.map(|w| &w[expert * inter * hidden..(expert + 1) * inter * hidden]),
            fc3_bias.map(|b| &b[expert * inter..(expert + 1) * inter]),
            fc1_size,
            hidden,
            inter,
            attributes,
        )?;
        for (local, slot) in slots.enumerate() {
            let row = plan.slot_row[slot] as usize;
            let weight = plan.slot_weight[slot];
            let src = &expert_out[local * hidden..(local + 1) * hidden];
            let dst = &mut output[row * hidden..(row + 1) * hidden];
            for (out, &value) in dst.iter_mut().zip(src) {
                *out += weight * value;
            }
        }
    }
    Ok(output)
}

/// Run every routed row through its expert and scatter the weighted results
/// back onto the output rows.
///
/// This is the whole operator between routing and the output write: gather,
/// FC1 (+ optional FC3 gate), activation, FC2, weighted accumulation. Each
/// stage is one pass over one buffer, so the only serial work left is the
/// routing itself.
#[allow(clippy::too_many_arguments)]
fn run_moe_grouped(
    plan: &RoutingPlan,
    x: &[f32],
    fc1_weights: &[f32],
    fc1_bias: Option<&[f32]>,
    fc2_weights: &[f32],
    fc2_bias: Option<&[f32]>,
    fc3_weights: Option<&[f32]>,
    fc3_bias: Option<&[f32]>,
    fc1_size: usize,
    hidden: usize,
    inter: usize,
    attributes: &MoeAttributes,
) -> Result<Vec<f32>> {
    let slots = plan.slots();
    let mut output = vec![0.0f32; plan.rows * hidden];
    if slots == 0 || hidden == 0 {
        return Ok(output);
    }
    debug_assert_eq!(
        plan.row_slots.len(),
        plan.rows * plan.k,
        "run_moe_grouped needs the row->slot inversion; call build_row_slots first"
    );

    let mut gathered = vec![0.0f32; slots * hidden];
    let slot_row = &plan.slot_row;
    for_each_row(&mut gathered, hidden, |slot, row| {
        let source = slot_row[slot] as usize * hidden;
        row.copy_from_slice(&x[source..source + hidden]);
    });

    let mut fc1_out = vec![0.0f32; slots * fc1_size];
    grouped_linear(&gathered, plan, fc1_weights, fc1_size, hidden, &mut fc1_out)?;

    let gate_source = if attributes.uses_separate_gate(fc3_weights.is_some()) {
        let mut fc3_out = vec![0.0f32; slots * inter];
        grouped_linear(
            &gathered,
            plan,
            fc3_weights.expect("validated separate gate FC3"),
            inter,
            hidden,
            &mut fc3_out,
        )?;
        Some(fc3_out)
    } else {
        None
    };
    drop(gathered);

    let activated = apply_activation(
        fc1_out,
        gate_source,
        plan,
        fc1_bias,
        fc3_bias,
        slots,
        fc1_size,
        inter,
        attributes,
    );

    let mut expert_out = vec![0.0f32; slots * hidden];
    grouped_linear(
        &activated,
        plan,
        fc2_weights,
        hidden,
        inter,
        &mut expert_out,
    )?;
    drop(activated);

    let (row_slots, slot_weight, slot_expert, k) = (
        &plan.row_slots,
        &plan.slot_weight,
        &plan.slot_expert,
        plan.k,
    );
    for_each_row(&mut output, hidden, |row, destination| {
        for &slot in &row_slots[row * k..(row + 1) * k] {
            if slot == u32::MAX {
                continue;
            }
            let slot = slot as usize;
            let weight = slot_weight[slot];
            let contribution = &expert_out[slot * hidden..(slot + 1) * hidden];
            match fc2_bias {
                Some(bias) => {
                    let expert = slot_expert[slot] as usize;
                    let bias = &bias[expert * hidden..(expert + 1) * hidden];
                    for ((value, &source), &bias) in
                        destination.iter_mut().zip(contribution).zip(bias)
                    {
                        *value += weight * (source + bias);
                    }
                }
                None => {
                    for (value, &source) in destination.iter_mut().zip(contribution) {
                        *value += weight * source;
                    }
                }
            }
        }
    });
    Ok(output)
}

/// Collapse FC1 (and the optional FC3 gate) into the `inter`-wide activated
/// tensor the second expert GEMM consumes.
///
/// The gated variants used to build the gate and linear halves as two fresh
/// `Vec`s element by element and then a third for the result. For a
/// 512-token Mixtral-shaped forward that is ~56 MiB of allocation and three
/// serial passes; here it is one parallel pass that writes each half directly
/// into the buffer it will be consumed from.
#[allow(clippy::too_many_arguments)]
fn apply_activation(
    mut fc1_out: Vec<f32>,
    gate_source: Option<Vec<f32>>,
    plan: &RoutingPlan,
    fc1_bias: Option<&[f32]>,
    fc3_bias: Option<&[f32]>,
    slots: usize,
    fc1_size: usize,
    inter: usize,
    attributes: &MoeAttributes,
) -> Vec<f32> {
    let slot_expert = &plan.slot_expert;
    fn expert_bias<'a>(
        bias: Option<&'a [f32]>,
        slot_expert: &[u32],
        slot: usize,
        width: usize,
    ) -> Option<&'a [f32]> {
        bias.map(|bias| {
            let expert = slot_expert[slot] as usize;
            &bias[expert * width..(expert + 1) * width]
        })
    }
    let gated = attributes.activation == Activation::Swiglu
        || (attributes.activation == Activation::Silu && gate_source.is_some());
    if !gated {
        for_each_row(&mut fc1_out, fc1_size, |slot, row| {
            let bias = expert_bias(fc1_bias, slot_expert, slot, fc1_size);
            for (index, value) in row.iter_mut().enumerate() {
                let biased = *value + bias.map_or(0.0, |bias| bias[index]);
                *value = activate(attributes.activation, biased);
            }
        });
        return fc1_out;
    }
    let mut activated = vec![0.0f32; slots * inter];
    match gate_source {
        // Unfused gate: FC1 is the gate, FC3 the linear half.
        Some(linear_part) => {
            for_each_row(&mut activated, inter, |slot, row| {
                let gate = &fc1_out[slot * inter..(slot + 1) * inter];
                let linear = &linear_part[slot * inter..(slot + 1) * inter];
                let gate_bias = expert_bias(fc1_bias, slot_expert, slot, inter);
                let linear_bias = expert_bias(fc3_bias, slot_expert, slot, inter);
                for (index, value) in row.iter_mut().enumerate() {
                    let g = gate[index] + gate_bias.map_or(0.0, |bias| bias[index]);
                    let l = linear[index] + linear_bias.map_or(0.0, |bias| bias[index]);
                    *value = attributes.swiglu(g, l);
                }
            });
        }
        // Fused gate: FC1 is `2 * inter` wide, interleaved when
        // `swiglu_fusion == 1` and split in halves when it is 2. The bias has
        // the same width and the same interleaving.
        None => {
            let interleaved = attributes.swiglu_fusion == 1;
            for_each_row(&mut activated, inter, |slot, row| {
                let source = &fc1_out[slot * fc1_size..(slot + 1) * fc1_size];
                let bias = expert_bias(fc1_bias, slot_expert, slot, fc1_size);
                for (index, value) in row.iter_mut().enumerate() {
                    let (gate_index, linear_index) = if interleaved {
                        (2 * index, 2 * index + 1)
                    } else {
                        (index, inter + index)
                    };
                    let g = source[gate_index] + bias.map_or(0.0, |bias| bias[gate_index]);
                    let l = source[linear_index] + bias.map_or(0.0, |bias| bias[linear_index]);
                    *value = attributes.swiglu(g, l);
                }
            });
        }
    }
    activated
}

/// Portable single-token reference expert: plain scalar dot products, no GEMM,
/// no transposes. Production now routes every row count through
/// [`run_expert_grouped`]; this is kept as the oracle the tests compare
/// against.
#[cfg(test)]
pub(super) fn run_expert(
    input: &[f32],
    fc1_weights: &[f32],
    fc1_bias: Option<&[f32]>,
    fc2_weights: &[f32],
    fc2_bias: Option<&[f32]>,
    fc3_weights: Option<&[f32]>,
    fc3_bias: Option<&[f32]>,
    fc1_size: usize,
    hidden: usize,
    inter: usize,
    attributes: &MoeAttributes,
) -> Vec<f32> {
    let mut fc1_out = linear(input, fc1_weights, fc1_bias, fc1_size, hidden);
    let activated = match attributes.activation {
        Activation::Swiglu => {
            let linear_part;
            let gate_part;
            if attributes.swiglu_fusion == 0 {
                gate_part = fc1_out;
                linear_part = linear(
                    input,
                    fc3_weights.expect("validated unfused swiglu FC3"),
                    fc3_bias,
                    inter,
                    hidden,
                );
            } else if attributes.swiglu_fusion == 1 {
                let mut gate = Vec::with_capacity(inter);
                let mut linear = Vec::with_capacity(inter);
                for pair in fc1_out.chunks_exact(2) {
                    gate.push(pair[0]);
                    linear.push(pair[1]);
                }
                gate_part = gate;
                linear_part = linear;
            } else {
                linear_part = fc1_out.split_off(inter);
                gate_part = fc1_out;
            }
            gate_part
                .into_iter()
                .zip(linear_part)
                .map(|(g, l)| attributes.swiglu(g, l))
                .collect()
        }
        Activation::Silu if fc3_weights.is_some() => {
            let linear_part = linear(
                input,
                fc3_weights.expect("validated SiLU gated FC3"),
                fc3_bias,
                inter,
                hidden,
            );
            fc1_out
                .into_iter()
                .zip(linear_part)
                .map(|(g, l)| attributes.swiglu(g, l))
                .collect()
        }
        activation => {
            for value in &mut fc1_out {
                *value = activate(activation, *value);
            }
            fc1_out
        }
    };
    linear(&activated, fc2_weights, fc2_bias, hidden, inter)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_expert_grouped(
    input: &[f32],
    rows: usize,
    fc1_weights: &[f32],
    fc1_bias: Option<&[f32]>,
    fc2_weights: &[f32],
    fc2_bias: Option<&[f32]>,
    fc3_weights: Option<&[f32]>,
    fc3_bias: Option<&[f32]>,
    fc1_size: usize,
    hidden: usize,
    inter: usize,
    attributes: &MoeAttributes,
) -> Result<Vec<f32>> {
    // A single routed token used to fall through to `run_expert`'s scalar dot
    // loop. That loop is ~50x slower per MAC than the GEMM, and decode - one
    // token, top-k experts - is the shape that matters most, so the grouped
    // path now serves `rows == 1` too. `run_expert` is retained as the
    // portable reference the tests compare against.
    let mut fc1_out = linear_grouped(input, rows, fc1_weights, fc1_bias, fc1_size, hidden)?;
    let activated = match attributes.activation {
        Activation::Swiglu => {
            let linear_part;
            let gate_part;
            if attributes.swiglu_fusion == 0 {
                gate_part = fc1_out;
                linear_part = linear_grouped(
                    input,
                    rows,
                    fc3_weights.expect("validated unfused swiglu FC3"),
                    fc3_bias,
                    inter,
                    hidden,
                )?;
            } else {
                let mut gate = Vec::with_capacity(rows * inter);
                let mut linear = Vec::with_capacity(rows * inter);
                for row in fc1_out.chunks_exact(fc1_size) {
                    if attributes.swiglu_fusion == 1 {
                        for pair in row.chunks_exact(2) {
                            gate.push(pair[0]);
                            linear.push(pair[1]);
                        }
                    } else {
                        gate.extend_from_slice(&row[..inter]);
                        linear.extend_from_slice(&row[inter..]);
                    }
                }
                gate_part = gate;
                linear_part = linear;
            }
            gate_part
                .into_iter()
                .zip(linear_part)
                .map(|(g, l)| attributes.swiglu(g, l))
                .collect()
        }
        Activation::Silu if fc3_weights.is_some() => {
            let linear_part = linear_grouped(
                input,
                rows,
                fc3_weights.expect("validated SiLU gated FC3"),
                fc3_bias,
                inter,
                hidden,
            )?;
            fc1_out
                .into_iter()
                .zip(linear_part)
                .map(|(g, l)| attributes.swiglu(g, l))
                .collect()
        }
        activation => {
            for value in &mut fc1_out {
                *value = activate(activation, *value);
            }
            fc1_out
        }
    };
    linear_grouped(&activated, rows, fc2_weights, fc2_bias, hidden, inter)
}

/// `output[rows, out_features] = input[rows, in_features] * weightsᵀ (+ bias)`.
///
/// Expert weights arrive as `[out_features, in_features]`, i.e. already
/// transposed relative to what a plain `A*B` GEMM wants. MLAS takes `transB`
/// directly, so the fast path hands it the weights as they are. The portable
/// path still has to materialize `[in_features, out_features]`, which for a
/// Mixtral-sized expert is a 29 MiB strided scatter per call - the single
/// largest term in the whole operator before this.
fn linear_grouped(
    input: &[f32],
    rows: usize,
    weights_nk: &[f32],
    bias: Option<&[f32]>,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; rows * out_features];
    #[cfg(feature = "mlas")]
    {
        if rows > 0 && out_features > 0 && in_features > 0 {
            mlas_sys::sgemm(
                false,
                true,
                rows,
                out_features,
                in_features,
                1.0,
                input,
                in_features,
                weights_nk,
                in_features,
                0.0,
                &mut output,
                out_features,
            );
        }
    }
    #[cfg(not(feature = "mlas"))]
    {
        let mut weights_kn = vec![0.0f32; weights_nk.len()];
        for output_feature in 0..out_features {
            for input_feature in 0..in_features {
                weights_kn[input_feature * out_features + output_feature] =
                    weights_nk[output_feature * in_features + input_feature];
            }
        }
        gemm(
            input,
            &weights_kn,
            &mut output,
            rows,
            in_features,
            out_features,
        )?;
    }
    if let Some(bias) = bias {
        for row in output.chunks_exact_mut(out_features) {
            for (value, bias) in row.iter_mut().zip(bias) {
                *value += bias;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
fn linear(
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    out_features: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; out_features];
    for o in 0..out_features {
        let mut sum = bias.map_or(0.0, |b| b[o]);
        for i in 0..in_features {
            sum += input[i] * weights[o * in_features + i];
        }
        output[o] = sum;
    }
    output
}

pub(super) fn routing_weights(
    logits: &[f32],
    aggregation_weights: Option<&[f32]>,
    k: usize,
    normalize: bool,
) -> Vec<(usize, f32)> {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));
    indices.truncate(k);
    if let Some(weights) = aggregation_weights {
        let denominator = if normalize {
            indices.iter().map(|&i| weights[i]).sum()
        } else {
            1.0
        };
        return indices
            .into_iter()
            .map(|i| {
                let weight = if denominator == 0.0 {
                    0.0
                } else {
                    weights[i] / denominator
                };
                (i, weight)
            })
            .collect();
    }

    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let all_sum: f32 = exponentials.iter().sum();
    let denominator = if normalize {
        indices.iter().map(|&i| exponentials[i]).sum()
    } else {
        all_sum
    };
    indices
        .into_iter()
        .map(|i| (i, exponentials[i] / denominator))
        .collect()
}

fn activate(activation: Activation, value: f32) -> f32 {
    match activation {
        Activation::Relu => value.max(0.0),
        Activation::Gelu => tanh_gelu(value),
        Activation::Silu => value * sigmoid(value),
        Activation::Identity => value,
        Activation::Swiglu => unreachable!(),
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn int_attr(node: &Node, name: &str, default: i64) -> Result<i64> {
    match node.attr(name) {
        Some(attr) => attr
            .as_int()
            .ok_or_else(|| error(format!("attribute {name} must be an integer"))),
        None => Ok(default),
    }
}

fn bool_attr(node: &Node, name: &str, default: bool) -> Result<bool> {
    match int_attr(node, name, i64::from(default))? {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(error(format!(
            "attribute {name} must be 0 or 1, got {value}"
        ))),
    }
}

fn float_attr(node: &Node, name: &str, default: f32) -> Result<f32> {
    match node.attr(name) {
        Some(attr) => attr
            .as_float()
            .ok_or_else(|| error(format!("attribute {name} must be a float"))),
        None => Ok(default),
    }
}

fn optional_dense(inputs: &[TensorView], index: usize) -> Result<Option<Vec<f32>>> {
    inputs
        .get(index)
        .filter(|v| !v.is_absent())
        .map(|v| to_dense_f32_widen("MoE", v).map(|c| c.into_owned()))
        .transpose()
}

fn validate_bias(
    name: &str,
    inputs: &[TensorView],
    index: usize,
    experts: usize,
    width: usize,
) -> Result<()> {
    if let Some(view) = inputs.get(index).filter(|v| !v.is_absent()) {
        require_exact_shape(name, view.shape, &[experts, width])?;
    }
    Ok(())
}

fn require_shape(name: &str, shape: &[usize], rank: usize) -> Result<()> {
    if shape.len() != rank {
        return Err(error(format!(
            "{name} must be {rank}-D, got shape {shape:?}"
        )));
    }
    Ok(())
}

fn require_exact_shape(name: &str, got: &[usize], expected: &[usize]) -> Result<()> {
    if got != expected {
        return Err(error(format!(
            "{name} must have shape {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("MoE: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuExecutionProvider;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{Attribute, Graph, NodeId, static_shape};
    use onnx_runtime_loader::Model;
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    fn model_node(
        input_shapes: &[Option<&[usize]>],
        output_shape: &[usize],
        attrs: &[(&str, Attribute)],
    ) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert("com.microsoft".into(), 1);
        let inputs = input_shapes
            .iter()
            .enumerate()
            .map(|(i, shape)| {
                shape.map(|shape| {
                    let value = graph.create_named_value(
                        format!("input_{i}"),
                        DataType::Float32,
                        static_shape(shape.iter().copied()),
                    );
                    graph.add_input(value);
                    value
                })
            })
            .collect();
        let output = graph.create_named_value(
            "output",
            DataType::Float32,
            static_shape(output_shape.iter().copied()),
        );
        let mut node = Node::new(NodeId(0), "MoE", inputs, vec![output]);
        node.domain = "com.microsoft".into();
        for (name, value) in attrs {
            node.attributes.insert((*name).into(), value.clone());
        }
        let node = graph.insert_node(node);
        graph.add_output(output);
        (graph, node)
    }

    fn kernel(graph: &Graph, node: NodeId) -> Box<dyn Kernel> {
        let model = Model::new(graph);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap()
    }

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (&got, &want)) in got.iter().zip(want).enumerate() {
            assert!(
                (got - want).abs() <= 1e-5,
                "index {i}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn grouped_moe_matches_per_token_dense_fallback_for_eight_experts_top2() {
        const ROWS: usize = 6;
        const HIDDEN: usize = 4;
        const INTER: usize = 6;
        const EXPERTS: usize = 8;
        let shapes = [
            Some(&[ROWS, HIDDEN][..]),
            Some(&[ROWS, EXPERTS]),
            Some(&[EXPERTS, INTER, HIDDEN]),
            None,
            Some(&[EXPERTS, HIDDEN, INTER]),
        ];
        let attrs = [
            ("k", Attribute::Int(2)),
            ("activation_type", Attribute::String(b"silu".to_vec())),
            ("normalize_routing_weights", Attribute::Int(1)),
        ];
        let (graph, node) = model_node(&shapes, &[ROWS, HIDDEN], &attrs);
        let input: Vec<f32> = (0..ROWS * HIDDEN)
            .map(|index| (index as f32 * 0.17).sin())
            .collect();
        let router: Vec<f32> = (0..ROWS * EXPERTS)
            .map(|index| ((index * 7 + index / EXPERTS * 3) % 19) as f32 - 9.0)
            .collect();
        let fc1: Vec<f32> = (0..EXPERTS * INTER * HIDDEN)
            .map(|index| ((index * 11 % 23) as f32 - 11.0) * 0.03125)
            .collect();
        let fc2: Vec<f32> = (0..EXPERTS * HIDDEN * INTER)
            .map(|index| ((index * 13 % 29) as f32 - 14.0) * 0.025)
            .collect();
        let x = Owned::f32(&[ROWS, HIDDEN], &input);
        let router_tensor = Owned::f32(&[ROWS, EXPERTS], &router);
        let fc1_tensor = Owned::f32(&[EXPERTS, INTER, HIDDEN], &fc1);
        let fc2_tensor = Owned::f32(&[EXPERTS, HIDDEN, INTER], &fc2);
        let mut grouped = Owned::zeros_f32(&[ROWS, HIDDEN]);
        kernel(&graph, node)
            .execute(
                &[
                    x.view(),
                    router_tensor.view(),
                    fc1_tensor.view(),
                    TensorView::absent(DataType::Float32),
                    fc2_tensor.view(),
                ],
                &mut [grouped.view_mut()],
            )
            .unwrap();

        let attributes = MoeAttributes::from_node(graph.node(node)).unwrap();
        let mut dense = vec![0.0f32; ROWS * HIDDEN];
        for row in 0..ROWS {
            for (expert, weight) in
                routing_weights(&router[row * EXPERTS..(row + 1) * EXPERTS], None, 2, true)
            {
                let expert_output = run_expert(
                    &input[row * HIDDEN..(row + 1) * HIDDEN],
                    &fc1[expert * INTER * HIDDEN..(expert + 1) * INTER * HIDDEN],
                    None,
                    &fc2[expert * HIDDEN * INTER..(expert + 1) * HIDDEN * INTER],
                    None,
                    None,
                    None,
                    INTER,
                    HIDDEN,
                    INTER,
                    &attributes,
                );
                for feature in 0..HIDDEN {
                    dense[row * HIDDEN + feature] += weight * expert_output[feature];
                }
            }
        }
        assert_close(&grouped.to_f32(), &dense);
    }

    fn measure_grouped_vs_dense(rows: usize, iterations: usize) -> (Duration, Duration) {
        const HIDDEN: usize = 128;
        const INTER: usize = 256;
        const EXPERTS: usize = 8;
        let attributes = MoeAttributes {
            k: 2,
            activation: Activation::Silu,
            normalize_routing_weights: true,
            swiglu_fusion: 0,
            activation_alpha: 1.0,
            activation_beta: 0.0,
            swiglu_limit: f32::INFINITY,
        };
        let input: Vec<f32> = (0..rows * HIDDEN)
            .map(|index| ((index * 17 % 101) as f32 - 50.0) * 0.01)
            .collect();
        let fc1: Vec<f32> = (0..EXPERTS * INTER * HIDDEN)
            .map(|index| ((index * 11 % 67) as f32 - 33.0) * 0.002)
            .collect();
        let fc2: Vec<f32> = (0..EXPERTS * HIDDEN * INTER)
            .map(|index| ((index * 13 % 71) as f32 - 35.0) * 0.002)
            .collect();
        let routes: Vec<[(usize, f32); 2]> = (0..rows)
            .map(|row| [(row % EXPERTS, 0.6), ((row + 1) % EXPERTS, 0.4)])
            .collect();

        let dense_start = Instant::now();
        for _ in 0..iterations {
            let mut selected = vec![0.0f32; rows * HIDDEN];
            for row in 0..rows {
                for expert in 0..EXPERTS {
                    let expert_output = run_expert(
                        &input[row * HIDDEN..(row + 1) * HIDDEN],
                        &fc1[expert * INTER * HIDDEN..(expert + 1) * INTER * HIDDEN],
                        None,
                        &fc2[expert * HIDDEN * INTER..(expert + 1) * HIDDEN * INTER],
                        None,
                        None,
                        None,
                        INTER,
                        HIDDEN,
                        INTER,
                        &attributes,
                    );
                    if let Some((_, weight)) = routes[row]
                        .iter()
                        .find(|&&(selected, _)| selected == expert)
                    {
                        for feature in 0..HIDDEN {
                            selected[row * HIDDEN + feature] += weight * expert_output[feature];
                        }
                    }
                }
            }
            black_box(selected);
        }
        let dense = dense_start.elapsed();

        let grouped_start = Instant::now();
        for _ in 0..iterations {
            let mut output = vec![0.0f32; rows * HIDDEN];
            for expert in 0..EXPERTS {
                let tasks: Vec<(usize, f32)> = routes
                    .iter()
                    .enumerate()
                    .flat_map(|(row, route)| {
                        route
                            .iter()
                            .filter(move |&&(selected, _)| selected == expert)
                            .map(move |&(_, weight)| (row, weight))
                    })
                    .collect();
                if tasks.is_empty() {
                    continue;
                }
                let mut grouped_input = Vec::with_capacity(tasks.len() * HIDDEN);
                for &(row, _) in &tasks {
                    grouped_input.extend_from_slice(&input[row * HIDDEN..(row + 1) * HIDDEN]);
                }
                let expert_output = run_expert_grouped(
                    &grouped_input,
                    tasks.len(),
                    &fc1[expert * INTER * HIDDEN..(expert + 1) * INTER * HIDDEN],
                    None,
                    &fc2[expert * HIDDEN * INTER..(expert + 1) * HIDDEN * INTER],
                    None,
                    None,
                    None,
                    INTER,
                    HIDDEN,
                    INTER,
                    &attributes,
                )
                .unwrap();
                for (grouped_row, &(row, weight)) in tasks.iter().enumerate() {
                    for feature in 0..HIDDEN {
                        output[row * HIDDEN + feature] +=
                            weight * expert_output[grouped_row * HIDDEN + feature];
                    }
                }
            }
            black_box(output);
        }
        (dense, grouped_start.elapsed())
    }

    #[test]
    #[ignore = "performance characterization; run with --release --ignored --nocapture"]
    fn grouped_moe_measures_benefit_over_dense_fallback_decode_and_prefill() {
        for (name, rows, iterations) in [("decode", 1, 50), ("prefill", 64, 2)] {
            let (dense, grouped) = measure_grouped_vs_dense(rows, iterations);
            eprintln!(
                "{name} M={rows}: dense={dense:?}, grouped={grouped:?}, speedup={:.2}x",
                dense.as_secs_f64() / grouped.as_secs_f64()
            );
            assert!(
                grouped < dense,
                "{name} grouped path did not beat dense fallback: grouped={grouped:?}, dense={dense:?}"
            );
        }
    }

    #[test]
    fn moe_gelu_top1_biases_selects_different_experts_per_row() {
        let shapes = [
            Some(&[2, 2][..]),
            Some(&[2, 2]),
            Some(&[2, 2, 2]),
            Some(&[2, 2]),
            Some(&[2, 2, 2]),
            Some(&[2, 2]),
        ];
        let (graph, node) = model_node(
            &shapes,
            &[2, 2],
            &[
                ("k", Attribute::Int(1)),
                ("activation_type", Attribute::String(b"gelu".to_vec())),
                ("normalize_routing_weights", Attribute::Int(0)),
            ],
        );
        let x = Owned::f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let router = Owned::f32(&[2, 2], &[4.0, 0.0, 0.0, 4.0]);
        let fc1 = Owned::f32(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
        let fc1_bias = Owned::f32(&[2, 2], &[0.5, -0.5, 1.0, 1.0]);
        let fc2 = Owned::f32(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
        let fc2_bias = Owned::f32(&[2, 2], &[0.25, -0.25, 0.5, 0.5]);
        let mut y = Owned::zeros_f32(&[2, 2]);
        kernel(&graph, node)
            .execute(
                &[
                    x.view(),
                    router.view(),
                    fc1.view(),
                    fc1_bias.view(),
                    fc2.view(),
                    fc2_bias.view(),
                ],
                &mut [y.view_mut()],
            )
            .unwrap();
        assert_close(
            &y.to_f32(),
            &[1.619_902, 1.128_895_2, 7.365_103_2, 9.329_131],
        );
    }

    #[test]
    fn moe_defaults_to_top1_relu_when_attributes_are_omitted() {
        let shapes = [
            Some(&[1, 2][..]),
            Some(&[1, 2]),
            Some(&[2, 2, 2]),
            None,
            Some(&[2, 2, 2]),
        ];
        let (graph, node) = model_node(&shapes, &[1, 2], &[]);
        let x = Owned::f32(&[1, 2], &[1.0, -2.0]);
        let router = Owned::f32(&[1, 2], &[0.0, 0.0]);
        let fc1 = Owned::f32(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
        let fc2 = Owned::f32(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
        let mut y = Owned::zeros_f32(&[1, 2]);
        kernel(&graph, node)
            .execute(
                &[
                    x.view(),
                    router.view(),
                    fc1.view(),
                    TensorView::absent(DataType::Float32),
                    fc2.view(),
                ],
                &mut [y.view_mut()],
            )
            .unwrap();
        assert_close(&y.to_f32(), &[0.5, 0.0]);
    }

    #[test]
    fn moe_top1_relu_runs_natively_in_bf16_matching_f32() {
        // Same node as the top1/relu default test, but Q/router/weights and the
        // output are bf16. MoE computes in f32 (widen on read, narrow on write),
        // so the bf16 result must match the f32 reference within bf16 tolerance.
        let shapes = [
            Some(&[1, 2][..]),
            Some(&[1, 2]),
            Some(&[2, 2, 2]),
            None,
            Some(&[2, 2, 2]),
        ];
        let (graph, node) = model_node(&shapes, &[1, 2], &[]);
        let x = Owned::bf16(&[1, 2], &[1.0, -2.0]);
        let router = Owned::bf16(&[1, 2], &[0.0, 0.0]);
        let fc1 = Owned::bf16(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
        let fc2 = Owned::bf16(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
        let mut y = Owned::zeros(DataType::BFloat16, &[1, 2]);
        kernel(&graph, node)
            .execute(
                &[
                    x.view(),
                    router.view(),
                    fc1.view(),
                    TensorView::absent(DataType::Undefined),
                    fc2.view(),
                ],
                &mut [y.view_mut()],
            )
            .unwrap();
        let got = y.to_bf16_as_f32();
        for (i, (&g, &w)) in got.iter().zip(&[0.5, 0.0]).enumerate() {
            assert!((g - w).abs() <= 0.05, "index {i}: got {g}, want {w}");
        }
    }

    #[test]
    fn moe_silu_top2_normalized_without_biases_preserves_3d_shape() {
        let shapes = [
            Some(&[1, 2, 2][..]),
            Some(&[2, 2]),
            Some(&[2, 2, 2]),
            None,
            Some(&[2, 2, 2]),
        ];
        let (graph, node) = model_node(
            &shapes,
            &[1, 2, 2],
            &[
                ("k", Attribute::Int(2)),
                ("activation_type", Attribute::String(b"silu".to_vec())),
                ("normalize_routing_weights", Attribute::Int(1)),
            ],
        );
        let x = Owned::f32(&[1, 2, 2], &[1.0, -1.0, 2.0, 1.0]);
        let router = Owned::f32(&[2, 2], &[0.0, 0.0, 2.0, 1.0]);
        let fc1 = Owned::f32(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
        let fc2 = Owned::f32(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5]);
        let mut y = Owned::zeros_f32(&[1, 2, 2]);
        kernel(&graph, node)
            .execute(
                &[
                    x.view(),
                    router.view(),
                    fc1.view(),
                    TensorView::absent(DataType::Float32),
                    fc2.view(),
                ],
                &mut [y.view_mut()],
            )
            .unwrap();
        let p0 = 2.0f32.exp() / (2.0f32.exp() + 1.0f32.exp());
        let p1 = 1.0 - p0;
        assert_close(
            &y.to_f32(),
            &[
                0.5 * activate(Activation::Silu, 1.0) + 0.5 * 0.5 * activate(Activation::Silu, 2.0),
                0.5 * activate(Activation::Silu, -1.0)
                    + 0.5 * 0.5 * activate(Activation::Silu, -2.0),
                p0 * activate(Activation::Silu, 2.0) + p1 * 0.5 * activate(Activation::Silu, 4.0),
                p0 * activate(Activation::Silu, 1.0) + p1 * 0.5 * activate(Activation::Silu, 2.0),
            ],
        );
    }

    #[test]
    fn moe_swiglu_unfused_fc3_with_biases() {
        let shapes = [
            Some(&[1, 2][..]),
            Some(&[1, 2]),
            Some(&[2, 1, 2]),
            Some(&[2, 1]),
            Some(&[2, 2, 1]),
            None,
            Some(&[2, 1, 2]),
            Some(&[2, 1]),
        ];
        let (graph, node) = model_node(
            &shapes,
            &[1, 2],
            &[
                ("k", Attribute::Int(1)),
                ("activation_type", Attribute::String(b"swiglu".to_vec())),
                ("swiglu_fusion", Attribute::Int(0)),
                ("normalize_routing_weights", Attribute::Int(1)),
            ],
        );
        let x = Owned::f32(&[1, 2], &[2.0, 1.0]);
        let router = Owned::f32(&[1, 2], &[0.0, 3.0]);
        let fc1 = Owned::f32(&[2, 1, 2], &[1.0, 0.0, 0.0, 1.0]);
        let fc1_bias = Owned::f32(&[2, 1], &[0.0, 1.0]);
        let fc2 = Owned::f32(&[2, 2, 1], &[1.0, 2.0, 3.0, 4.0]);
        let fc3 = Owned::f32(&[2, 1, 2], &[0.0, 1.0, 1.0, 1.0]);
        let fc3_bias = Owned::f32(&[2, 1], &[0.0, 0.5]);
        let mut y = Owned::zeros_f32(&[1, 2]);
        kernel(&graph, node)
            .execute(
                &[
                    x.view(),
                    router.view(),
                    fc1.view(),
                    fc1_bias.view(),
                    fc2.view(),
                    TensorView::absent(DataType::Float32),
                    fc3.view(),
                    fc3_bias.view(),
                ],
                &mut [y.view_mut()],
            )
            .unwrap();
        let expert = 2.0 * sigmoid(2.0) * 3.5;
        assert_close(&y.to_f32(), &[3.0 * expert, 4.0 * expert]);
    }

    #[test]
    fn moe_swiglu_fused_interleaved() {
        let shapes = [
            Some(&[1, 1][..]),
            Some(&[1, 1]),
            Some(&[1, 4, 1]),
            None,
            Some(&[1, 1, 2]),
        ];
        let (graph, node) = model_node(
            &shapes,
            &[1, 1],
            &[
                ("k", Attribute::Int(1)),
                ("activation_type", Attribute::String(b"swiglu".to_vec())),
                ("swiglu_fusion", Attribute::Int(1)),
            ],
        );
        let x = Owned::f32(&[1, 1], &[2.0]);
        let router = Owned::f32(&[1, 1], &[0.0]);
        let fc1 = Owned::f32(&[1, 4, 1], &[1.0, 3.0, 2.0, 4.0]);
        let fc2 = Owned::f32(&[1, 1, 2], &[1.0, 0.5]);
        let mut y = Owned::zeros_f32(&[1, 1]);
        kernel(&graph, node)
            .execute(
                &[
                    x.view(),
                    router.view(),
                    fc1.view(),
                    TensorView::absent(DataType::Float32),
                    fc2.view(),
                ],
                &mut [y.view_mut()],
            )
            .unwrap();
        let expected = 2.0 * sigmoid(2.0) * 6.0 + 0.5 * (4.0 * sigmoid(4.0) * 8.0);
        assert_close(&y.to_f32(), &[expected]);
    }

    #[test]
    fn moe_silu_with_fc3_uses_ort_mixtral_gated_form() {
        let shapes = [
            Some(&[1, 1][..]),
            Some(&[1, 1]),
            Some(&[1, 1, 1]),
            None,
            Some(&[1, 1, 1]),
            None,
            Some(&[1, 1, 1]),
        ];
        let (graph, node) = model_node(
            &shapes,
            &[1, 1],
            &[
                ("k", Attribute::Int(1)),
                ("activation_type", Attribute::String(b"silu".to_vec())),
            ],
        );
        let x = Owned::f32(&[1, 1], &[2.0]);
        let router = Owned::f32(&[1, 1], &[0.0]);
        let fc1 = Owned::f32(&[1, 1, 1], &[3.0]);
        let fc2 = Owned::f32(&[1, 1, 1], &[0.5]);
        let fc3 = Owned::f32(&[1, 1, 1], &[4.0]);
        let mut y = Owned::zeros_f32(&[1, 1]);
        kernel(&graph, node)
            .execute(
                &[
                    x.view(),
                    router.view(),
                    fc1.view(),
                    TensorView::absent(DataType::Float32),
                    fc2.view(),
                    TensorView::absent(DataType::Float32),
                    fc3.view(),
                ],
                &mut [y.view_mut()],
            )
            .unwrap();
        assert_close(&y.to_f32(), &[0.5 * (6.0 * sigmoid(6.0) * 8.0)]);
    }

    /// Reference MoE forward built entirely out of [`run_expert`]'s scalar dot
    /// products - no GEMM, no grouping, no transposes.
    #[allow(clippy::too_many_arguments)]
    fn reference_moe(
        rows: usize,
        hidden: usize,
        inter: usize,
        experts: usize,
        k: usize,
        fc1_size: usize,
        input: &[f32],
        router: &[f32],
        fc1: &[f32],
        fc2: &[f32],
        fc3: Option<&[f32]>,
        fc1_bias: Option<&[f32]>,
        fc2_bias: Option<&[f32]>,
        fc3_bias: Option<&[f32]>,
        attributes: &MoeAttributes,
    ) -> Vec<f32> {
        let mut dense = vec![0.0f32; rows * hidden];
        for row in 0..rows {
            for (expert, weight) in routing_weights(
                &router[row * experts..(row + 1) * experts],
                None,
                k,
                attributes.normalize_routing_weights,
            ) {
                let expert_output = run_expert(
                    &input[row * hidden..(row + 1) * hidden],
                    &fc1[expert * fc1_size * hidden..(expert + 1) * fc1_size * hidden],
                    fc1_bias.map(|bias| &bias[expert * fc1_size..(expert + 1) * fc1_size]),
                    &fc2[expert * hidden * inter..(expert + 1) * hidden * inter],
                    fc2_bias.map(|bias| &bias[expert * hidden..(expert + 1) * hidden]),
                    fc3.map(|fc3| &fc3[expert * inter * hidden..(expert + 1) * inter * hidden]),
                    fc3_bias.map(|bias| &bias[expert * inter..(expert + 1) * inter]),
                    fc1_size,
                    hidden,
                    inter,
                    attributes,
                );
                for feature in 0..hidden {
                    dense[row * hidden + feature] += weight * expert_output[feature];
                }
            }
        }
        dense
    }

    /// Both production drivers must agree **bit-for-bit**, because the choice
    /// between them is a pure performance gate: a shape that crosses the
    /// threshold must not change its numerics.
    ///
    /// The row counts cover both regimes - `rows == 1` puts every group at one
    /// row (the batched-GEMM case), `rows == 37` with `k == 3` over 5 experts
    /// produces groups of several different lengths.
    #[test]
    fn the_two_drivers_are_bit_identical() {
        const HIDDEN: usize = 6;
        const INTER: usize = 5;
        const EXPERTS: usize = 5;
        for &(rows, k) in &[(1usize, 2usize), (7, 1), (37, 3), (64, 5)] {
            for &(activation, fusion) in &[
                (Activation::Swiglu, 1),
                (Activation::Swiglu, 0),
                (Activation::Silu, 0),
                (Activation::Gelu, 0),
                (Activation::Relu, 0),
            ] {
                for &bias in &[false, true] {
                    let attributes = MoeAttributes {
                        k,
                        activation,
                        normalize_routing_weights: true,
                        swiglu_fusion: fusion,
                        activation_alpha: 1.702,
                        activation_beta: 1.0,
                        swiglu_limit: 7.0,
                    };
                    let fc1_size = attributes.checked_fc1_size(INTER, "MoE").unwrap();
                    let uses_fc3 = attributes.uses_separate_gate(true);
                    let x = ramp(rows * HIDDEN, 11, 0.5, 0.13);
                    let router = ramp(rows * EXPERTS, 13, 0.4, 0.31);
                    let fc1 = ramp(EXPERTS * fc1_size * HIDDEN, 17, 0.3, 0.017);
                    let fc2 = ramp(EXPERTS * HIDDEN * INTER, 19, 0.2, 0.023);
                    let fc3 = uses_fc3.then(|| ramp(EXPERTS * INTER * HIDDEN, 23, 0.25, 0.029));
                    let fc1_bias = bias.then(|| ramp(EXPERTS * fc1_size, 7, 0.15, 0.11));
                    let fc2_bias = bias.then(|| ramp(EXPERTS * HIDDEN, 5, 0.1, 0.07));
                    let fc3_bias = (bias && uses_fc3).then(|| ramp(EXPERTS * INTER, 3, 0.05, 0.05));
                    let mut plan = RoutingPlan::build(&router, rows, EXPERTS, k, true);
                    let args = (
                        fc1_bias.as_deref(),
                        fc2_bias.as_deref(),
                        fc3.as_deref(),
                        fc3_bias.as_deref(),
                    );
                    plan.build_row_slots();
                    let grouped = run_moe_grouped(
                        &plan,
                        &x,
                        &fc1,
                        args.0,
                        &fc2,
                        args.1,
                        args.2,
                        args.3,
                        fc1_size,
                        HIDDEN,
                        INTER,
                        &attributes,
                    )
                    .unwrap();
                    let per_expert = run_moe_per_expert(
                        &plan,
                        &x,
                        &fc1,
                        args.0,
                        &fc2,
                        args.1,
                        args.2,
                        args.3,
                        fc1_size,
                        HIDDEN,
                        INTER,
                        &attributes,
                    )
                    .unwrap();
                    assert_eq!(
                        grouped, per_expert,
                        "rows={rows} k={k} activation={activation:?} fusion={fusion} bias={bias}"
                    );
                }
            }
        }
    }

    /// The gate must pick the grouped driver exactly where it was measured to
    /// win. These are the calibration points from the 27-cell ORT A/B: the
    /// 512-token prefill shapes won by 1.35x-1.57x, the 1- and 32-token decode
    /// shapes lost by 5-16%.
    #[test]
    fn the_driver_gate_matches_the_measured_crossover() {
        // (hidden, inter, experts, k, tokens, expect_grouped)
        let cells = [
            (1024usize, 3584usize, 8usize, 2usize, 512usize, true),
            (2048, 768, 16, 8, 512, true),
            (2048, 6400, 4, 2, 512, true),
            (2048, 6400, 4, 2, 32, true),
            (1024, 3584, 8, 2, 32, false),
            (2048, 768, 16, 8, 32, false),
            (1024, 3584, 8, 2, 1, false),
            (2048, 768, 16, 8, 1, false),
            (2048, 6400, 4, 2, 1, false),
        ];
        for &(hidden, inter, experts, k, tokens, expect) in &cells {
            // A uniform router spreads the tokens evenly over the experts,
            // which is what the synthetic benchmark graphs do.
            let mut router = vec![0.0f32; tokens * experts];
            for row in 0..tokens {
                for expert in 0..experts {
                    router[row * experts + expert] = ((row + expert) % experts) as f32;
                }
            }
            let plan = RoutingPlan::build(&router, tokens, experts, k, true);
            let fc1_size = 2 * inter;
            // Asserted on the work estimate rather than on `use_grouped_driver`
            // so the test still means something on a single-core runner, where
            // the driver always answers "per-expert" regardless of shape.
            assert_eq!(
                grouped_work_units(&plan, fc1_size, hidden, inter) >= grouped_min_work(),
                expect,
                "hidden={hidden} inter={inter} experts={experts} k={k} tokens={tokens}"
            );
            if moe_worker_budget() >= 2 {
                assert_eq!(
                    use_grouped_driver(&plan, fc1_size, hidden, inter),
                    expect,
                    "driver disagrees with the work estimate"
                );
            }
        }
    }

    /// A collapsed router - one expert taking most of the rows while the rest
    /// take one each - still has one large GEMM to hide the elementwise stages
    /// behind, so the gate must look at the **largest** group and not the mean.
    ///
    /// This is the case the balanced-router calibration test cannot see: with
    /// an even distribution every summary statistic agrees.
    #[test]
    fn the_gate_reads_the_largest_group_not_the_average() {
        const EXPERTS: usize = 16;
        const HIDDEN: usize = 2048;
        const INTER: usize = 6400;
        let rows = EXPERTS;
        // Row 0 alone would be one row per expert; instead the first 16 rows all
        // pick expert 0, and rows 1..16 each add one distinct second expert.
        let mut router = vec![0.0f32; rows * EXPERTS];
        for row in 0..rows {
            router[row * EXPERTS] = 9.0;
            router[row * EXPERTS + row] = 8.0;
        }
        let plan = RoutingPlan::build(&router, rows, EXPERTS, 2, true);
        let largest = plan.groups.iter().map(|g| g.len).max().unwrap();
        let mean = plan.slots() / plan.groups.len();
        assert!(
            largest >= 4 * mean.max(1),
            "the fixture must be skewed: largest={largest} mean={mean}"
        );
        let fc1_size = 2 * INTER;
        assert_eq!(
            grouped_work_units(&plan, fc1_size, HIDDEN, INTER),
            largest as u64 * HIDDEN as u64 * (fc1_size + INTER) as u64
        );
        assert!(
            grouped_work_units(&plan, fc1_size, HIDDEN, INTER) >= grouped_min_work(),
            "the dominant group is large enough for the grouped driver"
        );
        // The same shape summarised by the mean would fall below the floor, so
        // this test fails if the statistic is ever changed back.
        let mean_work = (mean as u64) * (HIDDEN as u64) * ((fc1_size + INTER) as u64);
        assert!(mean_work < grouped_min_work());
    }

    /// The exact algorithm `run_moe_grouped` replaced: gather each expert's
    /// rows into a fresh buffer, run `run_expert_grouped`, then accumulate the
    /// weighted result into the output in ascending expert order.
    #[allow(clippy::too_many_arguments)]
    fn per_expert_loop_reference(
        rows: usize,
        hidden: usize,
        inter: usize,
        experts: usize,
        fc1_size: usize,
        input: &[f32],
        router: &[f32],
        fc1: &[f32],
        fc2: &[f32],
        fc3: Option<&[f32]>,
        fc1_bias: Option<&[f32]>,
        fc2_bias: Option<&[f32]>,
        fc3_bias: Option<&[f32]>,
        attributes: &MoeAttributes,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; rows * hidden];
        let mut tasks = BTreeMap::<usize, Vec<(usize, f32)>>::new();
        for row in 0..rows {
            for (expert, weight) in routing_weights(
                &router[row * experts..(row + 1) * experts],
                None,
                attributes.k,
                attributes.normalize_routing_weights,
            ) {
                tasks.entry(expert).or_default().push((row, weight));
            }
        }
        for (expert, expert_tasks) in tasks {
            let mut grouped_input = Vec::with_capacity(expert_tasks.len() * hidden);
            for &(row, _) in &expert_tasks {
                grouped_input.extend_from_slice(&input[row * hidden..(row + 1) * hidden]);
            }
            let expert_out = run_expert_grouped(
                &grouped_input,
                expert_tasks.len(),
                &fc1[expert * fc1_size * hidden..(expert + 1) * fc1_size * hidden],
                fc1_bias.map(|b| &b[expert * fc1_size..(expert + 1) * fc1_size]),
                &fc2[expert * hidden * inter..(expert + 1) * hidden * inter],
                fc2_bias.map(|b| &b[expert * hidden..(expert + 1) * hidden]),
                fc3.map(|w| &w[expert * inter * hidden..(expert + 1) * inter * hidden]),
                fc3_bias.map(|b| &b[expert * inter..(expert + 1) * inter]),
                fc1_size,
                hidden,
                inter,
                attributes,
            )
            .unwrap();
            for (grouped_row, &(row, weight)) in expert_tasks.iter().enumerate() {
                for feature in 0..hidden {
                    output[row * hidden + feature] +=
                        weight * expert_out[grouped_row * hidden + feature];
                }
            }
        }
        output
    }

    /// The grouped/batched driver must be **bit-identical** to the per-expert
    /// loop it replaced. Batching only changes how MLAS partitions threads and
    /// the slot layout only changes where rows live, so any difference here is
    /// a bug rather than a rounding artefact.
    ///
    /// The two row counts are the two regimes: `rows == 1` puts every group at
    /// exactly one row, which is the uniform-length case that becomes a single
    /// batched GEMM, while `rows == 37` with `k == 3` over 5 experts produces
    /// groups of several different lengths, exercising the length bucketing,
    /// the per-expert bias lookup and the multi-contribution scatter.
    #[test]
    fn grouped_driver_is_bit_identical_to_the_per_expert_loop() {
        const HIDDEN: usize = 24;
        const INTER: usize = 20;
        const EXPERTS: usize = 5;
        for &(rows, k) in &[(1usize, 3usize), (37, 3), (512, 2)] {
            for &(activation, fusion, with_fc3) in &[
                (Activation::Relu, 0usize, false),
                (Activation::Gelu, 0, false),
                (Activation::Silu, 0, true),
                (Activation::Swiglu, 0, true),
                (Activation::Swiglu, 1, false),
                (Activation::Swiglu, 2, false),
            ] {
                for with_bias in [false, true] {
                    let attributes = MoeAttributes {
                        k,
                        activation,
                        normalize_routing_weights: true,
                        swiglu_fusion: fusion,
                        activation_alpha: 1.702,
                        activation_beta: 1.0,
                        swiglu_limit: 7.0,
                    };
                    let fc1_size = attributes.checked_fc1_size(INTER, "MoE").unwrap();
                    let input = ramp(rows * HIDDEN, 23, 0.5, 0.031);
                    let router = ramp(rows * EXPERTS, 17, 0.4, 0.11);
                    let fc1 = ramp(EXPERTS * fc1_size * HIDDEN, 29, 0.5, 0.017);
                    let fc2 = ramp(EXPERTS * HIDDEN * INTER, 31, 0.5, 0.013);
                    let fc3 = with_fc3.then(|| ramp(EXPERTS * INTER * HIDDEN, 19, 0.5, 0.021));
                    let fc1_bias = with_bias.then(|| ramp(EXPERTS * fc1_size, 11, 0.5, 0.07));
                    let fc2_bias = with_bias.then(|| ramp(EXPERTS * HIDDEN, 13, 0.5, 0.05));
                    let fc3_bias =
                        (with_bias && with_fc3).then(|| ramp(EXPERTS * INTER, 7, 0.5, 0.03));

                    let want = per_expert_loop_reference(
                        rows,
                        HIDDEN,
                        INTER,
                        EXPERTS,
                        fc1_size,
                        &input,
                        &router,
                        &fc1,
                        &fc2,
                        fc3.as_deref(),
                        fc1_bias.as_deref(),
                        fc2_bias.as_deref(),
                        fc3_bias.as_deref(),
                        &attributes,
                    );
                    let mut plan = RoutingPlan::build(&router, rows, EXPERTS, k, true);
                    plan.build_row_slots();
                    let got = run_moe_grouped(
                        &plan,
                        &input,
                        &fc1,
                        fc1_bias.as_deref(),
                        &fc2,
                        fc2_bias.as_deref(),
                        fc3.as_deref(),
                        fc3_bias.as_deref(),
                        fc1_size,
                        HIDDEN,
                        INTER,
                        &attributes,
                    )
                    .unwrap();
                    assert_eq!(
                        got, want,
                        "rows={rows} k={k} activation={activation:?} fusion={fusion} \
                         fc3={with_fc3} bias={with_bias}"
                    );
                }
            }
        }
    }

    /// The slot layout has to survive experts that are never selected and rows
    /// whose top-k lands on a single expert repeatedly: `row_slots` is sized
    /// `rows * k` and the scatter must skip the entries that stay unfilled.
    #[test]
    fn routing_plan_groups_are_contiguous_ascending_and_cover_every_slot() {
        const ROWS: usize = 11;
        const EXPERTS: usize = 6;
        const K: usize = 2;
        // Row `r` prefers experts `r % 3` - experts 3..6 are never routed to.
        let mut router = vec![0.0f32; ROWS * EXPERTS];
        for row in 0..ROWS {
            router[row * EXPERTS + (row % 3)] = 5.0;
            router[row * EXPERTS + ((row + 1) % 3)] = 4.0;
        }
        let mut plan = RoutingPlan::build(&router, ROWS, EXPERTS, K, true);
        assert_eq!(plan.slots(), ROWS * K);
        assert_eq!(plan.groups.len(), 3, "only three experts are ever selected");
        let mut expected_start = 0;
        let mut previous_expert = None;
        for group in &plan.groups {
            assert_eq!(group.start, expected_start);
            assert!(previous_expert.is_none_or(|e| e < group.expert));
            for slot in group.start..group.start + group.len {
                assert_eq!(plan.slot_expert[slot] as usize, group.expert);
            }
            previous_expert = Some(group.expert);
            expected_start += group.len;
        }
        assert_eq!(expected_start, plan.slots());
        assert!(
            plan.row_slots.is_empty(),
            "the inversion is only built for the driver that reads it"
        );
        plan.build_row_slots();
        // Every slot is reachable from exactly one `row_slots` entry.
        let mut seen = vec![0usize; plan.slots()];
        for &slot in &plan.row_slots {
            if slot != u32::MAX {
                seen[slot as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&count| count == 1));
    }

    fn ramp(len: usize, modulus: usize, offset: f32, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|index| ((index * 7 + 3) % modulus) as f32 * scale - offset)
            .collect()
    }

    /// `rows == 1` used to short-circuit into the scalar `run_expert` path.
    /// Decode is exactly that shape, so the grouped GEMM now serves it and has
    /// to stay numerically indistinguishable from the reference.
    #[test]
    fn single_token_decode_matches_the_scalar_reference() {
        const ROWS: usize = 1;
        const HIDDEN: usize = 8;
        const INTER: usize = 12;
        const EXPERTS: usize = 4;
        let shapes = [
            Some(&[ROWS, HIDDEN][..]),
            Some(&[ROWS, EXPERTS]),
            Some(&[EXPERTS, INTER, HIDDEN]),
            None,
            Some(&[EXPERTS, HIDDEN, INTER]),
        ];
        let attrs = [
            ("k", Attribute::Int(2)),
            ("activation_type", Attribute::String(b"silu".to_vec())),
            ("normalize_routing_weights", Attribute::Int(1)),
        ];
        let (graph, node) = model_node(&shapes, &[ROWS, HIDDEN], &attrs);
        let input = ramp(ROWS * HIDDEN, 23, 0.4, 0.05);
        let router = ramp(ROWS * EXPERTS, 13, 5.0, 1.0);
        let fc1 = ramp(EXPERTS * INTER * HIDDEN, 29, 0.3, 0.02);
        let fc2 = ramp(EXPERTS * HIDDEN * INTER, 31, 0.3, 0.02);

        let x = Owned::f32(&[ROWS, HIDDEN], &input);
        let router_tensor = Owned::f32(&[ROWS, EXPERTS], &router);
        let fc1_tensor = Owned::f32(&[EXPERTS, INTER, HIDDEN], &fc1);
        let fc2_tensor = Owned::f32(&[EXPERTS, HIDDEN, INTER], &fc2);
        let mut got = Owned::zeros_f32(&[ROWS, HIDDEN]);
        kernel(&graph, node)
            .execute(
                &[
                    x.view(),
                    router_tensor.view(),
                    fc1_tensor.view(),
                    TensorView::absent(DataType::Float32),
                    fc2_tensor.view(),
                ],
                &mut [got.view_mut()],
            )
            .unwrap();

        let attributes = MoeAttributes::from_node(graph.node(node)).unwrap();
        let want = reference_moe(
            ROWS,
            HIDDEN,
            INTER,
            EXPERTS,
            2,
            INTER,
            &input,
            &router,
            &fc1,
            &fc2,
            None,
            None,
            None,
            None,
            &attributes,
        );
        assert_close(&got.to_f32(), &want);
    }

    /// All three swiglu layouts (unfused fc3, interleaved, split) go through
    /// the same grouped GEMM, with and without biases, and at both a decode
    /// row count and a prefill row count. Each combination has to match the
    /// scalar reference.
    #[test]
    fn every_swiglu_fusion_mode_matches_the_scalar_reference() {
        const HIDDEN: usize = 6;
        const INTER: usize = 4;
        const EXPERTS: usize = 4;
        for rows in [1usize, 5] {
            for fusion in [0i64, 1, 2] {
                for biased in [false, true] {
                    let fc1_size = if fusion == 0 { INTER } else { 2 * INTER };
                    let mut shapes = vec![
                        Some(vec![rows, HIDDEN]),
                        Some(vec![rows, EXPERTS]),
                        Some(vec![EXPERTS, fc1_size, HIDDEN]),
                        biased.then(|| vec![EXPERTS, fc1_size]),
                        Some(vec![EXPERTS, HIDDEN, INTER]),
                    ];
                    if fusion == 0 || biased {
                        shapes.push(biased.then(|| vec![EXPERTS, HIDDEN]));
                    }
                    if fusion == 0 {
                        shapes.push(Some(vec![EXPERTS, INTER, HIDDEN]));
                        shapes.push(biased.then(|| vec![EXPERTS, INTER]));
                    }
                    let shape_refs: Vec<Option<&[usize]>> =
                        shapes.iter().map(|s| s.as_deref()).collect();
                    let attrs = [
                        ("k", Attribute::Int(2)),
                        ("activation_type", Attribute::String(b"swiglu".to_vec())),
                        ("normalize_routing_weights", Attribute::Int(1)),
                        ("swiglu_fusion", Attribute::Int(fusion)),
                    ];
                    let (graph, node) = model_node(&shape_refs, &[rows, HIDDEN], &attrs);
                    let input = ramp(rows * HIDDEN, 19, 0.3, 0.04);
                    let router = ramp(rows * EXPERTS, 17, 6.0, 1.0);
                    let fc1 = ramp(EXPERTS * fc1_size * HIDDEN, 23, 0.25, 0.02);
                    let fc2 = ramp(EXPERTS * HIDDEN * INTER, 29, 0.25, 0.02);
                    let fc3 = ramp(EXPERTS * INTER * HIDDEN, 31, 0.25, 0.02);
                    let fc1_bias = ramp(EXPERTS * fc1_size, 13, 0.2, 0.05);
                    let fc2_bias = ramp(EXPERTS * HIDDEN, 11, 0.2, 0.05);
                    let fc3_bias = ramp(EXPERTS * INTER, 7, 0.2, 0.05);

                    let x = Owned::f32(&[rows, HIDDEN], &input);
                    let router_tensor = Owned::f32(&[rows, EXPERTS], &router);
                    let fc1_tensor = Owned::f32(&[EXPERTS, fc1_size, HIDDEN], &fc1);
                    let fc2_tensor = Owned::f32(&[EXPERTS, HIDDEN, INTER], &fc2);
                    let fc3_tensor = Owned::f32(&[EXPERTS, INTER, HIDDEN], &fc3);
                    let fc1_bias_tensor = Owned::f32(&[EXPERTS, fc1_size], &fc1_bias);
                    let fc2_bias_tensor = Owned::f32(&[EXPERTS, HIDDEN], &fc2_bias);
                    let fc3_bias_tensor = Owned::f32(&[EXPERTS, INTER], &fc3_bias);
                    let absent = TensorView::absent(DataType::Float32);
                    let mut got = Owned::zeros_f32(&[rows, HIDDEN]);
                    let mut views = vec![
                        x.view(),
                        router_tensor.view(),
                        fc1_tensor.view(),
                        if biased {
                            fc1_bias_tensor.view()
                        } else {
                            absent
                        },
                        fc2_tensor.view(),
                    ];
                    if fusion == 0 || biased {
                        views.push(if biased {
                            fc2_bias_tensor.view()
                        } else {
                            absent
                        });
                    }
                    if fusion == 0 {
                        views.push(fc3_tensor.view());
                        views.push(if biased {
                            fc3_bias_tensor.view()
                        } else {
                            absent
                        });
                    }
                    kernel(&graph, node)
                        .execute(&views, &mut [got.view_mut()])
                        .unwrap();

                    let attributes = MoeAttributes::from_node(graph.node(node)).unwrap();
                    let want = reference_moe(
                        rows,
                        HIDDEN,
                        INTER,
                        EXPERTS,
                        2,
                        fc1_size,
                        &input,
                        &router,
                        &fc1,
                        &fc2,
                        (fusion == 0).then_some(&fc3[..]),
                        biased.then_some(&fc1_bias[..]),
                        biased.then_some(&fc2_bias[..]),
                        (fusion == 0 && biased).then_some(&fc3_bias[..]),
                        &attributes,
                    );
                    let got = got.to_f32();
                    assert_eq!(got.len(), want.len());
                    for (index, (&got, &want)) in got.iter().zip(&want).enumerate() {
                        assert!(
                            (got - want).abs() <= 1e-5,
                            "rows={rows} swiglu_fusion={fusion} biased={biased} \
                             index {index}: got {got}, want {want}"
                        );
                    }
                }
            }
        }
    }

    /// Expert weights are borrowed when they are contiguous and copied
    /// otherwise. A strided view takes the copying branch, and both branches
    /// have to produce bit-identical results.
    #[test]
    fn borrowed_and_copied_expert_weights_agree() {
        const ROWS: usize = 3;
        const HIDDEN: usize = 4;
        const INTER: usize = 6;
        const EXPERTS: usize = 4;
        let shapes = [
            Some(&[ROWS, HIDDEN][..]),
            Some(&[ROWS, EXPERTS]),
            Some(&[EXPERTS, INTER, HIDDEN]),
            None,
            Some(&[EXPERTS, HIDDEN, INTER]),
        ];
        let attrs = [
            ("k", Attribute::Int(2)),
            ("activation_type", Attribute::String(b"silu".to_vec())),
            ("normalize_routing_weights", Attribute::Int(1)),
        ];
        let (graph, node) = model_node(&shapes, &[ROWS, HIDDEN], &attrs);
        let input = ramp(ROWS * HIDDEN, 19, 0.3, 0.0625);
        let router = ramp(ROWS * EXPERTS, 13, 5.0, 1.0);
        let fc1 = ramp(EXPERTS * INTER * HIDDEN, 23, 0.25, 0.0625);
        let fc2 = ramp(EXPERTS * HIDDEN * INTER, 29, 0.25, 0.0625);

        // The same values laid out with a gap after every element, exposed
        // through a stride-2 view. Identical logical tensor, non-contiguous
        // storage, so `to_dense_f32_widen` has to materialize it.
        let spread = |values: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; values.len() * 2];
            for (index, &value) in values.iter().enumerate() {
                out[index * 2] = value;
            }
            out
        };

        let x = Owned::f32(&[ROWS, HIDDEN], &input);
        let router_tensor = Owned::f32(&[ROWS, EXPERTS], &router);
        let fc1_contiguous = Owned::f32(&[EXPERTS, INTER, HIDDEN], &fc1);
        let fc2_contiguous = Owned::f32(&[EXPERTS, HIDDEN, INTER], &fc2);
        let fc1_strided = Owned::f32(&[EXPERTS * INTER * HIDDEN * 2], &spread(&fc1)).with_view(
            &[EXPERTS, INTER, HIDDEN],
            &[(INTER * HIDDEN * 2) as i64, (HIDDEN * 2) as i64, 2],
        );
        let fc2_strided = Owned::f32(&[EXPERTS * HIDDEN * INTER * 2], &spread(&fc2)).with_view(
            &[EXPERTS, HIDDEN, INTER],
            &[(HIDDEN * INTER * 2) as i64, (INTER * 2) as i64, 2],
        );

        let mut borrowed = Owned::zeros_f32(&[ROWS, HIDDEN]);
        kernel(&graph, node)
            .execute(
                &[
                    x.view(),
                    router_tensor.view(),
                    fc1_contiguous.view(),
                    TensorView::absent(DataType::Float32),
                    fc2_contiguous.view(),
                ],
                &mut [borrowed.view_mut()],
            )
            .unwrap();
        let mut copied = Owned::zeros_f32(&[ROWS, HIDDEN]);
        kernel(&graph, node)
            .execute(
                &[
                    x.view(),
                    router_tensor.view(),
                    fc1_strided.view(),
                    TensorView::absent(DataType::Float32),
                    fc2_strided.view(),
                ],
                &mut [copied.view_mut()],
            )
            .unwrap();
        assert_eq!(borrowed.to_f32(), copied.to_f32());
    }

    /// `linear_grouped` hands MLAS a `transB` GEMM instead of materializing
    /// `[in, out]`. Pin it against a hand-rolled transposed multiply.
    #[test]
    fn linear_grouped_matches_a_hand_rolled_transposed_gemm() {
        const ROWS: usize = 5;
        const IN: usize = 7;
        const OUT: usize = 9;
        let input = ramp(ROWS * IN, 23, 0.3, 0.05);
        let weights_nk = ramp(OUT * IN, 29, 0.3, 0.05);
        let bias = ramp(OUT, 11, 0.2, 0.1);
        for bias in [None, Some(&bias[..])] {
            let got = linear_grouped(&input, ROWS, &weights_nk, bias, OUT, IN).unwrap();
            let mut want = vec![0.0f32; ROWS * OUT];
            for row in 0..ROWS {
                for out in 0..OUT {
                    let mut acc = bias.map_or(0.0, |bias| bias[out]);
                    for k in 0..IN {
                        acc += input[row * IN + k] * weights_nk[out * IN + k];
                    }
                    want[row * OUT + out] = acc;
                }
            }
            assert_close(&got, &want);
        }
    }

    /// The zero-extent guard in front of the GEMM has to leave a correctly
    /// sized (empty) result rather than, say, panicking on the slice maths.
    /// MLAS itself also tolerates `M == 0`, so this pins the wrapper's
    /// contract, not MLAS's.
    #[test]
    fn linear_grouped_returns_an_empty_result_for_zero_rows() {
        let weights_nk = ramp(4 * 3, 13, 0.2, 0.05);
        let got = linear_grouped(&[], 0, &weights_nk, None, 4, 3).unwrap();
        assert!(got.is_empty());
    }
}
