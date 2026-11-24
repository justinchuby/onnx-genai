//! Float32 reference kernel for ORT 1.27 `com.microsoft::MoE`.
//!
//! The positional inputs are:
//! `input`, `router_probs`, `fc1_experts_weights`, `fc1_experts_bias?`,
//! `fc2_experts_weights`, `fc2_experts_bias?`, `fc3_experts_weights?`,
//! `fc3_experts_bias?`. Weights use ORT's expert-major canonical layout.

use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, Result, TensorMetadata, TensorMut, TensorView, WorkspaceLayout,
    WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Node};
// Only the MLAS bucketing and the `#[cfg(test)]` reference algorithm still
// need an ordered map; the driver itself orders by expert through `RoutingPlan`.
#[cfg(any(feature = "mlas", test))]
use std::collections::BTreeMap;

use super::check_arity;
use super::decoded_projection::ProjectionScratch;
use super::gelu::tanh_gelu;
#[cfg(not(feature = "mlas"))]
use super::matmul::gemm_bt;
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Relu,
    Gelu,
    Silu,
    Swiglu,
    Identity,
}

impl Activation {
    pub fn from_kernel_id(id: i32) -> Option<Self> {
        match id {
            0 => Some(Self::Relu),
            1 => Some(Self::Gelu),
            2 => Some(Self::Silu),
            3 => Some(Self::Swiglu),
            4 => Some(Self::Identity),
            _ => None,
        }
    }

    pub fn kernel_id(self) -> i32 {
        match self {
            Self::Relu => 0,
            Self::Gelu => 1,
            Self::Silu => 2,
            Self::Swiglu => 3,
            Self::Identity => 4,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Relu => "relu",
            Self::Gelu => "gelu",
            Self::Silu => "silu",
            Self::Swiglu => "swiglu",
            Self::Identity => "identity",
        }
    }
}

/// Finite default that is numerically equivalent to an unclipped SwiGLU for
/// every finite `f32` input while preserving the fail-closed attribute contract.
pub const DEFAULT_SWIGLU_LIMIT: f32 = f32::MAX;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeActivationAttributes {
    pub activation: Activation,
    pub swiglu_fusion: usize,
    pub activation_alpha: f32,
    pub activation_beta: f32,
    pub swiglu_limit: f32,
}

/// Single validation authority shared by CPU MoE/QMoE and CUDA
/// QMoE/BlockQuantizedMoE/planar-MoE.
pub fn validate_moe_activation_attributes(
    activation_name: &str,
    swiglu_fusion: i64,
    activation_alpha: f32,
    activation_beta: f32,
    swiglu_limit: f32,
) -> std::result::Result<MoeActivationAttributes, String> {
    let activation = match activation_name {
        "relu" => Activation::Relu,
        "gelu" => Activation::Gelu,
        "silu" => Activation::Silu,
        "swiglu" => Activation::Swiglu,
        "identity" => Activation::Identity,
        other => {
            return Err(format!(
                "unsupported activation_type '{other}' (supported: relu, gelu, silu, swiglu, identity)"
            ));
        }
    };
    if !(0..=2).contains(&swiglu_fusion) {
        return Err(format!(
            "swiglu_fusion must be 0, 1, or 2, got {swiglu_fusion}"
        ));
    }
    if activation != Activation::Swiglu && swiglu_fusion != 0 {
        return Err("swiglu_fusion is only valid when activation_type='swiglu'".into());
    }
    for (name, value) in [
        ("activation_alpha", activation_alpha),
        ("activation_beta", activation_beta),
    ] {
        if !value.is_finite() {
            return Err(format!("attribute {name} must be finite, got {value}"));
        }
    }
    if !swiglu_limit.is_finite() || swiglu_limit <= 0.0 {
        return Err(format!(
            "attribute swiglu_limit must be positive and finite, got {swiglu_limit}"
        ));
    }
    Ok(MoeActivationAttributes {
        activation,
        swiglu_fusion: swiglu_fusion as usize,
        activation_alpha,
        activation_beta,
        swiglu_limit,
    })
}

/// Factory for the ORT contrib `MoE` operator.
pub struct MoEFactory;

/// Float MoE kernel that groups routed rows by expert before expert GEMMs.
pub struct MoEKernel {
    attributes: MoeAttributes,
    // Fixed at creation so metadata bounds and dispatch use the same policy.
    grouped_min_work: u64,
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

/// Every attribute check `MoEFactory::create` performs, in one place so that
/// [`unsupported_reason`] and the factory cannot disagree.
fn float_moe_attributes(node: &Node) -> Result<MoeAttributes> {
    let attributes = MoeAttributes::from_node(node)?;
    if int_attr(node, "block_size", 0)? != 0 {
        return Err(error(
            "block_size is a QMoE quantization attribute and is unsupported by float MoE",
        ));
    }
    Ok(attributes)
}

/// Why this kernel cannot run a given `MoE` node, if it cannot.
///
/// Must be answered at *claim* time. `GetCapability` consults only the dtype
/// and shape filters, so a rejection raised later, in the kernel factory,
/// arrives after ORT has compiled the node onto this EP and is a **hard session
/// failure that no fallback recovers from**. ORT's own CPU `MoE` kernel runs
/// configurations this one does not yet implement — `use_sparse_mixer=1` most
/// notably, which real sparse-mixer mixtures set — so claiming and then failing
/// would take a model that works today and kill it at `CreateSession`.
///
/// Implemented by running the factory's own attribute parse, so a limit cannot
/// be added to `create` without appearing here too.
pub(crate) fn unsupported_reason(node: &Node) -> Option<String> {
    float_moe_attributes(node).err().map(|e| e.to_string())
}

impl KernelFactory for MoEFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(MoEKernel {
            attributes: float_moe_attributes(node)?,
            grouped_min_work: grouped_min_work(),
        }))
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
        let normalize = bool_attr(node, "normalize_routing_weights", false)?;
        if parse_sparse_mixer && bool_attr(node, "use_sparse_mixer", false)? {
            return Err(error(
                "use_sparse_mixer=1 is unsupported by the Phase-1 CPU reference kernel",
            ));
        }
        let swiglu_fusion = int_attr(node, "swiglu_fusion", 0)?;
        let activation_alpha = float_attr(node, "activation_alpha", 1.0)?;
        let activation_beta = float_attr(node, "activation_beta", 0.0)?;
        let swiglu_limit = float_attr(node, "swiglu_limit", DEFAULT_SWIGLU_LIMIT)?;
        let activation_attributes = validate_moe_activation_attributes(
            activation_name,
            swiglu_fusion,
            activation_alpha,
            activation_beta,
            swiglu_limit,
        )
        .map_err(error)?;
        Ok(Self {
            k: k as usize,
            activation: activation_attributes.activation,
            normalize_routing_weights: normalize,
            swiglu_fusion: activation_attributes.swiglu_fusion,
            activation_alpha: activation_attributes.activation_alpha,
            activation_beta: activation_attributes.activation_beta,
            swiglu_limit: activation_attributes.swiglu_limit,
        })
    }

    fn swiglu(&self, gate: f32, linear: f32) -> f32 {
        let g = gate.min(self.swiglu_limit);
        let l = linear.clamp(-self.swiglu_limit, self.swiglu_limit);
        g * sigmoid(self.activation_alpha * g) * (l + self.activation_beta)
    }
}

impl MoEKernel {
    fn workspace_layout(
        &self,
        shape: &[usize],
        inter: usize,
        has_fc3: bool,
    ) -> Result<WorkspaceLayout<3>> {
        if !matches!(shape.len(), 2 | 3) {
            return Err(error(format!(
                "input must be 2-D [rows, hidden] or 3-D [batch, sequence, hidden], got {shape:?}"
            )));
        }
        let invalid = || {
            error(format!(
                "execution workspace overflows for input {shape:?}, inter_size={inter}, k={}; reduce batch or projection dimensions",
                self.attributes.k
            ))
        };
        let rows = shape[..shape.len() - 1]
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(invalid)?;
        let hidden = shape[shape.len() - 1];
        let input_elements = rows.checked_mul(hidden).ok_or_else(invalid)?;
        let accumulator =
            std::alloc::Layout::array::<f32>(input_elements).map_err(|_| invalid())?;
        let fc1 = self.attributes.checked_fc1_size(inter, "MoE")?;
        let grouped_possible =
            projection_work_units(rows, fc1, hidden, inter) >= self.grouped_min_work;
        let bound = if hidden == 0 {
            0
        } else if grouped_possible {
            rows.checked_mul(self.attributes.k).ok_or_else(invalid)?
        } else {
            rows
        };
        let separate_gate = self.attributes.uses_separate_gate(has_fc3);
        let widths = if self.attributes.is_gated(has_fc3) {
            // A: gather -> activated; B: FC1+FC3 -> expert output.
            [
                hidden.max(inter),
                fc1.checked_add(if separate_gate { inter } else { 0 })
                    .ok_or_else(invalid)?
                    .max(hidden),
            ]
        } else {
            // A: gather -> expert output; B: FC1 -> in-place activation.
            [hidden, inter]
        };
        let regions = widths.map(|width| {
            let elements = bound.checked_mul(width).ok_or_else(invalid)?;
            std::alloc::Layout::array::<f32>(elements).map_err(|_| invalid())
        });
        let [a, b] = regions;
        WorkspaceLayout::new([a?, b?, accumulator])
    }
}

impl Kernel for MoEKernel {
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        _outputs: &[onnx_runtime_ep_api::OutputMetadata<'_>],
    ) -> Result<onnx_runtime_ep_api::WorkspacePreparation> {
        let required = |index: usize, name: &str| {
            inputs
                .get(index)
                .filter(|input| input.present)
                .ok_or_else(|| {
                    error(format!(
                        "required input {index} ('{name}') is absent during workspace preparation"
                    ))
                })
        };
        let input = required(0, "input")?;
        if !matches!(
            input.dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            return Err(error(format!(
                "input 0 requires Float32, Float16, or BFloat16, got {:?}",
                input.dtype
            )));
        }
        let fc2 = required(4, "fc2_experts_weights")?;
        require_shape("fc2_experts_weights", fc2.shape, 3)?;
        Ok(onnx_runtime_ep_api::WorkspacePreparation::Ready(
            WorkspaceRequirement {
                bytes: self
                    .workspace_layout(
                        input.shape,
                        fc2.shape[2],
                        inputs.get(6).is_some_and(|v| v.present),
                    )?
                    .bytes() as u64,
                alignment: std::mem::align_of::<f32>(),
                ..WorkspaceRequirement::NONE
            },
        ))
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.execute_with_workspace(inputs, outputs, None)
    }

    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
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
        let rows = x_shape[..x_shape.len() - 1]
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| {
                error(format!(
                    "flattened input row count overflows for {x_shape:?}; reduce batch dimensions"
                ))
            })?;
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

        let layout = self.workspace_layout(x_shape, inter, has_fc3)?;
        // SAFETY: exclusive Host backing lives through this synchronous call.
        let [input, projection, mut accumulator] = unsafe {
            ProjectionScratch::bind_layout("MoE execution workspace", layout, workspace)?
        };
        let mut scratch = MoeWorkspace { input, projection };

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
        let driver =
            if use_grouped_driver(&plan, expected_fc1, hidden, inter, self.grouped_min_work) {
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
            &mut scratch,
            &mut accumulator,
        )?;
        write_dense_f32_narrow("MoE", &mut outputs[0], output)
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

    fn is_gated(&self, has_fc3: bool) -> bool {
        self.activation == Activation::Swiglu || (self.activation == Activation::Silu && has_fc3)
    }

    fn gated_pair_indices(&self, index: usize, inter: usize, separate: bool) -> (usize, usize) {
        if separate {
            (index, index)
        } else if self.swiglu_fusion == 1 {
            (2 * index, 2 * index + 1)
        } else {
            (index, inter + index)
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
            let expert = &weights[group.expert * expert_stride..(group.expert + 1) * expert_stride];
            gemm_bt(
                &input[group.start * in_features..(group.start + group.len) * in_features],
                expert,
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
/// driver on for every shape with multiple workers. Read at kernel creation.
const MOE_GROUPED_MIN_WORK: u64 = 300_000_000;

/// Whether the grouped/batched driver is expected to beat the per-expert loop.
///
/// The discriminator is the work in *one* expert's GEMMs, not the total: the
/// grouped driver's win comes from parallelising the gather, activation and
/// scatter, which only matters once MLAS is already saturating the machine
/// inside each expert's GEMM. The largest group determines that opportunity;
/// `fc1_size + inter` counts both the FC1(+FC3) and FC2 passes.
fn use_grouped_driver(
    plan: &RoutingPlan,
    fc1_size: usize,
    hidden: usize,
    inter: usize,
    min_work: u64,
) -> bool {
    let groups = plan.groups.len();
    if groups == 0 {
        return true;
    }
    if moe_worker_budget() < 2 {
        // Single-threaded: the two drivers measured identically, so prefer the
        // one with fewer intermediate buffers.
        return false;
    }
    grouped_work_units(plan, fc1_size, hidden, inter) >= min_work
}

/// The gate's work estimate, split out so it can be pinned by a test on a
/// machine of any width.
///
/// The largest group rather than the mean: a collapsed router that sends most
/// of its rows to one expert still has one large GEMM to hide the elementwise
/// stages behind, and the mean would hide it.
fn grouped_work_units(plan: &RoutingPlan, fc1_size: usize, hidden: usize, inter: usize) -> u64 {
    let rows_per_group = plan.groups.iter().map(|group| group.len).max().unwrap_or(0);
    projection_work_units(rows_per_group, fc1_size, hidden, inter)
}

fn projection_work_units(rows: usize, fc1_size: usize, hidden: usize, inter: usize) -> u64 {
    (rows as u64)
        .saturating_mul(hidden as u64)
        .saturating_mul((fc1_size as u64).saturating_add(inter as u64))
}

fn grouped_min_work() -> u64 {
    // Read once per kernel, never process-global: later environment changes
    // apply to newly created kernels, not already prepared workspace plans.
    std::env::var("ONNX_GENAI_MOE_GROUPED_MIN_WORK")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(MOE_GROUPED_MIN_WORK)
}

/// The per-expert driver: gather each expert's rows into the shared lane, run
/// the pair of expert GEMMs, then accumulate the weighted result.
///
/// Kept as the small-shape path: compute one expert at a time, retaining the
/// serial gather, activation and scatter policy. Both scratch banks are reused
/// by every expert rather than allocating intermediate matrices per group.
///
/// Iterating `plan.groups` reproduces the ascending-expert order, and slots
/// inside a group are in ascending row order, so this is **bit-identical** to
/// the grouped driver rather than merely close.
#[allow(clippy::too_many_arguments)]
fn run_moe_per_expert<'out>(
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
    scratch: &mut MoeWorkspace<'_>,
    accumulator: &'out mut ProjectionScratch<'_>,
) -> Result<&'out [f32]> {
    let output = accumulator.zeroed_prefix(plan.rows * hidden)?;
    if plan.slots() == 0 || hidden == 0 {
        return Ok(output);
    }
    for group in &plan.groups {
        let expert = group.expert;
        let slots = group.start..group.start + group.len;
        let expert_out = scratch.run_dense_expert(
            x,
            &plan.slot_row[slots.clone()],
            DenseExpertWeights {
                fc1: &fc1_weights[expert * fc1_size * hidden..(expert + 1) * fc1_size * hidden],
                fc1_bias: fc1_bias.map(|b| &b[expert * fc1_size..(expert + 1) * fc1_size]),
                fc2: &fc2_weights[expert * hidden * inter..(expert + 1) * hidden * inter],
                fc2_bias: fc2_bias.map(|b| &b[expert * hidden..(expert + 1) * hidden]),
                fc3: fc3_weights
                    .map(|w| &w[expert * inter * hidden..(expert + 1) * inter * hidden]),
                fc3_bias: fc3_bias.map(|b| &b[expert * inter..(expert + 1) * inter]),
            },
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
fn run_moe_grouped<'out>(
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
    scratch: &mut MoeWorkspace<'_>,
    accumulator: &'out mut ProjectionScratch<'_>,
) -> Result<&'out [f32]> {
    let slots = plan.slots();
    let output = accumulator.zeroed_prefix(plan.rows * hidden)?;
    if slots == 0 || hidden == 0 {
        return Ok(output);
    }
    debug_assert_eq!(
        plan.row_slots.len(),
        plan.rows * plan.k,
        "run_moe_grouped needs the row->slot inversion; call build_row_slots first"
    );

    let gathered = match parallel_rows_per_task(slots, hidden) {
        Some(rows_per_task) => scratch.input.gather_rows_parallel(
            x,
            hidden,
            &plan.slot_row,
            std::num::NonZeroUsize::new(rows_per_task).expect("parallel row tasks are nonempty"),
        )?,
        None => {
            scratch
                .input
                .gather_rows(x, hidden, plan.slot_row.iter().map(|&row| row as usize))?
        }
    };
    let separate = attributes.uses_separate_gate(fc3_weights.is_some());
    let projections = scratch
        .projection
        .zeroed_prefix(slots * (fc1_size + if separate { inter } else { 0 }))?;
    let (fc1_out, gate) = projections.split_at_mut(slots * fc1_size);
    grouped_linear(gathered, plan, fc1_weights, fc1_size, hidden, fc1_out)?;
    let gate = if separate {
        grouped_linear(
            gathered,
            plan,
            fc3_weights.expect("validated separate gate FC3"),
            inter,
            hidden,
            gate,
        )?;
        Some(&*gate)
    } else {
        None
    };
    let expert_out = if attributes.is_gated(fc3_weights.is_some()) {
        let activated = scratch.input.zeroed_prefix(slots * inter)?;
        apply_gated_activation(
            fc1_out, gate, plan, fc1_bias, fc3_bias, fc1_size, inter, attributes, activated,
        );
        let expert_out = scratch.projection.zeroed_prefix(slots * hidden)?;
        grouped_linear(activated, plan, fc2_weights, hidden, inter, expert_out)?;
        expert_out
    } else {
        for_each_row(fc1_out, fc1_size, |slot, row| {
            let bias = expert_bias(fc1_bias, &plan.slot_expert, slot, fc1_size);
            for (index, value) in row.iter_mut().enumerate() {
                *value = activate(
                    attributes.activation,
                    *value + bias.map_or(0.0, |bias| bias[index]),
                );
            }
        });
        let expert_out = scratch.input.zeroed_prefix(slots * hidden)?;
        grouped_linear(fc1_out, plan, fc2_weights, hidden, inter, expert_out)?;
        expert_out
    };

    let (row_slots, slot_weight, slot_expert, k) = (
        &plan.row_slots,
        &plan.slot_weight,
        &plan.slot_expert,
        plan.k,
    );
    for_each_row(output, hidden, |row, destination| {
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

/// Per-expert projections have already added bias; grouped projections add it
/// here, including their existing +0 operation when an optional bias is absent.
/// Distinct static states keep that decision out of the element loop.
trait ActivationBias: Copy {
    fn value(self, linear: bool, index: usize, value: f32) -> f32;
}

#[derive(Clone, Copy)]
struct AppliedActivationBias;

impl ActivationBias for AppliedActivationBias {
    fn value(self, _linear: bool, _index: usize, value: f32) -> f32 {
        value
    }
}

#[derive(Clone, Copy)]
struct PendingActivationBias<'a> {
    fc1: Option<&'a [f32]>,
    linear: Option<&'a [f32]>,
}

impl ActivationBias for PendingActivationBias<'_> {
    fn value(self, linear: bool, index: usize, value: f32) -> f32 {
        value + if linear { self.linear } else { self.fc1 }.map_or(0.0, |bias| bias[index])
    }
}

fn gated_activation_row(
    output: &mut [f32],
    fc1: &[f32],
    fc3: Option<&[f32]>,
    inter: usize,
    attributes: &MoeAttributes,
    bias: impl ActivationBias,
) {
    let linear = fc3.unwrap_or(fc1);
    for (index, value) in output.iter_mut().enumerate() {
        let (g, l) = attributes.gated_pair_indices(index, inter, fc3.is_some());
        *value = attributes.swiglu(bias.value(false, g, fc1[g]), bias.value(true, l, linear[l]));
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_gated_activation(
    fc1_out: &[f32],
    gate_source: Option<&[f32]>,
    plan: &RoutingPlan,
    fc1_bias: Option<&[f32]>,
    fc3_bias: Option<&[f32]>,
    fc1_size: usize,
    inter: usize,
    attributes: &MoeAttributes,
    activated: &mut [f32],
) {
    for_each_row(activated, inter, |slot, row| {
        let fc1 = &fc1_out[slot * fc1_size..(slot + 1) * fc1_size];
        let fc3 = gate_source.map(|gate| &gate[slot * inter..(slot + 1) * inter]);
        let fc1_bias = expert_bias(fc1_bias, &plan.slot_expert, slot, fc1_size);
        let linear_bias = if fc3.is_some() {
            expert_bias(fc3_bias, &plan.slot_expert, slot, inter)
        } else {
            fc1_bias
        };
        gated_activation_row(
            row,
            fc1,
            fc3,
            inter,
            attributes,
            PendingActivationBias {
                fc1: fc1_bias,
                linear: linear_bias,
            },
        );
    });
}

/// Two exclusive banks, reused only after the previous phase's final read.
struct MoeWorkspace<'a> {
    input: ProjectionScratch<'a>,
    projection: ProjectionScratch<'a>,
}
struct DenseExpertWeights<'a> {
    fc1: &'a [f32],
    fc1_bias: Option<&'a [f32]>,
    fc2: &'a [f32],
    fc2_bias: Option<&'a [f32]>,
    fc3: Option<&'a [f32]>,
    fc3_bias: Option<&'a [f32]>,
}
impl MoeWorkspace<'_> {
    #[allow(clippy::too_many_arguments)]
    fn run_dense_expert(
        &mut self,
        x: &[f32],
        row_indices: &[u32],
        weights: DenseExpertWeights<'_>,
        fc1_size: usize,
        hidden: usize,
        inter: usize,
        attributes: &MoeAttributes,
    ) -> Result<&[f32]> {
        let rows = row_indices.len();
        let input = if rows == 1 {
            let start = row_indices[0] as usize * hidden;
            &x[start..start + hidden]
        } else {
            self.input
                .gather_rows(x, hidden, row_indices.iter().map(|&row| row as usize))?
        };
        let separate = attributes.uses_separate_gate(weights.fc3.is_some());
        let projections = self
            .projection
            .zeroed_prefix(rows * (fc1_size + if separate { inter } else { 0 }))?;
        let (fc1, fc3) = projections.split_at_mut(rows * fc1_size);
        linear_grouped_into(
            input,
            rows,
            weights.fc1,
            weights.fc1_bias,
            fc1_size,
            hidden,
            fc1,
        )?;
        let fc3 = if separate {
            linear_grouped_into(
                input,
                rows,
                weights.fc3.expect("validated separate FC3"),
                weights.fc3_bias,
                inter,
                hidden,
                fc3,
            )?;
            Some(&*fc3)
        } else {
            None
        };
        if attributes.is_gated(weights.fc3.is_some()) {
            let activated = self.input.zeroed_prefix(rows * inter)?;
            // Keep the per-expert activation serial, as before.
            if inter != 0 {
                for (row, target) in activated.chunks_mut(inter).enumerate() {
                    gated_activation_row(
                        target,
                        &fc1[row * fc1_size..(row + 1) * fc1_size],
                        fc3.map(|linear| &linear[row * inter..(row + 1) * inter]),
                        inter,
                        attributes,
                        AppliedActivationBias,
                    );
                }
            }
            let output = self.projection.zeroed_prefix(rows * hidden)?;
            linear_grouped_into(
                activated,
                rows,
                weights.fc2,
                weights.fc2_bias,
                hidden,
                inter,
                output,
            )?;
            Ok(output)
        } else {
            for value in fc1.iter_mut() {
                *value = activate(attributes.activation, *value);
            }
            let output = self.input.zeroed_prefix(rows * hidden)?;
            linear_grouped_into(
                fc1,
                rows,
                weights.fc2,
                weights.fc2_bias,
                hidden,
                inter,
                output,
            )?;
            Ok(output)
        }
    }
}

/// Portable single-token reference expert: plain scalar dot products, no GEMM,
/// no transposes. Production uses grouped GEMM projections at every row count;
/// this is kept as the independent oracle the tests compare against.
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
                for pair in fc1_out.as_chunks::<2>().0 {
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
    let fc1 = DenseExpertProjection {
        weights: fc1_weights,
        out_features: fc1_size,
        in_features: hidden,
    };
    let fc2 = DenseExpertProjection {
        weights: fc2_weights,
        out_features: hidden,
        in_features: inter,
    };
    let fc3 = fc3_weights.map(|weights| DenseExpertProjection {
        weights,
        out_features: inter,
        in_features: hidden,
    });
    run_expert_projections(
        input,
        rows,
        &fc1,
        fc1_bias,
        &fc2,
        fc2_bias,
        fc3.as_ref(),
        fc3_bias,
        attributes,
        |projection, input, rows, bias| projection.apply(input, rows, bias),
    )
}

/// Projection geometry shared by dense and compressed expert execution.
/// The sequential apply callback may borrow one exclusive scratch buffer; each
/// result remains owned while subsequent projections overwrite that scratch.
pub(super) trait ExpertProjection {
    fn out_features(&self) -> usize;
    fn in_features(&self) -> usize;
}
struct DenseExpertProjection<'a> {
    weights: &'a [f32],
    out_features: usize,
    in_features: usize,
}
impl ExpertProjection for DenseExpertProjection<'_> {
    fn out_features(&self) -> usize {
        self.out_features
    }
    fn in_features(&self) -> usize {
        self.in_features
    }
}
impl DenseExpertProjection<'_> {
    fn apply(&self, input: &[f32], rows: usize, bias: Option<&[f32]>) -> Result<Vec<f32>> {
        linear_grouped(
            input,
            rows,
            self.weights,
            bias,
            self.out_features,
            self.in_features,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_expert_projections<P: ExpertProjection>(
    input: &[f32],
    rows: usize,
    fc1: &P,
    fc1_bias: Option<&[f32]>,
    fc2: &P,
    fc2_bias: Option<&[f32]>,
    fc3: Option<&P>,
    fc3_bias: Option<&[f32]>,
    attributes: &MoeAttributes,
    mut apply: impl FnMut(&P, &[f32], usize, Option<&[f32]>) -> Result<Vec<f32>>,
) -> Result<Vec<f32>> {
    let fc1_size = fc1.out_features();
    let inter = fc2.in_features();
    // A single routed token used to fall through to `run_expert`'s scalar dot
    // loop. That loop is ~50x slower per MAC than the GEMM, and decode - one
    // token, top-k experts - is the shape that matters most, so the grouped
    // path now serves `rows == 1` too. `run_expert` is retained as the
    // portable reference the tests compare against.
    let mut fc1_out = apply(fc1, input, rows, fc1_bias)?;
    let activated = match attributes.activation {
        Activation::Swiglu => {
            let (gate_part, linear_part) = if attributes.swiglu_fusion == 0 {
                let linear = apply(
                    fc3.expect("validated unfused swiglu FC3"),
                    input,
                    rows,
                    fc3_bias,
                )?;
                (fc1_out, linear)
            } else {
                let mut gate = Vec::with_capacity(rows * inter);
                let mut linear = Vec::with_capacity(rows * inter);
                for row in fc1_out.chunks_exact(fc1_size) {
                    if attributes.swiglu_fusion == 1 {
                        for pair in row.as_chunks::<2>().0 {
                            gate.push(pair[0]);
                            linear.push(pair[1]);
                        }
                    } else {
                        gate.extend_from_slice(&row[..inter]);
                        linear.extend_from_slice(&row[inter..]);
                    }
                }
                (gate, linear)
            };
            gate_part
                .into_iter()
                .zip(linear_part)
                .map(|(g, l)| attributes.swiglu(g, l))
                .collect()
        }
        Activation::Silu if fc3.is_some() => {
            let linear_part = apply(
                fc3.expect("validated SiLU gated FC3"),
                input,
                rows,
                fc3_bias,
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
    apply(fc2, &activated, rows, fc2_bias)
}

/// `output[rows, out_features] = input[rows, in_features] * weightsᵀ (+ bias)`.
///
/// Expert weights arrive as `[out_features, in_features]`, i.e. already
/// transposed relative to what a plain `A*B` GEMM wants. Both arms consume that
/// native `[n][k]` layout directly: MLAS via its `transB` flag, and the portable
/// path via [`gemm_bt`], whose dot-product inner loop is *contiguous along `k`*
/// for both operands. Neither materializes the `[in_features, out_features]`
/// copy the old portable path built - a 29 MiB strided scatter per call for a
/// Mixtral-sized expert, previously the single largest term in the operator.
fn linear_grouped(
    input: &[f32],
    rows: usize,
    weights_nk: &[f32],
    bias: Option<&[f32]>,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; rows * out_features];
    linear_grouped_into(
        input,
        rows,
        weights_nk,
        bias,
        out_features,
        in_features,
        &mut output,
    )?;
    Ok(output)
}

/// The destination is an initialized zeroed prefix, including for a zero-K
/// projection. Both owned and workspace callers establish that invariant.
#[allow(clippy::too_many_arguments)]
fn linear_grouped_into(
    input: &[f32],
    rows: usize,
    weights_nk: &[f32],
    bias: Option<&[f32]>,
    out_features: usize,
    in_features: usize,
    output: &mut [f32],
) -> Result<()> {
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
                output,
                out_features,
                true,
            );
        }
    }
    #[cfg(not(feature = "mlas"))]
    {
        gemm_bt(input, weights_nk, output, rows, in_features, out_features)?;
    }
    if let Some(bias) = bias.filter(|_| out_features != 0) {
        for row in output.chunks_exact_mut(out_features) {
            for (value, bias) in row.iter_mut().zip(bias) {
                *value += bias;
            }
        }
    }
    Ok(())
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
mod activation_validation_tests {
    use super::*;

    #[test]
    fn shared_activation_validator_covers_every_activation_fusion_pair() {
        for activation in ["relu", "gelu", "silu", "swiglu", "identity"] {
            validate_moe_activation_attributes(activation, 0, 1.0, 0.0, f32::MAX)
                .unwrap_or_else(|error| panic!("{activation}/fusion=0 rejected: {error}"));
            for fusion in [1, 2] {
                let result =
                    validate_moe_activation_attributes(activation, fusion, 1.0, 0.0, f32::MAX);
                assert_eq!(
                    result.is_ok(),
                    activation == "swiglu",
                    "activation={activation} fusion={fusion}: {result:?}"
                );
            }
        }
        assert!(validate_moe_activation_attributes("unknown", 0, 1.0, 0.0, f32::MAX).is_err());
        for fusion in [-1, 3] {
            assert!(validate_moe_activation_attributes("swiglu", fusion, 1.0, 0.0, 1.0).is_err());
        }
    }

    #[test]
    fn shared_activation_validator_rejects_every_non_finite_or_non_positive_value() {
        for (name, alpha, beta, limit) in [
            ("activation_alpha NaN", f32::NAN, 0.0, 1.0),
            ("activation_alpha +Inf", f32::INFINITY, 0.0, 1.0),
            ("activation_alpha -Inf", f32::NEG_INFINITY, 0.0, 1.0),
            ("activation_beta NaN", 1.0, f32::NAN, 1.0),
            ("activation_beta +Inf", 1.0, f32::INFINITY, 1.0),
            ("activation_beta -Inf", 1.0, f32::NEG_INFINITY, 1.0),
            ("swiglu_limit NaN", 1.0, 0.0, f32::NAN),
            ("swiglu_limit +Inf", 1.0, 0.0, f32::INFINITY),
            ("swiglu_limit -Inf", 1.0, 0.0, f32::NEG_INFINITY),
            ("swiglu_limit zero", 1.0, 0.0, 0.0),
            ("swiglu_limit negative", 1.0, 0.0, -1.0),
        ] {
            assert!(
                validate_moe_activation_attributes("swiglu", 1, alpha, beta, limit).is_err(),
                "{name} must fail closed"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuExecutionProvider;
    use crate::kernels::testutil::{Owned, PreparedTestKernel};
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{Attribute, Graph, NodeId, static_shape};
    use onnx_runtime_loader::Model;
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    #[test]
    fn gated_row_bias_states_preserve_layouts_and_signed_zero() {
        let gate = black_box([-0.0, -2.0, 3.0]);
        let linear = black_box([1.0, 2.0, -3.0]);
        let gate_bias = black_box([0.0, 0.25, -0.5]);
        let linear_bias = black_box([0.0, -0.5, 0.75]);
        // Observe rounding at the bias boundary itself. Once inlined through
        // min/clamp, optimized activation code can fold zero expressions
        // differently from a compile-time-constant scalar oracle.
        let zero = black_box(-0.0_f32);
        assert_eq!(
            AppliedActivationBias.value(false, 0, zero).to_bits(),
            (-0.0_f32).to_bits()
        );
        for fc1 in [None, Some(&gate_bias[..])] {
            assert_eq!(
                PendingActivationBias { fc1, linear: None }
                    .value(false, 0, zero)
                    .to_bits(),
                0.0_f32.to_bits()
            );
        }
        for fusion in 0..=2 {
            let attributes = MoeAttributes {
                k: 1,
                activation: Activation::Swiglu,
                normalize_routing_weights: false,
                swiglu_fusion: fusion,
                activation_alpha: 1.0,
                activation_beta: 0.0,
                swiglu_limit: DEFAULT_SWIGLU_LIMIT,
            };
            let arrange = |a: &[f32], b: &[f32]| match fusion {
                0 => a.to_vec(),
                1 => a.iter().zip(b).flat_map(|(&a, &b)| [a, b]).collect(),
                _ => [a, b].concat(),
            };
            let fc1 = arrange(&gate, &linear);
            let fc3 = (fusion == 0).then_some(&linear[..]);
            let mut applied = [29.0; 3];
            gated_activation_row(
                &mut applied,
                &fc1,
                fc3,
                3,
                &attributes,
                AppliedActivationBias,
            );
            let expected = std::array::from_fn::<_, 3, _>(|i| {
                attributes.swiglu(black_box(gate[i]), black_box(linear[i]))
            });
            assert_eq!(applied.map(f32::to_bits), expected.map(f32::to_bits));
            assert_eq!(applied[0].to_bits(), (-0.0f32).to_bits());
            for biased in [false, true] {
                let fc1_bias = arrange(&gate_bias, &linear_bias);
                let mut pending = [29.0; 3];
                gated_activation_row(
                    &mut pending,
                    &fc1,
                    fc3,
                    3,
                    &attributes,
                    PendingActivationBias {
                        fc1: biased.then_some(fc1_bias.as_slice()),
                        linear: biased.then_some(if fusion == 0 {
                            &linear_bias[..]
                        } else {
                            fc1_bias.as_slice()
                        }),
                    },
                );
                let expected = std::array::from_fn::<_, 3, _>(|i| {
                    attributes.swiglu(
                        black_box(gate[i]) + if biased { black_box(gate_bias[i]) } else { 0.0 },
                        black_box(linear[i])
                            + if biased {
                                black_box(linear_bias[i])
                            } else {
                                0.0
                            },
                    )
                });
                assert_eq!(pending.map(f32::to_bits), expected.map(f32::to_bits));
                assert_eq!(pending[0], 0.0);
            }
        }
    }

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

    fn test_accumulator(backing: &mut [f32]) -> ProjectionScratch<'_> {
        let bytes = std::mem::size_of_val(backing);
        // SAFETY: this initialized Host slice is exclusively lent by the test.
        unsafe {
            ProjectionScratch::bind(
                "MoE test accumulator",
                bytes,
                Some(WorkspaceView::new(
                    onnx_runtime_ep_api::DevicePtrMut(backing.as_mut_ptr().cast()),
                    bytes,
                )),
            )
        }
        .unwrap()
    }

    fn test_scratch(backing: &mut [f32]) -> MoeWorkspace<'_> {
        let mid = backing.len() / 2;
        let (a, b) = backing.split_at_mut(mid);
        let bind = |slice: &mut [f32]| {
            let bytes = std::mem::size_of_val(slice);
            // SAFETY: each independent Host slice is exclusively lent.
            unsafe {
                ProjectionScratch::bind(
                    "MoE test",
                    bytes,
                    Some(WorkspaceView::new(
                        onnx_runtime_ep_api::DevicePtrMut(slice.as_mut_ptr().cast()),
                        bytes,
                    )),
                )
            }
            .unwrap()
        };
        MoeWorkspace {
            input: bind(a),
            projection: bind(b),
        }
    }

    fn kernel(graph: &Graph, node: NodeId) -> Box<dyn Kernel> {
        let model = Model::new(graph);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap()
    }

    fn workspace_kernel(min_work: u64) -> MoEKernel {
        let (graph, node) = model_node(
            &[],
            &[],
            &[
                ("k", Attribute::Int(2)),
                ("activation_type", Attribute::String(b"identity".to_vec())),
                ("normalize_routing_weights", Attribute::Int(1)),
            ],
        );
        MoEKernel {
            attributes: float_moe_attributes(graph.node(node)).unwrap(),
            grouped_min_work: min_work,
        }
    }

    #[test]
    fn execution_workspace_bounds_cover_threshold_skew_and_pool_changes() {
        use crate::kernels::testutil::RecordingGovernor;
        use onnx_runtime_ep_api::DevicePtrMut;
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{
            HostWorkspacePool, MemoryGovernor, ProcessMemoryManager,
        };
        use std::sync::Arc;
        const ROWS: usize = 3;
        const H: usize = 4;
        const E: usize = 4;
        const K: usize = 2;
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let four = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let input = Owned::f32(
            &[ROWS, H],
            &[1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.],
        );
        let fc1 = Owned::f32(&[E, H, H], &[1.; E * H * H]);
        let fc2 = Owned::f32(&[E, H, H], &[1.; E * H * H]);
        // The upper bound is exactly 3*4*(4+4)=96. Test both sides,
        // plus an explicit grouped override; policy lives in the kernel.
        for threshold in [97, 96, 0] {
            let kernel = workspace_kernel(threshold);
            let bytes = ROWS * (H + H) * 4 * if threshold <= 96 { K } else { 1 } + ROWS * H * 4;
            let books = RecordingGovernor::new_host(bytes as u64);
            let manager = ProcessMemoryManager::new().unwrap();
            let pool = HostWorkspacePool::governed(&manager, Arc::new(books.clone())).unwrap();
            let cpu = CpuExecutionProvider::with_resources(
                crate::CpuMemoryResources::default()
                    .with_workspace_pool(pool.clone())
                    .unwrap(),
            );
            let mut counts = None;
            for router_values in [
                [4., 4., 0., 0., 0., 0., 4., 4., 4., 0., 0., 4.],
                [4., 4., 0., 0.].repeat(ROWS).try_into().unwrap(),
            ] {
                let router = Owned::f32(&[ROWS, E], &router_values);
                let inputs = [
                    input.view(),
                    router.view(),
                    fc1.view(),
                    TensorView::absent(DataType::Float32),
                    fc2.view(),
                ];
                let metadata: Vec<_> = inputs
                    .iter()
                    .map(|v| TensorMetadata::new(v.dtype, v.shape, !v.is_absent()))
                    .collect();
                let req = one
                    .install(|| {
                        kernel
                            .workspace_requirement(&metadata, &[])
                            .and_then(onnx_runtime_ep_api::WorkspacePreparation::require_ready)
                    })
                    .unwrap();
                assert_eq!(req.bytes, bytes as u64);
                let plan = RoutingPlan::build(&router_values, ROWS, E, K, true);
                assert!(!one.install(|| use_grouped_driver(&plan, H, H, H, threshold)));
                assert_eq!(
                    four.install(|| use_grouped_driver(&plan, H, H, H, threshold)),
                    four.install(|| moe_worker_budget() >= 2)
                        && grouped_work_units(&plan, H, H, H) >= threshold
                );
                let mut output = Owned::f32(&[ROWS, H], &[-123.; ROWS * H]);
                let mut invalid_backing = vec![0f32; bytes / 4];
                for workspace in [
                    None,
                    Some(WorkspaceView::new(
                        DevicePtrMut(invalid_backing.as_mut_ptr().cast()),
                        bytes - 4,
                    )),
                    Some(WorkspaceView::new(
                        DevicePtrMut(
                            invalid_backing
                                .as_mut_ptr()
                                .cast::<u8>()
                                .wrapping_add(1)
                                .cast(),
                        ),
                        bytes,
                    )),
                ] {
                    let err = kernel
                        .execute_with_workspace(&inputs, &mut [output.view_mut()], workspace)
                        .unwrap_err();
                    assert!(err.to_string().contains("MoE execution workspace"), "{err}");
                    assert_eq!(output.to_f32(), [-123.; ROWS * H]);
                }
                for pool_threads in [&one, &four, &one, &four] {
                    for _ in 0..17 {
                        let mut backing = cpu
                            .allocate_workspace(req.bytes as usize, req.alignment, req.role())
                            .unwrap();
                        pool_threads
                            .install(|| {
                                let view = WorkspaceView::new(
                                    DevicePtrMut(backing.as_mut_ptr()),
                                    backing.len(),
                                );
                                let inputs = [
                                    input.view(),
                                    router.view(),
                                    fc1.view(),
                                    TensorView::absent(DataType::Float32),
                                    fc2.view(),
                                ];
                                kernel.execute_with_workspace(
                                    &inputs,
                                    &mut [output.view_mut()],
                                    Some(view),
                                )
                            })
                            .unwrap();
                        cpu.deallocate_workspace(backing).unwrap();
                        // Independent identity-weight oracle: FC2(FC1(x))=16*x.
                        assert_eq!(
                            output.to_f32(),
                            [16., 16., 16., 16., 32., 32., 32., 32., 48., 48., 48., 48.]
                        );
                        let current = (books.claims(), books.releases());
                        assert_eq!(
                            *counts.get_or_insert(current),
                            current,
                            "warm workspace must never call its issuer, including zero bytes"
                        );
                        assert_eq!(books.used(Tier::Host), bytes as u64);
                        assert_eq!(manager.snapshot().unwrap().allocations.len(), 1);
                    }
                }
            }
            pool.trim_idle(u64::MAX).unwrap();
            assert_eq!(books.used(Tier::Host), 0);
            assert_eq!(manager.process_used(Tier::Host), 0);
        }
    }

    #[test]
    fn execution_workspace_metadata_checks_empty_singleton_dtype_and_overflow() {
        let kernel = workspace_kernel(MOE_GROUPED_MIN_WORK);
        let requirement = |shape: &[usize], dtype, present| {
            kernel
                .workspace_requirement(
                    &[
                        TensorMetadata::new(dtype, shape, present),
                        TensorMetadata::new(dtype, &[], false),
                        TensorMetadata::new(dtype, &[], false),
                        TensorMetadata::new(dtype, &[], false),
                        TensorMetadata::new(dtype, &[4, 4, 4], true),
                    ],
                    &[],
                )
                .and_then(onnx_runtime_ep_api::WorkspacePreparation::require_ready)
        };
        for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
            assert_eq!(requirement(&[0, 4], dtype, true).unwrap().bytes, 0);
            assert_eq!(requirement(&[1, 4], dtype, true).unwrap().bytes, 48);
            assert_eq!(requirement(&[2, 3, 4], dtype, true).unwrap().bytes, 288);
        }
        for shape in [
            &[][..],
            &[4],
            &[1, 2, 3, 4],
            &[usize::MAX, 2, 4],
            &[2, usize::MAX],
        ] {
            assert!(
                requirement(shape, DataType::Float32, true).is_err(),
                "{shape:?}"
            );
        }
        assert!(requirement(&[3, 4], DataType::Int32, true).is_err());
        assert!(requirement(&[3, 4], DataType::Float32, false).is_err());
        assert!(
            kernel
                .workspace_requirement(&[], &[])
                .and_then(onnx_runtime_ep_api::WorkspacePreparation::require_ready)
                .unwrap_err()
                .to_string()
                .contains("input 0")
        );
        assert!(
            kernel
                .workspace_requirement(
                    &[TensorMetadata::new(DataType::Float32, &[3, 4], true)],
                    &[]
                )
                .and_then(onnx_runtime_ep_api::WorkspacePreparation::require_ready)
                .unwrap_err()
                .to_string()
                .contains("fc2_experts_weights")
        );
        let grouped = workspace_kernel(0);
        assert_eq!(
            grouped.workspace_layout(&[1, 4], 4, false).unwrap().bytes(),
            80
        );
        assert!(
            grouped
                .workspace_layout(&[isize::MAX as usize / 8, 2], 4, false)
                .is_err()
        );
    }

    #[test]
    fn projection_banks_cover_gates_asymmetric_widths_and_zero_axes() {
        use crate::kernels::testutil::RecordingGovernor;
        use onnx_runtime_ep_api::DevicePtrMut;
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{
            HostWorkspacePool, MemoryGovernor, ProcessMemoryManager,
        };
        use std::sync::Arc;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for (hidden, inter) in [(4, 7), (11, 3), (4, 0), (0, 3)] {
            for rows in [0, 1, 3] {
                for (activation, fusion, separate) in [
                    (Activation::Identity, 0, false),
                    (Activation::Silu, 0, true),
                    (Activation::Swiglu, 0, true),
                    (Activation::Swiglu, 1, false),
                    (Activation::Swiglu, 2, false),
                ] {
                    for threshold in [0, u64::MAX] {
                        let attributes = MoeAttributes {
                            k: 2,
                            activation,
                            normalize_routing_weights: true,
                            swiglu_fusion: fusion,
                            activation_alpha: 1.7,
                            activation_beta: 0.2,
                            swiglu_limit: 3.0,
                        };
                        let kernel = MoEKernel {
                            attributes,
                            grouped_min_work: threshold,
                        };
                        let experts = 4;
                        let fc1_size = if fusion == 0 { inter } else { 2 * inter };
                        let x_values = ramp(rows * hidden, 11, 0.2, 0.02);
                        let router_values = [4., 4., 0., 0.].repeat(rows);
                        let w1 = ramp(experts * fc1_size * hidden, 17, 0.1, 0.01);
                        let w2 = ramp(experts * hidden * inter, 19, 0.1, 0.01);
                        let w3 = ramp(experts * inter * hidden, 23, 0.1, 0.01);
                        let b1 = ramp(experts * fc1_size, 5, 0.1, 0.01);
                        let b2 = ramp(experts * hidden, 7, 0.1, 0.01);
                        let b3 = ramp(experts * inter, 3, 0.1, 0.01);
                        let expected = reference_moe(
                            rows,
                            hidden,
                            inter,
                            experts,
                            2,
                            fc1_size,
                            &x_values,
                            &router_values,
                            &w1,
                            &w2,
                            separate.then_some(&w3),
                            Some(&b1),
                            Some(&b2),
                            separate.then_some(&b3),
                            &attributes,
                        );
                        let x = Owned::f32(&[rows, hidden], &x_values);
                        let router = Owned::f32(&[rows, experts], &router_values);
                        let w1 = Owned::f32(&[experts, fc1_size, hidden], &w1);
                        let w2 = Owned::f32(&[experts, hidden, inter], &w2);
                        let w3 = Owned::f32(&[experts, inter, hidden], &w3);
                        let b1 = Owned::f32(&[experts, fc1_size], &b1);
                        let b2 = Owned::f32(&[experts, hidden], &b2);
                        let b3 = Owned::f32(&[experts, inter], &b3);
                        let inputs = [
                            x.view(),
                            router.view(),
                            w1.view(),
                            b1.view(),
                            w2.view(),
                            b2.view(),
                            if separate {
                                w3.view()
                            } else {
                                TensorView::absent(DataType::Float32)
                            },
                            if separate {
                                b3.view()
                            } else {
                                TensorView::absent(DataType::Float32)
                            },
                        ];
                        let metadata: Vec<_> = inputs
                            .iter()
                            .map(|v| TensorMetadata::new(v.dtype, v.shape, !v.is_absent()))
                            .collect();
                        let req = kernel
                            .workspace_requirement(&metadata, &[])
                            .unwrap()
                            .require_ready()
                            .unwrap();
                        let bound = if hidden == 0 {
                            0
                        } else {
                            rows * if threshold == 0 { 2 } else { 1 }
                        };
                        let gated = activation == Activation::Swiglu || separate;
                        let widths = if gated {
                            hidden.max(inter)
                                + (fc1_size + if separate { inter } else { 0 }).max(hidden)
                        } else {
                            hidden + inter
                        };
                        assert_eq!(req.bytes, ((bound * widths + rows * hidden) * 4) as u64);
                        let books = RecordingGovernor::new_host(req.bytes);
                        let manager = ProcessMemoryManager::new().unwrap();
                        let backing_pool =
                            HostWorkspacePool::governed(&manager, Arc::new(books.clone())).unwrap();
                        let cpu = CpuExecutionProvider::with_resources(
                            crate::CpuMemoryResources::default()
                                .with_workspace_pool(backing_pool.clone())
                                .unwrap(),
                        );
                        let mut output = Owned::zeros_f32(&[rows, hidden]);
                        let mut callbacks = None;
                        for _ in 0..3 {
                            if req.bytes == 0 {
                                kernel.execute(&inputs, &mut [output.view_mut()]).unwrap();
                            } else {
                                let mut backing = cpu
                                    .allocate_workspace(
                                        req.bytes as usize,
                                        req.alignment,
                                        req.role(),
                                    )
                                    .unwrap();
                                pool.install(|| {
                                    let inputs = [
                                        x.view(),
                                        router.view(),
                                        w1.view(),
                                        b1.view(),
                                        w2.view(),
                                        b2.view(),
                                        if separate {
                                            w3.view()
                                        } else {
                                            TensorView::absent(DataType::Float32)
                                        },
                                        if separate {
                                            b3.view()
                                        } else {
                                            TensorView::absent(DataType::Float32)
                                        },
                                    ];
                                    let workspace = WorkspaceView::new(
                                        DevicePtrMut(backing.as_mut_ptr()),
                                        backing.len(),
                                    );
                                    kernel.execute_with_workspace(
                                        &inputs,
                                        &mut [output.view_mut()],
                                        Some(workspace),
                                    )
                                })
                                .unwrap();
                                cpu.deallocate_workspace(backing).unwrap();
                            }
                            assert_close(&output.to_f32(), &expected);
                            assert_eq!(
                                *callbacks.get_or_insert((books.claims(), books.releases())),
                                (books.claims(), books.releases())
                            );
                            assert_eq!(books.used(Tier::Host), req.bytes);
                        }
                        backing_pool.trim_idle(u64::MAX).unwrap();
                        assert_eq!(books.used(Tier::Host), 0);
                        assert_eq!(manager.process_used(Tier::Host), 0);
                    }
                }
            }
        }
    }

    #[test]
    fn accumulator_preserves_strided_f32_f16_and_bf16_outputs() {
        const ROWS: usize = 3;
        const H: usize = 4;
        const E: usize = 4;
        fn tensor(dtype: DataType, shape: &[usize], values: &[f32]) -> Owned {
            match dtype {
                DataType::Float32 => Owned::f32(shape, values),
                DataType::Float16 => Owned::f16(shape, values),
                DataType::BFloat16 => Owned::bf16(shape, values),
                _ => unreachable!(),
            }
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
            for threshold in [0, u64::MAX] {
                let kernel = workspace_kernel(threshold);
                let input = tensor(
                    dtype,
                    &[ROWS, H],
                    &[1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.],
                );
                let router = tensor(dtype, &[ROWS, E], &[4., 4., 0., 0.].repeat(ROWS));
                let fc1 = tensor(dtype, &[E, H, H], &[1.; E * H * H]);
                let fc2 = tensor(dtype, &[E, H, H], &[1.; E * H * H]);
                let mut output = tensor(dtype, &[ROWS, H * 2], &[-123.; ROWS * H * 2])
                    .with_view(&[ROWS, H], &[(H * 2) as i64, 2]);
                pool.install(|| {
                    kernel.execute_prepared(
                        &[
                            input.view(),
                            router.view(),
                            fc1.view(),
                            TensorView::absent(dtype),
                            fc2.view(),
                        ],
                        &mut [output.view_mut()],
                    )
                })
                .unwrap();
                let physical = match dtype {
                    DataType::Float32 => output.to_f32(),
                    DataType::Float16 => output.to_f16_as_f32(),
                    DataType::BFloat16 => output.to_bf16_as_f32(),
                    _ => unreachable!(),
                };
                for (index, value) in physical.into_iter().enumerate() {
                    assert_eq!(
                        value,
                        if index % 2 == 0 {
                            16. * (index / (2 * H) + 1) as f32
                        } else {
                            -123.
                        },
                        "dtype={dtype:?}, offset={index}"
                    );
                }
            }
        }
    }

    #[test]
    fn accumulator_finishes_reading_input_before_writing_an_aliased_output() {
        use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
        use onnx_runtime_ir::DeviceId;
        let shape = [3, 4];
        let strides = [4, 1];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let router = Owned::f32(&[3, 4], &[4., 4., 0., 0.].repeat(3));
        let fc1 = Owned::f32(&[4, 4, 4], &[1.; 64]);
        let fc2 = Owned::f32(&[4, 4, 4], &[1.; 64]);
        for threshold in [0, u64::MAX] {
            let kernel = workspace_kernel(threshold);
            let mut values: Vec<f32> = vec![1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.];
            pool.install(|| {
                let pointer = values.as_mut_ptr().cast::<std::ffi::c_void>();
                // Both views share live Host backing. The kernel stages all
                // contributions before its final write, preserving this alias.
                let input = TensorView::new(
                    DevicePtr(pointer.cast_const()),
                    DataType::Float32,
                    &shape,
                    &strides,
                    DeviceId::cpu(),
                );
                let output = TensorMut::new(
                    DevicePtrMut(pointer),
                    DataType::Float32,
                    &shape,
                    &strides,
                    DeviceId::cpu(),
                );
                kernel.execute_prepared(
                    &[
                        input,
                        router.view(),
                        fc1.view(),
                        TensorView::absent(DataType::Float32),
                        fc2.view(),
                    ],
                    &mut [output],
                )
            })
            .unwrap();
            assert_eq!(
                values,
                [16., 16., 16., 16., 32., 32., 32., 32., 48., 48., 48., 48.]
            );
        }
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
            .execute_prepared(
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
            swiglu_limit: DEFAULT_SWIGLU_LIMIT,
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
            .execute_prepared(
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
            .execute_prepared(
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
            .execute_prepared(
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
            .execute_prepared(
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
            .execute_prepared(
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
            .execute_prepared(
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
            .execute_prepared(
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
                    let mut backing = vec![0f32; plan.slots() * (HIDDEN + 3 * INTER) * 2];
                    let mut scratch = test_scratch(&mut backing);
                    let mut grouped_backing = vec![f32::NAN; rows * HIDDEN];
                    let mut grouped_accumulator = test_accumulator(&mut grouped_backing);
                    let mut expert_backing = vec![f32::NAN; rows * HIDDEN];
                    let mut expert_accumulator = test_accumulator(&mut expert_backing);
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
                        &mut scratch,
                        &mut grouped_accumulator,
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
                        &mut scratch,
                        &mut expert_accumulator,
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
                grouped_work_units(&plan, fc1_size, hidden, inter) >= MOE_GROUPED_MIN_WORK,
                expect,
                "hidden={hidden} inter={inter} experts={experts} k={k} tokens={tokens}"
            );
            if moe_worker_budget() >= 2 {
                assert_eq!(
                    use_grouped_driver(&plan, fc1_size, hidden, inter, MOE_GROUPED_MIN_WORK),
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
            grouped_work_units(&plan, fc1_size, HIDDEN, INTER) >= MOE_GROUPED_MIN_WORK,
            "the dominant group is large enough for the grouped driver"
        );
        // The same shape summarised by the mean would fall below the floor, so
        // this test fails if the statistic is ever changed back.
        let mean_work = (mean as u64) * (HIDDEN as u64) * ((fc1_size + INTER) as u64);
        assert!(mean_work < MOE_GROUPED_MIN_WORK);
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
                    let mut backing = vec![0f32; plan.slots() * (HIDDEN + 3 * INTER) * 2];
                    let mut scratch = test_scratch(&mut backing);
                    let mut result_backing = vec![f32::NAN; rows * HIDDEN];
                    let mut result_accumulator = test_accumulator(&mut result_backing);
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
                        &mut scratch,
                        &mut result_accumulator,
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
            .execute_prepared(
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
                        .execute_prepared(&views, &mut [got.view_mut()])
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
            .execute_prepared(
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
            .execute_prepared(
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

    /// Pin the grouped driver's linear stage - the decode-critical path that
    /// now feeds expert weights straight into `gemm_bt` in their native
    /// `[out][in]` layout - against a hand-rolled transposed multiply computed
    /// the old way (dot each slot's input with the expert's `[out][in]` rows).
    /// This runs for both the MLAS and non-MLAS arms.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn grouped_linear_matches_a_hand_rolled_transposed_gemm() {
        const EXPERTS: usize = 3;
        const IN: usize = 10;
        const OUT: usize = 7;
        const ROWS: usize = 6;
        // k = 1 routing in a mixed pattern so the three experts interleave.
        let mut router = vec![0.0f32; ROWS * EXPERTS];
        for row in 0..ROWS {
            router[row * EXPERTS + (row % EXPERTS)] = 5.0;
        }
        let plan = RoutingPlan::build(&router, ROWS, EXPERTS, 1, true);
        let slots = plan.slots();
        let input = ramp(slots * IN, 31, 0.3, 0.05);
        // Expert weights in ORT canonical `[experts][out][in]` layout.
        let weights = ramp(EXPERTS * OUT * IN, 29, 0.2, 0.04);

        let mut output = vec![0.0f32; slots * OUT];
        grouped_linear(&input, &plan, &weights, OUT, IN, &mut output).unwrap();

        let stride = OUT * IN;
        let mut want = vec![0.0f32; slots * OUT];
        for group in &plan.groups {
            let expert = &weights[group.expert * stride..(group.expert + 1) * stride];
            for slot in group.start..group.start + group.len {
                for o in 0..OUT {
                    let mut acc = 0.0f32;
                    for i in 0..IN {
                        acc += input[slot * IN + i] * expert[o * IN + i];
                    }
                    want[slot * OUT + o] = acc;
                }
            }
        }
        assert_close(&output, &want);
    }
}
