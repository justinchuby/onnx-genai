//! Correctness-first ORT 1.27 `com.microsoft::QMoE` CPU kernel.
//!
//! Integer expert weights use ORT's expert-major layout
//! `[experts, out_features, in_features / pack_size]`, where
//! `pack_size = 8 / expert_weight_bits`. Values are packed least-significant
//! bits first along the input-feature (K) axis. Scales use
//! `[experts, out_features, in_features / block_size]`; optional zero points
//! pack block values least-significant bits first in
//! `[experts, out_features, ceil(blocks / pack_size)]`.
//!
//! By default this preserves the resident baseline. With
//! `ONNX_GENAI_WEIGHT_OFFLOAD=1`, external mmap-backed expert tensors are
//! validated as expert-major, routes are computed first, and only the batch's
//! unique selected expert slices are dequantized one at a time and released
//! after all routed tokens consume them. Non-pageable layouts fall back to the
//! resident path without changing results.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, RwLockReadGuard, Weak};

use onnx_runtime_ep_api::{
    EpError, ExternalMmapRegion, Kernel, KernelFactory, Result, TensorBacking, TensorMut,
    TensorView,
};
use onnx_runtime_ir::{DataType, Node};
use onnx_runtime_loader::{
    ExpertQuantization, ExpertStorageOrder, ExpertTensorLayout, WeightRegionCatalog,
};

use super::matmul_nbits::dequantize_nbits_row;
use super::moe::{
    ExpertProjection, MoeAttributes, routing_weights, run_expert_grouped, run_expert_projections,
};
use super::{check_arity, quantized_input_bytes, to_dense_f32, write_dense_f32};
use crate::DerivedWeightMemory;
use crate::resources::{DenseWeightGrant, WeightAdmission};
use crate::weight_offload::{
    WeightOffloadMode, checked_host_budget, effective_host_budget_u64, metrics,
    weight_offload_prefetch_default,
};

/// Factory for the ORT contrib `QMoE` operator.
pub struct QMoEFactory {
    host_cache: WeightOffloadHostCache,
}

impl QMoEFactory {
    pub(crate) fn new(host_cache: WeightOffloadHostCache) -> Self {
        Self { host_cache }
    }
}

/// One routed expert and its `(row, contribution_slot, route_weight)` tasks,
/// as materialised for the route-first offload loop.
type RoutedExpert = (usize, Vec<(usize, usize, f32)>);

/// Route-grouped, per-active-expert dequantizing integer QMoE kernel.
pub struct QMoEKernel {
    layer_id: u32,
    attributes: MoeAttributes,
    bits: usize,
    block_size: usize,
    weight_offload: WeightOffloadMode,
    host_cache: WeightOffloadHostCache,
    prefetch_experts: bool,
}

/// Why this kernel cannot run a given `QMoE` node, if it cannot.
///
/// This has to be answered at *claim* time, not in `create`. `GetCapability`
/// consults only the dtype and shape filters; a rejection raised later, in the
/// kernel factory, arrives after ORT has already compiled the node onto this EP
/// and is a **hard session failure that no fallback recovers from** — strictly
/// worse for the user than never claiming the node.
///
/// The sharpest case is column-wise quantization. ORT's schema leaves
/// `block_size` with no default, so an absent attribute means one scale per
/// output row (`[experts, out_features]`), while this kernel implements only
/// the blocked form (`[experts, out_features, in_features / block_size]`, block
/// a power of two ≥ 16). ORT's own CPU kernel *does* run the column-wise form,
/// so claiming and then failing would take a working model and kill it. The
/// same is true of `use_sparse_mixer=1` and of non-integer `quant_type`.
/// Declining here is a capability answer, not a performance one; the fix is to
/// implement the missing paths, tracked in the benchmark doc's §23.6.
///
/// Implemented by running the factory's own attribute parse, so a limit cannot
/// be added to `create` without appearing here too.
pub(crate) fn unsupported_reason(node: &Node) -> Option<String> {
    quantized_moe_attributes(node).err().map(|e| e.to_string())
}

/// Everything `QMoEFactory::create` validates, in one place so that
/// [`unsupported_reason`] and the factory cannot disagree.
fn quantized_moe_attributes(node: &Node) -> Result<(MoeAttributes, usize, usize)> {
    let attributes = MoeAttributes::from_node(node)?;
    let bits = int_attr(node, "expert_weight_bits", 4)?;
    if !matches!(bits, 1 | 2 | 4 | 8) {
        return Err(error(format!(
            "expert_weight_bits must be one of {{1, 2, 4, 8}}, got {bits}"
        )));
    }
    let block_size = int_attr(node, "block_size", 0)?;
    if block_size < 16 || !(block_size as usize).is_power_of_two() {
        return Err(error(format!(
            "block_size must be a power of two and at least 16, got {block_size}"
        )));
    }
    let quant_type = match node.attr("quant_type") {
        Some(attr) => attr
            .as_str()
            .ok_or_else(|| error("attribute quant_type must be a string"))?,
        None => "int",
    };
    if quant_type != "int" {
        return Err(error(format!(
            "quant_type='{quant_type}' is unsupported; this kernel implements integer affine QMoE only"
        )));
    }
    Ok((attributes, bits as usize, block_size as usize))
}

impl KernelFactory for QMoEFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let (attributes, bits, block_size) = quantized_moe_attributes(node)?;
        Ok(Box::new(QMoEKernel {
            layer_id: node.id.0,
            attributes,
            bits,
            block_size,
            weight_offload: WeightOffloadMode::from_env(),
            host_cache: self.host_cache.clone(),
            prefetch_experts: weight_offload_prefetch_default(),
        }))
    }
}

impl Kernel for QMoEKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("QMoE", inputs, outputs, 7, 21, 1)?;
        for (index, name) in [
            (0, "input"),
            (1, "router_probs"),
            (2, "fc1_experts_weights"),
            (3, "fc1_scales"),
            (5, "fc2_experts_weights"),
            (6, "fc2_scales"),
        ] {
            if inputs[index].is_absent() {
                return Err(error(format!(
                    "required input {index} ('{name}') is absent"
                )));
            }
        }
        require_dtype("input", &inputs[0], DataType::Float32)?;
        require_dtype("router_probs", &inputs[1], DataType::Float32)?;
        require_dtype("fc1_experts_weights", &inputs[2], DataType::Uint8)?;
        require_dtype("fc1_scales", &inputs[3], DataType::Float32)?;
        require_dtype("fc2_experts_weights", &inputs[5], DataType::Uint8)?;
        require_dtype("fc2_scales", &inputs[6], DataType::Float32)?;
        if outputs[0].dtype != DataType::Float32 {
            return Err(error(format!(
                "output requires Float32, got {:?}",
                outputs[0].dtype
            )));
        }
        for (index, name) in [
            (4, "fc1_experts_bias"),
            (7, "fc2_experts_bias"),
            (9, "fc3_scales"),
            (10, "fc3_experts_bias"),
            (14, "router_weights"),
        ] {
            if let Some(input) = optional_input(inputs, index) {
                require_dtype(name, input, DataType::Float32)?;
            }
        }
        for (index, name) in [
            (8, "fc3_experts_weights"),
            (11, "fc1_zero_points"),
            (12, "fc2_zero_points"),
            (13, "fc3_zero_points"),
        ] {
            if let Some(input) = optional_input(inputs, index) {
                require_dtype(name, input, DataType::Uint8)?;
            }
        }
        if let Some((index, _)) = inputs
            .iter()
            .enumerate()
            .skip(15)
            .find(|(_, input)| !input.is_absent())
        {
            return Err(error(format!(
                "input {index} is only used by FP4/FP8 QMoE modes, which are unsupported by this integer kernel"
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
        let rows = checked_product(&x_shape[..x_shape.len() - 1], "flattened input row count")?;
        for (index, input) in inputs
            .iter()
            .enumerate()
            .filter(|(_, input)| !input.is_absent())
        {
            checked_tensor_layout(&format!("input {index}"), input.shape, input.dtype)?;
        }
        let output_elements = checked_tensor_layout("output", outputs[0].shape, outputs[0].dtype)?;
        require_rank("router_probs", inputs[1].shape, 2)?;
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
        if !hidden.is_multiple_of(self.block_size) {
            return Err(error(format!(
                "hidden_size {hidden} must be divisible by block_size {}",
                self.block_size
            )));
        }

        require_rank("fc2_experts_weights", inputs[5].shape, 3)?;
        if inputs[5].shape[0] != experts || inputs[5].shape[1] != hidden {
            return Err(error(format!(
                "fc2_experts_weights must start with [experts={experts}, hidden={hidden}], got {:?}",
                inputs[5].shape
            )));
        }
        let pack_size = 8 / self.bits;
        let inter = inputs[5].shape[2]
            .checked_mul(pack_size)
            .ok_or_else(|| error("fc2 inter_size overflow"))?;
        if inter == 0 || inter % self.block_size != 0 {
            return Err(error(format!(
                "inferred inter_size {inter} must be non-zero and divisible by block_size {}",
                self.block_size
            )));
        }
        let fc1_size = self.attributes.checked_fc1_size(inter, "QMoE")?;

        let fc1 = QuantizedExperts::new(
            "fc1",
            &inputs[2],
            &inputs[3],
            optional_input(inputs, 11),
            experts,
            fc1_size,
            hidden,
            self.bits,
            self.block_size,
            self.weight_offload.enabled,
        )?;
        let fc2 = QuantizedExperts::new(
            "fc2",
            &inputs[5],
            &inputs[6],
            optional_input(inputs, 12),
            experts,
            hidden,
            inter,
            self.bits,
            self.block_size,
            self.weight_offload.enabled,
        )?;

        validate_bias("fc1_experts_bias", inputs, 4, experts, fc1_size)?;
        validate_bias("fc2_experts_bias", inputs, 7, experts, hidden)?;
        let fc1_bias = optional_dense(inputs, 4)?;
        let fc2_bias = optional_dense(inputs, 7)?;

        let has_fc3 = optional_input(inputs, 8).is_some();
        let uses_separate_gate = self.attributes.uses_separate_gate(has_fc3);
        let (fc3, fc3_bias) = if uses_separate_gate {
            let weights = optional_input(inputs, 8)
                .ok_or_else(|| error("unfused swiglu requires input 8 fc3_experts_weights"))?;
            let scales = optional_input(inputs, 9)
                .ok_or_else(|| error("fc3_experts_weights requires input 9 fc3_scales"))?;
            validate_bias("fc3_experts_bias", inputs, 10, experts, inter)?;
            (
                Some(QuantizedExperts::new(
                    "fc3",
                    weights,
                    scales,
                    optional_input(inputs, 13),
                    experts,
                    inter,
                    hidden,
                    self.bits,
                    self.block_size,
                    self.weight_offload.enabled,
                )?),
                optional_dense(inputs, 10)?,
            )
        } else {
            for (index, name) in [
                (8, "fc3_experts_weights"),
                (9, "fc3_scales"),
                (10, "fc3_experts_bias"),
                (13, "fc3_zero_points"),
            ] {
                if optional_input(inputs, index).is_some() {
                    return Err(error(format!(
                        "{name} is only valid for unfused swiglu or silu gated-GLU"
                    )));
                }
            }
            (None, None)
        };

        if let Some(router_weights) = optional_input(inputs, 14) {
            require_exact_shape("router_weights", router_weights.shape, &[rows, experts])?;
        }

        let x = to_dense_f32(&inputs[0])?;
        let router = to_dense_f32(&inputs[1])?;
        let aggregation = optional_dense(inputs, 14)?;
        let expected_output_elements = checked_product(&[rows, hidden], "output element count")?;
        if output_elements != expected_output_elements {
            return Err(error(format!(
                "output has {output_elements} elements, expected {expected_output_elements}"
            )));
        }
        let mut output = vec![0.0f32; output_elements];
        let mut routes = Vec::with_capacity(rows);
        let mut token_counts = BTreeMap::new();
        let mut tasks = BTreeMap::<usize, Vec<(usize, usize, f32)>>::new();
        let mut route_slots = 0usize;
        for row in 0..rows {
            let router_range = checked_range(row, experts, "router row")?;
            let route = routing_weights(
                &router[router_range.clone()],
                aggregation.as_deref().map(|weights| &weights[router_range]),
                self.attributes.k,
                self.attributes.normalize_routing_weights,
            );
            // Investigation probe (default-OFF): mirror of the CUDA route dump so
            // a CPU-vs-CUDA run can be diffed for benign borderline-argmax
            // reassociation vs a real router top-k divergence.
            if std::env::var_os("ONNX_GENAI_QMOE_ROUTE_DUMP").is_some() {
                static CALL: AtomicU64 = AtomicU64::new(0);
                let call = CALL.fetch_add(1, Ordering::Relaxed);
                let row_logits = &router[checked_range(row, experts, "router row")?];
                let order: Vec<usize> = route.iter().map(|&(e, _)| e).collect();
                let mut sorted = order.clone();
                sorted.sort_unstable();
                let selected_set: std::collections::HashSet<usize> =
                    order.iter().copied().collect();
                let min_selected = order
                    .iter()
                    .map(|&e| row_logits[e])
                    .fold(f32::INFINITY, f32::min);
                let max_rejected = (0..experts)
                    .filter(|e| !selected_set.contains(e))
                    .map(|e| row_logits[e])
                    .fold(f32::NEG_INFINITY, f32::max);
                let margin = min_selected - max_rejected;
                eprintln!(
                    "QMOE_ROUTE_CPU call={call} row={row} order={order:?} set={sorted:?} \
                     min_sel_logit={min_selected:.8e} max_rej_logit={max_rejected:.8e} \
                     margin={margin:.8e}"
                );
            }
            for &(expert, route_weight) in &route {
                *token_counts.entry(expert).or_insert(0usize) += 1;
                tasks
                    .entry(expert)
                    .or_default()
                    .push((row, route_slots, route_weight));
                route_slots = route_slots
                    .checked_add(1)
                    .ok_or_else(|| error("routed contribution count overflow"))?;
            }
            routes.push(route);
        }
        let contribution_elements =
            checked_product(&[route_slots, hidden], "routed contribution element count")?;
        checked_byte_count(
            contribution_elements,
            std::mem::size_of::<f32>(),
            "routed contribution byte count",
        )?;
        let mut contributions = vec![0.0f32; contribution_elements];
        let route_first = self.weight_offload.enabled
            && fc1.is_pageable()
            && fc2.is_pageable()
            && fc3.as_ref().is_none_or(QuantizedExperts::is_pageable);
        if route_first {
            let mut mapped_regions = Vec::new();
            mapped_regions.extend_from_slice(fc1.mapped_regions());
            mapped_regions.extend_from_slice(fc2.mapped_regions());
            if let Some(weights) = &fc3 {
                mapped_regions.extend_from_slice(weights.mapped_regions());
            }
            metrics()
                .record_mapped_regions(&mapped_regions)
                .map_err(error)?;
            metrics().record_routes(self.layer_id, &token_counts);

            let expanded_bytes = DequantizedExpert::expanded_bytes(&fc1, &fc2, fc3.as_ref())?;
            let ordered: Vec<RoutedExpert> = tasks.into_iter().collect();
            let cache_operation = self.host_cache.begin_operation(
                ExpertCacheKey::new(
                    self.layer_id,
                    0,
                    self.bits,
                    self.block_size,
                    &fc1,
                    &fc2,
                    fc3.as_ref(),
                ),
                experts,
                expanded_bytes,
            )?;
            // Every bank with an active operation or retained entry participates
            // in the capacity proof. The operation freezes that registry until
            // its worker finishes, so another layer cannot invalidate the proof
            // by registering and admitting a previously unseen bank midway.
            let compute = |expert: usize,
                           expert_tasks: &[(usize, usize, f32)],
                           weights: &ExpertExecution,
                           contributions: &mut [f32]|
             -> Result<()> {
                write_grouped_contributions(
                    contributions,
                    &x,
                    expert_tasks,
                    expert,
                    weights,
                    &fc1,
                    &fc2,
                    fc3.as_ref(),
                    fc1_bias.as_deref(),
                    fc2_bias.as_deref(),
                    fc3_bias.as_deref(),
                    fc1_size,
                    hidden,
                    inter,
                    &self.attributes,
                )
            };

            let prefetch_memory = if self.prefetch_experts
                && ordered.len() > 1
                && cache_operation.prefetch_eligibility().allows_pipeline()
            {
                cache_operation.prefetch_memory(ordered.iter().map(|(expert, _)| *expert))?
            } else {
                None
            };
            if let Some(prefetch_memory) = prefetch_memory {
                // Single-ahead prefetch pipeline: a dedicated worker leases
                // (and, on a miss, dequantizes) expert `n+1` while the main
                // thread computes expert `n`, overlapping the host-side
                // transfer with compute. Leases are still issued strictly in
                // ascending expert order (one request in flight). Combined with
                // the eviction-neutrality guard above, cache admission/eviction
                // and every counter stay byte-identical to the serial path —
                // only the wall-clock overlaps.
                use std::sync::mpsc;
                let host_cache = &cache_operation;
                let fc1_ref = &fc1;
                let fc2_ref = &fc2;
                let fc3_ref = fc3.as_ref();
                let ordered_ref = &ordered;
                let prefetch_memory = &prefetch_memory;
                std::thread::scope(|scope| -> Result<()> {
                    let (req_tx, req_rx) = mpsc::channel::<usize>();
                    let (res_tx, res_rx) = mpsc::channel::<Result<ExpertExecution>>();
                    scope.spawn(move || {
                        while let Ok(index) = req_rx.recv() {
                            let expert = ordered_ref[index].0;
                            let leased = lease_route_first_expert(
                                host_cache,
                                expert,
                                fc1_ref,
                                fc2_ref,
                                fc3_ref,
                                prefetch_memory,
                            );
                            if res_tx.send(leased).is_err() {
                                break;
                            }
                        }
                    });
                    req_tx
                        .send(0)
                        .map_err(|_| error("route-first prefetch worker unavailable"))?;
                    for index in 0..ordered_ref.len() {
                        let weights = res_rx
                            .recv()
                            .map_err(|_| error("route-first prefetch worker stopped"))??;
                        if index + 1 < ordered_ref.len() {
                            req_tx
                                .send(index + 1)
                                .map_err(|_| error("route-first prefetch worker unavailable"))?;
                        }
                        let (expert, expert_tasks) = &ordered_ref[index];
                        compute(*expert, expert_tasks, &weights, &mut contributions)?;
                    }
                    Ok(())
                })?;
            } else {
                for (expert, expert_tasks) in &ordered {
                    let weights = lease_route_first_expert(
                        &cache_operation,
                        *expert,
                        &fc1,
                        &fc2,
                        fc3.as_ref(),
                        &ExpertMemoryScope::OnDemand(self.host_cache.memory()),
                    )?;
                    compute(*expert, expert_tasks, &weights, &mut contributions)?;
                }
            }
        } else {
            for (expert, expert_tasks) in tasks {
                let weights = load_transient_expert(
                    expert,
                    &fc1,
                    &fc2,
                    fc3.as_ref(),
                    self.host_cache.memory(),
                    false,
                )?;
                write_grouped_contributions(
                    &mut contributions,
                    &x,
                    &expert_tasks,
                    expert,
                    &weights,
                    &fc1,
                    &fc2,
                    fc3.as_ref(),
                    fc1_bias.as_deref(),
                    fc2_bias.as_deref(),
                    fc3_bias.as_deref(),
                    fc1_size,
                    hidden,
                    inter,
                    &self.attributes,
                )?;
            }
        }
        let mut slot = 0usize;
        for (row, route) in routes.into_iter().enumerate() {
            let output_range = checked_range(row, hidden, "output row")?;
            let output_row = &mut output[output_range];
            for _ in route {
                let contribution_range = checked_range(slot, hidden, "routed contribution row")?;
                for (output_value, contribution) in output_row
                    .iter_mut()
                    .zip(&contributions[contribution_range])
                {
                    *output_value += contribution;
                }
                slot = slot
                    .checked_add(1)
                    .ok_or_else(|| error("routed contribution index overflow"))?;
            }
        }
        debug_assert_eq!(slot, route_slots);
        write_dense_f32(&mut outputs[0], &output)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

struct DequantizedExpert {
    fc1: Vec<f32>,
    fc2: Vec<f32>,
    fc3: Option<Vec<f32>>,
    _residency: Option<DequantizedExpertResidency>,
    // Payloads above are destroyed before returning their owned Host capacity.
    _grant: Option<Arc<DenseWeightGrant>>,
}

impl DequantizedExpert {
    fn load(
        expert: usize,
        fc1: &QuantizedExperts<'_>,
        fc2: &QuantizedExperts<'_>,
        fc3: Option<&QuantizedExperts<'_>>,
        track_residency: bool,
        grant: Option<Arc<DenseWeightGrant>>,
    ) -> Result<Self> {
        Ok(Self {
            fc1: fc1.dequantize_expert(expert)?,
            fc2: fc2.dequantize_expert(expert)?,
            fc3: fc3
                .map(|weights| weights.dequantize_expert(expert))
                .transpose()?,
            _residency: track_residency.then(DequantizedExpertResidency::new),
            _grant: grant,
        })
    }

    fn expanded_bytes(
        fc1: &QuantizedExperts<'_>,
        fc2: &QuantizedExperts<'_>,
        fc3: Option<&QuantizedExperts<'_>>,
    ) -> Result<usize> {
        let mut bytes = fc1.dequantized_bytes()?;
        bytes = bytes
            .checked_add(fc2.dequantized_bytes()?)
            .ok_or_else(|| error("selected expert expanded byte count overflow"))?;
        if let Some(weights) = fc3 {
            bytes = bytes
                .checked_add(weights.dequantized_bytes()?)
                .ok_or_else(|| error("selected expert expanded byte count overflow"))?;
        }
        if bytes > isize::MAX as usize {
            return Err(error(
                "selected expert expanded byte count exceeds isize::MAX",
            ));
        }
        Ok(bytes)
    }
}

struct DequantizedExpertResidency;

impl DequantizedExpertResidency {
    fn new() -> Self {
        metrics().record_dequantized_expert_materialized();
        Self
    }
}

impl Drop for DequantizedExpertResidency {
    fn drop(&mut self) {
        metrics().record_dequantized_expert_released();
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ExpertCacheKey {
    layer_id: u32,
    expert: usize,
    bits: usize,
    block_size: usize,
    out_and_in_features: Vec<(usize, usize)>,
    source_regions: Vec<ExternalMmapRegion>,
}

impl ExpertCacheKey {
    #[allow(clippy::too_many_arguments)]
    fn new(
        layer_id: u32,
        expert: usize,
        bits: usize,
        block_size: usize,
        fc1: &QuantizedExperts<'_>,
        fc2: &QuantizedExperts<'_>,
        fc3: Option<&QuantizedExperts<'_>>,
    ) -> Self {
        let mut out_and_in_features = vec![
            (fc1.out_features, fc1.in_features),
            (fc2.out_features, fc2.in_features),
        ];
        let mut source_regions = Vec::with_capacity(
            fc1.mapped_regions.len()
                + fc2.mapped_regions.len()
                + fc3.map_or(0, |weights| weights.mapped_regions.len()),
        );
        source_regions.extend_from_slice(&fc1.mapped_regions);
        source_regions.extend_from_slice(&fc2.mapped_regions);
        if let Some(weights) = fc3 {
            out_and_in_features.push((weights.out_features, weights.in_features));
            source_regions.extend_from_slice(&weights.mapped_regions);
        }
        Self {
            layer_id,
            expert,
            bits,
            block_size,
            out_and_in_features,
            source_regions,
        }
    }
}

/// Policy lookup and payload publication are separate from dense allocation.
enum ExpertLease {
    Hit(Arc<DequantizedExpert>),
    Filled(Arc<DequantizedExpert>),
    Transient,
    Refused(onnx_runtime_memory_api::MemoryError),
}

/// The resolved execution choice travels through the prefetch worker too.
enum ExpertExecution {
    Dense(Arc<DequantizedExpert>),
    RowStreamed(onnx_runtime_memory_api::MemoryError),
}

/// A streaming-only two-slot allowance is never attached to retained state.
/// Cacheable prefetch instead splits one exact payload grant for each initially
/// missing selected expert; active hits are pinned through the operation.
enum ExpertMemoryScope<'a> {
    OnDemand(&'a DerivedWeightMemory),
    StreamingWindow(Arc<DenseWeightGrant>),
    Selected {
        grants: Mutex<BTreeMap<usize, DenseWeightGrant>>,
        _pins: Vec<Arc<DequantizedExpert>>,
    },
}

enum ExpertGrantAdmission {
    Granted(Option<Arc<DenseWeightGrant>>),
    Declined(onnx_runtime_memory_api::MemoryError),
}

fn owned_expert_grant(grant: DenseWeightGrant) -> Option<Arc<DenseWeightGrant>> {
    match grant {
        DenseWeightGrant::Local => None,
        grant => Some(Arc::new(grant)),
    }
}

impl ExpertMemoryScope<'_> {
    fn grant(&self, expert: usize, bytes: usize, retained: bool) -> Result<ExpertGrantAdmission> {
        match self {
            Self::OnDemand(memory) => match memory.try_grant(bytes)? {
                WeightAdmission::Granted(grant) => {
                    Ok(ExpertGrantAdmission::Granted(owned_expert_grant(grant)))
                }
                WeightAdmission::Declined(reason) => Ok(ExpertGrantAdmission::Declined(reason)),
            },
            Self::StreamingWindow(grant) => {
                if retained {
                    return Err(error(
                        "a retained QMoE expert cannot borrow a streaming-only two-slot grant; prepare individual selected-expert grants before admitting retained payloads",
                    ));
                }
                Ok(ExpertGrantAdmission::Granted(Some(Arc::clone(grant))))
            }
            Self::Selected { grants, .. } => {
                let grant = grants.lock().unwrap_or_else(std::sync::PoisonError::into_inner).remove(&expert)
                    .ok_or_else(|| error(format!("QMoE expert {expert} has no unused preflight payload grant; selected experts must be unique and initially resident entries must stay pinned")))?;
                Ok(ExpertGrantAdmission::Granted(owned_expert_grant(grant)))
            }
        }
    }

    fn load(
        &self,
        expert: usize,
        fc1: &QuantizedExperts<'_>,
        fc2: &QuantizedExperts<'_>,
        fc3: Option<&QuantizedExperts<'_>>,
    ) -> Result<ExpertExecution> {
        match self.grant(
            expert,
            DequantizedExpert::expanded_bytes(fc1, fc2, fc3)?,
            false,
        )? {
            ExpertGrantAdmission::Granted(grant) => Ok(ExpertExecution::Dense(Arc::new(
                DequantizedExpert::load(expert, fc1, fc2, fc3, true, grant)?,
            ))),
            ExpertGrantAdmission::Declined(reason) => Ok(ExpertExecution::RowStreamed(reason)),
        }
    }
}

fn load_transient_expert(
    expert: usize,
    fc1: &QuantizedExperts<'_>,
    fc2: &QuantizedExperts<'_>,
    fc3: Option<&QuantizedExperts<'_>>,
    memory: &DerivedWeightMemory,
    track_residency: bool,
) -> Result<ExpertExecution> {
    match memory.try_grant(DequantizedExpert::expanded_bytes(fc1, fc2, fc3)?)? {
        WeightAdmission::Granted(grant) => {
            Ok(ExpertExecution::Dense(Arc::new(DequantizedExpert::load(
                expert,
                fc1,
                fc2,
                fc3,
                track_residency,
                match grant {
                    DenseWeightGrant::Local => None,
                    grant => Some(Arc::new(grant)),
                },
            )?)))
        }
        WeightAdmission::Declined(reason) => Ok(ExpertExecution::RowStreamed(reason)),
    }
}

/// Identity is derived from the canonical expert key, with expert zero as the
/// bank representative; the cardinality completes its capacity bound.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ExpertBankKey {
    representative: ExpertCacheKey,
    experts: usize,
}

struct ExpertBank {
    key: ExpertBankKey,
    expanded_bytes: usize,
}

type ExpertBanks = BTreeMap<ExpertBankKey, Weak<ExpertBank>>;

struct HostCacheEntry {
    // Retained payloads keep their bank in the shared capacity proof even after
    // the operation (or compiled kernel) that created them has finished.
    _bank: Arc<ExpertBank>,
    weights: Arc<DequantizedExpert>,
    expanded_bytes: usize,
    frequency: u64,
    last_used: u64,
    pin_until: u64,
}

#[derive(Default)]
struct AdmissionHistory {
    frequency: u64,
    last_used: u64,
}

#[derive(Default)]
struct HostExpertCache {
    entries: BTreeMap<ExpertCacheKey, HostCacheEntry>,
    history: BTreeMap<ExpertCacheKey, AdmissionHistory>,
    owned_bytes: usize,
    reported_owned_bytes: u64,
    reported_budget_bytes: u64,
    clock: u64,
}

const ADMISSION_FREQUENCY: u64 = 2;
const PIN_FREQUENCY: u64 = 3;
const PIN_WINDOW: u64 = 8;
const HISTORY_DECAY_WINDOW: u64 = 64;
const HISTORY_EXPIRY_WINDOW: u64 = HISTORY_DECAY_WINDOW * 256;
const MAX_ADMISSION_HISTORY_ENTRIES: usize = 4096;

struct HostCacheFill {
    key: ExpertCacheKey,
    bank: Arc<ExpertBank>,
    expanded_bytes: usize,
    frequency: u64,
    last_used: u64,
}

enum HostCachePlan {
    Ready(ExpertLease),
    Fill(HostCacheFill),
}

impl HostExpertCache {
    fn hit(
        &mut self,
        key: &ExpertCacheKey,
        budget_bytes: usize,
    ) -> Result<Option<Arc<DequantizedExpert>>> {
        let Some(entry) = self.entries.get_mut(key) else {
            return Ok(None);
        };
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or_else(|| error("host-cache access clock overflow"))?;
        let now = self.clock;
        entry.frequency = entry.frequency.saturating_add(1);
        entry.last_used = now;
        if entry.frequency >= PIN_FREQUENCY {
            entry.pin_until = now.saturating_add(PIN_WINDOW);
        }
        let weights = Arc::clone(&entry.weights);
        metrics().record_host_cache_hit();
        self.record_residency(budget_bytes)?;
        Ok(Some(weights))
    }

    /// Select policy under the payload lock; allocation and retired-payload
    /// destruction happen only after the caller has released that lock.
    fn plan_lease(
        &mut self,
        key: ExpertCacheKey,
        expanded_bytes: usize,
        budget_bytes: usize,
        bank: Arc<ExpertBank>,
    ) -> (Result<HostCachePlan>, Vec<HostCacheEntry>) {
        let mut retired = Vec::new();
        let result = (|| -> Result<HostCachePlan> {
            if let Some(weights) = self.hit(&key, budget_bytes)? {
                return Ok(HostCachePlan::Ready(ExpertLease::Hit(weights)));
            }
            self.clock = self
                .clock
                .checked_add(1)
                .ok_or_else(|| error("host-cache access clock overflow"))?;
            let now = self.clock;
            metrics().record_host_cache_miss();
            if budget_bytes == 0 || expanded_bytes == 0 || expanded_bytes > budget_bytes {
                self.record_residency(budget_bytes)?;
                return Ok(HostCachePlan::Ready(ExpertLease::Transient));
            }
            self.prune_history(now);
            let history = self.history.entry(key.clone()).or_default();
            if now.saturating_sub(history.last_used) > HISTORY_DECAY_WINDOW {
                history.frequency /= 2;
            }
            history.frequency = history.frequency.saturating_add(1);
            history.last_used = now;
            let candidate_frequency = history.frequency;
            if candidate_frequency < ADMISSION_FREQUENCY {
                self.record_residency(budget_bytes)?;
                return Ok(HostCachePlan::Ready(ExpertLease::Transient));
            }
            let required = self
                .owned_bytes
                .checked_add(expanded_bytes)
                .ok_or_else(|| error("owned host-cache byte count overflow"))?;
            let mut victims = Vec::new();
            if required > budget_bytes {
                let mut candidates = self
                    .entries
                    .iter()
                    .filter(|(_, entry)| {
                        Arc::strong_count(&entry.weights) == 1
                            && entry.pin_until < now
                            && candidate_frequency > entry.frequency
                    })
                    .map(|(key, entry)| {
                        (
                            entry.frequency,
                            entry.last_used,
                            key.clone(),
                            entry.expanded_bytes,
                        )
                    })
                    .collect::<Vec<_>>();
                candidates.sort_by_key(|(frequency, last_used, _, _)| (*frequency, *last_used));
                let mut reclaim = required - budget_bytes;
                for (_, _, victim, bytes) in candidates {
                    victims.push(victim);
                    reclaim = reclaim.saturating_sub(bytes);
                    if reclaim == 0 {
                        break;
                    }
                }
                if reclaim != 0 {
                    self.record_residency(budget_bytes)?;
                    return Ok(HostCachePlan::Ready(ExpertLease::Transient));
                }
            }
            for victim in &victims {
                if let Some(entry) = self.entries.remove(victim) {
                    let bytes = entry.expanded_bytes;
                    retired.push(entry);
                    self.owned_bytes = self.owned_bytes.checked_sub(bytes).ok_or_else(|| error(format!(
                        "host expert cache layer {} expert {} accounting underflow: cannot retire {bytes} bytes from {} owned bytes; recreate the CPU provider and report inconsistent cache accounting",
                        victim.layer_id, victim.expert, self.owned_bytes,
                    )))?;
                }
            }
            metrics()
                .record_host_cache_evictions(victims.len())
                .map_err(error)?;
            self.record_residency(budget_bytes)?;
            Ok(HostCachePlan::Fill(HostCacheFill {
                key,
                bank,
                expanded_bytes,
                frequency: candidate_frequency,
                last_used: now,
            }))
        })();
        (result, retired)
    }

    fn publish(
        &mut self,
        fill: HostCacheFill,
        weights: Arc<DequantizedExpert>,
        budget_bytes: usize,
    ) -> Result<ExpertLease> {
        let total = self
            .owned_bytes
            .checked_add(fill.expanded_bytes)
            .ok_or_else(|| error("owned host-cache byte count overflow"))?;
        if total > budget_bytes || self.entries.contains_key(&fill.key) {
            return Err(error(
                "host-cache fill publication lost its exclusive cold-fill policy slot; preserve the fill gate through allocation and publication",
            ));
        }
        self.history.remove(&fill.key);
        self.entries.insert(
            fill.key,
            HostCacheEntry {
                _bank: fill.bank,
                weights: Arc::clone(&weights),
                expanded_bytes: fill.expanded_bytes,
                frequency: fill.frequency,
                last_used: fill.last_used,
                pin_until: 0,
            },
        );
        self.owned_bytes = total;
        self.record_residency(budget_bytes)?;
        Ok(ExpertLease::Filled(weights))
    }

    #[cfg(test)]
    fn lease_registered<F>(
        &mut self,
        key: ExpertCacheKey,
        expanded_bytes: usize,
        budget_bytes: usize,
        load: F,
        bank: Arc<ExpertBank>,
    ) -> Result<ExpertLease>
    where
        F: FnOnce() -> Result<DequantizedExpert>,
    {
        let trimmed = self.trim_to_budget(budget_bytes)?;
        drop(trimmed);
        let (plan, retired) = self.plan_lease(key, expanded_bytes, budget_bytes, bank);
        drop(retired);
        match plan? {
            HostCachePlan::Ready(lease) => Ok(lease),
            HostCachePlan::Fill(fill) => self.publish(fill, Arc::new(load()?), budget_bytes),
        }
    }

    #[cfg(test)]
    fn lease<F>(
        &mut self,
        key: ExpertCacheKey,
        expanded_bytes: usize,
        budget_bytes: usize,
        load: F,
    ) -> Result<ExpertLease>
    where
        F: FnOnce() -> Result<DequantizedExpert>,
    {
        let mut representative = key.clone();
        representative.expert = 0;
        let bank = Arc::new(ExpertBank {
            key: ExpertBankKey {
                representative,
                experts: usize::MAX,
            },
            expanded_bytes,
        });
        self.lease_registered(key, expanded_bytes, budget_bytes, load, bank)
    }

    fn prune_history(&mut self, now: u64) {
        self.history
            .retain(|_, history| now.saturating_sub(history.last_used) <= HISTORY_EXPIRY_WINDOW);
        while self.history.len() >= MAX_ADMISSION_HISTORY_ENTRIES {
            let Some(victim) = self
                .history
                .iter()
                .min_by_key(|(_, history)| (history.frequency, history.last_used))
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            self.history.remove(&victim);
        }
    }

    #[cfg(test)]
    fn trim_to_budget(&mut self, budget_bytes: usize) -> Result<Vec<HostCacheEntry>> {
        if self.owned_bytes <= budget_bytes {
            self.record_residency(budget_bytes)?;
            return Ok(Vec::new());
        }
        let mut candidates = self
            .entries
            .iter()
            .filter(|(_, entry)| Arc::strong_count(&entry.weights) == 1)
            .map(|(key, entry)| {
                (
                    entry.last_used,
                    entry.frequency,
                    key.clone(),
                    entry.expanded_bytes,
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(last_used, frequency, _, _)| (*last_used, *frequency));
        let mut victims = Vec::new();
        let mut projected = self.owned_bytes;
        for (_, _, key, bytes) in candidates {
            victims.push(key);
            projected = projected
                .checked_sub(bytes)
                .ok_or_else(|| error("host-cache trim byte accounting underflow"))?;
            if projected <= budget_bytes {
                break;
            }
        }
        if projected > budget_bytes {
            return Err(error(format!(
                "cannot lower owned host-cache budget to {budget_bytes} bytes while {projected} bytes are leased"
            )));
        }
        let retired = victims
            .iter()
            .filter_map(|victim| self.entries.remove(victim))
            .collect();
        self.owned_bytes = projected;
        metrics()
            .record_host_cache_evictions(victims.len())
            .map_err(error)?;
        self.record_residency(budget_bytes)?;
        Ok(retired)
    }

    fn record_residency(&mut self, budget_bytes: usize) -> Result<()> {
        let (owned, budget) = metrics()
            .record_host_cache_residency(
                self.reported_owned_bytes,
                self.owned_bytes,
                self.reported_budget_bytes,
                budget_bytes,
            )
            .map_err(error)?;
        self.reported_owned_bytes = owned;
        self.reported_budget_bytes = budget;
        Ok(())
    }

    fn release_residency(&mut self) {
        metrics()
            .release_host_cache_residency(self.reported_owned_bytes, self.reported_budget_bytes);
        self.reported_owned_bytes = 0;
        self.reported_budget_bytes = 0;
    }
}

impl Drop for HostExpertCache {
    fn drop(&mut self) {
        self.release_residency();
    }
}

struct WeightOffloadHostCacheInner {
    cache: Mutex<HostExpertCache>,
    retention_bytes: u64,
    memory: DerivedWeightMemory,
    effective_retention: OnceLock<usize>,
    banks: RwLock<ExpertBanks>,
    // Serialize cold publication, never holding the payload lock across a
    // governor callback, allocation, dequantization, or retired-payload drop.
    fill: Mutex<()>,
}

impl crate::resources::IdleWeightCache for WeightOffloadHostCacheInner {
    fn reclaim_idle(&self, want: usize) -> Result<usize> {
        let mut retired = Vec::new();
        let result = {
            let mut cache = self
                .cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (|| {
                let mut released = 0usize;
                // Frequency pins guide ordinary retention, not physical pressure.
                // Active Arc readers always remain protected.
                while released < want {
                    let candidate = cache
                        .entries
                        .iter()
                        .filter(|(_, entry)| Arc::strong_count(&entry.weights) == 1)
                        .min_by_key(|(_, entry)| (entry.frequency, entry.last_used))
                        .map(|(key, _)| key.clone());
                    let Some(key) = candidate else {
                        break;
                    };
                    let entry = cache
                        .entries
                        .remove(&key)
                        .expect("selected idle expert exists");
                    let bytes = entry.expanded_bytes;
                    retired.push(entry);
                    cache.owned_bytes = cache.owned_bytes.checked_sub(bytes)
                        .ok_or_else(|| error("idle expert reclamation exceeded owned cache bytes; recreate the CPU provider and report inconsistent cache accounting"))?;
                    released += bytes;
                }
                metrics()
                    .record_host_cache_evictions(retired.len())
                    .map_err(error)?;
                let budget = usize::try_from(cache.reported_budget_bytes)
                    .expect("reported cache budget originated from an addressable usize");
                cache.record_residency(budget)?;
                Ok(released)
            })()
        };
        drop(retired);
        result
    }
}

/// Shared expert retention policy and payloads funded by their own Host grants.
#[derive(Clone)]
pub(crate) struct WeightOffloadHostCache {
    inner: Arc<WeightOffloadHostCacheInner>,
}

#[derive(Debug, PartialEq, Eq)]
enum HostPrefetchEligibility {
    Streaming,
    FullyResident,
    InsufficientCapacity {
        required_bytes: u128,
        budget_bytes: usize,
    },
    CapacityOverflow,
}

impl HostPrefetchEligibility {
    fn allows_pipeline(&self) -> bool {
        matches!(self, Self::Streaming | Self::FullyResident)
    }
}

enum HostBankSnapshot<'a> {
    // A frozen zero budget cannot admit anything, so no registry lock is needed.
    Streaming,
    Registered(RwLockReadGuard<'a, ExpertBanks>),
}

/// Borrow the immutable retention decision and freeze bank membership through
/// the routed operation. Workers borrow this proof rather than taking another
/// reader behind a pending cold registrar.
struct HostCacheOperation<'a> {
    banks: HostBankSnapshot<'a>,
    bank: Arc<ExpertBank>,
    cache: &'a WeightOffloadHostCacheInner,
    budget_bytes: usize,
}

impl HostCacheOperation<'_> {
    fn prefetch_eligibility(&self) -> HostPrefetchEligibility {
        let HostBankSnapshot::Registered(banks) = &self.banks else {
            return HostPrefetchEligibility::Streaming;
        };
        let Some(required_bytes) =
            banks
                .values()
                .filter_map(Weak::upgrade)
                .try_fold(0u128, |sum, bank| {
                    sum.checked_add(
                        (bank.key.experts as u128).checked_mul(bank.expanded_bytes as u128)?,
                    )
                })
        else {
            return HostPrefetchEligibility::CapacityOverflow;
        };
        if required_bytes <= self.budget_bytes as u128 {
            HostPrefetchEligibility::FullyResident
        } else {
            HostPrefetchEligibility::InsufficientCapacity {
                required_bytes,
                budget_bytes: self.budget_bytes,
            }
        }
    }

    fn prefetch_memory(
        &self,
        experts: impl Iterator<Item = usize>,
    ) -> Result<Option<ExpertMemoryScope<'_>>> {
        if !self.cache.memory.is_governed() {
            return Ok(Some(ExpertMemoryScope::OnDemand(&self.cache.memory)));
        }
        if self.budget_bytes == 0 {
            let Some(bytes) = self.bank.expanded_bytes.checked_mul(2) else {
                return Ok(None);
            };
            return match self.cache.memory.try_grant(bytes)? {
                WeightAdmission::Granted(grant) => {
                    Ok(Some(ExpertMemoryScope::StreamingWindow(Arc::new(grant))))
                }
                WeightAdmission::Declined(reason) => {
                    onnx_runtime_ep_api::record_kernel_variant_stage!(
                        "prefetch",
                        "qmoe.serial",
                        "two streaming expert slots require {bytes} Host bytes: {reason}"
                    );
                    Ok(None)
                }
            };
        }
        let (pins, missing) = {
            let cache = self
                .cache
                .cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut pins = Vec::new();
            let mut missing = Vec::new();
            let mut key = self.bank.key.representative.clone();
            for expert in experts {
                key.expert = expert;
                if let Some(entry) = cache.entries.get(&key) {
                    pins.push(Arc::clone(&entry.weights));
                } else {
                    missing.push(expert);
                }
            }
            (pins, missing)
        };
        let mut grants = BTreeMap::new();
        if !missing.is_empty() {
            let Some(bytes) = self.bank.expanded_bytes.checked_mul(missing.len()) else {
                return Ok(None);
            };
            let mut group = match self.cache.memory.try_grant(bytes)? {
                WeightAdmission::Granted(grant) => grant,
                WeightAdmission::Declined(reason) => {
                    onnx_runtime_ep_api::record_kernel_variant_stage!(
                        "prefetch",
                        "qmoe.serial",
                        "complete missing selected experts require {bytes} Host bytes: {reason}"
                    );
                    return Ok(None);
                }
            };
            for expert in missing {
                grants.insert(expert, group.split_off(self.bank.expanded_bytes)?);
            }
        }
        Ok(Some(ExpertMemoryScope::Selected {
            grants: Mutex::new(grants),
            _pins: pins,
        }))
    }

    #[cfg(test)]
    fn registered_bank_count(&self) -> usize {
        match &self.banks {
            HostBankSnapshot::Streaming => 0,
            HostBankSnapshot::Registered(banks) => banks.len(),
        }
    }

    fn lease<F>(
        &self,
        expert: usize,
        memory: &ExpertMemoryScope<'_>,
        load: F,
    ) -> Result<ExpertLease>
    where
        F: FnOnce(Option<Arc<DenseWeightGrant>>) -> Result<DequantizedExpert>,
    {
        if self.budget_bytes == 0 {
            metrics().record_host_cache_miss();
            return Ok(ExpertLease::Transient);
        }
        let mut key = self.bank.key.representative.clone();
        key.expert = expert;
        if let Some(weights) = self
            .cache
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .hit(&key, self.budget_bytes)?
        {
            return Ok(ExpertLease::Hit(weights));
        }
        let _fill = self
            .cache
            .fill
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (plan, retired) = self
            .cache
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .plan_lease(
                key,
                self.bank.expanded_bytes,
                self.budget_bytes,
                Arc::clone(&self.bank),
            );
        drop(retired);
        match plan? {
            HostCachePlan::Ready(lease) => Ok(lease),
            HostCachePlan::Fill(fill) => match memory.grant(expert, fill.expanded_bytes, true)? {
                ExpertGrantAdmission::Granted(grant) => {
                    let loaded = Arc::new(load(grant)?);
                    let result = self
                        .cache
                        .cache
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .publish(fill, Arc::clone(&loaded), self.budget_bytes);
                    // Keep this owner outside the guard, including publication errors.
                    drop(loaded);
                    result
                }
                ExpertGrantAdmission::Declined(reason) => Ok(ExpertLease::Refused(reason)),
            },
        }
    }
}

impl WeightOffloadHostCache {
    pub(crate) fn new(
        retention_bytes: u64,
        memory: DerivedWeightMemory,
    ) -> std::result::Result<Self, &'static str> {
        checked_host_budget(retention_bytes)?;
        let inner = Arc::new(WeightOffloadHostCacheInner {
            cache: Mutex::new(HostExpertCache::default()),
            retention_bytes,
            memory,
            effective_retention: OnceLock::new(),
            banks: RwLock::new(BTreeMap::new()),
            fill: Mutex::new(()),
        });
        inner.memory.register_idle_cache(&inner);
        Ok(Self { inner })
    }

    /// Configured retention ceiling, never an idle reservation on Host capacity.
    pub(crate) fn retention_budget_bytes(&self) -> u64 {
        self.inner.retention_bytes
    }

    pub(crate) fn memory(&self) -> &DerivedWeightMemory {
        &self.inner.memory
    }

    fn begin_operation(
        &self,
        mut representative: ExpertCacheKey,
        experts: usize,
        expanded_bytes: usize,
    ) -> Result<HostCacheOperation<'_>> {
        representative.expert = 0;
        let key = ExpertBankKey {
            representative,
            experts,
        };
        let budget_bytes = if let Some(&bytes) = self.inner.effective_retention.get() {
            bytes
        } else {
            let requested = effective_host_budget_u64(self.inner.retention_bytes)?;
            let candidate = checked_host_budget(requested).map_err(|reason| {
                EpError::Memory(onnx_runtime_memory_api::MemoryError::InvalidRequest {
                    tier: onnx_runtime_memory_api::Tier::Host.name(),
                    requested,
                    reason,
                })
            })?;
            // Resolve once on success. Invalid configuration leaves no policy
            // or payload state behind, so correcting it can be retried.
            *self.inner.effective_retention.get_or_init(|| candidate)
        };
        if budget_bytes == 0 {
            return Ok(HostCacheOperation {
                banks: HostBankSnapshot::Streaming,
                bank: Arc::new(ExpertBank {
                    key,
                    expanded_bytes,
                }),
                cache: &self.inner,
                budget_bytes,
            });
        }
        // The immutable policy needs no adoption/lifecycle locks. Registration
        // is cold; readers for known banks share the lock and compute concurrently.
        let mut banks = self
            .inner
            .banks
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let bank = if let Some(bank) = banks.get(&key).and_then(Weak::upgrade) {
            bank
        } else {
            drop(banks);
            let mut registry = self
                .inner
                .banks
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            registry.retain(|_, bank| bank.strong_count() != 0);
            let bank = registry
                .get(&key)
                .and_then(Weak::upgrade)
                .unwrap_or_else(|| {
                    let bank = Arc::new(ExpertBank {
                        key: key.clone(),
                        expanded_bytes,
                    });
                    registry.insert(key, Arc::downgrade(&bank));
                    bank
                });
            drop(registry);
            // Our strong reference keeps this registration live across the
            // write-to-read transition; another registrar cannot prune it.
            banks = self
                .inner
                .banks
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            bank
        };
        if bank.expanded_bytes != expanded_bytes {
            return Err(error(format!(
                "host expert bank layer {} ({} experts, {}-bit, block {}) changed expanded bytes from {} to {expanded_bytes} for the same source identity; use a distinct source layout",
                bank.key.representative.layer_id,
                bank.key.experts,
                bank.key.representative.bits,
                bank.key.representative.block_size,
                bank.expanded_bytes,
            )));
        }
        Ok(HostCacheOperation {
            banks: HostBankSnapshot::Registered(banks),
            bank,
            cache: &self.inner,
            budget_bytes,
        })
    }

    #[cfg(test)]
    fn lease<F>(&self, key: ExpertCacheKey, expanded_bytes: usize, load: F) -> Result<ExpertLease>
    where
        F: FnOnce() -> Result<DequantizedExpert>,
    {
        let expert = key.expert;
        self.begin_operation(key, usize::MAX, expanded_bytes)?
            .lease(
                expert,
                &ExpertMemoryScope::OnDemand(self.memory()),
                |grant| {
                    let mut weights = load()?;
                    weights._grant = grant;
                    Ok(weights)
                },
            )
    }

    #[cfg(test)]
    fn clear(&self) {
        let retired = {
            let mut cache = self
                .inner
                .cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            cache.release_residency();
            cache.history.clear();
            cache.owned_bytes = 0;
            cache.clock = 0;
            std::mem::take(&mut cache.entries)
        };
        drop(retired);
    }
}

fn lease_route_first_expert(
    host_cache: &HostCacheOperation<'_>,
    expert: usize,
    fc1: &QuantizedExperts<'_>,
    fc2: &QuantizedExperts<'_>,
    fc3: Option<&QuantizedExperts<'_>>,
    memory: &ExpertMemoryScope<'_>,
) -> Result<ExpertExecution> {
    let lease = host_cache.lease(expert, memory, |grant| {
        DequantizedExpert::load(expert, fc1, fc2, fc3, true, grant)
    })?;
    let (execution, read_from_mmap) = match lease {
        ExpertLease::Hit(weights) => (ExpertExecution::Dense(weights), false),
        ExpertLease::Filled(weights) => (ExpertExecution::Dense(weights), true),
        ExpertLease::Transient => (memory.load(expert, fc1, fc2, fc3)?, true),
        ExpertLease::Refused(reason) => (ExpertExecution::RowStreamed(reason), true),
    };
    if read_from_mmap {
        let mut bytes_read = fc1
            .expert_source_bytes(expert)?
            .checked_add(fc2.expert_source_bytes(expert)?)
            .ok_or_else(|| error("selected expert byte count overflow"))?;
        if let Some(source) = fc3 {
            bytes_read = bytes_read
                .checked_add(source.expert_source_bytes(expert)?)
                .ok_or_else(|| error("selected expert byte count overflow"))?;
        }
        metrics().record_bytes_read(bytes_read).map_err(error)?;
    }
    Ok(execution)
}

#[allow(clippy::too_many_arguments)]
fn write_grouped_contributions(
    contributions: &mut [f32],
    input: &[f32],
    tasks: &[(usize, usize, f32)],
    expert: usize,
    weights: &ExpertExecution,
    fc1: &QuantizedExperts<'_>,
    fc2: &QuantizedExperts<'_>,
    fc3: Option<&QuantizedExperts<'_>>,
    fc1_bias: Option<&[f32]>,
    fc2_bias: Option<&[f32]>,
    fc3_bias: Option<&[f32]>,
    fc1_size: usize,
    hidden: usize,
    inter: usize,
    attributes: &MoeAttributes,
) -> Result<()> {
    let fc1_bias_range = checked_range(expert, fc1_size, "fc1 bias expert row")?;
    let fc2_bias_range = checked_range(expert, hidden, "fc2 bias expert row")?;
    let fc3_bias_range = checked_range(expert, inter, "fc3 bias expert row")?;
    let mut grouped_input = Vec::with_capacity(tasks.len() * hidden);
    for &(row, _, _) in tasks {
        grouped_input.extend_from_slice(&input[checked_range(row, hidden, "input row")?]);
    }
    let expert_out = match weights {
        ExpertExecution::Dense(weights) => run_expert_grouped(
            &grouped_input,
            tasks.len(),
            &weights.fc1,
            fc1_bias.map(|bias| &bias[fc1_bias_range]),
            &weights.fc2,
            fc2_bias.map(|bias| &bias[fc2_bias_range]),
            weights.fc3.as_deref(),
            fc3_bias.map(|bias| &bias[fc3_bias_range]),
            fc1_size,
            hidden,
            inter,
            attributes,
        )?,
        ExpertExecution::RowStreamed(reason) => {
            onnx_runtime_ep_api::record_kernel_variant!(
                "qmoe.streamed_rows",
                "Host transient expert grant declined: {reason}"
            );
            let fc1 = QuantizedExpertProjection {
                source: fc1,
                expert,
            };
            let fc2 = QuantizedExpertProjection {
                source: fc2,
                expert,
            };
            let fc3 = fc3.map(|source| QuantizedExpertProjection { source, expert });
            run_expert_projections(
                &grouped_input,
                tasks.len(),
                &fc1,
                fc1_bias.map(|bias| &bias[fc1_bias_range]),
                &fc2,
                fc2_bias.map(|bias| &bias[fc2_bias_range]),
                fc3.as_ref(),
                fc3_bias.map(|bias| &bias[fc3_bias_range]),
                attributes,
            )?
        }
    };
    for (grouped_row, &(_, slot, route_weight)) in tasks.iter().enumerate() {
        let contribution =
            &mut contributions[checked_range(slot, hidden, "routed contribution row")?];
        for feature in 0..hidden {
            contribution[feature] = route_weight * expert_out[grouped_row * hidden + feature];
        }
    }
    Ok(())
}

struct QuantizedExperts<'a> {
    packed: Cow<'a, [u8]>,
    scales: Cow<'a, [f32]>,
    zero_points: Option<Cow<'a, [u8]>>,
    catalogs: Vec<WeightRegionCatalog>,
    mapped_regions: Vec<ExternalMmapRegion>,
    pageable: bool,
    experts: usize,
    out_features: usize,
    in_features: usize,
    packed_in: usize,
    blocks: usize,
    zero_point_bytes: usize,
    dequantized_elements: usize,
    bits: usize,
    block_size: usize,
}

impl<'a> QuantizedExperts<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: &str,
        packed: &'a TensorView<'a>,
        scales: &'a TensorView<'a>,
        zero_points: Option<&'a TensorView<'a>>,
        experts: usize,
        out_features: usize,
        in_features: usize,
        bits: usize,
        block_size: usize,
        prefer_mmap: bool,
    ) -> Result<Self> {
        for (operand, view) in [
            ("experts_weights", Some(packed)),
            ("scales", Some(scales)),
            ("zero_points", zero_points),
        ] {
            if view.is_some_and(|view| !view.device.is_host_accessible()) {
                return Err(error(format!(
                    "{name}_{operand} must be host-accessible for CPU QMoE; move this tensor to the CPU before execution"
                )));
            }
        }
        let pack_size = 8 / bits;
        if !in_features.is_multiple_of(pack_size) {
            return Err(error(format!(
                "{name} input features {in_features} must be divisible by pack_size {pack_size}"
            )));
        }
        if !in_features.is_multiple_of(block_size) {
            return Err(error(format!(
                "{name} input features {in_features} must be divisible by block_size {block_size}"
            )));
        }
        let packed_in = in_features / pack_size;
        let expert_rows = checked_product(
            &[experts, out_features],
            &format!("{name} expert-row count"),
        )?;
        let packed_elements = checked_product(
            &[expert_rows, packed_in],
            &format!("{name} packed-weight element count"),
        )?;
        checked_byte_count(
            packed_elements,
            std::mem::size_of::<u8>(),
            &format!("{name} packed-weight byte count"),
        )?;
        require_exact_shape(
            &format!("{name}_experts_weights"),
            packed.shape,
            &[experts, out_features, packed_in],
        )?;
        let blocks = in_features / block_size;
        let scale_elements = checked_product(
            &[expert_rows, blocks],
            &format!("{name} scale element count"),
        )?;
        let scale_bytes = checked_byte_count(
            scale_elements,
            std::mem::size_of::<f32>(),
            &format!("{name} scale byte count"),
        )?;
        require_exact_shape(
            &format!("{name}_scales"),
            scales.shape,
            &[experts, out_features, blocks],
        )?;
        let zero_point_bytes = blocks
            .checked_add(pack_size - 1)
            .ok_or_else(|| error(format!("{name} zero-point row byte count overflow")))?
            / pack_size;
        let zero_point_elements = checked_product(
            &[expert_rows, zero_point_bytes],
            &format!("{name} zero-point element count"),
        )?;
        if let Some(points) = zero_points {
            checked_byte_count(
                zero_point_elements,
                std::mem::size_of::<u8>(),
                &format!("{name} zero-point byte count"),
            )?;
            require_exact_shape(
                &format!("{name}_zero_points"),
                points.shape,
                &[experts, out_features, zero_point_bytes],
            )?;
        }
        preflight_row_ranges(expert_rows, packed_in, &format!("{name} packed-weight"))?;
        preflight_row_ranges(expert_rows, blocks, &format!("{name} scale"))?;
        preflight_row_ranges(expert_rows, zero_point_bytes, &format!("{name} zero-point"))?;
        let dequantized_elements = checked_product(
            &[out_features, in_features],
            &format!("{name} per-expert dequantized element count"),
        )?;
        checked_byte_count(
            dequantized_elements,
            std::mem::size_of::<f32>(),
            &format!("{name} per-expert dequantized byte count"),
        )?;
        preflight_row_ranges(
            out_features,
            in_features,
            &format!("{name} dequantized output"),
        )?;
        let quantization = Some(ExpertQuantization {
            bits,
            block_size,
            blocks_per_row: blocks,
        });
        let catalog_for = |view: &TensorView, rows_per_expert, elements_per_row, tensor_len| {
            WeightRegionCatalog::for_mapped_tensor_view(
                view.dtype,
                view.shape,
                tensor_len,
                ExpertTensorLayout {
                    version: 1,
                    experts,
                    rows_per_expert,
                    storage_elements_per_row: elements_per_row,
                    order: if view.is_contiguous() {
                        ExpertStorageOrder::ExpertMajor
                    } else {
                        ExpertStorageOrder::Interleaved
                    },
                    quantization,
                },
            )
        };
        let packed_catalog = catalog_for(packed, out_features, packed_in, packed_elements);
        let scale_catalog = catalog_for(scales, out_features, blocks, scale_bytes);
        let zero_point_catalog = zero_points
            .map(|points| catalog_for(points, out_features, zero_point_bytes, zero_point_elements));
        let mapped_region = |view: &TensorView| match view.backing {
            TensorBacking::ExternalMmap(region) => Some(region),
            TensorBacking::Opaque | TensorBacking::Sealed { .. } => None,
        };
        let pageable = prefer_mmap
            && packed_catalog.is_pageable()
            && scale_catalog.is_pageable()
            && zero_point_catalog
                .as_ref()
                .is_none_or(WeightRegionCatalog::is_pageable)
            && mapped_region(packed).is_some()
            && mapped_region(scales).is_some()
            && zero_points.is_none_or(|points| mapped_region(points).is_some());

        // Borrowability is a layout/device property, independent of whether
        // the complete bank qualifies for mmap paging. Materialize each strided
        // operand separately instead of copying every tensor in the bank.
        let packed_data = quantized_input_bytes(packed)?;
        let scale_data = crate::dtype::to_dense_f32_widen("QMoE scales", scales)?;
        let zero_point_data = zero_points.map(quantized_input_bytes).transpose()?;
        let mut catalogs = vec![packed_catalog, scale_catalog];
        catalogs.extend(zero_point_catalog);
        let mut mapped_regions = Vec::new();
        if pageable {
            mapped_regions.push(mapped_region(packed).expect("pageable packed mmap"));
            mapped_regions.push(mapped_region(scales).expect("pageable scale mmap"));
            if let Some(points) = zero_points {
                mapped_regions.push(mapped_region(points).expect("pageable zero-point mmap"));
            }
        }
        Ok(Self {
            packed: packed_data,
            scales: scale_data,
            zero_points: zero_point_data,
            catalogs,
            mapped_regions,
            pageable,
            experts,
            out_features,
            in_features,
            packed_in,
            blocks,
            zero_point_bytes,
            dequantized_elements,
            bits,
            block_size,
        })
    }

    fn is_pageable(&self) -> bool {
        self.pageable
    }

    fn mapped_regions(&self) -> &[ExternalMmapRegion] {
        &self.mapped_regions
    }

    fn expert_source_bytes(&self, expert: usize) -> Result<usize> {
        self.catalogs.iter().try_fold(0usize, |total, catalog| {
            let bytes = catalog
                .region(expert)
                .ok_or_else(|| error(format!("missing catalog range for expert {expert}")))?
                .len;
            total
                .checked_add(bytes)
                .ok_or_else(|| error("per-expert source byte count overflow"))
        })
    }

    fn dequantize_expert(&self, expert: usize) -> Result<Vec<f32>> {
        if expert >= self.experts {
            return Err(error(format!(
                "routed expert {expert} is out of range for {} experts",
                self.experts
            )));
        }
        let requested = self.dequantized_bytes()? as u64;
        let mut output = Vec::new();
        output
            .try_reserve_exact(self.dequantized_elements)
            .map_err(|failure| {
                EpError::Memory(onnx_runtime_memory_api::MemoryError::AllocationFailed { tier: "host", requested, reason: format!("QMoE expert {expert} could not allocate its admitted f32 projection: {failure}; free Host memory or lower concurrent expert demand") })
            })?;
        output.resize(self.dequantized_elements, 0.0f32);
        for row in 0..self.out_features {
            let output_range = checked_range(row, self.in_features, "dequantized output row")?;
            self.decode_expert_row(expert, row, &mut output[output_range])?;
        }
        Ok(output)
    }

    fn decode_expert_row(&self, expert: usize, row: usize, output: &mut [f32]) -> Result<()> {
        if expert >= self.experts || row >= self.out_features || output.len() != self.in_features {
            return Err(error(format!(
                "expert {expert}, output row {row}, decoded length {} do not match [{},{},{}]; use an in-range expert/row and a complete input-feature buffer",
                output.len(),
                self.experts,
                self.out_features,
                self.in_features
            )));
        }
        let expert_row = expert
            .checked_mul(self.out_features)
            .and_then(|offset| offset.checked_add(row))
            .ok_or_else(|| error("expert-row offset overflow"))?;
        let packed_range = checked_range(expert_row, self.packed_in, "packed-weight expert row")?;
        let scale_range = checked_range(expert_row, self.blocks, "scale expert row")?;
        let zero_point_range =
            checked_range(expert_row, self.zero_point_bytes, "zero-point expert row")?;
        dequantize_nbits_row(
            &self.packed[packed_range],
            &self.scales[scale_range],
            self.zero_points
                .as_ref()
                .map(|points| &points[zero_point_range]),
            output,
            self.bits,
            self.block_size,
        );
        Ok(())
    }

    fn dequantized_bytes(&self) -> Result<usize> {
        checked_byte_count(
            self.dequantized_elements,
            std::mem::size_of::<f32>(),
            "per-expert expanded byte count",
        )
    }
}

struct QuantizedExpertProjection<'a, 'b> {
    source: &'a QuantizedExperts<'b>,
    expert: usize,
}
impl ExpertProjection for QuantizedExpertProjection<'_, '_> {
    fn out_features(&self) -> usize {
        self.source.out_features
    }
    fn in_features(&self) -> usize {
        self.source.in_features
    }
    fn apply(&self, input: &[f32], rows: usize, bias: Option<&[f32]>) -> Result<Vec<f32>> {
        super::decoded_projection::linear_decoded_grouped(
            "QMoE NBits expert",
            input,
            rows,
            self.in_features(),
            self.out_features(),
            bias,
            || |row, output: &mut [f32]| self.source.decode_expert_row(self.expert, row, output),
            |input, row, output, rows, k| {
                super::matmul::gemm_bt_serial(input, row, output, rows, k, 1)
            },
            super::decoded_projection::rayon_decoded_columns,
        )
    }
}

fn checked_product(factors: &[usize], context: &str) -> Result<usize> {
    onnx_runtime_loader::weights::checked_product(factors, context)
        .map_err(|failure| error(failure.to_string()))
}

fn checked_byte_count(elements: usize, element_size: usize, context: &str) -> Result<usize> {
    onnx_runtime_loader::weights::checked_byte_count(elements, element_size, context)
        .map_err(|failure| error(failure.to_string()))
}

fn checked_tensor_layout(name: &str, shape: &[usize], dtype: DataType) -> Result<usize> {
    let elements = checked_product(shape, &format!("{name} element count"))?;
    checked_byte_count(elements, dtype.byte_size(), &format!("{name} byte count"))?;
    Ok(elements)
}

fn checked_range(index: usize, width: usize, context: &str) -> Result<std::ops::Range<usize>> {
    onnx_runtime_loader::weights::checked_range(index, width, context)
        .map_err(|failure| error(failure.to_string()))
}

fn preflight_row_ranges(rows: usize, width: usize, context: &str) -> Result<()> {
    if rows != 0 {
        checked_range(rows - 1, width, context)?;
    }
    Ok(())
}

fn optional_input<'a, 'b>(
    inputs: &'a [TensorView<'b>],
    index: usize,
) -> Option<&'a TensorView<'b>> {
    inputs.get(index).filter(|input| !input.is_absent())
}

fn optional_dense(inputs: &[TensorView], index: usize) -> Result<Option<Vec<f32>>> {
    optional_input(inputs, index).map(to_dense_f32).transpose()
}

fn validate_bias(
    name: &str,
    inputs: &[TensorView],
    index: usize,
    experts: usize,
    width: usize,
) -> Result<()> {
    if let Some(bias) = optional_input(inputs, index) {
        require_exact_shape(name, bias.shape, &[experts, width])?;
    }
    Ok(())
}

fn require_dtype(name: &str, input: &TensorView, dtype: DataType) -> Result<()> {
    if input.dtype != dtype {
        return Err(error(format!(
            "{name} requires {dtype:?}, got {:?}",
            input.dtype
        )));
    }
    Ok(())
}

fn require_rank(name: &str, shape: &[usize], rank: usize) -> Result<()> {
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

fn int_attr(node: &Node, name: &str, default: i64) -> Result<i64> {
    match node.attr(name) {
        Some(attr) => attr
            .as_int()
            .ok_or_else(|| error(format!("attribute {name} must be an integer"))),
        None => Ok(default),
    }
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("QMoE: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuExecutionProvider;
    use crate::kernels::testutil::Owned;
    use crate::weight_offload::{WeightOffloadMode, WeightOffloadStats, metrics};
    use onnx_runtime_ep_api::{DevicePtr, ExecutionProvider};
    use onnx_runtime_ir::{Attribute, DeviceId, Graph, NodeId, WeightRef, static_shape};
    use onnx_runtime_loader::{Model, Pageability, WeightStore, load_model_with_weights};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};

    struct Quantized {
        packed: Vec<u8>,
        scales: Vec<f32>,
        zero_points: Option<Vec<u8>>,
        dequantized: Vec<f32>,
    }

    fn quantize(
        experts: usize,
        out_features: usize,
        in_features: usize,
        bits: usize,
        block_size: usize,
        affine: bool,
    ) -> Quantized {
        let pack_size = 8 / bits;
        let blocks = in_features / block_size;
        let packed_in = in_features / pack_size;
        let zp_bytes = blocks.div_ceil(pack_size);
        let mask = if bits == 8 {
            u8::MAX
        } else {
            (1u8 << bits) - 1
        };
        let default_zp = 1u8 << (bits - 1);
        let mut packed = vec![0u8; experts * out_features * packed_in];
        let mut scales = vec![0.0f32; experts * out_features * blocks];
        let mut zero_points = affine.then(|| vec![0u8; experts * out_features * zp_bytes]);
        let mut dequantized = vec![0.0f32; experts * out_features * in_features];

        for expert in 0..experts {
            for row in 0..out_features {
                for block in 0..blocks {
                    let scale = 0.25 + 0.125 * ((expert + row + block) % 3) as f32;
                    scales[(expert * out_features + row) * blocks + block] = scale;
                    let zero_point = if affine {
                        default_zp.saturating_sub(((expert + row + block) % 2) as u8)
                    } else {
                        default_zp
                    };
                    if let Some(points) = &mut zero_points {
                        let index = (expert * out_features + row) * zp_bytes + block / pack_size;
                        points[index] |= zero_point << ((block % pack_size) * bits);
                    }
                    for offset in 0..block_size {
                        let depth = block * block_size + offset;
                        let centered = ((expert * 3 + row * 5 + depth * 7) % 7) as i16 - 3;
                        let quantized = (centered + zero_point as i16) as u8 & mask;
                        let packed_index =
                            (expert * out_features + row) * packed_in + depth / pack_size;
                        packed[packed_index] |= quantized << ((depth % pack_size) * bits);
                        dequantized[(expert * out_features + row) * in_features + depth] =
                            (quantized as f32 - zero_point as f32) * scale;
                    }
                }
            }
        }
        Quantized {
            packed,
            scales,
            zero_points,
            dequantized,
        }
    }

    #[test]
    fn quantized_inputs_borrow_each_contiguous_operand_independently_of_paging() {
        const EXPERTS: usize = 2;
        const OUTPUTS: usize = 3;
        const INPUTS: usize = 96;
        const BLOCK: usize = 16;
        for bits in [2, 4, 8] {
            for affine in [false, true] {
                let quantized = quantize(EXPERTS, OUTPUTS, INPUTS, bits, BLOCK, affine);
                for prefer_mmap in [false, true] {
                    for strided in 0usize..8 {
                        // Prefixes exercise byte_offset; independent stride masks
                        // ensure one strided operand does not force other copies.
                        let steps: [usize; 3] = std::array::from_fn(|i| 1 + ((strided >> i) & 1));
                        let padded_u8 = |values: &[u8], step: usize| {
                            let mut data = vec![0; 1 + values.len() * step];
                            for (i, value) in values.iter().enumerate() {
                                data[1 + i * step] = *value;
                            }
                            Owned::u8(&[data.len()], &data)
                        };
                        let packed = padded_u8(&quantized.packed, steps[0]);
                        let mut scale_data = vec![0.0; 1 + quantized.scales.len() * steps[1]];
                        for (i, value) in quantized.scales.iter().enumerate() {
                            scale_data[1 + i * steps[1]] = *value;
                        }
                        let scales = Owned::f32(&[scale_data.len()], &scale_data);
                        let points = quantized
                            .zero_points
                            .as_deref()
                            .map(|data| padded_u8(data, steps[2]));
                        let blocks = INPUTS / BLOCK;
                        let packed_in = INPUTS / (8 / bits);
                        let point_width = blocks.div_ceil(8 / bits);
                        let shapes = [
                            [EXPERTS, OUTPUTS, packed_in],
                            [EXPERTS, OUTPUTS, blocks],
                            [EXPERTS, OUTPUTS, point_width],
                        ];
                        let strides: [[i64; 3]; 3] = std::array::from_fn(|i| {
                            [
                                (OUTPUTS * shapes[i][2] * steps[i]) as i64,
                                (shapes[i][2] * steps[i]) as i64,
                                steps[i] as i64,
                            ]
                        });
                        let mut packed_view = packed.view();
                        packed_view.shape = &shapes[0];
                        packed_view.strides = &strides[0];
                        packed_view.byte_offset = 1;
                        let mut scale_view = scales.view();
                        scale_view.shape = &shapes[1];
                        scale_view.strides = &strides[1];
                        scale_view.byte_offset = 4;
                        let point_view = points.as_ref().map(|points| {
                            let mut view = points.view();
                            view.shape = &shapes[2];
                            view.strides = &strides[2];
                            view.byte_offset = 1;
                            view
                        });
                        let source = QuantizedExperts::new(
                            "fc1",
                            &packed_view,
                            &scale_view,
                            point_view.as_ref(),
                            EXPERTS,
                            OUTPUTS,
                            INPUTS,
                            bits,
                            BLOCK,
                            prefer_mmap,
                        )
                        .unwrap();
                        assert!(
                            !source.is_pageable(),
                            "opaque storage never becomes mmap-pageable"
                        );
                        assert_eq!(matches!(&source.packed, Cow::Borrowed(_)), steps[0] == 1);
                        assert_eq!(matches!(&source.scales, Cow::Borrowed(_)), steps[1] == 1);
                        if steps[0] == 1 {
                            assert_eq!(source.packed.as_ptr(), packed_view.data_ptr::<u8>());
                        }
                        if steps[1] == 1 {
                            assert_eq!(source.scales.as_ptr(), scale_view.data_ptr::<f32>());
                        }
                        assert_eq!(&*source.packed, &quantized.packed);
                        assert_eq!(&*source.scales, &quantized.scales);
                        match (&source.zero_points, &point_view) {
                            (Some(data), Some(view)) => {
                                assert_eq!(matches!(data, Cow::Borrowed(_)), steps[2] == 1);
                                if steps[2] == 1 {
                                    assert_eq!(data.as_ptr(), view.data_ptr::<u8>());
                                }
                                assert_eq!(
                                    data.as_ref(),
                                    quantized.zero_points.as_deref().unwrap()
                                );
                            }
                            (None, None) => {}
                            _ => panic!("optional zero-point storage changed"),
                        }
                        for expert in 0..EXPERTS {
                            let range = expert * OUTPUTS * INPUTS..(expert + 1) * OUTPUTS * INPUTS;
                            assert_eq!(
                                source.dequantize_expert(expert).unwrap(),
                                quantized.dequantized[range]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn quantized_inputs_refuse_nonhost_operands_before_reading_storage() {
        let packed = Owned::u8(&[1, 1, 8], &[0; 8]);
        let scales = Owned::f32(&[1, 1, 1], &[1.0]);
        let points = Owned::u8(&[1, 1, 1], &[0]);
        for (slot, name) in ["experts_weights", "scales", "zero_points"]
            .iter()
            .enumerate()
        {
            let mut views = [packed.view(), scales.view(), points.view()];
            views[slot].device = DeviceId::cuda(0);
            let failure = QuantizedExperts::new(
                "fc1",
                &views[0],
                &views[1],
                Some(&views[2]),
                1,
                1,
                16,
                4,
                16,
                false,
            )
            .err()
            .expect("CPU cannot read a device-only operand");
            let message = failure.to_string();
            assert!(message.contains(&format!("fc1_{name}")), "{message}");
            assert!(message.contains("move this tensor to the CPU"), "{message}");
        }
    }

    fn model_node(
        op: &str,
        inputs: &[Option<(DataType, &[usize])>],
        output_shape: &[usize],
        attrs: &[(&str, Attribute)],
    ) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert("com.microsoft".into(), 1);
        let inputs = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                input.map(|(dtype, shape)| {
                    let value = graph.create_named_value(
                        format!("input_{index}"),
                        dtype,
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
        let mut node = Node::new(NodeId(0), op, inputs, vec![output]);
        node.domain = "com.microsoft".into();
        for (name, value) in attrs {
            node.attributes.insert((*name).into(), value.clone());
        }
        let node = graph.insert_node(node);
        graph.add_output(output);
        (graph, node)
    }

    fn kernel(graph: &Graph, node: NodeId) -> Result<Box<dyn Kernel>> {
        let model = Model::new(graph);
        CpuExecutionProvider::with_resources(test_resources()).get_kernel(
            model.graph.node(node),
            &[],
            1,
        )
    }

    fn attributes(
        bits: usize,
        block_size: usize,
        k: usize,
        normalize: bool,
    ) -> Vec<(&'static str, Attribute)> {
        vec![
            ("expert_weight_bits", Attribute::Int(bits as i64)),
            ("block_size", Attribute::Int(block_size as i64)),
            ("k", Attribute::Int(k as i64)),
            ("activation_type", Attribute::String(b"identity".to_vec())),
            (
                "normalize_routing_weights",
                Attribute::Int(i64::from(normalize)),
            ),
        ]
    }

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
            assert!(
                (got - want).abs() <= 1e-5,
                "index {index}: got {got}, want {want}"
            );
        }
    }

    fn tiny_owned(dtype: DataType, shape: &[usize]) -> Owned {
        Owned {
            bytes: vec![0; dtype.byte_size().max(1)],
            shape: shape.to_vec(),
            strides: vec![0; shape.len()],
            dtype,
        }
    }

    fn overflow_test_kernel() -> Box<dyn Kernel> {
        let inputs = [
            Some((DataType::Float32, &[1, 16][..])),
            Some((DataType::Float32, &[1, 1])),
            Some((DataType::Uint8, &[1, 16, 8])),
            Some((DataType::Float32, &[1, 16, 1])),
            None,
            Some((DataType::Uint8, &[1, 16, 8])),
            Some((DataType::Float32, &[1, 16, 1])),
        ];
        let attrs = attributes(4, 16, 1, false);
        let (graph, node) = model_node("QMoE", &inputs, &[1, 16], &attrs);
        kernel(&graph, node).unwrap()
    }

    fn assert_kernel_failure_contains(result: Result<()>, expected: &str) {
        match result {
            Err(EpError::KernelFailed(message)) => {
                assert!(
                    message.contains(expected),
                    "expected error containing '{expected}', got '{message}'"
                );
            }

            Err(other) => panic!("expected KernelFailed, got {other}"),
            Ok(()) => panic!("overflowing QMoE input unexpectedly succeeded"),
        }
    }

    struct MmapRun {
        output: Vec<f32>,
        catalogs_pageable: bool,
        selected_source_bytes: usize,
    }

    fn interleave_u8(source: &[u8], experts: usize, rows: usize, cols: usize) -> Vec<u8> {
        let mut output = vec![0u8; source.len()];
        for expert in 0..experts {
            for row in 0..rows {
                for col in 0..cols {
                    output[(row * experts + expert) * cols + col] =
                        source[(expert * rows + row) * cols + col];
                }
            }
        }
        output
    }

    fn interleave_f32(source: &[f32], experts: usize, rows: usize, cols: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; source.len()];
        for expert in 0..experts {
            for row in 0..rows {
                for col in 0..cols {
                    output[(row * experts + expert) * cols + col] =
                        source[(expert * rows + row) * cols + col];
                }
            }
        }
        output
    }

    fn append_f32_bytes(target: &mut Vec<u8>, values: &[f32]) {
        target.extend(values.iter().flat_map(|value| value.to_le_bytes()));
    }

    fn test_external_path() -> PathBuf {
        static NEXT_FILE: AtomicU64 = AtomicU64::new(0);
        let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        std::env::current_dir()
            .expect("cwd")
            .join("target-leon")
            .join("weight-offload-tests")
            .join(format!("qmoe-{}-{id}.bin", std::process::id()))
    }

    fn mapped_tensor_view<'a>(
        store: &'a WeightStore,
        weight: &'a WeightRef,
        shape: &'a [usize],
        strides: &'a [i64],
        dtype: DataType,
    ) -> TensorView<'a> {
        let bytes = store.bytes(weight).expect("mapped weight bytes");
        let (mapping_id, offset, len) = store
            .external_mmap_provenance(weight)
            .expect("mapped weight provenance");
        TensorView::new(
            DevicePtr(bytes.as_ptr().cast()),
            dtype,
            shape,
            strides,
            DeviceId::cpu(),
        )
        .with_backing(TensorBacking::ExternalMmap(ExternalMmapRegion {
            mapping_id,
            offset,
            len,
        }))
    }

    fn run_mmap_qmoe(enabled: bool, interleaved: bool, input: &[f32], router: &[f32]) -> MmapRun {
        run_mmap_qmoe_with_k(enabled, interleaved, input, router, 1)
    }

    fn run_mmap_qmoe_with_k(
        enabled: bool,
        interleaved: bool,
        input: &[f32],
        router: &[f32],
        k: usize,
    ) -> MmapRun {
        run_mmap_qmoe_sequence(enabled, interleaved, input, &[router], k)
    }

    fn run_mmap_qmoe_sequence(
        enabled: bool,
        interleaved: bool,
        input: &[f32],
        routers: &[&[f32]],
        k: usize,
    ) -> MmapRun {
        run_mmap_qmoe_sequence_prefetch(
            enabled,
            interleaved,
            input,
            routers,
            k,
            true,
            test_resources(),
            1.0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_mmap_qmoe_sequence_prefetch(
        enabled: bool,
        interleaved: bool,
        input: &[f32],
        routers: &[&[f32]],
        k: usize,
        prefetch_experts: bool,
        resources: crate::CpuMemoryResources,
        scale_multiplier: f32,
    ) -> MmapRun {
        const EXPERTS: usize = 4;
        const ROWS: usize = 4;
        const HIDDEN: usize = 16;
        const INTER: usize = 16;
        const BITS: usize = 4;
        const BLOCK: usize = 16;
        const PACKED: usize = HIDDEN / 2;
        const BLOCKS: usize = 1;

        let mut fc1 = quantize(EXPERTS, INTER, HIDDEN, BITS, BLOCK, false);
        let mut fc2 = quantize(EXPERTS, HIDDEN, INTER, BITS, BLOCK, false);
        for scale in fc1.scales.iter_mut().chain(&mut fc2.scales) {
            *scale *= scale_multiplier;
        }
        let fc1_packed = if interleaved {
            interleave_u8(&fc1.packed, EXPERTS, INTER, PACKED)
        } else {
            fc1.packed
        };
        let fc1_scales = if interleaved {
            interleave_f32(&fc1.scales, EXPERTS, INTER, BLOCKS)
        } else {
            fc1.scales
        };
        let fc2_packed = if interleaved {
            interleave_u8(&fc2.packed, EXPERTS, HIDDEN, PACKED)
        } else {
            fc2.packed
        };
        let fc2_scales = if interleaved {
            interleave_f32(&fc2.scales, EXPERTS, HIDDEN, BLOCKS)
        } else {
            fc2.scales
        };

        let mut external = vec![0u8; 4];
        let fc1_packed_offset = external.len();
        external.extend_from_slice(&fc1_packed);
        let fc1_scales_offset = external.len();
        append_f32_bytes(&mut external, &fc1_scales);
        let fc2_packed_offset = external.len();
        external.extend_from_slice(&fc2_packed);
        let fc2_scales_offset = external.len();
        append_f32_bytes(&mut external, &fc2_scales);

        let path = test_external_path();
        std::fs::create_dir_all(path.parent().expect("external parent"))
            .expect("create external parent");
        std::fs::write(&path, &external).expect("write external weights");
        let fc1_packed_ref = WeightRef::External {
            path: path.clone(),
            offset: fc1_packed_offset,
            length: fc1_packed.len(),
            dtype: DataType::Uint8,
            dims: vec![EXPERTS, INTER, PACKED],
        };
        let fc1_scales_ref = WeightRef::External {
            path: path.clone(),
            offset: fc1_scales_offset,
            length: fc1_scales.len() * 4,
            dtype: DataType::Float32,
            dims: vec![EXPERTS, INTER, BLOCKS],
        };
        let fc2_packed_ref = WeightRef::External {
            path: path.clone(),
            offset: fc2_packed_offset,
            length: fc2_packed.len(),
            dtype: DataType::Uint8,
            dims: vec![EXPERTS, HIDDEN, PACKED],
        };
        let fc2_scales_ref = WeightRef::External {
            path: path.clone(),
            offset: fc2_scales_offset,
            length: fc2_scales.len() * 4,
            dtype: DataType::Float32,
            dims: vec![EXPERTS, HIDDEN, BLOCKS],
        };

        let order = if interleaved {
            ExpertStorageOrder::Interleaved
        } else {
            ExpertStorageOrder::ExpertMajor
        };
        let layout = |rows_per_expert, storage_elements_per_row| ExpertTensorLayout {
            version: 1,
            experts: EXPERTS,
            rows_per_expert,
            storage_elements_per_row,
            order,
            quantization: Some(ExpertQuantization {
                bits: BITS,
                block_size: BLOCK,
                blocks_per_row: BLOCKS,
            }),
        };
        let catalogs = [
            WeightRegionCatalog::classify(&fc1_packed_ref, layout(INTER, PACKED)),
            WeightRegionCatalog::classify(&fc1_scales_ref, layout(INTER, BLOCKS)),
            WeightRegionCatalog::classify(&fc2_packed_ref, layout(HIDDEN, PACKED)),
            WeightRegionCatalog::classify(&fc2_scales_ref, layout(HIDDEN, BLOCKS)),
        ];
        let selected_source_bytes = [1usize, 3usize]
            .into_iter()
            .flat_map(|expert| catalogs.iter().map(move |catalog| (catalog, expert)))
            .map(|(catalog, expert)| catalog.region(expert).map_or(0, |region| region.len))
            .sum();

        let mut store = WeightStore::new();
        store.map_external(&path).expect("map external weights");
        let packed_strides = if interleaved {
            [PACKED as i64, (EXPERTS * PACKED) as i64, 1]
        } else {
            [(INTER * PACKED) as i64, PACKED as i64, 1]
        };
        let fc2_packed_strides = if interleaved {
            [PACKED as i64, (EXPERTS * PACKED) as i64, 1]
        } else {
            [(HIDDEN * PACKED) as i64, PACKED as i64, 1]
        };
        let scale_strides = if interleaved {
            [BLOCKS as i64, (EXPERTS * BLOCKS) as i64, 1]
        } else {
            [(INTER * BLOCKS) as i64, BLOCKS as i64, 1]
        };
        let fc2_scale_strides = if interleaved {
            [BLOCKS as i64, (EXPERTS * BLOCKS) as i64, 1]
        } else {
            [(HIDDEN * BLOCKS) as i64, BLOCKS as i64, 1]
        };
        let fc1_packed_shape = [EXPERTS, INTER, PACKED];
        let fc1_scale_shape = [EXPERTS, INTER, BLOCKS];
        let fc2_packed_shape = [EXPERTS, HIDDEN, PACKED];
        let fc2_scale_shape = [EXPERTS, HIDDEN, BLOCKS];
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/qmoe_weight_offload/model.onnx.textproto");
        let (graph, fixture_store) =
            load_model_with_weights(&fixture).expect("load ONNX IR QMoE fixture");
        let node = graph
            .nodes
            .iter()
            .find_map(|(id, node)| (node.op_type == "QMoE").then_some(id))
            .expect("QMoE fixture node");
        for input in [2usize, 3, 5, 6] {
            let value = graph.node(node).inputs[input].expect("initializer input");
            assert!(matches!(
                graph.initializers.get(&value),
                Some(WeightRef::External { .. })
            ));
        }
        drop(fixture_store);
        let mut attributes = MoeAttributes::from_node(graph.node(node)).expect("attributes");
        attributes.k = k;
        let kernel = QMoEKernel {
            layer_id: graph.node(node).id.0,
            attributes,
            bits: BITS,
            block_size: BLOCK,
            weight_offload: WeightOffloadMode { enabled },
            host_cache: resources.expert_cache().clone(),
            prefetch_experts,
        };
        let x = Owned::f32(&[ROWS, HIDDEN], input);
        let mut final_output = None;
        for router in routers {
            let router = Owned::f32(&[ROWS, EXPERTS], router);
            let mut output = Owned::zeros_f32(&[ROWS, HIDDEN]);
            kernel
                .execute(
                    &[
                        x.view(),
                        router.view(),
                        mapped_tensor_view(
                            &store,
                            &fc1_packed_ref,
                            &fc1_packed_shape,
                            &packed_strides,
                            DataType::Uint8,
                        ),
                        mapped_tensor_view(
                            &store,
                            &fc1_scales_ref,
                            &fc1_scale_shape,
                            &scale_strides,
                            DataType::Float32,
                        ),
                        TensorView::absent(DataType::Float32),
                        mapped_tensor_view(
                            &store,
                            &fc2_packed_ref,
                            &fc2_packed_shape,
                            &fc2_packed_strides,
                            DataType::Uint8,
                        ),
                        mapped_tensor_view(
                            &store,
                            &fc2_scales_ref,
                            &fc2_scale_shape,
                            &fc2_scale_strides,
                            DataType::Float32,
                        ),
                    ],
                    &mut [output.view_mut()],
                )
                .expect("run mmap QMoE");
            final_output = Some(output.to_f32());
        }
        let output = final_output.expect("at least one QMoE execution");
        drop(store);
        std::fs::remove_file(path).expect("remove external weights");
        MmapRun {
            output,
            catalogs_pageable: catalogs
                .iter()
                .all(|catalog| catalog.pageability() == &Pageability::Pageable),
            selected_source_bytes,
        }
    }

    /// Configurable mmap-backed route-first QMoE workload used for A/B
    /// prefetch benchmarking. Unlike [`run_mmap_qmoe_sequence_prefetch`] (fixed
    /// 4×16 fixture dims), this builds arbitrarily sized expert-major weights so
    /// the host-side dequant cost is large enough to observe compute/transfer
    /// overlap. Returns the final output, a stats snapshot, and the wall-clock
    /// elapsed across all decode steps.
    #[allow(clippy::too_many_arguments)]
    fn run_route_first_bench(
        experts: usize,
        hidden: usize,
        inter: usize,
        bits: usize,
        block: usize,
        rows: usize,
        k: usize,
        host_cache_budget: u64,
        routers: &[Vec<f32>],
        prefetch_experts: bool,
    ) -> (Vec<f32>, WeightOffloadStats, std::time::Duration) {
        let pack_size = 8 / bits;
        let packed_fc1 = hidden / pack_size;
        let packed_fc2 = inter / pack_size;
        let blocks_fc1 = hidden / block;
        let blocks_fc2 = inter / block;

        let fc1 = quantize(experts, inter, hidden, bits, block, false);
        let fc2 = quantize(experts, hidden, inter, bits, block, false);

        let mut external = vec![0u8; 4];
        let fc1_packed_offset = external.len();
        external.extend_from_slice(&fc1.packed);
        let fc1_scales_offset = external.len();
        append_f32_bytes(&mut external, &fc1.scales);
        let fc2_packed_offset = external.len();
        external.extend_from_slice(&fc2.packed);
        let fc2_scales_offset = external.len();
        append_f32_bytes(&mut external, &fc2.scales);

        let path = test_external_path();
        std::fs::create_dir_all(path.parent().expect("external parent"))
            .expect("create external parent");
        std::fs::write(&path, &external).expect("write external weights");

        let fc1_packed_ref = WeightRef::External {
            path: path.clone(),
            offset: fc1_packed_offset,
            length: fc1.packed.len(),
            dtype: DataType::Uint8,
            dims: vec![experts, inter, packed_fc1],
        };
        let fc1_scales_ref = WeightRef::External {
            path: path.clone(),
            offset: fc1_scales_offset,
            length: fc1.scales.len() * 4,
            dtype: DataType::Float32,
            dims: vec![experts, inter, blocks_fc1],
        };
        let fc2_packed_ref = WeightRef::External {
            path: path.clone(),
            offset: fc2_packed_offset,
            length: fc2.packed.len(),
            dtype: DataType::Uint8,
            dims: vec![experts, hidden, packed_fc2],
        };
        let fc2_scales_ref = WeightRef::External {
            path: path.clone(),
            offset: fc2_scales_offset,
            length: fc2.scales.len() * 4,
            dtype: DataType::Float32,
            dims: vec![experts, hidden, blocks_fc2],
        };

        let layout =
            |rows_per_expert, storage_elements_per_row, blocks_per_row| ExpertTensorLayout {
                version: 1,
                experts,
                rows_per_expert,
                storage_elements_per_row,
                order: ExpertStorageOrder::ExpertMajor,
                quantization: Some(ExpertQuantization {
                    bits,
                    block_size: block,
                    blocks_per_row,
                }),
            };
        let catalogs = [
            WeightRegionCatalog::classify(&fc1_packed_ref, layout(inter, packed_fc1, blocks_fc1)),
            WeightRegionCatalog::classify(&fc1_scales_ref, layout(inter, blocks_fc1, blocks_fc1)),
            WeightRegionCatalog::classify(&fc2_packed_ref, layout(hidden, packed_fc2, blocks_fc2)),
            WeightRegionCatalog::classify(&fc2_scales_ref, layout(hidden, blocks_fc2, blocks_fc2)),
        ];
        assert!(
            catalogs
                .iter()
                .all(|catalog| catalog.pageability() == &Pageability::Pageable),
            "benchmark weights must be pageable to exercise route-first offload"
        );

        let mut store = WeightStore::new();
        store.map_external(&path).expect("map external weights");
        let fc1_packed_shape = [experts, inter, packed_fc1];
        let fc1_scale_shape = [experts, inter, blocks_fc1];
        let fc2_packed_shape = [experts, hidden, packed_fc2];
        let fc2_scale_shape = [experts, hidden, blocks_fc2];
        let fc1_packed_strides = [(inter * packed_fc1) as i64, packed_fc1 as i64, 1];
        let fc1_scale_strides = [(inter * blocks_fc1) as i64, blocks_fc1 as i64, 1];
        let fc2_packed_strides = [(hidden * packed_fc2) as i64, packed_fc2 as i64, 1];
        let fc2_scale_strides = [(hidden * blocks_fc2) as i64, blocks_fc2 as i64, 1];

        let (graph, node) = model_node(
            "QMoE",
            &[
                Some((DataType::Float32, &[rows, hidden][..])),
                Some((DataType::Float32, &[rows, experts])),
                Some((DataType::Uint8, &[experts, inter, packed_fc1])),
                Some((DataType::Float32, &[experts, inter, blocks_fc1])),
                None,
                Some((DataType::Uint8, &[experts, hidden, packed_fc2])),
                Some((DataType::Float32, &[experts, hidden, blocks_fc2])),
            ],
            &[rows, hidden],
            &attributes(bits, block, k, false),
        );
        let attributes = MoeAttributes::from_node(graph.node(node)).expect("attributes");
        reset_offload_test_state(host_cache_budget);
        let kernel = QMoEKernel {
            layer_id: graph.node(node).id.0,
            attributes,
            bits,
            block_size: block,
            weight_offload: WeightOffloadMode { enabled: true },
            host_cache: test_resources().expert_cache().clone(),
            prefetch_experts,
        };

        let input = pseudo_random_values(rows * hidden, 0xa5a5_1234);
        let x = Owned::f32(&[rows, hidden], &input);
        let mut final_output = None;
        let start = std::time::Instant::now();
        for router_values in routers {
            let router = Owned::f32(&[rows, experts], router_values);
            let mut output = Owned::zeros_f32(&[rows, hidden]);
            kernel
                .execute(
                    &[
                        x.view(),
                        router.view(),
                        mapped_tensor_view(
                            &store,
                            &fc1_packed_ref,
                            &fc1_packed_shape,
                            &fc1_packed_strides,
                            DataType::Uint8,
                        ),
                        mapped_tensor_view(
                            &store,
                            &fc1_scales_ref,
                            &fc1_scale_shape,
                            &fc1_scale_strides,
                            DataType::Float32,
                        ),
                        TensorView::absent(DataType::Float32),
                        mapped_tensor_view(
                            &store,
                            &fc2_packed_ref,
                            &fc2_packed_shape,
                            &fc2_packed_strides,
                            DataType::Uint8,
                        ),
                        mapped_tensor_view(
                            &store,
                            &fc2_scales_ref,
                            &fc2_scale_shape,
                            &fc2_scale_strides,
                            DataType::Float32,
                        ),
                    ],
                    &mut [output.view_mut()],
                )
                .expect("run mmap QMoE");
            final_output = Some(output.to_f32());
        }
        let elapsed = start.elapsed();
        let stats = crate::weight_offload_stats();
        drop(store);
        std::fs::remove_file(path).expect("remove external weights");
        (
            final_output.expect("at least one execution"),
            stats,
            elapsed,
        )
    }

    fn bench_routers(experts: usize, rows: usize, steps: usize, seed: u64) -> Vec<Vec<f32>> {
        (0..steps)
            .map(|step| pseudo_random_values(rows * experts, seed.wrapping_add(step as u64 * 97)))
            .collect()
    }

    #[test]
    fn prefetch_pipeline_is_byte_and_stat_identical_to_serial() {
        let _guard = hold_metrics_test_lock();
        const EXPERTS: usize = 8;
        const HIDDEN: usize = 64;
        const INTER: usize = 128;
        const ROWS: usize = 6;
        let routers = bench_routers(EXPERTS, ROWS, 6, 0xbead_1111);
        let budget: u64 = (3 * (INTER * HIDDEN + HIDDEN * INTER) * 4) as u64;
        let (serial_out, serial_stats, _) = run_route_first_bench(
            EXPERTS, HIDDEN, INTER, 4, 16, ROWS, 2, budget, &routers, false,
        );
        let (pipe_out, pipe_stats, _) = run_route_first_bench(
            EXPERTS, HIDDEN, INTER, 4, 16, ROWS, 2, budget, &routers, true,
        );
        assert_eq!(serial_out, pipe_out, "prefetch must be byte-identical");
        assert_eq!(serial_stats.host_cache_hits, pipe_stats.host_cache_hits);
        assert_eq!(serial_stats.host_cache_misses, pipe_stats.host_cache_misses);
        assert_eq!(
            serial_stats.host_cache_evictions,
            pipe_stats.host_cache_evictions
        );
        assert_eq!(
            serial_stats.bytes_read_from_mmap,
            pipe_stats.bytes_read_from_mmap
        );
        assert!(
            serial_stats.host_cache_misses > 0 && serial_stats.host_cache_evictions > 0,
            "workload must force real admission and eviction (misses={}, evictions={})",
            serial_stats.host_cache_misses,
            serial_stats.host_cache_evictions
        );
    }

    #[test]
    fn prefetch_pipeline_survives_cache_smaller_than_working_set() {
        let _guard = hold_metrics_test_lock();
        const EXPERTS: usize = 6;
        const HIDDEN: usize = 32;
        const INTER: usize = 64;
        const ROWS: usize = 6;
        // Budget for a single expert forces eviction whenever >1 expert is hot.
        let one_expert: u64 = ((INTER * HIDDEN + HIDDEN * INTER) * 4) as u64;
        let routers = bench_routers(EXPERTS, ROWS, 8, 0xfeed_2222);
        let (serial_out, _, _) = run_route_first_bench(
            EXPERTS, HIDDEN, INTER, 4, 16, ROWS, 3, one_expert, &routers, false,
        );
        let (pipe_out, stats, _) = run_route_first_bench(
            EXPERTS, HIDDEN, INTER, 4, 16, ROWS, 3, one_expert, &routers, true,
        );
        assert_eq!(serial_out, pipe_out);
        assert!(stats.owned_host_cache_bytes <= one_expert);
        assert!(stats.peak_owned_host_cache_bytes <= one_expert);
    }

    /// Multi-routing / multi-budget sweep proving the prefetch pipeline is
    /// observationally indistinguishable from the serial route-first loop.
    ///
    /// The single-ahead pipeline holds expert `n`'s lease while admitting
    /// `n+1`; the extra in-flight `Arc` protects entry `n` from eviction, so in
    /// the partial-cache regime the pipeline and serial retain different experts
    /// and diverge on hits/misses/evictions/bytes-read. The eviction-neutrality
    /// guard engages the pipeline only where no eviction can occur while a
    /// prefetch is in flight (`budget == 0` streaming, or `budget >= experts`
    /// fully resident) and otherwise falls back to the serial loop, so the two
    /// paths are byte- and stat-identical everywhere.
    ///
    /// The router seeds were mined from a guard-less run and reproduce Gaff's
    /// cited divergences on this fixture (e.g. seed `0x1000+3·131`/k=1/budget=3×
    /// read +3072 mmap bytes; seed `0x1000+8·131`/k=2/budget=3× diverged on
    /// misses). The sweep drives them across the engaged budgets (0× and the
    /// fully-resident 6×/8×) and the partial budgets that must fall back, and
    /// asserts for every combination:
    ///   * byte-identical output,
    ///   * identical hits/misses/evictions/bytes-read, and
    ///   * bytes-read is never a regression (`pipe <= serial`).
    ///
    /// Removing the guard lets the pipeline engage under a partial budget and
    /// reintroduces the divergences, failing this test.
    #[test]
    fn prefetch_matches_serial_across_routing_and_budget_sweep() {
        let _guard = hold_metrics_test_lock();
        // Dimensions chosen so the host-side working set exceeds a few-expert
        // budget, forcing real cross-step admission/eviction (the regime where
        // an extra in-flight lease can perturb victim selection).
        const EXPERTS: usize = 6;
        const HIDDEN: usize = 32;
        const INTER: usize = 64;
        const ROWS: usize = 6;
        const STEPS: usize = 16;
        let one_expert: u64 = ((INTER * HIDDEN + HIDDEN * INTER) * 4) as u64;

        // Router seeds whose guard-less single-ahead pipeline diverges from
        // serial on this fixture (mined offline); includes Gaff's cited
        // seed=3/k=1 and seed=8/k=2 regimes.
        let router_seeds: [u64; 8] = [
            0x1000 + 3 * 131,
            0x1000 + 7 * 131,
            0x1000 + 8 * 131,
            0x1000 + 13 * 131,
            0x1000 + 27 * 131,
            0x1000 + 33 * 131,
            0x1000 + 34 * 131,
            0x1000 + 39 * 131,
        ];
        let ks: [usize; 3] = [1, 2, 3];
        // 0× engages the streaming pipeline; 1×–4× are partial (serial
        // fallback); 6× (== EXPERTS) and 8× are fully resident (engaged, no
        // eviction). All must be indistinguishable from serial.
        let budget_experts: [u64; 6] = [0, 1, 3, 4, 6, 8];

        let mut saw_partial_eviction = false;
        let mut saw_engaged_streaming = false;
        for &seed in &router_seeds {
            for &k in &ks {
                let routers = bench_routers(EXPERTS, ROWS, STEPS, seed);
                for &mult in &budget_experts {
                    let budget = mult * one_expert;
                    let (serial_out, serial_stats, _) = run_route_first_bench(
                        EXPERTS, HIDDEN, INTER, 4, 16, ROWS, k, budget, &routers, false,
                    );
                    let (pipe_out, pipe_stats, _) = run_route_first_bench(
                        EXPERTS, HIDDEN, INTER, 4, 16, ROWS, k, budget, &routers, true,
                    );
                    let ctx = format!("seed={seed:#x} k={k} budget={mult}x");
                    assert_eq!(
                        serial_out, pipe_out,
                        "output must be byte-identical ({ctx})"
                    );
                    assert_eq!(
                        serial_stats.host_cache_hits, pipe_stats.host_cache_hits,
                        "hits diverged ({ctx})"
                    );
                    assert_eq!(
                        serial_stats.host_cache_misses, pipe_stats.host_cache_misses,
                        "misses diverged ({ctx})"
                    );
                    assert_eq!(
                        serial_stats.host_cache_evictions, pipe_stats.host_cache_evictions,
                        "evictions diverged ({ctx})"
                    );
                    assert_eq!(
                        serial_stats.bytes_read_from_mmap, pipe_stats.bytes_read_from_mmap,
                        "bytes-read diverged ({ctx})"
                    );
                    // The feature's own intent: never read more than serial.
                    assert!(
                        pipe_stats.bytes_read_from_mmap <= serial_stats.bytes_read_from_mmap,
                        "prefetch read more mmap bytes than serial ({ctx})"
                    );
                    // The partial budgets (which fall back to serial) must
                    // actually exercise eviction, else the guard-removal
                    // mutation would have nothing to diverge on.
                    if (1..EXPERTS as u64).contains(&mult) && serial_stats.host_cache_evictions > 0
                    {
                        saw_partial_eviction = true;
                    }
                    // The streaming budget engages the pipeline with real leases.
                    if mult == 0 && serial_stats.host_cache_misses > 0 {
                        saw_engaged_streaming = true;
                    }
                }
            }
        }
        assert!(
            saw_partial_eviction,
            "sweep must exercise eviction in a partial-budget (fallback) regime"
        );
        assert!(
            saw_engaged_streaming,
            "sweep must exercise the engaged streaming pipeline (budget == 0)"
        );
    }

    /// Mutation probe pinned to the exact regimes Gaff cited: seed=3/k=1 (the
    /// +3072-byte default-on regression) and seed=8/k=2 (evictions/misses
    /// divergence). It exercises both the engaged pipeline and the fallback,
    /// and fails if the eviction-neutrality guard is removed.
    ///
    /// At the partial budget (3×) these fixtures evict, and the original
    /// guard-less pipeline diverged from serial; the guard now forces serial
    /// fallback, so the counters match exactly — deleting the guard re-engages
    /// the pipeline here and reintroduces the divergence, failing the equality
    /// assertions. At `budget == 0` the pipeline is engaged (streaming) and is
    /// asserted byte- and stat-identical, proving the fix does not simply
    /// disable prefetch.
    #[test]
    fn prefetch_guard_holds_stats_in_gaff_divergent_regime() {
        let _guard = hold_metrics_test_lock();
        const EXPERTS: usize = 6;
        const HIDDEN: usize = 32;
        const INTER: usize = 64;
        const ROWS: usize = 6;
        const STEPS: usize = 16;
        let one_expert: u64 = ((INTER * HIDDEN + HIDDEN * INTER) * 4) as u64;

        // (router seed, k) pairs Gaff's sweep flagged as divergent under the
        // original guard-less pipeline on this fixture.
        let cases: [(u64, usize); 2] = [(0x1000 + 3 * 131, 1), (0x1000 + 8 * 131, 2)];
        let mut fallback_evicted = false;
        let mut streaming_missed = false;
        for &(seed, k) in &cases {
            let routers = bench_routers(EXPERTS, ROWS, STEPS, seed);
            // 3× is a partial budget: the guard falls back to serial here.
            // 0× is streaming: the guard engages the pipeline here.
            for &budget in &[3 * one_expert, 0] {
                let (serial_out, serial_stats, _) = run_route_first_bench(
                    EXPERTS, HIDDEN, INTER, 4, 16, ROWS, k, budget, &routers, false,
                );
                let (pipe_out, pipe_stats, _) = run_route_first_bench(
                    EXPERTS, HIDDEN, INTER, 4, 16, ROWS, k, budget, &routers, true,
                );
                let ctx = format!("seed={seed:#x} k={k} budget={budget}");
                assert_eq!(
                    serial_out, pipe_out,
                    "output must be byte-identical ({ctx})"
                );
                assert_eq!(
                    serial_stats.host_cache_misses, pipe_stats.host_cache_misses,
                    "miss count must match serial ({ctx})"
                );
                assert_eq!(
                    serial_stats.host_cache_evictions, pipe_stats.host_cache_evictions,
                    "eviction count must match serial ({ctx})"
                );
                assert_eq!(
                    serial_stats.bytes_read_from_mmap, pipe_stats.bytes_read_from_mmap,
                    "bytes-read must match serial — no default-on regression ({ctx})"
                );
                assert!(
                    pipe_stats.bytes_read_from_mmap <= serial_stats.bytes_read_from_mmap,
                    "prefetch must never read more mmap bytes than serial ({ctx})"
                );
                if budget > 0 {
                    fallback_evicted |= serial_stats.host_cache_evictions > 0;
                } else {
                    streaming_missed |= serial_stats.host_cache_misses > 0;
                }
            }
        }
        assert!(
            fallback_evicted,
            "the cited partial-budget regimes must evict for the guard-removal probe to bite"
        );
        assert!(
            streaming_missed,
            "the streaming regime must issue real leases so the engaged path is exercised"
        );
    }

    /// A/B micro-benchmark. Ignored by default (timing-sensitive); run with:
    /// `cargo test -p onnx-runtime-ep-cpu prefetch_ab_benchmark -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing benchmark; run explicitly with --ignored --nocapture"]
    fn prefetch_ab_benchmark() {
        let _guard = hold_metrics_test_lock();
        const EXPERTS: usize = 16;
        const HIDDEN: usize = 256;
        const INTER: usize = 512;
        const ROWS: usize = 8;
        const STEPS: usize = 64;
        const K: usize = 4;
        let routers = bench_routers(EXPERTS, ROWS, STEPS, 0x5eed_c0de);
        // Streaming host budget (0 = no host cache): every routed expert is
        // dequantized from mmap each step, so the pipeline overlaps the full
        // transfer with compute. This is the regime the eviction-neutrality
        // guard engages (no admission/eviction state to perturb), so the A/B is
        // both representative of the win and byte-/stat-identical.
        let budget: u64 = 0;

        // Warm both configurations once (JIT of page faults) then time.
        let mut serial = None;
        let mut pipe = None;
        for _ in 0..3 {
            serial = Some(run_route_first_bench(
                EXPERTS, HIDDEN, INTER, 4, 32, ROWS, K, budget, &routers, false,
            ));
            pipe = Some(run_route_first_bench(
                EXPERTS, HIDDEN, INTER, 4, 32, ROWS, K, budget, &routers, true,
            ));
        }
        let (serial_out, serial_stats, serial_elapsed) = serial.unwrap();
        let (pipe_out, pipe_stats, pipe_elapsed) = pipe.unwrap();
        assert_eq!(serial_out, pipe_out, "A/B outputs must be byte-identical");

        let tokens = (ROWS * STEPS) as f64;
        let serial_tps = tokens / serial_elapsed.as_secs_f64();
        let pipe_tps = tokens / pipe_elapsed.as_secs_f64();
        let hit_rate = |s: &WeightOffloadStats| {
            let total = s.host_cache_hits + s.host_cache_misses;
            if total == 0 {
                0.0
            } else {
                s.host_cache_hits as f64 / total as f64
            }
        };
        println!(
            "=== route-first QMoE prefetch A/B (experts={EXPERTS} hidden={HIDDEN} inter={INTER} rows={ROWS} steps={STEPS} k={K}) ==="
        );
        println!(
            "BEFORE (serial):    {:>8.1} tok/s | {:>7.2} ms | hits={} misses={} evictions={} hit-rate={:.1}%",
            serial_tps,
            serial_elapsed.as_secs_f64() * 1e3,
            serial_stats.host_cache_hits,
            serial_stats.host_cache_misses,
            serial_stats.host_cache_evictions,
            hit_rate(&serial_stats) * 100.0
        );
        println!(
            "AFTER  (prefetch):  {:>8.1} tok/s | {:>7.2} ms | hits={} misses={} evictions={} hit-rate={:.1}%",
            pipe_tps,
            pipe_elapsed.as_secs_f64() * 1e3,
            pipe_stats.host_cache_hits,
            pipe_stats.host_cache_misses,
            pipe_stats.host_cache_evictions,
            hit_rate(&pipe_stats) * 100.0
        );
        println!(
            "speedup: {:.2}x ({:+.1}%)",
            pipe_tps / serial_tps,
            (pipe_tps / serial_tps - 1.0) * 100.0
        );
    }

    fn pseudo_random_values(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((state >> 40) as u32 as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn metrics_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Serialize the offload tests on the shared process-global metrics, and do
    /// so *robustly against poisoning*.
    ///
    /// These tests all mutate the same process-wide `metrics()` singleton, so
    /// they must run one at a time. If any one of them panics (a genuine
    /// assertion failure) while holding this mutex, the standard library marks
    /// the mutex poisoned; every subsequent `lock().expect(...)` then panics
    /// with `PoisonError` instead of running. That amplifies a single real
    /// failure into a cascade of unrelated "failures" that hides the true
    /// culprit. We therefore recover the guard from a poisoned mutex: the
    /// offending test still reports its own failure, but the remaining tests
    /// continue to run and report honestly (one failure stays one failure).
    fn hold_metrics_test_lock() -> std::sync::MutexGuard<'static, ()> {
        metrics_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    thread_local! {
        static TEST_RESOURCES: std::cell::RefCell<crate::CpuMemoryResources> =
            std::cell::RefCell::new(crate::CpuMemoryResources::default());
    }

    fn test_resources() -> crate::CpuMemoryResources {
        TEST_RESOURCES.with(|resources| resources.borrow().clone())
    }

    fn reset_offload_test_state(host_cache_budget: u64) {
        TEST_RESOURCES.with(|resources| {
            let mut resources = resources.borrow_mut();
            resources.expert_cache().clear();
            *resources = crate::CpuMemoryResources::new(
                DerivedWeightMemory::default(),
                crate::CpuWeightCacheLimits {
                    expert_bytes: host_cache_budget,
                    ..Default::default()
                },
            )
            .unwrap();
        });
        metrics().reset();
    }

    fn cache_key(expert: usize) -> ExpertCacheKey {
        ExpertCacheKey {
            layer_id: 7,
            expert,
            bits: 4,
            block_size: 16,
            out_and_in_features: vec![(1, 1)],
            source_regions: Vec::new(),
        }
    }

    fn run_equivalence(
        bits: usize,
        hidden: usize,
        inter: usize,
        block_size: usize,
        k: usize,
        normalize: bool,
        affine: bool,
    ) {
        run_equivalence_shape(bits, hidden, inter, block_size, k, normalize, affine, 2, 2);
    }

    #[allow(clippy::too_many_arguments)]
    fn run_equivalence_shape(
        bits: usize,
        hidden: usize,
        inter: usize,
        block_size: usize,
        k: usize,
        normalize: bool,
        affine: bool,
        experts: usize,
        rows: usize,
    ) {
        let fc1 = quantize(experts, inter, hidden, bits, block_size, affine);
        let fc2 = quantize(experts, hidden, inter, bits, block_size, affine);
        let input: Vec<f32> = (0..rows * hidden)
            .map(|index| (index % 5) as f32 * 0.25 - 0.5)
            .collect();
        let router: Vec<f32> = (0..rows * experts)
            .map(|index| ((index * 7 + index / experts * 5) % 17) as f32 - 8.0)
            .collect();
        let pack_size = 8 / bits;
        let hidden_blocks = hidden / block_size;
        let inter_blocks = inter / block_size;

        let float_shapes = [
            Some((DataType::Float32, &[rows, hidden][..])),
            Some((DataType::Float32, &[rows, experts])),
            Some((DataType::Float32, &[experts, inter, hidden])),
            None,
            Some((DataType::Float32, &[experts, hidden, inter])),
        ];
        let float_attrs = [
            ("k", Attribute::Int(k as i64)),
            ("activation_type", Attribute::String(b"identity".to_vec())),
            (
                "normalize_routing_weights",
                Attribute::Int(i64::from(normalize)),
            ),
        ];
        let (float_graph, float_node) =
            model_node("MoE", &float_shapes, &[rows, hidden], &float_attrs);
        let x = Owned::f32(&[rows, hidden], &input);
        let router_tensor = Owned::f32(&[rows, experts], &router);
        let fc1_float = Owned::f32(&[experts, inter, hidden], &fc1.dequantized);
        let fc2_float = Owned::f32(&[experts, hidden, inter], &fc2.dequantized);
        let mut float_output = Owned::zeros_f32(&[rows, hidden]);
        kernel(&float_graph, float_node)
            .unwrap()
            .execute(
                &[
                    x.view(),
                    router_tensor.view(),
                    fc1_float.view(),
                    TensorView::absent(DataType::Float32),
                    fc2_float.view(),
                ],
                &mut [float_output.view_mut()],
            )
            .unwrap();
        let moe_attributes =
            crate::kernels::moe::MoeAttributes::from_node(float_graph.node(float_node)).unwrap();
        let mut dense_fallback = vec![0.0f32; rows * hidden];
        for row in 0..rows {
            for (expert, route_weight) in crate::kernels::moe::routing_weights(
                &router[row * experts..(row + 1) * experts],
                None,
                k,
                normalize,
            ) {
                let expert_output = crate::kernels::moe::run_expert(
                    &input[row * hidden..(row + 1) * hidden],
                    &fc1.dequantized[expert * inter * hidden..(expert + 1) * inter * hidden],
                    None,
                    &fc2.dequantized[expert * hidden * inter..(expert + 1) * hidden * inter],
                    None,
                    None,
                    None,
                    inter,
                    hidden,
                    inter,
                    &moe_attributes,
                );
                for feature in 0..hidden {
                    dense_fallback[row * hidden + feature] += route_weight * expert_output[feature];
                }
            }
        }
        assert_close(&float_output.to_f32(), &dense_fallback);

        let fc1_zero_point_shape = [experts, inter, hidden_blocks.div_ceil(pack_size)];
        let fc2_zero_point_shape = [experts, hidden, inter_blocks.div_ceil(pack_size)];
        let q_shapes = [
            Some((DataType::Float32, &[rows, hidden][..])),
            Some((DataType::Float32, &[rows, experts])),
            Some((DataType::Uint8, &[experts, inter, hidden / pack_size])),
            Some((DataType::Float32, &[experts, inter, hidden_blocks])),
            None,
            Some((DataType::Uint8, &[experts, hidden, inter / pack_size])),
            Some((DataType::Float32, &[experts, hidden, inter_blocks])),
            None,
            None,
            None,
            None,
            affine.then_some((DataType::Uint8, &fc1_zero_point_shape[..])),
            affine.then_some((DataType::Uint8, &fc2_zero_point_shape[..])),
        ];
        let q_attrs = attributes(bits, block_size, k, normalize);
        let (q_graph, q_node) = model_node("QMoE", &q_shapes, &[rows, hidden], &q_attrs);
        let fc1_packed = Owned::u8(&[experts, inter, hidden / pack_size], &fc1.packed);
        let fc1_scales = Owned::f32(&[experts, inter, hidden_blocks], &fc1.scales);
        let fc2_packed = Owned::u8(&[experts, hidden, inter / pack_size], &fc2.packed);
        let fc2_scales = Owned::f32(&[experts, hidden, inter_blocks], &fc2.scales);
        let fc1_zero_points = fc1
            .zero_points
            .as_ref()
            .map(|points| Owned::u8(&[experts, inter, hidden_blocks.div_ceil(pack_size)], points));
        let fc2_zero_points = fc2
            .zero_points
            .as_ref()
            .map(|points| Owned::u8(&[experts, hidden, inter_blocks.div_ceil(pack_size)], points));
        let q_inputs = [
            x.view(),
            router_tensor.view(),
            fc1_packed.view(),
            fc1_scales.view(),
            TensorView::absent(DataType::Float32),
            fc2_packed.view(),
            fc2_scales.view(),
            TensorView::absent(DataType::Float32),
            TensorView::absent(DataType::Uint8),
            TensorView::absent(DataType::Float32),
            TensorView::absent(DataType::Float32),
            fc1_zero_points
                .as_ref()
                .map_or_else(|| TensorView::absent(DataType::Uint8), Owned::view),
            fc2_zero_points
                .as_ref()
                .map_or_else(|| TensorView::absent(DataType::Uint8), Owned::view),
        ];
        let mut q_output = Owned::zeros_f32(&[rows, hidden]);
        kernel(&q_graph, q_node)
            .unwrap()
            .execute(&q_inputs, &mut [q_output.view_mut()])
            .unwrap();
        assert_close(&q_output.to_f32(), &float_output.to_f32());
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let expanded_bytes = (inter * hidden + hidden * inter) * std::mem::size_of::<f32>();
        for capacity in [0, expanded_bytes - 1, expanded_bytes] {
            let governor = LedgerGovernor::new(LeaseLedger::new(0, capacity as u64, 0));
            let provider = CpuExecutionProvider::with_resources(
                crate::CpuMemoryResources::new(
                    DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap(),
                    crate::CpuWeightCacheLimits::default(),
                )
                .unwrap(),
            );
            let governed = provider.get_kernel(q_graph.node(q_node), &[], 1).unwrap();
            governed
                .execute(&q_inputs, &mut [q_output.view_mut()])
                .unwrap();
            assert_close(&q_output.to_f32(), &float_output.to_f32());
            assert_eq!(
                governor.used(Tier::Host),
                0,
                "transient payloads must return their grants after execution"
            );
        }
    }

    #[test]
    fn transient_expert_grants_cover_complete_sets_and_stream_all_gates() {
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        const H: usize = 48;
        const I: usize = 48;
        const E: usize = 2;
        for bits in [2, 4, 8] {
            for affine in [false, true] {
                for (activation, fusion, separate) in [
                    ("identity", 0, false),
                    ("relu", 0, false),
                    ("gelu", 0, false),
                    ("silu", 0, false),
                    ("silu", 0, true),
                    ("swiglu", 0, true),
                    ("swiglu", 1, false),
                    ("swiglu", 2, false),
                ] {
                    let mut attrs = attributes(bits, 16, 2, true);
                    attrs
                        .iter_mut()
                        .find(|(key, _)| *key == "activation_type")
                        .unwrap()
                        .1 = Attribute::String(activation.as_bytes().to_vec());
                    attrs.push(("swiglu_fusion", Attribute::Int(fusion)));
                    let (graph, node) = model_node("QMoE", &[], &[2, H], &attrs);
                    let attrs = quantized_moe_attributes(graph.node(node)).unwrap().0;
                    let fc1_size = attrs.checked_fc1_size(I, "test QMoE").unwrap();
                    let dims = [(fc1_size, H), (H, I), (I, H)];
                    let mut q: [_; 3] = std::array::from_fn(|i| {
                        quantize(E, dims[i].0, dims[i].1, bits, 16, affine)
                    });
                    for bank in &mut q {
                        for scale in &mut bank.scales {
                            *scale *= 0.73145;
                        }
                    }
                    let p: [_; 3] = std::array::from_fn(|i| {
                        Owned::u8(&[E, dims[i].0, dims[i].1 / (8 / bits)], &q[i].packed)
                    });
                    let sc: [_; 3] = std::array::from_fn(|i| {
                        Owned::f32(&[E, dims[i].0, dims[i].1 / 16], &q[i].scales)
                    });
                    let zp: [_; 3] = std::array::from_fn(|i| {
                        let shape = [E, dims[i].0, (dims[i].1 / 16).div_ceil(8 / bits)];
                        let fallback = vec![0; shape.iter().product()];
                        Owned::u8(&shape, q[i].zero_points.as_deref().unwrap_or(&fallback))
                    });
                    let pv: [_; 3] = std::array::from_fn(|i| p[i].view());
                    let sv: [_; 3] = std::array::from_fn(|i| sc[i].view());
                    let zv: [_; 3] = std::array::from_fn(|i| zp[i].view());
                    let sources: [_; 3] = std::array::from_fn(|i| {
                        QuantizedExperts::new(
                            "test projection",
                            &pv[i],
                            &sv[i],
                            affine.then_some(&zv[i]),
                            E,
                            dims[i].0,
                            dims[i].1,
                            bits,
                            16,
                            false,
                        )
                        .unwrap()
                    });
                    let (fc1, fc2) = (&sources[0], &sources[1]);
                    let fc3 = separate.then_some(&sources[2]);
                    let bytes = DequantizedExpert::expanded_bytes(fc1, fc2, fc3).unwrap();
                    let input = pseudo_random_values(2 * H, 0xdeadbeef);
                    let tasks = [(1, 0, 0.25), (0, 1, 0.75), (1, 2, -0.5)];
                    let biases: [Vec<f32>; 3] = std::array::from_fn(|i| {
                        (0..E * dims[i].0).map(|j| (j % 3) as f32 * 0.125).collect()
                    });
                    let run = |execution: &ExpertExecution| {
                        let mut out = vec![0.; tasks.len() * H];
                        write_grouped_contributions(
                            &mut out,
                            &input,
                            &tasks,
                            1,
                            execution,
                            fc1,
                            fc2,
                            fc3,
                            Some(&biases[0]),
                            Some(&biases[1]),
                            separate.then_some(biases[2].as_slice()),
                            fc1_size,
                            H,
                            I,
                            &attrs,
                        )
                        .unwrap();
                        out
                    };
                    let governor = LedgerGovernor::new(LeaseLedger::new(0, bytes as u64, 0));
                    let memory =
                        DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap();
                    let dense = load_transient_expert(1, fc1, fc2, fc3, &memory, false).unwrap();
                    assert_eq!(governor.used(Tier::Host), bytes as u64);
                    let expected = run(&dense);
                    assert!(
                        matches!(
                            load_transient_expert(1, fc1, fc2, fc3, &memory, false).unwrap(),
                            ExpertExecution::RowStreamed(_)
                        ),
                        "an active full-set owner must keep the quota occupied"
                    );
                    for limit in [0, bytes - 1] {
                        let denied = LedgerGovernor::new(LeaseLedger::new(0, limit as u64, 0));
                        let source =
                            DerivedWeightMemory::governed(&denied, HolderId::new(13)).unwrap();
                        let streamed =
                            load_transient_expert(1, fc1, fc2, fc3, &source, false).unwrap();
                        assert!(matches!(streamed, ExpertExecution::RowStreamed(_)));
                        let actual = run(&streamed);
                        assert_close(&actual, &expected);
                        assert_eq!(denied.used(Tier::Host), 0);
                    }
                    let ExpertExecution::Dense(payload) = &dense else {
                        panic!("exact quota must admit a dense set")
                    };
                    let active = payload.clone();
                    drop(dense);
                    assert_eq!(governor.used(Tier::Host), bytes as u64);
                    drop(active);
                    assert_eq!(governor.used(Tier::Host), 0);
                }
            }
        }
    }

    #[test]
    fn transient_mmap_grants_preserve_prefetch_outputs_and_cache_policy() {
        use onnx_runtime_memory_api::{MemoryError, MemoryRole, Tier};
        use onnx_runtime_memory_governor::{
            HolderId, LeaseAccounting, LeaseLedger, MemoryAuthorityId, MemoryGovernor, MemoryLease,
        };
        #[derive(Debug)]
        struct ObservedBooks {
            ledger: Arc<LeaseLedger>,
            peak: AtomicU64,
            grants: Mutex<Vec<u64>>,
        }
        impl LeaseAccounting for ObservedBooks {
            fn authority_id(&self) -> Option<MemoryAuthorityId> {
                self.ledger.authority_id()
            }
            fn release(&self, tier: Tier, bytes: u64) {
                self.ledger.release(tier, bytes);
            }
            fn try_claim(
                &self,
                tier: Tier,
                bytes: u64,
                role: MemoryRole,
            ) -> std::result::Result<(), MemoryError> {
                self.ledger.try_claim(tier, bytes, role)?;
                self.peak
                    .fetch_max(self.ledger.used(tier), Ordering::Relaxed);
                self.grants.lock().unwrap().push(bytes);
                Ok(())
            }
        }
        struct ObservedGovernor(Arc<ObservedBooks>);
        impl MemoryGovernor for ObservedGovernor {
            fn authority_id(&self, _tier: Tier) -> MemoryAuthorityId {
                self.0.authority_id().unwrap()
            }
            fn reserve(
                &self,
                tier: Tier,
                bytes: u64,
                role: MemoryRole,
                holder: HolderId,
            ) -> std::result::Result<MemoryLease, MemoryError> {
                self.0.try_claim(tier, bytes, role)?;
                Ok(MemoryLease::new(tier, bytes, role, holder, self.0.clone()))
            }
            fn available(&self, tier: Tier) -> u64 {
                self.0.ledger.available(tier)
            }
            fn used(&self, tier: Tier) -> u64 {
                self.0.ledger.used(tier)
            }
        }
        let _guard = hold_metrics_test_lock();
        let input = pseudo_random_values(4 * 16, 0xcafe);
        let router = pseudo_random_values(4 * 4, 0xfeed);
        const BYTES: u64 = 2 * 16 * 16 * 4;
        reset_offload_test_state(0);
        let baseline = run_mmap_qmoe_sequence_prefetch(
            false,
            false,
            &input,
            &[&router],
            3,
            false,
            test_resources(),
            0.73145,
        );
        for cache_bytes in [0, 4 * BYTES] {
            for transient_bytes in [0, BYTES - 1, BYTES, 2 * BYTES] {
                let mut previous = None;
                let mut previous_output = None;
                for prefetch in [false, true] {
                    metrics().reset();
                    let books = Arc::new(ObservedBooks {
                        ledger: LeaseLedger::new(0, cache_bytes + transient_bytes, 0),
                        peak: AtomicU64::new(0),
                        grants: Mutex::new(Vec::new()),
                    });
                    let governor = ObservedGovernor(books.clone());
                    let resources = crate::CpuMemoryResources::new(
                        DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap(),
                        crate::CpuWeightCacheLimits {
                            expert_bytes: cache_bytes,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let cache = resources.expert_cache().clone();
                    let result = run_mmap_qmoe_sequence_prefetch(
                        true,
                        false,
                        &input,
                        &[&router, &router, &router],
                        3,
                        prefetch,
                        resources,
                        0.73145,
                    );
                    assert_close(&result.output, &baseline.output);
                    assert_eq!(
                        governor.used(Tier::Host),
                        cache_bytes,
                        "every transient grant must be returned after workers and compute finish"
                    );
                    if let Some(previous) = &previous_output {
                        assert_eq!(
                            &result.output, previous,
                            "prefetch must preserve exact output under the same Host capacity"
                        );
                    }
                    previous_output = Some(result.output.clone());
                    let transient_peak = if cache_bytes != 0 {
                        0
                    } else if prefetch && transient_bytes >= 2 * BYTES {
                        2 * BYTES
                    } else if transient_bytes >= BYTES {
                        BYTES
                    } else {
                        0
                    };
                    assert_eq!(
                        books.peak.load(Ordering::Relaxed),
                        cache_bytes + transient_peak,
                        "unexpected working-set grant peak for prefetch={prefetch}"
                    );
                    if prefetch && cache_bytes != 0 {
                        assert_eq!(
                            books
                                .grants
                                .lock()
                                .unwrap()
                                .iter()
                                .filter(|&&bytes| bytes == cache_bytes)
                                .count(),
                            2,
                            "the first two cold steps split complete selected-expert grants; the third warm step must bypass the issuer"
                        );
                    }
                    let stats = crate::weight_offload_stats();
                    let policy = (
                        stats.host_cache_hits,
                        stats.host_cache_misses,
                        stats.host_cache_evictions,
                        stats.owned_host_cache_bytes,
                    );
                    if let Some(previous) = previous {
                        assert_eq!(
                            policy, previous,
                            "prefetch must preserve cache admission and eviction counters"
                        );
                    }
                    previous = Some(policy);
                    drop(cache);
                    assert_eq!(governor.used(Tier::Host), 0);
                }
            }
        }
    }

    #[test]
    fn qmoe_int4_single_block_matches_float_moe() {
        run_equivalence(4, 16, 16, 16, 1, false, false);
    }

    #[test]
    fn grouped_int4_qmoe_matches_per_token_dense_fallback_for_eight_experts_top2() {
        run_equivalence_shape(4, 16, 32, 16, 2, true, true, 8, 12);
    }

    #[test]
    fn route_first_mmap_matches_full_dequant_on_pseudorandom_inputs() {
        let _guard = hold_metrics_test_lock();
        let input = pseudo_random_values(4 * 16, 0x1234_5678);
        let router = pseudo_random_values(4 * 4, 0x9876_5432);
        reset_offload_test_state(0);
        let baseline = run_mmap_qmoe(false, false, &input, &router);
        reset_offload_test_state(0);
        let route_first = run_mmap_qmoe(true, false, &input, &router);
        assert!(route_first.catalogs_pageable);
        assert_close(&route_first.output, &baseline.output);
    }

    #[test]
    fn route_first_preserves_exact_accumulation_order_for_top_k() {
        let _guard = hold_metrics_test_lock();
        let input = pseudo_random_values(4 * 16, 0x0a11_ce55);
        let router = pseudo_random_values(4 * 4, 0x5151_5151);
        let baseline = run_mmap_qmoe_with_k(false, false, &input, &router, 3);
        reset_offload_test_state(0);
        let route_first = run_mmap_qmoe_with_k(true, false, &input, &router, 3);
        assert_eq!(route_first.output, baseline.output);
    }

    #[test]
    fn route_first_reads_only_unique_selected_expert_ranges() {
        let _guard = hold_metrics_test_lock();
        let input = pseudo_random_values(4 * 16, 0xfeed_beef);
        let router = vec![
            0.0, 5.0, 1.0, 2.0, // expert 1
            1.0, 4.0, 0.0, 3.0, // expert 1
            0.0, 1.0, 2.0, 6.0, // expert 3
            2.0, 1.0, 0.0, 7.0, // expert 3
        ];
        reset_offload_test_state(0);
        let run = run_mmap_qmoe(true, false, &input, &router);
        let stats = crate::weight_offload_stats();
        assert_eq!(
            stats.bytes_read_from_mmap as usize,
            run.selected_source_bytes
        );
        assert_eq!(stats.layer_executions, 1);
        assert_eq!(stats.active_experts, 4);
        assert_eq!(stats.unique_experts_per_batch, 2);
        assert_eq!(stats.tokens_per_expert.get(&1), Some(&2));
        assert_eq!(stats.tokens_per_expert.get(&3), Some(&2));
        let layer = stats.per_layer.values().next().expect("per-layer stats");
        assert_eq!(layer.executions, 1);
        assert_eq!(layer.active_experts, 4);
        assert_eq!(layer.unique_experts, 2);
    }

    /// Route-first dequantized-expert residency must stay *bounded* — the point
    /// of the #364 offload design is that we never resident every selected
    /// expert at once, even when all of them are routed to.
    ///
    /// The peak residency depends on which path runs:
    ///
    ///   * The serial route-first loop holds exactly the working set: one
    ///     dequantized expert is materialized, computed, and dropped before the
    ///     next is leased, so the peak is deterministically `WORKING_SET`.
    ///   * The single-ahead prefetch pipeline (engaged at `budget == 0`) holds
    ///     the current expert's lease while the worker leases and dequantizes
    ///     the next one, so residency may transiently reach
    ///     `WORKING_SET + IN_FLIGHT_PREFETCH_DEPTH`. Whether it actually does is
    ///     a *timing* question: if the prefetch overlaps compute the peak is 2,
    ///     otherwise it stays 1. Under coverage instrumentation the overlap is
    ///     far more likely, which is exactly why a fragile `== 1` assertion
    ///     flaked in CI while passing locally.
    ///
    /// We therefore assert the honest invariant — residency is bounded to the
    /// working set plus at most one in-flight prefetch, and never grows to hold
    /// all selected experts — instead of an exact peak that is nondeterministic
    /// by design.
    #[test]
    fn route_first_bounds_dequantized_residency_when_all_experts_are_selected() {
        /// Route-first computes one expert at a time, so the working set is a
        /// single dequantized expert.
        const WORKING_SET: u64 = 1;
        /// The prefetch pipeline keeps a single lease in flight (single-ahead),
        /// so it can add at most one more resident expert on top of the working
        /// set.
        const IN_FLIGHT_PREFETCH_DEPTH: u64 = 1;
        /// Upper bound on concurrent dequantized-expert residency for the
        /// route-first path, whether or not the prefetch overlaps compute.
        const RESIDENCY_BOUND: u64 = WORKING_SET + IN_FLIGHT_PREFETCH_DEPTH;
        /// Every one of the four experts is routed to by this fixture.
        const SELECTED_EXPERTS: u64 = 4;

        let _guard = hold_metrics_test_lock();
        let input = pseudo_random_values(4 * 16, 0x1234);
        let router = vec![
            9.0, 0.0, 0.0, 0.0, // expert 0
            0.0, 9.0, 0.0, 0.0, // expert 1
            0.0, 0.0, 9.0, 0.0, // expert 2
            0.0, 0.0, 0.0, 9.0, // expert 3
        ];
        let baseline = run_mmap_qmoe(false, false, &input, &router);

        // Serial route-first (prefetch disabled): with no prefetch ever in
        // flight the peak is deterministically the working set. This pins the
        // lower edge of the bound and proves the accounting itself is correct.
        reset_offload_test_state(0);
        let serial = run_mmap_qmoe_sequence_prefetch(
            true,
            false,
            &input,
            &[&router],
            1,
            false,
            test_resources(),
            1.0,
        );
        let serial_stats = crate::weight_offload_stats();
        assert_close(&serial.output, &baseline.output);
        assert_eq!(serial_stats.unique_experts_per_batch, 4);
        assert_eq!(serial_stats.peak_dequantized_experts, WORKING_SET);

        // Single-ahead prefetch pipeline (engaged at `budget == 0`): residency
        // may reach `WORKING_SET + IN_FLIGHT_PREFETCH_DEPTH` when the prefetch
        // overlaps compute, but must never exceed it and must never resident
        // every selected expert. Asserting the bound — not an exact peak — is
        // stable regardless of scheduling, which is what makes this test
        // deterministic under coverage-instrumented timing.
        reset_offload_test_state(0);
        let route_first = run_mmap_qmoe(true, false, &input, &router);
        let stats = crate::weight_offload_stats();
        assert_close(&route_first.output, &baseline.output);
        assert_eq!(stats.unique_experts_per_batch, 4);
        assert!(
            (WORKING_SET..=RESIDENCY_BOUND).contains(&stats.peak_dequantized_experts),
            "route-first peak dequantized residency {} escaped the expected \
             bound {WORKING_SET}..={RESIDENCY_BOUND}",
            stats.peak_dequantized_experts,
        );
        assert!(
            stats.peak_dequantized_experts < SELECTED_EXPERTS,
            "route-first must bound residency, not resident all {SELECTED_EXPERTS} \
             selected experts at once (peak was {})",
            stats.peak_dequantized_experts,
        );
    }

    #[test]
    fn interleaved_expert_layout_falls_back_without_error() {
        let _guard = hold_metrics_test_lock();
        let input = pseudo_random_values(4 * 16, 0x55aa);
        let router = vec![
            0.0, 5.0, 1.0, 2.0, 1.0, 4.0, 0.0, 3.0, 0.0, 1.0, 2.0, 6.0, 2.0, 1.0, 0.0, 7.0,
        ];
        let baseline = run_mmap_qmoe(false, false, &input, &router);
        reset_offload_test_state(0);
        let fallback = run_mmap_qmoe(true, true, &input, &router);
        assert!(!fallback.catalogs_pageable);
        assert_close(&fallback.output, &baseline.output);
        assert_eq!(crate::weight_offload_stats().bytes_read_from_mmap, 0);
    }

    #[test]
    fn flag_off_preserves_legacy_path_and_counters() {
        let _guard = hold_metrics_test_lock();
        let input = pseudo_random_values(4 * 16, 7);
        let router = pseudo_random_values(4 * 4, 11);
        reset_offload_test_state(0);
        let legacy = run_mmap_qmoe(false, false, &input, &router);
        let stats = crate::weight_offload_stats();
        assert_eq!(stats.bytes_read_from_mmap, 0);
        assert_eq!(stats.layer_executions, 0);

        reset_offload_test_state(0);
        let route_first = run_mmap_qmoe(true, false, &input, &router);
        assert_close(&legacy.output, &route_first.output);
    }

    #[test]
    fn zero_byte_host_cache_matches_phase1_direct_mmap() {
        let _guard = hold_metrics_test_lock();
        let input = pseudo_random_values(4 * 16, 0x0bad_f00d);
        let router = vec![
            0.0, 5.0, 1.0, 2.0, 1.0, 4.0, 0.0, 3.0, 0.0, 1.0, 2.0, 6.0, 2.0, 1.0, 0.0, 7.0,
        ];
        reset_offload_test_state(0);
        let direct = run_mmap_qmoe(true, false, &input, &router);
        reset_offload_test_state(0);
        let repeated = run_mmap_qmoe_sequence(true, false, &input, &[&router, &router, &router], 1);
        let stats = crate::weight_offload_stats();
        assert_eq!(repeated.output, direct.output);
        assert_eq!(stats.host_cache_hits, 0);
        assert_eq!(stats.host_cache_misses, 6);
        assert_eq!(stats.owned_host_cache_bytes, 0);
        assert_eq!(
            stats.bytes_read_from_mmap as usize,
            repeated.selected_source_bytes * 3
        );
    }

    #[test]
    fn host_cache_enforces_expanded_byte_cap_for_oversubscribed_working_set() {
        const EXPANDED_EXPERT_BYTES: u64 = 2 * 16 * 16 * 4;
        let _guard = hold_metrics_test_lock();
        let input = pseudo_random_values(4 * 16, 0xcafe_babe);
        let router = vec![
            9.0, 0.0, 0.0, 0.0, // expert 0
            0.0, 9.0, 0.0, 0.0, // expert 1
            0.0, 0.0, 9.0, 0.0, // expert 2
            0.0, 0.0, 0.0, 9.0, // expert 3
        ];
        reset_offload_test_state(0);
        let direct = run_mmap_qmoe(true, false, &input, &router);
        reset_offload_test_state(EXPANDED_EXPERT_BYTES);
        let cached = run_mmap_qmoe_sequence(
            true,
            false,
            &input,
            &[&router, &router, &router, &router],
            1,
        );
        let stats = crate::weight_offload_stats();
        assert_eq!(cached.output, direct.output);
        assert!(stats.peak_owned_host_cache_bytes <= EXPANDED_EXPERT_BYTES);
        assert!(stats.owned_host_cache_bytes <= EXPANDED_EXPERT_BYTES);
        assert_eq!(stats.host_cache_budget_bytes, EXPANDED_EXPERT_BYTES);
    }

    #[test]
    fn repeated_routing_working_set_converges_to_host_cache_hits() {
        const TWO_EXPANDED_EXPERTS: u64 = 2 * 2 * 16 * 16 * 4;
        let _guard = hold_metrics_test_lock();
        let input = pseudo_random_values(4 * 16, 0x1234_abcd);
        let router = vec![
            0.0, 5.0, 1.0, 2.0, 1.0, 4.0, 0.0, 3.0, 0.0, 1.0, 2.0, 6.0, 2.0, 1.0, 0.0, 7.0,
        ];
        reset_offload_test_state(TWO_EXPANDED_EXPERTS);
        let run = run_mmap_qmoe_sequence(
            true,
            false,
            &input,
            &[&router, &router, &router, &router, &router],
            1,
        );
        let stats = crate::weight_offload_stats();
        assert_eq!(stats.host_cache_hits, 6);
        assert_eq!(stats.host_cache_misses, 4);
        assert_eq!(stats.host_cache_evictions, 0);
        assert_eq!(stats.owned_host_cache_bytes, TWO_EXPANDED_EXPERTS);
        assert!(
            stats.owned_host_cache_bytes as usize > run.selected_source_bytes,
            "cache accounting must charge expanded f32 bytes, not compressed source bytes"
        );
    }

    #[test]
    fn one_off_rare_route_does_not_evict_pinned_hot_expert() {
        const EXPANDED_EXPERT_BYTES: u64 = 2 * 16 * 16 * 4;
        let _guard = hold_metrics_test_lock();
        let input = pseudo_random_values(4 * 16, 0x55aa_aa55);
        let hot = vec![
            0.0, 9.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0,
        ];
        let rare = vec![
            0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 9.0,
        ];
        reset_offload_test_state(0);
        let direct = run_mmap_qmoe(true, false, &input, &hot);
        reset_offload_test_state(EXPANDED_EXPERT_BYTES);
        let cached = run_mmap_qmoe_sequence(
            true,
            false,
            &input,
            &[&hot, &hot, &hot, &hot, &rare, &hot],
            1,
        );
        let stats = crate::weight_offload_stats();
        assert_eq!(cached.output, direct.output);
        assert_eq!(stats.host_cache_hits, 3);
        assert_eq!(stats.host_cache_evictions, 0);
        assert_eq!(stats.owned_host_cache_bytes, EXPANDED_EXPERT_BYTES);
    }

    #[test]
    fn cached_and_uncached_routes_and_logits_are_bit_identical() {
        const TWO_EXPANDED_EXPERTS: u64 = 2 * 2 * 16 * 16 * 4;
        let _guard = hold_metrics_test_lock();
        let input = pseudo_random_values(4 * 16, 0xdec0_de01);
        let router = pseudo_random_values(4 * 4, 0xface_feed);
        reset_offload_test_state(0);
        let uncached = run_mmap_qmoe(true, false, &input, &router);
        reset_offload_test_state(TWO_EXPANDED_EXPERTS);
        let cached = run_mmap_qmoe_sequence(true, false, &input, &[&router, &router, &router], 1);
        assert_eq!(cached.output, uncached.output);
        assert!(crate::weight_offload_stats().host_cache_hits > 0);
    }

    #[test]
    fn two_engine_cache_partitions_respect_independent_budgets() {
        let _guard = hold_metrics_test_lock();
        metrics().reset();
        let engine_a = WeightOffloadHostCache::new(4, DerivedWeightMemory::default()).unwrap();
        let engine_b = WeightOffloadHostCache::new(8, DerivedWeightMemory::default()).unwrap();
        let weights = |value| DequantizedExpert {
            fc1: vec![value],
            fc2: Vec::new(),
            fc3: None,
            _residency: None,
            _grant: None,
        };

        for cache in [&engine_a, &engine_b] {
            drop(cache.lease(cache_key(0), 4, || Ok(weights(1.0))).unwrap());
            drop(cache.lease(cache_key(0), 4, || Ok(weights(1.0))).unwrap());
            drop(cache.lease(cache_key(1), 4, || Ok(weights(2.0))).unwrap());
            drop(cache.lease(cache_key(1), 4, || Ok(weights(2.0))).unwrap());
        }

        assert_eq!(engine_a.retention_budget_bytes(), 4);
        assert_eq!(engine_b.retention_budget_bytes(), 8);
        let cache_a = engine_a.inner.cache.lock().unwrap();
        let cache_b = engine_b.inner.cache.lock().unwrap();
        assert!(cache_a.owned_bytes <= 4);
        assert!(cache_b.owned_bytes <= 8);
        assert_eq!(cache_a.entries.len(), 1);
        assert_eq!(cache_b.entries.len(), 2);
    }

    fn lease_fixture_expert(
        operation: &HostCacheOperation<'_>,
        expert: usize,
    ) -> Result<ExpertLease> {
        operation.lease(
            expert,
            &ExpertMemoryScope::OnDemand(&operation.cache.memory),
            |grant| {
                let mut payload = expert_payload(1.0);
                payload._grant = grant;
                Ok(payload)
            },
        )
    }

    fn expert_payload(value: f32) -> DequantizedExpert {
        DequantizedExpert {
            fc1: vec![value],
            fc2: vec![],
            fc3: None,
            _residency: None,
            _grant: None,
        }
    }

    #[test]
    fn retained_grant_callbacks_observe_unlocked_payloads_and_hard_failures_are_transactional() {
        use onnx_runtime_memory_api::{MemoryError, MemoryRole, Tier};
        use onnx_runtime_memory_governor::{
            HolderId, LeaseAccounting, LeaseLedger, MemoryAuthorityId, MemoryGovernor, MemoryLease,
        };
        use std::sync::atomic::AtomicBool;
        #[derive(Debug)]
        struct Books {
            ledger: Arc<LeaseLedger>,
            cache: Mutex<Weak<WeightOffloadHostCacheInner>>,
            payload: Mutex<Weak<DequantizedExpert>>,
            callbacks_unlocked: AtomicBool,
            released_while_live: AtomicBool,
            reject: AtomicBool,
            grants: AtomicU64,
            refunds: AtomicU64,
        }
        impl Books {
            fn observe(&self) {
                if let Some(cache) = self.cache.lock().unwrap().upgrade() {
                    self.callbacks_unlocked
                        .fetch_and(cache.cache.try_lock().is_ok(), Ordering::Relaxed);
                }
            }
        }
        impl LeaseAccounting for Books {
            fn authority_id(&self) -> Option<MemoryAuthorityId> {
                self.ledger.authority_id()
            }
            fn try_claim(
                &self,
                tier: Tier,
                bytes: u64,
                role: MemoryRole,
            ) -> std::result::Result<(), MemoryError> {
                if bytes != 0 {
                    self.observe();
                    if self.reject.load(Ordering::Relaxed) {
                        return Err(MemoryError::InvalidRequest {
                            tier: tier.name(),
                            requested: bytes,
                            reason: "fixture rejects this allocator request; restore the source before retrying",
                        });
                    }
                    self.grants.fetch_add(1, Ordering::Relaxed);
                }
                self.ledger.try_claim(tier, bytes, role)
            }
            fn release(&self, tier: Tier, bytes: u64) {
                if bytes != 0 {
                    self.observe();
                    self.released_while_live.fetch_or(
                        self.payload.lock().unwrap().strong_count() != 0,
                        Ordering::Relaxed,
                    );
                    self.refunds.fetch_add(1, Ordering::Relaxed);
                }
                self.ledger.release(tier, bytes);
            }
        }
        struct Governor(Arc<Books>);
        impl MemoryGovernor for Governor {
            fn authority_id(&self, _tier: Tier) -> MemoryAuthorityId {
                self.0.authority_id().unwrap()
            }
            fn reserve(
                &self,
                tier: Tier,
                bytes: u64,
                role: MemoryRole,
                holder: HolderId,
            ) -> std::result::Result<MemoryLease, MemoryError> {
                self.0.try_claim(tier, bytes, role)?;
                Ok(MemoryLease::new(tier, bytes, role, holder, self.0.clone()))
            }
            fn available(&self, tier: Tier) -> u64 {
                self.0.ledger.available(tier)
            }
            fn used(&self, tier: Tier) -> u64 {
                self.0.ledger.used(tier)
            }
        }
        let _guard = hold_metrics_test_lock();
        let books = Arc::new(Books {
            ledger: LeaseLedger::new(0, 4, 0),
            cache: Mutex::new(Weak::new()),
            payload: Mutex::new(Weak::new()),
            callbacks_unlocked: AtomicBool::new(true),
            released_while_live: AtomicBool::new(false),
            reject: AtomicBool::new(false),
            grants: AtomicU64::new(0),
            refunds: AtomicU64::new(0),
        });
        let governor = Governor(books.clone());
        let cache = WeightOffloadHostCache::new(
            4,
            DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap(),
        )
        .unwrap();
        *books.cache.lock().unwrap() = Arc::downgrade(&cache.inner);
        for _ in 0..2 {
            drop(
                cache
                    .lease(cache_key(0), 4, || Ok(expert_payload(1.)))
                    .unwrap(),
            );
        }
        *books.payload.lock().unwrap() =
            Arc::downgrade(&cache.inner.cache.lock().unwrap().entries[&cache_key(0)].weights);
        // The third cold reference is frequent enough to evict the first entry.
        for _ in 0..3 {
            drop(
                cache
                    .lease(cache_key(1), 4, || Ok(expert_payload(2.)))
                    .unwrap(),
            );
        }
        assert_eq!(governor.used(Tier::Host), 4);
        assert!(
            cache
                .inner
                .cache
                .lock()
                .unwrap()
                .entries
                .contains_key(&cache_key(1))
        );
        *books.payload.lock().unwrap() =
            Arc::downgrade(&cache.inner.cache.lock().unwrap().entries[&cache_key(1)].weights);
        cache.clear();
        assert_eq!(governor.used(Tier::Host), 0);
        assert_eq!(books.grants.load(Ordering::Relaxed), 2);
        assert_eq!(books.refunds.load(Ordering::Relaxed), 2);
        assert!(
            books.callbacks_unlocked.load(Ordering::Relaxed),
            "reserve/refund callbacks must never run under the payload lock"
        );
        assert!(!books.released_while_live.load(Ordering::Relaxed));
        books.reject.store(true, Ordering::Relaxed);
        drop(
            cache
                .lease(cache_key(2), 4, || Ok(expert_payload(3.)))
                .unwrap(),
        );
        let failure = cache
            .lease(cache_key(2), 4, || Ok(expert_payload(3.)))
            .err()
            .unwrap();
        assert!(matches!(
            failure,
            EpError::Memory(MemoryError::InvalidRequest { .. })
        ));
        assert_eq!(cache.inner.cache.lock().unwrap().owned_bytes, 0);
        assert_eq!(governor.used(Tier::Host), 0);
        books.reject.store(false, Ordering::Relaxed);
        assert!(matches!(
            cache
                .lease(cache_key(2), 4, || Ok(expert_payload(3.)))
                .unwrap(),
            ExpertLease::Filled(_)
        ));
        cache.clear();
        assert_eq!(governor.used(Tier::Host), 0);
        for _ in 0..2 {
            drop(
                cache
                    .lease(cache_key(3), 4, || Ok(expert_payload(4.)))
                    .unwrap(),
            );
        }
        // Fault injection pins the retirement error path: even an inconsistent
        // local counter must return payloads outside the lock, never refund
        // through an observer while still holding that lock.
        cache.inner.cache.lock().unwrap().owned_bytes = 1;
        for _ in 0..2 {
            drop(
                cache
                    .lease(cache_key(4), 4, || Ok(expert_payload(5.)))
                    .unwrap(),
            );
        }
        let failure = cache
            .lease(cache_key(4), 4, || Ok(expert_payload(5.)))
            .err()
            .unwrap();
        assert!(failure.to_string().contains("accounting underflow"));
        assert_eq!(governor.used(Tier::Host), 0);
        assert!(books.callbacks_unlocked.load(Ordering::Relaxed));
        cache.clear();
    }

    #[test]
    fn concurrent_cold_references_publish_one_payload_and_one_grant() {
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        use std::sync::{Barrier, atomic::AtomicUsize};
        let _guard = hold_metrics_test_lock();
        let governor = LedgerGovernor::new(LeaseLedger::new(0, 4, 0));
        let cache = WeightOffloadHostCache::new(
            64,
            DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap(),
        )
        .unwrap();
        drop(
            cache
                .lease(cache_key(0), 4, || Ok(expert_payload(1.)))
                .unwrap(),
        );
        let loads = AtomicUsize::new(0);
        let start = Barrier::new(9);
        std::thread::scope(|scope| {
            let workers = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        let held = cache
                            .lease(cache_key(0), 4, || {
                                loads.fetch_add(1, Ordering::Relaxed);
                                Ok(expert_payload(1.))
                            })
                            .unwrap();
                        assert!(matches!(held, ExpertLease::Hit(_) | ExpertLease::Filled(_)));
                    })
                })
                .collect::<Vec<_>>();
            start.wait();
            for worker in workers {
                worker.join().unwrap();
            }
        });
        assert_eq!(loads.load(Ordering::Relaxed), 1);
        assert_eq!(governor.used(Tier::Host), 4);
        assert_eq!(cache.inner.cache.lock().unwrap().owned_bytes, 4);
        cache.clear();
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[test]
    fn governed_retention_reserves_only_live_payloads_and_active_arcs_keep_grants() {
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let _guard = hold_metrics_test_lock();
        let governor = LedgerGovernor::new(LeaseLedger::new(0, 8, 0));
        let cache = WeightOffloadHostCache::new(
            64,
            DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap(),
        )
        .unwrap();
        assert_eq!(cache.retention_budget_bytes(), 64);
        assert_eq!(governor.used(Tier::Host), 0);
        drop(
            cache
                .lease(cache_key(0), 4, || Ok(expert_payload(1.)))
                .unwrap(),
        );
        assert_eq!(
            governor.used(Tier::Host),
            0,
            "history alone allocates no retained payload"
        );
        let held = cache
            .lease(cache_key(0), 4, || Ok(expert_payload(1.)))
            .unwrap();
        assert_eq!(governor.used(Tier::Host), 4);
        drop(cache);
        assert_eq!(
            governor.used(Tier::Host),
            4,
            "an active expert outlives its cache partition"
        );
        drop(held);
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[test]
    fn shared_pressure_expert_and_dense_caches_reclaim_each_others_idle_payloads() {
        use super::super::dense_weight_cache::{
            DenseWeightCache, DenseWeightKey, DenseWeightLoad, DenseWeightSpec,
        };
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        struct Key;
        impl DenseWeightKey for Key {
            fn dense_bytes(&self) -> Result<usize> {
                Ok(4)
            }
        }
        let _guard = hold_metrics_test_lock();
        let governor = LedgerGovernor::new(LeaseLedger::new(0, 4, 0));
        let source = DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap();
        let experts = WeightOffloadHostCache::new(64, source.clone()).unwrap();
        let dense = DenseWeightCache::<Key>::new(source, 4);
        for _ in 0..4 {
            drop(
                experts
                    .lease(cache_key(0), 4, || Ok(expert_payload(2.0)))
                    .unwrap(),
            );
        }
        // Frequency pinning cannot keep an inactive payload physically resident
        // when the shared Host books refuse another weight allocation.
        assert_eq!(experts.inner.cache.lock().unwrap().owned_bytes, 4);
        let loaded = dense
            .load_group(&[DenseWeightSpec::cached(&Key).unwrap()], |_, row| {
                row.fill(3.0);
                Ok(())
            })
            .unwrap();
        let DenseWeightLoad::Ready(loaded) = loaded else {
            panic!("idle expert must fund dense payload");
        };
        assert_eq!(experts.inner.cache.lock().unwrap().owned_bytes, 0);
        assert_eq!(governor.used(Tier::Host), 4);
        assert!(matches!(
            experts
                .lease(cache_key(0), 4, || Ok(expert_payload(2.0)))
                .unwrap(),
            ExpertLease::Transient
        ));
        assert!(matches!(
            experts
                .lease(cache_key(0), 4, || Ok(expert_payload(2.0)))
                .unwrap(),
            ExpertLease::Refused(_)
        ));
        assert_eq!(loaded.weights[0].as_slice(), &[3.0]);
        drop(loaded);
        let held = experts
            .lease(cache_key(0), 4, || Ok(expert_payload(2.0)))
            .unwrap();
        assert!(matches!(held, ExpertLease::Filled(_)));
        assert_eq!(dense.stats(), (0, 0));
        assert_eq!(experts.inner.cache.lock().unwrap().owned_bytes, 4);
        let before = crate::weight_offload_stats();
        assert!(matches!(
            dense
                .load_group(&[DenseWeightSpec::transient(4)], |_, _| panic!(
                    "active expert cannot be reclaimed"
                ))
                .unwrap(),
            DenseWeightLoad::Declined(_)
        ));
        assert_eq!(
            crate::weight_offload_stats().host_cache_evictions,
            before.host_cache_evictions
        );
        experts.clear();
        assert_eq!(governor.used(Tier::Host), 4);
        drop(held);
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[test]
    fn governed_retention_competes_with_real_host_kv_without_idle_entitlement() {
        use onnx_runtime_memory_api::{MemoryRole, Tier};
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let _guard = hold_metrics_test_lock();
        let governor = LedgerGovernor::new(LeaseLedger::new(0, 12, 0));
        let kv = governor
            .reserve(Tier::Host, 8, MemoryRole::KvCache, HolderId::new(1))
            .unwrap();
        let cache = WeightOffloadHostCache::new(
            64,
            DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap(),
        )
        .unwrap();
        assert_eq!(governor.used(Tier::Host), 8);
        for _ in 0..2 {
            drop(
                cache
                    .lease(cache_key(0), 4, || Ok(expert_payload(1.)))
                    .unwrap(),
            );
        }
        assert_eq!(governor.used(Tier::Host), 12);
        let active = cache.lease(cache_key(0), 4, || unreachable!()).unwrap();
        drop(
            cache
                .lease(cache_key(1), 4, || Ok(expert_payload(2.)))
                .unwrap(),
        );
        assert!(matches!(
            cache
                .lease(cache_key(1), 4, || Ok(expert_payload(2.)))
                .unwrap(),
            ExpertLease::Refused(_)
        ));
        assert_eq!(cache.inner.cache.lock().unwrap().owned_bytes, 4);
        drop(active);
        assert!(matches!(
            cache
                .lease(cache_key(1), 4, || Ok(expert_payload(2.)))
                .unwrap(),
            ExpertLease::Filled(_)
        ));
        assert_eq!(governor.used(Tier::Host), 12);
        drop(kv);
        assert_eq!(governor.used(Tier::Host), 4);
        cache.clear();
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[test]
    fn selected_prefetch_grants_refund_each_retained_payload_independently() {
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let _guard = hold_metrics_test_lock();
        let governor = LedgerGovernor::new(LeaseLedger::new(0, 12, 0));
        let cache = WeightOffloadHostCache::new(
            12,
            DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap(),
        )
        .unwrap();
        let operation = cache.begin_operation(cache_key(0), 3, 4).unwrap();
        for expert in 0..3 {
            drop(
                operation
                    .lease(
                        expert,
                        &ExpertMemoryScope::OnDemand(cache.memory()),
                        |grant| {
                            let mut payload = expert_payload(expert as f32);
                            payload._grant = grant;
                            Ok(payload)
                        },
                    )
                    .unwrap(),
            );
        }
        let scope = operation.prefetch_memory(0..3).unwrap().unwrap();
        assert_eq!(governor.used(Tier::Host), 12);
        let mut active = None;
        for expert in 0..3 {
            let held = operation
                .lease(expert, &scope, |grant| {
                    let mut payload = expert_payload(expert as f32);
                    payload._grant = grant;
                    Ok(payload)
                })
                .unwrap();
            assert!(matches!(held, ExpertLease::Filled(_)));
            if expert == 0 {
                active = Some(held);
            }
        }
        drop(scope);
        assert_eq!(governor.used(Tier::Host), 12);
        cache.clear();
        assert_eq!(
            governor.used(Tier::Host),
            4,
            "two evicted payloads must not keep a shared full-group allowance alive"
        );
        drop(active);
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[test]
    fn process_metrics_aggregate_live_cache_residency_and_release_on_drop() {
        let _guard = hold_metrics_test_lock();
        metrics().reset();
        let engine_a = WeightOffloadHostCache::new(4, DerivedWeightMemory::default()).unwrap();
        let engine_b = WeightOffloadHostCache::new(8, DerivedWeightMemory::default()).unwrap();
        let weights = |value| DequantizedExpert {
            fc1: vec![value],
            fc2: Vec::new(),
            fc3: None,
            _residency: None,
            _grant: None,
        };

        for expert in 0..2 {
            for _ in 0..2 {
                drop(
                    engine_a
                        .lease(cache_key(expert), 4, || Ok(weights(expert as f32)))
                        .unwrap(),
                );
            }
        }
        for expert in 0..4 {
            for _ in 0..2 {
                drop(
                    engine_b
                        .lease(cache_key(expert), 4, || Ok(weights(expert as f32)))
                        .unwrap(),
                );
            }
        }

        let stats = crate::weight_offload_stats();
        assert_eq!(stats.owned_host_cache_bytes, 12);
        assert_eq!(stats.peak_owned_host_cache_bytes, 12);
        assert_eq!(stats.host_cache_budget_bytes, 12);

        drop(engine_a);
        let stats = crate::weight_offload_stats();
        assert_eq!(stats.owned_host_cache_bytes, 8);
        assert_eq!(stats.host_cache_budget_bytes, 8);
    }

    #[test]
    fn admission_history_skips_uncacheable_keys_and_expires_churn() {
        let _guard = hold_metrics_test_lock();
        metrics().reset();
        let mut cache = HostExpertCache::default();
        let weights = || DequantizedExpert {
            fc1: vec![1.0],
            fc2: Vec::new(),
            fc3: None,
            _residency: None,
            _grant: None,
        };

        drop(cache.lease(cache_key(0), 4, 0, || Ok(weights())).unwrap());
        drop(cache.lease(cache_key(1), 8, 4, || Ok(weights())).unwrap());
        assert!(cache.history.is_empty());

        for expert in 0..MAX_ADMISSION_HISTORY_ENTRIES + 32 {
            drop(
                cache
                    .lease(cache_key(expert), 4, 4, || Ok(weights()))
                    .unwrap(),
            );
        }
        assert_eq!(cache.history.len(), MAX_ADMISSION_HISTORY_ENTRIES);

        cache.clock = cache
            .clock
            .checked_add(HISTORY_EXPIRY_WINDOW)
            .and_then(|clock| clock.checked_add(1))
            .unwrap();
        drop(
            cache
                .lease(cache_key(MAX_ADMISSION_HISTORY_ENTRIES + 100), 4, 4, || {
                    Ok(weights())
                })
                .unwrap(),
        );
        assert_eq!(cache.history.len(), 1);
    }

    #[test]
    fn shared_bank_rejects_inconsistent_expanded_size_with_source_context() {
        let _guard = hold_metrics_test_lock();
        metrics().reset();
        let cache = WeightOffloadHostCache::new(8, DerivedWeightMemory::default()).unwrap();
        {
            let operation = cache.begin_operation(cache_key(0), 1, 4).unwrap();
            for _ in 0..2 {
                drop(lease_fixture_expert(&operation, 0).unwrap());
            }
        }
        let failure = cache
            .begin_operation(cache_key(0), 1, 8)
            .err()
            .expect("reject inconsistent source metadata");
        let message = failure.to_string();
        assert!(message.contains("layer 7"), "{message}");
        assert!(message.contains("from 4 to 8"), "{message}");
        assert!(message.contains("distinct source layout"), "{message}");
    }

    #[test]
    fn zero_budget_prefetch_needs_no_bank_registry_lock() {
        let _guard = hold_metrics_test_lock();
        let cache = WeightOffloadHostCache::new(0, DerivedWeightMemory::default()).unwrap();
        let first = cache.begin_operation(cache_key(0), 2, 4).unwrap();
        assert!(first.prefetch_eligibility().allows_pipeline());
        let registry = cache
            .inner
            .banks
            .try_write()
            .expect("streaming must not lock the registry");
        assert!(registry.is_empty());
        let mut foreign = cache_key(0);
        foreign.layer_id = 99;
        let second = cache.begin_operation(foreign, 2, 4).unwrap();
        assert!(second.prefetch_eligibility().allows_pipeline());
        assert_eq!(second.registered_bank_count(), 0);
    }

    #[test]
    fn shared_bank_prefetch_proof_counts_retained_foreign_payloads() {
        let _guard = hold_metrics_test_lock();
        metrics().reset();
        for (budget, expected) in [(0, true), (8, false), (12, true)] {
            let cache =
                WeightOffloadHostCache::new(budget, DerivedWeightMemory::default()).unwrap();
            let mut foreign = cache_key(0);
            foreign.layer_id = 99;
            {
                let operation = cache.begin_operation(foreign, 1, 4).unwrap();
                for _ in 0..2 {
                    drop(lease_fixture_expert(&operation, 0).unwrap());
                }
            }
            let operation = cache.begin_operation(cache_key(0), 2, 4).unwrap();
            assert_eq!(
                operation.prefetch_eligibility().allows_pipeline(),
                expected,
                "budget={budget}"
            );
            if budget == 8 {
                assert_eq!(
                    operation.prefetch_eligibility(),
                    HostPrefetchEligibility::InsufficientCapacity {
                        required_bytes: 12,
                        budget_bytes: 8,
                    }
                );
            }
            assert_eq!(
                operation.registered_bank_count(),
                if budget == 0 { 0 } else { 2 }
            );
        }
    }

    #[test]
    fn shared_bank_prefetch_proof_freezes_new_registrations_but_not_worker_leases() {
        use std::sync::mpsc;
        use std::time::Duration;
        let _guard = hold_metrics_test_lock();
        metrics().reset();
        let cache = WeightOffloadHostCache::new(8, DerivedWeightMemory::default()).unwrap();
        let operation = cache.begin_operation(cache_key(0), 2, 4).unwrap();
        assert!(operation.prefetch_eligibility().allows_pipeline());
        assert!(cache.inner.banks.try_write().is_err());
        let timeout = Duration::from_secs(5);
        std::thread::scope(|scope| {
            // Known-bank operations share the proof; they do not serialize on
            // a global operation mutex.
            let (known_tx, known_rx) = mpsc::channel();
            let known_cache = &cache;
            scope.spawn(move || {
                let known = known_cache.begin_operation(cache_key(1), 2, 4).unwrap();
                known_tx
                    .send(known.prefetch_eligibility().allows_pipeline())
                    .unwrap();
            });
            assert!(known_rx.recv_timeout(timeout).unwrap());
            let (started_tx, started_rx) = mpsc::channel();
            let (registered_tx, registered_rx) = mpsc::channel();
            let cache_ref = &cache;
            scope.spawn(move || {
                let mut foreign = cache_key(0);
                foreign.layer_id = 99;
                started_tx.send(()).unwrap();
                let foreign = cache_ref.begin_operation(foreign, 1, 4).unwrap();
                registered_tx
                    .send(foreign.prefetch_eligibility().allows_pipeline())
                    .unwrap();
            });
            started_rx.recv_timeout(timeout).unwrap();
            let (worker_tx, worker_rx) = mpsc::channel();
            std::thread::scope(|workers| {
                let borrowed = &operation;
                workers.spawn(move || {
                    for _ in 0..2 {
                        drop(lease_fixture_expert(borrowed, 0).unwrap());
                    }
                    worker_tx.send(()).unwrap();
                });
                worker_rx.recv_timeout(timeout).expect(
                    "borrowed worker must not reacquire the registry behind a pending writer",
                );
            });
            assert!(matches!(
                registered_rx.try_recv(),
                Err(mpsc::TryRecvError::Empty)
            ));
            drop(operation);
            assert!(
                !registered_rx.recv_timeout(timeout).unwrap(),
                "the retained original bank must participate after registration"
            );
        });
    }

    #[test]
    fn shared_bank_registry_prunes_dead_sources_and_retains_live_entries() {
        let _guard = hold_metrics_test_lock();
        metrics().reset();
        let cache = WeightOffloadHostCache::new(8, DerivedWeightMemory::default()).unwrap();
        {
            let retained = cache.begin_operation(cache_key(0), 1, 4).unwrap();
            for _ in 0..2 {
                drop(lease_fixture_expert(&retained, 0).unwrap());
            }
        }
        for layer_id in 100..132 {
            let mut key = cache_key(0);
            key.layer_id = layer_id;
            let operation = cache.begin_operation(key, 1, 4).unwrap();
            assert_eq!(operation.registered_bank_count(), 2);
            assert!(operation.prefetch_eligibility().allows_pipeline());
        }
        // Clearing payloads also ends their metadata lifetime. The next cold
        // registration removes all expired weak entries.
        cache.clear();
        let mut key = cache_key(0);
        key.layer_id = 200;
        let operation = cache.begin_operation(key, 1, 4).unwrap();
        assert_eq!(operation.registered_bank_count(), 1);
    }

    #[test]
    fn foreign_bank_invalidates_layer_only_prefetch_capacity_proof() {
        let _guard = hold_metrics_test_lock();
        metrics().reset();
        let run = |hold_current: bool| {
            let mut cache = HostExpertCache::default();
            let weights = || {
                Ok(DequantizedExpert {
                    fc1: vec![1.0],
                    fc2: Vec::new(),
                    fc3: None,
                    _residency: None,
                    _grant: None,
                })
            };
            let mut foreign = cache_key(0);
            foreign.layer_id = 99;
            let mut disposable = cache_key(0);
            disposable.layer_id = 100;
            for _ in 0..2 {
                drop(cache.lease(disposable.clone(), 4, 8, weights).unwrap());
            }
            for _ in 0..4 {
                drop(cache.lease(foreign.clone(), 4, 8, weights).unwrap());
            }
            drop(cache.lease(cache_key(0), 4, 8, weights).unwrap());
            for _ in 0..2 {
                drop(cache.lease(cache_key(1), 4, 8, weights).unwrap());
            }
            cache.trim_to_budget(4).unwrap();
            assert!(cache.entries.contains_key(&foreign));
            let current = cache.lease(cache_key(0), 4, 8, weights).unwrap();
            assert!(cache.entries.contains_key(&cache_key(0)));
            let current = if hold_current {
                Some(current)
            } else {
                drop(current);
                None
            };
            drop(cache.lease(cache_key(1), 4, 8, weights).unwrap());
            let admitted = cache.entries.contains_key(&cache_key(1));
            drop(current);
            admitted
        };
        assert!(run(false), "serial admits the next expert");
        assert!(!run(true), "prefetch's live Arc prevents the serial victim");
    }

    #[test]
    fn active_host_cache_lease_blocks_eviction_until_drop() {
        let _guard = hold_metrics_test_lock();
        reset_offload_test_state(4);
        let weights = |value| DequantizedExpert {
            fc1: vec![value],
            fc2: Vec::new(),
            fc3: None,
            _residency: None,
            _grant: None,
        };
        let mut cache = HostExpertCache::default();

        drop(
            cache
                .lease(cache_key(0), 4, 4, || Ok(weights(1.0)))
                .unwrap(),
        );
        let held = cache
            .lease(cache_key(0), 4, 4, || Ok(weights(1.0)))
            .unwrap();
        drop(
            cache
                .lease(cache_key(1), 4, 4, || Ok(weights(2.0)))
                .unwrap(),
        );
        drop(
            cache
                .lease(cache_key(1), 4, 4, || Ok(weights(2.0)))
                .unwrap(),
        );
        drop(
            cache
                .lease(cache_key(1), 4, 4, || Ok(weights(2.0)))
                .unwrap(),
        );
        assert!(cache.entries.contains_key(&cache_key(0)));
        assert!(!cache.entries.contains_key(&cache_key(1)));

        drop(held);
        drop(
            cache
                .lease(cache_key(1), 4, 4, || Ok(weights(2.0)))
                .unwrap(),
        );
        assert!(!cache.entries.contains_key(&cache_key(0)));
        assert!(cache.entries.contains_key(&cache_key(1)));
        assert_eq!(crate::weight_offload_stats().host_cache_evictions, 1);
    }

    #[test]
    fn qmoe_int2_single_block_matches_float_moe() {
        run_equivalence(2, 16, 16, 16, 1, false, false);
    }

    #[test]
    fn qmoe_int1_single_block_matches_float_moe() {
        run_equivalence(1, 16, 16, 16, 1, false, false);
    }

    #[test]
    fn qmoe_int8_matches_float_moe() {
        run_equivalence(8, 16, 16, 16, 1, false, false);
    }

    #[test]
    fn qmoe_int4_multiple_blocks_affine_matches_float_moe() {
        run_equivalence(4, 32, 32, 16, 1, false, true);
    }

    #[test]
    fn qmoe_int4_odd_blocks_affine_matches_float_moe() {
        run_equivalence(4, 48, 48, 16, 1, false, true);
    }

    #[test]
    fn qmoe_sub4_odd_blocks_affine_match_float_moe() {
        run_equivalence(2, 48, 48, 16, 1, false, true);
        run_equivalence(1, 48, 48, 16, 1, false, true);
    }

    #[test]
    fn qmoe_top2_normalized_matches_float_moe() {
        run_equivalence(4, 16, 16, 16, 2, true, false);
    }

    /// Independent oracle for the clipped-SwiGLU formula this kernel must
    /// match, transcribed directly from the CUDA reference
    /// (`swiglu_value` in `onnx-runtime-ep-cuda/src/kernels/qmoe.rs`):
    /// `min(gate, limit) * sigmoid(alpha * min(gate, limit)) * (clamp(linear, -limit, limit) + beta)`.
    /// Kept deliberately separate from `MoeAttributes::swiglu` so a bug
    /// introduced into the production formula does not silently pass by
    /// comparing against itself.
    fn reference_clipped_swiglu(gate: f32, linear: f32, alpha: f32, beta: f32, limit: f32) -> f32 {
        let bounded_gate = gate.min(limit);
        let bounded_linear = linear.clamp(-limit, limit);
        let sigmoid = 1.0 / (1.0 + (-(alpha * bounded_gate)).exp());
        bounded_gate * sigmoid * (bounded_linear + beta)
    }

    /// Builds a single-expert, unfused-gate (fc3) QMoE node quantized at
    /// `expert_weight_bits=8` (so the dequantization is exact and the test
    /// isolates activation-formula correctness, not quantization noise), runs
    /// it with the given `activation_alpha`/`activation_beta`/`swiglu_limit`,
    /// and returns the CPU kernel's output alongside an oracle computed
    /// directly from `reference_clipped_swiglu`.
    fn run_clipped_swiglu(alpha: f32, beta: f32, limit: f32) -> (Vec<f32>, Vec<f32>) {
        const HIDDEN: usize = 16;
        const INTER: usize = 16;
        const EXPERTS: usize = 1;
        const BLOCK: usize = 16;
        const BITS: usize = 8;

        let fc1 = quantize(EXPERTS, INTER, HIDDEN, BITS, BLOCK, false);
        let fc2 = quantize(EXPERTS, HIDDEN, INTER, BITS, BLOCK, false);
        let fc3 = quantize(EXPERTS, INTER, HIDDEN, BITS, BLOCK, false);
        let input: Vec<f32> = (0..HIDDEN).map(|i| (i as f32 * 0.2) - 1.5).collect();
        let router = vec![1.0f32];

        let q_shapes = [
            Some((DataType::Float32, &[1, HIDDEN][..])),
            Some((DataType::Float32, &[1, EXPERTS])),
            Some((DataType::Uint8, &[EXPERTS, INTER, HIDDEN])),
            Some((DataType::Float32, &[EXPERTS, INTER, 1])),
            None,
            Some((DataType::Uint8, &[EXPERTS, HIDDEN, INTER])),
            Some((DataType::Float32, &[EXPERTS, HIDDEN, 1])),
            None,
            Some((DataType::Uint8, &[EXPERTS, INTER, HIDDEN])),
            Some((DataType::Float32, &[EXPERTS, INTER, 1])),
        ];
        let q_attrs = vec![
            ("expert_weight_bits", Attribute::Int(BITS as i64)),
            ("block_size", Attribute::Int(BLOCK as i64)),
            ("k", Attribute::Int(1)),
            ("activation_type", Attribute::String(b"swiglu".to_vec())),
            ("swiglu_fusion", Attribute::Int(0)),
            ("activation_alpha", Attribute::Float(alpha)),
            ("activation_beta", Attribute::Float(beta)),
            ("swiglu_limit", Attribute::Float(limit)),
        ];
        let (q_graph, q_node) = model_node("QMoE", &q_shapes, &[1, HIDDEN], &q_attrs);
        let x = Owned::f32(&[1, HIDDEN], &input);
        let router_tensor = Owned::f32(&[1, EXPERTS], &router);
        let fc1_packed = Owned::u8(&[EXPERTS, INTER, HIDDEN], &fc1.packed);
        let fc1_scales = Owned::f32(&[EXPERTS, INTER, 1], &fc1.scales);
        let fc2_packed = Owned::u8(&[EXPERTS, HIDDEN, INTER], &fc2.packed);
        let fc2_scales = Owned::f32(&[EXPERTS, HIDDEN, 1], &fc2.scales);
        let fc3_packed = Owned::u8(&[EXPERTS, INTER, HIDDEN], &fc3.packed);
        let fc3_scales = Owned::f32(&[EXPERTS, INTER, 1], &fc3.scales);
        let mut q_output = Owned::zeros_f32(&[1, HIDDEN]);
        kernel(&q_graph, q_node)
            .unwrap()
            .execute(
                &[
                    x.view(),
                    router_tensor.view(),
                    fc1_packed.view(),
                    fc1_scales.view(),
                    TensorView::absent(DataType::Float32),
                    fc2_packed.view(),
                    fc2_scales.view(),
                    TensorView::absent(DataType::Float32),
                    fc3_packed.view(),
                    fc3_scales.view(),
                ],
                &mut [q_output.view_mut()],
            )
            .unwrap();

        // Reference: dequantized dense matmuls, then the clipped-SwiGLU
        // oracle applied elementwise, matching `run_expert`'s unfused-gate
        // path with the exact same weights.
        let (gate, linear_part): (Vec<f32>, Vec<f32>) = (0..INTER)
            .map(|row| {
                let mut g = 0.0f32;
                let mut l = 0.0f32;
                for (col, &x) in input.iter().enumerate() {
                    g += x * fc1.dequantized[row * HIDDEN + col];
                    l += x * fc3.dequantized[row * HIDDEN + col];
                }
                (g, l)
            })
            .unzip();
        let activated: Vec<f32> = gate
            .iter()
            .zip(&linear_part)
            .map(|(&g, &l)| reference_clipped_swiglu(g, l, alpha, beta, limit))
            .collect();
        let oracle: Vec<f32> = (0..HIDDEN)
            .map(|row| {
                activated
                    .iter()
                    .enumerate()
                    .map(|(col, &a)| a * fc2.dequantized[row * INTER + col])
                    .sum()
            })
            .collect();
        (q_output.to_f32(), oracle)
    }

    /// Default clipped-SwiGLU attributes (`alpha=1`, `beta=0`,
    /// `limit=f32::MAX`) must match the independent formula oracle. This pins
    /// the baseline shape of the activation before non-default attributes
    /// are exercised.
    #[test]
    fn qmoe_swiglu_default_clip_attrs_match_reference_formula() {
        let (got, want) = run_clipped_swiglu(1.0, 0.0, crate::kernels::moe::DEFAULT_SWIGLU_LIMIT);
        assert_close(&got, &want);
    }

    /// Non-default `activation_alpha`/`activation_beta`/`swiglu_limit` (the
    /// GLM-5.2/DeepSeek-V4-style clipped-SwiGLU parameters CUDA already
    /// supports) must be honored by the CPU kernel and match the same
    /// independent formula oracle, not merely run without error.
    #[test]
    fn qmoe_swiglu_non_default_clip_attrs_match_reference_formula() {
        let (got, want) = run_clipped_swiglu(1.702, 1.0, 3.5);
        assert_close(&got, &want);
    }

    /// Regression guard for the silent-ignore failure mode: if the kernel
    /// ever stopped reading `activation_alpha`/`activation_beta`/
    /// `swiglu_limit` from the node (e.g. a future refactor drops them from
    /// the parsed attributes or from the formula), this would produce the
    /// *same* output as the default attributes above and this test would
    /// catch that by asserting the non-default output differs.
    #[test]
    fn qmoe_swiglu_non_default_clip_attrs_change_output_vs_defaults() {
        let (default_output, _) =
            run_clipped_swiglu(1.0, 0.0, crate::kernels::moe::DEFAULT_SWIGLU_LIMIT);
        let (non_default_output, _) = run_clipped_swiglu(1.702, 1.0, 3.5);
        assert_ne!(
            default_output, non_default_output,
            "non-default activation_alpha/activation_beta/swiglu_limit must change QMoE output"
        );
    }

    /// A non-finite `activation_alpha`/`activation_beta`/`swiglu_limit`
    /// attribute must fail loudly at kernel-creation time rather than
    /// silently producing NaN/Inf-poisoned output.
    #[test]
    fn qmoe_rejects_non_finite_clip_attrs() {
        let inputs = [
            Some((DataType::Float32, &[1, 16][..])),
            Some((DataType::Float32, &[1, 1])),
            Some((DataType::Uint8, &[1, 16, 16])),
            Some((DataType::Float32, &[1, 16, 1])),
            None,
            Some((DataType::Uint8, &[1, 16, 16])),
            Some((DataType::Float32, &[1, 16, 1])),
        ];
        for (name, value) in [
            ("activation_alpha", f32::NAN),
            ("activation_alpha", f32::INFINITY),
            ("activation_alpha", f32::NEG_INFINITY),
            ("activation_beta", f32::NAN),
            ("activation_beta", f32::INFINITY),
            ("activation_beta", f32::NEG_INFINITY),
            ("swiglu_limit", f32::NAN),
            ("swiglu_limit", f32::INFINITY),
            ("swiglu_limit", f32::NEG_INFINITY),
            ("swiglu_limit", 0.0),
            ("swiglu_limit", -1.0),
        ] {
            let mut attrs = vec![
                ("expert_weight_bits", Attribute::Int(8)),
                ("block_size", Attribute::Int(16)),
                ("k", Attribute::Int(1)),
                ("activation_type", Attribute::String(b"swiglu".to_vec())),
                ("swiglu_fusion", Attribute::Int(1)),
            ];
            attrs.push((name, Attribute::Float(value)));
            let (graph, node) = model_node("QMoE", &inputs, &[1, 16], &attrs);
            let error = kernel(&graph, node)
                .err()
                .unwrap_or_else(|| panic!("{name}={value} must be rejected"));
            assert!(
                error.to_string().contains(name),
                "error for {name}={value} should name the offending attribute, got: {error}"
            );
        }
    }

    #[test]
    fn qmoe_rejects_unsupported_weight_bits() {
        let inputs = [
            Some((DataType::Float32, &[1, 16][..])),
            Some((DataType::Float32, &[1, 2])),
            Some((DataType::Uint8, &[2, 16, 8])),
            Some((DataType::Float32, &[2, 16, 1])),
            None,
            Some((DataType::Uint8, &[2, 16, 8])),
            Some((DataType::Float32, &[2, 16, 1])),
        ];
        for bits in [3, 5] {
            let attrs = attributes(bits, 16, 1, false);
            let (graph, node) = model_node("QMoE", &inputs, &[1, 16], &attrs);
            match kernel(&graph, node) {
                Err(EpError::KernelFailed(message)) => {
                    assert!(message.contains("expert_weight_bits must be one of {1, 2, 4, 8}"));
                    assert!(message.contains(&format!("got {bits}")));
                }
                Err(other) => panic!("expected KernelFailed for {bits}-bit QMoE, got {other}"),
                Ok(_) => panic!("unsupported {bits}-bit QMoE unexpectedly produced a kernel"),
            }
        }
    }

    #[test]
    fn qmoe_rejects_unsupported_block_size() {
        let inputs = [
            Some((DataType::Float32, &[1, 16][..])),
            Some((DataType::Float32, &[1, 2])),
            Some((DataType::Uint8, &[2, 16, 8])),
            Some((DataType::Float32, &[2, 16, 2])),
            None,
            Some((DataType::Uint8, &[2, 16, 8])),
            Some((DataType::Float32, &[2, 16, 2])),
        ];
        let attrs = attributes(4, 8, 1, false);
        let (graph, node) = model_node("QMoE", &inputs, &[1, 16], &attrs);
        let failure = match kernel(&graph, node) {
            Ok(_) => panic!("unsupported block_size unexpectedly produced a kernel"),
            Err(error) => error.to_string(),
        };
        assert!(failure.contains("block_size must be a power of two and at least 16"));
    }

    #[test]
    fn qmoe_rejects_flattened_rows_overflow_before_allocation() {
        let x = tiny_owned(DataType::Float32, &[usize::MAX, 2, 0]);
        let router = tiny_owned(DataType::Float32, &[0, 1]);
        let fc1_packed = tiny_owned(DataType::Uint8, &[1, 16, 0]);
        let fc1_scales = tiny_owned(DataType::Float32, &[1, 16, 0]);
        let fc2_packed = tiny_owned(DataType::Uint8, &[1, 0, 8]);
        let fc2_scales = tiny_owned(DataType::Float32, &[1, 0, 1]);
        let mut output = tiny_owned(DataType::Float32, &[usize::MAX, 2, 0]);

        let result = overflow_test_kernel().execute(
            &[
                x.view(),
                router.view(),
                fc1_packed.view(),
                fc1_scales.view(),
                TensorView::absent(DataType::Float32),
                fc2_packed.view(),
                fc2_scales.view(),
            ],
            &mut [output.view_mut()],
        );

        assert_kernel_failure_contains(result, "flattened input row count overflow");
    }

    #[test]
    fn qmoe_rejects_zero_masked_tensor_overflow_before_allocation() {
        let x = tiny_owned(DataType::Float32, &[0, usize::MAX, 2]);
        let router = tiny_owned(DataType::Float32, &[0, 1]);
        let fc1_packed = tiny_owned(DataType::Uint8, &[1, 16, 1]);
        let fc1_scales = tiny_owned(DataType::Float32, &[1, 16, 0]);
        let fc2_packed = tiny_owned(DataType::Uint8, &[1, 2, 8]);
        let fc2_scales = tiny_owned(DataType::Float32, &[1, 2, 1]);
        let mut output = tiny_owned(DataType::Float32, &[0, usize::MAX, 2]);

        let result = overflow_test_kernel().execute(
            &[
                x.view(),
                router.view(),
                fc1_packed.view(),
                fc1_scales.view(),
                TensorView::absent(DataType::Float32),
                fc2_packed.view(),
                fc2_scales.view(),
            ],
            &mut [output.view_mut()],
        );

        assert_kernel_failure_contains(result, "input 0 element count overflow");
    }

    #[test]
    fn qmoe_rejects_isize_max_exceeding_byte_count_before_allocation() {
        let hidden = isize::MAX as usize / std::mem::size_of::<f32>() + 1;
        let x = tiny_owned(DataType::Float32, &[1, hidden]);
        let router = tiny_owned(DataType::Float32, &[1, 1]);
        let fc1_packed = tiny_owned(DataType::Uint8, &[1, 16, hidden / 2]);
        let fc1_scales = tiny_owned(DataType::Float32, &[1, 16, hidden / 16]);
        let fc2_packed = tiny_owned(DataType::Uint8, &[1, hidden, 8]);
        let fc2_scales = tiny_owned(DataType::Float32, &[1, hidden, 1]);
        let mut output = tiny_owned(DataType::Float32, &[1, hidden]);

        let result = overflow_test_kernel().execute(
            &[
                x.view(),
                router.view(),
                fc1_packed.view(),
                fc1_scales.view(),
                TensorView::absent(DataType::Float32),
                fc2_packed.view(),
                fc2_scales.view(),
            ],
            &mut [output.view_mut()],
        );

        assert_kernel_failure_contains(result, "input 0 byte count exceeds isize::MAX");
    }

    #[test]
    fn qmoe_rejects_quantized_expert_layout_overflow_before_allocation() {
        let experts = usize::MAX / 16 + 1;
        let x = tiny_owned(DataType::Float32, &[1, 16]);
        let router = tiny_owned(DataType::Float32, &[1, experts]);
        let fc1_packed = tiny_owned(DataType::Uint8, &[experts, 16, 8]);
        let fc1_scales = tiny_owned(DataType::Float32, &[experts, 16, 1]);
        let fc2_packed = tiny_owned(DataType::Uint8, &[experts, 16, 8]);
        let fc2_scales = tiny_owned(DataType::Float32, &[experts, 16, 1]);
        let mut output = tiny_owned(DataType::Float32, &[1, 16]);

        let result = overflow_test_kernel().execute(
            &[
                x.view(),
                router.view(),
                fc1_packed.view(),
                fc1_scales.view(),
                TensorView::absent(DataType::Float32),
                fc2_packed.view(),
                fc2_scales.view(),
            ],
            &mut [output.view_mut()],
        );

        assert_kernel_failure_contains(result, "input 2 element count overflow");
    }
}
