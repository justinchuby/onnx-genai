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
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use onnx_runtime_ep_api::{
    EpError, ExternalMmapRegion, Kernel, KernelFactory, Result, TensorBacking, TensorMut,
    TensorView,
};
use onnx_runtime_ir::{DataType, Node};
use onnx_runtime_loader::{
    ExpertQuantization, ExpertStorageOrder, ExpertTensorLayout, WeightRegionCatalog,
};

use super::matmul_nbits::dequantize_nbits_row;
use super::moe::{MoeAttributes, routing_weights, run_expert};
use super::{
    check_arity, contiguous_f32_slice, contiguous_u8_slice, to_dense_bytes, to_dense_f32,
    write_dense_f32,
};
use crate::weight_offload::{
    WeightOffloadMode, checked_host_budget, metrics, weight_offload_host_budget,
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

/// Per-row block-dequantizing integer QMoE reference kernel.
pub struct QMoEKernel {
    layer_id: u32,
    attributes: MoeAttributes,
    bits: usize,
    block_size: usize,
    weight_offload: WeightOffloadMode,
    host_cache: WeightOffloadHostCache,
}

impl KernelFactory for QMoEFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
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
        Ok(Box::new(QMoEKernel {
            layer_id: node.id.0,
            attributes,
            bits: bits as usize,
            block_size: block_size as usize,
            weight_offload: WeightOffloadMode::from_env(),
            host_cache: self.host_cache.clone(),
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
        if hidden % self.block_size != 0 {
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
        let route_first = self.weight_offload.enabled
            && fc1.is_pageable()
            && fc2.is_pageable()
            && fc3.as_ref().is_none_or(QuantizedExperts::is_pageable);
        if route_first {
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

            let contribution_elements =
                checked_product(&[route_slots, hidden], "routed contribution element count")?;
            checked_byte_count(
                contribution_elements,
                std::mem::size_of::<f32>(),
                "routed contribution byte count",
            )?;
            let mut contributions = vec![0.0f32; contribution_elements];
            for (expert, expert_tasks) in tasks {
                let expanded_bytes = DequantizedExpert::expanded_bytes(&fc1, &fc2, fc3.as_ref())?;
                let key = ExpertCacheKey::new(
                    self.layer_id,
                    expert,
                    self.bits,
                    self.block_size,
                    &fc1,
                    &fc2,
                    fc3.as_ref(),
                );
                let weights = self.host_cache.lease(key, expanded_bytes, || {
                    DequantizedExpert::load(expert, &fc1, &fc2, fc3.as_ref())
                })?;
                metrics().record_dequantized_window(1);
                if weights.read_from_mmap {
                    let mut bytes_read = fc1
                        .expert_source_bytes(expert)?
                        .checked_add(fc2.expert_source_bytes(expert)?)
                        .ok_or_else(|| error("selected expert byte count overflow"))?;
                    if let Some(source) = &fc3 {
                        bytes_read = bytes_read
                            .checked_add(source.expert_source_bytes(expert)?)
                            .ok_or_else(|| error("selected expert byte count overflow"))?;
                    }
                    metrics().record_bytes_read(bytes_read).map_err(error)?;
                }
                for (row, slot, route_weight) in expert_tasks {
                    let input_range = checked_range(row, hidden, "input row")?;
                    let contribution_range =
                        checked_range(slot, hidden, "routed contribution row")?;
                    accumulate_expert(
                        &mut contributions[contribution_range],
                        &x[input_range],
                        expert,
                        route_weight,
                        &weights,
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
                    let contribution_range =
                        checked_range(slot, hidden, "routed contribution row")?;
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
        } else {
            for row in 0..rows {
                let router_range = checked_range(row, experts, "router row")?;
                let route = routing_weights(
                    &router[router_range.clone()],
                    aggregation
                        .as_deref()
                        .map(|weights| &weights[router_range.clone()]),
                    self.attributes.k,
                    self.attributes.normalize_routing_weights,
                );
                let input_range = checked_range(row, hidden, "input row")?;
                let input_row = &x[input_range];
                let output_range = checked_range(row, hidden, "output row")?;
                let output_row = &mut output[output_range];
                for (expert, route_weight) in route {
                    let weights = DequantizedExpert {
                        fc1: fc1.dequantize_expert(expert)?,
                        fc2: fc2.dequantize_expert(expert)?,
                        fc3: fc3
                            .as_ref()
                            .map(|weights| weights.dequantize_expert(expert))
                            .transpose()?,
                    };
                    accumulate_expert(
                        output_row,
                        input_row,
                        expert,
                        route_weight,
                        &weights,
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
        }
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
}

impl DequantizedExpert {
    fn load(
        expert: usize,
        fc1: &QuantizedExperts<'_>,
        fc2: &QuantizedExperts<'_>,
        fc3: Option<&QuantizedExperts<'_>>,
    ) -> Result<Self> {
        Ok(Self {
            fc1: fc1.dequantize_expert(expert)?,
            fc2: fc2.dequantize_expert(expert)?,
            fc3: fc3
                .map(|weights| weights.dequantize_expert(expert))
                .transpose()?,
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

struct ExpertLease {
    weights: Arc<DequantizedExpert>,
    read_from_mmap: bool,
}

impl Deref for ExpertLease {
    type Target = DequantizedExpert;

    fn deref(&self) -> &Self::Target {
        &self.weights
    }
}

struct HostCacheEntry {
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
    clock: u64,
}

const ADMISSION_FREQUENCY: u64 = 2;
const PIN_FREQUENCY: u64 = 3;
const PIN_WINDOW: u64 = 8;
const HISTORY_DECAY_WINDOW: u64 = 64;
const HISTORY_EXPIRY_WINDOW: u64 = HISTORY_DECAY_WINDOW * 256;
const MAX_ADMISSION_HISTORY_ENTRIES: usize = 4096;

impl HostExpertCache {
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
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or_else(|| error("host-cache access clock overflow"))?;
        let now = self.clock;
        self.trim_to_budget(budget_bytes)?;
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.frequency = entry.frequency.saturating_add(1);
            entry.last_used = now;
            if entry.frequency >= PIN_FREQUENCY {
                entry.pin_until = now.saturating_add(PIN_WINDOW);
            }
            metrics().record_host_cache_hit();
            metrics()
                .record_host_cache_residency(self.owned_bytes, budget_bytes)
                .map_err(error)?;
            return Ok(ExpertLease {
                weights: Arc::clone(&entry.weights),
                read_from_mmap: false,
            });
        }

        metrics().record_host_cache_miss();
        if budget_bytes == 0 || expanded_bytes == 0 || expanded_bytes > budget_bytes {
            metrics()
                .record_host_cache_residency(self.owned_bytes, budget_bytes)
                .map_err(error)?;
            return Ok(ExpertLease {
                weights: Arc::new(load()?),
                read_from_mmap: true,
            });
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
            metrics()
                .record_host_cache_residency(self.owned_bytes, budget_bytes)
                .map_err(error)?;
            return Ok(ExpertLease {
                weights: Arc::new(load()?),
                read_from_mmap: true,
            });
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
                metrics()
                    .record_host_cache_residency(self.owned_bytes, budget_bytes)
                    .map_err(error)?;
                return Ok(ExpertLease {
                    weights: Arc::new(load()?),
                    read_from_mmap: true,
                });
            }
        }

        let mut evicted_bytes = 0usize;
        for victim in &victims {
            let entry = self
                .entries
                .remove(victim)
                .expect("selected host-cache victim exists");
            evicted_bytes = evicted_bytes
                .checked_add(entry.expanded_bytes)
                .ok_or_else(|| error("evicted host-cache byte count overflow"))?;
        }
        self.owned_bytes = self
            .owned_bytes
            .checked_sub(evicted_bytes)
            .ok_or_else(|| error("owned host-cache byte accounting underflow"))?;
        self.owned_bytes = self
            .owned_bytes
            .checked_add(expanded_bytes)
            .ok_or_else(|| error("owned host-cache byte count overflow"))?;
        debug_assert!(self.owned_bytes <= budget_bytes);
        metrics()
            .record_host_cache_evictions(victims.len())
            .map_err(error)?;
        metrics()
            .record_host_cache_residency(self.owned_bytes, budget_bytes)
            .map_err(error)?;

        let loaded = match load() {
            Ok(loaded) => loaded,
            Err(failure) => {
                self.owned_bytes = self
                    .owned_bytes
                    .checked_sub(expanded_bytes)
                    .ok_or_else(|| error("host-cache reservation rollback underflow"))?;
                metrics()
                    .record_host_cache_residency(self.owned_bytes, budget_bytes)
                    .map_err(error)?;
                return Err(failure);
            }
        };
        let weights = Arc::new(loaded);
        self.history.remove(&key);
        self.entries.insert(
            key,
            HostCacheEntry {
                weights: Arc::clone(&weights),
                expanded_bytes,
                frequency: candidate_frequency,
                last_used: now,
                pin_until: 0,
            },
        );
        Ok(ExpertLease {
            weights,
            read_from_mmap: true,
        })
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

    fn trim_to_budget(&mut self, budget_bytes: usize) -> Result<()> {
        if self.owned_bytes <= budget_bytes {
            metrics()
                .record_host_cache_residency(self.owned_bytes, budget_bytes)
                .map_err(error)?;
            return Ok(());
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
                "cannot lower owned host-cache budget to {budget_bytes} bytes while {} bytes are leased",
                projected
            )));
        }
        for victim in &victims {
            self.entries
                .remove(victim)
                .expect("selected host-cache trim victim exists");
        }
        self.owned_bytes = projected;
        metrics()
            .record_host_cache_evictions(victims.len())
            .map_err(error)?;
        metrics()
            .record_host_cache_residency(self.owned_bytes, budget_bytes)
            .map_err(error)?;
        Ok(())
    }

    #[cfg(test)]
    fn clear(&mut self) {
        self.entries.clear();
        self.history.clear();
        self.owned_bytes = 0;
        self.clock = 0;
    }
}

struct WeightOffloadHostCacheInner {
    budget_bytes: AtomicU64,
    cache: Mutex<HostExpertCache>,
}

/// A governor-owned warm-host expert-cache partition.
#[derive(Clone)]
pub struct WeightOffloadHostCache {
    inner: Arc<WeightOffloadHostCacheInner>,
}

impl WeightOffloadHostCache {
    /// Create an independent cache partition with its own byte ceiling.
    pub fn new(budget_bytes: u64) -> std::result::Result<Self, &'static str> {
        checked_host_budget(budget_bytes)?;
        Ok(Self {
            inner: Arc::new(WeightOffloadHostCacheInner {
                budget_bytes: AtomicU64::new(budget_bytes),
                cache: Mutex::new(HostExpertCache::default()),
            }),
        })
    }

    /// Return this partition's governor-configured byte ceiling.
    pub fn configured_budget_bytes(&self) -> u64 {
        self.inner.budget_bytes.load(Ordering::Relaxed)
    }

    fn lease<F>(&self, key: ExpertCacheKey, expanded_bytes: usize, load: F) -> Result<ExpertLease>
    where
        F: FnOnce() -> Result<DequantizedExpert>,
    {
        let budget_bytes =
            weight_offload_host_budget(self.configured_budget_bytes()).map_err(error)?;
        self.inner
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .lease(key, expanded_bytes, budget_bytes, load)
    }

    pub(crate) fn reconfigure(&self, budget_bytes: u64) -> Result<()> {
        let checked = checked_host_budget(budget_bytes).map_err(error)?;
        self.inner
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .trim_to_budget(checked)?;
        self.inner
            .budget_bytes
            .store(budget_bytes, Ordering::Relaxed);
        Ok(())
    }

    #[cfg(test)]
    fn clear(&self) {
        self.inner
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }
}

pub(crate) fn default_weight_offload_host_cache() -> &'static WeightOffloadHostCache {
    static CACHE: OnceLock<WeightOffloadHostCache> = OnceLock::new();
    CACHE.get_or_init(|| WeightOffloadHostCache::new(0).expect("zero host-cache budget is valid"))
}

#[allow(clippy::too_many_arguments)]
fn accumulate_expert(
    output_row: &mut [f32],
    input_row: &[f32],
    expert: usize,
    route_weight: f32,
    weights: &DequantizedExpert,
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
    let expert_out = run_expert(
        input_row,
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
    );
    for feature in 0..hidden {
        output_row[feature] += route_weight * expert_out[feature];
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
            TensorBacking::Opaque => None,
        };
        let pageable = prefer_mmap
            && packed_catalog.is_pageable()
            && scale_catalog.is_pageable()
            && zero_point_catalog
                .as_ref()
                .is_none_or(WeightRegionCatalog::is_pageable)
            && packed.device.is_host_accessible()
            && scales.device.is_host_accessible()
            && zero_points.is_none_or(|points| points.device.is_host_accessible())
            && mapped_region(packed).is_some()
            && mapped_region(scales).is_some()
            && zero_points.is_none_or(|points| mapped_region(points).is_some());

        let (packed_data, scale_data, zero_point_data) = if pageable {
            (
                Cow::Borrowed(contiguous_u8_slice(packed)?),
                Cow::Borrowed(contiguous_f32_slice(scales)?),
                zero_points
                    .map(contiguous_u8_slice)
                    .transpose()?
                    .map(Cow::Borrowed),
            )
        } else {
            (
                Cow::Owned(to_dense_bytes(packed)?),
                Cow::Owned(to_dense_f32(scales)?),
                zero_points.map(to_dense_bytes).transpose()?.map(Cow::Owned),
            )
        };
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
        let mut output = Vec::new();
        output
            .try_reserve_exact(self.dequantized_elements)
            .map_err(|failure| {
                error(format!("failed to allocate dequantized expert: {failure}"))
            })?;
        output.resize(self.dequantized_elements, 0.0f32);
        for row in 0..self.out_features {
            let expert_row = expert
                .checked_mul(self.out_features)
                .and_then(|offset| offset.checked_add(row))
                .ok_or_else(|| error("expert-row offset overflow"))?;
            let packed_range =
                checked_range(expert_row, self.packed_in, "packed-weight expert row")?;
            let scale_range = checked_range(expert_row, self.blocks, "scale expert row")?;
            let zero_point_range =
                checked_range(expert_row, self.zero_point_bytes, "zero-point expert row")?;
            let output_range = checked_range(row, self.in_features, "dequantized output row")?;
            dequantize_nbits_row(
                &self.packed[packed_range],
                &self.scales[scale_range],
                self.zero_points
                    .as_ref()
                    .map(|points| &points[zero_point_range]),
                &mut output[output_range],
                self.bits,
                self.block_size,
            );
        }
        Ok(output)
    }

    fn dequantized_bytes(&self) -> Result<usize> {
        checked_byte_count(
            self.dequantized_elements,
            std::mem::size_of::<f32>(),
            "per-expert expanded byte count",
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
    use crate::weight_offload::{WeightOffloadMode, metrics};
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
        CpuExecutionProvider::new().get_kernel(model.graph.node(node), &[], 1)
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
        const EXPERTS: usize = 4;
        const ROWS: usize = 4;
        const HIDDEN: usize = 16;
        const INTER: usize = 16;
        const BITS: usize = 4;
        const BLOCK: usize = 16;
        const PACKED: usize = HIDDEN / 2;
        const BLOCKS: usize = 1;

        let fc1 = quantize(EXPERTS, INTER, HIDDEN, BITS, BLOCK, false);
        let fc2 = quantize(EXPERTS, HIDDEN, INTER, BITS, BLOCK, false);
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
            .join("tests/fixtures/qmoe_weight_offload/model.onnx");
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
            host_cache: default_weight_offload_host_cache().clone(),
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

    fn reset_offload_test_state(host_cache_budget: u64) {
        default_weight_offload_host_cache().clear();
        metrics().reset();
        crate::set_weight_offload_host_budget(host_cache_budget).expect("test host-cache budget");
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
        let experts = 2;
        let rows = 2;
        let fc1 = quantize(experts, inter, hidden, bits, block_size, affine);
        let fc2 = quantize(experts, hidden, inter, bits, block_size, affine);
        let input: Vec<f32> = (0..rows * hidden)
            .map(|index| (index % 5) as f32 * 0.25 - 0.5)
            .collect();
        let router = vec![3.0, 1.0, 0.5, 2.5];
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
        let mut q_output = Owned::zeros_f32(&[rows, hidden]);
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
                    TensorView::absent(DataType::Uint8),
                    TensorView::absent(DataType::Float32),
                    TensorView::absent(DataType::Float32),
                    fc1_zero_points
                        .as_ref()
                        .map_or_else(|| TensorView::absent(DataType::Uint8), Owned::view),
                    fc2_zero_points
                        .as_ref()
                        .map_or_else(|| TensorView::absent(DataType::Uint8), Owned::view),
                ],
                &mut [q_output.view_mut()],
            )
            .unwrap();
        assert_close(&q_output.to_f32(), &float_output.to_f32());
    }

    #[test]
    fn qmoe_int4_single_block_matches_float_moe() {
        run_equivalence(4, 16, 16, 16, 1, false, false);
    }

    #[test]
    fn route_first_mmap_matches_full_dequant_on_pseudorandom_inputs() {
        let _guard = metrics_test_lock().lock().expect("metrics test lock");
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
        let _guard = metrics_test_lock().lock().expect("metrics test lock");
        let input = pseudo_random_values(4 * 16, 0xa11c_e55);
        let router = pseudo_random_values(4 * 4, 0x5151_5151);
        let baseline = run_mmap_qmoe_with_k(false, false, &input, &router, 3);
        reset_offload_test_state(0);
        let route_first = run_mmap_qmoe_with_k(true, false, &input, &router, 3);
        assert_eq!(route_first.output, baseline.output);
    }

    #[test]
    fn route_first_reads_only_unique_selected_expert_ranges() {
        let _guard = metrics_test_lock().lock().expect("metrics test lock");
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

    #[test]
    fn route_first_bounds_dequantized_residency_when_all_experts_are_selected() {
        let _guard = metrics_test_lock().lock().expect("metrics test lock");
        let input = pseudo_random_values(4 * 16, 0x1234);
        let router = vec![
            9.0, 0.0, 0.0, 0.0, // expert 0
            0.0, 9.0, 0.0, 0.0, // expert 1
            0.0, 0.0, 9.0, 0.0, // expert 2
            0.0, 0.0, 0.0, 9.0, // expert 3
        ];
        let baseline = run_mmap_qmoe(false, false, &input, &router);
        reset_offload_test_state(0);
        let route_first = run_mmap_qmoe(true, false, &input, &router);
        let stats = crate::weight_offload_stats();
        assert_close(&route_first.output, &baseline.output);
        assert_eq!(stats.unique_experts_per_batch, 4);
        assert_eq!(stats.peak_dequantized_experts, 1);
    }

    #[test]
    fn interleaved_expert_layout_falls_back_without_error() {
        let _guard = metrics_test_lock().lock().expect("metrics test lock");
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
        let _guard = metrics_test_lock().lock().expect("metrics test lock");
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
        let _guard = metrics_test_lock().lock().expect("metrics test lock");
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
        let _guard = metrics_test_lock().lock().expect("metrics test lock");
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
        let _guard = metrics_test_lock().lock().expect("metrics test lock");
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
        let _guard = metrics_test_lock().lock().expect("metrics test lock");
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
        let _guard = metrics_test_lock().lock().expect("metrics test lock");
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
        let _guard = metrics_test_lock().lock().expect("metrics test lock");
        metrics().reset();
        let engine_a = WeightOffloadHostCache::new(4).unwrap();
        let engine_b = WeightOffloadHostCache::new(8).unwrap();
        let weights = |value| DequantizedExpert {
            fc1: vec![value],
            fc2: Vec::new(),
            fc3: None,
        };

        for cache in [&engine_a, &engine_b] {
            drop(cache.lease(cache_key(0), 4, || Ok(weights(1.0))).unwrap());
            drop(cache.lease(cache_key(0), 4, || Ok(weights(1.0))).unwrap());
            drop(cache.lease(cache_key(1), 4, || Ok(weights(2.0))).unwrap());
            drop(cache.lease(cache_key(1), 4, || Ok(weights(2.0))).unwrap());
        }

        assert_eq!(engine_a.configured_budget_bytes(), 4);
        assert_eq!(engine_b.configured_budget_bytes(), 8);
        let cache_a = engine_a.inner.cache.lock().unwrap();
        let cache_b = engine_b.inner.cache.lock().unwrap();
        assert!(cache_a.owned_bytes <= 4);
        assert!(cache_b.owned_bytes <= 8);
        assert_eq!(cache_a.entries.len(), 1);
        assert_eq!(cache_b.entries.len(), 2);
    }

    #[test]
    fn admission_history_skips_uncacheable_keys_and_expires_churn() {
        let _guard = metrics_test_lock().lock().expect("metrics test lock");
        metrics().reset();
        let mut cache = HostExpertCache::default();
        let weights = || DequantizedExpert {
            fc1: vec![1.0],
            fc2: Vec::new(),
            fc3: None,
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
    fn active_host_cache_lease_blocks_eviction_until_drop() {
        let _guard = metrics_test_lock().lock().expect("metrics test lock");
        reset_offload_test_state(4);
        let weights = |value| DequantizedExpert {
            fc1: vec![value],
            fc2: Vec::new(),
            fc3: None,
        };
        let mut cache = HostExpertCache::default();

        drop(cache.lease(cache_key(0), 4, 4, || Ok(weights(1.0))).unwrap());
        let held = cache.lease(cache_key(0), 4, 4, || Ok(weights(1.0))).unwrap();
        drop(cache.lease(cache_key(1), 4, 4, || Ok(weights(2.0))).unwrap());
        drop(cache.lease(cache_key(1), 4, 4, || Ok(weights(2.0))).unwrap());
        drop(cache.lease(cache_key(1), 4, 4, || Ok(weights(2.0))).unwrap());
        assert!(cache.entries.contains_key(&cache_key(0)));
        assert!(!cache.entries.contains_key(&cache_key(1)));

        drop(held);
        drop(cache.lease(cache_key(1), 4, 4, || Ok(weights(2.0))).unwrap());
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
