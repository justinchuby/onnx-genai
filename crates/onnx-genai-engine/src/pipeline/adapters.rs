mod peft;
use super::row_state::{RowScopedState, gather_rows};
use crate::engine::memory_plan::Holder;
use anyhow::Context;
use onnx_genai_metadata::{
    AdapterArtifact, AdapterScaleEncoding, AdapterServiceContract, AdapterWeightFormat,
};
use onnx_genai_ort::Value;
use onnx_runtime_memory_governor::{MemoryGovernor, MemoryLease};
use serde::Deserialize;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct AdapterActivation {
    pub adapter: String,
    pub scale: f32,
}

impl AdapterActivation {
    pub fn new(adapter: impl Into<String>, scale: f32) -> Self {
        Self {
            adapter: adapter.into(),
            scale,
        }
    }
}

/// Adapter composition for each row of the current batch, in row order.
///
/// Rows are positional. The runtime associates a batch row with a request
/// through its own private request table; no scheduler slot ID or request epoch
/// is serialized into metadata or carried through this type.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AdapterSelection {
    /// Immutable composition per batch row, indexed by row position.
    pub rows: Vec<Vec<AdapterActivation>>,
}

impl AdapterSelection {
    /// Append the composition for the next batch row.
    pub fn with_row(mut self, adapters: impl IntoIterator<Item = AdapterActivation>) -> Self {
        self.rows.push(adapters.into_iter().collect());
        self
    }

    pub(super) fn bind_inputs(
        &self,
        service: &AdapterServiceContract,
        cache: &AdapterCache,
    ) -> anyhow::Result<HashMap<String, Value>> {
        anyhow::ensure!(
            !self.rows.is_empty(),
            "adapter selection needs at least one request row"
        );
        let width = service.selection.max_adapters;
        let slots = self
            .rows
            .len()
            .checked_mul(width)
            .context("adapter selection dimensions overflow; reduce batch size")?;
        let bytes = slots
            .checked_mul(12)
            .and_then(|n| {
                self.rows
                    .len()
                    .checked_mul(9)
                    .and_then(|rows| n.checked_add(rows))
            })
            .context("adapter selection byte count overflow")?;
        let lease = Arc::new(cache.reserve(u64::try_from(bytes)?, Holder::Activations)?);
        let mut ids = vec![-1_i64; slots];
        let mut scales = vec![0_f32; slots];
        let mut counts = Vec::with_capacity(self.rows.len());
        for (row, activations) in self.rows.iter().enumerate() {
            anyhow::ensure!(
                activations.len() <= width,
                "adapter row {row} selects {} adapters, exceeding adapters.selection.max_adapters={width}",
                activations.len()
            );
            let mut seen = std::collections::HashSet::new();
            for (slot, activation) in activations.iter().enumerate() {
                anyhow::ensure!(
                    activation.scale.is_finite(),
                    "adapter '{}' scale must be finite",
                    activation.adapter
                );
                anyhow::ensure!(
                    seen.insert(&activation.adapter),
                    "adapter row {row} contains duplicate adapter '{}'",
                    activation.adapter
                );
                let artifact = service.artifacts.get(&activation.adapter).with_context(||format!("adapter row {row} selects undeclared adapter '{}'; use the package's declared adapter aliases",activation.adapter))?;
                ids[row * width + slot] =
                    i64::try_from(artifact.index).context("declared adapter ID exceeds int64")?;
                scales[row * width + slot] = activation.scale;
            }
            counts.push(i64::try_from(activations.len())?);
        }
        let batch = i64::try_from(self.rows.len())?;
        let width = i64::try_from(width)?;
        let mut result = HashMap::from([
            (
                service.selection.segments.clone(),
                governed_value(
                    Value::from_vec_i64(ids, &[batch, width])?,
                    Arc::clone(&lease),
                )?,
            ),
            (
                service.selection.adapter_counts.clone(),
                governed_value(Value::from_vec_i64(counts, &[batch])?, Arc::clone(&lease))?,
            ),
            (
                service.selection.scales.clone(),
                governed_value(
                    Value::from_vec_f32(scales, &[batch, width])?,
                    Arc::clone(&lease),
                )?,
            ),
        ]);
        if let Some(active) = &service.selection.active {
            result.insert(
                active.clone(),
                governed_value(
                    Value::from_raw_bytes(
                        vec![1; self.rows.len()],
                        &[batch],
                        onnx_genai_ort::DataType::Bool,
                    )?,
                    Arc::clone(&lease),
                )?,
            );
        }
        Ok(result)
    }
}

pub(super) fn selection_from_inputs(
    service: &AdapterServiceContract,
    values: &HashMap<String, Value>,
    batch_rows: usize,
) -> anyhow::Result<AdapterSelection> {
    let selection = &service.selection;
    let max_adapters = selection.max_adapters;
    let segments = values
        .get(&selection.segments)
        .with_context(|| format!("adapter segments input '{}' is absent", selection.segments))?
        .to_vec_i64()
        .with_context(|| {
            format!(
                "adapter segments input '{}' must be int64",
                selection.segments
            )
        })?;
    let adapter_counts = values
        .get(&selection.adapter_counts)
        .with_context(|| {
            format!(
                "adapter counts input '{}' is absent",
                selection.adapter_counts
            )
        })?
        .to_vec_i64()
        .with_context(|| {
            format!(
                "adapter counts input '{}' must be int64",
                selection.adapter_counts
            )
        })?;
    let scales = values
        .get(&selection.scales)
        .with_context(|| format!("adapter scales input '{}' is absent", selection.scales))?
        .to_vec_f32()
        .with_context(|| {
            format!(
                "adapter scales input '{}' must be float32",
                selection.scales
            )
        })?;
    let expected_slots = batch_rows
        .checked_mul(max_adapters)
        .context("adapter selection shape overflows usize")?;
    anyhow::ensure!(
        segments.len() == expected_slots && scales.len() == expected_slots,
        "adapter segments and scales must each contain batch * max_adapters = {expected_slots} values"
    );
    anyhow::ensure!(
        adapter_counts.len() == batch_rows,
        "adapter counts has {} rows but the batch has {batch_rows}",
        adapter_counts.len()
    );
    let aliases = service
        .artifacts
        .iter()
        .map(|(alias, artifact)| (artifact.index as i64, alias.as_str()))
        .collect::<HashMap<_, _>>();
    let mut rows = Vec::with_capacity(batch_rows);
    for (physical_row, declared_count) in adapter_counts.iter().enumerate().take(batch_rows) {
        let count = usize::try_from(*declared_count).with_context(|| {
            format!("adapter count for row {physical_row} must be non-negative")
        })?;
        anyhow::ensure!(
            count <= max_adapters,
            "adapter count {count} for row {physical_row} exceeds max_adapters {max_adapters}"
        );
        let mut activations = Vec::with_capacity(count);
        for slot in 0..max_adapters {
            let offset = physical_row * max_adapters + slot;
            if slot < count {
                let id = segments[offset];
                let alias = aliases.get(&id).with_context(|| {
                    format!("adapter ID {id} for row {physical_row} slot {slot} is undeclared")
                })?;
                activations.push(AdapterActivation::new(*alias, scales[offset]));
            } else {
                anyhow::ensure!(
                    segments[offset] == -1 && scales[offset] == 0.0,
                    "unused adapter slot {slot} for row {physical_row} must be padded with ID -1 and scale 0"
                );
            }
        }
        rows.push(activations);
    }
    Ok(AdapterSelection { rows })
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AdapterLifecycleDiagnostic {
    pub loads: u64,
    pub cache_hits: u64,
    pub evictions: u64,
    pub reloads: u64,
    pub capture_invalidations: u64,
    pub plan_variants: usize,
    pub replayed_plans: u64,
    pub cached: Vec<String>,
    pub active_plan_key: Option<String>,
    pub capability: Option<String>,
    pub portable_fallback: bool,
}

#[derive(Debug, Clone)]
pub(super) struct AdapterTargetWeights {
    pub input_features: usize,
    pub output_features: usize,
    pub output_offset: usize,
    pub output_width: usize,
    pub rank: usize,
    pub factor: f32,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

struct LoadedAdapter {
    targets: HashMap<(String, String), Vec<AdapterTargetWeights>>,
    _lease: MemoryLease,
    peft_config_identity: Option<[u8; 32]>,
}

#[derive(Debug, Deserialize)]
struct JsonBundle {
    targets: HashMap<String, JsonTarget>,
}

#[derive(Debug, Deserialize)]
struct JsonTarget {
    a: Vec<f32>,
    b: Vec<f32>,
}

#[derive(Default)]
pub(super) struct AdapterCache {
    entries: HashMap<String, LoadedAdapter>,
    authority: Option<Rc<dyn MemoryGovernor>>,
    lru: VecDeque<String>,
    previously_loaded: std::collections::HashSet<String>,
    plans: VecDeque<String>,
    diagnostic: AdapterLifecycleDiagnostic,
}

/// Resolved adapter composition for each batch row, in row order.
///
/// This is row-scoped native state, so it implements the mandatory
/// [`RowScopedState`] ABI: a scheduler that compacts the batch moves adapter
/// compositions with it, and a finished request releases its row.
#[derive(Debug, Clone, Default)]
pub(super) struct AdapterRunContext {
    pub rows: Vec<Vec<(String, f32)>>,
}

impl RowScopedState for AdapterRunContext {
    fn rows(&self) -> usize {
        self.rows.len()
    }

    fn compact(&mut self, selection: &[usize]) -> anyhow::Result<()> {
        self.rows = gather_rows(&self.rows, selection).context("adapter run context compaction")?;
        Ok(())
    }

    fn release(&mut self, row: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            row < self.rows.len(),
            "cannot release adapter row {row}; only {} rows exist",
            self.rows.len()
        );
        self.rows.remove(row);
        Ok(())
    }
}

impl AdapterCache {
    pub(super) fn new(authority: Rc<dyn MemoryGovernor>) -> Self {
        Self {
            authority: Some(authority),
            ..Self::default()
        }
    }
    pub(super) fn reserve(&self, bytes: u64, holder: Holder) -> anyhow::Result<MemoryLease> {
        self.authority.as_deref().context("adapter execution has no host-memory authority; reload the package with its model governor")?
            .reserve(onnx_runtime_memory_api::Tier::Host,bytes,holder.role(),holder.id()).map_err(Into::into)
    }
    pub(super) fn projection_rank(
        &self,
        context: &AdapterRunContext,
        component: &str,
        parameter: &str,
    ) -> anyhow::Result<usize> {
        let mut rank = 0;
        for row in &context.rows {
            for (alias, _) in row {
                for target in self.targets(alias, component, parameter)? {
                    rank = rank.max(target.rank);
                }
            }
        }
        Ok(rank)
    }
    pub fn diagnostic(&self) -> AdapterLifecycleDiagnostic {
        let mut diagnostic = self.diagnostic.clone();
        diagnostic.cached = self.lru.iter().cloned().collect();
        diagnostic
    }

    pub fn prepare(
        &mut self,
        root: &Path,
        service: &AdapterServiceContract,
        selection: &AdapterSelection,
        active_rows: &[bool],
    ) -> anyhow::Result<AdapterRunContext> {
        if active_rows.len() != selection.rows.len() {
            anyhow::bail!(
                "adapter active mask has {} rows but the selection has {}",
                active_rows.len(),
                selection.rows.len()
            );
        }
        let requested = selection
            .rows
            .iter()
            .zip(active_rows)
            .filter(|(_, active)| **active)
            .flat_map(|(activations, _)| activations)
            .filter(|activation| activation.scale != 0.0)
            .map(|activation| activation.adapter.as_str())
            .collect::<std::collections::HashSet<_>>();
        if requested.len() > service.cache.max_entries {
            anyhow::bail!(
                "request activates {} distinct adapters but the adapter cache holds only {}; increase adapters.cache.max_entries",
                requested.len(),
                service.cache.max_entries
            );
        }
        let mut rows = Vec::with_capacity(selection.rows.len());
        let mut plan_parts = Vec::with_capacity(selection.rows.len());
        for (physical_row, row_selection) in selection.rows.iter().enumerate() {
            let activations = if active_rows[physical_row] {
                row_selection.clone()
            } else {
                Vec::new()
            };
            let mut resolved = Vec::with_capacity(activations.len());
            let mut row_key = Vec::with_capacity(activations.len());
            let mut row_adapters = std::collections::HashSet::new();
            for activation in activations {
                if !row_adapters.insert(activation.adapter.clone()) {
                    anyhow::bail!(
                        "adapter selection for batch row {physical_row} contains duplicate \
                         adapter '{}'",
                        activation.adapter
                    );
                }
                if !activation.scale.is_finite() {
                    anyhow::bail!(
                        "adapter '{}' scale {} for batch row {physical_row} must be finite",
                        activation.adapter,
                        activation.scale
                    );
                }
                let artifact = service
                    .artifacts
                    .get(&activation.adapter)
                    .with_context(|| {
                        format!(
                            "adapter selection for batch row {physical_row} references \
                             undeclared adapter '{}'",
                            activation.adapter
                        )
                    })?;
                if activation.scale == 0.0 {
                    continue;
                }
                self.ensure_loaded(root, service, &activation.adapter, artifact)?;
                let scale = activation.scale;
                if !scale.is_finite() {
                    anyhow::bail!(
                        "adapter '{}' effective scale {scale} for batch row {physical_row} must be finite",
                        activation.adapter
                    );
                }
                row_key.push((
                    activation.adapter.clone(),
                    scale.to_bits(),
                    self.entries
                        .get(&activation.adapter)
                        .and_then(|entry| entry.peft_config_identity),
                ));
                resolved.push((activation.adapter, scale));
            }
            plan_parts.push(row_key);
            rows.push(resolved);
        }
        let plan_key = serde_json::to_string(&(&service.application_capability, plan_parts))?;
        if let Some(index) = self.plans.iter().position(|key| key == &plan_key) {
            self.diagnostic.replayed_plans += 1;
            self.plans.remove(index);
        } else if self.plans.len() >= service.cache.max_entries {
            self.plans.pop_front();
        }
        self.plans.push_back(plan_key.clone());
        self.diagnostic.plan_variants = self.plans.len();
        self.diagnostic.active_plan_key = Some(plan_key.clone());
        self.diagnostic.capability = Some(service.application_capability.clone());
        self.diagnostic.portable_fallback = service.portable_fallback;
        Ok(AdapterRunContext { rows })
    }

    fn ensure_loaded(
        &mut self,
        root: &Path,
        service: &AdapterServiceContract,
        name: &str,
        artifact: &AdapterArtifact,
    ) -> anyhow::Result<()> {
        if let Some(loaded) = self.entries.get(name) {
            if let Some(expected) = loaded.peft_config_identity {
                let weight = artifact
                    .weights
                    .iter()
                    .find(|weight| {
                        matches!(
                            weight.format,
                            AdapterWeightFormat::Json | AdapterWeightFormat::HfPeft
                        )
                    })
                    .context("cached adapter source disappeared")?;
                let authority = self
                    .authority
                    .as_deref()
                    .context("PEFT cache requires host-memory authority")?;
                let (configuration, _lease) = peft::configuration(root, weight, authority)?;
                anyhow::ensure!(
                    peft::identity(&configuration) == expected,
                    "PEFT adapter {name:?} configuration changed after loading; restore its immutable config or reload the package with a new adapter version before selecting it"
                );
            }
            self.diagnostic.cache_hits += 1;
            self.touch(name);
            return Ok(());
        }
        if self.entries.len() == service.cache.max_entries
            && let Some(evicted) = self.lru.pop_front()
        {
            self.entries.remove(&evicted);
            self.diagnostic.evictions += 1;
            if service.planning.invalidate_capture_on_eviction {
                self.diagnostic.capture_invalidations += 1;
                self.plans.clear();
            }
        }
        let authority=self.authority.as_deref().context("adapter weight cache has no host-memory authority; construct the runtime with its model governor")?;
        let loaded = load_adapter(root, service, artifact, authority)?;
        if !self.previously_loaded.insert(name.to_string()) {
            self.diagnostic.reloads += 1;
        }
        self.diagnostic.loads += 1;
        self.entries.insert(name.to_string(), loaded);
        self.lru.push_back(name.to_string());
        Ok(())
    }

    fn touch(&mut self, name: &str) {
        if let Some(index) = self.lru.iter().position(|candidate| candidate == name) {
            self.lru.remove(index);
        }
        self.lru.push_back(name.to_string());
    }

    pub fn targets(
        &self,
        adapter: &str,
        component: &str,
        parameter: &str,
    ) -> anyhow::Result<&[AdapterTargetWeights]> {
        let loaded = self
            .entries
            .get(adapter)
            .with_context(|| format!("adapter '{adapter}' is not loaded"))?;
        // An adapter may target only some declared projections. Missing patches
        // preserve the actual base output rather than inventing a zero result.
        Ok(loaded
            .targets
            .get(&(component.to_string(), parameter.to_string()))
            .map(Vec::as_slice)
            .unwrap_or(&[]))
    }
}

/// The tensor owns its host buffer and its credit together, including through aliases.
pub(super) fn governed_value(value: Value, lease: Arc<MemoryLease>) -> anyhow::Result<Value> {
    value.with_owner(lease).map_err(Into::into)
}

pub(super) struct ProjectionInputs<'a> {
    pub source: &'a [f32],
    pub base: &'a [f32],
    pub batch: usize,
    pub input_features: usize,
    pub output_features: usize,
}

pub(super) fn apply_parameter_overlay(
    cache: &AdapterCache,
    context: &AdapterRunContext,
    component: &str,
    parameter: &str,
    inputs: ProjectionInputs<'_>,
) -> anyhow::Result<(Vec<f32>, usize)> {
    let ProjectionInputs {
        source,
        base,
        batch,
        input_features,
        output_features,
    } = inputs;
    anyhow::ensure!(
        context.rows.len() == batch,
        "adapter selection has {} rows but projection has batch {batch}",
        context.rows.len()
    );
    anyhow::ensure!(
        batch.checked_mul(input_features) == Some(source.len()),
        "adapter activation shape does not match batch and input features"
    );
    anyhow::ensure!(
        batch.checked_mul(output_features) == Some(base.len()),
        "adapter base projection shape does not match batch and output features"
    );
    let mut result = base.to_vec();
    for (row, activations) in context.rows.iter().enumerate() {
        for (adapter, scale) in activations {
            for target in cache.targets(adapter, component, parameter)? {
                anyhow::ensure!(
                    target.input_features == input_features
                        && target.output_features == output_features,
                    "adapter '{adapter}' target '{component}.{parameter}' expects [{}, {}], got [{input_features}, {output_features}]",
                    target.input_features,
                    target.output_features
                );
                let factor = *scale * target.factor;
                anyhow::ensure!(
                    factor.is_finite(),
                    "adapter '{adapter}' scale times alpha/rank exceeds finite float32; reduce the requested scale"
                );
                let x = &source[row * input_features..(row + 1) * input_features];
                let mut low_rank = vec![0.0_f32; target.rank];
                for (rank, value) in low_rank.iter_mut().enumerate() {
                    for (feature, input) in x.iter().enumerate() {
                        *value += input * target.a[rank * input_features + feature];
                    }
                }
                for output in 0..target.output_width {
                    let mut delta = 0.0_f32;
                    for (rank, value) in low_rank.iter().enumerate() {
                        delta += target.b[output * target.rank + rank] * value;
                    }
                    result[row * output_features + target.output_offset + output] += factor * delta;
                }
            }
        }
    }
    Ok((result, output_features))
}

fn load_adapter(
    root: &Path,
    service: &AdapterServiceContract,
    artifact: &AdapterArtifact,
    authority: &dyn MemoryGovernor,
) -> anyhow::Result<LoadedAdapter> {
    let weight = artifact
        .weights
        .iter()
        .find(|weight| matches!(weight.format, AdapterWeightFormat::Json | AdapterWeightFormat::HfPeft))
        .with_context(|| {
            format!(
                "portable adapter fallback has no JSON or HF PEFT artifact for {}@{}; use an execution provider with '{}' support",
                artifact.identity, artifact.version, service.application_capability
            )
        })?;
    anyhow::ensure!(
        weight.format == AdapterWeightFormat::HfPeft
            || matches!(artifact.dtype.as_str(), "float32" | "fp32"),
        "portable JSON adapter normalization requires float32, but {}@{} declares {}; use F32 JSON factors or an explicitly typed HF PEFT safetensors artifact",
        artifact.identity,
        artifact.version,
        artifact.dtype
    );
    let package_root = root
        .canonicalize()
        .with_context(|| format!("failed to resolve package root '{}'", root.display()))?;
    let path = root
        .join(&weight.location)
        .canonicalize()
        .with_context(|| {
            format!(
                "failed to resolve adapter artifact '{}'",
                root.join(&weight.location).display()
            )
        })?;
    if !path.starts_with(&package_root) {
        anyhow::bail!(
            "adapter artifact '{}' resolves outside package root '{}'",
            path.display(),
            package_root.display()
        );
    }
    let file = fs::File::open(&path)
        .with_context(|| format!("open adapter artifact '{}'", path.display()))?;
    let metadata = file
        .metadata()
        .context("inspect adapter artifact before bounded loading")?;
    anyhow::ensure!(
        metadata.is_file(),
        "adapter artifact '{}' must be a regular file",
        path.display()
    );
    let length = metadata.len();
    let transient = length
        .checked_mul(16)
        .context("adapter container staging size overflow; reduce the artifact")?;
    let _staging = authority
        .reserve(
            onnx_runtime_memory_api::Tier::Host,
            transient,
            Holder::CpuDerivedWeights.role(),
            Holder::CpuDerivedWeights.id(),
        )
        .with_context(|| {
            format!(
                "reserve adapter container parsing memory for '{}'",
                path.display()
            )
        })?;
    let mut live_bytes = 0_u64;
    for binding in &artifact.bindings {
        let target = service
            .target_manifest
            .targets
            .iter()
            .find(|target| target.id == binding.target)
            .with_context(|| {
                format!(
                    "adapter binding targets undeclared projection '{}'",
                    binding.target
                )
            })?;
        let rank = u64::try_from(binding.rank.unwrap_or(artifact.rank))?;
        let weights = u64::try_from(target.input_features)?
            .checked_add(u64::try_from(
                target
                    .output_slice
                    .as_ref()
                    .map_or(target.output_features, |slice| slice.width),
            )?)
            .and_then(|n| n.checked_mul(rank))
            .and_then(|n| n.checked_mul(4))
            .context("adapter target tensor byte count overflow")?;
        let names = u64::try_from(target.component.len())?
            .checked_add(u64::try_from(target.initializer.len())?)
            .and_then(|n| n.checked_add(512))
            .context("adapter target descriptor size overflow")?;
        live_bytes = live_bytes
            .checked_add(weights)
            .and_then(|n| n.checked_add(names))
            .context("adapter weight cache size overflow")?;
    }
    let lease = authority
        .reserve(
            onnx_runtime_memory_api::Tier::Host,
            live_bytes,
            Holder::CpuDerivedWeights.role(),
            Holder::CpuDerivedWeights.id(),
        )
        .with_context(|| {
            format!(
                "reserve normalized adapter weights for {}@{}",
                artifact.identity, artifact.version
            )
        })?;
    use std::io::Read;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(
            usize::try_from(length)?
                .checked_add(1)
                .context("adapter file length overflow")?,
        )
        .context("allocate bounded adapter container input; free host memory and retry")?;
    file.take(
        length
            .checked_add(1)
            .context("adapter file length overflow")?,
    )
    .read_to_end(&mut bytes)
    .with_context(|| format!("read bounded adapter artifact '{}'", path.display()))?;
    anyhow::ensure!(
        bytes.len() as u64 == length,
        "adapter artifact '{}' changed size while loading; replace model artifacts atomically between loads",
        path.display()
    );
    let (bundles, peft_config_identity) = match weight.format {
        AdapterWeightFormat::Json => {
            let bundle: JsonBundle = serde_json::from_slice(&bytes).with_context(|| {
                format!("adapter artifact '{}' is invalid JSON", path.display())
            })?;
            (bundle.targets, None)
        }
        AdapterWeightFormat::HfPeft => {
            let (targets, identity) =
                peft::load(root, weight, artifact, service, &bytes, authority)?;
            (targets, Some(identity))
        }
        _ => unreachable!("supported artifact format selected above"),
    };
    let mut targets = HashMap::new();
    for binding in &artifact.bindings {
        let target = service
            .target_manifest
            .targets
            .iter()
            .find(|target| target.id == binding.target)
            .with_context(|| {
                format!(
                    "adapter {}@{} binding references undeclared target '{}'",
                    artifact.identity, artifact.version, binding.target
                )
            })?;
        let tensors = bundles.get(&binding.weight_key).with_context(|| {
            format!(
                "adapter {}@{} artifact has no target key '{}'",
                artifact.identity, artifact.version, binding.weight_key
            )
        })?;
        let rank = binding.rank.unwrap_or(artifact.rank);
        let alpha = binding.alpha.unwrap_or(artifact.alpha);
        let expected_a = rank
            .checked_mul(target.input_features)
            .context("adapter A tensor size overflow")?;
        let output_offset = target.output_slice.as_ref().map_or(0, |slice| slice.offset);
        let output_width = target
            .output_slice
            .as_ref()
            .map_or(target.output_features, |slice| slice.width);
        anyhow::ensure!(
            output_offset
                .checked_add(output_width)
                .is_some_and(|end| end <= target.output_features),
            "adapter target output slice exceeds its declared projection width"
        );
        let expected_b = output_width
            .checked_mul(rank)
            .context("adapter B tensor size overflow")?;
        if tensors.a.len() != expected_a || tensors.b.len() != expected_b {
            anyhow::bail!(
                "adapter target '{}.{}' shape mismatch: A has {} values (expected {expected_a}), B has {} values (expected {expected_b})",
                target.component,
                target.initializer,
                tensors.a.len(),
                tensors.b.len()
            );
        }
        let factor = match &weight.scale_encoding {
            AdapterScaleEncoding::AlphaOverRank => (alpha / rank as f64) as f32,
            AdapterScaleEncoding::Baked => 1.0,
        };
        anyhow::ensure!(
            factor.is_finite() && tensors.a.iter().chain(&tensors.b).all(|n| n.is_finite()),
            "adapter target '{}.{}' contains non-finite weights or alpha/rank factor; normalize finite tensors and scaling before loading",
            target.component,
            target.initializer
        );
        targets
            .entry((target.component.clone(), target.initializer.clone()))
            .or_insert_with(Vec::new)
            .push(AdapterTargetWeights {
                input_features: target.input_features,
                output_features: target.output_features,
                output_offset,
                output_width,
                rank,
                factor,
                a: tensors.a.clone(),
                b: tensors.b.clone(),
            });
    }
    Ok(LoadedAdapter {
        targets,
        _lease: lease,
        peft_config_identity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai_metadata::{
        AdapterCacheContract, AdapterDiscoveryFallback, AdapterEvictionPolicy,
        AdapterPlanningContract, AdapterProvenance, AdapterScaleEncoding, AdapterSelectionContract,
        AdapterTargetBinding, AdapterWeightArtifact, LoraTargetDescriptor, LoraTargetManifest,
    };

    fn test_cache() -> AdapterCache {
        AdapterCache::new(Rc::new(onnx_runtime_memory_governor::LedgerGovernor::new(
            onnx_runtime_memory_governor::LeaseLedger::new(0, 1024 * 1024 * 1024, 0),
        )))
    }
    fn artifact(name: &str, location: &str, _bytes: &[u8]) -> AdapterArtifact {
        AdapterArtifact {
            index: usize::from(name == "blue"),
            identity: name.to_string(),
            version: "1".to_string(),
            rank: 1,
            alpha: 1.0,
            dtype: "float32".to_string(),
            weights: vec![AdapterWeightArtifact {
                location: location.to_string(),
                loader_capability: "onnx-genai.adapters.json@1".to_string(),
                config_location: None,
                scale_encoding: AdapterScaleEncoding::AlphaOverRank,
                format: AdapterWeightFormat::Json,
            }],
            bindings: vec![AdapterTargetBinding {
                target: "projection".to_string(),
                weight_key: "projection".to_string(),
                rank: None,
                alpha: None,
            }],
            provenance: Some(AdapterProvenance {
                producer: "synthetic-test".to_string(),
                source: None,
                revision: None,
            }),
        }
    }

    fn service_contract(red: &[u8], blue: &[u8], max_entries: usize) -> AdapterServiceContract {
        AdapterServiceContract {
            target_manifest: LoraTargetManifest {
                targets: vec![LoraTargetDescriptor {
                    id: "projection".to_string(),
                    component: "decoder".to_string(),
                    initializer: "projection".to_string(),
                    layer_index: None,
                    node_name: "projection".to_string(),
                    output_name: "projection.output".to_string(),
                    activation_dtype: "float32".to_string(),
                    input_features: 2,
                    output_features: 2,
                    rank: Some(1),
                    alpha: Some(1.0),
                    output_slice: None,
                    graph_inputs: None,
                }],
            },
            discovery_fallback: AdapterDiscoveryFallback::Disabled,
            selection: AdapterSelectionContract {
                segments: "request.adapter_segments".to_string(),
                adapter_counts: "request.adapter_counts".to_string(),
                scales: "request.adapter_scales".to_string(),
                active: None,
                max_adapters: 2,
            },
            application_capability: "onnx-genai.adapters@1".to_string(),
            portable_fallback: true,
            artifacts: std::collections::BTreeMap::from([
                (
                    "red".to_string(),
                    artifact("red", "adapters/red/adapter.json", red),
                ),
                (
                    "blue".to_string(),
                    artifact("blue", "adapters/blue/adapter.json", blue),
                ),
            ]),
            cache: AdapterCacheContract {
                max_entries,
                eviction: AdapterEvictionPolicy::Lru,
            },
            planning: AdapterPlanningContract::default(),
        }
    }

    #[test]
    fn sparse_bindings_and_multiple_fused_slices_preserve_other_outputs() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        for alias in ["red", "blue"] {
            fs::create_dir_all(root.path().join("adapters").join(alias))?;
        }
        let red =
            br#"{"targets":{"head":{"a":[1.0,0.0],"b":[2.0]},"tail":{"a":[1.0,0.0],"b":[3.0]}}}"#;
        let blue = br#"{"targets":{"other":{"a":[0.0,1.0],"b":[1.0,2.0,3.0]}}}"#;
        fs::write(root.path().join("adapters/red/adapter.json"), red)?;
        fs::write(root.path().join("adapters/blue/adapter.json"), blue)?;
        let mut service = service_contract(red, blue, 2);
        let target = &mut service.target_manifest.targets[0];
        target.output_features = 3;
        target.output_slice = Some(onnx_genai_metadata::LoraTargetSlice {
            role: "first".into(),
            offset: 0,
            width: 1,
            rank: None,
            alpha: None,
        });
        let mut tail = target.clone();
        tail.id = "tail".into();
        tail.output_slice.as_mut().unwrap().role = "last".into();
        tail.output_slice.as_mut().unwrap().offset = 2;
        let mut other = target.clone();
        other.id = "other".into();
        other.initializer = "other".into();
        other.node_name = "other".into();
        other.output_name = "other.output".into();
        other.output_slice = None;
        service.target_manifest.targets.extend([tail, other]);
        let artifact = service.artifacts.get_mut("red").unwrap();
        artifact.bindings[0].weight_key = "head".into();
        artifact.bindings.push(AdapterTargetBinding {
            target: "tail".into(),
            weight_key: "tail".into(),
            rank: None,
            alpha: None,
        });
        let artifact = service.artifacts.get_mut("blue").unwrap();
        artifact.bindings[0].target = "other".into();
        artifact.bindings[0].weight_key = "other".into();
        let mut cache = test_cache();
        let context = cache.prepare(
            root.path(),
            &service,
            &AdapterSelection::default().with_row([
                AdapterActivation::new("red", 1.0),
                AdapterActivation::new("blue", 1.0),
            ]),
            &[true],
        )?;
        let inputs = || ProjectionInputs {
            source: &[1.0, 2.0],
            base: &[12.0, 15.0, 18.0],
            batch: 1,
            input_features: 2,
            output_features: 3,
        };
        assert_eq!(
            apply_parameter_overlay(&cache, &context, "decoder", "projection", inputs())?.0,
            vec![14.0, 15.0, 21.0]
        );
        assert_eq!(
            apply_parameter_overlay(&cache, &context, "decoder", "other", inputs())?.0,
            vec![14.0, 19.0, 24.0]
        );
        Ok(())
    }

    #[test]
    fn composition_history_is_bounded_and_preserves_exact_float_scale_bits() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        fs::create_dir_all(root.path().join("adapters/red"))?;
        let data = br#"{"targets":{"projection":{"a":[1.0,0.0],"b":[1.0,2.0]}}}"#;
        fs::write(root.path().join("adapters/red/adapter.json"), data)?;
        let service = service_contract(data, data, 2);
        let mut cache = test_cache();
        let small = 1e-9_f32;
        for scale in [small, f32::from_bits(small.to_bits() + 1)] {
            cache.prepare(
                root.path(),
                &service,
                &AdapterSelection::default().with_row([AdapterActivation::new("red", scale)]),
                &[true],
            )?;
        }
        assert_eq!(cache.diagnostic().replayed_plans, 0);
        for step in 1..32 {
            cache.prepare(
                root.path(),
                &service,
                &AdapterSelection::default().with_row([AdapterActivation::new("red", step as f32)]),
                &[true],
            )?;
        }
        assert_eq!(cache.diagnostic().plan_variants, 2);
        let mut cold = test_cache();
        fs::remove_file(root.path().join("adapters/red/adapter.json"))?;
        let disabled = cold.prepare(
            root.path(),
            &service,
            &AdapterSelection::default().with_row([AdapterActivation::new("red", 0.0)]),
            &[true],
        )?;
        assert!(disabled.rows[0].is_empty());
        assert_eq!(cold.diagnostic().loads, 0);
        Ok(())
    }

    #[test]
    fn governed_tensor_aliases_keep_host_credit_until_last_view() -> anyhow::Result<()> {
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{LeaseLedger, LedgerGovernor};
        let ledger = LeaseLedger::new(0, 64, 0);
        let governor = LedgerGovernor::new(ledger);
        let lease = Arc::new(governor.reserve(
            Tier::Host,
            8,
            Holder::Activations.role(),
            Holder::Activations.id(),
        )?);
        let value = governed_value(Value::from_vec_f32(vec![2.0, 3.0], &[2])?, lease)?;
        let alias = value
            .try_alias_clone()
            .expect("governed tensors are shared views")?;
        drop(value);
        assert_eq!(governor.used(Tier::Host), 8);
        assert_eq!(alias.to_vec_f32()?, vec![2.0, 3.0]);
        drop(alias);
        assert_eq!(governor.used(Tier::Host), 0);
        Ok(())
    }
    #[test]
    fn adapter_weight_cache_reserves_and_releases_host_memory() -> anyhow::Result<()> {
        use onnx_runtime_memory_api::{MemoryError, Tier};
        use onnx_runtime_memory_governor::{LeaseLedger, LedgerGovernor};
        let root = tempfile::tempdir()?;
        fs::create_dir_all(root.path().join("adapters/red"))?;
        let data = br#"{"targets":{"projection":{"a":[1.0,0.0],"b":[1.0,2.0]}}}"#;
        fs::write(root.path().join("adapters/red/adapter.json"), data)?;
        let service = service_contract(data, data, 2);
        let selection = AdapterSelection::default().with_row([AdapterActivation::new("red", 1.0)]);
        let governor = Rc::new(LedgerGovernor::new(LeaseLedger::new(0, 64, 0)));
        let mut cache = AdapterCache::new(governor.clone());
        let error = cache
            .prepare(root.path(), &service, &selection, &[true])
            .unwrap_err();
        assert!(error.downcast_ref::<MemoryError>().is_some(), "{error:#}");
        assert_eq!(governor.used(Tier::Host), 0);
        assert_eq!(cache.diagnostic().loads, 0);
        let governor = Rc::new(LedgerGovernor::new(LeaseLedger::new(0, 1024 * 1024, 0)));
        let mut cache = AdapterCache::new(governor.clone());
        cache.prepare(root.path(), &service, &selection, &[true])?;
        assert!(governor.used(Tier::Host) >= 16);
        drop(cache);
        assert_eq!(governor.used(Tier::Host), 0);
        Ok(())
    }

    #[test]
    fn overlay_uses_actual_non_identity_non_square_base_projection() -> anyhow::Result<()> {
        let red = br#"{"targets":{"projection":{"a":[1.0,0.0],"b":[1.0,2.0,3.0]}}}"#;
        let mut service = service_contract(red, red, 2);
        service.target_manifest.targets[0].output_features = 3;
        let directory = tempfile::tempdir()?;
        let root = directory.path().to_path_buf();
        fs::create_dir_all(root.join("adapters/red"))?;
        fs::write(root.join("adapters/red/adapter.json"), red)?;
        let mut cache = test_cache();
        let context = cache.prepare(
            &root,
            &service,
            &AdapterSelection::default().with_row([AdapterActivation::new("red", 0.5)]),
            &[true],
        )?;
        // [1,2] * [[2,3,4],[5,6,7]] = [12,15,18], independently of LoRA.
        let input = ProjectionInputs {
            source: &[1.0, 2.0],
            base: &[12.0, 15.0, 18.0],
            batch: 1,
            input_features: 2,
            output_features: 3,
        };
        assert_eq!(
            apply_parameter_overlay(&cache, &context, "decoder", "projection", input)?.0,
            vec![12.5, 16.0, 19.5]
        );
        let empty = AdapterRunContext { rows: vec![vec![]] };
        let input = ProjectionInputs {
            source: &[1.0, 2.0],
            base: &[12.0, 15.0, 18.0],
            batch: 1,
            input_features: 2,
            output_features: 3,
        };
        assert_eq!(
            apply_parameter_overlay(&cache, &empty, "decoder", "projection", input)?.0,
            vec![12.0, 15.0, 18.0]
        );
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn one_artifact_supports_heterogeneous_target_rank_and_alpha() {
        let root = std::env::current_dir()
            .expect("current directory")
            .join("target")
            .join(format!("adapter-heterogeneous-rank-{}", std::process::id()));
        fs::create_dir_all(root.join("adapters/red")).expect("create adapter test directory");
        let red = br#"{"targets":{"projection":{"a":[1.0,0.0],"b":[1.0,2.0]},"projection_2":{"a":[1.0,0.0,0.0,1.0],"b":[1.0,0.0,0.0,1.0]}}}"#;
        let blue = br#"{"targets":{"projection":{"a":[0.0,1.0],"b":[3.0,4.0]}}}"#;
        fs::write(root.join("adapters/red/adapter.json"), red).expect("write adapter");

        let mut service = service_contract(red, blue, 1);
        service.target_manifest.targets[0].alpha = Some(2.0);
        service.target_manifest.targets.push(LoraTargetDescriptor {
            id: "projection_2".to_string(),
            component: "decoder".to_string(),
            initializer: "projection_2".to_string(),
            layer_index: Some(1),
            node_name: "projection_2".to_string(),
            output_name: "projection_2.output".to_string(),
            activation_dtype: "float32".to_string(),
            input_features: 2,
            output_features: 2,
            rank: Some(2),
            alpha: Some(6.0),
            output_slice: None,
            graph_inputs: None,
        });
        let artifact = service.artifacts.get_mut("red").expect("red artifact");
        artifact.bindings[0].alpha = Some(2.0);
        artifact.bindings.push(AdapterTargetBinding {
            target: "projection_2".to_string(),
            weight_key: "projection_2".to_string(),
            rank: Some(2),
            alpha: Some(6.0),
        });

        let selection = AdapterSelection::default().with_row([AdapterActivation::new("red", 1.0)]);
        let mut cache = test_cache();
        let context = cache
            .prepare(&root, &service, &selection, &[true])
            .expect("load heterogeneous-rank artifact under one alias");
        let (rank_one, _) = apply_parameter_overlay(
            &cache,
            &context,
            "decoder",
            "projection",
            ProjectionInputs {
                source: &[1.0, 2.0],
                base: &[1.0, 2.0],
                batch: 1,
                input_features: 2,
                output_features: 2,
            },
        )
        .expect("apply rank-one target");
        let (rank_two, _) = apply_parameter_overlay(
            &cache,
            &context,
            "decoder",
            "projection_2",
            ProjectionInputs {
                source: &[1.0, 2.0],
                base: &[1.0, 2.0],
                batch: 1,
                input_features: 2,
                output_features: 2,
            },
        )
        .expect("apply rank-two target");
        assert_eq!(rank_one, vec![3.0, 6.0]);
        assert_eq!(rank_two, vec![4.0, 8.0]);

        fs::remove_dir_all(root).expect("remove adapter test directory");
    }

    #[test]
    fn heterogeneous_composition_compaction_and_cache_lifecycle() {
        let root = std::env::current_dir()
            .expect("current directory")
            .join("target")
            .join(format!("adapter-test-{}", std::process::id()));
        fs::create_dir_all(root.join("adapters/red")).expect("create red adapter test directory");
        fs::create_dir_all(root.join("adapters/blue")).expect("create blue adapter test directory");
        let red = br#"{"targets":{"projection":{"a":[1.0,0.0],"b":[1.0,2.0]}}}"#;
        let blue = br#"{"targets":{"projection":{"a":[0.0,1.0],"b":[3.0,4.0]}}}"#;
        fs::write(root.join("adapters/red/adapter.json"), red).expect("write red adapter");
        fs::write(root.join("adapters/blue/adapter.json"), blue).expect("write blue adapter");

        let service = service_contract(red, blue, 2);
        // Rows are positional: row 0 runs red, row 1 runs the base model, and
        // row 2 composes red and blue.
        let selection = AdapterSelection::default()
            .with_row([AdapterActivation::new("red", 1.0)])
            .with_row([])
            .with_row([
                AdapterActivation::new("red", 0.5),
                AdapterActivation::new("blue", 1.0),
            ]);
        let mut cache = test_cache();
        let context = cache
            .prepare(&root, &service, &selection, &[true, true, true])
            .expect("prepare heterogeneous adapters");
        let (output, width) = apply_parameter_overlay(
            &cache,
            &context,
            "decoder",
            "projection",
            ProjectionInputs {
                source: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                base: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                batch: 3,
                input_features: 2,
                output_features: 2,
            },
        )
        .expect("apply heterogeneous adapters");
        assert_eq!(width, 2);
        assert_eq!(output, vec![2.0, 4.0, 3.0, 4.0, 25.5, 35.0]);
        for (row, row_input, expected) in [
            (0, [1.0, 2.0], [2.0, 4.0]),
            (1, [3.0, 4.0], [3.0, 4.0]),
            (2, [5.0, 6.0], [25.5, 35.0]),
        ] {
            let isolated =
                AdapterSelection::default().with_row(selection.rows[row].iter().cloned());
            let single = cache
                .prepare(&root, &service, &isolated, &[true])
                .expect("prepare independent row");
            let (single_output, _) = apply_parameter_overlay(
                &cache,
                &single,
                "decoder",
                "projection",
                ProjectionInputs {
                    source: &row_input,
                    base: &row_input,
                    batch: 1,
                    input_features: 2,
                    output_features: 2,
                },
            )
            .expect("apply independent row");
            assert_eq!(single_output, expected);
        }

        let dynamic = AdapterSelection::default().with_row([AdapterActivation::new("red", 0.5)]);
        let dynamic_context = cache
            .prepare(&root, &service, &dynamic, &[true])
            .expect("prepare dynamic scale");
        let (dynamic_output, _) = apply_parameter_overlay(
            &cache,
            &dynamic_context,
            "decoder",
            "projection",
            ProjectionInputs {
                source: &[1.0, 2.0],
                base: &[1.0, 2.0],
                batch: 1,
                input_features: 2,
                output_features: 2,
            },
        )
        .expect("apply dynamic scale");
        assert_eq!(dynamic_output, vec![1.5, 3.0]);

        // Compaction is a positional permutation through the mandatory
        // RowScopedState ABI: no adapter identity is ever serialized, and the
        // composition follows its row to the new position.
        let mut compacted = context.clone();
        compacted
            .compact(&[2, 0])
            .expect("compaction keeps each row's composition");
        let (compacted_output, _) = apply_parameter_overlay(
            &cache,
            &compacted,
            "decoder",
            "projection",
            ProjectionInputs {
                source: &[5.0, 6.0, 1.0, 2.0],
                base: &[5.0, 6.0, 1.0, 2.0],
                batch: 2,
                input_features: 2,
                output_features: 2,
            },
        )
        .expect("apply compacted adapters");
        assert_eq!(compacted_output, vec![25.5, 35.0, 2.0, 4.0]);

        // Releasing a finished row leaves the survivors in order.
        let mut released = context.clone();
        released.release(1).expect("release the base row");
        assert_eq!(released.rows(), 2);
        let (released_output, _) = apply_parameter_overlay(
            &cache,
            &released,
            "decoder",
            "projection",
            ProjectionInputs {
                source: &[1.0, 2.0, 5.0, 6.0],
                base: &[1.0, 2.0, 5.0, 6.0],
                batch: 2,
                input_features: 2,
                output_features: 2,
            },
        )
        .expect("apply after release");
        assert_eq!(released_output, vec![2.0, 4.0, 25.5, 35.0]);

        let mut inactive_cache = test_cache();
        let inactive_selection =
            AdapterSelection::default().with_row([AdapterActivation::new("red", 1.0)]);
        let inactive = inactive_cache
            .prepare(&root, &service, &inactive_selection, &[false])
            .expect("prepare inactive adapter row");
        let (inactive_output, _) = apply_parameter_overlay(
            &inactive_cache,
            &inactive,
            "decoder",
            "projection",
            ProjectionInputs {
                source: &[1.0, 2.0],
                base: &[1.0, 2.0],
                batch: 1,
                input_features: 2,
                output_features: 2,
            },
        )
        .expect("inactive row uses immutable base");
        assert_eq!(inactive_output, vec![1.0, 2.0]);
        assert_eq!(inactive_cache.diagnostic().loads, 0);

        let duplicate = AdapterSelection::default().with_row([
            AdapterActivation::new("red", 1.0),
            AdapterActivation::new("red", 0.5),
        ]);
        let error = cache
            .prepare(&root, &service, &duplicate, &[true])
            .expect_err("duplicate row adapter must fail");
        assert!(error.to_string().contains("duplicate adapter"));

        // A one-entry cache evicts, reloads, and invalidates captured plan variants.
        let service = service_contract(red, blue, 1);
        let mut cache = test_cache();
        for adapter in ["red", "blue", "red"] {
            let selection =
                AdapterSelection::default().with_row([AdapterActivation::new(adapter, 1.0)]);
            cache
                .prepare(&root, &service, &selection, &[true])
                .expect("prepare adapter lifecycle request");
        }
        let diagnostic = cache.diagnostic();
        assert_eq!(diagnostic.loads, 3);
        assert_eq!(diagnostic.evictions, 2);
        assert_eq!(diagnostic.reloads, 1);
        assert_eq!(diagnostic.capture_invalidations, 2);

        fs::remove_dir_all(root).expect("remove adapter test directory");
    }

    #[test]
    fn wire_selection_is_strict_ordered_ssa() {
        let red = br#"{"targets":{"projection":{"a":[1.0,0.0],"b":[1.0,2.0]}}}"#;
        let blue = br#"{"targets":{"projection":{"a":[0.0,1.0],"b":[3.0,4.0]}}}"#;
        let service = service_contract(red, blue, 2);
        let values = HashMap::from([
            (
                "request.adapter_segments".to_string(),
                Value::from_slice_i64(&[0, -1, 1, 0], &[2, 2]).expect("adapter IDs"),
            ),
            (
                "request.adapter_counts".to_string(),
                Value::from_slice_i64(&[1, 2], &[2]).expect("adapter counts"),
            ),
            (
                "request.adapter_scales".to_string(),
                Value::from_slice_f32(&[0.5, 0.0, 1.0, -0.25], &[2, 2]).expect("adapter scales"),
            ),
        ]);
        let selection = selection_from_inputs(&service, &values, 2).expect("selection");
        assert_eq!(selection.rows[0], [AdapterActivation::new("red", 0.5)]);
        assert_eq!(
            selection.rows[1],
            [
                AdapterActivation::new("blue", 1.0),
                AdapterActivation::new("red", -0.25),
            ]
        );

        let invalid_padding = HashMap::from([
            (
                "request.adapter_segments".to_string(),
                Value::from_slice_i64(&[0, 1], &[1, 2]).expect("adapter IDs"),
            ),
            (
                "request.adapter_counts".to_string(),
                Value::from_slice_i64(&[1], &[1]).expect("adapter counts"),
            ),
            (
                "request.adapter_scales".to_string(),
                Value::from_slice_f32(&[1.0, 0.0], &[1, 2]).expect("adapter scales"),
            ),
        ]);
        let error = selection_from_inputs(&service, &invalid_padding, 1)
            .expect_err("non-canonical padding must fail");
        assert!(error.to_string().contains("must be padded with ID -1"));
    }

    #[test]
    fn artifact_replacement_is_allowed_but_shape_is_enforced() {
        let root = std::env::current_dir()
            .expect("current directory")
            .join("target")
            .join(format!("adapter-invalid-test-{}", std::process::id()));
        fs::create_dir_all(root.join("adapters/red")).expect("create red adapter test directory");
        fs::create_dir_all(root.join("adapters/blue")).expect("create blue adapter test directory");
        let valid = br#"{"targets":{"projection":{"a":[1.0,0.0],"b":[1.0,2.0]}}}"#;
        let replacement =
            br#"{ "targets": { "projection": { "a": [1.0, 0.0], "b": [1.0, 2.0] } } }"#;
        fs::write(root.join("adapters/red/adapter.json"), replacement)
            .expect("write replacement adapter");
        fs::write(root.join("adapters/blue/adapter.json"), valid).expect("write blue adapter");
        let service = service_contract(valid, valid, 2);
        let selection = AdapterSelection::default().with_row([AdapterActivation::new("red", 1.0)]);
        test_cache()
            .prepare(&root, &service, &selection, &[true])
            .expect("semantically compatible replacement bytes must load");

        let malformed = br#"{"targets":{"projection":{"a":[1.0],"b":[1.0,2.0]}}}"#;
        fs::write(root.join("adapters/red/adapter.json"), malformed)
            .expect("write malformed adapter");
        let service = service_contract(malformed, valid, 2);
        let error = test_cache()
            .prepare(&root, &service, &selection, &[true])
            .expect_err("shape mismatch must fail");
        assert!(error.to_string().contains("shape mismatch"));
        fs::remove_dir_all(root).expect("remove adapter test directory");
    }
}
