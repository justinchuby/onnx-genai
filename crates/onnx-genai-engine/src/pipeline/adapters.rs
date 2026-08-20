use super::row_state::{RowScopedState, gather_rows};
use anyhow::Context;
use onnx_genai_metadata::{
    AdapterArtifact, AdapterScaleEncoding, AdapterServiceContract, AdapterWeightFormat,
};
use onnx_genai_ort::Value;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::Path;

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
    pub rank: usize,
    pub factor: f32,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

#[derive(Debug, Clone)]
struct LoadedAdapter {
    targets: HashMap<(String, String), AdapterTargetWeights>,
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
    lru: VecDeque<String>,
    previously_loaded: std::collections::HashSet<String>,
    plans: std::collections::HashSet<String>,
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
                if !activation.scale.is_finite()
                    || activation.scale < -16.0
                    || activation.scale > 16.0
                {
                    anyhow::bail!(
                        "adapter '{}' scale {} for batch row {physical_row} must be finite and \
                         within [-16, 16]",
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
                self.ensure_loaded(root, service, &activation.adapter, artifact)?;
                let scale = activation.scale;
                if !scale.is_finite() || !(-16.0..=16.0).contains(&scale) {
                    anyhow::bail!(
                        "adapter '{}' effective scale {scale} for batch row {physical_row} must \
                         be finite and within [-16, 16]",
                        activation.adapter
                    );
                }
                row_key.push(format!("{}:{scale:.8}", activation.adapter));
                resolved.push((activation.adapter, scale));
            }
            plan_parts.push(format!("[{}]", row_key.join(",")));
            rows.push(resolved);
        }
        let plan_key = format!(
            "{}|{}",
            service.application_capability,
            plan_parts.join(";")
        );
        if !self.plans.insert(plan_key.clone()) {
            self.diagnostic.replayed_plans += 1;
        }
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
        if self.entries.contains_key(name) {
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
        let loaded = load_adapter(root, service, artifact)?;
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

    pub fn target(
        &self,
        adapter: &str,
        component: &str,
        parameter: &str,
    ) -> anyhow::Result<&AdapterTargetWeights> {
        self.entries
            .get(adapter)
            .with_context(|| format!("adapter '{adapter}' is not loaded"))?
            .targets
            .get(&(component.to_string(), parameter.to_string()))
            .with_context(|| {
                format!("adapter '{adapter}' does not target parameter '{component}.{parameter}'")
            })
    }
}

pub(super) fn apply_parameter_overlay(
    cache: &AdapterCache,
    context: &AdapterRunContext,
    component: &str,
    parameter: &str,
    source: &[f32],
    batch: usize,
    input_features: usize,
) -> anyhow::Result<(Vec<f32>, usize)> {
    if context.rows.len() != batch {
        anyhow::bail!(
            "adapter selection has {} rows but overlay input has batch {batch}",
            context.rows.len()
        );
    }
    let mut output_features = input_features;
    for activations in &context.rows {
        for (adapter, _) in activations {
            output_features = cache.target(adapter, component, parameter)?.output_features;
        }
    }
    let mut result = vec![0.0_f32; batch * output_features];
    for (row, activations) in context.rows.iter().enumerate() {
        if activations.is_empty() {
            if output_features != input_features {
                anyhow::bail!(
                    "base row without adapters requires matching input/output features, got {input_features} and {output_features}"
                );
            }
            result[row * output_features..(row + 1) * output_features]
                .copy_from_slice(&source[row * input_features..(row + 1) * input_features]);
            continue;
        }
        if output_features == input_features {
            result[row * output_features..(row + 1) * output_features]
                .copy_from_slice(&source[row * input_features..(row + 1) * input_features]);
        }
        for (adapter, scale) in activations {
            let target = cache.target(adapter, component, parameter)?;
            if target.input_features != input_features || target.output_features != output_features
            {
                anyhow::bail!(
                    "adapter '{adapter}' target '{component}.{parameter}' expects [{}, {}], got [{input_features}, {output_features}]",
                    target.input_features,
                    target.output_features
                );
            }
            let x = &source[row * input_features..(row + 1) * input_features];
            let mut low_rank = vec![0.0_f32; target.rank];
            for (rank, value) in low_rank.iter_mut().enumerate() {
                for (feature, input) in x.iter().enumerate() {
                    *value += input * target.a[rank * input_features + feature];
                }
            }
            for output in 0..output_features {
                let mut delta = 0.0_f32;
                for (rank, value) in low_rank.iter().enumerate() {
                    delta += target.b[output * target.rank + rank] * value;
                }
                result[row * output_features + output] += *scale * target.factor * delta;
            }
        }
    }
    Ok((result, output_features))
}

fn load_adapter(
    root: &Path,
    service: &AdapterServiceContract,
    artifact: &AdapterArtifact,
) -> anyhow::Result<LoadedAdapter> {
    anyhow::ensure!(
        matches!(artifact.dtype.as_str(), "float32" | "fp32"),
        "portable JSON adapter fallback requires float32, but {}@{} declares {}",
        artifact.identity,
        artifact.version,
        artifact.dtype
    );
    let weight = artifact
        .weights
        .iter()
        .find(|weight| weight.format == AdapterWeightFormat::Json)
        .with_context(|| {
            format!(
                "portable adapter fallback has no JSON artifact for {}@{}; use an execution provider with '{}' support",
                artifact.identity, artifact.version, service.application_capability
            )
        })?;
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
    let bytes = fs::read(&path)
        .with_context(|| format!("failed to load adapter artifact '{}'", path.display()))?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != weight.sha256 {
        anyhow::bail!(
            "adapter artifact '{}' checksum mismatch: expected {}, got {actual}",
            path.display(),
            weight.sha256
        );
    }
    let bundle: JsonBundle = serde_json::from_slice(&bytes)
        .with_context(|| format!("adapter artifact '{}' is invalid JSON", path.display()))?;
    let bundles = bundle.targets;
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
        let expected_a = rank * target.input_features;
        let expected_b = target.output_features * rank;
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
        targets.insert(
            (target.component.clone(), target.initializer.clone()),
            AdapterTargetWeights {
                input_features: target.input_features,
                output_features: target.output_features,
                rank,
                factor,
                a: tensors.a.clone(),
                b: tensors.b.clone(),
            },
        );
    }
    Ok(LoadedAdapter { targets })
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai_metadata::{
        AdapterCacheContract, AdapterDiscoveryFallback, AdapterEvictionPolicy,
        AdapterPlanningContract, AdapterProvenance, AdapterScaleEncoding, AdapterSelectionContract,
        AdapterTargetBinding, AdapterWeightArtifact, LoraTargetDescriptor, LoraTargetManifest,
    };

    fn artifact(name: &str, location: &str, bytes: &[u8]) -> AdapterArtifact {
        AdapterArtifact {
            index: usize::from(name == "blue"),
            identity: name.to_string(),
            version: "1".to_string(),
            base_model_fingerprint: format!(
                "onnx-genai-targeted-base-v1:sha256:{}",
                "a".repeat(64)
            ),
            rank: 1,
            alpha: 1.0,
            dtype: "float32".to_string(),
            weights: vec![AdapterWeightArtifact {
                location: location.to_string(),
                loader_capability: "onnx-genai.adapters.json@1".to_string(),
                sha256: format!("{:x}", Sha256::digest(bytes)),
                config_location: None,
                config_sha256: None,
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
            base_model_fingerprint: format!(
                "onnx-genai-targeted-base-v1:sha256:{}",
                "a".repeat(64)
            ),
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
        let mut cache = AdapterCache::default();
        let context = cache
            .prepare(&root, &service, &selection, &[true])
            .expect("load heterogeneous-rank artifact under one alias");
        let (rank_one, _) =
            apply_parameter_overlay(&cache, &context, "decoder", "projection", &[1.0, 2.0], 1, 2)
                .expect("apply rank-one target");
        let (rank_two, _) = apply_parameter_overlay(
            &cache,
            &context,
            "decoder",
            "projection_2",
            &[1.0, 2.0],
            1,
            2,
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
        let mut cache = AdapterCache::default();
        let context = cache
            .prepare(&root, &service, &selection, &[true, true, true])
            .expect("prepare heterogeneous adapters");
        let (output, width) = apply_parameter_overlay(
            &cache,
            &context,
            "decoder",
            "projection",
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            3,
            2,
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
            let (single_output, _) =
                apply_parameter_overlay(&cache, &single, "decoder", "projection", &row_input, 1, 2)
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
            &[1.0, 2.0],
            1,
            2,
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
            &[5.0, 6.0, 1.0, 2.0],
            2,
            2,
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
            &[1.0, 2.0, 5.0, 6.0],
            2,
            2,
        )
        .expect("apply after release");
        assert_eq!(released_output, vec![2.0, 4.0, 25.5, 35.0]);

        let mut inactive_cache = AdapterCache::default();
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
            &[1.0, 2.0],
            1,
            2,
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
        let mut cache = AdapterCache::default();
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
    fn artifact_checksum_and_shape_are_enforced() {
        let root = std::env::current_dir()
            .expect("current directory")
            .join("target")
            .join(format!("adapter-invalid-test-{}", std::process::id()));
        fs::create_dir_all(root.join("adapters/red")).expect("create red adapter test directory");
        fs::create_dir_all(root.join("adapters/blue")).expect("create blue adapter test directory");
        let valid = br#"{"targets":{"projection":{"a":[1.0,0.0],"b":[1.0,2.0]}}}"#;
        fs::write(root.join("adapters/red/adapter.json"), b"corrupt")
            .expect("write corrupt adapter");
        fs::write(root.join("adapters/blue/adapter.json"), valid).expect("write blue adapter");
        let service = service_contract(valid, valid, 2);
        let selection = AdapterSelection::default().with_row([AdapterActivation::new("red", 1.0)]);
        let error = AdapterCache::default()
            .prepare(&root, &service, &selection, &[true])
            .expect_err("checksum mismatch must fail");
        assert!(error.to_string().contains("checksum mismatch"));

        let malformed = br#"{"targets":{"projection":{"a":[1.0],"b":[1.0,2.0]}}}"#;
        fs::write(root.join("adapters/red/adapter.json"), malformed)
            .expect("write malformed adapter");
        let service = service_contract(malformed, valid, 2);
        let error = AdapterCache::default()
            .prepare(&root, &service, &selection, &[true])
            .expect_err("shape mismatch must fail");
        assert!(error.to_string().contains("shape mismatch"));
        fs::remove_dir_all(root).expect("remove adapter test directory");
    }
}
