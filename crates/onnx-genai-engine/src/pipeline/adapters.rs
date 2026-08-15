use anyhow::Context;
use onnx_genai_metadata::{AdapterArtifact, AdapterServiceContract, AdapterWeightFormat};
use onnx_genai_ort::Value;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct AdapterActivation {
    pub adapter: String,
    pub scale: f32,
    pub scale_input: Option<String>,
}

impl AdapterActivation {
    pub fn new(adapter: impl Into<String>, scale: f32) -> Self {
        Self {
            adapter: adapter.into(),
            scale,
            scale_input: None,
        }
    }

    pub fn with_scale_input(mut self, input: impl Into<String>) -> Self {
        self.scale_input = Some(input.into());
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AdapterSelection {
    /// Immutable composition keyed by semantic slot and its current request epoch.
    pub rows: BTreeMap<AdapterRowIdentity, Vec<AdapterActivation>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AdapterRowIdentity {
    pub row_id: i64,
    pub request_epoch: i64,
}

impl AdapterSelection {
    pub fn with_row(
        mut self,
        row_id: i64,
        request_epoch: i64,
        adapters: impl IntoIterator<Item = AdapterActivation>,
    ) -> Self {
        self.rows.insert(
            AdapterRowIdentity {
                row_id,
                request_epoch,
            },
            adapters.into_iter().collect(),
        );
        self
    }
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

#[derive(Debug, Clone, Default)]
pub(super) struct AdapterRunContext {
    pub rows: Vec<(AdapterRowIdentity, Vec<(String, f32)>)>,
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
        row_ids: &[i64],
        request_epochs: &[i64],
        active_rows: &[bool],
        request_inputs: &HashMap<String, Value>,
    ) -> anyhow::Result<AdapterRunContext> {
        if request_epochs.len() != row_ids.len() {
            anyhow::bail!(
                "adapter request_epochs has {} rows but row_ids has {}",
                request_epochs.len(),
                row_ids.len()
            );
        }
        if let Some(epoch) = request_epochs.iter().find(|epoch| **epoch < 0) {
            anyhow::bail!("adapter request epoch {epoch} must be non-negative");
        }
        if active_rows.len() != row_ids.len() {
            anyhow::bail!(
                "adapter active mask has {} rows but row_ids has {}",
                active_rows.len(),
                row_ids.len()
            );
        }
        let request_rows = row_ids
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        if request_rows.len() != row_ids.len() {
            anyhow::bail!("adapter semantic row_ids must be unique within a request");
        }
        let identities = row_ids
            .iter()
            .copied()
            .zip(request_epochs.iter().copied())
            .map(|(row_id, request_epoch)| AdapterRowIdentity {
                row_id,
                request_epoch,
            })
            .collect::<Vec<_>>();
        let active_request_rows = identities
            .iter()
            .copied()
            .zip(active_rows)
            .filter_map(|(identity, active)| active.then_some(identity))
            .collect::<std::collections::HashSet<_>>();
        let requested = selection
            .rows
            .iter()
            .filter(|(identity, _)| active_request_rows.contains(identity))
            .flat_map(|(_, activations)| activations)
            .map(|activation| activation.adapter.as_str())
            .collect::<std::collections::HashSet<_>>();
        if requested.len() > service.cache.max_entries {
            anyhow::bail!(
                "request activates {} distinct adapters but the adapter cache holds only {}; increase workflow.adapters.cache.max_entries",
                requested.len(),
                service.cache.max_entries
            );
        }
        let mut rows = Vec::with_capacity(row_ids.len());
        let mut plan_parts = Vec::with_capacity(row_ids.len());
        for (physical_row, identity) in identities.iter().copied().enumerate() {
            let activations = if active_rows[physical_row] {
                selection.rows.get(&identity).cloned().unwrap_or_default()
            } else {
                Vec::new()
            };
            let mut resolved = Vec::with_capacity(activations.len());
            let mut row_key = Vec::with_capacity(activations.len());
            for activation in activations {
                if !activation.scale.is_finite()
                    || activation.scale < -16.0
                    || activation.scale > 16.0
                {
                    anyhow::bail!(
                        "adapter '{}' scale {} for row {row_id} must be finite and within [-16, 16]",
                        activation.adapter,
                        activation.scale,
                        row_id = identity.row_id
                    );
                }
                let artifact = service
                    .artifacts
                    .get(&activation.adapter)
                    .with_context(|| {
                        format!(
                            "adapter selection for row {row_id} references undeclared adapter '{}'",
                            activation.adapter,
                            row_id = identity.row_id
                        )
                    })?;
                self.ensure_loaded(root, service, &activation.adapter, artifact)?;
                let dynamic = if let Some(input) = &activation.scale_input {
                    let scales = request_inputs
                        .get(input)
                        .with_context(|| {
                            format!("adapter dynamic scale input '{input}' is absent")
                        })?
                        .to_vec_f32()
                        .with_context(|| {
                            format!("adapter dynamic scale input '{input}' must be host float32")
                        })?;
                    *scales.get(physical_row).with_context(|| {
                        format!("adapter dynamic scale input '{input}' has no row {physical_row}")
                    })?
                } else {
                    1.0
                };
                let scale = activation.scale * dynamic;
                if !scale.is_finite() || !(-16.0..=16.0).contains(&scale) {
                    anyhow::bail!(
                        "adapter '{}' effective scale {scale} for row {row_id} must be finite and within [-16, 16]",
                        activation.adapter,
                        row_id = identity.row_id
                    );
                }
                row_key.push(format!("{}:{scale:.8}", activation.adapter));
                resolved.push((activation.adapter, scale));
            }
            plan_parts.push(format!(
                "{}@{}=[{}]",
                identity.row_id,
                identity.request_epoch,
                row_key.join(",")
            ));
            rows.push((identity, resolved));
        }
        let plan_key = plan_parts.join(";");
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
        let loaded = load_adapter(root, artifact, &service.application_capability)?;
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
    for (_, activations) in &context.rows {
        for (adapter, _) in activations {
            output_features = cache.target(adapter, component, parameter)?.output_features;
        }
    }
    let mut result = vec![0.0_f32; batch * output_features];
    for (row, (_, activations)) in context.rows.iter().enumerate() {
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
    artifact: &AdapterArtifact,
    application_capability: &str,
) -> anyhow::Result<LoadedAdapter> {
    let mut bundles = HashMap::new();
    for weight in &artifact.weights {
        if weight.format != AdapterWeightFormat::Json {
            anyhow::bail!(
                "portable adapter fallback cannot load {:?}; use an execution provider with '{}' support",
                weight.format,
                application_capability
            );
        }
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
        bundles.extend(bundle.targets);
    }
    let mut targets = HashMap::new();
    for target in &artifact.targets {
        let tensors = bundles.get(&target.weight_key).with_context(|| {
            format!(
                "adapter {}@{} artifact has no target key '{}'",
                artifact.identity, artifact.version, target.weight_key
            )
        })?;
        let expected_a = artifact.rank * target.input_features;
        let expected_b = target.output_features * artifact.rank;
        if tensors.a.len() != expected_a || tensors.b.len() != expected_b {
            anyhow::bail!(
                "adapter target '{}.{}' shape mismatch: A has {} values (expected {expected_a}), B has {} values (expected {expected_b})",
                target.component,
                target.parameter,
                tensors.a.len(),
                tensors.b.len()
            );
        }
        targets.insert(
            (target.component.clone(), target.parameter.clone()),
            AdapterTargetWeights {
                input_features: target.input_features,
                output_features: target.output_features,
                rank: artifact.rank,
                factor: (artifact.alpha / artifact.rank as f64) as f32,
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
        AdapterCacheContract, AdapterEvictionPolicy, AdapterPlanningContract, AdapterTargetBinding,
        AdapterWeightArtifact,
    };

    fn artifact(name: &str, location: &str, bytes: &[u8]) -> AdapterArtifact {
        AdapterArtifact {
            identity: name.to_string(),
            version: "1".to_string(),
            base_model_fingerprint: "base-sha256".to_string(),
            rank: 1,
            alpha: 1.0,
            dtype: "float32".to_string(),
            weights: vec![AdapterWeightArtifact {
                location: location.to_string(),
                sha256: format!("{:x}", Sha256::digest(bytes)),
                format: AdapterWeightFormat::Json,
            }],
            targets: vec![AdapterTargetBinding {
                component: "decoder".to_string(),
                parameter: "projection".to_string(),
                weight_key: "projection".to_string(),
                input_features: 2,
                output_features: 2,
            }],
            provenance: Some("synthetic-test".to_string()),
        }
    }

    fn service_contract(red: &[u8], blue: &[u8], max_entries: usize) -> AdapterServiceContract {
        AdapterServiceContract {
            base_model_fingerprint: "base-sha256".to_string(),
            row_ids: "request.row_ids".to_string(),
            request_epochs: "request.request_epochs".to_string(),
            active: None,
            application_capability: "onnx-genai.adapters".to_string(),
            portable_fallback: true,
            artifacts: BTreeMap::from([
                ("red".to_string(), artifact("red", "red.json", red)),
                ("blue".to_string(), artifact("blue", "blue.json", blue)),
            ]),
            cache: AdapterCacheContract {
                max_entries,
                eviction: AdapterEvictionPolicy::Lru,
            },
            planning: AdapterPlanningContract::default(),
        }
    }

    #[test]
    fn heterogeneous_composition_compaction_and_cache_lifecycle() {
        let root = std::env::current_dir()
            .expect("current directory")
            .join("target")
            .join(format!("adapter-test-{}", std::process::id()));
        fs::create_dir_all(&root).expect("create adapter test directory");
        let red = br#"{"targets":{"projection":{"a":[1.0,0.0],"b":[1.0,2.0]}}}"#;
        let blue = br#"{"targets":{"projection":{"a":[0.0,1.0],"b":[3.0,4.0]}}}"#;
        fs::write(root.join("red.json"), red).expect("write red adapter");
        fs::write(root.join("blue.json"), blue).expect("write blue adapter");

        let service = service_contract(red, blue, 2);
        let selection = AdapterSelection::default()
            .with_row(10, 0, [AdapterActivation::new("red", 1.0)])
            .with_row(
                30,
                0,
                [
                    AdapterActivation::new("red", 0.5),
                    AdapterActivation::new("blue", 1.0),
                ],
            );
        let mut cache = AdapterCache::default();
        let context = cache
            .prepare(
                &root,
                &service,
                &selection,
                &[10, 20, 30],
                &[0, 0, 0],
                &[true, true, true],
                &HashMap::new(),
            )
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
        for (row_id, row_input, expected) in [
            (10, [1.0, 2.0], [2.0, 4.0]),
            (20, [3.0, 4.0], [3.0, 4.0]),
            (30, [5.0, 6.0], [25.5, 35.0]),
        ] {
            let single = cache
                .prepare(
                    &root,
                    &service,
                    &selection,
                    &[row_id],
                    &[0],
                    &[true],
                    &HashMap::new(),
                )
                .expect("prepare independent row");
            let (single_output, _) =
                apply_parameter_overlay(&cache, &single, "decoder", "projection", &row_input, 1, 2)
                    .expect("apply independent row");
            assert_eq!(single_output, expected);
        }

        let dynamic = AdapterSelection::default().with_row(
            10,
            0,
            [AdapterActivation::new("red", 1.0).with_scale_input("request.adapter_scale")],
        );
        let dynamic_inputs = HashMap::from([(
            "request.adapter_scale".to_string(),
            Value::from_slice_f32(&[0.5], &[1]).expect("dynamic scale"),
        )]);
        let dynamic_context = cache
            .prepare(
                &root,
                &service,
                &dynamic,
                &[10],
                &[0],
                &[true],
                &dynamic_inputs,
            )
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

        // The semantic row IDs carry adapter identity through physical-row compaction.
        let compacted = cache
            .prepare(
                &root,
                &service,
                &selection,
                &[30, 10],
                &[0, 0],
                &[true, true],
                &HashMap::new(),
            )
            .expect("prepare compacted adapters");
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

        let mut inactive_cache = AdapterCache::default();
        let inactive = inactive_cache
            .prepare(
                &root,
                &service,
                &selection,
                &[10],
                &[0],
                &[false],
                &HashMap::new(),
            )
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

        // A one-entry cache evicts, reloads, and invalidates captured plan variants.
        let service = service_contract(red, blue, 1);
        let mut cache = AdapterCache::default();
        for (row_id, adapter) in [(1, "red"), (2, "blue"), (3, "red")] {
            let selection = AdapterSelection::default().with_row(
                row_id,
                0,
                [AdapterActivation::new(adapter, 1.0)],
            );
            cache
                .prepare(
                    &root,
                    &service,
                    &selection,
                    &[row_id],
                    &[0],
                    &[true],
                    &HashMap::new(),
                )
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
    fn artifact_checksum_and_shape_are_enforced() {
        let root = std::env::current_dir()
            .expect("current directory")
            .join("target")
            .join(format!("adapter-invalid-test-{}", std::process::id()));
        fs::create_dir_all(&root).expect("create adapter test directory");
        let valid = br#"{"targets":{"projection":{"a":[1.0,0.0],"b":[1.0,2.0]}}}"#;
        fs::write(root.join("red.json"), b"corrupt").expect("write corrupt adapter");
        fs::write(root.join("blue.json"), valid).expect("write blue adapter");
        let service = service_contract(valid, valid, 2);
        let selection =
            AdapterSelection::default().with_row(1, 0, [AdapterActivation::new("red", 1.0)]);
        let error = AdapterCache::default()
            .prepare(
                &root,
                &service,
                &selection,
                &[1],
                &[0],
                &[true],
                &HashMap::new(),
            )
            .expect_err("checksum mismatch must fail");
        assert!(error.to_string().contains("checksum mismatch"));

        let malformed = br#"{"targets":{"projection":{"a":[1.0],"b":[1.0,2.0]}}}"#;
        fs::write(root.join("red.json"), malformed).expect("write malformed adapter");
        let service = service_contract(malformed, valid, 2);
        let error = AdapterCache::default()
            .prepare(
                &root,
                &service,
                &selection,
                &[1],
                &[0],
                &[true],
                &HashMap::new(),
            )
            .expect_err("shape mismatch must fail");
        assert!(error.to_string().contains("shape mismatch"));
        fs::remove_dir_all(root).expect("remove adapter test directory");
    }
}
