//! Model-bound adapter defaults; every generation receives an immutable selection.
use anyhow::Context;
use onnx_genai_engine::{AdapterActivation, AdapterSelection, GenerateOptions};
use onnx_genai_metadata::{AdapterServiceContract, AdapterWeightFormat};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

#[derive(Debug, thiserror::Error)]
#[error("invalid LoRA selection")]
struct InvalidSelection;

pub(crate) fn api_error(error: anyhow::Error) -> crate::routes::ApiError {
    let message = format!("{error:#}");
    if error.is::<InvalidSelection>() {
        crate::routes::ApiError::invalid_request(message)
    } else {
        crate::routes::ApiError::internal(message)
    }
}

fn scale_or_zero<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
    let value = serde_json::Value::deserialize(deserializer)?;
    if let Some(value) = value.as_f64() {
        Ok(value as f32)
    } else {
        if !value.is_null() {
            tracing::warn!("lora.scale must be numeric; using the upstream default 0");
        }
        Ok(0.0)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct Activation {
    pub(crate) id: usize,
    #[serde(default, deserialize_with = "scale_or_zero")]
    pub(crate) scale: f32,
}
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ListedAdapter {
    pub(crate) id: usize,
    pub(crate) path: String,
    pub(crate) scale: f32,
}
#[derive(Debug)]
struct Entry {
    id: usize,
    alias: String,
    path: String,
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct AdapterCapabilities {
    pub(crate) generation: bool,
    pub(crate) sessions: bool,
    pub(crate) embeddings: bool,
}
#[derive(Debug)]
pub(crate) struct LoraState {
    entries: Vec<Entry>,
    max_active: usize,
    pub(crate) capabilities: AdapterCapabilities,
    defaults: RwLock<Vec<AdapterActivation>>,
}
impl LoraState {
    pub(crate) fn new(
        contract: &AdapterServiceContract,
        capabilities: AdapterCapabilities,
    ) -> anyhow::Result<Arc<Self>> {
        let mut entries = Vec::new();
        for (alias, artifact) in &contract.artifacts {
            let weight=artifact.weights.iter().find(|w| w.format == AdapterWeightFormat::Json)
                .with_context(||format!("adapter '{alias}' has no source supported by this executor; supply a declared portable JSON artifact"))?;
            entries.push(Entry {
                id: artifact.index,
                alias: alias.clone(),
                path: weight.location.clone(),
            });
        }
        entries.sort_by_key(|e| e.id);
        Ok(Arc::new(Self {
            entries,
            capabilities,
            max_active: contract
                .selection
                .max_adapters
                .min(contract.cache.max_entries),
            defaults: RwLock::new(Vec::new()),
        }))
    }
    pub(crate) fn initialize(&self, without_apply: bool) -> anyhow::Result<()> {
        let activations = self
            .entries
            .iter()
            .map(|e| Activation {
                id: e.id,
                scale: if without_apply { 0.0 } else { 1.0 },
            })
            .collect::<Vec<_>>();
        self.set(&activations).context("initialize LoRA defaults; use --lora-init-without-apply when all declared adapters cannot be active together")
    }
    fn resolve(&self, activations: &[Activation]) -> anyhow::Result<Vec<AdapterActivation>> {
        self.normalize(activations).context(InvalidSelection)
    }
    fn normalize(&self, activations: &[Activation]) -> anyhow::Result<Vec<AdapterActivation>> {
        // llama.cpp parses this array as an ID-sorted map, with the last value
        // winning for repeated IDs. Display fields from GET may be echoed back.
        let mut selected = std::collections::BTreeMap::new();
        for activation in activations {
            let index = self
                .entries
                .binary_search_by_key(&activation.id, |entry| entry.id)
                .map_err(|_| {
                    anyhow::anyhow!(
                        "lora adapter ID {} is undeclared; list /lora-adapters for this model",
                        activation.id
                    )
                })?;
            selected.insert(activation.id, (&self.entries[index], activation.scale));
        }
        let mut result = Vec::new();
        for (entry, scale) in selected.into_values() {
            anyhow::ensure!(
                scale.is_finite(),
                "lora adapter {} scale must be finite",
                entry.id
            );
            if scale != 0.0 {
                result.push(AdapterActivation::new(&entry.alias, scale));
            }
        }
        anyhow::ensure!(
            result.len() <= self.max_active,
            "lora selects {} active adapters, exceeding this model's composition/cache capacity {}; disable some adapters or update the declared capacities",
            result.len(),
            self.max_active
        );
        Ok(result)
    }
    pub(crate) fn set(&self, activations: &[Activation]) -> anyhow::Result<()> {
        let resolved = self.resolve(activations)?;
        *self
            .defaults
            .write()
            .map_err(|_| anyhow::anyhow!("LoRA defaults lock is poisoned; reload the model"))? =
            resolved;
        Ok(())
    }
    pub(crate) fn selection(
        &self,
        activations: Option<&[Activation]>,
    ) -> anyhow::Result<AdapterSelection> {
        let row = match activations {
            Some(values) => self.resolve(values)?,
            None => self
                .defaults
                .read()
                .map_err(|_| anyhow::anyhow!("LoRA defaults lock is poisoned; reload the model"))?
                .clone(),
        };
        Ok(AdapterSelection::default().with_row(row))
    }
    pub(crate) fn list(&self) -> anyhow::Result<Vec<ListedAdapter>> {
        let defaults = self
            .defaults
            .read()
            .map_err(|_| anyhow::anyhow!("LoRA defaults lock is poisoned; reload the model"))?;
        Ok(self
            .entries
            .iter()
            .map(|entry| ListedAdapter {
                id: entry.id,
                path: entry.path.clone(),
                scale: defaults
                    .iter()
                    .find(|a| a.adapter == entry.alias)
                    .map_or(0.0, |a| a.scale),
            })
            .collect())
    }
}

pub(crate) fn apply_request(
    request: &crate::ChatCompletionRequest,
    handle: &crate::registry::ModelHandle,
    options: &mut GenerateOptions,
    has_session: bool,
) -> Result<(), crate::routes::ApiError> {
    use crate::routes::ApiError;
    if has_session
        && handle
            .engine
            .adapters
            .as_ref()
            .is_some_and(|state| !state.capabilities.sessions)
    {
        return Err(ApiError::invalid_request(
            "LoRA requests cannot reuse this model's session-scoped tensors; send the full conversation stateless, or use a workflow that replays token history",
        ));
    }
    options.adapter_selection = resolve_request(request.extensions.get("lora"), handle)?;
    if options
        .adapter_selection
        .as_ref()
        .is_some_and(|selection| selection.rows.iter().any(|row| !row.is_empty()))
        && handle
            .engine
            .adapters
            .as_ref()
            .is_some_and(|state| !state.capabilities.generation)
    {
        return Err(ApiError::invalid_request(
            "this model admits adapters only for embedding workflows; select an adapter-aware generation workflow or send lora: []",
        ));
    }
    Ok(())
}

pub(crate) fn resolve_request(
    value: Option<&serde_json::Value>,
    handle: &crate::registry::ModelHandle,
) -> Result<Option<AdapterSelection>, crate::routes::ApiError> {
    use crate::routes::ApiError;
    let requested = value
        .map(|value| serde_json::from_value::<Vec<Activation>>(value.clone()))
        .transpose()
        .map_err(|e| {
            ApiError::invalid_request(format!(
                "invalid lora: {e}; use [{{\"id\":0,\"scale\":1.0}}] or [] to disable all adapters"
            ))
        })?;
    match &handle.engine.adapters {
        Some(state) => Ok(Some(
            state.selection(requested.as_deref()).map_err(api_error)?,
        )),
        None => {
            if requested.as_ref().is_some_and(|values| !values.is_empty()) {
                return Err(ApiError::invalid_request(format!(
                    "model '{}' has no admitted adapter service; select an adapter-aware package or omit lora",
                    handle.id
                )));
            }
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capacity_refusals_preserve_defaults_and_initialization_is_explicit() {
        let mut contract: onnx_genai_metadata::InferenceMetadata =
            serde_yaml::from_str(include_str!(
                "../../../tests/fixtures/onnx_genai_workflows/adapter/inference_metadata.yaml"
            ))
            .unwrap();
        let mut contract = contract.adapters.take().unwrap();
        contract.cache.max_entries = 1;
        let state = LoraState::new(
            &contract,
            AdapterCapabilities {
                generation: true,
                sessions: true,
                embeddings: false,
            },
        )
        .unwrap();
        assert!(
            state
                .initialize(false)
                .unwrap_err()
                .to_string()
                .contains("--lora-init-without-apply")
        );
        state.initialize(true).unwrap();
        state.set(&[Activation { id: 3, scale: 0.5 }]).unwrap();
        let before = serde_json::to_value(state.list().unwrap()).unwrap();
        let error = state
            .set(&[
                Activation { id: 0, scale: 1.0 },
                Activation { id: 3, scale: 1.0 },
            ])
            .unwrap_err();
        assert!(error.is::<InvalidSelection>());
        assert_eq!(serde_json::to_value(state.list().unwrap()).unwrap(), before);
        assert!(
            state
                .set(&[Activation {
                    id: 3,
                    scale: f32::INFINITY
                }])
                .is_err()
        );
    }
}
