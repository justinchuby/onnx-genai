//! Explicit HF PEFT saved linear LoRA factors; graph targets remain manifest-owned.
use super::{JsonTarget, MemoryGovernor, MemoryLease, factor_storage::FactorStorage};
use anyhow::{Context, ensure};
use onnx_genai_metadata::{AdapterArtifact, AdapterServiceContract, AdapterWeightArtifact};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    io::Read,
    path::Path,
};

pub(super) fn configuration(
    root: &Path,
    weight: &AdapterWeightArtifact,
    authority: &dyn MemoryGovernor,
) -> anyhow::Result<(Vec<u8>, MemoryLease)> {
    let location = weight
        .config_location
        .as_ref()
        .context("HF PEFT weights require config_location naming adapter_config.json")?;
    let root = root.canonicalize().context("resolve PEFT package root")?;
    let path = root
        .join(location)
        .canonicalize()
        .with_context(|| format!("resolve PEFT config {location:?}"))?;
    ensure!(
        path.starts_with(&root),
        "PEFT config {} resolves outside package root {}",
        path.display(),
        root.display()
    );
    let file = std::fs::File::open(&path)
        .with_context(|| format!("open PEFT config {}", path.display()))?;
    let metadata = file
        .metadata()
        .context("inspect PEFT config before bounded read")?;
    ensure!(
        metadata.is_file(),
        "PEFT config {} must be a regular file",
        path.display()
    );
    let length = metadata.len();
    let lease = authority
        .reserve(
            onnx_runtime_memory_api::Tier::Host,
            length
                .checked_mul(16)
                .context("PEFT config parsing byte budget overflow")?,
            super::Holder::CpuDerivedWeights.role(),
            super::Holder::CpuDerivedWeights.id(),
        )
        .context("reserve PEFT config parsing memory")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(
            usize::try_from(length)?
                .checked_add(1)
                .context("PEFT config length overflow")?,
        )
        .context("allocate bounded PEFT config input")?;
    file.take(
        length
            .checked_add(1)
            .context("PEFT config read length overflow")?,
    )
    .read_to_end(&mut bytes)
    .context("read PEFT config")?;
    ensure!(
        bytes.len() as u64 == length,
        "PEFT config {} changed size while loading; replace artifacts atomically",
        path.display()
    );
    Ok((bytes, lease))
}

pub(super) fn load(
    root: &Path,
    weight: &AdapterWeightArtifact,
    artifact: &AdapterArtifact,
    service: &AdapterServiceContract,
    bytes: &[u8],
    authority: &dyn MemoryGovernor,
) -> anyhow::Result<(HashMap<String, JsonTarget>, [u8; 32])> {
    ensure!(
        matches!(
            weight.scale_encoding,
            onnx_genai_metadata::AdapterScaleEncoding::AlphaOverRank
        ),
        "HF PEFT ordinary saved factors require scale_encoding alpha_over_rank; baked scales are a different artifact format and cannot be inferred"
    );
    let storage = FactorStorage::declared(artifact)?;
    let (config_bytes, _config_lease) = configuration(root, weight, authority)?;
    let config: Value = serde_json::from_slice(&config_bytes).with_context(|| {
        format!(
            "parse HF PEFT config {:?} for adapter {}@{}; provide valid JSON",
            weight.config_location, artifact.identity, artifact.version
        )
    })?;
    let config = config.as_object().with_context(|| {
        format!(
            "HF PEFT config {:?} for adapter {}@{} must be a JSON object",
            weight.config_location, artifact.identity, artifact.version
        )
    })?;
    ensure!(
        config.get("peft_type").and_then(Value::as_str) == Some("LORA"),
        "PEFT peft_type must be LORA; other adapter algorithms require an explicit implementation"
    );
    let rank = config
        .get("r")
        .and_then(Value::as_u64)
        .and_then(|r| usize::try_from(r).ok())
        .filter(|r| *r > 0)
        .context("PEFT r must be a positive supported integer")?;
    let alpha = config
        .get("lora_alpha")
        .and_then(Value::as_f64)
        .filter(|a| a.is_finite() && *a > 0.0)
        .context("PEFT lora_alpha must be finite and positive")?;
    let modules=config.get("target_modules").and_then(Value::as_array).context("PEFT target_modules must be an explicit module-name list; regex/all-linear discovery is not supported")?;
    ensure!(
        !modules.is_empty()
            && modules
                .iter()
                .all(|m| m.as_str().is_some_and(|s| !s.is_empty())),
        "PEFT target_modules must contain nonempty strings"
    );
    for (key, value) in config {
        let supported = match key.as_str() {
            "peft_type" | "r" | "lora_alpha" | "target_modules" => true,
            "bias" => value.as_str() == Some("none"),
            "fan_in_fan_out"
            | "use_dora"
            | "use_rslora"
            | "lora_bias"
            | "use_qalora"
            | "ensure_weight_tying" => value == false,
            "modules_to_save"
            | "layers_to_transform"
            | "layers_pattern"
            | "target_parameters"
            | "exclude_modules"
            | "layer_replication" => value.is_null() || value.as_array().is_some_and(Vec::is_empty),
            "rank_pattern" | "alpha_pattern" | "loftq_config" | "runtime_config" => {
                value.as_object().is_some_and(serde_json::Map::is_empty)
            }
            "eva_config"
            | "corda_config"
            | "megatron_config"
            | "alora_invocation_tokens"
            | "trainable_token_indices"
            | "arrow_config"
            | "velora_config"
            | "kasa_config"
            | "astra_config"
            | "lora_ga_config"
            | "monteclora_config" => value.is_null(),
            "use_bdlora" => value.is_null() || value == false,
            "qalora_group_size" => value.as_u64().is_some_and(|size| size > 0),
            "inference_mode" => value.is_boolean(),
            "init_lora_weights" => value.is_boolean() || value.as_str() == Some("gaussian"),
            "base_model_name_or_path" | "revision" | "task_type" | "peft_version" => {
                value.is_null() || value.is_string()
            }
            "auto_mapping" => {
                value.is_null()
                    || value.as_object().is_some_and(|map| {
                        map.iter().all(|(key, value)| {
                            matches!(key.as_str(), "base_model_class" | "parent_library")
                                && value.is_string()
                        })
                    })
            }
            "lora_dropout" => value
                .as_f64()
                .is_some_and(|p| p.is_finite() && (0.0..=1.0).contains(&p)),
            // These values have no inference effect unless the associated
            // unsupported algorithm is enabled; never silently enable it.
            "megatron_core" => value.as_str() == Some("megatron.core"),
            _ => false,
        };
        ensure!(
            supported,
            "PEFT configuration field {key:?}={value} is unsupported by ordinary F32 linear LoRA; DoRA/RSLoRA, quantized/variant adapters, nonstandard bias, discovery and per-target pattern overrides require explicit support"
        );
    }
    let views = onnx_runtime_loader::safetensors_views(bytes)
        .context("parse HF PEFT safetensors factors")?;
    let mut used = HashSet::new();
    let mut targets = HashMap::new();
    for binding in &artifact.bindings {
        let prefix = &binding.weight_key;
        ensure!(
            modules.iter().any(|m| {
                let m = m.as_str().unwrap();
                prefix == m || prefix.ends_with(&format!(".{m}"))
            }),
            "PEFT binding module {prefix:?} is outside adapter_config target_modules; supply exact saved module prefixes in bindings.weight_key"
        );
        ensure!(
            binding.rank.unwrap_or(artifact.rank) == rank
                && binding.alpha.unwrap_or(artifact.alpha) == alpha,
            "PEFT module {prefix:?} r={rank}/lora_alpha={alpha} disagrees with authoritative artifact/binding rank and alpha; update the manifest instead of guessing scales"
        );
        let target = service
            .target_manifest
            .targets
            .iter()
            .find(|target| target.id == binding.target)
            .context("PEFT binding references an undeclared target")?;
        let width = target
            .output_slice
            .as_ref()
            .map_or(target.output_features, |slice| slice.width);
        let mut factors = Vec::new();
        for (label, shape) in [("A", [rank, target.input_features]), ("B", [width, rank])] {
            let name = format!("{prefix}.lora_{label}.weight");
            ensure!(
                used.insert(name.clone()),
                "PEFT saved tensor {name:?} is bound more than once"
            );
            let view=views.get(&name).with_context(||format!("PEFT tensor {name:?} is missing; embedding/Conv or named-adapter tensor spellings are not supported"))?;
            ensure!(
                view.dtype == storage.dtype(),
                "PEFT tensor {name:?} storage dtype {:?} disagrees with adapter {}@{} declared dtype {:?}; export both A/B in the declared F32/F16/BF16 storage dtype or correct the manifest",
                view.dtype,
                artifact.identity,
                artifact.version,
                storage.dtype()
            );
            ensure!(
                view.shape == shape,
                "PEFT tensor {name:?} shape {:?} must equal {shape:?} from the manifest target and PEFT rank",
                view.shape
            );
            let mut factor = Vec::new();
            factor
                .try_reserve_exact(view.data.len() / storage.width())
                .context("allocate governed normalized PEFT factor")?;
            for value in view.data.chunks_exact(storage.width()) {
                let value = storage.decode(value);
                ensure!(
                    value.is_finite(),
                    "PEFT tensor {name:?} contains nonfinite {:?} factors after F32 normalization; export finite ordinary linear factors",
                    storage.dtype()
                );
                factor.push(value);
            }
            factors.push(factor);
        }
        let b = factors.pop().unwrap();
        let a = factors.pop().unwrap();
        targets.insert(prefix.clone(), JsonTarget { a, b });
    }
    ensure!(
        views.keys().all(|name| used.contains(name)),
        "PEFT safetensors contains tensors without exact manifest bindings; map every saved linear module and reject modules_to_save/embedding/unknown tensor semantics"
    );
    Ok((targets, super::content_digest(&config_bytes)))
}
