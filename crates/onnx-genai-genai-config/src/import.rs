//! Fail-closed, one-way `genai_config.json` -> `InferenceMetadata` import.
//!
//! The legacy config is a *source* format only. There is no export path and no
//! backward compatibility: nothing in this crate turns an [`InferenceMetadata`]
//! back into a `genai_config.json`, because a reverse synthesizer would have to
//! approximate facts the new contract states precisely.
//!
//! Import is fail-closed. Deserialization alone is not enough to decide that an
//! import was faithful, because serde silently discards keys the wire types do
//! not name. This module therefore walks the *raw* JSON and compares every key
//! path against [`CONSUMED_KEYS`] (paths the converter genuinely reads) and
//! [`KNOWN_DROPPED_KEYS`] (paths the new contract deliberately does not carry).
//! Anything else is an error.
//!
//! Callers that accept a lossy import pass [`ImportOptions::allow_lossy`], which
//! downgrades the error to a recorded list of dropped key paths in
//! [`ImportReport::dropped_keys`]. The list is the whole point: a lossy import
//! must be able to say exactly what it threw away.

use std::path::Path;

use onnx_genai_metadata::InferenceMetadata;

use crate::{GenAiConfig, GenAiConfigError, ModelGraphInfo, loading};

/// Key paths the converter reads. `*` matches one path segment (an array index
/// or a dynamic map key such as a pipeline stage name).
///
/// A path listed here covers itself and everything beneath it.
pub const CONSUMED_KEYS: &[&str] = &[
    "model.type",
    "model.context_length",
    "model.vocab_size",
    "model.pad_token_id",
    "model.bos_token_id",
    "model.eos_token_id",
    "model.sep_token_id",
    "model.decoder_start_token_id",
    "model.image_token_id",
    "model.video_token_id",
    "model.vision_start_token_id",
    "model.decoder.filename",
    "model.decoder.head_size",
    "model.decoder.num_attention_heads",
    "model.decoder.num_key_value_heads",
    "model.decoder.num_hidden_layers",
    "model.decoder.inputs.input_ids",
    "model.decoder.inputs.inputs_embeds",
    "model.decoder.inputs.attention_mask",
    "model.decoder.inputs.position_ids",
    "model.decoder.inputs.past_key_names",
    "model.decoder.inputs.past_value_names",
    "model.decoder.inputs.past_names",
    "model.decoder.inputs.cross_past_key_names",
    "model.decoder.inputs.cross_past_value_names",
    "model.decoder.inputs.encoder_hidden_states",
    "model.decoder.inputs.targets",
    "model.decoder.inputs.lstm_hidden_state",
    "model.decoder.inputs.lstm_cell_state",
    "model.decoder.outputs.logits",
    "model.decoder.outputs.present_key_names",
    "model.decoder.outputs.present_value_names",
    "model.decoder.outputs.present_names",
    "model.decoder.pipeline.*.*.filename",
    "model.decoder.pipeline.*.*.inputs",
    "model.decoder.pipeline.*.*.outputs",
    "model.encoder.filename",
    "model.encoder.num_attention_heads",
    "model.encoder.num_hidden_layers",
    "model.encoder.inputs.input_ids",
    "model.encoder.inputs.audio_features",
    "model.encoder.inputs.attention_mask",
    "model.encoder.outputs.encoder_hidden_states",
    "model.encoder.outputs.cross_present_key_names",
    "model.encoder.outputs.cross_present_value_names",
    "model.embedding.filename",
    "model.embedding.inputs.input_ids",
    "model.embedding.inputs.image_features",
    "model.embedding.inputs.audio_features",
    "model.embedding.outputs.inputs_embeds",
    "model.vision.filename",
    "model.vision.config_filename",
    "model.vision.spatial_merge_size",
    "model.vision.patch_size",
    "model.vision.inputs.pixel_values",
    "model.vision.inputs.image_sizes",
    "model.vision.inputs.image_grid_thw",
    "model.vision.outputs.image_features",
    "model.speech.filename",
    "model.speech.inputs.audio_embeds",
    "model.speech.inputs.attention_mask",
    "model.speech.outputs.audio_features",
    "model.joiner.filename",
    "model.joiner.inputs.encoder_outputs",
    "model.joiner.inputs.decoder_outputs",
    "model.joiner.outputs.logits",
    "model.vad.filename",
    "search.past_present_share_buffer",
    "search.max_length",
    "search.min_length",
    "search.do_sample",
    "search.temperature",
    "search.top_k",
    "search.top_p",
    "search.repetition_penalty",
    "search.num_beams",
    "search.num_return_sequences",
    "search.length_penalty",
    "search.no_repeat_ngram_size",
    "search.diversity_penalty",
    "search.early_stopping",
];

/// Key paths the new contract deliberately does not carry, with the reason.
///
/// These are not oversights. Each names a fact the redesign moved out of package
/// metadata (deployment policy the runtime owns) or a legacy field the typed
/// contract replaced. They still count as dropped: an import that hits one is
/// lossy and must be opted into.
pub const KNOWN_DROPPED_KEYS: &[(&str, &str)] = &[
    (
        "model.decoder.session_options",
        "execution provider, device, and session tuning are runtime deployment policy",
    ),
    (
        "model.encoder.session_options",
        "execution provider, device, and session tuning are runtime deployment policy",
    ),
    (
        "model.embedding.session_options",
        "execution provider, device, and session tuning are runtime deployment policy",
    ),
    (
        "model.vision.session_options",
        "execution provider, device, and session tuning are runtime deployment policy",
    ),
    (
        "model.speech.session_options",
        "execution provider, device, and session tuning are runtime deployment policy",
    ),
    (
        "model.decoder.run_options",
        "per-run execution tuning is runtime deployment policy",
    ),
    (
        "model.kv_cache",
        "KV allocator, paging, and storage mode are runtime deployment policy; graph-visible \
         state representation is inferred from the graph ABI instead",
    ),
    (
        "model.decoder.inputs.block_table",
        "paged-attention block tables are a kernel/runtime varlen ABI, not package metadata",
    ),
    (
        "model.decoder.inputs.slot_mapping",
        "paged-attention slot mapping is a kernel/runtime varlen ABI, not package metadata",
    ),
    (
        "model.decoder.inputs.cache_indirection",
        "beam cache indirection is a runtime-minted row selection, never serialized metadata",
    ),
    (
        "model.decoder.outputs.output_cross_qk_names",
        "cross-attention QK export is a diagnostic surface the runtime owns",
    ),
    (
        "model.decoder.inputs.cache_last_channel",
        "Conformer NeMo streaming caches have no declared state kind in this contract yet",
    ),
    (
        "model.decoder.inputs.cache_last_time",
        "Conformer NeMo streaming caches have no declared state kind in this contract yet",
    ),
    (
        "model.decoder.inputs.rnn_states",
        "RNN decoder state has no declared state kind in this contract yet",
    ),
    (
        "model.decoder.hidden_size",
        "hidden width is a graph-visible tensor dimension; the contract declares attention shape \
         and reads widths from the model's own ONNX ABI",
    ),
    (
        "model.encoder.hidden_size",
        "hidden width is a graph-visible tensor dimension; the contract declares attention shape \
         and reads widths from the model's own ONNX ABI",
    ),
    (
        "model.encoder.head_size",
        "encoder self-attention shape stays inside the encoder graph; the contract declares only \
         the attention shape that sizes the KV state a decoder carries",
    ),
    (
        "model.encoder.num_key_value_heads",
        "encoder self-attention shape stays inside the encoder graph; the contract declares only \
         the attention shape that sizes the KV state a decoder carries",
    ),
    (
        "model.vision.tokens_per_second",
        "video temporal sampling rate has no declared field in this contract yet",
    ),
];

/// How strict an import should be.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImportOptions {
    /// Record unrepresentable keys instead of failing.
    ///
    /// This is the `--allow-lossy` switch. It never changes what the converter
    /// produces; it only decides whether dropping a key is fatal.
    pub allow_lossy: bool,
}

/// What an import consumed and what it threw away.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImportReport {
    /// Key paths present in the source that the new contract does not carry,
    /// in sorted order. Empty for a faithful import.
    pub dropped_keys: Vec<String>,
}

impl ImportReport {
    /// Whether anything was dropped.
    pub fn is_lossy(&self) -> bool {
        !self.dropped_keys.is_empty()
    }
}

/// Key paths in `raw` that the import cannot carry into the new contract.
///
/// Walks leaves only: a key whose value is an object is judged by its children
/// unless the key path itself is claimed, which lets `CONSUMED_KEYS` claim whole
/// subtrees (`model.decoder.pipeline.*.*.inputs`) without listing array indices.
pub fn unrepresentable_keys(raw: &serde_json::Value) -> Vec<String> {
    let mut dropped = Vec::new();
    walk(raw, &mut Vec::new(), &mut dropped);
    dropped.sort();
    dropped.dedup();
    dropped
}

fn walk(value: &serde_json::Value, path: &mut Vec<String>, dropped: &mut Vec<String>) {
    if !path.is_empty() {
        if claims(CONSUMED_KEYS.iter().copied(), path) {
            return;
        }
        // A known-dropped pattern claims its whole subtree, and the claim root —
        // not each leaf under it — is what the report names, because the reason
        // is recorded against the root.
        if claims(KNOWN_DROPPED_KEYS.iter().map(|(key, _)| *key), path) {
            dropped.push(path.join("."));
            return;
        }
    }
    match value {
        serde_json::Value::Object(map) if !map.is_empty() => {
            for (key, child) in map {
                path.push(key.clone());
                walk(child, path, dropped);
                path.pop();
            }
        }
        serde_json::Value::Array(items) if !items.is_empty() => {
            for (index, child) in items.iter().enumerate() {
                path.push(index.to_string());
                walk(child, path, dropped);
                path.pop();
            }
        }
        _ if path.is_empty() => {}
        _ => dropped.push(path.join(".")),
    }
}

fn claims<'a>(patterns: impl Iterator<Item = &'a str>, path: &[String]) -> bool {
    patterns
        .into_iter()
        .any(|pattern| matches_pattern(pattern, path))
}

/// Whether `pattern` claims `path` or any ancestor of it.
fn matches_pattern(pattern: &str, path: &[String]) -> bool {
    let segments = pattern.split('.').collect::<Vec<_>>();
    if segments.len() > path.len() {
        return false;
    }
    segments
        .iter()
        .zip(path)
        .all(|(segment, actual)| *segment == "*" || *segment == actual.as_str())
}

/// Why a given dropped key is not representable, when the reason is recorded.
pub fn drop_reason(key: &str) -> Option<&'static str> {
    KNOWN_DROPPED_KEYS.iter().find_map(|(pattern, reason)| {
        let path = key.split('.').map(str::to_owned).collect::<Vec<_>>();
        matches_pattern(pattern, &path).then_some(*reason)
    })
}

/// Import a parsed `genai_config.json` fail-closed.
///
/// `raw` must be the same document `config` was parsed from; it carries the keys
/// serde discarded.
pub fn import(
    config: &GenAiConfig,
    raw: &serde_json::Value,
    kv_native_dtype: Option<&str>,
    decoder_graph: Option<&ModelGraphInfo>,
    options: ImportOptions,
) -> Result<(InferenceMetadata, ImportReport), GenAiConfigError> {
    let dropped_keys = unrepresentable_keys(raw);
    if !dropped_keys.is_empty() && !options.allow_lossy {
        return Err(GenAiConfigError::LossyImport {
            keys: describe(&dropped_keys),
        });
    }
    let metadata = match decoder_graph {
        Some(graph) => config.to_inference_metadata_with_graph(kv_native_dtype, graph)?,
        None => config.to_inference_metadata(kv_native_dtype)?,
    };
    Ok((metadata, ImportReport { dropped_keys }))
}

/// Import a `genai_config.json` from disk fail-closed.
pub fn import_from_path(
    path: &Path,
    kv_native_dtype: Option<&str>,
    decoder_graph: Option<&ModelGraphInfo>,
    options: ImportOptions,
) -> Result<(InferenceMetadata, ImportReport), GenAiConfigError> {
    let content = std::fs::read_to_string(path)?;
    let raw: serde_json::Value = serde_json::from_str(&content)?;
    let config: GenAiConfig = serde_json::from_str(&content)?;
    import(&config, &raw, kv_native_dtype, decoder_graph, options)
}

/// Import from a model directory, or `Ok(None)` when it has no legacy config.
pub fn import_from_dir(
    model_dir: &Path,
    kv_native_dtype: Option<&str>,
    decoder_graph: Option<&ModelGraphInfo>,
    options: ImportOptions,
) -> Result<Option<(InferenceMetadata, ImportReport)>, GenAiConfigError> {
    let Some(path) = loading::find_in_dir(model_dir) else {
        return Ok(None);
    };
    import_from_path(&path, kv_native_dtype, decoder_graph, options).map(Some)
}

fn describe(keys: &[String]) -> String {
    keys.iter()
        .map(|key| match drop_reason(key) {
            Some(reason) => format!("{key} ({reason})"),
            None => key.clone(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal() -> serde_json::Value {
        serde_json::json!({
            "model": {
                "type": "qwen2",
                "context_length": 4096,
                "vocab_size": 152064,
                "eos_token_id": 151645,
                "decoder": {
                    "filename": "model.onnx",
                    "head_size": 128,
                    "num_attention_heads": 14,
                    "num_key_value_heads": 2,
                    "num_hidden_layers": 24,
                    "inputs": {
                        "input_ids": "input_ids",
                        "attention_mask": "attention_mask",
                        "position_ids": "position_ids",
                        "past_key_names": "past_key_values.%d.key",
                        "past_value_names": "past_key_values.%d.value"
                    },
                    "outputs": {
                        "logits": "logits",
                        "present_key_names": "present.%d.key",
                        "present_value_names": "present.%d.value"
                    }
                }
            },
            "search": { "max_length": 4096, "past_present_share_buffer": true }
        })
    }

    #[test]
    fn a_fully_representable_config_drops_nothing() {
        assert!(unrepresentable_keys(&minimal()).is_empty());
    }

    #[test]
    fn deployment_policy_keys_are_reported_as_dropped() {
        let mut raw = minimal();
        raw["model"]["decoder"]["session_options"] =
            serde_json::json!({ "provider_options": [{ "cuda": {} }] });
        let dropped = unrepresentable_keys(&raw);
        assert_eq!(dropped, vec!["model.decoder.session_options".to_owned()]);
        assert!(drop_reason(&dropped[0]).is_some_and(|reason| reason.contains("runtime")));
    }

    #[test]
    fn unrecognized_keys_are_reported_without_a_reason() {
        let mut raw = minimal();
        raw["model"]["decoder"]["invented_field"] = serde_json::json!(7);
        let dropped = unrepresentable_keys(&raw);
        assert_eq!(dropped, vec!["model.decoder.invented_field".to_owned()]);
        assert_eq!(drop_reason(&dropped[0]), None);
    }

    #[test]
    fn import_fails_closed_by_default() {
        let mut raw = minimal();
        raw["model"]["kv_cache"] = serde_json::json!({ "native_dtype": "float16" });
        let config: GenAiConfig = serde_json::from_value(raw.clone()).expect("config");
        let error = import(&config, &raw, None, None, ImportOptions::default())
            .expect_err("lossy import must fail closed");
        assert!(
            matches!(error, GenAiConfigError::LossyImport { .. }),
            "{error}"
        );
        assert!(error.to_string().contains("model.kv_cache"), "{error}");
        assert!(error.to_string().contains("--allow-lossy"), "{error}");
    }

    #[test]
    fn allow_lossy_records_every_dropped_key() {
        let mut raw = minimal();
        raw["model"]["kv_cache"] = serde_json::json!({ "native_dtype": "float16" });
        raw["model"]["decoder"]["run_options"] = serde_json::json!({ "log_level": 2 });
        let config: GenAiConfig = serde_json::from_value(raw.clone()).expect("config");
        let (_metadata, report) = import(
            &config,
            &raw,
            None,
            None,
            ImportOptions { allow_lossy: true },
        )
        .expect("lossy import is allowed");
        assert!(report.is_lossy());
        assert_eq!(
            report.dropped_keys,
            vec![
                "model.decoder.run_options".to_owned(),
                "model.kv_cache".to_owned(),
            ]
        );
    }

    #[test]
    fn a_faithful_import_reports_no_loss() {
        let raw = minimal();
        let config: GenAiConfig = serde_json::from_value(raw.clone()).expect("config");
        let (_metadata, report) =
            import(&config, &raw, None, None, ImportOptions::default()).expect("faithful import");
        assert!(!report.is_lossy());
    }

    #[test]
    fn dynamic_pipeline_stage_names_are_claimed_by_wildcards() {
        let mut raw = minimal();
        raw["model"]["decoder"]["pipeline"] = serde_json::json!([
            { "embeds": { "filename": "embeds.onnx", "inputs": ["input_ids"], "outputs": ["h"] } },
            { "body": { "filename": "body.onnx", "inputs": ["h"], "outputs": ["logits"] } },
        ]);
        assert!(unrepresentable_keys(&raw).is_empty());
    }

    #[test]
    fn declared_search_defaults_reach_the_generation_contract() {
        // `search.*` is listed in CONSUMED_KEYS, so the import claims to read it.
        // Before the generation contract was wired up, only `search.max_length`
        // was actually read and every sampling field the package author declared
        // was silently discarded -- a package asking for temperature 1.0 / top-k
        // 40 got greedy decoding and no diagnostic. This pins the claim.
        let mut raw = minimal();
        raw["search"]["do_sample"] = serde_json::json!(true);
        raw["search"]["temperature"] = serde_json::json!(1.0);
        raw["search"]["top_k"] = serde_json::json!(40);
        raw["search"]["top_p"] = serde_json::json!(0.8);
        raw["search"]["repetition_penalty"] = serde_json::json!(1.1);
        let config: GenAiConfig = serde_json::from_value(raw.clone()).expect("config");
        let (metadata, report) =
            import(&config, &raw, None, None, ImportOptions::default()).expect("import");
        assert!(!report.is_lossy());
        let defaults = metadata
            .generation
            .as_ref()
            .and_then(|generation| generation.defaults.as_ref())
            .expect("generation defaults");
        assert_eq!(defaults.do_sample, Some(true));
        assert_eq!(defaults.temperature, Some(1.0));
        assert_eq!(defaults.top_k, Some(40));
        assert_eq!(defaults.top_p, Some(0.8));
        assert_eq!(defaults.repetition_penalty, Some(1.1));
    }

    #[test]
    fn numeric_token_facts_reach_the_tokenizer_package_authority() {
        let mut raw = minimal();
        raw["model"]["pad_token_id"] = serde_json::json!(0);
        raw["model"]["bos_token_id"] = serde_json::json!(1);
        raw["model"]["eos_token_id"] = serde_json::json!([2, 3]);
        raw["model"]["sep_token_id"] = serde_json::json!(4);
        raw["model"]["decoder_start_token_id"] = serde_json::json!(5);
        raw["model"]["image_token_id"] = serde_json::json!(6);
        raw["model"]["video_token_id"] = serde_json::json!(7);
        raw["model"]["vision_start_token_id"] = serde_json::json!(8);
        let config: GenAiConfig = serde_json::from_value(raw.clone()).expect("config");
        let (metadata, report) =
            import(&config, &raw, None, None, ImportOptions::default()).expect("import");
        assert!(!report.is_lossy());
        let tokens = metadata
            .package
            .expect("package facts")
            .tokenizer
            .expect("tokenizer facts")
            .special_tokens
            .expect("special token facts");
        assert_eq!(tokens.pad_token_id, Some(0));
        assert_eq!(tokens.bos_token_id, Some(1));
        assert_eq!(tokens.eos_token_id, [2, 3]);
        assert_eq!(tokens.sep_token_id, Some(4));
        assert_eq!(tokens.decoder_start_token_id, Some(5));
        assert_eq!(tokens.image_token_id, Some(6));
        assert_eq!(tokens.video_token_id, Some(7));
        assert_eq!(tokens.vision_start_token_id, Some(8));
    }

    #[test]
    fn undeclared_sampling_policy_is_not_invented() {
        // Absence must stay absent. `max_length` is carried because the author
        // wrote it, but every field they did NOT write stays `None` rather than
        // acquiring a plausible-looking default -- a fabricated `temperature`
        // would be indistinguishable from a declared one downstream.
        let mut raw = minimal();
        raw["search"] = serde_json::json!({ "max_length": 4096 });
        let config: GenAiConfig = serde_json::from_value(raw.clone()).expect("config");
        let (metadata, _) =
            import(&config, &raw, None, None, ImportOptions::default()).expect("import");
        let defaults = metadata
            .generation
            .as_ref()
            .and_then(|generation| generation.defaults.as_ref())
            .expect("declared max_length is a generation default");
        assert_eq!(defaults.max_length, Some(4096));
        assert_eq!(
            *defaults,
            onnx_genai_metadata::GenerationDefaults {
                max_length: Some(4096),
                ..Default::default()
            }
        );
    }

    #[test]
    fn a_config_that_declares_nothing_gets_no_generation_section() {
        let mut raw = minimal();
        raw.as_object_mut().expect("object").remove("search");
        let config: GenAiConfig = serde_json::from_value(raw.clone()).expect("config");
        let (metadata, _) =
            import(&config, &raw, None, None, ImportOptions::default()).expect("import");
        assert!(metadata.generation.is_none());
    }

    #[test]
    fn shared_buffer_declaration_becomes_permitted_aliasing() {
        // `past_present_share_buffer: true` is the legacy author's statement that
        // aliasing present onto past is legal for this graph. The new contract
        // spells that as the state group's aliasing, and it is the only thing that can let
        // an imported package take the shared-buffer decode path.
        let raw = minimal();
        let config: GenAiConfig = serde_json::from_value(raw.clone()).expect("config");
        let (metadata, _) =
            import(&config, &raw, None, None, ImportOptions::default()).expect("import");
        assert_eq!(
            metadata.decoder_io().and_then(|io| io.aliasing),
            Some(onnx_genai_metadata::StateAliasing::Permitted)
        );
    }

    #[test]
    fn silence_about_sharing_stays_forbidden() {
        // Omission is not permission. A config that never claimed its graph
        // tolerates aliasing must not acquire that claim through import.
        for value in [serde_json::Value::Bool(false), serde_json::Value::Null] {
            let mut raw = minimal();
            if value.is_null() {
                raw["search"]
                    .as_object_mut()
                    .expect("search object")
                    .remove("past_present_share_buffer");
            } else {
                raw["search"]["past_present_share_buffer"] = value;
            }
            let config: GenAiConfig = serde_json::from_value(raw.clone()).expect("config");
            let (metadata, _) =
                import(&config, &raw, None, None, ImportOptions::default()).expect("import");
            // The workflow's state group always states its aliasing, so silence
            // in the source config resolves to an explicit `Forbidden` rather
            // than to an absent field. The meaning is unchanged — every consumer
            // reads an absent aliasing as `Forbidden` — but the package now says
            // so instead of leaving a reader to know the default.
            assert_eq!(
                metadata.decoder_io().and_then(|io| io.aliasing),
                Some(onnx_genai_metadata::StateAliasing::Forbidden)
            );
        }
    }

    /// Every real Foundry Local package carries `model.decoder.hidden_size`, so
    /// an unexplained drop there made every one of them lossy for a reason the
    /// operator could not read. Classifying it does not make the import any less
    /// strict -- a classified drop is still a drop -- it only means the WARN says
    /// why.
    #[test]
    fn the_structural_size_keys_real_packages_carry_all_state_their_reason() {
        for key in [
            "model.decoder.hidden_size",
            "model.encoder.hidden_size",
            "model.encoder.head_size",
            "model.encoder.num_key_value_heads",
            "model.vision.tokens_per_second",
        ] {
            assert!(
                drop_reason(key).is_some(),
                "'{key}' is dropped without a recorded reason"
            );
        }
    }

    /// Classifying a key must not smuggle it past the fail-closed gate.
    #[test]
    fn a_classified_drop_is_still_lossy() {
        let raw = serde_json::json!({
            "model": {
                "type": "llama",
                "context_length": 128,
                "vocab_size": 32,
                "eos_token_id": 2,
                "decoder": {
                    "filename": "model.onnx",
                    "hidden_size": 64,
                    "head_size": 8,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 8,
                    "num_hidden_layers": 2,
                    "inputs": {"input_ids": "input_ids"},
                    "outputs": {"logits": "logits"}
                }
            },
            "search": {"max_length": 128}
        });
        let dropped = unrepresentable_keys(&raw);
        assert_eq!(dropped, vec!["model.decoder.hidden_size".to_string()]);

        let config: GenAiConfig = serde_json::from_value(raw.clone()).expect("config");
        let strict = import(&config, &raw, None, None, ImportOptions::default());
        assert!(
            strict.is_err(),
            "a documented drop must still fail a strict import"
        );

        let (_, report) = import(
            &config,
            &raw,
            None,
            None,
            ImportOptions { allow_lossy: true },
        )
        .expect("lossy import");
        assert!(report.is_lossy());
        assert_eq!(report.dropped_keys, dropped);
    }

    #[test]
    fn there_is_no_reverse_synthesizer() {
        // The import direction is one-way by construction: this crate exposes no
        // function that turns metadata back into a legacy config. Guarding it in
        // a test keeps a future "just add an exporter" change honest.
        for source in [
            include_str!("lib.rs"),
            include_str!("compatibility.rs"),
            include_str!("loading.rs"),
        ] {
            for banned in [
                "to_genai_config",
                "into_genai_config",
                "synthesize_genai_config",
            ] {
                assert!(
                    !source.contains(banned),
                    "reverse synthesizer '{banned}' reappeared"
                );
            }
        }
    }
}
