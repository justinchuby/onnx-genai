use std::collections::BTreeSet;

use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::{
    GenAiConfigError, GenAiVision, GraphTensorInfo, incomplete, required_str,
    unrepresentable_preprocessing,
};

#[derive(Debug, Deserialize)]
pub(crate) struct ProcessorConfig {
    pub(crate) processor: Processor,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Processor {
    pub(crate) transforms: Vec<ProcessorTransform>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ProcessorTransform {
    pub(crate) operation: ProcessorOperation,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ProcessorOperation {
    #[serde(rename = "type")]
    pub(crate) operation_type: String,
    #[serde(default)]
    pub(crate) attrs: Map<String, Value>,
}

pub(crate) fn processor_program_json(
    processor: &ProcessorConfig,
    vision: &GenAiVision,
    pixel_info: &GraphTensorInfo,
    grid_info: &GraphTensorInfo,
) -> Result<Value, GenAiConfigError> {
    let mut transforms = Vec::new();
    let mut seen = BTreeSet::new();
    for transform in &processor.processor.transforms {
        let operation = &transform.operation;
        match operation.operation_type.as_str() {
            "DecodeImage" | "ConvertRGB" => {
                if seen.insert("decode_rgb") {
                    transforms.push(json!({ "op": "decode_rgb" }));
                }
            }
            "Resize" => {
                let width = required_attr_u32(&operation.attrs, "width", "Resize")?;
                let height = required_attr_u32(&operation.attrs, "height", "Resize")?;
                let smart_resize = required_attr_flag(&operation.attrs, "smart_resize", "Resize")?;
                if smart_resize {
                    return Err(unrepresentable_preprocessing(
                        "processor Resize.attrs.smart_resize=true; smart resize is not representable by the runtime's stretch/crop/pad resize modes",
                    ));
                }
                transforms.push(json!({
                    "op": "resize",
                    "size": { "width": width, "height": height },
                    "mode": "stretch"
                }));
                seen.insert("resize");
            }
            "Rescale" => {
                let scale = operation
                    .attrs
                    .get("rescale_factor")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| incomplete("processor Rescale.attrs.rescale_factor"))?;
                transforms.push(json!({ "op": "rescale", "scale": scale }));
                seen.insert("rescale");
            }
            "Normalize" => {
                let mean = required_attr_f32_array(&operation.attrs, "mean", "Normalize")?;
                let std = required_attr_f32_array(&operation.attrs, "std", "Normalize")?;
                transforms.push(json!({ "op": "normalize", "mean": mean, "std": std }));
                seen.insert("normalize");
            }
            "PatchImage" => {
                let patch_size = required_attr_usize(&operation.attrs, "patch_size", "PatchImage")?;
                let temporal_patch_size =
                    required_attr_usize(&operation.attrs, "temporal_patch_size", "PatchImage")?;
                let merge_size = required_attr_usize(&operation.attrs, "merge_size", "PatchImage")?;
                if vision.patch_size != Some(patch_size) {
                    return Err(incomplete(format!(
                        "processor PatchImage patch_size ({patch_size}) must match model.vision.patch_size ({:?})",
                        vision.patch_size
                    )));
                }

                if vision.spatial_merge_size != Some(merge_size) {
                    return Err(incomplete(format!(
                        "processor PatchImage merge_size ({merge_size}) must match model.vision.spatial_merge_size ({:?})",
                        vision.spatial_merge_size
                    )));
                }
                transforms.push(json!({
                    "op": "patchify",
                    "patch_size": patch_size,
                    "temporal_patch_size": temporal_patch_size,
                    "merge_size": merge_size,
                    "channel_order": "channels_first",
                    "flatten": true
                }));
                seen.insert("patchify");
            }
            other => {
                return Err(incomplete(format!(
                    "processor operation '{other}' has no typed compatibility mapping"
                )));
            }
        }
    }
    for required in ["decode_rgb", "resize", "rescale", "normalize", "patchify"] {
        if !seen.contains(required) {
            return Err(incomplete(format!(
                "processor transform program operation '{required}'"
            )));
        }
    }

    let pixel_name = required_str(
        vision.inputs.pixel_values.as_deref(),
        "model.vision.inputs.pixel_values",
    )?;
    let grid_name = required_str(
        vision.inputs.image_grid_thw.as_deref(),
        "model.vision.inputs.image_grid_thw",
    )?;
    let outputs = vec![
        json!({
            "name": pixel_name,
            "content": "pixels",
            "dtype": pixel_info.dtype
        }),
        json!({
            "name": grid_name,
            "content": "grid_dimensions",
            "dtype": grid_info.dtype
        }),
    ];
    Ok(json!({
        "image": {
            "transforms": transforms,
            "outputs": outputs
        }
    }))
}

pub(crate) fn required_attr_flag(
    attrs: &Map<String, Value>,
    name: &str,
    operation: &str,
) -> Result<bool, GenAiConfigError> {
    match attrs.get(name).and_then(Value::as_u64) {
        Some(0) => Ok(false),
        Some(1) => Ok(true),
        _ => Err(incomplete(format!(
            "processor {operation}.attrs.{name} must be the numeric flag 0 or 1"
        ))),
    }
}

pub(crate) fn required_attr_u32(
    attrs: &Map<String, Value>,
    name: &str,
    operation: &str,
) -> Result<u32, GenAiConfigError> {
    let value = required_attr_usize(attrs, name, operation)?;
    u32::try_from(value)
        .map_err(|_| incomplete(format!("processor {operation}.attrs.{name} fits in u32")))
}

pub(crate) fn required_attr_usize(
    attrs: &Map<String, Value>,
    name: &str,
    operation: &str,
) -> Result<usize, GenAiConfigError> {
    attrs
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| incomplete(format!("processor {operation}.attrs.{name}")))
}

pub(crate) fn required_attr_f32_array(
    attrs: &Map<String, Value>,
    name: &str,
    operation: &str,
) -> Result<Vec<f32>, GenAiConfigError> {
    attrs
        .get(name)
        .and_then(Value::as_array)
        .and_then(|values| {
            values
                .iter()
                .map(|value| value.as_f64().map(|value| value as f32))
                .collect()
        })
        .ok_or_else(|| incomplete(format!("processor {operation}.attrs.{name}")))
}

pub(crate) fn expand_pattern(pattern: &str, layers: usize) -> Vec<String> {
    (0..layers)
        .map(|i| pattern.replace("%d", &i.to_string()))
        .collect()
}

/// Expand a self-attention KV name pattern for all layers.
///
/// A combined pattern yields one name per layer; separate key/value patterns
/// (falling back to the conventional defaults) interleave `[key_i, value_i]`.
pub(crate) fn expand_kv(
    combined: Option<&str>,
    key: Option<&str>,
    value: Option<&str>,
    default_key: &str,
    default_value: &str,
    layers: Option<usize>,
) -> Option<Vec<String>> {
    let layers = layers?;
    if layers == 0 {
        return None;
    }
    if let Some(combined) = combined {
        return Some(expand_pattern(combined, layers));
    }
    let key = key.unwrap_or(default_key);
    let value = value.unwrap_or(default_value);
    let mut out = Vec::with_capacity(layers * 2);
    for i in 0..layers {
        out.push(key.replace("%d", &i.to_string()));
        out.push(value.replace("%d", &i.to_string()));
    }
    Some(out)
}

/// Expand a cross-attention KV name pattern; requires both key and value to be
/// declared (no default injection). Interleaves `[key_i, value_i]` per layer.
pub(crate) fn expand_cross_kv(
    key: Option<&str>,
    value: Option<&str>,
    layers: Option<usize>,
) -> Option<Vec<String>> {
    let (key, value, layers) = (key?, value?, layers?);
    if layers == 0 {
        return None;
    }
    let mut out = Vec::with_capacity(layers * 2);
    for i in 0..layers {
        out.push(key.replace("%d", &i.to_string()));
        out.push(value.replace("%d", &i.to_string()));
    }
    Some(out)
}

pub(crate) fn component_json(filename: String, role: &str, io: Option<Value>) -> Value {
    let mut m = Map::new();
    m.insert("filename".into(), json!(filename));
    m.insert("type".into(), json!(role));
    if let Some(io) = io {
        m.insert("io".into(), io);
    }
    Value::Object(m)
}

pub(crate) fn edge(from: &str, to: &str) -> Value {
    json!({ "from": from, "to": to })
}

pub(crate) fn edge_with_dtype(from: &str, to: &str, dtype: &str) -> Value {
    json!({ "from": from, "to": to, "dtype": dtype })
}

pub(crate) fn run_on(phase: &str) -> Value {
    json!({ "run_on": phase })
}

/// A `composite` strategy: an optional single-pass encode stage followed by an
/// autoregressive decode stage.
pub(crate) fn composite_encode_decode(prompt_component: Option<&str>, decoder: &str) -> Value {
    let mut stages: Vec<Value> = Vec::new();
    if let Some(component) = prompt_component {
        stages.push(json!({
            "name": "encode",
            "strategy": { "kind": "single_pass", "model": component },
        }));
    }
    stages.push(json!({
        "name": "decode",
        "strategy": { "kind": "autoregressive", "decoder": decoder },
    }));
    json!({ "kind": "composite", "stages": stages })
}

pub(crate) fn pipeline_stage_role(name: &str) -> &'static str {
    match name {
        "embeddings" | "embedding" => "embedding",
        "language_model_head" | "lm_head" => "lm_head",
        _ => "decoder",
    }
}

pub(crate) fn filename_or(filename: &Option<String>, fallback: &str) -> String {
    filename.clone().unwrap_or_else(|| fallback.to_string())
}

pub(crate) fn insert_usize(map: &mut Map<String, Value>, key: &str, value: Option<usize>) {
    if let Some(value) = value {
        map.insert(key.into(), json!(value));
    }
}

pub(crate) fn insert_i64(map: &mut Map<String, Value>, key: &str, value: Option<i64>) {
    if let Some(value) = value {
        map.insert(key.into(), json!(value));
    }
}

pub(crate) fn insert_f32(map: &mut Map<String, Value>, key: &str, value: Option<f32>) {
    if let Some(value) = value {
        map.insert(key.into(), json!(value));
    }
}

pub(crate) fn insert_bool(map: &mut Map<String, Value>, key: &str, value: Option<bool>) {
    if let Some(value) = value {
        map.insert(key.into(), json!(value));
    }
}

/// Whether a dtype string denotes a KV dtype the share-buffer GQA path supports
/// (16- or 32-bit floating point). Mirrors the engine's gate.
pub(crate) fn is_share_buffer_kv_dtype(dtype: &str) -> bool {
    matches!(
        dtype.to_ascii_lowercase().as_str(),
        "float16" | "fp16" | "half" | "bfloat16" | "bf16" | "float32" | "fp32" | "float"
    )
}
