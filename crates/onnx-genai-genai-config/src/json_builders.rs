use serde_json::{Map, Value, json};

pub(crate) fn expand_pattern(pattern: &str, layers: usize) -> Vec<String> {
    (0..layers)
        .map(|index| pattern.replace("%d", &index.to_string()))
        .collect()
}

pub(crate) fn expand_kv(
    combined: Option<&str>,
    key: Option<&str>,
    value: Option<&str>,
    default_key: &str,
    default_value: &str,
    layers: Option<usize>,
) -> Option<Vec<String>> {
    let layers = layers.filter(|layers| *layers > 0)?;
    if let Some(combined) = combined {
        return Some(expand_pattern(combined, layers));
    }
    let mut names = Vec::with_capacity(layers * 2);
    for index in 0..layers {
        names.push(key.unwrap_or(default_key).replace("%d", &index.to_string()));
        names.push(
            value
                .unwrap_or(default_value)
                .replace("%d", &index.to_string()),
        );
    }
    Some(names)
}

pub(crate) fn expand_cross_kv(
    key: Option<&str>,
    value: Option<&str>,
    layers: Option<usize>,
) -> Option<Vec<String>> {
    let (key, value, layers) = (key?, value?, layers.filter(|layers| *layers > 0)?);
    let mut names = Vec::with_capacity(layers * 2);
    for index in 0..layers {
        names.push(key.replace("%d", &index.to_string()));
        names.push(value.replace("%d", &index.to_string()));
    }
    Some(names)
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

pub(crate) fn is_share_buffer_kv_dtype(dtype: &str) -> bool {
    matches!(
        dtype.to_ascii_lowercase().as_str(),
        "float16" | "fp16" | "half" | "bfloat16" | "bf16" | "float32" | "fp32" | "float"
    )
}
