use super::*;

#[derive(Debug, Clone)]
pub(super) struct KvPair {
    pub(super) past: String,
    pub(super) present: String,
    pub(super) input: TensorInfo,
    pub(super) seq_axis: usize,
}
pub(super) fn infer_kv_pairs(
    session: &Session,
    io: Option<&onnx_genai_metadata::ModelIoSpec>,
) -> Result<Vec<KvPair>> {
    if let Some(io) = io {
        return match (&io.kv_inputs, &io.kv_outputs) {
            (Some(inputs), Some(outputs)) if inputs.len() == outputs.len() => inputs
                .iter()
                .zip(outputs)
                .map(|(past, present)| kv_pair(session, past, present))
                .collect(),
            (Some(inputs), Some(outputs)) => Err(OrtError::InvalidArgument(format!(
                "io.kv_inputs ({}) and io.kv_outputs ({}) must have equal length",
                inputs.len(),
                outputs.len()
            ))),
            (None, None) => Ok(Vec::new()),
            _ => Err(OrtError::InvalidArgument(
                "io.kv_inputs and io.kv_outputs must be declared together".into(),
            )),
        };
    }
    let input_names = session.input_names();
    let mut pairs = Vec::new();
    for output in session.outputs() {
        if !name_contains_present_key_value(&output.name) {
            continue;
        }
        let Some(suffix) = kv_suffix(&output.name, KvNamingConvention::Dotted) else {
            continue;
        };
        let Some(past_name) = input_names.iter().find(|input| {
            kv_suffix(input, KvNamingConvention::Dotted).as_deref() == Some(suffix.as_str())
        }) else {
            continue;
        };
        pairs.push(kv_pair(session, past_name, &output.name)?);
    }
    Ok(pairs)
}

fn kv_pair(session: &Session, past_name: &str, present_name: &str) -> Result<KvPair> {
    let input = session
        .inputs()
        .iter()
        .find(|input| input.name == past_name)
        .cloned()
        .ok_or_else(|| {
            OrtError::InvalidArgument(format!("declared KV input '{past_name}' is not exposed"))
        })?;
    if !session
        .outputs()
        .iter()
        .any(|output| output.name == present_name)
    {
        return Err(OrtError::InvalidArgument(format!(
            "declared KV output '{present_name}' is not exposed"
        )));
    }
    if !matches!(
        input.dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        return Err(OrtError::InvalidArgument(format!(
            "KV input '{}' must be Float32, Float16, or BFloat16, got {:?}",
            input.name, input.dtype
        )));
    }
    if input.shape.len() < 3 {
        return Err(OrtError::InvalidArgument(format!(
            "KV input '{}' has unsupported shape {:?}",
            input.name, input.shape
        )));
    }
    let seq_axis = input.shape.len() - 2;
    Ok(KvPair {
        past: past_name.to_string(),
        present: present_name.to_string(),
        input,
        seq_axis,
    })
}

pub(super) fn detect_static_cache(
    session: &Session,
) -> Result<Option<(StaticCacheSignature, Vec<StaticCachePair>)>> {
    let has_write_indices = session
        .input_names()
        .iter()
        .any(|name| name == "write_indices");
    let has_nonpad = session
        .input_names()
        .iter()
        .any(|name| name == "nonpad_kv_seqlen");
    if !has_write_indices || !has_nonpad {
        return Ok(None);
    }

    let mut indices = session
        .inputs()
        .iter()
        .filter_map(|input| static_cache_suffix(&input.name, "key_cache."))
        .collect::<Vec<_>>();
    indices.sort_unstable();
    indices.dedup();
    if indices.is_empty() {
        return Ok(None);
    }

    let mut pairs = Vec::with_capacity(indices.len());
    let mut max_len = None;
    let mut kv_dim = None;
    let mut dtype = None;
    for index in indices {
        let key_name = format!("key_cache.{index}");
        let value_name = format!("value_cache.{index}");
        let key_output = format!("updated_key_cache.{index}");
        let value_output = format!("updated_value_cache.{index}");
        let key_input = session
            .inputs()
            .iter()
            .find(|input| input.name == key_name)
            .cloned()
            .ok_or_else(|| OrtError::InvalidArgument(format!("missing input '{key_name}'")))?;
        let value_input = session
            .inputs()
            .iter()
            .find(|input| input.name == value_name)
            .cloned()
            .ok_or_else(|| OrtError::InvalidArgument(format!("missing input '{value_name}'")))?;
        if !session
            .output_names()
            .iter()
            .any(|name| name == &key_output)
        {
            return Err(OrtError::InvalidArgument(format!(
                "missing output '{key_output}'"
            )));
        }
        if !session
            .output_names()
            .iter()
            .any(|name| name == &value_output)
        {
            return Err(OrtError::InvalidArgument(format!(
                "missing output '{value_output}'"
            )));
        }
        validate_static_cache_tensor(&key_input)?;
        validate_static_cache_tensor(&value_input)?;
        if key_input.shape[1..] != value_input.shape[1..] {
            return Err(OrtError::InvalidArgument(format!(
                "key/value cache shape mismatch for layer {index}: {:?} vs {:?}",
                key_input.shape, value_input.shape
            )));
        }
        if key_input.dtype != value_input.dtype {
            return Err(OrtError::InvalidArgument(format!(
                "key/value cache dtype mismatch for layer {index}: {:?} vs {:?}",
                key_input.dtype, value_input.dtype
            )));
        }
        let layer_max_len = key_input.shape[1] as usize;
        let layer_kv_dim = key_input.shape[2] as usize;
        if max_len.get_or_insert(layer_max_len) != &layer_max_len {
            return Err(OrtError::InvalidArgument(
                "static-cache layers have inconsistent max lengths".into(),
            ));
        }
        if kv_dim.get_or_insert(layer_kv_dim) != &layer_kv_dim {
            return Err(OrtError::InvalidArgument(
                "static-cache layers have inconsistent KV dims".into(),
            ));
        }
        if dtype.get_or_insert(key_input.dtype) != &key_input.dtype {
            return Err(OrtError::InvalidArgument(
                "static-cache layers have inconsistent dtypes".into(),
            ));
        }
        pairs.push(StaticCachePair {
            index,
            key_input,
            value_input,
            key_output,
            value_output,
        });
    }
    pairs.sort_by_key(|pair| pair.index);
    let signature = StaticCacheSignature {
        layers: pairs.len(),
        max_len: max_len
            .ok_or_else(|| OrtError::InvalidArgument("non-empty static cache pairs".into()))?,
        kv_dim: kv_dim
            .ok_or_else(|| OrtError::InvalidArgument("non-empty static cache pairs".into()))?,
        dtype: dtype
            .ok_or_else(|| OrtError::InvalidArgument("non-empty static cache pairs".into()))?,
        has_position_ids: session
            .input_names()
            .iter()
            .any(|name| name == "position_ids"),
    };
    Ok(Some((signature, pairs)))
}

fn static_cache_suffix(name: &str, prefix: &str) -> Option<usize> {
    name.strip_prefix(prefix)?.parse().ok()
}

fn validate_static_cache_tensor(info: &TensorInfo) -> Result<()> {
    if !matches!(
        info.dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        return Err(OrtError::InvalidArgument(format!(
            "static-cache tensor '{}' must be Float32, Float16, or BFloat16, got {:?}",
            info.name, info.dtype
        )));
    }
    if info.shape.len() != 3 || info.shape[1] <= 0 || info.shape[2] <= 0 {
        return Err(OrtError::InvalidArgument(format!(
            "static-cache tensor '{}' must have shape [B, MAX_LEN, KV_DIM], got {:?}",
            info.name, info.shape
        )));
    }
    Ok(())
}
