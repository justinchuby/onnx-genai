use std::collections::{BTreeMap, BTreeSet};

use crate::{GenAiConfigError, incomplete};

/// One graph tensor declaration supplied by the package loader.
///
/// This inventory comes from the ONNX graph interface itself, so compatibility
/// conversion can preserve actual sparse ports, ranks, and dtypes instead of
/// expanding architecture-sized guesses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphTensorInfo {
    pub name: String,
    pub dtype: String,
    /// One entry per axis; `None` denotes a symbolic dimension.
    pub dimensions: Vec<Option<usize>>,
}

/// Explicit input/output inventory for one ONNX component.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelGraphInfo {
    pub inputs: Vec<GraphTensorInfo>,
    pub outputs: Vec<GraphTensorInfo>,
}

/// ONNX graph inventories required to synthesize a strict multimodal pipeline.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PipelineGraphInfo {
    pub vision: ModelGraphInfo,
    pub embedding: ModelGraphInfo,
    pub decoder: ModelGraphInfo,
}

/// ONNX graph inventories required to synthesize a strict encoder-decoder
/// (audio/text sequence-to-sequence) pipeline, e.g. Whisper.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EncoderDecoderGraphInfo {
    pub encoder: ModelGraphInfo,
    pub decoder: ModelGraphInfo,
}

pub(crate) fn require_graph_input<'a>(
    graph: &'a ModelGraphInfo,
    name: &str,
    component: &str,
) -> Result<&'a GraphTensorInfo, GenAiConfigError> {
    graph
        .inputs
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| incomplete(format!("{component} ONNX input '{name}'")))
}

pub(crate) fn require_graph_output<'a>(
    graph: &'a ModelGraphInfo,
    name: &str,
    component: &str,
) -> Result<&'a GraphTensorInfo, GenAiConfigError> {
    graph
        .outputs
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| incomplete(format!("{component} ONNX output '{name}'")))
}

pub(crate) fn require_same_dtype(
    left: &GraphTensorInfo,
    right: &GraphTensorInfo,
    description: &str,
) -> Result<(), GenAiConfigError> {
    if left.dtype == right.dtype {
        Ok(())
    } else {
        Err(incomplete(format!(
            "{description} dtype agreement: '{}' is {}, but '{}' is {}",
            left.name, left.dtype, right.name, right.dtype
        )))
    }
}

/// Match a paired key/value `%d` name pattern against `tensors` and return the
/// ordered layer index set, the interleaved `[key_0, value_0, key_1, ...]`
/// names verified to exist in the graph, and the single shared cache dtype.
///
/// Unlike [`GenAiConfig::strict_decoder_state`], this does not require the key
/// and value patterns to share a common textual prefix (Whisper self/cross KV
/// use distinct `..._key_self_%d` / `..._value_self_%d` prefixes), and it does
/// not derive any fixed-state `state_pairs`, so encoder-decoder cross-attention
/// and cross-QK ports are never misread as recurrent state.
pub(crate) fn strict_indexed_kv(
    tensors: &[GraphTensorInfo],
    key_pattern: &str,
    value_pattern: &str,
    description: &str,
) -> Result<(Vec<usize>, Vec<String>, String), GenAiConfigError> {
    let keys = match_indexed_tensors(tensors, key_pattern)?;
    let values = match_indexed_tensors(tensors, value_pattern)?;
    let indices = exact_index_set(&[&keys, &values], description)?;
    if indices.is_empty() {
        return Err(incomplete(format!(
            "at least one {description} graph-port pair"
        )));
    }
    let mut names = Vec::with_capacity(indices.len() * 2);
    let mut dtype: Option<String> = None;
    for index in &indices {
        let key = keys[index];
        let value = values[index];
        require_same_dtype(key, value, description)?;
        match dtype.as_deref() {
            Some(canonical) if canonical != key.dtype => {
                return Err(incomplete(format!(
                    "all {description} tensors must use one dtype: canonical dtype is {canonical}, but '{}' is {}",
                    key.name, key.dtype
                )));
            }
            None => dtype = Some(key.dtype.clone()),
            _ => {}
        }
        names.push(key.name.clone());
        names.push(value.name.clone());
    }
    Ok((
        indices,
        names,
        dtype.expect("non-empty KV indices establish a dtype"),
    ))
}

pub(crate) fn split_indexed_pattern(pattern: &str) -> Result<(&str, &str), GenAiConfigError> {
    let Some((prefix, suffix)) = pattern.split_once("%d") else {
        return Err(incomplete(format!(
            "indexed tensor pattern '{pattern}' must contain %d"
        )));
    };
    if suffix.contains("%d") {
        return Err(incomplete(format!(
            "indexed tensor pattern '{pattern}' must contain exactly one %d"
        )));
    }
    Ok((prefix, suffix))
}

pub(crate) fn match_indexed_tensors<'a>(
    tensors: &'a [GraphTensorInfo],
    pattern: &str,
) -> Result<BTreeMap<usize, &'a GraphTensorInfo>, GenAiConfigError> {
    let (prefix, suffix) = split_indexed_pattern(pattern)?;
    let mut matched = BTreeMap::new();
    for tensor in tensors {
        let Some(index) = tensor
            .name
            .strip_prefix(prefix)
            .and_then(|name| name.strip_suffix(suffix))
        else {
            continue;
        };
        let index = index.parse::<usize>().map_err(|_| {
            incomplete(format!(
                "ONNX tensor '{}' matches pattern '{pattern}' but has a non-numeric index",
                tensor.name
            ))
        })?;
        if matched.insert(index, tensor).is_some() {
            return Err(incomplete(format!(
                "ONNX graph has duplicate tensors for pattern '{pattern}' index {index}"
            )));
        }
    }
    Ok(matched)
}

pub(crate) fn exact_index_set(
    maps: &[&BTreeMap<usize, &GraphTensorInfo>],
    description: &str,
) -> Result<Vec<usize>, GenAiConfigError> {
    let Some(first) = maps.first() else {
        return Ok(Vec::new());
    };
    let expected = first.keys().copied().collect::<Vec<_>>();
    if maps
        .iter()
        .skip(1)
        .all(|map| map.keys().copied().eq(expected.iter().copied()))
    {
        Ok(expected)
    } else {
        Err(incomplete(format!(
            "{description} do not have identical layer indices"
        )))
    }
}

pub(crate) fn common_pattern_prefix<'a>(
    first: &'a str,
    second: &'a str,
) -> Result<&'a str, GenAiConfigError> {
    let (first_prefix, _) = split_indexed_pattern(first)?;
    let (second_prefix, _) = split_indexed_pattern(second)?;
    if first_prefix == second_prefix {
        Ok(first_prefix)
    } else {
        Err(incomplete(format!(
            "key/value patterns '{first}' and '{second}' must share the same state prefix"
        )))
    }
}

pub(crate) fn suffix_tensor_map<'a>(
    tensors: &'a [GraphTensorInfo],
    prefix: &str,
    excluded: &BTreeSet<&str>,
    description: &str,
) -> Result<BTreeMap<String, &'a GraphTensorInfo>, GenAiConfigError> {
    let mut matched = BTreeMap::new();
    for tensor in tensors {
        if excluded.contains(tensor.name.as_str()) {
            continue;
        }
        let Some(suffix) = tensor.name.strip_prefix(prefix) else {
            continue;
        };
        if suffix.is_empty() {
            return Err(incomplete(format!(
                "{description} contains an empty suffix for '{}'",
                tensor.name
            )));
        }
        if matched.insert(suffix.to_owned(), tensor).is_some() {
            return Err(incomplete(format!(
                "{description} contains duplicate suffix '{suffix}'"
            )));
        }
    }
    Ok(matched)
}
