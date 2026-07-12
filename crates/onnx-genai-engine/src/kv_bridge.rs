//! KV model metadata, paged-cache mirroring, and rewind helpers.

use crate::config::SessionId;
use crate::decode::{DecodeState, is_kv_input, is_present_output, matching_past_input};
use crate::logits::TokenId;
use crate::session::{DraftModel, DraftSession, EngineSession};
use anyhow::Context;
use onnx_genai_kv::{KvCacheOps, KvDType, LayerKv, PageId, PageTensorConfig, PagedKvCache};
use onnx_genai_ort::{DataType, Session, TensorInfo, Value};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(crate) struct KvModelInfo {
    pub(crate) tensor_config: PageTensorConfig,
    pub(crate) layers: Vec<KvLayerIo>,
}

#[derive(Debug, Clone)]
pub(crate) struct KvLayerIo {
    pub(crate) key_present: String,
    pub(crate) value_present: String,
    pub(crate) key_past: String,
    pub(crate) value_past: String,
}

pub(crate) fn infer_kv_model_info(
    session: &Session,
    page_size: usize,
) -> anyhow::Result<Option<KvModelInfo>> {
    let mut key_outputs = Vec::new();
    let mut value_outputs = Vec::new();
    for info in session
        .outputs()
        .iter()
        .filter(|info| is_present_output(&info.name))
    {
        let lower = info.name.to_ascii_lowercase();
        if lower.contains("key") {
            key_outputs.push(info.clone());
        } else if lower.contains("value") {
            value_outputs.push(info.clone());
        }
    }

    if key_outputs.is_empty() && value_outputs.is_empty() {
        return Ok(None);
    }
    key_outputs.sort_by_key(|info| kv_layer_index(&info.name).unwrap_or(usize::MAX));
    value_outputs.sort_by_key(|info| kv_layer_index(&info.name).unwrap_or(usize::MAX));
    if key_outputs.len() != value_outputs.len() {
        anyhow::bail!(
            "model exposes mismatched present key/value outputs: {} keys, {} values",
            key_outputs.len(),
            value_outputs.len()
        );
    }

    let (num_kv_heads, head_dim) = infer_kv_heads_and_head_dim(&key_outputs[0])?;
    let config = PageTensorConfig {
        num_layers: key_outputs.len(),
        num_kv_heads,
        head_dim,
        page_size,
        dtype: KvDType::F32,
    };
    let kv_inputs = session
        .inputs()
        .iter()
        .filter(|info| is_kv_input(&info.name))
        .map(|info| info.name.clone())
        .collect::<Vec<_>>();
    let mut layers = Vec::with_capacity(key_outputs.len());
    for (key, value) in key_outputs.iter().zip(value_outputs.iter()) {
        if key.dtype != DataType::Float32 || value.dtype != DataType::Float32 {
            anyhow::bail!("KV present outputs must be Float32");
        }
        let key_past = matching_past_input(&key.name, &kv_inputs)
            .with_context(|| format!("missing past input for present output '{}'", key.name))?
            .clone();
        let value_past = matching_past_input(&value.name, &kv_inputs)
            .with_context(|| format!("missing past input for present output '{}'", value.name))?
            .clone();
        layers.push(KvLayerIo {
            key_present: key.name.clone(),
            value_present: value.name.clone(),
            key_past,
            value_past,
        });
    }

    Ok(Some(KvModelInfo {
        tensor_config: config,
        layers,
    }))
}

pub(crate) fn infer_kv_heads_and_head_dim(info: &TensorInfo) -> anyhow::Result<(usize, usize)> {
    if info.dtype != DataType::Float32 || info.shape.len() < 3 {
        anyhow::bail!(
            "present KV output '{}' must be Float32 rank >= 3, got {:?} {:?}",
            info.name,
            info.dtype,
            info.shape
        );
    }
    let head_dim = *info
        .shape
        .last()
        .filter(|dim| **dim > 0)
        .with_context(|| format!("cannot infer KV head_dim from '{}'", info.name))?
        as usize;
    let num_kv_heads = info
        .shape
        .iter()
        .enumerate()
        .find_map(|(idx, &dim)| {
            (idx != 0 && idx + 1 != info.shape.len() && dim > 0).then_some(dim as usize)
        })
        .with_context(|| format!("cannot infer KV heads from '{}'", info.name))?;
    Ok((num_kv_heads, head_dim))
}

pub(crate) fn mirror_present_kv_to_pages(
    session: &Session,
    kv_model: &KvModelInfo,
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    outputs: &[Value],
    past_len: usize,
    input_len: usize,
) -> anyhow::Result<()> {
    let output_lookup = session
        .output_names()
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.as_str(), idx))
        .collect::<HashMap<_, _>>();
    let layer_data = kv_model
        .layers
        .iter()
        .map(|layer| {
            let key = outputs[*output_lookup
                .get(layer.key_present.as_str())
                .with_context(|| format!("missing output '{}'", layer.key_present))?]
            .to_vec_f32()?;
            let key_shape = outputs[*output_lookup
                .get(layer.key_present.as_str())
                .with_context(|| format!("missing output '{}'", layer.key_present))?]
            .shape()
            .to_vec();
            let value = outputs[*output_lookup
                .get(layer.value_present.as_str())
                .with_context(|| format!("missing output '{}'", layer.value_present))?]
            .to_vec_f32()?;
            let value_shape = outputs[*output_lookup
                .get(layer.value_present.as_str())
                .with_context(|| format!("missing output '{}'", layer.value_present))?]
            .shape()
            .to_vec();
            Ok((key, key_shape, value, value_shape))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    for offset in 0..input_len {
        let token_pos = past_len + offset;
        let owned_layers = layer_data
            .iter()
            .map(|(key, key_shape, value, value_shape)| {
                Ok((
                    extract_present_token(key, key_shape, kv_model.tensor_config, token_pos)?,
                    extract_present_token(value, value_shape, kv_model.tensor_config, token_pos)?,
                ))
            })
            .collect::<anyhow::Result<Vec<(Vec<f32>, Vec<f32>)>>>()?;
        let borrowed = owned_layers
            .iter()
            .map(|(key, value)| LayerKv {
                key: key.as_slice(),
                value: value.as_slice(),
            })
            .collect::<Vec<_>>();
        kv_cache
            .append_token_kv(seq, &borrowed)
            .map_err(|e| anyhow::anyhow!("Failed to mirror present KV into pages: {}", e))?;
    }
    Ok(())
}

pub(crate) fn extract_present_token(
    data: &[f32],
    shape: &[i64],
    config: PageTensorConfig,
    token_pos: usize,
) -> anyhow::Result<Vec<f32>> {
    let axes = kv_tensor_axes(shape, config, token_pos)?;
    let strides = row_major_strides(shape);
    let mut token = Vec::with_capacity(config.num_kv_heads * config.head_dim);
    for head in 0..config.num_kv_heads {
        for dim in 0..config.head_dim {
            let mut indices = vec![0_usize; shape.len()];
            indices[axes.head] = head;
            indices[axes.sequence] = token_pos;
            indices[axes.head_dim] = dim;
            let flat = indices
                .iter()
                .zip(strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum::<usize>();
            token.push(
                *data
                    .get(flat)
                    .context("present KV tensor index out of bounds")?,
            );
        }
    }
    Ok(token)
}

pub(crate) fn load_materialized_past(
    session: &Session,
    kv_model: &KvModelInfo,
    decode_state: &mut DecodeState,
    materialized: &onnx_genai_kv::MaterializedKv,
) -> anyhow::Result<()> {
    let input_shapes = session
        .inputs()
        .iter()
        .map(|info| (info.name.as_str(), info.shape.as_slice()))
        .collect::<HashMap<_, _>>();
    decode_state.past.clear();
    for (idx, layer) in kv_model.layers.iter().enumerate() {
        let key_shape = past_shape(
            input_shapes
                .get(layer.key_past.as_str())
                .copied()
                .context("missing key past input shape")?,
            materialized.sequence_len,
        )?;
        let value_shape = past_shape(
            input_shapes
                .get(layer.value_past.as_str())
                .copied()
                .context("missing value past input shape")?,
            materialized.sequence_len,
        )?;
        decode_state.past.insert(
            layer.key_past.clone(),
            Value::from_vec_f32(materialized.layers[idx].key.clone(), &key_shape)?,
        );
        decode_state.past.insert(
            layer.value_past.clone(),
            Value::from_vec_f32(materialized.layers[idx].value.clone(), &value_shape)?,
        );
    }
    Ok(())
}

pub(crate) fn past_shape(shape: &[i64], sequence_len: usize) -> anyhow::Result<Vec<i64>> {
    if shape.len() < 3 {
        anyhow::bail!("KV past shape rank must be >= 3, got {:?}", shape);
    }
    let seq_axis = shape.len() - 2;
    Ok(shape
        .iter()
        .enumerate()
        .map(|(axis, &dim)| {
            if axis == 0 {
                1
            } else if axis == seq_axis {
                sequence_len as i64
            } else {
                dim
            }
        })
        .collect())
}

pub(crate) fn attach_pages_to_sequence(
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    page_ids: &[PageId],
    len: usize,
) -> anyhow::Result<()> {
    if !kv_cache
        .page_table
        .get_sequence(seq)
        .context("sequence not found")?
        .is_empty()
    {
        anyhow::bail!("cannot attach prefix pages to a non-empty sequence");
    }
    for &page_id in page_ids {
        kv_cache.page_table.push_page(seq, page_id);
    }
    kv_cache.page_table.set_sequence_len(seq, len);
    Ok(())
}

pub(crate) fn rewind_target_state_to_len(
    session: &Session,
    kv_model: Option<&KvModelInfo>,
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    state: &mut EngineSession,
    len: usize,
) -> anyhow::Result<()> {
    state.tokens.truncate(len);
    rewind_decode_state_to_len(
        session,
        kv_model,
        kv_cache,
        seq,
        &mut state.decode_state,
        &mut state.kv_token_count,
        len,
    )
}

pub(crate) fn trim_overmaterialized_target_kv(
    session: &Session,
    kv_model: Option<&KvModelInfo>,
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    state: &mut EngineSession,
) -> anyhow::Result<()> {
    if state.kv_token_count > state.tokens.len() {
        rewind_target_state_to_len(session, kv_model, kv_cache, seq, state, state.tokens.len())?;
    }
    Ok(())
}

pub(crate) fn rewind_draft_state_to_len(
    draft_model: &mut DraftModel,
    state: &mut DraftSession,
    len: usize,
) -> anyhow::Result<()> {
    state.tokens.truncate(len);
    rewind_decode_state_to_len(
        &draft_model.session,
        draft_model.kv_model.as_ref(),
        &mut draft_model.kv_cache,
        state.seq,
        &mut state.decode_state,
        &mut state.kv_token_count,
        len,
    )
}

pub(crate) fn common_prefix_len(left: &[TokenId], right: &[TokenId]) -> usize {
    left.iter()
        .zip(right.iter())
        .take_while(|(left, right)| left == right)
        .count()
}

pub(crate) fn rewind_decode_state_to_len(
    session: &Session,
    kv_model: Option<&KvModelInfo>,
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    decode_state: &mut DecodeState,
    kv_token_count: &mut usize,
    len: usize,
) -> anyhow::Result<()> {
    if !decode_state.use_kv {
        *kv_token_count = 0;
        return Ok(());
    }
    if *kv_token_count == len {
        return Ok(());
    }
    if decode_state.has_runner() {
        kv_cache
            .rewind_to(seq, len)
            .map_err(|e| anyhow::anyhow!("Failed to rewind KV sequence {seq} to {len}: {}", e))?;
        decode_state.rewind_runner(len)?;
        *kv_token_count = len;
        return Ok(());
    }
    if kv_model.is_none() && *kv_token_count != len {
        anyhow::bail!("cannot rewind ORT KV tensors without paged KV materialization");
    }
    kv_cache
        .rewind_to(seq, len)
        .map_err(|e| anyhow::anyhow!("Failed to rewind KV sequence {seq} to {len}: {}", e))?;
    *kv_token_count = len;
    if len == 0 {
        decode_state.past.clear();
        return Ok(());
    }
    let kv_model = kv_model.context("missing KV model after rewind check")?;
    let materialized = kv_cache
        .materialize_sequence(seq)
        .map_err(|e| anyhow::anyhow!("Failed to materialize rewound KV sequence {seq}: {}", e))?;
    load_materialized_past(session, kv_model, decode_state, &materialized)
}

pub(crate) fn sequence_pages_for_len(
    kv_cache: &PagedKvCache,
    seq: SessionId,
    len: usize,
) -> anyhow::Result<Vec<PageId>> {
    let pages_needed = len.div_ceil(kv_cache.page_table.page_size);
    Ok(kv_cache
        .page_table
        .get_sequence(seq)
        .with_context(|| format!("sequence {seq} not found"))?
        .iter()
        .copied()
        .take(pages_needed)
        .collect())
}

struct KvTensorAxes {
    head: usize,
    sequence: usize,
    head_dim: usize,
}

fn kv_tensor_axes(
    shape: &[i64],
    config: PageTensorConfig,
    token_pos: usize,
) -> anyhow::Result<KvTensorAxes> {
    let head_dim = shape
        .iter()
        .rposition(|&dim| dim == config.head_dim as i64)
        .context("KV tensor head_dim axis not found")?;
    let head = shape
        .iter()
        .enumerate()
        .find_map(|(idx, &dim)| {
            (idx != head_dim && dim == config.num_kv_heads as i64).then_some(idx)
        })
        .context("KV tensor head axis not found")?;
    let sequence = shape
        .iter()
        .enumerate()
        .find_map(|(idx, &dim)| {
            (idx != head && idx != head_dim && dim as usize > token_pos).then_some(idx)
        })
        .context("KV tensor sequence axis not found")?;
    Ok(KvTensorAxes {
        head,
        sequence,
        head_dim,
    })
}

pub(crate) fn row_major_strides(shape: &[i64]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for idx in (0..shape.len().saturating_sub(1)).rev() {
        strides[idx] = strides[idx + 1] * shape[idx + 1] as usize;
    }
    strides
}

pub(crate) fn kv_layer_index(name: &str) -> Option<usize> {
    name.split(|ch: char| !ch.is_ascii_digit())
        .find(|part| !part.is_empty())
        .and_then(|part| part.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{ModelDecodePath, run_decode_session_logits};
    use onnx_genai_kv::{KvCacheOps, MaterializedLayerKv};
    use onnx_genai_ort::{Environment, SessionOptions};
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, MutexGuard, OnceLock};

    fn model_test_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name)
            .join("model.onnx")
    }

    fn load_session(name: &str) -> anyhow::Result<(Environment, Session)> {
        let environment = Environment::new("kv-bridge-tests")?;
        let session = Session::new(
            &environment,
            &fixture(name),
            SessionOptions::default().with_intra_op_threads(1),
        )?;
        Ok((environment, session))
    }

    fn tensor_config() -> PageTensorConfig {
        PageTensorConfig {
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 2,
            page_size: 2,
            dtype: KvDType::F32,
        }
    }

    fn append_token(cache: &mut PagedKvCache, seq: SessionId, base: f32) {
        let key = [base, base + 1.0, base + 2.0, base + 3.0];
        let value = [base + 10.0, base + 11.0, base + 12.0, base + 13.0];
        cache
            .append_token_kv(
                seq,
                &[LayerKv {
                    key: &key,
                    value: &value,
                }],
            )
            .unwrap();
    }

    fn infers_past_present_model_and_rejects_static_cache_as_kv_bridge_model() -> anyhow::Result<()>
    {
        let _guard = model_test_lock();
        let (_environment, session) = load_session("tiny-llm")?;
        let info = infer_kv_model_info(&session, 4)?.expect("past/present KV model");
        assert_eq!(
            info.tensor_config,
            PageTensorConfig {
                num_layers: 1,
                num_kv_heads: 2,
                head_dim: 8,
                page_size: 4,
                dtype: KvDType::F32,
            }
        );
        assert_eq!(info.layers[0].key_present, "present.0.key");
        assert_eq!(info.layers[0].value_present, "present.0.value");
        assert_eq!(info.layers[0].key_past, "past_key_values.0.key");
        assert_eq!(info.layers[0].value_past, "past_key_values.0.value");

        let (_environment, static_session) = load_session("tiny-llm-scatter")?;
        assert!(infer_kv_model_info(&static_session, 4)?.is_none());
        Ok(())
    }

    #[test]
    fn validates_kv_metadata_and_shape_helpers() {
        let valid = TensorInfo {
            name: "present.7.key".into(),
            dtype: DataType::Float32,
            shape: vec![-1, 2, -1, 8],
        };
        assert_eq!(infer_kv_heads_and_head_dim(&valid).unwrap(), (2, 8));
        assert_eq!(kv_layer_index(&valid.name), Some(7));
        assert_eq!(kv_layer_index("present.key"), None);
        assert_eq!(past_shape(&[-1, 2, -1, 8], 3).unwrap(), [1, 2, 3, 8]);
        assert_eq!(row_major_strides(&[1, 2, 3, 4]), [24, 12, 4, 1]);

        let wrong_dtype = TensorInfo {
            dtype: DataType::Int64,
            ..valid.clone()
        };
        assert!(
            infer_kv_heads_and_head_dim(&wrong_dtype)
                .unwrap_err()
                .to_string()
                .contains("must be Float32 rank >= 3")
        );
        let unknown_head_dim = TensorInfo {
            shape: vec![-1, 2, -1, -1],
            ..valid
        };
        assert!(
            infer_kv_heads_and_head_dim(&unknown_head_dim)
                .unwrap_err()
                .to_string()
                .contains("cannot infer KV head_dim")
        );
        assert!(past_shape(&[1, 2], 3).is_err());
    }

    #[test]
    fn extracts_tokens_from_present_tensor_and_reports_bad_layouts() {
        let config = PageTensorConfig {
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 2,
            page_size: 2,
            dtype: KvDType::F32,
        };
        let data = (0..12).map(|value| value as f32).collect::<Vec<_>>();
        assert_eq!(
            extract_present_token(&data, &[1, 2, 3, 2], config, 1).unwrap(),
            [2.0, 3.0, 8.0, 9.0]
        );
        assert!(
            extract_present_token(&data, &[1, 2, 3, 4], config, 1)
                .unwrap_err()
                .to_string()
                .contains("head axis not found")
        );
        assert!(
            extract_present_token(&data[..4], &[1, 2, 3, 2], config, 2)
                .unwrap_err()
                .to_string()
                .contains("index out of bounds")
        );
    }

    fn mirrors_present_append_range_into_paged_cache() -> anyhow::Result<()> {
        let _guard = model_test_lock();
        let (_environment, session) = load_session("tiny-llm")?;
        let model = infer_kv_model_info(&session, 2)?.unwrap();
        let mut cache = PagedKvCache::new_with_tensor_config(model.tensor_config, 4);
        let seq = cache.create_sequence();

        let logits = Value::from_vec_f32(vec![0.0; 2 * 32], &[1, 2, 32])?;
        let key = (0..64).map(|value| value as f32).collect::<Vec<_>>();
        let value = (100..164).map(|value| value as f32).collect::<Vec<_>>();
        let outputs = vec![
            logits,
            Value::from_vec_f32(key, &[1, 2, 4, 8])?,
            Value::from_vec_f32(value, &[1, 2, 4, 8])?,
        ];

        mirror_present_kv_to_pages(&session, &model, &mut cache, seq, &outputs, 1, 2)?;
        let materialized = cache.materialize_sequence(seq)?;
        assert_eq!(materialized.sequence_len, 2);
        assert_eq!(
            materialized.layers[0].key,
            [
                8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
                22.0, 23.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0,
                52.0, 53.0, 54.0, 55.0,
            ]
        );
        assert_eq!(
            &materialized.layers[0].value[..8],
            &[108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0]
        );

        let missing_seq =
            mirror_present_kv_to_pages(&session, &model, &mut cache, 999, &outputs, 1, 1)
                .unwrap_err();
        assert!(
            missing_seq
                .to_string()
                .contains("Failed to mirror present KV")
        );
        Ok(())
    }

    fn materializes_past_values_in_model_input_layout() -> anyhow::Result<()> {
        let _guard = model_test_lock();
        let (_environment, session) = load_session("tiny-llm")?;
        let model = infer_kv_model_info(&session, 2)?.unwrap();
        let materialized = onnx_genai_kv::MaterializedKv {
            sequence_len: 2,
            num_kv_heads: 2,
            head_dim: 8,
            layers: vec![MaterializedLayerKv {
                key: (0..32).map(|value| value as f32).collect(),
                value: (100..132).map(|value| value as f32).collect(),
            }],
        };
        let mut state = DecodeState::new(&session)?;
        state
            .past
            .insert("stale".into(), Value::from_vec_f32(vec![0.0], &[1])?);

        load_materialized_past(&session, &model, &mut state, &materialized)?;

        assert!(!state.past.contains_key("stale"));
        let key = &state.past["past_key_values.0.key"];
        assert_eq!(key.shape(), &[1, 2, 2, 8]);
        assert_eq!(key.to_vec_f32()?, materialized.layers[0].key);
        assert_eq!(
            state.past["past_key_values.0.value"].to_vec_f32()?,
            materialized.layers[0].value
        );
        Ok(())
    }

    #[test]
    fn attaches_prefix_pages_and_selects_only_pages_needed_for_length() -> anyhow::Result<()> {
        let mut cache = PagedKvCache::new_with_tensor_config(tensor_config(), 8);
        let source = cache.create_sequence();
        for base in [0.0, 10.0, 20.0] {
            append_token(&mut cache, source, base);
        }
        let source_pages = sequence_pages_for_len(&cache, source, 3)?;
        assert_eq!(source_pages.len(), 2);
        assert_eq!(sequence_pages_for_len(&cache, source, 1)?.len(), 1);

        let target = cache.create_sequence();
        attach_pages_to_sequence(&mut cache, target, &source_pages, 3)?;
        assert_eq!(
            cache.materialize_sequence(target)?,
            cache.materialize_sequence(source)?
        );
        assert!(
            attach_pages_to_sequence(&mut cache, target, &source_pages, 3)
                .unwrap_err()
                .to_string()
                .contains("non-empty sequence")
        );
        assert!(attach_pages_to_sequence(&mut cache, 999, &source_pages, 3).is_err());
        assert!(sequence_pages_for_len(&cache, 999, 1).is_err());
        Ok(())
    }

    fn rewinds_materialized_ort_past_and_handles_edge_branches() -> anyhow::Result<()> {
        let _guard = model_test_lock();
        let (_environment, session) = load_session("tiny-llm")?;
        let model = infer_kv_model_info(&session, 2)?.unwrap();
        let mut cache = PagedKvCache::new_with_tensor_config(model.tensor_config, 8);
        let seq = cache.create_sequence();
        for base in [0.0, 10.0, 20.0] {
            let key = vec![base; 16];
            let value = vec![base + 1.0; 16];
            cache.append_token_kv(
                seq,
                &[LayerKv {
                    key: &key,
                    value: &value,
                }],
            )?;
        }
        let mut decode_state = DecodeState::new(&session)?;
        let mut count = 3;

        rewind_decode_state_to_len(
            &session,
            Some(&model),
            &mut cache,
            seq,
            &mut decode_state,
            &mut count,
            2,
        )?;
        assert_eq!(count, 2);
        assert_eq!(
            decode_state.past["past_key_values.0.key"].shape(),
            &[1, 2, 2, 8]
        );

        rewind_decode_state_to_len(
            &session,
            Some(&model),
            &mut cache,
            seq,
            &mut decode_state,
            &mut count,
            0,
        )?;
        assert_eq!(count, 0);
        assert!(decode_state.past.is_empty());

        count = 1;
        let error = rewind_decode_state_to_len(
            &session,
            None,
            &mut cache,
            seq,
            &mut decode_state,
            &mut count,
            0,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("without paged KV materialization")
        );

        decode_state.use_kv = false;
        rewind_decode_state_to_len(
            &session,
            None,
            &mut cache,
            seq,
            &mut decode_state,
            &mut count,
            5,
        )?;
        assert_eq!(count, 0);
        Ok(())
    }

    fn rewinds_static_and_past_present_decode_runners() -> anyhow::Result<()> {
        let _guard = model_test_lock();
        for (fixture_name, path) in [
            (
                "tiny-llm-scatter",
                ModelDecodePath::StaticCache { max_len: 16 },
            ),
            (
                "tiny-llm",
                ModelDecodePath::PastPresent {
                    shared_buffer: false,
                    max_len: None,
                },
            ),
        ] {
            let (_environment, session) = load_session(fixture_name)?;
            let mut state = DecodeState::new_for_path(&session, &path)?;
            let mut cache = PagedKvCache::new(2, 8);
            let seq = cache.create_sequence();
            run_decode_session_logits(&mut state, &[2, 4, 3], 0)?;
            cache.append(seq, 3)?;
            let mut count = 3;

            rewind_decode_state_to_len(&session, None, &mut cache, seq, &mut state, &mut count, 1)?;

            assert_eq!(count, 1);
            assert_eq!(state.runner_len(), 1);
            assert_eq!(cache.len(seq)?, 1);
        }
        Ok(())
    }

    fn rewinds_target_state_and_trims_overmaterialized_kv() -> anyhow::Result<()> {
        let _guard = model_test_lock();
        let (_environment, session) = load_session("tiny-llm-scatter")?;
        let mut cache = PagedKvCache::new(2, 8);
        let seq = cache.create_sequence();
        cache.append(seq, 4)?;
        let mut state = EngineSession {
            tokens: vec![2, 4, 3],
            kv_token_count: 4,
            decode_state: DecodeState::new(&session)?,
            draft: None,
        };

        trim_overmaterialized_target_kv(&session, None, &mut cache, seq, &mut state)?;
        assert_eq!(state.kv_token_count, 0);
        assert_eq!(cache.len(seq)?, 4);

        state.decode_state.use_kv = true;
        state.kv_token_count = 4;
        rewind_target_state_to_len(&session, None, &mut cache, seq, &mut state, 4)?;
        assert_eq!(state.tokens, [2, 4, 3]);
        assert_eq!(state.kv_token_count, 4);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 4, 5]), 2);
        Ok(())
    }

    #[test]
    fn model_backed_bridge_paths_are_deterministic() -> anyhow::Result<()> {
        infers_past_present_model_and_rejects_static_cache_as_kv_bridge_model()?;
        mirrors_present_append_range_into_paged_cache()?;
        materializes_past_values_in_model_input_layout()?;
        rewinds_materialized_ort_past_and_handles_edge_branches()?;
        rewinds_static_and_past_present_decode_runners()?;
        rewinds_target_state_and_trims_overmaterialized_kv()
    }
}
