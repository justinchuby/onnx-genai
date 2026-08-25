//! KV model metadata, paged-cache mirroring, and rewind helpers.

use crate::config::SessionId;
use crate::decode::DecodeState;
use crate::kv_sizing::{KvDimension, KvStorageType, KvTensorSpec};
use crate::logits::TokenId;
use crate::session::{DraftModel, DraftSession, EngineSession};
use anyhow::Context;
use onnx_genai_kv::{
    KvCacheOps, KvDType, KvLayerPayload, KvPayload, KvPayloadDtype, LayerKv, LayerTensorConfig,
    PageId, PageTensorConfig, PagedKvCache,
};
use onnx_genai_ort::{DataType, GraphIo, Session, TensorInfo, Value};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(crate) struct KvModelInfo {
    /// Representative (layer-0) geometry. Authoritative for `num_layers`,
    /// `page_size` and `dtype`; the `num_kv_heads`/`head_dim` fields mirror
    /// layer 0 and are only valid to read directly for uniform models. Use
    /// [`KvModelInfo::layer_configs`] / [`KvModelInfo::layer_tensor_config`] for
    /// per-layer geometry.
    pub(crate) tensor_config: PageTensorConfig,
    /// Per-layer KV geometry, indexed identically to [`KvModelInfo::layers`].
    /// Length equals the number of exported KV layers (post kv-sharing).
    pub(crate) layer_configs: Vec<LayerTensorConfig>,
    /// Native past tensors, retained independently so asymmetric K/V geometry
    /// and storage types are charged exactly as the allocator sizes them.
    #[cfg_attr(
        all(not(feature = "native-backend"), not(test)),
        expect(
            dead_code,
            reason = "native KV tensors are consumed only by the native backend"
        )
    )]
    pub(crate) native_kv_tensors: Vec<KvTensorSpec>,
    pub(crate) layers: Vec<KvLayerIo>,
}

impl KvModelInfo {
    /// The tensor config for a single exported KV layer, carrying that layer's
    /// own `num_kv_heads`/`head_dim` while inheriting the shared `num_layers`,
    /// `page_size` and `dtype`.
    pub(crate) fn layer_tensor_config(&self, layer: usize) -> PageTensorConfig {
        let geom = self.layer_configs[layer];
        PageTensorConfig {
            num_layers: self.tensor_config.num_layers,
            num_kv_heads: geom.num_kv_heads,
            head_dim: geom.head_dim,
            page_size: self.tensor_config.page_size,
            dtype: self.tensor_config.dtype,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct KvLayerIo {
    pub(crate) key_present: String,
    pub(crate) value_present: String,
    pub(crate) key_past: String,
    pub(crate) value_past: String,
}

/// One self-attention layer's four paged-KV port names, resolved WITHOUT
/// reading any tensor name.
struct ResolvedKvLayer {
    key_past: String,
    key_present: String,
    value_past: String,
    value_present: String,
    key_present_info: TensorInfo,
    key_past_info: TensorInfo,
    value_past_info: TensorInfo,
}

pub(crate) fn infer_kv_model_info(
    graph: &dyn GraphIo,
    io: Option<&onnx_genai_metadata::DecoderAbi>,
    page_size: usize,
    dtype: KvDType,
) -> anyhow::Result<Option<KvModelInfo>> {
    let Some(resolved_layers) = resolve_kv_layers(graph, io)? else {
        return Ok(None);
    };

    let key_outputs = resolved_layers
        .iter()
        .map(|layer| layer.key_present_info.clone())
        .collect::<Vec<_>>();
    let layer_configs = layer_configs_from_key_outputs(&key_outputs)?;
    let native_kv_tensors = resolved_layers
        .iter()
        .flat_map(|layer| [&layer.key_past_info, &layer.value_past_info])
        .map(native_kv_tensor_spec)
        .collect::<anyhow::Result<Vec<_>>>()?;
    // Representative geometry (layer 0). The paged cache is built from the full
    // per-layer `layer_configs`; `tensor_config` remains a single-value view for
    // uniform-only consumers (connector payloads, num_layers/dtype).
    let config = PageTensorConfig {
        num_layers: resolved_layers.len(),
        num_kv_heads: layer_configs[0].num_kv_heads,
        head_dim: layer_configs[0].head_dim,
        page_size,
        dtype,
    };
    let layers = resolved_layers
        .into_iter()
        .map(|layer| KvLayerIo {
            key_present: layer.key_present,
            value_present: layer.value_present,
            key_past: layer.key_past,
            value_past: layer.value_past,
        })
        .collect();

    Ok(Some(KvModelInfo {
        tensor_config: config,
        layer_configs,
        native_kv_tensors,
        layers,
    }))
}

fn native_kv_tensor_spec(info: &TensorInfo) -> anyhow::Result<KvTensorSpec> {
    if !is_supported_kv_dtype(info.dtype) {
        anyhow::bail!(
            "past KV input '{}' has dtype {:?}, which the paged KV cache cannot store \
             (its pages hold Float32, Float16, or BFloat16). The declaration is well \
             formed; this backend lacks the capability.",
            info.name,
            info.dtype
        );
    }
    let shape = info
        .shape
        .iter()
        .enumerate()
        .map(|(axis, &dim)| {
            if dim >= 0 {
                KvDimension::Fixed(dim as u64)
            } else if axis == 0 {
                KvDimension::PerSequenceBatch
            } else {
                KvDimension::Context
            }
        })
        .collect();
    Ok(KvTensorSpec {
        name: info.name.clone(),
        dtype: match info.dtype {
            DataType::Float32 => KvStorageType::Float32,
            DataType::Float16 => KvStorageType::Float16,
            DataType::BFloat16 => KvStorageType::BFloat16,
            _ => unreachable!("validated above"),
        },
        shape,
    })
}

/// Resolve the paged-KV self-attention layer ports strictly from explicit
/// metadata — the runtime never reads a tensor name to decide which port is
/// key/value, which layer it belongs to, or how past pairs with present.
///
/// `kv_inputs`/`kv_outputs` are equal-length, positionally paired
/// past<->present lists ordered as consecutive per-layer `[key_i, value_i]`
/// pairs (the exporter contract, matching `genai-config` `expand_kv`). Layer `l`
/// therefore binds index `2*l` (key) and `2*l + 1` (value).
///
/// Returns `Ok(None)` when the graph carries no explicitly declared paged KV
/// pairs: either the I/O metadata declares none (a static-cache or non-KV
/// graph), or no I/O metadata is present at all. Paged-KV geometry is built
/// only from an explicit `kv_inputs`/`kv_outputs` declaration — never inferred
/// from a tensor name or shape, since a growing paged present output is
/// shape-indistinguishable from a static-cache buffer or a logits/hidden
/// output. The decode-path resolver (`decode::resolved_io`) independently fails
/// closed, naming `kv_inputs`/`kv_outputs`, when a decoder genuinely
/// carries unpaired KV state without an explicit declaration.
fn resolve_kv_layers(
    graph: &dyn GraphIo,
    io: Option<&onnx_genai_metadata::DecoderAbi>,
) -> anyhow::Result<Option<Vec<ResolvedKvLayer>>> {
    let (kv_inputs, kv_outputs) = match io.map(|io| (io.kv_inputs.as_ref(), io.kv_outputs.as_ref()))
    {
        Some((Some(inputs), Some(outputs))) => (inputs, outputs),
        // Metadata is present and explicitly declares no growing KV pairs (a
        // static-cache or non-KV graph): there is no paged bridge to build.
        Some((None, None)) => return Ok(None),
        Some((Some(_), None)) | Some((None, Some(_))) => {
            anyhow::bail!("kv_inputs and kv_outputs must be declared together")
        }
        // No I/O metadata at all: never guess KV pairing from tensor names or
        // shapes. The engine simply builds no paged cache; KV correctness is
        // enforced by the decode-path resolver, which fails closed on unpaired
        // KV state and names the exact missing metadata key.
        None => return Ok(None),
    };

    let port_layers = pair_kv_ports(kv_inputs, kv_outputs)?;
    if port_layers.is_empty() {
        return Ok(None);
    }
    let mut layers = Vec::with_capacity(port_layers.len());
    for ports in port_layers {
        let key_present_info = require_present_kv_output(graph, &ports.key_present)?;
        require_present_kv_output(graph, &ports.value_present)?;
        let key_past_info = require_kv_input(graph, &ports.key_past)?;
        let value_past_info = require_kv_input(graph, &ports.value_past)?;
        layers.push(ResolvedKvLayer {
            key_past: ports.key_past,
            key_present: ports.key_present,
            value_past: ports.value_past,
            value_present: ports.value_present,
            key_present_info,
            key_past_info,
            value_past_info,
        });
    }
    Ok(Some(layers))
}

/// One layer's four KV port NAMES, before any ONNX-graph lookup.
#[derive(Debug)]
struct KvLayerPortNames {
    key_past: String,
    key_present: String,
    value_past: String,
    value_present: String,
}

/// Group the flat, positionally-paired `kv_inputs`/`kv_outputs` port lists into
/// per-layer `[key, value]` quadruples. Pure and name-free: it validates only
/// list STRUCTURE (equal length, even count) and never inspects the ONNX graph
/// or a tensor name. Errors name the exact offending metadata key so an
/// exporter can fix the declaration. An empty (but valid) list yields no layers.
fn pair_kv_ports(
    kv_inputs: &[String],
    kv_outputs: &[String],
) -> anyhow::Result<Vec<KvLayerPortNames>> {
    if kv_inputs.len() != kv_outputs.len() {
        anyhow::bail!(
            "kv_inputs ({}) and kv_outputs ({}) must have equal length",
            kv_inputs.len(),
            kv_outputs.len()
        );
    }
    // Each self-attention layer contributes a [key, value] pair, so the flat
    // positional lists must contain an even number of ports.
    if !kv_inputs.len().is_multiple_of(2) {
        anyhow::bail!(
            "kv_inputs/kv_outputs declare {} KV ports; a paged self-attention cache \
             pairs them as per-layer [key, value], which requires an even count",
            kv_inputs.len()
        );
    }

    Ok((0..kv_inputs.len() / 2)
        .map(|layer| {
            let key_index = layer * 2;
            let value_index = key_index + 1;
            KvLayerPortNames {
                key_past: kv_inputs[key_index].clone(),
                key_present: kv_outputs[key_index].clone(),
                value_past: kv_inputs[value_index].clone(),
                value_present: kv_outputs[value_index].clone(),
            }
        })
        .collect())
}

/// Look up a declared present-KV output and validate its element type, returning
/// its shape record for structural geometry inference. Errors name the exact
/// missing/invalid `kv_outputs` port.
fn require_present_kv_output(graph: &dyn GraphIo, name: &str) -> anyhow::Result<TensorInfo> {
    let info = graph
        .outputs()
        .iter()
        .find(|output| output.name == name)
        .cloned()
        .with_context(|| {
            format!("declared kv_outputs port '{name}' is not exposed by the graph")
        })?;
    if !is_supported_kv_dtype(info.dtype) {
        anyhow::bail!(
            "KV present output '{name}' has dtype {:?}, which the paged KV cache cannot \
             store (its pages hold Float32, Float16, or BFloat16). The declaration is \
             well formed; this backend lacks the capability.",
            info.dtype
        );
    }
    Ok(info)
}

/// Validate that a declared past-KV input exists; errors name the exact missing
/// `kv_inputs` port.
fn require_kv_input(graph: &dyn GraphIo, name: &str) -> anyhow::Result<TensorInfo> {
    graph
        .inputs()
        .iter()
        .find(|input| input.name == name)
        .cloned()
        .with_context(|| format!("declared kv_inputs port '{name}' is not exposed by the graph"))
}

/// Build the per-layer KV geometry from each exported present-KV output shape.
///
/// The returned vector has one [`LayerTensorConfig`] per exported KV layer, in
/// the same order as `key_outputs` (sorted by layer index). Because a model
/// exports only `num_hidden_layers - num_kv_shared_layers` KV entries — the last
/// `num_kv_shared_layers` layers reuse an earlier layer's K/V — this vector's
/// length equals the number of *exported* KV entries, and metadata
/// `shared_kv.target_layers` indices map directly onto these positions (post
/// kv-sharing). Different layers may declare different head_dim (e.g. Gemma-4
/// sliding=256 vs full=512); geometry is read structurally from the ONNX I/O
/// shapes, never from model names.
pub(crate) fn layer_configs_from_key_outputs(
    key_outputs: &[TensorInfo],
) -> anyhow::Result<Vec<LayerTensorConfig>> {
    key_outputs
        .iter()
        .map(|info| {
            let (num_kv_heads, head_dim) = infer_kv_heads_and_head_dim(info)?;
            Ok(LayerTensorConfig::new(num_kv_heads, head_dim))
        })
        .collect()
}

pub(crate) fn infer_kv_heads_and_head_dim(info: &TensorInfo) -> anyhow::Result<(usize, usize)> {
    if !is_supported_kv_dtype(info.dtype) || info.shape.len() < 3 {
        anyhow::bail!(
            "present KV output '{}' must be Float32, Float16, or BFloat16 rank >= 3, got {:?} {:?}",
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

/// KV element types the *paged* KV cache can store.
///
/// This is a storage capability of this backend, not a property of the metadata
/// format: `KvStorageType` has fp32 and 16-bit float pages and nothing narrower,
/// and the host paged-mirror path widens 16-bit float values to fp32 page
/// storage. A package may legitimately declare an FP8 cache — the format can
/// name `float8_e4m3fn` — and it will be refused here, by name, because these
/// pages cannot hold it. Callers surface that as an unsupported-capability
/// error so the reader knows to change backend or EP, not to edit the
/// document.
fn is_supported_kv_dtype(dtype: DataType) -> bool {
    matches!(
        dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    )
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
            .to_vec_f32_lossy()?;
            let key_shape = outputs[*output_lookup
                .get(layer.key_present.as_str())
                .with_context(|| format!("missing output '{}'", layer.key_present))?]
            .shape()
            .to_vec();
            let value = outputs[*output_lookup
                .get(layer.value_present.as_str())
                .with_context(|| format!("missing output '{}'", layer.value_present))?]
            .to_vec_f32_lossy()?;
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
            .enumerate()
            .map(|(layer_idx, (key, key_shape, value, value_shape))| {
                let layer_config = kv_model.layer_tensor_config(layer_idx);
                Ok((
                    extract_present_token(key, key_shape, layer_config, token_pos)?,
                    extract_present_token(value, value_shape, layer_config, token_pos)?,
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
            // Keep the `KvError` as the source rather than flattening it to a
            // string: callers distinguish a full pool (recoverable — stop
            // mirroring) from a real fault (not recoverable) by downcasting.
            .context("Failed to mirror present KV into pages")?;
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
    decode_state.set_past(
        materialized_past_values(session, kv_model, materialized)?,
        materialized.sequence_len,
    );
    Ok(())
}

fn materialized_past_values(
    session: &Session,
    kv_model: &KvModelInfo,
    materialized: &onnx_genai_kv::MaterializedKv,
) -> anyhow::Result<HashMap<String, Value>> {
    if materialized.start_position != 0 || materialized.sink_len != 0 {
        anyhow::bail!(
            "cannot load paged KV starting at absolute position {} (sink_len {}) into a contiguous past/present graph; discontinuous attention-sink positions require explicit position_ids support",
            materialized.start_position,
            materialized.sink_len
        );
    }
    let input_shapes = session
        .inputs()
        .iter()
        .map(|info| (info.name.as_str(), info.shape.as_slice()))
        .collect::<HashMap<_, _>>();
    let input_dtypes = session
        .inputs()
        .iter()
        .map(|info| (info.name.as_str(), info.dtype))
        .collect::<HashMap<_, _>>();
    let mut past = HashMap::new();
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
        // Paged storage holds KV widened to f32; narrow back to the graph's
        // declared past-input dtype (e.g. fp16 GQA exports) so the injected KV
        // matches the model contract. For an fp16 model this is the exact
        // inverse of the fp16->f32 widening done when mirroring present KV.
        let key_dtype = input_dtypes
            .get(layer.key_past.as_str())
            .copied()
            .context("missing key past input dtype")?;
        let value_dtype = input_dtypes
            .get(layer.value_past.as_str())
            .copied()
            .context("missing value past input dtype")?;
        past.insert(
            layer.key_past.clone(),
            Value::from_f32_slice_as(&materialized.layers[idx].key, &key_shape, key_dtype)?,
        );
        past.insert(
            layer.value_past.clone(),
            Value::from_f32_slice_as(&materialized.layers[idx].value, &value_shape, value_dtype)?,
        );
    }
    Ok(past)
}

pub(crate) fn past_shape(shape: &[i64], sequence_len: usize) -> anyhow::Result<Vec<i64>> {
    if shape.len() < 3 {
        anyhow::bail!("KV past shape rank must be >= 3, got {shape:?}");
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RewindRunnerPolicy {
    AllowRunnerRewind,
    RejectRunnerRewind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RewindRequest {
    pub(crate) len: usize,
    pub(crate) runner_policy: RewindRunnerPolicy,
}

impl RewindRequest {
    pub(crate) fn new(len: usize, runner_policy: RewindRunnerPolicy) -> Self {
        Self { len, runner_policy }
    }
}

pub(crate) fn rewind_target_state_to_len(
    session: &Session,
    kv_model: Option<&KvModelInfo>,
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    state: &mut EngineSession,
    request: RewindRequest,
) -> anyhow::Result<()> {
    validate_target_state_rewind_to_len(kv_model, kv_cache, seq, state, request)?;
    state.tokens.truncate(request.len);
    rewind_decode_state_to_len(
        session,
        kv_model,
        kv_cache,
        seq,
        &mut state.decode_state,
        &mut state.kv_token_count,
        request,
    )
}

pub(crate) fn validate_target_state_rewind_to_len(
    kv_model: Option<&KvModelInfo>,
    kv_cache: &PagedKvCache,
    seq: SessionId,
    state: &EngineSession,
    request: RewindRequest,
) -> anyhow::Result<()> {
    validate_decode_state_rewind_to_len(
        kv_model,
        kv_cache,
        seq,
        &state.decode_state,
        state.kv_token_count,
        request,
    )
}

pub(crate) fn trim_overmaterialized_target_kv(
    session: &Session,
    kv_model: Option<&KvModelInfo>,
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    state: &mut EngineSession,
    runner_policy: RewindRunnerPolicy,
) -> anyhow::Result<()> {
    if state.kv_token_count > state.tokens.len() {
        rewind_target_state_to_len(
            session,
            kv_model,
            kv_cache,
            seq,
            state,
            RewindRequest::new(state.tokens.len(), runner_policy),
        )?;
    }
    Ok(())
}

pub(crate) fn rewind_draft_state_to_len(
    draft_model: &mut DraftModel,
    state: &mut DraftSession,
    request: RewindRequest,
) -> anyhow::Result<()> {
    validate_draft_state_rewind_to_len(draft_model, state, request)?;
    state.tokens.truncate(request.len);
    rewind_decode_state_to_len(
        &draft_model.session,
        draft_model.kv_model.as_ref(),
        &mut draft_model.kv_cache,
        state.seq,
        &mut state.decode_state,
        &mut state.kv_token_count,
        request,
    )
}

pub(crate) fn validate_draft_state_rewind_to_len(
    draft_model: &DraftModel,
    state: &DraftSession,
    request: RewindRequest,
) -> anyhow::Result<()> {
    validate_decode_state_rewind_to_len(
        draft_model.kv_model.as_ref(),
        &draft_model.kv_cache,
        state.seq,
        &state.decode_state,
        state.kv_token_count,
        request,
    )
}

pub(crate) fn common_prefix_len(left: &[TokenId], right: &[TokenId]) -> usize {
    left.iter()
        .zip(right.iter())
        .take_while(|(left, right)| left == right)
        .count()
}

pub(crate) fn validate_decode_state_rewind_to_len(
    kv_model: Option<&KvModelInfo>,
    kv_cache: &PagedKvCache,
    seq: SessionId,
    decode_state: &DecodeState,
    kv_token_count: usize,
    request: RewindRequest,
) -> anyhow::Result<()> {
    let len = request.len;
    decode_state.validate_rewind_to_len(
        kv_token_count,
        len,
        kv_model.is_some(),
        request.runner_policy,
    )?;
    if decode_state.use_kv
        && kv_token_count != len
        && (decode_state.has_runner() || decode_state.is_windowed() || kv_model.is_some())
    {
        kv_cache.validate_rewind_to(seq, len).map_err(|e| {
            anyhow::anyhow!("Failed to validate KV sequence {seq} rewind to {len}: {e}")
        })?;
    }
    Ok(())
}

pub(crate) fn rewind_decode_state_to_len(
    session: &Session,
    kv_model: Option<&KvModelInfo>,
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    decode_state: &mut DecodeState,
    kv_token_count: &mut usize,
    request: RewindRequest,
) -> anyhow::Result<()> {
    let len = request.len;
    validate_decode_state_rewind_to_len(
        kv_model,
        kv_cache,
        seq,
        decode_state,
        *kv_token_count,
        request,
    )?;
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
            .map_err(|e| anyhow::anyhow!("Failed to rewind KV sequence {seq} to {len}: {e}"))?;
        decode_state.rewind_runner(len)?;
        *kv_token_count = len;
        return Ok(());
    }
    if decode_state.is_windowed() {
        decode_state.rewind_windowed(*kv_token_count, len)?;
        kv_cache
            .rewind_to(seq, len)
            .map_err(|e| anyhow::anyhow!("Failed to rewind KV sequence {seq} to {len}: {e}"))?;
        *kv_token_count = len;
        return Ok(());
    }
    if kv_model.is_none() && *kv_token_count != len {
        anyhow::bail!("cannot rewind ORT KV tensors without paged KV materialization");
    }
    if len == 0 {
        kv_cache
            .rewind_to(seq, len)
            .map_err(|e| anyhow::anyhow!("Failed to rewind KV sequence {seq} to {len}: {e}"))?;
        *kv_token_count = len;
        decode_state.set_past(HashMap::new(), 0);
        return Ok(());
    }
    let kv_model = kv_model.context("missing KV model after rewind check")?;
    // Materialize the rewound view *before* mutating anything. The previous
    // shape cloned the entire page pool, rewound the copy, and read from it --
    // duplicating every page's storage to answer a question about one
    // sequence. Reading first gives the same guarantee (a failed read leaves
    // the sequence untouched) without the copy.
    let materialized = kv_cache
        .materialize_sequence_to(seq, len)
        .map_err(|e| anyhow::anyhow!("Failed to materialize rewound KV sequence {seq}: {e}"))?;
    let past = materialized_past_values(session, kv_model, &materialized)?;
    kv_cache
        .rewind_to(seq, len)
        .map_err(|e| anyhow::anyhow!("Failed to rewind KV sequence {seq} to {len}: {e}"))?;
    *kv_token_count = len;
    decode_state.set_past(past, len);
    Ok(())
}

pub(crate) fn sequence_pages_for_len(
    kv_cache: &PagedKvCache,
    seq: SessionId,
    len: usize,
) -> anyhow::Result<Vec<PageId>> {
    let pages_needed = len.div_ceil(kv_cache.page_table.page_size);
    let pages = kv_cache
        .page_table
        .get_sequence(seq)
        .with_context(|| format!("sequence {seq} not found"))?;
    // Refuse to hand back a short list. These pages get published under a key
    // that says they hold `len` tokens, so silently returning fewer would let a
    // later request attach pages missing the KV they claim to have and decode
    // from whatever those pages held before — wrong output, no error anywhere.
    if pages.len() < pages_needed {
        anyhow::bail!(
            "What: cannot collect {pages_needed} KV page(s) covering {len} token(s) for sequence              {seq}; it holds only {}.              Why: the sequence was not mirrored to that length, so publishing it would claim KV              that was never written.              Fix: publish only the mirrored prefix, or raise the KV page budget so mirroring can              keep up.",
            pages.len()
        );
    }
    Ok(pages.iter().copied().take(pages_needed).collect())
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

// ---------------------------------------------------------------------------
// Connector KV payload <-> runner-KV conversion (DESIGN §38, K4)
// ---------------------------------------------------------------------------

/// One layer's exported past K/V as owned host floats plus their ORT shapes.
///
/// Produced from a `PastPresent` runner's [`crate::decode::DecodeState::export_runner_kv`]
/// so a chunk's token range can be sliced out into a portable [`KvPayload`].
pub(crate) struct ExportedLayerKv {
    pub(crate) key: Vec<f32>,
    pub(crate) key_shape: Vec<i64>,
    pub(crate) value: Vec<f32>,
    pub(crate) value_shape: Vec<i64>,
}

/// Collect exported runner KV into per-layer host buffers in `kv_model` layer
/// order, ready for chunk slicing via [`chunk_payload_from_exported`].
pub(crate) fn exported_layers_from_runner(
    kv_model: &KvModelInfo,
    exported: &[(String, onnx_genai_ort::Value)],
) -> anyhow::Result<Vec<ExportedLayerKv>> {
    let by_name = exported
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect::<HashMap<_, _>>();
    kv_model
        .layers
        .iter()
        .map(|layer| {
            let key_v = *by_name
                .get(layer.key_past.as_str())
                .with_context(|| format!("exported KV missing '{}'", layer.key_past))?;
            let value_v = *by_name
                .get(layer.value_past.as_str())
                .with_context(|| format!("exported KV missing '{}'", layer.value_past))?;
            Ok(ExportedLayerKv {
                key: key_v.to_vec_f32_lossy()?,
                key_shape: key_v.shape().to_vec(),
                value: value_v.to_vec_f32_lossy()?,
                value_shape: value_v.shape().to_vec(),
            })
        })
        .collect()
}

/// Slice the token range `[chunk_start, chunk_start + num_tokens)` out of the
/// exported per-layer KV into a portable [`KvPayload`].
///
/// The payload uses the head-major `[num_kv_heads, num_tokens, head_dim]` layout
/// documented on [`KvPayload`], read via the same [`extract_present_token`] axis
/// handling used to mirror present KV, so it matches the engine's KV contract
/// for arbitrary tensor axis orders.
pub(crate) fn chunk_payload_from_exported(
    layers: &[ExportedLayerKv],
    config: PageTensorConfig,
    chunk_start: usize,
    num_tokens: usize,
) -> anyhow::Result<KvPayload> {
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let per_layer = num_kv_heads * num_tokens * head_dim;
    let mut payload_layers = Vec::with_capacity(layers.len());
    for layer in layers {
        let mut key = vec![0.0_f32; per_layer];
        let mut value = vec![0.0_f32; per_layer];
        for t in 0..num_tokens {
            let abs = chunk_start + t;
            let key_tok = extract_present_token(&layer.key, &layer.key_shape, config, abs)?;
            let value_tok = extract_present_token(&layer.value, &layer.value_shape, config, abs)?;
            for head in 0..num_kv_heads {
                for dim in 0..head_dim {
                    let dst = (head * num_tokens + t) * head_dim + dim;
                    let src = head * head_dim + dim;
                    key[dst] = key_tok[src];
                    value[dst] = value_tok[src];
                }
            }
        }
        payload_layers.push(KvLayerPayload { key, value });
    }
    Ok(KvPayload {
        num_tokens,
        num_layers: layers.len(),
        num_kv_heads,
        head_dim,
        dtype: KvPayloadDtype::F32,
        layers: payload_layers,
    })
}

/// A fetched chunk's payload positioned at a token offset relative to the start
/// of the contiguous fetched region (`relative_start == 0` for the first chunk).
pub(crate) struct PlacedPayload<'a> {
    pub(crate) relative_start: usize,
    pub(crate) payload: &'a KvPayload,
}

/// Assemble contiguous fetched chunk payloads into full-length
/// `(past_key_values.* name, Value)` past tensors covering `[0, total_len)`,
/// ready for [`crate::decode::DecodeState::import_runner_kv`].
///
/// Each per-layer tensor is built in `[num_kv_heads, total_len, head_dim]`
/// head-major order and shaped with [`past_shape`], matching the past-tensor
/// contract [`load_materialized_past`] uses, so the injected KV is consumed by
/// the model exactly as if it had been recomputed.
pub(crate) fn past_kv_from_payloads(
    session: &Session,
    kv_model: &KvModelInfo,
    placed: &[PlacedPayload<'_>],
    total_len: usize,
) -> anyhow::Result<Vec<(String, Value)>> {
    let config = kv_model.tensor_config;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let input_shapes = session
        .inputs()
        .iter()
        .map(|info| (info.name.as_str(), info.shape.as_slice()))
        .collect::<HashMap<_, _>>();

    let mut out = Vec::with_capacity(kv_model.layers.len() * 2);
    for (idx, layer) in kv_model.layers.iter().enumerate() {
        let key_shape = past_shape(
            input_shapes
                .get(layer.key_past.as_str())
                .copied()
                .context("missing key past input shape")?,
            total_len,
        )?;
        let value_shape = past_shape(
            input_shapes
                .get(layer.value_past.as_str())
                .copied()
                .context("missing value past input shape")?,
            total_len,
        )?;
        let full = num_kv_heads * total_len * head_dim;
        let mut key = vec![0.0_f32; full];
        let mut value = vec![0.0_f32; full];
        for placed in placed {
            let layer_payload = placed
                .payload
                .layers
                .get(idx)
                .context("fetched payload missing a layer")?;
            let num_tokens = placed.payload.num_tokens;
            for head in 0..num_kv_heads {
                for t in 0..num_tokens {
                    for dim in 0..head_dim {
                        let dst = (head * total_len + placed.relative_start + t) * head_dim + dim;
                        let src = (head * num_tokens + t) * head_dim + dim;
                        key[dst] = layer_payload.key[src];
                        value[dst] = layer_payload.value[src];
                    }
                }
            }
        }
        out.push((
            layer.key_past.clone(),
            Value::from_vec_f32(key, &key_shape)?,
        ));
        out.push((
            layer.value_past.clone(),
            Value::from_vec_f32(value, &value_shape)?,
        ));
    }
    Ok(out)
}

/// Whether `kv_model`'s past KV inputs are plain f32 in the ORT graph — the
/// only dtype the connector payload round-trips today. Non-f32 KV (fp16/fp8/
/// int8) is skipped so a dtype mismatch can never corrupt injected output. The
/// check uses the model's actual past-input dtype (not the coarser
/// [`KvDType`], which folds fp16 into `F32`).
/// Whether the ORT decoder `session` declares loop-carried recurrent state
/// (hybrid conv/recurrent `fixed_state` alongside attention KV).
///
/// This mirrors the native [`has_recurrent_state`] gate (#700) onto the ORT
/// paged-KV reuse path (#701). Paged / materialized-past reuse
/// ([`load_materialized_past`]) restores only attention KV, so a hybrid
/// recurrent model reused this way would run a fresh-zero recurrent/conv state
/// against a reused attention prefix and silently emit wrong continuation
/// logits (the #695 failure #700 fixed for the native backend). When this
/// returns `true` the reuse decision must decline paged reuse and force a full
/// recompute — correct, if slower — until per-prefix recurrent-state restore
/// lands.
///
/// The signal is purely structural (RULES.md §2), never a model-name gate: a
/// state counts as recurrent only when the model's I/O spec declares it as a
/// loop-carried `state_pair` input AND that input's shape carries a static
/// penultimate (feature) axis, exactly as the native path classifies declared
/// state inputs. Attention KV past inputs carry a symbolic `past_sequence`
/// length on that axis and are never declared as state pairs, so this is a
/// guaranteed no-op for every attention-only model loadable today.
///
/// [`has_recurrent_state`]: crate::native_decode
pub(crate) fn ort_session_has_recurrent_state(
    session: &Session,
    io: Option<&onnx_genai_metadata::DecoderAbi>,
) -> bool {
    let Some(state_pairs) = io.and_then(|io| io.state_pairs.as_ref()) else {
        return false;
    };
    let declared = state_pairs
        .iter()
        .map(|pair| pair.input.as_str())
        .collect::<std::collections::HashSet<_>>();
    session
        .inputs()
        .iter()
        .filter(|info| declared.contains(info.name.as_str()))
        .any(|info| is_recurrent_state_input_shape(&info.shape))
}

/// Structural test matching `native_decode::is_recurrent_state_shape` for the
/// ORT session's concrete `i64` shapes: a declared loop-carried state input is
/// a fixed-size recurrent state (replaced wholesale each step) rather than a
/// growable KV cache when its sequence axis — the penultimate axis — is
/// statically sized (`>= 0`) rather than symbolic (`< 0`).
fn is_recurrent_state_input_shape(shape: &[i64]) -> bool {
    shape.len() >= 2 && shape[shape.len() - 2] >= 0
}

pub(crate) fn kv_model_past_is_f32(session: &Session, kv_model: &KvModelInfo) -> bool {
    let Some(layer) = kv_model.layers.first() else {
        return false;
    };
    session
        .inputs()
        .iter()
        .find(|info| info.name == layer.key_past)
        .map(|info| info.dtype == DataType::Float32)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{
        DecodeState, ModelDecodePath, SharedKvOffer, detect_model_decode_path,
        extract_next_token_logits_from_outputs, run_decode_session_logits, run_decode_step,
    };
    use onnx_genai_kv::{KvCacheOps, MaterializedLayerKv};
    use onnx_genai_ort::{Environment, SessionOptions};
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

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
            .join("model.onnx.textproto")
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

    fn fixture_io(name: &str) -> anyhow::Result<onnx_genai_metadata::DecoderAbi> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name)
            .join("inference_metadata.yaml");
        let metadata = onnx_genai_metadata::load_metadata(&path)?;
        metadata
            .decoder_io()
            .cloned()
            .with_context(|| format!("fixture '{name}' must declare a decode ABI"))
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

    fn load_named_session(dir: &str, model_file: &str) -> anyhow::Result<(Environment, Session)> {
        let environment = Environment::new("kv-bridge-tests")?;
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(dir)
            .join(model_file);
        let session = Session::new(
            &environment,
            &path,
            SessionOptions::default().with_intra_op_threads(1),
        )?;
        Ok((environment, session))
    }

    #[test]
    fn ort_recurrent_state_guard_is_noop_for_attention_only_model() -> anyhow::Result<()> {
        let _guard = model_test_lock();
        // An attention-only KV model declares no loop-carried `state_pairs`, so
        // the #701 ORT paged-reuse guard must not trip: reuse stays enabled.
        let (_environment, session) = load_session("tiny-llm")?;
        let io = fixture_io("tiny-llm")?;
        assert!(
            !ort_session_has_recurrent_state(&session, Some(&io)),
            "attention-only model must not be treated as recurrent (guard must be a no-op)"
        );
        // Absent I/O metadata is also treated as non-recurrent.
        assert!(!ort_session_has_recurrent_state(&session, None));
        Ok(())
    }

    #[test]
    fn ort_recurrent_state_guard_trips_for_hybrid_recurrent_model() -> anyhow::Result<()> {
        let _guard = model_test_lock();
        // A hybrid decoder declares loop-carried `state_a.in` / `state_b.in`
        // recurrent states (static penultimate axis) alongside its attention KV
        // past inputs. The guard must trip so paged/materialized-past reuse is
        // declined and a full recompute is forced (#701), mirroring the native
        // `has_recurrent_state` gate (#700).
        let (_environment, session) =
            load_named_session("tiny-multiaxis-state-decoder", "decoder.onnx.textproto")?;
        let io: onnx_genai_metadata::DecoderAbi = serde_json::from_value(serde_json::json!({
            "kv_inputs": ["past.3.key", "past.3.value", "past.11.key", "past.11.value"],
            "state_pairs": [
                {"input": "state_a.in", "output": "state_a.out"},
                {"input": "state_b.in", "output": "state_b.out"},
            ],
        }))?;
        assert!(
            ort_session_has_recurrent_state(&session, Some(&io)),
            "hybrid recurrent model must be gated out of ORT paged-KV reuse"
        );
        Ok(())
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

    #[test]
    fn speculative_target_rejection_allows_runner_backed_rewind_validation() -> anyhow::Result<()> {
        let mut cache = PagedKvCache::new(4, 8);
        let seq = cache.create_sequence();
        cache.append(seq, 4)?;
        let state = EngineSession {
            tokens: vec![10, 11, 12, 13],
            kv_token_count: 4,
            decode_state: DecodeState::for_test_runner_backed(),
            draft: None,
            sampled_fastpath_failed: false,
        };

        validate_target_state_rewind_to_len(
            None,
            &cache,
            seq,
            &state,
            RewindRequest::new(2, RewindRunnerPolicy::AllowRunnerRewind),
        )?;
        let error = validate_target_state_rewind_to_len(
            None,
            &cache,
            seq,
            &state,
            RewindRequest::new(2, RewindRunnerPolicy::RejectRunnerRewind),
        )
        .expect_err("public rewind policy must still reject runner-backed state");
        assert!(
            error.to_string().contains("runner-backed decoder state"),
            "unexpected error: {error:#}"
        );
        Ok(())
    }

    #[test]
    fn speculative_draft_realignment_allows_runner_backed_rewind_validation() -> anyhow::Result<()>
    {
        let mut cache = PagedKvCache::new(4, 8);
        let seq = cache.create_sequence();
        cache.append(seq, 5)?;
        let decode_state = DecodeState::for_test_runner_backed();

        validate_decode_state_rewind_to_len(
            None,
            &cache,
            seq,
            &decode_state,
            5,
            RewindRequest::new(3, RewindRunnerPolicy::AllowRunnerRewind),
        )
    }

    fn infers_past_present_model_and_rejects_static_cache_as_kv_bridge_model() -> anyhow::Result<()>
    {
        let _guard = model_test_lock();
        let (_environment, session) = load_session("tiny-llm")?;
        let info = infer_kv_model_info(&session, Some(&fixture_io("tiny-llm")?), 4, KvDType::F32)?
            .expect("past/present KV model");
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
        // Per-layer geometry for a uniform model is a single-entry vector equal
        // to the representative config's num_kv_heads/head_dim.
        assert_eq!(
            info.layer_configs,
            vec![onnx_genai_kv::LayerTensorConfig::new(2, 8)]
        );

        let (_environment, static_session) = load_session("tiny-llm-scatter")?;
        assert!(
            infer_kv_model_info(
                &static_session,
                Some(&fixture_io("tiny-llm-scatter")?),
                4,
                KvDType::F32
            )?
            .is_none()
        );

        // Without any I/O metadata, the paged resolver never guesses geometry
        // from tensor names or shapes: it builds no paged cache even for a graph
        // that clearly carries KV-shaped state inputs (a static-cache scatter
        // model, or a past/present model whose metadata was simply not loaded).
        // KV correctness is enforced separately by the decode-path resolver.
        assert!(
            infer_kv_model_info(&session, None, 4, KvDType::F32)?.is_none(),
            "no metadata must yield no paged KV, never a name/shape guess"
        );
        assert!(infer_kv_model_info(&static_session, None, 4, KvDType::F32)?.is_none());

        // Declaring only one side of the positional pairing is an actionable
        // error that names both required keys, never a silent half-guess.
        let mut half_declared = fixture_io("tiny-llm")?;
        half_declared.kv_outputs = None;
        let mismatch = infer_kv_model_info(&session, Some(&half_declared), 4, KvDType::F32)
            .expect_err("declaring kv_inputs without kv_outputs must fail");
        assert!(
            mismatch.to_string().contains("kv_inputs")
                && mismatch.to_string().contains("kv_outputs"),
            "the error must name both explicit keys: {mismatch}"
        );
        Ok(())
    }

    /// Per-layer geometry construction for a heterogeneous model, plus the
    /// `num_kv_shared_layers` fold. A model with `num_hidden_layers = 4` and
    /// `num_kv_shared_layers = 1` exports only `4 - 1 = 3` KV entries; the last
    /// full-attention layer here uses a larger head_dim (16) than the sliding
    /// layers (8), mirroring Gemma-4 E2B's 256/512 split at tiny scale.
    ///
    /// The resulting `layer_configs` therefore has exactly 3 entries (one per
    /// exported KV output), and metadata `shared_kv.target_layers` indices map
    /// directly onto these positions — index 2 selects the full layer's (3, 16)
    /// geometry, never the sliding layers' (2, 8). An off-by-one in this mapping
    /// would silently read the wrong layer's geometry.
    #[test]
    fn layer_configs_are_built_per_exported_kv_layer_with_shared_layer_fold() {
        // 3 EXPORTED present-key outputs = num_hidden_layers(4) - num_kv_shared_layers(1).
        let key_outputs = vec![
            TensorInfo {
                name: "present.0.key".into(),
                dtype: DataType::Float32,
                shape: vec![-1, 2, -1, 8], // sliding: num_kv_heads=2, head_dim=8
            },
            TensorInfo {
                name: "present.1.key".into(),
                dtype: DataType::Float32,
                shape: vec![-1, 2, -1, 8], // sliding: num_kv_heads=2, head_dim=8
            },
            TensorInfo {
                name: "present.2.key".into(),
                dtype: DataType::Float16,
                shape: vec![-1, 3, -1, 16], // full: num_kv_heads=3, head_dim=16
            },
        ];

        let configs = layer_configs_from_key_outputs(&key_outputs).unwrap();
        assert_eq!(
            configs,
            vec![
                onnx_genai_kv::LayerTensorConfig::new(2, 8),
                onnx_genai_kv::LayerTensorConfig::new(2, 8),
                onnx_genai_kv::LayerTensorConfig::new(3, 16),
            ],
            "one LayerTensorConfig per EXPORTED KV layer, with per-layer head_dim"
        );
        // Exported-entry count equals the folded layer count, not num_hidden_layers.
        assert_eq!(configs.len(), 3);

        // shared_kv.target_layers index the EXPORTED entries directly: the
        // full-attention group's last target layer (2) must pick (3, 16).
        let full_group_target: usize = *[0usize, 1, 2].last().unwrap();
        assert_eq!(configs[full_group_target].num_kv_heads, 3);
        assert_eq!(configs[full_group_target].head_dim, 16);
        // A sliding group targeting layer 0 must pick (2, 8), not the full geometry.
        assert_eq!(configs[0].head_dim, 8);
    }

    #[test]
    fn native_kv_specs_retain_asymmetric_past_tensor_geometry_and_dtype() -> anyhow::Result<()> {
        let graph = onnx_genai_ort::GraphIoMetadata::new(
            vec![
                TensorInfo {
                    name: "past_key_values.0.key".into(),
                    dtype: DataType::Float16,
                    shape: vec![-1, 2, -1, 4],
                },
                TensorInfo {
                    name: "past_key_values.0.value".into(),
                    dtype: DataType::Float32,
                    shape: vec![-1, 2, -1, 8],
                },
            ],
            vec![
                TensorInfo {
                    name: "present.0.key".into(),
                    dtype: DataType::Float16,
                    shape: vec![-1, 2, -1, 4],
                },
                TensorInfo {
                    name: "present.0.value".into(),
                    dtype: DataType::Float32,
                    shape: vec![-1, 2, -1, 8],
                },
            ],
        );
        let io = fixture_io("tiny-llm")?;

        let info = infer_kv_model_info(&graph, Some(&io), 16, KvDType::F32)?
            .context("declared KV pairs should produce model info")?;

        assert_eq!(info.native_kv_tensors.len(), 2);
        assert_eq!(
            crate::kv_sizing::kv_cache_bytes_for_tensors(&info.native_kv_tensors, 1)?,
            80,
            "f16 key (16 B/token) plus wider f32 value (64 B/token)"
        );
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
        assert_eq!(past_shape(&[-1, 2, -1, 8], 3).unwrap(), [1, 2, 3, 8]);
        assert_eq!(row_major_strides(&[1, 2, 3, 4]), [24, 12, 4, 1]);

        // fp16 present KV is accepted for on-device / share-buffer decode runners.
        let fp16 = TensorInfo {
            dtype: DataType::Float16,
            ..valid.clone()
        };
        assert_eq!(infer_kv_heads_and_head_dim(&fp16).unwrap(), (2, 8));

        let wrong_dtype = TensorInfo {
            dtype: DataType::Int64,
            ..valid.clone()
        };
        assert!(
            infer_kv_heads_and_head_dim(&wrong_dtype)
                .unwrap_err()
                .to_string()
                .contains("must be Float32, Float16, or BFloat16 rank >= 3")
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

    /// Helper: build owned `String` port lists from string literals.
    fn ports(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    #[test]
    fn pair_kv_ports_groups_interleaved_key_value_by_declared_order() -> anyhow::Result<()> {
        // Deliberately NON-conventional names prove the pairing is positional,
        // never derived from a substring like "key"/"value" or a layer index.
        let layers = pair_kv_ports(
            &ports(&["in_a", "in_b", "in_c", "in_d"]),
            &ports(&["out_a", "out_b", "out_c", "out_d"]),
        )?;
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].key_past, "in_a");
        assert_eq!(layers[0].key_present, "out_a");
        assert_eq!(layers[0].value_past, "in_b");
        assert_eq!(layers[0].value_present, "out_b");
        assert_eq!(layers[1].key_past, "in_c");
        assert_eq!(layers[1].value_present, "out_d");
        Ok(())
    }

    #[test]
    fn pair_kv_ports_accepts_empty_lists_as_no_layers() -> anyhow::Result<()> {
        assert!(pair_kv_ports(&ports(&[]), &ports(&[]))?.is_empty());
        Ok(())
    }

    #[test]
    fn pair_kv_ports_rejects_unequal_lengths_naming_both_keys() {
        let error = pair_kv_ports(&ports(&["a", "b"]), &ports(&["x"]))
            .expect_err("unequal kv_inputs/kv_outputs must fail");
        let message = error.to_string();
        assert!(message.contains("kv_inputs"), "{message}");
        assert!(message.contains("kv_outputs"), "{message}");
    }

    #[test]
    fn pair_kv_ports_rejects_odd_port_count_naming_the_key() {
        let error = pair_kv_ports(&ports(&["a", "b", "c"]), &ports(&["x", "y", "z"]))
            .expect_err("an odd KV port count cannot form [key, value] pairs");
        assert!(
            error.to_string().contains("kv_inputs/kv_outputs"),
            "the error must name the offending key: {error}"
        );
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
        let model = infer_kv_model_info(&session, Some(&fixture_io("tiny-llm")?), 2, KvDType::F32)?
            .unwrap();
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
        let model = infer_kv_model_info(&session, Some(&fixture_io("tiny-llm")?), 2, KvDType::F32)?
            .unwrap();
        let materialized = onnx_genai_kv::MaterializedKv {
            start_position: 0,
            sink_len: 0,
            sequence_len: 2,
            layers: vec![MaterializedLayerKv {
                key: (0..32).map(|value| value as f32).collect(),
                value: (100..132).map(|value| value as f32).collect(),
                num_kv_heads: 2,
                head_dim: 8,
            }],
        };
        let io = fixture_io("tiny-llm")?;
        let mut state = DecodeState::new_with_io(&session, Some(&io))?;
        state.set_past(
            HashMap::from([("stale".to_string(), Value::from_vec_f32(vec![0.0], &[1])?)]),
            1,
        );

        load_materialized_past(&session, &model, &mut state, &materialized)?;

        assert!(!state.past().contains_key("stale"));
        let key = &state.past()["past_key_values.0.key"];
        assert_eq!(key.shape(), &[1, 2, 2, 8]);
        assert_eq!(key.to_vec_f32()?, materialized.layers[0].key);
        assert_eq!(
            state.past()["past_key_values.0.value"].to_vec_f32()?,
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

    /// Asking for more tokens than a sequence holds must fail loudly.
    ///
    /// These pages are published under a key saying they cover that many
    /// tokens. Returning a short list instead would let a later request attach
    /// pages missing the KV they claim and decode from stale contents — wrong
    /// output with nothing in the logs. A sequence can be short whenever the
    /// page pool ran dry mid-generation and mirroring stopped early.
    #[test]
    fn refuses_to_collect_more_pages_than_a_sequence_holds() {
        let mut cache = PagedKvCache::new_with_tensor_config(tensor_config(), 8);
        let seq = cache.create_sequence();
        for base in [0.0, 10.0, 20.0] {
            append_token(&mut cache, seq, base);
        }

        // 3 tokens at 2 tokens/page is 2 pages; asking for 9 needs 5.
        let error = sequence_pages_for_len(&cache, seq, 9)
            .expect_err("collecting more pages than the sequence holds must fail")
            .to_string();
        assert!(
            error.contains("cannot collect 5 KV page(s) covering 9 token(s)"),
            "error should say how many pages were wanted: {error}"
        );
        assert!(
            error.contains("it holds only 2"),
            "error should say how many pages exist: {error}"
        );

        // The exactly-covering length still works.
        assert_eq!(sequence_pages_for_len(&cache, seq, 3).unwrap().len(), 2);
    }

    fn rewinds_materialized_ort_past_and_handles_edge_branches() -> anyhow::Result<()> {
        let _guard = model_test_lock();
        let (_environment, session) = load_session("tiny-llm")?;
        let model = infer_kv_model_info(&session, Some(&fixture_io("tiny-llm")?), 2, KvDType::F32)?
            .unwrap();
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
        let io = fixture_io("tiny-llm")?;
        let mut decode_state = DecodeState::new_with_io(&session, Some(&io))?;
        let mut count = 3;

        rewind_decode_state_to_len(
            &session,
            Some(&model),
            &mut cache,
            seq,
            &mut decode_state,
            &mut count,
            RewindRequest::new(2, RewindRunnerPolicy::AllowRunnerRewind),
        )?;
        assert_eq!(count, 2);
        assert_eq!(
            decode_state.past()["past_key_values.0.key"].shape(),
            &[1, 2, 2, 8]
        );

        rewind_decode_state_to_len(
            &session,
            Some(&model),
            &mut cache,
            seq,
            &mut decode_state,
            &mut count,
            RewindRequest::new(0, RewindRunnerPolicy::AllowRunnerRewind),
        )?;
        assert_eq!(count, 0);
        assert!(decode_state.past().is_empty());

        count = 1;
        let error = rewind_decode_state_to_len(
            &session,
            None,
            &mut cache,
            seq,
            &mut decode_state,
            &mut count,
            RewindRequest::new(0, RewindRunnerPolicy::AllowRunnerRewind),
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
            RewindRequest::new(5, RewindRunnerPolicy::AllowRunnerRewind),
        )?;
        assert_eq!(count, 0);
        Ok(())
    }

    fn rejects_static_and_past_present_decode_runner_rewind() -> anyhow::Result<()> {
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
                    sliding_window: None,
                    sink_tokens: None,
                },
            ),
        ] {
            let (_environment, session) = load_session(fixture_name)?;
            let session = Arc::new(session);
            let io = fixture_io(fixture_name)?;
            let mut state = DecodeState::new_for_path_with_io(&session, &path, Some(&io))?;
            let mut cache = PagedKvCache::new(2, 8);
            let seq = cache.create_sequence();
            run_decode_session_logits(&mut state, &[2, 4, 3], 0)?;
            cache.append(seq, 3)?;
            let mut count = 3;

            let error = rewind_decode_state_to_len(
                &session,
                None,
                &mut cache,
                seq,
                &mut state,
                &mut count,
                RewindRequest::new(1, RewindRunnerPolicy::RejectRunnerRewind),
            )
            .expect_err("runner-backed rewind must fail closed");

            assert!(
                error.to_string().contains("runner-backed decoder state"),
                "unexpected error: {error:#}"
            );
            assert_eq!(count, 3);
            assert_eq!(state.runner_len(), 3);
            assert_eq!(cache.len(seq)?, 3);
        }
        Ok(())
    }

    #[test]
    fn windowed_past_present_keeps_absolute_positions_with_bounded_past() -> anyhow::Result<()> {
        let _guard = model_test_lock();
        let (_environment, session) = load_session("tiny-llm")?;
        let session = Arc::new(session);
        let io = fixture_io("tiny-llm")?;
        let path = detect_model_decode_path(Some(&io), Some(2), 0, SharedKvOffer::default())?;
        assert!(matches!(
            path,
            ModelDecodePath::PastPresent {
                shared_buffer: false,
                max_len: None,
                sliding_window: Some(2),
                sink_tokens: None,
            }
        ));
        let mut state = DecodeState::new_for_path_with_io(&session, &path, Some(&io))?;

        run_decode_step(&session, &mut state, &[2, 4], 0)?;
        assert_eq!(state.retained_kv_len(2), 2);
        assert!(state.past().values().all(|value| value.shape()[2] == 2));

        run_decode_step(&session, &mut state, &[3], 2)?;
        assert_eq!(state.retained_kv_len(3), 2);
        assert!(state.past().values().all(|value| value.shape()[2] == 2));
        // Sliding-window attention always takes bounded functional
        // past/present dataflow; storage hints cannot override it.
        assert!(matches!(
            detect_model_decode_path(Some(&io), Some(2), 0, SharedKvOffer::default())?,
            ModelDecodePath::PastPresent {
                shared_buffer: false,
                max_len: None,
                sliding_window: Some(2),
                sink_tokens: None,
            }
        ));
        Ok(())
    }

    #[test]
    fn windowed_past_present_pins_attention_sink_rows() -> anyhow::Result<()> {
        let _guard = model_test_lock();
        let (_environment, session) = load_session("tiny-llm")?;
        let session = Arc::new(session);
        // window=2, sink=1 (StreamingLLM): retained buffer = 1 sink row + up to 2
        // window rows once the context exceeds sink + window.
        let path = ModelDecodePath::PastPresent {
            shared_buffer: false,
            max_len: None,
            sliding_window: Some(2),
            sink_tokens: Some(1),
        };
        let io = fixture_io("tiny-llm")?;
        let mut state = DecodeState::new_for_path_with_io(&session, &path, Some(&io))?;
        assert_eq!(state.sink_tokens(), 1);

        // present_len=2, window covers the whole buffer (window_start=0 <= sink).
        run_decode_step(&session, &mut state, &[2, 4], 0)?;
        assert_eq!(state.retained_kv_len(2), 2);
        assert!(state.past().values().all(|value| value.shape()[2] == 2));

        // present_len=3, window_start=1 == sink: still keep everything.
        run_decode_step(&session, &mut state, &[3], 2)?;
        assert_eq!(state.retained_kv_len(3), 3);
        assert!(state.past().values().all(|value| value.shape()[2] == 3));

        // present_len=4, window_start=2 > sink=1: pin sink row + trailing window.
        run_decode_step(&session, &mut state, &[5], 3)?;
        assert_eq!(state.retained_kv_len(4), 3);
        assert!(state.past().values().all(|value| value.shape()[2] == 3));
        Ok(())
    }

    /// Regression guard for metadata-driven graph-port binding: the
    /// `tiny-llm-explicit-io` fixture is the tiny-llm graph with every decode
    /// port renamed to a non-conventional name (`tokens`, `attn_mask_port`,
    /// `pos_port`, `out_logits`, `cache_k_in.0`/`cache_v_in.0` ->
    /// `cache_k_out.0`/`cache_v_out.0`). The runtime can only decode it by
    /// reading the explicit resolved decode ABI, proving it never infers a port by
    /// tensor name.
    #[test]
    fn explicit_io_binds_decode_ports_purely_from_metadata_names() -> anyhow::Result<()> {
        let _guard = model_test_lock();
        let (_environment, session) = load_session("tiny-llm-explicit-io")?;

        // Sanity: the graph exposes only the renamed, non-conventional names.
        assert!(session.input_names().iter().any(|name| name == "tokens"));
        assert!(
            session
                .output_names()
                .iter()
                .any(|name| name == "out_logits")
        );
        assert!(!session.input_names().iter().any(|name| name == "input_ids"));
        assert!(!session.output_names().iter().any(|name| name == "logits"));

        // Shape cannot distinguish the three integer sequence inputs, so
        // initialization fails closed and names are not consulted.
        let convention_err = match DecodeState::new_with_io(&session, None) {
            Ok(_) => panic!("ambiguous shapes must require metadata"),
            Err(error) => error,
        };
        assert!(
            convention_err
                .to_string()
                .contains("declare the port's role in"),
            "expected actionable metadata error, got: {convention_err}"
        );

        // Load the explicit `io` block from the fixture's inference metadata and
        // decode purely from the declared names.
        let metadata_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm-explicit-io/inference_metadata.yaml");
        let metadata = onnx_genai_metadata::load_metadata(&metadata_path)?;
        let io = metadata
            .decoder_io()
            .expect("fixture declares a decode ABI");

        let mut state = DecodeState::new_with_io(&session, Some(io))?;
        assert!(
            state.use_kv,
            "declared kv_inputs/kv_outputs enable KV cache"
        );

        // First step (prefill) over two tokens, then a single decode step that
        // must read the cached KV threaded via the declared kv pairs.
        let prefill = run_decode_step(&session, &mut state, &[2, 4], 0)?;
        let prefill_logits = extract_next_token_logits_from_outputs(
            &session,
            &prefill,
            io.logits_output.as_deref(),
        )?;
        assert_eq!(
            prefill_logits.len(),
            32,
            "logits bound from declared out_logits output must be vocab-sized"
        );

        let step = run_decode_step(&session, &mut state, &[3], 2)?;
        let step_logits =
            extract_next_token_logits_from_outputs(&session, &step, io.logits_output.as_deref())?;
        assert_eq!(step_logits.len(), 32);
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
            decode_state: {
                let io = fixture_io("tiny-llm-scatter")?;
                DecodeState::new_with_io(&session, Some(&io))?
            },
            draft: None,
            sampled_fastpath_failed: false,
        };

        trim_overmaterialized_target_kv(
            &session,
            None,
            &mut cache,
            seq,
            &mut state,
            RewindRunnerPolicy::AllowRunnerRewind,
        )?;
        assert_eq!(state.kv_token_count, 0);
        assert_eq!(cache.len(seq)?, 4);

        state.decode_state.use_kv = true;
        state.kv_token_count = 4;
        rewind_target_state_to_len(
            &session,
            None,
            &mut cache,
            seq,
            &mut state,
            RewindRequest::new(4, RewindRunnerPolicy::AllowRunnerRewind),
        )?;
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
        rejects_static_and_past_present_decode_runner_rewind()?;
        rewinds_target_state_and_trims_overmaterialized_kv()
    }

    /// Build a minimal TensorInfo set that looks like a past/present KV model and
    /// verify that `infer_kv_model_info` propagates the requested storage dtype
    /// directly into `PageTensorConfig.dtype` without any special-casing.
    ///
    /// These tests use the real tiny-llm fixture so that `Session::inputs/outputs`
    /// are fully populated; the dtype under test is a storage-layer knob that does
    /// not touch the model I/O schema.
    fn infer_kv_model_info_propagates_dtype_to_page_tensor_config(
        dtype: KvDType,
    ) -> anyhow::Result<()> {
        let _guard = model_test_lock();
        let (_environment, session) = load_session("tiny-llm")?;
        let info = infer_kv_model_info(&session, Some(&fixture_io("tiny-llm")?), 4, dtype)?
            .expect("tiny-llm exposes past/present KV outputs");
        assert_eq!(
            info.tensor_config.dtype, dtype,
            "PageTensorConfig.dtype must equal the requested storage dtype"
        );
        Ok(())
    }

    #[test]
    fn configured_dtype_f32_flows_into_page_tensor_config() -> anyhow::Result<()> {
        infer_kv_model_info_propagates_dtype_to_page_tensor_config(KvDType::F32)
    }

    #[test]
    fn configured_dtype_fp8_e4m3fn_flows_into_page_tensor_config() -> anyhow::Result<()> {
        infer_kv_model_info_propagates_dtype_to_page_tensor_config(KvDType::Fp8E4M3Fn)
    }

    #[test]
    fn configured_dtype_fp8_e5m2_flows_into_page_tensor_config() -> anyhow::Result<()> {
        infer_kv_model_info_propagates_dtype_to_page_tensor_config(KvDType::Fp8E5M2)
    }

    #[test]
    fn configured_dtype_int8_flows_into_page_tensor_config() -> anyhow::Result<()> {
        infer_kv_model_info_propagates_dtype_to_page_tensor_config(KvDType::Int8)
    }

    /// K4 multi-layer coverage: directly exercise [`chunk_payload_from_exported`]
    /// with **3 synthetic exported layers** (2 kv_heads, 4 head_dim, 8-token total
    /// exported window, chunk covering positions 3-6) and verify that every
    /// (layer, K/V slot, head, chunk-relative token, dim) cell is byte-identical
    /// to the input after extraction.
    ///
    /// Choosing `chunk_start = 3` ensures `token_pos ≥ 3` on every step, which
    /// prevents the benign batch-vs-sequence axis ambiguity in [`kv_tensor_axes`]
    /// for `token_pos == 0` and exercises the full code path cleanly.
    ///
    /// Pattern:
    ///   `key[l][h][t][d]  = 1000·l + 100·h + 10·t + d`  (absolute token `t`)
    ///   `val[l][h][t][d]  = -(1000·l + 100·h + 10·t + d)`
    ///
    /// Any layer swap, K/V swap, or [head, token, dim] transposition produces a
    /// value that differs from the expected → the mismatch is always detectable.
    #[test]
    fn chunk_payload_from_exported_multilayer_preserves_layer_head_token_dim_ordering() {
        const NUM_LAYERS: usize = 3;
        const NUM_KV_HEADS: usize = 2;
        const TOTAL_SEQ: usize = 8;
        const HEAD_DIM: usize = 4;
        const CHUNK_START: usize = 3;
        const NUM_TOKENS: usize = 4; // covers absolute positions 3,4,5,6

        let config = PageTensorConfig {
            num_layers: NUM_LAYERS,
            num_kv_heads: NUM_KV_HEADS,
            head_dim: HEAD_DIM,
            page_size: NUM_TOKENS,
            dtype: KvDType::F32,
        };

        // Shape [1, NUM_KV_HEADS, TOTAL_SEQ, HEAD_DIM] — standard past-KV form.
        let shape = vec![
            1_i64,
            NUM_KV_HEADS as i64,
            TOTAL_SEQ as i64,
            HEAD_DIM as i64,
        ];
        let exported: Vec<ExportedLayerKv> = (0..NUM_LAYERS)
            .map(|l| {
                // flat index: h*TOTAL_SEQ*HEAD_DIM + t*HEAD_DIM + d  (batch=0 → 0 offset)
                let size = NUM_KV_HEADS * TOTAL_SEQ * HEAD_DIM;
                let mut key = vec![0.0_f32; size];
                let mut value = vec![0.0_f32; size];
                for h in 0..NUM_KV_HEADS {
                    for t in 0..TOTAL_SEQ {
                        for d in 0..HEAD_DIM {
                            let flat = h * TOTAL_SEQ * HEAD_DIM + t * HEAD_DIM + d;
                            let sig = (1000 * l + 100 * h + 10 * t + d) as f32;
                            key[flat] = sig;
                            value[flat] = -sig;
                        }
                    }
                }
                ExportedLayerKv {
                    key,
                    key_shape: shape.clone(),
                    value,
                    value_shape: shape.clone(),
                }
            })
            .collect();

        let payload =
            chunk_payload_from_exported(&exported, config, CHUNK_START, NUM_TOKENS).unwrap();

        assert_eq!(payload.num_layers, NUM_LAYERS);
        assert_eq!(payload.num_kv_heads, NUM_KV_HEADS);
        assert_eq!(payload.num_tokens, NUM_TOKENS);
        assert_eq!(payload.head_dim, HEAD_DIM);
        assert!(payload.is_well_formed());

        for l in 0..NUM_LAYERS {
            for h in 0..NUM_KV_HEADS {
                for t in 0..NUM_TOKENS {
                    let abs_t = CHUNK_START + t; // absolute token index
                    for d in 0..HEAD_DIM {
                        let idx = (h * NUM_TOKENS + t) * HEAD_DIM + d;
                        let expected = (1000 * l + 100 * h + 10 * abs_t + d) as f32;
                        assert_eq!(
                            payload.layers[l].key[idx], expected,
                            "layer={l} head={h} token={t}(abs={abs_t}) dim={d}: key mismatch \
                             (got {}, expected {expected})",
                            payload.layers[l].key[idx]
                        );
                        assert_eq!(
                            payload.layers[l].value[idx], -expected,
                            "layer={l} head={h} token={t}(abs={abs_t}) dim={d}: value mismatch \
                             (got {}, expected {expected})",
                            payload.layers[l].value[idx]
                        );
                    }
                }
            }
        }
    }
}
