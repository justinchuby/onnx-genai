//! Faithful Rust port of `paged_attention_helper.h::CheckInputs` (ORT 1.29.0).
//!
//! Every rejection here returns [`PagedAttentionError::InvalidArgument`] and
//! corresponds one-to-one to an `ORT_MAKE_STATUS(..., INVALID_ARGUMENT, ...)` in
//! the helper. Line references in comments point at that file.

use crate::params::{
    PagedAttentionAttributes, PagedAttentionInputs, PagedAttentionParameters, Shape,
};
use crate::types::{KvCacheDtype, KvQuantType, PagedAttentionError};

type Res<T> = Result<T, PagedAttentionError>;

fn shape_size(shape: &Shape) -> i64 {
    shape.dims().iter().product()
}

/// Validate a `PagedAttention` node's attributes and input shapes exactly as
/// ORT 1.29.0 does, returning the derived [`PagedAttentionParameters`].
///
/// # Errors
/// Returns [`PagedAttentionError::InvalidArgument`] for any schema violation,
/// with a message mirroring the upstream helper.
#[allow(clippy::too_many_lines)]
pub fn check_inputs(
    attrs: &PagedAttentionAttributes,
    inputs: &PagedAttentionInputs,
) -> Res<PagedAttentionParameters> {
    let num_heads = attrs.num_heads;
    let kv_num_heads = attrs.kv_num_heads;
    if num_heads <= 0 {
        return Err(PagedAttentionError::invalid(
            "num_heads must be a positive integer.",
        ));
    }
    if kv_num_heads <= 0 {
        return Err(PagedAttentionError::invalid(
            "kv_num_heads must be a positive integer.",
        ));
    }

    let is_latent = attrs.kv_cache_layout.is_latent();
    let storage_dtype = inputs.key_cache_storage_dtype;
    let is_quantized_cache = storage_dtype.is_quantized_storage();

    // --- Q/K/V dispatch (helper.h Check_Q_K_Latent / Check_Q_K_V / Check_QKV) ---
    let (token_count, q_hidden_size, kv_hidden_size, head_size, is_packed_qkv) = if is_latent {
        check_q_k_latent(inputs, num_heads, kv_num_heads)?
    } else if inputs.key.is_some() {
        check_q_k_v(inputs, num_heads, kv_num_heads)?
    } else {
        check_qkv(inputs, num_heads, kv_num_heads)?
    };

    // --- v_head_size resolution (helper.h:453-466) ---
    let mut v_head_size = head_size;
    if attrs.v_head_size != 0 {
        if attrs.v_head_size < 1 || attrs.v_head_size > head_size {
            return Err(PagedAttentionError::invalid(format!(
                "'v_head_size' must be 0 (meaning head_size) or in [1, head_size] = [1, {head_size}], got {}",
                attrs.v_head_size
            )));
        }
        if attrs.v_head_size != head_size && !is_latent {
            return Err(PagedAttentionError::invalid(format!(
                "'v_head_size' ({}) may only differ from head_size ({head_size}) when 'kv_cache_layout' is 'LATENT'.",
                attrs.v_head_size
            )));
        }
        v_head_size = attrs.v_head_size;
    }

    // --- softmax-scale trap (helper.h:469-474) ---
    let has_explicit_scale = attrs.scale.is_some();
    if v_head_size != head_size && !has_explicit_scale {
        return Err(PagedAttentionError::invalid(format!(
            "An explicit 'scale' attribute is required when 'v_head_size' ({v_head_size}) differs from head_size ({head_size}): the default 1/sqrt(head_size) is not the intended scale for absorbed MLA."
        )));
    }

    // --- LATENT-specific constraints (helper.h:477-509) ---
    if is_latent {
        if inputs.value_cache.is_some() {
            return Err(PagedAttentionError::invalid(
                "Input 'value_cache' must be absent when 'kv_cache_layout' is 'LATENT': the value cache is the leading 'v_head_size' channels of 'key_cache'.",
            ));
        }
        if kv_num_heads != 1 {
            return Err(PagedAttentionError::invalid(format!(
                "'kv_num_heads' must be 1 when 'kv_cache_layout' is 'LATENT', got {kv_num_heads}"
            )));
        }
        if inputs.head_sink.is_some() {
            return Err(PagedAttentionError::invalid(
                "Input 'head_sink' is not supported when 'kv_cache_layout' is 'LATENT'.",
            ));
        }
        if inputs.q_norm_weight.is_some() || inputs.k_norm_weight.is_some() {
            return Err(PagedAttentionError::invalid(
                "Inputs 'q_norm_weight' / 'k_norm_weight' are not supported when 'kv_cache_layout' is 'LATENT': DeepSeek normalizes the latent projections in the graph, before absorption.",
            ));
        }
        if inputs.v_scale.is_some()
            || attrs.v_quant_type != KvQuantType::None
            || attrs.v_cache_dtype != KvCacheDtype::Default
        {
            return Err(PagedAttentionError::invalid(
                "Input 'v_scale' and attributes 'v_quant_type' / 'v_cache_dtype' must be unset when 'kv_cache_layout' is 'LATENT': the value elements are the key elements, so 'k_scale' and 'k_cache_dtype' describe both.",
            ));
        }
    } else if inputs.value_cache.is_none() {
        return Err(PagedAttentionError::invalid(
            "Input 'value_cache' is required unless 'kv_cache_layout' is 'LATENT'.",
        ));
    }

    // --- KV cache (helper.h CheckKVCache) ---
    let (num_blocks, block_size) = check_kv_cache(inputs, kv_num_heads, head_size, is_latent)?;

    // --- sequence-length tensors (helper.h CheckSequenceLengthTensors) ---
    let batch_size = check_sequence_length_tensors(inputs)?;

    // --- block table (helper.h CheckBlockTable) ---
    let max_num_blocks_per_seq = check_block_table(&inputs.block_table, batch_size)?;

    // --- slot mapping (helper.h CheckSlotMapping) ---
    if let Some(slot_mapping) = &inputs.slot_mapping {
        check_slot_mapping(slot_mapping, token_count)?;
    }

    // --- head sink & QK-Norm (helper.h CheckHeadSink / CheckQKNormWeights) ---
    if let Some(head_sink) = &inputs.head_sink {
        check_head_sink(head_sink, num_heads)?;
    }
    check_qk_norm_weights(
        inputs.q_norm_weight.as_ref(),
        inputs.k_norm_weight.as_ref(),
        head_size,
    )?;
    if inputs.q_norm_weight.is_some() && attrs.qk_norm_epsilon <= 0.0 {
        return Err(PagedAttentionError::invalid(format!(
            "qk_norm_epsilon must be positive, got {}",
            attrs.qk_norm_epsilon
        )));
    }

    // --- rotary cache (helper.h:540-560) ---
    let rotary_dim = match (&inputs.cos_cache, &inputs.sin_cache) {
        (Some(_), Some(_)) => inputs.rotary_dim,
        (None, None) => 0,
        _ => {
            return Err(PagedAttentionError::invalid(
                "Input 'cos_cache' and 'sin_cache' shall be both present or both absent.",
            ));
        }
    };
    if attrs.rotary_offset < 0 || attrs.rotary_offset % 8 != 0 {
        return Err(PagedAttentionError::invalid(format!(
            "'rotary_offset' must be non-negative and a multiple of 8, got {}",
            attrs.rotary_offset
        )));
    }
    if attrs.rotary_offset + rotary_dim > head_size {
        return Err(PagedAttentionError::invalid(format!(
            "'rotary_offset' + rotary_dim must not exceed head_size. Got {} + {rotary_dim} > {head_size}",
            attrs.rotary_offset
        )));
    }
    // The CUDA kernel (paged_attention.cc) requires the rotary caches when
    // do_rotary=1; without them it would silently skip RoPE and miscompute.
    if attrs.do_rotary && rotary_dim == 0 {
        return Err(PagedAttentionError::invalid(
            "do_rotary=1 requires both 'cos_cache' and 'sin_cache'.",
        ));
    }

    // --- quantized KV cache (helper.h CheckKVCacheQuantization) ---
    check_kv_cache_quantization(
        inputs.k_scale.as_ref(),
        "k_scale",
        "k_quant_type",
        attrs.k_quant_type,
        is_quantized_cache,
        kv_num_heads,
        head_size,
    )?;
    if !is_latent {
        check_kv_cache_quantization(
            inputs.v_scale.as_ref(),
            "v_scale",
            "v_quant_type",
            attrs.v_quant_type,
            is_quantized_cache,
            kv_num_heads,
            v_head_size,
        )?;
        // The kernel is instantiated for one T_CACHE, so both caches share it.
        if inputs.value_cache_storage_dtype != storage_dtype {
            return Err(PagedAttentionError::invalid(
                "'key_cache' and 'value_cache' must have the same element type.",
            ));
        }
    }
    check_kv_cache_dtype(attrs.k_cache_dtype, storage_dtype, "k_cache_dtype")?;
    check_kv_cache_dtype(attrs.v_cache_dtype, storage_dtype, "v_cache_dtype")?;

    // --- attention_metadata (helper.h:565-583) ---
    if let Some(meta) = &inputs.attention_metadata
        && (meta.rank() != 1 || meta.dims()[0] != 2)
    {
        return Err(PagedAttentionError::invalid(format!(
            "Input 'attention_metadata' must have shape (2), got {}",
            meta.to_string_compact()
        )));
    }

    let scale = attrs
        .scale
        .unwrap_or_else(|| 1.0f32 / (head_size as f32).sqrt());

    Ok(PagedAttentionParameters {
        batch_size,
        token_count,
        num_heads,
        kv_num_heads,
        head_size,
        v_head_size,
        hidden_size: q_hidden_size,
        v_hidden_size: num_heads * v_head_size,
        kv_hidden_size,
        is_latent_kv: is_latent,
        is_packed_qkv,
        rotary_offset: attrs.rotary_offset,
        rotary_dim,
        block_size,
        num_blocks,
        max_num_blocks_per_seq,
        scale,
        softcap: attrs.softcap,
        local_window_size: attrs.local_window_size,
        do_rotary: attrs.do_rotary,
        rotary_interleaved: attrs.rotary_interleaved,
        use_head_sink: inputs.head_sink.is_some(),
        use_qk_norm: inputs.q_norm_weight.is_some(),
        qk_norm_epsilon: attrs.qk_norm_epsilon,
        k_quant_type: attrs.k_quant_type,
        v_quant_type: attrs.v_quant_type,
    })
}

/// Returns `(token_count, q_hidden, kv_hidden, head_size, is_packed_qkv=false)`.
fn check_q_k_v(
    inputs: &PagedAttentionInputs,
    num_heads: i64,
    kv_num_heads: i64,
) -> Res<(i64, i64, i64, i64, bool)> {
    require_rank(&inputs.query, 2, "query")?;
    let token_count = inputs.query.dims()[0];
    let q_hidden = inputs.query.dims()[1];
    let head_size = q_hidden / num_heads;
    if head_size % 8 != 0 {
        return Err(PagedAttentionError::invalid(format!(
            "head_size must be a multiple of 8. Got head_size % 8 == {}",
            head_size % 8
        )));
    }
    let key = inputs.key.as_ref().ok_or_else(|| {
        PagedAttentionError::invalid(
            "Input 'key' and 'value' shall be both present, or both absent in the case of packed qkv.",
        )
    })?;
    let value = inputs.value.as_ref().ok_or_else(|| {
        PagedAttentionError::invalid(
            "Input 'key' and 'value' shall be both present, or both absent in the case of packed qkv.",
        )
    })?;
    require_rank_shape(key, 2, "key")?;
    if token_count != key.dims()[0] {
        return Err(PagedAttentionError::invalid(
            "Input 'query' and 'key' shall have same dim 0 (token count)",
        ));
    }
    let kv_hidden = key.dims()[1];
    if kv_hidden % kv_num_heads != 0 {
        return Err(PagedAttentionError::invalid(format!(
            "kv_hidden_size must be a multiple of kv_num_heads. Got kv_hidden_size % kv_num_heads == {}",
            kv_hidden % kv_num_heads
        )));
    }
    if kv_hidden / kv_num_heads != head_size {
        return Err(PagedAttentionError::invalid(format!(
            "kv_hidden_size / kv_num_heads must be equal to head_size. Got kv_hidden_size / kv_num_heads == {}",
            kv_hidden / kv_num_heads
        )));
    }
    require_rank_shape(value, 2, "value")?;
    if token_count != value.dims()[0] {
        return Err(PagedAttentionError::invalid(
            "Input 'query' and 'value' shall have same dim 0 (token count)",
        ));
    }
    if value.dims()[1] != kv_hidden {
        return Err(PagedAttentionError::invalid(
            "Input 'value' is expected to have same hidden size as key.",
        ));
    }
    Ok((token_count, q_hidden, kv_hidden, head_size, false))
}

/// LATENT mode (helper.h Check_Q_K_Latent).
fn check_q_k_latent(
    inputs: &PagedAttentionInputs,
    num_heads: i64,
    kv_num_heads: i64,
) -> Res<(i64, i64, i64, i64, bool)> {
    let key = inputs.key.as_ref().ok_or_else(|| {
        PagedAttentionError::invalid("Input 'key' is required when 'kv_cache_layout' is 'LATENT'.")
    })?;
    if inputs.value.is_some() {
        return Err(PagedAttentionError::invalid(
            "Input 'value' must be absent when 'kv_cache_layout' is 'LATENT': the value of every head is the leading 'v_head_size' channels of the latent key.",
        ));
    }
    require_rank(&inputs.query, 2, "query")?;
    let token_count = inputs.query.dims()[0];
    let q_hidden = inputs.query.dims()[1];
    if q_hidden % num_heads != 0 {
        return Err(PagedAttentionError::invalid(format!(
            "Input 'query' hidden size must be a multiple of num_heads. Got {q_hidden} % {num_heads} == {}",
            q_hidden % num_heads
        )));
    }
    let head_size = q_hidden / num_heads;
    if head_size % 8 != 0 {
        return Err(PagedAttentionError::invalid(format!(
            "head_size must be a multiple of 8. Got head_size % 8 == {}",
            head_size % 8
        )));
    }
    require_rank_shape(key, 2, "key")?;
    if token_count != key.dims()[0] {
        return Err(PagedAttentionError::invalid(
            "Input 'query' and 'key' shall have same dim 0 (token count)",
        ));
    }
    let kv_hidden = key.dims()[1];
    if kv_hidden != kv_num_heads * head_size {
        return Err(PagedAttentionError::invalid(format!(
            "Input 'key' is expected to have hidden size kv_num_heads * head_size = {} in 'LATENT' mode, got {kv_hidden}",
            kv_num_heads * head_size
        )));
    }
    Ok((token_count, q_hidden, kv_hidden, head_size, false))
}

/// Packed-QKV mode (helper.h Check_QKV).
fn check_qkv(
    inputs: &PagedAttentionInputs,
    num_heads: i64,
    kv_num_heads: i64,
) -> Res<(i64, i64, i64, i64, bool)> {
    require_rank(&inputs.query, 2, "query")?;
    let token_count = inputs.query.dims()[0];
    let packed_hidden = inputs.query.dims()[1];
    let head_size = packed_hidden / (num_heads + 2 * kv_num_heads);
    if head_size % 8 != 0 {
        return Err(PagedAttentionError::invalid(format!(
            "head_size must be a multiple of 8. Got head_size % 8 == {}",
            head_size % 8
        )));
    }
    if inputs.value.is_some() {
        return Err(PagedAttentionError::invalid(
            "Input 'key' and 'value' shall be both present, or both absent in the case of packed qkv.",
        ));
    }
    let q_hidden = head_size * num_heads;
    let kv_hidden = head_size * kv_num_heads;
    Ok((token_count, q_hidden, kv_hidden, head_size, true))
}

/// helper.h CheckKVCache — returns `(num_blocks, block_size)`.
fn check_kv_cache(
    inputs: &PagedAttentionInputs,
    kv_num_heads: i64,
    head_size: i64,
    is_latent: bool,
) -> Res<(i64, i64)> {
    let kc = &inputs.key_cache;
    if kc.rank() != 4 {
        return Err(PagedAttentionError::invalid(format!(
            "Input 'key_cache' is expected to have 4 dimensions, got {}",
            kc.rank()
        )));
    }
    let num_blocks = kc.dims()[0];
    let block_size = kc.dims()[1];
    if block_size < 16 || (block_size & (block_size - 1)) != 0 {
        return Err(PagedAttentionError::invalid(format!(
            "block_size must be a power of two and at least 16. Got block_size == {block_size}"
        )));
    }
    if kc.dims()[2] != kv_num_heads {
        return Err(PagedAttentionError::invalid(format!(
            "Input 'key_cache' shall have kv_num_heads, got {}",
            kc.dims()[2]
        )));
    }
    if kc.dims()[3] != head_size {
        return Err(PagedAttentionError::invalid(format!(
            "Input 'key_cache' dimension 3 should be same as head_size, got {}",
            kc.dims()[3]
        )));
    }
    if is_latent {
        return Ok((num_blocks, block_size));
    }
    let vc = inputs
        .value_cache
        .as_ref()
        .expect("value_cache presence checked before CheckKVCache");
    if vc.rank() != 4 {
        return Err(PagedAttentionError::invalid(format!(
            "Input 'value_cache' is expected to have 4 dimensions, got {}",
            vc.rank()
        )));
    }
    if vc.dims()[0] != num_blocks {
        return Err(PagedAttentionError::invalid(format!(
            "Input 'value_cache' dimension 0 should be num_blocks, got {}",
            vc.dims()[0]
        )));
    }
    if vc.dims()[1] != block_size {
        return Err(PagedAttentionError::invalid(format!(
            "Input 'value_cache' dimension 1 should be block_size, got {}",
            vc.dims()[1]
        )));
    }
    if vc.dims()[2] != kv_num_heads {
        return Err(PagedAttentionError::invalid(format!(
            "Input 'value_cache' shall have kv_num_heads, got {}",
            vc.dims()[2]
        )));
    }
    if vc.dims()[3] != head_size {
        return Err(PagedAttentionError::invalid(format!(
            "Input 'value_cache' dimension 3 should be same as head_size, got {}",
            vc.dims()[3]
        )));
    }
    Ok((num_blocks, block_size))
}

/// helper.h CheckSequenceLengthTensors — returns `batch_size`.
fn check_sequence_length_tensors(inputs: &PagedAttentionInputs) -> Res<i64> {
    let cs = &inputs.cumulative_sequence_length;
    if cs.rank() != 1 || cs.dims()[0] < 2 {
        return Err(PagedAttentionError::invalid(
            "cumulative_sequence_length must be shape (batch_size + 1).",
        ));
    }
    let batch_size = cs.dims()[0] - 1;
    let sl = &inputs.past_seqlens;
    if sl.rank() != 1 || sl.dims()[0] != batch_size {
        return Err(PagedAttentionError::invalid(
            "seqlens must be shape (batch_size).",
        ));
    }
    Ok(batch_size)
}

/// helper.h CheckBlockTable — returns `max_num_blocks_per_seq`.
fn check_block_table(block_table: &Shape, batch_size: i64) -> Res<i64> {
    if block_table.rank() != 2 {
        return Err(PagedAttentionError::invalid("block_table must be 2D."));
    }
    if block_table.dims()[0] != batch_size {
        return Err(PagedAttentionError::invalid(format!(
            "block_table dimension 0 should be batch_size, got {}",
            block_table.dims()[0]
        )));
    }
    Ok(block_table.dims()[1])
}

/// helper.h CheckSlotMapping.
fn check_slot_mapping(slot_mapping: &Shape, token_count: i64) -> Res<()> {
    if slot_mapping.rank() != 1 {
        return Err(PagedAttentionError::invalid(format!(
            "Input 'slot_mapping' is expected to have 1 dimension, got {}",
            slot_mapping.rank()
        )));
    }
    if slot_mapping.dims()[0] != token_count {
        return Err(PagedAttentionError::invalid(format!(
            "Input 'slot_mapping' dimension 0 should be token_count ({token_count}), got {}",
            slot_mapping.dims()[0]
        )));
    }
    Ok(())
}

/// helper.h CheckHeadSink.
fn check_head_sink(head_sink: &Shape, num_heads: i64) -> Res<()> {
    if head_sink.rank() != 1 {
        return Err(PagedAttentionError::invalid(
            "head_sink must be a 1D tensor",
        ));
    }
    if head_sink.dims()[0] != num_heads {
        return Err(PagedAttentionError::invalid(format!(
            "head_sink dimension 0 must be equal to the num heads, got {}",
            head_sink.dims()[0]
        )));
    }
    Ok(())
}

/// helper.h CheckQKNormWeights.
fn check_qk_norm_weights(
    q_norm_weight: Option<&Shape>,
    k_norm_weight: Option<&Shape>,
    head_size: i64,
) -> Res<()> {
    if q_norm_weight.is_some() != k_norm_weight.is_some() {
        return Err(PagedAttentionError::invalid(
            "Input 'q_norm_weight' and 'k_norm_weight' must be provided together.",
        ));
    }
    let (Some(q), Some(k)) = (q_norm_weight, k_norm_weight) else {
        return Ok(());
    };
    if q.rank() != 1 || q.dims()[0] != head_size {
        return Err(PagedAttentionError::invalid(format!(
            "Input 'q_norm_weight' must be a 1D tensor of shape (head_size) = ({head_size})."
        )));
    }
    if k.rank() != 1 || k.dims()[0] != head_size {
        return Err(PagedAttentionError::invalid(format!(
            "Input 'k_norm_weight' must be a 1D tensor of shape (head_size) = ({head_size})."
        )));
    }
    Ok(())
}

/// helper.h CheckKVCacheQuantization (one K or V side).
#[allow(clippy::too_many_arguments)]
fn check_kv_cache_quantization(
    scale: Option<&Shape>,
    scale_name: &str,
    quant_type_name: &str,
    quant_type: KvQuantType,
    is_quantized_cache: bool,
    kv_num_heads: i64,
    head_size: i64,
) -> Res<()> {
    if quant_type == KvQuantType::None {
        if scale.is_some() {
            return Err(PagedAttentionError::invalid(format!(
                "Input '{scale_name}' must not be provided when '{quant_type_name}' is 'NONE'."
            )));
        }
        if is_quantized_cache {
            return Err(PagedAttentionError::invalid(format!(
                "The KV cache has a quantized element type, so '{quant_type_name}' must be 'PER_TENSOR' or 'PER_CHANNEL'."
            )));
        }
        return Ok(());
    }
    if !is_quantized_cache {
        return Err(PagedAttentionError::invalid(format!(
            "'{quant_type_name}' is set, but the KV cache element type is not quantized. Use an int8 or float8e4m3fn cache, or set '{quant_type_name}' to 'NONE'."
        )));
    }
    let Some(scale) = scale else {
        return Err(PagedAttentionError::invalid(format!(
            "Input '{scale_name}' is required when '{quant_type_name}' is not 'NONE'."
        )));
    };
    let count = shape_size(scale);
    if quant_type == KvQuantType::PerTensor {
        if count != 1 {
            return Err(PagedAttentionError::invalid(format!(
                "Input '{scale_name}' must have exactly 1 element for PER_TENSOR quantization, got {count}"
            )));
        }
        return Ok(());
    }
    // PER_CHANNEL: count == kv_num_heads*head_size and trailing dim == head_size.
    if count != kv_num_heads * head_size
        || scale.dims().is_empty()
        || *scale.dims().last().expect("non-empty") != head_size
    {
        return Err(PagedAttentionError::invalid(format!(
            "Input '{scale_name}' must have shape (kv_num_heads, 1, head_size) = ({kv_num_heads}, 1, {head_size}) for PER_CHANNEL quantization, got {}",
            scale.to_string_compact()
        )));
    }
    Ok(())
}

/// helper.h CheckKVCacheDataType.
fn check_kv_cache_dtype(logical: KvCacheDtype, storage: KvCacheDtype, attr_name: &str) -> Res<()> {
    if logical == KvCacheDtype::Default || logical == storage {
        return Ok(());
    }
    if logical.is_sub_byte() {
        return Err(PagedAttentionError::invalid(format!(
            "'{attr_name}' == '{}' requires a uint8 packed cache, which is not enabled in this build.",
            logical.as_str()
        )));
    }
    Err(PagedAttentionError::invalid(format!(
        "'{attr_name}' is '{}', but the cache tensor's element type is '{}'. Leave the attribute at '' to use the tensor's element type.",
        logical.as_str(),
        storage.as_str()
    )))
}

fn require_rank(shape: &Shape, rank: usize, name: &str) -> Res<()> {
    if shape.rank() != rank {
        return Err(PagedAttentionError::invalid(format!(
            "Input '{name}' is expected to have {rank} dimensions, got {}",
            shape.rank()
        )));
    }
    Ok(())
}

fn require_rank_shape(shape: &Shape, rank: usize, name: &str) -> Res<()> {
    require_rank(shape, rank, name)
}
