//! Backend capability gate.
//!
//! [`check_inputs`](crate::validate::check_inputs) accepts everything the schema
//! accepts. A concrete kernel implements only a *subset* of those schema-valid
//! modes, so it must reject the rest with a typed
//! [`PagedAttentionError::NotImplemented`] — never silently miscompute. This is
//! exactly what the upstream WebGPU kernel does
//! (`contrib_ops/webgpu/bert/paged_attention.cc:500-560`), and what our native
//! CUDA `LATENT` subset does in the other direction.
//!
//! Keeping this separate from the schema validator is the `design-discipline`
//! rule: one question (is this *legal*?) and one mechanism; a second question
//! (can *this kernel* run it?) and a second mechanism.

use crate::params::{PagedAttentionInputs, PagedAttentionParameters};
use crate::types::{KvCacheLayout, PagedAttentionError};

/// Declares which schema-valid optional modes a particular kernel implements.
/// Anything set to `false` (or a layout not in `layouts`) is rejected by
/// [`check_backend_support`] with a typed `NOT_IMPLEMENTED`.
#[derive(Debug, Clone)]
pub struct NativeSubset {
    pub name: &'static str,
    pub separate: bool,
    pub latent: bool,
    pub softcap: bool,
    pub local_window: bool,
    pub narrower_v_head_size: bool,
    pub rotary_offset: bool,
    pub head_sink: bool,
    pub qk_norm: bool,
    pub quantized_cache: bool,
    pub slot_mapping: bool,
    pub attention_metadata: bool,
    pub packed_qkv: bool,
}

impl NativeSubset {
    /// The exact subset the upstream **WebGPU** kernel implements: `SEPARATE`
    /// only, every optional feature rejected. Packed-QKV *is* supported (the
    /// kernel splits it via `RunSplitPackedQKV`, `webgpu/bert/paged_attention.cc`),
    /// so it is allowed here. Encoded so the native side can be diffed against a
    /// known-good precedent.
    #[must_use]
    pub fn webgpu_separate() -> Self {
        Self {
            name: "webgpu-separate",
            separate: true,
            latent: false,
            softcap: false,
            local_window: false,
            narrower_v_head_size: false,
            rotary_offset: false,
            head_sink: false,
            qk_norm: false,
            quantized_cache: false,
            slot_mapping: false,
            attention_metadata: false,
            packed_qkv: true,
        }
    }

    /// The native first-slice target: GLM-5.2 `--glm-full-attention` dense MLA,
    /// i.e. the `LATENT` absorbed path with a narrower `v_head_size`, partial
    /// RoPE (`rotary_offset`), scheduler `slot_mapping`, and host
    /// `attention_metadata` for CUDA-graph capture. Every other optional mode is
    /// rejected until a later slice claims it.
    #[must_use]
    pub fn glm_dense_mla_latent() -> Self {
        Self {
            name: "glm-dense-mla-latent",
            separate: false,
            latent: true,
            softcap: false,
            local_window: false,
            narrower_v_head_size: true,
            rotary_offset: true,
            head_sink: false,
            qk_norm: false,
            quantized_cache: false,
            slot_mapping: true,
            attention_metadata: true,
            packed_qkv: false,
        }
    }
}

/// Reject any schema-valid mode this `subset` does not implement.
///
/// `params` must be the output of [`check_inputs`](crate::validate::check_inputs)
/// (i.e. the node is already schema-valid).
///
/// # Errors
/// Returns [`PagedAttentionError::NotImplemented`] for the first unsupported mode.
pub fn check_backend_support(
    params: &PagedAttentionParameters,
    inputs: &PagedAttentionInputs,
    subset: &NativeSubset,
) -> Result<(), PagedAttentionError> {
    let who = subset.name;
    let layout = if params.is_latent_kv {
        KvCacheLayout::Latent
    } else {
        KvCacheLayout::Separate
    };
    let layout_ok = match layout {
        KvCacheLayout::Separate => subset.separate,
        KvCacheLayout::Latent => subset.latent,
    };
    if !layout_ok {
        return Err(PagedAttentionError::unimplemented(format!(
            "PagedAttention ({who}): kv_cache_layout={} is not supported.",
            if params.is_latent_kv {
                "LATENT"
            } else {
                "SEPARATE"
            }
        )));
    }
    if params.is_packed_qkv && !subset.packed_qkv {
        return Err(PagedAttentionError::unimplemented(format!(
            "PagedAttention ({who}): packed QKV is not supported."
        )));
    }
    if params.softcap != 0.0 && !subset.softcap {
        return Err(PagedAttentionError::unimplemented(format!(
            "PagedAttention ({who}): non-zero softcap is not supported."
        )));
    }
    if params.local_window_size != -1 && !subset.local_window {
        return Err(PagedAttentionError::unimplemented(format!(
            "PagedAttention ({who}): local_window_size != -1 is not supported."
        )));
    }
    if params.v_head_size != params.head_size && !subset.narrower_v_head_size {
        return Err(PagedAttentionError::unimplemented(format!(
            "PagedAttention ({who}): v_head_size != head_size is not supported."
        )));
    }
    if params.rotary_offset != 0 && !subset.rotary_offset {
        return Err(PagedAttentionError::unimplemented(format!(
            "PagedAttention ({who}): rotary_offset != 0 is not supported."
        )));
    }
    if params.use_head_sink && !subset.head_sink {
        return Err(PagedAttentionError::unimplemented(format!(
            "PagedAttention ({who}): head_sink input is not supported."
        )));
    }
    if params.use_qk_norm && !subset.qk_norm {
        return Err(PagedAttentionError::unimplemented(format!(
            "PagedAttention ({who}): q_norm_weight/k_norm_weight inputs are not supported."
        )));
    }
    let quantized = !params.k_quant_type.is_none() || !params.v_quant_type.is_none();
    if quantized && !subset.quantized_cache {
        return Err(PagedAttentionError::unimplemented(format!(
            "PagedAttention ({who}): quantized KV cache is not supported."
        )));
    }
    if inputs.slot_mapping.is_some() && !subset.slot_mapping {
        return Err(PagedAttentionError::unimplemented(format!(
            "PagedAttention ({who}): slot_mapping input is not supported."
        )));
    }
    if inputs.attention_metadata.is_some() && !subset.attention_metadata {
        return Err(PagedAttentionError::unimplemented(format!(
            "PagedAttention ({who}): attention_metadata input is not supported."
        )));
    }
    Ok(())
}
