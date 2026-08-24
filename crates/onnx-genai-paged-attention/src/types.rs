//! Enums and the error type for the `PagedAttention` schema, faithful to
//! ORT 1.29.0 (`bert_defs.cc` attribute definitions and
//! `paged_attention_helper.h` string parsing).

/// Error categories mirroring the two status codes the op can return.
///
/// * [`InvalidArgument`](Self::InvalidArgument) — a *schema* violation, exactly
///   what `paged_attention_helper.h::CheckInputs` rejects.
/// * [`NotImplemented`](Self::NotImplemented) — a schema-valid mode that a
///   particular backend subset does not implement, exactly what the upstream
///   WebGPU kernel returns (`contrib_ops/webgpu/bert/paged_attention.cc`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PagedAttentionError {
    /// A schema-level rejection (`ORT_MAKE_STATUS(..., INVALID_ARGUMENT, ...)`).
    #[error("INVALID_ARGUMENT: {0}")]
    InvalidArgument(String),
    /// A backend capability rejection (`ORT_MAKE_STATUS(..., NOT_IMPLEMENTED, ...)`).
    #[error("NOT_IMPLEMENTED: {0}")]
    NotImplemented(String),
}

impl PagedAttentionError {
    pub(crate) fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidArgument(msg.into())
    }
    pub(crate) fn unimplemented(msg: impl Into<String>) -> Self {
        Self::NotImplemented(msg.into())
    }
    /// True for schema violations.
    #[must_use]
    pub fn is_invalid_argument(&self) -> bool {
        matches!(self, Self::InvalidArgument(_))
    }
    /// True for backend capability rejections.
    #[must_use]
    pub fn is_not_implemented(&self) -> bool {
        matches!(self, Self::NotImplemented(_))
    }
}

/// Physical layout of the KV cache (`kv_cache_layout` attribute).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvCacheLayout {
    /// Distinct `key_cache` and `value_cache` tensors, symmetric head dims.
    #[default]
    Separate,
    /// Absorbed Multi-head Latent Attention: a single cache, `kv_num_heads == 1`,
    /// V is the leading `v_head_size` channels of the same latent row as K.
    Latent,
}

impl KvCacheLayout {
    /// Parse the attribute string; ORT fails shape inference for anything else.
    ///
    /// # Errors
    /// Returns [`PagedAttentionError::InvalidArgument`] for an unknown value.
    pub fn parse(s: &str) -> Result<Self, PagedAttentionError> {
        match s {
            "SEPARATE" => Ok(Self::Separate),
            "LATENT" => Ok(Self::Latent),
            other => Err(PagedAttentionError::invalid(format!(
                "kv_cache_layout must be 'SEPARATE' or 'LATENT', got '{other}'."
            ))),
        }
    }
    #[must_use]
    pub fn is_latent(self) -> bool {
        matches!(self, Self::Latent)
    }
}

/// Quantization granularity of a KV cache side (`k_quant_type` / `v_quant_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvQuantType {
    #[default]
    None,
    PerTensor,
    PerChannel,
}

impl KvQuantType {
    /// # Errors
    /// Returns [`PagedAttentionError::InvalidArgument`] for an unknown value.
    pub fn parse(s: &str) -> Result<Self, PagedAttentionError> {
        match s {
            "NONE" => Ok(Self::None),
            "PER_TENSOR" => Ok(Self::PerTensor),
            "PER_CHANNEL" => Ok(Self::PerChannel),
            other => Err(PagedAttentionError::invalid(format!(
                "quant type must be 'NONE', 'PER_TENSOR' or 'PER_CHANNEL', got '{other}'."
            ))),
        }
    }
    #[must_use]
    pub fn is_none(self) -> bool {
        matches!(self, Self::None)
    }
}

/// Logical element type of a KV cache (`k_cache_dtype` / `v_cache_dtype`), and
/// also the concrete *storage* element type of the cache tensor.
///
/// `Default` (the empty attribute string) means "same as the tensor element
/// type". The sub-byte members (`Int4`, `Float4E2M1`) are reserved by the schema
/// but decoded by no backend, so they are always rejected by the dtype check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvCacheDtype {
    /// Empty attribute string: use the cache tensor's own element type.
    #[default]
    Default,
    Float16,
    BFloat16,
    Int8,
    Float8E4M3Fn,
    Int4,
    Float4E2M1,
}

impl KvCacheDtype {
    /// Parse a `k_cache_dtype` / `v_cache_dtype` attribute string.
    ///
    /// # Errors
    /// Returns [`PagedAttentionError::InvalidArgument`] for an unknown value.
    pub fn parse(s: &str) -> Result<Self, PagedAttentionError> {
        match s {
            "" => Ok(Self::Default),
            "float16" => Ok(Self::Float16),
            "bfloat16" => Ok(Self::BFloat16),
            "int8" => Ok(Self::Int8),
            "float8e4m3fn" => Ok(Self::Float8E4M3Fn),
            "int4" => Ok(Self::Int4),
            "float4e2m1" => Ok(Self::Float4E2M1),
            other => Err(PagedAttentionError::invalid(format!(
                "k_cache_dtype/v_cache_dtype must be one of '', 'float16', 'bfloat16', 'int8', \
                 'float8e4m3fn', 'int4', 'float4e2m1', got '{other}'."
            ))),
        }
    }

    /// Whether this *storage* element type is a quantized cache type
    /// (`IsQuantizedKVCacheDataType` in ORT). Only `int8` and `float8e4m3fn`
    /// are valid quantized storage types for the shipped kernels.
    #[must_use]
    pub fn is_quantized_storage(self) -> bool {
        matches!(self, Self::Int8 | Self::Float8E4M3Fn)
    }

    /// Whether this logical type is a sub-byte packed type.
    #[must_use]
    pub fn is_sub_byte(self) -> bool {
        matches!(self, Self::Int4 | Self::Float4E2M1)
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "",
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
            Self::Int8 => "int8",
            Self::Float8E4M3Fn => "float8e4m3fn",
            Self::Int4 => "int4",
            Self::Float4E2M1 => "float4e2m1",
        }
    }
}
