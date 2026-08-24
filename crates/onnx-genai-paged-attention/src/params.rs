//! Attribute set, input-shape set, and derived parameters for a
//! `PagedAttention` node. Shapes are concrete (`i64` dims) so the validator can
//! reproduce the helper's shape checks exactly.

use crate::types::{KvCacheDtype, KvCacheLayout, KvQuantType};

/// A concrete tensor shape (all dims known). ORT allows symbolic dims at graph
/// build time; the runtime validator this mirrors runs on concrete shapes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(pub Vec<i64>);

impl Shape {
    #[must_use]
    pub fn new(dims: impl Into<Vec<i64>>) -> Self {
        Self(dims.into())
    }
    #[must_use]
    pub fn rank(&self) -> usize {
        self.0.len()
    }
    #[must_use]
    pub fn dims(&self) -> &[i64] {
        &self.0
    }
    #[must_use]
    pub fn to_string_compact(&self) -> String {
        let inner = self
            .0
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!("({inner})")
    }
}

impl From<Vec<i64>> for Shape {
    fn from(v: Vec<i64>) -> Self {
        Self(v)
    }
}

/// The 15 `PagedAttention` v1 attributes (`bert_defs.cc:1545-1620`).
#[derive(Debug, Clone)]
pub struct PagedAttentionAttributes {
    pub num_heads: i64,
    pub kv_num_heads: i64,
    /// `None` means "unset" → the `1/sqrt(head_size)` default (which the helper
    /// forbids when `v_head_size != head_size`).
    pub scale: Option<f32>,
    pub softcap: f32,
    pub local_window_size: i64,
    pub do_rotary: bool,
    pub rotary_interleaved: bool,
    pub qk_norm_epsilon: f32,
    pub k_quant_type: KvQuantType,
    pub v_quant_type: KvQuantType,
    pub k_cache_dtype: KvCacheDtype,
    pub v_cache_dtype: KvCacheDtype,
    pub kv_cache_layout: KvCacheLayout,
    /// `0` means "same as head_size".
    pub v_head_size: i64,
    pub rotary_offset: i64,
}

impl Default for PagedAttentionAttributes {
    fn default() -> Self {
        Self {
            num_heads: 0,
            kv_num_heads: 0,
            scale: None,
            softcap: 0.0,
            local_window_size: -1,
            do_rotary: false,
            rotary_interleaved: false,
            qk_norm_epsilon: 1e-6,
            k_quant_type: KvQuantType::None,
            v_quant_type: KvQuantType::None,
            k_cache_dtype: KvCacheDtype::Default,
            v_cache_dtype: KvCacheDtype::Default,
            kv_cache_layout: KvCacheLayout::Separate,
            v_head_size: 0,
            rotary_offset: 0,
        }
    }
}

/// The 17 `PagedAttention` v1 inputs, expressed as shapes (`Some`) or absence
/// (`None`) for the optional inputs. `key_cache`/`value_cache` also carry their
/// concrete storage element type so the quantization contract can be checked.
#[derive(Debug, Clone)]
pub struct PagedAttentionInputs {
    pub query: Shape,
    pub key: Option<Shape>,
    pub value: Option<Shape>,
    pub key_cache: Shape,
    pub key_cache_storage_dtype: KvCacheDtype,
    pub value_cache: Option<Shape>,
    pub value_cache_storage_dtype: KvCacheDtype,
    pub cumulative_sequence_length: Shape,
    pub past_seqlens: Shape,
    pub block_table: Shape,
    pub cos_cache: Option<Shape>,
    pub sin_cache: Option<Shape>,
    pub slot_mapping: Option<Shape>,
    pub head_sink: Option<Shape>,
    pub q_norm_weight: Option<Shape>,
    pub k_norm_weight: Option<Shape>,
    pub k_scale: Option<Shape>,
    pub v_scale: Option<Shape>,
    pub attention_metadata: Option<Shape>,
    /// The rotary `rotary_dim` implied by `cos_cache`/`sin_cache` last dim × 2.
    /// `0` when there is no rotary cache. Kept here because the validator needs
    /// it to enforce `rotary_offset + rotary_dim <= head_size`.
    pub rotary_dim: i64,
}

/// Parameters derived by [`crate::validate::check_inputs`] once all invariants
/// hold — the exact analogue of ORT's `PagedAttentionParameters`. These feed the
/// CPU oracle and a native kernel's launch geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct PagedAttentionParameters {
    pub batch_size: i64,
    pub token_count: i64,
    pub num_heads: i64,
    pub kv_num_heads: i64,
    pub head_size: i64,
    pub v_head_size: i64,
    pub hidden_size: i64,
    pub v_hidden_size: i64,
    pub kv_hidden_size: i64,
    pub is_latent_kv: bool,
    pub is_packed_qkv: bool,
    pub rotary_offset: i64,
    pub rotary_dim: i64,
    pub block_size: i64,
    pub num_blocks: i64,
    pub max_num_blocks_per_seq: i64,
    /// The resolved softmax scale (explicit attribute, or `1/sqrt(head_size)`).
    pub scale: f32,
    pub softcap: f32,
    pub local_window_size: i64,
    pub do_rotary: bool,
    pub rotary_interleaved: bool,
    pub use_head_sink: bool,
    pub use_qk_norm: bool,
    pub qk_norm_epsilon: f32,
    pub k_quant_type: KvQuantType,
    pub v_quant_type: KvQuantType,
}
