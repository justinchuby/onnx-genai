//! Descriptor-driven KV-cache stride indexing for converted GQA paths.
//!
//! The CUDA EP used to select the KV cache physical layout with a two-valued
//! `kv_layout` integer (`0` = head-major BNSH, `1` = seq-major BSNH) that every
//! converted kernel branched on at runtime. This module replaces that branch
//! with a stride descriptor: each axis's element stride is stored symbolically
//! (the set of runtime dimensions it multiplies), and the kernels **generate**
//! their index arithmetic from the descriptor at NVRTC module-build time. A JIT
//! backend compiles a specialized module per descriptor — keyed into the module
//! cache — so there is no runtime indexing cost and no two-valued branch.
//!
//! Only the two named layouts are honored today; any descriptor the converted
//! path cannot honor is rejected with an error rather than silently
//! mis-indexing (see [`KvCacheStrides::require_converted_path_support`]).

use onnx_runtime_ep_api::{EpError, Result};

/// A runtime KV dimension an axis stride can be a multiple of. Mirrors the
/// metadata `KvStrideDim`; kept local so the EP does not depend on the metadata
/// crate for a three-value tag.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum KvDim {
    /// Number of KV heads.
    KvHeads,
    /// Sequence/token capacity of the growing axis.
    SeqCapacity,
    /// Per-token head width.
    HeadDim,
}

/// A GQA path that may touch the physical KV-cache layout.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum KvCachePath {
    FusedDecodePrep,
    FlashPrefillPrep,
    FlashPrefillRead,
    Fp16DecodeRead,
    Bf16DecodeRead,
    UnfusedDecodePrep,
    F32DecodeRead,
    ReferenceRead,
    Phase2aRead,
}

impl KvCachePath {
    fn name(self) -> &'static str {
        match self {
            Self::FusedDecodePrep => "fused fp16 decode prep",
            Self::FlashPrefillPrep => "flash prefill prep",
            Self::FlashPrefillRead => "flash prefill read",
            Self::Fp16DecodeRead => "fp16 split-K decode read",
            Self::Bf16DecodeRead => "bf16 split-K decode read",
            Self::UnfusedDecodePrep => "unfused decode prep",
            Self::F32DecodeRead => "f32 split-K decode read",
            Self::ReferenceRead => "reference attention read",
            Self::Phase2aRead => "phase2a attention read",
        }
    }
}

const CONVERTED_PATHS: &[KvCachePath] = &[
    KvCachePath::FusedDecodePrep,
    KvCachePath::FlashPrefillPrep,
    KvCachePath::FlashPrefillRead,
    KvCachePath::Fp16DecodeRead,
    KvCachePath::Bf16DecodeRead,
];

/// Symbolic per-axis strides of a KV-cache binding, general over layout.
///
/// The concrete element stride of an axis is the product of the runtime
/// dimensions in its factor list; an empty list is unit stride. `offset` and
/// `reservation_override` describe a binding that is a view into a larger
/// reservation (e.g. token-major); the converted path honors only whole-buffer
/// bindings and rejects the rest.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct KvCacheStrides {
    batch: Vec<KvDim>,
    head: Vec<KvDim>,
    seq: Vec<KvDim>,
    head_dim: Vec<KvDim>,
    offset_elements: u64,
    reservation_override: bool,
    /// Stable identity of a recognized named layout, used to pick a `'static`
    /// module-cache key. Empty for an unnamed (unrecognized) descriptor.
    named: &'static str,
}

impl Default for KvCacheStrides {
    fn default() -> Self {
        Self::head_major_bnsh()
    }
}

impl KvCacheStrides {
    /// Head-major BNSH `[batch, kv_heads, cap, head_dim]`.
    pub(crate) fn head_major_bnsh() -> Self {
        use KvDim::{HeadDim, KvHeads, SeqCapacity};
        KvCacheStrides {
            batch: vec![KvHeads, SeqCapacity, HeadDim],
            head: vec![SeqCapacity, HeadDim],
            seq: vec![HeadDim],
            head_dim: vec![],
            offset_elements: 0,
            reservation_override: false,
            named: "bnsh",
        }
    }

    /// Seq-major BSNH `[batch, cap, kv_heads, head_dim]`.
    pub(crate) fn seq_major_bsnh() -> Self {
        use KvDim::{HeadDim, KvHeads, SeqCapacity};
        KvCacheStrides {
            batch: vec![SeqCapacity, KvHeads, HeadDim],
            head: vec![HeadDim],
            seq: vec![KvHeads, HeadDim],
            head_dim: vec![],
            offset_elements: 0,
            reservation_override: false,
            named: "bsnh",
        }
    }

    /// Build the descriptor a native-backend GQA node's `kv_layout` attribute
    /// selects (`0` = head-major BNSH, `1` = seq-major BSNH). Any other value is
    /// rejected — the wire attribute only carries the two named layouts.
    pub(crate) fn from_attribute(kv_layout: i64) -> Result<Self> {
        match kv_layout {
            0 => Ok(Self::head_major_bnsh()),
            1 => Ok(Self::seq_major_bsnh()),
            other => Err(EpError::KernelFailed(format!(
                "cuda_ep GroupQueryAttention: kv_layout {other} must be 0 (BNSH) or 1 (BSNH)"
            ))),
        }
    }

    /// Whether this is the head-major default layout that every reader and
    /// writer honors. Non-default layouts are gated to converted paths.
    pub(crate) fn is_head_major(&self) -> bool {
        self.named == "bnsh"
    }

    /// Reject a descriptor/path pair that is not converted, rather than
    /// silently mis-indexing it. Converted paths require a contiguous
    /// (unit-stride) `head_dim` innermost axis and whole-buffer bindings.
    pub(crate) fn require_converted_path_support(&self, path: KvCachePath) -> Result<()> {
        if !self.head_dim.is_empty() {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: KV descriptor has a non-unit head_dim stride; the \
                 converted fp16 decode path requires a contiguous (half2-vectorizable) head_dim"
                    .into(),
            ));
        }
        if self.offset_elements != 0 || self.reservation_override {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: KV descriptor is a view into a larger reservation \
                 (non-zero offset or reservation-spanning seq extent); the converted path honors \
                 only whole-buffer bindings, not a token-major view"
                    .into(),
            ));
        }
        if self.named.is_empty() {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: KV descriptor does not match a layout with a cached \
                 specialized module; only head-major BNSH and seq-major BSNH are honored"
                    .into(),
            ));
        }
        if !self.is_head_major() && !CONVERTED_PATHS.contains(&path) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GroupQueryAttention: seq-major (BSNH) KV cannot use the {} path; \
                 converted paths are fused fp16 decode prep, flash prefill prep/read, and fp16 \
                 split-K decode read",
                path.name()
            )));
        }
        Ok(())
    }

    /// The `'static` module-cache key for the fp16 split-K decode read kernel.
    pub(crate) fn decode_module_key(&self) -> Result<&'static str> {
        match self.named {
            "bnsh" => Ok("gqa_decode_attention_f16_v8_bnsh"),
            "bsnh" => Ok("gqa_decode_attention_f16_v8_bsnh"),
            _ => Err(self.no_cached_module_err()),
        }
    }

    /// The `'static` module-cache key for the bf16 split-K decode read kernel.
    ///
    /// bf16 shares the fp16 kernel's structure and KV-stride prelude but emits a
    /// distinct NVRTC source (different element type and intrinsics), so it must
    /// use its own module-cache key to avoid colliding with the fp16 module.
    pub(crate) fn decode_module_key_bf16(&self) -> Result<&'static str> {
        match self.named {
            "bnsh" => Ok("gqa_decode_attention_bf16_v1_bnsh"),
            "bsnh" => Ok("gqa_decode_attention_bf16_v1_bsnh"),
            _ => Err(self.no_cached_module_err()),
        }
    }

    /// The `'static` module-cache key for a non-default f32 prep module.
    pub(crate) fn prep_f32_module_key(&self) -> Result<&'static str> {
        match self.named {
            "bsnh" => Ok("group_query_attention_prep_v4_bsnh"),
            _ => Err(self.no_cached_module_err()),
        }
    }

    /// The `'static` module-cache key for a non-default fp16/bf16 prep module.
    pub(crate) fn prep_half_module_key(&self) -> Result<&'static str> {
        match self.named {
            "bsnh" => Ok("group_query_attention_prep_half_v4_bsnh"),
            _ => Err(self.no_cached_module_err()),
        }
    }

    pub(crate) fn flash_f32_module_key(&self) -> Result<&'static str> {
        match self.named {
            "bnsh" => Ok("flash_attention_f32_v2_bnsh"),
            "bsnh" => Ok("flash_attention_f32_v2_bsnh"),
            _ => Err(self.no_cached_module_err()),
        }
    }

    pub(crate) fn flash_half_module_key(&self) -> Result<&'static str> {
        match self.named {
            "bnsh" => Ok("flash_attention_half_v3_bnsh"),
            "bsnh" => Ok("flash_attention_half_v3_bsnh"),
            _ => Err(self.no_cached_module_err()),
        }
    }

    fn no_cached_module_err(&self) -> EpError {
        EpError::KernelFailed(
            "cuda_ep GroupQueryAttention: no cached specialized module for this KV descriptor; \
             only head-major BNSH and seq-major BSNH are honored"
                .into(),
        )
    }

    /// The CUDA `#define` prelude that specializes the fp16 decode read kernel's
    /// KV base/stride arithmetic for this descriptor. The macros reference the
    /// kernel's in-scope `batch_index`, `kv_head`, `kv_heads`, `cache_capacity`,
    /// and `head_size` variables.
    pub(crate) fn decode_prelude(&self) -> String {
        let batch = product_expr(&self.batch, "kv_heads", "cache_capacity", "head_size");
        let head = product_expr(&self.head, "kv_heads", "cache_capacity", "head_size");
        let seq = product_expr(&self.seq, "kv_heads", "cache_capacity", "head_size");
        format!(
            "#define GQA_KV_BASE(b, h) ( (long)(b) * {batch} + (long)(h) * {head} )\n\
             #define GQA_KV_STRIDE ( {seq} )\n"
        )
    }

    /// The CUDA `#define` prelude that specializes flash/prefill KV reads.
    pub(crate) fn flash_prelude(&self) -> String {
        let batch = product_expr(&self.batch, "kv_heads", "kv_capacity", "dim");
        let head = product_expr(&self.head, "kv_heads", "kv_capacity", "dim");
        let seq = product_expr(&self.seq, "kv_heads", "kv_capacity", "dim");
        format!(
            "#define FLASH_KV_BASE(b, h) ( (long)(b) * {batch} + (long)(h) * {head} )\n\
             #define FLASH_KV_STRIDE ( {seq} )\n"
        )
    }

    /// The CUDA prelude that specializes prep cache reads/writes. The generic
    /// index macro accepts the runtime head count, capacity, and head dimension
    /// so the build path can address past and present buffers with different
    /// capacities.
    pub(crate) fn prep_prelude(&self) -> String {
        let batch = product_expr(&self.batch, "heads", "capacity", "dim");
        let head = product_expr(&self.head, "heads", "capacity", "dim");
        let seq = product_expr(&self.seq, "heads", "capacity", "dim");
        format!(
            "#define GQA_KV_INDEX(b, h, slot, heads, capacity, dim) ( (long)(b) * {batch} \
             + (long)(h) * {head} + (long)(slot) * {seq} )\n\
             #define GQA_KV_DST(b, h, slot) \
             GQA_KV_INDEX(b, h, slot, kv_heads, present_capacity, dim)\n"
        )
    }
}

/// Render a symbolic axis stride as a CUDA `long` product expression over the
/// named runtime-dimension variables. An empty factor list is unit stride.
fn product_expr(factors: &[KvDim], kv_heads: &str, capacity: &str, head_dim: &str) -> String {
    if factors.is_empty() {
        return "1".to_string();
    }
    let parts: Vec<String> = factors
        .iter()
        .map(|factor| match factor {
            KvDim::KvHeads => format!("(long){kv_heads}"),
            KvDim::SeqCapacity => format!("(long){capacity}"),
            KvDim::HeadDim => format!("(long){head_dim}"),
        })
        .collect();
    format!("({})", parts.join(" * "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attribute_maps_to_named_layouts() {
        assert!(KvCacheStrides::from_attribute(0).unwrap().is_head_major());
        assert!(!KvCacheStrides::from_attribute(1).unwrap().is_head_major());
        assert!(KvCacheStrides::from_attribute(2).is_err());
    }

    #[test]
    fn named_layouts_are_honored() {
        assert!(
            KvCacheStrides::head_major_bnsh()
                .require_converted_path_support(KvCachePath::ReferenceRead)
                .is_ok()
        );
        assert!(
            KvCacheStrides::seq_major_bsnh()
                .require_converted_path_support(KvCachePath::FlashPrefillRead)
                .is_ok()
        );
        assert!(
            KvCacheStrides::seq_major_bsnh()
                .require_converted_path_support(KvCachePath::ReferenceRead)
                .is_err()
        );
    }

    // A descriptor the converted path cannot honor must be rejected with an
    // error rather than silently mis-indexing.
    #[test]
    fn reservation_view_is_rejected() {
        let mut view = KvCacheStrides::seq_major_bsnh();
        view.offset_elements = 4096;
        view.reservation_override = true;
        let error = view
            .require_converted_path_support(KvCachePath::Fp16DecodeRead)
            .unwrap_err();
        assert!(matches!(error, EpError::KernelFailed(_)));
        assert!(view.decode_module_key().is_ok());
    }

    #[test]
    fn non_unit_head_dim_is_rejected() {
        let mut bad = KvCacheStrides::head_major_bnsh();
        bad.head_dim = vec![KvDim::KvHeads];
        assert!(
            bad.require_converted_path_support(KvCachePath::Fp16DecodeRead)
                .is_err()
        );
    }

    // The generated arithmetic must reproduce the two hand-written layouts.
    #[test]
    fn decode_prelude_matches_layout_formulae() {
        let bnsh = KvCacheStrides::head_major_bnsh().decode_prelude();
        assert!(bnsh.contains(
            "#define GQA_KV_BASE(b, h) ( (long)(b) * \
             ((long)kv_heads * (long)cache_capacity * (long)head_size) \
             + (long)(h) * ((long)cache_capacity * (long)head_size) )"
        ));
        assert!(bnsh.contains("#define GQA_KV_STRIDE ( ((long)head_size) )"));

        let bsnh = KvCacheStrides::seq_major_bsnh().decode_prelude();
        assert!(bsnh.contains("#define GQA_KV_STRIDE ( ((long)kv_heads * (long)head_size) )"));
    }

    #[test]
    fn prep_prelude_matches_layout_formulae() {
        let bnsh = KvCacheStrides::head_major_bnsh().prep_prelude();
        // Head-major dst == ((b*N + h)*cap + slot)*dim, written here as the
        // distributed sum b*(N*cap*dim) + h*(cap*dim) + slot*dim.
        assert!(bnsh.contains("(long)(slot) * ((long)dim)"));
        let bsnh = KvCacheStrides::seq_major_bsnh().prep_prelude();
        assert!(bsnh.contains("(long)(slot) * ((long)heads * (long)dim)"));
    }

    #[test]
    fn flash_prelude_matches_layout_formulae() {
        let bnsh = KvCacheStrides::head_major_bnsh().flash_prelude();
        assert!(bnsh.contains("#define FLASH_KV_STRIDE ( ((long)dim) )"));
        let bsnh = KvCacheStrides::seq_major_bsnh().flash_prelude();
        assert!(bsnh.contains("#define FLASH_KV_STRIDE ( ((long)kv_heads * (long)dim) )"));
    }
}
