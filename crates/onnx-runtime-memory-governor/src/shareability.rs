//! Layout-general prefix-shareability arithmetic (#777).
//!
//! # Why this exists
//!
//! Prefix sharing was long framed as "requires seq-major". That is a
//! **granule-relative** claim stated absolutely, and it is wrong. Whether a
//! shared KV prefix can be physically shared across sequences is **arithmetic**,
//! not a property of a named layout:
//!
//! ```text
//! fragment_bytes                 = prefix_len * (contiguous bytes per fragment in that layout)
//! shareable                      = fragment_bytes >= granule
//! shareable_granules_per_fragment= floor(fragment_bytes / granule)
//! multi_map_ops                  = fragments * shareable_granules_per_fragment
//! ```
//!
//! Layout sets `fragment_bytes` and the **cost** (how many multi-map operations
//! sharing takes), not the **possibility**. The genuine requirements are (a) the
//! KV buffer is VMM-backed — physical handles mappable at more than one virtual
//! address — and (b) `fragment_bytes >= granule` for the layout in use, on the
//! platform in use.
//!
//! Two consequences the "seq-major only" framing hid:
//!
//! * **Head-major becomes shareable when the arithmetic says so.** At a 2 MiB
//!   granule with `head_dim = 128` and fp16, a head-major fragment reaches one
//!   granule at a `granule / (head_dim * dtype)` = `2_097_152 / 256` =
//!   **8,192-token** prefix — realistic for RAG and long system prompts.
//! * **Granularity is a platform capability, queried not assumed.** #776
//!   measured CUDA `CU_MEM_ALLOC_GRANULARITY_MINIMUM == RECOMMENDED == 2 MiB` on
//!   this device; Level Zero and Vulkan sparse binding expose ~64 KiB, and CPU
//!   `mmap` 4 KiB. Callers pass the **queried** granule; this module never
//!   hardcodes one. This is the per-EP, per-platform capability #783 records.
//!
//! # What this module is *not*
//!
//! It is the shareability **decision**, not the mapping. It says whether — and
//! at what cost — a prefix can be shared for a given layout, geometry, prefix
//! length, and granule. The actual multi-map is
//! [`SharedMapping::commit_shared_prefix`](crate::SharedMapping::commit_shared_prefix);
//! a caller uses [`PrefixShareability::shareable`] to decide whether to attempt
//! it, and [`PrefixShareability::refusal_reason`] to refuse **with a reason**
//! rather than silently falling back to N private copies — the same "error,
//! never mis-map" discipline the capability's own methods keep.

/// Geometry of a model's per-layer KV cache, in the terms the fragment sizes
/// are computed from.
///
/// This is the layout-independent shape: how the per-token KV bytes decompose
/// into layers, heads, and per-head width. A [`KvLayout`] turns it into a
/// [`KvFragmentation`] (how those bytes are scattered into contiguous
/// fragments).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ModelKvGeometry {
    /// Number of transformer layers (each contributes a K and a V side).
    pub layers: u64,
    /// Number of KV heads (post-GQA grouping), per layer per side.
    pub kv_heads: u64,
    /// Per-head width in elements.
    pub head_dim: u64,
    /// Bytes per KV element (2 for fp16/bf16, 1 for fp8, etc.).
    pub dtype_bytes: u64,
}

impl ModelKvGeometry {
    /// Total KV bytes one token contributes across the whole model
    /// (`layers * 2 * kv_heads * head_dim * dtype_bytes`). This is invariant
    /// across layouts — layout only decides how these bytes are *fragmented*.
    pub fn bytes_per_token_total(&self) -> u64 {
        self.layers
            .saturating_mul(2)
            .saturating_mul(self.kv_heads)
            .saturating_mul(self.head_dim)
            .saturating_mul(self.dtype_bytes)
    }
}

/// How a physical KV layout scatters a token's KV bytes into independent
/// **contiguous fragments**.
///
/// A "fragment" is one maximal run of bytes that is contiguous in device
/// address space and grows along the sequence axis, so that a `prefix_len`-token
/// prefix occupies `prefix_len * contiguous_bytes_per_token` contiguous bytes
/// within it. Sharing maps whole granules, so only granules that fall entirely
/// inside such a run can be shared.
///
/// This is deliberately a *descriptor*, not a closed enum keyed to a named
/// layout: per #783 the layout is a per-EP, per-platform capability, and a JIT
/// backend can present a token-major view or any other stride arrangement. The
/// named constructors below cover the three cases measured in #777/#787; an
/// arbitrary descriptor is built directly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct KvFragmentation {
    /// How many independent contiguous fragments a full-model prefix occupies.
    pub fragments: u64,
    /// Contiguous bytes **one token** contributes to **each** fragment.
    pub contiguous_bytes_per_token: u64,
}

impl KvFragmentation {
    /// Head-major BNSH `[batch, kv_heads, seq, head_dim]`.
    ///
    /// Each `(layer, side, head)` owns its own `max_seq x head_dim` stripe, so a
    /// prefix is `layers * 2 * kv_heads` fragments, each contiguous over
    /// `head_dim` per token.
    pub fn head_major_bnsh(geometry: ModelKvGeometry) -> Self {
        Self {
            fragments: geometry
                .layers
                .saturating_mul(2)
                .saturating_mul(geometry.kv_heads),
            contiguous_bytes_per_token: geometry.head_dim.saturating_mul(geometry.dtype_bytes),
        }
    }

    /// Seq-major BSNH `[batch, seq, kv_heads, head_dim]`.
    ///
    /// Each `(layer, side)` buffer is dense across heads, so a prefix is
    /// `layers * 2` fragments, each contiguous over `kv_heads * head_dim` per
    /// token.
    pub fn seq_major_bsnh(geometry: ModelKvGeometry) -> Self {
        Self {
            fragments: geometry.layers.saturating_mul(2),
            contiguous_bytes_per_token: geometry
                .kv_heads
                .saturating_mul(geometry.head_dim)
                .saturating_mul(geometry.dtype_bytes),
        }
    }

    /// Token-major across all layers: the whole model's KV for a token is one
    /// dense run, so a prefix is a single fragment contiguous over
    /// `layers * 2 * kv_heads * head_dim` per token.
    pub fn token_major(geometry: ModelKvGeometry) -> Self {
        Self {
            fragments: 1,
            contiguous_bytes_per_token: geometry.bytes_per_token_total(),
        }
    }
}

/// The shareability decision for one `(layout, geometry, prefix_len, granule)`.
///
/// Every field is derived from the arithmetic in the module docs; nothing here
/// is keyed to a named layout.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PrefixShareability {
    /// Whether any granule of the prefix can be shared at all
    /// (`fragment_bytes >= granule`).
    pub shareable: bool,
    /// The queried device granule used for this decision, in bytes.
    pub granule: u64,
    /// Number of contiguous fragments the prefix occupies.
    pub fragments: u64,
    /// Contiguous prefix bytes in **one** fragment
    /// (`prefix_len * contiguous_bytes_per_token`).
    pub fragment_bytes: u64,
    /// Whole granules of **one** fragment that fall entirely inside the shared
    /// prefix and can therefore be shared: `floor(fragment_bytes / granule)`.
    pub shareable_granules_per_fragment: u64,
    /// Total multi-map operations sharing costs:
    /// `fragments * shareable_granules_per_fragment`. This is the layout-driven
    /// **cost** — 768 vs 96 vs 1 for the three named layouts on qwen14b — not a
    /// possibility gate.
    pub multi_map_ops: u64,
    /// Prefix bytes that **cannot** be shared because they land in a granule
    /// that also holds private continuation (the straddling boundary granule is
    /// private per sequence), summed across all fragments, **per sequence**.
    ///
    /// When `shareable` is false every prefix byte falls here (no whole granule
    /// fits); when `fragment_bytes` is an exact multiple of the granule this is
    /// zero (the boundary is clean).
    pub wasted_boundary_bytes_per_sequence: u64,
}

impl PrefixShareability {
    /// A caller-facing reason a prefix is not shareable, or `None` when it is.
    ///
    /// This is the string a KV path uses to **refuse with a reason** instead of
    /// silently making N private copies.
    pub fn refusal_reason(&self) -> Option<&'static str> {
        if self.shareable {
            None
        } else {
            Some(
                "prefix not shareable: each contiguous KV fragment is smaller than one mapping \
                 granule (fragment_bytes < granule), so no whole granule falls entirely inside \
                 the shared prefix; lengthen the prefix, use a layout with larger fragments, or \
                 run on a finer-granule device",
            )
        }
    }
}

/// Compute whether — and at what cost — a `prefix_len`-token prefix laid out as
/// `fragmentation` can be physically shared at `granule` bytes.
///
/// This is the single authority that replaces any "is this seq-major" check.
/// It performs the module's arithmetic and nothing else; it does not consult a
/// device, allocate, or map.
///
/// `granule` must be the **queried** platform granule (`> 0`). A zero granule is
/// treated as "no valid granule" and yields a non-shareable decision rather than
/// dividing by zero.
pub fn evaluate_prefix_shareability(
    fragmentation: KvFragmentation,
    prefix_len: u64,
    granule: u64,
) -> PrefixShareability {
    let fragment_bytes = prefix_len.saturating_mul(fragmentation.contiguous_bytes_per_token);
    let fragments = fragmentation.fragments;

    if granule == 0 {
        return PrefixShareability {
            shareable: false,
            granule,
            fragments,
            fragment_bytes,
            shareable_granules_per_fragment: 0,
            multi_map_ops: 0,
            wasted_boundary_bytes_per_sequence: fragments.saturating_mul(fragment_bytes),
        };
    }

    let shareable_granules_per_fragment = fragment_bytes / granule;
    let shareable = shareable_granules_per_fragment > 0;
    let boundary_remainder_per_fragment =
        fragment_bytes - shareable_granules_per_fragment * granule;
    let multi_map_ops = fragments.saturating_mul(shareable_granules_per_fragment);
    let wasted_boundary_bytes_per_sequence =
        fragments.saturating_mul(boundary_remainder_per_fragment);

    PrefixShareability {
        shareable,
        granule,
        fragments,
        fragment_bytes,
        shareable_granules_per_fragment,
        multi_map_ops,
        wasted_boundary_bytes_per_sequence,
    }
}

/// Convenience: evaluate shareability for a named-or-custom [`KvFragmentation`]
/// built from a model geometry.
///
/// Equivalent to `evaluate_prefix_shareability(fragmentation, prefix_len,
/// granule)`; kept so a call site reads as "is this geometry's prefix
/// shareable" without a separate construction line.
pub fn evaluate_geometry_shareability(
    fragmentation: KvFragmentation,
    prefix_len: u64,
    granule: u64,
) -> PrefixShareability {
    evaluate_prefix_shareability(fragmentation, prefix_len, granule)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;
    const KIB: u64 = 1024;

    /// qwen14b: 48 layers, 8 KV heads, head_dim 128, fp16.
    fn qwen14b() -> ModelKvGeometry {
        ModelKvGeometry {
            layers: 48,
            kv_heads: 8,
            head_dim: 128,
            dtype_bytes: 2,
        }
    }

    /// qwen2.5-0.5b: 24 layers, 2 KV heads, head_dim 64, fp16.
    fn qwen05b() -> ModelKvGeometry {
        ModelKvGeometry {
            layers: 24,
            kv_heads: 2,
            head_dim: 64,
            dtype_bytes: 2,
        }
    }

    #[test]
    fn total_bytes_per_token_is_layout_invariant() {
        // fragments * contiguous_bytes_per_token must equal the whole-model
        // per-token byte count for every layout — the invariant that proves the
        // fragment descriptors partition the same bytes.
        let g = qwen14b();
        let total = g.bytes_per_token_total();
        assert_eq!(total, 196_608, "qwen14b per-token KV bytes");
        for frag in [
            KvFragmentation::head_major_bnsh(g),
            KvFragmentation::seq_major_bsnh(g),
            KvFragmentation::token_major(g),
        ] {
            assert_eq!(frag.fragments * frag.contiguous_bytes_per_token, total);
        }
    }

    #[test]
    fn qwen14b_2000_token_prefix_at_2mib_matches_the_worked_table() {
        // The issue's worked table, corrected: seq-major at 2 MiB shares
        // floor(4_096_000 / 2_097_152) = 1 granule per fragment, NOT 2. The "2"
        // in the original table is ceil (granules the prefix *touches* /
        // residency), a different quantity from whole shareable granules.
        let g = qwen14b();
        let prefix = 2_000;

        let hm = evaluate_prefix_shareability(KvFragmentation::head_major_bnsh(g), prefix, 2 * MIB);
        assert_eq!(hm.fragments, 768);
        assert_eq!(hm.fragment_bytes, 512_000); // 2000 * 128 * 2
        assert!(!hm.shareable, "head-major @2MiB fragment < granule");
        assert_eq!(hm.shareable_granules_per_fragment, 0);
        assert_eq!(hm.multi_map_ops, 0);
        assert!(hm.refusal_reason().is_some());

        let sm = evaluate_prefix_shareability(KvFragmentation::seq_major_bsnh(g), prefix, 2 * MIB);
        assert_eq!(sm.fragments, 96);
        assert_eq!(sm.fragment_bytes, 4_096_000); // 2000 * 8 * 128 * 2
        assert!(sm.shareable);
        assert_eq!(
            sm.shareable_granules_per_fragment, 1,
            "floor(4_096_000 / 2_097_152) = 1, not the table's 2 (which is ceil)"
        );
        // Consistent with the doc's own 'seq-major = 96 multi-maps per sequence'.
        assert_eq!(sm.multi_map_ops, 96);
        assert!(sm.refusal_reason().is_none());

        let tm = evaluate_prefix_shareability(KvFragmentation::token_major(g), prefix, 2 * MIB);
        assert_eq!(tm.fragments, 1);
        assert_eq!(tm.fragment_bytes, 393_216_000); // 2000 * 196_608
        assert!(tm.shareable);
        assert_eq!(tm.shareable_granules_per_fragment, 187); // floor(393_216_000 / 2_097_152)
        assert_eq!(tm.multi_map_ops, 187);
    }

    #[test]
    fn qwen14b_2000_token_prefix_at_64kib_makes_head_major_shareable() {
        // At the ~64 KiB granule Level Zero / Vulkan sparse binding expose, the
        // same head-major fragments that were unshareable at 2 MiB span ~7
        // granules each and become shareable. Layout is not the gate.
        let g = qwen14b();
        let prefix = 2_000;

        let hm =
            evaluate_prefix_shareability(KvFragmentation::head_major_bnsh(g), prefix, 64 * KIB);
        assert!(hm.shareable, "head-major becomes shareable at 64 KiB");
        assert_eq!(hm.shareable_granules_per_fragment, 7); // floor(512_000 / 65_536)
        assert_eq!(hm.multi_map_ops, 768 * 7);

        let sm = evaluate_prefix_shareability(KvFragmentation::seq_major_bsnh(g), prefix, 64 * KIB);
        assert_eq!(sm.shareable_granules_per_fragment, 62); // floor(4_096_000 / 65_536)
    }

    #[test]
    fn head_major_threshold_is_exactly_8192_tokens_at_2mib() {
        // The realistic-for-RAG threshold: head-major fragment_bytes reaches one
        // 2 MiB granule at prefix = granule / (head_dim * dtype) = 8192 tokens.
        let g = qwen14b();
        let frag = KvFragmentation::head_major_bnsh(g);

        let below = evaluate_prefix_shareability(frag, 8_191, 2 * MIB);
        assert!(
            !below.shareable,
            "8191 tokens is one token short of a granule"
        );

        let at = evaluate_prefix_shareability(frag, 8_192, 2 * MIB);
        assert!(at.shareable, "8192 tokens reaches exactly one granule");
        assert_eq!(at.fragment_bytes, 2 * MIB);
        assert_eq!(at.shareable_granules_per_fragment, 1);
        assert_eq!(
            at.wasted_boundary_bytes_per_sequence, 0,
            "an exact-granule fragment wastes nothing at the boundary"
        );
    }

    #[test]
    fn small_model_needs_token_major_or_finer_granule_at_2mib() {
        // qwen2.5-0.5b at 2 MiB: even seq-major fragments (512 KB) are under a
        // granule, so only token-major shares; the arithmetic, not a layout
        // preference, says so.
        let g = qwen05b();
        let prefix = 2_000;

        assert!(
            !evaluate_prefix_shareability(KvFragmentation::head_major_bnsh(g), prefix, 2 * MIB)
                .shareable
        );
        assert!(
            !evaluate_prefix_shareability(KvFragmentation::seq_major_bsnh(g), prefix, 2 * MIB)
                .shareable
        );
        let tm = evaluate_prefix_shareability(KvFragmentation::token_major(g), prefix, 2 * MIB);
        assert!(tm.shareable);
        assert_eq!(tm.shareable_granules_per_fragment, 11); // floor(24_576_000 / 2_097_152)
    }

    #[test]
    fn cpu_mmap_4kib_granule_shares_essentially_any_prefix_in_any_layout() {
        // mmap granularity is 4 KiB, so fragment_bytes >= granule holds for
        // essentially any prefix in any layout — prefix sharing is
        // straightforwardly universal on the CPU backend.
        let g = qwen14b();
        let prefix = 64; // a short prefix
        for frag in [
            KvFragmentation::head_major_bnsh(g),
            KvFragmentation::seq_major_bsnh(g),
            KvFragmentation::token_major(g),
        ] {
            let d = evaluate_prefix_shareability(frag, prefix, 4 * KIB);
            assert!(d.shareable, "4 KiB granule shares even a 64-token prefix");
        }
    }

    #[test]
    fn boundary_waste_is_the_remainder_and_whole_fragment_when_unshareable() {
        let g = qwen14b();
        // Unshareable: the whole fragment is wasted, per fragment, per sequence.
        let hm = evaluate_prefix_shareability(KvFragmentation::head_major_bnsh(g), 2_000, 2 * MIB);
        assert_eq!(
            hm.wasted_boundary_bytes_per_sequence,
            768 * 512_000,
            "when nothing shares, every prefix byte is boundary-private"
        );
        // Shareable with a remainder: only the straddling remainder is wasted.
        let tm = evaluate_prefix_shareability(KvFragmentation::token_major(g), 2_000, 2 * MIB);
        let remainder = 393_216_000u64 - 187 * (2 * MIB);
        assert_eq!(tm.wasted_boundary_bytes_per_sequence, remainder);
    }

    #[test]
    fn a_zero_granule_is_refused_not_divided_by() {
        let g = qwen14b();
        let d = evaluate_prefix_shareability(KvFragmentation::token_major(g), 2_000, 0);
        assert!(!d.shareable);
        assert_eq!(d.shareable_granules_per_fragment, 0);
        assert!(d.refusal_reason().is_some());
    }
}
