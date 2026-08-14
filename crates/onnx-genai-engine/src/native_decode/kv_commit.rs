//! Layout-aware KV-cache commit geometry.
//!
//! `cuMemMap` maps whole granule-aligned windows of virtual address space onto
//! whole physical granules; partial mapping does not exist. So the *committed*
//! physical bytes of a KV binding are `granule × (windows touched by a live
//! byte)`, and which windows the live prefix touches is decided by the tensor
//! layout, not by the allocator (see `docs/MEMORY_ARCHITECTURE.md`, "KV layout
//! and residency").
//!
//! Before this module the native CUDA binding always committed a single flat
//! range `0..(capacity × kv_heads × head_dim × elem)` — the packed bucket —
//! regardless of layout. That made head-major and seq-major commit **identical**
//! physical bytes (measured in #794), because a flat range from offset 0 maps
//! the same contiguous granules whatever the axis order means. This module makes
//! the committed byte ranges follow the layout descriptor instead:
//!
//! * **Seq-major BSNH** `[batch, seq, kv_heads, head_dim]`: a token's whole KV
//!   (all heads) is contiguous, so the live prefix of `valid_len` tokens is one
//!   dense run `0..(valid_len × kv_heads × head_dim × elem)`. Committing that run
//!   under a *fixed full-context stride* touches `ceil(live_bytes / granule)`
//!   granules — the `layers × 2` floor — and the stride never changes as the
//!   sequence grows, so a captured graph is not invalidated.
//! * **Head-major BNSH** `[batch, kv_heads, seq, head_dim]`: each head owns its
//!   own `capacity × head_dim` stripe, so the live prefix scatters into one
//!   fragment per head. Under a fixed full-context stride the floor is therefore
//!   `kv_heads` windows per binding (`layers × 2 × kv_heads` across the model) —
//!   the `kv_heads×` penalty this whole design line is about. The engine keeps
//!   head-major on its historical *bucketed* single flat range (which tracks the
//!   sequence and re-captures on growth); the per-stripe floor form here is what
//!   the residency measurement compares seq-major against on the same
//!   stable-VA / fixed-stride regime.

// The geometry helpers below (`live_prefix_ranges`, `committed_granules`,
// `live_prefix_committed_bytes`, and `KvBindingGeometry`) are the layout-aware
// commit mechanism. They are exercised by this module's unit tests and mirrored
// by the driver-level GPU residency measurement in `onnx-runtime-cuda-memory`
// (`vmm_kv_layout_residency_gpu`). The engine's own seq-major fixed-stride
// commit path now consumes `live_prefix_ranges` directly
// (`DecodeCudaState::seq_major_kv_commit_requests`), so the live commit geometry
// and the measured residency floor are single-sourced here and cannot drift. The
// seq-major (BSNH) fixed-stride physical-shape build that hangs the dense-prefix
// commit on this geometry landed with #801/#812 (see
// `docs/MEMORY_ARCHITECTURE.md`, "KV layout and residency"). The residency
// projection helpers `committed_granules` / `live_prefix_committed_bytes` remain
// measurement-only (they model the granule floor the driver test verifies), so
// they carry a scoped dead-code allowance for non-test builds.

use std::ops::Range;

/// The physical KV-cache layout a binding is stored in. Absent metadata means
/// [`KvCommitLayout::HeadMajor`], preserving the historical behavior exactly.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum KvCommitLayout {
    /// Head-major BNSH `[batch, kv_heads, seq, head_dim]`. Live prefix scatters
    /// one fragment per head stripe.
    #[default]
    HeadMajor,
    /// Seq-major BSNH `[batch, seq, kv_heads, head_dim]`. Live prefix is one
    /// dense contiguous run across all heads.
    SeqMajor,
}

impl KvCommitLayout {
    pub(crate) fn is_seq_major(self) -> bool {
        matches!(self, KvCommitLayout::SeqMajor)
    }
}

/// Per-token geometry of a single KV binding (one `(layer, side)` key or value
/// buffer), in elements/bytes. `kv_heads × head_dim × elem_bytes` is the number
/// of physical bytes one token of this binding occupies, identical for both
/// layouts; only *where* those bytes land differs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct KvBindingGeometry {
    pub kv_heads: usize,
    pub head_dim: usize,
    pub elem_bytes: usize,
}

impl KvBindingGeometry {
    /// Physical bytes one token occupies in this binding.
    pub(crate) fn bytes_per_token(self) -> Option<usize> {
        self.kv_heads
            .checked_mul(self.head_dim)?
            .checked_mul(self.elem_bytes)
    }

    /// Bytes one head's per-token slice occupies (`head_dim × elem_bytes`). This
    /// is the head-major head-stripe row stride's live width per token.
    fn bytes_per_token_per_head(self) -> Option<usize> {
        self.head_dim.checked_mul(self.elem_bytes)
    }
}

/// The byte ranges (within one binding's allocation) that must be committed to
/// make the live prefix of `valid_len` tokens physically resident for each of
/// `batch` sequences, given the layout and a fixed full-context `capacity` (the
/// grow-axis stride).
///
/// * Seq-major returns one dense range per sequence, sequence `b` covering
///   `[b × capacity × bytes_per_token .. + valid_len × bytes_per_token]`. The
///   per-sequence stride is the fixed full-context `capacity`, so a sequence's
///   bytes never move as the *committed* prefix grows.
/// * Head-major returns one range per `(sequence, head)` stripe, stripe
///   `(b, h)` covering `[(b × kv_heads + h) × capacity × head_dim × elem .. +
///   valid_len × head_dim × elem]`. This is the per-stripe floor of a fixed
///   full-context stride.
///
/// **Batch generality (stage 2b-impl-2, #750):** the batch axis is the
/// *outermost* axis of both layouts, so a sequence's fragments are a fixed
/// `per_sequence_stride = capacity × bytes_per_token` apart and computing them
/// needs no relocation — this is why the *fixed full-context stride* commit path
/// is batch-general. It is only a *growing bucket* (the realloc / VMM
/// growing-bucket path, not this fixed-stride commit) whose seq-major
/// per-sequence stride depends on the mutable bucket capacity and would relocate
/// every sequence `b > 0` on growth; that case is refused explicitly in
/// [`super::cuda::kv_growth_byte_layout`]. At `batch == 1` this function is
/// byte-identical to the historical single-sequence form.
///
/// Returns `None` on arithmetic overflow. An empty vec means nothing to commit
/// (`valid_len == 0`).
pub(crate) fn live_prefix_ranges(
    layout: KvCommitLayout,
    geometry: KvBindingGeometry,
    batch: usize,
    capacity: usize,
    valid_len: usize,
) -> Option<Vec<Range<usize>>> {
    if valid_len == 0 || batch == 0 {
        return Some(Vec::new());
    }
    let valid_len = valid_len.min(capacity);
    match layout {
        KvCommitLayout::SeqMajor => {
            let bytes_per_token = geometry.bytes_per_token()?;
            let live_width = valid_len.checked_mul(bytes_per_token)?;
            let per_sequence_stride = capacity.checked_mul(bytes_per_token)?;
            let mut ranges = Vec::with_capacity(batch);
            for sequence in 0..batch {
                let start = sequence.checked_mul(per_sequence_stride)?;
                let end = start.checked_add(live_width)?;
                ranges.push(start..end);
            }
            Some(ranges)
        }
        KvCommitLayout::HeadMajor => {
            let head_stride = capacity.checked_mul(geometry.bytes_per_token_per_head()?)?;
            let per_sequence_stride = geometry.kv_heads.checked_mul(head_stride)?;
            let live_width = valid_len.checked_mul(geometry.bytes_per_token_per_head()?)?;
            let mut ranges = Vec::with_capacity(batch.checked_mul(geometry.kv_heads)?);
            for sequence in 0..batch {
                let sequence_base = sequence.checked_mul(per_sequence_stride)?;
                for head in 0..geometry.kv_heads {
                    let start = sequence_base.checked_add(head.checked_mul(head_stride)?)?;
                    let end = start.checked_add(live_width)?;
                    ranges.push(start..end);
                }
            }
            Some(ranges)
        }
    }
}

/// The number of physical granules the given byte ranges touch, i.e. the
/// committed physical bytes divided by the granule size. Two ranges that share a
/// granule window count that window once (this is why seq-major's dense run is
/// so much cheaper than head-major's scattered fragments at the same content).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn committed_granules(ranges: &[Range<usize>], granule: usize) -> usize {
    assert!(granule > 0, "granule size must be non-zero");
    let mut windows: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|range| range.end > range.start)
        .map(|range| {
            let first = range.start / granule;
            let last = (range.end - 1) / granule;
            (first, last)
        })
        .collect();
    if windows.is_empty() {
        return 0;
    }
    windows.sort_unstable();
    let mut total = 0usize;
    let mut cursor: Option<usize> = None;
    for (first, last) in windows {
        let start = match cursor {
            Some(prev_last) if first <= prev_last => prev_last + 1,
            _ => first,
        };
        if last >= start {
            total += last - start + 1;
        }
        cursor = Some(last.max(cursor.unwrap_or(last)));
    }
    total
}

/// Committed physical bytes for the live prefix of one binding under `layout`,
/// on a fixed full-context stride, rounded up to the granule. `None` on
/// overflow.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn live_prefix_committed_bytes(
    layout: KvCommitLayout,
    geometry: KvBindingGeometry,
    batch: usize,
    capacity: usize,
    valid_len: usize,
    granule: usize,
) -> Option<usize> {
    let ranges = live_prefix_ranges(layout, geometry, batch, capacity, valid_len)?;
    committed_granules(&ranges, granule).checked_mul(granule)
}

#[cfg(test)]
mod tests {
    use super::*;

    // qwen14b KV geometry per binding: 8 kv heads, head_dim 128, fp16.
    const QWEN14B: KvBindingGeometry = KvBindingGeometry {
        kv_heads: 8,
        head_dim: 128,
        elem_bytes: 2,
    };
    // qwen2.5-0.5b KV geometry per binding: 2 kv heads, head_dim 64, fp16.
    const QWEN05B: KvBindingGeometry = KvBindingGeometry {
        kv_heads: 2,
        head_dim: 64,
        elem_bytes: 2,
    };
    const GRANULE: usize = 2 * 1024 * 1024;

    #[test]
    fn seq_major_is_one_dense_run_from_zero() {
        let ranges = live_prefix_ranges(KvCommitLayout::SeqMajor, QWEN14B, 1, 32_768, 100).unwrap();
        assert_eq!(ranges, vec![0..(100 * 8 * 128 * 2)]);
    }

    #[test]
    fn head_major_is_one_fragment_per_head_stripe() {
        let capacity = 32_768;
        let valid = 100;
        let ranges =
            live_prefix_ranges(KvCommitLayout::HeadMajor, QWEN14B, 1, capacity, valid).unwrap();
        assert_eq!(ranges.len(), QWEN14B.kv_heads);
        let head_stride = capacity * QWEN14B.head_dim * QWEN14B.elem_bytes;
        let live_width = valid * QWEN14B.head_dim * QWEN14B.elem_bytes;
        for (head, range) in ranges.iter().enumerate() {
            assert_eq!(range.start, head * head_stride);
            assert_eq!(range.end, head * head_stride + live_width);
        }
    }

    // Batch>1 control (stage 2b-impl-2, #750): seq-major returns one dense run
    // per sequence, each a fixed `capacity × bytes_per_token` apart, so a
    // transposed axis or a capacity-dependent stride would show up here (it
    // could not at batch=1). No relocation: the per-sequence stride is the fixed
    // full-context `capacity`, independent of `valid_len`.
    #[test]
    fn seq_major_batch_n_is_one_dense_run_per_sequence() {
        let batch = 3;
        let capacity = 32_768;
        let valid = 100;
        let ranges =
            live_prefix_ranges(KvCommitLayout::SeqMajor, QWEN14B, batch, capacity, valid).unwrap();
        assert_eq!(ranges.len(), batch);
        let bpt = QWEN14B.kv_heads * QWEN14B.head_dim * QWEN14B.elem_bytes;
        let seq_stride = capacity * bpt;
        let live = valid * bpt;
        for (sequence, range) in ranges.iter().enumerate() {
            assert_eq!(range.start, sequence * seq_stride);
            assert_eq!(range.end, sequence * seq_stride + live);
        }
        // Sequence 0's fragment is byte-identical to the batch-1 result.
        let batch1 =
            live_prefix_ranges(KvCommitLayout::SeqMajor, QWEN14B, 1, capacity, valid).unwrap();
        assert_eq!(ranges[0], batch1[0]);
    }

    // Batch>1 control: head-major scatters `batch × kv_heads` fragments, batch
    // outermost, `(b, h)` at `(b × kv_heads + h) × head_stride`.
    #[test]
    fn head_major_batch_n_is_one_fragment_per_sequence_head() {
        let batch = 3;
        let capacity = 32_768;
        let valid = 100;
        let ranges =
            live_prefix_ranges(KvCommitLayout::HeadMajor, QWEN14B, batch, capacity, valid).unwrap();
        assert_eq!(ranges.len(), batch * QWEN14B.kv_heads);
        let head_stride = capacity * QWEN14B.head_dim * QWEN14B.elem_bytes;
        let live_width = valid * QWEN14B.head_dim * QWEN14B.elem_bytes;
        for (index, range) in ranges.iter().enumerate() {
            let start = index * head_stride; // dense (b*kv_heads + h) enumeration
            assert_eq!(range.start, start);
            assert_eq!(range.end, start + live_width);
        }
    }

    #[test]
    fn empty_prefix_commits_nothing() {
        assert!(
            live_prefix_ranges(KvCommitLayout::SeqMajor, QWEN14B, 1, 32_768, 0)
                .unwrap()
                .is_empty()
        );
        // Batch has no effect when there is nothing live to commit.
        assert!(
            live_prefix_ranges(KvCommitLayout::SeqMajor, QWEN14B, 4, 32_768, 0)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            live_prefix_committed_bytes(KvCommitLayout::HeadMajor, QWEN14B, 1, 32_768, 0, GRANULE)
                .unwrap(),
            0
        );
    }

    #[test]
    fn overlapping_windows_are_counted_once() {
        // Two ranges inside the same single granule → 1 granule, not 2.
        let ranges = vec![0..1024, 4096..8192];
        assert_eq!(committed_granules(&ranges, GRANULE), 1);
        // Two ranges straddling a shared boundary granule.
        let ranges = vec![0..(GRANULE + 10), GRANULE..(GRANULE + 20)];
        assert_eq!(committed_granules(&ranges, GRANULE), 2);
    }

    // The headline: on a fixed full-context stride, at the near-empty floor,
    // seq-major commits `kv_heads×` fewer physical bytes than head-major.
    #[test]
    fn near_empty_floor_is_kv_heads_times_smaller_seq_major_qwen14b() {
        let capacity = 32_768;
        let valid = 1; // one live token
        let head_major = live_prefix_committed_bytes(
            KvCommitLayout::HeadMajor,
            QWEN14B,
            1,
            capacity,
            valid,
            GRANULE,
        )
        .unwrap();
        let seq_major = live_prefix_committed_bytes(
            KvCommitLayout::SeqMajor,
            QWEN14B,
            1,
            capacity,
            valid,
            GRANULE,
        )
        .unwrap();
        // head-major: one granule per head → 8 granules; seq-major: one dense
        // run under a granule → 1 granule. 8× = kv_heads×.
        assert_eq!(head_major, 8 * GRANULE);
        assert_eq!(seq_major, GRANULE);
        assert_eq!(head_major / seq_major, QWEN14B.kv_heads);
    }

    #[test]
    fn near_empty_floor_is_kv_heads_times_smaller_seq_major_qwen05b() {
        let capacity = 32_768;
        let head_major = live_prefix_committed_bytes(
            KvCommitLayout::HeadMajor,
            QWEN05B,
            1,
            capacity,
            1,
            GRANULE,
        )
        .unwrap();
        let seq_major =
            live_prefix_committed_bytes(KvCommitLayout::SeqMajor, QWEN05B, 1, capacity, 1, GRANULE)
                .unwrap();
        assert_eq!(head_major / seq_major, QWEN05B.kv_heads); // 2×
    }

    // Above the seq-major crossover (granule / bytes_per_token) the dense run
    // fills whole granules and the layouts converge — the honest limit of the
    // win. qwen14b bytes/token = 2048, crossover ≈ 1024 tokens.
    #[test]
    fn above_crossover_the_layouts_converge() {
        let capacity = 32_768;
        let valid = 8192; // well above the ~1024-token crossover
        let head_major = live_prefix_committed_bytes(
            KvCommitLayout::HeadMajor,
            QWEN14B,
            1,
            capacity,
            valid,
            GRANULE,
        )
        .unwrap();
        let seq_major = live_prefix_committed_bytes(
            KvCommitLayout::SeqMajor,
            QWEN14B,
            1,
            capacity,
            valid,
            GRANULE,
        )
        .unwrap();
        // head-major per head: 8192×128×2 = 2 MiB = exactly 1 granule per head →
        // still 8 granules; seq-major: 8192×2048 = 16 MiB = 8 granules. Equal.
        assert_eq!(head_major, seq_major);
    }

    #[test]
    fn valid_len_is_clamped_to_capacity() {
        let ranges =
            live_prefix_ranges(KvCommitLayout::SeqMajor, QWEN14B, 1, 256, 100_000).unwrap();
        assert_eq!(ranges, vec![0..(256 * 8 * 128 * 2)]);
    }

    #[test]
    fn bytes_per_token_matches_both_layouts() {
        assert_eq!(QWEN14B.bytes_per_token().unwrap(), 8 * 128 * 2);
        assert_eq!(QWEN05B.bytes_per_token().unwrap(), 2 * 64 * 2);
    }
}
