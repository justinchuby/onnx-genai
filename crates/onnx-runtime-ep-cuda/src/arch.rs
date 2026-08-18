//! SM-version kernel-dispatch scaffolding.
//!
//! This module is the single insertion point for selecting kernel variants and
//! tiling by CUDA compute capability at runtime. It exists so the pending
//! RTX/consumer-GPU kernels (device-property tiling, split-K by SM count, Ada
//! L2-residency, shared `cp.async` staging) have a clean, arch-guarded seam to
//! plug into the moment that hardware lands — we currently develop on H200
//! (`sm_90`) only, so this is *scaffolding + correctness*, not live tuning.
//!
//! # Portability / no-regression contract
//!
//! Per the standing directive "rtx显卡也要优化", every performance path must
//! help consumer/edge RTX cards (Ada `sm_89` RTX 40, Ampere `sm_86` RTX 30,
//! Blackwell `sm_120` RTX 50), not just the datacenter parts. Two rules keep
//! that safe:
//!
//! 1. **Totality.** [`ArchTier::from_compute_capability`] maps *every* plausible
//!    compute capability to a tier without panicking, so an unseen future GPU
//!    can never crash dispatch — it falls back to the nearest known tier.
//! 2. **`sm_90` is frozen.** The [`ArchConfig`] returned for Hopper mirrors the
//!    values today's hardcoded selectors already use on H200
//!    (`qmoe_gemm::tile_for` → 8, `matmul_nbits` resident-warps → 64,
//!    tensor-core eligible). Nothing in the live kernel-selection path reads
//!    this module yet, so it *cannot* change current behavior; when a future
//!    kernel routes through here, the Hopper row guarantees byte-identical
//!    selection on our dev hardware.

use crate::runtime::CudaDeviceCapabilities;

/// Coarse architecture family a device belongs to. Kernel variant choices and
/// default tiling are keyed off this rather than raw `(major, minor)` so the
/// pending RTX kernels can express "Ada wants X" without re-deriving the family
/// at every call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArchTier {
    /// Pre-Volta and anything below `sm_70`. Portable CUDA-core paths only.
    Legacy,
    /// `sm_70`/`sm_72` — Volta (V100 / Xavier).
    Volta,
    /// `sm_75` — Turing (RTX 20 / T4).
    Turing,
    /// `sm_80`/`sm_86`/`sm_87` — Ampere (A100 datacenter, RTX 30 consumer).
    Ampere,
    /// `sm_89` — Ada Lovelace (RTX 40 / L4 / L40).
    Ada,
    /// `sm_90` — Hopper (H100 / H200). This is our current dev hardware.
    Hopper,
    /// `sm_100`+ / `sm_120` — Blackwell (B100/B200 datacenter, RTX 50 consumer).
    Blackwell,
}

impl ArchTier {
    /// Map a compute capability to its architecture tier. **Total**: never
    /// panics, and any capability newer than the ones enumerated here resolves
    /// to the newest known tier (so a future consumer part is treated like the
    /// closest thing we understand rather than crashing dispatch).
    #[must_use]
    pub fn from_compute_capability((major, minor): (u32, u32)) -> Self {
        match (major, minor) {
            (0..=6, _) => ArchTier::Legacy,
            (7, 0..=2) => ArchTier::Volta,
            (7, _) => ArchTier::Turing, // sm_75 and any other sm_7x
            (8, 0..=7) => ArchTier::Ampere,
            (8, _) => ArchTier::Ada, // sm_89 (and any later sm_8x)
            (9, _) => ArchTier::Hopper,
            // sm_100 (Blackwell datacenter) and sm_120 (RTX 50) both land here,
            // as does any unseen future major — newest known tier, no panic.
            (_, _) => ArchTier::Blackwell,
        }
    }

    /// `true` for tiers with the `mma.sync`/`cp.async` tensor-core machinery
    /// (SM80+). Mirrors `marlin_gemm::MARLIN_MIN_SM` so a future dispatch that
    /// routes tensor-core eligibility through the tier table stays consistent
    /// with the existing `device_supports_marlin` gate.
    #[must_use]
    pub fn has_tensor_cores(self) -> bool {
        matches!(
            self,
            ArchTier::Ampere | ArchTier::Ada | ArchTier::Hopper | ArchTier::Blackwell
        )
    }
}

/// Default, tier-derived kernel-configuration hints.
///
/// Every field is a **hint** for the pending RTX/arch kernels, not a wired-in
/// selector. The values are seeded from the choices today's hardcoded selectors
/// already make, so routing a kernel through this table later is a refactor, not
/// a behavior change (crucially on Hopper — see the module contract).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArchConfig {
    /// The tier these hints were derived from.
    pub tier: ArchTier,
    /// Preferred QMoE grouped-GEMM tile-M. Mirrors the `preferred` ladder in
    /// [`crate::kernels`] `qmoe_gemm::tile_for` (SM80+ → 8, SM70/75 → 4, else 2)
    /// before the shared-memory clamp is applied.
    ///
    // RTX/arch: device-property tiling (RTX-TILING) — the pending kernel should
    // clamp this against `CudaDeviceCapabilities::max_shared_memory_per_block*`
    // exactly like `qmoe_gemm::tile_for` does today, then plug the result in
    // here instead of re-deriving `preferred`.
    pub qmoe_tile_hint: u32,
    /// Resident warps per SM used for one-wave occupancy math. Mirrors the
    /// `resident_warps` ladder in `matmul_nbits` (`sm_80`/`sm_90+` datacenter →
    /// 64, consumer `sm_86`/`sm_89` → 48).
    ///
    // RTX/arch: split-K by SM count (RTX-SPLITK) — combine this with
    // `CudaDeviceCapabilities::multiprocessor_count()` to size the split-K
    // degree so consumer parts (fewer SMs) fill the grid without oversubscribing.
    pub resident_warps_per_sm: u32,
    /// Whether this tier is eligible for the tensor-core (Marlin/`mma.sync`)
    /// path. Mirrors `marlin_gemm::device_supports_marlin`.
    pub prefers_tensor_core: bool,
    /// Default dynamic shared-memory budget hint, in bytes. Conservative
    /// (non-opt-in) 48 KB ceiling that holds on every tier; a real launch should
    /// still clamp against the device's opt-in ceiling.
    ///
    // RTX/arch: shared cp.async staging (RTX-CPASYNC) — a future async-copy
    // path should raise this toward
    // `CudaDeviceCapabilities::max_shared_memory_per_block_optin()` per tier.
    pub smem_budget_bytes: u32,
    /// Whether the pending Ada L2-residency lever applies to this tier. Ada's
    /// large L2 (`sm_89`) is the primary target; other tiers default off.
    ///
    // RTX/arch: Ada L2-residency (RTX-L2RES) — gate the persisting-L2 window on
    // this flag and size it from `CudaDeviceCapabilities::l2_cache_size()`.
    pub l2_residency_candidate: bool,
}

/// Conservative dynamic shared-memory budget that holds on every architecture
/// (matches `runtime::SAFE_SHARED_MEMORY_PER_BLOCK_FALLBACK`).
const DEFAULT_SMEM_BUDGET_BYTES: u32 = 48 * 1024;

impl ArchConfig {
    /// Default hint set for a tier, independent of any specific device's probed
    /// limits. Used by [`ArchConfig::for_capabilities`] and directly by tests
    /// that simulate a tier without real hardware.
    #[must_use]
    pub fn for_tier(tier: ArchTier) -> Self {
        // `qmoe_tile_hint` reproduces `qmoe_gemm::tile_for`'s `preferred` ladder;
        // `resident_warps_per_sm` reproduces `matmul_nbits`'s ladder.
        let (qmoe_tile_hint, resident_warps_per_sm) = match tier {
            ArchTier::Legacy => (2, 48),
            ArchTier::Volta => (4, 64), // sm_70 datacenter → 64 resident warps
            ArchTier::Turing => (4, 48),
            ArchTier::Ampere => (8, 64), // sm_80 datacenter → 64 (sm_86 consumer clamps elsewhere)
            ArchTier::Ada => (8, 48),    // sm_89 consumer part → 48
            ArchTier::Hopper => (8, 64), // FROZEN: must match today's H200 selection
            ArchTier::Blackwell => (8, 64),
        };
        Self {
            tier,
            qmoe_tile_hint,
            resident_warps_per_sm,
            prefers_tensor_core: tier.has_tensor_cores(),
            smem_budget_bytes: DEFAULT_SMEM_BUDGET_BYTES,
            l2_residency_candidate: matches!(tier, ArchTier::Ada),
        }
    }

    /// Default hint set for a probed device. Currently derived purely from the
    /// tier; the probed `CudaDeviceCapabilities` (SM count, L2 size, opt-in smem)
    /// are the levers the pending RTX kernels will fold in at the insertion
    /// points above.
    #[must_use]
    pub fn for_capabilities(capabilities: CudaDeviceCapabilities) -> Self {
        Self::for_tier(capabilities.arch_tier())
    }
}

/// Per-SM resident-warp estimate used for the one-wave occupancy math that
/// drives the int4/accuracy_level=4 decode GEMV's tiling and split-K grid-fill.
///
/// This reproduces — **byte-for-byte** — the ladder the decode selectors in
/// `kernels::matmul_nbits` have used on H200 to date: the datacenter parts
/// `sm_80` (A100) and `sm_90`+ (Hopper/Blackwell datacenter) expose 64 warps/SM,
/// while consumer/edge parts (`sm_86`/`sm_87` Ampere, `sm_89` Ada, Turing and
/// older) are treated as 48. Centralizing it here is what lets a live selector
/// consume the arch layer without perturbing the frozen `sm_90` selection: on
/// Hopper this returns 64, exactly as the previous inline `match` did.
///
// RTX/arch: split-K by SM count (RTX-SPLITK, todo rtx-devprop-tiling) — this is
// the resident-warp input to the one-wave CTA target. A future RTX tuning pass
// adjusts the consumer (48) rungs here so lower-SM Ada/Ampere parts size their
// own split-K degree, WITHOUT touching the `sm_90` (64) rung that H200 depends
// on. Keep the `(8, 0) | (9.., _) => 64` arm frozen.
#[must_use]
pub fn decode_resident_warps_per_sm((major, minor): (u32, u32)) -> u32 {
    match (major, minor) {
        (8, 0) | (9.., _) => 64,
        _ => 48,
    }
}

/// Device-property-driven tiling/split-K profile for the int4 decode GEMV.
///
/// This is the single, unit-testable surface the pending `rtx-devprop-tiling`
/// kernels select through. It folds together the probed device properties
/// ([`CudaDeviceCapabilities`]) that the grid-fill heuristics need — the arch
/// tier, the per-SM resident-warp estimate, and the multiprocessor count — so a
/// selector can size its split-K degree from the SM count without re-deriving
/// the arch family at every call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeTilingProfile {
    /// Arch tier the profile was derived from.
    pub tier: ArchTier,
    /// Multiprocessor (SM) count of the probed device — the split-K grid-fill
    /// lever. Consumer/edge RTX parts have far fewer SMs than H200's 132, so a
    /// fixed launch that fills H200 leaves them either grid-starved or
    /// over-subscribed; this is the number a split-K selector divides against.
    pub multiprocessor_count: u32,
    /// Per-SM resident-warp estimate for one-wave occupancy math (see
    /// [`decode_resident_warps_per_sm`]). Byte-identical to today's ladder.
    pub resident_warps_per_sm: u32,
    /// Whether this tier opts into the **new** SM-count-driven split-K tuning
    /// lever. `sm_90`/Hopper is **frozen** to today's hardcoded selection and is
    /// deliberately excluded (`false`) so H200 behavior cannot change; the
    /// lower-SM consumer/edge tiers (Ada, Ampere, Turing, Legacy) opt in
    /// (`true`) so the pending kernels can tune split-K for them. Note this
    /// gates only the *future RTX tuning*; the existing multiprocessor-count
    /// grid-fill in `matmul_nbits` already runs on every tier and is unchanged.
    pub sm_count_split_k: bool,
}

impl DecodeTilingProfile {
    /// Build the decode tiling profile for a probed device.
    #[must_use]
    pub fn for_capabilities(capabilities: CudaDeviceCapabilities) -> Self {
        let tier = capabilities.arch_tier();
        Self {
            tier,
            multiprocessor_count: capabilities.multiprocessor_count(),
            resident_warps_per_sm: decode_resident_warps_per_sm(capabilities.compute_capability()),
            // Frozen on Hopper (H200) by construction; every other tier is a
            // candidate for the RTX split-K tuning lever.
            sm_count_split_k: !matches!(tier, ArchTier::Hopper),
        }
    }

    /// One-wave resident-CTA target for a launch of `threads_per_cta` threads,
    /// i.e. `SM_count * (resident_warps_per_sm / warps_per_cta)`. This is the
    /// arch-aware occupancy denominator the grid-fill/split-K selectors compare
    /// their CTA count against; sizing it from the probed SM count is what makes
    /// the split-K degree track the device instead of a hardcoded GPU.
    #[must_use]
    pub fn one_wave_ctas(self, threads_per_cta: u32) -> usize {
        let warps_per_cta = (threads_per_cta / 32).max(1) as usize;
        let resident_ctas = (self.resident_warps_per_sm as usize / warps_per_cta).max(1);
        (self.multiprocessor_count.max(1) as usize).saturating_mul(resident_ctas)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::CudaDeviceCapabilities;

    /// Every compute capability we could plausibly meet resolves to a tier
    /// without panicking, including messy/未来 values. Guards the "totality"
    /// half of the portability contract.
    #[test]
    fn arch_tier_mapping_is_total_and_panic_free() {
        for major in 0u32..=20 {
            for minor in 0u32..=16 {
                // Must not panic and must return one of the known tiers.
                let tier = ArchTier::from_compute_capability((major, minor));
                let _ = ArchConfig::for_tier(tier);
            }
        }
    }

    /// Spot-check the family mapping across the real parts we care about,
    /// including the consumer RTX targets called out in the standing directive.
    #[test]
    fn known_compute_capabilities_map_to_expected_tiers() {
        let cases = [
            ((6, 1), ArchTier::Legacy),     // Pascal GTX 10
            ((7, 0), ArchTier::Volta),      // V100
            ((7, 5), ArchTier::Turing),     // RTX 20 / T4
            ((8, 0), ArchTier::Ampere),     // A100
            ((8, 6), ArchTier::Ampere),     // RTX 30
            ((8, 9), ArchTier::Ada),        // RTX 40
            ((9, 0), ArchTier::Hopper),     // H100 / H200 (dev hardware)
            ((10, 0), ArchTier::Blackwell), // B100/B200
            ((12, 0), ArchTier::Blackwell), // RTX 50
        ];
        for (cc, expected) in cases {
            assert_eq!(
                ArchTier::from_compute_capability(cc),
                expected,
                "cc {cc:?} mapped to the wrong tier"
            );
        }
    }

    /// HARD no-change guard: `sm_90` must resolve to Hopper and produce the
    /// config that reproduces today's H200 kernel selection (tile 8, 64 resident
    /// warps, tensor-core eligible). If a future edit perturbs this row, this
    /// test fails and the reviewer knows the sm_90 freeze was broken.
    #[test]
    fn sm_90_hopper_config_is_frozen() {
        let cfg = ArchConfig::for_tier(ArchTier::Hopper);
        assert_eq!(cfg.tier, ArchTier::Hopper);
        assert_eq!(cfg.qmoe_tile_hint, 8, "sm_90 QMoE tile must stay 8");
        assert_eq!(
            cfg.resident_warps_per_sm, 64,
            "sm_90 resident warps must stay 64"
        );
        assert!(cfg.prefers_tensor_core, "sm_90 stays tensor-core eligible");
        assert_eq!(cfg.smem_budget_bytes, DEFAULT_SMEM_BUDGET_BYTES);
        assert!(
            !cfg.l2_residency_candidate,
            "L2 residency is an Ada lever, not a Hopper one"
        );

        // And the same values when reached through a synthetic sm_90 device,
        // proving the capabilities → tier → config chain is consistent.
        let caps = CudaDeviceCapabilities::for_test((9, 0), 132, 50 * 1024 * 1024);
        assert_eq!(caps.arch_tier(), ArchTier::Hopper);
        assert_eq!(caps.arch_config(), cfg);
    }

    /// Ada (`sm_89`, RTX 40) is the L2-residency candidate and a consumer part
    /// (48 resident warps), distinct from datacenter Hopper. Exercises a
    /// non-Hopper tier via a synthetic device — no real RTX hardware needed.
    #[test]
    fn ada_consumer_config_differs_from_hopper() {
        let ada = ArchConfig::for_tier(ArchTier::Ada);
        assert_eq!(ada.resident_warps_per_sm, 48);
        assert!(ada.l2_residency_candidate);
        assert!(ada.prefers_tensor_core);

        let hopper = ArchConfig::for_tier(ArchTier::Hopper);
        assert_ne!(ada, hopper);
    }

    /// Legacy/pre-SM80 parts must not advertise tensor cores (keeps the pending
    /// Marlin/mma path off hardware that lacks `mma.sync`).
    #[test]
    fn pre_sm80_tiers_have_no_tensor_cores() {
        for tier in [ArchTier::Legacy, ArchTier::Volta, ArchTier::Turing] {
            assert!(!ArchConfig::for_tier(tier).prefers_tensor_core);
        }
        for tier in [
            ArchTier::Ampere,
            ArchTier::Ada,
            ArchTier::Hopper,
            ArchTier::Blackwell,
        ] {
            assert!(ArchConfig::for_tier(tier).prefers_tensor_core);
        }
    }

    /// The arch-aware resident-warp ladder must reproduce, byte-for-byte, the
    /// inline `match` that `matmul_nbits::use_accuracy4_stage64` used before it
    /// routed through this module: `(8, 0) | (9.., _) => 64`, everything else
    /// `48`. This is the guard that keeps the decode-GEMV occupancy math (and
    /// therefore H200's selection) unchanged when a selector consumes the arch
    /// layer.
    #[test]
    fn decode_resident_warps_ladder_matches_frozen_selection() {
        // Datacenter parts: 64 warps/SM.
        for cc in [(8, 0), (9, 0), (10, 0), (12, 0)] {
            assert_eq!(
                decode_resident_warps_per_sm(cc),
                64,
                "cc {cc:?} must stay on the 64-warp rung"
            );
        }
        // Consumer/edge parts (the RTX targets) + older: 48 warps/SM.
        for cc in [(8, 6), (8, 7), (8, 9), (7, 5), (7, 0), (6, 1)] {
            assert_eq!(
                decode_resident_warps_per_sm(cc),
                48,
                "cc {cc:?} must stay on the 48-warp rung"
            );
        }
    }

    /// HARD no-change guard for the decode tiling profile: a synthetic H200
    /// (`sm_90`, 132 SMs) must map to Hopper, keep the 64-warp rung, and be
    /// **excluded** from the new SM-count split-K tuning lever — i.e. the frozen
    /// path. If a future edit lets Hopper opt into RTX tuning, this fails.
    #[test]
    fn sm_90_decode_profile_is_frozen_out_of_rtx_splitk() {
        let h200 = CudaDeviceCapabilities::for_test((9, 0), 132, 50 * 1024 * 1024);
        let profile = DecodeTilingProfile::for_capabilities(h200);
        assert_eq!(profile.tier, ArchTier::Hopper);
        assert_eq!(profile.resident_warps_per_sm, 64);
        assert_eq!(profile.multiprocessor_count, 132);
        assert!(
            !profile.sm_count_split_k,
            "sm_90/H200 must NOT opt into the RTX SM-count split-K lever (frozen)"
        );
    }

    /// RTX path (no hardware): simulated consumer parts must opt INTO the
    /// SM-count-driven split-K tiling, using their lower SM counts and the
    /// 48-warp consumer rung — while `sm_90` stays frozen. Exercised purely via
    /// synthetic `for_test` capabilities.
    #[test]
    fn rtx_consumer_profiles_opt_into_sm_count_split_k() {
        // Simulated RTX 4090-class Ada: sm_89, 128 SMs.
        let ada = DecodeTilingProfile::for_capabilities(CudaDeviceCapabilities::for_test(
            (8, 9),
            128,
            72 * 1024 * 1024,
        ));
        assert_eq!(ada.tier, ArchTier::Ada);
        assert_eq!(ada.resident_warps_per_sm, 48);
        assert!(
            ada.sm_count_split_k,
            "Ada consumer opts into SM-count split-K"
        );

        // Simulated L4-class Ada with far fewer SMs (58) — split-K must track
        // the SM count, so a fixed 256-thread launch reaches one wave sooner.
        let ada_l4 = DecodeTilingProfile::for_capabilities(CudaDeviceCapabilities::for_test(
            (8, 9),
            58,
            48 * 1024 * 1024,
        ));
        assert!(ada_l4.sm_count_split_k);
        assert!(
            ada_l4.one_wave_ctas(256) < ada.one_wave_ctas(256),
            "fewer SMs => smaller one-wave CTA target (split-K fills the grid sooner)"
        );

        // Simulated RTX 3080-class Ampere consumer: sm_86, 68 SMs.
        let ampere = DecodeTilingProfile::for_capabilities(CudaDeviceCapabilities::for_test(
            (8, 6),
            68,
            5 * 1024 * 1024,
        ));
        assert_eq!(ampere.tier, ArchTier::Ampere);
        assert_eq!(ampere.resident_warps_per_sm, 48);
        assert!(
            ampere.sm_count_split_k,
            "Ampere consumer opts into SM-count split-K"
        );

        // sm_90 remains frozen out of the lever alongside them.
        let h200 = DecodeTilingProfile::for_capabilities(CudaDeviceCapabilities::for_test(
            (9, 0),
            132,
            50 * 1024 * 1024,
        ));
        assert!(!h200.sm_count_split_k);
    }

    /// The one-wave CTA target must scale with the probed SM count (the split-K
    /// lever) and with the CTA width, so the split-K degree tracks the device.
    #[test]
    fn one_wave_ctas_tracks_sm_count_and_cta_width() {
        let profile = DecodeTilingProfile::for_capabilities(CudaDeviceCapabilities::for_test(
            (8, 9),
            100,
            48 * 1024 * 1024,
        ));
        // 48 warps/SM, 256-thread (8-warp) CTA => 6 resident CTAs/SM * 100 SMs.
        assert_eq!(profile.one_wave_ctas(256), 600);
        // A narrower single-warp CTA fits 48 resident CTAs/SM.
        assert_eq!(profile.one_wave_ctas(32), 4800);
        // Degenerate device is clamped to at least one SM / one resident CTA.
        let tiny =
            DecodeTilingProfile::for_capabilities(CudaDeviceCapabilities::for_test((8, 9), 0, 0));
        assert_eq!(tiny.one_wave_ctas(256), 6);
    }
}
