//! Environment configuration for weight residency and opt-in diagnostics.
//!
//! Kept separate from page ownership and admission. Cached knobs retain their
//! individual lazy initialization; `pin_refill_every` remains dynamically read.
//! The safety budget remains platform-specific and independent of probe flags.

use std::collections::HashSet;
use std::sync::OnceLock;

/// Parse `ONNX_GENAI_ZERO_COPY_HYBRID`. Defaults **OFF** (opt-in):
/// only `1`/`true`/`yes`/`on` (case/whitespace-insensitive) enable it.
pub(super) fn zero_copy_hybrid_from_env_value(value: Option<&str>) -> bool {
    match value {
        Some(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => false,
    }
}

/// Diagnostic knob (#864): perform the deferred weight's real streaming copy
/// instead of binding its host-mapped device pointer. Isolates the deferral/
/// admission flow from the host-mapped READ. Default OFF.
pub(super) fn zero_copy_copy_instead() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        zero_copy_hybrid_from_env_value(
            std::env::var("ONNX_GENAI_ZERO_COPY_HYBRID_COPY_INSTEAD")
                .ok()
                .as_deref(),
        )
    })
}

/// Diagnostic knob (#864): print per-bind zero-copy diagnostics. Default OFF.
pub(super) fn zero_copy_debug() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        zero_copy_hybrid_from_env_value(
            std::env::var("ONNX_GENAI_ZERO_COPY_HYBRID_DEBUG")
                .ok()
                .as_deref(),
        )
    })
}

/// Diagnostic knob (#864): pre-fault every page of the mapping on the CPU before
/// `cuMemHostRegister`, so pinning sees populated pages rather than racing the
/// lazy mmap's demand paging. Default OFF.
pub(super) fn zero_copy_prefault() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        zero_copy_hybrid_from_env_value(
            std::env::var("ONNX_GENAI_ZERO_COPY_HYBRID_PREFAULT")
                .ok()
                .as_deref(),
        )
    })
}

/// Diagnostic knob (#864): register the mapping with DEVICEMAP only (drop
/// READ_ONLY), to isolate a READ_ONLY-flag correctness problem. Default OFF.
pub(super) fn zero_copy_no_readonly() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        zero_copy_hybrid_from_env_value(
            std::env::var("ONNX_GENAI_ZERO_COPY_HYBRID_NO_READONLY")
                .ok()
                .as_deref(),
        )
    })
}

/// Outcome of reading a numeric diagnostic knob.
///
/// The three states are kept apart deliberately. Collapsing `Invalid` into
/// `Unset` is what makes the silent-fallback shape dangerous: a sweep that
/// writes `..._BUDGET_BYTES=2GB` (or leaves a digit separator, or a stray
/// character) would fall back to the conservative default and then report "no
/// corruption at 2 GiB" having never tested 2 GiB. A confident wrong answer is
/// worse than an error. This is the same failure as rendering "not determined"
/// as "determined to be unsafe" (#931), and the same family as the
/// `ASYNC_PAGEIN` trap where an unrecognized value silently selected the slow
/// path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum NumericEnv {
    Unset,
    Invalid(String),
    Value(u64),
}

impl NumericEnv {
    /// The configured value, warning on stderr when one was supplied but not
    /// understood, so a measurement taken under it is not mistaken for a
    /// measurement of the value the operator believed they set.
    fn or_default(self, name: &str, default: u64) -> u64 {
        match self {
            NumericEnv::Value(value) => value,
            NumericEnv::Unset => default,
            NumericEnv::Invalid(raw) => {
                eprintln!(
                    "cuda_ep: {name}={raw:?} is not a base-10 byte count; using {default} instead. \
                     Set a plain integer (e.g. 1073741824 for 1 GiB) — this value did NOT take \
                     effect, so any measurement taken under it describes the default."
                );
                default
            }
        }
    }

    fn into_option(self, name: &str) -> Option<u64> {
        match self {
            NumericEnv::Value(value) => Some(value),
            NumericEnv::Unset => None,
            NumericEnv::Invalid(raw) => {
                eprintln!(
                    "cuda_ep: {name}={raw:?} is not a base-10 integer; ignoring it. This value did \
                     NOT take effect."
                );
                None
            }
        }
    }
}

/// Read a numeric diagnostic knob, keeping "unset" and "set but not understood"
/// distinct. See [`NumericEnv`] for why that distinction is load-bearing.
pub(super) fn parse_numeric_env(name: &str) -> NumericEnv {
    let Ok(raw) = std::env::var(name) else {
        return NumericEnv::Unset;
    };
    match raw.trim().parse::<u64>() {
        Ok(value) => NumericEnv::Value(value),
        Err(_) => NumericEnv::Invalid(raw),
    }
}

/// Diagnostic knob (#864): cap how many weights are actually bound zero-copy
/// (the rest are copied). `Some(1)` isolates a single zero-copy read. Default
/// `None` (unbounded).
pub(super) fn zero_copy_max_binds() -> Option<u64> {
    const NAME: &str = "ONNX_GENAI_ZERO_COPY_HYBRID_MAX_BINDS";
    static V: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *V.get_or_init(|| parse_numeric_env(NAME).into_option(NAME))
}

/// Windows/WDDM default safety budget for **distinct** zero-copy bytes bound per
/// residency (256 MiB). This conservative value exists because of a measured
/// hardware limit (#864): on an RTX 4060 Laptop under WDDM, device reads through
/// a `cuMemHostRegister(READ_ONLY | DEVICEMAP)` mapping are bit-identical up to
/// ~0.44 GB of distinct host-mapped data read per decode step (32 cold weights
/// verified correct), but **silently corrupt** above that (48 weights / ~0.65 GB
/// collapsed generation 16 → 3 tokens — the #886 signature, but from stale
/// host-mapped reads, not eviction). Individual reads are always correct, so
/// this is an aggregate host-mapped-aperture ceiling, not a per-read fault. It
/// is a **WDDM/VidMm memory-manager artifact** (same family as #863, where
/// VidMm silently demoted our own VMM granules to host RAM), so the conservative
/// value is kept only on Windows.
const ZERO_COPY_SAFE_BUDGET_BYTES_WDDM: u64 = 256 * 1024 * 1024;

/// Non-Windows (Linux / discrete-GPU) default safety budget: **2 GiB**. #925
/// re-measured the hybrid on Linux (H200, driver 580.105.08, CUDA 13, kernel
/// 6.6, native VMM decode path) and found the WDDM aperture ceiling **absent**:
/// generation stayed byte-identical to the Step-0 baseline with `cuda_graph`
/// `fallbacks=0` up to **6.795 GB** of distinct host-mapped weights bound and
/// re-read in place every decode step (704 `cuMemHostRegister` binds; n=3, all
/// runs byte-identical) — ~15× the WDDM ~0.44 GB onset and ~10× the top of its
/// 0.44–0.65 GB corruption band. On Linux `cuMemHostRegister(READ_ONLY |
/// DEVICEMAP)` pins pages in the driver with no VidMm layer above it, so the
/// ceiling that motivates the Windows value does not apply here; the 256 MiB
/// default left an ~8× decode win unused in the over-budget regime (67 vs
/// ~8.5 tok/s median vs managed streaming).
///
/// 2 GiB is chosen deliberately **bounded, not unbounded**: it sits >3× below
/// the #925 measured-safe 6.795 GB (a comfortable margin, since only one GPU
/// class — Hopper/H200 — was actually tested), yet clears the entire WDDM
/// corruption band by >3× and covers both the observed ~0.85 GB per-step cold
/// working set and the deferred fraction of a realistically over-budget model
/// with headroom — which is what unlocks the win. Keeping it bounded avoids
/// over-committing pinned host RAM and avoids extrapolating past tested hardware
/// (portability); operators can raise or lower it per run via
/// `ONNX_GENAI_ZERO_COPY_HYBRID_BUDGET_BYTES`.
const ZERO_COPY_SAFE_BUDGET_BYTES_NON_WINDOWS: u64 = 2 * 1024 * 1024 * 1024;

/// Platform-selected default distinct-zero-copy-byte budget. Windows keeps the
/// conservative WDDM value ([`ZERO_COPY_SAFE_BUDGET_BYTES_WDDM`], #864); every
/// other platform uses the higher Linux-measured value
/// ([`ZERO_COPY_SAFE_BUDGET_BYTES_NON_WINDOWS`], #925). The `if cfg!(...)` form
/// references both named consts unconditionally, so each stays greppable and
/// unit-testable regardless of the host that compiled this crate. The default
/// is set under the platform's observed-safe ceiling so the opt-in knob can
/// never violate the byte-identical gate; override for investigation only.
const ZERO_COPY_SAFE_BUDGET_BYTES: u64 = if cfg!(target_os = "windows") {
    ZERO_COPY_SAFE_BUDGET_BYTES_WDDM
} else {
    ZERO_COPY_SAFE_BUDGET_BYTES_NON_WINDOWS
};

/// Per-residency distinct zero-copy byte budget (see
/// [`ZERO_COPY_SAFE_BUDGET_BYTES`]). Override via
/// `ONNX_GENAI_ZERO_COPY_HYBRID_BUDGET_BYTES` (e.g. `0` to force copy-only, or a
/// large value to reproduce the corruption on other hardware).
///
/// The default is **platform-selected** (see [`ZERO_COPY_SAFE_BUDGET_BYTES`]):
/// 256 MiB on Windows (the WDDM-derived figure, #864/#912) and 2 GiB elsewhere
/// (Linux/H200 measured byte-identical to 6.795 GB with `fallbacks=0`, #925).
/// Anyone sweeping this on other hardware should confirm the value took effect
/// by checking that `zero_copy_bytes_per_token` tracks it, rather than trusting
/// that the environment was read as intended.
pub(super) fn zero_copy_budget_bytes() -> u64 {
    const NAME: &str = "ONNX_GENAI_ZERO_COPY_HYBRID_BUDGET_BYTES";
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| parse_numeric_env(NAME).or_default(NAME, ZERO_COPY_SAFE_BUDGET_BYTES))
}

///
/// This exists purely to answer the #888 discriminating question: byte-aware
/// residency (#886) changed *two* independent things at once — it started
/// *retaining* large tensors that used to bypass, **and** it started evicting
/// the *smallest* resident instead of the front-of-order oldest one. If merely
/// changing the eviction victim — while keeping the shipped size-blind
/// always-bypass semantics — is enough to corrupt decode output, the surrounding
/// offload path harbours a latent order-dependent defect (explanation 2). If
/// only the full byte-aware policy (which also flips pages from bypass to
/// retained) corrupts, the defect is specific to that retain change
/// (explanation 1). Default (`lru`, or unset) is byte-identical to the shipped
/// path; every other value is opt-in and experimental.
pub const WEIGHT_OFFLOAD_EVICT_ORDER_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_EVICT_ORDER";

/// Eviction victim ordering for the size-blind admission path. Only the victim
/// changes; the always-bypass decision, the strong-count-1 evictable predicate,
/// and the pre-eviction stream drain are all identical to the shipped path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EvictOrderProbe {
    /// Front-of-order oldest evictable page — the shipped `next_evictable_key`
    /// behaviour. Byte-identical default.
    #[default]
    Lru,
    /// Most-recently-used evictable page (reverse recency). A "trivially
    /// different but still-correct" order: it only ever targets `strong_count
    /// == 1` pages, exactly like LRU.
    Mru,
    /// Smallest evictable page by bytes — the eviction *target* byte-aware uses,
    /// but decoupled from byte-aware's retain-large decision.
    Smallest,
    /// Largest evictable page by bytes.
    Largest,
}

/// Parse [`WEIGHT_OFFLOAD_EVICT_ORDER_ENV`]. Unset or unrecognised keeps the
/// shipped `Lru` order, so the default path stays byte-identical.
pub(crate) fn evict_order_from_env_value(value: Option<&str>) -> EvictOrderProbe {
    match value.map(|value| value.trim().to_ascii_lowercase()) {
        Some(value) => match value.as_str() {
            "mru" | "reverse" => EvictOrderProbe::Mru,
            "smallest" | "small" => EvictOrderProbe::Smallest,
            "largest" | "large" => EvictOrderProbe::Largest,
            _ => EvictOrderProbe::Lru,
        },
        None => EvictOrderProbe::Lru,
    }
}

/// Read [`WEIGHT_OFFLOAD_EVICT_ORDER_ENV`] from the process environment.
#[must_use]
pub fn evict_order_probe_from_env() -> EvictOrderProbe {
    evict_order_from_env_value(
        std::env::var(WEIGHT_OFFLOAD_EVICT_ORDER_ENV)
            .ok()
            .as_deref(),
    )
}

/// #888 diagnostic env: when truthy, drain the compute and copy streams
/// immediately before every H2D page-in fill. Default OFF (byte-identical).
/// Used to classify whether byte-aware residency's corruption is a
/// write-after-read hazard on the shared physical granule pool (would be fixed
/// by this drain) or a pure aliasing/logic bug (would not).
const WEIGHT_OFFLOAD_SYNC_BEFORE_FILL_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_SYNC_BEFORE_FILL";

pub(super) fn sync_before_fill_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        matches!(
            std::env::var(WEIGHT_OFFLOAD_SYNC_BEFORE_FILL_ENV)
                .ok()
                .as_deref()
                .map(|value| value.trim().to_ascii_lowercase()),
            Some(ref value) if matches!(value.as_str(), "1" | "true" | "yes" | "on")
        )
    })
}

/// #888 diagnostic env: when truthy, byte-aware admission never *bypasses* a key
/// that already owns a stable VA slot — it re-retains it into residency instead.
/// Default OFF. This closes the one `stable_slot`/residency disagreement that is
/// reachable only under byte-aware (a once-retained tensor squeezed below the
/// resident set and re-entering as a bypass), letting an A/B on the same binary
/// test whether that disagreement is byte-aware's corruption mechanism.
const WEIGHT_OFFLOAD_RETAIN_SLOTTED_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_RETAIN_SLOTTED";

pub(super) fn retain_slotted_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        matches!(
            std::env::var(WEIGHT_OFFLOAD_RETAIN_SLOTTED_ENV)
                .ok()
                .as_deref()
                .map(|value| value.trim().to_ascii_lowercase()),
            Some(ref value) if matches!(value.as_str(), "1" | "true" | "yes" | "on")
        )
    })
}

/// Static hot-set pin (#837 item 3, reviewer follow-up). A tensor whose byte
/// length is at least the threshold is admitted **once** and never evicted or
/// re-admitted, up to a total pin budget. This is the provably-safe retention
/// shape: unlike byte-aware residency (which evicts and can re-admit a large
/// stable-slot tensor — the #886/#888 corruption pattern), a pinned tensor never
/// enters the eviction population at all, so it can never be evicted-and-re-
/// admitted. It is served as a hit across steps, saving its bytes every step.
/// Both variables must be set to positive integers to engage; either unset or
/// zero leaves the shipped size-blind path byte-identical.
const WEIGHT_OFFLOAD_PIN_THRESHOLD_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_PIN_THRESHOLD_BYTES";
const WEIGHT_OFFLOAD_PIN_BUDGET_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_PIN_BUDGET_BYTES";

/// `(threshold_bytes, budget_bytes)` when static hot-set pinning is engaged, or
/// `None` on the shipped path. Cached once per process.
pub(super) fn static_pin_config() -> Option<(u64, u64)> {
    static CACHE: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let threshold = parse_numeric_env(WEIGHT_OFFLOAD_PIN_THRESHOLD_ENV)
            .into_option(WEIGHT_OFFLOAD_PIN_THRESHOLD_ENV)?;
        let budget = parse_numeric_env(WEIGHT_OFFLOAD_PIN_BUDGET_ENV)
            .into_option(WEIGHT_OFFLOAD_PIN_BUDGET_ENV)?;
        (threshold > 0 && budget > 0).then_some((threshold, budget))
    })
}

/// Explicit key allow-list for the static hot-set pin (#837 item 3, reviewer
/// follow-up). Comma-separated `u64` weight keys; when set, *exactly* those keys
/// are pinned (admitted once, never evicted, never re-admitted) regardless of
/// size or budget, and the size-threshold path above is ignored. This lets the
/// measurement pin the specific chronic-bypasser tensors identified by the
/// per-key trace, rather than whichever large tensors happen to arrive first.
/// Unset on the shipped path.
const WEIGHT_OFFLOAD_PIN_KEYS_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD_PIN_KEYS";

/// Parsed explicit pin allow-list, or `None` when the variable is unset/empty.
/// Cached once per process.
pub(super) fn static_pin_keys() -> Option<&'static HashSet<u64>> {
    static CACHE: OnceLock<Option<HashSet<u64>>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let raw = std::env::var(WEIGHT_OFFLOAD_PIN_KEYS_ENV).ok()?;
            let keys: HashSet<u64> = raw
                .split(',')
                .filter_map(|token| token.trim().parse::<u64>().ok())
                .collect();
            (!keys.is_empty()).then_some(keys)
        })
        .as_ref()
}

/// Physical VMM granule size on this build (2 MiB). The staleness probes below
/// hash and report the retained device copy at this granularity so a divergence
/// can be pinned to a specific granule / byte offset within the tensor (#945
/// experiment 3, the within-tensor bisection).
pub(super) const PIN_PROBE_GRANULE_BYTES: usize = 2 * 1024 * 1024;

/// #945 experiment 2/3 — granule-level checksums of a pinned page. Comma-separated
/// `u64` weight keys whose retained device copy is hashed on admission and
/// re-hashed on every subsequent hit, reporting the first step at which the bytes
/// diverge from the admission snapshot and the first granule that changed. This is
/// a pure read-back (device→host memcpy + FNV-1a); it needs no CUDA toolkit, which
/// is what blocked settling #886. Unset on the shipped path, so the size-blind
/// residency path is byte-identical with it off.
const WEIGHT_PIN_CHECKSUM_ENV: &str = "ONNX_GENAI_WEIGHT_PIN_CHECKSUM";

/// #945 experiment 1 — force a re-fill of a pinned page every `N` hits. When set to
/// a positive integer `N`, every `N`-th time a *pinned* key is served as a hit its
/// device copy is re-filled from the original host source (the same closure that
/// filled it on admission). If corruption disappears under this and the host bytes
/// are unchanged, the retained device copy was going stale — the decisive control
/// for the retained-page-staleness hypothesis. A **null (unset) value performs no
/// re-fill and reproduces the baseline byte-for-byte**, so a positive reading here
/// cannot be an artifact of the knob's mere presence (measurement discipline #1).
pub(super) const WEIGHT_PIN_REFILL_EVERY_ENV: &str = "ONNX_GENAI_WEIGHT_PIN_REFILL_EVERY";

/// Parsed checksum-probe key allow-list, cached once per process. `None` when the
/// variable is unset/empty (the shipped path).
pub(super) fn pin_checksum_keys() -> Option<&'static HashSet<u64>> {
    static CACHE: OnceLock<Option<HashSet<u64>>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let raw = std::env::var(WEIGHT_PIN_CHECKSUM_ENV).ok()?;
            let keys: HashSet<u64> = raw
                .split(',')
                .filter_map(|token| token.trim().parse::<u64>().ok())
                .collect();
            (!keys.is_empty()).then_some(keys)
        })
        .as_ref()
}

/// Parsed force-re-fill cadence. `None`/`Some(0)` on the shipped path means
/// never re-fill. This is read only for an opt-in pinned page, and remains
/// dynamic so fault-injection tests and diagnostic runs can scope the knob.
pub(super) fn pin_refill_every() -> Option<u64> {
    std::env::var(WEIGHT_PIN_REFILL_EVERY_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|n| *n > 0)
}

/// #945 control — isolate the *reused-slot bypass* hazard. When enabled, a key
/// that already owns a persistent stable-VA slot (#716) but that admission would
/// classify as a **bypass** is kept **resident** instead of re-entering as a
/// transient bypass. The bypass path reuses the slot's persistent VA for a
/// throwaway page whose physical is later returned to the shared granule pool
/// while the baked VA still references it — the `#888` `stable_slot`/residency
/// disagreement. Force-retaining a large tensor (via `PIN_KEYS` or the
/// byte-aware policy) is exactly what evicts a slotted key's physical while
/// keeping its slot, so a later step re-requests it as a slotted bypass.
///
/// A **null (unset) value keeps the shipped behavior byte-for-byte** — the
/// override only ever fires for the `reused_slot && bypass` case, which is
/// unreachable on the size-blind default path — so a positive reading here is
/// not an artifact of the knob's presence (measurement discipline #1).
///
/// RESULT (#945, 2026-08-14): measured **INSUFFICIENT**. Enabling this over
/// `PIN_KEYS=919` still collapses decode (`not deterministic across measured
/// runs`) and the `#888` diagnostic still fires. Keeping the re-requested
/// slotted key resident does not undo the corruption, because the damaging event
/// is the *earlier eviction* of that slotted key's physical granules (returned
/// to the shared pool and re-mapped under the forced large retention), not the
/// bookkeeping of the later re-admission. This ELIMINATES the reused-slot-bypass
/// re-admission as the fix site. Kept as a documented negative control.
const WEIGHT_SLOT_BYPASS_RETAIN_ENV: &str = "ONNX_GENAI_WEIGHT_SLOT_BYPASS_RETAIN";

/// Read [`WEIGHT_SLOT_BYPASS_RETAIN_ENV`], cached once per process. `false` on
/// the shipped path (unset/any non-truthy value).
pub(super) fn slot_bypass_retain_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        matches!(
            std::env::var(WEIGHT_SLOT_BYPASS_RETAIN_ENV)
                .ok()
                .as_deref()
                .map(|value| value.trim().to_ascii_lowercase()),
            Some(ref v) if v == "1" || v == "true" || v == "yes" || v == "on"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_copy_hybrid_env_defaults_off_and_opts_in() {
        // Opt-in, default OFF: the hybrid is byte-identical to the shipped path
        // until an operator explicitly enables it (#864). Unset and every
        // non-truthy spelling stay OFF.
        assert!(!zero_copy_hybrid_from_env_value(None));
        assert!(!zero_copy_hybrid_from_env_value(Some("0")));
        assert!(!zero_copy_hybrid_from_env_value(Some("false")));
        assert!(!zero_copy_hybrid_from_env_value(Some("")));
        assert!(!zero_copy_hybrid_from_env_value(Some("maybe")));
        // Only the canonical truthy spellings enable it.
        assert!(zero_copy_hybrid_from_env_value(Some("1")));
        assert!(zero_copy_hybrid_from_env_value(Some("true")));
        assert!(zero_copy_hybrid_from_env_value(Some("YES")));
        assert!(zero_copy_hybrid_from_env_value(Some("  On ")));
    }

    #[test]
    fn zero_copy_safe_budget_defaults_are_platform_aware() {
        // Bind the consts to locals so these stay runtime assertions (comparing
        // two consts directly trips clippy::assertions_on_constants).
        let wddm = ZERO_COPY_SAFE_BUDGET_BYTES_WDDM;
        let non_windows = ZERO_COPY_SAFE_BUDGET_BYTES_NON_WINDOWS;
        let selected = ZERO_COPY_SAFE_BUDGET_BYTES;

        // Windows/WDDM (#864): distinct host-mapped reads were byte-identical at
        // 32 cold weights (~0.44 GB/step) and corrupted at 48 (~0.65 GB/step).
        // The WDDM default must sit strictly under the observed-safe ceiling so
        // the opt-in knob can never violate the byte-identical gate there.
        let wddm_observed_safe_bytes: u64 = 436_633_600; // 32 cold weights, measured
        assert_eq!(wddm, 256 * 1024 * 1024);
        assert!(
            wddm < wddm_observed_safe_bytes,
            "WDDM default {wddm} must stay under the WDDM ceiling"
        );

        // Non-Windows (#925): Linux/H200 stayed byte-identical to the Step-0
        // baseline with cuda_graph fallbacks=0 up to 6.795 GB of distinct
        // host-mapped bytes bound and re-read per step (704 binds, n=3). The
        // non-Windows default is raised to 2 GiB: bounded and >3x under that
        // measured-safe figure, yet above the whole WDDM corruption band so the
        // lever #925 proved available on Linux is actually unlocked.
        let linux_measured_safe_bytes: u64 = 6_795_458_560; // #925, debug-summed
        assert_eq!(non_windows, 2 * 1024 * 1024 * 1024);
        assert!(
            non_windows < linux_measured_safe_bytes,
            "non-Windows default {non_windows} must stay under the #925 measured-safe ceiling"
        );
        assert!(
            non_windows > wddm_observed_safe_bytes,
            "non-Windows default must clear the WDDM corruption band to unlock the lever"
        );

        // The platform-selected default resolves to the right named const.
        #[cfg(target_os = "windows")]
        assert_eq!(selected, wddm);
        #[cfg(not(target_os = "windows"))]
        assert_eq!(selected, non_windows);
    }

    #[test]
    fn numeric_env_keeps_unset_and_unparseable_distinct() {
        // The hazard is a measurement one. A sweep that writes `2GB`, or leaves
        // a digit separator or a stray character, would silently fall back to
        // the conservative default -- and then report "no corruption at 2 GiB"
        // having never tested 2 GiB. A confident wrong answer is worse than an
        // error.
        //
        // The assertion that matters is that `Invalid` is distinguishable from
        // `Unset`. The previous implementation was `.parse().ok()`, which
        // collapsed both to `None`, so a test that only checked "bad input does
        // not yield a value" would have passed against it unchanged and proved
        // nothing. Same shape as rendering "not determined" as "determined to
        // be unsafe" (#931).
        const NAME: &str = "ONNX_GENAI_TEST_NUMERIC_ENV_PROBE";

        // SAFETY: single-threaded test-local variable with a unique name; it is
        // set and removed within this test and read by nothing else.
        unsafe {
            std::env::remove_var(NAME);
        }
        assert_eq!(parse_numeric_env(NAME), NumericEnv::Unset);

        // SAFETY: as above.
        unsafe {
            std::env::set_var(NAME, " 1073741824 ");
        }
        assert_eq!(
            parse_numeric_env(NAME),
            NumericEnv::Value(1_073_741_824),
            "a plain integer, surrounding whitespace included, must be honoured"
        );

        for bad in ["2GB", "1_073_741_824", "0x10", "", "1.5", "-1"] {
            // SAFETY: as above.
            unsafe {
                std::env::set_var(NAME, bad);
            }
            assert_eq!(
                parse_numeric_env(NAME),
                NumericEnv::Invalid(bad.to_string()),
                "{bad:?} must be reported as supplied-but-unusable, not as absent"
            );
        }

        // The default substitution still happens -- the point is that it is
        // announced rather than silent.
        assert_eq!(parse_numeric_env(NAME).or_default(NAME, 4096), 4096);
        assert_eq!(parse_numeric_env(NAME).into_option(NAME), None);

        // SAFETY: as above.
        unsafe {
            std::env::remove_var(NAME);
        }
        assert_eq!(parse_numeric_env(NAME).or_default(NAME, 4096), 4096);
    }

    #[test]
    fn evict_order_env_defaults_lru_and_parses_variants() {
        // Unset and unrecognised keep the shipped front-of-order LRU victim, so
        // the default path is byte-identical (#888).
        assert_eq!(evict_order_from_env_value(None), EvictOrderProbe::Lru);
        assert_eq!(evict_order_from_env_value(Some("")), EvictOrderProbe::Lru);
        assert_eq!(
            evict_order_from_env_value(Some("lru")),
            EvictOrderProbe::Lru
        );
        assert_eq!(
            evict_order_from_env_value(Some("nonsense")),
            EvictOrderProbe::Lru
        );
        // Opt-in experimental orders (case/whitespace-insensitive).
        assert_eq!(
            evict_order_from_env_value(Some("mru")),
            EvictOrderProbe::Mru
        );
        assert_eq!(
            evict_order_from_env_value(Some(" Reverse ")),
            EvictOrderProbe::Mru
        );
        assert_eq!(
            evict_order_from_env_value(Some("SMALLEST")),
            EvictOrderProbe::Smallest
        );
        assert_eq!(
            evict_order_from_env_value(Some("large")),
            EvictOrderProbe::Largest
        );
    }
}
