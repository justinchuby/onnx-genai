//! Automatic trace diagnosis and roofline analysis (§46.6 / §49.5).
//!
//! Reading a raw trace is expensive and error-prone, so this module turns a
//! collected timeline into **answers**: it scans the [`TraceEvent`] stream the
//! rest of the crate already produces and emits [`DiagnosedIssue`]s, and it
//! classifies individual kernels/ops against the device roofline. Everything
//! here obeys the project's flagship rule ([`RULES.md` #1](../../../RULES.md)):
//! every finding carries **WHAT** is wrong ([`description`](DiagnosedIssue::description)),
//! **WHY** ([`evidence`](DiagnosedIssue::evidence) +
//! [`root_cause`](DiagnosedIssue::root_cause)), and **HOW** to fix it
//! ([`suggestion`](DiagnosedIssue::suggestion)). Vague, unactionable findings
//! are treated as a bug, not a feature.
//!
//! ## Fitting the crate's real data model
//!
//! The §49.5 design sketch analyses a `GpuKernelMetrics` value with
//! `flop_count_hp` / `duration_ns` fields. The crate's *actual*
//! [`GpuKernelMetrics`](crate::cupti::GpuKernelMetrics) is a `cupti`-gated
//! Phase-2 **stub** with a different shape (occupancy / DRAM bytes, no FLOP
//! counts), and this module must build in the default, safe, no-`cupti`
//! configuration. So rather than invent a parallel data model or depend on a
//! feature-gated stub, the roofline analyzer consumes a small
//! [`KernelSample`] that is populated directly from the data the crate collects
//! today — a [`TraceEvent`]'s duration plus its
//! [`Args`](crate::Args) `flops` / `bytes` / `precision` fields. When those
//! metrics are absent (the common case until the CUPTI PM-counter path lands),
//! the analyzer **degrades gracefully** instead of dividing by zero or
//! fabricating numbers (see [`BoundType::Indeterminate`]).
//!
//! ## Quick start
//!
//! ```
//! use onnx_runtime_tracer::{Args, TraceContext};
//! use onnx_runtime_tracer::diagnose::{AutoDiagnosis, RooflineAnalyzer};
//!
//! let (ctx, mem) = TraceContext::in_memory();
//! for _ in 0..8 {
//!     let _s = ctx.span("MatMul", "compute")
//!         .with_args(Args::new().flops(34_359_738_368_u64).bytes(268_435_456));
//! }
//! let events = mem.events();
//!
//! // Whole-trace diagnosis → a human-readable, copy-pasteable report.
//! let diagnosis = AutoDiagnosis::analyze(&events);
//! println!("{}", diagnosis.report());
//!
//! // Per-kernel roofline classification against an H100.
//! let analyzer = RooflineAnalyzer::h100();
//! for ev in &events {
//!     let result = analyzer.analyze_event(ev);
//!     println!("{result:?}");
//! }
//! ```

use crate::args::{ARG_CHOSEN_KERNEL, ARG_FASTPATH_REJECTED_REASON, ARG_OPTIMIZED_CANDIDATE};
use crate::event::{TraceEvent, TracePhase};
use std::collections::BTreeMap;
use std::fmt;

// =============================================================================
// Roofline analysis (§49.5)
// =============================================================================

/// The numeric precision a kernel executed at, selecting which device FLOP
/// ceiling the roofline is measured against.
///
/// Tensor-core (fp16/bf16) and CUDA-core (fp32) peaks differ by several times,
/// so classifying against the wrong ceiling would mislabel efficiency. When the
/// precision is unknown we default to [`Precision::Fp16`], the common path for
/// LLM inference kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Precision {
    /// Half precision / bfloat16 — measured against `peak_flops_fp16`.
    #[default]
    Fp16,
    /// Single precision — measured against `peak_flops_fp32`.
    Fp32,
}

impl Precision {
    /// Parse a precision from a free-form dtype/precision string
    /// (`"fp16"`, `"f16"`, `"half"`, `"bf16"`, `"fp32"`, `"f32"`, `"float"`).
    ///
    /// Unrecognised strings fall back to the [`Default`] ([`Precision::Fp16`]).
    #[must_use]
    pub fn from_str_lenient(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "fp32" | "f32" | "float" | "float32" | "single" => Precision::Fp32,
            _ => Precision::Fp16,
        }
    }
}

/// A single kernel/op measurement fed to the [`RooflineAnalyzer`].
///
/// This is the seam between the trace data the crate collects **today**
/// (a [`TraceEvent`]'s duration plus its [`Args`] `flops`/`bytes`) and the
/// roofline model. Each metric is optional so a partially-instrumented op (the
/// normal case until the CUPTI PM-counter path lands) degrades cleanly rather
/// than being dropped or faked. Build one from a [`TraceEvent`] with
/// [`KernelSample::from_event`], or construct it directly for testing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelSample {
    /// Kernel or op name (e.g. `"MatMul_0"`).
    pub kernel_name: String,
    /// Correlated runtime node id, if the event carried one (`args.node_id`).
    pub node_id: Option<u64>,
    /// Estimated floating-point operations performed. `None` when not measured.
    pub flops: Option<u64>,
    /// Bytes moved to/from memory for this kernel. `None` when not measured.
    pub bytes_accessed: Option<u64>,
    /// Wall-clock duration in **nanoseconds**. `None` when not timed.
    pub duration_ns: Option<u64>,
    /// The precision the kernel ran at (selects the FLOP ceiling).
    pub precision: Precision,
}

impl KernelSample {
    /// Build a sample from a [`TraceEvent`], reading `flops`, `bytes`,
    /// `node_id`, and `precision`/`dtype` from its [`Args`] and its duration
    /// from `dur` (µs → ns). Missing fields stay `None` so the analyzer can
    /// degrade gracefully.
    #[must_use]
    pub fn from_event(ev: &TraceEvent) -> Self {
        let precision = arg_str(ev, "precision")
            .or_else(|| arg_str(ev, "dtype"))
            .map_or(Precision::default(), Precision::from_str_lenient);
        Self {
            kernel_name: ev.name.clone(),
            node_id: arg_u64(ev, "node_id"),
            flops: arg_u64(ev, "flops"),
            bytes_accessed: arg_u64(ev, "bytes"),
            duration_ns: ev.dur.map(|us| us.saturating_mul(1_000)),
            precision,
        }
    }
}

/// How a kernel is limited, per the roofline model (§49.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundType {
    /// Arithmetic intensity above the ridge point → limited by FLOP throughput.
    ComputeBound,
    /// Arithmetic intensity below the ridge point → limited by bandwidth.
    MemoryBound,
    /// Kernel too short — dominated by launch overhead (classifiable from
    /// duration alone, even without FLOP/byte metrics).
    LaunchBound,
    /// Near the ridge point — compute and memory are roughly balanced.
    Balanced,
    /// **Graceful-degradation state**: the metrics needed to classify (FLOPs
    /// and/or bytes, or a usable duration) were not available. Not a device
    /// property — a *measurement* gap, surfaced explicitly instead of guessing.
    Indeterminate,
}

impl BoundType {
    /// A short lowercase label matching the §49.7 CLI (`"compute-bound"`, …).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            BoundType::ComputeBound => "compute-bound",
            BoundType::MemoryBound => "memory-bound",
            BoundType::LaunchBound => "launch-bound",
            BoundType::Balanced => "balanced",
            BoundType::Indeterminate => "indeterminate",
        }
    }
}

impl fmt::Display for BoundType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// The result of classifying one kernel against the device roofline (§49.5).
///
/// [`arithmetic_intensity`](RooflineResult::arithmetic_intensity),
/// [`achieved_tflops`](RooflineResult::achieved_tflops), and
/// [`efficiency`](RooflineResult::efficiency) are **optional**: when the input
/// [`KernelSample`] lacks the metrics to compute them they are `None` rather
/// than a fabricated `0.0`, and [`bound`](RooflineResult::bound) is
/// [`BoundType::Indeterminate`]. A [`suggestion`](RooflineResult::suggestion) is
/// always populated with an actionable next step (including "how to get the
/// missing metrics" in the degraded case).
#[derive(Clone, Debug, PartialEq)]
pub struct RooflineResult {
    /// The kernel/op name this result describes.
    pub kernel_name: String,
    /// Correlated runtime node id, if the sample carried one.
    pub node_id: Option<u64>,
    /// Arithmetic intensity (FLOPs / bytes). `None` if not computable.
    pub arithmetic_intensity: Option<f64>,
    /// Achieved throughput in TFLOP/s. `None` if not computable.
    pub achieved_tflops: Option<f64>,
    /// Bottleneck classification.
    pub bound: BoundType,
    /// Fraction of the attainable roofline ceiling reached (`1.0` = at the
    /// roofline). `None` if not computable.
    pub efficiency: Option<f64>,
    /// An actionable suggestion. Always `Some` for this module's output.
    pub suggestion: Option<String>,
}

/// Automatically classify kernels as compute-, memory-, or launch-bound and
/// suggest how to improve them (§49.5).
///
/// The device ceilings are constructor parameters, so the same analyzer works
/// for any GPU. Use [`RooflineAnalyzer::h100`] for a ready-made example, or
/// [`RooflineAnalyzer::new`] with your device's numbers.
#[derive(Clone, Debug, PartialEq)]
pub struct RooflineAnalyzer {
    peak_flops_fp16: f64,
    peak_flops_fp32: f64,
    peak_bandwidth: f64,
    launch_threshold_ns: u64,
    low_efficiency: f64,
}

impl RooflineAnalyzer {
    /// Create an analyzer from a device's peak ceilings.
    ///
    /// * `peak_flops_fp16` — peak fp16/bf16 (tensor-core) FLOP/s.
    /// * `peak_flops_fp32` — peak fp32 (CUDA-core) FLOP/s.
    /// * `peak_bandwidth`  — peak memory bandwidth in **bytes/s**.
    ///
    /// Non-finite or non-positive ceilings are clamped to a tiny positive value
    /// so the analyzer can never divide by zero (it will simply report low
    /// efficiency); callers should pass real device numbers.
    #[must_use]
    pub fn new(peak_flops_fp16: f64, peak_flops_fp32: f64, peak_bandwidth: f64) -> Self {
        let sane = |v: f64| if v.is_finite() && v > 0.0 { v } else { f64::MIN_POSITIVE };
        Self {
            peak_flops_fp16: sane(peak_flops_fp16),
            peak_flops_fp32: sane(peak_flops_fp32),
            peak_bandwidth: sane(peak_bandwidth),
            launch_threshold_ns: DEFAULT_LAUNCH_THRESHOLD_NS,
            low_efficiency: DEFAULT_LOW_EFFICIENCY,
        }
    }

    /// A ready-made NVIDIA H100 (SXM, HBM3) profile for examples and tests:
    /// 989 TFLOPS fp16 tensor-core, 67 TFLOPS fp32, 3.35 TB/s HBM.
    #[must_use]
    pub fn h100() -> Self {
        Self::new(989e12, 67e12, 3.35e12)
    }

    /// Kernels shorter than this (nanoseconds) are classified
    /// [`BoundType::LaunchBound`] — launch overhead dominates. Default
    /// [`DEFAULT_LAUNCH_THRESHOLD_NS`].
    #[must_use]
    pub fn with_launch_threshold_ns(mut self, ns: u64) -> Self {
        self.launch_threshold_ns = ns;
        self
    }

    /// Efficiency below this fraction triggers the "low utilization" tuning
    /// suggestions. Default [`DEFAULT_LOW_EFFICIENCY`].
    #[must_use]
    pub fn with_low_efficiency(mut self, frac: f64) -> Self {
        self.low_efficiency = frac;
        self
    }

    /// The ridge point (FLOPs/byte) where the compute ceiling meets the memory
    /// ceiling, for the given precision.
    fn ridge_point(&self, precision: Precision) -> f64 {
        self.peak_flops(precision) / self.peak_bandwidth
    }

    fn peak_flops(&self, precision: Precision) -> f64 {
        match precision {
            Precision::Fp16 => self.peak_flops_fp16,
            Precision::Fp32 => self.peak_flops_fp32,
        }
    }

    /// Classify a single [`TraceEvent`] (convenience over
    /// [`KernelSample::from_event`] + [`analyze`](RooflineAnalyzer::analyze)).
    #[must_use]
    pub fn analyze_event(&self, ev: &TraceEvent) -> RooflineResult {
        self.analyze(&KernelSample::from_event(ev))
    }

    /// Classify a [`KernelSample`] against the roofline.
    ///
    /// Never panics: zero bytes, zero duration, or absent FLOP/byte metrics all
    /// degrade to [`BoundType::LaunchBound`] (from duration) or
    /// [`BoundType::Indeterminate`] (insufficient metrics) with an explanatory
    /// suggestion, rather than dividing by zero.
    #[must_use]
    pub fn analyze(&self, s: &KernelSample) -> RooflineResult {
        let dur_ns = s.duration_ns.filter(|&d| d > 0);
        let flops = s.flops.filter(|&f| f > 0);
        let bytes = s.bytes_accessed.filter(|&b| b > 0);

        // Full metrics available → real roofline classification.
        if let (Some(flops), Some(bytes), Some(dur_ns)) = (flops, bytes, dur_ns) {
            return self.analyze_full(s, flops, bytes, dur_ns);
        }

        // Degraded path 1: a duration but no FLOP/byte metrics. A very short
        // kernel is still confidently launch-bound from timing alone.
        if let Some(dur_ns) = dur_ns.filter(|&d| d < self.launch_threshold_ns) {
            return RooflineResult {
                    kernel_name: s.kernel_name.clone(),
                    node_id: s.node_id,
                    arithmetic_intensity: None,
                    achieved_tflops: None,
                    bound: BoundType::LaunchBound,
                    efficiency: None,
                    suggestion: Some(format!(
                        "Kernel ran in {:.2}µs (< {:.2}µs launch-overhead threshold) so \
                         launch cost dominates, even without FLOP/byte metrics. Fix: fuse it \
                         with neighbouring ops or capture the region in a CUDA graph to amortise \
                         launch latency.",
                        dur_ns as f64 / 1_000.0,
                        self.launch_threshold_ns as f64 / 1_000.0,
                    )),
                };
        }

        // Degraded path 2: not enough to classify. Say so explicitly.
        let missing = describe_missing(flops.is_some(), bytes.is_some(), dur_ns.is_some());
        RooflineResult {
            kernel_name: s.kernel_name.clone(),
            node_id: s.node_id,
            arithmetic_intensity: None,
            achieved_tflops: None,
            bound: BoundType::Indeterminate,
            efficiency: None,
            suggestion: Some(format!(
                "Insufficient metrics to place '{}' on the roofline ({missing}). This is expected \
                 until GPU FLOP/byte counters are collected. Fix: attach FLOP and byte estimates \
                 to the span's Args (Args::flops/Args::bytes), or profile in CUPTI `metrics` mode \
                 (§49.6) to capture them.",
                s.kernel_name,
            )),
        }
    }

    fn analyze_full(&self, s: &KernelSample, flops: u64, bytes: u64, dur_ns: u64) -> RooflineResult {
        let ai = flops as f64 / bytes as f64;
        let ridge = self.ridge_point(s.precision);
        let peak_flops = self.peak_flops(s.precision);

        // Achieved throughput and the attainable roofline ceiling at this AI
        // (min of the compute ceiling and the bandwidth-limited slope). One
        // formula covers both regimes, so efficiency is always in [0, ~1].
        let achieved = flops as f64 / dur_ns as f64 * 1e9;
        let attainable = peak_flops.min(ai * self.peak_bandwidth);
        let efficiency = achieved / attainable;

        let bound = if dur_ns < self.launch_threshold_ns {
            BoundType::LaunchBound
        } else if ai > ridge * 1.2 {
            BoundType::ComputeBound
        } else if ai < ridge * 0.8 {
            BoundType::MemoryBound
        } else {
            BoundType::Balanced
        };

        let low = efficiency < self.low_efficiency;
        let suggestion = Some(match bound {
            BoundType::LaunchBound => format!(
                "Kernel ran in {:.2}µs (< {:.2}µs launch-overhead threshold): launch cost \
                 dominates. Fix: fuse with neighbouring ops or capture the region in a CUDA graph.",
                dur_ns as f64 / 1_000.0,
                self.launch_threshold_ns as f64 / 1_000.0,
            ),
            BoundType::MemoryBound if low => format!(
                "Memory-bound at {:.0}% of the bandwidth roofline (AI={ai:.2} FLOP/byte, below the \
                 {ridge:.1} ridge point). Fix: (1) fuse with adjacent ops to cut memory traffic, \
                 (2) use in-place operations, (3) quantise to shrink the data moved.",
                efficiency * 100.0,
            ),
            BoundType::MemoryBound => format!(
                "Memory-bound (AI={ai:.2} FLOP/byte < ridge {ridge:.1}) but already at {:.0}% of the \
                 bandwidth roofline — near the practical limit. Fix: only a fusion or dtype change \
                 that reduces bytes moved will help further.",
                efficiency * 100.0,
            ),
            BoundType::ComputeBound if low => format!(
                "Compute-bound at {:.0}% of the FLOP roofline (AI={ai:.2} FLOP/byte, above the \
                 {ridge:.1} ridge point). Fix: (1) raise occupancy (fewer registers/less shared \
                 memory), (2) use Tensor Cores (fp16/bf16/int8), (3) enlarge tiles for better reuse.",
                efficiency * 100.0,
            ),
            BoundType::ComputeBound => format!(
                "Compute-bound (AI={ai:.2} FLOP/byte > ridge {ridge:.1}) at {:.0}% of the FLOP \
                 roofline — healthy. No action needed; keep this op on the tensor-core path.",
                efficiency * 100.0,
            ),
            BoundType::Balanced => format!(
                "Balanced near the ridge point (AI={ai:.2} FLOP/byte ≈ ridge {ridge:.1}) at {:.0}% \
                 efficiency. Fix: it is sensitive to both compute and bandwidth — improve whichever \
                 the surrounding kernels are more constrained by.",
                efficiency * 100.0,
            ),
            BoundType::Indeterminate => unreachable!("full-metric path always classifies"),
        });

        RooflineResult {
            kernel_name: s.kernel_name.clone(),
            node_id: s.node_id,
            arithmetic_intensity: Some(ai),
            achieved_tflops: Some(achieved / 1e12),
            bound,
            efficiency: Some(efficiency),
            suggestion,
        }
    }
}

/// Default launch-overhead threshold (nanoseconds): kernels shorter than this
/// are launch-bound (§49.5).
pub const DEFAULT_LAUNCH_THRESHOLD_NS: u64 = 5_000;

/// Default efficiency fraction below which tuning suggestions are emitted.
pub const DEFAULT_LOW_EFFICIENCY: f64 = 0.5;

fn describe_missing(flops: bool, bytes: bool, dur: bool) -> String {
    let mut missing = Vec::new();
    if !flops {
        missing.push("FLOP count");
    }
    if !bytes {
        missing.push("bytes accessed");
    }
    if !dur {
        missing.push("a non-zero duration");
    }
    format!("missing {}", missing.join(" and "))
}

/// Render a set of [`RooflineResult`]s as a human-readable report matching the
/// §49.7 CLI (`MatMul_0 : AI=128.5 compute-bound efficiency=72% → …`).
#[must_use]
pub fn render_roofline_report(results: &[RooflineResult]) -> String {
    if results.is_empty() {
        return "No kernels to analyze.\n".to_string();
    }
    let name_w = results.iter().map(|r| r.kernel_name.len()).max().unwrap_or(0).max(4);
    let mut out = String::new();
    for r in results {
        let ai = r
            .arithmetic_intensity
            .map_or_else(|| "AI=  n/a".to_string(), |v| format!("AI={v:>6.1}"));
        let eff = r
            .efficiency
            .map_or_else(|| "efficiency= n/a".to_string(), |v| format!("efficiency={:>3.0}%", v * 100.0));
        out.push_str(&format!(
            "{:<name_w$} : {ai}  {:<14} {eff}\n",
            r.kernel_name,
            r.bound.label(),
        ));
        if let Some(s) = &r.suggestion {
            out.push_str(&format!("    → {s}\n"));
        }
    }
    out
}

// =============================================================================
// Automatic diagnosis (§46.6)
// =============================================================================

/// How serious a [`DiagnosedIssue`] is (§46.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    /// Informational — worth knowing, not necessarily a problem.
    Info,
    /// A likely performance problem to address.
    Warning,
    /// A severe problem very likely hurting correctness or performance a lot.
    Critical,
}

impl Severity {
    /// The report icon for this severity (`ℹ️` / `⚠️` / `🛑`).
    #[must_use]
    pub fn icon(self) -> &'static str {
        match self {
            Severity::Info => "ℹ️",
            Severity::Warning => "⚠️",
            Severity::Critical => "🛑",
        }
    }
}

/// The category of a diagnosed problem (§46.6).
///
/// The variants marked *live* are derivable from the [`TraceEvent`] stream the
/// crate collects today; the *hook* variants need data (GPU PM counters,
/// placement/tier info, a reference run) that later phases will provide and are
/// documented as TODO seams on [`AutoDiagnosis`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IssueCategory {
    /// *Live.* An op much slower than its cohort (statistical outlier).
    SlowOp,
    /// *Live (data-conditional).* Frequent memory-pressure/eviction events.
    MemoryThrashing,
    /// *Live (data-conditional).* Transfers not hidden by compute.
    PrefetchStall,
    /// *Live.* An op runs with varying input shapes (buffer reallocations).
    ShapeInstability,
    /// *Live.* Shapes keep changing, preventing CUDA-graph capture.
    NoCudaGraph,
    /// *Live (data-conditional).* An op had an **optimized kernel available**
    /// (e.g. FlashAttention / fused SDPA / fused GEMM) but the runtime took the
    /// generic fallback instead. Fires when a kernel/EP reports the rejection
    /// via [`report_missed_fastpath`] / [`Args::missed_fastpath`](crate::Args::missed_fastpath).
    MissedFastPath,
    /// *Hook (needs placement/tier data).* Hot weight on the wrong device tier.
    SuboptimalPlacement,
    /// *Hook (needs a reference run).* Output diverges numerically.
    NumericalDivergence,
}

/// One automatically diagnosed problem (§46.6).
///
/// Extends the design sketch with [`root_cause`](DiagnosedIssue::root_cause) and
/// [`evidence_summary`](DiagnosedIssue::evidence_summary) so every issue can
/// render the full **WHAT / WHY / HOW** contract from `RULES.md` #1, not just a
/// bare label. [`evidence`](DiagnosedIssue::evidence) keeps the *actual* spans
/// so a tool or human can drill into the raw timeline.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosedIssue {
    /// How serious this is.
    pub severity: Severity,
    /// What kind of problem it is.
    pub category: IssueCategory,
    /// **WHAT** is wrong — the one-line headline.
    pub description: String,
    /// **WHY** — a human summary of the evidence.
    pub evidence_summary: String,
    /// **WHY** — the diagnosed root cause.
    pub root_cause: String,
    /// **WHY** — the actual spans/events backing this finding.
    pub evidence: Vec<TraceEvent>,
    /// **HOW** — the actionable fix.
    pub suggestion: String,
}

impl DiagnosedIssue {
    /// Render this issue in the §46.6 CLI style (icon + headline, then
    /// `Evidence:` / `Root cause:` / `Fix:` lines).
    #[must_use]
    pub fn render(&self) -> String {
        format!(
            "{}  {}\n   Evidence: {}\n   Root cause: {}\n   Fix: {}\n",
            self.severity.icon(),
            self.description,
            self.evidence_summary,
            self.root_cause,
            self.suggestion,
        )
    }
}

/// The result of scanning a trace for problems (§46.6).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AutoDiagnosis {
    /// Every issue found, most severe first.
    pub issues: Vec<DiagnosedIssue>,
}

/// Tunable thresholds for [`AutoDiagnosis`]. Sensible defaults via [`Default`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiagnosisConfig {
    /// Minimum cohort size before outlier statistics are trusted.
    pub min_cohort: usize,
    /// An op is a slow outlier only if it is at least this many times the
    /// cohort median.
    pub slow_op_ratio: f64,
    /// …and its robust (MAD-based) z-score exceeds this.
    pub slow_op_z: f64,
    /// A slow op at or above this ratio is escalated to [`Severity::Critical`].
    pub slow_op_critical_ratio: f64,
    /// Window (µs) over which memory-pressure events are counted for thrashing.
    pub thrash_window_us: u64,
    /// Minimum pressure events within one window to flag thrashing.
    pub thrash_min_events: usize,
}

impl Default for DiagnosisConfig {
    fn default() -> Self {
        Self {
            min_cohort: 4,
            slow_op_ratio: 3.0,
            slow_op_z: 3.5,
            slow_op_critical_ratio: 10.0,
            thrash_window_us: 500_000,
            thrash_min_events: 8,
        }
    }
}

impl AutoDiagnosis {
    /// Scan a slice of collected [`TraceEvent`]s with the default
    /// [`DiagnosisConfig`].
    #[must_use]
    pub fn analyze(events: &[TraceEvent]) -> Self {
        Self::analyze_with(events, &DiagnosisConfig::default())
    }

    /// Scan a slice of collected [`TraceEvent`]s with custom thresholds.
    #[must_use]
    pub fn analyze_with(events: &[TraceEvent], cfg: &DiagnosisConfig) -> Self {
        let mut issues = Vec::new();
        detect_slow_ops(events, cfg, &mut issues);
        detect_shape_instability(events, &mut issues);
        detect_no_cuda_graph(events, &mut issues);
        detect_memory_thrashing(events, cfg, &mut issues);
        detect_prefetch_stall(events, &mut issues);
        detect_missed_fastpath(events, &mut issues);
        // TODO(§46.6): SuboptimalPlacement needs weight-tier/hotness metadata
        // and NumericalDivergence needs a reference run — neither is present in
        // the trace stream yet. See `detect_suboptimal_placement` /
        // `detect_numerical_divergence` hooks below.

        // Most severe first, then by category for stable output.
        issues.sort_by(|a, b| {
            severity_rank(b.severity)
                .cmp(&severity_rank(a.severity))
                .then_with(|| format!("{:?}", a.category).cmp(&format!("{:?}", b.category)))
        });
        Self { issues }
    }

    /// Whether the scan found no problems.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.issues.is_empty()
    }

    /// Render the full report in the §46.6 CLI style. When nothing is wrong,
    /// says so warmly rather than printing an empty string (RULES.md #1).
    #[must_use]
    pub fn report(&self) -> String {
        if self.issues.is_empty() {
            return "✅ No issues detected — the trace looks healthy.\n".to_string();
        }
        let mut out = String::new();
        for (i, issue) in self.issues.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str(&issue.render());
        }
        out
    }
}

impl fmt::Display for AutoDiagnosis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.report())
    }
}

fn severity_rank(s: Severity) -> u8 {
    match s {
        Severity::Info => 0,
        Severity::Warning => 1,
        Severity::Critical => 2,
    }
}

// --- Detectors ---------------------------------------------------------------

/// Group the base op name out of an instance name: `"MatMul_7"` → `"MatMul"`.
fn cohort_key(name: &str) -> &str {
    if let Some(i) = name.rfind('_') {
        let (head, tail) = name.split_at(i);
        if !head.is_empty() && tail.len() > 1 && tail[1..].bytes().all(|b| b.is_ascii_digit()) {
            return head;
        }
    }
    name
}

fn completed(events: &[TraceEvent]) -> impl Iterator<Item = (&TraceEvent, u64)> {
    events
        .iter()
        .filter(|e| e.ph == TracePhase::Complete)
        .filter_map(|e| e.dur.map(|d| (e, d)))
}

fn median_sorted(sorted: &[u64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2] as f64
    } else {
        (sorted[n / 2 - 1] as f64 + sorted[n / 2] as f64) / 2.0
    }
}

/// SlowOp: an op N× slower than its cohort's robust centre (median + MAD).
fn detect_slow_ops(events: &[TraceEvent], cfg: &DiagnosisConfig, out: &mut Vec<DiagnosedIssue>) {
    let mut cohorts: BTreeMap<&str, Vec<(&TraceEvent, u64)>> = BTreeMap::new();
    for (ev, dur) in completed(events) {
        cohorts.entry(cohort_key(&ev.name)).or_default().push((ev, dur));
    }

    for (key, members) in cohorts {
        if members.len() < cfg.min_cohort {
            continue;
        }
        let mut durs: Vec<u64> = members.iter().map(|(_, d)| *d).collect();
        durs.sort_unstable();
        let median = median_sorted(&durs);
        if median <= 0.0 {
            continue;
        }
        // Median absolute deviation (robust spread).
        let mut abs_dev: Vec<u64> = durs.iter().map(|d| (*d as f64 - median).abs() as u64).collect();
        abs_dev.sort_unstable();
        let mad = median_sorted(&abs_dev);

        for (ev, dur) in &members {
            let ratio = *dur as f64 / median;
            if ratio < cfg.slow_op_ratio {
                continue;
            }
            // Robust z-score; when MAD is zero (identical cohort) rely on ratio.
            let z = if mad > 0.0 {
                0.6745 * (*dur as f64 - median) / mad
            } else {
                f64::INFINITY
            };
            if z < cfg.slow_op_z {
                continue;
            }
            let severity = if ratio >= cfg.slow_op_critical_ratio {
                Severity::Critical
            } else {
                Severity::Warning
            };
            let shape_note = arg_str(ev, "device")
                .map(|d| format!(", device={d}"))
                .unwrap_or_default();
            out.push(DiagnosedIssue {
                severity,
                category: IssueCategory::SlowOp,
                description: format!("Slow '{key}' op: '{}' is a latency outlier", ev.name),
                evidence_summary: format!(
                    "'{}' took {:.1}µs vs a cohort median of {:.1}µs across {} '{key}' ops \
                     ({ratio:.1}× slower, robust z={}){shape_note}.",
                    ev.name,
                    *dur as f64,
                    median,
                    members.len(),
                    if z.is_finite() { format!("{z:.1}") } else { "∞".to_string() },
                ),
                root_cause: format!(
                    "This '{key}' invocation is a statistical outlier — most cohort members run \
                     near {median:.1}µs, so a per-call factor (an unfused fallback path, a host \
                     sync, a cold cache, or a larger input) is inflating this one."
                ),
                evidence: vec![(*ev).clone()],
                suggestion: format!(
                    "Compare '{}'s Args (shapes/device/dtype) against the fast cohort members; \
                     check for an unexpected CPU fallback, an implicit device transfer, or a \
                     first-touch cold path, and pin it to the same fast kernel the cohort uses.",
                    ev.name,
                ),
            });
        }
    }
}

/// Collect, per op cohort, the distinct `shapes` signatures seen.
fn shape_signatures(events: &[TraceEvent]) -> BTreeMap<&str, (usize, Vec<String>, Vec<TraceEvent>)> {
    let mut map: BTreeMap<&str, (usize, Vec<String>, Vec<TraceEvent>)> = BTreeMap::new();
    for ev in events.iter().filter(|e| e.ph == TracePhase::Complete) {
        let Some(sig) = arg_json(ev, "shapes").map(|v| v.to_string()) else {
            continue;
        };
        let entry = map.entry(cohort_key(&ev.name)).or_default();
        entry.0 += 1;
        if !entry.1.contains(&sig) {
            entry.1.push(sig);
            entry.2.push(ev.clone());
        }
    }
    map
}

/// ShapeInstability: one op cohort runs with more than one input shape → the
/// runtime must reallocate buffers each time the shape changes.
fn detect_shape_instability(events: &[TraceEvent], out: &mut Vec<DiagnosedIssue>) {
    for (key, (runs, sigs, evidence)) in shape_signatures(events) {
        if sigs.len() < 2 || runs < 2 {
            continue;
        }
        out.push(DiagnosedIssue {
            severity: Severity::Warning,
            category: IssueCategory::ShapeInstability,
            description: format!("Shape instability on '{key}' (buffer reallocations)"),
            evidence_summary: format!(
                "'{key}' ran {runs} times with {} distinct input shapes: {}.",
                sigs.len(),
                sigs.join(", "),
            ),
            root_cause:
                "Varying input shapes force the runtime to free and reallocate work buffers on \
                 each new shape instead of reusing a stable allocation, adding allocator and \
                 planning latency to every changed call."
                    .to_string(),
            evidence,
            suggestion:
                "Pad inputs to a fixed shape (e.g. a static sequence length / batch size) so the \
                 same buffers are reused, or bucket inputs into a few fixed shape classes to bound \
                 the number of reallocations."
                    .to_string(),
        });
    }
}

/// NoCudaGraph: shapes change across runs, so a CUDA graph cannot be captured.
fn detect_no_cuda_graph(events: &[TraceEvent], out: &mut Vec<DiagnosedIssue>) {
    // Whole-trace shape stability: how many distinct shape signatures appear
    // among ops that recur (a graph needs *all* shapes stable).
    let sigs = shape_signatures(events);
    let mut changing = 0usize;
    let mut total_distinct = 0usize;
    let mut evidence = Vec::new();
    for (runs, s, ev) in sigs.values() {
        if *runs >= 2 && s.len() >= 2 {
            changing += 1;
            total_distinct += s.len();
            evidence.extend(ev.iter().cloned());
        }
    }
    if changing == 0 {
        return;
    }
    out.push(DiagnosedIssue {
        severity: Severity::Info,
        category: IssueCategory::NoCudaGraph,
        description: "CUDA graph not captured".to_string(),
        evidence_summary: format!(
            "{changing} op(s) changed shape across runs ({total_distinct} distinct shapes total); \
             a CUDA graph requires every op's shape to stay fixed."
        ),
        root_cause:
            "Variable-length / variable-shape inputs change kernel launch parameters between runs, \
             so the launch sequence cannot be recorded once and replayed as a graph."
                .to_string(),
        evidence,
        suggestion:
            "Stabilise shapes with padding (fixed sequence length / batch size) or set the batch \
             dimension to static (e.g. vmap.batch_size=static) so the graph can be captured once \
             and replayed."
                .to_string(),
    });
}

/// MemoryThrashing: many memory-pressure/eviction events packed into a short
/// window. *Data-conditional*: fires only when the trace carries such events
/// (category `"memory"` or a name/category mentioning evict/pressure/spill/
/// thrash) — until memory-tier tracing lands these simply won't appear.
fn detect_memory_thrashing(
    events: &[TraceEvent],
    cfg: &DiagnosisConfig,
    out: &mut Vec<DiagnosedIssue>,
) {
    let mut pressure: Vec<&TraceEvent> = events
        .iter()
        .filter(|e| is_pressure_event(e))
        .collect();
    if pressure.len() < cfg.thrash_min_events {
        return;
    }
    pressure.sort_by_key(|e| e.ts);

    // Sliding window over timestamps.
    let mut best: &[&TraceEvent] = &[];
    let mut lo = 0;
    for hi in 0..pressure.len() {
        while pressure[hi].ts.saturating_sub(pressure[lo].ts) > cfg.thrash_window_us {
            lo += 1;
        }
        if hi - lo + 1 > best.len() {
            best = &pressure[lo..=hi];
        }
    }
    if best.len() < cfg.thrash_min_events {
        return;
    }
    let span_us = best.last().unwrap().ts.saturating_sub(best[0].ts);
    out.push(DiagnosedIssue {
        severity: Severity::Warning,
        category: IssueCategory::MemoryThrashing,
        description: "Memory thrashing detected".to_string(),
        evidence_summary: format!(
            "{} memory-pressure events within {:.0}ms — the same regions are being evicted and \
             re-fetched.",
            best.len(),
            span_us as f64 / 1_000.0,
        ),
        root_cause:
            "Live working set exceeds the available memory tier, so recently-evicted data is \
             immediately needed again and refetched — competing allocations (e.g. KV cache vs \
             weights) churn the last free capacity."
                .to_string(),
        evidence: best.iter().map(|e| (*e).clone()).collect(),
        suggestion:
            "Raise the memory budget (e.g. memory.vram_limit_gb) or shrink the working set (reduce \
             context length / batch, or move a competing allocation to another tier) so the hot \
             set fits and stops being evicted."
                .to_string(),
    });
}

/// PrefetchStall: transfer time is not hidden by compute. *Data-conditional*:
/// fires only when the trace has both `"transfer"` and `"compute"` category
/// spans (i.e. once memory-transfer tracing is present).
fn detect_prefetch_stall(events: &[TraceEvent], out: &mut Vec<DiagnosedIssue>) {
    let mut transfer_us: u64 = 0;
    let mut compute_us: u64 = 0;
    let mut worst: Vec<(&TraceEvent, u64)> = Vec::new();
    for (ev, dur) in completed(events) {
        match ev.cat.as_str() {
            "transfer" | "memcpy" | "prefetch" => {
                transfer_us += dur;
                worst.push((ev, dur));
            }
            "compute" => compute_us += dur,
            _ => {}
        }
    }
    if transfer_us == 0 || compute_us == 0 || transfer_us <= compute_us {
        return;
    }
    worst.sort_by_key(|b| std::cmp::Reverse(b.1));
    worst.truncate(3);
    let evidence: Vec<TraceEvent> = worst.iter().map(|(e, _)| (*e).clone()).collect();
    out.push(DiagnosedIssue {
        severity: Severity::Warning,
        category: IssueCategory::PrefetchStall,
        description: "Prefetch stall — transfers not hidden by compute".to_string(),
        evidence_summary: format!(
            "Total transfer time {:.1}µs exceeds total compute time {:.1}µs, so compute waits on \
             data movement instead of overlapping it.",
            transfer_us as f64, compute_us as f64,
        ),
        root_cause:
            "Per-region transfer latency is larger than the compute it should overlap, so prefetch \
             cannot hide it and the pipeline stalls waiting for weights/activations to arrive."
                .to_string(),
        evidence,
        suggestion:
            "Keep the hottest layers' weights resident on the compute device (placement hint \
             \"force\") so they are not transferred each step, or start the prefetch earlier / \
             increase overlap depth so transfers finish before their consumer runs."
                .to_string(),
    });
}

fn is_pressure_event(e: &TraceEvent) -> bool {
    let hay = |s: &str| {
        let s = s.to_ascii_lowercase();
        s.contains("evict")
            || s.contains("pressure")
            || s.contains("spill")
            || s.contains("thrash")
            || s.contains("oom")
    };
    e.cat.eq_ignore_ascii_case("memory") && (hay(&e.name) || hay(&e.cat)) || hay(&e.name)
}

// --- MissedFastPath ----------------------------------------------------------

/// MissedFastPath: an op had an optimized kernel available but the runtime ran
/// the generic fallback. *Data-conditional*: fires only for events a kernel/EP
/// explicitly tagged with the missed-fast-path contract
/// ([`ARG_FASTPATH_REJECTED_REASON`] + [`ARG_OPTIMIZED_CANDIDATE`], normally via
/// [`report_missed_fastpath`] / [`Args::missed_fastpath`](crate::Args::missed_fastpath)).
///
/// Renders the full RULES.md #1 contract: **WHAT** op + which optimized kernel
/// was skipped, **WHY** (the reason string + the fallback that ran), and
/// **HOW** to hit the fast path (a fix derived from the reason).
fn detect_missed_fastpath(events: &[TraceEvent], out: &mut Vec<DiagnosedIssue>) {
    for ev in events {
        let Some(reason) = arg_str(ev, ARG_FASTPATH_REJECTED_REASON) else {
            continue;
        };
        let candidate = arg_str(ev, ARG_OPTIMIZED_CANDIDATE).unwrap_or("an optimized kernel");
        let chosen = arg_str(ev, ARG_CHOSEN_KERNEL);
        let node = if ev.name.is_empty() { "op" } else { ev.name.as_str() };

        let (severity, fix) = fastpath_fix(reason, candidate);

        let chosen_note = chosen
            .map(|c| format!(" the runtime ran the '{c}' fallback instead."))
            .unwrap_or_else(|| " the runtime ran a generic fallback instead.".to_string());

        out.push(DiagnosedIssue {
            severity,
            category: IssueCategory::MissedFastPath,
            description: format!(
                "Missed fast path on '{node}': '{candidate}' available but not used"
            ),
            evidence_summary: format!(
                "'{node}' could have used the optimized '{candidate}' kernel, but{chosen_note} \
                 Reason the fast path was rejected: {reason}."
            ),
            root_cause: format!(
                "The runtime evaluated '{candidate}' for '{node}' and rejected it because: \
                 {reason}. A rejected fast path means the op fell back to a slower generic \
                 kernel that does the same math with more time/memory."
            ),
            evidence: vec![ev.clone()],
            suggestion: fix,
        });
    }
}

/// Map a fast-path rejection reason to a severity and an actionable fix.
///
/// Known reason keywords get a tailored fix; anything else falls back to a
/// generic-but-actionable suggestion so the finding is never a dead end.
/// Reasons indicating the input is simply below a fast-path threshold are
/// [`Severity::Info`] (expected/benign); everything else is a
/// [`Severity::Warning`] the user can usually act on.
fn fastpath_fix(reason: &str, candidate: &str) -> (Severity, String) {
    let r = reason.to_ascii_lowercase();
    if r.contains("threshold") || r.contains("too small") || r.contains("below") {
        return (
            Severity::Info,
            format!(
                "Expected: the input is below '{candidate}'s fast-path threshold, so the generic \
                 kernel is the right choice here. Batch or grow the workload to cross the \
                 threshold if you want '{candidate}' to engage."
            ),
        );
    }
    let fix = if r.contains("dtype") || r.contains("precision") || r.contains("fp32") {
        format!(
            "'{candidate}' supports a narrower dtype set. Cast the op's inputs to a supported \
             dtype (commonly fp16/bf16) — e.g. run the model in half precision — so '{candidate}' \
             can take the inputs."
        )
    } else if r.contains("head_dim") || r.contains("multiple") || r.contains("align") {
        format!(
            "'{candidate}' needs an aligned shape (e.g. head_dim a multiple of 8). Pad the \
             offending dimension up to the required alignment so '{candidate}' accepts it."
        )
    } else if r.contains("mask") {
        format!(
            "'{candidate}' does not support this mask form. Use a supported mask (e.g. a plain \
             causal/boolean mask) or fold the mask into the score bias so '{candidate}' applies."
        )
    } else if r.contains("ep not enabled") || r.contains("provider") || r.contains("disabled") {
        format!(
            "'{candidate}' lives on an EP that is not enabled. Enable the execution provider that \
             owns '{candidate}' (and confirm the device/build supports it) so the op is placed on \
             the fast path."
        )
    } else {
        format!(
            "Adjust the op so it satisfies '{candidate}'s requirements ({reason}); compare its \
             Args (dtype/shapes/mask/device) against a call that does hit '{candidate}'."
        )
    };
    (Severity::Warning, fix)
}

/// Report — from a kernel or EP — that an **optimized kernel existed** for an op
/// but was **not** selected, so [`AutoDiagnosis`] can surface a
/// [`MissedFastPath`](IssueCategory::MissedFastPath) issue (§46.6).
///
/// This is the one-call emission API kernel authors should use at the point
/// they decide to fall back. It emits an instant `"fastpath"` event named after
/// the node, carrying the [`missed_fastpath`](crate::Args::missed_fastpath)
/// contract args. No-op when `ctx` is disabled.
///
/// * `node` — the op/node name (e.g. `"FusedAttention_3"`).
/// * `candidate` — the optimized kernel that could have run (e.g.
///   `"FlashAttention"`).
/// * `reason` — **why** it was skipped (e.g. `"unsupported dtype fp16"`,
///   `"head_dim=100 not a multiple of 8"`, `"mask shape not supported"`,
///   `"EP not enabled"`, `"shape too small — below fast-path threshold"`).
/// * `chosen` — the fallback kernel that actually ran (e.g. `"generic f32 SDPA"`).
///
/// # Example
///
/// ```
/// use onnx_runtime_tracer::TraceContext;
/// use onnx_runtime_tracer::diagnose::{report_missed_fastpath, AutoDiagnosis, IssueCategory};
///
/// let (ctx, mem) = TraceContext::in_memory();
/// // At a kernel's fallback point:
/// report_missed_fastpath(
///     &ctx,
///     "FusedAttention_3",
///     "FlashAttention",
///     "unsupported dtype fp32 (fast path is fp16/bf16 only)",
///     "generic f32 SDPA",
/// );
/// let diag = AutoDiagnosis::analyze(&mem.events());
/// assert!(diag.issues.iter().any(|i| i.category == IssueCategory::MissedFastPath));
/// ```
///
/// # Follow-up seams (real emission sites still to wire)
///
/// The tracer is not yet threaded into the CPU/CUDA kernels (§48.5), so the
/// concrete fallback points below should call this helper once a `TraceContext`
/// is available to them:
///
/// * `onnx-runtime-ep-cpu` `FusedAttentionKernel` — currently f32-only (it
///   `to_dense_f32`-converts): for fp16/bf16 inputs it should report the
///   FlashAttention/fused-SDPA fast path as rejected with reason
///   `"unsupported dtype"`.
/// * `onnx-runtime-ep-cpu` MatMul/GEMM backend selection (`CpuBackend`) — when
///   it falls back from the SIMD GEMM to the `Generic` blocked kernel.
/// * `onnx-runtime-ep-cuda` attention selection — when a masked/odd-shaped
///   attention cannot use the fused cuBLAS/flash path.
pub fn report_missed_fastpath(
    ctx: &crate::TraceContext,
    node: impl Into<String>,
    candidate: impl Into<String>,
    reason: impl Into<String>,
    chosen: impl Into<String>,
) {
    if !ctx.is_enabled() {
        return;
    }
    let args = crate::Args::new().missed_fastpath(candidate, reason, chosen);
    ctx.instant(node, "fastpath", Some(args));
}

// --- Args helpers ------------------------------------------------------------

fn arg_json<'a>(ev: &'a TraceEvent, key: &str) -> Option<&'a serde_json::Value> {
    ev.args.as_ref()?.get(key)
}

fn arg_u64(ev: &TraceEvent, key: &str) -> Option<u64> {
    arg_json(ev, key)?.as_u64()
}

fn arg_str<'a>(ev: &'a TraceEvent, key: &str) -> Option<&'a str> {
    arg_json(ev, key)?.as_str()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::Args;

    fn ev(name: &str, dur_us: u64, args: Option<Args>) -> TraceEvent {
        TraceEvent {
            name: name.to_string(),
            cat: "compute".to_string(),
            ph: TracePhase::Complete,
            ts: 0,
            dur: Some(dur_us),
            pid: 1,
            tid: 1,
            scope: None,
            args: args.map(Args::into_value),
        }
    }

    // ---- Roofline: compute-bound -------------------------------------------

    #[test]
    fn roofline_compute_bound_healthy() {
        // Big GEMM: 16 GFLOP over 4 MB → AI ~ 4096 FLOP/byte, well above ridge.
        let analyzer = RooflineAnalyzer::h100();
        let sample = KernelSample {
            kernel_name: "MatMul_0".to_string(),
            node_id: Some(0),
            flops: Some(16_000_000_000),
            bytes_accessed: Some(4_000_000),
            duration_ns: Some(30_000), // 30µs
            precision: Precision::Fp16,
        };
        let r = analyzer.analyze(&sample);
        assert_eq!(r.bound, BoundType::ComputeBound);
        assert!(r.arithmetic_intensity.unwrap() > 1000.0);
        assert!(r.achieved_tflops.unwrap() > 0.0);
        let s = r.suggestion.unwrap();
        assert!(!s.is_empty());
        assert!(s.to_lowercase().contains("compute-bound"));
    }

    #[test]
    fn roofline_compute_bound_low_efficiency_suggests_occupancy() {
        let analyzer = RooflineAnalyzer::h100();
        // High AI but very long duration → low efficiency.
        let sample = KernelSample {
            kernel_name: "MatMul_slow".to_string(),
            node_id: None,
            flops: Some(16_000_000_000),
            bytes_accessed: Some(4_000_000),
            duration_ns: Some(5_000_000), // 5ms → poor throughput
            precision: Precision::Fp16,
        };
        let r = analyzer.analyze(&sample);
        assert_eq!(r.bound, BoundType::ComputeBound);
        assert!(r.efficiency.unwrap() < 0.5);
        let s = r.suggestion.unwrap().to_lowercase();
        assert!(s.contains("occupancy") || s.contains("tensor core"));
    }

    // ---- Roofline: memory-bound --------------------------------------------

    #[test]
    fn roofline_memory_bound_suggests_fusion() {
        let analyzer = RooflineAnalyzer::h100();
        // LayerNorm-ish: few FLOPs, lots of bytes → AI ~ 0.5, below ridge.
        let sample = KernelSample {
            kernel_name: "LayerNorm_3".to_string(),
            node_id: Some(3),
            flops: Some(50_000_000),
            bytes_accessed: Some(100_000_000),
            duration_ns: Some(60_000),
            precision: Precision::Fp16,
        };
        let r = analyzer.analyze(&sample);
        assert_eq!(r.bound, BoundType::MemoryBound);
        assert!(r.arithmetic_intensity.unwrap() < 1.0);
        let s = r.suggestion.unwrap().to_lowercase();
        assert!(s.contains("fuse") || s.contains("quant") || s.contains("in-place"));
    }

    // ---- Roofline: launch-bound (tiny kernel) ------------------------------

    #[test]
    fn roofline_launch_bound_tiny_kernel_with_metrics() {
        let analyzer = RooflineAnalyzer::h100();
        let sample = KernelSample {
            kernel_name: "Add_7".to_string(),
            node_id: Some(7),
            flops: Some(1_000),
            bytes_accessed: Some(4_000),
            duration_ns: Some(800), // < 5µs threshold
            precision: Precision::Fp16,
        };
        let r = analyzer.analyze(&sample);
        assert_eq!(r.bound, BoundType::LaunchBound);
        let s = r.suggestion.unwrap().to_lowercase();
        assert!(s.contains("fuse") || s.contains("cuda graph"));
    }

    #[test]
    fn roofline_launch_bound_from_duration_alone() {
        // Graceful degradation: no FLOP/byte metrics, but short duration → still
        // confidently launch-bound.
        let analyzer = RooflineAnalyzer::h100();
        let sample = KernelSample {
            kernel_name: "Add_7".to_string(),
            node_id: None,
            flops: None,
            bytes_accessed: None,
            duration_ns: Some(900),
            precision: Precision::Fp16,
        };
        let r = analyzer.analyze(&sample);
        assert_eq!(r.bound, BoundType::LaunchBound);
        assert!(r.arithmetic_intensity.is_none());
        assert!(r.suggestion.unwrap().to_lowercase().contains("launch"));
    }

    // ---- Roofline: graceful degradation edge cases -------------------------

    #[test]
    fn roofline_zero_bytes_does_not_panic_and_is_indeterminate() {
        let analyzer = RooflineAnalyzer::h100();
        let sample = KernelSample {
            kernel_name: "Weird".to_string(),
            node_id: None,
            flops: Some(1_000_000),
            bytes_accessed: Some(0), // zero bytes must not divide-by-zero
            duration_ns: Some(50_000),
            precision: Precision::Fp16,
        };
        let r = analyzer.analyze(&sample);
        assert_eq!(r.bound, BoundType::Indeterminate);
        assert!(r.efficiency.is_none());
        assert!(r.suggestion.unwrap().to_lowercase().contains("insufficient metrics"));
    }

    #[test]
    fn roofline_zero_duration_is_indeterminate() {
        let analyzer = RooflineAnalyzer::h100();
        let sample = KernelSample {
            kernel_name: "Weird2".to_string(),
            node_id: None,
            flops: Some(1_000_000),
            bytes_accessed: Some(1_000_000),
            duration_ns: Some(0),
            precision: Precision::Fp16,
        };
        let r = analyzer.analyze(&sample);
        assert_eq!(r.bound, BoundType::Indeterminate);
        assert!(r.suggestion.unwrap().contains("Insufficient metrics"));
    }

    #[test]
    fn roofline_no_metrics_at_all_is_indeterminate() {
        let analyzer = RooflineAnalyzer::h100();
        let sample = KernelSample {
            kernel_name: "Bare".to_string(),
            ..Default::default()
        };
        let r = analyzer.analyze(&sample);
        assert_eq!(r.bound, BoundType::Indeterminate);
        let s = r.suggestion.unwrap();
        assert!(s.contains("FLOP") && s.contains("bytes"));
    }

    #[test]
    fn roofline_from_event_reads_args() {
        let analyzer = RooflineAnalyzer::h100();
        let event = ev(
            "MatMul_0",
            30, // 30µs
            Some(Args::new().flops(16_000_000_000).bytes(4_000_000).with("node_id", 0_u64)),
        );
        let r = analyzer.analyze_event(&event);
        assert_eq!(r.bound, BoundType::ComputeBound);
        assert_eq!(r.node_id, Some(0));
    }

    #[test]
    fn roofline_never_panics_on_bad_ceilings() {
        // Non-positive ceilings are clamped; must not panic.
        let analyzer = RooflineAnalyzer::new(0.0, -1.0, f64::NAN);
        let sample = KernelSample {
            kernel_name: "X".to_string(),
            node_id: None,
            flops: Some(1_000),
            bytes_accessed: Some(1_000),
            duration_ns: Some(50_000),
            precision: Precision::Fp32,
        };
        let r = analyzer.analyze(&sample);
        assert!(r.efficiency.is_some());
    }

    #[test]
    fn roofline_report_renders_and_is_nonempty() {
        let analyzer = RooflineAnalyzer::h100();
        let results = vec![
            analyzer.analyze(&KernelSample {
                kernel_name: "MatMul_0".to_string(),
                node_id: Some(0),
                flops: Some(16_000_000_000),
                bytes_accessed: Some(4_000_000),
                duration_ns: Some(30_000),
                precision: Precision::Fp16,
            }),
            analyzer.analyze(&KernelSample {
                kernel_name: "Add_7".to_string(),
                node_id: None,
                flops: None,
                bytes_accessed: None,
                duration_ns: Some(900),
                precision: Precision::Fp16,
            }),
        ];
        let report = render_roofline_report(&results);
        assert!(report.contains("MatMul_0"));
        assert!(report.contains("compute-bound"));
        assert!(report.contains("launch-bound"));
        assert!(report.contains("→"));
    }

    // ---- AutoDiagnosis: slow op outlier ------------------------------------

    #[test]
    fn diagnosis_detects_slow_op_outlier() {
        let mut events: Vec<TraceEvent> = (0..8)
            .map(|i| ev(&format!("MatMul_{i}"), 40, None))
            .collect();
        // One egregiously slow instance.
        events.push(ev("MatMul_99", 900, Some(Args::new().device("cpu"))));

        let d = AutoDiagnosis::analyze(&events);
        let slow: Vec<_> = d
            .issues
            .iter()
            .filter(|i| i.category == IssueCategory::SlowOp)
            .collect();
        assert_eq!(slow.len(), 1);
        let issue = slow[0];
        assert_eq!(issue.severity, Severity::Critical); // 900/40 = 22.5× ≥ 10
        assert!(!issue.suggestion.is_empty());
        assert!(issue.evidence_summary.contains("outlier") || issue.evidence_summary.contains("slower"));
        assert_eq!(issue.evidence.len(), 1);
        assert_eq!(issue.evidence[0].name, "MatMul_99");
        // Renders the WHAT/WHY/HOW contract.
        let r = issue.render();
        assert!(r.contains("Evidence:") && r.contains("Root cause:") && r.contains("Fix:"));
    }

    #[test]
    fn diagnosis_no_false_positive_on_uniform_cohort() {
        let events: Vec<TraceEvent> = (0..8).map(|i| ev(&format!("Op_{i}"), 50, None)).collect();
        let d = AutoDiagnosis::analyze(&events);
        assert!(d.issues.iter().all(|i| i.category != IssueCategory::SlowOp));
    }

    #[test]
    fn diagnosis_ignores_tiny_cohort() {
        // Only 2 members → below min_cohort, no outlier stats.
        let events = vec![ev("Op_0", 10, None), ev("Op_1", 500, None)];
        let d = AutoDiagnosis::analyze(&events);
        assert!(d.issues.iter().all(|i| i.category != IssueCategory::SlowOp));
    }

    // ---- AutoDiagnosis: shape instability / no cuda graph ------------------

    #[test]
    fn diagnosis_detects_shape_instability_and_no_cuda_graph() {
        let events = vec![
            ev("MatMul_0", 40, Some(Args::new().shapes(vec![vec![1_i64, 128, 4096]]))),
            ev("MatMul_0", 41, Some(Args::new().shapes(vec![vec![1_i64, 130, 4096]]))),
            ev("MatMul_0", 42, Some(Args::new().shapes(vec![vec![1_i64, 200, 4096]]))),
        ];
        let d = AutoDiagnosis::analyze(&events);
        assert!(d.issues.iter().any(|i| i.category == IssueCategory::ShapeInstability));

        let ncg = d
            .issues
            .iter()
            .find(|i| i.category == IssueCategory::NoCudaGraph)
            .expect("should flag no-cuda-graph");
        assert_eq!(ncg.severity, Severity::Info);
        assert!(ncg.suggestion.to_lowercase().contains("pad") || ncg.suggestion.contains("static"));
    }

    #[test]
    fn diagnosis_stable_shapes_no_instability() {
        let events: Vec<TraceEvent> = (0..3)
            .map(|_| ev("MatMul_0", 40, Some(Args::new().shapes(vec![vec![1_i64, 128, 4096]]))))
            .collect();
        let d = AutoDiagnosis::analyze(&events);
        assert!(
            d.issues
                .iter()
                .all(|i| i.category != IssueCategory::ShapeInstability
                    && i.category != IssueCategory::NoCudaGraph)
        );
    }

    // ---- AutoDiagnosis: memory thrashing -----------------------------------

    #[test]
    fn diagnosis_detects_memory_thrashing() {
        let mut events = Vec::new();
        for i in 0..12 {
            events.push(TraceEvent {
                name: "page_evict".to_string(),
                cat: "memory".to_string(),
                ph: TracePhase::Instant,
                ts: i * 20_000, // within a 500ms window
                dur: None,
                pid: 1,
                tid: 1,
                scope: Some('p'),
                args: None,
            });
        }
        let d = AutoDiagnosis::analyze(&events);
        let thr = d
            .issues
            .iter()
            .find(|i| i.category == IssueCategory::MemoryThrashing)
            .expect("should detect thrashing");
        assert_eq!(thr.severity, Severity::Warning);
        assert!(thr.suggestion.to_lowercase().contains("vram") || thr.suggestion.contains("budget"));
        assert!(thr.evidence.len() >= 8);
    }

    // ---- AutoDiagnosis: prefetch stall -------------------------------------

    #[test]
    fn diagnosis_detects_prefetch_stall() {
        let transfer = |dur| TraceEvent {
            name: "H2D".to_string(),
            cat: "transfer".to_string(),
            ph: TracePhase::Complete,
            ts: 0,
            dur: Some(dur),
            pid: 1,
            tid: 2,
            scope: None,
            args: None,
        };
        let events = vec![
            transfer(1_300),
            transfer(1_300),
            ev("MatMul_0", 500, None),
            ev("MatMul_1", 500, None),
        ];
        let d = AutoDiagnosis::analyze(&events);
        assert!(d.issues.iter().any(|i| i.category == IssueCategory::PrefetchStall));
    }

    // ---- Report / health ----------------------------------------------------

    #[test]
    fn diagnosis_healthy_report() {
        let events: Vec<TraceEvent> = (0..8).map(|i| ev(&format!("Op_{i}"), 50, None)).collect();
        let d = AutoDiagnosis::analyze(&events);
        assert!(d.is_healthy());
        assert!(d.report().contains("healthy"));
    }

    #[test]
    fn diagnosis_report_orders_critical_first() {
        let mut events: Vec<TraceEvent> = (0..8)
            .map(|i| ev(&format!("MatMul_{i}"), 40, None))
            .collect();
        events.push(ev("MatMul_99", 900, None)); // critical slow op
        events.push(ev("LayerNorm_0", 40, Some(Args::new().shapes(vec![vec![1_i64, 1]]))));
        events.push(ev("LayerNorm_0", 40, Some(Args::new().shapes(vec![vec![1_i64, 2]]))));

        let d = AutoDiagnosis::analyze(&events);
        assert!(d.issues.len() >= 2);
        assert_eq!(d.issues[0].severity, Severity::Critical);
    }

    #[test]
    fn cohort_key_strips_numeric_suffix() {
        assert_eq!(cohort_key("MatMul_0"), "MatMul");
        assert_eq!(cohort_key("MatMul_123"), "MatMul");
        assert_eq!(cohort_key("Softmax"), "Softmax");
        assert_eq!(cohort_key("layer_norm_final"), "layer_norm_final");
        assert_eq!(cohort_key("_5"), "_5");
    }

    #[test]
    fn precision_parse_is_lenient() {
        assert_eq!(Precision::from_str_lenient("fp32"), Precision::Fp32);
        assert_eq!(Precision::from_str_lenient("FLOAT"), Precision::Fp32);
        assert_eq!(Precision::from_str_lenient("bf16"), Precision::Fp16);
        assert_eq!(Precision::from_str_lenient("garbage"), Precision::Fp16);
    }

    // ---- MissedFastPath -----------------------------------------------------

    /// Build an instant-phase event carrying the missed-fast-path contract args,
    /// as `report_missed_fastpath` would emit.
    fn fastpath_ev(name: &str, candidate: &str, reason: &str, chosen: &str) -> TraceEvent {
        TraceEvent {
            name: name.to_string(),
            cat: "fastpath".to_string(),
            ph: TracePhase::Instant,
            ts: 0,
            dur: None,
            pid: 1,
            tid: 1,
            scope: Some('t'),
            args: Some(Args::new().missed_fastpath(candidate, reason, chosen).into_value()),
        }
    }

    #[test]
    fn missed_fastpath_renders_full_contract() {
        let events = vec![fastpath_ev(
            "FusedAttention_3",
            "FlashAttention",
            "unsupported dtype fp32 (fast path is fp16/bf16 only)",
            "generic f32 SDPA",
        )];
        let d = AutoDiagnosis::analyze(&events);
        let issue = d
            .issues
            .iter()
            .find(|i| i.category == IssueCategory::MissedFastPath)
            .expect("MissedFastPath issue expected");
        assert_eq!(issue.severity, Severity::Warning);

        let rendered = issue.render();
        // WHAT: names the node and the optimized kernel that was skipped.
        assert!(rendered.contains("FusedAttention_3"));
        assert!(rendered.contains("FlashAttention"));
        // WHY: the reason + the fallback that ran.
        assert!(rendered.contains("unsupported dtype fp32"));
        assert!(rendered.contains("generic f32 SDPA"));
        // HOW: a dtype-specific, actionable fix.
        assert!(rendered.to_lowercase().contains("cast"));
        assert!(rendered.to_lowercase().contains("dtype"));
    }

    #[test]
    fn missed_fastpath_alignment_reason_suggests_padding() {
        let events = vec![fastpath_ev(
            "Attention_7",
            "FusedSDPA",
            "head_dim=100 not a multiple of 8",
            "generic attention",
        )];
        let d = AutoDiagnosis::analyze(&events);
        let issue = d
            .issues
            .iter()
            .find(|i| i.category == IssueCategory::MissedFastPath)
            .expect("MissedFastPath issue expected");
        assert_eq!(issue.severity, Severity::Warning);
        assert!(issue.suggestion.to_lowercase().contains("pad"));
    }

    #[test]
    fn missed_fastpath_below_threshold_is_info() {
        let events = vec![fastpath_ev(
            "MatMul_1",
            "SIMD GEMM",
            "shape too small — below fast-path threshold",
            "generic blocked GEMM",
        )];
        let d = AutoDiagnosis::analyze(&events);
        let issue = d
            .issues
            .iter()
            .find(|i| i.category == IssueCategory::MissedFastPath)
            .expect("MissedFastPath issue expected");
        // Below-threshold fallback is expected/benign, not a warning.
        assert_eq!(issue.severity, Severity::Info);
        assert!(issue.suggestion.to_lowercase().contains("threshold"));
    }

    #[test]
    fn missed_fastpath_helper_emits_detectable_event() {
        let (ctx, mem) = crate::TraceContext::in_memory();
        report_missed_fastpath(
            &ctx,
            "FusedAttention_0",
            "FlashAttention",
            "EP not enabled",
            "generic f32 SDPA",
        );
        let d = AutoDiagnosis::analyze(&mem.events());
        let issue = d
            .issues
            .iter()
            .find(|i| i.category == IssueCategory::MissedFastPath)
            .expect("MissedFastPath issue expected from helper");
        assert!(issue.suggestion.to_lowercase().contains("enable"));
    }

    #[test]
    fn missed_fastpath_helper_is_noop_when_disabled() {
        let ctx = crate::TraceContext::noop();
        let (_ctx2, mem) = crate::TraceContext::in_memory();
        // Disabled context records nothing.
        report_missed_fastpath(&ctx, "Op", "Fast", "reason", "fallback");
        assert!(AutoDiagnosis::analyze(&mem.events()).is_healthy());
    }

    #[test]
    fn no_missed_fastpath_without_reason_arg() {
        // A plain op (no fastpath_rejected_reason) must not raise the issue.
        let events = vec![ev("MatMul_0", 50, Some(Args::new().device("cpu")))];
        let d = AutoDiagnosis::analyze(&events);
        assert!(!d.issues.iter().any(|i| i.category == IssueCategory::MissedFastPath));
    }
}
