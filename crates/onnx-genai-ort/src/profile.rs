//! Env-gated per-stage decode profiler.
//!
//! Enabled with `ONNX_GENAI_PROFILE=1`. When disabled every entry point is a
//! single relaxed-atomic load plus an early return, so production paths pay no
//! measurable cost. When enabled, [`Span`] accumulates wall-clock nanoseconds
//! and a call count per named stage into a process-global registry that
//! [`report`] renders as a table.
//!
//! This exists to answer one question: for each generated token, how much wall
//! time is spent inside the ORT kernels (`session.run`) versus our own
//! orchestration (tensor binding, KV rotation, logits copy, sampling,
//! detokenization). See `docs/benchmarks` and the CPU profiling decision note.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Returns whether profiling is enabled, reading `ONNX_GENAI_PROFILE` once.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("ONNX_GENAI_PROFILE").ok().as_deref(),
            Some("1") | Some("true") | Some("yes")
        )
    })
}

#[derive(Default, Clone, Copy)]
struct StageStat {
    total_ns: u128,
    count: u64,
}

fn registry() -> &'static Mutex<BTreeMap<&'static str, StageStat>> {
    static REGISTRY: OnceLock<Mutex<BTreeMap<&'static str, StageStat>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Add a measured duration for `stage`. No-op unless profiling is enabled.
pub fn record(stage: &'static str, nanos: u128) {
    if !enabled() {
        return;
    }
    if let Ok(mut reg) = registry().lock() {
        let entry = reg.entry(stage).or_default();
        entry.total_ns += nanos;
        entry.count += 1;
    }
}

/// A scoped timer that records its lifetime to `stage` on drop.
pub struct Span {
    stage: &'static str,
    start: Instant,
    active: bool,
}

impl Span {
    /// Start a span. Cheap and inert when profiling is disabled.
    #[must_use]
    pub fn new(stage: &'static str) -> Self {
        Self {
            stage,
            start: Instant::now(),
            active: enabled(),
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if self.active {
            record(self.stage, self.start.elapsed().as_nanos());
        }
    }
}

/// Open a profiling [`Span`] for the given static stage name.
#[macro_export]
macro_rules! prof_span {
    ($stage:expr) => {
        $crate::profile::Span::new($stage)
    };
}

/// Clear all accumulated stage statistics.
pub fn reset() {
    if let Ok(mut reg) = registry().lock() {
        reg.clear();
    }
}

/// Render the accumulated per-stage statistics as a text table.
///
/// `tokens` scales the per-token column; pass the number of generated tokens.
pub fn report(tokens: u64) -> String {
    let reg = match registry().lock() {
        Ok(reg) => reg,
        Err(_) => return String::from("<profiler registry poisoned>"),
    };
    let mut rows: Vec<(&'static str, StageStat)> =
        reg.iter().map(|(name, stat)| (*name, *stat)).collect();
    rows.sort_by_key(|row| std::cmp::Reverse(row.1.total_ns));

    let tokens = tokens.max(1);
    let mut out = String::new();
    out.push_str(&format!(
        "{:<26} {:>12} {:>10} {:>14} {:>12}\n",
        "stage", "total_ms", "calls", "us/call", "us/token"
    ));
    out.push_str(&format!("{}\n", "-".repeat(78)));
    for (name, stat) in &rows {
        let total_ms = stat.total_ns as f64 / 1_000_000.0;
        let us_per_call = if stat.count > 0 {
            (stat.total_ns as f64 / 1_000.0) / stat.count as f64
        } else {
            0.0
        };
        let us_per_token = (stat.total_ns as f64 / 1_000.0) / tokens as f64;
        out.push_str(&format!(
            "{:<26} {:>12.3} {:>10} {:>14.2} {:>12.2}\n",
            name, total_ms, stat.count, us_per_call, us_per_token
        ));
    }
    out
}
