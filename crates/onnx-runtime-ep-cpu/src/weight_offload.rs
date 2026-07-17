//! Huge-model weight-offload mode and lightweight process-wide observability.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use onnx_runtime_ep_api::ExternalMmapRegion;

/// Environment switch for the route-first mmap MoE path.
pub const WEIGHT_OFFLOAD_ENV: &str = "ONNX_GENAI_WEIGHT_OFFLOAD";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WeightOffloadMode {
    pub enabled: bool,
}

impl WeightOffloadMode {
    pub fn from_env() -> Self {
        Self::from_value(std::env::var_os(WEIGHT_OFFLOAD_ENV).as_deref())
    }

    fn from_value(value: Option<&OsStr>) -> Self {
        Self {
            enabled: value.is_some_and(|value| value == "1"),
        }
    }
}

/// Best-effort Linux process memory/page-fault counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinuxProcessMemoryStats {
    pub resident_rss_bytes: u64,
    pub minor_faults: u64,
    pub major_faults: u64,
}

/// Snapshot of route-first weight-offload activity.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WeightOffloadStats {
    pub mapped_bytes: u64,
    pub bytes_read_from_mmap: u64,
    pub layer_executions: u64,
    pub active_experts: u64,
    pub unique_experts_per_batch: u64,
    pub peak_dequantized_experts: u64,
    pub routed_tokens: u64,
    pub tokens_per_expert: BTreeMap<usize, u64>,
    pub per_layer: BTreeMap<u32, WeightOffloadLayerStats>,
    pub linux_process: Option<LinuxProcessMemoryStats>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WeightOffloadLayerStats {
    pub executions: u64,
    pub active_experts: u64,
    pub unique_experts: u64,
    pub tokens_per_expert: BTreeMap<usize, u64>,
}

#[derive(Default)]
pub(crate) struct WeightOffloadMetrics {
    mapped_regions: Mutex<MappedRegionState>,
    bytes_read_from_mmap: AtomicU64,
    layer_executions: AtomicU64,
    active_experts: AtomicU64,
    unique_experts_per_batch: AtomicU64,
    peak_dequantized_experts: AtomicU64,
    routed_tokens: AtomicU64,
    tokens_per_expert: Mutex<BTreeMap<usize, u64>>,
    per_layer: Mutex<BTreeMap<u32, WeightOffloadLayerStats>>,
}

#[derive(Default)]
struct MappedRegionState {
    regions: BTreeSet<ExternalMmapRegion>,
    total_bytes: u64,
}

impl WeightOffloadMetrics {
    pub fn record_mapped_regions(
        &self,
        regions: &[ExternalMmapRegion],
    ) -> Result<(), &'static str> {
        let mut state = self
            .mapped_regions
            .lock()
            .expect("weight-offload mapped-region lock poisoned");
        let mut additions = BTreeSet::new();
        let mut total = state.total_bytes;
        for &region in regions {
            let end = region
                .offset
                .checked_add(region.len)
                .ok_or("mapped region endpoint overflow")?;
            if end > isize::MAX as usize {
                return Err("mapped region endpoint exceeds isize::MAX");
            }
            if !state.regions.contains(&region) && additions.insert(region) {
                let len = u64::try_from(region.len).map_err(|_| "mapped region length overflow")?;
                total = total.checked_add(len).ok_or("mapped byte total overflow")?;
            }
        }
        state.regions.extend(additions);
        state.total_bytes = total;
        Ok(())
    }

    pub fn record_bytes_read(&self, bytes: usize) -> Result<(), &'static str> {
        let bytes = u64::try_from(bytes).map_err(|_| "mmap read byte count overflow")?;
        self.bytes_read_from_mmap
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |total| {
                total.checked_add(bytes)
            })
            .map_err(|_| "mmap read byte total overflow")?;
        Ok(())
    }

    pub fn record_dequantized_window(&self, experts: usize) {
        self.peak_dequantized_experts
            .fetch_max(experts as u64, Ordering::Relaxed);
    }

    pub fn record_routes(&self, layer: u32, token_counts: &BTreeMap<usize, usize>) {
        let active = token_counts.values().copied().sum::<usize>();
        self.layer_executions.fetch_add(1, Ordering::Relaxed);
        self.active_experts
            .fetch_add(active as u64, Ordering::Relaxed);
        self.unique_experts_per_batch
            .fetch_add(token_counts.len() as u64, Ordering::Relaxed);
        self.routed_tokens
            .fetch_add(active as u64, Ordering::Relaxed);
        let mut totals = self
            .tokens_per_expert
            .lock()
            .expect("weight-offload metrics lock poisoned");
        for (&expert, &tokens) in token_counts {
            let total = totals.entry(expert).or_default();
            *total = total.saturating_add(tokens as u64);
        }
        drop(totals);

        let mut layers = self
            .per_layer
            .lock()
            .expect("weight-offload layer metrics lock poisoned");
        let layer_stats = layers.entry(layer).or_default();
        layer_stats.executions = layer_stats.executions.saturating_add(1);
        layer_stats.active_experts = layer_stats.active_experts.saturating_add(active as u64);
        layer_stats.unique_experts = layer_stats
            .unique_experts
            .saturating_add(token_counts.len() as u64);
        for (&expert, &tokens) in token_counts {
            let total = layer_stats.tokens_per_expert.entry(expert).or_default();
            *total = total.saturating_add(tokens as u64);
        }
    }

    fn snapshot(&self) -> WeightOffloadStats {
        WeightOffloadStats {
            mapped_bytes: self
                .mapped_regions
                .lock()
                .expect("weight-offload mapped-region lock poisoned")
                .total_bytes,
            bytes_read_from_mmap: self.bytes_read_from_mmap.load(Ordering::Relaxed),
            layer_executions: self.layer_executions.load(Ordering::Relaxed),
            active_experts: self.active_experts.load(Ordering::Relaxed),
            unique_experts_per_batch: self.unique_experts_per_batch.load(Ordering::Relaxed),
            peak_dequantized_experts: self.peak_dequantized_experts.load(Ordering::Relaxed),
            routed_tokens: self.routed_tokens.load(Ordering::Relaxed),
            tokens_per_expert: self
                .tokens_per_expert
                .lock()
                .expect("weight-offload metrics lock poisoned")
                .clone(),
            per_layer: self
                .per_layer
                .lock()
                .expect("weight-offload layer metrics lock poisoned")
                .clone(),
            linux_process: linux_process_memory_stats(),
        }
    }

    #[cfg(test)]
    pub fn reset(&self) {
        *self
            .mapped_regions
            .lock()
            .expect("weight-offload mapped-region lock poisoned") = MappedRegionState::default();
        self.bytes_read_from_mmap.store(0, Ordering::Relaxed);
        self.peak_dequantized_experts.store(0, Ordering::Relaxed);
        self.layer_executions.store(0, Ordering::Relaxed);
        self.active_experts.store(0, Ordering::Relaxed);
        self.unique_experts_per_batch.store(0, Ordering::Relaxed);
        self.routed_tokens.store(0, Ordering::Relaxed);
        self.tokens_per_expert
            .lock()
            .expect("weight-offload metrics lock poisoned")
            .clear();
        self.per_layer
            .lock()
            .expect("weight-offload layer metrics lock poisoned")
            .clear();
    }
}

static METRICS: OnceLock<WeightOffloadMetrics> = OnceLock::new();

pub(crate) fn metrics() -> &'static WeightOffloadMetrics {
    METRICS.get_or_init(WeightOffloadMetrics::default)
}

/// Read current offload counters. The scheduler governor can poll this without
/// depending on kernel internals.
pub fn weight_offload_stats() -> WeightOffloadStats {
    metrics().snapshot()
}

#[cfg(target_os = "linux")]
fn linux_process_memory_stats() -> Option<LinuxProcessMemoryStats> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let resident_rss_bytes = status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(|kib| kib.checked_mul(1024))
        .unwrap_or(0);

    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let fields = stat.get(stat.rfind(')')?.checked_add(2)?..)?;
    let fields = fields.split_whitespace().collect::<Vec<_>>();
    Some(LinuxProcessMemoryStats {
        resident_rss_bytes,
        minor_faults: fields.get(7)?.parse().ok()?,
        major_faults: fields.get(9)?.parse().ok()?,
    })
}

#[cfg(not(target_os = "linux"))]
fn linux_process_memory_stats() -> Option<LinuxProcessMemoryStats> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_offload_flag_is_opt_in() {
        assert!(!WeightOffloadMode::from_value(None).enabled);
        assert!(!WeightOffloadMode::from_value(Some(OsStr::new("0"))).enabled);
        assert!(WeightOffloadMode::from_value(Some(OsStr::new("1"))).enabled);
    }

    #[test]
    fn mapped_bytes_sum_distinct_ranges_across_layers() {
        let metrics = WeightOffloadMetrics::default();
        let first = ExternalMmapRegion {
            mapping_id: 7,
            offset: 0,
            len: 100,
        };
        let second = ExternalMmapRegion {
            mapping_id: 7,
            offset: 100,
            len: 200,
        };
        metrics.record_mapped_regions(&[first]).unwrap();
        metrics.record_mapped_regions(&[second, first]).unwrap();
        assert_eq!(metrics.snapshot().mapped_bytes, 300);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_process_counters_are_best_effort_readable() {
        let stats = linux_process_memory_stats().expect("Linux /proc process counters");
        assert!(stats.resident_rss_bytes > 0);
    }
}
