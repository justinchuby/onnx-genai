//! Per-device rate profile: the **machine-specific** half of the cost model
//! (`docs/architecture/ORT2.md` §6.2 `DeviceProfile`).
//!
//! Everything here is a *rate* — a property of the hardware, not of any kernel.
//! Kernel *structure* (bytes moved, FLOPs) is reported by the EP and lives in
//! [`crate::KernelStructure`]; `cost = structure / rate`. Keeping the two apart
//! is what makes an EP binary correct on any host and makes offline planning
//! natural (issue #995).
//!
//! Two invariants are load-bearing:
//!
//! * **Unknown must be representable** (the #947 lesson). Every rate is
//!   `Option`-shaped: an unmeasured rate is *absent*, never a fabricated
//!   default. A caller that asks for a cost the model cannot compute gets
//!   `None`, not a confident wrong number.
//! * **Memory bandwidth is not a scalar.** Measured DRAM bandwidth on the #995
//!   box spans 10.07→49.28 GB/s between 1 and 20 threads — a 5× spread — so a
//!   single `memory_bandwidth: f64` would be wrong by 5× depending on load.
//!   [`MemoryBandwidth`] therefore stores the whole parallelism curve and every
//!   query names the parallelism it assumes.

use std::collections::BTreeMap;
use std::time::Duration;

use onnx_runtime_ir::DataType;
use serde::{Deserialize, Serialize};

/// Sustained memory bandwidth as a function of the number of concurrently
/// active threads, in **bytes per second**.
///
/// A scalar bandwidth is not a well-defined hardware property: DRAM controllers
/// saturate only under enough memory-level parallelism, so the achievable rate
/// depends on how many threads are streaming at once. This stores the measured
/// curve (thread count → bytes/sec) and requires every consumer to state the
/// parallelism it assumes, so the cost model can be right for both a lightly
/// loaded and a saturated machine instead of wrong for one of them.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MemoryBandwidth {
    /// Measured samples: thread count → bytes/sec. Empty means "unknown".
    samples: BTreeMap<u32, f64>,
}

impl MemoryBandwidth {
    /// An empty (fully unknown) bandwidth curve.
    pub fn unknown() -> Self {
        Self::default()
    }

    /// Build from measured `(threads, bytes_per_sec)` samples (e.g. the
    /// `roofline_bandwidth` probe output).
    pub fn from_samples(samples: impl IntoIterator<Item = (u32, f64)>) -> Self {
        Self {
            samples: samples
                .into_iter()
                .filter(|&(threads, bw)| threads > 0 && bw.is_finite() && bw > 0.0)
                .collect(),
        }
    }

    /// Whether any sample has been recorded.
    pub fn is_known(&self) -> bool {
        !self.samples.is_empty()
    }

    /// Record or overwrite one sample.
    pub fn insert(&mut self, threads: u32, bytes_per_sec: f64) {
        if threads > 0 && bytes_per_sec.is_finite() && bytes_per_sec > 0.0 {
            self.samples.insert(threads, bytes_per_sec);
        }
    }

    /// The measured samples, ascending by thread count.
    pub fn samples(&self) -> impl Iterator<Item = (u32, f64)> + '_ {
        self.samples.iter().map(|(&t, &bw)| (t, bw))
    }

    /// Bandwidth (bytes/sec) at a specific parallelism, or `None` if unknown.
    ///
    /// Exact samples are returned directly; between two measured points the
    /// value is linearly interpolated; outside the measured range it is clamped
    /// to the nearest endpoint (bandwidth is monotone-ish and saturating, so
    /// extrapolating a slope past the ends would invent capacity that was never
    /// measured).
    pub fn at_parallelism(&self, threads: u32) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        if let Some(&bw) = self.samples.get(&threads) {
            return Some(bw);
        }
        let below = self.samples.range(..threads).next_back();
        let above = self.samples.range(threads + 1..).next();
        match (below, above) {
            (Some((&t0, &bw0)), Some((&t1, &bw1))) => {
                let frac = (threads - t0) as f64 / (t1 - t0) as f64;
                Some(bw0 + (bw1 - bw0) * frac)
            }
            (Some((_, &bw)), None) | (None, Some((_, &bw))) => Some(bw),
            (None, None) => None,
        }
    }

    /// Single-threaded bandwidth, or `None` if unknown.
    pub fn single_thread(&self) -> Option<f64> {
        self.at_parallelism(1)
    }

    /// Peak measured bandwidth across all recorded parallelisms, or `None`.
    pub fn peak(&self) -> Option<f64> {
        self.samples
            .values()
            .cloned()
            .fold(None, |acc, bw| Some(acc.map_or(bw, |m: f64| m.max(bw))))
    }
}

/// Compute throughput (FLOP/s) per element data type.
///
/// Absence of an entry means "not measured for this dtype" — the cost model
/// then declines to produce a compute-bound estimate for that dtype rather than
/// substituting a plausible number. Keyed by the ONNX dtype integer so it
/// serializes as a plain JSON object.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ComputeThroughput {
    /// ONNX dtype integer → FLOP/s.
    by_dtype: BTreeMap<i32, f64>,
}

impl ComputeThroughput {
    /// An empty (fully unknown) throughput table.
    pub fn unknown() -> Self {
        Self::default()
    }

    /// Record throughput (FLOP/s) for a dtype.
    pub fn set(&mut self, dtype: DataType, flops_per_sec: f64) {
        if flops_per_sec.is_finite() && flops_per_sec > 0.0 {
            self.by_dtype.insert(dtype.to_onnx(), flops_per_sec);
        }
    }

    /// Throughput (FLOP/s) for a dtype, or `None` if not measured.
    pub fn get(&self, dtype: DataType) -> Option<f64> {
        self.by_dtype.get(&dtype.to_onnx()).copied()
    }

    /// Whether any dtype throughput is known.
    pub fn is_known(&self) -> bool {
        !self.by_dtype.is_empty()
    }
}

/// The machine-specific rate profile for one device.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DeviceProfile {
    /// Human-readable device name (e.g. `NVIDIA GeForce RTX 4060 Laptop GPU`).
    pub name: String,
    /// Compute throughput (FLOP/s) per dtype. Unknown dtypes are absent.
    pub compute_throughput: ComputeThroughput,
    /// Sustained memory bandwidth, parallelism-qualified. Unknown if empty.
    pub memory_bandwidth: MemoryBandwidth,
    /// Fixed kernel-launch / dispatch latency, or `None` if unmeasured.
    pub launch_overhead: Option<Duration>,
}

impl DeviceProfile {
    /// A named profile with every rate unknown. Rates are filled by calibration
    /// (see [`crate::calibration`]), never by literals.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            compute_throughput: ComputeThroughput::unknown(),
            memory_bandwidth: MemoryBandwidth::unknown(),
            launch_overhead: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_bandwidth_returns_none() {
        let bw = MemoryBandwidth::unknown();
        assert_eq!(bw.at_parallelism(4), None);
        assert_eq!(bw.peak(), None);
        assert!(!bw.is_known());
    }

    #[test]
    fn bandwidth_interpolates_and_clamps() {
        // The #995 box: 1→10.07, 4→26.50, 8→37.46, 20→49.28 GB/s.
        let bw = MemoryBandwidth::from_samples([
            (1, 10.07e9),
            (4, 26.50e9),
            (8, 37.46e9),
            (20, 49.28e9),
        ]);
        assert_eq!(bw.single_thread(), Some(10.07e9));
        assert_eq!(bw.peak(), Some(49.28e9));
        // Exact sample.
        assert_eq!(bw.at_parallelism(8), Some(37.46e9));
        // Interpolated between 4 and 8 threads.
        let mid = bw.at_parallelism(6).unwrap();
        assert!((mid - 31.98e9).abs() < 1e6, "{mid}");
        // Clamped past the top of the measured range.
        assert_eq!(bw.at_parallelism(64), Some(49.28e9));
        // Below the lowest measured point clamps to it (no invented capacity).
        assert_eq!(bw.at_parallelism(0), Some(10.07e9));
    }

    #[test]
    fn compute_throughput_unknown_dtype_is_none() {
        let mut t = ComputeThroughput::unknown();
        t.set(DataType::Float32, 1.0e12);
        assert_eq!(t.get(DataType::Float32), Some(1.0e12));
        assert_eq!(t.get(DataType::Int4), None);
    }

    #[test]
    fn bad_samples_are_rejected() {
        let bw = MemoryBandwidth::from_samples([(0, 1.0e9), (1, f64::NAN), (2, -5.0), (4, 8.0e9)]);
        assert_eq!(bw.samples().count(), 1);
        assert_eq!(bw.at_parallelism(4), Some(8.0e9));
    }
}
