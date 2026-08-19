//! [`PlacementCostModel`] — the assembled cost model (`docs/architecture/ORT2.md`
//! §6.2/§6.3/§6.4): per-device rate profiles, a transfer matrix, the cost
//! formulas that combine them with kernel structure, and `save`/`load`.
//!
//! `save`/`load` is a product requirement, not a convenience (issue #995): a
//! user calibrates on their own hardware, saves the artifact, and passes it in
//! to compute costs and plans **offline**. The serialized form is plain JSON so
//! it is inspectable and portable.
//!
//! Every formula returns `Option<Cost>` and yields `None` the moment a rate it
//! needs is missing. This is the load-bearing #947 lesson: it is always correct
//! to say "I cannot price this yet"; it is never correct to substitute a
//! plausible constant and report a confident wrong cost.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use onnx_runtime_ir::{DataType, DeviceId};
use serde::{Deserialize, Serialize};

use crate::device::DeviceKey;
use crate::profile::DeviceProfile;
use crate::structure::{Cost, KernelStructure};
use crate::transfer::{HostMemoryKind, TransferCostMatrix, TransferProfile};

/// Errors from loading or saving a cost model.
#[derive(Debug, thiserror::Error)]
pub enum CostModelError {
    /// The artifact could not be read or written.
    #[error("cost-model I/O error at {path}: {source}")]
    Io {
        /// The path involved.
        path: String,
        /// The underlying I/O error.
        source: std::io::Error,
    },
    /// The artifact was not valid JSON for a cost model.
    #[error("cost-model (de)serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

/// The assembled placement cost model.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlacementCostModel {
    #[serde(with = "crate::serde_util::map_as_seq")]
    device_profiles: BTreeMap<DeviceKey, DeviceProfile>,
    transfer_matrix: TransferCostMatrix,
}

impl PlacementCostModel {
    /// A model with no devices and no links — every cost is unknown until
    /// calibration supplies rates.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a device's rate profile.
    pub fn set_device_profile(&mut self, device: DeviceKey, profile: DeviceProfile) {
        self.device_profiles.insert(device, profile);
    }

    /// The rate profile for a device, or `None` if the device is unknown.
    pub fn device_profile(&self, device: &DeviceKey) -> Option<&DeviceProfile> {
        self.device_profiles.get(device)
    }

    /// Mutable access to a device profile, inserting a blank named profile if
    /// absent (used by calibration to fill rates incrementally).
    pub fn device_profile_mut(&mut self, device: DeviceKey) -> &mut DeviceProfile {
        self.device_profiles
            .entry(device)
            .or_insert_with(|| DeviceProfile::new(String::new()))
    }

    /// The transfer matrix.
    pub fn transfer_matrix(&self) -> &TransferCostMatrix {
        &self.transfer_matrix
    }

    /// Mutable transfer matrix (used by calibration).
    pub fn transfer_matrix_mut(&mut self) -> &mut TransferCostMatrix {
        &mut self.transfer_matrix
    }

    /// Record a directed transfer profile.
    pub fn set_transfer_profile(
        &mut self,
        src: DeviceKey,
        dst: DeviceKey,
        profile: TransferProfile,
    ) {
        self.transfer_matrix.set(src, dst, profile);
    }

    /// Known device keys.
    pub fn devices(&self) -> impl Iterator<Item = &DeviceKey> {
        self.device_profiles.keys()
    }

    // ---- Cost formulas (§6.3) ------------------------------------------------

    /// Estimate the cost of running a kernel with the given [`KernelStructure`]
    /// on `device`, assuming `parallelism` concurrently active threads/streams.
    ///
    /// Roofline: `time = max(memory_time, compute_time) + launch`, where
    ///   * `memory_time = bytes_moved / memory_bandwidth(parallelism)`,
    ///   * `compute_time = flops / compute_throughput(dtype)` (added to the max
    ///     only when both FLOPs and that dtype's throughput are known),
    ///   * `launch` is `launch_overhead` scaled by the launch count (0 if
    ///     unmeasured).
    ///
    /// Returns `None` when memory bandwidth at the requested parallelism is
    /// unknown for the device — the memory term is the load-bearing one for
    /// decode, so without it there is no defensible estimate. When FLOPs or the
    /// dtype throughput are unknown the compute term is omitted, and when
    /// `launch_overhead` is unmeasured the launch term is omitted; in either
    /// case the returned [`Cost`] is a **lower bound** ([`Cost::is_lower_bound`]
    /// is set) rather than a point estimate, because both omissions are
    /// optimistic (the true time can only be larger).
    pub fn op_cost(
        &self,
        structure: &KernelStructure,
        device: &DeviceKey,
        parallelism: u32,
    ) -> Option<Cost> {
        let profile = self.device_profiles.get(device)?;
        let bandwidth = profile.memory_bandwidth.at_parallelism(parallelism)?;
        let memory_time = structure.bytes_moved as f64 / bandwidth;

        let compute_time = match (structure.flops, structure.compute_dtype) {
            (Some(flops), Some(dtype)) => profile
                .compute_throughput
                .get(dtype)
                .map(|tp| flops as f64 / tp),
            _ => None,
        };

        let mut seconds = compute_time.map_or(memory_time, |ct| ct.max(memory_time));
        // The compute term is omitted whenever FLOPs / dtype throughput were
        // unknown; the launch term is omitted whenever launch latency was never
        // measured. Both omissions only ever *lower* the estimate, so a cost
        // missing either is a lower bound, not a point estimate.
        let compute_omitted = compute_time.is_none();
        let launch_omitted = profile.launch_overhead.is_none();
        if let Some(launch) = profile.launch_overhead {
            seconds += launch.as_secs_f64() * structure.launch_count.max(1) as f64;
        }

        let time = Duration::from_secs_f64(seconds);
        Some(if compute_omitted || launch_omitted {
            Cost::lower_bound(time, structure.bytes_moved)
        } else {
            Cost::estimate(time, structure.bytes_moved)
        })
    }

    /// Estimate a host<->device transfer of `bytes` from `src` to `dst` using
    /// the given host-memory kind.
    ///
    /// `time = latency_base + bytes / bandwidth(host_kind)`. Returns a
    /// zero-cost for a same-device "transfer" (a structural fact), and `None`
    /// when the link, or the requested host-memory kind's bandwidth, was never
    /// measured.
    ///
    /// # `latency_base` is an optimistic omission, made explicit
    ///
    /// Bandwidth is refused when unknown (that is the load-bearing term), but
    /// `latency_base` is only an *additive refinement*, so rather than reject the
    /// whole estimate over it, an unmeasured latency is treated as zero. That
    /// substitution is **optimistic** — it under-states transfer cost — and this
    /// is exactly the wrong direction to be silently wrong in: under-pricing a
    /// transfer biases a placement search toward moving data, which is the defect
    /// #994 exists to fix. The optimism is therefore not silent: when
    /// `latency_base` was unmeasured the returned [`Cost`] is flagged
    /// [`Cost::is_lower_bound`], so a reader cannot mistake it for a point
    /// estimate or use it to justify a move. When latency *was* measured the
    /// result is a full point estimate.
    pub fn transfer_cost(
        &self,
        bytes: u64,
        src: &DeviceKey,
        dst: &DeviceKey,
        host: HostMemoryKind,
    ) -> Option<Cost> {
        if src == dst {
            return Some(Cost::zero());
        }
        let profile = self.transfer_matrix.get(src, dst)?;
        let bandwidth = profile.bandwidth(host)?;
        // Unmeasured latency is treated as zero (an additive lower-bound term),
        // but the resulting cost is flagged as a lower bound so the optimism is
        // visible to the caller rather than silent.
        let latency = profile.latency_base.unwrap_or(Duration::ZERO);
        let seconds = latency.as_secs_f64() + bytes as f64 / bandwidth;
        let time = Duration::from_secs_f64(seconds);
        Some(if profile.latency_base.is_some() {
            Cost::estimate(time, bytes)
        } else {
            Cost::lower_bound(time, bytes)
        })
    }

    /// Convenience: the roofline cost of a dense `MatMul` (`[m,k] x [k,n]`),
    /// deriving the [`KernelStructure`] (`2*m*n*k` FLOPs, real dtype byte
    /// traffic) and applying [`op_cost`](Self::op_cost).
    pub fn matmul_cost(
        &self,
        m: u64,
        n: u64,
        k: u64,
        dtype: DataType,
        device: &DeviceKey,
        parallelism: u32,
    ) -> Option<Cost> {
        let elem = dtype.byte_size() as u64;
        let bytes_read = (m.saturating_mul(k).saturating_add(k.saturating_mul(n))) * elem;
        let bytes_written = m.saturating_mul(n) * elem;
        let structure = KernelStructure {
            bytes_moved: bytes_read.saturating_add(bytes_written),
            flops: Some(2u64.saturating_mul(m).saturating_mul(n).saturating_mul(k)),
            compute_dtype: Some(dtype),
            launch_count: 1,
        };
        self.op_cost(&structure, device, parallelism)
    }

    // ---- Persistence (§6.4) --------------------------------------------------

    /// Serialize the cost model to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String, CostModelError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize a cost model from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, CostModelError> {
        Ok(serde_json::from_str(json)?)
    }

    /// Save the cost model to `path` as JSON, so it can be reloaded for offline
    /// planning without re-calibrating (§6.4).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), CostModelError> {
        let path = path.as_ref();
        let json = self.to_json()?;
        std::fs::write(path, json).map_err(|source| CostModelError::Io {
            path: path.display().to_string(),
            source,
        })
    }

    /// Load a cost model previously written by [`save`](Self::save).
    pub fn load(path: impl AsRef<Path>) -> Result<Self, CostModelError> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path).map_err(|source| CostModelError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_json(&json)
    }
}

/// Convenience conversions so callers holding IR device handles can query the
/// model without constructing [`DeviceKey`]s by hand.
impl PlacementCostModel {
    /// [`device_profile`](Self::device_profile) keyed by an IR [`DeviceId`].
    pub fn device_profile_for(&self, device: DeviceId) -> Option<&DeviceProfile> {
        self.device_profiles.get(&DeviceKey::from(device))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::MemoryBandwidth;
    use crate::transfer::MeasuredLinkState;

    fn cuda_profile() -> DeviceProfile {
        let mut p = DeviceProfile::new("test-gpu");
        p.memory_bandwidth = MemoryBandwidth::from_samples([(1, 1.0e12)]);
        p.compute_throughput.set(DataType::Float16, 2.0e13);
        p.launch_overhead = Some(Duration::from_micros(5));
        p
    }

    #[test]
    fn op_cost_none_when_bandwidth_unknown() {
        let mut model = PlacementCostModel::new();
        model.set_device_profile(DeviceKey::cuda(0), DeviceProfile::new("gpu"));
        let s = KernelStructure::from_bytes(1_000_000);
        assert!(model.op_cost(&s, &DeviceKey::cuda(0), 1).is_none());
    }

    #[test]
    fn op_cost_memory_bound_when_flops_unknown() {
        let mut model = PlacementCostModel::new();
        model.set_device_profile(DeviceKey::cuda(0), cuda_profile());
        // 1e9 bytes / 1e12 B/s = 1 ms, plus 5 us launch.
        let s = KernelStructure::from_bytes(1_000_000_000);
        let cost = model.op_cost(&s, &DeviceKey::cuda(0), 1).unwrap();
        let expected = 1.0e-3 + 5.0e-6;
        assert!(
            (cost.time.as_secs_f64() - expected).abs() < 1e-9,
            "{cost:?}"
        );
    }

    #[test]
    fn op_cost_takes_roofline_max() {
        let mut model = PlacementCostModel::new();
        model.set_device_profile(DeviceKey::cuda(0), cuda_profile());
        // Compute-heavy: 2e13 FLOP / 2e13 = 1 s dominates the tiny memory term.
        let s = KernelStructure {
            bytes_moved: 8,
            flops: Some(20_000_000_000_000),
            compute_dtype: Some(DataType::Float16),
            launch_count: 1,
        };
        let cost = model.op_cost(&s, &DeviceKey::cuda(0), 1).unwrap();
        assert!(cost.time.as_secs_f64() >= 1.0, "{cost:?}");
    }

    #[test]
    fn transfer_same_device_is_free() {
        let model = PlacementCostModel::new();
        let cost = model
            .transfer_cost(
                1024,
                &DeviceKey::cpu(),
                &DeviceKey::cpu(),
                HostMemoryKind::Pinned,
            )
            .unwrap();
        assert_eq!(cost, Cost::zero());
    }

    #[test]
    fn transfer_none_when_link_unknown() {
        let model = PlacementCostModel::new();
        assert!(
            model
                .transfer_cost(
                    1024,
                    &DeviceKey::cpu(),
                    &DeviceKey::cuda(0),
                    HostMemoryKind::Pinned
                )
                .is_none()
        );
    }

    #[test]
    fn transfer_respects_host_kind() {
        let mut model = PlacementCostModel::new();
        model.set_transfer_profile(
            DeviceKey::cpu(),
            DeviceKey::cuda(0),
            TransferProfile {
                latency_base: Some(Duration::from_micros(10)),
                pinned_bandwidth: Some(10.0e9),
                pageable_bandwidth: None,
                is_async_capable: true,
                measured_link_state: MeasuredLinkState::Active,
            },
        );
        let pinned = model.transfer_cost(
            1_000_000_000,
            &DeviceKey::cpu(),
            &DeviceKey::cuda(0),
            HostMemoryKind::Pinned,
        );
        assert!(pinned.is_some());
        // Pageable bandwidth was not measured — unknown, not defaulted.
        let pageable = model.transfer_cost(
            1_000_000_000,
            &DeviceKey::cpu(),
            &DeviceKey::cuda(0),
            HostMemoryKind::Pageable,
        );
        assert!(pageable.is_none());
    }

    #[test]
    fn transfer_cost_is_point_estimate_when_latency_measured() {
        let mut model = PlacementCostModel::new();
        model.set_transfer_profile(
            DeviceKey::cpu(),
            DeviceKey::cuda(0),
            TransferProfile {
                latency_base: Some(Duration::from_micros(12)),
                pinned_bandwidth: Some(11.7e9),
                pageable_bandwidth: None,
                is_async_capable: true,
                measured_link_state: MeasuredLinkState::Active,
            },
        );
        let cost = model
            .transfer_cost(
                1_000_000_000,
                &DeviceKey::cpu(),
                &DeviceKey::cuda(0),
                HostMemoryKind::Pinned,
            )
            .unwrap();
        assert!(!cost.is_lower_bound, "{cost:?}");
    }

    #[test]
    fn transfer_cost_is_lower_bound_when_latency_unmeasured() {
        let mut model = PlacementCostModel::new();
        model.set_transfer_profile(
            DeviceKey::cpu(),
            DeviceKey::cuda(0),
            TransferProfile {
                latency_base: None,
                pinned_bandwidth: Some(11.7e9),
                pageable_bandwidth: None,
                is_async_capable: true,
                measured_link_state: MeasuredLinkState::Active,
            },
        );
        let cost = model
            .transfer_cost(
                1_000_000_000,
                &DeviceKey::cpu(),
                &DeviceKey::cuda(0),
                HostMemoryKind::Pinned,
            )
            .unwrap();
        // latency_base was never measured — the omission (treated as zero) is
        // optimistic, so the cost must announce itself as a lower bound.
        assert!(cost.is_lower_bound, "{cost:?}");
    }

    #[test]
    fn op_cost_is_lower_bound_when_compute_term_omitted() {
        let mut model = PlacementCostModel::new();
        model.set_device_profile(DeviceKey::cuda(0), cuda_profile());
        // FLOPs unknown → compute term omitted → lower bound.
        let s = KernelStructure::from_bytes(1_000_000_000);
        let cost = model.op_cost(&s, &DeviceKey::cuda(0), 1).unwrap();
        assert!(cost.is_lower_bound, "{cost:?}");
    }

    #[test]
    fn op_cost_is_point_estimate_when_fully_known() {
        let mut model = PlacementCostModel::new();
        model.set_device_profile(DeviceKey::cuda(0), cuda_profile());
        // FLOPs + dtype throughput + launch overhead all known → point estimate.
        let s = KernelStructure {
            bytes_moved: 8,
            flops: Some(1_000_000),
            compute_dtype: Some(DataType::Float16),
            launch_count: 1,
        };
        let cost = model.op_cost(&s, &DeviceKey::cuda(0), 1).unwrap();
        assert!(!cost.is_lower_bound, "{cost:?}");
    }
}
