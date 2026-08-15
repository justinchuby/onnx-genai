//! Cross-device transfer rates (`docs/architecture/ORT2.md` §6.2
//! `TransferCostMatrix` / `TransferProfile`).
//!
//! Like [`crate::DeviceProfile`], everything here is a machine-specific rate,
//! and unknown is representable: a link that has not been probed is absent from
//! the matrix, and an unmeasured field inside a profile is `None`.
//!
//! The one addition over the bare §6.2 sketch is that a `TransferProfile`
//! carries **both** the pinned and pageable host-memory bandwidths. Issue #995
//! is explicit that the two differ by a large factor (the driver must bounce
//! pageable host memory through an internal pinned staging buffer) and that the
//! cost model has to know *which one the runtime will actually use*. Collapsing
//! them to a single number would systematically misprice every host<->device
//! copy by ~2×.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::device::DeviceKey;

/// Which kind of host buffer backs a host<->device transfer. The achievable
/// bandwidth differs sharply between the two, so a transfer cost is undefined
/// until the caller says which the runtime uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HostMemoryKind {
    /// Ordinary pageable host memory (the driver stages it through a bounce
    /// buffer).
    Pageable,
    /// Page-locked (pinned) host memory (DMA'd directly).
    Pinned,
}

/// Sustained transfer rate for one ordered `(src, dst)` device pair.
///
/// `time = latency_base + bytes / bandwidth`, the exact two-parameter roofline
/// the `roofline_transfer` probe fits. Both fields are `Option`: a probe that
/// only measured bandwidth (e.g. a single large size) leaves `latency_base`
/// unknown rather than pretending it is zero.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TransferProfile {
    /// Fixed per-transfer latency, or `None` if unmeasured.
    pub latency_base: Option<Duration>,
    /// Sustained bandwidth (bytes/sec) from a **pinned** host buffer, or `None`.
    pub pinned_bandwidth: Option<f64>,
    /// Sustained bandwidth (bytes/sec) from a **pageable** host buffer, or
    /// `None`.
    pub pageable_bandwidth: Option<f64>,
    /// Whether the link can overlap this copy with compute (async DMA).
    pub is_async_capable: bool,
}

impl TransferProfile {
    /// A profile with every rate unknown.
    pub fn unknown() -> Self {
        Self::default()
    }

    /// Bandwidth (bytes/sec) for a given host-memory kind, or `None` if that
    /// kind was not measured.
    pub fn bandwidth(&self, host: HostMemoryKind) -> Option<f64> {
        match host {
            HostMemoryKind::Pinned => self.pinned_bandwidth,
            HostMemoryKind::Pageable => self.pageable_bandwidth,
        }
    }

    /// Whether either bandwidth is known.
    pub fn is_known(&self) -> bool {
        self.pinned_bandwidth.is_some() || self.pageable_bandwidth.is_some()
    }
}

/// A serializable `(src, dst)` device pair used as the transfer-matrix key.
///
/// JSON object keys must be strings, so the matrix is stored as a list of
/// `(key, profile)` entries rather than a map with a tuple key.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TransferKey {
    /// Source device.
    pub src: DeviceKey,
    /// Destination device.
    pub dst: DeviceKey,
}

impl TransferKey {
    /// Construct a directed transfer key.
    pub fn new(src: DeviceKey, dst: DeviceKey) -> Self {
        Self { src, dst }
    }
}

/// The full set of directed transfer profiles between devices.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TransferCostMatrix {
    #[serde(with = "crate::serde_util::map_as_seq")]
    entries: BTreeMap<TransferKey, TransferProfile>,
}

impl TransferCostMatrix {
    /// An empty matrix (every link unknown).
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the profile for a directed link, replacing any previous entry.
    pub fn set(&mut self, src: DeviceKey, dst: DeviceKey, profile: TransferProfile) {
        self.entries.insert(TransferKey::new(src, dst), profile);
    }

    /// The profile for a directed link, or `None` if it was never measured.
    ///
    /// A same-device transfer (`src == dst`) is free by definition and returns
    /// a synthesized zero-latency, effectively-infinite-bandwidth profile so
    /// callers do not have to special-case it — that is a structural fact, not
    /// a fabricated machine rate.
    pub fn get(&self, src: &DeviceKey, dst: &DeviceKey) -> Option<&TransferProfile> {
        self.entries
            .get(&TransferKey::new(src.clone(), dst.clone()))
    }

    /// All recorded links.
    pub fn entries(&self) -> impl Iterator<Item = (&TransferKey, &TransferProfile)> {
        self.entries.iter()
    }

    /// Number of recorded links.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no link has been recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_link_is_none() {
        let m = TransferCostMatrix::new();
        assert!(m.get(&DeviceKey::cpu(), &DeviceKey::cuda(0)).is_none());
    }

    #[test]
    fn direction_and_host_kind_are_distinct() {
        let mut m = TransferCostMatrix::new();
        m.set(
            DeviceKey::cpu(),
            DeviceKey::cuda(0),
            TransferProfile {
                latency_base: Some(Duration::from_micros(12)),
                pinned_bandwidth: Some(11.7e9),
                pageable_bandwidth: Some(6.0e9),
                is_async_capable: true,
            },
        );
        let p = m.get(&DeviceKey::cpu(), &DeviceKey::cuda(0)).unwrap();
        assert_eq!(p.bandwidth(HostMemoryKind::Pinned), Some(11.7e9));
        assert_eq!(p.bandwidth(HostMemoryKind::Pageable), Some(6.0e9));
        // The reverse direction was not recorded — unknown, not mirrored.
        assert!(m.get(&DeviceKey::cuda(0), &DeviceKey::cpu()).is_none());
    }
}
