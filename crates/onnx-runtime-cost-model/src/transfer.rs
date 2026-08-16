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

/// The device power/activity state a link's bandwidth was **measured under**.
///
/// A host<->device bandwidth is not a device constant — it is a rate that
/// depends on the power state of the link at the moment it was sampled. On the
/// #995 box (RTX 4060 Laptop, PCIe Gen4) a probe taken while the GPU had dropped
/// to NVIDIA pstate **P8** (210 MHz SM, 6.5 W) read a flat **~1.6 GB/s** H2D
/// pinned, versus **~11.7 GB/s** for the identical binary with the GPU active —
/// a ~7× swing from power state alone, because a parked laptop GPU downclocks
/// the PCIe link (Gen4→Gen1). During decode the GPU is *not* parked, so a
/// `Parked` measurement systematically **under-states** the decode-time link and
/// must not be trusted as the rate that applies to real inference traffic.
///
/// This is recorded in the profile — not just in prose — so a consumer can tell
/// whether a rate is representative of the regime it is about to price. Unknown
/// is the honest default: a probe that did not record the power state leaves it
/// `Unknown` rather than claiming the number is trustworthy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MeasuredLinkState {
    /// The device was in an active/high-power state (representative of a busy
    /// decode loop) when the bandwidth was measured. Trustworthy for placement.
    Active,
    /// The device was parked in a low-power state (e.g. NVIDIA P8) when the
    /// bandwidth was measured. The link was downclocked, so the number
    /// under-states the decode-time rate and must not drive placement.
    Parked,
    /// The power state at measurement time was not recorded.
    #[default]
    Unknown,
}

impl MeasuredLinkState {
    /// Whether a rate measured in this state is representative of the active
    /// decode regime. Only [`Active`](Self::Active) qualifies; both `Parked`
    /// (known-unrepresentative) and `Unknown` (unverified) do not.
    pub fn is_decode_representative(self) -> bool {
        matches!(self, MeasuredLinkState::Active)
    }
}

/// Sustained transfer rate for one ordered `(src, dst)` device pair.
///
/// `time = latency_base + bytes / bandwidth`, the exact two-parameter roofline
/// the `roofline_transfer` probe fits. Both fields are `Option`: a probe that
/// only measured bandwidth (e.g. a single large size) leaves `latency_base`
/// unknown rather than pretending it is zero.
///
/// `measured_link_state` records the device power state the bandwidths were
/// sampled under (see [`MeasuredLinkState`]); a `Parked` measurement is a known
/// under-estimate of the decode-time link and a consumer can consult
/// [`is_decode_representative`](TransferProfile::is_decode_representative) before
/// trusting the rate for placement.
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
    /// The device power state these bandwidths were measured under. Defaults to
    /// [`MeasuredLinkState::Unknown`].
    #[serde(default)]
    pub measured_link_state: MeasuredLinkState,
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

    /// Whether these rates were measured in a state representative of the active
    /// decode regime (see [`MeasuredLinkState`]). A `false` here means the rate
    /// is either a known under-estimate (measured while the device was parked)
    /// or unverified — a consumer should not use it to justify moving data.
    pub fn is_decode_representative(&self) -> bool {
        self.measured_link_state.is_decode_representative()
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
                measured_link_state: MeasuredLinkState::Active,
            },
        );
        let p = m.get(&DeviceKey::cpu(), &DeviceKey::cuda(0)).unwrap();
        assert_eq!(p.bandwidth(HostMemoryKind::Pinned), Some(11.7e9));
        assert_eq!(p.bandwidth(HostMemoryKind::Pageable), Some(6.0e9));
        assert!(p.is_decode_representative());
        // The reverse direction was not recorded — unknown, not mirrored.
        assert!(m.get(&DeviceKey::cuda(0), &DeviceKey::cpu()).is_none());
    }

    #[test]
    fn measured_link_state_gates_decode_trust() {
        // Unknown is the default and is not decode-representative.
        assert!(!MeasuredLinkState::default().is_decode_representative());
        assert!(!TransferProfile::unknown().is_decode_representative());
        // A parked measurement is a known under-estimate — not trustworthy.
        let parked = TransferProfile {
            pinned_bandwidth: Some(1.6e9),
            measured_link_state: MeasuredLinkState::Parked,
            ..TransferProfile::unknown()
        };
        assert!(parked.is_known());
        assert!(!parked.is_decode_representative());
        // Only an active measurement qualifies.
        assert!(MeasuredLinkState::Active.is_decode_representative());
    }
}
