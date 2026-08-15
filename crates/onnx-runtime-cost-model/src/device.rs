//! Serializable device identity for the cost model.
//!
//! [`onnx_runtime_ir::DeviceId`] is the runtime's in-memory device handle, but
//! it does not derive `serde` (it is a `Copy` enum + ordinal in the IR crate).
//! A cost model has to be **serializable** — the whole point of §6.4's
//! `save`/`load` is that a user can calibrate on their own hardware and pass the
//! artifact back in for offline planning — so this module defines a stable,
//! string-keyed [`DeviceKey`] that round-trips through JSON and converts to and
//! from `DeviceId` without depending on the IR crate growing a serde surface.

use onnx_runtime_ir::{DeviceId, DeviceType};
use serde::{Deserialize, Serialize};

/// A serialization-stable device identifier: the canonical device name plus its
/// ordinal index.
///
/// The `kind` string is exactly [`DeviceType::trace_name`], so a serialized key
/// reads the same way it does in a trace (`cpu`, `cuda`, `custom:7`, ...) and
/// there is a single owner for the spelling.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DeviceKey {
    /// Canonical device-class name (`cpu`, `cuda`, ...).
    pub kind: String,
    /// Device ordinal.
    pub index: u32,
}

impl DeviceKey {
    /// Construct a key from a canonical name and ordinal.
    pub fn new(kind: impl Into<String>, index: u32) -> Self {
        Self {
            kind: kind.into(),
            index,
        }
    }

    /// The host device, `cpu:0`.
    pub fn cpu() -> Self {
        Self::new("cpu", 0)
    }

    /// A CUDA device by ordinal.
    pub fn cuda(index: u32) -> Self {
        Self::new("cuda", index)
    }

    /// Convert to an IR [`DeviceId`], if the `kind` names a known device class.
    ///
    /// Returns `None` for an unrecognized name rather than guessing a device —
    /// a mislabeled key must not silently become `cpu`.
    pub fn to_device_id(&self) -> Option<DeviceId> {
        DeviceType::from_trace_name(&self.kind).map(|device_type| DeviceId {
            device_type,
            index: self.index,
        })
    }
}

impl From<DeviceId> for DeviceKey {
    fn from(id: DeviceId) -> Self {
        Self {
            kind: id.device_type.trace_name().into_owned(),
            index: id.index,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_device_id() {
        for id in [DeviceId::cpu(), DeviceId::cuda(0), DeviceId::cuda(3)] {
            let key = DeviceKey::from(id);
            assert_eq!(key.to_device_id(), Some(id));
        }
    }

    #[test]
    fn round_trips_through_json() {
        let key = DeviceKey::cuda(2);
        let json = serde_json::to_string(&key).unwrap();
        let back: DeviceKey = serde_json::from_str(&json).unwrap();
        assert_eq!(key, back);
    }

    #[test]
    fn unknown_kind_yields_no_device_id() {
        assert_eq!(DeviceKey::new("nonsense", 0).to_device_id(), None);
    }
}
