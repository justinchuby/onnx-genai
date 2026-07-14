//! Device types and placement identifiers (see `docs/ORT2.md` §4.2).
//!
//! Device placement is a first-class annotation on every [`crate::Value`] and
//! [`crate::Node`], enabling multi-device partitioning without side tables.

/// A class of compute device / execution backend.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Rocm,
    CoreMl,
    Mlx,
    WebGpu,
    Qnn,
    OpenVino,
    /// Vendor / experimental backend keyed by an opaque id.
    Custom(u32),
}

impl DeviceType {
    /// Whether tensors on this device share the host address space and can be
    /// accessed by CPU code without an explicit copy.
    pub fn is_host_accessible(self) -> bool {
        // MLX targets Apple unified memory; CPU is trivially host-accessible.
        matches!(self, DeviceType::Cpu | DeviceType::Mlx)
    }
}

/// A specific device instance: a [`DeviceType`] plus an ordinal index.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DeviceId {
    pub device_type: DeviceType,
    pub index: u32,
}

impl DeviceId {
    /// Construct a device id.
    pub fn new(device_type: DeviceType, index: u32) -> Self {
        Self { device_type, index }
    }

    /// The default host device (`CPU:0`).
    pub fn cpu() -> Self {
        Self::new(DeviceType::Cpu, 0)
    }

    /// A CUDA device by ordinal.
    pub fn cuda(index: u32) -> Self {
        Self::new(DeviceType::Cuda, index)
    }

    /// Whether this device is host-accessible (see [`DeviceType::is_host_accessible`]).
    pub fn is_host_accessible(self) -> bool {
        self.device_type.is_host_accessible()
    }
}

impl Default for DeviceId {
    fn default() -> Self {
        Self::cpu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_cpu0() {
        assert_eq!(DeviceId::default(), DeviceId::cpu());
        assert_eq!(DeviceId::default().index, 0);
    }

    #[test]
    fn host_accessibility() {
        assert!(DeviceId::cpu().is_host_accessible());
        assert!(DeviceId::new(DeviceType::Mlx, 0).is_host_accessible());
        assert!(!DeviceId::cuda(0).is_host_accessible());
    }
}
