//! Device types and placement identifiers (see `docs/architecture/ORT2.md` §4.2).
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
    /// The canonical lower-case name used in traces and diagnostics.
    ///
    /// One owner for these spellings so a trace produced by any execution
    /// provider labels its device the same way. `Custom` keeps its opaque id
    /// rather than collapsing every vendor backend to one indistinguishable
    /// name.
    pub fn trace_name(self) -> std::borrow::Cow<'static, str> {
        match self {
            DeviceType::Cpu => "cpu".into(),
            DeviceType::Cuda => "cuda".into(),
            DeviceType::Rocm => "rocm".into(),
            DeviceType::CoreMl => "coreml".into(),
            DeviceType::Mlx => "mlx".into(),
            DeviceType::WebGpu => "webgpu".into(),
            DeviceType::Qnn => "qnn".into(),
            DeviceType::OpenVino => "openvino".into(),
            DeviceType::Custom(id) => format!("custom:{id}").into(),
        }
    }

    /// The device a canonical name refers to, or `None` if it names none.
    ///
    /// The inverse of [`trace_name`](DeviceType::trace_name), kept beside it so
    /// the two cannot drift. A plugin execution provider is configured with a
    /// device name from package metadata and has to report which device it
    /// actually runs on; without this it could only guess, and guessing `Cpu`
    /// makes a trace claim that Metal work happened on the host.
    pub fn from_trace_name(name: &str) -> Option<Self> {
        let name = name.trim().to_ascii_lowercase();
        Some(match name.as_str() {
            "cpu" => DeviceType::Cpu,
            "cuda" => DeviceType::Cuda,
            "rocm" => DeviceType::Rocm,
            "coreml" => DeviceType::CoreMl,
            // The Metal plugin is named for the API it targets; the device
            // class it runs on is MLX.
            "mlx" | "metal" => DeviceType::Mlx,
            "webgpu" => DeviceType::WebGpu,
            "qnn" => DeviceType::Qnn,
            "openvino" => DeviceType::OpenVino,
            other => {
                let id = other.strip_prefix("custom:")?.parse().ok()?;
                DeviceType::Custom(id)
            }
        })
    }

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
