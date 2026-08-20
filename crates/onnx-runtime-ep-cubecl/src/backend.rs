//! Which CubeCL shader backend an execution provider instance drives.
//!
//! This module carries no CubeCL dependency so that name/identity questions —
//! "what is this provider called", "is it available on this build" — can be
//! answered by a host that was compiled without any GPU backend feature.

use onnx_runtime_ir::DeviceType;

/// The CubeCL shader backend a provider instance compiles kernels for.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CubeclBackend {
    /// WGSL shaders over any `wgpu` backend. Portable; the dtype floor is
    /// `f32`/`f16`/`i32`/`u32`.
    WebGpu,
    /// SPIR-V shaders pinned to the Vulkan backend. Adds `bf16` and 8/16-bit
    /// integers, and enables subgroup (plane) operations.
    Vulkan,
}

impl CubeclBackend {
    /// Every backend, in declaration order. Used by the plugin to advertise one
    /// factory per backend without duplicating the list.
    pub const ALL: [CubeclBackend; 2] = [CubeclBackend::WebGpu, CubeclBackend::Vulkan];

    /// The EP identifier reported by [`ExecutionProvider::name`], snake_case to
    /// match `cpu_ep` / `cuda_ep`.
    ///
    /// [`ExecutionProvider::name`]: onnx_runtime_ep_api::provider::ExecutionProvider::name
    pub const fn ep_name(self) -> &'static str {
        match self {
            CubeclBackend::WebGpu => "cubecl_webgpu_ep",
            CubeclBackend::Vulkan => "cubecl_vulkan_ep",
        }
    }

    /// The user-facing provider name accepted by `ONNX_GENAI_EP`.
    pub const fn provider_name(self) -> &'static str {
        match self {
            CubeclBackend::WebGpu => "cubecl-webgpu",
            CubeclBackend::Vulkan => "cubecl-vulkan",
        }
    }

    /// The name ORT registers this backend's EP factory under.
    pub const fn registration_name(self) -> &'static str {
        match self {
            CubeclBackend::WebGpu => "onnxruntime_cubecl_webgpu_ep",
            CubeclBackend::Vulkan => "onnxruntime_cubecl_vulkan_ep",
        }
    }

    /// The device class nodes placed on this provider are annotated with.
    pub const fn device_type(self) -> DeviceType {
        match self {
            CubeclBackend::WebGpu => DeviceType::WebGpu,
            CubeclBackend::Vulkan => DeviceType::Vulkan,
        }
    }

    /// Parse a user-supplied provider name. Accepts the canonical spelling and
    /// the `_`-separated variant, case-insensitively.
    pub fn from_provider_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "cubecl-webgpu" => Some(CubeclBackend::WebGpu),
            "cubecl-vulkan" => Some(CubeclBackend::Vulkan),
            _ => None,
        }
    }

    /// Whether this build can construct the backend, and if not, exactly why.
    ///
    /// This is a *compile-shape* check only: it reports whether the code paths
    /// exist, not whether a GPU is present. Device presence is discovered when
    /// the provider is constructed, because that is the first point at which it
    /// can be observed honestly.
    pub const fn availability(self) -> Result<(), UnavailableReason> {
        match self {
            CubeclBackend::WebGpu => {
                if cfg!(feature = "webgpu") {
                    Ok(())
                } else {
                    Err(UnavailableReason::FeatureDisabled)
                }
            }
            CubeclBackend::Vulkan => {
                if cfg!(target_os = "macos") {
                    Err(UnavailableReason::UnsupportedPlatform)
                } else if cfg!(feature = "vulkan") {
                    Ok(())
                } else {
                    Err(UnavailableReason::FeatureDisabled)
                }
            }
        }
    }

    /// A complete, actionable message for why this backend cannot be used, or
    /// `None` when it can (RULES.md §1).
    pub fn unavailable_message(self) -> Option<String> {
        let reason = self.availability().err()?;
        let provider = self.provider_name();
        Some(match reason {
            UnavailableReason::FeatureDisabled => format!(
                "execution provider '{provider}' is not compiled into this build of \
                 onnx-runtime-ep-cubecl. Rebuild the plugin with \
                 `--features {}` (the cdylib is built by \
                 `cargo build -p onnx-runtime-ep-cubecl-plugin --features {}`), or select a \
                 different provider with ONNX_GENAI_EP.",
                self.cargo_feature(),
                self.cargo_feature(),
            ),
            UnavailableReason::UnsupportedPlatform => format!(
                "execution provider '{provider}' is unavailable on macOS: CubeCL's SPIR-V \
                 compiler (cubecl-spirv) is not built for this target, so no Vulkan shader \
                 can be produced. Use ONNX_GENAI_EP=cubecl-webgpu, which runs the same \
                 kernels through WGSL on Metal."
            ),
        })
    }

    /// The cargo feature that enables this backend.
    pub const fn cargo_feature(self) -> &'static str {
        match self {
            CubeclBackend::WebGpu => "webgpu",
            CubeclBackend::Vulkan => "vulkan",
        }
    }
}

/// Why a [`CubeclBackend`] cannot be constructed in this build.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UnavailableReason {
    /// The backend's cargo feature was not enabled.
    FeatureDisabled,
    /// The backend cannot exist on this target at all.
    UnsupportedPlatform,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_names_round_trip() {
        for backend in CubeclBackend::ALL {
            assert_eq!(
                CubeclBackend::from_provider_name(backend.provider_name()),
                Some(backend),
            );
        }
    }

    #[test]
    fn provider_names_accept_underscore_and_case_variants() {
        assert_eq!(
            CubeclBackend::from_provider_name("CubeCL_WebGPU"),
            Some(CubeclBackend::WebGpu),
        );
        assert_eq!(
            CubeclBackend::from_provider_name("  cubecl_vulkan  "),
            Some(CubeclBackend::Vulkan),
        );
        assert_eq!(CubeclBackend::from_provider_name("cubecl"), None);
    }

    #[test]
    fn backends_are_distinguishable_end_to_end() {
        // Two providers that reported the same identity anywhere would be
        // indistinguishable in traces and placement.
        let webgpu = CubeclBackend::WebGpu;
        let vulkan = CubeclBackend::Vulkan;
        assert_ne!(webgpu.ep_name(), vulkan.ep_name());
        assert_ne!(webgpu.provider_name(), vulkan.provider_name());
        assert_ne!(webgpu.registration_name(), vulkan.registration_name());
        assert_ne!(webgpu.device_type(), vulkan.device_type());
    }

    #[test]
    fn unavailable_message_names_the_fix() {
        for backend in CubeclBackend::ALL {
            let Some(message) = backend.unavailable_message() else {
                continue;
            };
            assert!(
                message.contains(backend.provider_name()),
                "message must name the provider: {message}"
            );
            assert!(
                message.contains("ONNX_GENAI_EP") || message.contains("--features"),
                "message must name a concrete fix: {message}"
            );
        }
    }

    #[test]
    fn vulkan_is_unavailable_on_macos_regardless_of_feature() {
        if cfg!(target_os = "macos") {
            assert_eq!(
                CubeclBackend::Vulkan.availability(),
                Err(UnavailableReason::UnsupportedPlatform),
            );
        }
    }
}
