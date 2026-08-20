//! CubeCL client construction, one flavour per [`CubeclBackend`].
//!
//! # How the two backends are kept apart
//!
//! `WgpuRuntime<C>` is generic over its shader compiler, and that type
//! parameter — not a runtime flag — is what separates the two providers:
//!
//! * `cubecl-webgpu` instantiates `WgpuRuntime<WgslCompiler>`, so every kernel
//!   is emitted as WGSL no matter which native `wgpu` backend the host picks.
//! * `cubecl-vulkan` instantiates `WgpuRuntime<SpirvCompiler>`, which only
//!   exists when the `vulkan` feature is on, and that feature only builds off
//!   macOS because `cubecl-spirv` is not compiled for that target upstream.
//!
//! Choosing at the type level means a build either contains a backend's code
//! path or it does not; there is no runtime state in which a provider could
//! quietly emit the other language.

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime, WgslCompiler};
use onnx_runtime_ep_api::{EpError, Result};

use crate::backend::CubeclBackend;

/// The runtime type backing `cubecl-webgpu`.
pub type WebGpuRuntime = WgpuRuntime<WgslCompiler>;

/// The runtime type backing `cubecl-vulkan`.
#[cfg(feature = "vulkan")]
pub type VulkanRuntime = WgpuRuntime<cubecl_wgpu::SpirvCompiler>;

/// Translate a device ordinal into the CubeCL device selector.
///
/// Ordinal 0 means "whatever this host considers its default compute device",
/// which is what a caller that never named a device wants. Higher ordinals
/// address discrete GPUs explicitly so a multi-GPU host is addressable without
/// this EP inventing its own enumeration order.
pub fn wgpu_device(ordinal: u32) -> WgpuDevice {
    match ordinal {
        0 => WgpuDevice::DefaultDevice,
        n => WgpuDevice::DiscreteGpu(n as usize - 1),
    }
}

/// Build the compute client for `backend` on `ordinal`.
///
/// Returns an actionable error when the backend is not part of this build or
/// cannot exist on this platform, and propagates a device-initialisation panic
/// as an error rather than letting it unwind through the plugin's C boundary.
pub fn open_client<R: Runtime<Device = WgpuDevice>>(
    backend: CubeclBackend,
    ordinal: u32,
) -> Result<ComputeClient<R>> {
    if let Some(message) = backend.unavailable_message() {
        return Err(EpError::KernelFailed(message));
    }
    let device = wgpu_device(ordinal);
    // CubeCL surfaces "no adapter" as a panic from deep inside wgpu's async
    // init. The plugin boundary must never unwind, and "the GPU is missing" is
    // a user-facing configuration failure, not an invariant violation, so it is
    // converted into a `Result` here at the point where the context is known.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| R::client(&device))).map_err(|_| {
        EpError::KernelFailed(format!(
            "execution provider '{provider}' could not open a GPU device (ordinal {ordinal}). \
             No compatible {api} adapter was found, or the driver failed to initialise. \
             Check that a GPU driver is installed and visible to this process, select \
             another ordinal, or use ONNX_GENAI_EP=cpu.",
            provider = backend.provider_name(),
            api = match backend {
                CubeclBackend::WebGpu => "WebGPU/wgpu",
                CubeclBackend::Vulkan => "Vulkan",
            },
        ))
    })
}

/// A short description of the device a client is attached to, for logs and for
/// the `nxrt_ep_*` diagnostics the plugin exports.
pub fn describe_device<R: Runtime>(backend: CubeclBackend, client: &ComputeClient<R>) -> String {
    format!(
        "{provider} via {shader}",
        provider = backend.provider_name(),
        shader = R::name(client),
    )
}
