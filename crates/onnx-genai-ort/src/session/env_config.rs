use std::path::Path;

use onnx_genai_runtime_config::{
    CudaDevice, ExecutionProviderEntry, IntraOpThreads, runtime_config,
};

use super::CudaAttentionMode;
use super::ep_compat::{ResolvedEp, capability, resolve_execution_provider};
use super::options::SessionOptions;
use super::plugin::{
    plugin_registration_name_from_path, resolve_inline_plugin, resolve_plugin_selection,
};

pub(super) fn execution_providers_from_env() -> Option<Vec<ResolvedEp>> {
    let entries = &runtime_config().execution_providers;
    if entries.is_empty() {
        return None;
    }
    let providers = entries
        .iter()
        .filter_map(|entry| match entry {
            ExecutionProviderEntry::Builtin(selection) if selection.name == "plugin" => {
                let config = runtime_config();
                let library = config.ep_library.clone()?;
                Some(resolve_plugin_selection(
                    selection.clone(),
                    library.clone(),
                    config
                        .ep_registration_name
                        .clone()
                        .unwrap_or_else(|| plugin_registration_name_from_path(&library)),
                    config.ep_options.clone(),
                    config.ep_device.clone(),
                ))
            }
            ExecutionProviderEntry::Builtin(selection) => {
                Some(resolve_execution_provider(selection))
            }
            ExecutionProviderEntry::Plugin(spec) => resolve_inline_plugin(spec),
        })
        .collect::<Vec<_>>();
    (!providers.is_empty()).then_some(providers)
}

pub(super) fn requested_non_cpu_provider(options: &SessionOptions) -> bool {
    options
        .execution_providers
        .iter()
        .any(|ep| !ep.caps.is_host())
}

/// Whether `path` names an ONNX protobuf TextFormat fixture (`*.textproto`).
///
/// Textproto models are git-friendly text; ORT cannot read them from disk, so
/// [`Session::new`] converts them to binary bytes and loads them from memory.
pub(super) fn is_textproto_path(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("textproto"))
}

pub(super) fn requested_strict_provider(options: &SessionOptions) -> bool {
    options
        .execution_providers
        .iter()
        .any(ResolvedEp::is_strict)
}

pub(super) fn cuda_device_id_from_env() -> i32 {
    match &runtime_config().cuda_device {
        CudaDevice::Id(device_id) => *device_id,
        CudaDevice::Invalid(value) => {
            tracing::warn!(
                "Ignoring invalid ONNX_GENAI_CUDA_DEVICE={value}; expected a non-negative integer, using device 0"
            );
            0
        }
    }
}

/// Optional intra-op thread override from `ONNX_GENAI_INTRA_OP_THREADS`.
///
/// Only consulted when the caller left `intra_op_num_threads` at the default
/// (0 = "ORT decides"); an explicit `with_intra_op_threads` always wins. A
/// positive integer pins the ORT intra-op pool. This is the profiler-identified
/// CPU decode lever: ORT's default oversubscribes Apple-silicon efficiency
/// cores, so a 10-thread decode measured ~2x slower than a 6-8 performance-core
/// config. Invalid or non-positive values are ignored with a warning.
pub(super) fn intra_op_threads_from_env() -> Option<i32> {
    match &runtime_config().intra_op_threads {
        IntraOpThreads::Unset => None,
        IntraOpThreads::Count(threads) => Some(*threads),
        IntraOpThreads::Invalid(value) => {
            tracing::warn!(
                "Ignoring invalid ONNX_GENAI_INTRA_OP_THREADS={value}; expected a positive integer"
            );
            None
        }
    }
}

/// Effective ORT intra-op thread count for session creation.
///
/// Precedence is: explicit API > `ONNX_GENAI_INTRA_OP_THREADS` > platform
/// CPU default > ORT's own default (0).
pub(super) fn effective_intra_op_threads(options: &SessionOptions) -> i32 {
    if options.intra_op_num_threads > 0 {
        return options.intra_op_num_threads;
    }
    intra_op_threads_from_env()
        .or_else(|| default_cpu_ort_intra_op_threads(options))
        .unwrap_or(0)
}

fn default_cpu_ort_intra_op_threads(options: &SessionOptions) -> Option<i32> {
    if !options
        .execution_providers
        .iter()
        .all(|ep| ep.caps.is_host())
    {
        return None;
    }
    std::thread::available_parallelism()
        .ok()
        .and_then(|available| default_cpu_ort_intra_op_threads_for_available(available.get()))
}

pub(super) fn default_cpu_ort_intra_op_threads_for_available(available: usize) -> Option<i32> {
    default_cpu_ort_intra_op_threads_for_available_on(
        available,
        cfg!(all(target_os = "windows", target_arch = "aarch64")),
    )
}

pub(super) fn default_cpu_ort_intra_op_threads_for_available_on(
    available: usize,
    windows_arm64: bool,
) -> Option<i32> {
    if !windows_arm64 {
        return None;
    }

    // Match ONNX Runtime GenAI's default policy in src/models/model.cpp:
    // SetIntraOpNumThreads(min(max(1, hardware_concurrency() / 2), 16)).
    // That avoids all-core ORT CPU decode on Windows ARM64 Snapdragon parts,
    // where int4 decode is memory-bandwidth-bound rather than core-count-bound.
    //
    // TODO: refine with a cache-cluster cap once onnx-runtime-cpuinfo exposes
    // all cache records; its current wrapper only records the first L2/L3 entry,
    // so cluster detection is unreliable.
    Some(((available / 2).clamp(1, 16)) as i32)
}

/// Whether to disable WebGPU validation. Default true (safe overhead
/// reduction); set `ONNX_GENAI_WEBGPU_VALIDATION=1` to keep validation on.
pub(super) fn webgpu_disable_validation_from_env() -> bool {
    !runtime_config().webgpu_validation
}

/// CUDA attention mode from the typed runtime configuration registry.
///
/// ORT exposes the desired behavior as the CUDA provider option
/// `sdpa_kernel=16` (the standard math implementation), so this configuration
/// does not need to mutate ORT's process-wide attention environment variables.
pub(super) fn cuda_attention_mode_from_runtime_config() -> CudaAttentionMode {
    match &runtime_config().cuda_attention_mode {
        CudaAttentionMode::Invalid(invalid) => {
            tracing::warn!(
                "Ignoring invalid ONNX_GENAI_CUDA_ATTENTION={invalid}; expected 'auto', 'fused', or 'unfused'"
            );
            CudaAttentionMode::Auto
        }
        mode => mode.clone(),
    }
}

/// Whether device-resident KV buffers are enabled. Default **false**: on the
/// ORT 1.27 WebGPU EP, binding a user-pre-allocated `WebGPU_Buffer` device
/// tensor as a persistent in-place `past`/`present` share-buffer segfaults
/// (`EXC_BAD_ACCESS`, call through a null function pointer) during multi-step
/// decode. Set `ONNX_GENAI_DEVICE_KV=1` to opt in experimentally once ORT
/// supports external device KV tensors. See
/// `.squad/decisions/inbox/leon-device-resident-kv.md`.
pub(super) fn device_kv_enabled_from_env() -> bool {
    runtime_config().device_kv
}

/// Explicit operator opt-in that lets an otherwise unverified EP participate in
/// the fixed-capacity, pre-bound present-output (SharedBuffer) decode path.
///
/// WHAT: Reads `ONNX_GENAI_SHARED_KV_PRESENT_BINDING` and returns TRUE for the
/// usual truthy values (`1`/`true`/`yes`/`on`), FALSE otherwise (including
/// unset).
///
/// WHY: The verified-EP allowlist in [`fixed_capacity_present_binding_supported`]
/// gates the SharedBuffer path. The Metal plugin EP now implements the
/// fixed-capacity in-place-write GQA contract and is on that allowlist, so this
/// flag is no longer needed for Metal. It remains a global operator override so
/// an as-yet-unverified EP (e.g. CoreML) can opt into SharedBuffer without a
/// code change.
///
/// HOW: Consumed only by
/// [`Session::supports_fixed_capacity_present_binding`]; it overrides the
/// conservative capability allowlist.
pub(super) fn shared_kv_present_binding_opt_in_from_env() -> bool {
    runtime_config().shared_kv_present_binding
}

/// Resolve fixed-capacity present binding from EP capabilities, with an explicit
/// operator override for unverified EPs.
pub(super) fn fixed_capacity_present_binding_supported(
    providers: &[ResolvedEp],
    opt_in: bool,
) -> bool {
    opt_in
        || !providers.is_empty()
            && providers
                .iter()
                .all(|ep| ep.caps.has(capability::FIXED_CAPACITY_PRESENT_BINDING))
}
