use std::ffi::{CStr, CString};

use onnx_genai_runtime_config::{EpSelection, PluginSpec};

use crate::{Environment, OrtError, Result};

use super::ep_compat::{self, EpCapabilities, HardwareKind, ResolvedEp};
use super::providers::parse_hardware_device_type;

pub(super) fn resolve_inline_plugin(spec: &PluginSpec) -> Option<ResolvedEp> {
    if spec.library.as_os_str().is_empty() {
        tracing::warn!("Ignoring inline plugin entry with an empty library path");
        return None;
    }
    Some(resolve_plugin_selection(
        EpSelection::new("plugin"),
        spec.library.clone(),
        spec.registration_name
            .clone()
            .unwrap_or_else(|| plugin_registration_name_from_path(&spec.library)),
        spec.options.clone(),
        spec.device.clone(),
    ))
}

pub(super) fn resolve_plugin_selection(
    selection: EpSelection,
    library: std::path::PathBuf,
    registration_name: String,
    options: Vec<(String, String)>,
    device: Option<String>,
) -> ResolvedEp {
    let hardware = match device.as_deref().map(str::to_ascii_uppercase).as_deref() {
        Some("CPU") => HardwareKind::Cpu,
        Some("GPU") => HardwareKind::Gpu,
        Some("NPU") => HardwareKind::Npu,
        _ => HardwareKind::Other,
    };
    ResolvedEp {
        caps: EpCapabilities::new(selection.name.clone(), hardware, None, None, &[]),
        selection,
        strategy: ep_compat::AppendStrategy::PluginLibrary {
            lib: library,
            registration_name,
            options,
            device,
        },
        graph_capture_env: false,
        transitional_webgpu: false,
    }
}

/// Derive a stable registration handle for a plugin library from its file name.
///
/// This is only an opaque handle passed to ORT's
/// `RegisterExecutionProviderLibrary`; it does not need to match (and must not
/// be confused with) the provider's internal EP name.
pub(super) fn plugin_registration_name_from_path(path: &std::path::Path) -> String {
    path.file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .filter(|stem| !stem.is_empty())
        .unwrap_or_else(|| "onnx_genai_ep_plugin".to_string())
}

/// Register an ORT execution-provider plugin shared library and append every
/// device it contributes to `session_options`.
///
/// The plugin's provider is identified WITHOUT hardcoding its name: we snapshot
/// the environment's EP devices before registration and append only the devices
/// that appear afterwards. This mirrors the documented plugin-EP registration
/// flow (`RegisterExecutionProviderLibrary` + `GetEpDevices` +
/// `SessionOptionsAppendExecutionProvider_V2`) used by packages such as
/// `onnxruntime-ep-openvino`, so it works for any ORT >= 1.22 plugin EP
/// (OpenVINO, NV TensorRT RTX, QNN, ...).
pub(super) fn append_plugin_execution_provider(
    env: &Environment,
    session_options: *mut onnx_genai_ort_sys::OrtSessionOptions,
    registration_name: &str,
    plugin_path: &std::path::Path,
    options: &[(String, String)],
    device_class: Option<&str>,
) -> Result<()> {
    if !plugin_path.is_file() {
        return Err(OrtError::InvalidArgument(format!(
            "execution provider plugin library not found at {}",
            plugin_path.display()
        )));
    }

    let api = crate::error::api()?;
    let get_ep_devices = api
        .GetEpDevices
        .ok_or(OrtError::ApiUnavailable("GetEpDevices"))?;
    let ep_name = api
        .EpDevice_EpName
        .ok_or(OrtError::ApiUnavailable("EpDevice_EpName"))?;
    let append = api
        .SessionOptionsAppendExecutionProvider_V2
        .ok_or(OrtError::ApiUnavailable(
            "SessionOptionsAppendExecutionProvider_V2",
        ))?;

    // ORT plugin registration is process-global. Keep the device snapshot,
    // registration, and provider-name cache update atomic with respect to other
    // environments registering plugins concurrently.
    let discovery_guard = env.lock_plugin_discovery()?;

    // Query the environment's current EP devices as a list of raw pointers.
    let query_devices = || -> Result<Vec<*const onnx_genai_ort_sys::OrtEpDevice>> {
        let mut devices_ptr: *const *const onnx_genai_ort_sys::OrtEpDevice = std::ptr::null();
        let mut count = 0usize;
        // SAFETY: the environment is live; both output pointers are valid.
        crate::error::check_status(unsafe {
            get_ep_devices(env.as_ptr(), &mut devices_ptr, &mut count)
        })?;
        let mut out = Vec::new();
        if !devices_ptr.is_null() {
            for index in 0..count {
                // SAFETY: ORT returned an array of `count` entries.
                let device = unsafe { *devices_ptr.add(index) };
                if !device.is_null() {
                    out.push(device);
                }
            }
        }
        Ok(out)
    };
    // Read an EP device's provider name (discovered, never hardcoded).
    let name_of = |device: *const onnx_genai_ort_sys::OrtEpDevice| -> Option<String> {
        // SAFETY: `device` is owned by the live environment.
        let name_ptr = unsafe { ep_name(device) };
        if name_ptr.is_null() {
            return None;
        }
        // SAFETY: ORT EP names are NUL-terminated strings.
        Some(
            unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned(),
        )
    };

    // Snapshot the EP-name multiset before registering so we can identify the
    // devices the plugin contributes without knowing its name in advance.
    let mut before_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for device in query_devices()? {
        if let Some(name) = name_of(device) {
            *before_counts.entry(name).or_insert(0) += 1;
        }
    }
    let before = before_counts;

    let newly_registered =
        env.register_execution_provider_library(registration_name, plugin_path)?;

    // After registration, group the environment's EP devices by provider name.
    let after: Vec<(*const onnx_genai_ort_sys::OrtEpDevice, String)> = query_devices()?
        .into_iter()
        .filter_map(|device| name_of(device).map(|name| (device, name)))
        .collect();

    // Determine the plugin's provider name.
    //
    // On the first registration for this handle, the provider is discovered by
    // the before/after device diff: the name whose device count grew (never
    // hardcoded). When the library was already registered on this shared
    // environment (e.g. a second session such as a speculative-decode draft),
    // the diff is empty because the devices were already present, so we reuse
    // the provider name discovered on the first registration instead.
    let target_name = if newly_registered {
        let mut after_counts: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for (_, name) in &after {
            *after_counts.entry(name.as_str()).or_insert(0) += 1;
        }
        let mut new_names: Vec<&str> = after_counts
            .iter()
            .filter(|(name, count)| **count > before.get(**name).copied().unwrap_or(0))
            .map(|(name, _)| *name)
            .collect();
        new_names.sort_unstable();

        // A plugin may expose several provider groupings (e.g. OpenVINO registers
        // both `OpenVINOExecutionProvider` and virtual `OpenVINOExecutionProvider.AUTO`
        // devices). ORT requires every device appended in one call to share a single
        // EP, so choose one provider group deterministically: prefer the base
        // provider name (no `.` suffix) over virtual variants, else the first sorted.
        let discovered = new_names
            .iter()
            .find(|name| !name.contains('.'))
            .or_else(|| new_names.first())
            .map(|name| (*name).to_owned());
        match discovered {
            Some(name) => {
                env.cache_plugin_provider(registration_name, &name)?;
                name
            }
            None => {
                return Err(OrtError::InvalidArgument(format!(
                    "execution provider plugin '{registration_name}' registered from {} but contributed no new execution-provider devices",
                    plugin_path.display()
                )));
            }
        }
    } else {
        match env.cached_plugin_provider(registration_name)? {
            Some(name) => name,
            None => {
                return Err(OrtError::InvalidArgument(format!(
                    "execution provider plugin '{registration_name}' (from {}) was already registered but its provider name is unknown",
                    plugin_path.display()
                )));
            }
        }
    };
    drop(discovery_guard);

    let mut selected: Vec<*const onnx_genai_ort_sys::OrtEpDevice> = after
        .iter()
        .filter(|(_, name)| *name == target_name)
        .map(|(device, _)| *device)
        .collect();
    let selected_name = Some(target_name);

    if selected.is_empty() {
        return Err(OrtError::InvalidArgument(format!(
            "execution provider plugin '{registration_name}' registered from {} but contributed no execution-provider devices",
            plugin_path.display()
        )));
    }

    // If the caller asked for a specific hardware-device class (CPU/GPU/NPU),
    // narrow the selection to a single matching device. A plugin may expose one
    // EP name spanning several hardware devices (e.g. OpenVINO advertising both
    // GPU and CPU); ORT's `AppendExecutionProvider_V2` chooses a device from the
    // list it is given, so filtering here is how a portable device request is
    // honoured. The class is matched against ORT's generic `OrtHardwareDeviceType`
    // enum, never a provider-specific device string.
    if let Some(requested) = device_class {
        if let Some(wanted) = parse_hardware_device_type(requested) {
            match (api.EpDevice_Device, api.HardwareDevice_Type) {
                (Some(ep_device_device), Some(hw_type)) => {
                    let matching: Vec<*const onnx_genai_ort_sys::OrtEpDevice> = selected
                        .iter()
                        .copied()
                        .filter(|device| {
                            // SAFETY: `device` is owned by the live environment; the
                            // returned hardware handle is owned by ORT.
                            let hw = unsafe { ep_device_device(*device) };
                            !hw.is_null() && unsafe { hw_type(hw) } == wanted
                        })
                        .collect();
                    if matching.is_empty() {
                        return Err(OrtError::InvalidArgument(format!(
                            "execution provider plugin '{registration_name}' exposes no {requested} device; \
                             unset ONNX_GENAI_EP_DEVICE or choose an available hardware class"
                        )));
                    }
                    // Keep a single device so the plugin cannot silently fall back to
                    // a different one.
                    selected = vec![matching[0]];
                }
                _ => {
                    // The request cannot be honoured without device introspection;
                    // fail loudly rather than silently running on an arbitrary device.
                    return Err(OrtError::ApiUnavailable(
                        "EpDevice_Device/HardwareDevice_Type (required for ONNX_GENAI_EP_DEVICE selection)",
                    ));
                }
            }
        } else {
            tracing::warn!(
                requested,
                "ONNX_GENAI_EP_DEVICE is not a recognized hardware class (expected CPU, GPU, or NPU); ignoring"
            );
        }
    }

    // Provider options are provider-defined; pass keys/values through verbatim.
    let option_keys = options
        .iter()
        .map(|(key, _)| {
            CString::new(key.as_str())
                .map_err(|_| OrtError::InvalidArgument("EP option key contains NUL".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    let option_values = options
        .iter()
        .map(|(_, value)| {
            CString::new(value.as_str())
                .map_err(|_| OrtError::InvalidArgument("EP option value contains NUL".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    let key_ptrs = option_keys.iter().map(|k| k.as_ptr()).collect::<Vec<_>>();
    let value_ptrs = option_values.iter().map(|v| v.as_ptr()).collect::<Vec<_>>();
    let (key_ptr, value_ptr) = if options.is_empty() {
        (std::ptr::null(), std::ptr::null())
    } else {
        (key_ptrs.as_ptr(), value_ptrs.as_ptr())
    };

    // SAFETY: selected devices belong to the live environment; the session
    // options handle is valid; the key/value arrays each hold `options.len()`
    // NUL-terminated strings that outlive the call.
    crate::error::check_status(unsafe {
        append(
            session_options,
            env.as_ptr().cast_mut(),
            selected.as_ptr(),
            selected.len(),
            key_ptr,
            value_ptr,
            options.len(),
        )
    })?;
    tracing::info!(
        plugin = %plugin_path.display(),
        registration = registration_name,
        provider = selected_name.as_deref().unwrap_or("<unknown>"),
        devices = selected.len(),
        "Enabled ONNX Runtime execution provider plugin"
    );
    Ok(())
}
