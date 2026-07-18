//! Typed, process-wide configuration for ONNX GenAI runtime environment flags.
//!
//! Add new library-internal runtime flags to [`RuntimeConfig`] instead of
//! reading environment variables at their call sites.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::OnceLock;

const DEFAULT_SPEC_PROMPT: &str = "Once upon a time, there was a small robot who";
const DEFAULT_MB_PROMPT: &str = "<bos>The capital of France is";

/// An execution provider requested through `ONNX_GENAI_EP`.
///
/// Provider names are normalized but otherwise opaque. Provider-specific
/// options are forwarded unchanged by the ORT layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EpSelection {
    /// Normalized (trimmed, lowercased) provider name.
    pub name: String,
    /// Opaque provider options.
    pub options: BTreeMap<String, String>,
}

impl EpSelection {
    /// Construct a selection using the same normalization as the env parser.
    #[must_use]
    pub fn new(name: impl AsRef<str>) -> Self {
        Self {
            name: normalize_ep_name(name.as_ref()),
            options: BTreeMap::new(),
        }
    }

    /// Whether this selection names the implicit host provider.
    #[must_use]
    pub fn is_host_default(&self) -> bool {
        self.name.is_empty() || self.name == "cpu"
    }
}

/// A single execution-provider plugin fully described inline in the
/// `ONNX_GENAI_EP` priority list (e.g. `plugin:/path/ep.so|device=GPU`).
///
/// Unlike the bare `plugin` token (which is configured through the scalar
/// `ONNX_GENAI_EP_LIBRARY`/`_NAME`/`_OPTIONS`/`_DEVICE` variables and can only
/// appear once), an inline plugin carries its own library, registration name,
/// options, and device class, so several distinct plugins can be composed in
/// one `ONNX_GENAI_EP` list. Nothing here is provider-specific: the concrete
/// provider name is still discovered from the plugin at load time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PluginSpec {
    /// Path to the ORT execution-provider plugin shared library.
    pub library: PathBuf,
    /// Optional registration handle passed to ORT's
    /// `RegisterExecutionProviderLibrary`. Only an opaque handle; it does NOT
    /// need to match the provider's internal name. Defaults to one derived from
    /// the library file name when unset.
    pub registration_name: Option<String>,
    /// Provider-specific options passed straight through to ORT.
    pub options: Vec<(String, String)>,
    /// Optional hardware device class (`CPU`, `GPU`, `NPU`) used to narrow a
    /// plugin that exposes several devices down to one.
    pub device: Option<String>,
}

/// One entry in the `ONNX_GENAI_EP` execution-provider priority list.
///
/// The list is ordered by descending priority: ORT tries the first entry's
/// provider first, then falls back to later entries for nodes it cannot claim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionProviderEntry {
    /// A built-in provider or the bare `plugin` token (configured through the
    /// scalar `ONNX_GENAI_EP_*` variables).
    Builtin(EpSelection),
    /// A plugin fully described inline in the list.
    Plugin(PluginSpec),
}

/// Parsed `ONNX_GENAI_CUDA_DEVICE` value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaDevice {
    /// A valid non-negative CUDA device identifier.
    Id(i32),
    /// An invalid value retained so the ORT layer can preserve its warning.
    Invalid(String),
}

/// Parsed `ONNX_GENAI_INTRA_OP_THREADS` value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IntraOpThreads {
    /// No override was provided.
    Unset,
    /// A valid positive thread count.
    Count(i32),
    /// An invalid value retained so the ORT layer can preserve its warning.
    Invalid(String),
}

/// Complete registry of ONNX GenAI library-internal runtime knobs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeConfig {
    /// `ONNX_GENAI_EP` (`Vec<ExecutionProviderEntry>`, default: empty): the
    /// ordered execution-provider priority list. Parsed from a comma-separated
    /// list of tokens (built-ins such as `cuda`/`webgpu`, the bare `plugin`
    /// token, and/or inline `plugin:<path>|attr=..` plugins). Multiple plugins
    /// and built-ins can be composed; ORT tries them in order. Empty means no
    /// override (auto-detect).
    pub execution_providers: Vec<ExecutionProviderEntry>,
    /// `ONNX_GENAI_CUDA_DEVICE` (`CudaDevice`, default: device 0): selects the CUDA device.
    pub cuda_device: CudaDevice,
    /// `ONNX_GENAI_INTRA_OP_THREADS` (`IntraOpThreads`, default: unset): overrides ORT intra-op threads when positive.
    pub intra_op_threads: IntraOpThreads,
    /// `ONNX_GENAI_WEBGPU_VALIDATION` (`bool`, default: false): keeps WebGPU validation enabled.
    pub webgpu_validation: bool,
    /// `ONNX_GENAI_WEBGPU_GRAPH_CAPTURE` (`bool`, default: false): enables WebGPU graph capture.
    pub webgpu_graph_capture: bool,
    /// `ONNX_GENAI_CUDA_GRAPH` (`bool`, default: false): enables CUDA graph capture.
    pub cuda_graph: bool,
    /// Whether `ONNX_GENAI_CUDA_GRAPH` was explicitly set in the environment.
    ///
    /// Lets callers distinguish "unset" (so they may auto-enable CUDA graph
    /// capture for models that support it) from an explicit opt-out
    /// (`ONNX_GENAI_CUDA_GRAPH=0`), which must always be honored.
    pub cuda_graph_explicit: bool,
    /// `ONNX_GENAI_DEVICE_KV` (`bool`, default: false): opts the experimental WebGPU
    /// device-resident KV buffers in. CUDA device-resident KV is always on and does
    /// not require this flag.
    pub device_kv: bool,
    /// `ONNX_GENAI_SHARED_KV_PRESENT_BINDING` (`bool`, default: false): opts unverified EPs into fixed-capacity present binding.
    pub shared_kv_present_binding: bool,
    /// `ONNX_GENAI_METAL_EP_LIB` (`PathBuf`, default: unset): points to the external Metal EP dynamic library.
    pub metal_ep_lib: Option<PathBuf>,
    /// `ONNX_GENAI_EP_LIBRARY` (`PathBuf`, default: unset): points to a generic
    /// ORT execution-provider plugin shared library (e.g. the
    /// `onnxruntime-ep-openvino` plugin). Used when `ONNX_GENAI_EP=plugin`.
    pub ep_library: Option<PathBuf>,
    /// `ONNX_GENAI_EP_NAME` (`String`, default: unset): optional registration
    /// name for the plugin library. Only a handle passed to ORT's
    /// `RegisterExecutionProviderLibrary`; it does NOT need to match the
    /// provider's internal name. Defaults to one derived from the library file
    /// name when unset.
    pub ep_registration_name: Option<String>,
    /// `ONNX_GENAI_EP_OPTIONS` (`Vec<(String, String)>`, default: empty):
    /// provider-specific options for the plugin EP, parsed from a
    /// `key=value,key=value` list and passed straight through to ORT. The keys
    /// are provider-defined, so nothing here is hardcoded.
    pub ep_options: Vec<(String, String)>,
    /// `ONNX_GENAI_EP_DEVICE` (`String`, default: unset): optional hardware
    /// device class used to narrow a plugin EP that exposes several devices
    /// (e.g. OpenVINO advertising both GPU and CPU) down to one. Matched against
    /// ORT's generic `OrtHardwareDeviceType` (`CPU`, `GPU`, `NPU`) — a portable
    /// class, not a provider-specific device name, so nothing here is hardcoded.
    pub ep_device: Option<String>,
    /// `ONNX_GENAI_PROFILE` (`bool`, default: false): enables aggregate per-stage profiling.
    pub profile: bool,
    /// `ONNX_GENAI_TRACE` (`PathBuf`, default: unset): writes a Perfetto timeline to this non-empty path.
    pub trace: Option<PathBuf>,
    /// `ONNX_GENAI_FIM_MODEL_DIR` (`PathBuf`, default: unset): selects the manual FIM test model directory.
    pub fim_model_dir: Option<PathBuf>,
    /// `ONNX_GENAI_SPEC_TARGET` (`PathBuf`, default: unset): overrides the speculative benchmark target model.
    pub spec_target: Option<PathBuf>,
    /// `ONNX_GENAI_SPEC_DRAFT` (`PathBuf`, default: unset): overrides the speculative benchmark draft model.
    pub spec_draft: Option<PathBuf>,
    /// `ONNX_GENAI_SPEC_PROMPT` (`String`, default: built-in TinyStories prompt): sets the speculative benchmark prompt.
    pub spec_prompt: String,
    /// `ONNX_GENAI_SPEC_MAX_NEW_TOKENS` (`usize`, default: 32): sets the speculative benchmark token budget.
    pub spec_max_new_tokens: usize,
    /// `ONNX_GENAI_SPEC_K` (`usize`, default: 4, minimum: 1): sets speculative tokens per draft step.
    pub spec_k: usize,
    /// `ONNX_GENAI_SPEC_ALLOW_SLOW` (`bool`, default: false): disables the speculative speedup assertion when present.
    pub spec_allow_slow: bool,
    /// `ONNX_GENAI_MB_FULL` (`PathBuf`, default: unset): selects the merged Milestone B model package.
    pub mb_full: Option<PathBuf>,
    /// `ONNX_GENAI_MB_TARGET` (`PathBuf`, default: unset): selects the target-only Milestone B model package.
    pub mb_target: Option<PathBuf>,
    /// `ONNX_GENAI_MB_PROMPT` (`String`, default: built-in capital-of-France prompt): sets the Milestone B prompt.
    pub mb_prompt: String,
    /// `ONNX_GENAI_MB_MAX` (`usize`, default: 64): sets the Milestone B token budget.
    pub mb_max: usize,
}

impl RuntimeConfig {
    /// Parse a configuration snapshot from the current process environment.
    #[must_use]
    pub fn from_env() -> Self {
        Self::from_os_fn(|name| std::env::var_os(name))
    }

    /// Parse a configuration snapshot through a testable UTF-8 lookup function.
    #[must_use]
    pub fn from_fn<F>(lookup: F) -> Self
    where
        F: Fn(&str) -> Option<String>,
    {
        Self::from_os_fn(|name| lookup(name).map(OsString::from))
    }

    fn from_os_fn<F>(lookup: F) -> Self
    where
        F: Fn(&str) -> Option<OsString>,
    {
        let mut execution_providers = env_string(&lookup, "ONNX_GENAI_EP")
            .map(|value| parse_execution_provider_list(&value))
            .unwrap_or_default();
        let ep_options = env_string(&lookup, "ONNX_GENAI_EP_OPTIONS")
            .map(|value| parse_key_value_list(&value))
            .unwrap_or_default();
        for entry in &mut execution_providers {
            if let ExecutionProviderEntry::Builtin(selection) = entry {
                selection.options.extend(ep_options.iter().cloned());
            }
        }
        let cuda_device = match env_string(&lookup, "ONNX_GENAI_CUDA_DEVICE") {
            Some(value) => value
                .trim()
                .parse::<i32>()
                .ok()
                .filter(|device| *device >= 0)
                .map(CudaDevice::Id)
                .unwrap_or_else(|| CudaDevice::Invalid(value)),
            None => CudaDevice::Id(0),
        };
        let intra_op_threads = match env_string(&lookup, "ONNX_GENAI_INTRA_OP_THREADS") {
            Some(value) => match value.trim().parse::<i32>() {
                Ok(threads) if threads > 0 => IntraOpThreads::Count(threads),
                _ => IntraOpThreads::Invalid(value),
            },
            None => IntraOpThreads::Unset,
        };

        Self {
            execution_providers,
            cuda_device,
            intra_op_threads,
            webgpu_validation: env_bool(&lookup, "ONNX_GENAI_WEBGPU_VALIDATION", false),
            webgpu_graph_capture: env_bool(&lookup, "ONNX_GENAI_WEBGPU_GRAPH_CAPTURE", false),
            cuda_graph: env_bool(&lookup, "ONNX_GENAI_CUDA_GRAPH", false),
            cuda_graph_explicit: lookup("ONNX_GENAI_CUDA_GRAPH").is_some(),
            device_kv: env_bool(&lookup, "ONNX_GENAI_DEVICE_KV", false),
            shared_kv_present_binding: env_bool(
                &lookup,
                "ONNX_GENAI_SHARED_KV_PRESENT_BINDING",
                false,
            ),
            metal_ep_lib: env_path(&lookup, "ONNX_GENAI_METAL_EP_LIB")
                .filter(|path| !path.as_os_str().is_empty()),
            ep_library: env_path(&lookup, "ONNX_GENAI_EP_LIBRARY")
                .filter(|path| !path.as_os_str().is_empty()),
            ep_registration_name: env_string(&lookup, "ONNX_GENAI_EP_NAME")
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty()),
            ep_options,
            ep_device: env_string(&lookup, "ONNX_GENAI_EP_DEVICE")
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty()),
            profile: matches!(
                env_string(&lookup, "ONNX_GENAI_PROFILE").as_deref(),
                Some("1" | "true" | "yes")
            ),
            trace: env_utf8_path(&lookup, "ONNX_GENAI_TRACE")
                .filter(|path| !path.as_os_str().is_empty()),
            fim_model_dir: env_utf8_path(&lookup, "ONNX_GENAI_FIM_MODEL_DIR"),
            spec_target: env_path(&lookup, "ONNX_GENAI_SPEC_TARGET"),
            spec_draft: env_path(&lookup, "ONNX_GENAI_SPEC_DRAFT"),
            spec_prompt: env_string(&lookup, "ONNX_GENAI_SPEC_PROMPT")
                .unwrap_or_else(|| DEFAULT_SPEC_PROMPT.to_owned()),
            spec_max_new_tokens: env_usize(&lookup, "ONNX_GENAI_SPEC_MAX_NEW_TOKENS", 32),
            spec_k: env_usize(&lookup, "ONNX_GENAI_SPEC_K", 4).max(1),
            spec_allow_slow: lookup("ONNX_GENAI_SPEC_ALLOW_SLOW").is_some(),
            mb_full: env_utf8_path(&lookup, "ONNX_GENAI_MB_FULL"),
            mb_target: env_utf8_path(&lookup, "ONNX_GENAI_MB_TARGET"),
            mb_prompt: env_string(&lookup, "ONNX_GENAI_MB_PROMPT")
                .unwrap_or_else(|| DEFAULT_MB_PROMPT.to_owned()),
            mb_max: env_usize(&lookup, "ONNX_GENAI_MB_MAX", 64),
        }
    }
}

/// Return the process-wide runtime configuration, parsed on first use.
#[must_use]
pub fn runtime_config() -> &'static RuntimeConfig {
    static CONFIG: OnceLock<RuntimeConfig> = OnceLock::new();
    CONFIG.get_or_init(RuntimeConfig::from_env)
}

fn env_bool<F>(lookup: &F, name: &str, default: bool) -> bool
where
    F: Fn(&str) -> Option<OsString>,
{
    env_string(lookup, name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn env_usize<F>(lookup: &F, name: &str, default: usize) -> usize
where
    F: Fn(&str) -> Option<OsString>,
{
    env_string(lookup, name)
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_path<F>(lookup: &F, name: &str) -> Option<PathBuf>
where
    F: Fn(&str) -> Option<OsString>,
{
    lookup(name).map(PathBuf::from)
}

fn env_utf8_path<F>(lookup: &F, name: &str) -> Option<PathBuf>
where
    F: Fn(&str) -> Option<OsString>,
{
    env_string(lookup, name).map(PathBuf::from)
}

fn env_string<F>(lookup: &F, name: &str) -> Option<String>
where
    F: Fn(&str) -> Option<OsString>,
{
    lookup(name).and_then(|value| value.into_string().ok())
}

/// Normalize an execution-provider name consistently for env and API callers.
#[must_use]
pub fn normalize_ep_name(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

fn parse_execution_provider(value: &str) -> EpSelection {
    EpSelection::new(value)
}

/// Parse the `ONNX_GENAI_EP` value into an ordered execution-provider priority
/// list.
///
/// Tokens are comma-separated and tried in order. Each token is one of:
/// * a built-in provider name (`cpu`, `webgpu`, `cuda`, `metal`, `coreml`),
/// * the bare `plugin` token (configured through the scalar `ONNX_GENAI_EP_*`
///   variables — kept for backward compatibility, may appear once), or
/// * an inline plugin `plugin:<library>[|name=<n>][|device=<class>][|opt.<k>=<v>]...`.
///
/// Empty tokens are skipped; if the whole value has no usable token (e.g. an
/// explicit empty string) the list falls back to a single CPU entry, matching
/// the historical single-value behavior. Nothing here hardcodes a plugin's
/// provider name — inline plugins only carry a library path and passthrough
/// options; the concrete provider is discovered at load time.
fn parse_execution_provider_list(value: &str) -> Vec<ExecutionProviderEntry> {
    let mut entries: Vec<ExecutionProviderEntry> = value
        .split(',')
        .map(|token| token.trim())
        .filter(|token| !token.is_empty())
        .map(parse_execution_provider_entry)
        .collect();
    if entries.is_empty() {
        entries.push(ExecutionProviderEntry::Builtin(EpSelection::new("cpu")));
    }
    entries
}

fn parse_execution_provider_entry(token: &str) -> ExecutionProviderEntry {
    // Detect an inline plugin (`plugin:<...>`) case-insensitively on the scheme
    // only, so the library path itself keeps its original case (important on
    // case-sensitive filesystems and for Windows drive letters).
    if let Some((scheme, rest)) = token.split_once(':') {
        if scheme.trim().eq_ignore_ascii_case("plugin") {
            return ExecutionProviderEntry::Plugin(parse_inline_plugin_spec(rest));
        }
    }
    ExecutionProviderEntry::Builtin(parse_execution_provider(token))
}

/// Parse the portion of an inline plugin token after `plugin:`.
///
/// Layout: `<library>[|name=<n>][|device=<class>][|opt.<k>=<v>]...`. The first
/// `|`-separated segment is the library path; later segments are `key=value`
/// attributes: `name`, `device`, or `opt.<option-key>` (the option key keeps
/// its original case and is passed straight through to ORT). Unknown attribute
/// keys are ignored.
fn parse_inline_plugin_spec(rest: &str) -> PluginSpec {
    let mut segments = rest.split('|');
    let library = segments.next().unwrap_or("").trim();
    let mut registration_name = None;
    let mut device = None;
    let mut options = Vec::new();
    for attr in segments {
        let attr = attr.trim();
        if attr.is_empty() {
            continue;
        }
        let Some((key, val)) = attr.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let val = val.trim();
        let key_lc = key.to_ascii_lowercase();
        if key_lc == "name" {
            if !val.is_empty() {
                registration_name = Some(val.to_owned());
            }
        } else if key_lc == "device" {
            if !val.is_empty() {
                device = Some(val.to_owned());
            }
        } else if let Some(option_key) = key_lc
            .starts_with("opt.")
            .then(|| key[4..].trim())
        {
            if !option_key.is_empty() {
                options.push((option_key.to_owned(), val.to_owned()));
            }
        }
    }
    PluginSpec {
        library: PathBuf::from(library),
        registration_name,
        options,
        device,
    }
}

/// Parse a `key=value,key=value` list into ordered pairs.
///
/// Whitespace around keys, values, and separators is trimmed. Entries without
/// an `=`, or with an empty key, are skipped. Values may be empty. The parse is
/// intentionally provider-agnostic: keys and values are passed through verbatim
/// so no execution-provider option name is hardcoded.
fn parse_key_value_list(value: &str) -> Vec<(String, String)> {
    value
        .split(',')
        .filter_map(|entry| {
            let (key, val) = entry.split_once('=')?;
            let key = key.trim();
            if key.is_empty() {
                return None;
            }
            Some((key.to_owned(), val.trim().to_owned()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use super::{
        CudaDevice, EpSelection, ExecutionProviderEntry, IntraOpThreads, PluginSpec, RuntimeConfig,
    };

    fn config(entries: &[(&str, &str)]) -> RuntimeConfig {
        let values: HashMap<&str, &str> = entries.iter().copied().collect();
        RuntimeConfig::from_fn(|name| values.get(name).map(|value| (*value).to_owned()))
    }

    fn ep(name: &str) -> ExecutionProviderEntry {
        ExecutionProviderEntry::Builtin(EpSelection::new(name))
    }

    #[test]
    fn every_flag_has_its_existing_default() {
        let actual = config(&[]);
        assert_eq!(actual.execution_providers, Vec::new());
        assert_eq!(actual.cuda_device, CudaDevice::Id(0));
        assert_eq!(actual.intra_op_threads, IntraOpThreads::Unset);
        assert!(!actual.webgpu_validation);
        assert!(!actual.webgpu_graph_capture);
        assert!(!actual.cuda_graph);
        assert!(!actual.cuda_graph_explicit);
        assert!(!actual.device_kv);
        assert!(!actual.shared_kv_present_binding);
        assert_eq!(actual.metal_ep_lib, None);
        assert!(!actual.profile);
        assert_eq!(actual.trace, None);
        assert_eq!(actual.fim_model_dir, None);
        assert_eq!(actual.spec_target, None);
        assert_eq!(actual.spec_draft, None);
        assert_eq!(
            actual.spec_prompt,
            "Once upon a time, there was a small robot who"
        );
        assert_eq!(actual.spec_max_new_tokens, 32);
        assert_eq!(actual.spec_k, 4);
        assert!(!actual.spec_allow_slow);
        assert_eq!(actual.mb_full, None);
        assert_eq!(actual.mb_target, None);
        assert_eq!(actual.mb_prompt, "<bos>The capital of France is");
        assert_eq!(actual.mb_max, 64);
    }

    #[test]
    fn standard_boolean_flags_accept_existing_truthy_variants() {
        for truthy in ["1", "true", "yes", "on", " TRUE ", "On"] {
            let actual = config(&[
                ("ONNX_GENAI_WEBGPU_VALIDATION", truthy),
                ("ONNX_GENAI_WEBGPU_GRAPH_CAPTURE", truthy),
                ("ONNX_GENAI_CUDA_GRAPH", truthy),
                ("ONNX_GENAI_DEVICE_KV", truthy),
                ("ONNX_GENAI_SHARED_KV_PRESENT_BINDING", truthy),
            ]);
            assert!(actual.webgpu_validation);
            assert!(actual.webgpu_graph_capture);
            assert!(actual.cuda_graph);
            assert!(actual.device_kv);
            assert!(actual.shared_kv_present_binding);
        }

        let actual = config(&[
            ("ONNX_GENAI_WEBGPU_VALIDATION", "0"),
            ("ONNX_GENAI_DEVICE_KV", ""),
            ("ONNX_GENAI_CUDA_GRAPH", "no"),
        ]);
        assert!(!actual.webgpu_validation);
        assert!(!actual.device_kv);
        assert!(!actual.cuda_graph);
        // Explicit "no" still counts as an explicit opt-out.
        assert!(actual.cuda_graph_explicit);
    }

    #[test]
    fn profile_preserves_its_narrow_case_sensitive_truthy_semantics() {
        for truthy in ["1", "true", "yes"] {
            assert!(config(&[("ONNX_GENAI_PROFILE", truthy)]).profile);
        }
        for falsey in ["on", "TRUE", " true ", "0", ""] {
            assert!(!config(&[("ONNX_GENAI_PROFILE", falsey)]).profile);
        }
    }

    #[test]
    fn execution_provider_normalizes_and_retains_generic_names() {
        let cases = [
            ("", ""),
            (" CPU ", "cpu"),
            ("web-gpu", "web-gpu"),
            ("WEB_GPU", "web_gpu"),
            ("cuda", "cuda"),
            ("metal", "metal"),
            ("core-ml", "core-ml"),
            ("plugin", "plugin"),
            ("EP-Plugin", "ep-plugin"),
            (" Unknown-EP ", "unknown-ep"),
        ];
        for (value, expected_name) in cases {
            assert_eq!(
                config(&[("ONNX_GENAI_EP", value)]).execution_providers,
                vec![ep(expected_name)]
            );
        }
    }

    #[test]
    fn execution_provider_list_parses_ordered_priority_entries() {
        let actual = config(&[("ONNX_GENAI_EP", " cuda , webgpu ,cpu")]);
        assert_eq!(
            actual.execution_providers,
            vec![ep("cuda"), ep("webgpu"), ep("cpu")]
        );
    }

    #[test]
    fn execution_provider_list_skips_empty_tokens_but_keeps_bare_empty_as_cpu() {
        // A trailing comma yields an empty token that is skipped.
        assert_eq!(
            config(&[("ONNX_GENAI_EP", "cuda,")]).execution_providers,
            vec![ep("cuda")]
        );
        // An explicit empty value still resolves to a single CPU entry.
        assert_eq!(
            config(&[("ONNX_GENAI_EP", "")]).execution_providers,
            vec![ep("cpu")]
        );
    }

    #[test]
    fn execution_provider_list_parses_inline_plugins_with_attributes() {
        let actual = config(&[(
            "ONNX_GENAI_EP",
            "cuda,plugin:C:\\ep\\openvino.dll|device=GPU|name=ov|opt.device_type=GPU|opt.num_streams=2,\
             plugin:/opt/other_ep.so",
        )]);
        assert_eq!(
            actual.execution_providers,
            vec![
                ep("cuda"),
                ExecutionProviderEntry::Plugin(PluginSpec {
                    library: PathBuf::from("C:\\ep\\openvino.dll"),
                    registration_name: Some("ov".to_owned()),
                    options: vec![
                        ("device_type".to_owned(), "GPU".to_owned()),
                        ("num_streams".to_owned(), "2".to_owned()),
                    ],
                    device: Some("GPU".to_owned()),
                }),
                ExecutionProviderEntry::Plugin(PluginSpec {
                    library: PathBuf::from("/opt/other_ep.so"),
                    registration_name: None,
                    options: Vec::new(),
                    device: None,
                }),
            ]
        );
    }

    #[test]
    fn plugin_ep_library_name_and_options_parse() {
        let actual = config(&[
            ("ONNX_GENAI_EP", "plugin"),
            ("ONNX_GENAI_EP_LIBRARY", "/opt/onnxruntime_ep_openvino.so"),
            ("ONNX_GENAI_EP_NAME", " openvino_ep "),
            (
                "ONNX_GENAI_EP_OPTIONS",
                "device_type=CPU, num_streams=2 ,=skipme, valid=",
            ),
            ("ONNX_GENAI_EP_DEVICE", " GPU "),
        ]);
        assert_eq!(actual.execution_providers.len(), 1);
        let ExecutionProviderEntry::Builtin(selection) = &actual.execution_providers[0] else {
            panic!("expected built-in plugin selection");
        };
        assert_eq!(selection.name, "plugin");
        assert_eq!(selection.options.get("device_type").map(String::as_str), Some("CPU"));
        assert_eq!(
            actual.ep_library,
            Some(PathBuf::from("/opt/onnxruntime_ep_openvino.so"))
        );
        assert_eq!(actual.ep_registration_name.as_deref(), Some("openvino_ep"));
        assert_eq!(actual.ep_device.as_deref(), Some("GPU"));
        assert_eq!(
            actual.ep_options,
            vec![
                ("device_type".to_owned(), "CPU".to_owned()),
                ("num_streams".to_owned(), "2".to_owned()),
                ("valid".to_owned(), String::new()),
            ]
        );
    }

    #[test]
    fn plugin_ep_library_and_name_reject_empty() {
        let actual = config(&[
            ("ONNX_GENAI_EP", "plugin"),
            ("ONNX_GENAI_EP_LIBRARY", ""),
            ("ONNX_GENAI_EP_NAME", "   "),
        ]);
        assert_eq!(actual.ep_library, None);
        assert_eq!(actual.ep_registration_name, None);
        assert!(actual.ep_options.is_empty());
        assert_eq!(actual.ep_device, None);
    }

    #[test]
    fn integer_flags_preserve_validation_and_fallback_rules() {
        let actual = config(&[
            ("ONNX_GENAI_CUDA_DEVICE", " 3 "),
            ("ONNX_GENAI_INTRA_OP_THREADS", " 8 "),
            ("ONNX_GENAI_SPEC_MAX_NEW_TOKENS", "48"),
            ("ONNX_GENAI_SPEC_K", "0"),
            ("ONNX_GENAI_MB_MAX", "96"),
        ]);
        assert_eq!(actual.cuda_device, CudaDevice::Id(3));
        assert_eq!(actual.intra_op_threads, IntraOpThreads::Count(8));
        assert_eq!(actual.spec_max_new_tokens, 48);
        assert_eq!(actual.spec_k, 1);
        assert_eq!(actual.mb_max, 96);

        let invalid = config(&[
            ("ONNX_GENAI_CUDA_DEVICE", "-1"),
            ("ONNX_GENAI_INTRA_OP_THREADS", "0"),
            ("ONNX_GENAI_SPEC_MAX_NEW_TOKENS", " 48 "),
            ("ONNX_GENAI_SPEC_K", "bad"),
            ("ONNX_GENAI_MB_MAX", "-1"),
        ]);
        assert_eq!(invalid.cuda_device, CudaDevice::Invalid("-1".to_owned()));
        assert_eq!(
            invalid.intra_op_threads,
            IntraOpThreads::Invalid("0".to_owned())
        );
        assert_eq!(invalid.spec_max_new_tokens, 32);
        assert_eq!(invalid.spec_k, 4);
        assert_eq!(invalid.mb_max, 64);
    }

    #[test]
    fn paths_strings_and_presence_flags_preserve_existing_rules() {
        let actual = config(&[
            ("ONNX_GENAI_METAL_EP_LIB", "/opt/lib/libmlx.dylib"),
            ("ONNX_GENAI_TRACE", "trace.json"),
            ("ONNX_GENAI_FIM_MODEL_DIR", ""),
            ("ONNX_GENAI_SPEC_TARGET", "target"),
            ("ONNX_GENAI_SPEC_DRAFT", "draft"),
            ("ONNX_GENAI_SPEC_PROMPT", ""),
            ("ONNX_GENAI_SPEC_ALLOW_SLOW", ""),
            ("ONNX_GENAI_MB_FULL", "full"),
            ("ONNX_GENAI_MB_TARGET", "target-only"),
            ("ONNX_GENAI_MB_PROMPT", "prompt"),
        ]);
        assert_eq!(
            actual.metal_ep_lib,
            Some(PathBuf::from("/opt/lib/libmlx.dylib"))
        );
        assert_eq!(actual.trace, Some(PathBuf::from("trace.json")));
        assert_eq!(actual.fim_model_dir, Some(PathBuf::new()));
        assert_eq!(actual.spec_target, Some(PathBuf::from("target")));
        assert_eq!(actual.spec_draft, Some(PathBuf::from("draft")));
        assert_eq!(actual.spec_prompt, "");
        assert!(actual.spec_allow_slow);
        assert_eq!(actual.mb_full, Some(PathBuf::from("full")));
        assert_eq!(actual.mb_target, Some(PathBuf::from("target-only")));
        assert_eq!(actual.mb_prompt, "prompt");

        let empty = config(&[("ONNX_GENAI_METAL_EP_LIB", ""), ("ONNX_GENAI_TRACE", "")]);
        assert_eq!(empty.metal_ep_lib, None);
        assert_eq!(empty.trace, None);
    }
}
