//! ORT model-package directory parsing, variant selection, and path resolution.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Component as PathComponent, Path, PathBuf};

use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{Map, Value};
use thiserror::Error;

const MANIFEST_FILE: &str = "manifest.json";
const COMPONENT_FILE: &str = "component.json";
const MAX_JSON_BYTES: u64 = 4 * 1024 * 1024;

/// Errors produced while opening, selecting, resolving, or validating a package.
#[derive(Debug, Error)]
pub enum PackageError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid JSON in {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid model package: {0}")]
    Invalid(String),
    #[error("component '{0}' was not found")]
    UnknownComponent(String),
    #[error("variant '{variant}' was not found in component '{component}'")]
    UnknownVariant { component: String, variant: String },
    #[error("no variant in component '{component}' matches {request}")]
    NoMatchingVariant { component: String, request: String },
}

/// ORT package layout policy declared inside the (untrusted) manifest.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PackageLayout {
    /// All references must remain under the package root or a shared asset.
    #[default]
    Portable,
    /// References may use absolute paths or parent traversal, but only when the
    /// embedding application has separately opted in via
    /// [`HostTrust::AllowInstalledLayout`]. The manifest declaring this layout
    /// is never, on its own, sufficient to escape the package root.
    Installed,
}

/// Caller-supplied trust policy governing whether an `installed`-layout package
/// may resolve references outside the package root.
///
/// Packages are untrusted input (see `docs/genai/MODEL_PACKAGE.md` §7), so the
/// manifest's own `layout` field must never be able to grant itself
/// filesystem-escape privileges. Confinement is therefore always enforced
/// unless the embedding application explicitly passes
/// [`HostTrust::AllowInstalledLayout`], which it should only do for packages it
/// has itself installed into a trusted location.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HostTrust {
    /// Always confine every reference to the package root, regardless of the
    /// manifest `layout`. The safe default for downloaded/untrusted packages.
    #[default]
    Confined,
    /// Permit `installed`-layout manifests to use absolute paths and `..`.
    /// Only pass this for packages the host itself trusts.
    AllowInstalledLayout,
}

/// An ORT 1.x package manifest.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub schema_version: String,
    pub package_name: Option<String>,
    pub package_version: Option<String>,
    pub description: Option<String>,
    #[serde(default)]
    pub layout: PackageLayout,
    pub components: IndexMap<String, ComponentReference>,
    #[serde(default)]
    pub shared_assets: BTreeMap<String, String>,
    #[serde(default)]
    pub additional_metadata: Map<String, Value>,
    #[serde(flatten)]
    pub unknown_fields: Map<String, Value>,
}

/// An inline component or a path to a component JSON file/directory.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ComponentReference {
    Inline(Box<Component>),
    External(String),
}

/// A package component containing named executable variants.
#[derive(Debug, Clone, Deserialize)]
pub struct Component {
    pub component_name: String,
    pub variants: IndexMap<String, Variant>,
    #[serde(default)]
    pub additional_metadata: Map<String, Value>,
    #[serde(flatten)]
    pub unknown_fields: Map<String, Value>,
}

/// A named component variant.
#[derive(Debug, Clone, Deserialize)]
pub struct Variant {
    pub variant_directory: String,
    pub ep: Option<String>,
    pub device: Option<String>,
    pub compatibility_string: Option<String>,
    #[serde(default)]
    pub executor_info: Map<String, Value>,
    #[serde(default)]
    pub additional_metadata: Map<String, Value>,
    #[serde(flatten)]
    pub unknown_fields: Map<String, Value>,
}

/// Common executable fields understood from `executor_info["nxrt"]` or
/// `executor_info["ort"]`.
#[derive(Debug, Clone, Deserialize)]
pub struct ExecutorInfo {
    pub schema_version: Option<String>,
    pub model_file: String,
    pub genai_config: Option<String>,
    pub inference_metadata: Option<String>,
    pub tokenizer: Option<String>,
    #[serde(default)]
    pub session_options: BTreeMap<String, String>,
    #[serde(default)]
    pub provider_options: BTreeMap<String, String>,
    #[serde(flatten)]
    pub unknown_fields: Map<String, Value>,
}

/// Requested package variant attributes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SelectionRequest {
    pub variant: Option<String>,
    pub execution_provider: Option<String>,
    pub precision: Option<String>,
}

impl SelectionRequest {
    /// Select a specific named variant.
    #[must_use]
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            variant: Some(name.into()),
            ..Self::default()
        }
    }

    /// Select the first manifest-ordered variant matching an execution provider.
    #[must_use]
    pub fn for_execution_provider(execution_provider: impl Into<String>) -> Self {
        Self {
            execution_provider: Some(execution_provider.into()),
            ..Self::default()
        }
    }
}

/// Fully resolved files for one selected variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedVariant {
    pub component_name: String,
    pub variant_name: String,
    pub variant_directory: PathBuf,
    pub model_path: PathBuf,
    pub genai_config_path: Option<PathBuf>,
    pub inference_metadata_path: Option<PathBuf>,
    pub tokenizer_directory: Option<PathBuf>,
}

/// Open, parsed ORT model-package directory.
#[derive(Debug, Clone)]
pub struct ModelPackage {
    root: PathBuf,
    manifest: Manifest,
    trust: HostTrust,
}

impl ModelPackage {
    /// Open and structurally validate a package directory, confining every
    /// reference to the package root regardless of the manifest `layout`.
    ///
    /// This is the safe entry point for untrusted (e.g. downloaded) packages.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, PackageError> {
        Self::open_with_trust(root, HostTrust::Confined)
    }

    /// Open a package with an explicit caller-supplied trust policy.
    ///
    /// Pass [`HostTrust::AllowInstalledLayout`] only for packages the embedding
    /// application itself trusts; with it, `installed`-layout manifests may
    /// resolve absolute paths and `..` outside the package root. The default
    /// [`HostTrust::Confined`] (used by [`ModelPackage::open`]) never permits
    /// such escapes, even when the manifest declares `"layout": "installed"`.
    pub fn open_with_trust(root: impl AsRef<Path>, trust: HostTrust) -> Result<Self, PackageError> {
        let root = root.as_ref();
        if !root.is_dir() {
            return Err(PackageError::Invalid(format!(
                "package directory does not exist: {}",
                root.display()
            )));
        }
        let manifest_path = root.join(MANIFEST_FILE);
        let manifest: Manifest = read_json(&manifest_path)?;
        validate_version(&manifest.schema_version, "package")?;
        if manifest.components.is_empty() {
            return Err(PackageError::Invalid(
                "manifest components must not be empty".to_string(),
            ));
        }
        let root = root.canonicalize().map_err(|source| PackageError::Io {
            path: root.to_path_buf(),
            source,
        })?;
        Ok(Self {
            root,
            manifest,
            trust,
        })
    }

    /// Return the parsed package manifest.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Select and resolve one executable component variant.
    pub fn select(
        &self,
        component_name: &str,
        request: &SelectionRequest,
    ) -> Result<SelectedVariant, PackageError> {
        let component = self.load_component(component_name)?;
        let (variant_name, variant) = if let Some(requested_name) = &request.variant {
            component
                .variants
                .get_key_value(requested_name)
                .ok_or_else(|| PackageError::UnknownVariant {
                    component: component_name.to_string(),
                    variant: requested_name.clone(),
                })?
        } else {
            component
                .variants
                .iter()
                .find(|(_, variant)| variant_matches(variant, request))
                .ok_or_else(|| PackageError::NoMatchingVariant {
                    component: component_name.to_string(),
                    request: describe_request(request),
                })?
        };

        self.resolve_selected(component_name, variant_name, variant)
    }

    /// Validate every component, variant, executor payload, and referenced file.
    pub fn validate(&self) -> Result<(), PackageError> {
        for component_name in self.manifest.components.keys() {
            let component = self.load_component(component_name)?;
            if component.component_name != *component_name {
                return Err(PackageError::Invalid(format!(
                    "component map key '{component_name}' does not match component_name '{}'",
                    component.component_name
                )));
            }
            if component.variants.is_empty() {
                return Err(PackageError::Invalid(format!(
                    "component '{component_name}' has no variants"
                )));
            }
            for (variant_name, variant) in &component.variants {
                self.resolve_selected(component_name, variant_name, variant)?;
            }
        }
        Ok(())
    }

    fn load_component(&self, name: &str) -> Result<Component, PackageError> {
        let reference = self
            .manifest
            .components
            .get(name)
            .ok_or_else(|| PackageError::UnknownComponent(name.to_string()))?;
        match reference {
            ComponentReference::Inline(component) => Ok(component.as_ref().clone()),
            ComponentReference::External(reference) => {
                let mut path = self.resolve_path(&self.root, reference, true)?;
                if path.is_dir() {
                    // The implicit `component.json` under an external component
                    // directory must pass the same confinement check as any
                    // other reference; a symlinked component.json could
                    // otherwise escape the package root.
                    path = self.canonicalize_confined(&path.join(COMPONENT_FILE))?;
                }
                read_json(&path)
            }
        }
    }

    fn resolve_selected(
        &self,
        component_name: &str,
        variant_name: &str,
        variant: &Variant,
    ) -> Result<SelectedVariant, PackageError> {
        let variant_directory = self.resolve_path(&self.root, &variant.variant_directory, true)?;
        if !variant_directory.is_dir() {
            return Err(PackageError::Invalid(format!(
                "variant '{variant_name}' directory is not a directory: {}",
                variant_directory.display()
            )));
        }
        let executor_info = self.load_executor_info(variant_name, variant, &variant_directory)?;
        if let Some(version) = &executor_info.schema_version {
            validate_version(version, "executor")?;
        }
        let model_path = self.resolve_path(&variant_directory, &executor_info.model_file, true)?;
        ensure_file(&model_path, "model_file")?;
        let genai_config_path = self.resolve_optional_file(
            &variant_directory,
            executor_info.genai_config.as_deref(),
            "genai_config",
        )?;
        let inference_metadata_path = self.resolve_optional_file(
            &variant_directory,
            executor_info.inference_metadata.as_deref(),
            "inference_metadata",
        )?;
        let tokenizer_directory = executor_info
            .tokenizer
            .as_deref()
            .map(|reference| self.resolve_path(&variant_directory, reference, true))
            .transpose()?;
        if let Some(path) = &tokenizer_directory
            && !path.is_dir()
        {
            return Err(PackageError::Invalid(format!(
                "tokenizer reference is not a directory: {}",
                path.display()
            )));
        }
        for (key, reference) in &executor_info.session_options {
            if matches!(
                key.as_str(),
                "session.model_external_initializers_file_folder_path" | "ep.context_file_path"
            ) {
                self.resolve_path(&variant_directory, reference, true)?;
            }
        }
        Ok(SelectedVariant {
            component_name: component_name.to_string(),
            variant_name: variant_name.to_string(),
            variant_directory,
            model_path,
            genai_config_path,
            inference_metadata_path,
            tokenizer_directory,
        })
    }

    fn load_executor_info(
        &self,
        variant_name: &str,
        variant: &Variant,
        variant_directory: &Path,
    ) -> Result<ExecutorInfo, PackageError> {
        let (namespace, value) = variant
            .executor_info
            .get_key_value("nxrt")
            .or_else(|| variant.executor_info.get_key_value("ort"))
            .ok_or_else(|| {
                PackageError::Invalid(format!(
                    "variant '{variant_name}' has no executor_info.nxrt or executor_info.ort"
                ))
            })?;
        match value {
            Value::String(reference) => {
                let path = self.resolve_path(variant_directory, reference, true)?;
                read_json(&path)
            }
            Value::Object(_) => serde_json::from_value(value.clone()).map_err(|source| {
                PackageError::Json {
                    path: PathBuf::from(format!(
                        "{MANIFEST_FILE}:components.*.variants.{variant_name}.executor_info.{namespace}"
                    )),
                    source,
                }
            }),
            _ => Err(PackageError::Invalid(format!(
                "executor_info.{namespace} for variant '{variant_name}' must be an object or path"
            ))),
        }
    }

    fn resolve_optional_file(
        &self,
        base: &Path,
        reference: Option<&str>,
        field: &str,
    ) -> Result<Option<PathBuf>, PackageError> {
        reference
            .map(|reference| {
                let path = self.resolve_path(base, reference, true)?;
                ensure_file(&path, field)?;
                Ok(path)
            })
            .transpose()
    }

    /// Whether references must be confined to the package root. Confinement is
    /// always enforced unless the manifest declares the `installed` layout AND
    /// the caller explicitly opted in via [`HostTrust::AllowInstalledLayout`].
    fn confinement_enforced(&self) -> bool {
        !(self.manifest.layout == PackageLayout::Installed
            && self.trust == HostTrust::AllowInstalledLayout)
    }

    /// Canonicalize an existing path (resolving symlinks) and, when confinement
    /// is enforced, reject it if the real path escapes the package root.
    fn canonicalize_confined(&self, path: &Path) -> Result<PathBuf, PackageError> {
        if !path.exists() {
            return Err(PackageError::Invalid(format!(
                "referenced path does not exist: {}",
                path.display()
            )));
        }
        let canonical = path.canonicalize().map_err(|source| PackageError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if self.confinement_enforced() && !canonical.starts_with(&self.root) {
            return Err(PackageError::Invalid(format!(
                "package reference resolves outside package root: {}",
                path.display()
            )));
        }
        Ok(canonical)
    }

    fn resolve_path(
        &self,
        base: &Path,
        reference: &str,
        must_exist: bool,
    ) -> Result<PathBuf, PackageError> {
        let path = if let Some(asset_reference) = reference.strip_prefix("sha256:") {
            self.resolve_shared_asset(asset_reference)?
        } else {
            let reference_path = Path::new(reference);
            if self.confinement_enforced() {
                validate_portable_reference(reference_path)?;
            }
            if reference_path.is_absolute() {
                reference_path.to_path_buf()
            } else {
                base.join(reference_path)
            }
        };
        if must_exist {
            return self.canonicalize_confined(&path);
        }
        Ok(path)
    }

    fn resolve_shared_asset(&self, reference: &str) -> Result<PathBuf, PackageError> {
        let (digest, tail) = reference
            .split_once('/')
            .map_or((reference, None), |(digest, tail)| (digest, Some(tail)));
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(PackageError::Invalid(format!(
                "invalid sha256 shared-asset reference: sha256:{reference}"
            )));
        }
        let override_path = self
            .manifest
            .shared_assets
            .get(&format!("sha256:{digest}"))
            .or_else(|| self.manifest.shared_assets.get(digest));
        let root = override_path.map_or_else(
            || {
                Ok(self
                    .root
                    .join("shared_assets")
                    .join(format!("sha256-{digest}")))
            },
            |reference| {
                let path = Path::new(reference);
                if self.confinement_enforced() {
                    validate_portable_reference(path)?;
                }
                Ok(if path.is_absolute() {
                    path.to_path_buf()
                } else {
                    self.root.join(path)
                })
            },
        )?;
        let path = tail.map_or(root.clone(), |tail| root.join(tail));
        self.canonicalize_confined(&path)
    }
}

/// Probe whether a directory contains an ORT package manifest, while avoiding
/// unrelated fixture/application files also named `manifest.json`.
#[must_use]
pub fn is_model_package_directory(path: impl AsRef<Path>) -> bool {
    let path = path.as_ref();
    let Ok(bytes) = fs::read(path.join(MANIFEST_FILE)) else {
        return false;
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return has_package_suffix(path);
    };
    value.get("schema_version").is_some() && value.get("components").is_some()
        || has_package_suffix(path)
}

fn has_package_suffix(path: &Path) -> bool {
    path.extension().is_some_and(|extension| {
        extension.eq_ignore_ascii_case("ortpackage") || extension.eq_ignore_ascii_case("nxpackage")
    })
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, PackageError> {
    let metadata = fs::metadata(path).map_err(|source| PackageError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.len() > MAX_JSON_BYTES {
        return Err(PackageError::Invalid(format!(
            "JSON file exceeds {MAX_JSON_BYTES} bytes: {}",
            path.display()
        )));
    }
    let bytes = fs::read(path).map_err(|source| PackageError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| PackageError::Json {
        path: path.to_path_buf(),
        source,
    })
}

fn validate_version(version: &str, kind: &str) -> Result<(), PackageError> {
    let (major, minor) = version.split_once('.').ok_or_else(|| {
        PackageError::Invalid(format!("{kind} schema_version must be '<major>.<minor>'"))
    })?;
    let major = major.parse::<u64>().map_err(|_| {
        PackageError::Invalid(format!("{kind} schema_version has a non-numeric major"))
    })?;
    minor.parse::<u64>().map_err(|_| {
        PackageError::Invalid(format!("{kind} schema_version has a non-numeric minor"))
    })?;
    if major != 1 {
        return Err(PackageError::Invalid(format!(
            "unsupported {kind} schema major version {major}; supported major is 1"
        )));
    }
    Ok(())
}

fn validate_portable_reference(path: &Path) -> Result<(), PackageError> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                PathComponent::ParentDir | PathComponent::RootDir | PathComponent::Prefix(_)
            )
        })
    {
        return Err(PackageError::Invalid(format!(
            "portable package reference escapes its base: {}",
            path.display()
        )));
    }
    Ok(())
}

fn ensure_file(path: &Path, field: &str) -> Result<(), PackageError> {
    if path.is_file() {
        Ok(())
    } else {
        Err(PackageError::Invalid(format!(
            "{field} is not a file: {}",
            path.display()
        )))
    }
}

fn variant_matches(variant: &Variant, request: &SelectionRequest) -> bool {
    let execution_provider_matches = request.execution_provider.as_ref().is_none_or(|requested| {
        variant
            .ep
            .as_ref()
            .is_some_and(|actual| actual.eq_ignore_ascii_case(requested))
    });
    let precision_matches = request.precision.as_ref().is_none_or(|requested| {
        variant
            .additional_metadata
            .get("precision")
            .and_then(Value::as_str)
            .is_some_and(|actual| actual.eq_ignore_ascii_case(requested))
    });
    execution_provider_matches && precision_matches
}

fn describe_request(request: &SelectionRequest) -> String {
    let mut attributes = Vec::new();
    if let Some(execution_provider) = &request.execution_provider {
        attributes.push(format!("execution provider '{execution_provider}'"));
    }
    if let Some(precision) = &request.precision {
        attributes.push(format!("precision '{precision}'"));
    }
    if attributes.is_empty() {
        "the default selection".to_string()
    } else {
        attributes.join(" and ")
    }
}
