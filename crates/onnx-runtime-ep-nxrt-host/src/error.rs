//! Error types for the nxrt host loader.

use std::path::PathBuf;

/// Errors that can occur when loading or interacting with an nxrt plugin.
#[derive(Debug, thiserror::Error)]
pub enum NxrtHostError {
    /// The library file could not be opened.
    #[error("failed to load nxrt plugin library at {path}: {reason}")]
    LibraryLoadFailed { path: PathBuf, reason: String },

    /// A required symbol was not found in the loaded library.
    #[error("nxrt plugin at {path} is missing required symbol \"{symbol}\": {reason}")]
    SymbolNotFound {
        path: PathBuf,
        symbol: String,
        reason: String,
    },

    /// The plugin's ABI major version does not match what this host requires.
    #[error(
        "nxrt ABI version mismatch for plugin at {path}: \
         host requires major {host_major}, plugin reports major {plugin_major} \
         (plugin {plugin_major}.{plugin_minor}). \
         Rebuild the plugin against nxrt ABI {host_major}.x"
    )]
    AbiVersionMismatch {
        path: PathBuf,
        host_major: u32,
        host_minor: u32,
        plugin_major: u32,
        plugin_minor: u32,
    },

    /// The plugin's factory function returned an error.
    #[error("nxrt plugin at {path} factory failed: {status}")]
    FactoryFailed { path: PathBuf, status: String },

    /// The plugin reported zero available devices after successful creation.
    #[error(
        "nxrt plugin at {path} advertises zero devices; \
         an EP must expose at least one device to be usable"
    )]
    ZeroDevices { path: PathBuf },

    /// A call into the loaded plugin returned an error status.
    #[error("nxrt plugin call failed: {0}")]
    PluginCallFailed(String),
}

impl NxrtHostError {
    /// Convert to the EP-API error type for trait compatibility.
    pub fn into_ep_error(self) -> onnx_runtime_ep_api::EpError {
        match &self {
            Self::LibraryLoadFailed { path, reason } => {
                onnx_runtime_ep_api::EpError::EpLoadFailed {
                    path: path.clone(),
                    reason: reason.clone(),
                }
            }
            _ => onnx_runtime_ep_api::EpError::EpLoadFailed {
                path: PathBuf::new(),
                reason: self.to_string(),
            },
        }
    }
}
