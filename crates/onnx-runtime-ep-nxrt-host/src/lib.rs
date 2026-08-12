//! Dynamic loader and Rust-trait adapter for nxrt execution-provider plugins.
//!
//! This crate owns the **inbound** half of the §524 nxrt dynamic-loading
//! contract: it resolves a shared library on disk, validates the exported ABI
//! version, and wraps the loaded plugin as a `dyn ExecutionProvider` so callers
//! cannot distinguish it from an in-process Rust EP.
//!
//! # Lifetime safety
//!
//! The loaded `Library` is stored inside an `Arc` that is shared between the
//! host adapter and every object obtained from the plugin. The `Arc` ensures
//! the library is not unloaded while any live EP instance or kernel references
//! symbols inside it.

mod error;
mod loader;
mod provider_adapter;

pub use error::NxrtHostError;
pub use loader::{load_nxrt_plugin, NxrtPlugin};
pub use provider_adapter::NxrtExecutionProvider;
