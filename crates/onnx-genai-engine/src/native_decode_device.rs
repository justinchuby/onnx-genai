//! Which device a native decode session should run on.
//!
//! Deliberately **not** inside `native_decode`, which is gated on the
//! `native-backend` feature. This enum is configuration — a plain choice made
//! by a caller, with no dependency on the backend that honours it — and
//! ungated code needs to name it: `EngineConfig` stores it, and the pipeline
//! passes it to functions whose bodies are gated even though their signatures
//! are not.
//!
//! Keeping it behind the feature made those signatures fail to compile whenever
//! `native-backend` was off, which is how a CUDA-only CI job started failing on
//! a type it never uses.

/// Device requested for a native decode session.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum NativeDecodeDevice {
    #[default]
    Cpu,
    Cuda {
        index: Option<u32>,
    },
    Plugin {
        library: std::path::PathBuf,
        registration_name: Option<String>,
        provider_name: String,
    },
}
