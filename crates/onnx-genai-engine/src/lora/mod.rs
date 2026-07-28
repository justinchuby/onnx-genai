//! LoRA adapter loading and representation.

#[allow(
    dead_code,
    mismatched_lifetime_syntaxes,
    unused_imports,
    unsafe_op_in_unsafe_fn,
    clippy::all,
    clippy::pedantic
)]
mod adapter_schema_generated;
pub mod format;

/// The native-LoRA runtime manager and PEFT → session-spec bridge (design §D,
/// P4). Only available with the native backend, since it depends on the session
/// crate's injection types.
#[cfg(feature = "native-backend")]
pub mod manager;
