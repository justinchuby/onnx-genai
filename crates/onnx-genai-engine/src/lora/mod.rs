//! LoRA adapter loading and representation.

pub mod format;

/// The native-LoRA runtime manager and PEFT → session-spec bridge (design §D,
/// P4). Only available with the native backend, since it depends on the session
/// crate's injection types.
#[cfg(feature = "native-backend")]
pub mod manager;
