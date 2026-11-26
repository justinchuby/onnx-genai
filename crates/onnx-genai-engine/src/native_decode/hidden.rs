//! Immutable hidden conditioning retained by sessions and their snapshots.
use onnx_runtime_session::OwnedHostArray;
use std::{ops::Deref, sync::Arc};

/// There is no mutable accessor: a subsequent forward publishes a new owner,
/// so snapshots can share the old allocation and its entire funding lifetime.
#[derive(Clone)]
pub(super) enum NativeHidden {
    Cpu(Arc<OwnedHostArray<f32>>),
    // CUDA retains its existing explicit Host allocation boundary. This does
    // not assert that CUDA staging has joined the CPU managed-payload domain.
    #[cfg(feature = "native-cuda")]
    Host(Arc<Vec<f32>>),
}
impl NativeHidden {
    pub(super) fn from_cpu(values: OwnedHostArray<f32>) -> Self {
        Self::Cpu(Arc::new(values))
    }
    #[cfg(feature = "native-cuda")]
    pub(super) fn from_host(values: Vec<f32>) -> Self {
        Self::Host(Arc::new(values))
    }
    pub(super) fn as_slice(&self) -> &[f32] {
        match self {
            Self::Cpu(values) => values.as_slice(),
            #[cfg(feature = "native-cuda")]
            Self::Host(values) => values.as_slice(),
        }
    }
    pub(super) fn len(&self) -> usize {
        self.as_slice().len()
    }
}
impl Deref for NativeHidden {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        self.as_slice()
    }
}
impl PartialEq for NativeHidden {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl std::fmt::Debug for NativeHidden {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}
