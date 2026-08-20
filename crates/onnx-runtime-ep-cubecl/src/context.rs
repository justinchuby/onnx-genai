//! The state every kernel needs: a CubeCL client and the address table that
//! maps EP device pointers back onto CubeCL buffers.

use cubecl::prelude::*;
use onnx_runtime_ir::DeviceId;

use crate::backend::CubeclBackend;
use crate::memory::HandleTable;

/// Shared, immutable-after-construction provider state.
///
/// Kernels hold an `Arc` of this rather than a back-reference to the provider,
/// so a kernel stays valid for as long as it is held even if the provider is
/// moved, and so kernel code cannot reach mutable provider state.
pub struct CubeclContext<R: Runtime> {
    pub client: ComputeClient<R>,
    pub table: HandleTable,
    pub device: DeviceId,
    pub backend: CubeclBackend,
}

impl<R: Runtime> std::fmt::Debug for CubeclContext<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CubeclContext")
            .field("device", &self.device)
            .field("backend", &self.backend)
            .field("live_allocations", &self.table.live_allocations())
            .finish_non_exhaustive()
    }
}
