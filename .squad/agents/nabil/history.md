# Nabil — History (compacted 2026-08-12T06:00:00Z)

**Role:** Leads ORT plugin-EP integration for the Apple Metal/MPS EP and adjacent backend/runtime designs. The EP must cover onnx-genai/Mobius ops end-to-end, use ExecuTorch/PyTorch MPS references, and be tested through `ONNX_GENAI_EP`.

## Durable lessons
- ORT-schema model-package design was authored and remains the package-design baseline.
- Projection fusion: QKV is already packed; only gate/up `4864|4864→9728` pairs are candidates; ~125 MiB is a lower-bound payload cost.
- Native CUDA decode design needs a real non-null stream and serialized ownership of non-Send/Sync CUDA graphs.
- Weight offload design uses immutable mmap plus bounded host/VRAM caches through expert/page leases.
- QMoE fixes must preserve overflow checks, allocation addressability, odd affine-int4 blocks, int1/int2 gating/packing, zero-point tails, and sizing hardening.
- CUDA standard Attention validation: `Undefined` optional mask/past/nonpad slots mean absent; supplied tensors still need strict type/compatibility checks.
- MLX backend logging uses `log`, not `tracing`, because the cdylib has its own statics.
- `OrtMemoryInfo` lifetime: `EpDevice_AddAllocatorInfo` stores the raw pointer; do NOT call `ReleaseMemoryInfo` after success.
- `OrtGraph*`/`OrtNode*` must NOT be cached beyond their callback — copy all attributes into owned Rust during the callback.
- Deferrals must be backed by evidence. Verify generated bindings before deferring — `MemoryDevice_GetDeviceId` and `Session_GetEpGraphAssignmentInfo` were both claimed absent and both existed.

## Recent work (current wave, ~2026-08-11/12)

### 2026-08-11 — CUDA EP use-after-free + B1/B3/S4 under lockout (commit d64a49d59)
Replaced raw pointers from dropped `MutexGuard` with `Arc<Mutex<..>>` via `EpRef::Shared`. B3 CopyDirection via `Value_GetMemoryDevice`. S4 no-panic path for shared EP. B2 deferred citing `MemoryDevice_GetDeviceId` absent — **factually wrong** (API at `bindings.rs:6309`). B2 assigned to Batty.

### 2026-08-12 — B1+B2 absent-slot memory safety (PR #762, commit af45043fd)
**B1 (heap overflow):** Scratch dtype derived from `output_dtypes[slot]`; buffer sized `max(byte_size, 8)`; `TensorMut.absent` flag; fail-closed on Undefined. **B2 (routed path compaction):** `RoutedSlotKind` enum (Ort/Buffer/Absent) — every slot index aligned end-to-end. 277 passed / 0 failed. Miri clean. Clippy + fmt clean.

Full pre-compaction history in `history-archive.md`.
