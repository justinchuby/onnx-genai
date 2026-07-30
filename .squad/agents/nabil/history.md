# Nabil — History (compacted 2026-07-29)

**Role:** Leads ORT plugin-EP integration for the Apple Metal/MPS EP and adjacent backend/runtime designs. The EP must cover onnx-genai/Mobius ops end-to-end, use ExecuTorch/PyTorch MPS references, and be tested through `ONNX_GENAI_EP`.

## Durable lessons
- ORT-schema model-package design was authored and remains the package-design baseline.
- Projection fusion: QKV is already packed; only gate/up `4864|4864→9728` pairs are candidates; ~125 MiB is a lower-bound payload cost. Awaiting approval, not implemented.
- Native CUDA decode design needs a real non-null stream and serialized ownership of non-Send/Sync CUDA graphs; awaiting greenlight, not implemented.
- Weight offload design uses immutable mmap plus bounded host/VRAM caches through expert/page leases; no implementation started.
- CSA/MTP runtime design covers native CSA/index-op plus persistent iterative-MTP sidecar state; awaiting greenlight.
- QMoE fixes must preserve overflow checks, allocation addressability, odd affine-int4 blocks, int1/int2 gating/packing, zero-point tails, and sizing hardening.
- CUDA standard Attention validation: `Undefined` optional mask/past/nonpad slots mean absent; supplied tensors still need strict type/compatibility checks.
- MLX backend logging uses `log`, not `tracing`, because the cdylib has its own statics; default stderr is Warn-only, with Info via `VERBOSE=1` and Debug via `TRACE=<path>`.

## Recent work (current wave, ~2026-07-28/29)
- 2026-07-27: Replaced 12 MLX `eprintln!`/`eprint!` sites with the `log` facade plus minimal stderr logger; verified `-D warnings`, 1106 tests, and empty default stderr. PR https://github.com/justinchuby/onnxruntime-mlx/pull/9 remains unmerged.
- 2026-07-28: MLX logging decision note was merged into decisions for future backend logging work.

Full pre-compaction history in `history-archive.md`.
