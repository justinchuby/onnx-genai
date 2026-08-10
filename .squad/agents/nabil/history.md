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

## Recent work (current wave, ~2026-08-10)
- 2026-08-10: Completed outbound plugin-EP ABI recon and gap analysis. Verified ORT 1.27.0 / API v27 bindings. Enumerated all ORT plugin-EP structs (`OrtEpFactory`, `OrtEp`, `OrtNodeComputeInfo`, `OrtEpApi`, supporting opaque types) and their function-pointer fields from source. Produced lifecycle mapping table (2 DIRECT, 9 ADAPTABLE, 4 MISSING). Key finding: inbound `UnionFind`/`query_capabilities` machinery is fully reusable; `HostGraph`/`HostKernelContext` must be inverted. CPU EP export is mechanical; CUDA EP has hard blockers around context/stream/allocator ownership. Written to `docs/EP_PLUGIN_EXPORT_ABI_GAPS.md`.
- 2026-08-10: **Implemented outbound ORT plugin-EP export (v1).** Created `crates/onnx-runtime-ep-plugin` (adapter lib) and `crates/onnx-runtime-ep-cpu-plugin` (cdylib shim). CPU EP now exports `CreateEpFactories` / `ReleaseEpFactory` as real C symbols loadable by upstream ORT. `OrtEpFactory` vtable fully wired (GetName, GetSupportedDevices, CreateEp, ReleaseEp). `OrtEp` vtable wired (GetCapability via shared `query_capabilities`, Compile via `get_kernel`, ReleaseNodeComputeInfos). Compute callback returns fail-closed NOT_IMPLEMENTED (output shape bridging deferred). Removed dead `OrtPluginExport` placeholder. L2 dlopen integration test passing. Assumed export symbol: `CreateEpFactories`.
- 2026-08-10: **Closed the execute path.** Implemented `OutboundKernelContext` (kernel_ctx.rs) bridging ORT's `OrtKernelContext` ↔ `TensorView`/`TensorMut`. Wired `compute_execute` to read inputs via OrtApi, infer output shapes at runtime (broadcast or same-as-input), allocate outputs via `KernelContext_GetOutput`, and invoke `Kernel::execute`. Added `CompiledKernelEntry` with per-kernel metadata (num_inputs, num_outputs, output_dtype, ShapeInference). Added fail-closed version check: `GetApi(ORT_API_VERSION) == null` returns explicit OrtStatus via v1-API fallback. L2 test extended: drives Add kernel through Compute with value assertions (simple [4] and broadcast [2,3]+[3] cases). All 3 L2 tests pass. Clippy clean.

Full pre-compaction history in `history-archive.md`.
