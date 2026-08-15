# Nabil — History Archive

## Archived 2026-07-29 (full pre-compaction snapshot)

# Nabil — History

## 2026-07-12: Joined
Hired to lead the ORT plugin-EP integration for a new **Apple Metal/MPS execution provider** for ONNX Runtime (repo `../onnxruntime-mps`). Motivation: onnx-genai is ORT-kernel-bound on Apple Silicon (ORT's generic int4 CPU/WebGPU kernels lag llama.cpp's hand-tuned Metal); a custom MPS EP with hand-tuned kernels can beat everyone on Mac. The EP must support all ops onnx-genai/Mobius use: MatMulNBits (int4), GroupQueryAttention, GatherBlockQuantized, RoPE, RMSNorm, softmax, elementwise. Tested end-to-end by the onnx-genai runtime (`ONNX_GENAI_EP` selects it). Reference kernels: ExecuTorch + PyTorch MPS backends.

### 2026-07-16T00:00:00Z — Performance-and-design wave
Authored the ORT-schema-based model-package design document.

### 2026-07-16T00:00:03Z — Projection-fusion design recorded
Authored `docs/quantization/PROJECTION_FUSION.md` for conservative load-time gate/up MatMulNBits fusion. Fact Checker confirmed QKV is already packed, gate/up is the available `4864|4864→9728` target, and qualified the roughly 125 MiB payload as a lower-bound memory cost. The design is awaiting user approval and is not implemented.

### 2026-07-16T00:00:00Z — Native CUDA decode design
Authored `docs/execution/NATIVE_CUDA_DECODE.md` (`b416b7f`) and applied Fact Checker's stream/graph-ownership corrections (`33beb8d`). The fact-checked five-milestone `Arc<dyn ExecutionProvider>` design awaits user greenlight; implementation has not started.

## 2026-07-16T17:00:38+0000 — Weight offload design
- Authored `docs/memory/WEIGHT_OFFLOAD.md` (`f0d0890`): immutable mmap backing feeds bounded host and VRAM caches through weight-specific expert/page leases.
- The design awaits user greenlight; no implementation has started.

## 2026-07-16T19-27-57+0000 — Scribe session update

- Authored `docs/models/DEEPSEEK_CSA_MTP_RUNTIME.md` (`bca068c`), a native CSA/index-op and persistent iterative-MTP sidecar-state design. It awaits user greenlight.

## 2026-07-14T00:00:00Z — QMoE final approval

- Rejected the initial and first hardening revisions, then approved the final QMoE kernel once overflow checks, allocation addressability, and odd affine-int4 block handling were correct.

## 2026-07-17T02:24:32Z — QMoE int1/int2 review

- 🟢 Cleared `cdb4ee5`: factory gating, packing, zero-point tails, sizing, and existing hardening are correct; full crate suite passed (450 passed, 1 ignored).

## 2026-07-18T04-55-00Z — Scribe session update

- On lockout reassignment, fixed CUDA standard Attention claim validation (`8eb23f1`) so `Undefined` optional mask/past/nonpad slots mean absent while supplied tensors retain strict type and compatibility checks.

## 2026-07-27T18:20:00-07:00 — MLX EP logging framework

- Replaced all 12 `eprintln!`/`eprint!` sites in `onnxruntime-mlx/rust/` with the `log` crate facade + minimal in-crate stderr logger.
- Chose `log` over `tracing` because the plugin is a cdylib with its own statics; the subscriber model has no benefit and adds weight.
- Default: **Warn** only (panics + capture failures). Info via `VERBOSE=1`, Debug via `TRACE=<path>`.
- Verified: build clean (`-D warnings`), 1106 tests pass, stderr is empty by default.
- PR: https://github.com/justinchuby/onnxruntime-mlx/pull/9 (not merged)
- Decision: `.squad/decisions/inbox/nabil-mlx-logging.md`

## 2026-07-28T04-08-08+0000 — Wave 2 regression/roadmap update
- MLX logging decision note was merged into decisions for future backend logging work.

## ARCHIVED 2026-08-12T06:00:00Z (Scribe #762 memory-safety wave compaction)

### 2026-08-10 — Outbound plugin-EP ABI recon and gap analysis
Completed outbound plugin-EP ABI recon and gap analysis. Verified ORT 1.27.0 / API v27 bindings. Enumerated all ORT plugin-EP structs and function-pointer fields. Produced lifecycle mapping table (2 DIRECT, 9 ADAPTABLE, 4 MISSING). CPU EP export is mechanical; CUDA EP has hard blockers around context/stream/allocator ownership. Written to `docs/ep-plugin/EP_PLUGIN_EXPORT_ABI_GAPS.md`.

### 2026-08-10 — Implemented outbound ORT plugin-EP export (v1)
Created `crates/onnx-runtime-ep-plugin` and `crates/onnx-runtime-ep-cpu-plugin`. CPU EP exports `CreateEpFactories` / `ReleaseEpFactory` as real C symbols. OrtEpFactory vtable fully wired. L2 dlopen integration test passing.

### 2026-08-10 — Hardened crate for security audit findings (v2)
Fixed C1/H1/H2/H3 ship-blocking findings: catch_unwind on all 9 extern C callbacks, `AtomicPtr` for HOST_ORT_API, null check in ep_compile_inner, removed unsound `unsafe impl Send + Sync`. Added `invalid_arg_status` helper. 23 unit tests passing.

### 2026-08-10 — Fixed EP device enumeration
Root cause: `GetSupportedDevices` returned 0 devices AND factory vtable had null function pointers. Fixed via `HardwareDevice_Type`, `CreateEpDevice`, `EpDevice_AddAllocatorInfo`, and 20 vtable slots. Full chain verified: Register → GetEpDevices → CreateSession → Run → correct output [6,8,10,12].

### 2026-08-10 — Wired shape inference + fail-closed capability claims (v4)
Replaced `ShapeInference::for_op` with `for_node` in `ep_compile_inner`. Added fail-closed capability filter: nodes where `for_node` returns `Declined` excluded. Extended `OutboundGraphReader` with attribute reading and int64 initializer extraction. Multi-node subgraphs get `SubgraphRouting`. 82 unit tests pass.

### 2026-08-10 — Device-capable adapter surfaces for GPU/NPU EPs
Created `device.rs` with `DeviceSupport`, `DeviceAllocator`, `DeviceSyncStream`. Created `onnx-runtime-ep-cuda-plugin` shim crate. ~26 mock-device unit tests.

### 2026-08-10 — DeviceSupport integration & clippy fix
Fixed clippy `forget_non_drop` in `device.rs:142`. Integrated `DeviceSupport` into `factory.rs`. Added 7 generalized enumeration unit tests. 127 lib + 15 conformance tests pass.

### 2026-08-10 — Test-module mem::forget no-ops resolved
Replaced `mem::forget(buffer)` with `let _ = buffer;` in two test doubles. No ownership defect.

### 2026-08-11 — Implemented native nxrt dynamic EP ABI (§524)
Created `crates/onnx-runtime-ep-nxrt-abi/` — full ABI: version negotiation, factory creation, vtable-based lifecycle, panic containment, `export_nxrt_ep_factories!` macro. 19 unit tests pass.

### 2026-08-11 — Export macros + testing module
Shipped `export_nxrt_ep_factories!`, `export_nxrt_ep_negotiate_custom!`, `export_nxrt_ep_create_custom!`. 30 unit tests pass.

### 2026-08-11 — CUDA EP use-after-free fix (B1/B3/S4 revision)
Replaced raw pointers from dropped `MutexGuard` with `Arc<Mutex<..>>` clones via `EpRef::Shared`. B3: `CopyTensors` uses `Value_GetMemoryDevice` + `MemoryDevice_GetDeviceType`. S4: `create_ep_factories_for_shared_ep` takes `ep_name` directly. 173 targeted tests pass.

### 2026-08-11 — CUDA B1/B3/S4 under lockout (commit d64a49d59)
Under lockout. B1 EpRef::Shared, B3 CopyDirection::classify, S4 no-panic path. B2 deferred citing `MemoryDevice_GetDeviceId` absent — **this was factually wrong (API at bindings.rs:6309)**. B2 assigned to Batty.
