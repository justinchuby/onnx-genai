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
- 2026-08-10: **Hardened crate for security audit findings (v2).** Fixed 4 ship-blocking findings: (C1) verified `catch_unwind` is present on all 9 extern C callbacks; (H1) confirmed `AtomicPtr` already in use for `HOST_ORT_API`; (H2) confirmed + improved `graphs.is_null()` null check in `ep_compile_inner`, returning `ORT_INVALID_ARGUMENT` via new `invalid_arg_status` helper; (H3) removed unsound `unsafe impl Send + Sync` for `OutboundGraphReader`. Fixed 2 `unused_unsafe` warnings in `graph_reader.rs` (lines 40 and 501) by removing `unsafe {}` wrappers around non-unsafe `host_api()`. Fixed M1: `check()` now extracts the real ORT error message via `GetErrorMessage` before releasing the status. Added `invalid_arg_status` and `status_with_code` helpers to `status.rs`. Added 8 new unit tests (23 total, all passing). `cargo clippy -p onnx-runtime-ep-plugin -- -D warnings` clean. Decision record: `.squad/decisions/inbox/nabil-ep-plugin-hardening.md`.
- 2026-08-10: **Fixed EP device enumeration — ORT 1.27 now registers and runs our EP.** Root cause: `GetSupportedDevices` returned 0 devices AND factory vtable had null function pointers. ORT 1.27 segfaults on both. Fix: (1) implemented proper CPU device filtering via `HardwareDevice_Type`, `CreateEpDevice`, and `EpDevice_AddAllocatorInfo`; (2) populated ALL 20 vtable slots with real or no-op stubs (ORT calls them without null-checking); (3) `CreateAllocator` returns ORT's default allocator (ORT dereferences null output). Full chain verified: Register → GetEpDevices → CreateSession → Run → correct output [6,8,10,12]. Tests `ort_register_ep_library` and `ort_loads_our_ep_and_runs_model` pass. Decision record: `.squad/decisions/inbox/nabil-ep-device-enumeration.md`.

Full pre-compaction history in `history-archive.md`.
- 2026-08-10: **Wired shape inference into live path + fail-closed capability claims (v4).** Two integration gaps closed: (1) Replaced `ShapeInference::for_op` with `ShapeInference::for_node` in `ep_compile_inner` — Deckard's 22 attribute-aware rules now execute in the live path. (2) Added fail-closed capability filter in `ep_get_capability_inner`: nodes whose `for_node` returns `Declined` are excluded from claims, preventing over-claiming ops we cannot execute (e.g. NonZero with data-dependent output shape). Extended `OutboundGraphReader` with full attribute reading (via `Node_GetNumAttributes`/`Node_GetAttributes`/`ReadOpAttr`) and int64 initializer extraction (via `Graph_GetInitializers`/`ValueInfo_GetInitializerValue`/`GetTensorData`) — all data copied into owned Rust during the Compile frame, no ORT pointers cached. Opset-13 Unsqueeze/Squeeze axes resolved from constant initializer inputs. Multi-node fused subgraphs now get `SubgraphRouting` via `build_subgraph_routing()` so intermediates thread correctly. Tests: 82 unit tests pass, `ort_register_ep_library`/`ort_loads_our_ep_and_runs_model`/`ort_unsupported_op_declines_not_crashes` all pass. Decision: `.squad/decisions/inbox/nabil-ep-capability-integration.md`.
- 2026-08-10: **Device-capable adapter surfaces for GPU/NPU EPs.** Created `device.rs` module with generalized device surface: `DeviceSupport` config, `DeviceAllocator` (#[repr(C)] OrtAllocator vtable projection), `DeviceSyncStream` (#[repr(C)] OrtSyncStreamImpl vtable projection), fail-closed validators (`validate_device_support`, `validate_allocator_request`, `validate_stream_request`), and device-type mapping. Created `onnx-runtime-ep-cuda-plugin` shim crate (mirrors CPU plugin; feature-gated, compiles without CUDA toolkit). ~26 mock-device unit tests exercise allocator/stream/validator paths via `MockGpuEp`/`MockCpuEp` test doubles. `cargo check --workspace` passes. Blocked on `ep.rs:114` compile error (Deckard). Decision: `.squad/decisions/inbox/nabil-device-adapter-surfaces.md`.

## 2026-08-10 — DeviceSupport integration & clippy fix

- Fixed clippy `forget_non_drop` error in `device.rs:142`: `DeviceBuffer` has no `Drop`, so `mem::forget` was a no-op. Removed it; documented why the allocation is safe (owned by EP allocator, freed via `device_free`).
- Integrated `DeviceSupport` into `factory.rs`: generalized device enumeration, allocator creation, stream creation, and stream-awareness reporting. CPU path unchanged; GPU/NPU paths now functional via `create_ep_factories_with_device_support`.
- Added 7 generalized enumeration unit tests. Total: 127 lib tests pass, 15 conformance tests pass (including 25-cycle stress test).
- `cargo check --workspace` clean without CUDA.

## 2026-08-10 — Test-module mem::forget no-ops resolved

Clippy `--all-targets` flagged two more `forget_non_drop` errors (device.rs:465 and device.rs:569) in `#[cfg(test)]` mocks (`MockGpuEp::deallocate` and `MockCpuEp::deallocate`).

**Site analysis:**
- Both are `deallocate` implementations in test doubles. The pattern was: extract `ptr`/`size` from the `DeviceBuffer`, call `mem::forget(buffer)`, then manually `dealloc` the raw pointer. Intent: prevent double-free. Verdict: **dead code / no-op** — `DeviceBuffer` has no `Drop` impl, so there is no destructor to suppress. No ownership mistake; the pointer extraction + manual `dealloc` is already correct. The `forget` added no protection.
- Resolution: replaced `std::mem::forget(buffer)` with `let _ = buffer;` (idiomatic discard binding). Clippy accepts `let _ =` on non-`Drop` types because it is not a spurious lifecycle claim — it just moves the value into `_`. Neither `drop(buffer)` nor `mem::forget(buffer)` is accepted by clippy on a non-`Drop` type.
- No ownership defect, no decision record needed.

**Validation:**
- `cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings` → only Pris's 2 errors in `tests/trait_cabi_parity.rs`; zero errors from `device.rs`/`factory.rs`/`lib.rs`.
- `cargo test -p onnx-runtime-ep-plugin --lib` → **127 passed; 0 failed**.
- `cargo test -p onnx-runtime-ep-cpu-plugin` → **23 passed; 0 failed** (6 ABI + 17 conformance).
- `cargo check --workspace` → **Finished** without CUDA.

## 2026-08-11 — Implemented native nxrt dynamic EP ABI (§524)

Designed and implemented `crates/onnx-runtime-ep-nxrt-abi/` — the native half of the §524 extension contract. Full ABI surface: version negotiation (`NxrtNegotiate`), factory creation (`NxrtCreateEpFactories`), vtable-based lifecycle for factory/EP/kernel/allocator, explicit ownership rules, panic containment via `catch_status_panic`/`catch_void_panic`, and the `export_nxrt_ep_factories!` macro. All 19 unit tests pass. Clippy clean. Workspace check green. Added both `-nxrt-abi` and `-nxrt-host` to workspace members for Isidore's concurrent work.

## 2026-08-11 — Export macros + testing module for negative fixtures

Shipped three macros closing the "duplicate ABI" hole: `export_nxrt_ep_factories!` (standard plugins), `export_nxrt_ep_negotiate_custom!` (negative-test negotiate overrides), `export_nxrt_ep_create_custom!` (negative-test factory overrides). Added `testing` module exposing `NxrtNegotiateOverride` and `NxrtCreateFactoriesOverride` with `wrong_major`, `unknown_caps`, `panicking`, `zero`, `error` variants. Re-exported all capability constants and `validate_negotiation` at crate root. Macro hygiene: fully-qualified `::std::`/`$crate::` paths, `$constructor` evaluated outside unsafe block (clippy::macro_metavars_in_unsafe clean). Tests: **30 unit tests pass** (up from 19). `cargo check --workspace` green. Decision note updated.

## 2026-08-11 — CUDA EP use-after-free fix (B1/B3/S4 revision)

Took ownership after Gaff rejected Sapper's CUDA EP commits on #762 (reviewer lockout). Fixed three blocking defects:

- **B1 (use-after-free):** Replaced raw pointers from dropped `MutexGuard` with `Arc<Mutex<..>>` clones stored in each component via `EpRef::Shared`. Each callback locks briefly — no dangling pointers.
- **B3 (CopyTensors direction):** `transfer_full_copy_tensors` now calls `Value_GetMemoryDevice` + `MemoryDevice_GetDeviceType` to classify H2D/D2H/D2D, dispatching to `copy_from_host`/`copy_to_host`/`copy` correctly.
- **S4 (panic bomb):** New `create_ep_factories_for_shared_ep` takes `ep_name` directly — no constructor call. Fail-closed by design (actionable OrtStatus), not by accidental panic.

Also fixed: S1 (unknown ptr no-op), S2 (no `.unwrap()` across FFI), S3 (CreateEp uses shared_ep), N2 (vendor_id from config). Deferred B2 (pointer equality for same-device — fail-closed, not UB).

Added 3 regression tests: B1 (allocator outlives original Arc), S4 (no panic escape), B3 (direction matrix). All 173 targeted tests pass. Clippy clean. Plugin remains **fail-closed and unvalidated on hardware** — by design, not by circumstance.
