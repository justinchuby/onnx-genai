# Leon — History (compacted 2026-07-29)

**Role:** Engine/KV/runtime-buffer implementer. Runtime owns KV; model geometry from `inference_metadata.yaml`. Preserve device-buffer ownership, past/present aliasing, exact real-model comparison, reviewer lockouts.

**Historical summary through 2026-07-28:** Generalized shared KV, attention-sink SWA, connectors, prefix payload materialization (equal prefix keys prove content equality). Delivered heterogeneous Gemma4 E2B speculative execution (proposer inputs corrected). Hardened loaders/fusion (unsupported dtypes fail-closed, LayerNorm operand-order guarded, opset validation recursive, `nxrt_*` C ABI replaces `ort2_*`). Implemented weight-offload foundations, route-first QMoE, CUDA SparseKvGather D==0 validation, CPU CSA claim validation. Contributed to CUDA graph/capture correctness (SequenceAt/Scan parity, Phi decode lock, default-domain Attention and RoPE capture regressions) and PR #291 rewind policy split (public rewind rejects before mutation, internal speculative rewind allowed). Unified native CUDA/ORT KV capacity policy with transactional growth; real DeepSeek validation verified 4→8→16 growth/recapture.

Older detailed work (2026-07-14 through 2026-07-28) archived in `history-archive.md`.

## Recent work (2026-07-29)

### 2026-07-29T03:45:00+0000 — PR #382 CPU shared-buffer regression lock

- Under Benny's reviewer lockout, added `cpu_shared_buffer_continuous_batch_uses_declared_kv_pairs`, using `tiny-llm-sharedbuffer` and explicit float32 KV metadata.
- Engine-level CPU test runs continuous batching and compares sequential generation; fails at session construction if declared `model.io.kv_inputs` / `kv_outputs` stop reaching `BatchedSharedBufferDecodeSession`.
- Revert verification proved the test catches latent #380 regression previously hidden because equivalent CUDA E2E auto-skips without CUDA. Repair and test merged in `85b9ba15`.

### 2026-07-28T18:00:00-0700 — PR #385 re-scoped onto #392 (server + Python sampling wiring)

- #392 merged engine + CLI half of model-sampling-defaults work to `main` (`resolve_sampling_defaults`, `Option`-typed `SamplingOverrides`, CLI wiring). Strict precedence preserved: explicit override > model-declared > greedy fallback.
- Reset branch onto `origin/main`, re-applied only the delta #392 left missing: server + Python wiring, misnamed-test fix, resolver-level temperature-0 → greedy guard. Final diff: 7 files, +414/-49, single commit `b78d8bec`.
- Server/Python callers now decode stochastically against `do_sample: true` models, matching CLI. No greedy-assuming test broke.
- Gates green: engine lib 274, server sampling 116 pass (1 pre-existing fixture failure)

## 2026-07-29T12:30:00Z — tiny-reasoning-fixture rounds 2–3 (PR #411)

### Round 2 (replaced Batty after Gaff REJECT)
Authored statistical token-stream replacement. Luv ran it alone: 15/15 failures with
fix intact; one green in parallel suite was a fluke. Luv issued REJECT.

### Round 3 — resolved-policy surface (approved `f8ed4fb4`)
Surfaced sampling policy generation actually resolved into `--stats`/`--profile`.
`SamplingPolicy` captured from `turn_options` after `resolve_session_sampling`; same
struct moved into `TurnInput.options` (`:1352`) — no separate display-side resolution.
Two resolution sites unified: one helper called by both `/session` and every turn,
reading live backend on demand. No cache; no staleness across `/reload`/`/ep`/`/backend`.
`interactive.rs:1342-1347` + `generate.rs:122-127`.

Luv approved at `f8ed4fb4`. Mutation: both new tests FAIL 3/3; suite 42+2/44.
Mutated stats line `greedy=true temperature=1 top_k=0` — matches #385/#392 class.

### Delta (`88fa86b5`)
Moved capture inside `run_generation_turn` (`output.rs:206-211`). `turn` bound
immutably; moved into `backend.generate(turn, …)` at line 278 — no window between
capture and use. Divergence structurally impossible. Luv delta-approved after
mutation 3/3 red, isolation 10/10 green, full suite 44/44.

Also contributed to:
- Fixture `manifest.json`/generator string consistency fix (Batty's bug).
- Empty-answer invariant correction (manifest now accurately describes "drop
  whitespace-only" rather than asserting strict non-emptiness).

Durable rules:
- "Instrument the boundary you care about."
- "Two independent resolution sites for one policy is the defect, not an inconvenience."
- "Close a gap by construction rather than by comment where you can."
- "A checked-in fixture must be reproducible from its generator."
(`.squad/decisions.md`, reconstructed rules section, 2026-07-29)

Inbox drop `leon-reasoning-fixture-round3.md` was lost when the worktree was deleted

## 2026-08-10 — EP Plugin Compute Hardening (reviewer rejection fix)

**Context:** Holden's security re-audit flagged two findings (N1, N2) in `compute.rs`/`kernel_ctx.rs`. Deckard locked out; reassigned to Leon.

**N1 (CRITICAL — `compute_execute` panic guard):** Already present in current code — `catch_unwind` wraps the entire `compute_execute` body (confirmed at `compute.rs:~551`). Added test verifying the pattern.

**N2 (HIGH — negative dims wrap to usize::MAX):** Fixed in `kernel_ctx.rs`. Added `validate_dims()` helper that rejects any negative dimension with an actionable error message naming the dim index and value. Zero dims are accepted (legal ONNX).

**Additional hardening:**
- Element-count overflow: all `shape.iter().product()` replaced with `checked_mul` fold in `kernel_ctx.rs:validate_dims`, `compute.rs` intermediate buffer allocation, and `read_i64_tensor`.
- Byte-length overflow: `element_count * byte_size` uses `checked_mul`.
- Zero-dim null-ptr: zero-element tensors are allowed to have null data pointers; only non-zero-element tensors fail on null.
- 7 new unit tests: negative dim rejected, large negative rejected, element-count overflow, byte-length overflow, zero-dim accepted, scalar tensor, normal shape.
- 2 new compute tests: panic-guard pattern, contiguous_strides edge cases.

**Build status:** `cargo build -p onnx-runtime-ep-plugin` fails due to `graph_reader.rs` (Isidore's concurrent edits — missing fields/methods). All errors are confined to that file; `compute.rs` and `kernel_ctx.rs` have zero compile errors and pass clippy when graph_reader is stubbed. Noted for coordinator.

**No public API signatures changed.** `validate_dims` is `pub(crate)` only.
before Scribe ran; content reconstructed into `.squad/decisions.md`.
---

## 2026-08-10 — Clippy dead_code cleanup: validate_dims wired into read_inputs

**Branch:** squad/ep-plugin-export
**Triggered by:** Deckard validation gate failure; Reviewer Rejection Protocol prevented him from editing.

**Finding:** `validate_dims` in `kernel_ctx.rs:23` was reported as dead code by
`cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings`.

**Root cause — real gap, not cosmetic:** `validate_dims` was defined but never
called from `read_inputs`. The production path was silently casting ORT dims via
`.map(|&d| d as usize)` — a bare cast that passes negative dims as huge positive
values, bypassing the negative-dim rejection and overflow checks entirely.
This is exactly the "validation path never connected" scenario flagged in the mission.

**Fix:** Replaced the bare cast in `read_inputs` with a call to `validate_dims`,
so every set of ORT-supplied dims crossing the FFI boundary now goes through the
validated path. No `#[allow(dead_code)]` was used.

**Remaining clippy errors (not my files):**
- `lib.rs:184` — `unused-mut` (`let mut out_num`)
- `lib.rs:189` — `clippy::diverging-sub-expression` (panic! in test guard)
- `ep.rs:499` — `clippy::manual-dangling-ptr` (`1usize as *mut OrtEp`)
These are in Isidore's (`lib.rs`) and Deckard's (`ep.rs`) files; not touched.

**Test result:** `cargo test -p onnx-runtime-ep-plugin --lib` — 82 passed, 0 failed.

---

## 2026-08-10 — EP plugin parity wave: NEW-1 fix + f16/bf16 marshaling

**Branch:** squad/ep-plugin-parity-cuda

### TASK 1 — NEW-1: compute_release_state catch_unwind (compute.rs)

`compute_release_state` was the only `extern "C"` callback in the two owned
files lacking a `catch_unwind` guard. Wrapped the body in
`catch_unwind(AssertUnwindSafe(…))` with `let _ = result` to swallow any panic
(void return — no status channel). The other two callbacks (`compute_create_state`,
`compute_execute`) were already guarded; none were missed.

Added test `release_state_swallows_panic_safely` verifying the guard pattern.

### TASK 2 — f16/bf16 marshaling (kernel_ctx.rs)

Verified against `bindings.rs`:
- `ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10` → `DataType::Float16 = 10` ✅
- `ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 = 16` → `DataType::BFloat16 = 16` ✅
- Both have `byte_size() = 2` — existing `checked_mul` overflow guards cover them.

Exposed `CPU_EP_SUPPORTED_DTYPES: &[DataType]` as a public constant so Deckard
can import it (do not copy) for `GetKernelRegistry` type constraints.

Added 7 new tests covering f16/bf16 round-trip, byte-length, overflow guard,
unsupported-dtype fail-closed, and the supported-dtypes constant.

**Test results:**
- `cargo test -p onnx-runtime-ep-plugin --lib` → 90 passed (was 82)
- `cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings` → clean
- `cargo test -p onnx-runtime-ep-cpu-plugin` → 15 passed

---

## 2026-08-10 — M2-1/M2-2: stream EP memory leak + doc comment (device.rs)

**Branch:** squad/ep-plugin-parity-cuda  
**Triggered by:** Holden milestone-2 audit; Nabil locked out (reviewer rejection).

### M2-1 (MEDIUM): EP instance leaked in `stream_release`

**Root cause:** `factory_create_sync_stream` (factory.rs:668) creates a fresh EP via
`Box::into_raw(ep)` and stores the pointer in `DeviceSyncStream.ep`. The `stream_release`
callback only dropped the `DeviceSyncStream` Box but never reclaimed the EP pointer.

**Fix:** Added `Box::from_raw(stream.ep as *mut dyn ExecutionProvider)` in
`stream_release` after dropping the stream Box. The null guard prevents UB if
somehow called with a null EP.

**Double-free ruling:** ORT header (lines 207–216) confirms `Release` is called
exactly once per created stream. No other code reclaims the stream EP. The allocator
path has its own independent EP instance created in `factory_create_allocator`.

**Regression test:** `stream_release_reclaims_owned_ep_no_leak` — uses an EP whose
Drop increments a static `AtomicUsize`; asserts count goes from 0→1 after release.

**Test fix:** Updated 3 existing stream tests to use `Box::into_raw(Box::new(MockGpuEp))`
instead of stack references, matching the real factory path and preventing UB under
the new release logic.

### M2-2 (LOW): misleading doc comment on `DeviceAllocator::memory_info`

Comment claimed "Owned; freed on drop" but there is no `Drop` impl and the pointer
is ORT-borrowed (`EpDevice_AddAllocatorInfo` stores raw pointer; ORT releases via
`ReleaseEpDevice`). Fixed to: "Borrowed from ORT; NOT freed by this allocator."

### Audit — no other unpaired `into_raw` in device.rs

All `Box::into_raw` in device.rs are in tests and are paired with `Box::from_raw`
or `stream_release`. No factory.rs change needed for this fix.

### Validation

- `cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings` → clean
- `cargo test -p onnx-runtime-ep-plugin --lib` → 133 passed (132 + 1 new)
- `cargo test -p onnx-runtime-ep-cpu-plugin --all-targets` → 17 passed
- `cargo check --workspace` → success

## 2026-08-11 — Device data-transfer contract (`transfer.rs`)

**Branch:** `squad/ep-plugin-parity-cuda` (PR #762)

**Created:** `crates/onnx-runtime-ep-plugin/src/transfer.rs` — ORT `OrtDataTransferImpl` adapter.

**What:**
- `DeviceDataTransfer` (basic) and `DeviceDataTransferFull` (with OrtApi) adapters
- Copy-direction matrix: H→D, D→H, D→D(same) supported; cross-device + H→H rejected
- Stream-ordered copy via `copy_async` + `Fence` + `wait_fence`
- Ownership: Box::into_raw/from_raw lifecycle, EP borrowed not owned
- Mock device EP with non-host-dereferenceable address space for testing
- 21 new tests covering direction matrix, fail-closed CanCopy, ownership/leak detection, device-pointer guards

**Validation:**
- `cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings` → clean
- `cargo test -p onnx-runtime-ep-plugin` → 154 lib + 9 parity passed
- `cargo test -p onnx-runtime-ep-cpu-plugin` → 23 passed
- `cargo check --workspace` → success

**Not proven:** Nothing here proves CUDA works. Hardware-gated.
