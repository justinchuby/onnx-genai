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