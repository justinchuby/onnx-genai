# Batty — History (compacted 2026-07-29)

**Role:** Engine/EP implementer for the Rust ONNX runtime. Owns generation policy, logical KV, scheduler/default semantics, CLI maintainer harness wiring, and CPU/native EP correctness while preserving ORT ownership of physical forward execution/KV.

## Durable lessons
- Canonical ownership: ORT owns forward execution and physical KV; engine owns generation policy and logical KV.
- CPU kernels rely on session-side `strided::view_in_bounds` before dispatch.
- Metal-prefill hybrid measured slower than CPU and should not be productionized; connector KV import should eventually degrade to `Ok(None)` on import-runner failure.
- Optimizer fusions live under `com.microsoft` and must fail closed with strict decline-to-fuse guards.
- EPContext writer v1 was rejected for non-injective sidecar naming; Batty is locked out of that artifact. Gaff v3 merged as `0fa025e`.
- Batty remains locked out of H-D1 storage sizing, earlier fusion follow-ups, EPContext writer, `test/tiny-reasoning-fixture`, and any artifact explicitly reassigned by reviewers.
- `validate_model()` is the shared load-time validation path; empty graphs remain valid, dispatch-only checks stay out of load validation, and `Graph::validate` invariants are not duplicated.
- Native backend Auto detection is exact for `com.github.onnxruntime.genai::BlockQuantizedMatMul` opset-v1; unsupported speculation, pipelines, or non-CPU selection must error explicitly.
- Preserve the public `MtpConfig` struct contract through internal resolved config layers.
- CUDA EP work must remain capture-safe and correct across supported SM architectures, not only sm_90; Qwen decode locks in `common/decode_lock.rs` are co-owned with Leon/Pris.
- CLI `generate`/`run` absent `--max-new-tokens` fills remaining effective context; engine/server default remains 128; a 512-token warning fallback applies only when no limit is discoverable.
- Sampling flags disable greedy when temperature/top-p/top-k imply stochastic decoding unless `--temperature 0` or explicit `--greedy`; `--no-greedy` exists.
- Justin confirmed the CLI is a development/maintainer harness, not a consumer product; remote-client mode is out of scope and `docs/research/cli/00-backlog.md` is source of truth.
- ControlNet fixes bind real Mobius `controlnet_cond`; do not invent `conditioning_scale`; multi-ControlNet fails loudly until supported.
- Rewinds validate before session/token/KV mutation; unsupported sliding-window evicted positions and ORT-owned KV without paged materialization reject cleanly.
- Scheduler keeps DESIGN §26.4/§26.11 conservative reservation: reserve prompt + admitted max_tokens, cap only when prompt+one token fits, and make caps observable in result, stderr, stats/profile.
- Native `generate` bypasses scheduler admission; ORT direct/session generation goes through `drive_next_fcfs`.
- Tiny reasoning fixture trap: Batty's statistical token-stream replacement was rejected (15/15 failures with fix intact; one suite green was a fluke, distinct-output evidence came from stderr timestamps) and Batty is locked out.
- Empty assistant turns poison context like unclosed reasoning; closed paths must drop whitespace-only answers. Checked-in fixtures must be reproducible from generators.

## Recent work (current wave, ~2026-07-28/29)

### 2026-07-27 — ControlNet, rewind, backend, scheduler
- Fixed PR #283 / #50 after Bishop REQUEST-CHANGES: removed invented `conditioning_scale`, bound real mobius `controlnet_cond`, made multi-ControlNet fail loudly, and added contract-pinning tests. Bishop approved; PR merged as `687612f5`.
- Took over PR #291 after Leon's rejection under reviewer lockout and made failed rewinds validate before session mutation; unsupported sliding-window and ORT-owned-KV paths now leave logical tokens, `kv_token_count`, decode state, and paged KV unchanged.
- Added shared `--backend auto|ort|native` for `generate`/`run`; `run --backend` seeds `SessionSettings`, default stays `auto`, and explicit `native` reaches `EngineConfig` or fails clearly without native feature support. Gates passed: build, lib tests, fmt check, clippy.
- Scheduler regression work preserved conservative reservation, attached observable `budget_cap` metadata across swap-out/swap-in, reserved capped byte amounts atomically, and cleaned up both admitted/original requests on mismatch. Added model-free scheduler tests plus an engine test locking ORT-vs-native scheduler behavior.

### 2026-07-29T12:30:00Z — tiny-reasoning-fixture round 2 + empty-answer fix (PRs #410, #411)

#### Round 2 (PR #410 → locked out)
Authored round-2 replacement test: statistical token-stream assertion. Luv ran it alone:
15/15 failures with the fix intact; one green in full parallel suite was a fluke; the
supporting "8/8 distinct outputs" was a stderr-timestamp artifact — test compared stdout
only. Luv issued REJECT; Batty locked out of `test/tiny-reasoning-fixture`.

#### Empty-answer fix (shipped in PR #411)
Diagnosed and fixed: `quick --greedy --max-new-tokens 3` stopped on `</think>` and
committed an empty assistant turn. Commit path was unconditional on non-emptiness while
`manifest.json` asserted the invariant. Fix: closed path now drops whitespace-only answers
with a diagnostic distinct from "stopped inside reasoning". Also corrected the generator
description string that had diverged from `manifest.json` (found during merge-conflict
resolution — regenerating would have silently reverted the manifest correction).

Durable rules: "A committed turn with an empty answer poisons context exactly as an
unclosed one does." / "A checked-in fixture must be reproducible from its generator."
(`.squad/decisions.md`, reconstructed rules section, 2026-07-29)

Inbox drop `batty-reasoning-fixture-revision.md` was lost when the worktree was deleted
before Scribe ran; content reconstructed into `.squad/decisions.md`.

Full pre-compaction history in `history-archive.md`.

### 2026-08-11 — CUDA Upstream Audit: Both Candidates Dead

Audited the two CUDA candidates from PR #763 against upstream `main` @ `16b486a2`:

1. **MatMulNBits int4 block-128 GEMV** — DEAD. Upstream already has `block_size == 128`
   in the M=1 GEMV (`matmul_4bits_m1_impl.cuh:152`) and the fpA_intB GEMV/GEMM paths.
   Issue #23004 is about CPU, not CUDA.

2. **QMoE parallel routing** — DEAD. PR #28980 (optimize SoftmaxTopK) already merged.
   Issue #28987 tracks 8+ active Microsoft PRs on QMoE optimization.

Evaluated four next-best options (accuracy_level=4, GGUF formats, QMoE grouping,
graph capture/VMM/tiered KV) — all either already covered, not upstreamable, or
entangled with our runtime. Conclusion: no viable portable CUDA gap exists.

No kernel was implemented. No upstream PR is warranted.
Decision file: `.squad/decisions/inbox/batty-cuda-upstream-audit.md`

Durable lesson: The upstream CUDA EP is mature and actively staffed; our competitive
advantages are runtime-level (graph capture, VMM weight paging, tiered KV) and
architecturally not portable to upstream.

### 2026-08-11 — B1: Output dtype fix (PR #762 reviewer rejection)

Fixed silent wrong-answer bug: `CompiledKernelEntry.output_dtype` was a single dtype
guessed from the first input. Changed to `output_dtypes: Vec<DataType>` sourced from
ORT graph value info at Compile time. Added fail-closed Undefined-output-dtype filter
in GetCapability. No OrtGraph/OrtNode pointers cached past Compile.

Decision file: `.squad/decisions/inbox/batty-b1-output-dtype.md`

Durable lesson: Never infer output dtypes from inputs — always read from the graph's
declared value info. Ops like Cast, Where, Shape have output types unrelated to their
first input type.

### 2026-08-11 — B1 follow-up: Multi-output shape inference for LayerNorm family

Root cause: `ShapePreservingNorm` emitted input[0]'s shape for ALL outputs. Per ONNX spec,
LayerNormalization Mean/InvStdDev have reduced shape `[d[0]..d[axis-1], 1, .., 1]`.

Structural fix: replaced `ShapePreservingNorm` with `LayerNorm { axis, num_outputs,
full_shape_outputs }`. Resolves axis (including negative) at for_node time. Outputs 1+
get reduced shape unless listed in full_shape_outputs. for_op_domain declines these ops
(fail-closed: declined = correct-and-slower; wrong shape = silent corruption).

Checked: LayerNormalization, SimplifiedLayerNormalization, RMSNormalization,
SkipLayerNormalization, SkipSimplifiedLayerNormalization.

Test results:
- ep-plugin: 155 passed, 0 failed, 0 ignored (unit); 9 passed (integration)
- cpu-plugin with --include-ignored: 21 passed, 0 failed, 0 ignored
  (conformance_layer_norm_multi_output now passes)
- clippy: clean (RUSTFLAGS="-D warnings")
- fmt: clean

Durable lesson: Multi-output ops must not assume all outputs share input[0]'s shape.
Reduction outputs (Mean, InvStdDev) follow keepdims semantics over the normalised axes.

## 2026-08-11 — ARM64 Debug CI failure on onnxruntime PR #31973

**Task:** Root-cause the single failing CI job `Build Linux arm64 Debug / build_test_pipeline`.

**Finding:** Build reached [1452/1458] with zero compiler errors, then silently died during the final 6 link steps (onnxruntime_test_all, onnxruntime_provider_test, etc.). No error message, no exit code, no "ninja: build stopped" — classic OOM-kill signature. VM is Standard_D8pds_v5 (32 GB RAM). CCache missed, so all targets compiled fresh. Docker container killed mid-link.

**Code review:** All x86/ARM64 guards verified correct — `MlasLayerNormKernelAvx2` properly guarded by `MLAS_TARGET_AMD64` in both mlasi.h and platform.cpp. CMake entries properly in x86_64-only sections. No code bug found.

**Verdict:** STILL UNKNOWN — strong OOM circumstantial evidence but cannot conclusively prove without dmesg/kernel logs. No code fix applied. Recommended re-trigger to test flakiness.

**Decision doc:** `.squad/decisions/inbox/batty-arm64-31973.md`

## 2026-08-11 — B2 Fix: Device-ID Comparison for D2D Copies (PR #762)

Fixed the last CUDA EP blocker (B2). Added `is_same_device()` to `transfer.rs`
using `MemoryDevice_GetDeviceId` for device-id comparison when pointer equality
fails. Same-device D2D accepted; cross-device fails closed. 6 new tests, all
non-vacuous. CPU path unaffected. Compiles and type-checks; unvalidated on
hardware (blocked on #768).
