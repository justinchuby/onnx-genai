# batty — History

## Project context
Engine/EP implementer for the Rust ONNX runtime. Canonical ownership: ORT owns forward execution and physical KV; engine owns generation policy and logical KV. CPU kernels rely on session-side `strided::view_in_bounds` before dispatch.

## Summary through 2026-07-14T20:05:00Z

### Engine and KV foundations
Delivered generation, paged/prefix KV, constrained decoding, extensibility seams, prompt-lookup speculative decoding, SWA/sink hardening, mixed-layer KV groundwork, and early vision expansion. Metal-prefill hybrid was measured slower than CPU and should not be productionized. Connector KV import should eventually degrade to `Ok(None)` on import-runner failure.

### ORT2 EP and C API
Implemented the pure-Rust CPU EP foundation and expanded it through the Phase-1 kernel set; contributed the Phase-1 C ABI with opaque handles, panic fences, atomic run commit, and explicit error mapping. Hardening closed legacy Softmax semantics, NaN propagation, saturating casts, checked allocation geometry, dynamic-output guards, and shared Slice planning. Deckard completed the storage-byte overflow correction after Holden rejected the initial artifact.

### Optimizer and fused kernels
Generalized dispatch to `(domain, op_type)` and moved optimizer fusions to `com.microsoft`. Implemented executable, parity-tested LayerNorm, FusedMatMulBias, FusedGemm, FusedAttention, and Gelu paths with strict decline-to-fuse guards. bert_toy remained within reference tolerance; fusion Phase 2 is complete.

### EPContext
Implemented session consume and writer v1. Consume supports primary/reference resolution, external blobs, payload dedup, and executor bypass. Writer v1 was rejected for non-injective sidecar naming; Batty is locked out of that artifact. Leon and Gaff produced later revisions, with Gaff v3 merged as `0fa025e`. Remaining consume advisories include covered-node dedup, duplicate-primary diagnostics, and stronger traversal tests.

### Load-time validation
Unified validation behind `validate_model()` across disk/bytes and session load paths. Models now fail fast with actionable `UnsupportedControlFlow` or `DanglingTensorRef` errors. Empty graphs remain valid; per-kernel/shape-dependent checks remain dispatch concerns; IR invariants already enforced by `Graph::validate` are not duplicated. Holden reviewed 🟢; merged to `origin/main` as `2a99eec`.

### Reviewer lockouts and follow-ups
Batty is locked out of revising: H-D1 storage sizing, fusion follow-ups identified on earlier optimizer reviews, EPContext writer after v1 rejection, and other artifacts explicitly reassigned by reviewers. Preserve reviewer-protocol ownership when addressing advisories.

### 2026-07-16T00:00:00Z — Performance-and-design wave
Unified text codecs around TextCodec and renamed the text APIs.

### 2026-07-16T00:00:00Z — onnx-rs Python serialization bindings
Added the independent abi3-py310 `onnx-rs-python` crate, importing as `onnx_rs`, with opaque models, binary load/save, and text/JSON/TextProto codec functions (`1ae9a3d`). Freysa rejected the path conversion seam; Deckard's cleared path fix landed as `5b348b5`.


## 2026-07-16T19:27:57+0000Z — Native backend selector revision

Under Deckard's strict reviewer lockout, revised native serving in `2ae464b`: exact `com.github.onnxruntime.genai::BlockQuantizedMatMul` opset-v1 Auto detection, explicit errors for unsupported request speculation/pipelines/non-CPU selection, and regressions. Holden re-reviewed 🟢 CLEAR.

- 2026-07-18: Restored the pre-Phase-1 public MtpConfig struct contract via internal ResolvedMtpConfig; MTP Phase 1 re-review approved.
- 2026-07-19: Made BQMoE claim validation zero-allocation (`67abdb5`); hardened PR #30 retry safety and PR #34 capture gating.
- 2026-07-19T07:55:00Z: IndexShare v1's frozen ABI, exact CPU oracle, and interior-sentinel regression merged at `744a9a7`.


## 2026-07-19T07:42:20Z — CSA B2 nit fix landing

- Fixed B2 RMSNorm rounding parity and removed the redundant carry-reset loop in `2067504`; Chew re-reviewed 🟢 APPROVE and 14/14 GPU parity tests remained bit-exact.

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.


## 2026-07-21 — Wave-2 and CI milestone
CI now covers all 27 offline crates with warnings-as-errors and native Windows ARM64. Capture-safe native fp16 CUDA decode wave 2 stacked GQA prep fusion, warp-shuffle RMSNorm, and specialized down-projection GEMV on wave 1, reaching 663–672 tok/s on H200 versus ORT GenAI at 657, with zero fallbacks. All CUDA EP kernel work must remain correct and fast across supported SM architectures, not only sm_90.
- 2026-07-21T23:55Z — Native CUDA opset-24 ConstantOfShape/Gelu/OneHot landed; WP4 revision is the active correction after Zhora/Gaff lockout; clippy hygiene folded.

- 2026-07-22T00:00:00Z — CUDA graph auto-enable in native decode merged to main as `610bde0`; H200 Qwen2.5-0.5B improved 441.49→828.54 tok/s and Phi-4-mini 67.32→94.91 tok/s, token-exact with zero fallbacks. Leon reviewed 🟢.

## 2026-07-24T15:10:00Z — Qwen decode-correctness locks

Authored bit-exact native-CUDA-versus-ORT 64-token decode locks for Qwen2.5-0.5B and 7B. Chew approved; the shared `common/decode_lock.rs` helper is now co-owned with Leon and Pris for Qwen/Phi coverage.

## 2026-07-27T13:12:20+00:00 — Roadmap wave-5

- Under Leon's lockout, added bf16 harness support and ragged causal/non-causal VarlenAttention parity tests; Bishop approved PR #267.

## 2026-07-27T09:42:11-07:00 — CLI sampling/default-budget fixes

- Changed CLI `generate`/`run` absent `--max-new-tokens` semantics to fill the model's remaining effective context instead of imposing a fixed cap; engine/server default remains 128.
- REPL recomputes remaining-context ceiling per rendered turn and `/stats` shows context usage; no headroom reservation or pre-decode refusal policy.
- Added a 512-token warning fallback only when no metadata/decode-path/`--max-context` limit is discoverable, avoiding unbounded ORT decode.
- Fixed sampling flags so temperature/top-p/top-k disable greedy sampling unless `--temperature 0` or explicit `--greedy` applies; added `--no-greedy`.
- Added shared `--max-context` plumbing for `generate` and `run`.
### 2026-07-27 — CLI maintainer-tool backlog queued
Justin confirmed the onnx-genai CLI is a development/maintainer harness, not a consumer product. P0 CLI work in docs/research/cli/00-backlog.md is queued under that charter: live stats discoverability, structured maintainer output, batch/bench harnesses, explicit dev flags for engine behavior, and help snapshots/REPL help. Remote-client mode is out of scope.

## 2026-07-27T16:44:54Z — Wave 9 update
Owns PR #283 / #50 fix cycle after Bishop REQUEST-CHANGES; address conditioning_scale semantics and multi-ControlNet port backing before re-review.
- 2026-07-27T16:44:54Z — Fixed PR #283 / #50 after Bishop REQUEST-CHANGES: removed invented `conditioning_scale`, bound real mobius `controlnet_cond`, made multi-ControlNet fail loudly, and added contract-pinning tests. Bishop approved; PR merged as 687612f5.

## 2026-07-27T16:50:00-07:00 — PR #291 transactional rewind revision

- Took over after Leon's rejection under reviewer lockout and made failed rewinds validate before session mutation. Unsupported sliding-window evicted positions and ORT-owned KV without paged materialization now fail without changing logical tokens, `kv_token_count`, decode state, or paged KV.
- Added model-free regression coverage for both unsupported paths and updated the fork/rewind support matrix to say unsupported rewinds reject cleanly.
## 2026-07-27T14:08:06-07:00 — CLI backend flag

- Added shared `--backend auto|ort|native` plumbing for `generate` and `run`, reusing the REPL `/backend` parser.
- `run --backend` now seeds `SessionSettings`, so the initial load and later `/backend` switches use the same reload-bound backend state.
- Default remains `auto`; explicit `native` reaches `EngineConfig` and fails clearly without native feature support instead of falling back.
- Gates passed in `backend-flag`: `cargo build -p onnx-genai-cli`, `cargo test -p onnx-genai-cli --lib`, `cargo fmt -p onnx-genai-cli -- --check`, `cargo clippy -p onnx-genai-cli --all-targets -- -D warnings`.

## 2026-07-27T18:15:33-07:00 — Scheduler ceiling/reservation regression

- Confirmed by reading and scheduler regression test that PR #277 exposed an ORT-only path issue: native `generate` bypasses scheduler admission, while ORT direct/session generation goes through `drive_next_fcfs`; both load paths set `bytes_per_token`.
- Preserved DESIGN §26.4/§26.11 conservative reservation: scheduler still reserves full `prompt + admitted max_tokens` up front. If the requested ceiling cannot fit but prompt + at least one token can, admission caps `max_tokens` to what the byte budget can guarantee; the engine decodes with that admitted ceiling.
- Scheduler caps must be observable: expose cap metadata on `GenerateResult`, warn on stderr in the CLI, and include the cap in stats/profile output so benchmark runs do not silently shorten their generation budget.
- Kept error reporting actionable for true rejection: batch-full vs KV-budget cause, requested/minimum/available/used/limit/shortfall bytes, running/max batch counts, and concrete mitigation hints.
- Added model-free scheduler tests for full-context ceiling capping, long multi-turn ceiling growth, repeated-turn accounting non-leakage, and error text; added an engine unit test locking that ORT generate uses the scheduler while native generate bypasses it.
- Follow-up review fixes: `budget_cap` remains attached to a sequence across swap-out/swap-in; capped reservation now computes and reserves the adjusted byte amount atomically under the shared `ByteBudget` lock; mismatched scheduler admission cleanup cancels both the admitted request and the originally enqueued request.


## 2026-07-29T18:35:00-07:00 — Qwen3 no-op reshape elimination for ORT parity

- On `qwen3-perf-followups` / PR #398, landed no-op `Reshape`/`Identity` elimination (`76f116cf`) at graph optimization/load-time planning.
- The native profile lost the `Reshape` row entirely (previously 113 calls / ~0.86 ms per token), closing a graph-optimizer advantage ORT already had.
- This executor cleanup combined with Resch's kernel/threadpool work to move native CPU to ORT peak/p90 parity; median still depends on a future low-overhead work-stealing/threadpool lever.

## 2026-07-29T22:00:00-07:00 — Executor cleanup final
- Cleaned deferred TensorView/bookkeeping/output-buffer churn (`aa5b67e6`); subspans improved but final native 106.5/106.2/101.0 tok/s stayed behind ORT 109.5/109.3/107.3, confirming no single executor hotspot remained.
