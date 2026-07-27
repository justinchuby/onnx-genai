# pris — History

## Project Context (day 1)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **Stack:** Rust edition 2024, Cargo workspace, ORT backend, HF tokenizers.
- **Crates:** onnx-genai, -metadata, -kv, -scheduler, -engine, -ort, -server.
- **Requested by:** Justin Chu
- **Team formed:** 2026-07-12



## 2026-07-12T09:13:00-07:00 — Metadata tests and tiny LLM fixture delivered
- Delivered metadata parser tests for valid YAML/JSON, malformed/schema-invalid parse errors, and runtime capability validation.
- Added deterministic tiny GPT-2-style fixture at `tests/fixtures/tiny-llm/` for next-batch ORT/tokenizer/generation integration without external model downloads.

## 2026-07-12T09:20:00-07:00 — Tiny fixture enabled Phase 1 E2E
- The deterministic `tests/fixtures/tiny-llm/` model and tokenizer enabled the first end-to-end greedy generation smoke test through the facade CLI, engine, tokenizer, and ORT session.


## 2026-07-12T09:38:00-07:00 — Phase 2 complete
Pris delivered Phase 2 coverage for interleaved persistent sessions, reset isolation, KV fork CoW independence, same-session prefix hit (`prefix_cache_hit_len > 0`, warm hit observed as 6), and cross-session prefix reuse with matching greedy output.

## 2026-07-12T10:10:00-07:00 — Phase 3 complete
Delivered Phase 3 validation: real TinyStories coherent CLI/HTTP generation, 12-session KV pressure pass with no OOM, speculative correctness harness, and documented CPU/tiny-model speedup limitation.

## 2026-07-12T12:02:00-07:00 — Qwen, Hermes, VLM, and long-context validation delivered
Validated Qwen2.5-0.5B Mobius builds and coherent generation, HTTP tool use, Hermes/coding-agent tool-loop acceptance, tiny VLM fixture scaffolding, static-cache scatter models, and flat 25-27 ms/token long-context decode.

## 2026-07-12T13:14:00-07:00 — Harness hardening merged
Pris's coding-agent harness sandbox is now in decisions: workspace path confinement, no shell execution, argv allow-list, guarded Python scripts, symlink/traversal rejection, and passing self-test.


### 2026-07-12T14:50:00-07:00
Advanced fixture work is canonical: builders use onnxscript/onnx-ir, `tiny-mtp-full` provides ignored greedy-equivalence e2e MTP coverage, `tiny-eagle3` exists for future proposer work, and paged attention remains blocked by Mobius support.

## 2026-07-12T16:14:00-07:00 — Coverage baseline and vision follow-up logged
- Coverage baseline is canonical: 75.63% line / 74.34% region overall, with KV 93.63, Scheduler 91.70, Server 80.05, Engine 74.87, ORT 68.67 line coverage.
- `scripts/coverage.sh --fail-under-lines 75` is the proposed CI floor; prioritize engine `kv_bridge` and targeted ORT decode error fixtures.
- Vision endpoint routing exists, but real quality needs a mobius CLIP+decoder VLM package and processor metadata.

## 2026-07-20T00:00:00Z — §34 Router R1 (node status endpoint) landed
- Delivered `GET /v1/status` on `onnx-genai-server` implementing the §34.8 node-status contract (`NodeStatus` + `SessionStatus`).
- Added `--node-id` / `ONNX_GENAI_NODE_ID` with hostname fallback and CSPRNG `node-<hex>` default.
- Real fields: `node_id`, `healthy`, `queue_depth`, `active_sessions`; all placeholder fields documented `// not yet tracked`.
- Commit 050259f (initial R1); commit 74314e8 (f32 alignment fix — `kv_usage`/`batch_utilization` changed from `f64` to `f32` to match router's mirror struct).
- Chew's 🟡 review identified the f32 type mismatch; Pris addressed it directly.

## 2026-07-13T23:50:16Z — Pending: A1 multi-layer gold fixture (from Chew's K4 review)

**Advisory A1 (owner: Pris):** The `tiny-llm` fixture used in `local_tiered_connector_fetch_reuse_is_token_identical` has `num_hidden_layers = 1`. Cross-layer ordering in the extract→store→fetch→inject round-trip is not yet exercised. Layer handling is name-keyed and symmetric (export and inject both iterate `kv_model.layers` in order), so risk is low — but a multi-layer gold fixture would close the last layout dimension of the K4 correctness proof.


## 2026-07-14T02:37:00Z — Gemma4 speculative acceptance fix (co-author with Leon)
- **Commit:** 8089a1f — Reviewed 🟡 Chew
- Owned: fixture updates (W5a mixed-head_dim), K4 multi-layer KV coverage, Milestone B numerics sign-off.
- Verified fp16↔f32 conversion exactness and paged-path round-trip is a true inverse for fp16 KV.

- 2026-07-14T19:05:00Z — Reviewed Ana's `nxrt` PyO3 FFI/abi3 binding; verdict GREEN. Binding merged in `878559f`.

- 2026-07-15 — Performed third review of Range hardening; advisory outcome recorded for `29f0772`.

## 2026-07-15T00:00:00Z — Cross-agent session update

- Hardened the Range Float32 parity regression test; included in the opset-coverage consolidation.

### 2026-07-16T00:00:00Z — Performance-and-design wave
Reviewed Gather/Shape/Constant through three resolved rejection cycles.

## 2026-07-16T17:00:38+0000 — Mobius sub-4-bit export wiring
- Opened Mobius PR #406, preserving MXFP4 and IQ4_NL GGUF blocks in `BlockQuantizedMatMul` export nodes.
- Unsupported IQ formats remain on the dequantize/requantize fallback until runtime support lands.

## 2026-07-16T18:11:48+0000 — Mobius full IQ-family export

- Updated Mobius PR #406 to preserve all ten runtime-supported MXFP4/IQ formats as `BlockQuantizedMatMul` raw blocks; the PR remains open.
- Mariette 🟢 verified enum IDs, format strings, dimensions, byte strides, and fallback behavior.

## 2026-07-16T19-27-57+0000 — Scribe session update

- **Real-model sub-4-bit milestone:** Qwen2.5-0.5B IQ4_XS produced coherent CPU-native output through 144 `BlockQuantizedMatMul` nodes (120 IQ4_NL, 24 IQ4_XS), with both formats executed without fallback (`2f65135`).
- **Mobius #406 update:** commit `797fff9` fixes mixed-native scaffolding and emits genai-domain opset v1; 304 tests passed and the PR awaits user merge.

## 2026-07-18T01:20:34Z — PR #25 lifecycle regression landed
- Replaced simulated lifecycle coverage with an isolated child-process test using real `Environment` create/drop and plugin registration; Vasquez approved `dbff29c`, and PR #25 merged.

## 2026-07-19T13:35Z — test-staleness guard
- Hardened unsupported-op executor tests with `NxrtNeverRegisteredSentinelOp` so future real-op registrations cannot invalidate handler-miss diagnostics; landed as `6ba4d96` with 23/23 executor and full session suite passing.
## 2026-07-19T14:10Z — Bitwise/Hardmax rejected
- Added initial Bitwise* and Hardmax (`43df6c0`/integrated `43c0315`), but Luv 🔴 rejected missing fp16/bf16 Hardmax and weak broadcast/rejection tests. Pris is locked out; Deckard owned the revision.


- **2026-07-19T16:15:00Z — CPU-EP reduction wave:** Authored ReduceLogSumExp opset-18 axes-input support, boolean ReduceMax/Min, and empty-set ReduceSum handling in `dc229c1`; Deckard corrected omitted-axes/noop semantics and the fix landed as `6e97ee6`.


## 2026-07-19T18:20:00Z — CPU-EP op coverage 936→975

- Authored BitShift/OneHot/Compress CPU coverage (`9ca9375`); after review lockout, Sapper corrected OneHot bounds and BitShift required direction (`49d8827`), then Gaff approved.


## 2026-07-19T20:10Z — CPU-EP op coverage Batch 4

- Authored IsInf, EyeLike, and mixed-type Pow (`a32e08f`). Luv rejected EyeLike extreme-`k` overflow and dtype truncation; Pris was locked out, Deckard fixed the artifact, and the corrected work landed in `46b2e42`.

- 2026-07-19: Reworked Unique to O(n log n) grouping with canonical NaN and signed-zero behavior. Pinned matched one- and eight-thread Rust/ORT benchmark configurations in 59b17ad; medium-f32 MatMul measured 21.4× and 16.4× slower than ORT respectively.

## 2026-07-21T03:15:00Z — CUDA graph M4 validated
- Added native decode replay integration coverage and corrected the real-Qwen H200 smoke to assert capture success: 1 capture, 62 replays, zero fallbacks (`4755575`, `42b71f7`).

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.


## 2026-07-21 — Wave-2 and CI milestone
CI now covers all 27 offline crates with warnings-as-errors and native Windows ARM64. Capture-safe native fp16 CUDA decode wave 2 stacked GQA prep fusion, warp-shuffle RMSNorm, and specialized down-projection GEMV on wave 1, reaching 663–672 tok/s on H200 versus ORT GenAI at 657, with zero fallbacks. All CUDA EP kernel work must remain correct and fast across supported SM architectures, not only sm_90.

## 2026-07-21T11:15:00Z — SwiGLU fusion review
- 🟢 Approved Mariette's CUDA `Mul(Silu(gate), up)` fusion after verifying guards, fp16 parity, tracing, capture safety, and portability. Reproduced a real 256-token gain with zero fallbacks.

## 2026-07-21T13:15:00Z — MatMul fusion review
- 🟢 Approved Rachael’s QKV-bias and paired gate/up+SwiGLU fusion after bit-exactness, misfire guards, portability, capture safety, and H200 performance checks; stacked throughput reached ~759 tok/s at 256 and ~789 at 1024.
- 2026-07-21T23:55Z — Approved WP2 revised native image processor, DS-1 bounded shape propagation, and related dtype/opset reviews for the segment.

## 2026-07-22T15:05:00+0000 — WP-B1 optional-modality schema landed

Pris authored WP-B1 optional-modality metadata schema support and Bryant approved it; the work landed on origin/main as `a71c6f3`. Rachael's WP-B design note remains active for WP-B2/WP-B3 follow-up reference.

- 2026-07-22T23:20:00Z — Authored the persistent SPMD decode-pool profile/implementation; the default-off lever advanced native M=1 int4 decode to about 17.3 tok/s with bit parity, then entered review and revision.
## 2026-07-24T15:10:00Z — Shared decode-lock helper revision

Revised the Phi decode lock after Holden's rejection: environment-gated real-model coverage and generalized `common/decode_lock.rs` across Phi and Qwen. Holden approved; Pris co-owns the shared helper with Batty and Leon.
## 2026-07-26T22:38:02+00:00 — ORT2 remaining-work audit

- Recorded that ORT2 Phase 1 is complete, full ORT2 runtime vision is roughly 65–70% complete, and core GenAI functionality is roughly 70% complete; remaining work is breadth, compatibility, heterogeneous placement, packages, CI, and productization.

### 2026-07-27 — CLI maintainer-tool backlog queued
Justin confirmed the onnx-genai CLI is a development/maintainer harness, not a consumer product. P0 CLI work in docs/research/cli/00-backlog.md is queued under that charter: live stats discoverability, structured maintainer output, batch/bench harnesses, explicit dev flags for engine behavior, and help snapshots/REPL help. Remote-client mode is out of scope.
## 2026-07-26T21:06:24-07:00 — Mac CPU native-vs-ORT bench harness
- Extended `crates/onnx-genai-bench/src/bin/compare.rs` with direct native CPU EP vs ORT CPU EP comparison, JSON output, warmups, repeated medians, p10-p95 spread, and measured-roofline fraction.
- Measured Qwen2.5-0.5B on M1 Max: native decode median 3.83 tok/s (7.02% measured roofline) vs ORT CPU 45.45 tok/s (83.33% roofline), ratio 0.084x; model-load median native 108.5 ms vs ORT 1199.6 ms.
- Added M1 Max absolute release-harness regression floor `NATIVE_CPU_DECODE_FLOOR_TOK_PER_S = 3.50` plus non-rig Apple-Silicon roofline-fraction floor in `crates/onnx-genai-bench/tests/profile_native.rs`.

## 2026-07-27T07:35:00-07:00 — PR #227 reviewer-comment fixes
- Fixed `--decode-skip 0` inflating decode tok/s by subtracting TTFT instead of `Duration::ZERO`; extracted `decode_throughput()` helper.
- Fixed `--profile-json -` in non-direct mode emitting invalid JSON (markdown + JSON mixed on stdout); mirrored direct-mode stderr routing.
- Added `decode_throughput_skip_0_1_2` test with guard-break proof; all 9 `compare` tests pass.
- Published figures unaffected: profile README used `--decode-skip 2`.

## 2026-07-27T08:11:00-07:00 — SDPA test helpers cfg-gated for x86_64 CI
- Gated `deterministic_values`, `PatternBias`, `PatternMask`, `sdpa_f64_reference` with `#[cfg(target_arch = "aarch64")]` to match their consuming tests.
- Without gating, these helpers compiled as dead code on x86_64 and x86, causing `-D warnings` CI failure.
- Chose precise `cfg` gating over `#[allow(dead_code)]` to avoid silencing future genuine dead-code findings in this module.
- Verified: `cargo clippy --all-targets --target x86_64-apple-darwin -- -D warnings` passes; native aarch64 clippy and 13 SDPA tests pass.

## 2026-07-27T08:40:00-07:00 — Regression guard hardening: dispatch test + raised floors
- Added `fp16_m1_decode_reaches_neon_gemv_not_half_gemm` dispatch-reachability test with `GEMV_F16_TEST_HITS` atomic counter in `matmul.rs`. Uses f16×f16 M=1 tensors matching real model dtype.
- Guard-break verified: test fails on current HEAD (before Iran's M=1 gate in `try_matmul_half`); passes with gate applied locally.
- Raised FP32 absolute floor from 3.50 → 18.0 tok/s, roofline fraction from 0.30 → 0.35.
- Added new FP16 floor test: absolute 28.0 tok/s, roofline fraction 0.25. Would have caught the 4.5× regression (13.37 < 28).
- All machines check roofline fraction; measurement rig additionally checks absolute floor.
- x86_64 cross-compile clean; aarch64 clippy clean; 132/133 matmul tests pass (1 expected failure: dispatch test correctly fails until Iran's fix lands).
- Added aarch64-only `sdpa_f32_neon` parity coverage against scalar and f64 references on Qwen-style decode, odd/tail dimensions, masks/`-inf`, causal/softcap, and large-score softmax stability cases.
- Added a dispatcher reach test proving `sdpa_f32(...)` executes the NEON path on Apple Silicon when MLAS is not selected.
- Guard-break probe skipped the `dot_neon` scalar tail and the new parity test failed (`max_abs=9.221658e-4`, `max_rel=2.034264e0`); restored code passes.
- Tightened model-scale GEMV max-relative tolerance from 2.0% to 1.8%, based on Chew's 1.57% measured worst legitimate f32 accumulation-order drift.
