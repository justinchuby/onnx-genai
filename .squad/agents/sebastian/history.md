# Sebastian — History

## Project Context (joined day)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **State when joined:** Phases 1-4 done; tool use/grammar/chat-template; Qwen2.5-0.5B runs; Hermes agent E2E; long-context O(1)/token via static-cache in-place KV. Working on DESIGN §26 batched serving + reviews.
- **Requested by:** Justin Chu
- **Joined:** 2026-07-12

## 2026-07-12T13:14:00-07:00 — Performance review merged
Sebastian's perf review is now in decisions. §26 should prioritize active-row compaction, ORT KV as hot source of truth, fewer per-step allocations, direct/borrowed logits access, and explicit snapshot/import/export for paged KV.

## 2026-07-12T13:52:00-07:00 — §26 Stage A/B complete
- Sebastian delivered `Engine::generate_batched_static` and `ContinuousBatchManager`; fixed batched static-cache generation matches individual runs and measured 6.2x throughput on the tiny fixture.
- Future scheduler/perf work should preserve the `submit`/`step`/`poll` contract and use Deckard's active-row compaction when rows finish or new requests are admitted.

## 2026-07-12T14:28:00-07:00 — Batched-test ORT determinism fixed
- Sebastian added `SessionOptions::with_intra_op_threads` and `Engine::from_dir_with_session_options` so correctness tests can force single-thread ORT execution.
- Batched static-cache exact-equality tests now use `intra_op_threads=1`, eliminating reduction-order FP tie flakes while production defaults remain unchanged.
- Preserve this convention for future real-model exact-equality tests.

## 2026-07-12T16:14:00-07:00 — Benchmark and observability contracts logged
- `onnx-genai-bench` and `scripts/run_benchmarks.sh` are canonical for device-comparable Criterion runs; preserve stable scenario names and machine metadata.
- Observability core is canonical: atomic metrics, `/metrics`, `/v1/status`, request spans, trace IDs, driver/session/token/TTFT/latency/cache-hit/429 counters.
- Perfetto, OTLP, and full debug endpoints remain future work.

## 2026-07-12T17:30:00-07:00 — Audio DSP and cross-runtime benchmarks logged
- Native Whisper log-mel preprocessing and the OpenAI HTTP cross-runtime benchmark harness are canonical.
- True 1:1 GGUF benchmarking remains in progress and was intentionally not logged as complete.

## 2026-07-12T21:35:00-07:00 — H200 runbook + CPU decode profile
- Wrote `docs/benchmarks/H200-CUDA-runbook.md`: full build/run/benchmark procedure for the CUDA path on H200, assembled from Leon's CUDA-EP flags and Sapper's stacked CUDA model, with a coherence gate (Hopper/ORT garbled-token caveat), checklist, and troubleshooting.
- Profiled CPU decode: **98.9% of per-token time is ORT `session.run`**; orchestration ~1%. Gap is ORT-kernel-bound, not ours. CPU-vs-CPU (same GGUF): ours 43.6 vs LM Studio CPU 157 tok/s (~3.6x).
- Biggest addressable lever: fixed model ships a **544 MB fp32 `lm_head` MatMul** every token (~23% of per-token cost) — quantize embedding+head in Mobius (GatherBlockQuantized) like the CUDA stacked model.
- Added env-gated profiler (`ONNX_GENAI_PROFILE`) + `profile_decode` harness; added `ONNX_GENAI_INTRA_OP_THREADS` override (M1 Max: 6-8 perf cores optimal, 10 threads ~2x slower). Decision in `.squad/decisions/inbox/sebastian-cpu-profile.md`. Did NOT commit.

## 2026-07-13T07:12:00-07:00 — Foundry Local model-vs-runtime isolation (DECISIVE: parity, not FL win)
- Downloaded FL's exact CPU model `qwen2.5-0.5b-instruct-generic-cpu:4` (SHA `997228…cd21`, byte-identical to the 07-12 bench) and ran it through OUR CPU runtime.
- **Decisive result: decode PARITY.** OURS-on-FL-model ~215 tok/s ≈ OURS-on-our-model ~206 ≈ FL-on-FL-model ~200-212. Warm HTTP: short 211.8 (ours) vs 212.1 (FL); long **175.0 (ours) vs 159.8 (FL) — we lead** after the fp32-GQA shared-KV fix. The 07-12 "FL leads 202.7/165.8" gap was pre-KV-fix + thermal/under-warmed sampling (machine variance 85-216 tok/s unwarmed).
- **Graph diff:** FL fuses Q/K/V into one MatMulNBits (N=1152) → 121 MatMulNBits / 299 nodes vs our 169 / 394 (48 fewer dispatches/token). But decode is bandwidth-bound (M=1), so fused QKV is **decode-neutral** — measured neutral. Low priority for CPU decode; prefill-only.
- **Task B (FL C++):** FL sets **zero custom ORT SessionOptions** — delegates to onnxruntime-genai (`genai_model_instance.cc:29-58`); IO binding + `past_present_share_buffer` are inside that lib. Our runtime already matches (ORT_ENABLE_ALL, IO binding, shared-KV). No missing session option.
- **No code change** (none warranted). Follow-ups: warmup discipline + server startup priming (Leon/Seb), TTFT/prefill ~2-4% residual (Leon), fused-QKV low-prio (Sapper). Doc: `docs/benchmarks/2026-07-13-foundry-local-analysis.md`; decision inbox `sebastian-foundry-analysis.md`. Did NOT commit.

- 2026-07-14T19:05:00Z — DESIGN.md §26.11 Resource Governor merged in `d6736e1`, specifying live byte-denominated VRAM/RAM limits, transactional lowering, and actionable over-budget errors.

## 2026-07-15T00:00:00Z — Cross-agent session update

- Validated non-empty IR>=3 opset imports while preserving custom-only models; merged in the loader legality stack.

### 2026-07-16T00:00:00Z — Performance-and-design wave
Marked Gather non-capturable and fixed thread-count-aware MatMulNBits partitioning.

- 2026-07-16T00:00:01Z — 🟢 Approved Rachael's exact single-consumer `x * Sigmoid(x)`→SiLU fusion (`682c93d`); added multi-consumer non-fusion coverage in `d116a96`. Independent interleaved benchmark: 44.45→47.64 tok/s (+7.2%) with unchanged tokens.

### 2026-07-16T00:00:03Z — Safe decode-thread configuration fix
Revised the rejected decode-only Rayon pool with a pure `resolve_decode_threads(raw, available)` helper (`feea8e5`). Empty, invalid, zero, negative, and overflowing settings now retain default behavior; positive values cap at available parallelism. Holden cleared the change after 413 tests.

### 2026-07-16T00:00:00Z — nxrt Python Engine threading revision
Replaced Rachael's rejected `RefCell`/unsendable Python genai Engine with a sendable `Mutex<RustEngine>` wrapper (`41d8c31`). Engine work releases the GIL and `try_lock` makes concurrent or callback-reentrant access an actionable `RuntimeError`; Holden cleared the fix.

### 2026-07-16T00:00:00Z — GQA decode direct-write review
🟢 Cleared Leon's M=1 contiguous-f32 GQA writer (`1fdd1ec`): prefill, strided, and non-f32 outputs retain the generic writer; BSH/BNSH layouts, RoPE, KV behavior, and grouping are preserved. Independent profiling measured GQA 0.883→0.457 ms/step and throughput 51.58→59.42 tok/s with exact eight-token output; 413 CPU EP tests passed.

## 2026-07-16T00:00:00Z — CUDA M2 packed-GQA review cycle
- 🔴 Rejected Roy's initial packed-GQA artifact for bypassing real packed prefill and failing unsupported-PTX validation; strict lockout enforced.
- 🟢 Cleared Wallace's repaired `4a34c66`: real packed-prefill→aliased-decode coverage, shared SM90 CUBIN fallback, 6/6 GQA and 114/114 CUDA tests passing.

## 2026-07-16T14:20:00Z — M3 device-resident CUDA KV review
- 🟢 Cleared Roy's `398c536`: 48 persistent aliased K/V buffers remain stable and make no KV host transfers. M2 and M3 CUDA streams are byte-identical; the CPU mismatch starts at index 10 and is a pre-existing numerical-drift follow-up.

## 2026-07-16T15:39:27Z — Scribe session update

- Fixed `onnx-runtime-python` `onnx_type_string` exhaustiveness for Undefined/Complex64/Complex128 (`f058594`); this main commit includes the completed onnx-rs full-spec merge.

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.

## 2026-07-21T05:40:00Z — fp16 decode and cross-platform reconciliation

- Integrated the end-to-end fp16 native CUDA decode path (`c8741ba`): coherent H200 Qwen output at about 344 tok/s with zero CUDA-graph fallbacks; f32 remained near 200 tok/s. Holden approved.


## 2026-07-21 — Wave-2 and CI milestone
CI now covers all 27 offline crates with warnings-as-errors and native Windows ARM64. Capture-safe native fp16 CUDA decode wave 2 stacked GQA prep fusion, warp-shuffle RMSNorm, and specialized down-projection GEMV on wave 1, reaching 663–672 tok/s on H200 versus ORT GenAI at 657, with zero fallbacks. All CUDA EP kernel work must remain correct and fast across supported SM architectures, not only sm_90.

## 2026-07-21T11:15:00Z — Wave-3 long-context GQA
- Raised capture-safe fp16 GQA `MAX_SPLITS` 8→16; Holden approved and `3b972bf` merged. Independent H200 review measured about 647→693 tok/s at 1024 tokens (+7.1%), flat at 256, with identical tokens and zero fallbacks.

- 2026-07-22T23:20:00Z — Revised the rejected persistent SPMD pool under lockout; `cee3c20` added real 31-worker parity, precedence diagnostics, and panic-safe poisoning, then merged after approval.
