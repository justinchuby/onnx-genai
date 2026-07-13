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
