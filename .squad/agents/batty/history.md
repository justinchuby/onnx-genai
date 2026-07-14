# batty — History

## Project Context (day 1)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **Stack:** Rust edition 2024, Cargo workspace, ORT backend, HF tokenizers.
- **Crates:** onnx-genai, -metadata, -kv, -scheduler, -engine, -ort, -server.
- **Requested by:** Justin Chu
- **Team formed:** 2026-07-12



## 2026-07-12T09:13:00-07:00 — Generation API and engine loop shell delivered
- Delivered `GenerateRequest`, `GenerateOptions`, `GenerateResult`, `GenerateToken`, callback support, `FinishReason`, `StopSequence`, and `Engine::generate` / `generate_with_callback`.
- Key contract for next-batch wiring: processor order is repetition penalty, stop-sequence termination, temperature, top-k, top-p; remaining backend stubs are prompt tokenization, token detokenization, and next-token logits.

## 2026-07-12T09:20:00-07:00 — Phase 1 engine wiring completed
- Wired generation to real ORT session execution and HF tokenizer loading.
- Added graph input discovery for `input_ids`, `attention_mask`, `position_ids`, and past/present KV names; threads model-owned KV tensors when present and falls back to full-sequence reruns otherwise.
- Tiny-fixture CLI greedy generation now runs end-to-end; 13 engine tests pass.


## 2026-07-12T09:38:00-07:00 — Phase 2 complete
Batty delivered persistent engine sessions, stateless `generate` compatibility, minimal FCFS scheduler admission, paged-KV mirroring, same/cross-session prefix reuse, and `GenerateResult::prefix_cache_hit_len` for cache observability.

## 2026-07-12T10:10:00-07:00 — Phase 3 complete
Delivered Phase 3 engine work: greedy speculative decoding, priority scheduling with swap preemption, context-window guard, and real-draft KV rewind fix; real differing model speculation is target-greedy token-identical.

## 2026-07-12T12:02:00-07:00 — Phase 4 engine and decode migration delivered
Delivered constrained decoding (JSON FSM + llguidance JSON Schema/Regex/Lark), pipeline executor APIs, and engine migration to DecodeSession/StaticCacheDecodeSession for O(1)/token KV movement.

## 2026-07-12T13:14:00-07:00 — Samplers and FIM merged
Batty's §24 sampler processors and §23 FIM APIs are now in decisions. Upcoming engine work should align these paths with Sampler, DecodeBackend, and SpeculativeProposer abstractions.

## 2026-07-12T13:52:00-07:00 — §26 engine refactor and batched serving complete
- Batty's engine decomposition is now the foundation for batched serving: `DecodeBackend` and the shared decode loop are the stable seam for past/present, static-cache, and future speculative/paged-attention backends.
- Coordinate future §27/§28 work with Sebastian's `ContinuousBatchManager`, Deckard's active-row API, and Rachael's single-owner server driver.

## 2026-07-12T14:28:00-07:00 — §25 traits and §27 prompt-lookup complete
- Batty delivered behavior-preserving extensibility seams: `SpeculativeProposer`, `DraftModelProposer`, `Sampler` (`GreedySampler`/`CategoricalSampler`), and `ProcessorChain` builder/constraint registration APIs.
- Batty also delivered prompt-lookup speculative decoding through `NgramProposer` and `SpeculativeMode::PromptLookup`, with greedy-equivalent output and multi-token accepts on repetitive input.
- Remaining §27 advanced speculators (MTP/Medusa/EAGLE) need special models; coordinate future paths through the `SpeculativeProposer` verify/commit loop.


### 2026-07-12T14:50:00-07:00
Prompt-lookup speculation and `MtpProposer` are accepted canonical runtime milestones. MTP proposals go through shared greedy verification; future work is optimized hidden-output decode and EAGLE-3 proposer support.

## 2026-07-12T16:14:00-07:00 — Decode ownership and CI clippy convention logged
- Decode ownership is canonical: ORT owns forward execution plus KV buffers/cursors/rewind; engine owns generation policy, loops, stopping, constraints, logical KV policy, and `DecodeBackend`.
- CI clippy is blocking with `cargo clippy --workspace --all-targets -- -D warnings`.
- Engine `kv_bridge` is the largest coverage gap; future EAGLE-3 proposer work should preserve the ORT/engine boundary.

## 2026-07-12T17:30:00-07:00 — EAGLE/logprobs/sampling batch logged
- EAGLE-3 hidden-state contract, opt-in engine token logprobs, and real categorical sampling with per-request seedable RNG are now canonical decisions.
- Future server work should map engine `TokenLogprob` data onto OpenAI chat/completions logprob surfaces.

## 2026-07-12T19:05:00-07:00 — fp16 GQA WebGPU KV consumed
- Runtime now loads + decodes the fp16 `com.microsoft::GroupQueryAttention` WebGPU export. KV bridge accepts Float16 (not just Float32); GQA routes through the `DecodeSession` SharedBuffer runner (present aliased onto a max-length past buffer) instead of the fp32 host paged-mirror. New `genai_config.rs` reads `past_present_share_buffer`/`max_length` from `genai_config.json`; `Value::to_vec_f32_lossy` widens fp16 logits.
- Verified real model: WebGPU coherent ("Paris") at ~21 tok/s decode (up from 9); CPU ~38 tok/s. Remaining follow-up: shared KV buffer is CPU-allocated so ORT still copies host↔device each step — make it device-resident (+ graph-capture provider opts) to pass CPU. See `.squad/decisions/inbox/batty-fp16-gqa-kv.md`.

## 2026-07-12T19:38:00-07:00 — GQA KV path reworked to our own InferenceMetadata (no genai_config.json)
- Per Justin's correction: runtime no longer reads onnxruntime-genai's `genai_config.json`. Deleted `genai_config.rs`; the GQA runtime-owned share-buffer KV decision now comes from our own `inference_metadata.yaml` (`InferenceMetadata`): GQA `model.attention.type` + fp16 KV dtype (`kv_cache.native_dtype` or `model.runtime_configurable.kv_cache.dtype`) + `model.max_sequence_length` (sizes the runtime-owned KV buffer). GQA op = on-device attention compute only; runtime owns/manages KV. No schema.rs change needed.
- New `decode::shared_kv_buffer_len_from_metadata`; `detect_model_decode_path` now takes `shared_kv_max_len: Option<usize>`. Hand-wrote `models/qwen2.5-0.5b-gqa-webgpu/inference_metadata.yaml`. Verified: renamed genai_config.json away → WebGPU server loaded from yaml alone, "capital of France" → "Paris". Non-GQA fixtures still load w/o yaml. `cargo test -p onnx-genai-engine -p onnx-genai-ort` + clippy `-D warnings` → exit 0. Contract for Deckard's Mobius emitter in `.squad/decisions/inbox/batty-inference-metadata-gqa.md`.

## 2026-07-12T21:15:00-07:00 — Perf/CUDA backlog closeout
- Batty runtime decisions now supersede the earlier genai_config path with onnx-genai InferenceMetadata for GQA shared KV; sliding-window attention (§40) is committed as 097bd25.
- WebGPU GQA is coherent but still ORT-capped (~49.6 tok/s); CUDA/H200 follow-up is prepared through Leon's feature-gated CUDA EP/device-KV work.

## 2026-07-13T06:58:00-07:00 — Hybrid GPU-prefill/CPU-decode: premise falsified, KV-handoff proven
- Prototyped the hybrid (Metal-EP prefill → CPU-EP decode) via two `DecodeSession`s + new `DecodeSession::export_kv`/`import_kv` (ZeroCopyRebind cross-session KV handoff, `clone_value_to_owned` host-materialize). New example `crates/onnx-genai-ort/examples/hybrid_prefill_decode.rs` + roundtrip test.
- **Premise is FALSE:** Metal does not win prefill — TTFT is ~1.5–2× SLOWER than CPU at every length and the gap GROWS with prompt length. Root cause (confirmed in `onnxruntime-mps/src/kernels/matmulnbits.metal`): MatMulNBits is a decode GEMV replicated once per prompt token (`grid.y = M`), no `simdgroup_matrix` GEMM tiling, no weight reuse across M → prefill re-streams the full int4 weight set M times (M×-bandwidth-bound). So hybrid total > pure-CPU always. Recommendation: do NOT productionize; the real lever is a prefill GEMM kernel on the Metal EP (Mariette). The KV-handoff seam works, is coherent, and is ~free (0.2–2.8 ms unified memory) — ready for when Metal prefill wins.
- Coherence: pure-CPU/pure-Metal/hybrid all fluent ("The capital of France is Paris."), token-identical for 14 tokens then the known pre-existing fp32 MatMulNBits drift. Green: clippy `-D warnings` clean; `cargo test -p onnx-genai-ort -p onnx-genai-engine` pass. Decision: `.squad/decisions/inbox/batty-hybrid-prefill-decode.md`.


## 2026-07-13T18:30:00Z — Review/fix batch
- Landed initial issue #14 vision token-expansion wiring as `79a030a`; Luv later rejected multi-image accounting/guard gaps, so reviewer lockout applied and Leon owned the fix.

## 2026-07-13T20:55:00Z — SWA/sink hardening nits (Chew review fixes)
- Added two first-activation debug_assert! calls in paged_cache.rs: page_count >= sink_pages and keep_from >= sink_len_target.
- Fixed rewind_to correctness bug: was rejecting valid positions in pinned sink prefix [0, sink_len). Guard updated to allow [0, sink) rewinds; resets sink_len/retained_start to 0 when rewinding behind sink. New test: rewind_into_sink_discards_window_and_resets_gap_bookkeeping.
- Documented draft sink_tokens=0 rationale in engine.rs: no-op without sliding_window; drafts have independent KV constraints; correct fix path is loading draft's own inference_metadata.
- Commit: 4e51d59. Tests and clippy clean.

## 2026-07-13T23:50:16Z — Pending: A2 graceful recompute fallback (from Chew's K4 review)

**Advisory A2 (owner: Batty):** In `try_connector_kv_injection` (`engine.rs`), a failure from `past_kv_from_payloads`/`import_runner_kv` currently propagates via `?` and hard-fails the entire `generate` call. The path is tightly gated (f32 + ZeroCopyRebind + fresh session + successful fetch), so severity is low. Recommended fix: catch the error and return `Ok(None)` to gracefully fall back to full recompute instead of aborting generation. This matches the spirit of the existing `load_materialized_past` fallback pattern.

## 2026-07-14T00-49-37Z — Gemma4 E2B real-run batch (W2)

**W2 — Heterogeneous per-layer KV geometry** (commit 9db1a3c, combined with Leon W3)
- Added `LayerTensorConfig` to `onnx-genai-kv`; paged KV cache now carries per-layer `num_kv_heads`+`head_dim`
- `new_with_layer_configs` / `new_with_layer_quant_config` constructors; `MaterializedLayerKv` with per-layer dims
- Left uniform staging shim for Leon to remove in W3
- `cargo test -p onnx-genai-kv --lib` → 78 passed (4 new heterogeneous tests)
- **Chew review:** 🟡 SHIP-with-advisories — advisory: connector KvPayload path still uniform-only (dead code for E2B, fix before enabling on any mixed-geometry model)

## 2026-07-14T02:37:00Z — ORT2 ep-api + ep-cpu merged
- **ep-api (65ec9f6):** DeviceBuffer ownership hardening, DLPack alignment (byte_offset, i64 strides), Cost non_exhaustive. Reviewed 🟡 Holden. 17 tests.
- **ep-cpu (ea30279):** CpuExecutionProvider + 7 Phase-1 pure-Rust kernels (MatMul, Add, Relu, Reshape, Transpose, Gather, LayerNorm). Reviewed 🟡 Chew + 🟡 Holden. 39 tests.
- Track D (session) must call `strided::view_in_bounds` before kernel dispatch; kernels trust caller for storage bounds.

## 2026-07-14T05:04:00Z — ORT2 capi Track E + ep-cpu +17 kernels merged

- **squad/ort2-capi** (8c9c8fc): Phase-1 C ABI — opaque handles, null-guarded, catch_unwind-fenced, atomic `ort2_run` commit, `SessionError→OrtErrorCode` mapping. 12/12 tests; Miri-clean. Closes Phase 1. Reviewed 🟢 Holden.
- **squad/ort2-epcpu-ops** (e485a83): +17 bert_toy kernels — Sub/Mul/Div/Pow/Min, Sqrt/Erf/Tanh, Cast, ReduceMean, Softmax, Shape, Unsqueeze, Expand, Slice, Constant, Gemm. 90/90 tests; no new deps. Reviewed 🟡 Chew.
- Softmax uses opset-13 per-axis semantics (correct for bert_toy last-axis; opset-12 coerce guard advisory assigned Roy/Deckard — Batty locked on this advisory).
- Loader gaps flagged (Slice/Expand/Constant shape inference) → addressed by Deckard b6f032e.
