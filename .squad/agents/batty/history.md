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
