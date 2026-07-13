# Leon — History

## 2026-07-12: Joined
Hired as Engine Dev (KV & runtime buffers) to add capacity alongside Batty as the runtime grew (9 crates, concurrent engine/KV workstreams). Project: onnx-genai, a Rust ONNX Runtime generative-AI inference runtime. Key context: runtime owns the KV cache; use our own InferenceMetadata (`inference_metadata.yaml`) not ORT-GenAI `genai_config.json`; static-cache/GQA use device-resident buffers with present→past IoBinding aliasing; WebGPU decode needs GQA op + quantized (Q4 MatMulNBits) weights. Real-model exact-equality tests use `intra_op_threads=1`.

## 2026-07-13: Landed attention-sink SWA support
Extended sliding-window attention with StreamingLLM-style sink-token retention across metadata, engine decode state, runtime KV buffers, and paged-KV bookkeeping. Landed as commit `2371864`.


## 2026-07-13T18:30:00Z — Review/fix batch
- Owned Batty's reviewer-lockout fix for issue #14 and landed `458fb78`, adding multi-image over-count bailouts and `tokens_per_tile` guards.

## 2026-07-13T20:55:00Z — SharedKv generalization + robustness fix
- Reviewed Luv's 🟡 gap: malformed speculative metadata block aborted all model loading.
- Renamed all runtime identifiers from `Gemma4Assistant*` to `SharedKv*` across metadata, ORT, and engine (ProposalType, module, types, engine field, wire value). Canonical wire: `proposal_type: shared_kv`.
- Dropped back-compat `gemma4_assistant` alias entirely (pre-release): now degrades to `ProposalType::Unknown`.
- Fixed robustness gap: `resolve_shared_kv` degrades to `Unknown` when `shared_kv` is empty or any group has empty `target_layers` — malformed block no longer aborts load.
- Test `legacy_gemma4_assistant_proposal_type_degrades_to_unknown` added. All tests green; integration test token-identical to greedy. Commit: f101377.
