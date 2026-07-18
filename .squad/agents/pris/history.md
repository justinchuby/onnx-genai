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
