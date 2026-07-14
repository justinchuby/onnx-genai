# roy — History

## Project Context (day 1)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **Stack:** Rust edition 2024, Cargo workspace, ORT backend, HF tokenizers.
- **Crates:** onnx-genai, -metadata, -kv, -scheduler, -engine, -ort, -server.
- **Requested by:** Justin Chu
- **Team formed:** 2026-07-12



## 2026-07-12T09:13:00-07:00 — Phase 1 foundation plan delivered
- Assessed Phase 1 status and identified real ORT CPU execution, model/tokenizer discovery, and minimal greedy generation as the critical path.
- Shared context for next batch: Deckard supplied ORT/tokenizer contracts, Batty supplied the generation API, and Pris supplied deterministic metadata/fixture coverage.


## 2026-07-12T09:38:00-07:00 — Phase 2 complete
Roy's Phase 2 plan was executed successfully: paged KV tensor storage, prefix cache lifecycle/CoW, persistent multi-session engine APIs, HTTP/SSE session surface, and Pris's exit tests are now in place. Shared contracts include `prefix_cache_hit_len`, `X-Session-Id`, and standalone ORT runtime packaging.

## 2026-07-12T10:10:00-07:00 — Phase 3 complete
Phase 3 plan completed and executed. Team delivered speculative decoding, tiered/quantized KV, priority/preemption, streaming/accounting hardening, and validation; speedup limitation is environment-bound locally.

## 2026-07-12T12:02:00-07:00 — Phase 4 and long-context plans completed
Roy's Phase 4, tool-use/grammar, and long-context plans were executed: pipeline execution, constrained decoding, OpenAI tool use, Qwen/Hermes validation, and O(1)/token static-cache decode are now recorded. Next roadmap follows DESIGN §23-28 plus paged attention.

## 2026-07-12T13:14:00-07:00 — Architecture review merged
Roy's workspace review is now in decisions: crate split is sound, but engine.rs must be decomposed and §26 needs an engine loop/channel plus DecodeBackend before true batching. §27/§28 need SpeculativeProposer/verifier seams.

## 2026-07-20T00:00:00Z — §34 Router R2+R3+affinity+hardening landed
- R2 (commit 1f58099): Created `crates/onnx-genai-router/` — pure session-aware routing core. Modules: `config.rs`, `node.rs`, `router.rs`, `session_map.rs`, `prefix_map.rs`. Policies: AffinityThenLoad, PrefixThenLoad, LeastKvUsage, Weighted. FNV-1a 64-bit prefix hash; optional JSON session-map persistence. 36 unit tests, clippy clean.
- R3 (commit ee8e464): Runnable reverse-proxy binary with `node_poller`, `proxy`, `api`, `metrics`, `state`, `main`. hyper-util client for transparent SSE streaming; hand-rolled Prometheus text; draining semantic; lazy rebalance. `/router/status|sessions|metrics|drain|rebalance` endpoints; all else proxied. 67 tests, clippy clean.
- Affinity weight fix (commit 54e5363): `Weighted` policy corrected from binary gate to continuous scoring bonus per §34.5. Formula: `kv_usage × kv_weight + normalized_queue × queue_weight − bonus`, where `bonus = affinity_weight` if affinity node and below overload threshold.
- R3 hardening (commit a36cbbd, post Deckard 🟡 review): (1) concurrent poller via `join_all`; (2) miss-on-unknown-id; (3) 16 MiB response cap on session affinity capture; (4) rebalance overload guard (`least_loaded_node_below_threshold`). 73 tests total.


## 2026-07-14T02:37:00Z — ORT2 Phase 1 foundation merged
- **Commit:** 203161c — 6 crates scaffolded, `onnx-runtime-ir` with 34 passing tests
- IR gaps flagged by Deckard (Track A): `DataType::from_onnx` fp8/int4 numbering vs ONNX spec, no `DataType::Undefined`, no unknown-rank `Shape` sentinel. Roy to address before quantized-model work.
