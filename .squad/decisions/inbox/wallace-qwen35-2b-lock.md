### 2026-08-19 — Golden-lock enable: qwen3.5-2b-text hybrid moat is now regression-proof (PR squad/qwen35-2b-text-moat-lock)

**By:** Wallace (benchmarking / perf-fairness)

**What.**
Added a permanent native-CUDA byte-identity greedy decode-lock for the **Qwen3.5-2B-text hybrid** (linear-attention/SSM + gated attention), mirroring the gpt-oss #1418 lock. New test:
`crates/onnx-genai-engine/tests/qwen35_2b_text_decode_lock.rs`
- Reuses the shared `common/decode_lock.rs::assert_native_matches_golden` helper (native CUDA, `DecodePrecision::Model`, greedy, graph capture = production default) — same structure as `gpt_oss_20b_decode_lock.rs`.
- Env override `QWEN35_2B_TEXT_DIR` with on-box `DEFAULT_MODEL_DIR` fallback (mirrors the `QWEN35_0_8B_HYBRID_DIR` pattern) → runs on this box without env setup.
- Feature-gated `#![cfg(all(feature = "native-backend", feature = "cuda"))]` + `#[ignore]` (needs `--ignored`/`--include-ignored` + the real export + a CUDA device).
- Correctness lock only (no perf assertion — perf is machine-variable and lives in this note).

**Golden token IDs (prompt `"The capital of France is"`, 24 tokens, GPU2-pinned, `--test-threads=1`):**
```
[11751, 13, 561, 6511, 314, 279, 3516, 4042, 369, 6312, 11, 414, 707, 13,
 561, 6511, 314, 279, 3516, 14634, 369, 6924, 13, 561]
```
Decoded: `" Paris. The capital of the United States is Washington, D.C. The capital of the United Kingdom is London. The"` — coherent.

**Determinism / byte-identity verdict (the coordinator's honesty gate):** greedy decode on this hybrid **is deterministic** — locked byte-identical across:
- graph capture ON (production default) — 2 consecutive PASS (24.7s, 24.8s),
- graph capture OFF (`ONNX_GENAI_CUDA_GRAPH=0`, eager) — 1 PASS (25.2s), same 24 IDs.
No run-to-run drift, no graph-on-vs-off drift → a plain byte-identity lock is valid (no need to fall back to a logit-tolerance/coherence-only anchor).

**Moat A/B measured this session (context for why this lock matters — GPU2 idle, greedy, 96 tok/skip 8, medians-of-5, tok/s ± pstdev):**
| ctx | native g1 (capture) | ORT-CUDA (eager, graph-blocked) | native ÷ ORT |
|---|---|---|---|
| short ~5 | 174.89 ±1.86 | 164.60 ±3.74 | 1.06× |
| mid ~362 | 168.98 ±2.03 | 112.86 ±0.23 | 1.50× |
| deep ~1729 | 168.59 ±4.84 | 55.58 ±2.21 | **3.03×** |

Moat class = **graph-BLOCK** (ORT places ~1027/1037 nodes on CUDA but 25 `Memcpy` nodes → "unable to run CUDA graph", eager-only; native captures the whole hybrid, fallbacks=0). Native is context-flat (18/24 layers are constant-state linear-attn/SSM); ORT collapses with context → the win GROWS with context.

**Why.**
gpt-oss and DeepSeek moats are CPU-fallback; this is the first *graph-block* moat we've locked. Native runs qwen3.5-2b-text coherently on GPU today, so the risk is silent regression of the hybrid enablement (LinearAttention / CausalConvWithState / mrope / capture) — exactly what a byte-identity golden lock prevents. Locking the native stream (not an ORT reference) is correct here because ORT runs eager-only and is not a byte-identity oracle for the captured int4 hybrid decode (same rationale as gpt-oss/deepseek locks).

**Scale-up follow-up (NOT in this PR):** `qwen3.5-9b-generic-cpu-2` is the same hybrid arch (32 layers, 16h/4kv, head_size 256) but ships as a multimodal SPLIT export (`embedding.onnx` + `vision.onnx` + `text.onnx`); the native loader rejects multi-`.onnx` dirs (`multiple .onnx files found … expected decoder.onnx or exactly one`). Needs a text-only re-export → routed to Deckard (export/Mobius). Same blocker on qwen3.5-0.8b.

**Discipline:** fresh base `origin/main` 5282a5e3a; GPU2 pinned (re-probed idle); `--test-threads=1`; staged only the two new files (`qwen35_2b_text_decode_lock.rs` + this note) — no `git add -A`; target/ left; commit trailer `Co-authored-by: Copilot`. PR `squad/qwen35-2b-text-moat-lock` — did NOT self-merge; awaiting coordinator re-validation of the golden lock.
