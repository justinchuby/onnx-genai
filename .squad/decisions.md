# Decisions

> Current decision ledger. The prior reconciliation ledger is preserved in
> `.squad/decisions-archive/2026-07-23T20-30-00Z-pre-parity-and-deepseek-ledger.md`.

## Index

- `2026-07-23T00-00-00Z-pre-reconciliation-ledger.md`: earlier processed inbox source records.
- `2026-07.md`: monthly historical ledger.
- `2026-07-23T20-30-00Z-pre-parity-and-deepseek-ledger.md`: archived pre-parity/DeepSeek active ledger.

## 2026-07-23 — Native/ORT parity and real DeepSeek-V2-Lite

### Lock scoped native/ORT decode parity regression coverage
**By:** Roy; reviewed by Fact Checker and Gaff
**What:** `scripts/check_native_ort_parity.py` and `tests/parity/` lock 128-token native/ORT CUDA parity for Phi-4-mini and Qwen2.5-0.5B. Qwen2.5-1.5B and 7B lock their observed first divergences and require native's token to match an independent f32 oracle dequantizing the exact deployed symmetric block-32 Q4 weights.
**Why:** The evidence establishes exact parity for the two recorded fixtures and native agreement with the deployed-Q4 oracle at the two measured divergence positions. It does not claim global numerical equivalence or blanket backend superiority.

### Keep parity-oracle scope explicit and harden before package changes
**By:** Gaff and Fact Checker
**What:** The current Q4 oracle is correct for the locked Qwen packages: low-nibble-first packing, implicit zero point 8, float16 scales, block size 32, no explicit zero points, and no `g_idx`. Future package expansion must add graph-contract guards or generalize dequantization; keep oracle provenance independently captured and assert its split relationship explicitly.
**Why:** ORT-CPU agreement alone is not an independent ground truth. The exact-Q4 f32 oracle supports the bounded observed claims, while preventing unsupported extrapolation to different artifacts or later autoregressive positions.

### Record the real DeepSeek-V2-Lite int4 export contract
**By:** Batty
**What:** A real bf16 checkpoint was exported to f16/int4 ONNX with 27 decoder layers, 26 QMoE nodes, 27 Attention nodes, and 189 MatMulNBits nodes. It uses asymmetric block-128 quantization, 64 routed experts with top-6 routing, and MLA widths of K=192/V=128; ONNX structural validation passed.
**Why:** This is a full-depth real-weight artifact suitable for native semantic validation, superseding synthetic-only structural evidence.

### Block real DeepSeek native semantic conclusions on block-128 support
**By:** Marsten
**What:** The real artifact fails before token 1 at layer-0 `q_proj`: strict CUDA placement has zero CPU fallbacks, but native fp16 `MatMulNBits` accepts only the block-32 packed layout while all 189 dense nodes are block-128. QMoE and MLA are not reached, so no semantic conclusion about them is valid.
**Why:** Resolve the layout incompatibility by re-exporting dense projections as block-32 or adding native fp16 block-128 MatMulNBits support. The latter is in flight; a block-32 re-export is also in flight.

### 2026-07-23: CORRECTION — block-128 already on main; Deckard branch superseded; stale-checkout hazard fixed
**By:** Squad (Coordinator), for Justin Chu
**What:** The prior "block-128 unsupported" entries are WRONG — they came from builds on a STALE checkout. `origin/main` (569507c) DOES support fp16 int4 MatMulNBits `block_size != 32` via `matmul_nbits_gemv_f16_general_bs` (`0fa57b0` + `c04a622`, from Gaff's earlier `feat/matmulnbits-block128`); test `fp16_gemv_matches_dequant_reference_block128` passes on main and the real DeepSeek block-128 export decodes coherent text (Holden smoke, 12.62 tok/s, q_proj OK). Deckard's `perf/matmul-nbits-block128` (5821162) branched ~67 commits behind main and re-implemented block-128 redundantly → Holden 🔴 REJECT (SUPERSEDED, do NOT merge; his "158/2 pre-existing" was measured vs the stale parent — those 2 tests pass on main).
**Why / action:** ROOT CAUSE was TEAM ROOT `/home/justinchu/onnx-genai` left on a stale branch, so Marsten's DeepSeek runs built stale code (missing block-128 + QMoE/DeepSeek fixes) → false hard-fail then garbage. FIXED: detached TEAM ROOT at origin/main 569507c. RULE: always build/benchmark from origin/main (or a worktree based on it), never the `.squad` state branch. Authoritative DeepSeek semantic re-run in flight (Marsten) from clean build at `/home/justinchu/wt-ds-semantic`.

## 2026-07-23 — DeepSeek MLA dtod race fixed and merged

### DeepSeek-V2-Lite garbage decode root-caused to CUDA dtod stream ordering
**By:** Rachael; reviewed by Holden; merged by Coordinator
**What:** The real-weight DeepSeek-V2-Lite native CUDA garbage-decode bug was a device-stream ordering race in `CudaRuntime::dtod`: kernels produced K/V on the EP non-blocking stream, while D2D copies ran on the legacy default stream and could read stale `k_nope`/V columns in the MLA `kv_b→Split→Concat→Reshape→Attention` chain. Commit `1fe314f` synchronizes before `memcpy_dtod_sync`, adds `runtime::tests::dtod_waits_for_pending_stream_writes`, and was fast-forward merged to `origin/main`.
**Why:** This is a general CUDA correctness fix for every native `dtod` caller, not a DeepSeek-specific workaround. Holden's fresh review verified the diagnosis, capture safety, regression failure without the fix, CUDA gate `202/0`, clippy clean, and no meaningful Qwen/Phi perf regression.

### GLM-4-9b dense native decode remains coherent after the fix
**By:** Marsten
**What:** Real GLM-4-9b int4 native CUDA decode was coherent on both baseline `569507c` and fixed `1fe314f`, byte-identical, roughly 101→102 tok/s noise, with CUDA graph active (`captures=1`, `replays=61`, `fallbacks=0`). ORT-genai could not load this export because `genai_config.json` is absent, so the evidence is native-only.
**Why:** GLM dense native support remains valid, and the DeepSeek dtod fix is a broader CUDA correctness win without regressing dense cuda-graph decode.
