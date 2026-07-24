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

## 2026-07-24 — DeepSeek MLA determinism fixed on main

### Stream-ordered async copy_reshape is merged
**By:** Gaff; authored by Rachael; merged by Coordinator
**What:** Commit `24531c4` replaced the Reshape/Squeeze `copy_reshape` D2D copy path with stream-ordered `dtod_async` on the EP compute stream. Gaff's fresh review marked it merge-ready: the CUDA EP is single-stream, so same-stream async ordering preserves the prior race fix while removing roughly 200+ stream drains per token. CUDA gate was `203/0`.
**Why:** This is a low-risk eager performance and capture-readiness improvement. It does not by itself enable DeepSeek MLA graph capture; the structural Attention capture blockers remain.

### DeepSeek greedy decode nondeterminism root-caused and fixed
**By:** Rachael; reviewed by Holden; merged by Coordinator
**What:** Commit `621936f` fixed DeepSeek-V2-Lite greedy decode nondeterminism in default-domain `Attention` decode. Root cause was an in-place KV-aliasing RAW race in `build_kv`: aliased `present` wrote the grown cache at `total_seq` stride while reading `past` at `past_seq` stride, corrupting heads greater than 0. The fix stages aliased KV growth into disjoint scratch, runs attention from scratch, then copies the completed cache back. A regression test now proves aliased decode KV growth matches a non-aliased reference and is deterministic.
**Why:** Greedy decode must be deterministic. The fix is general for default-domain `Attention` aliased KV growth and leaves GQA unchanged.

### DeepSeek-V2-Lite MLA decode is now deterministic and coherent on main
**By:** Squad
**What:** DeepSeek-V2-Lite MLA decode is now **DETERMINISTIC + coherent** on `origin/main` via `621936f`, with stream-ordered async `copy_reshape` from `24531c4`. Validated coverage now includes DeepSeek-V2-Lite MoE, DeepSeek-Coder-1.3B, GLM-4-9b, and DeepSeek-R1-Distill-Qwen-1.5B exact native/ORT-genai token parity.
**Why:** The prior DeepSeek incoherence/nondeterminism is resolved. Remaining performance lever is MLA graph-capture enablement, now in progress on `perf/deepseek-mla-capture-enable`.

### MLA graph-capture enablement remains a separate in-progress performance project
**By:** Rachael
**What:** Rachael's capture-enablement plan identified five blockers: host-derived per-step control arrays, growing logical KV/mask extents, per-step synchronizations, synchronous copy-back needing `dtod_async` under capture, and ensuring D2H `nonpad_kv_seqlen` stays off decode-with-past. Proposed direction is fixed-capacity default-domain Attention KV, a device-side valid-length scalar, no per-step host sync, async copy-back, and eventually supported capture once shapes and setup are capture-safe.
**Why:** Correctness is already merged independently. Capture enablement is medium-large engine/kernel work and must be reviewed separately.


## 2025-06-14 — DeepSeek MLA fixed-capacity eager optimization merged

### Fixed-capacity default-domain Attention KV append is on main
**By:** Rachael; reviewed by Holden; merged by Coordinator
**What:** Commit `53afab0` (fast-forwarded to `origin/main`) makes capacity-bound present K/V outputs of default-domain `Attention` use their fixed physical capacity and appends only the new token at the valid fixed slot. The kernel retains logical `[0, total_seq)` read bounds, so padding is not attended; dense/unbound Attention remains staged and GQA is unchanged. The regression test `decode_kv_capacity_append_matches_reference_and_ignores_padding` uses non-zero padding garbage to prove that boundary.
**Why:** This removes DeepSeek MLA's per-token KV restride, scratch allocation, and copy-back while preserving the preceding alias-race correctness fix. Reported eager throughput improved from about 22.2 to 23.5 tok/s (blk32) and 25.2 to 26.0 tok/s (blk128); the CUDA gate was 205/0, greedy output was deterministic, and Qwen GQA capture was unchanged.

### DeepSeek MLA full capture is reachable purely in-engine
**By:** Rachael
**What:** The growing `Unsqueeze_18` Attention bias is a self-contained causal-and-padding mask island consumed only by the 27 default-domain Attention nodes. It can be bypassed for a fixed-capacity, on-device implementation using the existing fixed-capacity raw attention mask plus a device valid-length scalar; no Mobius/export change is required for this plain causal+pad DeepSeek export. This later assessment supersedes the earlier uncertainty that a capacity-length mask required export changes.
**Why:** Capture cannot use the current eager shape-derived valid length, which freezes under replay. Follow-up sequencing is: device valid-length scalar ABI, kernel-side causal/pad synthesis and mask-island pruning, device-side control to remove per-step D2H/H2D, then enable `capture_support()`.

## 2025-06-14 — QMoE decode kernel optimization merged

### Vectorized int4 unpack and layout specialization landed on main
**By:** Deckard; profiled by Pris; reviewed by Chew; merged by Coordinator
**What:** Commit `53f9df6` adds compile-time quant-layout specialization and vectorized int4 unpacking to the CUDA QMoE decode linear path. It was fast-forward merged to `origin/main` from `perf/qmoe-vectorized-unpack`. QMoE linear time fell from 6.36 to 2.51 ms/token (-60.5%); real DeepSeek-V2-Lite end-to-end throughput improved 23.66→26.48 tok/s (+11.9%) for block-32 and 24.25→27.60 tok/s (+13.8%) for block-128. The MoE-only path leaves dense behavior unchanged.
**Why:** Pris established that the former scalar path was instruction/dequant-ALU bound (about 73% SM throughput, 2.8% peak DRAM, 3.2 waves/SM), not HBM-bound or grid-starved, making vectorized unpacking and compile-time layout selection the appropriate general optimization. Chew independently confirmed the vectorized and forced-scalar implementations are byte-identical, specialization has a general non-fallthrough path, gate `206/0`, QMoE parity `27/0`, and three exact deterministic runs on both exports (token `8913`, “ Paris”).

## 2025-06-14 — DeepSeek MLA capture ABI remains in progress

### Device valid-length decode foundation landed on the capture branch, not main
**By:** Rachael
**What:** Commit `e14d7df` on `perf/deepseek-mla-capture` implements the verified (a′) device valid-length ABI for fixed-capacity, mask-masked, single-query default-domain Attention decode. `derive_len` writes a device i32 from the decode mask frontier; optional `dev_len` makes `build_kv` and `attention_row` use that device value while the null path remains bit-for-bit unchanged. It is **not merged to main** and capture remains disabled while bindings grow. Reported validation: CUDA gate `205/0`, clean clippy, deterministic DeepSeek block-32/block-128 greedy outputs, and unchanged Qwen GQA capture.
**Why:** This is a safe decode-only enabling refactor, not completion of MLA capture. The current mask-frontier inference cannot establish a prefill causal mask’s true length. The c1 continuation therefore needs a prefill+decode-capable explicit device seqlens/valid-length ABI, eager prefill followed by a fixed-capacity binding switch at the prefill/decode boundary, deferred capture gating, and removal of host synchronizations, per-op allocation, and HtoD controls before `capture_support()` can be enabled.
