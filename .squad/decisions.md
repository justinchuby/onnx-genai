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

<!-- scribe-merge-2026-07-24T00-00-00Z-capture-foundation -->
## 2026-07-24 — Capture foundation reconciliation

The following four primary-source inbox records are merged verbatim. Pris's scoped Stage 0 → 1A/1B/1C fan-out is the active roadmap; Tyrell, Deckard, and Leon scopes remain in flight and are not completion records. `25dbb60` is the merged capture foundation (Attention capture eligible, 25.87→27.71 tok/s), not the full 2.4× outcome.

<!-- source: .squad/decisions/inbox/rachael-mla-capture-c1-device-len-abi.md -->
# Rachael — DeepSeek MLA capture: (c1) device valid-length ABI (prefill+decode) landed & verified

**Branch:** `perf/deepseek-mla-capture` @ `9ab1c6e` (stacked on merged main `53afab0`; commits `e14d7df` (a′) + `9ab1c6e` (c1)). Do NOT merge — fresh reviewer verifies the whole device-valid-length ABI at once.

## What the ABI does
The default-domain `('','Attention')` MLA path (DeepSeek-V2-Lite; 27 Attention nodes, no `is_causal` attr, additive mask `Unsqueeze_18` fully encodes causal+padding) now derives its valid **key length from device memory** — mirroring how GQA reads `seqlens_k` — instead of from the KV tensor extent. This is the prerequisite that makes the path capture-eligible (the growing logical extent no longer needs to be the source of truth for length).

- `derive_len` NVRTC kernel scans the additive mask frontier (first fully-masked key) into a device `i32`.
- Optional `dev_len` pointer plumbed `native_decode → executor → StandardAttentionKernel` (`build_kv` + `attention_row` read length from device when non-null).
- **Null-pointer-safe:** when `dev_len` is null, the host path is bit-for-bit unchanged (eager/dense/GQA untouched).

## (c1) fix: robust for BOTH prefill and decode
The original (a′) `derive_len` scanned **row 0** of the mask → correct for single-token decode (returns `total_seq`) but **wrong for multi-token causal prefill** (row 0 is valid only for key 0 → returns 1, not `prompt_len`).

**Fix:** scan the **LAST query row** `i = q_seq-1`. At the final query position the causal+padding frontier equals the total valid key length → returns `total_seq` for decode AND `prompt_len` for prefill.
- `derive_len(mask, mask_kind, key_len, row_base, out_len)` scans `[row_base, row_base+key_len)`.
- `execute()` computes `row_base = last_row * key_len`, `last_row = mask_q.saturating_sub(1)`, `key_len = mask.dims[3]`, `mask_q = mask.dims[2]`. Eligibility relaxed from decode-only to prefill+decode.
- **General**, not DeepSeek-special-cased; gated to the default-domain Attention path via the null `dev_len` fallback.

**Why eager e2e stays bit-identical:** with growing bindings the mask last dim equals `total` (no padding region), so last-row frontier = total = host `total_seq`. Last-row only *differs* from row-0 under a padding region (future fixed-cap, c2/d) or a causal prefill mask — the latter is locked by the new unit test.

## Verification (all pass)
- **Prefill+decode correctness (unit test):** `derive_len_reads_valid_length_from_device_for_prefill_and_decode` — decode→`total`; prefill last-row→`prompt_len`; prefill row-0→1 (guard: fails if it reverts to decode-only / host-shape derivation).
- **Determinism, both exports (greedy, 12 tok, warmups=0):**
  - blk32 ×3 → identical `[8913,13,185,549,19305,280,7239,317,254,28071,13,185]`
  - blk128 ×2 → identical, same seq. Coherent: pos0=8913 ' Paris' — " Paris.\nThe currency of France is the Euro.\n"
- **Dense non-regression:** Qwen2.5-0.5b GQA → `cuda_graph enabled=true captures=4 replays=244 fallbacks=0`, coherent, 358.9 tok/s (GQA path untouched, null `dev_len`).
- **Eager DeepSeek tok/s (64 tok, runs=3, warmups=1, idle GPU 6):** blk32 21.96, blk128 22.39 — unchanged vs ~23 baseline (ABI-only pass; capture OFF as expected).
- **Gate:** `cargo test -p onnx-runtime-ep-cuda --features cuda --lib` = **206/0** (+1 new test over 205); `onnx-runtime-session --features cuda --lib` = **65/0**; `clippy -p onnx-runtime-ep-cuda --features cuda --lib -- -D warnings` clean.

## Capture status: OFF (expected this pass)
Deferred to the next passes. Remaining plan pinned to file:line:

**(c2) prune the `Unsqueeze_18` growing-mask island + synthesize causal+padding bias in-kernel at fixed capacity** (proven engine-side, no export/Mobius dependency):
- Island: `attention_mask → Shape/Unsqueeze/CumSum → GreaterOrEqual causal → And key-validity → Where{0,-65504}`; `Unsqueeze_18` consumed ONLY by the 27 Attention nodes (re-confirm on both exports before pruning).
- Synthesize additive causal+padding bias in `standard_attention.rs attention_row` from (c1)'s device valid-length + fixed-cap `attention_mask [1,max_len]`. Kernel already does causal masking + valid-length bounding internally, so this makes the redundant growing bias unnecessary.
- IR ops available: `create_named_value`/`add_input`/`remove_nodes`/`node_mut`.

**(d) flip `capture_support() → Supported` + remove per-op scratch alloc + entry/exit `synchronize()`** (mirror `gqa_decode.rs` capture-safe pattern: device seqlens ptr, fixed module-global scratch, no sync):
- `standard_attention.rs`: entry sync ~`836`, exit sync ~`1250`; `capture_support()` Unsupported ~`1267`; replace per-op `alloc_raw`/`free_raw` with fixed module-global scratch.
- `native_decode.rs`: capture gate ~`1919` (`!bindings.any(has_dynamic_logical_input_shape())`) requires fixed-cap bindings — needs prefill-eager + post-prefill fixed-cap binding switch + deferred capture gate; host prologue rework so `total_seq`/`capacity_key` still hold at fixed cap (`past_key.seq`=capacity).
- Then VERIFY capture engages: captures≥1, replays=N, fallbacks=0 both exports; headline eager ~24-26 → captured ~57-61 tok/s (Pris: 40.5ms/tok, only 15.5ms real GPU compute; ~60% orchestration removed by capture).

**Files:** `crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs` (ABI); reference `crates/onnx-runtime-ep-cuda/src/kernels/gqa_decode.rs` (capture-safe pattern); `crates/onnx-genai-engine/src/native_decode.rs` (bindings/gate for c2/d).


<!-- source: .squad/decisions/inbox/rachael-mla-capture-c2d-flip-outcome.md -->
# DeepSeek MLA capture — (c2)+(d) outcome: capture ENGAGES + non-regressing, but 2.4x is OUT OF (a′)(c)(d) SCOPE

**Agent:** Rachael (worker)
**Branch:** `perf/deepseek-mla-capture` @ `cf40abb` (stacked on merged a′+c1 base `9ab1c6e`) — pushed, NOT merged.
**Verdict:** Capture engages, correct, deterministic, dense-safe, gate 207/0. **BUT the whole-step ~57–61 tok/s (2.4x) is NOT reachable within the Attention-path (a′)(c)(d) scope** — decisive evidence below. Recommend re-sequencing the remaining win as a separate general/MoE-op capture-safety effort.

## What landed (staged commits)
- **(c2) `af907e3`** — freeze default-domain `('','Attention')` KV/mask bindings to fixed capacity at single-token decode:
  - `executor.rs::kernel_input_uses_physical_capacity` (~1122): default-domain Attention KV inputs 4,5 treated as physical-capacity (mask-driven, non-causal), mirroring GQA.
  - `executor.rs` present-shape widening (~3269): made capacity-aware (`kv_capacity_bound`) so a physically-bound past does not inflate present beyond the capacity buffer (fixes a prefill `accepts_output` overflow: past_phys 4096 + cur 5 = 4101 > cap 4096).
  - `native_decode.rs::extend_mask(start,end,expose_len)`: freezes mask binding logical to `[1,max_len]` at decode (prefill keeps growing + runs eager); construction mask logical `[1,max_len]` so `graph_enabled` isn't killed.
- **(d) `cf40abb`** — make the default-domain Attention decode capture-safe + flip `capture_support`:
  - Persistent `StdAttnWorkspace` (module-scoped scratch, no per-op alloc on capture path; `reserve()` refuses to grow during capture, Drop frees).
  - Entry/exit `synchronize()` guarded on `!is_capturing()`; control `htod` uploads guarded out of capture.
  - A warmed fixed-capacity device-valid-length single-token decode step records a capture signature; `capture_support()` → `Supported` only when the signature is present (eager/dense/growing decline).
  - **Regression test** `capture_support_gated_on_warmed_device_valid_length_signature` (general): fresh kernel declines; Supported only after a device-valid-length decode signature is warmed — fails if `capture_support` reverts to unconditional Supported or the device-length requirement is dropped.

Design note: I did NOT IR-prune the `Unsqueeze_18` island. I froze the *binding* logical shape at decode instead (de-risked, no export/IR surgery). This makes the KV/mask BINDINGS capture-eligible, but does NOT make the island's intermediate ops capturable — see seam evidence.

## VERIFICATION (all on idle GPU 6, `source .cudaenv.sh`)
### Capture ENGAGES — both real exports
```
blk128: cuda_graph: enabled=true captures=1 replays=9  fallbacks=0
blk32 : cuda_graph: enabled=true captures=4 replays=244 fallbacks=0
```
### Determinism + coherence (3× blk32, 2× blk128) — IDENTICAL every run
```
generated_token_ids: [8913, 13, 185, 549, 19305, 280, 7239, 317, 254, 28071, 13, 185]
```
pos0 = 8913 ' Paris' → " Paris.\nThe currency of France is the Euro." ✅

### Dense non-regression (Qwen2.5-0.5b GQA) — UNTOUCHED
```
cuda_graph: enabled=true captures=4 replays=84 fallbacks=0
throughput: 268.80 tok/s ; coherent (" Paris. It is the largest city ...")
```
GQA path bit-for-bit unchanged (my rule is default-domain-Attention-only).

### GATE
- `CUDA_VISIBLE_DEVICES=6 cargo test -p onnx-runtime-ep-cuda --features cuda --lib` → **207 passed / 0 failed** (206 baseline + 1 new capture test).
- clippy `--lib -D warnings` clean: ep-cuda, session, engine.
- clean worktree; committed + pushed.

## HEADLINE (the part that falls short) — eager vs captured, blk32, tokens=64/runs=3/warmups=1
```
EAGER  (ONNX_GENAI_CUDA_GRAPH=0): 25.87 tok/s, 38.66 ms/step  (captures=0)
CAPTURED                        : 27.71 tok/s, 36.09 ms/step  (captures=4 replays=244 fallbacks=0)
```
**+7% only** — NOT the ~57–61 tok/s (2.4x) target.

## WHY 2.4x is out of scope — DECISIVE seam evidence (`ONNX_GENAI_LOG_CAPTURE_SEGMENTS=1`, blk32, 1 decode step)
The decode step fragments into **~727 eager seams**, of which only ~36 are in (c2)/(d) scope. Per-op-type seam counts:
```
268 Reshape   ("copy path, not a capture-validated zero-copy view" / data-dependent shape)
 81 Split     ("reads runtime split sizes on host + trailing stream synchronize")
 54 Concat    ("trailing host stream synchronize")
 53 Cast      (data-dependent output shape unresolved)
 52 Mul
 27 Expand    ("allocates/uploads/frees per-call broadcast metadata + synchronize")
 27 Attention (data-dependent input shape unresolved — consume mask island output)
 26 QMoE / TopK / Softmax / ScatterElements / MatMul / GatherElements  (data-dependent)
  ~9 mask-island ops (CumSum/Slice/Unsqueeze/GreaterOrEqual/And/Where/Cast)
```
**~690 of the ~727 seams are inherently non-capture-safe general + MoE ops (Reshape/Split/Concat/Cast/Mul/Expand + QMoE/TopK/Softmax/Scatter/Gather/MatMul) that live entirely OUTSIDE the Attention path.** Each seam is an eager fallback (with per-op alloc/sync), so per-step orchestration overhead (Pris: ~60% of the 40.5 ms/token) persists. Even a *perfect* mask-island prune would only unblock the 27 Attention nodes (they seam because they consume the data-dependent mask island), leaving ~690 seams — the 2.4x cannot materialize until those general/MoE ops are made capture-safe.

## RECOMMENDATION / OWNER ROUTING
- **LAND** this branch as a verified, non-regressing **capture foundation**: the default-domain Attention decode path is now genuinely capture-eligible (device valid-length ABI + fixed-cap bindings + capture-safe scratch/sync), engages with fallbacks=0, is deterministic + coherent, and does not regress dense/GQA. It is the correct prerequisite for the whole-step win. (Fresh reviewer verifies; do NOT merge yet.)
- **RE-SEQUENCE the 2.4x** as a NEW, larger effort: capture-safety for the general-op kernels (Reshape zero-copy view, Split/Concat without host-side sync, Expand without per-call alloc, Cast) + the MoE path (QMoE/TopK/Softmax/ScatterElements/GatherElements/MatMul must resolve output shapes before capture — likely fixed-capacity/token-count bounds). This is cross-cutting kernel work (engine + QMoE owner + matmul), NOT Attention-path (a′)(c)(d).
- My honest read: I did NOT force a fake whole-step number, and I did NOT push a regressing half-state — this is a strictly-improving, verified, separable piece + the evidence you need to scope the real remaining work.

Pinned file:line for the follow-up:
- Seam sources: general-op kernels in `crates/onnx-runtime-ep-cuda/src/kernels/` (reshape/split/concat/cast/expand) + `qmoe.rs` + `matmul*.rs`; each needs a capture-safe (no host-sync, no per-op alloc, resolved-shape) path.
- Attention path is done: `standard_attention.rs` capture_support (~1594), workspace (~559+), device valid-length (a′/c1) already merged.


<!-- source: .squad/decisions/inbox/pris-capture-seams-scope.md -->
### 2026-07-24: DeepSeek capture-seam scope and prioritized fan-out
**By:** Pris

## Executive conclusion

Rachael's Attention work is functioning: the default-domain MLA Attention
kernel can capture. The remaining speedup is blocked by **727 eager seam
nodes, grouped into 190 contiguous eager segments between 191 tiny captured
segments**. Both DeepSeek-V2-Lite exports have exactly the same seam topology.

The most important correction to the proposed routing is that the 26 `MatMul`
seams are ordinary f32 `ai.onnx::MatMul` router projections handled by
`kernels/matmul.rs`; they are **not** `MatMulNBits` seams. Those 26 nodes alone
appear to account for roughly **7.7 ms/token** of recoverable eager overhead.

The work should fan out into four parallel implementation scopes after one
shared resolved-shape prerequisite:

1. executor warmed-shape capture binding;
2. movement kernels;
3. f32 M=1 `MatMul`;
4. MoE routing/indexing/softmax integration.

## Reproduction

Source baseline was fresh `origin/main` `53f9df6`; the actual segmentation
trace used Rachael's `perf/deepseek-mla-capture` `cf40abb`, because that branch
contains the capture-eligible Attention path being scoped.

```bash
source /home/justinchu/onnx-genai/.cudaenv.sh

ONNX_GENAI_LOG_CAPTURE_SEGMENTS=1 CUDA_VISIBLE_DEVICES=1 \
  /home/justinchu/wt-mla-capture/target/release/profile_native \
  --model /home/justinchu/ds-e2e-artifacts/deepseek-v2-lite-real-int4-blk32 \
  --ep cuda --steady --decode-skip 8 --warmups 0 --runs 1 --tokens 12
```

GPU 1 was the only 0%-utilization device during the run; it had another
process's resident allocation, so throughput from this run is not used for
small-delta claims. The seam topology is deterministic and matched Rachael's
trace. The block-128 export produced the identical 727-node list.

Exact seam counts:

| op | count | reported reason |
|---|---:|---|
| Reshape | 268 | 216 copy-path unsupported; 26 unresolved output; 26 unresolved input |
| Split | 81 | runtime split-size host read + trailing sync |
| Concat | 54 | trailing host sync |
| Cast | 53 | unresolved output shape |
| Mul | 52 | unresolved output shape |
| Expand | 27 | per-call metadata allocation/upload/free + sync |
| Attention | 27 | unresolved mask input shape |
| MatMul | 26 | unresolved output shape |
| Softmax | 26 | unresolved output shape |
| TopK | 26 | unresolved output shape |
| GatherElements | 26 | unresolved output shape |
| ScatterElements | 26 | unresolved output shape |
| QMoE | 26 | unresolved output shape |
| mask island | 9 | CumSum/Unsqueeze/Slice/comparison/Where/Cast |

Reason totals are 293 unresolved outputs, 53 unresolved inputs, 216 Reshape
copy-path declines, 81 Split declines, 54 Concat declines, 27 Expand declines,
two Unsqueeze kernel declines, and one CumSum decline.

## Cause classification

Legend:

- **A:** dynamic/unresolved logical shape;
- **B:** per-call scratch or metadata allocation;
- **C:** host synchronization or D2H;
- **D:** no truthful capture-supported kernel path.

| category | cause | evidence and required state |
|---|---|---|
| Reshape | **D**, plus **A** for 52 | The kernel performs an async D2D copy but explicitly declines capture because the copy path is not capture-validated (`movement.rs:257-326`). Implement a bounded zero-copy `view_outputs` path for contiguous reshape, or validate fixed-buffer async-copy capture. The MoE flatten/unflatten pair also needs warmed resolved shapes. |
| Split | **C+D** | A static capturable path already exists (`movement.rs:921-982`, `1081-1145`), but it accepts only the single-input form. All 81 export nodes use a second input that is a constant initializer, so none qualify. Canonicalize constant split sizes into the static plan/single-input form; dynamic Split may remain eager. |
| Concat | **C+D** | Launches stream-ordered copy kernels, then unconditionally synchronizes and reports unsupported (`movement.rs:788-886`). Fixed-shape Concat needs no host sync or workspace. |
| Expand | **B+C+D** | Rebuilds, allocates, uploads, synchronizes, and frees broadcast metadata every call (`movement.rs:381-429`, metadata helper `230-254`). Use a warmed persistent metadata cache keyed by exact shape. |
| Cast | **A only** | Kernel already skips synchronization during capture and advertises Supported (`cast.rs:299-318`). Once the exact warmed shape is seeded, these 53 nodes should fold into adjacent graphs without Cast code changes. |
| Mul | **A only** | Binary elementwise already has persistent broadcast metadata, exact warmed signatures, and Supported capture (`elementwise.rs:594-659`, `740-808`). The 52 declines are structural shape failures. |
| QMoE | **A only** | QMoE already pools scratch, refuses growth during capture, skips its trailing sync, and advertises Supported after warmup (`qmoe.rs:1197-1269`, `1378-1401`). No new per-step workspace design is required. |
| TopK | **A+C+D** | Reads scalar K D2H and synchronizes after the launch (`topk.rs:125-200`). In this graph K is a constant initializer (`k=6`). Cache/fold constant K and add an exact warmed-shape capture path; do not D2H during replay. |
| Softmax | **A+C** | `capture_support()` says Supported, but both cuDNN and NVRTC paths synchronize unconditionally (`softmax.rs:255-272`, `302-344`). Guard/remove the sync during capture and verify the warmed cuDNN descriptor/handle path records correctly. |
| GatherElements | **A+B+C+D** | Host-validates indices, allocates/uploads/frees indexing metadata, synchronizes, and reports unsupported (`indexing.rs:385-453`). Mirror Scatter's persistent metadata + device capture-error pattern. |
| ScatterElements | **A only** | It already has persistent metadata, an exact capture signature, async D2D initialization, and device-side capture error reporting (`indexing.rs:559-668`). Shape seeding should be sufficient. |
| MatMul | **A+B+C+D** | DeepSeek's router is f32 M=1. The existing capture path is only fp16 M=1; f32 falls into cuBLASLt with per-call workspace allocation, heuristic work, sync, and free (`matmul.rs:288-343`, `402-414`). Add a generic f32 M=1 GEMV or persistent preselected algorithm/workspace. This is not a `matmul_nbits.rs` task. |
| Attention/mask | **A** | Attention itself is capture-eligible on Rachael's branch. Its mask input remains unresolved. This is the final Attention-owned dependency, not the source of the other 690 seams. |

The executor rejects nodes before consulting their kernels whenever an input
or output is absent from `resolved` (`executor.rs:2814-2865`; EP default policy
`ep-api/provider.rs:363-385`). `resolve_soft` deliberately omits
data-dependent values and only external/control-flow shapes are currently
seeded for capture (`executor.rs:1951-1969`, `2601-2616`). This explains why
already-capture-safe Cast, Mul, QMoE, and Scatter still appear as seams.

## Time attribution and ranking

Current-main per-op timing used 511 steady decode steps:

```bash
ONNX_GENAI_CUDA_GRAPH=0 ONNX_GENAI_PROFILE_OPS=1 CUDA_VISIBLE_DEVICES=1 \
  ./target/release/profile_native \
  --model /home/justinchu/ds-e2e-artifacts/deepseek-v2-lite-real-int4-blk32 \
  --ep cuda --steady --decode-skip 8 --warmups 1 --runs 3 --tokens 128
```

Median executor time was 34.95 ms/token. A matching low-overhead nsys run
measured 11.72 ms/token of CUDA kernels. The table below estimates direct
capture recovery as per-op wall time minus associated kernel/copy time.

| rank | category | estimated direct recovery |
|---:|---|---:|
| 1 | f32 router MatMul, 26 | **7.5-7.8 ms** |
| 2 | Split, 81 | **2.1-2.3 ms** |
| 3 | Attention/mask, 27 | **1.5-1.8 ms** |
| 4 | Reshape, 268 | **1.3-1.6 ms** |
| 5 | TopK, 26 | **1.0-1.2 ms** |
| 6 | GatherElements, 26 | **0.9-1.1 ms** |
| 7 | QMoE orchestration, 26 | **0.7-0.9 ms** |
| 8 | Concat, 54 | **0.5-0.7 ms** |
| 9 | Expand, 27 | **0.5-0.7 ms** |
| 10 | ScatterElements, 26 | **0.4-0.6 ms** |
| 11 | 53 dynamic Cast seams | **0.4-0.6 ms** |
| 12 | Softmax, 26 | **0.4-0.6 ms** |
| 13 | Mul, 52 | **0.2-0.3 ms** |

The non-Attention categories sum to approximately **17.2 ms/token** of direct
attributed recovery. Removing them also merges 191 captured fragments and 190
eager segments, eliminating repeated graph-replay dispatch, output-buffer
management, and unattributed executor work. That consolidation is plausibly a
further **3-5 ms/token**, bringing the total into the prior approximately
22-24 ms orchestration envelope.

These are estimates, not additive guarantees. Per-op timers charge queued
device wait to the synchronization point that observes it; the ranges should
be used for prioritization, not claimed as isolated benchmark wins.

## Owner/scope mapping

1. **Resolved-shape capture binding — ORT session/executor owner**
   - `onnx-runtime-session/src/executor.rs`
   - Seed warmed JIT-resolved shapes for an exact decode binding signature,
     analogous to control-flow shape seeding.
   - Invalidate/re-capture when any shape or persistent pointer changes.
   - This must not weaken bounds validation or assume all data-dependent shapes
     are globally static.

2. **Movement — CUDA movement + executor-view owner**
   - `movement.rs` plus executor zero-copy view integration.
   - Reshape, Split, Concat, Expand.
   - Existing static-Split work is useful but incomplete for constant
     initializer second inputs.

3. **Elementwise — CUDA pointwise owner**
   - Cast and Mul need integration tests after shape seeding, not redesign.
   - Ensure exact warmed signatures survive segmented capture.

4. **MoE routing — QMoE/indexing owner**
   - QMoE, TopK, GatherElements, ScatterElements, Softmax.
   - QMoE and Scatter should become capturable from shape seeding alone.
   - TopK/Gather/Softmax require kernel work.

5. **Router MatMul — CUDA GEMV owner**
   - `matmul.rs`, ordinary f32 M=1 router projections.
   - Do not route this to a MatMulNBits-only implementation unless that owner
     explicitly also owns generic dense MatMul.

6. **Attention/mask — Rachael**
   - Keep isolated from the scopes above. Her Attention kernel is already
     capture-eligible; only the unresolved mask island remains.

## Recommended staged fan-out

### Stage 0 — shared shape prerequisite

Assign one executor specialist to warmed exact-shape seeding for capture.
Capture safety requires:

- exact resolved input/output shape signature from an eager warmup;
- persistent buffer capacity and pointer stability;
- recapture on mismatch;
- no per-step allocation when the signature matches;
- unchanged view/bounds validation.

Expected immediate beneficiaries without kernel redesign: 53 Cast, 52 Mul,
26 QMoE, and 26 Scatter nodes, plus structural admission for the remaining MoE
chain. Direct expected gain is roughly **2 ms/token**, but its main value is
unblocking every later scope.

### Stage 1 — launch three high-value scopes in parallel

**1A. Router f32 MatMul:** highest direct win, approximately 7.7 ms/token.
Implement fixed-shape f32 M=1 capture-safe GEMV or persistent selected
workspace/algorithm. No per-call allocation, heuristic query, or sync.

**1B. Movement bundle:** removes 430 seam nodes and approximately 4.8-5.2
ms/token directly.

Order inside the bundle:

1. fold constant-input Split into the existing static path;
2. remove Concat sync and advertise fixed-shape capture support;
3. validate Reshape zero-copy or fixed-buffer async-copy capture;
4. add persistent Expand metadata.

Reshape is especially valuable beyond its direct timing because 268 nodes are
fragmenting otherwise capturable regions.

**1C. MoE routing/indexing:** approximately 2.5-3.0 ms/token of new kernel
work.

1. TopK constant/device K with no D2H or sync;
2. GatherElements persistent metadata + device-side bounds error;
3. Softmax no-sync capture path;
4. integration-only validation that QMoE/Scatter/Mul/Cast now fold after Stage
   0.

### Stage 2 — Attention mask closure and integration

Rachael closes the remaining mask shape seam independently. Then run one
cross-scope integration/review pass with hard gates:

- both blk32 and blk128;
- coherent and deterministic token IDs;
- seam count reduced from 727 to a documented minimal remainder;
- captured/eager comparison at 128+ tokens;
- allocation/sync counters;
- Qwen/Phi dense capture non-regression.

The objective is not merely `fallbacks=0`; it is to collapse the 191/190
captured/eager fragmentation toward a small number of stable replay graphs.

## What was not done

- No engine, executor, or kernel source was modified.
- No attempt was made to overlap or revise Rachael's Attention-owned files.
- No implementation savings are claimed; all category savings are attributed
  estimates from current-main per-op and nsys measurements.
- The report does not assume every data-dependent value is safe to freeze.
  Exact warmed signature validation and recapture are mandatory.



<!-- source: .squad/decisions/inbox/marsten-scoreboard-postqmoe.md -->
### 2025-06-14: Post-QMoE native CUDA decode scoreboard
**By:** Marsten
**What:** Fresh `origin/main` (`53f9df6`, detached worktree `/home/justinchu/onnx-genai`) was built with `cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin profile_native`.  Native and ORT figures below are medians of three runs after one warmup, on H200 GPU 0.  The timed region is 120 decode tokens after an eight-token skip from the raw prompt `The capital of France is`; all native runs used `--tokens 128 --steady --decode-skip 8`.

| model | native tok/s | ORT GenAI tok/s | native/ORT | capture status | coherent |
|---|---:|---:|---:|---|---|
| DeepSeek-V2-Lite int4 blk32 | 27.95 (27.85–27.98) | N/A — no `genai_config.json` | N/A | off: 0 captures/replays/fallbacks (MLA capture remains in flight) | Y — token 0 `8913`, “ Paris” |
| DeepSeek-V2-Lite int4 blk128 | 31.08 (31.01–31.16) | N/A — no `genai_config.json` | N/A | off: 0 captures/replays/fallbacks (MLA capture remains in flight) | Y — token 0 `8913`, “ Paris” |
| GLM-4-9B int4 | 120.73 (120.65–120.82) | N/A — no `genai_config.json` | N/A | engaged: 2 captures, 26 replays, 0 fallbacks (16-token diagnostic) | Y — begins “ Paris.” |
| DeepSeek-Coder-1.3B int4 | 797.13 (796.06–797.32) | N/A — ORT GenAI 0.14.1 tokenizer rejects its regex (`Invalid regex: \s?\p{L}+`) | N/A | engaged: 2/26/0 | Y — “ Paris…Germany is Berlin…” |
| DeepSeek-R1-Distill-Qwen-1.5B int4 | 621.26 (610.42–621.98) | 440.52 (440.21–445.00) | 1.41×* | engaged: 2/26/0 | N* — both backends begin `**C iter**` and repeat under this raw prompt |
| Qwen2.5-0.5B int4 | 870.41 (817.92–899.44) | 586.39 (578.30–587.21) | 1.48× | engaged: measured 1 capture, 125 replays, 0 fallbacks | Y — begins “ Paris.” |
| Qwen2.5-1.5B int4 | 619.43 (608.51–619.45) | 441.90 (439.59–444.01) | 1.40×* | engaged: 2/26/0 | N* — native repeats; ORT continuation is fluent |
| Qwen2.5-7B int4 | 295.49 (294.39–295.83) | 262.63 (262.50–262.78) | 1.13× | engaged: 2/26/0 | Y — fluent Paris continuation |
| Phi-4-mini int4 | 320.84 (320.79–320.99) | 238.01 (234.90–238.06) | 1.35× | engaged: 2/26/0 | Y — begins “ Paris.” |

**Why:** This is the post-`53f9df6` scoreboard after vectorized QMoE unpack.  It confirms coherent, native-leading comparisons for Qwen 0.5B, Qwen 7B, and Phi-4-mini; Qwen 1.5B's apparent speed lead is not a valid win because its native continuation is repetitive.  The R1-Distill package is also not semantically suitable for this raw-prompt coherence check even though native and ORT produce the same defective-looking opening.

**ORT method:** ORT GenAI 0.14.1 CUDA, same GPU, raw same prompt, one warmup, three 120-token timed runs after eight untimed tokens.  Its CUDA 12 dependencies were supplied from the local Foundry/ana runtime libraries.  No ORT value is inferred: the three non-GenAI exports genuinely lack configuration, and Coder's ORT tokenizer fails before generation.

**Caveat:** GPU 0 was checked immediately before every native and ORT run and was `0%` utilization / `0 MiB`; GPUs 1–4 had other occupants. This is a shared CPU-perf/capture-build host, so treat these as idle-GPU snapshots and not small-delta regression thresholds.
