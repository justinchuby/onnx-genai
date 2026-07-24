# DeepSeek MLA eager perf: fixed-capacity KV + fixed-slot append

**Author:** Rachael (worker)
**Branch:** `perf/deepseek-mla-fixedcap` @ `53afab0` (off correctness fix `621936f`)
**Status:** ready for fresh review — do NOT merge. Scope = (a)+(b) eager deliverable; capture stays OFF (deferred (c) capacity-length mask + (d) capture_support flip).

## Cause (the eager overhead)
DeepSeek-V2-Lite (MLA) decode runs the **default-domain `Attention`** kernel (dense models use `com.microsoft/GroupQueryAttention`). The correctness fix `621936f` made every decode step, for the aliased KV cache, **repack the whole cache** at a wider per-head stride into a **disjoint scratch buffer** and **copy it back** (to break the in-place RAW race). Once the cache lives at a fixed physical capacity that per-step restride + scratch alloc + copy-back is pure overhead.

Proven with a runtime probe (`MLA_PROBE`) on the real blk32 export: the Attention kernel sees KV at the **logical growing extent** (`[1,16,5,192]→[1,16,6,192]`), past/present **aliased** (same ptr), and the attention mask fully encodes valid length.

## Change (files)
1. **`executor.rs` (`exec_kernel_node`)** — for default-domain `Attention`, expose the **present K/V outputs** (consumer-less terminal graph outputs bound to the growing cache) to the kernel at the **binding physical capacity** instead of the logical extent. Gated: only present slots (oi>0) that are `accepts_subshape` capacity bindings with matching rank and physical axis-2 ≥ logical (all other axes equal). Dense/unbound present keeps its inferred shape → **GQA (Qwen/Phi/GLM) untouched**. Does NOT touch `kernel_input_uses_physical_capacity` (avoids poisoning present-shape inference to `cap+cur > physical`).
2. **`standard_attention.rs`** — `build_kv`/`attention_row` take the physical **capacity** as the per-head KV seq stride; loop bounds stay the **valid** `total_seq` (= past+cur, derived from the mask/shapes, unchanged). When the present buffer is a wider capacity binding: **append only the new token's K/V into fixed slot `[past_seq]`** and read exactly `[0, total_seq)`. Dense-aliased present (`cap==total`) keeps the **staged rebuild** — the race fix from 621936f is preserved. No restride / scratch-alloc / copy-back on the capacity path.

General mechanism (any default-domain Attention decode with a capacity-bound present), **not DeepSeek-special-cased**.

## Determinism proof (greedy, 3× back-to-back, both exports)
Both exports, prompt "The capital of France is", identical every run:
```
blk32  x3:  [8913, 13, 185, 549, 19305, 280, 7239, 317, 254, 28071, 13, 185]
blk128 x3:  [8913, 13, 185, 549, 19305, 280, 7239, 317, 254, 28071, 13, 185]
```
Coherent: pos0 = **8913 ' Paris'**; decoded = " Paris.\nThe currency of France is the Euro.\n..."

Probe confirms the capacity/fixed-slot path is engaged (no staging):
```
alias_key=true capacity_key=true stage_key=false key_cap=4096 total_seq=5 past_seq=0   (prefill)
alias_key=true capacity_key=true stage_key=false key_cap=4096 total_seq=6 past_seq=5   (decode append @slot5)
alias_key=true capacity_key=true stage_key=false key_cap=4096 total_seq=7 past_seq=6
```

## Eager tok/s (before 621936f-base → after, same idle GPU 5, warmups=1 runs=3 tokens=64)
| export | before | after |
|--------|--------|-------|
| blk32  | ~22.2  | ~23.5 (+~6%) |
| blk128 | ~25.2  | ~26.0 (+~3-4%) |

Modest but real and consistent — removing restride/copy-back/scratch-alloc. The big win (capture, ~replay) is the deferred (c)+(d) follow-up; capture stays OFF here (expected).

## Dense non-regression
- Qwen2.5-0.5b: capture intact `captures=4 replays=116 fallbacks=0`, coherent (' Paris'), tok/s ~300 before → ~300 after (no regression).
- GLM-4-9b / deepseek-coder-1.3b / Qwen all use `com.microsoft/GroupQueryAttention` → never touch the modified default-domain Attention path (verified by ONNX op-domain inspection + 0 kernel-probe hits).

## Regression test (+1 over 204 → 205/0)
`decode_kv_capacity_append_matches_reference_and_ignores_padding` (standard_attention.rs `alias_tests`): builds an aliased present at a physical capacity wider than the valid length, fills the padding slots `[total, cap)` with **non-zero garbage**, and asserts the attention output equals the dense reference and is deterministic across runs. Fails if the kernel reverts to reading the KV **extent/capacity** as the sequence length (it would fold the garbage padding into the scores). Existing `decode_kv_growth_alias_matches_reference_and_is_deterministic` (dense staging race fix) still passes.

## Gate
- `cargo test -p onnx-runtime-ep-cuda --features cuda --lib` → **205 passed / 0 failed** (204 baseline + new test).
- `cargo test -p onnx-runtime-session --features cuda --lib` → 65/0.
- `cargo clippy -p onnx-runtime-ep-cuda --lib -D warnings` clean; `-p onnx-runtime-session --lib -D warnings` clean.
- Clean worktree `/home/justinchu/wt-mla-kv` off `621936f`.

## Known edge (documented, unreachable in practice)
Capacity detection uses `key_cap > total_seq`. At the exact context-full boundary (valid == max_len == 4096) a capacity-aliased buffer would be misread as dense. Generation bails at `total_len > max_len` and physical == max_len (default 4096), so with realistic prompts this boundary is not hit. A robust fix (explicit capacity flag or 1-slot reserve) belongs with the deferred capture (c)/(d) ABI work.

## Sequencing note
This branch is off `621936f` (Holden's correctness review target). If 621936f changes in review, rebase this onto updated main. My earlier `de5188c` async copy-back is **subsumed** by fixed-slot append (folded in / not carried separately).

---

## Assessment: can (c) the growing attention mask be done ENGINE-SIDE (no Mobius/export)?

**Read: YES — (c) is achievable purely in-engine (on-device fixed-capacity mask synthesis from a device valid-length scalar). No export/Mobius change is required for DeepSeek-V2-Lite.** The full 2.4x capture win is reachable in-engine. Evidence below.

### What the growing mask actually is (traced from the real blk32 ONNX)
The Attention mask input `v_model.Unsqueeze_18` is a standard **causal + padding additive bias** built in cumsum space, shared by all 27 Attention layers:
```
CumSum_5   = cumsum(attention_mask, axis=1)              # per-token valid-position index (left-pad robust)
Sub_9      = shape(attention_mask) - shape(input_ids)    # past_len
Slice_10   = CumSum_5[past_len:]                         # query positions
GreaterOrEqual_12 = query_cumpos >= key_cumpos           # causal triangle
And_15     = causal AND cast(attention_mask)             # + key validity (padding)
Where_16   = And ? 0.0 : -65504.0                        # additive bias
Unsqueeze_18 = [B, 1, cur, total]                        # grows with total  <-- the capture blocker
```
It is a **pure function of `(past_len, cur_len, attention_mask)`** — nothing model-weight- or data-value-dependent.

### Why it's engine-synthesizable at fixed capacity (no export change)
1. **The mask subgraph is a self-contained island.** `attention_mask` feeds only `Shape/Unsqueeze/CumSum`; `Unsqueeze_18` is consumed **only** by the 27 Attention nodes (verified). Nothing else depends on it → the engine can bypass/prune it from the captured region and substitute an on-device fixed-capacity equivalent with zero side effects.
2. **The raw `attention_mask` input is already fixed-capacity** on device (`[1, max_len]`, logical grows; native_decode.rs:1681-1688). The padding info is already there.
3. **The StandardAttentionKernel already applies causal masking and bounds the attention to a valid length internally** (attention_row loops `[0,total_seq)` + pad_limits) — it does NOT fundamentally need the precomputed `[B,1,cur,total]` bias for the standard causal+pad case; that bias is redundant with the kernel's own logic. So the kernel can synthesize the mask from `(valid_len, past_len)` + the fixed-cap attention_mask, and stop consuming the growing graph tensor.

### The pivotal prerequisite (and an honest caveat on my current (a))
My (a) as landed derives the **valid length from the mask TENSOR SHAPE** (`total_seq = mask.dims[3]`, host-side metadata read each step). **That is capture-unsafe**: under capture the shape is frozen at capture-time, so the replayed graph would use a stale length. For capture, (a) MUST become a true **on-device valid-length scalar** the kernel reads from device memory (updated per step by a tiny device-side increment or a single H2D of one int), mirroring how GQA gets `seqlens_k` on-device. This is THE foundational change; capture needs it regardless of the mask. (For the current eager deliverable the shape-derived length is correct and sufficient — capture is off.)

### The in-engine work for (c)+(d) (all executor/kernel, no Mobius)
1. Device valid-length scalar ABI = (a) done right (device scalar, not mask-shape) — foundational.
2. Kernel-internal causal+pad mask synthesis from that scalar + the fixed-cap attention_mask buffer (reuses existing attention_row causal logic; drop the growing-mask dependency). [Option B — cheaper than materializing a fixed-cap bias tensor.]
3. Executor graph-transform: prune/bypass the `Unsqueeze_18` mask island from the captured region so there is no data-dependent intermediate shape.
4. Replace per-step D2H (`nonpad_kv_seqlen`) + htod control arrays with the device scalar; make any copy-back async/stream-ordered (reuse the 24531c4 pattern).
5. Flip `StandardAttentionKernel::capture_support()` -> Supported.

### Where export WOULD be needed (not our case)
Only exotic mask semantics the kernel can't reconstruct from a valid-length scalar — ALiBi / sliding-window / custom learned bias / bidirectional segments. DeepSeek-V2-Lite is plain causal+pad, and the cumsum construction already handles left-padding, so even batched/left-padded serving stays engine-synthesizable (per-batch valid-length from an on-device cumsum — a small fixed-capacity vector, still no export change).

**Bottom line for sequencing:** the 2.4x capture prize is in-engine reachable. Recommend the follow-up land in this order: (1) device valid-length scalar ABI (replaces my shape-derived length), (2) kernel-side mask synthesis + prune Unsqueeze_18 island, (3) device-side control (kill per-step D2H/htod), (4) capture_support flip. No Mobius dependency.
