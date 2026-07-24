# DeepSeek MLA CUDA-graph capture — turnkey implementation blueprint + scoping decision

**Author:** Rachael (worker)
**Base:** `perf/deepseek-mla-capture` reset to stack on `perf/deepseek-mla-fixedcap` @ `53afab0` (correctness fix `621936f` + eager fixedcap).
**Status:** BLUEPRINT + honest scoping decision. **No capture code pushed** — see decision. All blockers verified against source this session.

## Decision (why no partial capture landed)
The ~2.4x prize (Pris: eager 40.5 ms/tok, only 15.5 ms real GPU; ~60% is per-op orchestration) is reachable **purely in-engine** — I re-confirmed the mask island (below) and that the IR has the graph-mutation API to do it. BUT capture is **all-or-nothing**: it only engages when `DeviceBinding::has_dynamic_logical_input_shape()` (tensor.rs:308) is false for EVERY bound input, which requires removing FIVE coupled blockers at once. **Any partial state breaks real-weight decode** (e.g. fixing bindings to fixed-capacity without the device valid-length scalar makes `total_seq = key_past_seq + k_cur.seq` resolve to `max_len` → attention reads padding → garbage/nondeterminism). Per Justin's mandate (real-weight decode MUST stay deterministic + coherent) and the user's explicit "land safely-separable and report, don't force it", I will not push a rushed, unverifiable partial capture stack. This should be built as a dedicated, GPU-iterated, reviewed effort using the blueprint below.

## The five coupled capture blockers (verified file:line, on `53afab0`)
1. **Mask binding grows.** native_decode.rs:1960-1961 `extend_mask` sets `bindings[0]` logical `[1, end]` (physical `[1, max_len]`) → dynamic → capture gate (native_decode.rs:1919-1922) declines.
2. **KV bindings grow.** native_decode.rs:1965-1972 `set_logical_len` grows KV logical axis-2 each step → dynamic. (My fixedcap exposed the *present output* at capacity but left the *input* logical, deliberately, to avoid shape-inference poisoning — see rachael-mla-fixedcap.md.)
3. **Valid length is host/shape-derived.** standard_attention.rs:846 `total_seq = key_past_seq + k_cur.seq` (from the growing past_key input shape) and :785 `key_past_seq = past_key.seq`. Frozen at capture-time → stale on replay. **This is the pivot (a').**
4. **Growing mask island feeds Attention.** Graph subgraph `attention_mask → Shape/Unsqueeze/CumSum → Slice → GreaterOrEqual(causal) → And(key-valid) → Where{0,-65504} → Unsqueeze_18 [B,1,cur,total]` (grows with total). If bindings are fixed-cap without pruning this, `Shape/CumSum(attention_mask)` read `[1,max_len]` and mark padding valid → wrong mask. **This is (c).**
5. **Sync + per-op scratch on the kernel path.** standard_attention.rs:738 entry `synchronize()`, :1250 exit `synchronize()`, :1029 per-call `alloc_raw`/`free_raw` for scores/staging/offsets/pad_limits, and per-step htod of offsets/pad_limits (:1093-1103). All capture-illegal. capture_support() = Unsupported (:1267). **This is (d).**

## Re-confirmed: the mask is a prunable island (both exports)
`attention_mask` consumers = ONLY `{Shape, Unsqueeze, CumSum}`; `v_model.Unsqueeze_18` consumers = ONLY the 27 `Attention` nodes (verified on blk32; blk128 identical export topology). Semantics = standard causal + padding in cumsum space (left-pad robust), a pure function of `(past_len, cur_len, attention_mask[1,max_len])`. Fully reconstructible in-kernel from a device valid-length scalar; **no Mobius/export change**.

## Turnkey plan — stage as 3 reviewable commits

### (a') Device valid-length scalar ABI  [do first; verify eager-correct in isolation]
- **native_decode.rs:** add a fixed-capacity device binding `__attn_valid_len` (int32 `[batch]` or scalar `[1]`), written each step (H2D via `write_bytes`, OUTSIDE any captured region) with the current `total_len` (or `past_len`; pick one convention). Owns + updates alongside `extend_mask`/`set_logical_len`.
- **Graph transform (IR API exists):** `create_named_value` + `add_input` for `__attn_valid_len`; append it as a new optional trailing input to every default-domain `Attention` node via `node_mut`. (Alternative: thread it through the executor's existing default-Attention hook in exec_kernel_node — but a real graph input mirrors GQA's `seqlens_k` and is cleaner for binding.)
- **executor.rs:** bind `__attn_valid_len` at fixed capacity; ensure it is NOT flagged dynamic.
- **standard_attention.rs:** bump `check_arity` max inputs 7→8; read the device scalar pointer (new slot); the CUDA kernels (`build_kv`, `attention_row`) take `const int* valid_len` and compute `total = valid_len[b]` / `past = valid_len[b] - q_seq` **on-device** (mirror GQA group_query_attention.rs:61,80 `total = seqlens_k[b] + 1`). Keep HOST sizing (scratch/grid) from shapes for now (eager) — capacity-fixed sizing lands in (d).
- **Verify eager:** blk32+blk128 3× identical + coherent (8913 ' Paris'); Qwen non-regression; gate ≥205/0. Capture still OFF (bindings still grow). Add a unit test asserting the kernel derives length from the device scalar (feed a scalar != shape extent → output tracks the scalar).

### (c) On-device fixed-capacity mask synthesis + prune the island
- **native_decode.rs:** make mask + KV bindings **fixed-capacity** (logical == physical == max_len; stop growing in `extend_mask`/`set_logical_len`) so blockers (1)(2) clear. Keep `attention_mask[1,max_len]` contents (1s for valid, 0 beyond) for padding.
- **Graph transform:** `remove_nodes` the `Unsqueeze_18` island (Attention nodes drop input 3); DCE reclaims `attention_mask` subgraph. Attention now masks purely from the device scalar (a') + causal, synthesized in `attention_row` (kernel already applies causal via offsets/pad_limits — replace host offsets/pad_limits with on-device derivation from `valid_len`). Keep it GENERAL (any default-domain Attention decode).
- **Verify:** same correctness sweep; capture gate now OPEN (no dynamic bindings) but capture_support still Unsupported → still eager. Confirm determinism/coherence unchanged.

### (d) Flip capture_support + kill sync/scratch
- **standard_attention.rs:** remove entry/exit `synchronize()` (rely on same-stream ordering — reuse the 24531c4 stream-ordered pattern for any copy-back); replace per-call scratch with fixed module-global/persistent arena (mirror gqa_decode.rs:18 "fixed module-global scratch allocated when NVRTC loads"); fixed launch geometry (grid from capacity, internal loop bound from `valid_len`); `capture_support()` → `Supported`.
- **Verify (headline):** DeepSeek MLA CAPTURES: captures>=1, replays=N, fallbacks=0 both exports; tok/s ~25 → target ~57-61 (idle GPU, nvidia-smi first); 3× determinism + coherence; Qwen/Phi still capture + no regression; gate ≥205/0 + a capture-eligibility regression test; clippy --lib clean.

## Environment / verification refs
`source /home/justinchu/onnx-genai/.cudaenv.sh`; idle GPU via nvidia-smi (CPU-perf team shares box); blk32 `/home/justinchu/ds-e2e-artifacts/deepseek-v2-lite-real-int4-blk32/`, blk128 `.../deepseek-v2-lite-real-int4/`, Qwen `/home/justinchu/qwen2.5-0.5b-int4-onnx-native/`. Correct tokens `[8913,13,185,549,19305,280,7239,317,254,28071,13,185]`. GQA capture-safe reference kernel: `crates/onnx-runtime-ep-cuda/src/kernels/gqa_decode.rs` + `group_query_attention.rs` (seqlens_k device pointer, fixed scratch, no sync). Gate: `cargo test -p onnx-runtime-ep-cuda --features cuda --lib` (205/0 on 53afab0).

## Recommendation
Green-lit and in-engine feasible, but it is a substantial coupled change (kernel rewrite + graph transform + binding/ABI + fixed scratch) that must be built with staged GPU verification under the determinism mandate — not rushed. Recommend sequencing it as a dedicated focused effort (my next session or a pairing) using this blueprint. (a') is the separable first commit; (c)+(d) must land together to flip capture.
