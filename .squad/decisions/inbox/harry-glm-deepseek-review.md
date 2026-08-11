# Review: native CUDA load unblock for GLM-4-9B & DeepSeek-V2-Lite

Date: 2026-08-11
Reviewer: Harry (code reviewer)
PRs: #770 (honor native CUDA KV capacity), #771 (accept Cast-backed QMoE scales)
Context: both OPEN, auto-merge on; reviews are not a merge gate. Genuine correctness audit.

## PR #770 — native KV reservation uses runtime CUDA hard cap first

Verdict: **APPROVE** (correct for the target models) with one recommended hardening follow-up.

### What was verified
1. **Precedence is sound for the enforced ceiling.** `hard_max_len == capacity.max_len`
   (`cuda.rs:2330`). The decode path rejects any request beyond that ceiling at every
   entry point (`total_len > state.capacity.max_len` at cuda.rs:469, 604, 669, 815, 843)
   and KV growth refuses past it (`ensure_capacity`: `if required > self.capacity.max_len`
   at cuda.rs:1056). So reserving at `hard_max_len` cannot admit a sequence the device
   later fails to carry — the original "discover mid-generation" hazard is avoided. The
   env/programmatic caps are themselves clamped by metadata in `resolve_cuda_kv_capacity`
   (cuda.rs:2771+), so `hard_max_len <= metadata max_sequence_length` always.
2. **Fallback path is correct.** When `cuda_kv_debug_stats()` is None (CPU/non-CUDA),
   `native_kv_reservation_max_context` falls back to metadata `max_sequence_length`
   (source "model.max_sequence_length"). Covered by `native_kv_reservation_falls_back_to_metadata`.
3. **No regression for normal declared-context models.** With a declared
   `max_sequence_length` and no env/programmatic cap, `resolve_cuda_kv_capacity` sets
   `hard_max_len == metadata` value, so the new path reserves exactly what the old path
   reserved. Identical behavior. With an env cap it reserves the smaller enforced value —
   strictly safer.

### Finding (LOW severity, narrow, recommended fix — non-blocking for GLM/DeepSeek)
`resolve_cuda_kv_capacity` returns `max_len = usize::MAX`, source
`"unbounded (model.max_sequence_length unavailable)"` when a CUDA model declares **no**
`max_sequence_length` and no env/programmatic cap is set (cuda.rs ~2807-2814). In that case
`cuda_kv_debug_stats()` is `Some`, so the new `native_kv_reservation_max_context` returns
`Some((usize::MAX, "unbounded..."))` and load.rs calls `kv_reservation(usize::MAX)`, which
saturates to `u64::MAX`. Behavior then splits by allocator:
- VMM CUDA path (`commits_on_demand() == true`): only warns, still loads.
- **cuMemAlloc CUDA path (`commits_on_demand() == false`, provider.rs:965 / vmm_allocator.rs:945):**
  `governor.reserve_on(NativeKvCache, Device, u64::MAX)` fails → **load fails**.

Previously this case hit the `None` arm (load.rs:529) which deliberately warns-and-loads
("the model runs fine"). So for an undeclared-context model on the non-VMM CUDA path this
is a genuine load regression.

Recommended guard: treat an unbounded/`usize::MAX` `hard_max_len` as "no cap" and fall
through to metadata (i.e. `native_capacity.filter(|(len, _)| *len != usize::MAX)`), which
reproduces the old warn-and-load behavior. Neither GLM-4-9B nor DeepSeek-V2-Lite hits this
(both declare context and GLM uses `ONNX_GENAI_CUDA_KV_MAX_LEN=4096`), so it does not block
Justin's priority — but it should be fixed before relying on this path for arbitrary models.

## PR #771 — QMoE placement accepts one-hop Cast(initializer) scales

Verdict: **APPROVE**. All three scrutiny points hold.

1. **No dtype/region corruption.** `plan_static_weight_placement` is explicitly advisory
   (placement.rs:22-27 doc): it produces a plan/report (device/host byte counts, explanation)
   and does **not** feed bytes into the QMoE kernel. The runtime kernel still consumes its
   graph inputs — the fp32 Cast output — unchanged. Region classification reads the backing
   fp16 initializer only for dimensions/regions; `blocks_per_row = scale_dims[2]` is a count
   and shape is Cast-invariant (Cast preserves shape), so the layout math is correct. The
   fp16 backing bytes are never handed to the kernel as fp32. Cohaagen's claim is verified.
   Forward-looking note (not a bug today): if/when static placement becomes *enforcing*, the
   classified region points at the fp16 backing tensor while the kernel reads the fp32 Cast
   output — a divergence to track — but the scale tensors are tiny (per-block) so accounting
   impact is negligible.
2. **one-hop / domain guard is right.** `initializer_or_initializer_cast` accepts only a
   default-domain (`node.domain.is_empty()`) `Cast` whose single input is a direct
   initializer. A multi-hop `Cast(Cast(init))` is correctly rejected (the inner value is not
   in `graph.initializers`). A `com.microsoft` Cast is rejected — correct, since Cast is a
   standard-domain op. Not checking the Cast `to` dtype is harmless here because placement
   uses the backing initializer's real dims and the runtime consumes whatever the Cast emits.
3. **DRY.** No model-name gates; generic runtime/placement behavior. Confirmed.

### Validation plausibility
GLM top-40 identical, max logprob delta ~0.00794; DeepSeek top-40 identical (low-rank
reorder only), max delta ~0.02848; both golden-greedy locks pass. Deltas are consistent with
fp accumulation-order differences on the CUDA path, not a numeric defect. Plausible.

## Bottom line
- #771: APPROVE, no changes needed.
- #770: APPROVE for the priority models; recommend guarding the `usize::MAX` unbounded case
  so undeclared-context models on the cuMemAlloc CUDA path don't regress from load to failure.
