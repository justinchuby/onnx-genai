# Gaff — MLA unequal-K/V GQA guard review

**Date:** 2026-07-23
**Reviewed commit:** Mobius `00309bece3de6c145e4f87164274994e3f21b0e6` (`feat/glm4-gptq-import`)
**Scope:** Only the MLA-guard commit; unrelated GLM4/GPTQ work was not reviewed.

## Verdict: 🟡 — correct for statically shaped exports, but dynamic/symbolic K or V head dimensions can still fuse into an incompatible GQA node

### Guard logic

`_has_unequal_kv_head_dimensions()` is structural and model-agnostic: both rewrite paths bind the actual K and V inputs, then compare their final dimensions and the final dimensions of `past_key`/`past_value`. It rejects only when a pair has two Python `int` dimensions that differ. The rotary path correctly checks `k_pre` (before RoPE) plus `v`; the fallback checks `k` plus `v`.

For DeepSeek-V2-Lite I independently inspected the exported Attention inputs:

```
k      [batch, sequence_len, 3072]
v      [batch, sequence_len, 2048]
past_k [batch, 16, past_sequence_len, 192]
past_v [batch, 16, past_sequence_len, 128]
```

The guard therefore rejects from both the projected K/V sizes and the cache head sizes. This is not a DeepSeek name special case. Standard equal-head GQA continues to fuse: Qwen3 (28), Phi-3 (32), and GLM-4 (40) each changed from Attention to the same count of GroupQueryAttention in independent builds.

### Finding: symbolic final dimensions under-decline

At `src/mobius/rewrite_rules/_group_query_attention.py:56-66`, any non-`int` final dimension becomes `None`. A symbolic/dynamic K or V head dimension consequently bypasses the mismatch test, so an Attention whose runtime K and V head sizes differ can be rewritten to GroupQueryAttention even though that kernel lacks a distinct `v_head_size` input. This is an under-decline (not a DeepSeek-specific issue).

**Required adjustment:** make fusion conservative for the final K/V dimension: fuse only when equality is proved (equal static integers, or equal symbolic dimension identities/values if the ONNX IR representation establishes equality); otherwise retain standard Attention. Add a focused symbolic-shape regression covering unequal/unknown K/V final dimensions. This preserves ordinary GQA exports because their head dimensions are static and equal.

### Edge cases verified

- **Biased QKV / Phi-style:** Existing biased-QKV regression tests pass; the new helper observes the same K/V values regardless of bias plumbing.
- **Interleaved RoPE / GLM:** Existing GLM regression passes; the guard runs only in `check` and does not change `rotary_interleaved` propagation.
- **Static standard GQA:** Equal dimensions are not declined.

### Validation

- `git diff --check 00309be^ 00309be`: clean.
- `.venv/bin/python -m pytest src/mobius/rewrite_rules/_group_query_attention_test.py`: **23 passed** (19.07s).
- `.venv/bin/python -m ruff check src/mobius/rewrite_rules/_group_query_attention.py`: **All checks passed**.
