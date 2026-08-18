### 2026-08-18: general head-size fused f32 GQA decode kernel (was head_size=256)

**By:** Deckard

**What:** GO. Generalized the f32 fused split-K GQA decode fast path over head size
instead of special-casing 256. The kernel is now templated on dims-per-lane
`DPL = ceil(head_dim / 32)`; the launcher selects the exact tier (1..=8) so each
head keeps its minimal register footprint. `supported()` covers head_dim 1..=256.

- Correctness (byte-identical / f64 CPU-reference oracle), same tolerance as the
  original head<=128 test (max_abs<1e-3, max_rel<5e-3). All GQA 8/2, cache lengths
  spanning the split-K boundaries:
    head64  (dpl2): max_abs=1.19e-7  max_rel=2.49e-7
    head80  (dpl3): max_abs=1.19e-7  max_rel=2.55e-7
    head96  (dpl3): max_abs=1.19e-7  max_rel=2.43e-7
    head112 (dpl4): max_abs=1.79e-7  max_rel=3.58e-7
    head128 (dpl4): max_abs=1.49e-7  max_rel=3.04e-7
    head192 (dpl6): max_abs=1.19e-7  max_rel=2.51e-7
    head256 (dpl8): max_abs=1.79e-7  max_rel=4.15e-7
  (non-multiple-of-32 dims 80/96/112 correctly mask partial lanes.)

- Perf headline (qwen3.5-2b-text, head_dim=256, idle H200, native, tokens=128
  warmups=2 runs=5 --steady --decode-skip 1, medians of 5):
    BEFORE (gqa_attention_reference_f32): 102.31 tok/s (9.774 ms/token)
    AFTER  (fused, dpl8):                 170.97 tok/s (5.849 ms/token)  -> 1.67x
  nsys: gqa_attention_reference_f32 share 31.2% -> 1.1% (only warmup calls remain).

- Regression guard (qwen3-0.6b, head_dim=128): DPL4 baseline 316.06 tok/s vs
  templated dpl4 312-316 tok/s across repeats -> statistically identical (the
  templated dpl4 path compiles to the same code as the pre-change DPL=4 kernel;
  no register regression). A naive single-tier DPL=8 kernel measured 314.69 —
  also within noise on this model, but the per-DPL specialization guarantees no
  regression on attention-bound shapes/longer contexts.

**Why:** head_dim=256 previously fell to the serial reference kernel (nsys #1
decode hotspot, 31.2%). Parameterizing over DPL removes the fallback for the
whole common head-size set (64/80/96/112/128/192/256) with one kernel, keeps
small heads at their original register footprint, and is byte-identical-eligible.

**Scoped follow-up (NOT in this PR): asymmetric v_head_dim != qk_head_dim
(Gemma dual-head / DeepSeek MLA).** Audited: the f32 decode kernel and the entire
`group_query_attention.rs` op thread a single `head_size` for Q/K/V. Standard ONNX
`com.microsoft.GroupQueryAttention` is symmetric, so no runtime model needs this
today. Adding it would require: (1) split `head_size` into `qk_head_size` (query,
key, butterfly dot-product loop) and `v_head_size` (value accumulate `acc[]`,
`warp_acc`, scratch stride, output write) in `gqa_decode.rs`; (2) a second template
param so `q_reg` is sized by `ceil(qk/32)` and `acc`/output by `ceil(v/32)`
(entry matrix grows to DPL_QK x DPL_V); (3) thread a separate `v_head_dim` through
`gqa_decode::run()` and the GQA op call site + KV-cache/RoPE prep. Deferred to keep
this PR scoped; the win here (symmetric heads) covers every model we currently run.
