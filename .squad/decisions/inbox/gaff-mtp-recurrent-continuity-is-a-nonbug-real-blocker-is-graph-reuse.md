### 2026-08-21: MTP recurrent-state "continuity bug" does not exist — MTP is already token-identical; the real blocker is zero CUDA-graph reuse

**By:** Gaff

**What:**
Investigated the prescribed fix — "give the main-exec Scan child-session correct
non-zero recurrent-state continuity" — for the native CUDA MTP self-spec decode
divergence on the real Qwen3.8-27B int4 hybrid (GDN+GQA). Conclusion, with
airtight CPU+GPU evidence:

1. **The prescribed bug does not exist.** The real artifact
   (`/home/justinchu/qwen38-27b-int4-mtp-cuda/model.onnx`) has **zero `Scan`
   nodes**. Its GDN recurrence is expressed as ordinary custom ops
   (`CausalConvWithState`, `LinearAttention`) with top-level `past/present.N.*`
   state I/O; the 16 GQA layers use the default-domain `Attention` op with in-op
   KV cache. The decode-inline Scan sibling never engages for this model — both
   greedy AND the m>1 verify run on the **same main executor**.

2. **The m>1 verify forward is already correct.** Using a clean, MTP-free oracle
   (`verify_logits_probe`: replay greedy, then re-run the same positions through
   `decode_verify` with the true greedy continuation as the draft), the M=K
   eager verify is **argmax-identical to M=1 greedy** on **both CPU and CUDA**
   (`flips=0`) at the production `num_speculative_tokens=1` (k=1) and at k∈{3,4}.

3. **The prior divergence was a PROBE ARTIFACT.** The probe rewound the
   attention KV before each verify but left the destructive recurrent/conv state
   stranded at its fully-advanced value, so verify ran from a **stale** state.
   Restoring the recurrent state to `S_base` before verify (exactly what the
   speculative driver already does via `snapshot_recurrent_state`) makes the
   divergence vanish entirely. The earlier "main-exec Scan produces wrong logits
   from non-zero recurrent state" root cause (and the `GAFF_DISABLE_INLINE`
   evidence) was a misattribution of this artifact.

4. **MTP E2E is token-identical to greedy.** On the real hybrid (GPU, CUDA-graph
   ON, `fallbacks=0`), MTP-on generation is **token-for-token identical (48/48)**
   to the target's true greedy stream (`NativeDecodeSession` greedy), which also
   matches the probe. The standing exactness rule already holds today.

5. **THE REAL BLOCKER — zero CUDA-graph reuse.** MTP is currently a **slowdown**,
   not a speedup, because every verify step invalidates the captured decode
   graph. Measured A/B on the same build/GPU (see env block below):
   - Greedy (MTP off): **55.91 tok/s**, `cuda_graph replays=645 fallbacks=0`.
   - MTP  (MTP on):  **10.64 tok/s** median (highly variable 1.33–14.60),
     `cuda_graph captures=120 replays=0 invalidations=1071 fallbacks=0`,
     acceptance **84.7%**.
   MTP does correct, token-identical work but **replays=0** ⇒ it pays full graph
   capture cost every step and never replays ⇒ ~5.3× slower than greedy (and
   ~6× below the 62.56 baseline). The path to a real MTP speedup is **CUDA-graph
   retention across the verify/commit rewind** (the dormant `retain_graph_on_rewind`
   / "option (c)" already scaffolded in `rewind_inner`), NOT recurrent-state
   correctness. That is a separate, capture-safety-sensitive workstream.

**Landed in this PR (correctness + tooling, no behavior change to greedy/MTP):**
- `verify_logits_probe`: made it a **valid oracle for recurrent models** —
  snapshot the recurrent state at each base length and restore it before each
  eager verify (mirrors the driver). Without this the probe reports false
  positives on any hybrid GDN target.
- New public, opaque API on `NativeDecodeSession`: `has_recurrent_state_public`,
  `snapshot_recurrent_state_public`, `restore_recurrent_state_public`
  (+ `NativeRecurrentSnapshot`) so out-of-crate diagnostics can restore
  recurrent state around a verify.
- Regression test `native_verify_logits_require_restored_recurrent_state`
  (state-coupled synthetic hybrid fixture, k∈{1,3}): verify from the committed
  recurrent state bit-matches an m=1 greedy step; verify from a **stale** state
  diverges (negative control proving the restore is load-bearing for the emitted
  logits); restoring reinstates bit-identity.
- Corrected the misleading "byte-identical main (Scan child-session)" comment at
  `native_decode/mod.rs` to state the accurate inline-vs-main property and that
  non-`Scan` custom-op GDN hybrids run entirely on the main executor.

**Env block for every tok/s number above:** H200, `CUDA_VISIBLE_DEVICES=1`
(ordinals 2–7 held 107 GB resident by idle peers at 0% util; ord 1 empty/idle —
ordinal differs from the 62.56 baseline's ord 5, which was occupied this
session), batch=1 greedy, 128-token measured steady window, 3 warmups,
median-of-5 (`profile_native --steady --tokens 131 --decode-skip 3 --warmups 3
--runs 5`), model `/home/justinchu/qwen38-27b-int4-mtp-cuda` int4 block-32,
CUDA-graph ON, build `--release --features bench-native,native-cuda,cuda-13000`,
`ORT_ROOT=.ort-cuda-1.28/root` (onnxruntime 1.28 cuda13), branch off origin/main
`81fc0060e`.

**Why:**
The campaign was chasing a recurrent-state correctness bug that isn't there — the
verify path is exact and MTP is already token-identical. Landing the corrected
root cause plus the valid-oracle probe fix and a pinning regression test stops
future sessions from re-deriving the same false positive, and redirects the
speedup effort to the actual bottleneck (CUDA-graph reuse across verify). The
new public snapshot/restore API and the state-coupled fixture make the true
contract testable without the (env-blocked) full E2E artifact.
