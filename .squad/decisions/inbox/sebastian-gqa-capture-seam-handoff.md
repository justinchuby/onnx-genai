### 2026-08-12: CUDA-graph capture ENGAGES for Muse-Glimmer native decode but is neutralized by 52 GQA eager seams — hand off the KV-seq-symbol pin to Leon

**By:** Sebastian

**What:**
With Deckard #848 (shared-buffer/fixed-capacity KV classification) and Batty #850
(native-CUDA embedding component → model loads + decodes end-to-end on
`--pipeline --backend native --ep cuda`) merged to main (29bd8a35), I measured the
capture (Blocker 3) directly. Capture now **engages** but delivers **no speedup**,
and I root-caused why with concrete numbers.

**Measured (H200, CUDA_VISIBLE_DEVICES=0, staged real model, steady decode, 128 tok):**
- Baseline, capture OFF (`ONNX_GENAI_CUDA_GRAPH` unset): **14.52 tok/s**, 68.86 ms/token.
- Capture ON (`ONNX_GENAI_CUDA_GRAPH=1`): **14.58 tok/s**, 68.57 ms/token — statistically identical.
- Capture ON + `ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS=1` (Inc3c embeds path): **14.51 tok/s** — no change.
- Parity preserved across all three: greedy ids `[24, 372, 1045, 10016, 328, 2885, 262, 5091, 8811, 511, 917, 4921, 768, 328, 2885, 262, ...]`.
- GPU utilization during steady decode: **0–2%, ~127W** (near-idle) — decode is
  definitively **dispatch/host-bound**, NOT GPU-bound, confirming the prior diagnosis
  on the pipeline path.

**Why capture doesn't help (root cause, file:line):**
`ONNX_GENAI_LOG_CAPTURE_SEGMENTS=1` shows the captured decode step is fragmented
into **54 captured segments separated by 53 eager seams**: all **52
GroupQueryAttention nodes** (one per layer) plus **1 SkipSimplifiedLayerNormalization**.
Every GQA node is force-declined by the capture classifier's CENTRAL HARD VETO
(`crates/onnx-runtime-session/src/executor/capture.rs:592-631`):

> "capture classifier disqualified this node: an input or output shape depends on a
> growing (KV/total-sequence-length) symbol, so capturing it would replay a stale
> launch grid — forced eager seam"

The seed comes from `compute_capture_growing_symbols` /
`collect_structural_growing_symbols`
(`crates/onnx-runtime-session/src/executor/kernel_cache.rs:186-260`): the decoder
graph's GQA present/past KV boundary tensors are rank-4 `[batch, kv_heads,
present_sequence, head_dim]` with a **symbolic (growing) penultimate seq axis**, so
the symbol is seeded growing and every GQA node stays eager — **even though the
runtime binds fixed-capacity, device-resident KV** (`DecodeCudaState`, physical
`[1, kv_heads, max_len, head_dim]`; `native_decode/cuda.rs`), and the GQA CUDA
kernel HAS a capture-safe fixed-capacity device-length path
(`crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs:1977` "Record the
fixed-capacity decode signature as capture-safe once a single-token step has run
through the device-length workspace path"; `capture_support()` at 2575).

So the veto is a **false-positive for the fixed-capacity device-length GQA path**:
the launch grid is capacity-sized (bounded by the runtime seqlens/device-length
input), not growing-seq-sized, so a captured replay is actually correct — but the
static-graph classifier can't see that because the IR KV seq dim is left symbolic.

The empirical impact: collapsing ~1600 launches/token → ~106 (54 segment replays +
52 eager GQA launches) yields **zero** improvement, because the 53 eager seams
re-serialize the step (segment-by-segment replay in
`crates/onnx-runtime-ep-cuda/src/runtime.rs:466-474`, GQA nodes eager between
segments). The per-step wall (~68 ms) is dominated by the 52–53 mandatory eager
GQA seams, not by the launch count the captured segments already collapse.

**The fix (Leon's KV & Buffers domain — I am handing off, not grinding, per the
coordinator's "hard KV wall → flag Leon" instruction):**
Pin the decoder graph's GQA present/past KV **sequence symbol to the fixed
`max_len` capacity** on the shared-buffer/fixed-capacity native decode path, before
the session builds its capture classifier, so `collect_structural_growing_symbols`
no longer seeds it as growing and the classifier admits the 52 GQA nodes. That
collapses the step to a **single captured full-graph → one replay/token → ~1
sync/token**, which on a 0-2% idle GPU should close the 14.5→40+ tok/s gap in one
move. This is deliberately KV-geometry + capture-classifier + IR-shape-inference
surgery with a real corruption-risk surface (the classifier's hard veto exists
precisely to prevent a stale-grid replay), so it must be done by the KV & Buffers
owner with parity + growth/rebucket regression coverage:
1. On the fixed-capacity path, rewrite/pin the decoder graph's `past_key_values.*`
   (and, via GQA share-buffer present=past aliasing, `present.*`) penultimate axis
   to the concrete `max_len` so shape inference resolves it to a constant, not a
   growing symbol. `DecodeCudaState::capacity.max_len` is the value; the seq symbol
   is minted per `kv_growing_symbol` on the rank-4 KV boundary.
2. Prove the GQA kernel's captured launch grid is capacity-sized for every replayed
   shape (it is, on the device-length workspace path), and that KV growth/rebucket
   still invalidates+recaptures correctly (the `growth_keeps`/invalidation machinery
   in `native_decode/cuda.rs`).
3. Keep byte-exact parity against the ids above and the ORT backend.
Secondary: the 1 `SkipSimplifiedLayerNormalization` eager seam ("shape/dtype
signature does not match the warmed single-group capture signature") — a minor
single-node warmup-signature mismatch, worth a look once GQA is captured.

**EXACT PIN SITE (found — makes Leon's target narrow):**
The runtime already has fixed-capacity present-shape pinning that makes KV
capture-stable, but it is **gated to the default-domain `Attention` op and
explicitly excludes `com.microsoft::GroupQueryAttention`**:
`crates/onnx-runtime-session/src/executor/dispatch.rs:764` —
`if node.is_default_domain() && node.op_type == "Attention" { ... widen present
K/V outputs (slots 1..) to the binding's physical capacity ... }`. The block's own
comment says it is "mirroring GroupQueryAttention, whose present rule takes
`past_capacity.max(total)`" — i.e. GQA is *assumed* already capacity-shaped, yet
the **build-time** capture classifier
(`compute_capture_disqualifying_symbols`, `executor/build.rs:478`, seeding via
`collect_structural_growing_symbols`, `executor/kernel_cache.rs:186-260`) still
seeds the GQA present/past seq symbol as growing from the *static* graph shape and
force-declines every GQA node (`executor/capture.rs:614`). So the two candidate
fixes for Leon, in order of preference:
1. At session/executor build time, pin the GQA `past_key_values.*`/`present.*`
   penultimate axis to the fixed `max_len` capacity (as the default-domain
   Attention path already does at runtime) so
   `collect_structural_growing_symbols` sees a constant and does NOT seed the
   growing symbol — the classifier then admits GQA. This is the clean parallel of
   the existing default-`Attention` treatment, extended to
   `com.microsoft::GroupQueryAttention`.
2. Alternatively, teach `node_capture_seq_independent` /
   `compute_capture_disqualifying_symbols` that a GQA node whose KV is bound
   fixed-capacity (physical == capacity, valid length via device seqlens) is
   seq-independent for launch-grid purposes — but this weakens the hard veto and
   is riskier, so (1) is preferred.
The default-`Attention` precedent at dispatch.rs:755-800 is the working template;
the corruption-risk reasoning (present = `past_capacity.max(total)`, valid length
on-device, overflow caught by `total_len > max_len`) is already spelled out there
and applies identically to GQA.

**Why:**
This is the last blocker between the merged load+classify prerequisites and the
40+ tok/s target, and it is squarely KV-geometry (pin the fixed-capacity KV seq
symbol so the capture classifier stops force-declining GQA). Getting it wrong is
silent decode corruption, so it belongs to the KV & Buffers owner (Leon), not a
blind perf-side patch. My contribution here is the airtight measured diagnosis +
the exact pin site; I'll re-measure captures/segments/tok/s the moment Leon's pin
lands.
