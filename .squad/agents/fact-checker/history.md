# Fact Checker — History (compacted 2026-07-29)

**Role:** Verifies onnx-genai design, implementation, documentation, and performance claims against authoritative sources and reproducible runs. Owns canonical corrections and caveats so claims ship with exact semantics, scope, and measurement conditions.

## Durable lessons
- EPContext `ep.context_embed_mode` runtime default is `0`, not `1`; keep ORT option defaults exact.
- KV insertion facts: ORT GQA shared-buffer use is sanctioned, standard ONNX Attention has cache semantics, and HF calls `cache.update()` inside attention.
- Projection fusion: QKV is already packed; the available target is 24 gate/up `4864|4864→9728` pairs; 124.6875 MiB fused B+scale payload is only a lower bound because alignment copies may add RSS.
- Native CUDA decode design required a real non-null stream and serialized ownership of non-Send/Sync CUDA graphs; virtual-dispatch cost remains unmeasured.
- CLI research found top gaps in model acquisition, conversion/quant/fine-tuning, and benchmark/batch commands; strongest counter-argument is that CLI polish can distract from CUDA/perf/model enablement.
- Native CPU EP's FP16 decode win reproduces only under quiet-host or forced/persistent-pool conditions and is decode-only; loaded auto-calibration can pick the flat path and make the headline false.
- SPMD auto-calibrator re-probing mid-generation causes FP non-associativity and token divergence around ~459 tokens; forced pool and forced flat are individually deterministic.

## Recent work (current wave, ~2026-07-28/29)
### 2026-07-27T09:17:00Z — Win verification: "Native CPU EP beats ORT by 1.27×"
- **Claim:** Iran reports native FP16 at 57.5 tok/s = 1.27× ORT's best (45.0 FP32). PR #227 headline.
- **Initial result: ❌ OVERSTATED — cannot reproduce.** Same harness/model/prompt/flags yielded native FP16 median 36.1 tok/s vs ORT FP32 45.7 tok/s; native/ORT ratio 0.79× decode, 0.31× end-to-end.
- FP16 GEMV path was active; output coherent at 100 tokens; 500-token nondeterminism pointed to SPMD auto-calibration.

### 2026-07-27T10:55:00Z — Re-verification: calibrator hypothesis confirmed
- **Result: ✅ TRUE-WITH-CAVEATS.** Quiet-machine rerun reproduced native FP16 58.69 tok/s vs ORT FP32 45.76 tok/s, a 1.28× decode-only ratio.
- 3-way experiment confirmed forced-pool=60.20, auto(quiet)=58.69, forced-flat=43.78, auto(loaded)=24.56 tok/s.
- Caveats remain: decode-only (end-to-end 0.42× ORT), TTFT 10× worse, quiet host or `PERSISTENT_POOL=1` required, Qwen 0.5B only.
- Updated `.squad/decisions/inbox/fact-checker-win-verification.md` with the re-verification section.

### 2026-08-11T16:15:00Z — Fifth review of PR #762 (commit `38625fb38`)
- **Claims verified:** B1 absent_outputs unforgeable (✅), rank preserved (⚠️ acceptable), fails closed (✅), optional outputs claimed with fallback disabled (✅, ran 27 tests), CUDA docs honest (✅).
- **Claim contradicted:** "ORT 1.27 has no per-node provider attribution" (❌). `Session_GetEpGraphAssignmentInfo` exists since ORT 1.24; `Node_GetEpName` since 1.23. Both in ORT 1.27 headers.
- **Substantive:** S1 — `conformance_mixed_partition` should use the existing ORT API or correct its comment.
- **Devil's Advocate:** `unwrap_or(0)` convention for dynamic dims is load-bearing but enforced only by comment, not by type system. New kernels could silently allocate zero bytes.
- **Verdict:** Not ready to leave draft until S1 is resolved (~1 hour of work).
- Output: `.squad/decisions/inbox/fact-checker-fifth-review-762.md`

Full pre-compaction history in `history-archive.md`.

## 2026-08-11 — PR #762 fifth review: API existence check

**Task:** Fifth review of PR #762. Verify factual claims.

**Key finding:** Contradicted the claim that ORT 1.27 lacks per-node provider attribution. `Session_GetEpGraphAssignmentInfo` has existed since ORT 1.24. Two prior deferrals (Freysa, Coco) cited a non-existent API gap. API present in the generated bindings; confirmed by examining ORT header and generated Rust bindings.

**Outcome:** Resch wired the API in; 8 conformance tests now assert specific op assignments to `"cpu_ep"`.

**Lesson reinforced:** Verify an API's absence before deferring on it. Check the generated bindings.

## 2026-08-13 — Lower-bit quant accuracy reality-check (parallel to Sebastian's #885 probe)
Independent accuracy-lens check on sub-4-bit for the 30B dense model (read-only, no kernels).
**Findings:** int3 (~3.5 bpw imatrix/AWQ) + SpQR-style mixed = 🟢 credible (small tax); int2 scalar
/Q2_K = 🔴 accuracy cliff; int2 via codebook/trellis (QuIP#/AQLM/QTIP) = 🟡 but adds LUT/trellis
decode that **spends the bandwidth win back** (accuracy & bandwidth coupled); 2:4 sparsity needs a
fine-tune + no M=1 HW benefit. **Load-bearing blocker:** we only HAVE int4 — EVERY sub-4-bit path
must re-quantize from the **fp16/bf16 SOURCE** (re-squeezing int4 → collapse); ORT-stack tooling
for sub-4-bit >7B is immature. Combined with Sebastian's measured byte-fold probe (−75% DRAM →
+2.8%), lower-bit quant is a **MEASURED 🟥 NO-GO** and the ceiling is **latency-bound on the ~2568-
node serial chain**, not weight-bandwidth-bound. Chew is the numerics gate if any path is funded.
