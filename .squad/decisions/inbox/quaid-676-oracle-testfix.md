# Decision: fix unsound teacher-forced sub-assertion in 35B-A3B QMoE oracle test (#676 follow-up)

**Author:** Quaid (EP/runtime + numerics)
**Date:** 2026-08-06
**Branch:** squad/fix-676-oracle-test (PR against main)
**Artifact:** `/home/justinchu/qwen36-35b-a3b-qmoe-artifacts` (QMoE), `/home/justinchu/qwen36-35b-a3b-artifacts` (dense), `/home/justinchu/qwen36-35b-a3b-fp32-oracle` (fp32 oracle)
**Follows:** #676 (oracle regression test, author cohaagen — under reviewer lockout for this artifact), reviewer Harry's finding, #684 (parallel QMoE kernel)
**File:** `crates/onnx-genai-engine/tests/qwen36_35b_a3b_qmoe_divergence.rs`

## Symptom
`qwen36_35b_a3b_qmoe_native_cuda_matches_fp32_oracle` panics at `:215`
(`QMoE teacher-forced argmax`): expected `33803`, got `279`. Harry reproduced
`279` byte-identically on both the serial and the parallel (#684) QMoE kernels,
proving it is not a kernel bug. Reproduced here on `origin/main` (33.9 s).

## Root cause (verified on the real 35B model, CUDA GPU0)
The teacher-forced sub-assertion (step 2) ran on the **same engine instance**
that had just autoregressively decoded 120 tokens (`greedy_stream`, step 1). For
this **hybrid Mamba** model, reusing that engine serves the teacher-forced step
from the prefix/decode caches, which restore attention KV but **not** the
conv/recurrent (`Mamba`) state. The step therefore predicts from a corrupted
state and argmaxes an unrelated token (`279`); `33803` drops out of the top-k
entirely.

Decisive experiment (diagnostic harness, since removed):
- **reused** engine (post-`greedy_stream`) teacher-forced → argmax `279`.
- **fresh** engine, same 120-token context → argmax **`33803`**, runner-up
  **`5342`**, `logit(33803) − logit(5342) = +0.094` (in `MARGIN_BAND 0.04..=0.14`).

The `+0.094` fresh-engine margin matches the module doc's QMoE row (`+0.0938`)
exactly — **the doc table was correct**; only the test's engine handling was
wrong. Independently, teacher-forcing the **dense** graph on a fresh engine also
re-derives `33803`/`5342` at `+0.078` (matches the doc's dense int4 CUDA row),
confirming `33803` is the fp32-correct next token for this context.

## Fix (minimal, surgical)
Drop the autoregressive QMoE engine after reconstructing the shared context, then
run the teacher-forced step on a **fresh** QMoE engine. No magic constants added;
every retained assertion is the pre-existing oracle-backed one. Added a module-doc
section + inline comment documenting that teacher-forcing a hybrid model must use
a fresh engine. The fp32-oracle cross-check (step 4) and dense autoregressive
divergence (step 3) already used fresh engines and are unchanged.

Retained invariants (all oracle-backed):
1. QMoE **autoregressive** `[119] == 33803` (adjudicated fp32-correct token).
2. QMoE **teacher-forced (fresh engine)** argmax `== 33803`, runner-up `5342`,
   margin in band.
3. Dense autoregressive `[119] == 5342`, agrees before divergence, `!= 33803`.
4. fp32 oracle (CPU, fresh engine) teacher-forced `== 33803` in band (optional,
   env-guarded).

## Verification
- Reproduced original failure at `:215` (279 vs 33803) on `origin/main`.
- Fresh-vs-reused diagnostic proved reused→279, fresh→33803 (+0.094).
- Dense teacher-forced (fresh) → 33803/5342 (+0.078); QMoE fresh → 33803/5342
  (+0.094) — both in `MARGIN_BAND`.
- Env-absent path: `test result: ok` (skips cleanly).
- `--no-run` compiles with no warnings; `cargo fmt --all --check` clean.
- Full `--ignored` run with QMoE+dense dirs: assertions individually confirmed
  passing (step 1 passes on origin/main; step 2 fresh-engine value confirmed;
  step 3 dense assertions confirmed). Full green run is slow due to
  host-offloaded weights streaming the dense autoregressive decode.

## Note / possible follow-up (out of scope)
Teacher-forcing a hybrid Mamba model on an engine that has already generated
returns logits from a state that omits the conv/recurrent component (prefix cache
restores attention KV only). This is fine for the test (fresh engines), but the
prefix/decode-cache reuse producing wrong next-token logits for hybrid models in
a continuation scenario may warrant a separate engine-correctness look.
