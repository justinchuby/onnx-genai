# Luv — History (compacted 2026-08-12)

**Role:** Code reviewer for correctness/safety gates across decode, sampling, KV, concurrency, CPU/CUDA EP coverage, and API contracts. Validate with real exit codes and mutation/guard-break evidence; rejection triggers strict reviewer lockout.

## Durable lessons
- Never approve on style alone: verify builds/tests/fmt/clippy or equivalent real exit codes, and prefer mutation/guard-break proof for regression tests.
- Reviewer lockout facts remain canonical: Batty locked out on #14 vision token expansion; Pris locked out on Bitwise/Hardmax until Deckard revision; Batty locked out on PR #287 backend flag until Deckard revision; Batty locked out on PR #411 round-2 fixture until Leon revision.
- MatMulNBits direct-int4 four/eight-column GEMV tiling regressed from register pressure, spills, and non-contiguous packed-weight streams; one-column GEMV remains production.
- Unique kernel history: unsafe String execution/UB was rejected; final support is numeric/bool/bf16 with String reported unsupported.
- CPU EP f64 matters: f32-only activations and f64 narrowing were rejected; true-f64 correction was required.
- All CUDA EP kernel work must remain correct and fast across supported SM architectures, not only sm_90.
- CUDA coverage #338 moved Pad/Range parity forward, but #67 remains open; ScatterND, quantization, and cuDNN work were deferred.
- CLI sampling/context review invariants: explicit `--max-new-tokens` preserved, per-turn REPL budget recomputed, finite unknown-context fallback, correct context arithmetic, sampling/greedy semantics, non-TTY stability.
- Backend flag contract: reporting must show resolved backend, not requested `auto`; clarify/test whether shared `--backend` on `transcribe` is intentional.
- Run a new test in isolation before believing it; a full parallel-suite green can be a fluke.
- A near-deterministic token stream cannot witness sampling through its tokens; PR #411 token-stream assertion was ~95% false-fail, and stderr timestamps did not prove stdout diversity.
- For sampling-policy regressions, instrument the generation boundary: `--stats`/`--profile` must reflect the policy actually resolved for the turn, not a separate or stale `/session` view.
- A reviewer's "SAFE" is not proof; verify the load-bearing claim independently.
- A reviewer's blocker can be a false positive; verify reviewer claims the same way we verify author claims.

## Historical context (pre-2026-08-12)

Wave 2026-07-28/29: CUDA coverage batch 7 merged (PR #338); PR #411 tiny-reasoning fixture rounds 2 and 3; reviewer lockout held throughout. Full detail in `history-archive.md`.

Wave 2026-08-11: Reviews of PR #31974 (BFloat16 LayerNorm/RMSNorm CPU EP, conditional approve → approve after S1 fix), B4+B6 test rework, PR #762 third review (no blockers, S1 optional-slot vacuity confirmed), PR #31985 (ort-docfix, NITS only), PR #32001 (Apple Accelerate CMake, three substantive, no blockers). Full detail in `history-archive.md`.

## Archive pointer

Older entries in `history-archive.md`.

## 2026-08-12 — Review #31973 v3 (AVX2 LayerNorm blocker fix)

Fresh adversarial review of `nxrt/mlas-avx2-layernorm` @ `72e02cd92c` after the architecture-specific dispatch threshold fix. Key findings:
- **Threshold fix is genuine**, not cosmetic — old test would not have caught the original RVV bug because both production and test shared the same wrong universal-8 assumption. New architecture-specific constant correctly encodes the real contract.
- **Three stale Welford comments** (test_layernorm.cpp:275,646,1055) describe the wrong algorithm after the Welford→centered-two-pass rewrite. Substantive — fix before leaving draft.
- **CatastrophicCancellation honestly fixed** — two new scenarios with condition < 1e7 make the accuracy branch actually execute.
- All five non-blocking fixes landed honestly. No leaks, no performance overclaims.
- **Not ready for draft exit** until stale Welford comments fixed.
- Head reviewed: `72e02cd92c`. Wrote `.squad/decisions/inbox/luv-review-31973-v3.md`.

## 2026-08-17T16:25Z — PR #1134 GEMV prefetch pipeline shipped; gate/up PF=2 no-go

- Delivered `b6a5648c`: PF=2 prefetch-pipelined int4 block-32 scales-fp16 decode GEMV, `ONNX_GENAI_GEMV_PIPELINE` default-ON. Byte-identical by raw-bit unit test and teacher-forced logprob comparison; E2E isolated qwen14b improved about +5.2%.
- Tried gate/up SwiGLU PF=2 at `8735904d`; byte-identical but not a speedup because the fused gate/up kernel is issue-bound with load latency already hidden by warp switching. Kept default-OFF on branch; not merged.
- Next kernel lever is fewer instructions per byte, especially 128-bit vectorized int4 loads for gate/up.

## 2026-08-17T17:10Z — PR #1137 SwiGLU dequant zero-point bias-fold shipped default-ON

- Delivered `e54cae31` (kernel+test) + default-ON flip `3d5888ef`, merged as PR #1137 squash `70cc06ad`. Fused the `-8` symmetric zero-point into the magic-bias-removal constants of `int4x8_to_half2x4_sym8` (bottom `x-1032`=`0x6408`, top `fma x*(1/16) -72`=`0xD480`), removing 8 f16x2 ops/iter: −7.4% issued instructions, kernel time −1.6%; byte-identical.
- Corrected the task hypothesis: "vectorize to `uint4`" is the wrong lever — layout is already 128B-coalesced-per-warp, so a per-lane `uint4` remaps four lanes' nibbles and breaks byte-identity. The kernel is issue-bound; the byte-identical win is fewer ALU ops, not wider loads.
- Honest perf: Wallace independent found 7b +1.4% clean, 14b break-even (my +0.82%/14b did not reproduce — big-model decode is memory-latency-bound). Rachael 🟢 (exact fp16 nibble equivalence).
- Next lever: RMS `__syncthreads` / shared-mem staging latency floor — the reason −7.4% instrs bought only −1.6% time. Fresh worktree off origin/main; byte-identity required.

## 2026-08-17T18:05Z — PR #1139 gate/up SwiGLU occupancy-raise shipped default-ON

- Delivered `11a01fae` (kernel+test) + default-ON flip `cfc5e812`, admin-merged as PR #1139 squash `0636a759`. Added `_vec_occ` siblings of the two symmetric RMS-fused `_vec` gate/up entries whose ONLY source difference is `__launch_bounds__(256, 8)` — caps registers at exactly 32/thread (65536/(256×8), no spill) → 8 blocks/SM = 100% theoretical occupancy. Byte-identical by construction.
- Diagnosis (ncu `--set full`): default `_vec` was latency-bound / warp-starved — 40 regs, 62.2% achieved occ (75% theoretical, register-limited), dominant Short Scoreboard ≈51% on the staged-activation LDS in `permute_activation_f16x8`. Lever = more resident warps, not fewer instrs (#1137). OCC=1: regs 40→32, occ 62.2%→81.7%, DRAM 29.7%→32.9%, isolated kernel 57.5→54.0µs (−6.2%).
- Symmetric-only (`occ = !has_zp && gate_up_occ_enabled()`); asymmetric `_zp` (48–56 regs) excluded (spills at 32). Complementary with #1137: `_vec_occ` = `_vec` (bias-fold) + launch_bounds.
- Perf: 14b +2.4% E2E, Wallace independently confirmed +2.6% (5/5 rounds); 7b flat/no-regression. Byte-identity test PASS; teacher-forced dump-logprobs OCC=1 == OCC=0. Rachael 🟢, Wallace GREEN. Largest single-kernel byte-identical decode win so far.
- Next lever: pre-permute the normalized staged activation ONCE into shared — still Short-Scoreboard-bound at 82% occ (all 8 warps redundantly re-permute the same sequence each K-tile). Dispatched off origin/main incl. `0636a759`; byte-identity-sensitive; fresh worktree.

## 2026-08-17T18:40Z — gate/up pre-permute shelved; redirect to memory-side lever

- `squad/gateup-preperm @ 6629f0aa` proved byte-identical and ncu showed a real isolated kernel win, but Wallace could not reproduce dependable E2E gain (14b +0.27% noisy, 7b flat). Shelved/not merged; next work redirects to cp.async weight-load double-buffering / memory-side attack.

- 2026-08-17T19:05Z — Gate/up cp.async weight staging shelved: byte-identical but 2.3× slower / −36% E2E; redirected to GEMV-family survey with 128-bit vectorized gate/up loads as fallback.

- 2026-08-17T20:35Z — GEMV-family survey completed; down-proj `ONNX_GENAI_DOWN_OCC` proved byte-identical but NO-GO (32-reg cap spilled and occupancy did not convert), branch shelved; pivoted to q/o `_pipe` occupancy lever.

- 2026-08-17T21:05Z — q/o `_pipe` occupancy (`ONNX_GENAI_QO_OCC`) shelved: byte-identical but 40→32 reg granularity spilled/regressed; occupancy vein mined out, pivoted to kernel-fusion scoping.

- 2026-08-17T21:40Z — Fusion-scope survey closed as NO-GO: QKV fusion already exists and reproduced −10.8% on qwen2.5-14b; batch-1 byte-identical decode vein is mined out, pivoting to GLM/DeepSeek CUDA-kernel support scope.
## 2026-08-17T22:20Z — GLM/DeepSeek scope delivered; QMoE Gap 2 underway

- Delivered GLM/DeepSeek scope: GLM-4-9B int4 (98.2 tok/s) and DeepSeek-R1-distill-Qwen-1.5B int4 (690 tok/s) already run native E2E; no GLM/DeepSeek dense-kernel gap.
- Established MLA is not a custom-op gap for current DeepSeek-V2-Lite export; QMoE, CSA, and IndexShare kernels already exist and are oracle-validated.
- Active split: Gap 1 workspace-planner blocker routed to Leon; Gap 2 QMoE expert-GEMV remains Luv's CUDA target under `ONNX_GENAI_QMOE_VEC`.
## 2026-08-18T00:35Z — V2-Lite oracle artifact authored, rebased, and merged

- Authored the oracle correction for DeepSeek-V2-Lite CPU-vs-CUDA divergence: wrong CPU-oracle premise, benign f32 accumulation-order drift, and token-5 top-k near-tie below fp32 resolution.
- Added the f64-bounded QMoE expert-GEMV test and explicit native-CUDA/f64 decode-golden rationale; rebased onto current `main` and confirmed the golden held through #1129's MoE-claiming change.
- PR #1150 merged as squash `e075a715`; resume QMoE optimization (`ONNX_GENAI_QMOE_OCC`/`QMOE_VEC`) on this corrected baseline.

