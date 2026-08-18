# Decisions — live standing directives

Last consolidated: 2026-08-18T04:15Z (Scribe processed #1189 / DeepSeek Native-vs-ORT follow-ups; archived detailed 2026-08 narrative to `.squad/decisions-archive/2026-08.md` to keep the live ledger below 20KB.)

Standing governance rules and active directives. Full narrative is archived; keep this file to current decisions plus durable rules.

## Ledger health rule

Archive by SIZE, not age. Age-only archiving can silently no-op during high-volume campaigns because most entries are recent. When the live ledger crosses the spawn-budget gate, preserve full history in an archive and keep `decisions.md` to standing directives, active decisions, and pointers. Assemble from inbox drops, dedupe, then delete merged drops; leave `decisions/inbox/README.md`.

## Active historical pointers

For detailed per-PR narrative, use archives rather than expanding this live file. Primary locations: `.squad/decisions-archive/2026-07.md`, `.squad/decisions-archive/2026-08.md`, and `.squad/decisions/archive/`. The detailed 2026-08 decode-vs-ORT and graph-capture campaign narrative, including the pre-#1189 live ledger and full processed inbox drops from this batch, is preserved in `.squad/decisions-archive/2026-08.md` under `2026-08-18T04:15Z`.

## Current decode campaign standing

Native int4 decode leads stock ORT CUDA EP in production because native owns full-decode CUDA-graph capture and device-resident sampling on dynamic-KV int4 paths that ORT cannot capture. Equalizable eager-vs-eager dense-model results show ORT kernels are comparable or sometimes faster; do not frame the dense wins as intrinsic per-kernel superiority.

For DeepSeek-V2-Lite int4 QMoE the finding is stronger and different: stock ORT CUDA EP cannot place `com.microsoft::QMoE` on GPU, so its run falls back through CPU EP for 26 MoE layers. Report this as a GPU-vs-CPU-fallback capability gap, not a per-kernel multiplier.

Batch-1 byte-identical single-kernel/fusion work is mined out for now. Further wins should come from structural capabilities (capture, device token loop, higher arithmetic intensity, model support) or explicitly default-off experimental levers.

## Native-vs-ORT fairness rule

Native-vs-ORT claims must compare the same artifact, quantization, accuracy level, and steady-state methodology with oracle-correct output. If one engine crashes, rejects the graph, runs CPU, disables CUDA graphs, or uses a different weight file/config, report a capability/config gap rather than a throughput multiplier. For ORT-genai decode, verify CUDA provider and share-buffer/cuda-graph fast path are active before quoting tok/s.

## Benchmark and profiling discipline

Separate measured, estimated, and projected. Same-run PR-vs-base deltas beat absolute numbers under shared-host load. For CUDA-graph decode, `ONNX_GENAI_PROFILE_OPS=1` is a host/eager dispatch view and can mis-rank kernels; use `nsys --cuda-graph-trace=node` for kernel mix and `ncu --graph-profiling node --set full` for stall mechanism. A SIMD/accelerated path without a reachability test is equivalent to an unwired placeholder.

## Numerics and portability discipline

Default-on CUDA decode optimizations must be portable or explicitly arch-gated with byte-identical fallback. Token byte-identity is an argmax stability claim, not a numeric invariant; numeric changes need oracle/tolerance justification. Preserve Rule 11: unsupported devices must fall back without behavior loss. Env knobs used for A/B must be documented, deterministic under capture, and not hide default regressions.

For int4 GEMV/QMoE reductions, CPU bit-identity is not an oracle when accumulation order differs. Correctness is bounded agreement with an independent higher-precision reference plus deterministic backend output and explicit golden rationale.

## Testing and CI standing directives

- `cargo test --workspace` silently truncates on failure; use `--no-fail-fast` for full-suite evidence.
- Run new tests in isolation before trusting full-suite green. Assert on what code did, not summaries.
- An agent self-report is not evidence; verify with code, command output, and tests.
- Reviewer lockout is enforced: authors do not revise their own rejected artifacts.
- CI is asynchronous; required local targeted tests/builds/hardware probes remain blocking, but do not idle solely waiting for CI.
- Never commit `.squad/` files to external repos; if that happens, purge history rather than only deleting in a follow-up commit.

## CUDA availability directive

The primary Windows development box has a working RTX 4060 CUDA path even though `nvcc`, `CUDA_PATH`, and default PATH probes may fail. A complete CUDA 13 runtime is available under anaconda site-packages; agents must distinguish absent from misconfigured before claiming CUDA is unavailable. On that box, add the `cu13` and `cudnn` bin directories to PATH and build with `--features native-cuda`.

## 2026-08-18 — V2-Lite MoE CUDA graph-capture and workspace fixes merged

PR #1181 landed on `main` as `c9c7f64c`, unlocking V2-Lite graph capture by fixing the additive-mask `_d1` workspace-planner path; Wallace measured capture ON vs eager OFF byte-identical over 320 tokens, **101.80 vs 56.94 tok/s = 1.79×**.

A separate long-context Engine `Attention` workspace under-plan then surfaced around KV-capacity growth. PR #1189 landed on `main` as `b416a3e0`, fixing Engine/native CUDA single-token decode to re-run governed workspace preparation whenever `ensure_capacity` grows the KV/mask bucket. Leon's A/B on the real V2-Lite path generated 340 token-identical tokens in eager and capture; eager measured 47.32 tok/s, capture 89.69 tok/s with captures=2, replays=336, fallbacks=0. Rachael approved the fix as strictly gated on capacity growth and correctly placed before eager/capture execution.

## 2026-08-18 — DeepSeek-V2-Lite Native-vs-ORT row closed

Wallace measured the real 27-layer DeepSeek-V2-Lite int4 QMoE export under pinned ORT CUDA 1.27 and identical base-decode conditions. Native CUDA serves the model on GPU at **57.15 tok/s eager** and **101.68 tok/s captured**. Stock ORT CUDA EP cannot run the 26 `com.microsoft::QMoE` nodes on GPU; with CPU fallback it inserts 104 host/device Memcpy nodes and reaches **0.17 tok/s**, while strict no-fallback refuses the graph. ORT + CUDA graph is categorically N/A because the graph is split across CPU and GPU. Frame the row as a hard capability gap: native is the only measured GPU engine for int4 QMoE here.

## 2026-08-18 — GLM-4-9B int4 graph-capture scope

# Wallace — GLM-4-9B int4: does the graph-capture moat already extend? → 🟢 YES (captures clean today)

- Author: Wallace (inference-engine specialist)
- Date: 2026-08-18T04:03Z
- Worktree: fresh detached off `origin/main` @ `b416a3e0` (incl. #1171 classifier, #1181 `_d1`, #1189 Engine long-context fix).
- Model: real GLM-4-9B int4 (dense GQA, partial-RoPE) —
  `/home/justinchu/glm-e2e-artifacts/cohaagen-glm-4-9b-int4-cuda-post434` (`model.onnx` + 6.7 GB `.onnx.data`).
- GPU: H200, `CUDA_VISIBLE_DEVICES=6` (nvidia-smi first — all 8 idle 0 MiB/0%; quiet box).
- Base decode, single-stream greedy, `ONNX_GENAI_ONGPU_ARGMAX=1` both arms. NOT spec-decode.

## VERDICT: 🟢 GO — GLM-4-9B ALREADY captures cleanly, no code change needed
The graph-capture stack we landed for V2-Lite (classifier #1171 + `_d1` #1181 + long-context #1189)
**already covers GLM-4-9B today.** Capture engages with **no classifier/planner bail**, is
**byte-identical** to eager, and delivers **1.64×**.

## Capture engagement (question 1)
`ONNX_GENAI_CUDA_GRAPH=1` → capture ENGAGES, no decline/fallback:
```
cuda_graph: enabled=true captures=3 replays=185 fallbacks=0 invalidations=3
cuda_graph_measured:            captures=2 replays=124 fallbacks=0 invalidations=2
```
- `captures>0`, `replays>0`, `fallbacks=0` → the classifier's capacity-form gate accepts GLM's
  attention-mask topology (present-KV inputs present; mask does not escape as a graph output), and the
  workspace planner does NOT under-plan (no prepare-only refusal). No bail to root-cause.
- (The `invalidations=3` are the expected steady-state recaptures as the KV/seq extent grows during
  warmup; they settle — `fallbacks=0` throughout, and the measured window shows stable replays.)

## A/B: byte-identity + tok/s (question 2)
**Byte-identity: 0.000% divergence** — capture-ON vs eager token streams compared position-by-position
over **256 generated tokens**, identical (0/256). (Repo also carries an independent golden lock
`crates/onnx-genai-engine/tests/glm4_9b_decode_lock.rs` asserting native-CUDA == golden greedy.)

tok/s, medians of 5 interleaved rounds, 128 tok, on-GPU argmax both arms:
| arm | median tok/s | range | pstdev |
|---|---:|---:|---:|
| capture ON (`CUDA_GRAPH=1`) | **211.74** | 211.57–212.48 | 0.33 |
| eager (`CUDA_GRAPH=0`) | **128.82** | 128.24–129.05 | 0.29 |

- **ratio = 1.64× (capture/eager)**
- zero-arm-overlap confirmed: min(capON)=211.57 > max(eager)=129.05 → clean separation.

## Notes / honest caveats
- **Eager baseline moved up.** The task's quoted 98.2 tok/s GLM eager is an older figure; on
  `b416a3e0` (post #1189, on-GPU argmax) eager is **128.8 tok/s** and capture is **211.7 tok/s** on
  this H200. The 1.64× lever is the durable number; both absolute figures are higher than the prior
  baseline (later kernels + argmax + this GPU). Reported honestly rather than reconciled to the old value.
- **No bail to scope.** Unlike the V2-Lite `_d1` case (which needed Leon's planner recovery before
  capture engaged), GLM-4-9B requires **nothing** — it is a dense GQA graph whose attention-mask cone
  already terminates at a genuine capacity-form `Attention[3]` with present-KV, so the #1171 classifier
  passes it. No `geometry.rs`/`bindings.rs` change needed; Leon/Deckard have nothing to implement here.
- Base decode, greedy, opt-in flag (`ONNX_GENAI_CUDA_GRAPH`, default-OFF). No spec-decode. No source edits.

## Standing-table implication
GLM-4-9B int4 is native-only (ORT can't load it — dense int4 export ORT rejects, same capability class
as V2-Lite QMoE). The capture moat therefore extends the native-only lead further:
**native GLM-4-9B = 128.8 tok/s eager / 211.7 tok/s captured (1.64×), byte-identical, opt-in.**

## Reproduce
```bash
source /home/justinchu/onnx-genai/.cudaenv.sh
export CUDA_VISIBLE_DEVICES=6 ONNX_GENAI_ONGPU_ARGMAX=1
M=/home/justinchu/glm-e2e-artifacts/cohaagen-glm-4-9b-int4-cuda-post434
ONNX_GENAI_CUDA_GRAPH=1 profile_native --model $M --ep cuda --steady --warmups 1 --runs 3 --tokens 128  # 211.7, captures=3 replays=185 fallbacks=0
ONNX_GENAI_CUDA_GRAPH=0 profile_native --model $M --ep cuda --steady --warmups 1 --runs 3 --tokens 128  # 128.8
# byte-identity: dump generated_token_ids at --tokens 256 for both, diff position-by-position -> 0/256
```

## Constraints honored
Fresh detached worktree off `origin/main b416a3e0`; no `git add -A`; no source edits (measure + scope only);
pinned idle GPU6 after nvidia-smi; base decode only; opt-in flag, no default flip; no /tmp writes; no
stray procs left; GPU returned to idle.

## 2026-08-18 — ORT-fairness dense int4 reconfirmed

Wallace reconfirmed the 2026-08-17 dense int4 Native-vs-ORT decomposition from the opposite direction by trying to enable ORT CUDA graph mode on the same three production exports. True graph-vs-graph is unattainable: Phi-4-mini hard-rejects ORT graph capture because control-flow nodes cannot be supported by CUDAExecutionProvider; qwen2.5-7b aborts at runtime with `ort_value must contain a constructed tensor`; qwen2.5-14b-zp accepts the flag but effectively no-ops because CPU-assigned shape nodes fragment capture (eager 96.9 vs graph 98.7 tok/s, byte-identical).

Eager-vs-eager medians again show this is architectural, not a broad per-kernel native win: Phi native/ORT **0.85×**, qwen2.5-7b **0.77×**, qwen2.5-14b-zp **1.19×**. Keep the deployment headline that native captured decode leads ORT eager **1.33× / 1.14× / 1.83×**, but always label it as CUDA-graph capture plus on-GPU argmax that ORT structurally cannot apply on these dynamic-KV int4 exports.


## 2026-08-18 — Gate-3 speculative verify remains shelved after Marlin

Luv re-probed Deckard's Gate-3 B\* framework on current `main` `923dc592` after Marlin landed. Captured verify break-even `B*=C_verify(M=K)/C_decode(M=1)` is still **NO-GO**: qwen2.5-14b-zp reports **17.5× / 18.4× / 20.0×** for K=2/4/8, and qwen2.5-7b reports **14.9× / 15.7× / 17.4×**. This is worse than the 2026-08-14 baseline (~8.5×) and far above the ≤~2 GO gate, so n-gram/prompt-lookup, EAGLE/MTP, and model-draft speculative-decode work stays shelved.

This updates the earlier spec-decode arc rather than reopening it: the old #957 cheap GQA/SkipSimplifiedLayerNorm residual seams no longer appear as M>1 `KernelCaptureUnsupported` blockers. The measured blocker is now solely `MatMulNBits` at M>1 launching `matmul_nbits_gemm_f16` eagerly; Marlin's capture-safe M>1 int4 GEMM is not selected for this MatMulNBits path. Reconsider only after a graph-safe M>1 MatMulNBits/Marlin path exists and this exact B\* probe is rerun.

## 2026-08-18 — Gate-3 Marlin M>1 opt-in follow-up still NO-GO

Luv re-ran the Gate-3 B\* verify-cost probe with `ONNX_GENAI_MARLIN_M_GT_1=1` and no code changes as a follow-up to the earlier post-Marlin Gate-3 NO-GO. The env gate fixed the capture problem completely: qwen2.5-14b-zp capture segments dropped **96→1**, qwen2.5-7b **29→1**, `KernelCaptureUnsupported` seams disappeared, K=8 byte-identity passed, and the hot path switched to `matmul_nbits_marlin_gemm_f16_splitk`.

The decision does **not** change: speculative decode remains shelved. B\* improved but is still NO-GO at **5.19× / 5.19× / 5.79×** for qwen2.5-14b-zp and **4.64× / 4.71× / 5.23×** for qwen2.5-7b at K=2/4/8, above the ≥~4 kill gate and far above the ≤~2 GO target. The spec-decode family — model-draft, n-gram/prompt-lookup, EAGLE/MTP — is now mined out across the three probes. Residual cost is Marlin M>1 GEMM/repack/reduce (`matmul_nbits_marlin_repack` observed in the hot path), not graph fragmentation.
