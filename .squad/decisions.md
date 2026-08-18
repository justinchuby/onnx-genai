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
