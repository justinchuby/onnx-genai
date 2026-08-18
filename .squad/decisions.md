# Decisions — live standing directives

Last consolidated: 2026-08-18T16:44Z (Scribe processed Tycho #1180 CUDA lib-name inbox drop; prior detailed 2026-08 narrative remains archived in `.squad/decisions-archive/2026-08.md` to keep the live ledger below 20KB.)

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

### 2026-08-18: #1180 — single CUDA lib-name table
**By:** Tycho
**What:** Moved the CUDA runtime (`cudart`) shared-library candidate names into one canonical, per-platform table in `onnx-genai-cuda-version-guard` (`CUDART_CANDIDATES_LINUX/MACOS/WINDOWS`, `HostOs`, `cudart_candidates_for`, `cudart_candidates`). Both loaders now read it: `onnx-genai-ort::cuda_rt` (deleted its private `CUDART_CANDIDATES` const) and `onnx-runtime-ep-cuda::dynamic_library` (its `Runtime` match arm delegates to the guard). Deleted the #1178 agreement test `every_cuda_major_the_ep_loads_is_also_resolvable_here` and repointed the EP's `generates_linux/windows_cuda_names` tests at the canonical constants. Added `onnx-genai-cuda-version-guard` as a normal (not just build) dependency of both crates.
**Why:** The two lists had drifted (EP had CUDA 13, `cuda_rt` did not), which made CUDA 13 hosts look CUDA-less (#1178). A test asserting two tables agree is weaker than one table — this makes the duplicate copy unrepresentable (same shape as #1170's prefix-reuse fix). The canonical table preserves the exact union of both prior lists (incl. CUDA 13 and legacy `cudart64_120.dll`), so no host regresses.

### 2026-08-18: Reasoning content is sent to the client; the privacy gate is withdrawn
**By:** Squad (Coordinator), on the owner's instruction (@justinchuby)
**What:** The server must stream and return reasoning content to callers. The
"keep it private" gate proposed in #1224 is cancelled. Any test asserting that
reasoning is *absent* from a response (e.g. `reasoning_never_streams`) is
asserting the wrong behaviour and must be inverted, covering the streamed
deltas and the final non-streamed message as separate paths.
**Why:** Owner decision. Agent clients are expected to see the reasoning turn.
Note the budget half of #1224 ("give a reasoning turn room to finish") is
unaffected and still wanted.

### 2026-08-18: Session-state policy is unified behind one `SessionStore` seam (PR #1255)

**By:** Coding agent (spawned by Squad coordinator), on the owner's standing
"no 区别对待" directive.

**What:** `crates/onnx-genai-engine/src/engine/session_state.rs` now owns the
backend-independent session policy (lookup + "session {id} not found" text, the
rewind bound-checks and their exact strings, checkpoint arithmetic). Native and
ORT both route the six public session methods through it via `SessionStore`
adapters in `runtime.rs`, exactly mirroring the `KvPrefixStore` precedent from
#1170. Error strings are authored once; rendered text is byte-identical.

**Guard (do not remove):** two-part DRY guard, like `ReusedPrefix` +
`KV_REWIND_CALLERS`. (1) compile-time `CheckedPosition` newtype — the only way to
obtain a rewind target is through the one shared bound check; (2) test-time
tripwire `the_rewind_bound_check_lives_only_in_the_shared_policy` fails if
`"cannot rewind session"` is open-coded outside `session_state.rs`. A future
third backend must implement `SessionStore`, not copy the policy. Do NOT widen
the tripwire allowlist to get a green test.

**Asymmetries deliberately kept:** `create_session` (object construction, not
policy); ORT `validate_rewind` validates draft/target/paged-KV while native's is
`Ok(())` (persistent in-process decoder, always admits); ORT token `truncate`
stays inside `rewind_target_state_to_len` because it is shared with the
speculative-decode hot path — the shared policy owns the bound check, not the
truncation.

**Why it is trustworthy:** falsified — inverting the single shared bound check
turns BOTH backends red (3 ORT `failed_rewind_of_*` + native
`native_session_rewind_by_truncates_logical_length`, a new test added to close a
real coverage gap); reverting restores green. Verified: 414 lib tests pass,
clippy clean (default + native), native-backend suite green except a pre-existing
host-RAM KV-budget test, and CUDA (RTX 4060) native_engine ran on-GPU 16 passed /
1 pre-existing unrelated failure.

**Status:** PR #1255 open, not merged, awaiting owner review.

# CPU EP task runtime replaces raw Rayon fan-out in native kernels

**By:** Sebastian (Performance Engineer) — 2026-08-18
**What:** Native CPU kernels no longer fan out through the global Rayon pool.
They call `onnx_runtime_ep_cpu::task_runtime`, which dispatches to ORT's
`KernelContext_ParallelFor` when running inside the plugin EP and to a purpose
-built native pool otherwise. RoPE, Softmax, Transpose and the elementwise
activation fallback are converted; the remaining raw-Rayon sites are unchanged
and still work.

**Why:**

- Rayon parks its workers between parallel regions. Measured on this host, a
  fan-out costs 67 µs back-to-back but 226 µs when it follows a 20 µs gap —
  which is exactly the shape of decode. Documented in
  `docs/benchmarks/2026-08-15-cpu-ep-vs-ort-attention-moe.md` §26/§27.
- The new pool holds its workers in an adaptive spin (20 µs → 500 µs, doubling
  on a catch and halving on a park) so back-to-back and decode-gap dispatch cost
  the same. Measured p50 4.8 µs at a 0 µs gap and 4.9 µs at a 100 µs gap —
  14× and 47× better than Rayon's two numbers, and, more importantly, flat.
- Inside the plugin EP we do not run our own threads at all. Using ORT's own
  intra-op pool is the only way to avoid oversubscribing a host that has already
  sized its pool, and it makes our kernels honour the session's
  `intra_op_num_threads` the way every other ORT kernel does.

**Consequences / rules this sets:**

1. **New parallel kernels use `task_runtime`, not `rayon`.** `for_each_range`,
   `chunk_runs_mut` and `chunks_mut` cover the existing shapes.
2. **Width is inferred, and SMT-capped only above 8 hardware threads.** An
   explicit budget (`set_task_thread_budget`, `ONNX_GENAI_CPU_TASK_THREADS`) is
   honoured exactly. The floor is empirical: below 16 logical CPUs the second
   SMT sibling still pays for these memory-bound kernels, above it does not.
3. **No env-var test hooks in production paths.** Determinism comes from
   `task_runtime::testing` (`force_serial`, `isolated_pool`, `counters`,
   `planned_backend`).
4. **Per-vector SIMD helpers that take a closure must be `#[inline(always)]`.**
   Not a hint: `avx2::map_ps` losing its inline made `Tanh` and `Sigmoid` 2×
   slower on inputs that never reach the parallel path, and the trigger was an
   edit in two unrelated files. `codegen-units = 1` does not prevent this.

### 2026-08-18: #1223 — workspace re-prepare on rebucket

**By:** Bishop

**What:** Gap is REAL and now fixed. Rebucketing did NOT re-prepare governed
workspace: `prepare_with_device_bindings` runs once per generation and latches
`workspace_preparation_required`, after which `execute_kernel` refused to
(re)allocate a prepared slot. Within one shape bucket #1221 made that safe; a KV
growth to a new bucket (or a prompt in a different bucket than its decode steps)
left the reserved `SessionPersistent`/`StepScoped` slot absent or undersized,
reproducing #1221's two failure modes cross-bucket ("workspace invariant
mismatch" / "reached execution without prepared workspace"). Fix: allow a
prepared session to re-prepare (grow) its governed workspace slot **on eager
(non-capture) dispatch** — exactly the dispatch a rebucket forces (the growing-KV
decode path declines capture; a capture-eligible model re-warms eagerly after the
KV-growth graph invalidation before it re-captures). Growth stays forbidden while
recording a captured segment, so a replayed graph's baked workspace pointer is
never invalidated under it. The change lives on the shared executor workspace
path (`dispatch.rs::execute_kernel`), so it is general to every
governed-workspace operator, not special-cased to `Attention`.

**Why:** `Attention`'s route-dependent lifetime classification (decode →
`SessionPersistent`, prefill → `StepScoped`) is what makes it hit this first, not
what makes it unique; the correct fix generalizes #1221 rather than adding another
Attention-specific prepare pass. Gating growth on the eager disposition keeps the
prepared-workspace invariant intact for capture/replay while letting the one safe
point (the rebucket re-warm) re-prepare.

**Evidence:** New executor unit test
`prepared_session_reprepares_workspace_when_execution_rebuckets` reserves a 2-row
`SessionPersistent` slot via prepare, then executes a 4-row bucket. Reverting the
one-line guard fails it with `workspace invariant mismatch: execute requires 4096
bytes aligned to 512, prepared 2048 bytes aligned to 256`; with the fix it grows
in place and passes. `cargo test -p onnx-runtime-session --lib` = 186 passed;
clippy + rustfmt clean.

## 2026-08-18 — V2-Lite MoE CUDA graph-capture and workspace fixes merged

PR #1181 landed on `main` as `c9c7f64c`, unlocking V2-Lite graph capture by fixing the additive-mask `_d1` workspace-planner path; Wallace measured capture ON vs eager OFF byte-identical over 320 tokens, **101.80 vs 56.94 tok/s = 1.79×**.

A separate long-context Engine `Attention` workspace under-plan then surfaced around KV-capacity growth. PR #1189 landed on `main` as `b416a3e0`, fixing Engine/native CUDA single-token decode to re-run governed workspace preparation whenever `ensure_capacity` grows the KV/mask bucket. Leon's A/B on the real V2-Lite path generated 340 token-identical tokens in eager and capture; eager measured 47.32 tok/s, capture 89.69 tok/s with captures=2, replays=336, fallbacks=0. Rachael approved the fix as strictly gated on capacity growth and correctly placed before eager/capture execution.

## 2026-08-18 — DeepSeek-V2-Lite Native-vs-ORT row closed

Wallace measured the real 27-layer DeepSeek-V2-Lite int4 QMoE export under pinned ORT CUDA 1.27 and identical base-decode conditions. Native CUDA serves the model on GPU at **57.15 tok/s eager** and **101.68 tok/s captured**. Stock ORT CUDA EP cannot run the 26 `com.microsoft::QMoE` nodes on GPU; with CPU fallback it inserts 104 host/device Memcpy nodes and reaches **0.17 tok/s**, while strict no-fallback refuses the graph. ORT + CUDA graph is categorically N/A because the graph is split across CPU and GPU. Frame the row as a hard capability gap: native is the only measured GPU engine for int4 QMoE here.

## 2026-08-18 — ORT-fairness dense int4 reconfirmed

Wallace reconfirmed the 2026-08-17 dense int4 Native-vs-ORT decomposition from the opposite direction by trying to enable ORT CUDA graph mode on the same three production exports. True graph-vs-graph is unattainable: Phi-4-mini hard-rejects ORT graph capture because control-flow nodes cannot be supported by CUDAExecutionProvider; qwen2.5-7b aborts at runtime with `ort_value must contain a constructed tensor`; qwen2.5-14b-zp accepts the flag but effectively no-ops because CPU-assigned shape nodes fragment capture (eager 96.9 vs graph 98.7 tok/s, byte-identical).

Eager-vs-eager medians again show this is architectural, not a broad per-kernel native win: Phi native/ORT **0.85×**, qwen2.5-7b **0.77×**, qwen2.5-14b-zp **1.19×**. Keep the deployment headline that native captured decode leads ORT eager **1.33× / 1.14× / 1.83×**, but always label it as CUDA-graph capture plus on-GPU argmax that ORT structurally cannot apply on these dynamic-KV int4 exports.

## 2026-08-18 — Gate-3 speculative verify remains shelved after Marlin

Luv re-probed Deckard's Gate-3 B\* framework on current `main` `923dc592` after Marlin landed. Captured verify break-even `B*=C_verify(M=K)/C_decode(M=1)` is still **NO-GO**: qwen2.5-14b-zp reports **17.5× / 18.4× / 20.0×** for K=2/4/8, and qwen2.5-7b reports **14.9× / 15.7× / 17.4×**. This is worse than the 2026-08-14 baseline (~8.5×) and far above the ≤~2 GO gate, so n-gram/prompt-lookup, EAGLE/MTP, and model-draft speculative-decode work stays shelved.

This updates the earlier spec-decode arc rather than reopening it: the old #957 cheap GQA/SkipSimplifiedLayerNorm residual seams no longer appear as M>1 `KernelCaptureUnsupported` blockers. The measured blocker is now solely `MatMulNBits` at M>1 launching `matmul_nbits_gemm_f16` eagerly; Marlin's capture-safe M>1 int4 GEMM is not selected for this MatMulNBits path. Reconsider only after a graph-safe M>1 MatMulNBits/Marlin path exists and this exact B\* probe is rerun.

## 2026-08-18 — Gate-3 Marlin M>1 opt-in follow-up still NO-GO

Luv re-ran the Gate-3 B\* verify-cost probe with `ONNX_GENAI_MARLIN_M_GT_1=1` and no code changes as a follow-up to the earlier post-Marlin Gate-3 NO-GO. The env gate fixed the capture problem completely: qwen2.5-14b-zp capture segments dropped **96→1**, qwen2.5-7b **29→1**, `KernelCaptureUnsupported` seams disappeared, K=8 byte-identity passed, and the hot path switched to `matmul_nbits_marlin_gemm_f16_splitk`.

The decision does **not** change: speculative decode remains shelved. B\* improved but is still NO-GO at **5.19× / 5.19× / 5.79×** for qwen2.5-14b-zp and **4.64× / 4.71× / 5.23×** for qwen2.5-7b at K=2/4/8, above the ≥~4 kill gate and far above the ≤~2 GO target. The spec-decode family — model-draft, n-gram/prompt-lookup, EAGLE/MTP — is now mined out across the three probes. Residual cost is Marlin M>1 GEMM/repack/reduce (`matmul_nbits_marlin_repack` observed in the hot path), not graph fragmentation.

## 2026-08-18 — Marlin M>1 default flip mined out

Luv completed the real prefill/TTFT A/B for `ONNX_GENAI_MARLIN_M_GT_1=1` versus the portable tiled GEMM path, closing the thread opened by the two prior Gate-3 Marlin entries: “Gate-3 speculative verify remains shelved after Marlin” and “Gate-3 Marlin M>1 opt-in follow-up still NO-GO.” The verdict is **NO-GO to flip the default**: Marlin M>1 stays opt-in.

E2E `profile_native` TTFT showed only marginal-to-neutral qwen2.5-14b-zp movement (**0.976× / 0.988× / 0.999×** Marlin/portable at M=128/512/2048) and neutral-to-worse qwen2.5-7b movement (**1.005× / 1.013× / 1.001×**). Argmax matched every arm, but full-vocab token-0 logprob dumps were not byte-identical (max Δ **0.017** qwen14, **0.168** qwen7), so the silent-default byte-identity bar fails. Treat the Marlin-M>1 vein as mined out: not a spec-decode win, not a prefill/TTFT win, and not eligible for a silent default.
