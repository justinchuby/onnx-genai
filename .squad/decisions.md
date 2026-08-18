# Decisions — live standing directives

Last consolidated: 2026-08-18T17:11Z (Scribe processed Deckard GQA head-size inbox drop; detailed 2026-08 narrative remains archived in `.squad/decisions-archive/2026-08.md` to keep the live ledger below 20KB.)

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

## 2026-08-18 — Marlin M>1 now default ON; byte-identity bar relaxed to argmax stability

Supersedes “Marlin M>1 default flip mined out” (2026-08-18). `ONNX_GENAI_MARLIN_M_GT_1`
now defaults **ON**, and `ONNX_GENAI_MARLIN_SPLITK` moves to opt-in (default OFF).

The earlier NO-GO rested on two findings. The first — that the win is marginal —
was measured only on qwen2.5-14b/7b (0.976–1.013x). It does not generalize: on an
A100-SXM4-80GB (SM80) serving Muse-Glimmer-30B int4, measured end-to-end through
the server with both arms on one binary, a 3247-token prompt goes **37.5s → 16.1s
(2.33x)**, prefill 87 → 202 tok/s; 1647 tok 23.7s → 10.4s; 647 tok 9.4s → 4.7s.
Neutral on some models and 2.33x on others argues for a default plus an opt-out,
not for hiding the win behind a variable nobody sets.

The second finding — full-vocab token-0 logprobs are not byte-identical (max Δ
0.017 qwen14, 0.168 qwen7 at M=128/512/2048) — **stands, and is not attributable
to split-K**. `choose_split_k` returns 1 for `m > SPLITK_MAX_M` (32), so prefill
never elected a split; that divergence is the direct tensor-core kernel's
accumulation order versus the tiled GEMM. An earlier draft of this change claimed
otherwise and was wrong.

So this entry records a deliberate **relaxation of the bar**, not a claim that the
bar is met: the shipping default is now *argmax-stable* rather than
*bit-identical*. Greedy token streams match (validated by `marlin_m_gt_1_e2e.rs`
parity plus an 826-token spot check across the flip: identical answer, one
sentence reworded at a near-tie). Consumers of the *distribution* rather than the
selected token — a logprobs API, beam search, spec-decode acceptance ratios —
should set `ONNX_GENAI_MARLIN_M_GT_1=0`.

Split-K goes opt-in because flipping the parent ON would otherwise newly ship a
second, independent divergence source (its fixed-order fp32 partial reduction) to
every default deployment. It is inert for prefill by construction; the shapes it
governs are M<=32 speculative verify, which is separately shelved. Recover it
with `ONNX_GENAI_MARLIN_SPLITK=1`.

Not addressed: Marlin is no longer the prefill bottleneck. With it on,
`MatMulNBits` is ~20% of a prefill forward (GEMMs at ~74 TFLOPS); attention and
elementwise ops are the other ~80%.
