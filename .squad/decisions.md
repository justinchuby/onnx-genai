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

---

### 2026-08-18: SM-version kernel dispatch scaffolding (arch tier table)

**By:** Batty

**What:** Added arch-guarded kernel-dispatch scaffolding to `onnx-runtime-ep-cuda`
so the pending RTX/consumer-GPU kernels have a clean insertion point, without
touching any live kernel selection. Three pieces:

1. **Device-property probe extension.** Built on the existing
   `runtime::CudaDeviceCapabilities` seam (it already exposes compute capability,
   multiprocessor count, and opt-in shared-mem ceiling). Added an L2 cache-size
   probe (`CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`) with a `0 = unknown` fallback and
   an `l2_cache_size()` getter. No duplicate probe struct created.
2. **Arch dispatch table** (`src/arch.rs`): `ArchTier`
   {Legacy, Volta, Turing, Ampere, Ada, Hopper, Blackwell} with a **total**,
   panic-free `from_compute_capability` mapping, plus an `ArchConfig` of per-tier
   default hints (QMoE tile, resident warps/SM, tensor-core eligibility, smem
   budget, Ada L2-residency candidate). Hint values are seeded from today's
   hardcoded selectors (`qmoe_gemm::tile_for`, `matmul_nbits` resident-warps,
   `marlin_gemm::device_supports_marlin`).
3. **Insertion points**: `// RTX/arch:` hooks tagged RTX-TILING, RTX-SPLITK,
   RTX-CPASYNC, RTX-L2RES mark exactly where the pending device-property tiling,
   split-K-by-SM-count, shared `cp.async` staging, and Ada L2-residency kernels
   plug in.

**Why:** No A100/Ada/Blackwell/RTX hardware is attached (H200 `sm_90` only), so
this is scaffolding + correctness, not live RTX benchmarking. The layer lets us
select kernel variants/tiling by compute capability the moment hardware lands,
honoring the standing "rtx显卡也要优化" directive (optimizations must help
consumer RTX 30/40/50, not just H200).

**sm_90 no-change proof (HARD):** (a) The scaffolding is *not wired into any live
selection path* — `arch_tier`/`arch_config`/`ArchTier`/`ArchConfig` are referenced
only by their own definitions in `runtime.rs` and by `arch.rs`; no kernel selector
(matmul_nbits, qmoe, marlin, gqa) calls them, so dispatch on every device is
byte-for-byte unchanged. (b) The only `CudaDeviceCapabilities` change is an
*additive* `l2_cache_size` field; existing getters return identical values and
existing selectors are untouched. (c) Regression-locked by
`sm_90_hopper_config_is_frozen` (tile 8, 64 resident warps, tensor-core eligible,
no L2-residency) reached both directly and through a synthetic sm_90 device.

**Tests:** `CUDA_VISIBLE_DEVICES=7 cargo test -p onnx-runtime-ep-cuda --lib -- arch:: capability_limits` → 7 passed. `cargo check` clean; `cargo clippy` clean on changed files (the single remaining warning is in the pre-existing dirty `matmul_nbits.rs`, not this change); `cargo fmt` applied.

**Merged PR:** #1287

---

### 2026-08-18: on-GPU argmax base-decode win — GO (byte-identical), tie-break option added

**By:** Deckard

**What:** GO. Promoted/validated the on-GPU device argmax as a byte-identical greedy
BASE-decode win and added a highest-index tie-break OPTION to the device argmax
reduction kernel. Branch `squad/ongpu-argmax-base`, PR #1293 (off origin/main
684c70d0, worktree ../ongpu-argmax-wt).

**Tie-break finding (supersedes the task's stated blocker):** the device argmax
kernel and the engine's host greedy reference BOTH already resolve ties to the
LOWEST token id (`sample_greedy` / `argmax_logits_tensor`, canonical ONNX ArgMax
`select_last_index=false`). They are therefore ALREADY byte-identical with the
default lowest-index tie-break — the feared ~72.9% fp16-ULP-tie divergence does
NOT occur on the shipping path. The `max_by`-based "highest index" references
noted in the task are in test/bench code, not the shipping greedy sampler. I
still delivered a `HighestIndex` kernel OPTION (default stays `LowestIndex`) so the
same reduction can match a `max_by` / `select_last_index=true` reference where
needed; threaded through `ExecutionProvider::device_argmax` → CUDA provider →
session `DeviceIoBinding` → NVRTC partials/finalize kernels. Proven distinct on
identical fp16 ULP-tie inputs (new unit test) + a low-vs-high regression lock.
All 7 ep-cuda device_argmax GPU tests pass (GPU0); engine tensor_argmax tests pass.

**Byte-identity result (0.000% token divergence, 128 tok greedy, PINNED):**
- qwen2.5-0.5b int4 (24 layers, head64), GPU0: divergence 0.000% — PASS
- qwen2.5-14b int4-zp (48 layers, head128), GPU1: divergence 0.000% — PASS

**A/B perf (medians of 5 runs, 128 tok, decode-skip 1, warmups 2, PINNED idle GPU):**
| model | host-argmax tok/s | on-GPU-argmax tok/s | net |
|-------|-------------------|---------------------|-----|
| qwen2.5-0.5b int4 (GPU0) | 500.07 | 638.53 | 1.277x (+27.7%) |
| qwen2.5-14b int4-zp (GPU1) | 155.80 | 169.49 | 1.088x (+8.8%) |

**Wiring status:** on-GPU device argmax is the default in the standalone native
greedy loop (`NativeDecodeSession` → `next_token_greedy` → `decode_cuda_greedy` →
`read_greedy_result` → `device_argmax`). The pipeline decoder (`NativePipelineDecoder::step`)
is deliberately NOT wired here to keep this PR byte-identity-safe; recommend a follow-up.

**Default recommendation:** keep on-GPU argmax ON by default for the native
greedy loop — byte-identical and a strict tok/s win on every model measured. Keep
`LowestIndex` tie-break default; expose `HighestIndex` only as an opt-in kernel option.

**Why:** eliminates the per-step logits D2H + host reduction on the greedy base
path with zero token drift, a pure BASE-decode win (no speculative).

**Merged PR:** #1293

---

### 2026-08-18: Device-property tiling for the int4 decode GEMV (consume arch dispatch)

**By:** Batty

**What:** Wired the SM-version dispatch scaffolding from #1287 into the
int4/accuracy_level=4 decode GEMV so tiling + split-K grid-fill are keyed off
probed device properties through one arch seam. Three pieces, all in
`onnx-runtime-ep-cuda`:

1. `arch.rs` — `decode_resident_warps_per_sm(cc)` reproduces the existing
   resident-warp ladder byte-for-byte (`(8,0)|(9..,_) => 64`, else `48`), and a
   new `DecodeTilingProfile { tier, multiprocessor_count, resident_warps_per_sm,
   sm_count_split_k }` + `one_wave_ctas(threads)` fold the arch tier, SM count
   and resident-warp estimate into the split-K decision the pending RTX kernels
   select through.
2. `runtime.rs` — `CudaDeviceCapabilities::decode_resident_warps_per_sm()`
   delegates to the arch layer.
3. `matmul_nbits.rs` — `use_accuracy4_stage64` now reads its resident-warp
   estimate from the arch helper instead of an inline `match`.

**Which tier gets which tiling:**
- **Hopper `sm_90` (H100/H200):** 64 resident warps/SM; **excluded** from the
  new SM-count split-K lever (`sm_count_split_k = false`) — frozen to today's
  exact selection.
- **Ada `sm_89` (RTX 40 / L4/L40), Ampere `sm_86`/`sm_87` (RTX 30):** 48
  resident warps/SM; opt **into** the SM-count-driven split-K lever
  (`sm_count_split_k = true`). `one_wave_ctas()` scales the occupancy target
  with the probed SM count, so lower-SM parts (e.g. an L4 with 58 SMs) reach one
  wave sooner and split-K fills the grid without oversubscription.
- **Turing/Volta/Legacy/Blackwell:** total, panic-free mapping; consumer tiers
  opt into the lever, Blackwell datacenter stays on the 64-warp rung.

**Why:** Standing directive "rtx显卡也要优化" — the decode GEMV's grid-fill must
track consumer/edge RTX SM counts, not just H200's 132 SMs. Much of the
SM-count-driven split-K already existed (the selectors read
`multiprocessor_count`); this change gives the resident-warp/occupancy input a
single arch-aware seam and a unit-testable `DecodeTilingProfile` for the pending
`rtx-devprop-tiling` kernels, without perturbing H200.

**H200 byte-identity evidence:** structural (arch helper returns same 64 on sm_90 as
old inline match) + tests (`accuracy4_resident_warps_matches_prior_inline_ladder`,
`sm_90_decode_profile_is_frozen_out_of_rtx_splitk`) + GPU parity (19/20
matmul_nbits_gpu decode tests, 1 pre-existing failure on pristine origin/main).

**Verdict:** Zero-H200-change, test-locked scaffolding-that-selects. No Ada/Ampere
hardware attached; RTX path validated via synthetic `for_test` capabilities only.
Benchmark when RTX hardware lands. **Merged PR #1298 (04cfb77e).**

---

### 2026-08-18: Rescue PR #976 — split-block device argmax landed on current main — GO

**By:** Deckard

**What:** Rescued and re-landed the valuable change from draft PR #976 ("Split the
device sampler argmax across blocks instead of one per row") in `onnx-genai-ort`'s
device sampler (`crates/onnx-genai-ort/src/device_sampler.rs`). New branch off
origin/main (04cfb77e): `squad/rescue-976-split-argmax`, PR #1306. Worktree
`/home/justinchu/wt-976`.

**What was dropped:** PR #976 had 3 commits; the bottom one (`2965895f` PTX/loader
prerequisite) already landed on main as #964 — dropped. Cherry-picked only
`0048f5fb` (split-block argmax kernels + dispatch) and `91e64a75`
(dtype/batch/width tests). Both applied cleanly onto current main and compose
correctly with main's post-branch rework.

**Reconciliation with multi-row-argmax rework:** split composes ON TOP —
`argmax_into` derives `parts = argmax_parts(vocab)` and when `parts > 1` launches
`argmax_part_{f16,bf16,f32}` with `grid=(parts, rows)` then `argmax_join` with
`grid=(rows)`. Narrow rows (`vocab <= 2*BLOCK`) return `parts=0` and stay on the
single-launch kernel unchanged.

**Byte-identity result (H200, GPU3, PINNED, release):** UNCHANGED vs current main.
All GPU tests pass (split_matches_single_launch, batched_rows, odd_widths, main's
own parity tests). 19 non-ignored + 6 GPU-gated device_sampler tests pass; 0 fail.

**Kernel A/B (H200, vocab=202048, medians of 5):**
| path | median |
|------|--------|
| one-block-per-row (parts=1) | 38.17 µs |
| split (parts=99, auto) | 23.49 µs |
Speedup **1.62×** end-to-end. Pure-kernel win larger; original PR measured 19.52 µs
→ 3.60 µs at this vocab.

**Verdict: GO.** Byte-identical argmax confirmed across dtype/batch/width on H200.
Split win is NOT superseded by multi-row rework — they are orthogonal (multi-row
spreads rows on y-axis; this adds per-row block split). **Merged PR #1306 (9ac981ca).**
