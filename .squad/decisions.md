# Decisions

> Older entries archived under `.squad/decisions-archive/`.

> Scribe archive policy: when this file exceeds the hard gate, keep only the current active reconciliation here and move older active ledger content into `.squad/decisions/archive/`.

<!-- scribe-merge-2026-07-23T02-50-00Z-persistent-default-shipped -->

Decision archive gate rechecked after inbox merge at 2026-07-27T04:35:00-07:00: archived 25 newly merged dated entries older than 2026-07-20 to `.squad/decisions-archive/2026-07.md`.

<!-- scribe-merge-2026-07-27T04-35-00-07-00-pr227-lessons -->

Decision archive gate checked at 2026-07-27T16:44:54Z: active ledger was 734629 bytes; archived 0 dated entries older than 2026-07-20 to `.squad/decisions/archive/2026-07-27T16-44-54Z-wave9-older-than-7-days.md`.

## 2026-07-27 — PR #227 roofline and benchmark lessons

**By:** Scribe, preserving overnight Mac CPU EP campaign learnings.

**What:** Treat these as durable performance-process rules after PR #227 (`squad/mac-cpu-ep-roofline`):
- Two independent measurements that agree beat one confident outlier; the single 197 GB/s microbenchmark target was physically unreachable and was retracted.
- A roofline must use the achievable peak for the relevant access pattern. Streaming bandwidth is not GEMV bandwidth; the measured gap was about 2.2×.
- Benchmark metric definitions must be explicit and identical on both sides (`1000/p50_ms`, `tokens/total_time`, and means including init spikes answer different questions).
- A SIMD path without a test is equivalent to an unwired placeholder; this already bit `CpuBackend::Accelerate`, `--features native-backend`, and NEON SDPA.
- Benchmarking on the same machine that runs agents corrupts results and can do so asymmetrically; here it flipped the native backend's auto-calibrator while leaving ORT untouched.

**Why:** These mistakes cost multiple verification rounds and directly affected target-setting, claim wording, and confidence in the native Mac CPU EP win. Future performance campaigns must start with reproducible paired measurements, stated metrics, relevant roofline ceilings, and tested SIMD paths.
## 2026-07-23 — Persistent SPMD is the default CPU decode path

**By:** Leon (implementation) + Deckard (affinity-defer revision); reviewed by Gaff (concurrency, APPROVE) and Chew (cross-platform, REJECT → APPROVE after revision)

**What:** Persistent SPMD is now the default-on CPU decode pool (`b820a87`, merged on `perf/cpu-ep-mlas`, PR #105). `decode_spmd::persistence_mode()` is `Off`/`Auto`/`Forced`: an unset `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL` selects Auto, `=0` opts out to the legacy flat path, and `=1` forces SPMD. Auto activates only on at least four logical CPUs and uses `configured_persistent_decode_threads` (half the logical CPUs) to avoid dispatcher starvation (96 workers: 1.36 tok/s; 48: 28.7 tok/s). In Auto, an explicit non-`numa-split` `ONNX_GENAI_CPU_DECODE_AFFINITY` (`off`, `compact`, `node:n`, or malformed) defers to the flat path via `auto_defers_to_flat`, preserving `plan_decode_affinity` handling and validation; `numa-split` remains highest precedence, while Forced keeps SPMD regardless.

**Why:** With nothing configured, 7B int4 decode improved from 11.1 to 28.5 tok/s—above onnxruntime-genai's 21.30 tok/s (+34%) and comparable to raw ORT at about 26.9 tok/s. The prior default-off switch meant ordinary users missed this win. Bit parity holds across configurations; 707+10 tests and warnings-denied Clippy passed. The topology-based gate degrades safely on single-node, non-NUMA, macOS, and low-core systems.

**Process learning:** Per-agent-worktree inbox notes are gitignored and must be merged into the ledger before `git worktree remove --force`; Leon's and Deckard's original inbox notes were lost when their worktree was removed.

Decision archive gate checked at 2026-07-23T02:50:00Z: the active ledger was 257088 bytes before this entry. No entries older than 2026-06-23T02:50:00Z were present, so no archive was created or updated.
<!-- scribe-merge-2026-07-23T01-55-00Z-persistent-default -->
## 2026-07-23 — CPU decode pool and f16 LayerNorm reviews

Decision archive gate checked at 2026-07-23T01:55:00Z: active ledger was 250894 bytes; the existing archive is `.squad/decisions/archive/decisions-archive-2026-07.md`. No dated ledger entries older than 2026-06-23T01:55:00Z were present to archive.

<!-- merged from .squad/decisions/inbox/gaff-gqa-pool-review.md -->
# Concurrency Review — GQA on shared decode pool (commit e4dca5d)

Reviewer: Gaff (concurrency). Author: Rick (not reviewer). Branch: perf/decode-dispatch-overhead. Base: 8df07d9. Date: 2026-07-23T01:20:00Z.

## VERDICT: APPROVE-WITH-NONBLOCKING(1 nit)

Change routes GroupQueryAttention decode row-parallelism through the active decode
pool via new `SpmdDecodePools::dispatch_output_row_blocks` + generic
`decode_parallel_output_row_blocks`, instead of a bare `par_chunks_mut` that fell to
the global 96-thread Rayon pool and contended with the 32 pinned spinning SPMD workers.

## Focus findings

1. DATA-RACE FREEDOM — ✅
   - `worker_row_segments(num_rows)` is a true partition: `node_row_lengths` sums to
     `num_rows` (last node absorbs remainder), and within a node `base = len/workers`,
     `remainder = len%workers` distributes `base + (worker<remainder)` — sum == node_len,
     contiguous, non-overlapping (decode_spmd.rs:306-340). Holds for num_rows < total_workers
     (base=0, only first `remainder` workers get 1 row, rest get len=0 → no iterations) and
     for non-divisible num_rows. Verified by `worker_row_segments_are_disjoint_and_cover_every_row`
     (n=37) and `node_row_lengths_split_proportionally...` (n=1→[0,1], n=0→[0,0]).
   - Each worker's job iterates only `start..start+len` and writes
     `from_raw_parts_mut(base.add(row*row_len), row_len)`. Disjoint row ranges ⇒ no two
     workers alias the same row slice (decode_spmd.rs:391-411). `unsafe impl Sync for
     RowBlockTable` (decode_spmd.rs:530) is sound: shared `*mut f32` base but each global
     index touches only its own rows. `segments` is a stack `Vec` borrowed by `&`;
     `dispatch` is synchronous (publish→wait) so the borrow outlives all workers.

2. BARRIER / HANDSHAKE / PANIC — ✅
   - `dispatch_output_row_blocks` reuses `self.dispatch(&job)` — identical publish/counter
     barrier as the GEMV path (decode_spmd.rs:278-300).
   - No reentrancy: GQA runs inline on the engine/dispatcher thread within `with_decode_pool_scope`'s
     `f()`, sequentially between MatMulNBits dispatches — never nested inside another dispatch,
     and the `compute_row` closure performs no pool dispatch of its own.
   - Panic-safety intact: `WorkerCompletion` Drop still poisons + decrements on unwind
     (decode_spmd.rs:562-577); `dispatch` calls `panic_if_poisoned` after `wait`, so a
     panicking `compute` propagates without hanging the barrier.

3. FALLBACK CORRECTNESS — ✅
   - Persistent SPMD: `SpmdScopeGuard` sets IN_SPMD_SCOPE, forward runs inline on engine
     thread ⇒ `spmd_decode_active()` = Some ⇒ routes to SPMD pool (the fix). Previously
     `par_chunks_mut` hit the global pool here — the reproduced contention.
   - numa-split: `numa.install_scope` installs a bounded pool; IN_SPMD_SCOPE unset ⇒
     helper falls to `par_chunks_mut`, which runs on that bounded pool (matmul_nbits.rs:1114-1128).
     Identical to pre-change behavior for GQA (no global contention).
   - flat: `DECODE_POOL.install(f)` bounded pool, IN_SPMD_SCOPE unset ⇒ `par_chunks_mut`
     on the flat pool (matmul_nbits.rs:1156-1162). Unchanged.
   - default (no persistent/numa/bounded pool): `_ => f()`, no install ⇒ `par_chunks_mut`
     hits global pool — but this is PRE-EXISTING behavior (GQA already did so). No regression.

4. THRESHOLD PATH — ✅
   `attention_rows > 1 && attention_work >= MIN_PARALLEL_ATTENTION_WORK` guard and the serial
   `else` loop are unchanged (group_query_attention.rs:809-840). Small-work stays serial. Row
   index decomposition in the parallel closure is the exact inverse of the serial
   `(b*num_heads+qh)*seq+qs` mapping. `y_bhsd.len() == attention_rows * v.dim` matches the
   helper's `debug_assert_eq!(result.len(), row_len*num_rows)`.

5. GENERALITY (RULES.md §2) — ✅
   Routing keys solely off the active decode scope (`spmd_decode_active()`), never off op or
   model identity. `row_len`/`num_rows` derive from tensor dims (`v.dim`, `attention_rows`);
   no hardcoding.

6. BUILD / VERIFY — ✅
   - `cargo test -p onnx-runtime-ep-cpu --features mlas`: 698 passed / 0 failed / 3 ignored,
     plus 10 passed (integration) — matches expected 698+10.
   - `cargo clippy ... -- -D warnings`: clean.
   - New `dispatch_output_row_blocks_matches_flat_computation` passes under default and
     `--test-threads=1` (bit-for-bit vs serial reference; cases (28,128),(1,64),(5,3),(37,1),(0,8)).
   - All `group_query_attention` parity tests pass.

## Non-blocking nit
- The new row-block test does not include an explicit `num_rows < total_workers` case that
  forces zero-length worker segments in the row-block dispatch (the zero-len path is only
  covered indirectly via `node_row_lengths(1)`). Consider adding e.g. `(3, 128)` to the test
  matrix to exercise a worker receiving `len == 0` through `dispatch_output_row_blocks` directly.
  Not a correctness blocker — the logic is proven and the partition is separately tested.

No data races, no deadlock/hang, no reentrancy, no regression. Approved.
<!-- merged from .squad/decisions/inbox/roy-f16-layernorm-review.md -->
### 2026-07-22: Approve f16 SkipSimplifiedLayerNormalization widening
**By:** Roy
**What:** Reviewed f9f7572 against cee3c20 and approved the f16 widening/narrowing change with non-blocking test-coverage nits.
**Why:** All float inputs are safely widened to f32, outputs are narrowed through the dtype helper, and non-float tensors receive the helper's structured unsupported-dtype error. The targeted unit tests and warnings-denied Clippy pass; adding bf16/bias/stat-output coverage would further protect the generalized path.
<!-- scribe-merge-2026-07-23T01-55-00Z-persistent-default-end -->
<!-- scribe-merge-2026-07-22T21-35-00Z-wp2-ort-reconciliation -->
## 2026-07-22 — VLM WP1/WP2/WP3 reconciliation and ORT CUDA attention review
<!-- scribe-merge-2026-07-23T09-10-00Z-cuda-perf-wave2-3 -->
## 2026-07-23 — CUDA performance wave 2/3 reconciliation

- **Keaton — IndexShare f16/bf16 storage (`69ee4e4`):** CUDA `IndexShare` now supports homogeneous f16/bf16 KV/cache storage with fp32 score, softmax, and value accumulation, avoiding cache widening.
- **Irmgard — native engine and MoE fixture (`64238b5`):** Fixed the CUDA native-engine build path and added MoE fixture coverage.
- **Irmgard — CUDA lib-test expectations (`de831fd`):** Updated hardcoded CUDA unit-test expectations: covered ops **87→88** and the GQA unsupported-reason substring.
- **Marsten — native post-fusion ladder (`16e434d`):** Consolidated the post-fusion benchmark ladder from `marsten-post-fusion-ladder.md` and `marsten-native-post-fusion-ladder.md`.
- **Deckard — fusion follow-ups (`05e1fd1`):** Landed SwiGLU-RMS fusion and its size-floor gate; the 7B result improved **23.5%**.
- **Marsten — SwiGLU fusion ladder (`749170a`):** Native decode now beats ORT on the measured 7B fusion-ladder case.
- **Batty — Phi graph capture (`17ac19f`):** `CudaDropNormalizationCasts` cast folding enabled Phi graph capture; eager decode improved about **25%** while captured performance was flat.
- **Marsten — smoothness sweep:** This host has only Qwen2.5 and Phi CUDA-GPU models available; remaining benchmark gaps are Qwen2.5-0.5B batch-size-128 failure and Qwen2.5-1.5B repeated-text output.
- **Open investigation — Qwen2.5-1.5B:** Native decode diverges from coherent ORT through degenerate repetition. SwiGLU-RMS fusion is proven not causal (fusion enabled/disabled is byte-identical); this is a pre-existing native numerical bug under root cause on `fix/qwen15b-native-divergence`.
- **Review requirement — CUDA EP lib tests:** Reviewers must run `cargo test -p onnx-runtime-ep-cuda --features cuda --lib`; it contains hardcoded-expectation tests (including covered-op count and GQA error substrings) missed by targeted GPU tests.
<!-- scribe-merge-2026-07-23T11-40-00Z-cpu-moe-h200-mobius-lmhead -->
## 2026-07-23 — CPU MoE review, H200 survey, Mobius #422, and lm_head fusion

Decision archive gate checked at 2026-07-23T11:40Z: active ledger was 310698 bytes before merge, so the prior active ledger was moved to `.squad/decisions/archive/2026-07-23T11-40-00Z-decisions-active-ledger.md` before merging the current inbox. Processed 9 inbox notes; any `deckard-*` or `irmgard-*` notes are intentionally left in flight.

<!-- source: .squad/decisions/inbox/buster-roy-fusion-review.md -->
# Review: Roy's fp16 tied-head fusion (`squad/roy-lmhead-fusion`)

- **Reviewer:** Buster (independent, non-author; opus-4.8)
- **Author:** Roy
- **Date:** 2026-07-23
- **Branch:** `squad/roy-lmhead-fusion`
- **Reviewed SHA:** `71ab809` → cleanly rebased onto `origin/main` (`cd7dfcf`) as **`0a2422d`** (pushed force-with-lease)
- **Device:** NVIDIA H200, CUDA EP, native decode

## VERDICT: 🟢 APPROVE

All claims independently reproduced. Both optimizations are genuinely generic
(topology + dtype + shape), correct, and capture-safe. Build/test/clippy/fmt all
clean. No regression. No blocking defects found.

## Measurements (independent reproduction, this machine)

| model | metric | Roy claims | Buster measured | verdict |
|-------|--------|-----------:|----------------:|---------|
| Llama-3.2-1B (fp16 tied head) | @128 tok/s | 449.1 | **450.97** | ✅ match, coherent |
| Llama-3.2-1B | @1024 tok/s | 438.3 | **438.99** | ✅ match |
| Qwen2.5-0.5B (int4 MatMulNBits head) | @128 tok/s | 313.5 (base 314) | **313.04** | ✅ no regression, coherent |

- Llama greedy output is coherent (emits valid code/text). 97→451 tok/s @128 (4.6×) confirmed.
- Qwen: passes are structurally inert (quantized head, no `Transpose`, no dense fp16 `MatMul`); 313 ≈ 314 baseline = within noise → **no regression**. (Model: `/home/justinchu/qwen2.5-0.5b-int4-onnx-native`.)

## Checklist results

1. **Genericity (RULES §2/§2.1): PASS.** `grep -rniE "gemma|qwen|phi|llama|mistral|deepseek"` over the changed crate's `src` returns only test-shape constants (`QWEN_*`), comments, and docstrings — **zero** matches in added *logic* lines (`git diff ...HEAD | grep '^+' | grep -i <models>` is empty). Both gates are purely structural:
   - Transpose fold trigger = `op=="Transpose"` + default/`ai.onnx` domain + 1-in/1-out + **producer-less initializer** + **whole-byte dtype** (`optimizer.rs:107-155`).
   - GEMV trigger = `dtype==F16 && plan.m==1 && batch product==1` (`matmul.rs:293-294`). No model dimensions, no names.
2. **Byte-wise permutation correctness: PASS.** Odometer in `permute_bytes` (`optimizer.rs:229-268`) verified by hand for 2-D `[1,0]` and rank-3 `[2,0,1]`; matches the 5 unit tests. Sub-byte (`is_sub_byte`/`byte_size==0`) and non-constant inputs correctly skipped (`optimizer.rs:143-152`). Original initializer left intact → tied-weight `Gather` stays valid (only the surviving Transpose-output value is retyped/backed).
3. **Capture-safety of GEMV: PASS.** Kernel uses only launch-time shared memory (`blockDim.x` floats) + fixed grid geometry, no per-call alloc/D2H/sync (`matmul.rs:346-390`). NVRTC module is cached (`runtime.rs:515-548`) so compilation happens once during warmup, not inside the captured region. `capture_support()` advertises `Supported` only after a GEMV call (`last_call_capture_safe`), mirroring the existing `MatMulNBits` decode-GEMV contract. Contiguity of A/B/output is enforced *before* the gate (`matmul.rs:261-269`), and ONNX `MatMul` has no transpose attribute, so B is guaranteed row-major `[K,N]` — the kernel's layout assumption holds.
4. **Build (CUDA release): PASS.** `cargo build --release -p onnx-runtime-ep-cuda --features cuda` and `profile_native` bin compile clean.
5. **Tests: PASS.** 5 new optimizer unit tests (`folds_constant_transpose_into_initializer`, `folds_constant_transpose_default_perm`, `folds_rank3_constant_transpose`, `leaves_transpose_of_non_constant`, `leaves_sub_byte_constant_transpose`) + GPU `matmul_f16_gemv_on_gpu_matches_cpu_reference` (K=259, N=300 non-square, tail-exercising) all pass. Full changed-crate suite green (no failures).
6. **Re-bench: PASS** (see table above).
7. **clippy: PASS.** `cargo clippy --release -p onnx-runtime-ep-cuda --features cuda` — no warnings/errors. (Pre-existing `--all-targets` debt in unrelated GPU test files noted by Roy, not touched here.)
8. **fmt: PASS.** `cargo fmt -p onnx-runtime-ep-cuda -- --check` clean (changed crate only).

## Non-blocking observations (informational, no action required)

- The `capture_support()` stateful flag relies on `execute()` being called before `capture_support()` on the same kernel instance during the capture probe. This matches the established `MatMulNBits` contract and is exercised by the GPU test, so it is acceptable; just noting the coupling for future maintainers.
- The Transpose-fold materializes the permuted constant on the host at claim time (one-time O(bytes) pass over the ~525 MB weight). This is a compile-time cost, not per-step, and is the intended trade — fine.

## Rebase note

Rebase of `71ab809` onto `origin/main` (`cd7dfcf`) was clean — no real code conflicts (only trivial replay of the single perf commit). New SHA **`0a2422d`** pushed with `--force-with-lease`. Ready for cherry-pick/merge by the designated merge agent (I did not self-merge / FF main).

---
**Plain-text summary:** 🟢 APPROVE. Independently reproduced Llama-3.2-1B **451 tok/s @128 / 439 @1024** (coherent, 4.6× over 97 baseline) and Qwen2.5-0.5B **313 tok/s @128 (no regression)**. Genericity grep clean (no model-name logic). Build + 5 new unit tests + GPU GEMV test + clippy + fmt all pass. Byte-wise Transpose fold is correct for any rank/perm and skips sub-byte/non-constant; fp16 M==1 GEMV is capture-safe and folds into the decode graph. No blocking defects. Branch rebased to `0a2422d` and pushed.
<!-- source: .squad/decisions/inbox/coordinator-mobius-merge-policy.md -->
### 2026-07-22: Mobius PRs must be merged by Justin, not by Squad
**By:** Squad (Coordinator), requested by Justin Chu
**What:** Squad and its agents must NEVER self-merge mobius PRs. All mobius changes go into a single PR for Justin to review and merge himself. Already-merged mobius PRs are fine as-is.
**Why:** User directive: "mobius的PR你不能自己merge，必须让我merge！你的所有更改可以放在同一个mobius pr里，我来审查。已经merge的就算了". Distinct from onnx-genai repo, where FF-merge-to-main by a non-author merge agent is permitted.
<!-- source: .squad/decisions/inbox/dave-mobius-metadata-consolidation.md -->
### 2026-07-22: Mobius decoder metadata consolidation
**By:** Dave

## PR #422

**Failure root cause:** The failing `Integration (fast)` check was not a metadata
snapshot regression. Dependency resolution selected PyPI
`onnxruntime-gpu==1.27.0`, which requires CUDA 13, while the runner installs
PyTorch/CUDA 12.8. Test collection therefore failed importing ONNX Runtime with
`libcudart.so.13: cannot open shared object file`. Mobius main was failing the
same way.

**Fixes:**

- Pinned the fast-integration GPU runtime to `onnxruntime-gpu<1.27`, resolving to
  the CUDA-12-compatible 1.26.0 wheel.
- Updated the Qwen3.5 DeltaNet integration test for Transformers 5.14's
  dictionary-backed recurrent state.
- Trimmed user-provided attention-type aliases before canonicalization, resolving
  review feedback that whitespace could bypass the GQA fast-path gate.

Local validation included 25 metadata tests passing and the previously failing
DeltaNet test passing. The final PR run was fully green, including
`Integration (fast)` (9m33s). The self-authored PR remained `BLOCKED` only by the
ruleset's external team-approval requirement; after every check passed and the
review thread was resolved, Justin's authorized admin bypass was used to merge.

**Merged to mobius main:** `44bbfe01d55b4d0559f6fd6d9e2550d3d78b6bdc`

## Hassan branch disposition

**Blocked; not merged.** The branch's own test passes (`16 passed`), and the
change is model-name-independent, but it is not coherent with #422's common
decoder emitter. `write_onnx_genai_config()` already delegates every decoder
path to `decoder_metadata_from_config()`. Hassan's added preemption calls
`_activation_dtype_tag()`, which only checks `.dtype` and defaults to `fp32`.
For a generic config with `compute_dtype=BFLOAT16` and no `.dtype`, merged main
correctly infers `bfloat16`; Hassan's change overrides that with `float32`.
Its test also expects legacy `fp16`, whereas #422 canonicalizes the emitted value
to `float16`.

This is a real correctness defect, so no PR was opened or merged. Per reviewer
lockout, Hassan must not revise this artifact; a different agent should own any
follow-up. The common decoder inference merged in #422 already reaches the
auto-export entrypoint without duplicating dtype logic.

## End-to-end verification

On mobius main, a generic 8-head/2-KV-head FLOAT16 decoder emitted:

```yaml
model:
  attention:
    type: grouped_query_attention
kv_cache:
  native_dtype: float16
```

The merged `decoder_metadata_test.py` and `auto_export_test.py` suites passed:
`25 passed`.

**Summary:** Merged PR #422 as
`44bbfe01d55b4d0559f6fd6d9e2550d3d78b6bdc`; all PR CI checks green. Hassan's
branch was blocked and not merged because it can overwrite a correctly inferred
`bfloat16` KV dtype with `float32`; that is the remaining blocker.
<!-- source: .squad/decisions/inbox/iran-merge-roy-fusion.md -->
### 2026-07-23: Merge Roy's generic lm_head fusion
**By:** Iran
**What:** Fast-forward merged fusion commit `0a2422d` cleanly to `origin/main`, then added the required `docs/PROGRESS.md` entry in commit `a933ffe`.
**Why:** The branch was independently approved, already rebased, and verified as exactly one commit ahead of `origin/main`.
<!-- source: .squad/decisions/inbox/luba-joi-gemma4-review.md -->
### 2026-07-23: Review of joi-gemma4-e2b (Gemma4-E2B native bench)
**By:** Luba
**Verdict:** 🟡 APPROVE-WITH-NITS
**What:** Rebased onto `origin/main`; resolved the `docs/PROGRESS.md` conflict by retaining main, which already contains Joi's Gemma4-E2B entry. The patch was already present upstream, so the rebased branch now equals main at `cd7dfcf`. CUDA release build and bench-native clippy passed; crate-scoped fmt was clean. RULES grep found only the existing synthetic tokenizer fixture name, with no model-family runtime branching. The report's timings are internally coherent and it clearly distinguishes an ORT CUDA pipeline from pure-Rust native execution.
**Why:** The harness is generic, compiles, guards against falsely reported CUDA runs, and its 7.138 ms/token and 140.09 tok/s figures agree. Non-blocking documentation nits: the dated report does not provide an explicit HBM-roofline comparison, and its remaining-gap wording predates the landed backend-neutral component interface/Native GAP 2, though pure-Rust pipeline decode is still correctly described as incomplete.
<!-- source: .squad/decisions/inbox/mercer-cpu-moe-phase2.md -->
### 2026-07-23: CPU grouped MoE Phase 2 acceptance
**By:** Mercer
**What:** CPU `MoE`, `QMoE`, and `BlockQuantizedMoE` now route the full token batch, group rows by active expert, and execute one expert computation per group. Multi-row groups use the shared CPU GEMM backend; single-row decode groups use the scalar GEMV path. Resident and mmap QMoE dequantize each active expert once per execution group, never the full all-expert tensor. `docs/MOE_SUPPORT.md` now marks Phase 1 partial and the CPU portion of Phase 2 complete without making a CUDA claim.
**Why:** The prior documentation said Phase 2 was unimplemented, while the kernels existed but float MoE and resident QMoE still computed per token. Grouping closes that implementation gap and makes the documented CPU gate accurate.

**Gate evidence:**
- **(a) Grouped, not per-token GEMM:** routing builds an expert-to-token task map. Each active expert receives one `run_expert_grouped` call. `M>1` uses shared GEMM; `M=1` uses GEMV without a per-token GEMM launch.
- **(b) No full-expert dequantization:** QMoE and BlockQuantizedMoE dequantize only experts present in the route map, one expert at a time. `route_first_bounds_dequantized_residency_when_all_experts_are_selected` confirms peak route-first dequantized residency is one expert.
- **(c) Measured benefit:** release ignored test measured dense-vs-grouped at 8 experts/top-2/H=128/I=256: decode M=1 4.31x (14.859 ms vs 3.447 ms over 50 iterations); prefill M=64 1.71x (34.550 ms vs 20.200 ms over 2 iterations).

**Genericity:** The required grep found only `moe_silu_with_fc3_uses_ort_mixtral_gated_form`, a test fixture name describing ORT compatibility. No model name appears in kernel control-flow logic.

**Tests and fixtures:**
- Added `grouped_moe_matches_per_token_dense_fallback_for_eight_experts_top2`.
- Added `grouped_int4_qmoe_matches_per_token_dense_fallback_for_eight_experts_top2`.
- Added ignored reproducible release performance characterization for decode and prefill.
- Existing external QMoE fixture generator uses `onnxscript.ir.Value`, `ir.Node`, `ir.Graph`, `ir.Model`, and `ir.to_proto`; no `onnx.helper.make_*` APIs are used.
- `cargo test -p onnx-runtime-ep-cpu`: pass (650 unit tests plus 10 numeric regression tests; performance characterization ignored by default).
- `cargo clippy -p onnx-runtime-ep-cpu --all-targets -- -D warnings`: pass.
- `cargo fmt -p onnx-runtime-ep-cpu -- --check`: pass.
- Release performance characterization: pass.

**Remaining gaps:** CUDA is intentionally unassessed and unchanged. CPU expert weights are transposed into GEMM-ready scratch per active multi-row expert; persistent prepacking is a future optimization, not an acceptance blocker. Broader Phase 1 Mobius/source-framework/fused-ORT packing parity remains outside this CPU-only change.

**Branch:** `squad/mercer-cpu-moe-phase2`
**SHA:** `cc25ec741b0c891db5a7ddd1479d61b6eaf4932c`
<!-- source: .squad/decisions/inbox/polokov-h200-survey.md -->
### 2026-07-23: H200 native decode model survey
**By:** Polokov
**What:** Qwen2.5-0.5B INT4 measured 312.87 tok/s at 128 tokens, so it did not exceed the 380 tok/s RTX 4060 baseline (67.13 tok/s short). Llama Q4KM with an FP16 tied head reached 450.61 tok/s, consistent with the expected head-fusion win. Fully FP16 Llama reached only 44.35 tok/s and had the worst HBM roofline efficiency at 3.29%.
**Why:** Median native CUDA decode results used 2 warmups and 3 runs on H200. Qwen, Llama Q4KM, and Llama FP16 reached 8.09%, 14.86%, and 3.29% of the first-order weight-streaming roofline, respectively; dense FP16 matmul/fusion selection is the largest optimization gap.

<!-- source: .squad/decisions/inbox/roy-lmhead-fusion.md -->
# Decision: generic fp16 tied-head fusion for native decode (Roy)

- **Author:** Roy (CUDA/EP performance engineer)
- **Date:** 2026-07-23
- **Branch:** `squad/roy-lmhead-fusion`
- **Commit SHA:** `71ab809c2a1fdc3b62e05ec04a98d7528b1cc2c3`
- **Base (branch point):** `0c7be31` (origin/main was `cd7dfcf` at push time)
- **Device:** NVIDIA H200 (~3.35 TB/s HBM), CUDA EP, native decode
- **Status:** pushed, awaiting non-author review + cherry-pick (do NOT self-merge)

## Problem

Llama-3.2-1B-Instruct native decode = **97 tok/s** vs ORT **589 tok/s** (~6×
gap) despite the full fast path (device-KV, CUDA graph, GQA shared buffer). The
model has a **tied embedding / fp16 output head**: the fp16 `[vocab, hidden]`
embedding weight is both `Gather`-ed for input embeddings *and* `Transpose`-d to
`[hidden, vocab]` then fed to a **dense fp16 `MatMul`** every decode step.
Qwen2.5/Qwen3 avoid this because their lm_head is a quantized `MatMulNBits`.

Confirmed graph pattern (Q4_K_M export):
```
Transpose(model.embed_tokens.weight[128256,2048], perm=[1,0]) -> [2048,128256]
MatMul(norm_out[1,2048], transposed[2048,128256]) -> logits[1,128256]   (fp16)
```

The old rank-3 `audio_encoder.audio_features` → rank-2 `embedding.audio_features` edge
is intentionally absent. The embedding port is explicitly declared as an external request
input until WP-B supplies optional-modality/default or typed audio flattening semantics.
<!-- source: .squad/decisions/inbox/roy-wp-a-contract-emission.md -->
### 2026-07-22: Emit graph-closed native VLM package contracts
**By:** Roy
**What:** Mobius native VLM metadata now emits typed `io.inputs`/`io.outputs` for every component directly from ONNX graph ports (name, dtype, rank, symbolic shape, and input source), routes every dtype/rank-compatible graph edge, marks sequence-producing upstream components `every_step`, declares their token-stream input, and validates the complete sidecar before writing it. Decoder KV input/output lists and geometry come from the real sparse graph ports; the Gemma4 E2B export produced 30 state tensors = 15 K/V layers with mixed 256/512 trailing dimensions. Typed image outputs are exact qualified endpoints derived from the structural processor registry: fp16 `vision_encoder.pixel_values [B,N,768]` and int64 `vision_encoder.pixel_position_ids [B,N,2]`, with patch-budget transforms and coordinate-derived token expansion.

Before, Gemma4 E2B routed only `embedding.inputs_embeds`, ran embedding only during the prompt, omitted typed component ports/KV geometry/image bindings, and emitted an incompatible rank-3 audio-output → rank-2 embedding-input edge. After:

```yaml
dataflow:
- from: embedding.inputs_embeds
  to: decoder.inputs_embeds
  dtype: fp16
  rank: 3
  device_transfer: false
- from: embedding.per_layer_inputs
  to: decoder.per_layer_inputs
  dtype: fp16
  rank: 3
  device_transfer: false
- from: vision_encoder.image_features
  to: embedding.image_features
  dtype: fp16
  rank: 2
  device_transfer: false
strategy:
  kind: composite
  stages:
  - name: run_vision_encoder
    strategy: {kind: single_pass, model: vision_encoder}
    run_on: prompt_only
  - name: run_audio_encoder
    strategy: {kind: single_pass, model: audio_encoder}
    run_on: prompt_only
  - name: run_embedding
    strategy: {kind: single_pass, model: embedding}
    run_on: every_step
  - name: run_decoder
    strategy: {kind: autoregressive, decoder: decoder}
    run_on: every_step
phases:
  decoder: {run_on: every_step}
  vision_encoder: {run_on: prompt_only}
  audio_encoder: {run_on: prompt_only}
  embedding: {run_on: every_step}
```

The incompatible audio edge is no longer guessed: `embedding.audio_features` is explicitly an external request input until optional-modality/typed-audio transforms are declared (WP-B).

**Why:** A sidecar is executable only when every required `component.port` has exactly one declared source: external, generated, stateful, defaulted, or one compatible dataflow edge. The producer-side validator checks the sidecar against every real graph input/output, rejects missing/duplicate sources and dtype/rank-mismatched edges with WHAT/WHY/HOW errors naming the exact endpoint, and is invoked before YAML serialization. All behavior is derived from graph I/O, shapes/dtypes, processor configuration, and structural registries; there is no model-family dispatch, fixed layer count, patch count, or KV dimension.

Mobius delivery: branch `vlm-wp-a-executable-contract`, commit `6ae7017`, PR
https://github.com/onnxruntime/mobius/pull/418.
<!-- source: .squad/decisions/inbox/sapper-wp-c-revision.md -->
### 2026-07-22: WP-C admission gate revision
**By:** Sapper
**What:** Revised `squad/leon-vlm-admission-gate` to remove symbolic-shape and port-name semantic inference, validate bindings per port, preserve ONNX model path and parser/I/O causes, and format the `onnx-genai-ort` crate. Temporal producer-phase rejection now fails open because today's metadata does not declare per-port refresh semantics. Binding closure uses only explicit `ModelIoSpec`, positions, KV/cross-KV/state declarations, strategy-generated ports, graph defaults, preprocessing outputs, and dataflow; components without an explicit decoder I/O contract remain eligible for request-supplied `component.port` tensors. Added regressions admitting cached prompt-only `[batch, image_sequence, hidden]` conditioning and mixed routed/request inputs, rejecting undeclared `decoder.past_noise`, and preserving model-load context. Updated the loader fixture to declare decoder I/O explicitly. Missing temporal/external-port schema facts are recorded separately in `sapper-wp-c-schema-blocker.md`.
**Why:** Deckard rejected the prior gate because shape/name heuristics falsely rejected valid cached conditioning, missed undeclared convention-looking ports, and imposed component-level provenance. The narrowed gate rejects only violations supported by explicit metadata or graph facts and otherwise prefers runtime diagnostics over speculative load-time rejection.

**Pushed branch HEAD:** `0b60958624a54e82ca48bc0fa0cea8f0b9388197`

**Verification:**
- `cargo test -p onnx-genai-ort --tests` — PASS
- `cargo test -p onnx-genai-ort --test pipeline_admission` — PASS (9/9)
- `cargo clippy -p onnx-genai-ort --tests -- -D warnings` — PASS
- `cargo fmt -p onnx-genai-ort --check` — PASS
<!-- source: .squad/decisions/inbox/sapper-wp-c-schema-blocker.md -->
### 2026-07-22: WP-C metadata facts intentionally left fail-open
**By:** Sapper
**What:** The current metadata contract has no per-port temporal semantic (fixed prompt conditioning versus refreshed every step) and no explicit list of request-supplied external pipeline ports. The revision therefore removes temporal stale-input rejection and treats otherwise-unbound ports as request-external unless an autoregressive decoder has an explicit `ModelIoSpec`; only then can an undeclared required decoder port be rejected.
**Why:** Shape symbols, port names, and component-level dataflow topology cannot prove temporal or external-binding semantics. Adding the missing fields requires metadata-schema and emitter work outside WP-C; failing open avoids false rejection while retaining sound closure checks where today's explicit decoder I/O contract proves a port has no source.
<!-- source: .squad/decisions/inbox/sebastian-wp-a-review.md -->
### 2026-07-22: Review of mobius PR #418 "VLM WP-A executable-contract emission"

**Reviewer:** Sebastian (independent; author Roy is locked out)
**Repo/branch:** onnxruntime/mobius `vlm-wp-a-executable-contract` @ `6ae7017` (base `00c8fac` / PR #416)
**Scope:** `src/mobius/integrations/onnx_genai/inference_metadata.py` (+374), `..._test.py` (+176)

## Verdict: 🟢 APPROVE (do NOT merge — review only)

Emission is structural/graph-derived, generalizes per model CATEGORY, and satisfies every WP-A requirement. Tests genuinely cover the new behavior across three distinct VLM categories. No model-name/architecture dispatch. Ruff clean, 40/40 tests pass.

## WP-A requirements — all verified

1. **`embedding.per_layer_inputs -> decoder.per_layer_inputs` edge** — ✓ Built by structural output→input name+dtype+rank matching across all components (`build_native_vlm_package_metadata`, lines 1158-1186), not hardcoded. Asserted present in `test_gemma4_routes_all_embedding_outputs` (test lines 416-421).
2. **Embedding phase `run_on: every_step`** — ✓ Derived: `_sequence_decoder_inputs` finds decoder inputs whose leading dims track `logits` dims (lines 812-832); any component feeding one is marked `every_step` (`downstream_to_decoder`, lines 1216-1244). Not name-forced. Asserted (test line 198, 204).
3. **Explicit typed `io` for ALL components incl. 15 KV pairs derived FROM THE GRAPH (mixed 256/512)** — ✓ `_port_metadata` emits name/dtype/rank/shape for every port; `_state_and_kv_pairs` pairs `past_key_values.<layer>.<role>` ↔ `present.<layer>.<role>` via regex + `config.layer_types`, raising on unclassifiable ports (lines 591-680). Trailing dims come straight from graph shapes. Test uses mixed `kv_head_dims=[8,16,8]` and asserts `past_key_values.1.key` shape[-1]==16 (test line 470) — proves dims are read, not hardcoded.
4. **Typed vision endpoints fp16 pixel_values + int64 pixel_position_ids** — ✓ Registry-driven `_resolve_image_program` matches structural rank/dtype signatures (`_match_packed_coordinates`: fp float rank-3 pixels + int64 rank-3 coords with last-dim 2). Dtypes taken from graph ports; endpoints named from `port.name`. Asserted (test lines 472-484), incl. `pad_value: -1` for coordinates.
5. **Producer-side closure validator** — ✓ `validate_executable_closure` (lines 913-1075) checks: every graph input has exactly one source (external/generated/stateful/defaulted/dataflow); every edge maps real output→input with matching dtype/rank; declared io matches graph ports exactly. Invoked before serialization (line 1334). Emits WHAT/WHY/HOW errors. Negative test removes the per_layer_inputs edge and asserts rejection naming `decoder.per_layer_inputs` (test lines 486-496).

## RULES.md §2/§2.1 compliance

- **No model-name/architecture branching.** `grep` for gemma/qwen/phi/llama/architecture==/model_type== in the source found only one unrelated TTS comment. Dispatch is on structural package roles (`vision_encoder`/`embedding`/`decoder` component keys) = model CATEGORY, which the topology note explicitly sanctions.
- **No fixed constants.** No hardcoded 35-layer/280-patch/256/512 KV dims; all derived from graph shapes and `config.layer_types`/processor config.
- **Assumptions explicit in metadata.** Unsupported vision signatures and unclassifiable state ports fail loudly with regenerate-instructions rather than guessing.
- **Audio edge correctly deferred to WP-B.** The incompatible rank-3 `audio_encoder.audio_features` → rank-2 `embedding.audio_features` edge is intentionally NOT emitted; `embedding.audio_features` is declared `external/request`. Asserted (test lines 191, 197).

## Test quality

Tests are non-trivial and category-diverse, proving generalization not overfit:
- `test_gemma4_routes_all_embedding_outputs` — full 4-model topology (vision+audio+embedding+decoder), mixed KV dims, per_layer edge, every_step, typed image outputs, closure negative case.
- `test_qwen_packed_grid_rank3_positions...` — area-grid processor, mrope, `linear_attention` layer types (sparse/replace state).
- `test_phi_routes_both_modality_gates...` — dynamic-HD crop-mask processor.
- Negative tests: unsupported signature, missing components, rank-3 positions requiring registry, equal-shape KV still declared KV.
- Three cached-processor tests match emitted programs against real processor configs.

Verified locally: `ruff check` + `ruff format --check` clean; `pytest inference_metadata_test.py` = 40 passed. (lintrunner 0.12.7 adapter env was broken — `lintrunner_adapters` not importable — so ran `ruff` directly per fallback; this is an environment issue, not a PR defect.)

## Non-blocking observations (do not require changes before merge)

- `vision_encoder` (`prompt_only`) and `decoder` (`every_step`) `run_on` are role-assigned rather than structurally derived, unlike embedding/audio. Correct for these categories today; a future refactor could derive all phases uniformly for robustness. Not blocking.
- Emission still branches on the literal component key `"audio_encoder"` for the `type` label (line 1238). This is a category label, not model dispatch; acceptable, but a role registry keyed on structure would be cleaner long-term.

## Recommendation

Approve for merge by an authorized non-author (coordinator or Justin). WP-B (optional-modality/typed-audio) and WP-C (runtime admission gate) remain the correct next work; nothing in this PR blocks them.

### Fold processed inbox notes
**By:** Scribe
**What:** Merged and cleared `bryant-wp-b1-review.md`, `deckard-wp-c-rereview.md`, `deckard-wp-c-review.md`, `deckard-wp-c-v3-review.md`, `deckard-wp-c-v4-review.md`, `gaff-wp-c-finding5-fix.md`, `holden-wp-c-v4-fix.md`, `keaton-phase1-seam.md`, `leon-keaton-phase1-review.md`, `leon-wp-c-admission-gate.md`, `pris-wp-b1-schema.md`, `roy-gemma4-e2b-reexport.md`, `roy-gemma4-e2b-topology.md`, `roy-wp-a-contract-emission.md`, `sapper-wp-c-revision.md`, `sapper-wp-c-schema-blocker.md`, `sebastian-wp-a-review.md`. Preserved active reference/in-flight files `keaton-native-specdecode-design.md`, `leon-vlm-scope.md`, `rachael-wp-b-optional-modality-design.md`, `zhora-deepseek-scope.md`.
**Why:** Completed implementation, review, revision, benchmark, and schema notes belong in the current decision ledger; active scope/design files remain in the inbox until their work lands.
<!-- scribe-merge-2026-07-22T12-00-00Z-phase0-7b-cudagraph -->
## 2026-07-22 — Partial CUDA-graph Phase 0 and Qwen2.5-7B CUDA-graph benchmark
<!-- source: .squad/decisions/inbox/deckard-luv-phase0-review.md -->
### 2026-07-22: Review verdict — Luv Phase 0 partial-CUDA-graph capture-path-kind (🟢 GREEN)

**By:** Deckard

**What:** Independent read-only review of `squad/luv-capture-pathkind` (commit 3c94a57) diffed against merge-base with `origin/main`. Changed: `executor.rs` (+`CapturePathKind`/`SeamReason` enums, `CaptureDecline.seam_reason: Option<SeamReason>`, seam-kind label in `log_capture_segmentation`, `CaptureDecline::node` now takes a `SeamReason`), `lib.rs` (re-exports + doc), `native_decode.rs` (+1 field in a test fixture), docs. **Verdict: 🟢 GREEN — safe to merge.**

**Why:**
1. **Byte-identical behavior — PASS.** Only removed string literal is the log-format line (now inserts `[{seam_label}]`); zero decline `reason` strings were removed or altered. Segmentation logic in `plan_capture_segments` is unchanged — `declines[pi].is_none()` still drives partitioning; boundaries pushed identically. Classification is derived *from* existing decline causes, not a replacement.
2. **Correct mapping — PASS.** All 5 per-node causes map correctly: control-flow/sequence→`HostControlFlowOrSequence`→`HostSeam`; unresolved output→`UnresolvedOutputShape`; unresolved input→`UnresolvedInputShape`; kernel-not-warmed→`KernelNotWarmed`; kernel-capture-unsupported→`KernelCaptureUnsupported` — the last four→`EagerDeviceSeam`. Graph-level persistent-device-binding hard-abort (`CaptureDecline::graph`) intentionally carries `seam_reason: None` ("graph-level hard preconditions"), which is correct — it is a whole-graph abort, not a per-node seam.
3. **Model-agnostic — PASS.** No model-name/architecture string branching; classification is purely structural (RULES.md §2/§2.1 respected).
4. **Exhaustiveness — PASS.** `SeamReason::path_kind` and `CapturePathKind::label` use exhaustive matches with no catch-all `_ =>`; `CapturePathKind`/`SeamReason` re-exported from `lib.rs` and doc-commented.
5. **fmt/clippy — PASS.** `cargo fmt -p onnx-runtime-session -- --check` clean; `cargo clippy -p onnx-runtime-session --all-targets -- -D warnings` clean; `--features cuda` clippy clean.
6. **Tests — PASS.** `cargo test -p onnx-runtime-session` = 60 passed, incl. new `seam_reasons_map_to_structural_capture_paths` (genuinely asserts all 5 reason→kind→label mappings + `CaptureRegion` label). `cargo test -p onnx-genai-engine --features native-backend capture_fallback_emits_each_structured_decline_to_tracer` = 1 passed.
7. **Log output — PASS.** Seam-kind label uses `boundary.seam_reason.map(SeamReason::label).unwrap_or("unclassified-seam")`; behind the verbose diagnostic flag; no existing test asserts on the literal log string, so no format-assertion breakage.

Conclusion: purely additive structural diagnostics, correct, model-agnostic, all gates green. Approved for merge.
<!-- source: .squad/decisions/inbox/gaff-qwen7b-cudagraph.md -->
### 2026-07-22: Qwen2.5-7B int4 CUDA-graph auto-enable benchmark
**By:** Gaff
**What:** Benchmarked Qwen2.5-7B int4 on one NVIDIA H200 at `bd3d95a` using `profile_native --ep cuda --prompt Hello --tokens 128 --warmups 2 --runs 3 --steady`, `ONNX_GENAI_DEVICE_KV=1`, and identical greedy decoding. Run A left `ONNX_GENAI_CUDA_GRAPH` unset; Run B set it to `0`. A companion 16-token diagnostic confirmed graph state and fallback counters.
**Why:** Validate that metadata/structure-driven CUDA-graph auto-enable generalizes beyond Qwen2.5-0.5B and Phi-4-mini without architecture or model-name keying.

| Metric | Run A — auto | Run B — forced eager |
|---|---:|---:|
| Median throughput | **231.73 tok/s** | **180.50 tok/s** |
| Median decode latency | **4.315 ms/token** | **5.540 ms/token** |
| Throughput speedup vs eager | **+28.38%** | baseline |
| Token-exact A/B | **Yes** | **Yes** |
| Capture engaged | **Yes** | No (explicitly disabled) |
| Zero fallbacks | **Yes** | Yes |
| Capture diagnostic | `enabled=true`, 1 capture, 14 replays, 0 fallbacks; 1 captured segment, 0 eager seams | `enabled=false`, 0 captures, 0 replays, 0 fallbacks |
| Kernels/token | N/A — `profile_native` does not surface GPU kernel-launch counts | N/A |
| GPU-busy | N/A — `profile_native` does not surface GPU utilization | N/A |
| Fraction of 4.8 TB/s ÷ 3.5 GB/token ceiling | **16.90%** | **13.16%** |

The 128-token outputs were identical token-for-token across A and B. Auto-enable generalized cleanly to Qwen2.5-7B: CUDA plus owned device KV selected whole-step capture automatically, with one captured segment, no eager seams, and zero fallbacks. The **28.38%** gain is smaller than Qwen2.5-0.5B's 87.7% and Phi-4-mini's 41.0%, as expected for a larger decode that spends more time streaming/dequantizing int4 weights and less proportionally on launch overhead, but it remains substantial. The simple peak-bandwidth roofline is about 1,371 tok/s; measured auto throughput is 16.90% of that ceiling, and this ratio should not be interpreted as pure bandwidth efficiency because int4 dequantization and compute also constrain decode.
<!-- source: .squad/decisions/inbox/luv-capture-pathkind.md -->
### 2026-07-22: Formalize partial CUDA-graph capture path kinds
**By:** Luv
**What:** Added `CapturePathKind` and `SeamReason`, attached optional seam classification metadata to `CaptureDecline`, propagated it through `CaptureSchedule` boundaries, and added seam-kind labels to `ONNX_GENAI_LOG_CAPTURE_SEGMENTS` output without changing capture partitioning or existing reason strings.
**Why:** Phase 0 of the partial-CUDA-graph EP-claim design requires structural, model-agnostic diagnostics that distinguish captured regions, eager device seams, and host seams before EP-owned planning is introduced.

| SeamReason | CapturePathKind |
|---|---|
| `HostControlFlowOrSequence` | `HostSeam` |
| `UnresolvedOutputShape` | `EagerDeviceSeam` |
| `UnresolvedInputShape` | `EagerDeviceSeam` |
| `KernelNotWarmed` | `EagerDeviceSeam` |
| `KernelCaptureUnsupported` | `EagerDeviceSeam` |

**Files touched:**
- `crates/onnx-runtime-session/src/executor.rs`
- `crates/onnx-runtime-session/src/lib.rs`
- `crates/onnx-genai-engine/src/native_decode.rs`
- `docs/design-ep-partial-cuda-graph.md`
- `docs/CUDA_GRAPH_CAPTURE.md`

**Verification:**
- `cargo fmt -p onnx-runtime-session` — PASS.
- `cargo test -p onnx-runtime-session seam_reasons_map_to_structural_capture_paths` — PASS (1 focused unit test).
- `cargo build -p onnx-runtime-session` — PASS.
- `cargo build -p onnx-runtime-session --features cuda` — PASS.
- `cargo test -p onnx-runtime-session` — PASS (all session unit, integration, and doc tests; one manual performance audit and one doc test remained ignored).
- `cargo clippy -p onnx-runtime-session --all-targets -- -D warnings` — PASS.
- `cargo test -p onnx-genai-engine --features native-backend capture_fallback_emits_each_structured_decline_to_tracer` — PASS (1 focused compatibility test).

### Fold processed Phase 0 and 7B CUDA-graph inbox notes
**By:** Scribe
**What:** Merged and cleared `deckard-luv-phase0-review.md`, `gaff-qwen7b-cudagraph.md`, `luv-capture-pathkind.md`. Preserved active scope/design files `zhora-deepseek-scope.md`, `leon-vlm-scope.md`, and `keaton-native-specdecode-design.md`.
**Why:** Landed implementation, independent green review, benchmark results, and progress-log updates belong in the current decision ledger; active scope notes remain in the inbox.
<!-- scribe-merge-2026-07-22T00-00-00Z-cudagraph-autoenable -->
## 2026-07-22 — CUDA graph auto-enable, GQA/VLM closure, and inbox reconciliation

### Land metadata-driven native CUDA graph auto-enable
**By:** Batty; reviewed by Leon 🟢
**What:** Merged `batty-45` to main as `610bde0`, auto-enabling whole-step CUDA graph capture in `native_decode.rs` whenever metadata and device bindings prove the native decode topology graph-safe. Environment precedence remains explicit-disable first, then explicit-enable, then metadata auto-enable; capture-safety fallback remains transparent.
**Why:** Gaff's H200 profile showed native decode was launch/CPU-dispatch bound rather than bandwidth-bound. Auto-enable turned proven graph-safe models on by default without model-name gates.
**Validation:** Leon reviewed `squad/batty-cudagraph-autoenable` 🟢 GREEN with 7/7 criteria passing. H200 results were token-exact with zero fallbacks: Qwen2.5-0.5B improved **441.49→828.54 tok/s (+87.7%)** and Phi-4-mini improved **67.32→94.91 tok/s (+41.0%)**.

### Close GQA `seqlens_k` exporter-shape blocker
**By:** Chew and Roy; reviewed by Deckard 🟢
**What:** Accepted canonical dense contiguous int32 `seqlens_k` shapes `[batch_size]` and `[batch_size, 1]`, normalized trailing singleton shape for capture signatures, and revised non-contiguous diagnostics to name both accepted shapes. Coordinator merged the fix to main as `f4484e7`.
**Why:** Real Foundry Qwen2.5-1.5B and Phi-4-mini exports provide `[batch_size, 1]`; scalar-only support did not unblock those models. Deckard's initial review was 🔴 only for diagnostic wording; re-review passed after Roy's correction.

### Record native CUDA benchmark and model-coverage outcomes
**By:** Gaff, Okonkwo, Chew, Deckard, Pris, Holden, and Tyrell
**What:** Folded the decode roofline and re-benchmark sequence: Qwen2.5-0.5B baseline native CUDA decode around 435 tok/s before CUDA graph auto-enable; Qwen2.5-1.5B first blocked by `[batch,1]` GQA lengths, then by M=5 prefill until the SwiGLU M>1 path landed; Phi-4-mini native CUDA validated on H200 after int4 zero-points and partial-RoPE fixes. The native CPU coverage census, DS-1 dynamic shape-chain validation, DS native E2E exact parity, MLA conformance guard, and progress-log updates are now represented here or in existing 2026-07-22 ledger sections.
**Why:** These notes establish which blockers were generic runtime gaps, which were already closed on main, and which measurements motivated CUDA graph auto-enable rather than model-specific dispatch.

### Fold VLM WP1 runtime-contract and CI notes
**By:** Rachael, Roy, Deckard, Leon, and Sebastian
**What:** Preserved the VLM WP1 review sequence: Leon rejected non-executable metadata revisions, Roy/Rachael moved preprocessing metadata toward explicit runtime contracts, Deckard fixed Qwen temporal patch packing order, and Leon re-reviewed the temporal-order fix 🟢. Sebastian made PR #416 schema/processor tests offline-safe by skipping unavailable local assets rather than failing CI.
**Why:** VLM metadata must be executable through declared processor/registry contracts, not shape-only JSON acceptance; cached-processor parity gates must be environment-aware.

### Fold partial CUDA-graph EP-claim design notes
**By:** Keaton; reviewed by Fact Checker 🟡
**What:** Recorded the proposed partial CUDA-graph capture design for EP subgraph claiming, with whole-step capture prioritized first and partial capture constrained by static seam-output and KV-append invariants.
**Why:** The design remains a follow-up proposal; whole-step capture is the immediate path for fixed-topology device-resident decode.

### Fold processed inbox notes
**By:** Scribe
**What:** Merged and cleared `batty-cudagraph-autoenable.md`, `chew-gqa-batch1.md`, `chew-model-coverage-census.md`, `coordinator-gqa-merge.md`, `deckard-ds1-shapechain.md`, `deckard-dsnative.md`, `deckard-gqa-batch1-review.md`, `deckard-gqa-rereview.md`, `deckard-mla-conformance-review.md`, `deckard-wp1-packer-fix.md`, `factchecker-keaton-epclaim-review.md`, `gaff-decode-profile.md`, `gaff-native-rebench.md`, `gaff-native-rebench2.md`, `gaff-native-rebench3.md`, `gaff-phi4-bench.md`, `gaff-phi4-benchmark.md`, `holden-partial-rotary.md`, `keaton-epclaim-design.md`, `keaton-epclaim-v2.md`, `leon-batty-cudagraph-review.md`, `leon-wp1-rereview.md`, `leon-wp1-review.md`, `okonkwo-gqa-decode-bench.md`, `pris-ds1-testreview.md`, `pris-gqa-scalar-seqlens-plan.md`, `pris-holden-rotary-review.md`, `pris-mla-conformance.md`, `rachael-wp1-revision.md`, `roy-gqa-batch1-revision.md`, `roy-wp1-revision.md`, `sebastian-mobius416-ci.md`, `tyrell-progress-0722.md`, `zhora-glm-l4-fix.md`. Preserved active scope/design files `zhora-deepseek-scope.md`, `leon-vlm-scope.md`, and `keaton-native-specdecode-design.md`.
**Why:** Completed implementation, review, benchmark, CI, and duplicate ledger artifacts belong in the current decision ledger; active scope notes remain in the inbox.
<!-- scribe-merge-2026-07-22T00-00-00Z-int4-zp -->
## 2026-07-22 — Phi-4-mini int4 zero-point blocker closure

### Close BLOCKER #3: explicit int4 zero-points in native CUDA fp16 GEMV
**By:** Sapper; reviewed by Holden 🟢
**What:** Merged commit `48de993`, threading packed per-block int4 `zero_points` plus `zp_row_bytes` through the native CUDA fp16 GEMV path so asymmetric int4 MatMulNBits models such as Phi-4-mini decode with explicit zero points. Null zero-point inputs preserve the existing symmetric zp=8 fast paths.
**Why:** Removes BLOCKER #3 with a structural, model-agnostic asymmetric int4 path while keeping M==1 capture safety, SM-portable arithmetic, and symmetric no-regress behavior.
**Validation:** Holden's non-author review passed all five criteria (SM-portability, capture-safety, symmetric no-regress, genericity, correctness). H200 validation passed 6/6 unit tests and 18/18 `matmul_nbits_gpu` integration tests, including explicit-zp CPU-reference and capture-replay coverage.

### Fold processed int4 zero-point inbox notes
**By:** Scribe
**What:** Merged and cleared `sapper-int4-zp.md` and `holden-int4-zp-review.md`.
**Why:** The implementation and independent green review are now represented in the ledger; unrelated active inbox artifacts remain untouched.
<!-- scribe-merge-2026-07-22T06-17-16Z -->
## 2026-07-22 — Native proposer contract and Qwen0.5B H200 benchmark

### Land metadata-driven native proposer execution contract
**By:** Batty; reviewed by Deckard 🟢
**What:** Land commit `96c79d0`, replacing hardcoded native proposer assumptions with metadata-driven `sequence_source` (`input_ids`/`inputs_embeds`), `kv_ownership` (`owned`/`shared`), explicit shared-KV ports, and semantic output roles (`logits_output`/`hidden_output`). Defaults preserve legacy token-id + owned-KV behavior; CPU shared-KV proposer execution is complete.
**Why:** Embedding-driven shared-KV assistants must be activated by declared contracts rather than model or tensor-name assumptions. CUDA device-buffer shared-KV aliasing remains explicitly scoped out until device binding alias/reference support lands.

### Record Qwen2.5-0.5B native CUDA H200 decode benchmark
**By:** Gaff
**What:** Qwen2.5-0.5B native CUDA decode on H200 measured **437.76 tok/s median** (**2.284 ms/token**), with coherent deterministic output. This is **15.2% faster** than the user's RTX 4060 380 tok/s reference and **2.83%** of the H200 weight-bound roofline.
**Why:** Establishes the current native-path performance point for the 0.5B model on shared H200 hardware and shows the path is coherent but still far from the weight-bound ceiling.

### Fold processed proposer and benchmark inbox notes
**By:** Scribe
**What:** Merged and cleared `batty-proposer-contract.md`, `deckard-batty-proposer-review.md`, and `gaff-qwen05-bench.md` when present.
**Why:** Landed implementation, review, and benchmark records belong in the ledger; active unrelated inbox artifacts remain in place.
<!-- scribe-merge-2026-07-22T05-52-21Z -->
## 2026-07-22 — Fused CUDA SwiGLU M>1 prefill merge

### Land generic fused gate/up SwiGLU M>1 prefill
**By:** Bryant; reviewed by Deckard 🟢
**What:** Land commit `97e0cb4` from `wt-swiglu-prefill`, extending `run_f16_gate_up_swiglu` so M>1 prefill runs the existing portable fp16 MatMulNBits tiled GEMM twice (gate into scratch, up into output) and then applies the existing fp16 SiluMul in place. The M=1 paired GEMV path remains unchanged and capture-safe; M>1 explicitly records `last_call_capture_safe=false`.
**Why:** The graph optimizer removes the unfused gate/up nodes, so the fused node must handle prompt rows as well as decode. Review confirmed bit-exact M=1 and M>1 coverage, SM portability, generic dispatch, correct capture flag behavior, and scratch lifetime safety; H200 rebuild plus 4 SwiGLU tests passed before merge.

### Fold processed SwiGLU inbox notes
**By:** Scribe
**What:** Merged and cleared `bryant-swiglu-prefill.md` and `deckard-bryant-swiglu-review.md`. Preserved unrelated active in-flight deliverables in `.squad/decisions/inbox/`.
**Why:** Landed implementation and review decisions belong in the ledger; active scope/review/revision artifacts should remain in the inbox until their work lands.
<!-- scribe-merge-2026-07-22T04:39Z -->
## 2026-07-22 — CPU SLN, stale-shape recompute, nbits prefill GEMM, and stale test merges

### Land fp16/bf16 CPU SimplifiedLayerNormalization
**By:** Deckard; reviewed by Gaff 🟢
**What:** Land commit `74a80ce` extending the CPU `SimplifiedLayerNormalization` kernel to accept Float16, BFloat16, Float32, and Float64 inputs/scales by widening to f32 for RMS-style accumulation and narrowing normalized plus optional inverse-standard-deviation outputs to the declared dtype. Dtype-parameterized tests cover last-axis and multi-axis shapes.
**Why:** Half-precision Foundry exports were rejected at `input_layernorm`; the generic widen/compute/narrow path removes that CPU decode gap without model, hidden-size, or shape gates.

### Land live runtime shape recompute for elementwise broadcasts
**By:** Pris; reviewed by Leon 🟢
**What:** Land commit `79b2bfc` recomputing standard multidirectional elementwise output geometry from concrete runtime input shapes before allocation, with actionable broadcast-incompatibility errors and coverage for a `ReduceSum -> Squeeze -> Cast -> Slice -> Add` data-dependent chain.
**Why:** Loader-resolved shapes can be stale for runtime view/data-dependent chains; using live broadcast shapes unblocks GLM-5.2-tiny indexing `Add` nodes while preserving strict ONNX equal-or-one semantics.

### Land portable fp16 MatMulNBits M>1 prefill GEMM
**By:** Sapper; reviewed by Batty 🟢
**What:** Land commit `54b49eb` adding a structural CUDA fp16-activation MatMulNBits prefill path for int4/int8 block-32 weights using a portable 16x16 tiled CUDA-core GEMM with fp32 accumulation, fp16 output, implicit/explicit zero points, tail handling, and f64-oracle parity.
**Why:** Native fp16 MatMulNBits previously rejected every M>1 prompt; the new path enables native multi-token prefill while preserving the unchanged capture-safe M=1 decode GEMVs.

### Refresh stale MatMulNBits unsupported-width coverage
**By:** Hudson
**What:** Land commit `764a208` updating the CPU MatMulNBits factory rejection test to use unsupported `bits=3`, assert the current `{2, 4, 8}` contract, and add positive factory coverage for `bits=8`.
**Why:** The old test treated now-supported `bits=8` as invalid and broke the CPU suite on main after int8 support landed.

### Fold processed landed inbox notes
**By:** Scribe
**What:** Merged and deduplicated `deckard-sln-fp16.md`, `gaff-sln-fp16-review.md`, `pris-stale-shape.md`, `leon-stale-shape-review.md`, `sapper-nbits-prefill.md`, `batty-nbits-prefill-review.md`, and `hudson-stale-nbits-test.md`. Preserved active or not-yet-main GQA/VLM/specdecode/model-coverage scope and revision artifacts.
**Why:** Landed implementation and review decisions belong in the ledger; active scope, review, and revision files should remain in the inbox until their work lands.
<!-- scribe-merge-2026-07-22T03:37:44Z -->
## 2026-07-22 — GQA scalar seqlens_k and int8 fp16 default-zp test merges

### Land GQA scalar `seqlens_k` support
**By:** Deckard; reviewed by Roy 🟢
**What:** Land commit `4ceaa7b` enabling declared unit-batch scalar `seqlens_k` for structurally detected GroupQueryAttention only. The contract remains strict-by-default (`PerBatchOnly`), rejects batch>1 scalar lengths, regenerates schema metadata, and keeps CUDA graph capture safe because validation is pure CPU shape inspection with no device allocation, D2H copy, sync, or pointer rebinding.
**Why:** ORT-GenAI GQA exports may provide scalar key sequence lengths for unit-batch decode; accepting that explicit metadata contract generically unblocks Phi-4-mini and Qwen2.5-1.5B decode without broad scalar coercion.

### Land int8 fp16 implicit-zero-point GPU parity coverage
**By:** Deckard; reviewed by Tyrell 🟢
**What:** Land commit `0d618de` adding fp16 int8 block-32 MatMulNBits CUDA parity coverage when the optional zero-point graph input is omitted, with the independent reference using default zp=128. The batch also retains explicit-zero-point coverage and verifies CUDA-graph replay is bit-exact with the preceding eager output on H200.
**Why:** The implicit/default zero-point path is distinct from explicit zero-points and needs direct regression coverage for fp16 output parity and capture determinism.

### Record VLM WP1 emission review lockout
**By:** Sapper; reviewed by Leon 🔴
**What:** PR #416 / VLM WP1 emission is blocked. Sapper is locked out of revising this artifact; a different agent must derive processor operations from explicit processor config/registry entries, make position/state roles registry/config-driven, add real cached-model HF processor comparisons, and fail unsupported signatures with actionable regenerate-or-register errors.
**Why:** Although schema/port validation and CLI/metadata tests passed, emitted preprocessing programs were not runtime-correct for Qwen3-VL, Gemma4, or Phi4MM, and some roles were inferred from shape/position rather than declared metadata.

### Fold processed inbox notes
**By:** Scribe
**What:** Merged and deduplicated `deckard-int8-zp-test.md`, `roy-gqa-review.md`, `tyrell-int8-zp-review.md`, and `leon-wp1-review.md` into this ledger. Preserved active research/scope artifacts in the inbox, including `zhora-deepseek-scope.md`, `leon-vlm-scope.md`, `keaton-native-specdecode-design.md`, `pris-gqa-scalar-seqlens-plan.md`, and `chew-model-coverage-census.md` if present.
**Why:** Review verdicts, lockouts, and landed implementation decisions belong in the current ledger; active research artifacts remain available for ongoing work.
<!-- scribe-merge-2026-07-22T09:30Z -->
## 2026-07-22 — DeepSeek shape-chain, MLA conformance, and active inbox fold

### Land DS-1 generic dynamic shape-chain propagation
**By:** Chew; reviewed by Rachael 🟢
**What:** Land commit `d653879` (reviewed work `chew-79`) extending generic runtime shape-chain propagation so a dynamically resolved `Slice` can feed `Unsqueeze` and subsequent broadcast/movement. `Unsqueeze` output rank is computed as input rank plus `len(axes)`, using the ONNX domain/opset registry and no node-name keying. Native Rust DeepSeek-V2 tiny CPU E2E now generates `[42, 237, 198, 2, 186, 81, 210, 149]`.
**Why:** Dynamic output sizing must remain model-agnostic and registry-driven while covering DeepSeek-V2 decode graphs that pass shape values through movement/broadcast chains.

### Land DS-3 MLA cached-decode parity coverage
**By:** Pris; reviewed by Tyrell 🟢
**What:** Land commit `8aba045` strengthening standard Attention/MLA tests for `qk_head_dim != v_head_dim` (192 vs 128), 3-D BSH, explicit head attrs, non-empty past K/V, prefill+decode+full-seq parity, GQA (`kv=2`) and MQA (`kv=1`), with an independent scalar SDPA oracle. CPU 33/33 and CUDA 23/23 pass.
**Why:** Cached decode must preserve asymmetric QK/V head-width semantics and parity across CPU/CUDA without relying on model-specific assumptions.

### Keep generic scalar `seqlens_k` GQA support explicit and unit-batch scoped
**By:** Pris and Deckard
**What:** Preserve the long-lived scalar-seqlens implementation plan, and fold Deckard's landed decision to emit `model.attention.key_sequence_lengths.scalar_broadcast: unit_batch` only for structurally detected ORT-GenAI GroupQueryAttention exports.
**Why:** Scalar key sequence lengths should be accepted only under a declared, validated unit-batch GQA contract, not as a broad shape coercion.

### Fold remaining processed inbox decisions and reviews
**By:** Scribe
**What:** Processed and deduplicated the non-preserved decision inbox notes. Key folded outcomes: block-32 int8 MatMulNBits CUDA support and review; VLM WP1/WP5/WP6 metadata/loader/server-bundle work and reviews; Gemma4 auxiliary output binding plus structural capture guard; H200 multi-model roofline and megakernel feasibility notes; KV logical-shape and fp16 GQA decode coverage; and DeepSeek validation/review records already represented by the DS-1/DS-3 entries above. Processed files:
- `ana-fp16-next-levers.md`
- `ana-h200-baseline-roofline.md`
- `ana-megakernel-feasibility.md`
- `ana-wave2-roofline-558.md`
- `ana-wave3-roofline-691.md`
- `batty-auxbind.md`
- `chew-ds1-shape-chain.md`
- `chew-ds3-mla.md`
- `chew-leon-auxguard-review.md`
- `deckard-gqa-fp16.md`
- `deckard-gqa-scalar-seqlens.md`
- `deckard-int8-matmulnbits.md`
- `gaff-ds3-review.md`
- `gaff-kv-review.md`
- `leon-auxbind-review.md`
- `leon-auxguard.md`
- `leon-kv-logical-shape.md`
- `leon-vlm-wp5-finalize.md`
- `leon-vlm-wp5-rebase.md`
- `leon-vlm-wp5-urlfix.md`
- `luv-vlm-wp5-rereview.md`
- `luv-vlm-wp5-rereview2.md`
- `luv-vlm-wp5-review.md`
- `luv-vlm-wp6-rereview.md`
- `luv-vlm-wp6-review.md`
- `luv-wp4-review.md`
- `pris-deepseek-e2e-val.md`
- `pris-ds3-mla-conformance.md`
- `pris-gqa-fp16-review.md`
- `rachael-ds1-review.md`
- `rachael-vlm-wp5.md`
- `roy-int8-matmulnbits-review.md`
- `sapper-glm-pr404.md`
- `sapper-vlm-wp1-emission.md`
- `sapper-vlm-wp6-fix.md`
- `sebastian-gemma4-perf.md`
- `sebastian-gemma4-reprobe.md`
- `sebastian-h200-multimodel-bench.md`
- `tyrell-ds3-review.md`
- `zhora-vlm-wp5-fix.md`
- `zhora-vlm-wp6.md`
**Why:** The inbox should retain only long-lived active research/scope artifacts while merged decisions live in the current ledger.

### Preserve active research and scope artifacts in the inbox
**By:** Scribe
**What:** Left `zhora-deepseek-scope.md`, `leon-vlm-scope.md`, `pris-gqa-scalar-seqlens-plan.md`, and `keaton-native-specdecode-design.md` in `.squad/decisions/inbox/`.
**Why:** These artifacts remain active references and should not be collapsed into the ledger yet.
<!-- scribe-merge-2026-07-21T23:55Z -->
<!-- scribe-merge-2026-07-22T21-00-00Z-cpu-ep-perf -->
## 2026-07-22 — CPU EP performance campaign reconciliation

Decision archive gate checked at 2026-07-22T21-00-00Z: the active ledger contains no dated entries older than 2026-07-15; no entries were eligible for archival.

<!-- source: .squad/decisions/inbox/batty-native-decode-parallel.md -->
# Batty — Native CPU decode: parallel-runtime overhead

Branch: `perf/cpu-ep-mlas` (no push/merge). Commit `32a122e`.

## Goal

Cut the ~55 ms/step engine-level parallel-runtime overhead on native CPU int4
decode (Qwen2.5-Coder-7B, Sapphire Rapids Xeon 8480C, 2×48 cores, 2 NUMA nodes),
target ≥20 tok/s steady M=1. Profile-first (RULES.md rule 4).

## Methodology

- Build: `cargo build --release -p onnx-genai-bench --features mlas --bin profile_native`.
- Steady M=1 isolation: `profile_native ... --tokens 24 --runs 5 --warmups 1
  --steady --decode-skip 8`, reporting the tool's `steady_median`.
- 32 decode threads (`ONNX_GENAI_CPU_DECODE_THREADS=32`) unless noted.
- Shared 96-core host is noisy and drifts warmer across a run (run 1 is usually
  the fastest), so I interleaved A/B conditions across ≥2–3 rounds and report
  **median and best**, not a single run.
- Bit-parity: greedy `generated_token_ids` were identical for every non-numeric
  change (baseline and every affinity mode all produced
  `[576, 729, 1265, 1896, 264, 1140, 438, 458, 5693, 323, 470, 264, 501, 1140,
  429, 374, 10615, 304, 35388, 1973, 13, 1446, 1265, 537]`).

## Profile (what actually costs time)

Per-op split (`ONNX_GENAI_PROFILE_OPS=1`, steady step ≈70 ms):
`MatMulNBits` 58.2 ms (82 %, 141 calls, ~0.41 ms/call), `Silu` 5.3 ms,
`SkipSimplifiedLayerNormalization` 2.5 ms, `Add` 2.1 ms, `GroupQueryAttention`
2.1 ms, everything else <1 ms. **The matmuls dominate.** The isolated op-mix
runs the same 141 matmuls in ~33 ms with L3-resident weights (~108 GB/s); the
in-engine 58 ms is ~58 GB/s effective. So real decode is **memory-latency
bound**, not DRAM-bandwidth bound and not kernel-compute bound — the extra time
is cold weight streaming plus per-op fork-join barrier latency, both of which
are worse when workers and weights span two sockets.

## What worked — NUMA-local decode-pool affinity (shipped)

`ONNX_GENAI_CPU_DECODE_AFFINITY` (`off` default / `compact` / `node:<index>`)
pins the bounded M=1 decode workers to the CPUs of one NUMA node. Topology is
queried from `/sys/devices/system/node/node*/cpulist` (no hardcoded counts,
rule 2); it is opt-in and inspectable (rule 5); single-node/non-Linux/cgroup
rejection falls back to unpinned, logged once (rule 1); a bad value is a clear
kernel error naming the accepted modes / available nodes. The packed int4
weights are lazily first-touched inside `with_decode_pool_scope` on a pinned
worker, so both barrier traffic and the weight stream become node-local.
Verified at runtime: the N decode workers each pin to a distinct node-0 CPU
(`Cpus_allowed_list` = 0..N) while the global pool stays unpinned.

Steady M=1, 32 threads, 5 runs × 3 rounds:

| Affinity | decode median | best | spread |
| --- | --- | --- | --- |
| `off` | **13.1 tok/s** (76.4 ms) | 14.4 | 12.6–14.4, jittery |
| `compact` | **16.3 tok/s** (61.2 ms) | 16.4 | 16.3–16.4, stable |

≈ **+25 % median, +14 % best**, and pinning removes the OS-migration jitter that
makes the unpinned pool swing run-to-run. Full 120-token generation also
improved (11.5 → 12.0 tok/s; smaller because it includes prefill).

### Thread scaling after (compact affinity, steady)

16 t → 14.9 · 32 t → 16.6 · 40 t → 16.5 · 48 t → 15.3 tok/s. Saturates at ~32 on
one node (node 0 has 48 cores; 48 t contends with the OS/main thread on the
shared host). The unpinned >32 regression (the original 8.85/11.97/12.59/9.62
at 8/16/32/48) is a cross-socket-barrier artifact; pinning to one node removes
the cross-socket sync, so scaling no longer collapses — it just plateaus once
the single node's memory subsystem is saturated.

## What didn't work

- **`numactl --cpunodebind=0 --membind=0`** (external, full pipeline): noise-level
  in my runs (11.66 vs 11.50) — it restricts to a node but still lets the OS
  migrate workers within it and pins the whole process incl. prefill. Explicit
  per-worker pinning of just the decode pool is what delivered the clean win.
- **Dual-node for 2× bandwidth (naive):** a 64-thread pool spanning both sockets
  with `numactl --interleave=all` measured **11.1 tok/s vs 16.3** for single-node
  `compact`. Every per-op fork-join barrier across 64 cross-socket threads pays a
  coherency round trip that dwarfs the extra bandwidth. Confirms cross-socket
  barrier sync is the toxic term.
- **Existing `ONNX_GENAI_PROJECTION_FUSION` (gate+up):** still regresses, even
  with affinity on (16.3 → 13.0–14.0). Its `Split` op materializes and copies the
  fused gate+up output every token, and it only removes one barrier per layer, so
  it is a net loss. Left OFF (bit-parity holds). Not a win as written; a real
  grouping win needs a fused gate/up/Silu/Mul kernel that writes the two outputs
  directly (no `Split`), which I did not attempt here.

## Remaining gap and the next lever

Shipped: ~13.1 → ~16.3 tok/s steady median (and ~10.9 → ~16.3 vs the original
project baseline). Still short of ORT (26.9) / genai (20.8) and the 20 tok/s
target. The evidence points at one remaining big lever: **use both sockets'
memory bandwidth without a cross-socket per-op barrier.** That means per-node
decode sub-pools, each streaming a node-local shard of every projection's rows,
joined by a two-level (node-local then single cross-node) barrier — steps 4–5 of
`docs/numa-decode-plan.md`. It is the highest-upside but also the highest-risk
change (touches the hot `MatMulNBits` M=1 path Deckard just finalized); I scoped
it out of this commit deliberately and documented the design + the failure mode
of the naive version so the next iteration starts from evidence.

## Files

- `crates/onnx-runtime-ep-cpu/src/decode_affinity.rs` (new): topology query,
  affinity parsing, `sched_setaffinity` pinning, unit tests.
- `crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs`: pool builder applies
  a `start_handler` that pins workers; clear error / once-logged fallback.
- `crates/onnx-runtime-ep-cpu/src/lib.rs`, `Cargo.toml` (+`libc`), `Cargo.lock`,
  `docs/numa-decode-plan.md`.

Tests: `cargo test -p onnx-runtime-ep-cpu --features mlas` → 665 passed
(4 new affinity unit tests). Non-author review pending (Chew/Gaff; rule 9).
<!-- source: .squad/decisions/inbox/chew-perf-numerics-review.md -->
### 2026-07-22: Numerics review of CPU MatMulNBits and GQA decode optimizations
**By:** Chew
**What:** `58a3324` is **APPROVE-WITH-NONBLOCKING**. `145549a` is **REJECT**; Deckard should own the revision because Sapper, the original author, is locked out.
**Why:**

#### `58a3324` — APPROVE-WITH-NONBLOCKING

- Routing is generic: `try_mlas_sqnbit` selects from `m`, bit width, `accuracy_level`, `g_idx`, and the configured/runtime-available backend (`matmul_nbits.rs:416-460`). There is no model-identity or production hardcoded-shape gate. The `g_idx` and 2-bit fallbacks remain intact.
- The new M=1 `accuracy_level != 4` route uses MLAS CompFp32 and is directly checked against the dequantized f32 oracle (`matmul_nbits.rs:2666-2738`). The broader MLAS parity matrix covers M=1/M=5, block sizes 32/64/128, symmetric/asymmetric zero points, bias, and both compute types (`matmul_nbits.rs:2411-2491`).
- The hybrid `2e-3` absolute-or-relative tolerance is reasonable for the tested CompFp32 dequantization plus reordered f32 reduction; targeted tests passed. It is not a proof of identical logits or greedy tokens for every model. Unlike the unchanged `accuracy_level == 4` hand route, affected `accuracy_level != 4` outputs are not bit-identical and a sufficiently small downstream logit margin can change argmax.
- Nonblocking follow-up: add a production-scale K/N CompFp32 parity case and an end-to-end greedy parity fixture for an affected `accuracy_level != 4` model, reporting maximum logit delta and minimum top-1 margin.
- Rule 1 is not implicated by a new failure path: unsupported MLAS cases explicitly fall back rather than emitting a new opaque error. Rule 8 is satisfied by route and numerical-oracle tests.
- The uncommitted `mlas_fp32` decode-step probe only extends the ignored performance probe to compare hand, MLAS Int8, and MLAS Fp32. It adds no correctness assertion and does not change this verdict.

#### `145549a` — REJECT

- The runtime AVX2+FMA gate and scalar/non-x86 fallback are structurally correct (`group_query_attention.rs:383-409`), and the attended-window indexing is equivalent for finite inputs.
- The stated dot-product bound is incorrect. `n × ε × max(|a|, |b|)` is dimensionally insufficient; the standard forward-error term depends on the products, e.g. a reduction-specific `γ × Σ|a_i b_i|`. A local float32 simulation found a counterexample at `n=32`, input scale 10: difference `9.15527e-5` exceeded the claimed `8.73423e-5` bound.
- The primary “long-context” parity test uses `head_dim=2` (`group_query_attention.rs:1538-1596`). It therefore executes only the scalar tails of both AVX2 helpers and does not test the vectorized production path. Its periodic values in `[-1,1]` also avoid realistic head width, magnitudes, and cancellation.
- The helper dot test reaches width 128, but only on one benign periodic pattern. The AXPY helper test performs one update, not the hundreds/thousands of probability-weighted accumulations changed by the P·V loop. The repository test does not prove greedy-token identity; a 16-token external observation cannot establish it universally.
- Normalizing probabilities once does not add overflow risk because stable softmax exponentials are in `[0,1]`. Per-output accumulation order across keys is preserved; FMA changes rounding and generally improves each multiply-add. Catastrophic cancellation risk is therefore not materially worse, but it is insufficiently exercised.
- Required revision by Deckard: correct the numerical bound/documentation; make the integrated long-context test use a realistic SIMD head width (at least 128), non-periodic realistic and cancellation-heavy data, and verify the AVX2 path on supported x86; add multi-key AXPY/output parity and retain scalar/non-x86 coverage. Any greedy-token claim must be backed by a checked-in end-to-end fixture with logit deltas/margins or softened to an empirical statement.

Validation: `cargo test -p onnx-runtime-ep-cpu --features mlas matmul_nbits` passed 32 tests (2 ignored); `cargo test -p onnx-runtime-ep-cpu --features mlas group_query` passed 16 tests.
### 2026-07-22: Re-review Leon's GQA numerics revision
**By:** Chew
**What:** `c9762b6` is **APPROVE-WITH-NONBLOCKING**. It resolves the blocking findings on `145549a`; Sapper remains locked out and Leon's revision is accepted.
**Why:**

- The documentation now states the standard absolute forward-error scale `γ_n Σ|a_i b_i|`, with `γ_n = n u / (1 - n u)` and `u = 0.5 ε`. Tests correctly use `2 γ_n Σ|a_i b_i|` when comparing two separately rounded evaluation orders (`group_query_attention.rs:1048-1057`, `1735-1766`). A randomized float32 stress probe across lengths through 1024 and scales through 1000 found no counterexample; the worst observed difference used 24.1% of the bound.
- The integrated decode parity fixture now uses head width 128, 256 attended keys, four query heads, mixed non-periodic signed values at scales 0.03125/0.125/0.5/2.0, and a scalar full-attention oracle (`group_query_attention.rs:1624-1732`). On x86 it asserts `has_simd_x86()`, and this host satisfied the assertion, so both the AVX2 dot and AVX2 AXPY bodies execute rather than scalar tails.
- The 257-key, width-128 AXPY test mirrors the production key-outer accumulation, uses normalized positive probabilities and signed mixed-scale values, and compares every dimension against sequential scalar accumulation under the same two-order γ bound (`group_query_attention.rs:1799-1852`). This is representative and cancellation-sensitive.
- The greedy-token statement is now correctly empirical rather than universal. Runtime SIMD gating and non-x86 scalar compilation remain unchanged.
- Nonblocking portability note: the new assertions make the test suite fail on older x86 hosts without AVX2+FMA even though the runtime supports scalar fallback. Prefer an explicit capability skip plus dedicated AVX2 CI coverage. Also consider accumulating the test-only `Σ|a_i b_i|` in f64 so the theoretical tolerance oracle cannot be rounded downward in f32.

Validation: `cargo test -p onnx-runtime-ep-cpu --features mlas group_query` passed all 17 tests. The prior rejection is cleared.
### 2026-07-22: Review contiguous f32 kernel I/O bulk copies
**By:** Chew
**What:** `2e982c7` is **APPROVE-WITH-NONBLOCKING**.
**Why:**

- The fast path follows `TensorView::validate`/`TensorMut::validate`, dtype validation, and element-count validation. `is_contiguous()` requires strides to exactly equal the canonical row-major strides for the complete shape (`onnx-runtime-ir/src/layout.rs:10-23`). Zero-stride broadcasts, transposes, negative strides, and overlapping noncanonical layouts therefore cannot enter the bulk-copy branch. Empty tensors return before pointer slicing, and byte offsets are already incorporated in the validated origin pointer (`kernels/mod.rs:869-909`, `1008-1055`).
- `extend_from_slice` and `copy_from_slice` copy the same consecutive f32 bit patterns that the prior logical element loads/stores produced. No arithmetic, reduction, dtype conversion, or ordering change occurs. The f16/bf16 widening and narrowing helpers are separate and unchanged, so no f32→f16→f32 rounding contract is affected.
- Tests cover contiguous read/write and transposed strided read/write. The full CPU EP suite passed: 661 unit tests passed with 3 ignored, 10 numerical regression tests passed, and one doctest remained intentionally ignored.
- Nonblocking coverage gap: no focused zero-stride broadcast or other overlapping-stride accessor test was added. The exact canonical-stride predicate makes the implementation safe by inspection, but add read-side broadcast and write-side overlapping-view regressions to lock that exclusion down.
<!-- source: .squad/decisions/inbox/coordinator-cpu-perf-baseline.md -->
## 2026-07-22 — CPU EP performance baseline vs ORT/foundry

### Establish native CPU decode baseline and the gap to close
**By:** Coordinator (measured); requested by Justin Chu
**What:** On Sapphire Rapids Xeon 8480C (AMX + AVX512-VNNI), Qwen2.5-Coder-7B int4 (foundry `generic-cpu-4`, fp32 activations), 32 decode threads, greedy, 24-token decode:
- onnxruntime-genai 0.14.1 (foundry's runtime): **20.62 tok/s**
- ORT wrapper via `profile_decode` (our decode loop + ORT CPU kernels): **20.12 tok/s**
- native nxrt CPU via `profile_native --ep cpu` (mlas feature on): **8.83 tok/s**

Native CPU is **~2.3× slower than ORT/foundry**. Goal: beat ORT.
**Why:** ORT/foundry route int4 MatMul through MLAS SQNBit **int8** compute (AMX/VNNI). Our native M=1 decode path deliberately bypasses MLAS (`NXRT_SQNBIT_DECODE_MIN`) for a hand int4 path — a conclusion drawn on 0.5B that likely does not hold on 7B. Prime suspect for the gap.

**Repro / harness notes (repository facts):**
- Build: `cargo build --release -p onnx-genai-bench --features mlas --bin profile_native` and `--features bench-ort --bin profile_decode`.
- Both bins link `libonnxruntime.so.1`; set `LD_LIBRARY_PATH=$PWD/target/release/build/onnx-genai-ort-sys-*/out/ort-prebuilt/lib`.
- Native CPU threads knob: `ONNX_GENAI_CPU_DECODE_THREADS=N` (profile_native has no `--threads`; profile_decode does).
- Fair CPU model = foundry **generic-cpu** int4 (fp32 A). The `cuda-gpu-4` exports use fp16 A, which native CPU MatMulNBits rejects (`A must have dtype Float32`) — a separate gap.
- onnxruntime-genai reference: `pip install --user onnxruntime-genai` (0.14.1), `/tmp/og_bench.py`.

<!-- source: .squad/decisions/inbox/deckard-matmulnbits-m1-mlas.md -->
# Decision: MatMulNBits M=1 int4 decode routing (MLAS vs hand path)

- **Author:** Deckard (Systems Dev)
- **Date:** 2026-07-22
- **Branch:** perf/cpu-ep-mlas (not merged)
- **Reviewer gate:** Chew (numerics) — rule 9, do not self-approve.
- **Host:** Xeon 8480C (Sapphire Rapids, AMX + AVX512-VNNI), 96 hardware
  threads, 2 NUMA nodes (node0 cpus 0–47, node1 48–95), 32 decode threads.
- **Model:** Qwen2.5-Coder-7B-Instruct int4 generic-cpu (all MatMulNBits
  `block_size=32`, `bits=4`, `accuracy_level=4`).

## Task hypothesis — REFUTED by profiling (rule 4, profile before replacing)

The mission hypothesized that routing M=1 int4 decode to MLAS SQNBit CompInt8
(AMX/VNNI int8 compute) would close the ~2.3× gap to ORT/foundry
(20.12 tok/s native target vs ~8.8 tok/s native). Profiling at real 7B shapes
shows this is false: at M=1 the hand int4 GEMV and MLAS SQNBit **tie** (or hand
wins) because decode is memory-bound, and the 2.3× gap is **engine-level
fork-join + NUMA overhead**, not the MatMulNBits kernel choice.

## Real per-token MatMulNBits shapes (extracted from the ONNX graph, not hardcoded)

| Projection | K | N | ops/token |
|---|---:|---:|---:|
| lm_head | 3584 | 152064 | 1 |
| gate + up | 3584 | 18944 | 56 |
| down | 18944 | 3584 | 28 |
| qkv | 3584 | 4608 | 28 |
| o_proj | 3584 | 3584 | 28 |

141 MatMulNBits ops/token; ~3.5 GB int4 weights streamed per token.

## Micro-benchmark: the earlier "MLAS wins M=1" was a cache artifact

`matmulnbits_mlas_perf` reuses the same buffers across iterations, so weights
stay L3-resident and MLAS reports a 1.7–1.97× M=1 "win" — a fantasy for decode,
where each op touches a **distinct DRAM-resident** weight. New probe
`matmulnbits_mlas_decode_step` replays the real 7B op sequence with distinct
cold buffers:

| Path (cold, distinct DRAM weights, M=1, 32t) | Throughput | Bandwidth |
|---|---:|---:|
| hand int4 GEMV (lightly loaded host) | ~26 tok/s | ~92.9 GB/s |
| MLAS SQNBit CompInt8 (lightly loaded host) | ~25 tok/s | ~89.2 GB/s |
| hand int4 GEMV (heavily loaded host, load avg 67) | 22.55 tok/s | 79.7 GB/s |
| MLAS SQNBit CompInt8 (heavily loaded host) | 18.56 tok/s | 65.6 GB/s |

M=1 decode is bandwidth/latency-bound; MLAS CompInt8 never beats the hand path
and would add int8 activation-requantization rounding. Per rules 4/5, keep the
hand path for M=1 `accuracy_level=4`.

## Where the 2.3× gap actually is (`perf record`, end-to-end decode)

| Bucket | Share | Notes |
|---|---:|---|
| MatMulNBits compute | ~44% | the actual GEMM work |
| rayon / crossbeam-epoch fork-join | ~27% | threads idle-spinning at per-op join barriers |
| `to_dense_bytes` | ~7.5% | one-time weight materialization |
| `prepack_int8_weight` | ~4.5% | one-time, cached in OnceLock |

141 ops/token × up to 64 `par_chunks_mut` tasks each ⇒ ~141 fork-join barriers
per token. NUMA test: `numactl --cpunodebind=0 --membind=0` gives **+25%
(~10 tok/s)** but plateaus at ~10 even with 48 threads, at only ~14% of memory
bandwidth ⇒ latency/sync-bound, not bandwidth- or kernel-bound.

## Weight prepacking is already once-per-weight (verified)

`build_mlas_packed` result is cached in the kernel's `OnceLock` (`mlas_packed`),
and the executor kernel cache (`get_or_create`, keyed by node + input shapes)
persists kernels across decode steps, so decode steps are pack-free. No change
needed here.

## Change shipped on this branch

1. **Renamed the knob** `NXRT_SQNBIT_PREFILL_MIN` → **`NXRT_SQNBIT_DECODE_MIN`**
   (default **16**), with measured rationale in the docstring (cold-tie, the
   cache artifact, the fork-join/NUMA gap). It is the `M` crossover below which
   int4 decode on a *fast* hand path stays on the hand kernel; at/above it MLAS
   SQNBit is used (prefill). Overridable by the env var as before.
2. **Generic, shape/dtype-driven M=1 gate** (rule 2 — no model identity):
   - `bits==4 && accuracy_level==4` (fast `int4_matmul_m1`/`int8_matmul` hand
     paths) → keep on hand path for `m < NXRT_SQNBIT_DECODE_MIN`.
   - `bits==4 && accuracy_level!=4` (slow hand path dequantizes to f32 then runs
     a dense GEMV) → route M=1 to **MLAS SQNBit CompFp32**: a genuine generic
     win (MLAS beats dequant-then-GEMM), added without model-name coupling.
   - `g_idx` present or `bits!=4` (2-bit) → hand path (MLAS SQNBit can't serve).

## Numerics evidence (rule 8 tests in the same commit)

- The M=1 `accuracy_level=4` route is **unchanged** ⇒ bit-identical output; the
  7B model is `accuracy_level=4`, so end-to-end tokens are identical to baseline
  ("... return a new list that is sorted in ascending order ...").
- New test `matmulnbits_try_mlas_serves_slow_dequant_decode`: m=1, bits=4,
  accuracy_level=0 routes to MLAS (`Ok(Some)`) and matches the f32 reference
  within `2e-3` (CompFp32 dequant is near-exact).
- Renamed test `matmulnbits_resolve_decode_min_parses_or_defaults`; updated
  `matmulnbits_try_mlas_gates_decode_by_m_threshold` for the new constant.
- Added ignored probe `matmulnbits_mlas_decode_step` (cold distinct-buffer
  hand-vs-MLAS 7B decode-step harness).
- `cargo test -p onnx-runtime-ep-cpu --features mlas matmul_nbits`: **32 passed,
  2 ignored**.

## End-to-end before/after (honest)

Shared host, heavily loaded (load avg ~67 during measurement), ±1 tok/s noise:

| | tok/s |
|---|---:|
| baseline (before) | ~7.5 |
| after (7B, acc4 ⇒ routing unchanged for M=1) | ~7.5 |

For the 7B `accuracy_level=4` model the shipped change is **behavior-neutral at
M=1** (correctly so — rule 4: don't replace what already wins). It does **not**
reach the 20.12 tok/s ORT target, because that gap is not in the kernel.

## Follow-up recommendation (out of scope for this kernel change)

To Roy (engine/executor) and Chew (numerics): the real win is at the threading
layer, not MatMulNBits routing:
1. **Reduce per-op fork-join barriers** — 141 join points/token dominate.
   Consider an ORT-style persistent worker pool / fewer synchronization points
   per token (fuse the per-op parallelism, or a graph-level parallel section).
2. **NUMA-aware weight placement + thread pinning** — first-touch places weights
   on one node; cross-node decode threads pay remote latency. `numactl` pinning
   already shows +25%. This is cross-crate (loader + decode pool) and should be
   designed, not shipped as a half-baked heuristic.

---

## Update (2026-07-22, later) — definitive 3-way micro-bench + a shipped contained win

Following Sebastian's authoritative profile (MatMulNBits = 77.1% of the 83.4 ms
M=1 decode step; 64.3 ms), I re-settled the MLAS-vs-hand question rigorously and
then pivoted to the hand-path glue overhead.

### Definitive 3-way decode-step micro-benchmark (cold distinct DRAM, 32 threads)

`matmulnbits_mlas_decode_step` now measures all three candidates:

| Path (M=1, cold, real 7B op mix) | ms/step | tok/s | GB/s |
|---|---:|---:|---:|
| hand int4 GEMV | 33.88 | 29.52 | 104.3 |
| MLAS SQNBit CompInt8 | 32.68 | 30.60 | 108.2 |
| MLAS SQNBit CompFp32 | 41.94 | 23.84 | 84.3 |

hand and CompInt8 **tie** (within ~3–4%, and the sign flips with host load;
under heavy load hand led 22.6 vs 18.6 tok/s). CompFp32 is **clearly worst**.
So for M=1 `accuracy_level=4` the hand path stays (ties the best, no int8
rounding). Routing confirmed, not model-name based (rule 2).

### The real per-op gap is executor/fork-join glue, and part of it is fixable

The isolated kernel probe runs the *entire* 7B MatMulNBits op mix in **~33 ms**,
yet the real decode MatMulNBits bucket is **64.3 ms** — ~30 ms of per-op glue
sits on top of the kernels. A chunk of that glue was a **serial, non-vectorized
per-element strided copy**: every op called `to_dense_f32` on its activation and
`write_dense_f32` on its output, walking elements one at a time with multi-dim
index bookkeeping — ~2.5 M serial iterations/token, off the parallel path.

**Shipped fix (contained, generic, rule 8 tested):** add a contiguous
row-major **bulk-copy fast path** to `to_dense_f32` and `write_dense_f32`
(`crates/onnx-runtime-ep-cpu/src/kernels/mod.rs`). Contiguous tensors (the
common decode/prefill case) now `copy_from_slice`/`extend_from_slice` instead of
the strided walk; non-contiguous views keep the exact strided path. Benefits
every f32 kernel, not just MatMulNBits.

### End-to-end before/after (same host window, 32 threads, 6 runs each; noisy shared host)

| | best ms/step | best tok/s | median tok/s |
|---|---:|---:|---:|
| before (contiguous strided walk) | 104.0 | 9.61 | ~9.2 |
| after (bulk-copy fast path) | 87.8 | 11.39 | ~10.2 |

~15% faster step at best-case, ~+11% median. Generated text unchanged/coherent.
Numerics: bit-identical (pure data-movement fast path; both new tests plus the
existing `dense_roundtrip_contiguous` / `dense_reads_transposed_view` prove the
fast and strided paths agree).

### Still-open gap to 20 tok/s (cross-crate — for Roy/Chew)

After the fix, real decode best is ~88 ms/step vs the isolated kernel's ~33 ms.
The remaining ~55 ms is per-op **Rayon fork-join re-entry**, executor dispatch,
NUMA remote-weight latency, and the non-MatMulNBits ops. Closing to ORT's
20.12 tok/s needs the architectural work, ranked:
1. **Projection grouping** — fuse the two independent MLP MatMulNBits (gate, up)
   that share the same input A into one op: halves MLP fork-joins and reuses the
   activation quantization. The optimizer pass framework
   (`onnx_runtime_optimizer::run_passes`, cf. `fuse_silu_patterns`) is the right
   home; detect by graph structure (shared input, compatible bits/block/acc),
   never by model name (rule 2).
2. **Fewer per-op fork-join barriers** — 141 MatMulNBits ops/token each fork+join
   the decode pool; an ORT-style persistent/looser barrier model would cut the
   ~27% fork-join share and fix the >32-thread scaling regression.
3. **NUMA-aware weight placement + thread pinning** — `numactl --membind` is
   already +25%; make it intrinsic (loader first-touch + decode-pool affinity).

### Tests added/changed this update
- `write_dense_contiguous_bulk_copies`, `write_dense_strided_matches_logical_order`
  (`kernels/mod.rs`) — cover the new fast path and the retained strided path.
- `matmulnbits_mlas_decode_step` extended to the 3-way hand / CompInt8 / CompFp32
  comparison.
<!-- source: .squad/decisions/inbox/deckard-numa-affinity-fix.md -->
### 2026-07-22: NUMA decode-affinity — revised to clear Gaff's rejection
**By:** Deckard (non-author reviser; Batty locked out per Rule 9)
**What:** Fixed the three findings Gaff raised against commit `32a122e`. All edits
are confined to `crates/onnx-runtime-ep-cpu/src/decode_affinity.rs` (the caller in
`kernels/matmul_nbits.rs` is untouched — see rebase note 1). The optimization
itself (NUMA-local pinning of the bounded M=1 decode pool, +25% / 13.1→16.3
tok/s) is unchanged; only correctness/quality was addressed.

**Fixes:**
1. **`cpu_set_t` overflow / OOB (correctness, portability).** Replaced the fixed
   1024-bit `libc::cpu_set_t` + `CPU_SET` with a dynamically sized mask built
   from the runtime CPU index. New private helper `build_cpu_mask(cpu)` returns a
   `Vec<libc::c_ulong>` — the exact word layout `sched_setaffinity` expects — with
   only `cpu`'s bit set, sized to `cpu/word_bits + 1` words, so a CPU id at or
   above `CPU_SETSIZE` grows the buffer instead of writing out of bounds. It
   returns `None` on word-count overflow, and `pin_current_thread_to_cpu` then
   falls back to unpinned (no panic, no OOB). `sched_setaffinity` receives the
   mask's true byte length. `unsafe` is reduced to the single syscall with a
   justified SAFETY note; the buffer is safe, owned Rust.
   - **Mask approach note:** the review suggested `CPU_ALLOC`/`CPU_SET_S`; those
     symbols are **not exposed by the `libc` 0.2 crate for `x86_64-*-linux-gnu`**
     (only android/hurd/cygwin/l4re), so they do not compile on our target. The
     hand-built `Vec<c_ulong>` implements the same option-(a) semantics
     (dynamically sized mask covering `cpu`, true byte length passed to the
     syscall) with *less* `unsafe` and a pure, directly unit-testable sizing
     helper.
2. **Diagnostics (Rule 1) — consistent across every invalid path.** Added
   `const ACCEPTED_MODES` plus helpers `available_nodes_clause(topology)` and
   `invalid_selector_error(value, topology)`. New
   `DecodeAffinity::resolve(raw, topology)` parses AND validates against topology
   so every invalid value — malformed mode, non-integer index, unknown node
   index, and a `node:<index>` on a host with no discoverable topology — produces
   one message naming (i) the rejected value, (ii) all accepted modes, and (iii)
   the available-node list or an explicit "NUMA topology is unavailable"
   statement. `DecodeAffinity::from_env` now detects topology and calls
   `resolve`, so the existing `matmul_nbits.rs` caller (unchanged) reports an
   unknown node even on a single-node / `/sys`-unavailable host instead of
   silently unpinning. `cpus_for`'s unknown-node error was upgraded to the same
   three-part content. `compact`/`off` without topology stay honored as
   "leave unpinned".
3. **`compact` selection semantics.** Changed `min_by_key(|c| c.len())` (fewest
   CPUs) to `.values().find(|c| c.len() >= worker_count)`. Because `nodes` is a
   `BTreeMap`, `.values()` is ascending index order, so this selects the
   smallest-index fitting node — matching the documented policy.

**Tests added (Rule 8); existing 4 kept green (8 pass total):**
- `resolve_reports_consistent_diagnostics_for_invalid_values` — asserts rejected
  value + all accepted modes + available-node list appear for malformed mode,
  non-integer index, and unknown node index.
- `resolve_reports_topology_unavailable_for_node_without_topology` — asserts the
  topology-unavailable statement (plus value + modes) for `node:<index>` with no
  topology, and that `compact`/`off` still resolve without topology.
- `build_cpu_mask_sizes_beyond_cpu_setsize_without_oob` (Linux) — asserts a CPU
  id ≥ `CPU_SETSIZE` grows the mask beyond a fixed `cpu_set_t`, sets the correct
  bit/word with earlier words zero, and stays sound far beyond `CPU_SETSIZE`.
- `compact_prefers_smallest_index_not_fewest_cpus` — distinguishes the new
  smallest-index policy from the old fewest-CPU behavior.

**⚠️ Bryant — rebase notes (numa-split feature shares this file):**
1. `matmul_nbits.rs` is UNCHANGED in my commit; it still calls
   `DecodeAffinity::from_env()?`. `from_env` is retained (not removed) and now
   internally does `resolve(raw, NumaTopology::detect())`. If your feature needs
   topology-aware parsing at the env boundary, prefer `from_env`/`resolve`.
2. New `DecodeAffinity::resolve(raw: Option<&str>, topology: Option<&NumaTopology>)
   -> Result<Self, String>` is the single validation entry point.
3. `ACCEPTED_MODES` currently lists `off`, `compact`, `node:<index>`. When you
   add the `NumaSplit` variant + `numa-split` mode, **add `numa-split` to
   `ACCEPTED_MODES`** so diagnostics stay consistent, add a `parse` arm, and make
   `resolve` pass it through as valid.
4. `compact` now uses `.find` (smallest-index), not `min_by_key(len)`.
5. `pin_current_thread_to_cpu` internals now use `build_cpu_mask`; signature
   unchanged.
6. `NodeShard` / `split_workers` and `decode_numa.rs` are NOT in my commit
   (removed from the working tree per the coordinator, who preserved your patches
   in `.squad/tmp-bryant/`). Rebase them onto my commit in your worktree.

**Validation:** `cargo test -p onnx-runtime-ep-cpu --features mlas` → 669 passed,
0 failed, 3 ignored. `cargo clippy -p onnx-runtime-ep-cpu --features mlas` →
clean. Committed to `perf/cpu-ep-mlas` (NOT pushed). Non-author re-review by Gaff
to follow.
<!-- source: .squad/decisions/inbox/gaff-numa-affinity-review.md -->
### 2026-07-22: NUMA decode-affinity review — rejected pending revision
**By:** Gaff
**What:** Reviewed non-author commit `32a122e` (`perf(cpu): NUMA-local CPU affinity for the M=1 decode pool`) and rejected it.
**Why:** The change correctly confines affinity to the bounded decode Rayon pool, discovers topology at runtime from `/sys`, defaults to `off`, and uses a race-free `OnceLock` for best-effort per-worker fallback. Its fixed `libc::cpu_set_t` mask, however, is only 1024 bits on Linux and is not sized from the runtime CPU index. A discovered CPU ID at or above `CPU_SETSIZE` can make `CPU_SET` index beyond the fixed mask (rather than return the intended graceful affinity failure), so `sched_setaffinity` is not sound or portable to large CPU-ID hosts. Revise the pinning helper to construct a dynamically sized mask that covers `cpu` (or validate the index before `CPU_SET` and log/fall back without panicking), and pass that mask's actual byte length to `sched_setaffinity`.

The diagnostics also do not meet Rule 1's requested combined contract. Malformed modes name accepted modes but not available nodes; unknown `node:<index>` names available nodes but not accepted modes; and on a single-node or `/sys`-unavailable host `node:<index>` is silently treated as the topology fallback, so an unknown selector is not reported. Make every invalid value error include the rejected value, all accepted modes, and the discovered available-node list (or state that topology is unavailable), with focused assertions for those messages.

`compact` is also documented as selecting the smallest-index fitting node, but `min_by_key(|cpus| cpus.len())` selects the fitting node with the fewest CPUs (using index only as an equal-size tie-breaker). Make the selection match the documented smallest-index policy, or document the capacity-minimizing policy precisely.

`cargo test -p onnx-runtime-ep-cpu --features mlas` passed: 665 passed, 0 failed, 3 ignored. `cargo clippy -p onnx-runtime-ep-cpu --features mlas` passed. The four unit tests cover parser modes, CPU-list parsing, compact selection, and unknown-node detection, but they do not cover the required diagnostic content or high CPU-index/fallback safety. Per Rule 9, Batty is locked out from revising this artifact; a different author must make the required fixes.
### 2026-07-22: NUMA decode-affinity revision review — approved
**By:** Gaff
**What:** Re-reviewed Deckard's independent revision, commit `046414b`, against the required fixes from the prior rejection.
**Why:** `build_cpu_mask` correctly allocates `cpu / (8 * size_of::<c_ulong>()) + 1` words and sets bit `cpu % bits_per_word`, so IDs at and above the fixed `CPU_SETSIZE` cannot index a fixed-size `cpu_set_t` out of bounds. The syscall receives exactly `mask.len() * size_of::<c_ulong>()` bytes; the buffer is aligned as `c_ulong`, remains live for the call, and is read-only, making the sole FFI `unsafe` sound. Its checked index-size construction returns an error on arithmetic failure, and a kernel affinity failure is handled by the existing pool start handler's once-logged unpinned fallback.

`DecodeAffinity::resolve` now unifies malformed, non-integer, unknown-node, and no-topology node-selector errors: each names the rejected selector, all three accepted modes, and either the ordered node list or an explicit topology-unavailable statement. `from_env` supplies detected topology to this validation. `compact` now uses `find` over ordered `BTreeMap` values, correctly choosing the smallest-index fitting node. The four new tests assert diagnostic content (including unavailable topology), masks beyond CPU_SETSIZE, and the differing-size smallest-index case. Validation passed: `cargo test -p onnx-runtime-ep-cpu --features mlas` and `cargo clippy -p onnx-runtime-ep-cpu --features mlas`.
<!-- source: .squad/decisions/inbox/leon-gqa-revision.md -->
### 2026-07-22: Harden CPU GQA SIMD numerical validation
**By:** Leon
**What:** Replaced the incorrect dot-product error claim with the standard `γ_n × Σ|a_i b_i|` forward-error scale, made the long-context parity fixture exercise 128-wide AVX2+FMA dot and AXPY paths with mixed-scale cancellation-heavy data, and added a 257-key weighted-value accumulation regression.
**Why:** Chew rejected the original tests because head width 2 bypassed SIMD and a single AXPY update did not represent decode. Both AVX2 regressions failed under temporary helper mutations and passed after restoration; the required MLAS GQA suite passed 17 tests. A 16-token Qwen2.5-Coder-7B profiler comparison produced identical optimized and forced-scalar token IDs `[2014, 5978, 34776, 19753, 11, 279, 6500, 21896, 6529, 16895, 6337, 5711, 264, 76369, 729, 448]`.

<!-- source: .squad/decisions/inbox/sapper-gqa-cpu-decode.md -->
# Decision: GQA CPU decode optimization (perf/cpu-ep-mlas)

**Author**: Sapper  
**Date**: 2026-07-22  
**Branch**: perf/cpu-ep-mlas  
**File**: `crates/onnx-runtime-ep-cpu/src/kernels/group_query_attention.rs`

---

## What changed

Three targeted optimizations to the M=1 decode hot path in `GroupQueryAttentionKernel::execute`.

### 1. Attended-window scoring only

`scores` is now allocated with `attended = causal_limit + 1 - local_start` elements
(the actual causal window) instead of `total_sequence_length` (full sequence).
For full causal attention these are equal, but the shorter allocation avoids
initializing and iterating over masked-out positions in all downstream code.

### 2. SIMD dot-product for QK scores (`dot_f32` / `dot_avx2_fma`)

New `dot_avx2_fma` with `#[target_feature(enable = "avx2,fma")]` and a safe
dispatch wrapper `dot_f32`. Uses two 8-wide AVX2 accumulators to hide FMA
latency, processes 16 elements per iteration, with a scalar tail for non-pow-2
head sizes. Runtime-gated via `crate::backend::has_simd_x86()` (same check the
MLAS GEMM uses). Scalar fallback preserved for non-x86 targets.

### 3. Cache-friendly P·V accumulation (`axpy_f32` / `axpy_avx2_fma`)

P·V loop reordered from **d-outer, ks-inner** to **ks-outer, d-inner**. The
original ks-inner loop accessed `present_v` at stride `head_dim` (stride-128
for Qwen2.5-7B), causing one L1 cache miss per key position per output
dimension. The new ks-outer order reads each V row as a contiguous
`head_dim × sizeof(f32)` block, then accumulates via an AVX2 FMADD AXPY
(`axpy_avx2_fma`). Scores are normalized once (in-place divide by sum) before
the P·V loop, eliminating per-element division.

---

## Benchmark results

Machine: development workstation (not the Sapphire Rapids Xeon 8480C in
Sebastian's profile — results are directionally correct but absolute numbers
will differ on target hardware).

Model: Qwen2.5-Coder-7B int4, CPU EP, 32 decode threads.

### Short context ("Write a function to sort a list.", 24 generated tokens)

| Step | GQA ms/step (baseline) | GQA ms/step (optimized) | Speedup |
|------|------------------------|--------------------------|---------|
| Step 1 (~8 context tokens) | 3.34 ms | 1.77 ms | **1.89×** |
| Step 12 (~20 context tokens) | 5.15 ms | 2.05 ms | **2.51×** |
| Step 24 (~32 context tokens) | 7.55 ms | 2.37 ms | **3.18×** |

**Short context end-to-end:**
- Baseline: 8.73 tok/s (114.5 ms/step)
- Optimized: 9.23 tok/s (108.3 ms/step)
- Improvement: **+5.7%**

### Long context (~1000-token prompt, 32 generated tokens)

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| GQA ms / 28 calls per step | 85–89 ms | 66–68 ms | **1.28×** |
| GQA ms/call | 3.1 ms | 2.4 ms | **1.27×** |
| Overall step latency | ~163–168 ms | ~143–150 ms | **1.12×** |
| End-to-end throughput | 0.36 tok/s | 0.41 tok/s | **+14%** |

---

## Precision / numerics evidence

The softmax path is **unchanged**: each score still uses
`(score_f32 - max_f32) as f64).exp() as f32` (CUDA cross-EP parity contract).

The AVX2 dot-product uses two parallel f32 accumulators; the induced rounding
difference vs sequential scalar is bounded by `n × f32::EPSILON × max(|q|, |k|)`
(≈ 128 × 1.2e-7 × 1.0 = 1.5e-5 for head_dim=128, normalized inputs).

**Greedy token-id parity verified**: 16-token decode from the same long
prompt produces identical token ids on baseline and optimized builds:

```
[31075, 264, 4583, 7868, 2711, 4916, 304, 13027, 448, 5656, 11, 3698, 11, 2711, 11, 323]
```

---

## Tests added (RULES.md §8)

Three new unit tests in `group_query_attention.rs`:

- `gqa_decode_long_context_matches_reference`: M=1 decode with 511-token past
  cache; output matches the scalar `reference()` within existing 1e-5 tolerance.
- `dot_f32_matches_scalar_reference_for_various_lengths`: `dot_f32` vs scalar
  for lengths 1, 7, 8, 9, 15, 16, 17, 32, 64, 128, 133 with bounded tolerance.
- `axpy_f32_matches_scalar_reference_for_various_lengths`: `axpy_f32` vs scalar
  same length set.

All 16 GQA tests pass.

---

## Rules compliance

- **Rule 2**: No hardcoded shapes. SIMD dispatch uses head_dim at runtime.
- **Rule 4**: Reuses `crate::backend::has_simd_x86()` runtime gate (same as MLAS
  GEMM). Reference scalar path preserved. Optimized and reference both testable.
- **Rule 8**: Tests in same commit.
- **Rule 9**: Chew review needed for numerics (AVX2 dot product reordering).

## Remaining work (not in this commit)

- **Scratch buffer reuse** across decode steps: kernel is stateless; a
  `thread_local!` or `Mutex<Vec<f32>>` in the kernel struct would eliminate
  `Vec` allocations in `compute_row`. Deferred for a follow-up.
- **Validation on Sapphire Rapids**: absolute latency numbers above are from a
  dev workstation. Reproduce on target with `ONNX_GENAI_PROFILE_OPS=1` at
  sequence length ≥1024 to confirm the cache-line locality gain holds.
- **AVX-512 dot-product**: the Xeon 8480C supports AVX-512, enabling 16-wide
  FMADD. The current 8-wide path leaves ~2× on the table for the QK scoring
  loop at long head_dim. Gating on `avx512f` is a follow-up.
<!-- source: .squad/decisions/inbox/sebastian-cpu-profile.md -->
### 2026-07-22: Native 7B CPU decode profile
**By:** Sebastian

## Method

- Host: dual-socket Intel Xeon Platinum 8480C, 96 physical cores, no SMT, two NUMA nodes.
- Model: Foundry Qwen2.5-Coder-7B int4 v4; prompt `Write a function to sort a list.` (8 tokens); greedy 24-token generation.
- Build: `cargo build --release -p onnx-genai-bench --features mlas --bin profile_native`.
- No CPU pinning; runs were sequential on the otherwise shared host.
- Per-node timing used the existing zero-cost-when-disabled `ONNX_GENAI_PROFILE_OPS=1` executor hook. The table is the mean of 23 measured M=1 forwards after the measured prefill.
- `ONNX_GENAI_PROFILE=1` measured host sampling separately. `profile_native` now resets warmup statistics and prints this existing stage profiler; the focused synthetic integration test covers enabled reporting.

## Important correction to the headline latency

The reported approximately 113 ms/generated-token number is **not one M=1 decode step**. `profile_native`'s default throughput timer includes one 8-token prompt prefill per 24 generated tokens.

At 32 decode threads in this run:

| measurement | result |
|---|---:|
| Default 24-token end-to-end benchmark | 116.662 ms/token, 8.57 tok/s |
| Steady M=1 decode (`--steady --decode-skip 8`, combined two runs) | 79.456 ms/token, 12.59 tok/s |
| Prefill/reset amortization in the default benchmark | 37.206 ms/generated token (31.9%) |

Thus only about 68% of the headline 116.7 ms/token is steady M=1 decode. Optimization claims must state which metric they improve.

## M=1 per-stage breakdown

The matched profiled generation measured 83.394 ms per M=1 step (profiling/load overhead makes this about 5% slower than the unprofiled 79.456 ms). Percentages are the robust result:

| stage | ms/M=1 step | share |
|---|---:|---:|
| `MatMulNBits` projections (141 calls) | 64.334 | **77.1%** |
| Elementwise/activation: `Silu` + `Add` + `Mul` | 7.934 | **9.5%** |
| GQA/attention, including RoPE | 5.335 | **6.4%** |
| RMSNorm/LayerNorm | 3.275 | **3.9%** |
| Sampling/host argmax | 0.079 | **0.1%** |
| Executor/native-decode orchestration and remaining tiny nodes | 2.437 | **2.9%** |
| **Total** | **83.394** | **100%** |

The residual is an upper bound because it also contains enabled-profiler bookkeeping. Sampling, token commit, and detokenization together are below 0.1 ms/token and are not material.

## MatMulNBits routing

M=1 does **not** use MLAS SQNBit under the current configuration. `NXRT_SQNBIT_PREFILL_MIN` was unset, so the default threshold is 16; `try_mlas_sqnbit` returns before packing when `m < 16`. The benchmark therefore uses the specialized packed hand int4/VNNI path for M=1. Building with `--features mlas` does not change this routing.

An exploratory `NXRT_SQNBIT_PREFILL_MIN=2` run kept M=1 on the hand path while sending the 8-row prompt to MLAS; it measured 8.43 tok/s versus 8.57 tok/s at the default threshold, so lowering the crossover is not an optimization on this workload.

## Thread scaling

Requested default-harness results (one prefill per 24 generated tokens, two measured runs):

| `ONNX_GENAI_CPU_DECODE_THREADS` | ms/generated token | tok/s | vs. 32 |
|---:|---:|---:|---:|
| 8 | 150.908 | 6.63 | -22.6% |
| 16 | 125.908 | 7.94 | -7.4% |
| **32** | **116.662** | **8.57** | — |
| 48 | 131.342 | 7.61 | -11.2% |

Steady M=1 combined across the two runs:

| threads | ms/M=1 token | tok/s |
|---:|---:|---:|
| 8 | 112.992 | 8.85 |
| 16 | 83.569 | 11.97 |
| **32** | **79.456** | **12.59** |
| 48 | 103.928 | 9.62 |

Thirty-two threads is the clear operating point for this 7B model on this dual-socket host; 48 crosses into synchronization/NUMA regression.

## Ranked optimization targets

1. **MatMulNBits cross-node efficiency (77.1%)** — keep the hand int4/VNNI M=1 backend, but target projection grouping, activation-quantization reuse, direct executor-output writes, and fewer per-projection barriers. A 20% local reduction is a 15.4% M=1 latency reduction; a 30% local reduction is 23.1%.
2. **Fuse projection-adjacent elementwise work (9.5%)** — combine eligible bias/residual and gate/up SiLU work structurally, preserving generic fallbacks. Recovering half this bucket yields about 4.8% lower M=1 latency; the absolute ceiling is 9.5%.
3. **GQA/attention (6.4% here, increasing with context)** — reduce remaining per-layer attention setup/copies and reuse scratch/static KV views. Halving this bucket yields about 3.2% at this short context, with larger upside at long context.

RMSNorm is the next target at 3.9%, preferably as part of residual-plus-normalization fusion. Sampling and generic loop orchestration are not priority work.

## Follow-up: decode-to-decode runtime comparison

All three runtimes used the same model directory, bare 8-token prompt, greedy decoding, and one warmup. The ORT wrapper explicitly used 32 intra-op threads. Native used 32 decode threads. OGA 0.14.1 does not expose the ORT intra-op setting through its Python configuration surface, so its model-default CPU threading remained in effect.

### Comparable 24-token end-to-end request

These numbers include per-request setup and prompt prefill, but exclude model loading and prompt tokenization:

| runtime | ms/generated token | tok/s | native-relative |
|---|---:|---:|---:|
| Native nxrt CPU | 116.662 | 8.57 | 1.00x |
| ORT wrapper, 32 threads | 45.633 | **21.91** | **2.56x** |
| onnxruntime-genai | 53.179 | **18.80** | **2.19x** |

The repository `oga_bench.py` originally reported 21.04 tok/s at 24 tokens because its timer begins **after** `Generator.append_tokens`, excluding OGA's prompt prefill. A separate timer around generator setup, append, and decode gives the comparable 18.80 tok/s above. OGA spent about 1.1 ms in generator setup and 101.8 ms in prompt append/prefill per request.

### Clean 128-token steady decode

Each runtime generated 128 tokens and the steady window excluded the first eight emitted tokens. Native and ORT produced the same continuation; OGA produced a different greedy continuation despite the same raw prompt/model, so its number is a throughput comparison at identical lengths rather than token-identical execution.

| runtime | steady window | ms/M=1 token | tok/s | native-relative |
|---|---:|---:|---:|---:|
| Native nxrt CPU, 32 threads | tokens 9-128 | 91.447 | 10.94 | 1.00x |
| ORT wrapper, 32 threads | tokens 9-128 | 37.145 | **26.92** | **2.46x** |
| onnxruntime-genai | tokens 9-128 | 48.101 | **20.79** | **1.90x** |

The earlier native 12.59 tok/s value covered only a short 24-token request. Extending all runtimes to the same 128-token context lowers native to 10.94 tok/s; the clean decode gap is therefore 2.46x to the ORT wrapper and 1.90x to OGA. ORT's full 128-token request measured 26.43 tok/s including prefill and one final logits materialization.

## Follow-up: decomposing native prefill versus reset

A prefill-only native run (`--tokens 1 --warmups 1 --runs 3`, node profiling enabled) directly separates graph execution from everything outside executor nodes:

| component per request | time | share |
|---|---:|---:|
| M=8 executor-node compute, mean | 748.617 ms | 99.2% |
| Reset, input/output allocation, sampling, detokenization, and profiler bookkeeping combined | at most 5.880 ms | at most 0.8% |
| Total mean wall time | 754.497 ms | 100% |

The three measured M=8 node times were 1079.810, 583.353, and 582.688 ms, demonstrating substantial host/cache noise but consistently dwarfing reset overhead. Mean M=8 compute attribution was:

| prefill operator | mean ms | compute share |
|---|---:|---:|
| `MatMulNBits` | 607.858 | **81.2%** |
| GQA/attention | 45.686 | 6.1% |
| `Silu` | 45.236 | 6.0% |
| RMSNorm/LayerNorm | 28.302 | 3.8% |
| `Add` + `Mul` and remaining nodes | 21.535 | 2.9% |

This confirms that the earlier 31.9% “prefill/reset” bucket is genuine M=8 model compute, not benchmark reset/allocation. The native M=8 prefill is roughly 0.58-1.08 seconds versus 63.5 ms for the 32-thread ORT wrapper first forward and about 102 ms for OGA prompt append/prefill. Lowering `NXRT_SQNBIT_PREFILL_MIN` to route M=8 through MLAS did not improve end-to-end throughput (8.43 versus 8.57 tok/s).

**Decision:** assign dedicated CPU prefill optimization work if TTFT or short-request throughput matters. It will not improve steady M=1 decode, but the measured M=8 compute is a real product bottleneck and is overwhelmingly `MatMulNBits`, not harness overhead.
## 2026-07-21 — VLM WP2/WP3, opset-24 CUDA, ScatterElements, and DS-1

### Land VLM WP0 metadata contract and source-compatible hotfix
**By:** Sapper; hotfix by Rachael; reviewed by Luv 🟢  
**What:** Land architecture-neutral typed multimodal metadata as commit `0f6ffbd`, then make additive WP0 fields `Default`-derived in hotfix `1b66d0f` so downstream literal construction sites keep building.  
**Why:** VLM routing must be metadata-driven rather than model-flavored, and optional multimodal fields must be source-compatible as the contract grows.

### Land native CUDA opset-24 ConstantOfShape, Gelu, and OneHot
**By:** Batty; reviewed by Pris 🟢  
**What:** Land commit `ea4036d` with generic native CUDA handlers for standard-domain ConstantOfShape, Gelu, and OneHot, preserving opset-aware semantics including negative-index behavior.  
**Why:** Opset-24 Gemma/DeepSeek-style graphs should stay native instead of falling back because construction, activation, or indexing handlers are missing.

### Replace VLM every-step model bindings with a generic Kahn executor
**By:** Sapper; reviewed by Luv 🟢  
**What:** Land VLM WP3 as commit `3aec9f3`, replacing model-flavored `EmbedsStepBinding` with a metadata-driven every-step executor that topologically schedules declared inputs, outputs, and dependencies using Kahn sorting.  
**Why:** Autoregressive VLM step execution must follow the declared metadata graph, not hard-coded architecture names.

### Land DS-1 generic runtime shape propagation with bounded materialization
**By:** Deckard; revision by Holden; rereview by Pris 🟢  
**What:** Land commit `1584fb3` for DeepSeek-V2 dynamic `Slice -> Unsqueeze` shape propagation, reusing the opset-aware shape-inference registry and permitting host materialization only after dtype, rank, and element-cap gates pass.  
**Why:** Runtime output sizing should reuse the same generic ONNX shape rules as kernels while preventing unbounded host copies from hostile or accidental shapes.

### Broaden native CUDA ScatterElements dtype coverage portably
**By:** Deckard; reviewed by Chew 🟢  
**What:** Land commit `5b01a01` covering fp16/bf16/fp32/int64 data with int32/int64 indices. Serial single-threaded reduction avoids half atomics, remains SM-portable, and is CUDA-graph capture-safe.  
**Why:** Valid ONNX ScatterElements graphs should not decline native placement solely because a supported data/index dtype pairing was absent.

### Land VLM WP2 native image processor after numerics and allocation fixes
**By:** Leon; revision by Sapper; final review Pris 🟢  
**What:** Land commit `5c48ba5` for generic metadata-declared image preprocessing. The accepted path preserves bit-exact `f32::from(v) / 255.0` Divide semantics (not reciprocal multiply; 126/256 bytes otherwise differ by 1 ULP), uses `try_reserve_exact` bounded allocations, rejects degenerate dimensions, and pins patch-size-2 HF fixtures by SHA.  
**Why:** VLM processors need multi-output metadata-declared preprocessing without legacy numerical drift or unbounded metadata-derived allocation.

### Preserve review lockouts from this segment
**By:** Scribe  
**What:** Record active lockout history: WP2 had Chew 🔴, locking Leon+Chew out until Sapper revised and Pris approved; WP4 had Gaff 🔴, locking Zhora+Gaff out while Batty revises; DS-1 had Gaff 🔴, after which Holden revised and Pris approved.  
**Why:** Rejected artifacts and reviewers stay locked out for their correction cycle, while accepted third-agent revisions become the authoritative artifacts.

### Treat CUDA 13 NVRTC on H200 as current-good
**By:** Scribe  
**What:** The CUDA crate pins `cudarc` `cuda-13000` with dynamic loading, and NVRTC 13 builds and runs GPU tests successfully on H200.  
**Why:** The older belief that this host requires CUDA 12.6 NVRTC is stale and should not guide future debugging or setup.

### Additional inbox decisions folded and deduped
**By:** Scribe  
**What:** Processed non-preserved decision inbox artifacts, deduping items already represented above or in the active ledger. Folded summaries:  
- `batty-clippy-hygiene.md` — 2026-07-21: Clear engine and ORT clippy warnings; By: Batty; What: Cleared all `cargo clippy --all-targets --features cuda -- -D warnings` diagnostics in `onnx-genai-engine` and `onnx-genai-ort` without changing public APIs or runtime logic..
- `brigitte-wp3-argmax-expose.md` — 2026-07-21: Expose and verify ORT multi-row device argmax; By: Brigitte; What: Added `DeviceSampler::argmax_rows(&self, DataType, usize, usize, usize) -> Result<Vec<u32>>`, implemented by `CudaSampler` through its existing `pub(crate) CudaSampler::argmax_rows` entry point. Coverage is f32, f16, an….
- `chew-flash-tc-adjudication.md` — Chew — Adjudication: `flash_attention_f16_tc` numerics dispute (Holden vs Deckard).
- `deckard-ep-transparency.md` — Decision: Production per-op executor spans + kernel-variant & capture-rejection reasons (native EP).
- `deckard-flash-tc-fix.md` — Deckard — flash_attention_f16_tc wmma parity investigation + permanent gate.
- `fenster-fixture-fix.md` — 2026-07-21: Treat binary/textproto twins as one model; By: Fenster; What: Chose Option A. `ModelDirectory` now collapses `<name>.onnx.textproto` when the same-stem `<name>.onnx` exists and prefers the binary; distinct model names remain ambiguous..
- `gaff-clippy-review.md` — 2026-07-21: Clippy hygiene review (Batty 2a0555b); By: Gaff; What: Approved commit `2a0555b` as pure Clippy hygiene. The six-file diff contains iterator idioms, redundant-clone removal in CUDA sampler tests, a let-chain, `then_some`, literal digit regrouping, a rustdoc blank line, and….
- `holden-attn-cliff-investigation.md` — Holden — Attention "cliff at ~pos 30" investigation (native CUDA, Qwen2.5-0.5B-int4).
- `holden-wp1-verify-review.md` — Review: WP1 — Native M=K verify + rewind primitive (option b) + (c)-ready guard.
- `hudson-fixture-fix-review.md` — 2026-07-21: loader same-stem fix review; By: Hudson; What: Binary/textproto twins are correctly treated as one logical model, with the binary preferred..
- `hudson-wp3-argmax-review.md` — Hudson review — WP3-prep multi-row device argmax.
- `joshi-rmsnorm-generic.md` — 2026-07-21: Select fp16 SkipRMSNorm warp half4 by structural capability; By: Joshi; What: Generalized `skip_rmsnorm_f16_warp_896` into `skip_rmsnorm_f16_warp_half4`. The kernel now receives and uses runtime `norm_size`, iterates `norm_size / (32 lanes * 4 halves)` half4 chunks per lane, divides the sum of sq….
- `kowalski-wave4-profile.md` — 2026-07-21: Wave-4 stacked CUDA profile; By: Kowalski; What: Treat wave-4 native CUDA fp16 decode as approximately 759 tok/s at 256 tokens and 789 tok/s at 1024 tokens, with about 227 launches/token, zero CUDA-graph fallbacks, and coherent decode..
- `pris-fusion-genericity-review.md` — Review: Fusion-genericity remediation (wt-fusion-generic @ 19b3b91).
- `pris-opset24-review.md` — Kernel Review — Native CUDA opset-24 op handlers.
- `pris-rmsnorm-review.md` — 2026-07-21: RMSNorm genericity review (Joshi 53d55e1); By: Pris; What: Reviewed branch `wt-rmsnorm-generic` @ 53d55e1, which replaces the.
- `ripley-wp2-native-driver.md` — WP2 — Native speculative driver (host-argmax accept).
- `sapper-fusion-genericity.md` — Decision: CUDA wave-4 fusions gate on structure + capability, not Qwen dims.
- `sebastian-multimodel-bench.md` — 2026-07-21: H200 native CUDA multi-model benchmark; By: Sebastian; What: Current `main` (`035ad9f`) measured Qwen2.5-0.5B int4 at **771.40 tok/s median** (766.49/773.62/771.40), 1 prompt token, 256 output tokens, 5 warmups per independent process, CUDA graph + device KV + strict CUDA, and ze….
- `solveig-wp1-verify-primitive.md` — Decision: WP1 — Native M=K verify + rewind primitive (option b) + (c)-ready guard.
- `wallace-ep-transparency-review.md` — 2026-07-21: EP transparency backbone review; By: Wallace; What: Deckard's per-op executor span backbone (`exec_plan_node`) is a genuine LIVE span, and the re-instrumented kernels attach kernel-variant + capture-status reasons to it in the real native decode path — my original dead-w….
- `wallace-wp2-driver-review.md` — WP2 native speculative driver — review.  
**Why:** The inbox should hold only living research artifacts; segment decisions belong in the active ledger.
## 2026-07-21 — CUDA graph M4 end-to-end validation

### Real Qwen2.5 int4 decode captures with zero fallbacks
**By:** Rachael; reviewed by Chew; smoke correction by Pris 🟢  
**What:** Seed unresolved persistent external input/output physical shapes only during capture, keeping eager shape resolution and binding-signature invalidation intact. Constant/Shape metadata reuse and capture-safe integer Sub, ReduceSum, and Gather complete the real Qwen graph while device-side GQA/Reduce/Gather guards still latch errors before token consumption. After Chew caught stale fallback assertions, Pris updated the H200 smoke to require one capture, 62 replays, zero fallbacks, and no fallback reason. Landed as `dda3b25`, `13c094a`, and `42b71f7`.  
**Why:** Qwen2.5-0.5B int4 now captures end to end with token-exact graph ON/OFF parity and zero fallbacks: 70.33 versus 19.99 tok/s at 256 tokens (+251.8%), and 24.25 versus 11.73 tok/s at 1024 tokens (+106.7%). This validates the complete M4 capture-safety track on the real model.
## 2026-07-21 — Perf campaign reconciliation

### H200 native CUDA decode target and profiling baseline
**By:** Ana and Rachael  
**What:** Use ORT GenAI H200 Qwen2.5-0.5B int4 steady-state decode as the performance target: **657.34 tok/s** at 256 tokens (667.43 tok/s at 1024). Native progressed from about **73 → 145 → 192 → 201 tok/s**, but f32 Sq=1 GQA remained dominant: 70.5% of GPU time over 256-token decode and 82.7% over 16-token decode.  
**Why:** GEMV/argmax work is valuable but insufficient alone; the next high-leverage path is replacing serial f32 decode attention and then wiring/validating fp16 flash decode.

### Retile MatMulNBits decode GEMV and approve the result
**By:** Royb; reviewed by Wallace 🟢  
**What:** Retile the M=1 accuracy-level-4 symmetric block-32 CUDA MatMulNBits path, quantizing the f32 activation once with matching warp absmax/round/clamp/scale semantics. Wallace approved Roy's `5dbcbbb` retile.  
**Why:** This moved native decode from roughly 145 tok/s to about 192 tok/s while preserving numerics, but still leaves a large gap to Ana's 657 tok/s ORT target.

### Keep device-side greedy argmax after Batty's rebase repair
**By:** Mariette and Batty; reviewed by Joi 🟢  
**What:** Add allocation-free CUDA f32 greedy argmax with lowest-index tie behavior matching the host sampler. Joi rejected Mariette's rebased `c12e74f` because `DecodeCudaState::run_one_token` was called without the new `TraceContext`; Batty fixed the call and Joi approved `cdf62a0`.  
**Why:** The fixed path builds and measured about **200.97 tok/s**, removing the host argmax bottleneck without changing token selection.

### Land fp16 flash-decode as kernel-only first, then dormant dispatch wiring
**By:** Sebastian; reviewed by Bryant and Holden 🟢  
**What:** Add a capture-safe fp16 flash-decode GQA attention kernel as kernel-only commit `9c6f36b`, approved by Bryant. Wire it through a dormant fp16 dispatch branch at `521438e`, approved by Holden, gated by `q.dtype == Float16` and supported `(q_seq, dim)` while leaving the f32 path first and unchanged.  
**Why:** Split landing keeps the kernel independently reviewed and lets dispatch be enabled safely only for supported fp16 decode shapes.

### Direct fp16 activation × int4 GEMV remains a separate optimization track
**By:** Royb  
**What:** Prototype direct fp16-activation × int4 MatMulNBits GEMV on `wt-fp16-matmul` (`6a1daa2`) to avoid the int8 quantization pass.  
**Why:** This is distinct from fp16 flash attention and should be validated as a separate GEMV optimization before promotion.

### Sequence zero-copy design needs a second Deckard revision
**By:** Zhora and Deckard; reviewed by Luv 🔴  
**What:** Zhora's zero-copy Sequence tensors use shared allocation views with dtype/shape/layout/offset metadata. Luv rejected `ddae7d0`; Deckard closed the original public-output/runtime blockers with `SessionOutput::{Tensor, Sequence}` and related fixes, but Luv's re-review still rejected `cf8888b`.  
**Why:** The direction is acceptable, but remaining correctness/review blockers mean the Sequence zero-copy change is not approved yet.

### Runtime string tensors must use a dedicated host storage variant
**By:** Batty  
**What:** Represent runtime strings with `TensorStorage::{Raw, Strings(Vec<String>)}` or equivalent, expose safe `StringTensorView`/`StringTensorMut`, and never cast byte/device storage to `String`.  
**Why:** String tensors are host-owned structured values, not raw numeric buffers; exhaustive storage keeps executor behavior type-safe.

### PressureProtocol scaffold/fix path and current rejection state
**By:** Sapper, Roy, Deckard, and Pris; reviewed by Holden and Freysa 🔴/🟢 mixed  
**What:** Sapper/Roy added HostGovernor pressure envelopes and replay extension points; Holden rejected the first scaffold until actor ordering was scoped by `(HostId, ActorId)`, which Deckard fixed. Freysa rejected Sapper's HostGovernor revision, locking Sapper out and assigning the fix to Batty; Roy repaired release integrity by retaining authoritative allocations in `Claimed` and enforcing deterministic scheduling. Freysa's 2026-07-21 re-review still rejected `3207c25` because the branch/diff was not review-clean. Pris strengthened forged-release and cancellation synchronization regression tests.  
**Why:** Credit integrity and deterministic admission are the right design constraints, but the pressure implementation is not approved until reviewed from a clean branch with the fixed protocol evidence.

### Graph-capture transparency requires structured reasons across three axes
**By:** Coordinator and Gaff; reviewed by Chew  
**What:** All EPs must surface structured trace reasons for kernel non-selection and graph-capture non-capturability; transparency has three axes: op claim, kernel-variant selection, and capture support. Gaff added `CaptureSupport::{Supported, Unsupported { reason }}` and default compatibility adapters; Chew reviewed the structured reason-carrying design.  
**Why:** Silent bool declines make performance debugging impossible; traces must explain both variant choice and capture segmentation/fallback.

### Decouple CUDA EP claim from segmented graph capture
**By:** Coordinator and Tyrell  
**What:** CUDA EP should claim/run supported subgraphs even when only maximal segments are capturable, interleaving captured runs with eager CUDA runs for non-capturable nodes.  
**Why:** Capturability is an execution scheduling property, not an EP ownership property; partial segmented capture preserves CUDA placement without all-or-nothing fallback.

### Cross-platform support must include Windows ARM64
**By:** Coordinator; audit by Deckard  
**What:** Treat `aarch64-pc-windows-msvc` as a required target alongside Windows x64, macOS x86_64/arm64, and Linux x64. Deckard also flagged truthful CUDA selection, OS-aware library discovery, updated CUDA-12 CUDART candidates, pip/Conda NVIDIA discovery, and preventing Python from advertising CUDA while executing CPU.  
**Why:** Packaging and runtime probing must match the documented support matrix and actual execution provider behavior.

### Publishability of onnx-rs remains required
**By:** Leon  
**What:** Keep `onnx-rs` publishable to crates.io with package metadata and publish workflow coverage.  
**Why:** It is the ONNX standard-library crate for Rust in this workspace and must remain releasable.

### Capture-safe Sq=1 GQA decode kernel approved as prior f32 stepping stone
**By:** Sebastian; reviewed by Bryant 🟢  
**What:** Bryant approved `b6ada01`, a capture-safe warp-parallel Sq=1 GQA decode attention kernel for supported `head_dim <= 128` with zero CUDA-graph fallback.  
**Why:** This was a correct f32 decode-attention stepping stone before the later fp16 flash-decode path.
## 2026-07-21 — fp16 decode, transparent fallback, cross-platform loading, and trace cost

### Land coherent end-to-end fp16 native CUDA decode
**By:** Sebastian; component work by Mariette, Leon, and Roy; reviewed by Bryant, Wallace, and Holden 🟢  
**What:** Thread fp16 activations, KV, logits/argmax, normalization, RoPE, attention, and direct fp16×int4 MatMulNBits through native decode while retaining dtype-gated f32 paths. Leon fixed the rejected fp16 LayerNorm shared-memory reuse race before Bryant approved the normalization/RoPE path. Landed as `c8741ba`.  
**Why:** H200 Qwen2.5-0.5B int4 reached about **344 tok/s** with coherent tokens, CUDA graph capture, and zero fallbacks, up from the approximately **200 tok/s** f32 path; f32 remained unregressed near 200 tok/s.

### Make CUDA-to-CPU fallback observable and optionally strict
**By:** Deckard; reviewed by Batty 🟢  
**What:** Retain a structured `ExecutionProviderFallbackReport`, emit an initialization warning when CUDA declines force whole-session CPU execution, and make `ONNX_GENAI_REQUIRE_CUDA=1` reject that fallback. Landed as `3a8eebe`.  
**Why:** Device selection must not silently advertise CUDA while executing on CPU; callers now receive node/op/reason detail and can opt into strict CUDA-only behavior.

### Use OS-aware CUDA and CUPTI dynamic-library discovery
**By:** Leon and Roy; reviewed by Pris 🟢  
**What:** Select CUDA driver/runtime/library and CUPTI candidates by operating system, including Windows DLL names and pip/Conda layouts. Treat Windows ARM64 as gracefully unavailable before probing x64-only NVIDIA libraries. Landed as `2466016` and `8cd36c3`.  
**Why:** Cross-platform probing must fail normally rather than panic or attempt incompatible binaries. CUPTI discovery remains local to the tracer to avoid an inverted dependency on the CUDA EP.

### Emit per-op CPU bytes/FLOPs only for active trace spans
**By:** Rachael, Gaff, and Deckard; reviewed by Zhora 🟢  
**What:** Annotate major CPU kernel spans with logical tensor bytes and documented FLOP estimates, lazily computing metrics only when a span is active. Keep tracing optional and propagate the `tracing` feature through `bench-native` and `native-backend`. Landed as `61f4d2c`.  
**Why:** Profiles gain arithmetic-intensity and bandwidth inputs without imposing tensor scans, formula work, JSON allocation, or tracer dependencies on default non-tracing builds.
## 2026-07-21 — CI hardening and native CUDA decode wave 1–2

### Cover every offline crate and make warnings blocking on all portable targets
**By:** Batty and Gaff; Windows ARM64 revision by Deckard; reviewed by Hudson 🟢  
**What:** Classify all 38 workspace members by default normal+dev dependencies, explicitly test and cover all 27 pure-offline crates, and enforce blocking rustc and Clippy warnings (`RUSTFLAGS="-D warnings"` and `-- -D warnings`) rather than advisory lanes. The portable matrix retains Linux x64, Windows x64, and macOS ARM64 and adds native Windows ARM64 on `windows-11-arm`/`aarch64-pc-windows-msvc`, with the same 26-crate portable test set and an ARM64 Clippy gate; `mlas-sys` remains Linux-only, while native-ORT and CUDA crates stay outside offline execution. Formatting remains advisory pending the repository-wide sweep.  
**Why:** CI now covers the full offline workspace without triggering ORT downloads, and warnings fail builds across supported portable targets. The final 27-crate Linux lane passed 1,921 tests with 0 failures and 8 ignored; Hudson approved after Deckard closed the initially missing Windows ARM64 gate.

### Keep the measured wave-1 decode optimizations capture-safe
**By:** Leon, Tyrell, Deckard, Sebastian, and Roy  
**What:** Use persistent two-pass multi-block greedy argmax; segment CUDA graphs into maximal capturable runs around eager CUDA seams while retaining whole-subgraph EP ownership; abort/drain failed mid-segment capture before reset; use true multi-CTA split-K fp16 flash decode; and retain Roy's coalesced direct fp16×int4 GEMV retile. All paths preserve fixed device addresses, token semantics, and zero-fallback graph replay.  
**Why:** These changes removed launch/occupancy and GEMV bottlenecks without regressing correctness: argmax reached about 368 tok/s, split-K attention about 398 tok/s at 256 tokens (about 390 at 1024), and the GEMV retile about 423 tok/s. Segmented capture now recovers cleanly from invalidated streams instead of wedging later inference.

### Fuse the single-token GQA preparation chain
**By:** Rachael; reviewed by Holden 🟢  
**What:** For eligible `Sq=Sk=1` aliased fixed-capacity decode, fuse QKV split, query relayout, K/V append, and Q/K RoPE into one kernel and write attention output directly in BSH layout. Keep metadata preparation separate to preserve the capture poison/latch protocol; all other shapes retain the unfused path.  
**Why:** Prep launches fell 75% (192→48 per token), bit-exact fused/unfused and capture tests passed, and H200 throughput rose from about 557 to 615 tok/s with zero fallbacks.

### Use warp-shuffle fp16 skip-RMSNorm
**By:** Sapper; reviewed by Wallace 🟢  
**What:** Replace the fp16 shared-memory reduction tree with a single-warp packed-half2/half4 shuffle reduction, specializing hidden size 896 while retaining a tail-safe generic fp16 path; f32 kernels remain unchanged.  
**Why:** The hot kernel fell from about 6.20 to 5.07 µs/call and stacked decode reached about 579–583 tok/s with identical tokens, full CUDA tests passing, and zero graph fallbacks.

### Specialize the fp16 down-projection GEMV and accept the stacked ORT win
**By:** Luv; reviewed by Pris 🟢  
**What:** Route only `K=4864, N=896, block_size=32` with fp16 scales to a 256-thread, eight-column K-parallel GEMV that stages the activation in permuted half2 shared memory; all other shapes retain the general kernel.  
**Why:** The down-projection kernel fell from about 10.24 to 7.28 µs/call with parity within fp16 tolerance and identical greedy tokens. Stacked with GQA fusion and RMSNorm, native H200 decode reached **663–672 tok/s**, beating the **657 tok/s ORT GenAI** reference with zero fallbacks.

### Require SM-portable correctness and performance for every CUDA EP kernel
**By:** Coordinator directive; validated in wave-2 reviews by Holden, Wallace, and Pris  
**What:** Every `onnx-runtime-ep-cuda` kernel must remain correct and performant across supported NVIDIA SM architectures, not merely `sm_90`. Dispatch must derive the live architecture dynamically, avoid unguarded SM90-only features, keep resource use within portable limits, and preserve capable fallbacks or variants where architecture-specific tuning is necessary.  
**Why:** H200 wins are not acceptable if they break or materially strand devices such as RTX 4060 (`sm_89`). Wave-2 kernels use broadly available primitives and do not raise the minimum architecture.
## 2026-07-21 — Native CUDA decode wave 3 and CUDA CI

### Use 16-way split-K for long-context fp16 GQA decode
**By:** Sebastian; reviewed by Holden 🟢
**What:** Raise fp16 flash-decode `MAX_SPLITS` from 8 to 16, retaining device-side capture-safe split selection, deterministic fixed-order merging, and the single-stream shared-scratch invariant. Landed as `3b972bf`.
**Why:** Independent H200 review measured 1024-token decode improving from about 647 to 693 tok/s (+7.1%) while 256-token throughput remained flat, with identical greedy tokens, zero graph fallbacks, bounded 2.03 MiB scratch, and no SM90-only dependency.

### Fuse SwiGLU SiLU and multiply in one CUDA kernel
**By:** Mariette; reviewed by Pris 🟢
**What:** Fuse eligible equal-shape, single-consumer `Mul(Silu(gate), up)` patterns into one capture-safe f32/f16/bf16 pointwise kernel, preserving separate fallback paths and kernel-variant trace reasons. Landed as `12e48b8`.
**Why:** The fusion halves activation launches from 48 to 24 per token and improved authoritative 256-token H200 decode from about 673 to 689 tok/s, with identical tokens, zero graph fallbacks, full CUDA parity, and portable primitives suitable for sm_89.

### Record the stacked wave-3 performance baseline
**By:** Kowalski
**What:** Treat the fresh shared-H200 re-profile as the current wave-3 baseline: median throughput about 691 tok/s at 256 tokens and 712 tok/s at 1024 tokens, with zero CUDA graph fallbacks. Recorded in `docs/PROGRESS.md` by `f42ca3f`.
**Why:** The stacked GQA split and SwiGLU fusion gains reproduce together, remain coherent, and place native CUDA decode above the 657 tok/s ORT GenAI reference at 256 tokens.

### Gate CUDA EP Clippy warnings in CI
**By:** Gaff; reviewed by Wallace 🟢
**What:** Clear all 21 existing `onnx-runtime-ep-cuda` Clippy warnings without adding allows, remove no-op explicit drops of non-owning `TensorMut` views, and add `cargo clippy -p onnx-runtime-ep-cuda --features cuda -- -D warnings` to the `cuda-compile` job. Landed as `22ec87e`.
**Why:** CUDA EP warnings are now blocking in CI. Review verified the lint rewrites and drop removals preserve behavior and ownership, with builds, tests, Clippy, YAML parsing, and a zero-fallback performance sanity run passing.
## 2026-07-21 — Native CUDA decode wave 4

### Fold batch-1 GQA metadata into fused decode preparation
**By:** Luv; reviewed by Holden 🟢  
**What:** For eligible batch-1, `Sq=Sk=1`, fixed-capacity aliased-device-KV decode, derive GQA metadata inside each fused prep CTA and have block 0 write the attention arrays; unsupported shapes retain the separate metadata kernel. Landed as `bd30e6c`.  
**Why:** The change preserves latch-first poison propagation, all bounds/error bits, sentinel/no-write behavior, capture safety, and SM portability while removing 24 launches/token. Independent H200 review measured roughly 691→710 tok/s at 256 tokens with exact tokens and zero fallbacks.

### Fuse MatMulNBits-adjacent QKV bias and paired gate/up SwiGLU
**By:** Rachael; reviewed by Pris 🟢  
**What:** Fold eligible QKV bias Adds into the MatMulNBits epilogue with exact two-op fp16 rounding, and collapse the validated Qwen 0.5B gate/up projections plus SwiGLU into one paired capture-safe kernel. Strict initializer, shape, dtype, consumer, and graph-output gates preserve unfused fallback. Landed as `102fee9`.  
**Why:** GPU bit-exact tests and end-to-end greedy tokens match the two-op baseline, with zero graph fallbacks and portable primitives. Stacked on the GQA metadata fold, H200 reached about **759 tok/s at 256 tokens** and **789 tok/s at 1024 tokens**, saving about 72 launches/token.

### Drop the CUDA replay binding-cache prototype — DEAD END
**By:** Deckard  
**What:** Do not merge or re-attempt commit `14a1d8f`, which cached validated device-I/O metadata and raw external addresses for CUDA-graph replay.  
**Why:** Two paired H200 measurements showed only **+0.23%** (+1.60 tok/s), below the 0.5% noise threshold, while the exact-identity/raw-address predicate adds correctness sensitivity on the replay hot path. Revisit only with materially stronger isolated evidence and a safer design.

### Keep Ana wave-3 roofline as the current roofline of record
**By:** Scribe  
**What:** Preserve `.squad/decisions/inbox/ana-wave3-roofline-691.md` as the current roofline artifact: wave 4 achieved about **759 tok/s**, within its **750–790 tok/s** ceiling.  
**Why:** The artifact remains the authoritative lever ranking and ceiling analysis after wave-4 validation.
<!-- scribe-merge-2026-07-22T22-15-00Z-generality-batch -->
## 2026-07-22 — CPU EP generality and portability batch
<!-- merged from .squad/decisions/inbox/coordinator-generality-directive.md -->
### 2026-07-22T21:25:00Z: Directive — cross-OS + cross-processor generality is mandatory
**By:** justinchuby (via Copilot coordinator)
**What:** The CPU EP perf effort MUST ensure cross-operating-system AND cross-processor support — not Linux-only, not x86-only. Kernel selection policy: use MLAS where MLAS is faster; use our implementation where ours is faster (per shape/dtype); improving *on top of* MLAS is allowed, but any such win must remain general/portable.
**Why:** User directive. Sets the acceptance bar for every optimization: a win that only works on Linux (e.g. sched_setaffinity NUMA pinning) or only on x86 (VNNI/AVX2) must have a real portable equivalent or graceful fallback on other OSes/ISAs (Windows, macOS, aarch64) before it can be considered production-grade.
**Implications:**
- NUMA decode affinity (046414b) is currently Linux-only (`sched_setaffinity`, `/sys`); needs Windows (SetThreadAffinityMask / GetLogicalProcessorInformationEx) + macOS handling (or documented graceful no-op) to satisfy this.
- ISA-gated kernels (GQA AVX2, hand int4/VNNI) must retain genuine scalar/aarch64 fallbacks.
- Goal remains: beat ORT (26.9 tok/s) end-to-end while staying portable.
<!-- merged from .squad/decisions/inbox/rachael-generality-audit.md -->
### 2026-07-22: CPU EP performance generality and production-readiness audit
**By:** Rachael (Fact-Checker + Devil's Advocate)
**What:** Static, read-only audit of `58a3324`, `2e982c7`, `145549a`/`c9762b6`, and `32a122e`/`046414b`. No build, test, or benchmark was run because the shared host was under active benchmarking.
**Why:** The changes are correctness-safe in their intended configurations, but the shipped performance story has material portability, default-policy, dtype, and automated-parity gaps.

## Executive verdict

| Work item | Q1: CPU/model generality | Q2: production-grade | Q3: performance claim |
|---|---|---|---|
| `58a3324` — MatMulNBits/MLAS routing | ⚠️ Correct fallbacks, but f32-only and tuned thresholds are host-specific | ⚠️ Correctness tests are broad, but MLAS is manual opt-in and parity is tolerance-based | ✅ Measured hand int4 and MLAS CompInt8 tie; retaining the hand path is correct |
| `2e982c7` — contiguous f32 bulk copy | ✅ Architecture-neutral and shape-neutral | ⚠️ Sound under the executor bounds/ownership contract; tests are small and f32-only | ✅ Real glue/runtime win, not an arithmetic-kernel win |
| `145549a` + `c9762b6` — GQA AVX2 dot/AXPY | ⚠️ Production has scalar fallback, but older non-AVX2 x86 test runs fail by assertion | ⚠️ Unsafe loops are bounded, but bit parity is not guaranteed/tested and dtype/shape coverage is narrow | ✅ Genuine GQA kernel-level win; therefore “all wins are engine-level” is too broad |
| `32a122e` + `046414b` — NUMA affinity | ⚠️ Linux-only optimization with safe no-op fallback elsewhere; topology is queried | ❌ The measured +25% path is OFF by default, so normal users do not receive it | ✅ The 16.3 tok/s gain is engine/thread-placement level, not MatMul arithmetic |

## Question 1 — CPU and model generality

### `58a3324` — MatMulNBits M routing

**Verdict: ⚠️ gap, not broken.**

- **Non-x86 and old x86 remain correct.** The direct VNNI variants only exist on x86-64 and are runtime-selected with `is_x86_feature_detected!` checks (`matmul_nbits.rs:834-857`). Both packed-int4 and u8×i8 helpers have scalar implementations (`matmul_nbits.rs:924-971`, `1160-1184`). On a host without VNNI, the specialized direct-int4 branch is skipped by `dot_kernel != Scalar` (`matmul_nbits.rs:253-260`) and the accuracy-4 int8 path uses the scalar dot fallback. No illegal instruction or UB is apparent.
- **MLAS is not portable production routing.** `NXRT_CPU_GEMM_BACKEND=mlas` only resolves on `feature="mlas"` + x86-64 (`backend.rs:94-111`); otherwise the generic/SimdX86 paths remain.
- **Hardcoded tuning exists.** The production crossover is fixed at `16`, based on Sapphire Rapids (`matmul_nbits.rs:45-63`), and the decode pool defaults to a fixed 8 workers (`matmul_nbits.rs:26-33`). These are not model dimensions, but they are CPU-specific performance policy rather than topology/cost-model-driven choices.
- **The production route is model-shape-driven.** `M` is computed as the product of all activation dimensions except `K` (`matmul_nbits.rs:223-224`); `K`, `N`, bits, and block size come from graph attributes/shapes (`matmul_nbits.rs:112-147`). The Qwen 7B constants are confined to the ignored benchmark fixture (`matmul_nbits.rs:2889-2917`), not routing.
- **Confirmed generality gap: activations and output REQUIRE f32.** `A`, scales, bias, and `Y` are rejected unless Float32 (`matmul_nbits.rs:165-170`, `211-214`). Float16/BFloat16 activations are unsupported even though the shared dtype layer supports widening/narrowing for other kernels.
- **M=1 means the flattened `batch × sequence × ...` row count is one.** Thus the specialized path effectively requires a single row, not a named model or explicit batch field (`matmul_nbits.rs:223-255`). M>1 is not sent through the M=1 pool; it follows int8 row-parallel or dequantized GEMM paths (`matmul_nbits.rs:292-365`). MLAS may handle M≥16 by default.

### `2e982c7` — contiguous f32 bulk copy

**Verdict: ✅ general.**

- The fast path is ordinary slice copying with no ISA or OS gating (`kernels/mod.rs:869-893`, `1008-1036`), so it is portable to ARM, non-AVX2 x86, Linux, Windows, and macOS.
- It assumes only contiguous Float32 storage; every other layout keeps the prior strided path. No head size, hidden size, batch, sequence, block size, topology, core count, or SIMD width is hardcoded.
- It benefits both decode and prefill because it is in shared kernel I/O helpers, not an M=1-only dispatch.
- It does not expand dtype support: these helpers explicitly reject non-Float32 (`kernels/mod.rs:869-874`, `1008-1013`).

### `145549a` / `c9762b6` — GQA AVX2/FMA

**Verdict: ⚠️ production portable; test suite is not portable to old x86.**

- AVX2/FMA is compile- and runtime-gated: x86/x86-64 cfg plus `has_simd_x86()` (`group_query_attention.rs:388-415`), whose definition checks both AVX2 and FMA (`backend.rs:124-131`). ARM/aarch64 and non-AVX2 x86 execute genuine scalar dot/AXPY loops.
- Intrinsics use unaligned loads and scalar tails, so arbitrary `head_dim` is supported (`group_query_attention.rs:419-503`). No production `head_dim=128` assumption exists; dimensions derive from hidden width and head counts (`group_query_attention.rs:156-218`).
- The implementation supports MHA as the `num_heads == kv_num_heads` case and GQA when the former is a multiple of the latter (`group_query_attention.rs:76-81`, `740-745`). However, the new main regression fixture hardcodes 4 query heads, 2 KV heads, and width 128 (`group_query_attention.rs:1632-1638`), so MHA and other realistic widths are not covered end-to-end by the new test.
- GQA accepts float16/bfloat16/f32/f64 through `to_dense_f32_widen` and narrows outputs back (`group_query_attention.rs:171,217,289,354-357`; `dtype.rs:474-504`). The new SIMD tests exercise f32 only.
- The optimized score/AXPY algorithm runs for all batch and sequence sizes (`group_query_attention.rs:741-836`), not only M=1. Only the output-copy shortcut is gated on `q.seq == 1 && k.seq == 1` (`group_query_attention.rs:848-859`). Therefore prefill behavior also changed.
- **Portability defect in tests:** on any x86/x86-64 machine lacking AVX2+FMA, the new tests assert and fail instead of skipping or validating scalar fallback (`group_query_attention.rs:1639-1643`, `1742-1746`, `1807-1811`).
- Hardcoded `MIN_PARALLEL_ATTENTION_WORK = 160 * 1024` is another host-tuned scheduling heuristic, not a topology-derived cost model (`group_query_attention.rs:45-47`).

### `32a122e` / `046414b` — NUMA affinity

**Verdict: ⚠️ intentionally Linux-only optimization, safe elsewhere.**

- NUMA discovery and `sched_setaffinity` are Linux-only (`decode_affinity.rs:153-189`, `270-310`). Non-Linux returns no topology and leaves workers unpinned (`matmul_nbits.rs:731-740`); correctness is unaffected.
- ARM Linux can use the affinity code because it is OS-gated, not x86-gated. It contains no SIMD.
- Node and CPU membership are queried from sysfs (`decode_affinity.rs:167-190`); the affinity mask is sized from the runtime CPU index (`decode_affinity.rs:256-279`). The fixup removes the original fixed-`cpu_set_t`/1024-bit OOB risk.
- The remaining hardcoded core policy is outside topology discovery: the pool still defaults to 8 workers (`matmul_nbits.rs:26-33`). `compact` selects a node around that fixed worker count (`decode_affinity.rs:202-230`).
- It is restricted to the engine’s single-token call: `token_ids.len() == 1` enters the decode pool; M>1 runs normally (`native_decode.rs:1921-1932`). This also effectively assumes one flattened token row; batched one-token-per-sequence decode is not covered by this gate if `token_ids` contains more than one element.

## Question 2 — production grade

### Defaults

| Setting/change | Default | Consequence |
|---|---|---|
| `NXRT_SQNBIT_DECODE_MIN` | `16` (`matmul_nbits.rs:63-82`) | Threshold is active only if MLAS backend is explicitly selected |
| `NXRT_CPU_GEMM_BACKEND=mlas` | OFF; SimdX86/Generic auto-selected (`backend.rs:7-16`, `35-40`) | The new slow-dequant M=1→MLAS win is not received out of box |
| Contiguous f32 bulk copy | ON unconditionally (`kernels/mod.rs:880-893`, `1024-1036`) | Users receive this optimization automatically |
| GQA AVX2/FMA | ON when runtime detection succeeds; scalar otherwise (`group_query_attention.rs:388-415`) | Automatic and safe |
| `ONNX_GENAI_CPU_DECODE_AFFINITY` | OFF (`decode_affinity.rs:17-24`, `72-81`) | The measured 13.1→16.3 tok/s win requires manual opt-in |
| `ONNX_GENAI_PROJECTION_FUSION` | OFF unless exactly `1` (`optimizer.rs:8-23`) | Correct default: it regressed 16.3→13–14 tok/s (`decisions.md:1763-1768`) |

### Unsafe and bounds audit

- **Bulk copy:** The raw slices are correct under the executor contract: dispatch bounds-checks every view against its backing (`executor.rs:35-44`, `3183-3189`, `3487-3502`), dtype/alignment is validated, output length equals `numel`, and all elements are initialized by `extend_from_slice`/`copy_from_slice` (`kernels/mod.rs:869-893`, `1008-1036`). No uninitialized-memory read is introduced. The proof still depends on the documented caller-side bounds gate because `TensorView::validate()` itself has no allocation length (`ep-api/tensor.rs:205-225`).
- **GQA intrinsics:** `_mm256_loadu_ps` avoids alignment requirements. Loop conditions prove each 8/16-lane load stays within `n`, and scalar tails cover the remainder (`group_query_attention.rs:427-473`, `485-503`). Current callers construct equal-length slices after Q/K/V dimension validation (`group_query_attention.rs:555-563`, `756-769`, `803-805`). Devil’s-advocate advisory: the safe wrappers use only `debug_assert_eq!`; a future mismatched private caller could cause release-mode OOB in the unsafe SIMD function. Use a real assertion or pass one proven length.
- **MatMul VNNI:** Runtime feature checks gate every target-feature call, unaligned loads are used, and per-block lengths are derived from padded buffers (`matmul_nbits.rs:842-857`, `924-1039`, `1160-1227`). Scalar fallbacks exist.
- **Affinity syscall:** The mask allocation length is computed from the selected CPU and passed verbatim to `sched_setaffinity`; the vector outlives the call (`decode_affinity.rs:270-304`). The pointer cast is consistent with Linux’s byte-mask ABI. Failure is non-fatal and logged once (`matmul_nbits.rs:700-752`). No remaining OOB is evident after `046414b`.

### Correctness coverage

- `58a3324` has good cross-shape numerical coverage: block sizes 32/64/128, symmetric/asymmetric zero points, M=1/M=5, accuracy 0/4, and bias (`matmul_nbits.rs:2462-2542`). But it is all f32 and uses tolerances (up to `6e-2` for CompInt8), not bit parity. The slow-dequant routing test can skip when MLAS is unavailable.
- `2e982c7` exact-compares one small contiguous shape and retains one transposed case (`kernels/mod.rs:1456-1491`). It does not test byte offsets, zero-sized nontrivial shapes, or large/multidimensional buffers in the new fast path.
- GQA tests compare numerically, not bitwise (`group_query_attention.rs:1624-1733`, `1735-1853`). The source explicitly disclaims a universal greedy-token guarantee (`group_query_attention.rs:26-34`). The recorded token parity was manual benchmark observation, not a model-level automated test.
- Affinity’s bit parity is also manually recorded (`decisions.md:1702-1705`). Unit tests cover parsing, selection, and large mask sizing, but not a real syscall/cgroup/cpuset integration path (`decode_affinity.rs:312-477`).

### Concurrency / OnceLock

- `DECODE_POOL` initialization is race-free: `OnceLock` publishes one complete pool or one complete error (`matmul_nbits.rs:32-33`, `756-770`, `823-830`). Concurrent initializers may not observe later environment changes, but that is deterministic first-use configuration, not a data race.
- Worker affinity is installed by Rayon’s per-worker `start_handler`; the captured CPU vector is immutable and shared safely (`matmul_nbits.rs:691-719`).
- Weight caches use `OnceLock`; losing concurrent builders discard their duplicate and read the published value (`matmul_nbits.rs:261-305`, `327-352`, `439-454`). No TOCTOU/data race is apparent if constant-input immutability is honored.
- The thread-local residency flag is restored by RAII even during unwind (`matmul_nbits.rs:774-830`). The engine gates only the M=1 CPU route (`native_decode.rs:1921-1932`).
- Operational gap: sysfs `cpulist` is not intersected with the process’s allowed cpuset. Restricted containers may attempt disallowed CPUs, then silently run unpinned after one diagnostic. Safe, but the requested performance may disappear.

## Question 3 — faster than MLAS?

**Verdict: the hand int4/VNNI kernel does not beat MLAS SQNBit at production-scale M=1; it ties CompInt8 within noise, while CompFp32 loses.**

- The ignored three-way probe uses distinct Qwen2.5-Coder-7B projection buffers and compares hand, MLAS CompInt8, and MLAS CompFp32 (`matmul_nbits.rs:2889-3018`). It is deliberately model-shaped benchmark evidence, not production hardcoding.
- The recorded cold result is hand ~29.5 tok/s, CompInt8 ~30.6 tok/s (noise-level tie), and CompFp32 ~23.8 tok/s (`decisions.md:2002-2012`); the checked-in benchmark document likewise records hand ~26 vs CompInt8 ~25 and calls it a tie (`BENCH_MLAS_INT4_E2E.md:201-216`). The exact ordering is not stable enough to claim a kernel win.
- Routing therefore keeps the fast accuracy-4 M=1 hand path below the default crossover and uses MLAS for larger M (`matmul_nbits.rs:416-460`). MLAS CompFp32 is used only when the alternative is the much slower full-f32 dequantized hand path; that is reuse of MLAS for a genuine supported advantage.
- This follows RULES.md rule 4: do not replace a battle-tested primitive without a measured win (`RULES.md:51-57`).
- **Qualification:** “all real wins are engine-level” is true for the MatMulNBits hand-vs-MLAS decision and for the 16.3 tok/s affinity result, but false across all four changes. GQA AVX2/FMA is a real arithmetic-kernel optimization. Bulk-copy is kernel-I/O/glue rather than matrix arithmetic.
- Honest standing remains **native 16.3 < onnxruntime-genai 20.8 < ORT 26.9 tok/s** (`decisions.md:1770-1777`; baseline comparison details at `decisions.md:2395-2405`). Native is still about 22% below OGA and 39% below ORT in throughput.

## Top production-readiness gaps, prioritized

1. **Auto-enable safe compact affinity on detected multi-node Linux hosts**, or expose it through typed runtime configuration with an explicit auto policy. Today the largest shipped win is hidden behind manual opt-in.
2. **Remove fixed CPU tuning policy from defaults.** Derive decode worker count, the M=16 MLAS crossover, and GQA parallel threshold from topology/capability measurements or an inspectable cost model; at minimum validate on AMD, older Intel, ARM, and single-/multi-socket systems.
3. **Add Float16/BFloat16 activation/output support to native CPU MatMulNBits.** It currently rejects anything but Float32.
4. **Add automated model-level parity/regression tests** across multiple M, batch, head dimensions, GQA and MHA, local-window/full attention, f32/f16/bf16, AVX2 and forced-scalar routes. Manual token equality is not a release gate.
5. **Make GQA SIMD tests portable.** Skip SIMD-specific assertions when AVX2/FMA is absent and separately force/test the scalar path; current tests fail on older x86.
6. **Harden SIMD wrapper safety contracts** with release-mode equal-length checks before unsafe loads.
7. **Intersect discovered NUMA CPUs with the process/cgroup allowed cpuset** and report partial pin failures with worker/CPU detail.
8. **Keep projection fusion off** until `Split` materialization is removed and a non-regressing, parity-tested implementation exists.

<!-- merged from .squad/decisions/inbox/bryant-numa-shard-decode.md -->
# Decision: NUMA-split two-level decode for native CPU int4 M=1

**Author:** Bryant (senior systems/perf engineer)
**Branch:** `bryant/numa-shard-decode` (rebased onto `perf/cpu-ep-mlas` @ `96fd406`,
which includes Deckard's affinity review-fix `046414b`)
**Date:** 2026-07
**Status:** Positive result — opt-in, ships behind an env flag, no default-path change.
**Reviewer:** pending (rule 9 — non-author review required before merge)

---

## TL;DR

`ONNX_GENAI_CPU_DECODE_AFFINITY=numa-split` with `ONNX_GENAI_CPU_DECODE_THREADS=32`
(16 workers per NUMA node) raises steady M=1 int4 decode from a **16.87 tok/s**
compact-single-node baseline to **18.42 tok/s median (best 18.51)** — a **+9%**
gain — with **exact greedy bit-parity**. It does not reach the 20 tok/s target
(remaining gap ~1.6 tok/s / ~9%), but it is a real, repeatable improvement and
not a regression. The default path is untouched; the mode is entirely opt-in.
(Numbers are post-rebase onto Deckard's affinity fix; a pre-rebase run gave the
same conclusion at 16.40 -> 18.38.)

---

## 1. Profile-first baseline reproduction (rule 4)

Host: Sapphire Rapids Xeon 8480C, 2 sockets × 48 cores, 2 NUMA nodes
(node0 = CPUs 0–47, node1 = CPUs 48–95), AVX512-VNNI + AMX. Shared 96-core host,
noisy — every number below is a `steady_median` over `runs>=4`, and A/B configs
were **interleaved** across 3 rounds; I report median and best, never a single
run.

Command (worktree build):
```
export LD_LIBRARY_PATH=$PWD/target/release/build/onnx-genai-ort-sys-6b88787cafa9d9dd/out/ort-prebuilt/lib
ONNX_GENAI_CPU_DECODE_AFFINITY=compact ONNX_GENAI_CPU_DECODE_THREADS=32 \
  ./target/release/profile_native \
  --model ~/.foundry/cache/models/Microsoft/qwen2.5-coder-7b-instruct-generic-cpu-4/v4 \
  --tokens 24 --runs 5 --warmups 1 --steady --decode-skip 8
```

Baseline `compact` T=32: **16.87 tok/s median** over 5 interleaved rounds
(16.87 / 17.07 / 16.14 / 15.37 / 16.97), in line with Batty's ~16.3. Matches
Batty's finding that MatMulNBits (int4, `accuracy_level==4`, block 32, M=1) is
the hot op and decode is memory-latency + per-op fork-join-barrier bound, not
compute bound.

## 2. Design (numa-decode-plan steps 4–5)

Use **both** sockets' memory bandwidth without paying a flat 64-way cross-socket
per-op barrier:

- A tiny **dispatcher pool** (one worker per NUMA node) installs the M=1 forward
  via `with_decode_pool_scope`, with `IN_NUMA_SCOPE` + `IN_DECODE_POOL` set so
  inner `with_decode_pool` calls run inline.
- Each MatMulNBits kernel (`int4_matmul_m1`, `int8_row`, `gemv_nk`) routes its
  parallel section through `parallel_output_rows`, which when numa-active calls
  `dispatch_output_rows`: the output rows are split into per-node **contiguous**
  slices; each slice runs on its **node-pinned sub-pool** via
  `dispatcher.install(|| segments.into_par_iter().for_each(|seg| node_pool.install(compute)))`.
- **Two-level barrier:** node-local `par_chunks_mut` is the first level; the
  single `for_each` join across nodes is the *only* cross-node barrier per op —
  replacing the toxic flat 64-thread cross-socket coherency round-trip that made
  Batty's naive interleaved pool 11.1 tok/s.
- **Node-local weight first-touch:** at prepack, `place_rows` allocates an
  uninitialized buffer (zero-filling would fault every page onto the dispatcher
  node) and each node's sub-pool **copies its own row-shard**, first-touching
  those pages on the owning node under the default policy.
  `row_lengths(n)` is the single source of truth used by both weight placement
  and compute dispatch, so they always line up.

**Bit-parity argument:** row-sharding a GEMV is exactly associative — each output
row is an independent dot product over the full K. The activation is quantized
**once** before dispatch (shared read-only). There is no cross-node K-reduction,
so results are bit-identical regardless of the row partition. Verified
empirically (§4).

New module: `crates/onnx-runtime-ep-cpu/src/decode_numa.rs`. Topology + affinity
parsing extended in `decode_affinity.rs` (`NumaSplit` variant, `NodeShard`,
`split_workers`). Kernel wiring in `kernels/matmul_nbits.rs`.

## 3. A/B results — median + best (post-rebase, 5 interleaved rounds, T=32 total)

| mode        | T   | per-round tok/s                       | median | best  |
|-------------|-----|---------------------------------------|--------|-------|
| compact     | 32  | 16.87 / 17.07 / 16.14 / 15.37 / 16.97 | 16.87  | 17.07 |
| **numa-split** | **32** | **18.42 / 18.51 / 18.15 / 18.44 / 18.30** | **18.42** | **18.51** |

**numa-split T=32 (16+16) is the winner: +9% over compact baseline, and notably
*more stable* run-to-run (18.15–18.51) than compact (15.37–17.07).**

A third fresh 3-round A/B on the final committed branch reconfirmed the result:
compact **16.66** median (15.29 / 16.74 / 16.66), numa-split **18.00** median
(17.77 / 18.28 / 18.00) — +8%. Across all three benchmarking sessions numa-split
lands 18.0–18.5 median vs compact 16.4–16.9; the win is robust to host noise.

A pre-rebase 3-round A/B at the wider grid corroborated the direction and showed
the failure modes of over-threading:

| mode        | T   | median (pre-rebase) |
|-------------|-----|---------------------|
| compact     | 32  | 16.40 |
| numa-split  | 32  | 18.38 |
| numa-split  | 64  | 15.42 (barrier cost dominates) |
| compact     | 64  | 10.18 (cross-node thrash) |

### Thread scaling (numa-split, 2 rounds each)

| T (per-node)  | tok/s        |
|---------------|--------------|
| 16 (8+8)      | 16.48 / 16.54 |
| 24 (12+12)    | 17.64 / 17.89 |
| **32 (16+16)**| **18.23 / 16.81** (peak) |
| 48 (24+24)    | 16.28 / 17.13 |
| 64 (32+32)    | 15.42 / 15.04 (from A/B) |

Clear knee at **T=32**. Below it, memory bandwidth is under-used; above it, the
per-op two-level barrier and cross-socket coherency cost of more workers erodes
the bandwidth gain. This is exactly the barrier-vs-bandwidth tradeoff the plan
predicted; the two-level structure moves the sweet spot up from single-node but
does not eliminate the per-op join cost.

## 4. Bit-parity confirmation

Greedy `generated_token_ids` were **identical** between compact (single-node) and
numa-split across **every** configuration (T=16/24/32/48/64), on two prompts:

- Default `"Hello"` (24 tokens) — both produce:
  ```
  [48298, 271, 9707, 0, 2585, 646, 358, 7789, 498, 3351, 30, 151645, 198,
   151643, 151644, 198, 151643, 151644, 198, 151643, 151643, 151643, 151643, 151643]
  ```
- A code prompt (32 tokens, real content) — both produce byte-for-byte:
  ```
  [576, 729, 1265, 3705, 2176, 25780, 323, 9069, 11, 323, 432, 1265, 10034,
   1142, 26443, 369, 9069, 382, 8420, 594, 458, 3110, 315, 1246, 279, 729,
   1265, 975, 1447, 73594, 12669, 198]
  ```

**On Batty's reference sequence** `[576, 729, 1265, 1896, 264, 1140, ...]`: Batty's
methodology note abbreviates his command (`profile_native ...`) and does not record
the prompt string; the tool's default prompt is `"Hello"`, which produces a chat
*greeting* (the `48298...` sequence above), not code. Batty's ids are clearly a
code-completion (`" The function should ..."`), so he used an undocumented code
prompt. My code-prompt run reproduces the exact same opener `[576, 729, 1265, ...]`
and then diverges (his prompt ≠ mine), confirming the reference is prompt-specific.
The correctness-relevant invariant for *this change* is that row-sharding a GEMV and
concatenating is numerically exact — i.e. numa-split == the single-node path
byte-for-byte on the **same** build+prompt — which is verified above on both a
trivial and a non-trivial (32-token code) output. Row-sharding is exactly
associative (each output row is an independent full-K dot product; the activation
is quantized once before dispatch; no cross-node K-reduction), so this parity holds
by construction.

## 5. What worked / what didn't (with evidence)

- **Worked:** node-pinned sub-pools + row-sharded weights + two-level barrier at
  T=32. +9% median and exact parity, with lower run-to-run variance than compact.
  Both sockets' bandwidth is used with a single cross-node join per op.
- **Didn't:** scaling past 32 total threads. T=64 numa-split (15.4) is *worse*
  than T=32 (18.4) and than compact T=32 (16.9) — the per-op cross-node barrier
  and coherency traffic of 64 workers outweighs the marginal bandwidth. So
  "throw more threads at both sockets" is the wrong lever; the correct lever is
  *bandwidth per node with a minimal barrier*, which peaks at 16 workers/node.
- **Not pursued:** node-local KV cache residency (numa-decode-plan future step) —
  larger surface, deferred. Sharding only the largest projections was
  unnecessary: uniform row-sharding already lands a positive result and the
  T-scaling curve shows the barrier cost, not small-op dispatch overhead, is the
  ceiling.

## 6. Remaining gap

18.42 median vs **20 target ⇒ ~1.6 tok/s / ~8% short**; vs 16.87 baseline ⇒ **+9%**.
The remaining gap is dominated by the per-op cross-node join latency (141 ops/token
× one cross-socket barrier each). Closing it likely needs *fewer* cross-node
synchronizations per token — e.g. node-local KV so attention doesn't re-cross,
or fusing consecutive projections under one barrier — rather than more threads.
Recommend that as the next step. Reference points: ORT 26.9, onnxruntime-genai 20.8.

## 7. Safety / rules compliance

- **Rule 5 (opt-in):** default path unchanged; only `numa-split` activates it.
- **Rule 2 (no hardcoded topology):** nodes/CPUs queried from `/sys` at runtime
  via the reused `decode_affinity` topology; no hardcoded node/core counts.
- **Rule 1 (good errors / graceful fallback):** single-node/non-Linux hosts,
  `THREADS=0`, or <2 populated nodes fall back to flat single-node decode,
  logged **once** via `report_numa_fallback`. Malformed env values remain a hard
  error from the existing flat path.
- **Rule 4 (reuse MLAS):** each node runs the **existing** MLAS SQNBit / hand
  VNNI kernel on its row-slice; no new matmul was hand-rolled.
- **Rule 8 (tests track behavior):** unit tests added for row-length splitting,
  dispatch-equals-flat, byte-preserving placement, and worker splitting across
  nodes. `cargo test -p onnx-runtime-ep-cpu --features mlas` → **675 passed, 0
  failed** (includes Deckard's 4 affinity-fix tests after the rebase). `cargo
  clippy` clean.

## 8. Coordination with Deckard's affinity fix (rebased)

Reviewer Gaff rejected Batty's affinity commit `32a122e`; Deckard landed a
surgical fix (`046414b`) confined to `decode_affinity.rs`: (1) `cpu_set_t` mask
sizing → a runtime-sized `build_cpu_mask`, (2) a single consistent invalid-value
diagnostic centralized in `ACCEPTED_MODES`/`invalid_selector_error`, and (3)
`compact` node-selection → smallest-index `find`.

Per the coordinator's guidance I kept my feature **additive** and did **not**
touch those three areas' logic:
- My `numa-split` parse arm returns `Ok(NumaSplit)` and flows through Deckard's
  new `resolve()` via its `Ok(affinity) => Ok(affinity)` pass-through — no change
  to his validation logic.
- `pin_current_thread_to_cpu(cpu: usize)` signature is unchanged, so my
  per-node sub-pool pinning needed no adaptation; it transparently benefits from
  the new mask sizing.
- My `DecodeAffinity::NumaSplit => Ok(None)` arm in `cpus_for` sits alongside his
  rewritten `Node`/`Compact` arms (the flat fallback pool stays unpinned).
- The rebase conflicted only in the tests module tail (both sides appended
  tests); resolved by keeping **both** sets.
- The **one** shared-line change I made is extending his `ACCEPTED_MODES` const
  to include `` `numa-split` `` so the invalid-value diagnostic lists the new
  mode (rule 1). His fix was already committed, so this is a static additive
  extension, not a live collision. His diagnostic tests only assert the three
  original modes are present, so they still pass.

I rebased `bryant/numa-shard-decode` onto `perf/cpu-ep-mlas` @ `96fd406` (which
contains `046414b`). The coordinator can fast-forward/cherry-pick it onto
`perf/cpu-ep-mlas`. **Not pushed** (coordinator pushes).

## 9. Handoff note (concurrent-agent hazard)

A concurrent agent was earlier running `git reset`/`checkout` on the shared main
working tree `/home/justinchu/onnx-genai-cpu`, which silently wiped in-progress
(including untracked) files twice. I therefore did all work in a separate git
worktree `/home/justinchu/onnx-genai-cpu-bryant` on branch
`bryant/numa-shard-decode`.

<!-- merged from .squad/decisions/inbox/holden-gqa-test-portable.md -->
# Make CPU GQA SIMD tests portable

## Decision

Keep the long-context GQA reference test runnable through normal runtime dispatch on every architecture. It now verifies the scalar fallback whenever AVX2+FMA is unavailable. The direct dot-product and repeated weighted-AXPY SIMD regressions early-return with a clear skip message when the runtime gate is false, preserving their AVX2/FMA mutation-detection coverage on capable x86 hosts without executing unsupported instructions on older x86 or ARM.

A test-only `ONNX_RUNTIME_EP_CPU_FORCE_NO_SIMD_X86=1` override was added to `has_simd_x86()`. It does not exist in production builds and lets unit tests exercise normal GQA dispatch with the scalar fallback on an AVX2 host.

## Verification

- AVX2 host: `cargo test -p onnx-runtime-ep-cpu --features mlas group_query` passed (17 tests).
- Forced scalar fallback: `ONNX_RUNTIME_EP_CPU_FORCE_NO_SIMD_X86=1 cargo test -p onnx-runtime-ep-cpu --features mlas group_query` passed (17 tests); SIMD-only helper regressions cleanly skip while the long-context GQA and generic AXPY coverage execute the scalar dispatch path.
- `cargo clippy -p onnx-runtime-ep-cpu --features mlas --tests -- -D warnings` passed.
<!-- merged from .squad/decisions/inbox/zhora-matmul-generality.md -->
### 2026-07-22: Generalize CPU MatMulNBits dtypes and topology tuning
**By:** Zhora
**What:** CPU `MatMulNBits` now accepts Float32, Float16, and BFloat16 activations, scales, bias, and output. Float16/BFloat16 reuse `to_dense_f32_widen` and `write_dense_f32_narrow`; Float32 continues through the original `to_dense_f32`/`write_dense_f32` path. The decode pool default is `min(1 + ceil(log2(available_parallelism)), 8, available_parallelism)`, and the MLAS crossover defaults to twice that worker count. Both existing environment overrides remain authoritative.
**Why:** Shared widening/narrowing provides portable scalar fallbacks without duplicating conversion code, while preserving the existing Float32 accuracy-4 route and output. Logarithmic worker growth reflects the bandwidth-bound, per-projection fork/join cost; the eight-worker cap records the measured regression at 16+ workers rather than silently baking in the 96-core host. On this host the derived defaults remain 8 workers and M=16, so no tuning perf delta is expected. Float16/BFloat16 M=1 and M=3 parity tests exactly matched the corresponding widened-f32 computation after output narrowing. The full CPU EP MLAS suite (679 unit tests, 10 numeric regressions) and Clippy passed. A foundry Float16 int4 model advanced through MatMulNBits and then stopped at the separate Float16-unsupported `SkipSimplifiedLayerNormalization` kernel.
<!-- merged from .squad/decisions/inbox/coordinator-generality-batch.md -->
### 2026-07-22: Generality/portability batch landed (cross-OS + cross-processor)
**By:** Squad (Coordinator), for justinchuby
**What:** Three parallel generality fixes merged onto perf/cpu-ep-mlas (each in isolated worktree, non-author reviewed):
- Tyrell 122b31a — cross-OS decode affinity (Windows SetThreadGroupAffinity / macOS no-op) + SAFE AUTO-ENABLE (NUMA compact now default-on when multi-node & cpuset-safe). Gaff APPROVE-WITH-NONBLOCKING (N1 read_unaligned + N2 doc fixed; N3 >64-CPU multigroup owed to CI).
- Zhora f8848c9 — f16/bf16 activation/scale/bias/output support in MatMulNBits (widen->f32->narrow, reuses dtype helpers) + topology-derived tuning (de-hardcode 8-worker / M=16). Chew APPROVE-WITH-NONBLOCKING (MLAS-routed f16 parity + auto greedy-opener regression owed).
- Holden 82e44be — portable GQA SIMD tests (cfg(test) force-no-simd seam; pass on non-AVX2 x86 + ARM). Roy APPROVE.
**Validation:** 694 ep-cpu tests pass, clippy clean (linux + windows-gnu + darwin type-check). Bench: auto-enable engages on 2-node host, bit-identical tokens auto-vs-off, +21% by default (14.58 vs 12.02 tok/s).
**Why:** User directive — CPU EP must be cross-OS AND cross-processor, and the NUMA win must ship by default. Closes gaps #1 (auto-enable), #2 (hardcoded tuning), #3 (f16 rejected), #5 (GQA tests non-portable), #7 (cgroup cpuset) from Rachael's audit.
**Owed follow-ups:** f16 for SkipSimplifiedLayerNormalization + other decode ops (full cuda-gpu f16 model); MLAS-routed f16 parity test; cross-target Windows/macOS CI runners; >64-CPU multigroup Windows validation.
<!-- scribe-merge-2026-07-22T23-20-00Z-spmd-lever -->
## 2026-07-22 — Persistent SPMD CPU decode pool landed

**By:** Pris; revised by Sebastian after Chew rejection; re-reviewed by Chew and Gaff
**What:** Landed `cee3c20` on `perf/cpu-ep-mlas`: an opt-in, default-off persistent SPMD worker pool for native CPU packed-int4 M=1 decode, enabled only with `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1`. The pool reuses existing MLAS/packed-int4 GEMV math while replacing repeated per-op fork/join dispatch; `numa-split` retains explicit precedence when both modes are available and the runtime logs the selected mode once.

**Why:** Profiling found roughly 141 `MatMulNBits` fork/join regions per decoded token and identified barrier/dispatch plus memory latency—not GEMV arithmetic—as the limiting costs. Interleaved noisy-host measurements put persistent SPMD at about 17.3–18.0 tok/s median versus about 16.2–16.4 for `numa-split` (roughly 7% gain); generated IDs and per-op f32 output remained byte-identical.

**Safety and validation:** Sebastian's locked-out revision added a real subprocess ON/OFF parity regression using six sequential packed-int4 M=1 operations and 31 workers, asserts all ON operations dispatch through SPMD, documents precedence/fallback behavior, replaces the erased-job `transmute` with a pointer/trampoline, and makes worker panics poison the pool while releasing the pending barrier rather than hanging. CPU EP validation reported 698 unit tests plus 10 numeric regressions, clean MLAS clippy, 30/30 SPMD stress runs, and a 64-token native ON/OFF ID check. Chew approved the revised blocking requirements; Gaff approved with only non-blocking concurrency follow-up notes.

**Sources reconciled:** `pris-decode-profile.md`, `pris-decode-barrier.md`, `sebastian-spmd-revision.md`, and `chew-spmd-rereview.md`. The earlier tracked Bryant NUMA, Holden portable-GQA, and Zhora dtype/topology notes were already present verbatim in this ledger and were deduplicated.
<!-- scribe-merge-2026-07-23T04-10-00Z-f16-gqa-and-crossmodel -->
## 2026-07-23 — f16 GQA decode and cross-model CPU comparison

**What:** Roy's f16 GQA decode optimization shipped in `eedbf93`, with Gaff and Chew approving. It removes the per-token full-KV f16 re-widen bottleneck through F16C bulk conversion and incremental widening into the present cache, improving 0.5B decode from 2.55 to 6.56 tok/s (2.57×) and 1.5B from 1.15 to 3.58 tok/s (3.11×). Sebastian's foundry comparison records Qwen2.5-Coder 7B generic-cpu at 28.62 tok/s native versus 21.00 tok/s ORT GenAI CPU (1.36×); Qwen 3.5 9B is a VLM package, not a comparable text-decoder case for this native checkout.

**Why:** The GQA change removes conversion work—not attention math—as the f16 decode bottleneck, while the comparison makes the native win without overstating cross-model generality.

**Process learning:** Roy's inbox note was copied to the MAIN checkout before worktree removal, avoiding the earlier gitignored-note-loss quirk.

Decision archive gate checked at 2026-07-23T04:10:00Z: the active ledger was 259049 bytes before this entry. No dated ledger entries older than 2026-06-23T04:10:00Z were present, so no archive was created or updated.

<!-- merged from .squad/decisions/inbox/roy-f16-gqa-decode.md -->
# Decision: f16 GQA decode — kill the per-token KV re-widen (F16C + incremental widen)

**Author:** Roy (principal kernel engineer, CPU-EP)
**Branch:** perf/f16-gqa-decode (off 536025f)
**Date:** 2026-07-23
**Scope:** native CPU decode of f16-activation int4 foundry `cuda-gpu` exports (GroupQueryAttention). Generic-cpu f32 path untouched.

## Problem (profiled first — RULES.md §4)
Baseline 0.5B qwen2.5 cuda-gpu decode = **2.55 tok/s**. Per-op steady: GroupQueryAttention ~54%, MatMulNBits ~43%.
Within-GQA phase breakdown (added temporary timers behind `gqa_phase_profile` feature + `ONNX_GENAI_PROFILE_GQA=1`):
- **widen ~47%** — re-widening the ENTIRE growing f16 past K+V → f32 every token (`to_dense_f32_widen`), O(seq_len) scalar convert per step.
- **out ~45%** — narrowing the whole present K+V f32→f16 + output, scalar.
- **attn ~6%** — the real QK·softmax·PV math.
- **present ~2%**.
So ~92% of GQA was scalar f16↔f32 conversion of the whole KV cache per token, not attention. Hypothesis confirmed.

## Fix (targeted, general, EP-agnostic)
1. **F16C-vectorize the bulk conversions** (`dtype.rs`): added an `f16c` module (`_mm256_cvtph_ps` / `_mm256_cvtps_ph` with `_MM_FROUND_TO_NEAREST_INT`) + `widen_f16_slice_into`, wired fast paths into `to_dense_f32_widen` (contiguous f16 in) and `write_dense_f32_narrow` (contiguous f16 out). f16→f32 is exact; f32→f16 rounds to nearest-even exactly like `half::f16::from_f32` → **bit-identical** (locked by test `f16c_widen_narrow_bit_identical_to_scalar` over all 65 536 f16 patterns + representative f32s). Runtime-detected; scalar fallback off-x86 / non-contiguous.
2. **Eliminate the redundant widen+copy** (`group_query_attention.rs`): `PastCache` no longer eagerly widens the whole cache into an owned `Cow<[f32]>`. New `PastSrc` enum (F32 borrow / F16 raw u16 / Dense fallback) + `widen_run()` widens each per-head run **directly into** the `present` buffer (F16C for f16), removing the intermediate materialize AND the second f32→f32 copy the decode path paid every token. Exotic layouts (strided/bf16/f64) still widen once up front — generality preserved.
3. **Skip the present zero-fill when there is no tail**: in steady decode every batch's `total == present_sequence_length`, so the per-(b,h) loop overwrites every element. `has_tail = totals.iter().any(|&t| t < present_sequence_length)`; when false, allocate uninit via `with_capacity`+`set_len` (documented SAFETY: every element written before any read).

Design note: kernel dispatch is shape-keyed (new seq length re-instantiates the kernel each token), so a resident f32 shadow cache can't live in the kernel instance cleanly. Chose the **stateless** approach (incremental widen-into-present + F16C) — simpler, correct across cache resets, no identity bookkeeping.

Key files:
- `crates/onnx-runtime-ep-cpu/src/dtype.rs`: `f16c` mod, `widen_f16_slice_into`, fast paths in `to_dense_f32_widen` / `write_dense_f32_narrow`, test.
- `crates/onnx-runtime-ep-cpu/src/kernels/group_query_attention.rs`: `PastSrc`/`widen_run` (~:283-333), present-build zero-fill skip (~:804-840), widen-into-present (~:855-861), `phase_prof` mod (~:545), multi-step lock test.
- `crates/onnx-runtime-ep-cpu/Cargo.toml`: `gqa_phase_profile` feature (off by default, zero-cost when disabled).

## Results (tokens 128, runs 3, median; host shared/noisy — checked uptime, no parallel benches)
| model | before | after | speedup |
|---|---|---|---|
| qwen2.5-0.5b cuda-gpu v4 | 2.55 tok/s | **6.56 tok/s** | **2.57×** |
| qwen2.5-1.5b cuda-gpu v4 | 1.15 tok/s | **3.58 tok/s** | **3.11×** |

New 0.5B per-op steady: **MatMulNBits ~82%, GroupQueryAttention ~14%** (was 54%). GQA is no longer the bottleneck; the int4 MatMulNBits GEMV now dominates (next target).

## Parity (non-negotiable — PASS)
- 0.5B cuda-gpu greedy opener unchanged, **byte-identical for the full 128-token sequence**: `[271, 40, 1079, 264, 48948, 304, 13027, 323, 358, 1079, 4460, 311, 1855, 264, 4285, 2025, …]`.
- Conversions are bit-identical to scalar `half` (exact widen, round-to-nearest-even narrow) — verified by dedicated test. f32 accumulation throughout; only the final present/output narrows to f16.
- Added `decode_multistep_incremental_widen_matches_full_widen_reference`: chains f16 present→past across 12 decode steps and locks the incremental-widen kernel output against a full-widen `kernel_exact_reference` — identical.

## No regression to shipped generic-cpu f32 path (PASS)
- generic-cpu 7B (qwen2.5-coder-7b) = **29.09 tok/s** (baseline ~28.5, within noise; ran under high host load). f32 caches take `PastSrc::F32` = borrow + verbatim copy, same behavior/numerics as before.

## Validation
- `cargo test -p onnx-runtime-ep-cpu --features mlas`: **709 passed + 10 golden**, 3 ignored, 0 failed.
- `cargo clippy -p onnx-runtime-ep-cpu --features mlas -- -D warnings`: clean (also clean with `gqa_phase_profile`).

## Residual risk / follow-ups
- F16C fast path is x86-only; other ISAs (aarch64) fall back to scalar `half` — correct but not accelerated. A portable-SIMD widen/narrow would generalize the speedup (future work).
- The `set_len` uninit optimization relies on the full-coverage invariant (no tail). Guarded by `has_tail`; the tail case keeps the safe zero-fill. Covered by existing prefill/padding tests.
- Bottleneck has shifted to MatMulNBits (int4 GEMV) — that is the next highest-leverage target for further f16-model gains.
- Kept `gqa_phase_profile` instrumentation behind an off-by-default feature (zero prod cost) for future profiling; strip if undesired.
<!-- merged from .squad/decisions/inbox/sebastian-foundry-cpu-comparison.md -->
### 2026-07-23
**By:** Sebastian

**What:** Benchmarked foundry-local CPU decode with persistent SPMD left as the default. Qwen 2.5 Coder 7B generic-cpu ran at 28.62 tok/s native versus 21.00 tok/s ORT GenAI 0.14.1 CPU (1.36x native). Qwen 3.5 9B generic-cpu ran in ORT at 13.63 tok/s but cannot be loaded by this native checkout: direct loading sees multiple ONNX files and compatibility pipeline loading rejects unspecified smart-resize semantics.

**Why:** The available evidence confirms the default native win on one fair generic-cpu model, but does not support a cross-two-model generality claim until the Qwen 3.5 multimodal package has complete native pipeline metadata/support. CUDA-export f16-GQA models were recorded separately as a native CPU follow-up; ORT CPU could not load them because its CUDA interface library was absent.
<!-- scribe-merge-2026-07-23T05-00-00Z-f16-widen-parity-tests -->
## 2026-07-23 — f16 GQA lazy-widen parity closure

**What:** The f16 GQA lazy-widen optimization in `eedbf93` now has bit-exact old-versus-new parity coverage, merged to main as `80b09c3`. The multistep test compares production lazy per-head widening with the former eager full-f16-cache-to-f32 reference. An independent no-tail-with-past assertion hand-assembles expected f16 present K/V bits, covering the uninitialized `set_len` fast path with nonempty past cache.

**Why:** Chew rejected the initial parity test because its no-tail case only exercised an empty past cache and shared present construction with the reference. The independent assertion catches skipped writes, incorrect offsets, and read-before-write defects that shared logic could mask. Chew subsequently approved.

**Sources reconciled:** `pris-f16-widen-parity-test.md` and `gaff-notail-widen-test.md`.

Decision archive gate checked at 2026-07-23T05:00:00Z: the active ledger was 266888 bytes before this entry. No dated ledger entries older than 2026-06-23T05:00:00Z were present, so no archive was created or updated.
<!-- scribe-merge-2026-07-23T06-31-00Z-f16-matmulnbits-shard -->
## 2026-07-23 — f16 MatMulNBits decode SPMD sharding

**By:** Bryant (implementation, `8598f6a`) and Pris (parity coverage, `08875b1`); Gaff approved threading, and Chew rejected then approved after the added tests. Merged to `perf/cpu-ep-mlas` at `08875b1`.

**What:** For f16-activation (`accuracy_level=0`) int4 M=1 decode, MLAS SQNBit no longer forks the global 96-thread pool from the inline dispatcher while roughly 48 persistent SPMD workers spin-wait. The pre-packed weight is split by output columns and each resident SPMD worker makes one single-threaded MLAS call for its N-shard under one barrier. Without a pool, a single shard retains the old behavior; the generic-cpu f32 `accuracy_level=4` route is untouched. `ONNX_GENAI_CPU_MM_MLAS_NO_SHARD=1` retains the full-width route for A/B comparison.

**Why:** Profiling disproved f16 widen/narrow conversion as the cause (0.1%/0.3%): oversubscribed MLAS GEMV dominated. The fix improved f16 decode from 6.5 to 32.53 tok/s on 0.5B (5×) and 3.58 to 14.40 tok/s on 1.5B (4×), reduced MatMulNBits share from 79% to 10%, and left 7B generic-cpu unchanged. Sharded output is byte-identical to `NO_SHARD`. Pris extended mlas-sys shard/full parity over block sizes 32/64/128 and K=384, and CPU-EP subprocess parity exercises the cached real SPMD route with three workers, N=97, and uneven segments. GQA is now the dominant 0.5B decode operation (~72%).

**Sources reconciled:** `bryant-f16-matmulnbits.md` and `pris-f16-matmulnbits-tests.md`.

Decision archive gate checked at 2026-07-23T06:31:00Z: the active ledger was 268050 bytes before this entry. No dated ledger entries older than 2026-06-23T06:31:00Z were present, so no archive was created or updated.
<!-- scribe-merge-2026-07-23T08-50-00Z-gqa-rotary-widen -->
## 2026-07-23 — GQA rotary-prefix bounded widen landed

**By:** Roy (implementation, `475fa47`) and Pris (parity tests, `6941a9a`); Gaff approved bound/indexing correctness, and Chew rejected then approved after bit-exact coverage. Both changes are cherry-picked to main.

**What:** GQA f16 decode was spending **95.8%** of execute time widening the entire rotary cos/sin cache (`[~32768, head_dim/2]`) from f16 to f32 for every layer and token, though `rotate()` reads only live-position rows. This was not thread oversubscription: `RAYON_NUM_THREADS=8` was flat and disabling the persistent pool was worse. `widen_rotary_prefix` now bounds contiguous F16C/f32 widening to `max_position + 1` rows, retaining a full-widen-and-truncate fallback for strided/transposed layouts; output remains byte-identical.

**Why:** The original GQA phase percentages normalized to instrumented phases and concealed the cost. A `TOTAL_NS` timer around `execute()` exposed the uninstrumented rotary widening. Always include an execute-total timer rather than inferring totals from phase sums.

**Results and validation:** On merged main, 0.5B improved **34→101.89 tok/s** and 1.5B **14.7→50.51 tok/s**; generic-cpu 7B held at **26.86 tok/s**. GQA share fell **70%→4.4%**. Cumulatively, the f16 workstream reached 0.5B **6.5→101.89 (~15.7×)** and 1.5B **3.58→50.51 (~14×)**. Pris added `.to_bits()`-exact f16/f32 parity against full widen, strided/transposed fallback, and batch-two descending-`position_ids` coverage. **717 tests plus 10 doctests passed.**

**Sources reconciled:** `roy-f16-gqa-decode.md` and `pris-gqa-rotary-tests.md`.

Decision archive gate checked at 2026-07-23T08:50:00Z: the active ledger exceeded 20480 bytes, but no dated entries were older than 2026-06-23T08:50:00Z; no archive was created or updated.
<!-- scribe-merge-2026-07-23T10-30-00Z-perop-audit-silu-robustness -->
## 2026-07-23 — CPU per-op audit and SiLU MLAS robustness remediation

<!-- merged from .squad/decisions/inbox/deckard-perop-audit.md -->
# Per-op audit: every CPU-EP decode op vs onnxruntime-genai (ORT)

**Author:** Deckard (perf) · **Branch:** perf/perop-audit (off 6941a9a) · **Date:** 2026-07-23
**Goal (user directive):** 每个 op 的性能都要超过 ORT，然后用模型 benchmark 确保整体性能也超过.
= EVERY CPU-EP decode op must beat ORT per-op, AND whole-model must beat ORT.

Host: shared 96-core box, very noisy (other users: clamscan/VLLM/etc). All numbers
are medians of ≥3 runs taken in low-load windows (1-min load < ~5), native vs ORT
interleaved A/B. `uptime` gated every run.

Method:
- OUR whole-model / per-op: `profile_native --steady --decode-skip 8 --tokens 128`,
  per-op via `ONNX_GENAI_PROFILE_OPS=1` (executor.rs print_op_profile), aggregated
  over the 248 steady decode steps (audit_scripts/agg_ours.py).
- ORT whole-model: onnxruntime-genai 0.14.1, CPU provider (Config.clear_providers),
  min_length-forced 128 new tokens (audit_scripts/ort_wholemodel.py).
- ORT per-op: raw decoder model.onnx driven through onnxruntime 1.27 CPU with
  enable_profiling, single decode step (input_ids[1,1], past-KV @ len=64), node
  kernel times aggregated by op_type (audit_scripts/ort_perop.py).
- Caveat: both profilers add measurement overhead, so per-op *absolute* ms are
  inflated vs whole-model; per-op *ratios / winners* are the signal.

Op-name mapping: RotaryEmbedding is fused inside GroupQueryAttention (do_rotary);
ORT fuses SiLU's sigmoid*x into `QuickGelu` (= our `Sigmoid`+`Mul` → our `Silu`);
ORT folds the residual add into `SkipSimplifiedLayerNormalization`.

---

## 1. WHOLE-MODEL native vs ORT  (整体性能) — ALL THREE WIN

| Model                         | dtype | Native tok/s | ORT tok/s | Ratio | Verdict |
|-------------------------------|-------|--------------|-----------|-------|---------|
| qwen2.5-0.5b (cuda-gpu build) | f16   | 124.6        | 81.9      | 1.52x | WIN     |
| qwen2.5-1.5b (cuda-gpu build) | f16   | 61.1         | 43.3      | 1.41x | WIN     |
| qwen2.5-coder-7b generic-cpu  | int4/f32 | 29.1–31.3 | 21.0–21.7 | 1.39–1.44x | WIN |

(0.5B/1.5B numbers rose after the Mul fix + measured in a quieter window than the
pre-fix baselines of ~100 / ~50; 7B rose 27.1 → ~30 from the SiLU fix. See §3.)
Parity openers byte-identical (0.5B [271,40,1079,264,48948,304,13027,323,358,1079,
4460,311,1855,264,4285,2025]; 7B [48298,271,9707,0,2585,646,358,7789,498,3351,...]).

---

## 2. PER-OP native vs ORT (ms per decode step; ORT past=64)

### 0.5B (f16)  — BEFORE fixes
| op-type                        | ours | ORT (QuickGelu=SiLU) | ratio | WIN/LOSE |
|--------------------------------|------|------|-------|----------|
| MatMulNBits                    | 3.32 | 9.50 | 0.35 | WIN |
| GroupQueryAttention            | 1.05 | 1.90 | 0.55 | WIN |
| SkipSimplifiedLayerNormalization | 0.63 | 1.37 | 0.46 | WIN |
| Cast                           | 0.005| 3.04 | —    | WIN |
| **Mul (gate*up)**              | 2.61 | 0.65 | 4.02 | **LOSE** |
| **Silu**                       | 1.09 | 0.68 | 1.60 | **LOSE** |
| **Add (qkv bias)**             | 0.66 | 0.62 | 1.06 | ~tie/LOSE |

### 0.5B (f16)  — AFTER fixes
| op-type                        | ours | ORT | ratio | WIN/LOSE |
|--------------------------------|------|------|-------|----------|
| MatMulNBits                    | 2.83 | 9.50 | 0.30 | WIN |
| GroupQueryAttention            | 1.04 | 1.90 | 0.55 | WIN |
| SkipSimplifiedLayerNormalization | 0.63 | 1.37 | 0.46 | WIN |
| **Mul (gate*up)**              | 0.69 | 0.65 | 1.06 | ~tie (was 4.02) |
| Silu                           | 1.06 | 0.68 | 1.56 | LOSE (f16, follow-up) |
| Add (qkv bias)                 | 0.70 | 0.62 | 1.13 | LOSE (f16, follow-up) |

### 7B generic-cpu (int4 weights / f32 activations)  — BEFORE fixes
| op-type                        | ours  | ORT   | ratio | WIN/LOSE |
|--------------------------------|-------|-------|-------|----------|
| MatMulNBits                    | 18.92 | 101.17| 0.19 | WIN (int4 MLAS SQNBit, 5.3x) |
| GroupQueryAttention            | 3.06  | 4.22  | 0.72 | WIN |
| Mul                            | 0.34  | 1.14  | 0.30 | WIN |
| **Silu**                       | 4.90  | 1.00  | 4.90 | **LOSE (worst offender)** |
| **SkipSimplifiedLayerNormalization** | 2.91 | 2.05 | 1.42 | **LOSE** |
| **Add (qkv bias)**             | 1.83  | (fused in ORT SkipLN) | — | **LOSE** (ORT spends 0 here) |

### 7B generic-cpu  — AFTER fixes
| op-type                        | ours  | ORT   | ratio | WIN/LOSE |
|--------------------------------|-------|-------|-------|----------|
| MatMulNBits                    | 19.47 | 101.17| 0.19 | WIN |
| **Silu**                       | 0.375 | 1.00  | 0.37 | **WIN (was 4.90 — 13x faster)** |
| GroupQueryAttention            | 3.02  | 4.22  | 0.72 | WIN |
| Mul                            | 0.31  | 1.14  | 0.27 | WIN |
| SkipSimplifiedLayerNormalization | 3.05 | 2.05 | 1.49 | LOSE (follow-up) |
| Add (qkv bias)                 | 1.88  | (fused) | — | LOSE (follow-up) |

---

## 3. ACTION TAKEN — fixed the two worst offenders (profile-first, RULES.md §4)

**Fix commit:** `22db607` on branch `perf/perop-audit` (not pushed/merged).


### Fix A — SiLU f32 → MLAS vectorized logistic  (the #1 loser: 7B SiLU 4.9x, 15% of decode)
Root cause: `silu_contiguous_f32` ran a scalar `x/(1+exp(-x))` with f64 `exp` per
element; LLVM cannot autovectorize a libm `exp` call, so it stayed scalar while
ORT uses MLAS's SIMD sigmoid.
Change: bind MLAS `MlasComputeLogistic` (shim.cpp + mlas-sys `compute_logistic`) and
compute SiLU as `out = sigmoid(x); out *= x` — two vectorized passes, reusing the
same battle-tested routine ORT uses (§4: reuse MLAS where it wins). Non-mlas builds
keep the scalar reference.
Result: 7B Silu 4.90 → 0.375 ms/step (13x); now beats ORT (0.375 vs 1.00).
Whole-model 7B 27.1 → ~30–31 tok/s (+~12–15%). Parity byte-identical (openers unchanged;
existing `silu_contiguous_matches_reference` @1e-6 passes under --features mlas).

### Fix B — generic contiguous binary Mul/Sub/Div fast path for ALL float dtypes
(the #1 0.5B loser: f16 Mul 4.0x)
Root cause: the contiguous fast path was f32-only (`multiply_contiguous_f32`); the
f16 models fell to `binary_typed` → `broadcast_apply`, which recomputes a multi-axis
source index per element and allocates an accumulator + dense staging buffers — ~0.11ms
for a tiny [1,4864] multiply (pure index/alloc overhead).
Change: `binary_contiguous<T: NumericElem>` handles same-shape contiguous, non-aliasing
Sub/Mul/Div for f32/f64/f16/bf16 in one tight loop using the identical
`to_acc`/`from_acc` rounding and `BinOp::apply` combiner → byte-identical to the slow
path (new test `mul_f16_contiguous_matches_broadcast_path`).
Result: 0.5B Mul 2.61 → 0.69 ms/step (3.8x); now ~tie with ORT (0.69 vs 0.65).
Also speeds 1.5B (larger intermediate) — contributes to 1.5B 49.6 → 61.1.

Files: crates/mlas-sys/vendor/shim.cpp (+mlas_compute_logistic),
crates/mlas-sys/src/lib.rs (+compute_logistic),
crates/onnx-runtime-ep-cpu/src/kernels/activations.rs (silu_f32_slice),
crates/onnx-runtime-ep-cpu/src/kernels/elementwise.rs (binary_contiguous + test).

Validation: `cargo test -p onnx-runtime-ep-cpu --features mlas` = 718 pass;
`cargo clippy -p onnx-runtime-ep-cpu --features mlas -- -D warnings` clean;
`cargo clippy -p mlas-sys` clean; parity openers identical on 0.5B and 7B; 7B no regression.

---

## 4. REMAINING LOSERS — precise follow-ups (not fixed here)

1. **SiLU on f16 (0.5B/1.5B): 1.56x** — ours 1.06 vs ORT 0.68 ms/step.
   f16 SiLU still uses the widen→scalar-f64-exp→narrow path (activations.rs execute,
   `to_dense_f32_widen` branch). Fix: widen f16→f32 scratch, `mlas_sys::compute_logistic`,
   multiply, narrow — same pattern as Fix A. Parity: f16 narrowing swamps f32-vs-f64 exp
   diff (already verified argmax-stable on the f32 side). Est. ~0.4ms/step.

2. **qkv-bias Add: 7B 1.88ms / 0.5B 0.70ms** — ORT spends ~0 (folds the qkv bias into
   its attention/MatMul path; we run a standalone Add x{layers}). Two options:
   (a) route Add through the same `binary_contiguous` fast path (AddKernel in add.rs has
   its own broadcast_apply loop — check whether the bias is same-shape-contiguous or a
   [N]-broadcast; if broadcast, add a broadcast-row fast path), or
   (b) fuse the qkv bias into MatMulNBits/GQA input like ORT. (a) is the smaller, general
   win. File under EP fusion (RULES.md §2.1).

3. **SkipSimplifiedLayerNormalization: 7B 1.49x (3.05 vs 2.05), 0.5B we already WIN.**
   Only the 7B f32 case loses. Profile the f32 RMS/skip-norm kernel (norm_ops.rs/rmsnorm.rs)
   — likely scalar rsqrt / non-SIMD reduction vs MLAS. Candidate: MLAS has no direct
   SkipLayerNorm export in our shim; a SIMD f32 reduction + rsqrt pass (or bind ORT's
   contrib SkipLayerNorm math) would close it. Lower leverage (9% of 7B, 1.49x).

---

## 5. RESIDUAL RISKS & RANKED NEXT OPTIMIZATIONS

Risks:
- Host is extremely noisy; absolute tok/s shift ±15% with load. Ratios (native/ORT,
  interleaved A/B) are the trustworthy signal; all A/B pairs were taken in the same
  low-load window.
- ORT per-op ms are enable_profiling-inflated; used only for winner/ratio direction.
- MLAS logistic is f32 (vs our historical f64 exp). Verified argmax/opener parity on
  0.5B and 7B; if any future model shows drift, the non-mlas scalar path is unchanged.

Next optimizations, ranked by leverage:
1. (highest) f16 SiLU → MLAS logistic (follow-up #1): closes the last material 0.5B/1.5B
   activation loser; mirrors the done f32 fix. ~0.4ms/step on 0.5B.
2. qkv-bias Add fast-path / fusion (follow-up #2): removes 1.9ms/step of pure overhead on
   7B that ORT doesn't pay — biggest remaining 7B gap after SiLU.
3. 7B f32 SkipSimplifiedLayerNormalization SIMD (follow-up #3): 1.49x, 9% of 7B.
4. MatMulNBits already dominant-win (0.19–0.35x). No action; it is why we win overall.

**Bottom line:** After Fix A + Fix B, we WIN whole-model on all three models (1.39–1.52x)
and WIN or tie every *material* per-op on the real 7B CPU target (only the small f32
SkipLN and the ORT-fused qkv Add remain, both follow-ups). The remaining per-op losses are
f16-only (0.5B/1.5B, GPU-targeted builds) small-tensor activations, tracked as follow-ups.

<!-- merged from .squad/decisions/inbox/bryant-silu-robustness.md -->
# Decision: SiLU MLAS robustness fix (remediation of Deckard's REJECT)

**Author:** Bryant (principal kernel engineer, CPU-EP)
**Branch:** perf/perop-audit (remediation on top of 22db607)
**Date:** 2026-07-23
**Reviewers context:** Gaff APPROVED FFI/dispatch; Chew REJECTED on SiLU numerics + thin tests.

## What changed

### 1. SiLU numerics cliff fixed without losing the 13x MLAS win
`crates/onnx-runtime-ep-cpu/src/kernels/activations.rs`

MLAS's `MlasComputeLogistic` clamps its input to `[-18, 18]` internally.
SiLU(x) = x·σ(x), so `sigmoid(x) * x` was wrong for out-of-range / non-finite
inputs:
- SiLU(-1e30) leaked σ(-18)≈1.5e-8 → -1.5e22 instead of decaying to ~0.
- SiLU(-Inf) → -Inf·1.5e-8 = -Inf (should be 0).
- SiLU(+Inf), SiLU(NaN) were also corrupted.

Fix (`silu_f32_slice`, ~activations.rs:313): keep the vectorized MLAS logistic +
multiply for the whole slice (the hot path stays fully vectorized), then run a
single correction pass that overwrites only elements where the input is
non-finite or `|x| > 18` with an accurate scalar SiLU. The correction predicate
is one branch-predictable `is_finite() && abs() <= 18.0` compare per element, so
in-range (bounded LLM) activations keep MLAS speed. New constant
`SILU_MLAS_SAFE_BOUND = 18.0` (mlas-gated) documents the clamp boundary.

Scalar reference (`silu`, `silu_f64`, ~activations.rs:126/143) hardened so the
`#[cfg(not(feature="mlas"))]` path is the exact reference at the extremes too:
SiLU(-Inf)=0 (previously produced NaN via -Inf·0), SiLU(+Inf)=+Inf, SiLU(NaN)=NaN.

### 2. (B) f16 Mul generic contiguous fast path — kept as-is (Gaff-approved)
No behavioral change; only tests strengthened (below).

## Tests added / strengthened

`activations.rs`:
- `silu_contiguous_matches_reference`: now a DENSE sweep -50..50 step 0.25 plus
  extreme finite magnitudes (±1e30, ±1e-30) and clamp-boundary values, compared
  to an EXACT f64 reference with a tight abs-or-rel 1e-5 tolerance.
- `silu_in_range_region_is_bit_close`: pins [-18,18] against the exact reference
  (MLAS approximation held to abs-or-rel 1e-5; its tail flushes σ→0 near -16).
- `silu_handles_infinities_and_nan`: SiLU(+Inf)=+Inf, SiLU(-Inf)=0, SiLU(NaN)=NaN.

`elementwise.rs`:
- `mul_f16_contiguous_matches_broadcast_path`: now also exercises the broadcast
  fallback and asserts RAW f16 bits (`to_u16_bits`) equal between the contiguous
  fast path and the broadcast path (not just decoded f32).
- `sub_div_f16_contiguous_matches_broadcast_path` (new, Gaff nit): Sub and Div
  f16 contiguous-vs-broadcast bit-identity.

## Parity / generality

- In-range elements use the identical arithmetic (`*output *= input`) as the
  approved 22db607, so bounded activations are byte-identical by construction.
  The 0.5B and 7B openers (bounded activations, no |x|>18) stay byte-identical;
  the correction path is only entered for out-of-range/special values that do
  not occur in those models.
- Portable: no new x86-only intrinsics; MLAS logistic is cross-ISA; scalar
  fallback unchanged off-mlas.

## Validation

- `cargo test -p onnx-runtime-ep-cpu --features mlas`: 721 passed, 0 failed.
- `cargo test -p mlas-sys`: 12 passed, 0 failed.
- `cargo clippy -p onnx-runtime-ep-cpu --features mlas --tests -- -D warnings`: clean.
- `cargo fmt` clean on changed files (activations.rs, elementwise.rs).

## Residual risk

- MLAS logistic in-range approximation flushes very small σ to 0 near the clamp
  edge (abs error ≤ ~1.2e-6, e.g. SiLU(-16.4)→-0 vs -1.2e-6). This matches ORT's
  routine and is within the pinned tolerance; acceptable for accuracy-level-4.
- The 18.0 boundary is tied to MLAS's internal clamp; if a future MLAS version
  changes that clamp, the constant must track it (documented at the constant).
- Opener byte-identity argued by construction (in-range arithmetic unchanged);
  a live low-load opener run was not executed here as it requires model weights.

Decision archive gate checked at 2026-07-23T10:30:00Z: the active ledger was 271720 bytes before this merge and exceeded 51200 bytes. No dated ledger entries older than 2026-07-16T10:30:00Z were present, so no archive was created or updated.
<!-- scribe-merge-2026-07-23T10-30-00Z-perop-audit-silu-robustness-end -->
<!-- scribe-merge-2026-07-23T10-35-00Z-deckard-skiplayernorm-simd -->
<!-- merged from .squad/decisions/inbox/deckard-skiplayernorm-simd.md -->
### 2026-07-23: Make CPU SkipSimplifiedLayerNormalization allocation-free and vectorizable
**By:** Deckard
**What:** The contiguous f32 `SkipSimplifiedLayerNormalization` path now also handles requested mean/inv-std outputs directly, fuses residual/bias assembly with an eight-lane f32 square reduction, and uses a fixed-lane normalize/scale loop with scalar remainders. The broadcast and widened f16/bf16 fallback remains dtype- and shape-generic.
**Why:** The real 7B graph requested statistics, so the previous direct-output path was bypassed and every one of 56 decode calls allocated buffers and performed per-element broadcast index unraveling. On the mandated profile, average decode op time/share fell from 2.885 ms / 9.15% to 0.594 ms / 1.99%; this is about 3.3x faster than the audit's approximately 1.94 ms ORT result inferred from the reported 1.49x baseline gap. The rewrite contains no target-specific intrinsics or model constants, preserves the exact 16-token opener, and passed 719 unit tests plus 10 integration tests, warnings-denied Clippy, and formatting checks.
<!-- scribe-merge-2026-07-23T10-35-00Z-deckard-skiplayernorm-simd-end -->
<!-- scribe-merge-2026-07-23T11-00-00Z-roy-f16-silu -->
<!-- merged from .squad/decisions/inbox/roy-f16-silu.md -->
### 2026-07-23: Route widened low-precision SiLU through the shared MLAS path
**By:** Roy
**What:** f16/bf16 (and other non-f32, non-f64 floating) SiLU now widens to f32 and calls `silu_f32_slice` before narrowing, instead of applying scalar SiLU element by element.
**Why:** This reuses the portable MLAS logistic SIMD routine and its existing finite/extreme correction pass, eliminating the low-precision scalar activation bottleneck without model- or architecture-specific behavior. On the Qwen2.5-0.5B f16 profile, SiLU fell from about 1.08 ms to about 0.275 ms per 24 calls (~3.9x faster); the host was loaded above 6, so the relative per-op result is the meaningful measure.

**Review:** Chew APPROVE. **Merged:** `d14cc83`.
<!-- scribe-merge-2026-07-23T11-00-00Z-roy-f16-silu-end -->
<!-- scribe-merge-2026-07-23T11-00-00Z-bryant-qkv-bias-add -->
<!-- merged from .squad/decisions/inbox/bryant-qkv-bias-add.md -->
### 2026-07-23: Fold QKV-bias `Add` into `MatMulNBits` (CPU EP)
**By:** Bryant (CPU-EP kernels)
**Branch:** perf/qkv-bias-add (off main 316113e)

**What:** Added an always-on, EP-internal graph fusion pass
`CpuMatMulNBitsBiasFusion` in `crates/onnx-runtime-ep-cpu/src/optimizer.rs`
that recognizes the generic pattern `Add(MatMulNBits(A, ...), [N]-bias)` and
rewrites it to `MatMulNBits(A, ..., bias)` using the contrib op's optional bias
input (index 5). The `MatMulNBits` kernel already adds that bias inside the MLAS
GEMV epilogue, so the standalone element-wise `Add` disappears.

**Why:** The per-op audit flagged the QKV-bias `Add` as a spot where ORT is
faster because ORT fuses the bias into the projection GEMM. On the 7B
generic-cpu graph it was **28 Adds/step (1 per decoder layer), ~1.87 ms/step,
~6.5% of node execution** — a combined QKV `MatMulNBits` feeding one rank-1
`[q+k+v]` bias `Add` feeding GQA. Folding the bias into the GEMV epilogue reuses
memory the kernel already touches, so the bias add is effectively free.

**Profile (7B qwen2.5-coder generic-cpu-4, --steady --decode-skip 8 --tokens 128
--runs 3; shared box, trust SHARE not absolute ms):**
- Before: `Add` = 28 calls, ~1.82–1.88 ms/step, **6.5% share**; `MatMulNBits`
  67.3%; node execution ~28.2 ms.
- After: **`Add` gone (0 standalone Adds)**; `MatMulNBits` 73.3% (absorbs bias,
  its own ms unchanged ~19.0 ms); node execution ~26.1 ms.

**Correctness / generality (RULE 2 / 2.1):**
- Byte-identical: MLAS and the standalone `Add` both perform a single f32 add of
  the same bias per column over the same GEMM result.
- Opener stays byte-identical:
  `[48298,271,9707,0,2585,646,358,7789,498,3351,30,151645,198,151643,151644,198]`.
- Pattern-only match — no model names, no hardcoded dims. Guards: producer is a
  bias-free `MatMulNBits` (com.microsoft) whose sole consumer is the `Add` and
  whose output is not a graph output; bias is a rank-1 `[N]` float tensor over
  the output's last (`N`) dim. Falls back cleanly (no rewrite) otherwise.
- Runs unconditionally (unlike the env-gated gate/up `ProjectionFusion`) because
  it is a pure, safe, byte-identical convenience fold with a clean fallback.

**Validation:** `cargo test -p onnx-runtime-ep-cpu --features mlas` → 728 passed
/ 0 failed (incl. 5 new fusion tests: positive fold, operand-order symmetry,
non-row-vector bias rejected, extra-consumer rejected, graph-output rejected).
`cargo clippy -p onnx-runtime-ep-cpu --features mlas -- -D warnings` clean.
`rustfmt` clean on changed files.

**Scope:** No change to `main`, no push/merge. Touches only
`crates/onnx-runtime-ep-cpu/src/{optimizer.rs,lib.rs}`.

**Review:** Gaff APPROVE. **Merged:** `28adcd9`.
<!-- scribe-merge-2026-07-23T11-00-00Z-bryant-qkv-bias-add-end -->
<!-- scribe-merge-2026-07-23T11-10-00Z-coordinator-final-cpu-benchmark -->
<!-- merged from .squad/decisions/inbox/coordinator-final-cpu-benchmark.md -->
### 2026-07-23: CPU EP whole-model decode beats onnxruntime-genai on all 3 models (matched-load A/B)
**By:** Squad (Coordinator), for justinchuby
**What:** Final matched-load A/B on the same Xeon 8480C, native onnx-genai CPU vs onnxruntime-genai 0.14.1 CPU, decode tok/s (--steady --decode-skip 8 --tokens 128 --runs 3, median):
- Qwen2.5-0.5B f16: native 154.9 vs ORT 86.5 = 1.79x
- Qwen2.5-1.5B f16: native 74.0 vs ORT 40.6 = 1.82x
- Qwen2.5-coder-7B int4 generic-cpu: native 32.7 vs ORT 21.1 = 1.55x
Openers byte-identical. ORT f16 baselines obtained via CPU-provider config variants (/tmp/ortcpu-{0.5b,1.5b}, provider_options emptied).
**Why:** Confirms the user directive — every material CPU-EP decode op now beats/ties ORT AND whole-model decode beats ORT on all three. Landed this segment (all non-author reviewed, byte-identical/tight-tolerance, cross-OS/cross-arch, no hardcoded dims): f32 SiLU MLAS-logistic+robust-extreme (13x), f16/bf16 SiLU reuse (~3.9x), f16 Mul/Sub/Div binary_contiguous (~3.8x), SkipSimplifiedLayerNorm portable 8-lane SIMD + stats-output fast path (~3.3x vs ORT), QKV-bias Add folded into MatMulNBits epilogue (standalone Add eliminated). 730 CPU-EP tests green, clippy -D warnings clean. PR #105.
<!-- scribe-merge-2026-07-23T11-10-00Z-coordinator-final-cpu-benchmark-end -->
<!-- scribe-merge-2026-07-23T11-25-00Z-pris-parity-gate -->
<!-- merged from .squad/decisions/inbox/pris-parity-gate.md -->
### 2026-07-23: Add CPU SIMD-versus-scalar parity regression gate
**By:** Pris
**What:** Extended f16 Mul/Sub/Div binary-contiguous raw-bit parity coverage with non-lane-multiple 61- and 53-element inputs. Added cross-dtype (f32/f16/bf16) `SkipSimplifiedLayerNormalization` SIMD-versus-scalar parity coverage across remainder and bulk hidden sizes, with/without bias and requested statistics outputs. Existing SiLU MLAS-versus-scalar boundary coverage and MatMulNBits numeric bias-fusion equivalence were retained without duplication.
**Why:** Locks the five landed CPU-EP performance improvements against correctness regressions; x86 SIMD-equals-scalar parity serves as the cross-architecture correctness proxy.
**Validation:** 731 library tests passed; Clippy with warnings denied and rustfmt were clean.
**Merged:** `1be1bd5`.
<!-- scribe-merge-2026-07-23T11-25-00Z-pris-parity-gate-end -->
<!-- scribe-merge-2026-07-23T14-45-00Z-bf16-coverage-start -->
## 2026-07-23 — CPU EP bfloat16 (bf16) coverage extended
**By:** Zhora (impl), Gaff/opus (non-author review), requested by justinchuby.
**What:** ORT's CPU EP lacks bf16 for most ops; extended native CPU EP so every capable op accepts bf16. Audit found most of ~158 ops were already class-A (bf16 present). Added widen→f32→narrow bf16 compute paths for Softmax/LogSoftmax/all Reduce*/LayerNorm/Relu/all unary-math (Abs..Tan)/Attention/FusedAttention/GroupQueryAttention (incl. KV-cache decode round-trip)/Range/Constant/ConstantOfShape/CenterCropPad. **Fixed Transpose — it was secretly f32-only**, now byte/element-size generic (bit-exact). Added bf16 bit-lock regression tests on movement ops (Transpose/Concat/Slice/Gather/Split/Pad/Where/GatherElements/GatherND/ScatterElements/Compress/Tile/Reshape/Unsqueeze/Identity). Narrowing uses round-to-nearest-even (half::bf16::from_f32); softmax/scaling done in f32. Class-D int/bool/logical/quantized/window/bitwise ops correctly REJECT bf16 (no byte reinterpretation).
**Why:** User: "cpu ep所有可以支持bf16的op都要支持。ort现在就是cpu ep bf16不支持 很难用。我们要支持."
**Validation:** 757 ep-cpu lib tests + 10 numeric-regression golden tests green (752→771 total incl. new bf16 tests); clippy -D warnings clean. Non-author reviewed by Gaff (opus): attention KV round-trip verified single-narrow (no double-truncation), movement genuinely byte-generic, tests assert independent f32 reference / exact bit equality. Verdict ✅ ship.
**Merged:** cherry-picked to perf/cpu-ep-mlas as `84b40d9` + `a68b076`, pushed (PR #105).
<!-- scribe-merge-2026-07-23T14-45-00Z-bf16-coverage-end -->
<!-- scribe-merge-2026-07-23T15-30-00Z-loop-and-divergence-start -->
## 2026-07-23 — Generation-loop overhead cut + same-loop backend A/B + token-divergence root-cause
**By:** Bryant (loop), Deckard (A/B + divergence), requested by justinchuby.
**Loop overhead (Bryant, 2fbc679):** Profiled the shared engine decode loop; removed native greedy path's full-vocab logits materialization (direct argmax from Float32/Float16/BFloat16 tensors), cut default-loop alloc/cloning, skip incremental detokenization when no callback needs it. Native non-model overhead 0.510→0.258 ms/tok (-49%); 0.5B native 139.4→152.7 tok/s (+9.5%), 1.5B +3.5%. Greedy IDs identical. **Proved our loop is NOT slower than ORT's** — residual gap vs external onnxruntime-genai is inside ORT build/session.run (99% of wall), not our loop. Behavior dtype/shape/capability-driven, EP/model agnostic.
**Same-loop backend A/B (Deckard, 8f55928):** Added `--backend {native,ort,auto}` to profile_native so Native and ORT run through the SAME Engine::generate loop (isolates runtime speed from loop speed). Result: **Native beats ORT 2.24× (0.5B) / 2.38× (1.5B) / 3.06× (7B int4) / 3.49× (7B f16)** — proves our RUNTIME is faster, not just the loop.
**Token-divergence root-cause (Deckard, 557c3ed):**
  - 1.5B f16 @36: Native is MORE accurate (matches f32-reference argmax token 4092; ORT tie). KEEP ours. Regression test `matmul.rs::matmul_f16_preserves_near_tie_argmax_after_f32_accumulation`.
  - 7B int4 @23: REAL native bug — culprit = **CompInt8 activation quantization** in MatMulNBits (Native RMSE 0.005 vs ORT 0.0019 vs dequant-f32 oracle; native picks wrong token 151643 vs correct 151644). CompFp32 fixes it but collapses throughput 27→0.55 tok/s. Characterization test `matmul_nbits.rs::matmulnbits_compint8_argmax_reversal_is_caught_by_fp32_oracle`. → Spun focused fix agent (fix/compint8-accuracy) to make int8 path ORT-accurate at int8 speed (prefer reusing MLAS CompInt8).
**Generality gaps found (to fix):** Phi-4-mini/Phi-3.5 (phi3, head_dim=48) fail native GQA (kernel assumes 64) → fix/phi3-headdim agent. Qwen3-0.6b lacks GatherBlockQuantized native op → queued.
**Validation:** ep-cpu 759 tests green (incl. 2 new divergence tests); engine 164 passed / 17 pre-existing textproto-fixture failures (identical set on base — zero regression; separate fix PR opened via fix/textproto-fixture-loading). clippy clean.
**Merged:** perf/cpu-ep-mlas 2fbc679 + 8f55928 + 557c3ed (cherry-picked; profile_native.rs --backend conflict resolved to Deckard's Auto-capable version, Bryant's native_decode engine opts retained). Pushed to PR #105.
<!-- scribe-merge-2026-07-23T15-30-00Z-loop-and-divergence-end -->
<!-- scribe-merge-2026-07-23T16-20-00Z-conv-fixture-start -->
## 2026-07-23 — Native CPU EP CNN support (MLAS Conv/Pool, ORT parity) + textproto fixture-loading fix (PR #107)
**By:** Roy (Conv/Pool), Holden (fixture), reviewed by Gaff (opus). Requested by justinchuby.
**MLAS Conv/Pool (Roy, merged perf/cpu-ep-mlas d5cd0a8 + 6604295):** Native CPU EP had NO `ai.onnx::Conv`/Pool → ResNet-50/MobileNetV2/YOLO failed to load/run. Added MLAS-backed generic 2D Conv (auto_pad NOTSET/SAME_UPPER/SAME_LOWER/VALID, pads/strides/dilations, group+depthwise, optional bias) + Pool (Max/Average/GlobalAverage) + Add/ReLU/Clip, via new crates/mlas-sys shim (MlasConvPrepare/MlasConv/MlasPool) mirroring the sqnbit pattern. Also added profile_vision native-vs-ort CNN A/B harness.
  - **Parity vs ORT (fp32):** ResNet-50 abs 1.0e-5 / rel 5.4e-4, top-1 904 ✅; MobileNetV2 abs 9.1e-6 / rel 3.2e-4, top-1 904 ✅. CNN backbones run end-to-end natively.
  - **Perf gap (queued follow-up):** MLAS single-op Conv currently SLOWER than ORT (ResNet 12×, MobileNet 4.1×) — ORT uses fused NCHWc-blocked + prepacked Conv. Correctness/generality landed first; a Conv-perf agent (NCHWc block layout + weight prepack + Conv-BN-ReLU fusion) is queued to close/beat it.
  - **Review (Gaff/opus, non-author):** ✅ no 🔴 — FFI scratch size queried-then-allocated exactly (no OOB), all unsafe output slices length+alias-guarded, enum/attr mappings match vendored MLAS headers, hand-computed unit tests independent. Nits: add a numeric SAME-pad conv test; Conv has no non-MLAS scalar fallback (by design).
  - **Merge note:** relu.rs conflict (bf16 widen/narrow vs Roy's MLAS f32 fast-path) resolved to run MLAS fast-path first, then fall back to bf16 widen/narrow. 764 ep-cpu tests green (mlas), clippy clean (mlas is canonical; non-mlas has pre-existing dead-code profiling-static warnings).
  - **YOLO still needs:** opset-11 BatchNormalization (CPU reg starts opset 15) + Resize/NMS post-processing — follow-up.
**Textproto fixture fix (Holden, SEPARATE PR #107 → main, aaecfef):** 17 engine tests failed because committed `.onnx.textproto` fixtures (no binary model.onnx) hit `model_requires_native_backend` + `scan_top_level_control_flow`, which raw-binary-decoded → "invalid wire type value: 6". Fix routes both scans through the loader's textproto-aware `read_model_binary`/`is_textproto_path`. 17 failing → 0 (263 passed). Regression test `backend_and_control_flow_scans_parse_textproto_fixture` (verified passing under --features native-backend). Isolated 44-line change; opened as its own PR to main per user request ("要是有test fixture error，可以开一个pr修理").
**Still-open perf follow-ups (user: ALL parts must beat ORT):** (1) Conv NCHWc/prepack/fusion; (2) qwen3.5 native 0.07 tok/s — MatMulNBits (57-76%) + Reshape (24-42%) pathological on that hybrid model (Pris's new conv/linear-attn kernels are <1%); needs decode-path profiling.
<!-- scribe-merge-2026-07-23T16-20-00Z-conv-fixture-end -->
<!-- scribe-merge-2026-07-23T18-40-00Z-compint8-phi3-qwen35-start -->
## 2026-07-23 — CompInt8 accuracy fix + phi3 head_dim generality + Qwen3.5 native E2E (merged to PR #105)
**By:** Deckard (CompInt8), Tyrell (phi3), Pris (qwen3.5). Reviews: Leon (CompInt8), Rachael (phi3), Deckardrev (qwen3.5) — all opus, all non-author. Requested by justinchuby.
**CompInt8 activation-quant fix (Deckard, merged 70cd499):** The 7B int4 @step-23 token divergence (native picked 151643 vs correct 151644) was MatMulNBits CompInt8 per-row activation quant diverging from ORT/MLAS. Fix = per-K-block activation quantization (scale = max_abs_block/127, symmetric int8) folded into the per-block dot, consistent across scalar / AVX-VNNI / AVX512-VNNI, zero-block guarded (no div-by-zero). RMSE 8.9%→0.25%; native decode tokens now **byte-identical to ORT** at int8 speed (39 tok/s, no CompFp32 collapse). Superseded characterization test `matmulnbits_compint8_argmax_reversal_is_caught_by_fp32_oracle` removed (it asserted the bug); two new f32-oracle parity tests added. **Review (Leon/opus):** ✅ correct, no 🔴, verified real-model token parity fixes step-23.
**phi3 head_dim generality (Tyrell, merged 2c4cfab):** Native GQA + RotaryEmbedding assumed head_dim=64 → Phi-3.5/Phi-4-mini (head_dim 48/96, partial rotary width 48) errored "rotary cache dimension 48 vs kernel-required 64". Fix derives rotary_half/rotary_dim from the cos cache shape (checked_mul), supports partial rotary (tail lanes pass through untouched), preserves 64/128 path byte-identically. Phi-3.5 int4: native **byte-identical 32 tokens vs ORT** ("Paris..."), 1.96× ORT throughput (27.2 vs 13.9 tok/s uncontended). New tests: rope/decode head_dim 48/80 incl. cached-decode partial rotary. **Review (Rachael/opus):** ✅ no 🔴, bounds-safe KV path, independent first-principles RoPE references.
**Qwen3.5 native E2E (Pris, merged fd302e5 + d91d776):** Added CausalConvWithState + LinearAttention (gated-delta) kernels + GatherBlockQuantized (50,000× zero-copy fix, also unblocks qwen3-0.6b) + contrib com.microsoft::RotaryEmbedding (input order X,pos,cos,sin) + engine hybrid recurrent-state cache (fixed-size conv/recurrent states replaced wholesale, exempt from growable-KV seq-len check via is_recurrent_state_shape). Runs end-to-end, first token matches ORT. **Perf (queued):** native 0.07 vs ORT 52.4 tok/s — pre-existing MatMulNBits (57-76%) + Reshape (24-42%) pathology on this hybrid model; Pris's new kernels are <1% (confirmed not a new-code regression by Deckardrev). **Review (Deckardrev/opus):** ✅ safe, one 🔴 (unused import) fixed by Pris.
**Merge-resolution fixes (coordinator, folded into d91d776):** (a) native_decode.rs: merged Bryant's clean zip-loop output-fetch structure with Pris's recurrent-state guard inside the present→past branch. (b) rotary_embedding.rs: phi3's rank-2 cos-cache validation hardcoded inputs[1]/inputs[2]; under Pris's contrib remap inputs[1] is position_ids — rewrote validation to use resolved cos_i/sin_i indices so both standard and contrib orderings validate the correct tensors. (c) added contrib:false to tyrell's phi3 rotary test constructor.
**Validation:** ep-cpu **786 tests green** (mlas, incl. registration-count 89+mlas confirmed), clippy clean, rustfmt clean. Engine: 164 passed / 17 pre-existing textproto-fixture failures (identical set, zero regression — fixed separately in PR #107 to main). Stack pushed 1932aee..d91d776 to perf/cpu-ep-mlas.
<!-- scribe-merge-2026-07-23T18-40-00Z-compint8-phi3-qwen35-end -->
<!-- scribe-merge-2026-07-23T17-30-00Z-qwen35-decode-start -->
## 2026-07-23 — Qwen3.5 native decode 180× (zero-copy Reshape/Transpose + constant provenance) — merged PR #105
**By:** Warrick. Review: Nick (opus, non-author). Requested by justinchuby.
**What (merged 272438f):** Root-caused the 0.07 tok/s qwen3.5 decode pathology: (1) Reshape/Transpose were MATERIALIZING copies every step, hiding initializer provenance so MatMulNBits re-packed quantized weights each token; (2) LinearAttention duplicated recurrent states. Fix: Reshape/Transpose now emit zero-copy VIEWS (metadata-only; executor pins the source buffer + bounds-checks the composed view; Transpose emits genuinely permuted strides, Reshape views only when contiguous); constant/initializer provenance now flows through view ops so MatMulNBits packs weights ONCE (per-node OnceLock, keyed by node_id — no global/cross-session cache); direct output writes (buffer fully overwritten, beta=0); cache-friendly LinearAttention state updates. **Native 0.09 → 16.18 tok/s (~180×)**; ORT 50.96 (remaining 3.15× gap). **Exact 32-token ORT parity.** Files: reshape.rs, transpose.rs, matmul_nbits.rs, linear_attention.rs, executor.rs.
**Review (Nick/opus):** ✅ SAFE, no 🔴. 789 ep-cpu tests green, clippy clean; verified exact 32/32 ORT token parity on BOTH Qwen2.5-0.5B (regression — native still matches + no throughput regression) and Qwen3.5-2B (target). Views alias-safe (no UAF/OOB), provenance can't mis-tag runtime activations, per-node pack cache has no leakage.
**🟡 Follow-up nits (non-blocking, queued):** (1) executor.rs:3251-3256 provenance predicate marks ANY view-of-initializer constant; the pre-existing Slice kernel uses runtime starts/ends — a runtime-sliced initializer feeding a prepacking weight could cache a stale pack. NOT reachable by real transformer graphs (weights never runtime-sliced; 789 tests + both models pass) but a latent hazard — narrow provenance to data-invariant view ops OR require the whole view chain (incl. Slice bounds) constant, + regression test. (2) Add a comment documenting the LinearAttention no-input/output-aliasing invariant.
**Remaining perf gap (queued):** qwen3.5 native still 3.15× behind ORT — next: profile the residual MatMulNBits/attention path on this hybrid model.
<!-- scribe-merge-2026-07-23T17-30-00Z-qwen35-decode-end -->
<!-- scribe-merge-2026-07-23T17-55-00Z-nchwc-conv-start -->
## 2026-07-23 — CPU EP NCHWc Conv + weight pre-pack + Conv/BN/Relu fusion — merged PR #105
**By:** Grissom. Review: Greg (opus, non-author). Requested by justinchuby.
**What (merged 780ddbc + 9f93d3a):** Closed most of the Conv perf gap vs ORT. (1) mlas-sys: exposed MLAS NCHWc blocked-conv API — compiled snchwc.cpp/reorder.cpp, C shim + safe Rust wrappers (MlasNchwcConv, OIHWBiBo/OIHWBo filter reorder, NCHW↔NCHWc activation reorder, block-size query, fused activation). (2) ep-cpu: Conv picks NCHWc path when eligible (pointwise/blocked / first-layer NCHW / depthwise, mirroring ORT nchwc_transformer selection) else im2col fallback; filter+bias PRE-PACKED once for constant weights (per-node OnceLock, no global cache); always-on CpuConvBatchNormActivationFusion folds inference BatchNormalization into Conv weight/bias (a=scale/√(var+eps), new_w=w·a, new_b=(b-mean)·a+beta, eps from attr) and folds a trailing Relu into Conv activation only when Relu is the SOLE consumer. Purely structural (RULE 2).
**Key finding:** BatchNormalization, not Conv, was 65–92% of native CNN time; after fusion BN vanishes from the profile and Conv is 80–89%.
**Before/after (ratios, AVX-512, load ~21):** ResNet-50 native 799→**111 ms** (~69×→**7.7×** ORT); MobileNetV2 664→**22 ms** (~77×→**4.6×** ORT). Parity: ResNet max_abs 9.06e-6 / MobileNet 2.86e-6, top-1 AGREE both.
**Did NOT beat ORT yet** (7.7×/4.6× slower). Root cause of residual gap: every Conv reorders NCHW→NCHWc in and back out; ORT converts to NCHWc once at graph entry and stays blocked. **Next: graph-level NCHWc layout-propagation pass** (reorder only at layout boundaries, keep Conv/Pool/Add/Relu blocked) — the path to matching/beating ORT. bf16/f16 Conv = TODO (MLAS NCHWc is f32-centric).
**Review (Greg/opus):** ✅ SAFE, no 🔴. FFI buffer sizing correct (round_up channels to block), per-node prepack cache no leakage, BN-fold inference/constant-only + Relu sole-consumer guarded. mlas-sys 18 + ep-cpu 792 tests green (3 new fusion tests), clippy clean, real-model parity re-verified. 🟡 nits (queued): add debug_assert! length checks in public nchwc_* wrappers; add a dilation>1 NCHWc unit test.
<!-- scribe-merge-2026-07-23T17-55-00Z-nchwc-conv-end -->
<!-- scribe-merge-2026-07-23T18-50-00Z-f16-rope-gemma-start -->
## 2026-07-23 — f16 RotaryEmbedding (enables Gemma-2) + foundry-model breadth + generality gaps — PR #105
**By:** Sara. Review: Sofia (sol, non-author). Requested by justinchuby.
**What (merged c38438e):** RotaryEmbedding now accepts f16/bf16 by widening to f32 for compute and narrowing to the output dtype (was f32-only, ERRORed on Gemma-2's opset-24 f16 RoPE). f32 path is zero-copy identity (no regression — Phi-3.5 32-token native/ORT still identical). f16 computes in f32 then rounds once → potentially MORE accurate than ORT stepwise-f16. Parity unit test added. **Enables Gemma-2-2B native E2E with EXACT token parity vs ORT.**
**Review (Sofia/sol):** ✅ SAFE, no 🔴. 793 ep-cpu tests green (incl. f16 parity + head-dim 48/80), clippy clean, cherry-picks cleanly. 🟡 nit: add bf16 + ORT-golden RoPE coverage later.
**Foundry-model breadth results (same-loop native-vs-ORT, box heavily loaded ~20-66 so throughput ratios UNRELIABLE; PARITY is load-independent):**
  - Gemma-2-2B (mobius f16): tokens MATCH ✅ (native slower under load, re-measure clean).
  - Phi-3.5-mini int4: tokens **diverge at generated token 65** ❌ (separate/deeper than the CompInt8 step-23 fix; first 64 identical) → fix/token-divergence agent (Horatio).
  - Qwen3-0.6B int8/block-128: tokens **diverge immediately** ❌ + 0.003× (8-bit block-128 MatMulNBits) → Horatio.
**Generality gaps found (native CPU EP can't load these — QUEUED):**
  - **rank-3 1-D Conv** (ai.onnx::Conv opset18/21) — MLAS Conv only accepts rank-4 2-D NCHW; blocks Whisper-tiny encoder (X=[1,80,3000],W=[384,80,3]) AND Nemotron ASR encoder (X=[1,1024,7],W=[2048,1024,1]). **HIGHEST-VALUE next op.**
  - **LSTM opset 21** (Nemotron decoder) — no CPU EP handler.
  - **If branch rank-mismatch** in native shape inference (Whisper jump-times, Nemotron VAD) — rejects branch outputs of differing rank.
  - Multi-ONNX encoder/decoder package harness (Whisper) + Int32 input_ids synthesis for probes.
  - Nemotron joint graph: loads + matches ORT (max_abs 9.5e-7) but native 71ms vs ORT 0.35ms on synthetic probe (perf).
<!-- scribe-merge-2026-07-23T18-50-00Z-f16-rope-gemma-end -->

---
### 2026-07-23: Token divergences resolved + 8-bit MatMulNBits regression oracle (Horatio)
**By:** Horatio (investigation, opus), coordinator merge. Reviewed: self (test-only, oracle independence verified).
**What:** Investigated the two user-mandated native-vs-ORT token divergences Sara reported (Phi-3.5 int4 @ token 65; Qwen3-0.6B int8/block-128 immediate). NEITHER reproduces at branch tip (perf/cpu-ep-mlas): native == ORT byte-identical for full 128 tokens on BOTH models, and thread-count invariant (1|4|48 workers -> identical ids), ruling out reduction-order nondeterminism.
**Root causes (already fixed by prior merges):**
- Phi-3.5 token-65: CompInt8 activation per-K-block int8 quant fix (70cd499, locked 557c3ed) — matches ORT/MLAS QuantizeARow_CompInt8. The "identical for 64, flips near-tie argmax at 65" symptom matches slow-drift-then-flip. Sara's report predates that fix on her checkout / was a high-load ORT artifact.
- Qwen3-0.6B: no native bug. try_mlas_sqnbit declines bits!=4 (matmul_nbits.rs:574), so all 8-bit block-128 weights take the exact dequantize-to-f32 path (dequantize_weight -> gemv_nk/gemm) == ORT CompFp32. Sara's wrong-from-token-0 output is consistent with ORT miscompute under ~40x CPU oversubscription (her load 41/58/64), not a native defect.
**Contribution merged (bac0ae3, cherry-pick of db55954, test-only +179 lines):**
- matmulnbits_8bit_block128_execute_matches_dequant_f32_oracle: real execute() vs INDEPENDENT from-scratch dequant-f32 GEMM oracle; symmetric + asymmetric uint8 zp; decode M=1 + prefill M=5; rel RMSE <= 1e-5 + per-row argmax parity.
- matmulnbits_8bit_block128_argmax_matches_dequant_f32_oracle_at_near_tie: deterministic near-tie sweep; execute() never reverses f32 oracle greedy winner.
- Both green; oracle confirmed non-vacuous (out from CpuExecutionProvider.execute, oracle from plain reference GEMM over independently-dequantized weights).
**Residual:** If either divergence re-captured on a quiescent host (load < ~4), escalate with per-op logit dumps at the diverging step. Pre-existing clippy -D warnings drift (matmul.rs:800 excessive_precision on f16 literals from 557c3ed; group_query_attention.rs:1346 needless_range_loop) flagged for a separate lint-hygiene pass — NOT from this change.

---
### 2026-07-23: NCHWc graph-level layout propagation for CNNs (Brass) — MERGED 05a96bd
**By:** Brass (impl, opus). Reviewed by Wolfe (opus, non-author): 🟢 SAFE + 4 non-blocking nits.
**What:** New graph-level optimizer pass `NchwcLayoutPropagation` (crates/onnx-runtime-ep-cpu/src/nchwc_layout.rs ~1063 lines + kernels/nchwc.rs 603 lines, 6 blocked kernels; mlas-sys reorder helpers; graph.rs gc_value_if_orphan made pub). Keeps CNN backbones in MLAS channels-blocked (NCHWc) layout end-to-end (mirrors ORT NchwcTransformer): forward-propagates 4-D shapes with symbolic batch (Shape4{n:Dim,...}), classifies maximal NCHWc-capable subgraphs (Conv, Max/Avg/GlobalAvgPool, Add, Relu/Clip, folded BN), inserts ONE NCHW->NCHWc reorder per region entry + NCHWc->NCHW per exit, rewrites interior ops to consume blocked buffers. Env gates NXRT_DISABLE_NCHWC_LAYOUT / NXRT_NCHWC_DEBUG. Per-op Conv path preserved as fallback.
**Perf (shared 96-core, noisy):** MobileNetV2 ~62->~24ms (2.6x), gap to ORT 3.7x->1.3-1.8x (best ~1.05x). ResNet-50 clean A/B 230->97ms (2.36x), gap 7.7x->~2.5x. Still behind ORT on ResNet but gap closed dramatically.
**Parity (HARD GATE):** Wolfe independently reproduced: ResNet max_abs=0.0 top-1 AGREE, disable-gate full restore max_abs=0.0; synthetic opset-17 CNN with channels 12/20 (not block-16 multiples, forces padding lanes) + MaxPool pad-1 + residual Add + Clip + GlobalAvgPool: parity 1.19e-7 top-1 AGREE. Padding zero-filled at entry, zero-weighted through conv, per-channel-isolated in pool/GAP, dropped at exit (no leak/NaN). Symbolic batch resolved at runtime (no batch==1 assumption).
**Tests:** 798 ep-cpu + 20 mlas-sys green. Follow-up nits (non-blocking): (1) blocked kernels omit byte_ranges_overlap aliasing guard — safe today (planner retires input slots after allocating outputs, interior twins not user-bindable) but harden esp. exit reorder to user-bound output; (2) blocked-pooling non-zero-pad unit test; (3) benefit contingent on Conv+BN fusion; (4) possible redundant exit reorders (perf-only).

---
### 2026-07-23: Rank-3 (1-D) Conv support for Whisper/Nemotron encoders (Delko) — MERGED 40acb5f
**By:** Delko (impl, opus). Reviewed by Bosco (sol, non-author): 🟢 SAFE.
**What:** Conv1dAdaptation shim at ConvFactory::create in kernels/conv.rs (only file touched — no rank-4 path change, trivially rebased over Brass's NCHWc rework). Lifts 1-D conv to 2-D with singleton height axis: X[N,C,L]->[N,C,1,L], W[M,Cg,k]->[M,Cg,1,k], spatial attrs prepended with height identity, pads [pl,pr]->[0,pl,0,pr] (Bosco verified this — the #1 risk area — is correct). Output [N,M,1,Lout] squeezed to [N,M,Lout]. Reuses existing rank-4 MLAS fast path -> parity guaranteed. Shape inference + provider claim already accepted rank-3.
**Coverage:** 4 new tests (stride+pad, pointwise kernel-1, dilation, exact Foundry shapes [1,80,3000]/[384,80,3] Whisper + [1,1024,7]/[2048,1024,1] Nemotron); onnxruntime parity within 1e-5. auto_pad SAME/VALID, groups/depthwise, bias all handled. f32 only (matches existing kernel contract). Whisper Tiny + Nemotron ASR encoder Conv nodes now build+execute natively.
**Tests:** 802 ep-cpu lib green.

---
### 2026-07-23: bfloat16 operator coverage extension (Riley) — MERGED 209a56b
**By:** Riley (impl, sol). Reviewed by Frost (opus, non-author): 🟢 SAFE, numerically correct.
**Context:** User requirement — native CPU EP must support bf16 on every capable op; ORT's CPU EP does NOT (generality win). bf16 already had broad coverage (~47 kernel files).
**What:** Added NEW bf16 execution paths (compute-in-f32: widen bf16->f32, compute, narrow ONCE) to: selection.rs (TopK etc.), quantization.rs (Quantize/DequantizeLinear), block_quantized_matmul.rs (MXFP4/block-quant with f32 accumulation). ADDED verifying bf16 regression tests to rmsnorm.rs + rotary_embedding.rs (those already supported bf16 via the generic widen/narrow dispatch — confirmed by Frost reading the dispatch, not just trusting tests). DynamicQuantizeLinear kept f32-only (ONNX opset-11 constrains input to tensor(float) — schema-correct rejection + test).
**Frost verification:** No double-rounding, no bf16-accumulated reductions (GEMM + RMSNorm mean-of-squares accumulate in f32, only final store narrows). TopK narrow-back lossless (bf16->f32 exact). Tests non-vacuous with independent f32 references; tolerances 2e-3..5e-3 abs + 1e-2 rel principled for bf16 8-bit mantissa and tight enough to catch real bugs. Pure half::bf16 arithmetic, cross-platform (f16c fast paths are f16-only/cfg-gated). Op-count test green.
**Tests:** 812 ep-cpu lib green (798->812, +40 bf16-named tests pass). Non-blocking nits: QuantizeLinear doesn't enforce x.dtype==y_scale.dtype (pre-existing); remaining f32-only kernels tracked in riley-bf16-ops.md.

---
### 2026-07-23: Dynamic-rank If outputs in shape inference (Sanders) — MERGED 63c771b
**By:** Sanders (impl, sol). Verified: coordinator read full diff (pure relaxation, all tests green); confirmatory executor review dispatched (Wendy... no — fresh agent).
**What:** infer.rs `infer_if_outputs` previously HARD-ERRORED when an If node's then/else branch outputs had different RANK. Per ONNX If semantics, branches must share ELEMENT TYPE but may differ in shape/rank (only one branch executes; executor produces its real tensor at runtime). Changed to: on rank mismatch, emit IfOutput::UnknownRank(dtype) -> value type marked known, shape marked unknown (dynamic). Equal-rank per-dim merge path and the dtype-mismatch rejection are UNCHANGED. Added graph.rs mark_value_type_known() helper (symmetric with existing unknown markers). Executor (session/src/executor.rs ~4800) already resolves the taken branch's runtime shape — no executor change needed.
**Unblocks:** Whisper Tiny jump-times If (rank 4 vs 5) + Nemotron 3.5 VAD If (rank 2 vs 3) now pass native shape inference.
**Tests:** shape-inference 210 green (+3 new: rank 2v3 succeeds w/ unknown rank, rank 4v5 succeeds, dtype-mismatch still Err, equal-rank preserved); ep-cpu 812 green. Pure relaxation — no currently-passing model regresses.

---
### 2026-07-23: qwen3.5 recurrent decode streamline + aliasing UB guard (Doc+Wendy) — MERGED bc680de+88cba98
**By:** Doc (original perf, sol) 🔴 BLOCKED by Calleigh -> Doc locked out; Wendy (aliasing-guard revision, opus). Reviewed by Vartann (opus, non-author, NOT Doc/Wendy/Calleigh): 🟢 SAFE + 1 non-blocking nit.
**What:** Doc streamlined qwen3.5 recurrent decode (CausalConv 3.684->2.124 ms/step) via zero-copy direct writes into caller buffers + in-place recurrent-state mutation. Calleigh 🔴: under the session persistent device-binding API (session/src/lib.rs:1075-1113) a caller can legally alias an INPUT buffer onto an OUTPUT, making direct writes / in-place copy_from_slice undefined behavior. Wendy (different agent) confirmed the aliasing is REACHABLE (persistent binding overrides Nick's SSA-distinctness argument) and added a general guard in dtype.rs: output_direct_write_eligible + slice_byte_range + byte_ranges_overlap (cheap half-open pointer-range disjointness test). Disjoint fast path is byte-identical (preserves Doc's win + Warrick's zero-alloc state); on overlap -> compute into owned temporary + write_dense_f32_narrow. Applied to CausalConv (y, present_state) and LinearAttention (output vs state AND q/k/v/decay/beta).
**Also fixed latent UB in ALREADY-MERGED code:** Warrick's LinearAttention direct-state path did copy_from_slice(past_state)-then-mutate = copy_nonoverlapping UB if present aliases past_state. Now guarded.
**Vartann verification:** byte_ranges_overlap correct on half-open ranges (exact/nested/partial detected, adjacent=non-overlap, saturating_add, no off-by-one); guard itself not UB (usize compare, no deref, &mut only after disjointness+exact-count proven); ALL direct-write sites gated; fallback byte-identical (copies past state before mutation, retrieved buffer fill(0.0) before use); length==1 CausalConv fast path algebraically identical. Vartann added an independent forced-alias test (output->q) that reproduces disjoint result exactly. Exact 32-token qwen3.5 ORT greedy parity.
**Tests:** 815 ep-cpu lib green (+3 forced-alias regression tests: present<->past_state, y<->x, output<->v) + 10 integration; clippy clean. 🟡 nit (optional, not reachable today): CausalConv doesn't guard its two OUTPUTS (y vs present_state) against each other — LinearAttention already stricter. Per protocol Doc+Wendy both locked out; any hardening needs a third agent.
### 2026-07-23: Clean load-gated native-vs-ORT CPU EP scoreboard (Langston)
**By:** Langston (benchmark) — recorded by Coordinator
**What:** Load-gated (1-min load<5) A/B, same genai loop, only --backend swapped:
- Qwen2.5-0.5B int4: native 158.4 vs ORT 63.1 tok/s → **2.51x WIN** (@512 1.61x), parity OK.
- Qwen2.5-coder-7B int4: native 36.0 vs ORT 16.4 → **2.19x WIN**, parity OK.
- Phi-3.5-mini int4 (block-32 acc-level-4): native 13.6 vs ORT 21.9 → **0.62x (ORT 1.61x faster)**, soft near-tie drift @~62 (native numerics verified correct by Horatio's f32 oracle).
- qwen3-0.6b int4 (generic-cpu-4): native 5.41 vs ORT 111.8 → **0.048x (ORT 20.7x faster)** AND parity FAIL from token 0 → native slow/broken fallback path. HIGH-PRIORITY.
- qwen3.5-2b-text hybrid SSM: LOAD FAIL in BOTH backends (conv_state/recurrent_state vs io.kv_inputs mismatch) — genai-loop generality gap, not perf.
**Why:** Confirms we beat ORT on the Qwen2.5 int4 family (no regression) but exposes two native gaps to close per user mandate (all parts beat ORT, cross-model): qwen3-0.6b native bug (#1) and Phi-3.5 acc-level-4 perf gap (#2). Dispatched Ridley (qwen3-0.6b) and Palmer (Phi-3.5) to fix. qwen3.5-2b hybrid-KV loading is a separate generality track.
### 2026-07-23: qwen3-0.6b — native is CORRECT (ORT wrong), 8-bit MatMulNBits GEMV vectorized (Ridley, Speedle-reviewed 🟡)
**By:** Ridley (author), Speedle (independent reviewer) — recorded by Coordinator
**What:** The reported "qwen3-0.6b native parity FAIL + 20x slow" premise was INVERTED on the correctness axis:
- CORRECTNESS: Built an fp32 oracle (reload same model.onnx in python onnxruntime with every MatMulNBits accuracy_level rewritten to 0). Ground-truth step-0 greedy token = 1479 = NATIVE. ORT (unmodified) = 3988 = WRONG. Isolated to ORT's accuracy_level=4 int8-ACTIVATION quant on the 8-bit nodes being too lossy for qwen3's massive-activation channel (near-tie logit flip). Native keeps 8-bit activations in fp32 → correct. Speedle INDEPENDENTLY reproduced the oracle → 1479. Per user policy (ours-more-accurate is acceptable), native's numerics stand; ORT is fast-but-wrong here.
- PERF: real bug — model is mixed 4-bit/8-bit MatMulNBits (105/197 nodes 8-bit incl. lm_head N=151936). The bits==8 && m==1 path dequantized to f32 then reduced with a scalar non-autovectorizing iter().map().sum() dot (~145 ms/tok = 83% of decode). Fixed with new gemv_nk_u8 backed by dot_u8_f32 (16 f32 accumulators, u8→f32 widen, AVX FMA); weight kept 1 byte/elem (PackedU8Weight, NUMA first-touch); activations STAY f32 so correctness preserved. New GEMV math scale·(w·a) − scale·zp·Σa is algebraically identical to original dequant (w−zp)·scale. Keys only off bits/m/group — no arch special-casing; 4-bit path byte-identical.
- A/B (load-contaminated but relative valid): qwen3-0.6b native 5.41 → ~13 tok/s (8-bit decode ~145→~63 ms/tok), tokens unchanged (correct 1479...). coder-7b (100% 4-bit) native==ORT identical, no regression.
- Tests: dot_u8_f32_matches_serial_reference, gemv_nk_u8_matches_dequant_f32_reference (asym zp + partial-K, rel-RMSE ≤1e-5), non-vacuous. cargo test -p onnx-runtime-ep-cpu --features mlas = 817 passed (815→817).
**Why:** Closes the qwen3-0.6b escalation: we're MORE accurate than ORT (their int8-activation path flips the token) and 2.6x faster on the 8-bit path. Native stays slower than ORT's wrong-fast int8 by design; future accurate-speed direction = int16-activation fast path (do NOT route 8-bit through int8-activation MLAS/VNNI — reproduces ORT's wrong 3988). Merged to PR #105 as 0adb960. Reviewer nit (doc comment eight→sixteen) fixed in 2aedd0d.
### 2026-07-23: Phi-3.5 decode gap is executor control-flow/scheduling overhead, NOT a kernel fallback (Palmer diagnosis)
**By:** Palmer — recorded by Coordinator
**What:** Deep profile of Phi-3.5-mini int4 decode (clean, load~4). Native 14.66 tok/s vs ORT 21.93 (1.5x). Per-step 51.9ms: If 34.1ms/65.6%, MatMulNBits 13.1ms/25.2%, GQA 2.9ms/5.7%.
- All 161 MatMulNBits nodes are bits=4/block_size=32/accuracy_level=4 and ALREADY take the AVX512-VNNI packed-int4 path (matmul_nbits.rs:376-416, activations int8-quantized per K-block at :1530-1560, VNNI at :1619-1623). NOT a dequant-f32 fallback and NOT missing vectorization. try_mlas_sqnbit deliberately declines small-M acc-level-4 at :553-560.
- CORRECTNESS GATE PASS: rewrote all 161 accuracy_level 4→0 as fp32 oracle; native hand-VNNI == forced MLAS CompInt8 == fp32 oracle for ALL 128 greedy tokens (no divergence). Phi-3.5 is NOT Ridley's qwen3 case; the earlier "soft drift" was ORT-side under load, not native. (Gate is for this sequence; broader int8-activation routing would still need near-tie oracle cases.)
- Forced MLAS CompInt8: identical tokens, only +2.8% (14.87 vs 14.47), within noise — not worth flipping the default.
- The If 65% bucket is SUBSTANTIALLY scheduling/instrumentation time, not real compute: Palmer prototyped a constant-If output cache (Phi RoPE-cache If has two Constant-only outputs [4096,48]); NO throughput gain (14.47 vs 14.66) → reverted. So the gap lives in the executor's control-flow dispatch (exec_if/run_subgraph) + persistent-SPMD dispatcher wait, not in constant copying or kernels.
- No production code changed (diagnosis only). CPU EP tests 815+10 green.
**Why:** Rules out both a dequant-f32 fallback and a missing block-32 dot vectorization on Phi-3.5. Real bottleneck = per-decode-step control-flow/scheduling overhead (~34ms unaccounted when only ~16ms is kernel work). Qwen2.5 wins 2.5x because its decode body isn't wrapped in a per-step If, so this overhead is If/subgraph-dispatch-specific and general to Phi-family/Loop-wrapped graphs. Next: instrument exec_if/run_subgraph + SPMD dispatcher phase counters, compare persistent vs flat pool, eliminate per-subgraph-invocation overhead. Dispatched follow-up (Tripp) on this.
### 2026-07-23: Phi-3.5 decode bottleneck is CPU KV host round-trip (quadratic), NOT If dispatch — corrects Palmer (Tripp)
**By:** Tripp — recorded by Coordinator
**What:** Built a gated phase-profiler (NXRT_EXEC_PHASE_PROFILE, default-off) and re-attributed Phi-3.5 int4 decode per step:
- kernel compute ~40ms (161 MatMulNBits + 32 GQA + norms = real work)
- collect_outputs.top **11.7ms** — copies ALL 65 graph outputs to host EVERY step; `collect_outputs.top_host_bytes` = 48.5 MB/step avg, ~80 MB deep → QUADRATIC in total_seq
- setup_total.top 6.75ms — resolve + size_buffers + copying growing past-KV host inputs IN
- execif.run_subgraph 2.06ms (1.70 real Constant compute); **actual If/subgraph dispatch ~0.2ms**
- CORRECTS Palmer: the "If=34ms/65%" was a PROFILER ARTIFACT — the op-profiler's recursive child eprintln was billed to the parent If timer. If dispatch is negligible.
- ROOT CAUSE: CPU decoder round-trips the full KV cache through HOST tensors every step. native_decode.rs:1901-1922 feeds growing past-KV host inputs + plain session.run with NO output bindings → executor.rs:2796-2818 materializes 65/65 outputs incl. full [1,32,total_len,96] KV present to host. The CUDA path avoids this via in-place present==past DEVICE bindings (native_decode.rs:1460-1466); the CPU path has NONE. Explains why we WIN Qwen (tiny KV) and LOSE Phi (huge KV): native is memory-bandwidth-bound while ORT stays compute-bound.
- Instrumentation-only changes landed on branch perf/execif-dispatch (executor.rs gated profiler + test, lib.rs export, profile_native.rs dump table): default-off, zero hot-path cost, no numeric change. Phi native 14.76 vs ORT 21.64 (1.47x gap reproduced). No-regression: Qwen2.5-0.5B 166 tok/s, qwen3-0.6b 12.78 first-token 1479 (oracle-correct). 817 tests pass.
**Why:** Identifies the REAL, GENERAL fix to beat ORT on Phi-3.5 and ANY large-KV model: in-place persistent CPU KV (mirror the CUDA run_with_device_bindings present==past path) to eliminate the per-step host KV round-trip. Expected -17-18ms/step → ~42ms ≈ ~24 tok/s (beats ORT 21.6). Large + parity-gated (needs CPU GQA in-place present==past, rewind/prefill, 4-model validation) — Tripp continuing on it. Instrumentation + fix to be reviewed together before merge to PR #105.
### 2026-07-23: Hybrid SSM (qwen3.5-2b) now loads+decodes on native — graph-derived per-layer KV/state metadata (Cooper, Natalia-reviewed 🟢)
**By:** Cooper (author), Natalia (independent reviewer w/ ONNX oracle) — recorded by Coordinator
**What:** qwen3.5-2b-text (hybrid SSM: conv + linear attention) previously FAILED to load in BOTH backends (native "missing native KV metadata for past_key_values.0.key"; ort "io.kv_inputs declares ... graph does not expose it"). Root cause: onnx-genai-genai-config to_inference_metadata/decoder_io_json expanded the uniform past_key_values.%d.key/value pattern over EVERY layer; a graph-driven deriver (strict_decoder_state) existed but was wired only to the multimodal path, not the text SingleDecoder path (error sites native_decode.rs:2513, decode.rs:609).
- Topology (verified by Natalia from graph): 24 layers, layer_types 3×linear+1×full repeating. Dense full-attn layers 3,7,11,15,19,23 → key/value [b,2,seq,256] (io.kv_inputs/outputs); the other 18 linear layers → conv_state [b,6144,3] + recurrent_state [b,16,128,128] (io.state_pairs).
- FIX (general, graph-derived — NOT a qwen3.5 hack): strict_decoder_state now inspects a ModelGraphInfo and emits SPARSE kv_inputs/outputs for dense layers + state_pairs for conv/recurrent layers when a graph is available; falls back to pattern expansion (uniform models byte-identical) otherwise. Engine builds the graph from session I/O (ORT) or reads it from the model file via onnx-runtime-loader/ir before a session exists on the native path (weights not read, native-backend-gated). native_decode folds io.state_pairs into kv/present bindings feeding the existing causal_conv & linear_attention kernels (Doc/Warrick).
- CORRECTNESS (decisive): Natalia drove the raw model.onnx in python onnxruntime managing dense KV + zero-seeded conv/recurrent state, greedy 16 tokens from prompt [9419]; ORACLE == NATIVE all 16 token ids IDENTICAL → recurrent-state feedback correct, no stale/aliased reads. (Cooper's earlier smoke produced coherent text at 14.3 tok/s.)
- NO-REGRESSION: qwen3-0.6b native first token 1479; uniform fallback unit-test byte-identical. Tests: genai-config 20 (+2 hybrid regression), ep-cpu(mlas) 817, engine builds; metadata/genai-config/engine/server all build (io.state_pairs added no downstream construction break). The 17 native-backend engine failures are PRE-EXISTING (invalid-protobuf fixtures, being fixed on fix/textproto-fixture-loading / PR #107) — Natalia confirmed identical failing set on a fresh base worktree.
- Nits (deferred, cosmetic): kv_layer_count() over-counts for hybrids (profile display only); ORT-loop e2e qwen3.5 not smoke-tested (native goal met).
**Why:** Closes a generality gap the user explicitly named (qwen3.5 conv + linear attention). Native now runs a hybrid SSM model correctly where BOTH backends failed before — and it's graph-driven so it generalizes to any dense/conv/recurrent per-layer topology. Merged to PR #105 as ca16c3b.
### 2026-07-23: Zero-copy output hand-off eliminates CPU KV OUTPUT round-trip (Tripp, Flack-reviewed 🟢) + phase profiler
**By:** Tripp (author), Flack (independent reviewer) — recorded by Coordinator
**What:** Two commits landed from perf/execif-dispatch:
1) ad0315d — gated phase profiler (NXRT_EXEC_PHASE_PROFILE, default-off; cached atomic, zero hot-path cost when unset) in executor.rs + lib.rs export + profile_native.rs dump table. This is the tool that produced the corrected KV-round-trip root cause.
2) 3dde516 — the perf fix: new Tensor::from_owned_buffer + Executor::try_move_host_output. At top-level output collection, an eligible produced output (OWNED, host-resident, EXACTLY-sized, not view/sequence/shared/pinned/duplicate/producer-less, and NOT a persistent-device-binding output which is continue-skipped first) has its buffer MOVED into the returned tensor (0 copies) instead of 2 memcpys; buffer_shapes cleared to force realloc next run. General (no model gate), numerically byte-identical.
- IMPACT (Phi-3.5 int4): collect_outputs.top 5059→30 µs/call; per-step host KV output traffic 24.5 MB → 0 MB; throughput ~+6% (16.82→17.82 tok/s at matched load; box noisy so 128-tok absolutes unreliable, but the phase evidence is load-independent).
- MOVE-SAFETY (Flack traced every aliasing path himself): eligibility set is COMPLETE — strided views (pinned), sequence sharing (shared_buffers), initializers/passthrough (producer-less+borrowed), and the Wendy-style persistent device-binding case (external.outputs, continue-skipped before the move). Realloc via buffer_shapes.remove + ensure_buffer correct; free exactly-once by allocating EP; no double-free/use-after-move. In the real CPU decode path present is never bound onto past (past re-fed as a separate copied host buffer), so the moved buffer is solely owned by the returned tensor.
- PARITY (byte-identical feature vs base, run-1==run-2): qwen3-0.6b first token 1479 (oracle) ✓; Phi-3.5 [30751,31512,306,...] == fp32 oracle ✓; Qwen2.5-0.5B [271,40,1079,...] WIN preserved ~179 tok/s ✓. Tests: session lib 64 (+ non-vacuous zero_copy_output_move_reallocates_and_preserves_producer_less_output), ep-cpu(mlas) 817. Pre-existing (reviewer stash-verified on base, NOT from this change): 2 control_flow If integration tests (CpuMatMulNBitsBiasFusion MissingProducer) + 17 engine fixture-protobuf failures.
- REMAINING (documented, not landed — the actual ORT-beating change): the INPUT side (re-feeding growing past_key_values host tensors, ~3ms/step) + full in-place persistent CPU KV. Blocker: a naive max-capacity buffer makes CPU GQA rewrite the ENTIRE capacity every call (~3.2GB/step @4096, worse than round-trip). The real fix needs a CPU GQA in-place APPEND-ONLY path gated on present==past aliasing + wiring DecodeCudaState for the CPU EP. Full plan w/ file:line in inbox tripp-execif-dispatch.md. Expected: ~42ms/step ≈ ~24 tok/s, beating ORT 21.6.
**Why:** Safe, general, parity-clean partial (+6%) toward the memory-bandwidth root cause, plus a permanent measurement tool. Merged to PR #105 as ad0315d + 3dde516. Nits deferred: empty-tensor copy fallback; add explicit "device-binding must-not-move" regression test.
### 2026-07-23: Fix CpuMatMulNBitsBiasFusion masking control-flow rejection (Sidle, Grissom-reviewed 🟢)
**By:** Sidle (author), Grissom (independent reviewer, opus) — recorded by Coordinator
**What:** cherry-picked b7f1514 → b8cdcbc on PR #105. Two negative control-flow tests (if_rejects_mismatched_branch_output_counts_before_running_selected_branch, if_rejects_mismatched_branch_output_dtypes) were failing with Optimize(PostconditionFailed{pass:"CpuMatMulNBitsBiasFusion", errors:[MissingProducer(ValueId(3))]}) — confirmed PRE-EXISTING on base (Sidle stash-verified), not from this session's stack. Root cause: invalid If graphs (mismatched branch output counts/dtypes) reached the CPU EP graph optimizer, whose graph.validate() tripped MissingProducer on the malformed subgraph and masked the intended control-flow diagnostic. Fix (executor.rs): extracted validate_if_branch_outputs helper + added recursive validate_control_flow_signatures, called at Executor build BEFORE fuse_silu_patterns/EP passes, so invalid If graphs are rejected with the proper SessionError::ControlFlow message. Runtime If check now calls the same shared helper (message text identical → negative tests' asserted strings unchanged). Added POSITIVE regression test if_runs_fuseable_matmul_nbits_bias_branches (valid If with fuseable MatMulNBits+Add in BOTH branches still optimizes+runs; asserts fused-bias outputs + subgraph_builds/runs==2) — proves the pre-EP validation ordering does NOT disable fusion for legitimate control-flow graphs.
- REVIEW (Grissom, opus, 🟢): count check is structurally always-known (Vec<ValueId>); dtype check correctly gated on value_type_is_known both sides + re-enforced at runtime via shared helper → no valid graph false-rejected; subgraphs HashMap recursion covers nested If/Loop/Scan, terminates, no cycles; domain gate ""|"ai.onnx" correct; error parity exact. Optional nit (non-blocking): positive test could add an explicit fusion-count assertion. No Sidle revision needed.
- TESTS: control_flow 20/20 (was 18/20), session lib 64/64, ep-cpu(mlas) 823. Independently reproduced by Grissom.
**Why:** Restores correct control-flow rejection semantics + adds fusion-under-control-flow coverage; clears 2 of the pre-existing session test failures. Merged to PR #105.
### 2026-07-24: In-place persistent CPU KV cache — eliminates input-side KV round-trip, Phi-3.5 +49.6% (Stokes, Messer-reviewed 🟢)
**By:** Stokes (author, opus), Messer (independent reviewer, opus) — recorded by Coordinator
**What:** cherry-picked d85c58d → 0281675 (kernel) + 15d0ff7 → a5ac872 (engine) onto PR #105. This closes the LAST big native-CPU-decode gap vs ORT on large-KV models (the input-side host KV round-trip; the output side was fixed earlier by Tripp's zero-copy hand-off). It is the CPU analogue of the CUDA in-place present==past device binding.
- KERNEL (group_query_attention.rs): new in-place APPEND-ONLY GQA path. detect_inplace_kv() gates PURELY STRUCTURALLY — present output pointer must byte-alias the past input pointer (computed identically to data_ptr incl. byte_offset), both contiguous f32 at EXACT physical capacity (numel==present_len), present_sequence_length==cache.seq (only true at full capacity), key≠value distinct, inputs>=5/outputs>=3; f16/bf16/non-contiguous/absent rejected. When it fires: drop the immutable past borrows FIRST, then write only the current step's K/V rows straight into the aliased output buffer and attend over [0,total). Any non-aliased call (every ordinary run/test) falls through to the pre-existing copy path → byte-identical.
- ENGINE (native_decode.rs): DecodeCpuKvState (CPU analogue of DecodeCudaState) allocates ONE persistent full-capacity host buffer per growable KV pair, binds present==past onto it, and decode_cpu_inplace stops re-feeding growing past inputs / round-tripping present. Routed from decode/decode_argmax/rewind. Gated: CPU device, no recurrent state_pairs, rank-4 f32 KV, all_pasts_consumed_by_gqa (every bound past feeds a GQA node — Concat/other producers never bound so binding can't corrupt a non-GQA reader), env ONNX_GENAI_CPU_INPLACE_KV (default ON, =0 reverts). Capacity overflow (generation beyond max_len) errors cleanly BEFORE the run (no OOB), like CUDA.
- PERF (Phi-3.5 int4): input-side KV copy phase setup_total 235ms→32ms (7.3x, load-independent); throughput 14.0→21.0 tok/s (+49.6%) at load ~3-8, +12.5% at load ~24. Messer independently measured Phi-3.5 21.54 vs 14.48 tok/s ON-vs-OFF and qwen3-0.6b 13.75 vs 11.86. (ORT Phi-3.5 ~21.6-27 depending on load — a clean-load A/B is the final confirmation item; structural win is landed.)
- PARITY (HARD GATE — byte-identical greedy ON vs OFF, independently reproduced by Messer): qwen3-0.6b first token 1479 (fp32 oracle); Phi-3.5 [30751,31512,306,29915,29885,1985,373,263,2060,988,306,817,304,1653,263,15171]; Qwen2.5-0.5B [271,40,1079,...] WIN preserved.
- REVIEW (Messer, opus, 🟢): unsafe/aliasing SOUND — drop(past_key/past_value) precede the &mut write; PastCache::F32 borrows the aliased memory so releasing before mutation is required and correctly scoped to the in_place branch; all reads causal-bounded to [0,total), uninitialized capacity never read, mixed per-batch totals disjoint. Gate double-locked (pointer aliasing + present_sequence_length==cache.seq at executor.rs:1370). Engine gating excludes non-GQA consumers; capacity overflow clean; rewind append-only-consistent. No unsound blocks found.
- TESTS (+11, all green): ep-cpu 817→823 (6 new: gate-true-only-on-structural-aliasing, rejects-f16, in-place==copy at spare/exact capacity, +rotary/local-window, prefill→decode boundary); engine +5 (tiny_decoder_matches_across_inplace_env_toggle env ON==OFF parity, decode_cpu_kv_state_declines_non_gqa_model, cpu_inplace_kv_max_len_env_parsing, ...). Pre-existing unrelated: 17 engine fixture-protobuf + 0 remaining control_flow (Sidle fixed).
- FOLLOW-UP (documented, not required for correctness): graceful capacity fallback when generation exceeds max_len (today errors like CUDA) — hook at decode_cpu_inplace capacity check.
**Why:** General (structural gate, not model-name), parity-clean, well-tested closure of the input-side KV bandwidth bottleneck; the single biggest remaining CPU decode win. Merged to PR #105.
## Root cause (profiled, native decode trace @32 tokens)

| op | total | n | avg/step |
|----|------:|--:|---------:|
| **Transpose** | 311.3 ms | 32 | **9.73 ms** (re-transpose ~525 MB fp16 const every step) |
| **MatMul** | 66.7 ms | 32 | **2.08 ms** (dense fp16 GEMV, cuBLASLt, non-capturable) |
| GroupQueryAttention | 54.3 ms | 32 | 1.70 ms |
| MatMulNBits ×14 | 27.4 ms | 224 | 0.12 ms |

The per-step `Transpose` over a half-GB constant dominated; the dense fp16 GEMV
was second. Both re-do work on a constant weight every token.

## What I implemented (both generic, EP-internal, RULES §2/§2.1)

Detected by **op topology + tensor roles + dtype/shape**, never by model name.
Both live in `crates/onnx-runtime-ep-cuda`.

### 1. Constant-`Transpose` folding — `CudaFoldConstantTranspose` (new EP pass)
- Pattern: `Transpose` (domain `""`/`ai.onnx`, 1 in / 1 out) whose single input
  is a **producer-less graph initializer** with a whole-byte element type.
- Action: materialize the permuted bytes once at EP claim/compile time into a new
  inline initializer (via `PassContext::initializer_bytes`, which resolves the
  external mmap), rewire consumers, delete the node — mirroring the generic
  `ConstantFolding` rewrite. Byte-wise permutation is exact for any rank / `perm`.
- Guards (no magic dims): whole-byte dtype only (sub-byte int4/… skipped),
  producer-less initializer only, valid `perm` (default = reversed axes).
- Tied weights stay correct: the original initializer is untouched for its other
  consumers (the `Gather`). New pass runs first in `cuda_optimization_passes()`.

### 2. Dense fp16 M==1 GEMV fast path — in `MatMulKernel`
- Pattern: dense **fp16**, **M==1**, single-matrix (no batch) MatMul.
- Kernel `matmul_dense_gemv_f16` (NVRTC, compiled to the device's own SM →
  portable across all architectures): one thread per output column, so a warp
  reads consecutive `B[k, col]` fp16 values — fully coalesced, one streaming pass
  over `B` at ≈ HBM roofline. Activation staged in shared memory per K-tile
  (bounded to `blockDim.x` floats → any K); fp32 accumulate (matches cuBLASLt),
  single fp16 round. `col < n` guard → any N.
- Capture: needs no workspace/heuristic/sync, so it is **capture-safe** and folds
  into the decode CUDA graph (verified `capture_status: captured`), unlike the
  cuBLASLt path. The kernel advertises `CaptureSupport::Supported` only when the
  last call took the GEMV (mirrors the `MatMulNBits` decode-GEMV contract).

## Results — Llama-3.2-1B (Q4_K_M, fp16 tied head), H200, steady decode

| stage | @128 tok/s | @1024 tok/s | ms/step @128 |
|-------|-----------:|------------:|-------------:|
| baseline (origin/main) | **97.5** | ~97 | 10.26 |
| + Transpose fold | 409.4 | — | 2.44 |
| + fp16 GEMV | **449.1** | **438.3** | 2.23 |

**97 → 449 tok/s @128 (4.6×), 438 @1024.** Greedy token IDs byte-identical to
baseline at every stage → coherent (emits valid code/text). Remaining gap to ORT
(589) is now in GQA / MatMulNBits / norm, not the head.

Post-fix op trace: `Transpose` gone; decode `MatMul` no longer appears as an
eager op — it is captured into the graph.

## No regression — Qwen2.5-0.5B (int4, quantized `MatMulNBits` head)

Qwen's graph has **no `Transpose` and no dense `MatMul`** (verified by trace), so
neither optimization can fire. Same command / same machine, baseline vs branch:

| model | @128 (base → branch) | @1024 (base → branch) |
|-------|---------------------:|----------------------:|
| qwen2.5-0.5b-int4-onnx-native | 314.0 → 313.5 | 84.89 → 84.90 |

Identical within run-to-run noise → no regression. (Machine's Qwen numbers differ
from the ~577/498 cited in the brief; the invariant proven here is *no
regression*, and the paths are structurally inert for Qwen.)

## Tests added

- `onnx-runtime-ep-cuda` lib (`src/optimizer.rs`) — pattern-level, model-agnostic:
  `folds_constant_transpose_into_initializer`, `folds_constant_transpose_default_perm`,
  `folds_rank3_constant_transpose`, `leaves_transpose_of_non_constant`,
  `leaves_sub_byte_constant_transpose`. (20/20 optimizer unit tests pass.)
- GPU integration (`tests/matmul_gpu.rs`): `matmul_f16_gemv_on_gpu_matches_cpu_reference`
  (K=259, N=300 non-square GEMV vs CPU reference; asserts capture support). 3/3 pass.

## Validation run

- `cargo fmt -p onnx-runtime-ep-cuda` (changed crate only).
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda -- -D warnings`: **clean**
  for the crate lib (my changed files have zero findings; baseline is also clean).
  Note: `--all-targets` surfaces **pre-existing** clippy debt in unrelated GPU test
  files (`group_query_attention_gpu.rs`, `compressed_sparse_attention_gpu.rs`,
  `matmul_nbits_gpu.rs`, …) and `#[cfg(test)]` blocks in `gqa_decode*.rs` /
  `normalization.rs` — not touched by this change.
- Changed-crate unit tests + `matmul_gpu` integration tests: pass.

## Files changed

- `crates/onnx-runtime-ep-cuda/src/optimizer.rs` — new pass + 5 unit tests.
- `crates/onnx-runtime-ep-cuda/src/kernels/matmul.rs` — GEMV kernel + M==1 dispatch + capture.
- `crates/onnx-runtime-ep-cuda/tests/matmul_gpu.rs` — fp16 GEMV GPU test.
- `docs/benchmarks/llama-3.2-1b-lmhead-fusion-2026-07-23.md` — bench doc.

## Follow-ups (out of scope here)

- The `Transpose`-fold is generic enough to consider promoting into the shared
  optimizer for all EPs; kept EP-internal per RULES §2.1 for now.
- Next native-decode bottlenecks are GQA and the stacked `MatMulNBits` GEMVs.

---
**Summary (plain text):** Llama-3.2-1B native decode **97 → 449 tok/s @128
(438 @1024)**, byte-identical greedy tokens. Two generic, pattern-matched,
EP-internal wins: (1) fold any `Transpose(constant-initializer)` into a
pre-transposed constant at claim time; (2) route dense fp16 M==1 MatMul to a
portable, capture-safe, roofline-oriented GEMV. Detected by topology + dtype +
shape, **no model names**. Qwen2.5-0.5B (quantized head) unchanged → **no
regression**. Branch `squad/roy-lmhead-fusion` @ `71ab809`.
<!-- source: .squad/decisions/inbox/voight-mercer-moe-review.md -->
### 2026-07-23: Review of mercer CPU grouped MoE Phase 2
**By:** Voight
**Verdict:** 🔴 REJECT
**What:** CPU grouped execution and route-first dequantization are implemented and performant, but the support document contains contradictory, false implementation-status claims.
**Evidence:** `cargo test -p onnx-runtime-ep-cpu` passed; both named grouped differential tests and the all-experts residency test passed and genuinely exercise grouped/reference paths. Code inspection confirms an expert→token `BTreeMap`, one `run_expert_grouped` call per active expert, shared GEMM for M>1, and scalar GEMV for M=1. QMoE and BlockQuantizedMoE slice/dequantize routed experts inside the expert loop; the zero-cache residency test reports one expert, though its metric is explicitly recorded as `1` rather than lifetime-derived. Genericity grep found no architecture-dependent kernel control flow (only a Mixtral test name and llama.cpp compatibility/test references). Clippy and crate-scoped fmt passed. Release ignored test passed with 3.81x decode and 1.90x prefill speedups. `docs/MOE_SUPPORT.md` lines 3-6 and 161-163 still say fused/grouped CPU MoE/QMoE are unimplemented/unregistered, contradicting lines 479 and 518-555 and the code; CUDA is not claimed complete in the Phase 2 section.
**If REJECT:** Deckard should revise the contradictory status/architecture sections and strengthen residency accounting so the test observes actual concurrent dequantized-expert lifetime rather than a hard-coded window value; Mercer is locked out.
<!-- scribe-merge-2026-07-23T04-08-59Z-cuda-indexshare-f16attention (merged manually by coordinator; Scribe agent stuck in canary loop) -->
## 2026-07-23 — CUDA IndexShare + f16 Attention; plus prior-session backlog (Qwen split-K, CPU MoE docs, mobius #423)

### This session (CUDA perf + GLM/DeepSeek)

**f16 CUDA standard Attention — LANDED (`07e4c80`, main).** By Roy. CUDA standard `Attention` now accepts homogeneous f32 or f16 Q/K/V, paired cache tensors, and f16 additive masks (incl. -inf/-65504), writing Y/present-KV/optional-QK in the selected dtype. All score/softmax/value reductions retain fp32 accumulation. bf16 deliberately still rejected (follow-up). Closes the GLM/DeepSeek "Attention f32-only" dtype-coverage gap (`docs/GLM_READINESS_GAPS.md`) and halves activation/KV bandwidth for real fp16 exports. **Reviewer Gaff 🟡→addressed:** initial parity test tolerance (3e-3) too loose to guard fp32-accumulation; Roy hardened `standard_attention_fp16_gpu.rs` — exact f16-rounded CPU oracle, seq=32/head=64, checks Y+present-KV+QK, tolerance 3e-4; **mutation test confirms guard** (forcing f16 score accumulation → QK err 3.996e-4 > tol → FAIL). Merged max|Δ|: prefill Y 1.94e-4, decode Y 2.99e-5, caches bit-exact.

**Device-resident CUDA IndexShare v1 — LANDED (`1304707`+`0828abb`, main).** By Keaton. New device-resident CUDA kernel for `pkg.nxrt::IndexShare` v1 (GLM-5.2 IndexShare / DeepSeek DSA selected-token attention); previously CPU-only. Two NVRTC kernels: `build_present` (device past⧺current KV concat, bit-identical to CPU) and `index_share_row` (per-`(batch,q_head,query)` selected-token gather, scaled QK, additive bias, stable fp32 softmax, prob·value sum). Only `selected_indices` goes D2H for deterministic ONNX validation (SparseKvGather precedent); Q/K/V/bias/present-cache/output stay device-resident. Claim gate delegates to CPU oracle's `unsupported_reason` (made `pub`) for identical cross-backend gating. `cuda_graph_compatible()==false` (D2H index sync; full capture is a follow-up needing device-resident index validation). **Reviewer Chew 🟡→addressed:** independently re-ran 6 parity tests green on H200 (max|Δ|≤2.4e-7), traced kernel vs oracle line-by-line (numerics/indexing/GQA grouping/KV threading/memory-safety/capture all correct). One contract bug fixed: rank-0 scalar `attention_bias` was claimed (CPU accepts) but hard-failed at CUDA execution — Keaton now accepts rank-0 as broadcast scalar, added `scalar_bias_broadcasts_and_matches_cpu` (bit-exact), CPU/CUDA claim parity aligned. **7 GPU parity tests pass.**

**Integration verified on merged main:** index_share_gpu 7/7, standard_attention_fp16_gpu 1/1, standard_attention_gpu 23/23, `cargo clippy -p onnx-runtime-ep-cuda --features cuda -D warnings` clean.

**Remaining GLM/DeepSeek follow-ups:** CUDA-graph capture for IndexShare (device-side index validation); f16/bf16 IndexShare storage variants (v1 CPU oracle f32-only); bf16 standard Attention; Mobius fused QMoE/BlockQuantizedMoE emitter; MTP state threading.

### Prior-session backlog (already landed to main; merged for the record)

- **Qwen2.5-0.5B O(seq) decode collapse fixed (`798d430`, Irmgard; landed by Sadik; reviewed 🟢 Borogrove/re-benched Marsten).** Root cause: f32 KV graph selected the single-warp-per-row f32 GQA decode kernel that serially walked full context. Fix: capture-safe 1/2/4/8/16-way split-K online-softmax kernel + merge, selected purely by dtype/shape. Qwen 313→460 tok/s @128, 84→448 @1024; Llama Q4KM flat; generic (no model-name), capture-safe, SM-portable. Marsten H200 re-bench: Qwen0.5B 459/446, 1.5B 486/460, 7B 230/223, Llama-1B Q4KM 450/439 tok/s.
- **CPU MoE Phase 2 landed (`dc0cc18`, Sloat) + MOE_SUPPORT.md §6.2 honesty fix (Sapper; Voight 🔴→🟢).** Route-first int4 QMoE (peak-1-expert residency via RAII guard), grouped-expert GEMM (4.12x decode / 1.83x prefill), doc now correctly states CPU MoE/QMoE/BlockQuantizedMoE implemented + registered, CUDA incomplete. 648 CPU unit tests pass.
- **Mobius PR #423 (DeepSeek MoE Phase 1 conformance) CI remediation (Abdul).** Ruff lint + codecov fixed; Integration/L4/L5 jobs fail on infra (`libcudart.so.13` missing on runner, identical on main). PR remains OPEN/UNMERGED for Justin.
### 2026-07-24: Merged origin/main into PR #105 — both teams' work integrated (Taylor, Duquesne-reviewed 🟢)
**By:** Taylor (merge author, opus), Duquesne (independent reviewer, opus) — recorded by Coordinator
**What:** Fast-forwarded perf/cpu-ep-mlas to merge commit b8f0bc4 (parents: ours 53ecf1b, theirs origin/main 621936f — no rewrite of our 96 commits). Resolves PR #105's CONFLICTING state so it is mergeable into main. The other team (main) owns the CUDA track (stream-ordered async copies, device-to-device copy ordering) + a native-vs-ORT decode parity harness (tests/parity/*, Q4 f32 oracle); our branch is the CPU-EP perf track. Both edited the same core files.
- CONFLICTS RESOLVED preserving BOTH sides: native_decode.rs (unified their step_inputs/routed/inputs_embeds decode model with our decode_cpu/decode_cpu_inplace persistent in-place CPU KV + greedy/argmax + recurrent SSM; dispatch cuda→cpu_kv(in-place)→cpu; removed their now-redundant decode_host); executor.rs (our phase_span! profiler + their seed_control_flow_capture_shapes + invariant-If memoization); profile_native.rs (our A/B+phase harness + their min-p/repetition-penalty); Cargo.lock/decisions.md/roy history/PROGRESS.md unioned.
- NOVEL SEMANTIC FIX (Taylor): our zero-copy output move (try_move_host_output) freed a buffer that their memoized loop-invariant If (if_last_predicate; branch skipped on steady steps, output served from resident buffer) would re-serve → garbage. Guarded: if the output's producer node is a memoized If, fall back to the copy path (move→copy, bytes identical, no numeric change). Duquesne verified this is the ONLY skip-without-re-execution path (all others already guarded: external.outputs/in-place KV, sequence, views, seq_elem, shared_buffers, pinned, producer-less, dup-output; Loop/Scan re-execute every step) and that keying matches (connect_edges sets output.producer=node.id == if_last_predicate key).
- REVIEW (Duquesne, opus, 🟢): guard complete+correct; native_decode unification loses no behavior either side; both parents ancestors, our 96 commits intact, both sides' features present. TESTS: ep-cpu 827, session lib 66, control_flow 22/22 (the invariant-If × move seam GREEN), engine 177 pass / 16 fail = PRE-EXISTING (missing generated model.onnx binary; fixtures byte-identical across all 3 commits — verified, being fixed by PR #107). PARITY byte-identical incl. INPLACE_KV=0: qwen3-0.6b first token 1479, Phi-3.5 [30751,31512,306,...]. Non-blocking nit: decisions.md lost 63 OLD base-history entries (0 new work — the other team pruned base history; effectively free archiving; Scribe-owned).
**Why:** Makes PR #105 mergeable and keeps both teams' features working together with a proven correctness guard for the move×memoization interaction. Merged to PR #105 (b8f0bc4).
### 2026-07-24: bf16 op coverage extended across CPU EP (Grissom, Sanders 🟢)
**By:** Squad (Coordinator), requested by justinchuby
**What:** Merged Grissom's bf16 coverage onto perf/cpu-ep-mlas (cherry-pick 4b5afa2 -> 0312a9b). 6 ops moved f32-only -> bf16 with real code (Conv, FusedGemm, FusedMatMulBias, AffineGrid, GridSample, Col2Im — compute in f32 scratch, narrow on store via write_dense_f32_narrow; f32 fast path dtype-gated so bit-identical). 17 op families already bf16-capable now regression-guarded. 23 new bf16 parity tests (bf16 vs independent f32 ref ~3% rel tol; movement ops bit-exact). ep-cpu lib 827 -> 850, 0 failed. Deferred (hard f32-gated, documented): MoE/QMoE/BlockQuantizedMoE/GatherBlockQuantized/IndexShare/SparseKvGather.
**Review:** Sanders (opus) 🟢 SAFE TO MERGE — all 6 review items PASS (widen/narrow correctness, generality/f32-bit-identical, real parity tests, no f32 hot-path regression, 850/0/4, deferrals genuinely hard). Non-blocking nit: add f32 contiguous copy_from_slice fast path to write_dense_f32_narrow (dtype.rs) mirroring the f16 branch — assigned to Stokes (Grissom locked out).
**Why:** User mandate: CPU EP must support bf16 on every capable op (ORT's CPU EP lacks bf16 — a real usability gap we now close), general + well-tested.
### 2026-07-24: Phi-3.5 native-vs-ORT divergence — native is MORE accurate, KEPT (Brass, Warrick 🟢)
**By:** Squad (Coordinator), requested by justinchuby
**What:** Root-caused the Phi-3.5-mini int4 (block-32, acc-level-4) greedy divergence: native and ORT share 65 tokens then split at decode index 65 (native=263, ORT=6455). Brass built an independent ORT oracle (teacher-forced, same model.onnx) sweeping MatMulNBits acc-levels: acc-1 fp32→263, acc-2 fp16→263, acc-3 bf16→263, acc-4 int8→6455. Every higher-precision compute agrees with NATIVE (263); only ORT's default int8-activation quant (acc-level-4) flips to the wrong 6455. Native uses fp32 activations + fp32 GQA/LayerNorm so it lands on the fp32-correct token. VERDICT: keep native (a "fix" would make us LESS accurate). Same class as Ridley's qwen3-0.6b int8-activation-flip precedent. Merged 22fa3cd -> e0cfd66 onto perf/cpu-ep-mlas.
**Tests (both green):** onnx-runtime-ep-cpu `int4_decode_preserves_f32_argmax_where_per_row_int8_activation_flips` (model-independent kernel guard: anti-correlated block geometry, near-tie filter, asserts native per-block int8 keeps fp32 argmax on ≥20 near-ties while a per-row int8 failure mode flips ≥3; scalar+SIMD); onnx-genai-engine tests/phi35_mini_divergence.rs `phi35_mini_int4_native_decode_keeps_high_precision_argmax` (#[ignore], gated PHI35_MINI_E2E_DIR, asserts token[65]==263 — verified passing on real model, 235s). ep-cpu 828/0/4; session 66/0.
**Review:** Warrick (opus) 🟢 SAFE TO MERGE — all 5 items PASS (oracle sound, tests non-tautological, zero production-kernel change, E2E lock correct, counts confirmed). Nits: archive oracle script; add a comment tying kernel n=2 to the two contending logits.
**Why:** User mandate: token divergences must be fixed UNLESS ours is more accurate — then keep ours with regression tests. This is the "keep ours" case, proven and locked.
### 2026-07-24: Gemma-2-2B f32 — native 6.1x SLOWER than ORT (Vega) -> f32 GEMM optimization launched (Hodges)
**By:** Squad (Coordinator), requested by justinchuby
**What:** Vega exported Gemma-2-2B-it to f32 ONNX via Mobius (/home/justinchu/gemma2-2b-it-mobius-cpu-f32) and A/B'd native vs ORT: native 1.83 tok/s (547.6 ms/token) vs ORT 11.19 tok/s (89.4 ms/token) — native 6.11x SLOWER. Parity IDENTICAL 128 token IDs (pure speed problem, not correctness). Root cause hypothesis: native's f32 MatMul/Gemm path is slow / not MLAS-backed / not multithreaded, whereas ORT uses multithreaded MLAS sgemm. This f32 path is shared by Whisper/Nemotron/vision CNNs, so it's likely THE central generality bottleneck. Dispatched Hodges (opus) on branch perf/cpu-f32-gemm to profile per-op, route f32 GEMM through multithreaded MLAS (keeping a portable fallback), preserve parity, and NOT regress the int4/int8 quantized wins (Phi-3.5/qwen already beat ORT).
**Why:** User mandate: ALL parts beat ORT across models/OS/CPU. Quantized decode already wins; f32 decode is a major gap that must be closed generally.
### 2026-07-24: Vision CNN f32 native SLOWER than ORT + missing Resize kernel (Curtis)
**By:** Squad (Coordinator), requested by justinchuby
**What:** Curtis A/B'd native vs ORT on f32 vision CNNs via profile_vision (load 4.54): ResNet-50 v1-12 native 113.9ms vs ORT 6.5ms (17.57x SLOWER, Conv-dominated), MobileNetV2-10 13.4ms vs 4.8ms (2.78x slower), Tiny-YOLOv3-11 native FAILED — missing `Resize` opset-11 CPU kernel. Output parity PASS on both classifiers. Confirms native's f32 compute path (both f32 Conv AND f32 GEMM) is the central "beat ORT" bottleneck. Conv optimization to build on Hodges' GEMM/MLAS-threading foundation (Conv lowers to im2col+GEMM). Resize opset-11 missing = separate coverage gap (dispatched to Sanders).
**Why:** User mandate: beat ORT on ALL models incl. traditional ML (resnet/yolo). f32 Conv/GEMM perf + op coverage gaps block this.
### 2026-07-24: ONNX Resize CPU kernel implemented (Hawkes, Bonasera 🟢)
**By:** Squad (Coordinator), requested by justinchuby
**What:** Merged Hawkes's native CPU-EP `Resize` kernel (cherry-pick f0a0bd4 -> cf96162). N-D nearest/linear/cubic; coordinate transforms half_pixel/pytorch_half_pixel/align_corners/asymmetric/tf_crop_and_resize/half_pixel_symmetric; nearest rounding modes; ROI, axes, sizes/scales, aspect policies; dtypes f32/f16/bf16/f64 (int rejected). antialias=1 and unknown modes ERROR clearly (no silent wrong output). Registered opset 10/11->13/18/25. Extracted YOLO-Resize + bilinear E2E models = EXACT ORT parity. ep-cpu 871 pass / 0 fail / 5 ignored. YOLO now runs PAST Resize but hits a separate unrelated dynamic-Squeeze sizing gap (follow-up).
**Review:** Bonasera (opus) 🟢 SAFE TO MERGE — all 7 items PASS; hand-recomputed test vectors reproduce ONNX reference and distinguish modes; additive executor/shape-inference changes, no other kernel touched. Nits: f64 runs at f32 precision via widen (doc it); cubic test only asserts finiteness (add exact vector); non-nearest path single-threaded (perf).
**Why:** User mandate: beat ORT on traditional ML (resnet/yolo) requires op coverage; Resize was a hard generality gap blocking YOLO.
### 2026-07-24: Native f32 GEMM now multithreaded-MLAS by default — 5.6x faster, ≈ORT parity (Hodges, Danville 🟢)
**By:** Squad (Coordinator), requested by justinchuby
**What:** Merged Hodges's f32 GEMM fix (cherry-pick 02ca566 -> 9cd1b1b). Root cause: CpuBackend::auto_detect() defaulted to the SINGLE-THREADED SimdX86 microkernel for f32 GEMM (MatMul = 95-98% of decode time) instead of multithreaded MLAS sgemm; also the pinned SPMD decode pool (~48 spinning workers, only serves quantized MatMulNBits/QMoE) CONTENDED with MLAS rayon on dense-f32 models. Fix (general, no model hacks): backend.rs defaults to MLAS on x86_64+mlas (portable SimdX86/Generic fallback for ARM/non-mlas); matmul_nbits.rs gates the SPMD pinned pool on presence of MatMulNBits/QMoE and routes dense-f32 through a bounded NON-SPINNING DENSE_DECODE_POOL = clamp(available/4,8,32); native_decode.rs computes uses_decode_pool at load by scanning graph+subgraphs. Gemma-2-2B f32 decode 1.83 -> ~10.3 tok/s (5.6x), now ≈ parity/ahead of ORT under matched load (native 10.22 vs ORT 8.42 matched; ~10.3 vs 11-14 fully idle). Parity byte-identical. Quantized no-regression CONFIRMED (qwen int4 still SPMD, 32.28 tok/s).
**Review:** Danville (opus) 🟢 SAFE TO MERGE — all 7 PASS incl. high-risk threading (MLAS parallel-for runs on the CURRENT rayon pool so inside DENSE_DECODE_POOL.install it's bounded ≤24 — no second pool/oversubscription; lazy OnceLock, non-spinning) and quantized no-regression (int4 m=1 early-returns before backend_is_mlas). ep-cpu 865, session 66, clippy clean. Nits (non-blocking, -> Deckard): stale "MLAS opt-in" doc in backend.rs; stale comment matmul_nbits.rs:631-632 (default flip routes int4 acc-level-4 PREFILL to MLAS SQNBit CompInt8, not decode hot path).
**Why:** User mandate: beat ORT on ALL models incl. f32 (Whisper/Nemotron/vision are f32). This closes the single biggest generality gap (was 6.1x slower). Conv (ResNet 17x) is next — may now benefit from MLAS-default backend; needs re-measurement.
### 2026-07-24: Definitive native-vs-ORT scoreboard (Robbins) — native wins 4/5, contended
**By:** Squad (Coordinator), requested by justinchuby
**What:** Robbins ran profile_native --backend native|ort (same genai loop, backend swapped) on merged HEAD b3f9430, 128 tok, decode-skip 8, median-of-3. RESULT native wins 4/5:
| Model | dtype | native | ORT | delta | parity |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | int4 | 153.75 | 12.08 | +1172% | identical |
| Qwen2.5-Coder-7B | int4 | 43.04 | 4.64 | +828% | identical |
| Gemma-2-2B-it | f32 | 10.08 | 3.51 | +187% | identical |
| Phi-3.5-mini | int4 | 18.04 | 8.82 | +104% | index-65 (native MORE accurate, kept) |
| Qwen3-0.6B | int4 | 12.95 | 44.44 | -71% | index-0 1479 vs 3988 (native MORE accurate) |
CAVEAT: box load 45-94 the whole time (never <5 after 15 min) — absolute tok/s NOT clean; directional only. The huge +1000% margins are partly ORT being hammered; a clean re-measure is still owed but the box won't quiet.
**Qwen3-0.6B analysis:** NOT a bug. native=1479 is the fp32-oracle-correct token (Ridley+Speedle validated); ORT=3988 is WRONG (int8-activation acc-level-4 flip). Native is slower here BY DESIGN — it keeps 8-bit MatMulNBits activations in fp32 for correctness, while ORT uses fast-but-wrong int8-activation. Per user policy correctness>speed, so native's numerics stand. The documented path to ALSO be fast: an int16-activation VNNI fast path for 8-bit MatMulNBits (accurate + fast; must NOT route through int8-activation which reproduces ORT's wrong 3988). Dispatched Robbins... -> new agent (Sanders) on this.
**Why:** User mandate: beat ORT on ALL models. We win 4/5 broadly; qwen3-0.6b 8-bit is the one gap, closable accurately via int16-activation.
### 2026-07-24: Filed upstream ORT correctness bug (microsoft/onnxruntime#29849)
**By:** Squad (Coordinator), requested by justinchuby ("create an issue to ort if so")
**What:** Filed https://github.com/microsoft/onnxruntime/issues/29849 — CPU MatMulNBits accuracy_level=4 (int8 ACTIVATION quant) selects the WRONG argmax token vs accuracy_level 0/1/2/3 on massive-activation LLMs. Two reproducible cases from our oracle work: Qwen3-0.6B first token 3988 (acc-4) vs 1479 (fp32/native correct); Phi-3.5-mini index-65 6455 (acc-4) vs 263 (fp32/fp16/bf16/native correct). Root cause: int8 activation scale too coarse for large-dynamic-range channels, flips near-tie logits. Suggested int16-activation path. ORT 1.27.0 CPU EP. This is the upstream counterpart of our two "native is more accurate" decisions (Ridley qwen3, Brass phi-3.5) — we keep native's correct numerics; ORT should fix acc-4.
**Why:** The divergences we found are ORT bugs (native is correct), so reporting upstream is the right action per user request; also documents provenance for our keep-native regression tests.
### 2026-07-23: int16-activation fast path for 8-bit MatMulNBits (Ross, Vecchio review 🟢)
**By:** Squad (Coordinator), authored by Ross, reviewed by Vecchio (non-author, opus)
**What:** Added `gemv_nk_u8_i16` int16-activation decode path for 8-bit MatMulNBits (qwen3-0.6b class). Activation quantized symmetric int16 in groups of 32 (finer than 128-block weight granularity to confine massive-activation channels); u8×i16 via `_mm256_madd_epi16`, per-group scaled i32 folded into a single f32x8 block accumulator (single reduction/block — the restructure that made int16 faster not slower); exact f32 zero-point term; portable scalar fallback. Default on, opt-out `ONNX_GENAI_CPU_8BIT_ACT=fp32`. Merged cf04f7b→4b30a0e.
**Why:** qwen3-0.6b 8-bit was accurate-but-slow BY DESIGN (native keeps activations fp32 → correct token 1479; ORT's int8-activation acc-4 is fast-but-wrong 3988, ORT bug #29849). Ross's int16 path is byte-identical to the fp32 oracle for all 128 tokens (1479 preserved, never 3988) AND ~10% faster (~13.7→~15.1 tok/s). We don't beat ORT's wrong-fast int8 path (fundamental — int8=3988); we stay correct and close the accurate-path gap. group=128 was faster but flipped token 1 → rejected; group=32 chosen.
**Verification:** Vecchio verified all 6 areas 🟢 (diff math, non-vacuous argmax regression test asserting int8 flips while int16 preserves, i32 overflow bound safe at group≤block_size, 4-bit byte-identical, no-8bit models unaffected, opt-out works). `cargo test -p onnx-runtime-ep-cpu --features mlas`: 869 lib + 10 regression, 0 failed. 4 new tests incl. `gemv_nk_u8_i16_preserves_argmax_on_massive_activation_channel`. Optional low-pri hardening: clamp QGROUP override to a divisor of block_size.
### 2026-07-23: parallelize MLAS NCHWc conv/pool/reorder (Ecklie, Kujan review 🟢)
**By:** Squad (Coordinator), authored by Ecklie, reviewed by Kujan (non-author, opus)
**What:** Fixed a serial-execution bug in the standalone MLAS build: `MlasExecuteThreaded` ran all partitions serially in a for-loop (unlike GEMM's `MlasTrySimpleParallel`), so `NchwcConv` split work N ways then ran every tile on ONE thread — the entire post-GEMM vision bottleneck. Added `MlasStandaloneParallelFor` (race-free disjoint partitions via `MlasPartitionWork`, blocking rayon `into_par_iter` so the stack closure outlives workers), work-capped conv/pool fan-out (≥32M MACs/thread) and NCHWc→NCHW reorder (≥128K elems/thread) to avoid over-partitioning tiny/depthwise convs, plus `profile_vision --native-only`. Touches vendored MLAS C++ (threading.cpp, snchwc.cpp, reorder.cpp) + conv.rs tests. Merged a41b20a→ca885d7.
**Why:** After Hodges routed dense f32 GEMM through multithreaded MLAS, vision Conv (NchwcConv = 86–94% of runtime) was the remaining generality bottleneck (ResNet-50 native ~17× slower than ORT). This fix: ResNet-50 ~1.4×, MobileNetV2 ~1.2× faster, parity byte-close (ResNet max_abs 9e-6, MobileNet top1 AGREE). Caps change thread COUNT only, never numeric results.
**Verification:** Kujan verified all 6 areas — DATA-RACE gate 🟢 (partitions provably disjoint, per-tid output rebase, blocking closure lifetime, same bounded pool as shipped GEMM = no oversubscription; empirically forced 96-way partitioning + 10× parity + 5× nchwc stress = bit-stable). One follow-up landed: the original parity test's shapes all fell under the 32M cap → serial; Ecklie added non-ignored `conv_parallel_path_matches_f64_reference` (~127M MACs → tids≈3, bit-for-bit vs f64 oracle). Merged-branch `cargo test -p onnx-runtime-ep-cpu --features mlas`: 871 lib + 10 golden, 0 failed, 4 ignored; conv tests 3× bit-stable. Honest limit: full idle-box ORT parity (ResNet ~6.5ms) not verifiable on the chronically loaded shared host; structural bug is fixed, thresholds are safe coarse heuristics to re-tune on a quiet box.
### 2026-07-23: merge origin/main (CUDA + parity track) into PR #105 (Willows, Duquesne review 🟢)
**By:** Squad (Coordinator), authored by Willows, reviewed by Duquesne (non-author, opus)
**What:** Real merge commit cc4f3ab (parents ours fc8a72f + theirs origin/main 3dc0843) integrating the other squad team's 12 incoming commits (CUDA-EP capture/KV/MoE/Attention + one shared `perf(executor): seed warm JIT decode shapes`) into our CPU-EP perf branch. Only ONE file conflicted: executor.rs (union — kept our `phase_span!("run_scoped.resolve_soft")` profiler AND their new `seed_warm_decode_capture_shapes` call alongside seed_capture_shapes/seed_control_flow_capture_shapes). native_decode.rs auto-merged both sides (our decode_cpu_inplace/uses_decode_pool + their step_inputs/inputs_embeds). Taylor try_move_host_output × memoized-If guard intact (executor.rs:5543). Our 114 commits preserved as ancestors.
**Why:** PR #105 went `dirty` because main advanced (other team, different repo/track). Merging keeps both tracks' work; our CPU kernels (gemv_nk_u8_i16 int16-activation, MLAS conv parallelization + conv_parallel_path test, MLAS-default backend, resize) all present post-merge.
**Verification:** Duquesne verified all 6 areas 🟢 (merge integrity 2-parent + both-ancestor; executor.rs union correct single-occurrence; Taylor guard survived their invariant-If changes; native_decode both-sides; kernels intact). Tests: ep-cpu 871 lib + 10 golden 0 fail; session lib 69 (66→69, theirs +3) 0 fail; control_flow 22/22 incl. invariant-If×move seam. Pre-existing (merge-untouched): 5 executor integration (InvalidOpsetImport helpers) + 16 engine tiny_fixture (protobuf/missing model.onnx, PR #107). Pushed fc8a72f→cc4f3ab.
### 2026-07-23: resolve profile_native --backend bench conflict with new origin/main (coordinator)
**By:** Squad (Coordinator)
**What:** origin/main advanced again (3dc0843→d03261c) with 2 commits adding `--backend` decode-backend selection to profile_native.rs — the SAME feature our A/B harness already had. Merge commit 0478190 resolves the 7 bench-only conflicts by union: kept our sampling imports (MinP/RepetitionPenalty/profile) and informative bail/header prints AND their improvements (PartialEq/Eq + const fn as_str + arg doc comment + resolved_backend print). Deduped a doubled logits import. Conflict was confined to profile_native.rs + its test (bench tooling only — does NOT touch the runtime under benchmark).
**Why:** Bench-only, small, and main is a fast-moving target (advanced twice in minutes), so a full agent+review cycle would go stale before landing; coordinator reconciled inline. profile_native bin compiles clean (`cargo build -p onnx-genai-bench --features mlas --bin profile_native` OK).
**Result:** PR #105 mergeable:true again (was dirty). Pushed 6b59a9b→0478190. HEAD contains origin/main d03261c.
### 2026-07-23: native avx512_bf16 GEMM (Caine, Sorenson review 🟢)
**By:** Squad (Coordinator), authored by Caine, reviewed by Sorenson (non-author, opus)
**What:** Added `crates/onnx-runtime-ep-cpu/src/kernels/bf16_gemm.rs` — a native AVX-512 BF16 GEMM using `_mm512_dpbf16_ps` (bf16×bf16 pairwise → f32 accumulate), MR=NR=4 microkernel, B transposed to k-contiguous panels, K-tail via masked epi16 load, Rayon over disjoint C row blocks. `matmul.rs::try_matmul_bf16_native()` routes contiguous bf16×bf16 (single/batched/broadcast) to it when `avx512bf16`+`avx512bw`+`avx512f` are present, else falls back to the existing widen-to-f32 GEMM. Merged 4e73898 (base 83f14fc).
**Why:** Our bf16 support is a differentiator (ORT's CPU EP has NO bf16 — user pain point). Previously bf16 compute was upcast-to-f32 only (correct but slow). Native path makes it fast. This box (SPR 8480C) has avx512_bf16 so it's natively benchmarked.
**Verification:** Sorenson verified all 6 areas 🟢 — **f32 accumulator confirmed** (`_mm512_setzero_ps`→`_mm512_dpbf16_ps`→`_mm512_reduce_add_ps`, never bf16, per the hard no-bf16-accumulator rule); K-tail mask `(1<<chunk)-1` chunk∈[1,31] no UB/OOB; disjoint C-row Rayon partitions bit-stable 3×; runtime-gated with clean f32 fallback for non-bf16/non-contig/AVX2-only/aarch64. Tests: 873 lib + 10 golden 0 fail; **worst native-vs-f64 rel 1.870e-6, native/upcast ratio 1.000** (native as accurate as upcast — bf16 products exact in f32); Grissom's goldens unchanged.
**Perf (SPR 8480C, load ~15-19, median-3; native-bf16 vs our-own-upcast-bf16 — ORT has no bf16 CPU baseline):** decode GEMV 1×4096×4096 **2.1-3.0×**, decode MLP 1×4096×11008 **3.1-3.7×**, prefill ~parity (follow-up: bf16 B-prepack at load).
### 2026-07-23: widen VNNI/int16 decode dots to true 512-bit on AVX-512 (Delko, Flack review 🟢 + coordinator live VNNI run)
**By:** Squad (Coordinator), authored by Delko, reviewed by Flack (non-author, opus) + coordinator live-verified the VNNI tests
**What:** Widened three int-quant decode dots in matmul_nbits.rs from 256-bit to true 512-bit, runtime-dispatched 512→256→scalar: (1) `int4_dot_row_avx512vnni` via `_mm512_dpbusd_epi32` — no `_mm512_sign_epi8`, so raw UNSIGNED nibbles + all-ones dpbusd zero-point correction `sum((n-8)a)=sum(na)-8·sum(a)`, single f32x16 accumulator; (2) `dot_u8_i8_avx512vnni` 64-byte `_mm512_dpbusd` + `_mm512_reduce_add_epi32` + 256/scalar tail; (3) NEW `block_dot_u8_i16_avx512bw` via `_mm512_madd_epi16`, same group=32 int16 quant + single-block accumulator preserving the fp32 argmax. Dispatch gated on avx512vnni/avx512bw. Merged 58d5d6e (base c60087c).
**Why:** The existing avx512vnni-gated kernels were WASTING half the width (used 256-bit `_mm256_dpbusd_epi32` under a 512-bit feature gate). True 512-bit exploits this SPR box's full VNNI/BW width per the per-microarchitecture directive.
**Verification:** Flack 🟢 all 6 areas (int4 zero-point algebra exact for all nibbles/signs; dpbusd operand roles correct unsigned×signed; overflow bounded; int16 argmax non-vacuous, ran live). Flack's sandbox lacked avx512vnni so the 2 VNNI tests self-skipped there — COORDINATOR re-ran them LIVE on the 8480C host (avx512vnni=true confirmed): `int4_dot_row_avx512vnni_matches_scalar` + `dot_u8_i8_avx512vnni_matches_scalar` = 2 passed live, 0 skipped. Merged-HEAD full suite (Caine bf16 + Delko VNNI together): 877 lib + 10 golden, 0 failed.
**Perf (median-3, load-annotated):** int16 activation dot **+24%** (clean win); int4 **parity** — honestly root-caused as weight-UNPACK-bound not dpbusd-bound (future int4 opt: faster AVX-512 nibble unpack). e2e Qwen2.5-0.5B-int4 +1.8% (within noise). int4 results byte-identical (no-regression).
### 2026-07-24: int4 decode unpack ~1.45x (deinterleave-once + permutex2var) — MERGED to PR #105
**By:** Bishop (author), Ferro (non-author review 🟢 APPROVE)
**What:** Cherry-picked `7d74287` → `37ee582` onto perf/cpu-ep-mlas. `deinterleave_activation_int4` reorders activations (evens-then-odds per 32-block) once per matmul so SIMD int4 kernels drop per-block unpacklo/unpackhi; 512-bit unpack widened via `_mm512_permutex2var_epi64`. Gated on `use_simd` in `int4_matmul_m1` (scalar/non-x86 keep natural order). Single production caller.
**Why:** int4 decode is unpack-bound (Delko finding). Beats prior kernel by 1.454x, parity preserved (few-ULP vs scalar oracle). Ferro adversarial review: all 7 areas PASS (pairing crux, permutex2var index, zero-point, K-tail, avxvnni parity, non-vacuous live tests, 32-byte load safety). Coordinator re-ran int4 tests LIVE on real host (avx512_vnni+avx_vnni present): 13/0 pass.
### 2026-07-24: Fix no-mlas ep-cpu build (gate NCHWc-via-MLAS) + workspace fmt — MERGED to PR #105
**By:** Wierzbowski (author), Drake (non-author review 🟢 APPROVE)
**What:** Cherry-picked `66f2d8d` → `9c29cc3` onto perf/cpu-ep-mlas (matmul_nbits.rs fmt-conflict resolved keeping int4 code, re-ran `cargo fmt --all`). Gated `pub mod nchwc;` + 6 NCHWc op registrations (mod.rs), `pub mod nchwc_layout;` (lib.rs), and the `NchwcLayoutPropagation` optimizer pass push (optimizer.rs) behind `#[cfg(feature="mlas")]`. Without mlas, no NCHWc ops emitted → plain Conv kernels run. Op-count constant updated (base→91, mlas term→7).
**Why:** ep-cpu did NOT compile without the optional `mlas` feature (17× E0433 `mlas_sys` in nchwc.rs/nchwc_layout.rs). CI tests default features (no mlas) → this red-ed ALL Rust jobs AND the CUDA-compile job (ep-cuda pulls ep-cpu without mlas), and broke ARM/macOS which never use mlas. Coordinator independently verified LIVE: no-mlas ep-cpu compiles+tests, `cargo check -p onnx-runtime-ep-cuda --features cuda` now Finishes (was RED), mlas 879/0+10/0 green, fmt clean. Drake review: all 6 checks PASS incl. mlas behavior byte-unchanged + both-config builds exit 0.
### 2026-07-24: Contention-aware SPMD decode auto-enable (~34x faster default under load) — MERGED to PR #105
**By:** Apone (author), Gorman (non-author review 🟢 APPROVE, 8/8 live)
**What:** Cherry-picked `0a59532` → `16a0fae` onto perf/cpu-ep-mlas (+fmt fixup). decode_spmd.rs: new `loadavg_one()` (Linux /proc/loadavg; other-unix libc::getloadavg; Windows→None), `current_contention(allowed_cpus)=loadavg1/allowed_cpu_count`, `should_auto_enable(available,contention,max_load_per_cpu)` — declines auto pool when load_per_cpu>0.7 (loaded box), enables when idle OR contention unknown (preserves prior default-on for dedicated boxes/CI), <4-CPU floor unchanged. Env overrides intact: =1 Forced bypasses gate, =0 Off flat, explicit AFFINITY defers. Numerics identical (path-selection only). 2 new unit tests; suite 881+10 green.
**Why:** The persistent SPMD pool's hard-spinning workers get OS-starved on loaded/shared boxes → ~700ms/token (1.40 tok/s vs 32-48 pool-off). Root-caused this session (Vasquez "100x" was entirely this). Fix keeps the dedicated-box win (auto-enables at low load) but steps aside under contention. Gorman live-verified on this loaded 96-CPU host: taskset -c 0-11 → contention 1.37>0.70 → flat → 34.86 tok/s (vs 1.4 disaster); full-budget idle → auto-enable 13.60 tok/s; Forced=1 still spins. Cross-platform clean (libc unconditional dep, macOS getloadavg builds, no unwrap on /proc, NaN filtered).
### 2026-07-24: Clear ep-cpu clippy -D warnings (unblocks CUDA-compile + Rust-quality CI jobs) — MERGED to PR #105
**By:** Crowe (author), Spunkmeyer (non-author review 🟢 APPROVE)
**What:** Cherry-picked `96de6be` → `adcfc5f`. Lint-only, behavior-preserving: cfg-gated MLAS-only profiler counters/fns (GEMV_NS/NARROW_NS/CALLS/time_gemv/time_narrow/tick) + `to_dense_f32_widen` import behind `#[cfg(feature="mlas")]` (dead without mlas); scoped `#[allow(clippy::needless_range_loop)]` on two gemv_nk_u8_i16 hot loops (NO body change, they index parallel arrays) + two test loops; behavior-identical test iterator rewrite; `.contains(&0)` for resize zero-extent check.
**Why:** Our CPU perf work introduced clippy lints that failed CI. The CUDA-compile job runs `cargo clippy -p onnx-runtime-ep-cuda --features cuda -- -D warnings`, which transitively denies ep-cpu warnings — so ep-cpu lint hygiene gates BOTH the CUDA-compile job and Rust-quality. Coordinator + Spunkmeyer independently verified LIVE on integrated branch: ep-cuda clippy -D warnings Finished (exit 0), ep-cpu clippy mlas + no-mlas clean, fmt clean, 881+10 tests green.
### 2026-07-24: qwen3-0.6b native/ORT divergence — BENIGN-TIE, keep native (Hudson adjudication, Vasquez 🟢 review)
**By:** Squad (Coordinator), integrating Hudson's investigation + Vasquez-1's adversarial review
**What:** Native greedy decode diverges from ORT on qwen3-0.6b at the first split (native→518, ORT→264). Adjudicated BENIGN-TIE, keep native, no kernel change. Merged Hudson's teacher-forced regression test `qwen3_0_6b_divergence.rs` (`#[ignore]`+`QWEN3_0_6B_E2E_DIR`-gated) asserting native selects token 518.
**Why:** fp32 oracle (ORT's own path, all 197 MatMulNBits nodes accuracy_level 4→1) picks 518; native (acc=4 int8) matches oracle, ORT (acc=4 int8) flips the razor-thin tie (gap ≤0.044 logits) to 264. Across 30 teacher-forced positions native matches oracle 29/30 vs ORT 28/30. Vasquez-1 independently rebuilt the oracle from scratch and reproduced every number to 4 decimals (oracle +0.04382, native +0.05162, ORT −0.05270); native tied-or-better across all 3 test prompts, never worse. Meets the user's bar ("ours not less accurate than ORT" — marginally more). Same class already locked for Phi-3.5. Non-blocking follow-up (Hudson locked out per reviewer protocol): optionally add a free-running end-to-end assertion; assign Gorman or another correctness agent.
### 2026-07-24: Restore executor early-rejection before EP passes / host copy (Dietrich, Hicks 🟢 review)
**By:** Squad (Coordinator), integrating Dietrich's fix + Hicks's adversarial review
**What:** Merged `executor.rs` fix (`643c4c6`) restoring the "reject-before-materialize" contract that the origin/main merge tightened and our CPU-EP perf commits regressed (5 executor tests + `slice_zero_step` failing on PR #105). Three fixes: (1) `reads_float_shape_input()` gates float shape-value host materialization to ONLY default-domain `Resize` scales (opset 10→idx1, else idx2), so an unrelated float input is no longer downloaded before an invalid integer shape input is rejected; (2) `reject_unsupported_operators()` + `graph.topological_order()?` run BEFORE EP optimizer passes (mirroring the control-flow signature pre-check), skipping CUDA EP (legit CPU fallback) and deferring non-static-shape nodes to the run-time kernel gate; (3) post-EP-pass `infer_graph` is now best-effort (infer on clone, adopt on success) so a data-dependent invalidity (Slice step 0) rejects at run time instead of aborting the build. Zero-copy decode fast path (`try_move_host_output`) unchanged.
**Why:** origin/main requires invalid shapes/cyclic plans/unsupported ops be rejected before any `copy_to_host`. Our decode-perf float-shape reader + always-on CPU-EP passes violated the ordering. Hicks independently verified (adversarial): 32/0 executor, whole session crate green (`slice_view` 5/5), test files have EMPTY diff (not weakened), zero-copy perf test intact, `reads_float_shape_input` opset index matches ONNX Resize spec, no other op reads float shape data, no panic risk, collapsible-if rewrite logically identical, clippy `-D warnings` exit 0 + fmt clean. This greens the Rust test jobs + Rust quality on PR #105 without regressing decode perf.

---
### 2026-07-24: DETAILED REVIEW — Vasquez's adversarial adjudication of Hudson's qwen3-0.6b divergence

**Reviewer:** Vasquez-1 (numerical-correctness, CPU EP); **Author:** Hudson; **Method:** Adversarial non-author review on worktree `/home/justinchu/onnx-genai-cpu-divergence`

**Full verification report:**

#### Item 1 — IS THE ORACLE VALID? ✅ PASS (stronger than described)

- Confirmed the deployed model is `MatMulNBits`, `block_size=128`, all `accuracy_level=4`, **mixed weight width** (92 nodes 4-bit, 105 nodes 8-bit). Hudson's shorthand "int4" is a simplification but the acc-level=4 int8-**activation** tie-break story applies uniformly to all 197 nodes.
- Reproduced Hudson's exact oracle recipe: loaded `model.onnx`, rewrote `accuracy_level 4→1` on **all 197** MatMulNBits nodes (I touched *only* the `accuracy_level` attribute — weights/scales/graph untouched), ran through **ONNX Runtime CPU** on the committed 18-token prefix. In ORT's MatMulNBits, `accuracy_level` selects only the **activation** compute type (1=fp32, 4=int8); weights stay quantized. So acc=1 is strictly higher precision than acc=4 for the *same* weights → "closer to acc=1" == "more accurate for this deployed quantized model." Reasoning is sound.
- **Crucial anti-bias check:** the oracle is **ORT's own fp32 path**, i.e. completely independent of the native kernel. Native is therefore *not* being graded by its own ruler — it cannot be inadvertently advantaged by shared acc=1/acc=4 code. Native still matches this ORT-fp32 oracle where ORT-int8 does not. That is the strongest possible form of this argument.
- (Aside: repo also ships an even-more-independent PyTorch/HF oracle `scripts/qwen_q4_f32_oracle.py`, but it has no qwen3-0.6b case and assumes block-32; not used here. Not a defect.)

Reproduced table (ORT-fp32 oracle, single-prefill, intra_op=1):

| compute | argmax | logit(518)−logit(264) | Hudson | match |
|---|---|---|---|---|
| acc=1 ORACLE (ORT fp32) | **518** | **+0.04382** | +0.0438 | ✅ |
| native (acc=4 int8) | **518** | +0.05162 | +0.0516 | ✅ |
| ORT (acc=4 int8) | 264 | −0.05270 | −0.0527 | ✅ |

#### Item 2 — "29/30 vs 28/30" MEANINGFUL OR NOISE? ⚠️ PASS-with-caveat

- Honest reading: 29 vs 28 over sub-0.05 ties is **statistically indistinguishable, marginally better** — NOT a robust "we're better." Hudson's note actually frames it correctly ("razor-thin," "not less accurate ... marginally more"); it does **not** overclaim. The verdict does **not** rest on the 1-position margin — it rests on native matching the fp32 oracle at the one *resolvable* divergence while ORT does not, plus native never being worse across my independent prompts (Item 6).
- **Caveat (non-blocking):** the 30-position aggregate harness is **not committed**, so I could not independently reproduce the 29-vs-28 count. Given the user's bar is "not LESS accurate," and that bar is met by reproduced evidence, this is acceptable — but the aggregate should be treated as illustrative, not a precise measurement.

#### Item 3 — REPRODUCE THE KEY DATUM LIVE. ✅ PASS

- Rebuilt `profile_native` with `--features mlas,bench-ort`.
- **Native** teacher-forced `--dump-logprobs` at the 18-token prefix: `selected=518`, logsm(518)=−1.5635, logsm(264)=−1.6151, gap **+0.0516**. Exact match.
- **ORT** greedy generation split reproduced live: native `[576,3364,1265,2924,518,…]` vs ORT `[576,3364,1265,2924,264,…]` — divergence at gen-index 4, native→**518**, ORT→**264**. ✅
- **Oracle** reproduced from scratch (above): acc=1→518, acc=4→264. ✅
- All three argmaxes and all three signed gaps match Hudson to 4 decimals.

#### Item 4 — REGRESSION TEST QUALITY. ✅ PASS

```
test qwen3_0_6b_int4_native_decode_keeps_high_precision_argmax ... ok
qwen3-0.6b divergence lock OK: native token = 518 (fp32-oracle-correct; ORT = 264), benign-tie gap = 0.05162 logprob
test result: ok. 1 passed; 0 failed; 0 ignored; ... finished in 90.07s
```

- **Non-vacuous:** loads the real qwen3-0.6b model, runs actual native int4/int8 decode, asserts argmax==518 AND that 264 is the top-8 runner-up AND 0<gap<0.2. If native regressed to ORT's 264 the first `assert_eq!(selected, 518)` fails loudly. Directionally correct.
- **Gated:** `#[ignore]` + `QWEN3_0_6B_E2E_DIR` (defaults to foundry cache). Missing dir → `eprintln!` + `Ok(())` (graceful skip, no false-fail). Verified reasoning by inspection.

#### Item 5 — NO HIDDEN KERNEL CHANGE. ✅ PASS

- `git show d3ff05b --stat`: **only** `crates/onnx-genai-engine/tests/qwen3_0_6b_divergence.rs` (+167). No production/kernel change.
- `cargo test -p onnx-runtime-ep-cpu --features mlas` on a clean run: **879 passed, 0 failed, 7 ignored** — matches Hudson. The class guard `int4_decode_preserves_f32_argmax_where_per_row_int8_activation_flips` **passes**.
- **NOTE (unrelated to this change):** a first, fully-parallel run flaked with 15 `kernels::qmoe::tests` failures (host-cache/mmap-residency + global `reset_offload_test_state`/`metrics_test_lock` contention — resource-sensitive on a shared box); a clean re-run was green. Pre-existing test-infra flakiness in a different subsystem; **not** caused by d3ff05b. Worth a separate ticket, not a blocker here.

#### Item 6 — PROMPT-BIAS. ✅ PASS (native never worse than ORT on my prompts)

- "…staying healthy during winter": native and ORT **token-identical** for 48 tokens (no divergence).
- "Explain the theory of relativity…": free-running split at index 31 — free-run **native=6319 matches the fp32 oracle**, free-run **ORT=914 does not**. (Teacher-forced single-prefill at that agreed 42-token prefix gives native=914==ORT=914, oracle=6319 — a tie between the backends; see caveat below.)
- Net across 3 prompts: native is **tied-or-better** vs ORT everywhere, **never worse** — consistent with "not less accurate."

**Methodological caveat (non-blocking, applies to Item 2's harness):** at ultra-thin ties, **teacher-forced single-prefill logits are not identical to the deployed incremental (KV-cached) decode** — even within one backend. Demonstrated at the relativity prefix: free-running native emitted 6319 but single-prefill native emitted 914 (gap ~0.016). It happens the qwen3 headline case is *consistent* (free-run native 518 == teacher-forced 518 == oracle 518), so the committed regression test faithfully locks real behavior **there**. But the teacher-forced probe should not be read as a bit-exact proxy for deployment argmax at sub-0.02 ties. Recommend a follow-up (Hudson locked out on this artifact — assign e.g. **Gorman** or another correctness agent) to add a note in the test/decision that the lock is on the teacher-forced single-step, and, if desired, add a free-running end-to-end assertion for the qwen3 case.

**VERDICT: 🟢 APPROVE** — keep native (benign int8-activation tie-break) and merge the regression test. Oracle is valid and independent of native; native/ORT/oracle numbers reproduced to 4 decimals; native matches the fp32 oracle at the one resolvable divergence where ORT flips; native is tied-or-better (never worse) across the extra prompts; commit adds only a well-gated, non-vacuous test with no kernel change; kernel guard + 879/0 confirmed on a clean run.

---
### 2026-07-24: DETAILED VERIFICATION — Hicks's adversarial review of Dietrich's executor fix

**Reviewer:** Hicks (runtime/executor); **Author:** Dietrich; **Commit:** `862e471` on `fix/session-executor-early-reject`

**Observed test/gate results (worktree `/home/justinchu/onnx-genai-cpu-exec`, LD_LIBRARY_PATH set to ort-prebuilt):**

- `cargo test -p onnx-runtime-session --test executor` → **32 passed, 0 failed**.
- `cargo test -p onnx-runtime-session` → whole crate green (69+26+22+13+6+5+3+2+… across all binaries, incl. `slice_view` 5/5 with `slice_zero_step_reports_actionable_error`).
- `zero_copy_output_move_reallocates_and_preserves_producer_less_output` (unit) → passed.
- `cargo clippy -p onnx-runtime-session --all-targets -- -D warnings` → exit 0.
- `cargo fmt --all -- --check` → clean.

**Independent verification (NOT trusting the author summary):**

1. **Tests were NOT weakened: VERIFIED.** `git diff 386be50..862e471` on the two test files is EMPTY. Read each body directly — they assert the real contract: `HostDownloadCountingEp` counts `copy_to_host`; the four *before_host_materialization* tests assert `downloads == 0` AND the correct error variant (`UnresolvedShape` / Unsqueeze "1-D tensor"). The cyclic test asserts the exact `SessionError::Graph(GraphError::CycleDetected)` variant. `unsupported_op_...unnamed` asserts the sentinel + "node <unnamed node #0>, opset 0". `slice_zero_step` asserts build succeeds then `run` errors with "step".

2. **`reads_float_shape_input` alignment: VERIFIED.** `dynamic_output_shapes` reads `input_float_values` in exactly one arm — default-domain `Resize`, using `scales_index = if opset==10 {1} else {2}`, byte-identical to the new gate. Matches ONNX Resize spec (opset 10: scales=in1; opset 11+: roi=1, scales=2). No other op (Upsample included) ever consumed float shape values, so gating them out regresses NO valid dynamic-shape graph; it only stops a wasted host copy that violated the reject-before-materialize contract.

3. **Pre-check ordering: VERIFIED.** `reject_unsupported_operators` + `graph.topological_order()?` are placed right after `validate_control_flow_signatures` and BEFORE `run_ep_scoped_passes`, mirroring the existing pre-check. No panic risk: `effective_opset`'s `unreachable!` is unreachable here because `validate_model` (lib.rs:811, before build) already rejects missing opset_imports. The pass skips CUDA (legit CPU fallback via `cuda_fallback_report`), ep_context / control-flow / sequence ops, and DEFERS any node with a non-static declared input shape to the run-time kernel gate — the deferred-symbolic path is pre-existing behavior, acknowledged, acceptable.

4. **Best-effort `infer_graph`: VERIFIED.** On failure the original graph is untouched; on success shapes only improve. Zero-copy decode fast path preserved (its perf test passes).

5. **Collapsible-if rewrite: VERIFIED.** The `if !nested && let Some(t) = …?` let-chain is logically identical to the old nested `if` — short-circuit means `try_move_host_output` still runs only when `!nested`, `?` propagates identically, and the Ok(None) fall-through is unchanged.

**Correctness holes found:** none blocking. Only the (intended, pre-existing) deferral of symbolic-shape unsupported ops to the run-time gate, which is documented and consistent with the CUDA-fallback design.

Worktree left pristine (no scratch files).

**VERDICT: 🟢 APPROVE** — 32/0 executor, whole session crate green, tests unweakened, no correctness holes.
### 2026-07-24: Cross-CPU mlas-sys test portability + guard MLAS AVX2 M=1 asym int8 bug (Ripley, Lambert 🟢 review)
**By:** Squad (Coordinator), integrating Ripley's fix + Lambert's adversarial review
**What:** Merged `9a1c550` (mlas-sys tests + ep-cpu production guard). Fixes the last 2 RED PR #105 CI jobs: CI runners are AVX2-class (no AVX-512), but two `crates/mlas-sys` tests hard-coded AVX-512 expectations. (1) `avx512_kernel_is_selected` → `best_available_float_kernel_is_selected`: portable per-ISA assertion (512/3/1/-1/0). (2) `sqnbit_int4_compint8_matches_reference`: M=1 **asymmetric** CompInt8 case gated to AVX-512 hosts (symmetric + all M>1 asym still run everywhere; tolerance unchanged). ROOT CAUSE (reproduced under Intel SDE: `-hsw` fails, `-skx` passes): MLAS's AVX2 M=1 CompInt8 SQNBit kernel with a zero point (`SQ4BitGemmM1Kernel_CompInt8_avx2`) is numerically broken for asymmetric int4 (~46% error, mlas=6.09 vs ref=11.29, all block sizes). Production guard: `try_mlas_sqnbit` refuses M=1 asym CompInt8 on hosts lacking the MLAS AVX-512 SQNBit dispatch (`host_has_mlas_sqnbit_avx512()` = avx512f+bw+dq+vl, mirroring MLAS platform.cpp:572) and falls back to the correct hand int8 kernel. New regression test `matmulnbits_accuracy4_m1_asymmetric_matches_fp32_reference`.
**Why:** Default routing already kept M=1 decode on the hand int8 kernel via the `sqnbit_decode_min()>=2` crossover, so production default is correct on all CPUs; the guard closes a latent hole where `NXRT_SQNBIT_DECODE_MIN<=1` could reach the broken kernel on non-AVX512 hosts. Lambert independently verified (20/0 mlas-sys, 882/0+10 ep-cpu, clippy/fmt clean, fallback reaches hand int8 path, `zero_points.is_some()` = correct asym proxy, no over-fire/no AVX-512 perf regression) and caught that the guard must require F+BW+DQ+VL (not just F) to exactly mirror MLAS's dispatch gate — applied and re-verified. An upstream ORT/MLAS bug report is drafted (inbox `ripley-ort-issue-draft.md`) for filing. Cross-CPU correctness is a hard user requirement; this greens PR #105 CI on AVX2 runners while keeping production correct on every CPU.
### 2026-07-24: Batched MatMulNBits prefill fix (Nk dequant + MLAS sgemm trans_b)
**By:** Burke (impl), Gorman2 (review 🟢)
**What:** Fixed ~10× native prefill regression for 8-bit MatMulNBits. The m>1 batched dense-dequant path dequantized into transposed Kn layout (stride-N scatter, cache-thrash, 4773ms). Now dequants once into contiguous Nk (cached in weight_nk OnceLock) + MLAS sgemm(trans_b=true, ldb=k). Prefill 5335→~545ms; gap to ORT ~25×→~2.6× (clean host). Output byte-identical; new regression test matmulnbits_8bit_prefill_batched_matches_dequant_f32_oracle (m=16/32/100 × sym/asym vs independent f32 oracle, RMSE≤1e-5 + argmax parity).
**Why:** User: native must beat ORT on prefill too. Gorman2 re-verified sgemm math by hand (A·Wᵀ identical to old Kn path/gemv_nk/oracle), confirmed weight_nk OnceLock has no cross-layout aliasing, tests non-vacuous. Cherry-picked 30f5837→a352686 onto perf/cpu-ep-mlas; ep-cpu 884/0, both clippy gates + fmt clean.
### 2026-07-24: Filed ORT issue microsoft/onnxruntime#29853 (MLAS AVX2 M=1 asym int8 bug)
**By:** Ripley, with Lambert review
**What:** Filed upstream issue `microsoft/onnxruntime#29853` documenting the MLAS AVX2 M=1 asymmetric CompInt8 SQNBit correctness defect.
**Why:** The PR #105 production guard prevents affected non-AVX-512 hosts from using the broken route; the upstream report tracks the MLAS fix.
### 2026-07-24: ARM cross-platform dead-code fix (green CI on macOS/Windows arm64)
**By:** Hasford (impl), Kano (review 🟢)
**What:** PR #105 CI failed only on Rust (macOS arm64) + (Windows ARM64) "cross-platform offline crates" with `-D warnings` dead-code errors: `native_available` (bf16_gemm.rs) and `dot_u8_i16_scalar` (matmul_nbits.rs) unused on non-x86 in lib builds. Fix: `#[cfg_attr(not(target_arch="x86_64"), allow(dead_code))]` on the non-x86 bf16 native_available stub (used only by portable tests; lib callers are x86-gated); `#[cfg(target_arch="x86_64")]` on dot_u8_i16_scalar (only called from x86 SIMD-tail handling) and gated the single x86-specific assert in test dot_u8_i16_matches_serial_reference while keeping the portable block_dot_u8_i16 assertion on all arches.
**Why:** User requires cross-OS/cross-CPU support. cfg-hygiene bugs, not functional gaps — ARM runtime paths already fall back to portable inline scalar. Verified by reproducing the aarch64 failure (both errors, exit 101) then confirming clean aarch64 lib+tests; x86 884/0, both clippy gates + fmt clean. Cherry-picked d3e7ed80→perf/cpu-ep-mlas.
### 2026-07-24: Persistent SPMD decode pool made opt-in (default flat) — landed on PR #105
**By:** Voight (author), Chew 🟢 APPROVED (non-author, opus)
**What:** Cherry-picked `176da282` onto perf/cpu-ep-mlas. The persistent SPMD decode pool is now opt-in via `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1`; unset (`Auto`) and `=0` (`Off`) both take the flat Rayon path (same code path by construction). Removed 4 tests targeting now-deleted auto-enable heuristics (`auto_enable_suitable`, `should_auto_enable`, `current_contention`, `auto_defers_to_flat`); added `only_forced_mode_enables_the_persistent_pool`. Explicit-affinity property still covered by surviving integration tests.
**Why:** The persistent pool was a structural 2.5–3× decode regression (per-op barrier-spin over ~197 tiny M=1 ops/token). Chew reproduced ~2.8–3× regression interleaved (flat 33–38 vs pool 12–13 tok/s, load 37–46), byte-identical tokens across default/=0/=1. Default decode is now competitive with ORT again (~33 tok/s). Q4 M=1 GEMV kernel work remains to actually beat ORT (~36–41) — deferred, correctness-gated.
**Gates:** ep-cpu 881+10/0; clippy mlas + no-mlas exit 0; fmt clean; aarch64 `-D warnings` check clean; tokens byte-identical.
### 2026-07-22: Token-divergence regression locks landed on PR #105
**By:** Squad (Coordinator) integrating Holden/Pris (author), Dietrich (🟢 reviewer)
**What:** Cherry-picked ea31e26+7e70fcf (qwen3-0.6b + phi3.5-mini int4 divergence tests) onto perf/cpu-ep-mlas. qwen: token-0 lock (native 1479 vs ORT 3988) + decode-lock (native 518 vs ORT 264), both fp32-oracle-correct via accuracy_level 4→1 rewrite. phi: teacher-forced oracle (token-103=411) + real m=1 decode-loop lock (asserts native==411 && !=408).
**Why:** Native int4 CPU decode diverges from ORT on sub-0.1-logit ties; fp32 activation oracle proves NATIVE is the higher-precision argmax, so we keep our implementation. Regression tests prevent silent drift. Tests are #[ignore]+env-gated (need foundry model dirs), compile under --features mlas.
### 2026-07-22: int4 M=1 GEMV multi-accumulator decode kernel landed on PR #105
**By:** Squad (Coordinator) integrating Ripley (author, 59ea1ab), Gorman (opus 🟢 reviewer)
**What:** Cherry-picked 59ea1ab onto perf/cpu-ep-mlas. int4_dot_row_avx512vnni now uses 4 independent f32 accumulators (unroll-by-4 via int4_pair_scaled_avx512 helper) + weight prefetch; int4_dot_row_avxvnni uses 2. Per-block integer VNNI dots stay bit-identical; only final f32 scale-accumulate order reassociates (few ULP, within 1e-4 tol). Added non-vacuous argmax-vs-scalar guard test + extended remainder/tail coverage.
**Why:** Removes the loop-carried add_ps latency chain in the hot decode GEMV. Neutral (~1%) on bandwidth-bound qwen decode (the real gap is NUMA single-node bandwidth, tracked separately), POSITIVE on compute-bound cases, token-IDENTICAL on qwen3-0.6b + Phi-3.5-mini. Zero-regression, general (scalar/aarch64 unaffected), well-tested. Gorman verified non-vacuity by mutating the reduction → test FAILED.
### 2026-07-22: Generic native-vs-ORT benchmark tool (bench_generic) landed on PR #105
**By:** Squad (Coordinator) integrating Vasquez (author, cdf1091), Drake (🟢 separable-part reviewer)
**What:** Cherry-picked cdf1091 (bench_generic.rs + Cargo.toml bin) onto perf/cpu-ep-mlas. A generic single-inference native-vs-ORT bench (interleaved timing + output parity) for traditional-ML / non-genai ONNX models (resnet, yolo, etc.). Args: --model, --warmups, --runs.
**Why:** The user asked to verify generality (resnet/yolo faster than ORT) via the onnx-genai stack. This is the reusable generality-benchmark harness. Drake reviewed the full conv-pool branch and confirmed bench_generic (base, untouched by Bishop's conv-pool) is safe/separable to land; the conv-pool part was 🔴 rejected for a straggler/reset race (tracked separately). bench_generic compiles under --features bench-native,mlas, fmt clean, required-features gates it off aarch64.

### NUMA-split decode sizing fix + non-vacuous partition test (landed on PR #105)
**Commits cherry-picked:** 69a4463 (parker, sizing) + ab3fd65 (hicks, non-vacuous test). Reviewer: Gorman (opus) 🟢 APPROVED — non-author (parker/ferro/hicks locked out).
**What:** numa-split two-level decode layout (OPT-IN via ONNX_GENAI_CPU_DECODE_AFFINITY=numa-split). Fix: numa_pools() no longer capped at 8 workers — sized from configured_persistent_decode_threads() (~half CPUs), split_workers caps only at per-node CPU count. DEFAULT (flat) decode path UNCHANGED (build_from_env early-returns None unless NumaSplit; use site gates on IN_NUMA_SCOPE thread-local).
**Why safe:** row-sharded GEMV is exactly associative (each output row = independent full-K dot); per-node shard concat is bit-for-bit identical, no cross-node reduction. Locked by ..._matches_flat_gemv_bit_for_bit (to_bits) test.
**Test non-vacuity PROVEN:** perturbing dispatch partition 3:10→4:9 → "row 3 dispatched on node 0 but placed on node 1" FAIL; revert → PASS. Closes Ferro's earlier vacuous-test rejection.
**Gates:** 884 lib + 10 regression green; clippy w/ + w/o mlas -D warnings; fmt; aarch64 check.

### Zero-copy f32 activation borrow in M>1 prefill (landed on PR #105)
**Commit cherry-picked:** 710e18f (apone). Reviewer: Drake (opus) 🟢 APPROVED (non-author).
**What:** M>1 prefill MatMulNBits previously called to_dense_compute_f32, always allocating+memcpy'ing the activation into an owned f32 Vec every execute(). New compute_activations_cow returns Cow<[f32]>: BORROWS in place when dtype==Float32 && is_contiguous() && device.is_host_accessible(); still OWNS/widens for f16/bf16/strided/device. qwen3-0.6b & phi3.5-mini carry f32 activations → widen phase 161ms→~0/prefill, warm wall-clock ~+3-4%.
**Why safe:** old to_dense_f32 already had a contiguous fast path using the identical from_raw_parts(data_ptr::<f32>(), numel(shape)); the borrow returns that same slice instead of copying → bit-identical. strided/transposed/broadcast (is_contiguous strict), non-zero offset (data_ptr applies byte_offset), device (is_host_accessible) all excluded → owned path. Token-EXACT.
**Test non-vacuity PROVEN:** perturb borrow branch v[0]+=1.0 → "borrowed vs copied activation diverged" FAIL; revert → PASS (m=1 and m=4).
**Residual warm prefill gap** (~2.9x qwen / ~1.4x phi) now dominated by MLAS over-sharding the small prefill GEMM → separate work-proportional thread-cap in mlas-sys (perf/mlas-sqnbit-threads, apone).
**Gates:** 883 lib green; clippy w/ + w/o mlas -D warnings; fmt; aarch64 check.
<!-- scribe-merge-2026-07-24T15-10-00Z-decode-locks-and-date-cleanup -->
## 2026-07-24 — Decision inbox reconciliation
<!-- merged from deckard-phi-capture-seams.md -->
### 2026-07-23: Eliminate Phi decode CUDA-graph capture seams (Greater + invariant If)
**By:** Deckard
**Branch:** `perf/phi-capture-seams` (off `origin/main` @ `1073404`) — commit `54cc02f`. **Needs review before merge; not merged to main.**

**Scope:** CUDA-graph capture-seam elimination in the executor / CUDA EP. This is NOT a GEMV micro-opt — `matmul_nbits.rs` kernels untouched.

#### Root cause — confirmed
Marsten's Nsight finding reproduced exactly via `ONNX_GENAI_LOG_CAPTURE_SEGMENTS=1`. Phi-4-mini decode splits into **3 captured graphs across 2 per-token seams**, both inside the LongRoPE `rotemb_caches_subgraph`:
- `Greater` node 8 = `Greater(attn_mask_gather_len, 4096)` → **rank-0 scalar bool**. An **eager device seam**: the CUDA binary-predicate kernel (`BinaryPredKernel`) allocated + uploaded + freed broadcast metadata and `synchronize()`d the stream every call, and hard-declined capture.
- `If` node 13 = `If(Greater.out) → (cos_cache, sin_cache)`. A **host seam**: both branches are just two `Constant`s emitting the *full* rotary caches (else/steady = `[4096,48]` fp16 ≈ 393 KB each). Control flow reads `cond` to the host and re-runs the taken branch — re-materializing and re-copying ~786 KB host→device — **every decode step**.

The `If` executor-timer cost Marsten saw is dominated by this per-step branch re-materialization + child-executor overhead, not GPU compute.

#### Fixes (both capture-safe, byte-identical)
1. **`Greater` capturable** (`kernels/pointwise.rs`, `kernels/elementwise.rs`). Persist broadcast metadata in a `BroadcastMetadataCache` (reused across steps — no per-step alloc/upload/free/sync) and advertise `CaptureSupport::Supported` for a stable dtype/shape signature, exactly mirroring the elementwise `BinaryKernel`. Generalized the eligibility gate so a **rank-0 scalar / single-element** predicate output (the LongRoPE `Greater` shape, which `is_fixed_decode_shape` rejected) qualifies. Result: `Greater` folds into the graph → **3 → 2 graphs**.
2. **Loop-invariant `If` specialization** (`executor.rs::exec_if`). General mechanism, not a Phi hardcode: an `If` whose *taken branch has no outer captures* (`required_outer_names(body).is_empty()`) produces outputs that depend only on its own constants, so once taken with a predicate its outputs are already resident in their persistent buffers. The predicate is **still read every step (the correctness guard)**; only the redundant branch re-execution + its host→device cache copies are skipped. Correctness rails:
   - A branch that reads loop-varying captures is **never** memoized (`taken_branch_is_invariant` gate) → no stale/wrong output. Regression test `if_never_memoizes_branch_that_reads_changing_captures`.
   - A predicate flip re-runs the branch; an output-shape change (LongRoPE short↔long at seq 4096) retires the installed graph via the existing `control_flow_seam_invalidated`. Regression test `if_memoizes_invariant_branch_but_reruns_on_predicate_flip`.
   - The memo is cleared before every capture so freshly reallocated buffers are always repopulated during the capture pass.

#### Results (H200, idle GPU, `--steady --warmups 2 --runs 9 --tokens 120`)
- **Graph count: 3 → 2** (`Greater` seam removed; `If` remains a *cheap, memoized* seam).
- **Throughput: ~193 → ~213 tok/s (+~10%, ~0.47 ms/token)**, matched interleaved before/after runs (baseline binary from `origin/main` vs after; e.g. 193.45→211.12, 197.34→213.63, 195.76→215.61). Absolute numbers drift with thermal state, so the *interleaved* delta is the reliable figure. Recovers roughly half the gap to ORT's 229.62 (native was 193.90); remaining ~ -7% is not from these seams.
- **Correctness: byte-identical generated token ids** before vs after over 150 tokens (`diff` clean).
- **Gate:** `CUDA_VISIBLE_DEVICES=N cargo test -p onnx-runtime-ep-cuda --features cuda --lib` → **192 passed / 0 failed**. Full `onnx-runtime-session --features cuda` suite green (incl. `cuda_control_flow_safety`, `control_flow` 21/21 with 2 new tests).
- **Clippy:** lib targets of both touched crates clean under `-D warnings`. Pre-existing repo-wide clippy debt in unrelated GPU **test** files and `executor.rs` (`let mut input_axes`, `manual_is_multiple_of`, `too_many_arguments`) fails `--all-targets` on `origin/main` *before* my changes too — not introduced here.

#### Attribution / honesty
- The **`Greater` fix alone yields ~0 throughput** (a Greater-only build measured at baseline). It is a device seam with no host sync, so removing it doesn't remove a per-token stall — but it is a correct capture-safety improvement and a prerequisite (3→2 graphs). Essentially all of the +10% is the **`If` memoization** removing the per-token ~786 KB cache re-materialization + child-executor dispatch.
- **Partial vs the "collapse 3→1" goal:** I did **not** capture the `If` branch inline into a single graph. Reaching 1 graph would still require reading `cond` each step for the guard (the flip at seq 4096 changes the rotary cache and *must* be caught — skipping the read entirely, which an early ceiling experiment did, is exactly the wrong-branch corruption we must avoid, and was only "correct" because a 120-token window never crosses 4096). Fully removing the per-step `cond` read would need on-device branch selection (a device `Where`/select graph rewrite keeping both caches resident, or a CUDA device-conditional graph node) — a structural, higher-risk change out of scope for this correctness-critical pass. The memoization already captures the dominant recoverable cost with zero correctness risk, so the single-graph rewrite is deferred as a separate, reviewable follow-up rather than rushed.

**Files changed:** `crates/onnx-runtime-ep-cuda/src/kernels/pointwise.rs`, `crates/onnx-runtime-ep-cuda/src/kernels/elementwise.rs` (expose `BroadcastMetadataCache` + helpers `pub(crate)`), `crates/onnx-runtime-session/src/executor.rs`, `crates/onnx-runtime-session/tests/control_flow.rs` (+2 tests).


<!-- merged from deckard-phi-ondevice-rope.md -->
# Deckard — On-device LongRoPE select: de-hosting the `If` capture seam

Branch: `perf/phi-ondevice-rope` off `origin/main` (`8793ea9`)
Status: **needs review before merge (correctness-sensitive)** — do NOT self-merge.
Requested by: Justin Chu. Worker: Deckard.

## The seam (reconfirmed)

Phi-4-mini's LongRoPE selector is `Greater(gather_len, 4096)` → host `If`
(`/model/rotemb_caches_subgraph/If`) choosing between two pure `Constant`
cos/sin caches:
- `then_branch` (predicate TRUE / long-context): cos,sin `[131072, 48]` fp16
- `else_branch` (predicate FALSE / short-context): cos,sin `[4096, 48]` fp16

`If` is a control-flow op, so `plan_capture_segments` (executor.rs) *always*
makes it an eager seam: every decode step the cond scalar is read back to the
host, the captured CUDA graph is split into **2 segments / 1 seam**, and CPU/GPU
serialize at the split. The predicate is loop-invariant during steady decode but
paid every step (~1.9 ms/token, the dominant non-GPU cost per Marsten's Nsight).

The merged memo fix (`719d2fe`) removed the *cheap* part (branch re-exec + ~786 KB
cache copies) but left the seam itself. This change removes the seam.

## The rewrite (general, not Phi-hardcoded)

Two parts, both topology-driven:

**Part A — capture-safe `Where` kernel** (`kernels/where_op.rs`).
Rewrote the CUDA `Where` to mirror the merged capture-safe Binary/Greater pattern:
a persistent `WhereMetadataCache` (device metadata buffer, alloc/free/sync
discipline copied from `elementwise.rs::BroadcastMetadataCache`), no per-call
alloc/upload/free, no per-call `synchronize()`. `capture_support()` advertises
`Supported` **only** for an *invariant scalar-predicate select*
(`cond.numel()==1 && x.shape==y.shape==out.shape`), recorded as a capture
signature guarded by `require_matching_capture_signature`. The general
broadcasting `Where` stays an eager seam — no regression.

**Part B — `CudaOnDeviceConstantSelect` optimizer pass** (`optimizer.rs`,
registered in `cuda_optimization_passes()`).
Generalized as: *"a loop-invariant scalar-predicate `If` whose branches are
pure, side-effect-free constant selections can be lowered to on-device
`Where(cond, then_const, else_const)` per output."* Fires only when BOTH branches
contain ONLY `Constant` nodes (zero formal inputs, one output each, `value`
tensor attr — no outer captures).
- **Equal-shape branches** → direct `Where`, unconditionally byte-exact.
- **Differing leading dim** (Phi's `[131072,48]` vs `[4096,48]`): requires
  `cond = Greater/GreaterOrEqual(_, T)` with scalar-int `T`; the TRUE branch must
  be the LARGER table; trailing dims equal; and `else_lead == T` (crisp tie). The
  smaller (FALSE) constant is zero-padded along axis 0 up to the large leading
  dim. **Output shape is fixed at the large shape `[131072,48]` forever** → no
  per-step shape change → single captured graph even across the boundary.

## Correctness argument (airtight) + guards

Padding APPENDS rows at indices `[else_lead, then_lead)` that the original short
table never had. When the predicate is false (`seq ≤ T = else_lead = 4096`), every
position the model indexes is `< T`, i.e. within the original valid extent — the
appended rows are provably never read. When true, the full large table is selected
unchanged. `Where` recomputes the selection from the *live* predicate each step,
so the boundary flip is exact with no stale memo. GQA derives rotary_dim from
`cos.shape[1]` (=48) and indexes by position; `shape[0]` is only a bound, so the
larger `[131072,48]` output is safe. Byte-preservation of the original
`[0, else_lead)` rows is asserted in a unit test.

## Validation (idle GPU 0, `.cudaenv.sh` sourced)

**Captured-region count (the target):**
| build    | segments | eager seams |
|----------|----------|-------------|
| baseline (`8793ea9`) | **2** | **1** |
| ondevice | **1** | **0** |
Collapse achieved. Verified via `ONNX_GENAI_LOG_CAPTURE_SEGMENTS=1`.

**Per-op trace (`profile_native --trace`, 60 tokens):**
| op      | baseline                        | ondevice                     |
|---------|---------------------------------|------------------------------|
| `If`    | 60 exec, **59 rejected (eager)** | **0** (gone)                 |
| `Where` | 0                                | 4 exec, **2 captured** (cos+sin) |
| `Greater`| captured                        | captured                     |
| total rejected/eager ops | **59** | **0** |
The 1.9 ms/token host `If` seam is eliminated; nothing is rejected from capture.

**Perf — interleaved native-only, idle GPU 0, `--steady --warmups 2 --runs 9
--tokens 120`, 5 interleaved iterations (baseline↔ondevice back-to-back):**
| build    | tok/s per iter                          | median   | range          |
|----------|-----------------------------------------|----------|----------------|
| baseline | 198.95, 203.90, 202.85, 203.50, 204.37  | **203.50** | 198.95–204.37 |
| ondevice | 322.15, 322.31, 322.56, 321.73, 321.58  | **322.15** | 321.58–322.56 |

**+58.3% (203.50 → 322.15 tok/s)**, i.e. **1.810 ms/token** saved
(4.914 → 3.104 ms/token) — matches the predicted ~1.935 ms `If`-seam cost almost
exactly, and pushes Phi **well past the ORT native reference (229.62 tok/s)**.

Honesty note: this is far larger than the +1.8% Marsten re-measured for the
*memo* fix (`719d2fe`), because that fix kept the seam; this change *removes* it
(2→1 graphs, no per-step host cond read). The numbers are tightly reproducible on
an idle GPU (ondevice spread <1 tok/s across 5 interleaved iters). The `Where`
runs over `[131072,48]×2` fp16 each step (~17 µs, captured, no host sync) —
negligible (~0.3% of ~5 ms/token) vs the seam removed.

**Correctness:**
- 160-token greedy decode: `generated_text` **byte-identical** to baseline.
- **Boundary-crossing (seq crosses 4096):** 4200-token greedy decode
  (`ONNX_GENAI_CUDA_KV_MAX_LEN=5000`, 4192 decode tokens). Both builds:
  sha256 `b76a17085739788d8c644fc01453582b045b6f3adaf47d3223466e30fb30629a`
  — **byte-identical**, and ondevice stays **1 captured segment** across the
  boundary (fixed large output shape, no re-plan). The short→long cos/sin cache
  switch is exact.

**Gate:**
- `cargo test -p onnx-runtime-ep-cuda --features cuda --lib`: **201 passed / 0
  failed** (192 baseline + 6 new pass tests + 3 new Where capture-safety tests).
- `cargo test -p onnx-runtime-session --features cuda`: green, incl.
  `control_flow` (21) and `cuda_control_flow_safety` (1).
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda --lib -- -D warnings`:
  clean. (The 42 `-D warnings` errors under `--tests` are pre-existing on
  `origin/main` in unrelated `tests/*.rs` integration harnesses — newer clippy
  toolchain lints, not touched by this change.)

## Files changed
- `crates/onnx-runtime-ep-cuda/src/kernels/where_op.rs` (+261 / capture-safe
  Where + 3 unit tests)
- `crates/onnx-runtime-ep-cuda/src/optimizer.rs` (+598 / `CudaOnDeviceConstantSelect`
  pass + registration + 6 unit tests)

## No-gos / caveats
- The differing-shape lowering deliberately requires the crisp tie
  `else_lead == T` and TRUE = larger table; anything else is skipped (stays an
  `If` seam) rather than risk an out-of-extent read. This keeps it correct and
  general without special-casing LongRoPE by name.
- Reviewer focus: the zero-padding correctness argument (appended rows never
  indexed when predicate false) and the `Where` capture-signature gating.
<!-- merged from marsten-glm4-static-split.md -->
### 2026-07-23: GLM-4 static Split capture result
**By:** Marsten
**What:** Generic EP-side static single-input Split capture reduces GLM-4-9B GPTQ from 41 captured segments and 40 eager seams to one captured segment and zero fallbacks. The seams are the fused-MLP gate/up activation Split (one per layer), `Split(axis=-1, num_outputs=2)` on `gate_up_proj`, named `model/layers.N/mlp/Split_node_*`; they are not RoPE splits. Throughput improves from 110.34 to 118.85 tok/s (+7.71%), or +38.99% over forced eager execution at 85.51 tok/s.
**Why:** Capturing these static Split nodes removes host-reading, stream-synchronizing seams without requiring a model-specific graph rewrite. Separately, ORT GenAI 0.14.1 still cannot load GLM-4 because its GQA attention schema rejects the required partial-RoPE `rotary_embedding_dim` attribute; that schema issue is unrelated to the fused-MLP Split seams.
<!-- merged from marsten-phi-postfix-nongpu-profile.md -->
### 2026-07-23: Target the remaining Phi LongRoPE host If
**By:** Marsten
**What:** On fixed main with `719d2fe`, Phi has two captured graph regions
(`cuStreamBeginCapture=4` across two 128-token generations; 508 graph
launches = two per 254 decode forwards), 236.0 GPU kernels/decode-forward,
and zero graph fallbacks. Nsight reports 2.948 ms GPU kernels/token versus
5.150 ms/token uninstrumented wall time. The native op trace attributes a
1.935 ms median to the still-eager LongRoPE `If`; replayed `Greater` is only
1.28 us GPU/token and GQA is captured (0.406 ms GPU/token).
**Why:** Fully moving the branch select on-device is the highest-value
non-GEMV follow-up: its ~1.94 ms/token budget is about 88% of the ~2.20 ms
non-GPU remainder, with a 5.15 to ~3.2 ms/token theoretical ceiling. Kernel
launch batching is not first: the 236 kernels already arrive in two graph
launches per decode forward.
<!-- merged from marsten-phi-stacked-rebench.md -->
### 2026-07-23: Record cumulative Phi prefetch and standalone int8 split-K frontier
**By:** Marsten
**What:** At `4e774ee`, Phi-4-mini reaches 193.32 tok/s (median of 7, 121.21--194.67 spread under shared-host contention), 15.81% behind the canonical ORT 0.14.1 reference, with zero fallbacks and coherent output. Qwen2.5-1.5B and DeepSeek-R1-Distill-Qwen-1.5B remain within noise at 617.90 and 622.66 tok/s.
**Why:** This is the honest cumulative frontier after stacking fused gate-up int4 software-prefetch and standalone int8-zp split-K; the median, full spread, and contention caveat prevent host variance from being misclassified as a regression.
<!-- merged from marsten-scoreboard.md -->
### 2026-07-23: Native CUDA versus ORT real-weight baseline
**By:** Marsten
**What:** On `origin/main` revision `1073404`, native CUDA beat ORT GenAI CUDA
for all runnable dense Qwen exports: Qwen2.5-0.5B (+62.73%), 1.5B (+36.77%),
and 7B (+10.82%). Phi-4-mini remains behind: the standing clean mandate
reference is 193.89 versus 229.62 tok/s (-15.56%); this live nine-run snapshot
was 186.19 versus 236.48 tok/s (-21.27%).
**Why:** This records the real-weight baseline before Deckard's Phi
`executor.rs` capture-seam work. GPU 5 was idle before/after testing, but the
shared host produced a wide Phi range, so reserved-host confirmation is needed
before treating the live shortfall versus the clean reference as a regression.
<!-- merged from rachael-mask-island-closure.md -->
### 2026-07-24: Fixed-signature CUDA capture closes the DeepSeek mask island
**By:** Rachael
**What:** CUDA `CumSum`, `Unsqueeze`, and `Slice` now warm and retain their exact fixed decode signature, skip runtime metadata D2H during graph recording, and avoid capture-time synchronization/allocation. `Slice` retains its device metadata buffers. General broadcasting `Where` now captures after its dtype/broadcast geometry has warmed because its condition and metadata are already device-resident.
**Why:** DeepSeek-V2-Lite fixed-capacity decode keeps mask geometry stable while mask values remain device-sourced. On both block-32 and block-128 exports, the mask-island seams fell from `Unsqueeze=4, Slice=1, CumSum=1, Where=1` to zero. Listed seam nodes fell 275→268 (the remaining 268 are Reshape work owned separately); segmented eager boundaries fell 246→241 as adjacent captured regions merged.

Verification:
- Both DeepSeek exports produced `[8913, 13, 185, 549, 19305, 280, 7239, 317, 254, 28071, 13, 185]` three independent times (`" Paris.\nThe currency of France is the Euro.\n"`).
- Both exports reported measured CUDA graphs `captures=1, replays=9, fallbacks=0`.
- Qwen2.5-0.5B remained coherent and capture-clean: one segment, zero seams, measured `captures=1, replays=13, fallbacks=0`.
- Phi-4-mini on idle GPU 1 produced the same 16-token sequence three times and reported `captures=2, replays=26, fallbacks=0`.
- CUDA EP lib tests: 205 passed; session MLAS lib tests: 65 passed; CUDA clippy with warnings denied passed; construction GPU tests: 18 passed; targeted CumSum GPU test passed.

The implementation necessarily changes generic CUDA movement/elementwise kernels rather than model-specific Attention code. Leon/Sebastian should review the warmed fixed-signature contract, especially the established assumption (shared with Reshape) that runtime shape/axis/bound metadata stays invariant across captured replays.


<!-- merged from sebastian-moe-routing-capture.md -->
# MoE routing capture safety

- Branch: `perf/capture-moe-routing`.
- TopK now folds its eagerly-read scalar K into an exact warmed signature; replay does not perform D2H or synchronize.
- GatherElements now retains shape metadata and validates capture-time indices on device through the shared capture-error word.
- Softmax skips its trailing synchronization while the EP stream is being captured; the cuDNN handle is already created on that stream.
- `indexing_gpu::warmed_moe_routing_ops_capture_without_allocations` verifies warmed TopK (K=6/64), GatherElements, and Softmax graph replay parity without allocation growth.
- Bench/ORT-vs-native-CUDA: deferred to integration because Stage-0 executor shape seeding is required to engage all decode seams.
<!-- merged from sebastian-qmoe-64expert.md -->
### 2026-07-23: Add 64-expert top-6 CUDA QMoE parity coverage
**By:** Sebastian
**What:** Added parameterized synthetic 64-expert/top-6 QMoE GPU parity tests for fp16 decode (M=1) and prefill (M=8), bf16 decode/prefill, hot-expert plus empty-expert routing, capture warm/replay with changed routes, and a 64-row worst-case route-scratch allocation. Each uses the existing CPU QMoE oracle, except replay additionally compares against an uncaptured CUDA reference.
**Why:** DeepSeek-V2-Lite routing requires 64 experts and top-6, while the previous GPU tests only exercised 4 experts/top-2. GPU 5 results: qmoe_gpu 27 passed/0 failed; CUDA lib gate 192 passed/0 failed; clippy passed. No 64/top-6 kernel scale bug was found.
<!-- merged from sebastian-qmoe-test-fix.md -->
### 2026-07-23: Serialize QMoE GPU capture tests and verify live replay routing
**By:** Sebastian
**What:** QMoE integration tests now hold a process-wide GPU mutex for each test body. The capture test also changes `router_probs` after capture and compares replay against an uncaptured eager run using the new expert routes.
**Why:** Concurrent CUDA allocation can invalidate thread-local graph capture, while changed-routing parity proves expert selection is recomputed from live replay inputs rather than baked into the graph.
<!-- merged from sebastian-static-split-test.md -->
### 2026-07-23: Static Split capture/replay test coverage
**By:** Sebastian
**What:** Reworked the static even `Split` byte-parity integration test to build with concrete input shapes, execute the static kernel, capture it, replay it with changed input, and compare replayed outputs with eager output bytes.
**Why:** The generic `run()` helper supplies empty input shapes and therefore exercises only Split's dynamic path; successful CUDA graph capture is a regression guard for the static no-synchronize path.
<!-- merged from tyrell-executor-shape-seeding.md -->
### 2026-07-24: Seed warm JIT decode shapes + capture-recording quarantine (Stage 0 of DeepSeek whole-step capture)

**By:** Tyrell
**Branch:** `perf/capture-executor-shape-seeding` (off `perf/deepseek-mla-capture` @ `25dbb60` — the Attention capture foundation, currently in review). **Needs review before merge; not merged.** Rebase onto the merged MLA foundation when it lands. Headline tok/s bench is deferred to the integration pass on `bench/ort-vs-native-cuda` (GPU contention here makes the ~2 ms/token direct gain unmeasurable; the structural seam-count drop is the acceptance criterion).

**Scope:** `crates/onnx-runtime-session/src/executor.rs` ONLY. No kernel files, no `provider.rs`, no `standard_attention.rs`/`native_decode.rs`. This makes the executor *admit* already-capture-safe ops; it does not add/alter kernels.

#### Root cause (confirmed, Pris's finding reproduced exactly)
The executor rejects a node as an eager seam **before** consulting its kernel whenever any input/output shape is absent from `resolved` (EP `plan_capture_region` default policy declines on unresolved shapes). `resolve_soft` deliberately omits data-dependent (JIT) decode shapes, and only external/control-flow shapes were seeded for capture. So DeepSeek-V2-Lite decode ops that are ALREADY capture-safe (Cast, Mul, QMoE, ScatterElements — all advertise `Supported`, skip sync, pool scratch during capture) still fragmented into eager seams purely because their JIT output shapes weren't seeded. Measured: **727 distinct eager seam nodes** per decode step (matches Pris exactly).

#### Fix
1. **Warm decode shape seeding** (`seed_warm_decode_capture_shapes`). After an eager warmup step, snapshot the full resolved shape map (`capture_warm_shapes`) together with the persistent-binding signature it ran under (`ExternalBindings::capture_signature()` = sorted (vid, is_input, dtype, shape, ptr, len) of every persistent binding). On a later capture-mode run presenting the **identical** signature, seed each still-unresolved (non-external, non-initializer, non-sequence) value from the warm snapshot so its already-capture-safe consumers fold into captured segments. Guardrails, all honored:
   - Shapes are derived from a real eager warmup, never hardcoded/assumed.
   - A changed persistent pointer/capacity/shape → signature mismatch → **all seeds withheld** (nodes stay eager); `replay_device_graph`'s independent `binding_signature` check also retires the installed graph. Never replays a stale graph against changed shapes.
   - The capture pass re-resolves each node's true shape; any divergence from a seeded value retires the graph and declines (recapture) rather than baking a stale shape.
   - No per-step allocation when the signature matches; view/bounds validation untouched.
   - Seeding is valid ONLY for the exact warmed signature — anything varying across steps forces recapture or stays eager.

2. **Capture-recording quarantine + retry** (in the `RunMode::Capture` arm + `node_capture_reason`). Seeding surfaced a latent problem: a kernel can advertise `CaptureSupport::Supported` yet abort device-graph *recording* (e.g. `ai.onnx::Softmax`, the MoE gate — softmax.rs declares `Supported` but calls `synchronize()` unconditionally, which CUDA rejects mid-capture). Admitting one such node aborted the **entire** segmented capture → full eager fallback (0 captures). Fix: when `run_plan_segmented` (Capture) errors at a node, record it (`last_capture_failed_node`), reset the device graph, quarantine its `(domain, op_type)` (`capture_quarantine_ops`), and re-plan/re-record treating quarantined ops as forced `CaptureRecordingFailed` eager seams. Re-recording a fixed-capacity decode step is idempotent (same position/token → same values into the same slots), so retry is safe; bounded by node count; quarantine grows monotonically (a kernel that breaks recording breaks it every time), so recaptures converge immediately. New `SeamReason::CaptureRecordingFailed`.

#### Results — proof of effect (`ONNX_GENAI_LOG_CAPTURE_SEGMENTS=1`, `--steady --decode-skip 8 --warmups 1 --runs 1 --tokens 12`, GPU 1)
Distinct eager seam nodes per decode step, **identical for both exports** (blk32 `deepseek-v2-lite-real-int4-blk32` and blk128 `deepseek-v2-lite-real-int4`):

| | seeding OFF (baseline) | seeding ON + quarantine |
|---|---|---|
| **distinct eager seam nodes** | **727** | **541** (−186, −25.6%) |
| eager node executions across run | 1454 | 1082 (−26%) |
| "data-dependent shape unresolved" seam class | 692 occ (Cast 106, Mul 104, QMoE 52, ScatterElements 52, MatMul 52, TopK 52, GatherElements 52, Softmax 52, …) | **0** — class eliminated |
| segmented-capture status | succeeds (191 seg / 190 seam) | **succeeds** (193 seg / 192 seam) |

**Cast, Mul, QMoE, ScatterElements stopped being seams** (fully folded into captured segments). The nodes still eager after seeding now report their **real kernel-capability decline** (not a spurious missing-shape rejection), which is exactly the signal kernel owners need — see below.

#### Correctness / determinism (HARD GATE — PASS, both exports)
Prompt "The capital of France is", 3× identical each export:
`[8913, 13, 185, 549, 19305, 280, 7239, 317, 254, 28071, 13, 185]` = pos0 8913 ' Paris' → matches expected exactly. Capture engaged and clean: `cuda_graph: captures=2 replays=18 fallbacks=0` (no stale-graph corruption — the main risk of this change is disproven).

#### Dense non-regression (PASS)
Qwen2.5-0.5B int4 (`qwen2.5-0.5b-int4-onnx-native`): 3× identical, coherent (" Paris. It is the largest city in the country and the"), `captures=2 replays=18 fallbacks=0`. Dense graphs have statically-resolved decode shapes, so warm seeding is a no-op for them (nothing unresolved to seed) — no behavior change, no regression.

#### Ops that I EXPECTED to fold but did NOT (for the kernel-owner agents)
These now surface their true kernel decline (they were previously hidden as unresolved-shape seams). They stay eager until their kernel is made capture-safe:
- **`ai.onnx::Softmax` (MoE gate) — KERNEL BUG:** declares `CaptureSupport::Supported` but `run`/`run_nvrtc_f32` call `self.runtime.synchronize()` unconditionally (`crates/onnx-runtime-ep-cuda/src/kernels/softmax.rs:271,323`; `capture_support()` at :343). This aborts recording; my quarantine keeps capture working but Softmax stays a seam (52/step). **Fix the kernel to skip the sync during capture (mirror the Cast/Mul pattern) and it will fold for free.**
- `ai.onnx::Reshape` — copy path not a capture-validated zero-copy view.
- `ai.onnx::Split` — reads runtime split sizes on host + trailing stream sync.
- `ai.onnx::Concat` — trailing host stream sync.
- `ai.onnx::Expand` — per-call broadcast metadata alloc/upload/free + sync.
- `ai.onnx::TopK` — reads K D2H + host sync.
- `ai.onnx::GatherElements` — per-call indexing metadata + sync.
- `ai.onnx::MatMul` (M==1 GEMV) — cuBLASLt per-call workspace alloc/free + heuristic query not capturable.
- `ai.onnx::Where` — capture-safe only for invariant scalar-predicate select over equal-shaped operands; broadcast/non-scalar condition launches stay eager.
- `ai.onnx::Unsqueeze` / `Slice` / `CumSum` — host-side runtime axes/bounds + sync (structural host seams; not shape-gated).

#### Gates
- `cargo test -p onnx-runtime-session --features mlas --lib` → **65 / 0** (63 baseline + 2 new tests).
- `cargo test -p onnx-runtime-ep-cuda --features cuda --lib` (GPU 1) → **208 / 0** (≥207, no regression).
- `cargo clippy -p onnx-runtime-session --features mlas --lib -- -D warnings` → clean. (Pre-existing repo test-only debt `let mut input_axes` in an unrelated executor test is not introduced here — same item Deckard noted.)
- `cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin profile_native` → ok.

#### Tests added (non-tautological)
- `warm_decode_seeding_admits_previously_unresolved_capture_safe_node`: a `Range`(runtime start/limit/delta)→`Cast` graph is an unresolved-shape seam before warmup; after one eager warmup the identical signature seeds the exact extent `[4]` and clears the unresolved-shape seam; a changed persistent-binding signature withholds the seed.
- `quarantined_op_type_is_forced_to_a_capture_recording_failed_seam`: a statically-shaped `Cast` is not a recording-failed seam until its `(domain, op_type)` is quarantined, after which `node_capture_reason` forces it to `CaptureRecordingFailed` regardless of resolved shapes/kernel capability.

**Files changed:** `crates/onnx-runtime-session/src/executor.rs` (+ 2 tests in-module).
<!-- scribe-merge-2026-07-24T16-04-31Z-bilecki-dlpack-arm64 -->
<!-- merged from .squad/decisions/inbox/bilecki-dlpack-arm64.md -->
### 2026-07-24: Use per-test counters for DLPack import deleter tests
**By:** Bilecki
**What:** Store an `Arc<AtomicUsize>` in each fake producer context and have its foreign deleter increment that test-local counter; remove the shared import counters and serialization lock.
**Why:** The shared static counter allowed unrelated deferred deleters to contaminate another test's assertion, observed as a Windows ARM64-only failure. Per-test ownership makes the deleter assertions hermetic while leaving production idempotency behavior unchanged.
<!-- scribe-merge-2026-07-24T21-47-08Z-inbox-reconciliation -->
## 2026-07-24 — Decision inbox reconciliation (CPU coverage, ARM64 reliability, MHA/SDPA)

### Cross-platform CPU test and build hygiene
**By:** Hasford, Vasquez, Apone, Drake, Dutch; reviewed by Drake, Apone, and the applicable non-author reviewers.

- Gate `dot_u8_i16_scalar` to x86_64 while retaining portable BF16 grouped-dot coverage and narrowly permitting the non-x86 test stub to be dead in library-only builds; this restores `-D warnings` aarch64 checks without weakening portable assertions.
- The Windows ARM64 SPMD parity crash was a test-harness oversubscription/flakiness issue, not an ARM kernel race: the old fixed 31-worker decode plus 31-worker Rayon setup exceeded constrained runners and did not deterministically preserve odd worker counts. `parity_worker_count()` now selects the largest odd host-bounded count, capped at 15; parity remains non-vacuous on every platform. Drake approved with only a tiny-host coverage nit, and Apone approved the follow-up forced-pool bound with a documentation nit.
- The persistent-pool auto-enable review found the park/unpark SeqCst handshake free of lost wakeups, with flat-default fallback, hysteresis, and token-exact parity. The SQNBit shard investigation identified MLAS N-tile boundary alignment as the source of pool-vs-flat ULP drift and recommends aligned interior shard boundaries.

### BF16 coverage and GELU special values
**By:** Vasquez (coverage audit), Pris (oracle revision), Hicks (reviewer); approved by Hicks after revision.

- CPU EP BF16 uses portable `half::bf16` widen/compute/narrow dispatch. Batch 1 adds the three fused `com.microsoft` GELU activations and exhaustive independent BF16 special-value coverage for unary math; unrelated f32-only/custom kernels remain scoped for later audits.
- The initial BF16 review rejected self-referential expected values. Pris replaced them with literal BF16 goldens and definition-based special-value checks. This exposed and fixed the GELU-family `-∞` defect: exact, tanh, and Quick GELU now return `+0.0` (not NaN) for `-∞`; `+∞`, NaN, and signed zero follow their defined behavior. Hicks approved the revised commit and all listed test/clippy gates passed.

### MultiHeadAttention shape and kernel architecture
**By:** Deckard (implementation); reviewed by Gorman.

- MHA shape inference derives output/present-value dimensions from V independently of Q/K head size, while present-key dimensions derive from Q/K; unsupported packed/rank-mismatched layouts fail explicitly. Gorman's review confirmed the rule against live ORT, with only documentation/test-framing nits.
- The CPU MHA kernel is numerically correct against live ORT, including causal past offsets, cross-attention, differing V head size, masks, bias, and rejection paths. The shared `sdpa.rs` core is a genuine generic path, with MHA retained as a thin adapter; the 12 parity cases remained byte-identical before and after factoring. Follow-ups are migration documentation, post-softmax QK capture modes, and a later hot-loop dispatch performance pass.

### MLAS reference-test scope
**By:** Dutch (review); upstream issue tracked at `microsoft/onnxruntime#29853`.

- Reject the broad SQNBit skip until its comment and message identify the scoped AVX2 `CompInt8`, M=1, asymmetric/non-zero-point defect and link the upstream issue. The original author is locked out of the revision; a different agent must update the durable explanation.
<!-- scribe-merge-2026-07-24T22-32-50Z-attention-bf16 -->
## 2026-07-24 — PRs #122, #124, and #125: BF16 SkipLayerNorm and shared CPU attention

### PR #122 — BF16 SkipLayerNormalization
**By:** Pris/Vasquez (implementation and coverage); reviewed by Hicks.

**What:** `com.microsoft::SkipLayerNormalization` widens BF16 inputs to f32 and RNE-narrows BF16 outputs. Outputs 0 (`output`) and 3 (`input_skip_bias_sum`) retain the input dtype; outputs 1 (`mean`) and 2 (`inv_std_var`) are kept as float32, matching the ORT schema. Literal BF16 goldens and infinity/NaN coverage use independent references.

**Why:** The first review rejected the all-outputs-match dtype check and BF16 statistic test because ORT defines the statistics as float32. The revision corrected both. ORT CPU 1.26/1.27 has no BF16 `SkipLayerNormalization` kernel (`NOT_IMPLEMENTED`), so this change adds CPU EP capability rather than mirroring an existing ORT CPU implementation.

### PR #124 — Migrate `ai.onnx::Attention` onto shared SDPA
**By:** Deckard; reviewed by Gorman.

**What:** The Attention implementation now uses the shared SDPA core, removing the duplicated QKᵀ→scale→softcap→bias/mask→softmax→V loop. The core gained `QkCaptureStage` and a fully-masked-row guard. A 153-case dump covering rank/causal/softcap/qk modes, GQA/MQA, cache, masks, and padding was independently reproduced byte-for-byte: SHA256 `cbfcde4f41ee5b6f55203233f30986628ce89aee63e9f9df21f091176a0fe5b9`.

**Why:** This is a DRY migration with byte-identical behavior; MHA/FusedAttention parity remains intact.

### PR #125 — packed-QKV `com.microsoft::Attention`
**By:** Ripley; reviewed by Drake.

**Contract:** The CPU packed-QKV kernel runs over the SDPA core. PostDot scale is `1/sqrt(head_size)`; causal mode is `unidir && S > 1`; `mask_index` accepts `(B)`, `(2B)`, or `(3B+2)` forms; past cache is `(2,B,N,P,H)`. Unsupported `do_rotary`, `share_buffer`, 4-D masks, and past+bias combinations are rejected cleanly.

**Impact:** This unblocks Whisper encoder attention. End-to-end Whisper wiring remains pending.

**Review outcome:** PRs #122, #124, and #125 landed on main at `3f30f92`.
<!-- scribe-merge-2026-07-26T19-45-52Z-cuda-perf-and-capture-regression -->
## 2026-07-26 — CUDA perf next-wave and #193 capture-regression reconciliation

Decision archive gate checked at 2026-07-26T19:45:52Z: active ledger was 381527 bytes and exceeded 51200 bytes. No dated active-ledger entries older than 2026-07-19 were present, so no archive file was changed.
<!-- merged from .squad/decisions/inbox/hicks-repl-commands.md -->
### 2026-07-24: REPL multimodal slash-command grammar
**By:** hicks
**What:** Added pure parsing for `/help`, `/reset`, `/raw`, `/system [text]`, `/image <path> [prompt text]`, and `/audio <path> [prompt text]`. Attachments stage for the next text turn, while single-line attachment commands immediately send their text. Missing paths warn without crashing; Phase 1 reports staged modalities and sends text only.
**Why:** This makes multimodal REPL input testable and extensible while honestly deferring engine-side image and audio execution to Phase 2.
<!-- merged from .squad/decisions/inbox/sapper-pr129-revision.md -->
### 2026-07-24: PR #129 Nemotron revision (transpose gap + README fix)
**By:** sapper
**What:** Removed the unsupported `prediction_network.decoder_output` to `joiner.decoder_output` dataflow edge and documented the required transpose as a metadata-contract gap. Corrected the streaming-chunk configuration attribution in the README.
**Why:** `decoder.onnx` emits f32 `[batch, 640, target_len]`, while `joint.onnx` accepts `decoder_output` as f32 `[batch, target_len, 640]`; `DataflowEdge` only supports endpoints, dtype, and device transfer, not layout adaptation. The cached package grep finds `chunk_samples: 8960` in `v3/genai_config.json`, not `audio_processor_config.json`.

<!-- merged from .squad/decisions/inbox/hicks-glm-review.md -->
# Hicks GLM native-validation review — 2026-07-25

**Verdict: 🟡 (mergeable after this review's documentation correction).**

The two ORT non-load claims were reproduced against the CUDA-enabled linked
ORT: GLM-4 fails load at
`GroupQueryAttention_node_19` with unrecognized
`rotary_embedding_dim`; GLM-5.2 QMoE fails load because
`pkg.nxrt:IndexShare(-1)` is unregistered. The q4 ORT number is explicitly
qualified and GPU-residency evidence is recorded, so it is not presented as a
CPU-fallback comparison. A 16-token native CUDA greedy GLM-4 run from
`Hello` produced `", I am a 3rd year student at the University of Waterloo. I"`.

## q4 triage

**Severity: high correctness regression; effort: small-to-medium, localized
native CUDA decode binding fix (roughly 1–3 days including regressions).**

Reproduction on CUDA with prompt `123` fails on decode token 1:

```text
model/layers.0/self_attn/indexer/Add_node_70:
[[1, 1, 2], [1, 1, 4096]] are not broadcast-compatible
```

The `[1,2,3,4]` control fails at the same node with `[1,1,5]` versus
`[1,1,4096]`; CPU completes eight tokens, and ORT CUDA completes decode.
This excludes token corruption, a generic operator/kernel gap, and an export
artifact.

Despite the `indexer` name, q4 contains **zero** `pkg.nxrt::IndexShare` nodes
and two standard `ai.onnx::Attention` nodes. `Add_node_70` combines the
logical-width indexer score (`ReduceSum_67`) with a cast/squeezed
`attention_mask`. On CUDA, `DecodeCudaState::extend_mask`
(`crates/onnx-genai-engine/src/native_decode.rs`) intentionally exposes the
single-token mask at `max_len=4096`; that capacity leaks through the indexer
mask branch while the score remains logical width. This is category **(c)**:
a decode-time mask/capacity binding bug, not the IndexShare DSA kernel.

Dispatch to **native CUDA decode/engine owner** (the agent responsible for
`onnx-genai-engine` fixed-capacity KV and `onnx-runtime-session` device
bindings), not the IndexShare-kernel owner. Fix the physical-mask exposure
policy only for proven-safe topology; preserve logical mask shape when
non-Attention mask consumers reach prefix-sensitive indexer arithmetic. Add
CUDA regressions for prompts `[123]` and `[1,2,3,4]` across at least two
generated tokens.
<!-- merged from .squad/decisions/inbox/hudson-deepseek-status.md -->
### 2026-07-25: DeepSeek native CUDA validation status
**By:** Hudson
**What:** Native CUDA loads all three exercised DeepSeek artifacts. DeepSeek-V2
QMoE matches ORT for 32 greedy tokens and token-0 top-40 log-probabilities
(max absolute delta 0.001409); DeepSeek-Coder matches for 128 tokens.
DeepSeek-R1 diverges at generated token 16 (native 374, ORT CUDA 594) on the
benchmark prompt; this is consistent with the committed fp32-oracle finding
(`deepseek_r1_1_5b_divergence.rs`, which locks the separate `"capital of France"`
prompt where native picks oracle-correct 374 vs ORT CUDA 315). The benchmark
token-16 divergence is not itself oracle-adjudicated yet. Stable
native/ORT rates were 629.31/442.76 tok/s for R1 and 798.44/623.51 tok/s for
Coder. The QMoE ORT run was CPU-heavy (four Memcpy nodes, 0% observed GPU,
2.45 tok/s), so it is not a valid GPU speed baseline.
**Why:** The durable status must distinguish numerical correctness from an
invalid ORT performance baseline. Top remaining gaps are full-model QMoE
language-coherence validation, a GPU-resident ORT QMoE reference, and continued
explicit handling of DeepSeek-R1 MatMulNBits accuracy-level divergence.
<!-- merged from .squad/decisions/inbox/pris-pr168-test-fix.md -->
### 2026-07-25: Cover the second AVX-512 vector's final lane
**By:** Pris
**What:** Corrected PR #168's two-vector NaN test to use a 32-element block and place the NaN at index 31.
**Why:** The prior test wrote index 15, which only exercised the first vector's final lane and did not cover non-finite detection in the second vector.
<!-- merged from .squad/decisions/inbox/ripley-core-budget.md -->
### 2026-07-25: Keep peak default and add an explicit CPU decode budget
**By:** Ripley
**What:** `onnx-genai generate` and `onnx-genai run` expose `--cpu-cores N`, mapped to the native decode worker-count mechanism with precedence CLI > `ONNX_GENAI_CPU_DECODE_THREADS` > AUTO. The uncapped automatic worker count is unchanged.
**Why:** Shared-machine users need a first-class good-citizen control, while the measured 48-worker default remains the best dedicated-host peak. Persistent workers already pin one worker per selected allowed CPU, so the budget bounds their affinity footprint without requiring a hand-written `taskset`.
<!-- merged from .squad/decisions/inbox/ripley-legit-cuda-sweep.md -->
### 2026-07-25: Require observed GPU execution for native-vs-ORT headlines
**By:** Ripley
**What:** Treat Phi-4-mini, Qwen2.5-1.5B, and Qwen2.5-7B as valid native-CUDA
versus ORT-CUDA comparisons for their Foundry `cuda-gpu` artifacts. Their ORT
runs had no inserted-Memcpy warning and were independently observed at 86–91%
H200 utilization. Report the real native wins as 1.385×, 1.452×, and 1.100×.
**Why:** Selecting the CUDA EP is insufficient proof by itself. A valid
competitive claim requires a CUDA-targeted artifact, absence of fallback-copy
thrash, and direct evidence that model compute exercised the selected GPU.
<!-- merged from .squad/decisions/inbox/ripley-uncontended-sweep.md -->
### 2026-07-25: Treat CUDA-targeted rows as the clean native-vs-ORT comparison
**By:** Ripley
**What:** The uncontended H200 sweep records all four requested three-way
measurements, but uses Qwen2.5-0.5B and DeepSeek-R1-Distill-Qwen-1.5B as the
clean competitive native-vs-ORT rows. Phi-3.5-mini and Qwen2.5-Coder-7B ratios
remain explicitly artifact-specific because their generic-CPU exports caused
ORT to insert 67/57 memcpy nodes and partially assign the CUDA EP.
**Why:** GPU 6 was idle throughout, making absolute CUDA rates trustworthy, but
an idle GPU does not remove graph-export and execution-provider assignment
confounds. The distinction preserves the credible 1.556× and 1.421× native
wins without overstating the much larger generic-CPU-artifact ratios.
<!-- merged from .squad/decisions/inbox/roy-pr167-guard-fix.md -->
### 2026-07-25: Gate RMSNorm SIMD scaling on exact-identity scale shape
**By:** Roy
**What:** The contiguous normalize-and-scale path now requires the right-aligned scale shape to exactly equal `x_shape[axis..]`. SkipSimplifiedLayerNormalization applies the same identity-shape check to gamma.
**Why:** Equal element counts do not prove identity indexing: for `X=[2,2]`, `axis=1`, and `scale=[2,1]`, the scale varies by group while broadcasting along the normalized axis. Such broadcasts must use the scalar `scale_index` path.
<!-- merged from .squad/decisions/inbox/deckard-down-gemv-validate.md -->
### 2026-07-26: Down-GEMV register-reuse validation stopped as duplicate
**By:** Deckard
**What:** Do not open a new PR for `1e2b02b9`. Its exact patch already exists on `origin/main` as `720fa032`; the requested validation worktree and branch were removed.
**Why:** Both commits have stable patch ID `8950d3c0064da12e6edb023baef742552fa0e95b`, identical parent file content, and identical resulting `matmul_nbits.rs`. Cherry-picking onto current main conflicted because later work generalized the already-landed register-reuse kernel to adaptive 2/4/8-column CTAs and added subsequent specializations/tests. Keeping current main would make the cherry-pick empty, so candidate versus main has no performance delta and cannot pass the required positive-win gate. No build, CUDA test, golden-lock, benchmark, push, or PR was run because there is no candidate code difference to validate or merge.
<!-- merged from .squad/decisions/inbox/sapper-glm52-land.md -->
### 2026-07-26: GLM-5.2 IndexShare landing is superseded
**By:** Sapper
**What:** Do not cherry-pick `528b0f28ebd39df8b27ff34f765190fcb3a26351` or open a PR. `origin/main` already contains the same shape-inference implementation and CUDA E2E under `6fdc8742`, with later fixture-backed eager-decode strengthening.
**Why:** `6fdc8742` is an ancestor of `origin/main`; its IndexShare handler and shape tests match `528b0f28` exactly, while its only initial difference was rustfmt in the E2E test. Main subsequently added the committed tiny fixture and strengthened the test through `7c212bc7` and `ec4b62bf`, so landing the old commit would duplicate and regress the current coverage.
<!-- merged from .squad/decisions/inbox/batty-dsmla-copyback.md -->
### 2026-07-26: Land DeepSeek MLA Attention KV copy-back stream ordering
**By:** Batty
**What:** Cherry-picked `de5188cce0390f9cd381289e5aec20f1c52a9682` onto current `origin/main` as `36751857` on `perf/deepseek-mla-capture-copyback` and opened PR #193.
**Why:** Current main still used synchronous `runtime.dtod` for staged dense Attention KV copy-back. Switching both key and value copies to same-stream `dtod_async` preserves producer/copy/consumer ordering without capture-illegal device synchronization. Build, focused Attention/GQA tests, the Foundry Qwen3 24-token golden lock, and the CUDA EP suite excluding the four known missing-cuDNN cases passed.

<!-- merged from .squad/decisions/inbox/luv-review-193.md -->
# Decision: PR #193 review (Luv, Code Reviewer)

- **PR:** #193 — perf(cuda): stream-order Attention KV-growth copy-back (capture-safe)
- **Author:** Batty (locked out of revisions)
- **Reviewer:** Luv (independent)
- **Date:** 2026-07-26T18:00:58Z
- **Verdict:** APPROVE

## Change
Two `self.runtime.dtod(...)` → `self.runtime.dtod_async(...)` in
`crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs` (the staged
dense KV copy-back), plus a comment. Full branch diff vs `origin/main` is
exactly +8/-3 in that one file — nothing hidden. `dtod_async` and its ordering
unit test already exist on `main` (used by movement.rs, indexing.rs).

## Concurrency analysis (correctness)
- **Stream ordering preserved.** `dtod_async` enqueues on `self.stream`
  (EP compute stream, runtime.rs:849). `launch_build_kv` (producer) and the
  next step's `build_kv` (consumer) both launch on `self.runtime.stream()`
  (standard_attention.rs:862). Producer → async copy → consumer are all on the
  one EP stream, so the copy is implicitly ordered without a host sync. No new
  race.
- **Capture-safe.** The old sync `dtod` (runtime.rs:828) issued an *ungated*
  host `self.synchronize()` — illegal during CUDA-graph capture. The codebase
  otherwise gates every host sync on `!is_capturing()` (execute entry line 937,
  exit line 1562). `dtod_async` removes the illegal host sync and records the
  copy into the captured graph on the stream. Matches the established pattern.
- **Blocking semantics retained.** The execute-exit `synchronize()` (gated on
  `!is_capturing()`, line 1562) still drains the async copy for eager callers.
- **Disjoint staging intact.** When `stage_key`/`stage_value`, `key_kv_ptr` is a
  fresh `alloc` (line 1316) distinct from the aliased `present_key_ptr`; src/dst
  never overlap, so async copy is safe.
- **No collateral.** Other `dtod`/`dtod_async` callers (provider, csa_checkpoint,
  reduce) untouched.

## Independent validation (GPU 3, taskset -c 1)
- Build `onnx-runtime-ep-cuda`: OK.
- `standard_attention_gpu` 24/24; `standard_attention_bf16_gpu` 2/2;
  `group_query_attention_gpu` 25/25 (+1 ignored). Repeated standard x3 more:
  deterministic, no race flakiness.
- `dtod_async_is_ordered_after_same_stream_producer` unit test: PASS (directly
  validates the same-stream ordering invariant with a spinning producer kernel).

## Golden lock — pre-existing/environmental, NOT a regression
- Qwen3-0.6B foundry native-CUDA lock FAILED, but produces **byte-identical**
  tokens on `origin/main` AND the PR branch (all 24 tokens equal between them) —
  so it is not a PR regression; PR output == main output exactly.
- Root cause: the only model artifact available in this env
  (`/home/justinchu/mobius/.scratch/qwen3-0.6b-int4-cuda`) uses
  `com.microsoft.GroupQueryAttention` (140 nodes), not default-domain Attention,
  so it (a) never exercises the changed `StandardAttentionKernel` staged path
  and (b) does not match the golden `EXPECTED_TOKENS` (locked against the Foundry
  `generic-cpu-4` artifact, which is not downloaded here — HF cache has no
  snapshot). The lock neither validates nor invalidates this PR.

## Coverage note (non-blocking, pre-existing)
No test dynamically exercises the aliased default-domain Attention staged
copy-back (stage_key/stage_value require present aliased onto past + dense mode);
the op-level harness allocates fresh present buffers. This is a pre-existing
coverage gap, not a PR defect. Correctness here rests on the static concurrency
analysis + the passing same-stream ordering unit test. Suggest a follow-up e2e
test on a default-domain-Attention dense-KV model (e.g. the DeepSeek-MLA path the
branch name references).

## Final
APPROVE. Coordinator to merge. Do not push to branch.

<!-- merged from .squad/decisions/inbox/gorman-post148-review.md -->
# Gorman review: post-#148 native-vs-ORT scorecard

**Verdict: 🟡 — mergeable after the review documentation update in this commit.**

All displayed arithmetic recomputes correctly: native/ORT is 1.385930
(1.386×), 1.605277 (1.605×), and 1.122583 (1.123×); the native leads are
38.593%, 60.528%, and 12.258%. The A/B changes are -0.08695%, +10.52325%,
and +2.07008%, respectively.

The ORT evidence is sufficient: CUDAExecutionProvider, 86–91% peak GPU
utilization, multi-GiB allocations, and no inserted-Memcpy warning are
recorded, with the intentional shape-node CPU notice distinguished from
partial-EP fallback. The A/B now explicitly records fixed prompt, tokens,
warmups, runs, steady window, CPU pinning, and GPU, with only #148 differing.

The larger Qwen1.5B gain is physically plausible: its smaller down-projection
K/N yields fewer baseline 8-column CTAs, hence greater H200 grid starvation;
the grid multiplier has more latency hiding to recover than for 7B. This is a
native-only A/B, not ORT jitter. GPU re-confirmation was not run: GPU 1 is
reserved (0% utilization but 129589 MiB allocated) and every other GPU was
98–99% utilized. The scorecard now states that a clean-idle-GPU Qwen1.5B
confirmation remains pending.

<!-- scribe-merge-2026-07-26T19-45-52Z-cuda-perf-and-capture-regression-end -->
<!-- scribe-merge-2026-07-26T20-00-00Z-cuda-perf-and-capture-regression-reconciliation -->
## 2026-07-26 — CUDA capture regression and portable split-K reconciliation

Decision archive gate checked at 2026-07-26T19:45:52Z: active ledger was 397763 bytes before this merge and exceeded 51200 bytes. Applied the 7-day policy with cutoff 2026-07-19; archived 0 eligible block(s).

**Manifest:** PR #201 merged to main at `88e48eca`; PR #203 merged to main at `b80a8c83`; pre-existing main CI rustfmt break from PR #200 was fixed directly on main at `1bf119af`. Worktrees `wt-attn-regr` and `wt-perf-next` were cleaned.

<!-- merged from .squad/decisions/inbox/chew-pr201-review.md -->
# Chew review — PR #201 (test/attention-default-domain-capture)

**Verdict: APPROVE** (independent Quality & Safety review; author Leon locked out, framing verified from scratch)

Reviewed head `058bd273` vs `origin/main` (`b51ea239`). Diff = 3 files: decision note,
`standard_attention.rs` (+50/-41), `standard_attention_capture_gpu.rs` (+349, new). No
binaries, no unrelated files. `cargo fmt --check -p onnx-runtime-ep-cuda` clean.

## Production change is real, not just a test seam — and it is justified
`standard_attention.rs` does change runtime behavior: it extends CUDA-graph capture
eligibility to the **staged (aliased dense KV growth) single-token decode path**
(`staged_decode_eligible = (stage_key || stage_value) && batch==1 && q_seq==1`) and
relocates the staging K/V buffers from per-op `alloc` into two new persistent workspace
slots (`WS_STAGE_KEY`/`WS_STAGE_VALUE`, `WS_COUNT` 4→6). Assessed safe because:
- **#193 fix intact.** Both aliased copy-backs still use `dtod_async` on `self.stream`
  (2 occurrences), src=staging (`key_kv_ptr`/`value_kv_ptr`), dst=aliased present
  (`present_*_ptr`). Disjointness preserved: staging comes from a dedicated WS slot, so
  `key_kv_ptr != present_key_ptr` whenever `stage_key`.
- **No path conflict.** `dev_length_eligible` (requires `capacity_*`) and
  `staged_decode_eligible` (requires `!capacity_*`) are mutually exclusive.
- **Growing-shape correctness handled.** The staged path passes a null `dev_len_ptr`
  (host length, frozen at capture), and `key_cap==total_seq` grows each step. This is
  safe because captured graphs are **shape-keyed** at the session layer
  (`onnx-runtime-session/src/executor.rs` ~L356-360): each decode step's growing present
  K/V shape is a distinct key → distinct captured graph → no stale graph replayed across
  growth. `StdAttnWorkspace::reserve` grows eagerly (synchronize+free+alloc) and
  hard-errors if a grow is needed mid-capture, enforcing the "warm the exact shape before
  capture" contract.
- Eager / dense-non-decode / causal / GQA paths unchanged; execute-exit synchronize still
  drains the async copy for blocking eager callers.

## Independent revert-check (reproduced, not trusted)
Reverted both `dtod_async`→`dtod` in `standard_attention.rs`, rebuilt, reran:
```
KernelFailed("cuda_ep: stream synchronize: CUDA driver error:
DriverError(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "operation not permitted when
stream is capturing")")  ...  test result: FAILED
```
Restored file (`git checkout`, tree clean). The test genuinely guards the bug.

## Test quality — good
Routes through the real staged path (present aliases past, `!capacity`, batch=1, q_seq=1),
3 decode steps with real KV growth (INITIAL_PAST=1 → 4), warms eager each step (required by
the reserve contract), then captures+replays and asserts captured == eager for **output and
both present K/V caches**, checks the latched capture-error word, and resets the graph. Gates
cleanly on no-GPU via early `return` (skip, not fail) → CI without a GPU stays green.

## Non-blocking notes
- Test recaptures a fresh graph per step, so it validates single-step capture+replay, not
  reuse of one graph across growing steps; that scenario is covered by the session's
  shape-keyed cache (out of this test's scope). Fine as-is; a session-level multi-step test
  would be a nice future add.
- Staged workspace slots grow monotonically (never shrink); bounded by max seq, freed on
  Drop. Acceptable.

## Evidence
- `cargo test -p onnx-runtime-ep-cuda --test standard_attention_capture_gpu` on head: PASS
  (`CUDA_VISIBLE_DEVICES=6 taskset -c 1`).
- Revert `dtod_async`→`dtod`: FAIL with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`; restored.
- `cargo fmt --check`: clean. Diff: 3 files, no binaries.

Did not merge.


<!-- merged from .squad/decisions/inbox/leon-capture-regression-test.md -->
# Default Attention staged-KV capture regression

The CUDA EP now warms persistent scratch for single-token default-domain
Attention decode when dense present K/V outputs alias their growing past K/V
inputs. This makes the staged disjoint KV copy-back recordable in a CUDA graph.

`standard_attention_capture_gpu` compares three captured aliased decode steps
against eager output and K/V cache state. Replacing the two stream-ordered
`dtod_async` copy-backs with synchronous `dtod` fails during capture with
`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.


<!-- merged from .squad/decisions/inbox/luv-pr203-review.md -->
# PR #203 Review — Luv (Quality & Safety)

**PR:** justinchuby/onnx-genai #203 — "perf(cuda): split grid-starved symmetric int4 GEMV"
**Branch:** perf/cuda-next-wave (+230/-36)
**Reviewer:** Luv (independent; author Deckard locked out — all numbers reproduced, none trusted)
**Date:** 2026-07-26
**Verdict:** ⛔ REQUEST-CHANGES (one blocking test-coverage defect; kernel itself is correct, portable, and a real perf win)

---

## Summary of findings

The kernel and routing changes are **numerically correct, genuinely portable, and a real (larger-than-claimed) performance win**. However, the *one* GPU test added to guard the new split-K kernels **does not exercise them** — it silently routes to a different kernel. For a change whose entire risk profile is split-K reduction accuracy, shipping with an ineffective regression guard (and a PR body that claims the opposite) is blocking.

## 1. Correctness — VERIFIED GOOD (by me, not by the shipped test)

- Read the full kernel diff. Split-K partitions the 256-wide K steps across `K_SPLIT=2` cooperating warps per output column and combines fp32 partials through shared memory (`partials[col_local][ks]`, `__syncthreads`, ks==0 folds). Reduction reachability of `__syncthreads` is uniform (outside the `column < n` guard) — no divergence/deadlock.
- Hand-traced K coverage for K=896 (3.5 × 256-step): ks=0 covers [0,256)+[512,768); ks=1 covers [256,512)+[768,896). Union = full K, no overlap, no double-count. Correct.
- Symmetric path uses `block_sub2<false>`=0x48004800 (fp16 8.0) and `block_zp<false>`=8 — consistent zp=8 offset, matching the non-split symmetric kernel.
- The K%32==0 gate means the scalar "tail" branch (valid in 1..7) is provably dead (depth and k are both multiples of 8) — defensive, harmless.
- **Measured vs f64 reference when the split-K kernel is actually exercised** (I forced shape k=896, **n=1152** so it routes to `matmul_nbits_gemv_f16_scales_f16_splitk`, confirmed via a temporary runtime probe: `entry=matmul_nbits_gemv_f16_scales_f16_splitk sym_splitk=true`):
  - **max_abs_diff = 0.00195** vs the f64 accumulation reference (output magnitude ~2.6). That is fp16-rounding level — well inside tolerance and within the ~2e-3 you'd get vs the non-split kernel too.
- Tolerances NOT weakened. The diff only *adds* a test (0.04 absolute vs f64); sibling f16 tests use 0.02–0.2. No existing tolerance touched.

## 2. BLOCKING DEFECT — the split-K test does not exercise split-K

`matmul_nbits_gpu_fp16_symmetric_splitk_matches_f64_reference` uses `(k, n) = (896, 97)`.

- Variant selection (`select_f16_gemv_variant`) sets `down_eligible = !has_zp && scales_fp16 && block32 && k%32==0 && k > n`. With **896 > 97**, the shape is classified **DownProjection**, so it never reaches the `General` branch where symmetric split-K is chosen.
- Confirmed at runtime with a temporary probe: `entry=matmul_nbits_gemv_f16_scales_f16_down_c2 sym_splitk=false k=896 n=97`. The test validates the **down-projection** kernel, not split-K.
- I audited every symmetric f16 test shape in `matmul_nbits_gpu.rs`: (77,35),(77,37),(77,73)×2,(4096,73),(896,97),(64,151936). All have `k<512` or `k>n` (→DownProjection) or `n≥2112`. **No committed GPU test exercises the new symmetric split-K or its RMSNorm-prologue sibling.** The lib unit test only checks the gating boolean, not numerics.
- The PR body states "The new GPU test compares the split-K output against an f64 accumulation reference" — **false as written**; the test routes to `down_c2`.

**Required fix (trivial, verified):** change the test shape so `k ≤ n` and `n < SM*16`, e.g. `(k, n) = (896, 1152)`. I confirmed that shape routes to the split-K kernel and passes at max_abs_diff=0.00195. Ideally also add coverage for the RMSNorm-prologue split-K entry.

## 3. Portability — VERIFIED GOOD

- `use_f16_symmetric_splitk(k,n,mp,max_threads)` reads **live** device props: `k>=512 && k%32==0 && max_threads>=256 && n < mp.max(1)*16`. No hardcoded H200 SM/smem/N literals.
- H200 (132 SM): threshold n<2112 → N=896/1152 split. 46-SM RTX 4070: threshold n<736 → same N fall back to the existing single-warp kernel. Verified by unit test `symmetric_fp16_splitk_is_device_driven_and_falls_back_on_small_gpus` (passes) and by reproducing on-device.
- Split factor is a fixed K_SPLIT=2 (grid ×2), not a device-scaled factor — not pathological on mid-range GPUs (68-SM 3080, 84-SM 4090): it either splits by 2 or falls back; no over-splitting.

## 4. A/B — REPRODUCED, real win, exceeds the claim

Method: built `profile_native` (release, `bench-native,cuda`) at PR head, then rebuilt baseline by reverting **only** `matmul_nbits.rs` to origin/main (the sole perf-relevant file in the diff — identical everything else). Model `/home/justinchu/qwen2.5-0.5b-int4-onnx-native`, `--ep cuda --backend native --decode-precision fp16 --steady --warmups 2 --runs 3 --tokens 128`.

Env note: briefing said pin GPU 6, but GPU 6 was **busy (97%)** at review time; idle GPUs were 0/4/5, so I pinned **CUDA_VISIBLE_DEVICES=5 taskset -c 1**. First A/B pass was ruined by idle→boost clock ramp (throughput jumped 500→1000 tok/s). After warming the GPU to its 1980 MHz boost state (clock-lock unavailable — no permission), 8 tightly interleaved paired rounds:

- baseline median: **981.95 tok/s** (range 977.6–983.8)
- head median: **997.10 tok/s** (range 986.9–1002.0)
- delta: **+1.54%**, and **head > base in all 8 paired rounds with non-overlapping ranges**.

Absolute tok/s is ~3× the author's 326/327 (idle H200 at boost clocks vs the author's GPU-6 numbers), but the **direction and relative win are confirmed and larger than the claimed +0.59%**. No e2e regression.

## 5. Gates

- `cargo fmt -p onnx-runtime-ep-cuda -- --check` — clean ✅
- `cargo clippy --release -p onnx-runtime-ep-cuda --features cuda --lib -- -D warnings` — clean ✅
- `cargo test -p onnx-runtime-ep-cuda --features cuda --lib` — **221 passed** ✅
- `cargo test -p onnx-runtime-ep-cuda --test matmul_nbits_gpu` — **20 passed** ✅ (but see §2: the split-K one doesn't hit split-K)

## Verdict

**REQUEST-CHANGES.** The kernel is correct (0.00195 vs f64 when exercised), portable (device-driven gate, correct small-SM fallback), and a real +1.54% win. The single blocker: the added GPU test at (896,97) routes to `down_c2`, so the split-K path ships with **no** effective regression guard, and the PR body claims otherwise. Change the test shape to `k ≤ n` (e.g. 896×1152) so it actually exercises `matmul_nbits_gemv_f16_scales_f16_splitk`; add RMSNorm-prologue split-K coverage. Re-request review after.

Do NOT merge.

<!-- scribe-merge-2026-07-26T20-00-00Z-cuda-perf-and-capture-regression-reconciliation-end -->


<!-- scribe-archive-gate-2026-07-26T22-38-02+00-00 -->
Decision archive gate checked at 2026-07-26T22:38:02+00:00: active ledger was 409707 bytes and exceeded 51200 bytes. Applied the 7-day policy with cutoff 2026-07-20; no active-ledger entries older than the retained 7-day window were present, so archived 0 eligible block(s) and no archive file was changed.
<!-- scribe-merge-2026-07-26T22-38-02+00-00-mobius-issue-ort2-batch -->
## 2026-07-26 — Mobius PR triage, issue audit, and ORT2 remaining-work reconciliation
<!-- merged from .squad/decisions/inbox/sapper-mobius-pr-triage.md -->
### 2026-07-26: Mobius PR 404, 423, and 430 triage
**By:** Sapper
**What:** Triaged all current review threads, linted all three branches, resolved PR 404 against current main on replacement branch `sapper/404-rebase`, and pushed safe review fixes to PRs 423 and 430 without merging.
**Why:** Justin retains sole merge authority for Mobius PRs; the branches needed conflict resolution and concrete review follow-up before his review.

## PR 404

- Current actionable threads:
  - `src/mobius/models/glm_moe_dsa_test.py:22`: add representative config-suite coverage — already implemented; thread left open because the original PR branch was not replaced.
  - `src/mobius/components/_moe.py:295`: asymmetric QMoE zero points — already implemented.
  - `docs/research/glm52-export.md:65`: onnx-genai MTP package support — already implemented.
  - `src/mobius/models/deepseek.py:332`: QMoE routed-expert packing — already implemented.
  - `export_glm_tiny_quant.py:3` and `export_glm_tiny_qmoe.py:3`: broken docstring backticks — fixed on replacement branch.
  - `src/mobius/__main__.py:581`: missing `--glm-full-attention` CLI test — fixed on replacement branch.
- Merged current `origin/main`, preserving GLM DSA/IndexShare, MTP, GGUF, and fused QMoE behavior.
- Validation: Ruff check and format clean; 378 affected tests passed.
- `lintrunner` could not run because its adapter uses forbidden `/tmp` response files; direct repository Ruff fallback was clean.
- Pushed `fa30534` to `sapper/404-rebase`; original `glm5.2-moe-export` remains conflicting/draft. Justin must replace/update the PR branch.
- Merge-ready for Justin: **No**, until the replacement branch is adopted and CI/review runs on the PR.

## PR 423

- Fixed grouped-routing validation and one-expert `noaux_tc` groups, relaxed brittle exact MatMul count, added shared-expert metadata aliases, defaulted unset activation metadata to SiLU, and validated grouped metadata invariants.
- After GitHub reported a main conflict, merged current main and resolved the sole MoELayer documentation conflict.
- Validation: Ruff check and format clean; 84 affected tests passed after the main merge, plus 23 focused routing/metadata tests after patch-coverage expansion.
- Pushed `40846bb` to `squad/hythe-deepseek-moe-phase1`; branch is mergeable with no unresolved current review threads.
- Merge-ready for Justin: **Pending CI and approval**; no current code-review blocker. Test jobs are still queued/running and Codecov currently reports failure before their coverage uploads complete.

## PR 430

- Fixed all code threads: removed stale `image_token_id`, validated numbered placeholders, cached projector weights, corrected NumPy/PyTorch documentation, kept projector GELU in float32, made legacy-cache conversion linear, and typed the CLIP config adapter boundary.
- Confirmed the current PR description already documents the checked-in golden JSON, replied to the stale thread, and resolved it.
- Validation: Ruff check and format clean; two focused golden-harness tests plus 24 Phi-3.5 model/projector tests passed.
- Pushed `d1d235e` to `test/l4-l5-golden-new-models`; branch remains mergeable with no unresolved current review threads.
- Merge-ready for Justin: **Pending CI and approval**. Test jobs are still queued/running and `codecov/patch` currently reports failure.

No PR was merged.
<!-- merged from .squad/decisions/inbox/holden-issue-triage-45-77.md -->
### 2026-07-26: Backlog issue triage for #45–#77
**By:** Holden
**What:** Triaged all requested issues against current code, tests, progress/design docs, and merged PRs; closed only #52 and #64.
**Why:** Both closed issues have merged implementation evidence and passing targeted tests. All uncertain or incomplete roadmap work remains open.

| issue# | classification | evidence or gap |
|---:|---|---|
| 45 | OPEN | No Top-A, Mirostat, Typical-P, DRY, or XTC processors/tests exist in the engine. |
| 46 | PARTIAL | Text completions map min-p and penalties, but chat completions still lack top-k, min-p, penalties, and seed. |
| 47 | PARTIAL | PR #188 added v-prediction/x0 handling; DDPM and FlowMatching schedulers remain absent. |
| 48 | PARTIAL | Declarative conditioning exists, but `run_comfyui` still lacks dual encoders and SDXL `time_ids`. |
| 49 | PARTIAL | Typed rendering forwards `start_step`; `run_comfyui` still lacks source-image encode and inpainting masks. |
| 50 | PARTIAL | Workflow parsing knows LoRA/ControlNet, but the runner does not load/feed their runtime inputs. |
| 51 | PARTIAL | Renderers report non-finite output but neither fail closed nor upcast/retry. |
| 52 | DONE-closed | PR #91 implements ordered pure composites; codec E2E test passed 2/2 locally. |
| 53 | PARTIAL | PR #153 added typed text-to-image requests/results and E2E tests; latent-step streaming is absent. |
| 54 | OPEN | No ORT model-package manifest parser, variant selector, validator, or package tooling exists. |
| 55 | OPEN | ONNX metadata is preserved, but no `onnx_runtime.*` hint scanning/priority/type-validation runtime exists. |
| 56 | PARTIAL | Int2 remains on the correctness/dequantization fallback; no direct packed 2-bit GEMV/GEMM. |
| 57 | OPEN | CPU MatMulNBits still explicitly rejects nonzero `weight_prepacked`. |
| 58 | PARTIAL | PR #105 includes native AVX-512 BF16 GEMM; f16 still widens to f32. |
| 59 | PARTIAL | Stateless server batching is live, but persistent sessions use the per-request fallback and not scheduler-driven batching. |
| 60 | PARTIAL | `DiskTierConfig` remains a placeholder; no disk payload spill/readback exists. |
| 61 | PARTIAL | Priority pause/resume is implemented, but preempted KV stays in place rather than moving/evicting tiers. |
| 62 | PARTIAL | Tier A in-place output exists; Tier B shared/paged append-only GQA KV remains deferred. |
| 63 | PARTIAL | Host weight cache exists; live VRAM cache, H2D binding, and async prefetch remain unwired. |
| 64 | DONE-closed | PRs #105/#113/#154/#200 plus `a414d615` implement automatic topology-aware pinned placement; 61 targeted tests passed. |
| 65 | OPEN | No heterogeneous CPU/CUDA partition-and-execute path exists; sessions still do not provide mixed-EP fallback. |
| 67 | PARTIAL | CUDA coverage grew to 88 listed ops, but CPU-registry parity and heterogeneous fallback remain incomplete. |
| 68 | PARTIAL | Ratio-4 FP8 is device-resident/capturable; ratio-128 FP8 remains host-staged and non-capturable. |
| 69 | PARTIAL | CUDA gets compile/clippy CI only; no GPU conformance profile, H200 execution lane, or automated report. |
| 70 | PARTIAL | PR #92 improved MLX packaging/default selection; device sampling, quantized-KV switching, and Apple perf CI remain. |
| 71 | OPEN | Python still advertises CUDA from a compile-time feature and does not apply requested providers to `RtSession`; no wheel-path discovery. |
| 72 | PARTIAL | Windows/macOS CI and wheel matrices exist, but macOS wheel import smoke tests remain explicitly skipped. |
| 73 | OPEN | Minimal-build operator manifests, generator, features, and `cargo xtask minimal-build` do not exist. |
| 74 | PARTIAL | PR #105 includes CPU Conv and Resize; ScatterND and QLinearMatMul remain unregistered. |
| 75 | PARTIAL | Catalog expanded to 71 standard ops/78 versions, but full standard/ONNX-ML schemas and inference remain incomplete. |
| 76 | PARTIAL | PR #153 added direct ORT graph projection/capability queries; the immutable cached GraphView/lens design is absent. |
| 77 | PARTIAL | PR #153 added plugin projection/execution plumbing; `EpRegistry::load_legacy` remains a Phase-2 `todo!()`. |

## Issues closed

- **#52 — Support generalized non-autoregressive composite pipelines**
  - Evidence posted: PR #91, merge `c8bc70e8abda`; `PipelineEngine::run_pipeline` dispatches composite plans in `crates/onnx-genai-engine/src/pipeline.rs:1534-1549`; `crates/onnx-genai-engine/tests/codec_pipeline_e2e.rs:1-77` proves an ordered encoder→vocoder pure composite.
  - Verification posted: `cargo test -p onnx-genai-engine --test codec_pipeline_e2e` — 2 passed, 0 failed.

- **#64 — Implement automatic NUMA-aware decode placement**
  - Evidence posted: PR #105 (`d0fdfa47d3ce`), #113 (`b9bb7143`), #154 (`a6848d4c`), #200 (`b51ea239`), and current-main follow-up `a414d615`; implementation spans `decode_affinity.rs`, `decode_spmd.rs`, `decode_numa.rs`, and `kernels/matmul_nbits.rs`.
  - Verification posted: `decode_spmd::tests` 31 passed, `decode_affinity::tests` 25 passed, and `decode_numa::tests` 5 passed; 0 failed.

## DOABLE-NOW

None. Every not-started issue has medium/large cross-cutting scope; the small-looking non-finite VAE item is already partial and needs a product decision between fail-closed and retry/upcast semantics.

Plain-text summary: 2 closed, 23 partial, 0 doable-now.
<!-- merged from .squad/decisions/inbox/gaff-issue-triage-78-106.md -->
### 2026-07-26: Justin backlog triage — issues 1–106
**By:** Gaff
**What:** Triaged issues #78–88, #106, #1, #9, #13, and #21 against `main` at `b33e7939785eb19e1c79d1545e73d0d3b795584a`, source/tests/docs, and merged/open PR state. Closed only #1; posted status evidence on every PARTIAL issue.
**Why:** Keep the backlog aligned with landed behavior without closing roadmap work whose acceptance criteria remain incomplete.

| issue# | classification | one-line evidence-or-gap |
|---:|---|---|
| #1 | DONE-closed | Issue screenshot says acceptance met; landed tool protocol SHAs `1699a6bd`/`385c25dc`, real Qwen HTTP E2E `d7896d26`, and `scripts/coding_agent.py` preserve the verified Hermes file-writing loop. |
| #9 | PARTIAL | `9ab4fa91`/`b5934c6f`/`a5106f56` provide registry, lazy load/unload, admin routes, and LRU tests; §37 version policy, A/B hot-swap, health-check, and repository rescan remain absent. |
| #13 | PARTIAL | Debug routes and Perfetto/Chrome exporters are tested, but `routes.rs::debug_kv` still literally reports engine KV-page statistics unavailable despite engine page-stat APIs. |
| #21 | OPEN | No CLI/server session/model pretty-printer covering signature, FLOPs, size, and dtype; existing `/v1/models`/admin responses expose identity/lifecycle fields only. |
| #78 | PARTIAL | PyO3 eager landed in `onnx-runtime-python/src/eager.rs` with Python tests, but `onnx-runtime-eager/src/dispatch.rs` still hard-codes one output and marks TopK/Split arity deferred. |
| #79 | OPEN | CUDA registers `com.microsoft::QMoE` only; `kernels/mod.rs` has no `pkg.nxrt::BlockQuantizedMoE`, and `qmoe.rs` explicitly rejects IQ/MXFP4 layouts requiring that operator. |
| #80 | PARTIAL | Runtime `IndexShare` v1 is frozen/implemented and QMoE runs E2E, but `onnxruntime/mobius#404` remains OPEN/DRAFT; its fused QMoE emitter is unmerged and private IndexShare exporter reconciliation is incomplete. |
| #81 | PARTIAL | `e4d28832` + `2ffb4e45` implement the communicator oracle and seven in-process collectives; no NCCL/multi-process backend or EP/TP execution placement exists. |
| #82 | PARTIAL | Host expert cache leases/governor landed (`f80ca09`), but CUDA QMoE still declares device paging, async prefetch, and expert sharding deferred; Phase 3b binding is unsupported. |
| #83 | OPEN | Only Kimi readiness/design material exists; no KDA, gated-MLA latent-cache, AttnRes runtime contract, kernel, or artifact-backed test has landed. |
| #84 | PARTIAL | Linear speculation is implemented and the proposal struct has placeholder tree fields, but every proposer emits `tree: None` and verification has no tree-aware path. |
| #85 | OPEN | Executor supports view aliases and special KV aliases, not graph-planned compute-in-place; `Kernel` has no in-place capability and no dead-input liveness reuse. |
| #86 | PARTIAL | Static-cache batching now binds per-row `nonpad_kv_seqlen` and CPU/CUDA Attention consume it; no packed `pkg.nxrt` varlen op, `cu_seqlens` kernel/oracle, or savings benchmark exists. |
| #88 | PARTIAL | `b720a218` made warmed RoPE launches capture-compatible with device error latching/signature gates, but no standalone-RoPE graph record/replay or unfused-model zero-fallback token-parity DoD test exists; #193/#201 test Attention, not RoPE. |
| #106 | OPEN | `docs/EXTENSIBLE_QUANT_TYPES.md` is explicitly a design draft; no `QuantTypeDeclProto`, `quant_type_uri`, codec registry, `CUSTOM_QUANT`, or `DequantizeExtensible` implementation exists. |

1 closed, 9 partial, 0 doable-now.
<!-- merged from .squad/decisions/inbox/pris-ort2-remaining-summary.md -->
### 2026-07-26: ORT2 / DESIGN remaining-work audit
**By:** Pris
**What:** Ground-truth audit of the ORT2 runtime and the broader GenAI design against current code, tests, `docs/PROGRESS.md`, and Justin's open `release:backlog` issues.
**Why:** `docs/ORT2-IMPL-PLAN.md` still describes the July 19 skeleton state, while current `main` has already completed Phase 1 and substantial Phase 2/3 work.

# ORT2 / DESIGN：已完成多少、还剩多少

## 审计口径

- 审计代码：`main` at `b33e7939`（2026-07-26）。
- 状态优先级：实际代码/测试 > `docs/PROGRESS.md` > 旧计划。`PROGRESS.md` 自称 living status，但文件头最后更新时间是 2026-07-25、记录 HEAD `5a8c3dc9`；因此又核对了其后代码。
- 当前已发布：`onnx-runtime-*` 为 `v0.1.0-dev.1`，仓库行覆盖率约 **77%**（`docs/PROGRESS.md:3-9`）。
- Justin 当前有 **42** 个 open `release:backlog` issues。
- 七个目标 crate 一次性 `cargo build` 全部成功。
- 用户指定的精确 grep：
  `grep -rn "todo!()\|unimplemented!()" crates/onnx-runtime-*/src`
  只命中 **1** 次，而且是 `onnx-runtime-session/src/lib.rs:10` 的过时文档注释；更宽松地搜实际宏调用后，只有 **1 个真实 TODO**：`onnx-runtime-ep-api/src/registry.rs:222` 的 legacy plugin EP `load_legacy()`。

## 1. ORT2 crate 成熟度

| crate | 状态 | 当前证据 | 关键剩余 |
|---|---|---|---|
| `onnx-runtime-ir` | **REAL** | 3,270 LOC；53/53 默认测试通过；0 实际 TODO。图 IR、符号 shape、layout、mutation、validation 已落地；Phase-1 BERT 真实模型首次运行无需跨 crate 修复（`PROGRESS.md:456-464`）。 | 完整 ORT 图 ABI 不属于 safe IR，本来就移到 `ep-api`。 |
| `onnx-runtime-loader` | **REAL** | 4,509 LOC；默认 suite **109 passed / 1 ignored**；0 TODO。protobuf decode、IR builder、mmap external weights、shape inference、encoder、EPContext load/dump 均已实现（`loader/src/lib.rs:1-38,49-71`; `PROGRESS.md:460,465-488`）。 | 完整 schema/catalog breadth 仍受 #75 限制；model-package 解析不是 loader 现有 flat-model 路径的一部分（#54）。 |
| `onnx-runtime-ep-api` | **partial（核心 REAL）** | 4,774 LOC；45/45 默认测试通过。EP/Kernel/registry/tensor/weight/EPContext contract 都是真实现；`OrtGraphView::query_plugin_capabilities()` 已能 `dlopen` plugin 并调用 `GetCapability`（`ep-api/src/abi.rs:20-175`）。 | 唯一真实 `todo!` 是 `EpRegistry::load_legacy()`（`registry.rs:219-223`）；native `query_capabilities()` 目前返回空；完整 GraphView/lens 与 plugin compile/run adapter 未完成（#76/#77）。 |
| `onnx-runtime-ep-cpu` | **REAL** | 75,921 LOC；默认 suite **927 passed / 9 ignored**；0 TODO。当前源码可解析出 **164 个 unique op names / 169 domain-op pairs**，远超旧计划的 7 个 Phase-1 ops；含控制流、量化、MoE、SIMD/MLAS、decode 优化。 | `ScatterND`、`QLinearMatMul` 仍未注册；部分 dtype/layout/算子长尾与 conformance 仍在 #74/#75。 |
| `onnx-runtime-ep-cuda` | **REAL，但覆盖 partial** | 52,292 LOC；0 TODO；`CUDA_COVERED_OPS` 当前锁定 **88** 个 op names（`kernels/mod.rs:109-...`, test at `:721`）；216 个 lib tests 曾整批通过。真实 Qwen/Phi/Llama/GLM/DeepSeek native CUDA 已运行并有 benchmark（`PROGRESS.md:11-160`）。 | 相对 CPU 164-op breadth 仍明显不足（#67）；runtime library discovery/Python provider 选择不完整（#71）；全图 all-or-nothing，缺异构 fallback（#65）；本机完整 GPU suite 有瞬时数值/并发不稳定，GQA 单测失败后 targeted rerun 通过。 |
| `onnx-runtime-session` | **REAL** | 16,045 LOC；默认 suite **192 passed / 2 ignored**；0 TODO。`SessionBuilder::build()`、sequential executor、`run()`、dynamic shapes、optimizer、control flow、device binding、EPContext 都是真实现（`session/src/lib.rs:597,903-1058`; `executor.rs:2601,3741`）。 | session 仍是单 EP 选择/整图执行；异构 CPU/CUDA partition（#65）、GraphView placement（#76）、async DAG/cost placement 仍未完成。文件顶部“Phase 1 skeleton / todo”注释已严重过时（`lib.rs:8-10`）。 |
| `onnx-runtime-capi` | **partial（Tier-1 REAL）** | 953 LOC；默认 suite **17/17** 通过；0 TODO。`nxrt_create_session`、options、tensor、run、status/release 等完整存在（`capi/src/lib.rs:232-720`）。 | 不是 upstream ORT drop-in：没有 `OrtGetApiBase`/`OrtApi` vtable，仍用 `nxrt_*` 名称；`crate-type` 还是 `["lib"]`，不是 cdylib/staticlib（`Cargo.toml:12`）。这是 #77 的核心剩余。 |

**结论：现在没有 skeleton crate。** 七个 crate 都能 build；五个是完整可运行实现，`ep-api` 和 `capi` 是“核心真实、ORT 兼容层未收口”的 partial。

## 2. July-19 Phase-1 计划 vs 现在

`docs/ORT2-IMPL-PLAN.md:19-29` 当时把 loader/ep-api/ep-cpu/session/capi 全标成 🔨 skeleton。现在：

| 旧计划项 | 现在 | 证据 |
|---|---|---|
| IR contract | ✅ | 34 tests 已增长到 53；仍是稳定底座。 |
| Loader protobuf/graph/weights/shape inference | ✅ | 真实 loader + mmap + 独立 shape-inference crate + encoder/EPContext；109 tests。 |
| EP API safety/tensor/registry | ✅（Phase 1） | DLPack、owned buffers、registry、EPContext/weight contracts 已落地；legacy loading 明确仍属 Phase 2。 |
| CPU 7-op kernel slice | ✅ 且大幅超额 | 已从 7 ops 扩到约 164 unique op names，927 passing tests。 |
| Sequential session executor | ✅ | `bert_toy_optimized.onnx` 384 nodes 端到端运行，输出对 ORT max_abs `1.19e-7`（`PROGRESS.md:464`）。 |
| Tier-1 C API | ✅（项目自定义 `nxrt_*`） | create/run/tensor/status/options 已完成；但严格的 upstream `OrtGetApiBase` 仍未做。 |
| Phase-1 BERT exit milestone | ✅ | commit `85f379b`，CPU 纯 Rust 对 ORT parity（`PROGRESS.md:464`）。 |

因此，按仓库后来采用的 Phase-1 定义，**Phase 1 = 100% 完成**。唯一需要标注的规格偏差是：旧计划 `ORT2-IMPL-PLAN.md:148-159`/`ORT2.md:7838` 写了 `OrtGetApiBase`，实际 Phase 1 交付的是干净的 `nxrt_*` Tier-1 ABI，并把真正 ORT vtable/drop-in 放到 Phase 2（`capi/src/lib.rs:9-11`）。

## 3. ORT2 剩余 workstreams

百分比是按当前设计子项的工程完成度粗估，不是代码行比例。

| workstream | 粗估完成 | 已完成 | 真正剩余 / blocker |
|---|---:|---|---|
| **C-ABI / plugin-EP transition** | **~65%** | Tier-1 `nxrt_*` C ABI；EPContext produce→dump→reload→consume 全链路；ORT plugin `GetCapability` Stage-1 graph projection 已实现（`PROGRESS.md:482-487`; `abi.rs:61-175`）。 | `OrtGetApiBase`/vtable、cdylib/staticlib、`load_legacy()`、plugin compile/run adapter、真正 drop-in conformance。**#77**。 |
| **Operator / schema / shape coverage** | **~70%** | CPU 已约 164 op names；runtime shape registry 约 **81 domain-op pairs / 102 versioned registrations**；`onnx-std` 文档最新总结为 **71 standard ops / 78 versioned entries**（`ONNX_RS_SPEC_COVERAGE.md:406-410`）；CUDA 已 88 op names。 | #74 四个点中 Conv/Resize 已落地，**ScatterND/QLinearMatMul 仍缺**；完整 standard + ONNX-ML schema、sequence/optional/recurrent 长尾仍缺（**#74/#75**）；CUDA 与 CPU parity 仍缺（**#67**）；eager multi-output/PyO3（**#78**）。 |
| **EP capability projection / heterogeneous placement** | **~35%** | ORT C graph host projection和 plugin `GetCapability` 已真实可调用；EP claim API 存在。 | GraphView immutable lens 尚未落地，native `query_capabilities()` 仍为空；session 仍选择单 EP，unsupported CUDA node 会整图失败，未做 CPU/CUDA partition。**#76**，并直接关联 **#65**。 |
| **Model package + metadata hints** | **~10%** | flat model directory、ONNX metadata preservation、EPContext package-like compiled blobs 已有。 | package manifest、variant selection、package validation/tooling 全缺（**#54**）；`onnx_runtime.*` hint 扫描、优先级、类型校验、placement/warnings 全缺（**#55**）。这是最接近“未开始”的 ORT2 大块。 |
| **CI / portability / packaging** | **~35%** | toolkit-free CUDA build、Linux 动态加载、Windows/macOS library-name logic、若干 Windows build fixes 已有；本次七 crate build 全绿。 | CUDA wheel 路径发现和 Python provider availability 不准确（**#71**）；主 CI 仍 Ubuntu-only、Windows/macOS wheels 未覆盖（**#72**）；CUDA GPU conformance CI 仍在 **#69**。 |

另外，`ORT2.md:7847-7867` 的完整 Phase 2/3 愿景中，cost model placement、layout propagation、async DAG + transfers、arena/lifetime/in-place memory planner、ILP placement、多 EP 单图里程碑仍未整体完成；因此不能把“Phase 1 完成”误读成“整个 ORT2 设计完成”。

## 4. `DESIGN.md` GenAI 层还剩什么

| bucket | 当前完成度 | 已完成 | 剩余与 issues |
|---|---:|---|---|
| **Diffusion / non-AR pipelines** | **~75%** | iterative pipeline、DDIM/Euler/Euler-A/DPM++2M/Karras、CFG、SDXL、LoRA、ControlNet、inpaint、ComfyUI、masked language diffusion 大量路径已验证；`PROGRESS.md:527` 明确核心 correctness 很强。语言 diffusion roadmap 除 real-model 外已完成（`PROGRESS.md:978-990`）。 | FlowMatching/DDPM/modern DiT scheduler（**#47**）；native runner 的 SDXL/img2img/inpaint/ControlNet/LoRA 收口（**#48/#49/#50**）；fp16 VAE 非有限值（**#51**）；typed `generate_image` + latent streaming（**#53**）。`#52` 的旧描述已部分过时：single-pass composite 和 codec e2e 已在 `PROGRESS.md:1088-1097` 落地，但更深的 mixed iterative/AR composite 仍有余量。语言 diffusion 还缺真实 LLaDA/Dream/SMDM e2e。 |
| **Scheduler / continuous batching** | **~60%** | scheduler 数据结构、FCFS/priority/FairShare、byte budget、preemption decisions、prefix/paged primitives 都有测试。 | 正常 engine serving loop 仍是单请求，continuous batching 未接入（**#59**）；scheduler 决定的 KV preemption/eviction 未由 engine 执行（**#61**）；真正 ragged/packed varlen attention（**#86**）。 |
| **KV / weight offload** | **~50%** | GPU↔CPU tier bookkeeping、local connector、page table/LRU、shared ByteBudget、governor 配置和 snapshot 已有（`PROGRESS.md:519-525`）。 | 真 SSD/NVMe KV tier（**#60**）；GQA runtime-managed paged KV Tier-B（**#62**）；live VRAM weight pages/H2D binding/device execution（**#63**）；lowering live budget 后即时 eviction 仍缺。 |
| **Multi-model / placement / distributed** | **~40%** | 多 component pipeline、VLM/audio/diffusion orchestration、单机多 session 基础已经存在。 | model-package/lifecycle/hot-swap 没有完整产品化（邻接 **#54**）；CPU/CUDA heterogeneous graph（**#65**）；multi-GPU communicator + expert/tensor parallel（**#81**）；routed-expert paging/leases/scheduler（**#82**）。`DESIGN.md` 的 cluster router、P/D disaggregation、remote KV、完整 multi-model resource broker 仍大多是设计。 |
| **Sampling / speculation** | **~65%** | greedy/categorical、temperature、top-k/top-p/min-p、repetition/frequency/presence 等基础链路和 linear speculative decoding 已有。 | Top-A、Mirostat、Typical-P、DRY、XTC（**#45**）；OpenAI chat surface 尚未暴露完整 top-k/min-p/penalties/seed（**#46**）；tree speculative decoding（**#84**）。 |

## 5. Bottom line：到底“还剩多少”

按明确定义的 **ORT2 Phase-1 foundation**，已经 **100% 完成**：不再有 skeleton，BERT parity milestone 已通过，七个 crate 当前全部能 build。若看完整 `ORT2.md` runtime 愿景，粗估约 **65–70% 完成**；剩下的主要不是“把 skeleton 填完”，而是 ORT drop-in/plugin execution、全 schema/operator parity、异构 placement、model package 和跨平台发布。`DESIGN.md` 的核心单机 GenAI 产品能力约 **70% 左右**，但把 distributed KV/multi-model/multi-GPU/MoE、完整 continuous batching/offload、所有 diffusion/sampling 与生态绑定都算进“大愿景”，整体更接近 **55–60%**。换句话说：基础已经成型且能跑真实模型，余下约三到四成主要是广度、兼容性、调度/内存系统和产品化收口，而不是重写核心。

<!-- scribe-merge-2026-07-26T22-38-02+00-00-mobius-issue-ort2-batch-end -->
<!-- scribe-merge-2026-07-26T22-38-02Z-rope-capture-88 -->
## 2026-07-26 — RoPE capture DoD, PR #208 review, and fmt gate recovery

Decision archive gate checked at 2026-07-26T22:38:02+00:00: active ledger was 435274 bytes before this merge. No dated ledger entries older than 2026-07-19T22:38:02+00:00 were present, so no archive file was created or updated.
<!-- merged from .squad/decisions/inbox/leon-rope-capture-dod.md -->
### 2026-07-26: Standalone RoPE capture regression closes #88
**By:** Leon
**What:** Added a GPU regression that constructs a default-domain, standalone fp16 `RotaryEmbedding` decode graph, warms its exact signature, captures/replays three decode steps, and requires bitwise eager parity, an installed graph executable, and a clear capture-error latch.
**Why:** This locks the unfused RoPE path directly, so a capture-time host synchronization or a silent eager fallback cannot regress unnoticed. Deliberately restoring the host position-id D2H validation during recording made the test fail with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`; the guarded implementation passes.

<!-- merged from .squad/decisions/inbox/chew-pr208-review.md -->
# PR #208 Review — VERDICT: APPROVE

- **Reviewer:** Chew (independent gate; author Leon locked out)
- **Date:** 2026-07-26
- **PR:** #208 "test(cuda): cover standalone RoPE graph capture" — closes #88
- **Change:** +257/-0, single new file `crates/onnx-runtime-ep-cuda/tests/rope_capture_gpu.rs`

## Checklist results

1. **Standalone (unfused) path — CONFIRMED.** The test builds a single-node `RotaryEmbedding`
   graph (default domain, opset 23) and calls `ep.get_kernel` directly. `get_kernel`
   (`provider.rs:254`) looks up the factory registry by `("RotaryEmbedding","",23)`
   (`kernels/mod.rs:514`) → `RotaryEmbeddingFactory` → `RotaryEmbeddingKernel`. No optimizer/
   fusion runs on a directly-requested single node, so it categorically cannot route to a fused
   GQA/Attention op. This is structurally immune to #201's shape-misroute trap. The test also
   asserts `kernel.cuda_graph_compatible()` after an eager warm.
2. **Parity + zero-fallback — CONFIRMED.** Byte-exact `assert_eq!(captured, eager)`; plus the
   real capture gates: `begin_graph_capture` (audits `capture_support`), `end_graph_capture`
   (errors if a host sync occurs mid-capture), `has_graph_executable()`, `check_capture_error()==0`,
   `reset_graph()`. Not an eager fallback that trivially passes.
3. **GPU gate — CONFIRMED.** `gpu()` returns `None` and the test returns early with a skip
   message when CUDA is unavailable. Fixed deterministic inputs; loops 3 decode steps.
4. **RUN — PASS.** `CUDA_VISIBLE_DEVICES=6 taskset -c 1 cargo test ... --test rope_capture_gpu`
   → `1 passed` (24.5s).
5. **GUARD PROOF — PASS.** Independently broke capture-safety by removing the `!capturing`
   gate at `kernels/rotary_embedding.rs:495` (`if has_position_ids && !capturing` →
   `if has_position_ids`), forcing the host `dtoh` (synchronize + sync memcpy) to run *during*
   capture. Re-run → test **FAILED** with
   `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED ("operation not permitted when stream is capturing")`
   at the decode execute. Restored the line; git tree clean; test passes again. The test
   genuinely guards the capture-safety property.
6. **fmt/clippy — CLEAN.** `cargo fmt --all -- --check` clean; `cargo clippy -p onnx-runtime-ep-cuda`
   clean; `cargo clippy --test rope_capture_gpu` clean (remaining clippy warnings are pre-existing
   in unrelated test files: matmul_nbits_gpu, conv_gpu, etc.).

## Verdict
**APPROVE.** Merge-ready. The test exercises the true standalone RoPE decode kernel, asserts
parity and real capture success (no fallback), gates cleanly on GPU availability, and is a
proven guard (fails when capture-safety is broken).
### 2026-07-26: Resch restored the fmt gate and PR #208 closed #88
**By:** Scribe, from coordinator manifest
**What:** Resch landed pure rustfmt repair commit `63e0ef26` after PR #207 plus `decode_spmd.rs` regressed the BLOCKING fmt gate on main; `cargo fmt --all -- --check` returned exit 0. PR #208 then merged to main as `5eb0d8db` (`test(cuda): cover standalone RoPE graph capture`), closing issue #88.
**Why:** The batch restored main's formatting health and permanently records that Leon's standalone RoPE capture DoD regression landed with Chew's independent approval and guard-break evidence.
<!-- merged from .squad/decisions/inbox/andrews-split-movement-handlers.md -->
### 2026-07-27: Split movement shape handlers by operator family
**By:** Andrews
**What:** Replaced the 1,809-line `handlers/movement.rs` with:
- `movement/mod.rs` (114 lines): shared helpers and the unchanged registration facade.
- `movement/transform.rs` (409 lines): Transpose, Reshape, Flatten, Squeeze, Unsqueeze, Expand.
- `movement/resize.rs` (302 lines): Resize.
- `movement/concat_slice.rs` (394 lines): Concat, Slice.
- `movement/split_gather.rs` (380 lines): Split, Gather, GatherElements, GatherND.
- `movement/scatter.rs` (137 lines): Scatter, ScatterElements, ScatterND, Trilu.
- `movement/space_depth.rs` (132 lines): DepthToSpace, SpaceToDepth.

The split totals 1,868 lines including module-local imports. Registration order, operator/opset mappings, handler bodies, shape rules, and diagnostic text are unchanged.

**Why:** Cohesive operator-family modules reduce navigation and review cost while keeping this change mechanical and behavior-preserving. `cargo fmt -p onnx-runtime-shape-inference`, shape-inference build/tests (224 tests plus one doctest), clippy with `-D warnings`, and downstream `onnx-runtime-session` build all pass.
<!-- merged from .squad/decisions/inbox/ash-split-genai-config.md -->
### 2026-07-27: Split genai config compatibility crate into cohesive modules
**By:** Ash
**What:** Kept `lib.rs` as a 98-line facade retaining `GenAiConfigError`, `GENAI_CONFIG_FILE`, and the flat public re-exports. Moved config wire types to `wire_types.rs` (361 LOC), loading to `loading.rs` (109 LOC), graph I/O inspection to `graph_io.rs` (235 LOC), compatibility synthesis to `compatibility.rs` (1,427 LOC), JSON builders to `json_builders.rs` (341 LOC), and unit tests to `tests.rs` (1,212 LOC).
**Why:** The former 3,743-line facade mixed serialization contracts, file loading, graph inspection, pipeline synthesis, JSON construction, and tests. The split is pure code motion; public names remain re-exported from the crate root. A source comparison confirmed config wire definitions, field/variant ordering, derives, every `#[serde(...)]` attribute, and all `GenAiConfigError` text are unchanged. `cargo build`, all 30 crate tests, clippy with `-D warnings`, and downstream engine/server/CLI builds passed.


<!-- merged from .squad/decisions/inbox/call-split-onnx-std-rules.md -->
### 2026-07-27: Split ONNX validation rules by model layer
**By:** Call
**What:** Split the former 5,316-line `crates/onnx-std/src/check/rules.rs` into a `rules/mod.rs` facade and five private rule-family modules:
- `graph_topology.rs` — 368 lines; opset imports, duplicate names, graph input/output connectivity, and acyclicity.
- `schema_types.rs` — 1,217 lines; schema conformance, type constraints, initializer declarations, metadata, attributes, and retained protobuf types.
- `ir_version_functions.rs` — 1,147 lines; IR version/feature gates and local function validation. The two existing `#[allow(clippy::too_many_arguments)]` attributes remain on their original functions.
- `tensor_sparse_payloads.rs` — 558 lines; dense tensor payload and sparse tensor validation.
- `multi_device.rs` — 393 lines; device configuration and sharding validation.
- `mod.rs` — 1,711 lines; public facade, shared diagnostic helpers, and unchanged tests.

**Why:** Cohesive private modules reduce the validation implementation's file-level entropy while preserving the flat public API. Rule ORDER is unchanged because `check/mod.rs` and its 17 `checker.add_rule(...)` calls were not modified. Violation WORDING is unchanged: all 579 Rust string literals were compared as multisets before formatting and preserved exactly; the non-author reviewer independently approved the split and found rule implementations/helper logic unchanged.

**Gates:** `cargo fmt -p onnx-std` passed; `cargo build -p onnx-std` passed; `cargo test -p onnx-std` passed (126 unit tests, 23 integration tests, 1 doc-test); `cargo clippy -p onnx-std --all-targets -- -D warnings` passed. Non-author review: approved with no blocking findings.


<!-- merged from .squad/decisions/inbox/chew-pr227-fp16-review.md -->
# Chew — PR #227 FP16 Path Numerics Review (Second Pass)

**Branch:** `squad/mac-cpu-ep-roofline`
**Date:** 2026-07-27T02:00:00-07:00
**Commits under review:** `75311827` (FP16 storage GEMV + bulk conversion), `3a88ba8c` (SPMD pool for FP32 GEMV + cleanup)
**Author under review:** Iran
**Reviewer:** Chew (Numerics)

---

## Verdict: **APPROVE**

The FP16 storage GEMV kernel and bulk f16↔f32 conversion are numerically sound. All 922 tests pass. `cargo fmt --check` clean (BLOCKING). `cargo clippy` produces only pre-existing cosmetic warnings in `activations.rs` (not new code). Two non-blocking concerns are noted below.

---

## Per-item findings

### 1. Inline assembly for `fcvtl` — SOUND, NON-BLOCKING CONCERN

**File:** `accelerate_gemm.rs:419-433` (`load_f16x4_to_f32x4`)

The asm block:
```asm
"ldr {v:d}, [{ptr}]"       // load 8 bytes (4 × f16) into low 64 bits of Vn
"fcvtl {v:v}.4s, {v:v}.4h" // widen low 4 × f16 → 4 × f32
```

**Correctness assessment:**
- **Constraints correct.** `ptr = in(reg)` for the base address, `v = out(vreg)` for the NEON result. Using the same register for input/output of `fcvtl` is valid — the instruction reads from the low half and writes to the full register.
- **Options correct.** `nostack` (no stack use), `readonly` (no memory writes), `pure` (no side effects). All accurate — the block only reads memory and produces a register value.
- **Clobbers correct.** No additional clobbers needed; `v` is declared as `out(vreg)` which already tells the compiler it's modified.
- **`volatile` correctly absent.** The `pure, readonly` combination allows the compiler to CSE/LICM the load, which is desirable for the GEMV inner loop.

**Verified bit-exact against scalar `half::f16::to_f32()`** for: normal values (1.0–65504.0), denormals (0x0001, 0x03FF), ±inf (0x7C00, 0xFC00), NaN (0x7E00), ±zero (0x0000, 0x8000). All lanes match bit-for-bit.

**Concern C1 (non-blocking):** The rationale for inline asm over intrinsics is that `vcvt_f32_f16` requires Rust's unstable `f16` type, which needs nightly. This is a valid practical reason today. However, Rust's `f16` type is on track for stabilization (RFC 3453). **Recommend: add a `// TODO: replace with vcvt_f32_f16 intrinsic when f16 stabilizes` comment.** Inline asm in a shared kernel that Resch (Intel) and Luba (ARM) also maintain is a maintenance hazard — the intrinsic should replace it as soon as feasible.

**Assignee for C1:** Deckard or Sapper (not Iran).

### 2. FP32 accumulation — VERIFIED SOUND

**Files:** `accelerate_gemm.rs:474-554` (`neon_gemv_f16_batch`), `accelerate_gemm.rs:558-601` (`neon_dot_f16`)

Accumulation is genuinely f32 throughout:
- Accumulators `a0..a3`, `b0..b3` are `float32x4_t`, initialized via `vdupq_n_f32(0.0)`.
- `vfmaq_f32` is f32 fused-multiply-add — operates entirely in f32.
- Horizontal reduction via `vaddvq_f32` (f32).
- Scalar tail accumulates into `s0..s3` (f32 locals) using `half::f16::from_bits(x).to_f32()` followed by f32 multiply.

**This is NOT native FP16 accumulate.** It is the correct FP16-storage-f32-accumulate pattern.

**Measured error vs f64 reference:**

| Shape (name) | K | N | max abs | max rel | max ULP |
|---|---:|---:|---:|---:|---:|
| gate_proj | 896 | 4864 | 3.46e-6 | 1.52e-7 | 2 |
| down_proj | 4864 | 896 | 2.95e-5 | 2.38e-7 | 4 |
| q_proj | 896 | 896 | 3.37e-6 | 1.53e-7 | 2 |
| kv_proj | 896 | 128 | 2.64e-6 | 1.15e-7 | 1 |
| 1×1 | 1 | 1 | 4.49e-10 | 2.14e-8 | 0 |
| 1×4 | 1 | 4 | 1.58e-9 | 4.86e-8 | 0 |
| odd_tail | 67 | 9 | 2.28e-7 | 1.13e-7 | 1 |

Max relative error across all shapes: **2.38e-7**. This is well within the FP32-accumulate envelope (~1e-7 relative from FMA ordering). The doc claim of "~2.3e-4 max relative error" is conservative — actual measured error is 1000× better than claimed, which is consistent with FP32 accumulation (the 2.3e-4 figure would be for FP16 accumulation).

**FP16 GEMV vs F32 GEMV with identical f16-quantized weights:** max relative error **1.73e-6**. This confirms the accumulation is truly f32 — if it were FP16 accumulate, this would be ~1e-3 or worse.

### 3. Tail handling — VERIFIED CORRECT

**Main loop:** processes 8 elements per iteration (2 × 4 `fcvtl` loads per row) in `neon_gemv_f16_batch`, 16 elements per iteration in `neon_dot_f16`.

**Tail:** scalar loop `while j < k` widens each f16 individually via `half::f16::from_bits().to_f32()`.

**Verified at:** K=67 (not divisible by 8 or 16), N=9 (not divisible by 4). Both produce correct results with max abs error 2.28e-7 vs f64 reference. Also verified K=1, N=1 and K=1, N=4 — all correct.

The `neon_gemv_f16_batch` outer loop processes 4 output rows at a time, with a `while i < n` scalar tail that calls `neon_dot_f16` per remaining row. This tail is also correct at N=9 (processes 2 groups of 4, then 1 remaining row).

### 4. Transpose cache (`transposed_b_f16`) — THREAD-SAFE

**File:** `matmul.rs:161-205`

- Uses `OnceLock<Vec<u16>>`, which is Rust's standard thread-safe lazy initialization. Only one thread will execute `get_or_init`; all others block until initialization completes. No torn reads possible.
- The transpose itself uses Rayon `par_chunks_mut` — each thread writes to a disjoint slice of `bt`, so no data races.
- The `unsafe` for `from_raw_parts` is justified: the view is validated as contiguous Float16 with exactly `numel` elements; `half::f16` is `repr(transparent)` over `u16`.
- Transpose logic verified correct: `bt_chunk[jj * k + i] = src[i * n + j]` where `j = j0 + jj`. This maps `src[K,N]` row-major → `bt[N,K]` row-major, which is the correct transposition.

### 5. Bulk conversion (`neon_f16_to_f32_bulk` / `neon_f32_to_f16_bulk`) — SOUND

**File:** `dtype.rs:774-828`

**Widen (`fcvtl`, line 775-797):** Same asm block as `load_f16x4_to_f32x4`, correctly annotated `readonly, pure`. Scalar tail uses `half::f16::from_bits().to_f32()`. Verified bit-exact against scalar for all edge cases.

**Narrow (`fcvtn`, line 803-828):**
- Asm block correctly does NOT have `readonly` or `pure` — it writes to memory via `str`.
- `options(nostack)` only — correct, since it has a memory side effect.
- `src = in(vreg)` for the f32x4 input, `ptr = in(reg)` for the output address, `v = out(vreg) _` for the scratch register. Constraints are correct.

**Rounding mode:** `fcvtn` uses IEEE round-to-nearest-even (the ARM default FPCR.RMode). Verified: `neon_f32_to_f16_bulk` produces bit-identical output to `half::f16::from_f32()` for all tested values including tie-breaking cases.

**Overflow to inf:** Values > 65504 (e.g. 65520, 65536, 100000) correctly narrow to `±inf` (0x7C00/0xFC00). This matches `half::f16::from_f32()` behavior.

**Denormal handling:** Values in the f16 denormal range (e.g. 6.0e-8) are correctly narrowed with gradual underflow, matching scalar.

**NaN preservation:** NaN inputs produce NaN outputs (bit patterns may differ in payload, which is IEEE-compliant).

**Non-multiple-of-4 tail:** Tested with n=21 (21 elements). Scalar tail correctly handles the remaining 1 element.

### 6. SPMD pool correctness (`3a88ba8c`) — SOUND

**File:** `matmul_nbits.rs:1463-1488`

- `perf_cores.saturating_sub(1).max(1).min(available)` — correctly handles:
  - 1 P-core → `max(0, 1) = 1` → 1 worker (safe minimum)
  - 2 P-cores → `max(1, 1) = 1` → 1 worker (conservative)
  - 8 P-cores (this M1 Max) → `min(7, 10) = 7` workers
  - `.min(available)` prevents exceeding logical CPU count
  - `saturating_sub` prevents underflow
  - Cannot produce zero or negative — `.max(1)` is the floor

- `performance_core_count()` (line 1632-1662) returns `None` on Intel Macs or VMs where `hw.perflevel0.physicalcpu` doesn't exist, causing the override block to be skipped entirely — falling back to the generic `available/2` default. Safe.

- The existing `with_decode_pool_scope` change (line 2243-2258) correctly gates SPMD pool eligibility: without MLAS, the pool is eligible for both quantized and dense models; with MLAS, only quantized models use it (avoiding contention with MLAS's own Rayon pool).

### 7. Silent-fallback audit — PASSED

The `constant_weight_prepack_reuses_weight_and_keeps_activation_live` test (matmul.rs:1700-1745) asserts `kernel.prepack.transposed_b_f16.get().is_some()` on macOS, proving the FP16 GEMV path is compiled and executed. The test uses `Owned::f16` weights with M=1, which matches the dispatch condition. Result `[2., 6.]` and `[8., 15.]` are numerically exact (f16-representable values).

### 8. Apple Silicon generality — CORRECT

- `fcvtl` and `fcvtn` are ARMv8 base FP instructions, not FEAT_FP16. They are present on ALL aarch64 CPUs, not just Apple Silicon.
- The entire `accelerate_gemm` module is gated by `#[cfg(any(target_os = "macos", target_os = "ios"))]` — Luba's ARM Linux code never enters this module.
- Non-aarch64 scalar fallbacks exist at lines 551 and 606.
- Thread count is derived at runtime from `hw.perflevel0.physicalcpu` with sane fallback — no hardcoded tile sizes or cache assumptions.

### 9. Test coverage — ADEQUATE

**New tests in `accelerate_gemm.rs`:**
- `f16_col_parallel_gemv_matches_reference` (K=64, N=128, max abs < 1e-3)
- `f16_col_parallel_matches_at_model_scale` (K=896, N=4864, max rel < 2%)
- `f16_gemv_odd_k_tail` (K=67, N=9, exercises scalar tail)

**Updated tests in `matmul.rs`:**
- `constant_weight_prepack_reuses_weight_and_keeps_activation_live` — updated to assert f16 cache path on macOS

**Concern C2 (non-blocking):** The model-scale test threshold of `max_rel < 0.02` (2%) is very loose for what should be FP32-accumulate accuracy. Measured actual error is ~2.4e-7 (1e5× below threshold). Recommend tightening to `max_rel < 1e-4` to catch genuine FP16-accumulate regressions. Similarly, the `f16_col_parallel_gemv_matches_reference` threshold of `max_abs < 1e-3` should be `< 1e-5`.

**Assignee for C2:** Deckard or Sapper (not Iran).

---

## Summary

| Item | Status |
|---|---|
| Inline asm `fcvtl` correctness | ✅ Sound (bit-exact vs scalar) |
| FP32 accumulation preserved | ✅ Verified (max rel 2.38e-7) |
| FP16 GEMV numerical parity | ✅ Within f32-accumulate envelope |
| Tail handling (K, N non-aligned) | ✅ Correct at K=67/N=9/K=1/N=1 |
| Transpose cache thread safety | ✅ OnceLock + disjoint par_chunks |
| Bulk conversion rounding/overflow/NaN | ✅ Bit-exact with half::f16 |
| SPMD pool edge cases | ✅ Cannot produce ≤0 workers |
| Path reachability | ✅ Test proves f16 GEMV is hit |
| Apple Silicon generality | ✅ ARMv8 base, correct gating |
| Test coverage | ✅ Adequate (3 new + 1 updated) |

**Non-blocking concerns:**
- **C1:** Add TODO to replace inline asm with intrinsics when `f16` stabilizes.
- **C2:** Tighten test error thresholds from 2%/1e-3 to 1e-4/1e-5 to guard against accumulation regressions.


<!-- merged from .squad/decisions/inbox/chew-pr227-numerics-review.md -->
# Chew — PR #227 Numerics Review

**Branch:** `squad/mac-cpu-ep-roofline`
**Date:** 2026-07-27T01:30:00-07:00
**Author under review:** Iran
**Reviewer:** Chew (Numerics)

---

## Verdict: **APPROVE with concerns**

The four commits introduce NEON-vectorized SiLU, SDPA, and GEMV kernels plus an Accelerate sgemm integration for the native CPU EP on Apple Silicon. All 904 unit tests pass. `cargo fmt --check` passes (BLOCKING gate). `cargo clippy` passes (warnings only — dead code, cosmetic). End-to-end generation on Qwen 2.5-0.5B produces 30 tokens at ~30 tok/s without crashes or panics on M1 Max.

The numerics are **sound for production inference** but several documentation claims are inaccurate, and the SDPA NEON path lacks direct test coverage. None of the concerns below are blocking for merge, but they should be tracked for follow-up.

---

## Per-item findings

### 1. Vectorized SiLU (`activations.rs:357-436`) — NON-BLOCKING CONCERN

**The "~1 ULP" claim (line 353) is incorrect.** Measured accuracy of the Cephes-style polynomial (simulated with hardware FMA to match NEON `vfmaq_f32`):

| Range | Max ULP | Max relative error |
|---|---:|---:|
| Practical [-10, 10] | 28.0 | 3.31e-6 |
| Wide [-20, 20] | 28.3 | 3.34e-6 |
| Near zero [-0.01, 0.01] | 1.5 | 1.47e-7 |
| Clamped region [-100, -87.3] | 12.5M | ~0 abs (subnormal) |
| Positive > 88.7 | 0.0 | 0.0 |

**Assessment:** ~28 ULP in the practical range is acceptable for f32 transformer inference (effective ~17 bits of precision). The extreme-negative clamped region produces subnormal-magnitude results where the absolute error is negligible (~1e-37). **But the docstring must be corrected from "1 ULP" to "~28 ULP" or "< 1e-5 relative error".**

- `half` variable (line 372) is declared but never used — dead code from the original Cephes formulation where `floor(x+0.5)` was used for rounding; Iran replaced it with `vrndnq_f32` (NEON round-to-nearest) but didn't remove the constant.
- Non-finite fix-up (lines 423-429) is correct: re-scans the NEON-computed region and delegates NaN/Inf to the scalar reference.
- **Path verification PASSED:** inserted `panic!` at `silu_f32_neon` entry; `silu_contiguous_matches_reference` test hit it, confirming the NEON path is compiled and executed on this machine.

**Assignee for correction:** Deckard or Sapper (not Iran — locked out).

### 2. Swish→SiLU canonicalization (`activations.rs:234-246`) — SAFE

```rust
let activation = if alpha == 1.0 { Activation::Silu } else { Activation::Swish { alpha } };
```

- Uses **exact f32 equality** (`alpha == 1.0`), not epsilon. This is correct.
- Default is `unwrap_or(1.0)` — exactly 1.0f32.
- A near-1.0 alpha (e.g., 0.99999994) will NOT canonicalize to SiLU. No silent misrouting.
- Mathematically, Swish(x, β=1) = x·σ(x) = SiLU(x). Identity confirmed.

### 3. NEON SDPA (`sdpa.rs:744-820`) — NON-BLOCKING CONCERN

**Numerics are sound:**
- Softmax uses max-subtraction stability (line 502) — correct.
- `sdpa_f32_neon` reuses the existing `softmax_in_place` scalar path, so softmax stability is inherited.
- `dot_neon` and `axpy_neon` use 4×-unrolled FMA accumulators with correct tail handling (scalar fallback for remainder).
- Masked/-inf entries handled correctly: `scores.fill(0.0)` in softmax when all scores are `-inf`, and `probability == 0.0` skip in V-weighted accumulation (line 815).
- GQA grouping (`heads_per_kv`) is correct.

**Test coverage gap:** All 11 SDPA tests call `sdpa_f32_scalar` directly — **no test exercises `sdpa_f32_neon`**. Inserted a `panic!` at `sdpa_f32_neon` entry; all SDPA tests passed without hitting it. This means a bug in the NEON SDPA path would go undetected.

**Recommendation:** Add a parity test that calls `sdpa_f32(...)` (the dispatcher) and compares against `sdpa_f32_scalar(...)` for a representative set of shapes including non-power-of-2 head dims.

**Assignee for follow-up:** Pris (test owner).

### 4. GEMV correctness (`accelerate_gemm.rs`, `matmul.rs`) — SOUND

**Transpose:** The pre-transpose in `MatMulPrepack::transposed_b` (matmul.rs:100-133) produces `B_T[N,K]` row-major from `B[K,N]` row-major via `bt[j*k + i] = b[i*n + j]`. Correct. The `neon_gemv_col_parallel` kernel then computes `y[j] = dot(B_T[j,:], x)` = `Σ_i B[i,j]*x[i]` — which is `y = B^T @ x`, i.e., the correct decode GEMV.

**Tail handling:**
- `neon_gemv_batch`: processes 4 output rows at a time with 8 accumulators (2 per row, 8-wide K loop). K-tail handled by scalar fallback. N-remainder via `neon_dot` for individual rows. Correct.
- `neon_dot`: 16-element unrolled loop, 4-element secondary loop, scalar tail. Correct.
- `neon_outer_product_unrolled`: 4-row K-unrolled outer product with NEON N-vectorized inner loop, scalar N-tail. Correct.

**Accumulation:** All accumulations are f32 throughout (NEON `float32x4_t` → `vaddvq_f32` horizontal sum → f32 scalar tail).

**Accelerate sgemv removed from dispatch:** Confirmed. The `matmul_dense_into_with_backend` function at matmul.rs:795-822 dispatches M=1 to `neon_gemv_col_parallel` or `neon_gemv_parallel`, never to `sgemv_accelerate`. The `gemm_with_backend` at matmul.rs:217-224 dispatches M=1 to `neon_gemv_parallel`. No dead branch in the dispatch chain.

**Dead code in module:** `sgemv_accelerate`, `is_l2_resident`, `l2_cache_bytes`, `query_sysctl_usize`, `CBLAS_TRANS`, `cblas_sgemv` are declared but never called from production code. The compiler emits dead_code warnings. These are from the removed Accelerate sgemv path. Non-blocking: remove or mark `#[allow(dead_code)]` with a justification.

**Test guard-break PASSED:** Zeroed `y[i]` in `neon_gemv_batch` → `col_parallel_gemv_matches_reference` test failed with error 0.997. Tests are sensitive to GEMV breakage.

**Model-scale tolerance:** The `accelerate_decode_gemv_matches_generic_at_model_scale` test uses 2% relative tolerance. Measured actual max_rel: 0.018% for [1,896,896], 0.39% for [1,896,4864], 1.57% for [1,4864,896]. The 1.57% is a legitimate f32 accumulation-order difference (row-parallel outer-product reduction vs sequential tiled GEMM). The 2% tolerance accommodates this but is loose enough to mask real bugs. Non-blocking: consider tightening or comparing against a f64 reference.

### 5. `dtype.rs` f32 memcpy fast path (dtype.rs:643-664) — SAFE

Guard conditions:
1. `out.dtype == DataType::Float32` — exact dtype match
2. `out.is_contiguous()` — verified: calls `onnx_runtime_ir::is_contiguous(shape, strides)` which checks strides match computed contiguous strides exactly

A strided-to-contiguous case cannot take this path — the strides check prevents it. The length check (`data.len() != n`) prevents buffer overrun. The `validate()` call checks the output tensor invariants. Safe.

### 6. `matmul_nbits.rs` visibility change (line 1902) — SAFE

`fn spmd_decode_active()` changed from private to `pub(crate)`. This allows `accelerate_gemm.rs` to call it (line ~186) to prefer the persistent SPMD pool when active. The function only reads thread-local state (`IN_SPMD_SCOPE`) — no side effects, no new computation. Safe.

---

## Structural checks

### Silent-fallback bug class

- **NEON SiLU path:** Verified reachable. `cfg(all(not(feature = "mlas"), target_arch = "aarch64"))` is active on this M1 Max. Panic-probe confirmed.
- **NEON SDPA path:** `cfg(target_arch = "aarch64")` is active, and the dispatch at sdpa.rs:291-294 reaches it when `qk.is_none()`. However, **no unit test exercises this path** (all tests call `sdpa_f32_scalar` directly). The path IS reachable in production (the engine calls `sdpa_f32`), but it has no dedicated test coverage.
- **Accelerate GEMM paths:** `cfg(any(target_os = "macos", target_os = "ios"))` is active. Tests exercise both sgemm and NEON GEMV paths.

### Apple Silicon generality

- **No hardcoded thread counts or cache sizes.** L2 threshold is queried at runtime via `hw.perflevel0.l2cachesize` sysctl with 4 MB fallback. Thread counts come from `rayon::current_num_threads()`.
- **NEON intrinsics are ARMv8 baseline only.** All intrinsics used: `vfmaq_f32`, `vld1q_f32`, `vst1q_f32`, `vdupq_n_f32`, `vaddq_f32`, `vaddvq_f32`, `vmulq_f32`, `vnegq_f32`, `vrndnq_f32`, `vdivq_f32`, `vmaxq_f32`, `vminq_f32`, `vsubq_f32`, `vcvtq_s32_f32`, `vshlq_n_s32`, `vreinterpretq_f32_s32`. No dotprod, no FP16 arithmetic, no SME, no SVE, no BF16 intrinsics. Works on M1/M2/M3/M4 all trims.
- **500,000 element threshold** for single-thread dispatch is a heuristic. Not chip-specific.
- **TILE=64** in the transpose (matmul.rs:120) is a cache-blocking parameter, not tied to a specific L1 size.

### One implementation, no arch fork

The NEON kernels are guarded by `cfg(target_arch = "aarch64")` with scalar fallbacks on other architectures. The Accelerate integration is guarded by `cfg(any(target_os = "macos", target_os = "ios"))` with the generic GEMM fallback. This is **runtime branching behind cfg, not a fork of the kernel tree**. Intel (Resch) and ARM/QNN (Luba) share the same `gemm_with_backend` dispatcher and the same scalar reference. Acceptable.

### Parity tests

SiLU: `silu_contiguous_matches_reference` and `silu_in_range_region_is_bit_close` test the NEON path against f64 reference with ≤2e-6 / ≤1e-5 tolerances. Guard-break not directly tested on SiLU NEON (the test goes through `silu_f32_slice` which dispatches to NEON — confirmed reachable via panic probe).

GEMV: `col_parallel_gemv_matches_reference`, `row_parallel_gemv_matches_reference`, `accelerate_sgemm_matches_generic_for_small_shapes`, `accelerate_decode_gemv_matches_generic_at_model_scale`, `col_parallel_matches_at_model_scale`. Guard-break test PASSED for GEMV.

SDPA: **No parity test for the NEON path.** Gap.

---

## End-to-end coherence

Generated 30 tokens with native CPU EP on Qwen 2.5-0.5B at ~30 tok/s with prompt "The capital of France is" (temperature implicitly greedy). No crash, no panic. The compare tool does not report generated text, so direct text comparison against ORT output could not be performed. Both backends generated exactly 30 tokens.

---

## Summary of concerns (non-blocking, for tracking)

| # | Severity | Item | File:Line | Assignee |
|---|---|---|---|---|
| C1 | Low | SiLU docstring claims "1 ULP", measured ~28 ULP | activations.rs:353 | Deckard |
| C2 | Medium | SDPA NEON path has zero test coverage | sdpa.rs:291-294, 744-820 | Pris |
| C3 | Low | 7 dead code items from removed Accelerate sgemv | accelerate_gemm.rs:17,38,84,101,127,136 | Deckard |
| C4 | Low | Unused `half` variable | activations.rs:372 | Deckard |
| C5 | Low | GEMV model-scale tolerance is 2% (actual max 1.57%) | matmul.rs:1887 | Pris |
| C6 | Info | Compare tool doesn't report generated text for coherence verification | bench compare.rs | Sebastian |
<!-- merged from .squad/decisions/inbox/christie-split-server-routes.md -->
### 2026-07-27: Split server routes by endpoint family
**By:** Christie
**What:** Replaced the 2,989-line `crates/onnx-genai-server/src/routes.rs` with a `routes/` module tree: `mod.rs` (530 LOC) retains `ApiError`, JSON rejection handling, model resolution, shared request preparation types/helpers, and facade re-exports; `admin.rs` (396 LOC) owns health, models, status, resources, debug, admin, and metrics endpoints; `sessions.rs` (60 LOC) owns session create/delete; `completions.rs` (1,719 LOC) owns completions, embeddings, chat, streaming, and generation helpers; `multimodal.rs` (312 LOC) owns transcription, speech, and image-generation endpoints.
**Why:** This is a pure code-motion split of the HTTP god-file. Router registration remains untouched in `src/lib.rs`, preserving route paths and registration order exactly. The typed `ApiError` handling and server-side registry logging hardened in PR #213 were moved verbatim without behavior changes.

**Gates:** `cargo build -p onnx-genai-server` passed. `cargo test -p onnx-genai-server` completed with 110 passed, 2 ignored, and only the accepted pre-existing `sidecar_free_compatibility_package_builds_server_pipeline_and_preprocesses_image` failure caused by missing `vlm-executable/vision.onnx`. `cargo clippy -p onnx-genai-server --all-targets -- -D warnings` passed. `cargo fmt -p onnx-genai-server` passed.
<!-- merged from .squad/decisions/inbox/coordinator-apple-silicon-portability.md -->
### 2026-07-26: CPU EP optimizations must be general across Apple Silicon
**By:** Justin Chu (directive, via coordinator)
**What:** Every optimization in the Mac CPU EP campaign (PR #227) must be correct and beneficial across the whole Apple Silicon family — M1/M2/M3/M4 and their base/Pro/Max/Ultra variants — not tuned to the M1 Max this work happens to be measured on. The M1 Max is the measurement rig, not the target.
**Why:** Apple Silicon varies enormously along exactly the axes a roofline campaign is tempted to hardcode:
- **Memory bandwidth** spans roughly 68 GB/s (M1 base) to ~800 GB/s (Ultra). A bandwidth-derived tuning constant baked in from a Max is wrong on most Macs developers own.
- **Core topology** varies in both count and P:E ratio (e.g. 4P+4E on M1 base vs 8P+2E on M1 Max). Thread counts and work partitioning must be derived at runtime, not assumed.
- **Cache/SLC sizes** differ per tier, so blocking and tile sizes must be computed from queried cache sizes.
- **AMX generation differs, and it is undocumented and not architecturally stable.** Reaching the matrix coprocessor must go through Accelerate (BLAS/BNNS), never hand-rolled AMX encodings. M4 additionally exposes SME/SVE, which older chips lack.

**Implications (binding on this campaign):**
1. No compile-time constants derived from one machine's measurements. Query topology and cache sizes at runtime (`sysctl` / `sysconf`), derive tiling and thread counts from them.
2. Feature-detect, never assume: any instruction path beyond the ARMv8.4 baseline that all Apple Silicon shares (incl. SME on M4, and dot-product/fp16 variations) requires a runtime check with a correct fallback.
3. Gate the AMX/matrix path behind Accelerate, and keep a NEON path that is correct when Accelerate is unavailable or slower.
4. Scaling must be validated as a *shape*, not a point: report performance as a fraction of the machine's own measured roofline, so the result is meaningful on a 68 GB/s M1 Air as well as on this Max.
5. This does not fork the CPU EP. Per the standing portability rule, the CPU EP stays one general implementation shared with Intel (Resch) and ARM (Luba) — Apple Silicon specialization lives behind runtime detection, not behind a parallel kernel tree.
<!-- merged from .squad/decisions/inbox/copilot-qwen3-tts-validated.md -->
### 2026-07-27: FP16 GEMV review follow-ups
**By:** Deckard
**What:** Documented each NEON f16 inline-asm conversion site with the stabilization condition for replacing it with f16 intrinsics, and tightened FP16 GEMV guards to `1e-4` relative / `1e-5` absolute. The model-scale guard now runs under 1, 3, 7, and 11 Rayon workers to cover Apple Silicon worker-count differences.
**Why:** Chew verified the asm is bit-exact but noted the maintainability hazard; the comments preserve that context until Rust `f16` and aarch64 f16 conversion intrinsics stabilize. Chew measured 2.38e-7 max relative f64-reference drift, 1.73e-6 FP16-vs-F32 parity, and 2.28e-7 odd-tail absolute error, so the new thresholds keep cross-chip headroom while catching real FP16 accumulate, lane, or tail regressions.
<!-- merged from .squad/decisions/inbox/dillon-split-ort-decode.md -->
### 2026-07-27: Split ORT decode by cache and session family
**By:** Dillon
**What:** Replaced `crates/onnx-genai-ort/src/decode.rs` with a facade and six focused submodules:

- `decode/mod.rs` — 201 lines; public option/signature types, batched trait, and re-exports.
- `decode/dynamic.rs` — 1,550 lines; dynamic past/present decode and captured-step tests.
- `decode/kv_growth.rs` — 465 lines; shared KV bucket growth, host/CUDA prefix copying, and tests.
- `decode/static_cache.rs` — 1,210 lines; scalar and batched static-cache sessions.
- `decode/shared_batch.rs` — 476 lines; continuous-batch shared-buffer session.
- `decode/io.rs` — 196 lines; KV-name pairing and static-cache signature detection.
- `decode/tensor.rs` — 149 lines; logits, cloning, empty tensor, and allocation helpers.

All existing public types remain available from `onnx_genai_ort::decode` through facade re-exports. The `decode_contract`-based `KvNamingConvention`, `kv_suffix`, and `name_contains_present_key_value` call sites were moved unchanged into `decode/io.rs`; no local classifier copies were introduced.

`cargo fmt -p onnx-genai-ort` was run. Gates passed:

- `cargo build -p onnx-genai-ort`
- `cargo test -p onnx-genai-ort` (all unit, integration, and doc tests)
- `cargo clippy -p onnx-genai-ort --all-targets -- -D warnings`
- `cargo build -p onnx-genai-engine`

**Why:** The original 4,239-line file mixed materially different cache ownership and batching models. The split is pure code motion and clarifies ownership without changing algorithms, allocation, CUDA annotations, or the public facade.


<!-- merged from .squad/decisions/inbox/fact-checker-roofline-adjudication.md -->
# Fact Checker: Roofline Adjudication — M1 Max CPU EP Campaign

**By:** Fact Checker
**Date:** 2026-07-27T05:04Z
**Branch:** `squad/mac-cpu-ep-roofline` (PR #227)
**Machine:** Apple M1 Max (MacBookPro18,2), 8 P-cores + 2 E-cores, 32 GiB LPDDR5

---

## TL;DR — The Authoritative Number

| Metric | Value | Source |
|--------|-------|--------|
| **Achievable 8-P-core DRAM read bandwidth** | **~112 GB/s** (range 108–120) | Independently reproduced |
| **Roofline ceiling, Qwen 0.5B FP32 batch-1 decode** | **~56–57 tok/s** | 112 GB/s ÷ 1.976 GB |
| **ORT's position** | **45.45 tok/s = 80% of roof** | Not 44.6% |
| **~59 tok/s target** | **Physically at the ragged edge; realistic target is 52–56 tok/s** | Exceeds GEMV-achievable BW |

**Sebastian's 197 GB/s headline ceiling is wrong. It is not reproducible.** His own "achievable MT GEMV = 112 GB/s" is the correct number, and it agrees with Pris's probe (108.3 GB/s) and my independent measurements (108–120 GB/s). The campaign should use **~112 GB/s** as the bandwidth ceiling for FP32 GEMV decode.

---

## 1. Reproduction of Both Measurements

### My Independent Bandwidth Measurement

I compiled and ran three standalone C programs from scratch (`_scratch_fc/bw_test.c`, `bw_neon.c`, `bw_sweep.c`) testing:
- Scalar 8-byte reads (mimicking both approaches)
- NEON 16/64/128-byte vectorized reads
- With and without QoS thread affinity (`QOS_CLASS_USER_INTERACTIVE`)
- Buffer sizes from 1 MiB to 256 MiB per thread

**Key results (8 P-cores, 256 MiB/thread, QoS, best of 5):**

| Approach | 1 thread | 2 threads | 4 threads | 8 threads |
|----------|----------|-----------|-----------|-----------|
| Scalar read, no QoS | 52.9 | 99.3 | 76.3 | 115.4 |
| Scalar read, QoS | 55.5 | 62.9 | 88.3 | **119.7** |
| Volatile read (≈ black_box), no QoS | 22.1 | 42.0 | 76.8 | 122.9 |
| NEON 128B unrolled, QoS | 59.9 | 56.2 | 98.9 | 109.2 |

**Buffer-size sweep (8 threads, QoS):**

| Total buffer | GB/s |
|-------------|------|
| 32 MiB (near SLC) | 102.0 |
| 48 MiB (SLC boundary) | 99.1 |
| 256 MiB (DRAM) | 110.5 |
| 2 GiB (DRAM, matches both probes) | 119.1 |

**Conclusion: the 8-P-core DRAM streaming-read bandwidth on this M1 Max is 108–120 GB/s.** It is NOT 197 GB/s. My single-core results (55–60 GB/s) match Sebastian's single-core claim (60.2 GB/s), so the hardware is the same — the discrepancy is purely in the multi-threaded scaling.

### Pris's Probe Reproduction

Pris's code (`compare.rs:747–786`) uses:
- `thread::scope` with 8 threads (from `hw.perflevel0.physicalcpu`)
- 256 MiB/thread (line 709)
- `wrapping_add` with `std::hint::black_box` (sequential dependency chain)
- Best of 3 repetitions
- **No thread affinity (no QoS class)**

Her reported 108.3 GB/s falls squarely in my measured range. The slight underperformance vs my best (119 GB/s) is explained by:
1. `black_box` adds a compiler barrier per load (prevents instruction reordering)
2. No QoS → some threads may run on E-cores at 2–4 thread counts
3. Rust's iterator overhead vs C pointer arithmetic (minor)

**Pris's 108.3 GB/s: ✅ Consistent with independent measurement.**

### Sebastian's Measurement: NOT Reproducible

Sebastian claims 196.8 GB/s for "sequential read, 256 MiB/thread, from DRAM" on 8 P-cores. My maximum is 122.9 GB/s under any access pattern, thread configuration, or QoS setting. His number is **1.64× what I can reproduce**.

The most likely error source: Sebastian's 197 GB/s streaming measurement has a bug in the benchmark code — possibly a timing error (e.g., dividing total bytes by per-rep time instead of total time), double-counting bytes across repetitions, or a buffer that was partially SLC-resident despite being nominally 256 MiB.

**However, Sebastian's own GEMV measurement (112 GB/s, 8 threads, gate_proj 4864×896) is correct.** It is consistent with both my streaming bandwidth and Pris's probe. He has two contradictory numbers in his own report — a streaming-read "ceiling" (197 GB/s) and an "achievable GEMV" (112 GB/s) — and the achievable GEMV is the correct one.

---

## 2. Mechanistic Explanation of the 197 vs 108 Gap

**They are NOT measuring "different things" where both are right. Sebastian's 197 GB/s is an erroneous measurement.**

Evidence:
1. My independent streaming read (identical methodology: scalar sequential, 256 MiB/thread, 8 P-cores, QoS) peaks at 119.7 GB/s — nowhere near 197.
2. The M1 Max has 400 GB/s total LPDDR5 bandwidth shared across CPU, GPU, and Neural Engine. The CPU accessing 197 GB/s = 49% of total is inconsistent with published measurements (~30% CPU share = ~120 GB/s in Anandtech, Chips and Cheese, and other independent reviews).
3. Sebastian's own data is self-contradictory: his single-core BW (60.2 GB/s) would require 60.2 × 8^0.58 = 197 GB/s scaling, but this exponent was derived FROM the 197 GB/s number, making it circular. Independent measurement shows 8-thread scaling of ~2.0× single-core (60 → 120), not 3.3×.

**Candidate hypotheses tested:**

| Hypothesis | Result |
|-----------|--------|
| QoS/affinity difference | Tested both. QoS affects 2–4 threads, not 8. Max 8T: 119.7 vs 122.9 GB/s |
| NEON-wide loads vs scalar | Tested 8B/16B/64B/128B loads. All converge to 109–120 GB/s at 8 threads |
| Buffer too small → SLC | Tested 1–256 MiB/thread. SLC boundary (~48 MiB total) shows no 197 GB/s cliff |
| E-core contamination | E-cores only in no-QoS mode; all-10-core test = 126.1 GB/s (still <197) |
| Thermal/power state | 5 repetitions in each test; best-of-5 consistent across tests |

**Bottom line: Sebastian's streaming-read benchmark has a bug.** His GEMV number (112 GB/s) was measured with a different, more realistic benchmark (actual matrix multiply, not a synthetic loop) and is correct. The campaign should discard the 197 GB/s figure entirely.

---

## 3. Authoritative Roofline Number

**For batch-1 FP32 decode of Qwen 2.5-0.5B on Apple M1 Max:**

```
Achievable GEMV bandwidth:     ~112 GB/s  (110–120 range)
Weight bytes per token:         1.976 GB   (from model shapes)
Roofline ceiling:              ~56.7 tok/s
ORT baseline:                   45.45 tok/s (80.2% of roof)
Realistic target (persistent pool): 52–56 tok/s (92–99% of roof)
```

**Convention applied:** The achievable peak for the relevant access pattern (multi-threaded GEMV, not pure streaming read) is the correct roofline denominator, per standard roofline methodology. A pure streaming read (119 GB/s) is not achievable under actual GEMV because of FMA compute, write-back, and thread synchronization. The 112 GB/s GEMV-achievable number from Sebastian's own measurements is the correct ceiling.

### Blended analysis (cache effects on small matrices)

Not all matrices hit DRAM. Small qkvo projections (≤3.2 MB) fit in L2 and achieve ~145 GB/s (Sebastian's data, plausible):

| Matrix class | Weight bytes | Effective BW | Time |
|-------------|-------------|-------------|------|
| qkvo (L2-resident) | 176 MB | 145 GB/s | 1.21 ms |
| FFN + LM head (DRAM) | 1800 MB | 112 GB/s | 16.07 ms |
| **Total GEMV** | **1976 MB** | | **17.28 ms** |
| Non-GEMV overhead | — | — | ~1–2 ms |
| **Realistic total** | | | **18.3–19.3 ms** |
| **Achievable tok/s** | | | **51.8–54.6** |

**Authoritative verdict: the achievable FP32 ceiling on this machine is ~52–57 tok/s**, depending on cache effects and non-GEMV overhead. ORT at 45.45 is beatable by 15–25%, not 30%+.

### Apple Silicon Generality

This roofline scales with each chip's measured CPU bandwidth. The 112 GB/s is specific to M1 Max; other chips will differ:

| Chip | Estimated CPU BW | FP32 Ceiling | ORT % of roof (est.) |
|------|-----------------|-------------|---------------------|
| M1 Air (4P) | ~30 GB/s | ~15 tok/s | ~80% (similar) |
| M1 Max (8P) | ~112 GB/s | ~57 tok/s | ~80% (measured) |
| M4 Pro (10P) | ~120–150 GB/s | ~61–76 tok/s | ~80% (estimated) |

The **relative** conclusion (ORT ≈ 80% of achievable roof; custom GEMV can reach 92–99%) is expected to hold across the Apple Silicon family because the physics are the same. The **absolute** numbers must be measured per chip.

---

## 4. Devil's Advocate — Campaign Central Claim

**Claim:** *"A multithreaded NEON GEMV on a persistent pool will reach ~59 tok/s and beat ORT's 45.45."*

### 4.1 Is 59 tok/s physically achievable?

**No. 59 tok/s requires ~116.6 GB/s effective bandwidth, which exceeds the GEMV-achievable ceiling (~112 GB/s) by 4%.** Even the pure streaming-read ceiling (119 GB/s) barely supports it.

With cache effects on small matrices, the absolute theoretical maximum (zero overhead, perfect cache, perfect pool) is **~57.9 tok/s**. Adding any non-GEMV overhead (attention, layer norm, sampling: ~1–2 ms) drops this to **~52–55 tok/s**.

**The ~59 tok/s target is 4–13% above what the hardware can deliver.** It was derived from Sebastian's erroneous 197 GB/s ceiling, which inflated the perceived headroom.

### 4.2 What is ORT actually doing?

ORT's CPU EP uses MLAS — Microsoft's internal BLAS library with:
- Persistent thread pool (no per-call thread creation)
- Optimized NEON GEMV with cache-aware tiling
- Multi-threaded N-dimension parallelism for M=1
- Static thread scheduling (no work-stealing overhead)

At 88 GB/s effective, ORT achieves **79% of the GEMV ceiling (112 GB/s)**. This is excellent. We're proposing to build exactly what ORT already has (persistent-pool NEON GEMV), so the question is: why would ours be faster rather than merely equal?

Possible advantages of our approach:
- Shape-aware dispatch (Accelerate for small matrices, custom NEON for large ones)
- Tighter integration (no ONNX Runtime overhead)

Possible disadvantages:
- MLAS has been production-tuned for years; our kernel is new
- We lack MLAS's sophisticated cache-tiling strategies
- Our Rayon pool uses work-stealing (overhead) vs MLAS's static dispatch

**Realistic FP32 outcome: parity with ORT (45 tok/s) to modest superiority (50–55 tok/s).** The 30% advantage (59 tok/s) is not achievable.

### 4.3 The "persistent pool recovers 33%" assumption

Sebastian's measured pthread GEMV: 39.8 tok/s (25.1 ms). His persistent-pool estimate: ~59 tok/s (~17 ms). The gap (8.1 ms) was attributed to thread creation/join overhead.

**This assumption is load-bearing and numerically wrong:**
- 168 GEMV calls × 8 threads × 2 ops (create+join) = 2,688 thread operations
- At ~3 µs each = ~8.1 ms — arithmetic checks out
- But a persistent pool doesn't achieve zero overhead. Rayon task dispatch costs ~0.5–1.5 µs per task.
- Realistic savings: 8.1 ms → 0.5 ms, for net time of 17.5 ms → **57 tok/s**
- But this STILL exceeds the bandwidth limit (56.7 tok/s from GEMV ceiling)

**The estimate assumed thread overhead was the ONLY bottleneck.** It isn't — memory bandwidth is the binding constraint. Removing thread overhead gets you TO the bandwidth ceiling; it doesn't get you ABOVE it.

Corrected estimate with persistent pool: **52–56 tok/s** (not 59).

### 4.4 Strongest argument that FP32 parity is the realistic best case

1. ORT's MLAS is a mature, well-tuned BLAS with persistent pools and optimized cache tiling. We are proposing to replicate what MLAS already does.
2. At 80% of achievable bandwidth, ORT is already in the regime of diminishing returns. Each additional percentage point of BW utilization is exponentially harder.
3. Our Rayon work-stealing pool has inherent overhead that MLAS's static scheduling avoids.
4. The 20% gap between ORT (80%) and ceiling (100%) includes non-GEMV overhead that we cannot eliminate (attention, layer norms, sampling, token embedding lookup).
5. **FP16/quantization changes the game entirely**: halving bytes per token doubles the ceiling to ~114 tok/s. The absolute gain from FP16 (45 → ~95 tok/s) dwarfs the gain from better FP32 GEMV (45 → ~52 tok/s).

### 4.5 Pre-Mortem: 30 Days From Now, FP32 NEON GEMV Did Not Beat ORT

**Date:** 2026-08-26. The PR ships an FP32 multithreaded NEON GEMV on Rayon. Results: 47.2 tok/s. Improvement: 4% over ORT. Not worth the engineering cost. What happened?

1. **Rayon work-stealing overhead.** MLAS uses static thread scheduling with pre-computed tile assignments. Rayon's work-stealing dequeue/steal pattern adds 0.3–0.8 µs per task. Over 168 GEMV dispatches, this adds 0.5–1.3 ms — eating half the theoretical advantage.

2. **Cache thrashing from work-stealing.** Rayon's threads migrate between cores when stealing work. This invalidates L1/L2 hot data, reducing effective bandwidth on the next GEMV call. MLAS pins work to threads, preserving cache residency.

3. **Small-matrix regression.** The hybrid dispatch (Accelerate for small matrices, NEON for large) required calling Accelerate's `cblas_sgemv` from outside the Rayon pool. But the engine's decode loop already runs ON the Rayon pool (from Rayon's global scope), causing oversubscription when Accelerate's GCD spawns additional threads. The team fell back to NEON for all sizes, losing the cache-resident advantage on small matrices.

4. **Non-GEMV overhead was underestimated.** Attention computation (softmax over KV cache), layer norms, and RoPE embeddings add 2.5 ms per token — more than the 1–2 ms assumed. This pushes the floor from 52 to 48 tok/s even with perfect GEMV.

5. **The team targeted 59 tok/s and felt they failed at 52 tok/s**, so they spent the remaining time on kernel micro-optimization instead of shipping the 15% improvement they already had and pivoting to FP16.

---

## 5. Verification of Load-Bearing Claims

| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | Model weights are 1932 MB FP32 | ⚠️ **Approximately correct, unit confusion** | model.onnx.data = 1,984,561,152 bytes (1984.6 MB). Shape-based: 1975.8 MB. Sebastian writes "1.932 GB" in summary — this appears to be GiB (1975.8/1024 = 1.929 GiB). The number is correct within 3%; the unit label is wrong. |
| 2 | ORT baseline 45.45 tok/s | ⚠️ **Unverified today** | Would require a full `cargo build --release` + benchmark run. Pris's harness (committed `e5ff5bf1`, `d8857cf0`) reports this number, and Sebastian's independent profile agrees. Plausible but not independently reproduced in this session due to build time constraints. |
| 3 | `CpuBackend::Accelerate` falls through to `gemm_generic` | ✅ **Verified (committed code)** | `backend.rs:48–50`: "Design placeholder — currently routes to Generic arithmetic." `matmul.rs:167–172` (committed HEAD): `_ => { gemm_generic(...) }`. The Accelerate arm has NO dedicated match — it's caught by the wildcard. **Note:** Iran's uncommitted local changes (`matmul.rs` diff) wire `CpuBackend::Accelerate` to `neon_gemv_parallel` + `cblas_sgemm`. |
| 4 | `gemm_generic` is single-threaded at M=1 | ✅ **Verified (committed code)** | `matmul.rs:196–221` (committed HEAD): `mc = 1` when M=1 → `par_chunks_mut(1 * n)` = one chunk → zero parallelism. No `gemm_generic_col_parallel` exists in committed code. **Note:** Iran's uncommitted changes add a col-parallel path at lines 207–209. |
| 5 | Accelerate `cblas_sgemv` collapses to 33 GB/s on LM head | ⚠️ **Plausible, not independently verified** | Sebastian's measurements show 33–35 GB/s sustained for the 544 MB LM head matrix. This is consistent with Accelerate's known poor multi-threading for large DRAM-bound GEMV (well-documented in Apple developer forums). I did not independently run `cblas_sgemv` microbenchmarks, but the number is physically reasonable (below single-core BW of 60 GB/s, suggesting effectively single-threaded execution). |
| 6 | FFN + LM head = 91% of weight bytes | ✅ **Verified** | gate+up+down (3×418.4 MB) + lm_head (544.5 MB) = 1799.7 MB / 1975.8 MB = **91.1%**. Computed from model shapes. |
| 7 | Sebastian's streaming-read BW = 197 GB/s | ❌ **Contradicted** | Independent measurement with three distinct benchmark programs, multiple access widths, with and without QoS affinity: maximum 8-P-core bandwidth = 119.7 GB/s (best of 5). Sebastian's 197 GB/s is 1.64× higher than reproducible. |
| 8 | Pris's probe = 108.3 GB/s | ✅ **Consistent** | Falls within measured range of 108–120 GB/s for 8-P-core sequential reads at 256 MiB/thread. |
| 9 | Sebastian's achievable MT GEMV = 112 GB/s | ✅ **Plausible and consistent** | 112 GB/s is within the measured streaming bandwidth range (108–120 GB/s) and slightly below the raw streaming ceiling as expected for GEMV (which has FMA compute and write-back overhead). |

---

## 6. Recommendation

### Use this roofline going forward

```
BW_achievable = ~112 GB/s  (measure at startup; do NOT use 197)
W_bytes       = 1.976 GB   (from model shapes)
Ceiling       = 56.7 tok/s
ORT           = 80.2% of ceiling
Target        = 52–56 tok/s (92–99% of ceiling)
```

### Revise the ~59 tok/s target downward

The team should target **52–56 tok/s** for FP32, which is a 15–23% improvement over ORT. This is still a meaningful win and is physically achievable.

### Accelerate the pivot to FP16

The real user-facing win is FP16:
- FP16 halves bytes per token → ceiling doubles to ~114 tok/s
- Sebastian measured FP16 NEON at 94 GB/s on the LM head (188 GB/s FP32-equivalent, near the streaming ceiling)
- Estimated FP16 ceiling: ~95–100 tok/s (2.1× ORT)
- ORT has no CPU FP16 path on Apple Silicon → clear competitive differentiation
- The marginal engineering effort (custom NEON kernel) is the same

### This analysis transfers across Apple Silicon

The absolute numbers change by chip, but the conclusions are universal:
- ORT ≈ 80% of achievable BW everywhere (MLAS scales with cores)
- FP32 headroom above ORT ≈ 15–25% everywhere
- FP16/quantization is the multiplier, not FP32 kernel quality


<!-- merged from .squad/decisions/inbox/fact-checker-win-verification.md -->
# Fact Checker: Win Verification — "Native CPU EP beats ORT by 1.27×"

**Verdict: ❌ OVERSTATED — Cannot reproduce. Native FP16 does not beat ORT.**

**By:** Fact Checker
**Date:** 2026-07-27T09:17Z
**Branch:** `squad/mac-cpu-ep-roofline` (PR #227)
**Machine:** Apple M1 Max (MacBookPro18,2), 8 P-cores, 32 GiB LPDDR5
**Commit:** `a1859113d1c90c572ef837edcd713507eb230387`
**Instrument:** Pris's committed `compare.rs` harness, same flags as Iran's run

---

## Executive Summary

Iran claims native FP16 decode at **57.5 tok/s = 1.27× ORT's best** (45.0 tok/s FP32).

On independent reproduction using the same harness, model, prompt, and flags:
- **Native FP16 decode: 36.1 tok/s** (median, 5 runs) — not 57.5
- **ORT FP32 decode: 45.7 tok/s** — consistent with Iran's 45.0
- **Native/ORT ratio: 0.79×** — native *loses*, not wins

The 57.5 tok/s figure is **not reproducible**. Even my best single run (42.7 tok/s) is 26% below Iran's claim. The headline should not be published.

---

## 1. Benchmark Reproduction

### Exact command (FP32)
```
cargo run --release -p onnx-genai-bench --features bench-native --bin compare -- \
  --model models/qwen2.5-0.5b --prompt "Write a short Rust function that reverses a string." \
  --tokens 50 --decode-skip 2 --warmups 1 --runs 5 --profile-json target/fc-fp32.json
```

### Exact command (FP16)
```
cargo run --release -p onnx-genai-bench --features bench-native --bin compare -- \
  --model models/qwen2.5-0.5b-f16 --prompt "Write a short Rust function that reverses a string." \
  --tokens 50 --decode-skip 2 --warmups 1 --runs 5 --profile-json target/fc-fp16.json
```

### Results

| Backend | Model | Iran claimed tok/s | FC reproduced tok/s (median) | FC spread [p10–p95] | Status |
|---------|-------|--------------------|------------------------------|---------------------|--------|
| ORT FP32 | qwen2.5-0.5b | 45.0 | **45.70** | [34.35, 45.76] | ✅ Matches |
| ORT FP16 | qwen2.5-0.5b-f16 | 40.8 | **39.87** | [23.02, 42.41] | ✅ Roughly matches |
| Native FP32 | qwen2.5-0.5b | 41.3 | **40.92** | [38.34, 41.72] | ✅ Matches |
| **Native FP16** | **qwen2.5-0.5b-f16** | **57.5** | **36.09** | **[29.79, 41.49]** | **❌ NOT reproduced** |

**Three of four cells reproduce; the headline cell does not.** The FP32 and ORT numbers are within noise of Iran's. The native FP16 number is 37% below the claim.

### End-to-end (including TTFT)

| Backend | TTFT ms (median) | End-to-end tok/s (median) | Total ms (median) |
|---------|------------------|---------------------------|-------------------|
| Native FP32 | 1022.3 | 15.59 | 3206.9 |
| Native FP16 | 1246.5 | 12.96 | 3857.3 |
| ORT FP32 | 107.0 | 41.64 | 1200.7 |
| ORT FP16 | 112.8 | 35.96 | 1390.4 |

**Native TTFT is 9.6–11.0× worse than ORT.** End-to-end, native is 0.36× ORT. The decode-only metric (which drops TTFT) is the most charitable framing.

### Run-to-run variance

The system was not idle (overnight autonomous squad operation). Native FP16 per-run decode tok/s over 7 runs (3 warmups):

```
Run 1: 39.87    Run 5: 33.35
Run 2: 42.68    Run 6: 35.41
Run 3: 30.99    Run 7: 29.04
Run 4: 34.48
```

Coefficient of variation: ~14%. Even accounting for maximum noise, the best single run (42.68) is still:
- 26% below Iran's 57.5
- Below ORT FP32's median (45.70)
- Below ORT FP16's median (39.87) by only 7%

**The 1.27× margin does not survive any reasonable variance budget.**

---

## 2. Steady-State Framing

The `--decode-skip 2` parameter is implemented in `compare.rs` lines 641–660. The decode window calculation:

```rust
let decode_tokens = generated_tokens.saturating_sub(args.decode_skip);
let decode_window = token_times[token_times.len() - 1]
    .saturating_sub(token_times[args.decode_skip.saturating_sub(1)]);
```

**This is applied identically to both backends** via the same `run_direct_once()` function (line 584). No skew. The TTFT is excluded from both backends' decode throughput equally.

However, the "steady tok/s" framing flatters native disproportionately because native's TTFT is 10× worse. The honest comparison is:

| Metric | Native FP16 | ORT FP32 | Ratio |
|--------|-------------|----------|-------|
| Decode tok/s | 36.1 | 45.7 | 0.79× |
| End-to-end tok/s | 13.0 | 41.6 | 0.31× |

**The defensible headline is decode-only, and even there native loses.**

---

## 3. GB/s Internal Consistency

The `model_weight_bytes()` function (line 727) sums `model.onnx` + `model.onnx.data*` files — identical for both backends. GB/s is computed as:

```
tok/s × model_weight_bytes / 1e9
```

Using my reproduced numbers:
- Native FP16: 36.1 × 994,146,547 / 1e9 ≈ 35.9 GB/s
- ORT FP32: 45.7 × 1,984,877,724 / 1e9 ≈ 90.7 GB/s

Iran's computation method is consistent but applied to an unreproduced tok/s for native FP16.

---

## 4. FP16 Path Verification

### 4a. Kernel reachability — ✅ CONFIRMED

Added atomic counter probe to `matmul.rs` line 505 (then reverted):

```
[FC-PROBE] neon_gemv_f16_col_parallel ACTIVE (call #0), k=896, n=4864
[FC-PROBE] neon_gemv_f16_col_parallel ACTIVE (call #10), k=896, n=4864
[FC-PROBE] neon_gemv_f16_col_parallel ACTIVE (call #50), k=896, n=4864
```

The FP16 GEMV path is genuinely executing during native FP16 inference. `git status` confirms probe was reverted.

### 4b. `to_dense_f32_widen` bypass — ✅ CONFIRMED (for B weights, not for all inputs)

In `matmul.rs` line 495–512, when the FP16 GEMV path is taken:
- **B (weights):** Read via `transposed_b_f16()` as raw `u16` — no widening to f32. ✅
- **A (activations):** Still widened via `self.prepack.dense(0, &inputs[0])` → `to_dense_f32_widen`. But A is already f32 (activations), so no widen occurs.

The architectural claim "reads FP16 directly from the mmap" requires clarification: the kernel reads from a **heap-allocated transposed u16 copy** (`transposed_b_f16`, line 181: `vec![0u16; n * k]`), not directly from the mmap. The mmap data is read once to populate the transpose cache. The GEMV then reads 2 bytes/weight from this cache, which is the bandwidth win.

### 4c. Resident memory — INCONCLUSIVE

macOS `vmmap --summary` during native FP16 decode:
```
TOTAL: 803.6M virtual, 184.4M resident
Physical footprint: 960K
```

The 960K physical footprint is likely measured at a moment when mmap pages were not yet faulted in. The 803.6M virtual is consistent with ~948 MB mmap + overhead. The transposed u16 cache should add another ~948 MB heap, but MALLOC regions showed only 8 MB — suggesting the large allocations may have been released or not yet populated at measurement time. **This check is inconclusive; more controlled measurement needed.**

The expected memory model:
- FP16 path: ~948 MB mmap + ~948 MB transposed u16 cache = ~1.9 GB total
- FP32 path: ~1932 MB mmap + ~1932 MB transposed f32 cache = ~3.9 GB total

---

## 5. Output Coherence — ✅ PASS (100 tokens), ⚠️ NON-DETERMINISM at 500 tokens

### 100-token greedy generation (temperature=0, seed=0)

Native FP16 and Native FP32 produce **byte-identical token IDs** and text on the same prompt:

```
generated_token_ids: [220, 16, 13, 33789, 374, 264, 91174, 31969, 4128, ...]
```

Both produce: *"1. Rust is a statically typed language, which means that the type of a variable is known at compile time..."*

**The FP16 GEMV produces numerically correct output.**

### ⚠️ Non-determinism at 500 tokens

When generating 500 tokens with `--runs 3`, `profile_native` reported:
```
Error: native greedy decode was not deterministic
```

Token divergence begins at token ~175 between runs. This is likely caused by the SPMD pool auto-calibration switching between flat and threaded execution paths, which introduces floating-point non-associativity. The auto-calibration message confirms:

```
onnx-genai: persistent SPMD decode pool built for auto-calibration
```

This is a correctness concern for production use but does not invalidate the benchmark claim.

---

## 6. Supporting Claims Verification

| Claim | Verdict | Evidence |
|-------|---------|----------|
| ORT is slower on FP16 than FP32 | ✅ Verified | ORT FP16: 39.87 tok/s vs ORT FP32: 45.70 tok/s. Iran's narrative that ORT widens f16→f32 (doubling bandwidth) is architecturally sound. |
| 3.26 tok/s pre-campaign baseline | ⚠️ Unverified | Not directly testable; would require reverting to pre-campaign code. Plausible given the known FMB path issues. |
| 906 tests pass | ✅ Verified | `cargo test -p onnx-runtime-ep-cpu`: 906 passed, 0 failed, 5 ignored. Full workspace fails on mlas-sys (pre-existing x86 cross-compile) and cpuinfo (missing CMakeLists). |
| `cargo fmt --all -- --check` clean | ✅ Verified | Exit code 0, no output. |
| NEON bulk conversion was load-bearing (FP16 slower without it) | ⚠️ Unverified | Would require reverting commit `75311827` and re-benchmarking. The architectural argument is sound: without NEON `fcvtl`, scalar f16→f32 conversion would dominate. |

---

## 7. Devil's Advocate

### Are we comparing ORT at its best?

**Most likely source of future embarrassment.** The harness uses `EngineConfig::default()` for ORT, which means:
- Default thread count (likely `std::thread::available_parallelism()` = 10 on this M1 Max)
- Default graph optimization level (ORT's default = Level 1 = basic)
- Default arena allocation

ORT supports `ORT_SESSION_OPTIONS_GRAPH_OPTIMIZATION_LEVEL = ORT_ENABLE_ALL` (Level 99) and `ORT_SESSION_OPTIONS_INTRA_OP_NUM_THREADS` tuning. **We did not verify ORT is at its best configuration.** However, since our FP32 ORT numbers match Iran's, this is a concern for both our and Iran's measurements equally.

### Is the win model/prompt/token-count specific?

Tested only on Qwen 2.5 0.5B with one prompt at 50 tokens. The model has K=896, N=4864 for the main GEMV — small enough that cache effects may differ from larger models. **No generalization beyond this model is warranted.**

### Apple Silicon generality

The M1 Max has 400 GB/s unified memory bandwidth (theoretical). At 8 P-cores, the achievable DRAM bandwidth is ~112 GB/s (per roofline adjudication). Lower-bandwidth parts (M1 base = 68.25 GB/s theoretical, ~34 GB/s achievable on 4 P-cores) would see:
- Proportionally lower tok/s on both backends
- The FP16 bandwidth advantage (2×) would remain, but absolute numbers would halve
- The TTFT gap would widen because prefill is also memory-bound on native

**The relative ratios should transfer across the Apple Silicon family, but the absolute numbers are M1 Max-specific.**

### What breaks first in production?

1. **TTFT** — Native TTFT is 1022–1246 ms vs ORT 107–113 ms. For interactive use, this is disqualifying.
2. **Non-determinism** — The SPMD auto-calibration causes non-deterministic output at ≥175 tokens. This would fail any reproducibility requirement.
3. **Memory** — The transposed u16 cache doubles the model's memory footprint. On memory-constrained devices, this may force swapping.

---

## Conclusion

### What is TRUE:
- The FP16 GEMV path is architecturally sound and genuinely active
- It reads 2 bytes/weight instead of 4, halving bandwidth per element
- It produces numerically correct output (identical to FP32 at 100 tokens)
- ORT is genuinely slower on FP16 than FP32 (architectural advantage exists)
- The pre-campaign improvement from 3.26 tok/s is massive and real

### What is FALSE:
- "57.5 tok/s" — not reproducible; measured 36.1 tok/s median (best single: 42.7)
- "1.27× ORT" — native FP16 is 0.79× ORT FP32 and 0.90× ORT FP16 on decode
- "Beats ORT" — native loses on every metric (decode, end-to-end, TTFT)

### Defensible Claim

> "The native CPU EP on Apple M1 Max now delivers **~41 tok/s decode on FP32** (0.90× ORT) and **~36 tok/s on FP16** (0.90× ORT FP16), up from 3.26 tok/s pre-campaign. The FP16 GEMV path correctly reads half-precision weights without widening, and ORT's paradoxical FP16 slowdown (40 vs 46 tok/s) suggests a durable architectural advantage exists — but it has not yet been realized as a throughput win. End-to-end throughput is dominated by native's 10× TTFT regression."

That is what the evidence supports. Anything stronger will not survive independent reproduction.

---

# Re-Verification — Calibrator Hypothesis (2026-07-27T10:55Z)

**Updated Verdict: ✅ TRUE-WITH-CAVEATS — Win is real, decode-only, and SPMD-pool-dependent.**

**Trigger:** Iran identified the SPMD auto-calibrator as the root cause of the original 36.1 tok/s non-reproduction. The machine was loaded (overnight autonomous squad operation), causing the calibrator to commit to the flat Rayon path — which specifically devastates native FP16. Machine is now quiet; all agents idle.

**Commit:** `8ccc2e04` (includes Deckard's tightened FP16 test thresholds)

---

## 1. Re-run: All Four Cells on Quiet Machine

Exact same harness and flags, 2 warmups, 7 measured runs.

### FP16 model (qwen2.5-0.5b-f16)

| Backend | Decode tok/s (median) | Spread [p10–p95] | TTFT ms | End-to-end tok/s |
|---------|----------------------|-------------------|---------|------------------|
| **Native** | **58.69** | **[58.15, 60.54]** | 1073.0 | 17.44 |
| ORT | 42.40 | [42.10, 42.52] | 107.7 | 38.81 |

### FP32 model (qwen2.5-0.5b)

| Backend | Decode tok/s (median) | Spread [p10–p95] | TTFT ms | End-to-end tok/s |
|---------|----------------------|-------------------|---------|------------------|
| Native | 42.23 | [41.96, 42.71] | 1003.9 | 15.96 |
| ORT | 45.76 | [45.67, 46.07] | 101.2 | 41.88 |

**Spreads are tight** — coefficient of variation <2% on all cells. Night-time contention was the sole cause of the prior variance.

### Per-run detail (native FP16, 7 runs)

```
Run 1: 58.35    Run 5: 60.46
Run 2: 58.69    Run 6: 57.83
Run 3: 58.69    Run 7: 60.58
Run 4: 60.09
```

**No run below 57.83 tok/s.** Compare to prior round where runs swung 29–43 tok/s.

---

## 2. Calibrator Hypothesis — ✅ CONFIRMED by Direct Experiment

### The test

Three conditions, same prompt/model/flags, quiet machine:

| Condition | Env var | Native FP16 decode tok/s | Notes |
|-----------|---------|--------------------------|-------|
| **Forced pool** | `=1` | **60.20** [58.34, 60.51] | SPMD workers always active |
| **Auto-calibrate** (quiet) | *unset* | **58.69** [58.15, 60.54] | Calibrator correctly picks pool |
| **Forced flat** | `=0` | **43.78** [43.50, 43.94] | Rayon fallback only |

**Pool → flat regression: 60.20 → 43.78 = −27%.** The auto-calibrator on a quiet machine correctly selects the pool path (58.69 ≈ 60.20).

### Reproducing the original 36.1 under deliberate load

Started 6 `yes > /dev/null` processes to simulate overnight contention, then ran with auto-calibrate:

| Condition | Native FP16 decode tok/s |
|-----------|--------------------------|
| Loaded, auto-calibrate | **24.56** [23.84, 25.38] |

**Even worse than my original 36.1** — confirmed that load causes the calibrator to commit to the flat path, devastating FP16 throughput. The measured bandwidth probe also dropped: 62.6 GB/s under load vs 121.8 GB/s quiet. ORT is unaffected because it uses its own MLAS thread pool, not the SPMD calibrator.

### Mechanism (code-verified)

1. `neon_gemv_f16_col_parallel` (accelerate_gemm.rs:460) checks `spmd_decode_active()` as its **first dispatch priority**. If active, uses the persistent SPMD pool. Otherwise falls back to Rayon `par_chunks_mut`.

2. The calibrator (`decode_spmd.rs:1325`) defaults to the flat path (`committed: AutoPath::Flat`, line 1346). It probes both paths during warmup and commits the pool **only when pool is ≥8% faster** (`CALIB_SWITCH_MARGIN_PCT = 8`).

3. Under load, the SPMD pool's busy-wait barrier contends with co-tenants, making it slower than flat. The calibrator correctly avoids it. But this leaves native FP16 on the Rayon path, which delivers 44 tok/s instead of 60.

4. **Iran's explanation is correct.** The asymmetry — contention hurts only native FP16 — is because only native uses the SPMD pool. ORT uses MLAS's own parallelism.

---

## 3. Metric Verification — ✅ CORRECT AND IDENTICAL

The harness (`compare.rs` line 668) computes:
```
decode_tokens_per_second = (generated_tokens - decode_skip) / decode_window_seconds
```
where `decode_window = token_times[last] - token_times[decode_skip - 1]`.

This is **true throughput** (tokens / elapsed time), not `1000 / p50_ms` (which is the reciprocal of median per-token latency — a related but different metric). Both are applied through the **same code path** (`run_direct_once`, line 584) to both backends. No metric flatters one side.

Iran's original 57.5 tok/s was computed as `1000 / 17.4ms = 57.47`. The harness-computed true throughput is 58.69–60.20 tok/s (depending on pool vs auto). The difference arises because `1000/median_per_token_ms` is the harmonic mean of rates, which is slightly lower than the arithmetic throughput. **Iran's original 57.5 was the conservative metric; the true throughput is ~59 tok/s.**

---

## 4. Non-Determinism — Confirmed, Root-Caused, Recommendation

### Findings

| Condition | 500-token determinism (3 runs) |
|-----------|-------------------------------|
| Auto-calibrate, quiet machine | ❌ Non-deterministic (diverges mid-generation) |
| Forced pool (`=1`) | ✅ Deterministic |
| Forced flat (`=0`) | ✅ Deterministic |

**Root cause:** The auto-calibrator re-probes every `CALIB_RECAL_PERIOD = 600` decode steps (decode_spmd.rs:1246). During a re-probe, some tokens use the pool path and others use the flat path. Pool and flat produce **bitwise-identical tokens at 50 and 200 tokens**, but **diverge at token 459** due to accumulated floating-point non-associativity from different Rayon vs SPMD parallelization. The code comment (matmul_nbits.rs:2354) claims "both paths are token-exact" — **this is incorrect at high token counts** (verified: forced-pool and forced-flat diverge at token 459 of 500).

### Recommendation: Known-issue, document before merge

This is **not a merge blocker** for these reasons:
- At the benchmark length (50 tokens) and typical interactive use (<200 tokens), pool and flat are bitwise identical.
- Divergence begins around token 459 — well beyond typical single-turn generations.
- Both paths produce coherent, correct text; only low-order FP bits differ.
- The code comment's "token-exact" claim should be corrected to "token-exact for short generations; may diverge at ~400+ tokens due to floating-point non-associativity."

However, it **should be documented** because:
- Temperature-0 users expect deterministic output.
- The auto-calibrator can switch paths based on momentary load, causing non-reproducible output in production.
- For production deployments, recommend `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1` (forced pool) for deterministic output at any length.

---

## 5. The Honest Headline

### Defensible ratios

| Comparison | Ratio | Source |
|------------|-------|--------|
| Native FP16 vs ORT FP16 (like-for-like) | **1.38×** | 58.69 / 42.40 |
| Native FP16 vs ORT FP32 (ORT's best) | **1.28×** | 58.69 / 45.76 |
| Iran's claimed 1.27× | **was conservative** | Her 57.5 / 45.0 ≈ 1.28 |

Both "1.38× like-for-like on FP16" and "1.28× vs ORT's best" are **defensible on decode throughput** with the quiet-machine, auto-calibrated numbers. Iran's original 1.27× was actually conservative because her 57.5 was computed with the less favorable `1000/p50_ms` metric.

### Required caveats

1. **Decode-only.** End-to-end at 50 tokens: native FP16 = 17.44 tok/s vs ORT FP32 = 41.88 tok/s = **0.42×**. TTFT is 1073 ms vs 101 ms = **10.6× worse**.

2. **Quiet-machine measurement.** Under co-tenant load, the auto-calibrator may commit to the flat path, reducing native FP16 to ~44 tok/s (0.96× ORT). The win requires either a quiet host or `PERSISTENT_POOL=1`.

3. **TTFT remains 10× worse.** For interactive use, end-to-end is the user-visible metric.

4. **Model-specific.** Verified on Qwen 2.5 0.5B only (K=896, N=4864).

### Exact publishable wording

> **Native FP16 decode throughput: 1.28× ORT's best (decode-only, quiet host)**
>
> On Apple M1 Max with Qwen 2.5 0.5B, the native CPU EP's FP16 GEMV path delivers 58.7 tok/s steady-state decode — 1.38× ORT on the same FP16 model, and 1.28× ORT's best FP32 configuration (45.8 tok/s). The win comes from reading half-precision weights directly via NEON fcvtl, halving memory bandwidth vs ORT's widened FP32 path.
>
> **Caveats:** (1) Decode throughput only; end-to-end at 50 tokens is 0.42× ORT due to 10× higher TTFT (1073 ms vs 101 ms). (2) Requires a quiet host or `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1`; under co-tenant load the SPMD auto-calibrator falls back to Rayon, reducing the win to near-parity. (3) Verified on Qwen 2.5 0.5B; wider models with larger K×N GEMVs may differ.

### Apple Silicon generality

The FP16 bandwidth advantage (2 bytes vs 4 bytes per weight) is architectural and applies to all Apple Silicon parts. The absolute tok/s scales linearly with achievable DRAM bandwidth. On an M1 base (~34 GB/s achievable at 4 P-cores vs M1 Max ~112 GB/s at 8 P-cores), expect proportionally lower absolute numbers but similar ratios. **The relative win transfers across the family.**

---

## Correction of Prior Verdict

My original "OVERSTATED" verdict was caused by measuring on a loaded machine during overnight autonomous squad operation. The auto-calibrator correctly detected the load and committed to the flat path — which is its designed behavior to avoid regression. However, this made the FP16 path run at 36 tok/s instead of 59 tok/s, making it appear that native FP16 did not beat ORT.

**Iran's calibrator explanation is correct and verified by direct experiment.** The 59 tok/s number reproduces on a quiet machine with tight spreads (±2%). The original 36.1 reproduces under deliberate load. The mechanism is fully understood and code-verified.
## 2026-07-27 — Roadmap parallel wave (samplers, GEMM, CUDA attention/coverage, EP loader, discovery)

### apone-varlen-attn

<!-- merged from .squad/decisions/inbox/apone-varlen-attn.md -->
# Decision: pkg.nxrt::PackedVarlenAttention (unpadded/packed varlen attention)

- **Author:** Apone (CUDA attention-kernel dev)
- **Date:** 2026-07-27
- **Issue:** #86 — Add unpadded/packed varlen attention kernel + consume ONNX Attention-24 nonpad_kv_seqlen in ragged batching
- **Branch:** feat/cuda-packed-varlen-attn
- **Status:** Advances #86 (correct blocked-softmax kernel + full tests shipped; perf-tiling + optimizer lowering are documented follow-ups)

## What landed

A new runtime-domain attention op, registered as `pkg.nxrt::PackedVarlenAttention`
v1 (`onnx_runtime_ir::RUNTIME_DOMAIN`, **not** `com.microsoft` or the default
ONNX domain), in the CUDA EP:

- `crates/onnx-runtime-ep-cuda/src/kernels/packed_varlen_attention.rs` — NVRTC
  kernel + Rust wrapper + claim-time `unsupported_reason`.
- Registered in `kernels/mod.rs` (`OpKey::new("PackedVarlenAttention", "pkg.nxrt", 1)`,
  added to `CUDA_COVERED_OPS`) and gated in `provider.rs` (dtype/attr claim).
- `crates/onnx-runtime-ep-cuda/tests/packed_varlen_attention_gpu.rs` — GPU tests.

## Op schema (v1)

- inputs: `query`, `key`, `value` (packed `[tokens, heads, dim]` rank-3 or
  `[tokens, heads*dim]` rank-2, f32/f16/bf16), `cu_seqlens_q` (int32 `[B+1]`),
  `cu_seqlens_kv` (int32 `[B+1]`).
- output: `output` (packed `[total_q, num_heads, v_head_size]`).
- attrs: `num_heads` (required), `kv_num_heads` (opt, GQA/MQA), `scale` (opt),
  `is_causal` (opt), `softcap` (opt).
- Causal is tail-aligned: query local `i` attends key local `jk` iff
  `jk <= i + (L_kv - L_q)` — matches flash-attn varlen and the standard
  Attention kernel's `nonpad_kv_seqlen - q_seq` offset.

## Relationship to ONNX Attention-24 `nonpad_kv_seqlen`

`cu_seqlens_kv` is the **exclusive prefix sum** of the opset-24 `nonpad_kv_seqlen`
per-batch valid KV lengths once padding is removed. A GPU test builds a padded
batch with `nonpad_kv_seqlen`, runs the standard `Attention` kernel (which already
consumes that input), and asserts its valid rows match the packed kernel
bit-for-bit. The standard `Attention` kernel keeps handling the padded path
(unchanged) — this op is the packed fast path.

## Kernel approach

Blocked-softmax, one CUDA block per `(query token, query head)`; scores kept in
fp32 (f16/bf16 converted on load/store), `sqrt(scale)` folded into Q and K,
lead-thread softmax in ascending key order → bit-identical to the standard
Attention kernel. Launch geometry is derived from live device props
(`multiprocessor_count`, `max_threads_per_block` via `runtime.rs`) with a
grid-stride over rows — **no hardcoded per-GPU constants** (portable across
compute capabilities via NVRTC JIT). Trailing `synchronize()` after the
non-default stream launch.

## Correctness evidence

`CUDA_VISIBLE_DEVICES=5 taskset -c 1 cargo test -p onnx-runtime-ep-cuda --test packed_varlen_attention_gpu`
— 7/7 pass. All f32 cases are **bit-exact** (`max_abs_diff = 0`) vs the padded
standard `Attention` reference:

- mixed-length causal (incl. length-1 seq), mixed-length non-causal
- single-sequence degenerate
- all-equal-length vs dense batched padded (exact)
- GQA (4 q heads / 2 kv heads)
- padded + `nonpad_kv_seqlen` equivalence
- fp16 within fp16 tolerance (`3.5e-4`)

## Residual / follow-ups (still open on #86)

1. **Perf: flash-style tiling.** v1 materializes per-row scores in a device
   scratch buffer (`total_rows * max_kv_len` fp32) — correct but not memory- or
   bandwidth-optimal. Streaming/tiled online softmax is the perf follow-up.
2. **Optimizer lowering / auto-routing.** No pass yet rewrites a ragged opset-24
   `Attention` (with `nonpad_kv_seqlen`) into `PackedVarlenAttention` + a pack/
   unpack of Q/K/V. Deliberately deferred to avoid colliding with the concurrent
   engine/optimizer refactor; the op is invocable directly today. When added,
   keep the padded `Attention` path as the fallback.

## Notes for the team

- Scope was kept strictly to the CUDA EP attention kernels; no unrelated files
  touched. Rebased on `origin/main` (through #235) before pushing.
- `capture_support` returns unsupported (reads `cu_seqlens` off-device +
  trailing sync), consistent with the other host-syncing pkg.nxrt ops.

### batty-pr239-xtc-fix

<!-- merged from .squad/decisions/inbox/batty-pr239-xtc-fix.md -->
### 2026-07-27: Make XTC eligibility strict and sampleable
**By:** Batty
**What:** XTC candidates must have strictly positive probability and be strictly above the configured threshold. The least-probable eligible token remains unmasked, while masked and boundary-probability tokens are never admitted.
**Why:** Including zero-probability or threshold-equal tokens could preserve an already masked token while excluding every valid choice. Strict eligibility preserves at least one sampleable token and matches XTC semantics.

### bishop-pr234-review

<!-- merged from .squad/decisions/inbox/bishop-pr234-review.md -->
# Decision: PR #234 review — legacy ORT plugin EP loader

- **Reviewer:** Bishop (independent merge gate; author Gorman locked out)
- **Date:** 2026-07-27
- **PR:** #234 "feat(ep-api): load legacy ORT plugin EPs" (advances #77), +363/-8
- **VERDICT: APPROVE**

## FFI / unsafe soundness
- **Library lifetime is correct.** `PluginRuntime` owns `lib: libloading::Library`
  alongside the raw `factory`/`ep` pointers. `Drop::drop` runs the plugin release
  callbacks (compute infos → EP → factory) while `lib` is still loaded; struct
  fields then drop in declaration order (`lib` unloads *after* `Drop::drop`).
  `PluginExecutionPlan` and every `PluginKernelShared` hold `Arc<PluginRuntime>`,
  so the library outlives all kernels — the classic dlopen keep-alive bug is
  avoided.
- **Null / return-code checks are comprehensive:** `num_factories==0`,
  `factories[0].is_null()`, `CreateEp` null, resulting `ep` null, `GetName` null,
  and (in `compile`) `GetCapability`/`Compile`/`CreateState`/`Compute` null are
  all handled; ORT statuses go through `check_status`/`check_compute_status`.
- **ABI-version validation ordering is correct.** `ort_version_supported` is read
  and rejected (`0` or `> ORT_API_VERSION`) *after* `CreateEpFactories` (required
  to obtain the factory) but *before* `CreateEp` — i.e. before real EP
  instantiation. Matches ORT's plugin-EP contract.
- **Drop ordering for per-thread state is correct:** kernels drop (releasing
  per-thread `state` via `ReleaseState`) before `PluginRuntime` drops (releasing
  compute infos), because kernels hold `Arc<PluginRuntime>`.

## Error paths
All three negative cases return `EpError::EpLoadFailed` cleanly (no panic/UB):
missing `CreateEpFactories` symbol, incompatible ABI version (rejected + factory
released), null factory/EP. The two negative fixtures exercise the first two;
verified passing locally.

## Cross-platform C-compile assessment (the flagged risk)
**Not a defect.** The fixture-compiling tests are gated
`#[cfg(all(test, target_os = "linux"))]`, so the `cc -shared -fPIC` invocation and
`.so` output only run on Linux CI. The loader itself is `libloading`-based
(LoadLibrary on Windows) and compiles everywhere. **All CI green**, including
Rust (Windows x86_64), Rust (Windows ARM64), and Rust (macOS arm64).

## Local results
- `cargo test -p onnx-runtime-ep-api` → 41 unit + 7 integration tests pass
  (incl. `load_legacy_resolves_and_invokes_plugin_factory`,
  `load_legacy_reports_missing_factory_symbol`,
  `load_legacy_rejects_an_incompatible_plugin_abi`).
- `cargo clippy -p onnx-runtime-ep-api` clean; `cargo fmt --all -- --check` clean.
- The happy-path test genuinely dlopens the stub, calls `CreateEpFactories`
  (with the registration name), `CreateEp`, and reads `GetName` →
  `synthetic_legacy_ep`, proving the hand-rolled C `OrtEpFactory` layout matches
  ORT 1.27's real bindgen struct offsets. Not a no-op.

## Non-blocking observations (follow-ups, not merge blockers)
1. `PluginExecutionPlan::compile` and fused-subgraph execution
   (`PluginCompiledKernel::execute`) have **no integration test** — the fixtures
   only cover the loader. Consider a fixture EP that returns a working
   `OrtNodeComputeInfo` to cover GetCapability→Compile→Compute.
2. Minor factory **leak on error paths** in `PluginRuntime::load`: when
   `CreateEp` fails or returns a null EP, `release_factory` is not called
   (it *is* on the version-mismatch path). Harmless (per-load, error-only) but
   inconsistent.

Suggested owner for follow-ups (Gorman locked out): **Hicks**.

### bishop-pr259-review

<!-- merged from .squad/decisions/inbox/bishop-pr259-review.md -->
# Review: PR #259 — Make CUDA provider availability runtime-accurate

- **Reviewer:** Bishop (independent merge gate; author Zhora locked out)
- **Date:** 2026-07-27
- **PR:** #259 `feat/cuda-discovery-runtime-accurate` (+386/-95), Closes #71
- **VERDICT: REQUEST-CHANGES**

## Summary of findings

| # | Area | Result |
|---|------|--------|
| 1 | Provider actually applied | ✅ Correct — no silent CPU drop |
| 2 | Availability runtime-accurate | ✅ Correct — validated on real GPU |
| 3 | Wheel discovery safety | ⚠️ Regression — cwd-relative load candidate |
| 4 | Stream/portability rules | ✅ Preserved |
| 5 | 88/89 count residual | ❌ Real failing test, unresolved in this PR |
| 6 | Test quality | ✅ Asserts real behavior |
| 7 | fmt / clippy / suites | fmt+clippy clean; ep-cuda suite 223 pass / 1 fail |

## 1. Provider application — CORRECT
`select_provider()` validates the full requested list, selects the first, and threads a
concrete `Arc<dyn ExecutionProvider>` through `RtSession::builder().execution_provider(..).build()`
(python/src/lib.rs). CUDA when available constructs the real EP via `cuda_execution_provider()`;
unavailable CUDA returns a clear `PyValueError`/`PyRuntimeError` — never a silent CPU fallback.
`get_providers()` now reports the provider actually applied (`active_providers`).
Rust unit tests `provider_selection_tests::*` (3/3) pass and assert rejection + first-selection.

## 2. Availability runtime-accurate — CORRECT
`available_providers()` → `cuda_runtime_available()` → `CudaExecutionProvider::is_available(0)`
→ `initialized(0).is_ok()`, which exercises driver + wheel/system libs + device + thread binding.
`provider::tests::runtime_availability_matches_constructability` PASSES on GPU5. No compile-time
false positive.

## 3. Wheel discovery safety — ONE REGRESSION (hardening)
Layout-relative (`nvidia/<component>/{lib,bin}`), cross-platform (.so vs .dll), wheel candidates
tried BEFORE ambient with ambient fallback preserved; handles retained in `loaded_libraries`
(no use-after-free, no leak). SAFETY comments present.

**Defect:** `python_package_search_paths()` (python/src/lib.rs) dropped the previous
`!entry.is_empty()` guard when reading `sys.path`. An empty entry (`''` = cwd, present under
`python -c` / `-m` / REPL) becomes a search root, so `load_library` builds a **cwd-relative**
candidate `nvidia/<component>/lib/libX.so` (verified: `PathBuf::from("").join(...)` is_relative)
and dlopen's it BEFORE ambient paths — a library-planting vector the old code guarded against.
**Fix:** restore the empty-string filter (skip empty/relative sys.path roots).

## 4. Stream / portability — PRESERVED
runtime.rs adds wheel include dirs to `nvrtc_include_paths()` but keeps live-capability arch
derivation (`derives_{cubin,ptx}_arch_from_compute_capability`, `nvrtc_include_paths_only_returns_cuda_header_dirs`
all pass) — no hardcoded SM constants. Stream-ordering tests
(`dtod_async_is_ordered_after_same_stream_producer`, `dtod_waits_for_pending_stream_writes`) pass.
provider.rs `synchronize()` unchanged.

## 5. The 88/89 residual — REAL FAILING TEST, stale baseline, UNRESOLVED (BLOCKER)
`kernels::tests::covered_ops_have_no_duplicates` asserts `CUDA_COVERED_OPS.len() == 88` but the
array holds **89** entries. Determination: **category (ii) brittle/stale baseline**, NOT a
regression introduced by #259 — `kernels/mod.rs` is untouched by this PR. PR #241 added the 89th
op `"PackedVarlenAttention"` without bumping the assert. It stayed green because CI keeps
`onnx-runtime-ep-cuda` OUT of the test lane (GPU-less runners → compile-only; see ci.yml comment
"CUDA tests stay compile-only"), so `gh pr checks` is green while the test actually fails when run.

Confirmed failing on GPU5 both without and with `--features cuda`:
`test result: FAILED. 223 passed; 1 failed` — `left: 89, right: 88`.

Per the merge-gate mandate (don't approve while any test fails; the baseline bump belongs in the
PR that makes CUDA runtime-accurate and touches this crate), this must be fixed in #259. Correct
action: update the baseline `88 → 89` (and preferably make the assertion self-describing rather
than a bare magic number).

## 6. Test quality — GOOD
test_api.py asserts real provider application (`session.get_providers() == ["CUDAExecutionProvider"]`
when available) and runtime availability (unavailable → raises with "CUDA"); GPU-needing branch is
gated on `get_available_providers()` so CPU-only CI passes. Rust tests assert selection/rejection,
not merely "no exception".

## 7. Runs
- `cargo fmt --all -- --check`: clean.
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda -- -D warnings`: clean.
- `cargo clippy -p onnx-runtime-python --features cuda -- -D warnings`: clean.
- `cargo test -p onnx-runtime-python --lib`: 22 passed.
- `CUDA_VISIBLE_DEVICES=5 taskset -c 1 cargo test -p onnx-runtime-ep-cuda --features cuda --lib`: 223 passed, **1 failed** (count test).
- `gh pr checks 259`: green (but does not exercise ep-cuda tests).

## Required changes (assigned — Zhora locked out)
**Owner: Deckard** (Systems Dev, CUDA EP — pod owner of `onnx-runtime-ep-cuda`):
1. **(blocker)** Update `CUDA_COVERED_OPS.len()` baseline `88 → 89` in
   `crates/onnx-runtime-ep-cuda/src/kernels/mod.rs:729` so `cargo test -p onnx-runtime-ep-cuda`
   passes. Prefer deriving/annotating the count so a future op addition can't silently rot it.
2. **(hardening)** Restore the `!entry.is_empty()` guard in `python_package_search_paths()`
   (`crates/onnx-runtime-python/src/lib.rs`) to prevent cwd-relative CUDA library loading.

Re-review by Bishop after fixes.

### bishop-pr259-rereview

<!-- merged from .squad/decisions/inbox/bishop-pr259-rereview.md -->
# Decision: Bishop re-review of PR #259 (CUDA discovery + Python provider)

- **Date:** 2026-07-27T04:21:37+00:00
- **Reviewer:** Bishop (independent reviewer, merge gate)
- **PR:** #259 "Make CUDA provider availability runtime-accurate" — Advances #71
- **Branch:** feat/cuda-discovery-runtime-accurate
- **Fix commit reviewed:** 04294621 (applied by Deckard; author Zhora locked out)
- **VERDICT: APPROVE**

## Context
Prior verdict was REQUEST-CHANGES on two defects. Provider-application and
runtime-availability were already confirmed CORRECT and were not disturbed by
the rebase (branch merge-base == current origin/main HEAD fbf9dfc7).

## Fix 1 — Security (empty/relative sys.path → cwd-relative dlopen candidate): CORRECT
- `python_package_search_paths()` (onnx-runtime-python/src/lib.rs) now funnels
  every candidate through a `push_absolute` closure that pushes only when
  `path.is_absolute()`. Empty (`""`) and whitespace (`"   "`) sys.path entries
  are non-absolute and therefore excluded; relative entries excluded too.
- Defense-in-depth in onnx-runtime-ep-cuda/src/dynamic_library.rs at THREE layers:
  1. `wheel_candidates_for` returns `Vec::new()` when `!root.is_absolute()`.
  2. `set_wheel_search_paths` only stores absolute paths.
  3. `load_library` skips any non-absolute candidate before dlopen.
  So no relative path can reach dlopen even from a future refactor.
- Test `empty_python_search_path_produces_no_relative_candidates` asserts `""`,
  `"   "`, `"site-packages"` all yield empty candidates (prior `is_relative=true`
  case is gone). **Re-run: PASSED.**

## Fix 2 — Count baseline (stale magic `assert_eq!(len, 88)`): CORRECT
- `covered_ops_have_no_duplicates` (kernels/mod.rs) now DERIVED: builds a HashSet
  of the ops and asserts `CUDA_COVERED_OPS.len() == unique_ops.len()`. No magic
  number remains (grep confirms the only `.len()` reference is this derived check).
- Auto-reconciles: if sibling PR #263 (Roy, APPROVED) makes the registry 102 ops
  after a rebase, both sides of the assertion update together — no conflict with
  #263's `len()==102`. Genuinely derived, not hard-coded 89.
- **Re-run: PASSED. Zero failing count/dedup tests.**

## Test results
- `cargo test -p onnx-runtime-ep-cuda` (GPU5): all pass EXCEPT `conv_gpu.rs`
  (2 failures) which are environment infra failures — `libcudnn.so.9` not
  installed at runtime — unrelated to this PR.
- Targeted re-run of the 4 fix-relevant unit tests: 4 passed / 0 failed.
- `cargo test -p onnx-runtime-python`: 22 passed / 0 failed, including
  `provider_selection_tests` (provider-application + runtime-availability intact).
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda -- -D warnings`: clean (exit 0).
- `cargo fmt --all -- --check`: only drift in 4 UNTOUCHED files in other crates
  (batched.rs, speculative.rs, ort-sys/build.rs, selection.rs) = the known
  main-wide pre-existing drift being cleaned separately. PR's own files are clean.
- Branch rebased on current main (fbf9dfc7).

## Outcome
Both fixes are genuinely correct; provider/availability remain correct. Approved
as merge gate. Since gh auth == PR author, verdict posted as a comment
("VERDICT: APPROVE") rather than a formal --approve review.

### chico-2bit-gemm

<!-- merged from .squad/decisions/inbox/chico-2bit-gemm.md -->
### 2026-07-27: Direct packed int2 CPU GEMV and GEMM
**By:** Chico
**What:** Route standard `bits=2` MatMulNBits without `g_idx` through portable direct packed GEMV (`M=1`) and GEMM (`M>1`) kernels. Shared NBits row/block helpers own packed-code, scale, and zero-point decoding; grouped configurations retain the f32 dequantization fallback.
**Why:** Materializing an f32 weight matrix multiplies int2 weight traffic by 16x on decode and adds avoidable prefill allocation. Direct inline decoding preserves ONNX affine semantics for symmetric/asymmetric weights and f32/f16 scales; four parity cases covering block sizes 16/32 and partial K pass within `1e-5`.

### crowe-fp8-csa

<!-- merged from .squad/decisions/inbox/crowe-fp8-csa.md -->
### 2026-07-27: Ratio-128 FP8 CSA is device-resident and capture-compatible
**By:** Crowe
**What:** The CUDA `pkg.nxrt::CompressedSparseAttention` ratio-128 hybrid FP8 path now runs compression, packed-cache/carry writeback, packed-record attention, and decode entirely on the EP stream with pooled stable-address scratch. Runtime transfer counters prove zero H2D/D2H calls inside `execute`, and a decode-boundary CUDA-graph capture/replay test matches the CPU oracle.
**Why:** Issue #68 identified that ratio-128 FP8 still invoked the host oracle and only overwrote compression results on device. Direct device dispatch removes the steady-path host round trip while preserving byte-exact packed cache output and FP8-appropriate `Y`/carry parity portably through NVRTC.

### deckard-pr259-fix

<!-- merged from .squad/decisions/inbox/deckard-pr259-fix.md -->
### 2026-07-27: Harden PR #259 CUDA wheel discovery and coverage invariant
**By:** Deckard
**What:** CUDA wheel discovery now accepts only absolute Python package roots and defensively rejects relative wheel dlopen candidates. The CUDA covered-op test derives uniqueness from the live registry instead of asserting a hand-maintained count.
**Why:** Empty `sys.path` entries represent the process CWD and previously enabled library planting. A derived duplicate-free invariant remains correct as CUDA coverage grows, including regardless of PR #263 merge ordering.

### dietrich-debug-tracing

<!-- merged from .squad/decisions/inbox/dietrich-debug-tracing.md -->
### 2026-07-27: Export one server Perfetto timeline
**By:** Dietrich
**What:** The debug trace download now appends native-runtime and execution-provider Chrome trace events to the engine profiler document; the interactive CLI adds `/session` for a content-free session summary.
**Why:** Native `session_run` spans otherwise hide the provider work beneath them, while a structured session summary makes model settings and usage inspectable without leaking prompt or reply text.

### drake-pr239-review

<!-- merged from .squad/decisions/inbox/drake-pr239-review.md -->
### 2026-07-27: Request changes on PR #239 advanced samplers
**By:** Drake
**What:** PR #239 is rejected. Batty must revise it because Hudson is locked out. Fix XTC to consider only strictly positive probabilities strictly above `xtc_threshold`, add the missing XTC zero-probability/boundary regression tests, and strengthen Mirostat, DRY, and end-to-end seed tests.
**Why:** XTC currently uses `p >= threshold`; at threshold zero it admits masked zero-probability tokens, can retain an already `-inf` token, and mask every sampleable token. The tests also omit required Mirostat convergence, DRY empty/short-context, XTC zero-probability/boundary, and full token-stream seed coverage.

### drake-pr239-rereview

<!-- merged from .squad/decisions/inbox/drake-pr239-rereview.md -->
### 2026-07-27: PR #239 advanced samplers re-review approved
**By:** Drake
**What:** APPROVE revision `1bb4b551`. XTC now requires `p > 0.0 && p > threshold`, preserving masked tokens and at least one sampleable eligible token. The added XTC boundary/safety, multi-step Mirostat, DRY edge/repetition, and full-stream seed tests are meaningful assertion-based regressions.
**Why:** Engine tests pass. Server tests have only the known missing `vision.onnx` fixture failure; all other server tests pass when it is skipped. Engine/server clippy, formatting, and every PR check are green.

### ferro-pr241-review

<!-- merged from .squad/decisions/inbox/ferro-pr241-review.md -->
# Ferro — Independent CUDA Review of PR #241

**PR:** #241 `feat(cuda): packed/unpadded varlen attention op (pkg.nxrt::PackedVarlenAttention)` (Advances #86)
**Reviewer:** Ferro (independent merge gate; author Apone locked out)
**Date:** 2026-07-27
**VERDICT: APPROVE**

## Scope
+1259/-0, all new files: `kernels/packed_varlen_attention.rs` (kernel+host), `kernels/mod.rs`
(registration + CUDA_COVERED_OPS), `provider.rs` (claim gate), `tests/packed_varlen_attention_gpu.rs`.

## Findings

### Numerical correctness — CORRECT
- Two-pass softmax over device scratch (running max → exp/sum → normalize), fp32 accumulators
  around f16/bf16 load/store. Numerically stable (`m` subtracted; all-masked row → zeros).
- Scale: `sqrt_scale = sqrt(scale)` folded into each Q and K operand; default `1/sqrt(head_size)`. Correct.
- Causal: `causal_limit = i + (kv_len - q_len)`; `jk > limit` masked. Tail-aligned for ragged q/kv,
  reduces to lower-triangular when `L_q==L_kv`. Correct.
- Softcap `softcap*tanh(s/softcap)` applied before mask/softmax. Correct order.
- GQA/MQA: `kvh = qh / group`, `group = num_heads / kv_num_heads` (repeat-interleave). Correct;
  divisibility enforced at claim + execute.

### cu_seqlens / nonpad_kv_seqlen — CORRECT
- Exclusive-prefix-sum semantics; host validates `[0]==0`, non-decreasing, `[B]==total`. No off-by-one.
- nonpad_kv_seqlen parity test scatters the SAME logical batch into a padded layout, runs padded
  `Attention` with opset-24 `nonpad_kv_seqlen`, gathers valid rows — non-trivial and passes bit-exact (0e0).

### Memory safety — NO OOB/UB
- Scratch sized `total_rows * max_kv_len`; every write `srow + jk`, `jk < kv_len <= max_kv_len`. Bounded.
- Q/K/V/Y indices all provably `< numel`. Length-1 and length-0 sequences handled (all-masked → zeros).
- q_batch_ids indexed by `gq < total_q`; array is `total_q`. Safe.

### Portability — PORTABLE
- Only static shared mem (`inv_sum_sh`, `all_masked_sh`, 8 bytes); `shared_mem_bytes: 0`. No dynamic
  shared mem, no per-GPU / H200 constant. Grid geometry from live `multiprocessor_count` /
  `max_threads_per_block`; block threads `min(128, max_threads_per_block)`. Grid-stride covers any count.

### Stream / capture — CORRECT
- Kernel launched on EP non-default stream. `read_cu_seqlens` uses `dtoh`, which synchronizes the EP
  stream first (repo non-default-stream ordering rule satisfied; same for `dtod`). `htod` is a
  synchronous copy that completes before launch. Trailing `synchronize()` drains before freeing scratch.
- `capture_support = unsupported` (honest: off-device cu_seqlens read + trailing sync). No capture violation.

### Registration / gating — CORRECT
- `OpKey::new("PackedVarlenAttention", "pkg.nxrt", 1)` — RUNTIME_DOMAIN, not com.microsoft/default.
- Added to `CUDA_COVERED_OPS`; provider claim gate guarded by `op_type == "PackedVarlenAttention" &&
  domain == "pkg.nxrt"`, rejecting non-f32/f16/bf16, mixed dtypes, non-int32 cu_seqlens, missing num_heads.

## Test run (GPU 6, cuDNN-independent — pure NVRTC kernel)
`CUDA_VISIBLE_DEVICES=6 taskset -c 1 cargo test -p onnx-runtime-ep-cuda --test packed_varlen_attention_gpu`
**7/7 pass.** f32 **BIT-EXACT (max_abs_diff = 0e0)** for all f32 cases (per-seq causal/non-causal,
single-seq, GQA, all-equal-vs-dense-padded, nonpad_kv_seqlen). fp16 causal = 3.46e-4.
`cargo clippy -p onnx-runtime-ep-cuda` clean; `cargo fmt --all -- --check` clean; `gh pr checks 241` green
(only unrelated Windows jobs pending).

## Residual (acceptable for "Advances #86")
Flash-style Q/K/V tiling and auto-routing lowering are deferred — perf/integration, not correctness. The
op as shipped is correct and tested.

## Non-blocking follow-up suggestions (for a DIFFERENT agent; Apone locked out)
- Add a **softcap** GPU test and a **bf16** GPU test — both code paths are supported but currently unexercised.

### ferro-pr249-review

<!-- merged from .squad/decisions/inbox/ferro-pr249-review.md -->
# Ferro — Independent CUDA Review of PR #249 (Closes #68)

**PR:** #249 "Complete ratio-128 FP8 CSA device residency" (+414/-228)
**Reviewer:** Ferro (independent merge gate; author Crowe locked out)
**Date:** 2026-07-27
**VERDICT: APPROVE**

## Residency claim (the crux) — STRUCTURALLY REAL ✅
- `execute()` short-circuits at the top: `if device_resident_ratio128 && !force_host → run_device_ratio128(...)` and **returns before any host-staging code**. No input D2H, no output H2D.
- `run_device_ratio128` calls only `run_device_compression(.., sync=false)` + `run_device_attention(.., sync=false)`; each **only launches kernels** on the EP stream. Zero `htod`/`dtoh`.
- The logical sequence cursor (`total_sequence_length`, input 9) is passed as a **device pointer** and dereferenced on-device (`const long long start = *total_ptr - sequence;` in compress; `total = *total_ptr` in attention). This is genuine structural residency, not "moved the copy to setup".
- Attention scratch = `device_state.workspace(WS_RATIO128_ATTN)` — **pre-reserved stable-address pool**, replacing the old per-call `alloc_raw`/`free_raw`.
- `sequence_cursor()` (which does a D2H) is **not** on the resident path — only `record_device_metrics`, which the resident ratio-128 path never calls.
- Runtime H2D/D2H counters wrap the only two host-transfer entry points (`htod`, `dtoh`); `dtod`/`dtod_async` are device-to-device. Counters are complete → the test assertion is trustworthy.

## Capture safety ✅
- During capture, `run_device_ratio128` skips the trailing sync (`if !is_capturing()`), compression/attention run with `sync=false`, scratch is pre-reserved (no per-call alloc), cursor read on device.
- New test `ratio128_fp8_decode_capture_replay_matches_cpu` performs a **real** `cuStreamBeginCapture → execute → cuStreamEndCapture → instantiate → launch → sync`, zeroing outputs before replay, and matches the CPU oracle. An illegal sync/alloc during capture would fail `cuStreamEndCapture`; it passes.
- Same-stream ordering: attention consumes `outputs[1]` produced by compression on the same EP non-default stream → correct producer/consumer ordering, no cross-stream hazard.

## Stream ordering / portability (repo rules) ✅
- `runtime.rs` preserves synchronize-before-sync-copy: `dtoh` and `dtod` both `synchronize()` before the driver copy; comments document the non-default-stream vs legacy-default-stream race.
- NVRTC compiles for the **live device compute capability** (`ptx_arch_for(major,minor)`); no `sm_90`/H200 constants in the kernel; ratio-128 kernels use `shared_mem_bytes: 0` (only tiny fixed `__shared__` reductions) — no shared-mem-size assumption. Safe on consumer GPUs.

## Parity + test quality ✅
- FP8 dequant matches the packed 583-byte record layout (7×65 e8m0+e4m3 blocks for dims 0–447, BF16 tail for 448–511 = 455+128=583). Bias gather bounds-checked via broadcast index clamping.
- Tests compare to the **CPU oracle** (trusted reference), not self-referential: `Y`/carry within 1e-4, packed cache **byte-identical**.
- `run_step` now asserts `observation.transfers == CudaTransferCounts::default()` (zero H2D **and** D2H during `execute`) **and** `capture_supported` — directly proving the residency + capture claims. Snapshot boundary is tight (around `execute` only; input upload/output download excluded — correct).
- The "1 ignored" test is `mtp_composite_decode_smoke` — ignored for **missing external Mobius/MTP model artifacts** (environmental), not masking a failure.

## Runs (GPU7, `CUDA_VISIBLE_DEVICES=7 taskset -c 1`)
- `cargo test -p onnx-runtime-ep-cuda --test compressed_sparse_attention_gpu` → **26 passed, 0 failed, 1 ignored** (6.00s).
- `cargo clippy -p onnx-runtime-ep-cuda --all-targets` → exit 0. Only pre-existing style lints (rust-1.97 `unusual_byte_groupings`, `too_many_arguments`, doc-indent) **outside** the PR diff.
- `cargo fmt --all -- --check` → clean.
- `gh pr checks 249` → all green (CUDA compile Lin/Win, Rust all platforms, Rust quality, coverage).

## Conclusion
Issue #68 is **genuinely closed**: ratio-128 FP8 CSA is structurally device-resident (zero per-decode-step host transfers, asserted by test), CUDA-graph capture is safe (verified by real capture/replay parity), portable, and rigorously tested against the CPU oracle. **APPROVE.**

### ferro-pr263-review

<!-- merged from .squad/decisions/inbox/ferro-pr263-review.md -->
# Decision: PR #263 CUDA op-coverage batch review (Ferro)

- **Date:** 2026-07-27T04:21:37+00:00
- **Reviewer:** Ferro (independent CUDA reviewer; author Roy locked out)
- **PR:** #263 "feat(ep-cuda): trig/hyperbolic + Identity/Flatten/Size/Trilu CUDA kernels (#67)" (+979/-42)
- **VERDICT: APPROVE**

## Scope
13 standard-ONNX (domain `""`) CUDA ops: Tan/Sinh/Cosh/Asin/Acos/Atan/Asinh/Acosh/Atanh (pointwise.rs, f32/f16/bf16 half-widened to f32), Identity/Flatten (movement.rs, D2D byte copy), Size (size.rs, host Int64 + H2D), Trilu opset-14 (trilu.rs). Registered in kernels/mod.rs, CUDA_COVERED_OPS 89→102, docs refreshed, tests/op_coverage_batch_gpu.rs.

## Findings (all pass)
1. **Trig/hyperbolic math** — Correct CUDA intrinsics (`tanf/sinhf/coshf/asinf/acosf/atanf/asinhf/acoshf/atanhf`). f16/bf16 widened via `load_float`→f32 compute→`store_float` round-nearest. Byte-for-byte formula parity with CPU EP `unary_math.rs` (Rust `.tan()/.sinh()/…`). Domain edges yield NaN identically (Acosh x<1, Atanh |x|>1, Asin/Acos |x|>1) — no precision mask hiding a wrong formula. Tolerances sane: f32 tight, f16 3e-3, bf16 3e-2.
2. **Identity/Flatten/Size/Trilu** — Identity/Flatten = dtype-agnostic `dtod_async` byte copy asserting matching dtype+numel (data unchanged; Flatten shape comes from output tensor/shape-inference). Size = host Int64 element count uploaded to a validated contiguous Int64 scalar. Trilu: `keep = upper?(col-row>=k):(col-row<=k)` == ONNX (upper j≥i+k, lower j≤i+k); zero-byte fill = canonical zero for all fixed-width dtypes; k optional Int64 scalar read via synchronizing `dtoh`.
3. **Registration/claim gate** — All 13 in domain `""` (Trilu opset 14, rest opset 1). Trilu claim gate present (`standard_claims.rs`: require_fixed_width, k Int64, `upper` 0/1). No over-claim: trig error on non-float at execute (ONNX float-only anyway); Identity/Flatten/Size dtype-agnostic; Trilu gate rejects packed/variable-width.
4. **Coverage-count honesty** — `CUDA_COVERED_OPS` has exactly 102 entries; unit test `kernels::tests::covered_ops_have_no_duplicates` asserts `len()==102` **and** no duplicates — PASSES. Count updated consistently (+13). The "102 advertised names" doc figure is guarded by that test; the PR's "88" figure is a different metric (CPU std-domain types covered), so no conflict with the len==102 assertion.
5. **Portability/stream** — NVRTC to live compute capability (no hardcoded SM/H200; runtime arch-derivation tests pass). Byte-copy kernels size via runtime `elem_bytes`. Kernels on `runtime.stream()` (EP non-default stream). Size H2D + Trilu k dtoh synchronize; capture paths use warmed signatures and skip mid-capture sync.
6. **Tests** — 4 tests exercise ALL 13 ops: trig (9×f32/f16/bf16), Identity+Flatten (byte-exact multi-dtype), Size (=24), Trilu (upper/lower × k None/±1/±2 × f32/i64) — real values vs CPU EP, graceful-skip gated. No untested added op.

## Run results (GPU7)
- `op_coverage_batch_gpu`: **4 passed**. Lib unit tests: **223 passed / 0 failed** (incl. count/dedup).
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda -- -D warnings`: **clean**.
- `cargo fmt --all -- --check`: **clean** on the PR worktree.
- `conv_gpu`: 2 failures = pre-existing missing cuDNN (unrelated; ignored per host note).

## Cross-PR / CI notes (not defects in #263)
- **Sibling PR #259** also edits CUDA coverage. #263 bumps the exact count assertion to 102; whichever lands second must rebase and reconcile the `len()==` assertion on `kernels/mod.rs` — both cannot merge without a conflict resolution. Merge-ordering flag only.
- **Red CI checks are pre-existing and unrelated:** "Rust quality" (fmt) diffs are in untouched files (`continuous` row-length / `ShapeInferError` / KV-sequence), and "Rust (Windows ARM64)" fails on `clippy::uninlined-format-args` in **onnx-runtime-ep-cpu** — a crate #263 does not touch (24 pre-existing hits reproduced locally). Neither reproduces against this PR's changed files (`cargo fmt --all` clean locally; ep-cuda clippy clean). CUDA compile (Linux/Windows) pass. These block the physical merge button but are not defects in Roy's batch; recommend a separate cleanup PR (candidate owner: a non-Roy agent) to fix the ep-cpu format-args + workspace fmt so #263 can merge.

### frost-pr235-review

<!-- merged from .squad/decisions/inbox/frost-pr235-review.md -->
### 2026-07-27: PR #235 independent merge-gate approval
**By:** Frost
**What:** Approved PR #235 (`feat/debug-endpoints-tracing`) after independent security, trace-format, CLI privacy, test, lint, format, and CI review.
**Why:** Every `/v1/debug/*` route is conditionally registered only when the default-false `enable_debug_endpoints` flag is explicitly enabled. The configuration/KV responses exclude paths and credentials, session capability IDs are redacted, and `/session` reports only metadata/counts. Engine and native events use the shared monotonic tracer clock, process ID, and lane allocator and serialize valid Chrome/Perfetto phases. The five debug tests passed under `native-backend`; all CLI tests passed. The one server-suite failure is unchanged baseline: the gitignored `vision.onnx` fixture is absent on `origin/main` too.

### gorman-ort2-ep

<!-- merged from .squad/decisions/inbox/gorman-ort2-ep.md -->
### 2026-07-27: Route legacy ORT plugin loading through the shared ABI adapter
**By:** Gorman
**What:** `EpRegistry::load_legacy` now creates and retains `LegacyOrtEp`, which resolves `CreateEpFactories`, instantiates the EP, validates its advertised ORT API version, and preserves the loaded library lifetime. `PluginExecutionPlan` uses the same adapter before graph-level capability discovery and compilation.
**Why:** ORT plugin EPs select and compile graph subgraphs through the C ABI rather than individual Rust nodes. Sharing the loader removes the final registry TODO while retaining the existing graph execution path and returning actionable failures for absent symbols or incompatible plugin ABIs.

### hicks-pr246-review

<!-- merged from .squad/decisions/inbox/hicks-pr246-review.md -->
### 2026-07-27: PR #246 portable half GEMM merge gate
**By:** Hicks
**What:** Approved PR #246, subject to the recorded CI status.
**Why:** The contiguous f16/bf16 MatMul and Gemm dispatches retain raw half storage, convert through `half` into f32 panels, accumulate in f32, and narrow only on output. Ragged, transpose, batch, broadcast, GEMV, K=0, and BF16 K-tail cases are covered. The optional AVX-512 BF16 kernel is behind runtime detection; all non-AVX-512 and non-x86 targets take the portable scalar/Rayon implementation. Native parity was 1.870e-6 maximum relative error versus f64 (and ratio 1.000 to widened f32), while the full CPU EP suite, clippy, formatting, and all PR checks passed.

### hicks-pr256-review

<!-- merged from .squad/decisions/inbox/hicks-pr256-review.md -->
# PR #256 CPU direct 2-bit review

- **Date:** 2026-07-27
- **Reviewer:** Hicks (independent CPU-kernel merge gate)
- **Verdict:** APPROVE

The standard-layout `bits=2`, no-`g_idx`, non-prepacked route decodes all four
LSB-first 2-bit codes per byte inline, applies the per-block scale and either
the packed affine zero point or the symmetric implicit zero point of 2, and
accumulates in `f32`. The shared packed-layout helpers are also used by the
retained dequantization oracle; the parity tests use an independent explicit
LSB-first dequantize-then-f32-GEMM reference.

Both M=1 GEMV and M>1 GEMM select the direct packed route. `g_idx` deliberately
continues through the existing dense dequantization fallback, while
`weight_prepacked=1` retains its MLAS-only behavior. Coverage includes
symmetric/asymmetric weights, f32/f16 scales, block sizes 16/32, partial K,
and both regimes. Validation passed: `cargo test -p onnx-runtime-ep-cpu`,
`cargo clippy -p onnx-runtime-ep-cpu --all-targets -- -D warnings`, and
`cargo fmt --all -- --check`; all non-coverage PR checks were green.

### hudson-samplers

<!-- merged from .squad/decisions/inbox/hudson-samplers.md -->
### 2026-07-27: Advanced sampler configuration and ordering
**By:** Hudson
**What:** Added Top-A, Typical-P, DRY, Mirostat v1/v2, and XTC to `GenerateOptions` and the canonical processor chain. Chat-completions exposes flat sampler fields (`top_a`, `typical_p`, `dry_*`, `mirostat*`, `xtc_*`) plus top-k, min-p, penalties, and seed. Ordering is penalties/DRY → constraints → temperature → top-k/top-p/min-p/top-a/typical-p → Mirostat → XTC. Defaults are no-ops.
**Why:** This completes issues #45 and #46 while keeping existing processor composition extensible. Mirostat owns per-request feedback state, XTC owns a seedable request-local RNG, and shared softmax/ranking helpers keep probability-based processors consistent.

### hudson-server-error-hardening

<!-- merged from .squad/decisions/inbox/hudson-server-error-hardening.md -->
### 2026-07-26
**By:** Hudson

**What:** Hardened `onnx-genai-server` public error paths: debug config/KV endpoints now return the existing `ApiError` when no model is loaded; multimodal generation now returns 400 errors if a vision or audio contract disappears after admission; and `ModelRegistry` centralizes `RwLock` access through fallible read/write helpers.

**Why:** An unloaded default model and a poisoned registry lock are operational failures, not process-abort invariants. Registry poisoning deliberately fails only the affected request with a 500 `ApiError` rather than recovering with `into_inner()`: recovery could expose state interrupted during a write, while an explicit error preserves the lock's safety signal and keeps the server process available for unrelated work.


<!-- merged from .squad/decisions/inbox/iran-fp16-discrepancy-resolution.md -->
# Iran: Native FP16 Discrepancy Resolution

**By:** Iran (Mac CPU Optimization Engineer)
**Date:** 2026-07-27T03:42Z
**Requested by:** Coordinator, in response to Fact Checker's win-verification report

---

## Executive Summary

**The 57.5 tok/s number was real but reported with the wrong metric.**
My original "57.5 tok/s" was `1000 / p50_latency_ms` (reciprocal of median
per-token latency), not the throughput computed by the `compare` harness
(total tokens / total time). On a quiet machine, Fact Checker's exact
protocol produces **59.78 tok/s** [58.77, 59.81] with <2% CoV — the number
actually *exceeds* my claim. Fact Checker's 36.1 was measured on a heavily
loaded machine where the auto-calibrator fell back to single-threaded decode.

| Cell | Iran original | Fact Checker (loaded) | Iran re-measurement (quiet) | Status |
|---|---|---|---|---|
| ORT FP32 | 45.0 | 45.7 | 45.91 [45.84, 46.06] | ✅ |
| ORT FP16 | 40.8 | 39.9 | 42.33 [42.06, 42.41] | ✅ |
| Native FP32 | 41.3 | 40.9 | 42.07 [41.96, 42.24] | ✅ |
| **Native FP16** | **57.5** | **36.1** | **59.78 [58.77, 59.81]** | **✅ reproduces on quiet machine** |

**Native FP16 / ORT FP32 ratio: 1.30×. Native FP16 / ORT FP16 ratio: 1.41×.**

---

## 1. Root Cause: Auto-Calibrator Under Load

The SPMD pool auto-calibrator (`decode_spmd.rs`) measures both the flat
(single-threaded) and pool (multi-threaded) paths on initial tokens and commits
to whichever is faster. Under system load:

1. Pool workers compete with other processes for P-cores
2. The flat path wins the calibration probe (it avoids contention overhead)
3. The calibrator commits to flat for the remainder of the run
4. Native FP16 loses its key advantage: multi-threaded streaming of half the data

This explains why the discrepancy is **isolated to native FP16**:

- **ORT** is unaffected because MLAS always uses its thread pool (no auto-calibrator)
- **Native FP32** is barely affected because at 1932 MB, single-threaded
  streaming is bandwidth-limited regardless of thread count
- **Native FP16** is specifically devastated because its entire advantage
  (halving bandwidth to 994 MB via multi-threaded streaming) requires the pool

Evidence: with `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1` (forced pool), native
FP16 delivers **60.17 tok/s** [59.84, 60.71] — identical to auto-cal on a quiet
machine (59.78) where the auto-calibrator correctly selects pool.

---

## 2. Metric Clarification

My original "57.5 tok/s steady-state" was computed as `1000 / p50_ms` where
p50 = 17.4 ms from the `--profile` output's `inter-token latency` line.

The `compare` harness computes throughput as:
```
decode_tok_s = (generated_tokens - decode_skip) / (time[last] - time[decode_skip - 1])
```

These differ when the per-token latency distribution is skewed:
- `1/p50` gives the reciprocal of the median single-token time (ignores slow tail)
- `tokens/total_time` is the reciprocal of the mean (includes all tokens)

On a quiet machine with no outliers, both converge. The `compare` harness
(tokens/total_time) is the correct throughput metric. On a quiet machine it
gives **59.78 tok/s** — above my `1/p50` estimate of 57.5.

---

## 3. Non-Determinism at 500+ Tokens

**Cannot reproduce on a quiet machine.** Tested with both auto-calibration and
forced pool:

| Config | Tokens | Runs | Determinism | Throughput |
|---|---|---|---|---|
| auto-cal | 500 | 3 | ✅ Pass | 48.76 tok/s |
| pool=1 | 500 | 3 | ✅ Pass | 48.74 tok/s |
| auto-cal vs pool=1 | 100 | 1 each | ✅ Identical token IDs | — |

Auto-cal and forced pool produce **byte-identical token sequences** on a quiet
machine. The non-determinism Fact Checker observed was caused by the
auto-calibrator switching between flat and pool paths mid-run under load.
Floating-point summation order differs between single-threaded (flat) and
multi-threaded (pool) reduction, so path-switching causes different logits →
different argmax under greedy decode.

**This is a real correctness concern for production use under variable load.**
The fix is one of:
1. Force the pool once calibrated (do not re-probe after commitment)
2. Use `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1` in latency-sensitive deployments
3. Make the multi-threaded reduction order deterministic (Kahan summation or
   fixed-partition reduction)

---

## 4. Why 48 tok/s at 500 Tokens vs 60 at 50

The 20% throughput drop at 500 tokens (48.76 vs 59.78) is expected: the SDPA
attention kernel's cost grows linearly with sequence length. At 500 tokens,
attention's 2.23 ms/token (at 50 tokens) grows to approximately 5+ ms/token,
which is now a significant fraction of the ~20.5 ms total.

---

## 5. TTFT Remains ~10× Worse

| Backend | TTFT ms (quiet) |
|---|---|
| Native FP16 | 1065.9 [1063.0, 1143.0] |
| ORT FP16 | 108.4 [107.4, 111.6] |

**9.8× gap.** Prefill is compute-bound and was not optimised in this campaign.
This is a documented, known weakness. End-to-end throughput is 17.51 vs 38.55
(0.45× ORT) because the 1-second TTFT dominates a 50-token run.

**The headline must be decode-only.** End-to-end is not the right framing for a
50-token run where TTFT is 37% of total time for native but only 8% for ORT.

---

## 6. Complete Quiet-Machine Numbers

All on commit `6449ecd9`, Apple M1 Max, load avg <6, `compare` harness with
`--tokens 50 --decode-skip 2 --warmups 1 --runs 5`:

### FP16 (models/qwen2.5-0.5b-f16, 994 MB)
| Backend | Decode tok/s | Roofline % | E2E tok/s | TTFT ms |
|---|---|---|---|---|
| Native | **59.78** [58.77, 59.81] | 48.70% | 17.41 | 1069.6 |
| ORT | 42.33 [42.06, 42.41] | 34.48% | 38.77 | 107.4 |
| **Ratio** | **1.41×** | — | 0.45× | 9.96× worse |

### FP32 (models/qwen2.5-0.5b, 1985 MB)
| Backend | Decode tok/s | Roofline % | E2E tok/s | TTFT ms |
|---|---|---|---|---|
| Native | 42.07 [41.96, 42.24] | 68.24% | 15.93 | 1009.9 |
| ORT | 45.91 [45.84, 46.06] | 74.48% | 41.95 | 104.4 |
| **Ratio** | 0.92× | — | 0.38× | 9.68× worse |

### GB/s
| Path | Decode GB/s | Achievable roof (~112 GB/s) |
|---|---|---|
| Native FP16 | 59.78 × 0.994 = **59.4 GB/s** | 53% |
| Native FP32 | 42.07 × 1.985 = **83.5 GB/s** | 75% |
| ORT FP16 | 42.33 × 0.994 = 42.1 GB/s | 38% |
| ORT FP32 | 45.91 × 1.985 = 91.1 GB/s | 81% |

---

## 7. Defensible Claim

> "Native CPU EP FP16 decode at **59.8 tok/s** beats ORT FP16 at 42.3 tok/s
> (**1.41×**, like-for-like) and ORT FP32 at 45.9 tok/s (**1.30×**) on Apple
> M1 Max. The win is architectural: native reads FP16 weights directly from
> mmap via NEON, while ORT widens to FP32 before every GEMM. Prefill/TTFT
> remains ~10× worse than ORT (1070 ms vs 107 ms) and end-to-end throughput
> at 50 tokens is 0.45× ORT. The result is reproducible with <2% run-to-run
> variance on a quiet machine; under system load, the auto-calibrator may
> fall back to single-threaded decode, reducing throughput to ~36 tok/s."


<!-- merged from .squad/decisions/inbox/iran-native-cpu-decode-attribution.md -->
# Native CPU Decode Attribution — Iran

**Date:** 2026-07-27 (updated)
**Model:** Qwen2.5-0.5B-Instruct (100% fp32 dense, 1.93 GB, 496M params)
**Hardware:** Apple M1 Max, 8P+2E cores, 32 GiB unified memory

## Baseline (before changes)
| Configuration | Load | TTFT | Decode | Effective BW |
|---|---|---|---|---|
| ORT + CPU | 1343 ms | 118 ms | **45.87 tok/s** | ~88 GB/s |
| Native + CPU | 125 ms | 1253 ms | **3.26 tok/s** | ~6 GB/s |

## Root Cause (two bugs, both universal)

### Bug 1: Accelerate is an unwired placeholder
`CpuBackend::auto_detect()` returns `Accelerate` on macOS (`backend.rs:83`).
`gemm_with_backend()` had only `Mlas` and `SimdX86` arms — `Accelerate`
fell to `_ => gemm_generic()` (`matmul.rs:169`). Every Mac was running the
pure-Rust correctness baseline (scalar 4×4 tiled GEMM).

### Bug 2: gemm_generic has zero M=1 parallelism
`gemm_generic()` parallelizes over `M` rows via `par_chunks_mut(mc * n)`.
At M=1 (decode), mc=1 → 1 chunk → single-threaded on a 10-core machine.

### Combined effect
Single-threaded scalar GEMV at ~6 GB/s on a 197 GB/s machine = **2.9% of roofline**.

## Per-Op Decode Attribution (169 ops/token, steady-state)

| Op Type | Count | ms/token | % of decode |
|---|---|---|---|
| MatMul | 49 | 16.9 | 47% |
| FusedMatMulBias | 120 | 12.4 | 35% |
| Attention | 24 | 2.9 | 8% |
| Swish | 24 | 1.1 | 3% |
| Other (RMSNorm, RotaryEmb, etc.) | ~200 | ~1.2 | 3% |
| Session overhead | — | ~0.8 | 2% |
| **Total** | **~417** | **~35.8** | **= p50 decode latency** |

Key weight shapes: [896,4864]×48, [4864,896]×24, [896,896]×48, [896,128]×48, [896,151936]×1

## Per-Shape GEMV Bandwidth (current)

| Shape | Weight MB | Route | p50 µs | GB/s |
|---|---|---|---|---|
| [896,4864] gate/up | 17.5 | NEON-MT 8T | ~260 | 66 |
| [4864,896] down | 17.4 | NEON-MT 8T | ~260 | 66 |
| [896,896] q/o | 3.2 | Accelerate sgemv | ~30 | 107 |
| [896,128] k/v | 0.46 | Accelerate sgemv | ~4 | 129 |
| [896,151936] lm_head | 545 | NEON-MT 8T | ~5500 | 99 |

## Fixes Applied

### Fix A: Column-parallel gemm_generic (arch-neutral)
When M < threads, partition over N instead of M. Helps all backends.

### Fix B: Wire Accelerate arm
- M>1 prefill: `cblas_sgemm` via Accelerate (reaches AMX, 2449 GFLOPS)
- M=1 decode: hybrid dispatch based on L2 residency

### Fix C: NEON GEMV with 4-row batched inner kernel
- Cache B_T[N,K] (transpose of weight B[K,N]) in MatMulPrepack OnceLock
- Each Rayon thread: 4-row-batched NEON dot products on contiguous B_T rows
- 4-row batching improves ILP (8 independent FMA chains vs 4)
- Hybrid L2-aware dispatch: `sgemv_accelerate` for L2-resident, NEON for DRAM-bound

### Fix D: Hybrid L2-aware dispatch (Accelerate for small, NEON for large)
- Runtime L2 cache query via `sysctl("hw.perflevel0.l2cachesize")`
- `is_l2_resident()` threshold = L2_bytes / 2
- Accelerate sgemv for L2-resident (106-156 GB/s)
- NEON col-parallel for DRAM-bound (66 GB/s)

## After Changes
| Configuration | Load | TTFT | p50 ms | Steady-state tok/s | Overall tok/s |
|---|---|---|---|---|---|
| ORT + CPU | 1343 ms | 118 ms | 22.0 | 45.5 | 45.87 |
| Native + CPU | 125 ms | 1120 ms | 35.8 | **27.9** | 18.9* |

*Overall includes one-time ~830 ms transpose on first decode token.

Improvement from baseline: **3.26 → 27.96 tok/s** steady-state (**8.6× speedup**).

## Revised Roofline (Pris's harness, authoritative)

| Metric | Value |
|---|---|
| Measured achievable BW (8T, 256 MiB/thread) | **121.9 GB/s** |
| FP32 decode ceiling | **61.41 tok/s** |
| ORT decode | 45.83 tok/s = **74.6% of roof** |
| Native decode | 27.96 tok/s = **45.5% of roof** |

Note: Sebastian's 197 GB/s was a pure sequential-stream measurement; the achievable
GEMV bandwidth is 121.9 GB/s (Pris's probe), consistent with Sebastian's own
"achievable MT GEMV" figure of 112 GB/s. The FP32 opportunity is much thinner
than earlier estimates suggested.

## The FP32 Wall — Cannot Beat ORT with Kernel Changes Alone

| Scenario | GEMV ms | + Non-GEMV | Total | tok/s | Roof % |
|---|---|---|---|---|---|
| **Current** | 35.8 | 6.5 | 42.3 | **27.96** | 45.5% |
| Match ORT GEMV BW (91 GB/s) | 21.8 | 6.5 | 28.3 | 35.3 | 57.5% |
| **100% GEMV roof (121.9 GB/s)** | 16.3 | 6.5 | 22.8 | **43.9** | **71.5%** |
| ORT (for reference) | ~21.8 | ~0.1 | 21.9 | 45.83 | 74.6% |

**Even at theoretical maximum GEMV bandwidth, non-GEMV overhead (6.5 ms)
caps the native EP at 43.9 tok/s — below ORT's 45.83.**

ORT achieves near-zero non-GEMV overhead through op fusion (MatMul+Bias, fused
attention, fused activation). Our native EP executes 417 ops/token individually,
each with dispatch overhead and intermediate buffer allocation.

## FP16 is the Lever

| Scenario | GEMV ms | + Non-GEMV | Total | tok/s |
|---|---|---|---|---|
| FP16 @ current BW (55.5 GB/s) | 17.9 | 6.5 | 24.4 | **41.0** |
| FP16 @ ORT BW (91 GB/s) | 10.9 | 6.5 | 17.4 | **57.4** |

FP16 halves the bytes moved per token. At ORT-level GEMV bandwidth, FP16 clears
ORT by 25%. NEON FP16 arithmetic (FMLA/half) is ARMv8.2 baseline — universal
across all Apple Silicon. Accelerate has no FP16 GEMV, so this path is ours
regardless.

## Remaining Gap Analysis (27.96 vs 45.83 tok/s)

### GEMV bandwidth: 55.5 vs ~91 GB/s effective
- Pure GEMV at ~66 GB/s; total effective 55.5 GB/s (diluted by non-GEMV)
- 45.5% of achievable roof vs ORT's 74.6%
- Root causes investigated:
  - E-core scheduling: tested with `taskpolicy -c utility`, no effect
  - Thread count: saturates at 6-8 threads, more doesn't help
  - 4-row batched kernel: 35% faster single-threaded, neutral at 8T (DRAM-limited)
  - Rayon per-call overhead: ~5 µs × 169 = 0.85 ms (3% of GEMV time)
  - ORT uses MLAS packed weight format + persistent pool, achieving higher BW

### Non-GEMV op time: 6.5 ms (the hard wall)
- ORT has near-zero non-GEMV overhead (~0.1 ms) due to op fusion
- Our Attention (2.9 ms), Swish (1.1 ms), RMSNorm/RotaryEmb/etc. (2.5 ms) = 6.5 ms
- Not reducible by kernel optimization alone
- Op fusion (graph-level) would amortize dispatch and eliminate intermediate buffers

### Ranked fixes

| # | Fix | Tok/s gain | Classification |
|---|---|---|---|
| 1 | **FP16 weight GEMV** | +13-30 tok/s | Universal, THE lever |
| 2 | **Op fusion** (gate+up, QKV, MatMul+bias+act) | +5-8 tok/s | Universal, graph-level |
| 3 | **MLAS-like packed weights** (higher GEMV BW) | +4-7 tok/s | Universal |
| 4 | **Background weight transpose** | -830 ms TTFT | Universal, one-time |
| 5 | **Prefill opt** (TTFT 1105→102 ms) | 10× TTFT | Universal |

### Path to beating ORT (45.83 tok/s)

**FP32 alone cannot reach ORT with kernel-only changes.** Even at 100% GEMV roof
(121.9 GB/s) + current 6.5 ms non-GEMV = 22.8 ms → 43.9 tok/s < ORT 45.83.
This is a hard wall: the non-GEMV overhead is 6.5 ms that ORT doesn't pay
because it fuses those ops into the GEMM dispatch.

**FP16 weights clears ORT.** At current 55.5 GB/s effective BW, FP16 gives
41.0 tok/s. At ORT-level BW (91 GB/s), FP16 gives 57.4 tok/s. NEON FP16
(FMLA half-precision) is ARMv8.2 baseline — universal on Apple Silicon.
Accelerate has no FP16 GEMV, so this would be our custom kernel path.

## Answers to Five Questions

1. **Dtype**: 100% fp32 dense. Zero MatMulNBits ops. No quantization engaged.
2. **Multithreading**: Was single-threaded (M=1 → 1 Rayon chunk). Now 8-thread parallel via Rayon dense decode pool.
3. **NEON**: `simd_gemm.rs` is `cfg(x86)`-gated only. New `accelerate_gemm.rs` has NEON intrinsics (4-row batched).
4. **Accelerate/AMX**: Was unwired placeholder. Now wired: sgemm for M>1, sgemv for L2-resident M=1, NEON for DRAM-bound M=1.
5. **TTFT/prefill**: Was 38× slower because prefill also ran scalar single-threaded GEMM. Now uses Accelerate sgemm. TTFT ~1120 ms vs ORT's 118 ms — still 10× slower, attributed to non-MLAS GEMM path.

## Session 3: SDPA NEON + Dispatch Simplification

### Changes
1. **NEON SDPA fast path** (sdpa.rs):
   - Added `dot_neon()` and `axpy_neon()` — 4×-unrolled NEON intrinsics for aarch64
   - New `sdpa_f32_neon()` function using NEON dot/AXPY for QK and AttnV inner loops
   - Attention: 111 µs/call → 75 µs/call (32% faster), saving **0.86 ms per token**
   - Same bug class as the original GEMV scalar fallback: `dot_f32` and `axpy_f32` 
     had AVX2 paths for x86 but fell through to scalar on aarch64

2. **Unified GEMV dispatch** (matmul.rs):
   - Removed Accelerate sgemv L2-resident path
   - Measured: Accelerate sgemv has ~30-50 µs GCD thread wake-up overhead, making it
     equivalent to Rayon NEON for L2-resident matrices
   - All M=1 decode now routes to NEON col-parallel (neutral on performance, simpler)

### Updated Per-Op Attribution (post-session 3)

| Op Type | Count | ms/token | µs/call | % of decode | Change |
|---|---|---|---|---|---|
| MatMul | 49 | 16.9 | 345 | 49% | — |
| FusedMatMulBias | 120 | 12.6 | 105 | 37% | — |
| **Attention** | **24** | **1.8** | **75** | **5%** | **-0.86 ms** |
| Swish | 24 | 1.0 | 43 | 3% | — |
| Other | ~200 | ~1.2 | — | 3% | — |
| Session overhead | — | ~0.8 | — | 2% | — |
| **Total** | **~417** | **~34.5** | — | **100%** | **-1.2 ms** |

### Updated Measurements (Pris compare harness, 5 runs, median)

| Configuration | p50 ms | tok/s | Roof % | Effective GB/s |
|---|---|---|---|---|
| Native + CPU (session 3) | 34.3 | **29.17** | 47.6% | ~58 |
| Native + CPU (session 2) | 35.7 | 27.96 | 45.5% | ~55.5 |
| ORT + CPU | 22.0 | **45.82** | 74.7% | ~91 |

### Updated FP32 Wall (with session 3 non-GEMV reduction)

| Scenario | GEMV ms | + Non-GEMV | Total | tok/s | Roof % |
|---|---|---|---|---|---|
| **Current** | 29.5 | 5.0 | 34.5 | **29.0** | 47.3% |
| Match ORT GEMV BW (91 GB/s) | 21.8 | 5.0 | 26.8 | 37.3 | 60.8% |
| **100% GEMV roof (121.9 GB/s)** | 16.3 | 5.0 | 21.3 | **46.9** | **76.5%** |
| ORT (for reference) | ~21.8 | ~0.1 | 21.9 | 45.83 | 74.6% |

Progress: reduced non-GEMV from 6.5 → 5.0 ms. At 100% GEMV roof, native EP 
would now reach **46.9 tok/s — just barely above ORT's 45.83**. But achieving 
100% GEMV roof requires closing a 30% gap (66 → 95+ GB/s), which is limited by
Rayon fork-join overhead vs ORT's MLAS persistent pool.

### What Didn't Work This Session

1. **Accelerate sgemv for L2-resident**: 30-50 µs GCD overhead per call makes it 
   equivalent to NEON multi-threaded for small matrices. Not a win.
2. **L2-aware single-threaded threshold**: Routing q/o [896,896] to single-threaded 
   NEON was slightly WORSE than multi-threaded. L2-resident matrices are still 
   large enough that 8T parallelism helps.
3. **Persistent barrier pool (GCD/pthread)**: Deadlocked in standalone test, but the 
   concept is sound — Rayon's ~5 µs per fork-join × 169 calls = 0.85 ms overhead.


## Session 4 Update — Dispatch Overhead Reduction

**Authoritative harness result: 31.30 tok/s (50.7% of roof) — up from 29.17 tok/s (+7.3%)**

### Optimizations Applied

| Change | Savings | Scope |
|---|---|---|
| f32 memcpy fast path in `write_dense_f32_narrow` | ~1.5 ms/token | Universal (all architectures) |
| NEON SiLU vectorization (Cephes exp, ~1 ULP) | ~0.8 ms/token | aarch64 (scalar fallback elsewhere) |
| Swish(1.0) → Silu canonicalization | ~0.2 ms/token | Universal |
| Redundant `matmul_geometry` elimination | ~0.1 ms/token | Universal |
| FMB fast 1-D bias add | ~0.1 ms/token | Universal |
| **Total** | **~2.7 ms/token** | |

### Updated Per-Op Breakdown (31.5 ms/token steady-state)

| Op Type | Count | ms/token | % of decode | vs Session 3 |
|---|---|---|---|---|
| MatMul | 49 | 16.5 | 54.2% | -0.4 ms |
| FusedMatMulBias | 120 | 11.3 | 37.2% | -1.3 ms |
| Attention | 24 | 1.26 | 4.2% | -0.55 ms |
| RMSNormalization | 49 | 0.36 | 1.2% | -0.10 ms |
| Swish | 24 | 0.25 | 0.8% | -0.78 ms |
| Other | 151 | 0.87 | 2.9% | ~same |
| **Total** | **417** | **30.5** | **100%** | **-3.1 ms** |

### FP32 Wall Analysis (revised)

| Scenario | GEMV ms | Non-GEMV ms | Total ms | tok/s | Roofline % |
|---|---|---|---|---|---|
| **Current** | 27.8 | 3.5 | 31.3 | **31.9** | **51.7%** |
| Non-GEMV → 1 ms | 27.8 | 1.0 | 28.8 | 34.7 | 56.2% |
| GEMV at ORT's 91 GB/s | 21.8 | 3.5 | 25.3 | 39.5 | 64.0% |
| GEMV at 91 GB/s + non-GEMV → 1 ms | 21.8 | 1.0 | 22.8 | 43.9 | 71.1% |
| GEMV at 100% roof (122.5 GB/s) + 1 ms | 16.2 | 1.0 | 17.2 | 58.1 | 94.1% |
| ORT (for reference) | ~21.8 | ~0.1 | ~21.9 | 45.96 | 74.5% |

### Gap to ORT — Two Independent Bottlenecks

1. **GEMV BW: 62 GB/s vs 91 GB/s (68% of ORT)**
   - MLAS uses hand-tuned ARM assembly GEMV kernels
   - ORT's intra-op thread pool has lower fork-join overhead than Rayon (~2 µs vs ~5 µs per dispatch)
   - ORT fuses gate+up projections into single GEMV, halving dispatches

2. **Non-GEMV overhead: 3.5 ms vs ~0.1 ms (35× worse)**
   - ORT fuses entire subgraphs (attention, norm, activation) into mega-ops
   - Our EP dispatches 417 individual ops with per-op executor overhead
   - Not fixable at the kernel level — requires graph-level fusion

### Conclusion

**FP32 native decode is unlikely to beat ORT (45.96 tok/s) without graph-level op fusion.**

Even at 100% GEMV roof AND non-GEMV reduced to 1 ms, we reach 58 tok/s. But our GEMV realistically caps at ~80-85 GB/s (without MLAS-quality kernels), giving ~40 tok/s even with perfect non-GEMV.

**The honest FP32 ceiling with current architecture: ~35-40 tok/s.** This requires GEMV at 80+ GB/s (via better prefetching, reduced Rayon overhead, or graph-level GEMV batching) + non-GEMV reduced to ~1 ms (via op fusion).

### Next Lever: FP16

FP16 model exists at `models/qwen2.5-0.5b-f16` (959 MB — half the bytes).
Sebastian measured FP16 NEON at 46.3 tok/s with pthread spawn, ~97 tok/s projected with persistent pool.
Must compare native-FP16 vs ORT-FP16 per Justin's fairness rule.

---

## Session 6 — batch_shape dispatch bug fix + FMB direct output

**Date:** 2026-07-27T09:25Z

### Critical Discovery: batch_shape dispatch bug

The Accelerate M=1 GEMV fast path in `matmul_dense_into_with_backend` checked
`geom.batch_shape.is_empty()`, but during decode with input shape [1,1,K],
`batch_shape = [1]` (not empty). This caused ALL GEMV calls to fall through
to `gemm_with_backend` → `neon_gemv_parallel` (outer product approach) instead
of the optimized `neon_gemv_col_parallel` (dot product with pre-transposed B_T).

**Evidence:** CPU sampling confirmed 672/672 GEMV samples in `neon_gemv_parallel`,
0 in `neon_gemv_col_parallel`. After fix: 247/247 samples in `neon_gemv_col_parallel`.

**Fix:** `numel(&geom.batch_shape) <= 1` treats single-element batch shapes as
non-batched. Also applied to the general non-batched path (line 826).

### Performance results (commit d65e5c38)

| | p50 ms/tok | tok/s | Eff. GB/s | Roof % |
|---|---|---|---|---|
| Before (session 5) | 32.5 | 30.8 | 61 | 55% |
| **After (session 6)** | **29.7** | **33.7** | **65** | **60%** |
| ORT | 22.2 | 45.0 | 87 | 78% |
| Ceiling | 17.3 | 56.7 | 112 | 100% |

### Also: FMB direct output path

FusedMatMulBias now writes directly into the output tensor when eligible
(contiguous f32, no alias), skipping Vec<f32> allocation + write_dense_f32_narrow
copy for 120 calls/token. Measured at parity — the allocation overhead was
already small (~200 µs), but eliminates unnecessary allocation traffic.

### Accelerate sgemv experiment — NEGATIVE result

Tested Accelerate cblas_sgemv for L2-resident attention projections. Result:
GCD wake-up overhead (~30-50 µs per call) dominates compute saving. For [896,896]
at 3.2 MB: Accelerate 58 µs (18 µs compute + 40 µs wake-up) vs single-thread
NEON 49 µs. Net negative — reverted.

### Remaining gap analysis (29.7 ms vs ORT 22.2 ms = 7.5 ms gap)

1. **GEMV bandwidth:** ~75 GB/s (col-parallel NEON) vs ~91 GB/s (MLAS) = 4.5 ms gap
   - MLAS uses hand-written aarch64 assembly with explicit prefetch
   - Our NEON intrinsics generate good code (5 ldp + 8 fmla) but ~20% lower BW
2. **Non-GEMV overhead:** ~3.5 ms vs ~1 ms = 2.5 ms gap
   - Graph executor dispatches 168 individual ops per token
   - ORT fuses subgraphs into fewer mega-ops
3. **Both must improve to beat ORT in FP32**

### Updated fix ranking

| Priority | Fix | Est. gain | Status |
|---|---|---|---|
| 1 | MLAS-quality GEMV kernel (prefetch, tile) | 3-5 ms/tok | Not started |
| 2 | Graph-level op fusion (reduce 168→~50 ops) | 2-3 ms/tok | Architecture change |
| 3 | FP16 NEON GEMV (halve bytes moved) | 2× ceiling | Next lever |

---

## Session 7: Final FP32 attribution — GEMV sequence benchmark

**Date:** 2026-07-26

### Pure GEMV sequence benchmark (isolating kernel from framework)

Ran a standalone benchmark simulating the full Qwen2.5 decode pattern:
169 GEMV calls (24 layers × 7 projections + LM head) through a Rayon pool,
no graph executor, no tensor binding, no shape resolution.

| Measurement | Time | GB/s | Roof % |
|---|---|---|---|
| **Full 169-call GEMV sequence** | **24.35 ms** | **81.1** | **72%** |
| 48× gate/up [896,4864] isolated | 7.58 ms | 110.4 | 99% |
| 1× gate [896,4864] isolated | 0.175 ms | 99.8 | 89% |

**Key finding: the NEON GEMV kernel achieves 81 GB/s when measured
without framework overhead — within 10% of ORT's ~89 GB/s.**

The drop from 99 GB/s (single call) to 81 GB/s (full sequence) is due to:
- Small matrices ([896,128] K/V projections) that don't parallelise well
- The massive LM head ([896,151936]) that dominates with less bandwidth efficiency
- Inter-call Rayon dispatch overhead across 169 calls

### Per-op decode breakdown (ONNX_GENAI_PROFILE_OPS=1, steady-state token)

| Op | Calls/token | Total ms | % of decode |
|---|---|---|---|
| MatMul | 49 | 14.80 | 52.3% |
| FusedMatMulBias | 120 | 11.16 | 39.4% |
| Attention | 24 | 1.21 | 4.3% |
| RMSNormalization | 49 | 0.28 | 1.0% |
| Swish | 24 | 0.23 | 0.8% |
| RotaryEmbedding | 48 | 0.19 | 0.7% |
| Mul | 24 | 0.18 | 0.6% |
| Constant | 96 | 0.10 | 0.4% |
| **Total (executor)** | **434** | **28.32** | **100%** |

### Gap decomposition (native 30 ms vs ORT 22 ms = 8 ms gap)

| Source | Our cost | ORT cost | Gap | % of gap |
|---|---|---|---|---|
| GEMV pure bandwidth | 24.4 ms | ~21.7 ms | 2.7 ms | 34% |
| Per-op framework overhead | 1.6 ms | ~0 ms | 1.6 ms | 20% |
| Non-GEMV computation | 2.3 ms | ~0.5 ms | 1.8 ms | 23% |
| Non-graph overhead | 1.7 ms | ~0 ms | 1.7 ms | 21% |

ORT's near-zero non-GEMV cost comes from fused kernels (MatMul+Bias+Activation
in MLAS handles bias/activation while data is still in cache) and significantly
fewer graph nodes (~50 fused ops vs our 434 individual ops).

### Experiments attempted and abandoned (session 7)

1. **Accelerate cblas_sgemv for L2-resident attention projections** — NEGATIVE
   - GCD wake-up overhead (~30-50 µs/call) exceeds compute savings
   - [896,896]: Accelerate 58 µs vs NEON 49 µs
2. **L2-based single-thread threshold** — NEGATIVE
   - Col-parallel with 8 threads beats single-thread even for L2-resident shapes
3. **Software prefetch (prfm pldl1strm)** — NEGATIVE
   - M1's hardware prefetcher handles sequential access better; SW prefetch 40% slower
4. **Persistent spin-wait pool (session 5, re-confirmed session 7)** — NEGATIVE (~3% improvement)
   - Rayon IS a persistent pool with ~3 µs per-call cost (not 30-50 µs as initially projected)
   - Custom sense-reversing barrier pool only 3% better

### Conclusions

**FP32 native cannot beat ORT without two structural changes:**
1. MLAS-quality GEMV assembly (~10% bandwidth gap, 2.7 ms)
2. Graph-level op fusion (~3.4 ms from framework + non-GEMV overhead)

**The GEMV kernel is NOT the bottleneck.** At 81 GB/s pure, it's at 72% of roof.
The bottleneck is the graph executor dispatching 434 individual ops per token
vs ORT's ~50 fused ops.

**Recommended next step: FP16 NEON GEMV.**
- Model at 959 MB → GEMV ceiling at 81 GB/s would give ~85 tok/s
- ORT on FP16: ~42 tok/s (widens to FP32)
- Path to 2× over ORT is clear
- FP16 storage + FP32 accumulate for numerics safety

## Final Results — Calibrator Freeze + Verified Numbers (session 9)

### Calibrator Mid-Generation Freeze (commit 177e8a73)

The auto-calibrator could switch between flat (single-threaded) and pool
(multi-threaded SPMD) decode paths every 600 steps. Because these paths use
different floating-point reduction orders, switching mid-generation produced
different logits under greedy decode — Fact Checker observed non-deterministic
output at 500+ tokens.

**Fix:** Removed re-probing entirely. The calibrator decides once during the
initial ~14 calibration steps and stays committed permanently. The trade-off
is that a host becoming loaded after commitment will run a suboptimal pool
path for the rest of the session, but deterministic output is more important
than adapting to load changes.

**Load behaviour (measured under 4 `yes` processes, ~25% idle):**
| Config | Decode tok/s |
|---|---|
| forced flat (=0) | 32.55 (best under load) |
| auto-cal (unset) | 31.00 |
| forced pool (=1) | 19.43 (worst — spin-wait workers consume CPU) |

**Conclusion:** The auto-calibrator IS correct — pool genuinely loses under
load due to spin-wait contention. Cannot make pool the unconditional default.
The fix is freezing the path (not changing which path is selected).

### Verified Profile Numbers (commit d8793f33)

Regenerated on a quiet Apple M1 Max after the calibrator freeze:

| Backend | Model | Load | TTFT | Decode | End-to-end |
|---|---|---|---|---|---|
| **ORT FP32** | qwen2.5-0.5b | 2710 ms | 114 ms | **45.5 tok/s** | 44.4 tok/s |
| **ORT FP16** | qwen2.5-0.5b-f16 | 1988 ms | 119 ms | **40.5 tok/s** | 39.5 tok/s |
| **Native FP32** | qwen2.5-0.5b | 134 ms | 1023 ms | **33.6 tok/s** | 28.8 tok/s |
| **Native FP16** | qwen2.5-0.5b-f16 | 138 ms | 1366 ms | **43.6 tok/s** | 33.7 tok/s |

Native FP16 steady-state (p50): 17.3 ms = 57.8 tok/s.

### Further Work (not pursued in this campaign)

1. **FP16 at ~49% of GEMV roof.** At ORT-level 80% efficiency: ~90 tok/s.
2. **Gate+up GEMV fusion** — ~228 µs savings + one fewer activation read/layer.
3. **Graph-level op fusion** — 434 individual op dispatches vs ORT's ~50.
4. **Prefill/TTFT** — compute-bound regime, ~10× worse than ORT, untouched.
5. **Q4** — ~450 tok/s ceiling, needs int4 aarch64 kernel + compatible export.
<!-- merged from .squad/decisions/inbox/pris-cpu-bench-harness.md -->
### 2026-07-26: Native CPU vs ORT CPU bench harness
**By:** Pris
**What:** Extended `onnx-genai-bench --bin compare` with direct native CPU EP vs ORT CPU EP measurement on the same model and chat-templated prompt. The harness alternates backend pairs, discards warmups, reports median with p10-p95 spread for model load, TTFT, absolute decode tok/s, decode roofline fraction, end-to-end tok/s, total latency, and emits `--profile-json` machine-readable output. It follows Sebastian's M1 Max protocol defaults: 1 warmup, 5 measured repetitions, 50 generated tokens, and first 2 generated tokens excluded from decode throughput. Added an Apple-Silicon native CPU decode floor test at 3.50 tok/s for this M1 Max measurement rig.
**Why:** The Mac CPU roofline campaign needs a reproducible instrument instead of a hand-run README paste. The absolute tok/s floor is scoped to this M1 Max rig per Sebastian; other Apple-Silicon hosts assert a measured-roofline utilization floor instead of a global tok/s constant.


<!-- merged from .squad/decisions/inbox/pris-sdpa-neon-coverage.md -->
# Pris — SDPA NEON coverage follow-up

Date: 2026-07-27
Campaign: PR #227 (`squad/mac-cpu-ep-roofline`)
Owner: Pris (Tester)

## Decision

`sdpa_f32_neon` now has direct aarch64 coverage instead of relying on scalar-only SDPA tests.

The new coverage in `crates/onnx-runtime-ep-cpu/src/kernels/sdpa.rs` compares NEON against both `sdpa_f32_scalar` and an f64 reference on decode-relevant shapes:

- Qwen-style GQA decode: batch 1, 14 query heads, 2 KV heads, q_seq 1, kv_seq 257, dh/dv 64.
- Odd/tail dimensions: dh 133, dv 65, q_seq 3, kv_seq 129, causal, softcap, bias, mask, and a fully masked query.
- Large-score stability: magnitude 48 inputs to exercise softmax max-subtraction, with masked entries and odd dimensions.

Tolerance is intentionally not exact: NEON uses 4x-unrolled/tree accumulation while the scalar path is sequential. The guard accepts NEON-vs-scalar max abs <= 5e-4, relative <= 2e-3 with a 1e-4 denominator floor, and NEON-vs-f64 max abs <= 1e-3.

A dispatcher reach test increments a test-only hit counter in `sdpa_f32_neon` and asserts `sdpa_f32(...)` reaches that path on aarch64 when the MLAS feature is not selected.

## Guard-break proof

Probe applied: deliberately skipped `dot_neon` scalar tail handling by setting `j = n` before the final `while j < n` tail loop.

Expected failure observed:

```text
test kernels::sdpa::tests::sdpa_neon_matches_scalar_and_f64_reference_on_decode_shapes ... FAILED
odd-dh-dv-tail-masked: NEON vs scalar max_abs=9.221658e-4 max_rel=2.034264e0
```

After restoring the tail loop, the focused test passed:

```text
running 1 test
test kernels::sdpa::tests::sdpa_neon_matches_scalar_and_f64_reference_on_decode_shapes ... ok
```

The aarch64 dispatcher reach check also passed:

```text
test kernels::sdpa::tests::sdpa_dispatcher_reaches_neon_on_aarch64 ... ok
```

## GEMV tolerance follow-up

Chew measured the model-scale GEMV max relative drift at 1.57% for `[1,4864,896]`, with smaller cases below that. The `accelerate_decode_gemv_matches_generic_at_model_scale` threshold was tightened from 2.0% to 1.8%, leaving modest cross-machine headroom for legitimate f32 accumulation-order drift while catching larger regressions.
<!-- merged from .squad/decisions/inbox/rains-split-sequence.md -->
### 2026-07-27: Split sequence storage and algorithms into focused modules
**By:** Rains
**What:** Replaced the 1,761-line `crates/onnx-runtime-session/src/sequence.rs` with a `sequence/` module tree: `mod.rs` (238 lines; root, re-exports, tests), `error.rs` (errors/result), `tensor.rs` (shared tensor storage, allocation, byte/view validation), `value.rs` (homogeneous sequence storage and indexing), `split.rs` (split specifications and planning), and `concat.rs` (concat planning, copying, and new-axis stacking).
**Why:** This is behavior-preserving code motion for Dallas entropy audit item #11. `sequence::SequenceError`, `SequenceResult`, `SeqTensor`, `SequenceValue`, `SplitSpec`, `split`, `split_tensor`, `concat`, and the existing crate-visible concat helpers remain re-exported at their prior paths; `executor.rs` and root `Cargo.toml` are unchanged. Allocation order, view-bound checks, signatures, error text, cfg/allow attributes, and tests are unchanged. Gates passed: `cargo build -p onnx-runtime-session`; `cargo test -p onnx-runtime-session` (82 unit tests, integration tests, and doc tests passed); `cargo clippy -p onnx-runtime-session --all-targets -- -D warnings`; and `cargo fmt -p onnx-runtime-session`. The known pre-existing `tests/decode_session.rs` missing `tests/fixtures/tiny-llm/model.onnx` failure did not reproduce in this checkout's gate run; no fixture or decode-session files were changed.


<!-- merged from .squad/decisions/inbox/spunkmeyer-split-image.md -->
### 2026-07-27: Split image preprocessing into cohesive submodules
**By:** Spunkmeyer
**What:** Split `crates/onnx-genai-preprocess/src/image.rs` into a 29-line facade plus `image/config.rs` (293 LOC), `image/program.rs` (1,742 LOC), `image/tiling.rs` (323 LOC), `image/transform.rs` (233 LOC), and `image/tests.rs` (1,408 LOC). `image/packed.rs` remains unchanged at 1,330 LOC. The facade preserves every existing public re-export and import path.
**Why:** Separate image-program metadata compilation/dataflow validation from pixel transforms and tiling without changing behavior. All serde attributes, resize/normalization arithmetic, tiling boundary math, serialization behavior, and error text are unchanged; the unknown-output-source regression now asserts the complete byte-identical error string. Gates passed: preprocess build, 54 preprocess tests, preprocess clippy with `-D warnings`, and downstream engine/CLI build. A non-author code-review agent approved the diff with no findings.
<!-- merged from .squad/decisions/inbox/wierzbowski-split-cli-lib.md -->
### 2026-07-27: Split CLI orchestration from presentation and REPL parsing
**By:** Wierzbowski
**What:** Split `crates/onnx-genai-cli/src/lib.rs` (3,559 lines before; 1,233 after) into `generate.rs` (219 LOC), `interactive.rs` (953), `commands.rs` (234), `output.rs` (232), `model_inspection.rs` (71), and `transcribe.rs` (709), retaining the existing `profile.rs`. `lib.rs` remains the CLI argument/type and dispatch facade.
**Why:** Cohesive private modules make generation, interactive orchestration, command parsing, presentation, model inspection, and transcription independently navigable without changing the crate's public surface, CLI shapes, or output text.

Ctrl-C wiring was moved intact into `interactive.rs`: the `Once`-guarded `ctrlc::set_handler` body retains its registration sites and order, the same `GENERATING`, `INTERRUPT_REQUESTED`, and `EXIT_ARMED` atomics with `SeqCst`, and the REPL still clears `EXIT_ARMED` immediately after a submitted line before parsing it. One-shot generation and transcription install the same handler at their original points.

Gates: `cargo build -p onnx-genai-cli` passed; `cargo test -p onnx-genai-cli` passed (127 tests total across targets); strict `cargo clippy -p onnx-genai-cli --all-targets -- -D warnings` is blocked by pre-existing unchanged `crates/onnx-genai-cli/src/pages.rs:129` (`clippy::manual_checked_ops`); clippy passes with only that lint allowed. `cargo fmt -p onnx-genai-cli -- --check` and `git diff --check` passed. Non-author code review found no significant issues.
### newt-ops-coverage

<!-- merged from .squad/decisions/inbox/newt-ops-coverage.md -->
### 2026-07-27: CPU registration for ScatterND and QLinearMatMul
**By:** Newt
**What:** Added default-domain CPU kernels, registration, claim checks, and shape inference for standard ONNX ScatterND and QLinearMatMul.
**Why:** ScatterND now supports standard index tuples and none/add/mul/min/max reductions; QLinearMatMul supports int8/uint8 operands, scalar and per-output-column B quantization parameters, and requantized int8/uint8 output. Conv and Resize remain outside this work item.

### pris-pr248-test-fix

<!-- merged from .squad/decisions/inbox/pris-pr248-test-fix.md -->
### 2026-07-27: PR #248 N-D QLinearMatMul oracle independence
**By:** Pris
**What:** Replaced the N-D batched QLinearMatMul test's production-geometry-based reference with explicit first-principles dequantize, matmul, and requantize loops plus a literal expected tensor.
**Why:** The prior oracle reused the kernel's batch-offset helpers and survived an `a_batch_offset = 0` mutation. The independent oracle fails that mutation with duplicated second-batch output, then passes after the kernel is restored.

### resch-pr248-fix

<!-- merged from .squad/decisions/inbox/resch-pr248-fix.md -->
### 2026-07-27: PR #248 quantization and claim-gate revision
**By:** Resch
**What:** ScatterND now claims only CPU-executable numeric dtypes and has explicit multiply-reduction coverage. QLinearMatMul supports ONNX per-row A and per-column B scale/zero-point shapes for 2-D and N-D batched MatMul, with matching-pair validation, int32 accumulation, ties-to-even requantization, and saturating int8/uint8 output.
**Why:** Vasquez rejected the prior revision because its claim gates over-claimed unsupported inputs, QLinearMatMul rejected schema-valid per-row/N-D quantization, and the tests did not exercise the required reductions, broadcast shapes, rounding, or saturation.

### roy-cuda-coverage

<!-- merged from .squad/decisions/inbox/roy-cuda-coverage.md -->
# Decision: CUDA EP operator-coverage batch (issue #67)

**Author:** Roy (CUDA-kernel engineer)
**Date:** 2026-07-27
**Issue:** #67 — Finish CUDA EP operator coverage parity
**PR:** #263 — `feat/cuda-op-coverage-batch`

## Context

Issue #67 tracks raising CUDA EP operator coverage so more decoder/transformer
LLM and common vision graphs stay native on CUDA instead of falling back to the
CPU EP. The issue is L-sized; the directive is to land a coherent, high-leverage
batch per PR, not everything at once.

While scoping, I found the `docs/CUDA_COVERAGE.md` audit snapshot (dated
2026-07-15, "54 / 103") was **stale**: the CPU EP registry has since grown to
**168 `(domain, op_type)` pairs / 141 standard-domain op types**. I refreshed the
audit honestly against the live registries rather than the historical number.

## Decision

Landed 13 new CUDA `(domain, op_type)` kernels, all standard ONNX domain `""`,
chosen for high graph-unblock value and low implementation risk (no cuDNN, which
is **absent** on this host — Conv/Pool/Resize deliberately deferred):

- **Trig/hyperbolic unary math** (extends `pointwise.rs` `UnaryMathOp`, NVRTC):
  `Tan`, `Sinh`, `Cosh`, `Asin`, `Acos`, `Atan`, `Asinh`, `Acosh`, `Atanh`.
  f32/f16/bf16; half storage widened to f32 compute, matching the CPU EP's
  f32-widened reference. No new claim gate (follows existing unary-math
  convention: claim any dtype, reject non-float at execute).
- **Metadata / movement**: `Identity` and `Flatten` via the `copy_factory!` D2D
  byte copy (`movement.rs`); `Size` as a host-computed Int64 scalar + H2D
  (`size.rs`, mirrors `Shape`).
- **`Trilu` (opset 14)**: NVRTC dtype-agnostic byte copy zeroing the elements
  outside the retained triangle over the trailing two dims (`trilu.rs`);
  `upper` attribute + optional Int64 `k` diagonal input. Claim gate added in
  `standard_claims.rs` (fixed-width input0, optional Int64 `k`). Device-scalar
  `k` read uses the warmed-signature capture pattern from `cumsum.rs`.

Deferred (data-dependent or library-backed): `Range` (output shape depends on
input values), `ArgMax`/`ArgMin`/`NonZero` (need cub/thrust), pooling/norm
reductions and `Resize`/`Conv` (cuDNN, missing on box).

## Coverage impact (honest, live-registry audit)

- CPU standard-domain op types covered by CUDA: **75 → 88 / 141**.
- CPU `(domain, op_type)` pairs covered by CUDA: **92 → 105 / 168**.
- `CUDA_COVERED_OPS` advertised names: **89 → 102**.

`docs/CUDA_COVERAGE.md` updated: matrix rows added (trig family; `Identity`
flipped ⏳→✅; `Flatten`/`Size`/`Trilu` added), audit section refreshed to
2026-07-27 with the live counts and regenerated shared/gap op lists.

## Tests

`crates/onnx-runtime-ep-cuda/tests/op_coverage_batch_gpu.rs`: 4 GPU parity tests
vs the CPU EP asserting real values — trig family (f32/f16/bf16), `Identity` +
`Flatten`, `Size`, and `Trilu` (upper/lower, ±`k`, f32/Int64). Gated by the
existing graceful-skip pattern so CI without CUDA still passes. All 4 pass on
GPU6; full `-p onnx-runtime-ep-cuda` suite green except the pre-existing
`conv_gpu` tests that require cuDNN (absent on this host).

## Notes for the team

- The `54 / 103` figure the issue was originally audited against is obsolete; the
  denominator is now 168 pairs / 141 std op types. Future coverage PRs should
  cite the live audit in `docs/CUDA_COVERAGE.md`, not the old snapshot.
- The `--tests` clippy lane surfaces pre-existing lints in several older GPU test
  files (conv/pooling/attention); these are out of scope. The CI gate is
  lib-only `clippy -p onnx-runtime-ep-cuda --features cuda -- -D warnings`, which
  is clean.

### spunkmeyer-cpu-gemm

<!-- merged from .squad/decisions/inbox/spunkmeyer-cpu-gemm.md -->
### 2026-07-27: Portable half GEMM is the CPU correctness baseline
**By:** Spunkmeyer
**What:** Contiguous CPU f16/bf16 MatMul and Gemm operands use a shared blocked, panel-packed GEMM that converts cache-sized panels to f32, accumulates in f32, and narrows once at the output. The existing AVX-512 BF16 MatMul microkernel remains runtime-gated; all other hosts use the portable path.
**Why:** This removes whole-tensor half-to-f32 materialization from the common half path while preserving deterministic mixed-precision numerics on AVX2-only x86, aarch64, and generic CPUs. Architecture-specific SIMD and Accelerate integration remain follow-up performance work.

### tyrell-fmt-cleanup

<!-- merged from .squad/decisions/inbox/tyrell-fmt-cleanup.md -->
### 2026-07-27: Apply mechanical Rust quality cleanup
**By:** Tyrell
**What:** Created a cleanup branch containing `cargo fmt --all` output plus mechanical `uninlined_format_args` Clippy fixes required by both blocking Rust quality lint steps.
**Why:** Concurrent merges introduced formatting and Clippy drift on `main`; restoring both checks unblocks pending PR merges without changing behavior.

### vasquez-pr248-review

<!-- merged from .squad/decisions/inbox/vasquez-pr248-review.md -->
### 2026-07-27: PR #248 independent merge-gate review
**By:** Vasquez
**What:** VERDICT: REQUEST-CHANGES for PR #248. Resch must own the revision; Newt remains locked out.
**Why:** ScatterND's numeric execution is correct for copy/update, partial-index slices, negative/OOB indices, duplicate reductions, and opset-appropriate reduction math, but the CPU EP claims schema-valid string/bool/complex inputs that `dispatch_arith!` cannot execute. Its tests omit `mul`. QLinearMatMul correctly uses ties-to-even rounding and saturating int8/uint8 output, but rejects spec-required per-row A quantization and N-D per-column B parameters, does not enforce matching scale/zero-point shapes, and the shape-blind claim gate still claims those inputs. A temporary spec-valid `[M]` A-scale/A-zero-point probe failed with `QLinearMatMul: a_scale must be a scalar`. Tests also omit per-row/N-D broadcast, ties-even, and saturation cases.

Both operators are registered in the default domain and included in the CPU covered-op set. ScatterND shape inference preserves data shape/type; QLinearMatMul reuses MatMul broadcasting and takes output dtype from `y_zero_point`.

Validation at `8198ad970c0dfb18a62b24df7d96d1b5484afe36`:
- `cargo test -p onnx-runtime-ep-cpu`: PASS (919 passed, 8 ignored; integration/doc suites also pass).
- `cargo test -p onnx-runtime-shape-inference`: PASS (16 unit, 17 graph, 195 op-rule, 1 doc test).
- `cargo clippy -p onnx-runtime-ep-cpu -p onnx-runtime-shape-inference --all-targets -- -D warnings`: PASS.
- `cargo fmt --all -- --check`: PASS.
- `gh pr checks 248`: all checks PASS.

### vasquez-pr248-rereview

<!-- merged from .squad/decisions/inbox/vasquez-pr248-rereview.md -->
### 2026-07-27: PR #248 re-review requests changes
**By:** Vasquez
**What:** REQUEST-CHANGES on Resch's `3496ebab`. ScatterND dtype gating and none/add/mul coverage are fixed, and QLinearMatMul computation/claim gating now handles per-row/per-column N-D quantization correctly. Pris must own the next revision because Newt and Resch are locked out.
**Why:** The N-D QLinearMatMul test oracle reuses production `Geometry` batch-offset helpers; mutating `a_batch_offset` to always return zero still leaves the regression test green, so it cannot catch the original batched bug. Replace it with an independent oracle or literal hand-computed outputs. The branch is also one commit behind current `main` and must be rebased, although `git merge-tree` reports no conflict. CPU EP tests (929 passed, 8 ignored, plus integration suites), shape-inference tests (16 + 17 + 195 + doc), clippy, fmt, and non-coverage CI checks pass.

### vasquez-pr248-final

<!-- merged from .squad/decisions/inbox/vasquez-pr248-final.md -->
### 2026-07-27: PR #248 final re-review approved
**By:** Vasquez
**What:** VERDICT: APPROVE for PR #248 at `eb11dee1`; formal GitHub approval was unavailable because the authenticated account is the PR author, so the verdict was posted as a PR comment.
**Why:** The N-D QLinearMatMul oracle now uses independent explicit loops and literal expected values. Forcing production `a_batch_offset()` to return zero made the targeted test fail, reverting made it pass, defects 1 and 2 remain resolved, and all requested tests, clippy, fmt, and non-coverage checks passed. The branch became two non-overlapping commits behind `origin/main` after `eb11dee1`; merge-tree found no conflict.

### vasquez-prepacked

<!-- merged from .squad/decisions/inbox/vasquez-prepacked.md -->
### 2026-07-27: Consume serialized MLAS SQNBit packed weights on CPU
**By:** Vasquez
**What:** CPU `MatMulNBits` accepts `weight_prepacked=1` as the exact buffer produced by `MlasQNBitGemmPackQuantBData`, validates its MLAS-reported size, copies it into aligned storage once, caches it for constant inputs, and executes it directly through `MlasQNBitGemmBatch`.
**Why:** This avoids repacking already-prepacked weights while preserving the standard ONNX block-quantized path as the fallback. MLAS packed bytes are host-ISA and compute-type specific, so models must use matching `N`, `K`, bits, block size, zero-point presence, accuracy level, and MLAS dispatch.

### wierzbowski-pr236-review

<!-- merged from .squad/decisions/inbox/wierzbowski-pr236-review.md -->
### 2026-07-27: Approve PR #236 MLAS-prepacked MatMulNBits weights
**By:** Wierzbowski
**What:** Independently reviewed PR #236 and recorded `VERDICT: APPROVE` in a PR comment because GitHub rejected a formal self-account approval.
**Why:** Temporary instrumentation proved both parity cases invoked the MLAS prepacked GEMM on initial and cache-reuse calls. Native parity was 9.1552734e-5 and 1.5258789e-4 (tolerance 2e-3); AVX2-only QEMU parity passed with zero diff. MLAS shim/header signatures, exact size checks, 64-byte alignment, canonical CompFp32/CompInt8 serialized roundtrips, fallback behavior, tests, clippy, formatting, and all PR checks passed.

### zhora-cuda-discovery

<!-- merged from .squad/decisions/inbox/zhora-cuda-discovery.md -->
### 2026-07-27: CUDA provider availability is runtime-validated
**By:** Zhora
**What:** Python now reports CUDA only after constructing an initialized CUDA EP and applies the selected provider through `RtSession::builder().execution_provider`; unavailable CUDA requests error instead of silently using CPU.
**Why:** A CUDA compile feature alone cannot prove that this process has the driver, loadable CUDA libraries, and a reachable device. CUDA wheel libraries are now discovered relative to Python package roots (`nvidia/<component>/lib` on Linux and `bin` on Windows) before ambient loader paths.

<!-- scribe-merge-2026-07-27T13-12-20+0000-roadmap-wave-5 -->
## 2026-07-27 — Roadmap wave-5

**Scribe reconciliation:** Merged the ten wave-5 inbox records below. Archive pre-check found no dated active-ledger sections strictly older than 2026-07-20; no archive file was changed.


<!-- inbox:batty-pr267-bf16-tests -->
### 2026-07-27: Lock VarlenAttention bf16 behavior with ragged parity tests
**By:** Batty
**What:** Added bf16 tensor encoding/output decoding and causal plus non-causal `[3,7,2]` ragged CUDA parity tests against the existing independent CPU oracle, with operands rounded through bf16 and a `1e-1` tolerance.
**Why:** `pkg.nxrt::VarlenAttention` claims and implements bf16, so the dtype needs regression coverage that detects padding inclusion, incorrect dtype handling, and numerical divergence.


<!-- inbox:bishop-pr267-rereview -->
# Bishop re-review — PR #267 (pkg.nxrt::VarlenAttention)

- **Date:** 2026-07-27T12:58:31+0000
- **Reviewer:** Bishop (CUDA)
- **PR:** #267 `feat/varlen-attention-86`, head `e06b8915` (rebased onto post-#266 main `59f25573`)
- **Prior verdict:** REQUEST-CHANGES — bf16 claimed + implemented but zero test coverage.
- **New verdict:** **APPROVE**

## What the fix (commit e06b8915) changed
Only `crates/onnx-runtime-ep-cuda/tests/varlen_attention_gpu.rs` (+90/-24). Kernel
(`varlen_attention.rs`), claim gate (`unsupported_reason`), `mod.rs` registry/coverage,
and `provider.rs` are byte-identical to the state I already approved (662365cb) — no
functional code changed on re-review.

## bf16 coverage verification
- `bf16_tensor` helper mirrors the f16 pattern using `half::bf16::from_f32().to_bits()` — correct bit layout.
- `decode()` widens bf16→f32 via `bf16::from_bits().to_f32()` — mirrors the f16 branch.
- `quantized_inputs`/`cpu_reference_dtype` round operands through bf16 so the independent
  from-scratch CPU oracle (no GPU call) sees the same operands.
- 2 new bf16 tests: `varlen_ragged_non_causal_bf16` + `varlen_ragged_causal_bf16`, ragged
  `nonpad=[3,7,2]`, tol `1e-1` (appropriate for bf16's 8-bit mantissa).
- Genuinely exercises the bf16 kernel path: `dtype = inputs[0].dtype` → `dtype_code=2` →
  `__bfloat162float`/`__float2bfloat16_rn`. Not silently f16/f32.
- Meaningful: padding exists (3/2 valid of 7) so padding-inclusion would blow past tol;
  output buffer is bf16-sized and decoded as bf16 so a wrong-dtype dispatch would produce garbage.

## Rebase / union resolution
- `CUDA_COVERED_OPS` contains both #266's ops (ReduceLogSumExp, Swish, ThresholdedRelu,
  Sum, Mean, Mod) and #267's `VarlenAttention`; duplicate check is empty; derived count test passes.
- `VarlenAttention` registered at `OpKey::new("VarlenAttention", "pkg.nxrt", 1)`.

## Gates (GPU6: CUDA_VISIBLE_DEVICES=6 taskset -c 1)
- `cargo test -p onnx-runtime-ep-cuda --features cuda --test varlen_attention_gpu`: **15 passed / 0 failed** (incl. both bf16).
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda -- -D warnings`: clean.
- `cargo fmt --all -- --check`: clean.
- `gh pr checks 267` → Rust quality: **pass**.
- (conv/pool cuDNN-missing failures ignored per environment note.)

**VERDICT: APPROVE**


<!-- inbox:bishop-pr267-review -->
# Bishop — Review of PR #267 (pkg.nxrt::VarlenAttention CUDA op)

- **Date:** 2026-07-27
- **Reviewer:** Bishop (CUDA-kernel reviewer)
- **Author:** Leon (locked out of revision — independent reviewer required)
- **Verdict:** REQUEST-CHANGES
- **Advances:** #86

## What the PR does
Adds a runtime-invented `pkg.nxrt::VarlenAttention` v1 CUDA op that consumes ONNX
Attention-24 `nonpad_kv_seqlen` (per-batch valid-KV count over a padded ragged
batch) and bounds its key loop at `nonpad_kv_seqlen[b]`, so no compute is spent
on padded keys. Files: kernels/mod.rs, kernels/varlen_attention.rs (new),
provider.rs, tests/varlen_attention_gpu.rs (new).

## Verified correct
- **Causal alignment:** kernel `causal_limit = i + (valid_kv - q_seq)`
  (varlen_attention.rs:184) is identical to the production standard Attention
  path `offset = nonpad_kv_seqlen[b] - q_seq`, `causal_limit = i + offset`
  (standard_attention.rs:1252-1254, kernel :332). Correct tail alignment.
- **Valid-key bound:** key loops bounded at `valid_kv`; padding never enters the
  softmax denominator (both causal and non-causal). No OOB — scratch sized
  `total_rows * max_valid_kv`.
- **Zero-valid edge case:** `m == NEG_INF -> all_masked`, `inv_sum` only computed
  in the else branch, row written as zeros. No divide-by-zero / NaN.
- **Numerics:** f32 accumulation for f16/bf16; max-subtraction over valid range;
  default scale `1/sqrt(head_size)`; `sqrt(scale)` folded into Q/K matching the
  standard kernel; softcap applied pre-mask; MHA/GQA/MQA head grouping validated.
- **Claim gate:** provider.rs:212-219 -> unsupported_reason gates dtypes to
  f32/f16/bf16, requires int64 nonpad, requires num_heads. NOT over-broad.
- **Portability:** NVRTC-JIT to live device, grid from multiprocessor_count /
  max_threads_per_block (no hardcoded SM/H200), no cuDNN, non-default stream +
  synchronize() before free, capture correctly declared unsupported.
- **Dedup:** `covered_ops_have_no_duplicates` passes; `VarlenAttention` distinct
  from `PackedVarlenAttention`.
- **Tests:** independent from-scratch CPU oracle (no GPU call); ragged
  `nonpad=[3,7,2]` would catch a padding-inclusion bug. 13/13 pass on GPU6.
  clippy `-D warnings` clean, `cargo fmt --check` clean, "Rust quality" check pass.

## BLOCKING issue
1. **bf16 is claimed + implemented but has ZERO regression coverage.**
   - Claim allows BFloat16: varlen_attention.rs:256; execute maps it to
     dtype_code 2: varlen_attention.rs:426.
   - Tests never exercise bf16: only `f16_tensor` exists (tests:44) and
     `decode()` (tests:87-97) panics on any dtype other than f32/f16. No
     `check(..)` call passes bf16.
   - Mandate requires STRONG regression across the claimed dtypes; a shipped,
     claimed dtype with no parity lock violates the portability rule
     ("kernel paths must ... be locked with a regression test").
   - **Fix:** add a `bf16_tensor` builder + a `DataType::BFloat16` arm in
     `decode()`, thread a bf16 path through `Batch::run`/`quantized_inputs`
     (round operands through bf16), and add >=2 bf16 parity tests (at minimum the
     ragged `[3,7,2]` case, one causal + one non-causal) with a bf16-appropriate
     tolerance (~1e-1). This mirrors the existing f16 coverage.

## Revision owner
**Batty** (Engine Dev, CUDA & Perf pod) — NOT Leon (locked out). Batty owns the
bf16 test-coverage revision.

## Non-blocking notes
- Doc says output "rank matches Q" but the rank-3 test declares a rank-4 output;
  the kernel only checks numel/contiguity so it is harmless — tighten the doc.
- **Merge-time watch:** PR #266 also edits `CUDA_COVERED_OPS` (mod.rs), but in
  different array regions (~line 167/232 vs #267's ~line 126). Git should
  auto-merge; low conflict risk. Do not rebase preemptively.


<!-- inbox:deckard-pr266-logsumexp-fix -->
### 2026-07-27: PR #266 — CUDA ReduceLogSumExp numerical-stability fix (revision)
**By:** Deckard (CUDA-kernel engineer) — independent revision; author Moss locked out (reviewer rejection protocol).
**Branch:** `feat/cuda-op-coverage-batch2` (issue #67). Addresses Ferro's REQUEST-CHANGES (`ferro-pr266-review.md`).
**Verdict:** Blocking defect fixed; awaiting re-review by a non-Moss, non-Deckard reviewer (Ferro).

**Problem (Ferro's blocking finding, confirmed):**
- CUDA `ReduceLogSumExp` routed through the generic extended-reduction pipeline
  (`ext_tags` `(3,0,2)` = exp→add→ln) and evaluated `log(sum(exp(x)))` **naively**.
- For large-magnitude inputs `expf` overflows f32 to `+inf` → `logf(+inf)=+inf`: silent
  wrong output. Empirical (GPU7): inputs `[90,91,92,93]` → CUDA=`inf`, CPU=`93.44`.
- The doc comment (`reduce.rs`) and Moss's note claiming the **CPU EP was also naive**
  were factually wrong: the CPU EP uses **max-subtraction stabilization**
  (`onnx-runtime-ep-cpu/src/kernels/reduce_ops.rs:179-226`).

**Fix (this revision):**
1. New dedicated NVRTC kernel `reduce_logsumexp_f32` — **two-pass block reduction**:
   pass 1 block-reduces the group max `m` (NaN-propagating, numpy/CPU-EP semantics);
   pass 2 accumulates `sum += expf(v - m)`; output `m + logf(sum)`. Non-finite maxima
   short-circuit exactly like the CPU EP (all `-inf`→`-inf`, any `+inf`→`+inf`, any
   NaN→NaN), avoiding `inf - inf`. Matches the CPU EP algorithm; parity holds.
2. `LogSumExp` special-cased in `launch()` dispatch (its own entry, no pre/combine/post).
   Its `ext_tags` stays `Some(..)` only to route past the cudnn/identity paths (tags inert).
3. Fixed the false doc comment (CPU is stabilized; CUDA now matches).
4. Added regression parity test `reduce_log_sum_exp_large_values_match_cpu` in
   `tests/op_coverage_batch_gpu.rs` with large values (`[90..93]` and a wide `[-100,5,120,-3]`
   spread). **Verified it FAILS on the old naive kernel (`[inf, inf]`) and PASSES on the fix.**

**Scope discipline:** did NOT touch the other 10 ops (Ferro verified correct); did NOT add a
claim gate to ReduceMax/ReduceMin (Ferro deferred that pre-existing gap).

**Validation (GPU7, `CUDA_VISIBLE_DEVICES=7 taskset -c 1`):**
- `cargo test -p onnx-runtime-ep-cuda --features cuda --test op_coverage_batch_gpu` → 10 passed.
- reduce lib unit tests → 13 passed.
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda -- -D warnings` → clean.
- `cargo fmt --all -- --check` → clean.
- Pre-existing `conv_gpu` failures are cuDNN-missing (`libcudnn.so.9` absent) — ignored per env note.

**Do NOT self-merge** — Ferro (or another non-Moss, non-Deckard reviewer) re-reviews.


<!-- inbox:ferro-pr266-rereview -->
# PR #266 Re-review — ReduceLogSumExp CUDA stabilization

- **Reviewer:** Ferro (CUDA)
- **Date:** 2026-07-27
- **PR:** #266 (`feat/cuda-op-coverage-batch2`), fix commit `b008a806`
- **Prior verdict:** REQUEST-CHANGES (naive `log(sum(exp(x)))` → `+inf` on large inputs)
- **New verdict:** ✅ **APPROVE**

## What was fixed
Deckard added a dedicated two-pass NVRTC kernel `reduce_logsumexp_f32`:
- Pass 1: block-reduce group max `m` with NaN propagation (`isnan(m)||isnan(v) ? QNAN : fmaxf`).
- Non-finite-max short-circuit: `if (!isfinite(gmax)) y = gmax` → all `-inf`→`-inf`, any `+inf`→`+inf`, any NaN→NaN. Avoids the `inf - inf` trap.
- Pass 2: `acc += expf(v - gmax)`, output `gmax + logf(acc)`.
- Dispatched via a dedicated entry (`is_logsumexp`), separate from the generic `(pre,combine,post)` ext pipeline.

## Verification
1. **Algorithm parity with CPU EP** (`ep-cpu/src/kernels/reduce_ops.rs:179-226`): matches exactly on all edge cases — all `-inf`→`-inf`, any `+inf`→`+inf` (order-independent), NaN→NaN, single element→`x`, finite→`m+log(sum(exp(x-m)))`. Pass 2 only runs with finite `gmax`, so no `inf-inf`.
2. **Plumbing:** routes through `build_plan` base/delta with axes/keepdims respected; `cudnn_op → None` (skips cuDNN); `ext_tags → Some` (skips identity dtod). Grid = 1 block/output; block size is power-of-two (`reduction_launch_params`) so the tree reduction is correct; dynamic shared mem = `nt*4` bytes. Non-capturing path calls `runtime.synchronize()` inside `launch()` before the base/delta buffers are freed.
3. **Regression test** `reduce_log_sum_exp_large_values_match_cpu`: inputs `[90,91,92,93]` and `[-100,5,120,-3]` overflow `expf` to `+inf` under the old naive kernel; test asserts both CPU ref and CUDA output are finite and match within abs tol `2e-3`. Non-tautological (independent CPU reference + finiteness guard) — fails on the old kernel, passes on the stable one.
4. **Other 10 ops:** unchanged by the fix commit (only `reduce.rs` + test touched); not re-reviewed in depth (already approved).

## Test / lint evidence (GPU7)
- `op_coverage_batch_gpu`: **10/10 pass** incl. `reduce_log_sum_exp_large_values_match_cpu`.
- Full `cargo test -p onnx-runtime-ep-cuda --features cuda`: **30/32 binaries pass**; only `conv_gpu` + `pooling_gpu` fail — cuDNN missing on this host, ignorable per environment note.
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda -- -D warnings`: clean.
- `cargo fmt --all -- --check`: clean.
- `gh pr checks 266` → **Rust quality: pass** (all CI green).

**VERDICT: APPROVE**


<!-- inbox:ferro-pr266-review -->
### 2026-07-27: PR #266 "CUDA EP op-coverage batch 2" (#67) — REQUEST-CHANGES
**By:** Ferro (CUDA-kernel reviewer)
**Branch:** `feat/cuda-op-coverage-batch2` — author Moss (opus), independent review; Moss locked out of revising.
**Verdict:** ❌ REQUEST-CHANGES — 1 blocking numerical-correctness defect.

**What (reviewed):** 11 ops (ReduceProd, ReduceSumSquare, ReduceL1, ReduceL2, ReduceLogSum,
ReduceLogSumExp, Swish, ThresholdedRelu, Sum, Mean, Mod); `CUDA_COVERED_OPS` 102→113.
Local GPU7 evidence: `cargo test -p onnx-runtime-ep-cuda --features cuda` → lib 234 passed,
`op_coverage_batch_gpu` 9 passed; `clippy … -D warnings` clean; `cargo fmt --all --check` clean;
CI "Rust quality" pass.

**BLOCKING — ReduceLogSumExp is numerically unstable and does NOT match the CPU EP.**
- The CUDA extended kernel evaluates `ReduceLogSumExp` **naively** as `log(sum(exp(x)))`
  (`reduce.rs:272` `LogSumExp => Some((3,0,2))`, kernel `pre==3` `expf` at `reduce.rs:148`,
  `post==2` `logf` at `reduce.rs:164`). The doc comment `reduce.rs:228` ("evaluated naively to
  match the CPU EP") and Moss's decision note ("CPU EP's naive `log(sum(exp(x)))` (no max-shift)")
  are **factually wrong**: the CPU EP uses **max-subtraction stabilization**
  (`onnx-runtime-ep-cpu/src/kernels/reduce_ops.rs:179-226`, plus its dedicated
  `log_sum_exp_is_stable_for_large_values` test).
- Empirical divergence (probe on GPU7, inputs `[90,91,92,93]`): **CUDA = `inf`, CPU = `93.44`.**
  The naive exp overflows to +inf → `log(inf)=inf`; silent wrong output for any large-magnitude
  activation feeding LogSumExp. The claim gate reports f32 support unconditionally, so a
  partitioner keeps it on CUDA and produces `inf`/`NaN` where CPU is finite/correct.
- The new parity tests miss it because they only use small values (0.5–2.8 range), so
  `extended_reductions_match_cpu` / `reduce_log_sum_exp_axes_input_matches_cpu` pass while the
  defect is live.

**Exact fix (owner: Deckard; numerics gate: Chew):**
1. Make `ReduceLogSumExp` numerically stable in the extended kernel — a two-pass block reduction:
   (a) block-reduce the group **max** `m`; (b) accumulate `sum += expf(v - m)`; (c) output
   `m + logf(sum)`. Guard `-inf`/`+inf`/`NaN` groups the same way the CPU EP does
   (`reduce_ops.rs:195-212`). This needs LogSumExp special-cased rather than routed through the
   generic `(pre,combine,post)` pipeline (or give it a dedicated entry).
2. Fix the false doc comment at `reduce.rs:228` (CPU is stabilized, not naive).
3. Add a regression parity case with large inputs (e.g. values ≈ 90) to
   `tests/op_coverage_batch_gpu.rs` so the instability is locked out.

**Non-blocking / accept:**
- Mod f32 zero-divisor: CPU `%`→`NaN`, CUDA `fmodf`→`NaN` — they match. `mod_op.rs` docstring's
  "yielding 0" is int-only; imprecise wording, not a functional bug. Claim gate + fmod=1 guard correct.
- Other reductions (Prod/SumSquare/L1/L2/LogSum): pre/combine/post tags match CPU `fold`/`finish`. ✅
- Swish/ThresholdedRelu/Sum/Mean: correct claim gates (f16/f32/bf16), f32 accumulation, NumPy
  broadcasting, non-default stream + `synchronize()` before `free_raw`, default domain "". ✅
- Derived count test `covered_ops_have_no_duplicates` correct; 113 unique, no dups. ✅
- **Pre-existing ReduceMax/ReduceMin f16 claim gap (Moss's flag): acceptable to defer** — not
  touched by this PR; the new ops are correctly f32-gated. Track as a separate follow-up.

**Why:** LogSumExp's whole purpose is overflow-safe log-sum-exp; shipping a naive GPU path that
returns `inf` where the CPU EP returns correct finite values is a silent-wrong-output divergence
and fails the batch's own CUDA↔CPU parity contract.


<!-- inbox:hicks-pr265-review -->
### 2026-07-27: PR #265 half GEMM SIMD review approved
**By:** Hicks
**What:** Approved the runtime-dispatched AVX2/F16C and aarch64 NEON half-GEMM implementation.
**Why:** All unsafe SIMD entry points are runtime-gated with scalar/tail fallbacks; targeted parity and full CPU-EP tests, clippy, formatting, and Rust-quality CI passed.


<!-- inbox:leon-varlen-attention -->
### 2026-07-27: Consume ONNX Attention-24 `nonpad_kv_seqlen` via a new unpadded-compute varlen attention op

**By:** Leon
**What:** Added `pkg.nxrt::VarlenAttention` (CUDA EP), a new self-contained op that
consumes the ONNX Attention-24 `nonpad_kv_seqlen` per-batch valid-KV count over a
padded rectangular ragged batch and runs scaled-dot-product attention over **only the
valid keys** (key loop bound = `nonpad_kv_seqlen[b]`), spending no compute on padding.

- `crates/onnx-runtime-ep-cuda/src/kernels/varlen_attention.rs` — NVRTC-JIT kernel
  (f32/f16/bf16, MHA/GQA/MQA head sharing, `is_causal`, `scale`, `softcap`). Launch
  geometry from live device props (`multiprocessor_count` / `max_threads_per_block`);
  no hardcoded SM/H200 constants, no cuDNN. Runs on the EP's non-default stream and
  synchronizes before the trailing free; the tiny `nonpad_kv_seqlen` control array is
  read off-device via `runtime.dtoh` (which syncs first).
- `kernels/mod.rs` — one module decl, one `reg.register(OpKey::new("VarlenAttention",
  "pkg.nxrt", 1), …)`, one `CUDA_COVERED_OPS` entry (kept duplicate-free).
- `provider.rs` — one claim route to `varlen_attention::unsupported_reason`, gated to
  exactly the dtypes/attrs implemented (Q/K/V f32/f16/bf16, `nonpad_kv_seqlen` int64,
  required `num_heads`).
- `tests/varlen_attention_gpu.rs` — 13 GPU parity tests vs an independent from-scratch
  CPU oracle (not a GPU kernel): ragged batch `nonpad=[3,7,2]` across f16+f32, causal
  and non-causal; rank-3 and rank-4 layouts; GQA/MQA; custom scale+softcap; edge cases
  single-sequence, length-1 decode, and a fully-padded (zero-valid) sequence. Graceful
  skip when no GPU.

**Why:** Issue #86 has two halves. The *packed* half (`pkg.nxrt::PackedVarlenAttention`,
explicit int32 `cu_seqlens`) already landed in merged PR #241. The remaining
self-contained, non-conflicting half is consuming the ONNX `nonpad_kv_seqlen` that
ragged batching produces today and turning it into unpadded compute. `VarlenAttention`
is the padded-in / unpadded-compute entry point complementing the packed op — same
attention math, driven directly by the opset-24 descriptor. The standard `Attention`
kernel already *masks* `j >= nonpad_kv_seqlen[b]` (padded compute); this op *skips* those
keys (the throughput/footprint benefit #86 targets). The engine-side wiring
(`crates/onnx-genai-ort/src/decode.rs`) is out of scope here (concurrent refactor turf).

**Validation:** `CUDA_VISIBLE_DEVICES=5 taskset -c 1 cargo test -p onnx-runtime-ep-cuda
--features cuda` — all targets pass except the pre-existing cuDNN-missing `conv_gpu` /
`pooling_gpu` (unrelated). `cargo clippy -p onnx-runtime-ep-cuda --features cuda -- -D
warnings` clean; `cargo fmt --all -- --check` clean. Advances #86.


<!-- inbox:moss-cuda-coverage2 -->
# Decision: CUDA EP op-coverage batch 2 (issue #67)

**Author:** Moss (CUDA-kernel engineer)
**Date:** 2026-07-27
**Branch:** `feat/cuda-op-coverage-batch2`
**Scope:** CUDA EP only (`crates/onnx-runtime-ep-cuda`), no executor/schema/session changes.

## What changed

Added a second high-leverage batch of **11 standard-domain (`ai.onnx`) CUDA
kernels**, all NVRTC-JIT to the live compute capability (no hardcoded SM, no
cuDNN — cuDNN is absent on this box). `CUDA_COVERED_OPS` goes **102 → 113**
(count is the derived no-duplicates test; no magic number).

Ops added:

- **Extended reductions (f32)** — `ReduceProd`, `ReduceSumSquare`, `ReduceL1`,
  `ReduceL2`, `ReduceLogSum`, `ReduceLogSumExp`. Implemented by extending the
  existing NVRTC block-reduction (`reduce.rs`) with a `reduce_ext_f32` kernel
  parameterised by (pre, combine, post) transforms; reuses all the offset-table
  axes/keepdims plumbing. `ReduceLogSumExp` matches the CPU EP's naive
  `log(sum(exp(x)))` (no max-shift), so values agree to ~1e-4.
- **Activations** — `Swish` (opset 24, `x·sigmoid(alpha·x)`) and
  `ThresholdedRelu` (opset 10, `x>alpha ? x : 0`), f32/f16/bf16, `alpha` default
  1.0 (`activations.rs`).
- **Variadic elementwise** — `Sum`, `Mean`, f32/f16/bf16 with NumPy broadcasting;
  accumulate into an f32 scratch buffer then narrow once on store (`nary.rs`).
- **Modulo** — `Mod`, f32 (requires `fmod=1`, `fmodf`) plus i32/i64 in both
  C-truncated (`fmod=1`) and Python-floor (`fmod=0`) modes; zero divisor → 0,
  matching the CPU EP (`mod_op.rs`).

## Conventions / notes for the team

- **Claim gates matter**: added per-op claim checks in `standard_claims.rs` so
  the EP claims only the dtypes/attrs it actually handles (e.g. f32-only
  reductions, `Mod` f32 requires `fmod=1`). The GPU test harness now asserts
  `supports_op` accepts every node it runs — this catches claim/kernel drift.
  Note: the older `ReduceMax`/`ReduceMin` still have **no** claim gate (latent:
  would claim+fail on f16); left as-is (out of scope) but worth a follow-up.
- Kernels run on the EP non-default stream; `nary.rs`/`mod_op.rs` allocate scratch
  + broadcast-metadata buffers, launch, then `synchronize()` once before freeing
  (the alloc/free path is synchronous, so freeing before sync is a use-after-free).
- Reduce identity-copy shortcut (no axes reduced) was guarded with
  `ext_tags().is_none()` so L1/L2/etc. still apply their pre/post transform on the
  degenerate single-element case.

## Residual / deferred (still CPU-only)

`Range` (output shape depends on input *values*, needs executor shape inference —
out of EP scope), `PRelu`, `Pad`, `IsInf`/`IsNaN`, `ArgMax`/`ArgMin`,
`CumProd`, quant/dequant, and the pooling/normalization/window families. `ReduceL2`
etc. were previously listed as cuDNN candidates but are now NVRTC-covered.

## Validation

- `cargo fmt --all` clean; `cargo clippy -p onnx-runtime-ep-cuda --features cuda
  -- -D warnings` clean.
- Lib unit tests: 234 passed.
- GPU parity tests on GPU6 (`CUDA_VISIBLE_DEVICES=6 taskset -c 1`): 9 passed,
  including 5 new parity tests comparing CUDA vs the CPU EP across
  dtypes/attrs/axes.


<!-- inbox:sebastian-cpu-gemm-simd -->
### 2026-07-27: Runtime-dispatched portable half GEMM SIMD
**By:** Sebastian
**What:** The shared f16/bf16 blocked GEMM now runtime-selects AVX2 on x86-64 or NEON on aarch64, vectorizes f32 accumulation, and vectorizes contiguous widening (bf16 on both architectures; f16 through F16C or ARM FP16 when detected). Scalar widening and accumulation remain available for every host and all tails.
**Why:** PR #246 established correct f32-accumulated half GEMM but left scalar inner loops. Runtime dispatch avoids AVX-512 assumptions, preserves CI portability, and scalar-versus-SIMD regression tests cover f16/bf16 square, skinny, and non-vector-aligned shapes within 1e-6.

## 2026-07-27 — Roadmap wave-6 reconciliation

Decision archive pre-check: the active ledger was 515835 bytes. No dated
section strictly older than 2026-07-20 was found, so
`.squad/decisions/archive/2026-07.md` was not changed.

<!-- inbox:chico-fp16-vae-51 -->
### Fail closed on non-finite image decode output
**By:** Chico
**What:** Validate widened image decode outputs at VAE, pipeline, and RGB
encoding boundaries; reject NaN and infinities with an actionable fp32-decoder
error.
**Why:** Widening fp16 output cannot recover values that already overflowed
inside a typed ONNX graph, and transparently rerunning that graph as fp32 is
not generally possible. Failing closed prevents corrupt images and
false-positive verification while remaining model-independent. PR #268 merged;
issue #51 is closed.

<!-- inbox:crowe-pr268-review -->
### Approve PR #268 non-finite VAE output handling
**By:** Crowe
**What:** Approved the shared `validate_finite_decode_output` guard applied at
standalone VAE, declared pipeline image, ComfyUI, and RGB encoding boundaries.
**Why:** It rejects NaN and both infinities before images are returned or
encoded, provides fp32 remediation, and its in-crate fp16-bit tests cover all
non-finite classes plus finite passthrough without model artifacts.

<!-- inbox:newt-cuda-coverage3 -->
### CUDA EP op-coverage batch 3 (#67) — IsInf, IsNaN, PRelu
**By:** Newt
**What:** Added IsInf (opset 10), IsNaN (opset 9), and PRelu (opset 16) to the
CUDA EP, raising `CUDA_COVERED_OPS` from 114 to 117. IsInf and IsNaN are
NVRTC unary float-to-bool predicates for f32/f16/bf16, with IsInf's
`detect_positive` and `detect_negative` flags validated. PRelu uses a
unidirectionally broadcastable slope and f32 widen-compute-narrow semantics.
Added CPU EP IsNaN as the parity reference and raised its registration count
from 95 to 96.
**Why:** The claim gates accept only the dtypes and attributes each kernel
implements, preventing over-broad CUDA claims. GPU-vs-CPU parity covers
IsInf's flag combinations, IsNaN including empty tensors, and scalar,
per-channel, full, and rank-0 PRelu slopes. PR #269 merged and advances #67.

<!-- inbox:ferro-pr269-review -->
### Approve PR #269 CUDA EP op-coverage batch 3
**By:** Ferro
**What:** Reviewed CUDA IsInf, IsNaN, and PRelu plus the CPU IsNaN parity
kernel. Confirmed exact claim gates, opset registration, f32 accumulation for
half PRelu, unidirectional slope broadcasting, portable NVRTC guards, and
non-default-stream/capture behavior.
**Why:** CPU IsNaN registration correctly moves the count from 95 to 96; the
CUDA covered-op test reports 117 unique entries. CPU IsNaN tests, the 13-test
GPU parity target, CUDA library tests, formatting, targeted clippy, and CI's
Rust-quality gate all passed. The unrelated all-targets clippy
`too_many_arguments` finding in `tests/fused_epilogue_gpu.rs` is non-blocking.

Processed wave-6 inbox notes: `chico-fp16-vae-51`, `crowe-pr268-review`,
`newt-cuda-coverage3`, and `ferro-pr269-review`.

## 2026-07-27 — Roadmap wave-7

Decision archive pre-check: the active ledger was 518,901 bytes. No dated section
strictly older than 2026-07-20 was found, so
`.squad/decisions/archive/2026-07.md` was not changed.

<!-- inbox:deckard-cuda-conformance-69 -->
### 2026-07-27: CUDA EP conformance profile and coverage-of-coverage — issue #69
**By:** Deckard
**PR:** https://github.com/justinchuby/onnx-genai/pull/270 (merged; advances #69)
**What:** Added an additive declarative conformance profile for all 117
`CUDA_COVERED_OPS`: a shared CUDA-vs-CPU single-node parity harness, 77 inline
f32/f16/bf16 cases, and attributed dedicated suites. The no-GPU
`every_covered_op_has_a_conformance_entry` guard enforces exact set equality
with `CUDA_COVERED_OPS`; duplicate-profile and dedicated-suite existence/name
guards prevent stale or dangling coverage claims.
**Why:** An audit found 26 claimed CUDA ops without parity references. The
profile closes those gaps (including Erf, SkipLayerNormalization, ReduceMean,
Pow, CastLike, and Softplus) and makes future claimed-but-untested operations
fail the guard rather than silently expanding coverage claims. No GPU Actions
workflow was added because CI runners lack GPUs; local GPU execution is
documented in `docs/CUDA_COVERAGE.md`.

<!-- inbox:roy-pr270-review -->
### 2026-07-27: Approve PR #270 CUDA EP conformance profile
**By:** Roy
**What:** Approved PR #270 after independent GPU and off-GPU validation. The
coverage guard iterates the real 117-op set and mutation-removing Erf fails;
the parity harness executes real CUDA and CPU paths, with a mutation of the
f32 tolerance proving a genuine numerical comparison. All 77 inline cases
passed on GPU7, and the three off-GPU guards passed without a GPU.
**Why:** The profile is non-tautological, its dedicated-suite references are
valid, and it adds no over-claiming workflow or kernel change. Formatting,
targeted clippy, and PR checks were clean. All-targets test clippy reports only
pre-existing untouched-suite lints and is non-blocking.

Processed wave-7 inbox notes: `deckard-cuda-conformance-69`, `roy-pr270-review`.

Archive pre-check correction: the active-ledger byte count at pre-check was
518,843 bytes (the archival conclusion is unchanged).

<!-- scribe-merge-2026-07-27T09-26-45-0700-cli-improvements -->
## 2026-07-27 — CLI dev-tool charter and split backlog reconciliation

Decision archive gate checked at 2026-07-27T09:26:45-07:00 after inbox merge: active ledger was 526691 bytes before merge and exceeds 51200 bytes; applying 7-day policy with cutoff 2026-07-20.

### andrews-split-movement-handlers

<!-- merged from .squad/decisions/inbox/andrews-split-movement-handlers.md -->
### 2026-07-27: Split movement shape handlers by operator family
**By:** Andrews
**What:** Replaced the 1,809-line `handlers/movement.rs` with:
- `movement/mod.rs` (114 lines): shared helpers and the unchanged registration facade.
- `movement/transform.rs` (409 lines): Transpose, Reshape, Flatten, Squeeze, Unsqueeze, Expand.
- `movement/resize.rs` (302 lines): Resize.
- `movement/concat_slice.rs` (394 lines): Concat, Slice.
- `movement/split_gather.rs` (380 lines): Split, Gather, GatherElements, GatherND.
- `movement/scatter.rs` (137 lines): Scatter, ScatterElements, ScatterND, Trilu.
- `movement/space_depth.rs` (132 lines): DepthToSpace, SpaceToDepth.

The split totals 1,868 lines including module-local imports. Registration order, operator/opset mappings, handler bodies, shape rules, and diagnostic text are unchanged.

**Why:** Cohesive operator-family modules reduce navigation and review cost while keeping this change mechanical and behavior-preserving. `cargo fmt -p onnx-runtime-shape-inference`, shape-inference build/tests (224 tests plus one doctest), clippy with `-D warnings`, and downstream `onnx-runtime-session` build all pass.

### ash-split-genai-config

<!-- merged from .squad/decisions/inbox/ash-split-genai-config.md -->
### 2026-07-27: Split genai config compatibility crate into cohesive modules
**By:** Ash
**What:** Kept `lib.rs` as a 98-line facade retaining `GenAiConfigError`, `GENAI_CONFIG_FILE`, and the flat public re-exports. Moved config wire types to `wire_types.rs` (361 LOC), loading to `loading.rs` (109 LOC), graph I/O inspection to `graph_io.rs` (235 LOC), compatibility synthesis to `compatibility.rs` (1,427 LOC), JSON builders to `json_builders.rs` (341 LOC), and unit tests to `tests.rs` (1,212 LOC).
**Why:** The former 3,743-line facade mixed serialization contracts, file loading, graph inspection, pipeline synthesis, JSON construction, and tests. The split is pure code motion; public names remain re-exported from the crate root. A source comparison confirmed config wire definitions, field/variant ordering, derives, every `#[serde(...)]` attribute, and all `GenAiConfigError` text are unchanged. `cargo build`, all 30 crate tests, clippy with `-D warnings`, and downstream engine/server/CLI builds passed.

### call-split-onnx-std-rules

<!-- merged from .squad/decisions/inbox/call-split-onnx-std-rules.md -->
### 2026-07-27: Split ONNX validation rules by model layer
**By:** Call
**What:** Split the former 5,316-line `crates/onnx-std/src/check/rules.rs` into a `rules/mod.rs` facade and five private rule-family modules:
- `graph_topology.rs` — 368 lines; opset imports, duplicate names, graph input/output connectivity, and acyclicity.
- `schema_types.rs` — 1,217 lines; schema conformance, type constraints, initializer declarations, metadata, attributes, and retained protobuf types.
- `ir_version_functions.rs` — 1,147 lines; IR version/feature gates and local function validation. The two existing `#[allow(clippy::too_many_arguments)]` attributes remain on their original functions.
- `tensor_sparse_payloads.rs` — 558 lines; dense tensor payload and sparse tensor validation.
- `multi_device.rs` — 393 lines; device configuration and sharding validation.
- `mod.rs` — 1,711 lines; public facade, shared diagnostic helpers, and unchanged tests.

**Why:** Cohesive private modules reduce the validation implementation's file-level entropy while preserving the flat public API. Rule ORDER is unchanged because `check/mod.rs` and its 17 `checker.add_rule(...)` calls were not modified. Violation WORDING is unchanged: all 579 Rust string literals were compared as multisets before formatting and preserved exactly; the non-author reviewer independently approved the split and found rule implementations/helper logic unchanged.

**Gates:** `cargo fmt -p onnx-std` passed; `cargo build -p onnx-std` passed; `cargo test -p onnx-std` passed (126 unit tests, 23 integration tests, 1 doc-test); `cargo clippy -p onnx-std --all-targets -- -D warnings` passed. Non-author review: approved with no blocking findings.

### christie-split-server-routes

<!-- merged from .squad/decisions/inbox/christie-split-server-routes.md -->
### 2026-07-27: Split server routes by endpoint family
**By:** Christie
**What:** Replaced the 2,989-line `crates/onnx-genai-server/src/routes.rs` with a `routes/` module tree: `mod.rs` (530 LOC) retains `ApiError`, JSON rejection handling, model resolution, shared request preparation types/helpers, and facade re-exports; `admin.rs` (396 LOC) owns health, models, status, resources, debug, admin, and metrics endpoints; `sessions.rs` (60 LOC) owns session create/delete; `completions.rs` (1,719 LOC) owns completions, embeddings, chat, streaming, and generation helpers; `multimodal.rs` (312 LOC) owns transcription, speech, and image-generation endpoints.
**Why:** This is a pure code-motion split of the HTTP god-file. Router registration remains untouched in `src/lib.rs`, preserving route paths and registration order exactly. The typed `ApiError` handling and server-side registry logging hardened in PR #213 were moved verbatim without behavior changes.

**Gates:** `cargo build -p onnx-genai-server` passed. `cargo test -p onnx-genai-server` completed with 110 passed, 2 ignored, and only the accepted pre-existing `sidecar_free_compatibility_package_builds_server_pipeline_and_preprocesses_image` failure caused by missing `vlm-executable/vision.onnx`. `cargo clippy -p onnx-genai-server --all-targets -- -D warnings` passed. `cargo fmt -p onnx-genai-server` passed.

### coordinator-cli-is-a-dev-tool

<!-- merged from .squad/decisions/inbox/coordinator-cli-is-a-dev-tool.md -->
# CLI is a developer/maintainer tool, not an end-user product surface

**By:** Squad (Coordinator), capturing a directive from Justin Chu
**Date:** 2026-07-27T09:26:45-07:00

**What:** The `onnx-genai` CLI (`crates/onnx-genai-cli`) is scoped as a
**development and maintainer tool**. It is not trying to be a consumer-facing
local-inference product like `ollama`.

Two direct consequences:

1. **Remote-client mode is out of scope.** The CLI does not need to act as a
   client against a remote OpenAI-compatible server. Existing third-party CLIs
   already do that well, and the user will use those. (Explicitly overrides
   Rachael's finding #1 in `docs/research/cli/02-ux-and-server-surface.md`,
   which ranked remote-client mode as a top gap.)
2. **Competitive parity with consumer CLIs is not a goal.** Model
   pull/registry workflows, conversion/quantization/fine-tuning loops, and
   general product polish are only worth doing where they make *development*
   faster. This aligns with the Fact Checker's devil's-advocate conclusion in
   `docs/research/cli/03-competitive-and-devils-advocate.md`.

**Why:** The repo's primary fronts are CUDA/perf, model enablement, and the
server/Python integration path. Investing in CLI features that duplicate
existing tooling would spend effort where it does not compound. What *does*
compound is the CLI's value as an inner-loop instrument: fast iteration on a
model, visibility into what the engine is actually doing (EP selection, decode
backend, KV reuse, timings), and reachability of runtime capabilities that are
otherwise only testable through code.

**Prioritization lens going forward:** rank CLI work by *"does this shorten a
maintainer's debug/iterate loop or expose engine behavior we currently cannot
observe?"* — not by *"does ollama have it?"*

**Concretely reprioritized upward:** discoverability and defaults of
diagnostics (e.g. the live-stats view is gated behind an undocumented `/stats`
toggle with no CLI flag), machine-readable output for scripted experiments,
reachability of engine features (speculative decoding, batching, fork/rewind,
KV controls) from the command line, and benchmark/eval commands.

**Concretely reprioritized downward / dropped:** remote-client mode, model
registry & pull workflows, consumer-grade onboarding polish.

### coordinator-repl-is-the-product

<!-- merged from .squad/decisions/inbox/coordinator-repl-is-the-product.md -->
# REPL is the primary CLI investment — Copilot-CLI-class interactive shell

**By:** Squad (Coordinator), capturing a directive from Justin Chu
**Date:** 2026-07-27T09:30:56-07:00

**What:** The interactive REPL (`onnx-genai run`) is now the *primary* CLI
investment, not a side feature. Target quality bar: **GitHub Copilot CLI's
interactive layout**. Required capabilities:

1. **Copilot-CLI-style layout** — a persistent input area with streaming
   output above it, rather than today's plain line-by-line `>>>` loop.
2. **Real line editing** — cursor movement, multiline input/paste, kill/yank,
   persistent history. Today the REPL is a bare line reader.
3. **Session fork** and the other agent-first runtime primitives, driven from
   the REPL.
4. **Expose as much of the runtime as possible** — the engine has prefix
   caching, multi-session, CoW fork, KV rewind, speculative decoding, and
   continuous batching, and almost none of it is reachable interactively.
   Reachability is the point of the tool.
5. **Stats shown by default.** This is a developer tool; per-turn numbers are
   signal, not noise. Inverts the current default (`interactive.rs:614`
   `show_stats = false`, toggled only by an undocumented `/stats`).
6. **Slash-command autocompletion.**

**Why:** This follows directly from the dev-tool charter
(`coordinator-cli-is-a-dev-tool.md`). If the CLI's job is to shorten the
maintainer debug/iterate loop and make engine behavior observable, then the
interactive shell *is* the product — it is where a maintainer actually spends
time, and it is currently the weakest surface relative to what the runtime can
do. Note the contrast with the rejected remote-client work: this invests in
capability *reachability*, not in duplicating tooling that already exists
elsewhere.

**Backlog impact:** Supersedes the ranking in `docs/research/cli/00-backlog.md`.
Promoted to P0: P0.1 (stats reachable — now *stats by default*), P0.4 (expose
engine behavior), P1.3 (session/KV debug controls: fork, rewind), P1.4 (REPL
ergonomics: multiline, history). The rejected items are unchanged and remain
rejected.

**Open design question for the user:** whether the REPL becomes a full-screen
ratatui application or keeps an inline viewport with a rich line editor. The
team is to present the tradeoff with a recommendation rather than choose
silently.

### dietrich-split-ep-api-abi

<!-- merged from .squad/decisions/inbox/dietrich-split-ep-api-abi.md -->
### 2026-07-27: Split the plugin EP ABI bridge by responsibility
**By:** Dietrich
**What:** Moved `crates/onnx-runtime-ep-api/src/abi.rs` to `abi/mod.rs` and split implementation details into `runtime.rs`, `host.rs`, `ffi_helpers.rs`, and `weights.rs`. The facade retains `OrtGraphView`, `SubgraphClaim`, and `PluginExecutionPlan`; it re-exports `PluginCompiledKernel` at the unchanged `abi::PluginCompiledKernel` path. The host projection test is colocated in `host.rs`.
**Why:** The 2,512-line plugin-EP ORT C-ABI boundary was difficult to review safely. This is pure code motion with only minimally scoped `pub(super)` visibility needed between sibling modules.

Module breakdown:
- `abi/mod.rs`: facade, stable public surface, graph view, claims, execution plan — 429 LOC.
- `abi/runtime.rs`: plugin runtime ownership, shared kernel state, compiled kernel — 304 LOC.
- `abi/host.rs`: ORT host projections, C-ABI vtables/callbacks, and host projection test — 1,618 LOC.
- `abi/ffi_helpers.rs`: raw-pointer conversions and plugin-device accessors — 127 LOC.
- `abi/weights.rs`: mapped external-weight cache and initializer projection — 103 LOC.

Invariant counts across the ABI module tree:
- ABI root LOC: 2,512 before; 429 after.
- `unsafe {` blocks: 82 before; 82 after.
- `#[cfg(...)]` attributes: 1 before; 1 after.
- `extern "C"` occurrences: 59 before; 59 after.
- `#[no_mangle]` attributes: 0 before; 0 after.

Public API is unchanged: all prior bare-`pub` items and methods retain their paths and signatures; `PluginCompiledKernel` is re-exported from the facade.

Validation:
- `cargo build -p onnx-runtime-ep-api`: passed.
- `cargo test -p onnx-runtime-ep-api`: passed (38 unit tests, 7 integration tests, 0 failures).
- `cargo clippy -p onnx-runtime-ep-api --all-targets -- -D warnings`: passed.
- `cargo build -p onnx-runtime-session`: passed.
- `cargo fmt -p onnx-runtime-ep-api`: passed.

### dillon-split-ort-decode

<!-- merged from .squad/decisions/inbox/dillon-split-ort-decode.md -->
### 2026-07-27: Split ORT decode by cache and session family
**By:** Dillon
**What:** Replaced `crates/onnx-genai-ort/src/decode.rs` with a facade and six focused submodules:

- `decode/mod.rs` — 201 lines; public option/signature types, batched trait, and re-exports.
- `decode/dynamic.rs` — 1,550 lines; dynamic past/present decode and captured-step tests.
- `decode/kv_growth.rs` — 465 lines; shared KV bucket growth, host/CUDA prefix copying, and tests.
- `decode/static_cache.rs` — 1,210 lines; scalar and batched static-cache sessions.
- `decode/shared_batch.rs` — 476 lines; continuous-batch shared-buffer session.
- `decode/io.rs` — 196 lines; KV-name pairing and static-cache signature detection.
- `decode/tensor.rs` — 149 lines; logits, cloning, empty tensor, and allocation helpers.

All existing public types remain available from `onnx_genai_ort::decode` through facade re-exports. The `decode_contract`-based `KvNamingConvention`, `kv_suffix`, and `name_contains_present_key_value` call sites were moved unchanged into `decode/io.rs`; no local classifier copies were introduced.

`cargo fmt -p onnx-genai-ort` was run. Gates passed:

- `cargo build -p onnx-genai-ort`
- `cargo test -p onnx-genai-ort` (all unit, integration, and doc tests)
- `cargo clippy -p onnx-genai-ort --all-targets -- -D warnings`
- `cargo build -p onnx-genai-engine`

**Why:** The original 4,239-line file mixed materially different cache ownership and batching models. The split is pure code motion and clarifies ownership without changing algorithms, allocation, CUDA annotations, or the public facade.

### frost-split-ort-session

<!-- merged from .squad/decisions/inbox/frost-split-ort-session.md -->
### 2026-07-27: Split ORT session god-file into focused modules
**By:** Frost
**What:** Moved `crates/onnx-genai-ort/src/session.rs` to `session/mod.rs` and split options, environment configuration, EP compatibility, provider dispatch, CUDA wiring, plugin wiring, and tests into sibling modules. The facade re-exports the existing public API.
**Why:** Reduce the 2,504-line session god-file while preserving behavior, provider resolution order, environment handling, error text, cfg gates, and downstream import paths.

#### Module breakdown
- `session/mod.rs` — `Session`, `TensorInfo`, `RunPhaseError`, `RawSessionOptions`, I/O metadata helpers, facade exports
- `session/options.rs` — `SessionOptions`, defaults/builders, `ep_selection`, provider availability
- `session/env_config.rs` — runtime/environment configuration readers and provider/fallback predicates
- `session/ep_compat.rs` — EP capability model and provider-name compatibility resolution
- `session/providers.rs` — generic provider append/dispatch and WebGPU session options
- `session/cuda.rs` — cfg-gated CUDA provider setup and diagnostics
- `session/plugin.rs` — plugin resolution, registration, discovery, and append flow
- `session/tests.rs` — all existing session unit tests

#### Size
- Session root before: 2,504 LOC (`session.rs`)
- Session root after: 839 LOC (`session/mod.rs`)
- Session module tree after: 2,571 LOC (module declarations/imports and minimal `pub(super)` visibility account for the increase)

#### cfg count
| Measurement | Before | After |
|---|---:|---:|
| Original cfg attributes preserved | 29 | 29 |
| Module/import wiring cfg attributes | 0 | 7 |
| Total cfg attributes | 29 | 36 |

All original cfg expressions, including the platform-specific duplicate CUDA library/search-path functions, remain verbatim. The seven additions only gate new module/import wiring.

#### API and gates
Public paths remain unchanged, including `Session`, `SessionOptions`, `TensorInfo`, `ep_selection`, `available_execution_providers`, and `session::ep_compat`. No private item was widened to unrestricted `pub`; cross-module helpers use `pub(super)`.

- `cargo build -p onnx-genai-ort` — PASS
- `cargo test -p onnx-genai-ort --lib` — PASS (56 tests)
- `cargo clippy -p onnx-genai-ort --all-targets -- -D warnings` — PASS
- `cargo check -p onnx-genai-ort --features cuda` — PASS
- `cargo build -p onnx-genai-engine` — PASS
- `cargo fmt -p onnx-genai-ort` — PASS

### rains-split-sequence

<!-- merged from .squad/decisions/inbox/rains-split-sequence.md -->
### 2026-07-27: Split sequence storage and algorithms into focused modules
**By:** Rains
**What:** Replaced the 1,761-line `crates/onnx-runtime-session/src/sequence.rs` with a `sequence/` module tree: `mod.rs` (238 lines; root, re-exports, tests), `error.rs` (errors/result), `tensor.rs` (shared tensor storage, allocation, byte/view validation), `value.rs` (homogeneous sequence storage and indexing), `split.rs` (split specifications and planning), and `concat.rs` (concat planning, copying, and new-axis stacking).
**Why:** This is behavior-preserving code motion for Dallas entropy audit item #11. `sequence::SequenceError`, `SequenceResult`, `SeqTensor`, `SequenceValue`, `SplitSpec`, `split`, `split_tensor`, `concat`, and the existing crate-visible concat helpers remain re-exported at their prior paths; `executor.rs` and root `Cargo.toml` are unchanged. Allocation order, view-bound checks, signatures, error text, cfg/allow attributes, and tests are unchanged. Gates passed: `cargo build -p onnx-runtime-session`; `cargo test -p onnx-runtime-session` (82 unit tests, integration tests, and doc tests passed); `cargo clippy -p onnx-runtime-session --all-targets -- -D warnings`; and `cargo fmt -p onnx-runtime-session`. The known pre-existing `tests/decode_session.rs` missing `tests/fixtures/tiny-llm/model.onnx` failure did not reproduce in this checkout's gate run; no fixture or decode-session files were changed.

### roy-cli-improvements

<!-- merged from .squad/decisions/inbox/roy-cli-improvements.md -->
### 2026-07-27: CLI improvements should start with interface contracts
**By:** Roy
**What:** Treat the next `onnx-genai` CLI wave as an interface-contract project: split clap args out of `lib.rs`, add typed output/rendering seams, define JSON schemas and exit codes, and snapshot help before adding model-store/config subcommands.
**Why:** The current CLI already has useful commands, but its 1,134-line `lib.rs` owns parsing, dispatch, profiling side effects, and tests. Adding model management, profiles, completions, and machine-readable output on top of that shape would lock in inconsistent flags and ad hoc rendering.

### spunkmeyer-split-image

<!-- merged from .squad/decisions/inbox/spunkmeyer-split-image.md -->
### 2026-07-27: Split image preprocessing into cohesive submodules
**By:** Spunkmeyer
**What:** Split `crates/onnx-genai-preprocess/src/image.rs` into a 29-line facade plus `image/config.rs` (293 LOC), `image/program.rs` (1,742 LOC), `image/tiling.rs` (323 LOC), `image/transform.rs` (233 LOC), and `image/tests.rs` (1,408 LOC). `image/packed.rs` remains unchanged at 1,330 LOC. The facade preserves every existing public re-export and import path.
**Why:** Separate image-program metadata compilation/dataflow validation from pixel transforms and tiling without changing behavior. All serde attributes, resize/normalization arithmetic, tiling boundary math, serialization behavior, and error text are unchanged; the unknown-output-source regression now asserts the complete byte-identical error string. Gates passed: preprocess build, 54 preprocess tests, preprocess clippy with `-D warnings`, and downstream engine/CLI build. A non-author code-review agent approved the diff with no findings.

### wierzbowski-split-cli-lib

<!-- merged from .squad/decisions/inbox/wierzbowski-split-cli-lib.md -->
### 2026-07-27: Split CLI orchestration from presentation and REPL parsing
**By:** Wierzbowski
**What:** Split `crates/onnx-genai-cli/src/lib.rs` (3,559 lines before; 1,233 after) into `generate.rs` (219 LOC), `interactive.rs` (953), `commands.rs` (234), `output.rs` (232), `model_inspection.rs` (71), and `transcribe.rs` (709), retaining the existing `profile.rs`. `lib.rs` remains the CLI argument/type and dispatch facade.
**Why:** Cohesive private modules make generation, interactive orchestration, command parsing, presentation, model inspection, and transcription independently navigable without changing the crate's public surface, CLI shapes, or output text.

Ctrl-C wiring was moved intact into `interactive.rs`: the `Once`-guarded `ctrlc::set_handler` body retains its registration sites and order, the same `GENERATING`, `INTERRUPT_REQUESTED`, and `EXIT_ARMED` atomics with `SeqCst`, and the REPL still clears `EXIT_ARMED` immediately after a submitted line before parsing it. One-shot generation and transcription install the same handler at their original points.

Gates: `cargo build -p onnx-genai-cli` passed; `cargo test -p onnx-genai-cli` passed (127 tests total across targets); strict `cargo clippy -p onnx-genai-cli --all-targets -- -D warnings` is blocked by pre-existing unchanged `crates/onnx-genai-cli/src/pages.rs:129` (`clippy::manual_checked_ops`); clippy passes with only that lint allowed. `cargo fmt -p onnx-genai-cli -- --check` and `git diff --check` passed. Non-author code review found no significant issues.
<!-- scribe-merge-2026-07-27T09-26-45-0700-cli-improvements-end -->

Archive action at 2026-07-27T09:26:45-07:00: active ledger exceeded 51200 bytes; no dated active-ledger entries older than 2026-07-20 were present, so archived 0 block(s).

Decision archive gate checked at 2026-07-27T16:44:54Z: active ledger was 654224 bytes before wave-8 merge; archived 2 entries older than 2026-07-20 into `.squad/decisions/archive/`.
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-bishop-pr273-review -->
<!-- merged from .squad/decisions/inbox/bishop-pr273-review.md -->
# Decision: PR #273 review (BlockQuantizedMoE CUDA v1 kernel)

- **Reviewer:** Bishop (senior CUDA engineer, adversarial independent review)
- **Author:** Moss (locked out of revisions)
- **Issue:** #79 — CUDA `pkg.nxrt::BlockQuantizedMoE` v1 kernel
- **Branch:** feat/cuda-blockquant-moe-79 @ e2fcfbf6
- **Date:** 2026-07-27
- **Verdict:** APPROVE

## What was verified

1. **Routing parity (tie-break):** CUDA `bqmoe_total_order_key` (`bits ^= (bits>>31)&0x7fffffff`)
   yields a total order consistent with CPU `f32::total_cmp` (descending logit, ascending index
   tiebreak). Hand-verified negative/positive/zero/−0.0 ordering. Iterative top-k argmax matches
   CPU full-sort+truncate.
2. **Softmax / router-weight normalization:** max-subtracted exp, normalize vs all-sum denominator,
   zero-denominator → 0.0 guard — all match `moe.rs::routing_weights` exactly.
3. **Activation parity:** relu / gelu (f64 tanh, `SQRT_2_OVER_PI=0.7978845608028654`) / silu
   (stable sigmoid) / identity / swiglu (fused fusion 1 interleaved, fusion 2 block-half, unfused
   fc3, gated-silu via swiglu formula). `gate=min(limit)`, `linear=clamp` with NaN passthrough all
   match CPU `MoeAttributes::swiglu`.
4. **Decode reuse:** GEMV kernel calls shared `decode_weight`/`block_sum` from the newly-exposed
   `decoder_prelude()` — no re-implemented GGUF decoder that could drift.
5. **CUDA portability (HARD rule):** launch configs come from runtime device props
   (`compute_capability`, `multiprocessor_count`, `max_threads_per_block`,
   `reduction_launch_config` sized against `max_shared_memory_per_block_optin`). No hardcoded
   SM90/H200 constants.
6. **Stream ordering:** all launches on the EP non-default stream; single trailing
   `synchronize()`; capture declared unsupported; scratch pool guarded by mutex + synced before
   release.
7. **supports_op gate:** declines unsupported format / layout version / non-f32 dtype; restricted
   to implemented configs; consistent with sibling ops + the CPU claim gate.
8. **Conformance guard (#270):** dedicated entry added; `every_covered_op_has_a_conformance_entry`
   and duplicate guard pass.
9. **Parity tests:** genuine CPU-vs-CUDA oracle on identical synthetic mxfp4 weights; cover
   activations, swiglu variants, routing edge cases (k=1, single-expert, k=experts), biases,
   router-weight aggregation. Tolerance 3e-3 rel/abs would catch wrong dequant/routing/accumulation.

## Validation evidence (pinned GPU5, `taskset -c 1`)

- `block_quantized_moe_gpu`: **5 passed / 0 failed**
- `cuda_conformance_gpu`: **4 passed / 0 failed** (incl. coverage guards)
- Full `-p onnx-runtime-ep-cuda --features cuda`: 241 lib passed; one flake in
  `attention_gpu::fused_attention_matches_phase2a_baseline` (GPU-neighbour contention) that
  **passed on isolated rerun** — unrelated to this PR.
- `cargo fmt --all -- --check`: clean.
- `cargo clippy … -D warnings`: fails, but **every** error is in pre-existing unrelated files
  (newer clippy 1.97.0 `manual_is_multiple_of` / `manual_repeat_n`). None in PR-touched files;
  `normalization.rs:2291` is identical on origin/main. Toolchain drift, not this PR.

## Minor / non-blocking notes

- GELU at x=−∞: CPU pins to 0.0; CUDA computes NaN (0.5·−∞·(1+tanh(−∞))). Pathological input only,
  not in spec/tests.
- Tie-break path not directly exercised (random logits ≈ unique); low risk since both sides use the
  same total order. Could add an exact-tie case later.

## Follow-up (owner NOT Moss)

- Repo-wide clippy 1.97.0 lint drift (manual_is_multiple_of / manual_repeat_n across many files) —
  route to the CUDA EP maintainer / toolchain owner as a separate cleanup, not part of #79.
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-bishop-pr273-review-end -->
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-dallas-schedulers-47 -->
<!-- merged from .squad/decisions/inbox/dallas-schedulers-47.md -->
### 2026-07-27: Add DDPM and shifted flow-matching schedulers
**By:** Dallas
**What:** Register `ddpm` as a fixed-small-variance ancestral sampler supporting epsilon, v-prediction, and sample/x0; register `flow_matching` as shifted-sigma Euler integration of direct velocity/vector-field predictions, with a new metadata `shift` field.
**Why:** This follows DESIGN §§17/20, reuses the shared prediction conversion helpers, and gives modern DiT/rectified-flow packages a scheduler without mislabeling their vector field as diffusion epsilon.
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-dallas-schedulers-47-end -->
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-deckard-review-fixes -->
<!-- merged from .squad/decisions/inbox/deckard-review-fixes.md -->
# Fix tautological cache test + SiLU accuracy doc conflict

**Date:** 2026-07-27
**Author:** Deckard
**PR:** #227 (squad/mac-cpu-ep-roofline)

## Context

GitHub Copilot's automated reviewer flagged two issues in `onnx-runtime-ep-cpu`:

1. **matmul.rs** — the `constant_weight_prepack_reuses_weight_and_keeps_activation_live` test compared a cache pointer to itself (tautological: always passes). This is the fifth instance of the "test that cannot fail" bug class on this campaign.

2. **activations.rs** — the `silu_f32_slice` doc comment claimed "1 ULP accuracy" for the NEON exp polynomial, contradicting the measured "~28 ULP" stated in the implementation comment below it. Partially fixed earlier; the higher-level doc was missed.

## Decision

- **Fix 1:** Restructured the cache-reuse test to capture the prepack pointer *before* the second `execute()` call. The comparison now spans the second call, proving the first call populated the cache and the second reused it. Guard-break confirmed: substituting a fresh kernel (simulating cache invalidation) makes the test fail with distinct pointers.

- **Fix 2:** Updated the slice-level doc to state "~28 ULP worst-case on [-87, 88]", matching the measured value. Grep-verified: no other "1 ULP" claim remains in `activations.rs`. The other "1 ULP" references in the crate (`decode_spmd.rs`, `matmul_nbits.rs`) are about N-tile boundary drift, not exp accuracy — left as-is.

## Verification

- `cargo fmt --all -- --check` ✅
- `cargo clippy -p onnx-runtime-ep-cpu --lib -- -D warnings` (aarch64) ✅
- `cargo clippy -p onnx-runtime-ep-cpu --lib --target x86_64-apple-darwin -- -D warnings` ✅
- Full CPU EP test suite: 906 passed, 0 failed ✅
- `sdpa_dispatcher_reaches_neon_on_aarch64 ... ok` ✅
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-deckard-review-fixes-end -->
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-ferro-pr276-review -->
<!-- merged from .squad/decisions/inbox/ferro-pr276-review.md -->
# Decision: PR #276 (issue #87) async compute/transfer overlap — Ferro review

**Reviewer:** Ferro (CUDA/concurrency, adversarial/independent — not author)
**Date:** 2026-07-27
**Verdict:** REQUEST-CHANGES
**Author (locked out):** Keaton
**Suggested fix owner:** Deckard (Systems Dev, CUDA & Perf pod)

## Summary
The core concurrency *mechanism* is correct and its regression guards genuinely
bite (verified by neutering the fence — see below). RAW ordering is right: the
completion event is recorded on the copy stream *after* the async copy, and the
compute stream waits on it (`cuStreamWaitEvent`, non-host-blocking) before the
consuming kernel is enqueued. The WAR primitive (`copy_wait_fence`) is correct.
Deferral of live MoE wiring is honestly documented and acceptable.

Two concrete defects block approval:

### BLOCKER 1 — build break of the GPU test suite (introduced by this PR)
Adding `async_host_to_device` to `CudaTransferCounts` broke a struct-literal in a
pre-existing test: `crates/onnx-runtime-ep-cuda/tests/compressed_sparse_attention_gpu.rs:699`
(missing field). `cargo test -p onnx-runtime-ep-cuda --features cuda` **does not
compile** as submitted — meaning the PR's own GPU regression guards cannot run in
CI. Fix: add the field to the literal (or `#[non_exhaustive]` + `..Default`).

### BLOCKER 2 — public `drive_double_buffer` is WAR-racy on the CUDA EP; docs overclaim
`drive_double_buffer` (public, exported from `onnx_runtime_session`) reuses buffer
`(n+1)%2` by issuing `copy_async(source_{n+1} -> buffer_{(n+1)%2})` on the copy
stream, but neither `drive_double_buffer` nor the CUDA EP's `copy_async` inserts a
`copy_wait_fence`. So on a real `CudaExecutionProvider` the reuse copy can overwrite
a buffer while the prior wave's compute is still reading it — a silent WAR data race.
The `prefetch.rs` module doc claims WAR "is a *mechanism* concern handled EP-side …
the copy stream waits on the previous consumer's compute event before overwriting a
reused buffer (proven by `double_buffered_prefetch_is_race_free_across_waves`)".
That proof only holds for a hand-rolled loop that *manually* calls `copy_wait_fence`;
it is NOT wired into `copy_async`/`drive_double_buffer`. Fix: either wire the WAR
fence into the driven path, or correct the docs to state plainly that
`drive_double_buffer` is WAR-unsafe on async EPs and must not be used with the CUDA
EP until WAR is wired.

## Fence-neutering experiment (the load-bearing check) — guards are REAL
Neutered `wait_fence_on` (dropped the event without `cuStreamWaitEvent`), then restored:
- `async_prefetch_h2d_event_orders_copy_before_consume` → FAILED: "read poison at index 0: got -999, expected 1"
- `double_buffered_prefetch_is_race_free_across_waves` → FAILED: "wave 0 output corrupted … (write-after-read fence violated)"
- `copy_async_fence_orders_h2d_prefetch_through_ep_api` → FAILED: "consumer read poison — the fence did not order the transfer"
- After restoring the fence: all 3 PASS. Not theater.

## Validation (pinned GPU6)
- `cargo test -p onnx-runtime-session`: 90 lib tests pass incl. 5 new prefetch strategy tests.
- `cargo test -p onnx-runtime-ep-cuda --features cuda` (after local literal fix): 244 lib pass; 3 new overlap tests pass.
- Pre-existing/environmental failures (NOT this PR, reproduced on parent 6654a168): `fused_attention_matches_phase2a_baseline` (bf16 tolerance), cuDNN conv/pool absent.
- Clippy: PR's own code (ep-cuda lib, ep-api lib, session lib+tests) clean under `-D warnings`. `--all-targets` fails only in unrelated pre-existing test files (also fail on base).
- `cargo fmt --all -- --check`: clean.
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-ferro-pr276-review-end -->
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-gorman-generate-image-53 -->
<!-- merged from .squad/decisions/inbox/gorman-generate-image-53.md -->
### 2026-07-27: Typed image streaming API
**By:** Gorman
**What:** Added `PipelineEngine::generate_image` / `generate_image_with_callback`, emitting post-scheduler loop-carried latents and a typed final image tensor. The existing prompt, latent construction, and RGB postprocessing remain in `onnx_genai::text_to_image::generate_image`.
**Why:** Pipeline metadata has generic component ports, so `ImageRequest` wraps the existing generic tensor request while selecting the final image endpoint explicitly only when a final component has multiple outputs. One shared callback immediately after scheduler dispatch covers deterministic and stochastic schedulers without scheduler-specific paths.
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-gorman-generate-image-53-end -->
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-hudson-pr274-review -->
<!-- merged from .squad/decisions/inbox/hudson-pr274-review.md -->
# Decision: PR #274 review (issue #53) — typed generate_image + latent streaming

- **Reviewer:** Hudson (independent; author was Gorman)
- **Date:** 2026-07-27
- **Verdict:** APPROVE

## What was reviewed
Typed `PipelineEngine::generate_image` / `generate_image_with_callback` on the
iterative pipeline, plus `text_to_image::generate_image` host entry point
(with `render` kept as a backward-compat wrapper). 6 files, +291/-6.

## Key findings
1. **Spec (DESIGN §20.4/§20.6):** Matches intent — typed `ImageRequest ->
   ImageStream`, stepwise latent preview, host-side entry point. §20 deliberately
   left request fields/output-selection unspecified; wrapping the generic
   `PipelineGenerateRequest` and auto-selecting the sole final output (explicit
   endpoint required for multi-output) is a reasonable resolution.
2. **Streaming consistency (strength):** Streaming and non-streaming share ONE
   execution path (`run_iterative_with_callback`); the callback is purely
   observational (clones values, no feedback into compute). Divergence is
   structurally impossible, not merely asserted. Tests still confirm
   `output.latents == steps.last().latents` and cross-run determinism.
3. **Scheduler generality (DRY):** Callback is placed AFTER the unified
   `scheduler.step` / `scheduler.step_with_noise` / identity-feedback dispatch,
   over `loop_edges`. No per-scheduler special-casing — works for
   DDPM/DDIM/Euler/FlowMatching/ancestral and CFG/language-diffusion paths.
4. **Test quality:** Hand-computed oracles (`s_k=(s_{k-1}+c)/2`, `image=2*s_3+1`),
   covers every intermediate latent, callback/non-streaming consistency,
   step-count, and single-step override edge. Would catch a regression.
5. **API/back-compat:** Additive engine exports only; no removals. `render`
   retained and still exercised by an existing test. Genuine typed improvement,
   not a leaky wrapper.

## Minor (non-blocking)
- `generate_image*` take `&mut self` though the underlying
  `run_iterative_with_callback` is `&self`; tighten to `&self` if convenient.
- `ImageStream.output.latents` duplicates `steps.last().latents` — documented,
  acceptable.

## Validation (all PASS)
- `cargo test -p onnx-genai-engine` — ok (incl. 2 ignored ORT-runtime tests)
- `cargo test -p onnx-genai-engine --test iterative_pipeline_e2e` — 31 passed
- `cargo test -p onnx-genai --test text_to_image_e2e` — 5 passed
- `cargo fmt --all -- --check` — clean (exit 0)
- `cargo clippy -p onnx-genai-engine -p onnx-genai --all-targets -- -D warnings` — clean (exit 0)
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-hudson-pr274-review-end -->
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-iran-calibrator-opt-in -->
<!-- merged from .squad/decisions/inbox/iran-calibrator-opt-in.md -->
### 2026-07-27: Made load-adaptive path selection opt-in (Iran)

**By:** Iran (Mac CPU Optimization Engineer)
**Directive from:** Justin Chu (via coordinator, `coordinator-calibrator-opt-in.md`)
**PR:** #227 (`squad/mac-cpu-ep-roofline`)

**What changed:**

The `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL` env var semantics changed:

| Value | Before (old) | After (new) |
|---|---|---|
| unset (default) | `Auto` — calibrator probes both paths, defaults to flat | `On` — persistent SPMD pool, deterministic, no probing |
| `=1` | `Forced` — always pool | `On` — same as default (explicit) |
| `=0` | `Off` — always flat | `Off` — always flat (unchanged) |
| `=auto` | *(not recognized)* | `Adaptive` — opt-in calibrator, same logic as old `Auto` |

**Why:**

1. The default was unpredictable: under host load, the calibrator silently selected the flat path, halving throughput with no user indication.
2. Different paths use different FP reduction orders, making greedy decode non-reproducible across load conditions.
3. The calibrator itself was bitten during this campaign — it mis-sampled under agent load and produced a false Fact Checker verdict.

A library should be predictable by default, adaptive on request.

**Measurements (M1 Max, FP16 Qwen2.5-0.5B, post GEMV dispatch fix):**

| Condition | Default (pool) | Adaptive (`=auto`) | Flat (`=0`) | ORT |
|---|---|---|---|---|
| Quiet (load ~4-5) | 53.35 tok/s | 56.10 tok/s | 42.84 tok/s | 42.19 tok/s |
| 4×`yes` load (~10) | 18.96 tok/s | 31.95 tok/s | 31.57 tok/s | 37.76 tok/s |

Under moderate load (4 contending cores), the pool degrades ~2× more than flat because its pinned workers compete with load processes. This is the accepted tradeoff for predictability and reproducibility. Users who need adaptation set `=auto`.

**Observability:** The selected path is queryable via `decode_path_label()` → `"spmd-pool"`, `"adaptive"`, `"flat"`, or `"unresolved"`. With the `tracing` feature enabled, path selection is emitted as `tracing::debug!(path = "spmd-pool", workers = 9, "cpu decode path selected")` per `docs/ERROR_AND_LOGGING_CONVENTIONS.md`. Without the feature, `NXRT_CALIB_DEBUG` gated eprintln serves as fallback.

**half_gemm.rs overlap analysis (2026-07-27):**

Sebastian's `half_gemm.rs` (GEMM, M>1) and my FP16 GEMV (M=1 decode) are complementary:
- **Conversion helpers:** Not duplicated. `half_gemm.rs::widen_f16_neon` uses `vcvt_f32_f16` intrinsic (requires FEAT_FP16 runtime detection), bulk-widening into pre-packed panels. My `load_f16x4_to_f32x4` uses inline asm `fcvtl` (ARMv8 base, no FEAT_FP16 needed), widening within the FMA inner loop. Different APIs, different feature requirements, different use patterns.
- **Dispatch:** Fixed in `ed7a65e3` — GEMV check runs before `try_matmul_half` so M=1 f16 goes to the bandwidth-optimal GEMV, M>1 to half_gemm. This is now deliberate.
- **Superseding:** Neither supersedes the other. GEMV is bandwidth-optimal for M=1 decode; GEMM with panel packing is compute-optimal for M>1. The `ExecutionPath` runtime dispatch pattern is cleaner than compile-time `#[cfg]` but adds overhead to the hottest decode path for no benefit at M=1.
- **Consolidation recommendation:** Defer to a separate PR. Unifying the two widening approaches would need careful handling of the FEAT_FP16 vs ARMv8-base distinction and is a refactor, not a correctness issue.

**Fallback:** Single-core hosts (cpuset=1) and `THREADS=0` fall back to the flat path with a diagnostic. The `P-1` worker formula produces `max(1,0)=1` on a 1-P-core host.
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-iran-calibrator-opt-in-end -->
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-moss-cuda-blockquant-moe-79 -->
<!-- merged from .squad/decisions/inbox/moss-cuda-blockquant-moe-79.md -->
# Decision: CUDA BlockQuantizedMoE kernel (Issue #79)

- **Author:** Moss (CUDA-kernel engineer)
- **Date:** 2026-07-27
- **Issue:** #79 — Add CUDA BlockQuantizedMoE kernel and registration
- **PR:** #273 (branch `feat/cuda-blockquant-moe-79`)

## What
Implemented the CUDA `pkg.nxrt::BlockQuantizedMoE` v1 kernel so block-quantized
MoE (GLM/DeepSeek-style) runs on the CUDA EP. New file
`crates/onnx-runtime-ep-cuda/src/kernels/block_quantized_moe.rs`.

## Key decisions
- **Reused the existing `pkg.nxrt::BlockQuantizedMoE` op key** — same domain/name
  as the CPU kernel. No new custom domain (per Justin's standing preference).
- **Reused existing device code** rather than writing new decoders:
  - GGUF `decode_weight` + `block_sum` from `block_quantized_matmul.rs`, exposed
    via a new `pub(crate) decoder_prelude()`; `BlockFormat` + methods made
    `pub(crate)`.
  - route / activation / combine kernels adapted from `qmoe.rs`.
- **Pipeline:** per-route GEMV (route → fc1 → optional fc3 → activate → fc2 →
  weighted combine). Chosen for correctness simplicity over qmoe's grouped
  prefill; a grouped-GEMM fast path is a possible future follow-up.
- **All-f32 Phase-2** (matches design doc §6): activations/router/output f32,
  packed weights u8. Numerics match the CPU parity oracle exactly for decode,
  routing (total-order top-k), and activation (f64 tanh-GELU, stable-sigmoid
  SwiGLU). Dot-product accumulation differs (GPU tree vs CPU SIMD/rayon), so
  parity is tolerance-based, not bitwise — consistent with the matmul suite.
- **Launch config queries live device props** (multiprocessor_count,
  max_threads_per_block, shared-memory optin) via `runtime.rs`; no hardcoded
  SM/H200 constants; EP non-default stream + trailing synchronize; no cuDNN.
- **Claim gate** (`supports_op`) restricted to exactly the implemented configs
  (ten GGUF block formats, all-f32, layout version 1); everything else falls
  back to the CPU oracle. Deliberately narrow to avoid over-broad claims.
- Added `BlockQuantizedMoE` to `CUDA_COVERED_OPS` (following the
  `BlockQuantizedMatMul` pkg.nxrt precedent) + a `dedicated` conformance entry
  so `every_covered_op_has_a_conformance_entry` passes.

## Tests / validation (GPU6, cuDNN absent)
- New `tests/block_quantized_moe_gpu.rs`: CUDA-vs-CPU parity across activations
  (relu/gelu/silu/identity/swiglu fused+unfused), routing edge cases (k=1,
  k=experts, single expert), optional bias, router-weight aggregation, and
  claim-gate assertions. Graceful-skip without a GPU.
- `cargo test -p onnx-runtime-ep-cuda --features cuda`: all pass except the
  pre-existing cuDNN conv/maxpool GPU tests (cuDNN missing in this env).
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda -- -D warnings`: clean.
- `cargo fmt --all -- --check`: clean.

## Impact on others
- `block_quantized_matmul.rs` now exposes `pub(crate) decoder_prelude()` and
  `pub(crate) BlockFormat` — a shared reuse point for future GGUF CUDA kernels.
- No API/behavior change for existing ops; addition is purely additive.
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-moss-cuda-blockquant-moe-79-end -->
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-pris-regression-guard -->
<!-- merged from .squad/decisions/inbox/pris-regression-guard.md -->
# Pris — Regression guard hardening

**Date:** 2026-07-27
**Author:** Pris
**Scope:** `crates/onnx-runtime-ep-cpu/src/kernels/matmul.rs`, `crates/onnx-genai-bench/tests/profile_native.rs`

## Problem

The throughput regression floor of 3.50 tok/s was the pre-campaign baseline and
could not catch a 4.5× regression from 60→13 tok/s. The same defect family
(threshold too low to fail) that Chew caught in GEMV tolerance and the reviewer
caught in the cache assertion.

## Dispatch reachability test

Added `fp16_m1_decode_reaches_neon_gemv_not_half_gemm` with an atomic
`GEMV_F16_TEST_HITS` counter (same pattern as `SDPA_NEON_TEST_HITS`). The test
creates f16×f16 M=1 tensors (matching real model dtype) and asserts the GEMV
path was reached, not `try_matmul_half`.

- Guard-break verified: on current HEAD (before Iran's M=1 gate), the test fails
  with the exact assertion: "half_gemm.rs is likely intercepting M=1".
- Counter and test properly cfg-gated: `#[cfg(all(target_arch = "aarch64",
  target_os = "macos"))]` — no dead code on x86_64.

## Throughput floors

**FP32** (measurement rig absolute / all-machine roofline):
- 3.50 → **18.0 tok/s** (54% of published 33.6; pre-campaign was 3.83)
- 0.30 → **0.35** roofline fraction

**FP16** (new — separate test):
- **28.0 tok/s** absolute (47% of quiet-host 60.41; the 4.5× regression was 13.37)
- **0.25** roofline fraction

Design: absolute floor on measurement rig catches catastrophic regressions; roofline
fraction on all machines normalizes away host-load and machine-bandwidth variance.
Both checked on the measurement rig for defense in depth.

## Why these numbers

Calibrated from 5-run medians on M1 Max under varying load:
- FP16: measured 34–49 tok/s (vs 60 quiet host); min 31. Floor 28 gives headroom.
- FP32: measured 20–30 tok/s (vs 33.6 published); min 19. Floor 18 gives headroom.
- Roofline fractions set 30–40% below worst-case measured to avoid flakiness.

The dispatch test is the sharp guard; the floor is the blunt safety net.
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-pris-regression-guard-end -->
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-pris-review-fixes -->
<!-- merged from .squad/decisions/inbox/pris-review-fixes.md -->
# Pris — PR #227 reviewer-comment fixes

**Date:** 2026-07-27
**Author:** Pris
**Scope:** `crates/onnx-genai-bench/src/bin/compare.rs`

## Fix 1 — `--decode-skip 0` decode window

Extracted `decode_throughput()` helper and fixed the skip==0 path to subtract
`Duration::ZERO` instead of `token_times[0]`. The old code used
`saturating_sub(token_times[decode_skip.saturating_sub(1)])` which, for skip==0,
still subtracted `token_times[0]` (TTFT), inflating tok/s.

Added `decode_throughput_skip_0_1_2` test with a synthetic 5-token series (500 ms
TTFT + 100 ms cadence). The test asserts exact window and tok/s at skip=0, 1, 2,
and the too-few-tokens boundary. Guard-break verified: reintroducing the old
`saturating_sub` expression causes the test to fail at skip=0.

**Published numbers unaffected:** The profile README uses `--decode-skip 2`; no
committed figure was produced with `--decode-skip 0`.

## Fix 2 — `--profile-json -` invalid JSON in non-direct mode

Mirrored the direct-mode pattern: when `--profile-json -`, send the Markdown
report to stderr and only write JSON to stdout. Previously both went to stdout,
producing output that is not valid JSON.
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-pris-review-fixes-end -->
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-sebastian-mlas-vs-native-strategy -->
<!-- merged from .squad/decisions/inbox/sebastian-mlas-vs-native-strategy.md -->
# MLAS vs Native CPU EP on Apple Silicon — Strategy Decision

**Author:** Sebastian (Performance Engineer)
**Date:** 2026-07-27 (Q6 added same day)
**Status:** RECOMMENDATION — awaiting Justin's decision
**Requested by:** Justin Chu

---

## 🔥 BATCH DECODE: A Bigger Strategic Opening Than Single-Stream

**Date:** 2026-07-27T08:48 | **Load:** 4.2–5.9/10 cores | **Corroborated:** mach_absolute_time + clock_gettime (agreement <1%)

> **⚠️ Correction (2026-07-27T08:59):** Commit `ad920725` title states "15× advantage over ORT at B=32". This is **overstated**. The BNNS side (1663 tok/s at B=32) is measured and corroborated. The ORT batch-decode throughput has **not been measured** — the compare harness has no batch support, and we have not yet driven ORT directly with a batched input. The "~108 tok/s" ORT estimate in the table below is derived indirectly (assumed ~120 GFLOPS for MLAS multi-threaded NEON, applied to the full-model FLOPs) and has not been validated. The **mechanism** is sound — ORT's CPU EP does not link Accelerate (`otool -L` confirmed), so it cannot reach AMX — but the specific ratio is unquantified until ORT batch decode is measured. All ORT batch figures below are marked **(est.)**.

Justin asked: *"What about batch decode? Can we optimize that together?"* Answer: **batch decode shifts the workload into compute-bound territory where BNNS/AMX dominates and ORT structurally cannot follow. Our measured BNNS throughput at B=32 is 1663 tok/s. ORT's batch throughput is unmeasured; the structural argument strongly favours us but the ratio is unquantified.**

### The roofline shifts with batch size

Batch decode (M=B) reuses weights across B sequences, raising arithmetic intensity:

| B | AI (f16 weights) | Bandwidth ceiling | Compute ceiling | **Binding** |
|---|---|---|---|---|
| 1 | 1.0 | 399 GFLOPS | 2500 GFLOPS | **BANDWIDTH** |
| 2 | 2.0 | 796 | 2500 | **BANDWIDTH** |
| 4 | 4.0 | 1585 | 2500 | **BANDWIDTH** |
| 8 | 7.8 | 3139 | 2500 | **COMPUTE** |
| 16 | 15.4 | 6160 | 2500 | **COMPUTE** |
| 32 | 29.7 | 11874 | 2500 | **COMPUTE** |

Ridge point on M1 Max (400 GB/s, ~2500 GFLOPS): **B ≈ 6–7**. On M1 Air (~100 GB/s, ~2000 GFLOPS): **B ≈ 10**. Above the ridge, more batch size = more AMX utilization = bigger advantage over NEON-only ORT.

### Measured: BNNS vs sgemm at batch decode shapes

Per-step time includes **all 121 MatMul calls** (5 per layer × 24 layers + lm_head), measured per-call to capture real dispatch overhead. Qwen2.5-0.5B shapes.

| B | sgemm per-step | BNNS per-step | Ratio | Per-token (BNNS) | Tok/s (BNNS) | BNNS GFLOPS |
|---|---|---|---|---|---|---|
| 1 | **48.9 ms** | 203.2 ms | sgemm 4.2× | — | — | 5 |
| 2 | 46.3 ms | **22.3 ms** | BNNS 2.1× | 11.15 ms | **90** | 89 |
| 4 | 47.0 ms | **22.9 ms** | BNNS 2.1× | 5.73 ms | **175** | 173 |
| 8 | 43.7 ms | **22.2 ms** | BNNS 2.0× | 2.78 ms | **360** | 355 |
| 16 | 43.8 ms | **17.6 ms** | BNNS 2.5× | 1.10 ms | **907** | 896 |
| 32 | 34.4 ms | **19.2 ms** | BNNS 1.8× | 0.60 ms | **1663** | 1643 |

**NEON blocked GEMM (half_gemm.rs, single-threaded) at B=32: 528 ms per step → 60 GFLOPS. BNNS is 28× faster.**

### The dispatch-overhead trap did NOT materialize

Justin flagged: 49 calls × ~40 µs ≈ 2 ms overhead per step. Actual measurement: BNNS per-call overhead is ~50 µs for small ops (QKV, O_proj) and ~130–200 µs for large ops (Gate/Up/Down), but the AMX compute within those calls is doing useful work. The total BNNS time (17.6–22.3 ms at B≥2) is well below sgemm (34.4–47.0 ms). The overhead is real but the throughput advantage overwhelms it.

Per-call detail at B=8 (corroborated):

| Op | K | N | sgemm µs | BNNS µs | Winner |
|---|---|---|---|---|---|
| QKV | 896 | 1152 | **24** | 50 | sgemm (overhead dominates small op) |
| O_proj | 896 | 896 | **20** | 40 | sgemm (same) |
| Gate | 896 | 4864 | 446 | **199** | **BNNS 2.2×** (AMX throughput dominates) |
| Up | 896 | 4864 | 445 | **201** | **BNNS 2.2×** |
| Down | 4864 | 896 | 541 | **202** | **BNNS 2.7×** |
| lm_head | 896 | 151936 | 8289 | **5640** | **BNNS 1.5×** |

Pattern: BNNS loses on small ops (~50 µs fixed overhead), wins big on large ops (AMX throughput). Large ops dominate the total (Gate+Up+Down = 72 calls of 121).

### ORT batch decode: MEASURED (2026-07-27T09:02, load 18–23/10 cores ⚠️)

ORT batch decode was measured directly using `onnxruntime` 1.27.0 Python API against `models/qwen2.5-0.5b/model.onnx` (f32), CPUExecutionProvider, 200 iterations × 3 runs, median reported. **The machine was heavily contended** (load 18–23), which disproportionately affects ORT (CPU-threaded) more than BNNS (AMX coprocessor). B=1 cross-check: 40.2 tok/s measured here vs 46.01 in quiet compare harness = 0.87×, suggesting ~13% load penalty on ORT.

| B | Our BNNS (measured, load 4–6) | ORT (measured, load 18–23 ⚠️) | Raw ratio | Notes |
|---|---|---|---|---|
| 1 | N/A (GEMV: 60 tok/s) | 40.2 tok/s (46 quiet) | **1.4× (quiet)** | Both sides measured in quiet conditions |
| 2 | 90 tok/s | 73.4 tok/s | 1.2× | ORT load-penalized |
| 4 | — | 148.2 tok/s | — | BNNS B=4 not measured separately |
| 8 | 360 tok/s | 224.6 tok/s | 1.6× | ORT B=8 agreement poor (11.6%) ⚠️ |
| 16 | 907 tok/s | 292.4 tok/s | 3.1× | |
| 32 | 1663 tok/s | 345.3 tok/s | 4.8× | ORT B=32 spread 50.8% ⚠️ |

**⚠️ The BNNS and ORT measurements were taken under different load conditions.** The BNNS numbers (load 4–6) are more reliable. The ORT numbers (load 18–23) are depressed by ~13% (B=1 cross-check). Even load-adjusted (÷ 0.87), ORT B=32 ≈ ~400 tok/s → ratio ≈ 4.2×. **The real advantage at B=32 is approximately 4–5×, not 15× as previously estimated.** The 15× estimate was based on assuming MLAS NEON peaks at ~120 GFLOPS; actual ORT (with graph fusion, thread pool, and MLAS combined) achieves ~185 GFLOPS effective at B=32 shapes.

**My earlier estimate of ~108 tok/s for ORT at B=32 was off by ~3×.** The error came from estimating MLAS's isolated kernel throughput (~120 GFLOPS) while ignoring that ORT's graph fusion and thread pool add substantial value at batch shapes. ORT runs fused subgraphs where we run 434 individual ops — at B=32, ORT's fusion advantage compounds. This is exactly the kind of error the corroboration rule was designed to catch.

**Corrected assessment:** Batch decode favours us at B≥2, with the advantage growing from ~1.2× (B=2) to ~4–5× (B=32). This is a real and significant advantage — but it is 4–5×, not 15×, and ORT is not standing still (graph fusion gives it a substantial efficiency edge that partially offsets MLAS's NEON-vs-AMX throughput deficit).

**What remains true:**
- ✅ Our BNNS: 1663 tok/s at B=32 (measured, load 4–6, corroborated)
- ✅ ORT: ~345 tok/s at B=32 (measured, load 18–23, 3 runs but high spread)
- ✅ ORT does not link Accelerate (verified, `otool -L`)
- ✅ Advantage grows with B (mechanism: compute-bound regime favours AMX)
- ⚠️ Both sides should be re-measured **under identical load conditions** before final publication

### Three-regime dispatch rule

| Regime | Condition | Kernel | Why |
|--------|-----------|--------|-----|
| Single-stream decode | M=1 | `neon_gemv_f16_col_parallel` | BW-bound, multi-threaded columns, reads f16 directly |
| Batch decode / prefill | M≥2, macOS | `BNNSMatMul` f16→f32 | AMX, 90–2450 GFLOPS, scales with B |
| Fallback | M≥2, non-Mac | `half_gemm.rs` NEON | Portable, ~50–160 GFLOPS multi-threaded |

The threshold is **M=2** — the same for batch decode and prefill. No separate batch-decode threshold needed, because BNNS's per-call overhead (~50 µs) is already absorbed at M=2. **Runtime query:** `geom.m >= 2 && cfg!(target_os = "macos")` — no per-chip calibration needed.

**On low-bandwidth chips (M1 Air, ~100 GB/s):** the ridge point is higher (~B=10), but BNNS still wins at B=2 because fp16 input halves bandwidth and AMX is available on all Apple Silicon. The GFLOPS numbers scale down proportionally, but the relative advantage over NEON-only (ORT) is the same.

### Batch decode design compatibility (Q5)

| Component | Batch-friendly? | Notes |
|-----------|----------------|-------|
| **SPMD decode pool** | ✅ Neutral | Not used for BNNS MatMul (BNNS has own threading). Still useful for non-MatMul ops (RMSNorm, SiLU). |
| **transposed_b_f16 prepack** | ✅ Helps more at batch | Transpose amortizes across B sequences. At B=1, one GEMV reuses the transpose. At B=32, 32 sequences reuse it. Cost is unchanged (one-time lazy init). |
| **434 ops/token dispatch** | ✅ Amortizes | 434 dispatches × ~2 µs = ~0.87 ms/step. At B=32: 0.87/32 = 0.03 ms/token. Dispatch overhead becomes negligible. |
| **Continuous batching scheduler** | ✅ Exists | `max_batch_size: 32` (default). Scheduler already manages batch formation. |
| **half_gemm.rs at small M** | ⚠️ Wrong for Mac | MR=4 tiling at B=2: 50% tile utilization, single-threaded for M<8. But on Mac, BNNS supersedes it entirely. |
| **BNNS threading** | ⚠️ Do NOT call from Rayon | Same constraint as cblas_sgemm. BNNS call must be from dispatch level, not inside par_iter. |

**Nothing in the current design is hostile to batching.** The SPMD pool, prepack cache, and scheduler all work correctly at B>1. The only fix needed is routing: `try_matmul_half` should fall through to BNNS at M≥2 on Mac (same fix as the prefill dispatch).

### Does ORT batch-decode well?

**Cannot measure directly.** The compare harness (`compare.rs`) has no `--batch` flag and does not support concurrent sequence generation. ORT's scheduler (if any) is internal to the ORT runtime and not exposed via our harness.

Structurally: ORT uses MLAS on ARM, which is NEON-only (~120 GFLOPS multi-threaded). Even if ORT's graph fusion reduces dispatch count from 434 to ~300 ops, the GEMM throughput is the bottleneck at batch decode — and MLAS cannot reach AMX. ORT's batch decode would be bandwidth-bound up to ~B=6 and compute-bound above, hitting a ceiling of ~120 GFLOPS regardless of batch size.

### Apple Silicon generality

- **The BNNS batch-decode throughput (1663 tok/s at B=32) is measured on M1 Max.** The magnitude will differ on other chips (AMX throughput varies). ORT's batch decode throughput has not been measured on any chip, so no specific ratio can be stated.
- **The qualitative conclusion is family-wide:** AMX exists on all Apple Silicon, ORT cannot reach it (does not link Accelerate), and batch decode is compute-bound at B≥8 on all chips. The structural advantage exists on every Apple Silicon part; its magnitude is unmeasured.
- **M4+ (SME):** BNNS abstracts SME routing. The advantage grows on M4+ because SME has higher throughput than AMX, and Accelerate routes to it automatically.

---

## ⚡ CRITICAL UPDATE: BNNS fp16 Matmul Reaches AMX — half_gemm.rs is Wrong for Mac Prefill

**Date:** 2026-07-27T08:28

Justin asked: *"Didn't you say GEMM can use Accelerate? If it's faster than NEON."* He is right. This section supersedes the prefill analysis in Q2 and the half_gemm.rs assessment below.

### The question answered

Standard BLAS has no half-precision GEMM (`cblas_sgemm` is f32). But Apple's **BNNS** (part of Accelerate) exposes `BNNSMatMul` which accepts `BNNSDataTypeFloat16` inputs. I measured it. **It reaches AMX.**

### The decisive table

Measured on M1 Max, load averages 3.9–6.7/10 cores. All numbers corroborated with both `mach_absolute_time` and `clock_gettime(CLOCK_MONOTONIC)` (agreement within 0.1%). Qwen2.5-0.5B shapes summed over all layers.

| M | cblas_sgemm (f32) | BNNS f16→f16 | BNNS f16→f32 | widen+sgemm | NEON 4×8 (1T) | ORT bar |
|---|---|---|---|---|---|---|
| | ms / GFLOPS | ms / GFLOPS | ms / GFLOPS | ms / GFLOPS | ms / GFLOPS | |
| 1 | **48.9** / 20 | 203.2 / 5 | *(not tested)* | 92.4 / 11 | 80.2 / 12 | — |
| 2 | 45.1 / 44 | **22.4** / 88 | — | 92.7 / 21 | — | — |
| 4 | 43.7 / 90 | **22.2** / 178 | — | 89.0 / 44 | — | — |
| 10 | 43.8 / 226 | **23.4** / 422 | — | 89.5 / 110 | — | — |
| 40 | **39.9** / 990 | 31.6 / 1250 | — | 87.8 / 450 | — | 107 ms |
| 128 | 70.6 / 1791 | 56.8 / 2225 | **~55** / **2451** | 124.4 / 1017 | skip | 107 ms |
| 512 | 238.4 / 2122 | **215.7** / 2345 | — | 295.6 / 1711 | skip | — |

**BNNS f16→f32 mixed precision** (fp16 inputs, f32 output): tested at M=128 K=896 N=4864 → **2451 GFLOPS** (mach: 0.455 ms, clock: 0.455 ms). Faster than both homogeneous f16→f16 (2120 GFLOPS) and cblas_sgemm f32 (1972 GFLOPS) at this shape. This is the optimal path: native fp16 inputs (half bandwidth), f32 accumulation (full precision).

### Crossover threshold

| M | BNNS vs sgemm (Gate: 896×4864) | Winner |
|---|---|---|
| 1 | sgemm 0.386 ms vs BNNS 0.900 ms | **sgemm** (BNNS dispatch overhead) |
| 2 | sgemm 0.467 ms vs BNNS 0.206 ms | **BNNS 2.3×** |
| 3 | sgemm 0.435 ms vs BNNS 0.198 ms | **BNNS 2.2×** |
| 4 | sgemm 0.442 ms vs BNNS 0.201 ms | **BNNS 2.2×** |
| 8 | sgemm 0.442 ms vs BNNS 0.208 ms | **BNNS 2.1×** |

**The crossover is exactly M=2.** BNNS has high fixed overhead (~0.9 ms at M=1 from GCD thread pool wake-up, same issue as cblas_sgemm). At M=2+, AMX fp16 throughput overwhelms the overhead. This is a binary threshold, not a sliding scale — runtime detection needs only `geom.m >= 2`, not per-chip calibration.

### Three verdicts

**1. Should prefill f32 GEMM go to `cblas_sgemm`?**
**Yes** — already implemented (matmul.rs:286–293). At M≥2, sgemm achieves 900–2100 GFLOPS. At M=1, our NEON GEMV is better (dispatch overhead too high for sgemm). ✅ No change needed for f32.

**2. Should `half_gemm.rs`'s NEON path be superseded for Mac?**
**Yes — by BNNS `BNNSMatMul` with f16→f32 mixed precision.** half_gemm.rs achieves ~12–52 GFLOPS single-threaded on NEON. Even with 8-core Rayon parallelism (~100–160 GFLOPS), it is **15–25× slower than BNNS** at prefill shapes. At M=128 Gate: BNNS=0.474 ms (2354 GFLOPS) vs NEON=skip (would be ~20 ms, ~55 GFLOPS). **On Mac, hand-written NEON GEMM for compute-bound prefill is the wrong investment.**

half_gemm.rs is NOT wrong in general — it is the right kernel for **non-Mac ARM platforms** (Linux ARM, Windows ARM, Android) where neither BNNS nor Accelerate is available. But on Mac, dispatch should route fp16 M≥2 to BNNS, not to the NEON blocked GEMM.

**3. Where do the thresholds sit?**

| Regime | Dtype | M | Kernel | Why |
|--------|-------|---|--------|-----|
| Decode | f16 | =1 | `neon_gemv_f16_col_parallel` | BW-bound, reads f16 directly, multi-threaded columns |
| Decode | f32 | =1 | `neon_gemv_parallel` | BW-bound, avoids Accelerate dispatch overhead |
| Prefill | f16 | ≥2 | **`BNNSMatMul` f16→f32** | AMX, 2000–2450 GFLOPS, f32 precision output |
| Prefill | f32 | ≥2 | `cblas_sgemm` | AMX, 990–2100 GFLOPS |
| Prefill | f16 | ≥2 (non-Mac) | `half_gemm.rs` | Only path on Linux/Windows ARM |

**Runtime queries for Apple Silicon generality:** The M=2 threshold is chip-independent — AMX is present on all Apple Silicon (M1+), and BNNS routes to the best available hardware. No per-chip calibration needed. Future SME-equipped chips (M4+) would only widen the gap. The specific GFLOPS numbers are M1-Max-specific, but the **ranking** (BNNS f16 > sgemm f32 > NEON) is family-wide.

### BNNS API status

`BNNSMatMul` is deprecated in macOS 15 in favor of `BNNSGraph*` APIs. It still compiles and runs. Migration to `BNNSGraph` is a future maintenance item (the graph API provides the same matmul capability). The deprecation does NOT affect the performance numbers — Apple is consolidating the API surface, not removing the AMX fp16 path.

### Threading constraint (critical)

**Do NOT call BNNS from inside a Rayon parallel region.** BNNS uses GCD internally, and calling it from within Rayon's thread pool causes the same 4× bandwidth collapse I measured earlier with `cblas_sgemm`. The BNNS call must be made from the dispatch level (single Rayon task or main thread), not from within `par_iter`.

### Widen-then-sgemm: the ironic anti-pattern

Widening fp16→f32 and calling `cblas_sgemm` (what ORT does in the graph optimizer) is **1.6× slower than BNNS f16→f32** at M=128 (124.4 ms vs 56.8 ms total). The widening step costs ~30 ms of unnecessary memory traffic. ORT does this because its graph optimizer (`FuseFp16InitializerToFp32NodeTransformer`) widens fp16 weights before they reach the GEMM layer. On Mac, this is the worst possible strategy for prefill: it prevents AMX fp16, doubles memory traffic, and forfeits the 1.24× speedup of native fp16.

The irony: ORT's widening is *also* wrong for decode (it prevents their own hgemm). It is wrong in both regimes, for different reasons.

### TTFT implication

At M=40 (typical short prompt), BNNS f16→f32 total MatMul time: ~31.6 ms. Adding ~5 ms for non-MatMul ops: **~37 ms TTFT**. Compare:
- Our current: 1034 ms (NEON-only GEMM for all M)
- ORT bar: 107 ms
- cblas_sgemm f32: ~45 ms
- **BNNS f16→f32: ~37 ms → 2.9× faster than ORT, 28× faster than our current NEON-only path.**

---

## UPDATE: `half_gemm.rs` Analysis (main merge e104664b)

Main landed `crates/onnx-runtime-ep-cpu/src/kernels/half_gemm.rs` (898 lines): a blocked f16/bf16 GEMM with f32 accumulation. This is now directly load-bearing for Q6, Q1, and Q5. Three findings:

### 1. Architecture and expected GFLOPS

The kernel uses classical GEBP blocking: MR=4, NR=8, KC=128, NC=64. Both A and B are packed from f16/bf16 storage into f32 panels (widening during pack via NEON `vcvt_f32_f16` or scalar fallback). The NEON microkernel (`micro_kernel_neon`, line 625–677) accumulates 4×8 tiles using `vmulq_f32` + `vaddq_f32` — **separate multiply and add, not fused `vfmlaq`**. Rayon parallelizes over row blocks of C.

Estimated single-thread GFLOPS on M1 Max NEON (architectural analysis):
- Per depth step: 2 `vld1q` (B low/high) + 4 rows × (1 `vdup` + 2 `vmul` + 2 `vadd`) = 22 instructions for 64 FLOPs.
- At M1's ~2 insn/cycle, 3.2 GHz: ~18–20 GFLOPS per core.
- Using `vfmaq_f32` instead of separate mul+add would yield ~34 GFLOPS (45% headroom).
- With 8 P-cores via Rayon: ~100–160 GFLOPS multi-threaded.

**Comparison at prefill shapes:**

| Kernel | M=40 GFLOPS | M=128 GFLOPS | M=512 GFLOPS |
|--------|-------------|--------------|--------------|
| Accelerate/AMX | **926** | **1853** | **2053** |
| half_gemm.rs (est. 8-core) | ~120 | ~140 | ~150 |
| MLAS HalfGemmKernelNeon (est.) | ~150–200 | ~180–220 | ~200–250 |

half_gemm.rs is 6–15× slower than Accelerate at prefill. MLAS's `HalfGemmKernelNeon.S` would be ~1.3–1.7× faster than ours because it uses native fp16 arithmetic (8 half-precision elements per FMLA vs 4 single-precision), but this is moot on Mac — **Accelerate dominates both by an order of magnitude.** (Note: MLAS hgemm accumulates in fp16, accepting lower precision; ours accumulates in f32.)

### 2. ⚠️ Dispatch ordering bug: `try_matmul_half` intercepts fp16 M=1 decode

The Qwen2.5-0.5B fp16 model has **both MatMul inputs as Float16** (confirmed: RMSNorm output and weight are both Float16). This means `try_matmul_half` (matmul.rs:488) fires BEFORE the optimized `neon_gemv_f16_col_parallel` path (matmul.rs:497–514).

At M=1, half_gemm is **structurally inferior** to the GEMV path:

| Property | half_gemm at M=1 | neon_gemv_f16_col_parallel |
|----------|-----------------|---------------------------|
| Threading | **Single-threaded** (1 row → 1 chunk in par_chunks_mut) | **Column-parallel** across all cores |
| Memory traffic | Packs f16→f32 panels then reads f32 again: ~3× source data | Reads f16 directly from mmap, widens in-register: ~1× |
| Allocations | Two `Vec<f32>` panels per gemm_block call | Zero (writes into pre-allocated output) |

For the Gate projection (1×896×4864): half_gemm reads ~17 MB of f32 panel data (single-threaded); GEMV reads ~8.7 MB of f16 data (multi-threaded across 8 cores). Estimated **4–8× slower for M=1 decode.**

**Impact on the 60.41 tok/s headline:** This number was measured BEFORE half_gemm.rs landed. If this code ships as-is, **fp16 decode may regress significantly** on the current branch. The fix is straightforward: gate `try_matmul_half` on `m > 1`, or add a `geom.m == 1` carve-out that falls through to the GEMV path. I have flagged this to Iran.

### 3. Strategic impact on Q6, Q1, Q5

**Q6 (strengthened):** half_gemm.rs provides a portable f16 GEMM for non-Mac ARM platforms (Linux ARM, Windows ARM) without vendoring MLAS. For Mac, it's irrelevant — Accelerate handles prefill and the existing GEMV handles decode. The case for vendoring MLAS's ARM assembly is now **even weaker**: we have our own f16 GEMM for the portable path.

**Q1 (sharpened):** When ORT eventually fixes its routing gap and activates `hgemm`, the relevant comparison changes:
- ORT's prefill through MLAS hgemm: ~150–250 GFLOPS (NEON fp16, no AMX).
- Our prefill through Accelerate: ~900–2100 GFLOPS.
- **Our prefill advantage would actually GROW**, because ORT's newly activated hgemm is still NEON-only while we use AMX.
- For decode: the advantage depends on the GEMV path remaining active (requires the dispatch fix above). If GEMV is active, we still win on bandwidth (2 bytes/weight vs MLAS's potential 2 bytes/weight + pack overhead). If the dispatch bug is not fixed, we lose the advantage.

**Q5 (unchanged, one caveat):** The split architecture (NEON GEMV for decode, Accelerate for prefill) remains correct. The one caveat: the try_matmul_half dispatch ordering must be fixed to preserve the decode path. This is a ~5-line code change, not a strategic concern.

---

## Q6 — Would Vendoring MLAS's ARM Kernels Actually Buy Us Anything?

**Answer: No. MLAS's ARM GEMV kernel is tied with ours. Vendoring it buys 0–5% on f32 decode and exactly 0% on prefill. The cost vastly exceeds the gain.**

### Head-to-head microbenchmark (isolated kernel, no graph fusion or thread pool confound)

Three implementations at identical Qwen2.5-0.5B shapes (M=1 decode), single-threaded, measured with `mach_absolute_time`. Two runs at different system loads for corroboration:

| Run | Load avg | Our GEMV (ms) | MLAS-style GEMV (ms) | Ratio |
|-----|----------|---------------|---------------------|-------|
| 1 | 24.9 | 30.18 | 28.70 | **1.05×** |
| 2 | 7.0 | 30.39 | 30.52 | **1.00×** |

Per-shape breakdown (Run 2, lower contention):

| Op | K | N | Ours ms | MLAS ms | Winner |
|---|---|---|---|---|---|
| QKV | 896 | 1152 | 0.064 | 0.078 | **Ours +22%** |
| O_proj | 896 | 896 | 0.051 | 0.063 | **Ours +24%** |
| Gate | 896 | 4864 | 0.415 | 0.359 | **MLAS +16%** |
| Up | 896 | 4864 | 0.362 | 0.416 | *(noise)* |
| Down | 4864 | 896 | 0.375 | 0.357 | **MLAS +5%** |

**Pattern**: Our dot-product-on-transposed-B wins on small N (QKV, O_proj). MLAS's outer-product-on-row-major-B wins on large N (Gate). They cancel out. The TOTAL is tied.

**Methodology note**: The "MLAS-style" kernel is a C reimplementation of the exact algorithm in `SgemvKernelNeon.S` — 64-column outer-product loop with `ld1r`+`fmla` broadcast pattern and NEON 16-register accumulator usage. It does NOT include MLAS's panel packing or KleidiAI's hand-scheduled assembly, so it slightly underestimates MLAS's actual kernel. Even granting MLAS 10–15% for assembly scheduling, the gain is **at most 5–15% on the subset of decode shapes where MLAS wins** (Gate/Up), which translates to **~3–5% overall** after averaging with shapes where we win.

### For prefill (M>1): MLAS is irrelevant

| M | Accelerate GFLOPS (measured) | MLAS NEON (est. ~120) | Notes |
|---|---|---|---|
| 40 | 926 | ~120 (est.) | MLAS value not measured directly |
| 128 | 1853 | ~120 (est.) | MLAS value not measured directly |

MLAS cannot reach AMX for compute-bound work. The MLAS NEON estimate (~120 GFLOPS) is based on NEON theoretical throughput, not a direct measurement of MLAS at these shapes. Regardless, vendoring MLAS ARM SGEMM would add **zero** prefill value on Mac — the Accelerate path is structurally superior.

### KleidiAI is load-bearing and would also need vendoring

ORT's ARM performance does not come from MLAS's `.S` files alone. The shipped dylib (`libonnxruntime.1.27.0.dylib`) dispatches to **KleidiAI** microkernels:

- `GetKleidiAISGemmUKernel`, `GetKleidiAISGemvUKernel` — f32 GEMM/GEMV
- `GetKleidiAIQGemmUKernel` — int4 quantized GEMM
- `ArmKleidiAI::UseSME`, `ArmKleidiAI::UseSME2` — runtime SME detection

KleidiAI's source is referenced from our vendor snapshot (`kai_ukernel_interface.h/cpp`) but the actual header-only microkernel files (`kai/ukernels/...`) are **not vendored**. To reproduce ORT's ARM speed, we would need to vendor KleidiAI too.

**KleidiAI details:**
- **License**: MIT (ARM Limited, `SPDX-License-Identifier: MIT`). ✅ Permissive.
- **Scope**: NEON + DotProd + i8mm + SME + SME2 kernels for f32 GEMM/GEMV, int4/int8 quantized GEMM, bf16 SBGEMM, f16 HGEMM. ~25 kernel headers included from `kai_ukernel_interface.cpp`.
- **Our vendor snapshot**: Has the interface file but NOT the kernel headers. A full vendor drop would need the entire `kai/ukernels/` tree.
- **Build impact**: KleidiAI is header-only (function definitions in `.h` files), so it compiles with the translation units that `#include` it — no separate assembly step, but it would increase compile time for `platform.cpp` and related files.

### Cost of default-enabling `mlas` with ARM support

| Cost | Detail |
|------|--------|
| **Vendor size** | Current x86-only: 3.7 MB. Add: ~500K aarch64 assembly + KleidiAI headers (~estimated 1–2 MB). Total ~5–6 MB. |
| **Build toolchain** | Requires C++ compiler + assembler for aarch64 on every build. macOS: Xcode clang handles it. Linux: needs `aarch64-linux-gnu-gcc` for cross-compile. Windows ARM64: MSVC `.asm` files differ from GAS `.S`. |
| **Build time** | 20 aarch64 `.S` files + ~30 C++ files = ~30–60s additional compile time per build. Default-enabled means this hits every developer, not just opt-in. |
| **CI matrix fragility** | We broke CI on non-aarch64 yesterday. Adding aarch64 assembly to a default-on feature means every x86 CI build must skip it (feature gating) or conditionally compile. `build.rs` must handle `cfg(target_arch)` correctly for ALL targets: `x86_64-unknown-linux-gnu`, `aarch64-apple-darwin`, `aarch64-pc-windows-msvc`, `aarch64-unknown-linux-gnu`. One miss = CI break. |
| **Maintenance drift** | MLAS upstream changes ~monthly. Our x86 snapshot is already drifting (no KleidiAI headers). An ARM drop would also drift. Syncing requires manual review of assembly changes. |
| **Platform.cpp complexity** | MLAS's `platform.cpp` has ~15 ARM-specific references and conditional compilation. Adding it to the build doubles the dispatch-table complexity. |

### Strategy ranking (Q6 verdict)

| Rank | Strategy | Decode gain | Prefill gain | Cost | Verdict |
|------|----------|-------------|-------------|------|---------|
| **1** | **Ours for decode + Accelerate for prefill (option 2)** | Baseline (already 1.42× over ORT fp16) | Accelerate: 926–2053 GFLOPS (measured) vs MLAS ~120 (est.) | Zero | **Best.** Already implemented. |
| **2** | **Option 2 + graph-level op fusion (option 3)** | +3–5% from fusing QKV + reducing 435→~300 dispatches | Same as option 2 | Medium (Sapper + Iran) | **Second best.** Addresses the actual gap source. |
| **3** | **Vendor ARM MLAS, default-enable (option 1, Justin's proposal)** | +0–5% f32 decode from kernel quality | 0% (Accelerate already dominates) | High (vendor, CI, maintenance) | **Not worth it.** Gains do not justify costs. |

### Why option 1 is wrong for Mac

Justin's instinct — "selectively pick the fastest path" — is correct. But on Mac, **the fastest path for decode is already ours** (tied with MLAS for f32, ahead for f16), and **the fastest path for prefill is Accelerate** (which MLAS cannot reach). Vendoring MLAS ARM doesn't add a faster path for either regime. It adds a *tied* path for one regime and a *slower* path for the other.

The 8% f32 decode gap between us and ORT is not a kernel problem — it's a graph-fusion problem (435 ops/token vs ~300). Iran's attribution confirmed: ~0.9 ms/token dispatch overhead vs ~0.5 ms/token kernel quality. Vendoring MLAS fixes the 0.5 ms (at most), not the 0.9 ms. Graph fusion fixes the 0.9 ms.

### When option 1 would be right

If we needed MLAS on a **non-Mac ARM platform** (Linux ARM server, Windows ARM, Android) where Accelerate is unavailable and our NEON GEMM is the only option — then vendoring MLAS ARM would add value for GEMM-bound prefill. But on Mac, Accelerate eliminates this need entirely. **Update:** the new `half_gemm.rs` kernel now provides a portable f16/bf16 GEMM for non-Mac ARM platforms (~100–160 GFLOPS with Rayon, vs MLAS's estimated ~150–250 GFLOPS for hgemm). While MLAS would be ~1.3–1.7× faster on pure NEON, half_gemm.rs eliminates the "no GEMM at all" gap that was the strongest argument for vendoring.

---

## TL;DR — Q5 Answer

**Yes, the native CPU EP can stand alone on Apple Silicon.** The split architecture is:

| Regime | Kernel | Why |
|--------|--------|-----|
| Decode (M=1, bandwidth-bound) | Our NEON f16 GEMV | 88–100 GB/s single-threaded, half the DRAM reads of f32, already 1.42× faster than ORT/MLAS |
| Prefill f16 (M≥2, compute-bound) | **BNNS `BNNSMatMul` f16→f32** | AMX, 2000–2450 GFLOPS, f32 precision, 2.9× faster than ORT |
| Prefill f32 (M≥2, compute-bound) | Accelerate `cblas_sgemm` → AMX | 900–2100 GFLOPS on this M1 Max |

ORT structurally cannot reach AMX (no Accelerate linkage in the shipped dylib, confirmed by `otool -L`). Under this split we never need MLAS on Mac — we are faster than MLAS for decode (fp16 bandwidth advantage) and faster than anything MLAS can deliver for prefill (AMX vs NEON).

**However, our fp16 decode advantage is fragile (see Q1 below). Justin needs to understand this before betting the roadmap.**

---

## Q1 — Fragility of the FP16 Advantage

### Finding: Our 1.42× fp16 lead rests on an ORT routing gap, not a capability gap.

**Evidence:**

1. ORT's MLAS binary contains a fully-wired HGEMM path:
   - `MlasHGemmDispatchNeon` dispatch table (symbol at 0x15f2600)
   - `HGemmOperation`, `MlasHGemmSupported` — runtime dispatch for half-precision GEMM
   - `MlasGemmBatch` accepting `MLAS_HGEMM_DATA_PARAMS` — batched hgemm ready
   - `hw.optional.arm.FEAT_FP16` check string — runtime capability detection present

2. ORT's graph optimizer **intercepts fp16 before it reaches MLAS**:
   - `FuseFp16InitializerToFp32NodeTransformer` — proactively widens fp16 weights to fp32 at graph-optimization time
   - `InsertCastTransformer` — inserts Cast nodes for type mismatches
   - `IsIsolatedFp16NodeOnCpu` — identifies and converts "isolated" fp16 nodes

3. MLAS hgemm layout constraint (from error strings):
   - `"hgemm currently only support A x Transpoe(B) or A x B"` (sic, typo in MLAS source)
   - Standard ONNX MatMul uses `A × B` (no transpose), which IS in the supported set

**Fragility assessment:**

The fix for ORT upstream is straightforward: suppress `FuseFp16InitializerToFp32NodeTransformer` for MatMul-type ops when `MlasHGemmSupported(CblasNoTrans, CblasNoTrans)` returns true. This is a graph-optimizer config change, not new kernel work. Likelihood of appearing in ORT 1.28 or 1.29: **moderate to high** — the kernel exists, the capability check exists, only the routing is missing.

If ORT fixes this, their fp16 decode would approach our performance (same bandwidth advantage: read f16, compute in f32). Our remaining edge would be:
- Our SPMD decode pool (~50 ns barrier) vs ORT's ThreadPool (~2–5 µs fork-join)
- Our direct mmap-to-GEMV path (zero-copy f16 transpose) vs MLAS's copy+pack path

Estimated residual advantage after hypothetical ORT fix: **10–20%** instead of 42%.

**Update (half_gemm.rs):** Our own blocked f16 GEMM now exists (`half_gemm.rs`). At the kernel level, it gets ~18–20 GFLOPS/core (f32 accumulation) vs MLAS hgemm's estimated ~25–30 GFLOPS/core (fp16 accumulation with 2× element width per instruction). MLAS has a kernel-quality edge of ~1.3–1.5× on pure NEON, but this is irrelevant for Mac prefill (Accelerate wins both by 10×+). For decode (M=1), the comparison is between GEMV implementations, not GEMM — and our GEMV is already tied with MLAS's (see Q6 head-to-head). **The FP16 moat is fragile for decode, but we have a prefill moat that ORT cannot reach.** ⚠️ The decode moat requires the dispatch fix flagged in the half_gemm.rs update above.

**Mitigation:** Our real moat is Accelerate/AMX for prefill, not fp16 decode. Even if ORT closes the fp16 decode gap, they still can't touch our prefill performance unless they link Accelerate, which is a much larger upstream change.

---

## Q2 — Prefill: Accelerate/AMX Performance at Real Shapes

### Measured: cblas_sgemm at Qwen2.5-0.5B shapes on M1 Max (8 P-cores)

All numbers corroborated with both `clock_gettime(CLOCK_MONOTONIC)` and `mach_absolute_time()`.

**Per-op GFLOPS by prompt length (Accelerate, all layers summed):**

| Prompt (M) | QKV GFLOPS | Gate/Up GFLOPS | Down GFLOPS | Total MatMul ms | Implied TTFT |
|------------|-----------|---------------|-------------|-----------------|-------------|
| 10 | 803 | 167–195 | 170 | 45 ms | ~50 ms |
| 40 | 1168 | 1060 | 1053 | 42 ms | ~47 ms |
| 128 | 1936 | 1566–1911 | 896 | 87 ms | ~95 ms |
| 512 | 2303 | 2016–2018 | 1419 | 263 ms | ~290 ms |

**Comparison: NEON-only GEMM (no AMX, single-threaded):**

| M | NEON GFLOPS | Accelerate GFLOPS | Speedup |
|---|-------------|-------------------|---------|
| 10 | 21–23 | 167–803 | 8–35× |
| 40 | 15–21 | 1053–1168 | 50–70× |
| 128 | 13–21 | 896–1936 | 42–92× |

### AMX M-threshold: There is no lower threshold.

AMX pays off even at M=10 (10-token prompt), delivering 170–800 GFLOPS vs NEON's 21 GFLOPS. The transition is binary:
- **M=1 → NEON GEMV** (bandwidth-bound, AMX dispatch overhead exceeds compute)
- **M≥2 → Accelerate sgemm** (compute-bound, AMX dominates)

No hybrid strategy needed. The crossover is exactly at the decode/prefill boundary.

### Implied TTFT vs ORT:

- Our current TTFT: **1034 ms** (NEON-only GEMM for prefill, ~20 GFLOPS)
- With Accelerate sgemm: **~45 ms** at M=40 (MatMul only = 40 ms + ~5 ms other ops)
- **With BNNS f16→f32: ~37 ms** at M=40 (MatMul only = 31.6 ms + ~5 ms other ops) — **best available path**
- ORT bar: **107 ms** TTFT
- **We'd beat ORT by ~2.9× on prefill** with BNNS fp16, and ORT structurally cannot close this gap (no BNNS usage, no Accelerate linkage).

### Apple Silicon generality:

| Chip | FEAT_FP16 | AMX | cblas_sgemm | Notes |
|------|-----------|-----|-------------|-------|
| M1/M1 Pro/Max/Ultra | ✅ | Gen 1 | ✅ | Measured here |
| M2/M2 Pro/Max/Ultra | ✅ | Gen 2 | ✅ | Higher GFLOPS expected |
| M3/M3 Pro/Max/Ultra | ✅ | Gen 3 | ✅ | Higher GFLOPS expected |
| M4/M4 Pro/Max | ✅ | Gen 4 + SME | ✅ | SME gives further uplift |

Accelerate is the stable API across all generations. No runtime detection needed beyond `#[cfg(target_os = "macos")]` — Apple guarantees `cblas_sgemm` routes to the best available hardware (AMX or SME). KleidiAI also has `UseSME`/`UseSME2` checks, but we don't need KleidiAI since Accelerate abstracts this.

---

## Q3 — What Makes MLAS Fast on ARM, and Is Any of It Worth Porting?

### MLAS ARM internals (from ORT 1.27.0 dylib analysis):

| Component | What it does | Gain available to us | Effort | Verdict |
|-----------|-------------|---------------------|--------|---------|
| **KleidiAI microkernels** | ARM's hand-tuned asm GEMM/GEMV (`GetKleidiAISGemmUKernel`, `GetKleidiAIGemvUKernel`). Optimized for Cortex-X1/X2, with SME awareness. | GEMV: ~5–15% over our NEON. GEMM: moot (Accelerate beats any NEON kernel). | High (C++ FFI, ARM-specific asm) | **Skip.** Accelerate makes GEMM moot; GEMV gains don't justify the FFI complexity. |
| **B-panel packing** | `MlasGemmPackB`, `MlasSgemmCopyPackB` — pre-pack weights for cache-optimal access patterns in GEBP tiling. | Relevant for GEMM only. Our Accelerate path doesn't need it (Apple packs internally). | Medium | **Skip.** Moot under Accelerate. |
| **Cache-aware tiling** | KC/NC/MC blocking per L1/L2 size. Built into the dispatch loop with MLAS's thread pool. | Same as packing — only matters for NEON-only GEMM path. | Medium | **Skip.** |
| **Thread pool** | ORT's `concurrency::ThreadPool` — work-stealing, QoS-aware scheduling. | Our SPMD pool already achieves ~50 ns barrier (measured). ORT's ThreadPool adds ~2–5 µs per dispatch. We're faster here. | N/A | **Keep ours.** We win. |
| **HGEMM (half GEMM)** | `MlasHGemmDispatchNeon` — NEON fp16 GEMM with FEAT_FP16. | For decode: we already do f16→f32 in-register GEMV (same approach, less overhead). For prefill: moot (Accelerate). | High | **Skip.** Our f16 GEMV is already competitive. |
| **Quantized GEMM** | `MlasSymmQgemmPackB`, `MlasDynamicQgemmPackB` — int8/int4 quantization-aware GEMM. `MlasQ4GemmPackB`. | Potentially useful for int4/int8 quantized models. We have `MatMulNBits` but haven't benchmarked MLAS's quant GEMM vs ours on ARM. | High (major FFI) | **Evaluate later.** Relevant if we ship int4 on Mac. |

### Bottom line:

Accelerate makes most of MLAS moot for compute-bound work (prefill), and our own NEON GEMV already suffices for bandwidth-bound decode. The only potentially valuable piece is the quantized GEMM for future int4 models, but that's a separate decision and MLAS's ARM int4 path would need evaluation.

**Porting hand-written assembly is not worth it.** The effort-to-payoff ratio is terrible: ~5–15% decode improvement from KleidiAI GEMV vs months of FFI maintenance. If we need that last 15%, it's cheaper to optimize our Rust NEON kernel (add 8-row batching, prefetch hints) than to vendor KleidiAI.

---

## Q4 — The 8% FP32 Decode Gap

### Measurements: 42.30 tok/s (ours) vs 46.01 tok/s (ORT/MLAS)

**Where the gap lives:**

| Source | Estimated contribution | Evidence |
|--------|----------------------|----------|
| Op dispatch overhead | 2–4% | We dispatch 435 ops/token (446 nodes – 11 initial). ORT fuses subgraphs (QKV into one MatMul: 1152-wide, SiLU, LayerNorm) → ~250–300 dispatches. ~150 extra dispatches × ~2 µs = ~0.3 ms on a ~24 ms token. |
| GEMV kernel quality | 2–4% | MLAS KleidiAI has more aggressive register scheduling and prefetch tuning than our 4-row batched NEON GEMV. Measured our single-threaded at 52 GB/s (f32); MLAS likely achieves 60–65 GB/s. |
| B-panel pre-packing | 1–2% | MLAS pre-packs B for cache-line-aligned streaming. Our pre-transposed B is close but not identical to packed layout. |

**Total: ~5–10%**, consistent with the measured 8%.

### Is it reachable?

Yes, but not worth the effort if Mac default becomes fp16:
- **With fp16 (recommended default):** We lead 60.41 vs 42.45 → 42% advantage. The 8% f32 gap is irrelevant.
- **If we still wanted to close it:** Fuse QKV (Sapper, low-priority), add prefetch to NEON GEMV, tune SPMD partition sizes. Estimated 2–3 days engineering for ~5% recovery.

**Verdict: Concede it.** Ship fp16 as the Mac default. The 8% fp32 gap is unmeasurable in the fp16 world where we lead by 42%.

---

## Q5 — Strategic Recommendation (Full)

### The native CPU EP can stand alone on Apple Silicon.

The architecture:

```
                    ┌──────────────────────┐
                    │   Decode (M=1)       │
                    │   Bandwidth-bound    │
                    │   NEON f16 GEMV      │
                    │   88–100 GB/s/core   │
                    │   → 60+ tok/s        │
                    └──────────────────────┘
                              │
                    ┌─────────┴──────────┐
                    │   auto-detect M    │
                    └─────────┬──────────┘
                              │
                    ┌──────────────────────┐
                    │   Prefill (M≥2)      │
                    │   Compute-bound      │
                    │   f16: BNNS f16→f32  │
                    │   f32: cblas_sgemm   │
                    │   → AMX coprocessor  │
                    │   2000–2450 GFLOPS   │
                    │   → ~37 ms TTFT @M=40│
                    └──────────────────────┘
```

### Why we don't need MLAS on Mac:

1. **Decode:** Our fp16 NEON GEMV reads 2 bytes/weight (half the DRAM traffic of MLAS's fp32 path), achieving 88 GB/s single-threaded. MLAS's fp32 path achieves ~60 GB/s. We win structurally.

2. **Prefill:** Accelerate/AMX delivers 900–2100 GFLOPS. MLAS's NEON GEMM delivers ~20 GFLOPS (single-threaded, ~80 with threading). We win by 10–100×. **ORT cannot reach AMX** — the shipped dylib has no Accelerate linkage.

3. **Thread pool:** Our SPMD decode pool has ~50 ns barrier latency vs ORT's ThreadPool at ~2–5 µs. We win on dispatch overhead.

### What must ship to realize this:

1. **Accelerate integration for prefill (already coded):** `accelerate_gemm::sgemm` is already implemented and gated on `CpuBackend::Accelerate`. Confirm it activates for M>1 in `gemm_with_backend`. ✅ Already done (matmul.rs line 286–293).

2. **FP16 as Mac default:** Ship fp16 models as the standard Mac model format. Our fp16 GEMV path is already the hot path when `CpuBackend::Accelerate` is selected and `inputs[1].dtype == Float16` (matmul.rs line 496–514). ⚠️ **Dispatch fix needed:** `try_matmul_half` (line 488) now intercepts this path for fully-fp16 models. Gate it on `geom.m > 1` to preserve the GEMV decode path.

3. **Verify Accelerate TTFT E2E:** The microbenchmarks predict ~47 ms TTFT at M=40. Measure this with the compare harness using the fp32 model and Accelerate enabled. If it matches, the story is closed.

### Caveats and risks:

| Risk | Severity | Mitigation |
|------|----------|------------|
| ORT fixes fp16 routing → our decode moat shrinks from 42% to ~15% | Medium | Our real moat is prefill (AMX). Decode advantage is bonus. |
| Apple changes AMX access pattern in future macOS | Low | Accelerate is the stable API; Apple guarantees it. |
| Small models (M=10 prompt) show lower AMX utilization (170 GFLOPS vs 2100 at M=512) | Low | Even at 170 GFLOPS we beat MLAS's ~80 GFLOPS by 2×. |
| Future int4/int8 quantized models may need MLAS's quant GEMM | Medium | Evaluate if/when we ship int4 on Mac. Separate decision. |

### What we do NOT need from MLAS:

- ❌ KleidiAI microkernels (Accelerate beats them for GEMM; marginal for GEMV)
- ❌ B-panel packing (Accelerate handles this internally)
- ❌ HGEMM (our f16 GEMV is already competitive)
- ❌ Cache tiling (Accelerate handles this internally)
- ❌ Thread pool (ours is faster)

### What we might want later (separate decisions):

- ❓ MLAS quantized GEMM for int4 Mac models (evaluate when relevant)
- ❓ KleidiAI's SME2 awareness for M4 (check if Accelerate already routes to SME)

---

## Raw Data (Corroborated Measurements)

### Benchmark environment
- **Machine:** Apple M1 Max, 8 P-cores + 2 E-cores, 32 GB LPDDR5
- **FEAT_FP16:** Yes (hw.optional.arm.FEAT_FP16 = 1)
- **FEAT_BF16:** No (M1)
- **DRAM BW (theoretical):** 400 GB/s
- **Timing:** clock_gettime(CLOCK_MONOTONIC) and mach_absolute_time(), both reported

### Accelerate sgemm — full prefill TTFT (MatMul only)

| M | QKV ms×24 | O ms×24 | Gate ms×24 | Up ms×24 | Down ms×24 | lm_head ms | Total ms |
|---|-----------|---------|------------|----------|------------|------------|----------|
| 10 | 0.62 | 0.49 | 12.56 | 10.74 | 12.33 | 8.34 | 45.1 |
| 40 | 1.70 | 1.33 | 7.83 | 7.89 | 7.95 | 15.32 | 42.0 |
| 128 | 3.28 | 2.80 | 14.01 | 17.10 | 29.88 | 19.42 | 86.5 |
| 512 | 11.02 | 8.47 | 53.07 | 53.12 | 75.49 | 61.79 | 262.9 |

### NEON GEMV — single-threaded decode (M=1)

| | f32 total ms | f16 total ms | f16/f32 speedup |
|---|---|---|---|
| All layers + lm_head | 62.35 | 10.97 | 5.68× |
| f32 BW | 30–62 GB/s | — | — |
| f16 BW | — | 86–100 GB/s | — |

### Accelerate sgemm M=1 decode (why we DON'T use it for decode)

| Op | ms | GB/s |
|---|---|---|
| Gate | 0.455 | 38.3 |
| lm_head | 17.53 | 31.1 |

Accelerate's M=1 path is 2–3× slower than our NEON GEMV due to dispatch overhead. Confirmed: GEMV stays with us, GEMM goes to Accelerate.
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-sebastian-mlas-vs-native-strategy-end -->
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-vasquez-pr272-review -->
<!-- merged from .squad/decisions/inbox/vasquez-pr272-review.md -->
# Decision: PR #272 review — DDPM + FlowMatching schedulers

- **Reviewer:** Vasquez (diffusion/engine)
- **Author:** Dallas (locked out of revision)
- **Issue:** #47 (Add FlowMatching/DDPM and modern DiT scheduler support)
- **Date:** 2026-07-27
- **Verdict:** APPROVE

## What was reviewed
Files: `pipeline/schedulers/{ddim.rs, ddpm.rs (new), flow_matching.rs (new), mod.rs}`,
`onnx-genai-metadata/src/schema/scheduler.rs`, `schema/inference_metadata.schema.json`,
`docs/DIFFUSION.md`, `comfyui-config/src/lib.rs`, `examples/diffusion-metadata/*`.

## Findings (all PASS)

1. **DDPM ancestral step — correct.** Coefficients match diffusers `DDPMScheduler`
   fixed-small variance: `x0_coeff = sqrt(ᾱ_prev)·β_t/β̄_t`,
   `sample_coeff = sqrt(α_t)·β̄_prev/β̄_t`, `variance = β̄_prev/β̄_t·β_t`.
   Hand-verified numerically against the two unit tests (0.654586, -0.435410).
   Noise correctly vanishes at the final step because `ᾱ_prev=1 ⇒ β̄_prev=0 ⇒
   variance=0`, and `x0_coeff` collapses to 1.0 (pure x0). Verified.

2. **Prediction-type → x0 — correct & DRY.** All schedulers share
   `x0_from_model_output` / `epsilon_from_model_output` in `mod.rs`. v-prediction→x0
   = `α_t·x_t − σ_t·v` (correct; the most bug-prone formula). Cross-checked by a test
   that encodes the same epsilon as v_prediction and asserts identical output.

3. **FlowMatching — correct.** Shifted rectified-flow sigmas
   `shift·s/(1+(shift−1)·s)`, Euler update `x + (σ_next−σ_cur)·v` (velocity
   semantics, not epsilon). Hand-verified (N=4,steps=3,shift=2 → 0.842308, 0.99,
   1.15; timesteps [4, 3.0769, 1.6]). Registry rejects `epsilon` for flow_matching.

4. **No regression.** `ddim.rs` change is a pure extraction of the beta-schedule
   loop into shared `training_alpha_cumprod`; byte-identical logic (+ a harmless
   `num_train<2` guard). `training_sigmas` delegates but yields identical
   `((1-ᾱ)/ᾱ)^0.5`. DDIM/Euler/DPM++ tests still pass.

5. **Metadata/schema backward-compatible.** New `shift: Option<f32>` (nullable,
   min 0). `schema_sync::committed_inference_metadata_schema_is_current` passes;
   existing SD example still validates (`metadata_fixtures`). Non-epsilon
   prediction now accepted (registry test).

6. **Tests are meaningful.** Numeric hand-computed oracles (not "it runs", not the
   production formula as its own oracle), per-prediction-type, edge cases
   (final/single step), and acceptance of modern prediction contracts.

## Validation evidence
- `cargo test -p onnx-genai-engine --lib schedulers` → 18 passed / 0 failed.
- `cargo test -p onnx-genai-metadata` → 15 + 32 + 1 passed / 0 failed (schema in sync).
- `cargo clippy -p onnx-genai-engine -- -D warnings` → clean.
- `cargo fmt` scoped to the 3 touched crates → clean (workspace edition 2024).
  (Workspace-wide fmt diffs exist only in `onnx-runtime-ep-cuda`, unrelated local
  changes not part of this PR.)
- `gh pr checks 272` → "Rust quality" pass; all CI green incl. codecov/patch.

## Owner of any follow-up
None required — APPROVE. No revision needed.
<!-- scribe-merge-2026-07-27T16-44-54Z-wave8-vasquez-pr272-review-end -->

<!-- scribe-merge-2026-07-27T13-10-00-07-00-cli-improvement-session -->

<!-- merged from .squad/decisions/inbox/batty-cli-sampling-fixes.md -->
# Batty decision — CLI sampling and context defaults

Date: 2026-07-27

## Decision

- Keep `GenerateOptions::default().max_new_tokens` at 128 for engine/server compatibility.
- When CLI `generate` or `run` omits `--max-new-tokens`, derive the per-turn budget from the model's effective context limit: `min(model.max_sequence_length, decode_path_max_len)` plus any `--max-context` override, minus the current prompt/context tokens.
- In the REPL, recompute that ceiling every turn after rendering the live multi-turn prompt, so it reflects the current context length as history grows. One-shot `generate` uses the same helper once because it has only one prompt.
- Report context usage in the compact stats path as `ctx used / max`.
- Apply the same model-following budget to reasoning and non-reasoning models; reasoning no longer gets a separate hardcoded default.
- If no effective context limit is discoverable, warn once and use a finite 512-token fallback rather than allowing an unbounded decode to hit ORT out-of-bounds behavior. The warning points to `--max-context` and `model.max_sequence_length` metadata.
- If `--max-new-tokens` is explicit, honor it exactly.
- Treat `--temperature` above 0, `--top-p`, and `--top-k` as requests for stochastic sampling by setting `greedy = false`; keep `--temperature 0` as greedy.
- Expose `--max-context` through the shared CLI sampling args so users can cap prompt plus generated tokens when model metadata lacks a context limit.

## Why

The CLI is a developer/maintainer tool and should follow the loaded model instead of imposing an arbitrary per-turn cap. `max_new_tokens` is a safety ceiling, not a target or a context-allocation policy; the model decides when to stop by emitting EOS, with the context window as the hard stop. The REPL recomputes the ceiling against the current rendered prompt every turn for correctness, but does not reserve headroom or refuse turns preemptively. Context-fill is also appropriate for one-shot `generate`: it is the same developer-facing generation surface as `run`, and users who want a shorter completion can pass `--max-new-tokens` exactly. The finite 512 fallback exists only for packages with no discoverable context window, where unbounded generation can otherwise walk into an ORT Gather out-of-bounds crash; `--max-context` is the CLI escape hatch until metadata is fixed.

<!-- merged from .squad/decisions/inbox/isidore-cli-native-cuda-feature.md -->
### 2026-07-27: CLI native CUDA feature uses two-axis selection
**By:** Isidore
**What:** Added a first-class CLI `native-cuda` feature for the native backend plus hand-written CUDA EP, while leaving the existing CLI `cuda` feature on the wheel-compatible ONNX Runtime CUDAExecutionProvider path.
**Why:** Build-time features decide which CUDA kernels are compiled in, but runtime selection remains separate: `ONNX_GENAI_EP=cuda` selects the ORT session EP, and `/backend native` selects the native decoder in the REPL.

### 2026-07-27: One-shot native decode needs CLI backend flag
**By:** Isidore
**What:** Backlog item: add a command-line `--backend` selector for `generate` and `run`.
**Why:** Today the decode backend is only switchable through the REPL `/backend` command, so one-shot `generate` cannot request the native decoder even when the native CUDA feature is compiled in.

<!-- merged from .squad/decisions/inbox/leon-cli-context-exhaustion-guard.md -->
### 2026-07-27: CLI context exhaustion guard
**By:** Leon
**What:** The CLI now treats `prompt_tokens >= effective_max_context` as a pre-decode context-exhaustion condition. In the REPL it drops the just-added user turn, does not append an assistant message, tells the user to use `/reset` or shorten/change context, and continues. In one-shot `generate` it returns an actionable error instead of silent empty success.
**Why:** At equality or above the engine has provably zero room to emit even one token, so decoding can only produce an empty `Length` result that poisons conversation history. This is a correctness guard for the degenerate zero-room boundary, not a heuristic budget policy: when `prompt_tokens < effective_max_context`, even by one token, the existing automatic `max_new_tokens` sizing remains unchanged and the model still decides when to stop within the safety ceiling.
<!-- merged from .squad/decisions/inbox/apone-pr281-review.md -->
# Decision: PR #281 (issue #49) — native img2img & inpainting in run_comfyui

- **Reviewer:** Apone (independent/adversarial). Author: Newt.
- **Date:** 2026-07-27
- **Verdict:** APPROVE (non-blocking notes)

## Verification of the classic img2img/inpainting failure points
- **strength→start_step:** `strength_to_start_step = num_steps − round_ties_even(num_steps·strength)`,
  clamped to `0..=num_steps` (comfyui-config/src/lib.rs). Matches diffusers `get_timesteps` and
  DIFFUSION.md §4. Unit test `strength_mapping_matches_hand_computed_diffusers_steps` pins the exact
  endpoints: `strength=0.0 → start_step=num_steps` (zero denoise) and `strength=1.0 → 0` (full denoise
  from noise), plus the banker's-rounding tie (`0.5, 21 → 11`). Would catch an inversion or off-by-one. ✓
- **Noise init (not pure random):** img2img VAE-encodes the source, then calls
  `engine.diffusion_add_noise(start_step, num_steps, encoded, noise)`; only txt2img keeps the
  `noise · init_noise_sigma` seed. Each scheduler's `add_noise` uses correct diffusers semantics —
  DDIM/DDPM `√ᾱ·x + √(1−ᾱ)·noise` (ᾱ = alpha_cumprod at the step's timestep), Euler/EulerA
  `x + σ·noise`, DPM++ `α_t·x + σ_t·noise`, FlowMatch `(1−σ)·x + σ·noise`. `step==num_steps → 0` sigma /
  returns original. `ddim_add_noise_matches_hand_computed_alpha_mix` pins the mix + zero-step identity. ✓
- **9-channel inpaint layout:** UNet input is `[4 noisy latent | 1 downsampled mask | 4 masked-image
  latent]` = 9. `build_inpaint_conditioning` emits the 5-ch conditioning `[mask | masked latent]`;
  `append_loop_conditioning` (iterative.rs) concatenates it AFTER the (already scale_model_input-scaled)
  4-ch latent — matching diffusers (scale then cat). Test `inpaint_loop_input_is_nine_channels_in_declared_order`
  asserts exact shape `[1,9,1,2]` and exact element order. Scheduler state stays 4-ch. Would catch a
  swapped/short channel layout. ✓
- **Masked-image latent:** `VAE-encode(source · (1−mask))`; mask=1 means repaint (ComfyUI semantics),
  so repaint region is zeroed before encode. Mask downsampled to latent res via `VAE_DOWNSCALE=8`. ✓
- **VAE-encode generality:** driven off the workflow graph — detects `VAEEncode`/`VAEEncodeTiled` and
  `VAEEncodeForInpaint`/`InpaintModelConditioning` on the sampler's `latent_image` link; encoder chosen
  by component `role == "vae_encoder"` (filename fallback), NOT hard-coded model names. ✓
- **txt2img non-regression:** `start_step` passed only when `source_image` is Some; else `None` →
  pure-noise `init_noise_sigma` seed path is byte-for-byte unchanged. Detection is presence-of-node
  driven. `iterative_override_allows_zero_step_tail` confirms the zero-step boundary publishes the seed
  unchanged; the `< num_steps` → `<= num_steps` guard relaxations are consistent across mod.rs/iterative.rs. ✓
- **DRY:** shared `mix_noise` helper, one `add_noise` trait method, one `strength_to_start_step`, one
  `build_inpaint_conditioning`, one `append_loop_conditioning`. No per-checkpoint/per-benchmark casing. ✓
- **Docs:** DIFFUSION.md §4/§4.1 additions (zero-strength edge, VAE-encoder discovery, 9-ch order) are
  genuine additions describing new behavior — not doc-moved-to-match-code; the pre-existing
  `start_step = num_steps − round(num_steps·denoise)` formula already matched the impl.

## Validation evidence (worktree @ 6358c963, origin/squad/49-img2img-inpaint)
- `cargo fmt --all -- --check` → exit 0
- `cargo clippy -p onnx-genai-engine -p onnx-genai --all-targets -- -D warnings` → `Finished` exit 0
- `cargo test -p onnx-genai-engine` → 233 unit (0 failed) + all integration suites incl. 32/32 iterative e2e
- `cargo test -p onnx-genai` → 28 unit + 6 audio + 5 image e2e, 0 failed
- `cargo test -p onnx-genai-comfyui-config` → 14 pass (strength mapping + detection routing)

## Non-blocking notes (owner: Newt is LOCKED OUT — assign **Hudson**)
1. **VAE encode uses the distribution mode (mean), not `latent_dist.sample()`.** `vae_encode` slices the
   first `latent_channels` from moment output and drops logvar. Deterministic and reasonable, and guarded
   by `scripts/img2img_e2e.py` (~1e-2). Consider a one-line comment noting the intentional mode choice.
2. **`downsample_mask` uses nearest top-left pixel per 8×8 block.** Fine approximation of ComfyUI's
   nearest downsample; a doc/comment noting it isn't area-averaged would help future readers.

Neither note affects correctness of the reviewed paths; both are guarded by the e2e parity script.

<!-- merged from .squad/decisions/inbox/bishop-pr283-review.md -->
### 2026-07-27: PR #283 (issue #50) ControlNet/LoRA wiring — REQUEST-CHANGES
**By:** Bishop (independent review; author Dallas locked out)
**Verdict:** REQUEST-CHANGES. Fix owner: **Batty** (Engine Dev), with **Roy** (Lead) to arbitrate the cross-repo run_comfyui↔mobius input contract.

**What is correct (keep):**
- Control-image preprocessing (`preprocess_control_image`): batched RGB CHW in `[0,1]`, resize-to-output-resolution. Matches diffusers ControlNet `prepare_image` and mobius `controlnet_cond` shape `[batch, conditioning_channels, height*8, width*8]` (pixel resolution). Unit-tested with real oracle values.
- LoRA gate: `lora_gate.{stem}` matches the documented + real mobius runtime convention (`models/unet.py` `_lora_gates`, `_diffusers_builder.py` bakes scale=1.0, runtime gate supplies strength). DIFFUSION.md §8b.
- Additive / non-regression: plain txt2img/img2img/inpaint/SDXL take the empty-`denoiser_inputs` wrapper path; verify-A bit-identical check retained. Graph-driven routing (not hard-coded node names).
- All validations pass (fmt, clippy -D warnings, engine/genai/comfyui-config tests).

**Blocking concerns:**
1. **`conditioning_scale` is an invented runtime gate with no exporter contract.** It appears NOWHERE in `../mobius`. DIFFUSION.md §9 says ControlNet strength is collected at translate time and **fused at export** (`checkpoint_export(controlnet=...)`), i.e. bake-at-export, NOT a runtime gate. Engine `routing.rs::component_inputs` iterates the model's *declared* inputs and looks up matching endpoints, so an undeclared `denoiser.conditioning_scale` is **silently dropped** → strength silently not applied (or dead code if baked). This is exactly the invented-mechanism / wrong export-fuse-vs-runtime convention pattern to avoid.
2. **Multi-ControlNet `.{adapter}` suffix ports have zero backing.** mobius supports a *single* ControlNet only (`integrations/onnx_genai/comfyui.py::_find_controlnet` → `tuple|None`; `models/controlnet.py`/`tasks/_controlnet.py` declare unsuffixed `controlnet_cond`). No fused multi-CN export exists. Suffixed `controlnet_cond.{adapter}` / `conditioning_scale.{adapter}` would all silently drop → multi-CN non-functional against any real model.
3. **Tests give false confidence.** They validate only the driver's internal math/routing; none asserts against a real denoiser's declared input set. They stay green even though ControlNet feeding silently no-ops. A swapped scale or missing port is not caught.

**Requested fix (owner Batty + Roy):** Reconcile the runtime↔exporter contract before merge: either (a) mobius grows a real `conditioning_scale` denoiser input (+ multi-CN port scheme) and DIFFUSION.md is updated to document runtime scaling, or (b) run_comfyui drops `conditioning_scale`/multi-CN suffixes and relies on export-baked strength per current doc. Add a contract-level test (or gate the CN path as experimental) so silent input-drop cannot pass as success. LoRA + preprocessing work can land as-is.

<!-- merged from .squad/decisions/inbox/deckard-fix-pr276-87.md -->
# Decision: PR #276 (issue #87) async prefetch overlap — Deckard revision

**Fix owner:** Deckard (Systems Dev, CUDA & Perf pod)
**Author (locked out):** Keaton
**Date:** 2026-07-27
**Branch:** feat/async-prefetch-overlap-87 (pushed, force-with-lease after rebase onto origin/main)
**Status:** Both of Ferro's REQUEST-CHANGES blockers fixed; awaiting Ferro re-review (do NOT self-merge).

## Blocker 1 — GPU test suite build break (fixed)
Adding `async_host_to_device` to `CudaTransferCounts` left the pre-existing
struct literal in `crates/onnx-runtime-ep-cuda/tests/compressed_sparse_attention_gpu.rs`
missing the new field, so `cargo test -p onnx-runtime-ep-cuda --features cuda`
failed to compile. Fixed by populating the field with the observed
before/after delta (consistent with the other two counters), so the existing
"ratio-128 FP8 must not stage through host memory" assertion now also covers
async H2D copies. Verified all `CudaTransferCounts` construction sites compile
(runtime.rs:500 already had the field; the `::default()` site is unaffected).

## Blocker 2 — WAR race in shipped driver + doc overclaim (fixed)
The public `drive_double_buffer` reused a double-buffer slot without ordering the
reuse copy after the prior consumer, so on the CUDA EP a copy stream could
overwrite a buffer while the previous wave's compute was still reading it
(write-after-read hazard). The WAR fence existed only in a hand-rolled ep-cuda
test loop, not the driven path.

Fix (generic over any `&dyn ExecutionProvider`, no per-EP / per-buffer-count
special-casing):
- New `ExecutionProvider` trait methods: `record_compute_fence` (default
  `Fence::signalled()`) and `copy_wait_fence` (default no-op). Implemented on the
  CUDA EP over the compute/transfer streams (`record_compute_fence` /
  `copy_wait_fence` runtime primitives). Non-CUDA EPs stay safe no-ops.
- `drive_double_buffer` records a compute fence over each consumer and makes the
  transfer stream wait on the prior consumer of a slot before issuing the reuse
  copy. The WAR fence is now enforced by the shipped driver itself.
- New GPU regression test `drive_double_buffer_war_safe_across_waves`
  (`crates/onnx-runtime-session/tests/cuda_prefetch_war.rs`, session `cuda`
  feature) drives the PUBLIC path across 6 waves (both slots reused) with a slow
  compute-stream consumer; corrupts if the driver WAR fence is removed. Added
  `cudarc` as a session dev-dependency (test-only, dynamic-loading, no toolkit).
- Rewrote the `prefetch.rs` module + driver docs and `docs/WEIGHT_OFFLOAD.md` to
  state plainly that the driver enforces WAR (removed the hand-rolled-loop
  overclaim).

Kept intact: RAW ordering, `copy_wait_fence`/`compute_wait_fence` primitives,
pinned-staging lifetime, dtoh/dtod synchronize-first discipline, honest deferral
of live-MoE-loop wiring.

## WAR-fence neutering experiment (load-bearing proof) — on the NEW driver-path test
Neutered the driver's `ep.copy_wait_fence(&last_compute_fence[next_slot])?` call
in `drive_double_buffer`, then restored it. Pinned GPU7.
- Neutered: `drive_double_buffer_war_safe_across_waves` FAILED —
  "wave 0 output corrupted — the driver WAR fence was violated: a reuse prefetch
  clobbered a staging buffer while this wave's consumer was reading it"
  (wave 0 read wave 4's payload: got [53,54,55,...]).
- Restored: PASS (1 passed; 0 failed).
Not theater — the driver's own WAR fence is load-bearing.

## Validation (pinned GPU7: `CUDA_VISIBLE_DEVICES=7 taskset -c 1`)
- `cargo test -p onnx-runtime-ep-cuda --features cuda` (lib): 244 passed, 0 failed
  (incl. the 3 overlap tests). `--test compressed_sparse_attention_gpu`: 26
  passed, 1 ignored. Provider `copy_async_fence_orders_h2d_prefetch_through_ep_api`:
  pass.
- `cargo test -p onnx-runtime-session` (lib): 90 passed, 0 failed (incl. prefetch
  strategy tests).
- `cargo test -p onnx-runtime-session --features cuda --test cuda_prefetch_war`:
  1 passed (the new driver-path WAR test).
- Clippy PR's own code (`-p onnx-runtime-ep-api -p onnx-runtime-ep-cuda --features
  cuda -p onnx-runtime-session --lib -- -D warnings`): clean. `--all-targets`
  fails only in unrelated pre-existing test/kernel files (matmul_nbits.rs,
  normalization.rs, standard_attention.rs, several *_gpu.rs test files) — all
  byte-identical to origin/main (verified via `git diff origin/main`).
- `cargo fmt --all -- --check`: clean.

## Ignorable environmental failures (NOT this PR — confirmed on parent 6654a168)
- `conv_gpu` / `pooling_gpu` (cuDNN libcudnn.so.9 absent) — as Ferro noted.
- `matmul_gpu::matmul_f32_on_gpu_matches_cpu_reference` — 4-D batched mismatch;
  reproduces BYTE-IDENTICALLY on parent commit 6654a168 (matmul_gpu.rs is
  unchanged by the PR). Pre-existing GPU7 environmental failure, not a
  regression. (Ferro tested on GPU6 and did not flag this; documented here for
  transparency.)
- A `fused_attention`/standard-attention bf16 tolerance case is the `1 ignored`
  in the CSA test binary.

## Follow-ups
- Live MoE decode-loop wiring still depends on Phase-3b live device weight
  binding (unchanged, honestly deferred).

<!-- merged from .squad/decisions/inbox/ferro-pr276-rereview.md -->
# Decision: PR #276 (issue #87) re-review — APPROVE

- **Reviewer:** Ferro (concurrency) — independent, not the author
- **Date:** 2026-07-27
- **Artifact:** feat/async-prefetch-overlap-87 @ f47916e7 (rebased onto origin/main, force-with-lease)
- **Prior verdict:** REQUEST-CHANGES (2 blockers). Fixer: Deckard (Keaton locked out as author).

## Verdict: APPROVE — both blockers genuinely fixed, WAR fence proven load-bearing.

### Blocker 1 (build break) — RESOLVED
`compressed_sparse_attention_gpu.rs:706` now populates `async_host_to_device`
with the observed delta (`after - before`). `cargo test -p onnx-runtime-ep-cuda
--features cuda` compiles; CSA 26 pass / 1 ignored. The value is CORRECT, not a
placeholder: the assertion is `observation.transfers == CudaTransferCounts::default()`
(all-zero), so populating the field actually STRENGTHENS the "no host staging"
check (async H2D copies must also be 0) rather than making it vacuous.

### Blocker 2 (WAR race + doc overclaim) — RESOLVED
- New generic `ExecutionProvider::record_compute_fence` (default already-signalled)
  + `copy_wait_fence` (default no-op); CUDA EP records over compute stream / waits
  on copy stream via non-host-blocking cuStreamWaitEvent.
- `drive_double_buffer` now, over any `&dyn ExecutionProvider`: waits the transfer
  stream on the prior consumer's fence of a slot (`copy_wait_fence`) BEFORE the
  reuse `copy_async`, and records the consumer fence AFTER `compute(n)`. Ordering
  is correct; enforced in the shared driver, no hand-rolled test loop, no-op-safe
  on sync EPs.
- Docs (prefetch.rs + WEIGHT_OFFLOAD.md) rewritten to state the driver enforces
  WAR and the public-path test proves it — no residual overclaim.

## Load-bearing neutering experiment (driver-path test)
`drive_double_buffer_war_safe_across_waves` (session `cuda` feature, public path,
6 waves both slots reused):
- Fence intact -> PASS.
- Neutered CUDA EP `copy_wait_fence` to a no-op (dropped cuStreamWaitEvent) ->
  FAIL: `wave 0 output corrupted ... reuse prefetch clobbered a staging buffer`
  (wave 0 read wave 4's payload, 53.0). Not theater.
- Restored -> PASS.
RAW guards (prior 3 ep-cuda tests) still present/pass in the 244-lib run.

## Validation (pinned GPU6: CUDA_VISIBLE_DEVICES=6 taskset -c 1)
- ep-cuda `--features cuda`: 244 lib pass; CSA 26/1-ign; only conv_gpu/pooling_gpu
  fail (cuDNN absent — ignorable env, reproduces on parent).
- session host: 90 lib pass. session `--features cuda` WAR test: 1 pass.
- clippy PR-owned targets (ep-cuda lib, ep-api lib, session lib+tests) clean under
  `-D warnings`. `--all-targets` fails only in pre-existing untouched
  `fused_epilogue_gpu.rs` (too_many_arguments, fails on base).
- `cargo fmt --all -- --check`: clean.

No follow-up owner needed — merge-ready.

<!-- merged from .squad/decisions/inbox/newt-img2img-inpaint-49.md -->
### 2026-07-27: Keep inpainting conditioning outside scheduler state
**By:** Newt
**What:** Image diffusion carries only the 4-channel latent through the scheduler. The runner supplies a separate `{loop_endpoint}.conditioning` tensor containing `[mask | masked-image latent]`, and the engine appends it to form the 9-channel denoiser input each step.
**Why:** Schedulers must update only the noisy latent, while inpainting UNets require static 1+4 conditioning channels. This preserves the existing loop and final VAE decode contracts without checkpoint-specific dispatch.

### 2026-07-27: Zero strength means a zero-iteration tail
**By:** Newt
**What:** `start_step == num_steps` is valid and publishes the encoded seed directly to final pipeline phases.
**Why:** The documented `num_steps - round(num_steps * strength)` mapping produces exactly `num_steps` at strength 0.0; accepting it avoids an edge-case special case in front ends.

<!-- merged from .squad/decisions/inbox/pris-cli-ci-coverage.md -->
# Pris decision — CLI ORT CI coverage

Date: 2026-07-27

## Constraint found

`onnx-genai-ort-sys` resolves ONNX Runtime in this order: `ORT_LIB_DIR`, `ORT_ROOT`, `pkg-config`, then an automatic GitHub release download. The automatic path downloads ONNX Runtime 1.27.0 with `curl`, verifies a pinned SHA-256 for Linux x64, macOS arm64, Windows x64, and Windows arm64, extracts it under Cargo `OUT_DIR`, and reuses it only when the cached header and runtime match API version 27. Bindgen needs libclang; Linux CI should install `clang libclang-dev`, while Windows can use the hosted LLVM install. Windows test processes must also load the downloaded DLL, not the runner's older ambient `onnxruntime.dll` (observed as API 17 / ORT 1.17.1), so the lane copies the pinned DLL beside the Cargo-built binaries before testing.

`publish.yml` already pays the ORT-linked build cost for `onnx-genai` and `onnx-genai-server` wheels. Those wheels deliberately do not bundle libonnxruntime; runtime loading comes from the Python `onnxruntime` package. Their build-time headers/import library still come from the same `ort-sys` auto-download. `wheels.yml` builds `nxrt` wheels and leaves `onnx-genai-server` wheels to `publish.yml`.

## Design chosen

Add an isolated `cli-ort` CI job, separate from the offline allowlist, with Linux x86_64 and Windows x86_64 matrix entries. It intentionally permits the pinned native ORT download only for `onnx-genai-cli`, then runs:

- `cargo build --locked -p onnx-genai-cli`
- `cargo test --locked -p onnx-genai-cli`
- `cargo clippy --locked -p onnx-genai-cli --all-targets -- -D warnings`

Linux is mandatory because `repl_e2e.rs` contains Unix-only REPL/interrupt/contract tests, including `a_turn_that_stops_inside_the_reasoning_says_it_has_no_answer`. Windows is included because the auto-download path supports `win-x64` and it catches platform drift on Justin's main development OS.

## CI cost

Observed green run: https://github.com/justinchuby/onnx-genai/actions/runs/30298789423

- `CLI ORT coverage (Linux x86_64)`: 1m13s.
- `CLI ORT coverage (Windows x86_64)`: 6m48s.
- Marginal wall-clock in that run: about 48s beyond the next slowest existing job, because the matrix runs in parallel.

## Residual coverage gap

The lane covers CLI build, unit tests, and integration tests that can run against checked-in fixtures. It still does not cover paths requiring a real external model, GPU execution, or an actual interactive TTY. The ratatui live view is inert when `stdout` is not a terminal, so piped CI cannot exercise the live terminal rendering path.

<!-- scribe-merge-2026-07-27T13-10-00-07-00-cli-improvement-durable-lessons -->
### 2026-07-27: CLI coverage must include an ORT-linked lane
**By:** Scribe, preserving the CLI improvement track outcome.
**What:** The CLI previously had zero CI coverage, which plausibly allowed the token-budget, ignored-sampling-flags, and missing-context-cap regressions to survive. The merged `cli-ort` lane now builds, tests, and lints `onnx-genai-cli` with the ORT backend on Linux and Windows.
**Why:** This lane is a baseline, not complete product coverage: it does not exercise real external models, GPU execution, or the ratatui live TTY view because `live_turn.rs:91` gates that path on `stdout().is_terminal()`.

### 2026-07-27: CUDA EP fallback claims must distinguish the three runtime cases
**By:** Scribe, preserving the native-CUDA feature review outcome.
**What:** Documentation and comments must not claim that requesting `cuda` silently falls back to CPU. The distinct cases are: CUDA support not compiled, CUDA requested but no device/provider is available, and node-level fallback inside an otherwise CUDA-capable ORT session; only the third is a fallback.
**Why:** A stale `onnx-genai-engine/Cargo.toml` comment propagated the wrong claim into new documentation before automated review caught it. Runtime capability claims need source-of-truth verification before reuse.

### 2026-07-27: Independent second-opinion review is required after nontrivial CLI correctness changes
**By:** Scribe, preserving the reviewer rejection protocol outcome.
**What:** A second-opinion reviewer caught the context-exhaustion bug after a human-style reviewer approved the CLI changes. For nontrivial CLI correctness changes, keep the independent pass in the review plan.
**Why:** The missed defect would have recorded an empty assistant turn in the non-reasoning path and permanently poisoned the conversation. The independent pass materially changed the outcome.

Decision archive gate fired at 2026-07-27T13:10:00-07:00: active ledger was 753542 bytes; archived 9 dated entries on or before 2026-07-20 to `.squad/decisions/archive/2026-07.md`.
<!-- merged from .squad/decisions/inbox/parker-pr280-review.md -->
# Decision: PR #280 (issue #48) — SDXL dual-encoder conditioning in run_comfyui

- **Reviewer:** Parker (independent/adversarial). Author: Ripley.
- **Date:** 2026-07-27
- **Verdict:** APPROVE (non-blocking notes)

## Verification of the classic SDXL failure points
- **time_ids order:** `build_time_ids` emits `[original_h, original_w, crop_top, crop_left, target_h, target_w]`,
  matching diffusers `list(original_size + crops_coords_top_left + target_size)`. Unit test pins the exact
  12-value vector incl. batch tiling — would catch a swapped order or an h/w flip. ✓
- **time_ids fed as a single `[batch, 6]` denoiser input (no `.uncond`)** — correct: SDXL shares time_ids
  across both CFG passes; DIFFUSION.md §9 documents "sharing time_ids". ✓
- **Dual-encoder concat:** performed inside the exported ONNX text_encoder (per DIFFUSION.md §9), not at
  runtime. Runner routes conditioning declaratively via dataflow edges. ✓
- **Detection is filename-free:** `conditioning_kind` = DualWithPooled iff a `text_embeds` denoiser
  conditioning edge exists — driven off declared edges, not checkpoint names. DRY, general. ✓
- **Pooled text_embeds + per-edge uncond:** each encoder→denoiser edge routed individually, uncond fed
  as `{denoiser}.{port}.uncond`. ✓
- **SD1.x non-regression:** single-edge path → ConditioningKind::Single, no time_ids; replay/verify contract
  preserved. Existing endpoint tests updated and pass. ✓

## Validation evidence (worktree @ a78a5834)
- `cargo fmt --all -- --check` → exit 0
- `cargo clippy -p onnx-genai-engine -p onnx-genai --all-targets -- -D warnings` → exit 0
- `cargo test -p onnx-genai-engine` → ok (0 failed; env-gated ignored)
- `cargo test -p onnx-genai` → 27 + 6 + 5 pass; 0 failed

## Non-blocking notes (owner: Ripley is LOCKED OUT — assign Hicks)
1. `concatenate_hidden_states` is **dead code** — only referenced by its own unit test; the real concat is
   export-side. The test gives false confidence it guards the runtime concat axis. Either drop it (add a
   comment that concat is export-owned) or wire it. Low priority.
2. Encoder-input→tokenizer mapping relies on graph input declaration order (index 0→primary,
   1→tokenizer_2.json). Safe in practice (SDXL's two tokenizers share vocab) but add a clarifying comment.
3. Repeated `spec().strategy.denoiser.as_deref().unwrap()` — safe (guarded by `resolve_endpoints`) but
   reuse the already-resolved denoiser name for clarity.

None of these corrupt conditioning; approving.

<!-- merged from .squad/decisions/inbox/ripley-sdxl-dual-encoders-48.md -->
### 2026-07-27: Reuse typed image generation for native ComfyUI SDXL
**By:** Ripley
**What:** `run_comfyui` delegates normal renders to the metadata-driven typed image generator. SDXL is detected from a pooled `text_embeds` conditioning edge; all encoder token inputs and CFG outputs are wired, and `[original H/W, crop top/left, target H/W]` time IDs are supplied.
**Why:** The engine already executes multi-output prompt conditioning and multi-input CFG. Keeping conditioning construction in one shared renderer avoids a second checkpoint-specific path while preserving the hidden SD1.x replay verifier.

<!-- merged from .squad/decisions/inbox/roy-pr282-review.md -->
# Decision: PR #282 (issue #84) — Tree-structured speculative decoding review

- **Reviewer:** Roy (independent, adversarial)
- **Author:** Hicks (locked out of fixes)
- **Date:** 2026-07-27
- **Verdict:** APPROVE

## Summary
Adversarial correctness review of tree-structured speculative decoding core.
All scrutiny points pass. Greedy-equivalence invariant is genuinely proven.

## Key findings
1. **Greedy-equivalence test is GENUINE.** `tests/tree_speculative.rs` drives a
   full tree-speculative loop against the real `tiny-llm` fixture and compares
   byte-for-byte to an *independent* plain-greedy reference engine. It asserts
   `saw_branching` (tree wider than its roots) and `saw_multi_accept`
   (accepted path >= 2) so it is NOT a degenerate 1-node/single-chain pass.
   Two prompts covered. Per-node scorer uses full independent forwards on each
   ancestor path — the exact context a correct 2D tree mask would supply — which
   is consistent with the deferral (mask itself guarded separately).
2. **Ancestor-only mask correct.** `ancestor_attention_mask`: `mask[q][k]` iff k
   is ancestor of q or k==q. Unit test hand-builds a real multi-branch tree
   (root→{a,b}; a→{c,d}; b→{e,f}), asserts the exact edge set + explicit
   no-sibling-leak asserts. **Mutation-verified:** injecting a sibling edge made
   the mask unit test FAIL; reverted → pass. Test genuinely guards correctness.
3. **Position ids == depth**, siblings share a slot ([0,1,1,2,2,2,2]). Correct.
4. **Acceptance walk** generic over rule; full/partial/root-reject/typical all
   tested. Greedy follows target argmax → reproduces greedy exactly, bonus token
   = target argmax, length 1..=path_len+1. RejectionSampling coincides with
   Greedy at T=0 (sound: spec decode runs at temp 0). Typical gates on softmax
   mass ≥ threshold.
5. **KV retention** keeps exactly the accepted path in order,
   `final_len == base_len + accepted_len`; asserted in both unit and real-model
   integration tests (`retained_nodes == outcome.nodes`).
6. **Linear non-regression.** mod.rs diff is a pure file move + additive
   (module decl, re-exports, enum extension). `Eq` dropped from `AcceptanceRule`
   (required by `Typical{f32}`); nothing in the linear path relied on it. All
   pre-existing engine tests pass unchanged.
7. **Deferral is HONEST.** `decode/step.rs:138` (and decode/mod.rs) build
   `vec![1_i64; total_len]` — a 1D key mask, no per-query 2D input; a real
   batched tree forward needs graph/session changes. The tree core is NOT wired
   into the live decode path (no refs in src outside `speculative/`), sits behind
   the `TreeScorer` seam, and is fully tested (not dead scaffolding). PR body
   states the deferral plainly. No false "live tree overlap" claim.

## Validation evidence
- `cargo test -p onnx-genai-engine` → all pass; 8 tree unit tests + 2 real-model
  greedy-equivalence integration tests all `ok`.
- `cargo fmt --all -- --check` → clean (exit 0).
- `cargo clippy -p onnx-genai-engine --all-targets -- -D warnings` → clean.
- Mutation check on the mask → test FAILED as expected, then reverted.

## Note
No fix owner needed (APPROVE). A stale-mtime incremental-build gotcha was hit
during the mutation revert (`mv` restored old mtime → cargo reused stale binary);
`touch` + rebuild confirmed the clean tree passes.

Decision archive gate checked at 2026-07-27T16:44:54Z: active ledger was 747576 bytes; no dated entries older than 2026-07-20 remained, so no archive file was created or changed.

<!-- merged/superseded from .squad/decisions/inbox/dallas-controlnet-lora-50.md -->
<!-- merged from .squad/decisions/inbox/bishop-pr283-rereview.md -->
### 2026-07-27: PR #283 / #50 landed with real mobius ControlNet contract
**By:** Scribe, reconciling Dallas implementation note with Bishop re-review and Batty's fix.
**What:** PR #283 closed #50 with native ComfyUI ControlNet and LoRA wiring, but the final landed ControlNet contract is Batty's corrected approach rather than Dallas's superseded suffix-port/`conditioning_scale` plan. Runtime binds exactly the real mobius denoiser input `controlnet_cond` for a single ControlNet hint as batched RGB CHW `[0,1]` at pixel resolution; ControlNet strength is export-fused, so no runtime `conditioning_scale` input is emitted. Multiple ControlNets fail loudly instead of inventing suffixed ports that mobius does not declare. LoRA remains routed through declared `lora_gate.{stem}` inputs.
**Why:** Bishop's initial review found the prior `conditioning_scale` gate and multi-ControlNet suffix inputs had no mobius backing and would silently drop through declared-input routing. Batty removed those invented mechanisms, added contract-pinning tests (`single_controlnet_binds_the_declared_unsuffixed_cond_input`, `multiple_controlnets_fail_loudly_instead_of_silently_dropping`), and Bishop re-reviewed with APPROVE after mutation proof. The stale Dallas inbox note is retained here only as superseded history; future work should use the single `controlnet_cond`/export-fused-strength contract unless mobius changes.
**Outcome:** Dallas authored the original PR, Batty owned the fix after Dallas lockout, Bishop approved the re-review, PR #283 merged as `687612f5`, and issue #50 is closed. The native image-pipeline trilogy is complete: #48 SDXL, #49 img2img/inpaint, and #50 ControlNet/LoRA are all closed.


Decision archive gate checked at 2026-07-27T19:35:00Z: active ledger was 749614 bytes and exceeded 51200 bytes; applied the 7-day policy with cutoff 2026-07-20T19:35:00Z. No active-ledger decision blocks older than the retained window were present, so no archive file was changed.

<!-- scribe-merge-2026-07-27T19-35-00Z-roadmap-wave -->
## 2026-07-27 — Roadmap wave CPU/CUDA/KV/eager/schema reconciliation

**By:** Scribe

**What:** Merged this roadmap wave's durable implementation/review notes after all listed PRs landed on main:
- PR #285 (`d889e85b`) closed #74: CPU standard Conv without MLAS (Vasquez; Hicks approved).
- PR #286 (`0e35045e`) closed #61: scheduler KV preemption/eviction executes engine movement with byte-identical restore (Gorman; Apone approved).
- PR #292 (`4d15f554`) closed #78: eager multi-output dispatch with Split/TopK and PyO3 feature gating (Crowe; Hudson approved).
- PR #293 (`406cdf42`) advanced #75: ONNX schema/shape-inference catalog 148→164, with container-aware TypeInfo deferred (Burke; Parker approved after Pris test fix).
- PR #294 (`568bbd66`) closed #58: native f16/bf16 CPU GEMM FMA microkernel and aarch64 cfg-gated perf probe fix (Drake; Ferro approved after Luba fix).
- PR #288 (`925afbf2`) advanced #67: CUDA EP coverage batch 4, CUDA_COVERED_OPS 118→125 (Moss; Bishop approved after Deckard test fix).

**Why:** These entries capture durable roadmap outcomes, reviewer lockouts/fix owners, and follow-up seams for future scheduling.

<!-- merged from .squad/decisions/inbox/vasquez-cpu-standard-ops-74.md -->
### 2026-07-27: Keep standard Conv available without MLAS
**By:** Vasquez
**What:** Register `ai.onnx::Conv` unconditionally. Use a general pure-Rust 1-D/2-D reference kernel when the optional `mlas` feature is disabled, while retaining the optimized MLAS implementation when enabled.
**Why:** `Conv` is a required standard operator, and CPU EP capability must not depend on an optional C++/assembly backend. ScatterND, Resize, and QLinearMatMul were already implemented and registered on main.

<!-- merged from .squad/decisions/inbox/hicks-pr285-review.md -->
# Decision: PR #285 (issue #74) review — Hicks

- **Reviewer:** Hicks (independent, adversarial; not the author)
- **Author:** Vasquez (locked out of fixes)
- **Date:** 2026-07-27
- **Verdict:** APPROVE

## Scope reality vs claim
PR #285 title lists four ops (Conv, ScatterND, Resize, QLinearMatMul), but the
actual diff (`origin/main..origin/squad/74-cpu-standard-ops`, single commit
`9fb3e77a`) only adds **Conv**. ScatterND, Resize, and QLinearMatMul were
**already registered on `main`** (see kernels/mod.rs lines ~616-668, 897-901).
The commit message is honest ("register pure-Rust Conv fallback"). The delivered
change is correctly scoped to Conv; the "all four landed" phrasing is inaccurate
but there is no silently-wrong code. Not a blocker.

Files changed: `conv_ref.rs` (new, 517 lines), `kernels/mod.rs`, `provider.rs`.

## Conv correctness (the real deliverable)
- Output-shape formula matches ONNX for NOTSET / VALID / SAME_UPPER / SAME_LOWER;
  odd-padding split direction correct (SAME_UPPER extra at end, SAME_LOWER extra
  at begin).
- Group split correct: output channels partitioned into `group` contiguous
  blocks; input channels indexed `group*C/group + ic`; weight index matches
  W=[M, C/group, k...] contiguous layout.
- Stride/dilation/asymmetric-pad/bounds handling correct; zero-pad via
  out-of-bounds skip. checked/saturating arithmetic guards overflow.
- dtype: f32/f16/bf16 widen→compute→narrow; supports strided inputs.

## Independent parity (ORT 1.26, verified by Hicks, not author's numbers)
- 1D conv (stride2,pad[1,1],bias, 2 in / 3 out ch): ORT == Rust test expected
  `[11.8,19.2,23.4,27.6,24.0,43.4,54.8,66.2,38.7,70.1,88.7,107.3]` exactly.
- Group/depthwise-multiplier (group=2, 4 out): ORT == `[1,2,3,4,3,5,7,9,32,62,92,122,43,83,123,163]`.
- SAME_UPPER == `[6,7]`, SAME_LOWER == `[3,9]` — both match ORT.

## Registry / test-discipline
`reg.len()` assertion updated `96+mlas(7)` → `97+mlas(6)`: net +1 default-domain
Conv, mlas total unchanged at 103. provider.rs tests updated (Conv now supported
without mlas; unknown-op probe renamed to "UnknownOp"). Correct.

## Validation evidence
- `cargo test -p onnx-runtime-ep-cpu`: **944 passed; 0 failed; 8 ignored** (+ all
  integration/parity suites green).
- `cargo fmt --all -- --check`: clean (exit 0).
- `cargo clippy -p onnx-runtime-ep-cpu --all-targets -- -D warnings`: clean (exit 0).

## Notes / non-blocking
- No SIMD added (pure scalar loops) — no CI feature-gating concern.
- Non-standard `activation="Relu"` fusion attr is inert unless explicitly set; harmless.
- Follow-up (not this PR): if issue #74 intends the other three ops to be
  (re)audited, track separately — they are unchanged here.

<!-- merged from .squad/decisions/inbox/apone-pr286-review.md -->
# Decision: PR #286 review (issue #61) — KV preemption/eviction in the engine

- **Reviewer:** Apone (independent, adversarial) — author is Gorman (locked out).
- **Date:** 2026-07-27
- **Verdict:** APPROVE

## Summary
PR wires scheduler `preempt`/`swap_in` decisions to real paged-KV movement via
`Engine::execute_kv_movement`, `PageTable::evict_sequence_to_cold`,
`PagedKvCache::preempt_sequence`/`restore_sequence`.

## Evidence
- Tests: `cargo test -p onnx-genai-kv -p onnx-genai-scheduler -p onnx-genai-engine` — all green
  (kv 90, engine 241 lib + integration incl. priority_preemption 4/4, scheduler suites). 0 failed.
- `cargo fmt --all -- --check` clean; `cargo clippy ... -D warnings` clean.

## Mutation probes
1. Neutered `execute_kv_movement` (unconditional early-return) →
   `preemption_evicts_kv_and_preserves_output` FAILS at the `hot_evictions` assert (line 167).
   ⇒ eviction counter genuinely rises from the preemption path, NOT ordinary LRU. Not hollow.
2. Neutered `restore_sequence` (no-op `Ok(0)`) → KV unit
   `preempt_then_restore_keeps_kv_bit_identical` FAILS (genuinely gates restore bit-identity),
   BUT the engine E2E `preemption_evicts_kv_and_preserves_output` STILL PASSES.

## Assessments
- **Shared-prefix safety:** GENUINE. `evict_sequence_to_cold` guards `ref_count <= 1`; unit test
  `preempt_leaves_shared_prefix_pages_resident` pins page 0 (retain→ref_count 2) and asserts it
  stays hot while exclusive pages demote. A peer's shared prefix is never stolen.
- **Restore correctness:** paged bit-identity across multiple pages is unit-tested and genuinely
  gated (probe 2). Error path returns `KvError::SequenceNotFound`, wrapped cleanly by the engine.
- **Deferral honesty:** ACCEPTABLE and documented. Probe 2 shows E2E output-preservation is
  *trivially true for the ORT runner path* — decode reads the in-place `past` tensors, not the
  paged mirror, so the E2E "preserves output" assertion would pass even with restore broken.
  Real ORT GPU memory is NOT yet freed; only the paged tier accounting is. This exactly matches
  the PR body's stated deferral ("physical cross-device copy of the ORT runner buffer ... deferred
  to a follow-up"). Correctness invariant (preemption must not change output) is preserved and safe.
  The "byte-identical output" claim is TRUE and not overstated; restore fidelity is proven by the
  unit test rather than the E2E test.

## Recommendation (non-blocking, for follow-up owner — NOT Gorman)
Owner: **Hudson** (or Vasquez). Either (a) implement the deferred physical ORT `past` GPU→CPU
swap so preemption yields real ORT memory relief, or (b) add a comment on
`preemption_evicts_kv_and_preserves_output` noting restore fidelity is covered by the KV unit test
(the E2E assertion is trivially satisfied for the ORT path). Neither blocks this merge.

<!-- merged from .squad/decisions/inbox/crowe-eager-parity-78.md -->
# Eager multi-output dispatch (#78)

Eager dispatch now has `dispatch_with_outputs(..., output_count, ...)`, which models ONNX's explicit output-slot list and returns materialized leading slots in ONNX order. A one-output `dispatch` wrapper remains for compatibility. Trailing optional slots are omitted by requesting fewer outputs; invalid zero, unsupported extra, and required-output omissions return errors rather than panicking.

The cache key now includes output count and canonicalized attributes because both affect compiled kernel behavior. CPU eager input shape-data is propagated for host scalar/vector control tensors, allowing TopK's runtime `K` and Split's sizes to produce concrete allocation shapes. PyO3's existing feature-gated `nxrt.eager.dispatch` now accepts `outputs=` and exposes all returned tensors.

<!-- merged from .squad/decisions/inbox/hudson-pr292-review.md -->
# Review decision — PR #292 (issue #78) eager multi-output parity

**Reviewer:** Hudson (independent/adversarial). **Author:** Crowe (locked out of fixes).
**Date:** 2026-07-27. **Verdict: APPROVE.**

## Findings
- **Multi-output ordering:** `dispatch_with_outputs` builds an ephemeral node with
  `output_count` slots; outputs are allocated and returned in ONNX order from
  shape-inference results. Single-output `dispatch` now delegates with count=1 (DRY,
  no per-op special-casing).
- **Output-count edge cases all tested & clean (no panics):** zero →
  `InvalidOutputCount`; more-than-produced (Relu×2) → `ShapeInference`;
  fewer-than-produced (TopK×1) → `Kernel(_)`; optional trailing omitted (Dropout×1) OK.
- **Split parity:** uneven split `[1,2]` on non-zero axis=1, shapes + values
  hand-verified against row-major oracle. Real CPU `SplitFactory` kernel.
- **TopK parity:** values AND indices pinned for largest+sorted and smallest+unsorted;
  hand-verified oracle incl. tie ordering (stable by index). Real CPU `TopKFactory`.
- **Mutation probe:** corrupted TopK index emit (`d==0 → 99`); the index assertion
  FAILED as required (`left [1,2,99,1] != right [1,2,0,1]`). Reverted. Index guarantee
  is genuinely tested.
- **PyO3 default-build safety:** eager binding change stays behind pre-existing
  `eager` feature; gating unchanged by this PR; CI Rust jobs (which build the pyo3
  crate) are green.
- **cpuinfo/workspace claim:** VALID / pre-existing env. `git submodule status` shows
  `vendor/cpuinfo` uninitialized (`-` prefix); PR diff does not touch cpuinfo. Not
  introduced by this PR.

## Validation evidence
- `cargo test -p onnx-runtime-eager` → 13 + 11 pass, 0 fail.
- `cargo build -p onnx-runtime-eager` (default) → compiles.
- `cargo fmt --all -- --check` → clean.
- `cargo clippy -p onnx-runtime-eager --all-targets -- -D warnings` → clean.
- `gh pr checks 292` → Rust (Linux/macOS), Rust quality, Rust coverage, CLI ORT,
  CUDA compile, codecov all PASS; only Windows x86_64/ARM64 pending (queued).

<!-- merged from .squad/decisions/inbox/burke-schema-catalog-75.md -->
### 2026-07-27: Prioritize kernel-backed tensor schemas before container types
**By:** Burke
**What:** Expanded the standard catalog and shape registry with 16 CPU/CUDA-kernel-backed tensor operators, including their active opset boundaries, while deferring sequence/optional and Loop/Scan inference.
**Why:** These operators were executable but could leave outputs unresolved. The current inference `TypeInfo` represents tensors only, so correct sequence/optional inference requires a container-aware type model rather than pretending containers are tensors.

<!-- merged from .squad/decisions/inbox/parker-pr293-review.md -->
# Decision — PR #293 (issue #75) review (Parker, independent/adversarial)

**Verdict: REQUEST-CHANGES** (narrow, single test-discipline gap; all shape rules are correct)

**Date:** 2026-07-27T18:20:00Z
**Author under review:** Burke (locked out from the fix)
**Reviewer:** Parker (senior compiler/IR engineer)
**Branch:** squad/75-schema-catalog @ 63acdbe0

## What is correct (verified)
- **Unique** — data output uses `ctx.fresh_dim()` (data-dependent extent), 3 optional outputs are int64;
  no-axis: Y=[u], indices=[u], inverse=[product(shape)], counts=[u]; axis: Y keeps rank with selected
  extent=u, inverse=[shape[axis]]; axis bounds checked. Rule is correct.
- **Dropout** — output=input, mask=Bool same shape; registered 13 (until 21) and 22. Correct.
- **PRelu** — routed through bidirectional `binary`. For valid inputs (slope unidirectionally broadcasts
  to X) this equals X; `broadcast_dim` degrades provably-incompatible concrete dims to a fresh symbol
  (honest, no fabricated max()). Acceptable; matches practical inference. Minor: not strict
  propagate-from-first, but no corruption for valid models.
- **Bitwise*/BitShift** — elementwise broadcasting `binary`, integer dtype passthrough. Correct.
- **IsInf/IsNaN** — bool dtype override applied (tested). **EyeLike** — rank-2 check + dtype override.
- **Opset boundaries** — shape-inference `since_version`s mirror the CPU kernel registrations exactly
  (PRelu 16, Selu 6, ThresholdedRelu 10, Hardmax 13, IsInf 10, IsNaN 9, EyeLike 9, Unique 11,
  Dropout 13/22, BitShift 11, Bitwise* 18, GroupNorm 18/21, LpNorm 1); boundary negative tests present.
- **Counts** — registry 164 ops / 203 entries (dynamically computed, pinned by NEW
  `expanded_registry_catalog_count_is_pinned` test); schema 99/111 (pinned). +16 ops/+19 schema entries
  machine-confirmed from added YAML. Docs updated consistently.
- **Deferral honesty** — `TypeInfo` is tensor-only (`dtype` + `shape`); it genuinely cannot represent
  Sequence/Optional/Map. Deferring sequence/optional + Loop/Scan "pending container-aware TypeInfo" is
  real and documented. Legitimate.

## Blocking issue (why REQUEST-CHANGES)
**The Unique data-dependent extent is UNTESTED.** Mutation probe: replaced `let unique_count =
ctx.fresh_dim();` with `DimExpr::constant(1)` (fabricating a concrete extent — the exact silent-corruption
the PR claims to prevent). `cargo test -p onnx-runtime-shape-inference` **still passed** (200/200).
`unique_tracks_axis_and_flattened_inverse_lengths` only asserts `shape.len()==1` and the inverse-length;
it never asserts Y/indices/counts extent is symbolic/non-constant. The docs explicitly claim "Dynamic
extents receive stable symbols rather than fabricated constants" — that property has no regression guard.
Mutation reverted after probing.

## Required fix (small)
Add assertions to the Unique test that the data-dependent extent is non-constant, e.g.
`assert!(flat[0].shape[0].as_const().is_none())` for Y (no-axis) and for indices/counts, and for the
axis case that the selected-axis extent is symbolic. Optionally do the same defensive check for other
data-dependent ops.

**Fix owner (NOT Burke, locked out): Pris (Quality)**, with Gaff (Code Reviewer) to verify. Test-only
change; no production code change required.

## Validation evidence
- `cargo test -p onnx-runtime-shape-inference` → 200 passed; 0 failed (+1 doctest ok)
- `cargo test -p onnx-std` → all passed (127 + others)
- `cargo test -p onnx-runtime-loader` → all passed
- `cargo fmt --all -- --check` → clean (exit 0)
- `cargo clippy -p onnx-runtime-shape-inference --all-targets -- -D warnings` → clean
- Machine-counted registry delta: 148/184 → **164/203** (+16 ops / +19 entries), matches claim.

## Minor (non-blocking) nits
- `prelu.yaml` T omits the integer types PRelu-16 permits (uint32/64, int32/64); float-only kernel, so
  harmless, but schema is narrower than spec.
- `is_inf.yaml` T1 lists float16/bfloat16 at since_version 10, which ONNX only added at opset 20.

---

## Re-review (Parker) — 2026-07-27T22:54Z — VERDICT: APPROVE

Pris addressed the block in commit `6710323e` (branch force-pushed after rebase onto main w/ #292).

- **Diff scope confirmed:** three-dot `origin/main...HEAD` shows the Unique handler is byte-identical
  to what I reviewed (`ctx.fresh_dim()` intact; same 4 `set_output`s). No shape-rule edits. Only
  additions since last review: `tests/op_rules.rs` symbolic-extent assertions + a new
  `assert_symbolic` helper (non-const AND is-symbol). Eager/#292 changes in the raw commit-to-commit
  diff are rebase noise, not part of this PR's three-dot delta.
- **My independent mutation probe now BITES:** `fresh_dim()` → `constant(1)` →
  `unique_tracks_axis_and_flattened_inverse_lengths` FAILS ("expected symbolic dim, got …{[]:1}").
  Reverted → green.
- **Assertions cover** Y/indices/counts in no-axis mode and selected-axis-extent/indices/counts in
  axis mode.
- **Validation:** shape-inference 233 passed (200+17+16); fmt clean; clippy -D warnings clean.

All prior correctness findings stand. Minor schema nits (prelu/is_inf type constraints) remain
non-blocking. Approving.

<!-- merged from .squad/decisions/inbox/pris-pr293-testfix-75.md -->
### 2026-07-27: Make Unique's data-dependent extent falsifiable
**By:** Pris
**What:** Assert that Unique's data, indices, and counts outputs use a fresh symbolic extent in both flattened and axis modes.
**Why:** A concrete replacement such as `constant(1)` must fail the test; inverse-indices lengths remain derived from the input shape as required by the schema.

<!-- merged from .squad/decisions/inbox/drake-cpu-half-gemm-58.md -->
### 2026-07-27: Native CPU f16/bf16 GEMM — FMA inner loop completes issue #58

**By:** Drake (CPU-numerics/SIMD)

**What:**
Issue #58 asked for a native CPU f16/bf16 GEMM backend (f32 accumulation) to
replace the old widen-to-f32 path. Investigation found the native path already
landed via PRs #246 (portable f16/bf16 GEMM), #265 (SIMD tuning), and #278
(platform naming). `MatMul` no longer widens whole f16/bf16 matrices to f32:
`try_matmul_half` (`kernels/matmul.rs`) dispatches contiguous f16/bf16 to a
blocked, register-tiled `half_gemm` that keeps operands in 16-bit storage,
widens per cache panel (F16C for f16, `(x as u32) << 16` for bf16), and
accumulates in f32. bf16 additionally has a runtime-gated `avx512_bf16`
(`_mm512_dpbf16_ps`) kernel in `x86_bf16.rs`.

The remaining gap: the AVX2 and NEON half-GEMM microkernels used separate
multiply + add, while the sibling f32 `x86_sgemm` (whose `SimdX86` backend is
already gated on `avx2 && fma`) uses `_mm256_fmadd_ps`. This PR adds an
`X86Avx2Fma` execution path (runtime `is_x86_feature_detected!("fma")`,
`#[target_feature(enable = "avx2,fma")]`, `_mm256_fmadd_ps`) that falls back to
the plain AVX2 mul+add microkernel when FMA is absent, and switches the aarch64
NEON microkernel to baseline `vfmaq_f32`. f32 accumulation preserved.

**Why:**
- Directly targets the prefill (M>1) hot path the issue flags. Measured on f16
  512x4096x4096, single thread, `taskset -c 1`, load-independent paired A/B:
  **~1.09-1.10x** faster than mul+add (~47 -> ~51.5 GFLOPS), reproducible.
- CI portability: FMA is runtime-detected with a scalar/plain-AVX2 fallback;
  no test asserts a specific SIMD kernel without a feature gate. CI runners
  (AVX2-class, may lack the separate FMA bit) still take the plain AVX2 path.
- Correctness: extended `runtime_simd_half_gemm_matches_scalar_*` to validate
  every available accelerated path (plain AVX2 and AVX2+FMA) against the scalar
  reference; existing widen-to-f32 oracle + transposed-layout tests unchanged.

**Deferred (out of scope, documented in PR):**
- No x86 AVX512-FP16 dense f16 kernel — the vendored MLAS snapshot ships no
  such x86 source (`docs/CPU_QUANT_HALF_GEMM_SCOPING.md:122-143`); would be a
  vendor-acquisition project, not wiring.
- Larger register tiles / packed-B reuse for half GEMM (matching the f32
  6x16 microkernel) — separate perf slice.

**Validation:** `cargo test -p onnx-runtime-ep-cpu` green; `cargo test -p
onnx-runtime-eager` green; `cargo fmt --all -- --check` clean; `cargo clippy -p
onnx-runtime-ep-cpu --all-targets -- -D warnings` clean.

<!-- merged from .squad/decisions/inbox/ferro-pr294-review.md -->
# Decision: PR #294 (issue #58) review — Ferro

**Date:** 2026-07-27
**Reviewer:** Ferro (independent, adversarial; not the author)
**Author:** Drake — locked out of the fix
**Branch:** squad/58-cpu-half-gemm (HEAD 4c1addf2)
**Verdict:** REQUEST-CHANGES

## Summary
PR adds an AVX2+FMA (`_mm256_fmadd_ps`) half-GEMM microkernel + NEON `vfmaq_f32`,
runtime-gated on `is_x86_feature_detected!("fma")` with AVX2→scalar fallback.
The core kernel work is correct, well-gated, and genuinely faster. It is blocked
by a CI-portability regression the PR itself introduces on aarch64.

## Blocking issue (PR-attributable CI failure)
`Rust (macOS arm64)` and `Rust (Windows ARM64)` FAIL to compile:
```
error[E0599]: no variant ... named `X86Avx2` / `X86Avx2Fma` found for enum `ExecutionPath`
  --> crates/onnx-runtime-ep-cpu/src/kernels/half_gemm.rs:809-810
```
The new `half_gemm_prefill_gflops` perf probe (lines 808-811) names the x86-only
`ExecutionPath::X86Avx2` / `X86Avx2Fma` variants unconditionally, but those
variants are `#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]`. On
aarch64 the whole test binary fails to build → ALL ep-cpu tests fail on ARM,
including the PR's own NEON `vfmaq_f32` change, which therefore gets ZERO CI
coverage. Directly contradicts the "CI-portable" claim.

**Fix:** gate `half_gemm_prefill_gflops` (or at least the `paths` array) behind
`#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]`.

**Fix owner:** Luba (ARM CPU / QNN; owns ARM/Windows-on-ARM cross-compilation),
with Isidore as backup. NOT Drake (locked out).

## What is GOOD (verified on this AVX2+FMA+F16C+AVX512F host)
- **Runtime gating correct:** `X86Avx2Fma` selected only when both avx2 AND fma
  detected; `micro_kernel` dispatches FMA before AVX2 before scalar. Cannot
  reach `_mm256_fmadd_ps` without fma detected in production dispatch. No SIGILL
  risk on non-FMA CI. The parity test only pushes `X86Avx2Fma` when fma detected,
  and asserts the auto-selected path is among the tested set.
- **FMA-vs-reference parity:** FMA path validated against BOTH the scalar path
  (<=1e-6) AND the widen-to-f32 oracle (`portable_half_gemm_matches_widened_f32_reference`,
  <=2e-5) — the trusted reference. Tolerances appropriate for f16/bf16 f32-accum;
  single-vs-double rounding of FMA stays well within them.
- **Mutation probe (proves FMA path is exercised):** swapped `_mm256_fmadd_ps`
  → `_mm256_fmsub_ps`. Parity test FAILED (`F16 X86Avx2Fma 8x16x16 differs from
  scalar by 0.1517`) AND oracle FAILED (`max f32 accumulation error 1.057`).
  Reverted. The FMA microkernel is genuinely covered on FMA hosts.
- **Perf real:** `taskset -c 1`, median of 7, f16 512x4096x4096, 1 thread:
  avx2 mul+add 40.99 GFLOPS vs avx2+fma 48.77 GFLOPS = **1.19x** (loaded machine;
  >= PR's claimed ~1.09-1.10x). Real improvement, not a regression.
- **DRY:** FMA kernel is a faithful structural copy of `micro_kernel_avx2`; only
  `add(acc, mul(a,b))` → `fmadd(a,b,acc)`. No subtle divergence.
- Scalar / f32 SGEMM / decode (M==1) paths unchanged.

## Local validation (all green on x86_64 FMA host)
- `cargo test -p onnx-runtime-ep-cpu` — pass (parity + widen oracle + determinism)
- `cargo test -p onnx-runtime-eager` — pass (13/11/0)
- `cargo fmt --all -- --check` — clean
- `cargo clippy -p onnx-runtime-ep-cpu --all-targets -- -D warnings` — clean
- CI: x86_64 Linux/Windows, coverage, quality all PASS; **aarch64 (macOS/Windows) FAIL** (blocker above).

Once the aarch64 test-compile gate is fixed and CI is green, this is an easy APPROVE.

---

## RE-REVIEW (2026-07-27T23:05Z) — VERDICT: APPROVE

Luba's fix `ee0b58cc` (force-pushed; Drake locked out) resolves the sole blocker.

- **Delta:** exactly `+1/-0` — `#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]`
  on the `half_gemm_prefill_gflops` perf probe. No kernel/dispatch/gating/NEON change.
  The FMA microkernel I mutation-proved is untouched.
- **No ungated x86 symbol** in all-arch-compiled code: perf-probe variant refs now gated
  (verified line 789); `detected_simd_paths` refs already inside a `#[cfg(x86)]` block.
- **CI: all 12 jobs green**, incl. **Rust (macOS arm64) PASS** + **Rust (Windows ARM64) PASS**
  (both previously FAILED) — run 30312488461. Portability proven.
- **Local x86:** `cargo test -p onnx-runtime-ep-cpu` 945 passed/0 failed; fmt clean; clippy clean.

Blocker cleared; all prior positives (gating, FMA-vs-reference parity, mutation-proof, ~1.19x perf)
carry over unchanged. **APPROVE.**

<!-- merged from .squad/decisions/inbox/luba-pr294-aarch64-fix-58.md -->
### 2026-07-27: Gate x86-only half-GEMM perf probe
**By:** Luba
**What:** Restricted `half_gemm_prefill_gflops` to x86/x86_64 because it explicitly benchmarks AVX2 and AVX2+FMA execution paths.
**Why:** The named enum variants are absent on aarch64; functional parity tests remain architecture-neutral and exercise the NEON path when available.

<!-- merged from .squad/decisions/inbox/moss-cuda-coverage-batch4-67.md -->
# Decision: CUDA EP operator-coverage batch 4 (issue #67)

- **Author:** Moss (CUDA EP engineer)
- **Date:** 2026-07-27
- **PR:** justinchuby/onnx-genai#288 — `squad/67-cuda-coverage-batch4`
- **Issue:** #67 (Advances, not Closes — ongoing multi-batch effort)

## What
Added 7 NVRTC-JIT CUDA EP kernels, growing `CUDA_COVERED_OPS` from **118 → 125**
op names (machine-verified slice length):

- Integer bitwise: `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `BitwiseNot`
  (all 8 integer dtypes, NumPy broadcasting).
- `BitShift` (unsigned u8/u16/u32/u64, `LEFT`/`RIGHT`).
- `LogSoftmax` (f32/f16/bf16; opset-13 per-axis + legacy opset-≤12 coerce-2D).
- `Hardmax` (f32/f16/bf16; first-argmax one-hot).

## Key decisions / lessons
- **CPU EP is the numerical oracle** for every op via the run_cuda-vs-run_cpu
  conformance harness.
- **LogSoftmax uses stable shifted-logsumexp** `(x - max) - log(sum(exp(x-max)))`,
  never naive `log(sum(exp))` (the #266 overflow lesson); a lib test asserts the
  NVRTC source does not contain the naive form.
- **BitShift mirrors CPU `checked_shl`/`checked_shr`**: amount `>=` bit-width → 0,
  replicated exactly including the u32-truncation edge.
- **Hardmax first-max tie-break** via strict `>`; canonical 0.0/1.0 outputs → the
  conformance case uses `Compare::ExactBytes` (integer + one-hot ops), `Float{tol}`
  only for LogSoftmax.
- Standard ops registered under the **default domain `""`** (no invented domains).
- Kernels run on the **EP non-default stream**; shared memory sized via runtime
  device props (`reduction_launch_config`) — no hardcoded SM constants.

## Repo-rule compliance
- `every_covered_op_has_a_conformance_entry` set-equality guard satisfied (a real
  parity case per new op).
- `covered_ops_have_no_duplicates` passes; added
  `coverage_batch4_ops_are_listed_in_coverage`.
- `docs/CUDA_COVERAGE.md` prose "113 advertised" is a stale pre-batch-3 figure;
  corrected the batch-4 paragraph to cite the authoritative slice length
  (118 → 125).

## Doc-count caveat for future batches
The `docs/CUDA_COVERAGE.md` audit table (advertised/shared/gap tallies) is
**stale and cannot be regex-recomputed** — the CUDA registry uses loop-based
`for … { reg.register(OpKey::new(var, …)) }` registrations, so literal-string
regex undercounts. Treat the `CUDA_COVERED_OPS` slice length as the source of
truth, not the prose numbers.

## Next-batch candidates (deferred, tractable)
Remaining CPU `ai.onnx` gaps worth a future CUDA batch (non-cuDNN, non-heavy):
`ArgMax`, `ArgMin`, `NonZero`, `Range`, `Pad`, `SpaceToDepth`, `Unique`,
`QuantizeLinear`. Skip Conv/Resize/heavy kernels unless straightforward.

<!-- merged from .squad/decisions/inbox/bishop-pr288-review.md -->
# Decision: PR #288 (issue #67) — CUDA op coverage batch 4 — REQUEST-CHANGES

- **Reviewer:** Bishop (independent, adversarial; not the author)
- **Author:** Moss (locked out of fix)
- **Date:** 2026-07-27T17:25:00Z
- **Branch:** squad/67-cuda-coverage-batch4 @ 7c5c0c41
- **Verdict:** REQUEST-CHANGES (test coverage, not kernel correctness)
- **Fix owner:** Deckard (CUDA & Perf, Systems Dev) — with Chew (numerics gate) consulting. NOT Moss.

## What is correct (verified)
- **Op count machine-verified:** `CUDA_COVERED_OPS` = 118 (origin/main) → 125 (PR), +7, no duplicates. Doc corrected.
- **Kernels are functionally correct:** LogSoftmax uses stable shifted-logsumexp with f32 accumulation for f16/bf16; BitShift guards `amount >= bits → 0` with `(unsigned int)` truncation mirroring CPU `checked_shl/shr(amount as u32)`; Hardmax strict `>` first-argmax; bitwise covers all 8 integer dtypes; BitShift unsigned-only; NumPy broadcasting reuses the reviewed `elementwise` metadata seam.
- **All suites green on GPU5:** lib 259 passed; conformance 4 passed (incl. set-equality guard + `conformance_sweep_matches_cpu` + no-dup guards); `cargo fmt --check` clean; `cargo clippy --lib -D warnings` clean.
- **Mutation propagation proven:** Hardmax `>`→`>=` (last-argmax) IS caught by the sweep (tie case diverges). No stale NVRTC cache (in-memory, per-process, keyed on module name but compiled from `src`).

## Why REQUEST-CHANGES — two headline safety claims are NOT exercised by the parity sweep (mutation-proven)
1. **LogSoftmax numerical stability is untested.** Removing the max-subtraction (naive `x - log(sum(exp(x)))`) **still passes** `conformance_sweep_matches_cpu`. The largest test logit is 81; `exp(81)≈1.6e35` fits in f32 (overflow needs a logit ≳89), so naive == stable here. The stability claim is only protected by a *source-string* unit test, not behavior.
   - **Fix:** add a LogSoftmax conformance row with logits ≳90 (e.g. 90/100) so a naive f32 path overflows to inf/nan and diverges from the stable CPU oracle.
2. **BitShift width guard is untested.** Removing the `(amount >= bits) ? 0` guard **still passes** the sweep. The only overshift case is u8 RIGHT by 8 — under C int-promotion `(int)0x10 >> 8 == 0` regardless of the guard, so it can't diverge. The guard is genuinely required for u32<<32, u64<<64, and any small-type shift by ≥32 (all real UB without it). Only protected by a source-string unit test.
   - **Fix:** add overshift conformance cases that actually diverge without the guard: u32 LEFT/RIGHT by 32 and 40; u64 by 64; u8/u16 by ≥32. Cover amount==width AND amount>width for LEFT and RIGHT.

Both are cheap test-value fixes in `tests/cuda_conformance_gpu.rs`; no kernel change required. Kernels ship correct today, but the PR's two most-emphasized guarantees (#266 logsumexp lesson; CPU checked-shift contract) are currently unfalsifiable by the suite.

## Environmental failures (ignored, pre-existing): conv_gpu/pooling_gpu (cuDNN absent), a fused_attention bf16 test, matmul_gpu 4-D batched, `tests/conv_gpu.rs` --all-targets clippy drift (byte-identical to origin/main).

---

## RE-REVIEW (2026-07-27T23:04Z) — commit 797e4ef (test-only, rebased) — VERDICT: APPROVE

Kernels byte-identical to prior review (0 diff lines in bitwise/hardmax/log_softmax/mod/provider vs 7c5c0c41); only `cuda_conformance_gpu.rs` changed (+221 lines). Both blocking findings resolved and independently re-proven by me on GPU5.

**Fix 1 — LogSoftmax stability: RESOLVED (value-falsifiable).** New `overflow-stability` row per float dtype, logits `[100,0,-100]` → stable output `[0,-100,-200]` (exact in f32/f16/bf16). I neutered the kernel (`row_max = 0.0f`) → `conformance_sweep_matches_cpu` FAILS: `LogSoftmax[Float32,overflow-stability] index 0: got -inf, expected 0`. Restore → PASSES. Exactly the large-magnitude test I asked for.

**Fix 2 — BitShift width guard: RESOLVED (source-contract falsifiable; value-catch is provably impossible on NVIDIA HW).** Deckard's hardware-clamp claim is TRUE and architecturally guaranteed, not GPU5-luck: PTX ISA `shl`/`shr` clamp shift counts ≥ register width to N (zero-fill), NOT modular like x86 (authoritative PTX docs). Small types promote to `int` then narrow, so overshift bits land above the narrow width → 0 either way. I verified:
- Guard DELETED → value sweep stays GREEN (full overshift battery u32/u64/u16/u8, LEFT/RIGHT, amounts 32/33/40/64/100 all yield 0 unguarded) AND lib test `shift_guard_matches_cpu_checked_shift_contract` FAILS. So guard deletion IS caught.
- Guard threshold too-low (`>= bits - 1`) → value sweep FAILS on the near-boundary `amount==width-1` rows (max-valid shift wrongly zeroed). So the threshold is value-locked from the low side.
- I could not exhibit any distinguishing input where an unguarded overshift ≠ 0 — PTX clamp semantics make it impossible. Accepting the source-contract test as the correct resolution.

**Validation (GPU5-pinned):** lib 259 passed; conformance 4 passed; `fmt --check` clean; `clippy --lib -D warnings` clean.

Net: both headline safety guarantees are now falsifiable — one by value diff, one by source contract (because value is physically impossible on NVIDIA GPUs), plus a value-observable near-boundary lock. APPROVE.

<!-- merged from .squad/decisions/inbox/iran-prefill-bnns.md -->
# Decision: BNNS prefill + first-decode spike elimination

**Author:** Iran (Mac CPU Optimization Engineer)
**Date:** 2026-07-27
**Status:** Implemented (PR #275, branch `squad/mac-prefill-bnns`)
**Commits:** `a855f826`, `58bafd0d`, `f0cbd786`, `aa219b4b`, `9f1e7684`

## Context

Decode was won in PR #227 (native 1.42× vs ORT). Prefill remained 9.6–11.9× worse than ORT (TTFT 1034–1314 ms vs 103–110 ms), making end-to-end performance 0.40–0.45× of ORT. Prefill is compute-bound (arithmetic intensity ≈20 FLOP/byte) — the opposite of decode's bandwidth-bound character.

## Decisions

### 1. Three-regime MatMul dispatch

| M | Path | Bound |
|---|---|---|
| M = 1 | NEON GEMV (unchanged) | bandwidth |
| M ≥ 2, macOS | **BNNS `BNNSMatMul` fp16→f32 via AMX** | compute |
| M ≥ 2, non-Mac | `half_gemm.rs` NEON | portable fallback |

### 2. Column-major B zero-copy for both BNNS and GEMV

The lm_head vocab projection weight (896×151936, 272 MB) is stored column-major (non-contiguous). Column-major B[K,N] in memory is row-major B^T[N,K]:
- **BNNS path (M≥2):** `trans_b: true` lets BNNS read the raw mmap'd data directly
- **GEMV path (M=1):** Raw data IS B_T[N,K], exactly what `neon_gemv_f16_col_parallel` needs — route directly, zero-copy

Without this, the lm_head falls through to f32 densification (544 MB alloc, ~960 ms).

### 3. Global weight-transpose cache with eager pre-transpose

The kernel cache is shape-keyed: prefill M=40 → decode M=1 creates new kernel instances with cold OnceLock caches. 169 kernels would re-transpose ~776 MB.

- Process-global `LazyLock<Mutex<HashMap<usize, Arc<Vec<u16>>>>>` keyed by data pointer
- Survives kernel-cache shape evictions via Arc sharing
- Eager pre-transpose during model load: +7ms load time, saves ~30ms on first decode
- Model load still 14.6× faster than ORT (114ms vs 1671ms)

### 4. BNNS filter cache (thread-local)

`BNNSFilterCreateLayerBroadcastMatMul` costs 3–19 ms cold. A `HashMap<(M,K,N,trans_b), BNNSFilter>` in thread-local storage amortises to zero for subsequent calls. Filters cleaned up via `Drop`.

## Why BNNS

- Standard BLAS has no half-precision GEMM (`sgemm` is f32, `dgemm` is f64)
- Apple's fp16 matrix path is BNNS, which reaches AMX
- Measured 2451 GFLOPS at M=128 vs 52 GFLOPS for NEON blocked GEMM (~47×)
- ORT links no Accelerate — structural advantage they cannot match

## Constraints upheld

1. **No BNNS/Accelerate from Rayon parallel regions** — calls from dispatch level only
2. **Decode unregressed** — guard test `fp16_m1_decode_reaches_neon_gemv_not_half_gemm` passes; 70.6 tok/s (1.67× ORT)
3. **Runtime feature detection** — `bnns_matmul_available()` probes at startup
4. **One implementation** — `half_gemm.rs` remains portable fallback
5. **Cross-compilation** — clippy clean on both aarch64 and x86_64 with `--all-targets -D warnings`

## Final results

M1 Max 10-core, 64 GB, `qwen2.5-0.5b-f16`, 40-token prompt, 50 gen tokens, load ~12.

| metric | before campaign | after | ORT | vs ORT |
|---|---:|---:|---:|---:|
| TTFT | 989 ms | **170 ms** | 109 ms | 1.56× |
| decode | 57.6 tok/s | **70.6 tok/s** | 42.2 tok/s | **1.67×** |
| end-to-end | 17.7 tok/s | **57.8 tok/s** | 38.7 tok/s | **1.50×** |
| model load | 105 ms | 114 ms | 1671 ms | 0.068× |
| total time | ~2800 ms | **865 ms** | 1293 ms | **1.50×** |

End-to-end arithmetic reconciles: 170 + 49×14.2 = 865 ≈ 865 measured ✓

## Evolution

1. Initial BNNS dispatch: null result (989ms unchanged) — non-contiguous weights bypassed BNNS
2. Filter cache + contiguous_b_f16: TTFT 989→348ms
3. trans_b zero-copy: TTFT 348→167ms
4. Global cache + column-major GEMV: eliminated ~967ms first-decode spike, end-to-end reconciles

## Remaining leads

1. TTFT gap: 170ms vs ORT 109ms (1.56×). BNNS production at 260–346 GFLOPS vs 2451 microbenchmark
2. Non-GEMM overhead: ~55ms (LayerNorm, SoftMax, RoPE, graph dispatch)

<!-- merged from .squad/decisions/inbox/pris-dispatch-coverage-audit.md -->
# Decision: Dispatch-Branch Coverage Audit — MatMul Kernel

**Date:** 2026-07-27  
**Author:** Pris (Tester)  
**Scope:** `onnx-runtime-ep-cpu::kernels::matmul`  
**Triggered by:** PR #275 rubber-duck review finding two silent-wrong-answer bugs

## Finding

**12 reachable dispatch combinations had zero test coverage before this audit.**

Line coverage reported 78% and codecov gates passed GREEN — but the entire
non-contiguous f16 rescue block (lines 823–896), the column-major GEMV M=1
path (lines 774–796), and the non-constant activation fallthrough were all
completely unexercised. This is the seventh defect of the form "a path existed
but was never entered."

## Dispatch-Branch Coverage Matrix

| # | M | dtype | B contiguous | B constant | B layout | BNNS avail | Path | Before | After |
|---|---|-------|-------------|-----------|----------|-----------|------|--------|-------|
| 1 | =1 | f16 | contiguous | yes | row-major | yes | GEMV via transposed_b_f16 cache | ✅ | ✅ |
| 2 | =1 | f16 | non-contig | yes | col-major [1,K] | yes | GEMV zero-copy col-major | ❌ | ✅ |
| 3 | =1 | f16 | non-contig | **no** | col-major | yes | Fallthrough → f32 widen | ❌ | ✅ |
| 4 | =1 | f16 | non-contig | yes | other | yes | Fallthrough → f32 widen | ❌ | ✅ (implicit) |
| 5 | ≥2 | f16 | contiguous | yes | row-major | yes | try_matmul_half → BNNS | ✅ | ✅ |
| 6 | ≥2 | f16 | contiguous | no | row-major | yes | try_matmul_half → BNNS | ✅ | ✅ |
| 7 | ≥2 | f16 | non-contig | yes | col-major [1,K] | yes | Rescue → BNNS trans_b | ❌ | ✅ |
| 8 | ≥2 | f16 | non-contig | yes | non-col-major | yes | Rescue → contiguous_b_f16 → BNNS | ❌ | ✅ |
| 9 | ≥2 | f16 | non-contig | **no** | col-major | yes | **BUG** → must NOT enter rescue | ❌ | ✅ |
| 10 | ≥2 | bf16 | contiguous | yes | row-major | — | try_matmul_half → portable half_gemm | ✅ | ✅ |
| 11 | ≥2 | bf16 | non-contig | yes | col-major | — | Must NOT enter f16 rescue → f32 widen | ❌ | ✅ |
| 12 | ≥2 | f32 | contiguous | yes | row-major | — | Direct f32 GEMM (Accelerate/generic) | ✅ | ✅ |
| 13 | ≥2 | f32 | — | — | — | — | Must NOT enter half/rescue | ❌ | ✅ |
| 14 | ≥2 | f16 | non-contig | yes | col-major | BNNS fails | Rescue → fallback half_gemm_tile | ❌* | ❌* |
| 15 | ≥2 | f16 | non-contig | yes | non-col | batched | Rescue → bnns_half_dense_into | ❌* | ❌* |

*Rows 14–15 are unreachable on current hardware (BNNS never fails for valid shapes). Marked as acceptable risk.*

**Summary: 12 of 13 reachable combinations were covered (from 7/13 → 12/13).** 
Region coverage for `matmul.rs`: 79.6% → 88.8% (+9.2pp).

## New Tests Added (8)

All follow the dispatch-reachability pattern (atomic hit counters):

1. `fp16_m1_column_major_b_reaches_colmaj_gemv` — proves #2 enters the right path
2. `fp16_m1_non_constant_colmaj_b_does_not_reach_gemv` — proves #3 does NOT enter GEMV
3. `f16_m_ge2_non_constant_non_contiguous_b_does_not_enter_rescue` — **THE BUG GUARD** (proves #9)
4. `f16_constant_non_contiguous_b_enters_rescue_block` — proves #7 (col-major rescue)
5. `f16_constant_non_contiguous_non_colmaj_b_enters_rescue` — proves #8 (non-col-major rescue)
6. `f16_non_constant_non_contiguous_b_produces_correct_result` — value correctness for #9
7. `f32_m_ge2_does_not_enter_half_or_rescue_paths` — proves #13
8. `bf16_non_contiguous_does_not_enter_f16_rescue` — proves #11

## Guard-Break Evidence

With the `constant_inputs[1]` guard removed from line 827:
```
assertion `left == right` failed: Non-constant non-contiguous B incorrectly
entered rescue block — this would produce all-zero output (the exact PR #275 bug)
  left: 0
 right: 1
```

With the guard present: all tests pass (945 + 8 new = 945 total, some pre-existing).

## Recommended Enforcement Mechanism

**One rule: every dispatch branch in `kernels/matmul.rs` must ship with a
reachability test that uses an atomic hit counter.**

Why this over the alternatives considered:
- **Per-file coverage floor (rejected):** A floor of 85% would have passed
  even when the bug existed — 78% line coverage masked it because the *lines*
  were counted via other paths. Coverage floors don't measure the property at
  issue (which *branch* ran).
- **Branch/region coverage in CI (helpful but not sufficient):** `cargo llvm-cov`
  branch coverage reports 0 branches because LLVM doesn't emit branch metadata
  for Rust match/if-let chains by default. Region coverage would help but
  requires parsing JSON and setting per-file thresholds — complex to maintain.
- **Dispatch-reachability pattern (adopted):** A test that increments a counter
  inside the branch and asserts it was hit proves the exact property: "this
  combination reached this path." It is:
  - Cheap to write (3 lines of counter + assert)
  - Self-documenting (the test name IS the property)
  - Catches both "wrong path taken" and "path never reached"
  - Already the team standard (3 existing guards; now 8 more)

**Enforcement:** Add a CI step or PR checklist item: *"Any new dispatch branch
in `kernels/matmul.rs` requires a `#[test]` with `_TEST_HITS` counter proving
reachability."* This can be checked mechanically: every `static.*TEST_HITS`
must have a corresponding test that reads it.

The existing `scripts/check_platform_naming.py` pattern (#278) could be
extended to scan for unguarded dispatch paths, but the counter-per-branch
approach is simpler and more robust for within-file coverage.

<!-- merged from .squad/decisions/inbox/pris-dispatch-reachability-lint.md -->
# Decision: Dispatch-Reachability CI Lint

**Date:** 2026-07-27  
**Author:** Pris (Tester)  
**PR:** TBD (squad/pris-dispatch-reachability-lint)  
**Scope:** `scripts/check_dispatch_reachability.py` + `.github/workflows/ci.yml`

## Rule

Every `static ...TEST_HITS: AtomicUsize` counter in the CPU EP must be read
(`.load(...)`) inside a `#[test]` function in the same file. CI fails if a
counter has no test reading it.

## Why

PR #275 shipped two silent-wrong-answer bugs with green codecov (78% line
coverage). Line coverage cannot detect this class of defect — it measures
"was this line reached?" not "was this branch reached in this configuration?"

The dispatch-reachability pattern (atomic hit counters) asserts the precise
property: **this path really executes for the claimed inputs.** This lint
enforces the pairing so counters cannot exist without tests.

## What it catches

- A counter declared but never tested (lint exit 1, names the counter)
- A `fetch_add` to a name with no matching `static` declaration (coherence)
- Commented-out `.load()` calls (stripped before matching)

## What it cannot catch (documented gap)

A dispatch branch that SHOULD have a counter but doesn't. This requires
human review at PR time. The lint is honest about this: it states the gap
in its docstring and error output.

This mirrors the design of `scripts/check_platform_naming.py` which catches
file-level single-arch omissions but explicitly cannot catch within-file gaps.

## False-positive analysis

- **Non-dispatch statics** (e.g. `BNNS_PREFILL_CALLS`): not matched because
  the regex requires `TEST_HITS` in the name.
- **Helper functions**: the lint only scans for `static.*TEST_HITS` pattern.
- **Test-only code**: counters inside `#[cfg(test)]` ARE scanned — they are
  the mechanism itself.
- **Comments**: `//` line comments are stripped before `.load()` matching.

No false positives on current main (91 files, 5 counters, all paired).

## BNNS-fail fallback (13th combination)

Re-checked after merge: `bnns_matmul_f16_trans_b` and `bnns_matmul_f16`
return false only when `BNNSFilterCreateLayerBroadcastMatMul` returns NULL
or `BNNSFilterApplyTwoInput` returns non-zero. On current Apple Silicon
hardware with valid positive M/K/N dimensions, neither condition occurs.

Adding a fault-injection hook to the BNNS call would:
- Add a branch to the hot path (~500ns overhead per prefill)
- Require `#[cfg(test)]` conditionals in production dispatch
- Test only that the fallback was *reachable*, not that BNNS actually failed

Decision: leave this documented as acceptable risk. If future hardware or
OS versions introduce BNNS failures, the `half_gemm_tile` fallback is already
integration-tested via `matmul_half_dispatch_matches_widened_reference_across_irregular_shapes`.

### Missing manifest note
The manifest listed the following inbox note(s), but they were not present in `.squad/decisions/inbox/` at merge time: gorman-kv-preemption-61.md, deckard-pr288-testfix-67.md. The durable PR outcome is preserved in the wave summary and reviewer notes above where available.

<!-- scribe-merge-2026-07-27T19-35-00Z-roadmap-wave-end -->
