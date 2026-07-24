# Decisions

> Current decision ledger. Older ledgers are under `.squad/decisions/archive/`.

> Scribe archive policy: when this file exceeds the hard gate, keep only the current active reconciliation here and move older active ledger content into `.squad/decisions/archive/`.

<!-- scribe-merge-2026-07-23T02-50-00Z-persistent-default-shipped -->
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

## 2026-07-20 — CPU decode: resident pool and guarded GQA row parallelism

### Keep persistent M=1 decode-pool residency
**By:** Sapper; reviewed by Luv 🟢  
**What:** Run the whole native CPU M=1 forward inside one bounded decode-pool `install`, using a worker-local, nested, panic-safe RAII residency guard so each MatMulNBits call executes inline rather than reinstalling the same pool. `ONNX_GENAI_CPU_DECODE_THREADS=0`, prefill, default-feature-off, and CUDA behavior remain unchanged. Landed on main as `cbacb75`.  
**Why:** Qwen2.5-0.5B int4 decode improved about 3–6% with bit-identical tokens. This proves install crossings were avoidable but not the dominant remaining cost. Luv verified TLS isolation, Rayon semantics, deadlock safety, feature gates, and the CPU/build test matrix.

### Parallelize sufficiently large CPU GQA attention rows
**By:** Roy; reviewed by Luv 🟢  
**What:** Parallelize independent `(batch, query_head, query_sequence)` rows with one Rayon fork-join only above a 163,840 `row × key × head-dimension` work guard; retain serial execution below it. Each task owns a disjoint output row and private score buffer while preserving each row's reduction order. Landed on main as `c391327`.  
**Why:** Short decode regressed when parallelism was unconditional. Guarded parallelism improved 512-token decode throughput by 8.6%, reduced profiled GQA time by 13.9%, and cut 225-token prefill GQA time by 88.3%, with bit-identical 1-thread/8-thread greedy output. A future coverage follow-up may force exact serial/parallel comparison for a large ragged batch.

### Retain Tier-A GQA KV copy cleanup, defer shared append-only KV
**By:** Roy; regression coverage by Pris  
**What:** Borrow contiguous f32 past caches, remove a redundant owned clone, and replace scalar cache materialization loops with contiguous slice copies. Keep attention math and the SSA output contract unchanged. Pris added f16-widening and ragged-per-batch cache-materialization regressions.  
**Why:** The cleanup is bit-identical and removes avoidable work, but measured end-to-end decode was neutral within noise. True O(1)-append shared KV requires runtime aliasing/lifecycle changes and remains deferred.

### Do not land the decode fork-join granularity prototype
**By:** Deckard  
**What:** Revert the coarser 8/12-task MatMulNBits prototype and profiling probes; no commit landed.  
**Why:** Long runs regressed 7.1–8.4%. Post-residency profiling showed serial GQA at about 20.58 ms/token exceeded MatMulNBits at about 15.51 ms/token, so reducing projection task count removed steal slack rather than solving the dominant bottleneck. Revisit only as graph-level projection fusion, after GQA.

## 2026-07-20 — CUDA fused flash attention

### Fuse standard Attention only on measured-winning shapes
**By:** Rachael; reviewed by Chew 🟡  
**What:** Add an NVRTC tiled online-softmax backend behind `AttentionKernel`, including f16 WMMA with f32 accumulation and scalar f32/f16/bf16 support for MHA/GQA/MQA, causal/non-causal attention, and additive mask planes. Auto dispatch retains Phase-2a for decode, `D>128`, unsupported layouts/features, and measured-slower long spans. Landed on main as `a67b7a5`.  
**Why:** H200 f16 S512 improved about 1.53–1.60× and removed 48 MiB score scratch; S2048 regressed heavily when forced, so fallback is part of the design. Chew found the online-softmax merge, WMMA masking/synchronization, numerics, and dispatch sound. Non-blocking coverage remains for explicit Auto fallback gates, non-multiple-of-16 f16 head dimensions, and per-batch/per-head masks.

### Fuse GroupQueryAttention prefill with distinct physical and causal origins
**By:** Bryant, corrected by Rachael after Chew rejection; final review Chew 🟢  
**What:** Reuse the shared flash kernel behind `com.microsoft::GroupQueryAttention` for measured-winning prefill. Cache append and implicit RoPE use `total_length - key_sequence_length`; attention causal masking uses the distinct query start `total_length - query_sequence_length`. The final parity matrix covers 40 scenarios across f32/f16/bf16, MHA/GQA/MQA, fresh/cached/ragged, RoPE, local window, softcap, generic non-WMMA routing, large scores, unequal Q/K lengths, and Auto fallback. Landed on main as `94fa2b6`.  
**Why:** Bryant's first revision incorrectly reused the K append origin for queries when `Sq != Sk`; Chew rejected it and locked that artifact. Rachael's revision made the failing `Sq=2,Sk=4` case pass, tightened tolerances, and preserved exact present K/V. H200 fresh Q512 is about 1.31× faster with 48 MiB scratch saved; cached/large slower shapes fall back. The corrected artifact is approved and no active lockout remains.

## 2026-07-20 — Issue #40 Phase 1 distributed-runtime foundation

### Slice 1a: shared protocol trace + ticketed non-blocking host pressure
**By:** Tyrell; reviewed by Gaff 🟡  
**What:** Add the unpublished `onnx-runtime-protocol-trace` crate with public protocol envelopes/identities and a conformance-only independent `ReplayChecker`; add `HostGovernor` ticketed pressure accounting to `onnx-genai-scheduler`. All state transitions and trace linearization points commit under one short ledger lock; waits occur only on ticket-local condition variables after capacity is atomically charged. Landed on main as `0d1d265`.  
**Why:** The implementation conforms to `PressureProtocol.tla` invariants through an independent deterministic replay campaign and snapshot invariant checks. Gaff approved with two non-blocking issues—terminal-entry reaping and cancel-granted wake-after-unlock—which were folded into slice 1b. The TLC model gate is CI-deferred because Java/TLA tooling is unavailable locally.

### Slice 1b: Communicator + in-process backend + BufferOwnership registry
**By:** Tyrell; reviewed by Gaff 🟢  
**What:** Add unpublished `onnx-runtime-comm` with the async `Communicator` trait, synchronous reference `InProcessCommunicator`, and one-lock `OwnershipRegistry` over read/write lease sets. Dropping an operation handle detaches but does not release storage; terminal completion/abort releases leases, and freed allocation IDs remain tombstoned to prevent reuse/ABA. Reuse the slice-1a trace framework and independently replay `BufferOwnership` events. Landed on main as `e4d2883`.  
**Why:** Gaff verified exactly-one-owner, conflict, release, transfer, generation/ABA, non-blocking-lock, linearization, barrier, mailbox, and deterministic-conformance obligations. Slice 1b also reaps terminal pressure entries and moves all pressure wakeups after unlock. Non-blocking follow-ups for 1c include abort waking barrier waiters, barrier-map cleanup, and documenting tombstone growth.

### Slice 1c: one topology-wide collective ordering authority — IN PROGRESS
**By:** Tyrell  
**What:** Implement direct host rendezvous collectives behind a shared `CollectiveSequencer`; keep canonical submit order independent per communicator group, use one slot for count exchange plus all-to-all-v data, freeze reduction member order with checked arithmetic and per-contribution f16/bf16 rounding, and bound free tombstones with an exact window plus allocator-proven epoch floors.  
**Why:** This maps to `CollectiveOrdering.tla`: ranks may progress asynchronously without divergent order, groups do not acquire a false global enqueue order, completion stays rank-local, and abort freezes submissions before backend wakeup. This slice is not yet landed.

### Phase-1 deferred gates and remaining phases
**By:** Scribe  
**What:** Keep the TLC model gate CI-deferred. After 1c, Phase-1 slice 1d weight residency remains pending; issue #40 Phases 2–4 remain pending.  
**Why:** The landed Rust conformance harnesses provide deterministic implementation-side evidence, but do not replace the configured CI model check or the remaining distributed-runtime roadmap.

## 2026-07-20 — Issue #40 collective ordering completion

### Land slice 1c with serialized abort wakes and broad equivalence coverage
**By:** Tyrell; reviewed by Gaff 🟢  
**What:** Land all seven in-process collectives behind one canonical per-group `CollectiveSequencer`, deterministic member-order reduction, additive independent replay checking, bounded allocation tombstones, and rank-local completion. Abort now holds each rendezvous mutex while notifying its paired condition variable, closing the review's notify-before-park race. Distributed-equals-single-device bitwise coverage spans all_reduce, reduce_scatter, all_gather, broadcast, all_to_all, and all_to_all_v. Landed as `2ffb4e4` with follow-up `128440d`.  
**Why:** Gaff found the architecture and TLA refinement sound but blocked the original revision on a rare abort-path lost wakeup. Tyrell's deterministic waiter gate proved the fix, all comm/trace/scheduler suites passed, and the broadened equivalence matrix preserves fixed-rank-order determinism. TLC remains CI-deferred.

## 2026-07-20 — CUDA graph M4 capture-safety

### Own the CUDA graph lifecycle and exercise native decode replay
**By:** Rachael and Deckard; replay coverage by Pris; reviewed by Chew 🟢  
**What:** Serialize one CUDA graph lifecycle inside `CudaRuntime`, capture/replay only on its dedicated stream, invalidate on generation/binding lifecycle changes, and split capture-end from instantiate so failed instantiation cannot leak the intermediate `CUgraph`. Native decode remains flag-gated and strict-audit: unsupported graphs fall back eagerly. A capture-safe synthetic decoder proves token-exact eager/replay parity across reset, stable addresses, O(1) scalar uploads, two captures, sixteen replays, and zero fallbacks. Landed as `637e247`, `5470c01`, `dd2d807`, and `4755575`.  
**Why:** The first Qwen test exercised only fallback and was rejected as replay evidence. The final synthetic integration test executes the real `NativeDecodeSession::decode_cuda` state machine and resolved the M4 decode-loop review blocker without weakening the all-kernel capture audit.

### Gate MatMulNBits M=1 capture safety to the proven decode path
**By:** Bryant  
**What:** Remove trailing GEMV synchronizations and advertise MatMulNBits capture compatibility only after a successful no-`g_idx`, M=1 decode warmup; prefill, grouped-index, unwarmed, and configuration-changing paths remain ineligible. Runtime D2H helpers explicitly order after the EP stream. Landed as `a210703`.  
**Why:** The proven GEMV path is allocation-free, D2H-free, and synchronization-free, while the excluded paths dequantize, allocate, or validate on the host.

### Make fixed-shape GQA decode capture-safe with detect-before-consume metadata guards
**By:** Deckard, Rachael, and Bryant; reviewed by Chew 🟢  
**What:** Persist GQA scratch and remove the trailing stream sync (`dcb4f1b`); move advancing decode metadata reads and derived lengths on-device (`77829b9`); preserve warmup rejection and add on-device replay bounds checks with sentinel no-write behavior (`82c249d`). The final shared sticky error latch poisons subsequent replay steps after any violation and is polled immediately after logits D2H, before token consumption; explicit graph reset clears it. Landed final as `ca50bae`.  
**Why:** Earlier revisions were rejected for silent clamping and then for allowing a later valid replay to resume over a skipped KV row. The final detect-before-consume latch makes invalid metadata a hard, deterministic failure while valid fixed-capacity f32 one-token replay remains byte-identical and allocation-free.

### Make four normalization variants capture-safe
**By:** Roy; reviewed by Chew 🟢  
**What:** Remove trailing synchronizations from LayerNormalization, RMS/SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization, and SkipLayerNormalization. Keep SkipSimplified broadcast metadata in a mutex-protected, shape-keyed persistent cache and permit capture only after successful single-group warmup. Landed as `6184d82`.  
**Why:** The warmed decode paths now have stable metadata and no per-step allocation, free, upload, host read, or stream synchronization; the full CUDA suite and direct capture/replay byte-parity test passed.

### Bind elementwise capture eligibility to exact warmed signatures
**By:** Sapper and Deckard; reviewed by Chew 🟢  
**What:** Make supported unary and binary floating-point decode kernels capture-safe using persistent broadcast metadata and removed trailing synchronizations. Replace the initial boolean eligibility gate with mutex-protected exact dtype/entry and shape signatures; prefill, i64, errors, and signature changes remain ineligible. Landed final as `85b6f4e`.  
**Why:** Chew rejected the boolean gate because a warmed kernel could later execute a different dtype or shape during capture. Exact signatures close that TOCTOU while preserving numerics and the approved persistent-metadata design.

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
