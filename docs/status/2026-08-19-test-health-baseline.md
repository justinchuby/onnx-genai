# Test-health baseline — `main`, 2026-08-19

> **This is a dated snapshot, not a permanent truth.** A test-health baseline
> decays: `main` moves constantly, issues get fixed, new reds appear. If you are
> reading this more than a week or two after the date above, treat every count as
> *stale until re-run* and re-measure before relying on it. Presented as current
> when it is old, this document is worse than none.

## What this is

An honest, whole-picture inventory of which test suites are green / red / ignored
on `main` **right now**, so an agent on day one can tell instantly whether a
failure is theirs or pre-existing. It is an **inventory, not an investigation** —
failures are named and classified, not diagnosed.

## Environment (applies to every number below)

| | |
|---|---|
| Commit measured | `641cbefa` (worktree `wt-batching`); `origin/main` tip at write time `5c98d53b`, a Scribe docs chore #1463, test-irrelevant |
| CPU | Intel i7-13800H, 14C/20T |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB (8188 MiB), driver 591.55 |
| CUDA | 13.1 runtime (no toolkit), WDDM |
| Toolchain | rustc 1.97.1 (8bab26f4f 2026-07-14) |
| OS | Windows |
| Box state | Shared and intermittently loaded (another agent's GPU A/B + heavy parallel builds). Counts noted as contention-sensitive where they are. |

**Worktree setup requirement:** a fresh worktree needs
`git submodule update --init --recursive` before anything builds — the vendored
`onnx-runtime-cpuinfo/vendor/cpuinfo` submodule is otherwise empty and the CPU-info
build script fails. This is environment setup, not a code defect.

## Summary

| Suite (invocation) | passed | failed | ignored | Verdict |
|---|---:|---:|---:|---|
| `onnx-runtime-ep-cpu` (default) | 1474 | 0 | 20 | ✅ clean |
| `onnx-runtime-ep-cuda` `--features cuda --lib -j1` (Marlin default-on) | 418 | 2 | 21 | ⚠️ 2 fail, both tracked #1305/#1405 |
| `onnx-runtime-ep-cuda` (default features, lib, parallel) | 410 | 3 | 21 | ⚠️ 3 fail (3rd is parallelism-sensitive), all tracked #1305/#1405 |
| `onnx-genai-server` (default) | 278 | 1 | 3 | ✅ effectively clean (1 fail is contention-flaky, passes isolated) |
| `onnx-genai-engine` `--features cuda,native-backend --lib` + `RUN_CUDA_SMOKE=1` | 565 | 0 | 4 | ✅ clean |
| `cargo test --workspace` (default) | — | — | — | ❌ **does not compile** — filed #1466 |

Counts are summed across lib + integration + doc-test targets per package. A suite
that reports `ok` with everything ignored proves nothing, so ignored counts are
recorded everywhere.

## Per-suite detail

### `onnx-runtime-ep-cpu` — ✅ 1474 / 0 / 20
`cargo test -p onnx-runtime-ep-cpu` (default features, CPU only, ~222 s).
Clean. 20 ignored are the usual GPU-runner / benchmark gates.

### `onnx-runtime-ep-cuda` — ⚠️ 418 / 2 / 21 (authoritative), 410 / 3 / 21 (parallel)
The crate's **lib unit tests auto-run on the real GPU** when a CUDA device is
present, regardless of the `cuda` cargo feature. Integration `tests/*_gpu.rs`
targets are all `ignored` unless the `gpu-tests` feature is enabled (not run here —
see boundaries).

- **`--features cuda --lib -- --test-threads=1`, Marlin default-on: 418 / 2 / 21** (754 s) — authoritative.
- **default features, lib, default parallelism: 410 / 3 / 21** (471 s).

Failures (all **tracked**, all under **#1305** and its follow-up **#1405**):

| Test | Marlin-dependent? | Notes |
|---|---|---|
| `kernels::matmul_nbits::tests::fp16_gate_up_swiglu_is_bit_exact_to_two_op_path` | **Yes** | #1305 failure #1. Passes with `ONNX_GENAI_MARLIN_M_GT_1=0`. Marlin int4 GEMM is documented non-byte-identical. |
| `kernels::group_query_attention::tests::reference_scores_path_gates_on_dtype_and_decode_support` | **No** | #1305 failure #4. Deterministic, pure-function, **still fails with `MARLIN=0`**. Asserts `gqa_reference_scores_path(f32, q_seq=1, head_dim=192)` is `true`, but it is `false` because `gqa_decode::supported(1, 192)` now returns `true` — a stale test vs. code. Independent of Marlin; not resolved by the `MARLIN=0` workaround. |
| `kernels::matmul_nbits::tests::fp16_gemv_matches_dequant_reference_block128` | **Yes (and parallelism-sensitive)** | Failed only in the parallel default-feature run; **did not reproduce** single-threaded, and passes with `MARLIN=0`. Same Marlin non-byte-identical family. |

This is the same set #1305 recorded (it reported 398 / 4 / 21 with `--features cuda`);
`main` has since moved — **#1404** landed and resolved the two capture-safe-gate
tests (`fused_gate_up_swiglu_rmsnorm_is_bit_exact...` #2/#3), so the count is now
418 / 2 / 21 single-threaded.

> **Note for #1305 readers:** #1305 frames all four failures as one class. Measured
> here: three of the four are Marlin-`split-K`-driven and vanish with
> `ONNX_GENAI_MARLIN_M_GT_1=0`, but failure **#4 (GQA reference-scores gate) is
> Marlin-independent** and remains red under `MARLIN=0`. It is a stale-assertion /
> routing-change mismatch, not a numeric-precision issue. Worth splitting out.

### `onnx-genai-server` — ✅ 278 / 1 / 3 (effectively clean)
`cargo test -p onnx-genai-server` (default). The single failure,
`tests::fim_stream_returns_headers_before_generation_finishes` (a tokio
`Elapsed(())` timeout at `src/tests.rs:2138`, an SSE-headers-before-generation
timing assertion), is **contention-induced flakiness**: re-run in isolation on an
unloaded box it **passes** (1 passed, 1.46 s). The box was under heavy parallel
cargo + GPU load when it tripped. Not a regression.

### `onnx-genai-engine` — ✅ 565 / 0 / 4
`cargo test -p onnx-genai-engine --features "cuda,native-backend" --lib --
--test-threads=1`, with `ONNX_GENAI_RUN_CUDA_SMOKE=1` and
`ONNX_GENAI_DEFER_EAGER_SYNC=0` (53 s, real GPU). Clean, and matches the #1284
landing state. The 4 ignored include the two capture tests gated by **#1284**
(the synthetic fixture routes growable KV through `Cast`, not a capacity-form
attention op, so `binding_consumers_use_physical_capacity` correctly declines
CUDA-graph capture; un-skipping them requires rewriting the fixture's KV to feed a
real CUDA attention kernel). The default-feature (CPU-only) engine run is not
separately meaningful — native-decode / CUDA tests are gated behind
`cuda,native-backend` + `RUN_CUDA_SMOKE`, so without them they simply don't run.

### `cargo test --workspace` — ❌ does not compile (filed #1466)
Under default features the workspace test build **fails to compile** and yields
**zero** test results. `crates/onnx-genai-bench/tests/fused_batch_prefill.rs`
unconditionally imports `bench-native`-gated symbols (`synthetic_decoder`,
`NativeDecodeSession`, `onnx_runtime_session`) with no top-level
`#![cfg(feature = "bench-native")]` guard (3× `error[E0432]`). Because cargo
compiles all selected test targets before running any, this aborts the whole
workspace run — even `--no-fail-fast` does not help (it skips test failures, not
compile failures). Correct invocation for that crate is
`cargo test -p onnx-genai-bench --features bench-native`. This was **untracked**;
filed as **#1466**. Per-package suites (above) are how real coverage was obtained.

## Configuration-dependent reds (read before blaming your change)

| Env var | Default | Effect on tests |
|---|---|---|
| `ONNX_GENAI_MARLIN_M_GT_1` | **on** (`marlin_m_gt_1_enabled()`) | Default-on causes the Marlin non-byte-identical `matmul_nbits` reds in #1305/#1405 (`fp16_gate_up_swiglu_is_bit_exact...`, `fp16_gemv_matches_dequant_reference_block128`). Set `=0` and they pass. The GQA #4 red is **not** affected. |
| `ONNX_GENAI_DEFER_EAGER_SYNC` | on | Set `=0` to avoid the **#1439** `CUDA_ERROR_ILLEGAL_ADDRESS` use-after-unmap crash on weight-paging / weight-lending paths. `=0` is byte-identical to known-good. All GPU runs here used `=0`. |

## Boundaries — what was **not** run, and why

- **`onnx-runtime-ep-cuda` integration `tests/*_gpu.rs` under `--features gpu-tests`.**
  These un-ignore ~real-device GPU integration tests (attention, block-quantized
  matmul, allocation counts, etc.). Not run: it needs a long, *quiet* GPU window,
  and the box was shared with another agent's timing-sensitive A/B and a
  weight-paging crash investigation (#1439). The lib unit tests already exercise
  the CUDA kernels on the real device, so the kernel-level picture is covered; the
  integration harness sweep is deferred.
- **H200-only cases.** Several benchmarks/tests are H200 performance gates
  (`ignored, H200 performance benchmark`). No H200 on this box — not runnable here.
- **Large streaming models** (`qwen14b-zp`, 27B, MoE fixtures) as test inputs.
  These stream on an 8 GB card and their timing is weight-transfer bound; not part
  of the unit/integration suites and not exercised here.
- **`cargo test --workspace` full green count.** Blocked by the #1466 compile
  break; a `--exclude onnx-genai-bench` full-workspace run (30+ min cold build)
  was not performed — per-package suites cover the same targets.

## Anything alarming?

No silent failures, no pass-for-wrong-reason, no vacuously-green suite surfaced.
The one previously-untracked red (the `--workspace` compile break) is **loud**, not
silent, and is now filed (#1466). Everything else red is tracked (#1305/#1405) or
environmental (contention-flaky server timeout).

## Issue cross-reference

| Issue | Status vs this baseline |
|---|---|
| #1305 | Open. 2–3 ep-cuda reds. Note above: failure #4 (GQA) is Marlin-independent, unlike #1/#2/#3. |
| #1405 | Open. Marlin M>1 default-on stales the capture-safe invariant / reds fused-SwiGLU tests. |
| #1404 | Landed — resolved the two capture-safe-gate reds formerly in #1305. |
| #1284 | Open (partial fix landed via #1458). 2 engine capture tests gated `#[ignore]`; fixture rewrite remains. |
| #1439 | Open. Weight-paging `CUDA_ERROR_ILLEGAL_ADDRESS`; avoided with `ONNX_GENAI_DEFER_EAGER_SYNC=0`. |
| #1466 | **Filed by this baseline.** `cargo test --workspace` compile break (bench `fused_batch_prefill`). |
