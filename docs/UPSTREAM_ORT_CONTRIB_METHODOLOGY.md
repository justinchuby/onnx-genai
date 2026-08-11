# Upstream ORT Contribution — Validation Methodology

> **Status:** Protocol definition (no implementation).
> **Owner:** Sebastian (benchmark validity), Chew (numerics), Challenger (reachability).
> **Scope:** Any kernel or optimization contributed from this project upstream to `microsoft/onnxruntime`.

---

## 1. Reachability Tests

**Requirement:** Prove the optimized path is actually dispatched — not a scalar fallback.

### Acceptance criteria

| # | Criterion |
|---|-----------|
| R1 | A unit test asserts the intended kernel/ISA path executes for every supported EP/dtype/shape combination. |
| R2 | The test must **fail** if dispatch regresses (e.g., the kernel is silently bypassed or a scalar fallback is taken). |
| R3 | Detection mechanism must be explicit: either an instrumentation counter, a provider-level dispatch log entry, or a compile-time static assertion on the selected template instantiation. |
| R4 | Silent fallback detection: run under a mode that traps any invocation of the generic/reference kernel when the optimized path is expected. A "claim diagnostic" (see `docs/EP_CLAIM_DIAGNOSTICS.md`) must accompany the kernel registration. |

### Lessons from this repo

This project has documented the failure mode **"was the feature actually on?"** extensively:

- **EP claim diagnostics** (`docs/EP_CLAIM_DIAGNOSTICS.md`): a kernel that reports `KernelMatch::Unsupported` without an actionable reason is indistinguishable from one that was never registered. Every decline must carry a diagnostic reason.
- **Performance claim discipline** (`.squad/decisions.md`): "A SIMD/accelerated path without a reachability test is equivalent to an unwired placeholder."
- **Instruments that lie**: measurements on a path that silently fell back to CPU/scalar are worse than no measurement. Require positive dispatch proof before any perf claim.

### Implementation pattern

```
// Pseudocode — not production code
#[test]
fn kernel_is_dispatched() {
    let session = build_session_with_profiling(model, target_ep);
    session.run(inputs);
    let profile = session.end_profiling();
    assert!(profile.contains_kernel("MyOptimizedKernel_AVX512"));
    assert!(!profile.contains_kernel("MyOp_ScalarFallback"));
}
```

### Reviewer: **Challenger**

---

## 2. Numeric Parity

**Requirement:** Optimized kernels must match the scalar/f64 reference within a justified, locked tolerance.

### Tolerance policy

| Dtype | Reference | Max relative error | Max ULP | Notes |
|-------|-----------|-------------------|---------|-------|
| fp64 | Authoritative reference | 0 | 0 | Used as ground truth |
| fp32 | Scalar fp64 | 1e-5 relative | 4 ULP | Per-element |
| fp16 | Scalar fp64 | 1e-3 relative | — | Accumulated error bounded separately |
| bf16 | Scalar fp64 | 5e-3 relative | — | Wide mantissa loss expected |
| int8 (quantized) | fp32 dequant→reference | Quantization-noise floor | — | Must not exceed theoretical quant error |

### Edge-case coverage (mandatory)

- **Denormals:** Test with subnormal inputs; kernel must not flush-to-zero unless documented.
- **NaN/Inf propagation:** IEEE 754 compliance; NaN in → NaN out, no silent corruption.
- **Zero-size tensors:** Empty dimension must not crash or produce garbage.
- **Extreme shapes:** {1×1}, {1×N} with N > 64K, non-power-of-2 dimensions, prime dimensions.
- **Reduction accumulation:** For reductions and long chains (softmax, layernorm over seq_len > 8192), bound accumulated error relative to the fp64 chain, not just per-element.

### Regression lock

Every tolerance is captured in a test with an explicit threshold constant. If a code change widens the error, the test fails. Tolerance may only be relaxed via a reviewed decision with justification.

### Reviewer: **Chew**

---

## 3. Representative Model-Level Benchmarks

**Requirement:** A microbenchmark win must be accompanied by a model-level measurement — or an explicit "end-to-end impact unmeasured" statement.

### What to benchmark

| Regime | Models | Batch sizes | Sequence lengths |
|--------|--------|-------------|-----------------|
| Prefill | Representative dense (e.g., Qwen2.5-0.5B, Phi-3-mini) | 1, 4, 16 | 128, 512, 2048 |
| Decode | Same models | 1, 4, 16 | 128, 512, 2048 context |
| Quantized | Same models in int4/int8/fp16 variants | 1, 4 | 512 |

### What to report

| Metric | Required |
|--------|----------|
| Latency p50, p95, p99 | ✓ |
| Throughput (tokens/sec) | ✓ |
| Peak memory (RSS or device) | ✓ |
| Variance (stddev or IQR) | ✓ |
| Number of repetitions | ✓ (minimum 30 after warmup) |
| Warmup iterations discarded | ✓ (minimum 5) |

### Rules

1. **No cherry-picking.** Report full distribution, not a single best run.
2. **Variance is mandatory.** If stddev > 5% of mean, investigate and disclose the source.
3. **Warmup is mandatory.** First N runs are discarded to allow JIT, cache warming, thermal stabilization.
4. **Microbenchmark-only claims** must be labeled: "Microbenchmark only — model-level impact not measured. Amdahl-estimated ceiling: X%."
5. **Amdahl sanity check.** If the optimized op is <5% of total model time, a 2× kernel speedup yields <2.5% model-level gain — state this.

### Reviewer: **Sebastian**

---

## 4. Honest Same-Artifact Methodology

**Requirement:** Every performance comparison must use identical artifacts, configurations, and conditions.

### "Same artifact" definition — all must match

| Dimension | Requirement |
|-----------|-------------|
| Model file & weights | Byte-identical ONNX file, same path |
| Quantization | Same scheme, same calibration, same group size |
| Batch size & sequence length | Identical |
| Sampling parameters | Same top-k/top-p/temperature/seed or greedy |
| Hardware | Same GPU/CPU SKU, same device index |
| Clock state | Fixed clocks (nvidia-smi -lgc) or documented boost policy |
| Thread count & affinity | Explicit ORT_NUM_THREADS, same pinning |
| Build type | Both Release, or both RelWithDebInfo — never mix |
| Build flags | Same compiler, same -O level, same CUDA arch |
| ORT version/commit | Identical or explicitly stated difference |
| Graph optimization level | Same ORT optimization level (e.g., ORT_ENABLE_ALL) |
| Cache state | Both warmed identically (same warmup procedure) |
| Thermal state | Controlled: fans at max, or wait for thermal steady-state |

### Red-flag list — comparisons that MUST be rejected in review

| # | Dishonest comparison | Why it lies |
|---|---------------------|-------------|
| 1 | Optimized build vs. unoptimized/debug upstream | Inflates speedup by penalizing baseline |
| 2 | Debug assertions enabled on one side only | Adds overhead to one side |
| 3 | Different threadpool sizes | More threads = more parallelism ≠ kernel improvement |
| 4 | Different graph optimization levels | Higher opt level fuses ops, confounding kernel measurement |
| 5 | Fused path vs. unfused baseline | Measures fusion benefit, not kernel quality |
| 6 | Shape nobody runs in production | Irrelevant speedup; state the shape and its real-world frequency |
| 7 | Different quantization or precision | Comparing int4 speed to fp32 speed is not a kernel improvement |
| 8 | Stale baseline build (not fresh origin/main) | May include known regressions already fixed upstream |
| 9 | Different CUDA driver/toolkit versions | Driver differences can shift perf ±10% |
| 10 | Reporting peak throughput without p95/p99 tail | Hides variance and thermal throttling |
| 11 | Host under different background load | Noisy neighbor invalidates comparison |
| 12 | Comparing one engine that crashes/falls back vs. one that succeeds | Report a capability gap, not a throughput multiplier |

---

## 5. Upstream ORT Expectations

### Verified requirements (from `microsoft/onnxruntime` repository)

Citations from the repo as of 2026-08:

| Requirement | Source |
|-------------|--------|
| Unit tests mandatory for all PRs | `CONTRIBUTING.md`: "New code *must* be accompanied by unit tests." |
| Code coverage target ≥80% | `docs/Coding_Conventions_and_Standards.md`: "aim at maintaining over 80% coverage" |
| PR must describe motivation and measurement methodology for perf fixes | `docs/PR_Guidelines.md` §2: "mention the improvement and how the measurement was done" |
| Follow Google C++ style (120 char lines, exceptions allowed) | `docs/Coding_Conventions_and_Standards.md` |
| Use lintrunner; CI must pass | `CONTRIBUTING.md`: git hooks, lintrunner |
| CLA required | `CONTRIBUTING.md`: CLA-bot decorates PRs |
| Non-trivial changes need a feature-request issue first | `CONTRIBUTING.md`: "use the feature request issue template to discuss it with the team" |
| Do not use PRs as scratch pads; build/test locally first | `docs/PR_Guidelines.md` §7 |
| Keep PRs small (<10 files for non-cosmetic changes) | `docs/PR_Guidelines.md` §8 |
| Separate cosmetic from functional changes | `docs/PR_Guidelines.md` §9 |

### Unverified / requires manual confirmation

- **Specific benchmarking CI infrastructure**: ORT has `onnxruntime_perf_test` but the exact CI benchmark harness and approval workflow for perf PRs is not documented in the files reviewed. Confirm with ORT maintainers before submitting.
- **Required EP coverage per kernel PR**: unclear whether a new CPU contrib op also requires CUDA/ROCm/WebGPU stubs or only the targeted EP. Confirm per-PR.
- **Benchmark regression gates**: whether ORT CI auto-rejects perf regressions is unverified.

---

## Pre-Flight Checklist

Before claiming any speedup in an upstream PR, the contributor MUST complete:

- [ ] **Reachability proven**: dispatch test passes; silent fallback trapped
- [ ] **Numeric parity locked**: tolerance test passes for all target dtypes against fp64 reference
- [ ] **Edge cases covered**: denormals, NaN/Inf, zero-size, extreme shapes tested
- [ ] **Model-level measurement done** (or explicitly stated as unmeasured with Amdahl estimate)
- [ ] **Same-artifact methodology verified**: all dimensions in §4 table match
- [ ] **Red-flag review**: no items from the red-flag list apply
- [ ] **Evidence template filled** (see below)
- [ ] **ORT CI green**: lintrunner passes, unit tests pass, no new warnings
- [ ] **CLA signed**
- [ ] **Feature issue filed** (if non-trivial)

---

## Required Evidence Template

Every perf claim in a PR description must include:

```
## Performance Evidence

**Hardware:** [e.g., NVIDIA A100 80GB, Intel Xeon 8380, ...]
**OS/Driver:** [e.g., Ubuntu 22.04, CUDA 12.4, driver 550.xx]
**ORT version/commit:** [exact commit SHA]
**Build type:** Release
**Build flags:** [cmake flags or build.py arguments]
**Thread count:** ORT_NUM_THREADS=X, affinity=[pinned/unpinned]
**Graph optimization level:** ORT_ENABLE_ALL
**Clock state:** [fixed at X MHz / boost enabled / documented]

**Model:** [name, source, ONNX opset]
**Quantization:** [scheme, group size, calibration method]
**Batch/Seq:** [batch_size=B, seq_len=S]
**Warmup:** [N iterations discarded]
**Repetitions:** [M iterations measured]

**Command line (reproducible):**
```
[exact command to reproduce]
```

### Results

| Metric | Baseline | Optimized | Δ | p-value or CI |
|--------|----------|-----------|---|---------------|
| Latency p50 (ms) | X | Y | -Z% | ... |
| Latency p95 (ms) | X | Y | -Z% | ... |
| Throughput (tok/s) | X | Y | +Z% | ... |
| Peak memory (MB) | X | Y | ±Z% | ... |

**Variance:** stddev=X ms (Y% of mean)
**Reachability:** [link to dispatch test / profiler output]
**Numeric parity:** [link to tolerance test results]
**Model-level confirmed:** [yes/no — if no, Amdahl estimate: X%]
```

---

## Review Ownership

| Domain | Reviewer | Responsibility |
|--------|----------|---------------|
| Numeric correctness & tolerance | **Chew** | Verify fp64 reference, ULP bounds, edge cases |
| Benchmark validity & methodology | **Sebastian** | Verify same-artifact, variance, model-level confirmation |
| Reachability & dispatch proof | **Challenger** | Verify kernel is actually taken, fallback detection works |

---

## Appendix: Standing Rules Referenced

From `.squad/decisions.md`:

1. "A per-layer or microbenchmark speedup is not a model-level claim."
2. "A SIMD/accelerated path without a reachability test is equivalent to an unwired placeholder."
3. "Native-vs-ORT claims must compare the same artifact, quantization, accuracy level, and steady-state methodology with oracle-correct output."
4. "Benchmarks for 35B-A3B must build from a fresh origin/main worktree."
5. SIMD/NPU/kernel paths must match scalar/f64 reference within justified tolerance and be locked with a regression test.
