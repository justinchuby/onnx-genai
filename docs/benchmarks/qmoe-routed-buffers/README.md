# Integer QMoE routed buffers: candidate and performance gate

**Status: candidate; do not enable default merge yet.** Single-row decode has a measurable cost in this experiment. Zero warm issuer callbacks does not establish zero inference overhead.

Baseline `08634f19a81ee9150951f66254c03fc1b221c52f` integrates the two float-MoE banks/accumulator and shared-view safety work. The measured workspace is pinned by measured-source.patch and provenance.json. The final CPU-only candidate.patch excludes the mistaken duplicate CUDA test described below; timed CPU/runtime bytes are unchanged.

## Layout and lifetime

One admitted lane contains three disjoint f32 regions: gather `(rows>1 ? rows*H : 0)`, contributions `rows*top_k*H`, and final accumulator `rows*H`. Counts, byte layouts and the composite sum are checked. There is no expert-count multiplier or new public EP interface. Each expert still writes its original route slots; final reduction keeps the original token/route order. Main-thread scratch stays alive through the prefetch worker join. The final output write remains last, preserving strided outputs and input/output aliases.

The resource test pre-admits one 3-row lane, then runs 3→1→3 rows. Every fitting checkout/execute/return makes zero actual issuer calls, including zero-byte calls; retained capacity stays charged and one backing remains owned. Weight grants use the existing separate source in that test, so it does not prove zero issuer calls for transient weight admission. Routing, input copies and expert intermediate results remain outside this workspace bound. This is not a full RSS cap.

## Protocol and results

Apple M1 Max/macOS arm64, native CPU EP, synthetic public InferenceSession prepare/run/output-drop. Seven shapes, three independent processes per baseline/after/identical-baseline-null arm, nine timed batches each, pilot-calibrated to 100ms CPU per batch. Pilots are excluded. Arm order rotates. The whole build/sweep holds the stable flock shared with the llama.cpp peer. Process-start snapshots are retained locally; no continuously idle-host claim is made. The same-binary Relu control gate is ±5%, and every timed/control batch must reach 20ms. All seven duration/control gates pass; no retries were needed.

Complete f32 outputs and row argmaxes match in all 63 formal processes. Requirements equal actual funding. With top_k=2 the new retained bytes are exactly the old gather requirement plus rows*H*4*3. Newly retained capacity replaces formerly independent temporary Vec allocations; it is not evidence of lower RSS.

| rows / H / I / threads | CPU after vs base | absolute CPU change | null vs base | workspace before → after |
|---|---:|---:|---:|---:|
| 1 / 32 / 32 / 1 | +1.01% | +0.289 µs | +0.02% | 0 → 384 |
| 1 / 64 / 64 / 1 | +1.99% | +1.717 µs | +0.02% | 0 → 768 |
| 1 / 256 / 256 / 1 | -1.78% | -21.843 µs | -1.91% | 0 → 3072 |
| 8 / 256 / 256 / 1 | +1.05% | +25.476 µs | +0.02% | 8192 → 32768 |
| 8 / 256 / 256 / 4 | -0.57% | -18.677 µs | -1.45% | 8192 → 32768 |
| 512 / 256 / 256 / 4 | +1.14% | +117.000 µs | +0.12% | 524288 → 2097152 |
| 1 / 64 / 64 / 4 | -0.32% | -0.565 µs | -0.99% | 0 → 768 |

Full CPU/wall medians and ranges, controls and raw batches are retained in summary.json/runs.json. In particular H64 single-row/T1 adds about 1.717µs (+1.99%) while its null is +0.02%; this cannot simply be called noise. H32 adds about 0.289µs (+1.01%). H256 after and null move together, so its negative delta is not a demonstrated speedup. Multirow results are closer to neutral but do not erase the single-row cost.

## Decision and next experiment

Keep this candidate reviewable, but defer default merging while addressing the single-row acceptance gap. Do not introduce a private kernel pool, skip charging singleton buffers, reuse capture pin state, remove release guards or rerun the same timings until favourable.

The next independent experiment should keep this same QMoE computation and compare per-run return with a common executor lane held across a bounded active decode scope. Time the loop and final return to the pool separately (physical trim is a distinct event); verify actual checkout/recycle/issuer counts. Return the owner at request end, phase transition or scheduling yield, and retain asynchronous/capture/quarantine restrictions. A narrow borrowed runner with explicit fallible finish is a design candidate; it has not been implemented or benchmarked by this report.

## Validation

- QMoE focused: 59 passed, one existing ignored. Full CPU: 2030 passed/20 existing ignored; MLAS reference: 2072 passed/35 existing ignored; session: 296 passed. Strict all-target CPU/CUDA/session and MLAS Clippy pass.
- New tests cover 3→1→3 warm reuse and retained charging, top-k product/composite layout overflow, strided output canaries and real input/output aliases. Existing exact route accumulation, all gates, affine and prefetch/cache-policy tests remain passing.
- The default qmoe_gpu run compiled but executed zero tests (57 ignored), as required for CUDA-named targets without gpu-tests. I initially misread this as missing coverage and duplicated an older host test into that target. That duplicate was reverted: the existing host test already lives in the separate qmoe_cpu_oracle target. The correct target is explicitly selected with --exact and its named cpu_qmoe_oracle_prepares_declared_workspace_without_cuda test passes (rows 1 and 6, exact unit-weight f64 reference). The CUDA source scan passes over 95 targets. Initial zero-execution and mistaken-duplicate logs remain labelled in the archive; they are not the evidence for the correct host target. The duplicate file was never linked into the CPU timing probe, and its removal changes no timed CPU/runtime source. CUDA device tests remain unexecuted.
- No new whole-kernel Miri claim is made here. The shared partition helper has its existing layout tests; threaded Miri limitations recorded in earlier source-scoped reports remain limitations.

## Active-lane attribution follow-up

The [isolated five-arm experiment](https://github.com/justinchuby/onnx-genai/blob/ed5bd92a828de562568fe1bda90483b2dc8417db/docs/benchmarks/active-execution-lane/README.md) retained 75 formal processes and
included setup/finish in total costs. Same-binary scoped/unscoped differences are
about -0.021% at H32 and -0.448% at H64, with overlapping ranges and +1.323%
original/null drift for H64. H256 failed its control gate. These results do not
cleanly attribute or erase the earlier cost. The functional scope prototype
passes its tests but is not adopted or integrated into the real engine. Keep
this QMoE PR draft; investigate the existing warm-fit mechanism before adding
public lifecycle surface. The prototype is preserved only on the experiment
branch, with hash-verified binary copies cleaned after the study.
