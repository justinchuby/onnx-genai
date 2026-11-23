# Float MoE accumulator workspace

Source checkpoint `47c4973d421e2b2f5f7ed16a2afb34f99a4cbd75`; baseline is `098f7d3f1052746d1600d91de2b436e359a7b2be` (two projection banks plus the separate unaligned-storage fix).

## Change and accounting scope

The final rows*hidden f32 accumulator moves from an independently allocated Vec into a third disjoint region of the existing admitted Host workspace lane. It is initialized once per dispatch and survives all expert groups. It has no top-k or expert-count multiplier. A/B reuse and accumulation order are unchanged. The final narrowing/strided write stays after all input reads, including aliased input/output storage. No new public EP interface or quota callback is added.

Each measured requirement increases by exactly rows*hidden*4 bytes, verified against actual funding. This accounts for an existing result allocation; it is not evidence that total live memory or RSS falls. Routing, widening, final output owners and GEMM internals remain outside this workspace cap. Larger retained lanes continue to be charged at physical capacity.

## Protocol

Apple M1 Max, macOS arm64, native CPU EP, synthetic public InferenceSession prepare/run/output-drop. Eight shapes, three independent processes per baseline/after/identical-baseline null arm, nine timed batches per process, 100 ms target CPU time per pilot-selected batch. Pilots are excluded. Arm order rotates across processes. The complete sweep holds the shared stable flock used by the llama.cpp peer; process-start load snapshots are retained locally. There is no continuous foreign-CPU telemetry. A same-binary Relu control and ±5% control gate expose drift; they do not prove an otherwise idle machine.

Complete f32 outputs and row argmaxes match in all 72 formal processes. This is a synthetic session comparison, not a model/token-throughput or CUDA claim. Binary/probe hashes and raw batches are in provenance.json and runs.json.

| rows / H / I / threads / variant | CPU after vs base | null vs base | control after / null | duration / control gate | workspace before → after |
|---|---:|---:|---:|---|---:|
| 1 / 64 / 64 / 1 / identity (threshold default) | +0.37% | +0.80% | 1.0142 / 0.9997 | True / True | 512 → 768 |
| 8 / 256 / 256 / 1 / identity (threshold default) | -0.09% | -0.01% | 1.0026 / 1.0071 | True / True | 16384 → 24576 |
| 512 / 256 / 256 / 4 / identity (threshold 0) | +1.23% | -3.96% | 0.9968 / 0.9919 | True / True | 2097152 → 2621440 |
| 512 / 1024 / 1024 / 4 / identity (threshold default) | -0.15% | -0.14% | 1.0088 / 0.9691 | True / True | 8388608 → 10485760 |
| 1 / 64 / 128 / 1 / swiglu1 (threshold default) | -0.89% | -0.65% | 1.0173 / 1.0277 | True / True | 1536 → 1792 |
| 8 / 256 / 32 / 1 / swiglu0 (threshold default) | +0.24% | +0.95% | 1.0278 / 1.0741 | True / False | 16384 → 24576 |
| 512 / 256 / 256 / 4 / swiglu2 (threshold 0) | +0.98% | +0.16% | 0.9897 / 0.9732 | True / True | 3145728 → 3670016 |
| 8 / 64 / 128 / 4 / silu3 (threshold 0) | -6.67% | -15.11% | 0.9571 / 1.0058 | True / True | 24576 → 26624 |

CPU and wall medians and complete ranges are retained per arm in summary.json. Interpret each timing only together with its null/control result. Failed gates remain visible; this report does not silently discard or replace cells.

## Fixed supplementary diagnostic

The original serial split-SwiGLU control failed its 5% gate, and parallel cells had noticeable null drift. One prescribed diagnostic kept the source and binaries unchanged, doubled repetitions and ran all six arm permutations across the four affected shapes (72 further processes, six per arm). Hashes were verified before running. The original 72 processes remain above and in runs.json; diagnostic results do not replace them. Complete outputs and argmaxes match across all 144 formal processes. No further retries were prescribed.

| rows / H / I / threads / variant | CPU after vs base | null vs base | wall after vs base | control after / null | duration / control gate |
|---|---:|---:|---:|---:|---|
| 8 / 256 / 32 / 1 / swiglu0 | +0.19% | -0.35% | +0.19% | 1.0245 / 1.0336 | True / True |
| 512 / 256 / 256 / 4 / identity | -0.83% | -2.92% | +1.60% | 1.0280 / 0.9841 | True / True |
| 512 / 256 / 256 / 4 / swiglu2 | +0.31% | +0.72% | +1.98% | 1.0427 / 1.0152 | True / True |
| 8 / 64 / 128 / 4 / silu3 | +2.37% | -15.37% | -2.15% | 1.0177 / 0.9926 | True / True |

Read timing differences alongside the null/control and per-process ranges in diagnostic/summary.json. Zero warm workspace issuer calls is a tested resource property; it does not establish exactly zero overall inference overhead.

The diagnostic serial split-SwiGLU CPU change is +0.19% (null -0.35%). The forced identity and split-SwiGLU cells have overlapping process ranges; their diagnostic CPU changes are -0.83% and +0.31%. No universal speedup or exactly-zero-cost claim follows.

The small parallel SiLU+FC3 cell remains inconclusive: diagnostic CPU after/base is +2.37%, wall after/base is -2.15%, and the identical-baseline null CPU change is -15.37%. The CPU ranges overlap and wall behavior does not follow the CPU median. Preserve that uncertainty rather than interpreting either the original negative CPU delta or the diagnostic positive delta as an established effect.

## Correctness and validation

- Native CPU library: 2025 passed, 20 existing ignored. MLAS reference configuration: 2067 passed, 35 existing ignored. Session: 296 passed; layout: 11 passed. Strict all-target native CPU/session and MLAS Clippy passed. Logs are compressed alongside this report.
- Existing scalar oracle and bit-exact driver comparisons remain; new accumulator tests preserve strided f32/f16/bf16 output gaps and real input/output alias behavior. The resource test `execution_workspace_bounds_cover_threshold_skew_and_pool_changes` observes real issuer calls, including zero-byte calls, and requires no additional calls for fitting warm checkouts across thresholds and pools. That workspace-only property does not imply zero callbacks for transient weight admission.
- Default Miri passes the separate aligned/unaligned f32 conversion regression test. Whole threaded alias execution under Tree Borrows reaches a passing test result but exits on Crossbeam leak reports; default Stacked Borrows stops in Crossbeam intrusive pointer conversion. All-dtype Miri stops at unsupported half AArch64 assembly. Failures are retained and none is labelled a successful Miri run.

## Interrupted build and cleanup

The first measurement attempt stopped at candidate compilation with ENOSPC before collecting timing data. Its error and partial provenance remain in failed-build-attempt/. Source was restored. Under the shared lock, the private Miri target (794,884,549 logical bytes) and this thread's CPU package build cache were removed; cleanup.json records the commands and observations. The same protocol was then restarted. The fixed supplementary diagnostic is reported separately above; neither study was repeated until a favourable result appeared.
