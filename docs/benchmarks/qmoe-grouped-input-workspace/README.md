# QMoE reuses one grouped-input workspace

The integer CPU QMoE kernel now declares one StepScoped rows*hidden*f32 gather
region and reuses it across serial expert groups. Singleton groups borrow their
already contiguous input row; a batch with zero or one row declares zero scratch.
This replaces per-expert grouped-input Vec allocations without introducing a
quota API, allocator, Mutex or scheduling authority. The bound has no expert-count
or top-k multiplier: each token selects an expert at most once.

## Ownership and scope

QMoE declares WorkspacePlanning::Metadata. Its sizing hook and execution share
a checked size helper; full byte addressability is checked even when singleton
scratch is zero. The existing ProjectionScratch binds supplied Host storage and
gathers only the used prefix. It performs no whole-lane clearing. Multirow groups
continue to gather; there is no general contiguous-window search from the
previously unadopted MoE input-window candidate.

Only the main compute thread borrows this lane. The prefetch worker still owns
weights only; single-ahead bounds, ascending lease order, active-reader lifetime,
worker join, contribution slots and accumulation order remain unchanged. Output
and FC intermediates remain owned, so no scratch reference escapes execution.
Missing, short or misaligned nonempty workspace fails before expert compute.
Rank, dtype, presence, multiplication and byte overflow are checked in planning;
all pre-existing tensor validation still precedes execution binding.

The real Host pool regression exercises sparse/noncontiguous and full-row-skew
routing, one peak lane, exact output bytes and stable charge over 17 calls per
routing pattern. Warm workspace checkout/compute/return makes zero issuer
callbacks, including zero-byte callbacks; drop/trim returns its charge to zero.
The test explicitly isolates workspace funding with a local kernel weight source.
It does not establish zero callbacks for transient expert weight admission.

Transient weight grants, contributions, routing tables, conversions, FC
intermediates and row-decoder scratch remain outside this migration. Do not
interpret it as a whole-inference zero-callback or full-process memory cap.

The CUDA parity suite's CPU oracle now prepares declared scratch. A new GPU-free
regression runs that oracle for one and six rows against an independent f64
reference. It failed with missing QMoE workspace before the oracle fix and passes
afterward; compiling ignored GPU tests alone could not catch the omission.
No CUDA kernel, allocation policy or device execution is changed or measured.

## Actual public-session evidence

Apple M1 Max / macOS 27.0.1, Rust 1.98.0 / LLVM 22.1.8, release arm64.
The probe times InferenceSession strict prepare -> run, including output drop.
Fixture: com.microsoft::QMoE, 4-bit weights, block size 16, four experts, top-k=2,
normalized routing, identity activation. Alternating rows select expert pairs
0/1 and 2/3. Inline nonpageable weights use the transient path; this does not
measure retained experts or mmap prefetch, full-model generation or GPU overlap.

CPU milliseconds/call: medians of three process medians, nine batches/process.
The first candidate allocated gather scratch even for one row. Its qualified
single-row point estimate was +2.20%, about 1.84 us, with null +1.57%; the
four-thread point estimate was +2.93% with overlapping process-median ranges.
That result motivated the singleton ownership refinement rather than a blanket
zero-overhead claim. Preserve all original data:

| Rows x H x I / threads | Before ms | After ms | After/base | Null/base | Controls |
|---|---:|---:|---:|---:|---|
| 1 x 64 x 64 / 1 | 0.083964 | 0.085807 | 1.0220 | 1.0157 | pass |
| 8 x 256 x 256 / 1 | 2.410000 | 2.391738 | 0.9924 | 0.9912 | pass |
| 8 x 256 x 256 / 4 | 3.056636 | 3.146061 | 1.0293 | 1.0016 | pass |

Independent rebuilt study of the refined source, same probe and fixture:

| Rows x H x I / threads | Before ms | After ms | After/base | Null/base | Controls |
|---|---:|---:|---:|---:|---|
| 1 x 64 x 64 / 1 | 0.085552 | 0.085120 | 0.9949 | 0.9868 | pass |
| 8 x 256 x 256 / 1 | 2.370976 | 2.390071 | 1.0081 | 1.0174 | pass |
| 8 x 256 x 256 / 4 | 3.050727 | 3.065182 | 1.0047 | 1.0019 | pass |

Both studies retain full ranges, wall time, raw batches and control ratios.
All six cells pass duration and control checks. Refined CPU point changes are
-0.51%, +0.81% and +0.47%, with overlapping before/after process ranges and
material null shifts. No distinguishable regression or gain is established on
these fixtures; this is not a universal zero-cost claim. All 54 formal full output
buffers and row argmaxes match within each shape, also across the two studies.

Baseline funded workspace is zero. Refined single-row workspace is also zero;
eight-row H=256 workspace is precisely 8,192 bytes for one or four threads.
The scoped 64 MiB Host domain funds actual transient expert payloads and the
workspace, not the other allocations listed above. No derived payload remains
retained after a call. Workspace charge is stable warm and session/EP drop plus
idle trim returns ledger use to zero. Used-byte stability does not prove issuer
callback counts; the separate recording-issuer test pins workspace callbacks.

## Protocol and reproducibility

Baseline: 911fcbd78ee1eb0a82b471b164aa02cff2d33c32.
Original source: 00004b707e37abe03821209d6e22a9fb03f6eca3.
Refined source: 1cc886fb498f67f1035179c4cd89067a2f4a8fc3.
Both arms are freshly built at the same source path after cleaning affected
release packages, independently for each study. Drivers temporarily add the same
libc dev dependency and restore source, manifest and lockfile before timing.
The null executable is an SHA-verified baseline copy. Baseline/after/null rotate
over three processes per shape (27 formal processes/study). Excluded warmup is
100 ms CPU; excluded pilots choose common repetitions targeting 100 ms/batch.
CPU and unchanged Relu(16384) control batches (nine x 100,000 calls) must last
>=20 ms; both after/base and null/base control median drift must stay within 5%.
Rayon uses the explicit one- or four-thread pool named in each cell.

All heavy work holds /tmp/onnx-genai-hostlock-darwin.flock, coordinating with
llama.cpp. Process snapshots supplement this protocol; they do not prove an idle
host. Absolute peak RSS is recorded without a savings claim. Probes have zero
MLAS symbols and required InferenceSession/main symbols; empty and only-main
negative controls check the symbol audit.

The original/ and refined/ directories commit raw batches, summaries, exact
probes/drivers, source/probe/executable hashes, full-buffer hash verification and
normalized source patches. Each source-reconstruction.json proves its patch
reconstructs the measured files exactly from baseline. Original logs, buffers,
process snapshots and symbols remain in /tmp/nxrt-qmoe-gather-session-perf and
/tmp/nxrt-qmoe-gather-refined-session-perf; consumed binaries may be removed after
SHA verification. Reproduce with cached ORT 1.32.0 in an isolated checkout under
the shared lock. Drivers preserve original paths and modify/rebuild that tree.

## Validation

Full initial native CPU: 2,018 passed / 20 ignored; session: 295. Optional MLAS
CPU: 2,060 passed / 35 ignored; session: 297. Session --tests: 443 passed / two
existing ignored across 25 binaries, including lib tests. Final singleton
refinement: native and MLAS QMoE each 57 passed / one ignored, session 295 passed,
CPU oracle one passed / 57 hardware tests ignored. Strict CPU/CUDA/session
all-target Clippy passes by default, and CPU/session Clippy passes with MLAS.
Formatting and whitespace checks pass. No CUDA, x86, Linux/Windows ARM64 or
whole-model generation was tested. The first integration attempt used a Python
without NumPy; the successful run prepended the existing Miniconda environment
with NumPy/ONNX. No dependency or project configuration was changed to hide it.


After timing, integration with merged plugin/ARM64 work and the reviewed
single-owner API passes native/MLAS QMoE (57 each, one ignored), native/MLAS
session (296/298), the GPU-free oracle and strict CPU/CUDA/session all-target
Clippy. integration-source.json records that checkpoint and all changed Rust
source hashes; the three measured QMoE files are unchanged. These checks do not
reinterpret the recorded timings as a fresh post-integration baseline. The parent
API's quarantine-test teardown also passes all 14 provider tests under local
Miri with leak checking enabled; no production guard is relaxed.


CI's CUDA source guard correctly requires every test in a CUDA-named target to
remain ignored without gpu-tests. The host-only oracle regression is therefore
in the separate qmoe_cpu_oracle target; qmoe_gpu retains all 57 device tests.
Both use the same common::execute_cpu_kernel helper, which honors the CPU
provider's declared workspace alignment and owner lifecycle rather than guessing
an alignment. The host regression passes with gpu-tests both off and on, without
a CUDA library/device. Source scan and checker self-tests plus strict CUDA
all-target Clippy pass. host-target-source.json records this test-only follow-up;
the two measured CPU runtime files remain unchanged. Historical measured CUDA
oracle files and source patches are preserved rather than rewritten.


The CUDA compile Linux lane explicitly executes qmoe_cpu_oracle with the cuda
feature and !cancelled(), independently of earlier compile-step success. It uses
scripts/test_step.sh to reject an empty/all-ignored run. Running that exact step
locally without CUDA reports the named oracle passed and TEST_STEP_EXECUTED=1.
Workflow integrity, required-tier and package/lint coverage checks pass.

### Windows CLI closure: physical external-file identity

The Windows CLI ORT job exposed a separate linked-model cache bug: equal-size
replacement weights could retain the same modification timestamp, so the
non-Unix key reused the old linked directory. The key now combines a physical
file handle's identity with its modification time, queried through the same
open file, using the already locked `same-file` crate. It does not read/copy the
weight bank and runs at model materialization, outside warm inference.
The existing regression now constructs equal size and equal modification time
before replacement; unchanged handles still share a directory and replacements
must obtain a new directory containing the new bytes.

Local validation: 12 island tests and 895 native-backend engine tests pass
(1 ignored), plus strict native-backend engine all-target Clippy. This is Host
validation; the corrected Windows execution remains a CI responsibility. QMoE
and projection production files and historical timing sources are unchanged.
