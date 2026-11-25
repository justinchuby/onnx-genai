# im2col uses one horizontal span per kernel column

The runtime change solves the valid sampling interval once and copies contiguous
horizontal spans when stride_width=1. Dilation changes the starting coordinate,
not the sample step. Other strides sample the same valid interval. See [the
principle and bit-preservation argument](DESIGN.md). There is no new allocation,
cache, quota, public API or change to GEMM arithmetic or staged output ownership.
BNNS, direct NEON and pointwise routes retain their existing selection.

## Complete-operator CPU comparison

Apple M1 Max, 10 physical/logical CPUs, 32 GiB, macOS arm64; release compiler and
LLVM recorded in build-metadata.json. Frozen pre-change and post-change factories
are linked into **one default-feature executable**, both with normal production
backend auto selection (Accelerate detected). This is not a forced Generic or
MLAS comparison. The shared GEMM/backend code remains identical. A temporary
module registers the frozen old source for this probe only; both that module
registration and temporary example are restored before measurement.

The body is complete warmed Kernel.execute through a black-boxed trait object,
including its conversions, temporary buffers and output staging. Parent terminal
wait4 user+system captures all child threads. Setup subtraction estimates CPU per
body call; this is not single-call latency or model throughput. Each process
initializes and validates both factories before warming its selected arm. All
complete before/after outputs match bits; Python independently reconstructs f64
results for all 12 arm/case outputs.

Three rotated repetitions retain setup, fixed work, double work and identical
A/A labels for each arm. All 216 formal and 75 calibration processes are retained.
Each repetition requires work CPU >=0.1s, setup/work <=25%, adjusted double/work
in [1.7,2.3] and main/A/A CPU per call in [0.95,1.05]. Every bar must pass for both
arms to qualify. Full ranges and every repetition are retained in summary.json.
Ratios are medians of paired after/before ratios, not ratios of marginal medians.

| Fixture | After CPU us/call | Before CPU us/call | Paired CPU ratio | Qualified | Disjoint full ranges |
|---|---:|---:|---:|---|---|
| ordinary 3x3 | 7.469 | 7.460 | 1.002 | yes | no |
| dilated 3x3 | 17.331 | 23.585 | 0.744 | yes | yes |
| grouped 3x3 | 41.954 | 48.630 | 0.862 | yes | yes |
| depthwise 3x3 | 11.186 | 11.009 | 1.016 | NO: after A/A | no |
| large-weight 1x1 | 7.167 | 7.190 | 0.996 | yes | no |
| rank3 kernel5 / stride2 | 2.544 | 2.532 | 1.005 | yes | no |

The dilated and grouped fixtures use about 26% and 14% less complete-operator CPU,
respectively, under these controls. The other qualified fixture ranges overlap;
they establish no improvement. Depthwise fails A/A for the after arm and remains
unqualified, with no performance inference or retry. These are hot, small finite
fixtures, the same six geometries recorded in [the preceding study](../conv-reference-gap/README.md#shapes-and-workload).
They do not establish DRAM behavior, a whole-Conv-family verdict, other-tier or
ISA performance, LLM throughput, CUDA execution or an RSS saving. This is a new
before/after experiment, not a retry or qualification of the preceding study's
failed production/reference dilated comparison.

Rust pool degree one is checked, VECLIB_MAXIMUM_THREADS=1 is requested, and other
system-framework worker degrees are not directly observed. Total child CPU
includes those workers. Both kernels remain resident through the body, so peak
RSS cannot rank the standalone arms. The shared flock is inherited, ownership
reasserted and independent acquisition refused for the complete matrix; load
observations remain in plan. Darwin foreign-CPU instrumentation is unavailable,
so neither lock nor A/A is an assertion that all foreign work was absent. Each
process also obeys CPU <= logical_cpus * elapsed_wall + 0.01s.

## Correctness, controls and reproduction

The independent signed-coordinate im2col test executes 6912 geometry combinations
with signed zero, infinities, distinct NaN payloads, empty axes, asymmetric/large
padding, dilation, nonunit strides and multiple channels. Every output starts
poisoned. Full sample bits and +0 padding must match, and adjacent canaries remain
untouched. Actual padding-negative-zero and no-write mutations each compile,
select exactly one test and fail its assertion; the exact source is restored.

Default and linked-native Conv each pass 47 tests (one existing benchmark ignored),
MLAS reference Conv passes 11, explicit Generic-driver, default/linked factory
pairing and actual native allocation checks pass. Strict all-target Clippy passes
with and without mlas. Thirteen reader controls retain one accepted real corpus,
eight rejected synthetic corruptions and four unqualified synthetic clock/control
violations. Synthetic data never becomes a measured observation.

Run `python3 verify.py` for historical evidence, optionally passing a checkout
root to additionally check the post-change runtime source hash. Verification reads
tar members without extracting paths or executing native binaries, checks all
hashes and raw outputs, recomputes numeric/control results and replays the reader
controls. Build/run/check drivers retain original machine paths and need deliberate
adaptation elsewhere. The build driver includes the recorded pre-measurement file
mode repair; sandbox hardware-read and initial mode failures launched no child and
produced no discarded performance samples. The historical baseline and binary SHA
are frozen. Existing Cargo caches and the shared lock file are preserved.

The implementation was then rebased onto the recorded integration parent after
Kernel trait and other CPU changes landed. All eleven related build/test/lint
commands pass again; integration-validation.json and separate logs retain this
second check. A pre-existing Constant-test type-complexity lint failure is retained
separately, along with the two lint passes after a local type alias repair and
a focused Constant test pass. The twelve successful commands cover the original
eleven checks plus that focused test. The runtime Conv source SHA is unchanged. The performance matrix
remains historical evidence for its frozen build, not a new timing measurement
on the integration parent.
