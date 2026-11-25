# Conv reference gap: production auto selection matters

Apple M1 Max, 10 physical/logical CPUs, 32 GiB, macOS arm64, release Rust/LLVM
as recorded in build-metadata.json. This study measures **setup-adjusted total
process CPU per body call**, including all child threads. It does not measure
single-call latency, inference-session or LLM throughput. The body is complete
Kernel.execute for a reused kernel, or factory create + constant-input setup +
execute + drop for a new kernel. All libraries are already initialized for both
boundaries. `new` is neither a cold process nor a first inference.

Three implementations are distinguished: a default build with normal production
auto selection, and native-Generic-GEMM / MLAS factories in the same linked
reference executable. Native Generic still retains BNNS/direct NEON routes.
Default/reference are different feature executables; only Generic/reference share
one executable. Single-arm child processes use that shared executable to obtain
terminal wait4 CPU instead of unreliable short in-process CPU deltas. Thus the
functional prelude runs both implementations in one address space, while the
formal work intervals use separate processes.

## Qualified production results

CPU microseconds per body call below are medians of three setup-adjusted process
values. Ratios are medians of paired repetition ratios: production / reference;
above one means more CPU for production. Full ranges and every raw repetition
remain in summary.json and receipts.json.gz. All listed comparison ranges are
disjoint and pass every qualification bar in all three repetitions.

| Geometry | Reuse: production / reference us | CPU ratio | New kernel: production / reference us | CPU ratio |
|---|---:|---:|---:|---:|
| ordinary 3x3 | 7.425 / 12.647 | 0.584 | 7.906 / 14.300 | 0.552 |
| grouped 3x3 | 49.979 / 27.289 | 1.832 | 50.132 / 29.179 | 1.721 |
| depthwise 3x3 | 10.936 / 4.606 | 2.374 | 11.347 / 5.326 | 2.123 |
| large-weight 1x1 | 7.226 / 11.167 | 0.646 | 7.716 / 11.970 | 0.645 |
| rank3 kernel5, stride2 | 2.563 / 2.190 | 1.170 | 3.041 / 4.186 | 0.729 |

The dilated 3x3 production cells fail the A/A 5% bar in at least one repetition
for both lifecycles. Their raw point estimates (23.352/23.178 us production and
12.692/14.382 us reference for reuse/new) are retained but **unqualified**; they
support no production/reference conclusion. Overall 10/12 production comparisons
qualify. No retry is used to replace the two failures.

The Generic/reference diagnostic qualifies 12/12 cells. Its largest reuse ratio
is 12.497 for the 1x1 geometry (139.341 vs 11.167 us), while the actual default
production build uses *less* CPU than the reference on that same geometry.
The Generic gap must not be described as a production gap. Ordinary 3x3 also
uses less production CPU; grouped and depthwise are the clearest remaining
production gaps in these fixtures. No family graduation follows from one tier.

## A reference mechanism assumption that did not survive inspection

The frozen linked executable's MlasNchwcGetBlockSize returns immediate one;
[nchwc-block-size.disasm](nchwc-block-size.disasm) binds that observation to the
same measured executable SHA. The existing ConvFactory gate requires block>=8,
so all reference cases here use the fallback ConvPlan, **not NCHWc**. Do not
attribute reuse/new differences to NCHWc packed-filter reuse. They include plan,
workspace, construction and teardown costs; no individual component was isolated.

Source inspection identifies candidate work, not a causal timing attribution.
The native depthwise 3x3 stride1 kernel processes 12 of the fixture's 16 horizontal
outputs with vector FMA, and four with scalar boundary/tail arithmetic on every
row. A boundary/tail counterfactual is worth profiling, but must preserve the
existing mixed rounding and skipped-padding semantics. The grouped path also
retains im2col work before native GEMM. These observations justify focused
experiments rather than adding a packed cache or another quota mechanism.

## Shapes and workload

All inputs/weights/bias are contiguous f32 and fixed deterministic finite values.
Weights and biases are declared constant; input is not. Data fit the small
working sets in this study. No large-model cold-weight/DRAM behavior is inferred.

| Case | X NCHW (rank3 shown NCL) | W OIHW | Groups | Stride | Dilation | Padding |
|---|---|---|---:|---:|---:|---:|
| 0 ordinary | 1,8,16,16 | 16,8,3,3 | 1 | 1 | 1 | 1 |
| 1 dilated | 1,8,16,16 | 16,8,3,3 | 1 | 1 | 2 | 2 |
| 2 grouped | 2,16,12,12 | 24,8,3,3 | 2 | 1 | 1 | 1 |
| 3 depthwise | 1,16,16,16 | 16,1,3,3 | 16 | 1 | 1 | 1 |
| 4 pointwise | 1,256,4,4 | 128,256,1,1 | 1 | 1 | 1 | 0 |
| 5 rank3 | 1,8,128 | 16,8,5 (OIL) | 1 | 2 | 1 | 2 |

The common prelude creates, executes and checks every available implementation,
then warms the selected arm. Source reachability recording is separate from
first/second wall diagnostics, and is disabled before all body loops. Wall
prepare/first/second records are unqualified diagnostics, not CPU conclusions.
Output is poisoned once before a positive body; every final element is checked
against f64 and complete bytes are retained. The independent Python reader
reconstructs the f64 oracle for all 18 distinct arm/case outputs. Output hashes
are stable across repetitions, lifecycles, work counts and null labels within
an implementation. Different arithmetic implementations need not match bits.

## Clocks, controls and limits

There are 174 retained calibration processes and 648 formal processes: six
labels (three arms plus an identical A/A label for each), six cases, two
lifecycles, setup/work/double kinds and three rotated repetitions. Calibration
selects fixed calls independently per arm/lifecycle; matching A/A labels use
identical calls. Counts remain fixed throughout the formal matrix.

For every repetition, work CPU must be >=0.1 s, setup <=25% of work, and the
setup-adjusted double/work scale in [1.7,2.3]. Main/null CPU-per-call ratios must
be in [0.95,1.05] in every repetition. Setup means zero *body* calls; the common
functional/warmup prelude remains. Reader controls reject missing/duplicate or
mislabeled processes, impossible physical CPU, missing native GEMM witnesses,
wrong build claims and unstable outputs. Synthetic qualification controls ensure
AA, setup, scaling and nonpositive-body failures remain unqualified. They are
explicitly synthetic and are never performance observations.

CPU comes solely from parent wait4 user+system at terminal child state. Every
process must fit logical_cpu_count * wall + 0.01 s. Rust and MLAS pools are
actually checked at one, VECLIB_MAXIMUM_THREADS=1 is set, and all framework worker
CPU is included; other system-framework workers are not directly observed.
The outer shared flock is held throughout each complete build/study. An inherited
file descriptor reasserts ownership, and an independent lock probe must fail.
Darwin foreign-CPU instrumentation is unavailable: the lock and A/A do not prove
foreign work absent. Load observations are retained, not used as an idle claim.

Absolute RSS is recorded per process. The linked prelude retains both kernels,
so its peak cannot rank standalone memory cost or establish an RSS saving.
Startup, f64 validation and output-file work are included in raw process CPU;
setup subtraction estimates the body under the explicit proportionality tests.
No model, multi-tier, other-ISA, CUDA or universally zero-overhead claim follows.
No runtime or resource policy is changed by this documentation-only study.

## Reproduction and verification

Run verify.py without arguments for the sealed historical data, or with a source
checkout root to additionally compare recorded runtime source hashes. It reads
the raw tar members as data and re-runs numeric/corpus qualification. It never
extracts paths or executes measured binaries. Build/run drivers retain original
local paths and need deliberate adaptation elsewhere. The archived original
reader-control script is retained separately; reader_controls.py exposes its
unchanged controls as a function so verification needs no original /tmp paths. Binaries are identified by
SHA and nonempty symbol inventories; default has no linked MLAS library symbols,
while the reference executable has positive MLAS symbols. Dynamic ORT libraries
are not audited as a whole process, and no ORT CPU EP executes this workload.

The temporary Cargo example was removed after builds. Consumed private executable
copies are removed only after all child handles are terminal and their hashes
recorded; Cargo dependency caches are preserved and cleanup never unlinks the
lock. The historical measurement and later archival cleanup observed different
lock inodes. Their receipts retain both identities; no continuous identity across
the intervening pause is asserted.
