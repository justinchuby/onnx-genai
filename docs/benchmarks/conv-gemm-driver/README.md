# One native Conv GEMM driver

The default native Conv implementation had separate ordinary and grouped GEMM
drivers. Group=1 rebuilt its im2col Vec for every batch, while the grouped
pointwise route allocated a col buffer it never used and rechecked geometry
inside each group. Both now use the same group-indexed driver. A unit group has
identical input/weight/output slices and M/K/N to the removed ordinary loop.

The existing pointwise predicate is resolved once per driver call. Such calls
borrow the original input. Other nonempty calls create one col Vec and reuse it
across all batches and groups: im2col overwrites every element before each GEMM.
Empty batches return before backend selection or scratch creation. Zero K still
runs GEMM and bias/ReLU; it is not treated as an empty output.

There is no change to BNNS/NEON selection, backend selection, GEMM reduction,
bias-then-ReLU order, output staging/publication, rank-3 promotion or generic
scalar fallback. The old private helper names are removed; ignored benchmark
callers use the unified signature. No cache, retained buffer, pool, public EP
API or grant/request/refund operation is added. The col Vec remains call-owned
scratch; this step does not establish complete Conv/process memory accounting.

## Checks

The permanent unit matrix runs 540 combinations: groups 1/2/3, batches 0/1/3,
normal/zero input or output channels, pointwise and dilated pointwise geometry,
asymmetric padding with stride/dilation, empty output spatial axes, bias and
ReLU. Small dyadic data make products and sums exact, allowing independent
scalar-reference bit comparisons without assuming all backends share a floating
reduction order. Existing random/tolerance Conv tests remain unchanged.

The ARM64 native-only integration test observes the complete actual Conv.execute
call on the calling thread. Inputs, output backing and factory setup are outside
the count, with an unobserved warm call. It has a positive allocation-size control,
full output checks, and batch counts 0/1/2/17. Group1 uses dilation to exclude
BNNS; group2 has multiple input/output channels per group to exclude the NEON
depthwise route. The serial small fixtures avoid parallel-worker accounting.
Col allocation sizes differ from output staging for every measured batch.
Nonempty complete-execute allocation counts must not grow with batch count.
The grouped 1x1 dilation1 case has no col allocation, while the unchanged
classifier routes dilation2 through one col buffer. This is allocation-call
counting, not total bytes, peak memory, external-library allocation or latency.

The feature-MLAS build selects a different Conv source file. Its tests cannot
stand in for this implementation. Current checks use actual default native builds,
and a separate process explicitly selects the portable Generic GEMM. No MLAS
family graduation or reference removal is claimed.

The controls directory retains the unmodified historical implementation as a
negative control for the actual-execution allocation test, plus temporary
same-image old/new driver comparisons with finite seeded f32 bit patterns over
the same 540 geometries. Production source is restored exactly after controls.
No historical driver or diagnostic branch is shipped. Reproduction drivers
retain their original local paths and need adaptation on another checkout.

Heavy checks are serialized under the shared host flock and coordinated with
the peer chat. No inference/LLM speedup, universally zero overhead, GPU behavior
or CUDA execution is inferred. No CUDA device was used.

## Recorded results and reproduction

The unoptimized dev/test profile on this M1 Max passes 46 native Conv tests
(one existing ignored timing benchmark), the focused 540-case Generic matrix,
both actual-execute allocation tests, and strict all-target CPU Clippy.
For the scoped native fixtures, non-pointwise complete execution has two Rust
allocation calls, pointwise has one, and empty batch has zero. The explicitly
selected Generic backend adds one environment-string allocation per nonempty
call; its counts remain independent of batch count too. Neither result observes
allocations hidden inside system libraries or other threads.

The historical source compiles and fails the single allocation test at group1,
batch2: two col allocations rather than one. An eager-col-only mutation compiles
and fails at the grouped pointwise case: one col allocation rather than zero.
Both same-image seeded-bit comparisons execute all 540 cases and pass, using the
auto-selected native and explicitly selected Generic backends separately.

The initial integration test incorrectly required positive total allocations
for an empty batch. It failed on the legitimate zero-allocation result. That
assertion now applies only to nonempty batches; full outputs and the independent
positive allocator control remain required. Its original source, source hash
and failure log are retained under initial/.

Run `python3 -B docs/benchmarks/conv-gemm-driver/verify.py .` from the checkout root
for hashes, result/selection counts, exact negative controls, historical function
identity and restored source. Omit the checkout argument after later code changes.
The retained validation/control drivers expect `/tmp/nxrt-prefetch-groups-check.py`
for their environment prelude: environment-source.py contains exactly that executed
prelude. Controls also expect baseline.rs.gz decompressed as
`/tmp/nxrt-conv-driver-base.rs`. Adapt paths deliberately before reproduction;
queued.py retains the shared flock wrapper. The original ORT 1.32 path, single
Rayon thread and target/profile settings are in the prelude. No timing instrument
is used by this study.
