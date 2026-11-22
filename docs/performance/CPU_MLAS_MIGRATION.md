# CPU EP native capability ledger, and the MLAS absorption roadmap

**Direction (repository owner, final):** the production CPU execution provider
is **native**. It does not proactively link, activate, or route to MLAS, on any
target, in any shipped artifact. MLAS is an **optional research / test /
benchmark reference only** — something we measure our own kernels against and
absorb capability from. The objective is to absorb MLAS's capabilities into
native code until there is nothing left to reference.

Rejecting MLAS-by-default is **not** a re-opening of ORT's built-in CPU
execution provider. That remains forbidden too. Neither route is permitted:
this EP claims its nodes and runs them with its own kernels.

This document is the roadmap for the absorption. It keeps the method from
[`ABSORBING_MLAS.md`](ABSORBING_MLAS.md) unchanged — the same-binary A/B,
port-the-mechanism doctrine is what makes an absorption verifiable — and adds
the per-family inventory, priority order and graduation rule below.

## How to read this document

| term | means |
|---|---|
| **Native** | what production runs. The only thing a shipped build contains. |
| **Mlas** | a route a `--features mlas` *research* build can reach. Outstanding absorption work — never something we ship. |
| **NativeOverMlas** | our chunking / parallelism / repair around an MLAS inner primitive, again research-only. |
| **Graduated** | a family where native is correct and at least as fast as the reference. Nothing is outstanding; the reference is no longer interesting. |

`crates/onnx-runtime-ep-cpu/src/dispatch_ledger.rs` is the machine-readable form
of the same table, and `effective_backend()` reports what *this* build can
actually reach — which, in every default build, is `Native` for every family.

## Why the default is native, and stays native

Two things are true at once, and the direction resolves them in favour of the
second.

1. MLAS is currently faster on several families. On this project's plugin-path
   A/B (AMD EPYC 9V74, AVX2, ORT 1.27, K=N=2048, p50 of 41 interleaved
   iterations) 4-bit `MatMulNBits` at M=128 took **81×** ORT's time natively and
   7.3× with MLAS linked; `QLinearMatMul` u8 at M=128 took **55×** natively and
   9.3× with it.
2. Shipping MLAS to close those gaps removes the only forcing function that
   closes them properly. Every native weakness becomes invisible in the
   configuration users actually run, the EP stops being independently fast
   standing alone, and "absorb MLAS" becomes a project with no deadline and no
   pain. The gaps in (1) are the work item, not an argument for a dependency.

#1091 — *the configuration we measured was not the configuration we shipped* —
is fixed by the other half of this policy: benchmarks that link the reference
must say so, the ledger records which route actually ran, and
`crates/onnx-runtime-ep-cpu-plugin/tests/default_artifacts_are_mlas_free.rs`
proves the shipped artifact contains none of it.

The absorption programme is unaffected by MLAS being non-default. A research
build (`--features mlas`) still puts both implementations in one binary, which
is all the same-binary A/B needs.

## Ownership boundary

This is the part that is easy to state and easy to lose.

| | who owns it |
|---|---|
| Node claiming / partitioning | **Ours.** Our EP claims the nodes it supports and never declines one to make ORT's CPU EP take it. |
| Kernel selection within a claimed node | **Ours.** `dispatch_ledger::PLAN` records which route runs and why. |
| The inner arithmetic of some routes | MLAS, called as a library from inside our kernel. |
| Threading | **Ours** for every family whose `threads` column says so; MLAS partitions only where it owns the whole GEMM. |
| ORT's `CPUExecutionProvider` | **Never.** Not a fallback, not a baseline, not a dependency. |

Enforced by:

- `crates/onnx-runtime-ep-cpu-plugin/tests/plugin_ort_e2e.rs` —
  `every_fixture_loads_with_cpu_fallback_disabled` and
  `no_supported_node_is_ever_left_to_the_ort_cpu_ep` run every fixture with
  `session.disable_cpu_ep_fallback=1`, so an unclaimed node fails session
  creation instead of quietly running on ORT's CPU EP. CI runs this suite in
  **both** feature configurations.
- `crates/onnx-runtime-ep-cpu-plugin/tests/default_artifacts_are_mlas_free.rs`
  — the stronger statement, at four levels: no `default` feature list activates
  MLAS, a default build resolves no `mlas-sys`, the default cdylib's **whole**
  symbol table contains **zero** MLAS symbols, and the wheel never asks cargo
  for the feature. Each probe is checked to be load-bearing before its claim is
  asserted, so a zero count from a probe that read nothing cannot pass for
  evidence.
- `dispatch_ledger::Backend` has three variants — `Native`, `Mlas`,
  `NativeOverMlas`. There is deliberately **no** `OrtCpuEp` variant: the type
  cannot express delegation.

## The dispatch ledger

`crates/onnx-runtime-ep-cpu/src/dispatch_ledger.rs` records, per kernel family,
the route we plan to take and the evidence for it: dtypes, ISA dispatch, thread
model, and the shape gate that chooses between routes.

```console
$ NXRT_CPU_DISPATCH_LEDGER=1 cargo test -p onnx-runtime-ep-cpu \
      --test native_vs_mlas_differential -- --nocapture
```

`dispatch_ledger::render_plan()` prints the table; `effective_backend(family)`
gives what *this build* can reach (every MLAS-bearing plan degrades to `Native`
without the feature, because the native path is always compiled — it is the
correctness baseline).

### What it costs when it is off

Off is the default, and off has to be free, or an observability tool becomes a
performance regression. The recorder is a `static AtomicU8` tri-state rather
than a `OnceLock<bool>`, so the steady-state cost on a dispatch path is **one
relaxed load and a predicted-not-taken branch** — no acquire fence, no opaque
call, no lazy-init guard re-checked forever to answer a question settled by the
first call.

Recording sites take a *closure*, not a value: `record_with(|| Observation::…)`
returns before the closure runs when recording is off. This is not
micro-optimisation — `Observation::gemm`/`elementwise` probe the host ISA and
call `mlas_sys::mlas_threading_degree()` through FFI, so building one eagerly
would put real work on the hot path. `record_with_does_not_build_an_observation_while_disabled`
pins this: it is mutation-verified (making `record_with` take a value instead of
a closure fails it, 10,000 constructions vs 0).

Measured, Linux x86-64, release, 200 M calls:

| | ns/call |
|---|---|
| `record_with` with recording off | **0.87** |

That is ~2 cycles. The smallest GEMM in our own bench suite (`1×256×256`) takes
~17,000 ns, so one recording site is **0.005%** of the cheapest operation it
observes. Production callers are NN and NT GEMM dispatch, the activation
routes, and softmax. Each records per invocation rather than per element;
the source-level guard counts every lazy call and rejects eager recording.

Current plan, condensed. `Native-over-MLAS` means our kernel owns the
structure — blocking, threading, fusion, special-value handling — and calls MLAS
for an inner primitive.

| family | route | graduation status |
|---|---|---|
| `MatMulF32` | MLAS | baseline |
| `GemmF32` | Native-over-MLAS | partial — epilogue ours, inner SGEMM is `MatMulF32` |
| `MatMulNBits` | Native-over-MLAS | partial — int4 acc0 decode absorbed (#1104); prefill still MLAS |
| `QLinearMatMul` | MLAS research route; native AVX2 / NEON | partial — native byte GEMM; packed/requantized MLAS reference remains |
| `Activations` | Native-over-MLAS | partial — chunking, threading, special-value repair ours |
| `Softmax` | MLAS | baseline |
| `Normalization` | Native | no MLAS primitive |
| `AttentionTranspose` | Native-over-MLAS | partial — everything but the two inner GEMMs is native |
| `Quantization` | Native-over-MLAS | partial — block/int4 formats already native |
| `MoE` | Native-over-MLAS | partial — blocked on `MatMulNBits` prefill |
| `Convolution` | MLAS | baseline |
| `Pooling` | MLAS | baseline |
| `Elementwise` | Native | **graduated** — native dense SIMD matched or beat `MlasEltwiseAdd`, MLAS is not called |

## Inventory of MLAS primitives

Bound in `crates/mlas-sys`. "Used" means a route in a `--features mlas`
**research** build reaches it — that is, the family has outstanding absorption
work. No shipped build reaches any of these.

### Used

| primitive | family | notes |
|---|---|---|
| `MlasGemm` (f32) | `MatMulF32`, `GemmF32` | the workhorse; also the alpha/beta epilogue's inner call |
| `MlasGemmBatch` | `AttentionTranspose`, `MoE` | batched QK^T / PV, per-expert GEMM |
| `MlasGemmPackB` / `PackedGemm` | `MatMulF32` | packed-B reuse for repeated weights |
| `MlasQGemm` (i32, packed) | `QLinearMatMul` | integer GEMM with i32 accumulation |
| `MlasQGemmRequantize` | `QLinearMatMul` | requantize inside MLAS, as ORT does (#1125) |
| `MlasComputeSoftmax` (in place) | `Softmax` | contiguous last-axis rows |
| `MlasComputeErf` / `GeluErf` | `Activations` | `len >= SIMD_MIN_LEN` |
| `MlasComputeLogistic` / `SiLU` | `Activations` | with native repair outside ±18 |
| `MlasConvPrepare` / `Run` | `Convolution` | plus plan lifetime management |
| `MlasNchwc*` (block size, reorder in/out/filter, conv) | `Convolution` | the 6 extra op registrations a *research* build gains; production uses `conv_ref.rs` |
| `MlasPool`, `MlasNchwcPool` | `Pooling` | f32 Max/Average |
| `MlasQNBitGemm` (+ available / packB / workspace) | `MatMulNBits` | prefill `M >= NXRT_SQNBIT_DECODE_MIN` |
| `MlasSetThreading` | all | the MLAS pool runs under *our* thread budget (#1054) |

### Available but deliberately not used

| primitive | why not |
|---|---|
| `MlasComputeTanh` | native beat MLAS-plus-clamp by up to **1.39×** at 256 Ki (#1121) — de-graduated from MLAS deliberately |
| `MlasComputeLogistic` for `Sigmoid` | same measurement as Tanh; the route was removed |
| `MlasComputeActivation` (fused) | our fusions cover more op pairs and keep the epilogue native |

### QLinearMatMul reference packing ownership

Optional QLinearMatMul packed weights obtain their actual packed-buffer bytes
from the provider's `DerivedWeightMemory` Host authority before allocation.
The payload owns its grant; an active call keeps both alive independently of
the kernel cache. Retention permission reserves no idle bytes. Temporary Host
capacity refusal leaves construction retryable and executes the existing
unpacked reference route; allocator/configuration errors propagate. First
construction is serialized, and a different weight identity declines packing
before allocating a discarded replacement. No arithmetic or ISA route changes.

The regression `provider_qlinear_pack_owns_its_host_grant` executes through the
provider registry: before this change the ARM64 macOS fixture retained a
6,528-byte pack while the supplied Host ledger reported zero. The corrected
path charges that payload, reuses its charge on a warm call, and releases it on
last-owner drop. `qlinear_pack_capacity_refusal_retries_and_active_owner_keeps_grant`
covers KV competition and retry with byte-identical outputs;
`qlinear_pack_concurrent_builders_share_one_granted_payload` checks eight
simultaneous builders obtain the same payload under a one-pack budget.

The existing executor constant-state seam shares one cache across activation
shape variants of the same session node. Adoption requires the same immutable
grant source, constant B and admitted retention; even independently constructed
sources over the same Host books/holder cannot exchange caches. Different
nodes and sessions retain their own state. The session's node prediction uses
the largest possible signedness layout for one logical B matrix, including
sign-domain translation, vector B and leading unit dimensions; no fixed count
of prefill/decode copies remains. Prediction and packing share the same matrix
shape rule. Overridable initializers and distinct weight batches are excluded.
The real executor regression previously charged 6,528 then 13,056 bytes when
activation shape changed; it now retains one pack across three shapes and
checks separate-session charges and complete release on drop.

Each shared cache enrolls weakly in the same Host pressure domain. Ordinary KV
reservation/growth and another weight admission can evict its entire idle
packed payload; active Arc owners prevent eviction.
Reclamation removes the entry under its state lock and releases the payload
and grant after unlocking. Construction is serialized separately, so admission
can synchronously visit the requesting cache without deadlocking. An evicted
entry can be rebuilt on later use, or run unpacked while capacity is occupied.
The bounded `qlinear_pack_pressure_reclaims_idle_without_deadlocking_builders`
parent runs a named child that exercises self-callback admission, KV reserve
and growth, active-owner exclusion, zero-pressure no-op, unrelated authorities,
rebuild and weak registration lifetime. The old implementation fails the idle
KV-reservation case; a deliberate admission-under-state-lock fault is rejected
by the named parent after its child reaches self-cache admission and times out.
The shared-source policy/lifetime tests cover adoption, idle reclamation after
the originating kernel drops, rebuilding, and an active payload outliving the
remaining kernel. Eight concurrent shape-kernel builders share one granted
payload under a one-pack budget. This does not graduate QLinearMatMul or retire
its MLAS reference. The governed amount is the packed payload, not all
transient operand/output or accumulator storage.

### Retired reference capabilities

`Elementwise` is graduated. Add now always uses its existing native vDSP/dense
or general dtype/broadcast path, including when the remaining MLAS reference is
linked. Its former optional route, Rust/C binding, f32/fp16 MLAS Add templates
and Add-only ARM dispatch/header/source leaves have been deleted. No compatibility
API remains. Native same-shape and suffix-broadcast route guards run in both
feature configurations, with IEEE special-value, tail and empty-storage checks.
Add validates both input views and the output before native pointer access,
preserves typed validation causes, and rejects non-Host-accessible storage.
The MLAS multiplication primitive remains required by the independent SiLU reference.

### Gaps — MLAS has no primitive

`Normalization` (LayerNorm / RMSNorm / SkipLayerNorm / GroupNorm), KV-cache
layout and masking, the `MoE` router, block/int4 quantize–dequantize, sequence
ops. These are native today and there is nothing to absorb: they are already
ours.

## Replacement priority

Ordered by *decode-path impact × distance from parity*, which is also
`KernelFamily::ALL`'s declaration order.

1. **`MatMulF32` / `GemmF32`.** Everything else in a transformer is downstream of
   it, and `AttentionTranspose`, `MoE` and `GemmF32` cannot fully graduate until
   it does. Hardest: MLAS is a mature, hand-tuned, multi-ISA SGEMM.
2. **Activations.** Already the closest — Tanh and Sigmoid *have* graduated. Erf
   and exact Gelu remain: on the `narrow` arm of the re-measurement below MLAS
   is **1.34–1.56×** on Erf and **1.10–1.12×** on exact Gelu (`wide`:
   **1.34–1.64×** and **1.12–1.18×**).
3. **Normalization.** Already ours; listed so it stays measured, not to be
   absorbed.
4. **Softmax.** *Largely closed.* This entry used to read "the widest measured
   gap and the easiest structural win", because the native route was a scalar
   `f32::exp` loop ~9× off MLAS. `softmax_avx2` (#1234) and then #1416 landed on
   `main`, and the re-measurement below puts the ratio at **0.82–0.87** across
   the three softmax cases on the `narrow` arm (**0.80–0.83** on `wide`). The
   then-vs-now attribution is a separate, width-matched comparison: measured on
   `wide` against the archived #1173 numbers, the MLAS control arm is stationary
   to within **4.2%**, so **1.24–1.27× is attributable native work**, matching
   what #1416 claimed for the row kernel. What remains is the
   `exp` polynomial and the two-pass row traversal, on the decode critical path
   for every token. Still short of the 1.05 graduation bar, but no longer the
   headline gap.
5. **Attention transpose / KV layout.** Native except for the two inner GEMMs;
   graduates with (1).
6. **Quantization / `MatMulNBits` prefill / `MoE`.** Highest value, most work.
   #1104 absorbed int4 acc0 decode (1.36× / 1.56×); prefill is the remainder,
   and `MoE` follows it.

Native byte-GEMM column-zero-point compensation consumes each A row sum where
it is used, rather than allocating an m-element sum buffer for every batch.
An all-zero B zero-point vector skips that compensation entirely. The raw dot
kernel, ISA selection, threading and each K-reduction order remain unchanged;
all arithmetic is still wrapping i32. Seeded-accumulator tests cover zero,
sparse/extreme and mixed column offsets against the integer oracle.
`qgemm_compensation_allocation` observes actual serial ARM64 QLinearMatMul
execute calls with per-row A quantization, which cannot select MLAS even in a
reference build. It checks that allocation count does not grow with batch
count, using a positive allocator control and every output byte. Caller-owned
inputs/output and factory setup are excluded, with one unobserved warm call.
For the ARM64 fixture, the old zero-B-offset implementation requests 25, 26,
and 41 allocations for 1, 2 and 17 batches in both debug and release. The new
implementation requests 24 in every cell for B offsets 0 and 1, in debug and
release with and without the optional reference. The reduction is one allocation
per native GEMM batch, with no retained replacement buffer.
These are calling-thread allocation-call counts for complete serial executes,
including remaining parameter/geometry storage, rather than allocated-byte or
RSS bounds. The regression removes batch-wise compensation churn. AVX2 retains
its independent A-pair and B-panel scratch. No latency or end-to-end throughput
claim follows from removing the sum buffer.

Native ARM64 per-row activation-scale requantization now uses the shared
8-lane NEON NaN/clamp/ties-to-even/pack core, forming `(a_scale * b_scale) /
y_scale` in registers with separate multiply/divide steps. Short rows and tails
remain portable, with no temporary scale vector. A mode-specific calling-thread
witness prevents shared-scale or concurrent tests from satisfying the per-row
coverage check. Positive finite rounding-boundary counterexamples, nonfinite/
subnormal scales, offset slices, saturation, batches and parallel rows preserve
output bytes. The [M1 Max serial-row diagnostic](../benchmarks/2026-11-22-per-row-requant.md)
retains 2,646 samples over three independent process sweeps: 42/42 CPU and
42/42 wall A/A cells qualify. The initial candidate’s failed cell remains in
its separately archived corpus. Qualified widths 65–2048 show CPU portable/native
ratios 1.549–1.859; shorter widths offer smaller gains. Test-only witness bookkeeping is included
in native timings. These row measurements establish neither GEMM/model gains
nor family graduation; the optional reference remains.

Native SiLU shares the runtime chunk scheduler with host-pool preference and
nested-pool avoidance. ARM64 retains its exp polynomial and stable scalar
formula, but records exact exceptional-lane positions in four u32 masks.
Empty repair sets skip rereads, full sets use a contiguous stable repair walk,
and partial sets decode only their set bits. The 128-element bound derives
from four lanes and u32 mask capacity; the bitmap is 16 bytes of local storage,
with no heap scratch or retained weight copy. Scalar tails and numerical guards
remain intact. [M1 Max diagnostics](../benchmarks/2026-11-21-silu-density-repair.md)
show ordinary/sparse latency gains over the previous native path; wide-pool
per-cell CPU timing fails its A/A control and does not establish CPU savings.
The [standalone clock-boundary diagnostic](../benchmarks/2026-11-21-cpu-clock-scope.md)
finds both process CPU APIs sensitive to completion scope. Waiting through worker
quiescence reduces the largest excursions but still fails some A/A cells; neither
a clock API swap nor batching alone qualifies as a CPU-cost measurement fix.
The [direct MLAS rational counterexample](../benchmarks/2026-11-21-silu-native.md)
near x=-16.435 still exceeds the existing native error budget. The MLAS research
route remains; other hardware/ISA and global graduation work are outstanding.
The CPU crate retains the absorbed SIMD coefficients' complete MIT notice in
`MLAS_LICENSE`, independently of eventual vendor/reference removal.

ARM64 native Erf and exact GELU now share the existing AVX2 coefficient set and
use four-lane NEON, including partial tails and fused BiasGelu. Runtime feature
detection and the existing short-slice scalar seam remain. The
[M1 Max absorption diagnostic](../benchmarks/2026-11-21-neon-erf.md) retains 15120
samples across 180 cells with native A/A ratios 0.994–1.010. It measures 14–18×
less wall time for Erf and 12–14× for exact GELU than the former scalar bodies,
but MLAS remains faster in most cells. This closes the ARM64 scalar capability
gap without graduating the family or removing its optional reference. BiasGelu
bitwise consistency is tested; its latency and end-to-end models were not
measured. The existing numerical bounds and NaN/signed-zero policies remain.

The [NaN propagation follow-up](../benchmarks/2026-11-21-neon-erf-nan.md) removes
redundant NEON masking/repair while preserving every f32 NaN payload/sign and
before/after output bits. Qualified M1 Max cells with at least 257 elements
show about 7–12% less wall time than the previous NEON body. All 18900 samples
are retained; 179/180 cells pass A/A and one large runtime GELU cell is
performance-unqualified. MLAS still wins some qualified cells, so the family
and its optional reference remain.

A [paired-vector loop experiment](../benchmarks/2026-11-21-neon-erf-paired.md)
was rejected: all 180 A/A cells pass, but small M1 Max Erf/GELU calls regress
and larger cases show no repeatable improvement. The candidate and 18900 raw
samples remain reproducible; production retains the single-vector loop. Generated
GELU code has a larger stack frame and more spills, so simple unrolling is not
an evidence-backed next step for this implementation.

Native Softmax row APIs take a buffer and column count, deriving row count from
complete slices instead of accepting an ignored independent `n`. They return
an explicit error for nonempty zero-width input, partial rows, or unequal source
and destination extents before writing. Empty storage is a no-op. A checked
`NonZeroUsize` width and exact-row iterators establish each raw kernel's extent
from its borrowed row. Native and research A/B callers use the same geometry
validation, including when MLAS is unavailable; no old signature or compatibility
alias remains. These boundary changes do not remeasure the frozen latency tables
below.

ARM64 contiguous native Softmax now uses a four-lane NEON max/exp/sum/normalize
core from eight elements onward, sharing the AVX2 polynomial coefficients while
preserving gradual underflow and the original scalar short-row path. Both
in-place and disjoint outputs share one row core. The
[M1 Max experiment](../benchmarks/2026-11-21-neon-softmax.md) retains all three
mechanisms and 31752 samples; 107/108 final A/A cells qualify. Qualified large
cases are 2.25–2.77x faster than the former scalar body, but MLAS still wins
those comparisons. This closes the ARM64 scalar-exp capability gap without
graduating Softmax or removing its optional reference. Other ISAs, model
throughput, and CPU-cost improvements are not inferred.

The generic native Pool window walker now reuses four rank-sized coordinate
buffers and a row-major cursor, preserving sum order and both MaxPool index
orders. Unrequested index output is no longer staged. The
[actual-execution allocation proof](../benchmarks/2026-11-21-pool-window-workspace.md)
observes 8 allocations per call across its four geometries, versus 261–146885
before, in optimized native builds; the MLAS-feature build reaches the same
native LpPool path and also observes 8. This is bounded coordinate workspace,
not whole-call allocation freedom, a latency comparison, or Pool graduation.
The optional MLAS Pool/NCHWc references remain.

Generic f32 GEMM's thin-M column-parallel path now borrows the original B
stride and caller-owned output rows. It retains the same scalar register tiles
and K-panel accumulation order, replacing full B/output element copies with
row-reference metadata. This is a memory-ownership improvement in the portable
fallback; see the [allocation proof and scoped diagnostic](../benchmarks/2026-11-21-generic-gemm-borrowed-columns.md).

The native `QLinearMatMul` byte GEMM now has AVX2 and row-reusing ARM64
NEON implementations. NEON borrows the original weight bytes and retains no
packed copy. Register tiles share each B load across up to four rows; tile
height preserves at least one task per worker, with the compiler-vectorized
portable loop for single-row or narrow work. This is progress toward absorption,
not graduation of the MLAS packed/requantized route. See the scoped
[M1 Max measurements](../benchmarks/2026-11-21-qgemm-neon.md). `Convolution` and `Pooling`
still need their own capability and performance comparisons.

The native ARM64 shared-A-scale requantization epilogue also uses eight-lane
NEON rounding/clipping, with exact scalar tails and non-finite handling. Per-row
A scales keep the scalar epilogue; other architectures retain their portable
path. This introduces no packed copy or retained buffer. Its separate hot-cache
row diagnostic and exact-byte contract are documented in
[the requantization note](../benchmarks/2026-11-21-qlinear-requant-neon.md).
This does not graduate the MLAS packed/fused GEMM or establish a session speedup.

The `bench_qgemm_ab` kernel diagnostic runs native, portable and a repeated
native A/A control in the same explicit Rayon pool, with rotating order. It
checks exact output equality before timing and reports elapsed and process CPU
time (CPU time is explicitly unavailable on non-Unix platforms). It measures
only `sum_k (a - za) * b`: row sums, B zero-point correction, requantization,
load-time packing and session overhead are outside its scope. Earlier versions
of this diagnostic ran portable outside the selected pool and compared its
partial accumulation against a complete native GEMM. Those measurements must
not establish a speedup or absorption verdict without remeasurement.

## The graduation rule

> A native route replaces an MLAS route **only** when it is (a) correct against
> the differential test, and (b) at least **5% faster, repeatably, outside the
> noise floor, across the supported shape/dtype/ISA range**. Otherwise MLAS
> stays.

Unpacked:

- **Correctness first.** `crates/onnx-runtime-ep-cpu/tests/native_vs_mlas_differential.rs`
  holds the two routes against each other in one binary, including special
  values (±inf, NaN, extreme rows) where MLAS clamps and we do not. Byte
  identity is the default requirement; an f32 GEMM change may not preserve it,
  in which case quantify the deviation against a reference and say so.
- **5%, not "faster".** Measured with
  `cargo bench -p onnx-runtime-ep-cpu --bench native_vs_mlas`, which runs both
  routes interleaved in one process and reports the median of 21 repetitions,
  then prints the verdict per case: `native-graduates` /
  `native-faster-but-under-5%` / `keep-mlas`. **One invocation's verdict is not
  a graduation.** That 21-rep median still moves between invocations by more
  than 5% on most cases in the table below, so a single `native-graduates` line
  is a sample rather than a result.
- **On wall time, with CPU time beside it.** The two routes do not use the same
  number of threads — our `x86_sgemm` parallelises over column strips while MLAS
  declines to parallelise some shapes — so a CPU-time-only comparison would
  block a graduation that genuinely lowers latency. Wall time decides. Process
  CPU time is printed as `cpu_ratio` because the opposite failure is equally
  real: a route that "wins" by recruiting the whole machine regresses every
  concurrent session on the box, and the bench labels that
  `native-graduates-but-costs-more-cpu` rather than passing it silently.
- **Repeatable, outside noise — and the noise has to be measured, not assumed.**
  Cross-worktree comparison on a shared runner showed a uniform 0.70–0.82×
  offset on *byte-identical* kernels — larger than most kernel wins. One binary,
  interleaved, or it does not count. But interleaving only equalises placement
  *between the two arms*; it does not make the ratio robust, because the two
  routes scale differently with width, so anything that changes the machine's
  effective width moves them by different amounts. Two requirements follow.
  1. **Report the ratio's spread across invocations, and graduate only when that
     spread is smaller than the claimed win.** A 5% win inside a 30% spread is
     not a win.
  2. **Discard invocations that did not get the CPU.** Take
     `(utime + stime) / wall` for the bench process from `os.wait4` rusage and
     drop any invocation materially below the median of its siblings. A rep that
     was descheduled is not a slow rep, it is an untrusted one — a distinction
     owed to #1809, where adding this guard moved an A/A null from 52% to
     0.04–0.56%. Note what it does and does not do here: **in-process
     interleaving is what protects the ratio**; this guard only catches
     *differential* descheduling between the arms. Contention that lands evenly
     on both arms passes it wholesale. It is what lets you keep measuring on a
     shared box instead of waiting for a quiet one that never arrives, not a
     substitute for the interleaving.
  3. **Better still, measure foreign CPU on your own core set.** #1814 landed
     the measurement on `main` after this table was taken (now the
     `onnx-runtime-hostmon` crate, and reported as `host_foreign` by
     `bench_generic`).
     It reads busy jiffies on the process's `Cpus_allowed_list` and subtracts
     the process's own CPU, so the remainder is *foreign load on the cores you
     were actually confined to*. That closes the hole in the guard above
     directly: contention landing evenly on both arms is invisible to a
     differential CPU-efficiency check but shows up here as a positive foreign
     column. It also demonstrates why the host-level gates are insufficient —
     one foreign thread on a 32-CPU box does not move `/proc/loadavg` field 4,
     yet on a barrier-synchronised dispatch it cost a clean 2× at N=2. The
     tables below predate it and were guarded by the weaker method; a
     re-measurement should use it.
- **At a stated width.** The verdict is a function of how many cores the process
  is given, for the reason in the previous bullet: the two routes do not scale
  alike. Measured directly, six idle physical cores in one L3 domain against all
  32 logical CPUs, same binary, arms interleaved — `matmul_f32 16×512×512` reads
  **1.581 narrow and 0.866 wide**, and `decode 1×2048×2048` reads **1.117 narrow
  and 0.934 wide**. Both cross the 1.05 line on width alone. Record the width;
  never compare rows taken at different ones; and **verify the width actually
  held** — #1815 found a sibling harness whose two arms ran on different CPU
  sets while reporting the same thread count, so read `Cpus_allowed_list` from
  `/proc/<pid>/task/*/status` during the run rather than trusting the `taskset`
  you typed.
- **Across the supported range.** A win at one shape and a loss at another is
  not a graduation; it is a shape gate, and it belongs in the ledger's
  `shape_gate` column with both routes kept.
- **A measured "not worth it" is a good outcome.** #1104's brief assumed the win
  was layout; measuring first showed it was register blocking and made the
  proposed workaround unnecessary rather than clever.

Precedent in both directions already exists: `Elementwise` graduated, Tanh and
Sigmoid graduated *away from* MLAS routes that were slower once their clamping
had to be repaired (#1121), and `MatMulNBits` decode graduated while its prefill
did not.

### Current gap, measured

`cargo bench -p onnx-runtime-ep-cpu --bench native_vs_mlas`, Linux x86-64,
AVX2+FMA, one binary, interleaved, median of 21 reps per invocation. Native
route is `SimdX86` (the AVX2 `x86_sgemm` we ship, not the portable fallback) for
GEMM. Times are per unit of work — per multiply-accumulate for GEMM, per element
otherwise. `ratio = mlas / native` in wall time; **above 1.0 means native is
faster**.

Re-measured on `main` after the decode-placement corrections (#1729, #1794,
#1811), which changed how many physical cores a process actually gets and so
changed every absolute number in the previous revision of this table — that
revision had been taken when the decode pool put 16 workers on 8 physical cores.

Taken with `scripts/bench_native_vs_mlas_width.py`, **6 reps per arm,
alternating which arm runs first**, on a host that was explicitly *not* quiet
(~4–5 cores of unrelated load throughout). Two arms:

- **narrow** — `taskset -c 16,20,22,26,28,30`: six distinct physical cores, one
  sibling each, all inside a single L3 domain, all measured idle before the run.
  **6/6 reps trusted.**
- **wide** — `taskset -c 0-31`: every logical CPU, which is what the previous
  revision used. **5/6 reps trusted**; one discarded at `cpu_per_wall` 13.14
  against a median of 14.17.

`spread` is `(max − min) / median` of the ratio across that arm's trusted reps.

**Read the `ratio` column as a median of per-rep ratios, not as the quotient of
the two columns beside it.** Each rep contributes its own `mlas / native`, and
the median is taken over those; the `ns/unit` columns are independently the
medians of their own times. Because a median does not distribute over division,
the two disagree slightly — `decode 1×2048×2048` shows 0.0684 / 0.0617 = 1.109
by column but 1.117 by rep. The per-rep form is the correct one to quote,
because it pairs each MLAS invocation with the native invocation it was
interleaved against, which is the whole point of interleaving; the quotient of
medians silently pairs measurements that were never adjacent. The gap between
the two is itself a smell — where it is large, the arm is unstable.

**The mask was verified to hold, not assumed.** #1815 observed the neighbouring
`bench_generic` harness spawning its ORT arm *outside* the affinity confinement
it applied to the native arm, so the two arms ran on different CPU sets while
reporting the same thread count. That hazard applies to any `taskset` claim,
including this one, so it was checked rather than asserted: sampling
`Cpus_allowed_list` from `/proc/<pid>/task/*/status` 40 times across a live
narrow-arm run gives **478 thread-observations, every one of them
`16,20,22,26,28,30`** — the five `mlas-sys-ws-N` work-stealing threads, the five
`nxrt-task-N` workers and the bench's own threads alike. Both routes were
confined identically. (The single `0-31` observation is the `taskset` process
itself, before it execs.) A width comparison is only as good as the confinement
it claims, and this one is checkable.

**narrow arm — 6 physical cores, one L3, idle.** This is the arm to quote,
because it is the one whose reps agree.

| family | case | native ns/unit | MLAS ns/unit | ratio | spread | cpu_ratio | verdict |
|---|---|---|---|---|---|---|---|
| `matmul_f32` | decode 1×2048×2048 | 0.0617 | 0.0684 | **1.117** | 21% | 0.875 | native-graduates-but-costs-more-cpu |
| `matmul_f32` | decode 1×4096×4096 | 0.1454 | 0.1161 | 0.800 | 20% | 0.742 | keep-mlas |
| `matmul_f32` | 16×512×512 | 0.0282 | 0.0435 | 1.581 | 41% | 1.083 | **unstable** |
| `matmul_f32` | prefill 128×2048×2048 | 0.0125 | 0.0090 | 0.687 | 42% | 0.566 | keep-mlas |
| `matmul_f32` | odd 37×1023×511 | 0.0266 | 0.0133 | 0.523 | 82% | 0.434 | keep-mlas |
| `softmax` | decode 1×32000 | 0.8092 | 0.6578 | 0.821 | 4% | 0.825 | keep-mlas |
| `softmax` | attn 32×512 | 0.6264 | 0.5454 | 0.869 | 12% | 0.865 | keep-mlas |
| `softmax` | prefill 128×4096 | 0.6192 | 0.5407 | 0.871 | 6% | 0.867 | keep-mlas |
| `activations` | erf 4 Ki | 1.1148 | 0.8321 | 0.746 | 17% | 0.746 | keep-mlas |
| `activations` | erf 1 Mi | 0.4263 | 0.2651 | 0.641 | 6% | 0.646 | keep-mlas |
| `activations` | gelu-exact 4 Ki | 1.3832 | 1.2235 | 0.889 | 4% | 0.889 | keep-mlas |
| `activations` | gelu-exact 1 Mi | 0.4751 | 0.4374 | 0.912 | 4% | 0.905 | keep-mlas |

**wide arm — all 32 logical CPUs**, same binary, same session, interleaved with
the rows above.

| family | case | ratio | spread | verdict | vs narrow |
|---|---|---|---|---|---|
| `matmul_f32` | decode 1×2048×2048 | 0.934 | 13% | keep-mlas | **flips** |
| `matmul_f32` | decode 1×4096×4096 | 0.515 | 74% | keep-mlas | |
| `matmul_f32` | 16×512×512 | 0.866 | **134%** | **unstable** | **flips** |
| `matmul_f32` | prefill 128×2048×2048 | 0.595 | 41% | keep-mlas | |
| `matmul_f32` | odd 37×1023×511 | 0.168 | 55% | keep-mlas | |
| `softmax` | decode 1×32000 | 0.826 | 22% | keep-mlas | |
| `softmax` | attn 32×512 | 0.800 | 12% | keep-mlas | |
| `softmax` | prefill 128×4096 | 0.832 | 10% | keep-mlas | |
| `activations` | erf 4 Ki | 0.748 | 20% | keep-mlas | |
| `activations` | erf 1 Mi | 0.611 | 8% | keep-mlas | |
| `activations` | gelu-exact 4 Ki | 0.890 | 5% | keep-mlas | |
| `activations` | gelu-exact 1 Mi | 0.846 | 6% | keep-mlas | |

Three things fall out of putting the two arms side by side, and all three are
about the *method*, not the kernels.

**1. Two cases change verdict on width alone.** `decode 1×2048×2048` reads 1.117
on six cores and 0.934 on thirty-two; `16×512×512` reads 1.581 and 0.866. Same
binary, same machine, same half-hour — only the CPU mask differs. The mechanism
is in the rule above: `x86_sgemm` parallelises over column strips and MLAS
declines to parallelise some shapes, so the two arms do not scale alike, and
interleaving the *routes* inside one process does nothing to protect against
this because it changes both routes at once. **A graduation claim without a
stated width is not checkable.**

**2. `16×512×512` disagrees with itself on both arms.** Its per-rep verdict
alternates between `keep-mlas` and `native-graduates` from a byte-identical
binary — 134% spread wide, 41% narrow. It is marked unstable rather than given a
verdict. This is the entire argument for the spread requirement: at a 5%
threshold and a 134% spread, the bench will eventually hand you a
`native-graduates` line, and nothing in that single invocation's output would
tell you it was a coin flip. Had the previous revision of this table been run
once more, it could have graduated a route on this row.

**3. The narrow arm is the more trustworthy one overall, though not on every
row.** Its spreads run **4–82%** against the wide arm's **5–134%**, and it lost
no reps to the efficiency guard. Isolation beat parallelism: six idle cores in
one L3 domain produce a more repeatable number than thirty-two logical CPUs
shared with someone else's build. But the honest qualifier is that `odd
37×1023×511` is an 82% coin flip on the narrow arm too — wider than eleven of
the twelve wide-arm rows — so "narrow is tighter" is an aggregate statement, not
a per-row guarantee. Prefer the arm whose reps agree **on the row you care
about**, and say which one you used.

Net: **`decode 1×2048×2048` is the first f32 GEMM case to show a real native
win**, at 1.117 on the narrow arm — but it is `native-graduates-but-costs-more-cpu`
(cpu_ratio 0.875, so native burns more CPU for that latency), it does not hold
at 32 threads, and its 21% spread is wider than its 12% win. **Under the rule
above that is not a graduation.** It is the strongest candidate on the board and
the right next thing to re-measure properly. Everything else stays `keep-mlas`.

**Before re-measuring that row, read #1827.** The explanation given above for
why it does not hold at 32 threads — "the two routes do not scale alike" — is
true but may be materially under-specified for this particular row. `m == 1`
takes `sgemm_simd_m1`, whose strip decomposition carries a `.max(8)` floor that
appears to cap it at `ceil(n / 256)` tasks *independently of pool width*: at
`n = 2048` that is 8 tasks at both 6 and 32 threads, which would mean the wide
arm added twenty-four spinning workers and no parallelism. If that holds, the
wide number is not "native scaled worse", it is "native was capped while the
pool spun", and this row is **better** than the table credits it for. That is
derived from source and **not yet measured** — #1827 states the falsifiable
prediction (native wall time flat from 8 threads upward at `n = 2048`, with the
knee moving to `ceil(n / 256)` at other shapes). Settle it before spending a
graduation argument on this row in either direction.

**Softmax moved, and the MLAS column tells you how much of it was ours.** An
earlier revision of this table recorded softmax at 5.17 ns/element and a ratio
near 0.10 — "~9–10× off, the widest measured gap" — because the native route was
a scalar `f32::exp` loop. `softmax_avx2` (#1234) and then #1416 have since
landed on `main`.

The trap in reading that as kernel progress is that the machine changed
underneath *both* arms between the two measurements. What makes it decomposable
is that **no vendored MLAS kernel has changed since #1173** — the only
`mlas-sys` edits are the additive straggler handshake in `work_stealing_pool.rs`
(#828, diagnosed in #1714). That provenance is a *prior*, not the proof: the
"it only adds waiting" argument is one-directional, and in any case a control
that silently got slower would inflate the ratio just as badly as one that got
faster. The load-bearing evidence is the **direct then→now measurement of the
MLAS column itself**, below — the control is shown to have held still rather
than assumed to have. Compare **like width to like width**: the previous
revision and the `wide` arm above were both taken on all 32 logical CPUs.

| softmax case | MLAS then → now (control) | native then → now | ratio |
|---|---|---|---|
| decode 1×32000 | 0.6038 → 0.6015 (**−0.4%**) | 0.9108 → 0.7198 (**1.27×**) | 0.66 → 0.83 |
| attn 32×512 | 0.6636 → 0.6357 (−4.2%) | 1.0083 → 0.7944 (1.27×) | 0.66 → 0.80 |
| prefill 128×4096 | 0.6368 → 0.6265 (−1.6%) | 0.9312 → 0.7527 (1.24×) | 0.68 → 0.83 |

(Ratios in this table are quotients of medians on both sides, since the archived
revision recorded only medians; that is why the "now" column differs in the last
digit from the median-of-ratios in the wide table above.)

The control is stationary to within 4.2% across all three, so the native column
is real work: **1.24–1.27×**, which is exactly what #1416 claimed for the row
kernel. That is the cleanest attribution in this document, and it only exists
because the comparison arm was left untouched. Prefer this decomposition over
raw absolute columns whenever the table is re-measured across an infrastructure
change.

**One activation row wants a second look.** `erf 1 Mi` is the only case where
the **native** arm went the wrong way at matched width: 0.4551 → 0.5167, 13.5%
slower. The nearest scatter figure available is the `wide` arm's **8%** spread,
but that is a spread of *ratios* and this is a move in a *native time*, so the
two are not strictly commensurable — read it as "larger than the run-to-run
scatter of anything else in this row", not as a significance test. And its MLAS
control is *not* stationary here
(0.3565 → 0.3174, 11% faster), so unlike softmax this cannot be attributed to
our kernel from these numbers alone — a 25% relative swing with both arms moving
is as consistent with a bandwidth or placement effect as with a regression.
**Flagged, pinned re-measurement needed, not yet a finding.**

The f32 GEMM rows carry **no attribution at all**, and it is worth being explicit
about why rather than quietly presenting the improved ratios. `decode 1×2048×2048`
went **0.238 → 0.885** at matched width (both computed as a quotient of medians,
since the archived revision recorded only medians — hence 0.885 rather than the
0.934 median-of-ratios in the wide table above), which looks like a large native
win, and the native column did improve (0.1309 → 0.0715, 1.83×). But the control
moved further: MLAS went 0.0311 → 0.0633, i.e. **2.0× slower with no kernel
change**.
When the control arm moves by more than the effect, the arms are no longer
comparable across revisions, and the only defensible statement is the current
ratio at a stated width with its spread attached. GEMM remains the widest gap
(0.17–1.58 depending on shape and width) and **the top priority**, and the shape
dependence is unchanged: `SimdX86` is closest at the large shapes it was tuned
for and furthest at small and awkwardly-shaped ones, where MLAS's packing and
blocking amortise better.

One caveat a future agent should not have to rediscover: this bench calls
`gemm_with_backend` directly, so **neither** route gets the prepacked constant
weights that `matmul_dense` gives them in a real session. That is deliberate —
it compares the kernels rather than the caching — but it means the bench
understates both routes at inference time, and it is not the instrument for
judging the prepack path.

The comparison against #1116's dense-f32 work is not contradictory for this
reason: that task measured the shipped path with prepacking on a quiet laptop
and found parity at prefill, while this measures the raw kernels on a noisy
shared box. Both are true of what they measured.

### The absorption that already shipped — and a table this document got wrong

#1091 identified, and #1116 ported, the mechanism behind MLAS's
`SgemmKernelM1Avx` — stream B in place at M=1 rather than packing panels that
are reused zero times — into `x86_sgemm::sgemm_simd`. It landed behind
`ONNX_GENAI_CPU_MM_SIMD_M1_GEMV`, default **off**, pending a measurement.

**That toggle no longer exists, and it had already been removed before the first
revision of this document described it as live.** #1183 shipped the GEMV on by
default. Today `sgemm_simd` calls `sgemm_simd_variant(a, b, c, m, k, n, true)`
unconditionally; `use_m1_gemv` is a plain function parameter that only the
in-process A/B harness ever passes as `false`. **No environment variable reaches
that route**, and grepping the EP source for the variable finds only prose.

**Retraction.** The first revision of this section carried a three-row table
captioned "same binary, same session, toggle the only difference", reporting
`decode 1×2048×2048` at 0.146 with the toggle off against 0.337 with it on, and
called turning it on "the obvious next slice". Setting that variable cannot
produce two different columns, because nothing reads it — an A/B run that way
measures the same route twice. **The table is withdrawn.** It is exactly the
failure this document's own graduation rule warns about — an arm that was not on
the route it was labelled with — committed by the document that wrote the rule,
and it survived review because a plausible number in a well-formed table is not
self-evidently unmeasured. That is the argument for falsifying the *route* and
not just reading the *result*: a poisoned-binary or control-arm check would have
caught it immediately, and no amount of extra reps would have.

The comparison is still worth making, and there is a correct instrument for it:
`bench_f32_gemm_ab` in `kernels/matmul.rs` drives `sgemm_simd_variant` with the
flag set both ways inside one process (`GEMM_AB_ARM=simd_packed` against
`simd_gemv`, default `both` interleaves them), and carries a built-in control —
the M≥2 rows, which an M==1-only route cannot move, so a run that shifts them is
a run to discard. Anyone re-opening this question should use that harness, and
should quote #1183's shipped end-to-end result (through an ORT session, `ours/ORT`
p50 7.57 → 1.18 at 1×2048×2048 f32, with the M=128 prefill control unmoved at
1.03) rather than the withdrawn table.

The consequence for the roadmap is that the "obvious next slice" is **already
done**. The f32 GEMM position stated above was measured with the GEMV in place,
so it is the post-absorption gap, not the pre-absorption one.

## What linking the reference costs (research builds only)

Linux x86-64, gcc 13.3.0, rustc/cargo 1.97.1, release cdylib
(`libonnx_runtime_ep_cpu_plugin.so`), 32 cores. Re-measured against merge base
`c55a3fab3`.

| | base `main` | this branch, default (shipped) | this branch, `--features mlas` (research) |
|---|---|---|---|
| cdylib size (release) | 6,236,344 B | 6,242,024 B | 7,733,368 B |
| delta vs base | — | **+5,680 B (+0.091%)** | +1,497,024 B (+24.0%) |
| MLAS symbols linked | **0** | **0** | 841 |
| exported (dynamic) symbols | 12 | **12** | 12 |
| clean release build | 67.2 s | 64.0 s | 71.3 s |

**The default column is the claim to check.** Adding the ledger costs **5,680
bytes and no exported symbols** — `.text` +2,448 B, `.strtab` +1,890 B (symbol
names), `.eh_frame` +420 B, `.rela.dyn` +192 B, everything else under 100 B. The
public ABI is byte-for-byte the same 12 symbols. That is the "unavoidable tiny
metadata" bound: no MLAS, no new dependency, no new export, +0.09% of an
artifact that is already 6 MB. Build time is unchanged (the 3 s difference is
below the run-to-run spread on this host).

The research column's ~70 s is paid once per target directory, and only by
someone who asked for a research build. The vendored C++ does not change, so
`mlas-sys` is never recompiled by ordinary Rust edits.

Nobody shipping this project pays either cost, and requiring a C++ toolchain is
not a condition of building or using the CPU EP. That is a second, smaller
reason the default is native: the production build has no C++/asm dependency at
all.

## ARM64 research builds: the link failure, and the fix carried here

Recorded because it is a real finding, because the fix is the one piece of
non-research code in this change, and because the *reason* it went unnoticed for
so long generalises.

**The defect.** `crates/mlas-sys/build.rs` assembled the MSVC ARM64 `.asm`
sources and had **no GNU/clang branch** for the GAS `.S` sources in
`lib/aarch64/`. Those files define `MlasSgemmKernelZero`, `MlasSgemmKernelAdd`,
`MlasGemvFloatKernel`, `MlasConvSymKernelNeon` and ~30 more. `sgemm.cpp` and
`convsym.cpp` reach them through *data tables of function pointers*
(`MlasConvSymS8DispatchNeon` and friends), so the C++ **compiled** and only the
**link** broke — on Linux ARM64, Windows ARM64 and macOS ARM64, in whatever
downstream crate happened to produce a cdylib, far from the file at fault.

**Why nothing caught it.** `scripts/check_cross_compile.sh` runs `cargo clippy`,
which stops at type-checking. *A clippy-only cross-compile check cannot see a
missing symbol.* Nothing ever linked an ARM64 MLAS binary, so nothing ever asked
the question. That is the generalisable lesson: a check that does not reach the
linker does not test linking.

**The fix, in this change.** `build.rs` grew `compile_aarch64_asm()`, which
assembles 17 GAS `.S` kernels for every non-MSVC aarch64 target — the MSVC
branch's 18-file list minus exactly one:

| excluded | why |
|---|---|
| `HalfGemmKernelNeon.S` | fp16 arithmetic needs `-march=armv8.2-a+fp16`, whose accepted spelling differs between GNU `as` and Apple's assembler, and nothing compiled here references `MlasHalfGemmKernelNeon` |

The `Smmla`, `Ummla` and `Bf16` kernels are not assembled by *either* branch and
need `+i8mm`/`+bf16` if they ever are. Excluding `HalfGemmKernelNeon` is safe
*and* checked: if a future change compiles a dispatcher that needs it, the link
error names the symbol and the fix is to add the file with its `-march` group.

**Why this is in scope for a research-only change.** Requirement: the explicit
research feature must actually work where it claims to. `--features mlas` is now
a documented entry point, and a documented entry point that cannot link on half
the supported targets is not a reference — it is a footnote. The fix is confined
to `crates/mlas-sys/build.rs`, runs only when `target_arch == "aarch64"` and
`target_env != "msvc"`, and is unreachable from a default build, which never
compiles `mlas-sys` at all.

**How it is tested.** `crates/mlas-sys/tests/aarch64_assembly_is_built.rs`
tests the *shape* of the defect rather than one instance of it. It parses
`build.rs` and asserts (a) every file in the MSVC ARMASM group has a same-stem
`.S` in the GAS group or an explicit, documented exclusion, and (b) every file
either group names actually exists in the vendored tree. So "one dialect wired
up, the other forgotten" cannot recur silently, and the lists cannot drift from
the vendored sources. Both tests run on x86-64, so they need no aarch64 host —
but they also cannot prove the link succeeds. Full link verification still
requires an aarch64 runner; that limit is stated in the test's own comments.

**Scope of the original defect.** No shipped artifact was ever affected — a
default build links no MLAS on any target, so this could not reach a user. It
only ever blocked research builds.

## Native compact NBits operand ownership

The native 2-bit direct GEMV/GEMM and non-VNNI x86 accuracy-4 packed-nibble
kernels consume the original ONNX wire layout. In flat execution they now
borrow contiguous host `u8` values, `f32` scales, and optional `u8` zero points,
including nonzero tensor offsets. Constant and dynamic operands both borrow
only for the duration of the call; no source slice is retained in a cache.
This removes the otherwise identical resident copy for constants and the
repeated temporary copy for dynamic operands. It does not change arithmetic,
ISA selection, activation quantization, or the broader MLAS graduation status.

Half/bfloat scales and strided inputs still require preparation. Constants
reuse that prepared allocation; dynamic inputs own it only during execution.
An active NUMA/SPMD placement scope also retains the existing first-touch
preparation, even on a pool with one node. Layout-transforming int4/int8 packs
are separate work and are not made borrowed by this change. All eight native
prepared-pack slots now own Host grants covering their retained buffers; build
admission also covers simultaneous conversion/placement sources. Their typed
cache registers with the same authority pressure mechanism as dense expansions.
An executing call holds an `Arc` to its pack: pressure evicts only a slot with no
external readers, drops its buffers and grant outside the cache lock, and the
next call can rebuild under fresh admission. Destroying the kernel does not
refund a pack still held by a caller. The normal cache read takes a short mutex
and clones an `Arc`; it allocates or copies no weight data. The existing separate
cold-fill gate remains, so memory admission and construction do not run under a
payload lock. This is a lifecycle change, not a throughput claim.

`native_pack_pressure_preserves_readers_reclaims_idle_and_rebuilds` exercises all
eight factory-wired slots using real Host/KV reservations, including pressure
from another thread. Existing native-pack tests continue to verify payload and
conversion-peak byte accounting, refusal before construction, failed-builder
refund, warm reuse, dynamic call ownership, and the actual selected operator
cache. Compact borrowing and numerical kernels are unchanged. Native layout slots and their cold-fill gate now share one owner among the
shape variants of an executor node, using the existing constant-state hook.
Adoption verifies fixed preparation parameters, constant-input policy and the
exact immutable grant source; separate nodes/sessions keep separate state.
The active-reader and idle-pressure lifetimes remain unchanged. The
[host regression](../benchmarks/NBITS_NODE_PACK_SHARING.md) checks a real
executor, all eight native layout slots and independent concurrent builders.
F32 NK expansions now share that node state too, including the existing dense
cache and its cold-fill gate. Adoption additionally requires the same immutable
dense-byte limit, so a disabled or smaller retention policy cannot inherit a
larger cache. The [dense ownership regression](../benchmarks/NBITS_DENSE_NODE_SHARING.md)
checks three actual executor shape variants, session isolation, pressure and
concurrent builders. The MLAS research layouts now share explicit node-scoped state and participate in Host pressure reclamation as described below.

## Reference reachability and numerical comparisons

Linkage is not a particular ISA, selected backend, or blocked-layout capability.
The A/B GEMM list includes every implemented route available in the build:
Generic, runtime-supported x86 SIMD, Apple Accelerate, and explicitly linked
MLAS. MLAS SGEMM can be compared on ARM64 too; its x86 microkernel ID is zero
there because that ID is architecture-specific instrumentation. An explicit
`NXRT_CPU_GEMM_BACKEND=mlas` research override can select the library on those
targets without changing the Apple default. Add/pooling route tests observe the
selected native or research path rather than assuming linkage leaves vDSP/BNNS
selected. The NCHWc rewrite consumes a resolved block width: tests exercise
8/16-lane graph rewrites and an unavailable layout without pretending the host
can execute blocked convolutions. Production still uses the actual MLAS query.

MLAS is also not a mathematical oracle. On Apple ARM64, the native scalar Tanh
at `f32::MIN_POSITIVE` preserves the input while the MLAS rational differs by
28 ULP. Scalar Sigmoid(-88.5) produces about 3.672302e-39 while MLAS rounds to
zero; their ULP distance is large because the outputs are tiny. Treat these as
finite approximation/underflow comparisons, not changes to signed-zero or
nonfinite semantics. No kernel arithmetic is changed to make two algorithms
imitate one another.

The activation differential test keeps the existing bitwise special-value and
one-ULP interior guards when the native AVX2 approximation runs. It keeps
nonfinite/signed-zero checks on every target. Both finite routes are also
checked against the existing independent f64 oracle and per-family accuracy
bounds, reused from the native accuracy tests. Scalar native finite
underflow/tail fixtures additionally match the rounded f64 oracle bitwise. This
preserves the native quality guarantee while exposing reference rounding;
feature linkage alone cannot certify numerical equivalence or performance.

## Opting *in*, for research only

MLAS is off everywhere by default. To reach the reference deliberately:

```console
# Differential correctness, native vs the reference, one binary:
cargo test -p onnx-runtime-ep-cpu --features mlas --test native_vs_mlas_differential

# The A/B graduation benchmark:
cargo bench -p onnx-runtime-ep-cpu --features mlas --bench native_vs_mlas

# The plugin cdylib, for symbol-level comparison:
cargo build -p onnx-runtime-ep-cpu-plugin --features mlas --release

# A research wheel. Not publishable; release automation never sets this.
NXRT_EP_CPU_RESEARCH_MLAS=1 pip wheel python/nxrt-ep-cpu
```

`nxrt_ep_build_features()` reports `"mlas"` or `""`, so a packaged cdylib can be
identified after the fact, and `check_wheel.py` compares it against what the
build asked for. Every shipped wheel reports `""`.

Note that a research build **claims more nodes** than the shipped one: with MLAS
linked the EP registers 6 additional NCHWc reorder ops and uses `conv.rs` instead
of `conv_ref.rs`. That is a difference to keep in mind when reading A/B numbers —
compare families, not whole-model claims.


## Rules that apply to every absorption

Carried forward from [`ABSORBING_MLAS.md`](ABSORBING_MLAS.md), unchanged:

- **Any resident buffer must be governed** (#1056): declared to the memory plan
  before allocation, in the bytes actually allocated, and declinable. Dense
  expansions use `WeightCache` with `DerivedWeightMemory`; each payload
  owns its Host grant until the final reader drops it. Native compact packs
  also own grants; an estimate alone does not fund an allocation.
- **Anything cached per kernel instance is retained at least twice** — the
  executor's `KernelCache` is shape-keyed, so prefill and decode are separate
  instances. Share eligible weights through the executor node's constant state
  using the exact immutable `CpuMemoryResources` grant source, rather than a
  process-global raw-address cache. Borrow unchanged source layouts when placement permits.
- **The memory-cheapest technique wins ties.** The MLAS int4 route holds a packed
  copy at ~2× the int4 bytes; on a 14B model that is 16.7 GB and the memory plan
  declines it (#1051), so the model that most needs the speed is the one that
  cannot afford it. A technique that allocates nothing is available everywhere.
- **A numerical change shipped as a performance change is how a correctness
  regression enters unnoticed.**


## Apple transposed-B dense prefill

The Apple native backend now exposes checked Accelerate NT SGEMM, reading
`B[N,K]` directly through the same NT capability predicate used by x86 and the
optional MLAS reference. Eligible grouped/dense NBits prefill reuses its granted
NK expansion across calls and activation shapes instead of rebuilding a KN
transpose for every prefill. Refusal and zero-retention behavior still follow
the existing cache, grant and row/KN fallback policies. NN and NT now both record
through the existing lazy dispatch ledger.

NN and NT may use different floating-point reductions. Equal matrix layouts do
not establish bit identity, and the old universal bit-identity comments are
removed. The [Apple NT evidence](../benchmarks/APPLE_NT_GEMM.md) records finite
f64-oracle bounds, actual NN/NT bit differences, offset/tail cases, a nonfinite
smoke test and real executor prefill/decode reuse. This is a numerical backend
extension, not a claim of unchanged model token IDs or measured throughput.

The common NN/NT BLAS boundary validates exact extents, checked products and
signed 32-bit ABI dimensions, handles zero axes without entering BLAS, and
returns actionable errors. Its validated operand type also serves GEMV; ABI
limits are enforced only for BLAS, preserving the native large-GEMV usize
extent domain. The optional MLAS reference is unchanged and no family is
thereby graduated.


### Resolve NT implementation and reuse policy together

The transposed-GEMM predicate and unchecked backend dispatcher are replaced by
one resolved `NtGemm` value with private fields. Resolution includes backend
availability (and runtime AVX2/FMA eligibility for the native x86 implementation)
and its prefill weight-reuse requirement. Execution consumes that value. Generic
NN remains available when NT cannot resolve; no compatibility functions remain.

The [real-execute comparison](../benchmarks/APPLE_NT_EXECUTE_AB.md) preserves an
initial transient-NT candidate with qualified wall regressions in its zero-cache
128x1024x1024 fixture. The Apple descriptor now requires admitted NK retention,
using the existing constant-input and dense-byte-limit rule rather than a model
or geometry threshold. Disabled/insufficient retention and mutable metadata keep
the established KN route. Other NT backend policies are unchanged. Protocol 2
records successful NT calls so the zero-retention null cannot silently measure
NT. Original results and failed A/A cells are retained separately.

These are instrumented real-kernel execute observations on one Apple tier, not
model throughput, production-latency guarantees or MLAS graduation. The separate
non-dyadic numerical contract and all grant/pressure/fallback guards remain.


### Node-scoped ownership for optional SQNBit reference packs

The optional MLAS SQNBit store is now owned by a node's shared constant state,
matching native layouts and dense expansions. The executor's existing adoption
hook shares it across eligible shape variants only after fixed preparation
parameters, constant-input policy, dense-byte limit, MLAS retention verdict and
exact immutable grant source agree. Operand addresses distinguish layouts inside
that owner; matching addresses or physical Host books never discover another
node or session's cache.

There is no process-global store, test-only thread-local replacement, installer
key list, or executor blanket-clear API. Retiring the original builder leaves
adopted variants warm; the final owner releases membership. Active packed-payload
readers keep their own grants until they drop. Cold-fill gates still serialize
identical builds without holding a value-map lock during admission or refunds.
Dynamic and serialized-input paths retain their existing eligibility rules;
serialized prepacked inputs are not newly eligible for constant-state adoption.

The regressions check explicit sharing after builder retirement, deterministic
weak-owner teardown in the recycled-address test, independent-node charges on the
same source operands/Host books, concurrent preparation and lock-safe refunds,
and a real executor running three activation shapes plus an independent session.
The executor test reruns the survivor after peer teardown and checks warm charges
and complete final refund. This changes ownership only; it does not change packed
arithmetic, claim performance parity, graduate a family or remove the remaining
optional reference dependency. Reference maps now support idle pressure eviction
through the same Host authority, as described below.

Validation on Apple M1 Max / ARM64 macOS, Rust 1.98, using cached ORT headers
and libraries: native CPU 1,969 passed / 20 ignored, native session 274 passed;
optional-MLAS CPU 1,997 passed / 34 ignored, optional-MLAS session 276 passed.
Strict all-target Clippy passes for both crates in both configurations. After
strengthening scale-key coverage and adding the session test's unsupported-tier
qualification, both named controls selected one test and passed again, and both
MLAS all-target Clippy checks passed again. The qualification did not skip the
session control on this host. Rust formatting, dispatch manifest/reachability,
platform naming and feature-gate coverage checks pass. These are correctness and
ownership results, with no CUDA execution or new performance measurement.


### Idle pressure reclaims SQNBit reference layouts

The node's MLAS layout maps now register weakly with the supplying
`DerivedWeightMemory` pressure domain. Redundant per-kernel strong `OnceLock`
memos are removed. Every executing call clones its layout `Arc` while holding
the corresponding map lock, which is also the lock used to decide whether an
entry is idle. Pressure removes only entries whose sole strong owner is that
map, releases all map locks, then drops payloads and refunds their grants. It
never takes a cold-fill gate; a build can safely trigger pressure on its own
empty map. Full-width and sharded entries account for their actual packed and
retained metadata bytes; MLAS workspace/scratch remains outside this scope.

After eviction, standard wire inputs can use the established native fallback
while Host capacity is occupied, then rebuild under fresh admission. Serialized
MLAS bytes retain their existing actionable capacity failure and retry because
they cannot be interpreted as ONNX wire weights. Active readers retain their
grants even after the node owner drops. The weak pressure registration does not
keep a retired node alive and follows adopted node state after its builder drops.

Contiguous immutable operands use complete operand-address layout keys inside
the node owner. Noncontiguous immutable operands use an explicit `NodeConstant`
source variant, so deleting the memo does not sacrifice their preparation reuse.
This logical slot relies on the existing constant-input contract and explicit
node adoption; dynamic operands never enter it, and no cross-node address lookup
is restored. Warm lookups respect the current call's sharded/full-width decision
and cannot bypass the asymmetric M=1 numerical guard.

A bounded subprocess regression drives real factory-created kernels through
sharded, full-width, serialized and strided-scale layouts. It verifies self-cache
admission, another node's idle donation, zero-pressure no-op, unrelated authority
isolation, warm MLAS execution while a reader holds the existing payload,
active-reader refusal, cross-thread KV reservation and growth, fallback/refusal
and retry, source-owner retirement and final grant refund. Standard-wire cases
also race four execute readers against repeated Host pressure. Grant/refund
callbacks are checked outside map and gate-registry locks. This is lifecycle
validation, not a performance or graduation claim.

Validated on Apple M1 Max / ARM64 macOS with Rust 1.98 and the cached ORT root:
native CPU 1,969 passed / 20 ignored, native session 274 passed; optional-MLAS
CPU 1,998 passed / 35 ignored, optional-MLAS session 276 passed. Strict all-target
Clippy passes for both crates in both configurations. The bounded pressure
parent is a passing ordinary test; it explicitly launches the named ignored
child and checks its success, so that child is not uncovered by the aggregate
ignored count. After adding warm prefill/unscoped-decode layout switching under
a one-pack budget, the parent selected one test and passed again, with explicit
markers proving all four layouts ran on this host; final CPU MLAS all-target
Clippy also passed again. Formatting and dispatch/reachability/platform/feature
coverage checks pass. No other ISA's execution or CUDA behavior is established.


### MatMul preparation obeys the captured retention verdict

Ordinary SGEMM packing is retained derived-weight state. The shared
`MatMulPrepack` f32 and once-widened-f16 MLAS packing entries now require both
constant B and an admitting retention verdict. Previously these two entries
looked only at constant-input status, so a provider configured to retain nothing
could still keep a packed panel. Declined calls take the existing unpacked/half
computation paths; admitted calls keep their existing packed reuse. MatMul,
Gemm and FusedMatMulBias consume these same preparation entries.

Native noncontiguous f16 preparation had the same retention gap. Its private
API now returns a grant-carrying reader: an admitted cache shares its existing
copy, while a declined call owns a transient until computation finishes. It
must not return `None` merely because retention is declined: the existing
constant-weight rescue branch still needs actual operands to compute. The BNNS
column-major zero-copy route remains unchanged. This makes the temporary
lifetime explicit without changing the arithmetic primitives.

Three regressions fail on their corresponding old behavior: retained f32 pack,
retained widened-f16 pack, and retained strided-f16 copy despite decline. Final
controls exercise both verdicts, repeated calls, changed activations and admitted
reuse. The strided control also executes the Apple rescue path and checks that
decline produces the expected result without filling either persistent half or
dense-f32 storage. Their short dyadic corpus is exactly representable; its exact
comparisons do not establish broad packed/unpacked bit identity or model token
parity. Named bounded child processes keep these new real-kernel calls from
perturbing existing libtest-wide platform dispatch observations.

This closes a retention-policy bypass. Ordinary constant f32 materializations
and native contiguous-half copies now obtain Host grants as described below.
SGEMM packed panels and transpose payloads still need authority/lifecycle work;
ordinary node-owned sharing is consolidated below. No
performance improvement is established by these retention checks.

Validation on Apple M1 Max / ARM64 macOS with Rust 1.98 and cached ORT:
native CPU 1,970 passed / 20 ignored and session 274 passed; optional-MLAS CPU
2,001 passed / 35 ignored and session 276 passed. Both crates/configurations
pass strict all-target Clippy. After isolating all new execution controls from
shared platform probe counters, the final `honors_retention` filter selected
three tests and all passed again; final MLAS CPU all-target Clippy also passed.
Formatting and dispatch manifest/reachability/platform/feature checks pass.
These results establish the tested retention behavior, not physical allocation
limits, production speed, full-model numerics or another ISA's execution.


### Ordinary MatMul constant f32 copies own actual Host grants

MatMul, Gemm, FusedMatMulBias and FusedGemm factories now consume immutable
`CpuMemoryResources`, replacing their bool-only/default preparation wiring.
Load-time preparation uses the same provider resources. The FusedGemm factory
previously constructed an ungoverned default prepack regardless of its provider;
it now captures the supplying source and retention policy too. The existing
dense-preparation predictor covers all four consumers in their registered
domains, for both constant operands. Node ownership below removes the historical
shape-instance multiplier.

The prepack's widened/densified constant slots use `WeightCache` and the
provider's `DerivedWeightMemory`. Full f32 payload admission precedes allocation
and conversion. A sealed operand reader carries the owned payload/grant through
computation, whether cache retention is allowed or declined. Cache removal and
kernel destruction do not refund an executing reader. Independent idle copies
can be reclaimed by ordinary KV reserve/growth through the existing weak
pressure domain. Warm fills use the cache's existing singleflight mechanism.
The legacy family retention verdict remains separate from the block/NBits dense
expansion ceiling; no implicit new 256-MiB limit is introduced for ordinary
MatMul. An explicitly ungoverned default source remains local ownership.

Contiguous f32 constants still borrow original storage and claim zero bytes.
Dynamic activation conversion remains scratch, outside this weight-grant scope.
A materialized constant must fit its actual Host grant even when retention is
disabled; capacity refusal reports the MatMul constant-input admission operation
and preserves the typed memory cause. It does not silently allocate an uncharged
full matrix. Later calls may retry after capacity returns.

Conversion can fill an already admitted destination directly. The shared dtype
wrapper uses the same converter, preserving its existing F16C/NEON f16 and x86
bf16 helpers. Scalar strided conversion reads original elements into f32 output
without a second full matrix in the source dtype, including f64. Rank-sized
index bookkeeping is outside payload accounting. This changes materialization
ownership, not GEMM arithmetic, and is not a latency/throughput claim.

A dependent legacy transpose still uses the conversion buffer's address as its
key. Its memo now holds a source reader until that address-keyed entry retires.
Pressure therefore cannot free/recycle the source address underneath a retained
transpose. Such dependent sources remain pinned until the transpose owner drops;
transpose payload grants and reclaiming that dependency as a unit remain work.
Ordinary MLAS SGEMM packed panels, graph activations, outputs and other scratch
are not covered by this f32-copy grant claim. Native half copies are funded in
the next step below.

The unchanged public-provider regression fails on parent 2aa11cfc7: a real
MatMul retains a 24-byte f32 copy while Host books report zero. The corrected
regression drives all four factories with retention on/off, KV competition,
refusal, retry and final release. Additional controls check call-owned/retained
reader lifetime, borrowed f32 at zero capacity, eight simultaneous builders under
one-payload capacity, the transpose/source dependency, and conversion with
positive/negative strides, offsets, dtype variation and rejected destinations.
No CUDA device or other-ISA execution is inferred from these host checks.

Validation: native CPU 1,977 passed / 20 ignored and session 274 passed;
optional-MLAS CPU 2,008 passed / 35 ignored and session 276 passed. Strict
all-target Clippy passes for both crates in both configurations. Seven selected
ownership/factory/conversion/dependency/prediction controls pass, with explicit
nonzero selected counts. Formatting and dispatch manifest/reachability/platform/
feature checks pass. Host: Apple M1 Max ARM64 macOS, Rust 1.98, cached ORT root.
The parent comparison selected the exact named provider test and failed on the
0-versus-24-byte assertion, rather than treating a compilation error or zero-test
filter as evidence. The temporary parent worktree was restored and removed;
raw validation transcripts remain separate from performance evidence.


### Native noncontiguous f16 payload funding

`WeightCache` now stores flat copyable payload elements, sharing its
existing admission, singleflight and idle-pressure implementation between f32
expansions and native `u16` half-bit copies. A required noncontiguous constant
f16 B copy obtains actual Host funding before allocating its payload, even when
retention is declined. Its reader keeps the grant until the last owner drops.
The existing parallel 2-D and scalar N-D copy routes write directly into the
admitted destination; empty layouts avoid zero-sized parallel chunks. Both
Apple rescue callers propagate admission errors instead of returning zero
results or allocating an uncharged fallback. BNNS zero-copy transposed input
and GEMM arithmetic remain unchanged.

On parent 3e2588329, the exact unchanged public-provider regression executes one
selected test and fails with ledger usage 0 versus 306 expected bytes. The new
controls check retained/transient lifetimes, negative strides and offsets, eight
concurrent fills under one-payload capacity, active-reader exclusion, KV idle
reclamation, refusal/retry/refund, invalid sources, empty payloads, and actual
Apple half rescue results. Baseline worktree is removed; logs remain available.
This is ownership/correctness evidence on Apple M1 Max ARM64, not a performance
claim or evidence about CUDA/other ISAs. Ordinary SGEMM panels, transpose
payloads and scratch remain separate work; shape-variant ownership is
consolidated below.

Validation: native CPU 1,981 passed / 20 ignored and session 274 passed;
optional-MLAS CPU 2,012 passed / 35 ignored and session 276 passed. Strict
all-target Clippy passes for both crates/configurations. Formatting and dispatch
manifest/reachability/platform/feature checks pass. These test totals include
the four new half-copy controls and the existing real rescue control, whose
Host grant assertions now run under both retention verdicts.


### A checked allocation plan for ordinary SGEMM panels

Before attaching Host admission to ordinary MLAS SGEMM panels, their allocation
size must be a reliable preflight value. The old constructor used unchecked
logical multiplication, an unchecked native padded-size query, allocation
`unwrap`/assertions and no error return. `PackedBPlan` now binds dimensions and
native layout without allocating payloads; packing consumes that checked layout
and returns `SgemmPackError`. Both the plan and pack expose owned heap bytes.

The shim derives padding/alignment from MLAS's own definitions and checks every
rounding/product before querying the native size. It refuses disagreement with
the standalone generic layout. Rust checks source extent and addressable
allocation size; empty operands allocate a one-byte sentinel without calling
native packing, and an empty packed reduction writes zeros without reading it.
The shim now uses MLAS's canonical half definition instead of redeclaring it.

MatMul's f32 and once-widened half callers propagate actionable typed errors.
Cold fills serialize separately from immutable warm lookup; a failed build is
not memoized. F32 warm reuse validates source extent and packed geometry. The
once-widened half caller plans before materializing its source. These are
allocation-contract changes; the following step adds actual Host funding and
idle reclamation for ordinary panels and half-packing scratch.

Validation on Apple M1 Max ARM64 macOS / Rust 1.98: native CPU 1,981 passed
/ 20 ignored and session 274 passed; optional-MLAS CPU 2,013 passed / 35 ignored
and session 276 passed. Binding suite: 54 passed / 3 ignored, followed by five
explicitly selected SGEMM controls (four new controls plus the existing parallel
SGEMM check). Strict all-target Clippy passes for bindings and both CPU/session
configurations; formatting and dispatch manifest/reachability/platform/feature
checks pass. The unchanged exact constructor regression fails on bc4bac366
with the old length assertion unwinding. After the comparison, shared-target
stale artifacts were rejected, the bindings rebuilt from the current source,
and all five selected controls passed with their names/counts checked. The
temporary baseline worktree is removed. Logs are correctness evidence, not
performance or CUDA/other-ISA validation.


### Funded ordinary SGEMM panel ownership

Ordinary f32 SGEMM panels and panels widened from constant half weights now use
the same `WeightCache` authority as dense f32 expansions and native half copies.
Its owned-payload contract verifies actual allocation bytes for flat vectors or
MLAS-aligned panels. Remove the dedicated SGEMM OnceLock/build-lock mechanism;
executing Arc readers retain the payload and grant together, while idle cache
membership is reclaimable by ordinary Host/KV pressure. Source, dtype and K/N
geometry distinguish payloads. Distinct sources obtain distinct funded packs,
never each other's values; warm same-source hits skip planning/cold filling.

Half packing reserves panel bytes plus the f32 conversion peak before building
either payload. Only panel bytes remain after the conversion drops. A denied
optional prepack performs no conversion or packing and falls through to existing
unpacked computation without memoizing refusal. Retention-off behavior and
numerical primitives stay unchanged. Default native builds remain MLAS-free.
The existing dense expansion ceiling remains a policy for its original consumers,
not a new limit on ordinary MatMul preparation.

Controls exercise real native panel execution through live readers, exact Host
charge, concurrent single construction, active-reader protection, KV idle
reclamation, refusal/retry/refund, and half conversion peak boundaries. Public
provider execution runs both f32 and f16 inputs with retention on/off, sufficient
and zero capacity, and competing KV ownership, checking packed and unpacked
results. The prior distinct-source regression now checks both products and
reuse rather than merely rejecting the second source. Legacy transpose grants,
other scratch and performance measurements remain separate work. Ordinary
shape-variant ownership is consolidated below.

Validation on Apple M1 Max ARM64 macOS / Rust 1.98 / cached ORT: native CPU
1,981 passed / 20 ignored, session 274 passed; optional-MLAS CPU 2,017 passed
/ 35 ignored, session 276 passed. Strict all-target Clippy passes for both
crates/configurations, as do formatting and dispatch manifest/reachability/
platform/feature checks. The same public-provider fixture fails on 2fcd486f1
with ledger 0 versus 320 native panel bytes. After that exact baseline test,
the CPU crate was explicitly cleaned/rebuilt and the three panel lifecycle/peak/
concurrency controls plus the exact provider control ran with checked nonzero
counts and passed; both CPU Clippy configurations passed again. The temporary
baseline worktree is removed and raw logs retained. No performance/CUDA or
other-ISA inference follows from these ownership tests.


### Share ordinary preparation across a node's shape variants

The four MatMul-prepack consumers now implement the existing Kernel constant-
state sharing seam. A node-owned Arc contains dense f32 materializations,
contiguous half copies, native SGEMM panels and transpose memos. Each compiled
kernel keeps its own policy/flags, geometry, attributes and FLOP estimate. The
executor's existing same-node state exchange performs adoption without adding
MatMul knowledge to generic session code.

A state is accepted only with the same immutable operand flags and exact
DerivedWeightMemory source. Unrecognized state and retention-off/runtime-only
variants decline sharing; foreign grant sources and incompatible flags report
actionable errors. Node/session boundaries remain the executor's sharing scope,
with fresh owners constructed for new nodes or sessions. Installed process-global
transpose keys retire with the last shared preparation owner rather than the
first evicted kernel, and active readers still retain actual payload grants.
This changes transpose-key ownership, not transpose payload funding.

The dense estimator now counts one copy per constant operand per node and
removes its fixed shape multiplier. Its regression executes three shape
variants, verifies one actual Host charge and numerical results, then checks
exported-owner lifetime/final refund. Factory controls cover all four ops and
both constant slots with a no-pressure authority and exactly one-payload budget.
Other controls cover foreign-source/flag rejection, last-owner transpose
retirement and actual native panel reuse across activation shapes.

An end-to-end executor control exercises several activation widths with one
and two graph nodes, then a second session on the same provider. It verifies
node-local sharing, node/session isolation, one-copy prediction and final
release. Pressure reclamation is unavailable in that fixture, so a repeat
allocation cannot be hidden by eviction and a denied optional SGEMM pack cannot
evict a peer's payload on another CPU backend. No kernel arithmetic, production
backend default or performance claim changes in this step.

Validation on Apple M1 Max ARM64 macOS / Rust 1.98 / cached ORT: native CPU
1,984 passed / 20 ignored, session 275 passed; optional-MLAS CPU 2,021 passed
/ 35 ignored, session 277 passed. Strict all-target Clippy passes for both
crates/configurations; formatting and dispatch manifest/reachability/platform/
feature checks pass. Session suites/Clippy were rerun after making the resource
fixture pressure-independent for portable footprint measurement.

The unchanged exact public-factory fixture fails on parent d65b889d8: the second
shape kernel requests another 24-byte constant copy while the first holds the
complete budget. After explicitly cleaning/rebuilding the CPU crate, the exact
provider control, a 15-test MatMul group containing the three new owner controls,
and the exact one-copy prediction control execute nonzero checked counts and
pass. Both CPU Clippy configurations pass again. The baseline worktree and
consumed editing helpers are removed; raw transcripts remain. These are
ownership/correctness results, not performance or CUDA/other-ISA evidence.

Downstream validation: engine load tests 29 passed and memory-strategy tests
29 passed with `--no-default-features --features native-backend` and the cached
ORT library. The first load filter selected zero tests and was rejected by the
validation driver; the corrected `engine::load::` filter actually ran 29 tests.
No zero-test result is counted as coverage.


### Load-time preparation joins node ownership

Static CPU transpose preparation now exports the same MatMul weight owner used
by runtime shape variants. The generic kernel cache offers node-scoped static
state to the first kernel after checking immutable-input metadata, including
fully static materialization. Unknown state retains its lifetime and falls
through to ordinary preparation. Immutable flags are explicit provider API
inputs, independent of whether host bytes are available for every operand.

Session controls cover two nodes, static/symbolic shapes, repeated execution,
recognized/opaque/incompatible state, correct numerical outputs and final owner
release. The Apple CPU control verifies exact owner adoption, transpose reuse
and half-input numerical execution with either immutable operand-A flag even
when its host descriptor is absent.

A separate f64 Mul-to-MatMul regression changes a producer-backed initializer's
runtime value between requests. The unchanged test fails numerically on parent
87930c4a5: the second output remains 10 instead of 20. It passes with the shared
immutable-initializer classification. This is correctness evidence; no kernel
performance improvement or CUDA coverage is claimed.

On Apple M1 Max ARM64 macOS / Rust 1.98, full CPU suites pass in native
(1,984 passed / 20 ignored) and optional MLAS (2,021 / 35 ignored) configurations.
Session suites pass with 277 native and 279 MLAS tests, including the final f64
regression. Strict all-target Clippy passes for both CPU/session configurations;
formatting and dispatch manifest/reachability/platform/feature checks pass.

After the old-version negative control, the shared target's API/CPU/session
packages were explicitly cleaned. Fresh current controls each selected and
passed one test (generic handoff, writable-default regression, Apple prewarm
adoption). API tests pass 96/96; API and native CPU/session strict all-target
Clippy pass again. Downstream native-backend engine load and memory-strategy
groups each pass 29 tests with the cached ORT library. The temporary negative
control worktree and consumed editing scripts are removed; validation drivers,
raw logs and the stable host-lock inode remain.


### Materialized copy identity follows the complete source view

Dense f32 and contiguous f16 copies now share a cache-key implementation with
effective address, source dtype, shape and signed strides. Immutable storage
does not make all its logical views identical. The previous keys could return
the first copy when a runtime view changed strides without changing address or
element count. Shape must also participate: equal-size views can have identical
stride vectors but different row boundaries.

The direct control exercises both dense operand slots and native-width half
copies across positive, broadcast and negative strides, a same-strides shape
change, and a nonzero byte offset. Five distinct views hold five actual payload
grants (24 bytes each for dense f32, 12 for f16); after node-owner adoption,
five identical views hit the original allocations without new grants and final
drop refunds all bytes. A public-provider control executes MatMul, Gemm,
FusedMatMulBias and FusedGemm for both immutable operand slots, changes layout
through a shared preparation owner, then returns to the original layout.

Both unchanged controls fail numerically on parent 5bc65dce7. In the public
MatMul operand-A case, the changed view incorrectly repeats [6,6,18,18] instead
of [9,9,12,12]. These are correctness/ownership controls, not performance results;
CUDA and other CPU ISAs were not exercised on the Apple host.

Validation after an explicit CPU/session package clean on Apple M1 Max ARM64
macOS / Rust 1.98: native CPU 1,986 passed / 20 ignored and session 277 passed;
optional-MLAS CPU 2,023 passed / 35 ignored and session 279 passed. Strict
all-target Clippy passes for both crates in both configurations. Formatting and
dispatch manifest/reachability/platform/feature checks pass. Both exact new
controls selected one test and passed before the old-version negative controls;
the rebuilt full suites include both afterward. Consumed insertion fragments
and the temporary baseline worktree are removed; validation drivers/logs and
the stable host lock remain.


### Derived-weight admission stays off warm cache hits

Ordinary dense/half materialization uses a single cache lookup on a warm hit,
with common-rank full-layout keys stored inline. It does not allocate group
result/status containers or call the accountant. Cold fills retain the existing
transactional group admission, serialized construction, reader lifetime and
pressure reclamation. Higher ranks spill correctly; the inline capacity is no
semantic rank restriction.

A public-provider control counts every claim (including zero-byte requests)
and release. After setup and the cold preparation, 4,096 real MatMul executions
make no further accounting calls. The fixture admits any checked optional MLAS
panel as well as the dense copy, so reference-backend admission failures cannot
pollute a warm-hit measurement. Retention-off execution is an active control:
each real transient allocation claims and refunds its bytes. Teardown refunds
all retained payloads exactly once.

The `weight_admission_ab` benchmark uses public factory/execute calls, rotated
local/governed/A-A/f32-control arms, retained and transient policies, warm-up
outside measurement, per-round samples, wall and process CPU time. On Apple
M1 Max ARM64 macOS / Rust 1.98 / native Accelerate / one Rayon worker, three
interleaved process pairs compare parent b0bf6f71e against this implementation.
Each process contributes eleven rotated rounds. The stable Darwin host lock
covers both builds and all six process runs. Numbers below are medians of the
three process medians; brackets give their full range, in nanoseconds per call.

| M,K,N | Policy | Governed before | Governed after |
| --- | --- | --- | --- |
| 1,16,16 | retained | 371.63 [370.52–372.18] | 256.76 [256.28–269.02] |
| 1,128,256 | retained | 2213.30 [2211.08–2253.82] | 2078.89 [2039.26–2098.07] |
| 8,128,256 | retained | 1482.05 [1477.93–1484.22] | 1361.20 [1353.46–1453.46] |
| 1,16,16 | transient | 1058.90 [1056.08–1059.16] | 1023.58 [1018.27–1024.60] |
| 1,128,256 | transient | 74640.66 [74592.97–74670.43] | 74597.33 [74556.19–74752.49] |
| 8,128,256 | transient | 80850.41 [80842.14–80986.46] | 80977.30 [80883.47–81164.88] |

The tiny retained case improves about 31%; its duplicate local arm improves
similarly, while its governed f32 control moves about 1.4%. This supports a
metadata-path improvement on that CPU/workload, not a model-wide speedup.
Larger M=8 local A/A arms show substantial layout/scheduling variation, and a
transient f32 control moves about 6%; those rows do not establish general
speedups or regressions. All timing rows retain numerical output checks.

Admission is not universally free: in the three current tiny transient runs,
governed execution costs 14.83–19.13 ns (1.47–1.90%) more than local ownership,
above that case's A/A difference. Warm hits have structurally zero accountant
calls, but actual cold/transient allocations, pressure misses and final release
still require truthful accounting. No CUDA, other-ISA, pressure-path or
end-to-end LLM latency claim follows from this benchmark.

Validation: full native CPU 1,988 passed / 20 ignored and session 277 passed;
optional-MLAS CPU 2,025 passed / 35 ignored and session 279 passed. Strict
all-target Clippy passes for both CPU/session configurations. After strengthening
the callback observer to include zero-byte requests and admitting the checked
research panel, the exact callback control and CPU Clippy pass again in both
configurations; explicitly forced MLAS and Generic callback controls each select
and pass one test. The high-rank fallback control passes as well. Formatting and
dispatch/platform/feature gates pass. The temporary baseline worktree, consumed
insertion fragment and copied benchmark executables are removed; raw timing
samples, metadata, validation drivers/logs and the stable host lock remain.
