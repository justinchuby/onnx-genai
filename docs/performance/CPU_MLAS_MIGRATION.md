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
observes. There are three sites in total (`matmul::gemm_with_backend`, the
`dispatch_mlas!` macro in `simd_activations.rs`, and softmax's
`record_softmax_route`), all per-call rather than per-element.

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
cache. Compact borrowing and numerical kernels are unchanged. Native packs are
still kernel-local; cross-kernel sharing and pressure eviction of the remaining
MLAS research memo/store owners are separate work.

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
  expansions use `DenseWeightCache` with `DerivedWeightMemory`; each payload
  owns its Host grant until the final reader drops it. Compact retained packs
  remain outstanding work; an estimate alone does not fund an allocation.
- **Anything cached per kernel instance is retained at least twice** — the
  executor's `KernelCache` is shape-keyed, so prefill and decode are separate
  instances. Share eligible weights through the owning `CpuMemoryResources`
  cache using explicit immutable source identity, rather than a process-global
  raw-address cache. Borrow unchanged source layouts when placement permits.
- **The memory-cheapest technique wins ties.** The MLAS int4 route holds a packed
  copy at ~2× the int4 bytes; on a 14B model that is 16.7 GB and the memory plan
  declines it (#1051), so the model that most needs the speed is the one that
  cannot afford it. A technique that allocates nothing is available everywhere.
- **A numerical change shipped as a performance change is how a correctness
  regression enters unnoticed.**
