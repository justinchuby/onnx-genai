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

Current plan, condensed. `Native-over-MLAS` means our kernel owns the
structure — blocking, threading, fusion, special-value handling — and calls MLAS
for an inner primitive.

| family | route | graduation status |
|---|---|---|
| `MatMulF32` | MLAS | baseline |
| `GemmF32` | Native-over-MLAS | partial — epilogue ours, inner SGEMM is `MatMulF32` |
| `MatMulNBits` | Native-over-MLAS | partial — int4 acc0 decode absorbed (#1104); prefill still MLAS |
| `QLinearMatMul` | MLAS | baseline |
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
| `MlasQNBitGemm` (+ available / packB / workspace) | `MatMulNBits` | prefill `M >= NXRT_SQNBIT_PREFILL_MIN` |
| `MlasSetThreading` | all | the MLAS pool runs under *our* thread budget (#1054) |

### Available but deliberately not used

| primitive | why not |
|---|---|
| `MlasEltwiseAdd` | measured no better than our dense SIMD `Add`; `Elementwise` is graduated |
| `MlasComputeTanh` | native beat MLAS-plus-clamp by up to **1.39×** at 256 Ki (#1121) — de-graduated from MLAS deliberately |
| `MlasComputeLogistic` for `Sigmoid` | same measurement as Tanh; the route was removed |
| `MlasComputeActivation` (fused) | our fusions cover more op pairs and keep the epilogue native |

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
   and exact Gelu remain (MLAS is 1.4–1.6× and 1.17× at 4 Mi).
3. **Normalization.** Already ours; listed so it stays measured, not to be
   absorbed.
4. **Softmax.** The widest measured gap and the easiest structural win: our
   native route is a scalar `f32::exp` loop, ~9× off MLAS's vectorised one
   (table below). A vectorised native `exp` closes most of it, and softmax is on
   the decode critical path for every token.
5. **Attention transpose / KV layout.** Native except for the two inner GEMMs;
   graduates with (1).
6. **Quantization / `MatMulNBits` prefill / `MoE`.** Highest value, most work.
   #1104 absorbed int4 acc0 decode (1.36× / 1.56×); prefill is the remainder,
   and `MoE` follows it.

`QLinearMatMul`, `Convolution` and `Pooling` are **not** priorities. They are
MLAS baselines with wide ISA-specific dispatch and no decode-path presence; the
graduation rule below applies if someone wants to try, but nothing is waiting on
them.

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
  `native-faster-but-under-5%` / `keep-mlas`.
- **On wall time, with CPU time beside it.** The two routes do not use the same
  number of threads — our `x86_sgemm` parallelises over column strips while MLAS
  declines to parallelise some shapes — so a CPU-time-only comparison would
  block a graduation that genuinely lowers latency. Wall time decides. Process
  CPU time is printed as `cpu_ratio` because the opposite failure is equally
  real: a route that "wins" by recruiting the whole machine regresses every
  concurrent session on the box, and the bench labels that
  `native-graduates-but-costs-more-cpu` rather than passing it silently.
- **Repeatable, outside noise.** Cross-worktree comparison on a shared runner
  showed a uniform 0.70–0.82× offset on *byte-identical* kernels — larger than
  most kernel wins. One binary, interleaved, or it does not count.
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
AVX2+FMA, 32 cores, one binary, interleaved, median of 21 reps. Native route is
`SimdX86` (the AVX2 `x86_sgemm` we ship, not the portable fallback) for GEMM.
Times are per unit of work — per multiply-accumulate for GEMM, per element
otherwise. `ratio = mlas / native` in wall time; **above 1.0 means native is
faster**.

| family | case | native ns/unit | MLAS ns/unit | ratio | cpu_ratio | verdict |
|---|---|---|---|---|---|---|
| `matmul_f32` | decode 1×2048×2048 | 0.1948 | 0.0288 | 0.148 | 0.089 | keep-mlas |
| `matmul_f32` | decode 1×4096×4096 | 0.1332 | 0.0777 | 0.583 | 0.345 | keep-mlas |
| `matmul_f32` | 16×512×512 | 0.1136 | 0.0051 | 0.045 | 0.019 | keep-mlas |
| `matmul_f32` | prefill 128×2048×2048 | 0.0061 | 0.0039 | 0.641 | 0.383 | keep-mlas |
| `matmul_f32` | odd 37×1023×511 | 0.0368 | 0.0040 | 0.110 | 0.035 | keep-mlas |
| `softmax` | decode 1×32000 | 5.17 | 0.52 | 0.100 | 0.100 | keep-mlas |
| `softmax` | attn 32×512 | 5.16 | 0.55 | 0.107 | 0.107 | keep-mlas |
| `softmax` | prefill 128×4096 | 5.19 | 0.56 | 0.107 | 0.106 | keep-mlas |
| `activations` | erf 4 Ki | 1.22 | 0.84 | 0.686 | 0.686 | keep-mlas |
| `activations` | erf 1 Mi | 0.56 | 0.38 | 0.672 | 0.725 | keep-mlas |
| `activations` | gelu-exact 4 Ki | 1.43 | 1.34 | 0.936 | 0.936 | keep-mlas |
| `activations` | gelu-exact 1 Mi | 0.65 | 0.54 | 0.827 | 0.847 | keep-mlas |

This is the honest state of the programme: **no family measured here is ready to
graduate**, and the ratios say where the work is. Activations are within
1.15–1.6× — reachable. Softmax is ~9–10× off but the cause is known and
structural (scalar `exp`). f32 GEMM is 1.5–22× off depending on shape, and the
spread is the finding: `SimdX86` is closest at the large shapes it was tuned for
(0.63–0.66 at 4096-decode and 128-prefill) and furthest at small and
awkwardly-shaped ones, where MLAS's packing and blocking amortise better.

Two caveats a future agent should not have to rediscover. First, these numbers
were taken on a **busy shared 32-core container**; repeat runs of the same
binary moved the GEMM ratios by up to 2× (`prefill_128x2048x2048` read 0.338,
0.561 and 0.662 across three consecutive runs) while softmax and activations
were stable to ~2%. Treat the GEMM row as an order of magnitude, not a
measurement, and re-run on a quiet pinned host before acting on it. Second, this
bench calls `gemm_with_backend` directly, so **neither** route gets the
prepacked constant weights that `matmul_dense` gives them in a real session.
That is deliberate — it compares the kernels rather than the caching — but it
means the bench understates both routes at inference time, and it is not the
instrument for judging the prepack path.

The comparison against #1116's dense-f32 work is not contradictory for this
reason: that task measured the shipped path with prepacking on a quiet laptop
and found parity at prefill, while this measures the raw kernels on a noisy
shared box. Both are true of what they measured.

### The one absorption already built but not switched on

#1116 ported the mechanism behind MLAS's `SgemmKernelM1Avx` — stream B in place
at M=1 rather than packing panels that are reused zero times — into
`x86_sgemm::sgemm_simd`, but left it behind `ONNX_GENAI_CPU_MM_SIMD_M1_GEMV`,
default **off**, pending a measurement. This bench is that instrument, so here
is the measurement, same binary, same session, toggle the only difference:

| case | ratio, toggle off | ratio, toggle on | native time change |
|---|---|---|---|
| decode 1×2048×2048 | 0.146 | **0.337** | 0.194 → 0.0796 ns/MAC (**2.4× faster**) |
| decode 1×4096×4096 | 0.625 | **0.843** | 0.127 → 0.104 ns/MAC (1.2× faster) |
| prefill 128×2048×2048 | 0.662 | 0.566 | unchanged (toggle only affects M=1) |

The toggle is a **large, one-sided, mechanism-explained win on the shape that
dominates decode**, and it moves nothing else — a native-over-native
improvement that needs no reference at all. Turning it on by default is the
obvious next slice and the shortest path to the first f32 GEMM graduation. It
is not done here: flipping a kernel default is a kernel change that deserves its
own before/after on a quiet pinned host, per the rule this document just wrote.
This document ships only the ledger, the harness and the roadmap.

## What linking the reference costs (research builds only)

Linux x86-64, gcc 13.3.0, cargo 1.97.1, release cdylib
(`libonnx_runtime_ep_cpu_plugin.so`).

| | default (shipped) | `--features mlas` (research) | delta |
|---|---|---|---|
| cdylib size (release) | 7,646,424 B | 9,250,096 B | **+1,603,672 B (+21.0%)** |
| MLAS symbols linked | **0** | 860 | the shipped column is what `default_artifacts_are_mlas_free.rs` asserts |
| one-time C++/asm compile | — | 70.2 s (`mlas-sys`, release, 32 cores) | **+70 s per target dir** |
| incremental relink | 17.7 s | 16.7 s | within noise |

The 70 s is paid once per target directory, and only by someone who asked for a
research build. The vendored C++ does not change, so `mlas-sys` is never
recompiled by ordinary Rust edits.

Nobody shipping this project pays either cost, and requiring a C++ toolchain is
not a condition of building or using the CPU EP. That is a second, smaller
reason the default is native: the production build has no C++/asm dependency at
all.

## Known limitation of research builds: `mlas-sys` does not link on ARM64

Recorded here because it is a real finding and because it is a *reason the
research reference is harder to use*, not a reason to ship it.

`crates/mlas-sys/build.rs` assembles the MSVC ARM64 `.asm` sources and has **no
GNU/clang branch** for the GAS `.S` sources in `lib/aarch64/`. Those files
define `MlasSgemmKernelZero`, `MlasSgemmKernelAdd`, `MlasGemvFloatKernel`,
`MlasConvSymKernelNeon` and ~30 more, all of which `sgemm.cpp` and `convsym.cpp`
call through dispatch tables. The C++ therefore *compiles* and only the **link**
breaks, on Linux ARM64, Windows ARM64 and macOS ARM64.

Nothing caught it because nothing ever linked an ARM64 MLAS binary:
`scripts/check_cross_compile.sh` runs `cargo clippy`, which stops at
type-checking. A clippy-only cross-compile check cannot see a missing symbol.

Consequences and scope:

- **No shipped artifact is affected.** A default build links no MLAS on any
  target, so this cannot reach a user.
- **A `--features mlas` research build on aarch64 will fail to link.** Run the
  differential suite and the A/B benchmarks on x86-64 until `build.rs` grows the
  GAS branch. The `.S` files do not share an ISA baseline (baseline, `+fp16`,
  `+i8mm`, `+bf16`), so the fix is four assembler groups with probed `-march`
  requirements rather than one.
- This work is **not** in scope for the native EP. It is tracked as research
  tooling, and is deliberately not carried here — adding a C++/asm ARM64 build
  matrix to CI for a reference we do not ship is the maintenance surface the
  direction declined.

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
  before allocation, in the bytes actually allocated, and declinable. Use
  `GovernedWeightCache<T>`.
- **Anything cached per kernel instance is retained at least twice** — the
  executor's `KernelCache` is shape-keyed, so prefill and decode are separate
  instances. Prefer a process-global cache keyed on weight identity.
- **The memory-cheapest technique wins ties.** The MLAS int4 route holds a packed
  copy at ~2× the int4 bytes; on a 14B model that is 16.7 GB and the memory plan
  declines it (#1051), so the model that most needs the speed is the one that
  cannot afford it. A technique that allocates nothing is available everywhere.
- **A numerical change shipped as a performance change is how a correctness
  regression enters unnoticed.**
