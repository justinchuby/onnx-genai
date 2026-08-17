# CPU EP → MLAS migration ledger

**Direction, set 2026-08-17 (repository owner):** MLAS is **enabled by default
inside our CPU execution provider**. It is an internal backend library that our
EP calls; it is *not* ORT's built-in CPU execution provider and nothing here
delegates a node to it. The long-term objective is unchanged and continues:
progressively absorb and replace MLAS capabilities with our own faster kernels
until this EP comprehensively outperforms MLAS.

This supersedes the *default-features* half of
[`ABSORBING_MLAS.md`](ABSORBING_MLAS.md) (2026-08-16), which said we do not
bundle MLAS. It does **not** supersede that document's method — the same-binary
A/B, process CPU time, port-the-mechanism doctrine is what makes absorption
verifiable, and everything below depends on it.

## Why the default flipped

The 2026-08-16 direction was a correct response to a real defect (#1091: *the
configuration we measured was not the configuration we shipped*), but it fixed
it in the direction that made the shipped build the slow one. ORT's CPU EP **is**
MLAS. A cdylib without MLAS is not a conservative build of our EP, it is a
handicapped one: on this project's plugin-path A/B (AMD EPYC 9V74, AVX2, ORT
1.27, K=N=2048, p50 of 41 interleaved iterations) 4-bit `MatMulNBits` at M=128
took **81×** ORT's time without MLAS and 7.3× with it; `QLinearMatMul` u8 at
M=128 took **55×** without and 9.3× with.

Flipping the default fixes #1091 the other way round: measured and shipped are
now the same configuration because the default *is* the measured one, and the
opt-out is what carries the caveat.

The absorption programme is unaffected. MLAS being present is what makes the
same-binary A/B possible on every developer machine and every CI lane — you
cannot measure the gap you are closing against a library you did not link.

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
- `crates/onnx-runtime-ep-cpu-plugin/tests/mlas_default_wiring.rs` —
  `mlas_is_statically_private_to_this_cdylib` asserts every MLAS symbol is local
  to our shared object: **zero exported, zero undefined**. We link our own copy;
  the loader cannot bind our calls to `libonnxruntime`'s MLAS or vice versa, so
  the two never share a thread pool or a dispatch table.
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

Bound in `crates/mlas-sys`. "Used" means some route reaches it in a default
build today.

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
| `MlasNchwc*` (block size, reorder in/out/filter, conv) | `Convolution` | the 6 extra op registrations a default build gains |
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
dominates decode**, and it moves nothing else. It still does not reach 1.05 on
this host, so it does not *graduate* MLAS away — but it is the shortest path to
the first f32 GEMM graduation, and turning it on by default is the obvious next
slice. Deliberately not done here: this PR is infrastructure, and flipping a
kernel default is a kernel change that deserves its own before/after on a quiet
pinned host, per the rule this document just wrote.

## What the default costs

Linux x86-64, gcc 13.3.0, cargo 1.97.1, release cdylib
(`libonnx_runtime_ep_cpu_plugin.so`).

| | pure Rust | default (MLAS) | delta |
|---|---|---|---|
| cdylib size (release) | 7,574,552 B | 9,166,600 B | **+1,592,048 B (+21.0%)** |
| MLAS symbols linked | 0 | 897 | all local — 0 exported, 0 undefined |
| one-time C++/asm compile | — | 70.2 s (`mlas-sys`, release, 32 cores) | **+70 s per target dir** |
| incremental relink | 17.7 s | 16.7 s | within noise |

The 70 s is paid once per target directory. The vendored C++ does not change, so
`mlas-sys` is never recompiled by ordinary Rust edits and incremental rebuilds
measure the same either way. The size cost buys the order-of-magnitude quantized-
matmul numbers at the top of this document.

Requiring the toolchain is deliberate: a build that cannot compile MLAS should
say so, not silently produce a cdylib that is 55–81× slower on the paths users
notice. The opt-out below is the supported way to build without a C++ compiler.

## Platform validation

Flipping the default is what first required an ARM64 *binary*, and that exposed
a latent break in `mlas-sys`.

`crates/mlas-sys/build.rs` assembled the MSVC ARM64 `.asm` sources and had no
GNU/clang branch for the GAS `.S` sources in `lib/aarch64/`. Those files define
`MlasSgemmKernelZero`, `MlasSgemmKernelAdd`, `MlasGemvFloatKernel`,
`MlasConvSymKernelNeon` and ~30 more, all of which `sgemm.cpp` and `convsym.cpp`
call through dispatch tables. The C++ therefore *compiled* and only the *link*
broke — and nothing linked an ARM64 binary, because `scripts/check_cross_compile.sh`
ran `cargo clippy`, which stops at type-checking. This would have broken the
Linux ARM64, Windows ARM64 and macOS ARM64 lanes.

`build.rs` now assembles the GAS sources in four groups, because they do not
share an ISA baseline (each requirement was probed against
`aarch64-linux-gnu-gcc`, not guessed):

| group | files | `-march` |
|---|---|---|
| baseline | 23 | none (two carry their own `.arch_extension bf16`) |
| fp16 | `HalfGemmKernelNeon.S` | `armv8.2-a+fp16` |
| i8mm | `QgemmS8S8KernelSmmla.S`, `QgemmU8X8KernelUmmla.S` | `armv8.2-a+i8mm` |
| bf16 | `SbgemmKernelNeon.S`, `SconvDepthwiseKernelNeonBf16.S` | `armv8.2-a+bf16` |

The i8mm and bf16 groups are Linux-gated, mirroring the vendor's own
`#if defined(__linux__)` around `MlasGemmU8X8DispatchUmmla` /
`MlasGemmS8S8DispatchSmmla` (`platform.cpp:780`) — and because the Mach-O
assembler's acceptance of those directives is unverified here.

`scripts/check_cross_compile.sh` gained a third pass that *builds* the
`onnx-runtime-ep-cpu` test binaries for `aarch64-unknown-linux-gnu`, so the
linker has to resolve every MLAS symbol the EP calls. Reverting `build.rs` makes
that pass fail with the original undefined references, which is the evidence
that the gate is real rather than decorative.

| target | status | how |
|---|---|---|
| Linux x86-64 | tests, clippy, bench | native |
| Linux aarch64 | **1300 lib tests pass**, clippy, link | `qemu-aarch64-static`, cross gcc 13 |
| Windows x86-64 / ARM64 | build config only | no hardware here; CI lanes |
| macOS arm64 | build config only | no hardware here; CI lanes |

Executing the ARM64 tests needs `QEMU_LD_PREFIX` as well as a cargo runner —
several parity tests re-exec their own binary, and the child goes through
binfmt, which does not inherit the runner's `-L`:

```console
QEMU_LD_PREFIX=/usr/aarch64-linux-gnu \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64-static -L /usr/aarch64-linux-gnu" \
cargo test --target aarch64-unknown-linux-gnu -p onnx-runtime-ep-cpu --lib
```

Running these tests on ARM64 for the first time also corrected three claims that
only ever held on x86-64:

- The NCHWc layout-propagation tests asserted that blocked regions form. MLAS
  implements NCHWc convolution for x86-64 only; on aarch64
  `MlasNchwcGetBlockSize()` returns 1 and the pass deliberately does nothing.
  They now assert the *documented fallback* on such a host — the graph comes
  back untouched — rather than being skipped.
- `mlas_matches_rust_simd_on_special_values` demanded bit-identity. That is a
  real property on x86-64, where the pure-Rust route mirrors MLAS's AVX2
  polynomial instruction for instruction, and it stays asserted there. On
  aarch64 MLAS dispatches its own NEON kernels; the dense sweep measures the
  worst disagreement at 4.8e-7 absolute, so that target gets a tolerance plus
  exact NaN/infinity/signed-zero semantics, allowing for AdvSIMD's
  flush-to-zero on subnormals.
- `matmulnbits_arm64_kai_qsi8_asymmetric_qwen_shape_is_reachable` asserted that
  KleidiAI SDOT specifically wins the bits=8/block128/M=1 decode. With MLAS
  linked, `prefer_arm64_mlas_qnbit_decode` claims that shape first on non-Apple
  aarch64. The contract is that *a licensed CompInt8 route* runs, not which one,
  so it now matches its bits=4 sibling and accepts either.

## Opting out

Two boundaries, both real, both tested in CI:

```console
# The EP crate, keeping every operator group:
cargo build -p onnx-runtime-ep-cpu --no-default-features --features full

# The shipped cdylib:
cargo build -p onnx-runtime-ep-cpu-plugin --no-default-features

# The wheel:
NXRT_EP_CPU_NO_MLAS=1 pip wheel python/nxrt-ep-cpu
```

The wheel also drops MLAS automatically on any target outside
`setup.py`'s `MLAS_TARGETS` (linux-x86_64, windows-amd64, windows-arm64,
darwin-arm64), because a wheel that fails to build is worse than a wheel that is
slow. `nxrt_ep_build_features()` reports `"mlas"` or `""` so a packaged cdylib
can be identified after the fact, and `check_wheel.py` compares it against what
the build asked for.

Opting out changes behaviour, not just speed: without MLAS the EP registers 6
fewer ops (the NCHWc reorders) and uses `conv_ref.rs` instead of `conv.rs`, so it
claims fewer nodes. It remains correct — every MLAS route is differentially
tested against the native one that a pure-Rust build takes.

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
