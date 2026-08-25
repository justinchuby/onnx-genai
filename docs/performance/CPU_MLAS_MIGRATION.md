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
| `MlasQNBitGemm` (+ available / packB / workspace) | `MatMulNBits` | prefill `M >= NXRT_SQNBIT_DECODE_MIN` |
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
