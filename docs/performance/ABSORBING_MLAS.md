# Absorbing MLAS: reference implementation, then replacement

> **Partly superseded, 2026-08-17.** The *packaging* half of the direction below
> was reversed by the repository owner: MLAS is now **on by default** inside our
> CPU EP, as an internal backend library we call — never a delegation to ORT's
> CPU EP. See [`CPU_MLAS_MIGRATION.md`](CPU_MLAS_MIGRATION.md) for the current
> ownership boundary, dispatch ledger and graduation rule.
>
> The *method* below is not superseded and is what the migration depends on:
> same-binary A/B, process CPU time, port the mechanism rather than the
> dependency. Two things change in practice. Step 4 said to re-measure in a build
> without the `mlas` feature "because that is what users run" — users now run the
> MLAS build, so the comparison is native-vs-MLAS **in one binary**
> (`cargo bench -p onnx-runtime-ep-cpu --bench native_vs_mlas`), which is
> strictly better evidence than two builds. And absorption is no longer a
> prerequisite for shipping speed; it is how we stop needing MLAS.

**Direction, set 2026-08-16:** we do not bundle MLAS by default. We progressively
absorb its optimizations into our own native kernels. MLAS stays in the tree as a
*measurement reference* — a second implementation in the same process that lets us
measure exactly how far our kernel is from a good one — and nothing ships behind
it.

## Why this became necessary

A run of PRs won large CPU speedups by routing kernels to MLAS. None of them
reached a default build: `mlas` was optional in `onnx-runtime-ep-cpu` and absent
from the default features of both the CLI and the server, and the CPU kernels
carry the routes behind `#[cfg(feature = "mlas")]`.

No individual PR did anything wrong — each was measured honestly with
`--features mlas`, and several said so. The problem was cumulative and structural:
**the configuration we measured was not the configuration we shipped**, and nothing
in the process noticed. Tracked as #1091.

That defect is now closed from the other side: the default build *is* the
measured build, and the pure-Rust configuration is the one carrying an explicit
flag and its own CI lanes.

There is a second reason, visible only on large models. The MLAS int4 route holds a
packed weight copy at roughly 2× the int4 bytes. On a 14B model that is predicted at
16.7 GB, over the residency ceiling, so the memory plan declines it (#1051) and the
model gets nothing. **The model that most needs the speed is the one that cannot
afford the memory.** A technique that allocates nothing is available everywhere,
including where a budget refuses everything else.

## The method that works

1. **Keep both paths in one binary**, behind a same-binary A/B environment toggle.
   This is the step that makes everything else possible: without both
   implementations in one process, on one host, under one load, the gap is not
   measurable and every absorption would be asserted rather than verified.
2. **Measure the gap in process CPU time**, never wall clock. This host varies
   enough that identical configurations have measured 39.3 / 25.8 / 16.1 s wall
   while `TotalProcessorTime` reproduced to ~2%. Report peak RSS beside every
   timing, polled by PID while the process runs.
3. **Read the reference against ours to find the mechanism.** Not to copy code —
   to answer "why is it faster", in terms specific enough to port.
4. **Port the mechanism, not the dependency**, and re-measure in a build that does
   *not* have the `mlas` feature, because that is what users run.

Step 3 is where the value is. The instinct is to skip to "prepack like they do",
which usually reintroduces the memory cost that made the dependency unattractive.

## Case study: int4 decode (#1104)

The gap on `MatMulNBits` int4 `accuracy_level=0` decode was **2.68×** before #1021
vectorised our borrowed path, and **1.25×** after. #1104 closed roughly 83% of what
remained.

The mechanism turned out to be **register/N-blocking, not layout**:

| | ours (before) | MLAS |
|---|---|---|
| output columns per pass | 1 | 4 |
| accumulators | 1 | 4 |
| activation vector loads | reloaded per column | loaded once, reused across the group |
| horizontal reduction | per block | one per column |

Only the shuffle-free nibble unpack is tied to MLAS's prepacked buffer, and that
contribution measured **under the noise floor**. So the port needed no repack and
no resident copy.

Measured on a build **without** the `mlas` feature, process CPU time, peak RSS
polled by PID:

| model | before | after | gain | peak RSS |
|---|---|---|---|---|
| qwen05b-symzp | 0.763 CPU s/tok | 0.560 | 1.36× | unchanged |
| qwen14b-symzp | 14.24 CPU s/tok | 9.14 | **1.56×** | 8196 → 8155 MB |

Output byte-identical on symmetric, asymmetric and 14B models.

**The methodological lesson is the durable part.** The brief for that task assumed
the win was layout, and proposed a transient per-thread packing tile as the way to
get locality without a resident copy. Measuring first showed the assumption was
wrong and made the workaround unnecessary rather than clever. A measured "this is
not worth building" is a good outcome and is harder to report than a patch.

## Ledger

| technique | measured with MLAS | status |
|---|---|---|
| int4 acc0 decode (#1027) | up to 33× vs the old scalar path | **absorbed** (#1104) |
| dense f32 MatMul, `MlasGemmBatch` (#1045) | 4.4×, ORT parity | in progress |
| QLinearMatMul integer GEMM (#1058, #1086) | 5–6× | not started |
| acc-4 M=1 decode gating (#1028) | up to 15× | not started; win is specific to hosts *without* VNNI, so it is unmeasurable on an AVX-VNNI box |
| MLAS pool sees the EP thread budget (#1054) | — | ours already; nothing to absorb |

Note on #1086's signedness translation: that one is **arithmetic, not MLAS**.
Because the kernel computes `sum_k (a_k - za)(b_k - zb)`, shifting an operand and
its zero point by the same constant leaves every `i32` accumulator bit-identical,
so an unsupported `u8 × i8` combination can be moved into the supported unsigned
domain exactly. It ports directly into a native kernel and should.

## Constraints that apply to every absorption

- **Any resident buffer must be governed** under #1056: declared to the memory plan
  before allocation, in the bytes actually allocated, and declinable. Use
  `GovernedWeightCache<T>`; a new bare `OnceLock` buffer is flagged by
  `.github/workflows/weight-cache-guard.yml`.
- **Anything cached per kernel instance is retained at least twice.** The
  executor's `KernelCache` is shape-keyed, so prefill (`m > 1`) and decode
  (`m == 1`) are separate instances with separate caches. Prefer a process-global
  cache keyed on weight identity, cleared at `Executor` drop.
- **Byte-identity is the default requirement.** An f32 GEMM change alters
  accumulation order and may not preserve it; if it cannot, quantify the deviation
  against a reference and say so. A numerical change shipped as a performance
  change is how a correctness regression enters unnoticed.
