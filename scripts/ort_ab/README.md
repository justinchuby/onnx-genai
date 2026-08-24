# Native-vs-ORT A/B harness

Generators and an interleaved driver for comparing this repo's CPU execution
provider against a **real ONNX Runtime CPU session** on the same host, the same
graph, the same thread count and the same inputs.

These are the scripts behind
[`docs/benchmarks/2026-08-15-cpu-ep-vs-ort-attention-moe.md`](../../docs/benchmarks/2026-08-15-cpu-ep-vs-ort-attention-moe.md).

## Why this exists

A kernel microbenchmark that compares a new kernel against the *old* kernel
answers "did I make our code faster", which is not the question that matters.
The question that matters is "is our execution provider faster than the runtime
a user would otherwise run". Those two answers can point in opposite
directions: a kernel can get 8× faster and still be 3× slower than ORT.

So every number here is a **native/ORT ratio measured inside one process, on one
host, on the same graph**, with node and session overhead included. Lower is
better; `1.0` is parity.

## Ground rules

* **Ratios, not absolutes.** Shared CI/dev hosts drift. Same-shape absolute
  timings on the reference host moved by more than 4× between sessions, while
  the paired ratio was far more stable. Only publish ratios — and only compare
  ratios that came out of the *same* driver invocation. The ratio is not
  perfectly session-invariant either: the same MHA graph measured as a control
  in one session and as a subject in another differed by ~3× in ratio, because
  `--runs`/`--warmups` change how much of ORT's one-off packing is amortised.
  Never build a table by pasting cells from two different runs.
* **Interleaved arms.** `ab.py` alternates the arms trial by trial, so drift
  hits both arms roughly equally instead of being attributed to whichever arm
  ran during a noisy minute.
* **Run the null control.** Pass `--null-control` and `ab.py` adds a third arm
  that is *the first arm's own binary under a second name*, interleaved with the
  others. It cannot measure the change, which is the point: whatever delta it
  reports is this host's noise floor for that cell, measured in the same
  invocation as the real comparison, and the summary marks any real delta no
  larger than it as `WITHIN NOISE`.

  This is not a formality. Twice now, a delta that looked like a result was the
  instrument: §36.3 of the ledger measured ±20–30% at 32 threads on cells the
  change could not reach, and §37.4 measured **~40% apart on the median at two
  threads between two binaries traced to be executing the identical code
  path**. Interleaving and alternating arms does not remove that; it only stops
  it from being systematic. A cross-arm claim without a null arm in the same
  invocation should be read as provisional.
* **Medians and dispersion.** Report `p50` with the observed `[min–max]` of the
  per-trial ratios. A win narrower than the dispersion is not a win.
* **Warmups.** Both runtimes get warmup iterations before the measured runs;
  first-touch page faults and lazy packing otherwise land entirely on whichever
  arm goes first.
* **Parity is checked on every cell.** The driver records the harness's
  `parity=PASS/FAIL` per trial, marks any cell containing a failure as
  `PARITY_FAIL=n/m` in the medians summary, and prints a warning at the end.
  A performance number from a cell that does not produce ORT's answer is not a
  performance number.
* **Production shapes first.** Head counts, KV-head counts, head dims, hidden
  sizes and expert geometry come from public model configs. Benchmarks are not
  chosen to flatter the kernel.

## Synthetic data

**No trained weights are downloaded or used.** The generators emit single-node
graphs whose *dimensions* come from public architecture configs (Llama-3-8B,
Phi-3-mini-4k, Qwen2.5-0.5B, Qwen3-0.6B, Qwen3-MoE, Mixtral, Phi-3.5-MoE,
BERT-base/large, CLIP, Whisper). Tensor contents are the benchmark harness's
deterministic synthetic pattern, fed identically to both runtimes. Where a full
expert bank would not fit in host memory as f32, the expert **count** is reduced
and the reduction is recorded in the file name (`e{N}`).

This measures kernel and scheduling behaviour at production geometry. It does
**not** measure end-to-end quality, and it cannot detect a data-dependent
performance cliff that only trained weights would trigger.

## Generators

| Script | Emits |
|---|---|
| `gen_gqa.py` | `com.microsoft::GroupQueryAttention`, one node, fully static shapes |
| `gen_grid.py` | the GQA decode/prefill grid across four model geometries |
| `gen_l3sweep.py` | GQA decode graphs whose per-head attended-KV working set lands on 1/2/4/8/16/32 MiB, for cache-topology sweeps |
| `gen_mha.py` | `com.microsoft::MultiHeadAttention` (the operator the vectorised `sdpa_f32` path serves), 16 cells: 7 bidirectional encoder/prefill shapes, 3 causal (`unidirectional=1`) decoder prefills, 5 decode shapes (`q_seq = 1`, KV 128/1024/4096, batched), and one 8-token chunk |
| `gen_moe.py` | `com.microsoft::MoE` / `QMoE`, top-k routing, grouped experts |
| `gen_transforms.py` | the transforms that *surround* attention: `Softmax`, `RotaryEmbedding`, KV-cache `Concat`, BSNH↔BNSH `Transpose` |
| `gen_f16_gemv.py` | decode-shaped (`M = 1`) f16 `MatMul` or `Gemm` (`--op`), sweeping the weight working set from L2-resident to past LLC |
| `gen_f16_nt.py` | f16 `Gemm` prefill cells emitted **twice** — `transB = 1` and `transB = 0` over the same array pre-transposed — so the storage layout is the only variable |

Each takes an output directory:

```bash
python3 scripts/ort_ab/gen_transforms.py --out /path/to/models/transforms
python3 scripts/ort_ab/gen_grid.py --out /path/to/models/grid
python3 scripts/ort_ab/gen_moe.py --out-dir /path/to/models/moe --tokens 1 32 512
```

`gen_f16_gemv.py` additionally takes `--op {matmul,gemm}`. Since #1613 both ops
take the half decode GEMV at every `m == 1` weight size -- the
`HALF_PREFILL_GEBP_MIN_WEIGHT` handover to the fused widen-pack GEBP was
retired for decode because it measured as a loss -- so neither op needs
`ONNX_GENAI_CPU_MM_HALF_GEBP=0` to reach the GEMV any more. `--op gemm` emits
`Gemm` with `transB=0`, which has no weight gate and therefore measures the
GEMV as a default build actually runs it.

`gen_gqa.py` bakes semantically-constrained integer inputs (`seqlens_k`,
`total_sequence_length`) as **initializers**, because the harness would
otherwise fill them with its generic synthetic integer pattern and both
runtimes would attend over a nonsensical KV length.

## Driver

Build the benchmark binary first (a `cuda-*` feature is required by the crate's
feature wiring even for a CPU-only run):

```bash
cargo build --release -p onnx-genai-bench \
  --no-default-features --features mlas,cuda-13000 --bin bench_generic
```

Then:

```bash
python3 scripts/ort_ab/ab.py \
  --arms base=/path/to/baseline/bench_generic new=./target/release/bench_generic \
  --models /path/to/models/transforms/*.onnx \
  --threads 1 8 16 \
  --trials 5 --runs 7 --warmups 3 \
  --null-control \
  --csv results/transforms.csv
```

* `--arms name=path` — one or more binaries. Two arms is the usual case: an
  exact single-commit baseline build and the branch build, so the arms differ
  *only* by the commits under test.
* `--arm-env name=KEY=VALUE` — per-arm environment, for A/B-ing an opt-in
  threshold or feature flag using one binary in both arms.
* `--null-control` — add an arm named `null` that runs the **first** arm's
  binary, with the first arm's `--arm-env`, under a second name. It costs one
  extra arm's wall time and buys the only thing that says whether a delta is a
  result: the same-invocation noise floor. The deltas table then reads

  ```
  moe_phi35moe_h2048_i6400_e4_t512 t=1    after:  -25.48%  > noise (0.19%)  null:   -0.19%
  moe_mixtral_h1024_i3584_e8_t512  t=1    after:   -2.38%  > noise (0.63%)  null:   -0.63%
  ```

  and any arm whose delta is inside the null delta is printed as
  `WITHIN NOISE` instead of a number to paste into a table.
* `--native-only` — run every arm with `--native-only` so no ORT session exists
  in the child at all, and compare native times directly instead of
  `native/ort`. **Use this for any native-vs-native A/B.** ORT's intra-op pool
  spin-waits, so a paired run steals cores from the native arm: on the f16 GEMV
  cells it depressed the native median by up to 6x and pushed the null control
  to 27%, which was larger than most of the effects under test. The CSV marks
  these runs with `native_only=1`, and in that mode the `ratio` column holds
  native milliseconds rather than a ratio.
* `--threads` — passed through as **both** `--native-threads N` and
  `--ort-intra-threads N`, so the two runtimes get matched pool widths.
  `--native-threads` sets `ONNX_GENAI_CPU_DECODE_THREADS` (a decode-pool width
  budget, read once into a `OnceLock` before any session is built);
  `--ort-intra-threads` sets the ORT session's intra-op thread count. Neither
  sets process CPU affinity — the comparison is between equally-sized thread
  pools on an otherwise unconstrained machine, not between processes pinned to
  N cores. On a contended host that means both arms see the same contention,
  which is what makes the comparison fair; it does **not** mean either runtime
  had N cores to itself.

The driver prints a per-trial line for every cell, a medians table, and — when
more than one arm is present — a deltas table against the first arm, then writes
the full per-trial CSV at exit.

## Reading a result

```
sm_decode_h32_kv8192  t=8  base: ratio_p50=71.099 [65.886-85.249] native_p50=2.308ms
                            new:  ratio_p50= 6.390 [ 5.973- 7.188] native_p50=0.178ms
```

The publishable claim is "13× closer to ORT, still 6.4× behind at 8 threads" —
not "13× faster". The `native_p50` column is diagnostic only; compare it across
arms within one run, never across sessions.

## Caveats

* **A paired run depresses the native arm on small cells.** ORT's intra-op pool
  spin-waits, so on short kernels the two arms are not merely measured together
  but *compete*; the effect has been measured at up to 6x on f16 GEMV cells.
  Where the claim is a native-vs-native A/B, or where the cell is short, measure
  the arms separately with `bench_generic --native-only` / `--ort-only` and use
  the paired mode only to confirm `parity=PASS`. `--ort-only` synthesizes the
  same dtypes as the paired path (f32, f16, i32, i64, u8, i8).
* The driver **raises** if a cell produces no result line, rather than silently
  dropping it. Under heavy host contention a cell can fail this way; re-run it
  standalone before concluding anything.
* Under `ONNX_GENAI_PROFILE_OPS=1` the GQA fusion decision inverts, so the
  op-level profiler is not trustworthy for measuring scheduling changes. Use it
  to *locate* a hot op, then measure the op in isolation with these graphs.
* `Softmax`/`RotaryEmbedding`/`Transpose`/`Concat` graphs are single-node, so
  fixed per-run session overhead (currently a fresh allocation of every graph
  output) is a large fraction of the smallest cells. Cells below roughly 100 µs
  are overhead-dominated and should be read as an upper bound on the kernel gap,
  not as the kernel gap.

## Before you measure: take the host lock

Every number produced by these scripts is a ratio between two arms timed on one
machine, and that machine is shared. Contention has moved the *same* cell by
**8.6x** between two windows here (197.2 vs 22.8 tok/s), and two *identical*
binaries in an A/A null have disagreed by **45%** on one cell. Neither run
looked wrong from the inside: intra-run spread stayed under 6% in some of the
corrupted samples, because a tight spread only says the contention was steady,
not that the host was quiet.

So announce, and prefer the mechanical form:

```sh
scripts/hostlock.sh status                       # is anybody benchmarking?
scripts/hostlock.sh run --owner leon \
    --reason "softmax 28-cell matrix" --gate 4 \
    -- python3 scripts/ort_ab/ab.py ...          # acquire, run, always release
```

`run` is the form to use: it releases on success, on failure and on Ctrl-C.
It is held by the harness for the whole run, so an interleaved A/B/null keeps
the lock across every arm — sampling `ps` between two arms of someone else's
A/B is how a host was declared free while a sweep was mid-flight.
`--gate N` additionally waits for the *instantaneous runnable count* to fall to
N before starting, which drains load from people who never took the lock.

`SIGKILL` (and a full-box crash) cannot be caught, so it leaves the lock
directory behind. Nothing wedges: the lock carries its holder's pid **and**
that pid's start time, and the next acquirer reclaims it as soon as that
process is gone — including the case where the holder is a zombie its parent
never reaped, which still resolves in `/proc`. The same is true of the
internal guard that serialises reclaiming, so there is no state a kill can
leave that requires a human with `rm -rf`. If you want to see it before
trusting it: `hostlock.sh status` distinguishes FREE / HELD / STALE /
EXPIRED, and `provenance` prints who holds it and since when.

The lock and the gate decide whether to **start**. They cannot tell you
afterwards whether the numbers are any good, because they sample instants: a
gate sampled either side of a 2 s arm reported "runnable 2-4, clean" for runs
that were getting 50-70% of a core, against a 52% A/A null. Add
`--expect-cores N --min-efficiency F` to decide whether to **believe** it:

```sh
scripts/hostlock.sh run --owner leon --reason "softmax 28-cell matrix" \
    --gate 4 --expect-cores 16 --min-efficiency 0.90 \
    -- python3 scripts/ort_ab/ab.py ...
```

Both knobs are **opt-in**, and deliberately so. Without them `run` measures
and prints (`verdict=unjudged`) but enforces nothing, because a shared,
co-tenanted host is the normal case here and on the edge devices this engine
targets — a tool that failed by default on a busy box would be asserting a
dedicated machine that nobody promised. A low efficiency is information about
one measurement, never a claim that the host owes you every core.

That compares the CPU the command actually consumed against `N x wall` and
exits 6 if it falls short, so an unattended harness stops instead of
publishing. It needs no quiet host -- it tells you which reps to throw away.
Set `F` from a measured quiet-host run of *your* workload, not from 1.0: a
benchmark with a deliberate inter-token gap is legitimately below 1.0 and is
not contended.

Gate on that runnable count (`cut -d' ' -f4 /proc/loadavg | cut -d/ -f1`), not
on the 1-minute load average: `loadavg` is an exponential moving average, so it
stays high for a minute after a heavy run has ended and reads low while a burst
is still in flight. It misleads in both directions.

The Rust benchmarks read the lock and put the answer in the row. `bench_generic`
and the `decode_gap_park_ab` matrix emit a `host_lock=` field covering the
**whole measured window** -- read before the first run and again after the last,
so a run that changed hands halfway through prints `changed` rather than naming
whichever holder happened to be there at the end:

| value | meaning |
|---|---|
| `mine:<owner>` | held throughout by a live anchor matching `HOSTLOCK_OWNER` -- the only value that certifies the row |
| `foreign:<owner>` / `held:<owner>` | held by someone else, or by an owner that cannot be attributed because `HOSTLOCK_OWNER` was unset |
| `unverified:<owner>` / `stale:<owner>` | held by an anchor whose liveness is unprovable, or provably gone |
| `changed` | the window spans a change of custody; no single holder describes it |
| `free` / `unknown` | nobody declared the host, or the lock could not be read -- deliberately not the same value |

Set `HOSTLOCK_OWNER` in the **environment**, not just `--owner` on the command
line: `run` does not export the flag, so a child that inherits neither cannot
tell your lock from a co-tenant's and reports `held:` rather than `mine:`.
`HOSTLOCK_OWNER=leon scripts/hostlock.sh run --reason "..." -- ...` sets both at
once, since `--owner` defaults to it.

There is deliberately no fallback to `$USER`, even though the script uses one.
Every agent on this host runs as the same user, so `$USER` cannot distinguish
one declaration from another -- defaulting to it would report `mine:` for
somebody else's lock, which is the one direction this field exists to prevent.
An unset `HOSTLOCK_OWNER` genuinely cannot be attributed, and says so.

This is orthogonal to the `foreign_%` / `sib_%` columns beside it and does not
replace them: those measure what the host *did*, `host_lock` records what
somebody *said they were doing*. Contention sampling reads instants and can
miss a co-tenant that starts and finishes between two snapshots; a declaration
covers the whole window but proves nothing about load. An unlocked run on a
genuinely idle box is fine, and a locked run beside somebody's unannounced
`cargo test` is not.

The lock is advisory. It cannot stop anyone from using the cores and does not
try to; it makes "is somebody benchmarking right now, and who?" cheap enough to
check that there is no excuse for not checking. Record the runnable count you
measured at, and mark absolute timings taken on a busy host as indicative --
interleaved *ratios* survive contention far better than absolute milliseconds,
but neither survives it silently.
