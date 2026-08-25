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

So announce, and take the lock. For any **saturating** run — a full benchmark
matrix, the EP test suite, a qemu leg — it is *required*, not preferred:

```sh
scripts/hostlock.sh status                       # is anybody benchmarking?
scripts/hostlock.sh run --owner leon \
    --reason "softmax 28-cell matrix" --gate 4 \
    -- python3 scripts/ort_ab/ab.py ...          # acquire, run, always release
```

`run` is the form to use: it releases on success, on failure and on Ctrl-C.
The holder must be the **outer harness**, spanning every arm of an interleaved
A/B/null — not each benchmark child. Wrapping the children individually leaves
a gap between arms in which the box looks idle, and that gap is not
hypothetical: a peer ran `ps`, saw no benchmark process, and started a sweep
*between two arms* of somebody else's A/B.

That is the general rule, and it is worth stating on its own. **Never conclude
the host is free from "nothing of mine is running", from `ps`, or from a single
`loadavg` reading.** The first two sample an instant, and occupancy is a
property of an interval. `loadavg` fails the other way — it is an exponential
moving average, so it stays high after a heavy run has ended and reads low
while a burst is still in flight (see below) — but the conclusion is the same:
the lock is the only statement about the interval, because it is a
*declaration* rather than a measurement. `--gate N` additionally waits for the
*instantaneous runnable count* to fall to N before starting, which drains load
from people who never took the lock; it is a start admission control and
nothing more.

`ab.py` **enforces this rather than documenting it.** Before it launches a
single arm it reads the lock and requires a declaration whose anchor is itself
or one of its ancestors; anything else stops the run with exit 3 and prints the
wrapping command. The ancestry test is what distinguishes the two shapes that
both look like "a lock is held": a lock held by an *ancestor* spans every arm,
while a lock held by a benchmark *child* is released between them, which is the
gap a peer's sweep once started in. A peer's lock stops the run for the
opposite reason — they declared the box.

```sh
scripts/hostlock.sh run --owner leon --reason "moe mt panel 6-cell" -- \
    python3 scripts/ort_ab/ab.py --arms base=./a mine=./b --null-control ...
```

Every CSV row then carries `host_lock`, `lock_owner`, `lock_anchor_pid` and
`runnable_at_start`, and the label covers the **whole window**: the lock is
read again at the end, and a run that changed hands halfway through is stamped
`changed` rather than named after whoever happened to hold it last. If that
second reading fails the row says `unverified-end`, because an unreadable lock
is not evidence of a handoff and is not evidence against one either.
`runnable_at_start` is a single sample taken before the first arm — a note
about the conditions at the start, not a property of the interval; see the
`--gate` paragraph above for why no threshold on it is honest here.
`--unlocked` runs anyway and stamps every row `unlocked:<state>` — for smoke
tests, never for anything publishable — but it will **not** run over a lock
somebody else declared, because that damage lands on their measurement, where
no label of ours can reach it. `scripts/ort_ab/test_ab_lock.py` covers the
admission table and runs in the `Host lock` workflow.

The gate itself lives in `scripts/ort_ab/hostlock_gate.py` so that every
driver passes the same one rather than each reimplementing an admission check
from this file. `sweep_decode.py` uses it too — a thread sweep is the most
saturating thing we run, and its `min`-of-`min` aggregation was written to
*survive* contention rather than exclude it, which is blind to an SMT sibling
and to steady external load. Two calls wire up a new harness:

```python
label, prov = hostlock_gate.require("python3 scripts/ort_ab/mine.py <args>")
...
end = hostlock_gate.window_label(label, prov, hostlock_gate.read_provenance())
```

`sweep_decode.py` also refuses to call a table a scaling curve when it is not
one. `bench_generic` reports the widths it actually realized, and the sweep
puts the answer in a `width_ok` column beside `host_lock`. It exists because
`--native-threads 1` takes the dispatcher's serial short-circuit rather than a
one-worker pool, so a `t=1` column is a *different code path* from every other
column in the same table — four decode rows were published before anyone
noticed.

The distinction the column draws is between a **capped** width and a
**different route**, and it is the whole design:

| `width_ok` | meaning | fails the sweep? |
| --- | --- | --- |
| `yes` | the lanes asked for came back, on one route | no |
| `capped` | fewer lanes than requested — an SMT or cpuset cap | **no** (see below) |
| `varied` | trials in one cell disagreed on route or width | yes (**6**) |
| `not-requested` | the bench never saw the request at all | yes (**6**) |
| `opted-out` | `--native-threads 0`, the documented opt-out | no |
| `absent` | this binary cannot report width | no, unverified |

`capped` does **not** certify that the cap was legitimate — `as_requested=no`
cannot tell an SMT or cpuset cap from a width bug. It declines to *fail*, and
it names the numbers (route, pool width, task width, host cpus) so a reader
can judge. That way round because a check only a large idle host can pass is
an exclusive-host assumption smuggled in as a correctness check, which is what
issue #1802 forbids — and it would brand as invalid the capped-scaling rows
the sweep exists to surface. Pass `--require-width` if you genuinely need
exact lanes and want those cells to fail; it is opt-in because it is a demand
on the *host*, not on the engine.

What is fatal is categorical and holds anywhere: trials inside a cell that did
not agree, a request that never reached the engine, and — the check that
catches the original defect — **columns that are not all the same route**.
Each `t=1` cell passes its own check (width 1 asked, width 1 delivered); only
the comparison *between* columns shows that the leftmost point is a different
program, so the sweep compares the reported `native_path` across the whole
table and exits **6** when a curve is drawn through more than one of them.
`unresolved` (the decode pool was never built — normal for a model that does
not take this path) and `absent` are *unknown* routes, not second ones, and
never split a table on their own.

Every row also carries a `route` column, because a stderr complaint does not
survive being pasted into a document and the route is the datum whose absence
let four serial-path rows be published as decode results.

For the same reason `1` is **not** in the default `--threads` list any more:
`--native-threads 1` confines the process to a single cpu, so decode takes the
flat route rather than a one-worker pool, and a default that always exits
non-zero would teach its readers to ignore the exit code. Sweep the serial
column on its own, or ask for the mixed table with `--allow-route-split`,
which acknowledges the split — it does not hide it, the complaint still
prints and the rows still carry their routes.

`sweep_decode.py` prints its rows as it goes, so a custody change cannot be
stamped onto them retroactively; the exit code carries it instead — **4** for
a handoff (the rows above span the change: discard them) and **5** when the
lock could not be re-read at the end (the rows may be sound, and nothing
establishes that they are). Those are deliberately different answers. Only one
code gets out, so the precedence is fixed rather than incidental: **4** wins
over everything (every row is discarded anyway, and a finding about columns
nobody will quote would bury the instruction), and **6** wins over **5** (a
route defect is a definite structural fact; an unreadable end of window is a
"cannot tell", and reporting the certainty as the doubt understates what is
known).

A bench child that wedges would hold the *shared host lock* for as long as it
hangs — squatting on the box is the one outcome the lock exists to prevent —
so each invocation is bounded by `--cell-timeout` (default 3600s, `0`
disables).
`ab.py` buffers, so it stamps the label on the rows.
`crates/onnx-runtime-ep-cpu/benches/acc0_*.py` are mostly not wired up yet —
of the 23 files there that start a benchmark, 3 hold the lock, 19 are recorded
gaps under issue #2043 and 1 is an `exec` wrapper whose caller carries the
gap. A recorded gap is one somebody can close; an absent one is one nobody can
see.

Which drivers take the lock is now checked rather than remembered.
`scripts/ort_ab/test_gate_conformance.py` requires every `.py` in this
directory to declare itself a driver, a generator, a library or a test: an
undeclared file fails, a driver that never calls `hostlock_gate.require`
fails, and a file declared harmless that imports `onnxruntime`, opens an
`InferenceSession` or *names* a binary under `target/` fails as contradicted.
Both halves read the parsed tree rather than the text: prose about a benchmark
is not a benchmark (the first pass of the #2043 audit counted `gen_gqa.py` as
a harness for describing one in its docstring), and a commented-out gate call
is not a gate call — that error would be fail-open, reporting an unprotected
driver as protected.

It detects a *literal* path under `target/`, not the spawn itself, so a driver
misdeclared as a generator that builds its path at runtime would pass. That
limit is written down at the top of the file. A generator that legitimately
loads the runtime declares `loads-runtime:` and keeps its reason, the same way
an ungated driver declares `known-gap:`.

The same file reads a second root, `crates/onnx-runtime-ep-cpu/benches/`, and
reads it **by behaviour instead of by declaration**: those are somebody else's
harnesses, and a per-file role list there would be a claim about another lane
that would have to be kept true by hand. A file there that starts a
benchmark — directly, or by calling a helper in a module it imports — must
hold the lock or carry a recorded reason: `known-gap:` (saturates, not gated
yet), `no-bench:` (starts something that is not a benchmark) or `wrapper:`
(`exec`s what it is handed; the caller holds the lock, and taking it here
would keep the pid and drop the release). A file that only reads JSON needs no
entry.

Resolving the delegation is what makes that root readable at all. Not one acc0
harness contains a `target/release/` literal — they take the binary as
`sys.argv[1]` — and seven of them do not call `subprocess` either; they call
`acc0_gap_matrix.native`, or a wrapper around it, one or two imports away. The
resolution runs to a fixpoint in **both** directions, across modules and
within one: `native` itself contains no `subprocess` call, it calls the
same-file helper `sh`. Cross-module resolution alone left `native` out of the
table, and `acc0_w16_steal_ab.py` — four arms at width 16, calling nothing
else — read as starting nothing and needed neither a gate nor an entry. *Importing* the harness
library is not enough: nineteen files there import it, mostly for its parsers,
and requiring an entry for every scorer would produce a ledger nobody
maintains, which is how a check ends up with a blanket exemption instead of a
line.

Both lock idioms count as held: `hostlock_gate.require` here, and the
`HostLock` context manager in `acc0_gap_matrix.py` that shells out to
`scripts/hostlock.sh`. A checker that knew only its author's idiom would have
reported three genuinely gated harnesses as ungated — a false alarm in
somebody else's lane, which gets a check deleted rather than obeyed.

Records are checked in **both** directions. A `known-gap:` whose file has since
been gated fails as **stale**, and one naming a file that is gone, or that no
longer starts anything, fails as **dead**. So closing a gap is a one-line edit
here, and the ledger cannot decay into a description of the tree as it was —
which matters most in the case nobody plans for, a file that stops gating and
is then quietly covered by an exemption nobody meant to still grant.

### Closing a gap: gating a harness in another root

The ledger asks nineteen files to gate. If gating meant writing a lock client,
we would get nineteen subtly different ones — that root already has two. So
`hostlock_gate.py` is reusable from anywhere: nothing in it reads the working
directory, `scripts/hostlock.sh` is resolved from the module's own file, and a
harness in `crates/onnx-runtime-ep-cpu/benches/` needs a path insert and two
calls:

```python
import pathlib
import sys

_here = pathlib.Path(__file__).resolve()
_root = next(p for p in _here.parents if (p / "scripts" / "ort_ab").is_dir())
sys.path.insert(0, str(_root / "scripts" / "ort_ab"))
import hostlock_gate

label, prov = hostlock_gate.require(
    "python3 crates/onnx-runtime-ep-cpu/benches/<this file> <args>",
    unlocked=args.unlocked,
)
...
label = hostlock_gate.window_label(label, prov, hostlock_gate.read_provenance())
row.update(hostlock_gate.lock_columns(label, prov))
```

The walk up is there instead of a `parents[3]` because that constant holds
only for a file sitting directly in the benches root — every one of the
nineteen does today, and the first harness in a subdirectory would get a path
insert pointing at `crates/` and a bare `ImportError`. A recipe that is
copied gets copied somewhere else.

Then delete that file's line from `EP_LEDGER` — the check **fails on a stale
record**, so the deletion is forced rather than remembered.

The gate *checks custody*; it never takes the lock. A lock taken by the driver
is released when the driver exits, so a matrix run as several processes would
be certified arm by arm and protected across none of them. What holds it is
the outer wrapper, which `remedy()` prints with your own command already in
it:

```
scripts/hostlock.sh run --owner <you> --reason "<what this measures>" -- \
    python3 crates/onnx-runtime-ep-cpu/benches/<this file> <args>
```

Three properties of that recipe are pinned by cells rather than by this
paragraph: every `hostlock_gate.*` name used in the docs exists, the snippet
above is *executed* from a file planted in that root (and one directory
deeper, past a decoy `scripts/`) rather than read, and importing the gate
from an unrelated working directory still finds `hostlock.sh`. Two more read
the gate itself: it must never run an acquiring subcommand — `acquire` or
`run` — because "the driver must not take the lock" is the whole point, and a
text search for the word `acquire` would not see `run`; and what it prints
when it refuses has to be the command that would have worked. Doc drift in a
safety instruction is worse than no doc — the reader concludes the gate is
broken and writes their own client, which is how this started.

One structural property of custody is checked with it: the lock must not be
acquired **inside** an arm loop. A lock taken per arm is released between
arms, so each arm is protected and the comparison between them is not — which
is precisely the gap that was sampled in when the host was read as clear
(#1803). The run-time half of the same question is the `host_lock` column on
every row; neither replaces the other. What none of this can see is a file
that saturates the box *in process* without starting anything —
`cpu_work_probe.py` is the honest example, bounded and single-threaded today.

A driver may be ungated *on purpose* — `ort_cuda_decode_bench.py` is, pending
its owner — but only as a `known-gap:` entry naming the issue. The distinction
is the point: a gap somebody wrote down can be closed, while a gap that is
merely absent is one nobody can see. The workflow triggers on
`scripts/ort_ab/**` and `crates/onnx-runtime-ep-cpu/benches/**` for the same
reason: a filter listing today's files would have skipped the job for exactly
the new harness the check exists to catch.

`SIGKILL` (and a full-box crash) cannot be caught, so it leaves the lock
directory behind. Nothing wedges: the lock carries its holder's pid **and**
that pid's start time, and the next acquirer reclaims it as soon as that
process is gone — including the case where the holder is a zombie its parent
never reaped, which still resolves in `/proc`. The same is true of the
internal guard that serialises reclaiming, so there is no state a kill can
leave that requires a human with `rm -rf`. If you want to see it before
trusting it: `hostlock.sh status` distinguishes FREE / HELD / STALE /
EXPIRED / UNUSABLE, and `provenance` prints who holds it and since when.

`UNUSABLE` is the answer that is neither "yours" nor "somebody else's": no
lock can be **created** at the configured path on this host, so nobody here
can participate. `status` names the reason (also `lock_dir_problem=` in
`--porcelain`, always emitted and empty when there is none), and `acquire`,
`wait` and `run` refuse with exit **7** — distinct from 1, because a
misconfigured host is not a bad argument, and distinct from 2 and 3, which
both assert that a peer holds the box. `run` refuses *without* running your
command, which is the whole point: the failure it replaces was a host that
reported `FREE`, ran the benchmark unlocked, and said nothing.

### Where the lock lives

`/tmp/onnx-genai-hostlock`, and **everyone on the box must resolve the same
path** or the lock coordinates nothing. If your host cannot use `/tmp` (it is
unwritable, `noexec`, or per-service under systemd `PrivateTmp=`), move it with
a machine-local config, which every invocation by every agent reads:

```sh
mkdir -p ~/.config/onnx-genai
echo 'lock_dir=/var/lib/onnx-genai/hostlock' > ~/.config/onnx-genai/hostlock.conf
```

You will be told when you need this rather than having to guess: on a host
where the path cannot be created, `status` reports `UNUSABLE` with the reason
and `acquire`/`run`/`wait` exit 7. Until that existed the same host reported
`FREE` and ran unlocked.

`$HOSTLOCK_DIR` also moves the path and is **not** the same thing: it is set per
process, so it does not move your peers with you. It is a **private** lock. It
acquires instantly every time, collides with nobody, and — before this was
fixed — reported `FREE` in bytes identical to a genuinely free shared host
while a peer held the real one. Every invocation now says so on stderr unless
`HOSTLOCK_PRIVATE_OK=1` acknowledges it, and `status --porcelain` and
`provenance` carry `lock_dir=` / `lock_scope=` / `lock_dir_source=` into the
row, so a recorded measurement says which lock its `declared=yes` is a claim
about. Use it for tests, not for measurements you intend to publish.

While a config is in effect, `acquire` and `run` also consult the old `/tmp`
path **read-only** and refuse (exit 2) while a live holder is there: a peer who
has not re-read the config cannot see the new lock, and taking it would put two
benchmarks on the box. That consult never reaps or writes the old path — it is
liveness-checked by pid **and** start time, so a crashed holder or a recycled
pid does not block the migration.

The Rust reader (`onnx_runtime_hostmon::hostlock`) resolves the path by the
same rule, and `tests/agrees_with_hostlock_sh.rs` holds both sides to it. A
reader that looked somewhere else would report `free` on a declared host,
convincingly, forever.

The lock and the gate decide whether to **start**. They cannot tell you
afterwards whether the numbers are any good, because they sample instants: a
gate sampled either side of a 2 s arm reported "runnable 2-4, clean" for runs
that were getting 50-70% of a core, against a 52% A/A null. Add
`--expect-cores N --min-efficiency F` to decide whether to **reject** it:

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
publishing. Set `F` from a measured quiet-host run of *your* workload, not from
1.0: a benchmark with a deliberate inter-token gap is legitimately below 1.0
and is not contended.

**Both are supplementary. Neither certifies a number, and neither replaces the
lock.** `(utime+stime)/wall` measures how much of the wall clock your process
spent *scheduled on a CPU*, and that is not the same as how much work it got
done:

* **An SMT sibling never deschedules you.** A competitor on the other
  hyperthread of your core shares the front end and the execution ports. You
  keep running, efficiency stays around 1.00, and your throughput falls anyway.
* **Neither does a neighbour off-core.** Memory bandwidth, LLC occupancy and
  turbo headroom are shared box-wide. A process saturating DRAM on other cores
  slows you without ever touching your runqueue.

The A/A null has the mirror-image blind spot. It is a *variance* measurement:
it sees contention that differs between the two arms, and it is blind to
contention that is steady across both. Both arms are depressed by the same
factor, the null comes out small, and the ratio you publish is a ratio of two
equally contaminated numbers — which is why a tight intra-run spread means only
that the contention was steady, never that the host was quiet.

So: the lock decides whether you may **start**, `--gate` drains stragglers who
never took it, and efficiency and the null throw away a *subset* of the reps
that were spoiled anyway. Only the lock says the box was yours.

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
| `unusable` | the lock could not exist here at all: no directory, and none creatable. Not `free` -- nobody declared the host **and nobody could have**, this row's peers included. The remedy is a working `lock_dir=` (see [Where the lock lives](#where-the-lock-lives)), not a retry |

`run --owner leon -- ...` is enough: since #1929 `run` exports the declared
owner into the wrapped command, so the obvious invocation is also the one that
certifies. Before that fix the flag was only a shell variable, the child
inherited nothing, and every row of an otherwise correctly locked matrix read
`held:leon` instead of `mine:leon` -- honest, but not certifying. Setting
`HOSTLOCK_OWNER=leon` in the environment still works and is equivalent, since
`--owner` defaults to it.

The export is deliberately **not** applied to an owner the script defaulted
from `$USER`. Every agent on this host runs as the same unix user, so a
`$USER`-derived owner cannot distinguish one declaration from another --
exporting it would make every agent's lock read `mine:` to every other agent,
which is the one direction this field exists to prevent. So the lock file still
records that defaulted owner (it is the best available answer to "who?"), but
nothing downstream is told to treat it as *its own*: an undeclared run reports
`held:` and says so. Declaring an owner is what makes a row attributable.

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
