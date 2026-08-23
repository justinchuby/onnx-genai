# The acc0 int4 decode gap against ORT is ~1.1x, not 1.84x

**Date:** 2026-08-23 · **Host:** AMD EPYC 9V74, 16 physical / 32 logical, single
NUMA, AVX2+FMA+F16C (no AVX-512) · **Main:** `e189244ba` · **Harness:**
`crates/onnx-runtime-ep-cpu/benches/int4_decode_loop_ab.rs` +
`benches/ort_matmulnbits_baseline.py`

## Result

llama3-8B projection chain, `block_size = 32`, `accuracy_level = 0` (the
production default), one session, `taskset` to physical cores, four arms per
cell interleaved.

The gap is `ORT tok/s ÷ native tok/s` computed **within** each launch, then
medianed across launches. It is a paired statistic on purpose: the two arms run
seconds apart on the same machine, so pairing cancels drift that a ratio of
two independently-medianed columns would keep.

| width | native tok/s | ORT tok/s | **gap** | gap range | trusted cells | A/A range |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27.9 | 31.2 | **1.120x** | 1.112–1.128 | 2 of 3 | 1.025–1.036 |
| 4 | 107.2 | 122.3 | **~1.15x** | 1.087–1.284 | 4 of 5 | 0.868–1.150 |
| 8 | 211.0 | 238.0 | **1.120x** | 1.089–1.145 | 3 of 3 | 0.997–1.028 |
| 16 | — | — | **not resolvable** | — | 0 of 3 | — |

**Read the `t=4` row as "about the same as its neighbours", not as a distinct
1.15x.** Its A/A null — two identical native arms in the same launch — spans
0.868 to 1.150, so the whole gap sits inside its own noise floor at that width.
Quoting it to four digits would be false precision, and an earlier draft of this
document did exactly that. Only `t=1` and `t=8`, whose A/A nulls are within
3.6% and 2.8% of unity, resolve the gap at all.

Counts are cells, not launches: `t=4` pools five cells across two script
invocations, of which four are trusted. `t=1` is discussed under
[the discard](#the-t1-discard-is-post-hoc-and-here-is-what-it-costs).

**On statistics, because these columns are not interchangeable.** tok/s above is
`tokens_s_total` on both arms — wall-derived over every measured token — and the
gap divides one by the other, so it is like for like. The native binary
*additionally* reports a median per-token latency (`ms_token`), and the two
native views do not agree at width:

| width | native `ms_token` (median latency) | native `1000 ÷ tok/s` | divergence |
|---:|---:|---:|---:|
| 1 | 35.61 ms | 35.84 ms | 0.6% |
| 4 | 8.95 ms | 9.33 ms | 4.2% |
| 8 | 4.56 ms | 4.74 ms | 3.9% |

`tokens_s_total` is wall-derived and carries the tail; `ms_token` is a median
and discards it. The tail grows with width, which is why the divergence does.

**A latency-over-latency gap cannot be formed today at all**: the current ORT
harness reports only `tokens_s_total`, so there is no ORT median to divide by.
Substituting native's `ms_token` into the numerator anyway — a mixed quantity,
given only to show the conclusion is not sensitive to the choice — yields
**1.113 / 1.098 / 1.085** at t=1 / 4 / 8, i.e. 0.6–2.4% below the throughput
gap and mildly *declining* with width rather than flat. An earlier draft of this
table did exactly that substitution without saying so, printed native `ms_token`
beside ORT `1000/tps`, and called the result a gap; its columns did not yield
its own gap figure. The headline above is throughput on both sides.

Note the old published `1.84x` was itself a latency-style ratio (native
`ms_token` over ORT's min-over-reps per-`Run` median), so the then-versus-now
comparison spans a statistic change. It does not matter here: on either
definition today's figure is ~1.1x.

The previously published figure for this cell was **1.84x** at `t = 1`
(`docs/benchmarks/2026-08-21-int4-acc4-execution-regime.md`), carried into the
ledger as the reason acc0 was the top remaining CPU MatMulNBits target.

**That figure is not mislabelled, it is stale.** It is a real measurement of a
tree that no longer exists.

## Why the old number moved: measured, not inferred

The `1.84x` was `native 56.307 ms` against `ORT 30.632 ms`, and `t=8` carried a
companion `native 14.091 ms`.

An earlier draft of this document argued from a control: ORT re-measures at
31.99 ms, within 4.4% of its published 30.632, therefore the harness is
comparable and *"the movement is entirely on our side."* **That argument does
not hold, and the reviewer who pushed on it was right to.** The ORT arm
reproducing shows the *ORT* ruler did not move. It says nothing about the
native ruler, which sits in a different binary and changed repeatedly over the
same window — including `81e611c03` (#1722), whose title is literally *"make
the acc0 native and ORT arms measure one quantity"*.

So the inference was replaced with a measurement. `e9754e7ef`'s tree is checked
out in a second worktree, its `int4_decode_loop_ab` is built, and it is run
**beside** current main's on the same host, in the same environment, with
`PROBE_REPS=1` on both so neither gets a rep loop the other lacks, arms
interleaved within each launch and the launch order alternated.

### First: both published figures reproduce to within 0.4%, which identifies them

| published | rebuilt `e9754e7ef` today | reps | delta |
|---|---:|---:|---:|
| `56.307 ms` (t=1) | **56.519** (56.402 / 56.519 / 56.878) | 3 | +0.4% |
| `14.091 ms` (t=8) | **14.115** (14.105 / 14.115 / 14.196) | 3 | +0.2% |

Both reproduce **unpinned**, in the `steady` **`ms_token`** column. That pins
down what the old numbers are, which matters because the two obvious
corrections both turn out to land elsewhere:

- **The documented ~11% warmup handicap does not apply here.** §27 of the
  ledger (#1712) records that the native arm's clock started before thread
  spawn and three warmup steps — at `tokens = 24`, 27 steps of work charged
  against 24 counted tokens. That bias is in **`tokens_s_total`**, the
  wall-derived column. `ms_token` is a median over per-token samples collected
  *after* the warmup steps, and it never carried it. The published figures are
  `ms_token`: `ort_matmulnbits_baseline.py`'s own docstring at that commit names
  its comparand as *"the native harness's `steady` column-2 median"*, and the
  reproduction above lands on it to 0.4%. Deducting 11% from `56.307` would give
  a number that no run of either tree produces.
- **There is a real statistic asymmetry, and it points the other way.** Old ORT
  reported `min` over reps of a per-`Run` median; old native reported a single
  rep's median with no rep loop at all. Best-of-N against single-shot flatters
  ORT, so it made the old gap look *worse* than it was. Calling the two arms
  "the same statistic", as the first draft did, was wrong; the direction of the
  error is the one that does not help the argument.

### Then: what actually moved, per width

| width | factor | measured | what it is |
|---:|---|---:|---|
| 1 | old unpinned vs old pinned | pinned is **1.7% slower** | placement, negligible |
| 1 | **old → new, paired, 12 interleaved cells** | **1.64x** [1.61–1.88] | **kernel** |
| 1 | end-to-end, unpinned both | 1.60x (56.519 → 35.361) | — |
| 8 | old unpinned → old pinned to 8 physical cores | **1.67x** (14.115 → 8.430) | **benchmark defect** |
| 8 | **old → new, paired, 6 interleaved cells** | **1.82x** [1.78–1.89] | **kernel** |
| 8 | end-to-end, unpinned both | 3.06x (14.115 → 4.619) | — |

**At `t=1` the apparent movement is kernel.** Measured old-vs-new is 1.60–1.64x
against the 1.59x the published pair implies. The ruler question was worth
asking and the answer is that it does not bite at this width.

**At `t=8` it is not, and I am retracting the `3.08x` this document previously
claimed.** 1.67x of it is a defect in the *old benchmark*, and the mechanism is
specific: the old bench never called `EpFactory::initialize`, so it never ran
`bound_process_to_decode_budget()` and its process was never confined. That
function — with its physical-core `select_budget_cpus` — **already existed at
`e9754e7ef`**; only the bench was missing the call, which `11cb8e5f3` (#1766)
added. So the old bench measured a thread topology **no served session ever ran
in**: eight decode workers scattered across 32 logical CPUs, landing on SMT
siblings, against a full-width prefill/MLAS Rayon pool.

The SMT mechanism is directly measurable. Forcing the same binaries onto
`0-7` — four physical cores plus their siblings — against `0,2,...,14`, eight
distinct physical cores:

| binary | 8 physical cores | 4 cores + SMT siblings | unpinned | penalty |
|---|---:|---:|---:|---:|
| `e9754e7ef` | 8.430 ms | 16.121 ms | **14.115 ms** | 1.91x |
| current main | 4.664 ms | 7.988 ms | **4.619 ms** | 1.71x |

The old binary's unpinned run sits between its two pinned extremes, which is
what "the scheduler chose for us" looks like. Today's binary is **unaffected by
the absence of a pin** (4.619 unpinned vs 4.664 pinned, 0.99x) because it
confines itself. That is the correct behaviour and it is what production always
did — but it is a *measurement* correction, not a speedup delivered to anyone,
and reporting it inside a kernel-improvement figure was the error.

**This effect was already documented, on this host, by me, and I still walked
into it.** [2026-08-21-decode-worker-cpu-placement.md](2026-08-21-decode-worker-cpu-placement.md)
(#1680, ledger §24) concluded that an apparent "t=8 wash" was worker-to-CPU
placement rather than the kernel, and the dormant-nblock record opens with
"unpinned multi-thread numbers on this host measure worker placement, not the
kernel". The `14.091` figure is an unpinned multi-thread number on this host.
Knowing the failure mode, having written it down, and citing it in a
neighbouring document was not enough to stop me quoting a 3.08x off exactly
that kind of number two days later.

The net effect on this document's conclusion is nil: the **gap** rows at the top
are measured today, on both arms, under matched pins. What changes is the
credit — **1.8x of kernel improvement at `t=8`, not 3.1x.**

### Why the paired ratio survives a busy host

The absolute levels above move with load; the ratios do not. One launch caught a
competitor mid-cell and came back at `old 13.263 ms / new 7.007 ms` — both arms
about 1.6x slow — and its ratio was **1.893**, inside the range of the five
clean launches. Interleaving the arms seconds apart inside one launch is what
buys that: whatever slows one arm has usually not left by the time the other
runs. It is the same reason the gap column at the top is a paired median rather
than a ratio of independently-medianed columns.

## Six merges landed between the old number and this one

The commit that published `56.307` is `e9754e7ef` (#1628). Every one of these
landed **after** it, verified with `git merge-base --is-ancestor`:

| commit | PR | what it did |
|---|---|---|
| `8aed77a17` | #1667 | broke the serial f32 reduction chain in the int4 decode GEMV (**5.75x t=1**) |
| `99f105d52` | #1679 | enabled the register-blocked int4 kernel **at `accuracy_level = 0`** (1.17–1.69x) |
| `4e17f2251` | #1783 | folded the int4 zero-point unpack |
| `c3f0b0afa` | #1728 | sized the SPMD decode grain from its own pool, not ambient Rayon |
| `0652fdd2e` | #1794 | sized the persistent decode pool by physical cores |
| `6fdc04d75` | #1746 | gave the reserved dispatcher CPU a compute lane |

The first three are direct acc0 kernel work and the last three change what a
width means. Two more changed the **ruler** rather than the tree, and belong
beside them:

| commit | PR | what it did to the measurement |
|---|---|---|
| `81e611c03` | #1722 | made the native and ORT acc0 arms measure one quantity (barrier; removed the warmup/spawn charge from `tokens_s_total`) |
| `11cb8e5f3` | #1766 | made the benches call `EpFactory::initialize`, so a `t=N` row finally runs in the topology a served session runs in — **this is the 1.67x above** |

A gap figure that predates eight such merges is not evidence about today's tree,
and **no amount of re-labelling would have fixed it** — the 2026-08-23 scope
note added to it (#1843) said the number was `t=1`-only and that the
production-width gap was unmeasured. Both statements were true. Both missed that
the `t=1` number itself was three kernel merges out of date.

**This is the more dangerous half of the "stale number" failure class.** A
number that is *wrong* gets challenged. A number that was *right when taken*
reads as evidence forever, because its provenance looks impeccable — and it
keeps a closed problem at the top of the priority list while the actual top
item goes unexamined.

## What the gap actually is now

Native scales slightly better than ORT across the range that is measurable.
Both columns are `1000 ÷ tok/s`, so the scaling ratios are like for like:

| width | native | native vs its own t=1 | ORT | ORT vs its own t=1 |
|---:|---:|---:|---:|---:|
| 1 | 35.84 ms | 1.00x | 32.00 ms | 1.00x |
| 4 | 9.33 ms | 3.84x | 8.17 ms | 3.91x |
| 8 | 4.74 ms | 7.56x | 4.20 ms | 7.62x |

**`t=1` is a different code path from the other two rows**, and any scaling
figure quoted against it should say so. At `allowed.len() == 1`,
`build_from_env` declines to build a pool at all, so the native `t=1` column is
serial-on-the-dispatcher (`path=flat`, read back from the runtime and checked
per cell) rather than a one-worker pool. It is "vs serial", not "vs one
worker". Sebastian measured the same thing from the other side on the acc4 path
(#1740): at `total_workers <= 1`, `dispatch_output_rows` short-circuits and the
spawned worker receives no dispatch at all, 0% busy over a six-second window.

So the gap is flat at ~1.12x at the two widths that resolve it: it is **not** a
scaling problem, and there is no width at which acc0 collapses. The remaining
~11–12% is a kernel efficiency difference, and it is small enough that it now
sits below several other open items rather than above them.

### The `t=1` discard is post-hoc, and here is what it costs

Three `t=1` cells were taken and **two are in the headline**. The discarded one
is the CPU-0 contention cell described under method §4 below: both arms confined
to CPU 0 ran ~2x slow (native 68.488, ORT 60.489) while the roaming arm beside
them was normal (32.409).

The discard rule was **not pre-registered** — it was written after seeing that
cell — so the retained-cell figures have to be published beside it:

| `t=1` | gap | range | spread | cells |
|---|---:|---:|---:|---:|
| headline (CPU-0 cell discarded) | 1.120x | 1.112–1.128 | **1.4%** | 2 |
| all trusted cells retained | 1.112x | 0.927–1.128 | **18.1%** | 3 |

The median barely moves, so the conclusion survives either way. What does not
survive is the *precision*: "1.4% across launches" and "`t=1` and `t=8` agree to
three digits" are both artefacts of the discard, and the honest version of the
`t=1` row is `1.11–1.12x` with one cell in three thrown out for a documented,
independently-corroborated reason.

**The rule, stated in advance for next time:** a cell is refused if any arm's
wide-pin counterpart beats its matched-pin counterpart by more than the
1–2% measured pin asymmetry, because that is a per-CPU contention signature the
host-level guard cannot see. Applying that rule prospectively would have
rejected this cell without anyone having to look at its gap.

## Method, and three things that had to be fixed before the number meant anything

### 1. The two arms were not getting the same machine

`ONNX_GENAI_CPU_DECODE_THREADS=w` does not merely size a pool. It confines the
whole process:

```
onnx-genai: CPU decode budget 4 confined the process to 4 CPUs [0, 2, 4, 6]
onnx-genai: CPU decode budget 4 bounded the global Rayon pool (prefill/MLAS parallelism capped at 4 workers)
```

`acc0_gap_matrix.py` pinned **both** arms to all 16 even CPUs at every thread
count. At `t=4` that gave ORT four threads free to roam sixteen cores — more
L3, more memory controllers — while native had four. On a workload that is
bandwidth-bound by construction, that is a machine-size comparison wearing a
thread-count label. It went unnoticed because the only published gap cell was
`t=1`, where ORT's `intra_op_num_threads=1` means one thread regardless, and
because at `t=16` the two pins coincide exactly.

Both pins were measured so the size of the effect is data rather than
argument, and **the honest answer is that it is small**: ORT gains
**1–2%** from the wider pin on a quiet host (t=8: 1.120x matched vs 1.145x
wide; t=4: 1.148x vs 1.160x; t=1: 1.120x vs 1.136x). The concern was legitimate
and the fix is correct, but it does not move any conclusion. Recording that
here so the next person does not re-litigate it.

### 2. The width had to be checked, not assumed

Each native arm reports the width read back out of the pool, and a cell is
refused unless it matches the request:

```
decode_width requested=4 realized=4 path=spmd-pool as_requested
```

Timings cannot detect a vacuous sweep. A harness that silently runs one width
in every row looks perfectly stable, because it *is* — it is the same
configuration each time. That is precisely how `t=1 ≡ t=2` survived into three
documents (#1740, #1837).

Width 1 legitimately reports `path=flat`: at `allowed.len() == 1`,
`build_from_env` declines to build a pool at all, so the t=1 column is
**serial-on-the-dispatcher**, not "a pool with one worker". The comparison at
that width is native-serial against ORT-single-thread.

### 3. A pre-check cannot see a competitor that arrives mid-cell

`wait_quiet` samples before the cell. During this matrix a sibling agent
started a `cargo test` on the same crate — a full 32-thread saturation — and
four cells that had passed the pre-check were measured against it, at intra-run
spreads of 20–63%. They are discarded.

The guard is now a `LoadWatch` sampling the **instantaneous runnable count**
(field 4 of `/proc/loadavg`, not load average, which is a one-minute
exponential average that lags a job that just started and stays high long after
one ends) every second for the duration of every arm, refusing the cell if the
peak exceeds `width + 4`. The threshold has to scale with width: at width `w`
our own arm legitimately contributes ~`w` runnable threads, so a constant
"runnable > 4" would refuse every honest cell at `w >= 8`.

**This is necessary, not sufficient.** Ten launches at width 16 split into a
fast and a slow mode 1.8x apart in wall time while burning *identical*
CPU-seconds (14.4 vs 14.1 CPU-s per wall-s): the affected threads were never
descheduled, they just retired fewer instructions per cycle. No load,
CPU-efficiency, or context-switch guard can see that.

### 4. The wide-pin arm turned out to be a contention detector

One `t=1` cell passed the load pre-check (runnable 3, no process above 150%
CPU) and still produced:

```
w=1 native      68.488 ms    ort_matched  60.489 ms    (both confined to CPU 0)
w=1 native_aa   61.077 ms    ort_wide     32.409 ms    (roams 16 CPUs)
```

Every arm confined to CPU 0 ran ~2x slow; the one arm free to roam was normal.
That is a single busy CPU, and no host-level guard can see it — the box really
was quiet in aggregate. The cell is discarded, and it was caught **only**
because an arm on a different pin was measured beside it.

Worth generalising: at narrow widths the confinement lands on specific CPUs
(`w=1` → CPU 0), and CPU 0 is also where the harness, the shell and assorted
daemons live. **A single-CPU pin is the most fragile cell in any width sweep**,
and it is the one every speedup is quoted against.

## What is still unresolved

**Width 16 could not be measured, again.** Six cells at that width span native
5.719–12.486 ms/token with A/A ratios from 0.969 to 1.295, all taken while the
sibling `cargo test` was running. This is the same width whose launch
distribution spans 1.476–9.064 ms (514%) with no identified mechanism — see
[2026-08-23-acc4-decode-width-remeasurement.md](2026-08-23-acc4-decode-width-remeasurement.md).

The contaminated cells do put ORT at 2.29–3.17 ms against native 5.72–6.00 at
that width, which would be a ~1.6x gap if it survived, and native was in its
slow mode for all of them. **That is a hypothesis, not a result**, and it is
the one cell that would matter most: `t=16` is closest to an unconfined
production process. It needs a dedicated quiet-host study with the launch
distribution treatment, not another row in a matrix.

## Reproduce

The gap matrix:

```bash
cargo build --release -p onnx-runtime-ep-cpu --bench int4_decode_loop_ab
python3 crates/onnx-runtime-ep-cpu/benches/acc0_gap_matrix.py \
    --binary target/release/deps/int4_decode_loop_ab-<hash> \
    --models llama --threads 1,4,8 --sessions 1 --acc 0 --block 32 \
    --tokens 1:64,4:192,8:384 --reps 2 --launches 3 --aa --ort-pin both
```

`--ort-pin matched` is the default and is the only setting that compares like
with like; `both` adds the wide-pin arm, which costs one extra ORT run per cell
and is worth it at narrow widths for the reason in §4. `--launches 3` is what
produces the distribution — reps inside one process cannot see the
launch-to-launch spread, which is the larger of the two. The per-width token
map exists because a flat count spends the most wall time on the narrowest
width and still gives it the fewest samples relative to its variance.

The published table came from three invocations of this script rather than one
(`t=4` pools two of them), which is why its cell counts are 3 / 5 / 3 rather
than a uniform `--launches 3`. The `gap` column it prints is the one quoted
here; `ratio` beside it is the reciprocal, native/ORT.

The old-versus-new kernel A/B:

```bash
git worktree add ../old-tree e9754e7ef
git -C ../old-tree submodule update --init crates/onnx-runtime-cpuinfo/vendor/cpuinfo
cargo build --release --manifest-path ../old-tree/Cargo.toml \
    -p onnx-runtime-ep-cpu --bench int4_decode_loop_ab
```

Then run both binaries alternately with `PROBE_REPS=1` on each (the old tree
has no rep loop, so matching it is the only way both report one statistic),
`PROBE_BLOCK=32 PROBE_ACCURACY=0 PROBE_SESSIONS=1 PROBE_LAYERS=1`,
`ONNX_GENAI_CPU_DECODE_THREADS=w`, comparing the `steady` row's **`ms_token`**
column. Do not compare `tokens_s_total` across the two trees: the old binary
starts its wall clock before thread spawn and three warmup steps, which is the
~11% bias #1722 removed, and it lives entirely in that column.

To reproduce the placement finding, run the *old* binary at `t=8` three ways —
`taskset -c 0,2,4,6,8,10,12,14` (eight physical cores), `taskset -c 0-7` (four
cores plus SMT siblings), and unpinned. The unpinned run is the one that
returns 14.09 ms.
