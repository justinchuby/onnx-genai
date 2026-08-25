# The acc0 gap at width 16 is ~1.78x, not 1.12x

**Date:** 2026-08-23
**Commit:** `0f84888b8`
**Harnesses:** `crates/onnx-runtime-ep-cpu/benches/acc0_w16_study.py`,
`crates/onnx-runtime-ep-cpu/benches/acc0_w8_w16_scaling.py`
**Supersedes the width-16 row of:** [2026-08-23-acc0-gap-vs-ort-by-width.md](2026-08-23-acc0-gap-vs-ort-by-width.md)

## Result

`accuracy_level = 0` — the production default — llama3-8B projection chain,
block 32, one session, both arms pinned to the same 16 physical cores.

| statistic | gap (ORT ÷ native) | n |
|---|---:|---:|
| run 1, paired median (384 tok/rep) | **1.782** [1.409–1.813] | 6 trusted of 14 |
| run 2, paired median (768 tok/rep) | **1.773** [1.279–2.176] | 11 trusted of 16 |
| both runs pooled, paired median | **1.775** [1.279–2.176] | 17 |
| best launch vs best launch | **1.770** (255.4 → 452.1 tok/s) | 17 |
| median launch vs median launch | 1.699 | 17 |
| the half of cells where **native ran fastest** | **1.650** | 9 |
| separate same-session scaling run, different harness | **1.793** [1.375–2.176] | 10 |

**At `t = 8` the same workload measures `1.120x`. At `t = 16` it measures
~`1.78x`.** Two independent runs, taken hours apart with different token
budgets, agree on the paired median to **0.5%** (1.782 vs 1.773).

**This reverses a conclusion merged earlier the same day.** `4b4dacc7e`
(PR #1852) established that the published 1.84x acc0 gap was stale and measured
1.120x at `t=1` and `t=8`, and concluded that acc0 was no longer the top CPU
MatMulNBits target — explicitly *conditional on `t=16`*, which that matrix could
not resolve. This study resolves it, and the condition fails. **acc0 at
production width is a ~1.78x gap and goes back to the top of the work list.**

## The pre-registered rule returned RANGE ONLY, and that verdict is honoured

The acceptance rule was written into the harness **before its first run**,
because the failure this study exists to avoid is choosing a threshold after
seeing which threshold licenses the conclusion:

```
ACCEPT a point estimate iff
  (1) n_trusted >= 8, and
  (2) aa_halfwidth <= 0.10, and
  (3) |gap_median - 1| >= 3 * aa_halfwidth
RANGE ONLY if (1) holds but (2) or (3) fails
REPORT NOTHING if (1) fails
```

Run 1 returned **REPORT NOTHING** (`n_trusted = 6`). Run 2 returned **RANGE
ONLY** (`aa_halfwidth = 0.377`). Neither run is re-scored under a looser rule,
and **no point estimate is claimed here.** The heading says "~1.78x" because
that is where five statistics land, not because the rule was cleared.

What the rule is protecting against is precision, and the range is genuinely
wide: **1.28x to 2.18x** across 17 trusted cells. What it is *not* protecting
against is the direction and order of magnitude, and those are not in doubt —
**the lowest single cell in either run, 1.279x, still exceeds the `t=8` figure
of 1.120x**, and the highest is 2.18x.

## Why the conclusion survives an A/A null that wide

An A/A half-width of 0.35–0.38 means a single cell cannot be quoted. It does not
mean the central tendency is unmeasurable, and three properties of the data say
the ~1.78x is not an artefact of the noise:

**1. The null is symmetric between the arms, so it is the width, not either
implementation.** The study alternates which arm gets run twice, so both nulls
are measured in the same launches:

| doubled arm | n | A/A half-width | range |
|---|---:|---:|---|
| native | 15 | 0.347 | 0.653–1.225 |
| ORT | 15 | **0.377** | 0.830–1.377 |

ORT's own null is the *wider* of the two. A noise process that inflates the
measured gap would have to be asymmetric; this one is not. The merged matrix
only ever measured a native A/A, which is why it could not tell these apart.

**2. Two independent runs agree to 0.5% on the median.** Run 2 doubled the token
budget (384 → 768) precisely to attack intra-run spread. The median did not
move. Noise that produced a spurious 1.78x would not reproduce to three digits
across a different token budget on a different quiet window.

**3. A contamination-resistant statistic agrees.** Comparing each arm's *best*
launch — the statistic least sensitive to transient contention, since
contention can only make a launch slower — gives **1.770x**, on top of the
paired median of 1.775x. These are near-independent estimators.

**4. Even native's best mode is far from parity.** Split the 17 trusted cells at
the median of native throughput: in the half where native ran *fastest*
(244–255 tok/s) the gap is still **1.650x**. The width-16 bimodality is real and
visible — the slow half gives 1.834x — but the gap does not vanish when native
is in its good mode. It only shrinks from 1.83x to 1.65x, and 1.65x is not
1.12x.

## The scaling wall is between t=8 and t=16, and only we hit it

Measured directly, both widths **inside the same launch** with the same token
budget, width order rotated per launch, by
`crates/onnx-runtime-ep-cpu/benches/acc0_w8_w16_scaling.py`. Its acceptance rule
was also pre-registered, and deliberately a **sign test on paired launches**
rather than a comparison of medians: at width 16 both arms carry a ±35% A/A
null, which can move a median-of-ratios but cannot flip a per-launch sign
symmetrically.

    ACCEPT "native scales worse" iff n_trusted >= 6 and native's scaling ratio
    is below ORT's in >= 80% of paired launches.

**Result: 10 of 10 paired launches, 100%.** Not one launch in either width order
had native scaling as well as ORT.

| arm | `t=8 → t=16` scaling | range over 10 launches |
|---|---:|---|
| ORT | **1.762x** | 1.485 – 1.932 |
| native | **1.319x** | 1.131 – 1.488 |

**ORT converts the doubling into 1.76x — close to the 2.0x ideal. We convert it
into 1.32x, losing a third of the added width.** The gap at `t=16` is therefore
not a new kernel deficiency appearing at that width: it is the `t=8` gap plus a
scaling failure that is ours alone.

The same run reproduces the width-16 gap from a different harness, a different
token budget (512) and a different session — **1.793x** [1.375–1.968], against
the study's 1.775x. That is a fourth independent estimate landing in the same
place.

**This rules out the obvious explanation.** The natural reading of a plateau at
half the logical CPUs is memory bandwidth, and there is a documented bandwidth
knee at exactly this width (`2026-08-22-decode-width-scaling.md`, acc4 decode
pool: 7.52x → 9.26x, only 1.23x for the last doubling). But a bandwidth ceiling
is a property of the *host*, and it would flatten both arms. **ORT scales 1.76x
across the same doubling on the same host in the same launch.** Whatever binds
us at `t=16` is in our pool or our kernel, not in the DRAM.

### What `t=8` and `t=16` physically are on this host

Added 2026-08-24 after a cross-agent report claimed the decode pool places two
workers per physical core on a single L3. The claim is false on this tree (see
below), but checking it established something about the scaling comparison that
was not previously written down, and it belongs beside the result.

Realized placement, read out of `/proc/<tid>/status` (`Cpus_allowed_list`) for
every `onnx-genai-spmd` thread on `0a668d54b` —
`crates/onnx-runtime-ep-cpu/benches/decode_placement_census.sh`. This is a
categorical read, not a timing: it does not need a quiet host.

| configuration | spawned workers | pinned CPUs | L3 spread |
|---|---:|---|---|
| default, no `taskset`, no env | 15 | `0,2,4,…,28` | 8 in L3#0, 7 in L3#1 |
| `THREADS=16` under the even mask | 15 | `0,2,4,…,28` | 8 in L3#0, 7 in L3#1 |
| `THREADS=8` under the even mask | 7 | `0,2,…,12` | **7 in L3#0** |

Host: `cpu0/topology/thread_siblings_list` = `0-1` (siblings adjacent), L3 in two
32 MiB instances, `shared_cpu_list` `0-15` and `16-31`.

So **one worker per physical core in every configuration**, with the reserved
dispatcher CPU (`30` at width 16, `14` at width 8) left clear, exactly as
`reserve_single_group_headroom` and `decode_affinity::order_pin_targets`
specify.

The part that bears on the scaling number: `ONNX_GENAI_CPU_DECODE_THREADS=8`
confines the process to `[0,2,4,6,8,10,12,14]`, and **all eight of those CPUs
are inside one 32 MiB L3 instance**, while width 16 spans both. The `t=8 → t=16`
doubling on this host is therefore a doubling of cores *and* of L3 *and* of
memory-controller reach — a bigger change than "twice the threads", on a
workload that is bandwidth-bound by construction.

**This does not confound the comparison, because both arms get the same CPUs at
each width.** `acc0_gap_matrix.ort()` defaults its pin to `native_pin(threads)`
— `EVEN[:threads]`, the set native confines itself to — and
`acc0_w8_w16_scaling.py` calls it without an override, so ORT ran on
`[0,2,4,6,8,10,12,14]` at `t=8` and on all sixteen even CPUs at `t=16`, the same
machine as native in each case. That symmetry was added in `4b4dacc7e` and both
the 1.762x and 1.319x figures were measured after it.

It does mean the 2.0x "ideal" quoted above is a *conservative* reference for
both arms: a doubling that also doubles cache and bandwidth could exceed it. The
direction of the finding is unchanged and if anything understated — ORT converts
this change into 1.76x and we convert it into 1.32x.

**The cross-agent claim itself was a stale build.** It reported 16 workers on
cpus `0-15` and offered PR #1729 as the fix. #1729 merged as `6e8c31ebd` on
2026-08-23T01:11:35Z, and the related width-halving report (`available / 2` in
`default_persistent_threads`) was fixed by #1794 (`0652fdd2e`) the same night.
Both are ancestors of `origin/main`; the census above is what the merged tree
does. Recorded because the claim was circulating with a measured-looking table
attached to it, and because the tell was visible in that table without any
re-measurement: its two arms reported 16 and 15 shard participants, and 15
spawned workers plus an inline dispatcher is precisely what current main
builds.

**A permanent per-CPU competitor was also reported and is not reproducible.**
The same report measured cpu 0 delivering 50.3% of a core with 344 involuntary
switches in 2 s, and cpu 1 at full CPU *share* but 55% of the *work* — the SMT
signature of a busy sibling. Re-run with an equivalent work-completed probe
(`benches/cpu_work_probe.py`, iterations against `CLOCK_THREAD_CPUTIME_ID`) on an
idle host, cpu 0 reads `cpu_share` 0.999–1.000 at 9429/9482/9489 iterations,
inside the 8744–9499 band spanned by cpus 1, 2, 3, 14, 15, 16, 17, 18, 30 and
31. One transient outlier appeared (cpu 16 at 6485) and did not survive two
re-probes. The original observation was real; "permanent" was an inference from
a single sweep, and on a host shared by several agents that is load, not
topology. **The instrument point stands and is the useful part**: a CPU that is
granted 100% and delivers 55% is invisible to every CPU-time instrument, which
is a second and independent reason not to read `Percent of CPU` as utilisation
here.

**One number in that run disagrees with the merged matrix and is not
suppressed.** Its `t=8` gap reads **1.264x** [1.191–1.660], against the
`1.120x` established in `4b4dacc7e`. The difference is on the native side —
native `t=8` gives ~190 tok/s here against 211.0 there, while ORT `t=8` is
unchanged (232–240 vs 238.0). The plausible cause is this harness's own design:
it runs a `t=16` arm immediately beside the `t=8` arm in the same launch, so the
`t=8` cell inherits whatever pool and cache state a 16-wide arm leaves behind,
which the single-width matrix never did. That makes `1.264x` a worse estimate of
the isolated `t=8` gap than `1.120x`, and it is **not** offered as a correction
to it. It does not touch the scaling result, which is paired within each launch
and so cancels any level shift common to both widths.

## Run 1's guard was wrong, and it is recorded rather than quietly widened

Run 1 refused 8 of 14 cells. **None of those refusals was contention.**
`competing_load()` returned empty for all 14 cells and the pre-check runnable
count was 2–4 every time — the host was demonstrably quiet throughout. The
refusals were the guard tripping on *our own* threads: the ceiling was set from
a structural estimate (16 workers + dispatcher + harness + sampler + shell ≈ 20,
so 22) and the cell's real peak reaches **25**.

Measured peaks across run 1's 14 cells: `18, 19, 19, 22, 22, 22, 23×6, 25, 25`.

Run 2 raised the ceiling to 26. **That number is fitted to run 1's peaks, which
makes it post-hoc with respect to that run**, and it is disclosed here rather
than presented as a fresh derivation. Two things make it defensible: the
quantity it was fitted to (peak runnable of our own cell) is not the outcome,
and no cell it admits had a competitor — there were none to admit. Run 1's
`REPORT NOTHING` verdict stands as issued and is not re-scored.

What the ceiling still buys is the case `competing_load` cannot see: a sibling
running many threads each under 150% CPU.

## Method

Every arm is a fresh process. Within a launch the three arms run seconds apart
so pairing cancels drift, and **arm order alternates by launch** —
`native, ORT, native` on even launches and `ORT, native, ORT` on odd ones. That
serves two purposes: a monotone within-launch drift cannot masquerade as an
effect, and whichever arm is doubled supplies that launch's A/A null, so both
arms' nulls are measured rather than only native's. The merged matrix always ran
native first and always doubled native.

The pre-check gates on the **instantaneous runnable count**, not the load
average. The in-tree `wait_quiet` uses `getloadavg()[0]`, and this study is why
that is the wrong instrument in practice as well as in principle: immediately
after its own `cargo build` the load average sat at **6.86** on a host whose
runnable count was **2**, and a `threshold = 3.0` pre-check slept through a
fully quiet window without measuring anything. Gating the pre-check on the same
quantity `LoadWatch` polices during the arm makes the two agree.

Native width non-vacuity is read back from the binary per cell
(`decode_width requested=16 realized=16 as_requested`) — timings cannot detect a
vacuous sweep.

## Reproduce

```bash
cargo build --release -p onnx-runtime-ep-cpu --bench int4_decode_loop_ab
BIN="$PWD/target/release/deps/int4_decode_loop_ab-<hash>"

# the width-16 gap
python3 crates/onnx-runtime-ep-cpu/benches/acc0_w16_study.py \
    --binary "$BIN" --launches 16 --tokens 768 --reps 2 --out w16.json

# the t=8 -> t=16 scaling of both arms, paired within each launch
python3 crates/onnx-runtime-ep-cpu/benches/acc0_w8_w16_scaling.py \
    --binary "$BIN" --launches 10 --tokens 512 --reps 2 --out scaling.json
```

The harness prints the pre-registered thresholds in its first line and applies
them to its own output. Raw per-cell records — including every refused cell,
both arms' intra-run spread, the peak runnable count and the competitor list —
are written incrementally to `--out`, so a run killed part-way still yields its
completed cells.

## What this changes

- **acc0 returns to the top of the CPU MatMulNBits work list.** The `1.12x`
  headline of `4b4dacc7e` is correct at `t=1` and `t=8` and does not describe
  the width closest to an unconfined production process.
- **The scaling wall is localised to us and not to the host.** Native converts
  the `t=8 → t=16` doubling into 1.32x where ORT gets 1.76x, in 10 of 10 paired
  launches. **This has since been split into its two causes** —
  [2026-08-23-acc0-width-16-cpu-attribution.md](2026-08-23-acc0-width-16-cpu-attribution.md)
  adds CPU-seconds attribution to both arms and finds ~30% more CPU burned per
  token at `t=16` *and* ~40% of the sixteen cores idle (`busy` 0.938 → 0.595),
  against ORT's `busy = 0.999`. It also found that the decode pool's 500 us
  spin/yield wait makes `busy` read 0.966 instead of 0.595 at the same
  throughput, so **occupancy readings taken at the default blocktime cannot be
  used to argue the pool is fed.** The remaining step is narrower than stated
  here: localise the *idle* half with #1859's per-worker straggler attribution.
- **The width-16 A/A null is itself a defect.** ±35% on two identical arms is
  not a property any other width in this matrix has (`t=8` is ±2.8%), and until
  it is understood every width-16 number costs 16 launches to state as a range.
  It is symmetric across arms and across implementations, which points at the
  host or the pinning rather than at either kernel.
