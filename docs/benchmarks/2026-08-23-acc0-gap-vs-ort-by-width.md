# The acc0 int4 decode gap against ORT is ~1.1x, not 1.84x

**Date:** 2026-08-23 · **Host:** AMD EPYC 9V74, 16 physical / 32 logical, single
NUMA, AVX2+FMA+F16C (no AVX-512) · **Main:** `e189244ba` · **Harness:**
`crates/onnx-runtime-ep-cpu/benches/int4_decode_loop_ab.rs` +
`benches/ort_matmulnbits_baseline.py`

## Result

llama3-8B projection chain, `block_size = 32`, `accuracy_level = 0` (the
production default), one session, `taskset` to physical cores, three
independent launches per width, four arms per cell interleaved.

| width | native ms/token | ORT ms/token | **gap (ORT tok/s ÷ native tok/s)** | launches | across-launch spread |
|---:|---:|---:|---:|---:|---:|
| 1 | 35.36 | 31.99 | **1.120x** | 2 | 1.4% |
| 4 | 8.96 | 8.18 | **1.148x** | 4 | **17.1%** |
| 8 | 4.57 | 4.20 | **1.120x** | 3 | 5.0% |
| 16 | — | — | **not resolvable** | — | — |

The previously published figure for this cell was **1.84x** at `t = 1`
(`docs/benchmarks/2026-08-21-int4-acc4-execution-regime.md`), carried into the
ledger as the reason acc0 was the top remaining CPU MatMulNBits target.

**That figure is not mislabelled, it is stale.** It is a real measurement of a
tree that no longer exists.

## Why the old number moved, and the control that proves it

The `1.84x` was `native 56.307 ms` against `ORT 30.632 ms`. Measuring both
sides again on current main:

| arm | then | now | change |
|---|---:|---:|---|
| ORT, `t=1` | 30.632 ms | 31.99 ms | **+4.4%** — reproduces |
| native, `t=1` | 56.307 ms | 35.36 ms | **1.59x faster** |
| native, `t=8` | 14.091 ms | 4.57 ms | **3.08x faster** |

**The ORT arm reproducing is the control.** It is the same binary, the same
graph, the same host and the same statistic, and it lands within 4.4% of a
number taken two days ago. So the harness, the shapes and the definition are
comparable, and the movement is entirely on our side. Had ORT moved too, this
would be a host or harness finding rather than a kernel one, and nothing below
could be claimed.

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
width means. A gap figure that predates six such merges is not evidence about
today's tree, and **no amount of re-labelling would have fixed it** — the
2026-08-23 scope note added to it (#1843) said the number was `t=1`-only and
that the production-width gap was unmeasured. Both statements were true. Both
missed that the `t=1` number itself was three kernel merges out of date.

**This is the more dangerous half of the "stale number" failure class.** A
number that is *wrong* gets challenged. A number that was *right when taken*
reads as evidence forever, because its provenance looks impeccable — and it
keeps a closed problem at the top of the priority list while the actual top
item goes unexamined.

## What the gap actually is now

Native scales slightly better than ORT across the range that is measurable:

| width | native ms/token | native vs its own t=1 | ORT ms/token | ORT vs its own t=1 |
|---:|---:|---:|---:|---:|
| 1 | 35.36 | 1.00x | 31.99 | 1.00x |
| 4 | 8.96 | 3.95x | 8.18 | 3.91x |
| 8 | 4.57 | 7.73x | 4.20 | 7.62x |

The `t=4` row is the weakest of the three: its gap spans 1.087–1.284 across
four launches (17.1%) and one of its A/A pairs came back at 0.868, so treat it
as "about the same as its neighbours" rather than as a distinct 1.15x. The
`t=1` and `t=8` rows agree to three digits (1.120x both) at 1.4% and 5.0%.

So the gap is flat at ~1.12x from t=1 to t=8: it is **not** a scaling problem,
and there is no width at which acc0 collapses. The remaining ~11% is a kernel
efficiency difference, and it is small enough that it now sits below several
other open items rather than above them.

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

```bash
cargo build --release -p onnx-runtime-ep-cpu --bench int4_decode_loop_ab
python3 crates/onnx-runtime-ep-cpu/benches/acc0_gap_matrix.py \
    --binary target/release/deps/int4_decode_loop_ab-<hash> \
    --models llama --threads 1,4,8 --sessions 1 --acc 0 --block 32 \
    --tokens 192 --reps 2 --aa --ort-pin both
```

`--ort-pin matched` is the default and is the only setting that compares like
with like; `both` adds the wide-pin arm, which costs one extra ORT run per cell
and is worth it at narrow widths for the reason in §4.
