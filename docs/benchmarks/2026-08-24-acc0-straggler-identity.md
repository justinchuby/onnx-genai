# The width-16 straggler is real, and it is not assignment, placement or layout

Date: 2026-08-24 · Owner: Roy · Base: `41bdccb27` (+ `squad/roy-straggler-lead`)
Probes:
`crates/onnx-runtime-ep-cpu/benches/acc0_w16_straggler_identity.py`,
`acc0_w16_straggler_aslr.py`,
`acc0_w16_straggler_window.py`

The previous record left the straggler as a lead with an open contradiction:
one worker held 0.565 of `last_arrivals` against a chance share of 0.067 and
`work_skew` was 0.562, while `output_chunk_len_for` returns `n.div_ceil(tasks)`
and every llama projection width divides evenly by 16 — so reading the source
predicted no skew at all. I recorded that contradiction rather than resolving
it by argument. This record resolves it by measurement, excludes two further
mechanisms, and then tests whether the phenomenon exists at all.

Three verdicts, in the order they were taken. Every rule was written into the
probe docstring before the first launch of that probe.

## 1. Assignment is exonerated — `ops_spread` is exactly zero

The existing instrument already reported `timed_ops` per worker and `derive()`
threw it away: it takes `workers[0]["timed_ops"]` as *the* op count, silently
assuming every lane got the same number. That assumption was the question, so
the probe stops assuming it and reads all fifteen.

24 launches, width 16, all trusted by the unmodified
`acc0_w16_worker_split.trusted()`:

| quantity | median | pre-registered bound |
|---|---|---|
| `ops_spread` = (max−min)/mean of `timed_ops` | **0.0000** | ACCEPT ≥ 0.10, REJECT ≤ 0.01 |
| `work_skew` = max(work_ns)/mean − 1 | **0.5702** | — |

`ops_spread` was **0.0000 in every one of the 24 launches**, not merely in the
median. So `output_chunk_len_for` splits the work exactly evenly, the static
reading of the source was right, and it is now measured rather than argued.

**The excess is execution time on equal work, not unequal assignment.** Every
lane is handed the same number of ops and one lane spends ~57% longer doing
them.

## 2. Placement and address layout are excluded as the selector

The same 24 launches give the lane→CPU map, read from the same profile records
as the timing, in the same launch — never a placement read paired with a timing
read from a different instant, which is the defect class that cost me four
probes.

```
lane->cpu maps distinct across 24 trusted launches: 1  (STABLE)
  [[0,0],[1,2],[2,4],[3,6],[4,8],[5,10],[6,12],[7,14],
   [8,16],[9,18],[10,20],[11,22],[12,24],[13,26],[14,28]]
```

One map, every launch: lane *i* on cpu *2i*, one worker per physical core. That
is #1729's placement working exactly as specified, and it is a categorical
read, not a timing.

Against that fixed placement the victim still moves:

```
straggler lane: idx1 5/24 = 0.208   (chance 0.067, ACCEPT needed >= 0.5)  REJECT
straggler cpu : cpu2 5/24 = 0.208   (chance 0.067, ACCEPT needed >= 0.5)  REJECT
lanes seen: {0:5, 1:5, 4:1, 5:1, 6:1, 8:2, 9:1, 10:3, 12:1, 14:4}
```

So the selector is neither a fixed lane nor a fixed CPU, under a placement that
is identical every launch.

That leaves a property that is fixed for a process, different between
processes, and able to make equal work take unequal time — which is a good
description of the **address layout**. ASLR re-bases the weight arena on each
exec and where a lane's slice falls relative to cache sets and page boundaries
is then fixed for that process. So: hold the layout still and look.

16 launches per arm, interleaved, alternating order, `setarch -R` versus
default:

| arm | top lane | concentration | median `work_skew` | median `ops_spread` | lane→cpu maps |
|---|---|---|---|---|---|
| `aslr` | idx1 4/15 | **0.267** | 0.393 | 0.0000 | 1 |
| `fixed` | idx8 4/15 | **0.267** | 0.482 | 0.0000 | 1 |

**A byte-identical address layout moves the victim exactly as much as a
randomised one** — the two concentrations are the same number. REJECT.

The knob was verified before any launch rather than trusted, because this
project already shipped `ONNX_GENAI_CPU_DECODE_AFFINITY` (#1792), a
user-facing placement control that is completely inert. A `setarch -R` that
did nothing would have produced `conc(fixed) == conc(aslr)` — which is
precisely the observed result — and been reported as a clean REJECT:

```
control: setarch -R bases ['555555554000-...', x3] -> CONSTANT
control: default    bases 6 distinct of 6         -> RANDOMISED
```

Both arms held `ops_spread` at 0.0000 and one lane→CPU map, so they differ only
in layout.

## 3. The assumption all three shared: is there a slow lane at all?

Three sharp hypotheses about *which lane is slow* failed in the same direction,
so the thing to test is what they share. All of them assumed a slow lane exists,
and that came from two numbers with no null model:

```
work_skew       = max(work_ns) / mean(work_ns) - 1
straggler_share = max(last_arrivals) / sum(last_arrivals)
```

`work_skew` is a **maximum over fifteen lanes**. Take fifteen samples of any
noisy quantity and the maximum sits above the mean; at 15 lanes a perfectly
symmetric jitter distribution yields a positive `work_skew` forever. The metric
cannot return zero, and I had been reading it as an imbalance across three
records.

`straggler_share` is cumulative over every op in the window, so **window length
separates the two stories** with no new EP counter:

* a genuinely slow lane is last on nearly every op → share roughly constant as
  the window grows, tending to 1.0
* max-of-noise is whichever lane won the most coin flips → excess above chance
  decays like 1/√ops, i.e. ×0.50 for a 4× window

10 launches per arm, interleaved, `--tokens 192` vs `768`:

| arm | median ops | median `straggler_share` | excess over chance | median `work_skew` |
|---|---|---|---|---|
| short | 960 | 0.4552 | **+0.3885** | 0.552 |
| long | 3840 | 0.7232 | **+0.6565** | 0.569 |

chance share = 1/15 = 0.0667; window growth 4.00× (control required ≥ 3×).

```
R = excess(long)/excess(short) = 1.690      (chance decay for 4x is ~0.50)
VERDICT: FIXED LANE
```

**The concentration does not decay — it rises.** At the long window one lane is
last on a median of **72% of 3840 ops** against a chance share of 6.7%. The
straggler survives its own null test in the direction that keeps it alive, and
`work_skew` stays flat at ~0.55–0.57 across a 4× window change, which is what a
persistently slower lane looks like rather than a max-of-noise.

This was the outcome that would have cost the most: NOISE would have retracted
the straggler lead and the 0.565-share figure in the ledger, and meant two
probes were built to find the mechanism of an artefact. It was written into the
rule for that reason and it is worth stating that it did not fire.

## Where this leaves the straggler

Established by measurement, not argument:

* **Real.** One lane is last on ~72% of ops over a 3840-op window, and the
  concentration strengthens with window length (R = 1.69).
* **Costly.** Straggler wait is ~0.31 of the width-16 window; every other lane
  waits on it at each barrier.
* **Not assignment.** `ops_spread` = 0.0000 in 24/24 launches.
* **Not placement.** One lane→CPU map across 24 launches; victim moves anyway.
* **Not address layout.** `setarch -R` gives the same concentration as ASLR.
* **Per-process and durable.** Persistent within a launch, different between
  launches.

What has *not* been established, and is the next question: what picks the
victim at startup. The remaining shape is something that varies between
processes but is not the address layout and not the CPU assignment — a startup
ordering effect is the obvious family (which lane registers first, what a lane
touches before steady state), but naming a specific mechanism here would repeat
the error of carrying "weight-arena placement across the two L3/CCX domains"
for two records on a single-NUMA host. It is left unnamed until it is measured.

## Method notes

* The probes add **no new instrument**. All three import
  `acc0_w16_worker_split` and call its `run_width`, `derive` and `trusted`
  unmodified; they only keep the raw per-worker records that `derive` discards.
  `one_launch` is deliberately not used because it drops them.
* Every arm comparison is interleaved launch-by-launch with alternating order,
  so host drift cannot be absorbed by one arm.
* The chance share 1/n is printed beside every concentration figure so a reader
  can see what "no effect" looks like without recomputing it.
* Each probe carries a control designed to fire when the manipulation did not
  take effect — the verified `setarch -R`, the ≥3× window-growth check, and the
  `ops_spread` equality check. Two of the three would have converted a silent
  non-manipulation into a confident wrong verdict.

## Reproducing

```bash
B=target/release/deps/int4_decode_loop_ab-<hash>   # absolute path required:
                                                   # the harness runs with cwd=HERE
scripts/hostlock.sh run --owner <you> --reason "straggler" --wait --gate 6 -- \
  python3 crates/onnx-runtime-ep-cpu/benches/acc0_w16_straggler_identity.py \
    --binary "$PWD/$B" --launches 24 --tokens 192 --reps 2 --out ident.json

scripts/hostlock.sh run --owner <you> --reason "straggler aslr" --wait --gate 6 -- \
  python3 crates/onnx-runtime-ep-cpu/benches/acc0_w16_straggler_aslr.py \
    --binary "$PWD/$B" --launches 16 --tokens 192 --reps 2 --out aslr.json

scripts/hostlock.sh run --owner <you> --reason "straggler window" --wait --gate 6 -- \
  python3 crates/onnx-runtime-ep-cpu/benches/acc0_w16_straggler_window.py \
    --binary "$PWD/$B" --launches 10 --short 192 --long 768 --out window.json

# re-score without running anything
python3 crates/onnx-runtime-ep-cpu/benches/acc0_w16_straggler_window.py \
    --binary x --replay window.json
```
