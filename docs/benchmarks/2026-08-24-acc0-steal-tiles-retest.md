# The +23% steal-tiles candidate was the bimodal null, measured 24 launches deep

**Date:** 2026-08-24
**Tree:** `16eab6214` (branch off `683861ff5`)
**Host:** AMD EPYC 9V74, 16 physical / 32 logical, SMT siblings adjacent, L3 in
two 32 MiB instances. AVX2 + FMA + F16C; no AVX-512, no VNNI.
**Knob:** `ONNX_GENAI_CPU_DECODE_STEAL_TILES_PER_WORKER`, 1 (shipped) vs 2.

## Verdict first

**REJECT, and this time the rejection is a measurement, not an abstention.**

| run | n | ratio (2 ÷ 1) | sign | A/A half-width | verdict |
|---|---:|---:|---:|---:|---|
| original, 2026-08-23 | 8 | **1.2327** | 88% | 0.2154 | REJECT — effect < 3x null |
| this run, unstratified | **24** | **0.9883** | 38% | 0.1478 | REJECT — no effect |
| this run, **fast mode only** | 8 | **0.9889** | 38% | **0.0323** | REJECT — no effect |

The original run refused the candidate because its effect (+0.2327) did not
clear three times its A/A half-width (0.6462). That was the right call for the
wrong reason: the honest reading at the time was "the instrument is too blunt to
see a +23% effect". **There is no +23% effect.** Three times the launches puts
the ratio at **0.9889 against an A/A half-width of 0.0323** — an instrument
**4.6x sharper** than the original, which would have resolved anything above
**+9.7%**, and the candidate does not move at all.

**The original +23% is quantitatively accounted for.** In that run the control
arm drew the slow mode in **4 of 8** launches and the test arm in **1 of 8**. A
slow control deflates the denominator of `tps(test)/tps(control)` by the
published **1.687x** mode ratio, so a net 37.5% imbalance can manufacture up to
**+0.2576** of ratio out of nothing. The observed effect was **+0.2327**. The
artifact bound exceeds the effect it is supposed to explain, with room to spare.

## The instrument that was missing, and the one that failed

`benches/acc0_w16_mode_stratified.py` does not contain a rule. It imports
`verdict()` from `acc0_w16_blocktime_ab.py` and calls it, so every threshold
(n≥6, ratio≥1.10, sign≥80%, effect≥3x A/A half-width), the mechanism claim and
the width-8 regression guard are the same code that produced the original
REJECT. The only thing it changes is which launches are fed in, and it prints
the unmodified verdict next to the stratified one so they can never be confused.

It classifies each width-16 sub-launch by **effective lanes**
(`cpu_s_per_token / ms_token`), fast iff ≥ **13.0**. That cut is not fitted
here: it is the midpoint of the 3.1-lane gap published from **A/A data alone**
in [`2026-08-24-acc0-w16-null-page-backing.md`](2026-08-24-acc0-w16-null-page-backing.md),
fixed before any A/B launch was re-scored.

**One of its two gates failed, and the failure is the interesting part.** The
gate written first computes the slow-mode rate per *configuration*, following
the harness's arm rotation, and refuses to stratify if the two differ by more
than 2x — the check for whether *filtering* is arm-selective. On the original
run it reports **33.3% vs 25.0%, ratio 1.33, PASS.** It is a correct check and
it is blind to the defect, because the ratio is formed from the **control and
test sub-launches specifically** and the rotation pools the A/A slot into
whichever configuration it ran. Pooled, the arms look balanced. Paired, they are
4 against 1.

So the file now prints a second diagnostic unconditionally:

```
PAIRED MODE IMBALANCE (control vs test, the arms the ratio is formed from)
   control: slow 4/8 = 50.0%
      test: slow 1/8 = 12.5%
  net imbalance 37.5% of launches; at the published 1.687x mode ratio this
  can manufacture up to +0.2576 of ratio on its own
  UNUSABLE -- bar is 15%
```

against this run's:

```
   control: slow 7/24 = 29.2%
      test: slow 9/24 = 37.5%
  net imbalance 8.3% ... up to +0.0573 ... USABLE
```

**A run whose control and test arms drew the slow mode at different rates cannot
support any claim about the ratio, accept or reject.** That check costs nothing,
needs no extra launches, and would have flagged the original run the day it was
taken. It is now the first thing printed.

## Stratification works; it just had nothing to find

The point of stratifying was to sharpen the instrument, and it does:

| | unstratified (n=24) | fast mode only (n=8) |
|---|---:|---:|
| A/A half-width | 0.1478 | **0.0323** |
| threshold the effect must clear | +0.4434 | **+0.0970** |
| measured effect | −0.0117 | −0.0111 |

Removing the mode shrinks the null **4.6x** and drops the bar from +44% to
+9.7%. This is the unblock the null record predicted, and it is now
demonstrated rather than asserted. It simply does not rescue *this* candidate,
because the candidate is flat in both modes.

Note the stratified n is 8 out of 24 launches. Each launch runs three
independent width-16 processes and each draws the mode independently, so at a
~35% slow rate only about a third of launches have all three arms fast. **The
stratified arm is expensive: budget roughly 3x the launches.**

## The mechanism claim also inverts

The original run's strongest non-verdict evidence was that the mechanism moved
in the predicted direction: `sys_frac` fell 0.280 → 0.192 at 88% sign
consistency, which is what removing a straggler wait should look like. At n=24
that becomes **+0.0111 at 46% sign** — a coin flip — and in the fast mode
**+0.0043 at 50%**. The `sys_frac` fall was the same artifact seen from the
other side: the slow mode *is* a high-`sys` mode (0.315 vs 0.140–0.200), so an
arm that drew fewer slow launches necessarily shows lower `sys_frac`. The
mechanism agreed with the effect because both were the mode.

**This is the cautionary part.** A directionally-correct, high-sign-consistency
mechanism reading did not provide independent confirmation, because it was
downstream of the same nuisance variable. A mechanism check only corroborates
if it is independent of what is confounding the primary metric.

## The control that held, and why it matters

Stealing redistributes work, it does not remove it, so **user CPU per token must
not move**. It does not, in either run:

| | steal=1 | steal=2 |
|---|---|---|
| original (n=12 each) | 0.04780 | 0.04801 |
| this run (n=37 / 35) | 0.04810 | 0.04836 |

Total spread **1.124x across both configurations and both modes**, against
1.69x on wall time. This is the null-immune metric the A/A record identified,
behaving exactly as predicted on data taken before that record existed — and it
is a genuine independent replication of "the modes differ in waiting, not in
work", from a run whose purpose was something else entirely.

It is also the right *shape* of control for a load-balance candidate: flat user
CPU says the candidate did not quietly add or remove work, which is what would
have made the wall-time reading mean something other than balance.

## What is and is not disposed of

- **`STEAL_TILES_PER_WORKER=2` is closed as a throughput candidate at width 16
  on the zero-gap decode loop.** Not "unproven" — measured flat at an
  instrument sharp enough to have resolved +9.7%.
- The width-8 regression guard does not fire in either run (1.0090 / 1.0039), so
  the in-tree comment about 2x tiling regressing Qwen3 projections is not
  reproduced here either. That comment concerns a different model and shape and
  is not contradicted by this.
- **The 22.2-point straggler wait measured in
  [`2026-08-23-acc0-width-16-worker-attribution.md`](2026-08-23-acc0-width-16-worker-attribution.md)
  is untouched and remains the open target.** What is now known is that making
  spare tiles available does not collect it. Since the A/A record showed the
  slow mode has identical user work and +170% `sys`, the straggler is not a
  partitioning problem that more tiles can fix — a worker that is slow for
  reasons of its own gets slower per tile too, so handing its work to others
  requires knowing it is slow, which static spare tiles do not.

## Reproducing

```bash
HOSTLOCK_OWNER=roy scripts/hostlock.sh run --gate 6 --wait \
  --reason "steal-tiles re-test" -- \
  python3 crates/onnx-runtime-ep-cpu/benches/acc0_w16_blocktime_ab.py \
    --binary target/release/deps/int4_decode_loop_ab-<hash> \
    --env-name ONNX_GENAI_CPU_DECODE_STEAL_TILES_PER_WORKER \
    --control 1 --test 2 --hold ONNX_GENAI_CPU_DECODE_BLOCKTIME_US=500 \
    --launches 26 --tokens 384 --reps 3 --deadline-min 105 \
    --out steal_ab.json

python3 crates/onnx-runtime-ep-cpu/benches/acc0_w16_mode_stratified.py \
    --json steal_ab.json          # scores, runs nothing
```

The A/B harness is unmodified — same file, same rule, `--env-name` pointed at a
different knob, which is what it was built to allow. The scorer runs nothing and
can be pointed at any JSON either harness has already produced, including
archived ones; that is how the original run was re-examined without re-running
it.
