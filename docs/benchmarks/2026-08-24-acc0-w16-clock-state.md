# The width-16 mode is not a clock/boost state

Date: 2026-08-24 · Owner: Roy · Base: `7bf32c5f4`
Probe: `crates/onnx-runtime-ep-cpu/benches/acc0_w16_clock_state.py`

## Question

Of the candidates for the width-16 per-launch bimodality (~4.0 vs ~6.0 ms/token),
placement, THP/page backing, foreign load and static spare-tile steal are all
closed. Two remained: weight-arena placement across the two L3/CCX domains, and
**per-launch clock/boost state**. This closes the second.

## This host has no direct clock instrument

Worth recording, because it constrains every future frequency question here:

| instrument | status |
|---|---|
| `cpufreq` sysfs | absent |
| `/proc/cpuinfo` `cpu MHz` | **constant 2870.7 in all 18 launches, both modes** — nominal, not a reading |
| hardware PMU (`perf stat -e cycles`) | `<not supported>` (no vPMU) |
| `/dev/cpu/*/msr` (APERF/MPERF) | `Permission denied` |

The probe carries an explicit nominal-field control that fires on the constant
`cpu MHz` and refuses to convert it into a free REJECT. That control earned its
place immediately: without it, a field that is constant *by construction* would
have produced a confident "the clock does not differ between modes" that was
really "this field never differs from anything".

## The bound that needs no frequency counter

A clock drop is separable from every other slowdown mechanism by what it does to
**CPU-time**. Lowering the clock by factor `k` makes fixed work occupy `1/k`
times as many wall-seconds *on-CPU*, so CPU-time per token rises by the same
factor as wall time. Contrast:

| mechanism | wall/token | CPU-time/token |
|---|---|---|
| lower clock | up | **up by the same factor** |
| SMT contention | up | **unchanged** (measured: 1.86x throughput cost, 0.0% CPU-time cost) |
| parking / not running | up | **down** |

So the hypothesis makes a sharp prediction from data already being collected.

## Result — REJECT

18 trusted launches, quiet host under `hostlock.sh`, pre-registered rule
(ACCEPT if ≥75% of the required CPU-time inflation is present, REJECT if ≤25%):

| quantity | fast | slow | ratio |
|---|---|---|---|
| wall / token | 3.9540 ms | 6.0200 ms | **1.5225** |
| user CPU / token | 0.04786 s | 0.04906 s | **1.0250** |
| total CPU / token | 0.06115 s | 0.06240 s | 1.0204 |
| sys CPU / token | 0.01302 s | 0.01333 s | 1.0240 |

A clock drop producing 1.5225x wall requires a user/token ratio of 1.5225.
Observed 1.0250 — **4.8% of the required inflation**, against a REJECT threshold
of 1.1306.

**VERDICT: REJECT.** The slow mode retires its work in essentially the same
CPU-time as the fast mode. A lower clock cannot leave CPU-time per token
unchanged.

This rejects on a magnitude bound, not a correlation, so it does not need a
balanced sample of the two modes — one slow launch that retired its work in the
usual CPU-time is enough to kill a 41% clock drop. This run drew 1 slow launch
in 18, which is why the bound form matters.

## The sharper reading: the missing lanes are not running at all

Both user *and* sys CPU per token are flat (1.025, 1.024) while wall is +52% and
realized lanes fall 15.5 → 12.2. The ~3.3 missing lanes are therefore consuming
no CPU in either mode — they are not running, rather than spending longer in the
kernel.

**This corrects an earlier reading of mine.** The 2026-08-24 null record
reported the slow mode as `+4.6% user, +170% sys`, and I generalised from it
that the mode difference was yield-loop time. In this run sys/token is flat to
2.4%. Both runs are real; what differs is which slow launch was drawn. The
defensible statement across both is the one that holds in each: **user CPU per
token is flat between modes** (+2.5% here, +4.6% there) **while wall moves
1.5–1.7x**. The sys behaviour is not stable across slow launches and should not
be carried as a mechanism until it is measured over several of them. Corrected
in the ledger rather than deleted.

## Remaining candidate

Weight-arena placement across the two L3/CCX domains is now the only live
hypothesis on the list. It is consistent with everything above — it would not
change worker placement, would not change CPU-time per token, and would leave
lanes idle waiting on a straggler whose loads cross the interconnect.
