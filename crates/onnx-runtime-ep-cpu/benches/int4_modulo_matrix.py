#!/usr/bin/env python3
"""Finish the `dequant_panel_avx2` modulo-elimination matrix (#1809, refs #1676).

#1809 merged the elimination with a partial matrix: block-16 decode at 1.015x,
prefill nulls at m = 64/256/512, and m = 1 and m = 8 **withheld** because their
A/A null came back at 5.31% and 4.62%. This fills the withheld rows in and
takes the sweep at every power of two from 8 to 512, plus the block-16 decode
row again on current main.

Three arms, three separately built binaries from one source tree
(`build_arms.sh`):

  before   let offset_in_block = (depth + q) % block_size;
  after    let offset_in_block = offset_base + q;
  aa       a byte-identical copy of `after`

`aa` is the null. It is a *separate file* rather than a second run of the same
path, so it is a genuinely independent launch and pays every per-launch cost
the real arms pay -- ASLR, page backing, first-touch. A null taken any other
way understates the noise floor it is supposed to bound.

Method, following #1809's correction: the host gate is the **CPU efficiency of
the run itself** -- `os.wait4` rusage `(utime + stime) / wall` -- and not an
instantaneous runnable count sampled at run boundaries. A 2-second run has room
for a burst that starts after the opening sample and ends before the closing
one; that is how a 52% A/A null once passed a "host clean" check. A process
pinned to one core that is not being descheduled spends ~1.00 CPU-seconds per
wall-second, so this measures the thing directly.

Arms are interleaved **at launch granularity and rotated per round**, and every
row is reported as a distribution over independent launches. A single paired
A/B is not reported at any width: the decode loop on this host is bimodal per
process launch, and one pairing can be dominated by which mode each side
landed in.

Co-tenancy (`--co-tenant`)
--------------------------
Everything above is a *quiet-host* instrument: it pins to an idle core, and
both of its host gates exist to throw away any launch that was not alone. The
recorded correction on #1729 says that is not sufficient evidence for a
default:

    CPU scheduling and performance policy must not assume exclusive access to
    the machine. [...] A policy that wins only under exclusive quiet-host
    conditions is not a valid default.

The change this harness measures (`dequant_panel_avx2`'s eliminated modulo,
#1809/#2102) ships **on by default with no opt-out**, so it owes that evidence.
`--co-tenant` supplies it by injecting load deliberately instead of gating it
out, in the two modes this host actually has:

  smt    one pinned scalar-throughput spinner on the measured core's SMT
         sibling. Contends for execution units without contending for
         timeslices -- the mode `(utime+stime)/wall` is structurally blind to.
  dram   pinned streaming-memcpy hogs on *other* physical cores. Contends for
         memory bandwidth and shared L3, not for the measured core at all.

Both arms stay interleaved and rotated, so the co-tenant is common-mode and the
ratio remains a fair A/B; what changes is the regime the kernel runs in.

The gate is inverted rather than removed, and that is the whole design: a
co-tenant arm whose injected load never materialized is a quiet-host arm
wearing a busy-host label, and it would report a clean pass for exactly the
claim it was built to test. So each contended launch must *demonstrate* its
contention -- measured busy fraction on the co-tenant's own cpus, floored, per
launch -- and the gates that are still meaningful in that mode keep firing
(see `admit`).
"""

import argparse
import json
import os
import random
import resource
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
# Written by `int4_modulo_arms.sh`; override together with `MOD_ARMS_OUT`.
BIN = os.environ.get("MOD_ARMS_OUT", os.path.join(ROOT, "target/int4-modulo-arms"))
sys.path.insert(0, HERE)
import acc0_gap_matrix as H  # noqa: E402

# cpu 0 has a permanent external competitor on this host and cpu 1 is its SMT
# sibling, so a run pinned there is contended by construction. cpu 4 is an even
# cpu (one per physical core) away from both.
PIN = os.environ.get("MOD_PIN", "4")
CPU_EFF_FLOOR = 0.95

# `(utime+stime)/wall` measures time spent ON a logical cpu. SMT contention
# steals throughput WITHOUT stealing time: a competitor on the pinned cpu's
# hyperthread sibling shares the physical core's execution units, so the
# benchmark keeps its timeslice, scores a perfect 1.000, and is admitted --
# while delivering roughly half the work. Measured on this host at PIN=4 with
# a spinner on cpu5: 0.536x throughput at eff=1.000, against 0.976x for the
# same spinner on a *different* physical core. So the efficiency gate is sound
# against descheduling and structurally blind to SMT, and these are two
# different contention modes needing two different instruments.
#
# Ceiling set from measurement, not taste: an idle sibling reads 0.007 with
# the driver parked and 0.040 with it unpinned, a loaded one reads 1.000.
# 0.15 sits ~4x above the noisy end and ~7x below a real competitor.
SMT_SIBLING_CEIL = float(os.environ.get("MOD_SMT_CEIL", "0.15"))

#: Modes for `--co-tenant`. See the module docstring for what each contends
#: for; `none` is the historical quiet-host instrument.
CO_TENANT_MODES = ("none", "smt", "dram")

#: The busy fraction an injected co-tenant must actually reach on its own cpus
#: for the launch to count. This is a *floor*, and it is the inverse of the
#: ceiling above: the ceiling exists because unasked-for contention invalidates
#: a quiet-host number, and the floor exists because absent contention
#: invalidates a busy-host one. A single spinner pinned to an otherwise idle
#: cpu reads ~1.000; 0.90 leaves room for the spawn/teardown edges of the
#: window without admitting a cpu that was idle for a tenth of the run.
CO_TENANT_FLOOR = float(os.environ.get("MOD_COTENANT_FLOOR", "0.90"))

#: Set once by `main`. `cpus` are the cpus the injected load is pinned to, and
#: are watched per launch exactly the way the SMT siblings are.
CO_TENANT = {"mode": "none", "cpus": []}


def parse_cpu_list(spec):
    """Expand a cpu-list ("4", "4,6", "0-3") into a set of ints."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def sibling_cpus(pin):
    """The SMT siblings of `pin` that `pin` does not already occupy.

    Empty means the gate below cannot fire -- either the host has no SMT, or
    the pin already covers every sibling of its physical cores. That is a
    meaningful state and `main` reports it, because a gate that cannot fail
    reads exactly like a gate that passed.
    """
    pinned = parse_cpu_list(pin)
    sibs = set()
    for cpu in pinned:
        path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        try:
            with open(path) as fh:
                sibs |= parse_cpu_list(fh.read().strip())
        except OSError:
            continue
    return sorted(sibs - pinned)


def cpu_busy_jiffies(cpus, path="/proc/stat"):
    """(busy, total) jiffies summed over `cpus`, or None if there are none."""
    if not cpus:
        return None
    want = {f"cpu{c}" for c in cpus}
    busy = total = 0
    with open(path) as fh:
        for line in fh:
            fields = line.split()
            if fields and fields[0] in want:
                vals = [int(x) for x in fields[1:11]]
                total += sum(vals)
                busy += sum(vals) - (vals[3] + vals[4])  # minus idle + iowait
    return busy, total


def sibling_busy_fraction(before, after):
    """Fraction of the window the SMT siblings spent NOT idle."""
    if before is None or after is None:
        return None
    span = after[1] - before[1]
    return ((after[0] - before[0]) / span) if span > 0 else 0.0


SIBLING_CPUS = sibling_cpus(PIN)


# ---------------------------------------------------------------------------
# The injected co-tenant
# ---------------------------------------------------------------------------
#
# Re-executes this file rather than importing a second module, so the load
# generator cannot drift away from the harness that documents it, and so the
# `benches/` gate conformance check keeps seeing one file that spawns and one
# file that holds the lock -- the same file.

def _cotenant_spin(parent):
    """Scalar-throughput load: contends for execution units, not for time.

    The loop is integer ALU work with a carried dependence, which is what a
    hyperthread sibling steals issue slots from. Sebastian's `cpu_work_probe.py`
    measures the same shape from the other side.
    """
    x = 0
    while True:
        for _ in range(20000):
            x = (x * 1103515245 + 12345) & 0xFFFFFFFF
        if os.getppid() != parent:
            return x


def _cotenant_stream(parent, mib=96):
    """Streaming-memcpy load: contends for memory bandwidth and shared L3.

    Both buffers are far larger than either 32 MiB L3 instance on this host, so
    the copy misses to DRAM rather than recirculating in cache -- the point is
    to take bandwidth away from the measured core, not to warm it.
    """
    src = bytearray(mib << 20)
    dst = bytearray(mib << 20)
    while True:
        dst[:] = src
        if os.getppid() != parent:
            return len(dst)


#: Worker entry point, dispatched from `main` before argparse. `parent` is
#: checked on every pass: a co-tenant that outlives the harness is a runaway
#: load generator with nobody left to stop it, and this box has eight agents
#: on it.
CO_TENANT_WORKERS = {"smt": _cotenant_spin, "dram": _cotenant_stream}


def default_hog_cpus(pin=None, sibs=None, count=8, online=None):
    """One hog per physical core, skipping the measured core and its sibling.

    Skipping the sibling is not tidiness: a hog placed there would make the
    `dram` arm quietly an `smt`+`dram` arm, and the two are being compared.

    Also skips cpu 0/1: cpu 0 has a permanent external competitor on this host,
    so a hog placed there is measuring somebody else's load as well as its own.
    Even cpus only -- one logical cpu per physical core, so `count` hogs take
    `count` cores rather than `count/2` of them twice over.
    """
    pinned = parse_cpu_list(PIN if pin is None else pin)
    avoid = pinned | set(SIBLING_CPUS if sibs is None else sibs) | {0, 1}
    if online is None:
        try:
            online = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            online = list(range(os.cpu_count() or 1))
    out = [c for c in online if c % 2 == 0 and c not in avoid]
    return out[:count]


class CoTenant:
    """Deliberate, pinned, and measured external load for the duration of a run.

    Not a substitute for the host lock -- it is the opposite. The lock keeps
    *other* agents' load off the box so that the only co-tenant is this one,
    which is what makes the contention a controlled variable instead of the
    uncontrolled one every gate in this file exists to reject.
    """

    def __init__(self, mode, cpus):
        self.mode = mode
        self.cpus = list(cpus)
        self.procs = []

    def __enter__(self):
        if self.mode == "none" or not self.cpus:
            return self
        for cpu in self.cpus:
            self.procs.append(subprocess.Popen(
                ["taskset", "-c", str(cpu), sys.executable, os.path.abspath(__file__),
                 "--co-tenant-worker", self.mode, str(os.getpid())],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ))
        # Let the load reach steady state before the first launch is timed,
        # rather than letting the first rep pay the ramp and then discarding it
        # for missing the floor.
        time.sleep(1.5)
        print(f"  co-tenant {self.mode}: {len(self.procs)} worker(s) on cpus {self.cpus}",
              flush=True)
        return self

    def __exit__(self, *exc):
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=10)
        self.procs = []
        return False


def admit(eff, sib, hog, mode="none", eff_floor=CPU_EFF_FLOOR,
          smt_ceil=SMT_SIBLING_CEIL, cot_floor=CO_TENANT_FLOOR):
    """Which gate rejects this launch, or `None` if it is admitted.

    One function so the modes are readable side by side, because the
    interesting part is which gates *stay on* under a co-tenant rather than
    which are lifted.

    * `smt`  -- the sibling is loaded on purpose, so its ceiling would discard
      every launch and is replaced by a floor on the same measurement. Nothing
      else is lifted: the efficiency floor still catches descheduling, which is
      a different failure and is not what was asked for.
    * `dram` -- the hogs are on other physical cores, so the measured core's
      SMT sibling is still supposed to be idle and its ceiling still means what
      it did. A spinner that wanders onto the sibling would otherwise be
      credited to the bandwidth arm.

    The floor is the reason this is worth having. Lifting a gate is one line;
    lifting it and asserting nothing in its place produces a busy-host arm that
    is silently a quiet-host arm, reports a clean pass, and reads identically
    to a real one in the artifact.
    """
    if mode == "smt":
        if sib is None or sib < cot_floor:
            return "cotenant"
    elif mode == "dram":
        if hog is None or hog < cot_floor:
            return "cotenant"
        if sib is not None and sib > smt_ceil:
            return "smt"
    else:
        if sib is not None and sib > smt_ceil:
            return "smt"
    if eff < eff_floor:
        return "eff"
    return None


def starved_gate(mode):
    """Name the gates that were actually on, for the "it ate an arm" message.

    Under a co-tenant the likeliest cause of total starvation is not a noisy
    host but a co-tenant that died or never started, and a message naming only
    the efficiency floor sends the reader to the wrong place.
    """
    parts = [f"cpu-efficiency floor {CPU_EFF_FLOOR}"]
    if mode == "smt":
        parts.append(f"co-tenant floor {CO_TENANT_FLOOR} on siblings {SIBLING_CPUS}")
    elif mode == "dram":
        parts.append(f"co-tenant floor {CO_TENANT_FLOOR} on cpus {CO_TENANT['cpus']}")
        parts.append(f"SMT ceiling {SMT_SIBLING_CEIL}")
    else:
        parts.append(f"SMT ceiling {SMT_SIBLING_CEIL}")
    return "the host gates (" + "; ".join(parts) + ")"


def cotenancy_verdict(rows):
    """Pre-registered before the first contended launch (2026-08-27).

    The question is not "is the win bigger or smaller under load" -- that is
    reported but decides nothing. It is the one the correction asks:
    **does the default regress on a busy host?**

      VOID  if any row's A/A interval excludes 1.000. The instrument is biased
            somewhere in a matrix this small; re-take it rather than read
            around it. Strict on purpose -- a rule that reads around a broken
            A/A can hide a loss behind it.
      FAIL  if any row's A/B interval lies entirely below 1.000, i.e. a `loss`
            verdict. The change is then a quiet-host-only win and is not valid
            as an unconditional default.
      PASS  otherwise: every row is a gain or a null under injected load.
    """
    unreadable = [r["label"] for r in rows if not r["aa_brackets_unity"]]
    if unreadable:
        return {"verdict": "VOID", "rows": unreadable,
                "why": "A/A excludes 1.000 at: " + ", ".join(unreadable)}
    losses = [r["label"] for r in rows if r["verdict"] == "loss"]
    if losses:
        return {"verdict": "FAIL", "rows": losses,
                "why": "interval entirely below 1.000 at: " + ", ".join(losses)}
    return {"verdict": "PASS", "rows": [r["label"] for r in rows],
            "why": "no row regresses under injected load"}



def _timed_launch(binary, env_extra, timeout, drop_probe_ms=False):
    """Run one pinned launch and measure the three things the gates read.

    Returns `(stdout, cpu_eff, sibling_busy, cotenant_busy)`. The co-tenant's
    cpus are watched exactly the way the SMT siblings are, and for the same
    reason: the harness should not have to take on trust that the load it asked
    for is the load the run actually saw.

    The co-tenant workers are children of this process and are never reaped
    until teardown, so none of their CPU time lands in these `RUSAGE_CHILDREN`
    deltas -- the efficiency figure stays the benchmark's own.
    """
    env = dict(os.environ)
    env.update(env_extra)
    if drop_probe_ms:
        env.pop("PROBE_MS", None)
    argv = ["taskset", "-c", PIN, binary, "--bench"]
    watch = CO_TENANT["cpus"]
    sib_before = cpu_busy_jiffies(SIBLING_CPUS)
    hog_before = cpu_busy_jiffies(watch)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    proc = subprocess.run(
        argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        timeout=timeout, check=True,
    )
    wall = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    sib = sibling_busy_fraction(sib_before, cpu_busy_jiffies(SIBLING_CPUS))
    hog = sibling_busy_fraction(hog_before, cpu_busy_jiffies(watch))
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    return proc.stdout.decode(), (cpu / wall if wall > 0 else 0.0), sib, hog


def launch(binary, env_extra, timeout=1800):
    """One prefill launch. `rows` maps m -> dict of the printed columns."""
    out, eff, sib, hog = _timed_launch(binary, env_extra, timeout, drop_probe_ms=True)
    rows = {}
    for line in out.splitlines():
        f = line.split()
        if len(f) == 8 and f[0].isdigit():
            rows[int(f[2])] = {
                "k": int(f[0]), "n": int(f[1]),
                "cold_ms": float(f[3]), "steady_ms": float(f[4]),
                "gflops": float(f[5]), "sum": f[6], "fnv": f[7],
            }
    return rows, eff, sib, hog


def decode_launch(binary, env_extra, timeout=1800):
    out, eff, sib, hog = _timed_launch(binary, env_extra, timeout)
    rec = {}
    for line in out.splitlines():
        f = line.split()
        # `  cold/steady  ms_token  ms_token_p90  tokens_s_total  spread_%`
        if len(f) == 5 and f[0] in ("cold", "steady"):
            rec[f[0]] = float(f[1])
        elif line.startswith("checksum="):
            rec["checksum"] = line.split("=", 1)[1].strip()
    rec["raw"] = out
    return rec, eff, sib, hog



#: Fixed so the published intervals are reproducible rather than merely
#: recomputable to something similar.
BOOTSTRAP_SEED = 20260825
BOOTSTRAP_RESAMPLES = 20000


def bootstrap_ratio(before_vals, after_vals, resamples=BOOTSTRAP_RESAMPLES):
    """Percentile bootstrap interval for `median(before) / median(after)`.

    The point estimate on its own is not reportable on this host. The
    launch-to-launch spread reaches 100% while the median A/A null is 0.14%, so
    a ratio of medians is stable and a ratio of any single pairing is not -- and
    only the interval says which of those the reader is looking at.

    Launches are resampled, not individual timings: the launch is the unit that
    varies (placement, page backing, which decode mode the process landed in),
    and resampling below it would understate the spread it is there to capture.

    Percentile rather than BCa. The estimator is a smooth ratio of medians on
    50+ independent samples per arm and the intervals here are wide relative to
    the bias a BCa correction would remove, so the extra machinery would buy
    precision the measurement does not have.
    """
    if not before_vals or not after_vals:
        return (float("nan"), float("nan"))
    rng = random.Random(BOOTSTRAP_SEED)
    out = []
    for _ in range(resamples):
        ra = [rng.choice(before_vals) for _ in before_vals]
        rb = [rng.choice(after_vals) for _ in after_vals]
        mb = statistics.median(rb)
        out.append(statistics.median(ra) / mb if mb else float("nan"))
    out.sort()
    return (out[int(0.025 * resamples)], out[int(0.975 * resamples)])


def verdict(lo, hi):
    """What the interval permits you to say, rather than what the point says."""
    if lo > 1.0:
        return "gain"
    if hi < 1.0:
        return "loss"
    return "null"


def ratio_stats(before_vals, after_vals):
    """Medians over independent launches, and the median ratio.

    Reported as `before / after`, so > 1.000 means `after` is faster.
    """
    mb, ma = statistics.median(before_vals), statistics.median(after_vals)
    lo, hi = bootstrap_ratio(before_vals, after_vals)
    return {
        "ci_lo": lo,
        "ci_hi": hi,
        "verdict": verdict(lo, hi),
        "before_median_ms": mb,
        "after_median_ms": ma,
        "speedup": mb / ma if ma else float("nan"),
        "before_n": len(before_vals),
        "after_n": len(after_vals),
        "before_spread_pct": (max(before_vals) - min(before_vals)) / mb * 100 if mb else 0,
        "after_spread_pct": (max(after_vals) - min(after_vals)) / ma * 100 if ma else 0,
    }


def admission_columns(discarded, attempts, smt=None, cot=None, contention=None):
    """What the efficiency gate admitted, per arm and as a rate.

    `total` alone is what this harness used to report, and it cannot answer the
    question a reader actually has. "Every launch in this document passed the
    gate" is true of any gated dataset by construction -- discarded launches
    are not in the document -- so it conveys nothing. Keeping 17 of 18 and
    keeping 17 of 40 are very different datasets, and only the second is at
    real risk of having selected its reps.

    `max_arm_rate - min_arm_rate` is the number that closes the selection
    question. A gate that discards evenly across arms costs precision; a gate
    that discards one arm twice as often is choosing which launches of that
    arm survive, and the surviving set is no longer a fair sample of it.
    Reported, never enforced: this harness does not know what an acceptable
    spread is on an arbitrary host, and a threshold invented here would be the
    same unmeasured number it exists to expose.
    """
    rates = {
        arm: (discarded[arm] / attempts[arm] if attempts[arm] else 0.0)
        for arm in attempts
    }
    cols = {
        "total": sum(discarded.values()),
        "by_arm": dict(discarded),
        "attempts_by_arm": dict(attempts),
        "rate_by_arm": rates,
        "rate_spread": (max(rates.values()) - min(rates.values())) if rates else 0.0,
    }
    if smt is not None:
        # Which gate did the discarding. The two catch different contention:
        # the efficiency floor catches descheduling, the sibling ceiling
        # catches SMT theft that the floor scores as a perfect 1.000. Merging
        # them into one count would hide which mode this host actually has.
        cols["smt_by_arm"] = dict(smt)
        cols["smt_total"] = sum(smt.values())
        cols["smt_gate"] = (
            f"active on cpus {SIBLING_CPUS} (ceil {SMT_SIBLING_CEIL})"
            if SIBLING_CPUS
            else f"INACTIVE: pin {PIN!r} has no unoccupied SMT sibling, "
                 "so no launch can be discarded for SMT contention"
        )
        if CO_TENANT["mode"] == "smt":
            cols["smt_gate"] = (
                f"REPLACED BY A FLOOR: cpus {SIBLING_CPUS} carry the injected "
                f"co-tenant, so contention there is the treatment and the "
                f"ceiling is inverted (floor {CO_TENANT_FLOOR})"
            )
    if cot is not None:
        # The busy-host arm's own validity gate, counted separately from the
        # quiet-host ones. A non-zero total here does not mean the host was
        # noisy; it means the load this run asked for was not there, which is
        # the failure that would otherwise publish a quiet-host number under a
        # busy-host heading.
        cols["cotenant_by_arm"] = dict(cot)
        cols["cotenant_total"] = sum(cot.values())
        cols["cotenant_gate"] = (
            f"active: mode {CO_TENANT['mode']} on cpus {CO_TENANT['cpus']} "
            f"(floor {CO_TENANT_FLOOR})"
            if CO_TENANT["mode"] != "none"
            else "INACTIVE: no co-tenant injected, so this is a quiet-host run"
        )
    if contention:
        # Measured, not asserted. `min` is the one that matters: the floor is
        # per launch, so this is the worst contention any admitted launch saw.
        merged = [v for vals in contention.values() for v in vals if v is not None]
        if merged:
            cols["contention_admitted"] = {
                "median": statistics.median(merged),
                "min": min(merged),
                "max": max(merged),
                "n": len(merged),
                "by_arm": {
                    a: (statistics.median([v for v in vals if v is not None])
                        if any(v is not None for v in vals) else None)
                    for a, vals in contention.items()
                },
            }
    return cols


def prefill_matrix(rounds, block, shape, m_list, extra_env=None):
    arms = ["before", "after", "aa"]
    bins = {a: os.path.join(BIN, "prefill_" + ("after" if a == "aa" else a)) for a in arms}
    bins["aa"] = os.path.join(BIN, "prefill_aa")
    env = {"PROBE_BITS": "4", "PROBE_BLOCK": str(block), "PROBE_SHAPE": shape,
           "PROBE_M_LIST": ",".join(str(m) for m in m_list)}
    env.update(extra_env or {})
    samples = {a: {m: [] for m in m_list} for a in arms}
    cold = {a: {m: [] for m in m_list} for a in arms}
    fnv = {a: {} for a in arms}
    # Per arm, not one total. A single counter answers "how many launches were
    # thrown away" but not "were they thrown away evenly", and only the second
    # closes the selection question: the arms have genuinely different
    # runtimes, so a fixed efficiency floor can admit them at different rates
    # and quietly select for the launches that happened to land in a quiet
    # window. Attempts are counted alongside, so the rate is readable from the
    # artifact instead of inferred from `rounds`.
    discarded = {a: 0 for a in arms}
    attempts = {a: 0 for a in arms}
    smt = {a: 0 for a in arms}
    cot = {a: 0 for a in arms}
    # The contention every *admitted* launch actually saw, so the arm's regime
    # is a measurement in the artifact rather than a claim in the heading.
    contention = {a: [] for a in arms}
    mode = CO_TENANT["mode"]
    for r in range(rounds):
        # Rotate, so no arm is permanently first in a round and no arm
        # permanently inherits another's cache and frequency state.
        order = arms[r % len(arms):] + arms[: r % len(arms)]
        for arm in order:
            rows, eff, sib, hog = launch(bins[arm], env)
            attempts[arm] += 1
            gate = admit(eff, sib, hog, mode)
            if gate is not None:
                discarded[arm] += 1
                if gate == "smt":
                    smt[arm] += 1
                elif gate == "cotenant":
                    cot[arm] += 1
                continue
            contention[arm].append(sib if mode == "smt" else hog)
            for m, row in rows.items():
                if m not in samples[arm]:
                    continue
                samples[arm][m].append(row["steady_ms"])
                cold[arm][m].append(row["cold_ms"])
                fnv[arm].setdefault(m, set()).add(row["fnv"])
        print(f"  round {r + 1}/{rounds} done", flush=True)

    # If the gate ate an entire arm there is nothing to compare, and the
    # symptom without this is `statistics.StatisticsError: no median for empty
    # data` from inside the ratio -- which names neither the arm nor the gate.
    # The whole point of the admission columns is that a one-sided gate is
    # visible; the totally one-sided case should not be the one that reports
    # worst.
    starved = [a for a in arms if attempts[a] and discarded[a] == attempts[a]]
    if starved:
        raise SystemExit(
            f"{starved_gate(mode)} discarded every launch of "
            f"{', '.join(starved)} -- no sample survives to compare. "
            f"admission={admission_columns(discarded, attempts, smt, cot, contention)}"
        )
    table = []
    for m in m_list:
        row = {"m": m, "block": block, "shape": shape}
        row.update(ratio_stats(samples["before"][m], samples["after"][m]))
        aa = ratio_stats(samples["aa"][m], samples["after"][m])
        row["aa_null_pct"] = abs(aa["speedup"] - 1.0) * 100
        row["aa_speedup"] = aa["speedup"]
        row["aa_ci_lo"], row["aa_ci_hi"] = aa["ci_lo"], aa["ci_hi"]
        # The null has to be shown to contain 1.000, not assumed to. An A/A arm
        # whose own interval excludes 1.000 says the instrument is biased at
        # this row, and every verdict in the row is then unreadable.
        row["aa_brackets_unity"] = aa["ci_lo"] <= 1.0 <= aa["ci_hi"]
        row["cold_speedup"] = (
            statistics.median(cold["before"][m]) / statistics.median(cold["after"][m])
        )
        row["fnv"] = {a: sorted(fnv[a].get(m, [])) for a in arms}
        # Raw per-launch steady medians, so the ratio can be given a bootstrap
        # interval rather than a bare point estimate. The launch-to-launch
        # spread on this host reaches 100% while the median A/A null is 0.1%,
        # so the point estimate is only meaningful next to its interval.
        row["samples"] = {a: samples[a][m] for a in arms}
        row["bit_identical"] = (
            len(fnv["before"].get(m, set()) | fnv["after"].get(m, set()) | fnv["aa"].get(m, set())) == 1
        )
        table.append(row)
    return table, admission_columns(discarded, attempts, smt, cot, contention)


def decode_matrix(rounds, block, tokens):
    arms = ["before", "after", "aa"]
    bins = {a: os.path.join(BIN, "decode_" + a) for a in arms}
    env = {"PROBE_BLOCK": str(block), "PROBE_TOKENS": str(tokens),
           "PROBE_SESSIONS": "1", "ONNX_GENAI_CPU_DECODE_THREADS": "1"}
    samples = {a: [] for a in arms}
    cold_samples = {a: [] for a in arms}
    checks = {a: set() for a in arms}
    raw = {}
    discarded = {a: 0 for a in arms}
    attempts = {a: 0 for a in arms}
    smt = {a: 0 for a in arms}
    cot = {a: 0 for a in arms}
    contention = {a: [] for a in arms}
    mode = CO_TENANT["mode"]
    for r in range(rounds):
        order = arms[r % len(arms):] + arms[: r % len(arms)]
        for arm in order:
            rec, eff, sib, hog = decode_launch(bins[arm], env)
            attempts[arm] += 1
            gate = admit(eff, sib, hog, mode)
            if gate is not None:
                discarded[arm] += 1
                if gate == "smt":
                    smt[arm] += 1
                elif gate == "cotenant":
                    cot[arm] += 1
                continue
            contention[arm].append(sib if mode == "smt" else hog)
            raw.setdefault(arm, rec["raw"])
            if "checksum" in rec:
                checks[arm].add(rec["checksum"])
            if "steady" in rec:
                samples[arm].append(rec["steady"])
            cold_samples[arm].append(rec.get("cold", float("nan")))
        print(f"  decode round {r + 1}/{rounds} done", flush=True)
    # Same guard as the prefill matrix, and it has to be here rather than only
    # there: a fully starved `before` or `aa` reaches `ratio_stats` and dies on
    # `no median for empty data`, while a fully starved `after` returns the
    # generic "no parseable decode samples" below -- which names neither the
    # gate nor the arm, and so reads as a parsing bug rather than an admission
    # one.
    starved = [a for a in arms if attempts[a] and discarded[a] == attempts[a]]
    if starved:
        raise SystemExit(
            f"{starved_gate(mode)} discarded every decode launch "
            f"of {', '.join(starved)} -- no sample survives to compare. "
            f"admission={admission_columns(discarded, attempts, smt, cot, contention)}"
        )
    if not samples["after"]:
        return (
            {"error": "no parseable decode samples", "raw": raw},
            admission_columns(discarded, attempts, smt, cot, contention),
        )
    out = {"block": block, "tokens": tokens}
    out.update(ratio_stats(samples["before"], samples["after"]))
    aa = ratio_stats(samples["aa"], samples["after"])
    out["aa_null_pct"] = abs(aa["speedup"] - 1.0) * 100
    out["aa_ci_lo"], out["aa_ci_hi"] = aa["ci_lo"], aa["ci_hi"]
    out["aa_brackets_unity"] = aa["ci_lo"] <= 1.0 <= aa["ci_hi"]
    out["checksums"] = {a: sorted(checks[a]) for a in arms}
    out["bit_identical"] = len(checks["before"] | checks["after"] | checks["aa"]) == 1
    out["samples"] = samples
    return out, admission_columns(discarded, attempts, smt, cot, contention)


def route_proof(m_list, shape):
    """Prove, per row, that the modified line is on the route being timed.

    A source-level A/B has to rebuild between arms, so a null is uninterpretable
    without this: a change that never executed and a change that executed and
    cost nothing produce the same table. Timing cannot separate them; a
    deliberately wrong build can.

    Expected:
      before == after   everywhere  (the elimination is exact, so it is free)
      poison != after   on every row whose route reaches the line
      poison == after   on block 32, m = 1 -- the built-in control, because
                        that row takes the N-blocked decode kernel and never
                        calls the pack at all

    That last expectation is hardcoded from `int4_prefill_gebp_min_rows`'s gate
    rather than derived, which makes it a **tripwire as well as a control**: if
    the dispatch ever changes so that block 32 at m = 1 does reach the pack, or
    some other row stops reaching it, this reports FAIL. Read such a failure as
    "the routing moved" first and "the kernel broke" second -- `before ==
    after`, which is checked on every row independently, is the half that speaks
    to correctness.
    """
    arms = ["before", "after", "poison"]
    rows = {}
    for block in (16, 32):
        for arm in arms:
            env = {"PROBE_BITS": "4", "PROBE_BLOCK": str(block), "PROBE_SHAPE": shape,
                   "PROBE_M_LIST": ",".join(str(m) for m in m_list)}
            got, _, _, _ = launch(os.path.join(BIN, f"prefill_{arm}"), env)
            for m, row in got.items():
                rows.setdefault((block, m), {})[arm] = row["fnv"]
    print(f"{'block':>6} {'m':>5} {'before==after':>14} {'poison moves':>13}")
    ok = True
    for (block, m), got in sorted(rows.items()):
        identical = got["before"] == got["after"]
        moved = got["poison"] != got["after"]
        # Block 32 at m = 1 is the one row that is *supposed* not to move.
        expect_move = not (block == 32 and m == 1)
        ok = ok and identical and moved == expect_move
        note = "" if moved == expect_move else "  <-- UNEXPECTED"
        print(f"{block:>6} {m:>5} {str(identical):>14} {str(moved):>13}{note}")
    ok = ok and _gebp_off_control(m_list, shape, rows)
    print("route proof:", "PASS" if ok else "FAIL")
    return {"rows": {f"{b}/{m}": g for (b, m), g in rows.items()}, "pass": ok}


def _gebp_off_control(m_list, shape, rows):
    """Assert that `ONNX_GENAI_CPU_MM_INT4_GEBP=0` really does take the pack off
    the route, instead of documenting that it does.

    `--env`'s help calls this "verified", and for a long time that word rested on
    a run nobody had repeated and nothing re-checked. A claim that only exists in
    a docstring is indistinguishable from one that was never true: the env var is
    the control used to argue that a sub-2% residual is layout rather than the
    change, so if it ever stopped removing the pack, every argument leaning on it
    would keep reading exactly the same.

    The observable: at a row that *does* move with the pack on, turning the pack
    off must collapse `poison` onto `after`, because the poisoned line is then
    unreachable. Checksums only -- no timing, so this is contention-independent.

    Also assert the pack-off output *differs* from the pack-on output at that
    row. Without it the control passes vacuously if the flag were ignored
    entirely: an ignored flag leaves poison != after, which fails the collapse,
    but a flag that silently disabled *both* kernels' output would not.
    """
    block, m = 32, 8
    if m not in m_list:
        print(f"  gebp-off control: skipped, m = {m} not in --m-list")
        return True
    env = {"PROBE_BITS": "4", "PROBE_BLOCK": str(block), "PROBE_SHAPE": shape,
           "PROBE_M_LIST": str(m), "ONNX_GENAI_CPU_MM_INT4_GEBP": "0"}
    off = {}
    for arm in ("after", "poison"):
        got, _, _, _ = launch(os.path.join(BIN, f"prefill_{arm}"), env)
        off[arm] = got[m]["fnv"]
    collapsed = off["after"] == off["poison"]
    on_moves = rows[(block, m)]["poison"] != rows[(block, m)]["after"]
    changed_route = off["after"] != rows[(block, m)]["after"]
    good = collapsed and on_moves and changed_route
    print(f"  gebp-off control, block {block} m {m}: "
          f"pack-on moves={on_moves} pack-off collapses={collapsed} "
          f"pack-off differs from pack-on={changed_route}"
          f"{'' if good else '  <-- UNEXPECTED'}")
    return good


def park_driver_off_measured_core(extra=()):
    """Keep this process off the cpu it is measuring, and off that cpu's sibling.

    The driver is runnable for the whole launch -- it shepherds the child and
    reads /proc -- and it was previously unpinned, so the scheduler was free to
    place it on the measured core's SMT sibling and contend with the very
    benchmark it is timing. Measured: an idle sibling reads 0.040 busy with the
    driver unpinned versus 0.007 parked, a 6x self-inflicted floor.

    `extra` is the co-tenant's cpus. The driver must stay off those too, for a
    second reason: their busy fraction is a *measurement* now, and driver time
    landing there would be counted as injected load.

    Best effort. If no cpu is left over, stay put rather than fail a run over
    placement, but say so -- silently not parking is how this went unnoticed.
    """
    try:
        allowed = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return None
    keep_off = parse_cpu_list(PIN) | set(SIBLING_CPUS) | set(extra)
    free = allowed - keep_off
    if not free:
        print(f"  NOTE: cannot park driver off cpus {sorted(keep_off)} -- "
              f"only {sorted(allowed)} allowed; driver may contend with its own run")
        return None
    os.sched_setaffinity(0, free)
    return sorted(free)


def self_test():
    """Arithmetic only: no binary, no injected load, no lock, no host.

    Written against the defects, not the happy path. Every case below is one a
    plausible implementation of `--co-tenant` actually commits, and each one
    produces a run that looks entirely normal in the artifact:

      * lift the sibling ceiling for `smt` and assert nothing in its place, so
        an arm whose spinner died reports a quiet-host number under a
        busy-host heading;
      * lift *every* contention gate whenever a co-tenant is present, so a
        stray competitor on the measured core's sibling is credited to the
        bandwidth arm;
      * drop the efficiency floor under load, on the reasoning that the run is
        contended anyway -- descheduling is a different failure and still
        invalidates the launch;
      * read around a broken A/A instead of voiding the matrix, which lets a
        loss hide behind an interval that was never trustworthy.
    """
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(f"{name}: got {got!r}, want {want!r}")

    # -- quiet host: unchanged behaviour ------------------------------------
    check("quiet/clean", admit(0.99, 0.01, None, "none"), None)
    check("quiet/smt-competitor", admit(1.00, 0.99, None, "none"), "smt")
    check("quiet/descheduled", admit(0.70, 0.01, None, "none"), "eff")
    # A host with no free sibling cannot fire the ceiling, and must not be
    # rejected for it either.
    check("quiet/no-sibling", admit(0.99, None, None, "none"), None)

    # -- smt arm: the ceiling becomes a floor -------------------------------
    check("smt/loaded", admit(0.99, 0.99, None, "smt"), None)
    check("smt/spinner-never-started", admit(0.99, 0.01, None, "smt"), "cotenant")
    check("smt/spinner-died-halfway", admit(0.99, 0.50, None, "smt"), "cotenant")
    check("smt/unmeasurable", admit(0.99, None, None, "smt"), "cotenant")
    # Descheduling is not what was asked for and still invalidates the launch.
    check("smt/loaded-but-descheduled", admit(0.70, 0.99, None, "smt"), "eff")

    # -- dram arm: floor on the hogs, ceiling still live on the sibling -----
    check("dram/loaded", admit(0.99, 0.01, 0.99, "dram"), None)
    check("dram/hogs-never-started", admit(0.99, 0.01, 0.10, "dram"), "cotenant")
    check("dram/hogs-unmeasurable", admit(0.99, 0.01, None, "dram"), "cotenant")
    check("dram/stray-smt-competitor", admit(1.00, 0.99, 0.99, "dram"), "smt")
    check("dram/loaded-but-descheduled", admit(0.70, 0.01, 0.99, "dram"), "eff")

    # -- hog placement ------------------------------------------------------
    hogs = default_hog_cpus(pin="4", sibs=[5], count=8, online=list(range(32)))
    check("hogs/avoid-pin", [c for c in hogs if c == 4], [])
    check("hogs/avoid-contended-cpu0", [c for c in hogs if c in (0, 1)], [])
    check("hogs/one-per-physical-core", [c for c in hogs if c % 2], [])
    check("hogs/count", len(hogs), 8)
    # The sibling case has to be taken from an *odd* pin, or the even-cpu
    # filter answers it for free: siblings are adjacent here, so an even pin's
    # sibling is always odd and is excluded by parity whether or not the
    # avoidance works. Asserting it from pin 4 was a vacuous check that
    # survived removing the avoidance entirely.
    odd = default_hog_cpus(pin="5", sibs=[4], count=8, online=list(range(32)))
    check("hogs/avoid-even-sibling", [c for c in odd if c == 4], [])

    # -- the pre-registered rule -------------------------------------------
    ok = {"label": "m=1", "verdict": "null", "aa_brackets_unity": True}
    gain = {"label": "m=8", "verdict": "gain", "aa_brackets_unity": True}
    loss = {"label": "m=16", "verdict": "loss", "aa_brackets_unity": True}
    broken = {"label": "decode", "verdict": "gain", "aa_brackets_unity": False}
    check("rule/all-null", cotenancy_verdict([ok, gain])["verdict"], "PASS")
    check("rule/any-loss", cotenancy_verdict([ok, gain, loss])["verdict"], "FAIL")
    check("rule/broken-aa", cotenancy_verdict([ok, broken])["verdict"], "VOID")
    # A loss must never be reachable through a matrix the instrument could not
    # read: VOID takes precedence, so the answer is "re-take it", not "PASS".
    check("rule/loss-behind-broken-aa",
          cotenancy_verdict([loss, broken])["verdict"], "VOID")
    check("rule/names-the-row", cotenancy_verdict([ok, loss])["rows"], ["m=16"])

    for f in fails:
        print(f"FAIL {f}")
    print(f"self-test: {len(fails)} failure(s)")
    return not fails


def main():
    # Before argparse: a co-tenant worker is this same file re-executed, and it
    # must not touch the lock, the arms, or the self-test.
    if len(sys.argv) >= 4 and sys.argv[1] == "--co-tenant-worker":
        CO_TENANT_WORKERS[sys.argv[2]](int(sys.argv[3]))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=31)
    ap.add_argument("--decode-rounds", type=int, default=15)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--shape", default="small")
    ap.add_argument("--m-list", default="8,16,32,64,128,256,512")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--skip-prefill", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="arithmetic self-test of the admission gates and the "
                         "pre-registered co-tenancy rule; no binary, no load, "
                         "no lock, so it is safe on a shared runner")
    ap.add_argument("--route-proof", action="store_true",
                    help="checksum-only; proves each row's route without timing it")
    ap.add_argument(
        "--co-tenant", choices=CO_TENANT_MODES, default="none",
        help="inject external load for the whole run instead of gating it out. "
             "`smt` puts a scalar spinner on the measured core's SMT sibling; "
             "`dram` puts streaming-memcpy hogs on other physical cores. Both "
             "arms stay interleaved, so the co-tenant is common-mode and the "
             "ratio stays a fair A/B -- what changes is the regime. Required "
             "by the #1729 correction for any result offered as evidence for a "
             "default: a win measured only on a quiet host is not one.")
    ap.add_argument(
        "--co-tenant-cpus", default=None,
        help="cpu list for `--co-tenant dram` (default: one per physical core, "
             "skipping the measured core, its sibling, and cpu 0/1)")
    ap.add_argument(
        "--env", action="append", default=[], metavar="K=V",
        help="extra env for prefill launches. `--env ONNX_GENAI_CPU_MM_INT4_GEBP=0` "
             "is the layout control: it takes the pack off the route entirely, "
             "so any difference left between `before` and `after` is code layout "
             "and not the change. That it really does remove the pack is not "
             "taken on trust -- `--route-proof` asserts it, by checking the "
             "poisoned arm collapses onto `after` at a row that moves with the "
             "pack on. That control has to be run on the *same* rows as the "
             "claim; a different row can be a different kernel with different "
             "layout sensitivity. Note the flag's *timing* delta is "
             "unattributable by construction -- it swaps a whole algorithm and "
             "moves cache behaviour and layout with it -- so route claims rest "
             "on the poisoned checksum, never on this.")
    ap.add_argument("--out", default=os.path.join(BIN, "modulo_matrix.json"))
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(0 if self_test() else 1)

    aa = os.path.join(BIN, "prefill_aa")
    if not os.path.exists(aa):
        raise SystemExit(
            f"missing {aa} -- run `int4_modulo_arms.sh`, then "
            "`cp prefill_after prefill_aa` and `cp decode_after decode_aa` in that "
            "directory. The null arm is a separate file on purpose: it is then a "
            "genuinely independent launch that pays every per-launch cost the real "
            "arms pay."
        )
    for a in ("before", "after", "aa"):
        for kind in ("prefill", "decode"):
            p = os.path.join(BIN, f"{kind}_{a}")
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}")

    m_list = [int(v) for v in args.m_list.split(",")]
    if args.co_tenant == "smt":
        cot_cpus = list(SIBLING_CPUS)
        if not cot_cpus:
            raise SystemExit(
                f"--co-tenant smt needs an SMT sibling to load, and pin {PIN!r} "
                "has none free. Without one the arm would run quiet and report "
                "itself contended, which is the one outcome it must not have."
            )
    elif args.co_tenant == "dram":
        cot_cpus = (sorted(parse_cpu_list(args.co_tenant_cpus))
                    if args.co_tenant_cpus else default_hog_cpus())
        if not cot_cpus:
            raise SystemExit("--co-tenant dram has no cpus left to place hogs on")
    else:
        cot_cpus = []
    CO_TENANT["mode"] = args.co_tenant
    CO_TENANT["cpus"] = cot_cpus
    parked = park_driver_off_measured_core(cot_cpus)
    if args.route_proof:
        # No lock: checksums do not depend on who else is on the machine, and
        # holding the whole host to compute one would be antisocial.
        proof = route_proof(m_list, args.shape)
        with open(args.out, "w") as fh:
            json.dump(proof, fh, indent=2)
        print(f"wrote {args.out}")
        raise SystemExit(0 if proof["pass"] else 1)

    result = {
        "pin": PIN,
        "cpu_eff_floor": CPU_EFF_FLOOR,
        "rounds": args.rounds,
        "smt_sibling_cpus": SIBLING_CPUS,
        "smt_sibling_ceil": SMT_SIBLING_CEIL,
        "co_tenant": {"mode": args.co_tenant, "cpus": cot_cpus,
                      "floor": CO_TENANT_FLOOR},
        "driver_parked_on": parked,
    }
    with H.HostLock(
        H.bench_owner(),
        f"dequant_panel_avx2 modulo matrix: block{args.block} m={args.m_list}"
        f" + block16 decode ({args.rounds} rounds, co-tenant {args.co_tenant})",
    ), CoTenant(args.co_tenant, cot_cpus):
        result["lock"] = H.lock_provenance()
        if not args.skip_prefill:
            print(f"prefill matrix block={args.block} shape={args.shape}", flush=True)
        extra_env = dict(kv.split("=", 1) for kv in args.env)
        result["extra_env"] = extra_env
        table, disc = [], admission_columns({}, {})
        if not args.skip_prefill:
            table, disc = prefill_matrix(args.rounds, args.block, args.shape, m_list, extra_env)
        result["prefill"] = table
        # Kept for anyone reading an older artifact; the per-arm breakdown
        # beside it is the one that says whether the gate was even-handed.
        result["prefill_discarded_launches"] = disc["total"]
        result["prefill_admission"] = disc
        if not args.skip_decode:
            print("decode matrix block=16", flush=True)
            dec, ddisc = decode_matrix(args.decode_rounds, 16, args.tokens)
            result["decode"] = dec
            result["decode_discarded_launches"] = ddisc["total"]
            result["decode_admission"] = ddisc
    if args.co_tenant != "none":
        rows = [{"label": f"block{args.block} m={r['m']}",
                 "verdict": r["verdict"],
                 "aa_brackets_unity": r["aa_brackets_unity"]} for r in table]
        dec = result.get("decode")
        if dec and "error" not in dec:
            rows.append({"label": "block16 decode", "verdict": dec["verdict"],
                         "aa_brackets_unity": dec["aa_brackets_unity"]})
        result["cotenancy"] = cotenancy_verdict(rows)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\npin=cpu{PIN}  cpu_eff floor={CPU_EFF_FLOOR}  "
          f"discarded={result['prefill_discarded_launches']}")
    print(f"smt siblings={SIBLING_CPUS or 'NONE -- sibling gate inactive'}  "
          f"ceil={SMT_SIBLING_CEIL}  driver parked on={parked}")
    print(f"co-tenant={args.co_tenant}"
          + (f" on cpus {cot_cpus} (floor {CO_TENANT_FLOOR})"
             if args.co_tenant != "none" else " -- quiet-host run"))
    for label in ("prefill", "decode"):
        adm = result.get(f"{label}_admission")
        if not adm or not adm["attempts_by_arm"]:
            continue
        kept = {
            a: adm["attempts_by_arm"][a] - adm["by_arm"][a] for a in adm["by_arm"]
        }
        print(
            f"  {label} admitted: "
            + ", ".join(
                f"{a} {kept[a]}/{adm['attempts_by_arm'][a]}"
                for a in sorted(adm["by_arm"])
            )
            + f"  (discard-rate spread {adm['rate_spread']:.3f}"
            + " -- an uneven gate selects which launches of an arm survive)"
        )
        if adm.get("cotenant_total"):
            print(f"    {adm['cotenant_total']} discarded for missing the "
                  f"co-tenant floor: {adm['cotenant_by_arm']}")
        con = adm.get("contention_admitted")
        if con:
            print(f"    co-tenant busy on its own cpus, admitted launches: "
                  f"median {con['median']:.3f}  min {con['min']:.3f}  "
                  f"max {con['max']:.3f}  (n={con['n']})")
    if table:
        print(f"\nprefill block {args.block} ({args.shape}), {args.rounds} "
              f"independent launches per arm")
        print(f"{'m':>5} {'before ms':>10} {'after ms':>10} {'speedup':>8} {'95% CI':>18} "
              f"{'verdict':>8} {'A/A':>7} {'A/A 95% CI':>18} {'A/A ok':>7} {'bit-id':>7}")
        for row in table:
            print(f"{row['m']:>5} {row['before_median_ms']:>10.3f} {row['after_median_ms']:>10.3f} "
                  f"{row['speedup']:>8.4f} [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] "
                  f"{row['verdict']:>8} {row['aa_speedup']:>7.4f} "
                  f"[{row['aa_ci_lo']:.4f}, {row['aa_ci_hi']:.4f}] "
                  f"{str(row['aa_brackets_unity']):>7} {str(row['bit_identical']):>7}")
    print(f"\nIntervals: percentile bootstrap over launches, "
          f"{BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}.")
    if not all(r["aa_brackets_unity"] for r in table):
        print("WARNING: an A/A interval excludes 1.000 -- the instrument is biased "
              "at that row and its verdict is not readable.")
    if not all(r["bit_identical"] for r in table):
        print("WARNING: an arm produced different output bytes -- this A/B is not "
              "measuring an exact transformation.")
    if not args.skip_decode and "error" not in result["decode"]:
        d = result["decode"]
        print(f"\ndecode block 16, {args.decode_rounds} independent launches per arm")
        print(f"  before {d['before_median_ms']:.3f}  after {d['after_median_ms']:.3f}  "
              f"speedup {d['speedup']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}] {d['verdict']}  "
              f"A/A null {d['aa_null_pct']:.2f}% [{d['aa_ci_lo']:.4f}, {d['aa_ci_hi']:.4f}]  "
              f"bit-identical {d['bit_identical']}")
    if "cotenancy" in result:
        c = result["cotenancy"]
        print(f"\nco-tenancy rule (pre-registered): {c['verdict']} -- {c['why']}")
        print("  PASS means the default does not regress under injected load. "
              "It is not a claim that the win is the same size; the magnitudes "
              "above are the claim about size, and they are reported whichever "
              "way the rule goes.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
