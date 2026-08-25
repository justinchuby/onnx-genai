#!/usr/bin/env python3
"""Does physical page backing pick the width-16 straggler?

WHY THIS EXISTS
---------------
`acc0_w16_straggler_identity.py`, `..._aslr.py` and `..._window.py` establish
four things about the width-16 straggler, each against a rule written before
its first launch:

    real            excess concentration RISES with a 4x window, R = 1.690;
                    one lane last on a median 72% of 3840 ops (chance 0.067)
    not assignment  `ops_spread` = 0.0000 in 24/24 launches
    not placement   one lane->cpu map over 24 launches, victim moves anyway
    not layout      `setarch -R` gives concentration 0.267, the same number
                    as ASLR's 0.267

and a post-hoc reading of the same 73 launches adds that the last-arriving lane
is also the highest-`work_ns` lane in 0.667/0.667/0.684 of them -- the victim
usually *computes* longer rather than merely starting late.

So the selector has to make identical work, on a fixed core, at fixed virtual
addresses, take longer, and it has to differ between processes. **Physical**
page assignment is the family that fits: `setarch -R` pins virtual addresses,
but the kernel still hands out different physical frames on every exec, and the
large caches on this part are physically indexed, so set and bank conflicts
vary per process while the virtual layout does not.

THE LEVER, AND WHY IT IS THE ONLY ONE HERE
------------------------------------------
Physical frames cannot be read on this host -- `/proc/self/pagemap` returns
PFN 0 without `CAP_SYS_ADMIN` -- and the sysfs THP control is not writable. The
one unprivileged lever that changes physical backing is
`prctl(PR_SET_THP_DISABLE)`, which is inherited across `exec`:

    thp    default. `/sys/kernel/mm/transparent_hugepage/enabled` is
           `[always]` here and the host has 2.18 GB of AnonHugePages live, so
           the weight arena is backed by 2 MiB pages: a lane's slice is
           physically contiguous, conflicts are structured, and the huge page's
           physical alignment is drawn fresh on each exec.

    nothp  4 KiB pages. Frames are scattered, so whatever structured physical
           conflict a 2 MiB backing creates is averaged away across the slice.

If physical backing selects the victim, removing the structure must reduce the
imbalance.

WHY `work_skew` IS THE RIGHT METRIC HERE
----------------------------------------
Disabling THP will make everything slower (more TLB pressure). `work_skew` is
`max(work_ns)/mean(work_ns) - 1`, which is **scale-invariant**: if every lane
slows by the same factor, it does not move at all. So a uniform slowdown cannot
manufacture a result in either direction, which a wall-time comparison could.
Wall time is still reported, because a change large enough to alter the
workload's character should be visible to a reader.

PRE-REGISTERED RULE (written before the first launch)
-----------------------------------------------------
Trust: `acc0_w16_worker_split.trusted()`, unmodified. `MIN_PER_ARM = 10`.

Let `S(arm)` be the median `work_skew` of that arm.

    ACCEPT (physical backing selects the straggler) iff
        S(nothp) <= 0.60 * S(thp)          -- imbalance falls by >= 40%
    REJECT iff
        S(nothp) >= 0.85 * S(thp)          -- imbalance essentially survives
    otherwise REPORT NOTHING.

`straggler_share` excess over chance is reported alongside but does not gate,
because it is a count-based statistic whose window this probe does not vary.

CONTROLS
--------
1.  **The lever is verified, not trusted**, before any launch: a child that
    faults a 2 MiB-aligned 256 MiB mapping must report non-zero
    `AnonHugePages` in `/proc/self/smaps_rollup` without the wrapper and zero
    with it. This control has already earned its place -- a first attempt to
    verify the wrapper appeared to show it doing nothing, and the real cause
    was an unaligned test mapping that THP had never backed in either arm. An
    unverified lever produces two identical arms and a free REJECT.
2.  **Assignment stays equal** (`ops_spread <= 0.01`) in both arms.
3.  **Placement is unchanged**: each arm must show exactly one lane->cpu map,
    so the arms differ in page backing rather than in where lanes ran.
4.  Arms interleaved launch-by-launch with alternating order.

NOTE ON A PREVIOUS THP RESULT
-----------------------------
THP was closed earlier as a candidate for the width-16 *bimodality* -- the
per-launch fast/slow modes. That is a different question from the *straggler*,
which is an imbalance between lanes within a launch and is present in both
modes. This probe does not reopen the bimodality finding and does not depend
on it.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import acc0_w16_straggler_identity as I  # noqa: E402
import acc0_w16_worker_split as W  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
WRAPPER = HERE / "acc0_nothp_exec.py"

MIN_PER_ARM = 10
ACCEPT_RATIO = 0.60
REJECT_RATIO = 0.85
OPS_SPREAD_MAX = 0.01
WIDTH = 16
ARMS = ("thp", "nothp")

THP_SELFTEST = r"""
import ctypes
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
N = 256 << 20
p = libc.mmap(ctypes.c_void_p(0), ctypes.c_size_t(N + (2 << 20)), 3, 0x02 | 0x20, -1,
              ctypes.c_size_t(0))
base = (p + (2 << 20) - 1) & ~((2 << 20) - 1)
libc.madvise(ctypes.c_void_p(base), ctypes.c_size_t(N), 14)
buf = (ctypes.c_char * N).from_address(base)
for off in range(0, N, 4096):
    buf[off] = b"x"
for line in open("/proc/self/smaps_rollup"):
    if line.startswith("AnonHugePages"):
        print(line.split()[1])
"""


def anon_huge_kb(argv):
    """`AnonHugePages` a child reports after faulting a 2 MiB-aligned mapping.

    Passed on stdin rather than through `python3 -c`: a first version embedded
    the source in a shell command line, where the escaped newlines survived the
    shell as literal backslash-n and the child died of a syntax error. Both
    arms returned `None`, and the control caught it -- which is what it is for,
    but the quoting was still worth removing rather than escaping harder.
    """
    r = subprocess.run(argv, input=THP_SELFTEST, capture_output=True,
                       text=True, timeout=300)
    try:
        return int(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def verify_lever():
    on = anon_huge_kb(["python3", "-"])
    off = anon_huge_kb(["python3", str(WRAPPER), "python3", "-"])
    print(f"control: AnonHugePages default={on} kB  wrapper={off} kB")
    if not on or off != 0:
        print("VERDICT: ABORT -- PR_SET_THP_DISABLE is not in effect; nothing measured.")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--launches", type=int, default=14, help="per arm")
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--blocktime", type=int, default=0)
    ap.add_argument("--quiet-limit", type=int, default=40)
    ap.add_argument("--out", default="bb/w16_straggler_thp.json")
    ap.add_argument("--replay", default=None)
    args = ap.parse_args()

    if args.replay:
        return report(json.loads(pathlib.Path(args.replay).read_text()), args.launches)
    if not verify_lever():
        return 1
    print()

    records = []
    for i in range(args.launches):
        order = ARMS if i % 2 == 0 else tuple(reversed(ARMS))
        for arm in order:
            sub = argparse.Namespace(**vars(args))
            sub.binary = (f"python3 {WRAPPER} " if arm == "nothp" else "") + args.binary
            with W.H.LoadWatch() as watch:
                workers = W.parse_workers(W.run_width(sub, WIDTH))
            rec = {"widths": {str(WIDTH): W.derive(workers)},
                   "peak": watch.peak, "peak_limit": args.quiet_limit}
            keep = W.trusted(rec)
            table = I.lane_table(workers) if keep else None
            if table:
                arr = [w["last_arrivals"] for w in workers]
                table["straggler_share"] = max(arr) / max(1, sum(arr))
            records.append({"launch": i, "arm": arm, "trusted": keep, "table": table})
            pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(args.out).write_text(json.dumps(records, indent=1))
            if table:
                print(f"L{i:<2} {arm:<5} wall={table['wall_s']:.3f} "
                      f"skew={table['work_skew']:.3f} share={table['straggler_share']:.4f} "
                      f"ops_spread={table['ops_spread']:.4f} idx{table['straggler_idx']}",
                      flush=True)
            else:
                print(f"L{i:<2} {arm:<5} UNTRUSTED (peak={watch.peak})", flush=True)
            time.sleep(1)

    return report(records, args.launches)


def report(records, attempted) -> int:
    by = {a: [r["table"] for r in records if r["arm"] == a and r["table"]] for a in ARMS}
    print()
    for a in ARMS:
        print(f"{a:<5} trusted {len(by[a])}/{attempted}")
    if any(len(by[a]) < MIN_PER_ARM for a in ARMS):
        print(f"VERDICT: REPORT NOTHING -- need {MIN_PER_ARM} trusted launches per arm")
        return 0

    n = by[ARMS[0]][0]["n"]
    chance = 1.0 / n
    st = {}
    for a in ARMS:
        c = collections.Counter(t["straggler_idx"] for t in by[a])
        maps = len({json.dumps(sorted((int(k), v) for k, v in t["lane_cpu"].items()))
                    for t in by[a]})
        st[a] = {
            "skew": statistics.median(t["work_skew"] for t in by[a]),
            "share": statistics.median(t["straggler_share"] for t in by[a]),
            "wall": statistics.median(t["wall_s"] for t in by[a]),
            "spread": statistics.median(t["ops_spread"] for t in by[a]),
            "maps": maps,
            "conc": c.most_common(1)[0][1] / len(by[a]),
        }
        print(f"{a:<5} median skew={st[a]['skew']:.4f}  share={st[a]['share']:.4f} "
              f"(excess {st[a]['share'] - chance:+.4f})  wall={st[a]['wall']:.3f}  "
              f"ops_spread={st[a]['spread']:.4f}  maps={maps}  top-lane conc={st[a]['conc']:.3f}")
    print(f"chance share = 1/{n} = {chance:.4f}")

    for a in ARMS:
        if st[a]["spread"] > OPS_SPREAD_MAX:
            print(f"CONTROL 2 FIRED: {a} ops_spread {st[a]['spread']:.4f} > {OPS_SPREAD_MAX}")
            print("VERDICT: REPORT NOTHING")
            return 0
        if st[a]["maps"] != 1:
            print(f"CONTROL 3 FIRED: {a} shows {st[a]['maps']} lane->cpu maps; placement moved")
            print("VERDICT: REPORT NOTHING")
            return 0

    ratio = st["nothp"]["skew"] / max(1e-9, st["thp"]["skew"])
    print(f"\nS(nothp)/S(thp) = {ratio:.3f}   "
          f"(ACCEPT <= {ACCEPT_RATIO}, REJECT >= {REJECT_RATIO})")
    print(f"wall cost of disabling THP: {st['nothp']['wall'] / st['thp']['wall']:.3f}x")
    if ratio <= ACCEPT_RATIO:
        print("VERDICT: ACCEPT -- removing 2 MiB physical backing removes most of the")
        print("         imbalance. The width-16 straggler is selected by physical page")
        print("         backing, which is why it survived fixed placement and fixed")
        print("         virtual layout.")
    elif ratio >= REJECT_RATIO:
        print("VERDICT: REJECT -- the imbalance survives 4 KiB backing essentially intact.")
        print("         Physical page backing is not the selector. Candidate list is empty")
        print("         again, and the honest record is that the straggler is real,")
        print("         costly and unexplained.")
    else:
        print(f"VERDICT: REPORT NOTHING -- ratio {ratio:.3f} landed between the bounds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
