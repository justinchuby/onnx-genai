#!/usr/bin/env python3
"""Work-completed probe: how much *work* a CPU delivers in a fixed CPU-second.

Sebastian reported (2026-08-24) a permanent external competitor on cpu 0 taking
50.3% of the core, with cpu 1 as SMT collateral -- 100% CPU *share* but 55% of
the *work*. That second case is the one that matters: a CPU-time instrument
(`/usr/bin/time -v`'s `Percent of CPU`, `getrusage`) cannot see it, because the
scheduler really is giving the thread the CPU; the core is contended in
hardware, below the scheduler's view.

So the instrument has to be work completed, not time granted. This counts
iterations of a fixed integer loop against CLOCK_THREAD_CPUTIME_ID and reports
both, plus involuntary context switches. Two CPUs on the same physical core
delivering very different iteration counts at the same cpu_share is the SMT
signature; a low cpu_share on its own is ordinary timesharing.

This matters to me because every acc0 row I have published is pinned to a CPU
set that begins at cpu 0.
"""
import os
import resource
import sys
import time

SECONDS = float(os.environ.get("PROBE_SECONDS", "2.0"))


def spin(seconds):
    t0 = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
    w0 = time.monotonic()
    iters = 0
    x = 0
    while True:
        for _ in range(2000):
            x = (x * 1103515245 + 12345) & 0xFFFFFFFF
        iters += 1
        if time.monotonic() - w0 >= seconds:
            break
    cpu = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID) - t0
    wall = time.monotonic() - w0
    return iters, cpu, wall, x


if __name__ == "__main__":
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    iters, cpu, wall, _ = spin(SECONDS)
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    print(f"cpu={sys.argv[1] if len(sys.argv) > 1 else '?':>3} "
          f"iters={iters:6d} cpu_share={cpu / wall:.3f} "
          f"ivcsw={r1.ru_nivcsw - r0.ru_nivcsw}")
