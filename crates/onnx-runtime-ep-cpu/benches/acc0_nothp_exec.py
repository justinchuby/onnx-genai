#!/usr/bin/env python3
"""exec a command with transparent huge pages disabled for the whole process.

`prctl(PR_SET_THP_DISABLE)` is unprivileged and is inherited across `exec` and
by every thread, which makes it the only lever on this host that changes the
*physical* backing of a benchmark's memory: the sysfs THP control
(`/sys/kernel/mm/transparent_hugepage/enabled`) is root-only here and
`/proc/self/pagemap` returns PFN 0 without `CAP_SYS_ADMIN`, so physical frames
cannot be read directly either.

Used by `acc0_w16_straggler_thp.py`, which verifies the lever actually takes
effect before it measures anything -- `AnonHugePages` in `smaps_rollup` must be
non-zero without this wrapper and zero with it. That check is not ceremony: a
first attempt to verify it appeared to show the wrapper doing nothing, and the
real reason was that the test mapping was not 2 MiB aligned so THP had never
backed it in either arm. An unverified lever would have produced two identical
arms and a free, confident REJECT.
"""

import ctypes
import os
import sys

PR_SET_THP_DISABLE = 41


def main() -> int:
    if len(sys.argv) < 2:
        return sys.exit("usage: acc0_nothp_exec.py <command> [args...]")
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
        return sys.exit(
            "prctl(PR_SET_THP_DISABLE) failed: " + os.strerror(ctypes.get_errno())
        )
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
