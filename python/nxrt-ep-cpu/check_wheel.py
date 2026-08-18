"""Smoke test for a built ``nxrt-ep-cpu`` wheel, run by cibuildwheel.

Asserts two things about the *installed* wheel: the cdylib is present, and it
is the build this platform asked cargo for. The second half matters because a
fallback (pure-Rust) build is invisible at runtime and costs an order of
magnitude on the quantized matmul operators — `setup.py` records the intent in
``nxrt_ep_cpu._build``, and the cdylib reports the truth through
``nxrt_ep_build_features``.
"""

from __future__ import annotations

import os

import nxrt_ep_cpu
from nxrt_ep_cpu import _build

path = nxrt_ep_cpu.get_library_path()
assert os.path.exists(path), path

reported = nxrt_ep_cpu.build_features()
assert reported == _build.EXPECTED_FEATURES, (
    f"bundled cdylib reports build features {reported!r}, but this wheel was "
    f"built asking for {_build.EXPECTED_FEATURES!r}"
)
print("OK", path, f"features={reported!r}")
