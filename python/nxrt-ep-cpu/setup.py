"""Build backend glue for the ``nxrt-ep-cpu`` wheel.

This package ships the compiled ORT plugin-EP cdylib produced by the
``onnx-runtime-ep-cpu-plugin`` crate (``libonnx_runtime_ep_cpu_plugin.so`` /
``.dll`` / ``.dylib``). The cdylib exports the ORT plugin-EP C ABI
(``CreateEpFactories`` / ``ReleaseEpFactory``); it is **not** a PyO3 Python
extension. We therefore build it with plain ``cargo`` and vendor the resulting
shared library into the wheel as package data, rather than going through
maturin. See ``.squad/decisions/inbox/sebastian-nxrt-ep-pypi.md`` for the
rationale (dual-symbol linker problem avoided by keeping the cdylib pure C ABI
and providing a thin pure-Python ``__init__``).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.dist import Distribution

# ── Crate / package wiring ──────────────────────────────────────────────────
CRATE = "onnx-runtime-ep-cpu-plugin"
LIB_STEM = "onnx_runtime_ep_cpu_plugin"
PACKAGE = "nxrt_ep_cpu"
# Extra cargo features to enable (none for CPU; the CUDA package sets "cuda").
CARGO_FEATURES: list[str] = []


def _lib_filenames() -> list[str]:
    """Candidate cdylib filenames for the current platform.

    Rust emits ``lib<stem>.so`` on Linux, ``lib<stem>.dylib`` on macOS, and
    ``<stem>.dll`` (no ``lib`` prefix) on Windows.
    """
    if sys.platform.startswith("win"):
        return [f"{LIB_STEM}.dll"]
    if sys.platform == "darwin":
        return [f"lib{LIB_STEM}.dylib"]
    return [f"lib{LIB_STEM}.so"]


def _workspace_root(start: Path) -> Path:
    """Walk up from ``start`` to the Cargo workspace root.

    cibuildwheel copies the whole repository into the build container, so the
    workspace manifest is reachable by walking up from this file.
    """
    start = start.resolve()
    for candidate in (start, *start.parents):
        manifest = candidate / "Cargo.toml"
        if manifest.exists() and "[workspace]" in manifest.read_text(encoding="utf-8"):
            return candidate
    raise RuntimeError(
        f"could not locate the Cargo workspace root by walking up from {start}"
    )


def _build_cdylib() -> Path:
    """Compile the plugin crate with cargo and return the built cdylib path."""
    root = _workspace_root(Path(__file__).parent)
    cmd = ["cargo", "build", "--release", "-p", CRATE]
    if CARGO_FEATURES:
        cmd += ["--features", ",".join(CARGO_FEATURES)]
    print(f"[nxrt-ep] running: {' '.join(cmd)} (cwd={root})", flush=True)
    subprocess.run(cmd, cwd=root, check=True)

    release_dir = root / "target" / "release"
    for name in _lib_filenames():
        candidate = release_dir / name
        if candidate.exists():
            return candidate
    raise RuntimeError(
        f"cargo build succeeded but no cdylib found in {release_dir} "
        f"(looked for {_lib_filenames()})"
    )


class build_py(_build_py):
    """Build the cdylib and drop it next to the Python package."""

    def run(self) -> None:
        super().run()
        lib = _build_cdylib()
        dest_dir = Path(self.build_lib) / PACKAGE
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / lib.name
        print(f"[nxrt-ep] bundling {lib} -> {dest}", flush=True)
        shutil.copy2(lib, dest)


class BinaryDistribution(Distribution):
    """Force a platform-specific (non-purelib) wheel: we ship a compiled .so."""

    def has_ext_modules(self) -> bool:  # noqa: D401 - setuptools hook
        return True


# ``bdist_wheel`` lives in setuptools (recent) or the standalone ``wheel`` pkg.
try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # pragma: no cover - older setuptools
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class bdist_wheel(_bdist_wheel):
    """Tag the wheel ``py3-none-<platform>``.

    The bundled cdylib is a plain C-ABI shared library with no CPython ABI
    dependency, so a single platform wheel works on any Python 3. auditwheel
    (Linux) / delvewheel (Windows) later retag it manylinux/win-compliant.
    """

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _python, _abi, plat = super().get_tag()
        return "py3", "none", plat


setup(
    cmdclass={"build_py": build_py, "bdist_wheel": bdist_wheel},
    distclass=BinaryDistribution,
)
