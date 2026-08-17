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

import ctypes
import os
import platform
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
# Extra cargo features to enable. ``mlas`` links the vendored ONNX Runtime MLAS
# kernels into the cdylib; see ``_mlas_features``.
MLAS_FEATURE = "mlas"

# (``platform.system()``, ``platform.machine()``) pairs whose MLAS build this
# repository's CI proves, lowercased. Every entry is compiled by a
# ``cargo build -p onnx-runtime-ep-cpu-plugin --features mlas``
# step on the matching lane in ``.github/workflows/ci.yml``; a target that is
# not proven there is not listed here, because a wheel that fails to build is
# worse than a wheel that is slow.
MLAS_TARGETS = frozenset(
    {
        ("linux", "x86_64"),
        ("windows", "amd64"),
        ("windows", "arm64"),
        ("darwin", "arm64"),
    }
)


def _mlas_features(system: str | None = None, machine: str | None = None) -> list[str]:
    """Cargo features for this wheel's target.

    ORT's own CPU execution provider *is* MLAS. A cdylib built without it does
    not lose a little speed, it loses an order of magnitude: on this project's
    plugin-path A/B (AMD EPYC 9V74, AVX2, ORT 1.27, K=N=2048, p50 of 41
    interleaved iterations), 4-bit ``MatMulNBits`` at M=128 takes 81x ORT's
    time without MLAS and 7.3x with it, and ``QLinearMatMul`` u8 at M=128
    takes 55x without and 9.3x with. Shipping the pure-Rust build is therefore
    not the conservative default; it is the slow one.

    Set ``NXRT_EP_CPU_NO_MLAS=1`` to build the pure-Rust cdylib anyway, for a
    toolchain with no C++ compiler or a target the vendored sources do not
    cover.
    """
    if os.environ.get("NXRT_EP_CPU_NO_MLAS") == "1":
        print("[nxrt-ep] NXRT_EP_CPU_NO_MLAS=1: building without MLAS", flush=True)
        return []
    target = (
        (system or platform.system()).lower(),
        (machine or platform.machine()).lower(),
    )
    if target not in MLAS_TARGETS:
        print(
            f"[nxrt-ep] no CI-proven MLAS build for {target}: building without MLAS",
            flush=True,
        )
        return []
    return [MLAS_FEATURE]


CARGO_FEATURES: list[str] = _mlas_features()


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
            _verify_features(candidate)
            return candidate
    raise RuntimeError(
        f"cargo build succeeded but no cdylib found in {release_dir} "
        f"(looked for {_lib_filenames()})"
    )


def _unload(handle: ctypes.CDLL) -> None:
    """Best-effort release of a library loaded only to read its identity."""
    try:
        if sys.platform == "win32":
            ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(handle._handle))
        else:
            ctypes.CDLL(None).dlclose(ctypes.c_void_p(handle._handle))
    except Exception:  # pragma: no cover - identity check already succeeded
        pass


def _verify_features(lib: Path) -> None:
    """Fail the build if the cdylib is not the one we asked cargo for.

    ``target/release`` is shared with every other build in the checkout, so the
    file that exists after ``cargo build`` is not necessarily the file that
    build produced -- a stale artifact from a different feature set is exactly
    the failure mode this catches, and it is invisible afterwards because a
    compiled library does not say what it was built from. The cdylib exports
    ``nxrt_ep_build_features`` for this purpose.
    """
    expected = "mlas" if MLAS_FEATURE in CARGO_FEATURES else ""
    handle = ctypes.CDLL(os.fspath(lib))
    try:
        entry = getattr(handle, "nxrt_ep_build_features", None)
        if entry is None:
            raise RuntimeError(
                f"{lib} does not export nxrt_ep_build_features; it was built from "
                "a source tree that predates the build-identity export"
            )
        entry.restype = ctypes.c_char_p
        reported = (entry() or b"").decode()
    finally:
        # Windows keeps a loaded DLL locked against replacement, and this build
        # may go on to rebuild the same target directory.
        _unload(handle)
    if reported != expected:
        raise RuntimeError(
            f"{lib} reports build features {reported!r} but this wheel asked "
            f"for {expected!r}. Refusing to ship a mislabelled cdylib."
        )
    print(f"[nxrt-ep] cdylib build features: {reported!r}", flush=True)


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
        # Record what this wheel *intended* to ship, so the smoke test can
        # compare it against what the cdylib actually reports without needing
        # a build backend in the test environment.
        expected = "mlas" if MLAS_FEATURE in CARGO_FEATURES else ""
        (dest_dir / "_build.py").write_text(
            '"""Generated by setup.py: the build this wheel asked cargo for."""\n\n'
            f'EXPECTED_FEATURES = {expected!r}\n',
            encoding="utf-8",
        )


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
