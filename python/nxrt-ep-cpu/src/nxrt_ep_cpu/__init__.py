"""nxrt-ep-cpu — bundled ONNX Runtime plugin-EP (CPU) shared library.

This package vendors the compiled ``onnx-runtime-ep-cpu-plugin`` cdylib and
exposes the absolute path to it, plus a thin helper to register it with an
installed ``onnxruntime`` via ``register_execution_provider_library``.

The cdylib exports the ORT plugin-EP C ABI (``CreateEpFactories`` /
``ReleaseEpFactory``); it is consumed by ONNX Runtime through
``RegisterExecutionProviderLibrary(registration_name, library_path)``.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

__all__ = [
    "build_features",
    "get_library_path",
    "register",
    "REGISTRATION_NAME",
    "__version__",
]

__version__ = "0.1.0.dev5"

#: Default registration name passed to ONNX Runtime.
REGISTRATION_NAME = "nxrt_ep_cpu"

# Candidate filenames for the bundled cdylib across platforms. Rust emits
# ``lib<stem>.so`` (Linux), ``lib<stem>.dylib`` (macOS) and ``<stem>.dll``
# (Windows, no ``lib`` prefix).
_LIB_STEM = "onnx_runtime_ep_cpu_plugin"
_LIB_NAMES = (
    f"lib{_LIB_STEM}.so",
    f"lib{_LIB_STEM}.dylib",
    f"{_LIB_STEM}.dll",
)


def get_library_path() -> str:
    """Return the absolute path to the bundled plugin-EP cdylib.

    Raises:
        FileNotFoundError: if the shared library was not packaged (e.g. a
            broken/pure-python install).
    """
    here = Path(__file__).resolve().parent
    for name in _LIB_NAMES:
        candidate = here / name
        if candidate.exists():
            return os.fspath(candidate)
    raise FileNotFoundError(
        f"nxrt_ep_cpu: bundled plugin library not found in {here}. "
        f"Looked for: {', '.join(_LIB_NAMES)}. This usually means the wheel "
        "was built without the compiled cdylib."
    )


def build_features() -> str:
    """Return the optional build features compiled into the bundled cdylib.

    ``"mlas"`` means the vendored ONNX Runtime MLAS kernels are linked in;
    an empty string means the pure-Rust fallback paths, which are an order of
    magnitude slower on the quantized matmul operators. A compiled library
    otherwise says nothing about how it was built, so this is the only way to
    tell an installed wheel apart from a fallback build.

    Raises:
        FileNotFoundError: if the shared library was not packaged.
        OSError: if the library cannot be loaded.
        AttributeError: if the bundled library predates this export.
    """
    handle = ctypes.CDLL(get_library_path())
    try:
        entry = handle.nxrt_ep_build_features
        entry.restype = ctypes.c_char_p
        return (entry() or b"").decode()
    finally:
        # This query must not leave the library loaded: on Windows that would
        # lock the file for the rest of the process.
        try:
            if sys.platform == "win32":
                ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(handle._handle))
            else:
                ctypes.CDLL(None).dlclose(ctypes.c_void_p(handle._handle))
        except Exception:  # pragma: no cover - the value was already read
            pass


def register(session_options=None, registration_name: str = REGISTRATION_NAME):
    """Register the bundled plugin EP library with ONNX Runtime.

    This is a thin convenience wrapper around ONNX Runtime's
    ``register_execution_provider_library`` API. The exact call surface differs
    between ONNX Runtime releases, so this inspects what is available:

    * ``SessionOptions.register_execution_provider_library(name, path)``
      (preferred, when ``session_options`` is provided), or
    * ``onnxruntime.register_execution_provider_library(name, path)`` (module
      level), when present.

    Args:
        session_options: an ``onnxruntime.SessionOptions`` instance. If given
            and it exposes ``register_execution_provider_library``, that method
            is used.
        registration_name: the name ORT associates with the library.

    Returns:
        The path that was registered.

    Raises:
        ImportError: if ``onnxruntime`` cannot be imported.
        RuntimeError: if no compatible ``register_execution_provider_library``
            entry point is available in the installed ONNX Runtime.
    """
    path = get_library_path()

    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "nxrt_ep_cpu.register requires the 'onnxruntime' package to be "
            "installed."
        ) from exc

    # Prefer the SessionOptions-scoped API when a session_options is supplied.
    if session_options is not None and hasattr(
        session_options, "register_execution_provider_library"
    ):
        session_options.register_execution_provider_library(registration_name, path)
        return path

    # Fall back to the module-level API.
    module_fn = getattr(ort, "register_execution_provider_library", None)
    if callable(module_fn):
        module_fn(registration_name, path)
        return path

    raise RuntimeError(
        "The installed onnxruntime does not expose a "
        "'register_execution_provider_library' entry point compatible with "
        "nxrt_ep_cpu.register(). Pass a SessionOptions that supports it, or "
        "register the library manually with get_library_path()."
    )
