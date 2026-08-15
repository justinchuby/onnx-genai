"""nxrt-ep-cuda — bundled ONNX Runtime plugin-EP (CUDA) shared library.

EXPERIMENTAL / PRE-RELEASE: the CUDA execution provider has been validated on
physical CUDA hardware (NVIDIA H200, CUDA 13.0) — it runs the Muse-Glimmer-30B
(int4) decoder end-to-end with zero CPU fallbacks, and the bundled cdylib
registers and executes ONNX graphs through ONNX Runtime on-device — but APIs and
packaging may still change without notice. This package vendors the compiled
``onnx-runtime-ep-cuda-plugin`` cdylib (built with the ``cuda`` cargo feature,
CUDA 13) and exposes the absolute path to it, plus a thin helper to register it
with an installed ``onnxruntime`` via ``register_execution_provider_library``.

The cdylib exports the ORT plugin-EP C ABI (``CreateEpFactories`` /
``ReleaseEpFactory``); it is consumed by ONNX Runtime through
``RegisterExecutionProviderLibrary(registration_name, library_path)``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["get_library_path", "register", "REGISTRATION_NAME", "__version__"]

__version__ = "0.1.0.dev5"

#: Default registration name passed to ONNX Runtime.
REGISTRATION_NAME = "nxrt_ep_cuda"

_LIB_STEM = "onnx_runtime_ep_cuda_plugin"
_LIB_NAMES = (
    f"lib{_LIB_STEM}.so",
    f"lib{_LIB_STEM}.dylib",
    f"{_LIB_STEM}.dll",
)


def get_library_path() -> str:
    """Return the absolute path to the bundled CUDA plugin-EP cdylib.

    Raises:
        FileNotFoundError: if the shared library was not packaged.
    """
    here = Path(__file__).resolve().parent
    for name in _LIB_NAMES:
        candidate = here / name
        if candidate.exists():
            return os.fspath(candidate)
    raise FileNotFoundError(
        f"nxrt_ep_cuda: bundled plugin library not found in {here}. "
        f"Looked for: {', '.join(_LIB_NAMES)}. This usually means the wheel "
        "was built without the compiled cdylib."
    )


def register(session_options=None, registration_name: str = REGISTRATION_NAME):
    """Register the bundled CUDA plugin EP library with ONNX Runtime.

    Thin wrapper around ONNX Runtime's ``register_execution_provider_library``.
    Prefers ``SessionOptions.register_execution_provider_library(name, path)``
    when a ``session_options`` is supplied, else the module-level
    ``onnxruntime.register_execution_provider_library``.

    Args:
        session_options: an ``onnxruntime.SessionOptions`` instance.
        registration_name: the name ORT associates with the library.

    Returns:
        The path that was registered.

    Raises:
        ImportError: if ``onnxruntime`` cannot be imported.
        RuntimeError: if no compatible entry point is available.
    """
    path = get_library_path()

    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "nxrt_ep_cuda.register requires the 'onnxruntime' package to be "
            "installed."
        ) from exc

    if session_options is not None and hasattr(
        session_options, "register_execution_provider_library"
    ):
        session_options.register_execution_provider_library(registration_name, path)
        return path

    module_fn = getattr(ort, "register_execution_provider_library", None)
    if callable(module_fn):
        module_fn(registration_name, path)
        return path

    raise RuntimeError(
        "The installed onnxruntime does not expose a "
        "'register_execution_provider_library' entry point compatible with "
        "nxrt_ep_cuda.register(). Pass a SessionOptions that supports it, or "
        "register the library manually with get_library_path()."
    )
