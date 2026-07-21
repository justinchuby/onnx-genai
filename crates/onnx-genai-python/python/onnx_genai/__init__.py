"""onnx_genai — ONNX Runtime-backed text generation.

Same API as ``nxrt.genai`` (``Engine`` / ``GenerateResult``), implemented on
ONNX Runtime so it stays ONNX Runtime compatible.

The ONNX Runtime shared library is intentionally **not** bundled in this wheel.
It is loaded at import time from the installed ``onnxruntime`` (CPU) or
``onnxruntime-gpu`` (CUDA) wheel, so ``onnx_genai`` automatically uses whichever
execution providers the user installed. Install the CUDA runtime with
``pip install onnx-genai[cuda]`` on Windows/Linux; on macOS the
``onnxruntime-ep-mlx`` plugin is pulled in by default for Apple Silicon.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from pathlib import Path

# Environment variable through which the native engine can locate the MLX
# execution-provider plugin (macOS). Spelled out in full for readability.
_MLX_EP_LIBRARY_ENVIRONMENT_VARIABLE = "ONNX_GENAI_MLX_EP_LIBRARY"


def _onnxruntime_capi_directory() -> Path:
    """Return the ``capi`` directory of the installed onnxruntime package.

    Raises a clear, actionable error if the ``onnxruntime`` (or
    ``onnxruntime-gpu``) wheel is not installed, since it supplies the ONNX
    Runtime shared library this extension links against.
    """
    try:
        import onnxruntime  # noqa: F401  (imported for its filesystem location)
    except ImportError as error:  # pragma: no cover - exercised via install docs
        raise ImportError(
            "onnx_genai requires the ONNX Runtime shared library, which is "
            "provided by the 'onnxruntime' package. Install it with "
            "'pip install onnxruntime' (CPU) or 'pip install onnx-genai[cuda]' "
            "(CUDA, Windows/Linux)."
        ) from error

    capi_directory = Path(onnxruntime.__file__).resolve().parent / "capi"
    if not capi_directory.is_dir():
        raise ImportError(
            f"Could not locate the onnxruntime native library directory at "
            f"{capi_directory}. The installed onnxruntime package appears to be "
            f"incomplete."
        )
    return capi_directory


def _preload_onnxruntime() -> None:
    """Load libonnxruntime from the onnxruntime package before the extension.

    Our compiled ``_onnx_genai`` module links libonnxruntime dynamically. By
    loading the exact shared library shipped in the onnxruntime wheel first (and,
    on Windows, adding its directory to the DLL search path), the operating
    system resolves the extension's dependency to that already-loaded library.
    """
    capi_directory = _onnxruntime_capi_directory()

    if sys.platform.startswith("win"):
        # Add the capi directory to the DLL search path, then load the DLL so
        # the extension's import-by-name (onnxruntime.dll) resolves to it.
        os.add_dll_directory(str(capi_directory))
        library = capi_directory / "onnxruntime.dll"
        if library.is_file():
            ctypes.CDLL(str(library))
        return

    if sys.platform == "darwin":
        patterns = ["libonnxruntime.*.dylib", "libonnxruntime.dylib"]
        flags = os.RTLD_GLOBAL | os.RTLD_NOW
    else:
        patterns = ["libonnxruntime.so.*", "libonnxruntime.so"]
        flags = os.RTLD_GLOBAL | os.RTLD_NOW

    for pattern in patterns:
        matches = sorted(glob.glob(str(capi_directory / pattern)))
        if matches:
            # Load with RTLD_GLOBAL so the extension's DT_NEEDED (tracked by the
            # library's SONAME / install name) is satisfied by this object.
            ctypes.CDLL(matches[-1], mode=flags)
            return

    raise ImportError(
        f"Could not find the ONNX Runtime shared library under {capi_directory}."
    )


def _configure_mlx_execution_provider() -> None:
    """Expose the MLX execution-provider plugin path to the native engine (macOS).

    ``onnxruntime-ep-mlx`` ships the plugin dylib bundled in the wheel. We record
    its absolute path in an environment variable so the engine can register it
    with ONNX Runtime; failures here are non-fatal (MLX simply stays disabled).
    """
    if sys.platform != "darwin":
        return
    if _MLX_EP_LIBRARY_ENVIRONMENT_VARIABLE in os.environ:
        return
    try:
        import onnxruntime_ep_mlx

        library_path = onnxruntime_ep_mlx.library_path()
    except Exception:  # pragma: no cover - MLX plugin is optional at runtime
        return
    if library_path and os.path.isfile(library_path):
        os.environ[_MLX_EP_LIBRARY_ENVIRONMENT_VARIABLE] = str(library_path)


_preload_onnxruntime()
_configure_mlx_execution_provider()

from ._onnx_genai import Engine, GenerateResult, __version__  # noqa: E402

__all__ = ["Engine", "GenerateResult", "__version__"]
