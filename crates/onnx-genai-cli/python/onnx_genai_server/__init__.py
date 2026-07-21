"""onnx_genai_server — the ``onnx-genai`` command-line tool and OpenAI-compatible server.

Installing the ``onnx-genai-server`` wheel provides a single ``onnx-genai``
console command (``serve``, ``generate``, ``run``, ``show``, ``list``,
``version``). The command runs in-process through a compiled extension module;
this launcher first loads the ONNX Runtime shared library from the installed
``onnxruntime`` / ``onnxruntime-gpu`` wheel so ONNX Runtime is **not** bundled —
the tool uses whichever execution providers the user installed. Install the CUDA
runtime with ``pip install onnx-genai-server[cuda]`` on Windows/Linux; on macOS
the ``onnxruntime-ep-mlx`` plugin is pulled in by default for Apple Silicon.
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
    """Return the ``capi`` directory of the installed onnxruntime package."""
    try:
        import onnxruntime  # noqa: F401  (imported for its filesystem location)
    except ImportError as error:  # pragma: no cover - exercised via install docs
        raise ImportError(
            "onnx-genai-server requires the ONNX Runtime shared library, which "
            "is provided by the 'onnxruntime' package. Install it with "
            "'pip install onnxruntime' (CPU) or "
            "'pip install onnx-genai-server[cuda]' (CUDA, Windows/Linux)."
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
    """Load libonnxruntime from the onnxruntime package before the extension."""
    capi_directory = _onnxruntime_capi_directory()

    if sys.platform.startswith("win"):
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
            ctypes.CDLL(matches[-1], mode=flags)
            return

    raise ImportError(
        f"Could not find the ONNX Runtime shared library under {capi_directory}."
    )


def _configure_mlx_execution_provider() -> None:
    """Expose the MLX execution-provider plugin path to the engine (macOS)."""
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


def main() -> int:
    """Console entry point for the ``onnx-genai`` command.

    Loads ONNX Runtime from the installed onnxruntime wheel, then dispatches to
    the compiled CLI. Returns the process exit code.
    """
    _preload_onnxruntime()
    _configure_mlx_execution_provider()
    from ._onnx_genai_server import _run_cli

    return _run_cli(sys.argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
