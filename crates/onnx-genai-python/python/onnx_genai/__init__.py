"""Native-default Python engine with inert optional reference-library discovery.

The engine selects Native by default. Python locates installed optional libraries
without importing their packages or loading any shared library itself.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# Windows loader search handles must outlive the compiled CLI invocation.
_DLL_DIRECTORY_HANDLES: dict[str, object] = {}


def _package_directory(name: str) -> Path | None:
    """Locate a top-level package without executing its initialization code."""
    specification = importlib.util.find_spec(name)
    if specification is None or specification.origin is None:
        return None
    return Path(specification.origin).resolve().parent


def _configure_optional_libraries() -> None:
    """Publish discovery hints; the selected Rust backend loads its libraries."""
    if not any(
        name in os.environ for name in ("ONNX_GENAI_ORT_LIB", "ONNX_GENAI_ORT_LIB_DIR")
    ):
        package = _package_directory("onnxruntime")
        if package is not None:
            capi = package / "capi"
            if capi.is_dir():
                os.environ["ONNX_GENAI_ORT_LIB_DIR"] = str(capi)
    if sys.platform.startswith("win"):
        library = os.environ.get("ONNX_GENAI_ORT_LIB")
        directory = (
            str(Path(library).resolve().parent)
            if library
            else os.environ.get("ONNX_GENAI_ORT_LIB_DIR")
        )
        if (
            directory
            and Path(directory).is_dir()
            and directory not in _DLL_DIRECTORY_HANDLES
        ):
            _DLL_DIRECTORY_HANDLES[directory] = os.add_dll_directory(directory)
    if sys.platform == "darwin" and "ONNX_GENAI_MLX_EP_LIBRARY" not in os.environ:
        package = _package_directory("onnxruntime_ep_mlx")
        if package is not None:
            library = package / "libonnxruntime_mlx_ep.dylib"
            if library.is_file():
                os.environ["ONNX_GENAI_MLX_EP_LIBRARY"] = str(library)


# Discovery records optional reference-library hints without importing ORT or MLX.
_configure_optional_libraries()

from ._onnx_genai import Engine, GenerateResult, __version__

__all__ = ["Engine", "GenerateResult", "__version__"]
