"""Installed Native binding acceptance; run in a fresh dependency-free environment."""

import importlib.metadata
import importlib.util
import os
import sys
import tempfile
from pathlib import Path


def main() -> None:
    fixture = Path(sys.argv[1]).resolve()
    for package in ("onnxruntime", "onnxruntime_ep_mlx"):
        if importlib.util.find_spec(package) is not None:
            raise RuntimeError(
                f"wheel acceptance requires no installed {package}; use a fresh venv"
            )
    distribution = importlib.metadata.distribution("onnx-genai")
    specification = importlib.util.find_spec("onnx_genai")
    installed = distribution.locate_file("onnx_genai/__init__.py").resolve()
    if (
        specification is None
        or specification.origin is None
        or Path(specification.origin).resolve() != installed
    ):
        raise RuntimeError(
            "binding acceptance imported a source checkout instead of the installed wheel"
        )
    requirements = distribution.requires or []
    if any("extra ==" not in requirement for requirement in requirements):
        raise RuntimeError(
            f"Native wheel declares a mandatory Python dependency: {requirements}"
        )
    os.environ.pop("ONNX_GENAI_BACKEND", None)
    os.environ["ONNX_GENAI_EP"] = "cpu"
    with tempfile.TemporaryDirectory(prefix="nxrt-binding-wheel-") as temporary:
        os.environ["ONNX_GENAI_ORT_LIB"] = str(
            Path(temporary) / "missing-reference-runtime"
        )
        import onnx_genai

        engine = onnx_genai.Engine.from_dir(fixture)
        result = engine.generate("hi", max_tokens=2, temperature=0)
        if len(result.token_ids) != 2 or not isinstance(result.text, str):
            raise RuntimeError(
                f"installed binding did not generate two actual tokens: {result}"
            )
        if any(name in sys.modules for name in ("onnxruntime", "onnxruntime_ep_mlx")):
            raise RuntimeError("Native binding imported a reference runtime package")
    print("Installed binding: import and Native generation passed without ORT or MLX.")


if __name__ == "__main__":
    main()
