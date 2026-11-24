"""Exercise the installed wheel with an unavailable ORT reference library."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    fixture = Path(sys.argv[1]).resolve()
    environment = os.environ.copy()
    environment.pop("ONNX_GENAI_BACKEND", None)
    environment["ONNX_GENAI_EP"] = "cpu"
    with tempfile.TemporaryDirectory(prefix="nxrt-wheel-smoke-") as temporary:
        profile = Path(temporary) / "profile.json"
        environment["ONNX_GENAI_ORT_LIB"] = str(
            Path(temporary) / "missing-reference-runtime"
        )

        def run(
            arguments: list[str], success: bool = True
        ) -> subprocess.CompletedProcess:
            result = subprocess.run(
                ["onnx-genai", *arguments],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            if (result.returncode == 0) != success:
                raise RuntimeError(
                    f"installed CLI returned {result.returncode} for {arguments}:\n"
                    f"{result.stdout}\n{result.stderr}"
                )
            return result

        run(["version"])
        run(
            [
                "--profile-json",
                str(profile),
                "generate",
                str(fixture),
                "hi",
                "--max-new-tokens",
                "2",
                "--greedy",
                "--no-stats",
                "--cpu-cores",
                "1",
            ]
        )
        actual = json.loads(profile.read_text())
        if (
            actual["decode_backend"] != "native"
            or actual["execution_provider"] != "native"
        ):
            raise RuntimeError(f"installed wheel did not execute Native: {actual}")
        rejected = run(
            [
                "generate",
                str(fixture),
                "hi",
                "--backend",
                "ort",
                "--max-new-tokens",
                "1",
                "--no-stats",
                "--cpu-cores",
                "1",
            ],
            success=False,
        )
        if "failed to load onnx runtime" not in rejected.stderr.lower():
            raise RuntimeError(
                f"explicit ORT failed for an unexpected reason: {rejected.stderr}"
            )
    print(
        "Installed wheel: version and Native generation passed; missing explicit ORT rejected."
    )


if __name__ == "__main__":
    main()
