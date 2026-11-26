"""Remux existing Opus packets and compare independent libopus PCM references."""

import hashlib
import json
import struct
import subprocess
from pathlib import Path


def pcm_floats(data: bytes) -> tuple[float, ...]:
    return struct.unpack("<" + "f" * (len(data) // 4), data)


def main() -> None:
    root = Path(__file__).resolve().parent
    cases = json.loads((root / "manifest.json").read_text())["cases"]
    receipt = []

    for case in cases:
        name = case["name"]
        output = root / (name + ".webm")
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(root / (name + ".ogg")),
            "-c:a",
            "copy",
            str(output),
        ]
        subprocess.run(command, check=True)
        pcm = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-c:a",
                "libopus",
                "-i",
                str(output),
                "-f",
                "f32le",
                "-c:a",
                "pcm_f32le",
                "-",
            ],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        reference = (root / (name + ".reference.f32")).read_bytes()
        actual_samples = pcm_floats(pcm)
        reference_samples = pcm_floats(reference)
        assert len(actual_samples) == len(reference_samples), (
            name,
            len(actual_samples),
            len(reference_samples),
        )
        error = max(
            abs(actual - expected)
            for actual, expected in zip(actual_samples, reference_samples)
        )
        assert error <= 1 / 32768, (name, error)
        receipt.append(
            {
                "name": name,
                "webm_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "reference_sha256": hashlib.sha256(reference).hexdigest(),
                "full_pcm_samples": len(actual_samples),
                "max_error": error,
                "command": command,
            }
        )

    toolchain = subprocess.run(
        ["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, text=True
    ).stdout.splitlines()[0]
    manifest = {
        "generator": (
            "actual FFmpeg stream copy of unchanged checked-in Opus packets; "
            "independent libopus PCM oracle"
        ),
        "toolchain": toolchain,
        "cases": receipt,
    }
    (root / "webm-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
