"""Regenerate original codec fixtures with ffmpeg/ffprobe (not needed by tests)."""

from pathlib import Path
import subprocess, struct, math, json, hashlib

root = Path(__file__).resolve().parent
import tempfile

workspace = tempfile.TemporaryDirectory()
raw = Path(workspace.name) / "source.f32le"
raw.write_bytes(
    b"".join(
        struct.pack(
            "<ff",
            0.4999999 * math.sin(2 * math.pi * 440 * i / 16000),
            0.24681357 * math.sin(2 * math.pi * 440 * i / 16000),
        )
        for i in range(1600)
    )
)
cases = [
    ("pcm-u8.wav", "pcm_u8", []),
    ("pcm-s16.wav", "pcm_s16le", []),
    ("pcm-s24.wav", "pcm_s24le", []),
    ("pcm-s32.wav", "pcm_s32le", []),
    ("pcm-f32.wav", "pcm_f32le", []),
    ("pcm-f64.wav", "pcm_f64le", []),
    ("ima-adpcm.wav", "adpcm_ima_wav", []),
    ("tone.flac", "flac", []),
    ("vorbis.ogg", "vorbis", ["-strict", "experimental", "-q:a", "5"]),
    ("aac.aac", "aac", ["-b:a", "64k"]),
    ("aac.m4a", "aac", ["-b:a", "64k"]),
    ("alac.m4a", "alac", []),
    ("pcm.aiff", "pcm_s16be", []),
    ("pcm.caf", "pcm_s16le", []),
    ("flac.mka", "flac", []),
    ("flac.oga", "flac", []),
    ("mpeg2.mp2", "mp2", ["-b:a", "64k"]),
]
manifest = []
for name, codec, extra in cases:
    file = root / name
    reference = root / (name + ".f32le")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "f32le",
            "-ar",
            "16000",
            "-ac",
            "2",
            "-i",
            str(raw),
            "-map_metadata",
            "-1",
            "-c:a",
            codec,
            *extra,
            "-y",
            str(file),
        ],
        check=True,
    )
    input_options = ["-ignore_editlist", "1"] if name.endswith(".m4a") else []
    filters = "aformat=sample_fmts=flt,pan=mono|c0=0.5*c0+0.5*c1"
    if name == "vorbis.ogg":
        filters += ",atrim=end_sample=1600"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            *input_options,
            "-i",
            str(file),
            "-af",
            filters,
            "-f",
            "f32le",
            "-y",
            str(reference),
        ],
        check=True,
    )
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "json",
                str(file),
            ]
        )
    )
    manifest.append(
        {
            "file": name,
            "codec": codec,
            "sample_rate": int(info["streams"][0]["sample_rate"]),
            "reference": reference.name,
            "frames": reference.stat().st_size // 4,
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "reference_policy": (
                "media_samples_without_mp4_edits"
                if name.endswith(".m4a")
                else "vorbis_eos_granule_1600"
                if name == "vorbis.ogg"
                else "decoded_stream"
            ),
            "max_error": (
                0.001
                if codec == "adpcm_ima_wav"
                else 0.0002
                if codec in ["vorbis", "aac", "mp2"]
                else 0.0000001
            ),
        }
    )
    print(
        name,
        manifest[-1]["sample_rate"],
        manifest[-1]["frames"],
        file.stat().st_size,
        flush=True,
    )
# Unsupported codec inside an otherwise supported OGG container.
subprocess.run(
    [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ar",
        "16000",
        "-ac",
        "2",
        "-i",
        str(raw),
        "-c:a",
        "libopus",
        "-y",
        str(root / "unsupported-opus.ogg"),
    ],
    check=True,
)
(root / "manifest.json").write_text(
    json.dumps(
        {
            "generator": "nxrt original stereo tone; ffmpeg independent PCM oracle",
            "ffmpeg_version": subprocess.check_output(
                ["ffmpeg", "-version"], text=True
            ).splitlines()[0],
            "cases": manifest,
        },
        indent=2,
    )
    + "\n"
)
