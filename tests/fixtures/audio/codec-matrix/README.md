# Original audio codec fixtures

`generate.py` creates a 100 ms, 16 kHz stereo sine tone and encodes it with
ffmpeg. All audio is generated here; no third-party recordings are used.
Run `python3 tests/fixtures/audio/codec-matrix/generate.py` with ffmpeg and
ffprobe installed. Tests use committed artifacts and need neither executable.
The manifest records encoded SHA-256, reference sample counts and tolerances.

Independent references convert each channel to float before taking its arithmetic
mean. Lossless formats use 1e-7 tolerance. Lossy codecs use 2e-4; IMA ADPCM uses
0.001 because FFmpeg and Symphonia round intermediate predictor steps differently.
Vorbis references end at the declared 1600-sample EOS granule.
MP4 references disable edit lists: Symphonia 0.6.1 does not apply them, so AAC
priming remains in that media stream. Packet durations still bound trailing data.
The Opus fixture pins an actionable unsupported-codec error.
