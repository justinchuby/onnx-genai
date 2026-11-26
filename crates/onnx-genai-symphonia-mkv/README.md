# onnx-genai-symphonia-mkv

A maintained MPL-2.0 Matroska/WebM reader derived from Symphonia 0.6.1. It uses
unchanged published `symphonia-core` and `symphonia-common` 0.6.1 types. This is
an independent GenAI-stack package, not an upstream Symphonia release.

The normal core `FormatReader::default_track(TrackType::Audio)` selection is
preserved. Only that default audio track, when Opus, is admitted and normalized:
48 kHz positioned channels, `Track.delay` descriptive, raw packet PTS and exact
packet trims. Consumers must decode raw PCM and apply those trims once. Approximate
Segment.Duration is not an exact PCM endpoint; normalized track duration, frame
count and aggregate padding remain unknown. Other tracks retain upstream behavior;
nondefault Opus tracks are not covered by the normalized presentation contract.

The initial profile is complete full-file packet concatenation with zero coded
origin, exact 48 kHz
conversion of declared timestamp/delay/padding units, unit TrackTimestampScale,
explicit CodecDelay matching OpusHead pre-skip, known consistent recording rate,
one packet per block, no CodecState changes and no seeking. Actual container PTS
minus delay is preserved independently of the raw PCM cursor. Quantized or gapped
PTS does not change sample cuts or inject silence. BlockDuration/DefaultDuration
are retained in debug diagnostics as presentation facts, never used as decoder
geometry. TOC and exact padding determine valid PCM. Signed prefix and terminal
suffix padding compose with priming as intersected raw-buffer ranges. Fractional
conversions, backwards timestamps, lacing or reconfiguration fail explicitly.

`opus` exposes the shared RFC packet geometry and bounded header/channel admission;
its generic 1–48 frame helper does not inherit an unrelated Ogg demuxer limitation.
The parser and decoder private heaps are not metered by a PCM payload budget.

## Source and license

Original copyright and MPL notices are preserved in the source and LICENSE.
`UPSTREAM.json` records the immutable upstream archive, revision and copied-file
hashes; the exact 50 KiB source archive is included under `upstream/`. Run
`python3 verify_upstream.py` to verify the archive and identify local modifications.
`CHANGES.md` describes this maintained variant. Corresponding modified source is
in this package and at the repository URL in Cargo.toml. Distributors must preserve
MPL notices and provide the covered source as required by MPL-2.0. No external
ffmpeg executable or C codec runtime is required by this reader.
