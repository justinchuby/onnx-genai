# Changes from symphonia-format-mkv 0.6.1

- Renamed the package to `onnx-genai-symphonia-mkv` using the workspace version;
  unchanged published core/common 0.6.1 remain the sole public type identity.
- Preserved CodecDelay presence, signed DiscardPadding and CodecState presence.
- After unchanged canonical default-audio selection, normalize only selected Opus
  headers and packet timestamps/trims to 48 kHz, with independent raw PCM cursor and full-file profile guards.
- Preserve all other tracks and codec/lacing paths; no nondefault Opus support claim.
- Preserve actual container PTS independently of packet-concatenated PCM; presentation BlockDuration/DefaultDuration never choose sample cuts.
- Clear approximate normalized-track endpoint fields; raw packet trims own cuts.
- Added one shared public RFC packet/header helper, authored reader/clock tests,
  real stream-copy WebM fixture and upstream provenance verification.
- Reject Opus seeking; unchanged legacy codecs retain their existing seek behavior.
