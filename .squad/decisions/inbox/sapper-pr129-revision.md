### 2026-07-24: PR #129 Nemotron revision (transpose gap + README fix)
**By:** sapper
**What:** Removed the unsupported `prediction_network.decoder_output` to `joiner.decoder_output` dataflow edge and documented the required transpose as a metadata-contract gap. Corrected the streaming-chunk configuration attribution in the README.
**Why:** `decoder.onnx` emits f32 `[batch, 640, target_len]`, while `joint.onnx` accepts `decoder_output` as f32 `[batch, target_len, 640]`; `DataflowEdge` only supports endpoints, dtype, and device transfer, not layout adaptation. The cached package grep finds `chunk_samples: 8960` in `v3/genai_config.json`, not `audio_processor_config.json`.
