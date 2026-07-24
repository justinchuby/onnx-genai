# Nemotron audio pipeline metadata example

This example targets NVIDIA Nemotron 3.5 ASR Streaming 0.6B, obtained on
2026-07-24 from Foundry Local, not converted locally:

```text
catalog alias: nemotron-3.5-asr-streaming-0.6b
artifact id:   nemotron-3.5-asr-streaming-0.6b-generic-cpu:3
publisher:     Microsoft
model type:    ONNX
source:        azureml://registries/azureml/models/nemotron-3.5-asr-streaming-0.6b-generic-cpu/versions/3
```

It was downloaded with `foundry-local-sdk` 1.2.3:

```python
manager.catalog.get_model("nemotron-3.5-asr-streaming-0.6b").download()
```

The downloaded package is intentionally not committed.  Its Foundry catalog
size is 756 MB. Recreate it with a cache directory outside the repository's
tracked files, then inspect it with:

```bash
python3 - <<'PY'
from pathlib import Path
import onnx
for path in sorted(Path("MODEL_DIR").glob("*.onnx")):
    graph = onnx.load(path, load_external_data=False).graph
    print(path, [v.name for v in graph.input], [v.name for v in graph.output])
PY
```

## Inspected package contract

`audio_processor_config.json` specifies 16-kHz PCM, a 512-point FFT, 400-sample
Hann window, 160-sample hop, 128 mel bins, pre-emphasis 0.97, and 8,960 samples
per streaming chunk. It produces `audio_signal` float32 `[1, 65, 128]`.

| Component | Inputs | Outputs |
|---|---|---|
| `encoder.onnx` | `audio_signal` f32 `[1,65,128]`; `length` i64 `[1]`; `cache_last_channel` f32 `[1,24,56,1024]`; `cache_last_time` f32 `[1,24,1024,8]`; `cache_last_channel_len` i64 `[1]`; `lang_id` i64 `[1]` | `outputs` f32 `[1,7,1024]`; `encoded_lengths` i64 `[1]`; `cache_last_channel_next` f32 `[1,24,56,1024]`; `cache_last_time_next` f32 `[1,24,1024,8]`; `cache_last_channel_len_next` i64 `[1]` |
| `decoder.onnx` (LSTM prediction network) | `targets` i64 `[batch,target_len]`; `h_in` f32 `[2,batch,640]`; `c_in` f32 `[2,batch,640]` | `decoder_output` f32 `[batch,640,target_len]`; `h_out` f32 `[2,batch,640]`; `c_out` f32 `[2,batch,640]` |
| `joint.onnx` | `encoder_output` f32 `[batch,time,1024]`; `decoder_output` f32 `[batch,target_len,640]` | `joint_output` f32 `[batch,time,target_len,13088]` |
| `silero_vad.onnx` | `input` f32 `[?,?]`; `state` f32 `[2,?,128]`; `sr` i64 `[]` | `output` f32 `[unk__54,1]`; `stateN` f32 `[unk__55,unk__56,unk__57]` |

The encoder has convolution/attention streaming caches and the prediction
network has LSTM state.  Both pairs are represented with `io.state_pairs`;
they are replacement state, not transformer KV cache. No encoder-parity
comparison was run, so this example makes no claim about the previously
reported approximately `1e-2` LSTM/encoder parity difference.

## Fit and gaps

**Fits cleanly:** `audio_encoder` component role; `audio_features`-named
front-end output; explicit component filenames; typed dataflow boundaries;
audio-presence and `PhaseRunOn::EveryStep` gates; and fixed, zero-initialized
recurrent state pairs for both encoder caches and LSTM state.

**Gaps:** the current component `io` schema has semantic bindings for decoder
ports and state pairs, but no generic typed port inventory. The table above is
therefore documentation for required terminal ports such as `lang_id`,
`encoded_lengths`, and VAD state rather than machine-actionable metadata.
There is also no native audio preprocessing program (FFT/mel/pre-emphasis), no
streaming chunk/window scheduling contract, no transducer blank/symbol loop
contract (`blank_id=13087`, `max_symbols_per_step=10`), and no way to state
that joiner argmax conditionally feeds the next LSTM token. The composite stages
describe their execution phases but do not make that conditional feedback edge
executable. A full E2E runner requires these additions; this example validates
only the metadata parse/schema/pipeline-DAG contract.
