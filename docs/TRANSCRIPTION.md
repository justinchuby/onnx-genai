# File transcription

Both `POST /v1/audio/transcriptions` and `/audio/transcriptions` accept multipart
`file`, `model`, `response_format`, `max_tokens` and `stream` fields. PCM16 WAV is
currently supported. The model is selected before execution; a package with no
admitted speech path returns `501 not_supported_error`.

```sh
curl http://localhost:8080/v1/audio/transcriptions \
  -F model=speech-model -F file=@recording.wav -F max_tokens=128 -F stream=true
```

Buffered JSON follows the pinned llama.cpp transcript contract:

```json
{
  "type": "transcript.text.done",
  "text": "recognized text",
  "usage": {
    "type": "tokens",
    "input_tokens": 120,
    "output_tokens": 4,
    "total_tokens": 124,
    "input_tokens_details": {"cached_tokens": 0}
  }
}
```

`stream=true` returns SSE `transcript.text.delta` events with `delta`, followed by
that final `transcript.text.done` object and `data: [DONE]`. Deltas concatenate to
the final text. Generative speech publishes decoder deltas as they arrive, uses
bounded delivery, reconciles the final text and releases the driver receiver on
client disconnect. A one-pass frame decoder naturally publishes its complete
delta after acoustic inference. Stream failures publish an error and terminator.
The existing `response_format=text` convenience response remains available for
buffered requests; streaming requires JSON event format.

`max_tokens` is a positive per-request output budget bounded by
`--max-output-tokens` (default 4096). Frame decoders stop committing classes at
that budget rather than returning a resource error for an ordinary truncated
response. Hard acoustic-frame and render-buffer limits remain separate.

## Non-generative frame profiles

A package with `kind: transcription`, version `"1.0"`, executes its acoustic
workflow once. It neither invents an autoregressive prompt nor runs token
sampling. The admitted profile declares:

```yaml
profiles:
  speech:
    kind: transcription
    version: "1.0"
    outputs: {logits: acoustic.logits, frame_lengths: acoustic.lengths}
    decoding:
      kind: ctc
      blank_id: 0
      collapse_repeats: true
      time_axis: 1
      class_axis: 2
      lengths: frame_lengths
      vocabulary:
        source: inline
        size: 4
        tokens: ["<blank>", "a", "|", "b"]
        word_delimiter: "|"
```

The workflow declares exactly one request-sourced runtime `media` input with a
`uint8 [encoded_bytes]` contract. Its authored preprocessing owns waveform or
feature extraction. Other required inputs need an explicit conditioning contract;
the file API does not infer them. A frame profile currently rejects nonempty
`language`/`prompt` conditioning rather than silently dropping it.

Logits are float16, bfloat16 or float32 tensors of rank at least 2 with explicit,
distinct time/class axes. Other axes must be singleton; axes may be permuted.
Vocabulary size must agree with the class axis. Tokenizer tables may derive that
size from a fixed positive class dimension when no explicit size is provided. A padded time axis and
`decoding.lengths` bind the same workflow value. Runtime lengths must be
nonnegative and no larger than the actual frame extent. Padded time companions
use the core tensor contract's exact outer-axis shape. If classes precede time,
those class positions must report the same valid prefix for the one audio row;
disagreement is rejected rather than choosing or clamping a count. Unpadded
profiles may declare a scalar or count with singleton axes. Padded frames are never decoded.

CTC chooses the per-frame argmax, collapses configured repeated class IDs before
removing blank/ignored classes, and normalizes declared word delimiters. A blank
therefore separates equal classes that must both survive. `greedy_argmax` also
executes according to its explicit collapse/blank settings. Ties choose the
lowest class ID; nonfinite logits in a decoded frame fail with frame/class
context. Inline tables are package data. Tokenizer tables are resolved from the
actual executing package, never a caller-supplied tokenizer; delimiter and ignore
selections must exist in that table.

`/props` reports `modalities.audio` and
`nxrt_capabilities.frame_transcription`. A frame-only model accepts file
transcription, not generative chat audio attachments. Its usage counts valid
acoustic frame positions as `input_tokens` and additionally exposes that count
as `input_tokens_details.audio_frames`; output tokens count committed classes
after blank/repeat/ignore processing. Generative speech reports its admitted
decoder-prompt IDs and returned token IDs, with the engine's cache count. Richer
encoder-token accounting requires authored model usage outputs.

The server limits physical acoustic output frames with
`--max-transcription-frames` (default 1,000,000, including padding) and UTF-8 render
buffer bytes with `--max-transcription-text-bytes` (default 4 MiB, before delimiter
normalization). Positive limits are checked during argument parsing. Violations
name the resource and tell clients to shorten audio or operators to raise its
configured allowance.

## Validation and remaining work

The maintained `transcription_ctc` package runs actual Native CPU WAV preprocessing
and an ONNX waveform-dependent head. Tests verify blank-separated repetitions,
nonblank padding exclusion, empty input, inline/tokenizer vocabularies, greedy
mode, malformed validity counts, all time/class axis layouts, resource limits,
HTTP aliases, JSON/text/SSE responses and incremental generative delivery.
These are dataflow/protocol checks, not recognition-quality or GPU-performance
claims.

MP3 and broader codecs remain open. The existing generative convenience path rejects nonempty `prompt` conditioning
rather than dropping it, and still needs a general model-authored ASR prompt/response projection and richer
media accounting; its legacy feature-input geometry and CLI transcription path
also need further generalization. Realtime sessions use their separate declared
[contract](REALTIME.md). This feature does not establish complete audio parity.
