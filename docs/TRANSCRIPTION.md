# File transcription

Both `POST /v1/audio/transcriptions` and `/audio/transcriptions` accept multipart
`file`, `model`, `response_format`, `max_tokens` and `stream` fields. Supported encoded audio is detected
by content, regardless of the upload filename; see [formats](AUDIO_PREPROCESSING.md#encoded-audio-formats). The model is
selected before execution; a package with no
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

## CLI files and live segments

`nxrt transcribe` executes the same declared CTC/greedy contract as the server,
without constructing generative prompt tokens. Generative speech retains its
separate decoder path. Both accept file audio; stdin remains PCM16/WAV.

```sh
nxrt transcribe path/to/speech-package recording.mp3 --format json --max-new-tokens 128
```

`--max-new-tokens` is a positive per-segment output budget. Frame profiles reject
`--language` when they declare no such conditioning. File and live segmentation
resolve the actual source rate before computing the package's input limit.
The compiled audio program follows every published branch, accounting for integer
resampling, padding and valid-sample trimming; unused branches do not impose a
limit. Default cuts preserve the exact integer sample allowance. Explicit longer
segments fail instead of letting the workflow silently trim speech.

If the program declares no live trimming window, pass `--segment-seconds`
explicitly. A feature tensor's mel/frame shape is not a duration declaration for
an encoded-audio program. Encoded waveform outputs need no mel-shaped contract.
Legacy server-owned feature extraction accepts arbitrary positive mel counts,
but retains its documented convenience analysis; authored preprocessing remains
the route for explicit analysis rate/FFT/hop/window choices.

## Validation and remaining work

The maintained `transcription_ctc` package runs actual Native CPU WAV preprocessing
and an ONNX waveform-dependent head. Tests verify blank-separated repetitions,
nonblank padding exclusion, empty input, inline/tokenizer vocabularies, greedy
mode, malformed validity counts, all time/class axis layouts, resource limits,
HTTP aliases, JSON/text/SSE responses and incremental generative delivery.
These are dataflow/protocol checks, not recognition-quality or GPU-performance
claims.

Additional codec coverage and richer generative media accounting remain open.
Feature-input analysis executes the named model-authored audio program, including
its analysis geometry and source-rate clip limit. CLI frame execution and
program-derived segmentation are implemented. Realtime sessions use their separate
[contract](REALTIME.md). This feature does not establish complete audio parity.

Encoded audio support and decoder limitations are documented in
[audio preprocessing](AUDIO_PREPROCESSING.md#encoded-audio-formats).

## Generative decoder prompting and transcript projection

A generative speech package declares `transcription_config.json` at its root.
This versioned convenience artifact is separate from the core workflow schema,
like tokenizer/chat-template assets. Missing configuration returns an actionable
501 from file transcription; runtime code never guesses start/task/language
control tokens from vocabulary spellings. `/props.nxrt_capabilities.generative_transcription`
reports the loaded contract; `frame_transcription` remains independent. Frame
profiles retain their own ABI.

```json
{
  "version": 1,
  "prompt_template": "{{ prompt or '' }} <speech>{% if language %}{{ language }}{% endif %}<answer>",
  "languages": {"kind": "choices", "values": ["en", "zh"]},
  "prompt_conditioning": true,
  "response_template": {
    "version": 1,
    "start_anchor": "<speech>",
    "fields": {"content": {"open": "<answer>", "close": "</answer>"}}
  }
}
```

The example spellings are model facts: author the real model's prompt and
boundaries. `prompt_template` uses the canonical bounded Jinja renderer with
`language` and `prompt` (null when absent). Tokenization disables automatic special
insertion and enables registered special-token parsing. Languages may declare
`unsupported` (the default), `any`, or distinct nonempty `choices`. Choice labels
match exactly; the template owns any case conversion or control-token mapping.
`prompt_conditioning` defaults to false. Undeclared conditioning fails before SSE.

An omitted `response_template` returns raw decoded text. A declaration uses the
same response-boundary parser and tokenizer framing policy as ordinary generated
responses. Only its `content` channel becomes the transcript; markers, reasoning
and opaque fields are not transcript text. Rendered prompt content seeds parser
state and is excluded from newly recognized speech. Both raw and projected output
are bounded. SSE deltas are reconciled against the committed raw generation,
and concatenate to the same text returned by buffered JSON/text. Usage counts
actual decoder input tokens and all generated token IDs, including framing.

```sh
curl http://localhost:8080/v1/audio/transcriptions \
  -F model=speech-model -F file=@recording.flac -F language=zh \
  -F prompt='prior transcript context' -F stream=true
nxrt transcribe MODEL recording.flac --language zh --prompt 'prior transcript context'
```

Server budgets are `--max-transcription-input-tokens`,
`--max-transcription-prompt-bytes`, `--max-transcription-template-operations` and
`--max-transcription-text-bytes`; input also respects remaining model context.
CLI transcription exposes `--max-input-tokens`, `--max-prompt-bytes`,
`--max-template-operations` and `--max-text-bytes`. Template fuel/output and token
limits fail during preparation, before generation. Raw output includes hidden
regions for resource accounting, even when only a short transcript is visible.

Audio chat with a normal `chat_template` retains its normal multimodal prompt
and response handling. Without a chat template, audio-only generation renders
the declared transcription prompt; one user message's text is supplied as the
optional context prompt. Multi-turn text/role history requires a chat template.
Chat response projection remains the standard tokenizer `response_template`;
file transcription/CLI `transcribe` use the ASR-specific declaration above.
CLI `generate --audio` uses declared conditioning and retains ordinary raw
`generate` output semantics.

The maintained `transcription_projected` Native fixture produces a
waveform/conditioning-dependent token wrapped in declared framing. HTTP and CLI
acceptance verifies actual conditioning, input/output usage, buffered/SSE parity,
absence of framing in transcripts, and rejection of undeclared/over-budget input.
