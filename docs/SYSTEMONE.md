# Non-generative SystemOne decisions

`POST /v1/systemone` evaluates a model-declared decision head. It does not ask a
chat model to generate an answer. A package must declare a `decision` v1, v2 or v3 task
profile; ordinary generation/embedding/reranking packages receive actionable
`501 not_supported_error`. `/props` advertises `nxrt_capabilities.decision`.

```json
{
  "model": "decision-model",
  "state": "The text or structured context to assess",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Choose the appropriate destination",
      "criteria": {"billing": "Account and invoice questions", "support": "Technical help"}
    },
    "quality": {
      "type": "score",
      "instructions": "Rate the quality",
      "criteria": ["poor", "adequate", "excellent"]
    },
    "acceptable": {"type": "noul", "instructions": "Is this acceptable?"}
  }
}
```

The response contains `model`, `answers` keyed by question ID, and
`usage: {input_tokens, output_tokens: 0}`. A choice answer has `choice`,
`probabilities` and `confidence`. A score has `score`, `probabilities`,
`confidence` and `legend`. A `noul` answer has its probability under `noul`.
Option ordering and defaults come from the package input policy. Score levels
are zero-based in array order; `noul` returns the probability of the `true` key
regardless of option order. Ties select the first option. All prompt
variants count toward input usage; tokenizer padding does not.

## Package contract

Decision profiles require at least `schema_version: v1.10`. New packages using
`schema_version: v1.11` must explicitly declare `input_policy`. Both parsed and
programmatically constructed metadata enforce the version gates. A v1.10
package retains its defined rules: required instructions, explicit question
type, lexical choices, score levels 2–10, and strict false-first boolean options.

```yaml
profiles:
  decisions:
    kind: decision
    version: '1'
    outputs:
      probabilities: calibrated.probabilities
    decision:
      prompt_template: |
        {{ state | tojson }}
        {{ instructions | tojson }}
        {% for option in options %}{{ option.key }}: {{ option.description | tojson }}
        {% endfor %}
      input_policy:
        state: payload_or_null
        instructions:
          kind: optional
          default: ""
          payload: payload
        default_type: choice
        max_questions: 64
        choice:
          kind: dynamic
          minimum: 2
          order: request
          descriptions: payload_or_null
          nonempty_keys: true
        score:
          minimum: 1
          maximum: 16
          descriptions: payload
          legend: canonical_json_text
        noul:
          order: criteria_true_first
          criteria:
            kind: permissive
            no_yes_aliases: true
      max_options: 16
      variants: direct
      choice_confidence: uniform_margin
      score_confidence: mode_distance
      score: expected_level
```

The example uses core schema v1.11. Omit a question kind from `input_policy` to
disable it. Instructions may be required, optional with an authored default, or
forbidden. Payload admission may allow any JSON, any non-null JSON, text only,
text/object/array (`payload`), those types plus null (`payload_or_null`), or
nonempty text/object/array (`nonempty_payload`; empty containers remain valid).
State is always present even when its policy permits null. Missing/null/non-string
question types use `default_type` when declared; unknown strings are rejected.

Dynamic choices declare their minimum cardinality, description payload, key
validation and request/lexical ordering. A fixed choice policy instead declares
`kind: fixed` and an ordered `options` array of `{key, description}`; requests
must provide that exact set. Score ranges are authored, further constrained by
the model head and operator limits. `structured` legends retain JSON values;
`canonical_json_text` keeps strings unchanged and serializes other values with
recursively sorted object keys. Boolean order may be `false_first`, `true_first`,
or `criteria_true_first` (true first when it is the first of two criteria).
Boolean criteria may be forbidden, strict with a declared description payload,
or permissive. Permissive criteria ignore unrelated keys and optionally accept
`no`/`yes` aliases only when the corresponding canonical key is absent; explicit
canonical null takes precedence. Missing optional instructions use the authored
default; explicit null is validated normally. Unknown request/question fields
are ignored, matching the upstream transport.

The model author supplies the prompt; there is no model-name or architecture
lookup. Its Jinja context contains `state`, `instructions`, `type`, `options`
(an ordered array of key/description objects), and zero-based `variant`.
The tokenizer and its postprocessor produce the actual input IDs and feature
rows. The profile uses a request-sourced `prompt_tokens` runtime workflow input.
Optional application inputs are `attention_mask`, `token_type_ids` (int64
`[1,sequence]` rows), and `option_count` (int64 scalar or `[1]`). Unknown required
inputs fail admission instead of receiving guessed values.

The workflow emits a host-readable float16/bfloat16/float32 probability tensor
with shape `[options]` or `[1,options]`. The runtime checks cardinality, range,
finiteness and total probability mass. Small dtype-rounding discrepancies are
normalized. Raw logits or independent sigmoid scores must be converted into a
distribution by the authored workflow; invalid mass/range, malformed shapes and
missing outputs fail. Calibration, head selection, option selection, temperature and
bias belong to the authored workflow. Fixed head/sequence dimensions constrain
requests explicitly; dynamic shapes support varying option/token counts.

`reverse_average` evaluates both direct and reversed option orders, realigns the
second result and averages probabilities. `direct` evaluates one variant.
Confidence policies are explicit: `uniform_margin`, `mode_distance`,
`normalized_entropy`, or `maximum_probability`. Score policies are
`expected_level` or `median_level`. No policy is inferred from model identity.
The decision workflow must not declare text/token outputs; its generation token
budget is zero. Model execution uses the ordinary bounded engine worker queue,
and dropped clients do not execute pending commands. Converted probabilities
retain canonical host-memory ownership through the returned engine result.

## Ordered image inputs (decision v2)

Schema v1.14 adds decision v2 image workflows. The profile declares
`decision.images` as an ordered array of workflow endpoint names and exposes
both `outputs.probabilities` and `outputs.prompt_tokens`. Image endpoints must be
opaque application-sourced `uint8 [encoded_bytes]` vectors; the authored workflow
owns image decoding, preprocessing and the probability head. Fixed encoded-byte
lengths and required/optional input positions are admitted before execution.
The template additionally receives `image_count`. Decision v1 stays text-only.

For example, a package with two image endpoints can receive:

```json
{
  "state": "Compare the two images",
  "images": ["data:image/png;base64,...", "data:image/png;base64,..."],
  "questions": {"match": {"type": "choice", "instructions": "Choose a result",
    "criteria": {"same": "Same subject", "different": "Different subjects"}}}
}
```

Input order matches the upstream transport: `files`, then its `images` alias,
then image parts in chat-state message order. State can be a messages array or
an object containing `messages`; `image_url` parts accept a data URI directly or
`{url: data_uri}`. Those parts are removed from textual state while other parts,
message fields and wrapper fields remain intact. Decision v2 admits image data URIs; v3 additionally admits the audio forms below.
This decision transport does not fetch remote URLs. Video requires further
executable metadata and runtime support.

`--max-decision-media-bytes` defaults to 64 MiB of unique retained encoded bytes
per request. Repeated image references share their allocation across questions
and variants. Position counts are checked before loading; all images must decode
and all questions/variants must prepare before any model work is submitted.
The existing bounded image loader is shared with reranking. Encoded tensor
staging reserves engine host memory before allocation; the tensor and its
no-copy aliases retain that reservation until their final owner is released.

The workflow reports total `prompt_tokens`, including text and model-authored
image expansion, once per variant. Counts must be nonnegative, cover admitted
nonpadding text tokens and fit the configured/model input limit. Usage sums the
actual counts over questions and variants; output tokens remain zero. No hidden
image-token formula or model identity determines the expansion. The
`decision_images` fixture executes Native CPU image-preprocess@2 and a real ONNX
probability head; changing ordered image positions changes its output. This is
functional evidence, not recognition-quality or device-performance evidence.

## Mixed audio and spoken instructions (decision v3)

Core schema v1.15 adds `decision.media`, an ordered list of
`{kind: image|audio, input: workflow_endpoint}` bindings, and an optional
`decision.spoken_instructions` audio endpoint. V3 omits the v2 `images` field;
state media and spoken instructions use distinct opaque application-sourced
`uint8 [encoded_bytes]` inputs. The same endpoint cannot bind twice. Required
endpoints must be supplied; the declared mixed-media order is preserved.
Both probabilities and actual prompt-token usage are required outputs.

State media order is `files`, then `images`, then chat-state parts. Top-level
entries accept image/audio base64 data URIs. Embedded `input_audio` parts and
spoken instructions accept `{data: base64_or_audio_data_uri}` or
`{url: base64_or_audio_data_uri}` inside `input_audio`; `data` takes precedence.
Audio currently requires PCM16 WAV. HTTP/file URLs, video and other codecs fail
with field-specific errors before inference. For example, with an image/audio
state contract and a spoken endpoint:

```json
{
  "state": "Compare the recording with the picture",
  "files": ["data:image/png;base64,...", "data:audio/wav;base64,..."],
  "questions": {"match": {
    "instructions": {"type": "input_audio", "input_audio": {"data": "..."}},
    "criteria": {"same": "Same subject", "different": "Different subjects"}
  }}
}
```

The template receives `media_types`, `image_count`, `audio_count` and
`spoken_instructions` (boolean). Spoken instructions render no encoded bytes;
the audio travels through the declared workflow endpoint. Each question has its
own spoken input, while state media and repeated identical references share the
request-local encoded-byte budget. All media and questions are prepared before
execution. `/props` exposes `decision_media_order`, `decision_image_limit`,
`decision_audio_limit` and `decision_spoken_instructions`.

The maintained `decision_audio` fixture runs Native CPU image-preprocess@2,
repeated audio-preprocess@2 and a real ONNX probability head. Independent state
and question audio alter probabilities; changing clip length changes authored
usage, and total media expansion obeys the input-token limit. This proves data
flow, not recognition quality. Log-mel bin count, sample rate, FFT length, hop and Hann window length are
package-authored geometry. Broader waveform dataflow, normalization modes and
codecs remain separate compatibility work.

## Limits and remaining work

`--max-decision-input-tokens` defaults to 8192 per prompt variant (also capped by
model context); `--max-decision-questions` defaults to 64 and
`--max-decision-options` to 128 (also capped by the package/head). They must be
positive. `--max-decision-prompt-bytes` defaults to 4 MiB before tokenization,
and `--max-decision-template-operations` to 1,000,000 Jinja operations per variant.
Rendered-byte and interpreter-fuel limits fail before model execution, including
large template loops. All questions/variants are prepared before any inference is submitted.
The package's exact tokenizer padding/truncation contract remains authoritative.

The v1 execution path supports text and structured JSON state/instructions; v2
adds ordered images; v3 adds mixed state audio and spoken instructions. Video,
joint/packed decision heads and shared generative /
decision workflow entry points still need explicit metadata and runtime support.
GGUF decision-head import is not implemented by registering this HTTP route.
The small native ONNX fixture proves actual input-dependent head execution,
not compatibility with every upstream decision model. These remain part of the
overall compatibility work; tool execution remains delegated to external harnesses.

The native `audio-preprocess@2` adapter can now be invoked repeatedly with
component-local outputs, so separate state and question clips can retain their
own features and valid lengths. This prerequisite does not yet admit audio at
`/v1/systemone`: ordered audio/spoken-question metadata, transport preparation
and broader waveform/log-mel transform support remain required.
