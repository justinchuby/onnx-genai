# onnx-genai

A Rust inference runtime for generative AI models, built on ONNX Runtime.

**Reference implementation** of the [ONNX Inference Metadata Standard](https://github.com/onnx/onnx/issues/8184).

## Features

- **Generation:** greedy and categorical sampling with temperature, top-k, top-p,
  min-p, repetition, frequency, and presence controls.
- **Speculative decoding:** separate draft models and model-free prompt
  lookup/n-gram proposals, with greedy target verification and KV rewind.
- **Structured generation:** complete JSON plus llguidance-backed JSON Schema,
  regex, and Lark constraints; fill-in-the-middle (FIM) for compatible coder
  tokenizers.
- **Agent serving:** OpenAI-compatible chat completions, SSE streaming, model
  discovery, persistent sessions, Hugging Face/MiniJinja chat templates
  (including ChatML-style models), and tool calling (`tools`, `tool_choice`,
  and `<tool_call>` parsing).
- **Concurrency:** multi-session generation, prefix reuse, priority scheduling,
  swap preemption, and continuous static-cache batching. A tiny CPU fixture
  measured about 6.2x aggregate fixed-batch throughput; this is not a
  real-model GPU performance claim.
- **KV and long context:** paged allocation, copy-on-write fork, rewind,
  prefix cache, tiered storage, and opt-in int8 KV pages. Mobius static-cache
  models use runtime-owned in-place KV buffers for O(1) work per decoded token
  with respect to context length.
- **Pipelines and models:** metadata-declared multi-model pipelines, a tested
  tiny vision-language pipeline fixture, and real Qwen2.5-0.5B-Instruct and
  TinyStories generation built through Mobius.
- **Execution providers:** select CPU, WebGPU, CUDA, CoreML, or any ORT plugin
  EP with `ONNX_GENAI_EP` — a comma-separated priority list is supported so
  several providers (including multiple plugins) can be composed; unavailable
  providers warn and fall back to CPU.
- **Extensibility:** public `Sampler`, `SpeculativeProposer`, logit processor
  registry, and KV/pipeline APIs, plus an internal `DecodeBackend` seam shared
  by dynamic and static-cache decoding.

## Architecture

```text
OpenAI HTTP server / CLI / Rust facade
                    │
     chat templates, tools, constraints
                    │
 Generation engine + shared decode loop
     ├── speculative proposers/samplers
     ├── scheduler + continuous batching
     └── pipeline executor
                    │
 KV management: pages, prefix trie, tiering,
 int8 storage, sessions, static-cache buffers
                    │
 ONNX Runtime sessions + Hugging Face tokenizers
```

The paged KV manager currently supplies allocation, sharing, tiering, and
materialization; true paged-attention kernels are not yet implemented.

## Quick Start

### Build a model with Mobius

`scripts/build_qwen.sh` builds `Qwen/Qwen2.5-0.5B-Instruct` using a local
[Mobius](https://github.com/justinchuby/mobius) checkout:

```bash
MOBIUS_DIR=/path/to/mobius scripts/build_qwen.sh
# Output: models/qwen2.5-0.5b
```

For bounded, in-place KV storage and efficient long-context decode, export a
static-cache model. `MAX_SEQ_LEN` fixes the cache capacity at build time:

```bash
MOBIUS_DIR=/path/to/mobius STATIC_CACHE=1 MAX_SEQ_LEN=8192 scripts/build_qwen.sh
# Output: models/qwen2.5-0.5b-scatter
```

### Run the CLI

The `onnx-genai` binary lives in the `onnx-genai-cli` package:

```bash
cargo build --release -p onnx-genai-cli

./target/release/onnx-genai generate \
  models/qwen2.5-0.5b \
  --max-new-tokens 64 \
  --temperature 0 \
  --stream \
  --prompt "Write a short Rust hello-world program."
```

When `--max-new-tokens` is omitted, `generate` and `run` use whatever budget
remains in the model's effective context window, stopping on EOS, a stop
sequence, or context length. Sampling flags such as `--temperature 0.7`,
`--top-p`, or `--top-k` enable stochastic sampling, while `--temperature 0` and
`--greedy` force argmax. If a package has no discoverable context limit, the CLI
warns and uses a finite fallback; pass `--max-context TOKENS` or declare
`model.max_sequence_length` in inference metadata to make context-fill automatic.
In the REPL this budget is recomputed for each turn as history grows; `/stats`
shows `ctx used / max`.

On a shared machine, add `--cpu-cores N` to `generate`, `run`, or `transcribe` to
cap native CPU decode to N workers (for example, `--cpu-cores 8`); where
supported, those workers are pinned to at most N allowed CPUs. The equivalent
environment variable is `ONNX_GENAI_CPU_DECODE_THREADS=N`; precedence is CLI flag >
environment variable > automatic sizing. Omitting both preserves the full
peak-throughput default. This is a decode-worker budget, not a hard cpuset for
prefill or ONNX Runtime threads; combine it with an OS cpuset/taskset when the
entire process must be confined.

`onnx-genai run <model>` starts an interactive REPL. In a terminal it uses a
rich line editor with cursor movement, persistent history, bracketed paste,
slash-command completion, and multiline input via Alt+Enter (Enter submits).
Interactive terminal text generation shows compact per-turn stats by default;
pass `--no-stats` to hide them. Stats are enabled only when stdin and stdout are
terminals, and they are printed to stderr; piped stdout remains byte-stable
generated text. If stdin or stdout is piped, the REPL keeps the original plain
`>>> ` line loop for script compatibility. One Ctrl-C stops the current
generation; two in a row exit. Slash commands control the session:

```text
>>> /help
>>> /system Be concise.
>>> /image ./cat.png What is in this image?
>>> /audio ./speech.wav
>>> /raw
>>> /stats                  # runtime toggle for compact per-turn stats
>>> /profile on
>>> /model ./other-model
>>> /ep cpu
>>> /backend ort
>>> /reset
```

`/model`, `/ep`, and `/backend` reload the model, because an ONNX session is
created against its execution provider and decode backend and cannot be moved
between them; the conversation is cleared with it, since a reply belongs to the
model that produced it. If the new selection fails to load, the message says so
and the previous session keeps running. With no argument each reports the
current setting, and `/ep` also lists what this build can select — provider
support is compiled in, so a provider left out of the build cannot be chosen at
runtime.

`/profile on` turns the report on mid-session, and starts a Perfetto timeline
with it at full detail — deciding you want a timeline usually happens after
something looks wrong, which is the one moment a startup-only switch cannot
serve. `/profile trace <path>` chooses where it goes (`trace off` stops it), and
`/profile verbosity <decisions|ops|full>` changes how much it records between
turns. Full adds a span per worker thread per operator and costs about 4%.

The per-stage ORT breakdown is the exception: it is switched on from the
environment before any thread starts, so it needs `--profile` at startup, and
the command says so rather than printing a report that is quietly missing its
most detailed section.

`/stats` toggles per-turn numbers, for watching throughput and cache behavior
without the full `--profile` report. `generate` uses the same compact line for
terminal text generation unless `--no-stats` is passed. While a REPL reply
streams, the numbers update live beneath it; when the turn ends they settle into
a deliberate two-line block: performance and termination first, cache/scheduler/
memory behavior second.

```text
[ 613 in · 64 out · backend native · 41.7 tok/s · e2e 39.3 tok/s · ttft 116 ms · finish stop-seq · cap 3.6k->128 ]
[ cache 598/613 98% · ctx 677/8.2k · mm 120 · enc 1/2 · pg +5/-2 hot 1 pref 3 fail 1 · rss 2.5 GiB ]
```

If the scheduler had to admit a smaller decode budget, or the KV page pool
allocated/freed/evicted pages during the turn, those appear in this block.

The live view is drawn with [ratatui](https://ratatui.rs) into an *inline*
viewport rather than an alternate screen, so finished lines spill into the
terminal's own scrollback and the conversation stays selectable, copyable, and
present after the session ends. Compact stats are used only on a terminal and go
to stderr for `generate`: a piped session, or one started with `--no-stats`, gets
exactly the plain streaming output it got before.

### Image and audio input

Vision-language and speech packages declare their preprocessing contract in
inference metadata (`preprocessing.image` + `pipeline.vision` for images, an
`input_features` component input for audio). `run` and `generate` accept those
modalities on any package that declares one:

```bash
./target/release/onnx-genai generate models/tiny-vlm \
  --image ./cat.png \
  --prompt "What is in this image?"

./target/release/onnx-genai generate models/whisper-tiny \
  --audio ./speech.wav \
  --prompt ""
```

Audio is transcription: the model's own decoder prompt replaces the typed text,
because the clip carries the content. A model that declares neither contract
rejects the attachment with an error naming what it does accept.

You do not have to know a model's image placeholder token. Pass the images and
write the prompt normally — one placeholder per image is prepended for you:

```bash
onnx-genai generate models/my-vlm \
  --image left.png --image right.png \
  --prompt "What changed between these two photos?"
```

To control where each image sits in the sentence, write the placeholders
yourself and they are honored verbatim:

```bash
onnx-genai generate models/my-vlm --image cat.png --image dog.png \
  --prompt "The first <image> is a cat and the second <image> is a dog. Compare them."
```

A *partial* set is rejected rather than topped up: once you start positioning
placeholders, guessing where the rest belong would silently change which image a
sentence refers to. The metadata contract behind this is documented in
[docs/genai/MODEL_METADATA.md](docs/genai/MODEL_METADATA.md#multimodal-input-the-placeholder-contract),
which also covers how audio input differs.

#### Asking again about the same attachment

An image costs a turn twice: the encoder forward pass, and a prompt in which
that one image has expanded into hundreds or thousands of tokens. Neither is
repeated when you keep talking about the same picture — the encoder's output is
memoized under a digest of the exact pixels that produced it, and the decoder
keeps the KV it already computed, prefilling only the tokens the new turn added.

Attach a *different* image and both are recomputed, because that digest is part
of the cache key. It has to be: placeholder expansion makes two different
photographs produce byte-identical token sequences, so a cache keyed on tokens
alone would answer fluently about a picture the model was never shown.
A prompt that diverges from the previous turn — a forked conversation, an edited
question, a reasoning model's stripped history — keeps the head it still shares
rather than starting over. The same applies across *different* conversations on
the server: many agents running under one long system prompt each reuse it,
because reuse is computed over the common prefix rather than requiring the new
prompt to extend the old one. `--profile` shows what was skipped:

```text
encoder cache                     1 hit / 0 run
multimodal prefix reuse          613 tokens
```

`EngineConfig::pipeline_cache_bytes` bounds the memoized encoder outputs
(512 MiB by default; `0` turns the cache off).

Every command here can be run without installing anything, against the tiny
ONNX fixtures in `tests/fixtures/`; see
[the CLI crate's README](crates/onnx-genai-cli/README.md#running-from-source).

### Profiling

`--profile` reports where the time went. It works on every subcommand:

```bash
onnx-genai --profile generate models/qwen2.5-0.5b --prompt "..." --max-new-tokens 40
```

```text
── profile ──────────────────────────────────
model                    models/qwen2.5-0.5b
execution provider       cpu
model load                   3598.2 ms
prompt tokens                    36
generated tokens                 20
time to first token           116.3 ms
generation wall time          599.9 ms
decode throughput             39.28 tok/s
end-to-end throughput         33.34 tok/s
inter-token latency      mean 24.2 / p50 23.0 / p90 27.5 / p99 36.5 / max 36.5 ms
finish reason            MaxTokens
peak resident memory        2.5 GiB
kv cache budget             7.2 GiB (314560 tokens)

per-stage breakdown:
stage                          total_ms      calls        us/call     us/token
------------------------------------------------------------------------------
ort.session_run                 890.337         36       24731.57     25438.19
ort.sampling                     53.813         35        1537.52      1537.52
...
```

Decode throughput excludes the prefill wait, and end-to-end includes it, because
a long prompt inflates the second without the model decoding any faster.
Percentiles sit next to the mean because a run that averages 24 ms/token but
stalls for 400 ms mid-sentence feels broken, and only the tail shows it. The
per-stage table answers "ORT kernels or our orchestration?".

Each mode adds its own counters: denoise steps and ms/step for `--output-image`,
audio produced and real-time factor for `--output-audio`, and segments, audio
transcribed, real-time factor and slowest segment for `transcribe`.

#### KV page activity

`--profile` also reports what the run did to the KV page pool, as a delta rather
than lifetime totals:

```text
kv page activity:
  allocated                       5
  freed                           5
  evicted from hot tier           3  (pool under pressure)
  reclaimed from prefixes         7
  allocation failures             1  (pool exhausted)
```

`--vram-limit` sets the ceiling those pages come out of — a byte count (`8GiB`),
a fraction of detected capacity (`0.9`), or `auto`. An explicit byte value is
authoritative: the runtime's device-capacity probe is still provisional, so this
is how you tell it what is really available. Raising it enlarges the KV cache and
therefore the context that fits. `--host-ram-limit` does the same for the warm
offload tier.

Evictions and allocation failures are the signal that a context no longer fits:
they explain a latency cliff that no per-token number does. The last three lines
appear only when they happen, and a run that touched no pages says nothing
rather than printing zeros.

#### What the memory numbers cover

Memory is reported from two independent sources, because neither alone is the
whole picture:

| line | source | covers |
|---|---|---|
| `peak resident memory` | the kernel's high-water mark for this process | model weights, KV pages, ONNX Runtime arenas, and transient tensors |
| `device memory in use` | the engine's resource governor | what the engine accounts as allocated on the device, against its own ceiling |
| `kv cache budget` | the engine's derived budget | bytes reserved for KV, and how many tokens that holds |
| `device memory breakdown` | the governor's fixed-reservation split | how the ceiling divides into model weights and KV cache |

The breakdown answers "why doesn't this fit": weights are fixed, so the KV
budget is what a longer context has to come out of.

```text
device memory breakdown:
  model weights             1.8 GiB   25.7%
  kv cache                  5.4 GiB   74.3%
  kv pages                    14612 x 384.0 KiB
  (activations and runtime overhead not yet measured by the engine)
```

Model weights are measured from the package on disk — the `.onnx` graph plus its
ONNX external-data blob — and the KV budget is derived from what is left, so the
engine no longer promises KV capacity the weights are already using. Activations
and runtime overhead still read zero and are named as unmeasured rather than
printed as `0 B`; they need runtime instrumentation.

Two platform caveats worth knowing before you read a number as "total memory":

- **Discrete GPU (CUDA):** device allocations do **not** appear in the host
  process's resident set, so `peak resident memory` excludes VRAM entirely.
  `device memory in use` is the figure to read — but it is the engine's own
  accounting, not the driver's, so it does not include allocator fragmentation
  or CUDA context overhead. A driver-level probe (`cudaMemGetInfo`) is not wired
  up yet.
- **Apple Silicon (unified memory):** there is no separate device pool, and GPU
  buffers are allocated in the process's address space, so `peak resident
  memory` already accounts for them. `device memory in use` stays absent rather
  than reporting a misleading zero.

A number that was never measured is omitted rather than printed as `0`, so an
absent line means "not accounted here", never "nothing was used".

```bash
# Machine-readable, for diffing runs or plotting in CI (`-` writes to stdout)
onnx-genai --profile-json bench.json generate models/qwen2.5-0.5b --prompt "..."

# Chrome Trace Event timeline, viewable at https://ui.perfetto.dev
onnx-genai --profile-trace trace.json generate models/qwen2.5-0.5b --prompt "..."
```

### Generate images

`generate --output-image` renders a prompt through a diffusion package (see
[docs/genai/DIFFUSION.md](docs/genai/DIFFUSION.md)):

```bash
./target/release/onnx-genai generate models/stable-diffusion-1.5 \
  --prompt "an astronaut riding a horse" \
  --negative-prompt "blurry, low quality" \
  --steps 25 --guidance-scale 7.5 --seed 0 \
  --width 512 --height 512 \
  --output-image out.png
```

Steps, guidance scale, and the sampler default to the values the package
declares. For packages whose pipeline stops at the latent instead of declaring
a final VAE phase, add `--vae-decoder <latent-to-image.onnx>` (and
`--vae-scaling-factor`).

### Reasoning models

Models that emit a chain of thought before the answer are handled automatically.
The delimiters are read from the package's own chat template — a template that
writes `<think>` is declaring the convention — so nothing is keyed off a model
name and a package that declares none is untouched.

In `run`, reasoning is dimmed as it streams (on a terminal only, so a piped
transcript stays clean), and **only the answer becomes conversation history**.
These models are trained with earlier turns' thinking removed: replaying it
degrades quality and inflates the context, since the reasoning of a long session
can dwarf the conversation itself.

When `--max-new-tokens` is omitted, the CLI gives reasoning models the same
model-following budget as any other model: the remaining context window. If an
explicit or fallback decode budget still runs out inside the reasoning, the turn
genuinely has no answer. The exchange is dropped rather than stored as an empty
reply, which would otherwise teach the model that questions go unanswered:

```text
note: generation stopped inside the model's reasoning after hitting
--max-new-tokens 2. No answer was produced, so this turn is not kept.
Try --max-new-tokens 4.
```

Because the thinking is stripped before history is replayed, a follow-up prompt
diverges from what the model last held in its KV cache, at the point the
thinking began. Reuse covers everything before that, so the conversation and any
attached image are still reused; only the discarded reasoning is recomputed.

### Transcribe speech

`transcribe` turns speech into text, from files or a live stream:

```bash
# One or more WAV files
onnx-genai transcribe models/whisper-tiny talk.wav --format srt

# Live: transcribed as it arrives, one segment at a time
ffmpeg -f avfoundation -i ":0" -ar 16000 -ac 1 -f wav - 2>/dev/null \
  | onnx-genai transcribe models/whisper-tiny -

# Headerless PCM16 works too, e.g. from arecord
arecord -f S16_LE -r 16000 -c 1 -t raw \
  | onnx-genai transcribe models/whisper-tiny - --sample-rate 16000
```

A speech encoder consumes a bounded window, so long audio is cut into segments:
at a silence when there is one, and at the model's declared window otherwise.
Each segment is printed as soon as it is recognized, which is what makes the
live path usable — latency is one segment, not one recording. Silence between
segments is skipped rather than transcribed, and the timestamps reflect that.

`--format json` emits one object per segment (`index`, `start`, `end`, `text`);
`--format srt` emits subtitles. Diagnostics, including a real-time factor
showing whether the model keeps up with live audio, go to stderr so stdout stays
a clean transcript. Tune segmentation with `--segment-seconds`,
`--silence-seconds`, `--silence-threshold`, and `--min-segment-seconds`.

Whole-file transcription is also available over HTTP as
`POST /v1/audio/transcriptions`.

### Generate speech

`generate --output-audio` synthesizes through a text-to-speech package — an
autoregressive decoder that emits audio codes followed by a `run_on: final_only`
vocoder stage:

```bash
./target/release/onnx-genai generate models/my-tts \
  --prompt "Hello from onnx-genai." \
  --output-audio speech.wav
```

The output sample rate is declared by the package as `pipeline.audio.sample_rate`
rather than assumed; a package that omits it is rejected with an error rather
than played back at a guessed pitch (pass `--sample-rate` to supply one).

The native CPU decode path also enables a steady-state **decode-plan memo** by
default: it caches the per-step shape/buffer plan and replays it token-to-token
(token-exact by construction, with an in-flight verify net and a graceful fall
back to rebuilding on any model where invariance can't be proven). Disable it
with `ONNX_GENAI_DECODE_MEMO=0` (also `false`/`off`) on resource-constrained
hosts or for debugging; any other value, including unset, keeps it on.

### Run the OpenAI-compatible server

```bash
./target/release/onnx-genai-server \
  --model models/qwen2.5-0.5b \
  --model-id qwen2.5-0.5b \
  --addr 127.0.0.1:8080
```

Available routes are `GET /health`, `GET /v1/models`,
`POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings`,
`POST /v1/audio/transcriptions`, `POST /v1/audio/speech`,
`POST /v1/images/generations`, `POST /v1/sessions`, and
`DELETE /v1/sessions/{id}`. Pass a session id as `X-Session-Id` on chat
requests to reuse persistent context.

#### Multimodal chat

`POST /v1/chat/completions` accepts OpenAI content parts. Images are sent as
`image_url` (a `data:` URI or an `http(s)` URL; `detail` is accepted and
ignored, since resizing and tiling are declared by the package's
`preprocessing.image` program). Audio is sent as `input_audio` with base64
PCM16 WAV, and is transcribed.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my-vlm",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }],
    "max_tokens": 64
  }'
```

A model that declares no image or audio contract rejects the part with a 400
naming what it does accept, and unknown part types are named in the error along
with the supported set.

#### Image generation

`POST /v1/images/generations` renders through a diffusion package's own declared
denoise loop. Only `b64_json` is returned — the server stores nothing, so it has
no URL to hand back. `negative_prompt`, `steps`, `guidance_scale`, and `seed`
are onnx-genai extensions; omitted sampling values fall back to the package's
declared `num_steps` / `guidance_scale`.

```bash
curl http://127.0.0.1:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "stable-diffusion-1.5",
    "prompt": "an astronaut riding a horse",
    "negative_prompt": "blurry, low quality",
    "size": "512x512",
    "n": 1,
    "steps": 25,
    "guidance_scale": 7.5,
    "seed": 0
  }'
```

#### Speech synthesis

`POST /v1/audio/speech` returns the audio bytes directly, as OpenAI does.
`max_tokens`, `temperature`, `seed`, and `sample_rate` are onnx-genai
extensions. Only `wav` and `pcm` are offered: a compressed format is refused
rather than silently substituted, so a client that asked for MP3 never receives
WAV under an MP3 content type.

```bash
curl http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "my-tts", "input": "Hello from onnx-genai.", "response_format": "wav"}' \
  --output speech.wav
```

Chat with constrained JSON output (`"stream": true` enables SSE):

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "Reply with a JSON greeting."}],
    "response_format": {"type": "json_object"},
    "temperature": 0,
    "max_tokens": 64,
    "stream": false
  }'
```

Tool use, with a grammar-enforced required function call:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "What is the weather in Seattle?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"],
          "additionalProperties": false
        }
      }
    }],
    "tool_choice": "required",
    "temperature": 0,
    "max_tokens": 128
  }'
```

The server returns parsed calls in OpenAI `message.tool_calls`; the client is
responsible for executing tools and sending the result in a later message with
role `tool`.

Any OpenAI-compatible agent can use `http://127.0.0.1:8080/v1` as its base
URL and `qwen2.5-0.5b` as the model. The repository also includes a constrained
Hermes-style demonstration harness:

```bash
python3 scripts/coding_agent.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model qwen2.5-0.5b \
  --workdir target/coding-agent-workspace \
  --clean \
  --task "Create hello.py, run it, and report the output."
```

### Execution provider

```bash
ONNX_GENAI_EP=cpu ./target/release/onnx-genai-server --model models/qwen2.5-0.5b
ONNX_GENAI_EP=webgpu ./target/release/onnx-genai-server --model models/qwen2.5-0.5b
ONNX_GENAI_EP=coreml ./target/release/onnx-genai-server --model models/qwen2.5-0.5b
```

#### Plugin execution providers (any ORT ≥ 1.22 EP)

Any ONNX Runtime execution-provider plugin shared library can be loaded
generically — the provider name is discovered from the plugin, never hardcoded.
For example, using the [`onnxruntime-ep-openvino`](https://pypi.org/project/onnxruntime-ep-openvino/)
pip package:

```bash
ONNX_GENAI_EP=plugin \
ONNX_GENAI_EP_LIBRARY=/path/to/onnxruntime_providers_openvino_plugin.dll \
ONNX_GENAI_EP_DEVICE=CPU \
ONNX_GENAI_EP_OPTIONS=num_streams=2 \
  ./target/release/onnx-genai-server --model models/qwen2.5-0.5b
```

- `ONNX_GENAI_EP_LIBRARY` (required): path to the plugin shared library.
- `ONNX_GENAI_EP_DEVICE` (optional): narrows a multi-device plugin to one
  hardware class — `CPU`, `GPU`, or `NPU` (ORT's generic device enum).
- `ONNX_GENAI_EP_OPTIONS` (optional): provider-defined `key=value,key=value`
  options passed straight through.
- `ONNX_GENAI_EP_NAME` (optional): registration handle; defaults to the library
  file name.

#### Multiple execution providers (priority list)

`ONNX_GENAI_EP` accepts a comma-separated **priority list** — ORT tries each
entry in order and falls back to later entries for nodes it cannot claim. Each
entry is a built-in (`cpu`, `webgpu`, `cuda`, `metal`, `coreml`), the bare
`plugin` token (configured through the scalar `ONNX_GENAI_EP_*` variables
above), or an **inline plugin** that carries its own library and options so
several distinct plugins can be composed at once:

```bash
# CUDA first, then two different plugins, then CPU as a final fallback.
ONNX_GENAI_EP='cuda,plugin:/path/openvino_plugin.dll|device=GPU|opt.num_streams=2,plugin:/path/other_ep.so,cpu' \
  ./target/release/onnx-genai-server --model models/qwen2.5-0.5b
```

Inline plugin syntax: `plugin:<library>[|name=<handle>][|device=<CPU|GPU|NPU>][|opt.<key>=<value>]...`.
No provider name is ever hardcoded — each plugin's concrete provider is still
discovered from the library at load time. Distinct plugins that share the same
library file name should each set an explicit `name=` to avoid a registration
handle collision.

## Security

The server defaults to `127.0.0.1:8080` and has **no built-in
authentication**. Do not bind it to a non-loopback address unless it is behind
an authenticated reverse proxy. Server caps limit requested output tokens and
resident sessions (`--max-output-tokens`, `--max-sessions`), session ids come
from the OS CSPRNG, context length is checked when declared by model metadata,
and automatically downloaded ONNX Runtime archives require pinned SHA-256
checksums. Tool execution is always the client's responsibility.

## Coverage

Run `scripts/coverage.sh` to install the required LLVM tools when missing and
print workspace coverage. The final `TOTAL` row is the overall percentage; the
preceding rows identify low-coverage source files. For annotated source, run
`scripts/coverage.sh --html --open` instead of the default summary output.

## Project Structure

```text
crates/
├── onnx-genai/            # Main library facade and CLI
├── onnx-genai-metadata/   # Inference metadata parser and validation
├── onnx-genai-kv/         # Paged, prefix, tiered, and quantized KV storage
├── onnx-genai-scheduler/  # Priority scheduling and preemption
├── onnx-genai-ort/        # ONNX Runtime, tokenizers, templates, decode sessions
├── onnx-genai-engine/     # Generation, constraints, speculation, pipelines
└── onnx-genai-server/     # OpenAI-compatible HTTP server
```

## Status

**Phases 1-3 are complete. Phase 4 is substantially complete.**

Completed work includes end-to-end ORT/tokenizer generation, the CLI and HTTP
server, multi-session/prefix reuse, paged/tiered/int8 KV management,
priority/preemption, draft-model and prompt-lookup speculation, structured
decoding, FIM, chat templates and tool use, multi-model/VLM pipeline execution,
static-cache O(1)-per-token long-context decode, and continuous batched serving.
The OpenAI tool loop has been verified end-to-end with a Hermes coding agent.

Remaining advanced work includes:

- MTP, Medusa/tree, EAGLE, and other DESIGN §27 speculative proposers.
- vLLM `speculators` discovery and compatibility from DESIGN §28.
- Stochastic/rejection-sampling speculative acceptance; current verification
  uses greedy target agreement.
- True paged-attention execution kernels; current paging manages KV storage.
- Automatic hardware-profile probing/matching beyond explicit EP selection and
  metadata capability validation.

See [docs/architecture/DESIGN.md](docs/architecture/DESIGN.md) for the design and roadmap.

## License

MIT
