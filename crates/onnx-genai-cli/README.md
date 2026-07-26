# onnx-genai-server

The `onnx-genai` command-line tool and OpenAI-compatible server, backed by
[ONNX Runtime](https://onnxruntime.ai/).

```bash
pip install onnx-genai-server            # CPU (all platforms; MLX on macOS)
pip install onnx-genai-server[cuda]      # + CUDA 13 / cuDNN 9 (Windows/Linux)
```

ONNX Runtime is **not** bundled. The command loads `libonnxruntime` from the
installed `onnxruntime` (CPU) or `onnxruntime-gpu` (CUDA) wheel at startup, so it
uses whichever execution providers you installed.

## Commands

```bash
onnx-genai serve --models-dir ./models       # OpenAI-compatible HTTP server
onnx-genai generate ./model --prompt "Hello"  # one-shot generation
onnx-genai run ./model                       # interactive REPL
onnx-genai show ./model                       # resolved files + metadata
onnx-genai list --models-dir ./models         # list models
onnx-genai version                            # version + execution providers
```

`generate`, `run`, and `show` accept either a model directory or a config file
inside it (a file resolves to its parent directory).

### Polite CPU decode

Use `--cpu-cores N` with `generate` or `run` to cap native CPU decode to N
persistent workers:

```bash
onnx-genai generate ./model --prompt "Hello" --cpu-cores 8
```

Where thread affinity is supported, decode workers are pinned to at most N CPUs
from the process's allowed CPU set, leaving the rest of a shared machine
available to other programs. The equivalent environment setting is
`ONNX_GENAI_CPU_DECODE_THREADS=N`. Precedence is explicit `--cpu-cores` >
environment variable > automatic sizing. With neither setting, the
peak-throughput automatic default is unchanged. This controls native decode
workers; use an OS cpuset/taskset as well when the entire process, including
prefill or ONNX Runtime work, must be hard-confined.

## Runtime selection

Choose an execution provider at runtime with `ONNX_GENAI_EP` (e.g. `cpu`,
`cuda`). CUDA requires the `[cuda]` extra (or a separately installed
`onnxruntime-gpu`). On Apple Silicon, the `onnxruntime-ep-mlx` plugin is
installed by default.

Python 3.11+ is required (the `onnxruntime` dependency ships no earlier wheels).

## Running from source

The binary is `onnx-genai` in the `onnx-genai-cli` package, so everything above
works under `cargo run` with a `--` separating cargo's flags from the tool's:

```bash
cargo run -p onnx-genai-cli --bin onnx-genai -- version
```

Add `--release` for anything where speed matters — a debug build decodes orders
of magnitude slower, which makes `--profile` numbers meaningless:

```bash
cargo run --release -p onnx-genai-cli --bin onnx-genai -- \
  generate ./model --prompt "Hello" --max-new-tokens 64
```

### Trying it without downloading a model

The repository's test fixtures are tiny hand-built ONNX models. They generate
nonsense, but they exercise every code path end to end and need no network, so
they are the fastest way to check that a change works. Run these from the
repository root:

```bash
# Text generation, and the interactive REPL
cargo run -p onnx-genai-cli --bin onnx-genai -- \
  generate tests/fixtures/tiny-llm --prompt "hello" --max-new-tokens 5 --raw
cargo run -p onnx-genai-cli --bin onnx-genai -- run tests/fixtures/tiny-llm

# Image input (the prompt needs one <image> placeholder per image)
cargo run -p onnx-genai-cli --bin onnx-genai -- \
  generate tests/fixtures/tiny-vlm-image-input \
  --image path/to/any.png --prompt "describe <image>" --max-new-tokens 3 --raw

# Image generation. This fixture's VAE is 8x8, hence the explicit size
cargo run -p onnx-genai-cli --bin onnx-genai -- \
  generate tests/fixtures/tiny-txt2img \
  --prompt "a cat" --output-image out.png --steps 2 --width 8 --height 8

# Speech synthesis, then transcription of what it produced
cargo run -p onnx-genai-cli --bin onnx-genai -- \
  generate tests/fixtures/tiny-tts --prompt "hello" --output-audio out.wav
cargo run -p onnx-genai-cli --bin onnx-genai -- \
  transcribe tests/fixtures/tiny-whisper out.wav

# The OpenAI-compatible server
cargo run -p onnx-genai-cli --bin onnx-genai -- \
  serve --model tests/fixtures/tiny-llm --addr 127.0.0.1:8123
curl localhost:8123/v1/models
```

Model loading logs to stderr; append `2>/dev/null` when you only want the
generated text.

### Checking the REPL's live view

`/stats` renders the reply and its live numbers together, and that path is only
taken when stdout is a real terminal. It therefore cannot be seen through a pipe
— run the REPL directly in a terminal and type `/stats`, then a prompt:

```bash
cargo run -p onnx-genai-cli --bin onnx-genai -- run tests/fixtures/tiny-llm --max-new-tokens 5
```

```text
>>> /stats
per-turn stats enabled
>>> hello
dogtok29over,dog
[ 7 in · 5 out · 2.0 tok/s · ttft 1 ms · rss 50.6 MiB ]
```

Piping a script into the REPL exercises the plain-text fallback instead, which is
what the tests do:

```bash
printf '/stats\nhello\n\n' | cargo run -p onnx-genai-cli --bin onnx-genai -- \
  run tests/fixtures/tiny-llm --max-new-tokens 5
```

### Changing the session without restarting

The REPL can reload the model under a different execution provider or decode
backend, and switch models outright:

```text
>>> /ep                     # current provider, and what this build can select
>>> /ep cpu
>>> /backend ort            # auto | ort | native
>>> /model ./another-model
>>> /profile on             # report timings, memory, and cache reuse per turn
```

On Apple Silicon the MLX/Metal execution provider is offered — and
auto-selected — when its plugin library is configured, which the Python packages
do for you. From a source build, point at it yourself:

```bash
ONNX_GENAI_METAL_EP_LIB=$(python -c 'import onnxruntime_mlx, os;
print(os.path.join(os.path.dirname(onnxruntime_mlx.__file__), "libonnxruntime_mlx_ep.dylib"))') cargo run -p onnx-genai-cli --bin onnx-genai -- run ./model
```

`/ep` then lists `metal` and `--profile` reports it. Selection is non-strict: if
the plugin fails to load, the session falls back to CPU rather than failing.

All but `/profile` reload the model and clear the conversation. A selection that
fails to load is reported and the previous session keeps running, so an
unavailable provider does not end the session.

### Seeing what the KV cache holds

`/pages` shows the page pool as it stands, which is the view you want when a
pool is filling up and the question is whether that is one long conversation or
many that should be sharing:

```text
kv pages   ▓▓▓▓████················  38%
           38 of 100 pages held · 16 tokens/page
           512 of 608 token slots used (84% of held pages)
           9 shared (23%) — pages more than one sequence or cached prefix holds

references per page
    1x      29 pages
    2x       7 pages
    3x       2 pages

live sequences
  sequence     pages   tokens   shared
  7               12      190        5
```

Shared pages are drawn first (`▓`) because they are the ones that would
otherwise have been duplicated. A model whose KV is not paged says so, rather
than reporting an empty pool.

### Tests

Always name packages explicitly. A bare `--workspace` pulls in `mlas-sys`, whose
build script is x86-64-Linux-only, and `onnx-runtime-cpuinfo`, which needs its
git submodule checked out:

```bash
cargo test -p onnx-genai-cli
cargo test -p onnx-genai-cli --test repl_e2e
```
