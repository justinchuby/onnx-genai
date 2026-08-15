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
onnx-genai generate ./model --prompt "Hello"  # one-shot generation (-p is short for --prompt)
onnx-genai run ./model                       # interactive REPL
onnx-genai show ./model                       # resolved files + metadata
onnx-genai list --models-dir ./models         # list models
onnx-genai version                            # version + execution providers
```

`generate`, `run`, and `show` accept either a model directory or a config file
inside it (a file resolves to its parent directory).

### Interactive REPL controls

In `onnx-genai run`, press **Ctrl-C** while a response is generating to cancel
that turn and return to the prompt. At an idle prompt, press **Ctrl-D** or
**Ctrl-C** (or enter an empty line) to exit. A one-shot `onnx-genai generate`
run is also cancelled by **Ctrl-C** mid-generation.

When both stdin and stdout are terminals, the REPL uses a rich line editor:
arrow-key cursor movement, persistent history, bracketed paste, slash-command
completion, and multiline input with **Alt+Enter** (plain **Enter** submits).
Interactive terminal text generation shows compact per-turn stats by default;
pass `--no-stats` to hide them. Stats are enabled only when stdin and stdout are
terminals, and they are printed to stderr; piped stdout remains byte-stable
generated text. When stdin or stdout is piped, the REPL also keeps the original
plain `>>> ` line-loop behavior for scripts and tests.

### Generation budget and sampling

When `--max-new-tokens` is omitted, `generate` and `run` follow the model's
effective context window: the CLI uses the remaining context after the prompt, so
generation stops on EOS, a stop sequence, or the full context. Pass
`--max-new-tokens` to override exactly. If neither metadata nor the decode path
reveals a context limit, the CLI warns and uses a finite fallback instead of
risking an ORT out-of-bounds decode; fix that with `--max-context TOKENS` or by
declaring `model.max_sequence_length` in inference metadata.

The REPL recomputes that remaining-context ceiling every turn as conversation
history grows. `/stats` includes a terse context meter (`ctx used / max`).

Sampling flags (`--temperature` above 0, `--top-p`, or `--top-k`) switch from
greedy argmax to stochastic sampling. `--temperature 0` remains greedy. Use
`--greedy` or `--no-greedy` to make the mode explicit.

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

For the ONNX Runtime CPU backend, use `ONNX_GENAI_INTRA_OP_THREADS=N` to
override ORT intra-op parallelism. On Windows ARM64, the automatic ORT CPU
default matches onnxruntime-genai: half of logical CPUs, capped at 16. Other
platforms leave ORT's own default unchanged.

### Decode-plan memo (default on)

On the native CPU decode path, a steady-state decode-plan memo caches the
per-step shape/buffer plan and replays it token-to-token, which is token-exact
by construction (shape-only bookkeeping, an in-flight verify net, and a
graceful fall back to rebuilding every step on any model where invariance can't
be proven). It is **on by default**. To disable it — for example on a
resource-constrained host or when debugging — set `ONNX_GENAI_DECODE_MEMO=0`
(also accepts `false`/`off`, case-insensitive). Any other value, including
unset, keeps it on.

## Runtime selection

### Selecting the ONNX Runtime library

There are two separate resolution steps:

- **Build time (`ort-sys/build.rs`)** finds headers and a link/development
  install in this order: `ORT_LIB_DIR`, `ORT_ROOT`, `pkg-config`, then a
  SHA-256-pinned ORT 1.27.0 download into Cargo's `OUT_DIR`.
- **Run time** chooses the shared library the process actually loads in this
  order: `ONNX_GENAI_ORT_LIB` (exact file), `ONNX_GENAI_ORT_LIB_DIR`
  (containing directory), active `CONDA_PREFIX`, active `VIRTUAL_ENV`, the
  pinned `ort-sys` download under `target/.../ort-prebuilt/lib`, then the
  platform dynamic-loader default search path.

Native `cargo run` / binary builds can honor the runtime variables because
`ort-sys` resolves `OrtGetApiBase` with `libloading` before the first ORT API
call instead of link-loading `onnxruntime` into the final binary. If that ever
changes back to a normal import-library link, runtime variables cannot override
the OS loader; put the desired directory first in `PATH` (Windows),
`LD_LIBRARY_PATH` (Linux), or `DYLD_LIBRARY_PATH` (macOS) before process start.

Ask the tool what it actually loaded:

```bash
onnx-genai version
cargo run -p onnx-genai-cli --bin onnx-genai -- version
```

The output includes the ORT path, ORT version, API version, and why that library
was selected. Any explicit or auto-discovered selection also prints the same
line on first ORT use. API mismatches name the loaded library path, loaded API
version, required API version, and concrete fixes.

Examples only — adapt paths to your machine:

```powershell
# Windows PowerShell, conda or venv with a pip onnxruntime wheel.
$prefix = if ($env:CONDA_PREFIX) { $env:CONDA_PREFIX } else { $env:VIRTUAL_ENV }
$env:ONNX_GENAI_ORT_LIB_DIR = "$prefix\Lib\site-packages\onnxruntime\capi"
$env:PATH = "$env:ONNX_GENAI_ORT_LIB_DIR;$env:PATH"
onnx-genai version
```

```bash
# Linux/macOS shell, pip onnxruntime inside the active Python environment.
export ONNX_GENAI_ORT_LIB_DIR="$(python - <<'PY'
import pathlib, onnxruntime
print(pathlib.Path(onnxruntime.__file__).resolve().parent / "capi")
PY
)"
export LD_LIBRARY_PATH="$ONNX_GENAI_ORT_LIB_DIR:${LD_LIBRARY_PATH:-}"  # Linux
export DYLD_LIBRARY_PATH="$ONNX_GENAI_ORT_LIB_DIR:${DYLD_LIBRARY_PATH:-}"  # macOS
onnx-genai version
```

```bash
# System package or from-source ORT install.
export ONNX_GENAI_ORT_LIB=/opt/onnxruntime/lib/libonnxruntime.so.1
onnx-genai version
```

```bash
# From-source build using the pinned ort-sys download: leave ORT vars unset.
unset ONNX_GENAI_ORT_LIB ONNX_GENAI_ORT_LIB_DIR
cargo run -p onnx-genai-cli --bin onnx-genai -- version
```

CUDA has two independent switches:

1. Build-time features select which CUDA code is compiled in. `--features cuda`
   enables ONNX Runtime's built-in `CUDAExecutionProvider` path only. `--features
   native-cuda` enables the native backend plus the project's hand-written CUDA
   EP (`onnx-runtime-ep-cuda`).
2. Runtime settings select which path to use. `ONNX_GENAI_EP=cuda` asks the ORT
   session to use CUDA, while `--backend auto|ort|native` selects the decoder
   for `generate`, `transcribe`, and the starting decoder for `run`. Speech
   transcription uses the same autoregressive pipeline decoder as multimodal
   generation, so backend selection is meaningful there too. In the REPL,
   `/backend` can still reload the model and switch the decoder without
   restarting. Profile output, `/session`, bare `/backend`, and `/stats` report
   the resolved backend (`auto` is shown separately as the request when it
   resolved to `ort` or `native`).

CUDA failure modes are intentionally distinct:

- If CUDA support was not compiled into the ORT layer, `ONNX_GENAI_EP=cuda`
  fails session creation with a "CUDA support not compiled in" error; request
  `cpu` (or rebuild with `--features cuda` / `--features native-cuda`) instead.
- If CUDA support was compiled in but the provider is unavailable at runtime
  (for example, no loadable CUDA provider library, driver, or GPU),
  `ONNX_GENAI_EP=cuda` also fails session creation and tells you to request
  `ONNX_GENAI_EP=cpu` when CPU execution is intentional.
- When CUDA is compiled in and available for the ORT/native session but the
  native CUDA EP cannot claim every executable node, the native runtime falls
  back to its CPU EP. Set `ONNX_GENAI_REQUIRE_CUDA=1` to reject that node-level
  CPU fallback. On Apple Silicon, the `onnxruntime-ep-mlx` plugin is installed
  by default.

Windows PowerShell example for the native CUDA path:

```powershell
$env:ONNX_GENAI_EP = "cuda"
$env:ONNX_GENAI_REQUIRE_CUDA = "1"
cargo run --release -p onnx-genai-cli --features native-cuda --bin onnx-genai -- `
  generate .\path\to\model --prompt "Hello" --max-new-tokens 64 --backend native --profile
```

PowerShell native-vs-ORT decode comparison, with the same ORT execution provider
for both runs:

```powershell
$env:ONNX_GENAI_EP = "cuda"  # or "cpu"
cargo run --release -p onnx-genai-cli --features native-cuda --bin onnx-genai -- `
  generate .\path\to\model --prompt "Hello" --max-new-tokens 128 --backend native --profile
cargo run --release -p onnx-genai-cli --features native-cuda --bin onnx-genai -- `
  generate .\path\to\model --prompt "Hello" --max-new-tokens 128 --backend ort --profile
```

If `--backend native` is passed to a binary built without native decoder support,
the command fails and tells you to rebuild with `--features native-backend` (CPU
native/backend path) or `--features native-cuda` (native CUDA EP). It does not
fall back to ORT.

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

Model loading logs and default text-generation stats go to stderr; append
`2>/dev/null` when you only want the generated text. `generate --output-image`,
`generate --output-audio`, and `transcribe` do not print the compact token stats
line by default; use `--profile` for their timing and throughput diagnostics.

### Checking the REPL's live view

Interactive terminal sessions render the reply and live numbers together by
default. That path is only taken when stdin and stdout are real terminals, so it
cannot be seen through a pipe. Run the REPL directly in a terminal and type a
prompt:

```bash
cargo run -p onnx-genai-cli --bin onnx-genai -- run tests/fixtures/tiny-llm --max-new-tokens 5
```

```text
>>> hello
dogtok29over,dog
[ 7 in · 5 out · backend ort · 2.0 tok/s · ttft 1 ms · rss 50.6 MiB ]
```

Use `/stats` to toggle the compact line at runtime, or start `run` or text
`generate` with `--no-stats`. Piping a script into the REPL exercises the
byte-stable plain-text fallback instead, which is what the tests do:

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
>>> /session                # model, sampling, message counts, and token totals
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
