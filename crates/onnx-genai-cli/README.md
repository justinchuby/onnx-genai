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
onnx-genai serve ./model                # OpenAI-compatible HTTP server
onnx-genai serve --models-dir ./models   # ... or serve every model under a directory
onnx-genai generate ./model "Hello"      # one-shot generation
onnx-genai show ./model                  # resolved files + metadata
onnx-genai list ./models                 # list models
onnx-genai version                       # version + execution providers
```

`generate` and `show` accept either a model directory or a config file
inside it (a file resolves to its parent directory).

For `generate`, `MODEL PROMPT` is the canonical invocation. `--prompt PROMPT`
is also supported; supply exactly one prompt, not both forms.

### Interactive chat

Run `onnx-genai serve ./model` and open **http://127.0.0.1:8080/**.
The server embeds the official Llama Chat frontend; the terminal `run` command
and its REPL have been removed. No frontend build or Node.js is needed at runtime.
See [protocol compatibility](../../docs/LLAMA_CPP_COMPATIBILITY.md).
One-shot `generate` still supports Ctrl-C cancellation, `--stream`, compact
terminal stats, and profiling.

### Generation budget and sampling

When `--max-new-tokens` is omitted, `generate` follows the model's
effective context window: the CLI uses the remaining context after the prompt, so
generation stops on EOS, a stop sequence, or the full context. Pass
`--max-new-tokens` to override exactly. If neither metadata nor the decode path
reveals a context limit, the CLI warns and uses a finite fallback instead of
risking an ORT out-of-bounds decode; fix that with `--max-context TOKENS` or by
declaring `model.max_sequence_length` in inference metadata.

Sampling flags (`--temperature` above 0, `--top-p`, or `--top-k`) switch from
greedy argmax to stochastic sampling. `--temperature 0` remains greedy. Use
`--greedy` or `--no-greedy` to make the mode explicit.

### Polite CPU decode

Use `--cpu-cores N` with `generate` to cap native CPU decode to N
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
  SHA-256-pinned ORT 1.29.0 download into Cargo's `OUT_DIR`.
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
   for `generate` and `transcribe`. Speech transcription uses the same
   autoregressive pipeline decoder as multimodal generation. Profile output
   reports the resolved backend.

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
# Text generation and browser chat
cargo run -p onnx-genai-cli --bin onnx-genai -- \
  generate tests/fixtures/tiny-llm --prompt "hello" --max-new-tokens 5 --raw
cargo run -p onnx-genai-cli --bin onnx-genai -- serve tests/fixtures/tiny-llm

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

### Tests

Always name packages explicitly. A bare `--workspace` pulls in `mlas-sys`, whose
build script is x86-64-Linux-only, and `onnx-runtime-cpuinfo`, which needs its
git submodule checked out:

```bash
cargo test -p onnx-genai-cli
cargo test -p onnx-genai-cli --test prompt_e2e
```
