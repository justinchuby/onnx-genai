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

Choose an execution provider at runtime with `ONNX_GENAI_EP` (e.g. `cpu`,
`cuda`). CUDA requires the `[cuda]` extra (or a separately installed
`onnxruntime-gpu`). On Apple Silicon, the `onnxruntime-ep-mlx` plugin is
installed by default.

Python 3.11+ is required (the `onnxruntime` dependency ships no earlier wheels).
