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
onnx-genai generate --model ./model "Hello"  # one-shot generation
onnx-genai run --model ./model               # interactive REPL
onnx-genai show ./model                       # resolved files + metadata
onnx-genai list --models-dir ./models         # list models
onnx-genai version                            # version + execution providers
```

`generate`, `run`, and `show` accept either a model directory or a config file
inside it (a file resolves to its parent directory).

## Runtime selection

Choose an execution provider at runtime with `ONNX_GENAI_EP` (e.g. `cpu`,
`cuda`). CUDA requires the `[cuda]` extra (or a separately installed
`onnxruntime-gpu`). On Apple Silicon, the `onnxruntime-ep-mlx` plugin is
installed by default.

Python 3.11+ is required (the `onnxruntime` dependency ships no earlier wheels).
