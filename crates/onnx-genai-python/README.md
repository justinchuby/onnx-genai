# onnx-genai

Native text generation for Python.

`onnx-genai` exposes the high-level `Engine` / `GenerateResult` API also offered
by `nxrt.genai`. The shared engine compiles and selects Native execution by
default. ORT is optional and used only for reference tests.

```python
import onnx_genai

engine = onnx_genai.Engine.from_dir("path/to/model_dir")

result = engine.generate("Hello, world!", max_tokens=64, temperature=0.8)
print(result.text)

# Streaming
def on_token(text, token_id, finish_reason):
    print(text, end="", flush=True)

engine.generate_stream("Tell me a story", on_token, max_tokens=128)
```

The model directory must contain the ONNX graph(s), `tokenizer.json`, and model
metadata (`inference_metadata.yaml` or `genai_config.json`).

## Wheels

Distributed as stable-ABI (`abi3`) wheels tagged `cp310-abi3`, so a single wheel
per platform loads on standard CPython 3.10 and newer. Wheels are built on
CPython 3.12; free-threaded wheels require a separate `abi3t` build.
Native CPU execution requires no `onnxruntime` or MLX Python package and no
ORT shared library. Importing `onnx_genai` only records optional installed
library paths; it does not import or load those reference packages.

The `ort`, `ort-cuda`, and `mlx` extras install optional reference dependencies.
The compiled `ort-cuda` Cargo feature enables ORT CUDA reference support;
installing a provider does not change the Native engine default.

## Related packages

- **`nxrt`** — the low-level nxrt runtime (inference sessions, eager ops) plus
  `nxrt.genai`, the same generation API backed by the native runtime.
- **`onnx-genai-server`** — the `onnx-genai` command-line tool, including the
  OpenAI-compatible server (`onnx-genai serve`).

See <https://github.com/justinchuby/onnx-genai>.
