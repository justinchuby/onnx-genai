# onnx-genai

ONNX Runtime-backed text generation for Python.

`onnx-genai` exposes a small, high-level generation API implemented on
[ONNX Runtime](https://onnxruntime.ai/). It is the ONNX Runtime-compatible
counterpart of `nxrt.genai`: both packages present the **same** `Engine` /
`GenerateResult` API, but `onnx_genai` runs on ONNX Runtime while `nxrt.genai`
runs on the native nxrt runtime.

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
per platform loads on CPython 3.10 and newer. The bundled ONNX Runtime shared
library is vendored into the wheel, so no separate ONNX Runtime installation is
required.

## Related packages

- **`nxrt`** — the low-level nxrt runtime (inference sessions, eager ops) plus
  `nxrt.genai`, the same generation API backed by the native runtime.
- **`onnx-genai-server`** — the `onnx-genai` command-line tool, including the
  OpenAI-compatible server (`onnx-genai serve`).

See <https://github.com/justinchuby/onnx-genai>.
