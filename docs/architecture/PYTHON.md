# Python API (`nxrt`)

`nxrt` is the PyO3/abi3 Python package for:

- ONNX model inference with `InferenceSession`;
- single-operator eager execution with `nxrt.eager`;
- local text generation with `nxrt.genai`.

The package does not include the HTTP webserver.

## Installation

Published wheels require Python 3.10 or newer:

```bash
python -m pip install nxrt
```

For a local development install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "maturin>=1.7,<2" numpy
# Linux only: maturin needs patchelf to repair the development wheel.
python -m pip install patchelf
cd crates/onnx-runtime-python
maturin develop --release
```

`maturin develop` requires an active virtual environment. On Linux, install
`patchelf` as shown above; macOS and Windows do not need it.

The shipped/default build enables both `eager` and `genai`. Developers can build
a smaller package with Cargo features, for example:

```bash
maturin develop --no-default-features
maturin develop --no-default-features --features eager
maturin develop --no-default-features --features genai
```

## InferenceSession

`InferenceSession` mirrors the common `onnxruntime.InferenceSession` workflow:

```python
import numpy as np
import nxrt

session = nxrt.InferenceSession("model.onnx")
inputs = {"x": np.ones((1, 3), dtype=np.float32)}
(output,) = session.run(None, inputs)
print(output)
```

Paths, `os.PathLike` values, and serialized ONNX model bytes are accepted.
`run()` returns copied NumPy arrays; `run_with_values()` provides the DLPack
zero-copy output path.

## Eager operator dispatch

`nxrt.eager.dispatch()` executes one ONNX operator without constructing a
model:

```python
import numpy as np
import nxrt

a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([10.0, 20.0, 30.0], dtype=np.float32)

(result,) = nxrt.eager.dispatch("Add", [a, b])
print(result)  # [11. 22. 33.]
```

The full signature is:

```python
nxrt.eager.dispatch(
    op_type,
    inputs,
    attributes=None,
    *,
    domain="",
    opset=None,
) -> list[numpy.ndarray]
```

Attributes accept booleans, integers, floats, strings, bytes, and homogeneous
non-empty sequences of those scalar types:

```python
(result,) = nxrt.eager.dispatch(
    "Softmax",
    [np.array([[1.0, 2.0]], dtype=np.float32)],
    {"axis": -1},
)
```

`nxrt.eager.opset()` and `nxrt.eager.LATEST_ONNX_OPSET` report the default ONNX
opset. `nxrt.eager.cache_stats()` returns `entries`, `hits`, and `misses` for
the compiled-kernel cache. The current eager backend is CPU-only and currently
materializes one output, so multi-output operators remain deferred.

## Generative AI

Load a model directory containing compatible ONNX graph(s), tokenizer files,
and `inference_metadata.yaml` or `genai_config.json`:

```python
import nxrt

engine = nxrt.genai.Engine.from_dir("models/qwen2.5-0.5b")
result = engine.generate(
    "Write a short Rust hello-world program.",
    max_tokens=64,
    temperature=0.0,
)

print(result.text)
print(result.token_ids)
print(result.finish_reason)
```

The generation signature is:

```python
engine.generate(
    prompt,
    *,
    max_tokens=128,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
    seed=None,
    stop=None,
) -> nxrt.genai.GenerateResult
```

`stop` is a list of text stop sequences. `GenerateResult` exposes `text`,
`token_ids`, `finish_reason`, and `prefix_cache_hit_len`.

Tokenization uses the same tokenizer as generation:

```python
token_ids = engine.tokenize("Hello")
```

### Streaming

`generate_stream()` invokes a callback for each generated token. The callback
receives `(text, token_id, finish_reason)`; `finish_reason` is `None` until a
terminal token event:

```python
def on_token(text, token_id, finish_reason):
    print(text, end="", flush=True)

result = engine.generate_stream(
    "Once upon a time",
    on_token,
    max_tokens=64,
    temperature=0.8,
    top_p=0.95,
)
print("\n", result.finish_reason)
```

When generation stops because `max_tokens` was reached, there is no separate
terminal token, so the token callbacks retain `finish_reason=None`; the returned
`GenerateResult.finish_reason` is `"max_tokens"`. A future iterator-style API
may add a dedicated completion event.

`Engine` is safe to move between Python threads, but one instance is not
re-entrant. Concurrent calls, including calls made by a generation callback,
raise `RuntimeError` immediately. Serialize calls or use one `Engine` per
thread.
