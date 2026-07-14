# nxrt — Python binding for the nxrt ONNX runtime

`nxrt` exposes the pure-Rust **nxrt** ONNX runtime to Python with an API that
mirrors `onnxruntime.InferenceSession`, so existing code and conformance suites
(e.g. [`cbourjau/onnx-tests`](https://github.com/cbourjau/onnx-tests)) can drive
nxrt's execution providers with a one-line runtime swap and plain `pytest`.

```python
import numpy as np
import nxrt

sess = nxrt.InferenceSession("model.onnx")          # path, os.PathLike, or bytes
outs = sess.run(None, {"x": np.ones((1, 3), np.float32)})
print(nxrt.__version__, nxrt.get_available_providers())
```

- **Minimum Python: 3.10.**
- Built on [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs) against
  CPython's stable ABI.

## Install

```bash
pip install nxrt                       # from a published wheel
```

Local development build (installs into the active venv/conda env):

```bash
# from crates/onnx-runtime-python
maturin develop            # debug build + install
maturin develop --release  # optimized
```

Or build a wheel and install it:

```bash
maturin build --release            # writes target/wheels/nxrt-*.whl
pip install target/wheels/nxrt-*.whl
```

`ml_dtypes` is an optional runtime dependency, needed only to return `bfloat16`
outputs as numpy arrays: `pip install nxrt[bfloat16]`.

## API

| Symbol | Description |
|---|---|
| `nxrt.InferenceSession(path_or_bytes, providers=["CPUExecutionProvider"])` | Load a model from a path/`os.PathLike` or raw model `bytes`. |
| `sess.run(output_names, input_feed) -> list[np.ndarray]` | `output_names` is a list of names or `None` (all outputs). `input_feed` is `{name: np.ndarray}`. |
| `sess.get_inputs()` / `sess.get_outputs()` | Lists of `NodeArg` with `.name`, `.type` (e.g. `"tensor(float)"`), `.shape`. |
| `sess.get_providers()` | The providers the session was created with. |
| `nxrt.get_available_providers()` | Providers this wheel can service. |
| `nxrt.__version__` | Package version. |

### Providers

The default is `["CPUExecutionProvider"]`, always available (pure-Rust,
offline). `"CUDAExecutionProvider"` is available **only** when the crate is
built with the `cuda` Cargo feature; requesting an unknown or unbuilt provider
raises a `ValueError` that lists what this build supports.

### Supported dtypes

numpy ⇆ nxrt covers `bool`, `int8/16/32/64`, `uint8/16/32/64`, `float16`,
`float32`, `float64`, and `bfloat16` (bf16/f16 are carried as opaque 2-byte
storage, never reinterpreted through `float32`). Any other numpy dtype (e.g.
`complex64`, `float8`) raises an actionable `TypeError` telling you exactly which
dtype was rejected and how to cast around it — see `RULES.md` §1.

Arrays are copied to C-contiguous, little-endian layout automatically
(`numpy.ascontiguousarray`), so non-contiguous views are accepted.

## Wheels: `abi3` vs `abi3t`

Two distinct wheels ship for two distinct CPython ABIs (see
`docs/PIPELINE.md` §12.4):

| Wheel | ABI | Tag | Floor | Interpreter |
|---|---|---|---|---|
| standard | `abi3` (stable ABI) | `cp310-abi3` | `Py_LIMITED_API` = 3.10 | CPython 3.10–3.14+ (GIL) |
| free-threaded | `abi3t` | `cp315t` | free-threaded 3.15 | free-threaded CPython |

- **Standard `abi3`** — a **single** wheel tagged `cp310-abi3` (the `cp310` is
  the *minimum* interpreter it loads on, not the one that built it). It is
  compiled against `Py_LIMITED_API = 0x030A0000` (PyO3's `abi3-py310` feature,
  configured in `pyproject.toml`/`Cargo.toml`) and can be built with any
  interpreter ≥ 3.10:

  ```bash
  maturin build --release --interpreter python3.12   # still tagged cp310-abi3
  ```

- **Free-threaded `abi3t`** — the no-GIL (`Py_GIL_DISABLED`) interpreter is
  **not** ABI-compatible with the standard `abi3` wheel, so a separate wheel is
  published (tagged e.g. `cp315t`). It must be built **with a free-threaded
  interpreter**; PyO3 does not yet emit a limited-API/`abi3`-style wheel that is
  simultaneously free-threaded, so this build drops the `abi3-py310` feature and
  targets the free-threaded interpreter directly:

  ```bash
  # requires a free-threaded CPython (e.g. python3.15t / a 3.13t build)
  maturin build --release --interpreter python3.15t
  ```

  The Rust source is already free-threaded-safe: `InferenceSession` guards its
  mutable runtime state with a `Mutex`, so concurrent `run` calls from multiple
  threads are sound without the GIL. **Status: configured, not executed here** —
  the offline build environment only ships CPython 3.12, so the `abi3t` wheel is
  documented and buildable but has not been produced/tested in this iteration.
  PyO3 caveat: free-threaded support is stabilizing; pin a PyO3 ≥ 0.23 that
  advertises free-threaded support and expect the wheel tag to track the
  free-threaded interpreter you build against.

## Running the onnx-tests conformance suite against nxrt

[`cbourjau/onnx-tests`](https://github.com/cbourjau/onnx-tests) selects the
runtime under test through the `RUN_CANDIDATE` environment variable — a function
`(onnx.ModelProto) -> dict[str, np.ndarray]`. This crate ships that adapter as
`tests/nxrt_runtime.py::run_nxrt`:

```bash
git clone https://github.com/cbourjau/onnx-tests
cd onnx-tests
pip install -e .            # + hypothesis, spox

# point the suite at nxrt (adapter is in this crate's tests/ dir)
export PYTHONPATH=/path/to/onnx-genai/crates/onnx-runtime-python/tests
RUN_CANDIDATE=nxrt_runtime.run_nxrt pytest tests/ -q
```

Ops nxrt's CPU EP implements pass against `onnx.reference.ReferenceEvaluator`;
ops it does not yet implement fail with an **actionable** nxrt error naming the
operator, so the suite report cleanly separates "wrong answer" from "not yet
implemented". This crate also ships a focused, self-contained slice in
`tests/test_conformance.py` (runs a handful of supported ops through the same
generators) and API/error-quality tests in `tests/test_api.py`:

```bash
maturin develop
pytest crates/onnx-runtime-python/tests/ -q
```

Full-matrix conformance over the entire ONNX opset is the documented scale-up
step (`RUN_CANDIDATE=nxrt_runtime.run_nxrt` over the whole `onnx-tests/tests`
tree), gated on the CPU EP's op coverage growing.
