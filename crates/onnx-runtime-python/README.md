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
| `sess.run_with_values(output_names, input_feed) -> list[NxrtValue]` | Same args as `run`, but returns zero-copy `NxrtValue` wrappers (see below) instead of copied numpy arrays. |
| `sess.get_inputs()` / `sess.get_outputs()` | Lists of `NodeArg` with `.name`, `.type` (e.g. `"tensor(float)"`), `.shape`. |
| `sess.get_providers()` | The providers the session was created with. |
| `nxrt.get_available_providers()` | Providers this wheel can service. |
| `nxrt.__version__` | Package version. |

### Ergonomic API

`run()` keeps the exact onnxruntime signature (`run(output_names, input_feed)`),
so drop-in code and conformance suites are unchanged. For everyday use nxrt adds
a friendlier surface layered on the same zero-copy core.

**`nxrt.load(path, *, device=None, providers=None)`** — a friendly loader that
returns a **callable** `InferenceSession`. `device` is sugar for provider
selection (`"cpu"` default, `"cuda"`/`"cuda:N"`, `"metal"`); pass `providers=[…]`
for full control (it wins over `device`).

**`session(...)`** — call the session like a function instead of
`run(None, {...})`. Inputs resolve like a normal Python call:

```python
import numpy as np, nxrt

sess = nxrt.load("model.onnx")            # device="cuda" / providers=[...] optional

y = sess(x)                                # one input  -> one output, returned directly
s = sess(a, b)                             # positional, mapped to inputs by order
s = sess(A=a, B=b)                         # keyword, mapped to inputs by name
s = sess({"A": a, "B": b})                 # explicit name->array feed (dict/Mapping)
s = sess(a, B=b)                           # positional + keyword mix
```

Values may be numpy / torch / cupy / jax arrays (anything exposing `__dlpack__`
or `__array__`) — they flow through the zero-copy DLPack import, no wrapper
needed. Mistakes raise actionable errors (unknown input, duplicate feed, missing
input, too many positionals).

Outputs are shaped for convenience:

- **one output** → the `NxrtValue` itself (not a list);
- **multiple outputs** → an `Outputs` container supporting `out[0]`,
  `out["logits"]`, `out.logits`, `len(out)`, unpacking (`a, b = sess(x)`), and
  `.keys()`/`.values()`/`.items()`.

`NxrtValue` behaves like a tensor: `np.asarray(v)` (via `__array__`), `v.shape`,
`v.dtype`, `len(v)`, plus the zero-copy `torch.from_dlpack(v)` / `v.numpy()`.

**`session.bind_outputs("logits", ...)`** — returns a `BoundSession` proxy whose
calls return only the selected outputs. It is a convenience filter; inference
still computes all graph outputs. `run()`/`run_with_values()` on the base session
are unaffected.

```python
with sess.bind_outputs("logits") as bound:
    logits = bound(x)                      # only "logits" is returned
```

`session.input_names` / `session.output_names` give the input/output names in
graph order (alongside the existing `get_inputs()`/`get_outputs()`).

### Zero-copy output via DLPack

`run()` keeps its onnxruntime-compatible contract: it returns freshly-copied
`numpy.ndarray`s. For **zero-copy** access, `run_with_values()` returns
`NxrtValue` objects that implement the [Array-API DLPack producer
protocol](https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html)
(`__dlpack__` / `__dlpack_device__`), so a consumer *borrows* nxrt's output
buffer instead of copying it:

```python
(v,) = sess.run_with_values(None, {"x": x})
a = np.from_dlpack(v)       # numpy ≥ 2.1: writable, shares nxrt's buffer
t = torch.from_dlpack(v)    # torch: same physical memory (t.data_ptr() == a.ctypes.data)
host = v.numpy()            # still available: an owned copy, like run()
```

nxrt emits the versioned `DLManagedTensorVersioned` (`"dltensor_versioned"`
capsule, writable flag) when the consumer advertises DLPack major ≥ 1, and the
unversioned `DLManagedTensor` (`"dltensor"`) otherwise. The capsule's `deleter`
owns a reference to the backing buffer, so an imported array stays valid even
after the `NxrtValue` is dropped. bf16 exports as DLPack `kDLBfloat/16`
(consumable by torch; numpy has no bf16 DLPack import).

#### CUDA (`kDLCUDA`) zero-copy — import & export

When nxrt is built with the `cuda` feature and runs on a
`CUDAExecutionProvider`, the DLPack path extends to GPU memory with **no host
round-trip**:

```python
sess = nxrt.InferenceSession(model, providers=["CUDAExecutionProvider"])

# Import: a CUDA torch tensor is borrowed directly as a device-resident input.
x = torch.rand(1024, device="cuda")
(out,) = sess.run(None, {"x": x})          # runs on the GPU, no H2D/D2H copy

# Export: a CUDA nxrt output is borrowed by torch on the same device.
(v,) = sess.run_with_values(None, {"x": x})
t = torch.from_dlpack(v)                    # t.is_cuda; t.data_ptr() aliases nxrt's buffer
```

* `__dlpack_device__()` returns `(kDLCUDA, ordinal)` for a CUDA output, so the
  consumer allocates/borrows on the correct GPU.
* **Stream contract.** DLPack requires the producer to order the exported
  buffer's device work with respect to the consumer's stream before handoff. On
  **export**, nxrt takes the conservative, always-correct end of the handshake:
  it **fully synchronizes the producing EP's stream** (`Tensor::sync`) before
  returning the capsule, so the data is valid regardless of which stream the
  consumer reads on. A consumer `stream` of `-1` (the Array-API "no
  synchronization" sentinel) is honored and skips the sync; any other value
  (`None`, `1`, `2`, or a real handle) triggers it. On **import**, nxrt passes
  the legacy-default CUDA stream (`1`) to the producer's `__dlpack__(stream=...)`
  so the producer orders its work before nxrt reads on its default stream.
* Contiguity, dtype and device-ordinal are validated exactly as on the CPU path;
  any unsupported case falls back to a copy (or, for a real CUDA tensor a host
  copy cannot service, surfaces the producer's own error) rather than silently
  aliasing device memory as host memory.

A CPU wheel (built without the `cuda` feature) has no `CUDAExecutionProvider`
and treats a CUDA input as a copy-fallback; the GPU tests
(`tests/test_dlpack_gpu.py`) skip unless both `torch.cuda.is_available()` and
`CUDAExecutionProvider` are present.

### Providers

The default is `["CPUExecutionProvider"]`, always available (pure-Rust,
offline). `"CUDAExecutionProvider"` is available **only** when the crate is
built with the `cuda` Cargo feature; requesting an unknown or unbuilt provider
raises a `ValueError` that lists what this build supports.

### GPU tracing (CUPTI) in the CUDA wheel

GPU kernel tracing is enabled by default when the wheel is built with
`maturin build --release --features cuda`; the CPU wheel keeps the tracer and
CUPTI loader disabled. Install the CUDA runtime extra with `pip install
nxrt[cuda]`, or install `nvidia-cuda-cupti-cu13` alongside a locally built CUDA
wheel. The loader checks both normal system library paths and the package's
`site-packages/nvidia/cuda_cupti/lib` directory. CUDA builds also expose
`nxrt.cupti_available() -> bool`.

The NVIDIA driver (`libcuda.so.1`) is still required to capture GPU activity,
and CUPTI's major version must match the CUDA 13 build. A missing driver,
missing CUPTI library, or version mismatch makes `cupti_available()` return
`False` and tracing skip gracefully; it never prevents `import nxrt`.

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

Ops nxrt's CPU EP implements can be compared with
`onnx.reference.ReferenceEvaluator`; unsupported ops fail with an
**actionable** nxrt error naming the operator. This crate also ships a focused,
self-contained slice in
`tests/test_conformance.py` (runs a handful of supported ops through the same
generators) and API/error-quality tests in `tests/test_api.py`:

```bash
maturin develop
pytest crates/onnx-runtime-python/tests/ -q
```

24 tests pass standalone (offline); the conformance suite (~10 property-based op
tests) additionally runs when `onnx-tests` is installed, for 34 total.

The full upstream matrix has also been run. See
[`conformance/README.md`](conformance/README.md) and
[`docs/EP_CONFORMANCE.md`](../../docs/EP_CONFORMANCE.md) for reproducible
commands and the current CPU pass/fail/skip results.
