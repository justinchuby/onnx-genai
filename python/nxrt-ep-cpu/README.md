# nxrt-ep-cpu

A pip-installable **ONNX Runtime plugin execution provider (CPU)**. The wheel
bundles the compiled `onnx-runtime-ep-cpu-plugin` shared library
(`libonnx_runtime_ep_cpu_plugin.{so,dll,dylib}`) and exposes the absolute path
to it so ONNX Runtime can load it via
`RegisterExecutionProviderLibrary(registration_name, library_path)`.

The bundled library exports the ORT plugin-EP C ABI
(`CreateEpFactories` / `ReleaseEpFactory`). It is **not** a Python extension —
it is a plain C-ABI shared library that ONNX Runtime `dlopen`s.

## Install

```bash
pip install nxrt-ep-cpu
```

## Usage

```python
import nxrt_ep_cpu

# Absolute path to the bundled cdylib (guaranteed to exist).
path = nxrt_ep_cpu.get_library_path()

# Register with ONNX Runtime (requires `onnxruntime` to be installed).
import onnxruntime as ort
so = ort.SessionOptions()
nxrt_ep_cpu.register(so)          # thin wrapper over register_execution_provider_library
# ...or register manually with the path from get_library_path().
```

`register()` prefers `SessionOptions.register_execution_provider_library` when a
`SessionOptions` is passed, and falls back to the module-level
`onnxruntime.register_execution_provider_library`. If `onnxruntime` is not
installed it raises a clear `ImportError`.

## Platforms

Linux (x86_64, manylinux), Windows (AMD64, ARM64) and macOS (arm64). The wheel
is platform-specific — it contains a compiled shared library and is not
pure-Python.

## License

MIT. See the [onnx-genai](https://github.com/justinchuby/onnx-genai) repository.
