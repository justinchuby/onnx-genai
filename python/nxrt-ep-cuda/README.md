# nxrt-ep-cuda (EXPERIMENTAL / PRE-RELEASE)

> ⚠️ **Experimental.** The CUDA execution provider bundled here has been
> validated on physical CUDA hardware (NVIDIA H200, driver 580.105.08,
> CUDA 13.0): it loads and runs the Muse-Glimmer-30B (int4) decoder end-to-end
> with **zero CPU fallbacks**, and the bundled plugin `.so` registers and
> executes ONNX graphs through ONNX Runtime on-device. It remains
> **pre-release**: APIs and packaging may change without notice, and it is not
> yet recommended for production.

A pip-installable **ONNX Runtime plugin execution provider (CUDA 13)**. The
wheel bundles the compiled `onnx-runtime-ep-cuda-plugin` shared library
(`libonnx_runtime_ep_cuda_plugin.{so,dll}`) built with the `cuda` cargo feature
and exposes the absolute path to it so ONNX Runtime can load it via
`RegisterExecutionProviderLibrary(registration_name, library_path)`.

The bundled library exports the ORT plugin-EP C ABI
(`CreateEpFactories` / `ReleaseEpFactory`). It is **not** a Python extension.

## Requirements

- **CUDA 13** runtime. The wheel declares the NVIDIA runtime libraries as
  dependencies (`nvidia-cuda-runtime>=13`, `nvidia-cublas>=13`,
  `nvidia-cuda-nvrtc>=13`, `nvidia-cuda-cupti>=13`), so they are installed
  automatically. The NVIDIA **driver** (`libcuda.so.1`) remains a host
  prerequisite.
- Linux (x86_64) and Windows (AMD64) only. There is no macOS build.

## Install

```bash
pip install nxrt-ep-cuda   # pre-release; may require --pre
```

## Usage

```python
import nxrt_ep_cuda

path = nxrt_ep_cuda.get_library_path()   # absolute path to the bundled cdylib

import onnxruntime as ort
so = ort.SessionOptions()
nxrt_ep_cuda.register(so)                # thin wrapper over register_execution_provider_library
```

`register()` prefers `SessionOptions.register_execution_provider_library` when a
`SessionOptions` is passed, and falls back to the module-level
`onnxruntime.register_execution_provider_library`. If `onnxruntime` is not
installed it raises a clear `ImportError`.

## License

MIT. See the [onnx-genai](https://github.com/justinchuby/onnx-genai) repository.
