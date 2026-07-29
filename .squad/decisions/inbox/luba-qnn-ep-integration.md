### 2026-07-28: Drive Qualcomm QNN from the native runtime through ORT plugin EP
**By:** Luba
**What:** Acquisition/inspection/design for loading the prebuilt Qualcomm QNN ORT plugin EP from the Python wheel and driving it through onnx-genai's existing plugin-EP path.
**Why:** Justin confirmed we should not build QNN from source; the QNN EP ships as a prebuilt ORT plugin library in a wheel and only needs native runtime registration/options wiring.

## 1. Acquisition and binary inventory

Requested command:

```powershell
python -m pip download onnxruntime-ep-qnn --dest C:\Users\justinchu\dev\qnn-ep-wheel --only-binary=:all:
```

Result:

```text
ERROR: Could not find a version that satisfies the requirement onnxruntime-ep-qnn (from versions: none)
ERROR: No matching distribution found for onnxruntime-ep-qnn
```

The win-arm64-targeted retry also failed:

```powershell
python -m pip download onnxruntime-ep-qnn --dest C:\Users\justinchu\dev\qnn-ep-wheel --only-binary=:all: --platform win_arm64 --python-version 312 --implementation cp --abi cp312
```

with the same `No matching distribution found` error. PyPI JSON for `onnxruntime-ep-qnn` returned 404.

The available PyPI package is `onnxruntime-qnn`, whose metadata describes it as "ONNX Runtime QNN is a plugin execution provider". Download command used, without installing into the active environment:

```powershell
python -m pip download onnxruntime-qnn --dest C:\Users\justinchu\dev\qnn-ep-wheel --only-binary=:all:
```

Downloaded:

- `C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64.whl`
- `C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime-1.28.0-cp312-cp312-win_arm64.whl`
- dependency wheels: `numpy-2.5.1-cp312-cp312-win_arm64.whl`, `protobuf-7.35.1-py3-none-any.whl`, `coloredlogs-15.0.1-py2.py3-none-any.whl`, `flatbuffers-25.12.19-py2.py3-none-any.whl`, `packaging-26.2-py3-none-any.whl`, `sympy-1.14.0-py3-none-any.whl`, `humanfriendly-10.0-py2.py3-none-any.whl`, `mpmath-1.3.0-py3-none-any.whl`, `pyreadline3-3.5.6-py3-none-any.whl`.

Extracted QNN wheel:

```text
C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64
```

QNN package metadata:

- Package: `onnxruntime-qnn`
- Version: `2.4.0`
- Wheel tag: `cp312-cp312-win_arm64`
- Requires Python: `>=3.11`
- `onnxruntime_qnn\build_and_package_info.py`: `qnn_version = '2.48.40'`
- `onnxruntime_qnn\__init__.py` exposes:
  - `EP_NAME = "QNNExecutionProvider"`
  - `get_library_path()` -> `onnxruntime_providers_qnn.dll`
  - `get_qnn_htp_path()` -> `QnnHtp.dll`
  - `get_qnn_gpu_path()` -> `QnnGpu.dll`
  - `get_qnn_cpu_path()` -> `QnnCpu.dll`, but `QnnCpu.dll` is not present in this win-arm64 wheel.

Exact extracted file inventory:

```text
onnxruntime_qnn\__init__.py                                            1,392
onnxruntime_qnn\build_and_package_info.py                                 79
onnxruntime_qnn\Genie.dll                                          8,941,264
onnxruntime_qnn\libQnnHtpV68Skel.so                               10,240,928
onnxruntime_qnn\libqnnhtpv73.cat                                      12,213
onnxruntime_qnn\libQnnHtpV73Skel.so                               11,531,512
onnxruntime_qnn\libqnnhtpv81.cat                                      12,214
onnxruntime_qnn\libQnnHtpV81Skel.so                               12,606,648
onnxruntime_qnn\LICENSE                                               1,140
onnxruntime_qnn\onnxruntime_providers_qnn.dll                     3,088,592
onnxruntime_qnn\Privacy.md                                            2,469
onnxruntime_qnn\QnnGpu.dll                                        7,558,352
onnxruntime_qnn\QnnHtp.dll                                        3,313,360
onnxruntime_qnn\QnnHtpNetRunExtensions.dll                         941,776
onnxruntime_qnn\QnnHtpPrepare.dll                                94,336,208
onnxruntime_qnn\QnnHtpV68Stub.dll                                  550,096
onnxruntime_qnn\QnnHtpV73Stub.dll                                  566,480
onnxruntime_qnn\QnnHtpV81Stub.dll                                  566,480
onnxruntime_qnn\QnnIr.dll                                        1,714,384
onnxruntime_qnn\QnnSaver.dll                                       600,784
onnxruntime_qnn\QnnSystem.dll                                    3,630,800
onnxruntime_qnn\Qualcomm_LICENSE.pdf                               147,577
onnxruntime_qnn\ThirdPartyNotices.txt                               69,810
onnxruntime_qnn-2.4.0.dist-info\licenses\LICENSE                     1,140
onnxruntime_qnn-2.4.0.dist-info\METADATA                             3,063
onnxruntime_qnn-2.4.0.dist-info\RECORD                               2,531
onnxruntime_qnn-2.4.0.dist-info\top_level.txt                           16
onnxruntime_qnn-2.4.0.dist-info\WHEEL                                  101
```

Shared-library roles and exports inspected:

```text
C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64\onnxruntime_qnn\onnxruntime_providers_qnn.dll
  size: 3,088,592
  export: CreateEpFactories
  role: ORT plugin EP registration library. This is the library to pass to RegisterExecutionProviderLibrary.

C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64\onnxruntime_qnn\QnnHtp.dll
  size: 3,313,360
  export: QnnInterface_getProviders
  role: HTP/NPU QNN backend library. This is the `backend_path` target for NPU.

C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64\onnxruntime_qnn\QnnGpu.dll
  size: 7,558,352
  export: QnnInterface_getProviders
  role: QNN GPU backend.

C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64\onnxruntime_qnn\QnnSystem.dll
  size: 3,630,800
  exports: QnnSystemInterfaceInternal_getProviders, QnnSystemInterface_getProviders
  role: QNN system/context support library.

C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64\onnxruntime_qnn\QnnSaver.dll
  size: 600,784
  exports: QnnInterface_getProviders, QnnSaver_initialize
  role: QNN saver/debug backend.

C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64\onnxruntime_qnn\QnnHtpPrepare.dll
  size: 94,336,208
  role: HTP graph prepare/finalize support.

C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64\onnxruntime_qnn\QnnHtpV68Stub.dll
C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64\onnxruntime_qnn\QnnHtpV73Stub.dll
C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64\onnxruntime_qnn\QnnHtpV81Stub.dll
  role: HTP stub libraries; exports include transport/platform support functions.

C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64\onnxruntime_qnn\libQnnHtpV68Skel.so
C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64\onnxruntime_qnn\libQnnHtpV73Skel.so
C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64\onnxruntime_qnn\libQnnHtpV81Skel.so
  role: HTP skeleton libraries.

C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64\onnxruntime_qnn\libqnnhtpv73.cat
C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64\onnxruntime_qnn\libqnnhtpv81.cat
  role: Windows catalog/signature sidecars.
```

The regular ORT wheel extracted to:

```text
C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime-1.28.0-cp312-cp312-win_arm64
```

Important runtime DLLs:

```text
C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime-1.28.0-cp312-cp312-win_arm64\onnxruntime\capi\onnxruntime.dll
C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime-1.28.0-cp312-cp312-win_arm64\onnxruntime\capi\onnxruntime_providers_shared.dll
```

## 2. How onnx-genai currently registers and drives external ORT plugin EPs

The runtime config already has a provider-agnostic plugin surface:

- `EpSelection` normalizes the provider name and carries opaque provider options; provider options are forwarded unchanged by the ORT layer (`crates\onnx-genai-runtime-config\src\lib.rs:14-23`).
- `PluginSpec` represents an inline plugin from `ONNX_GENAI_EP=plugin:<path>|...`; it carries `library`, optional registration name, provider options, and optional hardware device class (`crates\onnx-genai-runtime-config\src\lib.rs:43-65`).
- `RuntimeConfig` exposes generic plugin env vars: `ONNX_GENAI_EP_LIBRARY`, `ONNX_GENAI_EP_NAME`, `ONNX_GENAI_EP_OPTIONS`, and `ONNX_GENAI_EP_DEVICE` (`crates\onnx-genai-runtime-config\src\lib.rs:170-190`).
- `ONNX_GENAI_EP` parses an ordered priority list of built-ins and inline plugins; the concrete provider name is deliberately discovered at load time, not hardcoded in parser logic (`crates\onnx-genai-runtime-config\src\lib.rs:417-430`).
- Inline plugin syntax is `plugin:<library>[|name=<n>][|device=<class>][|opt.<k>=<v>]...`; only `opt.*` entries become provider options passed through to ORT (`crates\onnx-genai-runtime-config\src\lib.rs:456-501`).
- `ONNX_GENAI_EP_OPTIONS` parses a comma-separated `key=value` list with provider-agnostic keys/values (`crates\onnx-genai-runtime-config\src\lib.rs:504-521`).

The ORT-facing session layer resolves this config:

- `execution_providers_from_env()` turns a bare `ONNX_GENAI_EP=plugin` plus scalar env vars into `resolve_plugin_selection(...)`, or resolves inline `PluginSpec` entries directly (`crates\onnx-genai-ort\src\session\env_config.rs:14-43`).
- A `PluginLibrary` append strategy carries only `lib`, `registration_name`, `options`, and `device`; it is intentionally not provider-specific (`crates\onnx-genai-ort\src\session\ep_compat.rs:116-123`).
- The generic plugin bridge metadata is available through `native_plugin_bridge()` so native code can see the plugin library/registration/provider without duplicating EP-name logic (`crates\onnx-genai-ort\src\session\ep_compat.rs:164-190`).
- `append_execution_provider()` dispatches `AppendStrategy::PluginLibrary` to `append_plugin_execution_provider(...)` (`crates\onnx-genai-ort\src\session\providers.rs:61-126`).

The C ABI seam is in `plugin.rs` and `env.rs`:

- `append_plugin_execution_provider()` documents the intended ORT plugin flow: `RegisterExecutionProviderLibrary` + `GetEpDevices` + `SessionOptionsAppendExecutionProvider_V2`; the provider name is discovered by diffing devices before/after registration, so QNN does not need a hardcoded provider name for the generic path (`crates\onnx-genai-ort\src\session\plugin.rs:65-74`).
- It requires ORT C API functions `GetEpDevices`, `EpDevice_EpName`, and `SessionOptionsAppendExecutionProvider_V2` (`crates\onnx-genai-ort\src\session\plugin.rs:90-101`).
- It snapshots devices before registration, calls `env.register_execution_provider_library(...)`, groups devices by provider name after registration, and caches the discovered provider name (`crates\onnx-genai-ort\src\session\plugin.rs:143-197`).
- `Environment::register_execution_provider_library()` calls ORT's `RegisterExecutionProviderLibrary`; on Windows it passes the library path as UTF-16 `ORTCHAR_T*` (`crates\onnx-genai-ort\src\env.rs:160-185`).
- If `ONNX_GENAI_EP_DEVICE`/inline `device=` is set, plugin device selection narrows to ORT's generic hardware class `CPU`, `GPU`, or `NPU`, not a vendor-specific string (`crates\onnx-genai-ort\src\session\plugin.rs:233-278`; parser in `crates\onnx-genai-ort\src\session\providers.rs:152-165`).
- Provider options are passed verbatim as C key/value arrays to `SessionOptionsAppendExecutionProvider_V2` (`crates\onnx-genai-ort\src\session\plugin.rs:280-316`).

Session creation and driving:

- `RawSessionOptions::new()` appends execution providers before creating the ORT session (`crates\onnx-genai-ort\src\session\mod.rs:620-665`).
- Explicit plugin providers are strict: auto-selected providers may fall back to CPU, but explicitly requested strict providers fail instead of silently changing device on session creation failure (`crates\onnx-genai-ort\src\session\mod.rs:171-187`; strictness in `crates\onnx-genai-ort\src\session\ep_compat.rs:144-161`).
- Inference is driven through standard ORT `Run` and `RunWithBinding`; the latter is the IoBinding path used by decode loops (`crates\onnx-genai-ort\src\session\mod.rs:210-245`, `crates\onnx-genai-ort\src\session\mod.rs:265-275`).
- The engine keeps `Environment` as the last field so plugin EP factories outlive all sessions; this avoids plugin allocator/context use-after-free during teardown (`crates\onnx-genai-engine\src\engine\model.rs:64-68`).

Named/non-plugin providers:

- `NamedGeneric` already supports append-by-name with provider options through `SessionOptionsAppendExecutionProvider` (`crates\onnx-genai-ort\src\session\providers.rs:127-149`, `crates\onnx-genai-ort\src\session\providers.rs:167-222`).
- That is not sufficient for the wheel QNN package because the QNN EP is an external plugin DLL. `ONNX_GENAI_EP=qnn` currently falls into the unknown-provider path, attempts a by-name append, and does not register `onnxruntime_providers_qnn.dll` (`crates\onnx-genai-ort\src\session\ep_compat.rs:330-351`).

What a new prebuilt ORT plugin EP must provide to be driven by us:

1. A real shared library path, loadable by ORT's `RegisterExecutionProviderLibrary`.
2. The ORT plugin export expected by ORT. For this QNN wheel, `onnxruntime_providers_qnn.dll` exports `CreateEpFactories`.
3. Any dependent DLLs discoverable by the Windows loader. For QNN, keep the QNN package directory on `PATH` or load from that directory so `QnnHtp.dll`, `QnnSystem.dll`, stubs/skels, etc. resolve.
4. At least one EP device returned by ORT `GetEpDevices` after registration.
5. Provider options as strings. For QNN HTP/NPU the key one is `backend_path=<full path to QnnHtp.dll>` or `backend_type=htp` if supported; `backend_path` is safer because it names the exact wheel DLL.

## 3. QNN-specific gap analysis

### 3.1 What already works today with the generic plugin path

The current code should already be able to attempt QNN plugin loading if the operator supplies explicit generic-plugin env vars:

```powershell
$qnnDir = 'C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64\onnxruntime_qnn'
$ortDir = 'C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime-1.28.0-cp312-cp312-win_arm64\onnxruntime\capi'
$env:PATH = "$qnnDir;$ortDir;$env:PATH"
$env:ONNX_GENAI_ORT_LIB = "$ortDir\onnxruntime.dll"
$env:ONNX_GENAI_EP = "plugin"
$env:ONNX_GENAI_EP_LIBRARY = "$qnnDir\onnxruntime_providers_qnn.dll"
$env:ONNX_GENAI_EP_NAME = "onnxruntime_qnn_ep"
$env:ONNX_GENAI_EP_DEVICE = "NPU"
$env:ONNX_GENAI_EP_OPTIONS = "backend_path=$qnnDir\QnnHtp.dll,htp_performance_mode=burst"
```

Equivalent inline form:

```powershell
$env:ONNX_GENAI_EP = "plugin:$qnnDir\onnxruntime_providers_qnn.dll|name=onnxruntime_qnn_ep|device=NPU|opt.backend_path=$qnnDir\QnnHtp.dll|opt.htp_performance_mode=burst"
```

This is useful as a Wave-2 smoke path before adding a dedicated `qnn` alias.

### 3.2 What is missing for `ONNX_GENAI_EP=qnn`

`ONNX_GENAI_EP=qnn` is not currently a first-class provider:

- `selectable_execution_providers()` only exposes `cpu`, optionally `cuda`, and optionally `metal` (`crates\onnx-genai-ort\src\session\ep_compat.rs:193-219`).
- `resolve_execution_provider()` has cases for `cuda`, `webgpu`, `coreml`, and `metal`, but no `qnn`; unknown names use by-name append with conservative capabilities (`crates\onnx-genai-ort\src\session\ep_compat.rs:239-351`).
- The by-name path can pass options but cannot register the wheel's plugin DLL. QNN needs `PluginLibrary`, not `NamedGeneric`.

Needed wiring:

1. Add runtime config entries for QNN library discovery:
   - `ONNX_GENAI_QNN_EP_LIB` -> full path to `onnxruntime_providers_qnn.dll`
   - `ONNX_GENAI_QNN_BACKEND_PATH` -> full path to `QnnHtp.dll` (default NPU)
   - optional `ONNX_GENAI_QNN_BACKEND_TYPE` (`htp`, `gpu`, `cpu`, `saver`) but do not set both `backend_path` and `backend_type`.
   - optional `ONNX_GENAI_QNN_PERFORMANCE_MODE`, `ONNX_GENAI_QNN_VTCM_MB`, `ONNX_GENAI_QNN_HTP_ARCH`, `ONNX_GENAI_QNN_SOC_MODEL`, `ONNX_GENAI_QNN_CONTEXT_*`.
2. Add a `qnn` case in `resolve_execution_provider()` that returns `AppendStrategy::PluginLibrary` with:
   - `lib = qnn_ep_lib`
   - `registration_name = "onnxruntime_qnn_ep"`
   - provider options initialized from the `EpSelection` plus QNN defaults, especially `backend_path`
   - `device = Some("NPU")` by default for HTP, unless caller overrides.
3. Add `qnn` to `selectable_execution_providers()` only when the QNN plugin DLL path exists, mirroring the Metal pattern.
4. Keep capabilities conservative at first: `HardwareKind::Npu`, no `FIXED_CAPACITY_PRESENT_BINDING`, no `DEVICE_KV`, no `DEVICE_SAMPLING`, no `GRAPH_CAPTURE` until measured/verified. Capability flags are the only stable EP behavior vocabulary used by decode/allocation code (`crates\onnx-genai-ort\src\session\ep_compat.rs:15-29`).

### 3.3 Provider options required/likely for QNN

From ORT QNN EP docs and inspected strings, HTP/NPU provider options should start with:

```text
backend_path=<full path to QnnHtp.dll>
htp_performance_mode=burst | sustained_high_performance | high_performance | balanced | default | ...
device_id=0
htp_graph_finalization_optimization_mode=3 for best final graph, or 1 for faster bring-up
profiling_level=basic
profiling_file_path=<path>.csv
```

Additional QNN options to expose later:

```text
backend_type=htp | gpu | cpu | saver       # alternative to backend_path, not together
vtcm_mb=<MB>
rpc_control_latency=<microseconds>
qnn_context_priority=normal | normal_high | high | low
soc_model=<model number>
htp_arch=<architecture number>
enable_htp_fp16_precision=0|1
offload_graph_io_quantization=0|1
enable_htp_shared_memory_allocator=0|1
qnn_saver_path=<path to QnnSaver.dll>
```

For the first NPU attempt, prefer the explicit full backend path:

```text
backend_path=C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64\onnxruntime_qnn\QnnHtp.dll
htp_performance_mode=burst
```

Do not rely on `QnnCpu.dll` for this win-arm64 wheel; it is not present in the inventory.

### 3.4 Session config gap for strict QNN verification and EPContext

Provider options are not enough for two important QNN flows:

1. Full-NPU verification should set ORT session config `session.disable_cpu_ep_fallback=1` so ORT fails if unsupported nodes fall back to CPU. Our current "strict provider" only prevents session-creation failure from retrying CPU; it does not force every node onto QNN.
2. QNN context binary generation/use relies on ORT session config keys:
   - `ep.context_enable=1`
   - `ep.context_file_path=<model_ctx.onnx>`
   - `ep.context_embed_mode=0|1`

We have a generic helper for `AddSessionConfigEntry` (`crates\onnx-genai-ort\src\session\providers.rs:43-59`), but `SessionOptions` does not yet carry arbitrary session config entries. Today it is used only for WebGPU-specific entries (`crates\onnx-genai-ort\src\session\providers.rs:12-40`).

Wave 2 should add generic `SessionOptions.session_config_entries: Vec<(String, String)>` plus env parsing for something like:

```text
ONNX_GENAI_ORT_SESSION_OPTIONS=session.disable_cpu_ep_fallback=1,ep.context_enable=1,ep.context_embed_mode=1
```

or QNN-specific env vars that populate the generic entries.

### 3.5 EPContext/precompiled-context considerations

onnx-genai's native EP API already treats ORT `com.microsoft::EPContext` as the on-disk/interchange form and runtime `EpContext` as the in-memory compiled context form (`crates\onnx-runtime-ep-api\src\epcontext.rs:1-21`, `crates\onnx-runtime-ep-api\src\epcontext.rs:28-50`).

Dispatch for compiled contexts is model-agnostic: EPs declare `source` keys and the registry maps `source` -> `EpId`; there are no hardcoded vendor names (`crates\onnx-runtime-ep-api\src\epcontext.rs:71-76`, `crates\onnx-runtime-ep-api\src\epcontext.rs:153-159`).

For QNN specifically:

- The ORT QNN EP has its own context-binary mechanism exposed through ORT session config (`ep.context_*`). This is likely the first precompile/cache path to use.
- We do not need to implement a native `ExecutionProvider` for QNN to load the prebuilt ORT plugin; the ORT plugin owns QNN context creation/restore.
- If later we ingest QNN-generated `EPContext` nodes into the pure native EP API, the source-key registry is already compatible in principle, but that is separate from driving the ORT plugin.

### 3.6 Model shape, KV, and decode implications

QNN HTP/NPU has hard model-compat constraints:

- QNN HTP is primarily for quantized/QDQ models; f32/f16 Foundry ONNX models may not claim HTP partitions without quantization or precompiled QNN context.
- QNN EP does not support dynamic shapes. Static batch/sequence/KV shapes must be fixed before session creation.
- The QNN plugin strings include "QNN doesn't support dynamic shape" and "GroupQueryAttention is only supported with the GPU backend", so current Foundry GQA decode graphs may not be HTP-compatible as-is.

onnx-genai decode path interaction:

- Shared-buffer KV is selected only when metadata requests `model.io.kv_update: shared_buffer` and the active session declares fixed-capacity present-binding support (`crates\onnx-genai-engine\src\decode\metadata.rs:148-165`, `crates\onnx-genai-engine\src\decode\metadata.rs:244-265`).
- The metadata helper treats declared `kv_update: shared_buffer` as the model contract and uses `model.max_sequence_length` to size the buffer (`crates\onnx-genai-engine\src\decode\metadata.rs:224-242`).
- Session capability gating is conservative for unverified EPs; unsupported/unverified providers default away from the fixed-capacity pre-bound present path unless explicitly opted in (`crates\onnx-genai-ort\src\session\mod.rs:475-504`).
- Device-resident KV allocation is only enabled for EPs advertising `DEVICE_KV`; unsupported EPs keep CPU buffers (`crates\onnx-genai-ort\src\session\mod.rs:511-524`).

Implication: do not initially mark QNN as supporting `FIXED_CAPACITY_PRESENT_BINDING` or `DEVICE_KV`. First prove a static one-shot inference. Then prove a fixed-shape decode step. Only after QNN accepts our pre-bound fixed-capacity present outputs should we enable the shared-buffer fast path for QNN.

## 4. Ordered Wave-2 implementation plan

### Milestone 0: Manual generic-plugin smoke, no code changes

Goal: prove ORT can register the wheel plugin and create a QNN EP session.

1. Keep QNN and ORT DLL directories on `PATH` for dependency resolution:

   ```powershell
   $qnnDir = 'C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64\onnxruntime_qnn'
   $ortDir = 'C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime-1.28.0-cp312-cp312-win_arm64\onnxruntime\capi'
   $env:PATH = "$qnnDir;$ortDir;$env:PATH"
   $env:ONNX_GENAI_ORT_LIB = "$ortDir\onnxruntime.dll"
   $env:ONNX_GENAI_EP = "plugin"
   $env:ONNX_GENAI_EP_LIBRARY = "$qnnDir\onnxruntime_providers_qnn.dll"
   $env:ONNX_GENAI_EP_NAME = "onnxruntime_qnn_ep"
   $env:ONNX_GENAI_EP_DEVICE = "NPU"
   $env:ONNX_GENAI_EP_OPTIONS = "backend_path=$qnnDir\QnnHtp.dll,htp_performance_mode=burst"
   ```

2. Load a tiny static QDQ ONNX model with QNN. If available, set `session.disable_cpu_ep_fallback=1` after adding session-config plumbing; before that, inspect logs/profiling to verify QNN.
3. First verifiable success criterion: session creation succeeds, QNN provider is logged by `append_plugin_execution_provider`, and one inference produces correct output with QNN profiling or no-CPU-fallback enabled.

### Milestone 1: Add first-class `ONNX_GENAI_EP=qnn`

1. Extend `onnx-genai-runtime-config`:
   - Add `qnn_ep_lib: Option<PathBuf>`.
   - Add `qnn_backend_path: Option<PathBuf>`.
   - Add optional QNN tuning config fields or use selection/provider options.
   - Parse env vars: `ONNX_GENAI_QNN_EP_LIB`, `ONNX_GENAI_QNN_BACKEND_PATH`, `ONNX_GENAI_QNN_PERFORMANCE_MODE`.
   - Add tests adjacent to existing plugin parsing tests.
2. Extend `ep_compat.rs`:
   - Add a `"qnn" | "qnn-htp" | "qnn_htp"` case.
   - Return `HardwareKind::Npu`, conservative capability flags `[]`, and `AppendStrategy::PluginLibrary`.
   - Registration name: `"onnxruntime_qnn_ep"`.
   - Device class: `Some("NPU")` by default.
   - Options: merge caller-provided options with defaults; if no `backend_path`/`backend_type` exists, add `backend_path=<QnnHtp.dll>`.
3. Add `qnn` to `selectable_execution_providers()` only when the QNN EP DLL exists.
4. Add tests:
   - `resolve_execution_provider(ep_selection("qnn"))` produces `PluginLibrary`.
   - QNN options include `backend_path`.
   - QNN remains strict and conservative: no fixed-capacity present binding until explicitly changed.

### Milestone 2: Add generic ORT session-config entries

1. Add `session_config_entries` to `SessionOptions`.
2. Parse `ONNX_GENAI_ORT_SESSION_OPTIONS` as provider-agnostic `key=value,key=value`.
3. Apply entries through the existing `add_session_config_entry()` before EP append/session creation.
4. Add QNN-specific convenience:
   - `ONNX_GENAI_QNN_DISABLE_CPU_FALLBACK=1` -> `session.disable_cpu_ep_fallback=1`
   - `ONNX_GENAI_QNN_CONTEXT_ENABLE=1` -> `ep.context_enable=1`
   - `ONNX_GENAI_QNN_CONTEXT_FILE=<path>` -> `ep.context_file_path=<path>`
   - `ONNX_GENAI_QNN_CONTEXT_EMBED=1` -> `ep.context_embed_mode=1`
5. First real verification should use `session.disable_cpu_ep_fallback=1`; otherwise QNN may silently run unsupported nodes on CPU.

### Milestone 3: QNN model compatibility probe

1. Start with one small static QDQ model known to be QNN-compatible.
2. Run CPU vs QNN and verify numerical tolerance.
3. Turn on QNN profiling (`profiling_level=basic`, `profiling_file_path=...`) to prove HTP execution.
4. Generate or consume a QNN context model with `ep.context_enable=1` and compare cold-start.

### Milestone 4: Foundry model path

1. Inventory the exact Foundry ONNX graph:
   - static vs dynamic shapes
   - QDQ quantization present?
   - GQA op vs decomposed attention
   - KV input/output shapes and metadata `model.io.kv_update`.
2. If model uses dynamic shapes, fix shapes or use a context-binary/precompiled model per target shape.
3. If model uses GQA on HTP, verify whether current QNN 2.48.40 supports it. Inspected plugin strings say GroupQueryAttention is only supported with the GPU backend, so HTP may require export changes/decomposition.
4. First Foundry success criterion: one prefill or one decode-step inference on QNN HTP with CPU fallback disabled.
5. Only after fixed-shape present-output binding is verified, add QNN `FIXED_CAPACITY_PRESENT_BINDING` capability and enable shared-buffer decode. Until then, use static-cache/one-shot paths or keep QNN off the shared-buffer fast path.

### Milestone 5: Performance and productization

1. With Sebastian's benchmark window clear, measure CPU native vs ORT QNN vs native-driven QNN on the same model/shape.
2. Add a small README/troubleshooting section:
   - wheel package name is `onnxruntime-qnn`
   - set `PATH` for QNN DLL dependencies
   - set `ONNX_GENAI_QNN_EP_LIB` and `ONNX_GENAI_QNN_BACKEND_PATH`
   - HTP requires QDQ/static shapes
   - how to enable profiling/context binaries
3. Add a no-heavy-build CI/unit layer for config resolution only. Hardware E2E stays opt-in.

## 5. Blockers and risks

- Package-name mismatch: `onnxruntime-ep-qnn` was not found on PyPI; the usable package is `onnxruntime-qnn`.
- The win-arm64 QNN wheel has no `QnnCpu.dll`; do not use QNN CPU backend as the integration fallback on this wheel.
- QNN HTP likely requires QDQ quantized, fixed-shape graphs. Foundry f16/dynamic decode graphs may not run on NPU without export/quantization/context work.
- Current runtime lacks generic ORT session-config plumbing, so it cannot yet set `session.disable_cpu_ep_fallback=1` or QNN `ep.context_*` entries without code changes.
- `ONNX_GENAI_EP=qnn` does not currently register a plugin; generic `ONNX_GENAI_EP=plugin` can be used for first smoke, but first-class UX needs Wave-2 wiring.
- The downloaded ORT dependency is 1.28.0. onnx-genai's loader asks ORT for API 27 and accepts a runtime if `GetApi(27)` is available (`crates\onnx-genai-ort\ort-sys\src\lib.rs:240-250`, `crates\onnx-genai-ort\ort-sys\src\lib.rs:296-304`). If this runtime does not expose API 27, use an ORT 1.27 library or update bindings deliberately.

## 6. Wave-2 implementation results — 2026-07-28T22:50-07:00

Implemented first-class QNN selection and generic session-config plumbing in the owned crates only:

- `crates\onnx-genai-runtime-config\src\lib.rs:157-200` adds QNN env vars and `ONNX_GENAI_ORT_SESSION_OPTIONS`; `:346-367` parses them; `:860-910` tests them.
- `crates\onnx-genai-ort\src\session\ep_compat.rs:220-224` exposes `qnn` only when `ONNX_GENAI_QNN_EP_LIB` points at an existing plugin DLL; `:337-358` resolves `ONNX_GENAI_EP=qnn` to a strict `PluginLibrary` with conservative `HardwareKind::Npu` capabilities, registration handle `onnxruntime_qnn_ep`, default device `NPU`, and QNN provider options; `:389-476` seeds `backend_path=QnnHtp.dll` (or the configured backend path) and optional HTP tuning options.
- `crates\onnx-genai-ort\src\session\options.rs:52-53`, `:114`, and `:184-207` carry generic session config entries and add QNN defaults for `session.disable_cpu_ep_fallback`, `ep.context_enable`, `ep.context_file_path`, and `ep.context_embed_mode` when requested.
- `crates\onnx-genai-ort\src\session\mod.rs:663-665` applies generic session config entries through ORT `AddSessionConfigEntry` before provider append/session creation.
- `crates\onnx-genai-ort\src\session\tests.rs` adds QNN conservative-plugin and strict-provider coverage.
- `crates\onnx-genai-ort\examples\qnn_smoke.rs` is a tiny static-QDQ smoke harness used to prove `Session::new` + `Session::run` can drive QNN.

### Tiny static QDQ NPU proof

Generated a tiny static QDQ Conv model at:

```text
C:\Users\justinchu\dev\onnx-genai\target\qnn-smoke\tiny_qdq_conv.onnx
```

The model is one fixed-shape float input `[1,1,4,4]` wrapped as QDQ uint8 activation, QDQ uint8 weight, `Conv`, then QDQ output. CPU ORT reference output for input `0..15` was:

```text
[-4.0, -3.4000000953674316, -2.0, -1.399999976158142]
```

Ran it through our `onnx-genai-ort` `Session` path with the new `ONNX_GENAI_EP=qnn` alias using `cargo run -p onnx-genai-ort --example qnn_smoke -- $model`; the command path exercised our Rust `Environment`, `SessionOptions::default()`, `Session::new`, and `Session::run`.

Working environment/incantation:

```powershell
$repo='C:\Users\justinchu\dev\onnx-genai'
$qnn='C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime_qnn-2.4.0-cp312-cp312-win_arm64\onnxruntime_qnn'
$ort='C:\Users\justinchu\dev\qnn-ep-wheel\onnxruntime-1.28.0-cp312-cp312-win_arm64\onnxruntime\capi'
$model="$repo\target\qnn-smoke\tiny_qdq_conv.onnx"
$env:PATH="$qnn;$ort;$env:PATH"
$env:ONNX_GENAI_ORT_LIB="$ort\onnxruntime.dll"
$env:ONNX_GENAI_EP='qnn'
$env:ONNX_GENAI_QNN_EP_LIB="$qnn\onnxruntime_providers_qnn.dll"
$env:ONNX_GENAI_QNN_BACKEND_PATH="$qnn\QnnHtp.dll"
$env:ONNX_GENAI_QNN_DISABLE_CPU_FALLBACK='1'
$env:ONNX_GENAI_QNN_PERFORMANCE_MODE='burst'
$env:ONNX_GENAI_EP_OPTIONS="profiling_level=basic,profiling_file_path=$repo\target\qnn-smoke\qnn_profile.csv,enable_framework_op_trace=1,framework_op_trace_dir=$repo\target\qnn-smoke"
# then run our native ORT wrapper against the model
cargo run -p onnx-genai-ort --example qnn_smoke -- $model
```

Evidence:

- Loader selected the wheel ORT successfully: `onnx-genai: selected ONNX Runtime 1.28.0 (API 27) from ...\onnxruntime.dll (ONNX_GENAI_ORT_LIB)`.
- QNN emitted HTP graph-preparation/finalization stages and DDR summary, including `Starting stage: Graph Preparation Initializing`, `Starting stage: Finalizing Graph Sequence`, and `DDR bandwidth summary`.
- Our runtime output was `[-4.0, -3.3, -2.0, -1.3000001]`, matching the expected quantized tolerance.
- `target\qnn-smoke\qnn_op_trace.json` was generated. Its summary proves HTP/backend execution with no CPU fallback:

```json
{
  "backend_type": "htp",
  "compilation_target": { "device_id": 0, "htp_arch": "V73", "soc_model": 0 },
  "summary": {
    "qnn_subgraphs": 1,
    "supported_nodes": 12,
    "total_onnx_nodes": 12,
    "total_qnn_ops": 5,
    "unsupported_nodes": 0
  },
  "unsupported_nodes": []
}
```

`profiling_file_path` did not produce a CSV because the QNN EP reported `ETW enabled previously, but disabled now. Can't do the switch! Won't output any profiling.` The framework op trace was sufficient evidence: backend `htp`, one QNN subgraph, all ONNX nodes supported, zero unsupported nodes, while `session.disable_cpu_ep_fallback=1` was active.

The vendored/default ORT was not used for the smoke. The successful run explicitly used the wheel's ORT 1.28.0 DLL via `ONNX_GENAI_ORT_LIB`; this confirms ORT 1.28 exposes API 27 to our bindings and has the plugin-EP v2 surface needed by QNN. Keep this as the recommended QNN smoke setup unless/until the vendored ORT is updated/verified for plugin EP v2.

### Validation

Passed:

```text
cargo test -p onnx-genai-runtime-config
cargo test -p onnx-genai-ort session::tests
cargo fmt --all -- --check
cargo clippy -p onnx-genai-runtime-config -- -D warnings
cargo clippy -p onnx-genai-ort -- -D warnings
```

Full `cargo test -p onnx-genai-ort` result: 67 passed, 1 failed. Failure appears unrelated/pre-existing in `loader::model_package_tests::flat_directory_with_unrelated_manifest_remains_backward_compatible`, expecting `model.onnx.textproto` while resolving `model.onnx` for `tests/fixtures/tiny-llm-scatter`; the QNN/session tests all passed.
