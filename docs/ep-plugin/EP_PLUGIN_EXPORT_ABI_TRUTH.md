# EP Plugin Export ABI — Ground Truth from ORT 1.27.0 Headers

**Source:** `onnxruntime-linux-x64-1.27.0.tgz` (SHA-256: `547e40a48f1fe73e3f812d7c88a948612c23f896b91e4e2ee1e232d7b468246f`), files `include/onnxruntime_c_api.h` and `include/onnxruntime_ep_c_api.h`.

**Date:** 2026-08-10

---

## 1. Required Export Symbols

### Verdict: **Both Nabil AND Pris are partially right; the doc comment and typedef disagree on the C identifier, but the EXPORTED SYMBOL NAME is `CreateEpFactories`.**

The `onnxruntime_c_api.h` line 5579 doc comment on `RegisterExecutionProviderLibrary` states:

> ```
> The library must export 'CreateEpFactories' and 'ReleaseEpFactory' functions.
> ```

The `onnxruntime_ep_c_api.h` line 2637 doc comment on the typedef says:

> ```
> This must be available in a function called 'CreateEpFactories' in the execution provider library.
> ```

However, the **typedef** name is `CreateEpApiFactoriesFn` (ep_c_api.h:2654):

```c
typedef OrtStatus* (*CreateEpApiFactoriesFn)(
    _In_ const char* registered_name,
    _In_ const OrtApiBase* ort_api_base,
    _In_ const OrtLogger* default_logger,
    _Inout_ OrtEpFactory** factories,
    _In_ size_t max_factories,
    _Out_ size_t* num_factories);
```

And `ReleaseEpFactory` (ep_c_api.h:2661-2669):

> ```
> This must be available in a function called 'ReleaseEpFactory' in the execution provider library.
> ```

```c
typedef OrtStatus* (*ReleaseEpApiFactoryFn)(_In_ OrtEpFactory* factory);
```

### Conclusion

| What | Value |
|------|-------|
| **Exported symbol name (create)** | `CreateEpFactories` |
| **Exported symbol name (release)** | `ReleaseEpFactory` |
| **C typedef (create)** | `CreateEpApiFactoriesFn` |
| **C typedef (release)** | `ReleaseEpApiFactoryFn` |
| **Release required?** | **YES** — the doc says "must export" both |

**Nabil said `CreateEpFactories`** — **CORRECT** for the exported symbol name.
**Pris said `CreateEpApiFactories`** — **INCORRECT.** That is neither the exported symbol name nor the typedef name. The typedef is `CreateEpApiFactoriesFn` (note trailing `Fn`), but ORT `dlsym`s for `CreateEpFactories`.

---

## 2. Can Upstream ORT Load a Plugin EP at Runtime? — **YES**

### Claim C is **WRONG**.

The `OrtApi` struct (onnxruntime_c_api.h) contains these members:

**Line 5577-5592:**
```c
ORT_API2_STATUS(RegisterExecutionProviderLibrary,
    _In_ OrtEnv* env,
    _In_ const char* registration_name,
    _In_ const ORTCHAR_T* path);
```

**Line 5607:**
```c
ORT_API2_STATUS(UnregisterExecutionProviderLibrary,
    _In_ OrtEnv* env,
    _In_ const char* registration_name);
```

**Line 5621:**
```c
ORT_API2_STATUS(GetEpDevices,
    _In_ const OrtEnv* env,
    _Outptr_ const OrtEpDevice* const** ep_devices,
    _Out_ size_t* num_ep_devices);
```

**Line 5643:**
```c
ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_V2,
    _In_ OrtSessionOptions* session_options,
    _In_ OrtEnv* env,
    _In_reads_(num_ep_devices) const OrtEpDevice* const* ep_devices,
    _In_ size_t num_ep_devices,
    _In_reads_(num_op_options) const char* const* ep_option_keys,
    _In_reads_(num_op_options) const char* const* ep_option_vals,
    size_t num_ep_options);
```

These are **function pointers inside the OrtApi struct**, obtained via `OrtGetApiBase()->GetApi(ORT_API_VERSION)`. They are invisible to `nm -D` because they are not individual exported symbols — Pris's methodology was fundamentally wrong. The entire ORT C API surface (hundreds of functions) is accessed through this single vtable; only `OrtGetApiBase` is a real exported symbol.

All four functions exist since **Version 1.22** (per the `\since` tags).

### Exact Call Sequence for End-to-End Test

```c
// 1. Get the API
const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

// 2. Create environment
OrtEnv* env;
api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);

// 3. Register the plugin EP library (calls dlopen + dlsym("CreateEpFactories"))
api->RegisterExecutionProviderLibrary(env, "my_ep", L"/path/to/libmy_ep.so");

// 4. Enumerate available EP devices (includes those from the registered library)
const OrtEpDevice* const* ep_devices;
size_t num_ep_devices;
api->GetEpDevices(env, &ep_devices, &num_ep_devices);

// 5. Pick the right OrtEpDevice, create session options, append EP
OrtSessionOptions* opts;
api->CreateSessionOptions(&opts);
api->SessionOptionsAppendExecutionProvider_V2(opts, env, &ep_devices[i], 1, NULL, NULL, 0);

// 6. Create session and run
OrtSession* session;
api->CreateSession(env, L"model.onnx", opts, &session);
// ... run inference ...

// 7. Cleanup
api->UnregisterExecutionProviderLibrary(env, "my_ep");
```

---

## 3. Authoritative Vtable Definitions

### OrtEpFactory (ep_c_api.h, struct starts at line ~2675)

```
ort_version_supported: uint32_t
GetName:               const char* (const OrtEpFactory*)
GetVendor:             const char* (const OrtEpFactory*)
GetSupportedDevices:   OrtStatus* (OrtEpFactory*, const OrtHardwareDevice* const*, size_t, OrtEpDevice**, size_t, size_t*)
CreateEp:              OrtStatus* (OrtEpFactory*, const OrtHardwareDevice* const*, const OrtKeyValuePairs* const*, size_t, const OrtSessionOptions*, const OrtLogger*, OrtEp**)
ReleaseEp:             void (OrtEpFactory*, OrtEp*)
GetVendorId:           uint32_t (const OrtEpFactory*)
GetVersion:            const char* (const OrtEpFactory*)
ValidateCompiledModelCompatibilityInfo: OrtStatus* (...)
CreateAllocator:       OrtStatus* (OrtEpFactory*, const OrtMemoryInfo*, const OrtKeyValuePairs*, OrtAllocator**)
ReleaseAllocator:      void (OrtEpFactory*, OrtAllocator*)
CreateDataTransfer:    OrtStatus* (OrtEpFactory*, OrtDataTransferImpl**)
IsStreamAware:         bool (const OrtEpFactory*)
CreateSyncStreamForDevice: OrtStatus* (...)
GetHardwareDeviceIncompatibilityDetails: OrtStatus* (optional, since 1.24)
CreateExternalResourceImporterForDevice: OrtStatus* (optional, since 1.24)
GetNumCustomOpDomains: OrtStatus* (since 1.24)
GetCustomOpDomains:    OrtStatus* (since 1.24)
InitGraphicsInterop:   OrtStatus* (optional, since 1.25)
DeinitGraphicsInterop: OrtStatus* (optional, since 1.25)
```

### OrtEp (ep_c_api.h, struct starts around line 2479)

```
ort_version_supported: uint32_t
GetName:               const char* (const OrtEp*)
GetCapability:         OrtStatus* (OrtEp*, const OrtGraph*, OrtEpGraphSupportInfo*)
Compile:               OrtStatus* (OrtEp*, const OrtGraph**, const OrtNode**, size_t, OrtNodeComputeInfo**, OrtNode**)  [optional since 1.24]
ReleaseNodeComputeInfos: void (OrtEp*, OrtNodeComputeInfo**, size_t)  [optional since 1.24]
GetPreferredDataLayout: OrtStatus* (optional)
ShouldConvertDataLayoutForOp: OrtStatus* (optional)
SetDynamicOptions:     OrtStatus* (optional)
OnRunStart:            OrtStatus* (optional)
OnRunEnd:              OrtStatus* (optional)
CreateAllocator:       OrtStatus* (optional)
CreateSyncStreamForDevice: OrtStatus* (optional)
GetCompiledModelCompatibilityInfo: const char* (OrtEp*, const OrtGraph*)
GetKernelRegistry:     OrtStatus* (optional, since 1.24)
IsConcurrentRunSupported: OrtStatus* (optional, since 1.24)
Sync:                  OrtStatus* (optional, since 1.25)
CreateProfiler:        OrtStatus* (optional, since 1.25)
IsGraphCaptureEnabled: bool (optional, since 1.26)
IsGraphCaptured:       bool (since 1.26, required if IsGraphCaptureEnabled implemented)
ReplayGraph:           OrtStatus* (since 1.26, required if IsGraphCaptureEnabled implemented)
GetGraphCaptureNodeAssignmentPolicy: OrtGraphCaptureNodeAssignmentPolicy (optional, since 1.26)
GetAvailableResource:  OrtStatus* (optional, since 1.26)
OnSessionInitializationEnd: OrtStatus* (optional, since 1.27)
GetDefaultMemoryDevice: OrtStatus* (optional, since 1.27)
ReleaseCapturedGraph:  OrtStatus* (optional, since 1.27)
```

### OrtNodeComputeInfo (ep_c_api.h)

```
ort_version_supported: uint32_t
CreateState:           OrtStatus* (OrtNodeComputeInfo*, OrtNodeComputeContext*, void**)
Compute:               OrtStatus* (OrtNodeComputeInfo*, void*, OrtKernelContext*)
ReleaseState:          void (OrtNodeComputeInfo*, void*)
```

### OrtEpGraphSupportInfo

Not a user-defined struct — ORT creates it and passes it to `OrtEp::GetCapability`. The EP populates it via `OrtEpApi::EpGraphSupportInfo_AddNodesToFuse` and `OrtEpApi::EpGraphSupportInfo_AddSingleNode`.

### OrtEpDevice

Opaque. Created via `OrtEpApi::CreateEpDevice`.

### OrtHardwareDevice

Opaque. Created via `OrtEpApi::CreateHardwareDevice` (since 1.24):
```c
ORT_API2_STATUS(CreateHardwareDevice,
    _In_ OrtHardwareDeviceType type,
    _In_ uint32_t vendor_id,
    _In_ uint32_t device_id,
    _In_ const char* vendor_name,
    _In_opt_ const OrtKeyValuePairs* metadata,
    _Out_ OrtHardwareDevice** hardware_device);
```

---

## 4. Version-Negotiation Contract

Every struct (`OrtEp`, `OrtEpFactory`, `OrtNodeComputeInfo`, `OrtDataTransferImpl`, `OrtSyncStreamImpl`, etc.) has a `ort_version_supported` field:

> "Implementation should set to `ORT_API_VERSION`. ORT will use this to ensure it does not call functions that were not available when the library was compiled."

This means ORT reads the EP's `ort_version_supported` value and **skips calling any function pointer that was added in a version newer than what the EP reports**. If a plugin sets `ort_version_supported = 27` and ORT is version 28, ORT will not call any version-28 additions.

The doc does NOT describe an explicit "reject on mismatch" — it is forward-compatible by design. ORT silently avoids calling newer members. There is no fail-closed rejection; it is a **graceful degradation** model.

For **fail-closed behavior** (Justin's requirement), we would need to add our own version check inside `CreateEpFactories` — verify `OrtGetApiBase()->GetApi(ORT_API_VERSION)` returns non-NULL, and if it returns NULL (meaning ORT doesn't support our compiled API version), return an error status.

---

## 5. Key Corrections for Implementation

1. **Export `CreateEpFactories`** (not `CreateEpApiFactories`). The function signature must match `CreateEpApiFactoriesFn`.
2. **Export `ReleaseEpFactory`** (not `ReleaseEpApiFactory`). Required.
3. **End-to-end tests ARE possible.** Use `OrtApi::RegisterExecutionProviderLibrary` → `GetEpDevices` → `SessionOptionsAppendExecutionProvider_V2`. Available since ORT 1.22.
4. **`ort_version_supported`** is a forward-compat field, not a fail-closed gate. Add explicit version checking in `CreateEpFactories` if fail-closed is required.

---

## 6. Accurate Field Inventory — Verified from `bindings.rs` (Roy, 2026-08-10)

Source: `target/debug/build/onnx-genai-ort-sys-3b504ed789bb5e57/out/bindings.rs`
(generated from ORT 1.27.0 headers at build time).

> **Important note:** `ValidateCompiledModelCompatibilityInfo` is a member of
> `OrtEpFactory`, NOT `OrtEp`. `OrtEp` has `GetCompiledModelCompatibilityInfo`.
> The bindings confirm this. Any doc that says otherwise is wrong.

### `OrtEp` — 24 fields

| # | Field | Optional? | Since version | v1 CPU EP |
|---|-------|-----------|---------------|-----------|
| 1 | `ort_version_supported: u32` | required | 1.22 | Set to `ORT_API_VERSION` (27) |
| 2 | `GetName` | required | 1.22 | Implement |
| 3 | `GetCapability` | required | 1.23 | Implement |
| 4 | `Compile` | optional since 1.24 | 1.23 | Implement (no kernel registry) |
| 5 | `ReleaseNodeComputeInfos` | optional since 1.24 | 1.23 | Implement |
| 6 | `GetPreferredDataLayout` | optional | 1.23 | `None` |
| 7 | `ShouldConvertDataLayoutForOp` | optional | 1.23 | `None` |
| 8 | `SetDynamicOptions` | optional | 1.23 | `None` |
| 9 | `OnRunStart` | optional | 1.23 | `None` |
| 10 | `OnRunEnd` | optional | 1.23 | `None` |
| 11 | `CreateAllocator` | optional | 1.23 | `None` (host malloc; ORT allocates) |
| 12 | `CreateSyncStreamForDevice` | optional | 1.23 | `None` (CPU has no stream) |
| 13 | `GetCompiledModelCompatibilityInfo` | required-if-Compile | 1.23 | `None` for v1 |
| 14 | `GetKernelRegistry` | optional | 1.24 | `None` |
| 15 | `IsConcurrentRunSupported` | optional | 1.24 | `None` |
| 16 | `Sync` | optional | 1.25 | `None` |
| 17 | `CreateProfiler` | optional | 1.25 | `None` ← **missing in ep.rs:34** |
| 18 | `IsGraphCaptureEnabled` | optional | 1.26 | `None` ← **missing in ep.rs:34** |
| 19 | `IsGraphCaptured` | required if #18 | 1.26 | `None` ← **missing in ep.rs:34** |
| 20 | `ReplayGraph` | required if #18 | 1.26 | `None` ← **missing in ep.rs:34** |
| 21 | `GetGraphCaptureNodeAssignmentPolicy` | optional | 1.26 | `None` ← **missing in ep.rs:34** |
| 22 | `GetAvailableResource` | optional | 1.26 | `None` ← **missing in ep.rs:34** |
| 23 | `OnSessionInitializationEnd` | optional | 1.27 | `None` ← **missing in ep.rs:34** |
| 24 | `GetDefaultMemoryDevice` | optional | 1.27 | `None` ← **missing in ep.rs:34** |
| 25 | `ReleaseCapturedGraph` | optional | 1.27 | `None` ← **missing in ep.rs:34** |

> Fields 17–25 (9 fields, not 11 as reported in the compile error; the compile error
> message rounds at 8 "and 8 other fields") are the ones causing the compile failure.
> All are `Option<fn>` and must be set to `None`.

### `OrtEpFactory` — 19 fields

| # | Field | Optional? | Since | v1 CPU EP |
|---|-------|-----------|-------|-----------|
| 1 | `ort_version_supported: u32` | required | 1.22 | Set to `ORT_API_VERSION` (27) |
| 2 | `GetName` | required | 1.22 | Implement |
| 3 | `GetVendor` | required | 1.22 | Implement |
| 4 | `GetSupportedDevices` | required | 1.22 | Implement |
| 5 | `CreateEp` | required | 1.22 | Implement |
| 6 | `ReleaseEp` | required | 1.22 | Implement |
| 7 | `GetVendorId` | optional | 1.23 | `None` |
| 8 | `GetVersion` | optional | 1.23 | `None` |
| 9 | `ValidateCompiledModelCompatibilityInfo` | optional | 1.23 | `None` |
| 10 | `CreateAllocator` | optional | 1.23 | `None` |
| 11 | `ReleaseAllocator` | optional | 1.23 | `None` |
| 12 | `CreateDataTransfer` | optional | 1.23 | `None` |
| 13 | `IsStreamAware` | optional | 1.23 | `None` |
| 14 | `CreateSyncStreamForDevice` | optional | 1.23 | `None` |
| 15 | `GetHardwareDeviceIncompatibilityDetails` | optional | 1.24 | `None` |
| 16 | `CreateExternalResourceImporterForDevice` | optional | 1.24 | `None` |
| 17 | `GetNumCustomOpDomains` | optional | 1.24 | `None` |
| 18 | `GetCustomOpDomains` | optional | 1.24 | `None` |
| 19 | `InitGraphicsInterop` | optional | 1.25 | `None` |
| 20 | `DeinitGraphicsInterop` | optional | 1.25 | `None` |

### Current Compiler Blocker

`crates/onnx-runtime-ep-plugin/src/ep.rs:34` initializes `ort::OrtEp { ... }` but
omits fields 17–25 from the table above. Since Rust struct initialization with named
fields requires all fields to be present, this is a compile error.

**Fix:** Add each missing field as `fieldname: None` in the initializer, or use
`ort::OrtEp { ..Default::default() }` if the bindings derive `Default`
(check with `grep "derive.*Default" bindings.rs | grep OrtEp`). If no `Default`,
explicitly set all 9 missing fields to `None`.
