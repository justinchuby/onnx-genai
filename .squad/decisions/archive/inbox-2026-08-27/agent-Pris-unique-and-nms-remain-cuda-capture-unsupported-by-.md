### 2026-08-26T04-42-50: Unique and NMS remain CUDA-capture unsupported by their dynamic-output contract
**By:** Pris
**What:** Unique and NMS remain CUDA-capture unsupported by their dynamic-output contract
**References:** PR #2180, crates/onnx-runtime-ep-cuda/tests/capture_sync_contract.rs, crates/onnx-runtime-ep-cuda/src/kernels/unique.rs, crates/onnx-runtime-ep-cuda/src/kernels/non_max_suppression.rs
**Why:** PR #2180 keeps CUDA Unique and NonMaxSuppression explicitly capture-unsupported. Both use KernelSizedOutputPolicy::DeviceWorkspace: prepare launches device work, synchronously copies an 8-byte count D2H, then ORT allocates data-dependent outputs before materialize. The capture-sync contract now binds each allowlisted materialize sync to that owning Kernel impl, policy, two phases, and reason; real-GPU begin_graph_capture tests prove rejection occurs before either phase.
<!-- Archived from durable inbox source `Pris-unique-and-nms-remain-cuda-capture-unsupported-by-.md` by Scribe on 2026-08-27; original inbox content above is unchanged. -->
