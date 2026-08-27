### 2026-08-25T17-07-08: CUDA NMS uses bounded DeviceWorkspace selection with 8-byte D2H
**By:** Leon
**What:** CUDA NMS uses bounded DeviceWorkspace selection with 8-byte D2H
**References:** feat/cuda-nms, PR #2112, PR #2113, crates/onnx-runtime-ep-cuda/src/kernels/non_max_suppression.rs
**Why:** CUDA NonMaxSuppression reuses #2113's DeviceWorkspace dynamic-output policy and #2112's exact CPU row semantics. It claims static contiguous f32 boxes/scores only, bounded to 256 boxes and 256 batch×class groups. One block per group filters and bitonic-sorts scores in parallel, then performs deterministic bounded suppression; a second metadata kernel sums counts. Only the 8-byte selected count crosses D2H before one device materialization launch. Optional scalars remain device-resident, workspace is governed StepScoped memory, and capture fails closed. No second device-output policy or full-input D2H was added.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
