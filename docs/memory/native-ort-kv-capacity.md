# Native and ORT KV capacity unification

The runtime uses one KV capacity policy for ORT shared-buffer KV and native CUDA
KV: `onnx_genai_kv::kv_capacity_bucket(required, hard_max)`. The policy rounds
demand to a power-of-two bucket, applies the `ONNX_GENAI_KV_MIN_BUCKET` floor,
and clamps only to factual limits: model metadata or an explicit user cap.

## Hard maximum

The hard maximum is not a VRAM prediction. It comes from explicit caller
configuration first (`load_with_cuda_kv_max_len`, `ONNX_GENAI_CUDA_KV_MAX_LEN`,
or the ORT shared-buffer cap), otherwise from `model.max_sequence_length`. Those
are facts: the user asked for a cap, or the model cannot attend beyond its
declared context. Native CUDA does not derive a ceiling from a free-memory
snapshot; if metadata is unavailable and the user did not set a cap, it grows
until allocation fails and reports that failure transactionally.

## Growth

Both ORT shared-buffer KV and native CUDA pre-allocate at the minimum bucket
(256 tokens by default, overridden by `ONNX_GENAI_KV_MIN_BUCKET`) and grow on
demand by the shared bucket policy up to the hard maximum, never pre-allocating
the full declared context length. The orchestration lives in
`onnx_genai_kv::ensure_kv_capacity` over the `KvCapacityGrowthBackend` trait:
reject above the hard maximum, compute the next `kv_capacity_bucket`, build all
fallible replacement state while the old state remains live, invalidate graph
capture, then commit the new capacity. ORT implements the primitive seam with
`OrtValue` allocation/rebind and `grow_kv_value`; native CUDA implements it with
`DeviceIoBinding` allocation, direct device-to-device prefix copies, and mask
rewrites. Adding another backend should implement those primitives rather than
re-derive the growth policy.

Growth deliberately asks the device to allocate instead of predicting a safe
VRAM ceiling. A failed grow is expected to be graceful and transactional: the old
bucket, logical length, and capture state remain live, and the error reports the
target bucket, approximate new allocation, approximate transient peak, current
device free/total memory when available, bytes per token, and user levers. The
transient peak matters because growth keeps the old bucket alive while allocating
and filling the new bucket; peak KV+mask memory is roughly `(old + new) *
bytes_per_token`, not just `new * bytes_per_token`.

## CUDA graph capture

Growth and CUDA graph capture coexist by making growth a graph boundary. The
shared driver builds the new bucket and captured mask before it calls the
backend's capture-invalidation primitive, so failure leaves the old buffers and
old capture intact. ORT releases its captured graph id; native CUDA resets its
captured graph. The next captured step then binds/captures the new buffers and
subsequent steps replay at the new bucket. Bucket growth is logarithmically rare,
so the recapture cost is amortized.

The remaining limitation is CI coverage, not capability: the grow-and-recapture
path was exercised manually on an RTX 4060 Laptop GPU by forcing early buckets
with `ONNX_GENAI_KV_MIN_BUCKET=4`, observing native CUDA growth from 4→8→16
with graph recapture at each bucket and coherent continued generation. Automated
CUDA coverage still depends on a GPU runner and model fixtures being available.

On Windows/WDDM, forcing a real CUDA allocation failure is not reliable: the GPU
memory manager virtualizes device memory and may page instead of making
`cudaMalloc` fail, even after a separate process reserves enough VRAM for
`cudaMemGetInfo` to report little or no free memory. The graceful-failure path is
therefore covered by injected model-free tests at the shared
`KvCapacityGrowthBackend` seam: allocation, prefix-copy, mask-allocation, and
capture-invalidation failures assert actionable errors and unchanged session
state. Linux/TCC GPU runners should still exercise real allocator failures when
available.
