# Native and ORT KV capacity unification

The runtime uses one KV capacity policy for ORT shared-buffer KV and native CUDA
KV: `onnx_genai_kv::kv_capacity_bucket(required, hard_max)`. The policy rounds
demand to a power-of-two bucket, applies the `ONNX_GENAI_KV_MIN_BUCKET` floor,
and clamps to a hard maximum derived from model metadata and device capacity.

## Hard maximum

The hard maximum is not a model-family constant. It comes from explicit caller
configuration first (`load_with_cuda_kv_max_len`, `ONNX_GENAI_CUDA_KV_MAX_LEN`),
otherwise from `model.max_sequence_length`, clamped by a queried CUDA
free-memory growth budget using the graph's actual KV bytes per token. The
growth budget accounts for the allocation-before-free sequence used by both
backends: at a bucket boundary the old bucket remains live while the larger
bucket and mask are allocated and filled, so the default ceiling is the largest
capacity whose worst bucket transition fits inside the headroom budget. If
neither metadata nor a CUDA memory query can provide a limit, native CUDA fails
during load with an actionable configuration error instead of guessing.

## Growth

Both ORT shared-buffer KV and native CUDA grow on demand by the shared bucket
policy instead of pre-allocating the full context. The orchestration lives in
`onnx_genai_kv::ensure_kv_capacity` over the `KvCapacityGrowthBackend` trait:
reject above the hard maximum, compute the next `kv_capacity_bucket`, build all
fallible replacement state while the old state remains live, invalidate graph
capture, then commit the new capacity. ORT implements the primitive seam with
`OrtValue` allocation/rebind and `grow_kv_value`; native CUDA implements it with
`DeviceIoBinding` allocation, direct device-to-device prefix copies, and mask
rewrites. Adding another backend should implement those primitives rather than
re-derive the growth policy.

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
