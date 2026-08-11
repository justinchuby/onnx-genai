# Contiguous-VA-per-sequence KV: crux answered, floor stands

- The decode kernel's **read pattern**, not the VA reservation, forces the
  physical commit. Isolating GPU test
  (`onnx-runtime-cuda-memory/tests/vmm_kv_contiguous_tail_gpu.rs`): a read
  bounded to the live length leaves the reservation tail **uncommitted**; a read
  one byte into the tail faults `CUDA_ERROR_INVALID_VALUE` (non-sticky). This is
  #721 stage 4 in isolation.
- A **fixed full-context stride** (the "one flat VA, never re-strided" ideal)
  pays a `objects x granule` floor because KV is per-binding/head-major:
  qwen2.5-0.5b = 96 head-stripes x 2 MiB = **192 MiB for ~12 KiB of content**
  (1.5 GB at 32K, 32x over bucket growth). A length-bounded kernel is necessary
  but not sufficient — the grow-axis stride sets the floor.
- The landed `kv_commits_on_demand` path (#682/#740/#748) is the right
  realization at **bucket stride**: full-context VA reserved, only the current
  bucket committed, growth keeps `device_ptrs` stable and commits the bucket
  delta (verified on real qwen2.5-0.5b). Committed bytes at **parity with bucket
  growth** + stable VA, but it still re-strides / re-captures on growth.
- Low committed bytes AND no re-capture cannot both hold under head-major
  layout. Unlocking both needs either sub-bucket commit (open: does the ORT GQA
  CUDA kernel touch the bucket tail `[logical_len, bucket)`?) or a seq-major KV
  layout (needs re-exported models / custom kernel).
- Device KV paging is owned by the **CUDA VMM layer** (`CudaVmmAllocator`
  granule mapping under a fixed reservation + governor grants), not by
  `onnx-genai-kv`. The native CUDA decode path has **no** `PageTable`/
  `PagedKvCache` consumer; that machinery stays host-only (CPU paged-GQA, KV
  page store, prefix sharing, quant layout). For the CUDA batch path it is not
  needed as the device pager — a batch is N sequences each with its own stable
  VMM reservation, and the flat GQA kernel is unchanged.
