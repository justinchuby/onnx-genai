# Transactional mapped growth

- Mapped-capacity lending is coordinated by the shared memory authority through
  weakly registered `ReclaimableMappedHolder`s and RAII `MappedGrowthGrant`s.
- Victim allowance is transferred before callbacks; only
  `mapped_bytes - new_limit` is physically reclaimed.
- Native CUDA KV commits all binding ranges as one granule-union transaction.
- Governed CUDA workspace and native KV use the same requester/grant contract.
- An explicit byte `serve --vram-limit` selects managed no-spill VMM/pool mode;
  `ONNX_GENAI_DYNAMIC_KV_WEIGHT_LENDING=0` restores the non-VMM compatibility
  path. A 6 GiB qwen2.5-14b int4 live run loaded and generated correctly; its
  first physical KV growth transferred and attributed 201,326,592 bytes.
- Managed VMM/pool construction failure is fatal before model allocation;
  compatibility mode alone may fall back to ungoverned `cuMemAlloc`.
- Insufficient reclaim is a typed capacity refusal and maps to HTTP 429 with
  `Retry-After`, never an `InvalidRequest`/500.
