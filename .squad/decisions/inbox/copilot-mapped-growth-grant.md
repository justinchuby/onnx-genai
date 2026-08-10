# Transactional mapped growth

- Mapped-capacity lending is coordinated by the shared memory authority through
  weakly registered `ReclaimableMappedHolder`s and RAII `MappedGrowthGrant`s.
- Victim allowance is transferred before callbacks; only
  `mapped_bytes - new_limit` is physically reclaimed.
- Native CUDA KV commits all binding ranges as one granule-union transaction.
- Governed CUDA workspace and native KV use the same requester/grant contract.
- `serve --vram-limit` with weight offload enables the VMM/pool path unless
  `ONNX_GENAI_DYNAMIC_KV_WEIGHT_LENDING=0`.
