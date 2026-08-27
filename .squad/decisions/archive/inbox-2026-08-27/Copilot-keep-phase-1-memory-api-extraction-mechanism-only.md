### 2026-08-18T14-46-11: Keep Phase 1 memory API extraction mechanism-only
**By:** Copilot
**What:** Keep Phase 1 memory API extraction mechanism-only
**References:** #1186 Phase 1, #1247
**Why:** Phase 1 of #1186 will move only Tier, DeviceKey, AllocationCommitRange, MappedAllocation, SharedDevicePrefix, and SharedPrefixCommitInfo into onnx-runtime-memory-api. DeviceAllocator and HostAllocator remain in onnx-runtime-memory-governor because current method signatures use governor-owned MappedPhysicalCapacityToken and MemoryError. MemoryRole, MemoryError, authority/holder identities, ledgers, allowances, growth grants, leases, pressure responders, shareability analysis, and policy remain governor-owned. Existing governor root and allocator-module paths re-export the moved types. No capability split, identity/lifecycle design, or runtime behavior changes are included.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
