### 2026-08-09: Explicit mapped-holder reclaim for native CUDA KV growth
**By:** Copilot
**What:** Device memory authorities register weak `ReclaimableMappedHolder` participants. Native CUDA KV preflights mapped growth through the generic provider/authority contract; the authority tentatively shrinks weight allowances, calls eviction outside its claim gate, then verifies or restores accounting.
**Why:** Short requests should lend unused maximum-context KV capacity to reloadable weight mappings without a process-global callback, while recurrent state remains protected and KV never commits before successful weight reclaim.
