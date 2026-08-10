### 2026-08-10: Managed no-spill VMM becomes the default, with automatic weight streaming
**By:** Copilot (coordinator), at the owner's direction
**What:** The managed no-spill VMM path stops being opt-in behind an explicit
`serve --vram-limit` and becomes the default. When a model exceeds the resolved
device budget, the runtime automatically enables weight streaming/offload rather
than failing. The explicit flag becomes an override. Tracked in #755.
**Why:** Owner directive — "managed no-spill vmm要转成默认：而且自动offload所以
超预算也可以通过我们的机制stream weights". Hard OOM was recorded as *correct*
behavior only because there was nothing else to do; with automatic streaming the
answer to "the weights do not fit" is "stream them", and #723's scan-resistant
residency already does that well (hit rate 0% -> 74.18%, evictions 6,286 -> 0).

**Hard prerequisite — #716.** Weight offload and CUDA graph capture are mutually
exclusive today: the pager's alloc/copy/free ops are capture-illegal, so
enabling offload disables capture. Flipping the default before #716 lands would
silently disable graph capture for every model that needs offload — exactly the
models this is meant to help — and would give back the 154 -> 34 segment
collapse won by #708 and #728.

**Sequencing:** (1) #735 strategy inference runs unconditionally, not only when
an explicit limit is present, and reports the plan before applying it; (2) #716
makes offload capture-compatible under stable VA slots; (3) flip the default
with a one-release opt-out; (4) publish an interleaved same-session comparison
in committed physical bytes, page-ins/hit rate/evictions, graph segment count
and decode ms/tok.

**Must not regress:** no silent spill to WDDM shared system memory; capacity
refusal stays a pre-header 429 (#743); public budgets stay committed physical
bytes (a nominal 6 GB content budget physically consumed ~6.51 GB); a model that
fits in VRAM must not start paging just because the managed path is default.
