### 2026-08-12: PR #830 revision 2 — ORT Plugin EP workspace/buffer plumbing

**By:** Leon (Engine Dev, KV & Buffers). Independent revision after reviewer
rejection; Deckard (revision 1 author) was locked out and not consulted.

**Scope:** `crates/onnx-runtime-ep-plugin`, plus two new test-only crates.
Everything below is host-tested; **no NVIDIA GPU exists in this environment**
and #768 stays open as the sole hardware tracker.

---

**1. The executor must never call `Kernel::execute()` directly.**

`compute.rs` called `Kernel::execute()` for every dispatch, so any kernel
declaring a `workspace_requirement` (cuBLASLt GEMM, reductions, FlashAttention
scratch) failed. Both the single-node and routed paths now go through
`prepare_workspace` → `execute_with_workspace`. If you add a dispatch site,
route it through `prepare_workspace`; the E2E suite's `WorkspaceAddKernel`
fails loudly if you don't.

Workspaces are allocated **per dispatch**, not cached as `SessionPersistent`.
Caching needs locking, because ORT may `Run()` a single session concurrently
and a shared workspace would be a data race. This is a known perf cost, not a
correctness gap — revisit with a per-stream arena, not with a naive cache.

**2. Device kernels never get host pointers.**

Subgraph intermediates were `Vec<u8>` tagged `DeviceId::cpu()`. They are now
EP-allocated with RAII cleanup and tagged with the device the bytes actually
live on. Two rules worth carrying forward:

- **Zero-fill only host-accessible allocations.** A host `write_bytes` on a
  device pointer is UB. `EpAllocation::new` gates on
  `device_id().is_host_accessible()`.
- **Query ORT for per-tensor device, don't assume `ep.device_id()`.** ORT
  legitimately keeps shape tensors in CPU memory even for a GPU EP, so
  `read_inputs` uses `Value_GetMemoryDevice` / `MemoryDevice_GetDeviceType`.
  Host-accessible EPs skip the query so the CPU path stays byte-identical.

`ScratchAllocator::ensure_allocatable` fails closed when a non-host-accessible
device has no EP allocator, and the routed path calls it *before* any kernel
runs.

**3. Size-zero allocation is normalised at the adapter, not in the allocator.**

`Alloc(0)` must yield a unique, non-null, freeable pointer.
`cudaMalloc(0)`, RMM and PyTorch's caching allocator all disagree about this,
so the normalisation lives in `device.rs::normalize_alloc_size` (mirrored by
`MIN_SCRATCH_BYTES` in `compute.rs`). Do not push it down into an EP: the
falsifier is a mock EP whose `allocate()` *rejects* zero bytes.

**4. Shared-EP teardown: `ReleaseEpFactory` owns explicit shutdown.**

Four ORT surfaces hold `Arc` clones, so no individual `Release*` may call
`shutdown()`. `ReleaseEpFactory` is the only point after all of them.
`release_ep_factory_with_teardown` returns `ShutdownCalled` /
`ShutdownFailed` / `StillReferenced { .. }` / `NotShared` so tests can assert
on the path taken. `StillReferenced` deliberately does **not** shut down and
falls back to the codified Drop-only invariant.

---

**Two hazards other agents will hit.**

**(a) `cargo test --test X` does not rebuild the crate's cdylib.** It builds
the test binary and the rlib only. A deliberately regressed executor still
reported green because the suite loaded a stale `.so` from a previous build.
Any test that dlopens one of our plugins must force the rebuild.
`onnx_runtime_ort_testkit::find_plugin_cdylib` now always runs
`cargo build -p <package>` first (memoised per package;
`NXRT_SKIP_PLUGIN_REBUILD=1` opts out). **If you write a plugin integration
test, use the testkit — do not re-roll the resolver.**

**(b) `.gitignore` blanket-ignores `*.onnx` with a per-path allow-list.**
Copying fixtures into a new crate leaves them silently untracked: green
locally, red in CI. Reference the canonical
`crates/onnx-runtime-ep-cpu-plugin/tests/fixtures/` models via
`CARGO_MANIFEST_DIR/../` instead of copying, or extend the allow-list.

---

**New crates (both `publish = false`, `members` but not `default-members`):**

- `crates/onnx-runtime-ort-testkit` — the single source of truth for ORT and
  plugin-cdylib discovery. Replaced two byte-identical
  `tests/common/ort_discovery.rs` copies and `tests/ort_path.rs`. Add new
  discovery helpers here, not in a test file.
- `crates/onnx-runtime-ep-shared-mock-plugin` — a real shared-EP cdylib that
  ORT loads through `RegisterExecutionProviderLibrary` → `GetEpDevices` →
  `CreateSession` → `Run`. **It is deliberately CPU-typed:**
  `GetSupportedDevices` can only match hardware ORT actually enumerates, so a
  GPU-typed mock is never selected on a GPU-less host and the suite would
  silently degrade into a skip. Use this crate to test the plugin *protocol*;
  it says nothing about device memory.

---

**CI baseline as of this revision (for anyone diagnosing #830):** two jobs are
red on this branch and byte-for-byte identically red on `main`
(run `31631294175`) — `Rust quality → Run clippy on all offline crates`
(`onnx-genai-ort-sys` build.rs cannot download ORT 1.27.0; HTTP 503 / curl 56)
and `CLI ORT (Linux x86_64)` (the same two
`onnx-genai-server/tests/http.rs` JSON-object response-format tests). This
branch touches neither crate. Always do the base comparison before attributing
a red job to a branch.
