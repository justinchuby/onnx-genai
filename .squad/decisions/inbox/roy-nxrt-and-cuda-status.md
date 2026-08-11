# Decision: §524 completion status and honest CUDA position

**Author:** Roy (Lead)
**Date:** 2026-08-11
**HEAD at writing:** `4212e090e`
**Branch:** `squad/ep-plugin-parity-cuda` (draft PR #762)

---

## §524 completion status

Standing directive §524 requires: stable C ABI with dynamic loading, first-class
Rust trait, the two surfaces in sync, ORT ABI evolving toward nxrt, fail-closed.

| Requirement | Status | Evidence |
|---|---|---|
| Stable C ABI with dynamic loading | ✅ Complete | `CreateEpFactories`/`ReleaseEpFactory` exported; ORT `dlopen`s; 23 conformance tests pass |
| First-class Rust trait | ✅ Proven | 9 parity tests in `trait_cabi_parity.rs` confirm trait↔C-ABI agreement |
| Trait↔C-ABI parity rule | ✅ Pinned | `C_ABI_claims = trait_claims ∩ { for_node != Declined }` |
| Fail-closed | ✅ Complete | Shape-inference Declined path + `node_passes_dtype_filter()` dtype gate |
| ORT ABI evolves toward nxrt | ✅ In progress | Plugin adapter is a thin shim; ORT types translated at boundary |
| **Native nxrt dynamic ABI** | 🔴 **NOT IMPLEMENTED** | `crates/onnx-runtime-ep-nxrt-abi/` and `crates/onnx-runtime-ep-nxrt-host/` do not exist |

**§524 is NOT complete.** The native nxrt dynamic ABI gap is unresolved.

---

## Finding: EP-compatibility milestone partially landed (working tree only)

The EP-compatibility milestone was described as delivering `crates/onnx-runtime-ep-nxrt-abi/`
(Nabil) and `crates/onnx-runtime-ep-nxrt-host/` (Isidore), among other work.

On inspection of HEAD `4212e090e`, neither crate was committed. However, the
working tree (from parallel agents) does contain:
- `crates/onnx-runtime-ep-nxrt-abi/` (Nabil) — untracked
- `crates/onnx-runtime-ep-nxrt-host/` (Isidore) — untracked
- `crates/onnx-runtime-ep-nxrt-testplugin/` — untracked
- `crates/onnx-runtime-ep-plugin/src/transfer.rs` (Leon) — untracked

**Critical integration gap found:** Nabil's ABI crate exports `NxrtNegotiate` /
`NxrtCreateEpFactories` with a vtable-based ownership model. Isidore's host loader
expects `nxrt_abi_version` / `nxrt_create_ep` / `nxrt_destroy_ep` / `nxrt_ep_name`
/ `nxrt_device_count` with an opaque-handle model. These protocols are incompatible.
Isidore's `abi_contract.rs` acknowledges this: "When Nabil's crate lands, this
module should be replaced by a re-export from that crate." A reconciliation pass
is required before either crate can be committed. **This is the immediate blocker.**

---

## Honest CUDA position

Justin's hard constraint: do not claim working CUDA without real GPU validation.

- This host has no CUDA toolkit and no GPU.
- No CUDA capability has been validated on hardware.
- `prefetch_lazy_weight` in `crates/onnx-runtime-ep-cuda/src/provider.rs:564` is
  a real stub (Deckard decision: deferred to post-Phase-2a).
- `onnx-runtime-ep-cuda-plugin` exists as a scaffold behind `#[cfg(feature = "cuda")]`
  but is not a working CUDA ORT plugin.
- Five design blockers (context/stream sharing, device-pointer marshaling, cuBLAS
  rebinding, weight paging redesign, graph capture coordination) must be resolved
  before a working CUDA plugin is possible, regardless of hardware availability.

Full table: `docs/CUDA_EP_STATUS.md`.

---

## Action items

1. **Nabil + Isidore:** Reconcile symbol protocol. Isidore's host loader must adopt
   Nabil's `NxrtNegotiate`/`NxrtCreateEpFactories`/vtable model (or vice versa,
   jointly). Resolve before either crate is committed.
2. Commit the resolved nxrt-abi, nxrt-host, and transfer.rs to the branch.
3. Pris: commit the CUDA hardware conformance runner to the branch before any
   hardware-validated claim is made.
4. Deckard: schedule `prefetch_lazy_weight` implementation for post-Phase-2a.
5. PR #762 stays draft until items 1–3 are resolved.
