# EP Plugin Conformance — Final Status

**Date:** 2026-08-10  
**Author:** Pris (Tester)  
**Branch:** squad/ep-plugin-export  
**Commit context:** Deckard's use-after-free fix (c92838dba)

---

## What is now proven about upstream-ORT plugin-EP compatibility

The following properties are verified by real ORT loading our cdylib via
`RegisterExecutionProviderLibrary` / `GetEpDevices` / `CreateSession` / `Run`:

| Property | Test | Result |
|---|---|---|
| ORT can dlopen our cdylib and get the EP factory | `ort_register_ep_library` | ✅ |
| Full register → Run → output correct (f32 Add [1,4]) | `ort_loads_our_ep_and_runs_model` | ✅ |
| Two back-to-back Run calls on one session — no state corruption | `conformance_multiple_run_calls` | ✅ |
| Two independent sessions (add_1x4, add_broadcast) correct in isolation | `conformance_two_sessions` | ✅ |
| Broadcast Add [2,3]+[3] | `conformance_add_broadcast` | ✅ |
| Multi-node fused subgraph Add+Mul | `conformance_chain_add_mul` | ✅ |
| 2-D MatMul [2,3]×[3,2] | `conformance_matmul_2d` | ✅ |
| **Batched 3-D MatMul [2,3,4]×[2,4,2]** | `conformance_matmul_batched_nd` | ✅ |
| INT32 Add | `conformance_add_int32` | ✅ |
| Dynamic first dimension (-1, 4) | `conformance_add_dynamic_dim` | ✅ |
| Mixed partition (Add via our EP, NonZero via ORT fallback) | `conformance_mixed_partition` | ✅ |
| 25 complete register→Run→unregister cycles, no corruption | `stress_register_run_unregister_cycles` | ✅ |

All 15 tests pass in two independent consecutive runs. Order-dependence was
previously a concern (corruption appeared at cycle ≥6 in full-suite runs);
the stress test alone runs 25 cycles consecutively and verifies every output.

---

## Key correctness finding: `EpDevice_EpName` vs registration name

`EpDevice_EpName` (ORT C API) returns the name the factory declares via
`OrtEpFactory::GetName` — i.e. `"cpu_ep"` for our factory. It does **not**
return the registration key passed to `RegisterExecutionProviderLibrary`.
The two names are orthogonal: the registration key is an application-level
identifier; the EP name is a factory-level declaration.

`conformance_two_sessions` had a test bug: it compared `EpDevice_EpName`
against the registration key `"cpu_ep_2sess"` instead of the factory name
`"cpu_ep"`. Fixed in this session (test file only; no implementation change).

---

## Coverage gaps (not yet provable)

### f16 / bf16 via ORT plugin path

The CPU kernel layer (`crates/onnx-runtime-ep-cpu`) does implement f16 and
bf16 for Add and MatMul (see `kernels/add.rs:285`, `kernels/half_gemm.rs`).
However, ORT routes nodes to our EP only when `GetCapability` claims them.
Our EP does not currently register explicit half-dtype type-constraint metadata
with ORT's node-capability API (no `GetKernelRegistry` implementation in
`crates/onnx-runtime-ep-plugin/src/ep.rs`). Consequently it cannot be reliably
proven that ORT will dispatch an f16/bf16 Add or MatMul node to our EP in an
end-to-end ONNX model test.

**Owner of fix if desired:** Nabil / ep.rs + factory.rs  
**Pris action when fix lands:** add `conformance_add_f16` and `conformance_matmul_bf16` tests.

### Non-square / non-power-of-two MatMul shapes

Only [2,3]×[3,2] (2-D) and [2,3,4]×[2,4,2] (3-D) are covered. Extreme-K
(K≫M,N), M=1 decode path, and very large K (memory-limited) shapes are
untested via ORT. These are covered at the unit level in `onnx-runtime-ep-cpu`
but not through the full ORT dispatch path.
