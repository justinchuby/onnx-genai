# nxrt ABI and CUDA EP §524 Status — Roy (2026-08-11, HEAD 99560c876)

## §524 Requirement Status

| Requirement | Status |
|---|---|
| Stable C ABI with dynamic loading | ✅ Complete — `CreateEpFactories`/`ReleaseEpFactory` exports; proven by 23 ORT conformance tests |
| First-class Rust trait (proven) | ✅ Proven — 9 parity tests in `trait_cabi_parity.rs` |
| Trait↔C-ABI parity rule | ✅ `C_ABI_claims = trait_claims ∩ { for_node != Declined }`. Pinned and tested. |
| Fail-closed | ✅ Shape-inference Declined path + dtype filter + nxrt `NXRT_CAP_KNOWN_MASK` reject |
| Native nxrt dynamic ABI | 🟡 **Committed but not fully green.** 30/30 ABI unit tests pass. 1 host round-trip test failing (`full_lifecycle_negotiate_create_release` — env-var race between parallel negative tests). Pris's fixture isolation fix not yet landed. |

## Honest CUDA Status

No CUDA capability has been validated on hardware. This host has no GPU and no
CUDA toolkit. The VALIDATED-ON-HARDWARE column in `docs/CUDA_EP_STATUS.md` is
empty for every row. `scripts/cuda_conformance_runner.sh` is committed at
`99560c876` and ready to run on a GPU host; it has not been run here (exit code
would be 2 = UNVALIDATED).

## Single Source of Truth Rule (new lesson)

Consumers of the nxrt dynamic ABI must depend on `onnx-runtime-ep-nxrt-abi`.
Nobody redefines the contract locally. The failure mode — private `abi_contract.rs`
with a different protocol — was structurally undetectable by `cargo check` because
the testplugin was not a workspace member. Both conditions are now corrected.

## What Remains Before PR #762 Exits Draft

1. **Pris:** Fix `full_lifecycle_negotiate_create_release` test — isolate `NXRT_TEST_PANIC` / `NXRT_TEST_FACTORY_ERROR` env vars from parallel tests (use `--test-threads=1` or per-test env isolation or a temp env guard).
2. **Hardware validation:** Run `./scripts/cuda_conformance_runner.sh` on a real GPU host and record output. Until then, CUDA claims remain IMPLEMENTED/COMPILE-CHECKED only.
