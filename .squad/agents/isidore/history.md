# Isidore — History (compacted 2026-08-12)

**Role:** Mobile & Bindings Engineer — C ABI, Python (PyO3), Swift/Kotlin bindings, mobile/edge cross-compilation and packaging. CPU & Edge pod.

## Durable lessons
- nxrt plugin ABI: vtable model (`NxrtNegotiate` → `NxrtCreateEpFactories`); opaque-handle model is wrong and was deleted.
- `NxrtStatus.code` is raw `u32` (wire type); safe accessor `status_code() -> Option<NxrtStatusCode>` via `from_u32()`. Unknown discriminants → `None`. Transmute UB on unknown enum discriminant is forbidden.
- `struct_size` must be validated before accessing any vtable slot. Check `>= offset_of!(field) + size_of!(field_type)` before dereferencing.
- ORT host API must be set via `set_host_api(api)` from `api_base` **before** calling any diagnostic or error-reporting function. `fail_status()` returns null without it.
- CUDA plugin uses `api_base` field, not the EP-specific `api` parameter.
- **Prefer leaking to calling through an unvalidated vtable pointer.** Undersized `struct_size` means `release` may not exist. Deliberate leak > arbitrary code execution.
- `std::mem::offset_of!(Struct, field)` is authoritative over hand-computed arithmetic offsets.
- CUDA `end_version i32::MAX` is correct for version-agnostic dispatchers; `99` is an arbitrary cap that silently under-claims when ONNX opset exceeds 99.
- CUDA `__CUDA_ARCH__ < 530` fallback: use `memcpy` into named `half2` locals, not `reinterpret_cast<half2*>(&member)` member-punning.
- `#[path = "common/ort_discovery.rs"] mod ort_discovery;` avoids the extra test-binary problem from `tests/common/mod.rs`.

## Historical context (pre-2026-08-12)
Wave coverage through 2026-08-11: nxrt host loader (§524), ABI reconciliation (replaced duplicate `abi_contract.rs`), N3 panic-guard, ABI correctness (enum UB, struct_size, CUDA status loss), iOS_CI infra flake triage, PR #32001 S1/S2/S3 lockout revision, PR #32003 strict-aliasing complete fix. Full detail in `history-archive.md`.

## Recent entries

## 2026-08-12 — PR #762 CUDA version bounds + nxrt struct_size hardening

1. CUDA registry `end_version` → `i32::MAX` with per-family justification.
2. `memoffset_of_create_ep()` → `std::mem::offset_of!(NxrtEpFactoryVtable, create_ep)`.
3. Release guard in `NxrtExecutionProvider::drop`: validates struct_size covers `release`+`ctx`; undersized → deliberate leak. `undersized_factory_vtable_skips_release` test: non-vacuous (both atomic-flag arms exercised).

Tests: 283 passed, 0 failed across 5 EP crates. Commit: `7a2268021`.

## 2026-08-12 — PR #762 ready for review (CUDA bounds + vtable hardening)

All items confirmed by Gaff. 283 passed / 0 failed. PR #762 marked ready for review.

*Full pre-2026-08-12 history in `history-archive.md`.*
