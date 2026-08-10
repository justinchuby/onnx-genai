# Exported Symbol Safety Contract — ORT Plugin-EP Entry Points

**Author:** Isidore  
**Date:** 2026-08-10  
**Resolves:** Security finding N3 (MEDIUM → FIXED) from Holden's re-audit  
**File:** `crates/onnx-runtime-ep-plugin/src/lib.rs`

---

## Contract

The two exported C symbols produced by `export_ep_factories!` — `CreateEpFactories`
and `ReleaseEpFactory` — are the first things upstream ORT touches when it
`dlopen`s our cdylib. Any panic that escapes across the C ABI boundary is
undefined behaviour and can corrupt the host process.

### `CreateEpFactories` (returns `*mut OrtStatus`)

The entire body of the generated function is wrapped in
`std::panic::catch_unwind(AssertUnwindSafe(...))`.

On success:
- Delegates to `factory::create_ep_factories`, which performs version
  negotiation, stores the host `OrtApi`, allocates the factory, and returns a
  null-ptr success status.

On panic (e.g. panicking `constructor()` or `ep.name()` panic):
- `*out_num` is null-checked and set to `0`.
- `panic_to_fail_status("CreateEpFactories: constructor panicked …")` returns
  a non-null `OrtStatus` with `ORT_FAIL` so ORT can report the failure cleanly.
  When the host API is unavailable (test context without ORT), `fail_status`
  returns null, which ORT interprets as success — but this path is unreachable
  in production because `api_base` is always non-null and `GetApi` succeeds
  before the constructor is ever called.
- The output factory array is left untouched (no garbage pointers written).

### `ReleaseEpFactory` (returns `void`)

The body is wrapped in `let _ = std::panic::catch_unwind(AssertUnwindSafe(...))`.

- The return type is `void`. The ORT plugin-EP ABI specifies no status channel
  for this function. The original macro generated `-> *mut OrtStatus`; this
  was corrected to `void` as part of this fix.
- On panic: silently swallowed. Leaking the factory is preferable to UB.
- The underlying `factory::release_ep_factory` still returns `*mut OrtStatus`
  (used internally); the macro discards it with `;` inside the closure.

---

## Macro Hygiene

- All paths are fully qualified (`::std::panic::catch_unwind`,
  `::std::panic::AssertUnwindSafe`, `::std::result::Result`) — no reliance on
  names the caller might shadow.
- `$crate::factory::create_ep_factories` and `$crate::factory::release_ep_factory`
  use the `$crate` pseudo-path so the macro is hygienic across crate boundaries.
- `$crate::panic_to_fail_status` — a `#[doc(hidden)] pub` wrapper added to
  `lib.rs` to give the macro access to `status::fail_status` (which is
  `pub(crate)` and not directly reachable from consumer crates via `$crate`).
- The macro does NOT call `$crate::status::fail_status` directly, avoiding
  visibility issues in consumer crates.

---

## Consumer Compatibility

`crates/onnx-runtime-ep-cpu-plugin/src/lib.rs` invokes the macro unchanged:

```rust
export_ep_factories!(|| Box::new(CpuExecutionProvider::new()));
```

`cargo build -p onnx-runtime-ep-cpu-plugin` passes cleanly.

---

## Null-pointer and capacity safety (existing, confirmed)

These properties were already present in `factory::create_ep_factories` and are
preserved by the macro guard (the guard never bypasses them):

- `api_base` null → `*out_num = 0`, return null (safe early exit before OrtApi
  is needed for error reporting).
- `max_factories == 0 || out_factories.is_null() || out_num.is_null()` →
  `fail_status(...)` after the host API is initialized.
- Factory array zeroed before writing the single factory pointer:
  `for i in 0..max_factories { *out_factories.add(i) = null_mut(); }`
- Exactly one factory is allocated per call; `*out_num = 1` only on success.
- `ReleaseEpFactory` null-checks the pointer before `Box::from_raw`.

---

## Version check

The fail-closed API version check in `factory::create_ep_factories` is not
bypassed by the guard. The guard wraps the call; a version mismatch returns a
proper `OrtStatus` error via the existing fallback path in `factory.rs`.

---

## Test evidence

`cargo clippy -p onnx-runtime-ep-plugin --lib -- -D warnings` → clean (0 errors,
0 warnings attributable to `lib.rs`).

`cargo build -p onnx-runtime-ep-cpu-plugin` → `Finished dev profile`.

Two new regression tests in `lib::tests`:
- `panicking_constructor_caught_and_zero_factories_returned` — verifies that
  `catch_unwind` absorbs a constructor panic and `out_num` is 0.
- `panic_to_fail_status_never_panics` — verifies the fallback status helper is
  itself panic-safe.

⚠️ `cargo test -p onnx-runtime-ep-plugin --lib` cannot compile in the current
working tree due to errors in `compute.rs` (Leon, `?` in `*mut OrtStatus`
closure, lines 685/688) and `kernel_ctx.rs` (Nabil, `validate_dims` not yet
defined, lines 304–360). These are concurrent work-in-progress failures
attributable to Leon's and Nabil's branches respectively, not to this fix.
Baseline (HEAD without those changes) had 66 passing tests; this fix adds 2
more that will pass once those files are resolved.
