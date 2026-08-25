`main` has been red on **`Rust (Windows ARM64)`** and **`Rust coverage (Windows x86_64)`** since #2059 merged at 02:55Z ([its own run](https://github.com/justinchuby/onnx-genai/actions/runs/32798748066): [Windows ARM64](https://github.com/justinchuby/onnx-genai/actions/runs/32798748066/job/97657606070), [Windows x86_64 coverage](https://github.com/justinchuby/onnx-genai/actions/runs/32798748066/job/97657606068)). Every branch cut after it inherits both failures.

## What is wrong

`a_default_width_pool_on_leader_cpus_uses_every_core_it_was_given` narrows its child process to one CPU per physical core with `decode_affinity::set_current_thread_affinity`. That function has two `cfg` arms: the Linux one calls `sched_setaffinity`, and the other returns

```
Err("process-wide CPU affinity masking is only implemented on Linux (no-op)")
```

**unconditionally, by construction.** The child `.expect()`s it, so off Linux the test cannot do anything but fail:

```
restrict the child to leader CPUs: "process-wide CPU affinity masking is only implemented on Linux (no-op)"
```

The premise of the test — a process narrowed to a leader-only cpuset — is not constructible on those targets at all. Requiring it there can only ever produce a false failure, which is the state that gets a test deleted rather than a lane fixed.

## The fix

`PROCESS_AFFINITY_MASKING_SUPPORTED`, a compile-time constant next to the two implementations, for the reason `core_topology::DETECTION_SUPPORTED` is compile-time (#1916): a caller has to be able to distinguish *"this platform never had the capability"* from *"the call failed here"*, and the first answer must not be obtained by making the call. `restrict_self_to_leader_cpus` now reports whether the restriction is attemptable instead of panicking on a platform fact; a failure where the platform **does** implement masking stays fatal.

A constant that claims a capability is worse than no constant — a caller keying on `false` would skip on a platform that works. So `the_masking_capability_constant_agrees_with_what_this_platform_does` asserts the constant against the `cfg`-selected implementation on every lane, on a spawned thread and re-applying the process's own mask so it cannot leak an affinity into whatever test the runner schedules next.

## Second defect, on Linux, in the same test

The test inferred "the restriction happened" from `allowed == cores` — **a check that is satisfied by the failure it exists to catch.** `restrict_self_to_leader_cpus` gives up silently when the topology is unreadable, and the child's `cores` then falls back to `allowed`, so the equality holds on a completely unrestricted process.

Measured rather than argued. Suppressing the restriction and forcing the child's topology read to its fallback:

```
RealizedWidth { allowed: 8, cores: 8, restricted: false, workers: 4, ... }
```

`allowed == cores` passes there. What the run reports next is a **misattribution**: it fails at `workers == cores` naming #1780 — a resolver defect — on a host where the resolver was correct and the premise was never established. On a cpuset that already holds one CPU per core, it passes outright.

The child now reports `restricted=` and the parent requires it: failing under `NXRT_REQUIRE_PLACEMENT_TESTS=1`, skipping with a stated cause otherwise — the same fail-closed/skip split `require_host_for_placement` draws.

## Evidence

Every claim above is a mutation result, not a reading. All runs `taskset -c 16-23`, `CARGO_INCREMENTAL=0`, under `scripts/hostlock.sh` with a stated reason.

| # | mutation | expected | observed |
|---|---|---|---|
| baseline | none | green | **1783 passed / 0 failed** (merged with `origin/main`) |
| F2 | `PROCESS_AFFINITY_MASKING_SUPPORTED = false` | agreement test fails; width test skips | **agreement test FAILED** (`says false but re-applying this process's own mask returned ok=true`); width test skipped with its stated cause |
| F3 | const `false` **and** the Linux impl forced to the non-Linux `Err` — a faithful emulation of a Windows lane | stated skip, green | **2 passed**, skip line printed |
| F3b | same emulation against **pre-fix** code | reproduces the CI failure | **FAILED**: `restrict the child to leader CPUs: "process-wide CPU affinity masking is only implemented on Linux (no-op)"` — byte-identical to the Windows ARM64 log |
| F1 | restriction suppressed + topology unreadable, **with** the fix | fails under required mode | **FAILED**: `NXRT_REQUIRE_PLACEMENT_TESTS=1 but the child could not narrow itself to a leader-only cpuset` |
| F1b | same, against **pre-fix** code | shows the blind spot | `allowed == cores` **passed**; failed later blaming #1780 |

Also: `cargo fmt --all -- --check` clean, `cargo clippy --locked -p onnx-runtime-ep-cpu --all-targets -- -D warnings` clean, and `scripts/check_cross_compile.sh` green at **full scope** (`full offline set (aarch64 cross toolchain present)`), not the FFI-free subset.

## Limits of this evidence

- **No Windows target is compiled locally.** `onnx-runtime-ep-cpu` → `onnx-runtime-ep-api` → `onnx-genai-ort-sys`, whose build script bindgens the ORT headers, so `--target x86_64-pc-windows-msvc` dies in the build script before rustc sees this crate; `scripts/check_cross_compile.sh` documents the same Windows exclusion. F3/F3b emulate the platform's *behaviour* on Linux; only the Windows lanes on this PR can confirm the *compile*.
- `main`'s third red lane, **`CLI ORT (Linux x86_64)`**, is a **different** defect and is not addressed here: `plugin_ort_e2e::initializer_chain_still_fuses_into_one_claim` fails with `Only one instance of LoggingManager created with InstanceType::Default can exist at any point in time`. That is #2065 (and #1123 on ARM64), already open. Checked before saying so: all 57 `#[test]` fns in `plugin_ort_e2e.rs` that create an `OrtEnv` hold `ORT_EP_LOCK`, none via a `let _ =` that would drop the guard immediately, and the binary spawns no threads -- so the overlap is not an unserialised caller, and tests *after* the failure created their own `Env` and passed, which rules out a leaked one. Consistent with an `Env` whose teardown has not finished when the next `CreateEnv` runs, which the mutex does not wait for.

## Species sweep

Three call sites reach `set_current_thread_affinity` outside its own module, and the other two already do the right thing — production code at `matmul_nbits.rs:4379` logs the `Err` and carries on, and the budget-lane child at `decode_spmd.rs:8772` prints a **skip marker** with the reason, commented *"Only Linux implements a process-wide mask, so on other hosts there is no way to manufacture the reduction."* That is this fix, one file over, written before it. Only the #2059 child `.expect()`s the capability, and no other caller does.

## Review note

This is a fix to another author's test (#2059, @-author unnotified until now — main being red made the fix the priority over the conversation). If the intent was for the leader-cpuset premise to be constructible on Windows, the right change is a Windows implementation of process-wide masking behind that constant, not this skip — say so and I will withdraw this in favour of it.

🤖 Working as Gaff (Code Reviewer / Quality).
