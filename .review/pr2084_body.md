**Rescoped.** This opened as a fix for `main`'s two red Windows lanes; #2078 landed the same platform gate ~20 minutes earlier and I merged it in rather than compete with it. Its skip message and idiom are kept verbatim. What remains is the part #2078 does not cover — and one sentence in it that I can show is false.

## The claim I am correcting

#2078's comment on the skip reads:

> The `allowed == cores` guard below would then fail for the platform rather than for a defect.

**That guard does not fail when the restriction is absent. It passes.** It is satisfied by the exact failure it exists to catch.

`restrict_self_to_leader_cpus` gives up silently when the topology is unreadable, and the child's `cores` then falls back to `allowed`:

```rust
let cores = crate::core_topology::require_host_for_placement()
    .map_or(allowed, |topology| /* ... */);   // <- fallback is `allowed`
```

So on an entirely **unrestricted** process the two are trivially equal. Measured, with the narrowing suppressed and the child's topology read forced to its fallback:

```
RealizedWidth { allowed: 8, cores: 8, restricted: false, workers: 4, ... }
```

`allowed == cores` passes there. What follows is worse than a vacuous pass — it is a **misattribution**: the run fails at `workers == cores` and blames **#1780, a resolver defect**, on a host where the resolver was correct and the premise was never established. On a cpuset that already holds one CPU per core, it passes outright.

```
green run  -> tells you nothing
red run    -> points at the wrong file
```

## The fix

**The premise is reported by the process that established it**, not inferred from a consequence. The child prints `restricted=`, and the parent requires it:

- `NXRT_REQUIRE_PLACEMENT_TESTS=1` → **fail**, naming the three causes (unreadable allowed set, unreadable topology, leaderless cpuset);
- otherwise → skip with a stated cause, in #2078's `SKIP <test>:` idiom.

That is the same fail-closed/skip split `require_host_for_placement` and #1916's `DETECTION_SUPPORTED` already draw, and it is why `required` is a parameter rather than a global read.

The `allowed == cores` check stays, with its comment corrected: it still catches a leader set that is not one CPU per core *on a host that answered*, which is a different fault.

## Second change: one spelling of the platform fact

#2078 spells it `cfg!(target_os = "linux")` inline, in two places. This replaces both with `decode_affinity::PROCESS_AFFINITY_MASKING_SUPPORTED`, next to the two `cfg` arms it describes, for the reason `DETECTION_SUPPORTED` is a constant: a caller must be able to tell *"this platform never had the capability"* from *"the call failed here"* without making the call — and when Windows process-wide masking does land, there is one place to change instead of a `grep`.

A constant that lies about a capability is worse than no constant, so `the_masking_capability_constant_agrees_with_what_this_platform_does` asserts it against the implementation that actually compiled, on every lane. It runs on a spawned thread and re-applies the process's own mask, so it cannot leak an affinity into whatever test the runner schedules on that thread next.

## Evidence

Mutation results, not readings. All runs `taskset -c 16-23`, `CARGO_INCREMENTAL=0`, under `scripts/hostlock.sh` with a stated reason. The host was shared throughout — no claim of a quiet machine is made or needed, since none of these are timings.

| # | mutation | expected | observed |
|---|---|---|---|
| baseline | none (merged with `origin/main`) | green | **1796 passed / 0 failed** |
| F1 | narrowing suppressed + topology unreadable, **with** this PR | fails under required mode | **FAILED**: `NXRT_REQUIRE_PLACEMENT_TESTS=1 but the child could not narrow itself to a leader-only cpuset` |
| F1b | same, against the **pre-#2078** test | shows the blind spot | `allowed == cores` **passed**; the run then failed blaming **#1780** |
| F2 | `PROCESS_AFFINITY_MASKING_SUPPORTED = false` | agreement test fails; width test skips | **agreement test FAILED** (`says false but re-applying this process's own mask returned ok=true`); width test skipped, stated cause |
| F3 | const `false` **and** the Linux impl forced to the non-Linux `Err` — a faithful emulation of a Windows lane | stated skip, green | **2 passed**, `SKIP` line printed |
| F3b | same emulation against the **pre-fix** test | reproduces the CI failure | **FAILED**: `restrict the child to leader CPUs: "process-wide CPU affinity masking is only implemented on Linux (no-op)"` — byte-identical to the Windows ARM64 log |

F3b was how I confirmed the Windows diagnosis before #2078 was visible to me; it is kept because it is also the falsifier for the constant, which is new here.

Also: `cargo fmt --all -- --check` clean, `cargo clippy --locked -p onnx-runtime-ep-cpu --all-targets -- -D warnings` clean, `scripts/check_cross_compile.sh` green at **full scope** (`full offline set (aarch64 cross toolchain present)`) — not the FFI-free subset it silently falls back to without the cross toolchain.

## Species sweep

Three call sites reach `set_current_thread_affinity` outside its own module. The other two already do the right thing: production code at `matmul_nbits.rs:4379` logs the `Err` and carries on, and the budget-lane child at `decode_spmd.rs:8772` prints a skip marker with the reason — *"Only Linux implements a process-wide mask, so on other hosts there is no way to manufacture the reduction."* That is this shape, one file over, written before it. No other caller `.expect()`s the capability.

## Limits

- **No Windows target compiles locally**: `ep-cpu` → `ep-api` → `ort-sys`, whose build script bindgens the ORT headers, so `--target x86_64-pc-windows-msvc` dies in the build script before rustc sees this crate; `scripts/check_cross_compile.sh` documents the same Windows exclusion. F3/F3b emulate the platform's *behaviour* on Linux; only this PR's Windows lanes can confirm the compile.
- `main`'s third red lane, **`CLI ORT (Linux x86_64)`**, is unrelated and untouched here: `plugin_ort_e2e::initializer_chain_still_fuses_into_one_claim` fails with `Only one instance of LoggingManager created with InstanceType::Default can exist at any point in time` — already tracked as #2065 (and #1123 on ARM64). Checked before saying so: every `#[test]` in that file that creates an `OrtEnv` holds `ORT_EP_LOCK`, none via a `let _ =` that would drop the guard immediately, and the binary spawns no threads; tests *after* the failure created their own `Env` and passed, which rules out a leaked one.

🤖 Working as Gaff (Code Reviewer / Quality).
