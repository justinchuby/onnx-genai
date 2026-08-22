# nxmem — the dynamic-plugin memory ABI

Phase 6 of the memory architecture refactor (issue #1186).

A *memory plugin* is a shared library that publishes allocation **mechanisms**
to an ONNX GenAI host. This document is the contract; the machine-readable
version is [`nxmem_memory_abi.h`](../../crates/onnx-runtime-memory-abi/include/nxmem_memory_abi.h),
and the smallest correct implementation is
[`minimal_plugin.c`](../../crates/onnx-runtime-memory-abi/examples/minimal_plugin.c).

## Why an ABI at all

Phases 1–5 moved every allocation decision behind Rust traits
(`DeviceAllocator`, `VirtualBacking`, `SharedMapping`) and put a process-wide
governor in front of them. Those traits are Rust: they use trait objects,
`Arc`, Rust enum layouts, and unwinding, none of which have a stable
representation across a `dlopen` boundary or across compiler versions.

Phase 6 adds a C ABI that expresses the same capabilities, plus a host-side
adapter that wraps a C vtable back into those Rust traits. A plugin therefore
plugs into the *existing* governance machinery without the governor learning
that it is talking to a plugin at all.

```
onnx-runtime-memory-abi        #[repr(C)] structs, vtables, versioning.
   (no dependencies)           Shared verbatim by host and plugin.
        │
        ├── onnx-runtime-memory-host        loads a cdylib, negotiates, and
        │      (libloading)                 wraps its vtables in the traits
        │                                   from onnx-runtime-memory-api
        │
        └── onnx-runtime-memory-testplugin  a cdylib exercising every rule,
               (cdylib)                     including the ones about failing
```

## What may cross the boundary

Only `#[repr(C)]` structs from the ABI crate, by pointer or by value.

Explicitly **never**: a Rust trait object, an `Arc`, a Rust `enum`, a Rust
`String`/`Vec`, ownership of either side's allocator, or a panic. Every
`extern "C"` entry point on both sides wraps its body in `catch_unwind`
(`catch_status_panic` / `catch_void_panic`), so a panic becomes
`NXMEM_STATUS_INTERNAL_ERROR` instead of unwinding into foreign frames, which
is undefined behaviour.

Errors are `NxmemStatus`: a stable `u32` code plus a 256-byte inline message
buffer. Inline, because a heap-allocated message would mean one side freeing
the other side's allocation.

## Versioning and negotiation

`major` is a hard gate. `minor` may only **append** fields, so a larger struct
is always a strict superset of a smaller one.

Current: major 1, minor 1. Baseline supported minor: 0. Minor 1 appends exactly
one allocator slot, `release_allocation`, gated by
`NXMEM_CAP_STRUCTURED_RELEASE`.

The host calls `NxmemNegotiate` once. The agreed minor is a **ceiling**, not an
assignment: each vtable separately declares in its own `abi_minor` what it
really implements, so one module can ship a current mechanism beside one still
written to the baseline. The test plugin does exactly that (`lazy` at minor 1,
`legacy-1-0` at minor 0).

Reading an untrusted vtable goes through `read_prefix`, never a plain
dereference:

1. reject a null or misaligned pointer;
2. read `struct_size` (offset 0) and `abi_minor` (offset 4) unaligned — those
   two fields are at those offsets in every version, forever;
3. reject `struct_size` smaller than the level the sender *claims*: a sender
   contradicting itself is broken, not old;
4. clamp: `effective_minor = min(declared, negotiated)`, and rewrite
   `abi_minor` to it so callers need only consult the value they get back;
5. copy `min(struct_size, size_of::<Self>())` bytes into a zeroed local;
6. null out every slot above the effective level.

Step 4 is what makes an old host usable with a new plugin. Rejecting instead
would mean a plugin could never add a slot without breaking every existing
host, which defeats the purpose of a minor version.

A NULL optional slot means the capability is **absent** and surfaces as
`NXMEM_STATUS_UNSUPPORTED_CAPABILITY`. It is never a silently successful
no-op. The host treats a non-NULL capability vtable behind a *clear* capability
flag as a contract violation rather than a bonus.

## Ownership

| object | created by | released by |
| --- | --- | --- |
| factory vtable | `NxmemCreateAllocatorFactories` | host, via the factory's `release`, exactly once |
| allocator vtable | factory `open_allocator` | host, via the allocator's `release`; `retain` adds a reference |
| virtual-backing / shared-mapping vtable | plugin | the owning allocator — never separately by the host |
| allocation | allocator `allocate` | `deallocate` or `release_allocation` |
| shared prefix | `create_shared_prefix` | matching `release_shared_prefix` |
| host callbacks | host | host, after the last allocator *and* the last queued release retire |

Every pointer passed into a call is borrowed for that call only, with two
stated exceptions: `NxmemOpenRequest::callbacks` is borrowed for the whole
lifetime of the allocator opened with it, and a factory's `name` must outlive
the factory.

The host holds its callback table in a `Box` created *before* `open_allocator`
so its address is stable, and drops it only in `Drop for AllocatorCore`, after
the plugin's `release` has returned.

If the host refuses a vtable that `open_allocator` returned `Ok` for, it still
owes the plugin a `release` — it makes a best-effort call when the struct is
well-formed enough to locate that slot. A plugin publishing a vtable too
malformed to read must therefore not allocate state before returning it.

## Threading, re-entrancy, and locks

Every slot may be called concurrently from any thread; a plugin does its own
locking.

**The rule that matters most: no participant blocks, and no participant holds
one of its own locks, across a call into the other side.**

- The host never calls into a plugin while holding a governance lock. This is
  the same invariant Phases 1–5 established for Rust trait objects
  (`pressure` drops the lock before `on_pressure`; `allocate_with` holds no
  lock while running the caller's closure; `run_drain_callback_if_ready` uses
  `.take()` so the callback runs outside the lock) — an ABI call is strictly
  more dangerous than a trait-object call, so the rule tightens rather than
  relaxes.
- The host's `take_allocation` locks its live map, removes the record, drops
  the guard, and *only then* enters the plugin.
- A plugin must not hold a lock across `request_reclaim`, because the host may
  re-enter the same plugin on the same thread to satisfy it.
- `drain_releases` invokes `release_completed` per retired ticket, in enqueue
  order. Take the batch under your lock, drop the lock, then call the host.

## Deferred release

`enqueue_release` hands an allocation over without freeing it and yields a
ticket. Until that ticket retires:

- the allocation counts as live;
- it pins the allocator, which pins the module;
- the host's callback table stays alive, because `release_completed` will still
  be called through it.

The pinning is at two different granularities and both matter. An
`Arc<PluginModule>` keeps the plugin's *code* mapped; it says nothing about the
host's *callback context*, which is a separate heap object the plugin captured
a raw pointer to at open. A queued release holds both, so an allocator dropped
with releases still outstanding drains what the plugin is willing to retire
and, for anything left, deliberately leaks its callback table rather than free
memory a plugin thread may still write to. `AllocatorCore::leaked_callback_tables`
counts those; a non-zero value is a leak to be fixed at the call site, not a
crash.

**Not yet implemented:** the *provider/context* half. No execution provider is
wired to this ABI in this phase, so "deferred release keeps the provider and
its context pinned" has nothing to pin. Only the intra-boundary half — module
and callback table — is implemented and tested here.

## Unloading a plugin

`MemoryPlugin::try_unload` is the only clean shutdown. It consumes the plugin,
asks both sides what is still live, and hands the plugin back on refusal so the
caller can retire the outstanding work and try again.

**Dropping a `MemoryPlugin` is not a clean shutdown.** Because `try_unload`
consumes `self`, every early return, `?` and unwind reaches `Drop` instead. A
drop has no channel to refuse through, so when the gate is shut its only two
options are to unmap a module whose code a live object may still enter, or to
keep it mapped forever. It keeps it mapped, and counts the event in
`MemoryPlugin::forced_module_leaks`.

Do not rely on the platform to save you here. Whether `dlclose` actually
unmaps is an accident of the loader: glibc unmaps a refcount-zero DSO that is
not marked `DF_1_NODELETE`, and Rust `cdylib`s are not; macOS commonly declines
to unmap. "It did not crash on my machine" is not a safety property.

## Release outcomes

Three states, deliberately not interchangeable:

| state | meaning |
| --- | --- |
| `COMPLETE` | the memory is gone; `unmapped_bytes` may be credited back |
| `QUARANTINED` | the plugin still owns `residual_owned_bytes`; the address must never be reissued and the residue must not be refunded |
| `FAILED` | nothing was mutated; the allocation is as live as the caller left it |

A state code the host does not recognise is treated as **quarantine**. It is
the only interpretation that cannot corrupt either memory or accounting:
guessing `COMPLETE` would reissue live memory, and guessing `FAILED` would
leave the host convinced it still owns something the plugin freed.

## Cross-provider misuse

Every allocation carries `mechanism_id` and `device`. Allocation, release,
range, and shared-prefix calls all check both before acting, and report
`NXMEM_STATUS_WRONG_MECHANISM` / `NXMEM_STATUS_WRONG_DEVICE` rather than
operating on a stranger's pointer.

`allocation_id` is a host-assigned monotonic counter and is **never derived
from the address** — the same reasoning that made `AllocationGeneration`
non-pointer-derived in Phase 1. An address can be reused; an id cannot, so an
ABA race cannot make one allocation impersonate another.

## Unload gating

`NxmemQueryUnloadReadiness` reports live allocators, allocations, views,
capabilities, and queued releases. A *view* is a shared prefix committed into a
live allocation — a plugin-owned window whose life is bounded by the allocation
it looks into, and which the host has no way to count for itself. The host refuses or defers unload while any
count is non-zero, and it checks its own counters too, so a rejection carries
both sides' tallies and a misreporting plugin is visible rather than fatal.

A plugin must count conservatively: reporting zero while anything is reachable
invites the host to unmap code that is about to run.

## Test plugin

`onnx-runtime-memory-testplugin` is a `cdylib` loaded at runtime by the ABI
tests — out-of-tree in the way that matters (`dlopen`, no workspace linking),
in-tree in the way that does not (it lives in the repo so it is built and
linted with everything else). It publishes twelve named mechanisms rather than
switching on environment variables or globals, so the tests select behaviour by
name:

| mechanism | behaviour under test |
| --- | --- |
| `eager` | the minimal conforming mechanism: required slots only |
| `lazy` | virtual backing, shared mapping, deferred release, structured release |
| `short-struct` | a vtable claiming fewer bytes than the baseline prefix, in an allocation that really is that short |
| `poisoned-tail` | a minor-0 vtable whose declared prefix is followed by populated garbage inside the same allocation |
| `missing-slot` | returns `Ok` with real state, then a vtable with a required slot null: tests the release path for a vtable the host cannot even parse |
| `bad-tier` | returns `Ok` with real state, then publishes a device tier no host knows: tests the host's post-`Ok` release obligation |
| `self-retaining` | keeps a reference to its own allocator state after the host lets go, so only the plugin's report can refuse unload |
| `callback-probe` | calls `request_reclaim` and fails cleanly when the host refuses |
| `legacy-1-0` | built to minor 0: an older participant under a newer host |
| `quarantining` | keeps residual ownership on release |
| `future-state` | reports a release state from a later contract level |
| `sticky` | never retires a queued release, so unload stays refused |

It runs entirely on host memory, so the whole suite is portable and needs no
accelerator.

## Relationship to CUDA

Nothing in this phase is CUDA-specific, and no CUDA code changes here. The
mechanisms a CUDA plugin would publish (VMM-backed lazy commit, shared
prefixes, stream-ordered release) are exactly the optional capabilities the ABI
already carries. Phase 7 is where the built-in eager allocator moves behind
this boundary.
