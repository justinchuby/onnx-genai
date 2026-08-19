/*
 * nxmem — the stable dynamic-plugin memory ABI.
 *
 * A memory plugin is a shared library that publishes one or more allocation
 * *mechanisms* to an ONNX GenAI host. This header is the whole contract: a
 * plugin needs nothing else from the workspace, does not link against it, and
 * may be written in C.
 *
 * ─── What crosses the boundary ──────────────────────────────────────────────
 *
 * Only `struct`s declared in this header, and only by pointer or by value.
 * Nothing else may cross: no C++ exception, no Rust panic, no ownership of an
 * allocator's heap, no object whose layout is defined by a language runtime.
 *
 *   - Every struct begins with `uint32_t struct_size`, at offset 0, in every
 *     version, forever. That field is how a reader knows which prefix of the
 *     struct the writer actually filled in.
 *   - Every struct that a version can grow also carries `uint32_t abi_minor`
 *     or is reached through one that does.
 *   - Errors are `NxmemStatus` values: a stable numeric code plus an inline
 *     message buffer. No error ever propagates as an unwind.
 *
 * ─── Versioning ─────────────────────────────────────────────────────────────
 *
 * `major` is a hard compatibility gate: participants that disagree on it do
 * not talk. `minor` only ever *appends* fields to the end of a struct, so a
 * larger struct is always a strict superset of a smaller one.
 *
 * The host calls `NxmemNegotiate` first. The agreed minor is a *ceiling*.
 * Each vtable then declares, in its own `abi_minor`, the level it really
 * implements, so one plugin may ship a current mechanism beside one still
 * written to the baseline. A reader:
 *
 *   1. reads `struct_size` at offset 0 and `abi_minor` at offset 4;
 *   2. rejects a `struct_size` smaller than the level the writer *claims* —
 *     that writer contradicts itself;
 *   3. clamps the effective level to `min(declared, negotiated)`;
 *   4. copies only `min(struct_size, sizeof(local))` bytes into a zeroed
 *     local, so unknown trailing bytes are ignored and slots the writer does
 *     not have read back as NULL;
 *   5. never calls a slot introduced above the effective level.
 *
 * A NULL optional slot means "this capability is absent" and must surface as
 * NXMEM_STATUS_UNSUPPORTED_CAPABILITY. It must never behave as a no-op that
 * silently succeeds.
 *
 * ─── Ownership ──────────────────────────────────────────────────────────────
 *
 *   object                     created by            released by
 *   ─────────────────────────  ────────────────────  ─────────────────────────
 *   factory vtable             NxmemCreateAllocator  host, via factory
 *                              Factories             `release`, exactly once
 *   allocator vtable           factory `open_alloc`  host, via allocator
 *                                                    `release`; `retain` adds
 *                                                    a reference
 *   virtual-backing vtable     plugin, owned by the  the allocator; the host
 *   shared-mapping vtable      allocator             never releases it
 *                                                    separately
 *   allocation                 allocator `allocate`  `deallocate` or
 *                                                    `release_allocation`
 *   shared prefix              `create_shared_prefix` matching
 *                                                    `release_shared_prefix`
 *   host callbacks             host                  host, after the last
 *                                                    allocator and the last
 *                                                    queued release retire
 *
 * Every pointer passed *into* a call is borrowed for that call only, unless
 * this header says otherwise. The two exceptions are stated where they occur:
 * `NxmemOpenRequest::callbacks` is borrowed for the allocator's whole
 * lifetime, and a factory's `name` must outlive the factory.
 *
 * ─── Threading, re-entrancy, and the rule that matters most ─────────────────
 *
 * Every slot may be called concurrently from any thread. A plugin must do its
 * own locking.
 *
 * A plugin MUST NOT block indefinitely inside any slot, and MUST NOT hold one
 * of its own locks across a call back into the host. The host's governance
 * layer serialises accounting behind locks, and a host callback may re-enter
 * this plugin on the same thread. A plugin that holds a lock across
 * `request_reclaim` can therefore deadlock against itself. The host reciprocates
 * this rule: it never calls into a plugin while holding a governance lock.
 *
 * `drain_releases` invokes the host's `release_completed` callback. Retire
 * tickets in enqueue order. Take the batch under your lock, drop the lock, and
 * only then call the host.
 *
 * ─── Deferred release and unload ────────────────────────────────────────────
 *
 * `enqueue_release` hands an allocation to the plugin without freeing it yet
 * and yields a ticket. Until that ticket retires through `drain_releases`, the
 * allocation counts as live: it pins the allocator, and the allocator pins the
 * module.
 *
 * The host calls `NxmemQueryUnloadReadiness` before unloading. While any count
 * in `NxmemUnloadReport` is non-zero the host refuses or defers the unload. A
 * plugin must count conservatively — reporting zero while anything is
 * reachable invites the host to unmap code that is about to run.
 *
 * ─── Required exports ───────────────────────────────────────────────────────
 *
 *   NxmemNegotiate
 *   NxmemCreateAllocatorFactories
 *   NxmemQueryUnloadReadiness
 *
 * All three are required. A library missing any one of them is refused at load
 * — without the third, unload could not be gated at all.
 *
 * Each struct below carries a machine-readable layout annotation naming its
 * size in bytes on a 64-bit target. Those annotations are machine-checked against the Rust definitions by
 * `header_layout_matches_the_rust_definitions` in this crate. Update both
 * sides together or the build fails.
 */

#ifndef NXMEM_MEMORY_ABI_H
#define NXMEM_MEMORY_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── versions ─────────────────────────────────────────────────────────── */

#define NXMEM_ABI_VERSION_MAJOR 1u
#define NXMEM_ABI_VERSION_MINOR 1u
#define NXMEM_ABI_VERSION_MINOR_BASELINE 0u

/* ─── capabilities ─────────────────────────────────────────────────────── */

/** Plain allocate/free. Every mechanism must offer this. */
#define NXMEM_CAP_ALLOCATOR (1ull << 0)
/** Reserve address space now, commit physical pages later. */
#define NXMEM_CAP_VIRTUAL_BACKING (1ull << 1)
/** One physical prefix mapped into several allocations. */
#define NXMEM_CAP_SHARED_MAPPING (1ull << 2)
/** Stream-ordered release that retires later. */
#define NXMEM_CAP_DEFERRED_RELEASE (1ull << 3)
/** Release reporting a structured outcome. Added at minor 1. */
#define NXMEM_CAP_STRUCTURED_RELEASE (1ull << 4)

/* ─── status codes ─────────────────────────────────────────────────────── */

#define NXMEM_STATUS_OK 0u
#define NXMEM_STATUS_VERSION_MISMATCH 1u
#define NXMEM_STATUS_SHORT_STRUCT 2u
#define NXMEM_STATUS_UNSUPPORTED_CAPABILITY 3u
#define NXMEM_STATUS_INVALID_ARGUMENT 4u
#define NXMEM_STATUS_INTERNAL_ERROR 5u
#define NXMEM_STATUS_NOT_IMPLEMENTED 6u
#define NXMEM_STATUS_DEVICE_ERROR 7u
#define NXMEM_STATUS_OUT_OF_MEMORY 8u
#define NXMEM_STATUS_WRONG_DEVICE 9u
#define NXMEM_STATUS_WRONG_MECHANISM 10u
#define NXMEM_STATUS_UNKNOWN_ALLOCATION 11u
#define NXMEM_STATUS_RELEASE_QUARANTINED 12u
#define NXMEM_STATUS_CALLBACK_FAILED 13u
#define NXMEM_STATUS_BUSY 14u

/** Message bytes excluding the NUL terminator. */
#define NXMEM_STATUS_MESSAGE_MAX 255u
/** Size of the inline message buffer, including the NUL terminator. */
#define NXMEM_STATUS_MESSAGE_BUF (NXMEM_STATUS_MESSAGE_MAX + 1u)

/**
 * A status crossing the boundary.
 *
 * By value, with an inline buffer: no allocator is shared, so neither side can
 * free the other's heap. `message` is NUL-terminated UTF-8; only the first
 * `message_len` bytes are meaningful.
 *
 * NXMEM_LAYOUT: NxmemStatus size=264
 */
typedef struct NxmemStatus {
  uint32_t code;
  uint32_t message_len;
  uint8_t message[NXMEM_STATUS_MESSAGE_BUF];
} NxmemStatus;

/* ─── device identity ──────────────────────────────────────────────────── */

#define NXMEM_TIER_DEVICE 0u
#define NXMEM_TIER_HOST 1u
#define NXMEM_TIER_DISK 2u

/**
 * Which memory a mechanism serves.
 *
 * Carried on allocations and on every capability request so a mechanism can
 * reject an object that belongs to a different device or provider rather than
 * quietly acting on it.
 *
 * NXMEM_LAYOUT: NxmemDeviceId size=8
 */
typedef struct NxmemDeviceId {
  uint32_t tier;
  uint32_t index;
} NxmemDeviceId;

/* ─── plain records ────────────────────────────────────────────────────── */

/**
 * An allocation, named well enough to reject cross-provider misuse.
 *
 * `mechanism_id` and `device` must match the mechanism being called.
 * `allocation_id` is a host-assigned monotonic id. It is deliberately not
 * derived from `ptr`: an address can be reused, an id cannot.
 *
 * NXMEM_LAYOUT: NxmemAllocation size=56
 */
typedef struct NxmemAllocation {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t mechanism_id;
  uint64_t allocation_id;
  NxmemDeviceId device;
  uint8_t *ptr;
  uint64_t bytes;
  uint64_t align;
} NxmemAllocation;

/**
 * Half-open byte span inside an allocation.
 *
 * NXMEM_LAYOUT: NxmemByteRange size=16
 */
typedef struct NxmemByteRange {
  uint64_t offset;
  uint64_t bytes;
} NxmemByteRange;

/**
 * A request to allocate.
 *
 * `committed_ranges` is borrowed for the call. An empty list on a lazy
 * mechanism means "reserve address space, commit nothing".
 *
 * NXMEM_LAYOUT: NxmemAllocRequest size=64
 */
typedef struct NxmemAllocRequest {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t mechanism_id;
  uint64_t allocation_id;
  NxmemDeviceId device;
  uint64_t bytes;
  uint64_t align;
  const NxmemByteRange *committed_ranges;
  uint64_t committed_range_count;
} NxmemAllocRequest;

/**
 * What an allocation produced.
 *
 * `owned_bytes` is what the mechanism charged; `mapped_bytes` is what it
 * physically mapped. They differ for lazy mechanisms and the host keeps the
 * two accounting axes apart.
 *
 * NXMEM_LAYOUT: NxmemAllocResult size=32
 */
typedef struct NxmemAllocResult {
  uint32_t struct_size;
  uint32_t reserved;
  uint8_t *ptr;
  uint64_t owned_bytes;
  uint64_t mapped_bytes;
} NxmemAllocResult;

/**
 * One span of one allocation.
 *
 * NXMEM_LAYOUT: NxmemRangeRequest size=80
 */
typedef struct NxmemRangeRequest {
  uint32_t struct_size;
  uint32_t reserved;
  NxmemAllocation allocation;
  NxmemByteRange range;
} NxmemRangeRequest;

#define NXMEM_RELEASE_COMPLETE 0u
#define NXMEM_RELEASE_QUARANTINED 1u
#define NXMEM_RELEASE_FAILED 2u

/**
 * What a release actually did.
 *
 * The three states are not interchangeable:
 *
 *   COMPLETE     the memory is gone and `unmapped_bytes` may be credited back.
 *   QUARANTINED  the plugin still owns `residual_owned_bytes`. The address must
 *                never be reissued and the residue must not be refunded.
 *   FAILED       nothing was mutated. The allocation is exactly as live as the
 *                caller left it, and may be released again.
 *
 * A state code a reader does not recognise must be treated as quarantine: it
 * is the only interpretation that cannot corrupt memory or accounting.
 *
 * NXMEM_LAYOUT: NxmemReleaseOutcome size=296
 */
typedef struct NxmemReleaseOutcome {
  uint32_t struct_size;
  uint32_t state;
  uint64_t allocation_bytes;
  uint64_t unmapped_bytes;
  uint64_t residual_owned_bytes;
  NxmemStatus failure;
} NxmemReleaseOutcome;

/**
 * One retired deferred release, reported to the host.
 *
 * NXMEM_LAYOUT: NxmemReleaseCompletion size=328
 */
typedef struct NxmemReleaseCompletion {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t ticket;
  uint64_t mechanism_id;
  uint64_t allocation_id;
  NxmemReleaseOutcome outcome;
} NxmemReleaseCompletion;

/**
 * A plugin asking the host to make room.
 *
 * NXMEM_LAYOUT: NxmemReclaimRequest size=32
 */
typedef struct NxmemReclaimRequest {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t mechanism_id;
  NxmemDeviceId device;
  uint64_t bytes;
} NxmemReclaimRequest;

/**
 * What the plugin still owns.
 *
 * The host refuses or defers unload while any count is non-zero.
 *
 * NXMEM_LAYOUT: NxmemUnloadReport size=48
 */
typedef struct NxmemUnloadReport {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t live_allocators;
  uint64_t live_allocations;
  uint64_t live_views;
  uint64_t live_capabilities;
  uint64_t queued_releases;
} NxmemUnloadReport;

/**
 * A shared physical prefix.
 *
 * NXMEM_LAYOUT: NxmemSharedPrefixHandle size=64
 */
typedef struct NxmemSharedPrefixHandle {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t mechanism_id;
  uint64_t handle;
  NxmemDeviceId device;
  uint64_t device_ptr;
  uint64_t committed_physical_bytes;
  uint64_t mapped_bytes;
  uint64_t requested_bytes;
} NxmemSharedPrefixHandle;

/**
 * Map a shared prefix into one allocation.
 *
 * NXMEM_LAYOUT: NxmemSharedPrefixCommitRequest size=136
 */
typedef struct NxmemSharedPrefixCommitRequest {
  uint32_t struct_size;
  uint32_t reserved;
  NxmemSharedPrefixHandle prefix;
  NxmemAllocation allocation;
  uint64_t byte_offset;
} NxmemSharedPrefixCommitRequest;

/**
 * What a shared-prefix commit cost.
 *
 * `additional_owned_bytes` is charged once, for the first mapping only; later
 * mappings of the same prefix report zero.
 *
 * NXMEM_LAYOUT: NxmemSharedPrefixCommitInfo size=32
 */
typedef struct NxmemSharedPrefixCommitInfo {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t additional_owned_bytes;
  uint64_t newly_mapped_bytes;
  uint64_t granules;
} NxmemSharedPrefixCommitInfo;

/* ─── host callbacks ───────────────────────────────────────────────────── */

/**
 * What the host lets a plugin call back into.
 *
 * Borrowed for the whole lifetime of every allocator opened with it, and for
 * every queued release naming one of those allocators. The host keeps it alive
 * until the last of them retires. A NULL slot means the host does not offer
 * that callback; the plugin must cope rather than require it.
 *
 * A plugin must not hold any of its own locks across either callback.
 *
 * NXMEM_LAYOUT: NxmemHostCallbacks size=32
 *
 * Every field offset below is part of the contract too, not just the total
 * size. A prefix is only meaningful if the fields inside it stay where they
 * are: MIN_STRUCT_SIZE_MINOR_0 is derived from a field offset, so inserting a
 * field mid-struct would silently move it and every older peer would be
 * misread. Pinning the offsets makes that a build failure.
 * NXMEM_LAYOUT_FIELD: NxmemHostCallbacks.struct_size offset=0
 * NXMEM_LAYOUT_FIELD: NxmemHostCallbacks.abi_minor offset=4
 * NXMEM_LAYOUT_FIELD: NxmemHostCallbacks.host_ctx offset=8
 * NXMEM_LAYOUT_FIELD: NxmemHostCallbacks.request_reclaim offset=16
 * NXMEM_LAYOUT_FIELD: NxmemHostCallbacks.release_completed offset=24
 */
typedef struct NxmemHostCallbacks {
  uint32_t struct_size;
  uint32_t abi_minor;
  void *host_ctx;
  NxmemStatus (*request_reclaim)(void *host_ctx,
                                 const NxmemReclaimRequest *request,
                                 uint64_t *reclaimed_out);
  NxmemStatus (*release_completed)(void *host_ctx,
                                   const NxmemReleaseCompletion *completion);
} NxmemHostCallbacks;

/* ─── vtables ──────────────────────────────────────────────────────────── */

struct NxmemVirtualBackingVtable;
struct NxmemSharedMappingVtable;

/**
 * What the host asks for when opening an allocator.
 *
 * `abi_minor` is the negotiated ceiling. `callbacks` is borrowed for the
 * allocator's whole lifetime, not just this call, and may be NULL.
 *
 * NXMEM_LAYOUT: NxmemOpenRequest size=32
 */
typedef struct NxmemOpenRequest {
  uint32_t struct_size;
  uint32_t abi_minor;
  NxmemDeviceId device;
  uint64_t required_capability_flags;
  const NxmemHostCallbacks *callbacks;
} NxmemOpenRequest;

/**
 * Reserve address space now and commit pages later.
 *
 * Reached through `NxmemAllocatorVtable::virtual_backing`. Owned by the
 * allocator that published it and released with it — never separately.
 *
 * NXMEM_LAYOUT: NxmemVirtualBackingVtable size=72
 *
 * Every field offset below is part of the contract too, not just the total
 * size. A prefix is only meaningful if the fields inside it stay where they
 * are: MIN_STRUCT_SIZE_MINOR_0 is derived from a field offset, so inserting a
 * field mid-struct would silently move it and every older peer would be
 * misread. Pinning the offsets makes that a build failure.
 * NXMEM_LAYOUT_FIELD: NxmemVirtualBackingVtable.struct_size offset=0
 * NXMEM_LAYOUT_FIELD: NxmemVirtualBackingVtable.abi_minor offset=4
 * NXMEM_LAYOUT_FIELD: NxmemVirtualBackingVtable.mechanism_id offset=8
 * NXMEM_LAYOUT_FIELD: NxmemVirtualBackingVtable.ctx offset=16
 * NXMEM_LAYOUT_FIELD: NxmemVirtualBackingVtable.allocate_committed offset=24
 * NXMEM_LAYOUT_FIELD: NxmemVirtualBackingVtable.commit_range offset=32
 * NXMEM_LAYOUT_FIELD: NxmemVirtualBackingVtable.decommit_range offset=40
 * NXMEM_LAYOUT_FIELD: NxmemVirtualBackingVtable.mapped_bytes_for_ranges offset=48
 * NXMEM_LAYOUT_FIELD: NxmemVirtualBackingVtable.mapped_bytes_for_allocation offset=56
 * NXMEM_LAYOUT_FIELD: NxmemVirtualBackingVtable.committed_bytes offset=64
 */
typedef struct NxmemVirtualBackingVtable {
  uint32_t struct_size;
  uint32_t abi_minor;
  uint64_t mechanism_id;
  void *ctx;
  NxmemStatus (*allocate_committed)(void *ctx, const NxmemAllocRequest *request,
                                    NxmemAllocResult *result_out);
  NxmemStatus (*commit_range)(void *ctx, const NxmemRangeRequest *request);
  NxmemStatus (*decommit_range)(void *ctx, const NxmemRangeRequest *request,
                                uint64_t *unmapped_out);
  NxmemStatus (*mapped_bytes_for_ranges)(void *ctx,
                                         const NxmemRangeRequest *requests,
                                         uint64_t count, uint64_t *mapped_out);
  NxmemStatus (*mapped_bytes_for_allocation)(void *ctx,
                                             const NxmemAllocRequest *request,
                                             uint64_t *mapped_out);
  NxmemStatus (*committed_bytes)(void *ctx, const NxmemAllocation *allocation,
                                 uint64_t *committed_out);
} NxmemVirtualBackingVtable;

/**
 * One physical prefix mapped into several allocations.
 *
 * Owned by the allocator that published it, exactly as virtual backing is.
 *
 * NXMEM_LAYOUT: NxmemSharedMappingVtable size=64
 *
 * Every field offset below is part of the contract too, not just the total
 * size. A prefix is only meaningful if the fields inside it stay where they
 * are: MIN_STRUCT_SIZE_MINOR_0 is derived from a field offset, so inserting a
 * field mid-struct would silently move it and every older peer would be
 * misread. Pinning the offsets makes that a build failure.
 * NXMEM_LAYOUT_FIELD: NxmemSharedMappingVtable.struct_size offset=0
 * NXMEM_LAYOUT_FIELD: NxmemSharedMappingVtable.abi_minor offset=4
 * NXMEM_LAYOUT_FIELD: NxmemSharedMappingVtable.mechanism_id offset=8
 * NXMEM_LAYOUT_FIELD: NxmemSharedMappingVtable.ctx offset=16
 * NXMEM_LAYOUT_FIELD: NxmemSharedMappingVtable.create_shared_prefix offset=24
 * NXMEM_LAYOUT_FIELD: NxmemSharedMappingVtable.retain_shared_prefix offset=32
 * NXMEM_LAYOUT_FIELD: NxmemSharedMappingVtable.release_shared_prefix offset=40
 * NXMEM_LAYOUT_FIELD: NxmemSharedMappingVtable.incremental_owned_bytes offset=48
 * NXMEM_LAYOUT_FIELD: NxmemSharedMappingVtable.commit_shared_prefix offset=56
 */
typedef struct NxmemSharedMappingVtable {
  uint32_t struct_size;
  uint32_t abi_minor;
  uint64_t mechanism_id;
  void *ctx;
  NxmemStatus (*create_shared_prefix)(void *ctx, uint64_t mechanism_id,
                                      uint64_t bytes,
                                      NxmemSharedPrefixHandle *handle_out);
  NxmemStatus (*retain_shared_prefix)(void *ctx,
                                      const NxmemSharedPrefixHandle *handle);
  NxmemStatus (*release_shared_prefix)(void *ctx,
                                       const NxmemSharedPrefixHandle *handle);
  NxmemStatus (*incremental_owned_bytes)(void *ctx,
                                         const NxmemSharedPrefixHandle *handle,
                                         uint64_t *bytes_out);
  NxmemStatus (*commit_shared_prefix)(
      void *ctx, const NxmemSharedPrefixCommitRequest *request,
      NxmemSharedPrefixCommitInfo *info_out);
} NxmemSharedMappingVtable;

/**
 * One opened allocation mechanism.
 *
 * `allocate`, `deallocate`, `retain`, and `release` are required at every
 * level; a vtable missing one is refused. Everything else is optional and NULL
 * when unsupported. `release_allocation` exists only at minor 1 and above and
 * only when NXMEM_CAP_STRUCTURED_RELEASE is set.
 *
 * `name` must outlive the allocator. `ctx` is opaque to the host and is passed
 * back unchanged to every slot.
 *
 * NXMEM_LAYOUT: NxmemAllocatorVtable size=128
 *
 * Every field offset below is part of the contract too, not just the total
 * size. A prefix is only meaningful if the fields inside it stay where they
 * are: MIN_STRUCT_SIZE_MINOR_0 is derived from a field offset, so inserting a
 * field mid-struct would silently move it and every older peer would be
 * misread. Pinning the offsets makes that a build failure.
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorVtable.struct_size offset=0
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorVtable.abi_minor offset=4
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorVtable.mechanism_id offset=8
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorVtable.device offset=16
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorVtable.capability_flags offset=24
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorVtable.name offset=32
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorVtable.ctx offset=40
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorVtable.allocate offset=48
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorVtable.deallocate offset=56
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorVtable.retain offset=64
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorVtable.release offset=72
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorVtable.virtual_backing offset=80
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorVtable.shared_mapping offset=88
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorVtable.enqueue_release offset=96
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorVtable.drain_releases offset=104
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorVtable.pending_release_count offset=112
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorVtable.release_allocation offset=120
 */
typedef struct NxmemAllocatorVtable {
  uint32_t struct_size;
  uint32_t abi_minor;
  uint64_t mechanism_id;
  NxmemDeviceId device;
  uint64_t capability_flags;
  const uint8_t *name;
  void *ctx;

  /* required at minor 0 */
  NxmemStatus (*allocate)(void *ctx, const NxmemAllocRequest *request,
                          NxmemAllocResult *result_out);
  NxmemStatus (*deallocate)(void *ctx, const NxmemAllocation *allocation,
                            uint64_t *unmapped_bytes_out);
  void (*retain)(void *ctx);
  void (*release)(void *ctx);

  /* optional at minor 0 */
  const struct NxmemVirtualBackingVtable *virtual_backing;
  const struct NxmemSharedMappingVtable *shared_mapping;
  NxmemStatus (*enqueue_release)(void *ctx, const NxmemAllocation *allocation,
                                 uint64_t *ticket_out);
  NxmemStatus (*drain_releases)(void *ctx, uint64_t max, uint64_t *retired_out);
  NxmemStatus (*pending_release_count)(void *ctx, uint64_t *count_out);

  /* added at minor 1 */
  NxmemStatus (*release_allocation)(void *ctx,
                                    const NxmemAllocation *allocation,
                                    NxmemReleaseOutcome *outcome_out);
} NxmemAllocatorVtable;

/**
 * A named, device-scoped mechanism the host may open.
 *
 * Owned by the host once `NxmemCreateAllocatorFactories` returns it: the host
 * calls `release` exactly once, after the last allocator opened from it has
 * been released. `name` must outlive the factory.
 *
 * NXMEM_LAYOUT: NxmemAllocatorFactoryVtable size=56
 *
 * Every field offset below is part of the contract too, not just the total
 * size. A prefix is only meaningful if the fields inside it stay where they
 * are: MIN_STRUCT_SIZE_MINOR_0 is derived from a field offset, so inserting a
 * field mid-struct would silently move it and every older peer would be
 * misread. Pinning the offsets makes that a build failure.
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorFactoryVtable.struct_size offset=0
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorFactoryVtable.abi_minor offset=4
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorFactoryVtable.name offset=8
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorFactoryVtable.device offset=16
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorFactoryVtable.capability_flags offset=24
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorFactoryVtable.ctx offset=32
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorFactoryVtable.open_allocator offset=40
 * NXMEM_LAYOUT_FIELD: NxmemAllocatorFactoryVtable.release offset=48
 */
typedef struct NxmemAllocatorFactoryVtable {
  uint32_t struct_size;
  uint32_t abi_minor;
  const uint8_t *name;
  NxmemDeviceId device;
  uint64_t capability_flags;
  void *ctx;
  NxmemStatus (*open_allocator)(void *ctx, const NxmemOpenRequest *request,
                                const NxmemAllocatorVtable **allocator_out);
  void (*release)(void *ctx);
} NxmemAllocatorFactoryVtable;

/* ─── negotiation ──────────────────────────────────────────────────────── */

/**
 * An inclusive version range a participant supports.
 *
 * NXMEM_LAYOUT: NxmemVersionRange size=16
 */
typedef struct NxmemVersionRange {
  uint32_t major_min;
  uint32_t major_max;
  uint32_t minor_min;
  uint32_t minor_max;
} NxmemVersionRange;

/**
 * What the host offers.
 *
 * NXMEM_LAYOUT: NxmemNegotiateRequest size=32
 */
typedef struct NxmemNegotiateRequest {
  uint32_t struct_size;
  uint32_t reserved;
  NxmemVersionRange host_range;
  uint64_t host_capability_flags;
} NxmemNegotiateRequest;

/**
 * What the plugin agrees to.
 *
 * `capability_flags` must be a subset of what the host offered: a plugin may
 * not advertise a capability the host never named, and the agreed minor may
 * not exceed either side's range.
 *
 * NXMEM_LAYOUT: NxmemNegotiateResponse size=40
 */
typedef struct NxmemNegotiateResponse {
  uint32_t struct_size;
  uint32_t agreed_major;
  uint32_t agreed_minor;
  uint32_t reserved;
  NxmemVersionRange plugin_range;
  uint64_t capability_flags;
} NxmemNegotiateResponse;

/* ─── required exports ─────────────────────────────────────────────────── */

/**
 * Agree a version and capability set. Called once, before anything else.
 *
 * Neither pointer may be retained past the call.
 */
NxmemStatus NxmemNegotiate(const NxmemNegotiateRequest *request,
                           NxmemNegotiateResponse *response_out);

/**
 * Publish the mechanisms this module offers.
 *
 * Writes at most `max_factories` pointers and the real count to `out_count`.
 * Each pointer becomes owned by the host, which releases it exactly once.
 * Writing more than `max_factories` is a contract violation.
 */
NxmemStatus
NxmemCreateAllocatorFactories(const NxmemAllocatorFactoryVtable **out_factories,
                              uint64_t max_factories, uint64_t *out_count);

/**
 * Report what this module still owns, so the host can gate unload.
 *
 * Count conservatively. A module that reports zero while anything is still
 * reachable will be unmapped while its own code is about to run.
 */
NxmemStatus NxmemQueryUnloadReadiness(NxmemUnloadReport *report_out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NXMEM_MEMORY_ABI_H */
