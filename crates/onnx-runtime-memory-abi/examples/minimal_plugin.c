/*
 * A minimal nxmem memory plugin, in portable C.
 *
 * It implements exactly one mechanism, "minimal", with the required slots and
 * no optional capability at all. That is deliberate: it is the smallest thing
 * that is a *correct* plugin, so it doubles as documentation of what is truly
 * mandatory.
 *
 * It needs nothing from the ONNX GenAI workspace but the header next door, and
 * links against nothing. Build it standalone:
 *
 *   cc -shared -fPIC -I../include -o libminimal_plugin.so minimal_plugin.c
 *
 * `nxmem_c_example_compiles` in this crate compiles this file on every unix
 * test run, so the header cannot drift away from a real C consumer unnoticed.
 *
 * What this example is careful about, because a plugin that gets these wrong
 * corrupts the host rather than merely failing:
 *
 *   - it fills in `struct_size` on everything it writes;
 *   - it rejects an allocation whose `mechanism_id` or `device` is not its own,
 *     which is what stops one provider's pointer being freed by another;
 *   - it never blocks, never takes a lock across a host callback, and never
 *     lets an error escape as anything but a status;
 *   - it counts live objects and reports them from NxmemQueryUnloadReadiness,
 *     so the host will not unmap this code while it is still in use.
 */

#include "nxmem_memory_abi.h"

#include <stdlib.h>
#include <string.h>

/* ─── status helpers ───────────────────────────────────────────────────── */

static NxmemStatus nxmem_status(uint32_t code, const char *message) {
  NxmemStatus status;
  memset(&status, 0, sizeof(status));
  status.code = code;
  if (message != NULL) {
    size_t len = strlen(message);
    if (len > NXMEM_STATUS_MESSAGE_MAX) {
      len = NXMEM_STATUS_MESSAGE_MAX;
    }
    memcpy(status.message, message, len);
    status.message_len = (uint32_t)len;
  }
  return status;
}

static NxmemStatus nxmem_ok(void) { return nxmem_status(NXMEM_STATUS_OK, NULL); }

/* ─── module state ─────────────────────────────────────────────────────── */

#define MINIMAL_MECHANISM_ID 0x6D696E31ull /* "min1" */

/*
 * Live-object counts. A real plugin would make these atomic; this example is
 * kept single-file and dependency-free, and the host serialises its own calls
 * during load and unload.
 */
static uint64_t g_live_allocators;
static uint64_t g_live_allocations;

static const char kMechanismName[] = "minimal";

/* ─── allocator ────────────────────────────────────────────────────────── */

/*
 * `identity_ok` is the whole cross-provider-misuse defence. An allocation
 * describes which mechanism and which device it belongs to; a mechanism that
 * skips this check will happily free a pointer that belongs to someone else.
 */
static NxmemStatus identity_ok(const NxmemAllocation *allocation) {
  if (allocation == NULL) {
    return nxmem_status(NXMEM_STATUS_INVALID_ARGUMENT,
                        "minimal: null allocation");
  }
  if (allocation->mechanism_id != MINIMAL_MECHANISM_ID) {
    return nxmem_status(NXMEM_STATUS_WRONG_MECHANISM,
                        "minimal: that allocation belongs to another mechanism");
  }
  if (allocation->device.tier != NXMEM_TIER_HOST ||
      allocation->device.index != 0) {
    return nxmem_status(NXMEM_STATUS_WRONG_DEVICE,
                        "minimal: that allocation belongs to another device");
  }
  return nxmem_ok();
}

static NxmemStatus minimal_allocate(void *ctx, const NxmemAllocRequest *request,
                                    NxmemAllocResult *result_out) {
  void *ptr = NULL;
  size_t align = 0;
  size_t bytes = 0;

  (void)ctx;
  if (request == NULL || result_out == NULL) {
    return nxmem_status(NXMEM_STATUS_INVALID_ARGUMENT,
                        "minimal: null pointer in allocate");
  }
  if (request->mechanism_id != MINIMAL_MECHANISM_ID) {
    return nxmem_status(NXMEM_STATUS_WRONG_MECHANISM,
                        "minimal: allocate called on another mechanism");
  }

  align = (size_t)request->align;
  if (align < sizeof(void *)) {
    align = sizeof(void *);
  }
  /* aligned_alloc requires a size that is a multiple of the alignment. */
  bytes = (size_t)request->bytes;
  if (bytes == 0) {
    bytes = align;
  }
  bytes = ((bytes + align - 1) / align) * align;

  ptr = aligned_alloc(align, bytes);
  if (ptr == NULL) {
    return nxmem_status(NXMEM_STATUS_OUT_OF_MEMORY,
                        "minimal: the host allocator refused the request");
  }

  memset(result_out, 0, sizeof(*result_out));
  result_out->struct_size = (uint32_t)sizeof(*result_out);
  result_out->ptr = (uint8_t *)ptr;
  /* This mechanism is eager: everything it owns, it has also mapped. */
  result_out->owned_bytes = (uint64_t)bytes;
  result_out->mapped_bytes = (uint64_t)bytes;

  g_live_allocations += 1;
  return nxmem_ok();
}

static NxmemStatus minimal_deallocate(void *ctx,
                                      const NxmemAllocation *allocation,
                                      uint64_t *unmapped_bytes_out) {
  NxmemStatus status;

  (void)ctx;
  if (unmapped_bytes_out == NULL) {
    return nxmem_status(NXMEM_STATUS_INVALID_ARGUMENT,
                        "minimal: null out-parameter in deallocate");
  }
  status = identity_ok(allocation);
  if (status.code != NXMEM_STATUS_OK) {
    return status;
  }

  free(allocation->ptr);
  *unmapped_bytes_out = allocation->bytes;
  if (g_live_allocations > 0) {
    g_live_allocations -= 1;
  }
  return nxmem_ok();
}

/*
 * This example opens one shared, statically allocated allocator, so `retain`
 * and `release` only move a count. A plugin with per-open state would free that
 * state when the count reaches zero — and only then, and only if no queued
 * release still names it.
 */
static void minimal_retain(void *ctx) {
  (void)ctx;
  g_live_allocators += 1;
}

static void minimal_release(void *ctx) {
  (void)ctx;
  if (g_live_allocators > 0) {
    g_live_allocators -= 1;
  }
}

static const NxmemAllocatorVtable kAllocator = {
    (uint32_t)sizeof(NxmemAllocatorVtable),
    NXMEM_ABI_VERSION_MINOR_BASELINE,
    MINIMAL_MECHANISM_ID,
    {NXMEM_TIER_HOST, 0},
    NXMEM_CAP_ALLOCATOR,
    (const uint8_t *)kMechanismName,
    NULL,
    minimal_allocate,
    minimal_deallocate,
    minimal_retain,
    minimal_release,
    /* Every optional slot is NULL: this mechanism has no lazy backing, no
       shared mapping, no deferred release, and no structured release. NULL is
       how "unsupported" is spelled — never a stub that returns OK. */
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
};

/* ─── factory ──────────────────────────────────────────────────────────── */

static NxmemStatus
minimal_open_allocator(void *ctx, const NxmemOpenRequest *request,
                       const NxmemAllocatorVtable **allocator_out) {
  (void)ctx;
  if (request == NULL || allocator_out == NULL) {
    return nxmem_status(NXMEM_STATUS_INVALID_ARGUMENT,
                        "minimal: null pointer in open_allocator");
  }
  /* Refuse rather than return an allocator that is missing something the host
     said it needs. Silently handing back less is how a host ends up calling a
     NULL slot. */
  if ((request->required_capability_flags & ~(uint64_t)NXMEM_CAP_ALLOCATOR) !=
      0) {
    return nxmem_status(NXMEM_STATUS_UNSUPPORTED_CAPABILITY,
                        "minimal: this mechanism offers plain allocation only");
  }
  if (request->device.tier != NXMEM_TIER_HOST || request->device.index != 0) {
    return nxmem_status(NXMEM_STATUS_WRONG_DEVICE,
                        "minimal: this mechanism serves host memory only");
  }

  g_live_allocators += 1;
  *allocator_out = &kAllocator;
  return nxmem_ok();
}

static void minimal_factory_release(void *ctx) { (void)ctx; }

static const NxmemAllocatorFactoryVtable kFactory = {
    (uint32_t)sizeof(NxmemAllocatorFactoryVtable),
    NXMEM_ABI_VERSION_MINOR_BASELINE,
    (const uint8_t *)kMechanismName,
    {NXMEM_TIER_HOST, 0},
    NXMEM_CAP_ALLOCATOR,
    NULL,
    minimal_open_allocator,
    minimal_factory_release,
};

/* ─── required exports ─────────────────────────────────────────────────── */

NxmemStatus NxmemNegotiate(const NxmemNegotiateRequest *request,
                           NxmemNegotiateResponse *response_out) {
  if (request == NULL || response_out == NULL) {
    return nxmem_status(NXMEM_STATUS_INVALID_ARGUMENT,
                        "minimal: null pointer in negotiate");
  }
  /* Read only the prefix this plugin was built to understand. */
  if (request->struct_size < (uint32_t)sizeof(NxmemNegotiateRequest)) {
    return nxmem_status(NXMEM_STATUS_SHORT_STRUCT,
                        "minimal: the host's negotiate request is too short");
  }
  if (request->host_range.major_min > NXMEM_ABI_VERSION_MAJOR ||
      request->host_range.major_max < NXMEM_ABI_VERSION_MAJOR) {
    return nxmem_status(NXMEM_STATUS_VERSION_MISMATCH,
                        "minimal: this plugin implements nxmem major 1");
  }

  memset(response_out, 0, sizeof(*response_out));
  response_out->struct_size = (uint32_t)sizeof(*response_out);
  response_out->agreed_major = NXMEM_ABI_VERSION_MAJOR;
  /* This plugin uses only the baseline prefix, so it agrees to the baseline
     regardless of how new the host is. */
  response_out->agreed_minor = NXMEM_ABI_VERSION_MINOR_BASELINE;
  response_out->plugin_range.major_min = NXMEM_ABI_VERSION_MAJOR;
  response_out->plugin_range.major_max = NXMEM_ABI_VERSION_MAJOR;
  response_out->plugin_range.minor_min = NXMEM_ABI_VERSION_MINOR_BASELINE;
  response_out->plugin_range.minor_max = NXMEM_ABI_VERSION_MINOR_BASELINE;
  /* Never advertise a capability the host did not offer. */
  response_out->capability_flags =
      request->host_capability_flags & (uint64_t)NXMEM_CAP_ALLOCATOR;
  if (response_out->capability_flags == 0) {
    return nxmem_status(NXMEM_STATUS_UNSUPPORTED_CAPABILITY,
                        "minimal: the host did not offer plain allocation");
  }
  return nxmem_ok();
}

NxmemStatus
NxmemCreateAllocatorFactories(const NxmemAllocatorFactoryVtable **out_factories,
                              uint64_t max_factories, uint64_t *out_count) {
  if (out_factories == NULL || out_count == NULL) {
    return nxmem_status(NXMEM_STATUS_INVALID_ARGUMENT,
                        "minimal: null pointer in create_allocator_factories");
  }
  if (max_factories < 1) {
    return nxmem_status(NXMEM_STATUS_INVALID_ARGUMENT,
                        "minimal: no room for even one factory");
  }
  out_factories[0] = &kFactory;
  *out_count = 1;
  return nxmem_ok();
}

NxmemStatus NxmemQueryUnloadReadiness(NxmemUnloadReport *report_out) {
  if (report_out == NULL) {
    return nxmem_status(NXMEM_STATUS_INVALID_ARGUMENT,
                        "minimal: null report pointer");
  }
  memset(report_out, 0, sizeof(*report_out));
  report_out->struct_size = (uint32_t)sizeof(*report_out);
  report_out->live_allocators = g_live_allocators;
  report_out->live_allocations = g_live_allocations;
  return nxmem_ok();
}
