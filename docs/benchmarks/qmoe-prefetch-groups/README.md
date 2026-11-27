# QMoE canonical groups and prefetch protocol

The route-first task map already provides unique, ascending expert groups. Funding
now borrows its keys directly, enforcing uniqueness without an arbitrary iterator
callback under the payload lock. After preflight, the driver consumes that map;
plain row/slot/weight metadata retires as each group finishes. No second ordered
Vec or index-to-expert lookup remains. The scope lifetime borrows the cache
operation, allowing the task map to move independently.

The scoped worker receives `PrefetchExpertId`, an actual key. The caller receives
current weights, requests the next key, then computes the current group. Only
loader/weights require Send. Callback errors preserve their original EP variant,
close endpoints before scoped joining, and retire pending weights. Default thread
startup uses fallible Builder spawning with an actionable error and typed OS
source. No compatibility interface or test hook is shipped.

## Coverage and boundaries

- Pure protocol tests cover sparse IDs 3/11/41, singleton and empty groups,
  ascending requests, a synchronized current-plus-next peak, first/middle loader
  failures and compute failure while the next payload is live. Error paths are
  bounded by a five-second test harness; all created test payloads retire.
- Real mmap cases select only experts 2/5/7 with three score rotations and three
  capacities: zero-retention streaming, partial-bank serial fallback and full-bank
  residency. All output bits and complete statistics match serial, excluding the
  overlap-dependent dequantized peak and unrelated dynamic Linux process data.
  Exact selected-expert mmap bytes and token counts are checked.
- Existing eligibility, shared-bank, pressure, issuer-count, numerical and ownership
  cases remain covered. Warm pins introduce no issuer callbacks; governed streaming
  still requests/refunds one window. No row/tile workspace quotas were introduced.

The wrong-ID mutation changes only worker loading to expert zero. The exact sparse
protocol test must compile, run and fail with the weight/group ID mismatch (zero
passed, one failed). A separate host-only startup probe temporarily requests an
impossible 2^62-byte thread stack. On this 64-bit macOS host the OS rejects startup;
the exact single test verifies no callbacks run, actionable text and a downcastable
std::io::Error cause. Neither fault injection remains in production. Source bytes
are restored in finally and hashed.

## Validation

The final source passes native and MLAS QMoE suites (68 passed, one existing ignored
timing benchmark each), the host QMoE oracle (one passed), and strict all-target
CPU Clippy in both configurations. Nonempty test listing and result counts are
checked by the retained driver. Heavy commands use the shared host flock coordinated
with the peer chat. Drivers contain original local paths and need adaptation to
reproduce elsewhere.

This is an ownership/protocol simplification, not a speed measurement. Channels,
map metadata and reader pins still allocate. Other cached payloads can outlive the
driver's current/next pair. No total-pipeline zero-allocation, reduced RSS, inference
latency, CPU-time, or GPU performance claim is made. No CUDA device was used.

Run `python3 -B docs/benchmarks/qmoe-prefetch-groups/verify.py .` to verify sealed
payloads and current source. Omit the checkout argument for historical verification
after subsequent changes.
