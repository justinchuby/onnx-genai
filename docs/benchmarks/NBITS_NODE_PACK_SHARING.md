# Native NBits packs shared by node shape variants

The executor caches compiled kernels by node and resolved input shapes. Native
MatMulNBits previously held eight independent layout slots and a cold-fill gate
in each kernel, so the same immutable weight could be prepared and charged again
when activation shape changed. The slots and gate now form a shared node owner,
passed through the existing `Kernel` constant-state hook. There is no global
address cache or new memory governor.

## Scope and evidence

The real-executor regression uses a symbolic activation row count, K=33, N=3,
2-bit wire weights, block size 16 and half scales. Half scales require native
preparation, so this fixture cannot pass by taking the contiguous f32 wire-borrow
path. The 2-bit route also cannot dispatch to MLAS in a reference build.

Before the change, execution at M=3 charges 72 Host ledger bytes; switching to
M=1 charges 144 and fails the regression. The new implementation executes
M=3,1,5,1, creates at least three shape-specialized kernels, and retains 72 bytes
throughout. Each prepared payload owns 36 wire bytes and nine f32 scales. Every
output byte equals the independent exact fixture result (-0.53125). A separate
session charges another 72; dropping each session refunds its own payload.
These are governor ledger bytes, not allocation-event counts, RSS, process peak
memory, latency or model throughput measurements.

The constant-state identity includes K, N, bit width, block size, accuracy level,
source encoding and the constant-input policy. Adoption refuses mismatched
parameters or unrelated erased state and reports an actionable error for an
alien grant source. Separately constructed sources over the same books and
holder number are still alien; cloned sources remain valid. Serialized MLAS
weights do not export native state.

Four host tests additionally cover:

- All eight factory-wired native layouts: producer teardown, shared warm payload,
  active readers resisting pressure, idle reclaim after producer teardown,
  rebuild, and the final reader outliving all kernels.
- Eight distinct cold kernel builders under a one-payload quota. All obtain the
  same payload; its grant survives all kernel drops until the last reader ends.
- Unknown state, each preparation parameter and metadata policy mismatch,
  governed/ungoverned source separation and ungoverned peer sharing.
- Optional mutable zero points with half scales: changed values produce the
  corresponding exact outputs, leave both shared slots empty and refund each
  transient preparation.

The prior all-layout pressure, conversion-peak, failed-builder refund,
concurrent-fill and numerical/dispatch tests remain. A complete source comparison
of execution/capability hooks and native preparation/numerical kernels, after
normalizing the added cache-owner field path and formatting, finds no other
changes in those regions.

## Validation

Final host validation (source fingerprints are in the adjacent
[`evidence directory`](nbits-node-pack-sharing/)):

| Build | CPU library passed / ignored | Session library passed / ignored | Strict all-target Clippy |
|---|---|---|---|
| native | 1959 / 19 | 272 / 0 | both crates pass |
| mlas | 1987 / 33 | 273 / 0 | both crates pass |

The initial
full-suite failure caught the new dynamic-metadata test's missing shared
dispatch-probe lock. That lock is now taken; the existing verifier is unchanged,
and the initial failure log is retained separately from final results.

```sh
cargo test --locked -p onnx-runtime-session --lib \
  executor::tests::nbits_shape_variants_share_one_native_pack_within_the_session_node \
  -- --exact --list
cargo test --locked -p onnx-runtime-session --lib \
  executor::tests::nbits_shape_variants_share_one_native_pack_within_the_session_node \
  -- --exact --nocapture
cargo test --locked -p onnx-runtime-ep-cpu --lib \
  kernels::matmul_nbits::tests::native_pack_shared -- --list
cargo test --locked -p onnx-runtime-ep-cpu --lib \
  kernels::matmul_nbits::tests::native_pack_shared
# Repeat both crates' complete --lib suites and strict --all-targets Clippy
# with and without --features mlas.
```

Verification uses ARM64 macOS, the existing cached ORT prebuilt, and the shared
Darwin host lock. Source fingerprints and command transcripts are preserved.
No CUDA device, x86 execution, whole-model token parity or throughput result is
claimed. Dense f32 caches and MLAS memo/store sharing/pressure policy are not
changed, and the optional research reference remains.
