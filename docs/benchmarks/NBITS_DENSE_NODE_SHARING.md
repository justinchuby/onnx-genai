# NBits dense weights shared by executor node

The compact layout slots already shared node constant state. The f32 NK cache
remained kernel-local, so distinct activation shapes could retain multiple full
expansions of one immutable matrix. The same state now carries an `Arc` to the
existing dense cache, sharing its cold-fill gate and pressure enrollment. The
canonical decoder, arithmetic, selected backend and placement are unchanged.

The shared state is now named for all derived NBits weights. Its single
eligibility rule includes the dense-byte retention ceiling alongside preparation
attributes and constant-input policy. A limit mismatch declines adoption; it
cannot turn denied retention into an admitted cache. Exact grant-source identity
remains required. No compatibility alias, process-global store or new budget is
introduced.

## Actual executor regression

The fixture is a valid ONNX graph with unknown-rank f32 input/output, K=33, N=3,
4-bit weights, block size 16, and constant group indices. Grouped quantization
requires the canonical dense route rather than compact integer kernels or MLAS
SQNBit. Shapes `[33]`, `[1,33]`, `[1,1,33]`, `[1,33]` all have one activation row,
so each backend can run the resident NK path. This does not assume that every
backend supports transposed-B prefill GEMM.

Before the change, the first shape retains 396 Host ledger bytes; the second
retains 792 and fails the new regression. Afterward all four executions retain
396 bytes, at least three distinct shape-specialized kernels exist, and every
output byte equals the independent exact fixture result 0.5. Output rank follows
the activation rank. A separate session retains another 396; dropping each
session refunds its own expansion.

These are ledger ownership observations, not allocator byte counts, RSS,
process peak memory, latency, prefill speed or whole-model throughput results.
They do not establish model token parity or CUDA/other-ISA execution.

## Ownership and policy controls

- Export an already-warm factory state to another shape variant. Both hold the
  same dense payload, the producer can drop, active readers resist pressure,
  idle membership can be reclaimed, and fresh admission rebuilds the entry.
  A final reader keeps its grant after both kernels are gone.
- Eight independently compiled cold kernels obtain one dense payload under a
  one-payload Host quota; all readers retain that one grant until the last drops.
- Add a retention-limit mismatch to the existing source/parameter rejection
  suite, while retaining foreign-source, constant-policy and erased-state guards.
- Mutable group indices produce fresh exact results and retain no dense entry.
  Constant metadata under a zero retention limit likewise stays transient.

Existing all-eight-layout compact sharing, grant peak, refusal, failed-builder,
pressure, numerical and dispatch guards remain. The dense cache implementation
itself is unchanged; only its owner is shared. Active evicted weights keep their
owned grants independently of cache membership and of the nominal planner cost.

## Validation

Final commands, source fingerprints and results are preserved in the adjacent
[`nbits-dense-node-sharing`](nbits-dense-node-sharing/) directory.

| Build | CPU library passed / ignored | Session library passed / ignored | Strict all-target Clippy |
|---|---|---|---|
| native | 1962 / 19 | 273 / 0 | both crates pass |
| mlas | 1990 / 33 | 274 / 0 | both crates pass |


```sh
cargo test --locked -p onnx-runtime-session --lib \
  executor::tests::nbits_shape_variants_share_one_dense_weight_within_the_session_node \
  -- --exact --list
cargo test --locked -p onnx-runtime-session --lib \
  executor::tests::nbits_shape_variants_share_one_dense_weight_within_the_session_node \
  -- --exact --nocapture
cargo test --locked -p onnx-runtime-ep-cpu --lib \
  kernels::matmul_nbits::tests::dense_weight_shared_shapes -- --list
cargo test --locked -p onnx-runtime-ep-cpu --lib \
  kernels::matmul_nbits::tests::dense_weight_shared_shapes
# Complete --lib tests and strict --all-targets Clippy run for both crates,
# with and without the optional --features mlas reference.
```

Host verification uses ARM64 macOS and the existing cached ORT prebuilt, under
the shared Darwin lock. Dense f32 computation can call the already-selected
backend; linking the MLAS reference does not imply it ran. No reference is retired
or family graduated by this ownership change.
