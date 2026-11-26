# Independent Conv implementations in one reference binary

Previously the `mlas` feature replaced `conv_ref.rs` with the MLAS Conv module.
A reference-enabled binary therefore could not directly execute this project's
independent native convolution implementation alongside the reference. It now
also compiles `conv_ref.rs` privately as `kernels::conv_native` and exposes
`backend_ab::convolution_factories()` with named `native_generic_gemm` and `mlas`
factories. This API exists only with `mlas` and `ops-cnn`.

The default build and feature-selected operator registry are unchanged. Native
production still uses its original automatic GEMM backend selection. Only the
private reference-build native module fixes its GEMM fallback to Generic, at
compile time, so `NXRT_CPU_GEMM_BACKEND=mlas` cannot silently make both arms use
MLAS. BNNS and direct NEON remain independent native convolution routes. No
factory/kernel field, runtime production toggle, environment mutation, allocator,
quota interface, pool or packed-weight mechanism is added.

This pair is **native Conv with Generic GEMM versus MLAS Conv**. It is not a
comparison against every production auto-selected native backend, and no speed
or memory-ranking conclusion follows. Both factories use their existing
implementation's preparation/ownership, including the reference's existing
packing. The optional MLAS reference remains ungraduated.

## Actual execution and independent witnesses

The linked integration test creates both concrete kernels and executes each twice
with declared constant weights/biases. Nine rank3/rank4 geometries cover groups,
batches, pointwise, asymmetric padding, strides, dilation, depthwise and channel
multipliers. Bias absence/presence and identity/ReLU give 36 paired f32 cases,
144 complete executes. Every output element from both arms is checked against an
independent direct f64 convolution oracle with `2e-5*(1+abs(reference))` tolerance.
This does not promise output-bit identity across implementations or cover all
float16/bfloat16 semantics. Empty batch is exercised separately on native only;
no MLAS zero-batch preparation capability is inferred.

Concrete KernelType identities must differ. The linked registry must still create
the existing public MLAS Conv type, matching the explicit reference arm; a default
build independently verifies its existing native registration and absence of MLAS.
For a forced grouped/dilated GEMM case, the existing dispatch ledger must record
exactly eight `(M,N,K)=(3,25,12)` Native GEMMs across two executes, with no dropped
observations. The final linked run sets `NXRT_CPU_GEMM_BACKEND=mlas` externally.
This verifies a native arithmetic entry actually ran despite the hostile override;
symbol presence or numerically equal results alone do not establish independence.
The ledger is enabled only in the isolated correctness test process, not a timing
measurement or production path.

Two negative controls run the exact linked test. One points the native factory
at the reference and must fail the concrete-kernel independence guard. The other
replaces fixed Generic with ordinary auto detection and must fail the Native GEMM
witness under the explicit MLAS environment. Both must compile and execute one
failing test; compiler errors/empty selectors are rejected. All four source files
are restored exactly after controls. No diagnostic branch ships.

Default Conv/allocation guards and the newly compiled native module's existing
unit tests remain covered. Initial strict-lint findings (a nested test-oracle if
and API placement after a test module) are retained separately; neither requires
a lint exemption. The current API is placed before the test module.

No timing benchmark, LLM throughput comparison, RSS saving, global MLAS removal
or CUDA execution is claimed. Heavy checks use the shared host flock coordinated
with the peer chat. Reproduction drivers retain their original checkout, ORT 1.32
cache and target paths and need deliberate adaptation elsewhere. Run
`python3 -B docs/benchmarks/conv-same-binary/verify.py .` for sealed evidence and
current source; omit the checkout argument after later source changes.

Final validation on this M1 Max passes 46 default-native Conv tests (one existing
ignored timing benchmark), the default registry test, the default actual-execute
allocation guard, the linked 36-case differential, and 46 tests of the private
linked native module (one ignored). Strict all-target Clippy passes both feature
configurations. Both mutation controls compile and fail exactly one executed test
at their intended guard. The final source hashes match restored source bytes.
The default Conv reconstruction removes only the research selector's cfg/comment
region and recovers the baseline source byte-for-byte.
