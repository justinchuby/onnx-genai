# Block MoE FC projection banks

BlockQuantizedMoE composes one admitted lane from maximum decoded-projection
scratch, bank A, bank B and the final accumulator. FC1, optional FC3, activation
and expert output now borrow the same private two-bank driver as float MoE.
No allocator, pool, public EP interface or per-expert/per-projection grant is
added. Dense and streamed projection callbacks fill caller destinations.

For R total input rows, hidden H, intermediate I and the validated FC1 width:

- Nongated: A = R*H (gather → expert output), B = R*I (FC1 → activation).
- Gated: A = R*max(H,I) (gather → activation),
  B = R*max(FC1+separate_FC3,H) (FC1/FC3 → expert output).
- Decode scratch and the accumulator remain disjoint. Serial experts reuse the
  banks without multiplying them by expert count or top-k. A singleton group
  borrows its original dense row.

Every projection destination initializes its actual prefix, including zero-K
bias results. Gated activation keeps the shared pair-index and AppliedBias
semantics; bias is not added again. Final output is published only after all
experts succeed, in the original expert/route accumulation order. The private
layout uses safe short borrows; no new unsafe initialization primitive is added.

This is an accounting/ownership change, **not an RSS-saving claim**. Retained
banks now charge capacity that the previous owned Vec path omitted from the
workspace. A conservative bound can exceed the old declared requirement, and
an insufficient cap may now refuse instead of funding those intermediates
outside the lane. Routing, tensor widening, compressed strided copies, GEMM
internals and final tensor storage remain outside this bound. It is not a
process-memory cap. Dense weight retention keeps its existing authority.

## Evidence and scope

Apple M1 Max, 10 logical CPUs, 32 GiB, macOS 27; rustc 1.98.0 / LLVM 22.1.8.
Native optimized probe builds have no MLAS symbols. The same public
InferenceSession graph/strict prepare/run/output-drop probe is built before and
after the source change. The initial and inline experiments retain the **same
baseline executable bytes**; null is a separate invocation of that baseline.
These are two source-controlled builds, not one-address-space A/B.

The graph has one runtime-domain MXFP4 BlockQuantizedMoE, four experts, top-k 2,
deterministic nonzero inputs/weights and alternating row routes. Dense/streamed
and separate-gate configuration are explicit fixture conditions, not findings
about automatic placement. The derived-cache budget is 64 MiB or zero as
specified; the total scoped Host cap is 64 MiB. This does not cap RSS. The Rayon
width is observed in the probe; it is not a per-worker participation trace.

A shared stable Darwin flock covers each complete build and matrix. The parent
records terminal child user+system CPU from wait4, wall duration, absolute peak
RSS, loads, exit status, full-output digest, argmax, actual workspace and derived
funding, and final zero usage after trim. Per-core foreign-CPU monitoring is
unavailable on this Darwin host; an otherwise idle host is not asserted.

**CPU is for the entire process**: thread-pool/model/session setup, allocation,
encoding, five fixed warm calls, repeated strict preparation + inference + output
drop, full final-output checking/writing, and retirement/trim. It is not a pure
kernel timer, per-token latency or full-model throughput. No short-window
getrusage/process-clock delta is used. Peak RSS includes the executable and
probe; it is not attributed to kernel banks.

Each matrix has 144 processes: three repetitions × six shapes ×
(setup/work for baseline/candidate/null + double work for baseline/candidate).
Normal calls = floor(268435456 / (rows*H*I*2*projection_count)), with projection
count 2 or 3. Setup-only has zero repeated calls; double work has exactly twice
the normal count. Setup and validation are otherwise identical. Arm order and
work order rotate. Each fixed candidate gets one complete matrix, not reruns
until a favorable sample appears.

Predeclared aggregate quality bars: baseline/null median CPU ratio within 5%,
setup median ≤25% of work-process CPU for all three arms, setup-adjusted double
work in [1.7,2.3] for baseline/candidate, and each work process ≥50 ms CPU.
Every receipt must close against the 10-logical-CPU physical wall bound (with
one CPU of collection tolerance), contain the expected work and shape, fund the
exact independently computed layout/derived payload, and trim to zero. Complete
outputs are bit-identical across every arm, setup/work/double, and repetition.
Both readers accept complete synthetic matrices and reject ten malformed or
vacuous variants; synthetic failed A/A cells remain unqualified.

## Complete results

Both matrices pass all six **aggregate** quality cells. Ratios below are the
median of three paired candidate/baseline whole-process CPU ratios, with every
pair retained in the range. A/A is the predeclared ratio of null/baseline medians.
Bytes are actual retained workspace funding, independently checked against the
layout formulas; they are not a knob chosen as the measured result.

| Version | Rows × H / I | Width | Weight path | Gate | Candidate / baseline CPU | Pair range | A/A | Workspace bytes |
|---|---|---:|---|---|---:|---|---:|---|
| Initial | 1 × 32 / 64 | 1 | streamed | none | 0.9976 | 0.9759–0.9986 | 0.9868 | 1280 → 1536 |
| Initial | 1 × 64 / 128 | 1 | dense | none | 0.9807 | 0.9718–0.9921 | 0.9987 | 2560 → 3072 |
| Initial | 1 × 64 / 128 | 1 | streamed | separate | 1.0667 | 1.0544–1.0863 | 0.9901 | 2560 → 3840 |
| Initial | 8 × 128 / 256 | 1 | streamed | none | 1.0048 | 0.9747–1.0064 | 0.9788 | 18432 → 26624 |
| Initial | 8 × 128 / 256 | 4 | streamed | separate | 1.0541 | 0.9935–1.0756 | 1.0212 | 28672 → 49152 |
| Initial | 8 × 128 / 256 | 4 | dense | separate | 1.0288 | 1.0021–1.0541 | 0.9988 | 28672 → 49152 |
| Inline | 1 × 32 / 64 | 1 | streamed | none | 0.9695 | 0.9506–0.9775 | 0.9933 | 1280 → 1536 |
| Inline | 1 × 64 / 128 | 1 | dense | none | 0.9779 | 0.9761–0.9908 | 1.0080 | 2560 → 3072 |
| Inline | 1 × 64 / 128 | 1 | streamed | separate | 1.0145 | 0.9944–1.0377 | 0.9990 | 2560 → 3840 |
| Inline | 8 × 128 / 256 | 1 | streamed | none | 1.0013 | 0.9958–1.0023 | 0.9937 | 18432 → 26624 |
| Inline | 8 × 128 / 256 | 4 | streamed | separate | 0.9033 | 0.8904–0.9661 | 1.0206 | 28672 → 49152 |
| Inline | 8 × 128 / 256 | 4 | dense | separate | 0.9819 | 0.9804–1.1913 | 1.0021 | 28672 → 49152 |

The initial bank candidate has a consistent +5.4–8.6% paired single-row gated
regression (median +6.7%) and a +5.4% four-width streamed point estimate; it was
**not adopted**. Inspection found a 472-byte outlined AppliedActivationBias row
helper in the candidate where the original probe had no such outline. This
motivated only an `#[inline]` hint, retaining all layout and arithmetic rules.
The original source, binaries' hashes, data and disassembly remain archived.

The inline candidate's single-row gated point change is +1.4%, range -0.6% to
+3.8%; a small cost is not excluded. The former consistent 5–7% regression is
not reproduced in that matrix. There is still an outlined helper copy in the
inline binary, so symbol presence alone does not identify every changed call
site. No hardware-counter or per-instruction causal claim is made.

An aggregate pass does not make every repetition quiet. In the refined matrix,
width-4 streamed repetition 2 has null/baseline 1.117; width-4 dense repetition 0
has 1.252 and candidate/baseline 1.191. They remain in the raw data and pair
ranges; no sample is discarded. The width-4 -9.7% point estimate and smaller
negative point estimates are fixture observations with these limits, not a
general speedup promise. The change does not establish universally zero CPU
overhead or inference/model throughput on any other tier.

## Numerical and resource validation

The new exact-bit packed-versus-owned-driver regression covers two formats,
five asymmetric/zero-axis pairs, row counts 0/1/3, six activation/gate layouts,
reverse row gathering, bias (including -0), and two fresh bindings over poisoned
banks: 360 complete-output comparisons. Seven guard values remain untouched.
The old owned reference first fails for fused inter=0 at chunks_exact(0). The
fix skips channel iteration for an empty FC1 while preserving zero-K FC2 and
its bias, including legitimate QMoE callers of that owned storage contract.

Existing dense-vs-streamed format/activation/pressure regressions and float MoE
bank tests remain. The expanded governed Block MoE test covers interleaved,
split and separate gates, a stored one-shard decode plan used in a larger Rayon
pool, exact four-region charges, no warm issuer request/return calls across
repeated whole-lane reuse (including zero-byte callbacks), stable outputs,
dense hits leaving decode scratch untouched, and trim-to-zero. These callback
counts are direct RecordingGovernor observations, not inferred from a constant
net-used gauge. They do not claim an allocation-free entire inference call.

Final validation receipts/logs are under validation/. The code must pass native
and MLAS-reference CPU suites in both default and optimized configurations,
session and decoded-layout tests, and strict all-target Clippy for both feature
sets. Those receipts bind the same two runtime source hashes as the inline
performance matrix; documentation and archived evidence are the late additions.
No CUDA device execution or CUDA performance is asserted.

## Reproduction and retained failures

The manifest hashes every retained payload. initial/ and inline/ contain each
runner, reader, protocol, terminal resource receipts, probe source/manifest/lock,
Cargo artifact/build receipts, stdout/stderr/output archives, results and frozen
runtime source/reconstruction patch. Binaries are kept locally, not committed.
The inline baseline receipt explicitly binds the inherited initial executable.

The initial historical probe failed to compile before any benchmark child: the
current API returns WorkspacePreparationReport rather than one flat bytes field.
That failure/source is retained under excluded/preparation-api/. The corrected
probe calls require_complete(), verifies step bytes and an empty persistent
bound; no compatibility API was reintroduced. The first exact-test compile error
(private attribute fields) and actual chunks_exact(0) red/green sequence are
also retained as diagnostic provenance, not performance samples.

Runners preserve real host paths. Reproduce in an isolated worktree at the
recorded parent, apply the decompressed candidate patch, adapt all root/ORT/
probe/output paths and read hardware from the actual host. The initial runner
builds the parent and candidate by temporarily substituting only its two saved
Rust files and restoring them in finally; never run it over someone else's
worktree or alongside a build of that tree. Hold the same stable Darwin flock
for the entire build/matrix; the runner refuses an unclaimed invocation. The
inline runner reuses only a hash-verified original baseline with byte-identical
probe source. wait4/RSS units are Darwin-specific; other platforms need a
validated collector. Analyze unchanged raw receipts and keep all noisy samples.
