# Deckard — On-device LongRoPE select: de-hosting the `If` capture seam

Branch: `perf/phi-ondevice-rope` off `origin/main` (`8793ea9`)
Status: **needs review before merge (correctness-sensitive)** — do NOT self-merge.
Requested by: Justin Chu. Worker: Deckard.

## The seam (reconfirmed)

Phi-4-mini's LongRoPE selector is `Greater(gather_len, 4096)` → host `If`
(`/model/rotemb_caches_subgraph/If`) choosing between two pure `Constant`
cos/sin caches:
- `then_branch` (predicate TRUE / long-context): cos,sin `[131072, 48]` fp16
- `else_branch` (predicate FALSE / short-context): cos,sin `[4096, 48]` fp16

`If` is a control-flow op, so `plan_capture_segments` (executor.rs) *always*
makes it an eager seam: every decode step the cond scalar is read back to the
host, the captured CUDA graph is split into **2 segments / 1 seam**, and CPU/GPU
serialize at the split. The predicate is loop-invariant during steady decode but
paid every step (~1.9 ms/token, the dominant non-GPU cost per Marsten's Nsight).

The merged memo fix (`719d2fe`) removed the *cheap* part (branch re-exec + ~786 KB
cache copies) but left the seam itself. This change removes the seam.

## The rewrite (general, not Phi-hardcoded)

Two parts, both topology-driven:

**Part A — capture-safe `Where` kernel** (`kernels/where_op.rs`).
Rewrote the CUDA `Where` to mirror the merged capture-safe Binary/Greater pattern:
a persistent `WhereMetadataCache` (device metadata buffer, alloc/free/sync
discipline copied from `elementwise.rs::BroadcastMetadataCache`), no per-call
alloc/upload/free, no per-call `synchronize()`. `capture_support()` advertises
`Supported` **only** for an *invariant scalar-predicate select*
(`cond.numel()==1 && x.shape==y.shape==out.shape`), recorded as a capture
signature guarded by `require_matching_capture_signature`. The general
broadcasting `Where` stays an eager seam — no regression.

**Part B — `CudaOnDeviceConstantSelect` optimizer pass** (`optimizer.rs`,
registered in `cuda_optimization_passes()`).
Generalized as: *"a loop-invariant scalar-predicate `If` whose branches are
pure, side-effect-free constant selections can be lowered to on-device
`Where(cond, then_const, else_const)` per output."* Fires only when BOTH branches
contain ONLY `Constant` nodes (zero formal inputs, one output each, `value`
tensor attr — no outer captures).
- **Equal-shape branches** → direct `Where`, unconditionally byte-exact.
- **Differing leading dim** (Phi's `[131072,48]` vs `[4096,48]`): requires
  `cond = Greater/GreaterOrEqual(_, T)` with scalar-int `T`; the TRUE branch must
  be the LARGER table; trailing dims equal; and `else_lead == T` (crisp tie). The
  smaller (FALSE) constant is zero-padded along axis 0 up to the large leading
  dim. **Output shape is fixed at the large shape `[131072,48]` forever** → no
  per-step shape change → single captured graph even across the boundary.

## Correctness argument (airtight) + guards

Padding APPENDS rows at indices `[else_lead, then_lead)` that the original short
table never had. When the predicate is false (`seq ≤ T = else_lead = 4096`), every
position the model indexes is `< T`, i.e. within the original valid extent — the
appended rows are provably never read. When true, the full large table is selected
unchanged. `Where` recomputes the selection from the *live* predicate each step,
so the boundary flip is exact with no stale memo. GQA derives rotary_dim from
`cos.shape[1]` (=48) and indexes by position; `shape[0]` is only a bound, so the
larger `[131072,48]` output is safe. Byte-preservation of the original
`[0, else_lead)` rows is asserted in a unit test.

## Validation (idle GPU 0, `.cudaenv.sh` sourced)

**Captured-region count (the target):**
| build    | segments | eager seams |
|----------|----------|-------------|
| baseline (`8793ea9`) | **2** | **1** |
| ondevice | **1** | **0** |
Collapse achieved. Verified via `ONNX_GENAI_LOG_CAPTURE_SEGMENTS=1`.

**Per-op trace (`profile_native --trace`, 60 tokens):**
| op      | baseline                        | ondevice                     |
|---------|---------------------------------|------------------------------|
| `If`    | 60 exec, **59 rejected (eager)** | **0** (gone)                 |
| `Where` | 0                                | 4 exec, **2 captured** (cos+sin) |
| `Greater`| captured                        | captured                     |
| total rejected/eager ops | **59** | **0** |
The 1.9 ms/token host `If` seam is eliminated; nothing is rejected from capture.

**Perf — interleaved native-only, idle GPU 0, `--steady --warmups 2 --runs 9
--tokens 120`, 5 interleaved iterations (baseline↔ondevice back-to-back):**
| build    | tok/s per iter                          | median   | range          |
|----------|-----------------------------------------|----------|----------------|
| baseline | 198.95, 203.90, 202.85, 203.50, 204.37  | **203.50** | 198.95–204.37 |
| ondevice | 322.15, 322.31, 322.56, 321.73, 321.58  | **322.15** | 321.58–322.56 |

**+58.3% (203.50 → 322.15 tok/s)**, i.e. **1.810 ms/token** saved
(4.914 → 3.104 ms/token) — matches the predicted ~1.935 ms `If`-seam cost almost
exactly, and pushes Phi **well past the ORT native reference (229.62 tok/s)**.

Honesty note: this is far larger than the +1.8% Marsten re-measured for the
*memo* fix (`719d2fe`), because that fix kept the seam; this change *removes* it
(2→1 graphs, no per-step host cond read). The numbers are tightly reproducible on
an idle GPU (ondevice spread <1 tok/s across 5 interleaved iters). The `Where`
runs over `[131072,48]×2` fp16 each step (~17 µs, captured, no host sync) —
negligible (~0.3% of ~5 ms/token) vs the seam removed.

**Correctness:**
- 160-token greedy decode: `generated_text` **byte-identical** to baseline.
- **Boundary-crossing (seq crosses 4096):** 4200-token greedy decode
  (`ONNX_GENAI_CUDA_KV_MAX_LEN=5000`, 4192 decode tokens). Both builds:
  sha256 `b76a17085739788d8c644fc01453582b045b6f3adaf47d3223466e30fb30629a`
  — **byte-identical**, and ondevice stays **1 captured segment** across the
  boundary (fixed large output shape, no re-plan). The short→long cos/sin cache
  switch is exact.

**Gate:**
- `cargo test -p onnx-runtime-ep-cuda --features cuda --lib`: **201 passed / 0
  failed** (192 baseline + 6 new pass tests + 3 new Where capture-safety tests).
- `cargo test -p onnx-runtime-session --features cuda`: green, incl.
  `control_flow` (21) and `cuda_control_flow_safety` (1).
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda --lib -- -D warnings`:
  clean. (The 42 `-D warnings` errors under `--tests` are pre-existing on
  `origin/main` in unrelated `tests/*.rs` integration harnesses — newer clippy
  toolchain lints, not touched by this change.)

## Files changed
- `crates/onnx-runtime-ep-cuda/src/kernels/where_op.rs` (+261 / capture-safe
  Where + 3 unit tests)
- `crates/onnx-runtime-ep-cuda/src/optimizer.rs` (+598 / `CudaOnDeviceConstantSelect`
  pass + registration + 6 unit tests)

## No-gos / caveats
- The differing-shape lowering deliberately requires the crisp tie
  `else_lead == T` and TRUE = larger table; anything else is skipped (stays an
  `If` seam) rather than risk an out-of-extent read. This keeps it correct and
  general without special-casing LongRoPE by name.
- Reviewer focus: the zero-padding correctness argument (appended rows never
  indexed when predicate false) and the `Where` capture-signature gating.
