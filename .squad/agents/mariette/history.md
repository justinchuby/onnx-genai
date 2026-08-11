# Mariette — History (compacted 2026-07-29)

**Role:** Metal/MPS kernel engineer for the Apple Metal EP, owning heavy kernels such as MatMulNBits, GQA, softmax, RoPE, and RMSNorm. Correctness against CPU reference comes first, then simdgroup/threadgroup optimization, using ExecuTorch/PyTorch MPS references and onnx-genai end-to-end tests.

## Durable lessons
- Offline per-EP ONNX conformance harness and `docs/EP_CONFORMANCE.md` merged in `1dfab0d`; process-bridge design is recorded in decisions.
- Vendored `cpuinfo` beneath its crate so cargo publish succeeds.
- Mobius native block reviews require exact `BlockQuantizedMatMul` format/dimension/byte-preservation contracts, 4-bit/block-32 mixed-native scaffold, genai opset v1, and unchanged pure-Q8 behavior.
- Attention, CUDA CSA, and MTP reviews needed rejection/fix cycles before approval; keep reviewer-lockout corrections canonical.
- Omitted-optional dtype trap: reject CUDA standard-Attention optional past-KV claim regressions; Nabil's `8eb23f1` fix passed CUDA/session/CPU gates.
- CUDA claim-gate hardening must avoid GLM over-rejection, handle omitted optionals, scope standard-domain checks, and preserve CPU/GLM/CUDA parity.
- Perf campaign inbox decisions were consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.
- Wave-3 SwiGLU fusion halved activation launches from 48 to 24/token, merged as `12e48b8`, and measured about 673→689 tok/s at 256 tokens with zero fallbacks.
- WP-B2 engine runtime is the accepted presence/fallback/gating implementation feeding the completed epic.

## Recent work (current wave, ~2026-07-28/29)
- Latest live item is 2026-07-22: WP-B landed; Mariette's WP-B2 engine runtime remains the accepted implementation.

Full pre-compaction history in `history-archive.md`.

## 2026-08-11 — S1/S2/S3: Optional-Slot Liveness, Axis Bounds, Scratch Buffer

**Commit:** fbd565160
**Verdict:** Our EP WAS declining optional-slot nodes. With fallback disabled, 2/4 tests failed at session creation.

**Fixes applied:**
- Claim filter + dtype filter: carve-out for `__absent_output_*` sentinels
- Single-kernel fast path: inject absent sentinels using `input_slots` mapping
- Clip added to shape inference (SameAsInput(0))
- axis >= rank rejection (was >)
- Scratch buffer sized from primary output dtype

**Result:** 266 passed, 0 failed. All optional_slots tests pass with fallback disabled.

## 2026-08-11 — PR #762: S1–S3 optional-slot liveness proof

**Task:** Confirm and fix S1–S3 from Luv's review.

**Commits:** `fbd565160`, `4757e25b6`

**Findings:** With `disable_cpu_ep_fallback=1`, optional-slot tests failed at `CreateSession` — EP was declining nodes. Three root causes:
1. Claim filter (`ep.rs:275`) rejects `DataType::Undefined` outputs.
2. Dtype filter same rejection.
3. `Clip` missing from shape inference op lists.
4. Single-kernel fast path passed ORT inputs directly without injecting absent sentinels.

**Fixes:** Claim filter carve-out for absent outputs; Clip added to `SameAsInput(0)`; `input_slots` mapping in fast path; axis bounds: `>= rank`; scratch buffer: `numel * primary_output.byte_size()`.

**Outcome:** Challenger's review found the `__absent_output_*` sentinel was forgeable from model content. Locked out; Coco fixed.
