### 2026-08-21: MTP int4 draft LM-head (Option C) — landed + MTP runs E2E on real hybrid; token-identity blocked by a native main-Scan recurrent-state bug

**By:** Gaff

**What:**
Phase 3 delivered the on-GPU int4 draft LM-head projection and, wiring it in, got
native MTP self-speculative decode to run **end-to-end for the first time** on the
real Qwen3.8-27B int4 hybrid (GDN+GQA) artifact. Landed on
`squad/mtp-int4-draft-lmhead` (off origin/main `6e3590128`, includes #1637).

1. **int4 draft LM-head projection — Option C (self-contained), CPU-oracle-validated.**
   Rejected Option A (Roy re-exports a sidecar with a baked int4 lm_head; +1.27GB
   duplication, cross-agent) and the executor-surgery variant of Option B. Instead
   build a **standalone single-node `MatMulNBits` `InferenceSession`** that reuses
   the target model's *own* int4 lm_head initializers (weight/scales/zero_points)
   **zero-copy** via the same `WeightStore` mmap — no re-export, no weight
   duplication, no host-side dequant. New `QuantizedDraftLmHead` +
   `DraftProjectionDevice{Cpu,Cuda{index}}` + `build_quantized_draft_lm_head`
   (speculative/mod.rs), plus `InferenceSession::from_graph_with_provider`
   (onnx-runtime-session/src/lib.rs). The projection runs during **proposal,
   outside** the captured native decode step — CUDA-graph capture of the target
   step is untouched (fallbacks stayed 0 on GPU).
   - **Loader-lowering discovery:** the native loader lowers int4 `MatMulNBits`
     into an explicit BitShift/BitwiseAnd/Cast/Sub/Mul/MatMul dequant subgraph, so
     the loaded IR has **no `MatMulNBits` node** to clone. The 3 initializers do
     persist, so the builder finds them by name (`lm_head.weight` +
     `.scales`/`.zero_points`) and **shape-derives all quant geometry** — N, k_blocks,
     blob → K=hidden, block_size=K/k_blocks, bits=blob*8/block_size. Nothing hardcoded.
   - **bf16-scales discovery:** the CUDA `MatMulNBits` kernel requires **Float32
     scales**; the artifact stores **BFloat16**. Convert once at build time to an
     inline f32 initializer (bf16/f16 both handled). Draft exactness isn't required
     for correctness (drafts are verified against the target), so an imperfect
     projection only lowers acceptance.
   - Oracle: `quantized_draft_lm_head_projects_int4_argmax` (one-hot hidden isolates
     a single int4 column, code 15 vs code 8) — passes on the CPU int4 kernel; the
     GPU path shares the identical builder.

2. **Two wiring fixes needed to reach the MTP path (both correct, both kept):**
   - **Dispatch (runtime.rs `generate_with_callbacks`):** metadata-driven MTP
     (`self.mtp.is_some()`) now routes to the cold spec-capable path, mirroring
     `native_shared_kv_proposer`. Previously metadata MTP fell through to the warm
     in-session path and never engaged.
   - **Hidden-seed derivation (load.rs `from_native_model_directory`):** MTP
     artifacts declare the draft seed only in `speculative.target_hidden_output`,
     not `model.io.hidden_output`, so the native session never recorded it. Derive
     `model.io.hidden_output` from `speculative.target_hidden_output` when
     `proposal_type == Mtp` and the io field is empty; never overrides an explicit
     value; non-MTP models untouched.

3. **MTP ran E2E on the real hybrid** (H200 ord 5, CUDA-graph ON, fallbacks=0):
   acceptance **76.9%** (verify_steps=7, proposed=13, accepted=10). But output was
   **NOT token-identical** to greedy — root-caused below.

**The blocker (root-caused, GPU-proven — NOT in Phase 3 scope):**
The native **main Scan child-session executor** produces wrong logits when the
recurrent (GDN SSM + conv1d) state is **non-zero**, even at m=1. The
**decode-inline sibling exec** (normal greedy decode) handles non-zero state
correctly; it was purpose-built (comment at native_decode/mod.rs:~1220) for
"recurrent-state continuity across the prefill→decode boundary." Prefill uses the
main exec but from **zero** state, masking the bug. MTP verify (m>1) is forced onto
the main exec and thus inherits it — this is the first consumer to require
non-zero-state continuity there.
- **Decisive experiment:** a temporary `GAFF_DISABLE_INLINE` toggle forced greedy
  onto the main exec → it produced the **exact same wrong stream as MTP**
  (`[0, 57590, 13, 198, 760, ...]`), while inline greedy produced the correct
  `[0, 64, 0, 32011, 13, ...]`. This isolates the fault to the main-exec Scan's
  non-zero recurrent-state handling — **independent of MTP, int4 projection, and
  multi-row verify.** (Toggle removed before commit; evidence recorded here.)
- Pure-attention MTP is unaffected (verify uses prefix-sliced KV rewind, no
  recurrent state) and would be token-identical; **only hybrid-recurrent MTP** hits
  this wall.

**Why (landing decision):**
Per coordinator guidance ("land what's solid, report the precise blocker + GPU
evidence, I'll re-scope"). The int4 projection (Phase 3 deliverable), dispatch fix,
and hidden-seed derivation are independently correct, CPU-oracle-validated, and
necessary regardless — they also **exposed** the executor bug. The recurrent Scan
bug is a native onnx-runtime-session executor/Scan-lowering issue, out of Phase 3
scope, and a deep fix (the main exec was never designed to continue non-zero
device-bound recurrent state; the inline sibling was built specifically for that).
Full lib suite green (574 pass, greedy inert). No speedup number is reported — the
E2E number is gated on the recurrent-Scan fix, not on this PR.

**Suggested re-scope for the blocker:** either (a) give the main-exec Scan
non-zero device-bound recurrent-state continuity (match what the inline sibling
does), or (b) decompose MTP verify's m>1 forward into K sequential m=1 inline-exec
steps (correct recurrent continuity, K forwards instead of one). (a) fixes it for
all m>1 recurrent decode; (b) is localized to verify. Recommend (a) if the
executor owner has bandwidth; the false "byte-identical" comment at
native_decode/mod.rs:~1300 should be corrected either way.
