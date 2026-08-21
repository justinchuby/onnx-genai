### 2026-08-21: Native MTP self-spec E2E — Gap 2 solved + Gap 1 wired to a clean seam; int4 shared LM-head is the hard wall

**By:** Gaff

**What:**
Landed the two shippable pieces of the native MTP self-speculative decode campaign on `squad/mtp-native-e2e-gaps` (off origin/main `79d991f51`, which includes #1633):

1. **Gap 2 (the true blocker) — SOLVED + GPU-validated.** The per-step hidden-seed
   auxiliary output (`hidden_states.63`) was pinned to the captured decode step's
   `[1,1,5120]` persistent binding and could not resize for `m>1` eager
   prefill/verify (`[1,m,5120]`), which broke *even plain greedy* on the MTP
   artifact (dispatch.rs external-output shape rejection). Fix in
   `native_decode/cuda.rs`: exclude the auxiliary binding tail from the eager
   device-binding slice (`..auxiliary_binding_range.start`) so aux outputs
   materialize to host on the eager path, while the captured single-token path
   keeps the persistent binding (capture-safe). The declared hidden output's last
   row is recorded into `last_hidden` on both paths (`read_aux_hidden_last_row`),
   located by output name — no hardcoded layer/dim. Inert for models with no
   declared hidden output (empty aux range ⇒ slice byte-identical to before).
   GPU evidence (H200 ord 5, `--ep cuda`, raw greedy on the real 27B int4 MTP
   artifact): coherent output "The capital of France is Paris.", CUDA-graph ON,
   `captures=2 replays=26 fallbacks=0`, token ids identical to reference.

2. **Gap 1 — native MTP proposer + load/dispatch wiring, to a clean seam.**
   `from_native_model_directory` (engine/load.rs) no longer hard-bails on
   metadata speculation and no longer hard-sets `mtp: None`; it now resolves
   `ResolvedMtpConfig` for `ProposalType::Mtp` (`load_native_mtp_proposer`),
   loads the pure-attention MTP head on the **ORT CUDA EP** (head session options
   built from the native decode device — the head's mixed bf16/f32 graph cannot
   load on CPU EP), and sets `mtp: Some(..)`. Threaded through
   `decode_backend.rs` (reject/allowlist/kind/plan), `runtime.rs` (speculative
   mode injection + `NativeSpeculationKind::Mtp` driver arm), and
   `native_speculative.rs` (`NativeProposer::Mtp` reusing the generic
   `MtpProposer` via `last_hidden()` + `argmax(base_logits)`; #1633
   recurrent-commit fires on accept). Inert for every non-MTP model.

3. **Two correctness fixes surfaced en route:**
   - `mtp_state_output` is now a real `Option` end-to-end. The metadata parser
     previously defaulted the *optional* state-output name to `"mtp_state"` even
     when a head declared none, so a pure-attention proposal-local head (our
     artifact, hc_mult=1) got a phantom state output that `MtpDecodeSession`
     then required and rejected. Parser preserves the declared `Option`;
     `from_sidecar_descriptor` honors it. Explicitly-declared state outputs
     (e.g. hc_mult>1 hidden-threaded heads) still resolve to `Some` — existing
     deepseek test unchanged; added a proposal-local regression test.
   - `MtpDecodeSession` KV-input dtype check now accepts BFloat16 (was
     f32/f16 only); the real head's KV is bf16.

**The hard wall (precise blocker, GPU evidence):** the MTP draft head reuses the
target's **shared LM-head**, which in the real Qwen3.8-27B int4 artifact is an
**int4 MatMulNBits-quantised** initializer: `lm_head.weight` uint8 `[248320,160,16]`
+ `lm_head.scales` bf16 `[248320,160]` + `lm_head.zero_points` uint8 `[248320,80]`
(node K=5120 N=248320 bits=4 block_size=32). The MTP `TargetInitializerLmHead`
adapter ("Phase 1") only consumes a *dense* f32/f16/bf16 `[vocab,hidden]` matrix.
(The embedding `model.embed_tokens.weight` is dense bf16 and works.) This is a
distinct architectural gap beyond the original three: dequantising the full
lm_head is ~5 GB, and a host-side draft GEMV over ~1.27 GB of int4 weight *per
decode step* is not throughput-viable (would cap us far below the 62.56 tok/s
baseline). The correct solution is to run the draft LM-head projection **on the
GPU** — either bake the lm_head into the MTP head ONNX graph, or have the draft
share the target's on-device quantised kernel — which is Phase-3 scope. The seam
now fails fast with an actionable error at this exact point rather than a
misleading dense-shape mismatch.

**Validation:** native-backend lib suite **569 passed / 0 failed** (greedy inert);
`onnx-genai-ort` mtp_session integ (2/2) and speculative module (20/20) green;
new `proposal_local_head_without_declared_state_output_stays_none` test passes.
Engine now resolves `ProposalType::Mtp` (not None) and loads `mtp/model.onnx` on
the ORT CUDA EP — validation point (a) reached; (b) token-identical MTP-vs-greedy
and the tok/s number are **blocked** on the int4 lm_head above. No fabricated
speedup number is reported.

**Runtime note for the CUDA MTP head:** point the process at the CUDA-enabled ORT
1.28 (the ort-sys auto-download is CPU-only) via
`ONNX_GENAI_ORT_LIB_DIR=$ORT_ROOT/lib` (+ `$ORT_ROOT/lib` and CUDA on
`LD_LIBRARY_PATH`); otherwise the head's CUDAExecutionProvider request fails
because the linked ORT reports only CPU.

**Why:** Gap 2 is the standalone blocker that broke plain greedy on the artifact —
shipping it unblocks the model regardless of MTP, and is capture-safe with
`fallbacks=0` on GPU. Gap 1 wiring de-risks the remaining E2E work behind a seam
that is inert for non-MTP models and stops precisely at the one genuinely-missing
capability (on-GPU quantised draft LM-head). Landing both now, with the blocker
documented and evidenced, is the honest increment the campaign asked for rather
than a half-wired path or a fabricated number.
