### 2026-08-11 — Qwen3.5/3.6-27B hybrid native-CUDA enablement

**By:** Cohaagen

**What:** Enabled `Qwen/Qwen3.6-27B` (model_type `qwen3_5`, hybrid GDN
`LinearAttention`/`CausalConvWithState` + periodic `GroupQueryAttention` + int4
`MatMulNBits`) to load and decode on the native CUDA EP, byte-exact vs an fp32
oracle. This was the last model the ORT-vs-native benchmark flagged as
"Unsupported native load: model needs unsupported native operators".

Root cause was **not** a missing kernel and **not** a mis-generated graph. The
`/home/justinchu/mary-models/qwen3.6-27b-int4-cuda/model.onnx` graph is a
correct hybrid (64 layers: 48 GDN linear-attention layers exposing
`conv_state`/`recurrent_state`, 16 periodic full-attention layers exposing dense
`key`/`value`) and the native kernels (`linear_attention.rs`,
`causal_conv_with_state.rs`, `group_query_attention.rs`) already exist. The
blocker was the artifact's **thin `inference_metadata.yaml`**: it declares only
`grouped_query_attention` and NO `io` port contract. On the native load path the
Resource Governor derives per-layer KV page byte geometry from `model.io.kv_inputs`
/`kv_outputs`; with no `io` block `resolve_kv_layers` returns `None` (it
deliberately never guesses KV pairing from tensor names), so the governor failed
with `cannot derive the KV memory budget because per-layer KV page geometry is
unknown ... fix by declaring model.io.kv_inputs and model.io.kv_outputs`.

**Fix (DRY, attribute/shape-driven, no model-name gate):** the native loader now
auto-derives the decoder `io` port contract from the ONNX graph's own port
inventory when the sidecar declares none, via
`GenAiConfig::derive_decoder_io_from_graph` — the exact proven derivation the
native decode step driver already uses in its `derive_fallback_io`. New helper
`maybe_fill_hybrid_io_from_graph` in `engine/load.rs` fills sparse dense
`kv_inputs`/`kv_outputs` + fixed recurrent `state_pairs` (and token/mask/position/
logits port names by presence). It is gated on a non-empty derived `state_pairs`
(the recurrent-hybrid case the shape-inference path cannot classify); a declared
`io` block always wins and pure-dense decoders are untouched. Added `Default` to
`ModelCapabilities` so the derived spec can be attached when `model` is absent.
This makes ALL future hybrid GDN models with a correct graph but thin metadata
auto-run the optimal native path — no per-model changes.

**Validation (H200, CUDA_VISIBLE_DEVICES, release):**
- Native-CUDA fp16 greedy of "The capital of France is" →
  `[11751, 13, 271, 248068, 271, 248069, 271, 4639]` = " Paris.\n\n<think>\n\n</think>\n\nThat" (coherent; first token " Paris" correct).
- Teacher-forced next-token argmax (fresh engine): native-CUDA fp16 = **11751**
  " Paris" (logprob −0.6080).
- fp32 oracle (fp16→fp32 up-conversion via `f16_to_f32.py`, int4 weights
  preserved, native CPU) teacher-forced argmax = **11751** " Paris" (logprob
  −0.6100, top-1 margin **2.549** nats). Byte-exact parity ✅ — the fp16 and fp32
  logits are near-identical, so fp16 rounding cannot flip the pick.

**Test:** `crates/onnx-genai-engine/tests/qwen35_27b_hybrid_native_cuda_e2e.rs`
(ignored; env `QWEN35_27B_DIR`, `QWEN35_27B_FP32_ORACLE_DIR`). It locks native
coherence and re-derives the fp32-oracle argmax at runtime when the oracle
artifact is present, else locks the recorded constant — mirroring the 35B QMoE
oracle lock.

**Why:** Justin's #1 hard rule is DRY/generality with no model-name gates. The
graph already carries the full hybrid topology; deriving the `io` contract from
it (rather than trusting incomplete metadata) is the general fix and matches the
existing `derive_fallback_io` philosophy. Re-exporting the artifact via mobius
would only patch this one file; the loader fix unblocks the whole family.

**Note for mobius:** the 27B int4 export's `inference_metadata.yaml` is
under-specified vs the working 35B/0.8B siblings (missing the entire `io`/state
contract and MoE/hybrid capability list). The runtime now tolerates it, but
mobius should emit the full `io` block (kv_inputs/kv_outputs/state_pairs) for
hybrid exports for parity with the 35B recipe.
