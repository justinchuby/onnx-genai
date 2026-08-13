### 2026-08-12: CUDA-graph capture for Muse-Glimmer native decode is blocked by 3 stacked cross-domain issues — escalate to Deckard (KV geometry) + Batty (pipeline decode-loop)

**By:** Sebastian

**What:**
Investigated why CUDA-graph capture does not engage for Muse-Glimmer-30B native
CUDA decode (the only lever to close the 11→40 tok/s gap vs ORT). Root cause is
NOT a single kernel — it is three stacked blockers, two of which are squarely in
other specialists' domains. I did NOT land a code fix this session because (a) the
model cannot be loaded on any engine native-decode path today, so no before/after
tok/s could be measured (charter rule: every perf claim must be measured), and
(b) the tractable fix I could own (sliding-window misclassification) is unsafe to
land blind because it would touch the shared classifier that keeps Gemma/Mistral
SWA models correct, with no way to regression-measure Muse-Glimmer end-to-end.

**The 3 blockers (precise, file:line):**

1. **LOAD — native pipeline decode not wired for this pipeline (Batty's domain).**
   Muse-Glimmer's decoder is *embeds-driven* (graph input `inputs_embeds`, no
   `input_ids`), so it is inherently a pipeline (embedding→decoder), not a
   single decoder. `profile_native --model` (single-decoder path, feeds token
   ids) therefore cannot drive it, and a hand-staged decoder-only dir hits a role
   collision in `native_decode/load.rs:455-612` (embeds decoder + token-id
   feeding auto-detects `attention_mask` as both TokenIds and AttentionMask).
   The only viable engine path is `--pipeline`, but
   `crates/onnx-genai-engine/src/pipeline/mod.rs:338-415` only wires native decode
   for **flat-autoregressive** pipelines (GAP-3 Inc-A). Muse-Glimmer's multimodal
   (vision+embedding+decoder) pipeline routes its embedding component to ORT,
   which lacks a bf16 `Where(16)` impl → load fails
   ("Could not find an implementation for Where(16)"). NOTE: the `muse_decode`
   bench harness runs BOTH embedding+decoder on the native CUDA EP fine (raw
   sessions) — so the native EP *can* run this model; it is the ENGINE pipeline
   plumbing that routes embedding to ORT. Making PipelineEngine run the embedding
   component natively (as `muse_decode` does) dissolves this blocker.

2. **CLASSIFY — vestigial sliding_window forces the non-capturable growing path (Deckard's domain).**
   `detect_model_decode_path` (`crates/onnx-genai-engine/src/decode/metadata.rs:108-130`)
   hard-routes ANY model with `sliding_window.is_some()` to
   `PastPresent { shared_buffer:false }` (growing/paged, capture-unstable), on the
   documented assumption that "the graph remains responsible for local-attention
   masking." That assumption is FALSE for Muse-Glimmer: its 52 GroupQueryAttention
   ops carry NO `local_window_size` attribute (attrs: num_heads=32, kv_num_heads=2,
   scale, do_rotary=1, rotary_interleaved=0) → attention is GLOBAL. The
   `sliding_window:2048` exists ONLY in our generated `inference_metadata.yaml`;
   the model's own `genai_config.json` declares NO sliding_window and
   `past_present_share_buffer: true` (line 92). So the window is vestigial and the
   routing is a misclassification that blocks the capture-stable shared-buffer path.
   Correct fix: treat `sliding_window` as active only when the decoder graph
   actually enforces it (GQA `local_window_size` present, or a windowed-mask
   construction detected) — otherwise fall through to the shared-buffer branch.
   This is KV-geometry/classifier surgery that must be regression-tested against the
   real SWA models (Gemma/Mistral) it currently protects.

3. **CAPTURE — once on a fixed-capacity/shared-buffer path, engage capture.**
   `ONNX_GENAI_CUDA_GRAPH` defaults OFF
   (`crates/onnx-genai-runtime-config/src/lib.rs:137,333`), and the capture
   classifier vetoes any node whose shapes reference a growing KV symbol
   (`executor/capture.rs:592-631`, growing symbols seeded in
   `executor/kernel_cache.rs:186-195`). Once blockers 1+2 put decode on the
   shared-buffer path (fixed KV addresses/shapes), capture should engage. This is
   proven infra: the CUDA fixed-capacity present-binding shared-KV path took
   Qwen2.5-0.5B from ~11 to ~265 tok/s (see KV notes in
   `crates/onnx-genai-ort/src/session/mod.rs:593-640`). CUDA is on the
   `supports_fixed_capacity_present_binding` allowlist.

**Why:**
The 11→40 gap is dispatch/launch-overhead bound (my prior measurement: GPU ~99%
idle, ~1600 kernel launches/token — recorded in
`sebastian-native-cuda-decode-perf.md`, merged in #840). Collapsing those launches
into one captured graph replay is the only path to 40+. But capture cannot engage
until the model (a) loads on a native decode path and (b) is classified onto a
fixed-KV, capture-stable path. Blocker 1 = engine pipeline decode-loop (Batty);
blocker 2 = KV-geometry/decode-path classifier (Deckard). I can own the
sliding-window-enforcement detection (my portability remit) but it must land
*coordinated* with Deckard's classifier changes and Batty's native-pipeline-embedding
enablement so it can be measured before merge. Recommend the coordinator bring in
Deckard + Batty; I'll pair on the sliding-window detection + re-measure captures /
launches-per-token / tok/s once the load path is open.
