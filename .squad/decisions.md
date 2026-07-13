# Decisions

Canonical, append-only record of accepted team decisions. Only the Coordinator (via Scribe merge) writes here. Agents drop proposals in `decisions/inbox/`.

---

### 2026-07-12T14:50:00-07:00: Decisions archive rollover
**By:** Scribe
**What:** Archived the fully merged canonical decision file to `decisions/archive/2026-07-12T14-50-00Z-decisions.md` after the publish/audit merge pushed `decisions.md` above 20480 bytes.
**Why:** Keep the hot decisions file small while retaining the complete merged inbox record, including publish, CI, audit, security, and speculative-runtime notes, under `.squad/decisions/archive/`.

---

### 2026-07-12T14:50:00-07:00: PUBLISHED to crates.io: onnx-genai v0.1.0 and seven sub-crates
**By:** Scribe
**What:** Published `onnx-genai v0.1.0` plus all seven sub-crates to crates.io: `onnx-genai-metadata`, `onnx-genai-kv`, `onnx-genai-scheduler`, `onnx-genai-ort`, `onnx-genai-ort-sys`, `onnx-genai-engine`, and `onnx-genai-server`. The release contract is `.github/workflows/publish.yml` using the protected `crates` environment and `CARGO_REGISTRY_TOKEN`; it publishes leaves-first, checks crates.io before each publish, skips versions already present, uses a User-Agent header for registry API calls, and is safe to re-run idempotently. Future releases are performed by bumping the workspace version and re-running the workflow.
**CI/Audit:** `.github/workflows/ci.yml` runs fmt/build/test on push and PR; clippy remains non-blocking until warning cleanup makes `-D warnings` viable. `.github/workflows/audit.yml` runs weekly cargo-audit and on dependency changes; fresh `cargo audit` found 0 vulnerabilities. Audit code regularly via scheduled cargo-audit plus periodic review passes.
**Security:** Batched-driver DoS findings were fixed with bounded active+pending admission (`max_pending`, HTTP 429 + `Retry-After`) and non-blocking bounded delivery that drops slow/closed clients instead of stalling the shared driver.
**Speculative runtime:** §27/§28 includes prompt-lookup n-gram speculation (`NgramProposer`, greedy-identical), `MtpProposer` and `SpeculativeMode::Mtp`, a full ignored tiny-MTP package/e2e fixture (`tiny-mtp-full`) that matches greedy, an EAGLE-3 fixture with proposer TBD, and speculator config auto-discovery for vLLM-style metadata. Remaining optional speculative work: EAGLE-3 proposer and optimized full-MTP hidden-output decode.
**Model policy:** Agents use task-appropriate models with `gpt-5.5` as the floor.
**Why:** The repository is now published, continuously validated, routinely audited, and has the complete runtime milestone set recorded in the canonical team log.

---

### 2026-07-12T14:50:00-07:00: Release and audit conventions
**By:** Scribe
**What:** Publishing is via `.github/workflows/publish.yml` using the protected `crates` environment; bump the workspace version and re-run the workflow for future releases. Audit code regularly through scheduled cargo-audit plus periodic review passes.
**Why:** These contracts should be durable and visible to future agents before they change release or security workflows.

---

### 2026-07-12T14:56:00-07:00: Make workspace clippy blocking
**By:** Batty
**What:** Cleared workspace clippy warnings covering dereferenced types matching their origin, collapsible conditionals, a derivable `Default`, needless range indexing, a large enum variant, explicit `.into_iter()`, needless `Ok`/`?`, and unnecessary mutable bindings. The private server driver command now boxes its oversized generation request variant. No `#[allow]` attributes were added. CI clippy now runs with `-D warnings` and is blocking.
**Why:** A warning-free workspace lets CI enforce Rust idioms and prevent new clippy warnings without changing generation behavior or public API ergonomics.

---

### 2026-07-12T15:40:00-07:00: Decode ownership boundary
**By:** Batty
**What:** Audited ORT `decode.rs`/`mtp.rs` against engine `decode.rs`, `decode_loop.rs`, and `kv_bridge.rs`. ORT decode sessions correctly owned single-forward tensor binding plus KV buffers/cursors/rewind, but `MtpDecodeSession::propose`, target embedding/LM-head abstractions, and argmax token selection were generation policy leaked into ORT. Moved that proposal loop and its target-side helpers into engine `speculative.rs`; ORT `MtpDecodeSession` now exposes only detection, one-step hidden-state execution, KV state, reset, and rewind. Documented ORT as single-forward/KV-buffer ownership and engine as token selection, stopping, multi-step generation, constraints, and logical KV policy. The explicit adapter seam is engine's `DecodeBackend` trait; `ModelDecodePath` is the model-I/O strategy enum used to select its implementation, not a trait.
**Why:** This keeps runtime tensor/KV mechanics close to ORT while ensuring every decision about which token to generate or when generation ends remains in the engine, without changing proposal behavior.

---

### 2026-07-12T15:20:00-07:00: Establish LLVM coverage baseline and target high-value gaps
**By:** Pris
**What:** Installed `llvm-tools-preview` and `cargo-llvm-cov` 0.8.7, added `scripts/coverage.sh`, documented coverage usage, and added deterministic tests for speculative prompt-lookup rejection/configuration edges, JSON constraint finish/stop behavior, paged-KV validation/eviction/prefetch errors, int8 copy-on-write rewrites, and prefix-cache active/missing release/eviction behavior. Comparable pre/post coverage excluding concurrently changing `onnx-genai-server` and newly added `onnx-genai-bench` rose from 73.83% to 74.60% lines and 73.15% to 73.85% regions. In that scope, engine rose 73.21%→73.80% lines / 70.32%→70.87% regions and KV rose 90.41%→93.63% lines / 91.04%→92.90% regions. Final full-workspace coverage is 75.63% lines and 74.34% regions: onnx-genai 0.00%/0.00%, onnx-genai-bench 0.00%/0.00%, engine 74.87%/71.81%, KV 93.63%/92.90%, metadata 80.87%/79.35%, ORT 68.67%/70.54%, scheduler 91.70%/88.48%, server 80.05%/76.63% (line/region). `onnx-genai-ort-sys` generated bindings do not appear as instrumented workspace source. The initial full-workspace run was blocked by concurrent incomplete server edits, so no fabricated server/bench “before” number is reported.
**Why:** The largest remaining meaningful gaps are engine KV/ORT bridge failure and rewind paths (30.48% lines), engine decode variants and ORT error paths, ORT decode/value/allocator/chat-template failures, server driver shutdown/overload/session-error/streaming-tool-call branches, and untested CLI/benchmark entry points. Add a CI coverage job using `scripts/coverage.sh --fail-under-lines 75`, publish an HTML or LCOV artifact, and ratchet the floor upward without reducing coverage; prioritize getting engine above 75% and adding targeted ORT decode error-fixture tests. `cargo test --workspace` and blocking `cargo clippy --workspace --all-targets -- -D warnings` pass.

---

### 2026-07-12T15:20:00-07:00: Add OpenAI-compatible legacy completions endpoint
**By:** Rachael
**What:** Added `POST /v1/completions` with OpenAI legacy text-completion request/response mapping, including `prompt`, optional `suffix`, sampler parameters, stop sequences, usage, and SSE streaming chunks followed by `[DONE]`. Requests without `suffix` use normal generation; requests with `suffix` route through the engine's `generate_fim_with_config` path using the model's auto-detected `FimConfig`. Models without recognized FIM tokens return HTTP 400 when `suffix` is supplied.
**Why:** This exposes the engine's existing FIM/infilling capability for coding-agent clients while sharing the server generation driver, admission cap, context/output limits, error mapping, and plain-completion session handling.

---

### 2026-07-12T15:56:00-07:00: OpenAI-compatible vision input routing
**By:** Rachael
**What:** `/v1/chat/completions` now accepts message content as either a string or OpenAI-style `text`/`image_url` parts. Image URLs support base64 `data:image/...` URIs and bounded, timeout-protected HTTP(S) fetches. Pipeline startup inspects the loaded encoder's `pixel_values` metadata to derive endpoint, layout, and fixed image size; decoded RGB images are resized and normalized to `[0, 1]` f32 tensors. Vision requests use `PipelineGenerateRequest`, while single-model requests containing images return HTTP 400.
**Why:** The engine already supports external pipeline tensors, so the server can add vision without changing engine or ORT internals. `[0, 1]` is the only generic normalization available because current pipeline metadata does not declare processor mean/std. A real production vision model still needs a real CLIP-style encoder plus compatible decoder/image-feature contract packaged as a VLM pipeline via mobius; Pris should supply that model package and its processor metadata so model-specific normalization can replace the generic fallback.

---

### 2026-07-12T15:40:00-07:00: Split the HTTP server library into focused modules
**By:** Rachael
**What:** Reduced `onnx-genai-server/src/lib.rs` to public re-exports plus `app()`/`serve()` wiring. Request/response schemas moved to `types.rs`, handlers and request preparation to `routes.rs`, batch-driver ownership and event routing to `driver.rs`, session registry and caps to `session.rs`, streaming buffers/chunks to `sse.rs`, and state/config/model loading to `state.rs`; unit tests moved to `tests.rs`.
**Why:** Issue #2 called for maintainable module boundaries around the 2,713-line server library. The crate-root re-exports preserve the existing public Rust API, and endpoint behavior, streaming, tool handling, admission/session caps, and driver logic are unchanged.

---

### 2026-07-12T15:20:00-07:00: Add a device-comparable Criterion benchmark crate
**By:** Sebastian
**What:** Added the non-published `crates/onnx-genai-bench` workspace member. Its default Criterion target covers tokenizer encode/decode, greedy/top-k/top-p/min-p sampling, paged KV allocation/deallocation, a seven-processor logit chain, and llguidance grammar masking without loading a model. The `bench-ort` target covers tiny-model E2E generation, scatter-cache prefill lengths, and scatter static batches. Run `scripts/run_benchmarks.sh` (or add `--model`) to print machine metadata and a comparable scenario/metric table while retaining Criterion reports under `target/criterion`.
**Why:** A fixed suite, stable scenario names, explicit units, and machine metadata allow measurements from the same commit to be saved and diffed across CPUs and devices without placing benchmarks on the default test execution path. A short validation run on an Apple M1 Max (10 cores, Darwin 25.5.0 arm64, rustc 1.97.0) measured encode 3.47M tokens/s, decode 26.92M tokens/s, greedy 80.4µs/token, top-k 456.8µs/token, top-p 718.0µs/token, min-p 327.0µs/token, KV page cycling 532.8K pages/s, the processor chain 687.4µs/step, and grammar masking 78.5µs/step. These are smoke-run samples, not publication-quality measurements; use the runner's default Criterion duration for comparisons.

---

### 2026-07-12T16:06:00-07:00: Add low-overhead server observability core
**By:** Sebastian
**What:** Added a process-wide atomic metrics registry with `onnx_genai_*` request, token, TTFT, end-to-end latency, session, queue, batch, prefix-cache, and rejection metrics. Added unauthenticated `GET /metrics` behind the default-on `metrics` Cargo feature and always-on `GET /v1/status`. Instrumentation covers HTTP request spans/trace IDs, route status counters, driver queue/batch lifecycle, generated tokens, prefix-cache results, sessions, TTFT, and generation latency.
**Why:** DESIGN §§31–32 require production metrics and quick status introspection with negligible hot-path overhead. The implementation uses relaxed atomics and fixed-size histogram/status arrays with no per-request registry allocations or background exporter. Perfetto tracing, OTLP, and the full `/v1/debug/*` suite remain future work.

---

### 2026-07-12T16:14:00-07:00: Planning batch closed issues #2-5 and recorded next gaps
**By:** Scribe
**What:** Issues #2-5 are closed. #2 split `onnx-genai-server/src/lib.rs` into focused `routes`, `driver`, `sse`, `types`, `state`, and `session` modules while preserving public re-exports. #3 clarified decode ownership: ORT owns forward execution plus KV buffers/cursors/rewind, while the engine owns generation loops, token policy, stopping, constraints, and the `DecodeBackend` seam. #4 added OpenAI-compatible `POST /v1/completions` for legacy completions and FIM via `suffix`. #5 added the non-published `onnx-genai-bench` Criterion crate and device-comparable benchmark runner.
**OpenAI surface:** Current surface covers chat completions, tool calls, FIM through `/v1/completions`, image input parts for `/v1/chat/completions`, and streaming. Vision accepts base64 data URIs and bounded HTTP(S) image URLs, preprocesses to pipeline tensors, routes VLM pipeline requests, and returns HTTP 400 for image input on non-VLM single-model paths. Real vision quality still requires a production mobius CLIP+decoder VLM package and processor metadata.
**Observability:** Added low-overhead atomic metrics, request spans and trace IDs, default-on `metrics` feature with `GET /metrics` Prometheus output, and always-on `GET /v1/status`. Instrumentation covers request/status counts, driver queue/batch/session state, generated tokens, TTFT, end-to-end latency, prefix-cache hits, and 429 rejections. Perfetto, OTLP, and full debug endpoints remain deferred.
**Coverage:** `cargo-llvm-cov` is available via `scripts/coverage.sh`. Full workspace baseline is 75.63% line / 74.34% region overall: KV 93.63, Scheduler 91.70, Server 80.05, Engine 74.87, ORT 68.67 line coverage. Proposed CI floor is `scripts/coverage.sh --fail-under-lines 75`; biggest high-value gap is engine `kv_bridge`.
**Why:** This records the completed planning batch, preserves endpoint/metrics/benchmark/coverage contracts, and makes remaining follow-ups visible to future routing.

---

### 2026-07-12T16:14:00-07:00: CI and deterministic real-model test conventions
**By:** Scribe
**What:** CI clippy is blocking and must run as `cargo clippy --workspace --all-targets -- -D warnings`. Real-model exact-equality tests that compare floating-point generation outputs should force ORT single-thread execution with `intra_op_threads=1` to avoid reduction-order tie flakes; production defaults remain unchanged.
**Why:** Future agents need to treat clippy warnings as build failures and keep deterministic exact-equality tests stable across ORT scheduling differences.

---

### 2026-07-12: EAGLE-3 hidden-state and draft-state contract
**By:** Batty
**What:** Added `SpeculativeMode::Eagle3(Eagle3Config)` with exactly three ordered target hidden outputs (low, middle, high). The engine extracts each output's last-token row into the optional `SpeculativeProposerContext::target_hidden_layers`; `Eagle3Proposer` concatenates them for `fused_hidden`, uses the high layer as the initial `recycled_hidden`, then feeds each `next_hidden` output back autoregressively. The ORT `Eagle3DecodeSession` matches the committed fixture's `inputs_embeds`/`fused_hidden`/`recycled_hidden`/KV inputs and `draft_logits`/`next_hidden`/present-KV outputs. Accept and rewind reset draft-only state because every verification pass supplies a fresh target anchor.
**Why:** EAGLE-3 requires multi-layer verifier features rather than MTP's single last-layer hidden state, while optional context fields preserve other proposer behavior. Resetting between verification passes prevents rejected draft features or KV from becoming stale. The tiny fixture exercises a 48-wide fused input and hidden recycling, but its generated attention-mask path only runs with empty past (`HiddenThreaded`), its 16-logit reduced vocabulary has no mapping artifact, and it is not packaged with a target model exposing three matching hidden outputs plus target embedding weights. A fuller end-to-end fixture should support non-empty draft KV, include those verifier outputs and exact embedding weights, and provide draft-to-target vocabulary mapping when IDs are not identity-mapped.

---

### 2026-07-12T16:26:00-07:00: Add opt-in engine token logprobs
**By:** Batty
**What:** `GenerateOptions::top_logprobs: Option<usize>` enables logprob capture; `None` is disabled. `GenerateResult::logprobs` is `Option<Vec<TokenLogprob>>`, where each entry contains `token_id`, the chosen token's `logprob`, and descending `top: Vec<(TokenId, f32)>`. The top list contains the requested highest-probability tokens and always includes the chosen token. Values are natural-log softmax probabilities computed from finite logits after the complete processor chain, including temperature and sampling filters, exactly matching the distribution used to select the token.
**Why:** The engine must expose confidence data without allocations or log-softmax work for default requests. Rachael should map these entries to chat `choices[].logprobs.content[]` records (token text, chosen logprob, and `top_logprobs`) and legacy completions `choices[].logprobs` arrays/maps, using tokenizer-decoded token ids; streaming should attach each token's metadata to its corresponding chunk.

---

### 2026-07-12T16:16:00-07:00: Restore real categorical sampling
**By:** Batty
**What:** Fixed every engine generation path that passed a hardcoded `0.0` to categorical sampling. The engine now uses `rand` with an optional `GenerateOptions::seed`, advances a per-request RNG for each sampled token, and maintains independently seeded RNG state per static/continuous batch row using `seed + row_index`. Greedy selection still routes through `GreedySampler`, does not draw from the RNG, and retains its previous deterministic output.
**Why:** A zero inverse-CDF target always selected the first eligible token, making temperature, top-p, top-k, and min-p sampling deterministic and distributionally wrong. A 100,000-draw fixed-seed test for logits `[0, 1, 2]` passed with every empirical frequency within one percentage point of the expected softmax probabilities (approximately 9.00%, 24.47%, and 66.52%); seed reproducibility, seed sensitivity, greedy non-advancement, and independent seedable batch rows are also covered.

---

### 2026-07-12T16:40:00-07:00: Make ORT compaction comparisons tolerant and serialized
**By:** Deckard
**What:** Audited every real-model session in `crates/onnx-genai-ort/tests/decode_session.rs`: `load_session` and all three direct `Session::new` sites already use `deterministic_session_options()` with `intra_op_threads=1`, including the compaction test's individual traces, full-batch reference, compacted batch, and admitted replacement path. The remaining flaky comparisons were batched forwards versus batch-one individual forwards, whose GEMM reduction structures can differ despite single-thread execution. Those comparisons now require every logit to be within `1e-4`; differing argmaxes are accepted only when both outputs show the competing logits are tied within that tolerance. Compacted-versus-full-batch output still uses the tighter existing `1e-5` comparison. The five ORT-heavy tests in the file also share a mutex because concurrent test execution reproduced an ORT session-initialization race; the pure tensor round-trip test remains parallel.
**Why:** The test verifies that compaction preserves active-row numerical outputs and token choice, not bit-identical results between structurally different batch-one and batched matrix operations. Single-threading fixes reduction-order scheduling nondeterminism, tolerance handles legitimate batch-shape floating-point differences near ties, and serialization prevents independent ORT session setup in this test binary from racing. Final validation completed with 15/15 consecutive `cargo test -p onnx-genai-ort --test decode_session` runs and 8/8 consecutive full-parallel `cargo test --workspace` runs, all with zero failures. `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` also passed.

---

### 2026-07-12: Use LLaVA best-resolution scoring and tile-aware preprocess results
**By:** Deckard
**What:** `dynamic_anyres` treats each configured aspect ratio as a columns-by-rows tile grid, ignores grids whose local tile count exceeds `max_tiles`, then maximizes effective source pixels after an aspect-preserving fit and minimizes wasted canvas pixels, with configuration order as the final tie-breaker. Tiled output places an optional global thumbnail first, followed by row-major local tiles. `ImageTensor` now reports total and per-image tile counts, original sizes, and exposes each tile's normalized contiguous data.
**Why:** This matches the established LLaVA/Gemma any-resolution selection behavior, keeps tile ordering deterministic, and gives callers the post-preprocessing tile count needed to expand image tokens and allocate KV cache capacity.

---

### 2026-07-12T16:46:00-07:00: Paged / block-table KV cache in Mobius
**By:** Deckard
**What:** In the separate `onnxruntime/mobius` repo, branch `feat/paged-cache` was pushed and draft PR https://github.com/onnxruntime/mobius/pull/395 opened. The implementation adds a paged-cache static-cache variant using standard ONNX ops: `PagedCacheState`, `ScatterND` writes into K/V pools, `Gather` assembles logical pages, `Attention` consumes gathered K/V with `nonpad_kv_seqlen`, and CLI/task flags expose `--paged-cache`, `--page-size`, and `--num-pages`. The onnx-genai runtime-side contract is single active sequence (`batch == 1`) with `key_pool.{i}` / `value_pool.{i}` `[num_pages,page_size,kv_hidden]`, `block_table [num_blocks]`, `slot_mapping [seq_len]`, and `nonpad_kv_seqlen [batch]`; outputs are `logits` plus updated pools. Validation covered graph structure, shape inference, MoE/dense paths, scatter/gather parity, and checker runs; end-to-end paged execution remains CUDA-kernel gated and multi-sequence batching is TODO.
**Why:** This gives onnx-genai a concrete Mobius export format for paged/block-table attention that supports non-contiguous KV pools and RadixAttention-style shared pages while staying within standard opset-24 graph constructs.

---

### 2026-07-12T16:12:00-07:00: Extract native preprocessing into a dedicated crate
**By:** Deckard
**What:** Added publishable `onnx-genai-preprocess` with public `audio` and `image` modules. Audio exposes `LogMelExtractor` and `decode_wav_pcm16`; image exposes `ImagePreprocessor`, `ImagePreprocessConfig`, layout/resize/normalization types, and tensor output. Whisper log-mel/WAV code moved from the facade, while image decode/resize/crop/normalize and §35 metadata resolution moved from the server. `onnx-genai` re-exports the crate as `onnx_genai::preprocess`; the server retains only bounded data-URI/HTTP loading and calls the crate.
**Why:** Native §35 preprocessing must be reusable by server and CLI without ORT Extensions or embedding model transformations in transport/facade crates. The implementations and their tests moved unchanged in behavior, preserving audio features and image tensor results.

---

### 2026-07-12T16:22:00-07:00: Raise kv_bridge deterministic coverage
**By:** Pris
**What:** Added four deterministic `kv_bridge` unit-test entry points covering past/present metadata inference, static-cache exclusion, present-KV token extraction and paged mirroring, materialization back into ORT past tensors, prefix-page attachment/reuse, page selection, rewind/clear/error branches, overmaterialized target trimming, and both static-cache and past/present decode-runner rewind paths. Real-model cases use `tiny-llm` and `tiny-llm-scatter` with `intra_op_threads=1`, are serialized into one model-backed test, and passed five repeated runs. Comparable LLVM coverage moved `kv_bridge.rs` from 30.48% to 94.51% lines (27.70% to 89.31% regions); observed full-workspace line coverage moved from 74.53% to 77.27% (+2.74 points). The after snapshot also contains concurrent unrelated server/preprocessing edits, so the overall delta is observational rather than isolated. `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, and fmt pass. No engine bug was found for Batty.
**Why:** `kv_bridge.rs` was the largest high-value engine coverage gap. These tests exercise the bridge's real tensor layouts, paged-cache data preservation, prefix reuse, materialization, rewind behavior, and failure handling without changing production logic.

---

### 2026-07-12T16:26:00-07:00: Establish Whisper encoder-decoder model contract
**By:** Pris
**What:** Mobius supports `--task speech-to-text` (also auto-selected for `model_type=whisper`) and emits `encoder/model.onnx` plus `decoder/model.onnx`. The generic encoder contract is `input_features: fp32[B, num_mel_bins, audio_seq_len] -> encoder_hidden_states: fp32[B, audio_seq_len/2, d_model]`; the generated `openai/whisper-tiny` graph exposes `[B,80,audio_seq_len] -> [B,1500,384]` and therefore expects the standard 3000-frame input. Its decoder takes `decoder_input_ids: i64[B,S]`, `encoder_hidden_states: fp32[B,E,384]`, `position_ids: i64[B,S]`, and four layers of self-attention `past_key_values.{layer}.{key,value}: fp32[B,6,P,64]`, producing `logits: fp32[B,S,51865]` and `present.{layer}.{key,value}: fp32[B,6,P+S,64]`. Mobius recomputes cross-attention K/V from `encoder_hidden_states` on each decoder call; it does not expose separate cross-attention KV cache ports.
**Why:** Added a 80-mel, 8-frame synthetic package under `tests/fixtures/tiny-whisper/`, generated reproducibly by `scripts/build_tiny_whisper.py` with onnx-ir. Its composite metadata runs the encoder once at `prompt_only`, routes `encoder.encoder_hidden_states` to `decoder.encoder_hidden_states`, and runs the decoder autoregressively. PipelineEngine already routes this cross-attention tensor as a persistent decoder extra; the only engine gap was recognizing Mobius' `decoder_input_ids` token-input name, so decode input matching now accepts that alias. An ignored WAV -> native log-mel -> pipeline -> token test verifies the complete contract.

---

### 2026-07-12T16:54:00-07:00: Add OpenAI-compatible audio input and transcription routing
**By:** Rachael
**What:** `/v1/chat/completions` accepts one `{type:"input_audio",input_audio:{data,format}}` part with base64 PCM16 WAV data; MP3 currently returns HTTP 400. Added `POST /v1/audio/transcriptions` multipart handling for `file` plus optional `model`, `language`, and `response_format=json|text`. Audio pipeline startup detects a Float32 `input_features` encoder input, derives its mel/frame shape, preprocesses WAV audio with `onnx-genai-preprocess` log-mel extraction, and supplies the tensor to the existing encoder-to-decoder `PipelineEngine`. Decoder prompts use Whisper start/language/transcribe/no-timestamps tokens when present, and output token ids are decoded by the pipeline tokenizer.
**Why:** This exposes both requested OpenAI audio surfaces without changing engine or ORT ownership. Requests against non-audio models return HTTP 400, generation remains bounded by request/server/pipeline token caps, and the tiny Whisper fixture validates routing. A real Whisper package still needs production encoder/decoder weights, a complete Whisper tokenizer with language/task special tokens and EOS configuration, fixed 30-second `[1,80,3000]` (or model-declared) feature input, and model-appropriate chunking/timestamp behavior for long audio and accurate transcription.

---

### 2026-07-12T16:30:00-07:00: Make vision preprocessing metadata-driven
**By:** Rachael
**What:** Server vision preprocessing now reads `preprocessing.image` metadata for resize size/mode, crop, interpolation, and per-channel mean/std normalization. Bicubic maps to the image crate's Catmull-Rom filter; supported resize paths are shortest-edge plus center crop, fixed resize, and longest-edge plus padding. Defaults are model input dimensions, shortest-edge center crop, bicubic interpolation, and `[0, 1]` normalization when metadata is absent.
**Why:** CLIP and other vision encoders require their exported processor geometry and normalization for faithful embeddings, while metadata-driven parameters avoid baking one model's constants into the server and retain compatibility with legacy pipeline packages.

---

### 2026-07-12T16:20:00-07:00: DESIGN §1-§38 implementation audit
**By:** Roy
**What:** Read-only audit found core generation, serving, KV, scheduler, metadata, ORT wrapper, tools, grammar constraints, FIM, batching, metrics, and most native preprocessing architecture implemented. Remaining or partial gaps cluster around `/v1/embeddings`, logprobs, multi-model lifecycle, EAGLE-3 proposer completion, audio/Whisper surface, debug/Perfetto/OTLP, image preprocessing fidelity, diffusion, cluster routing, external KV connectors, and metadata-declared queue-depth backpressure. The audit prioritized backlog: embeddings, logprobs, model lifecycle, EAGLE-3, audio endpoints, native log-mel, debug/profiling, image preprocessing fidelity, fp8 KV, diffusion, cluster router, language diffusion, real Mobius VLM/ASR packages, distributed KV, and queue-depth metadata. `cargo test --workspace` passed during the read-only validation.
**Why:** Future routing should focus on the remaining high-value OpenAI-compatible runtime gaps rather than re-auditing already completed design sections. Audio was architecturally feasible through the existing encoder-decoder pipeline once native mel preprocessing and a Mobius Whisper package exist.

---

### 2026-07-12T16:30:00-07:00: Add native Whisper log-mel preprocessing
**By:** Sebastian
**What:** Added reusable `onnx_genai::preprocess::audio` APIs. `LogMelExtractor` resamples mono f32 PCM to 16 kHz, computes a centered STFT with a periodic Hann window (`n_fft=400`, `hop=160`), applies Slaney-normalized 80- or 128-bin mel filters over 0–8 kHz, then applies Whisper's log10, max-minus-8 clamp, and `(x+4)/4` normalization. Dynamic extraction returns `[1, n_mels, n_frames]`; fixed extraction pads/truncates to 30 seconds and returns `[1, n_mels, 3000]`. `decode_wav_pcm16` decodes 16-bit integer PCM WAV and averages channels to mono.
**Why:** This provides the pure-Rust, ORT-Extensions-free DSP foundation required by issue #11 in the facade crate so CLI and server integrations can share it. Issue #12 must next accept audio content, decode WAV bytes (MP3 remains deferred), construct `input_features`, route them through a Whisper encoder/decoder pipeline, and map transcription output to the API.

---

### 2026-07-12: Add an OpenAI HTTP cross-runtime benchmark harness
**By:** Sebastian
**What:** Added a Rust comparison binary plus `scripts/compare_runtimes.sh` to probe onnx-genai, Ollama, and LM Studio, then measure fixed Qwen2.5-0.5B-Instruct short/long prompts with greedy streaming TTFT, decode throughput, total latency, and estimated prefill throughput. The Apple M1 Max baseline used onnx-genai's 1.98 GB f32 dynamic-cache ONNX model on CPU and identical 531.1 MB Q8_0 GGUF bytes in Ollama and LM Studio.
**Why:** A common API-level harness with warmups, median/p90, exact runtime/model labels, machine metadata, graceful skips, and saved Markdown reports makes periodic comparisons reproducible and prevents cherry-picked claims. The baseline verdict is that onnx-genai is currently slower on every reported median metric; quantized/fused ONNX execution, persistent IoBinding/KV reuse, and prefill/thread/cache-shape tuning are the next priorities.

---

### 2026-07-12: Runtime consumes fp16 GroupQueryAttention (GQA) WebGPU KV via genai_config share-buffer
**By:** Batty
**What:** Made the onnx-genai runtime load and decode Deckard's fp16 `com.microsoft::GroupQueryAttention` WebGPU export (`models/qwen2.5-0.5b-gqa-webgpu/`). Four changes:
1. **Accept fp16 KV.** `crates/onnx-genai-engine/src/kv_bridge.rs` no longer hard-requires Float32 present/past KV: `infer_kv_model_info` and `infer_kv_heads_and_head_dim` accept Float32 **or** Float16 (new `is_supported_kv_dtype` helper), matching the eagle3/mtp pattern. This removes the `present.0.key must be Float32 rank >= 3, got Float16` load failure. The host paged-mirror path stays fp32-only and now bails with a clear message if fp16 KV ever reaches it (it never does for GQA — see #2).
2. **Bypass the fp32 host cache for share-buffer KV.** GQA models take the existing `ModelDecodePath::PastPresent` runner (`DecodeSession`), so `next_session_token_logits` skips `mirror_present_kv_to_pages` entirely; KV OrtValues stay ORT-owned and the host `PagedKvCache` is used only for logical length bookkeeping (`append`). New: the runner now selects `DecodeKvMode::SharedBuffer` for GQA (present aliased onto the same max-length past buffer), not the CPU-round-tripping `ZeroCopyRebind`.
3. **Consume `genai_config.json`.** New module `crates/onnx-genai-engine/src/genai_config.rs` reads `search.past_present_share_buffer`, `search.max_length`, and `model.context_length`. `detect_model_decode_path` now takes this config: a `past_present_share_buffer: true` declaration is authoritative and pre-sizes the shared KV buffer to `max_length` (4096), falling back to `context_length`/metadata. This also bounds the context-window guard via `decode_path_max_len`.
4. **fp16 logits/hidden reads.** GQA emits fp16 `logits`; added `Value::to_vec_f32_lossy()` (ORT crate, uses `half`) that widens Float16 → f32 and passes Float32 through. Engine logits/hidden extractors (`extract_next_token_logits_from_outputs`, `extract_logits_sequence`, `extract_logits_value_sequence`, `extract_last_hidden`) now use it. Fixes `requested f32 data from Float16 tensor`.

Non-GQA fp32 / static-cache / CPU paths are unchanged (draft models pass `None` genai_config; share-buffer still requires an explicit declaration or ORT metadata + max context).

**Verification (real model, per-request server path):**
- `ONNX_GENAI_EP=webgpu` load succeeds and is **coherent**: "What is the capital of France?" → `Paris`; 128-token Eiffel-Tower essay is fluent.
- **WebGPU decode ≈ 21 tok/s** (isolated by differencing 8 vs 136 max_tokens), up from the prior WebGPU-fp16 **9 tok/s** (~2.3×). CPU EP on the same GQA model is coherent at **≈ 38 tok/s** (prior CPU reference 43). WebGPU is now far ahead of the old plain-Attention path but still trails CPU.
- `cargo test -p onnx-genai-engine -p onnx-genai-ort` → exit 0 (all pass, incl. new genai_config + `to_vec_f32_lossy` tests). `cargo clippy -p onnx-genai-engine -p onnx-genai-ort --all-targets -- -D warnings` → exit 0.

**Why / remaining optimization (follow-up):** The shared KV buffer is currently allocated with the default **CPU** allocator (`Value::empty` → `Allocator::default_cpu`). In SharedBuffer mode ORT therefore still copies each layer's KV host↔device per step, so WebGPU has not yet overtaken CPU. The remaining win is a **device-resident (WebGPU) shared KV buffer** plus enabling the genai_config WebGPU provider options (`enableGraphCapture: 1`, `validationMode: disabled`), which should eliminate the last H2D/D2H copies and let WebGPU decode pass CPU. Sebastian's harness run vs LM Studio is a separate follow-up.

**Files changed:** `crates/onnx-genai-engine/src/kv_bridge.rs`, `crates/onnx-genai-engine/src/decode.rs`, `crates/onnx-genai-engine/src/engine.rs`, `crates/onnx-genai-engine/src/lib.rs`, `crates/onnx-genai-engine/src/genai_config.rs` (new), `crates/onnx-genai-ort/src/value.rs`, `crates/onnx-genai-ort/Cargo.toml` (+`half`), `crates/onnx-genai-ort/tests/decode_session.rs`, `Cargo.lock`.

---

# Decision: Q4 GGUF→ONNX correctness fix (invalid MatMulNBits output)

- **Author:** Batty (Engine)
- **Date:** 2026-07-12T18:05:00-07:00
- **Requested by:** Justin Chu
- **Scope:** Mobius `src/mobius/integrations/gguf/**` only (no onnx-genai runtime change needed)

## Problem
`mobius build-gguf` converted `qwen2.5-0.5b-instruct-q4_0.gguf` (sha256 7671c0c3…)
into a `MatMulNBits` ONNX graph that **loaded** (168 nodes, CPU + WebGPU) but produced
**deterministic garbage** on both EPs (e.g. `辱字母…EventData`). llama.cpp/LM Studio
run the same GGUF correctly, so the GGUF weights are good and the conversion was wrong.

## Root cause — TWO independent conversion bugs (both required to fix)

### 1. Missing attention biases (primary)
`gguf_to_config()` never inferred projection-bias flags, so `attn_qkv_bias` defaulted
to `False`. Qwen2 carries Q/K/V biases (`blk.N.attn_{q,k,v}.bias`, magnitudes q≈79,
k≈130). With the flag false, the graph builder omitted the bias `Add` after each
projection (`MatMulNBits → RotaryEmbedding` instead of `MatMulNBits → Add(bias) →
RotaryEmbedding`), destroying attention. The fp16 reference graph has the `Add`.

### 2. Spurious / mis-shaped Q/K reverse-permute (secondary, also fatal)
- Qwen2 was mapped to the Llama Q/K interleaved-rope reverse-permute (`_PROCESSORS`
  + name-based `_needs_qk_permute`). **Qwen2/Qwen3 use NEOX rope and are NOT permuted
  by llama.cpp** — reverse-permuting scrambles the heads.
- `_reverse_permute` also used the *forward* reshape `(n_head, 2, dim)` instead of the
  inverse `(n_head, dim, 2)` (HF `modeling_gguf_pytorch_utils._reverse_permute_weights`
  uses `(n_head, dim, 2)`). This only accidentally inverts when `dim == 2`
  (head_dim == 4); for head_dim 64 it corrupts Llama/Mistral Q/K too. The paired test
  was self-consistently wrong (its `_forward_permute` used the inverse reshape), so it
  passed while production was broken.

Isolation proof: with biases restored but the Qwen2 permute re-enabled, output is
garbage again — confirming both bugs are independently fatal.

The MatMulNBits weight repacking itself was **correct**: dequant(repacked layer-0
q_proj) == GGUF Q4_0 dequant exactly (max_abs_diff = 0.0). Nibble order, scale, zp
(default 8 for symmetric Q4_0), block_size=32, and N/K were all fine.

## Fix (files changed, mobius)
- `_config_mapping.py`: add `_infer_attn_qkv_bias/_infer_attn_o_bias/_infer_mlp_bias`
  (tensor-presence inference, mirroring `_infer_tie_embeddings`); wire into
  `ArchitectureConfig`. Qwen2.5 now → `attn_qkv_bias=True, attn_o_bias=False,
  mlp_bias=False`.
- `_tensor_processors.py`: correct `_reverse_permute` reshape to `(n_head, dim, 2)`;
  add `LLAMA_QK_PERMUTE_MODEL_TYPES = {llama, mistral}` + `needs_llama_qk_permute`;
  remove `qwen2`/`qwen3` from `_PROCESSORS`.
- `_builder.py`: gate the quantized inline `_needs_qk_permute` on `model_type` via
  `needs_llama_qk_permute`.

## Tests added
- `_tensor_processors_test.py`: HF-reference reverse-permute round-trip at head_dim=64;
  fixed the self-wrong `_forward_permute`; Qwen not-permuted regressions; builder gate.
- `_reader_test.py`: projection-bias inference from tensor names.

## Verification (real exit codes)
- `pytest src/mobius/integrations/gguf/ -q` → **151 passed** (exit 0).
- Re-converted `models/qwen2.5-0.5b-q4-onnx-fixed` and ran onnx-genai server (CPU EP):
  - Before: `辱字母\`\amaisten…EventData`
  - After:  `The capital of France is Paris.` / `12 plus 7 is 19.` (matches fp16).

## Notes / follow-ups
- Only `src/mobius/integrations/gguf/**` changed; Deckard's files untouched.
- The `_reverse_permute` reshape fix also corrects a latent Llama/Mistral bug for any
  head_dim != 4.
- No commits made (coordinator commits).

---

### 2026-07-12: Target WebGPU builds at GroupQueryAttention
**By:** Deckard
**What:** Mobius WebGPU causal-LM exports use the existing EP-aware build context: `--ep webgpu` resolves `EpCapabilities.gqa_dtypes`, and fp16 standard-RoPE decoders emit `com.microsoft::GroupQueryAttention` directly with Q/K/V, past K/V, sequence lengths, and RoPE tables. Added explicit WebGPU fp16 regression coverage. The Qwen2.5-0.5B export contains 24 GQA and zero Attention nodes; ORT placed all 24 GQA nodes on WebGPU, with 268 WebGPU nodes, 6 CPU shape/seqlen nodes, one H2D copy node, and zero D2H copy nodes.
**Why:** The previous plain-Attention graph placed 24 Attention nodes on CPU and inserted 121 D2H plus 74 H2D copies per token. GQA removes that decomposition and keeps fused attention/RoPE/KV update in the WebGPU partition. The onnx-genai server still cannot consume the fp16 GQA cache: `crates/onnx-genai-engine/src/kv_bridge.rs` unconditionally requires Float32 present KV and mirrors it through host `PagedKvCache`; loading fails with `present KV output 'present.0.key' must be Float32 rank >= 3, got Float16 [-1, 2, -1, 64]`. Follow-up must route GQA models through ORT-owned shared/device KV buffers, propagate the Mobius `past_present_share_buffer`/max-length contract into onnx-genai model metadata or config loading, and skip host KV mirroring for that path.

---

### 2026-07-12: Make same-source GGUF the primary cross-runtime benchmark
**By:** Sebastian
**What:** Cross-runtime reports must identify one GGUF by filename and SHA-256, load those exact bytes in llama.cpp runtimes, and convert the same file through Mobius with `--keep-quantized`. The verified 2026-07-12 path uses official Qwen2.5-0.5B-Instruct Q4_0 because Q4_K_M currently fails Mobius weight-shape validation. onnx-genai successfully loaded 168 `MatMulNBits` projection nodes, but current Mobius dequantizes the quantized embedding and output head to fp32. The measured ORT CPU stack remained 77-80% slower in decode and much slower in prefill than Ollama/LM Studio Metal.
**Why:** The earlier fp32-ONNX versus GGUF baseline confounded runtime and quantization. Same-source conversion removes the dominant projection-weight mismatch, while explicitly recording the remaining embedding/head and CPU-vs-Metal limitations prevents the result from being overstated as perfectly byte- and device-identical.

---

### 2026-07-12: WebGPU benchmark correctness gate and fallback
**By:** Sebastian
**What:** The primary periodic comparison is onnx-genai WebGPU EP versus LM Studio Metal only. ORT 1.27 assigned all 168 Q4 `MatMulNBits` projections to WebGPU rather than CPU, but the converted Q4 graph produced the same invalid deterministic output on CPU and WebGPU, so it failed correctness and was not benchmarked. The reported GPU number therefore uses a non-quantized fp16 ONNX graph with fp32 logits/KV boundary casts versus the exact Q4_0 GGUF in LM Studio, clearly labeled as non-parity.
**Why:** GPU-vs-GPU is the fair runtime/backend comparison requested, but incorrect model output cannot support a performance claim. The fp16 fallback preserves a usable WebGPU measurement while exposing the current runtime constraints: attention and KV plumbing remain on CPU, native fp16 logits/KV are unsupported, and decode is substantially slower than both LM Studio and the historical CPU-EP baseline.

---

### 2026-07-12T19:38:00-07:00: GQA share-buffer KV driven by our own InferenceMetadata (not genai_config.json)

**Author:** batty (Engine)
**Supersedes:** the earlier `batty-fp16-gqa-kv` approach (merged into decisions.md, "Runtime consumes fp16 GroupQueryAttention (GQA) WebGPU KV via genai_config share-buffer"). Per Justin's architectural correction, the runtime NO LONGER reads onnxruntime-genai's `genai_config.json`. We use our OWN config (`InferenceMetadata` from `inference_metadata.yaml`) and the runtime owns/manages the KV cache; the GQA op is used for on-device attention compute only.

**What changed (crates/onnx-genai-engine + onnx-genai-ort):**
- **Deleted** `crates/onnx-genai-engine/src/genai_config.rs` and its `pub(crate) mod genai_config;` in `lib.rs`. No `genai_config.json` reads anywhere in the runtime.
- **`decode.rs`**: `detect_model_decode_path(session, metadata_max_context, shared_kv_max_len: Option<usize>)` now takes a metadata-derived shared-KV capacity instead of a `GenaiRuntimeConfig`. New `shared_kv_buffer_len_from_metadata(&InferenceMetadata) -> Option<usize>` derives the runtime-owned share-buffer decision from our metadata. When it returns `Some(max_len)`, the model takes `ModelDecodePath::PastPresent { shared_buffer: true, max_len: Some(max_len) }` (runtime-owned max-length KV buffer, `present.*` aliased onto `past_key_values.*` across steps). Helpers: `is_group_query_attention`, `metadata_kv_is_fp16`, `is_fp16_dtype`.
- **`engine.rs`**: loads `shared_kv_max_len` from the already-loaded `InferenceMetadata` (via `crate::decode::shared_kv_buffer_len_from_metadata`) and passes it to `detect_model_decode_path`. Draft models pass `None`.
- **No schema.rs change was needed** — all consumed fields already exist in `crates/onnx-genai-metadata/src/schema.rs`.

**(a) Exact InferenceMetadata fields the runtime reads for the GQA / runtime-owned share-buffer KV path:**

The runtime enables a **runtime-owned, max-length, share-buffer KV path** iff ALL THREE hold:
1. `model.attention.type` denotes group-query attention — accepted (case-insensitive, `-`/space normalized to `_`): `group_query_attention`, `grouped_query_attention`, `gqa`.
2. KV native dtype is fp16 — from EITHER `kv_cache.native_dtype` OR any entry of `model.runtime_configurable.kv_cache.dtype`; accepted values (case-insensitive): `float16`, `fp16`, `half`.
3. `model.max_sequence_length` is present — its value **pre-sizes the runtime-owned KV buffer (in tokens)**.

Additional fields consumed elsewhere (unchanged, informational for the emitter):
- `required_capabilities` — validated against the runtime's supported set. IMPORTANT: only emit capabilities the runtime supports; the runtime's supported list is `kv_cache`, `grouped_query_attention`, `multi_head_attention`, `prefix_cache`, `continuous_batching`. Do **not** require `fp16-kv` or hyphenated `group-query-attention` — those are NOT in the supported set and will fail model load with "Unsupported capabilities". Use `grouped_query_attention` (underscored) if you want to require GQA.
- `model.attention.{num_kv_heads,num_attention_heads,head_dim}` — informational/geometry (the KV tensor geometry the engine actually binds is still inferred from the ONNX `present.*` output shapes; these should match).
- `model.max_sequence_length` also bounds the context-window guard.

Notes for the emitter:
- `max_sequence_length` = intended **serving KV capacity** in tokens (it sizes the buffer). Emit the capacity you want the runtime to allocate (e.g. 4096), NOT necessarily the model's theoretical context window (Qwen2.5 = 32768) — a large value allocates a proportionally large KV buffer.
- fp32 / static-cache / CPU / non-GQA models simply omit these (or omit `inference_metadata.yaml` entirely) and keep their existing decode paths; no metadata is required for them.

**(b) Sample `inference_metadata.yaml` — the contract (committed at `models/qwen2.5-0.5b-gqa-webgpu/inference_metadata.yaml`, verified to run):**

```yaml
# onnx-genai inference metadata (our own config; replaces onnxruntime-genai's
# genai_config.json). The runtime reads these fields to own and manage the KV
# cache itself for the fp16 GroupQueryAttention WebGPU export: it allocates a
# single max-length KV buffer and aliases present.* -> past_key_values.* across
# decode steps, using the GQA op for on-device attention compute only.
required_capabilities:
  - grouped_query_attention
model:
  attention:
    type: group_query_attention
    num_kv_heads: 2
    num_attention_heads: 14
    head_dim: 64
  # Sizes the runtime-owned shared KV buffer (tokens). Emitters should set this
  # to the intended serving KV capacity, not necessarily the model's theoretical
  # context window.
  max_sequence_length: 4096
  runtime_configurable:
    kv_cache:
      dtype:
        - float16
kv_cache:
  native_dtype: float16
```

(Qwen2.5-0.5B geometry: 24 layers, num_attention_heads=14, num_key_value_heads=2, head_size=64.)

**(c) schema.rs fields added:** NONE. Everything is expressed with existing schema fields (`ModelCapabilities.attention` / `AttentionConfig`, `ModelCapabilities.max_sequence_length`, `ModelCapabilities.runtime_configurable` / `RuntimeConfigurable.kv_cache` / `RuntimeKvConfig.dtype`, `KvCacheSpec.native_dtype`).

**Verification:**
- Renamed `genai_config.json` away and ran `ONNX_GENAI_EP=webgpu ./target/release/onnx-genai-server --model models/qwen2.5-0.5b-gqa-webgpu` → model loaded from `inference_metadata.yaml` ALONE; `/v1/chat/completions` "capital of France" → **"Paris"** (coherent). `genai_config.json` restored afterward.
- Non-GQA regression: tiny fixtures (static-cache, tiny-llm) load with NO `inference_metadata.yaml` (`shared_kv_buffer_len_from_metadata` returns `None`); all engine/ort tests green.
- `cargo test -p onnx-genai-engine -p onnx-genai-ort` → exit 0. `cargo clippy -p onnx-genai-engine -p onnx-genai-ort --all-targets -- -D warnings` → exit 0.

**Follow-up (unchanged from before, separate task):** the shared KV buffer is still allocated with the default CPU allocator, so WebGPU still round-trips KV host↔device per step (~19–21 tok/s). Making it device-resident (WebGPU allocator) + WebGPU graph-capture provider options is the remaining perf win. Provider options previously came from `genai_config.json`; if we want them we must source them from our own config/EP setup, not genai_config.

---

### 2026-07-12: Bound runtime KV for uniform sliding-window attention
**By:** Batty
**What:** The engine now consumes `model.attention.sliding_window`. Past/present models use an engine-owned windowed decode path that keeps absolute position IDs while trimming ORT past tensors to the newest W tokens; long prefills are chunked so retained KV stays O(W). `PagedKvCache` tracks absolute sequence length separately from its page-aligned retained range and frees complete leading pages after each step. Non-SWA decode paths are unchanged. Static-cache SWA and fp16 GQA share-buffer SWA are rejected rather than run with incorrect indexing.
**Why:** ONNX Runtime `GroupQueryAttention.local_window_size` limits attention computation but its current shared past/present buffer remains append-only and sized to the full sequence; shrinking that buffer to W would write out of bounds and lose absolute RoPE positions. Mobius must emit local-window masking plus a bounded circular-cache/write-offset contract (or an equivalent graph-managed rotating cache with absolute positions) before the runtime can enable SWA for GQA share-buffer/static-cache exports. The current metadata schema supports one uniform window only; §40 hybrid per-layer patterns and attention sinks require future schema/model contracts.

---

### 2026-07-12: Mobius emits native onnx-genai inference metadata
**By:** Deckard
**What:** Mobius branch `feat/onnx-genai-metadata-export` adds `mobius build --runtime onnx-genai`, which writes `inference_metadata.yaml` from `ModelPackage.config`. The emitter maps attention type, query/KV head counts, head dimension, optional sliding window, maximum sequence length, runtime-configurable KV dtypes, native KV dtype, and required GQA/KV-dtype capabilities. The Qwen2.5-0.5B WebGPU package now declares 14 attention heads, 2 KV heads, head dimension 64, maximum sequence length 32768, fp16 KV, and the `group-query-attention`/`fp16-kv` capabilities.
**Why:** onnx-genai should consume its own `InferenceMetadata` contract instead of depending on ORT-GenAI's `genai_config.json`. Keeping the exporter in a dedicated Mobius integration makes the runtime contract explicit and testable at model-build time.

---

### 2026-07-12: Feature-gated CUDA EP, CUDA graphs, and device-resident GQA KV
**By:** Leon
**What:** `onnx-genai-ort` now has a default-off `cuda` Cargo feature. `ONNX_GENAI_EP=cuda` selects `ExecutionProvider::Cuda` and `ONNX_GENAI_CUDA_DEVICE` selects the non-negative CUDA device id (default 0). With the feature enabled, session creation uses ORT's V2 CUDA provider API (`CreateCUDAProviderOptions`, `UpdateCUDAProviderOptions`, `SessionOptionsAppendExecutionProvider_CUDA_V2`) with `device_id`; `ONNX_GENAI_CUDA_GRAPH=1` adds `enable_cuda_graph=1`. Graph-capture state is EP-generic and still maps `ONNX_GENAI_WEBGPU_GRAPH_CAPTURE` to WebGPU's existing session config. Without the feature, requesting CUDA returns `CUDA support not compiled in; rebuild with --features cuda`.

`ONNX_GENAI_DEVICE_KV=1` now has a separate CUDA path: `MemoryInfo::cuda(device_id)` (`"Cuda"`, device allocator, default memory) is resolved through the session's EP allocator and the existing `DecodeSession` shared-buffer path allocates max-length fp16 GQA KV with `Value::empty_in`. WebGPU behavior is unchanged and remains opt-in/experimental because ORT 1.27 WebGPU external in-place share-buffer tensors can SIGSEGV. CUDA graph/device-KV correctness and performance require H200 validation.

The CUDA feature is intentionally declared only on `onnx-genai-ort` in this change because another agent owns engine/server files. Until the coordinator adds a server-level `cuda = ["onnx-genai-ort/cuda"]` forwarding alias, use Cargo's package-qualified dependency feature:

```bash
export ORT_ROOT=/opt/onnxruntime-gpu-1.27.0
export ONNX_GENAI_EP=cuda
export ONNX_GENAI_CUDA_DEVICE=0
export ONNX_GENAI_CUDA_GRAPH=1
export ONNX_GENAI_DEVICE_KV=1

cargo run --release -p onnx-genai-server \
  --features onnx-genai-ort/cuda -- \
  --model models/qwen2.5-0.5b-cuda \
  --model-id qwen2.5-0.5b-cuda \
  --addr 127.0.0.1:8080
```

After the forwarding alias is added, the requested shorthand is:

```bash
cargo run --release --features cuda -p onnx-genai-server -- \
  --model models/qwen2.5-0.5b-cuda \
  --model-id qwen2.5-0.5b-cuda \
  --addr 127.0.0.1:8080
```

With LM Studio serving the comparison model at `127.0.0.1:1234`, benchmark from a second shell:

```bash
export LM_STUDIO_MODEL_ID=qwen2.5-0.5b-instruct-q4_0
cargo run --release -p onnx-genai-bench --bin compare -- \
  --runs 5 --warmups 1 --max-tokens 128 \
  --tokenizer models/qwen2.5-0.5b-cuda/tokenizer.json \
  --output docs/benchmarks/h200-cuda-vs-lm-studio.md \
  --runtime "onnx-genai CUDA|http://127.0.0.1:8080/v1|qwen2.5-0.5b-cuda|ONNX Q4_0; fp16 GQA KV|H200 CUDA EP; enable_cuda_graph=1; device KV=1" \
  --runtime "LM Studio|http://127.0.0.1:1234/v1|${LM_STUDIO_MODEL_ID}|GGUF Q4_0|H200 CUDA; record GPU offload/context/speculation settings"
```

**Why:** The CUDA graph is byte-identical to the validated WebGPU graph and ORT CUDA provides mature GQA/quantized kernels, IoBinding, external device allocators, and CUDA graph capture. Reusing the runtime-owned share-buffer abstraction removes per-token host/device KV transfers without changing the model or engine contract, while compile-time gating preserves CPU/WebGPU-only Mac builds.

---

### 2026-07-12T20:12:00-07:00: Device-resident GQA KV + persistent IoBinding + graph-capture plumbing (device-KV blocked by ORT 1.27 WebGPU SIGSEGV)

**Author:** leon (Engine Dev — KV & runtime buffers)
**Follows:** `batty-inference-metadata-gqa` (runtime-owned GQA share-buffer KV) and the perf lever it identified — the shared GQA KV buffer was CPU-allocated (`Value::empty` → default CPU allocator), so in SharedBuffer mode ORT round-trips KV host↔device every decode step, capping WebGPU decode far below Metal.

**Goal:** Make the runtime-owned max-length GQA KV buffers device-resident (WebGPU allocator) and bound as a persistent, in-place `past_key_values.*`/`present.*` share-buffer IoBinding, so KV never leaves the device; enable ORT WebGPU provider options (`enableGraphCapture`, `validationMode=disabled`) that benefit from stable device buffers.

---

#### What was implemented (all in `crates/onnx-genai-ort/src/`)

- **allocator.rs**
  - `MemoryInfo::webgpu()` — builds the WebGPU EP `MemoryInfo` via `CreateMemoryInfo_V2` (legacy `CreateMemoryInfo` rejects the name with *"Specified device is not supported. Try CreateMemoryInfo_V2."*). Params: name `"WebGPU_Buffer"`, device type `GPU` (1), vendor id 0, device id 0, mem type `DEFAULT` (0), alignment 0, allocator type `OrtDeviceAllocator` (0).
  - `Allocator::for_session_device(session_ptr, MemoryInfo)` — wraps the session's EP allocator via `CreateAllocator` (`owned: true`).
- **value.rs**
  - `Value::empty` refactored to delegate to new `Value::empty_in(shape, dtype, &Allocator)` (`CreateTensorAsOrtValue` with a caller-supplied allocator). CPU default preserved; device tensor safely outlives the local `Allocator` wrapper (ORT tensor retains the underlying AllocatorPtr shared_ptr).
- **session.rs**
  - `SessionOptions` gained `webgpu_graph_capture` and `webgpu_disable_validation`; `apply_provider_defaults()`/`selects_webgpu()` set them for WebGPU.
  - `Session` now tracks effective `execution_providers` (accounts for CPU fallback); `is_webgpu()`, `device_kv_allocator() -> Result<Option<Allocator>>` (None for non-webgpu, None when `ONNX_GENAI_DEVICE_KV` not opted-in, None+warn on allocator-create failure).
  - `apply_webgpu_provider_options()` + `add_session_config_entry()` write `ep.webgpuexecutionprovider.enableGraphCapture` and `.validationMode` via `AddSessionConfigEntry` after EPs are appended.
  - Env gates: `device_kv_enabled_from_env()` (**DEFAULT FALSE — opt-in**), `webgpu_disable_validation_from_env()` (**DEFAULT TRUE**), `webgpu_graph_capture_from_env()` (**DEFAULT FALSE**).
- **decode.rs (ort)**
  - `allocate_shared_buffers` uses `session.device_kv_allocator()`; allocates KV via `Value::empty_in(device_allocator)` when present, else `Allocator::default_cpu()`.

`crates/onnx-genai-engine/src/{decode.rs, kv_bridge.rs}` were **not** modified — device residence is driven automatically by the session EP inside the ort `DecodeSession`, so no engine change was warranted (avoided unnecessary edits/regressions).

---

#### The ORT limitation that blocks the perf lever

Binding a user-pre-allocated `WebGPU_Buffer` device tensor (created via `CreateTensorAsOrtValue` on a `CreateAllocator`-derived WebGPU allocator) as a **persistent in-place `past_key_values.*`/`present.*` share-buffer** via IoBinding **segfaults** during multi-step decode on ORT **1.27.0** (ORT_API_VERSION 27) WebGPU EP:

```
EXC_BAD_ACCESS (code=1, address=0x0)
frame #0: 0x0000000000000000   (call through a null function pointer)
thread: onnx-genai-batch-driver  (multi-step decode)
```

- Short generations (8–16 tokens) sometimes survive; longer (≈120 tokens) reliably crash.
- Independent of `validationMode` (crashes with `disabled`, `basic`, `full`).
- GQA **requires** the SharedBuffer path (in-place `past==present`, max-length buffer) per the export contract — which is exactly the externally-pre-allocated in-place device tensor case that ORT does not support here.

Because of this, **device-resident KV is gated OFF by default** and is opt-in via `ONNX_GENAI_DEVICE_KV=1` (experimental). The shipped default keeps CPU-allocated KV buffers (coherent + stable).

`validationMode=disabled` **is** safe and ships **ON** by default for WebGPU (measured within noise, no regression). Graph capture is wired but **OFF** by default: (a) it only benefits fully device-resident I/O (blocked above), and (b) our decode binds a growing `attention_mask` (`[1, past+new]`) each step, so a captured graph cannot replay stable addresses. Verified `ONNX_GENAI_WEBGPU_GRAPH_CAPTURE=1` with CPU KV is harmless but ineffective (coherent + stable across 3×120-token runs; capture does not engage without on-device I/O).

---

#### Numbers (WebGPU EP, `models/qwen2.5-0.5b-q4-gqa-webgpu`, decode via differencing max_tokens 8 vs 136, `/usr/bin/time -p`)

| Config | Coherence | Decode tok/s | Notes |
|---|---|---|---|
| Task-stated baseline | — | **30.5** | prior measurement |
| Existing release binary (pre-change) | "The capital of France is Paris." | ~47 (median) | our machine baseline |
| Final default (CPU KV + `validationMode=disabled`) | "Paris" + stable 5×120-tok | ~49.6 (median) | within noise, **no regression** |
| `ONNX_GENAI_DEVICE_KV=1` (device KV) | crashes on long gen | — | **SIGSEGV** (see above) |

The intended lever (eliminating host↔device KV copies via device-resident KV) is **blocked** by the ORT limitation; decode did not meaningfully improve. The device-resident plumbing is complete and correct behind the opt-in flag; the correct shipped default is coherent, stable, and non-regressing.

---

#### Recommended follow-ups
1. **ORT-sanctioned device output path:** instead of creating an external device tensor, use `BindOutputToDevice(present.*, WebGPU_Buffer mem_info)` to let ORT allocate `present` on device, then rebind it as next-step `past` (ZeroCopy rebind-to-device). Avoids external-tensor creation but changes KV semantics for a share-buffer GQA model — validate coherence carefully.
2. Track ORT WebGPU EP support for externally-allocated in-place share-buffer device tensors; re-enable `ONNX_GENAI_DEVICE_KV` default once fixed.
3. If/when device-resident I/O works, revisit graph capture with a fixed-capacity (padded) `attention_mask` to satisfy stable-address replay.

**Env flags introduced:** `ONNX_GENAI_DEVICE_KV` (default off), `ONNX_GENAI_WEBGPU_VALIDATION` (validation; disabled by default for WebGPU), `ONNX_GENAI_WEBGPU_GRAPH_CAPTURE` (default off).

**Validation:** `cargo clippy -p onnx-genai-engine -p onnx-genai-ort --all-targets -- -D warnings` → exit 0 (clean). `cargo test -p onnx-genai-engine -p onnx-genai-ort` → 0 failed.

---

### 2026-07-12: CUDA-targeted stacked Qwen model is structurally valid
**By:** Sapper
**What:** Mobius branch `int/cuda-stacked` at `380acf2` combines Q4_K_M conversion and quantized embeddings, plus a shape/type stamp for `GatherBlockQuantized` so exported models pass ONNX checker. The H200-ready package is `models/qwen2.5-0.5b-cuda/`, built from Qwen2.5-0.5B-Instruct Q4_0 because no local Q4_K_M GGUF was available. It has 24 `GroupQueryAttention`, 168 `MatMulNBits`, one `GatherBlockQuantized`, zero `Attention`, fp16 KV I/O, and a 73.03 MiB packed embedding payload. `onnx.checker` and strict shape inference pass. Mobius metadata emission produced `inference_metadata.yaml` with `grouped_query_attention`, fp16 KV, and max length 4096. `build-gguf` does not yet accept `--runtime onnx-genai`, so the sidecar was emitted separately with the existing `feat/onnx-genai-metadata-export` emitter. CPU fallback generated “The capital of France is Paris.”
**Why:** For this Qwen graph, `--ep cuda` and `--ep webgpu` produce byte-identical ONNX and external-data files: both EP capability sets allow fp16 GQA and packed QKV, while Qwen does not trigger WebGPU-only graph-capture rewrites. ORT source registers CUDA kernels for `GroupQueryAttention`, `MatMulNBits`, and `GatherBlockQuantized`; Mobius does not disallow either quantized op. Runtime CUDA follow-up still needs actual `ONNX_GENAI_EP=cuda` parsing/EP attachment, CUDA `enable_cuda_graph` provider configuration, CUDA device-resident shared-KV allocation, and Cargo/build feature gating so default Mac builds retain CPU/WebGPU ORT. Current `session.rs` only defines `ExecutionProvider::Cuda`; the environment parser and append path still fall back to CPU, and `device_kv_allocator` is WebGPU-only. Validation: `lintrunner --revision origin/main` exit 0; GGUF pytest 156 passed, exit 0.

---

### 2026-07-12: Emit loadable onnx-genai metadata with bounded KV capacity
**By:** Sapper
**What:** Mobius `--runtime onnx-genai` now emits only runtime-supported attention capabilities: `grouped_query_attention` for GQA and `multi_head_attention` for MHA. KV dtype remains represented by `kv_cache.native_dtype` and `model.runtime_configurable.kv_cache.dtype`, not an unsupported capability. `model.max_sequence_length` is treated as serving KV capacity, defaults to `min(4096, max_position_embeddings)`, and can be overridden with `--max-length N` up to the model limit.
**Why:** onnx-genai rejects unknown required capabilities, and using Qwen2.5's theoretical 32768-token context would pre-size roughly 800 MiB of fp16 KV, exceeding WebGPU's 256 MiB buffer limit. The regenerated 4096-token metadata loaded without `genai_config.json` and coherently answered that the capital of France is Paris.

---

### 2026-07-12: Normalize mixed Q4_K_M GGUF projections to MatMulNBits int4
**By:** Sapper
**What:** Mobius now detects Q4_K-containing mixed presets as a 4-bit, block-32 asymmetric MatMulNBits target. Q4_K super-blocks are reference-dequantized across flattened tensor row boundaries and affine-requantized per 32 values; Q5_0, Q6_K, and Q8_0 projection tensors in Q4_K_M are likewise normalized to the same layout. The official `Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf` converted to 168 MatMulNBits nodes and generated “The capital of France is Paris.”
**Why:** Q4_K fractional offsets cannot be represented faithfully by simply rounding/clamping uint4 zero-points, and Q4_K_M mixes multiple GGUF quantization types while one ONNX graph requires a single packed initializer shape. Dequantize-then-requantize gives a bounded half-scale error and preserves the fast packed-uint8 MatMulNBits path.

---

### 2026-07-12: Preserve GGUF token embeddings with GatherBlockQuantized
**By:** Sapper
**What:** Mobius `build-gguf --keep-quantized` now detects a repackable GGUF `token_embd.weight`, enables `QuantizedEmbedding`, reshapes the existing MatMulNBits packed bytes to the 2-D GatherBlockQuantized layout, and emits `com.microsoft::GatherBlockQuantized` with `bits=4`, `block_size=32`, `gather_axis=0`, and `quantize_axis=1`. Tied embedding/LM-head models share the packed table and use `MatMulNBits` for the head. For Qwen2.5-0.5B, the embedding changed from one 272.27 MB fp16 initializer to a 68.07 MB uint8 initializer plus 8.51 MB fp16 scales. ORT 1.27 verbose placement assigned GatherBlockQuantized to WebGpuExecutionProvider, not CPU.
**Why:** Gathering and dequantizing only selected token rows avoids materializing the 259.66 MiB fp16 embedding table and keeps its largest initializer below WebGPU's 256 MiB buffer limit. CPU inference produced “The capital of France is Paris.” WebGPU inference also produced coherent “The capital of France is Paris,” but the current release CLI then exited with SIGSEGV during/after generation; the fp16 GQA package is additionally blocked by the stale release runtime's pre-existing fp16-KV validation before session creation. A direct ORT 1.27 WebGPU session for that fp16 package created successfully and placed GatherBlockQuantized on WebGPU, so the operator itself is supported on-device rather than falling back.

---

### 2026-07-12: Treat the GQA/fixed-Q4 report as the current performance checkpoint
**By:** Sebastian
**What:** Use `docs/benchmarks/2026-07-12-JustindeMacBook-Pro-gqa-fixed.md` as the current cross-runtime comparison. The valid medians are WebGPU GQA 19.40/19.07 tok/s and CPU fixed-Q4 40.17/31.53 tok/s for short/long prompts; the old Q4 numbers remain invalid-output history only.
**Why:** Both onnx-genai models now pass the coherence gate. The report adds the missing LM Studio CPU comparison and shows the next priorities are device-resident WebGPU KV plus graph capture, CPU prefill, and quantized embedding/output-head kernels.

---

### 2026-07-12: Q4+GQA WebGPU is valid but KV residency remains dominant
**By:** Sebastian
**What:** The same-source Q4_0 Mobius WebGPU build contains 168 MatMulNBits, 24 GroupQueryAttention, and zero Attention nodes; ORT places all quantized projections and GQA on WebGPU with 1 H2D/0 D2H graph copies. It is coherent and measures 30.52/29.21 tok/s versus LM Studio Metal at 201.60/221.82 tok/s.
**Why:** Quantization improves the prior fp16-GQA WebGPU decode by 57.3%/53.2%, but WebGPU remains 6.61x/7.59x behind Metal and still trails Q4 CPU. Prioritize a device-resident WebGPU KV allocator, persistent IoBinding, and graph capture before further graph-level tuning.

---

### 2026-07-12: Expose embeddings endpoint with an explicit engine capability gap
**By:** Zhora
**What:** Add the OpenAI `/v1/embeddings` request/response contract, input validation, float/base64 vector serialization, and route wiring. Until the engine exposes single-pass hidden-state or pooled-output inference, valid requests return HTTP 501 instead of fabricated vectors.
**Why:** The server can faithfully establish and test the public API now, but pooling, normalization, dimensions, batching, and usage accounting require real model outputs and must remain an engine-owned capability.

---

### 2026-07-13: Sliding Window Attention — attention-sink (StreamingLLM) support + documented ORT boundary
**By:** Leon
**What:** Extended SWA (DESIGN §40) with attention-sink token retention. Metadata gains `model.attention.sink_tokens: Option<usize>` (§40.9). The paged KV cache gains `PagedKvCache::apply_sliding_window_with_sinks(seq, window, sink_tokens)` — pinning the leading sink pages and evicting only the middle window pages (sink pinning is page-granular; `sink_tokens==0` delegates to the existing contiguous `apply_sliding_window`). The engine threads `sink_tokens` from metadata through `detect_model_decode_path` → `ModelDecodePath::PastPresent` → `DecodeState`, and `apply_window_after_step`/`rewind_windowed` keep `[0, sink) ∪ [window_start, len)` token-exactly in the runtime KV buffer that feeds ORT.
**Why:** Contiguous single-window SWA was already implemented end-to-end; the real §40 gap was attention sinks (§40.4), which are correctness-critical — dropping the first tokens under a naive window corrupts the attention distribution. The runtime KV buffer (exact) and the paged cache (page-granular bookkeeping for rewind/prefix) are decoupled, so buffer sinks can be token-exact while paged sinks stay page-aligned without conflict.
**Boundary deferred to Mobius/ORT:** (1) hybrid per-layer attention patterns (§40.3) need per-layer KV buffers and per-layer graph masks — not expressible with a single shared decode buffer today; (2) feeding **discontinuous** `position_ids` (§40.8) into a contiguous ORT graph after window/sink eviction requires model/EP support (rotating cache or `local_window_size` contract). `detect_model_decode_path` already refuses SWA + static-cache and SWA + shared-buffer combos, and `load_materialized_past` refuses windowed/sink materialize into a contiguous graph, so the runtime never silently produces wrong outputs — it declines the unsupported path instead.

---

### 2026-07-13: Add server debug introspection and queue-depth admission control
**By:** Zhora
**What:** Added `/v1/debug/config`, `/v1/debug/sessions`, `/v1/debug/kv`, and `/v1/debug/trace`; renamed the server's active-plus-queued generation cap to `max_queue_depth` (`--max-queue-depth` / `ONNX_GENAI_MAX_QUEUE_DEPTH`).
**Why:** The debug endpoints expose current server configuration, sessions, existing cache/admission metrics, and tracing status without creating new engine subsystems. The explicit queue-depth setting documents and configures the existing semaphore-based HTTP 429 admission boundary.
