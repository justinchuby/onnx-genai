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
