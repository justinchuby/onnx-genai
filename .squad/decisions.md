# Decisions

Canonical, append-only record of accepted team decisions. Only the Coordinator (via Scribe merge) writes here. Agents drop proposals in `decisions/inbox/`.

---

### 2026-07-12T08:56:27-07:00: Expand Rust and Python gitignore coverage
**By:** Deckard
**What:** Added standard Rust backup/debug-symbol patterns and comprehensive Python cache, build, packaging, virtual environment, coverage, test, type-checking, lint, notebook checkpoint, and native extension ignore patterns to `.gitignore`.
**Why:** The repository is a Rust Cargo workspace that may include Python tooling, scripts, model conversion, or tests, so ignoring common generated artifacts keeps source control focused on intentional source files while preserving existing `/target` and `Cargo.lock` behavior.

---

# Generation API surface for Phase 1 greedy generation

**Date:** 2026-07-12T09:00:00-07:00
**By:** Batty

## Decision

`onnx-genai-engine` now exposes the Phase 1 generation API from `crates/onnx-genai-engine/src/engine.rs` and re-exports it from `lib.rs`:

- `GeneratePrompt::{Text(String), TokenIds(Vec<TokenId>)}`
- `GenerateOptions { max_new_tokens, temperature, top_p, top_k, repetition_penalty, greedy, stop_sequences, eos_token_id, stop_on_eos }`
- `GenerateRequest { prompt, options }` plus `GenerateRequest::new(...)`
- `GenerateResult { text, token_ids, finish_reason }`
- `GenerateToken { token_id, text, finish_reason }`
- `GenerateTokenCallback<'a> = dyn FnMut(GenerateToken) -> anyhow::Result<()> + Send + 'a`
- `FinishReason::{MaxTokens, EosToken, StopSequence { index }}`
- `StopSequence::{Text(String), Tokens(Vec<TokenId>)}` and `TokenId = u32`
- `Engine::generate(&mut self, request: GenerateRequest) -> anyhow::Result<GenerateResult>`
- `Engine::generate_with_callback(&mut self, request: GenerateRequest, callback: Option<&mut GenerateTokenCallback<'_>>) -> anyhow::Result<GenerateResult>`

`Engine::generate` creates a scheduler sequence, validates options, builds the processor chain, applies processors to logits, selects greedy vs sampled decoding, checks EOS/stop termination, advances/completes the scheduler, and returns `GenerateResult`.

## Processor-chain order

The Phase 1 chain order is:

1. `RepetitionPenaltyProcessor`
2. `StopSequenceProcessor` (termination signal only; no logit mutation)
3. `TemperatureProcessor`
4. `TopKProcessor`
5. `TopPProcessor`

This follows DESIGN.md §3.6, with stop sequence checks placed in the constraints slot before sampling filters.

## Intentional stubs for next batch

Only integration points waiting on Deckard's ORT/tokenizer/model-loader API remain `todo!()`:

- `Engine::tokenize_prompt` for `GeneratePrompt::Text`
- `Engine::detokenize_token`
- `Engine::next_token_logits`

`GeneratePrompt::TokenIds` already bypasses tokenization, so next batch should fill the ORT forward and detokenization calls without changing the public API.

---

### 2026-07-12T09:00:00-07:00: ORT Phase 1 session, loader, and tokenizer API
**By:** Deckard
**What:** `onnx-genai-ort` now exposes a CPU ORT C API wrapper contract for Batty's generation-loop wiring:
- `Environment::new(name: &str) -> Result<Environment>` creates the ORT environment handle.
- `Session::new(env: &Environment, path: &Path, options: SessionOptions) -> Result<Session>` loads an ONNX model. Phase 1 accepts only `ExecutionProvider::Cpu`; non-CPU providers return `OrtError::InvalidArgument` until EP append paths are wired.
- `Session::run(&self, inputs: &[(&str, &Value)]) -> Result<Vec<Value>>` runs a forward pass. Input names must match the model graph names. Output `Vec<Value>` is ordered exactly as `Session::output_names()` / `Session::outputs()`.
- `Session::input_names() -> &[String]`, `Session::output_names() -> &[String]`, `Session::inputs() -> &[TensorInfo]`, and `Session::outputs() -> &[TensorInfo]` expose graph I/O metadata. `TensorInfo { name, dtype, shape }` uses negative dimensions for ORT dynamic axes.
- `Value::from_slice_i64(data: &[i64], shape: &[i64]) -> Result<Value>` and `Value::from_slice_f32(data: &[f32], shape: &[i64]) -> Result<Value>` create CPU tensors with owned backing storage. `Value::from_vec_i64` / `from_vec_f32` avoid a second caller-side allocation when the caller can transfer a Vec. `Value::to_vec_i64()` / `to_vec_f32()` copy ORT output tensor data back to Rust. `Value::shape()`, `Value::dtype()`, and `Value::numel()` describe tensors.
- `IoBinding::{new, bind_input, bind_output, bind_output_to_device, clear}` and `Session::run_with_binding(&self, &IoBinding) -> Result<()>` are real C API calls, but Phase 1 generation should prefer `Session::run` unless pre-bound KV buffers are needed.
- `ModelDirectory::load(root) -> Result<ModelDirectory>` resolves `decoder.onnx` (or exactly one `.onnx` fallback), required `tokenizer.json`, and optional `inference_metadata.{yaml,yml,json}`. Missing metadata is represented as `metadata_path: None` and is tolerated for Phase 1.
- `Tokenizer::from_file(path) -> Result<Tokenizer>`, `encode(&self, prompt: &str) -> Result<Vec<u32>>`, `encode_i64(&self, prompt: &str) -> Result<Vec<i64>>`, `decode(&self, ids: &[u32]) -> Result<String>`, `decode_i64(&self, ids: &[i64]) -> Result<String>`, `token_id(&self, token: &str) -> Option<u32>`, and `eos_token_id(&self) -> Option<u32>` wrap the HF `tokenizers` crate. `decode*` skips special tokens.
**Why:** Batty needs a stable Phase 1 backend boundary for greedy generation: resolve model/tokenizer files, encode prompts to i64 `input_ids`, build i64/f32 CPU input tensors by graph name, run ORT, read logits from ordered outputs, and decode generated token ids. The API intentionally copies CPU tensors for safety now; IoBinding remains available for later KV/device-buffer work without changing the session contract.

---

### 2026-07-12T09:00:00-07:00: Add metadata tests and tiny Mobius LLM fixture
**By:** Pris
**What:** Added fixture-based tests for `onnx-genai-metadata` covering valid YAML/JSON parsing, malformed/schema-invalid parse errors, and runtime capability validation. Produced a tiny GPT-2-style decoder-only ONNX fixture at `tests/fixtures/tiny-llm/` with deterministic random weights and a matching WordLevel `tokenizer.json`.
**Why:** Phase 1 requires deterministic metadata parser coverage and a committed tiny end-to-end generation fixture. The model was generated without downloading HuggingFace weights using Mobius from `/Users/justinc/Documents/GitHub/mobius` with: `PYTHONPATH=/Users/justinc/Documents/GitHub/mobius/src python tests/fixtures/tiny-llm/generate_tiny_llm.py`. Generated sizes: `model.onnx` 15,807 bytes, `model.onnx.data` 13,312 bytes, `tokenizer.json` 2,038 bytes.

---

### 2026-07-12T08:58:32-07:00: Phase 1 foundation status and next plan
**By:** Roy
**What:** Assessed `docs/DESIGN.md` Phase 1 against current crate sources. Workspace scaffolding exists; metadata, KV, ORT, scheduler, engine, logit processors, server, and facade crates are present, but end-to-end generation is blocked by stub ORT execution, no tokenizer wiring, and no engine generation API/loop. The next increment should prioritize making `onnx-genai-ort` actually load and run a CPU ONNX session, then wire tokenizer/model-dir discovery and a minimal greedy single-sequence engine path.
**Why:** DESIGN.md Phase 1 exit criteria is loading a Phi-4 ONNX model and generating greedily end-to-end. No other Phase 1 component can be validated against a real model until the ORT wrapper returns real output tensors; this is the highest-leverage unblocker for Deckard, Batty, Rachael, and Pris to parallelize follow-on work.
