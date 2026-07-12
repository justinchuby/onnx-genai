# onnx-genai Design Document

**Project:** onnx-genai — A Rust inference runtime for generative AI models
**Author:** Justin Chu's agent
**Date:** 2026-07-12
**Status:** Design

---

## 1. Vision

A Rust-native generative AI runtime built on ONNX Runtime, implementing the inference metadata standard proposed in [onnx/onnx#8184](https://github.com/onnx/onnx/issues/8184). This project serves as both:

1. **Reference implementation** of the ONNX inference metadata spec (proving the standard is implementable and useful)
2. **Production-quality alternative** to onnxruntime-genai (C++) — with memory safety, modern concurrency, and clean architecture

### Goals

- Agent-first: multi-turn, long-context, concurrent inference as the primary workload
- Safe: Rust ownership model guarantees KV cache lifecycle, no use-after-free, no data races in the scheduler
- Modular: each component (KV cache, scheduler, pipeline, speculative) is independently usable and testable
- Standard-driven: behavior derived from inference metadata declarations, not hardcoded model-type dispatch
- ORT as execution backend: delegate all NN computation to ONNX Runtime; this project manages everything above the session level

### Non-Goals

- Writing custom CUDA/Metal kernels (ORT handles this via Execution Providers)
- Replacing ORT's graph optimization or operator implementation
- Supporting non-ONNX model formats (use Mobius to generate ONNX models)
- Building a training framework

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Public API Layer                    │
│  ┌───────────────────┐  ┌────────────────────────┐  │
│  │ OpenAI-compatible │  │ Rust native API        │  │
│  │ HTTP server       │  │ (library crate)        │  │
│  └───────────────────┘  └────────────────────────┘  │
├─────────────────────────────────────────────────────┤
│                 Generation Engine                     │
│  ┌──────────┐ ┌────────────┐ ┌───────────────────┐ │
│  │Scheduler │ │ Speculative│ │ Logit Processors  │ │
│  │          │ │ Decoding   │ │ (chain)           │ │
│  └──────────┘ └────────────┘ └───────────────────┘ │
├─────────────────────────────────────────────────────┤
│                 Memory Management                     │
│  ┌──────────────────┐ ┌──────────────────────────┐  │
│  │ KV Cache Manager │ │ Prefix Cache (trie)      │  │
│  │ (paged, tiered)  │ │                          │  │
│  └──────────────────┘ └──────────────────────────┘  │
├─────────────────────────────────────────────────────┤
│                 Model Management                      │
│  ┌──────────────────┐ ┌──────────────────────────┐  │
│  │ Inference Meta   │ │ Pipeline Orchestrator    │  │
│  │ Parser/Validator │ │ (multi-model)            │  │
│  └──────────────────┘ └──────────────────────────┘  │
├─────────────────────────────────────────────────────┤
│                 Backend Layer                         │
│  ┌──────────────────┐ ┌──────────────────────────┐  │
│  │ ORT Session Mgr  │ │ HF Tokenizers           │  │
│  │ (ort crate)      │ │ (tokenizers crate)      │  │
│  └──────────────────┘ └──────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 3. Core Components

### 3.1 Inference Metadata Parser

**Responsibility:** Parse, validate, and provide access to the model's inference metadata (the spec from onnx/onnx#8184).

**Input:** `inference_metadata.yaml` (or JSON) shipped alongside the model's `.onnx` files and `tokenizer.json`.

**Key behaviors:**
- Parse all spec sections (capabilities, kv_cache, quantization, pipeline, strategy, structured_output, hardware_profile)
- Validate `required_capabilities` against runtime's supported capability set → fast-fail with clear error
- Handle unknown fields gracefully (ignore, per spec's forward-compatibility rule)
- Resolve `fallback_behavior` for unknown enum values
- Provide typed Rust structs for each section

```rust
pub struct InferenceMetadata {
    pub model: ModelCapabilities,
    pub kv_cache: Option<KvCacheSpec>,
    pub quantization: Option<QuantizationIntent>,
    pub pipeline: Option<PipelineSpec>,
    pub strategy: Option<StrategySpec>,
    pub structured_output: Option<StructuredOutputSpec>,
    pub hardware_requirements: Option<HardwareRequirements>,
    pub required_capabilities: Vec<String>,
}

impl InferenceMetadata {
    pub fn load(path: &Path) -> Result<Self, MetadataError>;
    pub fn validate_against(&self, runtime_caps: &RuntimeCapabilities) -> Result<(), Vec<UnsupportedCapability>>;
}
```

### 3.2 KV Cache Manager

**Responsibility:** Manage KV cache memory with paged allocation, tiered storage, and Copy-on-Write semantics.

**This is the most critical component for agent workloads.**

#### 3.2.1 Page Table

```rust
/// A page holds KV tensors for a fixed number of tokens (e.g., 16 tokens per page)
pub struct Page {
    id: PageId,
    key: Tensor,          // [num_heads, page_size, head_dim]
    value: Tensor,        // [num_heads, page_size, head_dim]
    ref_count: AtomicU32, // for CoW sharing
    device: Device,       // GPU | CPU | Disk
}

pub struct PageTable {
    /// Logical sequence → ordered list of physical pages
    sequences: HashMap<SequenceId, Vec<PageId>>,
    /// Physical page pool
    pages: Slab<Page>,
    /// Free page list per device tier
    free_lists: HashMap<Device, VecDeque<PageId>>,
    /// Page size in tokens
    page_size: usize,
}
```

#### 3.2.2 Operations (from spec §4c)

```rust
pub trait KvCacheOps {
    /// Truncate cache to position. O(pages_removed), not O(sequence_length).
    fn rewind_to(&mut self, seq: SequenceId, position: usize) -> Result<()>;

    /// Fork a sequence with CoW. Shared pages get ref_count++, no copy.
    fn fork(&mut self, source: SequenceId, position: usize) -> Result<SequenceId>;

    /// Save cache state for later restore.
    fn checkpoint(&self, seq: SequenceId) -> Result<CacheCheckpoint>;
    fn restore(&mut self, seq: SequenceId, checkpoint: CacheCheckpoint) -> Result<()>;

    /// Append new KV entries (after a forward pass).
    fn append(&mut self, seq: SequenceId, key: Tensor, value: Tensor) -> Result<()>;

    /// Evict pages to a lower tier (GPU→CPU→Disk) based on policy.
    fn evict(&mut self, policy: EvictionPolicy, target_free_pages: usize) -> Result<usize>;

    /// Prefetch pages from lower tier back to GPU.
    fn prefetch(&mut self, seq: SequenceId, range: Range<usize>) -> Result<()>;
}
```

#### 3.2.3 Tiered Storage

```
GPU HBM (hot)  ←→  CPU RAM (warm)  ←→  SSD (cold)
     ↑                    ↑                  ↑
  active gen         paused session      suspended session
```

Eviction policy options (configured per-model via metadata):
- **LRU** — least recently accessed page gets evicted
- **Priority** — lower-priority sequences evict first
- **Layer-aware** — metadata's `sensitive_layers` keeps those on GPU

#### 3.2.4 Quantized KV Cache

Per metadata spec, support runtime KV quantization:
```rust
pub struct KvQuantConfig {
    pub key_dtype: DataType,      // from metadata.kv_cache.quantization_tolerance.key
    pub value_dtype: DataType,
    pub sensitive_layers: Vec<usize>,  // keep these at native precision
}
```

Quantize on write (when appending to cache), dequantize on read (when feeding to attention). Sensitive layers bypass quantization.

### 3.3 Prefix Cache

**Responsibility:** Detect and share common prefixes across sequences to avoid redundant computation.

**Data structure:** Radix tree (trie) keyed by token sequences.

```rust
pub struct PrefixCache {
    root: TrieNode,
}

struct TrieNode {
    /// Token at this node
    token: Option<TokenId>,
    /// Children keyed by next token
    children: HashMap<TokenId, Box<TrieNode>>,
    /// Cached KV pages for the prefix ending here
    kv_pages: Option<Vec<PageId>>,
    /// How many active sequences share this prefix
    ref_count: usize,
}

impl PrefixCache {
    /// Find longest cached prefix for a token sequence.
    /// Returns (prefix_length, page_ids) — caller can skip prefill for these tokens.
    pub fn lookup(&self, tokens: &[TokenId]) -> (usize, Vec<PageId>);

    /// Insert a computed prefix into the cache.
    pub fn insert(&mut self, tokens: &[TokenId], pages: Vec<PageId>);

    /// Evict least-used prefixes to free pages.
    pub fn evict_lru(&mut self, target_pages: usize) -> Vec<PageId>;
}
```

**Agent scenario:** All sessions sharing the same system prompt → first `lookup` returns the full system prompt's KV → skip that prefill entirely on subsequent requests.

### 3.4 Scheduler

**Responsibility:** Decide which sequences to run, when, and in what batch — managing GPU resources across concurrent requests.

#### 3.4.1 Request Lifecycle

```
Arriving → Queued → Prefilling → Generating → Completed
                         ↓              ↓
                    Preempted ←──── Preempted
                         ↓
                    Swapped (KV on CPU)
```

#### 3.4.2 Scheduling Policy

```rust
pub struct Scheduler {
    /// Requests waiting to be processed
    waiting: PriorityQueue<Request>,
    /// Currently running (in the batch)
    running: Vec<RunningSequence>,
    /// Preempted (KV cache swapped to CPU)
    swapped: Vec<SwappedSequence>,
    /// Configuration
    config: SchedulerConfig,
}

pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub max_total_tokens: usize,       // total KV budget across all sequences
    pub preemption_policy: PreemptionPolicy,  // recompute | swap
    pub priority_policy: PriorityPolicy,      // fcfs | priority | fair_share
}

impl Scheduler {
    /// Called each iteration: decide what to run next.
    /// Returns: sequences to prefill, sequences to decode, sequences to preempt.
    pub fn schedule(&mut self) -> ScheduleDecision;
}

pub struct ScheduleDecision {
    pub prefill: Vec<PrefillRequest>,   // new sequences entering the batch
    pub decode: Vec<DecodeRequest>,     // continuing generation
    pub preempt: Vec<SequenceId>,       // kick out to make room
    pub swap_in: Vec<SequenceId>,       // bring back from CPU
}
```

#### 3.4.3 Continuous Batching

Each scheduling iteration:
1. Completed sequences leave the batch (free their pages)
2. Preempted sequences swap out if memory pressure
3. New sequences enter if budget allows
4. All active sequences get one decode step together

This means different sequences can be at different positions — no padding waste.

### 3.5 Speculative Decoding Engine

**Responsibility:** Implement the parameterized speculative decoding loop based on metadata's `strategy` spec.

```rust
pub struct SpeculativeEngine {
    pub config: SpeculativeConfig,
}

pub struct SpeculativeConfig {
    pub producer: DraftProducer,
    pub acceptance: AcceptanceRule,
    pub tokens_per_step: usize,    // K
    pub topology: Topology,         // Linear | Tree
}

pub enum DraftProducer {
    DraftModel { session: OrtSession },
    SelfSpeculative { depth: usize },
    Ngram { min_match: usize, max_draft: usize, window: usize },
    ExtraHeads { head_name: String },
}

pub enum AcceptanceRule {
    Greedy,
    RejectionSampling,
    Typical { threshold: f32 },
}

impl SpeculativeEngine {
    /// Run one speculative step:
    /// 1. Draft K tokens using producer
    /// 2. Verify all K in one target forward pass
    /// 3. Accept/reject using acceptance rule
    /// 4. Rewind KV cache for rejected tokens
    /// Returns: accepted tokens (1..=K+1)
    pub fn step(
        &self,
        target_session: &OrtSession,
        kv_cache: &mut dyn KvCacheOps,
        sequence: &mut Sequence,
    ) -> Result<Vec<TokenId>>;
}
```

**KV rollback integration:** After rejection, `kv_cache.rewind_to(seq, accepted_position)` — this is why paged KV with O(1) rewind matters.

### 3.6 Logit Processor Chain

**Responsibility:** Apply ordered transformations to logits before sampling.

```rust
pub trait LogitProcessor: Send + Sync {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext);
    fn name(&self) -> &str;
}

pub struct ProcessorChain {
    processors: Vec<Box<dyn LogitProcessor>>,
}

// Built-in processors:
pub struct TemperatureProcessor { temperature: f32 }
pub struct TopPProcessor { top_p: f32 }
pub struct TopKProcessor { top_k: usize }
pub struct RepetitionPenaltyProcessor { penalty: f32 }
pub struct FrequencyPenaltyProcessor { penalty: f32 }
pub struct PresencePenaltyProcessor { penalty: f32 }
pub struct GrammarProcessor { grammar: CompiledGrammar }  // for structured output
pub struct StopSequenceProcessor { sequences: Vec<Vec<TokenId>> }
```

Order matters (configured via metadata or API):
1. Repetition/frequency/presence penalties
2. Grammar constraint (mask invalid tokens)
3. Temperature scaling
4. Top-K filtering
5. Top-P (nucleus) filtering

### 3.7 Pipeline Orchestrator

**Responsibility:** Execute multi-model pipelines as declared in metadata's `pipeline` spec.

```rust
pub struct Pipeline {
    pub models: HashMap<String, ModelHandle>,
    pub dataflow: Vec<DataflowEdge>,
    pub phases: HashMap<String, PhaseConfig>,
}

pub struct DataflowEdge {
    pub from_model: String,
    pub from_output: String,
    pub to_model: String,
    pub to_input: String,
    pub dtype: DataType,
    pub device_transfer: bool,
}

pub enum Phase {
    PromptOnly,  // run once on initial input
    Always,      // run every step
    FinalOnly,   // run only when generating final output
}

impl Pipeline {
    /// Execute the pipeline for one step, respecting phase gating and dataflow.
    pub fn step(&mut self, phase: CurrentPhase, inputs: &Inputs) -> Result<Outputs>;
}
```

**Example:** Vision-Language Model
1. Phase=PromptOnly: Run CLIP encoder → image_features
2. Phase=Always: Feed (text + image_features) to decoder → generate tokens
3. Dataflow edge: `clip.image_features` → `decoder.encoder_hidden_states`

### 3.8 ORT Session Manager

**Responsibility:** Manage ORT InferenceSession lifecycle, I/O binding, and device placement.

```rust
pub struct SessionManager {
    env: Arc<ort::Environment>,
    sessions: HashMap<String, OrtSession>,
}

pub struct OrtSession {
    session: ort::Session,
    io_binding: Option<ort::IoBinding>,
    device: Device,
    metadata: SessionMetadata,
}

impl SessionManager {
    /// Load a model from ONNX file with specified execution provider.
    pub fn load(&mut self, name: &str, path: &Path, device: Device) -> Result<()>;

    /// Run inference with pre-bound I/O (avoids host↔device copies).
    pub fn run(&self, name: &str, inputs: &IoBindings) -> Result<Outputs>;

    /// Unload a model to free resources.
    pub fn unload(&mut self, name: &str) -> Result<()>;
}
```

---

## 4. Data Flow: A Single Generation Step

```
1. Scheduler.schedule()
   → decides which sequences to decode this iteration

2. For each sequence in batch:
   a. KvCacheManager provides page table mapping
   b. Construct attention mask from page table
   c. Bind inputs to ORT session (input_ids, attention_mask, KV pages)

3. ORT session.run()
   → returns logits [batch_size, vocab_size]

4. LogitProcessorChain.process(logits)
   → apply penalties, grammar mask, temperature, top-p/k

5. Sample token(s) from processed logits

6. For each sequence:
   a. KvCacheManager.append(new KV from this step)
   b. Check stop conditions
   c. If done: notify scheduler, free pages
   d. If speculative: run verify loop, rewind rejected

7. Yield generated tokens to streaming response
```

---

## 5. Concurrency Model

```rust
// Main generation loop runs on a dedicated thread
// API requests come in via async (tokio)
// Communication via channels

pub struct Engine {
    /// Async → Engine: new requests, cancellations
    request_rx: mpsc::Receiver<EngineRequest>,
    /// Engine → Async: streaming tokens, completion
    response_tx: broadcast::Sender<EngineResponse>,
    /// Core components (owned by engine thread, no sharing)
    scheduler: Scheduler,
    kv_cache: KvCacheManager,
    prefix_cache: PrefixCache,
    session_mgr: SessionManager,
}

impl Engine {
    /// Main loop: runs on its own thread, processes one batch per iteration.
    pub fn run_loop(&mut self) {
        loop {
            // Drain incoming requests
            self.process_new_requests();
            // Schedule
            let decision = self.scheduler.schedule();
            // Execute batch
            self.execute_batch(decision);
            // Stream results back
            self.emit_tokens();
        }
    }
}
```

**Key design choice:** The engine owns all mutable state (KV cache, scheduler, sessions) on a single thread. No locks needed. API layer is async (tokio) and communicates via channels. This is the pattern used by vLLM's engine and avoids the complexity of fine-grained locking.

---

## 6. Model Directory Structure

```
model_dir/
├── inference_metadata.yaml    # The standard metadata spec
├── tokenizer.json             # HF tokenizers format
├── decoder.onnx               # Main decoder model (from Mobius)
├── decoder_data/              # External weights (if model is >2GB)
│   ├── weights_0.safetensors
│   └── weights_1.safetensors
├── vision_encoder.onnx        # Optional: for VLMs
└── draft_model.onnx           # Optional: for speculative decoding
```

`inference_metadata.yaml` is the single source of truth for how to load and run the model. No hardcoded `model_type` string dispatch.

---

## 7. Crate Structure

```
onnx-genai/
├── Cargo.toml                 # workspace
├── crates/
│   ├── onnx-genai/            # Main library crate (re-exports everything)
│   │   ├── src/lib.rs
│   │   └── Cargo.toml
│   ├── onnx-genai-metadata/   # Inference metadata parser + types
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── schema.rs      # Typed structs for all spec sections
│   │   │   ├── parser.rs      # YAML/JSON parsing + validation
│   │   │   └── validation.rs  # required_capabilities check
│   │   └── Cargo.toml
│   ├── onnx-genai-kv/         # KV cache manager
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── page_table.rs
│   │   │   ├── paged_cache.rs
│   │   │   ├── prefix_cache.rs
│   │   │   ├── tiered.rs      # GPU→CPU→Disk eviction
│   │   │   └── quantized.rs   # KV quantization
│   │   └── Cargo.toml
│   ├── onnx-genai-scheduler/  # Continuous batching scheduler
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── scheduler.rs
│   │   │   ├── policy.rs      # FCFS, priority, fair-share
│   │   │   └── preemption.rs
│   │   └── Cargo.toml
│   ├── onnx-genai-engine/     # Generation engine (ties everything together)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── engine.rs      # Main loop
│   │   │   ├── speculative.rs
│   │   │   ├── pipeline.rs    # Multi-model orchestration
│   │   │   ├── logits.rs      # Logit processor chain
│   │   │   └── sampling.rs
│   │   └── Cargo.toml
│   └── onnx-genai-server/     # OpenAI-compatible HTTP server
│       ├── src/
│       │   ├── main.rs
│       │   ├── routes.rs      # /v1/chat/completions, /v1/completions
│       │   ├── streaming.rs   # SSE streaming
│       │   └── models.rs      # Request/response types
│       └── Cargo.toml
├── tests/
│   ├── integration/           # End-to-end tests with tiny models
│   └── fixtures/              # Test model dirs with metadata
├── docs/
│   ├── DESIGN.md              # This file
│   ├── ARCHITECTURE.md        # Component interaction diagrams
│   └── METADATA_SPEC.md       # Local copy of the spec
└── README.md
```

---

## 8. Dependencies

| Crate | Purpose | Version |
|---|---|---|
| `ort` | ONNX Runtime Rust bindings | latest |
| `tokenizers` | HuggingFace tokenizers | latest |
| `tokio` | Async runtime for HTTP server | 1.x |
| `axum` | HTTP framework (for OpenAI-compatible API) | 0.7+ |
| `serde` + `serde_yaml` + `serde_json` | Metadata parsing | latest |
| `tracing` | Structured logging | latest |
| `thiserror` | Error types | latest |

---

## 9. API Surface

### 9.1 Library API (Rust)

```rust
use onnx_genai::{Engine, EngineConfig, GenerateRequest, GenerateStream};

// Load model
let engine = Engine::from_dir("./models/phi-4/", EngineConfig::default())?;

// Generate (streaming)
let request = GenerateRequest {
    messages: vec![Message::user("What is 2+2?")],
    max_tokens: 100,
    temperature: 0.7,
    ..Default::default()
};

let mut stream = engine.generate(request)?;
while let Some(token) = stream.next().await {
    print!("{}", token.text);
}
```

### 9.2 HTTP API (OpenAI-compatible)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

### 9.3 Multi-session (Agent) API

```rust
// Create a session (KV cache persists across calls)
let session = engine.create_session("phi-4")?;

// First turn
let response1 = session.generate("You are a helpful assistant. What is 2+2?").await?;

// Second turn — prefix cache kicks in, only new tokens computed
let response2 = session.generate("Now multiply that by 3").await?;

// Fork a session (CoW — cheap, shares prefix KV)
let branch = session.fork()?;
let alt_response = branch.generate("Actually, divide by 2 instead").await?;
```

---

## 10. Implementation Phases

### Phase 1: Foundation (Target: working end-to-end for a single model)

- [ ] Workspace + crate scaffold
- [ ] Inference metadata parser (`onnx-genai-metadata`)
- [ ] ORT session loading + basic forward pass
- [ ] Tokenizer integration (HF tokenizers crate)
- [ ] Simple KV cache (non-paged, single sequence)
- [ ] Greedy generation loop
- [ ] Basic logit processors (temperature, top-p, stop sequences)
- [ ] CLI: `onnx-genai generate --model ./path "prompt"`

**Exit criteria:** Can load a Phi-4 ONNX model (from Mobius) and generate text greedily, end-to-end.

### Phase 2: Agent Essentials

- [ ] Paged KV cache (page table, free list, append/rewind)
- [ ] Prefix cache (radix trie, lookup, insert)
- [ ] Multi-session support (persistent KV across turns)
- [ ] CoW fork
- [ ] Continuous batching scheduler (basic FCFS)
- [ ] OpenAI-compatible HTTP server with streaming

**Exit criteria:** Can serve multiple concurrent chat sessions with prefix sharing. Second turn in same session is measurably faster (prefix cache hit).

### Phase 3: Performance

- [ ] Speculative decoding (draft model + greedy acceptance first)
- [ ] Tiered KV storage (GPU→CPU eviction under pressure)
- [ ] Priority-based scheduling + preemption
- [ ] KV cache quantization (fp8)
- [ ] Token streaming with early stopping

**Exit criteria:** Speculative decoding shows >1.5× speedup. System handles 10+ concurrent sessions without OOM.

### Phase 4: Pipeline + Advanced

- [ ] Multi-model pipeline orchestration
- [ ] Vision-language model support (image encoder + decoder)
- [ ] Grammar/JSON constrained decoding
- [ ] Rejection sampling acceptance rule
- [ ] Tree-structured speculative decoding
- [ ] Hardware profile matching

**Exit criteria:** Can run a VLM pipeline (CLIP + decoder) end-to-end via inference metadata declaration.

---

## 11. Testing Strategy

### Unit Tests
- Metadata parsing: valid/invalid/forward-compatible schemas
- KV cache: page allocation, rewind, fork, eviction, quantization
- Prefix cache: lookup, insert, LRU eviction
- Scheduler: policy decisions, preemption triggers
- Logit processors: each processor in isolation + chain ordering

### Integration Tests
- End-to-end generation with a tiny model (2-layer transformer, committed as test fixture)
- Multi-turn session with prefix cache validation
- Concurrent sessions via HTTP API
- Speculative decoding correctness (greedy spec == greedy baseline, token-for-token)

### Benchmarks
- Tokens/sec (single stream, batched)
- Time-to-first-token (TTFT)
- Prefix cache hit rate
- Memory utilization (pages allocated vs wasted)
- Speculative acceptance rate

---

## 12. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language | Rust | Memory safety for KV management, zero-cost concurrency, learning goal, portfolio value |
| NN backend | ORT (via `ort` crate) | Leverages existing EP ecosystem, don't rewrite kernels |
| Model format | ONNX (generated by Mobius) | Standard format, ORT native |
| Tokenizer | HF `tokenizers` crate | Rust native, battle-tested, same format as model ships with |
| HTTP framework | Axum | Performant, ergonomic, tokio-native |
| Concurrency | Single engine thread + async API | Avoids lock contention on hot path; proven pattern (vLLM) |
| Config format | YAML (inference_metadata.yaml) | Human-readable, good Rust serde support |
| KV page size | 16 tokens (configurable) | Balances fragmentation vs waste; matches vLLM default |

---

## 13. Relationship to Existing Projects

| Project | Relationship |
|---|---|
| **onnxruntime-genai** (C++) | This is the Rust alternative. Same scope, different implementation philosophy. |
| **ORT** | Backend dependency. We call ORT sessions; we don't modify or replace ORT. |
| **Mobius** | Model generation tool. Produces the ONNX models we consume. |
| **onnx/onnx#8184** | The spec this project implements. This is the reference implementation. |
| **vLLM** | Architectural inspiration (PagedAttention, continuous batching, scheduler design). |
| **HuggingFace tokenizers** | Direct dependency for tokenization. |

---

## 14. Open Questions

1. **IO Binding vs. Manual Tensor Management** — ORT's `IoBinding` API allows pre-allocating device buffers and avoiding copies. How deeply should we integrate with it? (Likely: deeply, it's the key to avoiding host↔device round-trips for KV cache.)

2. **KV Cache in ORT Session** — ORT sessions typically manage their own state. For paged KV, we need to pass page pointers as inputs. This requires the ONNX model to expose KV as explicit inputs/outputs (which Mobius models do), not hidden internal state.

3. **Multi-GPU** — Phase 1-3 target single GPU. Multi-GPU (tensor parallel) requires splitting model and KV cache across devices. Defer to Phase 4+, but design page table to support multi-device from day one.

4. **Benchmarking Baseline** — Compare against onnxruntime-genai (C++) on same model + same hardware to demonstrate competitive performance despite Rust overhead (expect: minimal difference since hot path is in ORT/CUDA).

---

## 15. Design Decisions (Confirmed)

| Decision | Choice | Notes |
|---|---|---|
| ORT binding | Custom C API wrapper (reference `ort` crate) | `ort` crate not fresh enough; use latest ORT C API directly via `ort-sys` |
| GPU memory | ORT-managed (via IoBinding) | Must support NPU/DirectML/etc, not just CUDA |
| KV cache I/O | Explicit input/output on ONNX model | Mobius generates models with KV as explicit I/O, using opset 24 tensor scatter |
| License | MIT | |
| MSRV | Latest stable | No old Rust version compat needed |
| Quantized models | Must support | INT4/INT8 weight models from Mobius |
| Diffusion | Must support | Not just LLM — diffusion pipelines too |

---

## 16. Quantized Model Support

### How it works with ORT

ORT handles quantized ops natively (MatMulNBits, QLinearConv, etc.). From our runtime's perspective:

- **Weight-only quantization (W4A16, W8A16):** Model weights stored in INT4/INT8, activations in FP16. ORT dequantizes on-the-fly during execution. We load the model as-is — no special handling needed beyond passing the right session options.
- **Weight + Activation quantization (W8A8, W4A8):** Both weights and activations quantized. ORT kernels handle this. We ensure correct input dtype.
- **KV cache quantization:** We quantize/dequantize at the cache boundary (on append/read). Controlled by metadata's `kv_cache.quantization_tolerance`.

```rust
/// Quantization config derived from inference metadata.
pub struct QuantConfig {
    /// Weight quantization scheme (informational — ORT handles execution).
    pub weight_scheme: Option<String>,  // "int4_group128", "awq", "gptq"
    /// KV cache runtime quantization (we manage this).
    pub kv_dtype: DataType,
    /// Layers exempt from KV quantization.
    pub kv_sensitive_layers: Vec<usize>,
}
```

### What we actually need to implement

1. **Model loading:** Pass correct session options (EP selection, optimization level) — ORT does the rest
2. **KV cache quantization:** We own this. Quantize FP16→FP8/INT8 on `append()`, dequantize on read for attention input
3. **Mixed precision metadata:** Read from `inference_metadata.yaml`, apply to KV cache manager

---

## 17. Diffusion Pipeline Support

### Architecture for Diffusion

Diffusion models are fundamentally different from LLM generation:
- No KV cache, no autoregressive loop
- Instead: iterative denoising loop (N steps, typically 20-50)
- Multiple models in a pipeline: text encoder → U-Net/DiT → VAE decoder

### Design

```rust
/// Diffusion pipeline executor.
pub struct DiffusionPipeline {
    /// Text encoder (CLIP/T5)
    text_encoder: OrtSession,
    /// Denoising model (U-Net or DiT)
    denoiser: OrtSession,
    /// VAE decoder (latent → pixel)
    vae_decoder: OrtSession,
    /// Noise scheduler
    scheduler: NoiseScheduler,
}

pub enum NoiseScheduler {
    DDPM { num_steps: usize, beta_schedule: BetaSchedule },
    DDIM { num_steps: usize, eta: f32 },
    EulerDiscrete { num_steps: usize },
    FlowMatching { num_steps: usize, shift: f32 },
}

impl DiffusionPipeline {
    pub fn generate(&self, prompt: &str, config: DiffusionConfig) -> Result<Image> {
        // 1. Encode text
        let text_embeddings = self.text_encoder.run(prompt_tokens)?;

        // 2. Initialize latent noise
        let mut latents = random_latent(config.height, config.width, config.seed);

        // 3. Denoising loop
        for t in self.scheduler.timesteps() {
            let noise_pred = self.denoiser.run(latents, t, text_embeddings)?;
            latents = self.scheduler.step(noise_pred, t, latents);

            // Optional: yield progress
        }

        // 4. Decode latent → image
        let image = self.vae_decoder.run(latents)?;
        Ok(image)
    }
}
```

### Unified via Pipeline Orchestrator

The existing pipeline spec from metadata handles this:

```yaml
pipeline:
  models:
    text_encoder: { type: text_encoder, filename: text_encoder.onnx }
    unet: { type: denoiser, filename: unet.onnx }
    vae: { type: vae_decoder, filename: vae_decoder.onnx }

  strategy:
    kind: diffusion
    scheduler: euler_discrete
    num_steps: 20
    guidance_scale: 7.5

  dataflow:
    - from: text_encoder.last_hidden_state
      to: unet.encoder_hidden_states
    - from: unet.output
      to: vae.latent_sample

  phases:
    text_encoder:
      run_on: prompt_only
    unet:
      run_on: denoise_loop    # new phase type for diffusion
    vae:
      run_on: final_only
```

### What this means for crate structure

Add a new crate or module:

```
crates/
├── onnx-genai-engine/
│   ├── src/
│   │   ├── engine.rs          # LLM generation engine
│   │   ├── diffusion.rs       # Diffusion pipeline engine (NEW)
│   │   ├── pipeline.rs        # Unified pipeline orchestrator (routes to LLM or diffusion)
```

The `pipeline.rs` orchestrator reads `strategy.kind` from metadata:
- `kind: speculative` → LLM generation path
- `kind: diffusion` → Diffusion denoising path
- `kind: null/none` → Simple single-model forward pass

### Key difference from LLM path

| | LLM | Diffusion |
|---|---|---|
| State management | KV cache (paged, persistent) | Latent tensor (temporary, fixed size) |
| Loop | Autoregressive (variable length) | Fixed N steps |
| Batching | Continuous batching across requests | Batch dimension within one request (CFG) |
| Memory pattern | Growing (context accumulates) | Fixed (same size each step) |
| Streaming | Token-by-token | Step-by-step progress |

So diffusion doesn't need the KV cache manager or the continuous batching scheduler. But it shares:
- ORT session management
- Pipeline orchestration
- Device placement
- Inference metadata schema
- The HTTP API layer

---

## 18. ORT C API Wrapper Design

### Why custom wrapper

The `ort` crate (pyke/ort):
- Stuck on older ORT versions
- Heavy abstraction that hides C API details we need (IoBinding, custom allocators)
- We need latest ORT for opset 24 support (tensor scatter for KV)

### Approach

Thin, safe wrapper over `onnxruntime-sys` (or our own bindgen):

```rust
// crates/onnx-genai-ort/src/lib.rs

/// Safe wrapper over ORT C API.
pub struct Environment { /* OrtEnv* */ }
pub struct Session { /* OrtSession* */ }
pub struct IoBinding { /* OrtIoBinding* */ }
pub struct Value { /* OrtValue* — tensor */ }
pub struct Allocator { /* OrtAllocator* */ }
pub struct MemoryInfo { /* OrtMemoryInfo* */ }

impl Session {
    pub fn new(env: &Environment, path: &Path, options: SessionOptions) -> Result<Self>;
    pub fn run(&self, inputs: &[(&str, &Value)]) -> Result<Vec<Value>>;
    pub fn run_with_binding(&self, binding: &IoBinding) -> Result<()>;
}

impl IoBinding {
    /// Bind a pre-allocated tensor to an input/output name.
    /// This is how we pass KV cache pages without copying.
    pub fn bind_input(&mut self, name: &str, value: &Value) -> Result<()>;
    pub fn bind_output(&mut self, name: &str, memory_info: &MemoryInfo) -> Result<()>;
}

impl Value {
    /// Create a tensor on a specific device (GPU/CPU/NPU).
    pub fn tensor(shape: &[i64], dtype: DataType, memory_info: &MemoryInfo) -> Result<Self>;
    /// Create from existing data (zero-copy when possible).
    pub fn from_slice<T>(data: &[T], shape: &[i64]) -> Result<Self>;
}
```

### New crate needed

```
crates/
├── onnx-genai-ort/            # ORT C API safe wrapper (NEW)
│   ├── src/
│   │   ├── lib.rs
│   │   ├── env.rs             # Environment
│   │   ├── session.rs         # Session + SessionOptions
│   │   ├── value.rs           # Tensor values
│   │   ├── binding.rs         # IoBinding
│   │   ├── allocator.rs       # Allocator + MemoryInfo
│   │   └── error.rs           # ORT status → Result
│   ├── ort-sys/               # Raw bindgen (or reference onnxruntime-sys)
│   └── Cargo.toml
```

---

## 19. Updated Crate Dependency Graph

```
onnx-genai-server (binary)
    └── onnx-genai (library, re-exports)
            ├── onnx-genai-engine
            │       ├── onnx-genai-kv
            │       ├── onnx-genai-scheduler
            │       ├── onnx-genai-metadata
            │       └── onnx-genai-ort (NEW)
            └── onnx-genai-metadata
```

---

## 20. Generalized Pipeline Architecture

### The Problem

Generative AI inference isn't just "text in → text out." Real workloads include:

| Pipeline | Input | Output | Models Involved |
|---|---|---|---|
| Text generation (LLM) | text | text | decoder |
| Vision-Language (VLM) | image + text | text | vision_encoder + decoder |
| Text-to-Speech (TTS) | text | audio | text_encoder + acoustic_model + vocoder |
| Speech-to-Text (ASR) | audio | text | feature_extractor + encoder + decoder |
| Audio-to-Audio | audio | audio | encoder + decoder (e.g., voice conversion) |
| Image generation | text | image | text_encoder + denoiser + vae_decoder |
| Image editing | image + text | image | image_encoder + text_encoder + denoiser + vae_decoder |
| Embedding | text/image/audio | vector | encoder |
| Reranking | text pairs | scores | cross_encoder |
| Classification | text/image | labels | encoder + classifier_head |
| OCR | image | text | vision_encoder + decoder |
| Video generation | text/image | video frames | text_encoder + temporal_denoiser + vae_decoder |

### Design Principle: Pipelines as DAGs with Loop Strategies

Every pipeline above is a **directed acyclic graph of models** with optional **loop strategies** controlling iteration:

```
Pipeline = DAG(models, dataflow) + Strategy(loop_kind, termination)
```

Three fundamental loop strategies cover all cases:

1. **Autoregressive** — generate one token at a time until stop condition (LLM, ASR decoder)
2. **Fixed-step iterative** — run N times with evolving state (diffusion, flow matching)
3. **Single-pass** — run once, no iteration (embedding, classification, encoder, vocoder)

### 20.1 Pipeline Spec Schema (Generalized)

```yaml
pipeline:
  # Models in this pipeline
  models:
    model_name:
      filename: "model.onnx"
      type: encoder | decoder | denoiser | vocoder | classifier | embedder
      # Optional: execution constraints
      device_preference: gpu | npu | cpu

  # How data flows between models
  dataflow:
    - from: model_a.output_name
      to: model_b.input_name
      dtype: fp16 | fp32 | int64 | string
      device_transfer: true | false  # needs D2H or H2D copy?

  # Execution strategy
  strategy:
    kind: autoregressive | iterative | single_pass | composite
    # ... kind-specific parameters below

  # Per-model phase gating (when does each model run?)
  phases:
    model_name:
      run_on: prompt_only | every_step | final_only | on_demand
```

### 20.2 Strategy Definitions

#### Autoregressive (LLM, ASR decoder)

```yaml
strategy:
  kind: autoregressive
  decoder: decoder_model_name
  max_tokens: 4096
  stop_conditions:
    - eos_token: true
    - stop_sequences: ["</s>", "<|end|>"]
    - max_tokens: true
  kv_cache:
    enabled: true
    # Links to top-level kv_cache spec for quantization/operations
  speculative:  # optional acceleration
    draft: { producer: ngram, tokens_per_step: 5 }
    acceptance: greedy
```

#### Iterative (Diffusion, Flow Matching)

```yaml
strategy:
  kind: iterative
  denoiser: unet_model_name
  scheduler: euler_discrete | ddim | ddpm | flow_matching
  num_steps: 20
  guidance_scale: 7.5   # classifier-free guidance
  state:
    name: latents
    init: random_normal   # or from_input (for img2img)
    shape: [1, 4, 64, 64]
    dtype: fp16
```

#### Single-Pass (Embedding, Classification, Feature Extraction)

```yaml
strategy:
  kind: single_pass
  model: encoder_model_name
  # No loop, just run once
  batching:
    max_batch_size: 64
    dynamic_batching: true
    padding_strategy: longest | max_length
```

#### Composite (Multi-strategy pipelines)

For pipelines that combine strategies (e.g., ASR = single_pass encoder + autoregressive decoder):

```yaml
strategy:
  kind: composite
  stages:
    - name: encode
      strategy: { kind: single_pass, model: encoder }
      run_on: prompt_only
    - name: decode
      strategy: { kind: autoregressive, decoder: decoder }
      run_on: every_step
```

### 20.3 Concrete Pipeline Examples

#### Text-to-Speech (TTS)

```yaml
pipeline:
  models:
    text_encoder:
      filename: text_encoder.onnx
      type: encoder
    acoustic:
      filename: acoustic_model.onnx
      type: decoder
    vocoder:
      filename: vocoder.onnx
      type: vocoder

  dataflow:
    - from: text_encoder.hidden_states
      to: acoustic.encoder_hidden_states
      dtype: fp16
    - from: acoustic.mel_spectrogram
      to: vocoder.mel_input
      dtype: fp32

  strategy:
    kind: composite
    stages:
      - name: encode_text
        strategy: { kind: single_pass, model: text_encoder }
      - name: generate_mel
        strategy:
          kind: autoregressive
          decoder: acoustic
          stop_conditions:
            - eos_token: true
      - name: synthesize_audio
        strategy: { kind: single_pass, model: vocoder }

  phases:
    text_encoder: { run_on: prompt_only }
    acoustic: { run_on: every_step }
    vocoder: { run_on: final_only }
```

#### Speech-to-Text (ASR / Whisper-style)

```yaml
pipeline:
  models:
    feature_extractor:
      filename: feature_extractor.onnx
      type: encoder
      # Mel spectrogram computation (or could be done in preprocessing)
    encoder:
      filename: encoder.onnx
      type: encoder
    decoder:
      filename: decoder.onnx
      type: decoder

  dataflow:
    - from: feature_extractor.mel_features
      to: encoder.input_features
      dtype: fp32
    - from: encoder.last_hidden_state
      to: decoder.encoder_hidden_states
      dtype: fp16

  strategy:
    kind: composite
    stages:
      - name: extract_features
        strategy: { kind: single_pass, model: feature_extractor }
      - name: encode_audio
        strategy: { kind: single_pass, model: encoder }
      - name: decode_text
        strategy:
          kind: autoregressive
          decoder: decoder
          max_tokens: 448
          stop_conditions:
            - eos_token: true
            - special_tokens: ["<|endoftext|>"]
```

#### Embedding (text, image, or audio)

```yaml
pipeline:
  models:
    encoder:
      filename: encoder.onnx
      type: embedder

  strategy:
    kind: single_pass
    model: encoder
    batching:
      max_batch_size: 128
      dynamic_batching: true
      padding_strategy: longest
    pooling: mean | cls | last_token  # how to get fixed-size embedding from variable-length output
    normalize: true  # L2 normalize output embeddings
```

#### Audio-to-Audio (Voice Conversion / Enhancement)

```yaml
pipeline:
  models:
    content_encoder:
      filename: content_encoder.onnx
      type: encoder
    speaker_encoder:
      filename: speaker_encoder.onnx
      type: encoder
    decoder:
      filename: decoder.onnx
      type: decoder
    vocoder:
      filename: vocoder.onnx
      type: vocoder

  dataflow:
    - from: content_encoder.content_features
      to: decoder.content_input
      dtype: fp16
    - from: speaker_encoder.speaker_embedding
      to: decoder.speaker_condition
      dtype: fp16
    - from: decoder.mel_output
      to: vocoder.mel_input
      dtype: fp32

  strategy:
    kind: composite
    stages:
      - name: encode_content
        strategy: { kind: single_pass, model: content_encoder }
      - name: encode_speaker
        strategy: { kind: single_pass, model: speaker_encoder }
      - name: decode
        strategy: { kind: autoregressive, decoder: decoder }
      - name: vocalize
        strategy: { kind: single_pass, model: vocoder }
```

#### Image-to-Text (OCR / Captioning)

```yaml
pipeline:
  models:
    vision_encoder:
      filename: vision_encoder.onnx
      type: encoder
    decoder:
      filename: decoder.onnx
      type: decoder

  dataflow:
    - from: vision_encoder.image_features
      to: decoder.encoder_hidden_states
      dtype: fp16

  strategy:
    kind: composite
    stages:
      - name: encode_image
        strategy: { kind: single_pass, model: vision_encoder }
      - name: generate_text
        strategy:
          kind: autoregressive
          decoder: decoder
          max_tokens: 1024
          kv_cache: { enabled: true }
```

#### Reranking / Cross-Encoder

```yaml
pipeline:
  models:
    cross_encoder:
      filename: cross_encoder.onnx
      type: classifier

  strategy:
    kind: single_pass
    model: cross_encoder
    batching:
      max_batch_size: 256
      dynamic_batching: true
    output:
      type: scores   # single float per input pair
      activation: sigmoid  # or none for raw logits
```

### 20.4 API Surface for Different Pipeline Types

```rust
/// Unified pipeline handle — different execution paths based on strategy.
pub enum Pipeline {
    Autoregressive(AutoregressivePipeline),
    Iterative(IterativePipeline),
    SinglePass(SinglePassPipeline),
    Composite(CompositePipeline),
}

/// Public API adapts to pipeline type:
impl Engine {
    // --- Text generation (autoregressive) ---
    pub fn generate(&self, request: GenerateRequest) -> GenerateStream;

    // --- Embedding (single_pass, batched) ---
    pub fn embed(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>>;

    // --- Image generation (iterative) ---
    pub fn generate_image(&self, request: ImageRequest) -> ImageStream;

    // --- Speech-to-text (composite: single_pass + autoregressive) ---
    pub fn transcribe(&self, audio: &AudioInput) -> TranscribeStream;

    // --- Text-to-speech (composite: single_pass + autoregressive + single_pass) ---
    pub fn synthesize(&self, text: &str, voice: &str) -> Result<AudioOutput>;

    // --- Generic (any pipeline) ---
    pub fn run_pipeline(&self, inputs: PipelineInputs) -> PipelineOutputStream;
}
```

### 20.5 Preprocessing and Postprocessing

Some pipelines need non-NN processing (audio feature extraction, image resizing, tokenization). These are NOT ORT sessions but host-side operations:

```yaml
pipeline:
  preprocessing:
    - name: audio_features
      kind: mel_spectrogram
      params: { sample_rate: 16000, n_mels: 80, n_fft: 400 }
    - name: image_resize
      kind: resize_and_normalize
      params: { size: [224, 224], mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] }

  postprocessing:
    - name: audio_output
      kind: griffin_lim  # or just pass-through if vocoder produces waveform
    - name: detokenize
      kind: tokenizer_decode
```

Implementation: a `ProcessingStep` trait:

```rust
pub trait ProcessingStep: Send + Sync {
    fn process(&self, input: &Tensor) -> Result<Tensor>;
    fn name(&self) -> &str;
}

// Built-in steps:
pub struct MelSpectrogram { config: MelConfig }
pub struct ImageResize { size: (usize, usize), normalize: bool }
pub struct TokenizerDecode { tokenizer: Tokenizer }
pub struct AudioResample { target_sample_rate: u32 }
```

### 20.6 Streaming Behavior per Strategy

| Strategy | Streaming Output | What's Streamed |
|---|---|---|
| Autoregressive | Yes, token-by-token | Each generated token |
| Iterative | Yes, step-by-step | Intermediate latent (denoised image preview) |
| Single-pass | No (or batch progress) | Final result only |
| Composite | Depends on final stage | Follows the last stage's streaming behavior |

### 20.7 Memory Management per Strategy

| Strategy | KV Cache | State Tensor | Batching |
|---|---|---|---|
| Autoregressive | Paged, growing | N/A | Continuous batching (scheduler) |
| Iterative | N/A | Fixed size, evolving | Request-level batching (CFG doubles batch) |
| Single-pass | N/A | N/A | Dynamic batching (accumulate → run) |
| Composite | Per-stage | Per-stage | Follows bottleneck stage |

This means:
- KV cache manager only activates for autoregressive stages
- Scheduler only needed for autoregressive stages
- Single-pass stages use a simpler "batch accumulator" pattern
- Iterative stages manage a fixed-size state tensor

### 20.8 Crate Structure Update

```
crates/onnx-genai-engine/src/
├── engine.rs              # Top-level engine (routes to strategy)
├── autoregressive.rs      # Autoregressive generation loop + KV + speculative
├── iterative.rs           # Diffusion/flow denoising loop
├── single_pass.rs         # Embedding, classification, single forward
├── composite.rs           # Multi-stage pipeline orchestrator
├── pipeline.rs            # Pipeline loading + DAG construction from metadata
├── preprocessing.rs       # Non-NN processing steps
├── logits.rs              # Logit processor chain (autoregressive only)
├── sampling.rs            # Token sampling
└── speculative.rs         # Speculative decoding (autoregressive only)
```

---

## 21. Tool Use / Function Calling

### 21.1 Overview

A local coding agent needs to:
1. See available tools (file_read, file_write, shell_exec, search, etc.)
2. Decide when to call a tool vs. continue generating text
3. Output a structured tool call (function name + arguments)
4. Receive tool results and continue generation

This must work with **any model** that supports tool use (Llama, Qwen, Mistral, Phi, etc.) — each has different chat templates and tool call formats.

### 21.2 Tool Schema

```rust
/// A tool available to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name (must match what model outputs).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema for the tool's parameters.
    pub parameters: serde_json::Value,
}

/// How the model should use tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolChoice {
    /// Model decides whether to call a tool.
    Auto,
    /// Model MUST call one of the provided tools.
    Required,
    /// Model must call this specific tool.
    Function { name: String },
    /// Model must NOT call any tool (plain text response).
    None,
}

/// A tool call emitted by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for this call (for matching results).
    pub id: String,
    /// Tool name.
    pub name: String,
    /// JSON arguments.
    pub arguments: serde_json::Value,
}

/// Result of a tool execution, fed back to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Matches ToolCall.id.
    pub tool_call_id: String,
    /// Tool output (string or structured).
    pub content: String,
    /// Whether the tool call errored.
    pub is_error: bool,
}
```

### 21.3 Chat Template Abstraction

Different models encode tool calls differently. We abstract this:

```rust
/// Formats messages + tools into model-specific token sequences.
pub trait ChatTemplate: Send + Sync {
    /// Format a conversation with tool definitions into a prompt.
    fn apply(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        tool_choice: &ToolChoice,
    ) -> Vec<TokenId>;

    /// Parse generated text to detect tool calls.
    /// Returns None if the output is plain text (no tool call).
    fn parse_tool_calls(&self, generated_text: &str) -> Option<Vec<ToolCall>>;

    /// Get stop tokens that indicate end-of-turn or end-of-tool-call.
    fn stop_tokens(&self) -> Vec<StopSequence>;

    /// Get the tool-call start marker (e.g., "<|tool_call|>", "<function_call>").
    fn tool_call_start_marker(&self) -> Option<&str>;
}

/// Built-in templates for common model families.
pub enum BuiltinTemplate {
    /// Llama 3.x / Llama 4 tool format
    Llama,
    /// Qwen 2.5 / Qwen 3 tool format
    Qwen,
    /// Mistral / Mixtral tool format  
    Mistral,
    /// Phi-4 tool format
    Phi,
    /// Generic: uses Jinja template from tokenizer_config.json
    Jinja { template: String },
}
```

### 21.4 Tool Call Detection During Generation

Two approaches, use both:

**A. Token-level detection (fast path):**
```rust
/// Watches generated tokens for tool-call start markers.
pub struct ToolCallDetector {
    /// Partial match buffer for multi-token markers.
    buffer: String,
    /// Known start markers for active template.
    start_markers: Vec<String>,
    /// Once detected, switch to constrained mode.
    detected: bool,
}

impl LogitProcessor for ToolCallDetector {
    fn signal(&self, context: &ProcessorContext) -> Option<ProcessorSignal> {
        if self.detected {
            Some(ProcessorSignal::ToolCallStart)
        } else {
            None
        }
    }
}
```

**B. Constrained generation after detection:**

Once we detect a tool call is starting, switch to JSON Schema constrained decoding to guarantee valid tool call JSON:

```rust
/// After tool_call_start_marker detected, constrain output to valid tool JSON.
pub struct ToolCallConstraint {
    /// JSON Schema built from available tool definitions.
    schema: JsonSchemaConstraint,
    /// Which tools are allowed (from ToolChoice).
    allowed_tools: Vec<String>,
}
```

### 21.5 Multi-Turn Tool Loop

```rust
/// A generation session with tool-use loop.
pub struct ToolSession {
    engine: Engine,
    tools: Vec<ToolDefinition>,
    template: Box<dyn ChatTemplate>,
    messages: Vec<ChatMessage>,
}

impl ToolSession {
    /// Run one generation step. Returns either text or tool calls.
    pub fn step(&mut self) -> Result<StepResult> { ... }
    
    /// Feed tool results back and continue generation.
    pub fn submit_tool_results(&mut self, results: Vec<ToolResult>) -> Result<StepResult> { ... }
}

pub enum StepResult {
    /// Model produced final text response.
    Text(String),
    /// Model wants to call tools. Caller executes them and calls submit_tool_results.
    ToolCalls(Vec<ToolCall>),
    /// Model produced text + tool calls (parallel).
    Mixed { text: String, tool_calls: Vec<ToolCall> },
}
```

### 21.6 HTTP API (OpenAI-compatible)

```json
POST /v1/chat/completions
{
  "model": "local-model",
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "file_read",
        "description": "Read file contents",
        "parameters": { "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] }
      }
    }
  ],
  "tool_choice": "auto"
}
```

Response with tool call:
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": { "name": "file_read", "arguments": "{\"path\": \"src/main.rs\"}" }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

### 21.7 KV Cache Across Tool Turns

Critical for coding agent performance — don't recompute the entire context each turn:

```
Turn 1: [system + tools + user] → generate → tool_call     (KV cached)
Turn 2: [... + tool_result + ...] → generate → text        (reuse Turn 1 KV, append new)
```

The multi-session support (already implemented) handles this. Each `SessionId` preserves KV state. Tool results are appended as new tokens, and only the new portion requires prefill.

---

## 22. Grammar-Based Constrained Decoding

### 22.1 Overview

JSON constraint exists. Now generalize to arbitrary grammars for:
- JSON Schema (specific shape, not just valid JSON)
- Regex patterns (dates, emails, enums)
- GBNF/EBNF grammars (custom DSLs, code, XML)
- Tool call formats (model-specific structured output)

### 22.2 Grammar Specification

```rust
/// A grammar that constrains generation output.
#[derive(Debug, Clone)]
pub enum Grammar {
    /// Any valid JSON.
    Json,
    /// JSON conforming to a specific schema.
    JsonSchema(serde_json::Value),
    /// Output must match this regex.
    Regex(String),
    /// Context-free grammar in GBNF notation (llama.cpp compatible).
    Gbnf(String),
    /// Choice between literal strings.
    Choice(Vec<String>),
}

/// Compiled grammar ready for token-level enforcement.
pub trait CompiledGrammar: Send + Sync {
    /// Given current generated text, which tokens are allowed next?
    fn allowed_tokens(&self, state: &GrammarState, vocab: &Vocabulary) -> TokenMask;
    
    /// Advance the grammar state after accepting a token.
    fn advance(&self, state: &mut GrammarState, token_text: &str);
    
    /// Is the grammar in an accepting state? (output is complete & valid)
    fn is_accepting(&self, state: &GrammarState) -> bool;
    
    /// Can the grammar still reach an accepting state? (or is it stuck)
    fn is_dead(&self, state: &GrammarState) -> bool;
}

/// Opaque grammar automaton state.
pub struct GrammarState {
    /// Stack of automaton states (for recursive grammars).
    stack: Vec<u32>,
    /// Current position in the generated output.
    position: usize,
}

/// Bit vector over vocabulary — true = token is allowed.
pub struct TokenMask {
    bits: Vec<u64>,
    vocab_size: usize,
}

impl TokenMask {
    pub fn apply_to_logits(&self, logits: &mut [f32]) {
        for (i, logit) in logits.iter_mut().enumerate() {
            if !self.is_set(i) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
}
```

### 22.3 JSON Schema Constraint

More powerful than plain JSON — ensures output matches a specific schema:

```rust
pub struct JsonSchemaConstraint {
    schema: serde_json::Value,
    /// Precomputed: for each schema node, what characters can appear next.
    automaton: SchemaAutomaton,
}

/// Handles:
/// - Required/optional properties
/// - Type enforcement (string, number, boolean, null, array, object)
/// - Enum values
/// - String patterns (regex within strings)
/// - Nested objects/arrays
/// - Min/max length, min/max items
/// - oneOf/anyOf/allOf
impl CompiledGrammar for JsonSchemaConstraint { ... }
```

### 22.4 GBNF Grammar Engine

GBNF (GGML BNF) is the de-facto standard for grammar-constrained generation (used by llama.cpp, vLLM, etc.):

```
root   ::= object
object ::= "{" ws members ws "}"
members ::= pair ("," ws pair)*
pair   ::= string ":" ws value
value  ::= string | number | "true" | "false" | "null" | object | array
...
```

```rust
pub struct GbnfEngine {
    /// Parsed grammar rules.
    rules: Vec<GbnfRule>,
    /// Rule name → index.
    rule_index: HashMap<String, usize>,
    /// Root rule name.
    root: String,
}

impl GbnfEngine {
    pub fn compile(grammar_text: &str) -> Result<Self>;
}

impl CompiledGrammar for GbnfEngine { ... }
```

### 22.5 Regex Constraint

```rust
pub struct RegexConstraint {
    /// DFA states compiled from the regex.
    dfa: CompiledDfa,
}

impl RegexConstraint {
    pub fn new(pattern: &str) -> Result<Self> {
        // Compile regex to DFA for efficient per-token stepping
        let dfa = compile_regex_to_dfa(pattern)?;
        Ok(Self { dfa })
    }
}

impl CompiledGrammar for RegexConstraint { ... }
```

### 22.6 Performance: Precomputed Token Masks

For large vocabularies (32K-128K tokens), computing `allowed_tokens` per step is expensive. Optimization: **precompute masks for common grammar states**.

```rust
/// Cache of grammar_state → allowed token mask.
pub struct GrammarCache {
    /// State hash → precomputed mask.
    cache: HashMap<u64, Arc<TokenMask>>,
    /// Maximum cache entries.
    max_size: usize,
}
```

For JSON Schema specifically, most states repeat (e.g., "inside a string", "expecting colon", "expecting value"). Precompute those ~20 common states → amortized O(1) per token.

### 22.7 Integration with Logit Processor Chain

```rust
/// Grammar-based logit processor (slots into existing chain).
pub struct GrammarProcessor {
    grammar: Box<dyn CompiledGrammar>,
    state: GrammarState,
    vocab: Arc<Vocabulary>,
    cache: GrammarCache,
}

impl LogitProcessor for GrammarProcessor {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) {
        let mask = self.cache.get_or_compute(&self.state, |state| {
            self.grammar.allowed_tokens(state, &self.vocab)
        });
        mask.apply_to_logits(logits);
    }
    
    fn signal(&self, _context: &ProcessorContext) -> Option<ProcessorSignal> {
        if self.grammar.is_accepting(&self.state) {
            Some(ProcessorSignal::GrammarComplete)
        } else {
            None
        }
    }
}
```

### 22.8 HTTP API

```json
POST /v1/chat/completions
{
  "model": "local-model",
  "messages": [...],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "tool_call",
      "schema": { "type": "object", "properties": { "name": { "type": "string" }, "args": { "type": "object" } }, "required": ["name", "args"] }
    }
  }
}
```

Or with GBNF:
```json
{
  "grammar": "root ::= \"{\" ws \"\\\"action\\\":\" ws action ...",
}
```

---

## 23. Conditional Generation (Fill-in-the-Middle / Infilling)

### 23.1 Overview

A coding agent doesn't just append text — it needs to:
- **Fill in the middle** (FIM): given prefix + suffix, generate the middle
- **Complete at cursor**: given code before and after cursor, generate the insertion
- **Constrained completion**: generate code that satisfies surrounding context (types, imports)

### 23.2 FIM Format

Models use special tokens for FIM. The format varies:

```rust
/// Fill-in-the-middle configuration.
#[derive(Debug, Clone)]
pub struct FimConfig {
    /// Token/string that marks the beginning of prefix.
    pub prefix_token: String,    // e.g., "<|fim_prefix|>" or "<PRE>"
    /// Token/string that marks the beginning of middle (to be generated).
    pub middle_token: String,    // e.g., "<|fim_middle|>" or "<MID>"
    /// Token/string that marks the beginning of suffix.
    pub suffix_token: String,    // e.g., "<|fim_suffix|>" or "<SUF>"
    /// Format: PSM (prefix-suffix-middle) or SPM (suffix-prefix-middle).
    pub format: FimFormat,
}

#[derive(Debug, Clone)]
pub enum FimFormat {
    /// Prefix → Suffix → Middle (most models: StarCoder, CodeLlama, Qwen)
    PSM,
    /// Suffix → Prefix → Middle (some older models)
    SPM,
}

impl FimConfig {
    /// Auto-detect from tokenizer_config.json or model metadata.
    pub fn from_tokenizer_config(config: &serde_json::Value) -> Option<Self>;
    
    /// Format a FIM prompt.
    pub fn format_prompt(&self, prefix: &str, suffix: &str) -> String {
        match self.format {
            FimFormat::PSM => format!(
                "{}{}{}{}{}",
                self.prefix_token, prefix, self.suffix_token, suffix, self.middle_token
            ),
            FimFormat::SPM => format!(
                "{}{}{}{}{}",
                self.suffix_token, suffix, self.prefix_token, prefix, self.middle_token
            ),
        }
    }
}
```

### 23.3 Coding Agent Request Types

```rust
/// A code generation request (superset of plain text generation).
pub struct CodeGenerateRequest {
    /// The generation mode.
    pub mode: CodeMode,
    /// Language hint (for stop heuristics).
    pub language: Option<String>,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Stop sequences (in addition to model defaults).
    pub stop: Vec<String>,
    /// Temperature (lower = more deterministic for code).
    pub temperature: f32,
}

#[derive(Debug, Clone)]
pub enum CodeMode {
    /// Standard completion: continue from prefix.
    Complete {
        prefix: String,
    },
    /// Fill-in-the-middle: generate between prefix and suffix.
    Infill {
        prefix: String,
        suffix: String,
    },
    /// Multi-file context: other files as context + FIM in target file.
    RepoContext {
        /// Other relevant files (path → content).
        context_files: Vec<(String, String)>,
        /// Target file prefix (before cursor).
        prefix: String,
        /// Target file suffix (after cursor).
        suffix: String,
        /// Target file path (for language detection).
        file_path: String,
    },
}
```

### 23.4 Stop Conditions for Code Generation

Code completion needs smarter stopping than plain text:

```rust
/// Code-aware stop conditions.
pub struct CodeStopConditions {
    /// Stop at end of logical block (matching braces/indentation).
    pub stop_at_block_end: bool,
    /// Stop after N complete lines.
    pub max_lines: Option<usize>,
    /// Stop if indentation returns to or before the starting level.
    pub stop_at_dedent: bool,
    /// Stop at these literal strings.
    pub stop_sequences: Vec<String>,
    /// Language-specific: stop at next function/class definition.
    pub stop_at_next_definition: bool,
}

/// Implements stop logic as a LogitProcessor.
pub struct CodeStopProcessor {
    config: CodeStopConditions,
    language: Language,
    start_indent: usize,
}

impl LogitProcessor for CodeStopProcessor {
    fn signal(&self, context: &ProcessorContext) -> Option<ProcessorSignal> {
        // Check indentation, brace matching, line count, etc.
        ...
    }
}
```

### 23.5 Suffix-Aware Constrained Generation

For infilling, the generated text must **merge cleanly** with the suffix. This means:

```rust
/// Ensures generated output transitions smoothly into the suffix.
pub struct SuffixConstraint {
    /// The suffix text that follows the generation.
    suffix: String,
    /// Characters of suffix we've already "eaten" (overlap detection).
    overlap_detected: usize,
}

impl LogitProcessor for SuffixConstraint {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext) {
        // If generated text is starting to reproduce the suffix,
        // boost EOS / stop probability.
        // Prevents: prefix + generated + suffix having duplicated content.
    }
    
    fn signal(&self, context: &ProcessorContext) -> Option<ProcessorSignal> {
        if self.detected_suffix_overlap(context) {
            Some(ProcessorSignal::SuffixOverlap)
        } else {
            None
        }
    }
}
```

### 23.6 Repo-Level Context (for Coding Agent)

A coding agent needs multi-file awareness:

```rust
/// Formats repository context for the model.
pub trait RepoContextFormatter: Send + Sync {
    /// Format context files + target FIM into a single prompt.
    fn format(
        &self,
        context_files: &[(String, String)],  // (path, content)
        target_path: &str,
        prefix: &str,
        suffix: &str,
        fim_config: &FimConfig,
    ) -> String;
    
    /// Maximum context window to use for repo context (tokens).
    fn max_context_tokens(&self) -> usize;
    
    /// Prioritize which files to include when context is limited.
    fn rank_context_files(
        &self,
        target_path: &str,
        available_files: &[(String, String)],
    ) -> Vec<usize>;
}

/// Default: recently edited files + imports/dependencies first.
pub struct DefaultRepoFormatter {
    max_tokens: usize,
}
```

### 23.7 HTTP API Extensions

```json
POST /v1/completions
{
  "model": "local-model",
  "prompt": "<|fim_prefix|>def fibonacci(n):\n    <|fim_suffix|>\n    return result<|fim_middle|>",
  "max_tokens": 100,
  "temperature": 0.2,
  "stop": ["\n\n"]
}
```

Higher-level code completion endpoint:
```json
POST /v1/code/completions
{
  "model": "local-model",
  "mode": "infill",
  "prefix": "def fibonacci(n):\n    ",
  "suffix": "\n    return result",
  "language": "python",
  "max_tokens": 100,
  "stop_at_dedent": true,
  "context_files": [
    { "path": "utils.py", "content": "..." }
  ]
}
```

### 23.8 Integration: Tool Use + Grammar + FIM Together

A coding agent request flow:

```
1. User: "Add error handling to this function"
2. Engine receives: system prompt + tools + code context
3. Model decides: call file_read tool → get current code
4. Tool result fed back (KV cache preserved)
5. Model decides: generate code edit (FIM mode, grammar-constrained to valid syntax)
6. Model outputs: tool_call(file_write, {path, content})  ← JSON Schema constrained
7. Agent executes write, feeds result back
8. Model: "Done. Added try/except with logging." (plain text)
```

All three systems compose:
- **Tool use**: decides *what* to do (read/write/search)
- **Grammar**: ensures *structured output* is valid (tool call JSON, code blocks)
- **FIM/conditional**: generates *code content* that fits surrounding context

### 23.9 Crate Structure Update

```
crates/onnx-genai-engine/src/
├── constraints/
│   ├── mod.rs              # CompiledGrammar trait + TokenMask
│   ├── json.rs             # Existing JSON constraint (refactored)
│   ├── json_schema.rs      # JSON Schema constraint
│   ├── regex.rs            # Regex → DFA constraint
│   ├── gbnf.rs             # GBNF grammar engine
│   └── choice.rs           # Simple enum/choice constraint
├── tools/
│   ├── mod.rs              # ToolDefinition, ToolCall, ToolResult types
│   ├── template.rs         # ChatTemplate trait + built-in templates
│   ├── detector.rs         # ToolCallDetector (logit processor)
│   └── session.rs          # ToolSession (multi-turn loop)
├── code/
│   ├── mod.rs              # CodeGenerateRequest, CodeMode
│   ├── fim.rs              # FimConfig, format detection
│   ├── stop.rs             # CodeStopProcessor
│   ├── suffix.rs           # SuffixConstraint
│   └── repo_context.rs     # RepoContextFormatter
├── engine.rs
├── pipeline.rs
├── logits.rs
├── sampling.rs
└── speculative.rs
```

---

## 24. Sampling Policy & Configuration

### 24.1 Current State

Already implemented:
- `TemperatureProcessor` — divide logits by T
- `TopKProcessor` — keep top-K, -inf rest
- `TopPProcessor` — nucleus sampling
- `RepetitionPenaltyProcessor` — penalize repeated tokens
- `StopSequenceProcessor` — detect stop strings
- `ConstraintProcessor` — grammar/JSON masking
- `sample_greedy` + `sample_categorical` — final token selection

### 24.2 Missing Samplers

```rust
/// Min-P sampling: only keep tokens with P >= min_p * P(top_token).
/// More adaptive than top-p for varying entropy distributions.
pub struct MinPProcessor {
    pub min_p: f32,
}

/// Frequency penalty: penalize based on count, not just presence.
/// penalty = -frequency_penalty * count(token)
pub struct FrequencyPenaltyProcessor {
    pub penalty: f32,
}

/// Presence penalty: flat penalty if token appeared at all.
/// Different from repetition_penalty (which scales the logit).
pub struct PresencePenaltyProcessor {
    pub penalty: f32,
}

/// Top-A sampling: adaptive threshold based on entropy.
pub struct TopAProcessor {
    pub top_a: f32,
}

/// Mirostat: target a specific perplexity (entropy) level.
/// Self-tuning temperature that adapts during generation.
pub struct MirostatProcessor {
    pub tau: f32,        // target entropy
    pub eta: f32,        // learning rate
    mu: f32,             // evolving state
    pub version: MirostatVersion,
}

pub enum MirostatVersion { V1, V2 }

/// Typical sampling: keep tokens within typical information content.
pub struct TypicalPProcessor {
    pub p: f32,
}

/// DRY (Don't Repeat Yourself): penalize n-gram repetitions.
/// More sophisticated than simple repetition penalty.
pub struct DryProcessor {
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: usize,
    pub sequence_breakers: Vec<TokenId>,
}

/// XTC (eXclude Top Choices): randomly exclude top-probability tokens
/// to increase diversity while maintaining coherence.
pub struct XtcProcessor {
    pub probability: f32,  // chance of excluding
    pub threshold: f32,    // only exclude above this probability
}
```

### 24.3 Sampling Configuration (User-Facing)

```rust
/// Complete sampling configuration exposed via API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    // --- Temperature family ---
    /// Temperature for logit scaling. 0 = greedy.
    pub temperature: f32,
    /// Dynamic temperature range [min, max] (optional).
    pub dynatemp_range: Option<(f32, f32)>,
    
    // --- Truncation family ---
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub top_a: Option<f32>,
    pub typical_p: Option<f32>,
    
    // --- Penalty family ---
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub dry: Option<DryConfig>,
    pub xtc: Option<XtcConfig>,
    
    // --- Adaptive ---
    pub mirostat: Option<MirostatConfig>,
    
    // --- Seed ---
    pub seed: Option<u64>,
}

impl SamplingConfig {
    /// Build the processor chain from this config.
    /// Order: repetition/DRY → constraints → temperature → truncation → XTC
    pub fn build_chain(&self) -> ProcessorChain { ... }
}
```

### 24.4 Processor Ordering

Order matters. Our canonical order (matches community consensus):

```
1. Repetition/Frequency/Presence/DRY penalties  (modify logits based on history)
2. Grammar/Constraint masking                    (hard mask: -inf invalid tokens)
3. Temperature                                   (scale distribution)
4. Top-K                                         (coarse truncation)
5. Top-P / Min-P / Top-A / Typical-P            (fine truncation)
6. XTC                                           (random exclusion)
7. [Sample from resulting distribution]
8. Mirostat feedback                             (adjust for next step)
```

This is configurable — users can reorder via explicit `processor_order` field.

---

## 25. Extensibility & Component Replacement

### 25.1 Design Philosophy

Every major subsystem is behind a **trait** (interface). Users can:
- Swap implementations at compile time (feature flags + generics)
- Swap at runtime (trait objects / `Box<dyn Trait>`)
- Extend without forking (register custom processors, caches, samplers)

### 25.2 Trait Contracts (Public API Surface)

```rust
// === KV Cache ===
// Users can replace paged cache with: ring buffer, offloaded (CPU/disk), quantized, etc.
pub trait KvCacheOps: Send + Sync {
    fn create_sequence(&mut self) -> SequenceId;
    fn delete_sequence(&mut self, seq: SequenceId) -> Result<(), KvError>;
    fn append(&mut self, seq: SequenceId, num_tokens: usize) -> Result<(), KvError>;
    fn rewind_to(&mut self, seq: SequenceId, position: usize) -> Result<(), KvError>;
    fn fork(&mut self, source: SequenceId, position: usize) -> Result<SequenceId, KvError>;
    fn checkpoint(&self, seq: SequenceId) -> Result<CacheCheckpoint, KvError>;
    fn restore(&mut self, seq: SequenceId, checkpoint: CacheCheckpoint) -> Result<(), KvError>;
    fn sequence_length(&self, seq: SequenceId) -> Option<usize>;
    fn total_pages_used(&self) -> usize;
    fn total_pages_available(&self) -> usize;
    
    // New: device placement
    fn device(&self) -> Device;
    // New: materialization for IoBinding
    fn materialize(&self, seq: SequenceId) -> Result<MaterializedKv, KvError>;
}

// === Logit Processing ===
// Already a trait — users register custom processors.
pub trait LogitProcessor: Send + Sync {
    fn process(&self, logits: &mut [f32], context: &ProcessorContext);
    fn name(&self) -> &str;
    fn signal(&self, _context: &ProcessorContext) -> Option<ProcessorSignal> { None }
    /// Priority in the chain (lower = earlier). Default = 100.
    fn priority(&self) -> u32 { 100 }
}

// === Sampling ===
pub trait Sampler: Send + Sync {
    /// Select a token from processed logits.
    fn sample(&mut self, logits: &[f32], context: &ProcessorContext) -> TokenId;
    fn name(&self) -> &str;
}

// Built-in samplers:
pub struct GreedySampler;
pub struct CategoricalSampler { rng: StdRng }
pub struct MirostatSampler { tau: f32, eta: f32, mu: f32 }

// === Scheduling ===
pub trait SchedulerPolicy: Send + Sync {
    /// Select which sequences run in the next batch.
    fn select_batch(
        &mut self,
        waiting: &[SequenceState],
        running: &[SequenceState],
        max_batch_tokens: usize,
    ) -> SchedulerDecision;
    
    fn name(&self) -> &str;
}

pub enum SchedulerDecision {
    /// Run these sequences.
    Run(Vec<SequenceId>),
    /// Preempt some running sequences to make room.
    Preempt { victims: Vec<SequenceId>, run: Vec<SequenceId> },
    /// No work to do.
    Idle,
}

// Built-in policies:
pub struct FcfsPolicy;           // First-come-first-served
pub struct PriorityPolicy;       // Priority queue with preemption  
pub struct FairnessPolicy;       // Round-robin with starvation prevention

// === Chat Template ===
pub trait ChatTemplate: Send + Sync {
    fn apply(&self, messages: &[ChatMessage], tools: &[ToolDefinition], tool_choice: &ToolChoice) -> Vec<TokenId>;
    fn parse_tool_calls(&self, generated_text: &str) -> Option<Vec<ToolCall>>;
    fn stop_tokens(&self) -> Vec<StopSequence>;
}

// === Preprocessing ===
pub trait ProcessingStep: Send + Sync {
    fn process(&self, input: &Tensor) -> Result<Tensor>;
    fn name(&self) -> &str;
}

// === Model Loading ===
pub trait ModelLoader: Send + Sync {
    fn load_session(&self, path: &Path, options: &SessionOptions) -> Result<Session>;
    fn supports_format(&self, path: &Path) -> bool;
}
```

### 25.3 Registry Pattern (Runtime Extensibility)

```rust
/// Global registry for user-provided components.
pub struct EngineBuilder {
    config: EngineConfig,
    // Replaceable components:
    kv_cache: Option<Box<dyn KvCacheOps>>,
    scheduler_policy: Option<Box<dyn SchedulerPolicy>>,
    sampler: Option<Box<dyn Sampler>>,
    chat_template: Option<Box<dyn ChatTemplate>>,
    model_loader: Option<Box<dyn ModelLoader>>,
    // Additive components:
    logit_processors: Vec<Box<dyn LogitProcessor>>,
    preprocessing_steps: Vec<Box<dyn ProcessingStep>>,
    postprocessing_steps: Vec<Box<dyn ProcessingStep>>,
    constraints: Vec<Box<dyn Constraint>>,
}

impl EngineBuilder {
    pub fn new(config: EngineConfig) -> Self { ... }
    
    // --- Replace core components ---
    pub fn with_kv_cache(mut self, cache: impl KvCacheOps + 'static) -> Self {
        self.kv_cache = Some(Box::new(cache));
        self
    }
    
    pub fn with_scheduler_policy(mut self, policy: impl SchedulerPolicy + 'static) -> Self {
        self.scheduler_policy = Some(Box::new(policy));
        self
    }
    
    pub fn with_sampler(mut self, sampler: impl Sampler + 'static) -> Self {
        self.sampler = Some(Box::new(sampler));
        self
    }
    
    pub fn with_chat_template(mut self, template: impl ChatTemplate + 'static) -> Self {
        self.chat_template = Some(Box::new(template));
        self
    }
    
    // --- Add extra processors ---
    pub fn add_logit_processor(mut self, processor: impl LogitProcessor + 'static) -> Self {
        self.logit_processors.push(Box::new(processor));
        self
    }
    
    pub fn add_preprocessing(mut self, step: impl ProcessingStep + 'static) -> Self {
        self.preprocessing_steps.push(Box::new(step));
        self
    }
    
    /// Build the engine with all configured components.
    pub fn build(self, model_dir: &Path) -> Result<Engine> { ... }
}
```

Usage:
```rust
let engine = EngineBuilder::new(config)
    .with_kv_cache(MyCustomQuantizedCache::new(...))
    .with_scheduler_policy(PriorityPolicy::new())
    .add_logit_processor(MyDomainSpecificFilter::new())
    .build(model_dir)?;
```

### 25.4 Feature Flags (Compile-Time Selection)

```toml
[features]
default = ["paged-kv", "json-constraint"]

# KV cache implementations
paged-kv = []           # Default paged cache
ring-kv = []            # Simple ring buffer (less memory, no fork)
offload-kv = []         # CPU/disk offloading for long contexts

# Constraint engines
json-constraint = []    # JSON/JSON Schema (lightweight)
gbnf = []               # Full GBNF grammar engine
regex-constraint = []   # Regex → DFA

# Sampling
full-samplers = []      # All sampler variants (mirostat, DRY, XTC, etc.)
minimal-samplers = []   # Just greedy + temperature + top-p

# Model formats
gguf = []               # Load GGUF models (needs gguf parser)
safetensors = []        # Load from safetensors (for weight inspection)
```

### 25.5 ABI Stability Contract

For C consumers and plugins loaded at runtime:

```rust
/// Stable C ABI for engine operations.
/// Versioned: breaking changes bump major version.
#[repr(C)]
pub struct OnnxGenaiApi {
    pub version: u32,  // ABI version (semver major)
    
    // Lifecycle
    pub engine_create: unsafe extern "C" fn(config: *const c_char) -> *mut Engine,
    pub engine_destroy: unsafe extern "C" fn(engine: *mut Engine),
    
    // Generation
    pub generate: unsafe extern "C" fn(
        engine: *mut Engine,
        request_json: *const c_char,
        callback: Option<TokenCallback>,
        user_data: *mut c_void,
    ) -> *mut c_char,  // returns JSON result (caller frees)
    
    // Session
    pub session_create: unsafe extern "C" fn(engine: *mut Engine) -> u64,
    pub session_destroy: unsafe extern "C" fn(engine: *mut Engine, session: u64),
    
    // Plugin registration
    pub register_logit_processor: unsafe extern "C" fn(
        engine: *mut Engine,
        name: *const c_char,
        process_fn: LogitProcessFn,
        user_data: *mut c_void,
    ),
    pub register_sampler: unsafe extern "C" fn(
        engine: *mut Engine,
        name: *const c_char,
        sample_fn: SampleFn,
        user_data: *mut c_void,
    ),
}

/// Token callback signature (C ABI).
pub type TokenCallback = unsafe extern "C" fn(
    token_id: u32,
    token_text: *const c_char,
    user_data: *mut c_void,
) -> bool;  // return false to cancel generation

/// Logit processor function (C ABI plugin).
pub type LogitProcessFn = unsafe extern "C" fn(
    logits: *mut f32,
    vocab_size: usize,
    context: *const ProcessorContextC,
    user_data: *mut c_void,
);

/// Sampler function (C ABI plugin).
pub type SampleFn = unsafe extern "C" fn(
    logits: *const f32,
    vocab_size: usize,
    user_data: *mut c_void,
) -> u32;
```

### 25.6 Plugin System (Dynamic Loading)

```rust
/// A dynamically loaded plugin (.so/.dylib/.dll).
pub struct Plugin {
    _lib: libloading::Library,
    metadata: PluginMetadata,
}

#[repr(C)]
pub struct PluginMetadata {
    /// Plugin name.
    pub name: *const c_char,
    /// Plugin version (semver string).
    pub version: *const c_char,
    /// Minimum engine ABI version required.
    pub min_abi_version: u32,
    /// Maximum engine ABI version supported.
    pub max_abi_version: u32,
}

/// Plugin entry point signature.
/// Called once when plugin is loaded. Plugin registers its components.
pub type PluginInitFn = unsafe extern "C" fn(api: *const OnnxGenaiApi) -> i32;

impl EngineBuilder {
    /// Load a plugin from a shared library path.
    pub fn load_plugin(mut self, path: &Path) -> Result<Self> {
        // dlopen, find "onnx_genai_plugin_init" symbol, call it
        ...
    }
}
```

### 25.7 Versioning & Compatibility Matrix

| Component | Trait | ABI-stable? | Hot-swappable? |
|---|---|---|---|
| KV Cache | `KvCacheOps` | No (Rust trait) | At build time |
| Sampler | `Sampler` | Yes (C ABI) | Runtime plugin |
| LogitProcessor | `LogitProcessor` | Yes (C ABI) | Runtime plugin |
| Scheduler | `SchedulerPolicy` | No (Rust trait) | At build time |
| ChatTemplate | `ChatTemplate` | No (Rust trait) | At build time |
| ModelLoader | `ModelLoader` | No (Rust trait) | At build time |
| Constraint | `Constraint` | Yes (via GBNF string) | Runtime (pass grammar string) |

**ABI stability rules:**
1. C ABI functions never change signature (add new functions instead)
2. Version field in API struct allows forward-compatibility checks
3. Struct layouts with `#[repr(C)]` never reorder fields
4. New features = new optional function pointers (NULL = not supported)

### 25.8 Example: Custom KV Cache Implementation

```rust
/// Example: Ring buffer KV cache for constrained memory environments.
/// Trades off: no fork, no prefix cache, but fixed memory footprint.
pub struct RingKvCache {
    ring_size: usize,  // max tokens before wrap
    heads: HashMap<SequenceId, RingHead>,
}

impl KvCacheOps for RingKvCache {
    fn append(&mut self, seq: SequenceId, num_tokens: usize) -> Result<(), KvError> {
        // Wrap around when ring is full (oldest tokens evicted)
        ...
    }
    
    fn fork(&mut self, _source: SequenceId, _position: usize) -> Result<SequenceId, KvError> {
        // Ring buffer doesn't support fork — return error
        Err(KvError::UnsupportedOperation("fork not supported by RingKvCache"))
    }
    ...
}

// Usage:
let engine = EngineBuilder::new(config)
    .with_kv_cache(RingKvCache::new(4096))  // 4K token window
    .build(model_dir)?;
```

---

## 26. Multi-Agent Serving

### 26.1 Problem

A local coding agent swarm (e.g., Flightdeck workers) means 5-20+ concurrent agents hitting one GPU. Naive sequential serving wastes compute. We need:

- Multiple agents generating simultaneously (batched forward passes)
- Agents waiting on tools don't block others
- Shared system prompt KV across agents (prefix cache)
- Memory budget so one runaway agent doesn't starve others
- Priority so interactive requests beat background work

### 26.2 Agent Lifecycle States

```
┌─────────┐    prefill     ┌───────────┐    token     ┌────────────┐
│ QUEUED  │──────────────→ │ DECODING  │────────────→ │ DECODING   │──→ ...
└─────────┘                └───────────┘              └────────────┘
                                │                           │
                         tool_call detected          finish/max_tokens
                                │                           │
                                ▼                           ▼
                        ┌──────────────┐           ┌────────────┐
                        │   PAUSED     │           │  COMPLETE  │
                        │ (waiting     │           └────────────┘
                        │  tool result)│
                        └──────────────┘
                                │
                         tool_result arrives
                                │
                                ▼
                        ┌──────────────┐
                        │  RE-QUEUED   │──→ back to DECODING (KV intact)
                        └──────────────┘
```

Key: PAUSED agents keep KV cache allocated but release their batch slot. This is the common state for coding agents (waiting on file I/O, shell commands, etc.).

### 26.3 Priority Classes

```rust
/// Priority levels for scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AgentPriority {
    /// Interactive: user-facing code completion, chat response.
    /// Preempts everything. Target: first-token < 100ms.
    Interactive = 0,
    /// Standard: normal agent generation (tool calls, code gen).
    Standard = 1,
    /// Background: indexing, summarization, speculative prefill.
    /// Gets remaining capacity. Can be fully preempted.
    Background = 2,
}

/// Per-session (per-agent) configuration.
pub struct SessionConfig {
    pub priority: AgentPriority,
    /// Maximum KV pages this session can hold.
    pub max_pages: Option<usize>,
    /// Maximum tokens per generation step.
    pub max_tokens_per_turn: u32,
    /// Timeout before evicting paused session's KV.
    pub pause_eviction_timeout: Duration,
}
```

### 26.4 Memory Budget & Eviction

```rust
/// Memory budget controller.
pub struct MemoryBudget {
    /// Total GPU pages available.
    total_pages: usize,
    /// Reserved for interactive requests (guaranteed headroom).
    interactive_reserve: usize,  // e.g., 20% of total
    /// Per-session page limits.
    session_limits: HashMap<SessionId, usize>,
    /// Eviction order when memory pressure hits.
    eviction_policy: EvictionPolicy,
}

pub enum EvictionPolicy {
    /// Evict lowest-priority sessions first, then LRU within same priority.
    PriorityThenLru,
    /// Evict largest sessions first (free most memory per eviction).
    LargestFirst,
    /// Evict sessions with best prefix cache hit potential last.
    PrefixAware,
}

impl MemoryBudget {
    /// Can this session allocate more pages?
    pub fn can_allocate(&self, session: SessionId, num_pages: usize) -> bool { ... }
    
    /// Evict sessions until `needed` pages are free.
    /// Returns evicted session IDs (their KV is offloaded to CPU or dropped).
    pub fn evict_for(&mut self, needed: usize, exclude: &[SessionId]) -> Vec<SessionId> { ... }
}
```

**Eviction tiers:**
1. Drop background sessions' KV (they can re-prefill cheaply)
2. Offload paused standard sessions to CPU (swap back on resume)
3. Preempt running standard sessions (recompute from last checkpoint)
4. Never touch interactive sessions unless OOM

### 26.5 Batched Prefill (Chunked)

When multiple agents start simultaneously (e.g., Flightdeck spawns 5 workers):

```rust
/// Chunked prefill: split long prompts into chunks, interleave with decoding.
pub struct ChunkedPrefillConfig {
    /// Maximum tokens per prefill chunk.
    pub chunk_size: usize,  // e.g., 512
    /// How many prefill chunks to run before yielding to decode.
    pub chunks_before_yield: usize,  // e.g., 2
}

// Scheduling within a single forward pass:
// [Agent A prefill chunk 512tok] + [Agent B decode 1tok] + [Agent C decode 1tok]
// Next iteration:
// [Agent A prefill chunk 512tok] + [Agent B decode 1tok] + [Agent C decode 1tok]
// ...until A's prefill is done, then A joins decode batch.
```

This prevents a new agent with a 4K prompt from blocking all existing agents for seconds.

### 26.6 Prefix Cache Sharing

Coding agents typically share:
- System prompt (instructions, tool definitions) — often 2-4K tokens
- Repository context (common files) — varies

```rust
/// Prefix cache aware of multi-agent sharing.
impl PrefixCache {
    /// Register a prefix as "shared" — never evict while any session references it.
    pub fn pin_shared_prefix(&mut self, tokens: &[TokenId]) -> PrefixId;
    
    /// Attach a session to a shared prefix (CoW fork from prefix end).
    pub fn attach_to_prefix(&mut self, prefix: PrefixId, session: SessionId) -> Result<()>;
    
    /// Stats: how many sessions share this prefix.
    pub fn prefix_refcount(&self, prefix: PrefixId) -> usize;
}

// Memory savings example:
// 10 agents × 3K token system prompt = 30K tokens of KV
// With prefix sharing: 3K tokens of KV (computed once, shared via CoW pages)
// Savings: 90% memory reduction for system prompt KV
```

### 26.7 Scheduling Algorithm

```rust
impl Scheduler {
    /// Core scheduling loop: called every iteration.
    pub fn schedule_iteration(&mut self, budget: &MemoryBudget) -> BatchPlan {
        let mut plan = BatchPlan::new();
        
        // 1. Always include interactive decode requests.
        for session in self.decoding_sessions(AgentPriority::Interactive) {
            plan.add_decode(session);
        }
        
        // 2. Include standard decode requests up to batch limit.
        let remaining = self.max_batch_tokens - plan.total_tokens();
        for session in self.decoding_sessions(AgentPriority::Standard) {
            if plan.total_tokens() + 1 > self.max_batch_tokens { break; }
            plan.add_decode(session);
        }
        
        // 3. Interleave prefill chunks if capacity remains.
        let prefill_budget = remaining.min(self.chunked_prefill.chunk_size);
        if let Some(prefilling) = self.next_prefill_session() {
            plan.add_prefill_chunk(prefilling, prefill_budget);
        }
        
        // 4. Background decode gets whatever's left.
        for session in self.decoding_sessions(AgentPriority::Background) {
            if plan.total_tokens() >= self.max_batch_tokens { break; }
            plan.add_decode(session);
        }
        
        plan
    }
}

pub struct BatchPlan {
    /// Decode steps: each contributes 1 token to the batch.
    pub decode: Vec<SessionId>,
    /// Prefill chunk: contributes N tokens from one session.
    pub prefill: Option<(SessionId, usize)>,
    /// Total tokens in this forward pass.
    pub total_tokens: usize,
}
```

### 26.8 HTTP API for Multi-Agent

```json
POST /v1/chat/completions
{
  "model": "local-model",
  "messages": [...],
  "session_id": "agent-worker-3",
  "x-priority": "standard",
  "x-max-pages": 256
}
```

Or via header:
```
X-Session-Id: agent-worker-3
X-Priority: interactive
```

### 26.9 Observability

```rust
/// Runtime metrics exposed via /v1/status endpoint.
#[derive(Serialize)]
pub struct ServerStatus {
    pub active_sessions: usize,
    pub decoding_sessions: usize,
    pub paused_sessions: usize,
    pub queued_requests: usize,
    pub gpu_pages_used: usize,
    pub gpu_pages_total: usize,
    pub gpu_pages_shared: usize,   // prefix cache shared pages
    pub batch_utilization: f32,     // avg tokens per forward / max batch tokens
    pub per_session: Vec<SessionStatus>,
}

pub struct SessionStatus {
    pub session_id: String,
    pub state: SessionState,       // decoding / paused / queued
    pub priority: AgentPriority,
    pub kv_pages: usize,
    pub tokens_generated: usize,
    pub time_in_queue_ms: u64,
}
```

### 26.10 Configuration

```yaml
# server_config.yaml
serving:
  max_batch_tokens: 4096       # max tokens per forward pass
  max_concurrent_sessions: 32  # hard limit on active sessions
  
  memory:
    total_gpu_pages: 2048
    interactive_reserve_pct: 20
    eviction_policy: priority_then_lru
    offload_to_cpu: true        # swap evicted KV to CPU RAM
    
  prefill:
    chunk_size: 512
    chunks_before_yield: 2
    
  priorities:
    interactive:
      max_queue_time_ms: 50
      preempt_others: true
    standard:
      max_queue_time_ms: 5000
      pause_eviction_timeout_s: 300  # 5 min idle before KV eviction
    background:
      max_queue_time_ms: 30000
      max_pages_per_session: 128     # limit background memory usage
      
  prefix_sharing:
    enabled: true
    min_refcount_to_pin: 2     # pin after 2+ sessions use same prefix
```

---

## 27. Multi-Token Speculative Decoding (MTS)

### 27.1 Taxonomy of Speculative Methods

| Method | Draft Source | Models | Verification | Example |
|---|---|---|---|---|
| **Draft model** (current) | Separate smaller model | 2 | Target verifies draft tokens | Llama-70B + Llama-8B |
| **Self-speculative** | Same model, early exit | 1 | Later layers verify early-exit tokens | LayerSkip |
| **Multi-token prediction (MTP)** | Extra prediction heads on target | 1 | Main head verifies auxiliary heads | DeepSeek-V3, Meta MTP |
| **Medusa** | Fine-tuned extra heads | 1 | Tree-structured verification | Medusa-2 |
| **EAGLE** | Feature-level draft with autoregression | 1+adapter | Target verifies feature-based proposals | EAGLE-2 |
| **N-gram / Prompt lookup** | Input token patterns | 0 | Target verifies repeated patterns | Already supported |
| **Lookahead** | Jacobi iteration on target | 1 | N-gram cache from Jacobi trajectories | Lookahead Decoding |

### 27.2 Multi-Token Prediction (MTP) Design

MTP models (DeepSeek-V3, Meta's multi-token) have additional output heads that predict future tokens:

```
Input: [t1, t2, t3, t4]

Normal head (position +1):  predicts t5    ← always correct (autoregressive)
MTP head 1 (position +2):   predicts t6    ← speculative
MTP head 2 (position +3):   predicts t7    ← speculative
MTP head 3 (position +4):   predicts t8    ← speculative
```

One forward pass → 4 token candidates. Verify MTP predictions on next step.

```rust
/// Multi-token prediction model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MtpConfig {
    /// Number of extra prediction heads (k).
    /// Model outputs k+1 token predictions per forward pass.
    pub num_speculative_heads: usize,
    /// Output names for each head in the ONNX model.
    /// Index 0 = main head (always trusted), 1..k = speculative heads.
    pub head_output_names: Vec<String>,
    /// Acceptance rule for speculative heads.
    pub acceptance: MtpAcceptance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MtpAcceptance {
    /// Accept if MTP head's argmax matches target's next-step argmax.
    Greedy,
    /// Stochastic acceptance (for sampling): accept with P(draft)/P(target) probability.
    Stochastic,
    /// Threshold: accept if top-1 probability from MTP head > threshold.
    Confidence { min_prob: f32 },
}
```

### 27.3 Medusa-Style Tree Verification

Medusa uses multiple heads but verifies them in a **tree structure** — not just sequential:

```
Step 1: Main model generates token A (head 0)
        Medusa head 1 proposes: [B1, B2, B3]  (top-3 for position +2)
        Medusa head 2 proposes: [C1, C2]       (top-2 for position +3)

Step 2: Verify tree of candidates in ONE forward pass:
        [A→B1→C1, A→B1→C2, A→B2→C1, A→B2→C2, A→B3→C1, A→B3→C2]
        
        Using tree attention mask — all candidates verified simultaneously.
```

```rust
/// Medusa tree-structured speculation.
pub struct MedusaConfig {
    /// Number of Medusa heads.
    pub num_heads: usize,
    /// Top-k candidates per head.
    pub top_k_per_head: Vec<usize>,  // e.g., [3, 2, 2] → 3×2×2 = 12 tree paths
    /// Tree attention: which positions attend to which.
    pub tree_structure: TreeStructure,
    /// Maximum tree candidates per verification step.
    pub max_candidates: usize,
}

#[derive(Debug, Clone)]
pub struct TreeStructure {
    /// Parent index for each node in the verification batch.
    /// Root has parent = -1.
    pub parent_indices: Vec<i32>,
    /// Depth of each node.
    pub depths: Vec<usize>,
}

impl MedusaConfig {
    /// Build the tree attention mask for verification.
    pub fn build_tree_attention_mask(&self) -> Vec<Vec<bool>> {
        // Each candidate attends to its ancestors in the tree.
        // Enables single-pass verification of all candidates.
        ...
    }
}
```

### 27.4 EAGLE (Feature-Level Speculation)

EAGLE doesn't predict tokens directly — it predicts **hidden states** then uses the target's LM head:

```
Target model: input → hidden_states → LM_head → token
EAGLE adapter: prev_hidden + prev_token_embed → draft_hidden → LM_head → draft_token
```

```rust
/// EAGLE speculation config.
pub struct EagleConfig {
    /// Path to EAGLE adapter model (lightweight autoregressive on features).
    pub adapter_model: String,
    /// Number of draft tokens to propose per step.
    pub draft_tokens: usize,
    /// Whether to use tree-structured verification (EAGLE-2).
    pub tree_verification: bool,
    /// Feature layer to tap from target model.
    pub feature_layer: i32,  // -1 = last hidden state
}
```

### 27.5 Self-Speculative (Early Exit)

Use the same model's early layers as draft:

```rust
/// Self-speculative: early layers as draft, full model verifies.
pub struct SelfSpecConfig {
    /// Exit at this layer for draft tokens.
    pub draft_exit_layer: usize,   // e.g., layer 8 of 32
    /// Number of draft tokens before full verification.
    pub draft_tokens: usize,
    /// Confidence threshold for early exit.
    pub exit_confidence: f32,
}
```

Requires model to expose intermediate layer outputs (ONNX: add early-exit output node at layer N).

### 27.6 Unified Speculation Interface

All methods share the same contract:

```rust
/// Unified interface for all speculative methods.
pub trait SpeculativeProposer: Send + Sync {
    /// Propose candidate tokens for speculation.
    /// Returns one or more candidate sequences (tree = multiple paths).
    fn propose(
        &mut self,
        context: &SpeculativeContext,
    ) -> Result<Vec<CandidateSequence>>;
    
    /// Update internal state after verification results.
    fn update(&mut self, accepted: &VerificationResult);
    
    /// Name for metrics/logging.
    fn name(&self) -> &str;
    
    /// Expected speedup ratio (for scheduling decisions).
    fn expected_speedup(&self) -> f32;
}

pub struct SpeculativeContext {
    /// Tokens generated so far.
    pub tokens: Vec<TokenId>,
    /// Last hidden state from target model (for EAGLE).
    pub last_hidden_state: Option<Value>,
    /// Sampling config (needed for stochastic acceptance).
    pub sampling: SamplingConfig,
}

pub struct CandidateSequence {
    /// Proposed token IDs.
    pub tokens: Vec<TokenId>,
    /// Confidence/probability for each token.
    pub probs: Vec<f32>,
    /// Tree parent index (-1 = root). Sequential = [−1, 0, 1, 2, ...].
    pub parent_indices: Vec<i32>,
}

pub struct VerificationResult {
    /// How many tokens from each candidate path were accepted.
    pub accepted_lengths: Vec<usize>,
    /// The bonus token from the verifier (token at first mismatch position).
    pub bonus_token: TokenId,
}
```

### 27.7 Verification Engine

Single verification engine handles all proposer types:

```rust
pub struct SpeculativeEngine {
    proposer: Box<dyn SpeculativeProposer>,
    verifier: VerifierSession,  // the target model
    config: SpeculativeEngineConfig,
}

pub struct SpeculativeEngineConfig {
    /// Max draft tokens before verification.
    pub max_draft_tokens: usize,
    /// Acceptance rule.
    pub acceptance: AcceptanceRule,
    /// Adaptive: disable speculation when acceptance rate drops below threshold.
    pub min_acceptance_rate: f32,
    /// Window for acceptance rate tracking.
    pub acceptance_window: usize,
}

#[derive(Debug, Clone)]
pub enum AcceptanceRule {
    /// Accept iff argmax(target) == draft_token.
    Greedy,
    /// Accept with probability min(1, P_target/P_draft). Preserves distribution.
    Stochastic,
    /// Accept if target top-1 prob for this token > threshold.
    Threshold(f32),
    /// Typical acceptance: accept if token is within typical set.
    Typical { tau: f32 },
}

impl SpeculativeEngine {
    /// Run one speculative step: propose → verify → accept.
    pub fn step(&mut self, context: &mut GenerationContext) -> Result<Vec<TokenId>> {
        // 1. Get proposals
        let candidates = self.proposer.propose(&context.spec_context())?;
        
        // 2. Build verification batch
        //    - Sequential: just concat [accepted_so_far + draft_tokens]
        //    - Tree: build tree attention mask, run all paths in one forward
        let verify_batch = self.build_verify_batch(&candidates);
        
        // 3. One target forward pass (verifies all candidates)
        let target_logits = self.verifier.forward(&verify_batch)?;
        
        // 4. Accept/reject using configured rule
        let result = self.verify(&candidates, &target_logits);
        
        // 5. Update proposer state
        self.proposer.update(&result);
        
        // 6. Return accepted tokens + bonus
        Ok(result.accepted_tokens())
    }
}
```

### 27.8 Tree Attention for Verification

Tree verification (Medusa/EAGLE-2) verifies multiple candidate paths in ONE forward pass using a custom attention mask:

```rust
/// Build tree attention mask for verification.
/// 
/// Example: 3 candidates from position 5: [A, B, C]
/// Candidate A continues: [A→D, A→E]
///
/// Attention mask (1 = can attend):
///     pos: 0  1  2  3  4  A  B  C  AD AE
/// A:       [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
/// B:       [1, 1, 1, 1, 1, 0, 1, 0, 0, 0]
/// C:       [1, 1, 1, 1, 1, 0, 0, 1, 0, 0]
/// AD:      [1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
/// AE:      [1, 1, 1, 1, 1, 1, 0, 0, 0, 1]
fn build_tree_attention_mask(
    context_len: usize,
    candidates: &[CandidateSequence],
) -> Vec<Vec<bool>> {
    // Each candidate attends to:
    // 1. All context tokens (0..context_len)
    // 2. Its own ancestor chain in the tree
    ...
}
```

### 27.9 Adaptive Speculation

Not all sequences benefit from speculation equally. Adapt at runtime:

```rust
pub struct AdaptiveSpeculation {
    /// Rolling acceptance rate.
    acceptance_history: VecDeque<bool>,
    /// Current draft length (adapts based on acceptance).
    current_draft_len: usize,
    /// Min/max bounds.
    min_draft: usize,  // 1
    max_draft: usize,  // e.g., 8
}

impl AdaptiveSpeculation {
    /// Adjust draft length based on recent acceptance rate.
    pub fn adapt(&mut self) {
        let rate = self.acceptance_rate();
        if rate > 0.8 {
            // High acceptance → try more draft tokens
            self.current_draft_len = (self.current_draft_len + 1).min(self.max_draft);
        } else if rate < 0.4 {
            // Low acceptance → reduce draft (or disable speculation)
            self.current_draft_len = (self.current_draft_len - 1).max(self.min_draft);
        }
    }
    
    /// Completely disable speculation if it's not helping.
    pub fn should_disable(&self) -> bool {
        // If acceptance rate < 30% over last 50 tokens, speculation adds overhead
        self.acceptance_rate() < 0.3 && self.acceptance_history.len() >= 50
    }
}
```

### 27.10 ONNX Model Requirements

For each speculation method, the ONNX model needs specific outputs:

```yaml
# inference_metadata.yaml
speculative:
  method: mtp
  mtp:
    num_heads: 3
    head_outputs: ["logits", "logits_head_1", "logits_head_2", "logits_head_3"]
    
# Or Medusa:
speculative:
  method: medusa
  medusa:
    num_heads: 3
    top_k: [3, 2, 2]
    head_outputs: ["medusa_head_0", "medusa_head_1", "medusa_head_2"]
    
# Or EAGLE:
speculative:
  method: eagle
  eagle:
    adapter_model: "eagle_adapter.onnx"
    feature_output: "hidden_states"  # target model must expose this
    draft_tokens: 5

# Or self-speculative:
speculative:
  method: self_speculative
  self_speculative:
    draft_exit_layer: 8
    exit_output: "layer_8_logits"  # model has early-exit output node
    
# Or draft model (existing):
speculative:
  method: draft_model
  draft:
    model: "draft_model.onnx"
    tokens_per_step: 5
```

### 27.11 Performance Characteristics

| Method | Speedup | Extra Memory | Model Changes Needed |
|---|---|---|---|
| Draft model | 2-3× | Full draft model | None (two models) |
| MTP | 2-3× | ~5% (extra head weights) | Export with MTP heads |
| Medusa | 2-3× | ~10% (trained heads) | Fine-tune + export heads |
| EAGLE | 2.5-4× | ~15% (adapter) | Train adapter |
| Self-speculative | 1.5-2× | None | Model with early-exit output |
| N-gram | 1.2-2× | None | None |
| Lookahead | 1.5-2.5× | None | None |

For coding agent: EAGLE or MTP gives best bang-for-buck. Code is highly predictable → high acceptance rates → larger speedups.

---

## 28. Speculator Model Compatibility (vLLM Speculators Integration)

### 28.1 Motivation

The vLLM `speculators` library (github.com/vllm-project/speculators) is becoming the standard for training and publishing speculative decoding models. RedHat has published EAGLE-3, P-EAGLE, and DFlash speculators for Qwen3, Llama, Gemma 4 on HuggingFace. We should be able to load and run these directly.

### 28.2 Auto-Discovery from HuggingFace Config

Speculators publish a `config.json` with a `speculator_config` field:

```json
{
  "architectures": ["EagleModel"],
  "speculator_config": {
    "proposal_type": "eagle3",
    "num_speculative_tokens": 4,
    "verifier": {
      "name_or_path": "Qwen/Qwen3-8B",
      "architectures": ["Qwen3ForCausalLM"]
    }
  }
}
```

Our loader detects this automatically:

```rust
/// Check if a model directory contains a speculator config.
pub fn detect_speculator(model_dir: &Path) -> Option<SpeculatorDetection> {
    let config_path = model_dir.join("config.json");
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(config_path).ok()?).ok()?;
    
    let spec_config = config.get("speculator_config")?;
    let proposal_type = spec_config.get("proposal_type")?.as_str()?;
    
    Some(SpeculatorDetection {
        proposal_type: proposal_type.to_string(),
        num_tokens: spec_config.get("num_speculative_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as usize,
        verifier_path: spec_config.get("verifier")
            .and_then(|v| v.get("name_or_path"))
            .and_then(|v| v.as_str())
            .map(String::from),
    })
}

pub struct SpeculatorDetection {
    pub proposal_type: String,    // "eagle3", "peagle", "dflash", "mtp"
    pub num_tokens: usize,
    pub verifier_path: Option<String>,
}
```

### 28.3 Draft Vocabulary Mapping

Speculators often train draft models with a **reduced vocabulary** for speed. A 128K vocab target might have a 32K vocab draft. Need bidirectional mapping:

```rust
/// Maps between draft model's reduced vocabulary and target's full vocabulary.
pub struct VocabMapping {
    /// Target token ID → Draft token ID (or None if not in draft vocab).
    /// Shape: [verifier_vocab_size]. True = token exists in draft vocab.
    target_to_draft_mask: Vec<bool>,
    /// Draft token ID → Target token ID.
    /// Shape: [draft_vocab_size].
    draft_to_target: Vec<u32>,
    /// Draft vocab size.
    pub draft_vocab_size: usize,
    /// Target vocab size.
    pub target_vocab_size: usize,
}

impl VocabMapping {
    /// Load from speculators-format files (t2d.pt, d2t.pt as safetensors/npy).
    pub fn load(model_dir: &Path) -> Result<Option<Self>> {
        let t2d_path = model_dir.join("t2d.safetensors");
        let d2t_path = model_dir.join("d2t.safetensors");
        if !t2d_path.exists() || !d2t_path.exists() {
            return Ok(None);  // Same vocab, no mapping needed
        }
        // Load tensors...
        ...
    }
    
    /// Convert target token IDs to draft token IDs for draft model input.
    pub fn to_draft(&self, target_ids: &[u32]) -> Vec<u32> {
        target_ids.iter().map(|&tid| {
            // Binary search or lookup in t2d mapping
            self.target_to_draft_index(tid).unwrap_or(0) // UNK if not in draft vocab
        }).collect()
    }
    
    /// Convert draft model output logits back to target vocab space.
    /// Unmapped positions get -inf.
    pub fn draft_logits_to_target(&self, draft_logits: &[f32]) -> Vec<f32> {
        let mut target_logits = vec![f32::NEG_INFINITY; self.target_vocab_size];
        for (draft_idx, &logit) in draft_logits.iter().enumerate() {
            let target_idx = self.draft_to_target[draft_idx] as usize;
            target_logits[target_idx] = logit;
        }
        target_logits
    }
}
```

### 28.4 Multi-Layer Feature Extraction (DFlash)

DFlash uses hidden states from **multiple layers** of the verifier as "anchors":

```rust
/// DFlash configuration.
pub struct DFlashConfig {
    /// Which verifier layers to extract hidden states from.
    pub anchor_layers: Vec<usize>,  // e.g., [8, 16, 24] for a 32-layer model
    /// Block size for anchored drafting.
    pub block_size: usize,
    /// Maximum number of anchor blocks per draft step.
    pub max_anchors: usize,
}

/// The ONNX model for DFlash needs intermediate layer outputs.
/// In inference_metadata.yaml:
///
/// ```yaml
/// speculative:
///   method: dflash
///   dflash:
///     anchor_layers: [8, 16, 24]
///     anchor_outputs: ["hidden_state_8", "hidden_state_16", "hidden_state_24"]
///     block_size: 4
/// ```
///
/// The verifier ONNX model must be exported with these intermediate outputs.
```

### 28.5 Sliding Window for Draft Model KV

Draft models don't need full-context attention — recent tokens are most predictive:

```rust
/// Draft model KV cache with sliding window.
pub struct DraftKvConfig {
    /// Window size (tokens). Draft only attends to last N tokens.
    pub sliding_window: usize,     // e.g., 512
    /// Layers that use full attention (indices). Rest use sliding window.
    pub full_attention_layers: Vec<usize>,  // e.g., [0] (first layer only)
}

// Memory savings:
// Target: 32 layers × 4K context × 128 dim = 16M params of KV
// Draft (window=512): 8 layers × 512 context × 64 dim = 256K params of KV
// → Draft KV is ~1.5% of target KV
```

### 28.6 P-EAGLE: Parallel Multi-Token Prediction

P-EAGLE extends EAGLE-3 with **parallel** token prediction via COD (Conditional-On-Distribution) sampling — predicts multiple draft tokens in a single forward pass rather than sequentially:

```rust
/// P-EAGLE: one forward → multiple draft tokens (not sequential).
pub struct PEagleConfig {
    /// Number of parallel draft tokens per forward pass.
    pub parallel_tokens: usize,
    /// COD (Conditional-On-Distribution) sampling for parallel independence.
    pub cod_sampling: bool,
    /// Base EAGLE config.
    pub eagle: EagleConfig,
}

/// COD Sampling: each draft position conditions on the probability distribution
/// of the previous position rather than a single sampled token.
/// This allows parallel prediction without sequential dependency.
///
/// Standard EAGLE: token_1 → token_2 → token_3 (sequential, 3 forwards)
/// P-EAGLE:        [token_1, token_2, token_3] (parallel, 1 forward)
///
/// Trade-off: slightly lower acceptance rate, but 3× fewer draft forwards.
```

### 28.7 Unified Loading (Speculators-Compatible)

```rust
impl Engine {
    /// Load a model with automatic speculator detection.
    pub fn from_dir_with_speculation(
        verifier_dir: &Path,
        speculator_dir: Option<&Path>,
        config: EngineConfig,
    ) -> Result<Self> {
        let mut engine = Self::from_dir(verifier_dir, config)?;
        
        // Auto-detect speculator
        let spec_dir = speculator_dir.or_else(|| {
            // Check if verifier has embedded speculator config
            detect_speculator(verifier_dir).map(|_| verifier_dir)
        });
        
        if let Some(dir) = spec_dir {
            let detection = detect_speculator(dir)
                .ok_or_else(|| anyhow!("No speculator config found"))?;
            
            let proposer: Box<dyn SpeculativeProposer> = match detection.proposal_type.as_str() {
                "eagle3" | "eagle" => Box::new(EagleProposer::load(dir, &engine)?),
                "peagle" => Box::new(PEagleProposer::load(dir, &engine)?),
                "dflash" => Box::new(DFlashProposer::load(dir, &engine)?),
                "mtp" => Box::new(MtpProposer::load(dir, &engine)?),
                "ngram" => Box::new(NgramProposer::new(detection.num_tokens)),
                other => anyhow::bail!("Unsupported speculator type: {}", other),
            };
            
            // Load vocab mapping if present
            let vocab_mapping = VocabMapping::load(dir)?;
            
            engine.enable_speculation(proposer, vocab_mapping);
        }
        
        Ok(engine)
    }
}
```

### 28.8 CLI Integration

```bash
# Serve verifier + speculator (auto-detected from HF format)
onnx-genai serve ./qwen3-8b --speculator ./qwen3-8b-speculator-eagle3

# Or if speculator is embedded in model dir:
onnx-genai serve ./qwen3-8b-with-speculator

# Benchmark speculation effectiveness:
onnx-genai bench ./model --speculator ./speculator --dataset ./prompts.jsonl
# Output: acceptance_rate, tokens/s, speedup_vs_baseline
```

### 28.9 Model Conversion (Out of Scope)

This repo does NOT do model conversion. ONNX export is handled by **Mobius** or upstream tooling. We only consume pre-built ONNX models.

### 28.10 Mobius Integration (ONNX Model Building)

The conversion path from trained speculators to ONNX is handled by **Mobius** (Microsoft's ONNX model builder for GenAI). Mobius already supports building speculator models (e.g., Gemma 4 DFlash).

**Actual pipeline:**
```
Speculators training (PyTorch) 
    → HF checkpoint (safetensors + config.json)
    → Mobius build (handles export, graph optimization, opset 24)
    → ONNX speculator model + inference_metadata.yaml
    → onnx-genai engine loads directly
```

**We do NOT need a separate `convert-speculator` script.** Mobius IS the converter. Our responsibility is purely on the consumption side:

1. **Load** the ONNX model Mobius produces
2. **Detect** speculator type from config/metadata
3. **Route** to the correct `SpeculativeProposer` implementation
4. **Handle** vocab mapping if Mobius includes t2d/d2t tensors
5. **Bind** multi-layer outputs (DFlash anchor hidden states are already graph outputs)

This means our `detect_speculator()` should also check for Mobius-style metadata in addition to HF `speculator_config`:

```rust
/// Check for speculator config in either:
/// 1. HF-style: config.json → speculator_config.proposal_type
/// 2. Mobius-style: inference_metadata.yaml → speculative.method
pub fn detect_speculator(model_dir: &Path) -> Option<SpeculatorDetection> {
    // Try our native metadata first
    if let Some(meta) = try_load_inference_metadata(model_dir) {
        if let Some(spec) = &meta.speculative {
            return Some(SpeculatorDetection::from_metadata(spec));
        }
    }
    // Fall back to HF config.json
    try_detect_from_hf_config(model_dir)
}
```

**Key advantage:** Mobius handles all the hard ONNX export problems (graph optimization, operator fusion, opset compatibility, quantization). We focus purely on fast inference of the resulting model.

---

## 29. Language Diffusion Models

### 29.1 What Is Language Diffusion

Language diffusion generates text by **iteratively denoising a sequence of masked/corrupted tokens**, rather than appending one token at a time. All positions are generated in parallel and refined over multiple steps.

```
Autoregressive (GPT-style):
  Step 0: [The]
  Step 1: [The, cat]
  Step 2: [The, cat, sat]
  Step 3: [The, cat, sat, down]
  → 4 steps for 4 tokens, strictly left-to-right

Language Diffusion (MDLM/Mercury-style):
  Step 0: [MASK, MASK, MASK, MASK]          ← fully masked
  Step 1: [The,  MASK, sat,  MASK]          ← high-confidence positions unmasked
  Step 2: [The,  cat,  sat,  MASK]          ← more positions resolved
  Step 3: [The,  cat,  sat,  down]          ← all resolved
  → 3 steps for 4 tokens, any-order, parallel
```

### 29.2 Key Models

| Model | Scale | Method | Key Innovation |
|---|---|---|---|
| **MDLM** | Research | Masked discrete diffusion | Continuous-time masked diffusion with score entropy loss |
| **SEDD** | Research | Score entropy discrete diffusion | Concrete score matching for discrete data |
| **Mercury** | LLaMA-scale (up to 10B+) | Masked diffusion | First production-scale language diffusion, 10× faster than AR |
| **Plaid** | 1B+ | Discrete diffusion | Efficient training + adaptive step scheduling |
| **DART** | Research | Non-autoregressive | Any-order autoregressive with diffusion |
| **Dream** | Research | Discrete denoising | Reparameterized discrete diffusion |
| **LLaDA** | 8B | Large language diffusion with adaptation | Competitive with LLaMA-3 on benchmarks |

### 29.3 How It Differs from Image Diffusion

| Aspect | Image Diffusion | Language Diffusion |
|---|---|---|
| **Space** | Continuous (float pixels/latents) | Discrete (token IDs from finite vocab) |
| **Corruption** | Gaussian noise addition | Token masking or uniform corruption |
| **State** | Noisy float tensor | Partially-masked token sequence |
| **Per-step output** | Denoised float tensor | Logits per position over vocab |
| **Schedule** | Noise schedule (β_t) | Masking schedule (what % masked at step t) |
| **Unmasking** | Deterministic denoise | Confidence-based: unmask positions where model is most certain |
| **KV cache** | N/A | N/A (no causal attention, full bidirectional) |

### 29.4 Pipeline Strategy: `discrete_diffusion`

Extends our strategy taxonomy (§20) with a fourth fundamental type:

```yaml
strategy:
  kind: discrete_diffusion
  model: denoiser
  
  # Masking schedule
  schedule:
    type: cosine | linear | sigmoid | adaptive
    total_steps: 64          # max denoising steps
    
  # Unmasking policy
  unmasking:
    policy: confidence | random | entropy | hybrid
    # confidence: unmask highest-probability positions first
    # random: unmask random subset each step
    # entropy: unmask lowest-entropy positions first
    # hybrid: confidence with random tiebreaking
    
    tokens_per_step: adaptive  # or fixed number
    min_confidence: 0.9        # only unmask if P(token) > threshold
    
  # Generation shape
  output:
    length: fixed | variable
    max_length: 2048
    # fixed: generate exactly max_length tokens (pad/truncate)
    # variable: stop when all positions unmasked
```

### 29.5 Core Engine Design

```rust
/// Discrete diffusion generation engine.
pub struct DiscreteDiffusionEngine {
    /// The denoiser model (bidirectional transformer, NOT causal).
    session: Session,
    /// Tokenizer.
    tokenizer: Tokenizer,
    /// Masking schedule.
    schedule: MaskSchedule,
    /// Unmasking policy.
    unmasking: UnmaskingPolicy,
    /// Mask token ID.
    mask_token_id: TokenId,
}

/// The state of a diffusion generation: a partially-masked sequence.
pub struct DiffusionState {
    /// Current token IDs. Masked positions hold mask_token_id.
    tokens: Vec<TokenId>,
    /// Which positions are still masked (true = masked).
    mask: Vec<bool>,
    /// Current diffusion timestep (decreasing: T → 0).
    timestep: usize,
    /// Confidence scores from last forward pass (per position).
    confidences: Vec<f32>,
}

impl DiscreteDiffusionEngine {
    /// Generate text via iterative denoising.
    pub fn generate(&self, request: DiffusionRequest) -> Result<DiffusionResult> {
        // 1. Initialize: all positions masked (or partially given for infilling)
        let mut state = self.initialize_state(&request);
        
        // 2. Iterative denoising loop
        for step in (0..self.schedule.total_steps).rev() {
            // Compute masking ratio for this timestep
            let target_unmask_ratio = self.schedule.ratio_at(step);
            
            // Forward pass: get logits for all positions
            // (bidirectional attention — every position sees every other position)
            let logits = self.forward(&state, step)?;
            
            // Decide which positions to unmask this step
            let to_unmask = self.unmasking.select_positions(
                &state,
                &logits,
                target_unmask_ratio,
            );
            
            // Sample tokens for unmasked positions
            for pos in to_unmask {
                let token = self.sample_position(&logits[pos], &request.sampling)?;
                state.tokens[pos] = token;
                state.mask[pos] = false;
            }
            
            // Optional: re-mask low-confidence positions (allows correction)
            if self.schedule.allows_remask(step) {
                self.remask_low_confidence(&mut state, &logits);
            }
            
            // Stream intermediate result if requested
            if let Some(cb) = &request.callback {
                cb(DiffusionStep {
                    step,
                    tokens: state.tokens.clone(),
                    mask: state.mask.clone(),
                    unmasked_ratio: state.unmasked_ratio(),
                })?;
            }
            
            // Early stop if all positions are unmasked
            if state.all_unmasked() {
                break;
            }
        }
        
        Ok(DiffusionResult {
            text: self.tokenizer.decode(&state.tokens)?,
            token_ids: state.tokens,
            steps_used: self.schedule.total_steps - state.timestep,
        })
    }
}
```

### 29.6 Masking Schedules

```rust
pub enum MaskSchedule {
    /// Linear: unmask uniformly across steps.
    Linear { total_steps: usize },
    /// Cosine: slow start, fast middle, slow end. Most common.
    Cosine { total_steps: usize },
    /// Sigmoid: sharp transition in the middle.
    Sigmoid { total_steps: usize },
    /// Adaptive: adjust steps based on sequence difficulty.
    Adaptive {
        max_steps: usize,
        /// Stop early if all positions have confidence > threshold.
        early_stop_confidence: f32,
    },
}

impl MaskSchedule {
    /// What fraction of tokens should be unmasked at timestep t.
    /// t goes from T (fully masked) to 0 (fully unmasked).
    pub fn ratio_at(&self, t: usize) -> f32 {
        match self {
            MaskSchedule::Linear { total_steps } => {
                1.0 - (t as f32 / *total_steps as f32)
            }
            MaskSchedule::Cosine { total_steps } => {
                let s = t as f32 / *total_steps as f32;
                1.0 - (s * std::f32::consts::FRAC_PI_2).cos()
            }
            ...
        }
    }
    
    /// Whether re-masking (correcting earlier decisions) is allowed at step t.
    pub fn allows_remask(&self, t: usize) -> bool {
        // Typically allow remask in early steps (high t), freeze in later steps
        t > self.total_steps() / 3
    }
}
```

### 29.7 Why Language Diffusion Matters for Coding Agents

**1. Native infilling without FIM tokens:**
```rust
// Autoregressive: needs special FIM tokens, model must be trained for it
// "<|fim_prefix|>def fib(n):<|fim_suffix|>\n    return result<|fim_middle|>"

// Language diffusion: just mask the positions you want filled
// "def fib(n):\n    [MASK MASK MASK MASK MASK]\n    return result"
// → The model fills in the masked positions directly.
```

**2. Parallel multi-position editing:**
```
// Need to change variable name from 'x' to 'count' in 5 places:
// AR: regenerate entire file or do 5 separate edits
// Diffusion: mask all 5 positions, denoise in parallel → all updated consistently
```

**3. Speed for long generations:**
```
// 512 tokens:
// AR: 512 forward passes (sequential)
// Diffusion: ~32-64 forward passes (parallel across all positions)
// With adaptive scheduling: even fewer if content is predictable
```

**4. Controllable generation length:**
```
// Know you need exactly 200 tokens of code? 
// Initialize 200 masked positions → denoise.
// No need for length-predicting heuristics.
```

### 29.8 Conditional Diffusion (Guided Generation)

```rust
/// Conditioning modes for language diffusion.
pub enum DiffusionCondition {
    /// Unconditional: generate from scratch.
    Unconditional { length: usize },
    
    /// Prefix-conditioned: given prefix, generate continuation.
    /// (Prefix positions are never masked.)
    Prefix { prefix: Vec<TokenId>, generate_length: usize },
    
    /// Infilling: given prefix + suffix, fill the middle.
    Infill {
        prefix: Vec<TokenId>,
        suffix: Vec<TokenId>,
        fill_length: usize,  // or estimate
    },
    
    /// Span corruption: multiple spans to fill (T5-style).
    SpanFill {
        tokens: Vec<TokenId>,
        /// (start, end) indices of spans to regenerate.
        spans: Vec<(usize, usize)>,
    },
    
    /// Editing: given full text, re-generate specific positions.
    Edit {
        original: Vec<TokenId>,
        /// Positions to re-generate (mask these, keep rest).
        edit_positions: Vec<usize>,
    },
}
```

### 29.9 Differences from Existing Strategies

| Aspect | Autoregressive | Image Diffusion (iterative) | Language Diffusion (discrete_diffusion) |
|---|---|---|---|
| KV Cache | Yes (paged) | No | No |
| Attention | Causal (lower triangular) | Full bidirectional | Full bidirectional |
| Continuous batching | Yes (different lengths) | Yes (same shape) | Yes (same shape within step) |
| Speculative decoding | Yes (draft model) | No | Possible (adaptive step skipping) |
| Streaming | Token-by-token | Step-by-step preview | Step-by-step (increasing resolution) |
| Position flexibility | Left-to-right only | All positions | Any subset of positions |
| Memory per step | O(1) new + cached KV | O(n) full state | O(n) full state |

### 29.10 Memory & Batching

**No KV cache needed** — every step is a full bidirectional forward pass. This simplifies memory management but means each step is a full O(n²) attention:

```rust
// Memory comparison for 2048-token generation:
// AR: KV cache grows each step, peak = full sequence KV
// Diffusion: constant memory per step (full 2048×2048 attention each time)
//
// AR total compute: Σ(i=1..2048) O(i) = O(n²) across all steps
// Diffusion total compute: 64 × O(2048²) = O(64 × n²)
// → Diffusion does more FLOPs but steps are parallelizable on GPU
```

**Batching:** Multiple diffusion requests at the same step can be batched naturally (all same shape). Requests at different steps can still batch if padded.

```rust
impl Scheduler {
    /// Schedule diffusion requests — simpler than AR since no KV state.
    fn schedule_diffusion(&self, requests: &[DiffusionRequest]) -> DiffusionBatch {
        // Group by sequence length (or pad to max)
        // All positions processed in one forward pass per step
        // No KV cache allocation/management needed
        ...
    }
}
```

### 29.11 Integration with Tool Use

Language diffusion + tool use works differently from AR:

```
// AR tool use: generate until tool_call token → pause → resume from that point
// Diffusion tool use: 
//   1. Generate full response in masked form
//   2. Detect tool_call pattern in partially-unmasked output
//   3. Unmask tool_call JSON first (high priority positions)
//   4. Execute tool
//   5. Insert tool result as fixed (unmasked) tokens
//   6. Re-run denoising on remaining masked positions with tool result as context
```

```rust
/// Tool-aware diffusion: detect and prioritize tool call positions.
pub struct ToolAwareDiffusion {
    /// Positions identified as likely tool call JSON.
    tool_call_positions: Option<Range<usize>>,
    /// After tool execution, these positions are fixed context.
    tool_result_positions: Option<Range<usize>>,
}
```

### 29.12 Metadata Schema

```yaml
# inference_metadata.yaml for a language diffusion model
model:
  type: discrete_diffusion
  attention: bidirectional    # NOT causal
  mask_token_id: 128256      # special [MASK] token
  
pipeline:
  strategy:
    kind: discrete_diffusion
    schedule:
      type: cosine
      total_steps: 64
    unmasking:
      policy: confidence
      min_confidence: 0.9
    output:
      max_length: 4096
      
# No kv_cache section — diffusion models don't use KV cache
# No speculative section — different acceleration methods apply
```

### 29.13 Acceleration Techniques (Diffusion-Specific)

| Technique | Description | Speedup |
|---|---|---|
| **Adaptive step scheduling** | Skip steps when confidence is already high | 2-4× |
| **Distillation** | Train fewer-step model from many-step teacher | 2-8× |
| **Caching unchanged positions** | Only recompute attention for recently-changed positions | 1.5-2× |
| **Progressive unmasking** | Unmask more tokens per step as confidence increases | 1.5-3× |
| **Parallel sample + verify** | Generate multiple candidates, pick best | Quality improvement |

### 29.14 Crate Structure

```
crates/onnx-genai-engine/src/
├── autoregressive.rs          # AR generation (existing)
├── iterative.rs               # Image diffusion (existing)
├── single_pass.rs             # Embedding/classification (existing)
├── composite.rs               # Multi-stage pipelines (existing)
├── discrete_diffusion.rs      # NEW: language diffusion engine
├── discrete_diffusion/
│   ├── mod.rs                 # DiscreteDiffusionEngine
│   ├── schedule.rs            # MaskSchedule variants
│   ├── unmasking.rs           # UnmaskingPolicy (confidence/entropy/hybrid)
│   ├── state.rs               # DiffusionState management
│   └── condition.rs           # DiffusionCondition (infill/edit/prefix)
└── ...
```

---

## 30. Strategic Positioning: Why Microsoft Frameworks Win

*Discussion between Justin and Claw, 2026-07-12.*

### 30.1 Core Advantages (Justin's Insight)

Microsoft's framework advantage distills to three pillars:

1. **Clarity** — Spec-first culture. ONNX succeeded not because it was the first model IR, but because the spec was clear enough for an entire industry to build on. The pattern: define the contract precisely, then implement.

2. **Reliability** — Enterprise customers choose "won't break in production" over "bleeding edge." ORT may not be the fastest on every benchmark, but Fortune 500 CTOs trust it with production workloads.

3. **Ongoing support** — The real moat. Open-source projects can be abandoned (and frequently are). Microsoft backing means security patches, version upgrades, and EP support continue for years. This compounds over time as switching costs grow.

These three reinforce each other: clarity enables reliability, reliability earns trust for ongoing investment, ongoing support maintains clarity as things evolve.

### 30.2 The Trade-Off (Claw's Observations)

These same strengths are weaknesses at the **frontier**:

- **Community projects iterate faster.** vLLM, llama.cpp, SGLang will always support the newest model/technique first. Clarity and reliability take time; the frontier doesn't wait.

- **The sweet spot is the bridge from innovation to production.** Community explores → Microsoft standardizes, stabilizes, and maintains long-term. ONNX is the textbook case. Not the first model IR, but the standard one.

- **onnx-genai's positioning follows this pattern.** Not competing with vLLM on who supports the latest trick first. Instead: the reliable, well-specified runtime for "I want to run genAI on ORT in production."

### 30.3 What Justin Might Not Have Considered

**1. The "good enough" trap.**
Reliability + support can become an excuse for slow iteration. Azure ML's model serving fell behind because "enterprise customers don't need that yet" — until they did, and the gap was too wide. The risk: clarity and reliability are only advantages if the *capability floor* stays competitive. If onnx-genai can't run Qwen3 within weeks of release (not months), the other advantages don't matter.

**2. Composability is an underrated Microsoft strength.**
Beyond clarity/reliability/support, Microsoft frameworks tend to be **composable with the rest of the Microsoft stack** — DirectML, Windows ML, Azure, VS Code, Copilot. This isn't just vendor lock-in; it's genuine reduced friction. onnx-genai on ORT on DirectML on Windows is a stack no one else can offer as a single coherent experience. This matters enormously for the "local coding agent on your laptop" use case.

**3. The spec can become the product.**
ONNX the spec is arguably more valuable than ORT the runtime. If `inference_metadata.yaml` becomes the standard way to describe how to serve a model (not just the model format, but the *serving contract* — KV cache strategy, quantization, pipeline topology), that's a spec-level contribution that outlasts any single runtime. This is the highest-leverage thing this project could produce.

**4. Trust is asymmetric to build and destroy.**
Microsoft's reliability reputation took decades to build. One bad ORT release that corrupts inference results, one security vulnerability in the C API, one breaking change that costs enterprise customers a migration — and years of trust evaporate. The ongoing support advantage requires *consistent* quality, not just current quality. This means the testing/validation bar for onnx-genai needs to be higher than vLLM's, not equal.

**5. The talent pipeline advantage.**
Microsoft can attract and retain framework engineers who want stability, impact at scale, and not burning out on startup pace. This is a real, underappreciated competitive advantage in a market where most AI infra talent is chasing startup equity. The people who build the best frameworks are often the ones who value craftsmanship over speed — and Microsoft's culture (at its best) supports that.

---

## 31. Observability, Logging & Profiling

### 31.1 Design Goals

1. **Zero-cost when off** — tracing/profiling in release builds must have zero overhead when disabled. No allocations, no syscalls, no branch mispredictions.
2. **Always-on structured logging** — every request gets a trace ID, every error has context.
3. **Perfetto trace export** — visualize inference execution on a timeline: forward passes, KV cache ops, scheduling, sampling, tool calls.
4. **Prometheus/OpenTelemetry metrics** — production monitoring: latency percentiles, throughput, cache hit rates, queue depths.
5. **Live introspection** — query engine state without restarting: active sessions, KV utilization, batch composition.

### 31.2 Layered Architecture

```
┌─────────────────────────────────────────────────────┐
│  Layer 4: Export                                     │
│  Perfetto (.pftrace) │ Chrome JSON │ OTLP │ stdout  │
├─────────────────────────────────────────────────────┤
│  Layer 3: Aggregation                               │
│  Metrics (counters, histograms) │ Span collector    │
├─────────────────────────────────────────────────────┤
│  Layer 2: Instrumentation                           │
│  Trace spans │ Events │ Counters │ Flow events      │
├─────────────────────────────────────────────────────┤
│  Layer 1: Core                                      │
│  tracing crate (Rust ecosystem standard)            │
└─────────────────────────────────────────────────────┘
```

### 31.3 Instrumentation Points

Every critical path gets a trace span:

```rust
/// Instrumented components and their span names.
///
/// Engine:
///   engine.generate              — full generation request (root span)
///   engine.prefill               — prompt encoding phase
///   engine.decode_step           — single decode iteration
///   engine.speculative_step      — draft + verify cycle
///   engine.diffusion_step        — single denoising step
///
/// ORT:
///   ort.session_run              — single ORT forward pass
///   ort.io_binding_bind          — binding inputs/outputs
///   ort.io_binding_run           — run with pre-bound tensors
///
/// KV Cache:
///   kv.allocate_page             — page allocation
///   kv.evict                     — page eviction (with reason)
///   kv.fork                      — CoW fork
///   kv.prefix_match              — prefix cache lookup (hit/miss)
///   kv.offload                   — GPU→CPU offload
///   kv.reload                    — CPU→GPU reload
///
/// Scheduler:
///   scheduler.iteration          — one scheduling round
///   scheduler.preempt            — preemption event
///   scheduler.batch_compose      — batch assembly
///
/// Sampling:
///   sampling.logit_process       — full processor chain
///   sampling.grammar_mask        — grammar constraint application
///   sampling.sample_token        — final token selection
///
/// Tool Use:
///   tool.detect                  — tool call detection
///   tool.parse                   — tool call JSON parsing
///   tool.execute                 — tool execution (external)
///   tool.resume                  — generation resume after tool result
///
/// Pipeline:
///   pipeline.stage               — multi-model pipeline stage
///   pipeline.dataflow_transfer   — tensor transfer between models
///
/// Server:
///   http.request                 — full HTTP request lifecycle
///   http.stream_chunk            — SSE chunk sent
```

### 31.4 Perfetto Trace Generation

Perfetto uses the Chrome Trace Event Format (JSON) or its own protobuf format. We generate both:

```rust
/// Perfetto-compatible trace writer.
pub struct PerfettoTracer {
    /// Ring buffer of trace events (bounded memory).
    events: RingBuffer<TraceEvent>,
    /// Whether tracing is currently active.
    active: AtomicBool,
    /// Process/thread ID mapping.
    thread_names: HashMap<u64, String>,
    /// Counter tracks (GPU pages, batch size, queue depth, etc.).
    counters: HashMap<String, CounterTrack>,
}

/// A single trace event (Chrome Trace Event Format).
#[derive(Serialize)]
pub struct TraceEvent {
    /// Event name.
    pub name: String,
    /// Category (engine, ort, kv, scheduler, sampling, tool, http).
    pub cat: String,
    /// Phase: B(begin), E(end), X(complete), C(counter), i(instant), f/s/t(flow).
    pub ph: char,
    /// Timestamp in microseconds.
    pub ts: u64,
    /// Duration in microseconds (for ph=X complete events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dur: Option<u64>,
    /// Process ID.
    pub pid: u64,
    /// Thread ID.
    pub tid: u64,
    /// Custom arguments.
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub args: HashMap<String, serde_json::Value>,
}

impl PerfettoTracer {
    /// Start recording trace events.
    pub fn start(&self) {
        self.active.store(true, Ordering::Release);
    }
    
    /// Stop recording and export to file.
    pub fn stop_and_export(&self, path: &Path, format: TraceFormat) -> Result<()> {
        self.active.store(false, Ordering::Release);
        match format {
            TraceFormat::ChromeJson => self.export_chrome_json(path),
            TraceFormat::Perfetto => self.export_perfetto_proto(path),
        }
    }
}

pub enum TraceFormat {
    /// Chrome JSON trace (open in chrome://tracing or Perfetto UI).
    ChromeJson,
    /// Perfetto protobuf (.pftrace, open in ui.perfetto.dev).
    Perfetto,
}
```

### 31.5 What a Trace Looks Like

```
Time →
┌─────────────────────────────────────────────────────────────────┐
│ http.request (session=agent-3, 847ms)                           │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ engine.generate (max_tokens=256)                            │ │
│ │ ┌──────────┐                                                │ │
│ │ │ prefill  │ 45ms, 1024 prompt tokens                      │ │
│ │ │ ┌──────┐ │                                                │ │
│ │ │ │ort.  │ │ 38ms, batch=1, tokens=1024                    │ │
│ │ │ │run   │ │                                                │ │
│ │ │ └──────┘ │                                                │ │
│ │ └──────────┘                                                │ │
│ │ ┌───┐┌───┐┌───┐┌───┐┌──────────────┐┌───┐┌───┐┌───┐       │ │
│ │ │d.1││d.2││d.3││d.4││ tool.execute ││d.5││d.6││d.7│ ...   │ │
│ │ │2ms││2ms││2ms││2ms││   120ms      ││2ms││2ms││2ms│       │ │
│ │ └───┘└───┘└───┘└───┘└──────────────┘└───┘└───┘└───┘       │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Counter: kv_pages_used ─────────────────────────                │
│          ▁▂▃▄▅▅▅▅▅▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇█████████                │
│ Counter: batch_size    ─────────────────────────                │
│          ▃▃▃▃▃▃▃▃▃▃▃▃▃▁▁▁▁▁▁▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃                │
│ Counter: queue_depth   ─────────────────────────                │
│          ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁                │
└─────────────────────────────────────────────────────────────────┘
```

### 31.6 Flow Events (Request Lifecycle Tracking)

Track a request across async boundaries (scheduler queue → batch → forward pass → response):

```rust
/// Flow events connect spans across threads/async boundaries.
/// Essential for seeing: "this request waited in queue 50ms, then got
/// batched with 3 others, then forward pass took 12ms."

// When request enters scheduler queue:
tracer.flow_start("request_flow", request_id, "scheduler.enqueue");

// When scheduler picks it for a batch:
tracer.flow_step("request_flow", request_id, "scheduler.batch_assign");

// When forward pass starts:
tracer.flow_step("request_flow", request_id, "ort.session_run");

// When token is sent back to client:
tracer.flow_end("request_flow", request_id, "http.stream_chunk");
```

### 31.7 Counter Tracks

Continuous metrics visualized as line charts in Perfetto:

```rust
/// Counters emitted every scheduling iteration.
pub struct EngineCounters {
    // Memory
    pub kv_pages_used: u64,
    pub kv_pages_total: u64,
    pub kv_pages_shared: u64,       // prefix cache shared pages
    pub kv_offloaded_pages: u64,    // pages on CPU
    
    // Throughput
    pub tokens_generated_total: u64,
    pub tokens_per_second: f64,     // rolling window
    pub prefill_tokens_per_second: f64,
    
    // Batching
    pub current_batch_size: u64,    // sequences in current batch
    pub current_batch_tokens: u64,  // total tokens in current forward
    pub queue_depth: u64,           // waiting requests
    
    // Sessions
    pub active_sessions: u64,
    pub paused_sessions: u64,       // waiting on tool results
    
    // Speculation
    pub speculation_acceptance_rate: f64,
    pub draft_tokens_per_step: f64,
    
    // Cache
    pub prefix_cache_hit_rate: f64,
    pub prefix_cache_entries: u64,
    
    // Grammar
    pub grammar_mask_time_us: u64,  // time spent in grammar constraint
}
```

### 31.8 Integration with `tracing` Crate

Use Rust's `tracing` ecosystem as the instrumentation layer. Custom subscriber exports to Perfetto:

```rust
use tracing::{instrument, info_span, Span};
use tracing_subscriber::Layer;

/// Instrument a decode step with full context.
#[instrument(
    level = "debug",
    skip(self, state),
    fields(
        session_id = %session_id,
        step = step,
        batch_size = batch_size,
        tokens_in_batch = tokens_in_batch,
    )
)]
pub fn decode_step(
    &mut self,
    session_id: SessionId,
    step: usize,
    batch_size: usize,
    tokens_in_batch: usize,
) -> Result<TokenId> {
    // ORT forward pass (sub-span created automatically)
    let logits = {
        let _ort_span = info_span!("ort.session_run",
            input_tokens = 1,
            kv_length = self.kv_length(session_id),
        ).entered();
        self.session.run(&inputs)?
    };
    
    // Logit processing
    let token = {
        let _sample_span = info_span!("sampling.logit_process",
            chain_len = self.chain.len(),
        ).entered();
        self.chain.process(&mut logits, &context);
        self.sampler.sample(&logits, &context)
    };
    
    token
}

/// Custom tracing subscriber that collects spans into Perfetto format.
pub struct PerfettoLayer {
    tracer: Arc<PerfettoTracer>,
}

impl<S: tracing::Subscriber> Layer<S> for PerfettoLayer {
    fn on_enter(&self, id: &span::Id, ctx: Context<'_, S>) {
        // Record span begin event with timestamp
    }
    
    fn on_exit(&self, id: &span::Id, ctx: Context<'_, S>) {
        // Record span end event, compute duration
    }
    
    fn on_event(&self, event: &tracing::Event<'_>, ctx: Context<'_, S>) {
        // Record instant events (counters, markers)
    }
}
```

### 31.9 Structured Logging

Every log line is structured JSON with trace context:

```rust
// What gets logged (structured, not printf):
{
    "timestamp": "2026-07-12T14:30:00.123Z",
    "level": "INFO",
    "target": "onnx_genai_engine::engine",
    "span": "engine.generate",
    "trace_id": "abc123",
    "session_id": "agent-worker-3",
    "fields": {
        "event": "generation_complete",
        "tokens_generated": 142,
        "time_to_first_token_ms": 45,
        "total_time_ms": 847,
        "tokens_per_second": 167.6,
        "finish_reason": "stop",
        "cache_hit_tokens": 512,
        "speculation_acceptance_rate": 0.82
    }
}
```

Log levels and what goes where:

```
ERROR — ORT failures, OOM, corruption (always on)
WARN  — preemption, eviction, slow forward pass, fallback paths
INFO  — request start/end, session lifecycle, config changes
DEBUG — per-step decode, cache operations, scheduling decisions
TRACE — per-token logits, individual processor timings, tensor shapes
```

### 31.10 Runtime Profiling Controls

```rust
/// Profiling can be started/stopped at runtime via HTTP API.
/// No restart needed.

// Start profiling:
// POST /v1/debug/trace/start
// { "duration_seconds": 30, "level": "debug" }

// Stop and download trace:
// POST /v1/debug/trace/stop
// → Returns .pftrace file (open in ui.perfetto.dev)

// Or via CLI:
// onnx-genai profile --duration 30s --output trace.pftrace
// onnx-genai profile --duration 10s --output trace.json --format chrome

/// HTTP endpoints for runtime introspection.
pub fn debug_routes() -> Router {
    Router::new()
        // Tracing
        .route("/v1/debug/trace/start", post(start_trace))
        .route("/v1/debug/trace/stop", post(stop_trace))
        
        // Live metrics
        .route("/v1/debug/metrics", get(prometheus_metrics))
        .route("/v1/debug/counters", get(live_counters))
        
        // Engine state
        .route("/v1/debug/sessions", get(list_sessions))
        .route("/v1/debug/kv", get(kv_cache_state))
        .route("/v1/debug/scheduler", get(scheduler_state))
        .route("/v1/debug/batch", get(current_batch_info))
        
        // Log level control
        .route("/v1/debug/log-level", put(set_log_level))
}
```

### 31.11 ORT-Level Profiling Integration

ORT has its own profiling support. We wire it in:

```rust
/// Enable ORT's built-in profiling and merge with our traces.
pub struct OrtProfilingBridge {
    /// ORT profiling output path.
    ort_profile_path: PathBuf,
}

impl OrtProfilingBridge {
    /// Enable ORT profiling on session options.
    pub fn enable(&self, options: &mut SessionOptions) {
        // ORT_ENABLE_PROFILING env var or API call
        // ORT outputs its own Chrome trace JSON with kernel-level timing
    }
    
    /// Merge ORT's trace with our engine trace.
    /// ORT trace has: kernel execution, memory allocation, EP selection.
    /// Our trace has: scheduling, KV cache, sampling, tool use.
    /// Combined = full picture from HTTP request to GPU kernel.
    pub fn merge_traces(
        engine_trace: &Path,
        ort_trace: &Path,
        output: &Path,
    ) -> Result<()> {
        // Align timestamps, merge into single Perfetto trace
        ...
    }
}
```

### 31.12 Prometheus / OpenTelemetry Metrics

```rust
/// Prometheus metrics for production monitoring.
/// Exposed at GET /v1/debug/metrics
///
/// # HELP onnx_genai_requests_total Total generation requests
/// # TYPE onnx_genai_requests_total counter
/// onnx_genai_requests_total{status="success"} 1234
/// onnx_genai_requests_total{status="error"} 5
///
/// # HELP onnx_genai_time_to_first_token_seconds TTFT histogram
/// # TYPE onnx_genai_time_to_first_token_seconds histogram
/// onnx_genai_time_to_first_token_seconds_bucket{le="0.05"} 800
/// onnx_genai_time_to_first_token_seconds_bucket{le="0.1"} 1100
///
/// # HELP onnx_genai_tokens_per_second Current generation throughput
/// # TYPE onnx_genai_tokens_per_second gauge
/// onnx_genai_tokens_per_second 167.6
///
/// # HELP onnx_genai_kv_pages_used Current KV cache page utilization
/// # TYPE onnx_genai_kv_pages_used gauge
/// onnx_genai_kv_pages_used 1847
///
/// Key metrics:
/// - onnx_genai_requests_total (counter, by status)
/// - onnx_genai_time_to_first_token_seconds (histogram)
/// - onnx_genai_inter_token_latency_seconds (histogram)
/// - onnx_genai_tokens_per_second (gauge)
/// - onnx_genai_tokens_generated_total (counter)
/// - onnx_genai_kv_pages_used (gauge)
/// - onnx_genai_kv_pages_total (gauge)
/// - onnx_genai_kv_evictions_total (counter, by reason)
/// - onnx_genai_kv_prefix_cache_hit_rate (gauge)
/// - onnx_genai_batch_size (histogram)
/// - onnx_genai_queue_depth (gauge)
/// - onnx_genai_queue_wait_seconds (histogram)
/// - onnx_genai_active_sessions (gauge, by priority)
/// - onnx_genai_speculation_acceptance_rate (gauge)
/// - onnx_genai_grammar_mask_seconds (histogram)
/// - onnx_genai_preemptions_total (counter)
/// - onnx_genai_ort_forward_seconds (histogram, by model)
```

### 31.13 Zero-Cost When Off

```rust
/// All instrumentation compiles to nothing when disabled.
/// 
/// Release build with default features:
///   - Structured logging: always on (INFO level, minimal overhead)
///   - Perfetto tracing: compiled in but inactive (atomic bool check only)
///   - Prometheus metrics: compiled in, near-zero cost (atomic counters)
///   - Debug endpoints: compiled in, no cost unless called
///
/// Release build with `--no-default-features`:
///   - Everything stripped via cfg

#[cfg(feature = "tracing")]
macro_rules! trace_span {
    ($name:expr, $($field:tt)*) => {
        tracing::info_span!($name, $($field)*)
    };
}

#[cfg(not(feature = "tracing"))]
macro_rules! trace_span {
    ($name:expr, $($field:tt)*) => {
        // Compiles to nothing
    };
}
```

### 31.14 Feature Flags

```toml
[features]
default = ["logging", "metrics", "tracing"]

# Structured logging (tracing crate)
logging = ["tracing", "tracing-subscriber"]

# Prometheus metrics
metrics = ["prometheus-client"]

# Perfetto trace generation
tracing = ["tracing", "tracing-subscriber"]
perfetto = ["tracing", "prost"]  # protobuf for native .pftrace format

# OpenTelemetry export
otel = ["opentelemetry", "opentelemetry-otlp", "tracing-opentelemetry"]

# Debug HTTP endpoints (/v1/debug/*)
debug-endpoints = []
```

### 31.15 Configuration

```yaml
# server_config.yaml
observability:
  logging:
    level: info           # error/warn/info/debug/trace
    format: json          # json or pretty (for development)
    output: stderr        # stderr, stdout, or file path
    
  metrics:
    enabled: true
    endpoint: /v1/debug/metrics
    
  tracing:
    enabled: false        # off by default, start via API
    buffer_size: 1000000  # max events in ring buffer
    default_level: debug  # what to capture when tracing is active
    
  profiling:
    ort_profiling: false       # enable ORT's built-in profiling
    merge_ort_traces: true     # auto-merge ORT + engine traces
```

### 31.16 Comprehensive Metrics Catalog (informed by vLLM/SGLang)

After studying vLLM v1's metrics system, here's the complete metrics catalog organized by category. We track everything vLLM tracks, plus our ORT-specific and multi-agent metrics.

#### A. Prefix Cache Metrics

vLLM uses a sliding-window `CachingMetrics` (last N requests) for hit rate. We adopt the same pattern:

```rust
/// Sliding-window cache metrics (mirrors vLLM's CachingMetrics).
pub struct CachingMetrics {
    /// Rolling window of (requests, queries, hits).
    window: VecDeque<(u64, u64, u64)>,
    /// Max requests in window.
    max_recent_requests: usize,  // default: 1000
    /// Aggregated totals within window.
    agg_requests: u64,
    agg_queries: u64,   // tokens queried
    agg_hits: u64,       // tokens that hit cache
}

/// Per-scheduling-step prefix cache stats.
pub struct PrefixCacheStats {
    // New requests
    pub requests: u64,           // number of requests in this update
    pub queries: u64,            // tokens queried against cache
    pub hits: u64,               // tokens found in cache (saved compute)
    
    // Previously preempted requests (re-scheduled)
    pub preempted_requests: u64,
    pub preempted_queries: u64,
    pub preempted_hits: u64,     // preempted requests often hit more (their KV was evicted but prefix might remain)
    
    pub reset: bool,             // cache was reset/cleared
}
```

**Prometheus metrics:**
```
# Prefix cache
onnx_genai_prefix_cache_hit_rate            gauge     # sliding window hit rate (0.0-1.0)
onnx_genai_prefix_cache_queries_total       counter   # total tokens queried
onnx_genai_prefix_cache_hits_total          counter   # total tokens found in cache
onnx_genai_prefix_cache_hit_tokens_saved    counter   # prefill tokens skipped thanks to cache
onnx_genai_prefix_cache_entries             gauge     # number of cached prefix entries
onnx_genai_prefix_cache_resets_total        counter   # times cache was fully reset
```

#### B. KV Cache Eviction Metrics

vLLM tracks per-eviction events with lifetime and reuse patterns. Critical for tuning cache size:

```rust
/// Single KV cache block eviction event.
pub struct KvEvictionEvent {
    /// How long this block lived in cache (seconds).
    pub lifetime_seconds: f64,
    /// How long since this block was last accessed.
    pub idle_seconds: f64,
    /// Gaps between reuses (if reused multiple times).
    pub reuse_gaps_seconds: Vec<f64>,
    /// Eviction reason.
    pub reason: EvictionReason,
    /// Priority of the evicted session.
    pub session_priority: AgentPriority,
}

pub enum EvictionReason {
    /// No free pages, need space for new request.
    Capacity,
    /// Lower-priority session preempted.
    Preemption,
    /// Session idle timeout expired.
    IdleTimeout,
    /// Explicit cache clear (API call).
    ManualReset,
    /// Memory pressure from other EP (e.g., ORT internal allocation).
    MemoryPressure,
}
```

**Prometheus metrics:**
```
# KV cache pages
onnx_genai_kv_pages_used                   gauge     # current pages in use
onnx_genai_kv_pages_total                  gauge     # total available pages
onnx_genai_kv_pages_shared                 gauge     # pages shared via prefix cache (CoW)
onnx_genai_kv_pages_offloaded              gauge     # pages on CPU (swapped out)
onnx_genai_kv_usage_ratio                  gauge     # used/total (0.0-1.0)

# Eviction
onnx_genai_kv_evictions_total              counter   # total eviction events {reason}
onnx_genai_kv_eviction_lifetime_seconds    histogram # how long evicted blocks lived
onnx_genai_kv_eviction_idle_seconds        histogram # idle time before eviction
onnx_genai_kv_offload_total                counter   # GPU→CPU offload events
onnx_genai_kv_reload_total                 counter   # CPU→GPU reload events
onnx_genai_kv_offload_latency_seconds      histogram # offload transfer time
```

#### C. Request Lifecycle Metrics

vLLM tracks detailed per-request timestamps. Essential for diagnosing latency:

```rust
/// Per-request lifecycle timestamps.
pub struct RequestTimestamps {
    /// When request arrived at HTTP layer (wall clock).
    pub arrival_time: Instant,
    /// When request entered scheduler queue (monotonic).
    pub queued_ts: Instant,
    /// When request was first scheduled into a batch.
    pub scheduled_ts: Instant,
    /// When first token was generated.
    pub first_token_ts: Instant,
    /// When last token was generated (request complete).
    pub last_token_ts: Instant,
}

/// Derived metrics from timestamps.
pub struct RequestLatencyBreakdown {
    /// Queue wait time: queued_ts → scheduled_ts
    pub queue_time_ms: f64,
    /// Prefill time: scheduled_ts → first_token_ts
    pub prefill_time_ms: f64,
    /// Time to first token: arrival → first_token (user-perceived)
    pub ttft_ms: f64,
    /// Inter-token latency: avg time between consecutive tokens
    pub itl_ms: f64,
    /// End-to-end: arrival → last_token
    pub e2e_latency_ms: f64,
    /// Total tokens generated
    pub num_generation_tokens: u32,
    /// Prompt tokens
    pub num_prompt_tokens: u32,
    /// Prompt tokens that were cached (not computed)
    pub num_cached_tokens: u32,
    /// Finish reason
    pub finish_reason: FinishReason,
}
```

**Prometheus metrics:**
```
# Latency
onnx_genai_time_to_first_token_seconds     histogram # TTFT (most important user-facing metric)
onnx_genai_inter_token_latency_seconds     histogram # per-token latency
onnx_genai_e2e_latency_seconds             histogram # end-to-end request latency
onnx_genai_queue_wait_seconds              histogram # time in scheduler queue
onnx_genai_prefill_seconds                 histogram # prefill phase duration

# Request counts
onnx_genai_requests_total                  counter   # {status=success|error|cancelled}
onnx_genai_requests_active                 gauge     # currently in-flight
onnx_genai_requests_waiting                gauge     # in scheduler queue
onnx_genai_requests_preempted_total        counter   # preempted and re-queued
```

#### D. Throughput Metrics

```
# Token throughput
onnx_genai_prompt_tokens_total             counter   # total prompt tokens processed
onnx_genai_prompt_tokens_computed           counter   # actually computed (excl. cached)
onnx_genai_generation_tokens_total         counter   # total tokens generated
onnx_genai_prompt_throughput               gauge     # prompt tokens/sec (rolling)
onnx_genai_generation_throughput           gauge     # generation tokens/sec (rolling)
```

#### E. Scheduler & Batching Metrics

```
# Scheduler
onnx_genai_scheduler_running_requests      gauge     # requests currently generating
onnx_genai_scheduler_waiting_requests      gauge     # requests waiting to be scheduled
onnx_genai_scheduler_skipped_requests      gauge     # deferred (e.g., LoRA adapter not loaded)
onnx_genai_scheduler_preemptions_total     counter   # total preemption events
onnx_genai_scheduler_step_counter          counter   # total scheduling iterations

# Batching
onnx_genai_batch_size                      histogram # sequences per forward pass
onnx_genai_batch_tokens                    histogram # total tokens per forward pass
onnx_genai_batch_utilization               gauge     # batch_tokens / max_batch_tokens
onnx_genai_prefill_chunk_tokens            histogram # tokens per chunked prefill
```

#### F. Speculative Decoding Metrics

vLLM tracks per-position acceptance rates — not just overall. This is critical for tuning:

```rust
/// Per-step speculation stats.
pub struct SpecDecodingStats {
    /// Number of speculative positions configured.
    pub num_spec_tokens: usize,
    /// Total draft rounds this step.
    pub num_drafts: u64,
    /// Total draft tokens proposed.
    pub num_draft_tokens: u64,
    /// Total draft tokens accepted.
    pub num_accepted_tokens: u64,
    /// Per-position acceptance counts.
    /// Index 0 = first speculative position, etc.
    /// Shows acceptance rate decay by position.
    pub accepted_per_position: Vec<u64>,
    pub drafted_per_position: Vec<u64>,
}

// Example log output (vLLM style):
// SpecDecoding metrics:
//   Mean acceptance length: 3.42
//   Accepted throughput: 534.2 tokens/s
//   Drafted throughput: 712.8 tokens/s
//   Per-position acceptance rate: 0.95, 0.82, 0.71, 0.55, 0.38
//   Avg Draft acceptance rate: 75.1%
```

**Prometheus metrics:**
```
onnx_genai_spec_drafts_total               counter   # total draft rounds
onnx_genai_spec_draft_tokens_total         counter   # total tokens drafted
onnx_genai_spec_accepted_tokens_total      counter   # total tokens accepted
onnx_genai_spec_acceptance_rate            gauge     # rolling acceptance rate
onnx_genai_spec_mean_acceptance_length     gauge     # avg accepted tokens per draft round
onnx_genai_spec_acceptance_per_position    gauge     # {position=0,1,2,...} per-position rate
onnx_genai_spec_draft_throughput           gauge     # draft tokens/sec
onnx_genai_spec_accepted_throughput        gauge     # accepted tokens/sec
```

#### G. ORT-Specific Metrics (unique to us)

```
# Execution Provider
onnx_genai_ort_forward_seconds             histogram # {model, ep} forward pass time
onnx_genai_ort_ep_selected                 gauge     # which EP is active {ep=CUDA|DirectML|CPU}
onnx_genai_ort_io_binding_bind_seconds     histogram # IoBinding setup time
onnx_genai_ort_io_binding_run_seconds      histogram # IoBinding run time

# Memory (ORT-level)
onnx_genai_ort_memory_allocated_bytes      gauge     # {device} ORT allocator usage
onnx_genai_ort_memory_arena_bytes          gauge     # arena allocator size
```

#### H. Multi-Agent Metrics (unique to us)

```
# Sessions
onnx_genai_sessions_active                 gauge     # {priority} total active sessions
onnx_genai_sessions_paused                 gauge     # sessions waiting on tool results
onnx_genai_sessions_kv_pages               histogram # pages per session
onnx_genai_session_lifetime_seconds        histogram # session duration

# Tool use
onnx_genai_tool_calls_total                counter   # {tool_name} total tool invocations
onnx_genai_tool_latency_seconds            histogram # {tool_name} tool execution time
onnx_genai_tool_turns_per_request          histogram # tool round-trips per request

# Grammar
onnx_genai_grammar_mask_seconds            histogram # time computing token mask
onnx_genai_grammar_cache_hit_rate          gauge     # grammar state cache hit rate
onnx_genai_grammar_type_active             gauge     # {type=json|json_schema|gbnf|regex}
```

#### I. Error & Health Metrics

```
onnx_genai_errors_total                    counter   # {type} ORT errors, OOM, timeout, etc.
onnx_genai_corrupted_logits_total          counter   # NaN/Inf in logits (model issue)
onnx_genai_oom_events_total                counter   # out-of-memory events
onnx_genai_model_load_seconds              histogram # model loading time
onnx_genai_uptime_seconds                  gauge     # server uptime
```

### 31.17 Logging Format for Periodic Stats

Following vLLM's pattern, log a summary line every N seconds:

```
INFO  Throughput: prompt=1234.5 tok/s, gen=567.8 tok/s.
      Running: 8 reqs, Waiting: 2 reqs.
      KV cache: 87.3% (1792/2048 pages, 256 shared, 64 offloaded).
      Prefix cache hit rate: 73.2% (last 1000 reqs).
      SpecDecode: acceptance=75.1%, mean_len=3.42, per_pos=[0.95, 0.82, 0.71, 0.55].
      Preemptions: 0, Evictions: 3.
```

## 32. Metrics Exposure API

§31 defined *what* we measure. This section defines *how* we expose it: the concrete HTTP endpoints, wire formats, security model, and integration points.

### 32.1 Endpoint Summary

| Path | Method | Auth | Purpose | Feature Flag |
|------|--------|------|---------|--------------|
| `/metrics` | GET | None | Prometheus scrape target | `metrics` |
| `/v1/status` | GET | API key | Quick health/status JSON | *(always on)* |
| `/v1/debug/sessions` | GET | Localhost | Active session introspection | `debug-endpoints` |
| `/v1/debug/kv` | GET | Localhost | KV cache state | `debug-endpoints` |
| `/v1/debug/scheduler` | GET | Localhost | Scheduler state | `debug-endpoints` |
| `/v1/debug/batch` | GET | Localhost | Current batch composition | `debug-endpoints` |
| `/v1/debug/events` | GET (SSE) | Localhost | Real-time event stream | `debug-endpoints` |
| `/v1/debug/trace/start` | POST | Localhost | Start Perfetto trace | `debug-endpoints` |
| `/v1/debug/trace/stop` | POST | Localhost | Stop trace & download | `debug-endpoints` |
| `/v1/debug/log-level` | PUT | Localhost | Change log level at runtime | `debug-endpoints` |
| `/v1/debug/config` | GET | Localhost | Running configuration dump | `debug-endpoints` |

### 32.2 Prometheus Metrics Endpoint

**`GET /metrics`**

Standard Prometheus exposition format. Served on the main HTTP port (not a separate port) for simplicity. No authentication — Prometheus expects unauthenticated scrape targets. Restrict via network policy or `observability.metrics.bind` config.

**Request:** No parameters.

**Response:** `text/plain; version=0.0.4; charset=utf-8`

```
# HELP onnx_genai_requests_total Total generation requests.
# TYPE onnx_genai_requests_total counter
onnx_genai_requests_total{status="success"} 12847
onnx_genai_requests_total{status="error"} 23
onnx_genai_requests_total{status="cancelled"} 5

# HELP onnx_genai_time_to_first_token_seconds Time to first token.
# TYPE onnx_genai_time_to_first_token_seconds histogram
onnx_genai_time_to_first_token_seconds_bucket{le="0.01"} 342
onnx_genai_time_to_first_token_seconds_bucket{le="0.025"} 1893
onnx_genai_time_to_first_token_seconds_bucket{le="0.05"} 8721
onnx_genai_time_to_first_token_seconds_bucket{le="0.1"} 11432
onnx_genai_time_to_first_token_seconds_bucket{le="0.25"} 12510
onnx_genai_time_to_first_token_seconds_bucket{le="0.5"} 12780
onnx_genai_time_to_first_token_seconds_bucket{le="1.0"} 12840
onnx_genai_time_to_first_token_seconds_bucket{le="+Inf"} 12847
onnx_genai_time_to_first_token_seconds_sum 412.83
onnx_genai_time_to_first_token_seconds_count 12847

# HELP onnx_genai_generation_throughput Current generation tokens/sec.
# TYPE onnx_genai_generation_throughput gauge
onnx_genai_generation_throughput 534.2

# HELP onnx_genai_kv_usage_ratio KV cache utilization (0.0-1.0).
# TYPE onnx_genai_kv_usage_ratio gauge
onnx_genai_kv_usage_ratio 0.873

# ... (full catalog from §31.16)
```

**Feature flag:** `metrics`. When disabled, `/metrics` returns `404`.

**Implementation notes:**
- Uses `prometheus-client` crate (not the deprecated `prometheus` crate).
- Metrics are collected into a `Registry` and rendered on each scrape — no background thread.
- Histogram buckets for latency: `[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.5, 5.0]`.
- Memory overhead: ~8 KB per unique metric family. Full catalog ≈ 200 KB total.

### 32.3 OpenTelemetry OTLP Export

Alternative to Prometheus pull model. Pushes metrics (and traces/logs) to any OTLP-compatible backend (Grafana Cloud, Datadog, Jaeger, etc.).

**Configuration:**

```yaml
observability:
  otlp:
    enabled: false
    endpoint: "http://localhost:4317"    # gRPC OTLP endpoint
    # endpoint: "http://localhost:4318"  # HTTP OTLP endpoint
    protocol: grpc                       # grpc | http
    headers:                             # custom headers (e.g., auth)
      Authorization: "Bearer <token>"
    export_interval_seconds: 15          # metric push interval
    export_timeout_seconds: 10
    resource_attributes:                 # OpenTelemetry resource
      service.name: onnx-genai
      service.version: "0.1.0"
      deployment.environment: production
    export:
      metrics: true
      traces: true                       # export tracing spans as OTLP traces
      logs: false                        # structured logs as OTLP log records
```

**Feature flag:** `otel`. When enabled, metrics are dual-exported (Prometheus *and* OTLP). When `metrics` is off but `otel` is on, only OTLP push is active.

**Crates:** `opentelemetry 0.28`, `opentelemetry-otlp`, `tracing-opentelemetry`.

**Graceful degradation:** If the OTLP endpoint is unreachable, export silently drops batches after `export_timeout_seconds`. A counter `onnx_genai_otlp_export_failures_total` tracks drops. No backpressure on the engine.

### 32.4 Status Endpoint

**`GET /v1/status`**

Quick health check returning server state. Suitable for load balancer health checks and monitoring dashboards. Authenticated via API key (same as `/v1/chat/completions`).

**Request parameters:** None.

**Response:** `application/json`, status `200 OK`.

```json
{
  "status": "ready",
  "version": "0.1.0",
  "uptime_seconds": 84321,
  "model": {
    "name": "microsoft/phi-4-mini",
    "parameters": "3.8B",
    "quantization": "int4-awq",
    "execution_provider": "CUDA",
    "max_context_length": 131072
  },
  "engine": {
    "requests_active": 8,
    "requests_waiting": 2,
    "requests_total": 12847,
    "tokens_generated_total": 1847293,
    "generation_throughput_tps": 534.2,
    "prompt_throughput_tps": 1234.5
  },
  "kv_cache": {
    "pages_used": 1792,
    "pages_total": 2048,
    "usage_ratio": 0.875,
    "pages_shared": 256,
    "pages_offloaded": 64,
    "prefix_cache_hit_rate": 0.732
  },
  "sessions": {
    "active": 8,
    "paused": 3,
    "total_created": 142
  },
  "speculation": {
    "enabled": true,
    "acceptance_rate": 0.751,
    "mean_acceptance_length": 3.42
  }
}
```

**Status values:**
- `"starting"` — model loading in progress
- `"ready"` — accepting requests
- `"draining"` — graceful shutdown, finishing in-flight requests
- `"error"` — unrecoverable error (detail in `error` field)

**Feature flag:** None. Always compiled in; it's the server's health endpoint.

### 32.5 Debug Endpoints

All debug endpoints require the `debug-endpoints` feature flag and are **localhost-only** by default. Remote access requires explicit opt-in via `observability.debug.allow_remote: true` plus API key auth.

#### 32.5.1 `GET /v1/debug/sessions`

List active sessions with KV cache and generation state.

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | u32 | 50 | Max sessions to return |
| `offset` | u32 | 0 | Pagination offset |
| `sort` | string | `"kv_pages"` | Sort by: `kv_pages`, `created`, `last_active`, `priority` |
| `priority` | string | *(all)* | Filter: `interactive`, `batch`, `background` |

**Response:**

```json
{
  "total": 11,
  "sessions": [
    {
      "session_id": "agent-worker-3",
      "priority": "interactive",
      "state": "generating",
      "created_at": "2026-07-12T14:30:00Z",
      "last_active_at": "2026-07-12T14:55:12Z",
      "kv_pages": 347,
      "context_length": 22208,
      "max_context_length": 131072,
      "tokens_generated": 1842,
      "pending_tool_calls": 0,
      "speculation": {
        "enabled": true,
        "acceptance_rate": 0.82
      }
    },
    {
      "session_id": "agent-lead",
      "priority": "interactive",
      "state": "paused_tool_call",
      "created_at": "2026-07-12T13:10:00Z",
      "last_active_at": "2026-07-12T14:52:30Z",
      "kv_pages": 892,
      "context_length": 57088,
      "max_context_length": 131072,
      "tokens_generated": 8421,
      "pending_tool_calls": 1,
      "speculation": {
        "enabled": true,
        "acceptance_rate": 0.79
      }
    }
  ]
}
```

#### 32.5.2 `GET /v1/debug/kv`

KV cache detailed state.

**Query parameters:** None.

**Response:**

```json
{
  "total_pages": 2048,
  "used_pages": 1792,
  "free_pages": 256,
  "shared_pages": 312,
  "offloaded_pages": 64,
  "usage_ratio": 0.875,
  "page_size_tokens": 64,
  "total_capacity_tokens": 131072,
  "prefix_cache": {
    "entries": 47,
    "hit_rate": 0.732,
    "hit_rate_window": 1000,
    "total_queries": 184293,
    "total_hits": 134903
  },
  "eviction_stats": {
    "total_evictions": 342,
    "by_reason": {
      "capacity": 298,
      "preemption": 31,
      "idle_timeout": 13,
      "manual_reset": 0
    },
    "avg_lifetime_seconds": 127.4,
    "avg_idle_before_eviction_seconds": 45.2
  },
  "per_session": [
    { "session_id": "agent-lead", "pages": 892, "tokens": 57088, "shared_pages": 128 },
    { "session_id": "agent-worker-3", "pages": 347, "tokens": 22208, "shared_pages": 89 }
  ]
}
```

#### 32.5.3 `GET /v1/debug/scheduler`

Scheduler internals.

**Response:**

```json
{
  "policy": "priority_fcfs",
  "max_batch_size": 32,
  "max_batch_tokens": 4096,
  "running": [
    { "session_id": "agent-worker-3", "priority": "interactive", "tokens_remaining": 214 }
  ],
  "waiting": [
    { "session_id": "batch-job-7", "priority": "batch", "queued_at": "2026-07-12T14:55:10Z", "prompt_tokens": 2048 }
  ],
  "preempted": [],
  "stats": {
    "total_iterations": 184293,
    "total_preemptions": 31,
    "avg_batch_size": 4.7,
    "avg_batch_tokens": 1247
  }
}
```

#### 32.5.4 `GET /v1/debug/batch`

Current batch composition (what's in the forward pass right now).

**Response:**

```json
{
  "batch_id": 184294,
  "timestamp": "2026-07-12T14:55:12.345Z",
  "sequences": 5,
  "total_tokens": 1389,
  "composition": [
    {
      "session_id": "agent-worker-3",
      "type": "decode",
      "tokens": 1,
      "kv_length": 22208
    },
    {
      "session_id": "new-request-42",
      "type": "prefill",
      "tokens": 1024,
      "kv_length": 0,
      "chunked": true,
      "chunk_index": 0,
      "total_chunks": 2
    },
    {
      "session_id": "agent-worker-3",
      "type": "speculative_draft",
      "tokens": 4,
      "kv_length": 22208
    }
  ],
  "padding_tokens": 0,
  "utilization": 0.339
}
```

#### 32.5.5 `GET /v1/debug/config`

Dump the running configuration (redacting secrets).

**Response:**

```json
{
  "model": { "path": "/models/phi-4-mini-int4", "max_context_length": 131072 },
  "engine": { "max_batch_size": 32, "max_batch_tokens": 4096, "scheduling_policy": "priority_fcfs" },
  "kv_cache": { "num_pages": 2048, "page_size": 64, "offload_enabled": true },
  "speculation": { "enabled": true, "draft_model": "/models/phi-4-mini-draft", "num_speculative_tokens": 5 },
  "observability": { "logging": { "level": "info" }, "metrics": { "enabled": true }, "otlp": { "enabled": false } }
}
```

### 32.6 SSE Event Stream

**`GET /v1/debug/events`**

Server-Sent Events stream for real-time monitoring. Localhost-only. Ideal for TUI dashboards or live Grafana panels via SSE data source.

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `filter` | string | `"*"` | Comma-separated event types: `request`, `batch`, `eviction`, `preemption`, `error`, `stats`, `tool` |
| `interval_ms` | u32 | 1000 | Min interval for `stats` events |

**Response:** `text/event-stream`

```
event: stats
data: {"timestamp":"2026-07-12T14:55:12Z","throughput_tps":534.2,"requests_active":8,"requests_waiting":2,"kv_usage":0.875,"batch_size":5}

event: request
data: {"type":"start","request_id":"req-abc123","session_id":"agent-worker-3","prompt_tokens":1024}

event: batch
data: {"batch_id":184294,"sequences":5,"tokens":1389,"prefill_seqs":1,"decode_seqs":4}

event: eviction
data: {"session_id":"idle-session-9","pages":128,"reason":"idle_timeout","lifetime_seconds":342.1}

event: request
data: {"type":"complete","request_id":"req-abc123","session_id":"agent-worker-3","tokens_generated":142,"ttft_ms":45.2,"e2e_ms":847.3,"finish_reason":"stop"}

event: error
data: {"type":"ort_error","message":"CUDA out of memory","session_id":"batch-job-12"}

event: tool
data: {"session_id":"agent-lead","tool":"web_search","state":"executing","elapsed_ms":230}
```

**Connection limits:** Max 4 concurrent SSE connections. Additional connections get `429 Too Many Requests`.

**Memory overhead:** Events are not buffered — they're written directly to connected SSE clients. No ring buffer for SSE (unlike Perfetto traces). If no clients are connected, events are not generated.

### 32.7 Per-Request Response Metadata

Every `/v1/chat/completions` response includes timing metadata — no extra config needed.

#### 32.7.1 Response Headers

```
X-Request-Id: req-abc123
X-Time-To-First-Token-Ms: 45
X-Queue-Wait-Ms: 3
X-Prefill-Ms: 42
X-Generation-Ms: 802
X-Tokens-Generated: 142
X-Tokens-Prompt: 1024
X-Tokens-Cached: 512
X-Speculation-Acceptance-Rate: 0.82
```

Headers are always present. Lightweight (no JSON parsing needed), compatible with proxies/load balancers.

#### 32.7.2 Usage Object in Response Body

Standard OpenAI-compatible `usage` field, plus extended timing:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "phi-4-mini",
  "choices": [ ... ],
  "usage": {
    "prompt_tokens": 1024,
    "completion_tokens": 142,
    "total_tokens": 1166,
    "prompt_tokens_details": {
      "cached_tokens": 512
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0
    }
  },
  "timing": {
    "queue_ms": 3,
    "prefill_ms": 42,
    "generation_ms": 802,
    "total_ms": 847,
    "time_to_first_token_ms": 45,
    "inter_token_latency_ms": 5.65,
    "tokens_per_second": 167.6
  }
}
```

For streaming responses, the `timing` and `usage` objects appear in the final `[DONE]`-preceding chunk:

```
data: {"id":"chatcmpl-abc123","choices":[{"delta":{},"finish_reason":"stop","index":0}],"usage":{...},"timing":{...}}

data: [DONE]
```

**Feature flag:** The `timing` field requires `metrics` feature. When disabled, only standard `usage` is returned. The `usage` field is always present (OpenAI API compat).

### 32.8 Periodic Log Summary

Following vLLM's pattern (§31.17), a summary line is logged every N seconds on a background timer.

**Configuration:**

```yaml
observability:
  logging:
    periodic_stats_interval_seconds: 10   # 0 to disable
```

**Format (structured JSON in production):**

```json
{
  "level": "INFO",
  "target": "onnx_genai::stats",
  "event": "periodic_stats",
  "interval_seconds": 10,
  "throughput": { "prompt_tps": 1234.5, "generation_tps": 567.8 },
  "requests": { "active": 8, "waiting": 2, "completed_interval": 12 },
  "kv_cache": { "usage_ratio": 0.873, "used": 1792, "total": 2048, "shared": 256, "offloaded": 64 },
  "prefix_cache": { "hit_rate": 0.732 },
  "speculation": { "acceptance_rate": 0.751, "mean_length": 3.42 },
  "evictions_interval": 3,
  "preemptions_interval": 0,
  "errors_interval": 0
}
```

**Pretty format (for development / `format: pretty`):**

```
INFO  [stats] Throughput: prompt=1234.5 tok/s, gen=567.8 tok/s. Running: 8, Waiting: 2. KV: 87.3% (1792/2048). PrefixCache: 73.2%. SpecDec: 75.1% accept, 3.42 mean. Evictions: 3, Preemptions: 0.
```

**Feature flag:** None (part of `logging`). Disable by setting interval to 0.

### 32.9 Security Model

Three tiers of endpoint protection:

| Tier | Endpoints | Auth | Network |
|------|-----------|------|---------|
| **Public** | `/metrics` | None | Configurable bind address |
| **API** | `/v1/status`, `/v1/chat/completions` | API key (`Authorization: Bearer`) | Any |
| **Debug** | `/v1/debug/*` | None (localhost) or API key (remote) | Localhost-only by default |

**Configuration:**

```yaml
observability:
  metrics:
    bind: "127.0.0.1:9090"     # separate bind for /metrics (optional)
    # If unset, /metrics is served on the main server port

  debug:
    enabled: true               # master switch for /v1/debug/*
    allow_remote: false          # if true, requires API key for /v1/debug/*
    # When allow_remote is false, debug endpoints reject non-loopback IPs
    # with 403 Forbidden regardless of auth headers
```

**Implementation (axum middleware):**

```rust
/// Middleware that restricts debug endpoints to localhost.
async fn localhost_only(
    req: Request<Body>,
    next: Next,
) -> Response {
    let remote_addr = req.extensions().get::<ConnectInfo<SocketAddr>>();
    match remote_addr {
        Some(ConnectInfo(addr)) if addr.ip().is_loopback() => next.run(req).await,
        _ => StatusCode::FORBIDDEN.into_response(),
    }
}

/// Debug route group with conditional auth.
pub fn debug_routes(config: &ObservabilityConfig) -> Router {
    let router = Router::new()
        .route("/v1/debug/sessions", get(list_sessions))
        .route("/v1/debug/kv", get(kv_cache_state))
        .route("/v1/debug/scheduler", get(scheduler_state))
        .route("/v1/debug/batch", get(current_batch_info))
        .route("/v1/debug/events", get(sse_events))
        .route("/v1/debug/trace/start", post(start_trace))
        .route("/v1/debug/trace/stop", post(stop_trace))
        .route("/v1/debug/log-level", put(set_log_level))
        .route("/v1/debug/config", get(dump_config));

    if config.debug.allow_remote {
        router.layer(middleware::from_fn(api_key_auth))
    } else {
        router.layer(middleware::from_fn(localhost_only))
    }
}
```

**Perfetto trace files:** Written to a temp directory with `0600` permissions. The `/v1/debug/trace/stop` response streams the file and then deletes it.

### 32.10 Graceful Degradation

When features are disabled at compile time or runtime:

| Feature off | Behavior |
|-------------|----------|
| `metrics` disabled | `/metrics` → `404`. `timing` field omitted from chat responses. Internal counters still run for periodic log stats. |
| `otel` disabled | No OTLP export. Prometheus still works if `metrics` is on. |
| `debug-endpoints` disabled | All `/v1/debug/*` → `404`. |
| `tracing` disabled | `trace_span!()` compiles to no-op. Perfetto endpoints → `404`. |
| `metrics` + `otel` both off | No metric collection overhead. Atomic counter increments are skipped. Periodic log line uses engine-internal rolling averages only. |

**Runtime disable:** `PUT /v1/debug/log-level` with `{"level": "off"}` suppresses periodic stats. Individual metric families cannot be disabled at runtime (not worth the complexity).

### 32.11 Memory Overhead

| Component | Steady-state memory | Notes |
|-----------|-------------------|-------|
| Prometheus registry | ~200 KB | All metric families + histogram buckets |
| Perfetto ring buffer | 0 (inactive) / 16 MB (active) | Configurable `buffer_size`, bounded |
| SSE event stream | ~0 (no buffer) | Write-through to clients |
| Per-request timestamps | ~128 bytes/request | Freed on completion |
| Prefix cache sliding window | ~24 KB | 1000 entries × 24 bytes |
| OTLP export buffer | ~1 MB | Batch buffer, bounded |

Total always-on overhead: **~250 KB**. With active tracing: **~17 MB** (dominated by Perfetto ring buffer).

### 32.12 Grafana Dashboard Compatibility

The metrics are designed to populate a standard Grafana dashboard with these panels:

**Row 1: Request Overview**
- Requests/sec: `rate(onnx_genai_requests_total[5m])`
- Error rate: `rate(onnx_genai_requests_total{status="error"}[5m]) / rate(onnx_genai_requests_total[5m])`
- Active requests: `onnx_genai_requests_active`
- Queue depth: `onnx_genai_requests_waiting`

**Row 2: Latency**
- TTFT p50/p95/p99: `histogram_quantile(0.95, rate(onnx_genai_time_to_first_token_seconds_bucket[5m]))`
- ITL p50/p95/p99: `histogram_quantile(0.95, rate(onnx_genai_inter_token_latency_seconds_bucket[5m]))`
- E2E latency: `histogram_quantile(0.95, rate(onnx_genai_e2e_latency_seconds_bucket[5m]))`
- Queue wait: `histogram_quantile(0.95, rate(onnx_genai_queue_wait_seconds_bucket[5m]))`

**Row 3: Throughput**
- Generation tokens/sec: `onnx_genai_generation_throughput`
- Prompt tokens/sec: `onnx_genai_prompt_throughput`
- Batch size distribution: `histogram_quantile(0.5, rate(onnx_genai_batch_size_bucket[5m]))`

**Row 4: KV Cache**
- KV utilization: `onnx_genai_kv_usage_ratio`
- Prefix cache hit rate: `onnx_genai_prefix_cache_hit_rate`
- Evictions/sec: `rate(onnx_genai_kv_evictions_total[5m])`
- Pages breakdown (stacked): `onnx_genai_kv_pages_used`, `onnx_genai_kv_pages_shared`, `onnx_genai_kv_pages_offloaded`

**Row 5: Speculation** (conditional)
- Acceptance rate: `onnx_genai_spec_acceptance_rate`
- Per-position acceptance: `onnx_genai_spec_acceptance_per_position{position="0"}` through `{position="4"}`
- Mean acceptance length: `onnx_genai_spec_mean_acceptance_length`

**Row 6: Sessions & Tools**
- Active sessions by priority: `onnx_genai_sessions_active{priority=~".*"}`
- Tool call rate: `rate(onnx_genai_tool_calls_total[5m])`
- Tool latency p95: `histogram_quantile(0.95, rate(onnx_genai_tool_latency_seconds_bucket[5m]))`

A reference dashboard JSON will ship as `dashboards/onnx-genai.json` in the repo.

### 32.13 Implementation: Axum Router Assembly

```rust
use axum::{Router, routing::{get, put, post}};

pub fn observability_routes(config: &Config) -> Router {
    let mut router = Router::new();

    // /v1/status — always on
    router = router.route("/v1/status", get(status_handler));

    // /metrics — Prometheus scrape
    #[cfg(feature = "metrics")]
    {
        router = router.route("/metrics", get(prometheus_handler));
    }

    // /v1/debug/* — debug introspection
    #[cfg(feature = "debug-endpoints")]
    if config.observability.debug.enabled {
        let debug = Router::new()
            .route("/v1/debug/sessions", get(sessions_handler))
            .route("/v1/debug/kv", get(kv_handler))
            .route("/v1/debug/scheduler", get(scheduler_handler))
            .route("/v1/debug/batch", get(batch_handler))
            .route("/v1/debug/events", get(sse_handler))
            .route("/v1/debug/config", get(config_handler))
            .route("/v1/debug/trace/start", post(trace_start_handler))
            .route("/v1/debug/trace/stop", post(trace_stop_handler))
            .route("/v1/debug/log-level", put(log_level_handler));

        let debug = if config.observability.debug.allow_remote {
            debug.layer(middleware::from_fn(api_key_auth))
        } else {
            debug.layer(middleware::from_fn(localhost_only))
        };

        router = router.merge(debug);
    }

    router
}
```

---

## 33. ORT vs llama.cpp: PC Deployment Gap Analysis

*From real user feedback on Foundry Local vs llama.cpp, 2026-07-12.*

### 33.1 The Problem

Users comparing Foundry Local (ORT-based) against llama.cpp for local PC deployment report:
1. **Higher VRAM usage** for the same quantized model
2. **Larger binary/DLL size** (ORT CUDA EP)

Both matter critically on consumer GPUs (8-12 GB VRAM) where every MB counts.

### 33.2 VRAM Gap Root Causes

| Factor | llama.cpp | ORT (CUDA EP) | Gap |
|---|---|---|---|
| **Quantized matmul** | Compressed-domain kernel: operates directly on Q4/Q8 weights, no dequantize | MatMulNBits likely dequantizes to FP16 → cuBLAS matmul → extra intermediate tensor | ORT allocates full FP16 weight tensor temporarily |
| **KV cache dtype** | Default FP16, supports Q8_0/Q4_0 KV | Likely FP32 KV cache by default | 2× KV memory if FP32 |
| **Memory allocator** | Custom slab allocator, tight control | ORT arena allocator, may over-provision | Fragmentation + headroom waste |
| **Op fusion** | Hand-fused attention + FFN kernels | Relies on graph optimizer for fusion, not always optimal | Missed fusions = extra intermediate buffers |
| **Context overhead** | Minimal runtime state | ORT session, graph state, EP state | Fixed overhead per model load |

### 33.3 Binary Size Gap Root Causes

| Component | llama.cpp | ORT (CUDA EP) |
|---|---|---|
| Core binary | ~5-10 MB | onnxruntime.dll ~200-500 MB |
| CUDA kernels | Only ~20 ops needed for transformers | Hundreds of ops compiled for CUDA |
| Dependencies | cuBLAS optional, no cuDNN | cuBLAS + cuDNN required |
| Op coverage | Transformer-only | General purpose (CNN, RNN, transformer, etc.) |
| Build granularity | Single target arch | Multi-arch, multi-EP support compiled in |

### 33.4 Implications for onnx-genai

**What we can control (our runtime layer):**
- Default KV cache to FP16 (not FP32)
- Implement KV cache quantization (FP8/Q8 as designed in §16)
- Aggressive memory reuse patterns in paged cache
- Profile and document actual VRAM breakdown per component

**What needs ORT-level changes (upstream):**
- Compressed-domain quantized matmul kernels (avoid dequantize round-trip)
- Build-time op stripping (only include ops used by loaded model)
- Reduced binary: `onnxruntime-genai-slim` with only transformer-relevant CUDA kernels
- Better arena allocator tuning defaults for genai workloads

**Where we have an advantage over llama.cpp:**
- **DirectML EP** — llama.cpp's DirectML/Vulkan support is weak. AMD/Intel GPU users get a poor experience with llama.cpp but good support from ORT
- **NPU support** — QNN EP for Snapdragon, CoreML for Apple Silicon. llama.cpp has no NPU path
- **Multi-EP** — same model, automatic fallback: CUDA → DirectML → CPU
- **Quantization variety** — ORT supports more quantization formats via ONNX ops
- **Enterprise features** — ONNX model signing, sandboxing, telemetry, compliance

### 33.5 Competitive Strategy for PC Deployment

```
Short term:
  - Don't compete with llama.cpp on CUDA-only VRAM efficiency (they win)
  - Win on DirectML (AMD/Intel GPUs) and NPU (Snapdragon/Apple Silicon)
  - Win on "just works" model loading (ONNX is standard format)

Medium term:
  - Push ORT team for compressed-domain kernels
  - Push for binary size reduction (op stripping, slim builds)
  - Our KV cache quantization helps close VRAM gap

Long term:
  - If ORT kernels catch up, our runtime + ORT = competitive everywhere
  - Enterprise features (security, compliance, telemetry) differentiate
  - Multi-modal pipeline support (not just LLM) is broader than llama.cpp
```

### 33.6 Tracking Metrics

Add specific metrics to track competitive position:

```
# Per-model VRAM breakdown
onnx_genai_vram_model_weights_bytes       gauge  # static weight memory
onnx_genai_vram_kv_cache_bytes            gauge  # KV cache memory
onnx_genai_vram_activations_bytes         gauge  # intermediate activation memory
onnx_genai_vram_ort_overhead_bytes        gauge  # ORT runtime overhead
onnx_genai_vram_total_bytes               gauge  # total GPU memory used

# Binary size (build-time, document in CI)
# onnxruntime.dll size, total deployment size
```

This gives users (and us) transparency into where memory goes — and helps identify optimization targets.
