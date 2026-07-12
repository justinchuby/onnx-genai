# onnx-genai Design Document

**Project:** onnx-genai — A Rust inference runtime for generative AI models  
**Author:** Justin Chu  
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
