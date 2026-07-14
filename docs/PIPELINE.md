# Pipeline & Generate — Transformers-Parity Convenience Layer

> A `transformers`-style task facade for nxrt. `pipeline("text-generation", model)` and a
> library-level `messages=[...]` chat API, layered as thin wrappers over the existing
> `onnx_genai_engine::Engine`. No new inference machinery — just HF-parity ergonomics.

**Scope:** This document covers the high-level convenience API (task facade, chat/messages,
`GenerationConfig`, Python surface). The low/mid-level generation stack (sampling, speculative
decoding, sessions, batching) is covered in [DESIGN.md](./DESIGN.md); the Python binding
mechanics are covered in [ORT2.md](./ORT2.md) §24 and [EAGER.md](./EAGER.md) §11.

---

## Table of Contents

1. [Status & Motivation](#1-status--motivation)
2. [What Exists Today](#2-what-exists-today)
3. [The Gap](#3-the-gap)
4. [Naming: Avoiding the `PipelineEngine` Collision](#4-naming-avoiding-the-pipelineengine-collision)
5. [Task Facade — Rust API](#5-task-facade--rust-api)
6. [Chat / Messages Layer](#6-chat--messages-layer)
7. [`GenerationConfig` ↔ `GenerateOptions` Mapping](#7-generationconfig--generateoptions-mapping)
8. [Feature-Extraction / Embeddings](#8-feature-extraction--embeddings)
9. [Fill-in-the-Middle](#9-fill-in-the-middle)
10. [Streaming](#10-streaming)
11. [Where It Lives (Crates & Modules)](#11-where-it-lives-crates--modules)
12. [Python Surface](#12-python-surface)
13. [Server De-duplication Plan](#13-server-de-duplication-plan)
14. [Design Decisions & Open Questions](#14-design-decisions--open-questions)

---

## 1. Status & Motivation

**Status:** Design proposal. No code exists for this layer yet. All lower-level primitives it
composes already ship.

Users coming from Hugging Face `transformers` expect two entry points:

```python
from transformers import pipeline
pipe = pipeline("text-generation", model="...")
pipe("Hello", max_new_tokens=64)
```

and

```python
out = model.generate(**inputs, max_new_tokens=64)
```

nxrt already has a generation stack that is arguably a *superset* of `model.generate`
(speculative decoding, constrained decoding, continuous batching, prefix caching, sessions).
What is missing is the **thin, task-typed convenience facade** on top, plus a **library-level
`messages` / chat API**. Chat templating exists but is currently reachable only through the
server crate. This document designs that facade so a library or Python user gets
transformers-grade ergonomics without reaching into `onnx-genai-server`.

---

## 2. What Exists Today

We are **layering, not rewriting**. Every type below is real and verified.

### 2.1 Engine (`crates/onnx-genai-engine/src/engine.rs`)

- `Engine::from_dir(model_dir: &Path, config: EngineConfig) -> anyhow::Result<Self>` — loads
  model + tokenizer + metadata from a directory (`engine.rs:124`). Also
  `from_dir_with_session_options` (`engine.rs:129`).
- `Engine::generate(&mut self, request: GenerateRequest) -> anyhow::Result<GenerateResult>`
  — the core call (`engine.rs:544`). Text prompts are tokenized **internally** via the **private**
  `tokenize_prompt` (`engine.rs:992`), so text-in already works **for generation**. There is no
  *public* tokenizer accessor, which is why feature-extraction needs an explicit seam (§8.1).
- `generate_with_callback(&mut self, request, callback: Option<&mut GenerateTokenCallback<'_>>)`
  — streaming (`engine.rs:591`).
- `generate_fim(&mut self, prefix, suffix, options)` (`engine.rs:563`) and
  `generate_fim_with_config` (`engine.rs:577`).
- Sessions: `create_session` / `reset_session` / `close_session` / `session_token_count`
  (`engine.rs:829, 856, 901, 920`), plus `generate_in_session[_with_priority|_with_callback]`
  (`engine.rs:607, 616, 626`).
- `Engine::embed(&mut self, input_ids: &[TokenId])` and `embed_with_options(..., EmbeddingOptions)`
  (`crates/onnx-genai-engine/src/embedding.rs:31, 36`).
- `Engine::metadata(&self) -> &InferenceMetadata` (`engine.rs:928`).
- Continuous batching: `ContinuousBatchManager` (`engine/src/batched.rs`, re-exported at
  `engine/src/lib.rs:26`).

### 2.2 Request / result types (`crates/onnx-genai-engine/src/config.rs`)

- `GeneratePrompt::{ Text(String), TokenIds(Vec<TokenId>) }` with `From<&str/String/Vec<TokenId>>`
  (`config.rs:221`).
- `GenerateOptions` (`config.rs:248`): `max_new_tokens, temperature, top_p, top_k, min_p,
  repetition_penalty, frequency_penalty, presence_penalty, greedy, seed, stop_sequences,
  eos_token_id, stop_on_eos, max_context, num_speculative_tokens, speculative_mode, constraint,
  top_logprobs`. `Default` at `config.rs:292` (`max_new_tokens: 128`, `greedy: true`).
- `GenerateConstraint::{ Json, JsonSchema(String), Regex(String), Lark(String) }` (`config.rs:319`).
- `GenerateRequest { prompt, options }` with `GenerateRequest::new(impl Into<GeneratePrompt>)`
  (`config.rs:444, 452`).
- `GenerateResult { text: String, token_ids: Vec<TokenId>, finish_reason: FinishReason,
  prefix_cache_hit_len: usize, logprobs: Option<Vec<TokenLogprob>> }` (`config.rs:497`).
  `text` is already **detokenized**.
- `FinishReason::{ MaxTokens, EosToken, StopSequence { index }, Length }` (`config.rs:484`).
- `GenerateToken { token_id, text, finish_reason: Option<FinishReason> }` (`config.rs:523`) and
  `GenerateTokenCallback<'a> = dyn FnMut(GenerateToken) -> anyhow::Result<()> + Send + 'a`
  (`config.rs:530`).

### 2.3 Embeddings (`crates/onnx-genai-engine/src/embedding.rs`)

- `EmbeddingPooling::{ Mean (default), LastToken }` (`embedding.rs:10`).
- `EmbeddingOptions { pooling, normalize, hidden_state_output }` (`embedding.rs:20`).
- Re-exported at `engine/src/lib.rs:27`.

### 2.4 Chat templating — **exists, but only at the ORT layer** (`crates/onnx-genai-ort/src/chat_template.rs`)

- `ChatTemplate` (`chat_template.rs:17`) with
  `ChatTemplate::from_model_dir(model_dir: &Path) -> Result<Self>` (`chat_template.rs:125`) and
  `render(&self, messages: &[ChatMessage], tools: Option<&str>, add_generation_prompt: bool) -> Result<String>`
  (`chat_template.rs:159`). It is a minijinja-based renderer.
- **`from_model_dir` ALWAYS returns a template.** It prefers a standalone `chat_template.jinja`
  (`chat_template.rs:126–131`), else the `chat_template` string inside `tokenizer_config.json`
  (`chat_template.rs:133–147`), and **otherwise falls back to a built-in `DEFAULT_CHAT_TEMPLATE`**
  (`chat_template.rs:149–151`). It never signals "no model template present." This matters for the
  server-parity fallback policy in §6 — the server deliberately behaves differently (§2.4.1).
- `ChatMessage { role, content, tool_calls: Option<Value> }` (`chat_template.rs:87–92`) with
  `::system/::user/::assistant` constructors (`chat_template.rs:103–113`) and
  `.with_tool_calls(Value)` (`chat_template.rs:115–118`). **`tool_calls` is a per-message input
  field** (assistant tool-call history), serialized inside each message. It is **distinct** from the
  separate `tools` render argument.
- `render` exposes exactly three template variables (`chat_template.rs:180–184`): `messages` (each
  carrying its own `tool_calls`), `tools` (the offered tool catalog, from the `tools: Option<&str>`
  JSON argument — `chat_template.rs:162`, `165–170`, `182`), and `add_generation_prompt`. There is
  **no** `tool_choice` and **no** `tool_call_id` render variable — those live only in the server's
  own fallback (§2.4.1).
- `ChatRole::{ System, User, Assistant, Tool, Other(String) }` (`chat_template.rs:23`),
  `From<&str>` + `Serialize`/`Deserialize`.
- Re-exported at `crates/onnx-genai-ort/src/lib.rs:27`.
- **Consumed today only by the server** (`render_prompt` at `routes.rs:2029`, calling
  `template.render(..., true)` at `routes.rs:2056–2058`; `load_chat_template` at `state.rs:282`). A
  *library* user of `Engine` must therefore hand-format the prompt or depend on the server crate.
  The message-conversion + template-rendering seam is the duplication we eliminate — **not** the
  server's request handling (§13).

#### 2.4.1 The server's template behavior differs from `ChatTemplate`'s default — on purpose

`load_chat_template` (`state.rs:282–297`) returns `Ok(None)` when **neither** a standalone
`chat_template.jinja` **nor** a `tokenizer_config.json` `chat_template` string exists — it does not
construct a `ChatTemplate` in that case. `render_prompt` (`routes.rs:2029–2061`) then branches: with
a template it converts OpenAI DTO messages → `ChatMessage` and calls `render(..., true)`; with
`None` it uses the server's own role-tagged fallback `build_prompt` (`routes.rs:2066–2101`).

That fallback carries request context the ORT template variables never expose: a `<|tools|>` block,
a `<|tool_choice|>` block (`routes.rs:2073–2077`), per-message `tool_call_id` lines
(`routes.rs:2082–2086`), and serialized message `tool_calls` (`routes.rs:2090–2096`). So swapping
`load_chat_template` for `ChatTemplate::from_model_dir` is **behavior-changing**: it would replace
"no template ⇒ server fallback with tool context" with "no template ⇒ ORT `DEFAULT_CHAT_TEMPLATE`,
dropping `tool_choice`/`tool_call_id`." §6 defines an explicit fallback policy that preserves the
server's current behavior.

### 2.5 The **other** pipeline — `PipelineEngine` (do not confuse)

`crates/onnx-genai-engine/src/pipeline.rs` defines `PipelineEngine` (`pipeline.rs:66`),
`PipelineGenerateRequest` (`pipeline.rs:28`), and the `PipelineTensors` type alias
(`pipeline.rs:25`). It **imports** `PipelineSpec` and `PipelineStrategy` from the metadata crate —
they are defined in `crates/onnx-genai-metadata/src/schema.rs` (`PipelineSpec` at `schema.rs:404`,
`PipelineStrategy` at `schema.rs:534`) and merely `use`d by `pipeline.rs` (`pipeline.rs:13–14`).
`PipelineEngine` is a **multi-model dataflow orchestrator** (e.g. vision-encoder → decoder, with
`component.input_name` endpoints driven by model metadata). It is **not** the HF task-typed
`pipeline("text-generation")` convenience. See §4 for how we keep the names distinct.

### 2.6 Umbrella re-exports (`crates/onnx-genai/src/lib.rs`)

The `onnx-genai` umbrella crate re-exports `Engine, EngineConfig, GenerateOptions,
GeneratePrompt, GenerateRequest, GenerateResult, GenerateToken, GenerateTokenCallback,
FinishReason, …`. It does **not** yet re-export `ChatTemplate`/`ChatMessage` (those live in
`onnx_genai_ort`, re-exported as `onnx_genai::ort::…`).

### 2.7 Python bindings — planned, not built

`docs/EAGER.md` §11–§12 and `docs/ORT2.md` §24 describe the PyO3/maturin binding layer living
in `bindings/python/`, exposing the module **`nxrt`** (e.g. `import nxrt; nxrt.tensor(...)`).
The GenAI convenience API designed here will attach to that same `nxrt` module.

---

## 3. The Gap

Primitives exist; the transformers-like convenience layer does not. Concretely we need:

1. A **task-typed facade** mirroring `transformers.pipeline(task, model)` — text/messages in,
   text out, a thin wrapper over `Engine`.
2. A **library-level chat/`messages` API** built on a shared **message-conversion + template
   rendering** seam, so the server and the library/Python facade share **one rendering
   implementation** (not one generation path — the server keeps its driver/session/SSE stack, §13).
3. **HF-parity ergonomics**: `from_pretrained`, a callable/`run`, and a `GenerationConfig` that
   maps onto the existing `GenerateOptions`.
4. A **Python surface** on `nxrt` that mirrors `transformers` 1:1.
5. **Streaming** surfaced through the facade (Rust callback/closure, Python iterator).

---

## 4. Naming: Avoiding the `PipelineEngine` Collision

`PipelineEngine` (§2.5) already owns the word "pipeline" for multi-model dataflow. To mirror
`transformers` without colliding:

- The **free function** is `pipeline(task, model_dir, options)` — this matches HF exactly and
  does not collide with the *type* `PipelineEngine`.
- The **trait** returned is **`TaskPipeline`** (not `Pipeline`), and concrete task types are
  suffixed `…Pipeline`: `TextGenerationPipeline`, `ChatPipeline`, `FeatureExtractionPipeline`,
  `FillInMiddlePipeline`.
- Everything lives in a new module **`onnx_genai_engine::task`** and is re-exported from the
  umbrella crate as `onnx_genai::task::*` and (for the common names) at the crate root.

Rationale: "task facade" and "dataflow engine" are different concepts. Prefixing the facade
types with the *task name* and grouping them under `task::` makes the distinction obvious at
every call site (`task::TextGenerationPipeline` vs `PipelineEngine`), while the free function
`pipeline(...)` preserves muscle memory from `transformers`.

---

## 5. Task Facade — Rust API

### 5.1 Task enum + dispatch

```rust
// crates/onnx-genai-engine/src/task/mod.rs

/// Supported high-level tasks, mirroring `transformers` task strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Task {
    /// `"text-generation"` → Engine::generate
    TextGeneration,
    /// `"chat"` / `"conversational"` → chat template + Engine::generate
    Chat,
    /// `"feature-extraction"` → Engine::embed(_with_options)
    FeatureExtraction,
    /// `"fill-in-the-middle"` → Engine::generate_fim
    FillInMiddle,
}

impl std::str::FromStr for Task {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<Self> {
        Ok(match s {
            "text-generation" => Task::TextGeneration,
            "chat" | "conversational" => Task::Chat,
            "feature-extraction" | "embeddings" => Task::FeatureExtraction,
            "fill-in-the-middle" | "fim" => Task::FillInMiddle,
            other => anyhow::bail!("unknown task {other:?}"),
        })
    }
}
```

The enum is kept small and extensible: future tasks (`fill-mask`, `token-classification`) add a
variant + a concrete `…Pipeline` type without touching existing ones.

### 5.2 The `TaskPipeline` trait and the `pipeline()` free function

```rust
/// Options shared by every task pipeline at construction time.
#[derive(Debug, Clone, Default)]
pub struct PipelineOptions {
    /// Engine/session configuration passed through to `Engine::from_dir`.
    pub engine: EngineConfig,
    /// Default generation config; per-call overrides merge on top (see §7).
    pub generation: GenerationConfig,
}

/// A callable, task-typed facade over `Engine`. Text (or messages) in → text out.
pub trait TaskPipeline {
    /// The natural per-call input for this task (String, messages, prefix/suffix…).
    type Input;
    /// The natural per-call output for this task (String, Vec<f32>, GenerateResult…).
    type Output;

    /// Run one request, merging `overrides` onto the pipeline's default config.
    fn run(&mut self, input: Self::Input, overrides: GenerationConfig)
        -> anyhow::Result<Self::Output>;
}

/// Construct a boxed task pipeline from a task string, mirroring
/// `transformers.pipeline(task, model)`. Returns an enum wrapper so callers can hold one
/// value regardless of task; downcast helpers expose the concrete type when needed.
pub fn pipeline(
    task: &str,
    model_dir: &Path,
    options: PipelineOptions,
) -> anyhow::Result<AnyPipeline>;

/// Type-erased holder returned by `pipeline()`, since tasks have different in/out types.
pub enum AnyPipeline {
    TextGeneration(TextGenerationPipeline),
    Chat(ChatPipeline),
    FeatureExtraction(FeatureExtractionPipeline),
    FillInMiddle(FillInMiddlePipeline),
}
```

### 5.3 Concrete pipelines with `from_pretrained` + a callable

Each concrete type owns the `Engine`, the `model_dir` (needed to load the chat template lazily),
and the default `GenerationConfig`.

```rust
pub struct TextGenerationPipeline {
    engine: Engine,
    model_dir: PathBuf,
    defaults: GenerationConfig,
}

impl TextGenerationPipeline {
    /// HF-style constructor.
    pub fn from_pretrained(model_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        Self::from_pretrained_with(model_dir, PipelineOptions::default())
    }
    pub fn from_pretrained_with(
        model_dir: impl AsRef<Path>,
        options: PipelineOptions,
    ) -> anyhow::Result<Self>;

    /// The callable. `text` may be `&str`/`String`/`Vec<TokenId>` (via `Into<GeneratePrompt>`).
    /// Returns the full `GenerateResult`; `.text` is the detokenized string.
    pub fn generate(
        &mut self,
        prompt: impl Into<GeneratePrompt>,
        overrides: GenerationConfig,
    ) -> anyhow::Result<GenerateResult>;

    /// Streaming variant (see §10).
    pub fn generate_stream(
        &mut self,
        prompt: impl Into<GeneratePrompt>,
        overrides: GenerationConfig,
        on_token: &mut GenerateTokenCallback<'_>,
    ) -> anyhow::Result<GenerateResult>;
}

impl TaskPipeline for TextGenerationPipeline {
    type Input = GeneratePrompt;
    type Output = GenerateResult;
    fn run(&mut self, input: GeneratePrompt, overrides: GenerationConfig)
        -> anyhow::Result<GenerateResult> {
        self.generate(input, overrides)
    }
}
```

Rust has no `__call__`; `run`/`generate` are the idiomatic equivalent. The Python binding
(§12) wires these to `__call__`.

---

## 6. Chat / Messages Layer

We extract exactly **one seam** from the server: **message construction + template discovery +
rendering** (with an explicit fallback policy). Everything the server layers *around* that seam —
OpenAI DTO mapping, request constraints, token counting/context caps, session selection, multimodal
dispatch, the async `EngineDriver`, SSE buffering, and output tool-call parsing — **stays in the
server** (§13). Two independent consumers then share the seam: the server (which keeps its own
driver/session/SSE path) and the `ChatPipeline` facade (which composes the seam with a direct
`Engine::generate`). Neither replaces the other.

### 6.1 Where it lives

New module **`onnx_genai_engine::chat`**. The engine crate already depends on `onnx_genai_ort`
(the engine's `embedding.rs:6` already imports `onnx_genai_ort` types), so it can use
`onnx_genai_ort::{ChatTemplate, ChatMessage, ChatRole}` directly — no new crate is warranted.

### 6.2 The shared seam: discovery + rendering (with fallback signaling)

The seam must be able to report **"no model template present"** so the server can keep applying its
own tool-aware fallback (§2.4.1), instead of silently adopting `ChatTemplate`'s
`DEFAULT_CHAT_TEMPLATE`. That is the crux of preserving current server behavior.

```rust
// crates/onnx-genai-engine/src/chat/mod.rs
use onnx_genai_ort::{ChatMessage, ChatTemplate};

/// Discover a model chat template *without* falling back to a built-in default.
///
/// Returns `Ok(None)` when neither a standalone `chat_template.jinja` nor a
/// `tokenizer_config.json` `chat_template` string exists — mirroring the server's
/// `state.rs:282` `load_chat_template` semantics so the server (and any caller that
/// wants its own fallback) can detect "no model template" instead of getting ORT's
/// `DEFAULT_CHAT_TEMPLATE`. Distinct from `ChatTemplate::from_model_dir`, which
/// *always* returns a template (`chat_template.rs:149–151`).
pub fn discover_chat_template(model_dir: &Path) -> anyhow::Result<Option<ChatTemplate>>;

/// Explicit policy for what to do when a model ships no chat template.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TemplateFallback {
    /// Error out with an actionable message (RULES.md #1). The library facade default:
    /// a chat model without any template is a configuration problem worth surfacing.
    #[default]
    Error,
    /// Render with ORT's built-in default template (opt-in convenience), obtained via the
    /// new public `ChatTemplate::builtin_default()` accessor (§6.2.1) — *not* by reaching
    /// into ORT's private `DEFAULT_CHAT_TEMPLATE` const.
    Default,
    /// Report "no model template"; the *caller* renders its own fallback. This is the
    /// mode the server selects so it keeps `build_prompt` (`routes.rs:2066`) with its
    /// `tools` / `tool_choice` / `tool_call_id` / message `tool_calls` context intact.
    CallerHandles,
}

/// Chat-specific knobs layered on top of `GenerateOptions`.
#[derive(Debug, Clone)]
pub struct ChatRenderOptions {
    /// The **offered tool catalog** — serialized JSON exposed to the template as the
    /// separate `tools` variable (`chat_template.rs:162,182`). This is NOT the per-message
    /// `tool_calls` history, and NOT `tool_choice`.
    pub tools: Option<String>,
    /// Append the assistant generation prompt (HF `add_generation_prompt`). Default true.
    pub add_generation_prompt: bool,
}

impl Default for ChatRenderOptions {
    fn default() -> Self {
        Self { tools: None, add_generation_prompt: true }
    }
}

/// Outcome of rendering: either a prompt string, or an explicit "no model template" signal
/// so the caller can apply its own fallback.
pub enum RenderedPrompt {
    Prompt(String),
    NoModelTemplate,
}

/// Render `messages` with a discovered template, honoring `fallback`. Each `ChatMessage`
/// carries its own per-message `tool_calls` inside `messages`; `options.tools` is the
/// separate offered-catalog variable. Neither is `tool_choice` (a server request-policy
/// concept with no template variable — see §2.4).
pub fn render_messages(
    template: Option<&ChatTemplate>,
    messages: &[ChatMessage],
    options: &ChatRenderOptions,
    fallback: TemplateFallback,
) -> anyhow::Result<RenderedPrompt>;
```

Notes:

- `discover_chat_template` is the **shared** replacement for the server's `load_chat_template`
  (§13); it returns `None` on absence rather than an ORT default, so both the server and the facade
  agree on precedence in exactly one place.
- `render_messages` never tokenizes and never calls the engine. It only turns messages + a template
  (or a fallback decision) into a prompt string. Tokenization, options, sessions, and dispatch are
  the *caller's* job (the server's driver path, or the facade's `Engine::generate`).
- Under `TemplateFallback::CallerHandles`, `render_messages` returns `NoModelTemplate` when
  `template` is `None`; the server then routes to its own `build_prompt`. Under `Error`, it produces
  an actionable message per RULES.md #1 (`chat_template.rs`-style context: which model dir, that
  neither `chat_template.jinja` nor a `tokenizer_config.json` `chat_template` was found, and how to
  supply one).

### 6.2.1 Required ort-crate change: expose the built-in default template

`TemplateFallback::Default` renders with ORT's built-in default when a model ships no template.
That default currently lives **inside** the ort crate and is unreachable from the engine crate,
so the seam as drawn is not buildable without a small ort-crate addition:

- `DEFAULT_CHAT_TEMPLATE` is a **private** module const (`chat_template.rs:12`).
- `ChatTemplate`'s `template` field is **private** (`chat_template.rs:17–19`), and the only
  constructor is `ChatTemplate::from_model_dir` (`chat_template.rs:125`), which requires a model
  directory on disk and only yields the default as a side effect of finding *nothing*
  (`chat_template.rs:149–151`). There is **no** public string/`Default` constructor.

The design therefore requires **one minimal, explicit ort-crate addition** — the same
"engine/ort-crate change required" convention this doc uses for the §8.1 tokenization seam:

```rust
// crates/onnx-genai-ort/src/chat_template.rs (new public accessor on ChatTemplate)
impl ChatTemplate {
    /// A `ChatTemplate` backed by ORT's built-in default (`DEFAULT_CHAT_TEMPLATE`).
    /// Model-independent — needs no model directory.
    pub fn builtin_default() -> Self;
}
```

The built-in default is **model-independent** — it only emits `role: content` lines plus an
optional generation prompt (`chat_template.rs:12–13`) — so a zero-argument accessor is
sufficient and `render_messages` does **not** need a `model_dir` parameter to serve
`TemplateFallback::Default`. Under that fallback with `template == None`, `render_messages`
calls `ChatTemplate::builtin_default()` and renders through the existing `ChatTemplate::render`
(`chat_template.rs:159`). The three fallback branches are distinct: `Default` renders ORT's
built-in default via `ChatTemplate::builtin_default()` and `ChatTemplate::render`; `Error` returns
an actionable error per RULES.md #1; and `CallerHandles` returns
`RenderedPrompt::NoModelTemplate`, after which the caller (for example, the server) supplies its
own prompt build via `build_prompt` (§13.2) without going through `ChatTemplate::render`.

### 6.3 `ChatPipeline` — the facade consumer of the seam

`ChatPipeline` is the HF-facade type. It composes the shared renderer with a **direct**
`Engine::generate` / `generate_with_callback` (`engine.rs:544`, `591`). It is a *synchronous*,
single-process consumer — it does **not** and cannot stand in for the server's async
driver/session/SSE path (§13).

```rust
pub struct ChatPipeline {
    engine: Engine,
    template: Option<ChatTemplate>,   // via chat::discover_chat_template at construction
    fallback: TemplateFallback,       // facade default: Error (see §6.2)
    defaults: GenerationConfig,
}

impl ChatPipeline {
    pub fn from_pretrained(model_dir: impl AsRef<Path>) -> anyhow::Result<Self>;
    pub fn from_pretrained_with(model_dir: impl AsRef<Path>, options: PipelineOptions)
        -> anyhow::Result<Self>;

    /// messages in → assistant `GenerateResult` out. Renders via the shared seam, then
    /// runs `Engine::generate` directly.
    pub fn chat(
        &mut self,
        messages: &[ChatMessage],
        render: ChatRenderOptions,
        overrides: GenerationConfig,
    ) -> anyhow::Result<GenerateResult> {
        let prompt = match chat::render_messages(
            self.template.as_ref(), messages, &render, self.fallback,
        )? {
            RenderedPrompt::Prompt(p) => p,
            // With the facade's default `Error` fallback this arm is unreachable; a caller
            // that opts into `CallerHandles` supplies its own fallback before this point.
            RenderedPrompt::NoModelTemplate => anyhow::bail!(
                "no chat template for this model; set a fallback or provide chat_template.jinja"
            ),
        };
        // Tokenize once via the §8.1 seam so we know prompt_len for max_length handling,
        // then reuse the ids as the prompt (avoids a second tokenization in the engine).
        let token_ids = self.engine.tokenize(&prompt)?;         // §8.1 public seam
        let prompt_len = token_ids.len();
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(token_ids));
        // Merge per-call overrides onto pipeline defaults, then lower to engine options.
        request.options = overrides.lower_onto(self.base_options(), prompt_len)?; // §7.2
        self.engine.generate(request)
    }

    pub fn chat_stream(
        &mut self,
        messages: &[ChatMessage],
        render: ChatRenderOptions,
        overrides: GenerationConfig,
        on_token: &mut GenerateTokenCallback<'_>,
    ) -> anyhow::Result<GenerateResult>;  // forwards to Engine::generate_with_callback
}
```

The facade does **not** parse tool calls out of the model output, does not buffer for stop
boundaries, and does not emit SSE — those are OpenAI-endpoint concerns that remain server-owned
(§3-blocking-finding-3, §13). A library user who wants tool-call parsing composes it themselves on
top of `GenerateResult.text`.

### 6.4 Defaults sourced from the model directory

- **EOS**: the engine already fills `options.eos_token_id` from the tokenizer when unset
  (`engine.rs:650–651`; FIM path at `engine.rs:938–944`), so chat inherits correct stop behavior
  for free.
- **`generation_config.json` / `genai_config.json`**: `Engine::from_dir` already derives inference
  metadata from `genai_config.json` when `inference_metadata.yaml` is absent (`engine.rs:154–163`).
  The facade may *additionally* read HF-style `generation_config.json` (if present) into the default
  `GenerationConfig` (temperature, top_p, top_k, repetition_penalty, max_new_tokens). See §7.4 and
  OQ-3.
- **`add_generation_prompt`**: default `true` — matching the server, which hard-codes `true` at
  `routes.rs:2057`.
- **Message `tool_calls` vs offered `tools` vs `tool_choice` vs `tool_call_id`** — four distinct
  concepts, do not conflate:
  - **message `tool_calls`** (`chat_template.rs:91`, `115–118`): assistant tool-call *history*
    attached to a specific message via `ChatMessage::with_tool_calls`; serialized inside `messages`.
  - **offered `tools`** (`chat_template.rs:162`, `182`): the tool *catalog* the model may call,
    passed as the separate `tools` render argument (`ChatRenderOptions::tools`).
  - **`tool_choice`**: a request *policy* (auto/required/specific). It has **no** template variable;
    the server enforces it as a decoding constraint (`routes.rs:1913–1915`, `1919`) and only prints
    it in its own fallback (`routes.rs:2073–2077`). It stays server-side.
  - **`tool_call_id`**: correlates a `role:"tool"` result message back to a prior call; also only in
    the server fallback (`routes.rs:2082–2086`). Server-side.

  The earlier claim that `with_tool_calls(Value)` "carries tool calls into the template's `tools`
  variable" was **wrong**: `with_tool_calls` populates a *per-message* field inside `messages`; the
  `tools` variable is a wholly separate render argument.

---

## 7. `GenerationConfig` ↔ `GenerateOptions` Mapping

### 7.1 The HF-parity config

```rust
/// HF-parity generation config. Every field is optional so per-call overrides merge cleanly
/// onto the pipeline defaults (missing = "inherit"). Lowered to `GenerateOptions` before use.
#[derive(Debug, Clone, Default)]
pub struct GenerationConfig {
    pub max_new_tokens: Option<usize>,
    pub max_length: Option<usize>,        // HF total length; converted vs prompt_len (§7.3)
    pub do_sample: Option<bool>,          // HF; maps to `greedy = !do_sample`
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,               // nxrt extension beyond HF
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,   // OpenAI-style extension
    pub presence_penalty: Option<f32>,    // OpenAI-style extension
    pub seed: Option<u64>,
    pub stop: Option<Vec<String>>,        // HF `stop_strings` → StopSequence::Text (APPEND, §7.2)
    pub stop_replace: bool,               // if true, `stop` replaces inherited stops (§7.2)
    pub eos_token_id: Option<TokenId>,
    pub top_logprobs: Option<usize>,
    pub constraint: Option<GenerateConstraint>, // nxrt extension (JSON/schema/regex/grammar)

    // Native nxrt knobs kept reachable through the facade (no HF equivalent) — see §7.5.
    pub stop_on_eos: Option<bool>,             // GenerateOptions::stop_on_eos override
    pub max_context: Option<usize>,            // GenerateOptions::max_context override
    pub num_speculative_tokens: Option<usize>, // speculative-decoding draft length
    pub speculative_mode: Option<SpeculativeMode>,
}
```

`stop_replace` is a plain `bool` (default `false` = append) rather than `Option<bool>` because its
only role is to modify how `stop` merges; a missing `stop` makes it a no-op.

### 7.2 Lowering

```rust
impl GenerationConfig {
    /// Merge `self` onto `base`, then lower to engine `GenerateOptions`.
    ///
    /// `prompt_len` is the tokenized prompt length, needed to convert HF `max_length`
    /// (total) into the engine's native `max_new_tokens`.
    pub fn lower_onto(&self, mut base: GenerateOptions, prompt_len: usize)
        -> anyhow::Result<GenerateOptions> {
        if let Some(v) = self.temperature        { base.temperature = v; }
        if let Some(v) = self.top_p              { base.top_p = v; }
        if let Some(v) = self.top_k              { base.top_k = v; }
        if let Some(v) = self.min_p              { base.min_p = v; }
        if let Some(v) = self.repetition_penalty { base.repetition_penalty = v; }
        if let Some(v) = self.frequency_penalty  { base.frequency_penalty = v; }
        if let Some(v) = self.presence_penalty   { base.presence_penalty = v; }
        if let Some(v) = self.seed               { base.seed = Some(v); }
        if let Some(v) = self.eos_token_id       { base.eos_token_id = Some(v); }
        if let Some(v) = self.top_logprobs       { base.top_logprobs = Some(v); }
        if let Some(c) = &self.constraint        { base.constraint = Some(c.clone()); }

        // Native nxrt overrides preserved (see §7.5) — not part of HF, but accessible.
        if let Some(v) = self.stop_on_eos        { base.stop_on_eos = v; }
        if let Some(v) = self.max_context        { base.max_context = Some(v); }
        if let Some(v) = self.num_speculative_tokens { base.num_speculative_tokens = Some(v); }
        if let Some(m) = self.speculative_mode   { base.speculative_mode = Some(m); }

        // `stop`: APPEND to inherited stop sequences by default (union semantics), so a
        // per-call `stop` never silently drops model/default stops. `stop_replace = true`
        // opts into replacing them wholesale.
        if let Some(stop) = &self.stop {
            if self.stop_replace {
                base.stop_sequences.clear();
            }
            base.stop_sequences.extend(stop.iter().cloned().map(StopSequence::Text));
        }

        // do_sample ↔ greedy (HF `do_sample=False` == greedy argmax).
        if let Some(do_sample) = self.do_sample { base.greedy = !do_sample; }

        // max_new_tokens takes precedence; else derive from max_length with validation.
        if let Some(v) = self.max_new_tokens {
            base.max_new_tokens = v;
        } else if let Some(total) = self.max_length {
            if total <= prompt_len {
                // RULES.md #1: what failed, why, and how to fix — never silently clamp to 1.
                anyhow::bail!(
                    "max_length ({total}) must be greater than the prompt length ({prompt_len}) \
                     so at least one new token can be generated. Increase max_length to at least \
                     {} (prompt_len + desired new tokens), or set max_new_tokens directly to say \
                     how many tokens to generate on top of the prompt.",
                    prompt_len + 1
                );
            }
            base.max_new_tokens = total - prompt_len;
        }
        Ok(base)
    }
}
```

### 7.3 `max_length` vs `max_new_tokens`

`transformers` historically defaulted to `max_length` (total). nxrt's native knob is
`max_new_tokens` (`config.rs:250`). We accept **both**: `max_new_tokens` wins; otherwise we derive
`max_new_tokens = max_length - prompt_len`. **If `max_length <= prompt_len` we reject the request
with an actionable error** (§7.2) rather than silently forcing a single new token — a prompt that
already meets or exceeds the requested total is a caller mistake worth surfacing (RULES.md #1). New
code should prefer `max_new_tokens`.

### 7.4 HF knobs: add vs skip

| HF `GenerationConfig` field | Decision | Mapping / rationale |
|---|---|---|
| `max_new_tokens` | **keep** | 1:1 → `GenerateOptions::max_new_tokens` |
| `max_length` | **add (convert)** | derive `max_new_tokens` from prompt length (§7.3) |
| `do_sample` | **add** | `greedy = !do_sample` |
| `temperature`, `top_p`, `top_k` | **keep** | 1:1 |
| `repetition_penalty` | **keep** | 1:1 |
| `stop_strings` / `stop` | **keep** | → `StopSequence::Text`, **appended** to inherited stops unless `stop_replace` (§7.2) |
| `eos_token_id` | **keep** | 1:1 (else tokenizer default, `engine.rs:650–651`) |
| `min_p` | **keep (extension)** | nxrt supports it; HF has it too now |
| `frequency_penalty`, `presence_penalty` | **keep (extension)** | OpenAI-style, engine supports |
| `stop_on_eos`, `max_context`, `num_speculative_tokens`, `speculative_mode` | **keep (native)** | no HF equivalent; kept reachable through the facade (§7.5) |
| `num_return_sequences` | **skip v1** | engine returns one sequence per call; emulate later by looping with distinct seeds (OQ-2) |
| `num_beams`, `length_penalty`, `early_stopping` | **skip** | engine is sampling-based; no beam search |
| `bad_words_ids`, `suppress_tokens`, `forced_*_token_id` | **skip v1** | expressible later as `LogitProcessor`s |
| `pad_token_id` | **skip** | single-sequence facade; batching handles padding internally |
| `penalty_alpha` (contrastive) | **skip** | unsupported decoding strategy |
| `constraint` (JSON/schema/regex/grammar) | **keep (extension)** | nxrt-only power feature via `GenerateConstraint` |

### 7.5 Native nxrt knobs stay reachable

`GenerateOptions` carries knobs with no HF `GenerationConfig` analogue: `stop_on_eos` (default
`true`, field at `config.rs:274`, default at `config.rs:307`), `max_context` (`config.rs:277`),
`num_speculative_tokens`
(`config.rs:279`), and `speculative_mode` (`config.rs:281`). Rather than lose them behind the
HF-parity surface, `GenerationConfig` exposes optional overrides for each (§7.1) that lower straight
onto `GenerateOptions` (§7.2). Left unset, they inherit whatever the engine/model defaults provide
(the server sets `max_context` from model metadata at `routes.rs:1128`/`1280`).

## 8. Feature-Extraction / Embeddings

`FeatureExtractionPipeline` wraps `Engine::embed` / `embed_with_options`.

```rust
pub struct FeatureExtractionPipeline {
    engine: Engine,
    defaults: EmbeddingOptions,
}

impl FeatureExtractionPipeline {
    pub fn from_pretrained(model_dir: impl AsRef<Path>) -> anyhow::Result<Self>;

    /// Text (tokenized via the new seam) or token ids → pooled embedding vector.
    pub fn embed(&mut self, input: impl Into<GeneratePrompt>) -> anyhow::Result<Vec<f32>>;
    pub fn embed_with_options(
        &mut self,
        input: impl Into<GeneratePrompt>,
        options: EmbeddingOptions,
    ) -> anyhow::Result<Vec<f32>>;
}
```

### 8.1 The missing tokenization seam (must be added — not hand-waved)

`Engine::embed` / `embed_with_options` accept **only** `&[TokenId]` (`embedding.rs:31`, `36`), and
there is currently **no public way to tokenize text through the engine**:

- `Engine::tokenize_prompt` is **private** (`engine.rs:992`, no `pub`).
- The engine's `tokenizer` field is `pub(crate)` (`engine.rs:74`) — there is **no** public
  `Engine::tokenizer()` accessor.

So "reuse the tokenizer path the engine already owns" is not possible from outside the crate today.
We must add a real seam. Two options, in preference order:

1. **Preferred — an `Engine::embed_text` method** (engine crate, next to `embed`): tokenizes with
   the engine's own tokenizer, then calls the existing hidden-state path. This keeps tokenization
   private and gives the facade a single call:

   ```rust
   // crates/onnx-genai-engine/src/embedding.rs (new methods on Engine)
   impl Engine {
       /// Tokenize `text` with the model's tokenizer, then embed.
       pub fn embed_text(&mut self, text: &str) -> anyhow::Result<Vec<f32>>;
       pub fn embed_text_with_options(
           &mut self, text: &str, options: EmbeddingOptions,
       ) -> anyhow::Result<Vec<f32>>;
   }
   ```

2. **Alternative — expose tokenization directly**: make `tokenize_prompt` `pub` (or add
   `pub fn tokenize(&self, text: &str) -> anyhow::Result<Vec<TokenId>>`) so the facade tokenizes
   then calls `embed`. This is also what the §7.3 `max_length` conversion and the `ChatPipeline`
   `prompt_len` computation need, so a public `Engine::tokenize` is the more broadly useful seam.

Either way the seam is an **explicit engine-crate change**, not an existing capability. §8's `embed`
callable dispatches on `GeneratePrompt`: `TokenIds` → `Engine::embed_with_options` unchanged; `Text`
→ the new `embed_text_with_options` (or `tokenize` + `embed`). Pooling
(`EmbeddingPooling::{Mean, LastToken}`, `embedding.rs:10`) and `normalize` are surfaced through
`EmbeddingOptions` (`embedding.rs:20`).

---

## 9. Fill-in-the-Middle

`FillInMiddlePipeline` wraps `Engine::generate_fim` (`engine.rs:563`).

```rust
pub struct FillInMiddlePipeline { engine: Engine, defaults: GenerationConfig }

impl FillInMiddlePipeline {
    pub fn from_pretrained(model_dir: impl AsRef<Path>) -> anyhow::Result<Self>;

    /// prefix + suffix → generated middle. Requires the model's tokenizer_config.json to
    /// declare FIM tokens (else `generate_fim` errors, per engine.rs:569–572).
    pub fn infill(
        &mut self,
        prefix: impl AsRef<str>,
        suffix: impl AsRef<str>,
        overrides: GenerationConfig,
    ) -> anyhow::Result<GenerateResult>;
}
```

---

## 10. Streaming

The engine's streaming primitive is `generate_with_callback(request, Option<&mut
GenerateTokenCallback<'_>>)` where `GenerateTokenCallback = dyn FnMut(GenerateToken) ->
anyhow::Result<()> + Send` (`config.rs:530`). Each `GenerateToken { token_id, text,
finish_reason }` is delivered incrementally; returning `Err` aborts generation.

- **Rust:** every `…_stream` method takes `on_token: &mut GenerateTokenCallback<'_>` and forwards
  to `Engine::generate_with_callback` (`engine.rs:591`). No new streaming machinery. This is a
  **synchronous, in-process** callback — it is *not* the server's async SSE path. The server does
  not use this facade callback: it streams via the async `EngineDriver` (`DriverEvent::Token`,
  `routes.rs:1332`) with its own stop-boundary buffering and tool-call detection (§13). The two
  streaming surfaces are independent consumers.
- **Python:** exposed as an **iterator**. The binding runs generation on a worker thread, the
  callback pushes each `GenerateToken.text` into a bounded channel, and the Python-side
  `TextIteratorStreamer`-style object yields chunks. Mirrors `transformers`'
  `TextIteratorStreamer`:

```python
for chunk in pipe("Hello", max_new_tokens=64, stream=True):
    print(chunk, end="", flush=True)
```

---

## 11. Where It Lives (Crates & Modules)

**No new crate.** The facade composes `Engine` (engine crate) and `ChatTemplate` (ort crate),
and the engine crate already depends on the ort crate.

```
crates/onnx-genai-engine/src/
├── task/
│   └── mod.rs        # Task, TaskPipeline, AnyPipeline, pipeline(), *Pipeline types
├── chat/
│   └── mod.rs        # shared seam: discover_chat_template, render_messages,
│                     #   ChatRenderOptions, TemplateFallback, RenderedPrompt
│                     #   (rendering ONLY — no engine/driver/SSE logic)
├── embedding.rs      # + new Engine::embed_text / Engine::tokenize seam (§8.1)
└── lib.rs            # pub mod task; pub mod chat;
                      # pub use task::{pipeline, Task, TaskPipeline, TextGenerationPipeline, ...};
                      # pub use chat::{discover_chat_template, render_messages,
                      #                ChatRenderOptions, TemplateFallback, RenderedPrompt};

crates/onnx-genai/src/lib.rs   # umbrella re-exports:
                      # pub use onnx_genai_engine::task::{...};
                      # pub use onnx_genai_engine::chat::{...};
                      # pub use onnx_genai_ort::{ChatMessage, ChatRole, ChatTemplate};
```

- **`chat`** is a sibling of `embedding` (same layering: an engine module that uses ort types). It
  is a **pure rendering seam** — it does not own the engine, the driver, sessions, or SSE.
- **`task`** depends on `chat` for `ChatPipeline` (the facade consumer, §6.3).
- The **server** (`onnx-genai-server`) is the *other* consumer: it swaps its `state.rs:282`
  `load_chat_template` for `chat::discover_chat_template` and its `routes.rs:2029` message-building
  loop for `chat::render_messages`, but keeps everything else (§13).
- The umbrella crate additionally re-exports `ChatMessage/ChatRole/ChatTemplate` from ort so library
  users get the whole surface from `onnx_genai::…` without naming the ort crate.

---

## 12. Python Surface

The GenAI convenience API attaches to the planned **`nxrt`** module (§2.7). Package name
**`nxrt` is already decided** (`docs/EAGER.md` §11–12, `docs/ORT2.md` §24). To keep the runtime
(`nxrt.tensor`, `nxrt.ops`) and the GenAI layer clearly separated we default to a **`nxrt.genai`
submodule**, while also re-exporting the two headline entry points (`pipeline`, `generate`) at
`nxrt` top level for exact `transformers` parity. (Top-level-only vs submodule is OQ-4.)

### 12.1 Sketch

```python
import nxrt

# transformers.pipeline parity
pipe = nxrt.pipeline("text-generation", "path/to/model")
out = pipe("Hello", max_new_tokens=64, temperature=0.7)   # __call__
print(out.text)

# chat / messages
chat = nxrt.pipeline("chat", "path/to/model")
reply = chat.chat(messages=[
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Hi"},
], max_new_tokens=128)

# embeddings / feature-extraction
emb = nxrt.pipeline("feature-extraction", "path/to/model")
vec = emb("some text")                       # -> list[float]

# fill-in-the-middle
fim = nxrt.pipeline("fill-in-the-middle", "path/to/model")
mid = fim(prefix="def add(a, b):\n    return ", suffix="\n")

# functional generate (model.generate parity)
res = nxrt.generate("path/to/model", "Hello", max_new_tokens=32)

# streaming (iterator, TextIteratorStreamer-style)
for chunk in pipe("Tell me a story", max_new_tokens=200, stream=True):
    print(chunk, end="", flush=True)
```

### 12.2 Parity table

| `transformers` | `nxrt` |
|---|---|
| `pipeline("text-generation", model)` | `nxrt.pipeline("text-generation", model)` |
| `pipe("Hello", max_new_tokens=64)` | `pipe("Hello", max_new_tokens=64)` |
| `pipeline("conversational" / chat)` | `nxrt.pipeline("chat", model)` |
| `pipe(messages)` / `apply_chat_template` | `chat.chat(messages=[...])` |
| `pipeline("feature-extraction", model)` | `nxrt.pipeline("feature-extraction", model)` |
| `model.generate(**inputs, ...)` | `nxrt.generate(model, prompt, ...)` |
| `GenerationConfig(...)` | kwargs on the call, or `nxrt.GenerationConfig(...)` |
| `TextIteratorStreamer` | `pipe(..., stream=True)` → Python iterator |
| `do_sample=True/False` | `do_sample=True/False` (→ `greedy`) |
| `max_length` / `max_new_tokens` | both accepted (§7.3) |
| _n/a_ | `response_format` / grammar via `constraint=` (nxrt extension) |

### 12.3 Rust → Python return-type mapping

| Rust | Python |
|---|---|
| `GenerateResult` | object with `.text: str`, `.token_ids: list[int]`, `.finish_reason: str`, `.prefix_cache_hit_len: int`, `.logprobs: Optional[list]` |
| `FinishReason` | `str` (`"max_tokens"`, `"eos_token"`, `"stop_sequence"`, `"length"`) |
| `Vec<f32>` (embedding) | `list[float]` (or numpy array when numpy available) |
| `GenerateToken` (stream) | yielded `str` (`.text`); full object opt-in |
| `anyhow::Error` | raised `nxrt.GenAIError` |

The Rust design and the Python surface are intentionally 1:1: `pipeline()` ↔ `nxrt.pipeline`,
`GenerationConfig` fields ↔ Python kwargs, `…_stream` callback ↔ Python iterator.

### 12.4 Wheel build & ABI policy (mandatory)

The GenAI Python surface ships in the same `nxrt` wheels as the runtime, under the constraints
already set for the binding layer (`docs/EAGER.md` §11–12, `docs/ORT2.md` §24). Concretely:

- **Minimum Python: 3.10.** `nxrt` requires Python **≥ 3.10** (`requires-python = ">=3.10"`). No
  support for 3.9 or earlier.
- **Standard wheels use the stable ABI (`abi3`).** We ship a **single `abi3` wheel** tagged
  **`cp310-abi3`**. In a stable-ABI tag the `cp310` denotes the **minimum** CPython the wheel loads
  on — *not* the interpreter it was built with — so one wheel installs on CPython 3.10, 3.11, 3.12,
  3.13, and 3.14 without rebuilding per minor version. It is compiled against the limited API pinned
  to 3.10 (`Py_LIMITED_API = 0x030A0000`, i.e. PyO3's `abi3-py310` feature) and may be **built with
  any interpreter ≥ 3.10** (e.g. `maturin build --interpreter python3.12`); the build interpreter
  does **not** change the `cp310-abi3` tag. A `cp312-abi3` tag would be **wrong** with a 3.10 floor:
  it refuses to install on CPython 3.10 and 3.11. (`abi3` = CPython's stable ABI / limited API.)
- **Free-threaded builds ship SEPARATE `abi3t` wheels.** The free-threaded (no-GIL / `Py_GIL_DISABLED`)
  interpreter is **not** ABI-compatible with the standard `abi3` wheel, so we publish **distinct
  `abi3t` wheels targeting py315** (tagged for the free-threaded `t` ABI, e.g. `cp315t`). These are
  built and released alongside the standard `abi3` wheels; a free-threaded interpreter picks up the
  `abi3t` wheel, a standard interpreter picks up the `abi3` wheel.

| Wheel | ABI | Wheel tag | Limited-API floor | Interpreter |
|---|---|---|---|---|
| standard | `abi3` (stable ABI) | `cp310-abi3` | `Py_LIMITED_API` = 3.10 (`abi3-py310`) | CPython 3.10–3.14+ (GIL) |
| free-threaded | `abi3t` | `cp315t` | free-threaded 3.15 | free-threaded CPython |

Rationale: one `abi3` wheel tagged at its 3.10 floor spans CPython 3.10→3.14+ from a single build,
minimizing the build matrix and install surprises, while the free-threaded ecosystem's separate ABI
requires its own `abi3t` artifact rather than overloading the standard wheel.

---

## 13. Server De-duplication Plan

### 13.1 What the server actually does today (and keeps doing)

The server does **not** have an inline `template.render(...) + Engine::generate` sequence that a
synchronous chat helper could replace. Its chat/completions path is:

1. **Build the prompt** — `prepare_generate_request` (`routes.rs:1873–1895`) calls `render_prompt`
   (`routes.rs:2029`), which converts OpenAI DTO messages → `ChatMessage`, resolves the offered
   `tools` catalog, and either renders the model template or applies the server's own tool-aware
   fallback `build_prompt` (`routes.rs:2066`) when there is no template (§2.4.1).
2. **Tokenize + count** — the server tokenizes the prompt itself and counts prompt tokens
   (`routes.rs:1884–1887`), producing `GeneratePrompt::TokenIds` (not `Text`).
3. **Apply request policy** — builds `GenerateOptions` from request JSON, applies tokenizer stop
   ids, request constraints (`response_format`, forced `tool_choice` → `GenerateConstraint`,
   `routes.rs:1897–1917`), and enforces the context cap (`enforce_context_cap`,
   `routes.rs:1121`/`1273`; `max_context` from model metadata, `routes.rs:1128`/`1280`).
4. **Route through the async `EngineDriver`** — `handle.engine.generate(session_lookup, req)` or
   `generate_pipeline(req, pipeline_input)` (`routes.rs:1158–1170` non-stream, `1308–1320` stream).
   The `EngineDriver` (`driver.rs:29`, `generate` at `driver.rs:167`, `generate_pipeline` at
   `driver.rs:196`) is **required** for queueing / continuous batching, persistent sessions
   (`get_or_create_session`, `routes.rs:1129`/`1282`), the multimodal `PipelineEngine` path, and
   async `DriverEvent` delivery.
5. **Stream + parse tool output** — the SSE task (`routes.rs:1323–1429`) buffers stream text under
   tool detection (`buffer_for_tool_detection`, `routes.rs:1328–1329`), parses tool calls
   (`parse_tool_calls`, `routes.rs:1358`/`1392`), emits `tool_calls` chunks with a matching
   `"tool_calls"` finish reason (`routes.rs:1383–1385`/`1412–1414`), does stop-boundary buffering
   (`StopBoundaryBuffer`, `routes.rs:1326`/`1337`), and has special logprob / JSON-object paths
   (`routes.rs:1333`, `1356`, `1417`).

A synchronous `&mut Engine` + callback helper **cannot** replace steps 2–5. So there is no
`chat::generate_chat[_stream]` direct-generation helper, and the server does not call one.

### 13.2 What the server actually shares

Only the **message-conversion + template-discovery + rendering** seam (§6.2). Plan:

1. Land `onnx_genai_engine::chat` (§6) as the shared **rendering** seam — `discover_chat_template`,
   `render_messages`, `ChatRenderOptions`, `TemplateFallback`, `RenderedPrompt`.
2. Replace the server's `load_chat_template` (`state.rs:282`) with `chat::discover_chat_template`
   (same `None`-on-absence precedence), and replace the message-building/render loop inside
   `render_prompt` (`routes.rs:2033–2058`) with a call to `chat::render_messages(...,
   TemplateFallback::CallerHandles)`. On `RenderedPrompt::NoModelTemplate` the server keeps calling
   its existing `build_prompt` fallback (`routes.rs:2060`/`2066`) — preserving `<|tools|>`,
   `<|tool_choice|>`, `tool_call_id`, and message `tool_calls` behavior exactly.
3. Everything else stays **server-owned**: OpenAI DTO mapping, `tools`/`tool_choice`/`tool_call_id`
   policy (`validate_tool_choice` `routes.rs:1674`, `forced_tool_choice_constraint`
   `routes.rs:1919`), token counting / context caps, session selection, multimodal dispatch, the
   `EngineDriver`, SSE buffering, and output tool-call parsing.

Per-request system messages and `add_generation_prompt=true` are preserved by the shared renderer:
system messages flow through `messages` unchanged, and `render_messages` passes
`add_generation_prompt` (default `true`, matching the server's hard-coded `true` at
`routes.rs:2057`).

### 13.3 This is a behavior-preserving refactor, **not** a no-op

Because `discover_chat_template` returns `None` (not an ORT default) and the server retains its
`build_prompt` fallback under `TemplateFallback::CallerHandles`, the server's observable behavior is
unchanged. The naive alternative — swapping in `ChatTemplate::from_model_dir`, which *always*
returns `DEFAULT_CHAT_TEMPLATE` (`chat_template.rs:149–151`) — **would** change behavior by dropping
the fallback-only `tool_choice`/`tool_call_id` context. The explicit fallback policy exists
precisely to avoid that. The payoff is exactly one place (`chat`) that decides template precedence,
message conversion, and `add_generation_prompt`, shared by the server and the `ChatPipeline` facade.

---

## 14. Design Decisions & Open Questions

### Decisions

- **D1 — Facade name.** Free function `pipeline(task, model_dir, opts)`; trait `TaskPipeline`;
  concrete `TextGenerationPipeline` / `ChatPipeline` / `FeatureExtractionPipeline` /
  `FillInMiddlePipeline`, all under module `task::`. This mirrors `transformers` while keeping a
  visible distance from the existing dataflow `PipelineEngine` (§4).
- **D2 — Chat rendering lives in the engine; generation does not move.** `onnx_genai_engine::chat`
  is a **pure message-conversion + template-discovery + rendering seam** over
  `onnx_genai_ort::ChatTemplate`. No new crate; the engine already depends on ort. **Two independent
  consumers** share the seam: the server (which keeps its async `EngineDriver`, sessions, multimodal
  dispatch, SSE buffering, and output tool-call parsing) and the `ChatPipeline` facade (which
  composes the seam with a direct `Engine::generate`/`generate_with_callback`). The seam never calls
  the engine itself (§6, §13).
- **D2a — Explicit template fallback policy.** `discover_chat_template` returns `None` on absence
  (no ORT default), and `render_messages` honors `TemplateFallback::{Error, Default, CallerHandles}`.
  The server uses `CallerHandles` to preserve its tool-aware `build_prompt` fallback; the facade
  defaults to `Error`. This makes the server change behavior-preserving (§2.4.1, §13.3).
  `TemplateFallback::Default` requires **one minimal ort-crate addition** — a public
  `ChatTemplate::builtin_default()` accessor — because ORT's `DEFAULT_CHAT_TEMPLATE` const and
  `ChatTemplate.template` field are private (`chat_template.rs:12`, `17–19`); the built-in default
  is model-independent, so no `model_dir` is threaded through `render_messages` (§6.2.1).
- **D3 — No new inference machinery.** Facade is a thin wrapper; all sampling/streaming/FIM/
  embedding behavior is the existing engine's. The one required engine-crate addition is a public
  **tokenization seam** for feature extraction (`Engine::embed_text` / `Engine::tokenize`, §8.1),
  since `tokenize_prompt` is private (`engine.rs:992`) and there is no public tokenizer accessor.
- **D4 — `GenerationConfig` is optional-fields + merge.** Per-call overrides merge onto pipeline
  defaults, then lower to `GenerateOptions` (§7). Add `do_sample` and `max_length` conversion (which
  **rejects** `max_length <= prompt_len` with an actionable error, §7.2); `stop` **appends** to
  inherited stops unless `stop_replace`; native knobs (`stop_on_eos`, `max_context`, speculative)
  stay reachable (§7.5); skip beam-search-only knobs.
- **D5 — Model-agnostic, no back-compat shims.** No hardcoded model/vendor/EP names; pre-release,
  so no legacy aliases.
- **D6 — Python package `nxrt`, `abi3`/`abi3t`, Python ≥ 3.10.** Already decided in EAGER.md/ORT2.md;
  GenAI API attaches there, defaulting to a `nxrt.genai` submodule with `pipeline`/`generate` also
  re-exported top-level. Wheels: standard **`abi3`** (tagged `cp310-abi3`, `Py_LIMITED_API` = 3.10 /
  `abi3-py310` floor, buildable with any interpreter ≥ 3.10) + separate free-threaded **`abi3t`**
  (py315 target, `cp315t`); minimum Python **3.10** (§12.4).
- **D7 — v1 task list:** `text-generation`, `chat`/`conversational`, `feature-extraction`
  (embeddings), `fill-in-the-middle`. Enum + `…Pipeline` types are extensible for `fill-mask` /
  `token-classification` later. The `task::TextGenerationPipeline`/`ChatPipeline` names do **not**
  collide with `PipelineEngine`/`PipelineGenerateRequest`/`PipelineTensors` (§4, §2.5).

### Open questions (for @justinchuby)

- **OQ-1 — `AnyPipeline` vs generic trait objects.** Tasks have different in/out types, so
  `pipeline()` returns an `AnyPipeline` enum. Acceptable, or prefer separate
  `TextGenerationPipeline::from_pretrained(...)` as the primary Rust entry and reserve the
  stringly-typed `pipeline()` mainly for the Python binding?
- **OQ-2 — `num_return_sequences`.** Skip in v1 and emulate later by looping with distinct seeds,
  or wire it now (the engine returns one sequence per call)?
- **OQ-3 — `generation_config.json` ingestion.** `Engine::from_dir` reads
  `genai_config.json`/`inference_metadata.yaml`; should the facade *also* parse HF
  `generation_config.json` for defaults, or require callers to pass a `GenerationConfig`?
- **OQ-4 — Python namespace.** `nxrt.pipeline` top-level (max parity) vs `nxrt.genai.pipeline`
  (clean separation from the runtime's `nxrt.tensor`/`nxrt.ops`)? Default here is *both*
  (submodule + top-level re-export).
- **OQ-5 — Tokenization/embedding seam shape (RESOLVED direction, confirm exact API).** §8.1
  requires a public tokenization seam because `tokenize_prompt` is private (`engine.rs:992`) and the
  `tokenizer` field is `pub(crate)` (`engine.rs:74`). Preferred: add `Engine::embed_text` **and** a
  public `Engine::tokenize` (the latter also serves `max_length`/`prompt_len` needs). Confirm
  whether to expose `Engine::tokenize` broadly or keep tokenization behind task-specific methods
  only.
