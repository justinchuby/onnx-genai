# Native Dynamic LoRA Adapter Support — Design (v2, CPU-first)

Status: **Design / RFC** · Version: **v2** · Revised by: Melina (systems eng) ·
Original author: Ripley (v1) · Requested by: @justinchuby
Scope: runtime, hot-swappable LoRA for LLM text generation inside **our own**
native genai runtime (a custom native runtime — **not** ONNX Runtime GenAI, and
**not** built on ORT GenAI's Adapters API). ORT GenAI's on-disk contract is
referenced as prior art only. Offline weight-merge is **out of scope** — this is
strictly runtime application.

> **v2 scope decision: CPU-first.** Phase 1 targets **CPU only**. CUDA is
> explicitly deferred to a later phase behind its own contract, because two of
> the design's load-bearing assumptions are **fatal on CUDA today** (r=0
> base-only zero-fill; CUDA-graph capture of overridable adapter inputs). See
> §0 and the CUDA phase (§F, P5–P7). Every claim about our code below cites the
> `file:line` that was read and verified for this revision.

---

## Changes from v1 (review response)

A design review found six issues in v1. Each is addressed here; the mechanism
was re-verified against the source before rewriting.

| # | Finding (v1) | Severity | What changed in v2 |
|---|---|---|---|
| 1 | W0 "add initializer-backed input to `input_index` with an owned resizable buffer" was presented as a small, localized executor change. | **FATAL** | Re-verified: it is **not** localized. `bind_input_shape` rejects the override because the declared dim is statically `0` (`bindings.rs:44-60`), and `materialize_initializers` **overwrites** the declared shape with the concrete `[0,K]` (`build.rs:707-711`). Redesigned W0 around an explicit **`OptionalOverride`** descriptor (§B.3) that keeps a **declared dynamic** shape, a default concrete shape+bytes, and a per-run active shape. Enumerated every `input_index` touch-point and a full test matrix. |
| 2 | Graph injection survives the optimizer; math is "MatMul→MatMul". | **SOUND-with-concern** | Confirmed it survives **today's** optimizer (const-fold is integer/shape only — `constant_folding.rs:11-20,84-89`; DCE keeps output-feeding nodes — `dead_node.rs:11-45`). But (a) an initializer-backed input is "constant" to optimizer APIs → a future float-fold could legally erase the branch, and (b) the math was **underspecified**: PEFT tensors are `[r,K]`/`[N,r]`, ONNX `MatMul` has no transpose attribute, and CUDA `MatMul` **rejects strided transpose views** (`ep-cuda/.../matmul.rs:407-412`). Fixed: **transpose at load time** so adapter storage is already `[K,r]`/`[r,N]` and plain `MatMul` works; added an **optimizer-survival regression** to the test plan (§E). |
| 3 | r=0 base-only default yields a no-op `Add`. | **FATAL on CUDA** (mitigated by CPU-first) | CPU zero-fills `k==0` correctly (`ep-cpu/.../matmul.rs:1032-1035` half path, `:1259` generic path pre-zeroes the result). CUDA does **not**: `k==0` misses the capture-safe GEMV (gated on `plan.k > 0`, `ep-cuda/.../matmul.rs:445-448`), falls to workspace cuBLASLt with **no zero-fill** (`:462-510`), and marks itself non-capturable (`:618-624`). v2 documents r=0 base-only as a **CPU-phase** decision and lists the explicit CUDA prerequisite. |
| 4 | CUDA-graph capture reuses the CPU `Routed` plumbing on activation. | **LIKELY-FLAW** (deferred) | Confirmed native CUDA **rejects all `Routed` step inputs today** (`native_decode/load.rs:353-365`); the CPU `Routed` seam **cannot** be reused for CUDA. Same-shape content replacement *is* capture-safe in principle (replay signature is pointer+shape, not contents — `state.rs:562-606`, `bindings.rs:245-257`), but a rank change forces rebind+recapture. v2 moves **all** CUDA work to an explicit later phase with its own contract; adapter inputs go directly into `DecodeCudaState` bindings, not through `Routed`. |
| 5 | q/k/v are separate `MatMulNBits` nodes. | **FATAL — factual error** | **Wrong for the principal Qwen2.5 export.** Verified against a real model: Qwen2.5 **fuses** q/k/v into **one** `qkv_proj` `MatMulNBits` (real names in §H). Three independent PEFT factors must map onto one fused projection with verified Q/K/V slice offsets. Redesigned around a **target manifest** (§C) and combined B-factors into fused output slices. **New risk (missed by review):** fusion is **not universal** — Qwen3 keeps q/k/v split, and Qwen3.5-2B uses a different linear-attention projection entirely (§H). |
| 6 | Phasing put per-request `adapter_id` and CUDA early. | **LIKELY-FLAW** | Rephased **CPU-first** with correct dependency order (§F): OptionalOverride → CPU injected-LoRA correctness → target manifest → single fixed adapter/session → (LATER) CUDA capture → scheduler partitioning → per-request `adapter_id`. Per-request `adapter_id` is **not** exposed before scheduler isolation. |

---

## 0. Established facts (what the code actually does)

**Native decode is a thin stateful wrapper over the graph-IR session.**
`NativeDecodeSession` holds an `InferenceSession`, the carried KV `past`, and
`current_len`; each step calls `self.session.run(&bindings)`
(`crates/onnx-genai-engine/src/native_decode/mod.rs:66-81`,
`crates/onnx-genai-engine/src/native_decode/cpu.rs:271-330`, run call at
`cpu.rs:303`). CUDA decode goes through `decode_cuda`
(`mod.rs:268-275`, `native_decode/cuda.rs`).

**The graph is a fixed IR executed node-by-node.** The executor dispatches one
plan node at a time and resolves a `Kernel` per node
(`crates/onnx-runtime-session/src/executor/dispatch.rs` → `exec_kernel_node`).
Kernels are registered by `(op_type, domain, since_version)`
(`crates/onnx-runtime-ep-cpu/src/kernels/mod.rs:258-262` for
`MatMulNBits`/`com.microsoft` and `MatMul`, `:313` for `Add`).

**The int4 projection output is produced by `MatMulNBits`.** Inputs are
`A`(0, float activations), `B`(1, `uint8` packed weights), `scales`(2), optional
`zero_points`(3), `g_idx`(4), `bias`(5); single output `Y = A @ dequant(B)^T`.
**This `Y` value is the injection point**: a LoRA delta `scale · (x @ A) @ B`
(with A/B **stored already-transposed**, see §A) is added to it. Which
`MatMulNBits` nodes are LoRA targets is **export-specific** (§C, §H) — it is
**not** always one node per q/k/v.

**The graph is mutable at load time and there is a precedent for exactly this
kind of rewrite.** `Graph::insert_node` (`onnx-runtime-ir/src/graph.rs:434`),
`create_value` (`:187`), `add_input` (`:205`), `set_initializer` (`:288`),
`node_mut` (`:81`). The session's fp16-decode pass rewrites the freshly-loaded
graph in place *before* optimization (`fp16_decode.rs`, invoked from
`InferenceSession::build` at `crates/onnx-runtime-session/src/lib.rs:674`,
immediately before `optimize_graph` at `lib.rs:691`). It locates every
`MatMulNBits` by walking `graph.nodes` and inspecting input slot 2
(`fp16_decode.rs:75-88`) — the same discovery a LoRA pass needs.

**Hot-swap infrastructure exists on the device path.** Persistent device
buffers back graph inputs and are rebound between runs; the device-graph replay
signature is **pointer + shape, not contents** (`bindings.rs:245-257`,
`state.rs:562-606`), so same-shape content replacement is capture-safe **in
principle**. This is a CUDA-phase asset, not a Phase-1 dependency.

**CUDA cannot take our proposed adapter inputs today.** Native CUDA decode
**rejects any `Routed` (or embedding) step input** at load
(`native_decode/load.rs:353-365`), and its bindings are built once in
`DecodeCudaState::new`. The CPU `Routed` seam (`native_decode/cpu.rs`) is **not**
reusable for CUDA. → CUDA LoRA needs its own binding contract (§F, P5).

**Per-request state is single-sequence for native decode; batching lives above
it.** `NativeDecodeSession` carries one sequence's `past`/`current_len`.
Multi-sequence state is in the scheduler: `Request`/`RunningSequence`
(`crates/onnx-genai-scheduler/src/lib.rs:45-73`) and the engine's
`ContinuousBatchRow` (`crates/onnx-genai-engine/src/batched.rs:72`). An
`adapter_id` would attach here — but only **after** scheduler isolation (§F, P7).

**Package/metadata plumbing does not exist yet.** `docs/MODEL_PACKAGE.md:302-309`
and `:643-651` describe `adapters/`, `support.lora`, and
`ModelAssets.adapters: Vec<PathBuf>` — none implemented. `InferenceMetadata`
(`crates/onnx-genai-metadata/src/schema/mod.rs:57`) has no adapter field.
`safetensors` is **not** a current dependency.

---

## A. Adapter wire format

### Prior art (ORT GenAI `.onnx_adapter`)
ORT GenAI delegates adapter loading to ONNX Runtime's version-1 FlatBuffers
container (`adapter_schema.fbs`, file identifier `TORT`). The container carries
format/adapter/model versions and named tensor parameters, but **does not carry
rank, alpha, or scale metadata**. Parameter names equal graph-input names. The
current ORT GenAI model builder emits names such as
`model.layers.0.attn.q_proj.lora_A.MatMul.weight`, stores A/B already in
`[K,r]`/`[r,N]` orientation, and multiplies B by the PEFT scale before export.
Our interoperable reader therefore derives rank from the paired dimensions and
records `scale = 1` (`alpha = rank`) without rescaling the already-scaled B
tensor. Arbitrary graph-input naming and graphs that apply scale separately are
not inferable from the container alone and are rejected rather than guessed.

### Recommendation: direct PEFT/safetensors ingestion (primary)
Load stock PEFT adapters directly — `adapter_config.json`
(`r`, `lora_alpha`, `target_modules`, `fan_in_fan_out`, and the per-module
`rank_pattern` / `alpha_pattern`) + `adapter_model.safetensors` (keys like
`base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight`). One small
dependency (`safetensors`: header + mmap of raw tensors, no graph parser). A
native container (our own `.onnx_adapter`-style file, or an ONNX-initializer
file via `onnx_runtime_loader::load_model_bytes_with_weights`) is an optional
signed-package format a converter can emit from the PEFT form.

### Tensor layout — **transposed at load** (v2 fix, Finding 2)
PEFT stores `lora_A` as `[r, K]` and `lora_B` as `[N, r]` and computes
`delta = (x @ A^T) @ B^T`. ONNX `MatMul` has **no** transpose attribute, and the
CUDA kernel **rejects strided (transpose) views** (`ep-cuda/.../matmul.rs:407-412`
`"MatMul with a non-contiguous (strided) input"`). Therefore the **loader
transposes A and B once at load time** and stores them **already-transposed**:

- store `A_t` with shape `[K, r]` (transpose of PEFT `lora_A[r,K]`),
- store `B_t` with shape `[r, N]` (transpose of PEFT `lora_B[N,r]`),

so the injected graph is plain, contiguous `MatMul(x[M,K], A_t[K,r]) -> [M,r]`
then `MatMul([M,r], B_t[r,N]) -> [M,N]` — no Transpose nodes, no strided views,
CPU-and-CUDA-portable. `fan_in_fan_out=True` adapters are already stored
`[K,r]`/`[r,N]` and skip the transpose (validate and record which case applies).

**Dtype:** A/B in fp16 or fp32. Delta math runs in fp16/fp32; the int4 base is
never touched. Base-only default is `r = 0` (a **CPU-phase** decision — see §0
Finding 3 and §F P5 for the CUDA prerequisite).

**Per-module rank/alpha.** Honor PEFT `rank_pattern`/`alpha_pattern`: rank and
scale can differ per target module, so the injected A/B inputs are sized
per-module, not from a single global `r`.

---

## B. Injection architecture

### B.1 Approach: graph-native named-input override

The base int4 `MatMulNBits` stays untouched. A **separate** fp16/fp32 delta
branch is added and its A/B factors are exposed as **runtime-overridable graph
inputs**:

```
   x ──► MatMulNBits(int4 base)  ──► Y_base ───────────────────────────┐
   │                                                                    ▼
   └─► MatMul(x, A_t) ─► r ─► MatMul(r, B_t) ─► d ─► Mul(d, scale) ─► Add ─► Y_new
        (A_t input: default [K,0])   (B_t input: default [0,N])     consumers: Y_base → Y_new
```

- **int4 base untouched, no requantize.** A_t/B_t are two dense contiguous
  MatMuls; `Mul(scale)` + `Add`. Base-only default: `r=0` ⇒ empty delta ⇒ `Add`
  no-op (CPU zero-fills the empty inner-dim MatMul — §0 Finding 3).
- **Hot-swap = override named inputs.** Activate adapter *k* = bind
  `A_t_k`/`B_t_k` (and `scale`) to the named override inputs; deactivate =
  restore the `r=0` defaults (or `scale=0`).
- **No new kernel** — reuses `MatMul`, `Mul`, `Add`.

Rejected alternatives (unchanged from v1): (b) executor-hook in `dispatch.rs`
(edits the hottest path, complicates capture); (c) a fused `MatMulNBitsLora`
kernel (duplicates the two most tuned kernels).

### B.2 Model-source paths
- **B1 — consume baked-structure models** (exporter emits the branch + overridable
  inputs). Requires a Mobius exporter change (cross-repo dependency).
- **B2 — load-time injection** (`onnx-runtime-session/src/lora_inject.rs`, sibling
  of `fp16_decode.rs`), wired into `InferenceSession::build` after the fp16 pass
  (`lib.rs:674`) and **before** `optimize_graph` (`lib.rs:691`), using the
  `Graph::insert_node/create_value/add_input/set_initializer` helpers
  (`graph.rs:434,187,205,288`). CPU-first ships **B2** (exporter-independent).

**Optimizer-survival requirement (v2 fix, Finding 2).** Because an
initializer-backed input reads as "constant" to optimizer APIs, the injection
must be safe against future passes. Two options, pick one and assert it:
1. Mark overridable-initializer inputs **non-constant** to all passes, **or**
2. Inject **after** device-independent `optimize_graph` and rely on the existing
   EP-scoped re-inference (`build.rs:11-40` runs EP passes then re-infers shapes;
   EP placement is `build.rs:419-420`) — i.e. inject before `place_graph`.

Today's optimizer is already safe (const-fold is integer/shape only —
`constant_folding.rs:84-89`; DCE keeps output-feeding nodes and never removes
inputs/initializers — `dead_node.rs:16-18,53-60`), and the terminal `Add` keeps
the whole branch live. §E adds a regression that **locks this in**.

### B.3 The executor change — `OptionalOverride` (v2 redesign of W0, Finding 1)

**Why v1's W0 was fatal.** v1 proposed adding an initializer-backed input to
`input_index` with an owned resizable buffer, calling it "small and localized."
Verified — it is not:

- `bind_input_shape` compares the fed shape against the **declared** shape and
  rejects any dim where `Dim::Static(n) != actual` (`bindings.rs:52-60`). With a
  declared `[0,K]`, feeding `[r,K]` fails on the `0 != r` static-dim check.
- `materialize_initializers` **overwrites** the value's declared shape with the
  concrete initializer dims: `value_shapes.insert(vid, dims.iter().map(Dim::Static)…)`
  (`build.rs:707-711`). So the declared shape *becomes* `[0,K]` — there is no
  symbolic dim left to bind an override against.
- Symbol bindings come **only** from supplied inputs (`bind_symbols`,
  `bindings.rs:11-25`); an absent input contributes no symbol value.
- `build_name_indexes` **skips** any graph input that is also an initializer
  (`continue` at `build.rs:928`), excluding it from both `required_inputs` and
  `input_index`.

So a single value cannot simultaneously (a) declare a dynamic rank dim, (b) carry
a concrete default when unfed, and (c) accept a different-rank override when fed.
We need a first-class descriptor.

**The descriptor.** Introduce an *overridable optional input* class carried
alongside the executor state (sketch — illustrative, not final Rust):

```
struct OptionalOverride {
    vid: ValueId,
    name: String,
    dtype: DataType,
    declared_shape: Shape,     // DYNAMIC in the override dim, e.g. [K, sym("lora_r")]
    default_shape: Vec<usize>, // concrete fallback, e.g. [K, 0]
    default_bytes: Arc<[u8]>,  // owned; used when unfed (never a read-only mmap alias)
    active_shape: Vec<usize>,  // per-run: default_shape when unfed, override shape when fed
    override_state: OverrideState, // None | Host{ptr,len} | Device{binding}
}
```

**Symbol-binding rule.** Bind the descriptor's symbol(s) from its **active**
shape each run: when unfed, from `default_shape`; when fed (host or device),
from the override's shape. Do this in `bind_symbols` **before** node shape
resolution so downstream shapes (the `[M,r]` intermediate, the `Add`) resolve
consistently. Crucially, **do not overwrite `declared_shape` during
initializer materialization** — keep the symbolic dim.

**Exact code touch-points (verified line ranges):**
1. `build.rs:684-711` (`materialize_initializers`) — for override values, **do
   not** insert a static `value_shapes` entry from the initializer dims; keep the
   declared dynamic shape and stash the default bytes/shape on the descriptor.
   Force an **owned** buffer (never the borrowed mmap alias path at
   `build.rs:739-761`) so per-run override can rewrite/resize.
2. `build.rs:925-935` (`build_name_indexes`) — register override values in
   `input_index` (feedable by name) while keeping them **out of**
   `required_inputs` (still optional).
3. `bindings.rs:11-25` (`bind_symbols`) / `:28-81` (`bind_input_shape`) — teach
   symbol binding to seed override symbols from the active shape, and skip the
   `Dim::Static(0) != actual` rejection for override dims (they are declared
   symbolic, so they already take the `Dim::Symbolic` branch at `:62-77`).
4. `state.rs:36-37` — `required_inputs` stays "inputs minus initializers"; add a
   third documented category (the override set) so the invariants are explicit.
5. Host override path (`executor/run.rs` `prepare_run_buffers` /
   `validate_required_inputs`) — when an override is fed by name, resize+bind it
   like an external input; when absent, reinstate `default_bytes`/`default_shape`.
6. Device override path (`bindings.rs:259-320` `prepare_external_bindings`,
   persistent-input resolution) — CUDA phase only (§F P5); an override fed as a
   `DeviceIoBinding` resolves through `input_index` exactly like other device
   inputs, and its identity flows into the capture signature
   (`state.rs:562-606`).
7. Binding capacity / decode-memo signatures that enumerate persistent inputs
   (device-graph signature `bindings.rs:245-257`) must include override values so
   a rank change invalidates and re-arms capture rather than replaying stale
   geometry. **CUDA phase only.**

**Byte-identity guarantee.** A graph with **no** override values must build and
run byte-for-byte as today: the new code paths are all gated on membership in the
override set, which is empty for un-injected graphs.

**Test matrix for W0 (CPU Phase 1 unless noted):**
| Test | Asserts |
|---|---|
| unfed → default | override absent ⇒ `active_shape == default_shape` (`[K,0]`), `Add` is a true no-op, output bit-identical to base-only. |
| host override | feeding `A_t[K,r]`/`B_t[r,N]` by name binds the symbol, resizes the owned buffer, and produces `Y_base + scale·(x@A_t)@B_t`. |
| override → default restoration | after a fed run, a subsequent unfed run restores `default_bytes` exactly (no residual adapter state). |
| rank change | feeding `r=8` then `r=16` rebinds the symbol and resizes without error; stale-shape reuse is rejected. |
| byte-identity | a graph with no override values is unchanged vs. baseline (golden run hash). |
| capture invalidation (**CUDA phase**) | a rank/address change flips the device-graph signature (`state.rs:562-606`) and forces recapture. |
| decode-memo interaction (**CUDA phase**) | override membership is part of the persistent-binding set; a membership change re-arms capture. |

---

## C. Target discoverability — the manifest (v2 redesign, Finding 5)

**The v1 claim "q/k/v are separate `MatMulNBits` nodes" is false for our
principal model family.** Verified against a real int4 export (§H): Qwen2.5
**fuses** q/k/v into **one** `qkv_proj` `MatMulNBits`, while PEFT adapters carry
**separate** `q_proj`/`k_proj`/`v_proj` `lora_A`/`lora_B`. So three independent
PEFT factors must be combined onto **one** fused projection's output slices.

**New risk the review missed: fusion is not universal.** Also verified (§H):
Qwen3 keeps q/k/v **split** (and uses a different weight-name suffix), and
Qwen3.5-2B "text" uses a **linear-attention** projection (`in_proj_qkv`, `in_proj_a/b/z`)
that is neither the fused nor the split QKV shape. Name-pattern guessing is
therefore unsafe. Discovery **must** be manifest-driven.

**Target manifest.** Build, at load time, a per-model manifest mapping each
semantic PEFT module to a concrete graph target:

```
TargetEntry {
    semantic: "model.layers.0.self_attn.q_proj",
    node_id / value_id: <the MatMulNBits Y value to inject after>,
    orientation: RowMajor | fan_in_fan_out,
    k: usize,                 // input features
    n: usize,                 // this module's output width
    fused_group: Option<FusedGroup>, // Some for qkv_proj
}
FusedGroup {
    node: "qkv_proj",
    slices: [ ("q_proj", 0..3584), ("k_proj", 3584..4096), ("v_proj", 4096..4608) ],
    order: [Q, K, V],         // verified against GQA packing (§H)
}
```

**Combining PEFT factors into a fused projection.** For a fused `qkv_proj` with
output width `N = N_q + N_k + N_v`, each PEFT module has its own `A_t_i[K,r_i]`
and `B_t_i[r_i, N_i]`. The injected delta must place module *i*'s contribution
into columns `slice_i` of the `[M,N]` delta and zeros elsewhere. Two equivalent
constructions:
1. **Block-diagonal concat**: stack the three A_t into `A_t[K, r_q+r_k+r_v]`
   (block-diagonal is unnecessary — A is shared over K only if ranks are merged;
   keep three separate `MatMul(x, A_t_i)` to respect per-module `rank_pattern`),
   and assemble a `B_t[Σr_i, N]` whose block *i* writes only into `slice_i`
   (zero-padded into the other column ranges). One `[M,Σr]·[Σr,N]` then yields the
   correctly-sliced fused delta.
2. **Per-module MatMul + scatter-add** into the fused delta columns, then one
   `Add` to `Y_base`.

Construction (1) is preferred (fewer nodes, plays well with the overridable-input
model: the concatenated `A_t`/`B_t` are the override tensors). Either way the
**Q/K/V ordering and slice offsets must be validated** against the actual export
(§H); a wrong offset silently corrupts attention.

**Fail loud.** If any target module in `adapter_config.json` cannot be resolved
to a validated manifest entry (unknown layout, missing node, ambiguous fusion,
dim mismatch), **reject the adapter at load** with an actionable error. Never
silently skip a projection.

---

## D. Runtime API + lifecycle (CPU Phase 1)

### `LoraManager` (proposed `crates/onnx-genai-engine/src/lora/mod.rs`)
- `load(path) -> AdapterId` — parse container (§A), **transpose A/B at load**,
  validate against the base via the target manifest (§C), reject on any
  unresolved module.
- LRU cache of decoded adapters (transposed A_t/B_t host bytes, keyed by
  `AdapterId`), byte-budgeted like the KV `ByteBudget` pattern
  (`scheduler/src/pressure.rs`).
- `activate(session, AdapterId)` — bind A_t/B_t into the session's override
  inputs and set `scale`; `deactivate` restores the `r=0` defaults (or `scale=0`).

### Phase-1 selection model — **single fixed adapter per session**
The whole running batch shares **one** active adapter (or none), applied via the
named-input override. This matches the current single-sequence native decode and
needs no new kernel. **No per-request `adapter_id` API in Phase 1** — see §F for
why it is gated behind scheduler isolation.

---

## E. Numerics & correctness plan (CPU)

**Golden reference test.** Build a tiny base `MatMulNBits` (e.g. `K=64, N=64,
block_size=32, bits=4`), a known small PEFT `A[r,K]`/`B[N,r]`, run native CPU
decode with the adapter active through the **real `lora_inject` pass**, and
assert:

```
Y_native  ≈  Y_base  +  scale · (x @ A^T) @ B^T
```

- fp32 delta path: tight tolerance (~1e-4 relative).
- fp16 delta path: looser tolerance (~1e-2), documented.
- `scale = 0` **and** `r = 0` ⇒ `Y_native == Y_base` **bit-for-bit** (deactivation
  and base-only are true no-ops; relies on the CPU zero-filled result buffer,
  `ep-cpu/.../matmul.rs:1032-1035,1259`).
- **Transpose correctness**: assert the loader's `[r,K]→[K,r]` / `[N,r]→[r,N]`
  transpose reproduces the PEFT delta (guards against a silent axis swap).

**Fused-QKV test.** With three separate PEFT factors and a fused `qkv_proj`
manifest entry, assert each of the Q/K/V output slices equals its independent
per-module delta and the other slices are untouched (validates §C offsets).

**Optimizer-survival regression (v2 addition, Finding 2).** After injection, run
**every** optimization level and the EP-placement pass and assert that all A_t/B_t
override inputs and every injected LoRA node (both `MatMul`s, `Mul`, `Add`)
**survive** and remain wired to the graph output. Lock in that no const-fold/DCE
change can erase the branch.

**Fixture home.** Extend
`crates/onnx-genai-engine/src/native_decode/tests.rs` with a `lora_*` module;
build the tiny graph in-memory via the IR builder and inject through the same
pass, so tests exercise the real path, not a mock.

---

## F. Phased implementation plan (CPU-first)

Ordered by dependency. **P1–P4 are CPU Phase 1. P5–P7 are deferred.**

**P1 — `OptionalOverride` shape semantics + tests (the core).** The descriptor
and executor changes of §B.3. Files:
`executor/{build.rs:684-711,925-935}`, `executor/bindings.rs:11-81`,
`executor/run.rs`, `executor/state.rs:36-37`. Deps: none. Risk: **med** (run
path; must stay byte-identical when no override exists). Size: L.

**P2 — CPU injected-LoRA correctness.** PEFT/safetensors loader with
**transpose-at-load** (§A) + `lora_inject.rs` pass (B2) + r=0 base-only default +
the §E golden numerics vs a PEFT reference. Files:
`onnx-genai-engine/src/lora/format.rs` (+`safetensors` dep),
`onnx-runtime-session/src/lora_inject.rs`, `native_decode/tests.rs`.
Deps: P1. Risk: med. Size: L.

**P3 — Target manifest incl. fused QKV (§C).** Semantic→node/value mapping,
fused slice offsets with verified Q/K/V ordering, per-module `rank_pattern`/
`alpha_pattern`, fail-loud on unresolved modules. Deps: P2. Risk: med. Size: M.

**P4 — Single fixed adapter per session (§D).** `LoraManager` load/LRU/activate;
CLI `--adapter <path>`; runtime `activate_adapter`/`deactivate_adapter`;
metadata/package plumbing (§G). **No per-request API.** Deps: P2, P3. Risk: low.
Size: M.

**P5 — (LATER) CUDA persistent bindings + capture policy.** Prerequisites, all
verified as blocking today:
- Add a **capture-safe `k==0` zero-fill** CUDA MatMul path (or use fixed-rank
  zero-filled buffers with `scale=0`) **before** any CUDA LoRA — today `k==0`
  misses the GEMV (`ep-cuda/.../matmul.rs:445-448`) and the fallback neither
  zero-fills nor captures (`:462-510,618-624`).
- Feed A_t/B_t as **direct `DecodeCudaState` bindings**, not `Routed` — native
  CUDA rejects `Routed` today (`native_decode/load.rs:353-365`).
- Fixed rank/capacity per session (zero-padded, stable addresses) **or** session
  pools keyed by rank, each warmed/captured; activation must **invalidate/re-arm
  capture** on shape/address/membership change (signature at `state.rs:562-606`).
Deps: P4. Risk: **high**. Size: XL.

**P6 — (LATER) Scheduler partitioning / adapter-homogeneous batching.** Group
requests by adapter so a batch shares one active adapter; or session pools per
adapter. Deps: P5. Risk: med. Size: L.

**P7 — (LATER) Per-request `adapter_id`.** Only **after** P6. Add
`adapter_id: Option<AdapterId>` to `scheduler::Request`/`RunningSequence`
(`scheduler/src/lib.rs:45-73`) and `ContinuousBatchRow` (`batched.rs:72`).
**Do not expose per-request `adapter_id` before scheduler isolation** — requests
sharing execution state would otherwise receive the wrong adapter. Deps: P6.
Risk: med. Size: M.

Deferred entirely (gate on real multi-tenant demand): SGMV/BGMV grouped-GEMM EP
op + paged adapter pool (Punica [2310.18547], S-LoRA [2311.03285]).

---

## G. Metadata / package integration (CPU Phase 1, additive)

- **`ModelAssets`**: implement the documented `adapters: Vec<PathBuf>` field
  (`docs/MODEL_PACKAGE.md:643-651`) so `adapters/*.onnx_adapter` and the
  `support.lora` marker (`MODEL_PACKAGE.md:302-309`) are discovered.
- **`InferenceMetadata`**: add an optional `adapters` block (`schema/mod.rs:57`),
  e.g. `pub adapters: Option<LoraCapabilities>` declaring default adapter(s),
  target-module policy, and hot-swap support. Purely additive → same schema major
  version.
- **CLI/API**: `--adapter <path>` to preload + activate; engine-handle
  `activate_adapter(id)` / `deactivate_adapter()`. Per-request `adapter` field is
  **not** added until P7.

---

## H. Verified model export facts (Finding 5)

Inspected with `onnx==1.22` (graph only, `load_external_data=False`).

**Qwen2.5 (principal family) — FUSED QKV.**
Model: `~/.foundry/cache/models/Microsoft/qwen2.5-coder-7b-instruct-generic-cpu-4/v4/model.onnx`
(same 347-node / 141-`MatMulNBits` structure as
`qwen2.5-1.5b-instruct-cuda-gpu-4/v4/model.onnx`). Layer-0 attention:
- Node `/model/layers.0/attn/qkv_proj/MatMul_Q4`, packed weight
  `model.layers.0.attn.qkv_proj.MatMul.weight_Q4`, attrs `K=3584`, `N=4608`,
  `bits=4`, `block_size=32`.
- MLP keeps **separate** `gate_proj` / `up_proj` / `down_proj` `MatMul_Q4`;
  attention `o_proj` is its own node.
- `GroupQueryAttention` attrs: `num_heads=28`, `kv_num_heads=4` ⇒ `head_dim=128`.
  Fused `N=4608 = Q(28·128=3584) + K(4·128=512) + V(4·128=512)`. GQA takes the
  **packed** projection as its first input (key/value inputs empty), so the
  **Q,K,V order and offsets** are: `q_proj=[0:3584]`, `k_proj=[3584:4096]`,
  `v_proj=[4096:4608]`. These are the manifest slice offsets for this export.

**Qwen3 — SPLIT QKV (different layout AND naming).**
Model: `~/.foundry/cache/models/Microsoft/qwen3-0.6b-generic-cpu-4/v4/model.onnx`.
Layer-0 has separate `/model/layers.0/attn/{q,k,v}_proj/MatMulNBits` with weights
`model.layers.0.attn.{q,k,v}_proj.MatMulNBits.qweight` (note the different node
op-name **`MatMulNBits`** and weight suffix **`.qweight`** vs Qwen2.5's
`MatMul_Q4` / `.weight_Q4`). Dims: `q_proj K=1024 N=2048`, `k_proj`/`v_proj`
`K=1024 N=1024`.

**Qwen3.5-2B "text" — DIFFERENT projection entirely.**
Model: `~/.foundry/cache/models/Microsoft/qwen3.5-2b-text-generic-cpu-1/v1/model.onnx`
uses linear-attention nodes `/model/layers.0/linear_attn/in_proj_qkv/MatMul_Q4`,
`in_proj_a`, `in_proj_b`, `in_proj_z` — neither the fused nor the split standard
QKV shape.

**Implication.** The QKV mapping is **per-export**, and even node/weight name
suffixes differ across families. The target manifest (§C) must be built from the
actual graph, must validate dims/offsets, and must **fail loud** when a layout is
unrecognized. (A local DeepSeek-Qwen export was cited by the team as sharing the
fused Qwen2.5 layout; not present in this environment, so treat its exact node
names as **to confirm against that export**.)

---

## I. Open questions / risks (honest)

1. **W0 is a prerequisite, not a given.** Redesigned as `OptionalOverride`
   (§B.3). Must be provably byte-identical for graphs without override values.
2. **Fusion is export-specific (NEW, review missed it).** Qwen2.5 fuses, Qwen3
   splits, Qwen3.5-2B differs (§H). Manifest-driven discovery + fail-loud is
   mandatory; name-pattern heuristics are unsafe.
3. **CUDA r=0 zero-fill + capture (deferred, blocking for P5).** `k==0` misses
   the capture-safe GEMV and the fallback neither zero-fills nor captures
   (`ep-cuda/.../matmul.rs:445-448,462-510,618-624`).
4. **CUDA cannot take `Routed` adapter inputs today** (`load.rs:353-365`); needs a
   direct `DecodeCudaState` binding contract (P5).
5. **int4 base + fp16 delta accuracy.** fp16 accumulation on wide `N` could flip
   razor-thin greedy-argmax ties (same sensitivity as MatMulNBits N-tile shard
   boundaries). Needs the §E golden **plus** an on-model argmax check before eval
   parity is claimed.
6. **q/k norm placement (Qwen3).** Qwen3 applies RMSNorm to q/k **after** the
   projection; PEFT LoRA targets the projection output, which is exactly where we
   inject (before the norm) — consistent with PEFT, but assert this in the manifest
   for split-QKV exports.
7. **Optimizer future-proofing.** Overridable-initializer inputs read as
   "constant"; mark them non-constant or inject post-optimize (§B.2). The §E
   survival regression guards against regressions.
8. **Plugin/OpenVINO EPs** must support the injected `MatMul`/`Mul`/`Add`, or fall
   back.

---

## Summary of recommendations

- **Scope:** **CPU-first.** CUDA deferred behind its own contract (two CUDA
  FATALs: r=0 zero-fill + capture).
- **A (format):** direct PEFT/safetensors ingestion; **transpose A/B at load** to
  store `[K,r]`/`[r,N]` so plain contiguous `MatMul` works on CPU and CUDA.
- **B (injection):** graph-native named-input override; base stays int4
  `MatMulNBits`; delta is a separate `MatMul→MatMul→Mul→Add` with
  runtime-overridable A_t/B_t inputs. Core deliverable is the executor
  **`OptionalOverride`** class (W0), redesigned so the declared shape stays
  dynamic and is bound from the active (default-or-override) shape.
- **C (targets):** a validated per-export **target manifest**; combine PEFT
  q/k/v factors into fused `qkv_proj` output slices with verified ordering; honor
  `rank_pattern`/`alpha_pattern`; **fail loud** on unresolved modules.
- **F (phasing):** P1 OptionalOverride → P2 CPU injected correctness → P3 manifest
  → P4 single fixed adapter → **(LATER)** P5 CUDA → P6 scheduler partitioning →
  P7 per-request `adapter_id`.
- **Top risks:** (1) OptionalOverride executor change correctness/byte-identity;
  (2) export-specific QKV layout (fused vs split vs linear-attn); (3) CUDA r=0
  zero-fill + capture (deferred). Close behind: int4+fp16 argmax accuracy.
