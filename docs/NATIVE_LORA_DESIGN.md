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

---

## Changes from Phase 1 design (P2 addendum)

Phase 1 (§A–§I) shipped on this branch (P1–P4 committed): a **single fixed
adapter per session**, applied by feeding the injected `MatMul→MatMul→Mul→Add`
delta branch's `A_t`/`B_t` through the executor `OptionalOverride` mechanism
(`executor/state.rs:286`, `build_with_overrides`), with a fail-loud target
manifest (`lora_inject.rs`). §J below adds **Phase 2: many adapters live at once
in one batch**, executed by a dedicated **LoRA subsystem** — a paged A/B weight
pool plus a grouped custom op (SGMV/BGMV) — reached as a **custom operator**
injected into the base graph and run in the **single existing executor plan**.
Nothing in §J re-litigates the settled §A–§I decisions; it changes only two
things and unifies a third:

1. **Injection target.** Phase 1 injects a 4-node delta subgraph per projection;
   Phase 2 injects **one grouped custom op per target projection** (§J.1). The
   4-node subgraph becomes the *degenerate pool-of-one* case of the same op
   (§J.5) — the recommendation is to **unify**, not to keep two code paths.
2. **Weight delivery.** Phase 1 feeds one adapter's `A_t`/`B_t` as override
   inputs; Phase 2 delivers a **handle to a paged pool of many adapters** through
   the existing lazy-weight seam (`ep-api/src/weight.rs:95`, dispatch at
   `dispatch.rs:335-342`) plus a **per-batch-row segment descriptor** (§J.1,
   §J.2). Per-request `adapter_id` threads scheduler→op (§J.4).
3. **Manifest source.** Phase 1 derives the manifest from the graph (fail-loud);
   Phase 2 makes the **exporter-declared manifest in `InferenceMetadata` the
   primary source**, with graph discovery demoted to a fail-loud fallback (§J.6).

---

## J. Phase 2: multi-adapter subsystem (grouped LoRA op)

**Settled architecture (do not re-litigate).** Multi-adapter LoRA execution is a
**dedicated LoRA subsystem** — a paged A/B weight pool plus an SGMV/BGMV grouped
kernel — exposed as a **custom operator injected into the base graph and executed
in the one existing executor plan**. It is **not** separate per-adapter sessions
and **not** per-layer session interleaving. Rationale: LoRA acts mid-layer on
`q/k/v/o/gate/up/down` and feeds straight back into the forward pass, so an
opaque per-adapter session would force a cross-session round-trip **per targeted
projection per token** during decode — exactly the hot path. Keeping the delta
inline as a graph op means the batched activations never leave the plan. This is
the Punica ([2310.18547]) / S-LoRA ([2311.03285]) industrial pattern: one batched
LoRA op + a paged adapter pool, with grouped GEMM (SGMV) for prefill and batched
gather-GEMV (BGMV) for decode.

### J.1 The custom batched-LoRA op

**Contract (per targeted projection).** One op instance per target projection
(`q`/`k`/`v`/`o`/`gate`/`up`/`down`), placed exactly where Phase 1 places its
delta branch — reading the base projection's activation `x` and producing a delta
that is **`Add`-ed onto the base `MatMulNBits` output**, reusing the **Phase-1
`Add` wiring verbatim** (`lora_inject.rs` `inject`, the terminal
`Y = Add(Y_base, scaled)`). The base int4 `MatMulNBits` is **never touched**.

```
// Illustrative op contract (NOT final Rust). Domain pkg.nxrt, op GroupedLoraDelta.
GroupedLoraDelta(
    x:        Tensor[tokens, K],       // slot 0: base projection activation (fp16/fp32)
    segments: Tensor[tokens] (i32)     // slot 1: per-row adapter/segment routing descriptor
                                       //         (row -> pool page id; see §J.2/§J.4)
    pool:     WeightHandle (lazy)      // slot 2: handle/ref to the paged A/B pool (§J.2)
) -> delta: Tensor[tokens, N]          // added onto the base projection output by the reused Add
// Attributes (baked at injection, from the manifest §J.6):
//   k, n                : this projection's dims (n = slice width for a fused qkv target)
//   fused_slice         : Option<(offset,width)> — the qkv column range this op owns
//   target_module_id    : selects which A/B factor within each adapter's page this op reads
//   max_rank            : column budget for the intermediate [tokens, r] (capacity/padding)
```

- **Output is the delta only.** The op emits `[tokens, N]`; the existing `Add`
  applies it. For a **fused `qkv_proj`** target we keep the Phase-1 decision: the
  op owns one Q/K/V **column slice** (`fused_slice` from `FusedGroup`,
  `lora_inject.rs:134` — `slices: [(role, offset, width); 3]`), so three op
  instances (or one op with a per-role segment field) scatter into the fused
  `[.., N]` output and a single `Add` folds them onto `Y_base`. MLP `gate/up/down`
  and attention `o_proj` are standalone (`Placement::Direct`).
- **`x` is the whole batch's rows for that projection**, already laid out
  `[tokens, K]` by the plan; `segments[row]` says which adapter (which pool page)
  row *r* uses. Mixed adapters in one batch is the entire point (§J.3, §J.4).

**Where it plugs into our op-registration + dispatch (exact seam).**
1. **Registration.** Add a factory in
   **`crates/onnx-runtime-ep-cpu/src/kernels/mod.rs`**, in `build_cpu_registry()`
   (`mod.rs:250`), alongside the existing custom ops:
   ```rust
   reg.register(OpKey::new("GroupedLoraDelta", "pkg.nxrt", 1),
                Box::new(grouped_lora::GroupedLoraDeltaFactory));
   ```
   This is byte-for-byte the same seam used for `BlockQuantizedMatMul`
   (`mod.rs:260`), `IndexShare`, `SparseKvGather`, `CompressedSparseAttention`
   (`mod.rs:312-319`). Custom ops live in the private **`pkg.nxrt`** domain by
   convention. The kernel is a new `kernels/grouped_lora.rs` implementing
   `KernelFactory::create` (`ep-api/src/registry.rs:39-40`) → `Box<dyn Kernel>`
   (`ep-api/src/kernel.rs:448`).
2. **Dispatch.** No change to the dispatch loop: `exec_kernel_node`
   (`dispatch.rs:293`) resolves the op through `cache.get_or_create`
   (`dispatch.rs:415`) by `(op_type, domain, opset)` and runs it via
   `ctx.execute_kernel` (`dispatch.rs:499`). The op reads `x` and `segments` as
   ordinary `TensorView`s.
3. **Pool delivery through the lazy-weight seam.** The pool handle (slot 2) is
   delivered exactly like the block-quantized-MoE offload weight: extend
   **`LazyWeightBoundary`** (`ep-api/src/weight.rs:95`) with a `GroupedLora`
   variant and its `matches("pkg.nxrt","GroupedLoraDelta")` (`weight.rs:101`).
   The dispatch already gates lazy delivery on
   `LazyWeightBoundary::_.matches(domain, op_type)` and routes lazy inputs to
   `Kernel::execute_with_inputs(&[KernelInput], …)` (`dispatch.rs:335-342`,
   `kernel.rs:466`). The kernel receives the pool as a `WeightHandle::Lazy`
   (`weight.rs:166`) and indexes pages per row — **no per-token copy of adapter
   weights into the plan**, and the pool is never materialized wholesale.
4. **Overrides retire for the pool case.** `segments` is a fresh per-run input
   (fed by name, like today's overrides). The pool handle is bound once per
   session and rebound on eviction (§J.2). The Phase-1 `OptionalOverride` A_t/B_t
   inputs are **not** used by the grouped op (§J.5).

**Constant-input safety.** `segments` and the pool handle must be excluded from
the per-dispatch `constant_inputs` set so the kernel never prepacks stale routing
or weights — the same guard Phase 1 already applies to overrides
(`dispatch.rs:360,400-412`). For the pool handle this is automatic (a lazy
`WeightHandle` is delivered, not a constant initializer); for `segments`, register
it like an override (feedable-by-name, non-constant).

### J.2 Paged adapter weight pool

**Layout.** One host arena holding many adapters' already-transposed factors
(§A: `A_t[K,r]`, `B_t[r,N]`, contiguous, fp16/fp32). The natural unit is a
**page = one (adapter, target_module) factor pair**, because ranks differ
per-module (`rank_pattern`) and per-adapter:

```
Pool {
  arena:  Vec<u8>,                        // one contiguous host allocation (aligned)
  pages:  Slab<PageId, PagePlacement>,    // PagePlacement { adapter_id, module_id, kind: A|B,
                                          //   byte_offset, k, r, n, dtype }
  index:  HashMap<(AdapterId, ModuleId), (PageId /*A*/, PageId /*B*/)>,
  lru:    IntrusiveLru<PageId>,           // eviction order
  budget: ByteBudget,                     // scheduler/src/byte_budget.rs:84
}
```

- **Adapter-major vs rank-major.** Store **adapter-major** (an adapter's A_t/B_t
  factors are contiguous per module) so activating an adapter touches one
  cache-warm region and BGMV decode gathers one page per row. Rank-major (all
  A_t of equal rank co-located) only helps if we ever pad every adapter to a
  single global rank, which the per-module `rank_pattern` forbids (§A). Keep
  adapter-major; the grouped kernel indexes by `(page.byte_offset, r, n)`.
- **Alignment / contiguity.** The kernel needs each `A_t`/`B_t` page **contiguous
  and 64-byte aligned** (MLAS/AVX GEMM tile requirement; the CPU MatMul already
  assumes contiguous inputs, §A "CUDA rejects strided views"). The loader already
  produces contiguous transposed factors (`lora_inject.rs` `LoraModuleSpec.a_t`
  is `[K,r]` contiguous `TensorData`); the pool copies each into an aligned arena
  slot at load. A per-page 0-padding of `r` up to a small alignment (e.g. next
  multiple of 8) simplifies the SGMV group stride without merging ranks.
- **Load / evict.** LRU, **byte-budgeted through the existing `ByteBudget`**
  (`byte_budget.rs:84` — `try_reserve`/`release`/`reconfigure`, saturating,
  thread-safe, live-reconfigurable), the same primitive the KV scheduler uses.
  Loading an adapter `try_reserve(a_bytes + b_bytes)`; on shortfall, evict LRU
  cold pages (not the ones referenced by any live batch row — see §J.4 pinning)
  until it fits, then `release` the evicted bytes. Reuse the `LoraManager` LRU
  scaffolding (`onnx-genai-engine/src/lora/manager.rs`) — it already caches
  decoded adapters under a byte budget; Phase 2 promotes that cache into the
  arena-backed pool and adds the page index.
- **Per-row indexing.** `segments[row] -> AdapterId` (or directly a `PageId`
  pair) → `index[(adapter, this_op.module_id)]` → `(A_page, B_page)` →
  `(byte_offset, r, n)`. The op computes `delta[row] = scale · (x[row] @ A_t) @
  B_t` reading straight from the arena at those offsets. Decode gathers one
  `(A,B)` pair per row (BGMV); prefill groups rows by page (SGMV).

### J.3 SGMV (prefill) vs BGMV (decode) kernels

Both are variants of the **same** `GroupedLoraDelta` kernel; the op picks by the
batch's token/segment shape.

- **BGMV — decode (M≈1 per sequence, one token each, many sequences).** Each row
  is one token that may use a different adapter → a **batched gather + GEMV**:
  for row *r*, gather `(A_t, B_t)` for `segments[r]`, compute
  `x[r,K] @ A_t[K,r] -> t[1,r]`, then `t @ B_t[r,N] -> delta[1,N]`. This is the
  Punica BGMV shape (bandwidth-bound; one page-pair per row). Rows are
  independent → trivially parallel over the batch. This is the steady-state hot
  path and must be allocation-free per token (reuse a `[batch, max_rank]`
  intermediate scratch, sized by the `max_rank` attribute).
- **SGMV — prefill (variable-length segments, one adapter per prompt).** A prompt
  is a **contiguous segment** of many tokens sharing one adapter → a **segmented
  GEMM**: sort/group rows by adapter, then per group do a dense
  `X_g[m_g,K] @ A_t -> T_g[m_g,r]`, `T_g @ B_t -> D_g[m_g,N]`. This amortizes the
  GEMM over the segment (compute-bound, tiles well). The **segment descriptor**
  is a CSR-like `(group_ptr[num_groups+1], group_adapter[num_groups])` derived
  from `segments` — Punica's SGMV layout.

**Grouping / segment descriptor.** The op consumes `segments: [tokens]` (page/
adapter id per row). SGMV builds group offsets from runs of equal id (prefill is
already adapter-homogeneous per prompt, so runs are long); BGMV uses `segments`
directly as a per-row gather index. The descriptor is produced by the scheduler
(§J.4) and fed by name each run.

**Numerics — fp32 accumulators are mandatory.** Accumulate both GEMMs in **fp32**
even when `A_t`/`B_t` and `x` are fp16. This is the **flash-attention lesson from
this codebase**: fp16 accumulation flips razor-thin greedy-argmax ties at
realistic activation scale (§I item 5; CUDA flash-attention required fp32 accum,
`docs/CUDA_FLASH_ATTENTION.md`). The delta rides on a wide-`N` int4 projection,
so fp16 accum over `K` then `r` compounds error. **Contract: inputs/weights may
be fp16; the two matmul accumulators and the `scale·` multiply are fp32; the
delta is cast to the branch dtype only at the final store**, matching the §E
golden tolerance. Reuse the existing golden test harness (§E) with mixed adapters.

**Single-adapter fallback (must be cheap).** When a whole batch resolves to **one
adapter** (`segments` is constant, the common single-tenant case), the op must
**collapse to the Phase-1-equivalent dense path** — one `X[M,K] @ A_t @ B_t`, no
grouping, no gather, no per-row indexing overhead — i.e. exactly the two dense
MatMuls Phase 1 emits. Detect a constant `segments` (or a `num_groups==1` SGMV
descriptor) and take the dense branch. This guarantees Phase 2 is never slower
than Phase 1 for the dominant single-adapter workload (§J.5).

### J.4 Scheduler routing / per-request `adapter_id`

**Threading `adapter_id`.** Add `adapter_id: Option<AdapterId>` to
`scheduler::Request` and `RunningSequence` (`scheduler/src/lib.rs:45,62`) and to
the engine's `ContinuousBatchRow` (`batched.rs:72`). Each step, the batch builder
emits the **segment descriptor** by reading each row's `adapter_id` → `PageId`
(pool index, §J.2) and writing `segments[physical_row]`. That descriptor is the
op's slot-1 input. `adapter_id == None` maps to a reserved **null page** whose
factors are the `r=0` empty delta (the base-only row), so mixed base+adapter
batches are free.

- **Two batching modes, phased.**
  - **P2c: adapter-homogeneous batching (simpler first step).** Group requests so
    a batch shares one adapter (or base-only). `segments` is then constant and the
    op takes the §J.3 dense fallback — correct, and it validates the whole
    subsystem (pool, injection, manifest, descriptor plumbing) **without** needing
    the grouped kernel to be fast. This is the safe on-ramp.
  - **P2e: true mixed-adapter batch (the payoff).** Different rows carry different
    adapters in one decode step; BGMV/SGMV do the grouped work. This is the whole
    reason for grouped GEMM and is where throughput under many hot adapters wins.
- **Resource pressure / admission.** When many adapters are hot, admission must
  consider **pool bytes**, not just KV bytes. Gate adapter activation on the
  pool's `ByteBudget` (§J.2); if a request's adapter cannot be paged in without
  evicting a page **referenced by a live row in the current batch**, either defer
  the request or evict a cold non-referenced adapter. **Pin** every page
  referenced by a running row (a small refcount over `RunningSequence.adapter_id`)
  so eviction never pulls weights out from under an in-flight step. Surface pool
  pressure through the existing pressure protocol (`scheduler/src/pressure.rs`)
  the same way KV pressure is surfaced.

### J.5 Reconciliation with Phase 1 — **unify** (recommended)

**Recommendation: the grouped op subsumes the 4-node subgraph; retire the
override A/B path for LoRA.** The `GroupedLoraDelta` op handles the
single-adapter case as a **pool-of-one** (§J.3 dense fallback), so Phase 1's
`MatMul→MatMul→Mul→Add` becomes a strict special case with no behavioral
difference (same math, same `Add`, same fp32 accumulation). Keeping two injection
shapes doubles the surface that must stay numerically identical and byte-safe.

- **What stays:** the injection *pass* (`lora_inject.rs`), the **manifest** and
  `FusedGroup`/`Placement` types (`lora_inject.rs:134-160`), the **transpose-at-
  load** factor format (§A), the reused terminal **`Add`** wiring, the
  `LoraManager` LRU/`ByteBudget`, and every §E correctness test (they now target
  the op). The `OptionalOverride` executor mechanism **stays as a general
  executor capability** (it is not LoRA-specific — `state.rs:286` note) and is
  still available; LoRA simply stops being its consumer.
- **What changes:** `inject` emits **one `GroupedLoraDelta` per target** instead
  of four nodes per target; A/B come from the **pool handle**, not override
  inputs; `scale` folds into the op (fp32). The `OverrideFeed` type
  (`lora_inject.rs`) is replaced by the `segments` descriptor + pool binding.
- **Migration.** Land the op behind the manifest so P4's single-adapter path
  keeps working through the **dense-fallback op** first (P2a/P2b), prove
  bit-parity against the Phase-1 4-node subgraph on the §E golden (both must
  produce identical output for one adapter), then delete the 4-node emission
  (P2f). Retiring, not coexisting, avoids a permanent two-path numerics burden.
  **If** the grouped op slips, the fallback is simply *not deleting* the 4-node
  path — coexistence is the contingency, unification is the goal.

### J.6 Manifest from `InferenceMetadata` (declared-primary)

Phase 2 makes the **exporter-declared manifest the primary source**; graph
discovery (Phase 1's `build_manifest`) becomes the **fail-loud fallback**.

- **Schema.** Extend `LoraCapabilities`
  (`crates/onnx-genai-metadata/src/schema/adapters.rs:17`) — today advisory only
  (`available`/`default`/`target_module_policy`/`supports_hot_swap`) — with an
  optional **declared target manifest**: per semantic module → `{ layer, node
  name / value name, K, N, fused_slice: Option<(offset,width)>, role, per-module
  rank/alpha policy }`. This is the machine form of §C's `TargetEntry`/
  `FusedGroup`, authored by the exporter instead of rediscovered. Purely additive
  → same schema major version (the module's existing forward-compat rule).
- **How it drives injection.** When present, `lora_inject` builds its
  `LoraManifest` **directly from the declared entries** (resolving node/value
  names to `NodeId`/`ValueId`), skipping structural discovery. It still
  **validates** declared dims/offsets against the actual graph (K/N from the base
  `MatMulNBits` attrs, fused width sums to `N`) and **fails loud** on any mismatch
  — the declaration is trusted for *intent* but verified for *correctness*, so a
  stale export can never silently corrupt attention (§C invariant preserved).
- **Absent-manifest behavior.** No declared manifest ⇒ fall back to Phase-1
  structural discovery (`build_manifest`), which is already fail-loud on
  unrecognized layouts (fused vs split vs linear-attn, §C/§H). So a model exported
  before the Mobius change still works on the split/fused Qwen families and still
  refuses the layouts it cannot prove.
- **Cross-repo dependency (Mobius exporter).** Emitting the declared manifest is a
  **Mobius exporter change** (write the target manifest into `InferenceMetadata`
  at export time). Track it as an external dependency; Phase 2 does **not** block
  on it because the graph-discovery fallback covers the known families. Flag it as
  the recommended long-term source of truth (it removes the per-export structural
  guessing risk entirely for linear-attention and future layouts).

### J.7 CUDA sub-phase (clearly scoped later)

Deferred, unchanged prerequisites from Phase 1 (§F P5, §I items 3–4), **all
verified blocking today**:

- **Capture-safe `k==0` / null-page zero-fill.** CUDA `MatMul` misses the
  capture-safe GEMV when `k==0` and the fallback neither zero-fills nor captures
  (`ep-cuda/src/kernels/matmul.rs`, GEMV gated on `plan.k > 0`). The pool's null
  page (base-only rows) needs a capture-safe zero delta — either a fixed-rank
  zero-padded page with `scale=0`, or a `k==0` fast path added first.
- **Persistent device bindings + adapter pool on device.** Native CUDA rejects
  `Routed` step inputs (`native_decode/load.rs:353-365`); the pool handle and
  `segments` must be **direct `DecodeCudaState` bindings**, and the pool arena
  must live in **device** memory (a mirror of §J.2 on-device, paged by the same
  LRU/`ByteBudget`).
- **Capture invalidation on adapter-set change.** The device-graph replay
  signature is pointer+shape, not contents (`state.rs:562-606`,
  `bindings.rs:245-257`), so **same-shape** content swaps (a fixed-capacity pool)
  are capture-safe, but any change to the **set of resident pages / their
  addresses / `segments` shape** must invalidate and re-arm capture. Fixed device
  pool capacity (zero-padded ranks, stable addresses) is the simplest capture-safe
  design; a page swap keeps addresses and only rewrites contents.

CUDA is **P2g+**, gated on the CPU subsystem (P2a–P2f) landing and on real
multi-tenant GPU demand.

### J.8 Phased plan (P2a…P2g)

Ordered by dependency. P2a–P2f are CPU; P2g+ is CUDA.

| Phase | Deliverable | Deps | Risk | Size |
|---|---|---|---|---|
| **P2a** | **Paged pool + `ByteBudget` + page index** (host arena, adapter-major, LRU, pin refcount). Promote `LoraManager` cache into the arena. | P4 | med | L |
| **P2b** | **`GroupedLoraDelta` op — dense single-adapter path only** (the §J.3 fallback), registered at `mod.rs:250` in `pkg.nxrt`, pool delivered via a new `LazyWeightBoundary::GroupedLora`. Prove **bit-parity vs the Phase-1 4-node subgraph** on the §E golden. | P2a | **high** | L |
| **P2c** | **Adapter-homogeneous batching**: `adapter_id` on `Request`/`RunningSequence`/`ContinuousBatchRow`, constant-`segments` descriptor, pool pinning + pressure admission. End-to-end many-adapters-one-at-a-time. | P2b | med | M |
| **P2d** | **BGMV decode kernel** (per-row gather-GEMV, fp32 accum, allocation-free scratch) + **SGMV prefill kernel** (segmented GEMM, CSR descriptor). Numerics: mixed-adapter golden + on-model argmax check. | P2b | **high** | XL |
| **P2e** | **True mixed-adapter batch**: variable `segments`, scheduler emits per-row descriptor, fused-QKV scatter across three slices. The throughput payoff. | P2c, P2d | med | L |
| **P2f** | **Unify / retire Phase-1 4-node path** (§J.5) once P2b/P2e prove parity. Delete the 4-node emission; keep `OptionalOverride` as a general executor capability. | P2e | low | M |
| **P2-mobius** | **Declared manifest in `InferenceMetadata`** (§J.6) + exporter emit (cross-repo). Runtime consumes declared-primary, graph-discovery fallback. Independent of P2d. | P2b | med | M (runtime) + external |
| **P2g+** | **CUDA sub-phase** (§J.7): device pool, direct bindings, capture-safe null page, capture invalidation. | P2f | **high** | XL |

**Biggest risk — the BGMV decode kernel's numerics-and-speed both at once (P2d).**
It is on the steady-state per-token hot path, must be allocation-free, and must
carry fp32 accumulators without regressing decode latency or flipping argmax ties
on real models. A grouped gather-GEMV that is either too slow (worse than serial
per-adapter) or too loose (fp16 accum) sinks the value proposition. Mitigation:
land the dense fallback (P2b) and adapter-homogeneous mode (P2c) first so the
subsystem is correct and useful **before** the grouped kernel exists, and gate
P2d behind the §E golden **plus** an on-model argmax-parity check.

**Biggest unknown — real mixed-adapter batch composition, and whether grouped
GEMM actually beats group-by-adapter at our batch sizes.** Punica/S-LoRA numbers
assume many concurrent adapters and large batches; our decode batches may be
small and often single-adapter (§J.3 fallback covers that). If production traffic
is mostly one-adapter-per-batch, P2e's grouped kernel is dead weight and P2c is
the real product. We do **not** yet have workload traces to size this. **Honest
recommendation: build P2a–P2c first, measure adapter-mixing in real traffic, and
only then commit to P2d/P2e.** Do not build the grouped kernel on faith.

**Where the decided architecture meets a real codebase constraint (owner should
know):**
1. **The lazy-weight seam is currently hard-coded to one boundary.**
   `LazyWeightBoundary` is an enum with a single `BlockQuantizedMoe` variant and
   `matches` is a `matches!(self, Self::BlockQuantizedMoe) && domain=="pkg.nxrt"
   && op_type=="BlockQuantizedMoE"` literal (`ep-api/src/weight.rs:95-105`).
   Delivering the pool handle this way needs a **new variant + a second `matches`
   arm**, and the dispatch gate (`dispatch.rs:335-342`) generalizing from one
   boundary to a set. Small, but it is a real API touch, not free.
2. **`segments` must be a first-class per-run input, not an initializer.** It
   changes every step and must be excluded from `constant_inputs`
   (`dispatch.rs:360,400-412`) exactly like an override — otherwise a kernel could
   prepack stale routing. Registering a non-constant, feedable-by-name **runtime**
   input (not an override with a default) is a slightly different shape than the
   P1 `OptionalOverride`; confirm the executor can carry a plain optional runtime
   input alongside the override set.
3. **Fused-QKV forces either three ops or a per-role segment.** A single op cannot
   cleanly own three disjoint output slices with three independent ranks; the
   clean encodings are (a) three op instances (one per Q/K/V slice, each with its
   `fused_slice`) sharing one `Add`, or (b) one op with a per-role sub-descriptor.
   (a) is simpler and reuses the Phase-1 slice logic (`FusedGroup.slices`); it
   means up to 3× the op count on fused exports. Acceptable, but note it.

### J.9 What is wired end-to-end today (multi-adapter reachability)

*Status as of 2026-07-29 (branch `feat/native-lora-p2`, PR #374).* The grouped
subsystem (§J.1 op, §J.2 pool, §J.6 declared-manifest resolution) was fully
built and unit-tested but had **zero production callers** (notably the
`BudgetedLoraPool` control plane); this revision wires the shared byte budget
into grouped admission (see "Shared byte-budget governance" below). This section
records what is now reachable end-to-end versus what remains deferred. Nothing
here re-litigates §J.1–§J.8; it only states the current wiring.

**User-facing surface (CPU, native backend).**

* **CLI.** The existing `--adapter <PATH>` (single, always-on) is unchanged. A
  new **repeatable** `--adapters <NAME=PATH>` preloads several named adapters at
  once, and `--select-adapter <NAME>` on `generate`/`run` chooses which
  preloaded adapter applies to that request (omit ⇒ base model). `--adapter` and
  `--adapters` are mutually exclusive. A **single** `--adapters` entry collapses
  to the same always-on single-adapter fast path as `--adapter` (see fast-path
  note below) — the plural, per-request-selectable path engages only with **two
  or more** adapters.
* **Config.** `EngineConfig.lora_adapters: Vec<(String, PathBuf)>` (the plural,
  grouped form) sits beside the existing `lora_adapter: Option<PathBuf>` (single,
  DIRECT). `GenerateOptions.adapter: Option<String>` selects a preloaded adapter
  by its identifier per request. The identifier is the explicit `NAME`, else the
  file stem.

**Fast-path preservation (the hard perf gate).** The grouped pool, the
`GroupedLoraDelta` op, and the `lora.segments` feed are constructed **only when
two or more adapters are configured**. Zero adapters and a single adapter both
stay on the Phase-1 DIRECT 4-node path — the pool/registry/segments machinery is
never built for ≤1 adapter, so the no-adapter and single-adapter paths are
byte-for-byte the code they were before this change.

**How an adapter identifier threads request → segments.**
`GenerateOptions.adapter` (a name) → `NativeDecodeSession::select_lora_adapter`
resolves it to an `AdapterId` via the session's name→id map (unknown name ⇒
typed `UnknownLoraAdapter` error at admission, **never** a silent base
fallback) → the resolved id is held as the session's *active route* → each
decode step writes that route into a **reused** `lora.segments` Int32 buffer
(cleared and refilled in place; capacity retained across steps, so there is no
per-token heap allocation for the routing tensor after warmup) and binds it as
the non-constant runtime input the grouped op reads. A `None` route (no adapter)
writes `-1`, which the kernel treats as base-only.

**Injection generalization.** `inject_grouped_multi` admits **N** adapters under
distinct `AdapterId`s sharing one op set per target projection; adapter identity
comes **only** from `segments`. All adapters in one grouped session must target
an identical module set (same module name + layer index per position), enforced
fail-loud via `AdapterModuleSetMismatch`; this is an honest constraint for this
pass, not a silent truncation.

**Shared byte-budget governance (now wired).** Grouped-adapter admission is
routed through `BudgetedLoraPool` (engine `lora/pool.rs`) via a `LoraPoolSink`
control-plane trait (`onnx-runtime-ep-api`). Before the data-plane
`LoraWeightPool` admits each `(adapter, module)` factor pair, the pool reserves
that pair's page-aligned resident bytes from the **shared** `ByteBudget` — the
same instance the KV/device subsystem uses (`EngineResourceGovernor::byte_budget`,
threaded from `engine/load.rs` through `NativeDecodeSession` into the session
builder). An over-budget adapter set fails loud with a typed
`LoraInjectError::PoolBudgetExceeded { requested, used, limit, available,
shortfall }` instead of over-committing device memory. The reservation is
attached to the finished pool as its residency owner, so it releases exactly
once on session drop (preserving the RAII release). The ≤1-adapter DIRECT fast
path never builds the pool, so it is not budgeted (and stays byte-for-byte
Phase-1). NOTE: this is admission-time reservation + fail-loud only; the §J.2
LRU **eviction/paging** of cold pages under budget pressure remains deferred
(P2a). Regression coverage: engine `tests/lora_grouped_budget.rs`
(over-budget admission fails loud and leaks nothing; a successful load reserves
and then releases the shared budget to the exact prior level on drop).

**Collapsed single adapter + `--select-adapter`.** When a single `--adapters
NAME=PATH` collapses to the DIRECT fast path, the selectable `NAME` is retained
(`EngineConfig.lora_adapter_name`). A request `--select-adapter NAME` for that
same adapter is a **no-op** (it is already applied to every token); any other
name fails loud with a message that NAMES the actually-loaded adapter, rather
than the previous misleading "session was not loaded with a grouped pool" error.

**Wired (reachable + tested).**

* config → CLI → engine load of N adapters → grouped injection + `BudgetedLoraPool`
  registration (RAII-owned for the session lifetime, so budget/registry release
  on teardown — no reintroduced leaks) → session build → per-request selection →
  decode-loop `segments` feed.
* **Per-request adapter selection on the native single-session backend**: each
  `generate` call runs under one selected adapter (or base). Because a single
  generation is one sequence, every run hits the kernel's uniform-batch fast
  path. This is the shipped product surface.
* Tests: `onnx-runtime-session` `grouped_two_adapters_route_per_row` (per-row
  routing `[A,B,base,A]` through the executor) and
  `grouped_multi_adapter_module_mismatch_fails_loud`; engine
  `engine_multi_adapter_grouped_selects_per_request` (builds one session
  preloading two adapters via `SessionBuilder::lora_adapters`, then selects
  adapter A vs B vs base per run and asserts each output equals that adapter's
  delta, A ≠ B, and an unknown name fails loud). The single-adapter
  `engine_lora_path_applies_and_reverts_adapter_delta` still passes unchanged.
* **Mixed-adapter rows within ONE continuous batch** (§J.4/§J.5 P2e): the
  `ContinuousBatchManager` now threads a per-row `adapter_id` end to end. Each
  `PendingContinuousRequest` / `ContinuousBatchRow` carries the `lora_route`
  (`i32` segment id, `-1` == base) it was admitted with; the route is resolved at
  `submit` time from the request's `GenerateOptions.adapter` against the loaded
  adapter→route map (`resolve_lora_route`), failing loud on an unknown name
  (reusing the "unknown LoRA adapter" message that names the loaded adapters) and
  on an explicit adapter when no grouped pool is configured — never a silent base
  fallback. Before every decode call the manager builds the `lora.segments`
  tensor per row into a **reused scratch buffer** (`lora_route_scratch`, cleared
  and refilled in place with no per-step heap allocation after warmup) and feeds
  it through the new `BatchedDecodeSession::set_lora_routes` trait method:
  `feed_physical_lora_routes` fills physical-row-indexed segments (empty slots →
  base) before `step_select` / prefill, and `feed_active_lora_routes` fills
  active-row-ordered segments before `step_active`, matching how each call orders
  its rows. `set_lora_routes` defaults to a no-op, and the whole path is skipped
  when `lora_adapter_routes` is empty, so the ≤1-adapter / all-base fast path
  stays byte-for-byte Phase-1 and the ORT `BatchedStaticCacheDecodeSession`
  (which has no grouped op) is unaffected.
* Test: engine `continuous_batch_routes_mixed_adapters_per_row`
  (`native-backend`) drives the **real** `ContinuousBatchManager` with three
  submitted requests — bound to adapter A, adapter B, and base — through a
  decode-session double (`GroupedProbeSession`) that runs the **real** grouped
  `GroupedLoraDelta` kernel via a native `InferenceSession`. The zero base model
  makes each row's argmax depend solely on the per-row route: the A row emits A's
  delta argmax, the B row emits B's, the base row emits token 0. It is
  non-tautological — a whole-batch-to-base manager emits token 0 for every row
  and fails (verified by temporarily forcing all routes to base).
  `continuous_batch_unknown_adapter_fails_loud` covers the admission-time
  fail-loud path.

**Deferred (honestly not done in this pass).**

* **Native engine-level continuous-batch decode backend**: the per-row routing
  wiring above lives in `ContinuousBatchManager` + `BatchedDecodeSession` and is
  proven against a decode-session double running the real grouped kernel. It is
  NOT yet reachable from a production native continuous-batch backend, because
  today's `ContinuousBatchManager` runs only on the ORT backend
  (`BatchedStaticCacheDecodeSession`), whose `libonnxruntime.so` session has no
  `GroupedLoraDelta` op, and the native backend does not yet expose a
  continuous-batch decode session. `generate_batched_static` remains base-only.
  Wiring a native KV-cache continuous-batch decode session that implements
  `set_lora_routes` against the grouped `InferenceSession` is the remaining step
  to ship mixed-adapter batching as a product surface.
* **CUDA** grouped LoRA (§J.7): the grouped path is CPU-only and fails loud if a
  grouped pool meets a CUDA execution provider.
* **Speculative decoding + grouped adapter** combined: rejected fail-loud (the
  draft/verify paths do not yet thread the route).
