//! # `onnx-runtime-memory`
//!
//! Liveness-based **activation memory planning** for the ORT 2.0 runtime.
//!
//! ## Why this crate exists
//!
//! The user's north-star goal is to *run any-size model even when VRAM/RAM is
//! insufficient, as long as the whole system has enough storage*. Zero-copy
//! weight streaming already removes weight copies from the budget. The next big
//! lever is **activation memory**.
//!
//! Today the executor allocates **one buffer per graph value for its whole
//! lifetime**, so peak activation memory is `SUM(every intermediate tensor)`
//! rather than the *concurrent* peak. Most intermediates are dead long before
//! the run ends. A liveness-based planner shares one physical buffer among
//! values whose lifetimes do not overlap, cutting peak activation memory from
//! `O(N nodes)` to `O(max concurrent live set)` — often a multiple-× reduction
//! and the key to fitting big models.
//!
//! This crate is the **pure, deterministic planning algorithm**, deliberately
//! decoupled from the risky executor surgery (which lands later). It depends
//! only on [`onnx_runtime_ir`], contains no `unsafe`, and is free of PyO3 / EP /
//! session dependencies so it is trivially testable in isolation.
//!
//! ## What it computes
//!
//! Given a [`Graph`](onnx_runtime_ir::Graph), a [`ViewMap`] of zero-copy view
//! aliases, and a **size oracle** (`Fn(ValueId) -> Option<usize>`), the planner
//! produces an [`ActivationPlan`]:
//!
//! * `assignments: ValueId -> SlotId` — which reusable slot backs each value.
//! * `slots: Vec<SlotInfo>` — each slot's byte capacity.
//! * `peak_bytes` — the arena size the executor must allocate (sum of slot
//!   capacities) — the shared, concurrent-peak footprint.
//! * `naive_bytes` + `savings_ratio` — the one-buffer-per-value baseline and the
//!   proven reduction.
//!
//! The three-step algorithm is: **(1)** compute each buffer owner's live
//! interval `[def, use_end]` in topological order, folding view consumers and
//! graph-output liveness into the root owner; **(2)** size every owner via the
//! oracle, returning [`PlanStatus::Deferred`] if any size is symbolic; **(3)**
//! greedily walk nodes, allocating each output's slot (best-fit reuse of a
//! retired slot, else a new one) and retiring inputs *after* the node so a
//! node's own inputs and any graph output are never clobbered.
//!
//! ## Static vs. dynamic shapes
//!
//! The same algorithm serves **build-time** and **run-time** planning through
//! the size oracle. [`plan_activations_static`] plans from fully-static shapes;
//! any symbolic-shaped activation makes the whole plan [`PlanStatus::Deferred`]
//! so the executor can re-plan once shapes resolve for a run. A run-time caller
//! passes its own oracle backed by resolved shapes.
//!
//! ## Zero-copy view aliasing
//!
//! The executor treats layout/movement-op outputs (`Slice`, `Reshape`,
//! `Transpose`, …) as zero-copy views that own no buffer and alias a *source*
//! buffer, pinning the source so it outlives every alias. The planner mirrors
//! this: a view gets **no slot**; instead it extends its source's live interval
//! (transitively — a view of a view folds to the root buffer owner). Correct
//! view-liveness folding is exactly what stops a reused buffer from clobbering a
//! still-live alias. The caller supplies the `view -> source` edges via
//! [`ViewMap`]; op names are never hardcoded here.
//!
//! ## Intended executor integration (deferred follow-up — out of scope now)
//!
//! The follow-up PR wires this into `onnx-runtime-session`'s executor:
//!
//! 1. **Build the [`ViewMap`]** from the executor's existing view plan (the
//!    `views`/`pinned` machinery in `executor.rs`), mapping each view value to
//!    its source (root) buffer owner.
//! 2. **Call the planner** with a size oracle backed by the run's *resolved*
//!    shapes (build-time static shapes where available; a per-run oracle
//!    otherwise). On [`PlanStatus::Deferred`], re-plan once shapes resolve.
//! 3. **Allocate `peak_bytes`** as one arena (or `num_slots` `DeviceBuffer`s),
//!    then map each [`SlotId`] to an offset/allocation.
//! 4. **Hand each value a `TensorMut` window** into its assigned slot instead of
//!    the current per-value `buffers: HashMap<ValueId, DeviceBuffer>`.
//!
//! Two concerns are explicitly **out of scope** for both this crate and the
//! current planning contract, and must be handled by the integration PR:
//!
//! * **In-place ops** — an op that may safely overwrite an input (e.g. an
//!   elementwise unary) can share its input's slot for its output. Detecting
//!   this requires per-op semantics the planner does not model; until then the
//!   planner conservatively gives each output its own (reused) slot.
//! * **Fragmentation** — `peak_bytes` is the sum of slot capacities. Packing
//!   slots into a single arena with alignment/offset assignment (and any
//!   resulting internal fragmentation) is the executor's responsibility.
//!
//! ## Correctness invariants (enforced by [`validate`])
//!
//! * No two values with overlapping live intervals share a slot.
//! * Every graph output has a slot that is never reused after its def.
//! * A pinned source outlives every view aliasing it (fold correctness).

#![forbid(unsafe_code)]

mod error;
mod liveness;
mod options;
mod oracle;
mod plan;
mod validate;
mod view_map;

pub use error::{PlanError, ValidateError};
pub use liveness::{compute_liveness, Interval, Liveness};
pub use options::PlanOptions;
pub use oracle::{static_size, static_size_oracle};
pub use plan::{
    plan_activations, plan_activations_static, ActivationPlan, PlanStatus, SlotId, SlotInfo,
};
pub use validate::{validate, validate_static};
pub use view_map::ViewMap;
