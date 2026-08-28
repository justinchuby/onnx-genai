//! Named owners for the three kinds of state a workflow runtime holds.
//!
//! `docs/architecture/SESSION_CONCURRENCY.md` §3 classifies everything
//! reachable from a session handle into exactly one of three kinds, and states
//! that the classification *is* the design. Until now the classification lived
//! only in that document: [`super::WorkflowRuntime`] held the compiled plan, the
//! backend handles, the caches derived from them and the per-pass counters as
//! one flat field list, so which category a field belonged to was a fact about
//! the comment above it rather than about its type.
//!
//! This module gives each category an owner:
//!
//! | Category | Owner | Rule |
//! |---|---|---|
//! | §3.1 immutable plan | [`WorkflowPlan`] | frozen at load, shareable behind `Arc` — and `Send + Sync` because it is frozen, not because a lock guards it |
//! | §3.2 backend handles | [`WorkerBackend`] | sessions, islands and native components, owned by exactly one thread |
//! | §3.2 storage under §3.3 access | [`WorkerRuntimeState`] | bindings, allocators, device values, caches, counters and session cells the owning worker mutates |
//! | §3.3 per-execution | [`WorkflowPass`] | created when a pass starts, destroyed when it ends |
//!
//! Nothing here changes what the runtime does. At `W = 1` there is still one
//! worker and one thread; what changes is that a second worker can no longer be
//! added by accident, because the state it would have to own now has a name and
//! is structurally `!Send`.

use anyhow::Context;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use onnx_genai_metadata::decoder_workflow::IterationPolicy;
use onnx_genai_metadata::{CompiledWorkflow, PreprocessingSpec, WorkflowSpec};
use onnx_genai_ort::PipelineModels;

use crate::{EngineDecodeBackend, MemoryStrategyPlan};

use super::WorkflowOutputPublication;
use super::islands::ExecutionIsland;
use super::speculative::EmbeddingTable;
use super::turn_transaction::CommittedOutputState;
use super::workflow::{
    ComponentBindingKey, ComponentOutputKey, StableComponentBinding, WorkflowPerformanceCounters,
    WorkflowRunTelemetry,
};
use super::{WorkflowRuntime, adapters};

/// Makes the struct that holds it structurally `!Send` and `!Sync`.
///
/// §3.2's rule is that backend handles are "owned by exactly one worker, never
/// referenced from another thread". A rule like that is worth what the compiler
/// will enforce of it, and today most of these types are *incidentally* thread
/// bound: `RefCell` is `!Sync`, `Rc` is neither. Incidental is not a contract —
/// replacing one `Rc` with an `Arc` during an unrelated refactor would silently
/// hand a future `W > 1` the right to move ORT bindings between threads.
///
/// This marker states it instead, and costs nothing at runtime. It is a
/// `PhantomData`, not an `unsafe impl`: it *removes* an auto trait rather than
/// asserting one, so it cannot be wrong.
pub(crate) type ThreadBound = PhantomData<*const ()>;

/// Fails to compile if `$type` implements every trait listed.
///
/// The `!Send`/`!Sync` half of §3.2 cannot be written as an ordinary assertion,
/// because there is no `T: !Send` bound to test. Two blanket impls that overlap
/// exactly when the type does implement the traits turn it into an inference
/// ambiguity, which *is* a compile error. Nothing is evaluated at run time; the
/// `const` exists so the check is part of `cargo check`.
macro_rules! assert_not_impl_all {
    ($type:ty: $($trait:path),+ $(,)?) => {
        const _: fn() = || {
            trait AmbiguousIfImpl<A> {
                fn assertion() {}
            }
            impl<T: ?Sized> AmbiguousIfImpl<()> for T {}
            impl<T: ?Sized $(+ $trait)+> AmbiguousIfImpl<u8> for T {}
            // Ambiguous — and therefore a compile error — exactly when `$type`
            // implements all of the listed traits.
            let _ = <$type as AmbiguousIfImpl<_>>::assertion;
        };
    };
}

/// Fails to compile unless `$type` implements every trait listed.
macro_rules! assert_impl_all {
    ($type:ty: $($trait:path),+ $(,)?) => {
        const _: fn() = || {
            fn assert<T: ?Sized $(+ $trait)+>() {}
            assert::<$type>();
        };
    };
}

// The assertions in this module reach the macros by path; the re-export is what
// lets the rest of the crate's test modules state the same contracts.
#[cfg(test)]
pub(crate) use {assert_impl_all, assert_not_impl_all};

/// What identifies one cached embedding table: the component and initializer it
/// was read from, the residency it was made resident in, and the bit pattern of
/// the declared normalizer applied to its rows. Bits rather than the float
/// itself because a key has to be hashable and total.
pub(crate) type EmbeddingTableKey = (String, String, i32, Option<u32>);

/// The immutable half of a loaded package (§3.1).
///
/// The declared workflow, its compiled graph, the properties derived from that
/// graph once at load, the memory strategy plan and the resolved backend. None
/// of it is mutated after construction, which is why it is shareable behind an
/// `Arc` rather than copied per worker: cloning a plan per worker would
/// multiply host memory by the worker count for data that cannot change.
///
/// It is `Sync` because it is frozen. The static assertion below is the part
/// that keeps it true — a mutable cell added here would fail the build rather
/// than quietly making the plan unshareable.
pub(crate) struct WorkflowPlan {
    /// Directory the package was loaded from.
    pub(crate) package_root: PathBuf,
    /// The workflow this runtime executes.
    pub(crate) workflow: WorkflowSpec,
    /// The lowered graph, typed contracts and resolved bindings.
    pub(crate) compiled_workflow: CompiledWorkflow,
    /// Outputs this workflow fills one request row at a time, derived once from
    /// the compiled graph so every emit into one output agrees.
    pub(crate) row_wise_outputs: HashSet<String>,
    pub(crate) movable_emit_values: HashSet<String>,
    pub(crate) device_bridge_components: HashSet<String>,
    pub(crate) memory_strategy_plan: MemoryStrategyPlan,
    pub(crate) decode_backend: EngineDecodeBackend,
    /// Canonical construction-time decision reused by every execution entry.
    pub(crate) execution_admission: super::WorkflowExecutionAdmission,
    pub(crate) adapter_service: Option<onnx_genai_metadata::AdapterServiceContract>,
    pub(crate) preprocessing: Option<PreprocessingSpec>,
    /// The package's speculative compatibility contract, when it declares one.
    /// The chained proposal driver in [`super::speculative`] reads every field
    /// it needs from here, so proposal execution is metadata-driven rather than
    /// keyed on a model name.
    pub(crate) speculative: Option<onnx_genai_metadata::SpeculativeContract>,
}

// §3.1: "shared by every worker behind `Arc`, never mutated after load. These
// are `Sync` because they are frozen, not because a lock guards them."
assert_impl_all!(WorkflowPlan: Send, Sync);

/// The backend handles one worker owns (§3.2).
///
/// ORT component sessions, the fused execution islands built over them, and the
/// native component sessions when the native backend is compiled in. §3.2's rule
/// is that these are "owned by exactly one worker, never referenced from another
/// thread, constructed on the worker thread and dropped on the worker thread",
/// and [`ThreadBound`] is what makes the middle clause structural rather than
/// editorial — ORT's `Session` is itself `Send + Sync`, so without the marker
/// this struct would advertise a portability its bindings, allocators and values
/// do not have.
pub(crate) struct WorkerBackend {
    /// Component sessions and the package directory they were loaded from.
    pub(crate) models: PipelineModels,
    /// Fused multi-component subgraphs, each owning its own `Arc<Session>`.
    pub(crate) execution_islands: Vec<ExecutionIsland>,
    /// Native (pure-Rust) component sessions, present only when the engine was
    /// built for `EngineDecodeBackend::Native`. The universal interpreter drives
    /// these through the same seam it uses for ORT sessions; see
    /// `docs/architecture/NATIVE_WORKFLOW_BACKEND.md`.
    #[cfg(feature = "native-backend")]
    pub(crate) native_components: Option<RefCell<super::native_component::NativeComponentSet>>,
    pub(crate) thread_bound: ThreadBound,
}

// §3.2: backend handles never cross a thread boundary, and are not shared.
assert_not_impl_all!(WorkerBackend: Send);
assert_not_impl_all!(WorkerBackend: Sync);

impl WorkerBackend {
    pub(crate) fn new(
        models: PipelineModels,
        execution_islands: Vec<ExecutionIsland>,
        #[cfg(feature = "native-backend")] native_components: Option<
            RefCell<super::native_component::NativeComponentSet>,
        >,
    ) -> Self {
        Self {
            models,
            execution_islands,
            #[cfg(feature = "native-backend")]
            native_components,
            thread_bound: PhantomData,
        }
    }
}

/// Counters and diagnostics accumulated over a worker's life.
///
/// They were previously loose `Cell`/`RefCell` fields on the runtime, which made
/// "who may read these" a question about the whole runtime rather than about the
/// worker that owns them. Grouping them says it once.
#[derive(Default)]
pub(crate) struct WorkerCounters {
    /// Device→host materializations this worker performed.
    ///
    /// A proposal chain's whole point is that its per-token work stays on the
    /// device that produced it, and "stays on the device" is not something a
    /// throughput number diagnoses: a reintroduced copy shows up as a slower
    /// run months later, attributed to anything but the line that caused it.
    /// Counting the transfers makes it a property a test can hold.
    pub(crate) host_staging_count: Cell<u64>,
    /// Bytes this worker read back out of device memory deliberately.
    ///
    /// The counter above says *how many times* a whole tensor came down; this
    /// one says how much came down on the one path that is allowed to bring
    /// anything down at all — the token id a device argmax produces. Four bytes
    /// per row is the budget, and stating it as a number is what stops "only
    /// the token ids" from quietly becoming "the token ids and one small
    /// tensor".
    pub(crate) device_readback_bytes: Cell<u64>,
    /// How many times an embedding table was read out of an artifact.
    pub(crate) embedding_table_loads: Cell<u64>,
    /// Nodes this worker executed through a declared contract, by contract id.
    ///
    /// Selection of an algorithmic executor is supposed to come from what the
    /// workflow *authors*, and a claim like that is worth nothing unless
    /// something counts it. A test can assert that a package whose body names
    /// one contract routed its nodes there and nowhere else, which is a
    /// statement about the interpreter's dispatch rather than about which
    /// function a caller happened to call.
    pub(crate) contract_executions: RefCell<BTreeMap<String, u64>>,
    /// Timings folded in when a pass ends.
    pub(crate) workflow_performance: RefCell<WorkflowPerformanceCounters>,
}

/// Everything one worker mutates while it executes a plan (§3.2 storage under
/// §3.3 access discipline).
///
/// Every field here was a field of [`super::WorkflowRuntime`]. Gathering them
/// buys three things the flat list could not:
///
/// 1. The `Rc`/`RefCell`/`Cell` interior mutability is stated as *worker* state
///    rather than as an implementation detail of whichever field declared it.
/// 2. `ThreadBound` makes the whole group structurally `!Send` and `!Sync`, so
///    the compiler refuses to let a future pool hand it to a second thread.
/// 3. The values and allocators that [`super::WorkflowRuntime`]'s `Drop` must
///    release before the sessions they came from are in one place, so that
///    teardown is a statement about one owner rather than about a field list.
pub(crate) struct WorkerRuntimeState {
    /// Fixed-address ORT bindings reused across equal-shaped invocations. Each
    /// co-owns the session it was created from.
    pub(crate) component_bindings: RefCell<HashMap<ComponentBindingKey, StableComponentBinding>>,
    /// Device allocators created per component, co-owning their session.
    pub(crate) component_allocators:
        RefCell<HashMap<String, Arc<onnx_genai_ort::Allocator<'static>>>>,
    /// Stable output values allocated from those allocators. Released before
    /// them: a `Value` has no back-reference to its allocator (§3.4).
    pub(crate) component_outputs: RefCell<HashMap<ComponentOutputKey, Arc<onnx_genai_ort::Value>>>,
    /// Embedding tables read out of a component's artifact, cached for the
    /// worker's life.
    ///
    /// Re-reading a `[vocab, hidden]` initializer off disk once per proposal is
    /// pure waste — the file cannot change under a loaded package — and at real
    /// vocabularies it is the dominant cost of starting a proposal. The
    /// residency is part of the key because a device mirror and the host copy
    /// it was uploaded from are different tensors answering the same question,
    /// and the mirror must be uploaded once, not once per draft token. The
    /// declared normalizer joins them for the same reason: it changes what the
    /// cached rows are, not merely where they live.
    pub(crate) embedding_tables: RefCell<HashMap<EmbeddingTableKey, Rc<EmbeddingTable>>>,
    /// Immutable non-embedding initializers borrowed across components by a
    /// declared speculative contract.
    ///
    /// DFlash passes the target LM head into the proposer as a read-only input.
    /// Loading that matrix once per proposal would turn a metadata lookup into
    /// the dominant draft cost, while copying it into the proposer artifact
    /// would violate the declared shared-weight relationship.
    pub(crate) shared_initializers: RefCell<HashMap<(String, String), Rc<onnx_genai_ort::Value>>>,
    /// Session-scoped workflow cells, keyed by `(session id, cell)`.
    ///
    /// Per-session state living on the owning worker: §3.2 storage under §3.3
    /// access discipline, which is exactly what §3.3 says a conversation cell
    /// is.
    pub(crate) session_state: RefCell<HashMap<(String, String), onnx_genai_ort::Value>>,
    /// Durable effect cursors. Their payloads remain in the workflow's SSA
    /// working set; only this transaction-addressable progression is committed.
    pub(crate) session_effects: RefCell<HashMap<(String, String), u64>>,
    /// Durable output heads, cursors, lineage and closure facts. Output values
    /// stay pass-local until the enclosing transaction commits.
    pub(crate) session_outputs:
        RefCell<HashMap<(String, String, super::OutputStreamId), CommittedOutputState>>,
    /// Ordered transport-neutral publications from the last committed pass.
    /// This worker is thread-bound, so the execution plan can take the journal
    /// immediately without a second synchronization protocol.
    pub(crate) last_output_publications: RefCell<Vec<WorkflowOutputPublication>>,
    /// Sessions with a pass in flight, for leases declared `policy: exclusive`.
    ///
    /// Two turns of one conversation that both read the history before either
    /// writes it produce a last-write-wins conversation: the first turn's
    /// prompt and generation vanish. The declaration says the lease is
    /// exclusive, so this is what makes that true rather than assumed — a
    /// second concurrent turn is refused with a name, not silently lost.
    ///
    /// This is the *inner* lease of §4.2: it covers one worker's own passes.
    /// The routing-layer lease that covers the decode-core path too is Phase 2
    /// and is not in this state.
    pub(crate) session_leases: RefCell<HashSet<String>>,
    /// Monotonic committed-turn versions. A reusable execution plan records
    /// this before binding a continuation and must be rebuilt if another turn
    /// commits first.
    pub(crate) session_turn_versions: RefCell<HashMap<String, u64>>,
    /// Sibling interpreters over this package's loop re-authored for another
    /// iteration policy, built on first use and keyed by the policy.
    ///
    /// A continuous batch and a speculative block are *different iteration
    /// algorithms over the same declared generation*, and which one runs is
    /// answered by the contract the loop body's node names. The bodies that
    /// name them are authored by
    /// [`onnx_genai_metadata::decoder_workflow::iteration_variant`] from this
    /// package's own declaration — the same module that authored the
    /// single-token body — so the three cannot become three different
    /// statements of one loop.
    ///
    /// A variant's body invokes binding components only: the executor
    /// registered for its contract owns the session, the cache and the
    /// sampling. So a variant interpreter holds no models, no allocators and no
    /// governor, and building one costs a workflow compile rather than a load.
    pub(crate) iteration_runtimes: RefCell<BTreeMap<IterationPolicy, Rc<WorkflowRuntime>>>,
    pub(crate) adapter_cache: RefCell<adapters::AdapterCache>,
    pub(crate) active_adapter_context: RefCell<Option<adapters::AdapterRunContext>>,
    pub(crate) counters: WorkerCounters,
    /// Mints the id of each pass this worker runs (§3.3).
    pass_ids: PassIdAllocator,
    turn_transaction_ids: Cell<u64>,
    /// Mints CUDA graph capture ids for the component bindings above.
    graph_capture_ids: GraphCaptureIdAllocator,
    pub(crate) thread_bound: ThreadBound,
}

// §3.2/§3.3: worker state is owned by one thread. `Rc` already makes this true
// today; the assertion is what keeps it true after the next refactor.
assert_not_impl_all!(WorkerRuntimeState: Send);
assert_not_impl_all!(WorkerRuntimeState: Sync);

impl Default for WorkerRuntimeState {
    fn default() -> Self {
        Self {
            component_bindings: RefCell::new(HashMap::new()),
            component_allocators: RefCell::new(HashMap::new()),
            component_outputs: RefCell::new(HashMap::new()),
            embedding_tables: RefCell::new(HashMap::new()),
            shared_initializers: RefCell::new(HashMap::new()),
            session_state: RefCell::new(HashMap::new()),
            session_effects: RefCell::new(HashMap::new()),
            session_outputs: RefCell::new(HashMap::new()),
            last_output_publications: RefCell::new(Vec::new()),
            session_leases: RefCell::new(HashSet::new()),
            session_turn_versions: RefCell::new(HashMap::new()),
            iteration_runtimes: RefCell::new(BTreeMap::new()),
            adapter_cache: RefCell::new(adapters::AdapterCache::default()),
            active_adapter_context: RefCell::new(None),
            counters: WorkerCounters::default(),
            pass_ids: PassIdAllocator::default(),
            turn_transaction_ids: Cell::new(0),
            graph_capture_ids: GraphCaptureIdAllocator::for_component_bindings(),
            thread_bound: PhantomData,
        }
    }
}

impl WorkerRuntimeState {
    /// Open a pass and give it the next id in this worker's namespace.
    pub(crate) fn begin_pass(&self, max_iterations_only: bool) -> WorkflowPass {
        WorkflowPass {
            id: self.pass_ids.mint(),
            telemetry: WorkflowRunTelemetry::started(max_iterations_only),
        }
    }

    /// The pass this worker most recently opened, or [`PassId::NONE`] before the
    /// first one.
    pub(crate) fn current_pass(&self) -> PassId {
        self.pass_ids.current()
    }

    /// Mint a transaction identity before an admitted turn can mutate state.
    pub(crate) fn next_turn_transaction_id(&self) -> super::turn_transaction::TurnTransactionId {
        let next = self.turn_transaction_ids.get().saturating_add(1);
        self.turn_transaction_ids.set(next);
        super::turn_transaction::TurnTransactionId(next)
    }

    /// The capture id the next component binding owns.
    pub(crate) fn next_graph_capture_id(&self) -> anyhow::Result<GraphCaptureId> {
        self.graph_capture_ids.mint()
    }

    /// Release the ORT state that outranks its owner's field order.
    ///
    /// Values first, then the allocators they were allocated from: a `Value` is
    /// a bare `OrtValue` handle with no back-reference to its allocator, so ORT
    /// still requires it to be released first and nothing in the type system
    /// says so (§3.4). Bindings co-own their `Arc<Session>` and could be left to
    /// the field list, but they hold bound values, so they go first too.
    ///
    /// Called from [`super::WorkflowRuntime`]'s `Drop`, which is what #2019's
    /// contract requires and what §13 Phase 1 explicitly retains.
    pub(crate) fn release_ort_state(&mut self) {
        self.component_bindings.get_mut().clear();
        self.component_outputs.get_mut().clear();
        self.component_allocators.get_mut().clear();
    }
}

/// Identifies one execution pass of one worker.
///
/// Previously a bare `u64` named `workflow_execution_generation`, compared
/// against an equally bare `service_generation` on cached bindings. A newtype
/// makes "is this binding from the current pass?" a question that cannot be
/// asked of the wrong counter, which matters more once more than one worker can
/// mint one.
///
/// The namespace is *worker-local by declaration*: two workers may each mint
/// pass 7, and that is sound precisely because a pass id is only ever compared
/// against state the same worker owns. Nothing outside the worker observes it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct PassId(u64);

impl PassId {
    /// Before a worker has run anything. Cached state carrying this id belongs
    /// to no pass, so it always compares unequal to a live one.
    pub(crate) const NONE: Self = Self(0);
}

/// Mints [`PassId`]s for one worker, on the thread that owns it.
///
/// A `Cell`, not an atomic: this id never leaves the worker, and using an atomic
/// would advertise a sharing that §3.2 forbids. The counter wraps rather than
/// saturating, matching the pre-split behaviour — ids are compared for equality
/// against state written at most one pass ago, never ordered.
#[derive(Default)]
pub(crate) struct PassIdAllocator {
    next: Cell<u64>,
    thread_bound: ThreadBound,
}

assert_not_impl_all!(PassIdAllocator: Send);
assert_not_impl_all!(PassIdAllocator: Sync);

impl PassIdAllocator {
    pub(crate) fn mint(&self) -> PassId {
        let id = self.next.get().wrapping_add(1);
        self.next.set(id);
        PassId(id)
    }

    pub(crate) fn current(&self) -> PassId {
        PassId(self.next.get())
    }
}

/// A CUDA graph capture id — ORT's `gpu_graph_id` run-config entry.
///
/// The namespace is one ORT session: ORT keys captured graphs by this id
/// *within* a session, so two sessions may both use 0 and a single session must
/// never reuse one for a different binding. Both halves of that were previously
/// implicit — one call site used the length of a `HashMap`, another seeded a
/// counter at `island_id * 1000` — which is a partitioning scheme with no
/// keeper. Naming the type puts the rule where the id is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct GraphCaptureId(i32);

impl GraphCaptureId {
    /// Run without capturing or replaying, which ORT spells `-1`.
    pub(crate) const UNCAPTURED: Self = Self(-1);

    /// The value ORT is handed.
    pub(crate) fn get(self) -> i32 {
        self.0
    }
}

/// Mints [`GraphCaptureId`]s inside one session's namespace, on the thread that
/// owns that session.
///
/// Worker-local by declaration, like [`PassIdAllocator`]: a capture id is only
/// meaningful to the session it is replayed against, and §3.2 gives that session
/// to exactly one worker. Exhaustion is an error rather than a wrap, because
/// wrapping would silently replay a graph captured for a different binding.
pub(crate) struct GraphCaptureIdAllocator {
    next: Cell<i32>,
    thread_bound: ThreadBound,
}

assert_not_impl_all!(GraphCaptureIdAllocator: Send);
assert_not_impl_all!(GraphCaptureIdAllocator: Sync);

impl GraphCaptureIdAllocator {
    /// The allocator for a worker's component bindings, which start at 0.
    pub(crate) fn for_component_bindings() -> Self {
        Self {
            next: Cell::new(0),
            thread_bound: PhantomData,
        }
    }

    /// The allocator for one execution island.
    ///
    /// Islands are numbered from 0 and each stride of 1000 belongs to one
    /// island, which is the partition the pre-split code applied by hand. Each
    /// island owns its own session, so the stride is not what makes the ids
    /// unique — it is kept because it makes a captured graph's id say which
    /// island captured it when it turns up in an ORT log.
    pub(crate) fn for_island(island: usize) -> Self {
        Self {
            next: Cell::new(
                i32::try_from(island)
                    .unwrap_or(i32::MAX)
                    .saturating_mul(1000),
            ),
            thread_bound: PhantomData,
        }
    }

    pub(crate) fn mint(&self) -> anyhow::Result<GraphCaptureId> {
        let id = self.next.get();
        anyhow::ensure!(
            id >= 0,
            "CUDA graph capture ids for this session are exhausted: the next id would be {id}, \
             and ORT reserves negative ids for uncaptured runs"
        );
        // Advanced before the id is handed out, so an allocator that cannot name
        // its successor refuses rather than handing out an id it would repeat.
        let next = id
            .checked_add(1)
            .context("CUDA graph capture id exceeds i32")?;
        self.next.set(next);
        Ok(GraphCaptureId(id))
    }
}

/// State that lives for exactly one interpreter pass (§3.3).
///
/// "Created at the start of a turn, owned by the executing worker, destroyed at
/// the end of the turn — including on error, cancellation, and panic." The pass
/// id and the telemetry it accumulates are the two things that were previously
/// separate locals with no statement that they belonged to the same pass.
pub(crate) struct WorkflowPass {
    pub(crate) id: PassId,
    pub(crate) telemetry: WorkflowRunTelemetry,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_ids_are_unique_within_a_worker() {
        let allocator = PassIdAllocator::default();
        assert_eq!(
            allocator.current(),
            PassId::NONE,
            "a worker that has run nothing is in no pass"
        );
        let minted = (0..1_000).map(|_| allocator.mint()).collect::<Vec<_>>();
        let unique = minted.iter().copied().collect::<HashSet<_>>();
        assert_eq!(unique.len(), minted.len(), "pass ids collided");
        assert!(
            !unique.contains(&PassId::NONE),
            "a minted pass must never be mistaken for 'no pass'"
        );
        assert_eq!(allocator.current(), *minted.last().unwrap());
    }

    #[test]
    fn separate_workers_mint_their_own_pass_namespace() {
        // Worker-local by declaration: two workers minting the same number is
        // sound because a pass id is only compared against state the same
        // worker owns. This test states that, so a future change that starts
        // comparing ids across workers has to break it deliberately.
        let one = PassIdAllocator::default();
        let two = PassIdAllocator::default();
        assert_eq!(one.mint(), two.mint());
    }

    #[test]
    fn component_binding_capture_ids_are_unique_and_non_negative() {
        let allocator = GraphCaptureIdAllocator::for_component_bindings();
        let minted = (0..64)
            .map(|_| allocator.mint().expect("capture ids available"))
            .collect::<Vec<_>>();
        assert_eq!(
            minted.first().copied(),
            Some(GraphCaptureId(0)),
            "the first component binding owns capture id 0, as it did before the split"
        );
        let unique = minted.iter().copied().collect::<HashSet<_>>();
        assert_eq!(unique.len(), minted.len(), "capture ids collided");
        assert!(
            minted.iter().all(|id| id.get() >= 0),
            "a capture id must never collide with the uncaptured sentinel"
        );
        assert!(!unique.contains(&GraphCaptureId::UNCAPTURED));
    }

    #[test]
    fn island_capture_id_namespaces_do_not_overlap() {
        let first = GraphCaptureIdAllocator::for_island(0);
        let second = GraphCaptureIdAllocator::for_island(1);
        let first_ids = (0..1_000)
            .map(|_| first.mint().expect("capture ids available"))
            .collect::<HashSet<_>>();
        let second_ids = (0..1_000)
            .map(|_| second.mint().expect("capture ids available"))
            .collect::<HashSet<_>>();
        assert!(
            first_ids.is_disjoint(&second_ids),
            "island capture id strides overlapped"
        );
        assert_eq!(
            second.mint().expect("capture ids available"),
            GraphCaptureId(2000),
            "an island that outgrows its stride keeps minting unique ids"
        );
    }

    #[test]
    fn capture_id_exhaustion_is_an_error_rather_than_a_wrap() {
        let allocator = GraphCaptureIdAllocator {
            next: Cell::new(i32::MAX - 1),
            thread_bound: PhantomData,
        };
        assert_eq!(allocator.mint().unwrap(), GraphCaptureId(i32::MAX - 1));
        assert!(
            allocator.mint().is_err(),
            "exhaustion must be reported, not wrapped into the uncaptured sentinel"
        );
    }
}
