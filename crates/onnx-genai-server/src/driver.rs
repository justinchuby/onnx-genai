use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use anyhow::Context;
use onnx_genai::{
    Engine, GenerateOptions, GenerateRequest, GenerateResult, GenerateToken, SessionId, TokenId,
};
use onnx_genai_engine::{
    BatchingCapability, ContinuousBatchAdmission, ContinuousBatchEvent, ContinuousBatchHandle,
    ContinuousBatchManager, EmbeddingOptions, EncodedAudio, EngineGovernorError,
    EngineMemoryAccounting, FimConfig, GovernorSnapshot, KvNotApplicable, KvTelemetry,
    MemoryStrategyPlan, OrtEngineWorkerFactory, PipelineGenerateRequest, ResourceLimit,
    SchedulerAdmissionError,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot};

use crate::image_generation::{ImageExecutionRequest, ProducedImage};
use crate::lease::{ModelKey, ModelSessionPlacement, SessionLeaseGuard};
use crate::metrics::GenerationMetrics;
use crate::multimodal::MultimodalInput;
use crate::worker::{
    SessionCloseAccounting, SessionPlacement, SessionPlacementReservation, WorkerHandle, WorkerId,
    WorkerPool, WorkerStartError, WorkerStatusSnapshot, WorkerTurnGuard, WorkerUnavailable,
};

const DRIVER_OUTPUT_BUFFER: usize = 16;
const MICROBATCH_MIN_WAIT: Duration = Duration::from_millis(2);
const MICROBATCH_MAX_WAIT: Duration = Duration::from_millis(12);
const MICROBATCH_SETTLE_WAIT: Duration = Duration::from_millis(1);
const MICROBATCH_POLL_WAIT: Duration = Duration::from_micros(250);

#[derive(Clone)]
pub(crate) struct EngineDriver {
    /// The worker threads that own the engine. Exactly one today; every command
    /// names the worker it must reach rather than assuming there is only one.
    pub(crate) workers: WorkerPool,
    /// Which model this engine was loaded as, bound once by
    /// [`ModelHandle::new`](crate::registry::ModelHandle::new).
    ///
    /// The routing layer's lease and session maps span every loaded model, and
    /// a placement is only unique within one engine, so every key they build
    /// has to name the model too. Reading that name off the driver — rather
    /// than off whatever id the caller was carrying — is what makes the key a
    /// caller cannot get wrong.
    pub(crate) model: ModelKey,
    pub(crate) generation_capacity: Arc<Semaphore>,
    pub(crate) generation_capacity_size: u32,
    /// Lock-free mirror of the KV page pool, readable during a generation.
    pub(crate) kv_telemetry: Arc<KvTelemetry>,
    /// Latest engine-ledger snapshot, readable without a driver-thread round trip.
    pub(crate) resource_snapshot: Arc<Mutex<Option<GovernorSnapshot>>>,
    pub(crate) memory_strategy_plan: Arc<MemoryStrategyPlan>,
    /// Live model-wide device/host/disk ledgers shared by every worker.
    pub(crate) memory_accounting: Option<EngineMemoryAccounting>,
    /// Honest, decode-path-sourced batching report for this engine, resolved at
    /// startup. Surfaced over `/v1/resources` and `/v1/debug/kv` so an operator
    /// sees `batch_supported=false` / effective max batch = 1 directly instead of
    /// inferring it from a debug-level "using per-request engine path" log line.
    pub(crate) batching: Arc<BatchingReport>,
    /// What the loaded package's serialized workflow says, read once at
    /// startup. Facts about the document, not about which executor runs it.
    pub(crate) workflow_facts: WorkflowFacts,
    /// One lock-free KV mirror per worker. The legacy aggregate fields continue
    /// to report worker 0; debug/status expose this full list.
    pub(crate) worker_kv_telemetry: Arc<Vec<Arc<KvTelemetry>>>,
}

/// The serialized workflow, as an operator-facing summary.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WorkflowFacts {
    pub(crate) components: usize,
    pub(crate) graph_components: usize,
    pub(crate) declares_generation_loop: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct WorkerRuntimeStatus {
    pub(crate) worker: WorkerStatusSnapshot,
    pub(crate) kv: onnx_genai_engine::KvTelemetrySnapshot,
}

/// Server-facing summary of an engine's batching capability, combining the
/// engine's structural [`BatchingCapability`] with the operator's requested
/// `--max-batch` and the width that actually took effect.
#[derive(Debug, Clone)]
pub(crate) struct BatchingReport {
    /// Whether the decode path can advance more than one sequence per step.
    pub(crate) supported: bool,
    /// The `--max-batch` width the operator asked for (or the default).
    pub(crate) requested_max_batch: usize,
    /// The width that actually takes effect after clamping to what the decode
    /// path can honor.
    pub(crate) effective_max_batch: usize,
    /// Operator-facing reason string naming the backend / decode path.
    pub(crate) reason: String,
}

impl BatchingReport {
    fn from_capability(capability: &BatchingCapability, requested: usize) -> Self {
        Self {
            supported: capability.supports_batching(),
            requested_max_batch: requested,
            effective_max_batch: capability.effective_max_batch(requested),
            reason: capability.reason().to_string(),
        }
    }

    /// A placeholder report for registry test doubles that drive no engine.
    #[cfg(test)]
    pub(crate) fn single_sequence_stub() -> Self {
        Self {
            supported: false,
            requested_max_batch: 1,
            effective_max_batch: 1,
            reason: "test stub handle: no engine, single-sequence".to_string(),
        }
    }
}

pub(crate) enum DriverCommand {
    CreateSession {
        reservation: SessionPlacementReservation,
        acknowledgement: std::sync::mpsc::Receiver<()>,
        response: tokio::sync::oneshot::Sender<anyhow::Result<SessionId>>,
    },
    CloseSession {
        lease: SessionLeaseGuard,
        accounting: SessionCloseAccounting,
        response: tokio::sync::oneshot::Sender<anyhow::Result<()>>,
    },
    SessionTokenCount {
        session_id: SessionId,
        response: tokio::sync::oneshot::Sender<anyhow::Result<usize>>,
    },
    /// Tokens this session will put in front of the next turn's prompt.
    SessionPrefillCarry {
        session_id: SessionId,
        response:
            tokio::sync::oneshot::Sender<anyhow::Result<onnx_genai_engine::SessionPrefillCarry>>,
    },
    /// Generate, from whatever the request carried.
    ///
    /// One command, because there is one runtime beneath it. A session id is
    /// the conversation this request continues; `input` is the application
    /// tensors it arrived with (an image, an audio window). Either may be
    /// absent, and neither says anything about what kind of package is loaded —
    /// which is the point: a route no longer has to know.
    Generate {
        session_id: Option<SessionId>,
        /// The exclusive turn lease this generation holds, for a session-bound
        /// turn. It travels with the command rather than being released by the
        /// submitting task, so every way this command can end — accepted and
        /// run, never accepted because the channel closed, dropped because the
        /// worker stopped — releases the lease by dropping the guard.
        lease: Option<SessionLeaseGuard>,
        request: Box<GenerateRequest>,
        input: Option<MultimodalInput>,
        admission: oneshot::Sender<Result<(), DriverFailure>>,
        events: mpsc::Sender<DriverEvent>,
        permit: WorkerPermit,
    },
    SynthesizeSpeech {
        request: Box<GenerateRequest>,
        audio_output: String,
        reply: oneshot::Sender<anyhow::Result<EncodedAudio>>,
        permit: WorkerPermit,
    },
    GenerateImage {
        request: Box<ImageExecutionRequest>,
        reply: oneshot::Sender<anyhow::Result<ProducedImage>>,
        permit: WorkerPermit,
        track_metrics: bool,
    },
    GenerateFim {
        prefix: String,
        suffix: String,
        fim_config: FimConfig,
        options: Box<GenerateOptions>,
        admission: oneshot::Sender<Result<(), DriverFailure>>,
        events: mpsc::Sender<DriverEvent>,
        permit: WorkerPermit,
    },
    Embed {
        input_ids: Vec<TokenId>,
        options: EmbeddingOptions,
        reply: tokio::sync::oneshot::Sender<anyhow::Result<Vec<f32>>>,
        permit: WorkerPermit,
    },
    #[cfg(test)]
    ResourceSnapshot(tokio::sync::oneshot::Sender<anyhow::Result<GovernorSnapshot>>),
    SetVramLimit {
        limit: ResourceLimit,
        reply: tokio::sync::oneshot::Sender<
            anyhow::Result<Result<GovernorSnapshot, EngineGovernorError>>,
        >,
    },
    #[cfg(test)]
    BarrierGenerate {
        session_id: SessionId,
        barrier: Arc<std::sync::Barrier>,
        request: Box<GenerateRequest>,
        response: std::sync::mpsc::SyncSender<anyhow::Result<GenerateResult>>,
    },
    #[cfg(test)]
    Block {
        entered: std::sync::mpsc::SyncSender<()>,
        release: std::sync::mpsc::Receiver<()>,
    },
    #[cfg(test)]
    Panic,
}

#[derive(Debug)]
pub(crate) enum DriverEvent {
    Token(GenerateToken),
    Finished(GenerateResult),
    Error(DriverFailure),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DriverFailureKind {
    Internal,
    MemoryOverload,
    WorkerUnavailable,
    /// The caller asked the loaded package for something it cannot serve as
    /// asked. Carried as the engine's own type so a status code never depends
    /// on the wording of a message.
    PackageCapability(onnx_genai_engine::PackageCapabilityError),
}

#[derive(Debug, Clone)]
pub(crate) struct DriverFailure {
    pub(crate) message: String,
    pub(crate) kind: DriverFailureKind,
}

pub(crate) struct DriverGeneration {
    pub(crate) admission: oneshot::Receiver<Result<(), DriverFailure>>,
    pub(crate) events: mpsc::Receiver<DriverEvent>,
}

struct PendingSessionCreation {
    worker: WorkerId,
    acknowledgement: std::sync::mpsc::SyncSender<()>,
    response: oneshot::Receiver<anyhow::Result<SessionId>>,
}

impl std::fmt::Display for DriverFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl DriverFailure {
    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: DriverFailureKind::Internal,
        }
    }

    pub(crate) fn from_engine_error(error: &anyhow::Error) -> Self {
        let memory_overload = error.chain().any(|source| {
            matches!(
                source.downcast_ref::<SchedulerAdmissionError>(),
                Some(SchedulerAdmissionError::ByteBudget { .. })
            ) || matches!(
                source.downcast_ref::<onnx_runtime_memory_governor::MemoryError>(),
                Some(
                    onnx_runtime_memory_governor::MemoryError::TierExhausted { .. }
                        | onnx_runtime_memory_governor::MemoryError::CapacityUnavailable { .. },
                )
            )
        });
        let capability = onnx_genai_engine::package_capability_error(error);
        Self {
            // Anyhow's Display shows only the outermost context, which for a
            // decode failure is the generic "forward pass failed" wrapper. The
            // alternate form keeps the whole chain, and this message is the
            // only thing the client ever sees.
            message: match &capability {
                // A capability refusal already says exactly what happened; the
                // chain would add the interpreter frames it happened in, which
                // are not the caller's business.
                Some(capability) => capability.to_string(),
                None => format!("{error:#}"),
            },
            kind: match (capability, memory_overload) {
                (Some(capability), _) => DriverFailureKind::PackageCapability(capability),
                (None, true) => DriverFailureKind::MemoryOverload,
                (None, false) => DriverFailureKind::Internal,
            },
        }
    }

    fn from_worker_unavailable(error: WorkerUnavailable) -> Self {
        Self {
            message: error.to_string(),
            kind: DriverFailureKind::WorkerUnavailable,
        }
    }
}

#[derive(Debug)]
pub(crate) enum GenerateSubmitError {
    Overloaded,
    DriverStopped,
    Failed(DriverFailure),
}

fn generate_submit_worker_error(error: WorkerUnavailable) -> GenerateSubmitError {
    match error {
        WorkerUnavailable::Failed(_) => {
            GenerateSubmitError::Failed(DriverFailure::from_worker_unavailable(error))
        }
        WorkerUnavailable::Unknown { .. } | WorkerUnavailable::Stopped(_) => {
            GenerateSubmitError::DriverStopped
        }
    }
}

struct DriverRoute {
    admission: Option<oneshot::Sender<Result<(), DriverFailure>>>,
    events: mpsc::Sender<DriverEvent>,
    _permit: WorkerPermit,
    /// Held for as long as the row is: a route that is retired, rejected, or
    /// abandoned mid-stream drops it, and the session is leasable again.
    _lease: Option<SessionLeaseGuard>,
    metrics: GenerationMetrics,
}

struct PendingGeneration {
    request: GenerateRequest,
    lease: Option<SessionLeaseGuard>,
    admission: oneshot::Sender<Result<(), DriverFailure>>,
    events: mpsc::Sender<DriverEvent>,
    permit: WorkerPermit,
}

pub(crate) struct WorkerPermit {
    _admission: OwnedSemaphorePermit,
    _turn: WorkerTurnGuard,
}

impl WorkerPermit {
    fn new(admission: OwnedSemaphorePermit, turn: WorkerTurnGuard) -> Self {
        Self {
            _admission: admission,
            _turn: turn,
        }
    }

    #[cfg(test)]
    pub(crate) fn untracked(admission: OwnedSemaphorePermit) -> Self {
        let (commands, _rx) = mpsc::channel(1);
        let pool = WorkerPool::single(WorkerHandle::detached(WorkerId::PRIMARY, commands));
        Self::new(
            admission,
            pool.reserve_turn(WorkerId::PRIMARY)
                .expect("detached test worker is reachable"),
        )
    }
}

#[derive(Clone, Copy)]
struct MicrobatchAdmission<'a> {
    max_queue_depth: usize,
    generation_capacity: &'a Semaphore,
    worker_count: usize,
}

/// State constructed, used, and destroyed entirely inside one worker thread.
///
/// `DropProbe` is `()` in production. Tests use the final field to observe that
/// the preceding `Engine` has already been dropped on the worker thread.
struct EngineWorkerState<DropProbe> {
    engine: Engine,
    commands: Option<mpsc::Receiver<DriverCommand>>,
    continuous_batch_supported: bool,
    max_batch: usize,
    max_queue_depth: usize,
    generation_capacity: Arc<Semaphore>,
    resource_snapshot: Arc<Mutex<Option<GovernorSnapshot>>>,
    worker_count: usize,
    _not_send: std::marker::PhantomData<std::rc::Rc<()>>,
    _drop_probe: DropProbe,
}

struct EngineWorkerReady {
    batching: Arc<BatchingReport>,
    kv_telemetry: Arc<KvTelemetry>,
    resource_snapshot: Arc<Mutex<Option<GovernorSnapshot>>>,
    memory_strategy_plan: Arc<MemoryStrategyPlan>,
    memory_accounting: Option<EngineMemoryAccounting>,
    workflow_facts: WorkflowFacts,
}

fn prepare_engine_worker<DropProbe>(
    mut engine: Engine,
    rx: mpsc::Receiver<DriverCommand>,
    max_batch: usize,
    max_queue_depth: usize,
    generation_capacity: Arc<Semaphore>,
    drop_probe: DropProbe,
    worker_count: usize,
) -> (EngineWorkerState<DropProbe>, EngineWorkerReady) {
    let batching = Arc::new(BatchingReport::from_capability(
        &engine.batching_capability(),
        max_batch,
    ));
    let effective_max_batch = batching.effective_max_batch;
    let continuous_batch_supported = engine.continuous_batch_manager(effective_max_batch).is_ok();
    let memory_accounting = Some(engine.governor().memory_accounting());
    let kv_telemetry = Arc::new(KvTelemetry::default());
    if engine.attach_kv_telemetry(Arc::clone(&kv_telemetry)) {
        kv_telemetry.set_applicable();
    } else {
        kv_telemetry.set_not_applicable(KvNotApplicable::CacheCannotPage);
    }
    let memory_strategy_plan = Arc::new(engine.memory_strategy_plan().clone());
    let workflow_facts = WorkflowFacts {
        components: engine.workflow_component_count(),
        graph_components: engine.workflow_graph_component_count(),
        declares_generation_loop: engine.workflow_declares_generation_loop(),
    };
    let resource_snapshot = Arc::new(Mutex::new(Some(engine.resource_snapshot())));
    let ready = EngineWorkerReady {
        batching,
        kv_telemetry,
        resource_snapshot: Arc::clone(&resource_snapshot),
        memory_strategy_plan,
        memory_accounting,
        workflow_facts,
    };
    (
        EngineWorkerState {
            engine,
            commands: Some(rx),
            continuous_batch_supported,
            max_batch: effective_max_batch,
            max_queue_depth,
            generation_capacity,
            resource_snapshot,
            worker_count,
            _not_send: std::marker::PhantomData,
            _drop_probe: drop_probe,
        },
        ready,
    )
}

fn worker_start_error(error: WorkerStartError<anyhow::Error>) -> anyhow::Error {
    match error {
        WorkerStartError::ThreadSpawn(error) => {
            anyhow::Error::new(error).context("failed to spawn engine worker thread")
        }
        WorkerStartError::Initialization(error) => {
            error.context("engine worker initialization failed")
        }
        WorkerStartError::InitializationPanicked(message) => {
            anyhow::anyhow!("engine worker initialization panicked: {message}")
        }
        WorkerStartError::InitializationChannelClosed => {
            anyhow::anyhow!("engine worker exited before reporting initialization")
        }
    }
}

impl EngineDriver {
    /// Start one worker and construct its engine inside that OS thread.
    ///
    /// The construction closure may return a small, `Send`-safe payload derived
    /// from the engine for the surrounding model handle. The engine itself
    /// never crosses the thread boundary.
    pub(crate) fn start<Ready, Build>(
        build_engine: Build,
        max_batch: usize,
        max_queue_depth: usize,
    ) -> anyhow::Result<(Self, Ready)>
    where
        Ready: Send + 'static,
        Build: FnOnce() -> anyhow::Result<(Engine, Ready)> + Send + 'static,
    {
        Self::start_inner(
            move || {
                let (engine, ready) = build_engine()?;
                Ok((engine, ready, None))
            },
            1,
            max_batch,
            max_queue_depth,
            (),
            Arc::new(|_| Ok(())),
        )
    }

    /// Start an opt-in pool whose first worker loads the shared immutable ORT
    /// resources and whose remaining workers construct only worker-local state
    /// from that factory.
    pub(crate) fn start_ort_workers<Ready, Build>(
        build_engine: Build,
        worker_count: usize,
        max_batch: usize,
        max_queue_depth: usize,
    ) -> anyhow::Result<(Self, Ready)>
    where
        Ready: Send + 'static,
        Build: FnOnce() -> anyhow::Result<(Engine, Ready, Arc<OrtEngineWorkerFactory>)>
            + Send
            + 'static,
    {
        Self::start_inner(
            move || {
                let (engine, ready, factory) = build_engine()?;
                Ok((engine, ready, Some(factory)))
            },
            worker_count,
            max_batch,
            max_queue_depth,
            (),
            Arc::new(|_| Ok(())),
        )
    }

    fn start_inner<Ready, Build, DropProbe>(
        build_engine: Build,
        worker_count: usize,
        max_batch: usize,
        max_queue_depth: usize,
        drop_probe: DropProbe,
        before_additional_worker: Arc<dyn Fn(WorkerId) -> anyhow::Result<()> + Send + Sync>,
    ) -> anyhow::Result<(Self, Ready)>
    where
        Ready: Send + 'static,
        Build: FnOnce() -> anyhow::Result<(Engine, Ready, Option<Arc<OrtEngineWorkerFactory>>)>
            + Send
            + 'static,
        DropProbe: Send + 'static,
    {
        anyhow::ensure!(worker_count > 0, "engine worker count must be nonzero");
        let (commands, rx) = mpsc::channel(max_queue_depth);
        let generation_capacity = Arc::new(Semaphore::new(max_queue_depth));
        let driver_capacity = generation_capacity.clone();
        let worker_id = WorkerId::PRIMARY;
        let (worker, (ready_payload, factory, worker_ready)) = WorkerHandle::spawn(
            worker_id,
            format!("onnx-genai-batch-driver-{worker_id}"),
            commands,
            move || -> anyhow::Result<_> {
                let (engine, ready_payload, factory) = build_engine()
                    .with_context(|| format!("failed to construct engine on worker {worker_id}"))?;
                let (state, worker_ready) = prepare_engine_worker(
                    engine,
                    rx,
                    max_batch,
                    max_queue_depth,
                    driver_capacity,
                    drop_probe,
                    worker_count,
                );
                Ok((state, (ready_payload, factory, worker_ready)))
            },
            run_engine_driver,
        )
        .map_err(worker_start_error)
        .with_context(|| format!("failed to start engine worker {worker_id}"))?;
        let mut worker_handles = vec![worker];
        let mut worker_readies = vec![worker_ready];
        if worker_count > 1 {
            let factory = factory.ok_or_else(|| {
                anyhow::anyhow!(
                    "ORT worker factory was not returned for requested worker count {worker_count}"
                )
            })?;
            for index in 1..worker_count {
                let worker_id = WorkerId::new(index);
                let (commands, rx) = mpsc::channel(max_queue_depth);
                let factory = Arc::clone(&factory);
                let driver_capacity = Arc::clone(&generation_capacity);
                let before_additional_worker = Arc::clone(&before_additional_worker);
                let started = WorkerHandle::spawn(
                    worker_id,
                    format!("onnx-genai-batch-driver-{worker_id}"),
                    commands,
                    move || -> anyhow::Result<_> {
                        before_additional_worker(worker_id)?;
                        let engine = factory.build_worker().with_context(|| {
                            format!("failed to construct ORT engine on worker {worker_id}")
                        })?;
                        Ok(prepare_engine_worker(
                            engine,
                            rx,
                            max_batch,
                            max_queue_depth,
                            driver_capacity,
                            (),
                            worker_count,
                        ))
                    },
                    run_engine_driver,
                )
                .map_err(worker_start_error)
                .with_context(|| format!("failed to start engine worker {worker_id}"));
                match started {
                    Ok((worker, ready)) => {
                        worker_handles.push(worker);
                        worker_readies.push(ready);
                    }
                    Err(error) => {
                        WorkerPool::new(worker_handles).shutdown();
                        return Err(error.context(format!(
                            "ORT worker-pool startup was rolled back after {index} of \
                             {worker_count} worker(s) became ready"
                        )));
                    }
                }
            }
        }
        let first_ready = worker_readies
            .first()
            .expect("at least one worker reported ready");
        for (index, ready) in worker_readies.iter().enumerate().skip(1) {
            if ready.batching.supported != first_ready.batching.supported
                || ready.batching.effective_max_batch != first_ready.batching.effective_max_batch
                || ready.workflow_facts.components != first_ready.workflow_facts.components
                || ready.workflow_facts.graph_components
                    != first_ready.workflow_facts.graph_components
            {
                WorkerPool::new(worker_handles).shutdown();
                anyhow::bail!(
                    "ORT worker {index} resolved capabilities that differ from worker 0; \
                     all workers were shut down and the model was not loaded"
                );
            }
        }
        tracing::info!(
            batch_supported = first_ready.batching.supported,
            requested_max_batch = first_ready.batching.requested_max_batch,
            effective_max_batch = first_ready.batching.effective_max_batch,
            reason = %first_ready.batching.reason,
            "resolved batching capability",
        );
        let worker_kv_telemetry = Arc::new(
            worker_readies
                .iter()
                .map(|ready| Arc::clone(&ready.kv_telemetry))
                .collect::<Vec<_>>(),
        );
        let first_ready = worker_readies
            .into_iter()
            .next()
            .expect("at least one worker reported ready");
        let workers = WorkerPool::new(worker_handles);
        tracing::info!(workers = workers.len(), "engine worker pool started",);
        Ok((
            Self {
                // Rebound by `ModelHandle::new`, which is the first thing that
                // knows the id this engine was loaded as.
                model: ModelKey::new(""),
                workers,
                generation_capacity,
                generation_capacity_size: u32::try_from(max_queue_depth).unwrap_or(u32::MAX),
                kv_telemetry: Arc::clone(&first_ready.kv_telemetry),
                resource_snapshot: first_ready.resource_snapshot,
                memory_strategy_plan: first_ready.memory_strategy_plan,
                memory_accounting: first_ready.memory_accounting,
                batching: first_ready.batching,
                workflow_facts: first_ready.workflow_facts,
                worker_kv_telemetry,
            },
            ready_payload,
        ))
    }

    #[cfg(test)]
    fn start_with_drop_probe<Ready, Build, DropProbe>(
        build_engine: Build,
        max_batch: usize,
        max_queue_depth: usize,
        drop_probe: DropProbe,
    ) -> anyhow::Result<(Self, Ready)>
    where
        Ready: Send + 'static,
        Build: FnOnce() -> anyhow::Result<(Engine, Ready)> + Send + 'static,
        DropProbe: Send + 'static,
    {
        Self::start_inner(
            move || {
                let (engine, ready) = build_engine()?;
                Ok((engine, ready, None))
            },
            1,
            max_batch,
            max_queue_depth,
            drop_probe,
            Arc::new(|_| Ok(())),
        )
    }

    /// Stop every worker and wait for the engine to be destroyed.
    ///
    /// Idempotent. **Blocking**: callers on an async task must go through
    /// `spawn_blocking`, because this waits for the worker to finish whatever
    /// it is running. When it returns, the engine's ORT sessions, KV pages, and
    /// device allocations have been released on the thread that created them,
    /// and every later command fails with the usual "engine driver stopped".
    pub(crate) fn shutdown(&self) {
        self.workers.shutdown();
    }

    /// The sender for a session-bound command, addressed to the worker that
    /// owns the session's state.
    fn session_commands(
        &self,
        session: SessionPlacement,
    ) -> Result<mpsc::Sender<DriverCommand>, WorkerUnavailable> {
        self.workers.sender_for(session.worker)
    }

    /// The sender for a command bound to no session.
    fn primary_commands(&self) -> Result<mpsc::Sender<DriverCommand>, WorkerUnavailable> {
        self.workers.primary_sender()
    }

    fn reserve_stateless_command(
        &self,
    ) -> Result<(mpsc::Sender<DriverCommand>, WorkerPermit), GenerateSubmitError> {
        let (worker, turn) = self
            .workers
            .reserve_stateless_turn()
            .map_err(generate_submit_worker_error)?;
        let admission = self
            .generation_capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| GenerateSubmitError::Overloaded)?;
        let commands = self
            .workers
            .sender_for(worker)
            .map_err(generate_submit_worker_error)?;
        Ok((commands, WorkerPermit::new(admission, turn)))
    }

    pub(crate) fn worker_statuses(&self) -> Vec<WorkerRuntimeStatus> {
        self.workers
            .statuses()
            .into_iter()
            .enumerate()
            .map(|(index, worker)| WorkerRuntimeStatus {
                worker,
                kv: self.worker_kv_telemetry[index].snapshot(),
            })
            .collect()
    }

    /// What the loaded package's serialized workflow declares.
    pub(crate) fn workflow_facts(&self) -> WorkflowFacts {
        self.workflow_facts
    }

    /// The KV page pool mirror.
    ///
    /// Deliberately not a command round-trip. `/metrics` and `/v1/resources`
    /// must answer while the driver thread is inside an inline generation, and
    /// a command could not be serviced until that generation finished -- which
    /// is precisely when the pool is worth reading.
    pub(crate) fn kv_telemetry(&self) -> &Arc<KvTelemetry> {
        &self.kv_telemetry
    }

    /// The resolved, decode-path-sourced batching report for this engine.
    pub(crate) fn batching(&self) -> &BatchingReport {
        &self.batching
    }

    /// Bind this engine to the id it was loaded as.
    ///
    /// Called once, by [`ModelHandle::new`](crate::registry::ModelHandle::new),
    /// which is the only constructor of a routable model. Everything that keys
    /// a session or a lease reads the model from here, so the handle's id and
    /// the driver's are the same string by construction rather than by
    /// convention.
    pub(crate) fn bind_model(&mut self, id: &str) {
        self.model = ModelKey::new(id);
    }

    /// The model whose sessions this engine owns.
    pub(crate) fn model_key(&self) -> &ModelKey {
        &self.model
    }

    /// A conversation in *this* engine, as the globally unique key the routing
    /// layer's maps use.
    pub(crate) fn binding(&self, placement: SessionPlacement) -> ModelSessionPlacement {
        ModelSessionPlacement::new(self.model.clone(), placement)
    }

    /// Open an engine session and report where it lives.
    ///
    /// The placement, not the bare id, is what a caller stores: an engine
    /// session id means nothing away from the worker that issued it.
    pub(crate) async fn create_session(&self) -> anyhow::Result<SessionPlacement> {
        let pending = self.enqueue_session_creation().await?;
        let engine_session_id = pending
            .response
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))??;
        pending
            .acknowledgement
            .send(())
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?;
        Ok(SessionPlacement::new(pending.worker, engine_session_id))
    }

    async fn enqueue_session_creation(&self) -> anyhow::Result<PendingSessionCreation> {
        let reservation = self
            .workers
            .reserve_session_placement()
            .map_err(anyhow::Error::new)?;
        let worker = reservation.worker();
        let (response, rx) = tokio::sync::oneshot::channel();
        let (acknowledgement, acknowledgement_rx) = std::sync::mpsc::sync_channel(1);
        self.workers
            .sender_for(worker)
            .map_err(anyhow::Error::new)?
            .send(DriverCommand::CreateSession {
                reservation,
                acknowledgement: acknowledgement_rx,
                response,
            })
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?;
        Ok(PendingSessionCreation {
            worker,
            acknowledgement,
            response: rx,
        })
    }

    /// Close a session, under the lease that proves no turn is in flight on it.
    ///
    /// Taking the guard by value rather than a placement is the point: closing
    /// a conversation is a mutation, and a close that raced a live turn would
    /// destroy the state that turn is mid-way through writing. The lease is
    /// released when this returns, and never before the engine has answered.
    pub(crate) async fn close_session(&self, lease: SessionLeaseGuard) -> anyhow::Result<()> {
        self.enqueue_session_close(lease)
            .await?
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?
    }

    async fn enqueue_session_close(
        &self,
        lease: SessionLeaseGuard,
    ) -> anyhow::Result<oneshot::Receiver<anyhow::Result<()>>> {
        anyhow::ensure!(
            lease.model() == &self.model,
            "session lease for model '{}' cannot be closed on model '{}'",
            lease.model(),
            self.model,
        );
        let session = lease.placement();
        let accounting = self
            .workers
            .worker(session.worker)?
            .session_close_accounting();
        let (response, rx) = tokio::sync::oneshot::channel();
        self.session_commands(session)
            .map_err(anyhow::Error::new)?
            .send(DriverCommand::CloseSession {
                lease,
                accounting,
                response,
            })
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?;
        Ok(rx)
    }

    /// Tokens attended ahead of the prompt and the subset re-prefilled.
    pub(crate) async fn session_prefill_carry(
        &self,
        session: SessionPlacement,
    ) -> anyhow::Result<onnx_genai_engine::SessionPrefillCarry> {
        let (response, rx) = tokio::sync::oneshot::channel();
        self.session_commands(session)
            .map_err(anyhow::Error::new)?
            .send(DriverCommand::SessionPrefillCarry {
                session_id: session.engine_session_id,
                response,
            })
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?
    }

    pub(crate) async fn session_token_count(
        &self,
        session: SessionPlacement,
    ) -> anyhow::Result<usize> {
        let (response, rx) = tokio::sync::oneshot::channel();
        self.session_commands(session)
            .map_err(anyhow::Error::new)?
            .send(DriverCommand::SessionTokenCount {
                session_id: session.engine_session_id,
                response,
            })
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?
    }

    /// Submit one generation, whatever the package shape.
    ///
    /// The runtime resolves how to execute it, so a route no longer chooses
    /// between a text command and a pipeline command: it passes the session it
    /// has (ignored by a workflow package, which owns no engine sessions) and
    /// the multimodal attachments it has (ignored by a decoder package, which
    /// takes its prompt from the request).
    ///
    /// A session-bound turn arrives holding its exclusive lease, taken by the
    /// routing layer before this call. Nothing here can take it: by the time a
    /// command is being built the caller has already occupied a queue slot, and
    /// a refusal decided here would be a refusal decided too late.
    pub(crate) async fn submit_generation(
        &self,
        session: Option<SessionLeaseGuard>,
        request: GenerateRequest,
        input: Option<MultimodalInput>,
    ) -> Result<DriverGeneration, GenerateSubmitError> {
        self.generate(session, request, input).await
    }

    pub(crate) async fn generate(
        &self,
        session: Option<SessionLeaseGuard>,
        request: GenerateRequest,
        input: Option<MultimodalInput>,
    ) -> Result<DriverGeneration, GenerateSubmitError> {
        // A session-bound generation goes to the worker holding its KV state;
        // a stateless one goes to the worker that serves unbound work.
        debug_assert!(
            session
                .as_ref()
                .is_none_or(|lease| lease.model() == &self.model),
            "a turn's lease names the engine it runs on",
        );
        let placement = session.as_ref().map(SessionLeaseGuard::placement);
        let (worker, turn) = match placement {
            Some(placement) => (
                placement.worker,
                self.workers
                    .reserve_turn(placement.worker)
                    .map_err(generate_submit_worker_error)?,
            ),
            None => self
                .workers
                .reserve_stateless_turn()
                .map_err(generate_submit_worker_error)?,
        };
        let admission_permit = self
            .generation_capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| GenerateSubmitError::Overloaded)?;
        let permit = WorkerPermit::new(admission_permit, turn);
        let commands = self
            .workers
            .sender_for(worker)
            .map_err(generate_submit_worker_error)?;
        let (events, rx) = mpsc::channel(DRIVER_OUTPUT_BUFFER);
        let (admission, admission_rx) = oneshot::channel();
        crate::metrics::generation_queued();
        // The guard moves into the command, and the command moves into `send`.
        // A send that fails hands both back on this task, where the guard drops
        // and the session is immediately leasable again — no release call to
        // forget on the one exit path that never reaches the worker.
        if commands
            .send(DriverCommand::Generate {
                session_id: placement.map(|placement| placement.engine_session_id),
                lease: session,
                request: Box::new(request),
                input,
                admission,
                events,
                permit,
            })
            .await
            .is_err()
        {
            crate::metrics::generation_queue_cancelled();
            return Err(GenerateSubmitError::DriverStopped);
        }
        Ok(DriverGeneration {
            admission: admission_rx,
            events: rx,
        })
    }

    /// Run a small generation while blocking the calling thread. Used only by
    /// startup and the administrative warmup path, never request generation.
    pub(crate) fn warmup(&self, request: GenerateRequest) -> anyhow::Result<()> {
        let (worker, turn) = self
            .workers
            .reserve_stateless_turn()
            .map_err(anyhow::Error::new)?;
        let admission_permit = self
            .generation_capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| anyhow::anyhow!("generation capacity exceeded"))?;
        let permit = WorkerPermit::new(admission_permit, turn);
        let (events, mut receiver) = mpsc::channel(DRIVER_OUTPUT_BUFFER);
        let (admission, _admission_rx) = oneshot::channel();
        let command = DriverCommand::Generate {
            session_id: None,
            lease: None,
            request: Box::new(request),
            input: None,
            admission,
            events,
            permit,
        };
        self.workers
            .sender_for(worker)
            .map_err(anyhow::Error::new)?
            .blocking_send(command)
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?;
        while let Some(event) = receiver.blocking_recv() {
            match event {
                DriverEvent::Token(_) => {}
                DriverEvent::Finished(_) => return Ok(()),
                DriverEvent::Error(error) => anyhow::bail!(error.message),
            }
        }
        anyhow::bail!("generation stream ended before result")
    }

    pub(crate) async fn synthesize_speech(
        &self,
        request: GenerateRequest,
        audio_output: String,
    ) -> Result<EncodedAudio, GenerateSubmitError> {
        let (commands, permit) = self.reserve_stateless_command()?;
        let (reply, response) = oneshot::channel();
        crate::metrics::generation_queued();
        if commands
            .send(DriverCommand::SynthesizeSpeech {
                request: Box::new(request),
                audio_output,
                reply,
                permit,
            })
            .await
            .is_err()
        {
            crate::metrics::generation_queue_cancelled();
            return Err(GenerateSubmitError::DriverStopped);
        }
        response
            .await
            .map_err(|_| GenerateSubmitError::DriverStopped)?
            .map_err(|error| GenerateSubmitError::Failed(DriverFailure::from_engine_error(&error)))
    }

    pub(crate) async fn generate_image(
        &self,
        request: ImageExecutionRequest,
    ) -> Result<anyhow::Result<ProducedImage>, GenerateSubmitError> {
        let (commands, permit) = self.reserve_stateless_command()?;
        let (reply, receiver) = oneshot::channel();
        crate::metrics::generation_queued();
        if commands
            .send(DriverCommand::GenerateImage {
                request: Box::new(request),
                reply,
                permit,
                track_metrics: true,
            })
            .await
            .is_err()
        {
            crate::metrics::generation_queue_cancelled();
            return Err(GenerateSubmitError::DriverStopped);
        }
        receiver
            .await
            .map_err(|_| GenerateSubmitError::DriverStopped)
    }

    pub(crate) fn warmup_image(&self, request: ImageExecutionRequest) -> anyhow::Result<()> {
        let (commands, permit) = self
            .reserve_stateless_command()
            .map_err(|error| match error {
                GenerateSubmitError::Overloaded => anyhow::anyhow!("generation capacity exceeded"),
                GenerateSubmitError::DriverStopped => anyhow::anyhow!("engine driver stopped"),
                GenerateSubmitError::Failed(failure) => anyhow::anyhow!(failure.message),
            })?;
        let (reply, receiver) = oneshot::channel();
        commands
            .blocking_send(DriverCommand::GenerateImage {
                request: Box::new(request),
                reply,
                permit,
                track_metrics: false,
            })
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?;
        receiver
            .blocking_recv()
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))??;
        Ok(())
    }

    pub(crate) async fn generate_fim(
        &self,
        prefix: String,
        suffix: String,
        fim_config: FimConfig,
        options: GenerateOptions,
    ) -> Result<DriverGeneration, GenerateSubmitError> {
        let (commands, permit) = self.reserve_stateless_command()?;
        let (events, rx) = mpsc::channel(DRIVER_OUTPUT_BUFFER);
        let (admission, admission_rx) = oneshot::channel();
        crate::metrics::generation_queued();
        if commands
            .send(DriverCommand::GenerateFim {
                prefix,
                suffix,
                fim_config,
                options: Box::new(options),
                admission,
                events,
                permit,
            })
            .await
            .is_err()
        {
            crate::metrics::generation_queue_cancelled();
            return Err(GenerateSubmitError::DriverStopped);
        }
        Ok(DriverGeneration {
            admission: admission_rx,
            events: rx,
        })
    }

    pub(crate) async fn embed(
        &self,
        input_ids: Vec<TokenId>,
        options: EmbeddingOptions,
    ) -> anyhow::Result<Vec<f32>> {
        self.enqueue_embedding(input_ids, options)
            .await?
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?
    }

    async fn enqueue_embedding(
        &self,
        input_ids: Vec<TokenId>,
        options: EmbeddingOptions,
    ) -> anyhow::Result<oneshot::Receiver<anyhow::Result<Vec<f32>>>> {
        let (worker, turn) = self
            .workers
            .reserve_stateless_turn()
            .map_err(anyhow::Error::new)?;
        let admission = Arc::clone(&self.generation_capacity)
            .acquire_owned()
            .await
            .map_err(|_| anyhow::anyhow!("engine admission semaphore closed"))?;
        let permit = WorkerPermit::new(admission, turn);
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.workers
            .sender_for(worker)
            .map_err(anyhow::Error::new)?
            .send(DriverCommand::Embed {
                input_ids,
                options,
                reply,
                permit,
            })
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?;
        Ok(rx)
    }

    pub(crate) async fn resource_snapshot(&self) -> anyhow::Result<GovernorSnapshot> {
        let mut snapshot = self
            .resource_snapshot
            .lock()
            .expect("resource snapshot mirror lock poisoned")
            .clone()
            .ok_or_else(|| anyhow::anyhow!("resource governor is not available for this model"))?;
        if let Some(accounting) = &self.memory_accounting {
            accounting.refresh_snapshot(&mut snapshot);
        }
        Ok(snapshot)
    }

    pub(crate) fn memory_strategy_plan(&self) -> Arc<MemoryStrategyPlan> {
        Arc::clone(&self.memory_strategy_plan)
    }

    pub(crate) async fn set_vram_limit(
        &self,
        limit: ResourceLimit,
    ) -> anyhow::Result<Result<GovernorSnapshot, EngineGovernorError>> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.primary_commands()
            .map_err(anyhow::Error::new)?
            .send(DriverCommand::SetVramLimit { limit, reply })
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?;
        let mut result = rx
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))??;
        if let Ok(snapshot) = &mut result
            && let Some(accounting) = &self.memory_accounting
        {
            accounting.refresh_snapshot(snapshot);
        }
        Ok(result)
    }
}

fn run_engine_driver<DropProbe>(mut worker: EngineWorkerState<DropProbe>) {
    let commands = worker
        .commands
        .take()
        .expect("engine worker command receiver is present");
    // Which driver runs is a capability question the engine answers: a package
    // whose decode path can advance several rows per forward pass gets the
    // continuous-batch driver, and everything else — including a package whose
    // components own separate caches — gets the per-request one. Neither driver
    // asks what kind of package it holds.
    if worker.continuous_batch_supported {
        tracing::info!(
            max_batch = worker.max_batch,
            "continuous batch driver enabled"
        );
        run_static_engine_driver(
            &mut worker.engine,
            commands,
            worker.max_batch,
            worker.max_queue_depth,
            &worker.generation_capacity,
            &worker.resource_snapshot,
            worker.worker_count,
        );
    } else {
        tracing::info!(
            batch_supported = false,
            effective_max_batch = 1,
            "continuous batch driver disabled; using per-request engine path (single-sequence decode)"
        );
        run_fallback_engine_driver(&mut worker.engine, commands, &worker.resource_snapshot);
    }
}

fn run_fallback_engine_driver(
    engine: &mut Engine,
    mut rx: mpsc::Receiver<DriverCommand>,
    resource_snapshot: &Mutex<Option<GovernorSnapshot>>,
) {
    while let Some(command) = rx.blocking_recv() {
        handle_driver_command(engine, command);
        refresh_resource_snapshot(resource_snapshot, engine);
    }
}

fn run_static_engine_driver(
    engine: &mut Engine,
    mut rx: mpsc::Receiver<DriverCommand>,
    max_batch: usize,
    max_queue_depth: usize,
    generation_capacity: &Semaphore,
    resource_snapshot: &Mutex<Option<GovernorSnapshot>>,
    worker_count: usize,
) {
    // The current ContinuousBatchManager API accepts GenerateRequest only.
    // X-Session-Id requests keep using the driver's per-request engine path so
    // persistent engine KV/session semantics are preserved until the manager
    // grows a SessionId-aware submit API.
    let mut deferred = std::collections::VecDeque::new();
    loop {
        while let Ok(command) = rx.try_recv() {
            match command {
                command @ DriverCommand::Generate {
                    session_id: None,
                    input: None,
                    ..
                } => deferred.push_back(command),
                command => deferred.push_back(command),
            }
        }

        let Some(first) = deferred.pop_front().or_else(|| rx.blocking_recv()) else {
            break;
        };

        match first {
            DriverCommand::Generate {
                session_id: None,
                lease,
                request,
                input: None,
                admission,
                events,
                permit,
            } => {
                run_static_batch_until_idle(
                    engine,
                    &mut rx,
                    &mut deferred,
                    max_batch,
                    MicrobatchAdmission {
                        max_queue_depth,
                        generation_capacity,
                        worker_count,
                    },
                    resource_snapshot,
                    PendingGeneration {
                        request: *request,
                        lease,
                        admission,
                        events,
                        permit,
                    },
                );
                refresh_resource_snapshot(resource_snapshot, engine);
            }
            command => {
                handle_driver_command(engine, command);
                refresh_resource_snapshot(resource_snapshot, engine);
            }
        }
    }
}

fn refresh_resource_snapshot(resource_snapshot: &Mutex<Option<GovernorSnapshot>>, engine: &Engine) {
    *resource_snapshot
        .lock()
        .expect("resource snapshot mirror lock poisoned") = Some(engine.resource_snapshot());
}

fn run_static_batch_until_idle(
    engine: &mut Engine,
    rx: &mut mpsc::Receiver<DriverCommand>,
    deferred: &mut VecDeque<DriverCommand>,
    max_batch: usize,
    admission: MicrobatchAdmission<'_>,
    resource_snapshot: &Mutex<Option<GovernorSnapshot>>,
    first: PendingGeneration,
) {
    let mut initial = vec![first];

    let started = Instant::now();
    let hard_deadline = started + MICROBATCH_MAX_WAIT;
    let mut soft_deadline = started + MICROBATCH_MIN_WAIT;
    let mut saw_pending_sibling = false;
    loop {
        let mut accepted = 0_usize;
        while initial.len() < max_batch {
            match deferred.pop_front() {
                Some(DriverCommand::Generate {
                    session_id: None,
                    lease,
                    request,
                    input: None,
                    admission,
                    events,
                    permit,
                }) => {
                    initial.push(PendingGeneration {
                        request: *request,
                        lease,
                        admission,
                        events,
                        permit,
                    });
                    accepted += 1;
                }
                Some(command) => {
                    deferred.push_front(command);
                    break;
                }
                None => break,
            }
        }
        while initial.len() < max_batch {
            match rx.try_recv() {
                Ok(DriverCommand::Generate {
                    session_id: None,
                    lease,
                    request,
                    input: None,
                    admission,
                    events,
                    permit,
                }) => {
                    initial.push(PendingGeneration {
                        request: *request,
                        lease,
                        admission,
                        events,
                        permit,
                    });
                    accepted += 1;
                }
                Ok(command) => deferred.push_back(command),
                Err(_) => break,
            }
        }
        if accepted > 0 {
            let settle_deadline = Instant::now() + MICROBATCH_SETTLE_WAIT;
            soft_deadline = soft_deadline.max(settle_deadline.min(hard_deadline));
        }
        if initial.len() >= max_batch {
            break;
        }

        let expected_this_batch = if admission.worker_count == 1 {
            // Preserve the pre-sharding batching heuristic byte-for-byte for
            // the default configuration.
            let in_flight = admission
                .max_queue_depth
                .saturating_sub(admission.generation_capacity.available_permits());
            let deferred_permit_holders = deferred_permit_holder_count(deferred);
            in_flight
                .saturating_sub(deferred_permit_holders)
                .min(max_batch)
        } else {
            // The global admission semaphore also counts turns assigned to
            // other workers. They must never influence this worker's batch.
            initial.len()
        };
        // Re-evaluated every iteration rather than latched: a sibling that was
        // expected and then drained (or was never batchable to begin with)
        // must stop holding this request at the hard deadline. Latching meant
        // one transient over-count pinned a lone request to the slow path for
        // the rest of the window.
        saw_pending_sibling = expected_this_batch > initial.len();
        if saw_pending_sibling {
            soft_deadline = hard_deadline;
        }

        let now = Instant::now();
        if now >= hard_deadline {
            break;
        }
        let step = admission_step(
            initial.len(),
            expected_this_batch,
            accepted,
            deferred.is_empty(),
            now >= soft_deadline,
        );
        if step == AdmissionStep::Admit {
            break;
        }
        let deadline = if step == AdmissionStep::WaitForSibling {
            hard_deadline
        } else {
            soft_deadline
        };
        let sleep_for = deadline
            .saturating_duration_since(now)
            .min(MICROBATCH_POLL_WAIT);
        if sleep_for.is_zero() {
            thread::yield_now();
        } else {
            thread::sleep(sleep_for);
        }
    }

    if initial.len() == 1 && !saw_pending_sibling {
        let pending = initial
            .pop()
            .expect("the first generation request was queued");
        run_generation(engine, None, None, pending);
        return;
    }

    let formed_batch = if saw_pending_sibling || initial.len() > 1 {
        max_batch
    } else {
        initial.len().max(1).min(max_batch)
    };
    let mut manager = match engine.continuous_batch_manager(formed_batch) {
        Ok(manager) => manager,
        Err(err) => {
            crate::metrics::generation_queue_cancelled();
            for pending in initial {
                let failure =
                    DriverFailure::internal(format!("continuous batch setup failed: {err}"));
                let _ = pending.admission.send(Err(failure.clone()));
                let _ = pending.events.try_send(DriverEvent::Error(failure));
            }
            return;
        }
    };
    let mut routes: HashMap<usize, DriverRoute> = HashMap::new();
    let mut abandoned = HashMap::new();
    let mut reported_occupancy = onnx_genai_engine::BatchOccupancy::default();
    for pending in initial {
        submit_to_continuous_manager(&mut manager, &mut routes, &mut abandoned, pending);
    }

    loop {
        while routes.len() + abandoned.len() < manager.max_batch() {
            match deferred.pop_front() {
                Some(DriverCommand::Generate {
                    session_id: None,
                    lease,
                    request,
                    input: None,
                    admission,
                    events,
                    permit,
                }) => {
                    submit_to_continuous_manager(
                        &mut manager,
                        &mut routes,
                        &mut abandoned,
                        PendingGeneration {
                            request: *request,
                            lease,
                            admission,
                            events,
                            permit,
                        },
                    );
                }
                Some(command) => {
                    deferred.push_front(command);
                    break;
                }
                None => break,
            }
        }
        while let Ok(command) = rx.try_recv() {
            match command {
                DriverCommand::Generate {
                    session_id: None,
                    lease,
                    request,
                    input: None,
                    admission,
                    events,
                    permit,
                } => {
                    if routes.len() + abandoned.len() < manager.max_batch() {
                        submit_to_continuous_manager(
                            &mut manager,
                            &mut routes,
                            &mut abandoned,
                            PendingGeneration {
                                request: *request,
                                lease,
                                admission,
                                events,
                                permit,
                            },
                        );
                    } else {
                        deferred.push_back(DriverCommand::Generate {
                            session_id: None,
                            lease,
                            request,
                            input: None,
                            admission,
                            events,
                            permit,
                        });
                    }
                }
                // MUST stay above the catch-all: anything this returns `None`
                // for has been answered, and anything it hands back is parked
                // until the batch drains.
                command => {
                    if let Some(deferred_command) =
                        handle_or_defer_during_batch(resource_snapshot, command)
                    {
                        deferred.push_back(deferred_command);
                    }
                }
            }
        }

        cancel_abandoned_pending_admissions(&mut manager, &mut routes);
        manager.admit_pending();
        route_continuous_admissions(manager.poll_admissions(), &mut routes);
        if let Err(err) = manager.step() {
            let mut failure = DriverFailure::from_engine_error(&err);
            failure.message = format!("continuous batch generation failed: {err:#}");
            for (_, mut route) in routes.drain() {
                if let Some(sender) = route.admission.take() {
                    let _ = sender.send(Err(failure.clone()));
                }
                let _ = route.events.try_send(DriverEvent::Error(DriverFailure {
                    message: failure.message.clone(),
                    kind: failure.kind.clone(),
                }));
            }
            reported_occupancy = publish_batch_occupancy(manager.occupancy(), reported_occupancy);
            break;
        }
        route_continuous_admissions(manager.poll_admissions(), &mut routes);
        route_continuous_events(manager.poll(), &mut routes, &mut abandoned);
        reported_occupancy = publish_batch_occupancy(manager.occupancy(), reported_occupancy);
        if manager.is_idle() {
            // Ending the batch's declared pass is what reconciles its emit with
            // the tokens its executor committed, row by row. Doing it here
            // rather than dropping the manager is what keeps that check on the
            // path this server actually takes.
            //
            // The failure is *logged*, not routed: by the time the batch is
            // idle every route has been removed by the terminal event that
            // retired it, so there is nobody left to tell. What the log says is
            // that the tokens this batch already delivered are not the tokens
            // its declared workflow published — which is an operator-facing
            // fact about the build, not a per-request error.
            if let Err(err) = manager.drain() {
                tracing::error!(
                    error = %format!("{err:#}"),
                    steps = reported_occupancy.steps,
                    "the continuous batch's declared emit disagrees with the tokens its                      executor committed"
                );
            }
            break;
        }
    }
    tracing::info!(
        steps = reported_occupancy.steps,
        rows_advanced = reported_occupancy.rows_advanced,
        peak_rows = reported_occupancy.max_rows_in_step,
        max_batch = reported_occupancy.max_batch,
        mean_rows_per_step = reported_occupancy.mean_rows_per_step(),
        "continuous batch group drained"
    );
}

/// Publish the forwards issued since the last report to the process metrics.
///
/// The manager reports cumulative occupancy for its own lifetime, but the
/// registry accumulates across batch groups, so only the delta is added.
fn publish_batch_occupancy(
    current: onnx_genai_engine::BatchOccupancy,
    reported: onnx_genai_engine::BatchOccupancy,
) -> onnx_genai_engine::BatchOccupancy {
    crate::metrics::batch_forwards_observed(
        current.steps.saturating_sub(reported.steps),
        current.rows_advanced.saturating_sub(reported.rows_advanced),
        current.max_rows_in_step,
        current.max_batch,
    );
    current
}

/// What the admission loop should do this iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdmissionStep {
    /// Stop collecting and run what has been gathered.
    Admit,
    /// A sibling is outstanding; keep waiting until the hard deadline.
    WaitForSibling,
    /// Nothing is outstanding but the settle window has not closed.
    WaitToSettle,
}

/// Decide whether to keep collecting arrivals or admit what we have.
///
/// Extracted from the loop so the decision is testable without a live engine:
/// the loop needs a real driver thread, but this is a pure function of the
/// counts, and it is where both scheduling regressions lived.
fn admission_step(
    collected: usize,
    expected_this_batch: usize,
    accepted_this_iteration: usize,
    deferred_is_empty: bool,
    settled: bool,
) -> AdmissionStep {
    // Re-derived every iteration rather than latched. A sibling that was
    // expected and then drained (or was never batchable) must stop holding this
    // request, otherwise one transient over-count pins a lone request to the
    // slow path for the rest of the window.
    if expected_this_batch > collected {
        return AdmissionStep::WaitForSibling;
    }
    // Nothing arriving, nothing outstanding: genuinely solo, so admit now
    // rather than sleeping out the settle window. Waiting here added that delay
    // to the first token of every solo generation on a batching-capable engine,
    // which is the opposite of what a solo fast path is for.
    if accepted_this_iteration == 0 && deferred_is_empty {
        return AdmissionStep::Admit;
    }
    if settled {
        AdmissionStep::Admit
    } else {
        AdmissionStep::WaitToSettle
    }
}

/// How many deferred commands are holding a `generation_capacity` permit.
///
/// This is subtracted from the in-flight count to estimate how many *other*
/// requests are still arriving, so it must match the set of commands that
/// actually take a permit — every one of `generate`, `generate_pipeline`,
/// `generate_fim`, `render_images`, `synthesize_speech`, and `embed` does.
///
/// Counting only the text-generation commands understated the deferred total,
/// which inflated `expected_this_batch` and latched `saw_pending_sibling` for a
/// sibling that did not exist. A lone request then waited out the hard deadline
/// and ran through the continuous-batch path by itself — the slow path this
/// admission logic exists to avoid.
fn deferred_permit_holder_count(deferred: &VecDeque<DriverCommand>) -> usize {
    deferred
        .iter()
        .filter(|command| {
            matches!(
                command,
                DriverCommand::Generate { .. }
                    | DriverCommand::GenerateImage { .. }
                    | DriverCommand::GenerateFim { .. }
                    | DriverCommand::SynthesizeSpeech { .. }
                    | DriverCommand::Embed { .. }
            )
        })
        .count()
}

/// Decides what the static-batch loop does with a command that is not a new
/// generation: `None` means it was answered here and now, `Some` means it is
/// parked until the batch drains.
///
/// Read-only observability must be answered here. `/v1/resources` exists to
/// report on a running batch, so deferring its snapshot makes it readable only
/// when there is no batch left to report on -- the request appears to hang for
/// the entire duration of every in-flight generation.
///
/// Commands that *reconfigure* engine state are deferred until the batch drains by design.
/// Only read-only observability is answered immediately here.
pub(crate) fn handle_or_defer_during_batch(
    _resource_snapshot: &Mutex<Option<GovernorSnapshot>>,
    command: DriverCommand,
) -> Option<DriverCommand> {
    match command {
        // The variant itself is `#[cfg(test)]`: in production `/v1/resources`
        // reads the same mirror directly rather than routing through the driver.
        #[cfg(test)]
        DriverCommand::ResourceSnapshot(reply) => {
            // Served from the mirror, not from `&Engine`: during a batch the
            // engine is mutably borrowed by the ContinuousBatchManager, and the
            // point of answering here rather than deferring is that
            // `/v1/resources` must not appear to hang until every in-flight
            // generation completes. The mirror is the same value that endpoint
            // serves elsewhere, refreshed by `refresh_resource_snapshot`.
            let snapshot = _resource_snapshot
                .lock()
                .ok()
                .and_then(|guard| guard.clone());
            if let Some(snapshot) = snapshot {
                let _ = reply.send(Ok(snapshot));
                return None;
            }
            // No mirror yet: defer rather than fabricate one.
            Some(DriverCommand::ResourceSnapshot(reply))
        }
        other => Some(other),
    }
}

fn submit_to_continuous_manager(
    manager: &mut ContinuousBatchManager<'_>,
    routes: &mut HashMap<usize, DriverRoute>,
    abandoned: &mut HashMap<usize, DriverRoute>,
    pending: PendingGeneration,
) {
    let PendingGeneration {
        request,
        lease,
        admission,
        events,
        permit,
    } = pending;
    match manager.submit(request) {
        Ok(handle) => {
            routes.insert(
                handle.id,
                DriverRoute {
                    admission: Some(admission),
                    events,
                    _permit: permit,
                    // The row now owns the lease: whichever way it is retired —
                    // finished, rejected, or abandoned by a disconnected client —
                    // the route is dropped and the session becomes leasable again.
                    _lease: lease,
                    metrics: GenerationMetrics::start(),
                },
            );
            route_continuous_admissions(manager.poll_admissions(), routes);
            route_continuous_events(manager.poll(), routes, abandoned);
        }
        Err(err) => {
            crate::metrics::generation_queue_cancelled();
            let failure = DriverFailure::from_engine_error(&err);
            let _ = admission.send(Err(failure.clone()));
            let _ = events.try_send(DriverEvent::Error(failure));
        }
    }
}

fn route_continuous_admissions(
    admissions: Vec<ContinuousBatchAdmission>,
    routes: &mut HashMap<usize, DriverRoute>,
) {
    for admission in admissions {
        match admission {
            ContinuousBatchAdmission::Assigned { handle } => {
                if let Some(route) = routes.get_mut(&handle.id)
                    && let Some(sender) = route.admission.take()
                {
                    let _ = sender.send(Ok(()));
                }
            }
            ContinuousBatchAdmission::Rejected { handle, error } => {
                let Some(mut route) = routes.remove(&handle.id) else {
                    continue;
                };
                let failure = DriverFailure::from_engine_error(&error);
                if let Some(sender) = route.admission.take() {
                    let _ = sender.send(Err(failure.clone()));
                }
                let _ = route.events.try_send(DriverEvent::Error(failure));
            }
        }
    }
}

fn cancel_abandoned_pending_admissions(
    manager: &mut ContinuousBatchManager<'_>,
    routes: &mut HashMap<usize, DriverRoute>,
) {
    let cancelled = routes
        .iter()
        .filter_map(|(&id, route)| {
            route
                .admission
                .as_ref()
                .is_some_and(oneshot::Sender::is_closed)
                .then_some(id)
                .filter(|_| route.events.is_closed())
        })
        .collect::<Vec<_>>();
    for id in cancelled {
        if manager.cancel_pending(ContinuousBatchHandle { id }) {
            routes.remove(&id);
            crate::metrics::generation_queue_cancelled();
        }
    }
}

fn route_continuous_events(
    events: Vec<ContinuousBatchEvent>,
    routes: &mut HashMap<usize, DriverRoute>,
    abandoned: &mut HashMap<usize, DriverRoute>,
) {
    for event in events {
        match event {
            ContinuousBatchEvent::Token { handle, token } => {
                // A *disconnected* consumer loses its route immediately: the driver
                // keeps stepping every other row while the manager retires the
                // abandoned one. A consumer that is merely behind gets backpressure
                // instead — dropping its route would discard the tokens it has not
                // read yet along with the terminal event, turning a live request
                // into a truncated stream once the bounded channel filled up.
                let delivery_failed = if let Some(route) = routes.get_mut(&handle.id) {
                    route.metrics.token();
                    deliver_driver_event(&route.events, DriverEvent::Token(token), DELIVERY_GRACE)
                        .is_err()
                } else {
                    false
                };
                if delivery_failed && let Some(route) = routes.remove(&handle.id) {
                    abandoned.insert(handle.id, route);
                }
            }
            ContinuousBatchEvent::Finished { handle, result } => {
                if let Some(mut route) = routes.remove(&handle.id) {
                    route
                        .metrics
                        .result(result.token_ids.len(), result.prefix_cache_hit_len);
                    // The terminal event must reach the consumer even when the
                    // channel is momentarily full; losing it closes the stream and
                    // the caller reports "generation stream ended before result".
                    let _ = deliver_driver_event(
                        &route.events,
                        DriverEvent::Finished(result),
                        DELIVERY_GRACE,
                    );
                } else if let Some(mut route) = abandoned.remove(&handle.id) {
                    route
                        .metrics
                        .result(result.token_ids.len(), result.prefix_cache_hit_len);
                }
            }
            ContinuousBatchEvent::Aborted {
                handle,
                outcome,
                error,
            } => {
                let failure = DriverFailure::internal(format!(
                    "generation turn aborted to its committed baseline ({outcome:?}): {error}"
                ));
                if let Some(mut route) = routes.remove(&handle.id) {
                    if let Some(sender) = route.admission.take() {
                        let _ = sender.send(Err(failure.clone()));
                    }
                    let _ = deliver_driver_event(
                        &route.events,
                        DriverEvent::Error(failure),
                        DELIVERY_GRACE,
                    );
                }
                abandoned.remove(&handle.id);
            }
        }
    }
}

fn handle_driver_command(engine: &mut Engine, command: DriverCommand) {
    match command {
        DriverCommand::CreateSession {
            reservation,
            acknowledgement,
            response,
        } => {
            let session_id = match engine.create_session() {
                Ok(session_id) => session_id,
                Err(error) => {
                    let _ = response.send(Err(error));
                    return;
                }
            };
            let committed = match reservation.commit() {
                Ok(committed) => committed,
                Err(error) => {
                    let _ = engine.close_session(session_id);
                    let _ = response.send(Err(anyhow::Error::new(error)));
                    return;
                }
            };
            if response.send(Ok(session_id)).is_err() || acknowledgement.recv().is_err() {
                let _ = engine.close_session(session_id);
                return;
            }
            committed.persist();
        }
        DriverCommand::CloseSession {
            lease,
            accounting,
            response,
        } => {
            let held_lease = lease;
            let result = engine.close_session(held_lease.placement().engine_session_id);
            if result.is_ok() {
                accounting.session_closed();
            }
            drop(held_lease);
            let _ = response.send(result);
        }
        DriverCommand::SessionTokenCount {
            session_id,
            response,
        } => {
            let _ = response.send(engine.session_token_count(session_id));
        }
        DriverCommand::SessionPrefillCarry {
            session_id,
            response,
        } => {
            let _ = response.send(engine.session_prefill_carry(session_id));
        }
        DriverCommand::Generate {
            session_id,
            lease,
            request,
            input,
            admission,
            events,
            permit,
        } => run_generation(
            engine,
            session_id,
            input,
            PendingGeneration {
                request: *request,
                lease,
                admission,
                events,
                permit,
            },
        ),
        DriverCommand::GenerateFim {
            prefix,
            suffix,
            fim_config,
            options,
            admission,
            events,
            permit,
        } => run_fim_generation(
            engine,
            prefix,
            suffix,
            fim_config,
            *options,
            (admission, events, permit),
        ),
        // Speech and image are workflow outputs: the package either declares
        // one with the right role or it does not, and the runtime says which.
        // Refusing them up front by package kind would have been a guess about
        // the same question the workflow already answers.
        DriverCommand::SynthesizeSpeech {
            request,
            audio_output,
            reply,
            permit,
        } => {
            let _permit = permit;
            let result = engine
                .run_pipeline_outputs(PipelineGenerateRequest::new(*request))
                .and_then(|outputs| engine.encode_audio_output(&outputs, &audio_output));
            let _ = reply.send(result);
        }
        DriverCommand::GenerateImage {
            request,
            reply,
            permit: _permit,
            track_metrics,
        } => {
            let _metrics = track_metrics.then(GenerationMetrics::start);
            let result = (|| {
                let outputs = engine.run_pipeline_outputs(request.into_pipeline()?)?;
                let image = engine
                    .structured_output_for_role(
                        &outputs,
                        onnx_genai_engine::pipeline::WorkflowOutputRole::Image,
                    )
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "workflow completed without emitting its declared image output"
                        )
                    })?;
                Ok(ProducedImage {
                    values: image.to_vec_f32_lossy()?,
                    shape: image.shape().to_vec(),
                })
            })();
            let _ = reply.send(result);
        }
        DriverCommand::Embed {
            input_ids,
            options,
            reply,
            permit: _permit,
        } => {
            let _ = reply.send(engine.embed_with_options(&input_ids, options));
        }
        #[cfg(test)]
        DriverCommand::ResourceSnapshot(reply) => {
            let _ = reply.send(Ok(engine.resource_snapshot()));
        }
        DriverCommand::SetVramLimit { limit, reply } => {
            let result = engine
                .set_vram_limit(limit)
                .map(|_| engine.resource_snapshot());
            let _ = reply.send(Ok(result));
        }
        #[cfg(test)]
        DriverCommand::BarrierGenerate {
            session_id,
            barrier,
            request,
            response,
        } => {
            barrier.wait();
            let _ = response.send(engine.generate_in_session(session_id, *request));
        }
        #[cfg(test)]
        DriverCommand::Block { entered, release } => {
            let _ = entered.send(());
            let _ = release.recv();
        }
        #[cfg(test)]
        DriverCommand::Panic => panic!("injected worker failure"),
    }
}

/// How long a consumer may hold up the driver before it counts as stalled.
///
/// The driver serves every generation — batched rows and solo requests alike —
/// from one thread, so a consumer that stops reading holds up everyone behind
/// it. Two bounds pin this value. It must be far longer than the microseconds
/// an awake receiver needs to drain a `DRIVER_OUTPUT_BUFFER`-sized burst, so a
/// consumer that is merely *behind* is never mistaken for a dead one. And it
/// must stay well inside the time a caller is willing to wait for an unrelated
/// request, because abandoning a consumer that has genuinely stopped reading is
/// what preserves the guarantee that no single route can wedge the driver.
const DELIVERY_GRACE: Duration = Duration::from_secs(1);

/// How often to retry a full channel while waiting out the grace period.
const DELIVERY_RETRY_INTERVAL: Duration = Duration::from_millis(1);

/// Deliver a driver event, waiting briefly for capacity instead of dropping the
/// route the moment its channel is full.
///
/// `try_send` alone cannot tell "the consumer went away" from "the consumer is
/// briefly behind", and the bounded channel holds only `DRIVER_OUTPUT_BUFFER`
/// events. Treating a full buffer as a disconnect aborted every generation
/// longer than the buffer — "stream receiver closed: no available capacity" —
/// while the client was still connected and reading, and then dropped the
/// terminal event into that same full buffer, so the caller was told
/// "generation stream ended before result" instead of what actually went wrong.
///
/// Driver generation runs on a dedicated OS thread, never on the async runtime,
/// so waiting here parks that one thread rather than stalling the reactor. The
/// wait is bounded by `grace` so a consumer that has genuinely stopped reading
/// still loses its route and cannot wedge other rows.
fn deliver_driver_event(
    events: &mpsc::Sender<DriverEvent>,
    event: DriverEvent,
    grace: Duration,
) -> Result<(), DriverDeliveryError> {
    let mut pending = match events.try_send(event) {
        Ok(()) => return Ok(()),
        Err(mpsc::error::TrySendError::Full(event)) => event,
        Err(mpsc::error::TrySendError::Closed(_)) => {
            return Err(DriverDeliveryError::Disconnected);
        }
    };
    let deadline = Instant::now() + grace;
    loop {
        if Instant::now() >= deadline {
            return Err(DriverDeliveryError::Stalled);
        }
        thread::sleep(DELIVERY_RETRY_INTERVAL);
        pending = match events.try_send(pending) {
            Ok(()) => return Ok(()),
            Err(mpsc::error::TrySendError::Full(event)) => event,
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return Err(DriverDeliveryError::Disconnected);
            }
        };
    }
}

/// Deliver one event to a single request's own output channel.
///
/// The solo paths own their channel outright and have nothing to report a
/// typed stall to, so they take the default grace and surface either failure as
/// an ordinary error that ends the generation.
fn deliver_event(events: &mpsc::Sender<DriverEvent>, event: DriverEvent) -> anyhow::Result<()> {
    deliver_driver_event(events, event, DELIVERY_GRACE).map_err(anyhow::Error::new)
}

#[cfg(test)]
mod image_metrics_tests {
    use std::path::PathBuf;

    use super::*;
    use crate::state::AppState;

    #[tokio::test]
    async fn failed_image_execution_restores_pending_metric() {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/comfyui_workflows/txt2img_sd15");
        let state = AppState::load(&model_dir, Some("image-metrics-error".to_string())).unwrap();
        let handle = state
            .registry
            .resolve("image-metrics-error")
            .unwrap()
            .unwrap();
        let baseline = crate::metrics::snapshot().pending_requests;

        // Omitting the required application-owned negative-prompt input makes
        // the workflow fail after the image command starts.
        let result = handle
            .engine
            .generate_image(ImageExecutionRequest {
                request: GenerateRequest {
                    prompt: onnx_genai::GeneratePrompt::TokenIds(vec![2, 3]),
                    options: GenerateOptions {
                        max_new_tokens: 1,
                        seed: Some(1),
                        ..GenerateOptions::default()
                    },
                },
                inputs: Vec::new(),
            })
            .await
            .expect("image command submitted");

        assert!(
            result.is_err(),
            "malformed image workflow request must fail"
        );
        assert_eq!(crate::metrics::snapshot().pending_requests, baseline);
    }
}

/// Why a driver event could not be handed to its consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DriverDeliveryError {
    /// The receiver was dropped.
    Disconnected,
    /// The receiver is still alive but stopped draining within the grace period.
    Stalled,
}

impl std::fmt::Display for DriverDeliveryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disconnected => formatter.write_str("stream receiver closed"),
            Self::Stalled => {
                formatter.write_str("stream consumer stalled: output buffer stayed full")
            }
        }
    }
}

impl std::error::Error for DriverDeliveryError {}

/// Run one generation, from whatever the request carried.
///
/// One function, because there is one runtime beneath it. A session id names
/// the conversation to continue and the engine resolves what continues it; the
/// bound tensors are what the request arrived with, and a request carrying them
/// is bound as a workflow request because that is the only way to deliver them.
/// Neither branch asks what kind of package is loaded.
fn run_generation(
    engine: &mut Engine,
    session_id: Option<SessionId>,
    input: Option<MultimodalInput>,
    pending: PendingGeneration,
) {
    let PendingGeneration {
        request,
        lease,
        admission,
        events,
        permit: _permit,
    } = pending;
    let mut metrics = GenerationMetrics::start();
    let bound = match input
        .map(|input| input.bind(PipelineGenerateRequest::new(request.clone())))
        .transpose()
    {
        Ok(bound) => bound,
        Err(error) => {
            let failure = DriverFailure::internal(format!("{error:#}"));
            let _ = admission.send(Err(failure.clone()));
            let _ = events.try_send(DriverEvent::Error(failure));
            return;
        }
    };
    let mut admission = Some(admission);
    let mut callback = |token: GenerateToken| -> anyhow::Result<()> {
        metrics.token();
        deliver_event(&events, DriverEvent::Token(token))
    };
    let result = {
        let mut admitted = || {
            if let Some(sender) = admission.take() {
                let _ = sender.send(Ok(()));
            }
        };
        match (bound, session_id) {
            // A request carrying tensors is still a request in a conversation.
            // Dropping the session here would lose every `scope: session` cell
            // the package declares — silently, since the generation still
            // succeeds — which is exactly the multimodal multi-turn case
            // sessions were opened up for.
            (Some(bound), session) => engine.generate_with_pipeline_callbacks(
                match session {
                    Some(session) => bound.with_session_id(session.to_string()),
                    None => bound,
                },
                Some(&mut admitted),
                Some(&mut callback),
            ),
            (None, Some(session_id)) => engine.generate_in_session_with_callbacks(
                session_id,
                request,
                Some(&mut admitted),
                Some(&mut callback),
            ),
            (None, None) => {
                engine.generate_with_callbacks(request, Some(&mut admitted), Some(&mut callback))
            }
        }
    };
    // The engine has committed whatever this turn wrote to the session, so the
    // conversation is consistent and the next turn may take the lease. Released
    // *before* the terminal event rather than after this function returns: a
    // client that reads "finished" and immediately sends the next turn would
    // otherwise race the guard's drop and be told its own completed turn is
    // still in flight. A turn admitted at this instant still runs after this
    // one — the worker that would run it is this thread, and it is here.
    drop(lease);
    match result {
        Ok(result) => {
            metrics.result(result.token_ids.len(), result.prefix_cache_hit_len);
            let _ = deliver_event(&events, DriverEvent::Finished(result));
        }
        Err(err) => {
            let failure = DriverFailure::from_engine_error(&err);
            if let Some(sender) = admission.take() {
                let _ = sender.send(Err(failure.clone()));
            }
            let _ = deliver_event(&events, DriverEvent::Error(failure));
        }
    }
}

fn run_fim_generation(
    engine: &mut Engine,
    prefix: String,
    suffix: String,
    fim_config: FimConfig,
    options: GenerateOptions,
    delivery: (
        oneshot::Sender<Result<(), DriverFailure>>,
        mpsc::Sender<DriverEvent>,
        WorkerPermit,
    ),
) {
    let (admission, events, _permit) = delivery;
    let mut metrics = GenerationMetrics::start();
    let mut admission = Some(admission);
    let result = {
        let mut admitted = || {
            if let Some(sender) = admission.take() {
                let _ = sender.send(Ok(()));
            }
        };
        engine.generate_fim_with_config_and_callbacks(
            prefix,
            suffix,
            options,
            &fim_config,
            Some(&mut admitted),
            None,
        )
    };
    match result {
        Ok(result) => {
            metrics.result(result.token_ids.len(), result.prefix_cache_hit_len);
            let _ = deliver_event(&events, DriverEvent::Finished(result));
        }
        Err(err) => {
            let failure = DriverFailure::from_engine_error(&err);
            if let Some(sender) = admission.take() {
                let _ = sender.send(Err(failure.clone()));
            }
            let _ = deliver_event(&events, DriverEvent::Error(failure));
        }
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    struct IsSend<T: ?Sized>(std::marker::PhantomData<T>);
    trait SendFallback {
        const VALUE: bool = false;
    }
    impl<T: ?Sized> SendFallback for IsSend<T> {}
    #[allow(dead_code)]
    impl<T: ?Sized + Send> IsSend<T> {
        const VALUE: bool = true;
    }

    struct EngineDropThread(std::sync::mpsc::SyncSender<thread::ThreadId>);

    impl Drop for EngineDropThread {
        fn drop(&mut self) {
            let _ = self.0.send(thread::current().id());
        }
    }

    async fn block_worker(driver: &EngineDriver) -> std::sync::mpsc::SyncSender<()> {
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        driver
            .workers
            .primary_sender()
            .unwrap()
            .send(DriverCommand::Block {
                entered: entered_tx,
                release: release_rx,
            })
            .await
            .unwrap();
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("worker entered blocking command");
        release_tx
    }

    async fn wait_for_prior_commands(driver: &EngineDriver) {
        let release = block_worker(driver).await;
        release.send(()).unwrap();
    }

    #[test]
    fn engine_worker_state_is_structurally_not_send() {
        const {
            assert!(
                !IsSend::<EngineWorkerState<()>>::VALUE,
                "engine worker state must not cross threads"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn two_ort_workers_run_distinct_sessions_concurrently_with_colliding_local_ids() {
        let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm");
        let model_directory =
            onnx_genai_ort::ModelDirectory::load(&model_dir).expect("resolve tiny fixture");
        let config = onnx_genai_engine::EngineConfig {
            decode_backend: onnx_genai_engine::EngineDecodeBackend::Ort,
            ..onnx_genai_engine::EngineConfig::default()
        };
        let build_config = config.clone();
        let (driver, ()) = EngineDriver::start_ort_workers(
            move || {
                let engine = Engine::from_dir(&model_dir, build_config.clone())?;
                let factory = Arc::new(
                    engine
                        .ort_worker_factory(model_directory, build_config)
                        .expect("CPU ORT session supports concurrent Run"),
                );
                Ok((engine, (), factory))
            },
            2,
            1,
            8,
        )
        .expect("start two ORT workers");

        let first = driver.create_session().await.expect("first session");
        let second = driver.create_session().await.expect("second session");
        assert_eq!(first.worker, WorkerId::new(0));
        assert_eq!(second.worker, WorkerId::new(1));
        assert_eq!(
            first.engine_session_id, second.engine_session_id,
            "worker-local session ids may collide; placement keeps them distinct"
        );

        let barrier = Arc::new(std::sync::Barrier::new(2));
        let mut request = GenerateRequest::new(onnx_genai::GeneratePrompt::TokenIds(vec![1]));
        request.options.max_new_tokens = 2;
        request.options.stop_on_eos = false;
        let (first_tx, first_rx) = std::sync::mpsc::sync_channel(1);
        let (second_tx, second_rx) = std::sync::mpsc::sync_channel(1);
        driver
            .workers
            .sender_for(first.worker)
            .unwrap()
            .send(DriverCommand::BarrierGenerate {
                session_id: first.engine_session_id,
                barrier: Arc::clone(&barrier),
                request: Box::new(request.clone()),
                response: first_tx,
            })
            .await
            .unwrap();
        driver
            .workers
            .sender_for(second.worker)
            .unwrap()
            .send(DriverCommand::BarrierGenerate {
                session_id: second.engine_session_id,
                barrier,
                request: Box::new(request),
                response: second_tx,
            })
            .await
            .unwrap();

        let deadline = Duration::from_secs(10);
        assert!(
            first_rx
                .recv_timeout(deadline)
                .expect("worker 0 crossed the barrier")
                .is_ok()
        );
        assert!(
            second_rx
                .recv_timeout(deadline)
                .expect("worker 1 crossed the barrier")
                .is_ok()
        );
        driver.shutdown();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn w2_resource_snapshot_tracks_worker1_host_release_without_worker0_command() {
        let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm");
        let model_directory =
            onnx_genai_ort::ModelDirectory::load(&model_dir).expect("resolve tiny fixture");
        let config = onnx_genai_engine::EngineConfig {
            decode_backend: onnx_genai_engine::EngineDecodeBackend::Ort,
            ..onnx_genai_engine::EngineConfig::default()
        };
        let build_config = config.clone();
        let (driver, ()) = EngineDriver::start_ort_workers(
            move || {
                let engine = Engine::from_dir(&model_dir, build_config.clone())?;
                let factory = Arc::new(
                    engine
                        .ort_worker_factory(model_directory, build_config)
                        .expect("CPU ORT session supports concurrent Run"),
                );
                Ok((engine, (), factory))
            },
            2,
            1,
            8,
        )
        .expect("start two ORT workers");

        let worker0_at_ready = driver
            .resource_snapshot
            .lock()
            .unwrap()
            .clone()
            .expect("worker 0 readiness snapshot");
        let both_workers = driver.resource_snapshot().await.unwrap();
        assert!(
            both_workers.host_ram.used > worker0_at_ready.host_ram.used,
            "worker 1 KV allocation must appear without a worker 0 refresh: worker0={}, live={}",
            worker0_at_ready.host_ram.used,
            both_workers.host_ram.used,
        );

        driver
            .workers
            .sender_for(WorkerId::new(1))
            .unwrap()
            .send(DriverCommand::Panic)
            .await
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while driver.worker_statuses()[1].worker.health != crate::worker::WorkerHealth::Failed {
            assert!(Instant::now() < deadline, "worker 1 did not report failure");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        let after_worker1_failure = driver.resource_snapshot().await.unwrap();
        assert_eq!(
            after_worker1_failure.host_ram.used, worker0_at_ready.host_ram.used,
            "worker 1 KV release must disappear without executing a worker 0 command",
        );

        driver.shutdown();
        let after_teardown = driver.resource_snapshot().await.unwrap();
        assert_eq!(after_teardown.host_ram.used, 0);
        assert_eq!(after_teardown.vram.used, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn one_worker_failure_loses_only_its_sessions() {
        let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm");
        let model_directory =
            onnx_genai_ort::ModelDirectory::load(&model_dir).expect("resolve tiny fixture");
        let config = onnx_genai_engine::EngineConfig {
            decode_backend: onnx_genai_engine::EngineDecodeBackend::Ort,
            ..onnx_genai_engine::EngineConfig::default()
        };
        let build_config = config.clone();
        let (driver, ()) = EngineDriver::start_ort_workers(
            move || {
                let engine = Engine::from_dir(&model_dir, build_config.clone())?;
                let factory = Arc::new(
                    engine
                        .ort_worker_factory(model_directory, build_config)
                        .expect("CPU ORT session supports concurrent Run"),
                );
                Ok((engine, (), factory))
            },
            2,
            1,
            8,
        )
        .expect("start two ORT workers");
        let lost = driver.create_session().await.expect("worker 0 session");
        let survivor = driver.create_session().await.expect("worker 1 session");

        driver
            .workers
            .sender_for(lost.worker)
            .unwrap()
            .send(DriverCommand::Panic)
            .await
            .unwrap();
        let timeout = Instant::now() + Duration::from_secs(5);
        loop {
            if driver.worker_statuses()[0].worker.health == crate::worker::WorkerHealth::Failed {
                break;
            }
            assert!(Instant::now() < timeout, "worker 0 did not report failure");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            driver.worker_statuses()[0].worker.live_sessions,
            0,
            "failed-worker placements are invalidated"
        );
        let leases = crate::lease::SessionLeases::with_shards(1);
        let lease = leases
            .acquire(driver.binding(lost), "lost-session")
            .expect("the failed worker cannot strand the routing lease");
        assert!(matches!(
            driver.generate(Some(lease), warmup_request(), None).await,
            Err(GenerateSubmitError::Failed(DriverFailure {
                kind: DriverFailureKind::WorkerUnavailable,
                ..
            }))
        ));

        let lost_error = driver
            .session_token_count(lost)
            .await
            .expect_err("the failed worker's session is lost");
        assert!(
            lost_error
                .to_string()
                .contains("sessions placed on this worker"),
            "{lost_error:#}"
        );
        assert!(
            driver.session_token_count(survivor).await.is_ok(),
            "worker 1 remains available"
        );
        let replacement = driver
            .create_session()
            .await
            .expect("healthy worker accepts");
        assert_eq!(replacement.worker, WorkerId::new(1));
        driver.shutdown();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_session_and_embedding_calls_finish_worker_owned_cleanup() {
        let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm");
        let (driver, ()) = EngineDriver::start(
            move || {
                Ok((
                    Engine::from_dir(&model_dir, onnx_genai_engine::EngineConfig::default())?,
                    (),
                ))
            },
            1,
            8,
        )
        .expect("load tiny fixture");

        // Cancellation after enqueue drops the acknowledgement and response.
        // The worker creates, closes, and releases the pending reservation.
        let create_release = block_worker(&driver).await;
        let pending_create = driver.enqueue_session_creation().await.unwrap();
        drop(pending_create);
        create_release.send(()).unwrap();
        wait_for_prior_commands(&driver).await;
        assert_eq!(driver.worker_statuses()[0].worker.live_sessions, 0);

        // Even after the response was delivered, ownership is not transferred
        // until the submitting future sends its non-awaiting acknowledgement.
        let mut delivered_create = driver.enqueue_session_creation().await.unwrap();
        (&mut delivered_create.response).await.unwrap().unwrap();
        drop(delivered_create);
        wait_for_prior_commands(&driver).await;
        assert_eq!(driver.worker_statuses()[0].worker.live_sessions, 0);

        // A close command owns both the exclusive lease and its accounting.
        // Dropping the response receiver cannot admit another turn early or
        // leave the live-session count behind.
        let session = driver.create_session().await.unwrap();
        let leases = crate::lease::SessionLeases::with_shards(1);
        let binding = driver.binding(session);
        let lease = leases.acquire(binding.clone(), "cancel-close").unwrap();
        let close_release = block_worker(&driver).await;
        let close_response = driver.enqueue_session_close(lease).await.unwrap();
        drop(close_response);
        assert!(leases.is_held(&binding));
        close_release.send(()).unwrap();
        wait_for_prior_commands(&driver).await;
        assert!(!leases.is_held(&binding));
        assert_eq!(driver.worker_statuses()[0].worker.live_sessions, 0);

        // Embedding admission and turn accounting travel with the queued work.
        // They stay charged after the caller disconnects and release only once
        // the worker has actually attempted the embedding.
        let embed_release = block_worker(&driver).await;
        let embed_response = driver
            .enqueue_embedding(vec![1], EmbeddingOptions::default())
            .await
            .unwrap();
        drop(embed_response);
        assert_eq!(driver.generation_capacity.available_permits(), 7);
        assert_eq!(driver.worker_statuses()[0].worker.active_turns, 1);
        embed_release.send(()).unwrap();
        wait_for_prior_commands(&driver).await;
        assert_eq!(driver.generation_capacity.available_permits(), 8);
        assert_eq!(driver.worker_statuses()[0].worker.active_turns, 0);

        driver.shutdown();
    }

    #[test]
    fn partial_worker_startup_rolls_back_and_joins_ready_workers() {
        let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm");
        let model_directory =
            onnx_genai_ort::ModelDirectory::load(&model_dir).expect("resolve tiny fixture");
        let config = onnx_genai_engine::EngineConfig {
            decode_backend: onnx_genai_engine::EngineDecodeBackend::Ort,
            ..onnx_genai_engine::EngineConfig::default()
        };
        let build_config = config.clone();
        let caller = thread::current().id();
        let (dropped_tx, dropped_rx) = std::sync::mpsc::sync_channel(1);

        let error = match EngineDriver::start_inner(
            move || {
                let engine = Engine::from_dir(&model_dir, build_config.clone())?;
                let factory = Arc::new(
                    engine
                        .ort_worker_factory(model_directory, build_config)
                        .expect("CPU ORT session supports concurrent Run"),
                );
                Ok((engine, (), Some(factory)))
            },
            2,
            1,
            8,
            EngineDropThread(dropped_tx),
            Arc::new(|worker| {
                anyhow::ensure!(
                    worker != WorkerId::new(1),
                    "injected worker startup failure"
                );
                Ok(())
            }),
        ) {
            Ok(_) => panic!("the injected second-worker failure must abort pool startup"),
            Err(error) => error,
        };

        assert!(
            format!("{error:#}").contains("startup was rolled back after 1 of 2"),
            "{error:#}"
        );
        assert_ne!(
            dropped_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("ready worker was shut down and joined"),
            caller,
            "the first engine must be destroyed on its owner thread"
        );
    }

    fn pending_route() -> (
        DriverRoute,
        oneshot::Receiver<Result<(), DriverFailure>>,
        mpsc::Receiver<DriverEvent>,
    ) {
        let (admission, admission_rx) = oneshot::channel();
        let (events, events_rx) = mpsc::channel(1);
        let permit = Arc::new(Semaphore::new(1)).try_acquire_owned().unwrap();
        (
            DriverRoute {
                admission: Some(admission),
                events,
                _permit: WorkerPermit::untracked(permit),
                _lease: None,
                metrics: GenerationMetrics::start(),
            },
            admission_rx,
            events_rx,
        )
    }

    #[tokio::test]
    async fn continuous_admission_waits_for_row_assignment_not_queue_insertion() {
        let (route, mut admission_rx, mut events_rx) = pending_route();
        let mut routes = HashMap::from([(7, route)]);

        assert!(matches!(
            admission_rx.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        assert!(events_rx.try_recv().is_err(), "no token is required");

        route_continuous_admissions(
            vec![ContinuousBatchAdmission::Assigned {
                handle: ContinuousBatchHandle { id: 7 },
            }],
            &mut routes,
        );

        assert!(admission_rx.await.unwrap().is_ok());
        assert!(events_rx.try_recv().is_err(), "headers precede first token");
    }

    #[tokio::test]
    async fn continuous_row_failure_rejects_before_headers_without_memory_misclassification() {
        let (route, admission_rx, mut events_rx) = pending_route();
        let mut routes = HashMap::from([(9, route)]);

        route_continuous_admissions(
            vec![ContinuousBatchAdmission::Rejected {
                handle: ContinuousBatchHandle { id: 9 },
                error: anyhow::anyhow!("row assignment failed"),
            }],
            &mut routes,
        );

        let failure = admission_rx.await.unwrap().unwrap_err();
        assert_eq!(failure.kind, DriverFailureKind::Internal);
        assert!(routes.is_empty());
        assert!(matches!(
            events_rx.recv().await,
            Some(DriverEvent::Error(DriverFailure {
                kind: DriverFailureKind::Internal,
                ..
            }))
        ));
    }

    #[test]
    fn governed_capacity_failures_are_memory_overload() {
        let memory_error: anyhow::Error = SchedulerAdmissionError::ByteBudget {
            request_id: 1,
            seq_id: 2,
            prompt_tokens: 10,
            max_tokens: 20,
            bytes_per_token: 1024,
            requested: 30_720,
            minimum_required: 11_264,
            used: 4096,
            limit: 8192,
            available: 4096,
            shortfall: 7168,
            running: 1,
            max_batch_size: 32,
        }
        .into();
        assert_eq!(
            DriverFailure::from_engine_error(&memory_error).kind,
            DriverFailureKind::MemoryOverload
        );
        let workspace_error: anyhow::Error =
            onnx_runtime_memory_governor::MemoryError::TierExhausted {
                tier: "device",
                requested: 4096,
                used: 8192,
                limit: 8192,
                available: 0,
                role: onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped: false },
            }
            .into();
        assert_eq!(
            DriverFailure::from_engine_error(&workspace_error).kind,
            DriverFailureKind::MemoryOverload
        );
        let mapped_physical_error: anyhow::Error =
            onnx_runtime_memory_governor::MemoryError::CapacityUnavailable {
                tier: "device",
                requested: 4096,
                available: 0,
                role: onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped: false },
                detail: "cuMemMap could not grow the physical handle pool".into(),
                // Shaped the way the allocator actually reports this: the
                // governor's refusal is carried whole, so classification must
                // not depend on the wording of `detail`.
                source: Some(Box::new(
                    onnx_runtime_memory_governor::MemoryError::TierExhausted {
                        tier: "device",
                        requested: 4096,
                        used: 0,
                        limit: 0,
                        available: 0,
                        role: onnx_runtime_memory_governor::MemoryRole::Workspace {
                            step_scoped: false,
                        },
                    },
                )),
            }
            .into();
        assert_eq!(
            DriverFailure::from_engine_error(&mapped_physical_error).kind,
            DriverFailureKind::MemoryOverload
        );
        let invalid_error: anyhow::Error =
            onnx_runtime_memory_governor::MemoryError::InvalidRequest {
                tier: "device",
                requested: 1,
                reason: "invalid allocation range",
            }
            .into();
        assert_eq!(
            DriverFailure::from_engine_error(&invalid_error).kind,
            DriverFailureKind::Internal
        );

        let batch_error: anyhow::Error = SchedulerAdmissionError::BatchFull {
            running: 32,
            max_batch_size: 32,
        }
        .into();
        assert_eq!(
            DriverFailure::from_engine_error(&batch_error).kind,
            DriverFailureKind::Internal
        );
    }

    #[tokio::test]
    async fn resource_snapshot_uses_mirror_even_when_command_queue_is_full() {
        let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm");
        let engine = Engine::from_dir(&model_dir, onnx_genai_engine::EngineConfig::default())
            .expect("load tiny fixture");
        let snapshot = engine.resource_snapshot();
        let (commands, _rx) = mpsc::channel(1);
        let (reply, _reply_rx) = tokio::sync::oneshot::channel();
        commands
            .try_send(DriverCommand::ResourceSnapshot(reply))
            .expect("fill command queue");
        let driver = EngineDriver {
            workflow_facts: crate::driver::WorkflowFacts {
                components: 0,
                graph_components: 0,
                declares_generation_loop: false,
            },
            workers: WorkerPool::single(WorkerHandle::detached(WorkerId::PRIMARY, commands)),
            model: ModelKey::new("stub"),
            generation_capacity: Arc::new(Semaphore::new(1)),
            generation_capacity_size: 1,
            kv_telemetry: Arc::new(KvTelemetry::default()),
            worker_kv_telemetry: Arc::new(vec![Arc::new(KvTelemetry::default())]),
            resource_snapshot: Arc::new(Mutex::new(Some(snapshot.clone()))),
            memory_strategy_plan: Arc::new(engine.memory_strategy_plan().clone()),
            memory_accounting: None,
            batching: Arc::new(BatchingReport::from_capability(
                &engine.batching_capability(),
                1,
            )),
        };

        assert_eq!(driver.resource_snapshot().await.unwrap(), snapshot);
    }

    /// The driver must hand the admission callback to the *tensor-bound* arm.
    ///
    /// `run_generation`'s success arm never touches the admission sender: only
    /// its error arm does. So on a request that succeeds, the `Some(&mut
    /// admitted)` argument at the `(Some(bound), session)` call site is the only
    /// thing that resolves this oneshot with `Ok`, and the server is holding the
    /// other end waiting to send its response headers. Pass `None` there and
    /// every successful tensor-bound request 500s at the awaiting end while the
    /// generation itself completes perfectly -- the loudest possible failure in
    /// the quietest possible place.
    ///
    /// #1995: the engine's own guards (`a_tensor_bound_request_signals_
    /// admission_exactly_once` and its refuse-direction sibling) assert that the
    /// *engine* invokes the callback it is given. Neither can see this defect,
    /// because this is the *driver* not giving it one. Measured rather than
    /// argued: replacing `Some(&mut admitted)` with `None` on that arm alone
    /// left all 296 tests across all four server targets passing, byte-identical
    /// to baseline, while nulling the whole closure fails four -- all of them on
    /// the `(None, None)` prompt arm. The gap is this arm specifically.
    ///
    /// The generation is *not* required to succeed for this to discriminate,
    /// which is what makes the test writable at all -- the engine-side test
    /// records the opposite conclusion, that an end-to-end version needs a
    /// package the interpreter can finish. It does not, because the admission
    /// signal precedes the work and the error arm's `take()` is what separates
    /// the two cases:
    ///
    /// - given the callback: it fires at admission, `take()`s the sender, and
    ///   the error arm finds `None` -- the receiver already holds `Ok(())`;
    /// - denied the callback: the sender survives to the error arm, which sends
    ///   `Err(failure)`.
    ///
    /// So the received *value* is the observable, and asserting `Ok(())` rather
    /// than "resolved" is load-bearing: a test that only checked the receiver
    /// completed would pass under the mutation, since a denied callback still
    /// resolves it -- with the wrong answer.
    #[tokio::test]
    async fn a_tensor_bound_request_is_told_it_was_admitted() {
        let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm");
        let mut engine = Engine::from_dir(&model_dir, onnx_genai_engine::EngineConfig::default())
            .expect("load tiny fixture");

        // `encoded: true` is the branch that hands the package its bytes
        // untouched, so this needs no feature extractor and no mel-shaped
        // fixture -- it is the shortest route to a genuinely bound tensor.
        let spec = crate::audio_input::AudioInputSpec {
            endpoint: "audio_bytes".to_string(),
            n_mels: 80,
            n_frames: 100,
            max_tokens: None,
            encoded: true,
        };
        let input = MultimodalInput::from_samples(&spec, &[0.0f32; 16], 16_000)
            .expect("an encoded-audio spec binds the samples it is handed");

        // Assert the precondition rather than trusting the test's name: a
        // request whose inputs went missing is prompt-only, takes the
        // `(None, None)` arm, and would then pass for a reason that has nothing
        // to do with binding tensors -- the exact vacuity the four existing
        // streaming tests already sit in.
        let probe = input
            .bind(PipelineGenerateRequest::new(GenerateRequest::new(
                onnx_genai::GeneratePrompt::TokenIds(vec![1]),
            )))
            .expect("binding an encoded tensor cannot fail");
        assert!(
            !probe.inputs.is_empty(),
            "this test is only about the branch a tensor-carrying request takes"
        );
        let input = MultimodalInput::from_samples(&spec, &[0.0f32; 16], 16_000)
            .expect("an encoded-audio spec binds the samples it is handed");

        let mut request = GenerateRequest::new(onnx_genai::GeneratePrompt::TokenIds(vec![1]));
        request.options.max_new_tokens = 1;
        request.options.stop_on_eos = false;

        let (admission, admission_rx) = oneshot::channel();
        let (events, _events_rx) = mpsc::channel(64);
        let permit = Arc::new(Semaphore::new(1)).try_acquire_owned().unwrap();

        run_generation(
            &mut engine,
            None,
            Some(input),
            PendingGeneration {
                request,
                lease: None,
                admission,
                events,
                permit: WorkerPermit::untracked(permit),
            },
        );

        let admitted = admission_rx
            .await
            .expect("the admission sender must be resolved, not dropped");
        if let Err(failure) = admitted {
            panic!(
                "a tensor-bound request that the scheduler admitted must be told so, \
                 got Err({failure:?}): this oneshot is what the server waits on before \
                 sending headers, and the bound arm's `Some(&mut admitted)` is its only \
                 resolver on success"
            );
        }
    }

    #[test]
    fn native_cold_and_prompt_lookup_abort_without_driver_token_or_finished_events() {
        fn pending(
            request: GenerateRequest,
        ) -> (
            PendingGeneration,
            oneshot::Receiver<Result<(), DriverFailure>>,
            mpsc::Receiver<DriverEvent>,
        ) {
            let (admission, admission_rx) = oneshot::channel();
            let (events, events_rx) = mpsc::channel(DRIVER_OUTPUT_BUFFER);
            let permit = Arc::new(Semaphore::new(1)).try_acquire_owned().unwrap();
            (
                PendingGeneration {
                    request,
                    lease: None,
                    admission,
                    events,
                    permit: WorkerPermit::untracked(permit),
                },
                admission_rx,
                events_rx,
            )
        }

        for (route, speculative) in [("cold native", false), ("native prompt lookup", true)] {
            let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/tiny-native-engine");
            let mut engine = Engine::from_dir(
                &model_dir,
                onnx_genai_engine::EngineConfig {
                    decode_backend: onnx_genai_engine::EngineDecodeBackend::Native,
                    native_device: Some(onnx_genai_engine::NativeDecodeDevice::Cpu),
                    ..onnx_genai_engine::EngineConfig::default()
                },
            )
            .expect("load native test fixture");
            let mut request = GenerateRequest::new(onnx_genai::GeneratePrompt::TokenIds(vec![0]));
            request.options.max_new_tokens = 3;
            request.options.temperature = 0.0;
            request.options.greedy = true;
            request.options.stop_on_eos = false;
            if speculative {
                request.options.speculative_mode =
                    Some(onnx_genai_engine::SpeculativeMode::PromptLookup {
                        ngram: 1,
                        max_tokens: 2,
                    });
            }

            // The test hook fails only when the second selected token attempts
            // to enter the commit-only journal, so token one was already staged.
            engine.set_native_output_staging_limit_for_test(Some(1));
            let (pending_generation, mut admission_rx, mut events_rx) = pending(request.clone());
            run_generation(&mut engine, None, None, pending_generation);
            assert!(
                admission_rx
                    .try_recv()
                    .expect("admitted native generation must resolve admission")
                    .is_ok(),
                "{route} must report scheduler admission before executing"
            );
            match events_rx
                .try_recv()
                .expect("aborted generation must report its error")
            {
                DriverEvent::Error(failure) => assert!(
                    failure.message.contains(
                        "output staging exhausted its 1-token admission after buffering 1"
                    ),
                    "{route} must fail after staging token one: {}",
                    failure.message
                ),
                DriverEvent::Token(token) => {
                    panic!(
                        "{route} leaked DriverEvent::Token({}) from an aborted turn",
                        token.token_id
                    )
                }
                DriverEvent::Finished(_) => {
                    panic!("{route} leaked a success completion from an aborted turn")
                }
            }
            assert!(
                events_rx.try_recv().is_err(),
                "{route} must publish no token or success completion after its error"
            );

            engine.set_native_output_staging_limit_for_test(None);
            let (pending_generation, mut admission_rx, mut events_rx) = pending(request);
            run_generation(&mut engine, None, None, pending_generation);
            assert!(
                admission_rx
                    .try_recv()
                    .expect("retry admission must resolve")
                    .is_ok(),
                "{route} retry must be admitted"
            );
            let mut tokens = Vec::new();
            let mut finished = None;
            while let Ok(event) = events_rx.try_recv() {
                match event {
                    DriverEvent::Token(token) => tokens.push(token.token_id),
                    DriverEvent::Finished(result) => finished = Some(result),
                    DriverEvent::Error(failure) => {
                        panic!("{route} deterministic retry failed: {}", failure.message)
                    }
                }
            }
            let finished = finished.expect("{route} retry must publish success completion");
            assert_eq!(
                tokens, finished.token_ids,
                "{route} retry token event order"
            );
            assert_eq!(tokens, vec![1, 1, 1], "{route} retry must be deterministic");
        }
    }

    /// Shutting a driver down destroys the engine on the thread that owns it,
    /// before the call returns — and says so the same way twice.
    #[tokio::test]
    async fn shutdown_joins_the_engine_worker_and_is_idempotent() {
        let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm");
        let caller_thread = thread::current().id();
        let (constructed_tx, constructed_rx) = std::sync::mpsc::sync_channel(1);
        let (dropped_tx, dropped_rx) = std::sync::mpsc::sync_channel(1);
        let (driver, ()) = EngineDriver::start_with_drop_probe(
            move || {
                constructed_tx
                    .send(thread::current().id())
                    .expect("record construction thread");
                Ok((
                    Engine::from_dir(&model_dir, onnx_genai_engine::EngineConfig::default())?,
                    (),
                ))
            },
            1,
            2,
            EngineDropThread(dropped_tx),
        )
        .expect("load tiny fixture on worker");
        let construction_thread = constructed_rx
            .recv()
            .expect("worker reports construction thread");
        assert_ne!(
            construction_thread, caller_thread,
            "the engine must be constructed on the worker, not the caller"
        );
        assert_eq!(driver.workers.len(), 1, "phase one runs a pool of one");
        let worker = Arc::clone(driver.workers.primary());
        assert!(worker.is_running());

        // A session is opened on, and named by, the pool's only worker.
        let session = driver.create_session().await.expect("open session");
        assert_eq!(session.worker, WorkerId::PRIMARY);
        let leases = crate::lease::SessionLeases::with_shards(1);
        let lease = leases
            .acquire(driver.binding(session), "sess-shutdown")
            .expect("an idle session is leasable");
        driver.close_session(lease).await.expect("close session");

        driver.shutdown();
        assert!(worker.is_joined(), "shutdown must join the worker thread");
        assert!(!worker.is_running());
        assert_eq!(
            dropped_rx.recv().expect("record engine teardown thread"),
            construction_thread,
            "the engine must be destroyed on the thread that constructed it",
        );

        // Idempotent: a second shutdown neither panics nor waits on anything.
        driver.shutdown();
        assert!(worker.is_joined());

        // And every command now fails the way a stopped driver always has.
        let error = driver
            .create_session()
            .await
            .expect_err("a stopped driver opens no sessions");
        assert_eq!(error.to_string(), "engine driver stopped");
        assert!(matches!(
            driver.generate(None, warmup_request(), None).await,
            Err(GenerateSubmitError::DriverStopped)
        ));
        assert!(matches!(
            driver.session_token_count(session).await,
            Err(error) if error.to_string() == "engine driver stopped"
        ));
    }

    #[test]
    fn initialization_panic_is_returned_as_a_driver_start_error() {
        let error = match EngineDriver::start(
            || -> anyhow::Result<(Engine, ())> {
                panic!("fixture engine initializer panicked");
            },
            1,
            1,
        ) {
            Ok(_) => panic!("a panicked initializer must not produce a driver"),
            Err(error) => format!("{error:#}"),
        };

        assert!(
            error.contains("failed to start engine worker 0")
                && error.contains("engine worker initialization panicked")
                && error.contains("fixture engine initializer panicked"),
            "panic must retain worker and initialization context: {error}",
        );
    }

    /// A session-bound command addressed to a worker the pool does not have is
    /// refused by name, not routed to whichever worker happens to exist.
    #[tokio::test]
    async fn a_session_on_an_unknown_worker_is_refused() {
        let (commands, _rx) = mpsc::channel(1);
        let driver = EngineDriver {
            workflow_facts: WorkflowFacts {
                components: 0,
                graph_components: 0,
                declares_generation_loop: false,
            },
            workers: WorkerPool::single(WorkerHandle::detached(WorkerId::PRIMARY, commands)),
            model: ModelKey::new("stub"),
            generation_capacity: Arc::new(Semaphore::new(1)),
            generation_capacity_size: 1,
            kv_telemetry: Arc::new(KvTelemetry::default()),
            worker_kv_telemetry: Arc::new(vec![Arc::new(KvTelemetry::default())]),
            resource_snapshot: Arc::new(Mutex::new(None)),
            memory_strategy_plan: Arc::new(onnx_genai_engine::MemoryStrategyPlan::unknown(
                0,
                None,
                "driver test stub",
            )),
            memory_accounting: None,
            batching: Arc::new(BatchingReport::single_sequence_stub()),
        };
        let elsewhere = SessionPlacement::new(WorkerId::new(4), 1);

        let leases = crate::lease::SessionLeases::with_shards(1);
        let lease = leases
            .acquire(driver.binding(elsewhere), "sess-elsewhere")
            .expect("a lease is taken before the worker is even consulted");
        let error = driver
            .close_session(lease)
            .await
            .expect_err("worker 4 is not in a pool of one");
        assert_eq!(
            error.to_string(),
            "engine worker 4 is not in this pool (1 worker(s) loaded)"
        );
        assert!(
            !leases.is_held(&driver.binding(elsewhere)),
            "a close that failed still gave the lease back"
        );
    }

    /// A turn that never reaches a worker leaves nothing leased.
    ///
    /// The guard travels *inside* the command, which is what makes every
    /// terminal path release it without a release call to remember. This pins
    /// the two exits that never reach the engine at all: a full admission
    /// gate, and a send onto a channel whose worker is gone. Leaking either
    /// would make the session permanently unusable — every later turn on it
    /// would be refused 409 forever, with no turn in flight to justify it.
    #[tokio::test]
    async fn a_turn_that_never_reaches_a_worker_releases_its_lease() {
        let (commands, rx) = mpsc::channel(1);
        let driver = EngineDriver {
            workflow_facts: WorkflowFacts {
                components: 0,
                graph_components: 0,
                declares_generation_loop: false,
            },
            workers: WorkerPool::single(WorkerHandle::detached(WorkerId::PRIMARY, commands)),
            model: ModelKey::new("stub"),
            // One permit, so the second acquisition below finds the gate shut.
            generation_capacity: Arc::new(Semaphore::new(1)),
            generation_capacity_size: 1,
            kv_telemetry: Arc::new(KvTelemetry::default()),
            worker_kv_telemetry: Arc::new(vec![Arc::new(KvTelemetry::default())]),
            resource_snapshot: Arc::new(Mutex::new(None)),
            memory_strategy_plan: Arc::new(onnx_genai_engine::MemoryStrategyPlan::unknown(
                0,
                None,
                "driver test stub",
            )),
            memory_accounting: None,
            batching: Arc::new(BatchingReport::single_sequence_stub()),
        };
        let session = SessionPlacement::new(WorkerId::PRIMARY, 7);
        let leases = crate::lease::SessionLeases::with_shards(1);

        // 1. Refused by the admission gate, before a command is ever built.
        let hog = Arc::clone(&driver.generation_capacity)
            .try_acquire_owned()
            .expect("the only permit");
        let lease = leases
            .acquire(driver.binding(session), "sess-overloaded")
            .expect("an idle session is leasable");
        assert!(leases.is_held(&driver.binding(session)));
        assert!(matches!(
            driver.generate(Some(lease), warmup_request(), None).await,
            Err(GenerateSubmitError::Overloaded)
        ));
        assert!(
            !leases.is_held(&driver.binding(session)),
            "a turn refused admission is not a turn in flight"
        );
        drop(hog);

        // 2. Accepted by the gate, then handed to a worker that is gone.
        drop(rx);
        let lease = leases
            .acquire(driver.binding(session), "sess-stopped")
            .expect("still leasable");
        assert!(matches!(
            driver.generate(Some(lease), warmup_request(), None).await,
            Err(GenerateSubmitError::DriverStopped)
        ));
        assert_eq!(
            leases.held(),
            0,
            "a stopped driver leaves no session leased forever"
        );

        // And the session is ordinary again: the refusals cost it nothing.
        drop(
            leases
                .acquire(driver.binding(session), "sess-after")
                .expect("leasable after both refusals"),
        );
    }

    fn warmup_request() -> GenerateRequest {
        GenerateRequest {
            prompt: onnx_genai::GeneratePrompt::TokenIds(vec![0]),
            options: GenerateOptions {
                max_new_tokens: 1,
                ..GenerateOptions::default()
            },
        }
    }

    /// A lone request with nothing else outstanding must be admitted at once.
    ///
    /// This is the regression that made the "solo fast path" not fast: the loop
    /// had no exit for this case except the settle deadline, so every solo
    /// generation on a batching-capable engine paid MICROBATCH_MIN_WAIT of
    /// first-token latency before taking a path that was supposed to skip the
    /// batching machinery entirely.
    #[test]
    fn a_solo_request_is_admitted_without_waiting_to_settle() {
        assert_eq!(
            admission_step(1, 1, 0, true, false),
            AdmissionStep::Admit,
            "a solo request must not wait out the settle window"
        );
    }

    /// The same request must still be admitted once the window has closed.
    #[test]
    fn a_solo_request_is_admitted_after_the_settle_window_too() {
        assert_eq!(admission_step(1, 1, 0, true, true), AdmissionStep::Admit);
    }

    /// An outstanding sibling holds the batch open to the hard deadline.
    #[test]
    fn an_outstanding_sibling_holds_the_batch_open() {
        assert_eq!(
            admission_step(1, 2, 0, true, false),
            AdmissionStep::WaitForSibling
        );
        assert_eq!(
            admission_step(1, 2, 0, true, true),
            AdmissionStep::WaitForSibling,
            "a sibling outranks the settle deadline"
        );
    }

    /// A sibling that stops being outstanding must release the request.
    ///
    /// `saw_pending_sibling` used to latch, so a single transient over-count
    /// pinned a lone request to the slow continuous-batch path for the rest of
    /// the admission window even after the supposed sibling had drained.
    #[test]
    fn a_sibling_that_drains_stops_holding_the_batch() {
        assert_eq!(
            admission_step(1, 2, 0, true, false),
            AdmissionStep::WaitForSibling
        );
        // Next iteration: the sibling is gone.
        assert_eq!(
            admission_step(1, 1, 0, true, false),
            AdmissionStep::Admit,
            "the decision must be re-derived, not latched"
        );
    }

    /// Having just accepted an arrival, wait briefly for its neighbours.
    #[test]
    fn a_fresh_arrival_waits_out_the_settle_window() {
        assert_eq!(
            admission_step(2, 2, 1, true, false),
            AdmissionStep::WaitToSettle,
            "an arrival suggests more may be in flight"
        );
        assert_eq!(
            admission_step(2, 2, 1, true, true),
            AdmissionStep::Admit,
            "but only until the window closes"
        );
    }

    /// Queued commands mean more work is available, so do not admit early.
    #[test]
    fn queued_commands_prevent_the_early_solo_exit() {
        assert_eq!(
            admission_step(1, 1, 0, false, false),
            AdmissionStep::WaitToSettle,
            "a non-empty deferred queue is not a solo request"
        );
    }
}

#[cfg(test)]
mod driver_delivery_tests {
    use super::*;
    use onnx_genai_engine::FinishReason;

    fn token(id: u32) -> GenerateToken {
        GenerateToken {
            token_id: id,
            text: format!("t{id}"),
            finish_reason: None,
        }
    }

    fn result(tokens: usize) -> GenerateResult {
        GenerateResult {
            text: "done".to_string(),
            token_ids: (0..tokens as u32).collect(),
            finish_reason: FinishReason::EosToken,
            prefix_cache_hit_len: 0,
            logprobs: None,
            budget_cap: None,
        }
    }

    /// The defect: a generation longer than the output buffer aborted, and the
    /// terminal event was then dropped into the same full buffer, so the client
    /// was told the stream ended rather than what happened. A single request
    /// owns its channel, so the driver must wait for capacity instead.
    #[test]
    fn a_generation_longer_than_the_output_buffer_still_delivers_every_event() {
        let (events, mut rx) = mpsc::channel(DRIVER_OUTPUT_BUFFER);
        let produced = DRIVER_OUTPUT_BUFFER * 3;
        let sender = thread::spawn(move || {
            for _ in 0..produced {
                deliver_event(&events, DriverEvent::Token(token(0)))
                    .expect("a live receiver must accept every token");
            }
            deliver_event(&events, DriverEvent::Finished(result(0)))
                .expect("the terminal event must not be dropped for a full buffer");
        });

        let mut tokens = 0;
        let mut finished = false;
        while let Some(event) = rx.blocking_recv() {
            match event {
                DriverEvent::Token(_) => tokens += 1,
                DriverEvent::Finished(_) => {
                    finished = true;
                    break;
                }
                DriverEvent::Error(error) => panic!("unexpected failure: {}", error.message),
            }
        }
        sender.join().expect("sender thread panicked");
        assert_eq!(tokens, produced);
        assert!(finished, "the result must reach the caller");
    }

    /// A receiver that is genuinely gone must still be reported, so a
    /// disconnected client stops the generation instead of blocking it forever.
    #[test]
    fn a_dropped_receiver_is_reported_as_a_closed_stream() {
        let (events, rx) = mpsc::channel::<DriverEvent>(DRIVER_OUTPUT_BUFFER);
        drop(rx);
        let error = deliver_event(&events, DriverEvent::Token(token(0)))
            .expect_err("a dropped receiver cannot accept events");
        assert!(error.to_string().contains("stream receiver closed"));
    }

    /// Waiting for capacity must stay bounded. These paths run on the driver's
    /// one dedicated thread, so a consumer that holds its receiver but never
    /// reads it has to be abandoned, or every later request waits behind it.
    /// It is reported as stalled, not as closed, because those differ.
    #[test]
    fn a_receiver_that_never_reads_is_abandoned_within_the_budget() {
        let (events, _held) = mpsc::channel::<DriverEvent>(1);
        deliver_event(&events, DriverEvent::Token(token(0))).expect("the first event fits");

        let started = Instant::now();
        let error = deliver_event(&events, DriverEvent::Token(token(0)))
            .expect_err("a receiver that never reads cannot be waited on forever");
        let waited = started.elapsed();

        assert!(error.to_string().contains("stream consumer stalled"));
        assert!(waited >= DELIVERY_GRACE, "gave up too early: {waited:?}");
        assert!(
            waited < DELIVERY_GRACE * 3,
            "held the driver far past the budget: {waited:?}"
        );
    }

    fn route(capacity: usize) -> (DriverRoute, mpsc::Receiver<DriverEvent>) {
        let (events, events_rx) = mpsc::channel(capacity);
        let permit = Arc::new(Semaphore::new(1)).try_acquire_owned().unwrap();
        (
            DriverRoute {
                admission: None,
                events,
                _permit: WorkerPermit::untracked(permit),
                _lease: None,
                metrics: GenerationMetrics::start(),
            },
            events_rx,
        )
    }

    #[tokio::test]
    async fn full_channel_applies_backpressure_instead_of_failing() {
        // Regression: a generation longer than the bounded channel used to abort
        // with "stream receiver closed: no available capacity" even though the
        // consumer was alive and reading.
        let (events, mut events_rx) = mpsc::channel(2);
        let producer = tokio::task::spawn_blocking(move || {
            for id in 0..8 {
                deliver_driver_event(&events, DriverEvent::Token(token(id)), DELIVERY_GRACE)
                    .expect("live consumer must not be treated as disconnected");
            }
        });

        let mut seen = Vec::new();
        while let Some(DriverEvent::Token(token)) = events_rx.recv().await {
            seen.push(token.token_id);
        }
        producer.await.unwrap();
        assert_eq!(seen, (0..8).collect::<Vec<_>>(), "no token may be dropped");
    }

    #[tokio::test]
    async fn dropped_consumer_reports_disconnect() {
        let (events, events_rx) = mpsc::channel(1);
        drop(events_rx);
        let failed = tokio::task::spawn_blocking(move || {
            deliver_driver_event(&events, DriverEvent::Token(token(0)), DELIVERY_GRACE).is_err()
        })
        .await
        .unwrap();
        assert!(
            failed,
            "a closed channel must surface as a delivery failure"
        );
    }

    #[tokio::test]
    async fn slow_batch_consumer_keeps_route_and_receives_terminal_event() {
        let handle = ContinuousBatchHandle { id: 7 };
        let (driver_route, mut events_rx) = route(2);
        let mut routes = HashMap::from([(handle.id, driver_route)]);
        let mut abandoned = HashMap::new();

        let mut events: Vec<_> = (0..6)
            .map(|id| ContinuousBatchEvent::Token {
                handle,
                token: token(id),
            })
            .collect();
        events.push(ContinuousBatchEvent::Finished {
            handle,
            result: result(6),
        });

        let routing = tokio::task::spawn_blocking(move || {
            route_continuous_events(events, &mut routes, &mut abandoned);
            (routes.len(), abandoned.len())
        });

        let mut tokens = Vec::new();
        let mut finished = false;
        while let Some(event) = events_rx.recv().await {
            match event {
                DriverEvent::Token(token) => tokens.push(token.token_id),
                DriverEvent::Finished(_) => finished = true,
                DriverEvent::Error(error) => panic!("unexpected failure: {error:?}"),
            }
        }

        let (live, abandoned) = routing.await.unwrap();
        assert_eq!(tokens, (0..6).collect::<Vec<_>>());
        assert!(finished, "the terminal event must survive a full channel");
        assert_eq!(abandoned, 0, "a reading consumer must not be abandoned");
        assert_eq!(live, 0, "the finished route is retired, not abandoned");
    }

    #[tokio::test]
    async fn aborted_continuous_turn_publishes_only_an_error() {
        let handle = ContinuousBatchHandle { id: 13 };
        let (driver_route, mut events_rx) = route(2);
        let mut routes = HashMap::from([(handle.id, driver_route)]);
        let mut abandoned = HashMap::new();

        route_continuous_events(
            vec![ContinuousBatchEvent::Aborted {
                handle,
                outcome: onnx_genai_engine::pipeline::TurnTransactionOutcome::AbortToBaseline {
                    transaction: onnx_genai_engine::pipeline::TurnTransactionId(3),
                    baseline: onnx_genai_engine::pipeline::TurnBaselineId(3),
                    reason: onnx_genai_engine::pipeline::TurnAbortReason::ExecutionFailure,
                },
                error: "injected later execution failure".to_string(),
            }],
            &mut routes,
            &mut abandoned,
        );

        match events_rx.recv().await {
            Some(DriverEvent::Error(failure)) => assert!(
                failure
                    .message
                    .contains("aborted to its committed baseline")
                    && failure.message.contains("injected later execution failure"),
                "abort error must preserve actionable turn context: {}",
                failure.message
            ),
            Some(DriverEvent::Token(token)) => {
                panic!("aborted turn leaked DriverEvent::Token({})", token.token_id)
            }
            Some(DriverEvent::Finished(_)) => {
                panic!("aborted turn leaked a success completion")
            }
            None => panic!("aborted turn must publish an error"),
        }
        assert!(
            events_rx.recv().await.is_none(),
            "aborted turn must not publish token or success completion after its error"
        );
        assert!(routes.is_empty(), "aborted route must be retired");
        assert!(
            abandoned.is_empty(),
            "aborted route must not remain retained"
        );
    }

    #[tokio::test]
    async fn stalled_consumer_is_abandoned_after_the_grace_period() {
        // A consumer that stops draining entirely must lose its route so it
        // cannot hold up the rows batched alongside it.
        let handle = ContinuousBatchHandle { id: 11 };
        let (driver_route, _held_rx) = route(1);
        let mut routes = HashMap::from([(handle.id, driver_route)]);
        let mut abandoned = HashMap::new();

        let elapsed = tokio::task::spawn_blocking(move || {
            let started = Instant::now();
            route_continuous_events(
                (0..4)
                    .map(|id| ContinuousBatchEvent::Token {
                        handle,
                        token: token(id),
                    })
                    .collect(),
                &mut routes,
                &mut abandoned,
            );
            (started.elapsed(), routes.len(), abandoned.len())
        })
        .await
        .unwrap();

        let (duration, live, abandoned) = elapsed;
        assert_eq!(live, 0, "a stalled route is dropped");
        assert_eq!(abandoned, 1);
        assert!(
            duration < DELIVERY_GRACE * 3,
            "the driver must not wait out the grace period once per token, took {duration:?}"
        );
    }

    #[tokio::test]
    async fn disconnected_batch_consumer_is_abandoned() {
        let handle = ContinuousBatchHandle { id: 3 };
        let (driver_route, events_rx) = route(4);
        drop(events_rx);
        let mut routes = HashMap::from([(handle.id, driver_route)]);
        let mut abandoned = HashMap::new();

        route_continuous_events(
            vec![ContinuousBatchEvent::Token {
                handle,
                token: token(0),
            }],
            &mut routes,
            &mut abandoned,
        );

        assert!(routes.is_empty(), "a dead consumer loses its route");
        assert_eq!(abandoned.len(), 1);
    }
}
