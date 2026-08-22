use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use onnx_genai::{
    Engine, GenerateOptions, GenerateRequest, GenerateResult, GenerateToken, SessionId, TokenId,
};
use onnx_genai_engine::{
    BatchingCapability, ContinuousBatchAdmission, ContinuousBatchEvent, ContinuousBatchHandle,
    ContinuousBatchManager, DeviceMemoryAuthority, EmbeddingOptions, EncodedAudio,
    EngineGovernorError, FimConfig, GovernorSnapshot, KvNotApplicable, KvTelemetry,
    MemoryStrategyPlan, PipelineEngine, PipelineGenerateRequest, ResourceLimit,
    SchedulerAdmissionError,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot};

use crate::image_generation::{ImageExecutionRequest, ProducedImage};
use crate::metrics::GenerationMetrics;
use crate::multimodal::MultimodalInput;

const DRIVER_OUTPUT_BUFFER: usize = 16;
const MICROBATCH_MIN_WAIT: Duration = Duration::from_millis(2);
const MICROBATCH_MAX_WAIT: Duration = Duration::from_millis(12);
const MICROBATCH_SETTLE_WAIT: Duration = Duration::from_millis(1);
const MICROBATCH_POLL_WAIT: Duration = Duration::from_micros(250);

#[derive(Clone)]
pub(crate) struct EngineDriver {
    pub(crate) commands: mpsc::Sender<DriverCommand>,
    pub(crate) generation_capacity: Arc<Semaphore>,
    pub(crate) generation_capacity_size: u32,
    /// Lock-free mirror of the KV page pool, readable during a generation.
    pub(crate) kv_telemetry: Arc<KvTelemetry>,
    /// Latest engine-ledger snapshot, readable without a driver-thread round trip.
    pub(crate) resource_snapshot: Arc<Mutex<Option<GovernorSnapshot>>>,
    pub(crate) memory_strategy_plan: Arc<MemoryStrategyPlan>,
    pub(crate) device_authority: Option<DeviceMemoryAuthority>,
    /// Honest, decode-path-sourced batching report for this engine, resolved at
    /// startup. Surfaced over `/v1/resources` and `/v1/debug/kv` so an operator
    /// sees `batch_supported=false` / effective max batch = 1 directly instead of
    /// inferring it from a debug-level "using per-request engine path" log line.
    pub(crate) batching: Arc<BatchingReport>,
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

    /// The report for a pipeline engine, which always serves one request at a
    /// time (its components own their own caches rather than one batched pass).
    fn pipeline() -> Self {
        Self {
            supported: false,
            requested_max_batch: 1,
            effective_max_batch: 1,
            reason: "pipeline engines serve one request at a time; their \
                     components own separate caches rather than a shared batched \
                     forward pass"
                .to_string(),
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
    CreateSession(tokio::sync::oneshot::Sender<anyhow::Result<SessionId>>),
    CloseSession {
        session_id: SessionId,
        response: tokio::sync::oneshot::Sender<anyhow::Result<()>>,
    },
    SessionTokenCount {
        session_id: SessionId,
        response: tokio::sync::oneshot::Sender<anyhow::Result<usize>>,
    },
    Generate {
        session_id: Option<SessionId>,
        request: Box<GenerateRequest>,
        admission: oneshot::Sender<Result<(), DriverFailure>>,
        events: mpsc::Sender<DriverEvent>,
        permit: OwnedSemaphorePermit,
    },
    GeneratePipeline {
        request: Box<GenerateRequest>,
        input: Option<MultimodalInput>,
        admission: oneshot::Sender<Result<(), DriverFailure>>,
        events: mpsc::Sender<DriverEvent>,
        permit: OwnedSemaphorePermit,
    },
    SynthesizeSpeech {
        request: Box<GenerateRequest>,
        reply: oneshot::Sender<anyhow::Result<EncodedAudio>>,
        permit: OwnedSemaphorePermit,
    },
    GenerateImage {
        request: Box<ImageExecutionRequest>,
        reply: oneshot::Sender<anyhow::Result<ProducedImage>>,
        permit: OwnedSemaphorePermit,
        track_metrics: bool,
    },
    GenerateFim {
        prefix: String,
        suffix: String,
        fim_config: FimConfig,
        options: Box<GenerateOptions>,
        admission: oneshot::Sender<Result<(), DriverFailure>>,
        events: mpsc::Sender<DriverEvent>,
        permit: OwnedSemaphorePermit,
    },
    Embed {
        input_ids: Vec<TokenId>,
        options: EmbeddingOptions,
        reply: tokio::sync::oneshot::Sender<anyhow::Result<Vec<f32>>>,
    },
    #[cfg(test)]
    ResourceSnapshot(tokio::sync::oneshot::Sender<anyhow::Result<GovernorSnapshot>>),
    SetVramLimit {
        limit: ResourceLimit,
        reply: tokio::sync::oneshot::Sender<
            anyhow::Result<Result<GovernorSnapshot, EngineGovernorError>>,
        >,
    },
}

#[derive(Debug)]
pub(crate) enum DriverEvent {
    Token(GenerateToken),
    Finished(GenerateResult),
    Error(DriverFailure),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DriverFailureKind {
    Internal,
    MemoryOverload,
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
        Self {
            // Anyhow's Display shows only the outermost context, which for a
            // decode failure is the generic "forward pass failed" wrapper. The
            // alternate form keeps the whole chain, and this message is the
            // only thing the client ever sees.
            message: format!("{error:#}"),
            kind: if memory_overload {
                DriverFailureKind::MemoryOverload
            } else {
                DriverFailureKind::Internal
            },
        }
    }
}

enum EngineBackend {
    Single(Box<Engine>),
    Pipeline(Box<PipelineEngine>),
}

struct EngineOwner(EngineBackend);

#[derive(Debug)]
pub(crate) enum GenerateSubmitError {
    Overloaded,
    DriverStopped,
    Failed(DriverFailure),
}

struct DriverRoute {
    admission: Option<oneshot::Sender<Result<(), DriverFailure>>>,
    events: mpsc::Sender<DriverEvent>,
    _permit: OwnedSemaphorePermit,
    metrics: GenerationMetrics,
}

struct PendingGeneration {
    request: GenerateRequest,
    admission: oneshot::Sender<Result<(), DriverFailure>>,
    events: mpsc::Sender<DriverEvent>,
    permit: OwnedSemaphorePermit,
}

#[derive(Clone, Copy)]
struct MicrobatchAdmission<'a> {
    max_queue_depth: usize,
    generation_capacity: &'a Semaphore,
}

// SAFETY: The engine is moved exactly once into the dedicated driver thread.
// All ORT runners, sessions, KV state, and the continuous batch manager stay
// owned by that thread and are accessed only by processing channel commands.
unsafe impl Send for EngineOwner {}

impl EngineDriver {
    pub(crate) fn start(engine: Engine, max_batch: usize, max_queue_depth: usize) -> Self {
        let (commands, rx) = mpsc::channel(max_queue_depth);
        let generation_capacity = Arc::new(Semaphore::new(max_queue_depth));
        let driver_capacity = generation_capacity.clone();
        // Attach before the engine moves onto the driver thread: this is the
        // last point at which it is reachable from here, and the mirror must
        // outlive that move because reading it is the whole reason it exists.
        let mut engine = engine;
        // Resolve the honest batching capability from the decode path (not from
        // whether an ORT session exists) while the engine is still borrowable,
        // then clamp the requested width to what the path can actually honor.
        let batching = Arc::new(BatchingReport::from_capability(
            &engine.batching_capability(),
            max_batch,
        ));
        let effective_max_batch = batching.effective_max_batch;
        tracing::info!(
            batch_supported = batching.supported,
            requested_max_batch = batching.requested_max_batch,
            effective_max_batch = batching.effective_max_batch,
            reason = %batching.reason,
            "resolved batching capability",
        );
        let device_authority = Some(engine.governor().device_authority());
        let kv_telemetry = Arc::new(KvTelemetry::default());
        if engine.attach_kv_telemetry(Arc::clone(&kv_telemetry)) {
            kv_telemetry.set_applicable();
        } else {
            kv_telemetry.set_not_applicable(KvNotApplicable::CacheCannotPage);
        }
        let memory_strategy_plan = Arc::new(engine.memory_strategy_plan().clone());
        let owner = EngineOwner(EngineBackend::Single(Box::new(engine)));
        let resource_snapshot = Arc::new(Mutex::new(Some(match &owner.0 {
            EngineBackend::Single(engine) => engine.resource_snapshot(),
            EngineBackend::Pipeline(_) => unreachable!("single-engine owner just constructed"),
        })));
        let driver_snapshot = Arc::clone(&resource_snapshot);
        thread::Builder::new()
            .name("onnx-genai-batch-driver".to_string())
            .spawn(move || {
                run_engine_driver(
                    owner,
                    rx,
                    effective_max_batch,
                    max_queue_depth,
                    driver_capacity,
                    driver_snapshot,
                )
            })
            .expect("failed to spawn onnx-genai engine driver");
        Self {
            commands,
            generation_capacity,
            generation_capacity_size: u32::try_from(max_queue_depth).unwrap_or(u32::MAX),
            kv_telemetry,
            resource_snapshot,
            memory_strategy_plan,
            device_authority,
            batching,
        }
    }

    pub(crate) fn start_pipeline(engine: PipelineEngine, max_queue_depth: usize) -> Self {
        let (commands, rx) = mpsc::channel(max_queue_depth);
        let generation_capacity = Arc::new(Semaphore::new(max_queue_depth));
        let driver_capacity = generation_capacity.clone();
        let device_authority = Some(engine.device_authority());
        let memory_strategy_plan = Arc::new(engine.memory_strategy_plan().clone());
        let owner = EngineOwner(EngineBackend::Pipeline(Box::new(engine)));
        let resource_snapshot = Arc::new(Mutex::new(Some(match &owner.0 {
            EngineBackend::Pipeline(engine) => engine.resource_snapshot(),
            EngineBackend::Single(_) => unreachable!("pipeline owner just constructed"),
        })));
        let driver_snapshot = Arc::clone(&resource_snapshot);
        // A pipeline engine owns its components' caches rather than one page
        // table, so there is nothing here to mirror. Reported as an explicit
        // not-applicable rather than an all-zero pool, which would read as an
        // idle paged cache instead of an absent one.
        let kv_telemetry = Arc::new(KvTelemetry::default());
        kv_telemetry.set_not_applicable(KvNotApplicable::CacheCannotPage);
        thread::Builder::new()
            .name("onnx-genai-pipeline-driver".to_string())
            .spawn(move || {
                run_engine_driver(
                    owner,
                    rx,
                    1,
                    max_queue_depth,
                    driver_capacity,
                    driver_snapshot,
                )
            })
            .expect("failed to spawn onnx-genai pipeline driver");
        Self {
            commands,
            generation_capacity,
            generation_capacity_size: u32::try_from(max_queue_depth).unwrap_or(u32::MAX),
            kv_telemetry,
            resource_snapshot,
            memory_strategy_plan,
            device_authority,
            batching: Arc::new(BatchingReport::pipeline()),
        }
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

    pub(crate) async fn create_session(&self) -> anyhow::Result<SessionId> {
        let (response, rx) = tokio::sync::oneshot::channel();
        self.commands
            .send(DriverCommand::CreateSession(response))
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?
    }

    pub(crate) async fn close_session(&self, session_id: SessionId) -> anyhow::Result<()> {
        let (response, rx) = tokio::sync::oneshot::channel();
        self.commands
            .send(DriverCommand::CloseSession {
                session_id,
                response,
            })
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?
    }

    pub(crate) async fn session_token_count(&self, session_id: SessionId) -> anyhow::Result<usize> {
        let (response, rx) = tokio::sync::oneshot::channel();
        self.commands
            .send(DriverCommand::SessionTokenCount {
                session_id,
                response,
            })
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?
    }

    pub(crate) async fn generate(
        &self,
        session_id: Option<SessionId>,
        request: GenerateRequest,
    ) -> Result<DriverGeneration, GenerateSubmitError> {
        let permit = self
            .generation_capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| GenerateSubmitError::Overloaded)?;
        let (events, rx) = mpsc::channel(DRIVER_OUTPUT_BUFFER);
        let (admission, admission_rx) = oneshot::channel();
        crate::metrics::generation_queued();
        if self
            .commands
            .send(DriverCommand::Generate {
                session_id,
                request: Box::new(request),
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
    pub(crate) fn warmup(&self, request: GenerateRequest, pipeline: bool) -> anyhow::Result<()> {
        let permit = self
            .generation_capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| anyhow::anyhow!("generation capacity exceeded"))?;
        let (events, mut receiver) = mpsc::channel(DRIVER_OUTPUT_BUFFER);
        let (admission, _admission_rx) = oneshot::channel();
        let command = if pipeline {
            DriverCommand::GeneratePipeline {
                request: Box::new(request),
                input: None,
                admission,
                events,
                permit,
            }
        } else {
            DriverCommand::Generate {
                session_id: None,
                request: Box::new(request),
                admission,
                events,
                permit,
            }
        };
        self.commands
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

    pub(crate) async fn generate_pipeline(
        &self,
        request: GenerateRequest,
        input: Option<MultimodalInput>,
    ) -> Result<DriverGeneration, GenerateSubmitError> {
        let permit = self
            .generation_capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| GenerateSubmitError::Overloaded)?;
        let (events, rx) = mpsc::channel(DRIVER_OUTPUT_BUFFER);
        let (admission, admission_rx) = oneshot::channel();
        crate::metrics::generation_queued();
        if self
            .commands
            .send(DriverCommand::GeneratePipeline {
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

    pub(crate) async fn synthesize_speech(
        &self,
        request: GenerateRequest,
    ) -> Result<EncodedAudio, GenerateSubmitError> {
        let permit = self
            .generation_capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| GenerateSubmitError::Overloaded)?;
        let (reply, response) = oneshot::channel();
        crate::metrics::generation_queued();
        if self
            .commands
            .send(DriverCommand::SynthesizeSpeech {
                request: Box::new(request),
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
        let permit = self
            .generation_capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| GenerateSubmitError::Overloaded)?;
        let (reply, receiver) = oneshot::channel();
        crate::metrics::generation_queued();
        if self
            .commands
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
        let permit = self
            .generation_capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| anyhow::anyhow!("generation capacity exceeded"))?;
        let (reply, receiver) = oneshot::channel();
        self.commands
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
        let permit = self
            .generation_capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| GenerateSubmitError::Overloaded)?;
        let (events, rx) = mpsc::channel(DRIVER_OUTPUT_BUFFER);
        let (admission, admission_rx) = oneshot::channel();
        crate::metrics::generation_queued();
        if self
            .commands
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
        let _permit = Arc::clone(&self.generation_capacity)
            .acquire_owned()
            .await
            .map_err(|_| anyhow::anyhow!("engine admission semaphore closed"))?;
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.commands
            .send(DriverCommand::Embed {
                input_ids,
                options,
                reply,
            })
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?
    }

    pub(crate) async fn resource_snapshot(&self) -> anyhow::Result<GovernorSnapshot> {
        let mut snapshot = self
            .resource_snapshot
            .lock()
            .expect("resource snapshot mirror lock poisoned")
            .clone()
            .ok_or_else(|| anyhow::anyhow!("resource governor is not available for this model"))?;
        if let Some(authority) = &self.device_authority {
            let used = authority.used_bytes();
            let limit = authority.limit_bytes();
            snapshot.vram.used = used;
            snapshot.vram.limit = limit;
            snapshot.vram.headroom = limit.saturating_sub(used);
            // `resolved_limits.vram_bytes` is the *resolved device (VRAM)
            // capacity limit*, which stays `None` when the device capacity could
            // not be measured (#947). The shared authority's ceiling on such a
            // box is the host-RAM-derived advisory bound, not a measured VRAM
            // capacity, so it is surfaced through `vram.limit` only and must not
            // be relabelled here as a resolved VRAM capacity. When the device WAS
            // measured, refresh it to the shared authority's live ceiling.
            if snapshot.resolved_limits.vram_bytes.is_some() {
                snapshot.resolved_limits.vram_bytes = Some(limit);
            }
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
        self.commands
            .send(DriverCommand::SetVramLimit { limit, reply })
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?
    }
}

fn run_engine_driver(
    owner: EngineOwner,
    rx: mpsc::Receiver<DriverCommand>,
    max_batch: usize,
    max_queue_depth: usize,
    generation_capacity: Arc<Semaphore>,
    resource_snapshot: Arc<Mutex<Option<GovernorSnapshot>>>,
) {
    let mut engine = match owner.0 {
        EngineBackend::Single(engine) => *engine,
        EngineBackend::Pipeline(mut pipeline) => {
            run_pipeline_driver(&mut pipeline, rx, &resource_snapshot);
            return;
        }
    };
    let continuous_batch_supported = engine.continuous_batch_manager(max_batch).is_ok();
    if continuous_batch_supported {
        tracing::info!(max_batch, "continuous batch driver enabled");
        run_static_engine_driver(
            &mut engine,
            rx,
            max_batch,
            max_queue_depth,
            &generation_capacity,
            &resource_snapshot,
        );
    } else {
        tracing::info!(
            batch_supported = false,
            effective_max_batch = 1,
            "continuous batch driver disabled; using per-request engine path (single-sequence decode)"
        );
        run_fallback_engine_driver(&mut engine, rx, &resource_snapshot);
    }
}

fn run_pipeline_driver(
    engine: &mut PipelineEngine,
    mut rx: mpsc::Receiver<DriverCommand>,
    resource_snapshot: &Mutex<Option<GovernorSnapshot>>,
) {
    while let Some(command) = rx.blocking_recv() {
        match command {
            DriverCommand::GeneratePipeline {
                request,
                input,
                admission,
                events,
                permit,
            } => run_pipeline_generation(engine, *request, input, admission, events, permit),
            DriverCommand::SynthesizeSpeech {
                request,
                reply,
                permit,
            } => {
                let _permit = permit;
                let result = engine
                    .run_pipeline_outputs(PipelineGenerateRequest::new(*request))
                    .and_then(|outputs| engine.encode_audio_output(&outputs));
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
            DriverCommand::CreateSession(response) => {
                let _ = response.send(Err(anyhow::anyhow!(
                    "sessions are not supported by pipeline models"
                )));
            }
            DriverCommand::CloseSession { response, .. } => {
                let _ = response.send(Err(anyhow::anyhow!(
                    "sessions are not supported by pipeline models"
                )));
            }
            DriverCommand::SessionTokenCount { response, .. } => {
                let _ = response.send(Err(anyhow::anyhow!(
                    "sessions are not supported by pipeline models"
                )));
            }
            DriverCommand::Generate {
                admission, events, ..
            }
            | DriverCommand::GenerateFim {
                admission, events, ..
            } => {
                crate::metrics::generation_queue_cancelled();
                let failure =
                    DriverFailure::internal("invalid generation route for pipeline model");
                let _ = admission.send(Err(failure.clone()));
                let _ = events.try_send(DriverEvent::Error(failure));
            }
            DriverCommand::Embed { reply, .. } => {
                let _ = reply.send(Err(anyhow::anyhow!(
                    "embeddings are not supported by pipeline models"
                )));
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
        }
        if let Ok(mut snapshot) = resource_snapshot.lock() {
            *snapshot = Some(engine.resource_snapshot());
        }
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
                    session_id: None, ..
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
                request,
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
                    },
                    resource_snapshot,
                    PendingGeneration {
                        request: *request,
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
                    request,
                    admission,
                    events,
                    permit,
                }) => {
                    initial.push(PendingGeneration {
                        request: *request,
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
                    request,
                    admission,
                    events,
                    permit,
                }) => {
                    initial.push(PendingGeneration {
                        request: *request,
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

        let in_flight = admission
            .max_queue_depth
            .saturating_sub(admission.generation_capacity.available_permits());
        let deferred_permit_holders = deferred_permit_holder_count(deferred);
        let expected_this_batch = in_flight
            .saturating_sub(deferred_permit_holders)
            .min(max_batch);
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
        run_fallback_generation(
            engine,
            None,
            pending.request,
            pending.admission,
            pending.events,
            pending.permit,
        );
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
        submit_to_continuous_manager(
            &mut manager,
            &mut routes,
            &mut abandoned,
            pending.request,
            pending.admission,
            pending.events,
            pending.permit,
        );
    }

    loop {
        while routes.len() + abandoned.len() < manager.max_batch() {
            match deferred.pop_front() {
                Some(DriverCommand::Generate {
                    session_id: None,
                    request,
                    admission,
                    events,
                    permit,
                }) => {
                    submit_to_continuous_manager(
                        &mut manager,
                        &mut routes,
                        &mut abandoned,
                        *request,
                        admission,
                        events,
                        permit,
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
                    request,
                    admission,
                    events,
                    permit,
                } => {
                    if routes.len() + abandoned.len() < manager.max_batch() {
                        submit_to_continuous_manager(
                            &mut manager,
                            &mut routes,
                            &mut abandoned,
                            *request,
                            admission,
                            events,
                            permit,
                        );
                    } else {
                        deferred.push_back(DriverCommand::Generate {
                            session_id: None,
                            request,
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
                    kind: failure.kind,
                }));
            }
            reported_occupancy = publish_batch_occupancy(manager.occupancy(), reported_occupancy);
            break;
        }
        route_continuous_admissions(manager.poll_admissions(), &mut routes);
        route_continuous_events(manager.poll(), &mut routes, &mut abandoned);
        reported_occupancy = publish_batch_occupancy(manager.occupancy(), reported_occupancy);
        if manager.is_idle() {
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
/// `generate_fim`, `render_images`, and `synthesize_speech` does.
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
                    | DriverCommand::GeneratePipeline { .. }
                    | DriverCommand::GenerateImage { .. }
                    | DriverCommand::GenerateFim { .. }
                    | DriverCommand::SynthesizeSpeech { .. }
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
    request: GenerateRequest,
    admission: oneshot::Sender<Result<(), DriverFailure>>,
    events: mpsc::Sender<DriverEvent>,
    permit: OwnedSemaphorePermit,
) {
    match manager.submit(request) {
        Ok(handle) => {
            routes.insert(
                handle.id,
                DriverRoute {
                    admission: Some(admission),
                    events,
                    _permit: permit,
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
        }
    }
}

fn handle_driver_command(engine: &mut Engine, command: DriverCommand) {
    match command {
        DriverCommand::CreateSession(response) => {
            let _ = response.send(engine.create_session());
        }
        DriverCommand::CloseSession {
            session_id,
            response,
        } => {
            let _ = response.send(engine.close_session(session_id));
        }
        DriverCommand::SessionTokenCount {
            session_id,
            response,
        } => {
            let _ = response.send(engine.session_token_count(session_id));
        }
        DriverCommand::Generate {
            session_id,
            request,
            admission,
            events,
            permit,
        } => run_fallback_generation(engine, session_id, *request, admission, events, permit),
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
        DriverCommand::GeneratePipeline {
            admission, events, ..
        } => {
            crate::metrics::generation_queue_cancelled();
            let failure =
                DriverFailure::internal("invalid pipeline generation route for single model");
            let _ = admission.send(Err(failure.clone()));
            let _ = events.try_send(DriverEvent::Error(failure));
        }
        DriverCommand::SynthesizeSpeech { reply, .. } => {
            let _ = reply.send(Err(anyhow::anyhow!(
                "speech synthesis requires a workflow pipeline model"
            )));
        }
        DriverCommand::GenerateImage { reply, .. } => {
            crate::metrics::generation_queue_cancelled();
            let _ = reply.send(Err(anyhow::anyhow!(
                "image generation requires a metadata-declared pipeline model"
            )));
        }
        DriverCommand::Embed {
            input_ids,
            options,
            reply,
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

fn run_pipeline_generation(
    engine: &mut PipelineEngine,
    request: GenerateRequest,
    input: Option<MultimodalInput>,
    admission: oneshot::Sender<Result<(), DriverFailure>>,
    events: mpsc::Sender<DriverEvent>,
    _permit: OwnedSemaphorePermit,
) {
    let mut metrics = GenerationMetrics::start();
    let pipeline_request = match input
        .map(|input| input.bind(PipelineGenerateRequest::new(request.clone())))
        .transpose()
    {
        Ok(Some(bound)) => bound,
        Ok(None) => PipelineGenerateRequest::new(request),
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
        engine.generate_with_callbacks(pipeline_request, Some(&mut admitted), Some(&mut callback))
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

fn run_fallback_generation(
    engine: &mut Engine,
    session_id: Option<SessionId>,
    request: GenerateRequest,
    admission: oneshot::Sender<Result<(), DriverFailure>>,
    events: mpsc::Sender<DriverEvent>,
    _permit: OwnedSemaphorePermit,
) {
    let mut metrics = GenerationMetrics::start();
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
        match session_id {
            Some(session_id) => engine.generate_in_session_with_callbacks(
                session_id,
                request,
                Some(&mut admitted),
                Some(&mut callback),
            ),
            None => {
                engine.generate_with_callbacks(request, Some(&mut admitted), Some(&mut callback))
            }
        }
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

fn run_fim_generation(
    engine: &mut Engine,
    prefix: String,
    suffix: String,
    fim_config: FimConfig,
    options: GenerateOptions,
    delivery: (
        oneshot::Sender<Result<(), DriverFailure>>,
        mpsc::Sender<DriverEvent>,
        OwnedSemaphorePermit,
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
                _permit: permit,
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
            commands,
            generation_capacity: Arc::new(Semaphore::new(1)),
            generation_capacity_size: 1,
            kv_telemetry: Arc::new(KvTelemetry::default()),
            resource_snapshot: Arc::new(Mutex::new(Some(snapshot.clone()))),
            memory_strategy_plan: Arc::new(engine.memory_strategy_plan().clone()),
            device_authority: None,
            batching: Arc::new(BatchingReport::from_capability(
                &engine.batching_capability(),
                1,
            )),
        };

        assert_eq!(driver.resource_snapshot().await.unwrap(), snapshot);
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
                _permit: permit,
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
