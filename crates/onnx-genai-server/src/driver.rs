use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use anyhow::Context;
use onnx_genai::text_to_audio::{SynthesizedAudio, TextToAudioRequest};
use onnx_genai::text_to_image::{RenderedImage, TextToImageRequest};
use onnx_genai::{
    Engine, GenerateOptions, GenerateRequest, GenerateResult, GenerateToken, SessionId, TokenId,
};
use onnx_genai_engine::{
    ContinuousBatchEvent, ContinuousBatchManager, EmbeddingOptions, EngineGovernorError, FimConfig,
    GovernorSnapshot, KvNotApplicable, KvTelemetry, PipelineEngine, PipelineGenerateRequest,
    ResourceLimit,
};
use onnx_genai_ort::Tokenizer;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};

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
    /// Lock-free mirror of the KV page pool, readable during a generation.
    pub(crate) kv_telemetry: Arc<KvTelemetry>,
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
        events: mpsc::Sender<DriverEvent>,
        permit: OwnedSemaphorePermit,
    },
    GeneratePipeline {
        request: Box<GenerateRequest>,
        input: Option<MultimodalInput>,
        events: mpsc::Sender<DriverEvent>,
        permit: OwnedSemaphorePermit,
    },
    GenerateFim {
        prefix: String,
        suffix: String,
        fim_config: FimConfig,
        options: Box<GenerateOptions>,
        events: mpsc::Sender<DriverEvent>,
        permit: OwnedSemaphorePermit,
    },
    Embed {
        input_ids: Vec<TokenId>,
        options: EmbeddingOptions,
        reply: tokio::sync::oneshot::Sender<anyhow::Result<Vec<f32>>>,
    },
    RenderImages {
        pipeline_dir: PathBuf,
        request: Box<TextToImageRequest>,
        reply: tokio::sync::oneshot::Sender<anyhow::Result<Vec<RenderedImage>>>,
    },
    SynthesizeSpeech {
        tokenizer: Arc<Tokenizer>,
        request: Box<TextToAudioRequest>,
        reply: tokio::sync::oneshot::Sender<anyhow::Result<SynthesizedAudio>>,
    },
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
    Error(String),
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
}

struct DriverRoute {
    events: mpsc::Sender<DriverEvent>,
    _permit: OwnedSemaphorePermit,
    metrics: GenerationMetrics,
}

struct PendingGeneration {
    request: GenerateRequest,
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
        let kv_telemetry = Arc::new(KvTelemetry::default());
        if engine.attach_kv_telemetry(Arc::clone(&kv_telemetry)) {
            kv_telemetry.set_applicable();
        } else {
            kv_telemetry.set_not_applicable(KvNotApplicable::CacheCannotPage);
        }
        let owner = EngineOwner(EngineBackend::Single(Box::new(engine)));
        thread::Builder::new()
            .name("onnx-genai-batch-driver".to_string())
            .spawn(move || {
                run_engine_driver(owner, rx, max_batch, max_queue_depth, driver_capacity)
            })
            .expect("failed to spawn onnx-genai engine driver");
        Self {
            commands,
            generation_capacity,
            kv_telemetry,
        }
    }

    pub(crate) fn start_pipeline(engine: PipelineEngine, max_queue_depth: usize) -> Self {
        let (commands, rx) = mpsc::channel(max_queue_depth);
        let generation_capacity = Arc::new(Semaphore::new(max_queue_depth));
        let driver_capacity = generation_capacity.clone();
        let owner = EngineOwner(EngineBackend::Pipeline(Box::new(engine)));
        // A pipeline engine owns its components' caches rather than one page
        // table, so there is nothing here to mirror. Reported as an explicit
        // not-applicable rather than an all-zero pool, which would read as an
        // idle paged cache instead of an absent one.
        let kv_telemetry = Arc::new(KvTelemetry::default());
        kv_telemetry.set_not_applicable(KvNotApplicable::CacheCannotPage);
        thread::Builder::new()
            .name("onnx-genai-pipeline-driver".to_string())
            .spawn(move || run_engine_driver(owner, rx, 1, max_queue_depth, driver_capacity))
            .expect("failed to spawn onnx-genai pipeline driver");
        Self {
            commands,
            generation_capacity,
            kv_telemetry,
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
    ) -> Result<mpsc::Receiver<DriverEvent>, GenerateSubmitError> {
        let permit = self
            .generation_capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| GenerateSubmitError::Overloaded)?;
        let (events, rx) = mpsc::channel(DRIVER_OUTPUT_BUFFER);
        crate::metrics::generation_queued();
        if self
            .commands
            .send(DriverCommand::Generate {
                session_id,
                request: Box::new(request),
                events,
                permit,
            })
            .await
            .is_err()
        {
            crate::metrics::generation_queue_cancelled();
            return Err(GenerateSubmitError::DriverStopped);
        }
        Ok(rx)
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
        let command = if pipeline {
            DriverCommand::GeneratePipeline {
                request: Box::new(request),
                input: None,
                events,
                permit,
            }
        } else {
            DriverCommand::Generate {
                session_id: None,
                request: Box::new(request),
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
                DriverEvent::Error(message) => anyhow::bail!(message),
            }
        }
        anyhow::bail!("generation stream ended before result")
    }

    pub(crate) async fn generate_pipeline(
        &self,
        request: GenerateRequest,
        input: Option<MultimodalInput>,
    ) -> Result<mpsc::Receiver<DriverEvent>, GenerateSubmitError> {
        let permit = self
            .generation_capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| GenerateSubmitError::Overloaded)?;
        let (events, rx) = mpsc::channel(DRIVER_OUTPUT_BUFFER);
        crate::metrics::generation_queued();
        if self
            .commands
            .send(DriverCommand::GeneratePipeline {
                request: Box::new(request),
                input,
                events,
                permit,
            })
            .await
            .is_err()
        {
            crate::metrics::generation_queue_cancelled();
            return Err(GenerateSubmitError::DriverStopped);
        }
        Ok(rx)
    }

    /// Render images on the engine thread that owns the pipeline.
    ///
    /// Diffusion is a single long synchronous call rather than a token stream,
    /// so this is a plain request/response command instead of an event channel.
    pub(crate) async fn render_images(
        &self,
        pipeline_dir: PathBuf,
        request: TextToImageRequest,
    ) -> Result<anyhow::Result<Vec<RenderedImage>>, GenerateSubmitError> {
        let _permit = self
            .generation_capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| GenerateSubmitError::Overloaded)?;
        let (reply, rx) = tokio::sync::oneshot::channel();
        if self
            .commands
            .send(DriverCommand::RenderImages {
                pipeline_dir,
                request: Box::new(request),
                reply,
            })
            .await
            .is_err()
        {
            return Err(GenerateSubmitError::DriverStopped);
        }
        rx.await.map_err(|_| GenerateSubmitError::DriverStopped)
    }

    /// Synthesize speech on the engine thread that owns the pipeline.
    ///
    /// Like image rendering, this is one long synchronous call rather than a
    /// token stream, so it is a plain request/response command.
    pub(crate) async fn synthesize_speech(
        &self,
        tokenizer: Arc<Tokenizer>,
        request: TextToAudioRequest,
    ) -> Result<anyhow::Result<SynthesizedAudio>, GenerateSubmitError> {
        let _permit = self
            .generation_capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| GenerateSubmitError::Overloaded)?;
        let (reply, rx) = tokio::sync::oneshot::channel();
        if self
            .commands
            .send(DriverCommand::SynthesizeSpeech {
                tokenizer,
                request: Box::new(request),
                reply,
            })
            .await
            .is_err()
        {
            return Err(GenerateSubmitError::DriverStopped);
        }
        rx.await.map_err(|_| GenerateSubmitError::DriverStopped)
    }

    pub(crate) async fn generate_fim(
        &self,
        prefix: String,
        suffix: String,
        fim_config: FimConfig,
        options: GenerateOptions,
    ) -> Result<mpsc::Receiver<DriverEvent>, GenerateSubmitError> {
        let permit = self
            .generation_capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| GenerateSubmitError::Overloaded)?;
        let (events, rx) = mpsc::channel(DRIVER_OUTPUT_BUFFER);
        crate::metrics::generation_queued();
        if self
            .commands
            .send(DriverCommand::GenerateFim {
                prefix,
                suffix,
                fim_config,
                options: Box::new(options),
                events,
                permit,
            })
            .await
            .is_err()
        {
            crate::metrics::generation_queue_cancelled();
            return Err(GenerateSubmitError::DriverStopped);
        }
        Ok(rx)
    }

    pub(crate) async fn embed(
        &self,
        input_ids: Vec<TokenId>,
        options: EmbeddingOptions,
    ) -> anyhow::Result<Vec<f32>> {
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
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.commands
            .send(DriverCommand::ResourceSnapshot(reply))
            .await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("engine driver stopped"))?
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
) {
    let mut engine = match owner.0 {
        EngineBackend::Single(engine) => *engine,
        EngineBackend::Pipeline(mut pipeline) => {
            run_pipeline_driver(&mut pipeline, rx);
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
        );
    } else {
        tracing::info!("continuous batch driver disabled; using per-request engine path");
        run_fallback_engine_driver(&mut engine, rx);
    }
}

fn run_pipeline_driver(engine: &mut PipelineEngine, mut rx: mpsc::Receiver<DriverCommand>) {
    while let Some(command) = rx.blocking_recv() {
        match command {
            DriverCommand::GeneratePipeline {
                request,
                input,
                events,
                permit,
            } => run_pipeline_generation(engine, *request, input, events, permit),
            DriverCommand::RenderImages {
                pipeline_dir,
                request,
                reply,
            } => {
                let _ = reply.send(onnx_genai::text_to_image::render(
                    &pipeline_dir,
                    engine,
                    &request,
                ));
            }
            DriverCommand::SynthesizeSpeech {
                tokenizer,
                request,
                reply,
            } => {
                let _ = reply.send(onnx_genai::text_to_audio::synthesize(
                    engine, &tokenizer, &request,
                ));
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
            DriverCommand::Generate { events, .. } | DriverCommand::GenerateFim { events, .. } => {
                crate::metrics::generation_queue_cancelled();
                let _ = events.try_send(DriverEvent::Error(
                    "invalid generation route for pipeline model".to_string(),
                ));
            }
            DriverCommand::Embed { reply, .. } => {
                let _ = reply.send(Err(anyhow::anyhow!(
                    "embeddings are not supported by pipeline models"
                )));
            }
            DriverCommand::ResourceSnapshot(reply) => {
                let _ = reply.send(Err(anyhow::anyhow!(
                    "resource governor is not available for pipeline models"
                )));
            }
            DriverCommand::SetVramLimit { reply, .. } => {
                let _ = reply.send(Err(anyhow::anyhow!(
                    "resource governor is not available for pipeline models"
                )));
            }
        }
    }
}

fn run_fallback_engine_driver(engine: &mut Engine, mut rx: mpsc::Receiver<DriverCommand>) {
    while let Some(command) = rx.blocking_recv() {
        handle_driver_command(engine, command);
    }
}

fn run_static_engine_driver(
    engine: &mut Engine,
    mut rx: mpsc::Receiver<DriverCommand>,
    max_batch: usize,
    max_queue_depth: usize,
    generation_capacity: &Semaphore,
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
                    PendingGeneration {
                        request: *request,
                        events,
                        permit,
                    },
                );
            }
            command => handle_driver_command(engine, command),
        }
    }
}

fn run_static_batch_until_idle(
    engine: &mut Engine,
    rx: &mut mpsc::Receiver<DriverCommand>,
    deferred: &mut VecDeque<DriverCommand>,
    max_batch: usize,
    admission: MicrobatchAdmission<'_>,
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
                    events,
                    permit,
                }) => {
                    initial.push(PendingGeneration {
                        request: *request,
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
                    events,
                    permit,
                }) => {
                    initial.push(PendingGeneration {
                        request: *request,
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
                let _ = pending.events.try_send(DriverEvent::Error(format!(
                    "continuous batch setup failed: {err}"
                )));
            }
            return;
        }
    };
    let mut routes: HashMap<usize, DriverRoute> = HashMap::new();
    let mut abandoned = HashMap::new();
    for pending in initial {
        submit_to_continuous_manager(
            &mut manager,
            &mut routes,
            &mut abandoned,
            pending.request,
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
                    events,
                    permit,
                }) => {
                    submit_to_continuous_manager(
                        &mut manager,
                        &mut routes,
                        &mut abandoned,
                        *request,
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
                    events,
                    permit,
                } => {
                    if routes.len() + abandoned.len() < manager.max_batch() {
                        submit_to_continuous_manager(
                            &mut manager,
                            &mut routes,
                            &mut abandoned,
                            *request,
                            events,
                            permit,
                        );
                    } else {
                        deferred.push_back(DriverCommand::Generate {
                            session_id: None,
                            request,
                            events,
                            permit,
                        });
                    }
                }
                // MUST stay above the catch-all: anything this returns `None`
                // for has been answered, and anything it hands back is parked
                // until the batch drains.
                command => {
                    if let Some(deferred_command) = handle_or_defer_during_batch(engine, command) {
                        deferred.push_back(deferred_command);
                    }
                }
            }
        }

        if let Err(err) = manager.step() {
            let message = format!("continuous batch generation failed: {err}");
            for (_, route) in routes.drain() {
                let _ = route.events.try_send(DriverEvent::Error(message.clone()));
            }
            break;
        }
        route_continuous_events(manager.poll(), &mut routes, &mut abandoned);
        if manager.is_idle() {
            break;
        }
    }
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
                    | DriverCommand::GenerateFim { .. }
                    | DriverCommand::RenderImages { .. }
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
    engine: &Engine,
    command: DriverCommand,
) -> Option<DriverCommand> {
    match command {
        DriverCommand::ResourceSnapshot(reply) => {
            let _ = reply.send(Ok(engine.resource_snapshot()));
            None
        }
        other => Some(other),
    }
}

fn submit_to_continuous_manager(
    manager: &mut ContinuousBatchManager<'_>,
    routes: &mut HashMap<usize, DriverRoute>,
    abandoned: &mut HashMap<usize, DriverRoute>,
    request: GenerateRequest,
    events: mpsc::Sender<DriverEvent>,
    permit: OwnedSemaphorePermit,
) {
    match manager.submit(request) {
        Ok(handle) => {
            routes.insert(
                handle.id,
                DriverRoute {
                    events,
                    _permit: permit,
                    metrics: GenerationMetrics::start(),
                },
            );
            route_continuous_events(manager.poll(), routes, abandoned);
        }
        Err(err) => {
            crate::metrics::generation_queue_cancelled();
            let _ = events.try_send(DriverEvent::Error(err.to_string()));
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
                // A slow or disconnected consumer loses its route immediately. The
                // driver never waits for output capacity; it keeps stepping every
                // other row while the manager retires the abandoned row.
                let delivery_failed = if let Some(route) = routes.get_mut(&handle.id) {
                    route.metrics.token();
                    route.events.try_send(DriverEvent::Token(token)).is_err()
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
                    let _ = route.events.try_send(DriverEvent::Finished(result));
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
            events,
            permit,
        } => run_fallback_generation(engine, session_id, *request, events, permit),
        DriverCommand::GenerateFim {
            prefix,
            suffix,
            fim_config,
            options,
            events,
            permit,
        } => run_fim_generation(engine, prefix, suffix, fim_config, *options, events, permit),
        DriverCommand::GeneratePipeline { events, .. } => {
            crate::metrics::generation_queue_cancelled();
            let _ = events.try_send(DriverEvent::Error(
                "invalid pipeline generation route for single model".to_string(),
            ));
        }
        DriverCommand::Embed {
            input_ids,
            options,
            reply,
        } => {
            let _ = reply.send(engine.embed_with_options(&input_ids, options));
        }
        DriverCommand::RenderImages { reply, .. } => {
            let _ = reply.send(Err(anyhow::anyhow!(
                "What: image generation was routed to a single-model engine. \
                 Why: only a declared diffusion pipeline can run a denoise loop. \
                 How: request a model whose package declares `strategy.denoiser`."
            )));
        }
        DriverCommand::SynthesizeSpeech { reply, .. } => {
            let _ = reply.send(Err(anyhow::anyhow!(
                "What: speech synthesis was routed to a single-model engine. \
                 Why: only a declared pipeline can run a post-decode vocoder stage. \
                 How: request a model whose package declares a `run_on: final_only` waveform stage."
            )));
        }
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

fn run_pipeline_generation(
    engine: &mut PipelineEngine,
    request: GenerateRequest,
    input: Option<MultimodalInput>,
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
            let _ = events.try_send(DriverEvent::Error(format!("{error:#}")));
            return;
        }
    };
    let mut callback = |token: GenerateToken| -> anyhow::Result<()> {
        metrics.token();
        events
            .try_send(DriverEvent::Token(token))
            .context("stream receiver closed")
    };
    match engine.generate_with_callback(pipeline_request, Some(&mut callback)) {
        Ok(result) => {
            metrics.result(result.token_ids.len(), result.prefix_cache_hit_len);
            let _ = events.try_send(DriverEvent::Finished(result));
        }
        Err(err) => {
            let _ = events.try_send(DriverEvent::Error(err.to_string()));
        }
    }
}

fn run_fallback_generation(
    engine: &mut Engine,
    session_id: Option<SessionId>,
    request: GenerateRequest,
    events: mpsc::Sender<DriverEvent>,
    _permit: OwnedSemaphorePermit,
) {
    let mut metrics = GenerationMetrics::start();
    let mut callback = |token: GenerateToken| -> anyhow::Result<()> {
        metrics.token();
        events
            .try_send(DriverEvent::Token(token))
            .context("stream receiver closed")
    };
    let result = match session_id {
        Some(session_id) => {
            engine.generate_in_session_with_callback(session_id, request, Some(&mut callback))
        }
        None => engine.generate_with_callback(request, Some(&mut callback)),
    };
    match result {
        Ok(result) => {
            metrics.result(result.token_ids.len(), result.prefix_cache_hit_len);
            let _ = events.try_send(DriverEvent::Finished(result));
        }
        Err(err) => {
            let _ = events.try_send(DriverEvent::Error(err.to_string()));
        }
    }
}

fn run_fim_generation(
    engine: &mut Engine,
    prefix: String,
    suffix: String,
    fim_config: FimConfig,
    options: GenerateOptions,
    events: mpsc::Sender<DriverEvent>,
    _permit: OwnedSemaphorePermit,
) {
    let mut metrics = GenerationMetrics::start();
    match engine.generate_fim_with_config(prefix, suffix, options, &fim_config) {
        Ok(result) => {
            metrics.result(result.token_ids.len(), result.prefix_cache_hit_len);
            let _ = events.try_send(DriverEvent::Finished(result));
        }
        Err(err) => {
            let _ = events.try_send(DriverEvent::Error(err.to_string()));
        }
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;

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

    /// Every command that takes a `generation_capacity` permit must be counted.
    ///
    /// `in_flight` is derived from that semaphore, and this count is subtracted
    /// from it to estimate arrivals. Missing a permit-holding variant here
    /// understates the deferred total, inflates the estimate, and forces a lone
    /// request onto the slow path -- which is exactly what happened when only
    /// the three text-generation commands were counted.
    #[test]
    fn image_commands_are_counted_as_permit_holders() {
        let (image_reply, _image_rx) = tokio::sync::oneshot::channel();
        let mut deferred: VecDeque<DriverCommand> = VecDeque::new();
        deferred.push_back(DriverCommand::RenderImages {
            pipeline_dir: PathBuf::new(),
            request: Box::new(TextToImageRequest::default()),
            reply: image_reply,
        });

        assert_eq!(
            deferred_permit_holder_count(&deferred),
            1,
            "RenderImages holds a generation_capacity permit and must be \
             subtracted from the in-flight estimate"
        );
    }

    /// A deferred permit holder must not be mistaken for an incoming sibling.
    ///
    /// This is the end-to-end shape of the bug: one image request queued behind
    /// a lone generation used to make `expected_this_batch` exceed `collected`,
    /// which held the generation to the hard deadline and then ran it through
    /// the continuous-batch path by itself.
    #[test]
    fn a_deferred_image_request_does_not_force_a_lone_generation_onto_the_slow_path() {
        let (image_reply, _image_rx) = tokio::sync::oneshot::channel();
        let mut deferred: VecDeque<DriverCommand> = VecDeque::new();
        deferred.push_back(DriverCommand::RenderImages {
            pipeline_dir: PathBuf::new(),
            request: Box::new(TextToImageRequest::default()),
            reply: image_reply,
        });

        // Two permits are held: our lone generation, and the queued image.
        let in_flight = 2usize;
        let max_batch = 4usize;
        let expected_this_batch = in_flight
            .saturating_sub(deferred_permit_holder_count(&deferred))
            .min(max_batch);

        assert_eq!(
            expected_this_batch, 1,
            "the queued image is accounted for, not counted as an arrival"
        );
        assert_ne!(
            admission_step(1, expected_this_batch, 0, deferred.is_empty(), false),
            AdmissionStep::WaitForSibling,
            "a lone generation must not wait for a sibling that is really a \
             queued image request"
        );
    }
}
