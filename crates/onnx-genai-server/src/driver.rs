use std::{collections::HashMap, path::PathBuf, sync::Arc, thread};

use anyhow::Context;
use onnx_genai::text_to_audio::{SynthesizedAudio, TextToAudioRequest};
use onnx_genai::text_to_image::{RenderedImage, TextToImageRequest};
use onnx_genai::{
    Engine, GenerateOptions, GenerateRequest, GenerateResult, GenerateToken, SessionId, TokenId,
};
use onnx_genai_engine::{
    ContinuousBatchEvent, ContinuousBatchManager, EmbeddingOptions, EngineGovernorError,
    EngineResourceGovernor, FimConfig, GovernorSnapshot, KvNotApplicable, KvTelemetry,
    PipelineEngine, PipelineGenerateRequest, ResourceLimit,
};
use onnx_genai_ort::Tokenizer;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};

use crate::batch_telemetry::BatchTelemetry;

use crate::metrics::GenerationMetrics;
use crate::multimodal::MultimodalInput;

const DRIVER_OUTPUT_BUFFER: usize = 16;

#[derive(Clone)]
pub(crate) struct EngineDriver {
    pub(crate) commands: mpsc::Sender<DriverCommand>,
    pub(crate) generation_capacity: Arc<Semaphore>,
    /// Shared with the driver thread's engine. Read directly by HTTP handlers,
    /// never through the command channel.
    kv_telemetry: Arc<KvTelemetry>,
    /// Shared with the driver thread's engine, for the same reason as
    /// `kv_telemetry`: `/v1/resources` must answer while the driver thread is
    /// blocked inside a generation. `None` for pipeline engines, which have no
    /// governor to share.
    governor: Option<Arc<EngineResourceGovernor>>,
    /// Occupancy of the batch the decoder is stepping, published by the driver
    /// thread. Shared for the same reason as `kv_telemetry`: the status route
    /// must answer while that thread is blocked inside a generation.
    batch_telemetry: Arc<BatchTelemetry>,
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

// SAFETY: The engine is moved exactly once into the dedicated driver thread.
// All ORT runners, sessions, KV state, and the continuous batch manager stay
// owned by that thread and are accessed only by processing channel commands.
unsafe impl Send for EngineOwner {}

impl EngineDriver {
    pub(crate) fn start(engine: Engine, max_batch: usize, max_queue_depth: usize) -> Self {
        let (commands, rx) = mpsc::channel(max_queue_depth);
        let generation_capacity = Arc::new(Semaphore::new(max_queue_depth));
        // Taken BEFORE the engine moves to the driver thread: this is the only
        // point at which the server can still touch it.
        let governor = Some(engine.governor_handle());
        // Decided BEFORE the engine moves, on the caller's thread, for the same
        // reason the governor handle is taken here: it is the last point at
        // which the server can still ask the engine anything. Deciding it here
        // means the width published below is the width that will actually run,
        // so there is one publisher and one decision rather than a fast wrong
        // answer racing a slow right one.
        let continuous_batch_supported = engine.continuous_batch_manager(max_batch).is_ok();
        let owner = EngineOwner(EngineBackend::Single(Box::new(engine)));
        let kv_telemetry = Arc::new(KvTelemetry::default());
        let driver_telemetry = Arc::clone(&kv_telemetry);
        let batch_telemetry = Arc::new(BatchTelemetry::default());
        // `max_batch` is the configured CEILING, not the width this driver will
        // run at: the per-request fallback is one row wide however large the
        // ceiling is. Publishing the ceiling here while the driver thread
        // published the truth made `/v1/status` return 4 or 1 depending on
        // scheduling -- a capacity published before the decode path is known is
        // a guess wearing a measurement's name.
        //
        // Publishing synchronously is still right: an absent field costs every
        // reader a `pending` frame. What was wrong was publishing a value we
        // had not yet earned. Now the path is known first, so the field is both
        // immediate and true.
        let published_capacity = if continuous_batch_supported {
            max_batch
        } else {
            1
        };
        batch_telemetry.publish(0, 0, published_capacity);
        let driver_batch = Arc::clone(&batch_telemetry);
        thread::Builder::new()
            .name("onnx-genai-batch-driver".to_string())
            .spawn(move || {
                run_engine_driver(
                    owner,
                    rx,
                    max_batch,
                    continuous_batch_supported,
                    driver_telemetry,
                    driver_batch,
                )
            })
            .expect("failed to spawn onnx-genai engine driver");
        Self {
            commands,
            generation_capacity,
            kv_telemetry,
            governor,
            batch_telemetry,
        }
    }

    /// An [`EngineDriver`] with no driver thread behind it, for tests that
    /// need a `ModelHandle` without loading a model. Its telemetry reports
    /// not-applicable, because there is no pool.
    #[cfg(test)]
    pub(crate) fn detached_for_test(commands: mpsc::Sender<DriverCommand>) -> Self {
        Self {
            commands,
            generation_capacity: Arc::new(Semaphore::new(0)),
            kv_telemetry: Arc::new(KvTelemetry::default()),
            governor: None,
            // No driver thread, so nothing ever publishes: the status route
            // omits the batch fields, which is the truth about this handle.
            batch_telemetry: Arc::new(BatchTelemetry::default()),
        }
    }

    /// An [`EngineDriver`] holding a real engine's governor handle but with NO
    /// driver thread behind it, so nothing ever drains the command channel.
    ///
    /// This is what a driver thread blocked inside `generate` looks like from
    /// the outside, made deterministic: any request that needs the thread waits
    /// forever. A test using [`Self::start`] cannot model this, because that
    /// spawns a live thread which promptly drains whatever it is sent.
    #[cfg(test)]
    pub(crate) fn detached_with_governor_for_test(
        engine: &Engine,
        commands: mpsc::Sender<DriverCommand>,
    ) -> Self {
        Self {
            commands,
            generation_capacity: Arc::new(Semaphore::new(0)),
            kv_telemetry: Arc::new(KvTelemetry::default()),
            governor: Some(engine.governor_handle()),
            batch_telemetry: Arc::new(BatchTelemetry::default()),
        }
    }

    /// Live paged-KV counters, readable while a generation is running.
    ///
    /// Deliberately not routed through [`DriverCommand`]: both driver loops run
    /// generation inline, so a command cannot be serviced mid-generation, and
    /// paged-KV behaviour is only interesting *during* one.
    pub(crate) fn kv_telemetry(&self) -> &Arc<KvTelemetry> {
        &self.kv_telemetry
    }

    /// Live batch occupancy, or `None` until a driver loop has published one.
    pub(crate) fn batch_telemetry(&self) -> &Arc<BatchTelemetry> {
        &self.batch_telemetry
    }

    pub(crate) fn start_pipeline(engine: PipelineEngine, max_queue_depth: usize) -> Self {
        let (commands, rx) = mpsc::channel(max_queue_depth);
        let generation_capacity = Arc::new(Semaphore::new(max_queue_depth));
        let owner = EngineOwner(EngineBackend::Pipeline(Box::new(engine)));
        let kv_telemetry = Arc::new(KvTelemetry::default());
        let driver_telemetry = Arc::clone(&kv_telemetry);
        let batch_telemetry = Arc::new(BatchTelemetry::default());
        // A pipeline runs one generation at a time, so its batch is one row
        // wide BY CONSTRUCTION -- nothing needs to be asked of the engine and
        // no decode path has to be chosen first. Published synchronously, on
        // the caller's thread, so the field is never absent, and published only
        // here so there is exactly one publisher of this pool's width.
        batch_telemetry.publish(0, 0, 1);
        let driver_batch = Arc::clone(&batch_telemetry);
        thread::Builder::new()
            .name("onnx-genai-pipeline-driver".to_string())
            .spawn(move || run_engine_driver(owner, rx, 1, false, driver_telemetry, driver_batch))
            .expect("failed to spawn onnx-genai pipeline driver");
        Self {
            commands,
            generation_capacity,
            kv_telemetry,
            // Pipeline engines expose no resource governor, so `/v1/resources`
            // keeps the command path and its honest "not available" error.
            governor: None,
            batch_telemetry,
        }
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

    /// A resource snapshot, read WITHOUT the driver thread when possible.
    ///
    /// `/v1/resources` is observability: it exists to report on a server that
    /// is busy, so it must not queue behind the work it describes. The fallback
    /// driver runs generation INLINE on its command loop, so a `DriverCommand`
    /// is only serviced between generations -- and under sustained load, with
    /// arrivals overlapping departures, those gaps close and the endpoint stops
    /// answering altogether. Measured before this handle existed: 5 of 6 polls
    /// timed out at 10s, worst 7.9s, while `/v1/status` answered 1046 times
    /// with a 2.3ms median in the same window.
    ///
    /// The governor's accessors take `&self`, so a shared handle is both
    /// always-available and always-LIVE. That is why this is a shared read and
    /// not a published mirror: a mirror refreshed between generations is
    /// staleest exactly when the server is busiest, which is the condition this
    /// endpoint exists for.
    ///
    /// Pipeline engines expose no governor, so they keep the command path and
    /// its honest "not available" error rather than inventing a snapshot.
    pub(crate) async fn resource_snapshot(&self) -> anyhow::Result<GovernorSnapshot> {
        if let Some(governor) = &self.governor {
            return Ok(governor.snapshot());
        }
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

/// What to publish about a paged pool once the decode path is known.
///
/// Extracted so the rule has exactly one home. Both driver arms route through
/// it, and it is total over its two inputs, so there is no combination that
/// falls through to a default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvApplicabilityDecision {
    Applicable,
    NotApplicable(KvNotApplicable),
}

impl KvApplicabilityDecision {
    fn apply(self, telemetry: &KvTelemetry) {
        match self {
            Self::Applicable => telemetry.set_applicable(),
            Self::NotApplicable(reason) => telemetry.set_not_applicable(reason),
        }
    }
}

/// Decide whether a paged pool is the mechanism actually in play.
///
/// Applicability is a CONJUNCTION of two independent facts, and the bug this
/// replaces used the second alone as a proxy for both:
///
///   1. `paged` -- the pool can hold KV tensors at all. Reported by the attach,
///      never inferred.
///   2. `!continuous_batch_supported` -- the paged path is the one that will
///      run. Continuous batching and paged KV are mutually exclusive.
///
/// **Neither implies the other, in either direction.**
///
/// Fact 2 alone was the shipped bug. `continuous_batch_manager` also fails when
/// `max_batch` is zero, when the ORT decoder session is absent, and when the
/// batched session cannot be constructed -- none of which say anything about
/// KV -- so `!continuous_batch_supported` converts any of those into a positive
/// claim that paged KV is in play. That is a capability derived from the
/// absence of a different capability, which is never sound.
///
/// Fact 1 alone is just as wrong the other way: on the batching path the pool
/// is fully tensor-backed, so `paged` is `true` while the decoder never
/// consults it. Reporting that pool as applicable would render a structure that
/// can never move as one that is merely idle.
///
/// `paged` is checked first because it is the stronger claim: a cache that
/// cannot page is not paging *regardless* of which decode path runs, so
/// naming the decode path in that case would explain the wrong mechanism.
pub(crate) fn classify_kv_applicability(
    paged: bool,
    continuous_batch_supported: bool,
) -> KvApplicabilityDecision {
    if !paged {
        KvApplicabilityDecision::NotApplicable(KvNotApplicable::CacheCannotPage)
    } else if continuous_batch_supported {
        KvApplicabilityDecision::NotApplicable(KvNotApplicable::ContinuousBatching)
    } else {
        KvApplicabilityDecision::Applicable
    }
}

fn run_engine_driver(
    owner: EngineOwner,
    rx: mpsc::Receiver<DriverCommand>,
    max_batch: usize,
    // Decided by the caller, before the engine moved, and already published as
    // the batch width. Passed rather than recomputed: two evaluations of the
    // same predicate is a divergence waiting to happen, and the published
    // capacity would be the thing that diverged.
    continuous_batch_supported: bool,
    kv_telemetry: Arc<KvTelemetry>,
    batch_telemetry: Arc<BatchTelemetry>,
) {
    let mut engine = match owner.0 {
        EngineBackend::Single(engine) => *engine,
        EngineBackend::Pipeline(mut pipeline) => {
            // A pipeline decoder pages only when its KV can be paged; the
            // accessor reports which, so we never claim applicability we
            // haven't checked. A pipeline never batches continuously, so the
            // second fact is a constant here rather than an inference.
            let paged = pipeline.attach_kv_telemetry(Arc::clone(&kv_telemetry));
            classify_kv_applicability(paged, false).apply(&kv_telemetry);
            run_pipeline_driver(&mut pipeline, rx);
            return;
        }
    };

    // Attach either way. The pool's capacity is a real number even when the
    // mechanism is inactive; what must not be guessed is whether it will move.
    let paged = engine.attach_kv_telemetry(Arc::clone(&kv_telemetry));
    classify_kv_applicability(paged, continuous_batch_supported).apply(&kv_telemetry);

    if continuous_batch_supported {
        tracing::info!(max_batch, "continuous batch driver enabled");
        run_static_engine_driver(&mut engine, rx, max_batch, &batch_telemetry);
    } else {
        tracing::info!("continuous batch driver disabled; using per-request engine path");
        run_fallback_engine_driver(&mut engine, rx, &batch_telemetry);
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

fn run_fallback_engine_driver(
    engine: &mut Engine,
    mut rx: mpsc::Receiver<DriverCommand>,
    batch_telemetry: &BatchTelemetry,
) {
    // This path runs generations inline, one at a time. Its capacity is not
    // `max_batch` -- no batch exists -- it is one. Publishing that is what
    // stops the status route pairing a count from this path with a width no
    // code here honours.
    batch_telemetry.publish(0, 0, 1);
    while let Some(command) = rx.blocking_recv() {
        let generating = matches!(command, DriverCommand::Generate { .. });
        if generating {
            batch_telemetry.publish(1, 0, 1);
        }
        handle_driver_command(engine, command);
        if generating {
            batch_telemetry.publish(0, 0, 1);
        }
    }
}

fn run_static_engine_driver(
    engine: &mut Engine,
    mut rx: mpsc::Receiver<DriverCommand>,
    max_batch: usize,
    batch_telemetry: &BatchTelemetry,
) {
    // The current ContinuousBatchManager API accepts GenerateRequest only.
    // A batch of known width exists the moment this loop starts, before any
    // request arrives. Publishing it here means a live node always carries
    // both terms -- an idle batch reports `0 of 4`, which is a measurement --
    // and only a driver with no thread behind it omits them.
    batch_telemetry.publish(0, 0, max_batch);

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
                    *request,
                    events,
                    permit,
                    batch_telemetry,
                );
            }
            command => handle_driver_command(engine, command),
        }
    }
}

/// Runs one static batch to completion.
///
/// `pub(crate)` so tests can drive the real loop rather than re-implementing
/// its intake: the AC31 hang lived in this function's queue handling, and a
/// test that calls the decision helper directly cannot observe it.
pub(crate) fn run_static_batch_until_idle(
    engine: &Engine,
    rx: &mut mpsc::Receiver<DriverCommand>,
    deferred: &mut std::collections::VecDeque<DriverCommand>,
    max_batch: usize,
    first_request: GenerateRequest,
    first_events: mpsc::Sender<DriverEvent>,
    first_permit: OwnedSemaphorePermit,
    batch_telemetry: &BatchTelemetry,
) {
    let mut manager = match engine.continuous_batch_manager(max_batch) {
        Ok(manager) => manager,
        Err(err) => {
            crate::metrics::generation_queue_cancelled();
            let _ = first_events.try_send(DriverEvent::Error(format!(
                "continuous batch setup failed: {err}"
            )));
            return;
        }
    };
    let mut routes: HashMap<usize, DriverRoute> = HashMap::new();
    let mut abandoned = HashMap::new();
    submit_to_continuous_manager(
        &mut manager,
        &mut routes,
        &mut abandoned,
        first_request,
        first_events,
        first_permit,
    );

    loop {
        // Commands the OUTER loop already drained out of `rx` live here, and
        // are invisible to `rx.try_recv()` below. Draining them first is what
        // makes concurrent arrivals join the *current* batch: without this,
        // every request the outer loop parked waits for this batch to go idle
        // and then runs as its own one-row batch -- continuous batching
        // silently degrades to strict serialisation under exactly the
        // concurrent load it exists to overlap. It also strands read-only
        // commands like `/v1/resources` behind every parked generation.
        for command in std::mem::take(deferred) {
            intake_during_batch(
                engine,
                &mut manager,
                &mut routes,
                &mut abandoned,
                deferred,
                command,
            );
        }
        while let Ok(command) = rx.try_recv() {
            intake_during_batch(
                engine,
                &mut manager,
                &mut routes,
                &mut abandoned,
                deferred,
                command,
            );
        }

        if let Err(err) = manager.step() {
            let message = format!("continuous batch generation failed: {err}");
            for (_, route) in routes.drain() {
                let _ = route.events.try_send(DriverEvent::Error(message.clone()));
            }
            break;
        }
        // Both terms come from `manager` at the same instant. `max_batch()`
        // is the length of the row array and `active_len()` counts occupied
        // slots of that same array, so the pair cannot exceed one.
        batch_telemetry.publish(
            manager.active_len(),
            manager.pending_len(),
            manager.max_batch(),
        );

        route_continuous_events(manager.poll(), &mut routes, &mut abandoned);
        if manager.is_idle() {
            break;
        }
    }

    // The batch is gone. Leaving the last reading in place would leave the
    // status route reporting rows that no longer exist, which reads as a
    // wedged server rather than an idle one.
    batch_telemetry.publish(0, 0, max_batch);
}

/// Takes one command into a running batch, from either source.
///
/// Extracted so the parked queue and the live channel cannot drift apart: the
/// AC31 hang was caused by these two intake paths being written separately and
/// only one of them learning to batch.
fn intake_during_batch(
    engine: &Engine,
    manager: &mut ContinuousBatchManager<'_>,
    routes: &mut HashMap<usize, DriverRoute>,
    abandoned: &mut HashMap<usize, DriverRoute>,
    deferred: &mut std::collections::VecDeque<DriverCommand>,
    command: DriverCommand,
) {
    match command {
        DriverCommand::Generate {
            session_id: None,
            request,
            events,
            permit,
        } => submit_to_continuous_manager(manager, routes, abandoned, *request, events, permit),
        // MUST stay above the catch-all: anything this returns `None` for has
        // been answered, and anything it hands back is parked until the batch
        // drains.
        command => {
            if let Some(deferred_command) = handle_or_defer_during_batch(engine, command) {
                deferred.push_back(deferred_command);
            }
        }
    }
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
/// Commands needing `&mut Engine` cannot be served while `ContinuousBatchManager`
/// holds its borrow, so they stay deferred by design.
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
