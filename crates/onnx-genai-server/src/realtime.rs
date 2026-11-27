//! Worker-owned native realtime connection. WebSocket events are rendered by
//! the route; this layer carries typed quanta and cancellation only.
mod conditioning;
mod conversation;
mod events;
mod pcm;
mod settings;
mod tools;
pub(crate) mod websocket;
use crate::{
    driver::{DriverCommand, EngineDriver},
    worker::WorkerRealtimeGuard,
};
use anyhow::Context;
use onnx_genai_engine::{
    Engine, GenerationControl,
    realtime::{RealtimeInput, RealtimeOutput, RealtimeTokenBudget},
};
use onnx_genai_metadata::RealtimeMode;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::{OwnedSemaphorePermit, mpsc, oneshot};

/// Model-independent transport and prompt admission budgets.
#[derive(Debug, Clone)]
pub struct RealtimeLimits {
    pub message_bytes: std::num::NonZeroUsize,
    pub source_bytes: std::num::NonZeroUsize,
    pub native_samples: std::num::NonZeroUsize,
    pub audio_event_samples: std::num::NonZeroUsize,
    pub text_bytes: std::num::NonZeroUsize,
    pub prompt_bytes: std::num::NonZeroUsize,
    pub prompt_tokens: std::num::NonZeroUsize,
    pub template_operations: std::num::NonZeroU64,
    pub write_timeout: std::time::Duration,
}
impl Default for RealtimeLimits {
    fn default() -> Self {
        use std::num::{NonZeroU64, NonZeroUsize};
        Self {
            message_bytes: NonZeroUsize::new(1024 * 1024).unwrap(),
            source_bytes: NonZeroUsize::new(4 * 1024 * 1024).unwrap(),
            native_samples: NonZeroUsize::new(1024 * 1024).unwrap(),
            audio_event_samples: NonZeroUsize::new(65536).unwrap(),
            text_bytes: NonZeroUsize::new(4 * 1024 * 1024).unwrap(),
            prompt_bytes: NonZeroUsize::new(1024 * 1024).unwrap(),
            prompt_tokens: NonZeroUsize::new(8192).unwrap(),
            template_operations: NonZeroU64::new(1_000_000).unwrap(),
            write_timeout: std::time::Duration::from_secs(5),
        }
    }
}

pub(crate) enum RealtimeInputOwned {
    Audio(Vec<f32>),
    Text(Vec<u32>),
    Commit,
    Generate {
        tokens: Vec<u32>,
        constraint: Option<onnx_genai_engine::GenerateConstraint>,
    },
    Configure {
        mode: RealtimeMode,
        tokens: Vec<u32>,
    },
    Poll,
    Cancel,
}
impl RealtimeInputOwned {
    fn execute(
        &self,
        session: &mut onnx_genai_engine::realtime::RealtimeSession<'_>,
        budget: RealtimeTokenBudget,
        control: GenerationControl,
    ) -> anyhow::Result<RealtimeOutput> {
        let input = match self {
            Self::Configure { mode, tokens } => return session.configure(*mode, tokens, control),
            Self::Audio(samples) => RealtimeInput::Audio(samples),
            Self::Text(tokens) => RealtimeInput::Text(tokens),
            Self::Generate { tokens, constraint } => {
                session.set_constraint(constraint.clone())?;
                RealtimeInput::Generate(tokens)
            }
            Self::Commit => RealtimeInput::Commit,
            Self::Poll => RealtimeInput::Poll,
            Self::Cancel => RealtimeInput::Cancel,
        };
        session.step(input, budget, control)
    }
}

struct Step {
    input: RealtimeInputOwned,
    budget: RealtimeTokenBudget,
    reply: oneshot::Sender<anyhow::Result<RealtimeOutput>>,
}
enum ConnectionRequest {
    PrepareDecoder {
        preserved: std::collections::BTreeSet<u32>,
        max_bytes: std::num::NonZeroUsize,
        reply: oneshot::Sender<anyhow::Result<onnx_genai_engine::tokenization::TokenTextDecoder>>,
    },
    Quantum(Step),
    ArmConstraint {
        constraint: Option<onnx_genai_engine::GenerateConstraint>,
        reply: oneshot::Sender<anyhow::Result<()>>,
    },
}
struct ConnectionCancellation(Arc<AtomicBool>);
impl Drop for ConnectionCancellation {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// A single producer with one in-flight request and a bounded native mailbox.
/// Dropping it cancels a running quantum and closes the worker's receiver.
pub(crate) struct RealtimeConnection {
    steps: mpsc::Sender<ConnectionRequest>,
    _cancellation: ConnectionCancellation,
    budget: RealtimeTokenBudget,
}
impl RealtimeConnection {
    pub(crate) async fn prepare_decoder(
        &mut self,
        preserved: std::collections::BTreeSet<u32>,
        max_bytes: std::num::NonZeroUsize,
    ) -> anyhow::Result<onnx_genai_engine::tokenization::TokenTextDecoder> {
        let (reply, response) = oneshot::channel();
        self.steps
            .send(ConnectionRequest::PrepareDecoder {
                preserved,
                max_bytes,
                reply,
            })
            .await
            .context("native realtime owner closed before publication admission")?;
        response
            .await
            .context("native realtime owner closed while preparing publication decoder")?
    }
    /// Prepare a response-local constraint on the native owner without running
    /// a quantum or consuming input/usage. Generate retains its existing path.
    pub(crate) async fn arm_constraint(
        &mut self,
        constraint: Option<onnx_genai_engine::GenerateConstraint>,
    ) -> anyhow::Result<()> {
        let (reply, response) = oneshot::channel();
        self.steps
            .send(ConnectionRequest::ArmConstraint { constraint, reply })
            .await
            .context("native realtime connection closed before constraint admission")?;
        response
            .await
            .context("native realtime worker closed before arming response sampling")?
    }
    pub(crate) fn set_budget(&mut self, budget: RealtimeTokenBudget) {
        self.budget = budget;
    }
    pub(crate) async fn step(
        &mut self,
        input: RealtimeInputOwned,
    ) -> anyhow::Result<RealtimeOutput> {
        let (reply, response) = oneshot::channel();
        self.steps
            .send(ConnectionRequest::Quantum(Step {
                input,
                budget: self.budget,
                reply,
            }))
            .await
            .context("native realtime connection closed before quantum admission")?;
        response
            .await
            .context("native realtime worker closed before returning the quantum")?
    }
}

pub(crate) struct RealtimeCommand {
    mode: RealtimeMode,
    initial_tokens: Vec<u32>,
    ready: oneshot::Sender<anyhow::Result<RealtimeOutput>>,
    steps: mpsc::Receiver<ConnectionRequest>,
    cancelled: Arc<AtomicBool>,
    // Release admission before publishing the worker as idle. Fields drop in
    // declaration order, including initialization failures and disconnects.
    _admission: OwnedSemaphorePermit,
    runtime: tokio::runtime::Handle,
    guard: WorkerRealtimeGuard,
}
impl EngineDriver {
    pub(crate) async fn open_realtime(
        &self,
        mode: RealtimeMode,
        initial_tokens: Vec<u32>,
    ) -> anyhow::Result<(RealtimeConnection, RealtimeOutput)> {
        let contract = self.realtime.as_ref().context(
            "model does not declare a realtime quantum profile; choose a compatible model package",
        )?;
        anyhow::ensure!(
            contract.supports(mode),
            "model realtime profile '{}' does not support {mode:?}; select one of its declared modes",
            contract.profile()
        );
        let admission = self.generation_capacity.clone().try_acquire_owned()
            .context("realtime generation capacity is full; retry after a connection closes or increase configured capacity")?;
        let guard = self
            .workers
            .reserve_realtime()
            .context("reserving an idle native worker for realtime")?;
        let sender = self.workers.realtime_sender(&guard)?;
        let (ready, response) = oneshot::channel();
        let (steps, receiver) = mpsc::channel(1);
        let cancellation = ConnectionCancellation(Arc::new(AtomicBool::new(false)));
        sender
            .send(DriverCommand::Realtime(RealtimeCommand {
                mode,
                initial_tokens,
                ready,
                steps: receiver,
                cancelled: Arc::clone(&cancellation.0),
                guard,
                _admission: admission,
                runtime: tokio::runtime::Handle::current(),
            }))
            .await
            .map_err(|_| {
                anyhow::anyhow!("native realtime worker stopped before connection admission")
            })?;
        let initial = response
            .await
            .context("native realtime worker stopped during session initialization")??;
        Ok((
            RealtimeConnection {
                steps,
                _cancellation: cancellation,
                budget: RealtimeTokenBudget::Unlimited,
            },
            initial,
        ))
    }
}
impl RealtimeCommand {
    pub(crate) fn run(mut self, engine: &mut Engine) {
        let mut shutdown = self.guard.shutdown();
        let control = || {
            let cancelled = Arc::clone(&self.cancelled);
            let shutdown = shutdown.clone();
            GenerationControl::from_cancellation_probe(move || {
                cancelled.load(Ordering::Acquire) || *shutdown.borrow()
            })
        };
        if self.ready.is_closed() || *shutdown.borrow() {
            return;
        }
        let (mut session, initial) =
            match engine.open_realtime_session(self.mode, &self.initial_tokens, control()) {
                Ok(opened) => opened,
                Err(error) => {
                    let _ = self.ready.send(Err(error));
                    return;
                }
            };
        if self.ready.send(Ok(initial)).is_err() {
            return;
        }
        loop {
            let next = self.runtime.block_on(async {
                if *shutdown.borrow() {
                    return None;
                }
                tokio::select! {
                    biased;
                    _ = shutdown.changed() => None,
                    step = self.steps.recv() => step,
                }
            });
            let step = match next {
                None => break,
                Some(ConnectionRequest::PrepareDecoder {
                    preserved,
                    max_bytes,
                    reply,
                }) => {
                    if !reply.is_closed()
                        && reply
                            .send(session.prepare_text_decoder(&preserved, max_bytes))
                            .is_err()
                    {
                        break;
                    }
                    continue;
                }
                Some(ConnectionRequest::ArmConstraint { constraint, reply }) => {
                    // Compilation is transactional and does not execute the
                    // workflow. Admission rejection need not poison native KV.
                    if !reply.is_closed() && reply.send(session.set_constraint(constraint)).is_err()
                    {
                        break;
                    }
                    continue;
                }
                Some(ConnectionRequest::Quantum(step)) => step,
            };
            if step.reply.is_closed() {
                continue;
            }
            let cancelled = Arc::clone(&self.cancelled);
            let shutdown_probe = shutdown.clone();
            let control = GenerationControl::from_cancellation_probe(move || {
                cancelled.load(Ordering::Acquire) || *shutdown_probe.borrow()
            });
            let result = step.input.execute(&mut session, step.budget, control);
            let failed = result.is_err();
            if step.reply.send(result).is_err() || failed {
                break;
            }
        }
        // Native session drops before the worker reservation or package owner.
        drop(session);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn driver() -> EngineDriver {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime");
        EngineDriver::start(
            move || {
                Ok((
                    Engine::from_dir(&root, onnx_genai_engine::EngineConfig::default())?,
                    (),
                ))
            },
            1,
            8,
        )
        .unwrap()
        .0
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pcm_frames_execute_and_publish_native_audio_before_input_commit() -> anyhow::Result<()>
    {
        use base64::Engine as _;
        use std::num::{NonZeroU32, NonZeroUsize};
        let driver = driver();
        let spec = driver.realtime.as_ref().unwrap().spec();
        let frame_samples = spec.input_frame_samples;
        let native_rate = NonZeroU32::new(spec.input_sample_rate_hz).unwrap();
        let mut pcm = super::pcm::PcmInput::new(
            native_rate,
            native_rate,
            NonZeroUsize::new(frame_samples).unwrap(),
            NonZeroUsize::new(8192).unwrap(),
            NonZeroUsize::new(8192).unwrap(),
        )?;
        let (mut native, _) = driver.open_realtime(RealtimeMode::Duplex, vec![]).await?;
        let (mut response, _) = super::events::ResponseEvents::new(
            "resp_native".into(),
            "item_native".into(),
            serde_json::json!(["audio"]),
            NonZeroUsize::new(4096).unwrap(),
            NonZeroUsize::new(frame_samples).unwrap(),
        );
        let first = frame_samples / 2;
        for samples in [first, frame_samples - first] {
            pcm.append(&base64::engine::general_purpose::STANDARD.encode(vec![0; samples * 2]))?;
        }
        let frame = pcm
            .next_frame()?
            .context("complete native PCM frame must be ready before commit")?;
        let output = native.step(RealtimeInputOwned::Audio(frame)).await?;
        response.usage(output.output_tokens)?;
        let events = response.audio(&output.audio)?;
        let audio = events
            .iter()
            .find(|event| event["type"] == "response.output_audio.delta")
            .context("actual native quantum emitted no audio delta")?;
        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(audio["delta"].as_str().unwrap())?
                .len(),
            frame_samples * 2
        );
        pcm.commit()?;
        assert!(pcm.next_frame()?.is_none());
        assert!(native.step(RealtimeInputOwned::Cancel).await?.done);
        assert_eq!(
            response
                .finish(events::ResponseEnd::Cancelled)
                .last()
                .unwrap()["response"]["status"],
            "cancelled"
        );
        drop(native);
        tokio::task::spawn_blocking(move || driver.shutdown()).await?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_connection_owns_worker_until_disconnect_cleanup_finishes() -> anyhow::Result<()>
    {
        let driver = driver();
        let (mut connection, initial) = driver.open_realtime(RealtimeMode::Duplex, vec![4]).await?;
        assert_eq!(initial.tokens, [4]);
        assert_eq!(driver.worker_statuses()[0].worker.active_turns, 1);
        assert!(
            driver
                .open_realtime(RealtimeMode::Duplex, vec![])
                .await
                .is_err()
        );
        assert_eq!(
            connection
                .step(RealtimeInputOwned::Audio(vec![0.1, 0.2]))
                .await?
                .audio,
            [0.1, 0.2]
        );
        assert!(
            (connection
                .step(RealtimeInputOwned::Audio(vec![0.2]))
                .await?
                .audio[0]
                - 0.5)
                .abs()
                < 1e-6
        );
        assert_eq!(
            connection
                .step(RealtimeInputOwned::Text(vec![5, 6]))
                .await?
                .tokens,
            [5, 6]
        );
        assert!(!connection.step(RealtimeInputOwned::Commit).await?.done);
        assert!(
            connection
                .step(RealtimeInputOwned::Generate {
                    tokens: vec![],
                    constraint: None
                })
                .await?
                .done
        );
        assert!(!connection.step(RealtimeInputOwned::Poll).await?.done);
        assert!(connection.step(RealtimeInputOwned::Cancel).await?.done);
        drop(connection);
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while driver.worker_statuses()[0].worker.active_turns != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .context("native disconnect did not release its worker")?;
        assert_eq!(
            driver.generation_capacity.available_permits(),
            driver.generation_capacity_size as usize
        );
        let (mut fresh, _) = driver
            .open_realtime(RealtimeMode::Transcription, vec![])
            .await?;
        assert_eq!(
            fresh
                .step(RealtimeInputOwned::Audio(vec![0.1]))
                .await?
                .audio,
            [0.1]
        );
        drop(fresh);
        tokio::task::spawn_blocking(move || driver.shutdown()).await?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn model_shutdown_cancels_a_live_native_connection_without_waiting_for_client()
    -> anyhow::Result<()> {
        let driver = driver();
        let (mut connection, _) = driver.open_realtime(RealtimeMode::Duplex, vec![]).await?;
        let shutdown_driver = driver.clone();
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            tokio::task::spawn_blocking(move || shutdown_driver.shutdown()),
        )
        .await
        .context("model shutdown waited for the realtime client")??;
        assert!(connection.step(RealtimeInputOwned::Poll).await.is_err());
        assert_eq!(
            driver.generation_capacity.available_permits(),
            driver.generation_capacity_size as usize
        );
        Ok(())
    }
}
