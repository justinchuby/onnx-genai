//! Status of an admitted physical Native executor and owner-thread actions.
use crate::slot_storage::{SlotFilename, SlotStorage};
use anyhow::Context;
use serde_json::Value;
#[cfg(feature = "native-backend")]
use serde_json::json;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy)]
pub(crate) struct SlotSnapshot {
    pub(crate) is_processing: bool,
    pub(crate) n_past: usize,
    pub(crate) n_ctx: Option<usize>,
    pub(crate) state_files: bool,
}
#[derive(Debug)]
pub(crate) struct NativeSlot {
    snapshot: Mutex<SlotSnapshot>,
}
impl NativeSlot {
    #[cfg(feature = "native-backend")]
    pub(crate) fn from_engine(engine: &onnx_genai::Engine) -> Option<Arc<Self>> {
        let state = engine.native_slot_state()?;
        Some(Arc::new(Self {
            snapshot: Mutex::new(SlotSnapshot {
                is_processing: false,
                n_past: state.materialized_tokens,
                n_ctx: state.max_context,
                state_files: state.state_files,
            }),
        }))
    }
    pub(crate) fn snapshot(&self) -> anyhow::Result<SlotSnapshot> {
        self.snapshot.lock().map(|s|*s).map_err(|_|anyhow::anyhow!("Native slot status lock is poisoned; reload the model to rebuild its executor status"))
    }
    pub(crate) fn begin(self: &Arc<Self>) -> SlotExecution {
        if let Ok(mut state) = self.snapshot.lock() {
            state.is_processing = true;
        }
        SlotExecution {
            slot: Arc::clone(self),
            completed: false,
        }
    }
    fn finish(&self, engine: &onnx_genai::Engine) {
        if let Ok(mut state) = self.snapshot.lock() {
            #[cfg(feature = "native-backend")]
            if let Some(actual) = engine.native_slot_state() {
                state.n_past = actual.materialized_tokens;
                state.n_ctx = actual.max_context;
                state.state_files = actual.state_files;
            }
            #[cfg(not(feature = "native-backend"))]
            let _ = engine;
            state.is_processing = false;
        }
    }
}
pub(crate) struct SlotExecution {
    slot: Arc<NativeSlot>,
    completed: bool,
}
impl SlotExecution {
    pub(crate) fn finish(&mut self, engine: &onnx_genai::Engine) {
        if !self.completed {
            self.slot.finish(engine);
            self.completed = true;
        }
    }
}
impl Drop for SlotExecution {
    fn drop(&mut self) {
        if !self.completed
            && let Ok(mut state) = self.slot.snapshot.lock()
        {
            state.is_processing = false;
        }
    }
}

// The shared command schema remains available in reference-only builds.
#[cfg_attr(not(feature = "native-backend"), allow(dead_code))]
pub(crate) enum SlotOperation {
    Save {
        storage: Arc<SlotStorage>,
        filename: SlotFilename,
    },
    Restore {
        storage: Arc<SlotStorage>,
        filename: SlotFilename,
    },
    Erase,
}
impl SlotOperation {
    pub(crate) fn run(self, engine: &mut onnx_genai::Engine) -> anyhow::Result<Value> {
        #[cfg(feature = "native-backend")]
        {
            let start = std::time::Instant::now();
            match self {
                Self::Save { storage, filename } => {
                    let limits = onnx_genai_engine::NativeStateFileLimits {
                        max_bytes: storage.limits.file_bytes,
                        ..Default::default()
                    };
                    let report = storage.save(&filename, |writer| {
                        engine.write_native_slot_state(writer, limits)
                    })?;
                    Ok(
                        json!({"id_slot":0,"filename":filename.as_str(),"n_saved":report.logical_tokens,"n_written":report.bytes,"timings":{"save_ms":start.elapsed().as_secs_f64()*1000.0}}),
                    )
                }
                Self::Restore { storage, filename } => {
                    let limits = onnx_genai_engine::NativeStateFileLimits {
                        max_bytes: storage.limits.file_bytes,
                        ..Default::default()
                    };
                    let report = storage.restore(&filename, |reader| {
                        engine.restore_native_slot_state(reader, limits)
                    })?;
                    Ok(
                        json!({"id_slot":0,"filename":filename.as_str(),"n_restored":report.logical_tokens,"n_read":report.bytes,"timings":{"restore_ms":start.elapsed().as_secs_f64()*1000.0}}),
                    )
                }
                Self::Erase => {
                    Ok(json!({"id_slot":0,"n_erased":engine.erase_native_slot_state()?}))
                }
            }
        }
        #[cfg(not(feature = "native-backend"))]
        {
            let _ = (self, engine);
            anyhow::bail!("slot actions require the Native backend; rebuild with native-backend")
        }
    }
}

impl crate::driver::EngineDriver {
    pub(crate) fn slot_status(&self) -> anyhow::Result<SlotSnapshot> {
        self.workers
            .primary_sender()
            .context("Native slot owner is unavailable; reload the model")?;
        self.native_slot.as_ref().context("this executor has no admitted serial Native slot; use a single-sequence Native decoder")?.snapshot()
    }
    pub(crate) async fn slot_operation(&self, operation: SlotOperation) -> anyhow::Result<Value> {
        self.slot_status()?;
        let permit = Arc::clone(&self.generation_capacity)
            .try_acquire_owned()
            .context("slot action admission is full; retry after current engine requests finish")?;
        let sender = self
            .workers
            .primary_sender()
            .context("Native slot owner is unavailable; reload the model")?;
        let (response, reply) = tokio::sync::oneshot::channel();
        sender
            .send(crate::driver::DriverCommand::Slot {
                operation,
                response,
                _permit: permit,
            })
            .await
            .map_err(|_| anyhow::anyhow!("Native slot command channel closed; reload the model"))?;
        reply
            .await
            .context("Native slot worker stopped before replying; reload the model")?
    }
}
