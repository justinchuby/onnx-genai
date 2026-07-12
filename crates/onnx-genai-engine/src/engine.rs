//! Main generation engine.

use onnx_genai_kv::{PagedKvCache, KvCacheOps, SequenceId};
use onnx_genai_metadata::InferenceMetadata;
use onnx_genai_scheduler::{Scheduler, SchedulerConfig, Priority};
use std::path::Path;

/// Engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Number of GPU pages for KV cache.
    pub num_gpu_pages: usize,
    /// Tokens per KV page.
    pub page_size: usize,
    /// Scheduler config.
    pub scheduler: SchedulerConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            num_gpu_pages: 1024,
            page_size: 16,
            scheduler: SchedulerConfig::default(),
        }
    }
}

/// The generation engine.
pub struct Engine {
    /// Model inference metadata.
    metadata: InferenceMetadata,
    /// KV cache manager.
    kv_cache: PagedKvCache,
    /// Batch scheduler.
    scheduler: Scheduler,
    // ORT session for the decoder model.
    // session: ort::Session,  // TODO: uncomment when ort is configured
    // Tokenizer.
    // tokenizer: tokenizers::Tokenizer,  // TODO: uncomment when ready
}

impl Engine {
    /// Load a model from a directory.
    pub fn from_dir(model_dir: &Path, config: EngineConfig) -> anyhow::Result<Self> {
        // Load metadata
        let metadata_path = model_dir.join("inference_metadata.yaml");
        let metadata = if metadata_path.exists() {
            onnx_genai_metadata::load_metadata(&metadata_path)
                .map_err(|e| anyhow::anyhow!("Failed to load metadata: {}", e))?
        } else {
            // Fall back to JSON
            let json_path = model_dir.join("inference_metadata.json");
            if json_path.exists() {
                onnx_genai_metadata::load_metadata(&json_path)
                    .map_err(|e| anyhow::anyhow!("Failed to load metadata: {}", e))?
            } else {
                tracing::warn!("No inference metadata found, using defaults");
                InferenceMetadata {
                    required_capabilities: vec![],
                    model: None,
                    kv_cache: None,
                    quantization: None,
                    pipeline: None,
                    strategy: None,
                    structured_output: None,
                    hardware_requirements: None,
                }
            }
        };

        // Validate capabilities
        let runtime_caps = onnx_genai_metadata::RuntimeCapabilities::default();
        if let Err(unsupported) = onnx_genai_metadata::validate(&metadata, &runtime_caps) {
            anyhow::bail!("Unsupported capabilities: {:?}", unsupported);
        }

        // Initialize KV cache
        let kv_cache = PagedKvCache::new(config.page_size, config.num_gpu_pages);

        // Initialize scheduler
        let scheduler = Scheduler::new(config.scheduler);

        // TODO: Load ORT session
        // TODO: Load tokenizer

        Ok(Self {
            metadata,
            kv_cache,
            scheduler,
        })
    }

    /// Create a new generation session.
    pub fn create_session(&mut self) -> SequenceId {
        self.kv_cache.create_sequence()
    }

    /// Get the loaded metadata.
    pub fn metadata(&self) -> &InferenceMetadata {
        &self.metadata
    }
}
