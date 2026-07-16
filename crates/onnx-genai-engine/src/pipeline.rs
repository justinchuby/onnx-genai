//! Multi-model pipeline orchestrator.

use crate::decode::{
    DecodeState, clone_value, extract_next_token_logits, run_decode_step_with_extra,
};
use crate::decode_loop::{DecodeLoopBackend, DecodeLoopState, run_decode_loop};
use crate::engine::{Engine, EngineConfig, model_requires_native_backend};
use crate::kv_bridge::infer_kv_model_info;
use crate::logits::TokenId;
use crate::processors::build_processor_chain;
use crate::{
    EngineDecodeBackend, GeneratePrompt, GenerateRequest, GenerateResult, GenerateTokenCallback,
};
use anyhow::Context;
use onnx_genai_metadata::{
    DataflowEdge, PhaseRunOn, PipelineSpec, PipelineStrategy, PipelineStrategyKind,
    PipelineVisionConfig,
};
use onnx_genai_ort::{
    PipelineModelDirectory, PipelineModels, Session, SessionOptions, Tokenizer, Value,
};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

/// Named tensors supplied to or produced by pipeline components.
///
/// Keys are fully-qualified endpoints of the form `component.input_name` or
/// `component.output_name`.
pub type PipelineTensors = HashMap<String, Value>;

/// A pipeline generation request.
pub struct PipelineGenerateRequest {
    pub request: GenerateRequest,
    /// External tensors keyed by `component.input_name`.
    pub inputs: PipelineTensors,
    /// Number of image tiles represented by the external vision tensor.
    ///
    /// This is known only after preprocessing and must be supplied before
    /// decoder KV allocation for encoder-free multimodal pipelines.
    pub num_image_tiles: Option<usize>,
}

impl PipelineGenerateRequest {
    pub fn new(request: GenerateRequest) -> Self {
        Self {
            request,
            inputs: HashMap::new(),
            num_image_tiles: None,
        }
    }

    pub fn with_input(mut self, endpoint: impl Into<String>, value: Value) -> Self {
        self.inputs.insert(endpoint.into(), value);
        self
    }

    pub fn with_image_tile_count(mut self, num_image_tiles: usize) -> Self {
        self.num_image_tiles = Some(num_image_tiles);
        self
    }
}

impl From<GenerateRequest> for PipelineGenerateRequest {
    fn from(request: GenerateRequest) -> Self {
        Self::new(request)
    }
}

/// Engine for metadata-declared multi-model pipelines.
pub struct PipelineEngine {
    models: PipelineModels,
    plan: PipelinePlan,
    decoder_state: DecodeState,
    tokenizer_component: String,
}

impl Engine {
    /// Load a metadata-declared pipeline directory.
    ///
    /// The returned [`PipelineEngine`] keeps the existing single-model `Engine`
    /// API stable while exposing a separate end-to-end pipeline path.
    pub fn from_pipeline_dir(
        pipeline_dir: &Path,
        config: EngineConfig,
    ) -> anyhow::Result<PipelineEngine> {
        PipelineEngine::from_dir_with_config(pipeline_dir, config)
    }
}

impl PipelineEngine {
    /// Load all pipeline sessions with default CPU ORT options.
    pub fn from_dir(pipeline_dir: &Path) -> anyhow::Result<Self> {
        Self::from_dir_with_config(pipeline_dir, EngineConfig::default())
    }

    pub fn from_dir_with_config(pipeline_dir: &Path, config: EngineConfig) -> anyhow::Result<Self> {
        if config.decode_backend == EngineDecodeBackend::Native {
            anyhow::bail!("native backend not supported for pipeline models");
        }
        if config.decode_backend == EngineDecodeBackend::Auto {
            let directory = PipelineModelDirectory::load(pipeline_dir)
                .map_err(|e| anyhow::anyhow!("Failed to resolve pipeline models: {}", e))?;
            for (component, model_path) in &directory.model_paths {
                if model_requires_native_backend(model_path)? {
                    anyhow::bail!(
                        "native backend not supported for pipeline models: component '{component}' requires the native backend"
                    );
                }
            }
        }
        let models = PipelineModels::load_with_options(pipeline_dir, SessionOptions::default())
            .map_err(|e| anyhow::anyhow!("Failed to load pipeline models: {}", e))?;
        let plan = PipelinePlan::from_spec(&models.directory.spec)?;
        let decoder = models
            .session(&plan.decoder)
            .with_context(|| format!("pipeline decoder '{}' was not loaded", plan.decoder))?;
        let _kv_model = infer_kv_model_info(decoder, config.page_size, config.kv_cache_dtype)?;
        let decoder_state = DecodeState::new(decoder)?;
        let tokenizer_component = plan.decoder.clone();
        Ok(Self {
            models,
            plan,
            decoder_state,
            tokenizer_component,
        })
    }

    /// Generate text from a pipeline with no extra non-text tensors.
    pub fn generate(&mut self, request: GenerateRequest) -> anyhow::Result<GenerateResult> {
        self.generate_with_pipeline_request(request.into())
    }

    /// Generate text while supplying external component inputs, such as
    /// `vision_encoder.pixel_values` for a VLM encoder.
    pub fn generate_with_pipeline_request(
        &mut self,
        pipeline_request: PipelineGenerateRequest,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_with_callback(pipeline_request, None)
    }

    /// Generate text and optionally stream tokens.
    pub fn generate_with_callback(
        &mut self,
        pipeline_request: PipelineGenerateRequest,
        callback: Option<&mut GenerateTokenCallback<'_>>,
    ) -> anyhow::Result<GenerateResult> {
        let mut options = pipeline_request.request.options.clone();
        options.validate()?;
        if options.eos_token_id.is_none() {
            options.eos_token_id = self.tokenizer()?.eos_token_id();
        }
        let prompt_tokens = tokenize_with(self.tokenizer()?, &pipeline_request.request.prompt)?;
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt must contain at least one token");
        }
        if pipeline_request.num_image_tiles == Some(0) {
            anyhow::bail!("image tile count must be greater than zero");
        }
        // TODO(#14): Pipeline metadata must declare the image placeholder token
        // and tokens-per-tile contract. Expand that placeholder here using
        // `num_image_tiles` before DecodeState/KV allocation. The server vision
        // seam should pass ImageTensor::num_tiles via with_image_tile_count().

        let prompt_tokens = expand_image_placeholders_count_based(
            prompt_tokens,
            pipeline_request.num_image_tiles,
            self.models.directory.spec.vision.as_ref(),
        )?;

        let mut tensors = pipeline_request.inputs;
        self.run_prompt_phase_components(&mut tensors)?;
        let decoder_extras = self.decoder_extra_inputs(&tensors)?;

        let chain = build_processor_chain(&options, Some(self.tokenizer()?))?;
        self.decoder_state = {
            let decoder = self.models.session(&self.plan.decoder).with_context(|| {
                format!("pipeline decoder '{}' was not loaded", self.plan.decoder)
            })?;
            DecodeState::new(decoder)?
        };

        let decoder = self
            .models
            .session(&self.plan.decoder)
            .with_context(|| format!("pipeline decoder '{}' was not loaded", self.plan.decoder))?;
        let tokenizer = self
            .models
            .tokenizer_for(&self.tokenizer_component)
            .with_context(|| {
                format!("no tokenizer available for '{}'", self.tokenizer_component)
            })?;
        let mut backend = PipelineDecodeLoopBackend {
            decoder,
            decoder_state: &mut self.decoder_state,
            decoder_extras: &decoder_extras,
            context_tokens: prompt_tokens,
            prompt_len: 0,
            generated_count: 0,
        };
        backend.prompt_len = backend.context_tokens.len();
        let mut loop_state = DecodeLoopState::new(0, options.seed, options.top_logprobs);
        run_decode_loop(
            &mut backend,
            &mut loop_state,
            &options,
            &chain,
            tokenizer,
            None,
            callback,
        )
    }

    pub fn spec(&self) -> &PipelineSpec {
        &self.models.directory.spec
    }

    fn tokenizer(&self) -> anyhow::Result<&Tokenizer> {
        self.models
            .tokenizer_for(&self.tokenizer_component)
            .with_context(|| format!("no tokenizer available for '{}'", self.tokenizer_component))
    }

    fn run_prompt_phase_components(&self, tensors: &mut PipelineTensors) -> anyhow::Result<()> {
        for component in &self.plan.prompt_components {
            let session = self
                .models
                .session(component)
                .with_context(|| format!("pipeline component '{component}' was not loaded"))?;
            let inputs = self.component_inputs(component, session, tensors)?;
            let refs = inputs
                .iter()
                .map(|(name, value)| (name.as_str(), value))
                .collect::<Vec<_>>();
            let outputs = session
                .run(&refs)
                .map_err(|e| anyhow::anyhow!("ORT pipeline component '{component}' failed: {e}"))?;
            for (name, value) in session.output_names().iter().zip(outputs) {
                tensors.insert(format!("{component}.{name}"), value);
            }
        }
        Ok(())
    }

    fn component_inputs(
        &self,
        component: &str,
        session: &Session,
        tensors: &PipelineTensors,
    ) -> anyhow::Result<Vec<(String, Value)>> {
        let mut inputs = Vec::new();
        for info in session.inputs() {
            let endpoint = format!("{component}.{}", info.name);
            let routed = self
                .plan
                .dataflow
                .iter()
                .find(|edge| edge.to == endpoint)
                .and_then(|edge| tensors.get(&edge.from));
            let value = tensors
                .get(&endpoint)
                .or(routed)
                .with_context(|| format!("missing pipeline input '{endpoint}'"))?;
            inputs.push((info.name.clone(), clone_value(value)?));
        }
        Ok(inputs)
    }

    fn decoder_extra_inputs(
        &self,
        tensors: &PipelineTensors,
    ) -> anyhow::Result<Vec<(String, Value)>> {
        let mut extras = Vec::new();
        for edge in self
            .plan
            .edges_to_component(&self.plan.decoder)
            .filter(|edge| {
                endpoint_component(&edge.from).is_some_and(|from| from != self.plan.decoder)
            })
        {
            let (_, input) = parse_endpoint(&edge.to)?;
            let value = tensors
                .get(&edge.from)
                .with_context(|| format!("missing routed pipeline tensor '{}'", edge.from))?;
            extras.push((input.to_string(), clone_value(value)?));
        }
        Ok(extras)
    }
}

fn tokenize_with(tokenizer: &Tokenizer, prompt: &GeneratePrompt) -> anyhow::Result<Vec<TokenId>> {
    match prompt {
        GeneratePrompt::TokenIds(tokens) => Ok(tokens.clone()),
        GeneratePrompt::Text(text) => tokenizer
            .encode(text)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {}", e)),
    }
}

/// Expand the single image placeholder token in `prompt_tokens` into
/// `tokens_per_tile * num_tiles` copies of that same token.
///
/// Returns the input unchanged when `num_image_tiles` is `None`.
///
/// Only a **single** placeholder occurrence is supported. `num_image_tiles` is
/// an aggregate tile count across all images in the request, so expanding
/// multiple placeholders by that aggregate would produce the wrong number of
/// image-token slots. Richer per-image metadata (and row/column separator
/// tokens) requires the full preprocessing path; this count-based path targets
/// separator-free single-image models only.
///
/// Errors when:
/// - `num_image_tiles` is `Some` but the pipeline metadata declares no vision
///   contract (`image_placeholder_token_id` or `tokens_per_tile` missing).
/// - The placeholder token ID does not fit in `TokenId` (u32).
/// - `tokens_per_tile` is zero.
/// - The prompt contains no placeholder token, or more than one.
/// - Arithmetic would overflow, or the expanded sequence is empty.
fn expand_image_placeholders_count_based(
    prompt_tokens: Vec<TokenId>,
    num_image_tiles: Option<usize>,
    vision: Option<&PipelineVisionConfig>,
) -> anyhow::Result<Vec<TokenId>> {
    let num_tiles = match num_image_tiles {
        None => return Ok(prompt_tokens),
        Some(n) => n,
    };

    let (placeholder_i64, tokens_per_tile) = match vision {
        Some(v) => match (v.image_placeholder_token_id, v.tokens_per_tile) {
            (Some(id), Some(tpt)) => (id, tpt),
            _ => anyhow::bail!(
                "image tile count supplied but pipeline metadata vision contract is incomplete: \
                 both image_placeholder_token_id and tokens_per_tile must be set"
            ),
        },
        None => anyhow::bail!(
            "image tile count supplied but pipeline metadata declares no vision section; \
             add pipeline.vision with image_placeholder_token_id and tokens_per_tile"
        ),
    };

    if tokens_per_tile == 0 {
        anyhow::bail!(
            "pipeline metadata tokens_per_tile is 0; must be at least 1"
        );
    }

    let placeholder_id: TokenId = u32::try_from(placeholder_i64)
        .with_context(|| {
            format!(
                "image_placeholder_token_id {placeholder_i64} is out of range for token ID (u32)"
            )
        })?;

    let placeholder_count = prompt_tokens
        .iter()
        .filter(|&&t| t == placeholder_id)
        .count();
    if placeholder_count == 0 {
        anyhow::bail!(
            "num_image_tiles supplied but prompt contains no image placeholder token \
             (id={placeholder_id}); the prompt must contain exactly one placeholder"
        );
    }
    if placeholder_count > 1 {
        anyhow::bail!(
            "multi-image count-based expansion is not supported: found {placeholder_count} image \
             placeholders (id={placeholder_id}) but only an aggregate tile count is available; \
             supply a single image or thread per-image tile counts"
        );
    }

    let expansion: usize = tokens_per_tile
        .checked_mul(num_tiles)
        .context("image token expansion overflow: tokens_per_tile * num_image_tiles is too large")?;

    // The single placeholder expands to `expansion` copies; all other tokens are kept.
    let new_len = prompt_tokens
        .len()
        .checked_sub(1)
        .and_then(|base| base.checked_add(expansion))
        .context("expanded prompt token sequence length overflows")?;

    let mut expanded = Vec::new();
    expanded
        .try_reserve_exact(new_len)
        .context("failed to allocate expanded prompt token sequence")?;

    for token in prompt_tokens {
        if token == placeholder_id {
            for _ in 0..expansion {
                expanded.push(placeholder_id);
            }
        } else {
            expanded.push(token);
        }
    }

    if expanded.is_empty() {
        anyhow::bail!(
            "image placeholder expansion produced an empty token sequence; \
             check that num_image_tiles > 0 and the prompt contains non-placeholder tokens"
        );
    }

    Ok(expanded)
}

struct PipelineDecodeLoopBackend<'a> {
    decoder: &'a Session,
    decoder_state: &'a mut DecodeState,
    decoder_extras: &'a [(String, Value)],
    context_tokens: Vec<TokenId>,
    prompt_len: usize,
    generated_count: usize,
}

impl DecodeLoopBackend for PipelineDecodeLoopBackend<'_> {
    fn context_len(&self) -> usize {
        self.context_tokens.len()
    }

    fn processor_prompt_tokens(&self) -> Vec<TokenId> {
        self.context_tokens.clone()
    }

    fn next_logits(&mut self) -> anyhow::Result<Vec<f32>> {
        let past_len = if self.decoder_state.use_kv {
            self.context_tokens
                .len()
                .saturating_sub(if self.generated_count == 0 {
                    self.prompt_len
                } else {
                    1
                })
        } else {
            0
        };
        let input_tokens = if self.decoder_state.use_kv && self.generated_count > 0 {
            self.context_tokens[self.context_tokens.len() - 1..].to_vec()
        } else {
            self.context_tokens.clone()
        };
        let outputs = run_decode_step_with_extra(
            self.decoder,
            self.decoder_state,
            &input_tokens,
            past_len,
            self.decoder_extras,
        )?;
        extract_next_token_logits(self.decoder, outputs)
    }

    fn commit_token(&mut self, token_id: TokenId) -> anyhow::Result<()> {
        self.context_tokens.push(token_id);
        self.generated_count += 1;
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct PipelinePlan {
    decoder: String,
    prompt_components: Vec<String>,
    dataflow: Vec<DataflowEdge>,
}

impl PipelinePlan {
    fn from_spec(spec: &PipelineSpec) -> anyhow::Result<Self> {
        let decoder = autoregressive_decoder(&spec.strategy)
            .context("pipeline strategy must contain an autoregressive decoder")?;
        if !spec.models.contains_key(&decoder) {
            anyhow::bail!("pipeline decoder '{decoder}' is not declared in models");
        }

        let mut prompt_components = Vec::new();
        for component in topological_components(spec)? {
            if component == decoder {
                continue;
            }
            match component_phase(spec, &component, &decoder) {
                PhaseRunOn::PromptOnly => prompt_components.push(component),
                PhaseRunOn::EveryStep | PhaseRunOn::OnDemand | PhaseRunOn::FinalOnly => {}
                PhaseRunOn::Other(value) => {
                    anyhow::bail!(
                        "unsupported phase '{value}' for pipeline component '{component}'"
                    )
                }
            }
        }

        Ok(Self {
            decoder,
            prompt_components,
            dataflow: spec.dataflow.clone(),
        })
    }

    fn edges_to_component<'a>(
        &'a self,
        component: &'a str,
    ) -> impl Iterator<Item = &'a DataflowEdge> + 'a {
        self.dataflow
            .iter()
            .filter(move |edge| endpoint_component(&edge.to) == Some(component))
    }
}

fn autoregressive_decoder(strategy: &PipelineStrategy) -> Option<String> {
    match strategy.kind {
        PipelineStrategyKind::Autoregressive => strategy.decoder.clone(),
        PipelineStrategyKind::Composite => strategy
            .stages
            .iter()
            .find_map(|stage| autoregressive_decoder(&stage.strategy)),
        PipelineStrategyKind::Iterative
        | PipelineStrategyKind::SinglePass
        | PipelineStrategyKind::Other(_) => None,
    }
}

fn component_phase(spec: &PipelineSpec, component: &str, decoder: &str) -> PhaseRunOn {
    spec.phases
        .get(component)
        .map(|phase| phase.run_on.clone())
        .unwrap_or_else(|| {
            if component == decoder {
                PhaseRunOn::EveryStep
            } else {
                PhaseRunOn::PromptOnly
            }
        })
}

fn topological_components(spec: &PipelineSpec) -> anyhow::Result<Vec<String>> {
    let mut remaining = spec.models.keys().cloned().collect::<BTreeSet<_>>();
    let mut ordered = Vec::new();
    while !remaining.is_empty() {
        let ready = remaining
            .iter()
            .find(|component| {
                spec.dataflow.iter().all(|edge| {
                    endpoint_component(&edge.to) != Some(component.as_str())
                        || endpoint_component(&edge.from)
                            .is_some_and(|from| !remaining.contains(from))
                })
            })
            .cloned();
        let Some(component) = ready else {
            anyhow::bail!("pipeline dataflow contains a cycle");
        };
        remaining.remove(&component);
        ordered.push(component);
    }
    Ok(ordered)
}

fn parse_endpoint(endpoint: &str) -> anyhow::Result<(&str, &str)> {
    endpoint
        .split_once('.')
        .filter(|(component, port)| !component.is_empty() && !port.is_empty())
        .with_context(|| format!("pipeline endpoint must be component.port: {endpoint}"))
}

fn endpoint_component(endpoint: &str) -> Option<&str> {
    parse_endpoint(endpoint)
        .ok()
        .map(|(component, _)| component)
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai_metadata::{PhaseConfig, PipelineComponentSpec, PipelineStrategyStage};
    use std::collections::BTreeMap;

    fn component(role: &str) -> PipelineComponentSpec {
        PipelineComponentSpec {
            filename: format!("{role}.onnx"),
            role: role.to_string(),
            device_preference: None,
            tokenizer: None,
        }
    }

    #[test]
    fn explicit_native_backend_is_rejected_before_loading_pipeline_models() {
        let error = PipelineEngine::from_dir_with_config(
            Path::new("does-not-need-to-exist"),
            EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                ..EngineConfig::default()
            },
        )
        .err()
        .expect("native pipeline backend must be rejected");
        assert!(
            error
                .to_string()
                .contains("native backend not supported for pipeline models")
        );
    }

    #[cfg(feature = "native-backend")]
    #[test]
    fn auto_backend_rejects_pipeline_component_requiring_native() -> anyhow::Result<()> {
        use onnx_runtime_loader::proto::{
            ModelProto,
            onnx::{GraphProto, NodeProto, OperatorSetIdProto},
        };
        use prost::Message;

        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/test-fixtures/pipeline-native-backend-rejection");
        std::fs::create_dir_all(&root)?;
        let model = ModelProto {
            opset_import: vec![OperatorSetIdProto {
                domain: "com.github.onnxruntime.genai".to_string(),
                version: 1,
            }],
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    domain: "com.github.onnxruntime.genai".to_string(),
                    op_type: "BlockQuantizedMatMul".to_string(),
                    ..NodeProto::default()
                }],
                ..GraphProto::default()
            }),
            ..ModelProto::default()
        };
        std::fs::write(root.join("decoder.onnx"), model.encode_to_vec())?;
        std::fs::write(
            root.join("inference_metadata.yaml"),
            r#"
pipeline:
  models:
    decoder:
      filename: decoder.onnx
      type: decoder
  dataflow: []
  strategy:
    kind: autoregressive
    decoder: decoder
"#,
        )?;

        let error = PipelineEngine::from_dir_with_config(&root, EngineConfig::default())
            .err()
            .expect("Auto must reject native-only pipeline components");
        assert!(
            error
                .to_string()
                .contains("native backend not supported for pipeline models")
        );
        Ok(())
    }

    #[test]
    fn plan_routes_prompt_encoder_outputs_to_decoder_inputs() -> anyhow::Result<()> {
        let spec = PipelineSpec {
            models: BTreeMap::from([
                ("vision_encoder".to_string(), component("encoder")),
                ("decoder".to_string(), component("decoder")),
            ]),
            dataflow: vec![DataflowEdge {
                from: "vision_encoder.image_features".to_string(),
                to: "decoder.encoder_hidden_states".to_string(),
                dtype: Some("fp32".to_string()),
                device_transfer: Some(false),
            }],
            strategy: PipelineStrategy {
                kind: PipelineStrategyKind::Composite,
                decoder: None,
                max_tokens: None,
                stop_conditions: None,
                kv_cache: None,
                speculative: None,
                model: None,
                batching: None,
                denoiser: None,
                scheduler: None,
                num_steps: None,
                guidance_scale: None,
                state: None,
                stages: vec![
                    PipelineStrategyStage {
                        name: "encode".to_string(),
                        strategy: Box::new(PipelineStrategy {
                            kind: PipelineStrategyKind::SinglePass,
                            decoder: None,
                            max_tokens: None,
                            stop_conditions: None,
                            kv_cache: None,
                            speculative: None,
                            model: Some("vision_encoder".to_string()),
                            batching: None,
                            denoiser: None,
                            scheduler: None,
                            num_steps: None,
                            guidance_scale: None,
                            state: None,
                            stages: vec![],
                        }),
                        run_on: Some(PhaseRunOn::PromptOnly),
                    },
                    PipelineStrategyStage {
                        name: "decode".to_string(),
                        strategy: Box::new(PipelineStrategy {
                            kind: PipelineStrategyKind::Autoregressive,
                            decoder: Some("decoder".to_string()),
                            max_tokens: None,
                            stop_conditions: None,
                            kv_cache: None,
                            speculative: None,
                            model: None,
                            batching: None,
                            denoiser: None,
                            scheduler: None,
                            num_steps: None,
                            guidance_scale: None,
                            state: None,
                            stages: vec![],
                        }),
                        run_on: Some(PhaseRunOn::EveryStep),
                    },
                ],
            },
            phases: BTreeMap::from([
                (
                    "vision_encoder".to_string(),
                    PhaseConfig {
                        run_on: PhaseRunOn::PromptOnly,
                    },
                ),
                (
                    "decoder".to_string(),
                    PhaseConfig {
                        run_on: PhaseRunOn::EveryStep,
                    },
                ),
            ]),
            vision: None,
        };

        let plan = PipelinePlan::from_spec(&spec)?;
        assert_eq!(plan.prompt_components, ["vision_encoder"]);
        assert_eq!(plan.decoder, "decoder");
        let routed = plan.edges_to_component("decoder").collect::<Vec<_>>();
        assert_eq!(routed.len(), 1);
        assert_eq!(
            parse_endpoint(&routed[0].to)?,
            ("decoder", "encoder_hidden_states")
        );
        assert_eq!(routed[0].from, "vision_encoder.image_features");
        Ok(())
    }

    fn vision_config(placeholder_id: i64, tpt: usize) -> PipelineVisionConfig {
        PipelineVisionConfig {
            image_placeholder_token_id: Some(placeholder_id),
            tokens_per_tile: Some(tpt),
        }
    }

    #[test]
    fn image_placeholder_expansion_replaces_tokens() {
        // [1, PLACEHOLDER, 2] with 2 tiles × 3 tokens/tile → [1, IMG, IMG, IMG, IMG, IMG, IMG, 2]
        let tokens: Vec<TokenId> = vec![1, 100, 2];
        let cfg = vision_config(100, 3);
        let expanded =
            expand_image_placeholders_count_based(tokens, Some(2), Some(&cfg)).unwrap();
        assert_eq!(expanded, vec![1, 100, 100, 100, 100, 100, 100, 2]);
    }

    #[test]
    fn image_placeholder_expansion_multiple_placeholders_errors() {
        // Count-based path only supports a single placeholder; >1 must error.
        let tokens: Vec<TokenId> = vec![100, 5, 100];
        let cfg = vision_config(100, 4);
        let err =
            expand_image_placeholders_count_based(tokens, Some(1), Some(&cfg)).unwrap_err();
        assert!(
            err.to_string().contains("multi-image count-based expansion is not supported"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn image_placeholder_expansion_none_tiles_is_noop() {
        let tokens: Vec<TokenId> = vec![1, 100, 2];
        let cfg = vision_config(100, 256);
        let result = expand_image_placeholders_count_based(tokens.clone(), None, Some(&cfg)).unwrap();
        assert_eq!(result, tokens);
    }

    #[test]
    fn image_placeholder_expansion_no_vision_config_with_tiles_errors() {
        let tokens: Vec<TokenId> = vec![1, 100, 2];
        let err =
            expand_image_placeholders_count_based(tokens, Some(1), None).unwrap_err();
        assert!(err.to_string().contains("no vision section"));
    }

    #[test]
    fn image_placeholder_expansion_incomplete_contract_errors() {
        let tokens: Vec<TokenId> = vec![1, 100, 2];
        let cfg = PipelineVisionConfig {
            image_placeholder_token_id: Some(100),
            tokens_per_tile: None,
        };
        let err =
            expand_image_placeholders_count_based(tokens, Some(1), Some(&cfg)).unwrap_err();
        assert!(err.to_string().contains("vision contract is incomplete"));
    }

    #[test]
    fn image_placeholder_expansion_missing_placeholder_errors() {
        let tokens: Vec<TokenId> = vec![1, 2, 3];
        let cfg = vision_config(100, 4);
        let err =
            expand_image_placeholders_count_based(tokens, Some(1), Some(&cfg)).unwrap_err();
        assert!(err.to_string().contains("no image placeholder token"));
    }

    #[test]
    fn image_placeholder_expansion_negative_id_errors() {
        let tokens: Vec<TokenId> = vec![1, 2];
        let cfg = PipelineVisionConfig {
            image_placeholder_token_id: Some(-1),
            tokens_per_tile: Some(4),
        };
        let err =
            expand_image_placeholders_count_based(tokens, Some(1), Some(&cfg)).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn image_placeholder_expansion_tokens_per_tile_zero_errors() {
        let tokens: Vec<TokenId> = vec![1, 100, 2];
        let cfg = vision_config(100, 0);
        let err =
            expand_image_placeholders_count_based(tokens, Some(1), Some(&cfg)).unwrap_err();
        assert!(
            err.to_string().contains("tokens_per_tile is 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn image_placeholder_expansion_zero_tiles_produces_empty_errors() {
        // tokens_per_tile=4, num_tiles=0 → expansion=0 → prompt becomes empty
        let tokens: Vec<TokenId> = vec![100];
        let cfg = vision_config(100, 4);
        let err =
            expand_image_placeholders_count_based(tokens, Some(0), Some(&cfg)).unwrap_err();
        assert!(
            err.to_string().contains("empty token sequence"),
            "unexpected error: {err}"
        );
    }
}
