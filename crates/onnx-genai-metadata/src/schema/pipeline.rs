use super::*;

/// Multi-model pipeline represented as a directed acyclic dataflow graph.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
pub struct PipelineSpec {
    /// North-star component-centric SSA workflow.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workflow: Option<WorkflowSpec>,

    /// Typed inputs exposed by the complete package.
    #[serde(default)]
    pub inputs: BTreeMap<String, TensorContract>,

    /// Typed outputs produced by the complete package.
    #[serde(default)]
    pub outputs: BTreeMap<String, TensorContract>,

    /// Named model components in the pipeline DAG; at least one component is required.
    #[serde(default)]
    #[schemars(extend("minProperties" = 1))]
    pub models: BTreeMap<String, PipelineComponentSpec>,

    /// Directed tensor or data edges between component ports.
    #[serde(default)]
    pub dataflow: Vec<DataflowEdge>,

    /// Explicit fan-in reducers keyed by destination endpoint.
    #[serde(default)]
    pub reducers: BTreeMap<String, ReducerSpec>,

    /// Universal nested control-flow program.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control: Option<ControlFlow>,

    /// General tensor state, including loop-carried and persistent session state.
    #[serde(default)]
    pub states: BTreeMap<String, StateDeclaration>,

    /// Named data-only sampler, scheduler, solver, and tensor programs.
    #[serde(default)]
    pub programs: BTreeMap<String, Program>,

    /// Typed resource contracts for named components.
    #[serde(default)]
    pub resources: BTreeMap<String, ResourceContract>,

    /// Package batching contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batching: Option<BatchingContract>,

    /// Declarative output materialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub postprocessing: Option<PostprocessingSpec>,

    /// Loop and execution strategy for the pipeline.
    #[serde(default)]
    pub strategy: PipelineStrategy,

    /// Auxiliary-component lifecycle scheduling, keyed by component name.
    ///
    /// Models referenced directly by strategy control-flow fields (`decoder`,
    /// `model`, `denoiser`, `outer`, or `inner`) must not appear here. Every
    /// other model must have exactly one phase entry.
    #[serde(default)]
    pub phases: BTreeMap<String, PhaseConfig>,

    /// Vision-language model token-expansion contract.
    ///
    /// When present, the engine uses these fields to replace each image
    /// placeholder token in the prompt with the declared expanded image-token
    /// sequence before KV-cache allocation.
    #[serde(default)]
    pub vision: Option<PipelineVisionConfig>,

    /// Waveform contract for a pipeline whose final stage emits audio.
    ///
    /// Present for text-to-speech and any other package that produces sound.
    /// The sample rate is model DATA: a runtime cannot infer it from a tensor,
    /// and guessing it silently changes playback pitch and duration.
    #[serde(default)]
    pub audio: Option<PipelineAudioConfig>,

    /// Declared position-id generation and prefill→decode continuation program.
    ///
    /// Generic and architecture-neutral: parameterized by rank, axis labels, and
    /// section sizes so it expresses both ordinary rank-2 linear positions and
    /// rank-N multimodal coordinates as data — never a model-family branch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positions: Option<PositionProgram>,
}

/// Waveform contract for a pipeline stage that emits audio.
///
/// Architecture-neutral: the endpoint is an arbitrary `component.output` name
/// carried in the package's metadata, and the sample rate is a declared number.
/// Neither is inferred from a model or vendor name.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema)]
pub struct PipelineAudioConfig {
    /// Sample rate, in hertz, of the waveform the pipeline emits.
    #[schemars(range(min = 1))]
    pub sample_rate: Option<u32>,

    /// Endpoint carrying the waveform, in `component.output` form.
    ///
    /// When absent, the runtime uses the sole output of the final-phase
    /// component, which is unambiguous for the common single-vocoder shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,

    /// Number of interleaved channels in the waveform. Defaults to 1 (mono).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub channels: Option<u16>,
}

/// Declared position-id program for a decoder graph.
///
/// The runtime constructs the position tensor from these declared parameters
/// instead of assuming a fixed rank-2 layout. `rank` 1 (with a single axis)
/// expresses ordinary linear positions; `rank` N expresses multi-axis
/// multimodal coordinates. Axis labels and section sizes are opaque DATA — the
/// runtime never infers them from a model name.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema)]
pub struct PositionProgram {
    /// Graph input port that receives the position ids (arbitrary name, DATA).
    #[schemars(length(min = 1))]
    pub input: String,

    /// Number of coordinate streams carried by the position tensor.
    ///
    /// `1` is an ordinary linear position stream; values `> 1` describe
    /// multi-axis multimodal coordinates. The physical ONNX tensor rank is
    /// declared separately by `tensor_rank`.
    #[schemars(range(min = 1))]
    pub rank: usize,

    /// Physical ONNX tensor rank.
    ///
    /// Rank 2 declares a conventional `[batch, sequence]` linear input. Higher
    /// ranks declare an explicit coordinate axis in addition to batch/sequence
    /// axes. Absent preserves the legacy mapping (`rank == 1` means tensor rank
    /// 2; otherwise tensor rank 3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 2))]
    pub tensor_rank: Option<usize>,

    /// How the position values are generated for prefill.
    ///
    /// `linear` generates ordinary sequence positions. `processor_coordinates`
    /// consumes the declared processor summaries to construct multi-axis
    /// coordinates. Future generation programs remain extensible capability
    /// strings rather than model-family branches.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<schema_vocabulary::PositionGeneration>")]
    pub generation: Option<String>,

    /// Optional coordinate-stream labels, one per stream (DATA).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(inner(length(min = 1)))]
    pub axes: Option<Vec<String>>,

    /// Optional section sizes for sectioned rotary position embeddings.
    ///
    /// Opaque list of per-section widths; their meaning is model DATA, not a
    /// runtime branch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sections: Option<Vec<usize>>,

    /// Declared dtype of the position tensor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<schema_vocabulary::TensorDType>")]
    pub dtype: Option<String>,

    /// How positions continue from the prompt (prefill) into per-token decode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<schema_vocabulary::PositionContinuation>")]
    pub continuation: Option<String>,

    /// Optional processor-summary endpoints this program reads to compute
    /// multi-axis coordinates (e.g. a declared grid-dimensions output). Each
    /// entry is an arbitrary endpoint name (DATA), never a model-family hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(inner(length(min = 1)))]
    pub processor_summaries: Option<Vec<String>>,
}

/// Image placeholder token-expansion contract for encoder-free VLM pipelines.
///
/// Every field is optional and additive: legacy documents that declare only
/// `image_placeholder_token_id` and `tokens_per_tile` keep working. The richer
/// fields mirror the generic expansion the preprocessor already models
/// (separate emitted image token, per-tile/per-patch count source, per-image
/// correspondence, optional row/column separators, and thumbnail order). All of
/// it is generic data — no field names or values reference a model family.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, JsonSchema)]
pub struct PipelineVisionConfig {
    /// Token ID of the image placeholder in the tokenized prompt.
    ///
    /// The engine replaces every occurrence of this token with the expanded
    /// image token sequence before sequence-length and KV-cache sizing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_placeholder_token_id: Option<i64>,

    /// Number of image tokens each tile expands to.
    ///
    /// The total per-tile expansion is `tokens_per_tile * num_tiles`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub tokens_per_tile: Option<usize>,

    /// Token ID emitted for each expanded image position.
    ///
    /// Distinct from `image_placeholder_token_id`: the placeholder marks WHERE
    /// to expand, while this is the token actually written into the expanded
    /// sequence. When absent, the placeholder token itself is repeated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_token_id: Option<i64>,

    /// Where the per-placeholder token count comes from (per tile, per patch, or
    /// a declared grid). Generic selector, never a model-family branch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<schema_vocabulary::ImageTokenCountSource>")]
    pub token_count_source: Option<String>,

    /// Named preprocessing value that supplies per-image counts or grid
    /// dimensions when `token_count_source` is data-derived.
    ///
    /// This is an arbitrary processor output name. A runtime resolves the name
    /// from the declared preprocessing program; it never dispatches on familiar
    /// tensor names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub token_count_summary: Option<String>,

    /// Number of image tokens each patch expands to, used when the count source
    /// is per patch. Declared data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub tokens_per_patch: Option<usize>,

    /// Whether each placeholder occurrence corresponds to one input image in
    /// prompt order. Absent means the historical one-placeholder-per-image rule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placeholder_per_image: Option<bool>,

    /// How prompt placeholders correspond to input images.
    ///
    /// `prompt_order` pairs each placeholder with the next input image.
    /// `explicit_indices` reads correspondence from `correspondence_summary`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<schema_vocabulary::ImageCorrespondence>")]
    pub image_correspondence: Option<String>,

    /// Named preprocessing value containing explicit image correspondence data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub correspondence_summary: Option<String>,

    /// Optional token ID emitted between rows of a tiled image grid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_separator_token_id: Option<i64>,

    /// Optional token ID emitted between columns within a grid row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub column_separator_token_id: Option<i64>,

    /// Order of the optional global thumbnail tile relative to the local grid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<schema_vocabulary::ThumbnailOrder>")]
    pub thumbnail_order: Option<String>,
}

/// Declared, architecture-neutral input preprocessing programs.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, JsonSchema)]
pub struct PreprocessingSpec {
    /// Typed image preprocessing transform program and its named tensor outputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<ImagePreprocessingProgram>,
}

/// Generic image preprocessing program: an ordered transform pipeline plus the
/// named workflow SSA tensor outputs it emits.
///
/// The program is expressed entirely as parameterized, architecture-neutral
/// data. Transform operations are generic (decode, resize, rescale, normalize,
/// tile, patchify, pad). In workflow metadata, outputs are materialized by a
/// manifest-pinned preprocessing adapter invocation and bind processor-local
/// values to typed SSA names. A package may name an output `pixel_position_ids`,
/// `image_grid_thw`, or anything else without introducing runtime model-family
/// dispatch.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct ImagePreprocessingProgram {
    /// Ordered list of generic transform operations applied to decoded pixels.
    #[serde(default)]
    pub transforms: Vec<ImageTransform>,

    /// Named tensor outputs the program emits, each bound to a workflow SSA value.
    #[schemars(length(min = 1))]
    pub outputs: Vec<ImageOutputBinding>,
}

/// One generic image transform operation.
///
/// `op` selects the operation from a generic vocabulary; the remaining fields
/// are the parameters that operation reads (only the relevant ones are set).
/// Every parameter is model DATA — concrete sizes, patch sizes, means, and so on
/// live in a model's fixture, never as constants baked into this schema.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct ImageTransform {
    /// Generic operation selector (e.g. `resize`, `normalize`, `patchify`).
    #[schemars(with = "schema_vocabulary::ImageTransformOp")]
    pub op: String,

    /// Named values consumed by this transform.
    ///
    /// Absent means the operation consumes the immediately preceding value.
    /// Explicit names allow branching programs without tensor-name heuristics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1), inner(length(min = 1)))]
    pub inputs: Option<Vec<String>>,

    /// Named values produced by this transform.
    ///
    /// These names are processor-local data. Final graph bindings select them
    /// through `ImageOutputBinding::source`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1), inner(length(min = 1)))]
    pub outputs: Option<Vec<String>>,

    /// Target size for a `resize` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<ImageSizeSpec>,

    /// Resize/crop mode (e.g. `pad`, `crop`, `stretch`) — generic string data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub mode: Option<String>,

    /// Interpolation filter for a `resize` operation — generic string data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub interpolation: Option<String>,

    /// Minimum pixel area for an aspect-preserving `pixel_area` resize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub min_pixels: Option<usize>,

    /// Maximum pixel area for an aspect-preserving `pixel_area` resize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub max_pixels: Option<usize>,

    /// Required divisibility of both resized dimensions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub size_multiple: Option<usize>,

    /// Maximum number of spatial patches for a patch-budget resize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub max_patches: Option<usize>,

    /// Spatial pooling edge used when resolving a patch-budget resize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub pooling_kernel_size: Option<usize>,

    /// Scalar multiplier for a `rescale` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,

    /// Per-channel mean for a `normalize` operation (length is model data).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean: Option<Vec<f32>>,

    /// Per-channel standard deviation for a `normalize` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub std: Option<Vec<f32>>,

    /// Edge length of a square tile for a `tile` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub tile_size: Option<usize>,

    /// Maximum number of local tiles for a `tile` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub max_tiles: Option<usize>,

    /// Whether a `tile` operation also emits a global thumbnail tile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_thumbnail: Option<bool>,

    /// Ordering of a global thumbnail relative to local tiles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<schema_vocabulary::ThumbnailOrder>")]
    pub thumbnail_order: Option<String>,

    /// Interpolation filter used specifically for a global thumbnail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub thumbnail_interpolation: Option<String>,

    /// RGB canvas fill value applied before dynamic tiling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canvas_pad_value: Option<f64>,

    /// Pixel edge represented by one validity-mask cell.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub mask_patch_size: Option<usize>,

    /// Edge length of a square patch for a `patchify` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub patch_size: Option<usize>,

    /// Number of identical temporal frames packed into each spatial patch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub temporal_patch_size: Option<usize>,

    /// Spatial patch-group edge controlling packed patch traversal order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub merge_size: Option<usize>,

    /// Flattened patch feature order (`channels_first` or `channels_last`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub channel_order: Option<String>,

    /// Patch-coordinate component order (`yx` or `xy`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub coordinate_order: Option<String>,

    /// Whether `patchify` flattens each patch into a single feature vector.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flatten: Option<bool>,

    /// Fill value for a `pad` operation, or sentinel for padded coordinates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pad_value: Option<f64>,

    /// Exact first-axis length produced by a `pad` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub target_length: Option<usize>,
}

/// A square size or an explicit width/height for an image transform.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum ImageSizeSpec {
    /// A single edge length applied to both dimensions.
    Square(u32),
    /// Explicit width and height.
    Dimensions {
        /// Target width in pixels.
        width: u32,
        /// Target height in pixels.
        height: u32,
    },
}

/// One named tensor output produced by an image preprocessing program.
///
/// The output binds a processor-local value to a typed workflow SSA name.
/// Neither the name nor the content role is inferred from a model identity.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct ImageOutputBinding {
    /// Named processor-local value produced by a transform.
    #[schemars(length(min = 1))]
    pub source: String,

    /// Workflow SSA value produced by the preprocessing adapter invocation.
    #[schemars(length(min = 1), example = &"image.pixel_values")]
    pub name: String,

    /// Generic content role this tensor carries (pixels, coordinates, grid,
    /// original size, or validity mask) — never a model-family label.
    #[schemars(with = "schema_vocabulary::ImageOutputContent")]
    pub content: String,

    /// Declared output dtype. Always explicit; never inferred from the model.
    #[schemars(with = "schema_vocabulary::TensorDType")]
    pub dtype: String,

    /// Full workflow tensor contract. Required when `pipeline.workflow` is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract: Option<crate::schema::TensorContract>,

    /// Optional sentinel/pad value for padded entries (e.g. `-1` coordinates).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pad_value: Option<f64>,

    /// Whether the runtime may omit this output when a model does not need it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optional: Option<bool>,
}

/// One executable ONNX model in a pipeline.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct PipelineComponentSpec {
    /// Non-empty ONNX filename relative to the model package root.
    #[schemars(length(min = 1), example = &"decoder.onnx")]
    pub filename: String,

    /// Component role, for example `encoder`, `decoder`, `draft`, `denoiser`, or `vocoder`.
    #[serde(rename = "type")]
    #[schemars(with = "schema_vocabulary::PipelineRole")]
    pub role: String,

    /// Optional execution or device preference declared by the model package.
    #[schemars(with = "Option<schema_vocabulary::DevicePreference>")]
    pub device_preference: Option<String>,

    /// Tokenizer filename relative to the package root.
    ///
    /// If absent, loaders may use a shared top-level `tokenizer.json`.
    #[schemars(length(min = 1), example = &"tokenizer.json")]
    pub tokenizer: Option<String>,

    /// Explicit graph I/O port bindings for this pipeline component.
    ///
    /// The runtime binds decode-step ports from the declared names. A port that
    /// is not declared is resolved ONLY from an unambiguous io-shape signal;
    /// when the shape is ambiguous the runtime fails with an actionable error
    /// naming the key to declare, and never guesses from a tensor name.
    #[serde(default)]
    pub io: Option<ModelIoSpec>,

    /// Typed graph inputs and outputs exposed by this component.
    #[serde(default)]
    pub ports: ComponentPorts,
}

/// Directed connection between two pipeline component ports.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct DataflowEdge {
    /// Source package input or endpoint in `component.output_name` form.
    #[schemars(regex(pattern = r"^[^.]+(?:\.[^.]+)?$"), example = &"encoder.hidden_states")]
    pub from: String,

    /// Destination package output or endpoint in `component.input_name` form.
    #[schemars(regex(pattern = r"^[^.]+(?:\.[^.]+)?$"), example = &"decoder.encoder_hidden_states")]
    pub to: String,

    /// Scalar or logical data type at the component boundary.
    #[schemars(with = "Option<schema_vocabulary::TensorDType>")]
    pub dtype: Option<String>,

    /// Whether the runtime must move the value between execution devices.
    pub device_transfer: Option<bool>,
}

/// Phase gate for one pipeline component.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct PhaseConfig {
    /// Pipeline phase in which the component runs.
    pub run_on: PhaseRunOn,

    /// Opaque presence key required for this component to run.
    #[serde(
        default,
        deserialize_with = "deserialize_optional_non_empty_string",
        skip_serializing_if = "Option::is_none"
    )]
    #[schemars(length(min = 1))]
    pub when_present: Option<String>,
}

/// Pipeline phase gate.
///
/// Known values are enumerated while future strings remain valid.
#[derive(Debug, Clone, PartialEq, Eq, JsonSchema)]
#[schemars(with = "String", transform = schema_helpers::phase_run_on)]
pub enum PhaseRunOn {
    /// Run only while processing the initial prompt.
    PromptOnly,
    /// Run at every pipeline step; `always` is accepted as an alias.
    EveryStep,
    /// Run only when producing the final output.
    FinalOnly,
    /// Run only when explicitly requested by the application.
    OnDemand,
    /// Future phase gate not recognized by this runtime version.
    Other(String),
}

impl<'de> Deserialize<'de> for PhaseRunOn {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "prompt_only" => Self::PromptOnly,
            "every_step" | "always" => Self::EveryStep,
            "final_only" => Self::FinalOnly,
            "on_demand" => Self::OnDemand,
            _ => Self::Other(value),
        })
    }
}

impl Serialize for PhaseRunOn {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(match self {
            Self::PromptOnly => "prompt_only",
            Self::EveryStep => "every_step",
            Self::FinalOnly => "final_only",
            Self::OnDemand => "on_demand",
            Self::Other(value) => value,
        })
    }
}

/// Parameterized execution strategy for a pipeline or composite stage.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
pub struct PipelineStrategy {
    /// Strategy family; determines which strategy-specific fields are meaningful.
    pub kind: PipelineStrategyKind,

    /// Autoregressive decoder component name.
    pub decoder: Option<String>,

    /// Maximum number of tokens generated by an autoregressive stage.
    #[schemars(range(min = 1))]
    pub max_tokens: Option<usize>,

    /// Runtime-specific stop-condition declarations.
    pub stop_conditions: Option<Vec<serde_json::Value>>,

    /// Runtime-specific KV-cache strategy parameters.
    pub kv_cache: Option<serde_json::Value>,

    /// Runtime-specific speculative execution parameters.
    pub speculative: Option<serde_json::Value>,

    /// Single-pass component name.
    pub model: Option<String>,

    /// Runtime-specific batching parameters.
    pub batching: Option<serde_json::Value>,

    /// Iterative or diffusion denoiser component name.
    pub denoiser: Option<String>,

    /// Scheduler identifier for iterative or diffusion execution.
    pub scheduler: Option<String>,

    /// Number of iterative or diffusion steps.
    #[schemars(range(min = 1))]
    pub num_steps: Option<usize>,

    /// Denoiser input port that receives the per-step timestep/sigma scalar.
    ///
    /// When set, the iterative loop feeds this input a rank-1 `float32` value
    /// each step (from `timesteps` when provided, otherwise the 0-based step
    /// index), so a step-aware denoiser can condition on the current step.
    #[serde(default)]
    pub timestep_input: Option<String>,

    /// Explicit per-step timestep/sigma schedule for an iterative strategy.
    ///
    /// When present its length must equal `num_steps`; when absent the loop
    /// uses the 0-based step index. Requires `timestep_input` to have any effect.
    #[serde(default)]
    pub timesteps: Option<Vec<f32>>,

    /// First step index for a partial (img2img) denoise loop.
    ///
    /// When set, the iterative loop runs `start_step..num_steps` instead of the
    /// full `0..num_steps`, and the seed (`denoiser` sample input) is expected to
    /// already be the encoded image noised to `timesteps[start_step]`. Matches
    /// diffusers' img2img `get_timesteps(num_steps, strength)` skip. Default 0.
    #[serde(default)]
    pub start_step: Option<usize>,

    /// Optional diffusion scheduler applied to the denoiser's loop-carried
    /// output (treating it as a noise prediction) each step.
    #[serde(default)]
    pub scheduler_config: Option<SchedulerSpec>,

    /// Denoiser conditioning input port zeroed for the unconditional pass of
    /// classifier-free guidance. Required when `guidance_scale` != 1.0.
    #[serde(default)]
    pub cfg_conditioning_input: Option<String>,

    /// Classifier-free guidance scale or equivalent strategy-specific multiplier.
    #[schemars(range(min = 0.0))]
    pub guidance_scale: Option<f32>,

    /// Runtime-specific iterative state declaration.
    pub state: Option<serde_json::Value>,

    /// Ordered child stages for a composite strategy.
    #[serde(default)]
    pub stages: Vec<PipelineStrategyStage>,

    /// Outer autoregressive decoder for a `nested_autoregressive` stage.
    ///
    /// The multi-decoder TTS shape: one outer step is one
    /// audio frame. The outer decoder (talker) produces a per-frame
    /// `last_hidden_state` that seeds the inner loop (see `inner`).
    #[serde(default)]
    pub outer: Option<String>,

    /// Inner autoregressive decoder for a `nested_autoregressive` stage.
    ///
    /// The code_predictor: for each outer frame it runs a short inner AR loop of
    /// `num_code_groups` steps over the residual codebooks, seeded at inner step
    /// 0 by the outer decoder's `last_hidden_state` (routed via a dataflow edge
    /// `outer.last_hidden_state -> inner.inputs_embeds`) and threading its own
    /// per-step code embedding on later steps.
    #[serde(default)]
    pub inner: Option<String>,

    /// Inner-loop depth (RVQ residual codebook count) for a
    /// `nested_autoregressive` stage: the number of code tokens collected per
    /// outer frame. Must be at least 1.
    #[schemars(range(min = 1))]
    #[serde(default)]
    pub num_code_groups: Option<usize>,

    /// Inner decoder output port threaded across inner steps for a
    /// `nested_autoregressive` stage.
    ///
    /// Each inner step consumes the previous step's per-code embedding as its
    /// `inputs_embeds` seed; this names the inner decoder OUTPUT port that
    /// produces that embedding. It is declared explicitly because the port is
    /// shape-indistinguishable from other float outputs — the runtime must not
    /// infer it by tensor name. Absent on a nested stage ⇒ actionable error
    /// naming `pipeline.strategy.inner_embedding_output`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inner_embedding_output: Option<String>,

    /// Optional pre-embedder component driving the outer decoder (talker) of a
    /// `nested_autoregressive` stage through `inputs_embeds` instead of
    /// `input_ids`.
    ///
    /// A codec-driven TTS talker is not driven by token ids: each step's
    /// `inputs_embeds` is materialized from the PREVIOUS frame's codes as
    /// `codec_sum(+ text_embed)` (where
    /// `codec_sum = codec_embed(code_0) + Σ_i cp_codec_weights[i][codes[i+1]]`).
    /// When this field names such a component (inputs
    /// `frame_codes [batch, num_code_groups]` int64 `[+ text_embed [batch, 1,
    /// hidden]]` → output `inputs_embeds [batch, 1, hidden]`), the runtime builds
    /// the outer decoder's per-step `inputs_embeds` through it, keeping the engine
    /// generic. Requires a dataflow edge
    /// `{pre_embedder}.inputs_embeds -> {outer}.inputs_embeds`.
    ///
    /// When absent the outer loop is `input_ids`-driven (backward compatible).
    ///
    /// All graph-specific port bindings (the pre-embedder's `frame_codes` /
    /// optional `text_embed` inputs and the output feeding the outer decoder)
    /// are declared explicitly in [`PreEmbedderSpec`]; the runtime never guesses
    /// them by tensor name or dtype.
    #[serde(default)]
    pub pre_embedder: Option<PreEmbedderSpec>,

    /// Optional prefill embedder component that supplies the outer decoder
    /// (talker) with its real frame-0 PREFILL sequence and the per-frame
    /// trailing-text conditioning of a `nested_autoregressive` stage.
    ///
    /// The talker is prefilled with a multi-position embedding
    /// sequence built from the tokenized prompt, and each subsequent frame is
    /// conditioned on one trailing-text embedding. This component materializes
    /// both from `text_ids`: inputs `text_ids [batch, text_len]` int64 → outputs
    /// `prefill_embeds [batch, prefill_len, hidden]` float (fed DIRECTLY to the
    /// talker's `inputs_embeds` on frame 0) and `trailing_text_embeds [batch,
    /// trailing_len, hidden]` float (one vector consumed per outer frame `k >= 1`
    /// as the pre-embedder's `text_embed`). It runs once in the prompt phase
    /// (`run_on: prompt_only`); its `text_ids` input is auto-seeded from the
    /// tokenized prompt.
    ///
    /// Only meaningful together with [`Self::pre_embedder`] (the frame-`k >= 1`
    /// path feeds the trailing-text vectors through it). When absent, frame 0
    /// uses a zero seed and every `text_embed` is zero (backward compatible).
    ///
    /// All graph-specific port bindings (the prompt input plus the prefill and
    /// trailing-text outputs) are declared explicitly in [`PrefillEmbedderSpec`];
    /// the runtime never guesses them by tensor name or dtype.
    #[serde(default)]
    pub prefill_embedder: Option<PrefillEmbedderSpec>,
}

/// Structured binding for the optional pre-embedder that drives the outer
/// decoder (talker) of a `nested_autoregressive` stage via `inputs_embeds`.
///
/// Every graph-specific port the runtime touches is declared here, so the
/// engine never infers a port by tensor name or dtype.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct PreEmbedderSpec {
    /// Declared model name of the pre-embedder component.
    #[schemars(length(min = 1))]
    pub component: String,

    /// Pre-embedder input port receiving the previous frame's codes
    /// (`int64 [batch, num_code_groups]`).
    #[schemars(length(min = 1))]
    pub frame_codes_input: String,

    /// Optional pre-embedder input port receiving the per-frame trailing-text
    /// conditioning vector (`float [batch, 1, hidden]`). When absent, the
    /// pre-embedder exposes no trailing-text input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_embed_input: Option<String>,
}

/// Structured binding for the optional prefill embedder that supplies the outer
/// decoder (talker) of a `nested_autoregressive` stage with its frame-0 PREFILL
/// sequence and per-frame trailing-text conditioning.
///
/// Every graph-specific port the runtime touches is declared here, so the
/// engine never infers a port by tensor name or dtype.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct PrefillEmbedderSpec {
    /// Declared model name of the (prompt-phase) prefill embedder component.
    #[schemars(length(min = 1))]
    pub component: String,

    /// Prefill-embedder input port receiving the tokenized prompt
    /// (`int64 [batch, text_len]`, e.g. `text_ids`).
    #[schemars(length(min = 1))]
    pub prompt_input: String,

    /// Prefill-embedder output port carrying the talker's frame-0 multi-position
    /// PREFILL sequence (`float [batch, prefill_len, hidden]`), fed DIRECTLY to
    /// the outer decoder's `inputs_embeds` on frame 0.
    #[schemars(length(min = 1))]
    pub prefill_output: String,

    /// Prefill-embedder output port carrying the per-frame trailing-text vectors
    /// (`float [batch, trailing_len, hidden]`), one sliced per outer frame
    /// `k >= 1` into the pre-embedder's `text_embed`.
    #[schemars(length(min = 1))]
    pub trailing_output: String,
}

/// Named child stage of a composite pipeline strategy.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(deny_unknown_fields)]
pub struct PipelineStrategyStage {
    /// Non-empty stage name unique within its containing composite.
    #[schemars(length(min = 1))]
    pub name: String,

    /// Execution strategy for this stage.
    pub strategy: Box<PipelineStrategy>,
}
