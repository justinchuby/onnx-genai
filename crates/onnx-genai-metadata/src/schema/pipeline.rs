use super::*;

/// Internal decoder position program retained for the bare single-model engine.
///
/// Workflow packages express position tensors through ordinary typed values and
/// component invocations; this type is not referenced by `PipelineSpec` and is
/// therefore not part of the workflow metadata schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionProgram {
    pub input: String,
    pub rank: usize,
    pub tensor_rank: Option<usize>,
    pub generation: Option<String>,
    pub axes: Option<Vec<String>>,
    pub sections: Option<Vec<usize>>,
    pub dtype: Option<String>,
    pub continuation: Option<String>,
    pub processor_summaries: Option<Vec<String>>,
}

/// Executable package described by the universal typed workflow IR.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(deny_unknown_fields)]
pub struct PipelineSpec {
    /// Required component-centric SSA workflow.
    pub workflow: WorkflowSpec,
}

/// Declared, architecture-neutral input preprocessing programs.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, JsonSchema)]
pub struct PreprocessingSpec {
    /// Typed still-image preprocessing program and its named tensor outputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<VisionPreprocessingProgram>,

    /// Typed video preprocessing program and its named tensor outputs.
    ///
    /// A video is a still image plus a temporal axis, so it declares the same
    /// program type: the spatial transforms and their parameters are identical,
    /// and the temporal ones (`sample_frames`, `pad_frames`) are further
    /// operations in the same generic vocabulary. Splitting the two would
    /// duplicate every spatial parameter for no gain, while keeping them one
    /// key would hide that a package preprocesses stills and clips differently.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video: Option<VisionPreprocessingProgram>,

    /// Typed audio preprocessing transform program and its named tensor outputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioPreprocessingProgram>,
}

/// Generic pixel preprocessing program: an ordered transform pipeline plus the
/// named workflow SSA tensor outputs it emits.
///
/// The program is expressed entirely as parameterized, architecture-neutral
/// data. Transform operations are generic (decode, resize, rescale, normalize,
/// tile, patchify, pad, and for clips sample_frames and pad_frames). In workflow
/// metadata, outputs are materialized by a manifest-pinned preprocessing adapter
/// invocation and bind processor-local values to typed SSA names. A package may
/// name an output `pixel_position_ids`, `image_grid_thw`, `video_grid_thw`, or
/// anything else without introducing runtime model-family dispatch.
///
/// One program type covers stills and clips, the way one audio program covers
/// every audio family: a frame is an image, so a video program declares the
/// same spatial work plus the temporal operations that select and pad frames.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct VisionPreprocessingProgram {
    /// Ordered list of generic transform operations applied to decoded pixels.
    #[serde(default)]
    pub transforms: Vec<VisionTransform>,

    /// Named tensor outputs the program emits, each bound to a workflow SSA value.
    #[schemars(length(min = 1))]
    pub outputs: Vec<VisionOutputBinding>,
}

/// One generic pixel transform operation.
///
/// `op` selects the operation from a generic vocabulary; the remaining fields
/// are the parameters that operation reads (only the relevant ones are set).
/// Every parameter is model DATA — concrete sizes, patch sizes, means, and so on
/// live in a model's fixture, never as constants baked into this schema.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct VisionTransform {
    /// Generic operation selector (e.g. `resize`, `normalize`, `patchify`).
    #[schemars(with = "schema_vocabulary::VisionTransformOp")]
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
    /// through `VisionOutputBinding::source`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1), inner(length(min = 1)))]
    pub outputs: Option<Vec<String>>,

    /// Target size for a `resize` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<VisionSizeSpec>,

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

    /// Relative nesting of the temporal and channel axes inside a flattened
    /// `channels_first` patch (`channel_major` or `temporal_major`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub temporal_order: Option<String>,
    /// Order patches are emitted in (`merge_groups`, the default, or `raster`).
    /// Independent of `merge_size`, which only sets how many patches collapse
    /// into one vision token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub patch_order: Option<String>,

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

    /// Exact first-axis length produced by a `pad` operation, or the exact frame
    /// count produced by a `pad_frames` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub target_length: Option<usize>,

    /// Frame rate a `sample_frames` operation resamples a clip to.
    ///
    /// Temporal parameters, like every other parameter here, are model DATA: a
    /// package that preprocesses clips states its own sampling rate and frame
    /// budget instead of a runtime inferring them from a model family.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,

    /// Exact number of frames a `sample_frames` operation selects from a clip.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub num_frames: Option<usize>,

    /// Stride, in decoded frames, between the frames `sample_frames` selects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub frame_stride: Option<usize>,
}

/// A square size or an explicit width/height for a pixel transform.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum VisionSizeSpec {
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

/// One named tensor output produced by a pixel preprocessing program.
///
/// The output binds a processor-local value to a typed workflow SSA name.
/// Neither the name nor the content role is inferred from a model identity.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct VisionOutputBinding {
    /// Named processor-local value produced by a transform.
    #[schemars(length(min = 1))]
    pub source: String,

    /// Workflow SSA value produced by the preprocessing adapter invocation.
    #[schemars(length(min = 1), example = &"image.pixel_values")]
    pub name: String,

    /// Generic content role this tensor carries (pixels, coordinates, grid,
    /// original size, validity mask, or the offsets/owner map of a packed batch
    /// at item, frame, or clip granularity) — never a model-family label.
    #[schemars(with = "schema_vocabulary::VisionOutputContent")]
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

/// Generic audio preprocessing program: an ordered transform pipeline plus the
/// named workflow SSA tensor outputs it emits.
///
/// The program is expressed entirely as parameterized, architecture-neutral
/// data, mirroring `VisionPreprocessingProgram`. Transform operations are
/// generic (decode, resample, downmix, rescale, normalize, pad, frame,
/// spectrogram, log_mel). In workflow metadata, outputs are materialized by a
/// manifest-pinned preprocessing adapter invocation and bind processor-local
/// values to typed SSA names. A package may name an output `input_values`,
/// `input_features`, `attention_mask`, or anything else without introducing
/// runtime model-family dispatch.
///
/// One program type covers every audio family. A CTC acoustic model declares
/// resample/downmix/zero_mean_unit_variance over raw samples; an
/// encoder-decoder speech model declares resample/pad/log_mel over a fixed
/// window. The runtime reads the same fields either way.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct AudioPreprocessingProgram {
    /// Ordered list of generic transform operations applied to decoded audio.
    #[serde(default)]
    pub transforms: Vec<AudioTransform>,

    /// Named tensor outputs the program emits, each bound to a workflow SSA value.
    #[schemars(length(min = 1))]
    pub outputs: Vec<AudioOutputBinding>,
}

/// One generic audio transform operation.
///
/// `op` selects the operation from a generic vocabulary; the remaining fields
/// are the parameters that operation reads (only the relevant ones are set).
/// Every parameter is model DATA — concrete sample rates, channel counts, mel
/// bin counts, and so on live in a model's fixture, never as constants baked
/// into this schema.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct AudioTransform {
    /// Generic operation selector (e.g. `resample`, `zero_mean_unit_variance`,
    /// `log_mel`).
    #[schemars(with = "schema_vocabulary::AudioTransformOp")]
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
    /// through `AudioOutputBinding::source`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1), inner(length(min = 1)))]
    pub outputs: Option<Vec<String>>,

    /// Target sample rate in Hz for a `resample` operation, and the analysis
    /// rate a mel filterbank is built for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub sample_rate: Option<u32>,

    /// Target channel count for a `downmix` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub channels: Option<usize>,

    /// Numerical stabilizer added to the variance for `zero_mean_unit_variance`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epsilon: Option<f64>,

    /// Scalar multiplier for a `rescale` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,

    /// Padding side / normalization mode selector — generic string data
    /// (e.g. `right`, `left`, `fixed_window`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub mode: Option<String>,

    /// Fill value used by a `pad` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pad_value: Option<f64>,

    /// Fixed target length in samples or frames for a `pad`/`trim` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub target_length: Option<usize>,

    /// Mel filterbank size for a `log_mel`/`log_mel_spectrogram` operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub num_mel_bins: Option<usize>,

    /// FFT size for a spectrogram operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub n_fft: Option<usize>,

    /// Hop length in samples for a spectrogram operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub hop_length: Option<usize>,

    /// Analysis window length in samples for a spectrogram operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub win_length: Option<usize>,

    /// Analysis window function — generic string data (e.g. `hann`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub window: Option<String>,

    /// Mel scale convention — generic string data (e.g. `slaney`, `htk`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub mel_scale: Option<String>,
}

/// One named tensor output produced by an audio preprocessing program.
///
/// The output binds a processor-local value to a typed workflow SSA name.
/// Neither the name nor the content role is inferred from a model identity.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
pub struct AudioOutputBinding {
    /// Workflow SSA value this output binds to.
    #[schemars(length(min = 1), example = &"audio.input_features")]
    pub name: String,

    /// Program-local value produced by a transform.
    #[schemars(length(min = 1))]
    pub source: String,

    /// Generic content role of this output.
    #[schemars(with = "schema_vocabulary::AudioOutputContent")]
    pub content: String,

    /// Element type of the emitted tensor.
    #[schemars(with = "schema_vocabulary::TensorDType")]
    pub dtype: String,

    /// Full workflow tensor contract. Required when `pipeline.workflow` is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract: Option<crate::schema::TensorContract>,

    /// Optional sentinel/pad value for padded entries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pad_value: Option<f64>,

    /// Whether the runtime may omit this output when a model does not need it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optional: Option<bool>,
}
