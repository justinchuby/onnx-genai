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
