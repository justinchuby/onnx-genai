use std::{collections::BTreeMap, path::Path};

use anyhow::Context;
use image::DynamicImage;
use onnx_genai_metadata::{
    ImageOutputBinding, ImagePreprocessingProgram, ImageSizeSpec, ImageTransform,
};
use serde::Deserialize;

use super::{
    CHANNELS, ImageLayout, ImagePreprocessConfig, ImageTensorBundle, ImageTensorDType,
    ImageTilingConfig, Interpolation, MAX_ASPECT_RATIOS, MAX_IMAGE_COUNT, MAX_IMAGE_OUTPUTS,
    MAX_IMAGE_PIXELS, MAX_IMAGE_TRANSFORMS, MAX_TENSOR_ELEMENTS, MAX_TILES_PER_IMAGE,
    Normalization, ResizeMode, ThumbnailPosition, TileGrid, TilingMode, packed, tiling::tile_image,
    transform::normalize_tile,
};
use packed::{PackSpec, PreparedImage};

/// Reusable image preprocessor resolved from a model input and §35 metadata.
#[derive(Debug, Clone)]
pub struct ImagePreprocessor {
    shape: Vec<i64>,
    layout: ImageLayout,
    config: ImagePreprocessConfig,
    program: ImageProgram,
}

#[derive(Debug, Deserialize)]
struct MetadataDocument {
    preprocessing: Option<PreprocessingMetadata>,
}

#[derive(Debug, Deserialize)]
struct PreprocessingMetadata {
    image: Option<ImageMetadata>,
}

#[derive(Debug, Deserialize)]
struct ImageMetadata {
    resize: Option<ResizeMetadata>,
    tiling: Option<TilingMetadata>,
    normalize: Option<NormalizeMetadata>,
    #[serde(default)]
    transforms: Vec<ImageTransformMetadata>,
    #[serde(default)]
    outputs: Vec<ImageOutputMetadata>,
}

#[derive(Debug, Deserialize)]
struct ResizeMetadata {
    mode: Option<String>,
    size: Option<ImageSize>,
    interpolation: Option<String>,
    crop: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ImageSize {
    Square(u32),
    Dimensions { width: u32, height: u32 },
}

#[derive(Debug, Deserialize)]
struct NormalizeMetadata {
    mean: [f32; 3],
    std: [f32; 3],
}

#[derive(Debug, Deserialize)]
struct TilingMetadata {
    mode: Option<String>,
    tile_size: Option<u32>,
    max_tiles: Option<usize>,
    aspect_ratios: Option<Vec<[u32; 2]>>,
    include_thumbnail: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ImageTransformMetadata {
    op: String,
    inputs: Option<Vec<String>>,
    outputs: Option<Vec<String>>,
    size: Option<ImageSize>,
    mode: Option<String>,
    interpolation: Option<String>,
    min_pixels: Option<usize>,
    max_pixels: Option<usize>,
    size_multiple: Option<usize>,
    max_patches: Option<usize>,
    pooling_kernel_size: Option<usize>,
    scale: Option<f64>,
    mean: Option<Vec<f32>>,
    std: Option<Vec<f32>>,
    tile_size: Option<usize>,
    max_tiles: Option<usize>,
    include_thumbnail: Option<bool>,
    thumbnail_order: Option<String>,
    thumbnail_interpolation: Option<String>,
    canvas_pad_value: Option<f64>,
    mask_patch_size: Option<usize>,
    patch_size: Option<usize>,
    temporal_patch_size: Option<usize>,
    merge_size: Option<usize>,
    channel_order: Option<String>,
    temporal_order: Option<String>,
    patch_order: Option<String>,
    coordinate_order: Option<String>,
    flatten: Option<bool>,
    pad_value: Option<f64>,
    target_length: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ImageOutputMetadata {
    source: Option<String>,
    name: String,
    content: String,
    dtype: String,
    pad_value: Option<f64>,
    optional: Option<bool>,
}

#[derive(Debug, Clone)]
pub(super) struct ImageProgram {
    value_ops: Vec<ValueOp>,
    named_value_ops: Option<BTreeMap<String, Vec<ValueOp>>>,
    patchify: Option<PatchifySpec>,
    pad_value: Option<f64>,
    target_length: Option<usize>,
    pub(super) dynamic_resize: Option<DynamicResize>,
    pub(super) dynamic_hd: Option<DynamicHdSpec>,
    outputs: Vec<OutputSpec>,
}

#[derive(Debug, Clone)]
struct OutputSpec {
    source: Option<String>,
    packed: packed::OutputSpec,
}

#[derive(Debug, Clone)]
pub(super) struct PatchifySpec {
    pub(super) patch_size: usize,
    pub(super) temporal_patch_size: usize,
    pub(super) merge_size: usize,
    pub(super) channel_order: PatchChannelOrder,
    pub(super) temporal_order: PatchTemporalOrder,
    pub(super) patch_order: PatchOrder,
    pub(super) coordinate_order: CoordinateOrder,
}

/// The order patches are emitted in, which is independent of `merge_size`.
///
/// Qwen2-VL packs each `merge_size x merge_size` spatial group contiguously, so
/// the model's merge reshape sees a whole group per row. Some exports instead
/// expect plain row-major patch order and do the grouping inside the graph.
/// `merge_size` still governs how many patches collapse into one image token,
/// so the two knobs cannot be folded together: emitting raster order by setting
/// `merge_size: 1` would also quadruple the placeholder count.
#[derive(Debug, Clone, Copy)]
pub(super) enum PatchOrder {
    MergeGroups,
    Raster,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum PatchChannelOrder {
    ChannelsFirst,
    ChannelsLast,
}

/// Where the temporal axis sits relative to the channel axis inside one
/// flattened `channels_first` patch.
///
/// Qwen2-VL repeats each frame inside its channel block, giving `[C, T, H, W]`.
/// Muse Glimmer keeps whole frames contiguous instead, giving `[T, C, H, W]`.
/// The two agree only when a model has a single temporal frame, and feeding one
/// layout to a model trained on the other scrambles colour while leaving spatial
/// structure intact, so the packer cannot guess.
#[derive(Debug, Clone, Copy)]
pub(super) enum PatchTemporalOrder {
    ChannelMajor,
    TemporalMajor,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum CoordinateOrder {
    Yx,
    Xy,
}

#[derive(Debug, Clone)]
pub(super) enum DynamicResize {
    PixelArea {
        min_pixels: usize,
        max_pixels: usize,
        size_multiple: usize,
    },
    PatchBudget {
        patch_size: usize,
        max_patches: usize,
        pooling_kernel_size: usize,
    },
}

#[derive(Debug, Clone)]
pub(super) struct DynamicHdSpec {
    pub(super) mask_patch_size: usize,
    pub(super) canvas_pad_value: u8,
    pub(super) thumbnail_interpolation: Interpolation,
}

#[derive(Debug, Clone)]
pub(super) enum ValueOp {
    Divide(f32),
    Rescale(f32),
    Normalize { mean: [f32; 3], std: [f32; 3] },
}

impl ImagePreprocessor {
    /// Resolves preprocessing directly from a typed metadata program.
    pub fn from_input_and_program(
        shape: &[i64],
        program: &ImagePreprocessingProgram,
    ) -> anyhow::Result<Self> {
        Self::from_metadata_document(
            shape,
            Some(MetadataDocument {
                preprocessing: Some(PreprocessingMetadata {
                    image: Some(Self::image_metadata_from_program(program)),
                }),
            }),
        )
    }

    /// Resolves preprocessing from a model pixel input and optional metadata file.
    pub fn from_input_and_metadata(
        shape: &[i64],
        metadata_path: Option<&Path>,
    ) -> anyhow::Result<Self> {
        let document = metadata_path
            .map(std::fs::read_to_string)
            .transpose()
            .context("failed to read preprocessing metadata")?
            .map(|content| {
                serde_yaml::from_str::<MetadataDocument>(&content)
                    .context("failed to parse preprocessing metadata")
            })
            .transpose()?;
        Self::from_metadata_document(shape, document)
    }

    fn image_metadata_from_program(program: &ImagePreprocessingProgram) -> ImageMetadata {
        ImageMetadata {
            resize: None,
            tiling: None,
            normalize: None,
            transforms: program
                .transforms
                .iter()
                .map(Self::image_transform_metadata)
                .collect(),
            outputs: program
                .outputs
                .iter()
                .map(Self::image_output_metadata)
                .collect(),
        }
    }

    fn image_transform_metadata(transform: &ImageTransform) -> ImageTransformMetadata {
        ImageTransformMetadata {
            op: transform.op.clone(),
            inputs: transform.inputs.clone(),
            outputs: transform.outputs.clone(),
            size: transform.size.as_ref().map(|size| match size {
                ImageSizeSpec::Square(edge) => ImageSize::Square(*edge),
                ImageSizeSpec::Dimensions { width, height } => ImageSize::Dimensions {
                    width: *width,
                    height: *height,
                },
            }),
            mode: transform.mode.clone(),
            interpolation: transform.interpolation.clone(),
            min_pixels: transform.min_pixels,
            max_pixels: transform.max_pixels,
            size_multiple: transform.size_multiple,
            max_patches: transform.max_patches,
            pooling_kernel_size: transform.pooling_kernel_size,
            scale: transform.scale,
            mean: transform.mean.clone(),
            std: transform.std.clone(),
            tile_size: transform.tile_size,
            max_tiles: transform.max_tiles,
            include_thumbnail: transform.include_thumbnail,
            thumbnail_order: transform.thumbnail_order.clone(),
            thumbnail_interpolation: transform.thumbnail_interpolation.clone(),
            canvas_pad_value: transform.canvas_pad_value,
            mask_patch_size: transform.mask_patch_size,
            patch_size: transform.patch_size,
            temporal_patch_size: transform.temporal_patch_size,
            merge_size: transform.merge_size,
            channel_order: transform.channel_order.clone(),
            temporal_order: transform.temporal_order.clone(),
            patch_order: transform.patch_order.clone(),
            coordinate_order: transform.coordinate_order.clone(),
            flatten: transform.flatten,
            pad_value: transform.pad_value,
            target_length: transform.target_length,
        }
    }

    fn image_output_metadata(output: &ImageOutputBinding) -> ImageOutputMetadata {
        ImageOutputMetadata {
            source: output.source.clone(),
            name: output.name.clone(),
            content: output.content.clone(),
            dtype: output.dtype.clone(),
            pad_value: output.pad_value,
            optional: output.optional,
        }
    }

    fn from_metadata_document(
        shape: &[i64],
        document: Option<MetadataDocument>,
    ) -> anyhow::Result<Self> {
        if shape.is_empty() {
            anyhow::bail!("vision pixel input shape must not be empty");
        }
        if shape
            .iter()
            .any(|dimension| *dimension == 0 || *dimension < -1)
        {
            anyhow::bail!("vision pixel input shape contains an invalid dimension: {shape:?}");
        }
        let metadata = document
            .and_then(|document| document.preprocessing)
            .and_then(|preprocessing| preprocessing.image);
        let is_typed_program = metadata
            .as_ref()
            .is_some_and(|image| !image.transforms.is_empty() || !image.outputs.is_empty());
        let (layout, model_width, model_height) = if shape.len() == 4 {
            let layout = match (shape[1], shape[3]) {
                (3, _) => ImageLayout::Nchw,
                (_, 3) => ImageLayout::Nhwc,
                _ if is_typed_program => ImageLayout::Nchw,
                _ => anyhow::bail!(
                    "vision input must declare an RGB channel dimension, but the model declares {shape:?}"
                ),
            };
            let (height, width) = match layout {
                ImageLayout::Nchw => (shape[2], shape[3]),
                ImageLayout::Nhwc => (shape[1], shape[2]),
            };
            (layout, width, height)
        } else if is_typed_program {
            (ImageLayout::Nchw, -1, -1)
        } else {
            anyhow::bail!(
                "legacy image preprocessing requires a rank-4 vision input, but the model declares {shape:?}; packed inputs require preprocessing.image.transforms and outputs"
            );
        };
        let (config, program) = if is_typed_program {
            typed_program_from_metadata(
                metadata.context("typed image preprocessing metadata is missing")?,
                model_width,
                model_height,
            )?
        } else {
            let config = preprocessing_from_metadata(metadata, model_width, model_height)?;
            let program = legacy_program(&config)?;
            (config, program)
        };
        let mut resolved_shape = shape.to_vec();
        if resolved_shape.len() == 4 {
            match layout {
                ImageLayout::Nchw => {
                    resolved_shape[2] = i64::from(config.height);
                    resolved_shape[3] = i64::from(config.width);
                }
                ImageLayout::Nhwc => {
                    resolved_shape[1] = i64::from(config.height);
                    resolved_shape[2] = i64::from(config.width);
                }
            }
        }
        Ok(Self {
            shape: resolved_shape,
            layout,
            config,
            program,
        })
    }

    /// Resolves preprocessing using model dimensions and default §35 behavior.
    pub fn from_input(shape: &[i64]) -> anyhow::Result<Self> {
        Self::from_input_and_metadata(shape, None)
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn layout(&self) -> ImageLayout {
        self.layout
    }

    pub fn config(&self) -> &ImagePreprocessConfig {
        &self.config
    }

    /// Decodes encoded images and preprocesses them into a named tensor bundle.
    pub fn preprocess_encoded<I, B>(&self, images: I) -> anyhow::Result<ImageTensorBundle>
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        let mut decoded = Vec::new();
        for (image_index, bytes) in images.into_iter().enumerate() {
            if image_index == MAX_IMAGE_COUNT {
                anyhow::bail!(
                    "image batch contains more than the supported limit of {MAX_IMAGE_COUNT} images; split the request into smaller batches"
                );
            }
            decoded
                .try_reserve(1)
                .context("failed to allocate decoded image batch")?;
            decoded.push(
                image::load_from_memory(bytes.as_ref())
                    .with_context(|| format!("failed to decode image {image_index}"))?,
            );
        }
        self.preprocess(&decoded)
    }

    /// Preprocesses decoded images into a named typed tensor bundle.
    pub fn preprocess(&self, images: &[DynamicImage]) -> anyhow::Result<ImageTensorBundle> {
        if images.is_empty() {
            anyhow::bail!("at least one image is required");
        }
        if images.len() > MAX_IMAGE_COUNT {
            anyhow::bail!(
                "image batch contains {} images, exceeding the supported limit of {MAX_IMAGE_COUNT}; split the request into smaller batches",
                images.len()
            );
        }

        if let Some(named_value_ops) = &self.program.named_value_ops {
            let mut merged = None;
            let mut total_elements = 0usize;
            for output in &self.program.outputs {
                let source = output.source.as_deref().context(
                    "explicitly named image output lost its source during program compilation",
                )?;
                let value_ops = named_value_ops.get(source).with_context(|| {
                    format!(
                        "image output '{}' selects unknown compiled source '{source}'",
                        output.packed.name
                    )
                })?;
                let mut branch =
                    self.preprocess_with_ops(images, value_ops, vec![output.packed.clone()])?;
                total_elements =
                    branch
                        .tensors
                        .iter()
                        .try_fold(total_elements, |total, tensor| {
                            total
                                .checked_add(tensor.data.len())
                                .context("total image output element count overflowed")
                        })?;
                if total_elements > MAX_TENSOR_ELEMENTS {
                    anyhow::bail!(
                        "image output bundle requires {total_elements} elements across declared tensors, exceeding the safety limit of {MAX_TENSOR_ELEMENTS}; reduce image dimensions, tile count, patch count, batch size, or duplicate pixel outputs"
                    );
                }
                match &mut merged {
                    Some(bundle) => {
                        ensure_compatible_output_branch(bundle, &branch, &output.packed.name)?;
                        bundle.tensors.append(&mut branch.tensors);
                    }
                    None => merged = Some(branch),
                }
            }
            return merged.context("preprocessing.image.outputs must contain at least one output");
        }

        self.preprocess_with_ops(
            images,
            &self.program.value_ops,
            self.program
                .outputs
                .iter()
                .map(|output| output.packed.clone())
                .collect(),
        )
    }

    fn preprocess_with_ops(
        &self,
        images: &[DynamicImage],
        value_ops: &[ValueOp],
        outputs: Vec<packed::OutputSpec>,
    ) -> anyhow::Result<ImageTensorBundle> {
        let mut prepared_elements = 0usize;
        let mut prepared = Vec::new();
        prepared
            .try_reserve_exact(images.len())
            .context("failed to allocate prepared image batch")?;
        for (image_index, image) in images.iter().enumerate() {
            validate_source_image(image, image_index)?;
            let tiled = tile_image(image, &self.config, &self.program)?;
            let grid = tiled.grid;
            let image_tiles = tiled.tiles;
            if image_tiles.len() > MAX_TILES_PER_IMAGE + 1 {
                anyhow::bail!(
                    "image {image_index} produced {} tiles, exceeding the supported limit of {}; reduce max_tiles",
                    image_tiles.len(),
                    MAX_TILES_PER_IMAGE + 1
                );
            }
            let image_elements = image_tiles.iter().try_fold(0usize, |total, tile| {
                let elements = checked_image_elements(
                    tile.width() as usize,
                    tile.height() as usize,
                    "normalized image tile",
                )?;
                total
                    .checked_add(elements)
                    .context("prepared image element count overflowed")
            })?;
            prepared_elements = prepared_elements
                .checked_add(image_elements)
                .context("prepared image batch element count overflowed")?;
            if prepared_elements > MAX_TENSOR_ELEMENTS {
                anyhow::bail!(
                    "prepared image batch requires {prepared_elements} fp32 elements, exceeding the safety limit of {MAX_TENSOR_ELEMENTS}; reduce image dimensions, tile count, or batch size"
                );
            }
            let mut tiles = Vec::new();
            tiles
                .try_reserve_exact(image_tiles.len())
                .context("failed to allocate normalized image tiles")?;
            for tile in &image_tiles {
                tiles.push(normalize_tile(
                    tile,
                    tile.width() as usize,
                    tile.height() as usize,
                    value_ops,
                )?);
            }
            prepared.push(PreparedImage {
                original_size: (image.width(), image.height()),
                transformed_size: tiled.transformed_size,
                tile_grid: grid,
                tile_size: image_tiles
                    .first()
                    .map(|tile| (tile.width() as usize, tile.height() as usize))
                    .context("image preprocessing produced no tiles")?,
                tiles,
                validity_masks: tiled.validity_masks,
            });
        }
        packed::build_bundle(
            prepared,
            &PackSpec {
                layout: self.layout,
                patchify: self.program.patchify.clone(),
                pad_value: self.program.pad_value,
                target_length: self.program.target_length,
                outputs,
                declared_pixel_shape: self.shape.clone(),
            },
            self.config.tiling.thumbnail_position,
        )
    }
}

fn ensure_compatible_output_branch(
    expected: &ImageTensorBundle,
    actual: &ImageTensorBundle,
    output_name: &str,
) -> anyhow::Result<()> {
    if expected.images != actual.images
        || expected.num_tiles != actual.num_tiles
        || expected.tiles_per_image != actual.tiles_per_image
        || expected.tile_grids != actual.tile_grids
        || expected.thumbnail_position != actual.thumbnail_position
    {
        anyhow::bail!(
            "image output '{output_name}' selects a branch with incompatible image ordering or geometry"
        );
    }
    Ok(())
}

fn legacy_program(config: &ImagePreprocessConfig) -> anyhow::Result<ImageProgram> {
    let value_ops = match &config.normalization {
        Normalization::ZeroToOne => vec![ValueOp::Divide(255.0)],
        Normalization::MeanStd { mean, std } => vec![
            ValueOp::Divide(255.0),
            ValueOp::Normalize {
                mean: *mean,
                std: *std,
            },
        ],
    };
    Ok(ImageProgram {
        value_ops,
        named_value_ops: None,
        patchify: None,
        pad_value: None,
        target_length: None,
        dynamic_resize: None,
        dynamic_hd: None,
        outputs: vec![OutputSpec {
            source: None,
            packed: packed::OutputSpec {
                name: "pixels".to_owned(),
                content: "pixels".to_owned(),
                dtype: ImageTensorDType::Fp32,
                pad_value: None,
                optional: false,
            },
        }],
    })
}

fn typed_program_from_metadata(
    metadata: ImageMetadata,
    model_width: i64,
    model_height: i64,
) -> anyhow::Result<(ImagePreprocessConfig, ImageProgram)> {
    if metadata.transforms.len() > MAX_IMAGE_TRANSFORMS {
        anyhow::bail!(
            "preprocessing.image.transforms contains {} entries, exceeding the supported limit of {MAX_IMAGE_TRANSFORMS}",
            metadata.transforms.len()
        );
    }
    if metadata.outputs.len() > MAX_IMAGE_OUTPUTS {
        anyhow::bail!(
            "preprocessing.image.outputs contains {} entries, exceeding the supported limit of {MAX_IMAGE_OUTPUTS}",
            metadata.outputs.len()
        );
    }
    if metadata.transforms.is_empty() {
        anyhow::bail!(
            "preprocessing.image.transforms must not be empty when typed image outputs are declared"
        );
    }
    if metadata.outputs.is_empty() {
        anyhow::bail!(
            "preprocessing.image.outputs must not be empty when typed image transforms are declared"
        );
    }
    if metadata.resize.is_some() || metadata.tiling.is_some() || metadata.normalize.is_some() {
        anyhow::bail!(
            "preprocessing.image cannot mix legacy resize/tiling/normalize fields with typed transforms"
        );
    }
    let explicit_dataflow = validate_program_dataflow(&metadata.transforms, &metadata.outputs)?;
    let named_value_ops = explicit_dataflow
        .then(|| compile_named_value_ops(&metadata.transforms))
        .transpose()?;

    let mut resize = None;
    let mut tiling = None;
    let mut value_ops = Vec::new();
    let mut patchify = None;
    let mut pad_value = None;
    let mut target_length = None;
    let mut dynamic_resize = None;
    let mut dynamic_hd = None;
    let mut decoded = false;
    let mut flattened = false;
    let mut patchified = false;
    let mut padded = false;
    for transform in metadata.transforms {
        match transform.op.as_str() {
            "decode" | "decode_rgb" => {
                if decoded || resize.is_some() || !value_ops.is_empty() || patchified || padded {
                    anyhow::bail!("decode_rgb must be the first image transform");
                }
                decoded = true;
            }
            "convert_rgb" => {
                if !decoded || resize.is_some() || !value_ops.is_empty() || patchified || padded {
                    anyhow::bail!("convert_rgb must follow decode and precede image transforms");
                }
            }
            "resize" => {
                if resize.is_some()
                    || tiling.is_some()
                    || !value_ops.is_empty()
                    || patchified
                    || padded
                {
                    anyhow::bail!(
                        "resize must occur once and before tile, rescale, normalize, patchify, or pad"
                    );
                }
                let mode_name = transform.mode.as_deref().unwrap_or("stretch");
                let mode = match mode_name {
                    "stretch" | "fixed" | "fixed_size" => ResizeMode::Fixed,
                    "crop" | "shortest_edge" | "shortest_edge_center_crop" => {
                        ResizeMode::ShortestEdgeCenterCrop
                    }
                    "pad" | "longest_edge_pad" => ResizeMode::LongestEdgePad,
                    "pixel_area" => {
                        let min_pixels = transform
                            .min_pixels
                            .context("pixel_area resize requires min_pixels metadata")?;
                        let max_pixels = transform
                            .max_pixels
                            .context("pixel_area resize requires max_pixels metadata")?;
                        let size_multiple = transform
                            .size_multiple
                            .context("pixel_area resize requires size_multiple metadata")?;
                        if min_pixels == 0
                            || max_pixels == 0
                            || size_multiple == 0
                            || min_pixels > max_pixels
                        {
                            anyhow::bail!(
                                "pixel_area resize requires 0 < min_pixels <= max_pixels and size_multiple > 0"
                            );
                        }
                        dynamic_resize = Some(DynamicResize::PixelArea {
                            min_pixels,
                            max_pixels,
                            size_multiple,
                        });
                        ResizeMode::PixelArea
                    }
                    "aspect_ratio_patch_budget" => {
                        let patch_size = transform.patch_size.context(
                            "aspect_ratio_patch_budget resize requires patch_size metadata",
                        )?;
                        let max_patches = transform.max_patches.context(
                            "aspect_ratio_patch_budget resize requires max_patches metadata",
                        )?;
                        let pooling_kernel_size = transform.pooling_kernel_size.context(
                            "aspect_ratio_patch_budget resize requires pooling_kernel_size metadata",
                        )?;
                        if patch_size == 0 || max_patches == 0 || pooling_kernel_size == 0 {
                            anyhow::bail!(
                                "aspect_ratio_patch_budget resize parameters must be greater than zero"
                            );
                        }
                        dynamic_resize = Some(DynamicResize::PatchBudget {
                            patch_size,
                            max_patches,
                            pooling_kernel_size,
                        });
                        ResizeMode::PatchBudget
                    }
                    other => anyhow::bail!(
                        "unsupported image resize transform mode '{other}'; expected stretch, crop, pad, pixel_area, or aspect_ratio_patch_budget"
                    ),
                };
                let size = match mode {
                    ResizeMode::PixelArea | ResizeMode::PatchBudget => {
                        if transform.size.is_some() {
                            anyhow::bail!(
                                "{mode_name} resize computes its target from the source image and must not declare size"
                            );
                        }
                        ImageSize::Square(
                            u32::try_from(transform.size_multiple.unwrap_or_else(|| {
                                transform
                                    .patch_size
                                    .unwrap_or(1)
                                    .saturating_mul(transform.pooling_kernel_size.unwrap_or(1))
                            }))
                            .context("dynamic resize alignment is too large")?,
                        )
                    }
                    _ => transform
                        .size
                        .context("image resize transform requires size metadata")?,
                };
                let interpolation = parse_interpolation(transform.interpolation.as_deref())?;
                resize = Some((size, mode, interpolation));
            }
            "rescale" => {
                if patchified || padded {
                    anyhow::bail!("rescale must occur before patchify or pad");
                }
                let scale = transform
                    .scale
                    .context("image rescale transform requires scale metadata")?;
                let scale = scale as f32;
                if !scale.is_finite() {
                    anyhow::bail!("image rescale scale must be finite and representable as fp32");
                }
                value_ops.push(ValueOp::Rescale(scale));
            }
            "normalize" => {
                if patchified || padded {
                    anyhow::bail!("normalize must occur before patchify or pad");
                }
                let mean = channel_values("mean", transform.mean)?;
                let std = channel_values("std", transform.std)?;
                if mean.iter().any(|value| !value.is_finite())
                    || std.iter().any(|value| !value.is_finite() || *value <= 0.0)
                {
                    anyhow::bail!(
                        "image normalization mean/std values must be finite and std must be greater than zero"
                    );
                }
                value_ops.push(ValueOp::Normalize { mean, std });
            }
            "tile" => {
                if tiling.is_some() || patchified || padded {
                    anyhow::bail!("tile must occur once and before patchify or pad");
                }
                let tile_size = transform
                    .tile_size
                    .context("image tile transform requires tile_size metadata")?;
                if tile_size == 0 {
                    anyhow::bail!("image tile transform tile_size must be greater than zero");
                }
                let tile_size = u32::try_from(tile_size).context("image tile_size is too large")?;
                let max_tiles = transform.max_tiles.unwrap_or(6);
                if max_tiles == 0 {
                    anyhow::bail!("image tile transform max_tiles must be greater than zero");
                }
                if max_tiles > MAX_TILES_PER_IMAGE {
                    anyhow::bail!(
                        "image tile transform max_tiles {max_tiles} exceeds the supported limit of {MAX_TILES_PER_IMAGE}; reduce max_tiles"
                    );
                }
                let mode = match transform.mode.as_deref().unwrap_or("dynamic_anyres") {
                    "dynamic_anyres" => TilingMode::DynamicAnyres,
                    "fixed_grid" => TilingMode::FixedGrid,
                    "dynamic_hd" => TilingMode::DynamicHd,
                    other => anyhow::bail!(
                        "unsupported image tile transform mode '{other}'; expected dynamic_anyres, fixed_grid, or dynamic_hd"
                    ),
                };
                let thumbnail_position =
                    parse_thumbnail_position(transform.thumbnail_order.as_deref())?;
                let include_thumbnail = transform.include_thumbnail.unwrap_or(false);
                if include_thumbnail != (thumbnail_position != ThumbnailPosition::None) {
                    anyhow::bail!(
                        "image tile include_thumbnail={include_thumbnail} conflicts with thumbnail_order={thumbnail_position:?}"
                    );
                }
                if mode == TilingMode::DynamicHd {
                    let mask_patch_size = transform
                        .mask_patch_size
                        .context("dynamic_hd tile requires mask_patch_size metadata")?;
                    if mask_patch_size == 0 || !(tile_size as usize).is_multiple_of(mask_patch_size)
                    {
                        anyhow::bail!(
                            "dynamic_hd mask_patch_size must be greater than zero and divide tile_size"
                        );
                    }
                    let canvas_pad_value = exact_u8(
                        transform
                            .canvas_pad_value
                            .context("dynamic_hd tile requires canvas_pad_value metadata")?,
                        "dynamic_hd canvas_pad_value",
                    )?;
                    dynamic_hd = Some(DynamicHdSpec {
                        mask_patch_size,
                        canvas_pad_value,
                        thumbnail_interpolation: parse_interpolation(
                            transform.thumbnail_interpolation.as_deref(),
                        )?,
                    });
                }
                tiling = Some(ImageTilingConfig {
                    mode,
                    tile_size,
                    max_tiles,
                    aspect_ratios: default_anyres_grids(),
                    include_thumbnail,
                    thumbnail_position,
                });
            }
            "patchify" => {
                if patchified || padded {
                    anyhow::bail!("patchify must occur once and before pad");
                }
                flattened = transform.flatten.unwrap_or(true);
                let size = transform
                    .patch_size
                    .context("image patchify transform requires patch_size metadata")?;
                if size == 0 {
                    anyhow::bail!("image patchify patch_size must be greater than zero");
                }
                let temporal_patch_size = transform.temporal_patch_size.unwrap_or(1);
                let merge_size = transform.merge_size.unwrap_or(1);
                if temporal_patch_size == 0 || merge_size == 0 {
                    anyhow::bail!(
                        "image patchify temporal_patch_size and merge_size must be greater than zero"
                    );
                }
                let channel_order = match transform
                    .channel_order
                    .as_deref()
                    .unwrap_or("channels_first")
                {
                    "channels_first" | "chw" => PatchChannelOrder::ChannelsFirst,
                    "channels_last" | "hwc" => PatchChannelOrder::ChannelsLast,
                    other => anyhow::bail!(
                        "unsupported image patchify channel_order '{other}'; expected channels_first or channels_last"
                    ),
                };
                let temporal_order = match transform
                    .temporal_order
                    .as_deref()
                    .unwrap_or("channel_major")
                {
                    "channel_major" | "cthw" => PatchTemporalOrder::ChannelMajor,
                    "temporal_major" | "tchw" => PatchTemporalOrder::TemporalMajor,
                    other => anyhow::bail!(
                        "unsupported image patchify temporal_order '{other}'; expected channel_major or temporal_major"
                    ),
                };
                let patch_order = match transform.patch_order.as_deref().unwrap_or("merge_groups") {
                    "merge_groups" | "merge_blocks" => PatchOrder::MergeGroups,
                    "raster" | "row_major" => PatchOrder::Raster,
                    other => anyhow::bail!(
                        "unsupported image patchify patch_order '{other}'; expected merge_groups or raster"
                    ),
                };
                let coordinate_order = match transform.coordinate_order.as_deref().unwrap_or("yx") {
                    "yx" => CoordinateOrder::Yx,
                    "xy" => CoordinateOrder::Xy,
                    other => anyhow::bail!(
                        "unsupported image patchify coordinate_order '{other}'; expected yx or xy"
                    ),
                };
                patchify = Some(PatchifySpec {
                    patch_size: size,
                    temporal_patch_size,
                    merge_size,
                    channel_order,
                    temporal_order,
                    patch_order,
                    coordinate_order,
                });
                patchified = true;
            }
            "flatten" => {
                if !patchified || flattened || padded {
                    anyhow::bail!(
                        "flatten must follow one unflattened patchify transform and precede pad"
                    );
                }
                flattened = true;
            }
            "emit_original_size"
            | "emit_transformed_size"
            | "emit_validity_mask"
            | "emit_patch_coordinates"
            | "emit_grid_coordinates" => {}
            "pad" => {
                if !patchified {
                    anyhow::bail!("image pad transform requires a preceding patchify transform");
                }
                if padded {
                    anyhow::bail!("image pad transform may occur only once");
                }
                let value = transform.pad_value.unwrap_or(0.0);
                if !value.is_finite() {
                    anyhow::bail!("image pad transform pad_value must be finite");
                }
                pad_value = Some(value);
                target_length = Some(
                    transform
                        .target_length
                        .context("image pad transform requires target_length metadata")?,
                );
                padded = true;
            }
            other => anyhow::bail!(
                "unsupported required image transform '{other}'; supported operations are decode, decode_rgb, convert_rgb, resize, rescale, normalize, tile, patchify, flatten, emit_original_size, emit_transformed_size, emit_validity_mask, emit_patch_coordinates, emit_grid_coordinates, and pad"
            ),
        }
    }
    if patchified && !flattened {
        anyhow::bail!(
            "image patchify flatten=false requires a following flatten transform before packed output"
        );
    }

    let (width, height, resize_mode, interpolation) = match resize {
        Some((ImageSize::Square(size), mode, interpolation)) => (size, size, mode, interpolation),
        Some((ImageSize::Dimensions { width, height }, mode, interpolation)) => {
            (width, height, mode, interpolation)
        }
        None => (
            resolve_dimension("width", model_width, None)?,
            resolve_dimension("height", model_height, None)?,
            ResizeMode::Fixed,
            Interpolation::Bicubic,
        ),
    };
    validate_image_dimensions(width, height, "image resize")?;
    let tiling = match tiling {
        Some(tiling) => tiling,

        None => ImageTilingConfig {
            mode: TilingMode::None,
            tile_size: width,
            max_tiles: 1,
            aspect_ratios: vec![TileGrid {
                columns: 1,
                rows: 1,
            }],
            include_thumbnail: false,
            thumbnail_position: ThumbnailPosition::None,
        },
    };
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(metadata.outputs.len())
        .context("failed to allocate image output specifications")?;
    for output in metadata.outputs {
        outputs.push(OutputSpec {
            source: output.source,
            packed: packed::OutputSpec {
                name: output.name,
                content: output.content,
                dtype: ImageTensorDType::parse(&output.dtype)?,
                pad_value: output.pad_value,
                optional: output.optional.unwrap_or(false),
            },
        });
    }
    Ok((
        ImagePreprocessConfig {
            width,
            height,
            resize_mode,
            interpolation,
            tiling,
            // Typed programs execute value transforms in declared order.
            normalization: Normalization::ZeroToOne,
        },
        ImageProgram {
            value_ops,
            named_value_ops,
            patchify,
            pad_value,
            target_length,
            dynamic_resize,
            dynamic_hd,
            outputs,
        },
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProgramValueKind {
    Raster,
    Tiles,
    Patches,
    FlatPatches,
    PaddedPatches,
    Coordinates,
    PaddedCoordinates,
    Grid,
    OriginalSize,
    TransformedSize,
    ValidityMask,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ProgramStructure {
    resized: bool,
    tiled: bool,
    patchified: bool,
    flattened: bool,
    padded: bool,
}

fn validate_program_dataflow(
    transforms: &[ImageTransformMetadata],
    outputs: &[ImageOutputMetadata],
) -> anyhow::Result<bool> {
    let explicit = transforms
        .iter()
        .any(|transform| transform.inputs.is_some() || transform.outputs.is_some())
        || outputs.iter().any(|output| output.source.is_some());
    if !explicit {
        return Ok(false);
    }

    let mut values = BTreeMap::<String, ProgramValueKind>::new();
    let mut structures = BTreeMap::<String, ProgramStructure>::new();
    let mut global_structure = ProgramStructure::default();
    let mut previous = Vec::<String>::new();
    for (index, transform) in transforms.iter().enumerate() {
        let inputs = match &transform.inputs {
            Some(inputs) => inputs.clone(),
            None if previous.len() == 1 => previous.clone(),
            None if matches!(transform.op.as_str(), "decode" | "decode_rgb") => Vec::new(),
            None => anyhow::bail!(
                "image transform {index} ('{}') must declare inputs because the preceding transform produced {} values",
                transform.op,
                previous.len()
            ),
        };
        let input_kinds = inputs
            .iter()
            .map(|input| {
                values.get(input).copied().with_context(|| {
                    format!(
                        "image transform {index} ('{}') consumes unknown value '{input}'; declare a preceding transform output with that name",
                        transform.op
                    )
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let output_kinds = descriptor_output_kinds(index, transform, &input_kinds)?;
        let input_structures = inputs
            .iter()
            .map(|input| {
                structures.get(input).copied().with_context(|| {
                    format!(
                        "image transform {index} ('{}') consumes unknown value '{input}'",
                        transform.op
                    )
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let output_structures = descriptor_output_structures(index, transform, &input_structures)?;
        update_program_structure(&mut global_structure, &transform.op);
        let names = transform.outputs.as_ref().with_context(|| {
            format!(
                "image transform {index} ('{}') must declare outputs in an explicitly named preprocessing program",
                transform.op
            )
        })?;
        if names.len() != output_kinds.len() {
            anyhow::bail!(
                "image transform {index} ('{}') declares {} outputs, but this operation produces {}; fix preprocessing.image.transforms[{index}].outputs",
                transform.op,
                names.len(),
                output_kinds.len()
            );
        }
        previous.clear();
        for ((name, kind), structure) in names.iter().zip(output_kinds).zip(output_structures) {
            if name.is_empty() {
                anyhow::bail!("image transform {index} declares an empty output name");
            }
            if values.insert(name.clone(), kind).is_some() {
                anyhow::bail!(
                    "image transform {index} ('{}') redefines value '{name}'; transform output names must be unique",
                    transform.op
                );
            }
            structures.insert(name.clone(), structure);
            previous.push(name.clone());
        }
    }

    let has_patchify = transforms
        .iter()
        .any(|transform| transform.op == "patchify");
    let has_pad = transforms.iter().any(|transform| transform.op == "pad");
    for output in outputs {
        let Some(source) = output.source.as_deref() else {
            anyhow::bail!(
                "image output '{}' must declare source in an explicitly named preprocessing program",
                output.name
            );
        };
        let kind = values.get(source).with_context(|| {
            format!(
                "image output '{}' selects unknown source '{source}'; choose a declared transform output",
                output.name
            )
        })?;
        let structure = structures
            .get(source)
            .copied()
            .context("validated image output source lost its structure")?;
        if !content_accepts_kind(&output.content, *kind, has_patchify, has_pad) {
            anyhow::bail!(
                "image output '{}' declares content '{}' but source '{source}' carries {kind:?}; bind the output to a compatible transform value",
                output.name,
                output.content
            );
        }
        validate_supported_output_structure(output, source, structure, global_structure)?;
    }
    Ok(true)
}

fn descriptor_output_structures(
    index: usize,
    transform: &ImageTransformMetadata,
    inputs: &[ProgramStructure],
) -> anyhow::Result<Vec<ProgramStructure>> {
    if matches!(transform.op.as_str(), "decode" | "decode_rgb") {
        return Ok(vec![ProgramStructure::default()]);
    }
    if transform.op == "pad" {
        return Ok(inputs
            .iter()
            .copied()
            .map(|mut structure| {
                structure.padded = true;
                structure
            })
            .collect());
    }
    let &[mut structure] = inputs else {
        anyhow::bail!(
            "image transform {index} ('{}') expects one input, got {}",
            transform.op,
            inputs.len()
        );
    };
    update_program_structure(&mut structure, &transform.op);
    Ok(vec![structure])
}

fn update_program_structure(structure: &mut ProgramStructure, op: &str) {
    match op {
        "resize" => structure.resized = true,
        "tile" => structure.tiled = true,
        "patchify" => structure.patchified = true,
        "flatten" => structure.flattened = true,
        "pad" => structure.padded = true,
        _ => {}
    }
}

fn validate_supported_output_structure(
    output: &ImageOutputMetadata,
    source: &str,
    structure: ProgramStructure,
    global: ProgramStructure,
) -> anyhow::Result<()> {
    let supported = match output.content.as_str() {
        "pixels" => structure == global,
        "transformed_size" | "validity_mask" => {
            structure.resized == global.resized && structure.tiled == global.tiled
        }
        "patch_coordinates" | "grid_dimensions" => {
            structure.resized == global.resized
                && structure.tiled == global.tiled
                && structure.patchified == global.patchified
        }
        "original_size" => true,
        _ => true,
    };
    if !supported {
        anyhow::bail!(
            "image output '{}' selects source '{source}' from a structural branch that the current packer cannot execute independently; resize, tile, patchify, flatten, and pad must follow the output's selected dataflow path",
            output.name
        );
    }
    Ok(())
}

fn compile_named_value_ops(
    transforms: &[ImageTransformMetadata],
) -> anyhow::Result<BTreeMap<String, Vec<ValueOp>>> {
    let mut values = BTreeMap::<String, Vec<ValueOp>>::new();
    let mut previous = Vec::<String>::new();
    for (index, transform) in transforms.iter().enumerate() {
        let inputs = match &transform.inputs {
            Some(inputs) => inputs.clone(),
            None if previous.len() == 1 => previous.clone(),
            None if matches!(transform.op.as_str(), "decode" | "decode_rgb") => Vec::new(),
            None => anyhow::bail!(
                "image transform {index} ('{}') has ambiguous implicit inputs",
                transform.op
            ),
        };
        let input_ops = inputs
            .iter()
            .map(|input| {
                values.get(input).cloned().with_context(|| {
                    format!(
                        "image transform {index} ('{}') consumes unknown value '{input}'",
                        transform.op
                    )
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let names = transform.outputs.as_ref().with_context(|| {
            format!(
                "image transform {index} ('{}') must declare outputs in an explicitly named preprocessing program",
                transform.op
            )
        })?;
        let mut output_ops = if transform.op == "pad" {
            input_ops
        } else if matches!(transform.op.as_str(), "decode" | "decode_rgb") {
            vec![Vec::new()]
        } else {
            let [mut ops] = input_ops.try_into().map_err(|inputs: Vec<Vec<ValueOp>>| {
                anyhow::anyhow!(
                    "image transform {index} ('{}') expects one input, got {}",
                    transform.op,
                    inputs.len()
                )
            })?;
            match transform.op.as_str() {
                "rescale" => {
                    let scale = transform
                        .scale
                        .context("image rescale transform requires scale metadata")?
                        as f32;
                    ops.push(ValueOp::Rescale(scale));
                }
                "normalize" => {
                    ops.push(ValueOp::Normalize {
                        mean: channel_values("mean", transform.mean.clone())?,
                        std: channel_values("std", transform.std.clone())?,
                    });
                }
                _ => {}
            }
            vec![ops]
        };
        if names.len() != output_ops.len() {
            anyhow::bail!(
                "image transform {index} ('{}') declares {} outputs, but compiled {} value branches",
                transform.op,
                names.len(),
                output_ops.len()
            );
        }
        previous.clear();
        for (name, ops) in names.iter().zip(output_ops.drain(..)) {
            values.insert(name.clone(), ops);
            previous.push(name.clone());
        }
    }
    Ok(values)
}

fn descriptor_output_kinds(
    index: usize,
    transform: &ImageTransformMetadata,
    inputs: &[ProgramValueKind],
) -> anyhow::Result<Vec<ProgramValueKind>> {
    let require = |expected: &[ProgramValueKind]| -> anyhow::Result<ProgramValueKind> {
        let [actual] = inputs else {
            anyhow::bail!(
                "image transform {index} ('{}') expects one input, got {}",
                transform.op,
                inputs.len()
            );
        };
        if !expected.contains(actual) {
            anyhow::bail!(
                "image transform {index} ('{}') received incompatible input {actual:?}",
                transform.op
            );
        }
        Ok(*actual)
    };
    match transform.op.as_str() {
        "decode" | "decode_rgb" => {
            if !inputs.is_empty() {
                anyhow::bail!(
                    "image transform {index} ('{}') must not declare inputs",
                    transform.op
                );
            }
            Ok(vec![ProgramValueKind::Raster])
        }
        "convert_rgb" | "resize" => {
            require(&[ProgramValueKind::Raster])?;
            Ok(vec![ProgramValueKind::Raster])
        }
        "rescale" | "normalize" => {
            let kind = require(&[ProgramValueKind::Raster, ProgramValueKind::Tiles])?;
            Ok(vec![kind])
        }
        "tile" => {
            require(&[ProgramValueKind::Raster])?;
            Ok(vec![ProgramValueKind::Tiles])
        }
        "patchify" => {
            require(&[ProgramValueKind::Raster, ProgramValueKind::Tiles])?;
            Ok(vec![ProgramValueKind::Patches])
        }
        "flatten" => {
            require(&[ProgramValueKind::Patches])?;
            Ok(vec![ProgramValueKind::FlatPatches])
        }
        "emit_original_size" => {
            require(&[ProgramValueKind::Raster])?;
            Ok(vec![ProgramValueKind::OriginalSize])
        }
        "emit_transformed_size" => {
            require(&[ProgramValueKind::Raster, ProgramValueKind::Tiles])?;
            Ok(vec![ProgramValueKind::TransformedSize])
        }
        "emit_validity_mask" => {
            require(&[ProgramValueKind::Tiles])?;
            Ok(vec![ProgramValueKind::ValidityMask])
        }
        "emit_patch_coordinates" => {
            require(&[ProgramValueKind::Patches, ProgramValueKind::FlatPatches])?;
            Ok(vec![ProgramValueKind::Coordinates])
        }
        "emit_grid_coordinates" => {
            require(&[ProgramValueKind::Patches, ProgramValueKind::FlatPatches])?;
            Ok(vec![ProgramValueKind::Grid])
        }
        "pad" => {
            if inputs.is_empty() {
                anyhow::bail!("image transform {index} ('pad') expects at least one input");
            }
            inputs
                .iter()
                .map(|kind| match kind {
                    ProgramValueKind::Patches | ProgramValueKind::FlatPatches => {
                        Ok(ProgramValueKind::PaddedPatches)
                    }
                    ProgramValueKind::Coordinates => Ok(ProgramValueKind::PaddedCoordinates),
                    other => anyhow::bail!(
                        "image transform {index} ('pad') cannot pad input {other:?}; expected patches or coordinates"
                    ),
                })
                .collect()
        }
        other => anyhow::bail!(
            "unsupported required image transform '{other}' at preprocessing.image.transforms[{index}]"
        ),
    }
}

fn content_accepts_kind(
    content: &str,
    kind: ProgramValueKind,
    has_patchify: bool,
    has_pad: bool,
) -> bool {
    match content {
        "pixels" if has_pad => kind == ProgramValueKind::PaddedPatches,
        "pixels" if has_patchify => {
            matches!(
                kind,
                ProgramValueKind::Patches | ProgramValueKind::FlatPatches
            )
        }
        "pixels" => matches!(kind, ProgramValueKind::Raster | ProgramValueKind::Tiles),
        "patch_coordinates" if has_pad => kind == ProgramValueKind::PaddedCoordinates,
        "patch_coordinates" => kind == ProgramValueKind::Coordinates,
        "grid_dimensions" => kind == ProgramValueKind::Grid,
        "original_size" => kind == ProgramValueKind::OriginalSize,
        "transformed_size" => kind == ProgramValueKind::TransformedSize,
        "validity_mask" => kind == ProgramValueKind::ValidityMask,
        _ => false,
    }
}

fn parse_interpolation(value: Option<&str>) -> anyhow::Result<Interpolation> {
    match value.unwrap_or("bicubic") {
        "bicubic" => Ok(Interpolation::Bicubic),
        "bilinear" => Ok(Interpolation::Bilinear),
        "lanczos" | "lanczos3" => Ok(Interpolation::Lanczos3),
        other => anyhow::bail!("unsupported image interpolation '{other}'"),
    }
}

fn channel_values(name: &str, values: Option<Vec<f32>>) -> anyhow::Result<[f32; 3]> {
    let values = values.with_context(|| format!("image normalize transform requires {name}"))?;
    values.try_into().map_err(|values: Vec<f32>| {
        anyhow::anyhow!(
            "image normalize transform {name} must contain 3 RGB values, got {}",
            values.len()
        )
    })
}

fn preprocessing_from_metadata(
    metadata: Option<ImageMetadata>,
    model_width: i64,
    model_height: i64,
) -> anyhow::Result<ImagePreprocessConfig> {
    let declared_size = metadata
        .as_ref()
        .and_then(|image| image.resize.as_ref())
        .and_then(|resize| resize.size.as_ref())
        .map(|size| match size {
            ImageSize::Square(size) => (*size, *size),
            ImageSize::Dimensions { width, height } => (*width, *height),
        });
    let width = resolve_dimension("width", model_width, declared_size.map(|size| size.0))?;
    let height = resolve_dimension("height", model_height, declared_size.map(|size| size.1))?;
    validate_image_dimensions(width, height, "image resize")?;

    let resize = metadata.as_ref().and_then(|image| image.resize.as_ref());
    let mode = resize.and_then(|resize| resize.mode.as_deref());
    let crop = resize.and_then(|resize| resize.crop.as_deref());
    let resize_mode = match mode.unwrap_or("shortest_edge") {
        "shortest_edge" => match crop.unwrap_or("center") {
            "center" | "center_crop" => ResizeMode::ShortestEdgeCenterCrop,
            other => anyhow::bail!("unsupported shortest_edge crop mode '{other}'"),
        },
        "fixed" | "fixed_size" => {
            if crop.is_some_and(|crop| crop != "none") {
                anyhow::bail!("fixed resize only supports crop mode 'none'");
            }
            ResizeMode::Fixed
        }
        "longest_edge_pad" => {
            if crop.is_some_and(|crop| crop != "none") {
                anyhow::bail!("longest_edge_pad only supports crop mode 'none'");
            }
            ResizeMode::LongestEdgePad
        }
        other => anyhow::bail!("unsupported image resize mode '{other}'"),
    };
    let interpolation = match resize
        .and_then(|resize| resize.interpolation.as_deref())
        .unwrap_or("bicubic")
    {
        "bicubic" => Interpolation::Bicubic,
        "bilinear" => Interpolation::Bilinear,
        "lanczos" | "lanczos3" => Interpolation::Lanczos3,
        other => anyhow::bail!("unsupported image interpolation '{other}'"),
    };
    let tiling = tiling_from_metadata(
        metadata.as_ref().and_then(|image| image.tiling.as_ref()),
        width,
        height,
    )?;
    let normalization = match metadata.and_then(|image| image.normalize) {
        Some(normalize) => {
            if normalize.std.iter().any(|value| *value <= 0.0) {
                anyhow::bail!("image normalization std values must be greater than zero");
            }
            Normalization::MeanStd {
                mean: normalize.mean,
                std: normalize.std,
            }
        }
        None => Normalization::ZeroToOne,
    };

    Ok(ImagePreprocessConfig {
        width,
        height,
        resize_mode,
        interpolation,
        tiling,
        normalization,
    })
}

fn tiling_from_metadata(
    metadata: Option<&TilingMetadata>,
    width: u32,
    height: u32,
) -> anyhow::Result<ImageTilingConfig> {
    let mode = match metadata.and_then(|tiling| tiling.mode.as_deref()) {
        None | Some("none") => TilingMode::None,
        Some("fixed_grid") => TilingMode::FixedGrid,
        Some("dynamic_anyres") => TilingMode::DynamicAnyres,
        Some(other) => anyhow::bail!("unsupported image tiling mode '{other}'"),
    };
    if mode == TilingMode::None {
        return Ok(ImageTilingConfig {
            mode,
            tile_size: width,
            max_tiles: 1,
            aspect_ratios: vec![TileGrid {
                columns: 1,
                rows: 1,
            }],
            include_thumbnail: false,
            thumbnail_position: ThumbnailPosition::None,
        });
    }

    let tile_size = match metadata.and_then(|tiling| tiling.tile_size) {
        Some(0) => anyhow::bail!("image tiling tile_size must be greater than zero"),
        Some(tile_size) => tile_size,
        None if width == height => width,
        None => anyhow::bail!("non-square tiled image inputs require tiling.tile_size metadata"),
    };
    if width != tile_size || height != tile_size {
        anyhow::bail!(
            "tiling tile_size {tile_size} must match model tile dimensions {width}x{height}"
        );
    }
    let max_tiles = metadata.and_then(|tiling| tiling.max_tiles).unwrap_or(6);
    if max_tiles == 0 {
        anyhow::bail!("image tiling max_tiles must be greater than zero");
    }
    if max_tiles > MAX_TILES_PER_IMAGE {
        anyhow::bail!(
            "image tiling max_tiles {max_tiles} exceeds the supported limit of {MAX_TILES_PER_IMAGE}; reduce max_tiles"
        );
    }

    let configured_ratios = metadata.and_then(|tiling| tiling.aspect_ratios.as_ref());
    if configured_ratios.is_some_and(|ratios| ratios.len() > MAX_ASPECT_RATIOS) {
        anyhow::bail!(
            "image tiling aspect_ratios exceeds the supported limit of {MAX_ASPECT_RATIOS} entries"
        );
    }
    let aspect_ratios = match (mode, configured_ratios) {
        (TilingMode::FixedGrid, None) => vec![TileGrid {
            columns: 1,
            rows: 1,
        }],
        (TilingMode::DynamicAnyres, None) => default_anyres_grids(),
        (TilingMode::DynamicHd, None) => default_anyres_grids(),
        (_, Some(ratios)) => {
            let mut grids = Vec::new();
            grids
                .try_reserve_exact(ratios.len())
                .context("failed to allocate image tiling aspect ratios")?;
            for [columns, rows] in ratios {
                grids.push(TileGrid {
                    columns: *columns,
                    rows: *rows,
                });
            }
            grids
        }
        (TilingMode::None, _) => unreachable!("none returned above"),
    };
    if aspect_ratios.is_empty() {
        anyhow::bail!("image tiling aspect_ratios must not be empty");
    }
    for grid in &aspect_ratios {
        if grid.columns == 0 || grid.rows == 0 {
            anyhow::bail!("image tiling aspect ratios must contain positive grid dimensions");
        }
        grid.tile_count()?;
    }
    if mode == TilingMode::FixedGrid {
        if aspect_ratios.len() != 1 {
            anyhow::bail!("fixed_grid tiling requires exactly one aspect_ratios entry");
        }
        let count = aspect_ratios[0].tile_count()?;
        if count > max_tiles {
            anyhow::bail!(
                "fixed_grid produces {count} local tiles, exceeding max_tiles {max_tiles}"
            );
        }
    } else if !aspect_ratios
        .iter()
        .any(|grid| grid.tile_count().is_ok_and(|count| count <= max_tiles))
    {
        anyhow::bail!("no dynamic_anyres aspect ratio fits max_tiles {max_tiles}");
    }

    let include_thumbnail = metadata
        .and_then(|tiling| tiling.include_thumbnail)
        .unwrap_or(true);
    Ok(ImageTilingConfig {
        mode,
        tile_size,
        max_tiles,
        aspect_ratios,
        include_thumbnail,
        thumbnail_position: if include_thumbnail {
            ThumbnailPosition::Prepend
        } else {
            ThumbnailPosition::None
        },
    })
}

fn parse_thumbnail_position(value: Option<&str>) -> anyhow::Result<ThumbnailPosition> {
    match value.unwrap_or("none") {
        "none" => Ok(ThumbnailPosition::None),
        "prepend" => Ok(ThumbnailPosition::Prepend),
        "append" => Ok(ThumbnailPosition::Append),
        other => anyhow::bail!(
            "unsupported image thumbnail_order '{other}'; expected none, prepend, or append"
        ),
    }
}

fn exact_u8(value: f64, description: &str) -> anyhow::Result<u8> {
    if !value.is_finite() || value.fract() != 0.0 || !(0.0..=255.0).contains(&value) {
        anyhow::bail!("{description} must be an integer in the range 0..=255, got {value}");
    }
    Ok(value as u8)
}

fn default_anyres_grids() -> Vec<TileGrid> {
    [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2)]
        .into_iter()
        .map(|(columns, rows)| TileGrid { columns, rows })
        .collect()
}

pub(super) fn validate_image_dimensions(
    width: u32,
    height: u32,
    description: &str,
) -> anyhow::Result<()> {
    if width == 0 || height == 0 {
        anyhow::bail!("{description} dimensions must be greater than zero, got {width}x{height}");
    }
    let pixels = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .context("image dimensions are too large for this platform")?;
    if pixels > MAX_IMAGE_PIXELS {
        anyhow::bail!(
            "{description} dimensions {width}x{height} require {pixels} pixels, exceeding the safety limit of {MAX_IMAGE_PIXELS}; reduce the configured image size"
        );
    }
    Ok(())
}

pub(super) fn validate_source_image(
    image: &DynamicImage,
    image_index: usize,
) -> anyhow::Result<()> {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 {
        anyhow::bail!(
            "source image {image_index} has degenerate dimensions {width}x{height}; provide an image with nonzero width and height"
        );
    }
    let pixels = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .context("source image dimensions are too large for this platform")?;
    if pixels > MAX_IMAGE_PIXELS {
        anyhow::bail!(
            "source image {image_index} dimensions {width}x{height} contain {pixels} pixels, exceeding the safety limit of {MAX_IMAGE_PIXELS}; resize the image before preprocessing"
        );
    }
    Ok(())
}

pub(super) fn checked_image_elements(
    width: usize,
    height: usize,
    description: &str,
) -> anyhow::Result<usize> {
    let elements = CHANNELS
        .checked_mul(width)
        .and_then(|value| value.checked_mul(height))
        .with_context(|| format!("{description} element count overflowed"))?;
    if elements > MAX_TENSOR_ELEMENTS {
        anyhow::bail!(
            "{description} requires {elements} elements, exceeding the safety limit of {MAX_TENSOR_ELEMENTS}; reduce image dimensions"
        );
    }

    Ok(elements)
}

fn resolve_dimension(name: &str, model: i64, configured: Option<u32>) -> anyhow::Result<u32> {
    if model == 0 || model < -1 {
        anyhow::bail!("vision input has invalid {name} dimension {model}");
    }

    let model_dimension = (model > 0)
        .then(|| {
            u32::try_from(model)
                .with_context(|| format!("vision input {name} dimension {model} is too large"))
        })
        .transpose()?;
    match (model_dimension, configured) {
        (Some(model), Some(configured)) if model != configured => anyhow::bail!(
            "preprocessing {name} {configured} does not match model input {name} {model}"
        ),
        (_, Some(0)) => anyhow::bail!("preprocessing {name} must be greater than zero"),
        (_, Some(configured)) => Ok(configured),
        (Some(model), None) => Ok(model),
        (None, None) => anyhow::bail!(
            "dynamic vision input {name} requires preprocessing.image.resize.size metadata"
        ),
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
