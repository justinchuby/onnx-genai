//! Typed image tensor bundles and generic patch packing.

use std::collections::BTreeSet;

use anyhow::Context;

use super::{
    CoordinateOrder, ImageLayout, MAX_IMAGE_COUNT, MAX_TENSOR_ELEMENTS, PatchChannelOrder,
    PatchifySpec, ThumbnailPosition, TileGrid,
};

/// Declared tensor element type for an image processor output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageTensorDType {
    Fp32,
    Fp16,
    Bf16,
    Int64,
    Int32,
    Int8,
    Uint8,
    Bool,
}

impl ImageTensorDType {
    pub(super) fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "float32" | "fp32" => Ok(Self::Fp32),
            "float16" | "fp16" | "half" => Ok(Self::Fp16),
            "bfloat16" | "bf16" => Ok(Self::Bf16),
            "int64" => Ok(Self::Int64),
            "int32" => Ok(Self::Int32),
            "int8" => Ok(Self::Int8),
            "uint8" => Ok(Self::Uint8),
            "bool" => Ok(Self::Bool),
            other => anyhow::bail!(
                "image output declares unsupported dtype '{other}'; supported dtypes are fp32, fp16, bf16, int64, int32, int8, uint8, and bool"
            ),
        }
    }
}

/// Contiguous storage for one typed image processor output.
#[derive(Debug, Clone, PartialEq)]
pub enum ImageTensorData {
    Fp32(Vec<f32>),
    /// IEEE-754 binary16 bit patterns.
    Fp16(Vec<u16>),
    /// IEEE-754 bfloat16 bit patterns.
    Bf16(Vec<u16>),
    Int64(Vec<i64>),
    Int32(Vec<i32>),
    Int8(Vec<i8>),
    Uint8(Vec<u8>),
    /// ONNX bool values stored as contiguous `0`/`1` bytes.
    Bool(Vec<u8>),
}

impl ImageTensorData {
    /// Number of scalar elements in this tensor.
    pub fn len(&self) -> usize {
        match self {
            Self::Fp32(values) => values.len(),
            Self::Fp16(values) | Self::Bf16(values) => values.len(),
            Self::Int64(values) => values.len(),
            Self::Int32(values) => values.len(),
            Self::Int8(values) => values.len(),
            Self::Uint8(values) => values.len(),
            Self::Bool(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            Self::Fp32(values) => Some(values),
            _ => None,
        }
    }
}

/// One named tensor emitted by the metadata-declared image program.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedImageTensor {
    pub name: String,
    pub content: String,
    pub dtype: ImageTensorDType,
    pub shape: Vec<i64>,
    pub data: ImageTensorData,
}

/// Per-image information used by placeholder expansion and downstream routing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageExpansionSummary {
    pub image_index: usize,
    /// Original image size as `(width, height)`.
    pub original_size: (u32, u32),
    pub tile_grid: TileGrid,
    pub tile_count: usize,
    /// Number of real patches, or the tile count for a rank-4 pixel output.
    pub expansion_count: usize,
    /// Start of this image in the packed or padded pixel tensor.
    pub tensor_offset: usize,
    /// Number of entries reserved for this image, including padding.
    pub tensor_length: usize,
}

/// Named typed tensors plus image-order-preserving expansion metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageTensorBundle {
    pub tensors: Vec<NamedImageTensor>,
    pub images: Vec<ImageExpansionSummary>,
    pub num_tiles: usize,
    pub tiles_per_image: Vec<usize>,
    pub tile_grids: Vec<TileGrid>,
    pub thumbnail_position: ThumbnailPosition,
}

impl ImageTensorBundle {
    pub fn tensor(&self, name: &str) -> Option<&NamedImageTensor> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    pub fn tensor_by_content(&self, content: &str) -> Option<&NamedImageTensor> {
        self.tensors.iter().find(|tensor| tensor.content == content)
    }

    /// Returns one fp32 rank-4 tile in the declared channel layout.
    pub fn tile_data(&self, index: usize) -> Option<&[f32]> {
        let pixels = self.tensor_by_content("pixels")?;
        if pixels.shape.len() != 4 {
            return None;
        }
        let data = pixels.data.as_f32_slice()?;
        let values_per_tile = data.len().checked_div(self.num_tiles)?;
        let start = index.checked_mul(values_per_tile)?;
        data.get(start..start.checked_add(values_per_tile)?)
    }

    pub fn tiling_summary(&self) -> super::ImageTilingSummary<'_> {
        super::ImageTilingSummary {
            num_tiles: self.num_tiles,
            tiles_per_image: &self.tiles_per_image,
            tile_grids: &self.tile_grids,
            thumbnail_position: self.thumbnail_position,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct OutputSpec {
    pub name: String,
    pub content: String,
    pub dtype: ImageTensorDType,
    pub pad_value: Option<f64>,
    pub optional: bool,
}

#[derive(Debug)]
pub(super) struct PreparedImage {
    pub original_size: (u32, u32),
    pub transformed_size: (u32, u32),
    pub tile_grid: TileGrid,
    pub tile_size: (usize, usize),
    /// Channel-first normalized tiles.
    pub tiles: Vec<Vec<f32>>,
    pub validity_masks: Option<Vec<Vec<u8>>>,
}

#[derive(Debug)]
pub(super) struct PackSpec {
    pub layout: ImageLayout,
    pub patchify: Option<PatchifySpec>,
    pub pad_value: Option<f64>,
    pub target_length: Option<usize>,
    pub outputs: Vec<OutputSpec>,
    pub declared_pixel_shape: Vec<i64>,
}

#[derive(Debug)]
struct PackedImage {
    patches: Vec<f32>,
    coordinates: Vec<i64>,
    patch_count: usize,
    grid: [i64; 3],
}

pub(super) fn build_bundle(
    prepared: Vec<PreparedImage>,
    spec: &PackSpec,
    thumbnail_position: ThumbnailPosition,
) -> anyhow::Result<ImageTensorBundle> {
    validate_outputs(&spec.outputs)?;
    if prepared.len() > MAX_IMAGE_COUNT {
        anyhow::bail!(
            "image batch contains {} images, exceeding the supported limit of {MAX_IMAGE_COUNT}; split the request into smaller batches",
            prepared.len()
        );
    }
    let num_tiles = prepared.iter().try_fold(0usize, |total, image| {
        total
            .checked_add(image.tiles.len())
            .context("image tile count is too large")
    })?;
    ensure_element_limit(num_tiles, "image tile count")?;
    let mut tiles_per_image = try_vec_with_capacity(prepared.len(), "image tile counts")?;
    let mut tile_grids = try_vec_with_capacity(prepared.len(), "image tile grids")?;
    for image in &prepared {
        tiles_per_image.push(image.tiles.len());
        tile_grids.push(image.tile_grid);
    }

    let (packed, feature_size) = match &spec.patchify {
        Some(patchify) => {
            let patch_size = patchify.patch_size;
            if patch_size == 0 {
                anyhow::bail!("image patchify patch_size must be greater than zero");
            }
            let feature_size = checked_element_product(
                "image patch feature dimension",
                &[3, patch_size, patchify.temporal_patch_size, patch_size],
            )?;
            let mut packed = try_vec_with_capacity(prepared.len(), "packed image batch")?;
            for image in &prepared {
                packed.push(pack_image(image, patchify)?);
            }
            let total_patch_count = packed.iter().try_fold(0usize, |total, image| {
                total
                    .checked_add(image.patch_count)
                    .context("total image patch count overflowed")
            })?;
            checked_element_product(
                "packed image pixel storage",
                &[total_patch_count, feature_size],
            )?;
            checked_element_product("packed image coordinate storage", &[total_patch_count, 2])?;
            (Some(packed), Some(feature_size))
        }
        None => (None, None),
    };

    let actual_max_patches = packed
        .as_ref()
        .and_then(|images| images.iter().map(|image| image.patch_count).max())
        .unwrap_or(0);
    let max_patches = match spec.target_length {
        Some(target) if target < actual_max_patches => anyhow::bail!(
            "image pad target_length {target} is smaller than the produced patch count {actual_max_patches}"
        ),
        Some(target) => target,
        None => actual_max_patches,
    };
    let total_patches = packed
        .as_ref()
        .map(|images| {
            images.iter().try_fold(0usize, |total, image| {
                total
                    .checked_add(image.patch_count)
                    .context("total image patch count overflowed")
            })
        })
        .transpose()?
        .unwrap_or(0);
    ensure_element_limit(total_patches, "total image patch count")?;
    let padded = spec.patchify.is_some() && spec.target_length.is_some();
    validate_total_output_elements(
        &prepared,
        packed.as_deref(),
        feature_size,
        max_patches,
        total_patches,
        num_tiles,
        padded,
        spec,
    )?;

    let mut tensors = try_vec_with_capacity(spec.outputs.len(), "image output tensor list")?;
    for output in &spec.outputs {
        let produced = match output.content.as_str() {
            "pixels" => Some(build_pixels(
                &prepared,
                packed.as_deref(),
                feature_size,
                max_patches,
                total_patches,
                padded,
                spec,
                output,
            )?),
            "patch_coordinates" => match packed.as_deref() {
                Some(images) => Some(build_coordinates(
                    images,
                    max_patches,
                    total_patches,
                    padded,
                    output,
                )?),
                None if output.optional => None,
                None => anyhow::bail!(
                    "required image output '{}' with content patch_coordinates requires a patchify transform",
                    output.name
                ),
            },
            "grid_dimensions" => match packed.as_deref() {
                Some(images) => Some(build_grid(images, output)?),
                None if output.optional => None,
                None => anyhow::bail!(
                    "required image output '{}' with content grid_dimensions requires a patchify transform",
                    output.name
                ),
            },
            "original_size" => Some(build_original_sizes(&prepared, output)?),
            "transformed_size" => Some(build_transformed_sizes(&prepared, output)?),
            "validity_mask" => Some(build_validity_mask(
                &prepared,
                packed.as_deref(),
                max_patches,
                total_patches,
                padded,
                output,
            )?),
            _ if output.optional => None,
            other => anyhow::bail!(
                "required image output '{}' uses unsupported content role '{other}'",
                output.name
            ),
        };
        if let Some(tensor) = produced {
            tensors.push(tensor);
        }
    }

    let images = expansion_summaries(&prepared, packed.as_deref(), max_patches, padded)?;
    Ok(ImageTensorBundle {
        tensors,
        images,
        num_tiles,
        tiles_per_image,
        tile_grids,
        thumbnail_position,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_total_output_elements(
    prepared: &[PreparedImage],
    packed: Option<&[PackedImage]>,
    feature_size: Option<usize>,
    max_patches: usize,
    total_patches: usize,
    num_tiles: usize,
    padded: bool,
    spec: &PackSpec,
) -> anyhow::Result<()> {
    let mut total = 0usize;
    for output in &spec.outputs {
        let count = match output.content.as_str() {
            "pixels" => match feature_size {
                Some(feature_size) if padded => checked_element_product(
                    &format!("padded pixel output '{}'", output.name),
                    &[prepared.len(), max_patches, feature_size],
                )?,
                Some(feature_size) => checked_element_product(
                    &format!("concatenated pixel output '{}'", output.name),
                    &[total_patches, feature_size],
                )?,
                None => checked_element_product(
                    &format!("rank-4 pixel output '{}'", output.name),
                    &[
                        num_tiles,
                        3,
                        prepared.first().map_or(0, |image| image.tile_size.0),
                        prepared.first().map_or(0, |image| image.tile_size.1),
                    ],
                )?,
            },
            "patch_coordinates" => match packed {
                Some(_) if padded => checked_element_product(
                    &format!("padded coordinate output '{}'", output.name),
                    &[prepared.len(), max_patches, 2],
                )?,
                Some(_) => checked_element_product(
                    &format!("coordinate output '{}'", output.name),
                    &[total_patches, 2],
                )?,
                None if output.optional => 0,
                None => anyhow::bail!(
                    "required image output '{}' with content patch_coordinates requires a patchify transform",
                    output.name
                ),
            },
            "grid_dimensions" => match packed {
                Some(_) => checked_element_product(
                    &format!("grid output '{}'", output.name),
                    &[prepared.len(), 3],
                )?,
                None if output.optional => 0,
                None => anyhow::bail!(
                    "required image output '{}' with content grid_dimensions requires a patchify transform",
                    output.name
                ),
            },
            "original_size" => checked_element_product(
                &format!("original-size output '{}'", output.name),
                &[prepared.len(), 2],
            )?,
            "transformed_size" => checked_element_product(
                &format!("transformed-size output '{}'", output.name),
                &[prepared.len(), 2],
            )?,
            "validity_mask" => {
                if prepared.iter().any(|image| image.validity_masks.is_some()) {
                    prepared.iter().try_fold(0usize, |total, image| {
                        let masks = image.validity_masks.as_ref().with_context(|| {
                            format!(
                                "image output '{}' requires validity masks for every image",
                                output.name
                            )
                        })?;
                        let mask_elements = masks.iter().try_fold(0usize, |count, mask| {
                            count
                                .checked_add(mask.len())
                                .context("validity mask element count overflowed")
                        })?;
                        total
                            .checked_add(mask_elements)
                            .context("validity mask element count overflowed")
                    })?
                } else if padded {
                    checked_element_product(
                        &format!("padded validity mask '{}'", output.name),
                        &[prepared.len(), max_patches],
                    )?
                } else if packed.is_some() {
                    total_patches
                } else {
                    num_tiles
                }
            }
            _ if output.optional => 0,
            other => anyhow::bail!(
                "required image output '{}' uses unsupported content role '{other}'",
                output.name
            ),
        };
        total = total
            .checked_add(count)
            .context("total image output element count overflowed")?;
        if total > MAX_TENSOR_ELEMENTS {
            anyhow::bail!(
                "image output bundle requires {total} elements across declared tensors, exceeding the safety limit of {MAX_TENSOR_ELEMENTS}; reduce image dimensions, tile count, patch count, batch size, or duplicate pixel outputs"
            );
        }
    }
    Ok(())
}

fn checked_element_product(description: &str, factors: &[usize]) -> anyhow::Result<usize> {
    let elements = factors.iter().try_fold(1usize, |product, factor| {
        product
            .checked_mul(*factor)
            .with_context(|| format!("{description} element count overflowed"))
    })?;
    ensure_element_limit(elements, description)?;
    Ok(elements)
}

fn ensure_element_limit(elements: usize, description: &str) -> anyhow::Result<()> {
    if elements > MAX_TENSOR_ELEMENTS {
        anyhow::bail!(
            "{description} requires {elements} elements, exceeding the safety limit of {MAX_TENSOR_ELEMENTS}; reduce image dimensions, tile count, patch count, or batch size"
        );
    }
    Ok(())
}

fn try_vec_with_capacity<T>(capacity: usize, description: &str) -> anyhow::Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .with_context(|| format!("failed to allocate {description} with {capacity} elements"))?;
    Ok(values)
}

fn try_filled_vec<T: Clone>(length: usize, value: T, description: &str) -> anyhow::Result<Vec<T>> {
    ensure_element_limit(length, description)?;
    let mut values = try_vec_with_capacity(length, description)?;
    values.resize(length, value);
    Ok(values)
}

fn validate_outputs(outputs: &[OutputSpec]) -> anyhow::Result<()> {
    if outputs.is_empty() {
        anyhow::bail!("preprocessing.image.outputs must contain at least one output binding");
    }
    let mut names = BTreeSet::new();
    for output in outputs {
        if output.name.is_empty() {
            anyhow::bail!("image output binding name must not be empty");
        }
        if !names.insert(output.name.as_str()) {
            anyhow::bail!("image output binding name '{}' is duplicated", output.name);
        }
        if output.content == "pixels"
            && !matches!(
                output.dtype,
                ImageTensorDType::Fp32 | ImageTensorDType::Fp16 | ImageTensorDType::Bf16
            )
        {
            anyhow::bail!(
                "pixel output '{}' must declare fp32, fp16, or bf16, not {:?}",
                output.name,
                output.dtype
            );
        }
        if output.pad_value.is_some_and(|value| !value.is_finite()) {
            anyhow::bail!(
                "image output '{}' declares a non-finite pad value",
                output.name
            );
        }
    }
    Ok(())
}

fn pack_image(image: &PreparedImage, patchify: &PatchifySpec) -> anyhow::Result<PackedImage> {
    let (width, height) = image.tile_size;
    let patch_size = patchify.patch_size;
    let patches_w = width / patch_size;
    let patches_h = height / patch_size;
    if !width.is_multiple_of(patch_size) || !height.is_multiple_of(patch_size) {
        anyhow::bail!(
            "image dimensions {width}x{height} are not divisible by declared patch_size {patch_size}"
        );
    }
    if !patches_w.is_multiple_of(patchify.merge_size)
        || !patches_h.is_multiple_of(patchify.merge_size)
    {
        anyhow::bail!(
            "image patch grid {patches_w}x{patches_h} is not divisible by merge_size {}",
            patchify.merge_size
        );
    }
    let per_tile = patches_w
        .checked_mul(patches_h)
        .context("image patch count is too large")?;
    let patch_count = image
        .tiles
        .len()
        .checked_mul(per_tile)
        .context("image patch count is too large")?;
    ensure_element_limit(patch_count, "image patch count")?;
    let feature_size = checked_element_product(
        "image patch feature dimension",
        &[3, patch_size, patchify.temporal_patch_size, patch_size],
    )?;
    let patch_elements =
        checked_element_product("packed image pixel output", &[patch_count, feature_size])?;
    let coordinate_elements =
        checked_element_product("packed image coordinates", &[patch_count, 2])?;
    let expected_tile_elements =
        checked_element_product("normalized image tile", &[3, width, height])?;
    let mut patches = try_vec_with_capacity(patch_elements, "packed image pixels")?;
    let mut coordinates = try_vec_with_capacity(coordinate_elements, "packed image coordinates")?;
    for (tile_index, tile) in image.tiles.iter().enumerate() {
        if tile.len() != expected_tile_elements {
            anyhow::bail!(
                "normalized image tile has {} values, expected {} for RGB {}x{}",
                tile.len(),
                expected_tile_elements,
                width,
                height
            );
        }
        for group_y in 0..patches_h / patchify.merge_size {
            for group_x in 0..patches_w / patchify.merge_size {
                for local_y in 0..patchify.merge_size {
                    for local_x in 0..patchify.merge_size {
                        let patch_y = group_y * patchify.merge_size + local_y;
                        let patch_x = group_x * patchify.merge_size + local_x;
                        match patchify.channel_order {
                            PatchChannelOrder::ChannelsFirst => {
                                for channel in 0..3 {
                                    let channel_offset = channel * width * height;
                                    for _ in 0..patchify.temporal_patch_size {
                                        for y in 0..patch_size {
                                            let row = (patch_y * patch_size + y) * width;
                                            let start = channel_offset + row + patch_x * patch_size;
                                            patches.extend_from_slice(
                                                &tile[start..start + patch_size],
                                            );
                                        }
                                    }
                                }
                            }
                            PatchChannelOrder::ChannelsLast => {
                                for _ in 0..patchify.temporal_patch_size {
                                    for y in 0..patch_size {
                                        let row = (patch_y * patch_size + y) * width;
                                        for x in 0..patch_size {
                                            for channel in 0..3 {
                                                patches.push(
                                                    tile[channel * width * height
                                                        + row
                                                        + patch_x * patch_size
                                                        + x],
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        let y = tile_index
                            .checked_mul(patches_h)
                            .and_then(|value| value.checked_add(patch_y))
                            .context("image patch row coordinate overflowed")?;
                        let y =
                            i64::try_from(y).context("image patch row coordinate is too large")?;
                        let x = i64::try_from(patch_x)
                            .context("image patch column coordinate is too large")?;
                        match patchify.coordinate_order {
                            CoordinateOrder::Yx => coordinates.extend_from_slice(&[y, x]),
                            CoordinateOrder::Xy => coordinates.extend_from_slice(&[x, y]),
                        }
                    }
                }
            }
        }
    }
    Ok(PackedImage {
        patches,
        coordinates,
        patch_count,
        grid: [
            i64::try_from(image.tiles.len()).context("image grid depth is too large")?,
            i64::try_from(patches_h).context("image grid height is too large")?,
            i64::try_from(patches_w).context("image grid width is too large")?,
        ],
    })
}

#[allow(clippy::too_many_arguments)]
fn build_pixels(
    prepared: &[PreparedImage],
    packed: Option<&[PackedImage]>,
    feature_size: Option<usize>,
    max_patches: usize,
    total_patches: usize,
    padded: bool,
    spec: &PackSpec,
    output: &OutputSpec,
) -> anyhow::Result<NamedImageTensor> {
    let (shape, values) = if let Some(packed) = packed {
        let feature_size = feature_size.context("missing packed image feature size")?;
        if padded {
            let fill = output.pad_value.or(spec.pad_value).unwrap_or_default() as f32;
            let element_count = checked_element_product(
                &format!("padded pixel output '{}'", output.name),
                &[prepared.len(), max_patches, feature_size],
            )?;
            let mut values = try_filled_vec(
                element_count,
                fill,
                &format!("padded pixel output '{}'", output.name),
            )?;
            for (image_index, image) in packed.iter().enumerate() {
                let start = checked_element_product(
                    "padded image pixel offset",
                    &[image_index, max_patches, feature_size],
                )?;
                let end = start
                    .checked_add(image.patches.len())
                    .context("padded image pixel range overflowed")?;
                values[start..end].copy_from_slice(&image.patches);
            }
            (
                vec![
                    to_i64(prepared.len(), "image batch")?,
                    to_i64(max_patches, "padded patch count")?,
                    to_i64(feature_size, "patch feature size")?,
                ],
                values,
            )
        } else {
            let element_count = checked_element_product(
                &format!("concatenated pixel output '{}'", output.name),
                &[total_patches, feature_size],
            )?;
            let mut values = try_vec_with_capacity(
                element_count,
                &format!("concatenated pixel output '{}'", output.name),
            )?;
            for image in packed {
                values.extend_from_slice(&image.patches);
            }
            (
                vec![
                    to_i64(total_patches, "total patch count")?,
                    to_i64(feature_size, "patch feature size")?,
                ],
                values,
            )
        }
    } else {
        let (width, height) = prepared
            .first()
            .map(|image| image.tile_size)
            .context("rank-4 pixel output requires at least one prepared image")?;
        if prepared
            .iter()
            .any(|image| image.tile_size != (width, height))
        {
            anyhow::bail!(
                "rank-4 pixel output '{}' requires a common tile size across the image batch",
                output.name
            );
        }
        let tile_count = prepared.iter().try_fold(0usize, |total, image| {
            total
                .checked_add(image.tiles.len())
                .context("image tile count overflowed")
        })?;
        let element_count = checked_element_product(
            &format!("rank-4 pixel output '{}'", output.name),
            &[tile_count, 3, width, height],
        )?;
        let pixels_per_tile = checked_element_product("image tile spatial size", &[width, height])?;
        let mut values = try_vec_with_capacity(
            element_count,
            &format!("rank-4 pixel output '{}'", output.name),
        )?;
        for image in prepared {
            for tile in &image.tiles {
                match spec.layout {
                    ImageLayout::Nchw => values.extend_from_slice(tile),
                    ImageLayout::Nhwc => {
                        for pixel in 0..pixels_per_tile {
                            for channel in 0..3 {
                                values.push(tile[channel * pixels_per_tile + pixel]);
                            }
                        }
                    }
                }
            }
        }
        let shape = match spec.layout {
            ImageLayout::Nchw => vec![
                to_i64(tile_count, "image tile count")?,
                3,
                to_i64(height, "image height")?,
                to_i64(width, "image width")?,
            ],
            ImageLayout::Nhwc => vec![
                to_i64(tile_count, "image tile count")?,
                to_i64(height, "image height")?,
                to_i64(width, "image width")?,
                3,
            ],
        };
        (shape, values)
    };
    validate_declared_shape(&output.name, &shape, &spec.declared_pixel_shape)?;
    Ok(NamedImageTensor {
        name: output.name.clone(),
        content: output.content.clone(),
        dtype: output.dtype,
        shape,
        data: convert_f32(values, output.dtype, &output.name)?,
    })
}

fn build_coordinates(
    packed: &[PackedImage],
    max_patches: usize,
    total_patches: usize,
    padded: bool,
    output: &OutputSpec,
) -> anyhow::Result<NamedImageTensor> {
    let sentinel = output.pad_value.unwrap_or(-1.0);
    let sentinel = exact_i64(sentinel, &output.name)?;
    let (shape, values) = if padded {
        let element_count = checked_element_product(
            &format!("padded coordinate output '{}'", output.name),
            &[packed.len(), max_patches, 2],
        )?;
        let mut values = try_filled_vec(
            element_count,
            sentinel,
            &format!("padded coordinate output '{}'", output.name),
        )?;
        for (image_index, image) in packed.iter().enumerate() {
            let start = checked_element_product(
                "padded image coordinate offset",
                &[image_index, max_patches, 2],
            )?;
            let end = start
                .checked_add(image.coordinates.len())
                .context("padded image coordinate range overflowed")?;
            values[start..end].copy_from_slice(&image.coordinates);
        }
        (
            vec![
                to_i64(packed.len(), "image batch")?,
                to_i64(max_patches, "padded patch count")?,
                2,
            ],
            values,
        )
    } else {
        let element_count = checked_element_product(
            &format!("coordinate output '{}'", output.name),
            &[total_patches, 2],
        )?;
        let mut values = try_vec_with_capacity(
            element_count,
            &format!("coordinate output '{}'", output.name),
        )?;
        for image in packed {
            values.extend_from_slice(&image.coordinates);
        }
        (vec![to_i64(total_patches, "total patch count")?, 2], values)
    };
    Ok(NamedImageTensor {
        name: output.name.clone(),
        content: output.content.clone(),
        dtype: output.dtype,
        shape,
        data: convert_i64(values, output.dtype, &output.name)?,
    })
}

fn build_grid(packed: &[PackedImage], output: &OutputSpec) -> anyhow::Result<NamedImageTensor> {
    let element_count = checked_element_product(
        &format!("grid output '{}'", output.name),
        &[packed.len(), 3],
    )?;
    let mut values =
        try_vec_with_capacity(element_count, &format!("grid output '{}'", output.name))?;
    for image in packed {
        values.extend_from_slice(&image.grid);
    }
    Ok(NamedImageTensor {
        name: output.name.clone(),
        content: output.content.clone(),
        dtype: output.dtype,
        shape: vec![to_i64(packed.len(), "image batch")?, 3],
        data: convert_i64(values, output.dtype, &output.name)?,
    })
}

fn build_original_sizes(
    prepared: &[PreparedImage],
    output: &OutputSpec,
) -> anyhow::Result<NamedImageTensor> {
    let element_count = checked_element_product(
        &format!("original-size output '{}'", output.name),
        &[prepared.len(), 2],
    )?;
    let mut values = try_vec_with_capacity(
        element_count,
        &format!("original-size output '{}'", output.name),
    )?;
    for image in prepared {
        values.push(i64::from(image.original_size.1));
        values.push(i64::from(image.original_size.0));
    }
    Ok(NamedImageTensor {
        name: output.name.clone(),
        content: output.content.clone(),
        dtype: output.dtype,
        shape: vec![to_i64(prepared.len(), "image batch")?, 2],
        data: convert_i64(values, output.dtype, &output.name)?,
    })
}

fn build_transformed_sizes(
    prepared: &[PreparedImage],
    output: &OutputSpec,
) -> anyhow::Result<NamedImageTensor> {
    let element_count = checked_element_product(
        &format!("transformed-size output '{}'", output.name),
        &[prepared.len(), 2],
    )?;
    let mut values = try_vec_with_capacity(
        element_count,
        &format!("transformed-size output '{}'", output.name),
    )?;
    for image in prepared {
        values.push(i64::from(image.transformed_size.1));
        values.push(i64::from(image.transformed_size.0));
    }
    Ok(NamedImageTensor {
        name: output.name.clone(),
        content: output.content.clone(),
        dtype: output.dtype,
        shape: vec![to_i64(prepared.len(), "image batch")?, 2],
        data: convert_i64(values, output.dtype, &output.name)?,
    })
}

fn build_validity_mask(
    prepared: &[PreparedImage],
    packed: Option<&[PackedImage]>,
    max_patches: usize,
    total_patches: usize,
    padded: bool,
    output: &OutputSpec,
) -> anyhow::Result<NamedImageTensor> {
    let dynamic_masks = prepared
        .iter()
        .map(|image| image.validity_masks.as_ref())
        .collect::<Option<Vec<_>>>();
    let (shape, values) = if let Some(masks_by_image) = dynamic_masks {
        let first_mask = masks_by_image
            .first()
            .and_then(|masks| masks.first())
            .context("dynamic validity mask output requires at least one mask")?;
        let mask_edge = (first_mask.len() as f64).sqrt() as usize;
        if mask_edge * mask_edge != first_mask.len() {
            anyhow::bail!("dynamic validity mask cells must form a square");
        }
        let total_masks = masks_by_image.iter().try_fold(0usize, |total, masks| {
            total
                .checked_add(masks.len())
                .context("dynamic validity mask count overflowed")
        })?;
        let mut values = try_vec_with_capacity(
            total_masks * mask_edge * mask_edge,
            &format!("dynamic validity mask '{}'", output.name),
        )?;
        for masks in masks_by_image {
            for mask in masks {
                if mask.len() != mask_edge * mask_edge {
                    anyhow::bail!(
                        "dynamic validity masks for output '{}' have inconsistent shapes",
                        output.name
                    );
                }
                values.extend(mask.iter().map(|value| i64::from(*value)));
            }
        }
        (
            vec![
                to_i64(total_masks, "dynamic validity mask count")?,
                to_i64(mask_edge, "dynamic validity mask height")?,
                to_i64(mask_edge, "dynamic validity mask width")?,
            ],
            values,
        )
    } else {
        match packed {
            Some(packed) if padded => {
                let element_count = checked_element_product(
                    &format!("padded validity mask '{}'", output.name),
                    &[packed.len(), max_patches],
                )?;
                let mut values = try_filled_vec(
                    element_count,
                    0_i64,
                    &format!("padded validity mask '{}'", output.name),
                )?;
                for (image_index, image) in packed.iter().enumerate() {
                    let start = image_index
                        .checked_mul(max_patches)
                        .context("padded validity-mask offset overflowed")?;
                    let end = start
                        .checked_add(image.patch_count)
                        .context("padded validity-mask range overflowed")?;
                    values[start..end].fill(1);
                }
                (
                    vec![
                        to_i64(packed.len(), "image batch")?,
                        to_i64(max_patches, "padded patch count")?,
                    ],
                    values,
                )
            }
            Some(_) => {
                ensure_element_limit(total_patches, "validity mask element count")?;
                (
                    vec![to_i64(total_patches, "total patch count")?],
                    try_filled_vec(
                        total_patches,
                        1,
                        &format!("validity mask '{}'", output.name),
                    )?,
                )
            }
            None => {
                let count = prepared.iter().try_fold(0usize, |total, image| {
                    total
                        .checked_add(image.tiles.len())
                        .context("image tile count overflowed")
                })?;
                ensure_element_limit(count, "validity mask element count")?;
                (
                    vec![to_i64(count, "image tile count")?],
                    try_filled_vec(count, 1, &format!("validity mask '{}'", output.name))?,
                )
            }
        }
    };
    Ok(NamedImageTensor {
        name: output.name.clone(),
        content: output.content.clone(),
        dtype: output.dtype,
        shape,
        data: convert_i64(values, output.dtype, &output.name)?,
    })
}

fn expansion_summaries(
    prepared: &[PreparedImage],
    packed: Option<&[PackedImage]>,
    max_patches: usize,
    padded: bool,
) -> anyhow::Result<Vec<ImageExpansionSummary>> {
    let mut offset = 0;
    let mut summaries = try_vec_with_capacity(prepared.len(), "image expansion summaries")?;
    for (image_index, image) in prepared.iter().enumerate() {
        let expansion_count = packed
            .map(|packed| packed[image_index].patch_count)
            .unwrap_or(image.tiles.len());
        let tensor_length = if padded { max_patches } else { expansion_count };
        summaries.push(ImageExpansionSummary {
            image_index,
            original_size: image.original_size,
            tile_grid: image.tile_grid,
            tile_count: image.tiles.len(),
            expansion_count,
            tensor_offset: offset,
            tensor_length,
        });
        offset = offset
            .checked_add(tensor_length)
            .context("image expansion tensor offset overflowed")?;
    }
    Ok(summaries)
}

fn validate_declared_shape(name: &str, actual: &[i64], declared: &[i64]) -> anyhow::Result<()> {
    if actual.len() != declared.len() {
        anyhow::bail!(
            "image pixel output '{name}' produced rank {} shape {actual:?}, but the model declares rank {} shape {declared:?}",
            actual.len(),
            declared.len()
        );
    }
    for (axis, (&actual_dimension, &declared_dimension)) in actual.iter().zip(declared).enumerate()
    {
        if declared_dimension == 0 || declared_dimension < -1 {
            anyhow::bail!(
                "image pixel output '{name}' has invalid declared dimension {declared_dimension} at axis {axis}"
            );
        }
        if declared_dimension > 0 && actual_dimension != declared_dimension {
            anyhow::bail!(
                "image pixel output '{name}' produced shape {actual:?}, but the model declares {declared:?}; axis {axis} expected {declared_dimension}, got {actual_dimension}"
            );
        }
    }
    Ok(())
}

fn convert_f32(
    values: Vec<f32>,
    dtype: ImageTensorDType,
    name: &str,
) -> anyhow::Result<ImageTensorData> {
    match dtype {
        ImageTensorDType::Fp32 => Ok(ImageTensorData::Fp32(values)),
        ImageTensorDType::Fp16 => {
            let mut converted =
                try_vec_with_capacity(values.len(), &format!("fp16 pixel output '{name}'"))?;
            converted.extend(values.into_iter().map(f32_to_f16_bits));
            Ok(ImageTensorData::Fp16(converted))
        }
        ImageTensorDType::Bf16 => {
            let mut converted =
                try_vec_with_capacity(values.len(), &format!("bf16 pixel output '{name}'"))?;
            converted.extend(values.into_iter().map(f32_to_bf16_bits));
            Ok(ImageTensorData::Bf16(converted))
        }
        _ => anyhow::bail!(
            "image pixel output '{name}' must declare fp32, fp16, or bf16, not {dtype:?}"
        ),
    }
}

fn convert_i64(
    values: Vec<i64>,
    dtype: ImageTensorDType,
    name: &str,
) -> anyhow::Result<ImageTensorData> {
    match dtype {
        ImageTensorDType::Fp32 => {
            let mut converted =
                try_vec_with_capacity(values.len(), &format!("fp32 image output '{name}'"))?;
            converted.extend(values.into_iter().map(|value| value as f32));
            Ok(ImageTensorData::Fp32(converted))
        }
        ImageTensorDType::Fp16 => {
            let mut converted =
                try_vec_with_capacity(values.len(), &format!("fp16 image output '{name}'"))?;
            converted.extend(
                values
                    .into_iter()
                    .map(|value| f32_to_f16_bits(value as f32)),
            );
            Ok(ImageTensorData::Fp16(converted))
        }
        ImageTensorDType::Bf16 => {
            let mut converted =
                try_vec_with_capacity(values.len(), &format!("bf16 image output '{name}'"))?;
            converted.extend(
                values
                    .into_iter()
                    .map(|value| f32_to_bf16_bits(value as f32)),
            );
            Ok(ImageTensorData::Bf16(converted))
        }
        ImageTensorDType::Int64 => Ok(ImageTensorData::Int64(values)),
        ImageTensorDType::Int32 => {
            let mut converted =
                try_vec_with_capacity(values.len(), &format!("int32 image output '{name}'"))?;
            for value in values {
                converted.push(i32::try_from(value).with_context(|| {
                    format!("image output '{name}' value {value} does not fit declared int32")
                })?);
            }
            Ok(ImageTensorData::Int32(converted))
        }
        ImageTensorDType::Int8 => {
            let mut converted =
                try_vec_with_capacity(values.len(), &format!("int8 image output '{name}'"))?;
            for value in values {
                converted.push(i8::try_from(value).with_context(|| {
                    format!("image output '{name}' value {value} does not fit declared int8")
                })?);
            }
            Ok(ImageTensorData::Int8(converted))
        }
        ImageTensorDType::Uint8 => {
            let mut converted =
                try_vec_with_capacity(values.len(), &format!("uint8 image output '{name}'"))?;
            for value in values {
                converted.push(u8::try_from(value).with_context(|| {
                    format!("image output '{name}' value {value} does not fit declared uint8")
                })?);
            }
            Ok(ImageTensorData::Uint8(converted))
        }
        ImageTensorDType::Bool => {
            let mut converted =
                try_vec_with_capacity(values.len(), &format!("bool image output '{name}'"))?;
            for value in values {
                converted.push(match value {
                    0 => 0,
                    1 => 1,
                    _ => anyhow::bail!(
                        "image output '{name}' value {value} cannot be represented as bool; expected 0 or 1"
                    ),
                });
            }
            Ok(ImageTensorData::Bool(converted))
        }
    }
}

fn exact_i64(value: f64, name: &str) -> anyhow::Result<i64> {
    if !value.is_finite()
        || value.fract() != 0.0
        || value < i64::MIN as f64
        || value > i64::MAX as f64
    {
        anyhow::bail!(
            "image output '{name}' pad value {value} cannot be represented as an integer"
        );
    }
    Ok(value as i64)
}

fn to_i64(value: usize, description: &str) -> anyhow::Result<i64> {
    i64::try_from(value).with_context(|| format!("{description} is too large"))
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;

    if exponent == 0xff {
        return sign | if mantissa == 0 { 0x7c00 } else { 0x7e00 };
    }

    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x80_0000;
        let shift = 14 - half_exponent;
        let rounded = (mantissa + (1 << (shift - 1)) - 1 + ((mantissa >> shift) & 1)) >> shift;
        return sign | rounded as u16;
    }

    let rounded = mantissa + 0xfff + ((mantissa >> 13) & 1);
    if rounded & 0x80_0000 != 0 {
        let exponent = half_exponent + 1;
        if exponent >= 0x1f {
            sign | 0x7c00
        } else {
            sign | ((exponent as u16) << 10)
        }
    } else {
        sign | ((half_exponent as u16) << 10) | ((rounded >> 13) as u16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_products_reject_arithmetic_overflow() {
        let error = checked_element_product("test image tensor", &[usize::MAX, 2]).unwrap_err();
        assert!(error.to_string().contains("element count overflowed"));
    }

    #[test]
    fn allocation_products_reject_the_explicit_element_limit() {
        let error =
            checked_element_product("test image tensor", &[MAX_TENSOR_ELEMENTS + 1]).unwrap_err();
        assert!(error.to_string().contains("exceeding the safety limit"));
    }

    #[test]
    fn temporal_patches_repeat_the_complete_spatial_tile() {
        let image = PreparedImage {
            original_size: (2, 2),
            transformed_size: (2, 2),
            tile_grid: TileGrid {
                columns: 1,
                rows: 1,
            },
            tile_size: (2, 2),
            tiles: vec![vec![
                1.0, 2.0, 3.0, 4.0, // channel 0
                5.0, 6.0, 7.0, 8.0, // channel 1
                9.0, 10.0, 11.0, 12.0, // channel 2
            ]],
            validity_masks: None,
        };
        let patchify = |channel_order| PatchifySpec {
            patch_size: 2,
            temporal_patch_size: 2,
            merge_size: 1,
            channel_order,
            coordinate_order: CoordinateOrder::Yx,
        };

        let channels_first =
            pack_image(&image, &patchify(PatchChannelOrder::ChannelsFirst)).unwrap();
        assert_eq!(
            channels_first.patches,
            vec![
                1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0, 9.0, 10.0, 11.0, 12.0,
            ]
        );

        let channels_last = pack_image(&image, &patchify(PatchChannelOrder::ChannelsLast)).unwrap();
        assert_eq!(
            channels_last.patches,
            vec![
                1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0, 1.0, 5.0, 9.0, 2.0,
                6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
            ]
        );
    }

    #[test]
    fn output_bundle_rejects_an_aggregate_above_the_element_limit() {
        let output = |name: &str| OutputSpec {
            name: name.to_owned(),
            content: "pixels".to_owned(),
            dtype: ImageTensorDType::Fp32,
            pad_value: None,
            optional: false,
        };
        let spec = PackSpec {
            layout: ImageLayout::Nchw,
            patchify: None,
            pad_value: None,
            target_length: None,
            outputs: vec![output("pixels_a"), output("pixels_b")],
            declared_pixel_shape: vec![-1, 3, 4_096, 4_096],
        };
        let prepared = [PreparedImage {
            original_size: (4_096, 4_096),
            transformed_size: (4_096, 4_096),
            tile_grid: TileGrid {
                columns: 1,
                rows: 1,
            },
            tile_size: (4_096, 4_096),
            tiles: Vec::new(),
            validity_masks: None,
        }];

        let error = validate_total_output_elements(&prepared, None, None, 0, 0, 1, false, &spec)
            .unwrap_err();
        assert!(error.to_string().contains("across declared tensors"));
    }
}
