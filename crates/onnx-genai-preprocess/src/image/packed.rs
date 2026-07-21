//! Typed image tensor bundles and generic patch packing.

use std::collections::BTreeSet;

use anyhow::Context;

use super::{ImageLayout, ThumbnailPosition, TileGrid};

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
    pub tile_grid: TileGrid,
    /// Channel-first normalized tiles.
    pub tiles: Vec<Vec<f32>>,
}

#[derive(Debug)]
pub(super) struct PackSpec {
    pub width: usize,
    pub height: usize,
    pub layout: ImageLayout,
    pub patch_size: Option<usize>,
    pub pad_value: Option<f64>,
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
    let num_tiles = prepared.iter().try_fold(0usize, |total, image| {
        total
            .checked_add(image.tiles.len())
            .context("image tile count is too large")
    })?;
    let tiles_per_image = prepared.iter().map(|image| image.tiles.len()).collect();
    let tile_grids = prepared.iter().map(|image| image.tile_grid).collect();

    let (packed, feature_size) = match spec.patch_size {
        Some(patch_size) => {
            if patch_size == 0 {
                anyhow::bail!("image patchify patch_size must be greater than zero");
            }
            if !spec.width.is_multiple_of(patch_size) || !spec.height.is_multiple_of(patch_size) {
                anyhow::bail!(
                    "image dimensions {}x{} are not divisible by declared patch_size {patch_size}",
                    spec.width,
                    spec.height
                );
            }
            let feature_size = 3usize
                .checked_mul(patch_size)
                .and_then(|value| value.checked_mul(patch_size))
                .context("image patch feature dimension is too large")?;
            let packed = prepared
                .iter()
                .map(|image| pack_image(image, spec.width, spec.height, patch_size))
                .collect::<anyhow::Result<Vec<_>>>()?;
            (Some(packed), Some(feature_size))
        }
        None => (None, None),
    };

    let max_patches = packed
        .as_ref()
        .and_then(|images| images.iter().map(|image| image.patch_count).max())
        .unwrap_or(0);
    let total_patches = packed
        .as_ref()
        .map(|images| images.iter().map(|image| image.patch_count).sum())
        .unwrap_or(0);
    let padded = spec.patch_size.is_some() && spec.pad_value.is_some();

    let mut tensors = Vec::with_capacity(spec.outputs.len());
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

    let images = expansion_summaries(&prepared, packed.as_deref(), max_patches, padded);
    Ok(ImageTensorBundle {
        tensors,
        images,
        num_tiles,
        tiles_per_image,
        tile_grids,
        thumbnail_position,
    })
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

fn pack_image(
    image: &PreparedImage,
    width: usize,
    height: usize,
    patch_size: usize,
) -> anyhow::Result<PackedImage> {
    let patches_w = width / patch_size;
    let patches_h = height / patch_size;
    let per_tile = patches_w
        .checked_mul(patches_h)
        .context("image patch count is too large")?;
    let patch_count = image
        .tiles
        .len()
        .checked_mul(per_tile)
        .context("image patch count is too large")?;
    let feature_size = 3 * patch_size * patch_size;
    let mut patches = Vec::with_capacity(patch_count * feature_size);
    let mut coordinates = Vec::with_capacity(patch_count * 2);
    for (tile_index, tile) in image.tiles.iter().enumerate() {
        if tile.len() != 3 * width * height {
            anyhow::bail!(
                "normalized image tile has {} values, expected {} for RGB {}x{}",
                tile.len(),
                3 * width * height,
                width,
                height
            );
        }
        for patch_y in 0..patches_h {
            for patch_x in 0..patches_w {
                for channel in 0..3 {
                    let channel_offset = channel * width * height;
                    for y in 0..patch_size {
                        let row = (patch_y * patch_size + y) * width;
                        let start = channel_offset + row + patch_x * patch_size;
                        patches.extend_from_slice(&tile[start..start + patch_size]);
                    }
                }
                coordinates.push(
                    i64::try_from(tile_index * patches_h + patch_y)
                        .context("image patch row coordinate is too large")?,
                );
                coordinates.push(
                    i64::try_from(patch_x).context("image patch column coordinate is too large")?,
                );
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
            let mut values = vec![fill; prepared.len() * max_patches * feature_size];
            for (image_index, image) in packed.iter().enumerate() {
                let start = image_index * max_patches * feature_size;
                values[start..start + image.patches.len()].copy_from_slice(&image.patches);
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
            let mut values = Vec::with_capacity(total_patches * feature_size);
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
        let mut values = Vec::with_capacity(
            prepared
                .iter()
                .map(|image| image.tiles.len())
                .sum::<usize>()
                * 3
                * spec.width
                * spec.height,
        );
        for image in prepared {
            for tile in &image.tiles {
                match spec.layout {
                    ImageLayout::Nchw => values.extend_from_slice(tile),
                    ImageLayout::Nhwc => {
                        for pixel in 0..spec.width * spec.height {
                            for channel in 0..3 {
                                values.push(tile[channel * spec.width * spec.height + pixel]);
                            }
                        }
                    }
                }
            }
        }
        let tile_count = prepared.iter().map(|image| image.tiles.len()).sum();
        let shape = match spec.layout {
            ImageLayout::Nchw => vec![
                to_i64(tile_count, "image tile count")?,
                3,
                to_i64(spec.height, "image height")?,
                to_i64(spec.width, "image width")?,
            ],
            ImageLayout::Nhwc => vec![
                to_i64(tile_count, "image tile count")?,
                to_i64(spec.height, "image height")?,
                to_i64(spec.width, "image width")?,
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
        let mut values = vec![sentinel; packed.len() * max_patches * 2];
        for (image_index, image) in packed.iter().enumerate() {
            let start = image_index * max_patches * 2;
            values[start..start + image.coordinates.len()].copy_from_slice(&image.coordinates);
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
        let mut values = Vec::with_capacity(total_patches * 2);
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
    let values = packed
        .iter()
        .flat_map(|image| image.grid)
        .collect::<Vec<_>>();
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
    let values = prepared
        .iter()
        .flat_map(|image| {
            [
                i64::from(image.original_size.1),
                i64::from(image.original_size.0),
            ]
        })
        .collect::<Vec<_>>();
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
    let (shape, values) = match packed {
        Some(packed) if padded => {
            let mut values = vec![0_i64; packed.len() * max_patches];
            for (image_index, image) in packed.iter().enumerate() {
                let start = image_index * max_patches;
                values[start..start + image.patch_count].fill(1);
            }
            (
                vec![
                    to_i64(packed.len(), "image batch")?,
                    to_i64(max_patches, "padded patch count")?,
                ],
                values,
            )
        }
        Some(_) => (
            vec![to_i64(total_patches, "total patch count")?],
            vec![1; total_patches],
        ),
        None => {
            let count = prepared.iter().map(|image| image.tiles.len()).sum();
            (vec![to_i64(count, "image tile count")?], vec![1; count])
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
) -> Vec<ImageExpansionSummary> {
    let mut offset = 0;
    prepared
        .iter()
        .enumerate()
        .map(|(image_index, image)| {
            let expansion_count = packed
                .map(|packed| packed[image_index].patch_count)
                .unwrap_or(image.tiles.len());
            let tensor_length = if padded { max_patches } else { expansion_count };
            let summary = ImageExpansionSummary {
                image_index,
                original_size: image.original_size,
                tile_grid: image.tile_grid,
                tile_count: image.tiles.len(),
                expansion_count,
                tensor_offset: offset,
                tensor_length,
            };
            offset += tensor_length;
            summary
        })
        .collect()
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
        ImageTensorDType::Fp16 => Ok(ImageTensorData::Fp16(
            values.into_iter().map(f32_to_f16_bits).collect(),
        )),
        ImageTensorDType::Bf16 => Ok(ImageTensorData::Bf16(
            values.into_iter().map(f32_to_bf16_bits).collect(),
        )),
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
        ImageTensorDType::Fp32 => Ok(ImageTensorData::Fp32(
            values.into_iter().map(|value| value as f32).collect(),
        )),
        ImageTensorDType::Fp16 => Ok(ImageTensorData::Fp16(
            values
                .into_iter()
                .map(|value| f32_to_f16_bits(value as f32))
                .collect(),
        )),
        ImageTensorDType::Bf16 => Ok(ImageTensorData::Bf16(
            values
                .into_iter()
                .map(|value| f32_to_bf16_bits(value as f32))
                .collect(),
        )),
        ImageTensorDType::Int64 => Ok(ImageTensorData::Int64(values)),
        ImageTensorDType::Int32 => values
            .into_iter()
            .map(|value| {
                i32::try_from(value).with_context(|| {
                    format!("image output '{name}' value {value} does not fit declared int32")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .map(ImageTensorData::Int32),
        ImageTensorDType::Int8 => values
            .into_iter()
            .map(|value| {
                i8::try_from(value).with_context(|| {
                    format!("image output '{name}' value {value} does not fit declared int8")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .map(ImageTensorData::Int8),
        ImageTensorDType::Uint8 => values
            .into_iter()
            .map(|value| {
                u8::try_from(value).with_context(|| {
                    format!("image output '{name}' value {value} does not fit declared uint8")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .map(ImageTensorData::Uint8),
        ImageTensorDType::Bool => values
            .into_iter()
            .map(|value| match value {
                0 => Ok(0),
                1 => Ok(1),
                _ => anyhow::bail!(
                    "image output '{name}' value {value} cannot be represented as bool; expected 0 or 1"
                ),
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .map(ImageTensorData::Bool),
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
