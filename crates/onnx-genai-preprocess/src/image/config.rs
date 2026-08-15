use anyhow::Context;

/// Tensor channel layout declared by the model input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageLayout {
    Nchw,
    Nhwc,
}

/// Image resizing strategy selected by §35 metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMode {
    ShortestEdgeCenterCrop,
    Fixed,
    LongestEdgePad,
    PixelArea,
    PatchBudget,
}

/// Resize interpolation selected by §35 metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Bicubic,
    Bilinear,
    Lanczos3,
}

/// Image tiling strategy selected by §35 metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TilingMode {
    None,
    FixedGrid,
    DynamicAnyres,
    DynamicHd,
}

/// A tile grid expressed as columns × rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileGrid {
    pub columns: u32,
    pub rows: u32,
}

impl TileGrid {
    pub(super) fn tile_count(self) -> anyhow::Result<usize> {
        let count = self
            .columns
            .checked_mul(self.rows)
            .context("image tile grid is too large")?;
        usize::try_from(count).context("image tile grid is too large")
    }
}

/// Placement of an optional global-thumbnail token segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThumbnailPosition {
    None,
    Prepend,
    Append,
}

/// Configuration for expanding one prompt image placeholder per preprocessed image.
///
/// Each local tile emits `tokens_per_tile` copies of `image_token_id`. Optional
/// column separators are emitted between tiles in a row, and optional row
/// separators are emitted between rows. A global thumbnail emits one additional
/// tile-sized segment before or after the local grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenExpansionConfig {
    pub image_placeholder_token_id: i64,
    pub image_token_id: i64,
    pub tokens_per_tile: usize,
    pub thumbnail_position: ThumbnailPosition,
    pub row_separator_token_id: Option<i64>,
    pub column_separator_token_id: Option<i64>,
}

/// Tile metadata required to expand image placeholders without image tensor data.
#[derive(Debug, Clone, Copy)]
pub struct ImageTilingSummary<'a> {
    pub num_tiles: usize,
    pub tiles_per_image: &'a [usize],
    /// Local grids corresponding one-to-one with `tiles_per_image`.
    pub tile_grids: &'a [TileGrid],
    /// Thumbnail position as stored in the image tensor.
    ///
    /// This is the authoritative ordering: the thumbnail tile appears at this
    /// position within each image's tile slice of the tensor. Token expansion
    /// must use the same ordering so that token indices line up with tile
    /// (embedding) indices. Must match `TokenExpansionConfig::thumbnail_position`.
    pub thumbnail_position: ThumbnailPosition,
}

/// Resolved image tiling parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageTilingConfig {
    pub mode: TilingMode,
    pub tile_size: u32,
    /// Maximum local grid tiles; an enabled global thumbnail is additional.
    pub max_tiles: usize,
    pub aspect_ratios: Vec<TileGrid>,
    pub include_thumbnail: bool,
    pub thumbnail_position: ThumbnailPosition,
}

/// Pixel normalization selected by §35 metadata.
#[derive(Debug, Clone, PartialEq)]
pub enum Normalization {
    ZeroToOne,
    MeanStd { mean: [f32; 3], std: [f32; 3] },
}

/// Resolved image preprocessing parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ImagePreprocessConfig {
    pub width: u32,
    pub height: u32,
    pub resize_mode: ResizeMode,
    pub interpolation: Interpolation,
    pub tiling: ImageTilingConfig,
    pub normalization: Normalization,
}

/// Replaces each image placeholder in a prompt with its image's tile token sequence.
///
/// Placeholders are matched to images in prompt order. The number of placeholder
/// occurrences must exactly match `tiling.tiles_per_image.len()`. The returned
/// token IDs are ready for the caller to pass to its scheduler/decoder; wiring
/// this function between tokenization and sequence-length/KV allocation is the
/// responsibility of the engine or server.
pub fn expand_image_placeholders(
    prompt_token_ids: &[i64],
    tiling: ImageTilingSummary<'_>,
    config: &TokenExpansionConfig,
) -> anyhow::Result<Vec<i64>> {
    validate_token_expansion(tiling, config)?;

    let placeholder_count = prompt_token_ids
        .iter()
        .filter(|token_id| **token_id == config.image_placeholder_token_id)
        .count();
    if placeholder_count != tiling.tiles_per_image.len() {
        anyhow::bail!(
            "prompt contains {placeholder_count} image placeholder(s), but preprocessing produced {} image(s)",
            tiling.tiles_per_image.len()
        );
    }

    let mut replacements = Vec::with_capacity(tiling.tile_grids.len());
    let mut replacement_tokens = 0usize;
    for grid in tiling.tile_grids {
        let replacement = expanded_image_tokens(*grid, tiling.thumbnail_position, config)?;
        replacement_tokens = replacement_tokens
            .checked_add(replacement.len())
            .context("expanded image token sequence is too large")?;
        replacements.push(replacement);
    }

    let output_len = prompt_token_ids
        .len()
        .checked_sub(placeholder_count)
        .and_then(|length| length.checked_add(replacement_tokens))
        .context("expanded prompt token sequence is too large")?;
    let mut expanded = Vec::new();
    expanded
        .try_reserve_exact(output_len)
        .context("failed to allocate expanded prompt token sequence")?;
    let mut image_index = 0usize;
    for token_id in prompt_token_ids {
        if *token_id == config.image_placeholder_token_id {
            expanded.extend_from_slice(&replacements[image_index]);
            image_index += 1;
        } else {
            expanded.push(*token_id);
        }
    }
    Ok(expanded)
}

fn validate_token_expansion(
    tiling: ImageTilingSummary<'_>,
    config: &TokenExpansionConfig,
) -> anyhow::Result<()> {
    if config.tokens_per_tile == 0 {
        anyhow::bail!("tokens_per_tile must be greater than zero");
    }
    if tiling.tiles_per_image.len() != tiling.tile_grids.len() {
        anyhow::bail!(
            "tiles_per_image has {} entries, but tile_grids has {}",
            tiling.tiles_per_image.len(),
            tiling.tile_grids.len()
        );
    }
    // The config thumbnail position must match the actual tensor layout so that
    // emitted token indices align with tile (embedding) indices in the tensor.
    if config.thumbnail_position != tiling.thumbnail_position {
        anyhow::bail!(
            "config thumbnail_position {:?} does not match tensor thumbnail_position {:?}; \
             token order must match the tile order stored in the image tensor",
            config.thumbnail_position,
            tiling.thumbnail_position,
        );
    }

    let thumbnail_tiles = usize::from(tiling.thumbnail_position != ThumbnailPosition::None);
    let mut total_tiles = 0usize;
    for (image_index, (&actual_tiles, grid)) in tiling
        .tiles_per_image
        .iter()
        .zip(tiling.tile_grids)
        .enumerate()
    {
        if grid.columns == 0 || grid.rows == 0 {
            anyhow::bail!("image {image_index} tile grid dimensions must be greater than zero");
        }
        let expected_tiles = grid
            .tile_count()?
            .checked_add(thumbnail_tiles)
            .context("image tile count is too large")?;
        if actual_tiles != expected_tiles {
            anyhow::bail!(
                "image {image_index} reports {actual_tiles} tile(s), but its {}x{} grid and thumbnail configuration require {expected_tiles}",
                grid.columns,
                grid.rows
            );
        }
        total_tiles = total_tiles
            .checked_add(actual_tiles)
            .context("total image tile count is too large")?;
    }
    if total_tiles != tiling.num_tiles {
        anyhow::bail!(
            "tiling summary reports {} total tile(s), but tiles_per_image sums to {total_tiles}",
            tiling.num_tiles
        );
    }
    Ok(())
}

fn expanded_image_tokens(
    grid: TileGrid,
    thumbnail_position: ThumbnailPosition,
    config: &TokenExpansionConfig,
) -> anyhow::Result<Vec<i64>> {
    let local_tiles = grid.tile_count()?;
    let thumbnail_tiles = usize::from(thumbnail_position != ThumbnailPosition::None);
    let separator_count = usize::from(config.column_separator_token_id.is_some())
        .checked_mul(local_tiles.saturating_sub(grid.rows as usize))
        .and_then(|count| {
            usize::from(config.row_separator_token_id.is_some())
                .checked_mul((grid.rows as usize).saturating_sub(1))
                .and_then(|rows| count.checked_add(rows))
        })
        .context("expanded image separator count is too large")?;
    let capacity = local_tiles
        .checked_add(thumbnail_tiles)
        .and_then(|tiles| tiles.checked_mul(config.tokens_per_tile))
        .and_then(|tokens| tokens.checked_add(separator_count))
        .context("expanded image token sequence is too large")?;
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(capacity)
        .context("failed to allocate expanded image token sequence")?;

    let emit_tile = |tokens: &mut Vec<i64>| {
        tokens.extend(std::iter::repeat_n(
            config.image_token_id,
            config.tokens_per_tile,
        ));
    };
    if thumbnail_position == ThumbnailPosition::Prepend {
        emit_tile(&mut tokens);
    }
    for row in 0..grid.rows {
        for column in 0..grid.columns {
            emit_tile(&mut tokens);
            if column + 1 < grid.columns
                && let Some(separator) = config.column_separator_token_id
            {
                tokens.push(separator);
            }
        }
        if row + 1 < grid.rows
            && let Some(separator) = config.row_separator_token_id
        {
            tokens.push(separator);
        }
    }
    if thumbnail_position == ThumbnailPosition::Append {
        emit_tile(&mut tokens);
    }
    Ok(tokens)
}
