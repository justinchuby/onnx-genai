use anyhow::Context;
use image::{DynamicImage, Rgb, RgbImage};

use super::{
    ImagePreprocessConfig, MAX_TILES_PER_IMAGE, ThumbnailPosition, TileGrid, TilingMode,
    program::{DynamicHdSpec, ImageProgram, validate_image_dimensions},
    transform::{resize_image, resize_image_to, resize_rgb},
};

pub(super) struct TiledImage {
    pub(super) grid: TileGrid,
    pub(super) transformed_size: (u32, u32),
    pub(super) tiles: Vec<RgbImage>,
    pub(super) validity_masks: Option<Vec<Vec<u8>>>,
}

pub(super) fn tile_image(
    image: &DynamicImage,
    config: &ImagePreprocessConfig,
    program: &ImageProgram,
) -> anyhow::Result<TiledImage> {
    match config.tiling.mode {
        TilingMode::None => {
            let resized = resize_image(image, config, program.dynamic_resize.as_ref())?;
            let transformed_size = resized.dimensions();
            Ok(TiledImage {
                grid: TileGrid {
                    columns: 1,
                    rows: 1,
                },
                transformed_size,
                tiles: vec![resized],
                validity_masks: None,
            })
        }
        TilingMode::FixedGrid => {
            let grid = config.tiling.aspect_ratios[0];
            tiled_image_for_grid(image, config, grid)
        }
        TilingMode::DynamicAnyres => {
            let grid = select_best_grid(
                image.width(),
                image.height(),
                config.tiling.tile_size,
                config.tiling.max_tiles,
                &config.tiling.aspect_ratios,
            )?;
            tiled_image_for_grid(image, config, grid)
        }
        TilingMode::DynamicHd => dynamic_hd_image(
            image,
            config,
            program
                .dynamic_hd
                .as_ref()
                .context("dynamic_hd tile metadata is missing")?,
        ),
    }
}

fn dynamic_hd_image(
    image: &DynamicImage,
    config: &ImagePreprocessConfig,
    dynamic_hd: &DynamicHdSpec,
) -> anyhow::Result<TiledImage> {
    let tile_size = config.tiling.tile_size;
    let mut columns = image.width().div_ceil(tile_size);
    let mut rows = image.height().div_ceil(tile_size);
    if usize::try_from(columns)
        .ok()
        .and_then(|columns| {
            usize::try_from(rows)
                .ok()
                .and_then(|rows| columns.checked_mul(rows))
        })
        .is_none_or(|count| count > config.tiling.max_tiles)
    {
        let aspect_ratio = image.width() as f64 / image.height() as f64;
        let area = u64::from(image.width()) * u64::from(image.height());
        let mut candidates = Vec::new();
        for candidate_columns in 1..=config.tiling.max_tiles {
            for candidate_rows in 1..=config.tiling.max_tiles / candidate_columns {
                candidates.push((candidate_columns, candidate_rows));
            }
        }
        candidates.sort_unstable_by_key(|&(columns, rows)| (columns * rows, columns, rows));
        candidates.dedup();
        let mut best = None;
        for (candidate_columns, candidate_rows) in candidates {
            let difference =
                (aspect_ratio - candidate_columns as f64 / candidate_rows as f64).abs();
            let prefer_larger = area
                > u64::from(tile_size)
                    * u64::from(tile_size)
                    * candidate_columns as u64
                    * candidate_rows as u64
                    / 2;
            if best.is_none_or(|(_, _, best_difference): (usize, usize, f64)| {
                difference < best_difference || (difference == best_difference && prefer_larger)
            }) {
                best = Some((candidate_columns, candidate_rows, difference));
            }
        }
        let (best_columns, best_rows, _) =
            best.context("dynamic_hd could not resolve a tile grid")?;
        columns = u32::try_from(best_columns).context("dynamic_hd grid width is too large")?;
        rows = u32::try_from(best_rows).context("dynamic_hd grid height is too large")?;
    }
    let grid = TileGrid { columns, rows };
    let canvas_width = columns
        .checked_mul(tile_size)
        .context("dynamic_hd canvas width is too large")?;
    let canvas_height = rows
        .checked_mul(tile_size)
        .context("dynamic_hd canvas height is too large")?;
    validate_image_dimensions(canvas_width, canvas_height, "dynamic_hd canvas")?;

    let width_ratio = canvas_width as f64 / image.width() as f64;
    let height_ratio = canvas_height as f64 / image.height() as f64;
    let (resized_width, resized_height) = if width_ratio < height_ratio {
        (
            canvas_width,
            (image.height() as f64 * width_ratio).floor() as u32,
        )
    } else {
        (
            (image.width() as f64 * height_ratio).floor() as u32,
            canvas_height,
        )
    };
    if resized_width < 10 || resized_height < 10 {
        anyhow::bail!(
            "dynamic_hd resize produced extreme dimensions {resized_width}x{resized_height}; provide a less extreme source image"
        );
    }
    let resized = resize_rgb(
        &image.to_rgb8(),
        resized_width,
        resized_height,
        config.interpolation,
    )?;
    let mut canvas = RgbImage::from_pixel(
        canvas_width,
        canvas_height,
        Rgb([dynamic_hd.canvas_pad_value; 3]),
    );
    image::imageops::replace(&mut canvas, &resized, 0, 0);

    let mask_edge = tile_size as usize / dynamic_hd.mask_patch_size;
    let mask_width = columns as usize * mask_edge;
    let mask_height = rows as usize * mask_edge;
    let mut canvas_mask = vec![1_u8; mask_width * mask_height];
    let padding_width = canvas_width - resized_width;
    let padding_height = canvas_height - resized_height;
    let invalid_columns = padding_width as usize / dynamic_hd.mask_patch_size;
    let invalid_rows = padding_height as usize / dynamic_hd.mask_patch_size;
    if invalid_columns > 0 {
        for row in 0..mask_height {
            canvas_mask[row * mask_width + mask_width - invalid_columns..(row + 1) * mask_width]
                .fill(0);
        }
    }
    if invalid_rows > 0 {
        canvas_mask[(mask_height - invalid_rows) * mask_width..].fill(0);
    }

    let local_count = grid.tile_count()?;
    let total_count = local_count
        .checked_add(usize::from(config.tiling.include_thumbnail))
        .context("dynamic_hd tile count overflowed")?;
    let mut tiles = Vec::new();
    let mut masks = Vec::new();
    tiles
        .try_reserve_exact(total_count)
        .context("failed to allocate dynamic_hd tiles")?;
    masks
        .try_reserve_exact(total_count)
        .context("failed to allocate dynamic_hd masks")?;
    let thumbnail = resize_rgb(
        &canvas,
        tile_size,
        tile_size,
        dynamic_hd.thumbnail_interpolation,
    )?;
    let thumbnail_mask = vec![1_u8; mask_edge * mask_edge];
    if config.tiling.thumbnail_position == ThumbnailPosition::Prepend {
        tiles.push(thumbnail.clone());
        masks.push(thumbnail_mask.clone());
    }
    for row in 0..rows {
        for column in 0..columns {
            tiles.push(
                image::imageops::crop_imm(
                    &canvas,
                    column * tile_size,
                    row * tile_size,
                    tile_size,
                    tile_size,
                )
                .to_image(),
            );
            let mut mask = Vec::with_capacity(mask_edge * mask_edge);
            for mask_row in 0..mask_edge {
                let start = (row as usize * mask_edge + mask_row) * mask_width
                    + column as usize * mask_edge;
                mask.extend_from_slice(&canvas_mask[start..start + mask_edge]);
            }
            masks.push(mask);
        }
    }
    if config.tiling.thumbnail_position == ThumbnailPosition::Append {
        tiles.push(thumbnail);
        masks.push(thumbnail_mask);
    }
    Ok(TiledImage {
        grid,
        transformed_size: (canvas_width, canvas_height),
        tiles,
        validity_masks: Some(masks),
    })
}

fn tiled_image_for_grid(
    image: &DynamicImage,
    config: &ImagePreprocessConfig,
    grid: TileGrid,
) -> anyhow::Result<TiledImage> {
    let tile_size = config.tiling.tile_size;
    let width = grid
        .columns
        .checked_mul(tile_size)
        .context("tiled image width is too large")?;
    let height = grid
        .rows
        .checked_mul(tile_size)
        .context("tiled image height is too large")?;
    validate_image_dimensions(width, height, "tiled image canvas")?;
    let resized = resize_image_to(image, config, width, height)?;
    let local_count = grid.tile_count()?;
    let tile_count = local_count
        .checked_add(usize::from(config.tiling.include_thumbnail))
        .context("image tile count overflowed")?;
    if tile_count > MAX_TILES_PER_IMAGE + 1 {
        anyhow::bail!(
            "image tiling produces {tile_count} tiles, exceeding the supported limit of {}; reduce max_tiles or the configured grid",
            MAX_TILES_PER_IMAGE + 1
        );
    }
    let mut tiles = Vec::new();
    tiles
        .try_reserve_exact(tile_count)
        .context("failed to allocate image tile batch")?;
    let thumbnail = resize_image_to(image, config, tile_size, tile_size)?;
    if config.tiling.thumbnail_position == ThumbnailPosition::Prepend {
        tiles.push(thumbnail.clone());
    }
    for row in 0..grid.rows {
        for column in 0..grid.columns {
            tiles.push(
                image::imageops::crop_imm(
                    &resized,
                    column * tile_size,
                    row * tile_size,
                    tile_size,
                    tile_size,
                )
                .to_image(),
            );
        }
    }
    if config.tiling.thumbnail_position == ThumbnailPosition::Append {
        tiles.push(thumbnail);
    }
    Ok(TiledImage {
        grid,
        transformed_size: (width, height),
        tiles,
        validity_masks: None,
    })
}

/// Selects the LLaVA-style best resolution.
///
/// Candidates exceeding `max_tiles` are ignored. Remaining candidates maximize
/// effective source pixels after aspect-preserving fit, then minimize padded or
/// cropped canvas pixels. Configuration order breaks any remaining tie.
pub(super) fn select_best_grid(
    image_width: u32,
    image_height: u32,
    tile_size: u32,
    max_tiles: usize,
    grids: &[TileGrid],
) -> anyhow::Result<TileGrid> {
    let original_area = u64::from(image_width) * u64::from(image_height);
    let mut best = None;
    for grid in grids.iter().copied() {
        let Some((effective, wasted)) = (|| {
            let tile_count = grid.tile_count().ok()?;
            if tile_count > max_tiles {
                return None;
            }
            let candidate_width = grid.columns.checked_mul(tile_size)?;
            let candidate_height = grid.rows.checked_mul(tile_size)?;
            let scale = (candidate_width as f64 / image_width as f64)
                .min(candidate_height as f64 / image_height as f64);
            let fitted_width = (image_width as f64 * scale).floor() as u64;
            let fitted_height = (image_height as f64 * scale).floor() as u64;
            let effective = (fitted_width * fitted_height).min(original_area);
            let candidate_area = u64::from(candidate_width) * u64::from(candidate_height);
            let wasted = candidate_area.saturating_sub(effective);
            Some((effective, wasted))
        })() else {
            continue;
        };
        if best.is_none_or(|(_, best_effective, best_wasted)| {
            effective > best_effective || (effective == best_effective && wasted < best_wasted)
        }) {
            best = Some((grid, effective, wasted));
        }
    }
    best.map(|(grid, _, _)| grid)
        .context("no image tiling aspect ratio fits max_tiles")
}
