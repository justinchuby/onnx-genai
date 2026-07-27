use anyhow::Context;
use image::{DynamicImage, Rgb, RgbImage, imageops::FilterType};

use super::{
    CHANNELS, ImagePreprocessConfig, Interpolation, ResizeMode,
    program::{DynamicResize, ValueOp, checked_image_elements, validate_image_dimensions},
};

pub(super) fn resize_image(
    image: &DynamicImage,
    config: &ImagePreprocessConfig,
    dynamic_resize: Option<&DynamicResize>,
) -> anyhow::Result<RgbImage> {
    let (width, height) = match dynamic_resize {
        Some(DynamicResize::PixelArea {
            min_pixels,
            max_pixels,
            size_multiple,
        }) => pixel_area_size(
            image.width(),
            image.height(),
            *min_pixels,
            *max_pixels,
            *size_multiple,
        )?,
        Some(DynamicResize::PatchBudget {
            patch_size,
            max_patches,
            pooling_kernel_size,
        }) => patch_budget_size(
            image.width(),
            image.height(),
            *patch_size,
            *max_patches,
            *pooling_kernel_size,
        )?,
        None => (config.width, config.height),
    };
    resize_image_to(image, config, width, height)
}

fn pixel_area_size(
    width: u32,
    height: u32,
    min_pixels: usize,
    max_pixels: usize,
    multiple: usize,
) -> anyhow::Result<(u32, u32)> {
    let aspect = width.max(height) as f64 / width.min(height) as f64;
    if aspect > 200.0 {
        anyhow::bail!("pixel_area resize requires absolute aspect ratio <= 200, got {aspect}");
    }
    let multiple = u32::try_from(multiple).context("pixel_area size_multiple is too large")?;
    let mut resized_height = round_to_multiple_ties_even(height, multiple);
    let mut resized_width = round_to_multiple_ties_even(width, multiple);
    let area = u64::from(resized_width) * u64::from(resized_height);
    if area > max_pixels as u64 {
        let beta = ((u64::from(width) * u64::from(height)) as f64 / max_pixels as f64).sqrt();
        resized_height =
            ((height as f64 / beta / multiple as f64).floor() as u32).max(1) * multiple;
        resized_width = ((width as f64 / beta / multiple as f64).floor() as u32).max(1) * multiple;
    } else if area < min_pixels as u64 {
        let beta = (min_pixels as f64 / (u64::from(width) * u64::from(height)) as f64).sqrt();
        resized_height = ((height as f64 * beta / multiple as f64).ceil() as u32).max(1) * multiple;
        resized_width = ((width as f64 * beta / multiple as f64).ceil() as u32).max(1) * multiple;
    }
    validate_image_dimensions(resized_width, resized_height, "pixel_area resize")?;
    Ok((resized_width, resized_height))
}

pub(super) fn round_to_multiple_ties_even(value: u32, multiple: u32) -> u32 {
    let quotient = value / multiple;
    let remainder = value % multiple;
    let rounded = match remainder.cmp(&(multiple - remainder)) {
        std::cmp::Ordering::Less => quotient,
        std::cmp::Ordering::Greater => quotient + 1,
        std::cmp::Ordering::Equal if quotient.is_multiple_of(2) => quotient,
        std::cmp::Ordering::Equal => quotient + 1,
    };
    rounded.saturating_mul(multiple)
}

fn patch_budget_size(
    width: u32,
    height: u32,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> anyhow::Result<(u32, u32)> {
    let target_pixels = max_patches
        .checked_mul(patch_size)
        .and_then(|value| value.checked_mul(patch_size))
        .context("patch-budget target pixel count overflowed")?;
    let factor = (target_pixels as f64 / (u64::from(width) * u64::from(height)) as f64).sqrt();
    let side_multiple = patch_size
        .checked_mul(pooling_kernel_size)
        .context("patch-budget size multiple overflowed")?;
    let mut target_height =
        (height as f64 * factor / side_multiple as f64).floor() as usize * side_multiple;
    let mut target_width =
        (width as f64 * factor / side_multiple as f64).floor() as usize * side_multiple;
    if target_height == 0 && target_width == 0 {
        anyhow::bail!(
            "patch-budget resize for {width}x{height} rounded both dimensions to zero at size multiple {side_multiple}"
        );
    }
    let pooled_patch_count = pooling_kernel_size
        .checked_mul(pooling_kernel_size)
        .context("patch-budget pooling area overflowed")?;
    let max_side_length = max_patches
        .checked_div(pooled_patch_count)
        .and_then(|value| value.checked_mul(side_multiple))
        .context("patch-budget maximum side length overflowed")?;
    if target_height == 0 {
        target_height = side_multiple;
        target_width = ((width / height) as usize)
            .saturating_mul(side_multiple)
            .min(max_side_length);
    } else if target_width == 0 {
        target_width = side_multiple;
        target_height = ((height / width) as usize)
            .saturating_mul(side_multiple)
            .min(max_side_length);
    }
    if target_height
        .checked_mul(target_width)
        .is_none_or(|pixels| pixels > target_pixels)
    {
        anyhow::bail!(
            "patch-budget resize {target_width}x{target_height} exceeds max_patches {max_patches} at patch_size {patch_size}"
        );
    }
    let target_width = u32::try_from(target_width).context("patch-budget width is too large")?;
    let target_height = u32::try_from(target_height).context("patch-budget height is too large")?;
    validate_image_dimensions(target_width, target_height, "patch-budget resize")?;
    Ok((target_width, target_height))
}

pub(super) fn resize_rgb(
    rgb: &RgbImage,
    width: u32,
    height: u32,
    interpolation: Interpolation,
) -> anyhow::Result<RgbImage> {
    validate_image_dimensions(width, height, "resized image")?;
    let filter = match interpolation {
        Interpolation::Bicubic => FilterType::CatmullRom,
        Interpolation::Bilinear => FilterType::Triangle,
        Interpolation::Lanczos3 => FilterType::Lanczos3,
    };
    Ok(image::imageops::resize(rgb, width, height, filter))
}

pub(super) fn resize_image_to(
    image: &DynamicImage,
    config: &ImagePreprocessConfig,
    width: u32,
    height: u32,
) -> anyhow::Result<RgbImage> {
    validate_image_dimensions(width, height, "resized image")?;
    let rgb = image.to_rgb8();
    let filter = match config.interpolation {
        Interpolation::Bicubic => FilterType::CatmullRom,
        Interpolation::Bilinear => FilterType::Triangle,
        Interpolation::Lanczos3 => FilterType::Lanczos3,
    };
    match config.resize_mode {
        ResizeMode::Fixed | ResizeMode::PixelArea | ResizeMode::PatchBudget => {
            Ok(image::imageops::resize(&rgb, width, height, filter))
        }
        ResizeMode::ShortestEdgeCenterCrop => {
            let scale =
                (width as f64 / rgb.width() as f64).max(height as f64 / rgb.height() as f64);
            let resized_width = ((rgb.width() as f64 * scale).round() as u32).max(width);
            let resized_height = ((rgb.height() as f64 * scale).round() as u32).max(height);
            validate_image_dimensions(
                resized_width,
                resized_height,
                "center-crop intermediate image",
            )?;
            let resized = image::imageops::resize(&rgb, resized_width, resized_height, filter);
            Ok(image::imageops::crop_imm(
                &resized,
                (resized_width - width) / 2,
                (resized_height - height) / 2,
                width,
                height,
            )
            .to_image())
        }
        ResizeMode::LongestEdgePad => {
            let scale =
                (width as f64 / rgb.width() as f64).min(height as f64 / rgb.height() as f64);
            let resized_width = ((rgb.width() as f64 * scale).round() as u32).clamp(1, width);
            let resized_height = ((rgb.height() as f64 * scale).round() as u32).clamp(1, height);
            let resized = image::imageops::resize(&rgb, resized_width, resized_height, filter);
            let mut padded = RgbImage::from_pixel(width, height, Rgb([0, 0, 0]));
            image::imageops::replace(
                &mut padded,
                &resized,
                i64::from((width - resized_width) / 2),
                i64::from((height - resized_height) / 2),
            );
            Ok(padded)
        }
    }
}

pub(super) fn normalize_tile(
    image: &RgbImage,
    width: usize,
    height: usize,
    operations: &[ValueOp],
) -> anyhow::Result<Vec<f32>> {
    let element_count = checked_image_elements(width, height, "normalized image tile")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(element_count)
        .context("failed to allocate normalized image tile")?;
    for channel in 0..CHANNELS {
        values.extend(image.pixels().map(|pixel| {
            operations.iter().fold(
                f32::from(pixel[channel]),
                |value, operation| match operation {
                    ValueOp::Divide(divisor) => value / divisor,
                    ValueOp::Rescale(scale) => value * scale,
                    ValueOp::Normalize { mean, std } => (value - mean[channel]) / std[channel],
                },
            )
        }));
    }
    Ok(values)
}
