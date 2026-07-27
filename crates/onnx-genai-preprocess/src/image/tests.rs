use super::*;
use image::{Rgb, RgbImage};

use crate::image::{
    ImageTensorData, ImageTilingSummary, TokenExpansionConfig, expand_image_placeholders,
    tiling::select_best_grid,
    transform::{normalize_tile, resize_image, round_to_multiple_ties_even},
};

mod hf_reference {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/hf_vlm_reference.rs"
    ));
}

fn token_expansion_config() -> TokenExpansionConfig {
    TokenExpansionConfig {
        image_placeholder_token_id: 99,
        image_token_id: 7,
        tokens_per_tile: 2,
        thumbnail_position: ThumbnailPosition::None,
        row_separator_token_id: None,
        column_separator_token_id: None,
    }
}

#[test]
fn expands_single_untiled_image_placeholder() {
    let config = token_expansion_config();
    let tiles_per_image = [1];
    let grids = [TileGrid {
        columns: 1,
        rows: 1,
    }];

    let expanded = expand_image_placeholders(
        &[1, 99, 2],
        ImageTilingSummary {
            num_tiles: 1,
            tiles_per_image: &tiles_per_image,
            tile_grids: &grids,
            thumbnail_position: ThumbnailPosition::None,
        },
        &config,
    )
    .unwrap();

    assert_eq!(expanded, [1, 7, 7, 2]);
}

#[test]
fn expands_single_image_local_tiles_in_row_major_order() {
    let config = token_expansion_config();
    let tiles_per_image = [6];
    let grids = [TileGrid {
        columns: 3,
        rows: 2,
    }];

    let expanded = expand_image_placeholders(
        &[99],
        ImageTilingSummary {
            num_tiles: 6,
            tiles_per_image: &tiles_per_image,
            tile_grids: &grids,
            thumbnail_position: ThumbnailPosition::None,
        },
        &config,
    )
    .unwrap();

    assert_eq!(expanded, [7; 12]);
}

#[test]
fn expands_tiles_with_appended_global_thumbnail() {
    let mut config = token_expansion_config();
    config.thumbnail_position = ThumbnailPosition::Append;
    config.column_separator_token_id = Some(8);
    let tiles_per_image = [3];
    let grids = [TileGrid {
        columns: 2,
        rows: 1,
    }];

    // tiling.thumbnail_position must match config; here both say Append so
    // that this test exercises the Append code path in expanded_image_tokens.
    let expanded = expand_image_placeholders(
        &[99],
        ImageTilingSummary {
            num_tiles: 3,
            tiles_per_image: &tiles_per_image,
            tile_grids: &grids,
            thumbnail_position: ThumbnailPosition::Append,
        },
        &config,
    )
    .unwrap();

    assert_eq!(expanded, [7, 7, 8, 7, 7, 7, 7]);
}

#[test]
fn inserts_column_and_row_separators_between_local_tiles() {
    let mut config = token_expansion_config();
    config.tokens_per_tile = 1;
    config.thumbnail_position = ThumbnailPosition::Prepend;
    config.column_separator_token_id = Some(8);
    config.row_separator_token_id = Some(9);
    let tiles_per_image = [5];
    let grids = [TileGrid {
        columns: 2,
        rows: 2,
    }];

    let expanded = expand_image_placeholders(
        &[99],
        ImageTilingSummary {
            num_tiles: 5,
            tiles_per_image: &tiles_per_image,
            tile_grids: &grids,
            thumbnail_position: ThumbnailPosition::Prepend,
        },
        &config,
    )
    .unwrap();

    assert_eq!(expanded, [7, 7, 8, 7, 9, 7, 8, 7]);
}

#[test]
fn matches_multiple_placeholders_to_images_in_prompt_order() {
    let mut config = token_expansion_config();
    config.tokens_per_tile = 1;
    let tiles_per_image = [2, 3];
    let grids = [
        TileGrid {
            columns: 2,
            rows: 1,
        },
        TileGrid {
            columns: 1,
            rows: 3,
        },
    ];

    let expanded = expand_image_placeholders(
        &[10, 99, 11, 99, 12],
        ImageTilingSummary {
            num_tiles: 5,
            tiles_per_image: &tiles_per_image,
            tile_grids: &grids,
            thumbnail_position: ThumbnailPosition::None,
        },
        &config,
    )
    .unwrap();

    assert_eq!(expanded, [10, 7, 7, 11, 7, 7, 7, 12]);
}

#[test]
fn rejects_inconsistent_token_expansion_inputs() {
    let grids = [TileGrid {
        columns: 2,
        rows: 1,
    }];

    let mut config = token_expansion_config();
    config.tokens_per_tile = 0;
    let error = expand_image_placeholders(
        &[99],
        ImageTilingSummary {
            num_tiles: 2,
            tiles_per_image: &[2],
            tile_grids: &grids,
            thumbnail_position: ThumbnailPosition::None,
        },
        &config,
    )
    .unwrap_err();
    assert!(error.to_string().contains("tokens_per_tile"));

    config.tokens_per_tile = 1;
    let error = expand_image_placeholders(
        &[99, 99],
        ImageTilingSummary {
            num_tiles: 2,
            tiles_per_image: &[2],
            tile_grids: &grids,
            thumbnail_position: ThumbnailPosition::None,
        },
        &config,
    )
    .unwrap_err();
    assert!(error.to_string().contains("2 image placeholder"));

    let error = expand_image_placeholders(
        &[99],
        ImageTilingSummary {
            num_tiles: 3,
            tiles_per_image: &[2],
            tile_grids: &grids,
            thumbnail_position: ThumbnailPosition::None,
        },
        &config,
    )
    .unwrap_err();
    assert!(error.to_string().contains("reports 3 total tile"));

    let error = expand_image_placeholders(
        &[99],
        ImageTilingSummary {
            num_tiles: 1,
            tiles_per_image: &[1],
            tile_grids: &grids,
            thumbnail_position: ThumbnailPosition::None,
        },
        &config,
    )
    .unwrap_err();
    assert!(error.to_string().contains("require 2"));
}

#[test]
fn bicubic_shortest_edge_resize_center_crops_to_target_dimensions() {
    let config = ImagePreprocessConfig {
        width: 4,
        height: 4,
        resize_mode: ResizeMode::ShortestEdgeCenterCrop,
        interpolation: Interpolation::Bicubic,
        tiling: ImageTilingConfig {
            mode: TilingMode::None,
            tile_size: 4,
            max_tiles: 1,
            aspect_ratios: vec![TileGrid {
                columns: 1,
                rows: 1,
            }],
            include_thumbnail: false,
            thumbnail_position: ThumbnailPosition::None,
        },
        normalization: Normalization::ZeroToOne,
    };
    let image = DynamicImage::ImageRgb8(RgbImage::from_fn(12, 6, |x, _| {
        if x < 6 {
            Rgb([255, 0, 0])
        } else {
            Rgb([0, 0, 255])
        }
    }));
    assert_eq!(
        resize_image(&image, &config, None).unwrap().dimensions(),
        (4, 4)
    );
}

#[test]
fn clip_mean_std_normalization_matches_known_pixel() {
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 1, Rgb([255, 128, 0])));
    let preprocessor = ImagePreprocessor {
        shape: vec![1, 3, 1, 1],
        layout: ImageLayout::Nchw,
        config: ImagePreprocessConfig {
            width: 1,
            height: 1,
            resize_mode: ResizeMode::Fixed,
            interpolation: Interpolation::Bicubic,
            tiling: ImageTilingConfig {
                mode: TilingMode::None,
                tile_size: 1,
                max_tiles: 1,
                aspect_ratios: vec![TileGrid {
                    columns: 1,
                    rows: 1,
                }],
                include_thumbnail: false,
                thumbnail_position: ThumbnailPosition::None,
            },
            normalization: Normalization::MeanStd {
                mean: [0.48145466, 0.4578275, 0.40821073],
                std: [0.26862954, 0.261_302_6, 0.275_777_1],
            },
        },
        program: ImageProgram {
            value_ops: vec![
                ValueOp::Rescale(1.0 / 255.0),
                ValueOp::Normalize {
                    mean: [0.48145466, 0.4578275, 0.40821073],
                    std: [0.26862954, 0.261_302_6, 0.275_777_1],
                },
            ],
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
        },
    };
    let tensor = preprocessor.preprocess(&[image]).unwrap();
    let expected = [
        (1.0 - 0.48145466) / 0.26862954,
        (128.0 / 255.0 - 0.4578275) / 0.261_302_6,
        (0.0 - 0.40821073) / 0.275_777_1,
    ];
    let pixels = tensor.tensor_by_content("pixels").unwrap();
    let actual = pixels.data.as_f32_slice().unwrap();
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn metadata_selects_target_resize_and_normalization() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/image_preprocessing.yaml");
    let preprocessor =
        ImagePreprocessor::from_input_and_metadata(&[-1, 3, -1, -1], Some(&path)).unwrap();
    assert_eq!(preprocessor.shape(), &[-1, 3, 2, 2]);
    assert_eq!(preprocessor.config().resize_mode, ResizeMode::Fixed);
    assert_eq!(preprocessor.config().interpolation, Interpolation::Bicubic);
    assert_eq!(
        preprocessor.config().normalization,
        Normalization::MeanStd {
            mean: [0.1, 0.2, 0.3],
            std: [0.4, 0.5, 0.6],
        }
    );
    assert_eq!(preprocessor.config().tiling.mode, TilingMode::None);
}

#[test]
fn metadata_selects_dynamic_anyres_tiling() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/image_tiling.yaml");
    let preprocessor =
        ImagePreprocessor::from_input_and_metadata(&[-1, 3, 2, 2], Some(&path)).unwrap();

    assert_eq!(
        preprocessor.config().tiling,
        ImageTilingConfig {
            mode: TilingMode::DynamicAnyres,
            tile_size: 2,
            max_tiles: 4,
            aspect_ratios: vec![
                TileGrid {
                    columns: 1,
                    rows: 1
                },
                TileGrid {
                    columns: 2,
                    rows: 1
                },
                TileGrid {
                    columns: 1,
                    rows: 2
                },
                TileGrid {
                    columns: 2,
                    rows: 2
                },
            ],
            include_thumbnail: true,
            thumbnail_position: ThumbnailPosition::Prepend,
        }
    );
}

#[test]
fn metadata_selects_fixed_grid_tiling() {
    let document = serde_yaml::from_str::<MetadataDocument>(
        r#"
preprocessing:
  image:
    resize:
      mode: fixed
      size: 2
      crop: none
    tiling:
      mode: fixed_grid
      tile_size: 2
      max_tiles: 6
      aspect_ratios: [[3, 2]]
"#,
    )
    .unwrap();
    let config = preprocessing_from_metadata(
        document
            .preprocessing
            .and_then(|preprocessing| preprocessing.image),
        2,
        2,
    )
    .unwrap();

    assert_eq!(config.tiling.mode, TilingMode::FixedGrid);
    assert_eq!(
        config.tiling.aspect_ratios,
        [TileGrid {
            columns: 3,
            rows: 2
        }]
    );
    assert!(config.tiling.include_thumbnail);
}

#[test]
fn missing_metadata_uses_bicubic_center_crop_and_zero_to_one() {
    let preprocessor = ImagePreprocessor::from_input(&[1, 3, 4, 4]).unwrap();
    assert_eq!(
        (preprocessor.config().width, preprocessor.config().height),
        (4, 4)
    );
    assert_eq!(
        preprocessor.config().resize_mode,
        ResizeMode::ShortestEdgeCenterCrop
    );
    assert_eq!(
        preprocessor.config().normalization,
        Normalization::ZeroToOne
    );
}

fn tiled_preprocessor(
    mode: TilingMode,
    grids: Vec<TileGrid>,
    max_tiles: usize,
) -> ImagePreprocessor {
    ImagePreprocessor {
        shape: vec![-1, 3, 2, 2],
        layout: ImageLayout::Nchw,
        config: ImagePreprocessConfig {
            width: 2,
            height: 2,
            resize_mode: ResizeMode::Fixed,
            interpolation: Interpolation::Bicubic,
            tiling: ImageTilingConfig {
                mode,
                tile_size: 2,
                max_tiles,
                aspect_ratios: grids,
                include_thumbnail: true,
                thumbnail_position: ThumbnailPosition::Prepend,
            },
            normalization: Normalization::ZeroToOne,
        },
        program: ImageProgram {
            value_ops: vec![ValueOp::Rescale(1.0 / 255.0)],
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
        },
    }
}

#[test]
fn none_tiling_preserves_one_output_per_image() {
    let preprocessor = ImagePreprocessor::from_input(&[-1, 3, 2, 2]).unwrap();
    let images = [
        DynamicImage::ImageRgb8(RgbImage::from_pixel(3, 2, Rgb([255, 0, 0]))),
        DynamicImage::ImageRgb8(RgbImage::from_pixel(2, 3, Rgb([0, 0, 255]))),
    ];
    let tensor = preprocessor.preprocess(&images).unwrap();
    let pixels = tensor.tensor_by_content("pixels").unwrap();

    assert_eq!(pixels.shape, [2, 3, 2, 2]);
    assert_eq!(tensor.num_tiles, 2);
    assert_eq!(tensor.tiles_per_image, [1, 1]);
    assert_eq!(
        tensor.tile_grids,
        [
            TileGrid {
                columns: 1,
                rows: 1
            },
            TileGrid {
                columns: 1,
                rows: 1
            }
        ]
    );
    assert_eq!(
        tensor
            .images
            .iter()
            .map(|image| image.original_size)
            .collect::<Vec<_>>(),
        [(3, 2), (2, 3)]
    );
    assert_eq!(pixels.data.len(), 2 * 3 * 2 * 2);
}

#[test]
fn fixed_grid_produces_grid_tiles_and_global_thumbnail() {
    let preprocessor = tiled_preprocessor(
        TilingMode::FixedGrid,
        vec![TileGrid {
            columns: 3,
            rows: 2,
        }],
        6,
    );
    let image = DynamicImage::ImageRgb8(RgbImage::from_fn(6, 4, |x, y| {
        Rgb([(x * 20) as u8, (y * 30) as u8, 0])
    }));
    let tensor = preprocessor.preprocess(&[image]).unwrap();
    let pixels = tensor.tensor_by_content("pixels").unwrap();

    assert_eq!(pixels.shape, [7, 3, 2, 2]);
    assert_eq!(tensor.num_tiles, 7);
    assert_eq!(tensor.tiles_per_image, [7]);
    assert_eq!(
        tensor.tile_grids,
        [TileGrid {
            columns: 3,
            rows: 2
        }]
    );
    assert_eq!(pixels.data.len(), 7 * 3 * 2 * 2);
    assert_eq!(tensor.tile_data(0).unwrap().len(), 3 * 2 * 2);
    assert_eq!(tensor.tile_data(6).unwrap().len(), 3 * 2 * 2);
    assert!(tensor.tile_data(7).is_none());
}

#[test]
fn dynamic_anyres_selects_expected_representative_grids() {
    let grids = default_anyres_grids();
    assert_eq!(
        select_best_grid(1200, 400, 336, 6, &grids).unwrap(),
        TileGrid {
            columns: 3,
            rows: 1
        }
    );
    assert_eq!(
        select_best_grid(400, 1200, 336, 6, &grids).unwrap(),
        TileGrid {
            columns: 1,
            rows: 3
        }
    );
    assert_eq!(
        select_best_grid(800, 800, 336, 6, &grids).unwrap(),
        TileGrid {
            columns: 2,
            rows: 2
        }
    );
}

#[test]
fn dynamic_anyres_respects_max_tiles_and_adds_thumbnail() {
    let preprocessor = tiled_preprocessor(
        TilingMode::DynamicAnyres,
        vec![
            TileGrid {
                columns: 3,
                rows: 2,
            },
            TileGrid {
                columns: 2,
                rows: 2,
            },
            TileGrid {
                columns: 2,
                rows: 1,
            },
        ],
        4,
    );
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(800, 800, Rgb([64, 128, 255])));
    let tensor = preprocessor.preprocess(&[image]).unwrap();
    let pixels = tensor.tensor_by_content("pixels").unwrap();

    assert_eq!(pixels.shape, [5, 3, 2, 2]);
    assert_eq!(tensor.num_tiles, 5);
    assert_eq!(tensor.tiles_per_image, [5]);
    assert_eq!(
        tensor.tile_grids,
        [TileGrid {
            columns: 2,
            rows: 2
        }]
    );
}

// --- Tests specifically for thumbnail position / tile ordering alignment ---

/// Regression test for the bug reported by Gaff: when the preprocessor
/// includes a thumbnail it is always placed FIRST in the tensor
/// (`ThumbnailPosition::Prepend`).  `tiling_summary()` must report this so
/// that callers can drive token expansion with the correct ordering.
#[test]
fn tiling_summary_reports_prepend_thumbnail_position_matching_tensor_layout() {
    let preprocessor = tiled_preprocessor(
        TilingMode::FixedGrid,
        vec![TileGrid {
            columns: 2,
            rows: 1,
        }],
        2,
    );
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 2, Rgb([100, 150, 200])));
    let tensor = preprocessor.preprocess(&[image]).unwrap();

    // The pipeline stores thumbnail first (index 0) then local tiles.
    assert_eq!(
        tensor.thumbnail_position,
        ThumbnailPosition::Prepend,
        "tensor thumbnail_position must be Prepend to match tiled_image_for_grid layout"
    );
    assert_eq!(
        tensor.tiling_summary().thumbnail_position,
        ThumbnailPosition::Prepend,
    );
    // tiles_per_image = [thumbnail + 2 local] = 3
    assert_eq!(tensor.tiles_per_image, [3]);
}

/// Token order must match tile order when thumbnail is first in the tensor.
///
/// With tokens_per_tile=1 and a 2×1 grid + prepended thumbnail the expected
/// token sequence is [thumbnail, local(0,0), local(0,1)].  Previously this
/// would be silently wrong if a caller accidentally used `Append` in config.
#[test]
fn prepend_thumbnail_token_order_matches_tensor_tile_order() {
    let mut config = token_expansion_config();
    config.tokens_per_tile = 1;
    config.thumbnail_position = ThumbnailPosition::Prepend;
    config.column_separator_token_id = Some(8);
    let tiles_per_image = [3];
    let grids = [TileGrid {
        columns: 2,
        rows: 1,
    }];
    // tiling.thumbnail_position=Prepend matches actual tensor layout.
    let expanded = expand_image_placeholders(
        &[99],
        ImageTilingSummary {
            num_tiles: 3,
            tiles_per_image: &tiles_per_image,
            tile_grids: &grids,
            thumbnail_position: ThumbnailPosition::Prepend,
        },
        &config,
    )
    .unwrap();
    // Expected: thumbnail first, then local tile 0, col_sep, local tile 1.
    assert_eq!(expanded, [7, 7, 8, 7]);
}

/// Token expansion must reject a config whose thumbnail_position contradicts
/// the tensor layout reported by the tiling summary.  This is the exact
/// failure mode described in Gaff's review: tensor has thumbnail FIRST but
/// config says LAST, silently producing misaligned embeddings.
#[test]
fn mismatched_thumbnail_position_config_vs_tiling_is_rejected() {
    let mut config = token_expansion_config();
    config.thumbnail_position = ThumbnailPosition::Append; // wrong for a Prepend tensor
    let tiles_per_image = [3];
    let grids = [TileGrid {
        columns: 2,
        rows: 1,
    }];
    let error = expand_image_placeholders(
        &[99],
        ImageTilingSummary {
            num_tiles: 3,
            tiles_per_image: &tiles_per_image,
            tile_grids: &grids,
            thumbnail_position: ThumbnailPosition::Prepend, // actual tensor layout
        },
        &config,
    )
    .unwrap_err();
    let msg = error.to_string();
    assert!(
        msg.contains("thumbnail_position"),
        "error should mention thumbnail_position mismatch, got: {msg}"
    );
}

/// Verify that token expansion driven by the real ImageTensor tiling summary
/// (thumbnail_position=Prepend) produces token order [thumbnail, local…],
/// which aligns with how tiled_image_for_grid lays out pixels in the tensor.
#[test]
fn token_expansion_from_real_tensor_summary_matches_tile_layout() {
    let preprocessor = tiled_preprocessor(
        TilingMode::FixedGrid,
        vec![TileGrid {
            columns: 2,
            rows: 1,
        }],
        2,
    );
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 2, Rgb([10, 20, 30])));
    let tensor = preprocessor.preprocess(&[image]).unwrap();
    // 3 tiles: thumbnail (index 0), local (index 1), local (index 2).
    assert_eq!(tensor.num_tiles, 3);
    assert_eq!(tensor.thumbnail_position, ThumbnailPosition::Prepend);

    let summary = tensor.tiling_summary();
    let mut config = token_expansion_config();
    config.tokens_per_tile = 1;
    // Config must match the tensor layout reported by tiling_summary.
    config.thumbnail_position = summary.thumbnail_position;

    let expanded = expand_image_placeholders(&[99], summary, &config).unwrap();
    // 3 tokens total: first corresponds to thumbnail (tensor index 0),
    // then the two local tiles in row-major order.
    assert_eq!(expanded.len(), 3);
    assert_eq!(expanded, [7, 7, 7]);
}

fn typed_preprocessor(shape: &[i64], image_yaml: &str) -> ImagePreprocessor {
    let document = serde_yaml::from_str::<MetadataDocument>(image_yaml).unwrap();
    ImagePreprocessor::from_metadata_document(shape, Some(document)).unwrap()
}

fn packed_test_images() -> [DynamicImage; 2] {
    [
        DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 2, Rgb([255, 0, 0]))),
        DynamicImage::ImageRgb8(RgbImage::from_pixel(2, 2, Rgb([0, 0, 255]))),
    ]
}

#[test]
fn named_operation_descriptors_select_declared_output_sources() {
    const PROGRAM: &str = r#"
preprocessing:
  image:
    transforms:
      - op: decode
        outputs: [decoded]
      - op: convert_rgb
        inputs: [decoded]
        outputs: [rgb]
      - op: resize
        inputs: [rgb]
        outputs: [resized]
        size: 4
        mode: stretch
      - op: rescale
        inputs: [resized]
        outputs: [scaled]
        scale: 0.00392156862745098
      - op: patchify
        inputs: [scaled]
        outputs: [patches]
        patch_size: 2
        flatten: false
      - op: flatten
        inputs: [patches]
        outputs: [flat_patches]
      - op: emit_patch_coordinates
        inputs: [patches]
        outputs: [coordinates]
      - op: emit_grid_coordinates
        inputs: [patches]
        outputs: [grid]
      - op: emit_original_size
        inputs: [rgb]
        outputs: [original_size]
      - op: pad
        inputs: [flat_patches, coordinates]
        outputs: [padded_patches, padded_coordinates]
        target_length: 5
        pad_value: 0
    outputs:
      - source: padded_patches
        name: arbitrary_pixels
        content: pixels
        dtype: fp32
      - source: padded_coordinates
        name: arbitrary_coordinates
        content: patch_coordinates
        dtype: int64
        pad_value: -1
      - source: grid
        name: arbitrary_grid
        content: grid_dimensions
        dtype: int32
      - source: original_size
        name: arbitrary_size
        content: original_size
        dtype: int64
"#;
    let preprocessor = typed_preprocessor(&[1, 5, 12], PROGRAM);
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(3, 2, Rgb([255, 0, 0])));
    let bundle = preprocessor.preprocess(&[image]).unwrap();

    assert_eq!(bundle.tensor("arbitrary_pixels").unwrap().shape, [1, 5, 12]);
    assert_eq!(
        bundle.tensor("arbitrary_coordinates").unwrap().shape,
        [1, 5, 2]
    );
    assert_eq!(
        bundle.tensor("arbitrary_grid").unwrap().data,
        ImageTensorData::Int32(vec![1, 2, 2])
    );
    assert_eq!(
        bundle.tensor("arbitrary_size").unwrap().data,
        ImageTensorData::Int64(vec![2, 3])
    );
}

#[test]
fn named_rescale_branches_execute_independently() {
    const PROGRAM: &str = r#"
preprocessing:
  image:
    transforms:
      - op: decode_rgb
        outputs: [decoded]
      - op: rescale
        inputs: [decoded]
        outputs: [half]
        scale: 0.5
      - op: rescale
        inputs: [decoded]
        outputs: [quarter]
        scale: 0.25
    outputs:
      - source: half
        name: half_pixels
        content: pixels
        dtype: fp32
      - source: quarter
        name: quarter_pixels
        content: pixels
        dtype: fp32
"#;
    let preprocessor = typed_preprocessor(&[1, 3, 1, 1], PROGRAM);
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 1, Rgb([200, 100, 40])));
    let bundle = preprocessor.preprocess(&[image]).unwrap();

    assert_eq!(
        bundle
            .tensor("half_pixels")
            .unwrap()
            .data
            .as_f32_slice()
            .unwrap(),
        [100.0, 50.0, 20.0]
    );
    assert_eq!(
        bundle
            .tensor("quarter_pixels")
            .unwrap()
            .data
            .as_f32_slice()
            .unwrap(),
        [50.0, 25.0, 10.0]
    );
}

#[test]
fn named_structural_branches_are_rejected_instead_of_misexecuted() {
    const PROGRAM: &str = r#"
preprocessing:
  image:
    transforms:
      - op: decode_rgb
        outputs: [decoded]
      - op: resize
        inputs: [decoded]
        outputs: [resized]
        size: 2
      - op: rescale
        inputs: [decoded]
        outputs: [unresized]
        scale: 0.5
    outputs:
      - source: unresized
        name: pixels
        content: pixels
        dtype: fp32
"#;
    let document = serde_yaml::from_str::<MetadataDocument>(PROGRAM).unwrap();
    let error =
        ImagePreprocessor::from_metadata_document(&[1, 3, 2, 2], Some(document)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("structural branch that the current packer cannot execute independently")
    );
}

#[test]
fn named_operation_descriptors_reject_unknown_output_source() {
    const PROGRAM: &str = r#"
preprocessing:
  image:
    transforms:
      - op: decode
        outputs: [decoded]
    outputs:
      - source: missing
        name: pixels
        content: pixels
        dtype: fp32
"#;
    let document = serde_yaml::from_str::<MetadataDocument>(PROGRAM).unwrap();
    let error =
        ImagePreprocessor::from_metadata_document(&[1, 3, 2, 2], Some(document)).unwrap_err();
    assert_eq!(
        error.to_string(),
        "image output 'pixels' selects unknown source 'missing'; choose a declared transform output"
    );
}

#[test]
fn checked_in_wp0_named_program_executes_without_identity_dispatch() {
    let document = serde_yaml::from_str::<MetadataDocument>(include_str!(
        "../../../onnx-genai-metadata/tests/fixtures/vlm_packed_valid.yaml"
    ))
    .unwrap();
    let preprocessor =
        ImagePreprocessor::from_metadata_document(&[1, 4096, 588], Some(document)).unwrap();
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(32, 16, Rgb([64, 128, 255])));
    let bundle = preprocessor.preprocess(&[image]).unwrap();

    assert_eq!(
        bundle.tensor("vision_encoder.pixel_values").unwrap().shape,
        [1, 4096, 588]
    );
    assert_eq!(
        bundle
            .tensor("vision_encoder.pixel_position_ids")
            .unwrap()
            .shape,
        [1, 4096, 2]
    );
    assert_eq!(bundle.images.len(), 1);
    assert!(bundle.images[0].expansion_count > 0);
    assert_eq!(bundle.images[0].tensor_length, 4096);
}

const PADDED_PROGRAM: &str = r#"
preprocessing:
  image:
    transforms:
      - op: decode_rgb
      - op: resize
        size: 2
        mode: stretch
        interpolation: bilinear
      - op: tile
        tile_size: 2
        max_tiles: 2
      - op: rescale
        scale: 0.00392156862745098
      - op: patchify
        patch_size: 1
        flatten: true
      - op: pad
        pad_value: 0
        target_length: 8
    outputs:
      - name: image_pixels
        content: pixels
        dtype: fp32
      - name: image_coordinates
        content: patch_coordinates
        dtype: int64
        pad_value: -1
"#;

// Small checked-in vectors generated once from equivalent HF processor
// operations (RGB conversion, resize, rescale, CHW patchify, and padding).
const HF_PADDED_PIXELS: [f32; 48] = [
    1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];
const HF_PADDED_COORDINATES: [i64; 32] = [
    0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1, 3, 0, 3, 1, 0, 0, 0, 1, 1, 0, 1, 1, -1, -1, -1, -1, -1, -1,
    -1, -1,
];

#[test]
fn gemma4_shaped_padded_patches_and_sentinel_coordinates_match_fixture() {
    let preprocessor = typed_preprocessor(&[2, 8, 3], PADDED_PROGRAM);
    let bundle = preprocessor.preprocess(&packed_test_images()).unwrap();
    let pixels = bundle.tensor("image_pixels").unwrap();
    let coordinates = bundle.tensor("image_coordinates").unwrap();

    assert_eq!(pixels.shape, [2, 8, 3]);
    assert_eq!(
        pixels.data.as_f32_slice().unwrap(),
        HF_PADDED_PIXELS.as_slice()
    );
    assert_eq!(coordinates.shape, [2, 8, 2]);
    assert_eq!(
        coordinates.data,
        ImageTensorData::Int64(HF_PADDED_COORDINATES.to_vec())
    );
    assert_eq!(
        bundle
            .images
            .iter()
            .map(|summary| (
                summary.image_index,
                summary.expansion_count,
                summary.tensor_offset,
                summary.tensor_length,
            ))
            .collect::<Vec<_>>(),
        [(0, 8, 0, 8), (1, 4, 8, 8)]
    );
}

#[test]
fn qwen_shaped_concatenated_patches_emit_per_image_grid() {
    const PROGRAM: &str = r#"
preprocessing:
  image:
    transforms:
      - op: decode_rgb
      - op: resize
        size: 4
        mode: stretch
        interpolation: bilinear
      - op: rescale
        scale: 0.00392156862745098
      - op: patchify
        patch_size: 2
        flatten: true
    outputs:
      - name: image_pixels
        content: pixels
        dtype: fp32
      - name: image_grid
        content: grid_dimensions
        dtype: int64
"#;
    let images = [
        DynamicImage::ImageRgb8(
            RgbImage::from_raw(4, 4, hf_reference::QWEN_IMAGE_0.to_vec()).unwrap(),
        ),
        DynamicImage::ImageRgb8(
            RgbImage::from_raw(4, 4, hf_reference::QWEN_IMAGE_1.to_vec()).unwrap(),
        ),
    ];
    let preprocessor = typed_preprocessor(&[8, 12], PROGRAM);
    let bundle = preprocessor.preprocess(&images).unwrap();
    let pixels = bundle.tensor("image_pixels").unwrap();
    let grid = bundle.tensor("image_grid").unwrap();

    assert_eq!(pixels.shape, [8, 12]);
    assert_eq!(
        pixels
            .data
            .as_f32_slice()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        hf_reference::QWEN_PIXEL_BITS
    );
    assert_eq!(grid.shape, [2, 3]);
    assert_eq!(
        grid.data,
        ImageTensorData::Int64(hf_reference::QWEN_GRID.to_vec())
    );
    assert_eq!(
        bundle
            .images
            .iter()
            .map(|summary| summary.tensor_offset)
            .collect::<Vec<_>>(),
        [0, 4]
    );
}

#[test]
fn phi_shaped_outputs_include_original_sizes_and_patch_validity() {
    const PROGRAM: &str = r#"
preprocessing:
  image:
    transforms:
      - op: decode_rgb
      - op: resize
        size: 4
        mode: stretch
        interpolation: bilinear
      - op: tile
        tile_size: 4
        max_tiles: 2
      - op: rescale
        scale: 0.00392156862745098
      - op: normalize
        mean: [0.48145466, 0.4578275, 0.40821073]
        std: [0.26862954, 0.26130258, 0.27577711]
      - op: patchify
        patch_size: 2
        flatten: true
      - op: pad
        pad_value: 0
        target_length: 8
    outputs:
      - name: image_pixels
        content: pixels
        dtype: bf16
      - name: image_pixels_fp16
        content: pixels
        dtype: fp16
      - name: image_sizes
        content: original_size
        dtype: int64
      - name: patch_mask
        content: validity_mask
        dtype: bool
"#;
    let images = [
        DynamicImage::ImageRgb8(
            RgbImage::from_raw(8, 4, hf_reference::PHI_IMAGE_0.to_vec()).unwrap(),
        ),
        DynamicImage::ImageRgb8(
            RgbImage::from_raw(4, 4, hf_reference::PHI_IMAGE_1.to_vec()).unwrap(),
        ),
    ];
    let preprocessor = typed_preprocessor(&[2, 8, 12], PROGRAM);
    let bundle = preprocessor.preprocess(&images).unwrap();

    let pixels = bundle.tensor("image_pixels").unwrap();
    assert_eq!(pixels.dtype, ImageTensorDType::Bf16);
    assert_eq!(
        pixels.data,
        ImageTensorData::Bf16(hf_reference::PHI_BF16_BITS.to_vec())
    );
    let fp16_pixels = bundle.tensor("image_pixels_fp16").unwrap();
    assert_eq!(fp16_pixels.dtype, ImageTensorDType::Fp16);
    assert_eq!(
        fp16_pixels.data,
        ImageTensorData::Fp16(hf_reference::PHI_FP16_BITS.to_vec())
    );
    assert_eq!(
        bundle.tensor("image_sizes").unwrap().data,
        ImageTensorData::Int64(hf_reference::PHI_SIZES.to_vec())
    );
    assert_eq!(
        bundle.tensor("patch_mask").unwrap().data,
        ImageTensorData::Bool(hf_reference::PHI_MASK.to_vec())
    );
}

#[test]
fn qwen_area_resize_and_temporal_patch_packing_are_executable() {
    const PROGRAM: &str = r#"
preprocessing:
  image:
    transforms:
      - op: decode_rgb
      - op: resize
        mode: pixel_area
        min_pixels: 65536
        max_pixels: 16777216
        size_multiple: 32
        interpolation: bicubic
      - op: rescale
        scale: 0.00392156862745098
      - op: normalize
        mean: [0.5, 0.5, 0.5]
        std: [0.5, 0.5, 0.5]
      - op: patchify
        patch_size: 16
        temporal_patch_size: 2
        merge_size: 2
        channel_order: channels_first
        flatten: true
    outputs:
      - name: pixel_values
        content: pixels
        dtype: fp32
      - name: image_grid_thw
        content: grid_dimensions
        dtype: int64
"#;
    let preprocessor = typed_preprocessor(&[-1, 1536], PROGRAM);
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(500, 300, Rgb([0, 0, 0])));
    let bundle = preprocessor.preprocess(&[image]).unwrap();

    assert_eq!(bundle.tensor("pixel_values").unwrap().shape, [576, 1536]);
    assert_eq!(
        bundle.tensor("image_grid_thw").unwrap().data,
        ImageTensorData::Int64(vec![1, 18, 32])
    );
}

#[test]
fn pixel_area_rounding_matches_python_ties_to_even() {
    assert_eq!(round_to_multiple_ties_even(16, 32), 0);
    assert_eq!(round_to_multiple_ties_even(48, 32), 64);
}

#[test]
fn gemma_patch_budget_resize_pads_to_declared_2520_patches() {
    const PROGRAM: &str = r#"
preprocessing:
  image:
    transforms:
      - op: decode_rgb
      - op: resize
        mode: aspect_ratio_patch_budget
        patch_size: 16
        max_patches: 2520
        pooling_kernel_size: 3
        interpolation: bicubic
      - op: rescale
        scale: 0.00392156862745098
      - op: patchify
        patch_size: 16
        channel_order: channels_last
        coordinate_order: xy
        flatten: true
      - op: pad
        pad_value: 0
        target_length: 2520
    outputs:
      - name: pixel_values
        content: pixels
        dtype: fp32
      - name: pixel_position_ids
        content: patch_coordinates
        dtype: int64
        pad_value: -1
"#;
    let preprocessor = typed_preprocessor(&[-1, 2520, 768], PROGRAM);
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(500, 300, Rgb([0, 0, 0])));
    let bundle = preprocessor.preprocess(&[image]).unwrap();

    assert_eq!(bundle.tensor("pixel_values").unwrap().shape, [1, 2520, 768]);
    assert_eq!(
        bundle.tensor("pixel_position_ids").unwrap().shape,
        [1, 2520, 2]
    );
    assert_eq!(bundle.images[0].expansion_count, 2268);
    assert_eq!(bundle.images[0].tensor_length, 2520);
}

#[test]
fn dynamic_hd_emits_transformed_size_and_patch_validity_masks() {
    const PROGRAM: &str = r#"
preprocessing:
  image:
    transforms:
      - op: decode_rgb
      - op: tile
        mode: dynamic_hd
        tile_size: 448
        max_tiles: 36
        include_thumbnail: true
        thumbnail_order: prepend
        interpolation: bilinear
        thumbnail_interpolation: bicubic
        canvas_pad_value: 255
        mask_patch_size: 14
      - op: rescale
        scale: 0.00392156862745098
      - op: normalize
        mean: [0.5, 0.5, 0.5]
        std: [0.5, 0.5, 0.5]
    outputs:
      - name: pixel_values
        content: pixels
        dtype: fp32
      - name: image_sizes
        content: transformed_size
        dtype: int64
      - name: image_attention_mask
        content: validity_mask
        dtype: fp32
        pad_value: 0
"#;
    let preprocessor = typed_preprocessor(&[-1, 3, 448, 448], PROGRAM);
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(500, 300, Rgb([0, 0, 0])));
    let bundle = preprocessor.preprocess(&[image]).unwrap();

    assert_eq!(
        bundle.tensor("pixel_values").unwrap().shape,
        [3, 3, 448, 448]
    );
    assert_eq!(
        bundle.tensor("image_sizes").unwrap().data,
        ImageTensorData::Int64(vec![448, 896])
    );
    assert_eq!(
        bundle.tensor("image_attention_mask").unwrap().shape,
        [3, 32, 32]
    );
    let mask = bundle
        .tensor("image_attention_mask")
        .unwrap()
        .data
        .as_f32_slice()
        .unwrap();
    assert!(mask.contains(&0.0));
    assert!(mask.contains(&1.0));
}

#[test]
fn rank4_nchw_values_remain_unchanged_in_bundle() {
    let preprocessor = ImagePreprocessor::from_input(&[1, 3, 1, 2]).unwrap();
    let image =
        DynamicImage::ImageRgb8(RgbImage::from_raw(2, 1, vec![255, 0, 128, 0, 64, 255]).unwrap());
    let bundle = preprocessor.preprocess(&[image]).unwrap();
    let pixels = bundle.tensor("pixels").unwrap();

    assert_eq!(pixels.shape, [1, 3, 1, 2]);
    assert_eq!(
        pixels.data.as_f32_slice().unwrap(),
        [1.0, 0.0, 0.0, 64.0 / 255.0, 128.0 / 255.0, 1.0]
    );
}

#[test]
fn legacy_zero_to_one_is_bit_exact_for_every_u8() {
    let preprocessor = ImagePreprocessor::from_input(&[1, 3, 1, 256]).unwrap();
    let image = RgbImage::from_fn(256, 1, |x, _| {
        let value = x as u8;
        Rgb([value, value, value])
    });
    let values = normalize_tile(&image, 256, 1, &preprocessor.program.value_ops).unwrap();

    for channel in 0..CHANNELS {
        for value in 0u8..=u8::MAX {
            let actual = values[channel * 256 + usize::from(value)].to_bits();
            let expected = (f32::from(value) / 255.0).to_bits();
            assert_eq!(
                actual, expected,
                "legacy normalization changed byte {value}: actual {actual:#010x}, expected {expected:#010x}"
            );
        }
    }
}

#[test]
fn rejects_degenerate_source_images_before_resize() {
    let preprocessor = ImagePreprocessor::from_input(&[1, 3, 2, 2]).unwrap();
    let image = DynamicImage::ImageRgb8(RgbImage::new(0, 2));
    let error = preprocessor.preprocess(&[image]).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("degenerate dimensions 0x2; provide an image with nonzero width and height")
    );
}

#[test]
fn rejects_oversized_center_crop_intermediates_before_allocation() {
    let preprocessor = ImagePreprocessor::from_input(&[1, 3, 4_096, 4_096]).unwrap();
    let image = DynamicImage::ImageRgb8(RgbImage::new(16_384, 1));
    let error = preprocessor.preprocess(&[image]).unwrap_err();

    assert!(error.to_string().contains("center-crop intermediate image"));
    assert!(error.to_string().contains("exceeding the safety limit"));
}

#[test]
fn rejects_metadata_dimensions_above_the_pixel_limit() {
    let yaml = format!(
        r#"
preprocessing:
  image:
    transforms:
      - op: decode_rgb
      - op: resize
        size: {{width: {}, height: 1}}
      - op: patchify
        patch_size: 1
    outputs:
      - name: pixels
        content: pixels
        dtype: fp32
"#,
        MAX_IMAGE_PIXELS + 1
    );
    let document = serde_yaml::from_str::<MetadataDocument>(&yaml).unwrap();
    let error = ImagePreprocessor::from_metadata_document(&[-1, 3], Some(document)).unwrap_err();

    assert!(error.to_string().contains("exceeding the safety limit"));
}
