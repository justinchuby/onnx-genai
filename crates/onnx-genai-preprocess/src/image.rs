//! Metadata-driven RGB image preprocessing.

mod config;
pub mod packed;

pub use config::{
    ImageLayout, ImagePreprocessConfig, ImageTilingConfig, ImageTilingSummary, Interpolation,
    Normalization, ResizeMode, ThumbnailPosition, TileGrid, TilingMode, TokenExpansionConfig,
    expand_image_placeholders,
};
pub use packed::{
    ImageExpansionSummary, ImageTensorBundle, ImageTensorDType, ImageTensorData, NamedImageTensor,
};
const CHANNELS: usize = 3;
pub(super) const MAX_IMAGE_COUNT: usize = 1_024;
pub(super) const MAX_IMAGE_PIXELS: usize = 16 * 1024 * 1024;
pub(super) const MAX_TENSOR_ELEMENTS: usize = 64 * 1024 * 1024;
const MAX_IMAGE_OUTPUTS: usize = 64;
const MAX_IMAGE_TRANSFORMS: usize = 64;
const MAX_TILES_PER_IMAGE: usize = 4_096;
const MAX_ASPECT_RATIOS: usize = 4_096;

mod tiling;

mod transform;

mod program;
pub use program::ImagePreprocessor;
use program::{CoordinateOrder, PatchChannelOrder, PatchifySpec};
