use std::collections::{BTreeMap, BTreeSet};

use onnx_genai_metadata::{
    BatchLayout, PACK_OFFSETS_CONTENT, PACK_OWNER_CONTENT, TensorDimension, VisionOutputBinding,
    VisionPreprocessingProgram,
};

use crate::batching::{EncoderBatchingError, PackedOwnership, RequestOwnership, RequestSpan};

use super::{
    ImagePreprocessor, ImageTensorBundle, ImageTensorDType, ImageTensorData, NamedImageTensor,
    is_grouping_output_content,
};

/// One logical media item borrowing its encoded physical parts.
///
/// A one-level image item has exactly one part. A two-level clip item has one
/// part per frame. The encoded payload bytes remain owned by the caller.
#[derive(Debug, Clone)]
pub struct MediaItem<'a> {
    pub parts: Vec<&'a [u8]>,
}

impl<'a> MediaItem<'a> {
    pub fn single(encoded: &'a [u8]) -> Self {
        Self {
            parts: vec![encoded],
        }
    }

    pub fn nested(parts: impl IntoIterator<Item = &'a [u8]>) -> Self {
        Self {
            parts: parts.into_iter().collect(),
        }
    }
}

/// Ordered media items contributed by one request row.
#[derive(Debug, Clone, Default)]
pub struct MediaRequest<'a> {
    pub items: Vec<MediaItem<'a>>,
}

impl<'a> MediaRequest<'a> {
    pub fn new(items: impl IntoIterator<Item = MediaItem<'a>>) -> Self {
        Self {
            items: items.into_iter().collect(),
        }
    }
}

/// Tensor outputs plus the ownership chain used to split them back to requests.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupedVisionBundle {
    pub tensors: ImageTensorBundle,
    pub ownership: Option<PackedOwnership>,
}

impl GroupedVisionBundle {
    pub fn request_spans(&self) -> Vec<RequestSpan> {
        self.ownership
            .as_ref()
            .map_or_else(Vec::new, PackedOwnership::request_spans)
    }

    pub fn request_local(
        &self,
        request_index: usize,
    ) -> Result<RequestOwnership, EncoderBatchingError> {
        self.ownership
            .as_ref()
            .ok_or_else(|| EncoderBatchingError::UnsupportedExecution {
                component: "preprocessing".to_owned(),
                detail: "the authored outputs declare no token_packed ownership chain".to_owned(),
            })?
            .request_local(request_index)
    }
}

/// Executes spatial preprocessing over flattened physical parts and emits every
/// declared padding/ownership companion.
#[derive(Debug, Clone)]
pub struct GroupedVisionPreprocessor {
    component: String,
    processor: ImagePreprocessor,
    outputs: Vec<VisionOutputBinding>,
    ownership_pairs: Vec<(String, String)>,
    pad_target_length: Option<usize>,
}

impl GroupedVisionPreprocessor {
    pub fn from_input_and_program(
        component: impl Into<String>,
        shape: &[i64],
        program: &VisionPreprocessingProgram,
    ) -> Result<Self, EncoderBatchingError> {
        let component = component.into();
        if let Some(transform) = program
            .transforms
            .iter()
            .find(|transform| matches!(transform.op.as_str(), "sample_frames" | "pad_frames"))
        {
            return Err(EncoderBatchingError::UnsupportedExecution {
                component,
                detail: format!(
                    "temporal transform '{}' requires encoded-container decode/sample support; \
                     the executable grouped path accepts already ordered encoded frames and never \
                     skips a declared temporal transform",
                    transform.op
                ),
            });
        }
        let ownership_pairs = declared_ownership_pairs(&component, &program.outputs)?;
        let pad_target_length = program
            .transforms
            .iter()
            .rev()
            .find(|transform| transform.op == "pad")
            .and_then(|transform| transform.target_length);
        let processor =
            ImagePreprocessor::from_input_and_grouped_program(shape, program).map_err(|error| {
                EncoderBatchingError::Preprocessing {
                    detail: format!("{error:#}"),
                }
            })?;
        Ok(Self {
            component,
            processor,
            outputs: program.outputs.clone(),
            ownership_pairs,
            pad_target_length,
        })
    }

    /// Preprocesses request rows without copying their encoded payload buffers.
    ///
    /// The authored output layout selects the topology: no packed level accepts
    /// one item per row, one level accepts items in rows, and two levels accepts
    /// physical parts in items in rows.
    pub fn preprocess_encoded(
        &self,
        requests: &[MediaRequest<'_>],
    ) -> Result<GroupedVisionBundle, EncoderBatchingError> {
        let level_count = self.ownership_pairs.len();
        let mut flattened = Vec::<&[u8]>::new();
        let mut one_level_counts = Vec::with_capacity(requests.len());
        let mut nested_counts = Vec::with_capacity(requests.len());
        for (request_index, request) in requests.iter().enumerate() {
            one_level_counts.push(request.items.len());
            let mut request_nested = Vec::with_capacity(request.items.len());
            if level_count == 0 && request.items.len() != 1 {
                return Err(EncoderBatchingError::UnsupportedExecution {
                    component: self.component.clone(),
                    detail: format!(
                        "request {request_index} contributes {} media items, but the authored \
                         outputs declare no token_packed ownership level; zero or multiple items \
                         cannot be represented without fabricating request boundaries",
                        request.items.len()
                    ),
                });
            }
            for (item_index, item) in request.items.iter().enumerate() {
                match level_count {
                    0 | 1 if item.parts.len() != 1 => {
                        return Err(EncoderBatchingError::UnsupportedExecution {
                            component: self.component.clone(),
                            detail: format!(
                                "request {request_index} item {item_index} contains {} physical \
                                 parts, but the authored output contract declares {level_count} \
                                 ownership level(s); a multi-part item requires two levels",
                                item.parts.len()
                            ),
                        });
                    }
                    0 | 1 => {}
                    2 => request_nested.push(item.parts.len()),
                    _ => unreachable!("metadata limits ownership to two levels"),
                }
                flattened.extend(item.parts.iter().copied());
            }
            nested_counts.push(request_nested);
        }
        let ownership = match level_count {
            0 => None,
            1 => Some(PackedOwnership::one_level(&one_level_counts)?),
            2 => Some(PackedOwnership::two_levels(&nested_counts)?),
            _ => unreachable!("metadata limits ownership to two levels"),
        };
        let bundle = if flattened.is_empty() {
            self.empty_bundle(requests.len())?
        } else {
            self.processor
                .preprocess_encoded(flattened)
                .map_err(|error| EncoderBatchingError::Preprocessing {
                    detail: format!("{error:#}"),
                })?
        };
        self.finish_bundle(bundle, ownership)
    }

    fn finish_bundle(
        &self,
        mut bundle: ImageTensorBundle,
        ownership: Option<PackedOwnership>,
    ) -> Result<GroupedVisionBundle, EncoderBatchingError> {
        if let Some(ownership) = &ownership {
            self.validate_packed_extents(&bundle, ownership)?;
        }
        let structural = self.structural_tensors(&bundle, ownership.as_ref())?;
        let mut tensors = bundle
            .tensors
            .drain(..)
            .chain(structural)
            .map(|tensor| (tensor.name.clone(), tensor))
            .collect::<BTreeMap<_, _>>();
        let mut ordered = Vec::with_capacity(self.outputs.len());
        for output in &self.outputs {
            if let Some(tensor) = tensors.remove(&output.name) {
                ordered.push(tensor);
            } else if !output.optional.unwrap_or(false) {
                return Err(EncoderBatchingError::Preprocessing {
                    detail: format!(
                        "declared preprocessing output '{}' was not materialized",
                        output.name
                    ),
                });
            }
        }
        if !tensors.is_empty() {
            return Err(EncoderBatchingError::Preprocessing {
                detail: format!(
                    "preprocessing produced undeclared outputs: {}",
                    tensors.into_keys().collect::<Vec<_>>().join(", ")
                ),
            });
        }
        bundle.tensors = ordered;
        Ok(GroupedVisionBundle {
            tensors: bundle,
            ownership,
        })
    }

    fn validate_packed_extents(
        &self,
        bundle: &ImageTensorBundle,
        ownership: &PackedOwnership,
    ) -> Result<(), EncoderBatchingError> {
        for output in &self.outputs {
            let Some(contract) = &output.contract else {
                continue;
            };
            let BatchLayout::TokenPacked { axis, .. } = contract.batch_layout else {
                continue;
            };
            let tensor =
                bundle
                    .tensor(&output.name)
                    .ok_or_else(|| EncoderBatchingError::Preprocessing {
                        detail: format!(
                            "packed output '{}' was not produced before ownership validation",
                            output.name
                        ),
                    })?;
            let actual = usize::try_from(tensor.shape[axis]).map_err(|_| {
                EncoderBatchingError::Preprocessing {
                    detail: format!(
                        "packed output '{}' has negative extent {} on axis {axis}",
                        output.name, tensor.shape[axis]
                    ),
                }
            })?;
            if actual != ownership.physical_count() {
                return Err(EncoderBatchingError::UnsupportedExecution {
                    component: self.component.clone(),
                    detail: format!(
                        "output '{}' materialized {actual} positions on its declared packed axis, \
                         but the request topology contains {} physical parts; this runtime only \
                         emits ownership when each input part remains one packed position",
                        output.name,
                        ownership.physical_count()
                    ),
                });
            }
        }
        Ok(())
    }

    fn structural_tensors(
        &self,
        bundle: &ImageTensorBundle,
        ownership: Option<&PackedOwnership>,
    ) -> Result<Vec<NamedImageTensor>, EncoderBatchingError> {
        let mut tensors = Vec::new();
        for output in self
            .outputs
            .iter()
            .filter(|output| is_grouping_output_content(&output.content))
        {
            let contract =
                output
                    .contract
                    .as_ref()
                    .ok_or_else(|| EncoderBatchingError::Preprocessing {
                        detail: format!(
                            "structural preprocessing output '{}' must declare a TensorContract",
                            output.name
                        ),
                    })?;
            if output.dtype != "int64" || contract.dtype != "int64" {
                return Err(EncoderBatchingError::Preprocessing {
                    detail: format!(
                        "structural preprocessing output '{}' must be int64 in both its binding \
                         and TensorContract",
                        output.name
                    ),
                });
            }
            if contract.rank != 1 {
                return Err(EncoderBatchingError::UnsupportedExecution {
                    component: self.component.clone(),
                    detail: format!(
                        "structural output '{}' has rank {}, but this grouped preprocessing slice \
                         emits rank-1 ownership companions and one valid length per physical part",
                        output.name, contract.rank
                    ),
                });
            }
            let values = match output.content.as_str() {
                PACK_OFFSETS_CONTENT => self
                    .ownership_values(&output.name, ownership, true)?
                    .to_vec(),
                PACK_OWNER_CONTENT => self
                    .ownership_values(&output.name, ownership, false)?
                    .to_vec(),
                _ => self.valid_lengths(&output.name, bundle)?,
            };
            tensors.push(NamedImageTensor {
                name: output.name.clone(),
                content: output.content.clone(),
                dtype: ImageTensorDType::Int64,
                shape: vec![i64::try_from(values.len()).map_err(|_| {
                    EncoderBatchingError::Preprocessing {
                        detail: format!("structural output '{}' length exceeds int64", output.name),
                    }
                })?],
                data: ImageTensorData::Int64(values),
            });
        }
        Ok(tensors)
    }

    fn ownership_values<'a>(
        &self,
        name: &str,
        ownership: Option<&'a PackedOwnership>,
        offsets: bool,
    ) -> Result<&'a [i64], EncoderBatchingError> {
        let ownership = ownership.ok_or_else(|| EncoderBatchingError::Preprocessing {
            detail: format!(
                "structural output '{name}' is an ownership companion, but no token_packed \
                 output contract declares an ownership chain"
            ),
        })?;
        for (level_index, (offset_name, owner_name)) in self.ownership_pairs.iter().enumerate() {
            let matches = if offsets {
                offset_name == name
            } else {
                owner_name == name
            };
            if matches {
                let level = &ownership.levels()[level_index];
                return Ok(if offsets {
                    &level.offsets
                } else {
                    &level.owner
                });
            }
        }
        Err(EncoderBatchingError::Preprocessing {
            detail: format!(
                "structural output '{name}' is not referenced by any declared ownership level"
            ),
        })
    }

    fn valid_lengths(
        &self,
        name: &str,
        bundle: &ImageTensorBundle,
    ) -> Result<Vec<i64>, EncoderBatchingError> {
        let mut resolved = None;
        for output in &self.outputs {
            let Some(contract) = &output.contract else {
                continue;
            };
            for padding in contract
                .padding
                .iter()
                .filter(|padding| padding.valid_lengths == name)
            {
                let tensor = bundle.tensor(&output.name).ok_or_else(|| {
                    EncoderBatchingError::Preprocessing {
                        detail: format!(
                            "padded output '{}' was not produced before valid-length generation",
                            output.name
                        ),
                    }
                })?;
                let shape =
                    contract
                        .shape
                        .as_ref()
                        .ok_or_else(|| EncoderBatchingError::Preprocessing {
                            detail: format!(
                                "padded output '{}' requires a declared shape",
                                output.name
                            ),
                        })?;
                let axis = shape
                    .iter()
                    .position(|dimension| {
                        matches!(dimension, TensorDimension::Symbol(symbol) if symbol == &padding.dimension)
                    })
                    .ok_or_else(|| EncoderBatchingError::Preprocessing {
                        detail: format!(
                            "padded output '{}' does not contain declared dimension '{}'",
                            output.name, padding.dimension
                        ),
                    })?;
                let outer = tensor.shape[..axis]
                    .iter()
                    .try_fold(1usize, |total, extent| {
                        let extent = usize::try_from(*extent).map_err(|_| {
                            EncoderBatchingError::Preprocessing {
                                detail: format!(
                                    "padded output '{}' has negative outer extent {extent}",
                                    output.name
                                ),
                            }
                        })?;
                        total.checked_mul(extent).ok_or_else(|| {
                            EncoderBatchingError::Preprocessing {
                                detail: format!(
                                    "padded output '{}' outer extent overflows usize",
                                    output.name
                                ),
                            }
                        })
                    })?;
                if outer != bundle.images.len() {
                    return Err(EncoderBatchingError::UnsupportedExecution {
                        component: self.component.clone(),
                        detail: format!(
                            "valid_lengths '{}' needs {outer} entries for output '{}' dimension \
                             '{}', but preprocessing has {} physical-part summaries; this runtime \
                             will not guess how to expand or reduce length rows",
                            name,
                            output.name,
                            padding.dimension,
                            bundle.images.len()
                        ),
                    });
                }
                let padded_extent = usize::try_from(tensor.shape[axis]).map_err(|_| {
                    EncoderBatchingError::Preprocessing {
                        detail: format!(
                            "padded output '{}' has negative padded extent {}",
                            output.name, tensor.shape[axis]
                        ),
                    }
                })?;
                if bundle
                    .images
                    .iter()
                    .any(|image| image.tensor_length != padded_extent)
                {
                    return Err(EncoderBatchingError::UnsupportedExecution {
                        component: self.component.clone(),
                        detail: format!(
                            "valid_lengths '{}' names output '{}' dimension '{}', but that \
                             dimension is not the processor-owned right-padded physical-part \
                             extent; refusing to derive lengths from unrelated image geometry",
                            name, output.name, padding.dimension
                        ),
                    });
                }
                let lengths = bundle
                    .images
                    .iter()
                    .enumerate()
                    .map(|(index, image)| {
                        if image.expansion_count > padded_extent {
                            return Err(EncoderBatchingError::Preprocessing {
                                detail: format!(
                                    "output '{}' item {index} valid length {} exceeds padded \
                                     dimension '{}' extent {padded_extent}",
                                    output.name, image.expansion_count, padding.dimension
                                ),
                            });
                        }
                        i64::try_from(image.expansion_count).map_err(|_| {
                            EncoderBatchingError::Preprocessing {
                                detail: format!(
                                    "output '{}' item {index} valid length exceeds int64",
                                    output.name
                                ),
                            }
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                match &resolved {
                    Some(previous) if previous != &lengths => {
                        return Err(EncoderBatchingError::Preprocessing {
                            detail: format!(
                                "valid_lengths output '{name}' is referenced by padded outputs \
                                 with different runtime lengths"
                            ),
                        });
                    }
                    Some(_) => {}
                    None => resolved = Some(lengths),
                }
            }
        }
        resolved.ok_or_else(|| EncoderBatchingError::Preprocessing {
            detail: format!(
                "structural length output '{name}' is not referenced by any padding declaration"
            ),
        })
    }

    fn empty_bundle(
        &self,
        request_count: usize,
    ) -> Result<ImageTensorBundle, EncoderBatchingError> {
        let mut tensors = Vec::new();
        for output in self
            .outputs
            .iter()
            .filter(|output| !is_grouping_output_content(&output.content))
        {
            tensors.push(self.empty_tensor(output, request_count)?);
        }
        Ok(ImageTensorBundle {
            tensors,
            images: Vec::new(),
            num_tiles: 0,
            tiles_per_image: Vec::new(),
            tile_grids: Vec::new(),
            thumbnail_position: self.processor.config().tiling.thumbnail_position,
        })
    }

    fn empty_tensor(
        &self,
        output: &VisionOutputBinding,
        request_count: usize,
    ) -> Result<NamedImageTensor, EncoderBatchingError> {
        let contract =
            output
                .contract
                .as_ref()
                .ok_or_else(|| EncoderBatchingError::Preprocessing {
                    detail: format!(
                        "empty grouped output '{}' requires a declared TensorContract",
                        output.name
                    ),
                })?;
        let declared_shape =
            contract
                .shape
                .as_ref()
                .ok_or_else(|| EncoderBatchingError::Preprocessing {
                    detail: format!(
                        "empty grouped output '{}' requires a declared shape",
                        output.name
                    ),
                })?;
        let shape = declared_shape
            .iter()
            .enumerate()
            .map(|(axis, dimension)| match dimension {
                TensorDimension::Fixed(extent) => Ok(*extent),
                TensorDimension::Symbol(symbol)
                    if contract.batch_layout.packed_axis() == Some(axis) =>
                {
                    Ok(0)
                }
                TensorDimension::Symbol(_)
                    if contract.batch_layout.request_axis() == Some(axis) =>
                {
                    i64::try_from(request_count).map_err(|_| EncoderBatchingError::Preprocessing {
                        detail: "request count exceeds int64".to_owned(),
                    })
                }
                TensorDimension::Symbol(symbol)
                    if contract
                        .padding
                        .iter()
                        .any(|padding| padding.dimension == *symbol) =>
                {
                    i64::try_from(self.pad_target_length.unwrap_or(0)).map_err(|_| {
                        EncoderBatchingError::Preprocessing {
                            detail: format!(
                                "empty output '{}' pad target exceeds int64",
                                output.name
                            ),
                        }
                    })
                }
                TensorDimension::Symbol(_) => Ok(0),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let elements = shape.iter().try_fold(1usize, |total, extent| {
            let extent =
                usize::try_from(*extent).map_err(|_| EncoderBatchingError::Preprocessing {
                    detail: format!(
                        "empty output '{}' has negative extent {extent}",
                        output.name
                    ),
                })?;
            total
                .checked_mul(extent)
                .ok_or_else(|| EncoderBatchingError::Preprocessing {
                    detail: format!("empty output '{}' shape overflows usize", output.name),
                })
        })?;
        let dtype = ImageTensorDType::parse(&output.dtype).map_err(|error| {
            EncoderBatchingError::Preprocessing {
                detail: format!("{error:#}"),
            }
        })?;
        Ok(NamedImageTensor {
            name: output.name.clone(),
            content: output.content.clone(),
            dtype,
            shape,
            data: zero_data(dtype, elements),
        })
    }
}

fn declared_ownership_pairs(
    component: &str,
    outputs: &[VisionOutputBinding],
) -> Result<Vec<(String, String)>, EncoderBatchingError> {
    let mut declared = None;
    for output in outputs
        .iter()
        .filter(|output| !is_grouping_output_content(&output.content))
    {
        let Some(contract) = &output.contract else {
            continue;
        };
        let BatchLayout::TokenPacked { levels, .. } = &contract.batch_layout else {
            continue;
        };
        let pairs = levels
            .iter()
            .map(|level| (level.offsets.clone(), level.owner.clone()))
            .collect::<Vec<_>>();
        match &declared {
            Some(previous) if previous != &pairs => {
                return Err(EncoderBatchingError::UnsupportedExecution {
                    component: component.to_owned(),
                    detail: format!(
                        "packed outputs do not share one ownership chain: output '{}' declares \
                         {pairs:?}, while another output declares {previous:?}",
                        output.name
                    ),
                });
            }
            Some(_) => {}
            None => declared = Some(pairs),
        }
    }
    let pairs = declared.unwrap_or_default();
    let companion_names = outputs
        .iter()
        .filter(|output| {
            matches!(
                output.content.as_str(),
                PACK_OFFSETS_CONTENT | PACK_OWNER_CONTENT
            )
        })
        .map(|output| output.name.as_str())
        .collect::<BTreeSet<_>>();
    for (offsets, owner) in &pairs {
        for name in [offsets.as_str(), owner.as_str()] {
            if !companion_names.contains(name) {
                return Err(EncoderBatchingError::Preprocessing {
                    detail: format!(
                        "ownership companion '{name}' is referenced by a packed output but is not \
                         declared as a preprocessing output"
                    ),
                });
            }
        }
    }
    if pairs.is_empty() && !companion_names.is_empty() {
        return Err(EncoderBatchingError::Preprocessing {
            detail: "ownership companion outputs are declared without a token_packed payload"
                .to_owned(),
        });
    }
    Ok(pairs)
}

fn zero_data(dtype: ImageTensorDType, elements: usize) -> ImageTensorData {
    match dtype {
        ImageTensorDType::Fp32 => ImageTensorData::Fp32(vec![0.0; elements]),
        ImageTensorDType::Fp16 => ImageTensorData::Fp16(vec![0; elements]),
        ImageTensorDType::Bf16 => ImageTensorData::Bf16(vec![0; elements]),
        ImageTensorDType::Int64 => ImageTensorData::Int64(vec![0; elements]),
        ImageTensorDType::Int32 => ImageTensorData::Int32(vec![0; elements]),
        ImageTensorDType::Int8 => ImageTensorData::Int8(vec![0; elements]),
        ImageTensorDType::Uint8 => ImageTensorData::Uint8(vec![0; elements]),
        ImageTensorDType::Bool => ImageTensorData::Bool(vec![0; elements]),
    }
}
