use std::collections::{BTreeMap, BTreeSet};

use onnx_genai_metadata::{
    BatchLayout, TensorDimension, VisionOutputBinding, VisionPreprocessingProgram,
};

use crate::batching::{EncoderBatchingError, PackedOwnership, RequestOwnership, RequestSpan};

use super::{
    ImagePreprocessor, ImageTensorBundle, ImageTensorDType, ImageTensorData, NamedImageTensor,
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
    valid_length_names: BTreeSet<String>,
    synthesized_output_names: BTreeSet<String>,
    request_factor: Option<usize>,
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
        let grouping = declared_grouping_references(&component, &program.outputs)?;
        let request_factor = declared_request_factor(
            &component,
            &program.outputs,
            &grouping.synthesized_output_names,
        )?;
        let pad_target_length = program
            .transforms
            .iter()
            .rev()
            .find(|transform| transform.op == "pad")
            .and_then(|transform| transform.target_length);
        let processor = ImagePreprocessor::from_input_and_grouped_program(
            shape,
            program,
            &grouping.synthesized_output_names,
        )
        .map_err(|error| EncoderBatchingError::Preprocessing {
            detail: format!("{error:#}"),
        })?;
        Ok(Self {
            component,
            processor,
            outputs: program.outputs.clone(),
            ownership_pairs: grouping.ownership_pairs,
            valid_length_names: grouping.valid_length_names,
            synthesized_output_names: grouping.synthesized_output_names,
            request_factor,
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
        let mut request_physical_counts = Vec::with_capacity(requests.len());
        for (request_index, request) in requests.iter().enumerate() {
            one_level_counts.push(request.items.len());
            let mut request_nested = Vec::with_capacity(request.items.len());
            let expected_unpacked = self.request_factor.unwrap_or(1);
            if level_count == 0 && request.items.len() != expected_unpacked {
                return Err(EncoderBatchingError::UnsupportedExecution {
                    component: self.component.clone(),
                    detail: format!(
                        "request {request_index} contributes {} media items, but the authored \
                         outputs declare no token_packed ownership level and require exactly \
                         {expected_unpacked} request row(s)",
                        request.items.len(),
                    ),
                });
            }
            let mut request_physical_count = 0usize;
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
                request_physical_count = request_physical_count
                    .checked_add(item.parts.len())
                    .ok_or_else(|| EncoderBatchingError::Preprocessing {
                        detail: format!(
                            "request {request_index} physical media count overflows usize"
                        ),
                    })?;
                flattened.extend(item.parts.iter().copied());
            }
            request_physical_counts.push(request_physical_count);
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
        self.finish_bundle(bundle, ownership, &request_physical_counts)
    }

    fn finish_bundle(
        &self,
        mut bundle: ImageTensorBundle,
        ownership: Option<PackedOwnership>,
        request_physical_counts: &[usize],
    ) -> Result<GroupedVisionBundle, EncoderBatchingError> {
        self.validate_output_layouts(&bundle, ownership.as_ref(), request_physical_counts)?;
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

    fn validate_output_layouts(
        &self,
        bundle: &ImageTensorBundle,
        ownership: Option<&PackedOwnership>,
        request_physical_counts: &[usize],
    ) -> Result<(), EncoderBatchingError> {
        let request_count = request_physical_counts.len();
        let physical_count = request_physical_counts
            .iter()
            .try_fold(0usize, |total, count| {
                total
                    .checked_add(*count)
                    .ok_or_else(|| EncoderBatchingError::Preprocessing {
                        detail: "group physical media count overflows usize".to_owned(),
                    })
            })?;
        for output in &self.outputs {
            if self.synthesized_output_names.contains(&output.name) {
                continue;
            }
            let Some(contract) = &output.contract else {
                continue;
            };
            let Some(tensor) = bundle.tensor(&output.name) else {
                if output.optional.unwrap_or(false) {
                    continue;
                }
                return Err(EncoderBatchingError::Preprocessing {
                    detail: format!(
                        "output '{}' was not produced before layout validation",
                        output.name
                    ),
                });
            };
            let (axis, expected_extent) = match contract.batch_layout {
                BatchLayout::Shared => {
                    if request_count > 1 || physical_count > 1 {
                        return Err(EncoderBatchingError::UnsupportedExecution {
                            component: self.component.clone(),
                            detail: format!(
                                "output '{}' declares shared layout, but preprocessing received \
                                 {physical_count} physical media parts across {request_count} \
                                 requests; media-derived rows cannot be broadcast as shared",
                                output.name
                            ),
                        });
                    }
                    continue;
                }
                BatchLayout::RequestAligned { axis } => {
                    require_request_contribution(
                        &self.component,
                        &output.name,
                        request_physical_counts,
                        1,
                        "request_aligned",
                    )?;
                    (axis, request_count)
                }
                BatchLayout::RequestExpanded { axis, factor } => {
                    require_request_contribution(
                        &self.component,
                        &output.name,
                        request_physical_counts,
                        factor,
                        "request_expanded",
                    )?;
                    let expected = request_count.checked_mul(factor).ok_or_else(|| {
                        EncoderBatchingError::Preprocessing {
                            detail: format!(
                                "output '{}' request-expanded extent overflows usize",
                                output.name
                            ),
                        }
                    })?;
                    (axis, expected)
                }
                BatchLayout::TokenPacked { axis, .. } => {
                    let ownership =
                        ownership.ok_or_else(|| EncoderBatchingError::UnsupportedExecution {
                            component: self.component.clone(),
                            detail: format!(
                                "output '{}' declares token_packed layout, but no request \
                                 ownership chain was assembled",
                                output.name
                            ),
                        })?;
                    (axis, ownership.physical_count())
                }
                BatchLayout::RuntimeSequenceState => {
                    return Err(EncoderBatchingError::UnsupportedExecution {
                        component: self.component.clone(),
                        detail: format!(
                            "output '{}' declares runtime_sequence_state, which a media \
                             preprocessing adapter cannot produce",
                            output.name
                        ),
                    });
                }
            };
            let Some(actual_extent) = tensor.shape.get(axis) else {
                return Err(EncoderBatchingError::Preprocessing {
                    detail: format!(
                        "output '{}' layout axis {axis} is outside produced shape {:?}",
                        output.name, tensor.shape
                    ),
                });
            };
            let actual_extent = usize::try_from(*actual_extent).map_err(|_| {
                EncoderBatchingError::Preprocessing {
                    detail: format!(
                        "output '{}' has negative extent {actual_extent} on layout axis {axis}",
                        output.name
                    ),
                }
            })?;
            if actual_extent == expected_extent {
                continue;
            }
            return Err(EncoderBatchingError::UnsupportedExecution {
                component: self.component.clone(),
                detail: format!(
                    "output '{}' materialized {actual_extent} positions on layout axis {axis}, \
                     but its authored {} layout requires {expected_extent} for these request \
                     contributions",
                    output.name,
                    contract.batch_layout.kind_name()
                ),
            });
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
            .filter(|output| self.synthesized_output_names.contains(&output.name))
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
            let values = if self
                .ownership_pairs
                .iter()
                .any(|(offsets, _)| offsets == &output.name)
            {
                self.ownership_values(&output.name, ownership, true)?
                    .to_vec()
            } else if self
                .ownership_pairs
                .iter()
                .any(|(_, owner)| owner == &output.name)
            {
                self.ownership_values(&output.name, ownership, false)?
                    .to_vec()
            } else if self.valid_length_names.contains(&output.name) {
                self.valid_lengths(&output.name, bundle)?
            } else {
                return Err(EncoderBatchingError::Preprocessing {
                    detail: format!(
                        "output '{}' was selected for runtime synthesis without a contract \
                         reference",
                        output.name
                    ),
                });
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
            .filter(|output| !self.synthesized_output_names.contains(&output.name))
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
                TensorDimension::Symbol(_) if contract.batch_layout.packed_axis() == Some(axis) => {
                    Ok(0)
                }
                TensorDimension::Symbol(_) => match contract.batch_layout {
                    BatchLayout::RequestAligned { axis: request_axis } if request_axis == axis => {
                        i64::try_from(request_count).map_err(|_| {
                            EncoderBatchingError::Preprocessing {
                                detail: "request count exceeds int64".to_owned(),
                            }
                        })
                    }
                    BatchLayout::RequestExpanded {
                        axis: request_axis,
                        factor,
                    } if request_axis == axis => {
                        let expanded = request_count.checked_mul(factor).ok_or_else(|| {
                            EncoderBatchingError::Preprocessing {
                                detail: format!(
                                    "empty output '{}' request-expanded extent overflows usize",
                                    output.name
                                ),
                            }
                        })?;
                        i64::try_from(expanded).map_err(|_| EncoderBatchingError::Preprocessing {
                            detail: format!(
                                "empty output '{}' request-expanded extent exceeds int64",
                                output.name
                            ),
                        })
                    }
                    _ => self.empty_inner_extent(output, contract, dimension),
                },
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

    fn empty_inner_extent(
        &self,
        output: &VisionOutputBinding,
        contract: &onnx_genai_metadata::TensorContract,
        dimension: &TensorDimension,
    ) -> Result<i64, EncoderBatchingError> {
        let TensorDimension::Symbol(symbol) = dimension else {
            unreachable!("empty inner extent is called only for symbolic dimensions");
        };
        if contract
            .padding
            .iter()
            .any(|padding| padding.dimension == *symbol)
        {
            let target = self.pad_target_length.ok_or_else(|| {
                EncoderBatchingError::UnsupportedExecution {
                    component: self.component.clone(),
                    detail: format!(
                        "empty output '{}' padded dimension '{}' has no exact processor pad \
                         target; refusing to invent extent zero",
                        output.name, symbol
                    ),
                }
            })?;
            return i64::try_from(target).map_err(|_| EncoderBatchingError::Preprocessing {
                detail: format!("empty output '{}' pad target exceeds int64", output.name),
            });
        }
        Err(EncoderBatchingError::UnsupportedExecution {
            component: self.component.clone(),
            detail: format!(
                "empty output '{}' symbolic inner dimension '{}' has no exact static or \
                 processor-derived extent; refusing to invent extent zero",
                output.name, symbol
            ),
        })
    }
}

#[derive(Debug, Default)]
struct GroupingReferences {
    ownership_pairs: Vec<(String, String)>,
    valid_length_names: BTreeSet<String>,
    synthesized_output_names: BTreeSet<String>,
}

fn declared_grouping_references(
    component: &str,
    outputs: &[VisionOutputBinding],
) -> Result<GroupingReferences, EncoderBatchingError> {
    let mut declared = None;
    let mut valid_length_names = BTreeSet::new();
    for output in outputs {
        let Some(contract) = &output.contract else {
            continue;
        };
        if let BatchLayout::TokenPacked { levels, .. } = &contract.batch_layout {
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
        for padding in &contract.padding {
            valid_length_names.insert(padding.valid_lengths.clone());
        }
    }
    let ownership_pairs = declared.unwrap_or_default();
    let output_names = outputs
        .iter()
        .map(|output| output.name.clone())
        .collect::<BTreeSet<_>>();
    let mut referenced_roles = BTreeMap::<String, &'static str>::new();
    let mut register = |name: &str, role: &'static str| -> Result<(), EncoderBatchingError> {
        if let Some(previous) = referenced_roles.insert(name.to_owned(), role)
            && previous != role
        {
            return Err(EncoderBatchingError::Preprocessing {
                detail: format!(
                    "preprocessing output '{name}' is referenced as both {previous} and {role}; \
                     one value cannot carry two structural meanings"
                ),
            });
        }
        if !output_names.contains(name) {
            return Err(EncoderBatchingError::Preprocessing {
                detail: format!(
                    "structural companion '{name}' is referenced by a payload contract but is not \
                     declared as a preprocessing output"
                ),
            });
        }
        Ok(())
    };
    for (offsets, owner) in &ownership_pairs {
        register(offsets, "ownership offsets")?;
        register(owner, "ownership owner map")?;
    }
    for name in &valid_length_names {
        register(name, "padding valid lengths")?;
    }
    Ok(GroupingReferences {
        ownership_pairs,
        valid_length_names,
        synthesized_output_names: referenced_roles.into_keys().collect(),
    })
}

fn declared_request_factor(
    component: &str,
    outputs: &[VisionOutputBinding],
    synthesized_output_names: &BTreeSet<String>,
) -> Result<Option<usize>, EncoderBatchingError> {
    let mut declared = None;
    for output in outputs {
        if synthesized_output_names.contains(&output.name) {
            continue;
        }
        let Some(contract) = &output.contract else {
            continue;
        };
        let factor = match contract.batch_layout {
            BatchLayout::RequestAligned { .. } => 1,
            BatchLayout::RequestExpanded { factor, .. } => factor,
            BatchLayout::Shared
            | BatchLayout::TokenPacked { .. }
            | BatchLayout::RuntimeSequenceState => continue,
        };
        if factor == 0 {
            return Err(EncoderBatchingError::Preprocessing {
                detail: format!(
                    "output '{}' declares request_expanded factor 0",
                    output.name
                ),
            });
        }
        match declared {
            Some(previous) if previous != factor => {
                return Err(EncoderBatchingError::UnsupportedExecution {
                    component: component.to_owned(),
                    detail: format!(
                        "request-scoped preprocessing outputs require different physical rows per \
                         request: a previous output requires {previous}, while '{}' requires \
                         {factor}",
                        output.name
                    ),
                });
            }
            Some(_) => {}
            None => declared = Some(factor),
        }
    }
    Ok(declared)
}

fn require_request_contribution(
    component: &str,
    output: &str,
    request_physical_counts: &[usize],
    expected: usize,
    layout: &str,
) -> Result<(), EncoderBatchingError> {
    if request_physical_counts
        .iter()
        .all(|count| *count == expected)
    {
        return Ok(());
    }
    Err(EncoderBatchingError::UnsupportedExecution {
        component: component.to_owned(),
        detail: format!(
            "output '{output}' declares {layout} layout requiring {expected} physical row(s) per \
             request, but ordered request contributions are {request_physical_counts:?}; the \
             processor cannot aggregate or fabricate per-request rows"
        ),
    })
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
