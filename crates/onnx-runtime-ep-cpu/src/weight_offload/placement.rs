//! Pure Phase-3a VRAM placement, arbitration, and transfer-tile sizing.

use onnx_runtime_loader::{Pageability, WeightRegionCatalog};

/// One model layer and the expert regions that must share its compute device.
#[derive(Clone, Copy, Debug)]
pub struct LayerWeightRegions<'a> {
    pub layer_index: usize,
    pub name: &'a str,
    pub regions: &'a [WeightRegionCatalog],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    Device,
    Host,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionPlacement {
    pub region_index: usize,
    pub bytes: u64,
    pub placement: Placement,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostFallbackReason {
    NonPageableRegion {
        region_index: usize,
        reason: String,
    },
    InsufficientBudget {
        required_bytes: u64,
        remaining_bytes: u64,
    },
    GpuLayersLimit {
        requested_layers: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerPlacement {
    pub layer_index: usize,
    pub name: String,
    pub bytes: u64,
    pub placement: Placement,
    pub regions: Vec<RegionPlacement>,
    pub fallback_reason: Option<HostFallbackReason>,
}

/// Translation of the compatibility `gpu_layers:N` request into bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuLayersOverrideReport {
    pub requested_layers: usize,
    pub available_layers: usize,
    pub equivalent_bytes: u64,
}

/// Deterministic, human-explainable whole-layer placement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacementPlan {
    pub coordinated_weight_budget_bytes: u64,
    pub effective_budget_bytes: u64,
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub gpu_layers_override: Option<GpuLayersOverrideReport>,
    pub layers: Vec<LayerPlacement>,
    pub explanation: String,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum PlacementError {
    #[error("layer {layer_index} ({name}) has no weight regions")]
    EmptyLayer { layer_index: usize, name: String },
    #[error("{context} overflow")]
    ByteOverflow { context: &'static str },
}

/// Plan device residency greedily in model layer order.
///
/// `gpu_layers` is translated to the bytes needed by the requested layer
/// prefix. It may reduce the effective budget, but can never exceed
/// `coordinated_weight_budget_bytes`, which the caller obtains from the memory
/// governor -- this planner decides *which* layers go on the device, not how
/// much of the device it may have.
pub fn plan_placement(
    layers: &[LayerWeightRegions<'_>],
    coordinated_weight_budget_bytes: u64,
    gpu_layers: Option<usize>,
) -> Result<PlacementPlan, PlacementError> {
    let override_report = gpu_layers
        .map(|requested_layers| {
            let available_layers = requested_layers.min(layers.len());
            let mut equivalent_bytes = 0u64;
            for layer in layers.iter().take(available_layers) {
                equivalent_bytes = equivalent_bytes
                    .checked_add(layer_tensor_bytes(layer)?)
                    .ok_or(PlacementError::ByteOverflow {
                        context: "gpu_layers equivalent byte count",
                    })?;
            }
            Ok(GpuLayersOverrideReport {
                requested_layers,
                available_layers,
                equivalent_bytes,
            })
        })
        .transpose()?;
    let effective_budget_bytes = override_report
        .map(|report| coordinated_weight_budget_bytes.min(report.equivalent_bytes))
        .unwrap_or(coordinated_weight_budget_bytes);

    let mut remaining = effective_budget_bytes;
    let mut device_bytes = 0u64;
    let mut host_bytes = 0u64;
    let mut placements = Vec::with_capacity(layers.len());
    let mut details = Vec::with_capacity(layers.len());

    for (order, layer) in layers.iter().enumerate() {
        if layer.regions.is_empty() {
            return Err(PlacementError::EmptyLayer {
                layer_index: layer.layer_index,
                name: layer.name.to_owned(),
            });
        }
        let bytes = layer_tensor_bytes(layer)?;
        let non_pageable = layer
            .regions
            .iter()
            .enumerate()
            .find_map(|(index, catalog)| match catalog.pageability() {
                Pageability::Pageable => None,
                Pageability::NonPageable(reason) => Some((index, format!("{reason:?}"))),
            });
        let fallback_reason = if let Some(report) = override_report
            && order >= report.available_layers
        {
            Some(HostFallbackReason::GpuLayersLimit {
                requested_layers: report.requested_layers,
            })
        } else if let Some((region_index, reason)) = non_pageable {
            Some(HostFallbackReason::NonPageableRegion {
                region_index,
                reason,
            })
        } else if bytes > remaining {
            Some(HostFallbackReason::InsufficientBudget {
                required_bytes: bytes,
                remaining_bytes: remaining,
            })
        } else {
            None
        };
        let placement = if fallback_reason.is_none() {
            remaining = remaining
                .checked_sub(bytes)
                .ok_or(PlacementError::ByteOverflow {
                    context: "remaining placement budget",
                })?;
            device_bytes = device_bytes
                .checked_add(bytes)
                .ok_or(PlacementError::ByteOverflow {
                    context: "device placement byte count",
                })?;
            Placement::Device
        } else {
            host_bytes = host_bytes
                .checked_add(bytes)
                .ok_or(PlacementError::ByteOverflow {
                    context: "host placement byte count",
                })?;
            Placement::Host
        };
        let mut regions = Vec::with_capacity(layer.regions.len());
        for (region_index, catalog) in layer.regions.iter().enumerate() {
            let bytes = catalog_tensor_bytes(catalog)?;
            regions.push(RegionPlacement {
                region_index,
                bytes,
                placement,
            });
        }
        let reason = match &fallback_reason {
            None => format!("device ({bytes} bytes; {remaining} bytes remain)"),
            Some(HostFallbackReason::NonPageableRegion {
                region_index,
                reason,
            }) => format!("host (region {region_index} is non-pageable: {reason})"),
            Some(HostFallbackReason::InsufficientBudget {
                required_bytes,
                remaining_bytes,
            }) => format!("host (needs {required_bytes} bytes; {remaining_bytes} bytes remain)"),
            Some(HostFallbackReason::GpuLayersLimit { requested_layers }) => {
                format!("host (outside gpu_layers:{requested_layers} prefix)")
            }
        };
        details.push(format!(
            "layer {} ({}) -> {reason}",
            layer.layer_index, layer.name
        ));
        placements.push(LayerPlacement {
            layer_index: layer.layer_index,
            name: layer.name.to_owned(),
            bytes,
            placement,
            regions,
            fallback_reason,
        });
    }

    let source = override_report.map_or_else(
        || "byte budget".to_owned(),
        |report| {
            format!(
                "gpu_layers:{} = {} bytes ({} layers available)",
                report.requested_layers, report.equivalent_bytes, report.available_layers
            )
        },
    );
    let explanation = format!(
        "VRAM placement: source={source}; coordinated={} bytes; effective={} bytes; \
         device={} bytes; host={} bytes. {}",
        coordinated_weight_budget_bytes,
        effective_budget_bytes,
        device_bytes,
        host_bytes,
        details.join("; ")
    );

    Ok(PlacementPlan {
        coordinated_weight_budget_bytes,
        effective_budget_bytes,
        device_bytes,
        host_bytes,
        gpu_layers_override: override_report,
        layers: placements,
        explanation,
    })
}

fn layer_tensor_bytes(layer: &LayerWeightRegions<'_>) -> Result<u64, PlacementError> {
    let mut bytes = 0u64;
    for catalog in layer.regions {
        let region_bytes = catalog_tensor_bytes(catalog)?;
        bytes = bytes
            .checked_add(region_bytes)
            .ok_or(PlacementError::ByteOverflow {
                context: "layer weight byte count",
            })?;
    }
    Ok(bytes)
}

fn catalog_tensor_bytes(catalog: &WeightRegionCatalog) -> Result<u64, PlacementError> {
    let layout = catalog.layout();
    let elements = layout
        .experts
        .checked_mul(layout.rows_per_expert)
        .and_then(|value| value.checked_mul(layout.storage_elements_per_row))
        .ok_or(PlacementError::ByteOverflow {
            context: "weight region element count",
        })?;
    let bytes =
        catalog
            .dtype()
            .checked_storage_bytes(elements)
            .ok_or(PlacementError::ByteOverflow {
                context: "weight region byte count",
            })?;
    u64::try_from(bytes).map_err(|_| PlacementError::ByteOverflow {
        context: "weight region byte count",
    })
}
// The coordinated weight/KV/scratch VRAM arbitration that used to live here is
// gone. The memory governor's ledger answers the same question -- one authority
// deciding how a device tier is divided between holders -- and two answers to
// that question is what the ledger work exists to end.
//
// The *placement* planner above stays. Deciding which layers live on the device
// is a different question, nothing else answers it, and Phase 3b is waiting on
// it (docs/memory/WEIGHT_OFFLOAD.md). What it must not do is carry its own budget:
// when it is wired, it should ask the governor how much room there is rather
// than arbitrating for itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IqFormat {
    Iq1S,
    Iq1M,
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    Iq3Xxs,
    Iq3S,
    Iq4Nl,
    Iq4Xs,
}

impl IqFormat {
    pub const fn encoded_block_bytes(self) -> u64 {
        match self {
            Self::Iq1S => 50,
            Self::Iq1M => 56,
            Self::Iq2Xxs => 66,
            Self::Iq2Xs => 74,
            Self::Iq2S => 82,
            Self::Iq3Xxs => 98,
            Self::Iq3S => 110,
            Self::Iq4Nl => 18,
            Self::Iq4Xs => 136,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantTileFormat {
    Mxfp4,
    Iq(IqFormat),
    AffineInt2 { block_size: u64 },
    AffineInt4 { block_size: u64 },
}

impl QuantTileFormat {
    pub fn encoded_block_bytes(self) -> Result<u64, TileSizeError> {
        match self {
            Self::Mxfp4 => Ok(17),
            Self::Iq(format) => Ok(format.encoded_block_bytes()),
            Self::AffineInt2 { block_size } => affine_block_bytes(block_size, 2),
            Self::AffineInt4 { block_size } => affine_block_bytes(block_size, 4),
        }
    }

    pub fn minimum_tile_bytes(self) -> Result<u64, TileSizeError> {
        self.encoded_block_bytes()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnappedTileSize {
    pub requested_bytes: u64,
    pub snapped_bytes: u64,
    pub quant_block_bytes: u64,
    pub minimum_bytes: u64,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum TileSizeError {
    #[error("quantization block size must be non-zero")]
    ZeroBlockSize,
    #[error("quantization tile byte count overflow")]
    ByteOverflow,
}

/// Snap a byte override down to whole quant blocks, clamping requests smaller
/// than the format minimum up to one complete block.
pub fn snap_transfer_tile_bytes(
    requested_bytes: u64,
    format: QuantTileFormat,
) -> Result<SnappedTileSize, TileSizeError> {
    let quant_block_bytes = format.encoded_block_bytes()?;
    let minimum_bytes = format.minimum_tile_bytes()?;
    let snapped_bytes = if requested_bytes < minimum_bytes {
        minimum_bytes
    } else {
        (requested_bytes / quant_block_bytes)
            .checked_mul(quant_block_bytes)
            .ok_or(TileSizeError::ByteOverflow)?
    };
    Ok(SnappedTileSize {
        requested_bytes,
        snapped_bytes,
        quant_block_bytes,
        minimum_bytes,
    })
}

fn affine_block_bytes(block_size: u64, bits: u64) -> Result<u64, TileSizeError> {
    if block_size == 0 {
        return Err(TileSizeError::ZeroBlockSize);
    }
    let packed_bits = block_size
        .checked_mul(bits)
        .ok_or(TileSizeError::ByteOverflow)?;
    packed_bits
        .checked_add(7)
        .ok_or(TileSizeError::ByteOverflow)
        .map(|bits| bits / 8)
}

#[cfg(test)]
mod tests {
    use onnx_runtime_ir::DataType;
    use onnx_runtime_loader::{
        ExpertQuantization, ExpertStorageOrder, ExpertTensorLayout, WeightRegionCatalog,
    };

    use super::*;

    fn catalog(experts: usize, bytes_per_expert: usize) -> WeightRegionCatalog {
        let tensor_bytes = experts
            .checked_mul(bytes_per_expert)
            .expect("test catalog byte count");
        WeightRegionCatalog::for_mapped_tensor_view(
            DataType::Uint8,
            &[experts, 1, bytes_per_expert],
            tensor_bytes,
            ExpertTensorLayout {
                version: 1,
                experts,
                rows_per_expert: 1,
                storage_elements_per_row: bytes_per_expert,
                order: ExpertStorageOrder::ExpertMajor,
                quantization: Some(ExpertQuantization {
                    bits: 4,
                    block_size: 16,
                    blocks_per_row: 1,
                }),
            },
        )
    }

    #[test]
    fn placement_is_greedy_explainable_and_whole_layer() {
        let layer0 = [catalog(2, 50), catalog(2, 25)];
        let layer1 = [catalog(2, 40)];
        let layer2 = [catalog(1, 25)];
        let layers = [
            LayerWeightRegions {
                layer_index: 0,
                name: "moe.0",
                regions: &layer0,
            },
            LayerWeightRegions {
                layer_index: 1,
                name: "moe.1",
                regions: &layer1,
            },
            LayerWeightRegions {
                layer_index: 2,
                name: "moe.2",
                regions: &layer2,
            },
        ];
        let plan = plan_placement(&layers, 180, None).unwrap();
        assert_eq!(plan.device_bytes, 175);
        assert_eq!(plan.layers[0].placement, Placement::Device);
        assert!(
            plan.layers[0]
                .regions
                .iter()
                .all(|region| region.placement == Placement::Device)
        );
        assert_eq!(plan.layers[1].placement, Placement::Host);
        assert_eq!(plan.layers[2].placement, Placement::Device);
        assert!(
            plan.explanation
                .contains("layer 1 (moe.1) -> host (needs 80 bytes; 30 bytes remain)")
        );
    }

    #[test]
    fn gpu_layers_override_reports_bytes_and_respects_coordinated_cap() {
        let layer0 = [catalog(1, 100)];
        let layer1 = [catalog(1, 80)];
        let layer2 = [catalog(1, 20)];
        let layers = [
            LayerWeightRegions {
                layer_index: 0,
                name: "0",
                regions: &layer0,
            },
            LayerWeightRegions {
                layer_index: 1,
                name: "1",
                regions: &layer1,
            },
            LayerWeightRegions {
                layer_index: 2,
                name: "2",
                regions: &layer2,
            },
        ];
        let plan = plan_placement(&layers, 150, Some(2)).unwrap();
        assert_eq!(
            plan.gpu_layers_override,
            Some(GpuLayersOverrideReport {
                requested_layers: 2,
                available_layers: 2,
                equivalent_bytes: 180,
            })
        );
        assert_eq!(plan.effective_budget_bytes, 150);
        assert_eq!(plan.layers[0].placement, Placement::Device);
        assert_eq!(plan.layers[1].placement, Placement::Host);
        assert_eq!(
            plan.layers[2].fallback_reason,
            Some(HostFallbackReason::GpuLayersLimit {
                requested_layers: 2
            })
        );
    }

    #[test]
    fn non_pageable_region_keeps_its_host_byte_accounting() {
        let region = WeightRegionCatalog::for_mapped_tensor_view(
            DataType::Uint8,
            &[1, 1, 64],
            64,
            ExpertTensorLayout {
                version: 1,
                experts: 1,
                rows_per_expert: 1,
                storage_elements_per_row: 64,
                order: ExpertStorageOrder::Interleaved,
                quantization: None,
            },
        );
        let layer_regions = [region];
        let layers = [LayerWeightRegions {
            layer_index: 7,
            name: "resident-only",
            regions: &layer_regions,
        }];
        let plan = plan_placement(&layers, 128, None).unwrap();
        assert_eq!(plan.device_bytes, 0);
        assert_eq!(plan.host_bytes, 64);
        assert_eq!(plan.layers[0].bytes, 64);
        assert!(matches!(
            plan.layers[0].fallback_reason,
            Some(HostFallbackReason::NonPageableRegion {
                region_index: 0,
                ..
            })
        ));
    }

    #[test]
    fn transfer_tiles_snap_to_whole_format_blocks() {
        assert_eq!(
            snap_transfer_tile_bytes(100, QuantTileFormat::Mxfp4).unwrap(),
            SnappedTileSize {
                requested_bytes: 100,
                snapped_bytes: 85,
                quant_block_bytes: 17,
                minimum_bytes: 17,
            }
        );
        assert_eq!(
            snap_transfer_tile_bytes(1, QuantTileFormat::Iq(IqFormat::Iq2S))
                .unwrap()
                .snapped_bytes,
            82
        );
        assert_eq!(
            snap_transfer_tile_bytes(70, QuantTileFormat::AffineInt4 { block_size: 32 })
                .unwrap()
                .snapped_bytes,
            64
        );
        assert_eq!(
            snap_transfer_tile_bytes(13, QuantTileFormat::AffineInt2 { block_size: 32 })
                .unwrap()
                .snapped_bytes,
            8
        );
    }
}
