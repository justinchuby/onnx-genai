//! Floating input snapshots share the complete expert-operation workspace.
use super::super::{decoded_projection::ProjectionScratch, projection_input::ProjectionInput};
use super::{INPUT_NAMES, OP, error, optional_input, optional_metadata};
use onnx_runtime_ep_api::{Result, TensorMetadata, TensorView};
use onnx_runtime_ir::DataType;
use std::alloc::Layout;

pub(super) struct MoeInputs<'a> {
    input: ProjectionInput<'a>,
    router_logits: ProjectionInput<'a>,
    router_weights: Option<ProjectionInput<'a>>,
    fc1_bias: Option<ProjectionInput<'a>>,
    fc2_bias: Option<ProjectionInput<'a>>,
    fc3_bias: Option<ProjectionInput<'a>>,
}

pub(super) struct MoeInputValues<'a> {
    pub(super) input: &'a [f32],
    pub(super) router_logits: &'a [f32],
    pub(super) router_weights: Option<&'a [f32]>,
    pub(super) fc1_bias: Option<&'a [f32]>,
    pub(super) fc2_bias: Option<&'a [f32]>,
    pub(super) fc3_bias: Option<&'a [f32]>,
}

impl<'a> MoeInputs<'a> {
    pub(super) fn new(
        inputs: &'a [TensorView<'a>],
        metadata: &[TensorMetadata<'_>],
    ) -> Result<Self> {
        let get = |slot: usize| {
            let view = optional_input(inputs, slot).ok_or_else(|| error(format!(
                "input {slot} ('{}') needs readable Host storage before live workspace preparation; bind its tensor view", INPUT_NAMES[slot]
            )))?;
            // The shared descriptor also serves half-capable MatMul. This op's
            // F32-only runtime contract must hold for the actual view as well.
            if view.dtype != DataType::Float32 {
                return Err(error(format!(
                    "input {slot} ('{}') must be Float32, got {:?}; bind F32 storage for BlockQuantizedMoE",
                    INPUT_NAMES[slot], view.dtype
                )));
            }
            ProjectionInput::new(OP, INPUT_NAMES[slot], view)
        };
        let optional = |slot| {
            if optional_metadata(metadata, slot).is_some() || optional_input(inputs, slot).is_some()
            {
                get(slot).map(Some)
            } else {
                Ok(None)
            }
        };
        Ok(Self {
            input: get(0)?,
            router_logits: get(1)?,
            router_weights: optional(8)?,
            fc1_bias: optional(3)?,
            fc2_bias: optional(5)?,
            fc3_bias: optional(7)?,
        })
    }

    pub(super) fn layouts(&self) -> Result<[Layout; 6]> {
        let sources = [
            (Some(&self.input), "input"),
            (Some(&self.router_logits), "router_logits"),
            (self.router_weights.as_ref(), "router_weights"),
            (self.fc1_bias.as_ref(), "fc1_experts_bias"),
            (self.fc2_bias.as_ref(), "fc2_experts_bias"),
            (self.fc3_bias.as_ref(), "fc3_experts_bias"),
        ];
        let mut layouts = [Layout::new::<()>(); 6];
        for (layout, (source, name)) in layouts.iter_mut().zip(sources) {
            let count = source.map_or(0, ProjectionInput::staging_elements);
            *layout = Layout::array::<f32>(count).map_err(|_| error(format!(
                "{name} conversion workspace for {count} f32 values exceeds addressable capacity; reduce input dimensions"
            )))?;
        }
        Ok(layouts)
    }

    pub(super) fn read<'b>(
        &'b self,
        lanes: &'b mut [ProjectionScratch<'_>; 6],
    ) -> Result<MoeInputValues<'b>> {
        let [input, logits, weights, fc1, fc2, fc3] = lanes;
        Ok(MoeInputValues {
            input: self.input.read(input)?,
            router_logits: self.router_logits.read(logits)?,
            router_weights: self
                .router_weights
                .as_ref()
                .map(|s| s.read(weights))
                .transpose()?,
            fc1_bias: self.fc1_bias.as_ref().map(|s| s.read(fc1)).transpose()?,
            fc2_bias: self.fc2_bias.as_ref().map(|s| s.read(fc2)).transpose()?,
            fc3_bias: self.fc3_bias.as_ref().map(|s| s.read(fc3)).transpose()?,
        })
    }
}
