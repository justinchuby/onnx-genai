//! Named floating sources and their snapshots in the whole-operation workspace.
use super::super::{decoded_projection::ProjectionScratch, projection_input::ProjectionInput};
use super::error;
use onnx_runtime_ep_api::{Result, TensorMetadata, TensorView};
use std::alloc::Layout;

pub(super) struct MoeInputs<'a> {
    input: ProjectionInput<'a>,
    router: ProjectionInput<'a>,
    fc1: ProjectionInput<'a>,
    fc1_bias: Option<ProjectionInput<'a>>,
    fc2: ProjectionInput<'a>,
    fc2_bias: Option<ProjectionInput<'a>>,
    fc3: Option<ProjectionInput<'a>>,
    fc3_bias: Option<ProjectionInput<'a>>,
}
pub(super) struct MoeInputValues<'a> {
    pub(super) input: &'a [f32],
    pub(super) router: &'a [f32],
    pub(super) fc1: &'a [f32],
    pub(super) fc1_bias: Option<&'a [f32]>,
    pub(super) fc2: &'a [f32],
    pub(super) fc2_bias: Option<&'a [f32]>,
    pub(super) fc3: Option<&'a [f32]>,
    pub(super) fc3_bias: Option<&'a [f32]>,
}
const NAMES: [&str; 8] = [
    "input",
    "router_probs",
    "fc1_experts_weights",
    "fc1_experts_bias",
    "fc2_experts_weights",
    "fc2_experts_bias",
    "fc3_experts_weights",
    "fc3_experts_bias",
];
impl<'a> MoeInputs<'a> {
    pub(super) fn new(
        inputs: &'a [TensorView<'a>],
        metadata: &[TensorMetadata<'_>],
    ) -> Result<Self> {
        let get = |slot: usize| {
            let view = inputs.get(slot).filter(|v| !v.is_absent()).ok_or_else(|| error(format!(
                "input {slot} ('{}') needs a readable Host tensor view before live workspace preparation; bind its storage", NAMES[slot]
            )))?;
            ProjectionInput::new("MoE", NAMES[slot], view)
        };
        let optional = |slot: usize| {
            if inputs.get(slot).is_some_and(|v| !v.is_absent())
                || metadata.get(slot).is_some_and(|v| v.present)
            {
                get(slot).map(Some)
            } else {
                Ok(None)
            }
        };
        Ok(Self {
            input: get(0)?,
            router: get(1)?,
            fc1: get(2)?,
            fc1_bias: optional(3)?,
            fc2: get(4)?,
            fc2_bias: optional(5)?,
            fc3: optional(6)?,
            fc3_bias: optional(7)?,
        })
    }
    pub(super) fn layouts(&self, empty_output: bool) -> Result<[Layout; 8]> {
        let sources = [
            Some(&self.input),
            Some(&self.router),
            Some(&self.fc1),
            self.fc1_bias.as_ref(),
            Some(&self.fc2),
            self.fc2_bias.as_ref(),
            self.fc3.as_ref(),
            self.fc3_bias.as_ref(),
        ];
        let mut layouts = [Layout::new::<()>(); 8];
        for ((layout, source), name) in layouts.iter_mut().zip(sources).zip(NAMES) {
            let count = if empty_output {
                0
            } else {
                source.map_or(0, ProjectionInput::staging_elements)
            };
            *layout = Layout::array::<f32>(count).map_err(|_| error(format!(
                "{name} conversion workspace for {count} f32 values exceeds addressable capacity; reduce input dimensions"
            )))?;
        }
        Ok(layouts)
    }
    pub(super) fn read<'b>(
        &'b self,
        lanes: &'b mut [ProjectionScratch<'_>; 8],
    ) -> Result<MoeInputValues<'b>> {
        let [input, router, fc1, b1, fc2, b2, fc3, b3] = lanes;
        Ok(MoeInputValues {
            input: self.input.read(input)?,
            router: self.router.read(router)?,
            fc1: self.fc1.read(fc1)?,
            fc1_bias: self.fc1_bias.as_ref().map(|s| s.read(b1)).transpose()?,
            fc2: self.fc2.read(fc2)?,
            fc2_bias: self.fc2_bias.as_ref().map(|s| s.read(b2)).transpose()?,
            fc3: self.fc3.as_ref().map(|s| s.read(fc3)).transpose()?,
            fc3_bias: self.fc3_bias.as_ref().map(|s| s.read(b3)).transpose()?,
        })
    }
}
