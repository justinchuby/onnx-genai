//! Inference BatchNormalization with one whole-call storage snapshot.

use super::{norm_ops::NormAcc, typed_workspace::TypedWorkspace};
use crate::{
    dispatch_float,
    dtype::{
        ComputeDomain, NumericElem, to_dense_into, validate_f32_widen_input, validate_host_output,
        write_dense,
    },
};
use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, OutputMetadata, Result, TensorMetadata, TensorMut, TensorView,
    WorkspacePreparation, WorkspaceView,
};
use onnx_runtime_ir::Node;

const OP: &str = "BatchNormalization";
const INPUTS: [&str; 5] = ["X", "scale", "B", "input_mean", "input_var"];
fn fail(why: impl std::fmt::Display) -> EpError {
    EpError::KernelFailed(format!("{OP}: {why}"))
}

pub struct BatchNormalizationFactory;
impl KernelFactory for BatchNormalizationFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let training = node
            .attr("training_mode")
            .and_then(|a| a.as_int())
            .unwrap_or(0);
        if training != 0 {
            return Err(fail(format!(
                "training_mode={training} is unsupported by the inference-only CPU EP; set training_mode=0 or select an EP supporting training"
            )));
        }
        Ok(Box::new(BatchNormalization {
            epsilon: node
                .attr("epsilon")
                .and_then(|a| a.as_float())
                .unwrap_or(1e-5),
        }))
    }
}
struct BatchNormalization {
    epsilon: f32,
}
struct Geometry {
    count: usize,
    channels: usize,
    spatial: usize,
}
impl Geometry {
    fn new(inputs: &[TensorMetadata<'_>], outputs: &[OutputMetadata<'_>]) -> Result<Self> {
        let [x, scale, bias, mean, variance] = inputs else {
            return Err(fail(format!(
                "expected exactly 5 present inputs {INPUTS:?}, got {}; bind all five inputs",
                inputs.len()
            )));
        };
        if !x.present || !super::FLOAT_DTYPES.contains(&x.dtype) {
            return Err(fail(format!(
                "X must be a present floating tensor, got present={} dtype {:?}; bind supported floating X",
                x.present, x.dtype
            )));
        }
        if x.shape.len() < 2 {
            return Err(fail(format!(
                "X shape {:?} must have rank at least 2 (N,C,...); provide batch and channel axes",
                x.shape
            )));
        }
        let channels = x.shape[1];
        let product = |dims: &[usize]| {
            dims.iter()
                .rev()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| {
                    fail(format!(
                        "X shape {:?} overflows dense indexing; reduce dimensions",
                        x.shape
                    ))
                })
        };
        let count = product(x.shape)?;
        let spatial = product(&x.shape[2..])?;
        if channels == 0 || spatial == 0 {
            return Err(fail(format!(
                "X shape {:?} has empty channel or spatial dimensions; provide nonempty C and spatial axes (empty N is supported)",
                x.shape
            )));
        }
        for (name, parameter) in INPUTS[1..].iter().zip([scale, bias, mean, variance]) {
            if !parameter.present || parameter.dtype != x.dtype || parameter.shape != [channels] {
                return Err(fail(format!(
                    "{name} must be present with X dtype {:?} and shape [{channels}], got present={} dtype {:?} shape {:?}; bind a matching channel vector",
                    x.dtype, parameter.present, parameter.dtype, parameter.shape
                )));
            }
        }
        let [y] = outputs else {
            return Err(fail(format!(
                "inference mode produces exactly 1 output, got {}; bind only Y",
                outputs.len()
            )));
        };
        match y {
            OutputMetadata::Tensor { dtype, shape }
                if *dtype == x.dtype && shape.as_ref() == x.shape => {}
            OutputMetadata::Unresolved { dtype } if *dtype == x.dtype => {}
            _ => {
                return Err(fail(format!(
                    "Y {y:?} must match X dtype {:?} and complete shape {:?}; bind a matching output",
                    x.dtype, x.shape
                )));
            }
        }
        // Even empty N retains real channel parameters; validate their capacity.
        dispatch_float!(x.dtype, OP, T => {
            std::alloc::Layout::array::<T>(channels).map_err(|_| fail(format!("channel count {channels} exceeds {:?} storage capacity; reduce C", x.dtype)))?;
            Ok(())
        })?;
        Ok(Self {
            count,
            channels,
            spatial,
        })
    }
    fn workspace<T: NumericElem + Default>(&self) -> Result<TypedWorkspace<T, 1>> {
        TypedWorkspace::new(OP, [("X/Y snapshot", self.count)])
    }
}

// Prove all offset arithmetic used by the existing scalar gather/scatter fits
// the host pointer domain, including negative strides and packed byte origins.
// Allocation bounds remain the owning EP's responsibility, as for TensorView.
fn addresses(
    name: &str,
    base: usize,
    byte_offset: usize,
    shape: &[usize],
    strides: &[i64],
    width: usize,
) -> Result<()> {
    let error = || {
        fail(format!(
            "{name} shape {shape:?}, strides {strides:?}, byte_offset {byte_offset} has an unaddressable storage range; bind valid Host backing with representable strides and offsets"
        ))
    };
    let origin = base.checked_add(byte_offset).ok_or_else(error)?;
    if shape.contains(&0) {
        return Ok(());
    }
    let (mut low, mut high) = (0i128, 0i128);
    for (&dim, &stride) in shape.iter().zip(strides) {
        let extent = ((dim - 1) as i128)
            .checked_mul(i128::from(stride))
            .ok_or_else(error)?;
        if extent < 0 {
            low = low.checked_add(extent).ok_or_else(error)?;
        } else {
            high = high.checked_add(extent).ok_or_else(error)?;
        }
    }
    let span = high
        .checked_sub(low)
        .and_then(|v| v.checked_add(1))
        .and_then(|v| v.checked_mul(width as i128))
        .ok_or_else(error)?;
    if span > isize::MAX as i128 {
        return Err(error());
    }
    let first = low
        .checked_mul(width as i128)
        .and_then(|v| isize::try_from(v).ok())
        .ok_or_else(error)?;
    let end = high
        .checked_add(1)
        .and_then(|v| v.checked_mul(width as i128))
        .and_then(|v| isize::try_from(v).ok())
        .ok_or_else(error)?;
    origin.checked_add_signed(first).ok_or_else(error)?;
    origin.checked_add_signed(end).ok_or_else(error)?;
    Ok(())
}

/// Validated, read-only rank-one parameter. No parameter tensor is copied;
/// real X/Y stay untouched until all parameter reads and computations finish.
struct ChannelReader<T> {
    origin: *const T,
    stride: isize,
    count: usize,
}
impl<T: NumericElem> ChannelReader<T> {
    fn new(name: &str, view: &TensorView, channels: usize) -> Result<Self> {
        validate_f32_widen_input(OP, view).map_err(|e| {
            fail(format!(
                "invalid {name}: {e}; bind a valid Host channel vector"
            ))
        })?;
        if view.dtype != T::DTYPE || view.shape != [channels] {
            return Err(fail(format!(
                "{name} must be {:?} shape [{channels}], got {:?} {:?}; bind the matching channel vector",
                T::DTYPE,
                view.dtype,
                view.shape
            )));
        }
        addresses(
            name,
            view.data.0 as usize,
            view.byte_offset,
            view.shape,
            view.strides,
            size_of::<T>(),
        )?;
        let stride = if channels == 1 {
            0
        } else {
            isize::try_from(view.strides[0]).map_err(|_| {
                fail(format!(
                    "{name} stride {} is not host-addressable; materialize a valid channel vector",
                    view.strides[0]
                ))
            })?
        };
        Ok(Self {
            origin: view.data_ptr::<T>(),
            stride,
            count: channels,
        })
    }
    fn get(&self, channel: usize) -> T::Acc {
        debug_assert!(channel < self.count);
        // SAFETY: construction proves Host access, dtype, complete rank-one
        // geometry and all offset/address arithmetic; callers index within C.
        // EP owns backing bounds/lifetime; arbitrary byte alignment is allowed.
        unsafe {
            self.origin
                .offset(channel as isize * self.stride)
                .read_unaligned()
        }
        .to_acc()
    }
}

impl BatchNormalization {
    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let metadata = inputs
            .iter()
            .map(|x| TensorMetadata::new(x.dtype, x.shape, !x.is_absent()))
            .collect::<smallvec::SmallVec<[_; 5]>>();
        let facts = outputs
            .iter()
            .map(OutputMetadata::from_view)
            .collect::<smallvec::SmallVec<[_; 1]>>();
        let geometry = Geometry::new(&metadata, &facts)?;
        validate_f32_widen_input(OP, &inputs[0])
            .map_err(|e| fail(format!("invalid X: {e}; bind valid floating Host input")))?;
        validate_host_output(OP, &outputs[0])?;
        for (name, base, offset, shape, strides, width) in [
            (
                "X",
                inputs[0].data.0 as usize,
                inputs[0].byte_offset,
                inputs[0].shape,
                inputs[0].strides,
                inputs[0].dtype.byte_size(),
            ),
            (
                "Y",
                outputs[0].data.0 as usize,
                outputs[0].byte_offset,
                outputs[0].shape,
                outputs[0].strides,
                outputs[0].dtype.byte_size(),
            ),
        ] {
            addresses(name, base, offset, shape, strides, width)?;
        }
        dispatch_float!(inputs[0].dtype,OP,T=>self.typed::<T>(inputs,&mut outputs[0],&geometry,workspace))
    }
    fn typed<T: NumericElem + Default>(
        &self,
        inputs: &[TensorView],
        output: &mut TensorMut,
        geometry: &Geometry,
        workspace: Option<WorkspaceView>,
    ) -> Result<()>
    where
        T::Acc: NormAcc,
    {
        let scale = ChannelReader::<T>::new("scale", &inputs[1], geometry.channels)?;
        let bias = ChannelReader::<T>::new("B", &inputs[2], geometry.channels)?;
        let mean = ChannelReader::<T>::new("input_mean", &inputs[3], geometry.channels)?;
        let variance = ChannelReader::<T>::new("input_var", &inputs[4], geometry.channels)?;
        let plan = geometry.workspace::<T>()?;
        let epsilon = <T::Acc as NormAcc>::from_f32(self.epsilon);
        // SAFETY: executor lends exclusive live Host scratch disjoint from all
        // tensors. The helper validates the complete binding before initializing.
        unsafe {
            plan.with_buffers(workspace, |[snapshot]| {
                to_dense_into(OP, &inputs[0], snapshot)?;
                for (instance_channel, slice) in
                    snapshot.chunks_exact_mut(geometry.spatial).enumerate()
                {
                    let channel = instance_channel % geometry.channels;
                    let scale = scale.get(channel);
                    let bias = bias.get(channel);
                    let mean = mean.get(channel);
                    let variance = variance.get(channel);
                    for value in slice {
                        *value = T::from_acc(
                            value
                                .to_acc()
                                .c_sub(mean)
                                .c_div(variance.c_add(epsilon).sqrt())
                                .c_mul(scale)
                                .c_add(bias),
                        );
                    }
                }
                // Every source and parameter read finishes before real Y is changed.
                write_dense(output, snapshot)
            })
        }
    }
}
impl Kernel for BatchNormalization {
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        let geometry = Geometry::new(inputs, outputs)?;
        let requirement =
            dispatch_float!(inputs[0].dtype,OP,T=>Ok(geometry.workspace::<T>()?.requirement()))?;
        Ok(WorkspacePreparation::Ready(requirement))
    }
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }
    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.run(inputs, outputs, workspace)
    }
    fn supports_strided_input(&self, _index: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests;
