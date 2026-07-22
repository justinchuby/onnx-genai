//! Native (pure-Rust nxrt) implementation of the backend-neutral
//! [`ComponentSession`](onnx_genai_metadata::ComponentSession) interface.
//!
//! [`NativeComponentSession`] wraps an [`InferenceSession`] so that
//! `PipelineEngine` can construct and query a pipeline component through the
//! native backend using the exact same seam it uses for ONNX Runtime. This is
//! what lets pipeline construction become backend-neutral: every declared
//! component is loaded through this adapter when the native backend is selected,
//! and its graph I/O metadata and named-tensor execution are exposed without any
//! ORT type appearing on the engine's construction path.
//!
//! Note this covers GAP 1 (the component-session seam) only. Wiring these
//! backend-neutral sessions into the pipeline decode loop — which still routes
//! state through ORT `Value`/`Session` — is the remaining native work (GAPs
//! 2/3); see [`crate::pipeline`].

use crate::native_decode::NativeDecodeDevice;
use onnx_genai_metadata::{
    ComponentDataType, ComponentError, ComponentIo, ComponentSession, ComponentTensor,
};
use onnx_runtime_ir::{DataType as IrDataType, Dim};
use onnx_runtime_session::{DevicePreference, InferenceSession, IoMeta, Tensor};
use std::path::Path;

const BACKEND: &str = "native";

fn ir_dtype_to_component(dtype: IrDataType) -> Result<ComponentDataType, ComponentError> {
    Ok(match dtype {
        IrDataType::Float32 => ComponentDataType::Float32,
        IrDataType::Float16 => ComponentDataType::Float16,
        IrDataType::BFloat16 => ComponentDataType::BFloat16,
        IrDataType::Float8E4M3FN => ComponentDataType::Float8E4M3,
        IrDataType::Float8E5M2 => ComponentDataType::Float8E5M2,
        IrDataType::Int8 => ComponentDataType::Int8,
        IrDataType::Int16 => ComponentDataType::Int16,
        IrDataType::Int32 => ComponentDataType::Int32,
        IrDataType::Int64 => ComponentDataType::Int64,
        IrDataType::Uint8 => ComponentDataType::Uint8,
        IrDataType::Uint16 => ComponentDataType::Uint16,
        IrDataType::Uint32 => ComponentDataType::Uint32,
        IrDataType::Uint64 => ComponentDataType::Uint64,
        IrDataType::Bool => ComponentDataType::Bool,
        other => {
            return Err(ComponentError::UnsupportedDataType {
                observed: format!("{other:?}"),
            });
        }
    })
}

fn component_dtype_to_ir(dtype: ComponentDataType) -> IrDataType {
    match dtype {
        ComponentDataType::Float32 => IrDataType::Float32,
        ComponentDataType::Float16 => IrDataType::Float16,
        ComponentDataType::BFloat16 => IrDataType::BFloat16,
        ComponentDataType::Float8E4M3 => IrDataType::Float8E4M3FN,
        ComponentDataType::Float8E5M2 => IrDataType::Float8E5M2,
        ComponentDataType::Int8 => IrDataType::Int8,
        ComponentDataType::Int16 => IrDataType::Int16,
        ComponentDataType::Int32 => IrDataType::Int32,
        ComponentDataType::Int64 => IrDataType::Int64,
        ComponentDataType::Uint8 => IrDataType::Uint8,
        ComponentDataType::Uint16 => IrDataType::Uint16,
        ComponentDataType::Uint32 => IrDataType::Uint32,
        ComponentDataType::Uint64 => IrDataType::Uint64,
        ComponentDataType::Bool => IrDataType::Bool,
    }
}

/// Declared shape as neutral `i64` axes: symbolic/dynamic dims become `-1`
/// (the ORT convention shared by [`ComponentIo`]).
fn declared_shape(shape: &[Dim]) -> Vec<i64> {
    shape
        .iter()
        .map(|dim| match dim {
            Dim::Static(size) => *size as i64,
            Dim::Symbolic(_) => -1,
        })
        .collect()
}

fn component_io(meta: &IoMeta) -> Result<ComponentIo, ComponentError> {
    Ok(ComponentIo {
        name: meta.name.clone(),
        dtype: ir_dtype_to_component(meta.dtype)?,
        shape: declared_shape(&meta.shape),
    })
}

/// A pipeline component backed by the native nxrt [`InferenceSession`].
pub struct NativeComponentSession {
    session: InferenceSession,
    inputs: Vec<ComponentIo>,
    outputs: Vec<ComponentIo>,
}

impl NativeComponentSession {
    /// Wrap an already-built native [`InferenceSession`] as a neutral component.
    pub fn new(session: InferenceSession) -> Result<Self, ComponentError> {
        let inputs = session
            .inputs()
            .iter()
            .map(component_io)
            .collect::<Result<_, _>>()?;
        let outputs = session
            .outputs()
            .iter()
            .map(component_io)
            .collect::<Result<_, _>>()?;
        Ok(Self {
            session,
            inputs,
            outputs,
        })
    }

    /// Load an ONNX model on the requested native device as a neutral component.
    pub fn load(path: &Path, device: NativeDecodeDevice) -> anyhow::Result<Self> {
        let preference = match device {
            NativeDecodeDevice::Cpu => DevicePreference::Cpu,
            NativeDecodeDevice::Cuda { index } => DevicePreference::Gpu { index },
        };
        let session = InferenceSession::builder()
            .model(path)
            .device(preference)
            .build()
            .map_err(|err| {
                anyhow::anyhow!(
                    "failed to load pipeline component '{}' on the native backend: {err}",
                    path.display()
                )
            })?;
        Self::new(session).map_err(anyhow::Error::from)
    }
}

fn to_native_tensor(
    component: &str,
    name: &str,
    tensor: &ComponentTensor,
) -> Result<Tensor, ComponentError> {
    let dtype = component_dtype_to_ir(tensor.dtype());
    let shape: Vec<usize> = tensor.shape().iter().map(|&dim| dim as usize).collect();
    Tensor::from_raw(dtype, shape, tensor.as_bytes()).map_err(|err| ComponentError::Backend {
        component: component.to_string(),
        backend: BACKEND,
        detail: format!("failed to build native input tensor '{name}': {err}"),
    })
}

fn from_native_tensor(component: &str, tensor: &Tensor) -> Result<ComponentTensor, ComponentError> {
    let dtype = ir_dtype_to_component(tensor.dtype)?;
    let shape: Vec<i64> = tensor.shape.iter().map(|&dim| dim as i64).collect();
    ComponentTensor::from_raw(dtype, shape, tensor.as_bytes().to_vec()).map_err(|err| {
        ComponentError::Backend {
            component: component.to_string(),
            backend: BACKEND,
            detail: format!("native output tensor is malformed: {err}"),
        }
    })
}

impl ComponentSession for NativeComponentSession {
    fn inputs(&self) -> &[ComponentIo] {
        &self.inputs
    }

    fn outputs(&self) -> &[ComponentIo] {
        &self.outputs
    }

    fn run(
        &mut self,
        inputs: &[(&str, &ComponentTensor)],
    ) -> Result<Vec<(String, ComponentTensor)>, ComponentError> {
        let component = self
            .outputs
            .first()
            .map(|io| io.name.as_str())
            .unwrap_or("<native-component>")
            .to_string();
        let tensors: Vec<(&str, Tensor)> = inputs
            .iter()
            .map(|(name, tensor)| Ok((*name, to_native_tensor(&component, name, tensor)?)))
            .collect::<Result<_, ComponentError>>()?;
        let borrowed: Vec<(&str, &Tensor)> = tensors
            .iter()
            .map(|(name, tensor)| (*name, tensor))
            .collect();
        let output_names: Vec<String> = self.outputs.iter().map(|io| io.name.clone()).collect();
        let outputs = self
            .session
            .run(&borrowed)
            .map_err(|err| ComponentError::Backend {
                component: component.clone(),
                backend: BACKEND,
                detail: err.to_string(),
            })?;
        output_names
            .into_iter()
            .zip(outputs.iter())
            .map(|(name, tensor)| Ok((name, from_native_tensor(&component, tensor)?)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{DataType as IrDt, Dim, Graph, Node, NodeId};

    /// A minimal `Add(x, y) -> z` graph over static rank-2 f32 tensors.
    fn tiny_add_session() -> InferenceSession {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 11);
        let shape = || -> Vec<Dim> { vec![Dim::Static(2), Dim::Static(3)] };
        let x = graph.create_named_value("x", IrDt::Float32, shape());
        let y = graph.create_named_value("y", IrDt::Float32, shape());
        let z = graph.create_named_value("z", IrDt::Float32, shape());
        graph.add_input(x);
        graph.add_input(y);
        graph.insert_node(Node::new(NodeId(0), "Add", vec![Some(x), Some(y)], vec![z]));
        graph.add_output(z);
        InferenceSession::from_graph(graph).expect("build tiny add session")
    }

    fn f32_tensor(shape: Vec<i64>, values: &[f32]) -> ComponentTensor {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        ComponentTensor::from_raw(ComponentDataType::Float32, shape, bytes).expect("tensor")
    }

    fn as_f32(tensor: &ComponentTensor) -> Vec<f32> {
        tensor
            .as_bytes()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    #[test]
    fn exposes_graph_io_metadata() {
        let component = NativeComponentSession::new(tiny_add_session()).expect("component");
        assert_eq!(component.input_names(), vec!["x", "y"]);
        assert_eq!(component.output_names(), vec!["z"]);
        for io in component.inputs() {
            assert_eq!(io.dtype, ComponentDataType::Float32);
            assert_eq!(io.rank(), 2);
            assert_eq!(io.shape, vec![2, 3]);
        }
        assert_eq!(component.outputs()[0].name, "z");
        assert_eq!(component.outputs()[0].dtype, ComponentDataType::Float32);
    }

    #[test]
    fn named_tensor_run_round_trip() {
        let mut component = NativeComponentSession::new(tiny_add_session()).expect("component");
        let x = f32_tensor(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = f32_tensor(vec![2, 3], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let outputs = component
            .run(&[("x", &x), ("y", &y)])
            .expect("component run");
        assert_eq!(outputs.len(), 1);
        let (name, tensor) = &outputs[0];
        assert_eq!(name, "z");
        assert_eq!(tensor.dtype(), ComponentDataType::Float32);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(as_f32(tensor), vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
    }
}
