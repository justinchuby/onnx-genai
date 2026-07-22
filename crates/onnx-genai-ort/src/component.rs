//! ONNX Runtime implementation of the backend-neutral
//! [`ComponentSession`](onnx_genai_metadata::ComponentSession) interface.
//!
//! [`OrtComponentSession`] wraps an existing ORT [`Session`] so that
//! `PipelineEngine` can construct and drive a pipeline component through the
//! same neutral seam it uses for the native backend. It is a behavior-preserving
//! adapter: [`OrtComponentSession::run`] forwards to [`Session::run`], so a
//! pipeline routed through this wrapper produces identical results to one that
//! calls the concrete session directly.

use crate::error::OrtError;
use crate::session::TensorInfo;
use crate::{DataType, Environment, Session, SessionOptions, Value};
use onnx_genai_metadata::{
    ComponentDataType, ComponentError, ComponentIo, ComponentSession, ComponentTensor,
};
use std::path::Path;

const BACKEND: &str = "ort";

impl From<DataType> for ComponentDataType {
    fn from(dtype: DataType) -> Self {
        match dtype {
            DataType::Float32 => ComponentDataType::Float32,
            DataType::Float16 => ComponentDataType::Float16,
            DataType::BFloat16 => ComponentDataType::BFloat16,
            DataType::Float8E4M3 => ComponentDataType::Float8E4M3,
            DataType::Float8E5M2 => ComponentDataType::Float8E5M2,
            DataType::Int8 => ComponentDataType::Int8,
            DataType::Int16 => ComponentDataType::Int16,
            DataType::Int32 => ComponentDataType::Int32,
            DataType::Int64 => ComponentDataType::Int64,
            DataType::Uint8 => ComponentDataType::Uint8,
            DataType::Uint16 => ComponentDataType::Uint16,
            DataType::Uint32 => ComponentDataType::Uint32,
            DataType::Uint64 => ComponentDataType::Uint64,
            DataType::Bool => ComponentDataType::Bool,
        }
    }
}

impl From<ComponentDataType> for DataType {
    fn from(dtype: ComponentDataType) -> Self {
        match dtype {
            ComponentDataType::Float32 => DataType::Float32,
            ComponentDataType::Float16 => DataType::Float16,
            ComponentDataType::BFloat16 => DataType::BFloat16,
            ComponentDataType::Float8E4M3 => DataType::Float8E4M3,
            ComponentDataType::Float8E5M2 => DataType::Float8E5M2,
            ComponentDataType::Int8 => DataType::Int8,
            ComponentDataType::Int16 => DataType::Int16,
            ComponentDataType::Int32 => DataType::Int32,
            ComponentDataType::Int64 => DataType::Int64,
            ComponentDataType::Uint8 => DataType::Uint8,
            ComponentDataType::Uint16 => DataType::Uint16,
            ComponentDataType::Uint32 => DataType::Uint32,
            ComponentDataType::Uint64 => DataType::Uint64,
            ComponentDataType::Bool => DataType::Bool,
        }
    }
}

fn component_io(info: &TensorInfo) -> ComponentIo {
    ComponentIo {
        name: info.name.clone(),
        dtype: info.dtype.into(),
        shape: info.shape.clone(),
    }
}

/// A pipeline component backed by an ONNX Runtime [`Session`].
pub struct OrtComponentSession {
    session: Session,
    inputs: Vec<ComponentIo>,
    outputs: Vec<ComponentIo>,
}

impl OrtComponentSession {
    /// Wrap an already-loaded ORT [`Session`] as a backend-neutral component.
    pub fn new(session: Session) -> Self {
        let inputs = session.inputs().iter().map(component_io).collect();
        let outputs = session.outputs().iter().map(component_io).collect();
        Self {
            session,
            inputs,
            outputs,
        }
    }

    /// Load an ONNX model as a backend-neutral component.
    pub fn load(env: &Environment, path: &Path, options: SessionOptions) -> crate::Result<Self> {
        Ok(Self::new(Session::new(env, path, options)?))
    }

    /// Borrow the underlying ORT session (execution paths that need the concrete
    /// ORT surface — device bindings, captured graphs — reach it through here).
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Consume the wrapper and recover the underlying ORT session.
    pub fn into_session(self) -> Session {
        self.session
    }
}

fn to_value(component: &str, tensor: &ComponentTensor) -> Result<Value, ComponentError> {
    let dtype: DataType = tensor.dtype().into();
    Value::from_raw_bytes(tensor.as_bytes().to_vec(), tensor.shape(), dtype).map_err(|err| {
        ComponentError::Backend {
            component: component.to_string(),
            backend: BACKEND,
            detail: format!("failed to build ORT input tensor: {err}"),
        }
    })
}

fn from_value(component: &str, value: &Value) -> Result<ComponentTensor, ComponentError> {
    let dtype: ComponentDataType = value.dtype().into();
    let bytes = value
        .to_raw_bytes()
        .map_err(|err| ComponentError::Backend {
            component: component.to_string(),
            backend: BACKEND,
            detail: format!("failed to read ORT output tensor: {err}"),
        })?;
    ComponentTensor::from_raw(dtype, value.shape().to_vec(), bytes)
}

impl ComponentSession for OrtComponentSession {
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
        // The component name is only available for diagnostics via the first
        // declared output/input; fall back to a stable label otherwise.
        let component = self
            .outputs
            .first()
            .map(|io| io.name.as_str())
            .unwrap_or("<ort-component>");
        let values: Vec<(&str, Value)> = inputs
            .iter()
            .map(|(name, tensor)| Ok((*name, to_value(component, tensor)?)))
            .collect::<Result<_, ComponentError>>()?;
        let borrowed: Vec<(&str, &Value)> =
            values.iter().map(|(name, value)| (*name, value)).collect();
        let outputs =
            self.session
                .run(&borrowed)
                .map_err(|err: OrtError| ComponentError::Backend {
                    component: component.to_string(),
                    backend: BACKEND,
                    detail: err.to_string(),
                })?;
        self.session
            .output_names()
            .iter()
            .zip(outputs.iter())
            .map(|(name, value)| Ok((name.clone(), from_value(component, value)?)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::OnceLock;

    fn tiny_whisper_encoder_textproto() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-whisper/encoder.onnx.textproto")
    }

    fn test_environment() -> &'static Environment {
        static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
        ENVIRONMENT.get_or_init(|| Environment::new("ort-component-test").expect("env"))
    }

    #[test]
    fn exposes_graph_io_metadata() {
        let path = tiny_whisper_encoder_textproto();
        if !path.exists() {
            eprintln!("exposes_graph_io_metadata: fixture absent, skipping");
            return;
        }
        let component = OrtComponentSession::load(
            test_environment(),
            &path,
            SessionOptions::default().with_intra_op_threads(1),
        )
        .expect("component loaded");

        assert_eq!(component.input_names(), vec!["input_features"]);
        assert_eq!(component.output_names(), vec!["encoder_hidden_states"]);

        let input = &component.inputs()[0];
        assert_eq!(input.name, "input_features");
        assert_eq!(input.dtype, ComponentDataType::Float32);
        assert_eq!(input.rank(), input.shape.len());

        let output = &component.outputs()[0];
        assert_eq!(output.name, "encoder_hidden_states");
        assert_eq!(output.dtype, ComponentDataType::Float32);
    }

    #[test]
    fn named_tensor_run_round_trip_matches_session() {
        let path = tiny_whisper_encoder_textproto();
        if !path.exists() {
            eprintln!("named_tensor_run_round_trip_matches_session: fixture absent, skipping");
            return;
        }
        // Reference: run the raw ORT session directly.
        let session = Session::new(
            test_environment(),
            &path,
            SessionOptions::default().with_intra_op_threads(1),
        )
        .expect("session");
        let features = Value::from_slice_f32(&vec![0.25f32; 80 * 8], &[1, 80, 8]).expect("input");
        let reference = session
            .run(&[("input_features", &features)])
            .expect("reference run");
        let reference_bytes = reference[0].to_raw_bytes().expect("reference bytes");

        // Through the backend-neutral component seam.
        let mut component = OrtComponentSession::load(
            test_environment(),
            &path,
            SessionOptions::default().with_intra_op_threads(1),
        )
        .expect("component");
        let input_bytes: Vec<u8> = vec![0.25f32; 80 * 8]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let input =
            ComponentTensor::from_raw(ComponentDataType::Float32, vec![1, 80, 8], input_bytes)
                .expect("component input");
        let outputs = component
            .run(&[("input_features", &input)])
            .expect("component run");

        assert_eq!(outputs.len(), 1);
        let (name, tensor) = &outputs[0];
        assert_eq!(name, "encoder_hidden_states");
        assert_eq!(tensor.dtype(), ComponentDataType::Float32);
        assert_eq!(tensor.shape(), reference[0].shape());
        assert_eq!(tensor.as_bytes(), reference_bytes.as_slice());
    }
}
