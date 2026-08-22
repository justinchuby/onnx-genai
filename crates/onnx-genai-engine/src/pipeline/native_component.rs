//! Native (pure-Rust) execution backend for the universal workflow interpreter.
//!
//! The interpreter in [`super::workflow`] is backend-agnostic: it threads
//! `onnx_genai_ort::Value` as its tensor currency and asks a backend to run
//! each declared ONNX component. This module supplies the native answer — a
//! set of [`onnx_runtime_session::InferenceSession`]s, one per declared
//! component graph — so `EngineDecodeBackend::Native` drives the *same*
//! interpreter (loops, branches, emits, loop-carried and shared state,
//! adapters, checkpoints) without any second workflow implementation.
//!
//! # Value seam, not a value copy-through
//!
//! `Value` is a device-capable *handle*, so it is the neutral currency the
//! interpreter already carries; this executor abstracts *execution*, not the
//! value type. On CPU a component boundary bridges `Value` ⇄
//! [`onnx_runtime_session::Tensor`] through the raw little-endian element
//! bytes both already agree on. It never routes a tensor through the
//! host-resident `ComponentTensor` seam, and it holds every session for the
//! life of the engine, so a recurring component edge (a decoder re-invoked each
//! loop iteration) reuses one native session rather than reloading or
//! re-serializing — observable through [`NativeComponentSet::run_count`].
//!
//! The CPU path is faithful and fail-closed on dtype; the CUDA device-resident
//! zero-copy handoff (native `Tensor::device_ptr` ⇆ `Value::from_external_memory`)
//! is the staged follow-up boundary A in
//! `docs/architecture/NATIVE_WORKFLOW_BACKEND.md`.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use onnx_genai_ort::{DataType, Value};
use onnx_runtime_ir::DataType as IrDataType;
use onnx_runtime_session::{DevicePreference, InferenceSession, Tensor};

use crate::native_decode::NativeDecodeDevice;

/// Faithful ORT→IR element-type mapping for a value entering a native graph.
///
/// Total over the ORT dtype vocabulary: every `Value` the interpreter can hold
/// has a native equivalent, so this never fails.
fn ir_dtype_from_ort(dtype: DataType) -> IrDataType {
    match dtype {
        DataType::Float32 => IrDataType::Float32,
        DataType::Float16 => IrDataType::Float16,
        DataType::BFloat16 => IrDataType::BFloat16,
        // ORT's single-precision f8 spellings are the ONNX FN/(non-UZ) variants.
        DataType::Float8E4M3 => IrDataType::Float8E4M3FN,
        DataType::Float8E5M2 => IrDataType::Float8E5M2,
        DataType::Int8 => IrDataType::Int8,
        DataType::Int16 => IrDataType::Int16,
        DataType::Int32 => IrDataType::Int32,
        DataType::Int64 => IrDataType::Int64,
        DataType::Uint8 => IrDataType::Uint8,
        DataType::Uint16 => IrDataType::Uint16,
        DataType::Uint32 => IrDataType::Uint32,
        DataType::Uint64 => IrDataType::Uint64,
        DataType::Bool => IrDataType::Bool,
    }
}

/// Faithful IR→ORT element-type mapping for a native output re-entering the
/// interpreter's value pool.
///
/// Partial: the native runtime can carry element types (e.g. `Float64`,
/// `String`, the UZ float8 variants, sub-byte ints) that the workflow value
/// currency does not. Rather than coerce silently (Rule 4), an out-of-vocabulary
/// output fails with an actionable diagnostic naming the component, port, and
/// observed type.
fn ort_dtype_from_ir(component: &str, port: &str, dtype: IrDataType) -> anyhow::Result<DataType> {
    Ok(match dtype {
        IrDataType::Float32 => DataType::Float32,
        IrDataType::Float16 => DataType::Float16,
        IrDataType::BFloat16 => DataType::BFloat16,
        IrDataType::Float8E4M3FN => DataType::Float8E4M3,
        IrDataType::Float8E5M2 => DataType::Float8E5M2,
        IrDataType::Int8 => DataType::Int8,
        IrDataType::Int16 => DataType::Int16,
        IrDataType::Int32 => DataType::Int32,
        IrDataType::Int64 => DataType::Int64,
        IrDataType::Uint8 => DataType::Uint8,
        IrDataType::Uint16 => DataType::Uint16,
        IrDataType::Uint32 => DataType::Uint32,
        IrDataType::Uint64 => DataType::Uint64,
        IrDataType::Bool => DataType::Bool,
        other => anyhow::bail!(
            "native workflow component '{component}' output '{port}' has element type {other:?}, \
             which the workflow value currency does not carry; the pipeline value pool supports \
             float{{32,16}}, bfloat16, float8{{e4m3fn,e5m2}}, int{{8,16,32,64}}, uint{{8,16,32,64}}, \
             and bool. Run this component on the ORT backend (decode_backend = Ort) or narrow the \
             graph's output dtype."
        ),
    })
}

/// A concrete (fully static) tensor shape as the native runtime expects it.
///
/// The interpreter only ever holds materialized tensors, so a dynamic
/// (negative) axis here is an invariant violation, reported as such rather than
/// silently truncated to zero.
fn native_shape(component: &str, port: &str, shape: &[i64]) -> anyhow::Result<Vec<usize>> {
    shape
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| {
                anyhow::anyhow!(
                    "native workflow component '{component}' input '{port}' has a dynamic or \
                     negative axis in shape {shape:?}; a tensor crossing the component boundary \
                     must be fully static"
                )
            })
        })
        .collect()
}

/// Bridge a workflow value into a native input tensor.
///
/// A host-resident value is bridged through its little-endian element bytes —
/// the same on-wire form the native `Tensor` stores, and for a CPU device there
/// is no round-trip (both sides are host memory). A device-resident value (a
/// CUDA `Value`) is **not** silently copied to the host: the device-resident
/// `Value → Tensor` bridge is not wired yet, so this fails closed with an
/// actionable diagnostic rather than reading device memory as host bytes. The
/// native runtime has the device-tensor APIs to build that bridge
/// (`Tensor::from_borrowed_parts_with_guard`), so this is an explicit
/// not-yet-wired boundary, not a permanent CPU-only limit.
fn value_to_native_tensor(component: &str, port: &str, value: &Value) -> anyhow::Result<Tensor> {
    if !value.is_host_resident().with_context(|| {
        format!(
            "native workflow component '{component}' could not classify input '{port}' residency"
        )
    })? {
        anyhow::bail!(
            "native workflow component '{component}' received a device-resident input for port \
             '{port}' (device {}); the device-resident Value->Tensor bridge is not wired yet, so \
             the native backend fails closed rather than silently host-copying. Run this workflow \
             on the CPU native device or the ORT backend, or wire the device bridge \
             (Tensor::from_borrowed_parts_with_guard).",
            value.device_id().unwrap_or(-1)
        );
    }
    let dtype = ir_dtype_from_ort(value.dtype());
    let shape = native_shape(component, port, value.shape())?;
    let bytes = value.to_raw_bytes().with_context(|| {
        format!("native workflow component '{component}' could not read input '{port}' bytes")
    })?;
    Tensor::from_raw(dtype, shape, &bytes).with_context(|| {
        format!(
            "native workflow component '{component}' could not build input tensor for port \
             '{port}'"
        )
    })
}

/// Bridge a native output tensor back into a workflow value.
///
/// A host-accessible tensor is bridged through its element bytes. A
/// device-resident output (a native CUDA tensor) is **not** read as host bytes
/// (`Tensor::as_bytes` is documented host-only and would be unsound); the
/// device-resident `Tensor → Value` bridge is not wired yet, so this fails
/// closed. The building blocks exist (`Tensor::device_ptr` +
/// `Value::from_external_memory`), so this is an explicit not-yet-wired
/// boundary, not a permanent CPU-only limit.
fn native_tensor_to_value(component: &str, port: &str, tensor: &Tensor) -> anyhow::Result<Value> {
    if !tensor.device().is_host_accessible() {
        anyhow::bail!(
            "native workflow component '{component}' produced a device-resident output '{port}' \
             on device {:?}; the device-resident Tensor->Value bridge is not wired yet, so the \
             native backend fails closed rather than reading device memory as host bytes. Run on \
             the CPU native device or the ORT backend, or wire the device bridge \
             (Value::from_external_memory).",
            tensor.device()
        );
    }
    let dtype = ort_dtype_from_ir(component, port, tensor.dtype)?;
    let shape: Vec<i64> = tensor.shape.iter().map(|&dim| dim as i64).collect();
    Value::from_raw_bytes(tensor.as_bytes().to_vec(), &shape, dtype).with_context(|| {
        format!(
            "native workflow component '{component}' could not publish output '{port}' into the \
             value pool"
        )
    })
}

/// One native component graph plus the output port names it publishes.
struct NativeComponent {
    session: InferenceSession,
    output_names: Vec<String>,
}

/// Every declared ONNX component of a workflow package, loaded as a native
/// [`InferenceSession`] bound to the engine's resolved native device/EP. Held
/// for the life of the [`PipelineEngine`], so a component invoked once per loop
/// iteration reuses one session.
pub(crate) struct NativeComponentSet {
    components: HashMap<String, NativeComponent>,
    device_label: String,
    run_count: u64,
}

impl NativeComponentSet {
    /// Load a native session for every component model file in the package,
    /// bound to the resolved native `device` (and its execution provider) —
    /// **not** ORT and **not** an auto-detected CPU EP.
    ///
    /// These are the same on-disk graphs the ORT `PipelineModels` inspects for
    /// I/O, so the native and ORT backends execute byte-identical component
    /// logic; only the executor differs.
    pub(crate) fn load(
        model_paths: &BTreeMap<String, PathBuf>,
        device: &NativeDecodeDevice,
    ) -> anyhow::Result<Self> {
        let mut components = HashMap::with_capacity(model_paths.len());
        for (component, path) in model_paths {
            let session = load_native_component(component, path, device)?;
            let output_names = session.outputs().iter().map(|io| io.name.clone()).collect();
            components.insert(
                component.clone(),
                NativeComponent {
                    session,
                    output_names,
                },
            );
        }
        Ok(Self {
            components,
            device_label: native_device_label(device),
            run_count: 0,
        })
    }

    /// Diagnostic label of the native device these sessions run on, used by
    /// `execution_provider_status` so a Native engine reports its real device.
    pub(crate) fn device_label(&self) -> &str {
        &self.device_label
    }

    /// Number of native component invocations performed so far.
    ///
    /// Tests assert this is non-zero after a native run (proving the native
    /// sessions, not an ORT fallback, executed the workflow).
    pub(crate) fn run_count(&self) -> u64 {
        self.run_count
    }

    /// Run one declared ONNX component over resolved input values, returning
    /// every declared graph output as a workflow value — the same
    /// `(name, Value)` contract the ORT plain-run path returns, so the caller's
    /// output routing is backend-independent.
    pub(crate) fn run_component(
        &mut self,
        component: &str,
        resolved: &[(&str, &Value)],
        _selected_outputs: &BTreeMap<String, String>,
    ) -> anyhow::Result<Vec<(String, Value)>> {
        let native = self.components.get_mut(component).with_context(|| {
            format!(
                "native workflow backend has no session for ONNX component '{component}'; it was \
                 not present in the package's component model files"
            )
        })?;

        let inputs = resolved
            .iter()
            .map(|(port, value)| {
                Ok((
                    (*port).to_string(),
                    value_to_native_tensor(component, port, value)?,
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let input_refs: Vec<(&str, &Tensor)> = inputs
            .iter()
            .map(|(port, tensor)| (port.as_str(), tensor))
            .collect();

        let outputs = native
            .session
            .run(&input_refs)
            .with_context(|| format!("native workflow component '{component}' failed to run"))?;
        self.run_count += 1;

        anyhow::ensure!(
            outputs.len() == native.output_names.len(),
            "native workflow component '{component}' returned {} tensors but declares {} outputs",
            outputs.len(),
            native.output_names.len(),
        );

        native
            .output_names
            .iter()
            .zip(outputs.iter())
            .map(|(name, tensor)| {
                Ok((
                    name.clone(),
                    native_tensor_to_value(component, name, tensor)?,
                ))
            })
            .collect()
    }
}

/// Human-readable label of a resolved native device, for EP-status reporting.
fn native_device_label(device: &NativeDecodeDevice) -> String {
    match device {
        NativeDecodeDevice::Cpu => "native-cpu".to_string(),
        NativeDecodeDevice::Cuda { index } => format!("native-cuda:{}", index.unwrap_or(0)),
        NativeDecodeDevice::Plugin { provider_name, .. } => {
            format!("native-plugin:{provider_name}")
        }
    }
}

/// Build a native `InferenceSession` for one component bound to `device`'s
/// **explicit** execution provider — never ORT and never an auto-detected CPU
/// EP (which would silently ignore a requested CUDA device). Mirrors the
/// device→EP selection the native decode path uses in `native_decode/load.rs`.
fn load_native_component(
    component: &str,
    path: &Path,
    device: &NativeDecodeDevice,
) -> anyhow::Result<InferenceSession> {
    let mut builder = InferenceSession::builder().model(path);
    match device {
        NativeDecodeDevice::Cpu => {
            builder = builder
                .device(DevicePreference::Cpu)
                .execution_provider(Arc::new(onnx_runtime_ep_cpu::CpuExecutionProvider::new()));
        }
        #[cfg(feature = "native-cuda")]
        NativeDecodeDevice::Cuda { index } => {
            let ordinal = index.unwrap_or(0);
            let ep = onnx_runtime_ep_cuda::CudaExecutionProvider::initialized(ordinal)
                .with_context(|| {
                    format!(
                        "initialize native CUDA execution provider (device {ordinal}) for \
                         workflow component '{component}'"
                    )
                })?;
            builder = builder
                .device(DevicePreference::Gpu {
                    index: Some(ordinal),
                })
                .execution_provider(Arc::new(ep));
        }
        #[cfg(not(feature = "native-cuda"))]
        NativeDecodeDevice::Cuda { .. } => {
            anyhow::bail!(
                "native workflow component '{component}' requested a CUDA device, but this build \
                 lacks the `native-cuda` feature. Rebuild with --features native-cuda, or select \
                 the CPU native device / the ORT backend."
            );
        }
        NativeDecodeDevice::Plugin { .. } => {
            anyhow::bail!(
                "native workflow component '{component}' requested a plugin execution provider, \
                 which the native workflow backend does not support yet; use the CPU or CUDA \
                 native device, or the ORT backend."
            );
        }
    }
    builder.build().with_context(|| {
        format!(
            "native workflow backend failed to load component '{component}' from '{}'. If this \
             component uses operators the native backend does not implement, run the workflow on \
             the ORT backend (decode_backend = Ort / ONNX_GENAI_BACKEND=ort).",
            path.display()
        )
    })
}
