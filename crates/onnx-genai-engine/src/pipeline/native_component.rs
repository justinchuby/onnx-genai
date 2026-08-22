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
//! The CPU path bridges each boundary through host element bytes. The CUDA path
//! keeps tensors device-resident end-to-end: a device-resident input `Value` is
//! bound zero-copy into the native session through an external-memory
//! [`onnx_runtime_session::DeviceIoBinding`], and each device output
//! [`onnx_runtime_session::Tensor`] is wrapped in a `Value` that *owns* it
//! (`Value::from_external_memory_with_owner`), so a recurring or loop-carried
//! edge hands its device buffer to the next component with no host round-trip.
//! Host staging happens only at genuine host inputs (e.g. token ids) and
//! explicit host-materialization boundaries.

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

/// Bridge a workflow value into a native input tensor on the CPU path.
///
/// A host-resident value is bridged through its little-endian element bytes —
/// the same on-wire form the native `Tensor` stores, and for a CPU device there
/// is no round-trip (both sides are host memory). A device-resident value is
/// never silently copied to the host here: on the CPU native device a
/// device-resident input cannot legitimately arise, so this fails closed with
/// an actionable diagnostic rather than reading device memory as host bytes.
/// The CUDA path never routes device inputs through this function — it binds
/// them zero-copy instead (see [`run_native_component_cuda`]).
fn value_to_native_tensor(component: &str, port: &str, value: &Value) -> anyhow::Result<Tensor> {
    if !value.is_host_resident().with_context(|| {
        format!(
            "native workflow component '{component}' could not classify input '{port}' residency"
        )
    })? {
        anyhow::bail!(
            "native workflow component '{component}' received a device-resident input for port \
             '{port}' (device {}) on the CPU native path, which bridges through host bytes; run \
             this workflow on a CUDA native device (which binds device inputs zero-copy) or the \
             ORT backend.",
            value.device_id().unwrap_or(-1)
        );
    }
    host_value_to_tensor(component, port, value)
}

/// Build a native input tensor from a value already known to be host-resident.
///
/// Shared by the CPU path (after its residency guard) and the CUDA path (for the
/// genuinely host inputs it feeds through `run_with_device_bindings`'s host
/// input slot, e.g. token ids that originate on the host).
fn host_value_to_tensor(component: &str, port: &str, value: &Value) -> anyhow::Result<Tensor> {
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

/// Bridge a host-accessible native output tensor back into a workflow value.
///
/// A host-accessible tensor is bridged through its element bytes. A
/// device-resident output is **not** read as host bytes here (`Tensor::as_bytes`
/// is documented host-only and would be unsound): on the CUDA path a
/// device-resident output is instead wrapped zero-copy into an owning `Value`
/// by [`device_tensor_to_value`]. Reaching the guard below therefore means a
/// device tensor surfaced on a path that cannot own it (the CPU path), so it
/// fails closed rather than reading device memory as host bytes.
fn native_tensor_to_value(component: &str, port: &str, tensor: &Tensor) -> anyhow::Result<Value> {
    if !tensor.device().is_host_accessible() {
        anyhow::bail!(
            "native workflow component '{component}' produced a device-resident output '{port}' \
             on device {:?} on a path that bridges through host bytes; run this workflow on a CUDA \
             native device (which keeps the output device-resident) or the ORT backend.",
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
/// for the life of the [`WorkflowRuntime`], so a component invoked once per loop
/// iteration reuses one session.
pub(crate) struct NativeComponentSet {
    components: HashMap<String, NativeComponent>,
    device_label: String,
    /// CUDA device ordinal when the resolved native device is CUDA; `None` for
    /// CPU (or any build without `native-cuda`). Selects the device-resident
    /// execution path over the host-bytes path in [`Self::run_component`].
    cuda_ordinal: Option<u32>,
    run_count: u64,
    /// Number of device-resident inputs bound **zero-copy** into a component
    /// (an intermediate or recurring/state tensor that entered a component still
    /// resident on the device, with no host round-trip). Non-zero after a
    /// multi-component CUDA run proves the device-resident edge is real.
    device_input_bindings: u64,
    /// Number of device-resident outputs a component produced and published as
    /// an owning device `Value` (kept on the device for the next component).
    device_outputs: u64,
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
            cuda_ordinal: native_cuda_ordinal(device),
            run_count: 0,
            device_input_bindings: 0,
            device_outputs: 0,
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

    /// `(device_input_bindings, device_outputs)` accumulated so far.
    ///
    /// `device_input_bindings > 0` proves an intermediate or recurring/state
    /// tensor entered a component **still device-resident** (bound zero-copy,
    /// no host round-trip); `device_outputs > 0` proves a component's output
    /// stayed on the device for the next component. Both are always zero on the
    /// CPU native device. Tests read these to prove end-to-end device residency.
    pub(crate) fn device_residency_counts(&self) -> (u64, u64) {
        (self.device_input_bindings, self.device_outputs)
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
        output_shapes: &BTreeMap<String, Vec<usize>>,
    ) -> anyhow::Result<Vec<(String, Value)>> {
        let cuda_ordinal = self.cuda_ordinal;
        let (produced, device_inputs, device_outputs) = {
            let native = self.components.get_mut(component).with_context(|| {
                format!(
                    "native workflow backend has no session for ONNX component '{component}'; it \
                     was not present in the package's component model files"
                )
            })?;
            match cuda_ordinal {
                // A CUDA native device keeps intermediate and loop-carried/state
                // tensors device-resident across the component boundary.
                #[cfg(feature = "native-cuda")]
                Some(ordinal) => {
                    let outcome = run_native_component_cuda(
                        component,
                        native,
                        resolved,
                        output_shapes,
                        ordinal,
                    )?;
                    (
                        outcome.produced,
                        outcome.device_input_bindings,
                        outcome.device_outputs,
                    )
                }
                #[cfg(not(feature = "native-cuda"))]
                Some(_) => unreachable!("cuda_ordinal is only set under the native-cuda feature"),
                // A CPU native device bridges each boundary through host bytes.
                None => {
                    let _ = output_shapes;
                    (
                        run_native_component_host(component, native, resolved)?,
                        0,
                        0,
                    )
                }
            }
        };
        self.run_count += 1;
        self.device_input_bindings += device_inputs;
        self.device_outputs += device_outputs;
        Ok(produced)
    }
}

/// CPU path: bridge every tensor through host bytes.
fn run_native_component_host(
    component: &str,
    native: &mut NativeComponent,
    resolved: &[(&str, &Value)],
) -> anyhow::Result<Vec<(String, Value)>> {
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

/// CUDA device ordinal when `device` is a CUDA device (and this build can reach
/// it), else `None`. `None` selects the host-bytes path in `run_component`.
fn native_cuda_ordinal(device: &NativeDecodeDevice) -> Option<u32> {
    match device {
        #[cfg(feature = "native-cuda")]
        NativeDecodeDevice::Cuda { index } => Some(index.unwrap_or(0)),
        _ => None,
    }
}

/// CUDA path: keep tensors device-resident across the component boundary.
///
/// Device-resident inputs (a recurring/loop-carried/state tensor produced by an
/// earlier component on this device) are bound **zero-copy** into the session
/// through an external-memory [`DeviceIoBinding`]; genuinely host-resident
/// inputs (token ids and other host-origin tensors) are uploaded once through
/// the run's host-input slot. Every graph output is left unbound, so the session
/// returns it as a device-resident [`Tensor`] which is wrapped into a `Value`
/// that *owns* it — no host round-trip on any recurring edge.
///
/// Cross-session correctness: each component is its own session (and stream), so
/// before an output is handed to the next component this synchronizes the
/// producing stream (the conservative DLPack-export handshake). That is a
/// device-side barrier, not a host copy.
/// The outcome of one native CUDA component run: every declared output as a
/// workflow value, plus the device-residency counts this invocation contributed
/// (`device_input_bindings` device-resident inputs bound zero-copy, and
/// `device_outputs` device-resident outputs published).
#[cfg(feature = "native-cuda")]
struct CudaRunOutcome {
    produced: Vec<(String, Value)>,
    device_input_bindings: u64,
    device_outputs: u64,
}

#[cfg(feature = "native-cuda")]
fn run_native_component_cuda(
    component: &str,
    native: &mut NativeComponent,
    resolved: &[(&str, &Value)],
    output_shapes: &BTreeMap<String, Vec<usize>>,
    ordinal: u32,
) -> anyhow::Result<CudaRunOutcome> {
    use onnx_runtime_session::ExternalMemorySpec;

    let device_id = i32::try_from(ordinal).map_err(|_| {
        anyhow::anyhow!("native workflow CUDA device ordinal {ordinal} does not fit in an i32")
    })?;

    // Partition inputs by residency: host inputs feed the run's host slot;
    // device inputs are bound zero-copy so they never round-trip the host.
    let mut host_inputs: Vec<(String, Tensor)> = Vec::new();
    let mut device_specs: Vec<ExternalMemorySpec> = Vec::new();
    for (port, value) in resolved {
        let host_resident = value.is_host_resident().with_context(|| {
            format!(
                "native workflow component '{component}' could not classify input '{port}' \
                 residency"
            )
        })?;
        if host_resident {
            host_inputs.push((
                (*port).to_string(),
                host_value_to_tensor(component, port, value)?,
            ));
            continue;
        }
        let value_device = value.device_id().with_context(|| {
            format!(
                "native workflow component '{component}' could not read device of input '{port}'"
            )
        })?;
        anyhow::ensure!(
            value_device == device_id,
            "native workflow component '{component}' input '{port}' is resident on device \
             {value_device}, but this component runs on CUDA device {device_id}; cross-device \
             workflow edges are not supported (route both components to the same native device)"
        );
        let dtype = ir_dtype_from_ort(value.dtype());
        let shape = native_shape(component, port, value.shape())?;
        let numel: usize = shape.iter().product();
        let len_bytes = numel
            .checked_mul(value.dtype().size_of())
            .ok_or_else(|| anyhow::anyhow!("input '{port}' byte size overflows usize"))?;
        let ptr = value.data_ptr_addr().with_context(|| {
            format!(
                "native workflow component '{component}' could not read device pointer of '{port}'"
            )
        })? as *mut std::ffi::c_void;
        device_specs.push(ExternalMemorySpec::input(
            (*port).to_string(),
            None::<String>,
            dtype,
            shape.clone(),
            shape,
            ptr,
            len_bytes,
        ));
    }

    let mut bindings = Vec::with_capacity(device_specs.len());
    for spec in device_specs {
        let port = spec.input_name.clone();
        // SAFETY: `spec.ptr` names device memory on this session's device
        // (validated equal to `device_id`) and covers `spec.len_bytes` (>=
        // physical_shape). The source `Value` is borrowed for this whole call,
        // so it outlives the run; the output sync below drains this stream
        // before we return, so no kernel reads the borrowed input afterwards.
        let binding = unsafe { native.session.device_binding_from_external_memory(spec) }
            .with_context(|| {
                format!(
                    "native workflow component '{component}' could not bind device input '{port}'"
                )
            })?;
        bindings.push(binding);
    }
    let device_input_bindings = bindings.len() as u64;

    // Bind a device output buffer for every output whose concrete shape the
    // interpreter already resolved from the bound input symbols. Its bytes then
    // stay on the device (returned as `None` from the run) instead of being
    // host-materialized, so a recurring/state output hands its device buffer to
    // the next component with no host round-trip. Outputs with a genuinely
    // dynamic shape are left unbound and host-materialized (the correct fallback
    // — we cannot size a device buffer we cannot yet shape).
    let output_dtypes: HashMap<&str, IrDataType> = native
        .session
        .outputs()
        .iter()
        .map(|io| (io.name.as_str(), io.dtype))
        .collect();
    let input_binding_count = bindings.len();
    let mut output_binding_ports: Vec<String> = Vec::new();
    for name in &native.output_names {
        let Some(shape) = output_shapes.get(name) else {
            continue;
        };
        let Some(&dtype) = output_dtypes.get(name.as_str()) else {
            continue;
        };
        let binding = native
            .session
            .allocate_device_output_binding(name.clone(), dtype, shape.clone(), shape.clone())
            .with_context(|| {
                format!(
                    "native workflow component '{component}' could not allocate device output \
                     buffer for '{name}'"
                )
            })?;
        bindings.push(binding);
        output_binding_ports.push(name.clone());
    }

    let host_refs: Vec<(&str, &Tensor)> = host_inputs
        .iter()
        .map(|(port, tensor)| (port.as_str(), tensor))
        .collect();
    let outputs = native
        .session
        .run_with_device_bindings(&host_refs, &mut bindings)
        .with_context(|| {
            format!("native workflow component '{component}' failed to run on CUDA")
        })?;

    anyhow::ensure!(
        outputs.len() == native.output_names.len(),
        "native workflow component '{component}' returned {} tensors but declares {} outputs",
        outputs.len(),
        native.output_names.len(),
    );

    // The device output bindings are the tail of `bindings`; move them out and
    // key them by port. The remaining (input) bindings borrow external memory
    // and free nothing when dropped at the end of this function.
    let output_bindings = bindings.split_off(input_binding_count);
    let mut output_binding_map: HashMap<String, onnx_runtime_session::DeviceIoBinding> =
        output_binding_ports
            .into_iter()
            .zip(output_bindings)
            .collect();

    let mut produced = Vec::with_capacity(outputs.len());
    let mut device_outputs = 0u64;
    for (name, output) in native.output_names.iter().zip(outputs) {
        if let Some(binding) = output_binding_map.remove(name) {
            // Device-resident bound output: its bytes stayed in our buffer.
            anyhow::ensure!(
                output.is_none(),
                "native workflow component '{component}' bound output '{name}' but the run also \
                 returned a tensor for it"
            );
            produced.push((
                name.clone(),
                value_from_output_binding(component, name, binding, device_id)?,
            ));
            device_outputs += 1;
            continue;
        }
        // Unbound output (dynamic shape): host-materialized by the run. Bridge
        // it through the shared helper, which keeps a device tensor resident if
        // one ever surfaces here.
        let tensor = output.ok_or_else(|| {
            anyhow::anyhow!(
                "native workflow component '{component}' output '{name}' was neither bound nor \
                 returned by the run"
            )
        })?;
        tensor.sync().with_context(|| {
            format!("native workflow component '{component}' could not synchronize output '{name}'")
        })?;
        if !tensor.device().is_host_accessible() {
            device_outputs += 1;
        }
        produced.push((
            name.clone(),
            device_tensor_to_value(component, name, tensor, device_id)?,
        ));
    }
    Ok(CudaRunOutcome {
        produced,
        device_input_bindings,
        device_outputs,
    })
}

/// Publish a device output binding's buffer as an owning device-resident value.
///
/// The returned `Value` takes ownership of the whole [`DeviceIoBinding`]; the
/// binding's `Drop` frees its device allocation, and (through
/// `Value::from_external_memory_with_owner`) that happens only after ORT
/// releases the `OrtValue`, so the buffer outlives every ORT use of it with no
/// leak. The producing stream is drained first so the next component reads
/// completed bytes.
#[cfg(feature = "native-cuda")]
fn value_from_output_binding(
    component: &str,
    port: &str,
    binding: onnx_runtime_session::DeviceIoBinding,
    device_id: i32,
) -> anyhow::Result<Value> {
    use onnx_genai_ort::MemoryInfo;

    // Device-side barrier: drain the stream that wrote this output.
    binding.allocator().sync().with_context(|| {
        format!("native workflow component '{component}' could not synchronize output '{port}'")
    })?;
    let dtype = ort_dtype_from_ir(component, port, binding.dtype)?;
    let shape_usize = binding.physical_shape().to_vec();
    let shape: Vec<i64> = shape_usize.iter().map(|&dim| dim as i64).collect();
    let numel: usize = shape_usize.iter().product();
    // A zero-element output has no allocation to alias; publish it host-empty.
    if numel == 0 {
        return Value::from_raw_bytes(Vec::new(), &shape, dtype).with_context(|| {
            format!(
                "native workflow component '{component}' could not publish empty output '{port}'"
            )
        });
    }
    let len_bytes = numel
        .checked_mul(dtype.size_of())
        .ok_or_else(|| anyhow::anyhow!("output '{port}' byte size overflows usize"))?;
    let ptr = binding.device_ptr() as *mut std::ffi::c_void;
    let memory_info = MemoryInfo::cuda(device_id).with_context(|| {
        format!(
            "native workflow component '{component}' could not describe CUDA device {device_id}"
        )
    })?;
    // SAFETY: `ptr` is `binding`'s own device allocation on `device_id`, valid
    // for `len_bytes` (numel * dtype size). The returned `Value` takes ownership
    // of the whole binding, so the allocation outlives the `Value` and every ORT
    // use of it; shape/dtype are the binding's own. The stream was synchronized
    // above, so the bytes are valid.
    let value = unsafe {
        Value::from_external_memory_with_owner(
            ptr,
            len_bytes,
            &shape,
            dtype,
            &memory_info,
            Box::new(binding),
        )
    }
    .with_context(|| {
        format!(
            "native workflow component '{component}' could not wrap device output '{port}' into a \
             workflow value"
        )
    })?;
    Ok(value)
}

/// Wrap a native output tensor into a workflow value, keeping a device tensor
/// device-resident.
///
/// A host-accessible tensor uses the host-bytes bridge. A device-resident tensor
/// is wrapped **zero-copy** into a `Value` that takes ownership of the tensor
/// (`Value::from_external_memory_with_owner`): the tensor's device allocation is
/// freed only when the `Value` (and every alias derived from it) drops, after
/// ORT releases the `OrtValue`, so a loop-carried/recurring edge can hold the
/// device buffer across iterations with no leak and no use-after-free.
#[cfg(feature = "native-cuda")]
fn device_tensor_to_value(
    component: &str,
    port: &str,
    tensor: Tensor,
    device_id: i32,
) -> anyhow::Result<Value> {
    use onnx_genai_ort::MemoryInfo;

    if tensor.device().is_host_accessible() {
        return native_tensor_to_value(component, port, &tensor);
    }
    let dtype = ort_dtype_from_ir(component, port, tensor.dtype)?;
    let shape: Vec<i64> = tensor.shape.iter().map(|&dim| dim as i64).collect();
    let numel = tensor.numel();
    // A zero-element device tensor has no allocation to alias (its device
    // pointer is null), so publish it as an empty host value of the right type.
    if numel == 0 {
        return Value::from_raw_bytes(Vec::new(), &shape, dtype).with_context(|| {
            format!(
                "native workflow component '{component}' could not publish empty output '{port}'"
            )
        });
    }
    let len_bytes = numel
        .checked_mul(dtype.size_of())
        .ok_or_else(|| anyhow::anyhow!("output '{port}' byte size overflows usize"))?;
    let ptr = tensor.device_ptr() as *mut std::ffi::c_void;
    let memory_info = MemoryInfo::cuda(device_id).with_context(|| {
        format!(
            "native workflow component '{component}' could not describe CUDA device {device_id}"
        )
    })?;
    // SAFETY: `ptr` is `tensor`'s own device allocation on `device_id`, valid for
    // `len_bytes` (numel * dtype size). The returned `Value` takes ownership of
    // `tensor` via the guard box, so the allocation outlives the `Value` and
    // every ORT use of it; shape/dtype are the tensor's own. The stream was
    // synchronized by the caller, so the bytes are valid.
    let value = unsafe {
        Value::from_external_memory_with_owner(
            ptr,
            len_bytes,
            &shape,
            dtype,
            &memory_info,
            Box::new(tensor),
        )
    }
    .with_context(|| {
        format!(
            "native workflow component '{component}' could not wrap device output '{port}' into a \
             workflow value"
        )
    })?;
    Ok(value)
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
