use std::collections::HashMap;

use onnx_runtime_eager::{EagerError, LATEST_ONNX_OPSET, Tensor as EagerTensor, global_context};
use onnx_runtime_ir::Attribute;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule, PySequence, PyString};

use crate::{non_host_read_error, numpy_array_parts, raw_tensor_to_numpy};

fn eager_error(op_type: &str, domain: &str, err: EagerError) -> PyErr {
    let message = format!(
        "eager dispatch failed for operator {op_type:?} in domain {domain:?}: {err}. \
         Verify the operator/domain/opset is supported by nxrt's CPU execution provider, \
         and that input dtypes and shapes satisfy the ONNX schema."
    );
    match err {
        EagerError::NoKernel { .. }
        | EagerError::MixedDeviceInputs { .. }
        | EagerError::NoEpForDevice(_)
        | EagerError::ShapeInference { .. }
        | EagerError::ShapeInferEngine(_)
        | EagerError::Ir(_) => PyValueError::new_err(message),
        EagerError::Kernel(_) => PyRuntimeError::new_err(message),
    }
}

fn sequence_attribute(name: &str, value: &Bound<'_, PyAny>) -> PyResult<Attribute> {
    let seq = value.downcast::<PySequence>().map_err(|_| {
        PyTypeError::new_err(format!(
            "eager attribute {name:?} must be bool, int, float, str, bytes, or a \
             homogeneous non-empty sequence of int, float, str, or bytes"
        ))
    })?;
    if seq.len()? == 0 {
        return Err(PyValueError::new_err(format!(
            "eager attribute {name:?} is an empty sequence, whose ONNX element type is \
             ambiguous. Pass a non-empty homogeneous sequence."
        )));
    }

    let items: Vec<Bound<'_, PyAny>> = seq.try_iter()?.collect::<PyResult<_>>()?;
    if items.iter().all(|item| item.extract::<i64>().is_ok()) {
        return Ok(Attribute::Ints(
            items
                .iter()
                .map(|item| item.extract::<i64>())
                .collect::<PyResult<_>>()?,
        ));
    }
    if items.iter().all(|item| item.extract::<f32>().is_ok()) {
        return Ok(Attribute::Floats(
            items
                .iter()
                .map(|item| item.extract::<f32>())
                .collect::<PyResult<_>>()?,
        ));
    }
    if items.iter().all(|item| item.downcast::<PyString>().is_ok()) {
        return Ok(Attribute::Strings(
            items
                .iter()
                .map(|item| Ok(item.downcast::<PyString>()?.to_str()?.as_bytes().to_vec()))
                .collect::<PyResult<_>>()?,
        ));
    }
    if items.iter().all(|item| item.downcast::<PyBytes>().is_ok()) {
        return Ok(Attribute::Strings(
            items
                .iter()
                .map(|item| Ok(item.downcast::<PyBytes>()?.as_bytes().to_vec()))
                .collect::<PyResult<_>>()?,
        ));
    }
    Err(PyTypeError::new_err(format!(
        "eager attribute {name:?} has a mixed or unsupported sequence type. Pass only \
         homogeneous int, float, str, or bytes values."
    )))
}

fn attribute_from_python(name: &str, value: &Bound<'_, PyAny>) -> PyResult<Attribute> {
    if value.is_instance_of::<pyo3::types::PyBool>() {
        return Ok(Attribute::Int(i64::from(value.extract::<bool>()?)));
    }
    if let Ok(v) = value.extract::<i64>() {
        return Ok(Attribute::Int(v));
    }
    if let Ok(v) = value.extract::<f32>() {
        return Ok(Attribute::Float(v));
    }
    if let Ok(v) = value.downcast::<PyString>() {
        return Ok(Attribute::String(v.to_str()?.as_bytes().to_vec()));
    }
    if let Ok(v) = value.downcast::<PyBytes>() {
        return Ok(Attribute::String(v.as_bytes().to_vec()));
    }
    sequence_attribute(name, value)
}

fn attributes_from_dict(
    attributes: Option<&Bound<'_, PyDict>>,
) -> PyResult<HashMap<String, Attribute>> {
    let mut attrs = HashMap::new();
    if let Some(attributes) = attributes {
        for (key, value) in attributes.iter() {
            let key: String = key.extract().map_err(|_| {
                PyTypeError::new_err(
                    "eager attribute names must be strings, for example {'axis': 0}",
                )
            })?;
            attrs.insert(key.clone(), attribute_from_python(&key, &value)?);
        }
    }
    Ok(attrs)
}

#[pyfunction]
#[pyo3(signature = (op_type, inputs, attributes=None, *, domain="", opset=None))]
fn dispatch(
    py: Python<'_>,
    op_type: &str,
    inputs: &Bound<'_, PyAny>,
    attributes: Option<&Bound<'_, PyDict>>,
    domain: &str,
    opset: Option<u64>,
) -> PyResult<Vec<Py<PyAny>>> {
    if op_type.is_empty() {
        return Err(PyValueError::new_err(
            "op_type must not be empty; pass an ONNX operator name such as 'Add'",
        ));
    }
    let input_seq = inputs.downcast::<PySequence>().map_err(|_| {
        PyTypeError::new_err(
            "eager.dispatch inputs must be a list or tuple of numpy-compatible arrays",
        )
    })?;
    let np = py.import("numpy")?;
    let values: Vec<Bound<'_, PyAny>> = input_seq.try_iter()?.collect::<PyResult<_>>()?;
    let tensors: Vec<EagerTensor> = values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let name = format!("input[{index}]");
            let (dtype, shape, bytes) = numpy_array_parts(py, &np, &name, value)?;
            EagerTensor::from_raw(dtype, shape.clone(), &bytes).map_err(|err| {
                PyValueError::new_err(format!(
                    "{name}: failed to build eager tensor with dtype {dtype:?} and \
                     shape {shape:?}: {err}. Ensure the array is C-contiguous or \
                     convertible with numpy.asarray()."
                ))
            })
        })
        .collect::<PyResult<_>>()?;
    let refs: Vec<&EagerTensor> = tensors.iter().collect();
    let attrs = attributes_from_dict(attributes)?;
    let outputs = global_context()
        .dispatch(op_type, domain, &refs, &attrs, opset)
        .map_err(|err| eager_error(op_type, domain, err))?;

    outputs
        .iter()
        .enumerate()
        .map(|(index, tensor)| {
            let name = format!("{op_type} output[{index}]");
            if let Some(message) = non_host_read_error(&name, tensor.device()) {
                return Err(PyRuntimeError::new_err(message));
            }
            raw_tensor_to_numpy(
                py,
                &np,
                &name,
                tensor.dtype(),
                tensor.shape(),
                tensor.as_bytes(),
                tensor.device(),
            )
        })
        .collect()
}

#[pyfunction]
fn opset() -> u64 {
    LATEST_ONNX_OPSET
}

#[pyfunction]
fn cache_stats(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let stats = global_context().cache_stats();
    let result = PyDict::new(py);
    result.set_item("entries", stats.entries)?;
    result.set_item("hits", stats.hits)?;
    result.set_item("misses", stats.misses)?;
    Ok(result.unbind())
}

pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let module = PyModule::new(py, "eager")?;
    module.add(
        "__doc__",
        "Single-op eager execution over numpy-compatible arrays.",
    )?;
    module.add("LATEST_ONNX_OPSET", LATEST_ONNX_OPSET)?;
    module.add_function(wrap_pyfunction!(dispatch, &module)?)?;
    module.add_function(wrap_pyfunction!(opset, &module)?)?;
    module.add_function(wrap_pyfunction!(cache_stats, &module)?)?;
    parent.add_submodule(&module)?;
    py.import("sys")?
        .getattr("modules")?
        .downcast_into::<PyDict>()?
        .set_item("nxrt.eager", &module)?;
    Ok(())
}
