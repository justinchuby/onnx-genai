//! `pkg.nxrt::SparseKvGather` v1: device-native sparse KV gather for the
//! DeepSeek/GLM compressed sparse-attention building block.
//!
//! Mirrors the CPU reference in
//! `crates/onnx-runtime-ep-cpu/src/kernels/sparse_kv_gather.rs`. The public
//! operator is strict: negative and out-of-range indices are hard errors
//! (`out_of_range='error'`, `index_layout_version=1`). Given a key/value cache
//! `[B, G, C, D]` and a candidate index table `[B, G, Q, K]`, it produces the
//! gathered records `[B, G, Q, K, D]`, preserving index order and duplicates.
//!
//! The value copy runs entirely on device (an NVRTC grid-stride byte-copy
//! kernel over the source cache and the index buffer). Only the small index /
//! `valid_lengths` tensors are read back to the host for the ONNX-required
//! deterministic range check, matching the `Gather` kernel's precedent.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const OP: &str = "SparseKvGather";
const INDEX_LAYOUT_VERSION: i64 = 1;
const BLOCK: u32 = 256;

const SOURCE: &str = r#"
extern "C" __global__ void sparse_kv_gather_bytes(
    const unsigned char* cache, const void* indices, unsigned char* output,
    long long output_bytes, long long row_bytes, long long cache_len,
    long long records_per_bg, int index_is_i64) {
  for (long long byte = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       byte < output_bytes; byte += (long long)gridDim.x * blockDim.x) {
    long long record = byte / row_bytes;
    long long within = byte - record * row_bytes;
    long long index = index_is_i64
        ? ((const long long*)indices)[record]
        : (long long)((const int*)indices)[record];
    long long bg = record / records_per_bg;
    long long source = (bg * cache_len + index) * row_bytes + within;
    output[byte] = cache[source];
  }
}
"#;

pub struct SparseKvGatherFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for SparseKvGatherFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let version = node
            .attr("index_layout_version")
            .and_then(|attribute| attribute.as_int())
            .unwrap_or(INDEX_LAYOUT_VERSION);
        if version != INDEX_LAYOUT_VERSION {
            return Err(error(format!(
                "index_layout_version must be {INDEX_LAYOUT_VERSION}, got {version}"
            )));
        }
        let out_of_range = match node.attr("out_of_range") {
            Some(attribute) => attribute
                .as_str()
                .ok_or_else(|| error("attribute out_of_range must be a UTF-8 string"))?,
            None => "error",
        };
        if out_of_range != "error" {
            return Err(not_implemented(format!(
                "{OP}: out_of_range='{out_of_range}' is unsupported; v1 requires 'error'"
            )));
        }
        Ok(Box::new(SparseKvGatherKernel {
            runtime: self.runtime.clone(),
        }))
    }
}

#[derive(Debug)]
pub struct SparseKvGatherKernel {
    runtime: Arc<CudaRuntime>,
}

impl Kernel for SparseKvGatherKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if !(2..=3).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(error(format!(
                "expected 2 or 3 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let cache = &inputs[0];
        let indices = &inputs[1];
        let output = &mut outputs[0];

        // Value copy is a raw byte gather, so any fixed-width element dtype is
        // supported as long as the cache and output agree. The CPU reference is
        // f32-only; f16/bf16 fall out for free since values are never
        // interpreted, only copied.
        if !matches!(
            cache.dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            return Err(not_implemented(format!(
                "{OP}: cache dtype {:?} unsupported; expected Float32, Float16, or BFloat16",
                cache.dtype
            )));
        }
        if output.dtype != cache.dtype {
            return Err(error(format!(
                "selected dtype {:?} must match cache dtype {:?}",
                output.dtype, cache.dtype
            )));
        }
        if !matches!(indices.dtype, DataType::Int32 | DataType::Int64) {
            return Err(not_implemented(format!(
                "{OP}: indices dtype {:?} unsupported; expected Int32 or Int64",
                indices.dtype
            )));
        }
        if !cache.is_contiguous() || !indices.is_contiguous() || !output.is_contiguous() {
            return Err(not_implemented(format!(
                "{OP} with non-contiguous input/output"
            )));
        }

        let cache_shape = shape4("cache", cache.shape)?;
        let indices_shape = shape4("indices", indices.shape)?;
        let [batch, groups, cache_len, dim] = cache_shape;
        let [index_batch, index_groups, queries, selections] = indices_shape;
        if (batch, groups) != (index_batch, index_groups) {
            return Err(error(format!(
                "cache and indices batch/group dimensions must match, got [{batch},{groups}] and [{index_batch},{index_groups}]"
            )));
        }
        let expected_output = [batch, groups, queries, selections, dim];
        if output.shape != expected_output {
            return Err(error(format!(
                "selected must have shape {expected_output:?}, got {:?}",
                output.shape
            )));
        }

        // Optional per-batch valid length limit (input 2, shape [B]).
        let valid_lengths = match inputs.get(2) {
            Some(view) if !view.is_absent() => {
                Some(self.read_valid_lengths(view, batch, cache_len)?)
            }
            _ => None,
        };

        let elem_bytes = cache.dtype.byte_size();
        let records = batch
            .checked_mul(groups)
            .and_then(|value| value.checked_mul(queries))
            .and_then(|value| value.checked_mul(selections))
            .ok_or_else(|| error("selected record count overflow"))?;
        let records_per_bg = queries
            .checked_mul(selections)
            .ok_or_else(|| error("per-group record count overflow"))?;
        let row_bytes = dim
            .checked_mul(elem_bytes)
            .ok_or_else(|| error("row byte count overflow"))?;
        let output_bytes = records
            .checked_mul(row_bytes)
            .ok_or_else(|| error("selected byte count overflow"))?;

        // An empty selection (K=0 or D=0) is a valid, empty, contiguous output.
        if output_bytes == 0 {
            return Ok(());
        }

        // A device-side out-of-range index would be an out-of-bounds load, so the
        // small index tensor is validated on the host before launch to keep the
        // ONNX-required error deterministic. The cache values never leave the
        // device.
        let host_indices = self.read_indices(indices, records)?;
        for (record, raw) in host_indices.iter().enumerate() {
            validate_index(
                *raw,
                record,
                records_per_bg,
                groups,
                cache_len,
                &valid_lengths,
            )?;
        }

        for (value, name) in [
            (output_bytes, "output byte count"),
            (row_bytes, "row byte count"),
            (cache_len, "cache length"),
            (records_per_bg, "per-group record count"),
        ] {
            i64::try_from(value).map_err(|_| error(format!("{name} exceeds i64")))?;
        }

        let func =
            self.runtime
                .nvrtc_function("sparse_kv_gather", SOURCE, "sparse_kv_gather_bytes")?;
        let output_bytes_i64 = output_bytes as i64;
        let row_bytes_i64 = row_bytes as i64;
        let cache_len_i64 = cache_len as i64;
        let records_per_bg_i64 = records_per_bg as i64;
        let index_is_i64 = i32::from(indices.dtype == DataType::Int64);
        let cache_ptr = cuptr(cache.data_ptr::<u8>() as *const c_void);
        let indices_ptr = cuptr(indices.data_ptr::<u8>() as *const c_void);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let grid = (output_bytes as u64)
            .div_ceil(BLOCK as u64)
            .min(65_535)
            .max(1) as u32;
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&cache_ptr)
            .arg(&indices_ptr)
            .arg(&output_ptr)
            .arg(&output_bytes_i64)
            .arg(&row_bytes_i64)
            .arg(&cache_len_i64)
            .arg(&records_per_bg_i64)
            .arg(&index_is_i64);
        // SAFETY: argument types and order match `sparse_kv_gather_bytes`; all
        // pointers refer to live contiguous device allocations validated above,
        // and every index was range-checked on the host.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err("launch sparse_kv_gather_bytes", e))?;
        self.runtime.synchronize()
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn cuda_graph_compatible(&self) -> bool {
        // Host-side index validation copies the index tensor D2H, and execution
        // synchronizes the stream before returning. Neither is legal during CUDA
        // graph capture.
        false
    }
}

impl SparseKvGatherKernel {
    fn read_indices(&self, view: &TensorView, records: usize) -> Result<Vec<i64>> {
        let byte_len = view.dtype.storage_bytes(records);
        let mut host = vec![0u8; byte_len];
        if !host.is_empty() {
            // SAFETY: `view` is a live contiguous device tensor and the host
            // buffer is exactly its fixed-width storage size.
            unsafe {
                self.runtime
                    .dtoh(&mut host, cuptr(view.data_ptr::<u8>() as *const c_void))?
            };
        }
        Ok(host
            .chunks_exact(view.dtype.byte_size())
            .map(|raw| match view.dtype {
                DataType::Int32 => i32::from_ne_bytes(raw.try_into().unwrap()) as i64,
                DataType::Int64 => i64::from_ne_bytes(raw.try_into().unwrap()),
                _ => unreachable!("index dtype was validated"),
            })
            .collect())
    }

    fn read_valid_lengths(
        &self,
        view: &TensorView,
        batch: usize,
        cache_len: usize,
    ) -> Result<Vec<usize>> {
        if !matches!(view.dtype, DataType::Int32 | DataType::Int64) {
            return Err(not_implemented(format!(
                "{OP}: valid_lengths dtype {:?} unsupported; expected Int32 or Int64",
                view.dtype
            )));
        }
        if view.shape != [batch] {
            return Err(error(format!(
                "valid_lengths must have shape [{batch}], got {:?}",
                view.shape
            )));
        }
        let raw = self.read_indices(view, batch)?;
        raw.into_iter()
            .enumerate()
            .map(|(b, length)| {
                let length = usize::try_from(length)
                    .map_err(|_| error(format!("valid_lengths[{b}] must be non-negative")))?;
                if length > cache_len {
                    return Err(error(format!(
                        "valid_lengths[{b}]={length} exceeds cache length {cache_len}"
                    )));
                }
                Ok(length)
            })
            .collect()
    }
}

fn validate_index(
    raw: i64,
    record: usize,
    records_per_bg: usize,
    groups: usize,
    cache_len: usize,
    valid_lengths: &Option<Vec<usize>>,
) -> Result<()> {
    let bg = if records_per_bg == 0 {
        0
    } else {
        record / records_per_bg
    };
    let b = if groups == 0 { 0 } else { bg / groups };
    let g = if groups == 0 { 0 } else { bg % groups };
    let within = if records_per_bg == 0 {
        0
    } else {
        record % records_per_bg
    };
    let valid_length = valid_lengths
        .as_ref()
        .map_or(cache_len, |lengths| lengths[b]);
    if raw < 0 {
        return Err(error(format!(
            "negative index {raw} at [batch={b}, group={g}, index={within}]"
        )));
    }
    if raw >= valid_length as i64 {
        return Err(error(format!(
            "index {raw} at [batch={b}, group={g}, index={within}] is out of range for valid length {valid_length}"
        )));
    }
    Ok(())
}

fn shape4(name: &str, shape: &[usize]) -> Result<[usize; 4]> {
    shape
        .try_into()
        .map_err(|_| error(format!("{name} must be rank 4, got shape {shape:?}")))
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("cuda_ep {OP}: {}", message.into()))
}
