//! CUDA `pkg.nxrt::GroupedLoraDelta` group-by-adapter dense kernel.
//!
//! Factor pages are copied from the shared host pool into persistent device
//! allocations on first use. Each invocation reads the small routing descriptor
//! back to the host, groups rows by adapter, and launches one dense fused
//! `X_group @ A_t @ B_t` kernel per adapter. Both dot products accumulate in
//! fp32; narrowing happens only at the final output store.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    AdapterId, EpError, Kernel, KernelFactory, LoraModuleId, LoraPagePair, LoraPoolId,
    LoraPoolRegistry, LoraWeightPool, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{DataType, Node};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const OPERATOR: &str = "GroupedLoraDelta";
const MODULE: &str = "grouped_lora_delta";
const ZERO_ENTRY: &str = "grouped_lora_zero";
const FLOAT32_ENTRY: &str = "grouped_lora_delta_float32";
const FLOAT16_ENTRY: &str = "grouped_lora_delta_float16";
const BFLOAT16_ENTRY: &str = "grouped_lora_delta_bfloat16";
const THREADS: u32 = 256;

const SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float grouped_lora_load_factor(
    const unsigned char* factor, unsigned long long index, int data_type)
{
    if (data_type == 0) {
        return ((const float*)factor)[index];
    }
    if (data_type == 1) {
        return __half2float(((const __half*)factor)[index]);
    }
    return __bfloat162float(((const __nv_bfloat16*)factor)[index]);
}

template <typename Value>
__device__ __forceinline__ float grouped_lora_load_value(
    const Value* values, unsigned long long index);

template <>
__device__ __forceinline__ float grouped_lora_load_value<float>(
    const float* values, unsigned long long index)
{
    return values[index];
}

template <>
__device__ __forceinline__ float grouped_lora_load_value<__half>(
    const __half* values, unsigned long long index)
{
    return __half2float(values[index]);
}

template <>
__device__ __forceinline__ float grouped_lora_load_value<__nv_bfloat16>(
    const __nv_bfloat16* values, unsigned long long index)
{
    return __bfloat162float(values[index]);
}

template <typename Value>
__device__ __forceinline__ Value grouped_lora_store_value(float value);

template <>
__device__ __forceinline__ float grouped_lora_store_value<float>(float value)
{
    return value;
}

template <>
__device__ __forceinline__ __half grouped_lora_store_value<__half>(float value)
{
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 grouped_lora_store_value<__nv_bfloat16>(
    float value)
{
    return __float2bfloat16_rn(value);
}

extern "C" __global__ void grouped_lora_zero(
    unsigned char* output, unsigned long long output_bytes)
{
    for (unsigned long long index =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < output_bytes;
         index += (unsigned long long)gridDim.x * blockDim.x) {
        output[index] = 0;
    }
}

template <typename Value>
__device__ void grouped_lora_delta(
    const Value* input,
    const int* rows,
    const unsigned char* factor_a,
    const unsigned char* factor_b,
    Value* output,
    unsigned long long group_rows,
    int input_width,
    int output_width,
    int rank,
    int factor_a_data_type,
    int factor_b_data_type,
    float scale)
{
    const unsigned long long elements =
        group_rows * (unsigned long long)output_width;
    for (unsigned long long element =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         element < elements;
         element += (unsigned long long)gridDim.x * blockDim.x) {
        const unsigned long long group_row = element / output_width;
        const int output_column = (int)(element % output_width);
        const int input_row = rows[group_row];
        float output_accumulator = 0.0f;
        for (int rank_column = 0; rank_column < rank; ++rank_column) {
            float intermediate_accumulator = 0.0f;
            for (int input_column = 0; input_column < input_width; ++input_column) {
                const float input_value = grouped_lora_load_value(
                    input,
                    (unsigned long long)input_row * input_width + input_column);
                const float factor_value = grouped_lora_load_factor(
                    factor_a,
                    (unsigned long long)input_column * rank + rank_column,
                    factor_a_data_type);
                intermediate_accumulator += input_value * factor_value;
            }
            const float factor_value = grouped_lora_load_factor(
                factor_b,
                (unsigned long long)rank_column * output_width + output_column,
                factor_b_data_type);
            output_accumulator += intermediate_accumulator * factor_value;
        }
        output[(unsigned long long)input_row * output_width + output_column] =
            grouped_lora_store_value<Value>(output_accumulator * scale);
    }
}

extern "C" __global__ void grouped_lora_delta_float32(
    const float* input, const int* rows,
    const unsigned char* factor_a, const unsigned char* factor_b, float* output,
    unsigned long long group_rows, int input_width, int output_width, int rank,
    int factor_a_data_type, int factor_b_data_type, float scale)
{
    grouped_lora_delta(
        input, rows, factor_a, factor_b, output, group_rows, input_width,
        output_width, rank, factor_a_data_type, factor_b_data_type, scale);
}

extern "C" __global__ void grouped_lora_delta_float16(
    const __half* input, const int* rows,
    const unsigned char* factor_a, const unsigned char* factor_b, __half* output,
    unsigned long long group_rows, int input_width, int output_width, int rank,
    int factor_a_data_type, int factor_b_data_type, float scale)
{
    grouped_lora_delta(
        input, rows, factor_a, factor_b, output, group_rows, input_width,
        output_width, rank, factor_a_data_type, factor_b_data_type, scale);
}

extern "C" __global__ void grouped_lora_delta_bfloat16(
    const __nv_bfloat16* input, const int* rows,
    const unsigned char* factor_a, const unsigned char* factor_b,
    __nv_bfloat16* output, unsigned long long group_rows, int input_width,
    int output_width, int rank, int factor_a_data_type, int factor_b_data_type,
    float scale)
{
    grouped_lora_delta(
        input, rows, factor_a, factor_b, output, group_rows, input_width,
        output_width, rank, factor_a_data_type, factor_b_data_type, scale);
}
"#;

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{OPERATOR}: {}", message.into()))
}

pub struct GroupedLoraDeltaFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for GroupedLoraDeltaFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let input_width = required_positive_attribute(node, "K")?;
        let output_width = required_positive_attribute(node, "N")?;
        let module_identifier = required_nonnegative_attribute(node, "module_id")?;
        let maximum_rank = required_positive_attribute(node, "max_rank")?;
        let pool_identifier = required_nonnegative_attribute(node, "pool_id")? as u64;
        let pool = LoraPoolRegistry::global()
            .get(LoraPoolId(pool_identifier))
            .ok_or_else(|| {
                error(format!(
                    "no adapter pool registered under pool_id {pool_identifier}; register the \
                     pool with LoraPoolRegistry before building the session"
                ))
            })?;
        Ok(Box::new(GroupedLoraDeltaKernel {
            runtime: self.runtime.clone(),
            input_width,
            output_width,
            module_identifier: LoraModuleId(module_identifier as u32),
            maximum_rank,
            pool,
            device_state: Mutex::new(DeviceState::default()),
        }))
    }
}

struct DeviceFactorPair {
    factor_a_pointer: CUdeviceptr,
    factor_b_pointer: CUdeviceptr,
    factor_a_data_type: i32,
    factor_b_data_type: i32,
    rank: usize,
    scale: f32,
}

#[derive(Default)]
struct DeviceState {
    factor_pairs: HashMap<AdapterId, DeviceFactorPair>,
    row_pointer: CUdeviceptr,
    row_capacity: usize,
}

pub struct GroupedLoraDeltaKernel {
    runtime: Arc<CudaRuntime>,
    input_width: usize,
    output_width: usize,
    module_identifier: LoraModuleId,
    maximum_rank: usize,
    pool: Arc<LoraWeightPool>,
    device_state: Mutex<DeviceState>,
}

impl Kernel for GroupedLoraDeltaKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(error(format!(
                "expected 2 inputs and 1 output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
        let input = &inputs[0];
        let segments = &inputs[1];
        let output = &mut outputs[0];
        require_value_data_type(input.dtype, "x")?;
        if output.dtype != input.dtype {
            return Err(error(format!(
                "delta dtype {:?} must match x dtype {:?}",
                output.dtype, input.dtype
            )));
        }
        if !matches!(segments.dtype, DataType::Int32 | DataType::Int64) {
            return Err(error(format!(
                "segments must be Int32 or Int64, got {:?}",
                segments.dtype
            )));
        }
        if !input.is_contiguous() || !segments.is_contiguous() || !output.is_contiguous() {
            return Err(error(
                "x, segments, and delta must be contiguous on the CUDA execution provider",
            ));
        }
        if input.shape.is_empty() || input.shape[input.shape.len() - 1] != self.input_width {
            return Err(error(format!(
                "x must have rank >= 1 with last dimension K={}, got shape {:?}",
                self.input_width, input.shape
            )));
        }
        let tokens = checked_product(&input.shape[..input.shape.len() - 1])?;
        let expected_output =
            [&input.shape[..input.shape.len() - 1], &[self.output_width]].concat();
        if output.shape != expected_output.as_slice() {
            return Err(error(format!(
                "delta shape {:?} must be {:?}",
                output.shape, expected_output
            )));
        }
        if checked_product(segments.shape)? != tokens {
            return Err(error(format!(
                "segments must have {tokens} elements (one per token), got shape {:?}",
                segments.shape
            )));
        }

        let segment_identifiers = self.read_segments(segments, tokens)?;
        let mut grouped_rows: Vec<(AdapterId, Vec<i32>)> = Vec::new();
        for (row, segment_identifier) in segment_identifiers.into_iter().enumerate() {
            let Some(adapter) = adapter_from_segment(segment_identifier) else {
                continue;
            };
            let row = i32::try_from(row)
                .map_err(|_| error("token row index exceeds the CUDA Int32 row table"))?;
            if let Some((_, rows)) = grouped_rows
                .iter_mut()
                .find(|(candidate, _)| *candidate == adapter)
            {
                rows.push(row);
            } else {
                grouped_rows.push((adapter, vec![row]));
            }
        }

        let mut device_state = self
            .device_state
            .lock()
            .map_err(|_| error("device factor cache mutex was poisoned"))?;
        for (adapter, _) in &grouped_rows {
            self.ensure_device_factor_pair(*adapter, &mut device_state)?;
        }
        self.zero_output(output)?;
        for (adapter, rows) in grouped_rows {
            self.launch_group(input, output, adapter, &rows, &mut device_state)?;
        }
        self.runtime.synchronize()
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "GroupedLoraDelta reads dynamic segments on the host and lazily binds persistent \
             factor pages; execute it as an eager CUDA-graph seam",
        )
    }
}

impl GroupedLoraDeltaKernel {
    fn read_segments(&self, segments: &TensorView, tokens: usize) -> Result<Vec<i64>> {
        let byte_count = tokens
            .checked_mul(segments.dtype.byte_size())
            .ok_or_else(|| error("segments byte count overflow"))?;
        let mut bytes = vec![0_u8; byte_count];
        if byte_count != 0 {
            let pointer = cuptr(segments.data_ptr::<u8>() as *const c_void);
            // SAFETY: the validated contiguous segments tensor contains exactly
            // `byte_count` readable bytes.
            unsafe { self.runtime.dtoh(&mut bytes, pointer) }?;
        }
        Ok(match segments.dtype {
            DataType::Int32 => bytes
                .chunks_exact(4)
                .map(|value| i32::from_ne_bytes(value.try_into().unwrap()) as i64)
                .collect(),
            DataType::Int64 => bytes
                .chunks_exact(8)
                .map(|value| i64::from_ne_bytes(value.try_into().unwrap()))
                .collect(),
            _ => unreachable!("segments dtype validated above"),
        })
    }

    fn ensure_device_factor_pair(
        &self,
        adapter: AdapterId,
        device_state: &mut DeviceState,
    ) -> Result<()> {
        if device_state.factor_pairs.contains_key(&adapter) {
            return Ok(());
        }
        let pair = self
            .pool
            .pair(adapter, self.module_identifier)
            .ok_or_else(|| {
                error(format!(
                    "adapter {} module {} has no resident page in the pool",
                    adapter.0, self.module_identifier.0
                ))
            })?;
        let (rank, factor_a_data_type, factor_b_data_type) = self.validate_pair(adapter, &pair)?;
        let factor_a_pointer = self.runtime.alloc_raw(pair.a.bytes.len())?;
        let factor_b_pointer = match self.runtime.alloc_raw(pair.b.bytes.len()) {
            Ok(pointer) => pointer,
            Err(failure) => {
                // SAFETY: the first allocation has not escaped or been launched.
                let _ = unsafe { self.runtime.free_raw(factor_a_pointer) };
                return Err(failure);
            }
        };
        let upload = unsafe {
            // SAFETY: both allocations exactly cover their immutable pool pages.
            self.runtime.htod(pair.a.bytes, factor_a_pointer)?;
            self.runtime.htod(pair.b.bytes, factor_b_pointer)
        };
        if let Err(failure) = upload {
            // SAFETY: neither pointer escaped the cache after a failed upload.
            let _ = unsafe { self.runtime.free_raw(factor_b_pointer) };
            let _ = unsafe { self.runtime.free_raw(factor_a_pointer) };
            return Err(failure);
        }
        device_state.factor_pairs.insert(
            adapter,
            DeviceFactorPair {
                factor_a_pointer,
                factor_b_pointer,
                factor_a_data_type,
                factor_b_data_type,
                rank,
                scale: pair.scale,
            },
        );
        Ok(())
    }

    fn validate_pair(
        &self,
        adapter: AdapterId,
        pair: &LoraPagePair<'_>,
    ) -> Result<(usize, i32, i32)> {
        let rank = pair.a.cols;
        if rank == 0 {
            return Err(error(format!(
                "adapter {} module {} has rank zero, which is unsupported on CUDA",
                adapter.0, self.module_identifier.0
            )));
        }
        if pair.a.rows != self.input_width {
            return Err(error(format!(
                "adapter {} module {}: A_t K {} != op K={}",
                adapter.0, self.module_identifier.0, pair.a.rows, self.input_width
            )));
        }
        if pair.b.rows != rank {
            return Err(error(format!(
                "adapter {} module {}: A_t rank {rank} != B_t rank {}",
                adapter.0, self.module_identifier.0, pair.b.rows
            )));
        }
        if pair.b.cols != self.output_width {
            return Err(error(format!(
                "adapter {} module {}: B_t width {} != op width N={}",
                adapter.0, self.module_identifier.0, pair.b.cols, self.output_width
            )));
        }
        if rank > self.maximum_rank {
            return Err(error(format!(
                "adapter {} module {}: rank {rank} exceeds max_rank {}",
                adapter.0, self.module_identifier.0, self.maximum_rank
            )));
        }
        Ok((
            rank,
            factor_data_type_identifier(pair.a.dtype)?,
            factor_data_type_identifier(pair.b.dtype)?,
        ))
    }

    fn zero_output(&self, output: &mut TensorMut) -> Result<()> {
        let output_bytes = checked_product(output.shape)?
            .checked_mul(output.dtype.byte_size())
            .ok_or_else(|| error("delta byte count overflow"))?;
        if output_bytes == 0 {
            return Ok(());
        }
        let function = self.runtime.nvrtc_function(MODULE, SOURCE, ZERO_ENTRY)?;
        let output_pointer = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let output_bytes = u64::try_from(output_bytes)
            .map_err(|_| error("delta byte count exceeds CUDA u64 indexing"))?;
        let grid = output_bytes.div_ceil(THREADS as u64).clamp(1, 65_535) as u32;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder.arg(&output_pointer).arg(&output_bytes);
        // SAFETY: the ABI matches `grouped_lora_zero`, and the output allocation
        // covers `output_bytes`.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|failure| driver_err("launch GroupedLoraDelta zero fill", failure))
    }

    fn launch_group(
        &self,
        input: &TensorView,
        output: &mut TensorMut,
        adapter: AdapterId,
        rows: &[i32],
        device_state: &mut DeviceState,
    ) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let row_bytes = std::mem::size_of_val(rows);
        if device_state.row_capacity < row_bytes {
            let replacement = self.runtime.alloc_raw(row_bytes)?;
            if device_state.row_pointer != 0 {
                // SAFETY: the prior row table belongs exclusively to this cache.
                unsafe { self.runtime.free_raw(device_state.row_pointer) }?;
            }
            device_state.row_pointer = replacement;
            device_state.row_capacity = row_bytes;
        }
        let row_bytes = unsafe {
            // SAFETY: i32 has no padding and the byte slice retains `rows`' lifetime.
            std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), row_bytes)
        };
        // SAFETY: the persistent row allocation covers `row_bytes`.
        unsafe {
            self.runtime.htod(row_bytes, device_state.row_pointer)?;
        }

        let factor_pair = device_state
            .factor_pairs
            .get(&adapter)
            .expect("factor pair was validated and uploaded before launch");
        let entry = match input.dtype {
            DataType::Float32 => FLOAT32_ENTRY,
            DataType::Float16 => FLOAT16_ENTRY,
            DataType::BFloat16 => BFLOAT16_ENTRY,
            _ => unreachable!("input dtype validated above"),
        };
        let function = self.runtime.nvrtc_function(MODULE, SOURCE, entry)?;
        let input_pointer = cuptr(input.data_ptr::<u8>() as *const c_void);
        let output_pointer = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let group_rows =
            u64::try_from(rows.len()).map_err(|_| error("adapter group size exceeds u64"))?;
        let input_width =
            i32::try_from(self.input_width).map_err(|_| error("K exceeds CUDA Int32 indexing"))?;
        let output_width =
            i32::try_from(self.output_width).map_err(|_| error("N exceeds CUDA Int32 indexing"))?;
        let rank = i32::try_from(factor_pair.rank).map_err(|_| error("rank exceeds CUDA Int32"))?;
        let elements = group_rows
            .checked_mul(self.output_width as u64)
            .ok_or_else(|| error("adapter group output element count overflow"))?;
        let grid = elements.div_ceil(THREADS as u64).clamp(1, 65_535) as u32;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&input_pointer)
            .arg(&device_state.row_pointer)
            .arg(&factor_pair.factor_a_pointer)
            .arg(&factor_pair.factor_b_pointer)
            .arg(&output_pointer)
            .arg(&group_rows)
            .arg(&input_width)
            .arg(&output_width)
            .arg(&rank)
            .arg(&factor_pair.factor_a_data_type)
            .arg(&factor_pair.factor_b_data_type)
            .arg(&factor_pair.scale);
        // SAFETY: arguments match the selected typed entry point; tensor shapes,
        // row indices, factor geometry, and device allocation sizes were checked.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|failure| driver_err("launch GroupedLoraDelta group", failure))
    }
}

impl Drop for GroupedLoraDeltaKernel {
    fn drop(&mut self) {
        let device_state = self
            .device_state
            .get_mut()
            .expect("GroupedLoraDelta device cache mutex poisoned");
        if device_state.row_pointer != 0 {
            // SAFETY: this pointer was allocated by this runtime and is owned here.
            let _ = unsafe { self.runtime.free_raw(device_state.row_pointer) };
            device_state.row_pointer = 0;
        }
        for (_, factor_pair) in device_state.factor_pairs.drain() {
            // SAFETY: both pointers were allocated by this runtime and are owned here.
            let _ = unsafe { self.runtime.free_raw(factor_pair.factor_b_pointer) };
            let _ = unsafe { self.runtime.free_raw(factor_pair.factor_a_pointer) };
        }
    }
}

fn adapter_from_segment(identifier: i64) -> Option<AdapterId> {
    if identifier < 0 {
        return None;
    }
    let adapter = AdapterId(identifier as u64);
    (!adapter.is_null()).then_some(adapter)
}

fn factor_data_type_identifier(data_type: DataType) -> Result<i32> {
    match data_type {
        DataType::Float32 => Ok(0),
        DataType::Float16 => Ok(1),
        DataType::BFloat16 => Ok(2),
        other => Err(error(format!(
            "adapter factor dtype {other:?} is unsupported; expected Float32, Float16, or BFloat16"
        ))),
    }
}

fn require_value_data_type(data_type: DataType, name: &str) -> Result<()> {
    match data_type {
        DataType::Float32 | DataType::Float16 | DataType::BFloat16 => Ok(()),
        other => Err(error(format!(
            "{name} must be Float32, Float16, or BFloat16, got {other:?}"
        ))),
    }
}

fn checked_product(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1_usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or_else(|| error("tensor element count overflow"))
    })
}

fn required_positive_attribute(node: &Node, name: &str) -> Result<usize> {
    let value = required_integer_attribute(node, name)?;
    if value <= 0 {
        return Err(error(format!(
            "attribute '{name}' must be positive, got {value}"
        )));
    }
    usize::try_from(value).map_err(|_| error(format!("attribute '{name}' exceeds usize")))
}

fn required_nonnegative_attribute(node: &Node, name: &str) -> Result<i64> {
    let value = required_integer_attribute(node, name)?;
    if value < 0 {
        return Err(error(format!(
            "attribute '{name}' must be non-negative, got {value}"
        )));
    }
    Ok(value)
}

fn required_integer_attribute(node: &Node, name: &str) -> Result<i64> {
    node.attr(name)
        .and_then(|attribute| attribute.as_int())
        .ok_or_else(|| error(format!("missing required integer attribute '{name}'")))
}

#[cfg(test)]
mod tests {
    use onnx_runtime_ep_api::{
        DevicePtr, DevicePtrMut, ExecutionProvider, KernelFactory, LoraFactorInput,
    };
    use onnx_runtime_ir::{Attribute, DeviceId, NodeId, compute_contiguous_strides};

    use super::*;
    use crate::CudaExecutionProvider;

    fn bytes_of<T>(values: &[T]) -> &[u8] {
        // SAFETY: test values are plain numeric types and the byte slice retains
        // the source lifetime.
        unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        }
    }

    fn node(pool_identifier: LoraPoolId, input_width: usize, output_width: usize) -> Node {
        let mut node = Node::new(NodeId(0), OPERATOR, vec![None, None], vec![]);
        node.attributes
            .insert("K".into(), Attribute::Int(input_width as i64));
        node.attributes
            .insert("N".into(), Attribute::Int(output_width as i64));
        node.attributes
            .insert("module_id".into(), Attribute::Int(0));
        node.attributes.insert("max_rank".into(), Attribute::Int(8));
        node.attributes
            .insert("pool_id".into(), Attribute::Int(pool_identifier.0 as i64));
        node
    }

    fn reference_delta(
        input: &[f32],
        input_width: usize,
        output_width: usize,
        rank: usize,
        factor_a: &[f32],
        factor_b: &[f32],
        scale: f32,
    ) -> Vec<f32> {
        let rows = input.len() / input_width;
        let mut output = vec![0.0_f32; rows * output_width];
        for row in 0..rows {
            for output_column in 0..output_width {
                let mut output_accumulator = 0.0_f64;
                for rank_column in 0..rank {
                    let mut intermediate_accumulator = 0.0_f64;
                    for input_column in 0..input_width {
                        intermediate_accumulator += input[row * input_width + input_column] as f64
                            * factor_a[input_column * rank + rank_column] as f64;
                    }
                    output_accumulator += intermediate_accumulator
                        * factor_b[rank_column * output_width + output_column] as f64;
                }
                output[row * output_width + output_column] =
                    (output_accumulator * scale as f64) as f32;
            }
        }
        output
    }

    #[test]
    fn mixed_adapters_match_cpu_and_closed_form_on_device() {
        let Ok(execution_provider) = CudaExecutionProvider::initialized(0) else {
            eprintln!("skipping GroupedLoraDelta parity test: CUDA device unavailable");
            return;
        };
        let input_width = 4;
        let output_width = 3;
        let rank = 2;
        let factor_a0 = vec![-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let factor_b0 = vec![0.0, 0.05, 0.1, 0.15, 0.2, 0.25];
        let factor_a1 = vec![0.1, 0.03, -0.04, -0.11, -0.18, -0.25, -0.32, -0.39];
        let factor_b1 = vec![0.3, 0.28, 0.26, 0.24, 0.22, 0.2];
        let mut pool = LoraWeightPool::with_capacity_bytes(1 << 20);
        for (adapter, factor_a, factor_b, scale) in [
            (
                AdapterId(0),
                factor_a0.as_slice(),
                factor_b0.as_slice(),
                0.5,
            ),
            (
                AdapterId(1),
                factor_a1.as_slice(),
                factor_b1.as_slice(),
                1.5,
            ),
        ] {
            pool.admit(
                adapter,
                LoraModuleId(0),
                LoraFactorInput {
                    dtype: DataType::Float32,
                    rows: input_width,
                    cols: rank,
                    bytes: bytes_of(factor_a),
                },
                LoraFactorInput {
                    dtype: DataType::Float32,
                    rows: rank,
                    cols: output_width,
                    bytes: bytes_of(factor_b),
                },
                scale,
            )
            .unwrap();
        }
        let pool = Arc::new(pool);
        let registration = LoraPoolRegistry::global().register_owned(pool);
        let node = node(registration.pool_id(), input_width, output_width);
        let cuda_kernel = GroupedLoraDeltaFactory {
            runtime: execution_provider.runtime().clone(),
        }
        .create(&node, &[])
        .unwrap();
        let cpu_kernel = onnx_runtime_ep_cpu::kernels::grouped_lora::GroupedLoraDeltaFactory
            .create(&node, &[])
            .unwrap();

        let input: Vec<f32> = (0..4 * input_width)
            .map(|index| index as f32 * 0.3 - 1.0)
            .collect();
        let segments = [0_i32, 1, -1, 0];
        let input_shape = [4, input_width];
        let output_shape = [4, output_width];
        let segment_shape = [4];
        let input_strides = compute_contiguous_strides(&input_shape);
        let output_strides = compute_contiguous_strides(&output_shape);
        let segment_strides = compute_contiguous_strides(&segment_shape);

        let input_buffer = execution_provider
            .allocate(std::mem::size_of_val(input.as_slice()), 256)
            .unwrap();
        let segment_buffer = execution_provider
            .allocate(std::mem::size_of_val(segments.as_slice()), 256)
            .unwrap();
        let mut output_buffer = execution_provider
            .allocate(
                output_shape.iter().product::<usize>() * std::mem::size_of::<f32>(),
                256,
            )
            .unwrap();
        unsafe {
            execution_provider
                .runtime()
                .htod(bytes_of(&input), cuptr(input_buffer.as_ptr()))
                .unwrap();
            execution_provider
                .runtime()
                .htod(bytes_of(&segments), cuptr(segment_buffer.as_ptr()))
                .unwrap();
        }
        let cuda_inputs = [
            TensorView::new(
                DevicePtr(input_buffer.as_ptr()),
                DataType::Float32,
                &input_shape,
                &input_strides,
                execution_provider.device_id(),
            ),
            TensorView::new(
                DevicePtr(segment_buffer.as_ptr()),
                DataType::Int32,
                &segment_shape,
                &segment_strides,
                execution_provider.device_id(),
            ),
        ];
        let cuda_output = TensorMut::new(
            DevicePtrMut(output_buffer.as_mut_ptr()),
            DataType::Float32,
            &output_shape,
            &output_strides,
            execution_provider.device_id(),
        );
        cuda_kernel
            .execute(&cuda_inputs, &mut [cuda_output])
            .unwrap();
        let mut cuda_output_bytes =
            vec![0_u8; output_shape.iter().product::<usize>() * std::mem::size_of::<f32>()];
        unsafe {
            execution_provider
                .runtime()
                .dtoh(&mut cuda_output_bytes, cuptr(output_buffer.as_ptr()))
                .unwrap();
        }
        let cuda_values: Vec<f32> = cuda_output_bytes
            .chunks_exact(4)
            .map(|value| f32::from_ne_bytes(value.try_into().unwrap()))
            .collect();

        let mut cpu_values = vec![0.0_f32; output_shape.iter().product()];
        let cpu_inputs = [
            TensorView::new(
                DevicePtr(input.as_ptr().cast::<c_void>()),
                DataType::Float32,
                &input_shape,
                &input_strides,
                DeviceId::cpu(),
            ),
            TensorView::new(
                DevicePtr(segments.as_ptr().cast::<c_void>()),
                DataType::Int32,
                &segment_shape,
                &segment_strides,
                DeviceId::cpu(),
            ),
        ];
        let cpu_output = TensorMut::new(
            DevicePtrMut(cpu_values.as_mut_ptr().cast::<c_void>()),
            DataType::Float32,
            &output_shape,
            &output_strides,
            DeviceId::cpu(),
        );
        cpu_kernel.execute(&cpu_inputs, &mut [cpu_output]).unwrap();

        let mut closed_form = vec![0.0_f32; cpu_values.len()];
        for (row, segment) in segments.iter().copied().enumerate() {
            let expected = match segment {
                0 => reference_delta(
                    &input[row * input_width..(row + 1) * input_width],
                    input_width,
                    output_width,
                    rank,
                    &factor_a0,
                    &factor_b0,
                    0.5,
                ),
                1 => reference_delta(
                    &input[row * input_width..(row + 1) * input_width],
                    input_width,
                    output_width,
                    rank,
                    &factor_a1,
                    &factor_b1,
                    1.5,
                ),
                _ => vec![0.0; output_width],
            };
            closed_form[row * output_width..(row + 1) * output_width].copy_from_slice(&expected);
        }
        for (index, ((cuda_value, cpu_value), expected)) in cuda_values
            .iter()
            .zip(&cpu_values)
            .zip(&closed_form)
            .enumerate()
        {
            assert!(
                (cuda_value - cpu_value).abs() < 1e-5,
                "CUDA/CPU mismatch at {index}: CUDA={cuda_value}, CPU={cpu_value}"
            );
            assert!(
                (cuda_value - expected).abs() < 1e-5,
                "CUDA/closed-form mismatch at {index}: CUDA={cuda_value}, expected={expected}"
            );
        }
        execution_provider.deallocate(input_buffer).unwrap();
        execution_provider.deallocate(segment_buffer).unwrap();
        execution_provider.deallocate(output_buffer).unwrap();
    }
}
