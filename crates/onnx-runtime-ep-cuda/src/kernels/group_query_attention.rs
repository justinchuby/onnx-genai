//! CUDA implementation of `com.microsoft::GroupQueryAttention`.
//!
//! BSH query/key/value inputs are prepared into BNSH buffers with NVRTC kernels,
//! then the shared cuBLASLt + stable-softmax attention engine is used. Present
//! key/value outputs remain BNSH and preserve a fixed past-cache capacity.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::attention::run_attention_f32;
use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const PREP_SRC: &str = r#"
extern "C" __global__ void gqa_transpose_bsh_to_bnsh(
    const float* src, float* dst, int batch, int seq, int heads, int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch * heads * seq * dim;
    if (idx >= count) return;
    int x = idx;
    const int d = x % dim; x /= dim;
    const int s = x % seq; x /= seq;
    const int h = x % heads; const int b = x / heads;
    dst[idx] = src[((b * seq + s) * heads + h) * dim + d];
}

extern "C" __global__ void gqa_build_cache(
    const float* current, const float* past, float* present,
    const int* past_lengths, int batch, int seq, int heads, int dim,
    int past_capacity, int present_capacity)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch * heads * present_capacity * dim;
    if (idx >= count) return;
    int x = idx;
    const int d = x % dim; x /= dim;
    const int s = x % present_capacity; x /= present_capacity;
    const int h = x % heads; const int b = x / heads;
    const int past_len = past_lengths[b];
    float value = 0.0f;
    if (s < past_len && past) {
        value = past[((b * heads + h) * past_capacity + s) * dim + d];
    } else if (s >= past_len && s < past_len + seq) {
        const int current_s = s - past_len;
        value = current[((b * seq + current_s) * heads + h) * dim + d];
    }
    present[idx] = value;
}

extern "C" __global__ void gqa_rope_bnsh(
    float* tensor, const float* cos_cache, const float* sin_cache,
    const long long* position_ids, const int* past_lengths,
    int batch, int seq, int heads, int dim, int tensor_capacity,
    int current_offset, int cache_rows, int interleaved)
{
    const int half = dim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch * heads * seq * half;
    if (idx >= count) return;
    int x = idx;
    const int k = x % half; x /= half;
    const int s = x % seq; x /= seq;
    const int h = x % heads; const int b = x / heads;
    const int pos = position_ids
        ? (int)position_ids[b * seq + s]
        : past_lengths[b] + s;
    if (pos < 0 || pos >= cache_rows) return;
    const int d0 = interleaved ? 2 * k : k;
    const int d1 = interleaved ? 2 * k + 1 : k + half;
    const int tensor_s = current_offset ? past_lengths[b] + s : s;
    const size_t base = ((size_t)(b * heads + h) * tensor_capacity + tensor_s) * dim;
    const float x0 = tensor[base + d0];
    const float x1 = tensor[base + d1];
    const float c = cos_cache[pos * half + k];
    const float sn = sin_cache[pos * half + k];
    tensor[base + d0] = c * x0 - sn * x1;
    tensor[base + d1] = sn * x0 + c * x1;
}

extern "C" __global__ void gqa_transpose_bnsh_to_bsh(
    const float* src, float* dst, int batch, int seq, int heads, int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = batch * seq * heads * dim;
    if (idx >= count) return;
    int x = idx;
    const int d = x % dim; x /= dim;
    const int h = x % heads; x /= heads;
    const int s = x % seq; const int b = x / seq;
    dst[idx] = src[((b * heads + h) * seq + s) * dim + d];
}
"#;

const PREP_MODULE: &str = "group_query_attention_prep";
const BLOCK: u32 = 256;

pub struct GroupQueryAttentionFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for GroupQueryAttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let required_heads = |name: &str| -> Result<usize> {
            let value = node.attr(name).and_then(|a| a.as_int()).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: missing required `{name}` attribute"
                ))
            })?;
            usize::try_from(value)
                .ok()
                .filter(|&v| v > 0)
                .ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "cuda_ep GroupQueryAttention: `{name}` must be > 0"
                    ))
                })
        };
        let num_heads = required_heads("num_heads")?;
        let kv_num_heads = required_heads("kv_num_heads")?;
        if !num_heads.is_multiple_of(kv_num_heads) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GroupQueryAttention: num_heads {num_heads} must be a multiple of kv_num_heads {kv_num_heads}"
            )));
        }
        for name in ["k_quant_type", "v_quant_type"] {
            if let Some(value) = node.attr(name)
                && value.as_str() != Some("NONE")
            {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: `{name}` other than NONE is not supported"
                )));
            }
        }
        for (name, message) in [
            ("kv_cache_bit_width", "quantized KV cache"),
            ("qk_output", "qk_output"),
            ("smooth_softmax", "smooth_softmax"),
        ] {
            if node.attr(name).and_then(|a| a.as_int()).unwrap_or(0) != 0 {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: {message} is not supported"
                )));
            }
        }
        let softcap = node
            .attr("softcap")
            .and_then(|a| a.as_float())
            .unwrap_or(0.0);
        if softcap < 0.0 {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: softcap must be non-negative".into(),
            ));
        }
        Ok(Box::new(GroupQueryAttentionKernel {
            runtime: self.runtime.clone(),
            num_heads,
            kv_num_heads,
            scale: node.attr("scale").and_then(|a| a.as_float()),
            do_rotary: node.attr("do_rotary").and_then(|a| a.as_int()).unwrap_or(0) != 0,
            rotary_interleaved: node
                .attr("rotary_interleaved")
                .and_then(|a| a.as_int())
                .unwrap_or(0)
                != 0,
            local_window_size: node
                .attr("local_window_size")
                .and_then(|a| a.as_int())
                .unwrap_or(-1),
            softcap,
        }))
    }
}

#[derive(Debug)]
pub struct GroupQueryAttentionKernel {
    runtime: Arc<CudaRuntime>,
    num_heads: usize,
    kv_num_heads: usize,
    scale: Option<f32>,
    do_rotary: bool,
    rotary_interleaved: bool,
    local_window_size: i64,
    softcap: f32,
}

struct Scratch<'a> {
    runtime: &'a CudaRuntime,
    ptr: CUdeviceptr,
}

impl<'a> Scratch<'a> {
    fn new(runtime: &'a CudaRuntime, bytes: usize) -> Result<Self> {
        Ok(Self {
            runtime,
            ptr: runtime.alloc_raw(bytes)?,
        })
    }
}

impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        // SAFETY: `ptr` is uniquely owned by this guard.
        let _ = unsafe { self.runtime.free_raw(self.ptr) };
    }
}

fn checked_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        EpError::KernelFailed(format!(
            "cuda_ep GroupQueryAttention: {name} {value} exceeds i32"
        ))
    })
}

fn require_dense(view: &TensorView, name: &str, dtype: DataType) -> Result<()> {
    if view.dtype != dtype {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep GroupQueryAttention: {name} must have dtype {dtype:?}, got {:?}",
            view.dtype
        )));
    }
    if !view.is_contiguous() {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep GroupQueryAttention: non-contiguous {name} is not supported; materialise it first"
        )));
    }
    Ok(())
}

fn read_i32(runtime: &CudaRuntime, view: &TensorView, name: &str) -> Result<Vec<i32>> {
    require_dense(view, name, DataType::Int32)?;
    let mut bytes = vec![0u8; view.numel() * 4];
    // SAFETY: the source tensor has exactly `bytes.len()` bytes.
    unsafe {
        runtime.dtoh(&mut bytes, cuptr(view.data_ptr::<u8>() as *const c_void))?;
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|x| i32::from_ne_bytes([x[0], x[1], x[2], x[3]]))
        .collect())
}

fn read_i64(runtime: &CudaRuntime, view: &TensorView, name: &str) -> Result<Vec<i64>> {
    require_dense(view, name, DataType::Int64)?;
    let mut bytes = vec![0u8; view.numel() * 8];
    // SAFETY: the source tensor has exactly `bytes.len()` bytes.
    unsafe {
        runtime.dtoh(&mut bytes, cuptr(view.data_ptr::<u8>() as *const c_void))?;
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|x| i64::from_ne_bytes([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]]))
        .collect())
}

fn bytes_of_i32(values: &[i32]) -> &[u8] {
    // SAFETY: i32 has no padding and the returned slice borrows `values`.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

macro_rules! launch_1d {
    ($runtime:expr, $entry:expr, $count:expr, $builder:ident, $args:block) => {{
        let function = $runtime.nvrtc_function(PREP_MODULE, PREP_SRC, $entry)?;
        let grid = u32::try_from(($count).div_ceil(BLOCK as usize)).map_err(|_| {
            EpError::KernelFailed("cuda_ep GroupQueryAttention: launch grid exceeds u32".into())
        })?;
        let mut $builder = $runtime.stream().launch_builder(&function);
        $args
        // SAFETY: each invocation supplies the argument ABI for its entry point;
        // buffers and scalar arguments remain live through synchronization.
        unsafe {
            $builder.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err(&format!("launch {}", $entry), e))?;
    }};
}

impl GroupQueryAttentionKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if !(7..=14).contains(&inputs.len()) || !(1..=3).contains(&outputs.len()) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GroupQueryAttention: expected 7..14 inputs and 1..3 outputs, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        if inputs[1].is_absent() || inputs[2].is_absent() {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: packed QKV/packed KV is not supported; provide unpacked query, key, and value".into(),
            ));
        }
        for (index, feature) in [
            (10, "attention_bias"),
            (11, "head_sink"),
            (12, "quantized-cache k_scale"),
            (13, "quantized-cache v_scale"),
        ] {
            if inputs.get(index).is_some_and(|v| !v.is_absent()) {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: {feature} is not supported"
                )));
            }
        }
        if self.local_window_size == 0 || self.local_window_size < -1 {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: local_window_size must be -1 or a positive integer"
                    .into(),
            ));
        }

        let (q, k, v) = (&inputs[0], &inputs[1], &inputs[2]);
        for (view, name) in [(q, "query"), (k, "key"), (v, "value")] {
            require_dense(view, name, DataType::Float32)?;
            if view.shape.len() != 3 {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: unpacked {name} must be rank 3 [B,S,H*D], got {:?}",
                    view.shape
                )));
            }
        }
        let (batch, q_seq, q_hidden) = (q.shape[0], q.shape[1], q.shape[2]);
        let (k_batch, k_seq, k_hidden) = (k.shape[0], k.shape[1], k.shape[2]);
        if batch == 0
            || q_seq == 0
            || k_seq == 0
            || q_hidden == 0
            || k_hidden == 0
            || !q_hidden.is_multiple_of(self.num_heads)
            || !k_hidden.is_multiple_of(self.kv_num_heads)
            || v.shape != [batch, k_seq, k_hidden]
            || k_batch != batch
        {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: incompatible query/key/value batch, sequence, or hidden dimensions".into(),
            ));
        }
        let dim = q_hidden / self.num_heads;
        if k_hidden / self.kv_num_heads != dim {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: query and key/value head sizes must match".into(),
            ));
        }
        if outputs[0].dtype != DataType::Float32
            || outputs[0].shape != [batch, q_seq, q_hidden]
            || !outputs[0].is_contiguous()
        {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GroupQueryAttention: output must be contiguous f32 [B,S,H*D] = [{batch},{q_seq},{q_hidden}], got {:?}",
                outputs[0].shape
            )));
        }

        let has_past_key = !inputs[3].is_absent();
        let has_past_value = !inputs[4].is_absent();
        if has_past_key != has_past_value {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: past_key and past_value must be provided together"
                    .into(),
            ));
        }
        let past_capacity = if has_past_key {
            for (view, name) in [(&inputs[3], "past_key"), (&inputs[4], "past_value")] {
                require_dense(view, name, DataType::Float32)?;
                if view.shape.len() != 4
                    || view.shape[0] != batch
                    || view.shape[1] != self.kv_num_heads
                    || view.shape[3] != dim
                {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep GroupQueryAttention: {name} must be BNSH [{batch},{},{},{}], got {:?}",
                        self.kv_num_heads, view.shape[2], dim, view.shape
                    )));
                }
            }
            if inputs[3].shape != inputs[4].shape {
                return Err(EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: past_key and past_value shapes must match".into(),
                ));
            }
            inputs[3].shape[2]
        } else {
            0
        };

        let seqlens = read_i32(&self.runtime, &inputs[5], "seqlens_k")?;
        if inputs[5].shape != [batch] || seqlens.iter().any(|&x| x < 0) {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: seqlens_k must be non-negative int32 [batch_size]"
                    .into(),
            ));
        }
        let total_scalar = read_i32(&self.runtime, &inputs[6], "total_sequence_length")?;
        if total_scalar.len() != 1 || total_scalar[0] < 0 {
            return Err(EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: total_sequence_length must be one non-negative int32 scalar".into(),
            ));
        }
        let total_sequence_length = total_scalar[0] as usize;
        let totals: Vec<i32> = seqlens
            .iter()
            .map(|&length| length.checked_add(1))
            .collect::<Option<_>>()
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: seqlens_k + 1 overflows int32".into(),
                )
            })?;
        if totals.iter().copied().max().unwrap_or(0) as usize != total_sequence_length {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep GroupQueryAttention: total_sequence_length {total_sequence_length} must equal max(seqlens_k + 1)"
            )));
        }
        let mut past_lengths = Vec::with_capacity(batch);
        for &total in &totals {
            let past = total.checked_sub(checked_i32(k_seq, "key sequence length")?).ok_or_else(
                || {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: seqlens_k + 1 is shorter than current key sequence".into(),
                    )
                },
            )?;
            if past as usize > past_capacity {
                return Err(EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: effective past length exceeds past cache extent"
                        .into(),
                ));
            }
            past_lengths.push(past);
        }
        let present_capacity = past_capacity.max(total_sequence_length);
        let expected_cache_shape = [batch, self.kv_num_heads, present_capacity, dim];
        for (index, name) in [(1, "present_key"), (2, "present_value")] {
            if let Some(output) = outputs.get(index)
                && (output.dtype != DataType::Float32
                    || output.shape != expected_cache_shape
                    || !output.is_contiguous())
            {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: {name} must be contiguous f32 BNSH {:?}, got {:?}",
                    expected_cache_shape, output.shape
                )));
            }
        }

        let explicit_positions = inputs.get(9).filter(|view| !view.is_absent());
        let (cos_ptr, sin_ptr, positions_ptr, cache_rows) = if self.do_rotary {
            if !dim.is_multiple_of(2) {
                return Err(EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: do_rotary requires an even head_size".into(),
                ));
            }
            if q_seq != k_seq {
                return Err(EpError::KernelFailed(
                    "cuda_ep GroupQueryAttention: do_rotary requires equal query/key sequence lengths".into(),
                ));
            }
            let cos = inputs
                .get(7)
                .filter(|view| !view.is_absent())
                .ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: do_rotary=1 requires cos_cache".into(),
                    )
                })?;
            let sin = inputs
                .get(8)
                .filter(|view| !view.is_absent())
                .ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: do_rotary=1 requires sin_cache".into(),
                    )
                })?;
            require_dense(cos, "cos_cache", DataType::Float32)?;
            require_dense(sin, "sin_cache", DataType::Float32)?;
            if cos.shape.len() != 2 || sin.shape != cos.shape || cos.shape[1] != dim / 2 {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep GroupQueryAttention: cos_cache/sin_cache must have shape [max_sequence_length,{}]",
                    dim / 2
                )));
            }
            let position_ptr = if let Some(position_ids) = explicit_positions {
                let ids = read_i64(&self.runtime, position_ids, "position_ids")?;
                if position_ids.shape != [batch, q_seq]
                    || ids
                        .iter()
                        .any(|&position| position < 0 || position as usize >= cos.shape[0])
                {
                    return Err(EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: position_ids must be valid non-negative int64 [batch_size, sequence_length]".into(),
                    ));
                }
                cuptr(position_ids.data_ptr::<u8>() as *const c_void)
            } else {
                if past_lengths
                    .iter()
                    .any(|&past| past as usize + q_seq > cos.shape[0])
                {
                    return Err(EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: rotary position exceeds cache rows".into(),
                    ));
                }
                0
            };
            (
                cuptr(cos.data_ptr::<u8>() as *const c_void),
                cuptr(sin.data_ptr::<u8>() as *const c_void),
                position_ptr,
                checked_i32(cos.shape[0], "rotary cache rows")?,
            )
        } else {
            (0, 0, 0, 0)
        };

        let totals_gpu = Scratch::new(&self.runtime, totals.len() * 4)?;
        let past_lengths_gpu = Scratch::new(&self.runtime, past_lengths.len() * 4)?;
        // SAFETY: scratch allocations exactly match the uploaded slices.
        unsafe {
            self.runtime.htod(bytes_of_i32(&totals), totals_gpu.ptr)?;
            self.runtime
                .htod(bytes_of_i32(&past_lengths), past_lengths_gpu.ptr)?;
        }
        let q_bnsh = Scratch::new(&self.runtime, q.numel() * 4)?;
        let out_bnsh = Scratch::new(&self.runtime, outputs[0].numel() * 4)?;
        let owned_present_k = (outputs.len() < 2)
            .then(|| {
                Scratch::new(
                    &self.runtime,
                    expected_cache_shape.iter().product::<usize>() * 4,
                )
            })
            .transpose()?;
        let owned_present_v = (outputs.len() < 3)
            .then(|| {
                Scratch::new(
                    &self.runtime,
                    expected_cache_shape.iter().product::<usize>() * 4,
                )
            })
            .transpose()?;
        let present_k_ptr = if let Some(output) = outputs.get_mut(1) {
            cuptr(output.data_ptr_mut::<u8>() as *const c_void)
        } else {
            owned_present_k
                .as_ref()
                .ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: internal present-key allocation missing"
                            .into(),
                    )
                })?
                .ptr
        };
        let present_v_ptr = if let Some(output) = outputs.get_mut(2) {
            cuptr(output.data_ptr_mut::<u8>() as *const c_void)
        } else {
            owned_present_v
                .as_ref()
                .ok_or_else(|| {
                    EpError::KernelFailed(
                        "cuda_ep GroupQueryAttention: internal present-value allocation missing"
                            .into(),
                    )
                })?
                .ptr
        };

        let batch_i = checked_i32(batch, "batch")?;
        let q_seq_i = checked_i32(q_seq, "query sequence length")?;
        let k_seq_i = checked_i32(k_seq, "key sequence length")?;
        let heads_i = checked_i32(self.num_heads, "num_heads")?;
        let kv_heads_i = checked_i32(self.kv_num_heads, "kv_num_heads")?;
        let dim_i = checked_i32(dim, "head_size")?;
        let past_capacity_i = checked_i32(past_capacity, "past capacity")?;
        let present_capacity_i = checked_i32(present_capacity, "present capacity")?;
        let local_window_i = i32::try_from(self.local_window_size.max(0)).map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep GroupQueryAttention: local_window_size exceeds i32".into(),
            )
        })?;
        let q_ptr = cuptr(q.data_ptr::<u8>() as *const c_void);
        launch_1d!(
            self.runtime,
            "gqa_transpose_bsh_to_bnsh",
            q.numel(),
            builder,
            {
                builder
                    .arg(&q_ptr)
                    .arg(&q_bnsh.ptr)
                    .arg(&batch_i)
                    .arg(&q_seq_i)
                    .arg(&heads_i)
                    .arg(&dim_i);
            }
        );

        let past_k_ptr = has_past_key
            .then(|| cuptr(inputs[3].data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let past_v_ptr = has_past_value
            .then(|| cuptr(inputs[4].data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        for (current, past, present) in [
            (
                cuptr(k.data_ptr::<u8>() as *const c_void),
                past_k_ptr,
                present_k_ptr,
            ),
            (
                cuptr(v.data_ptr::<u8>() as *const c_void),
                past_v_ptr,
                present_v_ptr,
            ),
        ] {
            launch_1d!(
                self.runtime,
                "gqa_build_cache",
                expected_cache_shape.iter().product::<usize>(),
                builder,
                {
                    builder
                        .arg(&current)
                        .arg(&past)
                        .arg(&present)
                        .arg(&past_lengths_gpu.ptr)
                        .arg(&batch_i)
                        .arg(&k_seq_i)
                        .arg(&kv_heads_i)
                        .arg(&dim_i)
                        .arg(&past_capacity_i)
                        .arg(&present_capacity_i);
                }
            );
        }

        if self.do_rotary {
            let interleaved_i: i32 = self.rotary_interleaved.into();
            for (tensor, seq_i, heads, capacity, current_offset) in [
                (q_bnsh.ptr, q_seq_i, heads_i, q_seq_i, 0i32),
                (present_k_ptr, k_seq_i, kv_heads_i, present_capacity_i, 1i32),
            ] {
                let count = batch * (heads as usize) * (seq_i as usize) * (dim / 2);
                launch_1d!(self.runtime, "gqa_rope_bnsh", count, builder, {
                    builder
                        .arg(&tensor)
                        .arg(&cos_ptr)
                        .arg(&sin_ptr)
                        .arg(&positions_ptr)
                        .arg(&past_lengths_gpu.ptr)
                        .arg(&batch_i)
                        .arg(&seq_i)
                        .arg(&heads)
                        .arg(&dim_i)
                        .arg(&capacity)
                        .arg(&current_offset)
                        .arg(&cache_rows)
                        .arg(&interleaved_i);
                });
            }
        }

        let scale = self
            .scale
            .filter(|&scale| scale != 0.0)
            .unwrap_or_else(|| 1.0 / (dim as f32).sqrt());
        run_attention_f32(
            &self.runtime,
            self.num_heads,
            self.kv_num_heads,
            true,
            batch,
            q_seq,
            total_sequence_length,
            dim,
            present_capacity,
            self.num_heads / self.kv_num_heads,
            scale,
            q_bnsh.ptr,
            present_k_ptr,
            present_v_ptr,
            out_bnsh.ptr,
            0,
            0,
            totals_gpu.ptr,
            past_lengths_gpu.ptr,
            local_window_i,
            self.softcap,
        )?;

        let output_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        launch_1d!(
            self.runtime,
            "gqa_transpose_bnsh_to_bsh",
            outputs[0].numel(),
            builder,
            {
                builder
                    .arg(&out_bnsh.ptr)
                    .arg(&output_ptr)
                    .arg(&batch_i)
                    .arg(&q_seq_i)
                    .arg(&heads_i)
                    .arg(&dim_i);
            }
        );
        self.runtime.synchronize()
    }
}

impl Kernel for GroupQueryAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}
