//! CUDA parity tests for the standard `Attention` operator in bf16 storage.
//!
//! Mirrors the fp16 harness ([`standard_attention_fp16_gpu`]): the CUDA kernel
//! runs on bf16 Q/K/V (plus bf16 additive mask and bf16 present-KV/output) and
//! is checked against a CPU f32 oracle built from the *exact* bf16-rounded
//! inputs, so the only expected divergence is bf16 storage round-off — not a
//! different arithmetic path. Every bulk output is checked: `Y`, both
//! `present_key`/`present_value` cache tensors, and the `qk_matmul_output`.
//!
//! A dedicated mutation guard ([`standard_attention_bf16_accumulates_in_fp32`])
//! constructs an ill-conditioned `probs·V` reduction whose value collapses if
//! the kernel accumulated in bf16 instead of fp32, and asserts the CUDA result
//! tracks the fp32 oracle rather than the bf16-accumulation oracle.

use half::bf16;
use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, Result, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

struct Tensor {
    dtype: DataType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn raw<T: Copy>(values: &[T]) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)).to_vec()
    }
}

fn tensor<T: Copy>(dtype: DataType, shape: &[usize], values: &[T]) -> Tensor {
    Tensor {
        dtype,
        shape: shape.to_vec(),
        bytes: raw(values),
    }
}

/// Run an op on CUDA. Individual input slots may be `None` to model an omitted
/// optional ONNX input (an empty-string input name → an absent [`TensorView`]).
fn run_opt(
    op: &str,
    opset: u64,
    inputs: &[Option<Tensor>],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> Vec<Vec<u8>> {
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let input_values = inputs
        .iter()
        .enumerate()
        .map(|(i, input)| {
            input.as_ref().map(|input| {
                let value = graph.create_named_value(
                    format!("input_{i}"),
                    input.dtype,
                    static_shape(input.shape.iter().copied()),
                );
                graph.add_input(value);
                value
            })
        })
        .collect::<Vec<_>>();
    let output_values = outputs
        .iter()
        .enumerate()
        .map(|(i, (dtype, shape))| {
            graph.create_named_value(
                format!("output_{i}"),
                *dtype,
                static_shape(shape.iter().copied()),
            )
        })
        .collect::<Vec<_>>();
    let mut node = Node::new(
        NodeId(0),
        op,
        input_values.into_iter().collect(),
        output_values.clone(),
    );
    for (name, value) in attrs {
        node.attributes.insert((*name).into(), value.clone());
    }
    let node_id = graph.insert_node(node);
    for output in output_values {
        graph.add_output(output);
    }
    let model = Model::new(&graph);
    let kernel = ep
        .get_kernel(model.graph.node(node_id), &[], opset)
        .expect("kernel");

    let input_buffers = inputs
        .iter()
        .map(|input| -> Result<Option<DeviceBuffer>> {
            let Some(input) = input else {
                return Ok(None);
            };
            let buffer = ep.allocate(input.bytes.len(), 256)?;
            if !input.bytes.is_empty() {
                unsafe { ep.runtime().htod(&input.bytes, cuptr(buffer.as_ptr()))? };
            }
            Ok(Some(buffer))
        })
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let input_strides = inputs
        .iter()
        .map(|input| {
            input
                .as_ref()
                .map(|input| compute_contiguous_strides(&input.shape))
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    let input_views = inputs
        .iter()
        .zip(&input_buffers)
        .zip(&input_strides)
        .map(|((input, buffer), strides)| match (input, buffer) {
            (Some(input), Some(buffer)) => TensorView::new(
                DevicePtr(buffer.as_ptr()),
                input.dtype,
                &input.shape,
                strides,
                ep.device_id(),
            ),
            _ => TensorView::absent(DataType::Float32),
        })
        .collect::<Vec<_>>();
    let mut output_buffers = outputs
        .iter()
        .map(|(dtype, shape)| -> Result<DeviceBuffer> {
            ep.allocate(dtype.storage_bytes(shape.iter().product()), 256)
        })
        .collect::<Result<Vec<DeviceBuffer>>>()
        .unwrap();
    let output_strides = outputs
        .iter()
        .map(|(_, shape)| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();
    let mut output_views = outputs
        .iter()
        .zip(output_buffers.iter_mut())
        .zip(&output_strides)
        .map(|(((dtype, shape), buffer), strides)| {
            TensorMut::new(
                DevicePtrMut(buffer.as_mut_ptr()),
                *dtype,
                shape,
                strides,
                ep.device_id(),
            )
        })
        .collect::<Vec<_>>();
    kernel.execute(&input_views, &mut output_views).unwrap();

    let result = outputs
        .iter()
        .zip(&output_buffers)
        .map(|((dtype, shape), buffer)| -> Result<Vec<u8>> {
            let mut bytes = vec![0; dtype.storage_bytes(shape.iter().product())];
            if !bytes.is_empty() {
                unsafe { ep.runtime().dtoh(&mut bytes, cuptr(buffer.as_ptr()))? };
            }
            Ok(bytes)
        })
        .collect::<Result<Vec<_>>>()
        .unwrap();
    for buffer in input_buffers.into_iter().flatten() {
        ep.deallocate(buffer).unwrap();
    }
    for buffer in output_buffers {
        ep.deallocate(buffer).unwrap();
    }
    result
}

fn run_cpu_opt(
    op: &str,
    opset: u64,
    inputs: &[Option<Tensor>],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> Vec<Vec<u8>> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let input_values = inputs
        .iter()
        .enumerate()
        .map(|(i, input)| {
            input.as_ref().map(|input| {
                let value = graph.create_named_value(
                    format!("input_{i}"),
                    input.dtype,
                    static_shape(input.shape.iter().copied()),
                );
                graph.add_input(value);
                value
            })
        })
        .collect::<Vec<_>>();
    let output_values = outputs
        .iter()
        .enumerate()
        .map(|(i, (dtype, shape))| {
            graph.create_named_value(
                format!("output_{i}"),
                *dtype,
                static_shape(shape.iter().copied()),
            )
        })
        .collect::<Vec<_>>();
    let mut node = Node::new(
        NodeId(0),
        op,
        input_values.into_iter().collect(),
        output_values.clone(),
    );
    for (name, value) in attrs {
        node.attributes.insert((*name).into(), value.clone());
    }
    let node_id = graph.insert_node(node);
    for output in output_values {
        graph.add_output(output);
    }
    let model = Model::new(&graph);
    let kernel = CpuExecutionProvider::new()
        .get_kernel(model.graph.node(node_id), &[], opset)
        .unwrap();

    let input_strides = inputs
        .iter()
        .map(|input| {
            input
                .as_ref()
                .map(|input| compute_contiguous_strides(&input.shape))
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    let input_views = inputs
        .iter()
        .zip(&input_strides)
        .map(|(input, strides)| match input {
            Some(input) => TensorView::new(
                DevicePtr(input.bytes.as_ptr().cast()),
                input.dtype,
                &input.shape,
                strides,
                DeviceId::cpu(),
            ),
            None => TensorView::absent(DataType::Float32),
        })
        .collect::<Vec<_>>();
    let mut output_buffers = outputs
        .iter()
        .map(|(dtype, shape)| vec![0; dtype.storage_bytes(shape.iter().product())])
        .collect::<Vec<_>>();
    let output_strides = outputs
        .iter()
        .map(|(_, shape)| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();
    let mut output_views = outputs
        .iter()
        .zip(output_buffers.iter_mut())
        .zip(&output_strides)
        .map(|(((dtype, shape), bytes), strides)| {
            TensorMut::new(
                DevicePtrMut(bytes.as_mut_ptr().cast()),
                *dtype,
                shape,
                strides,
                DeviceId::cpu(),
            )
        })
        .collect::<Vec<_>>();
    kernel.execute(&input_views, &mut output_views).unwrap();
    drop(output_views);
    output_buffers
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
        .collect()
}

fn bf16s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|v| bf16::from_bits(u16::from_ne_bytes(v.try_into().unwrap())).to_f32())
        .collect()
}

fn round_bf16(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .copied()
        .map(|value| bf16::from_f32(value).to_f32())
        .collect()
}

fn bf16_tensor(shape: &[usize], rounded_values: &[f32]) -> Tensor {
    let values = rounded_values
        .iter()
        .copied()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    tensor(DataType::BFloat16, shape, &values)
}

/// Compare a bf16 GPU output against an f32 oracle. Large-magnitude negatives
/// (masked `qk_matmul_output` entries where a big additive bias dominates) are
/// re-rounded to bf16 before comparison because their absolute round-off is
/// intrinsically coarse and carries no useful signal.
fn assert_bf16_matches_f32(label: &str, got: &[u8], expected: &[u8], tolerance: f32) -> f32 {
    let got = bf16s(got);
    let expected = f32s(expected);
    assert_eq!(got.len(), expected.len(), "{label}: element count");
    let mut maximum_error = 0.0_f32;
    for (index, (got, expected)) in got.iter().zip(&expected).enumerate() {
        if expected.is_infinite() {
            assert_eq!(
                got.is_infinite() && got.is_sign_negative(),
                expected.is_sign_negative(),
                "{label}[{index}]: expected {expected}, got {got}"
            );
        } else {
            let expected = if *expected <= -1000.0 {
                bf16::from_f32(*expected).to_f32()
            } else {
                *expected
            };
            let error = (got - expected).abs();
            maximum_error = maximum_error.max(error);
            assert!(
                error <= tolerance,
                "{label}[{index}]: error={error:e}, tolerance={tolerance:e}, expected={expected:e}, got={got:e}"
            );
        }
    }
    maximum_error
}

#[test]
fn standard_attention_bf16_matches_exact_rounded_f32_oracle() {
    let Ok(_ep) = CudaExecutionProvider::new_default() else {
        eprintln!("skip: no CUDA GPU available");
        return;
    };

    const QUERY_HEADS: usize = 4;
    const KEY_VALUE_HEADS: usize = 2;
    const SEQUENCE: usize = 32;
    const HEAD_SIZE: usize = 64;
    // bf16 carries an 8-bit significand (~8x the fp16 ulp), so the exact-input
    // parity bound is ~8x the fp16 test's 3e-4. The observed max abs error is
    // ~1e-3 (printed below); 1e-2 leaves an order-of-magnitude of headroom
    // without masking an accidental precision regression.
    const TOLERANCE: f32 = 1e-2;
    let query = round_bf16(
        &(0..QUERY_HEADS * SEQUENCE * HEAD_SIZE)
            .map(|index| 0.1 + (index * 17 % 97) as f32 / 1024.0)
            .collect::<Vec<_>>(),
    );
    let key = round_bf16(
        &(0..KEY_VALUE_HEADS * SEQUENCE * HEAD_SIZE)
            .map(|index| 0.1 + (index * 29 % 101) as f32 / 1024.0)
            .collect::<Vec<_>>(),
    );
    let value = round_bf16(
        &(0..KEY_VALUE_HEADS * SEQUENCE * HEAD_SIZE)
            .map(|index| ((index * 43 % 113) as f32 - 56.0) / 64.0)
            .collect::<Vec<_>>(),
    );
    let additive_mask = round_bf16(
        &(0..SEQUENCE * SEQUENCE)
            .map(|index| match (index / SEQUENCE, index % SEQUENCE) {
                (row, column) if column > row => f32::NEG_INFINITY,
                (_, 7 | 23) => -30_000.0,
                (row, column) if (row + column).is_multiple_of(11) => -0.5,
                _ => 0.0,
            })
            .collect::<Vec<_>>(),
    );
    let attrs = [
        ("q_num_heads", Attribute::Int(QUERY_HEADS as i64)),
        ("kv_num_heads", Attribute::Int(KEY_VALUE_HEADS as i64)),
        ("is_causal", Attribute::Int(0)),
        ("qk_matmul_output_mode", Attribute::Int(2)),
    ];
    let prefill_outputs_f32 = [
        (DataType::Float32, vec![1, QUERY_HEADS, SEQUENCE, HEAD_SIZE]),
        (
            DataType::Float32,
            vec![1, KEY_VALUE_HEADS, SEQUENCE, HEAD_SIZE],
        ),
        (
            DataType::Float32,
            vec![1, KEY_VALUE_HEADS, SEQUENCE, HEAD_SIZE],
        ),
        (DataType::Float32, vec![1, QUERY_HEADS, SEQUENCE, SEQUENCE]),
    ];
    let prefill_outputs_bf16 = prefill_outputs_f32
        .iter()
        .map(|(_, shape)| (DataType::BFloat16, shape.clone()))
        .collect::<Vec<_>>();
    let cpu_prefill = run_cpu_opt(
        "Attention",
        23,
        &[
            Some(tensor(
                DataType::Float32,
                &[1, QUERY_HEADS, SEQUENCE, HEAD_SIZE],
                &query,
            )),
            Some(tensor(
                DataType::Float32,
                &[1, KEY_VALUE_HEADS, SEQUENCE, HEAD_SIZE],
                &key,
            )),
            Some(tensor(
                DataType::Float32,
                &[1, KEY_VALUE_HEADS, SEQUENCE, HEAD_SIZE],
                &value,
            )),
            Some(tensor(
                DataType::Float32,
                &[1, 1, SEQUENCE, SEQUENCE],
                &additive_mask,
            )),
        ],
        &prefill_outputs_f32,
        &attrs,
    );
    let gpu_prefill = run_opt(
        "Attention",
        23,
        &[
            Some(bf16_tensor(&[1, QUERY_HEADS, SEQUENCE, HEAD_SIZE], &query)),
            Some(bf16_tensor(
                &[1, KEY_VALUE_HEADS, SEQUENCE, HEAD_SIZE],
                &key,
            )),
            Some(bf16_tensor(
                &[1, KEY_VALUE_HEADS, SEQUENCE, HEAD_SIZE],
                &value,
            )),
            Some(bf16_tensor(&[1, 1, SEQUENCE, SEQUENCE], &additive_mask)),
        ],
        &prefill_outputs_bf16,
        &attrs,
    );
    let prefill_y_error =
        assert_bf16_matches_f32("prefill Y", &gpu_prefill[0], &cpu_prefill[0], TOLERANCE);
    let prefill_key_error = assert_bf16_matches_f32(
        "prefill present_key",
        &gpu_prefill[1],
        &cpu_prefill[1],
        TOLERANCE,
    );
    let prefill_value_error = assert_bf16_matches_f32(
        "prefill present_value",
        &gpu_prefill[2],
        &cpu_prefill[2],
        TOLERANCE,
    );
    let prefill_qk_error =
        assert_bf16_matches_f32("prefill QK", &gpu_prefill[3], &cpu_prefill[3], TOLERANCE);

    let decode_query = &query[..QUERY_HEADS * HEAD_SIZE];
    let decode_key = &key[..KEY_VALUE_HEADS * HEAD_SIZE];
    let decode_value = &value[..KEY_VALUE_HEADS * HEAD_SIZE];
    let past_key = round_bf16(
        &(0..KEY_VALUE_HEADS * (SEQUENCE - 1) * HEAD_SIZE)
            .map(|index| 0.1 + (index * 31 % 103) as f32 / 1024.0)
            .collect::<Vec<_>>(),
    );
    let past_value = round_bf16(
        &(0..KEY_VALUE_HEADS * (SEQUENCE - 1) * HEAD_SIZE)
            .map(|index| ((index * 47 % 109) as f32 - 54.0) / 64.0)
            .collect::<Vec<_>>(),
    );
    let decode_outputs_f32 = [
        (DataType::Float32, vec![1, QUERY_HEADS, 1, HEAD_SIZE]),
        (
            DataType::Float32,
            vec![1, KEY_VALUE_HEADS, SEQUENCE, HEAD_SIZE],
        ),
        (
            DataType::Float32,
            vec![1, KEY_VALUE_HEADS, SEQUENCE, HEAD_SIZE],
        ),
        (DataType::Float32, vec![1, QUERY_HEADS, 1, SEQUENCE]),
    ];
    let decode_outputs_bf16 = decode_outputs_f32
        .iter()
        .map(|(_, shape)| (DataType::BFloat16, shape.clone()))
        .collect::<Vec<_>>();
    let cpu_decode = run_cpu_opt(
        "Attention",
        23,
        &[
            Some(tensor(
                DataType::Float32,
                &[1, QUERY_HEADS, 1, HEAD_SIZE],
                decode_query,
            )),
            Some(tensor(
                DataType::Float32,
                &[1, KEY_VALUE_HEADS, 1, HEAD_SIZE],
                decode_key,
            )),
            Some(tensor(
                DataType::Float32,
                &[1, KEY_VALUE_HEADS, 1, HEAD_SIZE],
                decode_value,
            )),
            None,
            Some(tensor(
                DataType::Float32,
                &[1, KEY_VALUE_HEADS, SEQUENCE - 1, HEAD_SIZE],
                &past_key,
            )),
            Some(tensor(
                DataType::Float32,
                &[1, KEY_VALUE_HEADS, SEQUENCE - 1, HEAD_SIZE],
                &past_value,
            )),
        ],
        &decode_outputs_f32,
        &attrs,
    );
    let gpu_decode = run_opt(
        "Attention",
        23,
        &[
            Some(bf16_tensor(&[1, QUERY_HEADS, 1, HEAD_SIZE], decode_query)),
            Some(bf16_tensor(&[1, KEY_VALUE_HEADS, 1, HEAD_SIZE], decode_key)),
            Some(bf16_tensor(
                &[1, KEY_VALUE_HEADS, 1, HEAD_SIZE],
                decode_value,
            )),
            None,
            Some(bf16_tensor(
                &[1, KEY_VALUE_HEADS, SEQUENCE - 1, HEAD_SIZE],
                &past_key,
            )),
            Some(bf16_tensor(
                &[1, KEY_VALUE_HEADS, SEQUENCE - 1, HEAD_SIZE],
                &past_value,
            )),
        ],
        &decode_outputs_bf16,
        &attrs,
    );
    let decode_y_error =
        assert_bf16_matches_f32("decode Y", &gpu_decode[0], &cpu_decode[0], TOLERANCE);
    let decode_key_error = assert_bf16_matches_f32(
        "decode present_key",
        &gpu_decode[1],
        &cpu_decode[1],
        TOLERANCE,
    );
    let decode_value_error = assert_bf16_matches_f32(
        "decode present_value",
        &gpu_decode[2],
        &cpu_decode[2],
        TOLERANCE,
    );
    let decode_qk_error =
        assert_bf16_matches_f32("decode QK", &gpu_decode[3], &cpu_decode[3], TOLERANCE);

    eprintln!(
        "bf16 Attention exact-input max_abs: prefill_y={prefill_y_error:e}, prefill_key={prefill_key_error:e}, prefill_value={prefill_value_error:e}, prefill_qk={prefill_qk_error:e}, decode_y={decode_y_error:e}, decode_key={decode_key_error:e}, decode_value={decode_value_error:e}, decode_qk={decode_qk_error:e}"
    );
}

/// Mutation guard: prove the kernel accumulates `probs·V` in fp32, not bf16.
///
/// The scenario forces a uniform softmax over `SEQUENCE` keys whose per-key
/// contribution (`VALUE / SEQUENCE`) is far below the bf16 ulp near the running
/// total, so a bf16 accumulator stalls well short of the true sum while an fp32
/// accumulator reaches it. The test asserts (1) the fp32 and bf16-accumulation
/// oracles are separated by a wide margin (the scenario is genuinely sensitive)
/// and (2) the CUDA result tracks the fp32 oracle, catching any mutation of the
/// `float acc` accumulators in `attention_row` to the storage (bf16) type.
#[test]
fn standard_attention_bf16_accumulates_in_fp32() {
    let Ok(_ep) = CudaExecutionProvider::new_default() else {
        eprintln!("skip: no CUDA GPU available");
        return;
    };

    const SEQUENCE: usize = 1024;
    const HEAD_SIZE: usize = 8;
    const VALUE_HEAD_SIZE: usize = 4;
    const VALUE: f32 = 100.0;

    // Constant Q and K make every score equal, so the softmax is exactly
    // uniform (probs = 1/SEQUENCE). VALUE is bf16-exact, so the only variable
    // under test is how the probs·V reduction accumulates.
    let query = vec![0.1_f32; HEAD_SIZE];
    let key = vec![0.1_f32; SEQUENCE * HEAD_SIZE];
    let value = vec![VALUE; SEQUENCE * VALUE_HEAD_SIZE];
    assert_eq!(bf16::from_f32(VALUE).to_f32(), VALUE, "VALUE must be bf16-exact");

    let attrs = [
        ("q_num_heads", Attribute::Int(1)),
        ("kv_num_heads", Attribute::Int(1)),
        ("is_causal", Attribute::Int(0)),
    ];
    let outputs_bf16 = [(DataType::BFloat16, vec![1, 1, 1, VALUE_HEAD_SIZE])];
    let gpu = run_opt(
        "Attention",
        23,
        &[
            Some(bf16_tensor(&[1, 1, 1, HEAD_SIZE], &query)),
            Some(bf16_tensor(&[1, 1, SEQUENCE, HEAD_SIZE], &key)),
            Some(bf16_tensor(&[1, 1, SEQUENCE, VALUE_HEAD_SIZE], &value)),
        ],
        &outputs_bf16,
        &attrs,
    );
    let gpu = bf16s(&gpu[0]);

    // fp32-accumulation oracle: uniform probs over a constant channel sum back
    // to VALUE exactly.
    let probability = 1.0_f32 / SEQUENCE as f32;
    let fp32_oracle = VALUE;
    // bf16-accumulation oracle: the same reduction with the running total
    // rounded to bf16 after every add — the mutation this test guards against.
    let mut bf16_accumulator = 0.0_f32;
    for _ in 0..SEQUENCE {
        bf16_accumulator = bf16::from_f32(bf16_accumulator + probability * VALUE).to_f32();
    }

    let oracle_gap = (fp32_oracle - bf16_accumulator).abs();
    eprintln!(
        "bf16 accumulation guard: fp32_oracle={fp32_oracle}, bf16_accum_oracle={bf16_accumulator}, gap={oracle_gap}, gpu={:?}",
        gpu
    );

    // The scenario must be genuinely sensitive to the accumulation dtype.
    assert!(
        oracle_gap > 20.0,
        "scenario not sensitive: fp32 and bf16-accumulation oracles differ by only {oracle_gap}"
    );

    // The bf16 output ulp near VALUE=100 is 100*2^-8 ≈ 0.39; allow a small
    // multiple for the final store rounding.
    const GPU_TOLERANCE: f32 = 1.0;
    for (channel, &got) in gpu.iter().enumerate() {
        assert!(
            (got - fp32_oracle).abs() <= GPU_TOLERANCE,
            "channel {channel}: CUDA output {got} strayed from the fp32 oracle {fp32_oracle} \
             (bf16-accumulation would give {bf16_accumulator}); the kernel is not accumulating in fp32"
        );
        assert!(
            (got - bf16_accumulator).abs() > oracle_gap / 2.0,
            "channel {channel}: CUDA output {got} matches the bf16-accumulation oracle \
             {bf16_accumulator}; the kernel appears to accumulate probs·V in bf16"
        );
    }
}
