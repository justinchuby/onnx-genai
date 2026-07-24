//! ORT 1.26 parity for the Qwen3.5 hybrid `CausalConvWithState` and
//! `LinearAttention` operators through the public CPU execution-provider API.
//!
//! Re-generate goldens with:
//! `python3 crates/onnx-runtime-ep-cpu/tests/qwen35_parity/generate.py`

use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{Attribute, DataType, DeviceId, Node, NodeId, compute_contiguous_strides};

include!("qwen35_parity/cases.rs");

fn bits(values: &[u32]) -> Vec<f32> {
    values.iter().map(|&value| f32::from_bits(value)).collect()
}

fn run_f32(
    node: &Node,
    input_shapes: &[Vec<usize>],
    input_data: &[Vec<f32>],
    output_shapes: &[Vec<usize>],
) -> Vec<Vec<f32>> {
    let input_strides: Vec<_> = input_shapes
        .iter()
        .map(|shape| compute_contiguous_strides(shape))
        .collect();
    let inputs: Vec<_> = input_data
        .iter()
        .zip(input_shapes)
        .zip(&input_strides)
        .map(|((data, shape), strides)| {
            TensorView::new(
                DevicePtr(data.as_ptr().cast()),
                DataType::Float32,
                shape,
                strides,
                DeviceId::cpu(),
            )
        })
        .collect();

    let mut output_data: Vec<Vec<f32>> = output_shapes
        .iter()
        .map(|shape| vec![0.0; shape.iter().product()])
        .collect();
    let output_strides: Vec<_> = output_shapes
        .iter()
        .map(|shape| compute_contiguous_strides(shape))
        .collect();
    let mut outputs: Vec<_> = output_data
        .iter_mut()
        .zip(output_shapes)
        .zip(&output_strides)
        .map(|((data, shape), strides)| {
            TensorMut::new(
                DevicePtrMut(data.as_mut_ptr().cast()),
                DataType::Float32,
                shape,
                strides,
                DeviceId::cpu(),
            )
        })
        .collect();

    CpuExecutionProvider::new()
        .get_kernel(node, input_shapes, 1)
        .expect("Qwen3.5 kernel should be registered")
        .execute(&inputs, &mut outputs)
        .expect("Qwen3.5 kernel execution should succeed");
    output_data
}

fn run_u16(
    node: &Node,
    dtype: DataType,
    input_shapes: &[Vec<usize>],
    input_data: &[Vec<u16>],
    output_shapes: &[Vec<usize>],
) -> Vec<Vec<u16>> {
    let input_strides: Vec<_> = input_shapes
        .iter()
        .map(|shape| compute_contiguous_strides(shape))
        .collect();
    let inputs: Vec<_> = input_data
        .iter()
        .zip(input_shapes)
        .zip(&input_strides)
        .map(|((data, shape), strides)| {
            TensorView::new(
                DevicePtr(data.as_ptr().cast()),
                dtype,
                shape,
                strides,
                DeviceId::cpu(),
            )
        })
        .collect();

    let mut output_data: Vec<Vec<u16>> = output_shapes
        .iter()
        .map(|shape| vec![0; shape.iter().product()])
        .collect();
    let output_strides: Vec<_> = output_shapes
        .iter()
        .map(|shape| compute_contiguous_strides(shape))
        .collect();
    let mut outputs: Vec<_> = output_data
        .iter_mut()
        .zip(output_shapes)
        .zip(&output_strides)
        .map(|((data, shape), strides)| {
            TensorMut::new(
                DevicePtrMut(data.as_mut_ptr().cast()),
                dtype,
                shape,
                strides,
                DeviceId::cpu(),
            )
        })
        .collect();

    CpuExecutionProvider::new()
        .get_kernel(node, input_shapes, 1)
        .expect("Qwen3.5 kernel should be registered")
        .execute(&inputs, &mut outputs)
        .expect("Qwen3.5 kernel execution should succeed");
    output_data
}

fn causal_conv_node(silu: bool) -> Node {
    let mut node = Node::new(NodeId(0), "CausalConvWithState", vec![], vec![]);
    node.domain = "com.microsoft".to_string();
    node.attributes.insert("ndim".into(), Attribute::Int(1));
    node.attributes.insert(
        "activation".into(),
        Attribute::String(if silu { b"silu" } else { b"none" }.to_vec()),
    );
    node
}

fn linear_attention_node(heads: usize, scale: f32) -> Node {
    let mut node = Node::new(NodeId(0), "LinearAttention", vec![], vec![]);
    node.domain = "com.microsoft".to_string();
    node.attributes
        .insert("q_num_heads".into(), Attribute::Int(heads as i64));
    node.attributes
        .insert("kv_num_heads".into(), Attribute::Int(heads as i64));
    node.attributes.insert(
        "update_rule".into(),
        Attribute::String(b"gated_delta".to_vec()),
    );
    node.attributes
        .insert("scale".into(), Attribute::Float(scale));
    node
}

fn assert_close(actual: &[f32], expected_bits: &[u32], tag: &str) {
    let expected = bits(expected_bits);
    assert_eq!(actual.len(), expected.len(), "{tag}: output length");
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        let abs = (actual - expected).abs();
        let rel = abs / expected.abs().max(1e-6);
        assert!(
            abs <= 1e-4 || rel <= 1e-4,
            "{tag}[{index}]: got {actual}, want {expected} (abs {abs}, rel {rel})"
        );
    }
}

#[test]
fn causal_conv_with_state_matches_onnxruntime_1_26() {
    use conv as g;

    let cases = [
        (
            "CONVA",
            g::CONVA_DIMS,
            g::CONVA_SILU,
            &g::CONVA_X[..],
            &g::CONVA_W[..],
            &g::CONVA_B[..],
            &g::CONVA_STATE[..],
            &g::CONVA_Y[..],
            &g::CONVA_PRESENT[..],
        ),
        (
            "CONVB",
            g::CONVB_DIMS,
            g::CONVB_SILU,
            &g::CONVB_X[..],
            &g::CONVB_W[..],
            &g::CONVB_B[..],
            &g::CONVB_STATE[..],
            &g::CONVB_Y[..],
            &g::CONVB_PRESENT[..],
        ),
        (
            "CONVC",
            g::CONVC_DIMS,
            g::CONVC_SILU,
            &g::CONVC_X[..],
            &g::CONVC_W[..],
            &g::CONVC_B[..],
            &g::CONVC_STATE[..],
            &g::CONVC_Y[..],
            &g::CONVC_PRESENT[..],
        ),
    ];

    for (name, [batch, channels, kernel, sequence], silu, x, weight, bias, state, y, present) in
        cases
    {
        let node = causal_conv_node(silu);

        let pad = kernel - 1;
        let input_shapes = vec![
            vec![batch, channels, sequence],
            vec![channels, 1, kernel],
            vec![channels],
            vec![batch, channels, pad],
        ];
        let input_data = vec![bits(x), bits(weight), bits(bias), bits(state)];
        let output_shapes = vec![vec![batch, channels, sequence], vec![batch, channels, pad]];
        let outputs = run_f32(&node, &input_shapes, &input_data, &output_shapes);
        assert_close(&outputs[0], y, &format!("{name} output"));
        assert_close(&outputs[1], present, &format!("{name} present_state"));
    }
}

#[test]
fn linear_attention_matches_onnxruntime_1_26() {
    use la as g;

    let cases = [
        (
            "LAA",
            g::LAA_DIMS,
            g::LAA_SCALE,
            &g::LAA_Q[..],
            &g::LAA_K[..],
            &g::LAA_V[..],
            &g::LAA_STATE[..],
            &g::LAA_G[..],
            &g::LAA_BETA[..],
            &g::LAA_O[..],
            &g::LAA_PRESENT[..],
        ),
        (
            "LAB",
            g::LAB_DIMS,
            g::LAB_SCALE,
            &g::LAB_Q[..],
            &g::LAB_K[..],
            &g::LAB_V[..],
            &g::LAB_STATE[..],
            &g::LAB_G[..],
            &g::LAB_BETA[..],
            &g::LAB_O[..],
            &g::LAB_PRESENT[..],
        ),
        (
            "LAC",
            g::LAC_DIMS,
            g::LAC_SCALE,
            &g::LAC_Q[..],
            &g::LAC_K[..],
            &g::LAC_V[..],
            &g::LAC_STATE[..],
            &g::LAC_G[..],
            &g::LAC_BETA[..],
            &g::LAC_O[..],
            &g::LAC_PRESENT[..],
        ),
    ];

    for (
        name,
        [batch, heads, d_k, d_v, sequence],
        scale,
        q,
        k,
        v,
        state,
        decay,
        beta,
        output,
        present,
    ) in cases
    {
        let node = linear_attention_node(heads, scale);

        let input_shapes = vec![
            vec![batch, sequence, heads * d_k],
            vec![batch, sequence, heads * d_k],
            vec![batch, sequence, heads * d_v],
            vec![batch, heads, d_k, d_v],
            vec![batch, sequence, heads],
            vec![batch, sequence, heads],
        ];
        let input_data = vec![
            bits(q),
            bits(k),
            bits(v),
            bits(state),
            bits(decay),
            bits(beta),
        ];
        let output_shapes = vec![
            vec![batch, sequence, heads * d_v],
            vec![batch, heads, d_k, d_v],
        ];
        let outputs = run_f32(&node, &input_shapes, &input_data, &output_shapes);
        assert_close(&outputs[0], output, &format!("{name} output"));
        assert_close(&outputs[1], present, &format!("{name} present_state"));
    }
}

#[test]
fn causal_conv_half_precision_widens_and_narrows_through_ep() {
    use conv as g;

    let [batch, channels, kernel, sequence] = g::CONVC_DIMS;
    let pad = kernel - 1;
    let input_shapes = vec![
        vec![batch, channels, sequence],
        vec![channels, 1, kernel],
        vec![channels],
        vec![batch, channels, pad],
    ];
    let output_shapes = vec![vec![batch, channels, sequence], vec![batch, channels, pad]];
    let inputs = [&g::CONVC_X[..], &g::CONVC_W, &g::CONVC_B, &g::CONVC_STATE];
    let node = causal_conv_node(false);
    let expected = bits(&g::CONVC_Y);

    let bf16_inputs: Vec<Vec<u16>> = inputs
        .iter()
        .map(|values| {
            values
                .iter()
                .map(|&value| half::bf16::from_f32(f32::from_bits(value)).to_bits())
                .collect()
        })
        .collect();
    let bf16_outputs = run_u16(
        &node,
        DataType::BFloat16,
        &input_shapes,
        &bf16_inputs,
        &output_shapes,
    );
    for (index, (&actual, &expected)) in bf16_outputs[0].iter().zip(&expected).enumerate() {
        let actual = half::bf16::from_bits(actual).to_f32();
        assert!(
            (actual - expected).abs() <= 0.05 * expected.abs().max(1.0) + 0.05,
            "bf16 output[{index}]: got {actual}, want {expected}"
        );
    }

    let f16_inputs: Vec<Vec<u16>> = inputs
        .iter()
        .map(|values| {
            values
                .iter()
                .map(|&value| half::f16::from_f32(f32::from_bits(value)).to_bits())
                .collect()
        })
        .collect();
    let f16_outputs = run_u16(
        &node,
        DataType::Float16,
        &input_shapes,
        &f16_inputs,
        &output_shapes,
    );
    for (index, (&actual, &expected)) in f16_outputs[0].iter().zip(&expected).enumerate() {
        let actual = half::f16::from_bits(actual).to_f32();
        assert!(
            (actual - expected).abs() <= 2e-2,
            "f16 output[{index}]: got {actual}, want {expected}"
        );
    }
}

#[test]
fn linear_attention_bf16_matches_widened_f32_reference_through_ep() {
    use la as g;

    let [batch, heads, d_k, d_v, sequence] = g::LAA_DIMS;
    let input_shapes = vec![
        vec![batch, sequence, heads * d_k],
        vec![batch, sequence, heads * d_k],
        vec![batch, sequence, heads * d_v],
        vec![batch, heads, d_k, d_v],
        vec![batch, sequence, heads],
        vec![batch, sequence, heads],
    ];
    let output_shapes = vec![
        vec![batch, sequence, heads * d_v],
        vec![batch, heads, d_k, d_v],
    ];
    let inputs = [
        &g::LAA_Q[..],
        &g::LAA_K,
        &g::LAA_V,
        &g::LAA_STATE,
        &g::LAA_G,
        &g::LAA_BETA,
    ];
    let f32_inputs: Vec<_> = inputs.iter().map(|values| bits(values)).collect();
    let node = linear_attention_node(heads, g::LAA_SCALE);
    let f32_outputs = run_f32(&node, &input_shapes, &f32_inputs, &output_shapes);
    let bf16_inputs: Vec<Vec<u16>> = inputs
        .iter()
        .map(|values| {
            values
                .iter()
                .map(|&value| half::bf16::from_f32(f32::from_bits(value)).to_bits())
                .collect()
        })
        .collect();
    let bf16_outputs = run_u16(
        &node,
        DataType::BFloat16,
        &input_shapes,
        &bf16_inputs,
        &output_shapes,
    );

    for (output_index, (reference, actual)) in f32_outputs.iter().zip(&bf16_outputs).enumerate() {
        for (index, (&reference, &actual)) in reference.iter().zip(actual).enumerate() {
            let actual = half::bf16::from_bits(actual).to_f32();
            assert!(
                (reference - actual).abs() <= 0.06 * reference.abs().max(1.0) + 0.02,
                "bf16 output {output_index}[{index}]: got {actual}, want {reference}"
            );
        }
    }
}
