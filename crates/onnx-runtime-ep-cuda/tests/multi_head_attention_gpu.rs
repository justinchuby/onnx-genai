#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::uninlined_format_args
)]
//! GPU parity for `com.microsoft::MultiHeadAttention`: the CUDA EP separate-QKV
//! multi-head attention adapter, checked tol-exact against the CPU EP oracle
//! across fp32/fp16/bf16, decode (`S=1`) and prefill (`S>1`), with/without
//! `bias`, with/without an in-op past-KV cache, with/without `key_padding_mask`
//! (including a fully-padded batch row, whose ORT-defined near-uniform
//! distribution is a real behaviour to match), with/without `attention_bias`,
//! and causal (`unidirectional`) vs non-causal.

mod common;

use std::sync::{Mutex, MutexGuard};

use common::{
    absent_input, assert_close, decode_floats, float_input, input, require_cuda, run_cpu, run_cuda,
};
use onnx_runtime_ir::{Attribute, DataType};

static MHA_GPU_LOCK: Mutex<()> = Mutex::new(());

fn lock_mha_gpu() -> MutexGuard<'static, ()> {
    MHA_GPU_LOCK.lock().unwrap_or_else(|poisoned| {
        eprintln!(
            "WARNING: MHA_GPU_LOCK was poisoned by a prior test panic — recovering. \
             Investigate the original failure above."
        );
        poisoned.into_inner()
    })
}

const OP: &str = "MultiHeadAttention";
const DOMAIN: &str = "com.microsoft";
const OPSET: u64 = 1;

fn tolerance(dtype: DataType) -> f32 {
    match dtype {
        DataType::Float32 => 2e-5,
        DataType::Float16 => 4e-3,
        DataType::BFloat16 => 4e-2,
        _ => 0.0,
    }
}

/// Deterministic mixed-magnitude values in roughly `[-1.5, 1.5]` — enough spread
/// that a transpose/layout bug cannot hide behind a smooth ramp, but small
/// enough that the softmax stays well away from saturation (a saturated softmax
/// would pass even against a broken kernel).
fn seeded(count: usize, seed: u64) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed;
            x ^= x >> 33;
            x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            x ^= x >> 29;
            let unit = ((x >> 11) as f64) / ((1u64 << 53) as f64);
            ((unit - 0.5) * 3.0) as f32
        })
        .collect()
}

#[derive(Default)]
struct Case {
    num_heads: usize,
    batch: usize,
    q_seq: usize,
    kv_seq: usize,
    head_size: usize,
    /// Rank-4 BNSH key/value (cross-attention) instead of rank-3 BSH.
    cross_bnsh: bool,
    bias: bool,
    /// `(shape, values, dtype)` for the optional `key_padding_mask` (slot 4).
    key_padding_mask: Option<(Vec<usize>, Vec<i64>, DataType)>,
    /// `(shape, values)` for the optional additive `attention_bias` (slot 5).
    attention_bias: Option<(Vec<usize>, Vec<f32>)>,
    /// Past sequence length; a non-zero value drives the KV-cache path.
    past_seq: usize,
    unidirectional: bool,
    want_present: bool,
}

fn build_inputs(case: &Case, dtype: DataType) -> Vec<common::Tensor> {
    let Case {
        num_heads,
        batch,
        q_seq,
        kv_seq,
        head_size,
        cross_bnsh,
        bias,
        past_seq,
        ..
    } = *case;
    let hidden = num_heads * head_size;

    let q = float_input(
        dtype,
        &[batch, q_seq, hidden],
        &seeded(batch * q_seq * hidden, 1),
    );
    let (key, value) = if cross_bnsh {
        (
            float_input(
                dtype,
                &[batch, num_heads, kv_seq, head_size],
                &seeded(batch * num_heads * kv_seq * head_size, 2),
            ),
            float_input(
                dtype,
                &[batch, num_heads, kv_seq, head_size],
                &seeded(batch * num_heads * kv_seq * head_size, 3),
            ),
        )
    } else {
        (
            float_input(
                dtype,
                &[batch, kv_seq, hidden],
                &seeded(batch * kv_seq * hidden, 2),
            ),
            float_input(
                dtype,
                &[batch, kv_seq, hidden],
                &seeded(batch * kv_seq * hidden, 3),
            ),
        )
    };

    let mut inputs = vec![q, key, value];

    // Determine the highest present optional slot so intervening omitted
    // optionals are materialised as absent placeholders (positional arity).
    let has_bias = bias;
    let has_mask = case.key_padding_mask.is_some();
    let has_abias = case.attention_bias.is_some();
    let has_past = past_seq > 0;
    let max_slot = if has_past {
        7
    } else if has_abias {
        5
    } else if has_mask {
        4
    } else if has_bias {
        3
    } else {
        2
    };

    for slot in 3..=max_slot {
        match slot {
            3 => inputs.push(if has_bias {
                let v_hidden = num_heads * head_size;
                float_input(
                    dtype,
                    &[2 * hidden + v_hidden],
                    &seeded(2 * hidden + v_hidden, 4),
                )
            } else {
                absent_input(dtype)
            }),
            4 => inputs.push(match &case.key_padding_mask {
                Some((shape, values, mask_dtype)) => match mask_dtype {
                    DataType::Int32 => input(
                        DataType::Int32,
                        shape,
                        &values.iter().map(|&v| v as i32).collect::<Vec<_>>(),
                    ),
                    _ => input(DataType::Int64, shape, values),
                },
                None => absent_input(DataType::Int64),
            }),
            5 => inputs.push(match &case.attention_bias {
                Some((shape, values)) => float_input(dtype, shape, values),
                None => absent_input(dtype),
            }),
            6 => inputs.push(if has_past {
                float_input(
                    dtype,
                    &[batch, num_heads, past_seq, head_size],
                    &seeded(batch * num_heads * past_seq * head_size, 6),
                )
            } else {
                absent_input(dtype)
            }),
            7 => inputs.push(if has_past {
                float_input(
                    dtype,
                    &[batch, num_heads, past_seq, head_size],
                    &seeded(batch * num_heads * past_seq * head_size, 7),
                )
            } else {
                absent_input(dtype)
            }),
            _ => unreachable!(),
        }
    }
    inputs
}

fn check(case: Case, dtype: DataType, ep: &onnx_runtime_ep_cuda::CudaExecutionProvider) {
    let inputs = build_inputs(&case, dtype);
    let hidden = case.num_heads * case.head_size;
    let total = case.past_seq + case.kv_seq;
    let mut outputs = vec![(dtype, vec![case.batch, case.q_seq, hidden])];
    if case.want_present {
        outputs.push((
            dtype,
            vec![case.batch, case.num_heads, total, case.head_size],
        ));
        outputs.push((
            dtype,
            vec![case.batch, case.num_heads, total, case.head_size],
        ));
    }
    let mut attrs = vec![("num_heads", Attribute::Int(case.num_heads as i64))];
    if case.unidirectional {
        attrs.push(("unidirectional", Attribute::Int(1)));
    }

    let cuda = run_cuda(ep, OP, DOMAIN, OPSET, &inputs, &outputs, &attrs);
    let cpu = run_cpu(OP, DOMAIN, OPSET, &inputs, &outputs, &attrs);
    let tol = tolerance(dtype);
    for (idx, (c, r)) in cuda.iter().zip(&cpu).enumerate() {
        assert_close(
            &format!("{OP}[out{idx}]"),
            dtype,
            &decode_floats(c, dtype),
            &decode_floats(r, dtype),
            tol,
        );
    }
}

fn all_dtypes() -> [DataType; 3] {
    [DataType::Float32, DataType::Float16, DataType::BFloat16]
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn self_attention_decode_and_prefill() {
    let _suite_lock = lock_mha_gpu();
    let ep = require_cuda();
    for dtype in all_dtypes() {
        // Prefill: S=L=4.
        check(
            Case {
                num_heads: 3,
                batch: 2,
                q_seq: 4,
                kv_seq: 4,
                head_size: 5,
                ..Default::default()
            },
            dtype,
            &ep,
        );
        // Decode: S=1, L=6.
        check(
            Case {
                num_heads: 2,
                batch: 2,
                q_seq: 1,
                kv_seq: 6,
                head_size: 8,
                ..Default::default()
            },
            dtype,
            &ep,
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn causal_prefill_matches_cpu() {
    let _suite_lock = lock_mha_gpu();
    let ep = require_cuda();
    for dtype in all_dtypes() {
        check(
            Case {
                num_heads: 2,
                batch: 1,
                q_seq: 5,
                kv_seq: 5,
                head_size: 4,
                unidirectional: true,
                ..Default::default()
            },
            dtype,
            &ep,
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn with_bias_matches_cpu() {
    let _suite_lock = lock_mha_gpu();
    let ep = require_cuda();
    for dtype in all_dtypes() {
        check(
            Case {
                num_heads: 2,
                batch: 2,
                q_seq: 3,
                kv_seq: 3,
                head_size: 4,
                bias: true,
                ..Default::default()
            },
            dtype,
            &ep,
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn cross_attention_bnsh_kv_matches_cpu() {
    let _suite_lock = lock_mha_gpu();
    let ep = require_cuda();
    for dtype in all_dtypes() {
        check(
            Case {
                num_heads: 2,
                batch: 2,
                q_seq: 3,
                kv_seq: 4,
                head_size: 6,
                cross_bnsh: true,
                ..Default::default()
            },
            dtype,
            &ep,
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn past_kv_cache_matches_cpu() {
    let _suite_lock = lock_mha_gpu();
    let ep = require_cuda();
    for dtype in all_dtypes() {
        // Decode step attending a 4-long cache plus 1 new key, present emitted.
        check(
            Case {
                num_heads: 2,
                batch: 2,
                q_seq: 1,
                kv_seq: 1,
                head_size: 8,
                past_seq: 4,
                want_present: true,
                ..Default::default()
            },
            dtype,
            &ep,
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn key_padding_mask_forms_match_cpu() {
    let _suite_lock = lock_mha_gpu();
    let ep = require_cuda();
    for dtype in all_dtypes() {
        // KeyLen (B,): both rows keep a real (well-conditioned) subset of keys.
        // A *fully*-padded row is exercised separately at f32 — see
        // `fully_padded_row_matches_cpu_f32` for why it is precision-bound at
        // f16/bf16.
        check(
            Case {
                num_heads: 2,
                batch: 2,
                q_seq: 2,
                kv_seq: 4,
                head_size: 4,
                key_padding_mask: Some((vec![2], vec![3, 2], DataType::Int32)),
                ..Default::default()
            },
            dtype,
            &ep,
        );
        // Raw 2-D (B, T) mask, int64; every row keeps at least one key.
        check(
            Case {
                num_heads: 2,
                batch: 2,
                q_seq: 2,
                kv_seq: 3,
                head_size: 4,
                key_padding_mask: Some((vec![2, 3], vec![1, 1, 0, 1, 0, 0], DataType::Int64)),
                ..Default::default()
            },
            dtype,
            &ep,
        );
    }
}

/// A fully key-padded batch row (`len = 0`). ORT adds `mask_filter_value`
/// (`-10000`) to every logit; the constant cancels under softmax, so the row
/// resolves to `softmax(raw scores)`. At f32 the CUDA EP reproduces the CPU
/// oracle bit-for-tolerance. This is deliberately **f32-only**: the shared
/// cuBLASLt/softmax softmax writes each masked logit (`raw − 10000`) back to the
/// low-precision score buffer, and at f16/bf16 every value rounds to exactly
/// `-10000` (the mantissa step near 10000 is 8), collapsing the row to a uniform
/// distribution — a measured precision artifact of the shared core, not a logic
/// error (documented in the PR).
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn fully_padded_row_matches_cpu_f32() {
    let _suite_lock = lock_mha_gpu();
    let ep = require_cuda();
    check(
        Case {
            num_heads: 2,
            batch: 2,
            q_seq: 2,
            kv_seq: 4,
            head_size: 4,
            // Batch 0 keeps 3 keys; batch 1 keeps none (fully padded).
            key_padding_mask: Some((vec![2], vec![3, 0], DataType::Int64)),
            ..Default::default()
        },
        DataType::Float32,
        &ep,
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn attention_bias_broadcasts_match_cpu() {
    let _suite_lock = lock_mha_gpu();
    let ep = require_cuda();
    for dtype in all_dtypes() {
        let (batch, num_heads, q_seq, kv_seq) = (2usize, 2usize, 3usize, 3usize);
        // Full (B, N, S, T) additive bias.
        check(
            Case {
                num_heads,
                batch,
                q_seq,
                kv_seq,
                head_size: 4,
                attention_bias: Some((
                    vec![batch, num_heads, q_seq, kv_seq],
                    seeded(batch * num_heads * q_seq * kv_seq, 9),
                )),
                ..Default::default()
            },
            dtype,
            &ep,
        );
        // Broadcast (1, 1, S, T) bias.
        check(
            Case {
                num_heads,
                batch,
                q_seq,
                kv_seq,
                head_size: 4,
                attention_bias: Some((vec![1, 1, q_seq, kv_seq], seeded(q_seq * kv_seq, 10))),
                ..Default::default()
            },
            dtype,
            &ep,
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn bias_mask_and_present_together_match_cpu() {
    let _suite_lock = lock_mha_gpu();
    let ep = require_cuda();
    for dtype in all_dtypes() {
        check(
            Case {
                num_heads: 2,
                batch: 2,
                q_seq: 3,
                kv_seq: 3,
                head_size: 4,
                bias: true,
                key_padding_mask: Some((vec![2], vec![2, 3], DataType::Int64)),
                want_present: true,
                ..Default::default()
            },
            dtype,
            &ep,
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn mha_prepared_workspace_reuses_backing_and_preserves_present_output_holes() {
    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView};
    use onnx_runtime_ir::{Node, NodeId, ValueId, compute_contiguous_strides};
    let _guard = lock_mha_gpu();
    let ep = require_cuda();
    let shape = [1usize, 2, 4];
    let input_shapes = vec![shape.to_vec(); 3];
    let strides = compute_contiguous_strides(&shape);
    let cache_shape = [1usize, 1, 2, 4];
    let cache_strides = compute_contiguous_strides(&cache_shape);
    let mut qk = ep.allocate(32, 256).unwrap();
    let mut value = ep.allocate(32, 256).unwrap();
    let mut output = ep.allocate(32, 256).unwrap();
    let mut present_value = ep.allocate(32, 256).unwrap();
    ep.copy_from_host(&common::raw(&[0f32; 8]), &mut qk)
        .unwrap();
    ep.copy_from_host(
        &common::raw(&[1f32, 2., 3., 4., 5., 6., 7., 8.]),
        &mut value,
    )
    .unwrap();
    let inputs = [
        TensorView::new(
            DevicePtr(qk.as_ptr()),
            DataType::Float32,
            &shape,
            &strides,
            ep.device_id(),
        ),
        TensorView::new(
            DevicePtr(qk.as_ptr()),
            DataType::Float32,
            &shape,
            &strides,
            ep.device_id(),
        ),
        TensorView::new(
            DevicePtr(value.as_ptr()),
            DataType::Float32,
            &shape,
            &strides,
            ep.device_id(),
        ),
    ];
    let node = |outputs| {
        let mut node = Node::new_with_output_slots(
            NodeId(0),
            OP,
            vec![Some(ValueId(0)), Some(ValueId(1)), Some(ValueId(2))],
            outputs,
        );
        node.domain = DOMAIN.into();
        node.attributes
            .insert("num_heads".into(), Attribute::Int(1));
        node
    };
    let only_y = node(vec![Some(ValueId(10))]);
    let kernel = ep.get_kernel(&only_y, &input_shapes, OPSET).unwrap();
    let error = kernel
        .execute(
            &inputs,
            &mut [TensorMut::new(
                DevicePtrMut(output.as_mut_ptr()),
                DataType::Float32,
                &shape,
                &strides,
                ep.device_id(),
            )],
        )
        .unwrap_err();
    assert!(
        error.to_string().contains("execute_with_workspace"),
        "{error}"
    );
    let mut workspace = None;
    ep.copy_from_host(&common::raw(&[-31f32; 8]), &mut output)
        .unwrap();
    common::execute_kernel(
        &ep,
        kernel.as_ref(),
        &inputs,
        &mut [TensorMut::new(
            DevicePtrMut(output.as_mut_ptr()),
            DataType::Float32,
            &shape,
            &strides,
            ep.device_id(),
        )],
        &mut workspace,
    )
    .unwrap();
    let identity = workspace.as_ref().unwrap().as_ptr();
    let mut bytes = [0u8; 32];
    ep.copy_to_host(&output, &mut bytes).unwrap();
    assert_eq!(
        decode_floats(&bytes, DataType::Float32),
        [3f32, 4., 5., 6., 3., 4., 5., 6.]
    );

    let with_hole = node(vec![Some(ValueId(10)), None, Some(ValueId(12))]);
    let kernel = ep.get_kernel(&with_hole, &input_shapes, OPSET).unwrap();
    let error = kernel
        .execute(
            &inputs,
            &mut [TensorMut::new(
                DevicePtrMut(output.as_mut_ptr()),
                DataType::Float32,
                &shape,
                &strides,
                ep.device_id(),
            )],
        )
        .unwrap_err();
    assert!(error.to_string().contains("compiled slots"), "{error}");
    ep.copy_from_host(&common::raw(&[-31f32; 8]), &mut output)
        .unwrap();
    ep.copy_from_host(&common::raw(&[-19f32; 8]), &mut present_value)
        .unwrap();
    common::execute_kernel(
        &ep,
        kernel.as_ref(),
        &inputs,
        &mut [
            TensorMut::new(
                DevicePtrMut(output.as_mut_ptr()),
                DataType::Float32,
                &shape,
                &strides,
                ep.device_id(),
            ),
            TensorMut::new(
                DevicePtrMut(std::ptr::null_mut()),
                DataType::Undefined,
                &[],
                &[],
                ep.device_id(),
            ),
            TensorMut::new(
                DevicePtrMut(present_value.as_mut_ptr()),
                DataType::Float32,
                &cache_shape,
                &cache_strides,
                ep.device_id(),
            ),
        ],
        &mut workspace,
    )
    .unwrap();
    assert_eq!(
        workspace.as_ref().unwrap().as_ptr(),
        identity,
        "the fitting workspace must retain its physical backing"
    );
    ep.copy_to_host(&output, &mut bytes).unwrap();
    assert_eq!(
        decode_floats(&bytes, DataType::Float32),
        [3f32, 4., 5., 6., 3., 4., 5., 6.]
    );
    ep.copy_to_host(&present_value, &mut bytes).unwrap();
    assert_eq!(
        decode_floats(&bytes, DataType::Float32),
        [1f32, 2., 3., 4., 5., 6., 7., 8.]
    );
    ep.deallocate_workspace(workspace.unwrap()).unwrap();
    for buffer in [qk, value, output, present_value] {
        ep.deallocate(buffer).unwrap();
    }
}
