#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::unusual_byte_groupings,
    clippy::doc_lazy_continuation,
    clippy::uninlined_format_args,
    clippy::cloned_ref_to_slice_refs,
    clippy::type_complexity,
    clippy::drop_non_drop,
    clippy::manual_repeat_n,
    clippy::manual_is_multiple_of,
    clippy::err_expect,
    clippy::clone_on_copy
)]
//! GPU parity regressions for `com.microsoft::LinearAttention` (Gated DeltaNet /
//! gated delta-rule linear attention, the recurrent attention of the Qwen3.5 /
//! Qwen3-Next hybrid family).
//!
//! Every case builds small synthetic query/key/value/past_state/decay/beta
//! tensors, runs the **CPU EP** kernel as the parity oracle (never a
//! self-comparison), and asserts the CUDA `LinearAttention` kernel reproduces
//! both the `output` and the `present_state` within tolerance. Coverage spans
//! the four `update_rule` variants, standard and inverse GQA, key-head sharing
//! (`n_k < H_kv`), per-head vs per-key-dim decay, per-head vs shared beta,
//! multi-timestep recurrence (state carry), a non-trivial past_state, and the
//! Float32 / Float16 / BFloat16 dtypes. CPU-only runs report these as ignored
//! unless `gpu-tests` is enabled.

mod common;

use common::{
    Tensor, assert_close, build_graph, decode_floats, encode_floats, float_input, require_cuda,
    run_cpu, run_cuda,
};
use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{Attribute, DataType, compute_contiguous_strides};
use onnx_runtime_loader::Model;

const DOMAIN: &str = "com.microsoft";
const OPSET: u64 = 1;

/// Deterministic pseudo-random f32 values in `[lo, hi)` from a splitmix64 seed.
fn values(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let unit = (z >> 40) as f32 / (1u64 << 24) as f32; // [0, 1)
            lo + unit * (hi - lo)
        })
        .collect()
}

struct Config {
    label: &'static str,
    batch: usize,
    seq: usize,
    d_k: usize,
    d_v: usize,
    q_num_heads: usize,
    kv_num_heads: usize,
    n_k_heads: usize,
    update_rule: &'static str,
    scale: f32,
    decay_per_key_dim: bool,
    beta_shared: bool,
    with_past: bool,
}

impl Config {
    fn output_hidden(&self) -> usize {
        self.q_num_heads.max(self.kv_num_heads) * self.d_v
    }
}

/// Assemble the six-input tensor list (query, key, value, past_state, decay,
/// beta) for one config in the given dtype. Trailing optionals are always
/// present here to exercise the full path.
fn build_inputs(cfg: &Config, dtype: DataType) -> Vec<Tensor> {
    let bt = cfg.batch * cfg.seq;
    let q = float_input(
        dtype,
        &[cfg.batch, cfg.seq, cfg.q_num_heads * cfg.d_k],
        &values(1, bt * cfg.q_num_heads * cfg.d_k, -1.0, 1.0),
    );
    let k = float_input(
        dtype,
        &[cfg.batch, cfg.seq, cfg.n_k_heads * cfg.d_k],
        &values(2, bt * cfg.n_k_heads * cfg.d_k, -1.0, 1.0),
    );
    let v = float_input(
        dtype,
        &[cfg.batch, cfg.seq, cfg.kv_num_heads * cfg.d_v],
        &values(3, bt * cfg.kv_num_heads * cfg.d_v, -1.0, 1.0),
    );
    let past = if cfg.with_past {
        float_input(
            dtype,
            &[cfg.batch, cfg.kv_num_heads, cfg.d_k, cfg.d_v],
            &values(
                4,
                cfg.batch * cfg.kv_num_heads * cfg.d_k * cfg.d_v,
                -0.5,
                0.5,
            ),
        )
    } else {
        float_input(
            dtype,
            &[cfg.batch, cfg.kv_num_heads, cfg.d_k, cfg.d_v],
            &vec![0.0; cfg.batch * cfg.kv_num_heads * cfg.d_k * cfg.d_v],
        )
    };
    // decay is a log-gate; keep it mildly negative so exp(g) stays in (0, 1].
    let decay_last = if cfg.decay_per_key_dim {
        cfg.kv_num_heads * cfg.d_k
    } else {
        cfg.kv_num_heads
    };
    let decay = float_input(
        dtype,
        &[cfg.batch, cfg.seq, decay_last],
        &values(5, bt * decay_last, -0.6, -0.02),
    );
    let beta_last = if cfg.beta_shared { 1 } else { cfg.kv_num_heads };
    let beta = float_input(
        dtype,
        &[cfg.batch, cfg.seq, beta_last],
        &values(6, bt * beta_last, 0.1, 0.9),
    );
    vec![q, k, v, past, decay, beta]
}

fn attrs(cfg: &Config) -> Vec<(&'static str, Attribute)> {
    vec![
        ("q_num_heads", Attribute::Int(cfg.q_num_heads as i64)),
        ("kv_num_heads", Attribute::Int(cfg.kv_num_heads as i64)),
        (
            "update_rule",
            Attribute::String(cfg.update_rule.as_bytes().to_vec()),
        ),
        ("scale", Attribute::Float(cfg.scale)),
    ]
}

fn tolerance(dtype: DataType) -> f32 {
    match dtype {
        DataType::Float32 => 2e-4,
        DataType::Float16 => 5e-2,
        DataType::BFloat16 => 2e-1,
        _ => unreachable!(),
    }
}

/// Run one config on CUDA and the CPU oracle, comparing every output.
fn check(cfg: &Config, dtype: DataType) {
    let ep = require_cuda();
    run_check(&ep, cfg, dtype, DOMAIN);
}

fn run_check(ep: &CudaExecutionProvider, cfg: &Config, dtype: DataType, domain: &str) {
    let inputs = build_inputs(cfg, dtype);
    let outputs = vec![
        (dtype, vec![cfg.batch, cfg.seq, cfg.output_hidden()]),
        (dtype, vec![cfg.batch, cfg.kv_num_heads, cfg.d_k, cfg.d_v]),
    ];
    let a = attrs(cfg);
    let cuda = run_cuda(ep, "LinearAttention", domain, OPSET, &inputs, &outputs, &a);
    let cpu = run_cpu("LinearAttention", domain, OPSET, &inputs, &outputs, &a);
    assert_eq!(cuda.len(), 2, "{}: expected 2 outputs", cfg.label);
    assert_eq!(cpu.len(), 2, "{}: CPU expected 2 outputs", cfg.label);
    let tol = tolerance(dtype);
    for (idx, name) in ["output", "present_state"].iter().enumerate() {
        let got = decode_floats(&cuda[idx], dtype);
        let want = decode_floats(&cpu[idx], dtype);
        assert_close(
            &format!("{} [{name}] {dtype:?} ({domain})", cfg.label),
            dtype,
            &got,
            &want,
            tol,
        );
    }
}

/// The full parity matrix. Shapes are small but non-trivial (multi-timestep so
/// the recurrent state genuinely carries; GQA both directions; key sharing).
fn configs() -> Vec<Config> {
    vec![
        // Standard GQA, gated_delta, per-head decay + per-head beta, with a
        // non-zero past_state — the real Qwen3.5 0.8b/2b decode config in
        // miniature (q == kv, multi-step recurrence).
        Config {
            label: "gated_delta/gqa1/past",
            batch: 2,
            seq: 4,
            d_k: 6,
            d_v: 5,
            q_num_heads: 3,
            kv_num_heads: 3,
            n_k_heads: 3,
            update_rule: "gated_delta",
            scale: 0.7,
            decay_per_key_dim: false,
            beta_shared: false,
            with_past: true,
        },
        // Standard GQA with grouping (q > kv) + key-head sharing (n_k < kv).
        Config {
            label: "gated_delta/gqa4/keyshare",
            batch: 1,
            seq: 5,
            d_k: 4,
            d_v: 4,
            q_num_heads: 8,
            kv_num_heads: 4,
            n_k_heads: 2,
            update_rule: "gated_delta",
            scale: 0.0, // resolves to 1/sqrt(d_k)
            decay_per_key_dim: false,
            beta_shared: false,
            with_past: true,
        },
        // Inverse GQA (q < kv) — the Qwen3.5 9b config shape (q=16, kv=32).
        Config {
            label: "gated_delta/inverse_gqa",
            batch: 1,
            seq: 3,
            d_k: 4,
            d_v: 5,
            q_num_heads: 2,
            kv_num_heads: 4,
            n_k_heads: 2,
            update_rule: "gated_delta",
            scale: 1.0,
            decay_per_key_dim: false,
            beta_shared: false,
            with_past: true,
        },
        // Per-key-dim decay layout ([B, T, H_kv·d_k]).
        Config {
            label: "gated_delta/decay_per_key",
            batch: 1,
            seq: 4,
            d_k: 3,
            d_v: 3,
            q_num_heads: 2,
            kv_num_heads: 2,
            n_k_heads: 2,
            update_rule: "gated_delta",
            scale: 0.5,
            decay_per_key_dim: true,
            beta_shared: false,
            with_past: false,
        },
        // Shared beta ([B, T, 1]).
        Config {
            label: "delta/beta_shared",
            batch: 2,
            seq: 3,
            d_k: 4,
            d_v: 4,
            q_num_heads: 2,
            kv_num_heads: 2,
            n_k_heads: 2,
            update_rule: "delta",
            scale: 0.9,
            decay_per_key_dim: false,
            beta_shared: true,
            with_past: false,
        },
        // Gated (decay, no delta).
        Config {
            label: "gated/no_delta",
            batch: 1,
            seq: 4,
            d_k: 5,
            d_v: 4,
            q_num_heads: 2,
            kv_num_heads: 2,
            n_k_heads: 2,
            update_rule: "gated",
            scale: 0.6,
            decay_per_key_dim: false,
            beta_shared: false,
            with_past: true,
        },
        // Plain linear (no gates at all).
        Config {
            label: "linear/plain",
            batch: 1,
            seq: 5,
            d_k: 4,
            d_v: 3,
            q_num_heads: 3,
            kv_num_heads: 3,
            n_k_heads: 3,
            update_rule: "linear",
            scale: 0.8,
            decay_per_key_dim: false,
            beta_shared: false,
            with_past: false,
        },
    ]
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn linear_attention_f32_parity() {
    for cfg in configs() {
        check(&cfg, DataType::Float32);
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn linear_attention_f16_parity() {
    for cfg in configs() {
        check(&cfg, DataType::Float16);
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn linear_attention_bf16_parity() {
    for cfg in configs() {
        check(&cfg, DataType::BFloat16);
    }
}

/// The standard ONNX-domain spelling (`""`) must dispatch to the SAME fused
/// kernel and match the CPU oracle exactly like the `com.microsoft` spelling —
/// proving the dual-domain registration (onnx/onnx#7689) is wired end to end.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn linear_attention_standard_domain_parity() {
    let ep = require_cuda();
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        for cfg in configs() {
            run_check(&ep, &cfg, dtype, "");
        }
    }
}

/// The two domain spellings are semantically identical: for the same inputs the
/// standard-domain (`""`) and `com.microsoft` ops must produce byte-identical
/// CUDA outputs (same kernel, no numeric drift).
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn linear_attention_both_domains_identical() {
    let ep = require_cuda();
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        for cfg in configs() {
            let inputs = build_inputs(&cfg, dtype);
            let outputs = vec![
                (dtype, vec![cfg.batch, cfg.seq, cfg.output_hidden()]),
                (dtype, vec![cfg.batch, cfg.kv_num_heads, cfg.d_k, cfg.d_v]),
            ];
            let a = attrs(&cfg);
            let msft = run_cuda(
                &ep,
                "LinearAttention",
                "com.microsoft",
                OPSET,
                &inputs,
                &outputs,
                &a,
            );
            let std = run_cuda(&ep, "LinearAttention", "", OPSET, &inputs, &outputs, &a);
            assert_eq!(msft.len(), std.len(), "{}: output arity", cfg.label);
            for (idx, name) in ["output", "present_state"].iter().enumerate() {
                assert_eq!(
                    msft[idx], std[idx],
                    "{} [{name}] {dtype:?}: com.microsoft vs standard-domain bytes differ",
                    cfg.label
                );
            }
        }
    }
}

/// The recurrent state must genuinely carry across timesteps: running two
/// half-sequences chained through `past_state` must equal one full-sequence
/// run. This proves the CUDA present_state is a faithful continuation state,
/// not just a per-step artifact.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn linear_attention_state_carry_matches_chained() {
    let ep = require_cuda();
    let dtype = DataType::Float32;
    let (batch, d_k, d_v, heads) = (1usize, 5usize, 4usize, 2usize);
    let a = vec![
        ("q_num_heads", Attribute::Int(heads as i64)),
        ("kv_num_heads", Attribute::Int(heads as i64)),
        ("update_rule", Attribute::String(b"gated_delta".to_vec())),
        ("scale", Attribute::Float(0.7)),
    ];
    let out_hidden = heads * d_v;

    // Full 6-timestep run.
    let full_cfg = Config {
        label: "carry/full",
        batch,
        seq: 6,
        d_k,
        d_v,
        q_num_heads: heads,
        kv_num_heads: heads,
        n_k_heads: heads,
        update_rule: "gated_delta",
        scale: 0.7,
        decay_per_key_dim: false,
        beta_shared: false,
        with_past: false,
    };
    let full_inputs = build_inputs(&full_cfg, dtype);
    let full_outputs = vec![
        (dtype, vec![batch, 6, out_hidden]),
        (dtype, vec![batch, heads, d_k, d_v]),
    ];
    let full = run_cuda(
        &ep,
        "LinearAttention",
        DOMAIN,
        OPSET,
        &full_inputs,
        &full_outputs,
        &a,
    );
    let full_out = decode_floats(&full[0], dtype);

    // Split the same q/k/v/decay/beta into two halves of 3 timesteps and chain
    // the present_state of the first into the past_state of the second.
    let split_at = 3usize;
    let slice_seq = |t: &Tensor, start: usize, len: usize| -> Tensor {
        let per = t.shape[2];
        let row = t.dtype.storage_bytes(per);
        let mut bytes = Vec::new();
        for s in start..start + len {
            bytes.extend_from_slice(&t.bytes[s * row..(s + 1) * row]);
        }

        Tensor {
            dtype: t.dtype,
            shape: vec![t.shape[0], len, per],
            bytes,
        }
    };

    let first_inputs: Vec<Tensor> = {
        let mut v: Vec<Tensor> = full_inputs[..3]
            .iter()
            .map(|t| slice_seq(t, 0, split_at))
            .collect();
        v.push(full_inputs[3].clone()); // zero past_state
        v.push(slice_seq(&full_inputs[4], 0, split_at));
        v.push(slice_seq(&full_inputs[5], 0, split_at));
        v
    };
    let first_outputs = vec![
        (dtype, vec![batch, split_at, out_hidden]),
        (dtype, vec![batch, heads, d_k, d_v]),
    ];
    let first = run_cuda(
        &ep,
        "LinearAttention",
        DOMAIN,
        OPSET,
        &first_inputs,
        &first_outputs,
        &a,
    );

    let second_inputs: Vec<Tensor> = {
        let mut v: Vec<Tensor> = full_inputs[..3]
            .iter()
            .map(|t| slice_seq(t, split_at, 6 - split_at))
            .collect();
        // past_state = present_state from the first half.
        v.push(Tensor {
            dtype,
            shape: vec![batch, heads, d_k, d_v],
            bytes: first[1].clone(),
        });
        v.push(slice_seq(&full_inputs[4], split_at, 6 - split_at));
        v.push(slice_seq(&full_inputs[5], split_at, 6 - split_at));
        v
    };
    let second_outputs = vec![
        (dtype, vec![batch, 6 - split_at, out_hidden]),
        (dtype, vec![batch, heads, d_k, d_v]),
    ];
    let second = run_cuda(
        &ep,
        "LinearAttention",
        DOMAIN,
        OPSET,
        &second_inputs,
        &second_outputs,
        &a,
    );

    let first_out = decode_floats(&first[0], dtype);
    let second_out = decode_floats(&second[0], dtype);
    let mut chained = first_out;
    chained.extend_from_slice(&second_out);
    assert_close(
        "state_carry chained==full",
        dtype,
        &chained,
        &full_out,
        2e-4,
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn linear_attention_capture_replay_preserves_in_place_recurrent_state() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let cfg = Config {
        label: "capture-in-place-state",
        batch: 1,
        seq: 1,
        d_k: 5,
        d_v: 4,
        q_num_heads: 2,
        kv_num_heads: 2,
        n_k_heads: 2,
        update_rule: "gated_delta",
        scale: 0.5,
        decay_per_key_dim: false,
        beta_shared: false,
        with_past: true,
    };
    let inputs = build_inputs(&cfg, DataType::Float32);
    let outputs = vec![
        (
            DataType::Float32,
            vec![cfg.batch, cfg.seq, cfg.output_hidden()],
        ),
        (
            DataType::Float32,
            vec![cfg.batch, cfg.kv_num_heads, cfg.d_k, cfg.d_v],
        ),
    ];
    let attrs = attrs(&cfg);
    let (graph, node_id) = build_graph("LinearAttention", DOMAIN, OPSET, &inputs, &outputs, &attrs);
    let model = Model::new(&graph);
    let concrete_shapes = inputs
        .iter()
        .map(|tensor| tensor.shape.clone())
        .collect::<Vec<_>>();
    let kernel = ep
        .get_kernel(model.graph.node(node_id), &concrete_shapes, OPSET)
        .unwrap();
    let eager_kernel = ep
        .get_kernel(model.graph.node(node_id), &concrete_shapes, OPSET)
        .unwrap();
    let upload_inputs = || {
        inputs
            .iter()
            .map(|tensor| {
                let buffer = ep.allocate(tensor.bytes.len(), 256).unwrap();
                unsafe { runtime.htod(&tensor.bytes, cuptr(buffer.as_ptr())).unwrap() };
                buffer
            })
            .collect::<Vec<DeviceBuffer>>()
    };
    let mut captured_inputs = upload_inputs();
    let mut eager_inputs = upload_inputs();
    let input_strides = inputs
        .iter()
        .map(|tensor| compute_contiguous_strides(&tensor.shape))
        .collect::<Vec<_>>();
    let captured_views = inputs
        .iter()
        .zip(&captured_inputs)
        .zip(&input_strides)
        .map(|((tensor, buffer), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                tensor.dtype,
                &tensor.shape,
                strides,
                ep.device_id(),
            )
        })
        .collect::<Vec<_>>();
    let eager_views = inputs
        .iter()
        .zip(&eager_inputs)
        .zip(&input_strides)
        .map(|((tensor, buffer), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                tensor.dtype,
                &tensor.shape,
                strides,
                ep.device_id(),
            )
        })
        .collect::<Vec<_>>();
    let output_strides = outputs
        .iter()
        .map(|(_, shape)| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();
    let output_bytes = outputs[0].0.storage_bytes(outputs[0].1.iter().product());
    let state_bytes = inputs[3].bytes.len();
    let mut captured_output = ep.allocate(output_bytes, 256).unwrap();
    let mut eager_output = ep.allocate(output_bytes, 256).unwrap();
    let mut captured_outputs = vec![
        TensorMut::new(
            DevicePtrMut(captured_output.as_mut_ptr()),
            outputs[0].0,
            &outputs[0].1,
            &output_strides[0],
            ep.device_id(),
        ),
        TensorMut::new(
            DevicePtrMut(captured_inputs[3].as_mut_ptr()),
            outputs[1].0,
            &outputs[1].1,
            &output_strides[1],
            ep.device_id(),
        ),
    ];
    let mut eager_outputs = vec![
        TensorMut::new(
            DevicePtrMut(eager_output.as_mut_ptr()),
            outputs[0].0,
            &outputs[0].1,
            &output_strides[0],
            ep.device_id(),
        ),
        TensorMut::new(
            DevicePtrMut(eager_inputs[3].as_mut_ptr()),
            outputs[1].0,
            &outputs[1].1,
            &output_strides[1],
            ep.device_id(),
        ),
    ];
    let read = |buffer: &DeviceBuffer, len: usize| {
        let mut bytes = vec![0; len];
        unsafe {
            runtime.dtoh(&mut bytes, cuptr(buffer.as_ptr())).unwrap();
        }
        bytes
    };
    let overwrite = |buffer: &DeviceBuffer, bytes: &[u8]| unsafe {
        runtime.htod(bytes, cuptr(buffer.as_ptr())).unwrap();
    };

    kernel
        .execute(&captured_views, &mut captured_outputs)
        .unwrap();
    eager_kernel
        .execute(&eager_views, &mut eager_outputs)
        .unwrap();
    assert_eq!(
        read(&captured_inputs[3], state_bytes),
        read(&eager_inputs[3], state_bytes),
        "warmup in-place recurrent state must match eager"
    );

    let allocation_counts = runtime.allocation_counts();
    let kernels = [kernel.as_ref()];
    runtime.begin_graph_capture(&kernels).unwrap();
    kernel
        .execute(&captured_views, &mut captured_outputs)
        .unwrap();
    runtime.end_graph_capture().unwrap();

    for step in 1_u64..=4 {
        for index in [0_usize, 1, 2, 4, 5] {
            let count = inputs[index].shape.iter().product();
            let mut replacement = inputs[index].clone();
            replacement.bytes = encode_floats(
                &values(100 * step + index as u64, count, -0.8, 0.8),
                replacement.dtype,
            );
            overwrite(&captured_inputs[index], &replacement.bytes);
            overwrite(&eager_inputs[index], &replacement.bytes);
        }
        eager_kernel
            .execute(&eager_views, &mut eager_outputs)
            .unwrap();
        runtime.replay_graph().unwrap();
        assert_eq!(
            read(&captured_output, output_bytes),
            read(&eager_output, output_bytes),
            "captured output diverged at recurrent decode step {step}"
        );
        assert_eq!(
            read(&captured_inputs[3], state_bytes),
            read(&eager_inputs[3], state_bytes),
            "captured in-place recurrent state diverged at decode step {step}"
        );
    }

    assert_eq!(runtime.allocation_counts(), allocation_counts);
    assert!(runtime.reset_graph().unwrap());
    drop(captured_outputs);
    drop(eager_outputs);
    for buffer in captured_inputs {
        ep.deallocate(buffer).unwrap();
    }
    for buffer in eager_inputs {
        ep.deallocate(buffer).unwrap();
    }
    ep.deallocate(captured_output).unwrap();
    ep.deallocate(eager_output).unwrap();
}
