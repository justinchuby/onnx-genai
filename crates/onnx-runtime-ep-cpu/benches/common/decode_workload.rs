//! The decode projection chain both decode-loop benches drive.
//!
//! Moved here verbatim from `int4_decode_loop_ab` so the gap-aware harness in
//! `decode_gap_park_ab` measures **the same work**, not a re-implementation of
//! it. That is load-bearing rather than tidy: the gap harness's zero-gap row is
//! only a control for the existing bench if the two chains are byte-identical,
//! and two copies of a 120-line weight setup drift silently. A control that has
//! quietly stopped controlling is the failure mode this whole line of work keeps
//! running into.

use onnx_runtime_ep_api::{ExecutionProvider, Kernel};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{Attribute, Node, NodeId};

use super::{FloatDType, Tensor};

/// One decode step's projections for a llama3-8B-shaped model, as `(k, n)`.
/// A decode token pays all of these back to back, which is what makes the
/// per-op fork/join cost a per-token cost rather than a one-off.
pub const PROJECTIONS: &[(usize, usize, &str)] = &[
    (4096, 6144, "qkv"),
    (4096, 4096, "o"),
    (4096, 14336, "gate"),
    (4096, 14336, "up"),
    (14336, 4096, "down"),
];

/// Qwen2.5-7B's decode projections. Not a cosmetic second model: its GQA head
/// layout makes `qkv` **narrow** (n = 4608 against a k of 3584) and its MLP
/// **much wider** relative to the hidden size (18944 vs llama's 14336 on a
/// larger k). Both differences move the `n`-loop trip count, which is exactly
/// the axis the N-blocked kernel's four-column grouping divides. A conclusion
/// drawn only from llama shapes has not been tested against a different
/// n/k ratio at all, and the tail behaviour at `n % 4` is invisible in a set
/// where every `n` is a multiple of 4.
pub const PROJECTIONS_QWEN: &[(usize, usize, &str)] = &[
    (3584, 4608, "qkv"),
    (3584, 3584, "o"),
    (3584, 18944, "gate"),
    (3584, 18944, "up"),
    (18944, 3584, "down"),
];

pub fn projections() -> &'static [(usize, usize, &'static str)] {
    match std::env::var("PROBE_MODEL")
        .unwrap_or_else(|_| "llama".into())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "qwen" => PROJECTIONS_QWEN,
        "llama" => PROJECTIONS,
        other => panic!("PROBE_MODEL must be llama or qwen, got {other:?}"),
    }
}

pub fn packed_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        })
        .collect()
}

pub fn floats(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * 0.0137 + seed).sin() * 0.5)
        .collect()
}

pub fn build_kernel(k: usize, n: usize, block_size: usize, accuracy: i64) -> Box<dyn Kernel> {
    let blocks = k.div_ceil(block_size);
    let blob = block_size / 2;
    let shapes = vec![vec![1, k], vec![n, blocks, blob], vec![n, blocks]];
    let mut node = Node::new(NodeId(0), "MatMulNBits", vec![], vec![]);
    node.domain = "com.microsoft".into();
    for (name, value) in [
        ("K", Attribute::Int(k as i64)),
        ("N", Attribute::Int(n as i64)),
        ("bits", Attribute::Int(4)),
        ("block_size", Attribute::Int(block_size as i64)),
        ("accuracy_level", Attribute::Int(accuracy)),
    ] {
        node.attributes.insert(name.into(), value);
    }
    let mut kernel = CpuExecutionProvider::new()
        .get_kernel(&node, &shapes, 1)
        .expect("CPU EP must register MatMulNBits");
    kernel.set_constant_inputs(&[false, true, true]);
    kernel
}

/// A weight, shared across sessions the way a served model's weights are.
pub struct Weight {
    pub b: Tensor,
    pub scales: Tensor,
    pub k: usize,
    pub n: usize,
}

/// Build the shared, deterministic weight set for one configuration.
pub fn weights(block_size: usize) -> Vec<Weight> {
    projections()
        .iter()
        .enumerate()
        .map(|(index, &(k, n, _))| {
            let blocks = k.div_ceil(block_size);
            let blob = block_size / 2;
            let packed = packed_bytes(n * blocks * blob, 7 + index as u64);
            let scales = floats(n * blocks, 0.3)
                .into_iter()
                .map(|v| v.abs().max(0.01) * 0.02)
                .collect::<Vec<_>>();
            Weight {
                b: Tensor::u8(&[n, blocks, blob], &packed),
                scales: Tensor::floats(FloatDType::F32, &[n, blocks], &scales),
                k,
                n,
            }
        })
        .collect()
}
