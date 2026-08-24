//! Standalone activation microbenchmark, used to justify
//! `kernels::simd_activations`.
//!
//! Prints one `name<TAB>ns_per_element` line per case so two builds of this
//! benchmark can be run **interleaved** and diffed. Cross-worktree comparison
//! is not valid on a shared runner (a uniform 0.70-0.82x offset was observed
//! on byte-identical kernels), so the intended workflow is:
//!
//! ```text
//! cargo build --release --bench activation_bench -p onnx-runtime-ep-cpu
//! cp target/release/deps/activation_bench-* after
//! git stash && cargo build --release --bench activation_bench -p onnx-runtime-ep-cpu
//! cp target/release/deps/activation_bench-* before && git stash pop
//! for i in $(seq 10); do taskset -c 0-15 ./before; taskset -c 0-15 ./after; done
//! ```
//!
//! `Sqrt` and `Erf` are included as in-run noise controls: they are untouched
//! by this change, so any reported speedup must exceed their concurrently
//! measured spread to be real.

mod common;

use std::hint::black_box;
use std::time::Instant;

use common::{FloatDType, Tensor};
use onnx_runtime_ep_api::{ExecutionProvider, Kernel, TensorView};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{Attribute, Node, NodeId};

/// `common::make_kernel` does not set `node.domain`, which contrib ops need.
fn kernel(
    op: &str,
    domain: &str,
    attrs: &[(&str, Attribute)],
    shapes: &[Vec<usize>],
    opset: u64,
) -> Box<dyn Kernel> {
    let mut node = Node::new(NodeId(0), op, vec![], vec![]);
    node.domain = domain.to_string();
    for (name, value) in attrs {
        node.attributes.insert((*name).into(), value.clone());
    }
    CpuExecutionProvider::new()
        .get_kernel(&node, shapes, opset)
        .expect("kernel must build")
}

/// Deterministic inputs spanning the interesting range of every kernel here,
/// including both saturation bands and values on either side of them.
fn inputs(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = (i as f32) / (len as f32);
            (t * 44.0 - 22.0) * (1.0 + 0.37 * ((i % 17) as f32 / 17.0))
        })
        .collect()
}

fn time(name: &str, kernel: &dyn Kernel, ins: &[TensorView], out: &mut Tensor, elems: usize) {
    // Warm up caches, branch predictors and any first-call dispatch.
    for _ in 0..8 {
        kernel
            .execute(black_box(ins), black_box(&mut [out.view_mut()]))
            .unwrap();
    }
    let iters = (1 << 24) / elems.max(1) + 4;
    let start = Instant::now();
    for _ in 0..iters {
        kernel
            .execute(black_box(ins), black_box(&mut [out.view_mut()]))
            .unwrap();
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "{name}\t{:.4}",
        elapsed * 1e9 / (iters as f64 * elems as f64)
    );
}

fn main() {
    // Match the decode thread topology a served session runs in (#1749).
    common::init_decode_topology();
    // Opened before anything else runs, so the window covers warmup too: a
    // warmup that shared cores with somebody else's run leaves caches and
    // frequency in a state the timed region inherits.
    let host_lock = common::open_host_lock_window();

    // Decode (`[1, 1, H]`), prefill (`[1, S, H]`) and two sizes around the
    // vector dispatch threshold.
    let shapes: [(&str, Vec<usize>); 5] = [
        ("tiny16", vec![16]),
        ("small64", vec![64]),
        ("decode3072", vec![1, 1, 3072]),
        ("decode4096", vec![1, 1, 4096]),
        ("prefill512x4096", vec![1, 512, 4096]),
    ];

    for dtype in [FloatDType::F32, FloatDType::F16, FloatDType::Bf16] {
        for (label, shape) in &shapes {
            let elems: usize = shape.iter().product();
            let x = Tensor::floats(dtype, shape, &inputs(elems));
            let mut out = Tensor::zeros(dtype, shape);

            for (op, domain, attrs, opset) in [
                ("Tanh", "", vec![], 13),
                ("Sigmoid", "", vec![], 13),
                ("QuickGelu", "com.microsoft", vec![], 1),
                ("FastGelu", "com.microsoft", vec![], 1),
                (
                    "Gelu",
                    "",
                    vec![("approximate", Attribute::String("tanh".into()))],
                    20,
                ),
                // Controls: untouched by this change.
                ("Sqrt", "", vec![], 13),
                ("Erf", "", vec![], 13),
            ] {
                let k = kernel(op, domain, &attrs, std::slice::from_ref(shape), opset);
                time(
                    &format!("{op}/{}/{label}", dtype.name()),
                    k.as_ref(),
                    &[x.view()],
                    &mut out,
                    elems,
                );
            }
        }
    }

    // Last, so the second reading covers everything above it.
    common::report_host_lock(host_lock);
}
