//! P2d measurement harness (design `docs/NATIVE_LORA_DESIGN.md` §J) — the
//! deliverable that decides whether the XL fused BGMV/SGMV grouped kernel is
//! worth building.
//!
//! Measures, on CPU, in fp32 (both paths use fp32 accumulators regardless of
//! factor dtype, so f32 factors isolate the group-by-adapter overhead from
//! widening cost):
//!
//! (a) single-adapter: the `GroupedLoraDelta` group-by-adapter path vs the
//!     Phase-1 delta path (two `MatMul`s + a scale `Mul` — the two share the
//!     final `Add` onto the base, so it is excluded from both), decode-shaped
//!     (tokens = 1) and prefill-shaped (tokens = 128), across representative
//!     K/N/rank. GATE: the grouped path must be <= the Phase-1 path.
//!
//! (b) multi-adapter batch: the group-by-adapter path at batch sizes {2,4,8,16}
//!     with {2,4,8} distinct adapters — the data that decides whether a fused
//!     grouped kernel is worth building. Reports tokens/sec and per-token
//!     overhead.
//!
//! Run with:
//! ```text
//! cargo run -p onnx-runtime-ep-cpu --release --bin lora_grouped_bench
//! ```

use std::sync::Arc;
use std::time::Instant;

use onnx_runtime_ep_api::{
    AdapterId, DevicePtr, DevicePtrMut, ExecutionProvider, Kernel, LoraFactorInput, LoraModuleId,
    LoraPoolId, LoraPoolRegistry, LoraWeightPool, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{compute_contiguous_strides, Attribute, DataType, DeviceId, Node, NodeId};

/// An owned, contiguous f32 tensor with cached strides.
struct F32Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<i64>,
}

impl F32Tensor {
    fn new(shape: &[usize], data: Vec<f32>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        Self {
            data,
            shape: shape.to_vec(),
            strides: compute_contiguous_strides(shape),
        }
    }

    fn filled(shape: &[usize]) -> Self {
        let len = shape.iter().product::<usize>();
        let data = (0..len)
            .map(|i| ((i % 251) as i32 - 125) as f32 / 64.0)
            .collect();
        Self::new(shape, data)
    }

    fn zeros(shape: &[usize]) -> Self {
        let len = shape.iter().product::<usize>();
        Self::new(shape, vec![0.0; len])
    }

    fn view(&self) -> TensorView<'_> {
        TensorView::new(
            DevicePtr(self.data.as_ptr().cast()),
            DataType::Float32,
            &self.shape,
            &self.strides,
            DeviceId::cpu(),
        )
    }

    fn view_mut(&mut self) -> TensorMut<'_> {
        TensorMut::new(
            DevicePtrMut(self.data.as_mut_ptr().cast()),
            DataType::Float32,
            &self.shape,
            &self.strides,
            DeviceId::cpu(),
        )
    }

    fn bytes(&self) -> Vec<u8> {
        self.data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

struct I32Tensor {
    data: Vec<i32>,
    shape: Vec<usize>,
    strides: Vec<i64>,
}

impl I32Tensor {
    fn new(shape: &[usize], data: Vec<i32>) -> Self {
        Self {
            data,
            shape: shape.to_vec(),
            strides: compute_contiguous_strides(shape),
        }
    }

    fn view(&self) -> TensorView<'_> {
        TensorView::new(
            DevicePtr(self.data.as_ptr().cast()),
            DataType::Int32,
            &self.shape,
            &self.strides,
            DeviceId::cpu(),
        )
    }
}

fn matmul_kernel(m: usize, k: usize, n: usize) -> Box<dyn Kernel> {
    let node = Node::new(NodeId(0), "MatMul", vec![], vec![]);
    CpuExecutionProvider::new()
        .get_kernel(&node, &[vec![m, k], vec![k, n]], 17)
        .expect("MatMul kernel")
}

fn mul_kernel(m: usize, n: usize) -> Box<dyn Kernel> {
    let node = Node::new(NodeId(0), "Mul", vec![], vec![]);
    CpuExecutionProvider::new()
        .get_kernel(&node, &[vec![m, n], vec![]], 14)
        .expect("Mul kernel")
}

fn grouped_kernel(k: usize, n: usize, max_rank: usize, pool_id: LoraPoolId) -> Box<dyn Kernel> {
    let mut node = Node::new(NodeId(0), "GroupedLoraDelta", vec![], vec![]);
    node.domain = "pkg.nxrt".to_string();
    node.attributes.insert("K".into(), Attribute::Int(k as i64));
    node.attributes.insert("N".into(), Attribute::Int(n as i64));
    node.attributes.insert("module_id".into(), Attribute::Int(0));
    node.attributes
        .insert("max_rank".into(), Attribute::Int(max_rank as i64));
    node.attributes
        .insert("pool_id".into(), Attribute::Int(pool_id.0 as i64));
    CpuExecutionProvider::new()
        .get_kernel(&node, &[vec![1, k], vec![1]], 1)
        .expect("GroupedLoraDelta kernel")
}

/// Register a pool holding `adapters` distinct adapters, each with one module of
/// the given rank/shape, under module id 0. Returns the pool id.
fn register_pool(adapters: usize, k: usize, n: usize, rank: usize) -> (LoraPoolId, Arc<LoraWeightPool>) {
    let a = F32Tensor::filled(&[k, rank]);
    let b = F32Tensor::filled(&[rank, n]);
    let a_bytes = a.bytes();
    let b_bytes = b.bytes();
    let per_pair = LoraWeightPool::page_pair_resident_bytes(a_bytes.len(), b_bytes.len());
    let mut pool = LoraWeightPool::with_capacity_bytes(per_pair * adapters as u64 + 4096);
    for adapter in 0..adapters as u64 {
        pool.admit(
            AdapterId(adapter),
            LoraModuleId(0),
            LoraFactorInput {
                dtype: DataType::Float32,
                rows: k,
                cols: rank,
                bytes: &a_bytes,
            },
            LoraFactorInput {
                dtype: DataType::Float32,
                rows: rank,
                cols: n,
                bytes: &b_bytes,
            },
            1.0,
        )
        .expect("admit adapter");
    }
    let pool = Arc::new(pool);
    let id = LoraPoolRegistry::global().register(Arc::clone(&pool));
    (id, pool)
}

/// Time `f` adaptively: warm up, then loop until at least `min_seconds` elapse,
/// returning mean seconds per call.
fn timed(min_seconds: f64, mut f: impl FnMut()) -> f64 {
    for _ in 0..5 {
        f();
    }
    let mut iters: u64 = 0;
    let start = Instant::now();
    loop {
        f();
        iters += 1;
        if start.elapsed().as_secs_f64() >= min_seconds {
            break;
        }
    }
    start.elapsed().as_secs_f64() / iters as f64
}

/// One (K, N, rank) shape the benchmark sweeps.
struct Shape {
    label: &'static str,
    k: usize,
    n: usize,
    rank: usize,
}

const SHAPES: &[Shape] = &[
    Shape { label: "attn q_proj 2048x2048 r8", k: 2048, n: 2048, rank: 8 },
    Shape { label: "attn q_proj 2048x2048 r16", k: 2048, n: 2048, rank: 16 },
    Shape { label: "attn q_proj 4096x4096 r16", k: 4096, n: 4096, rank: 16 },
    Shape { label: "mlp gate 4096x11008 r16", k: 4096, n: 11008, rank: 16 },
];

fn bench_single_adapter(tokens: usize) {
    println!("\n### Single-adapter, tokens = {tokens}\n");
    println!("| shape | K | N | rank | Phase-1 (µs) | grouped (µs) | grouped/Phase-1 |");
    println!("|---|---|---|---|---|---|---|");
    for shape in SHAPES {
        let (k, n, rank) = (shape.k, shape.n, shape.rank);
        let (pool_id, _pool) = register_pool(1, k, n, rank);

        let x = F32Tensor::filled(&[tokens, k]);
        let a = F32Tensor::filled(&[k, rank]);
        let b = F32Tensor::filled(&[rank, n]);
        let scale = F32Tensor::new(&[], vec![0.5]);
        let mut r_buf = F32Tensor::zeros(&[tokens, rank]);
        let mut delta_buf = F32Tensor::zeros(&[tokens, n]);
        let mut scaled_buf = F32Tensor::zeros(&[tokens, n]);
        let segments = I32Tensor::new(&[tokens], vec![0; tokens]);
        let mut grouped_out = F32Tensor::zeros(&[tokens, n]);

        let mm1 = matmul_kernel(tokens, k, rank);
        let mm2 = matmul_kernel(tokens, rank, n);
        let mul = mul_kernel(tokens, n);
        let grouped = grouped_kernel(k, n, rank, pool_id);

        let phase1 = timed(0.3, || {
            mm1.execute(&[x.view(), a.view()], &mut [r_buf.view_mut()]).unwrap();
            mm2.execute(&[r_buf.view(), b.view()], &mut [delta_buf.view_mut()]).unwrap();
            mul.execute(&[delta_buf.view(), scale.view()], &mut [scaled_buf.view_mut()]).unwrap();
        });
        let grouped_time = timed(0.3, || {
            grouped
                .execute(&[x.view(), segments.view()], &mut [grouped_out.view_mut()])
                .unwrap();
        });

        LoraPoolRegistry::global().unregister(pool_id);
        println!(
            "| {} | {} | {} | {} | {:.2} | {:.2} | {:.2}x |",
            shape.label,
            k,
            n,
            rank,
            phase1 * 1e6,
            grouped_time * 1e6,
            grouped_time / phase1,
        );
    }
}

fn bench_multi_adapter() {
    println!("\n### Multi-adapter batch (group-by-adapter path)\n");
    println!("| shape | batch | adapters | total (µs) | per-token (µs) | tokens/sec |");
    println!("|---|---|---|---|---|---|");
    // A representative attention projection and an MLP projection.
    let shapes: &[&Shape] = &[&SHAPES[2], &SHAPES[3]];
    for shape in shapes {
        let (k, n, rank) = (shape.k, shape.n, shape.rank);
        for &batch in &[2usize, 4, 8, 16] {
            for &adapters in &[2usize, 4, 8] {
                if adapters > batch {
                    continue;
                }
                let (pool_id, _pool) = register_pool(adapters, k, n, rank);
                let x = F32Tensor::filled(&[batch, k]);
                // Round-robin rows across the distinct adapters.
                let seg: Vec<i32> = (0..batch).map(|i| (i % adapters) as i32).collect();
                let segments = I32Tensor::new(&[batch], seg);
                let mut out = F32Tensor::zeros(&[batch, n]);
                let grouped = grouped_kernel(k, n, rank, pool_id);

                let total = timed(0.3, || {
                    grouped
                        .execute(&[x.view(), segments.view()], &mut [out.view_mut()])
                        .unwrap();
                });
                LoraPoolRegistry::global().unregister(pool_id);
                println!(
                    "| {} | {} | {} | {:.2} | {:.3} | {:.0} |",
                    shape.label,
                    batch,
                    adapters,
                    total * 1e6,
                    total * 1e6 / batch as f64,
                    batch as f64 / total,
                );
            }
        }
    }
}

fn main() {
    println!("# LoRA Phase-2 group-by-adapter measurements\n");
    println!(
        "CPU: {} logical cores. All timings are mean wall-clock per kernel call, fp32.",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0)
    );
    bench_single_adapter(1);
    bench_single_adapter(128);
    bench_multi_adapter();
}
