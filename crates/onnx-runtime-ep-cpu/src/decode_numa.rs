//! Two-level NUMA-split decode layout for the bounded M=1 decode pool.
//!
//! [`crate::decode_affinity`] can pin the whole decode pool to a *single* NUMA
//! node (`compact`), which keeps every per-op fork-join barrier and the streamed
//! int4 weights node-local but leaves the second socket's memory bandwidth
//! unused. Reaching both sockets' bandwidth naively -- one flat pool spanning
//! both nodes -- regresses badly, because every one of the ~141 per-token
//! `MatMulNBits` projections then closes with a barrier whose participants span
//! both sockets, so each op pays a cross-socket cache-coherency round trip.
//!
//! `numa-split` instead builds one node-pinned sub-pool per NUMA node and shards
//! each projection's *output rows* across them. Row-sharding a GEMV is exactly
//! associative -- each output row is an independent dot product over the whole
//! K dimension, so concatenating the per-node row slices reproduces the flat
//! result bit-for-bit, with no cross-node reduction. The per-node weight rows
//! are first-touched on their owning node, so each sub-pool streams node-local
//! memory. The only cross-socket synchronization is a *single* join per op
//! (a two-level barrier: node-local reductions first, then one cross-node
//! combine) instead of a flat N-way cross-socket barrier.
//!
//! Topology is queried at runtime (never hardcoded) and the whole layout is
//! opt-in behind `ONNX_GENAI_CPU_DECODE_AFFINITY=numa-split`; a single-node or
//! non-Linux host, an opted-out pool, or a topology that cannot be split falls
//! back (logged once) to the flat single-node decode path.

use std::sync::OnceLock;

use rayon::prelude::*;

use crate::decode_affinity::{DecodeAffinity, NodeShard, NumaTopology};
use crate::kernels::matmul_nbits::output_chunk_len;

/// One NUMA node's decode sub-pool, pinned to that node's CPUs.
struct NodePool {
    /// Number of workers pinned to this node (drives proportional row splits).
    workers: usize,
    /// The node-pinned Rayon pool that runs this node's row shard.
    pool: rayon::ThreadPool,
}

/// The `numa-split` decode layout: a tiny dispatcher pool that fans each op out
/// to one node-pinned sub-pool per NUMA node, joined by a single cross-node
/// barrier.
pub struct NumaDecodePools {
    /// Drives the per-op fan-out; sized to one worker per node so all node
    /// sub-pools run concurrently under a single join.
    dispatcher: rayon::ThreadPool,
    nodes: Vec<NodePool>,
    total_workers: usize,
}

impl NumaDecodePools {
    /// Build the per-node sub-pools plus the dispatcher pool from a worker
    /// split. Each node pool pins its workers to distinct CPUs of its node.
    fn build(shards: &[NodeShard]) -> std::result::Result<Self, String> {
        let mut nodes = Vec::with_capacity(shards.len());
        let mut total_workers = 0;
        for shard in shards {
            let pool = build_node_pool(shard)?;
            total_workers += shard.workers;
            nodes.push(NodePool {
                workers: shard.workers,
                pool,
            });
        }
        let dispatcher = rayon::ThreadPoolBuilder::new()
            .num_threads(nodes.len())
            .thread_name(|index| format!("onnx-genai-decode-dispatch-{index}"))
            .build()
            .map_err(|err| format!("failed to build numa-split dispatcher pool: {err}"))?;
        Ok(Self {
            dispatcher,
            nodes,
            total_workers,
        })
    }

    /// Total decode workers across all node sub-pools.
    pub fn total_workers(&self) -> usize {
        self.total_workers
    }

    /// Number of NUMA nodes in the layout.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Run the whole single-token forward on the dispatcher pool so each op's
    /// per-node fan-out (`dispatch_output_rows`) executes on already-woken
    /// dispatcher workers rather than crossing in from an external thread.
    pub fn install_scope<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        self.dispatcher.install(f)
    }

    /// Split `n` output rows across the nodes proportionally to their worker
    /// counts, contiguous and non-overlapping, the last node absorbing the
    /// remainder. Weight placement and compute dispatch both use this so a
    /// node's workers only ever read that node's weight rows.
    fn row_lengths(&self, n: usize) -> Vec<usize> {
        let mut lengths = Vec::with_capacity(self.nodes.len());
        let mut assigned = 0;
        for (position, node) in self.nodes.iter().enumerate() {
            let rows = if position + 1 == self.nodes.len() {
                n - assigned
            } else {
                (n.saturating_mul(node.workers)) / self.total_workers
            };
            assigned += rows;
            lengths.push(rows);
        }
        lengths
    }

    /// Shard `result`'s output rows across the node sub-pools and run `compute`
    /// on each node's contiguous slice, joined by a single cross-node barrier.
    ///
    /// `compute(output_start, outputs)` fills `outputs` with the projection rows
    /// `output_start .. output_start + outputs.len()`; it is the same closure
    /// the flat path passes to `par_chunks_mut`, so the math is identical.
    pub fn dispatch_output_rows<F>(&self, result: &mut [f32], k: usize, compute: &F)
    where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        let n = result.len();
        let lengths = self.row_lengths(n);
        let mut segments: Vec<(usize, usize, &mut [f32])> = Vec::with_capacity(self.nodes.len());
        let mut rest = result;
        let mut output_start = 0;
        for (position, len) in lengths.into_iter().enumerate() {
            let (head, tail) = rest.split_at_mut(len);
            segments.push((position, output_start, head));
            output_start += len;
            rest = tail;
        }
        // The dispatcher pool has one worker per node, so each segment installs
        // onto its node sub-pool concurrently; the `for_each` join is the only
        // cross-node synchronization point (the second barrier level).
        self.dispatcher.install(|| {
            segments
                .into_par_iter()
                .for_each(|(position, output_start, segment)| {
                    self.nodes[position].pool.install(|| {
                        let chunk = output_chunk_len(segment.len(), k);
                        if chunk < segment.len() {
                            segment.par_chunks_mut(chunk).enumerate().for_each(
                                |(chunk_index, outputs)| {
                                    compute(output_start + chunk_index * chunk, outputs)
                                },
                            );
                        } else {
                            compute(output_start, segment);
                        }
                    });
                });
        });
    }

    /// Copy `src` into a fresh buffer whose per-node row shards are first-touched
    /// on their owning node, so each sub-pool later streams node-local memory.
    ///
    /// `src` is a row-major `[n, stride]` weight component; the row split matches
    /// [`Self::row_lengths`] exactly so it lines up with `dispatch_output_rows`.
    pub fn place_rows<T: Copy + Send + Sync>(&self, src: &[T], n: usize) -> Vec<T> {
        if n == 0 || src.is_empty() {
            return src.to_vec();
        }
        let stride = src.len() / n;
        debug_assert_eq!(stride * n, src.len());
        let mut dst: Vec<T> = Vec::with_capacity(src.len());
        // Deliberately leave the buffer uninitialized: zero-filling here would
        // fault every page onto the *current* (dispatcher) node, defeating the
        // node-local first-touch performed by the pinned workers below.
        // SAFETY: `T: Copy` has no `Drop`, and every element is overwritten by
        // the `copy_from_slice` calls below before the buffer is observed. The
        // capacity was just reserved for exactly `src.len()` elements.
        #[allow(clippy::uninit_vec)]
        unsafe {
            dst.set_len(src.len());
        }
        let lengths = self.row_lengths(n);
        let mut dst_rest = dst.as_mut_slice();
        let mut src_offset = 0;
        let mut segments: Vec<(usize, &mut [T], &[T])> = Vec::with_capacity(self.nodes.len());
        for (position, rows) in lengths.into_iter().enumerate() {
            let len = rows * stride;
            let (dst_head, dst_tail) = dst_rest.split_at_mut(len);
            segments.push((position, dst_head, &src[src_offset..src_offset + len]));
            src_offset += len;
            dst_rest = dst_tail;
        }
        // First-touch each shard on its node: writing from a node-pinned worker
        // faults the destination pages onto that node under the default policy.
        self.dispatcher.install(|| {
            segments
                .into_par_iter()
                .for_each(|(position, dst_segment, src_segment)| {
                    self.nodes[position].pool.install(|| {
                        dst_segment.copy_from_slice(src_segment);
                    });
                });
        });
        dst
    }
}

/// Build one node-pinned Rayon sub-pool for `shard`.
fn build_node_pool(shard: &NodeShard) -> std::result::Result<rayon::ThreadPool, String> {
    let cpus = shard.cpus.clone();
    let node_index = shard.index;
    rayon::ThreadPoolBuilder::new()
        .num_threads(shard.workers)
        .thread_name(move |worker| format!("onnx-genai-decode-n{node_index}-{worker}"))
        .start_handler(move |worker_index| {
            let cpu = cpus[worker_index % cpus.len()];
            if let Err(message) = crate::decode_affinity::pin_current_thread_to_cpu(cpu) {
                report_numa_fallback(&format!(
                    "node {node_index} worker could not pin to cpu {cpu}: {message}"
                ));
            }
        })
        .build()
        .map_err(|err| format!("failed to build numa-split node {node_index} pool: {err}"))
}

/// Build the `numa-split` layout when the environment requests it and the host
/// can be split, returning `None` (logged once) for every fallback case so the
/// caller keeps the flat single-node decode path.
pub fn build_from_env(threads: Option<usize>) -> Option<NumaDecodePools> {
    // A malformed value is surfaced as a hard error by the flat pool builder
    // (`decode_affinity_cpus`); here we only need to detect a valid `numa-split`
    // request, so we resolve against no topology (`None`) -- topology-dependent
    // validation of other modes is the flat path's job. Using `resolve` (rather
    // than the deprecated `from_env`) keeps this forward-compatible with the
    // env-boundary API.
    let raw = std::env::var(crate::decode_affinity::DECODE_AFFINITY_ENV).ok();
    if DecodeAffinity::resolve(raw.as_deref(), None).ok()? != DecodeAffinity::NumaSplit {
        return None;
    }
    let Some(total) = threads else {
        report_numa_fallback(
            "ONNX_GENAI_CPU_DECODE_THREADS=0 opts out of the bounded pool; \
             numa-split needs a bounded worker count -- using the global pool",
        );
        return None;
    };
    let Some(topology) = NumaTopology::detect() else {
        report_numa_fallback(
            "host exposes no multi-node topology (single node or a platform without \
             discoverable NUMA topology); numa-split falls back to flat single-node decode",
        );
        return None;
    };
    // cgroup/cpuset/taskset safety (gap #7): only ever pin to CPUs the process is
    // actually allowed to run on. Intersect the discovered topology with the
    // process's allowed CPU set before splitting so a container-restricted host
    // never tries to pin outside its cpuset.
    let allowed = crate::decode_affinity::allowed_cpus();
    let topology = topology.restrict_to_allowed(allowed.as_deref());
    let Some(shards) = topology.split_workers(total) else {
        report_numa_fallback(
            "fewer than two NUMA nodes received workers (single-node host, or the process \
             cpuset spans fewer than two nodes); numa-split falls back to flat single-node decode",
        );
        return None;
    };
    match NumaDecodePools::build(&shards) {
        Ok(pools) => Some(pools),
        Err(message) => {
            report_numa_fallback(&message);
            None
        }
    }
}

/// Log the first numa-split fallback/pinning problem once so a restricted or
/// unsupported host surfaces the reason without spamming every worker/op.
fn report_numa_fallback(message: &str) {
    static REPORTED: OnceLock<()> = OnceLock::new();
    if REPORTED.set(()).is_ok() {
        eprintln!("onnx-genai: numa-split decode unavailable; {message}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_node_pools() -> NumaDecodePools {
        let shards = vec![
            NodeShard {
                index: 0,
                cpus: vec![0, 1],
                workers: 2,
            },
            NodeShard {
                index: 1,
                cpus: vec![2, 3],
                workers: 2,
            },
        ];
        NumaDecodePools::build(&shards).expect("build test numa pools")
    }

    fn asymmetric_two_node_pools() -> NumaDecodePools {
        let shards = vec![
            NodeShard {
                index: 0,
                cpus: vec![0],
                workers: 1,
            },
            NodeShard {
                index: 1,
                cpus: vec![1, 2, 3],
                workers: 3,
            },
        ];
        NumaDecodePools::build(&shards).expect("build asymmetric test numa pools")
    }

    #[test]
    fn row_lengths_split_proportionally_and_cover_all_rows() {
        let pools = two_node_pools();
        let lengths = pools.row_lengths(100);
        assert_eq!(lengths.len(), 2);
        assert_eq!(lengths.iter().sum::<usize>(), 100);
        // Equal worker counts split evenly.
        assert_eq!(lengths, vec![50, 50]);
        // The last node absorbs an odd remainder.
        assert_eq!(pools.row_lengths(101), vec![50, 51]);
        // Fewer rows than nodes still covers exactly, last node takes the rest.
        assert_eq!(pools.row_lengths(1), vec![0, 1]);
    }

    #[test]
    fn dispatch_output_rows_matches_flat_computation() {
        let pools = two_node_pools();
        let n = 37usize;
        let k = 8usize;
        // A deterministic per-row function so we can check exact concatenation.
        let compute = |output_start: usize, outputs: &mut [f32]| {
            for (offset, out) in outputs.iter_mut().enumerate() {
                *out = (output_start + offset) as f32 * 3.0 - 1.0;
            }
        };
        let mut sharded = vec![0.0f32; n];
        pools.dispatch_output_rows(&mut sharded, k, &compute);
        let mut flat = vec![0.0f32; n];
        compute(0, &mut flat);
        assert_eq!(sharded, flat);
    }

    #[test]
    fn place_rows_preserves_bytes() {
        let pools = two_node_pools();
        let n = 5usize;
        let stride = 4usize;
        let src: Vec<u8> = (0..(n * stride) as u8).collect();
        let placed = pools.place_rows(&src, n);
        assert_eq!(placed, src);

        let scales: Vec<f32> = (0..n).map(|row| row as f32 * 0.5).collect();
        let placed_scales = pools.place_rows(&scales, n);
        assert_eq!(placed_scales, scales);
    }

    /// Regression guard for the core `numa-split` correctness claim: sharding a
    /// GEMV's output rows across the per-node sub-pools and joining them must
    /// reproduce the flat single-threaded GEMV **bit-for-bit**. A GEMV row is an
    /// independent dot product over the whole K dimension, so row-sharding is
    /// exactly associative -- there is no cross-row reduction and therefore no
    /// floating-point re-association. Any divergence here means a real
    /// cross-node reduction or a non-associative combine has crept in.
    #[test]
    fn dispatch_output_rows_matches_flat_gemv_bit_for_bit() {
        let pools = two_node_pools();
        let n = 129usize; // not a multiple of the node/worker counts
        let k = room_k();
        // Deterministic, non-trivial weights and activation (values chosen so the
        // partial sums are order-sensitive if a wrong reduction is introduced).
        let weight: Vec<f32> = (0..n * k)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.3125 + 0.1)
            .collect();
        let activation: Vec<f32> = (0..k).map(|j| ((j % 13) as f32 - 6.0) * 0.5).collect();

        // The same closure the flat path passes to `par_chunks_mut`: fill each
        // output row `output_start + offset` with its dot product.
        let gemv_row = |output_start: usize, outputs: &mut [f32]| {
            for (offset, out) in outputs.iter_mut().enumerate() {
                let row = output_start + offset;
                let base = row * k;
                let mut acc = 0.0f32;
                for j in 0..k {
                    acc += weight[base + j] * activation[j];
                }
                *out = acc;
            }
        };

        let mut sharded = vec![0.0f32; n];
        pools.dispatch_output_rows(&mut sharded, k, &gemv_row);

        let mut flat = vec![0.0f32; n];
        gemv_row(0, &mut flat);

        // Bit-for-bit equality (not approximate): row-sharding must not perturb
        // any single row's dot-product accumulation order.
        assert_eq!(sharded.len(), flat.len());
        for (row, (got, want)) in sharded.iter().zip(flat.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "row {row} diverged: sharded={got} flat={want}"
            );
        }
    }

    /// The row split used to place weights (`place_rows`) and to dispatch compute
    /// (`dispatch_output_rows`) must be the *same* partition, or a node's workers
    /// would read another node's weight rows.
    #[test]
    fn place_rows_and_dispatch_share_the_same_partition() {
        let pools = asymmetric_two_node_pools();
        let n = 13usize;
        let lengths = pools.row_lengths(n);
        assert_eq!(lengths, vec![3, 10]);

        let source_rows: Vec<f32> = (0..n).map(|row| row as f32 + 0.25).collect();
        let placed = pools.place_rows(&source_rows, n);
        let mut placement_nodes = Vec::with_capacity(n);
        for (node, &length) in lengths.iter().enumerate() {
            placement_nodes.extend(std::iter::repeat_n(node, length));
        }

        let stamp_source_row = |output_start: usize, outputs: &mut [f32]| {
            let current_thread = std::thread::current();
            let thread_name = current_thread
                .name()
                .expect("dispatch output must run on a node-pinned worker");
            let node = thread_name
                .strip_prefix("onnx-genai-decode-n")
                .and_then(|suffix| suffix.split_once('-'))
                .and_then(|(node, _)| node.parse::<usize>().ok())
                .expect("dispatch output must run on a named node-pinned worker");
            for (offset, output) in outputs.iter_mut().enumerate() {
                let row = output_start + offset;
                assert_eq!(
                    node, placement_nodes[row],
                    "row {row} was dispatched on node {node}, but was placed on node {}",
                    placement_nodes[row]
                );
                *output = placed[row];
            }
        };

        let mut output = vec![0.0f32; n];
        pools.dispatch_output_rows(&mut output, 1, &stamp_source_row);
        for (row, (got, source)) in output.iter().zip(source_rows.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                source.to_bits(),
                "row {row} did not retain its placed source-row identity"
            );
        }
    }

    /// A K large enough that `output_chunk_len` splits a node's segment into
    /// several chunks, exercising the inner `par_chunks_mut` branch of
    /// `dispatch_output_rows` (not just the whole-segment branch).
    fn room_k() -> usize {
        4096
    }
}
