//! Falsifiers for the graph-output view materialization path.
//!
//! A `Transpose`/`Reshape`/`Slice` whose result is a graph output emits a
//! strided `ViewOutput` rather than gathered bytes, so the executor is what
//! actually turns it into the tensor the caller receives. That gather used to
//! run a scalar odometer into a `Vec` and then hand the `Vec` to
//! `Tensor::from_raw`, which allocated the bytes a second time and memcpyd
//! between them; it now collapses the geometry, copies the largest contiguous
//! block it can, fans out across rayon above a threshold, and writes straight
//! into the tensor's own allocation.
//!
//! Every one of those four changes can be wrong in a way unit tests on the
//! `Transpose` *kernel* cannot see, because on this path the kernel never runs.
//! So these tests go through the public `InferenceSession` surface and check the
//! values the caller actually gets:
//!
//! * a 4-D attention-layout permute large enough to cross the parallel
//!   threshold, against an independently computed reference — catches a wrong
//!   collapse, a wrong block size, and any row a worker failed to write;
//! * the same permute at a size *below* the threshold — the serial path must
//!   agree bit-for-bit with the parallel one, so a fan-out bug cannot hide
//!   behind a threshold;
//! * a permute that is the identity — must not corrupt anything when the
//!   collapse reduces the whole thing to one memcpy;
//! * a transposed view listed as a graph output *twice* — both tensors must be
//!   correct and independently owned, which is what fails if the gather ever
//!   starts handing out the producer's buffer instead of a copy;
//! * a reversed (negative-stride) view, where "largest contiguous block" is one
//!   element and the odometer is the only thing keeping the answer right.

use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef, static_shape,
};
use onnx_runtime_session::InferenceSession;

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i64_bytes(data: &[i64]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_init(g: &mut Graph, name: &str, dims: &[usize], data: &[f32]) -> ValueId {
    let vid = g.create_named_value(name, DataType::Float32, static_shape(dims.iter().copied()));
    g.set_initializer(
        vid,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            dims.to_vec(),
            f32_bytes(data),
        )),
    );
    vid
}

fn i64_init(g: &mut Graph, name: &str, dims: &[usize], data: &[i64]) -> ValueId {
    let vid = g.create_named_value(name, DataType::Int64, static_shape(dims.iter().copied()));
    g.set_initializer(
        vid,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            dims.to_vec(),
            i64_bytes(data),
        )),
    );
    vid
}

fn op(
    g: &mut Graph,
    op_type: &str,
    inputs: &[ValueId],
    out_dims: &[usize],
    attrs: &[(&str, Attribute)],
) -> ValueId {
    g.opset_imports.entry(String::new()).or_insert(17);
    let out = g.create_value(DataType::Float32, static_shape(out_dims.iter().copied()));
    let mut node = Node::new(
        NodeId(0),
        op_type,
        inputs.iter().map(|&v| Some(v)).collect(),
        vec![out],
    );
    for (k, v) in attrs {
        node.attributes.insert((*k).to_string(), v.clone());
    }
    g.insert_node(node);
    out
}

/// Reference permute: independent of the executor, deliberately the slow
/// obvious loop so it cannot share a bug with the thing under test.
fn reference_permute(src: &[f32], shape: &[usize], perm: &[usize]) -> Vec<f32> {
    let rank = shape.len();
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let mut in_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * shape[i + 1];
    }
    let n: usize = shape.iter().product();
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; rank];
    for _ in 0..n {
        let mut off = 0usize;
        for (d, &i) in idx.iter().enumerate() {
            off += i * in_strides[perm[d]];
        }
        out.push(src[off]);
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

/// Build `Transpose(data, perm)` as the sole graph output and run it.
fn run_transpose(shape: &[usize], perm: &[i64], data: &[f32]) -> Vec<f32> {
    let mut g = Graph::new();
    let src = f32_init(&mut g, "data", shape, data);
    let out_dims: Vec<usize> = perm.iter().map(|&p| shape[p as usize]).collect();
    let t = op(
        &mut g,
        "Transpose",
        &[src],
        &out_dims,
        &[("perm", Attribute::Ints(perm.to_vec()))],
    );
    g.add_output(t);
    let mut session = InferenceSession::from_graph(g).expect("build");
    let out = session.run(&[]).expect("run");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].shape, out_dims);
    out[0].to_vec_f32()
}

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 * 0.25 - 3.0).collect()
}

/// The attention layout permute `[B, S, H, D] -> [B, H, S, D]` at a size that
/// crosses the parallel-gather threshold (2 * 512 * 8 * 64 * 4 B = 2 MiB).
#[test]
fn attention_permute_graph_output_matches_reference_above_parallel_threshold() {
    let shape = [2usize, 512, 8, 64];
    let n: usize = shape.iter().product();
    let data = ramp(n);
    let got = run_transpose(&shape, &[0, 2, 1, 3], &data);
    let want = reference_permute(&data, &shape, &[0, 2, 1, 3]);
    assert_eq!(got.len(), want.len());
    assert!(
        got == want,
        "parallel gather disagrees with reference at {} of {} elements",
        got.iter().zip(&want).filter(|(a, b)| a != b).count(),
        n
    );
}

/// The same permutation below the threshold. If the serial and parallel paths
/// ever diverge, one of these two tests fails while the other passes, which
/// localises the bug to the fan-out rather than the geometry.
#[test]
fn attention_permute_graph_output_matches_reference_below_parallel_threshold() {
    let shape = [1usize, 4, 2, 8];
    let n: usize = shape.iter().product();
    let data = ramp(n);
    let got = run_transpose(&shape, &[0, 2, 1, 3], &data);
    let want = reference_permute(&data, &shape, &[0, 2, 1, 3]);
    assert_eq!(got, want);
}

/// An identity permutation collapses to a single memcpy. A collapse that drops
/// or reorders an axis it should not shows up here as scrambled output even
/// though no bytes are "missing".
#[test]
fn identity_permute_graph_output_is_unchanged() {
    let shape = [3usize, 128, 64];
    let n: usize = shape.iter().product();
    let data = ramp(n);
    let got = run_transpose(&shape, &[0, 1, 2], &data);
    assert_eq!(got, data);
}

/// Trailing size-1 axes are dropped by the collapse before fusion, which is
/// what lets otherwise-separated axes become neighbours. Getting that wrong
/// silently produces a correct-length, wrong-order result.
#[test]
fn permute_with_unit_axes_matches_reference() {
    let shape = [1usize, 6, 1, 32, 1];
    let n: usize = shape.iter().product();
    let data = ramp(n);
    let perm = [3usize, 0, 4, 1, 2];
    let got = run_transpose(&shape, &[3, 0, 4, 1, 2], &data);
    let want = reference_permute(&data, &shape, &perm);
    assert_eq!(got, want);
}

/// The same view listed twice as a graph output. Both tensors must hold the
/// right values and must be separately owned: if materialization ever returned
/// the producer's buffer instead of a copy, the second output would be freed
/// twice or observe the first one's mutations.
#[test]
fn view_listed_twice_as_graph_output_yields_two_correct_tensors() {
    let shape = [2usize, 64, 4, 16];
    let n: usize = shape.iter().product();
    let data = ramp(n);
    let mut g = Graph::new();
    let src = f32_init(&mut g, "data", &shape, &data);
    let t = op(
        &mut g,
        "Transpose",
        &[src],
        &[2, 4, 64, 16],
        &[("perm", Attribute::Ints(vec![0, 2, 1, 3]))],
    );
    g.add_output(t);
    g.add_output(t);
    let mut session = InferenceSession::from_graph(g).expect("build");
    let out = session.run(&[]).expect("run");
    assert_eq!(out.len(), 2);
    let want = reference_permute(&data, &shape, &[0, 2, 1, 3]);
    assert_eq!(out[0].to_vec_f32(), want);
    assert_eq!(out[1].to_vec_f32(), want);
}

/// A reversed slice gives the outer axis a negative stride. The inner axis is
/// still stride-1, so the gather copies whole 32-element rows -- but it must
/// walk them *backwards*, and the collapse must refuse to fuse the two axes
/// because `strides[0] == -32 != strides[1] * shape[1] == 32`. A fast path that
/// ignores stride sign returns the rows in forward order and passes every
/// length check.
#[test]
fn negative_stride_view_graph_output_is_reversed() {
    reversed_rows_roundtrip(64, 32);
}

/// The same reversal at 2 MiB, i.e. above the parallel-gather threshold, so the
/// negative-stride offset arithmetic is exercised by the fan-out and not only by
/// the serial walk. The two tests together mean a sign bug cannot survive by
/// living on only one of the two paths.
#[test]
fn negative_stride_view_graph_output_is_reversed_above_parallel_threshold() {
    reversed_rows_roundtrip(2048, 256);
}

fn reversed_rows_roundtrip(rows: usize, cols: usize) {
    let mut g = Graph::new();
    let data = ramp(rows * cols);
    let src = f32_init(&mut g, "data", &[rows, cols], &data);
    let starts = i64_init(&mut g, "starts", &[1], &[(rows - 1) as i64]);
    let ends = i64_init(&mut g, "ends", &[1], &[i64::MIN]);
    let axes = i64_init(&mut g, "axes", &[1], &[0]);
    let steps = i64_init(&mut g, "steps", &[1], &[-1]);
    let sliced = op(
        &mut g,
        "Slice",
        &[src, starts, ends, axes, steps],
        &[rows, cols],
        &[],
    );
    g.add_output(sliced);
    let mut session = InferenceSession::from_graph(g).expect("build");
    let out = session.run(&[]).expect("run");
    let got = out[0].to_vec_f32();
    let want: Vec<f32> = (0..rows)
        .rev()
        .flat_map(|r| data[r * cols..(r + 1) * cols].to_vec())
        .collect();
    assert_eq!(got, want);
}

/// The gather treats element size as a bare multiplier, so every dtype whose
/// storage is not 4 bytes exercises block arithmetic the f32 tests never reach:
/// a wrong `esize` still produces a correctly *shaped* tensor, just with the
/// bytes sheared. `Int64` (8 B) and `Float16` (2 B) bracket f32 on both sides.
#[test]
fn non_f32_dtypes_permute_correctly_through_the_gather() {
    for &(dtype, esize) in &[(DataType::Int64, 8usize), (DataType::Float16, 2usize)] {
        let shape = [2usize, 130, 3, 17];
        let n: usize = shape.iter().product();
        // Distinct per element, and distinct in every byte lane, so a shear by
        // any amount changes the value.
        let bytes: Vec<u8> = (0..n)
            .flat_map(|i| {
                let v = (i as u64).wrapping_mul(0x0101_0101_0101_0101).to_le_bytes();
                v[..esize].to_vec()
            })
            .collect();

        let mut g = Graph::new();
        let vid = g.create_named_value("data", dtype, static_shape(shape.iter().copied()));
        g.set_initializer(
            vid,
            WeightRef::Inline(TensorData::from_raw(dtype, shape.to_vec(), bytes.clone())),
        );
        g.opset_imports.entry(String::new()).or_insert(17);
        let out_dims = vec![2usize, 3, 130, 17];
        let out = g.create_value(dtype, static_shape(out_dims.iter().copied()));
        let mut node = Node::new(NodeId(0), "Transpose", vec![Some(vid)], vec![out]);
        node.attributes
            .insert("perm".to_string(), Attribute::Ints(vec![0, 2, 1, 3]));
        g.insert_node(node);
        g.add_output(out);

        let mut session = InferenceSession::from_graph(g).expect("build");
        let res = session.run(&[]).expect("run");
        let got = res[0].as_bytes().to_vec();
        assert_eq!(res[0].shape, out_dims, "{dtype:?}");
        assert_eq!(got.len(), n * esize, "{dtype:?}");

        // Reference: permute element indices, then splice the source bytes.
        let mut want = Vec::with_capacity(n * esize);
        for b in 0..shape[0] {
            for h in 0..shape[2] {
                for s in 0..shape[1] {
                    for d in 0..shape[3] {
                        let src = ((b * shape[1] + s) * shape[2] + h) * shape[3] + d;
                        want.extend_from_slice(&bytes[src * esize..(src + 1) * esize]);
                    }
                }
            }
        }
        assert!(got == want, "{dtype:?} gather sheared the bytes");
    }
}
