//! GPU conformance checks for CUDA data-movement and metadata kernels.

use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ep_cuda::{CudaExecutionProvider, subgraph_graph_capturable};
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, TensorData, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

fn bytes<T: Copy>(values: &[T]) -> Vec<u8> {
    // SAFETY: primitive test values are plain old data.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)).to_vec()
    }
}

fn gpu() -> CudaExecutionProvider {
    CudaExecutionProvider::new_default().expect("CUDA runtime must be available for this GPU test")
}

fn upload(ep: &CudaExecutionProvider, bytes: &[u8]) -> DeviceBuffer {
    let buffer = ep.allocate(bytes.len(), 256).unwrap();
    unsafe { ep.runtime().htod(bytes, cuptr(buffer.as_ptr())).unwrap() };
    buffer
}

fn download(ep: &CudaExecutionProvider, buffer: &DeviceBuffer, len: usize) -> Vec<u8> {
    let mut bytes = vec![0; len];
    unsafe {
        ep.runtime()
            .dtoh(&mut bytes, cuptr(buffer.as_ptr()))
            .unwrap()
    };
    bytes
}

fn run_gather(
    ep: &CudaExecutionProvider,
    axis: i64,
    data: &[f32],
    data_shape: &[usize],
    index_bytes: &[u8],
    index_type: DataType,
    index_shape: &[usize],
) -> Vec<f32> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 13);
    let data_value = graph.create_named_value(
        "data",
        DataType::Float32,
        static_shape(data_shape.iter().copied()),
    );
    let indices = graph.create_named_value(
        "indices",
        index_type,
        static_shape(index_shape.iter().copied()),
    );
    let axis = if axis < 0 {
        axis + data_shape.len() as i64
    } else {
        axis
    } as usize;
    let mut output_shape = data_shape[..axis].to_vec();
    output_shape.extend_from_slice(index_shape);
    output_shape.extend_from_slice(&data_shape[axis + 1..]);
    let output = graph.create_named_value(
        "output",
        DataType::Float32,
        static_shape(output_shape.iter().copied()),
    );
    graph.add_input(data_value);
    graph.add_input(indices);
    let mut node = Node::new(
        NodeId(0),
        "Gather",
        vec![Some(data_value), Some(indices)],
        vec![output],
    );
    node.attributes
        .insert("axis".into(), Attribute::Int(axis as i64));
    let node = graph.insert_node(node);
    graph.add_output(output);
    let model = Model::new(&graph);
    let kernel = ep.get_kernel(model.graph.node(node), &[], 13).unwrap();
    assert!(
        !subgraph_graph_capturable(&[kernel.as_ref()]),
        "a subgraph containing Gather must be rejected by the CUDA graph eligibility gate"
    );

    let data_buffer = upload(ep, &bytes(data));
    let index_buffer = upload(ep, index_bytes);
    let output_bytes = output_shape.iter().product::<usize>() * 4;
    let mut output_buffer = ep.allocate(output_bytes, 256).unwrap();
    let data_strides = compute_contiguous_strides(data_shape);
    let index_strides = compute_contiguous_strides(index_shape);
    let output_strides = compute_contiguous_strides(&output_shape);
    let inputs = [
        TensorView::new(
            DevicePtr(data_buffer.as_ptr()),
            DataType::Float32,
            data_shape,
            &data_strides,
            ep.device_id(),
        ),
        TensorView::new(
            DevicePtr(index_buffer.as_ptr()),
            index_type,
            index_shape,
            &index_strides,
            ep.device_id(),
        ),
    ];
    let output = TensorMut::new(
        DevicePtrMut(output_buffer.as_mut_ptr()),
        DataType::Float32,
        &output_shape,
        &output_strides,
        ep.device_id(),
    );
    kernel.execute(&inputs, &mut [output]).unwrap();
    let got = download(ep, &output_buffer, output_bytes)
        .chunks_exact(4)
        .map(|value| f32::from_ne_bytes(value.try_into().unwrap()))
        .collect();
    ep.deallocate(data_buffer).unwrap();
    ep.deallocate(index_buffer).unwrap();
    ep.deallocate(output_buffer).unwrap();
    got
}

#[test]
fn gather_gpu_embedding_i64_negative_index_exact() {
    let ep = gpu();
    let data = [0., 1., 10., 11., 20., 21., 30., 31.];
    let indices = [-1_i64, 1];
    let got = run_gather(
        &ep,
        0,
        &data,
        &[4, 2],
        &bytes(&indices),
        DataType::Int64,
        &[2],
    );
    let expected = vec![30., 31., 10., 11.]; // independent embedding lookup reference
    assert_eq!(got, expected);
    eprintln!("Gather GPU executed: exact match (max abs error 0)");
}

#[test]
fn gather_gpu_axis1_i32_exact() {
    let ep = gpu();
    let data = [0., 1., 2., 10., 11., 12.];
    let indices = [2_i32, 0];
    let got = run_gather(
        &ep,
        1,
        &data,
        &[2, 3],
        &bytes(&indices),
        DataType::Int32,
        &[2],
    );
    let expected = vec![2., 0., 12., 10.]; // columns 2 then 0
    assert_eq!(got, expected);
    eprintln!("Gather GPU executed: exact match (max abs error 0)");
}

#[test]
fn gather_gpu_two_dimensional_indices_exact() {
    let ep = gpu();
    let data = [0., 1., 2., 3., 4., 5., 6., 7.];
    let indices = [3_i64, 0, 2, 1];
    let got = run_gather(
        &ep,
        0,
        &data,
        &[4, 2],
        &bytes(&indices),
        DataType::Int64,
        &[2, 2],
    );
    let expected = vec![6., 7., 0., 1., 4., 5., 2., 3.];
    assert_eq!(got, expected);
    eprintln!("Gather GPU executed: exact match (max abs error 0)");
}

fn run_shape(ep: &CudaExecutionProvider, start: i64, end: Option<i64>) -> Vec<i64> {
    let input_shape = [2, 3, 5, 7];
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 15);
    let input = graph.create_named_value("input", DataType::Float32, static_shape(input_shape));
    let dims = &input_shape[(if start < 0 { start + 4 } else { start }).clamp(0, 4) as usize
        ..(end.unwrap_or(4).let_clamp(4)) as usize];
    let output = graph.create_named_value("output", DataType::Int64, static_shape([dims.len()]));
    graph.add_input(input);
    let mut node = Node::new(NodeId(0), "Shape", vec![Some(input)], vec![output]);
    node.attributes
        .insert("start".into(), Attribute::Int(start));
    if let Some(end) = end {
        node.attributes.insert("end".into(), Attribute::Int(end));
    }
    let node = graph.insert_node(node);
    graph.add_output(output);
    let model = Model::new(&graph);
    let kernel = ep.get_kernel(model.graph.node(node), &[], 15).unwrap();
    let input_buffer = ep.allocate(2 * 3 * 5 * 7 * 4, 256).unwrap();
    let mut output_buffer = ep.allocate(dims.len() * 8, 256).unwrap();
    let input_strides = compute_contiguous_strides(&input_shape);
    let output_shape = [dims.len()];
    let output_strides = compute_contiguous_strides(&output_shape);
    let input_view = TensorView::new(
        DevicePtr(input_buffer.as_ptr()),
        DataType::Float32,
        &input_shape,
        &input_strides,
        ep.device_id(),
    );
    let output_view = TensorMut::new(
        DevicePtrMut(output_buffer.as_mut_ptr()),
        DataType::Int64,
        &output_shape,
        &output_strides,
        ep.device_id(),
    );
    kernel.execute(&[input_view], &mut [output_view]).unwrap();
    let got = download(ep, &output_buffer, dims.len() * 8)
        .chunks_exact(8)
        .map(|value| i64::from_ne_bytes(value.try_into().unwrap()))
        .collect();
    ep.deallocate(input_buffer).unwrap();
    ep.deallocate(output_buffer).unwrap();
    got
}

trait ClampEnd {
    fn let_clamp(self, rank: i64) -> i64;
}
impl ClampEnd for i64 {
    fn let_clamp(self, rank: i64) -> i64 {
        (if self < 0 { self + rank } else { self }).clamp(0, rank)
    }
}

#[test]
fn shape_gpu_full_and_negative_slice_exact() {
    let ep = gpu();
    assert_eq!(run_shape(&ep, 0, None), vec![2, 3, 5, 7]);
    assert_eq!(run_shape(&ep, -3, Some(-1)), vec![3, 5]);
    eprintln!("Shape GPU executed: exact match (max abs error 0)");
}

fn run_constant(
    ep: &CudaExecutionProvider,
    value: Attribute,
    dtype: DataType,
    shape: &[usize],
) -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 13);
    let output = graph.create_named_value("output", dtype, static_shape(shape.iter().copied()));
    let mut node = Node::new(NodeId(0), "Constant", vec![], vec![output]);
    node.attributes.insert("value".into(), value);
    let node = graph.insert_node(node);
    graph.add_output(output);
    let model = Model::new(&graph);
    let kernel = ep.get_kernel(model.graph.node(node), &[], 13).unwrap();
    let output_bytes = dtype.storage_bytes(shape.iter().product());
    let mut output_buffer = ep.allocate(output_bytes, 256).unwrap();
    let strides = compute_contiguous_strides(shape);
    let output = TensorMut::new(
        DevicePtrMut(output_buffer.as_mut_ptr()),
        dtype,
        shape,
        &strides,
        ep.device_id(),
    );
    kernel.execute(&[], &mut [output]).unwrap();
    let got = download(ep, &output_buffer, output_bytes);
    ep.deallocate(output_buffer).unwrap();
    got
}

#[test]
fn constant_gpu_fp32_and_i64_tensor_exact() {
    let ep = gpu();
    let floats = [1.5_f32, -2.25, 3.75, 0.5];
    let float_value = TensorData::from_raw(DataType::Float32, vec![2, 2], bytes(&floats));
    assert_eq!(
        run_constant(
            &ep,
            Attribute::Tensor(float_value),
            DataType::Float32,
            &[2, 2]
        ),
        bytes(&floats)
    );
    let integers = [7_i64, -9, 42];
    let int_value = TensorData::from_raw(DataType::Int64, vec![3], bytes(&integers));
    assert_eq!(
        run_constant(&ep, Attribute::Tensor(int_value), DataType::Int64, &[3]),
        bytes(&integers)
    );
    eprintln!("Constant GPU executed: exact match (max abs error 0)");
}
