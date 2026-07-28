//! Tests for heterogeneous multi-provider partitioning and execution (#65).
//!
//! The correctness oracle is byte-identical output between a graph executed
//! heterogeneously (nodes split across two providers on two logical devices)
//! and the same graph executed entirely on a single reference provider. Both
//! providers are CPU-backed here so CI can run the placement / partition /
//! transfer-insertion logic without a GPU; the "accelerator" provider reports a
//! distinct, host-accessible logical device ([`DeviceType::Mlx`]) and a
//! restricted capability set, which is exactly the two-logical-CPU-provider
//! shape the design sanctions for CPU testing. A real CUDA end-to-end check is
//! gated separately (GPU-only) and is not part of CI.

use std::sync::Arc;

use onnx_runtime_ep_api::{
    DeviceBuffer, DeviceId, DeviceType, EpConfig, EpId, ExecutionProvider, Fence, Kernel,
    KernelMatch, Result as EpResult,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{DataType, Graph, Node, NodeId, Shape, TensorLayout, static_shape};
use onnx_runtime_loader::WeightStore;

use super::*;
use crate::executor::Executor;

/// A host-backed "accelerator" provider: it executes on the CPU but advertises a
/// distinct, host-accessible logical device and only supports a fixed op set, so
/// a graph must be split across it and the CPU EP.
struct AcceleratorEp {
    inner: CpuExecutionProvider,
    allowed: Vec<&'static str>,
    device: DeviceId,
}

impl AcceleratorEp {
    fn new(allowed: Vec<&'static str>) -> Self {
        let mut inner = CpuExecutionProvider::new();
        inner.initialize(&EpConfig::default()).unwrap();
        Self {
            inner,
            allowed,
            // Mlx is host-accessible, so the CPU execution path stays valid while
            // the device id differs from CPU:0 for transfer planning.
            device: DeviceId::new(DeviceType::Mlx, 0),
        }
    }
}

impl ExecutionProvider for AcceleratorEp {
    fn name(&self) -> &str {
        "accel_ep"
    }
    fn device_type(&self) -> DeviceType {
        DeviceType::Mlx
    }
    fn device_id(&self) -> DeviceId {
        self.device
    }
    fn initialize(&mut self, config: &EpConfig) -> EpResult<()> {
        self.inner.initialize(config)
    }
    fn shutdown(&mut self) -> EpResult<()> {
        self.inner.shutdown()
    }
    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        shapes: &[Shape],
        input_dtypes: &[DataType],
        layouts: &[TensorLayout],
    ) -> KernelMatch {
        if self.allowed.contains(&op.op_type.as_str()) {
            self.inner
                .supports_op(op, opset, shapes, input_dtypes, layouts)
        } else {
            KernelMatch::unsupported("accelerator does not support this op")
        }
    }
    fn get_kernel(
        &self,
        op: &Node,
        shapes: &[Vec<usize>],
        opset: u64,
    ) -> EpResult<Box<dyn Kernel>> {
        self.inner.get_kernel(op, shapes, opset)
    }
    fn allocate(&self, size: usize, alignment: usize) -> EpResult<DeviceBuffer> {
        self.inner.allocate(size, alignment)
    }
    fn deallocate(&self, buffer: DeviceBuffer) -> EpResult<()> {
        self.inner.deallocate(buffer)
    }
    fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> EpResult<()> {
        self.inner.copy(src, dst, size)
    }
    fn copy_async(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> EpResult<Fence> {
        self.inner.copy_async(src, dst, size)
    }
    fn sync(&self) -> EpResult<()> {
        self.inner.sync()
    }
}

fn cpu_slot(ep: u32) -> ProviderPlacement {
    let mut inner = CpuExecutionProvider::new();
    inner.initialize(&EpConfig::default()).unwrap();
    ProviderPlacement {
        ep: EpId(ep),
        provider: Arc::new(inner),
    }
}

fn accel_slot(ep: u32, allowed: Vec<&'static str>) -> ProviderPlacement {
    ProviderPlacement {
        ep: EpId(ep),
        provider: Arc::new(AcceleratorEp::new(allowed)),
    }
}

fn weights() -> Arc<WeightStore> {
    Arc::new(WeightStore::new())
}

/// Build a unary op chain `x -> op0 -> op1 -> ... -> out`.
fn build_chain(ops: &[&str]) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let mut current = graph.create_named_value("x", DataType::Float32, static_shape([4]));
    graph.add_input(current);
    for (i, op) in ops.iter().enumerate() {
        let out = graph.create_named_value(format!("t{i}"), DataType::Float32, static_shape([4]));
        graph.insert_node(Node::new(NodeId(0), *op, vec![Some(current)], vec![out]));
        current = out;
    }
    graph.add_output(current);
    graph
}

/// Reference execution: the whole graph on a single CPU EP.
fn reference(graph: &Graph, inputs: &[(&str, &Tensor)]) -> Vec<Tensor> {
    let ep = cpu_slot(0).provider;
    let mut exec = Executor::build(graph.clone(), weights(), ep).unwrap();
    exec.run(inputs).unwrap()
}

fn assert_byte_identical(a: &[Tensor], b: &[Tensor]) {
    assert_eq!(a.len(), b.len(), "output arity differs");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.as_bytes(),
            y.as_bytes(),
            "output {i} is not byte-identical (f32 bits differ)"
        );
    }
}

#[test]
fn pure_cpu_single_partition() {
    // Accelerator supports nothing; every node lands on the CPU EP as one
    // partition, with no cross-device transfers.
    let graph = build_chain(&["Relu", "Abs", "Neg"]);
    let providers = vec![accel_slot(0, vec![]), cpu_slot(1)];
    let plan = plan(&graph, &providers).unwrap();

    assert_eq!(plan.partitions.len(), 1);
    assert_eq!(plan.partitions[0].ep, EpId(1));
    assert!(plan.transfers.is_empty());

    let x = Tensor::from_f32(&[4], &[-1.0, 2.0, -3.0, 4.0]).unwrap();
    let out = execute(&plan, &graph, &weights(), &providers, &[("x", &x)]).unwrap();
    assert_byte_identical(&out, &reference(&graph, &[("x", &x)]));
}

#[test]
fn fully_accelerator_single_partition() {
    // Accelerator supports every op; one partition on the Mlx device. Only the
    // graph input must be staged host->Mlx.
    let graph = build_chain(&["Relu", "Abs", "Neg"]);
    let providers = vec![accel_slot(0, vec!["Relu", "Abs", "Neg"]), cpu_slot(1)];
    let plan = plan(&graph, &providers).unwrap();

    assert_eq!(plan.partitions.len(), 1);
    assert_eq!(plan.partitions[0].ep, EpId(0));
    assert_eq!(plan.partitions[0].device, DeviceId::new(DeviceType::Mlx, 0));
    // Exactly one transfer: the graph input onto the accelerator device.
    assert_eq!(plan.transfers.len(), 1);
    assert_eq!(plan.transfers[0].to, DeviceId::new(DeviceType::Mlx, 0));

    let x = Tensor::from_f32(&[4], &[-1.0, 2.0, -3.0, 4.0]).unwrap();
    let out = execute(&plan, &graph, &weights(), &providers, &[("x", &x)]).unwrap();
    assert_byte_identical(&out, &reference(&graph, &[("x", &x)]));
}

#[test]
fn mixed_multiple_partition_boundaries() {
    // Relu(accel) -> Abs(cpu) -> Sqrt(accel) -> Neg(cpu): four partitions and
    // three device crossings, plus the host->accel input transfer.
    let graph = build_chain(&["Relu", "Abs", "Sqrt", "Neg"]);
    let providers = vec![accel_slot(0, vec!["Relu", "Sqrt"]), cpu_slot(1)];
    let plan = plan(&graph, &providers).unwrap();

    assert_eq!(plan.partitions.len(), 4, "each op is its own partition");
    let eps: Vec<_> = plan.partitions.iter().map(|p| p.ep).collect();
    assert_eq!(eps, vec![EpId(0), EpId(1), EpId(0), EpId(1)]);

    let x = Tensor::from_f32(&[4], &[1.0, 4.0, 9.0, 16.0]).unwrap();
    let hetero = execute(&plan, &graph, &weights(), &providers, &[("x", &x)]).unwrap();
    let reference = reference(&graph, &[("x", &x)]);
    assert_byte_identical(&hetero, &reference);

    // --- Mutation probe 1: misplace execution order (drop the topological
    // guarantee). Running a consumer partition before its producer leaves a
    // boundary input unmaterialized, so execution must fail. ---------------
    let mut misordered = plan.clone();
    misordered.partitions.reverse();
    let broken = execute(&misordered, &graph, &weights(), &providers, &[("x", &x)]);
    assert!(
        broken.is_err(),
        "reversing partition order must break execution (topological order is load-bearing)"
    );

    // --- Mutation probe 2: drop a boundary transfer by clearing a producer
    // partition's outputs, so the downstream partition never receives it. ---
    let mut dropped = plan.clone();
    dropped.partitions[0].outputs.clear();
    let broken2 = execute(&dropped, &graph, &weights(), &providers, &[("x", &x)]);
    assert!(
        broken2.is_err(),
        "dropping a boundary transfer must break execution"
    );

    // --- Restore: the unmutated plan is byte-identical again. --------------
    let restored = execute(&plan, &graph, &weights(), &providers, &[("x", &x)]).unwrap();
    assert_byte_identical(&restored, &reference);

    // --- Oracle self-check: a single flipped bit is detected. --------------
    let mut corrupted = reference.clone();
    let mut bytes = corrupted[0].as_bytes().to_vec();
    bytes[0] ^= 0x01;
    corrupted[0] = Tensor::from_raw(DataType::Float32, vec![4], &bytes).unwrap();
    assert_ne!(
        hetero[0].as_bytes(),
        corrupted[0].as_bytes(),
        "oracle must detect a one-bit difference"
    );
}

#[test]
fn fan_out_boundary_transfer_is_deduplicated() {
    // Relu(accel) produces `y`; `y` fans out to Abs(cpu) and Neg(cpu), which are
    // two separate CPU partitions on the same device. The Mlx->CPU transfer of
    // `y` must be materialized exactly once, not once per consumer.
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = graph.create_named_value("x", DataType::Float32, static_shape([4]));
    graph.add_input(x);
    let y = graph.create_named_value("y", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(x)], vec![y]));
    let a = graph.create_named_value("a", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Abs", vec![Some(y)], vec![a]));
    let b = graph.create_named_value("b", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Neg", vec![Some(y)], vec![b]));
    graph.add_output(a);
    graph.add_output(b);

    let providers = vec![accel_slot(0, vec!["Relu"]), cpu_slot(1)];
    let plan = plan(&graph, &providers).unwrap();

    // Three partitions: {Relu}@Mlx, {Abs}@cpu, {Neg}@cpu.
    assert_eq!(plan.partitions.len(), 3);
    let cpu_dev = DeviceId::cpu();
    let mlx = DeviceId::new(DeviceType::Mlx, 0);
    // Exactly one transfer of `y` onto the CPU device (fan-out dedup), plus one
    // host->Mlx transfer of `x`.
    let y_to_cpu: Vec<_> = plan
        .transfers
        .iter()
        .filter(|t| t.value == y && t.to == cpu_dev)
        .collect();
    assert_eq!(
        y_to_cpu.len(),
        1,
        "fan-out value must be transferred to CPU exactly once"
    );
    assert_eq!(y_to_cpu[0].from, mlx);
    assert_eq!(plan.transfers.iter().filter(|t| t.value == x).count(), 1);

    let x_val = Tensor::from_f32(&[4], &[-1.0, 2.0, -3.0, 4.0]).unwrap();
    let out = execute(&plan, &graph, &weights(), &providers, &[("x", &x_val)]).unwrap();
    assert_byte_identical(&out, &reference(&graph, &[("x", &x_val)]));
}

#[test]
fn topological_order_preserved() {
    // Every producing partition precedes its consumers in the plan order.
    let graph = build_chain(&["Relu", "Abs", "Sqrt", "Neg"]);
    let providers = vec![accel_slot(0, vec!["Relu", "Sqrt"]), cpu_slot(1)];
    let plan = plan(&graph, &providers).unwrap();

    let mut produced_by: std::collections::HashMap<ValueId, usize> = Default::default();
    for (i, part) in plan.partitions.iter().enumerate() {
        for &node in &part.nodes {
            for &out in &graph.node(node).outputs {
                produced_by.insert(out, i);
            }
        }
    }
    for (j, part) in plan.partitions.iter().enumerate() {
        for &input in &part.inputs {
            if let Some(&i) = produced_by.get(&input) {
                assert!(i < j, "partition {j} consumes a value produced by {i}");
            }
        }
    }
}

#[test]
fn empty_graph_has_no_partitions() {
    // A graph with a single pass-through value (input == output) and no nodes:
    // no partitions, no transfers; execution returns the input unchanged.
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let v = graph.create_named_value("x", DataType::Float32, static_shape([3]));
    graph.add_input(v);
    graph.add_output(v);

    let providers = vec![accel_slot(0, vec![]), cpu_slot(1)];
    let plan = plan(&graph, &providers).unwrap();
    assert!(plan.partitions.is_empty());
    assert!(plan.transfers.is_empty());

    let x = Tensor::from_f32(&[3], &[7.0, 8.0, 9.0]).unwrap();
    let out = execute(&plan, &graph, &weights(), &providers, &[("x", &x)]).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].as_bytes(), x.as_bytes());
}

#[test]
fn node_unsupported_by_all_providers_fails_before_execution() {
    // No provider supports Sqrt (both slots only advertise Relu), so planning
    // must fail with an actionable error before any execution.
    let graph = build_chain(&["Relu", "Sqrt"]);
    let providers = vec![accel_slot(0, vec!["Relu"]), accel_slot(1, vec!["Relu"])];
    let err = plan(&graph, &providers).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Sqrt"),
        "error should name the unsupported op: {msg}"
    );
}

#[test]
fn placement_is_deterministic_across_runs() {
    let graph = build_chain(&["Relu", "Abs", "Sqrt", "Neg"]);
    let providers = vec![accel_slot(0, vec!["Relu", "Sqrt"]), cpu_slot(1)];
    let a = plan(&graph, &providers).unwrap();
    let b = plan(&graph, &providers).unwrap();

    assert_eq!(a.node_placement, b.node_placement);
    assert_eq!(a.partitions, b.partitions);
    assert_eq!(a.transfers, b.transfers);
}
