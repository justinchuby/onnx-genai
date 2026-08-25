//! ORT graph ABI bridge for legacy plugin EPs (§3.4, §4.5).
//!
//! Projects our IR through the small ONNX Runtime C graph API subset that a
//! plugin EP uses during capability discovery. The bridge deliberately owns the
//! raw FFI boundary here so the rest of the native runtime can keep using safe
//! Rust graph data.

use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString, c_char};
use std::path::Path;
use std::ptr;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use onnx_genai_ort_sys as ort;
use onnx_runtime_ir::{Graph, GraphView, NodeId, NodeIndex, ValueId, ValueIndex};

use crate::error::{EpError, Result};
use crate::kernel::{Kernel, KernelMatch};
use crate::provider::{DeviceBuffer, EpConfig, EpId, ExecutionProvider, Fence};
mod ffi_helpers;
mod host;
mod runtime;
mod weights;

use ffi_helpers::{MAX_PLUGIN_EP_DEVICES, ep_device_hardware, ep_device_metadata};
use host::{HostGraph, HostNode, HostSupportInfo, check_status, ort_api_base, release_status};
pub use runtime::PluginCompiledKernel;
use runtime::{PluginKernelShared, PluginRuntime};

/// A read-only projection of a [`Graph`] exposed through the ORT C graph API.
pub struct OrtGraphView<'view, 'graph> {
    view: &'view GraphView<'graph>,
}

/// Disjoint-set forest used to fuse supported nodes into convex partitions.
///
/// Union is deterministic (union by size, lowest index wins ties) so capability
/// claims stay stable across runs, and `find` avoids path compression so the
/// structure can be shared behind `&self` and cheaply cloned for trial merges.
#[derive(Clone)]
struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            size: vec![1; len],
        }
    }

    fn find(&self, mut node: usize) -> usize {
        while self.parent[node] != node {
            node = self.parent[node];
        }
        node
    }

    fn union(&mut self, a: usize, b: usize) {
        let (root_a, root_b) = (self.find(a), self.find(b));
        if root_a == root_b {
            return;
        }
        let (root, child) = if self.size[root_a] > self.size[root_b]
            || (self.size[root_a] == self.size[root_b] && root_a < root_b)
        {
            (root_a, root_b)
        } else {
            (root_b, root_a)
        };
        self.parent[child] = root;
        self.size[root] += self.size[child];
    }
}

/// An EP's claim over a subgraph it wants to compile and run.
#[derive(Clone, Debug)]
pub struct SubgraphClaim {
    /// The runtime-local EP identifier that produced the claim.
    pub ep_id: EpId,
    /// Claimed nodes, in the plugin's requested fused group.
    pub node_ids: Vec<NodeId>,
    /// Boundary values entering the claimed subgraph from outside it.
    pub input_values: Vec<ValueId>,
    /// Boundary values leaving the claimed subgraph to the rest of the graph.
    pub output_values: Vec<ValueId>,
    /// Optional plugin-specific metadata for a future compiled node.
    pub meta_def: Option<String>,
}

/// A compiled ORT plugin-EP subgraph that can be dispatched by the native executor.
///
/// ORT plugin EPs return opaque `OrtNodeComputeInfo` callbacks instead of native
/// Rust [`Kernel`]s. This wrapper owns the plugin library, EP instance, compute
/// infos, and per-subgraph compute states so the executor can treat each fused
/// subgraph as a normal kernel while preserving the ORT callback lifetimes.
pub struct PluginExecutionPlan {
    // Kept as a lifetime anchor so plugin callbacks remain loaded until all kernels are dropped.
    #[allow(dead_code)]
    inner: Arc<PluginRuntime>,
    kernels: Vec<Arc<PluginKernelShared>>,
}

/// An ORT plugin EP loaded through its stable C ABI.
///
/// Plugin EPs claim and compile *graphs*, rather than individual Rust
/// [`Node`]s. Consequently, [`ExecutionProvider::supports_op`] intentionally
/// declines individual-node dispatch; use [`PluginExecutionPlan::compile`] to
/// query its graph-level capabilities and obtain runnable kernels.
pub struct LegacyOrtEp {
    inner: PluginRuntime,
    name: String,
}

impl LegacyOrtEp {
    /// Load and instantiate an ORT plugin EP from `path`.
    ///
    /// `legacy.registration_name`, when present in `config`, is passed to the
    /// plugin's `CreateEpFactories` entry point. This is the only loader option
    /// interpreted by the ABI bridge; all provider-specific options remain the
    /// plugin's responsibility when it compiles a graph.
    pub fn load(path: impl AsRef<Path>, config: &EpConfig) -> Result<Self> {
        let path = path.as_ref();
        let registration_name = config
            .options
            .get("legacy.registration_name")
            .map(|name| {
                CString::new(name.as_str()).map_err(|_| EpError::EpLoadFailed {
                    path: path.to_path_buf(),
                    reason: "legacy.registration_name contains an interior NUL byte".into(),
                })
            })
            .transpose()?;
        Self::load_with_registration_name(path, registration_name.as_deref())
    }

    fn load_with_registration_name(path: &Path, registration_name: Option<&CStr>) -> Result<Self> {
        let inner = PluginRuntime::load(path, registration_name)?;
        let name = unsafe {
            (*inner.factory).GetName.and_then(|get_name| {
                let name = get_name(inner.factory);
                (!name.is_null()).then(|| CStr::from_ptr(name).to_string_lossy().into_owned())
            })
        }
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .map_or_else(|| "legacy_ort_ep".into(), |stem| format!("legacy_{stem}"))
        });
        Ok(Self { inner, name })
    }
}

impl ExecutionProvider for LegacyOrtEp {
    fn consume_route_residency_at_boundary(&self) -> Result<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn device_type(&self) -> onnx_runtime_ir::DeviceType {
        // The plugin-EP ABI does not expose a portable device-type query.
        onnx_runtime_ir::DeviceType::Custom(0)
    }

    fn device_id(&self) -> onnx_runtime_ir::DeviceId {
        onnx_runtime_ir::DeviceId::new(self.device_type(), 0)
    }

    fn initialize(&mut self, _config: &EpConfig) -> Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }

    fn supports_op(
        &self,
        _op: &onnx_runtime_ir::Node,
        _opset: u64,
        _shapes: &[onnx_runtime_ir::Shape],
        _input_dtypes: &[onnx_runtime_ir::DataType],
        _layouts: &[onnx_runtime_ir::TensorLayout],
    ) -> KernelMatch {
        KernelMatch::unsupported(
            "legacy ORT plugin EPs select graph subgraphs through PluginExecutionPlan, not individual nodes",
        )
    }

    fn get_kernel(
        &self,
        _op: &onnx_runtime_ir::Node,
        _shapes: &[Vec<usize>],
        _opset: u64,
    ) -> Result<Box<dyn Kernel>> {
        Err(EpError::KernelFailed(format!(
            "{} requires graph-level PluginExecutionPlan compilation before dispatch",
            self.name
        )))
    }

    fn allocate(&self, _size: usize, _alignment: usize) -> Result<DeviceBuffer> {
        Err(EpError::KernelFailed(format!(
            "{} does not expose allocation through the legacy plugin EP ABI",
            self.name
        )))
    }

    fn deallocate(&self, _buffer: DeviceBuffer) -> Result<()> {
        Err(EpError::KernelFailed(format!(
            "{} does not expose deallocation through the legacy plugin EP ABI",
            self.name
        )))
    }

    fn copy(&self, _src: &DeviceBuffer, _dst: &mut DeviceBuffer, _size: usize) -> Result<()> {
        Err(EpError::KernelFailed(format!(
            "{} does not expose copies through the legacy plugin EP ABI",
            self.name
        )))
    }

    fn copy_async(
        &self,
        _src: &DeviceBuffer,
        _dst: &mut DeviceBuffer,
        _size: usize,
    ) -> Result<Fence> {
        Err(EpError::KernelFailed(format!(
            "{} does not expose asynchronous copies through the legacy plugin EP ABI",
            self.name
        )))
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }
}

impl PluginExecutionPlan {
    /// Load `library_path`, run capability discovery, compile the selected fused
    /// groups, and return the claims plus runnable kernels in the same order.
    pub fn compile(
        graph: &Graph,
        library_path: impl AsRef<Path>,
        registration_name: Option<&CStr>,
    ) -> Result<(Vec<SubgraphClaim>, Self)> {
        Self::compile_with_device_label(graph, library_path, registration_name, "plugin")
    }

    /// Like [`Self::compile`], but uses `device_label` in fused-dispatch trace spans.
    pub fn compile_with_device_label(
        graph: &Graph,
        library_path: impl AsRef<Path>,
        registration_name: Option<&CStr>,
        device_label: impl Into<String>,
    ) -> Result<(Vec<SubgraphClaim>, Self)> {
        let library_path = library_path.as_ref();
        let device_label: Arc<str> = Arc::from(device_label.into());
        let mut support = HostSupportInfo::default();
        let legacy = LegacyOrtEp::load_with_registration_name(library_path, registration_name)?;
        let mut runtime = legacy.inner;
        let host = HostGraph::new(graph).map_err(|reason| EpError::EpLoadFailed {
            path: library_path.to_path_buf(),
            reason,
        })?;

        // SAFETY: The EP pointer and graph projection are valid for the duration
        // of the call. The plugin fills only the support-info object we pass.
        let status = unsafe {
            let get_capability = (*runtime.ep).GetCapability.ok_or_else(|| EpError::EpLoadFailed {
                path: library_path.to_path_buf(),
                reason: "OrtEp.GetCapability is null; fix by using a plugin EP that implements capability discovery".into(),
            })?;
            get_capability(
                runtime.ep,
                host.as_ort_graph(),
                &mut support as *mut HostSupportInfo as *mut ort::OrtEpGraphSupportInfo,
            )
        };
        check_status(library_path, "OrtEp.GetCapability", status)?;
        let claims = support.into_claims(graph, EpId(0));
        if claims.is_empty() {
            return Err(EpError::EpLoadFailed {
                path: library_path.to_path_buf(),
                reason:
                    "plugin loaded but GetCapability claimed no nodes; fix by selecting a model/operator set the plugin supports or using a different provider"
                        .into(),
            });
        }

        let compile = unsafe { (*runtime.ep).Compile }.ok_or_else(|| EpError::EpLoadFailed {
            path: library_path.to_path_buf(),
            reason:
                "OrtEp.Compile is null; fix by using a plugin EP that returns OrtNodeComputeInfo for claimed subgraphs"
                    .into(),
        })?;

        let mut subgraphs = Vec::with_capacity(claims.len());
        let mut fused_nodes = Vec::with_capacity(claims.len());
        for (index, claim) in claims.iter().enumerate() {
            subgraphs.push(HostGraph::new_for_claim(graph, claim).map_err(|reason| {
                EpError::EpLoadFailed {
                    path: library_path.to_path_buf(),
                    reason,
                }
            })?);
            fused_nodes.push(HostNode::fused(graph, claim, index).map_err(|reason| {
                EpError::EpLoadFailed {
                    path: library_path.to_path_buf(),
                    reason,
                }
            })?);
        }
        let mut graph_ptrs: Vec<*const ort::OrtGraph> =
            subgraphs.iter().map(HostGraph::as_ort_graph).collect();
        let mut fused_ptrs: Vec<*const ort::OrtNode> = fused_nodes
            .iter()
            .map(|node| (&**node as *const HostNode).cast::<ort::OrtNode>())
            .collect();
        let mut infos: Vec<*mut ort::OrtNodeComputeInfo> = vec![ptr::null_mut(); claims.len()];
        let mut ep_context_nodes: Vec<*mut ort::OrtNode> = vec![ptr::null_mut(); claims.len()];

        // SAFETY: All arrays contain `claims.len()` entries and point to host
        // graph/fused-node projections that remain alive until Compile returns.
        let status = unsafe {
            compile(
                runtime.ep,
                graph_ptrs.as_mut_ptr(),
                fused_ptrs.as_mut_ptr(),
                claims.len(),
                infos.as_mut_ptr(),
                ep_context_nodes.as_mut_ptr(),
            )
        };
        check_status(library_path, "OrtEp.Compile", status)?;
        if infos.iter().any(|info| info.is_null()) {
            return Err(EpError::EpLoadFailed {
                path: library_path.to_path_buf(),
                reason:
                    "OrtEp.Compile returned a null OrtNodeComputeInfo; fix the plugin compile implementation"
                        .into(),
            });
        }
        runtime.compute_infos = infos.clone();
        let inner = Arc::new(runtime);
        let mut kernels = Vec::with_capacity(infos.len());
        for (index, info) in infos.into_iter().enumerate() {
            let create_state = unsafe { (*info).CreateState }.ok_or_else(|| EpError::EpLoadFailed {
                path: library_path.to_path_buf(),
                reason: format!(
                    "OrtNodeComputeInfo[{index}].CreateState is null; fix the plugin compute-info callbacks"
                ),
            })?;
            let compute = unsafe { (*info).Compute }.ok_or_else(|| EpError::EpLoadFailed {
                path: library_path.to_path_buf(),
                reason: format!(
                    "OrtNodeComputeInfo[{index}].Compute is null; fix the plugin compute-info callbacks"
                ),
            })?;
            kernels.push(Arc::new(PluginKernelShared {
                runtime: Arc::clone(&inner),
                info,
                create_state,
                compute,
                release_state: unsafe { (*info).ReleaseState },
                states: Mutex::new(HashMap::new()),
                index,
                calls: AtomicU64::new(0),
                device_label: Arc::clone(&device_label),
            }));
        }
        Ok((claims, Self { inner, kernels }))
    }

    /// Return a runnable kernel for the claim at `index`.
    pub fn kernel(&self, index: usize) -> Option<PluginCompiledKernel> {
        self.kernels.get(index).map(|shared| PluginCompiledKernel {
            shared: Arc::clone(shared),
        })
    }

    /// Number of fused subgraph compute calls observed through this plan.
    pub fn compute_calls(&self) -> u64 {
        self.kernels
            .iter()
            .map(|kernel| kernel.calls.load(Ordering::Relaxed))
            .sum()
    }
}

impl<'view, 'graph> OrtGraphView<'view, 'graph> {
    /// Wrap an immutable cached graph lens for ABI projection.
    pub fn new(view: &'view GraphView<'graph>) -> Self {
        Self { view }
    }

    /// Ask a native Rust EP which subgraphs it can handle.
    ///
    /// Supported nodes are fused into deterministic, **convex** partitions.
    /// Adjacent supported nodes are co-partitioned only when the merge keeps
    /// every claim convex, so no directed path leaves a claim, passes through an
    /// excluded node, and re-enters it. Unsupported nodes and merges that would
    /// break convexity split the region into several claims, preserving ORT's
    /// `GetCapability` schedulability contract rather than flattening by EP.
    pub fn query_capabilities(&self, ep: &dyn ExecutionProvider) -> Vec<SubgraphClaim> {
        self.query_capabilities_filtered(ep, |_| true)
    }

    /// Like [`Self::query_capabilities`], but additionally requires every claimed
    /// node to satisfy `node_admissible`.
    ///
    /// The predicate is applied *before* convex partitioning, so a rejected node
    /// is excluded from the supported set and ORT partitions *around* it — the
    /// rest of the graph can still be claimed. This is the correct granularity
    /// for graph-level admissibility constraints that a per-op `supports_op`
    /// cannot express (e.g. a plugin whose compile path cannot route a node's
    /// weight-initializer inputs): declining the node here keeps it out of every
    /// convex claim rather than dropping a whole partition after the fact.
    pub fn query_capabilities_filtered<F>(
        &self,
        ep: &dyn ExecutionProvider,
        node_admissible: F,
    ) -> Vec<SubgraphClaim>
    where
        F: Fn(NodeIndex) -> bool,
    {
        let supported: Vec<_> = self
            .view
            .nodes()
            .filter(|&node| {
                if !node_admissible(node) {
                    return false;
                }
                let opset = self
                    .view
                    .graph()
                    .effective_opset(self.view.node(node))
                    .unwrap_or(0);
                ep.supports_node(self.view, node, opset).is_supported()
            })
            .collect();
        if supported.is_empty() {
            return Vec::new();
        }

        let node_count = self.view.nodes().len();
        let mut membership = vec![false; node_count];
        for node in supported {
            membership[node.as_usize()] = true;
        }

        // Fuse adjacent supported nodes, but only when the merge keeps the
        // partition convex. Two supported nodes may share a claim only if
        // merging their partitions introduces no directed path that leaves the
        // claim, passes through an excluded (e.g. unsupported) node, and
        // re-enters it. Such a non-convex partition would form a dependency
        // cycle with the excluded node and be unschedulable.
        let mut partitions = UnionFind::new(node_count);
        for node in self.view.nodes() {
            if !membership[node.as_usize()] {
                continue;
            }
            for &output in self.view.node_outputs(node) {
                for consumer in self.view.consumers(output) {
                    let neighbor = consumer.node;
                    if !membership[neighbor.as_usize()]
                        || partitions.find(node.as_usize()) == partitions.find(neighbor.as_usize())
                    {
                        continue;
                    }
                    if self.merge_preserves_convexity(&partitions, node, neighbor) {
                        partitions.union(node.as_usize(), neighbor.as_usize());
                    }
                }
            }
        }

        // Emit one claim per partition, ordered by first (topological) node and
        // with deterministic node membership.
        let mut order = Vec::new();
        let mut components: HashMap<usize, Vec<NodeIndex>> = HashMap::new();
        for node in self.view.nodes() {
            if !membership[node.as_usize()] {
                continue;
            }
            let root = partitions.find(node.as_usize());
            let component = components.entry(root).or_default();
            if component.is_empty() {
                order.push(root);
            }
            component.push(node);
        }
        order
            .into_iter()
            .map(|root| {
                let mut component = components.remove(&root).expect("root has a component");
                component.sort_unstable();
                self.claim_for_component(&component)
            })
            .collect()
    }

    /// Whether fusing the partitions containing `left` and `right` keeps every
    /// claim convex, i.e. the condensed graph (partition representatives as
    /// vertices, cross-partition value dependencies as edges) stays acyclic. A
    /// cycle would mean a directed path leaves the fused claim and returns to
    /// it, which is unschedulable.
    fn merge_preserves_convexity(
        &self,
        partitions: &UnionFind,
        left: NodeIndex,
        right: NodeIndex,
    ) -> bool {
        let mut trial = partitions.clone();
        trial.union(left.as_usize(), right.as_usize());
        self.condensed_graph_is_acyclic(&trial)
    }

    /// Kahn's algorithm over the condensed graph whose vertices are the current
    /// partition representatives and whose edges follow the original graph's
    /// value dependencies across partitions.
    fn condensed_graph_is_acyclic(&self, partitions: &UnionFind) -> bool {
        let mut edges: HashSet<(usize, usize)> = HashSet::new();
        let mut successors: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut in_degree: HashMap<usize, usize> = HashMap::new();

        for node in self.view.nodes() {
            in_degree
                .entry(partitions.find(node.as_usize()))
                .or_insert(0);
        }
        for node in self.view.nodes() {
            let from = partitions.find(node.as_usize());
            for &output in self.view.node_outputs(node) {
                for consumer in self.view.consumers(output) {
                    let to = partitions.find(consumer.node.as_usize());
                    if from != to && edges.insert((from, to)) {
                        successors.entry(from).or_default().push(to);
                        *in_degree.entry(to).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut ready: Vec<usize> = in_degree
            .iter()
            .filter(|&(_, &degree)| degree == 0)
            .map(|(&vertex, _)| vertex)
            .collect();
        let mut resolved = 0usize;
        while let Some(vertex) = ready.pop() {
            resolved += 1;
            if let Some(children) = successors.get(&vertex) {
                for &child in children {
                    let degree = in_degree.get_mut(&child).expect("edge target is a vertex");
                    *degree -= 1;
                    if *degree == 0 {
                        ready.push(child);
                    }
                }
            }
        }
        resolved == in_degree.len()
    }

    fn claim_for_component(&self, nodes: &[NodeIndex]) -> SubgraphClaim {
        let mut membership = vec![false; self.view.nodes().len()];
        for &node in nodes {
            membership[node.as_usize()] = true;
        }
        let mut inputs = Vec::<ValueIndex>::new();
        let mut outputs = Vec::<ValueIndex>::new();
        for &node in nodes {
            for &input in self.view.node_inputs(node).iter().flatten() {
                if self
                    .view
                    .producer(input)
                    .is_none_or(|producer| !membership[producer.as_usize()])
                    && !inputs.contains(&input)
                {
                    inputs.push(input);
                }
            }
            for &output in self.view.node_outputs(node) {
                let crosses_partition = self
                    .view
                    .consumers(output)
                    .iter()
                    .any(|use_| !membership[use_.node.as_usize()]);
                if (crosses_partition || self.view.value(output).is_graph_output)
                    && !outputs.contains(&output)
                {
                    outputs.push(output);
                }
            }
        }
        SubgraphClaim {
            ep_id: EpId(0),
            node_ids: nodes.iter().map(|&node| self.view.node_id(node)).collect(),
            input_values: inputs
                .into_iter()
                .map(|value| self.view.value_id(value))
                .collect(),
            output_values: outputs
                .into_iter()
                .map(|value| self.view.value_id(value))
                .collect(),
            meta_def: None,
        }
    }

    /// Load an ORT plugin-EP dynamic library and run its `GetCapability` method.
    ///
    /// The plugin sees this graph as an `OrtGraph` and reports fused node groups
    /// via `OrtEpGraphSupportInfo_AddNodesToFuse`. This does not compile or run
    /// the groups; it is the Stage-1 capability discovery boundary.
    pub fn query_plugin_capabilities(
        &self,
        library_path: impl AsRef<Path>,
        registration_name: Option<&CStr>,
    ) -> Result<Vec<SubgraphClaim>> {
        let library_path = library_path.as_ref();
        let graph = self.view.graph();
        let host = HostGraph::new(graph).map_err(|reason| EpError::EpLoadFailed {
            path: library_path.to_path_buf(),
            reason,
        })?;
        let mut support = HostSupportInfo::default();

        // SAFETY: Loading a user-selected plugin library is the required ORT
        // plugin mechanism. We keep the Library alive until factory/EP release
        // completes, and resolve only the documented C ABI symbols.
        let lib = unsafe { libloading::Library::new(library_path) }.map_err(|err| {
            EpError::EpLoadFailed {
                path: library_path.to_path_buf(),
                reason: format!(
                    "failed to open plugin dynamic library ({err}); fix by building the plugin dylib and passing the correct absolute path"
                ),
            }
        })?;

        type CreateEpFactories = unsafe extern "C" fn(
            *const c_char,
            *const ort::OrtApiBase,
            *const ort::OrtLogger,
            *mut *mut ort::OrtEpFactory,
            usize,
            *mut usize,
        ) -> *mut ort::OrtStatus;
        type ReleaseEpFactory = unsafe extern "C" fn(*mut ort::OrtEpFactory) -> *mut ort::OrtStatus;

        // SAFETY: Symbol types match ONNX Runtime's plugin EP C ABI. The symbol
        // is called before `lib` is dropped, so the function pointer remains valid.
        let create = unsafe { lib.get::<CreateEpFactories>(b"CreateEpFactories") }.map_err(|err| {
            EpError::EpLoadFailed {
                path: library_path.to_path_buf(),
                reason: format!(
                    "CreateEpFactories symbol was not found ({err}); fix by using an ONNX Runtime plugin-EP library built against the plugin EP C ABI"
                ),
            }
        })?;
        // SAFETY: Optional release symbol from the same plugin ABI. Absence is
        // tolerated because some hosts keep factories alive for process lifetime.
        let release_factory = unsafe { lib.get::<ReleaseEpFactory>(b"ReleaseEpFactory") }.ok();

        let mut factories: [*mut ort::OrtEpFactory; 1] = [ptr::null_mut()];
        let mut num_factories = 0usize;
        let name_ptr = registration_name.map_or(ptr::null(), CStr::as_ptr);
        // SAFETY: All out-pointers reference live stack storage, and the API base
        // points at process-lifetime vtables built below.
        let status = unsafe {
            create(
                name_ptr,
                ort_api_base(),
                ptr::null(),
                factories.as_mut_ptr(),
                factories.len(),
                &mut num_factories,
            )
        };
        check_status(library_path, "CreateEpFactories", status)?;
        if num_factories == 0 || factories[0].is_null() {
            return Err(EpError::EpLoadFailed {
                path: library_path.to_path_buf(),
                reason: "CreateEpFactories returned no factories; fix by checking that the plugin supports this platform and ORT API version".into(),
            });
        }

        let factory = factories[0];

        // Ask the factory which devices it supports, and hand exactly those
        // back when creating the provider.
        //
        // An earlier version passed `num_devices = 1` with both arrays null,
        // which happened to work because the one plugin tested does not
        // dereference them. The contract says these are the devices the
        // provider "was selected to use", so any plugin is entitled to read
        // them, and a count that describes no array is a crash waiting for a
        // different plugin.
        let mut ep_devices: [*mut ort::OrtEpDevice; MAX_PLUGIN_EP_DEVICES] =
            [ptr::null_mut(); MAX_PLUGIN_EP_DEVICES];
        let mut num_ep_devices = 0usize;
        // SAFETY: the factory came from the plugin; the out-array and count
        // reference live stack storage sized by `max_ep_devices`. The input
        // device list is empty, which the API permits: a plugin that owns its
        // own hardware (rather than describing one ORT enumerated) creates its
        // devices here.
        let status = unsafe {
            let get_devices =
                (*factory)
                    .GetSupportedDevices
                    .ok_or_else(|| EpError::EpLoadFailed {
                        path: library_path.to_path_buf(),
                        reason: "OrtEpFactory.GetSupportedDevices is null; fix by using a \
                                 complete plugin EP factory"
                            .into(),
                    })?;
            get_devices(
                factory,
                ptr::null(),
                0,
                ep_devices.as_mut_ptr(),
                ep_devices.len(),
                &mut num_ep_devices,
            )
        };
        check_status(library_path, "OrtEpFactory.GetSupportedDevices", status)?;
        if num_ep_devices == 0 {
            return Err(EpError::EpLoadFailed {
                path: library_path.to_path_buf(),
                reason: "the plugin reported no supported devices; fix by checking that its \
                         hardware is present and its driver is installed"
                    .into(),
            });
        }
        let num_ep_devices = num_ep_devices.min(ep_devices.len());

        // Unpack each OrtEpDevice into the hardware device and metadata that
        // CreateEp wants, keeping the two arrays index-aligned.
        let mut hardware: Vec<*const ort::OrtHardwareDevice> = Vec::with_capacity(num_ep_devices);
        let mut metadata: Vec<*const ort::OrtKeyValuePairs> = Vec::with_capacity(num_ep_devices);
        for device in ep_devices.iter().take(num_ep_devices) {
            // SAFETY: each entry was written by the factory above and is
            // non-null for indices below `num_ep_devices`. The accessors are
            // pure reads on the plugin-owned device.
            unsafe {
                hardware.push(ep_device_hardware(*device));
                metadata.push(ep_device_metadata(*device));
            }
        }

        let mut ep: *mut ort::OrtEp = ptr::null_mut();
        // SAFETY: the factory came from the plugin, and the two arrays are
        // index-aligned and exactly `num_ep_devices` long, matching the count.
        let status = unsafe {
            let create_ep = (*factory).CreateEp.ok_or_else(|| EpError::EpLoadFailed {
                path: library_path.to_path_buf(),
                reason: "OrtEpFactory.CreateEp is null; fix by using a complete plugin EP factory"
                    .into(),
            })?;
            create_ep(
                factory,
                hardware.as_ptr(),
                metadata.as_ptr(),
                num_ep_devices,
                ptr::null(),
                ptr::null(),
                &mut ep,
            )
        };
        check_status(library_path, "OrtEpFactory.CreateEp", status)?;
        if ep.is_null() {
            return Err(EpError::EpLoadFailed {
                path: library_path.to_path_buf(),
                reason: "OrtEpFactory.CreateEp returned a null EP; fix by checking plugin device requirements and options".into(),
            });
        }

        // SAFETY: The EP pointer is valid until released by the factory. The graph
        // and support objects outlive the call and are never mutated concurrently.
        let status = unsafe {
            let get_capability = (*ep).GetCapability.ok_or_else(|| EpError::EpLoadFailed {
                path: library_path.to_path_buf(),
                reason: "OrtEp.GetCapability is null; fix by using a plugin EP that implements capability discovery".into(),
            })?;
            get_capability(
                ep,
                host.as_ort_graph(),
                &mut support as *mut HostSupportInfo as *mut ort::OrtEpGraphSupportInfo,
            )
        };
        let result = check_status(library_path, "OrtEp.GetCapability", status)
            .map(|()| support.into_claims(graph, EpId(0)));

        // SAFETY: Release callbacks belong to the factory/EP returned above and
        // are invoked before the dynamic library is unloaded.
        unsafe {
            if let Some(release_ep) = (*factory).ReleaseEp {
                release_ep(factory, ep);
            }
            if let Some(release_factory) = release_factory {
                let st = release_factory(factory);
                if !st.is_null() {
                    release_status(st);
                }
            }
        }

        result
    }
}
