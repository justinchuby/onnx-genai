//! `OutboundGraphReader` — reads ORT's `OrtGraph*` through the ORT C API.
//!
//! This is the inverse of `abi/host.rs` which projects our IR as a fake
//! `OrtGraph*`. Here we read a REAL `OrtGraph*` from upstream ORT and build
//! a nxrt `Graph` for capability discovery.

use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::ptr;

use onnx_genai_ort_sys as ort;
use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, Shape, ValueId};

/// A reader that extracts nxrt IR from an ORT `OrtGraph*`.
pub struct OutboundGraphReader {
    graph: Graph,
    /// Maps ORT node index (position in the `Graph_GetNodes` array) → our NodeId.
    ort_index_to_node_id: Vec<NodeId>,
    /// The original ORT node pointers, kept for passing back to
    /// `EpGraphSupportInfo_AddNodesToFuse`.
    ort_node_ptrs: Vec<*const ort::OrtNode>,
    /// Owned copies of small int64 initializer tensors, keyed by value name.
    /// Used to resolve opset-13 Unsqueeze/Squeeze axes from constant inputs.
    initializer_int64: HashMap<String, Vec<i64>>,
    /// Out-of-band set of ValueIds that represent absent optional output slots.
    /// This replaces the previous in-band string sentinel (`__absent_output_*`)
    /// which was forgeable from model content. ValueIds are graph-internal
    /// identifiers not derivable from untrusted model data.
    absent_outputs: HashSet<ValueId>,
}

// OutboundGraphReader intentionally does NOT implement Send or Sync.
// The raw OrtNode pointers it holds are only valid within the ORT callback
// frame in which the reader was constructed. Granting Send/Sync would allow
// the reader (and its dangling pointers) to escape to other threads.
// Use it stack-locally inside a single GetCapability/Compile callback only.

impl OutboundGraphReader {
    /// Read an `OrtGraph*` and produce an IR `Graph`.
    ///
    /// # Safety
    ///
    /// `graph_ptr` must be a valid `OrtGraph*` from ORT, and the host ORT API
    /// must have been initialized via [`crate::status::set_host_api`].
    pub unsafe fn from_ort_graph(graph_ptr: *const ort::OrtGraph) -> Result<Self, String> {
        let api = crate::status::host_api();
        if api.is_null() {
            return Err("host ORT API not initialized".into());
        }

        let mut ir_graph = Graph::new();
        let mut ort_index_to_node_id = Vec::new();

        // Get nodes.
        let num_nodes = unsafe { Self::graph_num_nodes(api, graph_ptr)? };

        let mut ort_nodes: Vec<*const ort::OrtNode> = vec![ptr::null(); num_nodes];
        if num_nodes > 0 {
            unsafe { Self::graph_get_nodes(api, graph_ptr, ort_nodes.as_mut_ptr(), num_nodes)? };
        }

        // Get graph inputs (value infos).
        let num_inputs = unsafe { Self::graph_num_inputs(api, graph_ptr)? };
        let mut input_infos: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); num_inputs];
        if num_inputs > 0 {
            unsafe {
                Self::graph_get_inputs(api, graph_ptr, input_infos.as_mut_ptr(), num_inputs)?
            };
        }

        // Get graph outputs.
        let num_outputs = unsafe { Self::graph_num_outputs(api, graph_ptr)? };
        let mut output_infos: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); num_outputs];
        if num_outputs > 0 {
            unsafe {
                Self::graph_get_outputs(api, graph_ptr, output_infos.as_mut_ptr(), num_outputs)?
            };
        }

        // Build a name→ValueId map for wiring edges.
        let mut value_map: HashMap<String, ValueId> = HashMap::new();

        // Create graph input values.
        for info in &input_infos {
            if info.is_null() {
                continue;
            }
            let (name, dtype, shape) = unsafe { Self::read_value_info(api, *info)? };
            let vid = ir_graph.create_named_value(&name, dtype, shape);
            ir_graph.add_input(vid);
            value_map.insert(name, vid);
        }

        // First pass: create output values for all nodes.
        let mut node_output_names: Vec<Vec<(String, DataType)>> = Vec::with_capacity(num_nodes);
        let mut node_input_names: Vec<Vec<String>> = Vec::with_capacity(num_nodes);

        for ort_node in &ort_nodes {
            if ort_node.is_null() {
                node_output_names.push(Vec::new());
                node_input_names.push(Vec::new());
                continue;
            }

            // Read node outputs.
            let n_out = unsafe { Self::node_num_outputs(api, *ort_node)? };
            let mut out_infos: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); n_out];
            if n_out > 0 {
                unsafe {
                    Self::node_get_outputs(api, *ort_node, out_infos.as_mut_ptr(), n_out)?;
                }
            }

            let mut out_names = Vec::with_capacity(n_out);
            for info in &out_infos {
                if info.is_null() {
                    out_names.push((String::new(), DataType::Undefined));
                    continue;
                }
                let (name, dtype, shape) = unsafe { Self::read_value_info(api, *info)? };
                if !name.is_empty() && !value_map.contains_key(&name) {
                    let vid = ir_graph.create_named_value(&name, dtype, shape);
                    value_map.insert(name.clone(), vid);
                }
                out_names.push((name, dtype));
            }
            node_output_names.push(out_names);

            // Read node inputs.
            let n_in = unsafe { Self::node_num_inputs(api, *ort_node)? };
            let mut in_infos: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); n_in];
            if n_in > 0 {
                unsafe {
                    Self::node_get_inputs(api, *ort_node, in_infos.as_mut_ptr(), n_in)?;
                }
            }

            let mut in_names = Vec::with_capacity(n_in);
            for info in &in_infos {
                if info.is_null() {
                    in_names.push(String::new());
                    continue;
                }
                let (name, dtype, shape) = unsafe { Self::read_value_info(api, *info)? };
                if !name.is_empty() && !value_map.contains_key(&name) {
                    let vid = ir_graph.create_named_value(&name, dtype, shape);
                    value_map.insert(name.clone(), vid);
                }
                in_names.push(name);
            }
            node_input_names.push(in_names);
        }

        // Read initializers and copy small int64 tensors into owned data.
        let initializer_int64 =
            unsafe { Self::read_initializers_int64(api, graph_ptr).unwrap_or_default() };

        let mut absent_outputs: HashSet<ValueId> = HashSet::new();

        // Second pass: create nodes with proper edges and attributes.
        for (i, ort_node) in ort_nodes.iter().enumerate() {
            if ort_node.is_null() {
                continue;
            }

            let op_type = unsafe { Self::node_op_type(api, *ort_node)? };
            let domain = unsafe { Self::node_domain(api, *ort_node)? };
            let since_version = unsafe { Self::node_since_version(api, *ort_node)? };
            let name = unsafe { Self::node_name(api, *ort_node).unwrap_or_default() };

            let inputs: Vec<Option<ValueId>> = node_input_names[i]
                .iter()
                .map(|n| {
                    if n.is_empty() {
                        None
                    } else {
                        value_map.get(n).copied()
                    }
                })
                .collect();

            // Preserve positional output slots: ONNX addresses outputs by
            // position, so omitted optional outputs (empty name) must remain
            // as placeholder ValueIds rather than being compacted away, which
            // would shift downstream positions. Absent slots now carry the
            // ORT-declared dtype (not Undefined) so downstream scratch buffers
            // can be sized correctly for f16/bf16 ops.
            let outputs: Vec<ValueId> = node_output_names[i]
                .iter()
                .enumerate()
                .map(|(slot, (n, slot_dtype))| {
                    if n.is_empty() {
                        // Absent output slot — create a placeholder value so the
                        // position is preserved. The ValueId is recorded in
                        // `absent_outputs` (out-of-band) rather than using a
                        // magic name prefix that could be forged from model content.
                        // Carry the actual ORT-declared dtype so scratch buffers
                        // match the kernel's element size.
                        let vid = ir_graph.create_named_value(
                            format!("_absent_{i}_{slot}"),
                            *slot_dtype,
                            vec![],
                        );
                        absent_outputs.insert(vid);
                        vid
                    } else {
                        *value_map.get(n).unwrap_or_else(|| {
                            panic!("BUG: output name {n:?} not found in value_map")
                        })
                    }
                })
                .collect();

            let mut node = Node::new(NodeId(0), &op_type, inputs, outputs);
            node.name = name;
            node.domain = domain.clone();
            if since_version > 0 {
                node.version = Some(since_version as i64);
            }

            // Read node attributes from ORT and populate IR node.
            if let Ok(attrs) = unsafe { Self::read_node_attributes(api, *ort_node) } {
                node.attributes = attrs;
            }

            // For opset-13+ Unsqueeze/Squeeze: if axes attribute is missing but
            // input[1] is a constant initializer, inject it as an attribute so
            // ShapeInference::for_node can use it.
            if (op_type == "Unsqueeze" || op_type == "Squeeze")
                && since_version >= 13
                && !node.attributes.contains_key("axes")
                && let Some(axes_input_name) = node_input_names[i].get(1)
                && let Some(axes_data) = initializer_int64.get(axes_input_name)
            {
                node.attributes
                    .insert("axes".to_string(), Attribute::Ints(axes_data.clone()));
            }

            let nid = ir_graph.insert_node(node);
            ort_index_to_node_id.push(nid);
        }

        // Set graph outputs.
        for info in &output_infos {
            if info.is_null() {
                continue;
            }
            let (name, _dtype, _shape) = unsafe { Self::read_value_info(api, *info)? };
            if let Some(&vid) = value_map.get(&name) {
                ir_graph.add_output(vid);
            }
        }

        // Set default opset.
        ir_graph
            .opset_imports
            .insert(String::new(), ort::ORT_API_VERSION as u64);

        Ok(Self {
            graph: ir_graph,
            ort_index_to_node_id,
            ort_node_ptrs: ort_nodes,
            initializer_int64,
            absent_outputs,
        })
    }

    /// Get the IR graph for capability discovery.
    pub fn to_ir_graph(&self) -> &Graph {
        &self.graph
    }

    /// Out-of-band set of ValueIds representing absent optional output slots.
    /// These are not derivable from model content — they are assigned by the
    /// reader at graph construction time.
    pub fn absent_outputs(&self) -> &HashSet<ValueId> {
        &self.absent_outputs
    }

    /// Access the owned int64 initializer data (copied from ORT during construction).
    pub fn initializer_int64(&self) -> &HashMap<String, Vec<i64>> {
        &self.initializer_int64
    }

    /// Map our internal `NodeId` back to the ORT node's index in the original
    /// `Graph_GetNodes` array, for reporting claims.
    pub fn node_id_to_ort_index(&self, node_id: NodeId) -> usize {
        self.ort_index_to_node_id
            .iter()
            .position(|&nid| nid == node_id)
            .unwrap_or(0)
    }

    /// Get the original ORT `OrtNode*` pointer for one of our `NodeId`s.
    pub fn node_id_to_ort_ptr(&self, node_id: NodeId) -> *const ort::OrtNode {
        let idx = self.node_id_to_ort_index(node_id);
        self.ort_node_ptrs.get(idx).copied().unwrap_or(ptr::null())
    }

    // ─── ORT API wrappers ───────────────────────────────────────────────

    unsafe fn graph_num_nodes(
        api: *const ort::OrtApi,
        graph: *const ort::OrtGraph,
    ) -> Result<usize, String> {
        let f = unsafe { (*api).Graph_GetNumNodes }.ok_or("OrtApi.Graph_GetNumNodes is null")?;
        let mut count = 0usize;
        let status = unsafe { f(graph, &mut count) };
        Self::check(status)?;
        Ok(count)
    }

    unsafe fn graph_get_nodes(
        api: *const ort::OrtApi,
        graph: *const ort::OrtGraph,
        out: *mut *const ort::OrtNode,
        count: usize,
    ) -> Result<(), String> {
        let f = unsafe { (*api).Graph_GetNodes }.ok_or("OrtApi.Graph_GetNodes is null")?;
        let status = unsafe { f(graph, out, count) };
        Self::check(status)
    }

    unsafe fn graph_num_inputs(
        api: *const ort::OrtApi,
        graph: *const ort::OrtGraph,
    ) -> Result<usize, String> {
        let f = unsafe { (*api).Graph_GetNumInputs }.ok_or("OrtApi.Graph_GetNumInputs is null")?;
        let mut count = 0usize;
        let status = unsafe { f(graph, &mut count) };
        Self::check(status)?;
        Ok(count)
    }

    unsafe fn graph_get_inputs(
        api: *const ort::OrtApi,
        graph: *const ort::OrtGraph,
        out: *mut *const ort::OrtValueInfo,
        count: usize,
    ) -> Result<(), String> {
        let f = unsafe { (*api).Graph_GetInputs }.ok_or("OrtApi.Graph_GetInputs is null")?;
        let status = unsafe { f(graph, out, count) };
        Self::check(status)
    }

    unsafe fn graph_num_outputs(
        api: *const ort::OrtApi,
        graph: *const ort::OrtGraph,
    ) -> Result<usize, String> {
        let f =
            unsafe { (*api).Graph_GetNumOutputs }.ok_or("OrtApi.Graph_GetNumOutputs is null")?;
        let mut count = 0usize;
        let status = unsafe { f(graph, &mut count) };
        Self::check(status)?;
        Ok(count)
    }

    unsafe fn graph_get_outputs(
        api: *const ort::OrtApi,
        graph: *const ort::OrtGraph,
        out: *mut *const ort::OrtValueInfo,
        count: usize,
    ) -> Result<(), String> {
        let f = unsafe { (*api).Graph_GetOutputs }.ok_or("OrtApi.Graph_GetOutputs is null")?;
        let status = unsafe { f(graph, out, count) };
        Self::check(status)
    }

    unsafe fn node_num_inputs(
        api: *const ort::OrtApi,
        node: *const ort::OrtNode,
    ) -> Result<usize, String> {
        let f = unsafe { (*api).Node_GetNumInputs }.ok_or("OrtApi.Node_GetNumInputs is null")?;
        let mut count = 0usize;
        let status = unsafe { f(node, &mut count) };
        Self::check(status)?;
        Ok(count)
    }

    unsafe fn node_get_inputs(
        api: *const ort::OrtApi,
        node: *const ort::OrtNode,
        out: *mut *const ort::OrtValueInfo,
        count: usize,
    ) -> Result<(), String> {
        let f = unsafe { (*api).Node_GetInputs }.ok_or("OrtApi.Node_GetInputs is null")?;
        let status = unsafe { f(node, out, count) };
        Self::check(status)
    }

    unsafe fn node_num_outputs(
        api: *const ort::OrtApi,
        node: *const ort::OrtNode,
    ) -> Result<usize, String> {
        let f = unsafe { (*api).Node_GetNumOutputs }.ok_or("OrtApi.Node_GetNumOutputs is null")?;
        let mut count = 0usize;
        let status = unsafe { f(node, &mut count) };
        Self::check(status)?;
        Ok(count)
    }

    unsafe fn node_get_outputs(
        api: *const ort::OrtApi,
        node: *const ort::OrtNode,
        out: *mut *const ort::OrtValueInfo,
        count: usize,
    ) -> Result<(), String> {
        let f = unsafe { (*api).Node_GetOutputs }.ok_or("OrtApi.Node_GetOutputs is null")?;
        let status = unsafe { f(node, out, count) };
        Self::check(status)
    }

    unsafe fn node_op_type(
        api: *const ort::OrtApi,
        node: *const ort::OrtNode,
    ) -> Result<String, String> {
        let f =
            unsafe { (*api).Node_GetOperatorType }.ok_or("OrtApi.Node_GetOperatorType is null")?;
        let mut ptr: *const std::ffi::c_char = std::ptr::null();
        let status = unsafe { f(node, &mut ptr) };
        Self::check(status)?;
        if ptr.is_null() {
            return Err("Node_GetOperatorType returned null".into());
        }
        Ok(unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned())
    }

    unsafe fn node_domain(
        api: *const ort::OrtApi,
        node: *const ort::OrtNode,
    ) -> Result<String, String> {
        let f = unsafe { (*api).Node_GetDomain }.ok_or("OrtApi.Node_GetDomain is null")?;
        let mut ptr: *const std::ffi::c_char = std::ptr::null();
        let status = unsafe { f(node, &mut ptr) };
        Self::check(status)?;
        if ptr.is_null() {
            return Ok(String::new());
        }
        Ok(unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned())
    }

    unsafe fn node_since_version(
        api: *const ort::OrtApi,
        node: *const ort::OrtNode,
    ) -> Result<i32, String> {
        let f =
            unsafe { (*api).Node_GetSinceVersion }.ok_or("OrtApi.Node_GetSinceVersion is null")?;
        let mut version: i32 = 0;
        let status = unsafe { f(node, &mut version) };
        Self::check(status)?;
        Ok(version)
    }

    unsafe fn node_name(
        api: *const ort::OrtApi,
        node: *const ort::OrtNode,
    ) -> Result<String, String> {
        let f = unsafe { (*api).Node_GetName }.ok_or("OrtApi.Node_GetName is null")?;
        let mut ptr: *const std::ffi::c_char = std::ptr::null();
        let status = unsafe { f(node, &mut ptr) };
        Self::check(status)?;
        if ptr.is_null() {
            return Ok(String::new());
        }
        Ok(unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned())
    }

    /// Read dtype and shape from an `OrtValueInfo*`.
    unsafe fn read_value_info(
        api: *const ort::OrtApi,
        info: *const ort::OrtValueInfo,
    ) -> Result<(String, DataType, Shape), String> {
        // Get name.
        let get_name =
            unsafe { (*api).GetValueInfoName }.ok_or("OrtApi.GetValueInfoName is null")?;
        let mut name_ptr: *const std::ffi::c_char = ptr::null();
        let status = unsafe { get_name(info, &mut name_ptr) };
        Self::check(status)?;
        let name = if name_ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned()
        };

        // Get type info.
        let get_type_info =
            unsafe { (*api).GetValueInfoTypeInfo }.ok_or("OrtApi.GetValueInfoTypeInfo is null")?;
        let mut type_info: *const ort::OrtTypeInfo = ptr::null();
        let status = unsafe { get_type_info(info, &mut type_info) };
        Self::check(status)?;

        if type_info.is_null() {
            return Ok((name, DataType::Undefined, Vec::new()));
        }

        // Cast to tensor info.
        let cast_fn = unsafe { (*api).CastTypeInfoToTensorInfo }
            .ok_or("OrtApi.CastTypeInfoToTensorInfo is null")?;
        let mut tensor_info: *const ort::OrtTensorTypeAndShapeInfo = ptr::null();
        let status = unsafe { cast_fn(type_info, &mut tensor_info) };
        Self::check(status)?;

        if tensor_info.is_null() {
            return Ok((name, DataType::Undefined, Vec::new()));
        }

        // Get element type.
        let get_elem_type =
            unsafe { (*api).GetTensorElementType }.ok_or("OrtApi.GetTensorElementType is null")?;
        let mut elem_type: ort::ONNXTensorElementDataType = 0;
        let status = unsafe { get_elem_type(tensor_info, &mut elem_type) };
        Self::check(status)?;
        let dtype = DataType::from_onnx(elem_type as i32).unwrap_or(DataType::Undefined);

        // Get dimensions.
        let get_dims_count =
            unsafe { (*api).GetDimensionsCount }.ok_or("OrtApi.GetDimensionsCount is null")?;
        let mut dims_count = 0usize;
        let status = unsafe { get_dims_count(tensor_info, &mut dims_count) };
        Self::check(status)?;

        let shape = if dims_count > 0 {
            let get_dims = unsafe { (*api).GetDimensions }.ok_or("OrtApi.GetDimensions is null")?;
            let mut dims: Vec<i64> = vec![0; dims_count];
            let status = unsafe { get_dims(tensor_info, dims.as_mut_ptr(), dims_count) };
            Self::check(status)?;

            Shape::from(
                dims.iter()
                    .map(|&d| {
                        if d < 0 {
                            onnx_runtime_ir::Dim::Symbolic(onnx_runtime_ir::SymbolId(0))
                        } else {
                            onnx_runtime_ir::Dim::Static(d as usize)
                        }
                    })
                    .collect::<Vec<_>>(),
            )
        } else {
            Vec::new()
        };

        Ok((name, dtype, shape))
    }

    // ─── Attribute reading ─────────────────────────────────────────────

    /// Read all attributes from an ORT node and return them as owned IR
    /// `Attribute` values. Copies all data during the Compile call frame.
    unsafe fn read_node_attributes(
        api: *const ort::OrtApi,
        node: *const ort::OrtNode,
    ) -> Result<HashMap<String, Attribute>, String> {
        let get_num = unsafe { (*api).Node_GetNumAttributes }
            .ok_or("OrtApi.Node_GetNumAttributes is null")?;
        let mut num_attrs = 0usize;
        let status = unsafe { get_num(node, &mut num_attrs) };
        Self::check(status)?;

        if num_attrs == 0 {
            return Ok(HashMap::new());
        }

        let get_attrs =
            unsafe { (*api).Node_GetAttributes }.ok_or("OrtApi.Node_GetAttributes is null")?;
        let mut attr_ptrs: Vec<*const ort::OrtOpAttr> = vec![ptr::null(); num_attrs];
        let status = unsafe { get_attrs(node, attr_ptrs.as_mut_ptr(), num_attrs) };
        Self::check(status)?;

        let get_name = unsafe { (*api).OpAttr_GetName }.ok_or("OrtApi.OpAttr_GetName is null")?;
        let get_type = unsafe { (*api).OpAttr_GetType }.ok_or("OrtApi.OpAttr_GetType is null")?;
        let read_attr = unsafe { (*api).ReadOpAttr }.ok_or("OrtApi.ReadOpAttr is null")?;

        let mut result = HashMap::with_capacity(num_attrs);

        for &attr_ptr in &attr_ptrs {
            if attr_ptr.is_null() {
                continue;
            }

            // Get name.
            let mut name_ptr: *const std::ffi::c_char = ptr::null();
            let status = unsafe { get_name(attr_ptr, &mut name_ptr) };
            Self::check(status)?;
            if name_ptr.is_null() {
                continue;
            }
            let name = unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned();

            // Get type.
            let mut attr_type: ort::OrtOpAttrType = 0;
            let status = unsafe { get_type(attr_ptr, &mut attr_type) };
            Self::check(status)?;

            let attr_value = unsafe { Self::read_attr_value(read_attr, attr_ptr, attr_type) };

            if let Some(value) = attr_value {
                result.insert(name, value);
            }
        }

        Ok(result)
    }

    /// Read a single attribute value from its OrtOpAttr pointer.
    unsafe fn read_attr_value(
        read_attr: unsafe extern "C" fn(
            *const ort::OrtOpAttr,
            ort::OrtOpAttrType,
            *mut std::ffi::c_void,
            usize,
            *mut usize,
        ) -> *mut ort::OrtStatus,
        attr_ptr: *const ort::OrtOpAttr,
        attr_type: ort::OrtOpAttrType,
    ) -> Option<Attribute> {
        match attr_type {
            ort::ORT_OP_ATTR_INT => {
                let mut val: i64 = 0;
                let mut out_size = 0usize;
                let status = unsafe {
                    read_attr(
                        attr_ptr,
                        ort::ORT_OP_ATTR_INT,
                        (&raw mut val).cast(),
                        std::mem::size_of::<i64>(),
                        &mut out_size,
                    )
                };
                if status.is_null() {
                    Some(Attribute::Int(val))
                } else {
                    Self::check(status).ok();
                    None
                }
            }
            ort::ORT_OP_ATTR_INTS => {
                // First call with zero len to get required size.
                let mut required = 0usize;
                let status = unsafe {
                    read_attr(
                        attr_ptr,
                        ort::ORT_OP_ATTR_INTS,
                        ptr::null_mut(),
                        0,
                        &mut required,
                    )
                };
                // Status is non-null (buffer too small) but `required` has the count.
                if !status.is_null() {
                    Self::check(status).ok();
                }
                if required == 0 {
                    return Some(Attribute::Ints(Vec::new()));
                }
                let count = required / std::mem::size_of::<i64>();
                let mut buf: Vec<i64> = vec![0; count];
                let mut out_size = 0usize;
                let status = unsafe {
                    read_attr(
                        attr_ptr,
                        ort::ORT_OP_ATTR_INTS,
                        buf.as_mut_ptr().cast(),
                        required,
                        &mut out_size,
                    )
                };
                if status.is_null() {
                    Some(Attribute::Ints(buf))
                } else {
                    Self::check(status).ok();
                    None
                }
            }
            ort::ORT_OP_ATTR_FLOAT => {
                let mut val: f32 = 0.0;
                let mut out_size = 0usize;
                let status = unsafe {
                    read_attr(
                        attr_ptr,
                        ort::ORT_OP_ATTR_FLOAT,
                        (&raw mut val).cast(),
                        std::mem::size_of::<f32>(),
                        &mut out_size,
                    )
                };
                if status.is_null() {
                    Some(Attribute::Float(val))
                } else {
                    Self::check(status).ok();
                    None
                }
            }
            ort::ORT_OP_ATTR_STRING => {
                // First call to get required size.
                let mut required = 0usize;
                let status = unsafe {
                    read_attr(
                        attr_ptr,
                        ort::ORT_OP_ATTR_STRING,
                        ptr::null_mut(),
                        0,
                        &mut required,
                    )
                };
                if !status.is_null() {
                    Self::check(status).ok();
                }
                if required == 0 {
                    return Some(Attribute::String(Vec::new()));
                }
                let mut buf: Vec<u8> = vec![0; required];
                let mut out_size = 0usize;
                let status = unsafe {
                    read_attr(
                        attr_ptr,
                        ort::ORT_OP_ATTR_STRING,
                        buf.as_mut_ptr().cast(),
                        required,
                        &mut out_size,
                    )
                };
                if status.is_null() {
                    // Trim trailing null if present.
                    if buf.last() == Some(&0) {
                        buf.pop();
                    }
                    Some(Attribute::String(buf))
                } else {
                    Self::check(status).ok();
                    None
                }
            }
            // FLOATS, STRINGS, GRAPH, TENSOR — not needed for shape inference;
            // skip gracefully.
            _ => None,
        }
    }

    // ─── Initializer reading ────────────────────────────────────────────

    /// Read graph initializers and extract small int64 tensors as owned data.
    /// This copies the data during the callback frame; no ORT pointers are cached.
    unsafe fn read_initializers_int64(
        api: *const ort::OrtApi,
        graph: *const ort::OrtGraph,
    ) -> Result<HashMap<String, Vec<i64>>, String> {
        let get_num = unsafe { (*api).Graph_GetNumInitializers }
            .ok_or("OrtApi.Graph_GetNumInitializers is null")?;
        let mut num_init = 0usize;
        let status = unsafe { get_num(graph, &mut num_init) };
        Self::check(status)?;

        if num_init == 0 {
            return Ok(HashMap::new());
        }

        let get_inits = unsafe { (*api).Graph_GetInitializers }
            .ok_or("OrtApi.Graph_GetInitializers is null")?;
        let mut init_infos: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); num_init];
        let status = unsafe { get_inits(graph, init_infos.as_mut_ptr(), num_init) };
        Self::check(status)?;

        let get_init_val = unsafe { (*api).ValueInfo_GetInitializerValue }
            .ok_or("OrtApi.ValueInfo_GetInitializerValue is null")?;
        let get_name =
            unsafe { (*api).GetValueInfoName }.ok_or("OrtApi.GetValueInfoName is null")?;

        let get_type_info =
            unsafe { (*api).GetValueInfoTypeInfo }.ok_or("OrtApi.GetValueInfoTypeInfo is null")?;
        let cast_tensor = unsafe { (*api).CastTypeInfoToTensorInfo }
            .ok_or("OrtApi.CastTypeInfoToTensorInfo is null")?;
        let get_elem_type =
            unsafe { (*api).GetTensorElementType }.ok_or("OrtApi.GetTensorElementType is null")?;
        let get_dims_count =
            unsafe { (*api).GetDimensionsCount }.ok_or("OrtApi.GetDimensionsCount is null")?;
        let get_data = unsafe { (*api).GetTensorData }.ok_or("OrtApi.GetTensorData is null")?;
        let get_tensor_shape = unsafe { (*api).GetTensorTypeAndShape }
            .ok_or("OrtApi.GetTensorTypeAndShape is null")?;
        let get_shape_elem_count = unsafe { (*api).GetTensorShapeElementCount }
            .ok_or("OrtApi.GetTensorShapeElementCount is null")?;

        let mut result = HashMap::new();

        for &info in &init_infos {
            if info.is_null() {
                continue;
            }

            // Get name.
            let mut name_ptr: *const std::ffi::c_char = ptr::null();
            let status = unsafe { get_name(info, &mut name_ptr) };
            if !status.is_null() || name_ptr.is_null() {
                Self::check(status).ok();
                continue;
            }
            let name = unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned();
            if name.is_empty() {
                continue;
            }

            // Check if this is an int64 tensor and small enough to copy.
            let mut type_info: *const ort::OrtTypeInfo = ptr::null();
            let status = unsafe { get_type_info(info, &mut type_info) };
            if !status.is_null() || type_info.is_null() {
                Self::check(status).ok();
                continue;
            }
            let mut tensor_info: *const ort::OrtTensorTypeAndShapeInfo = ptr::null();
            let status = unsafe { cast_tensor(type_info, &mut tensor_info) };
            if !status.is_null() || tensor_info.is_null() {
                Self::check(status).ok();
                continue;
            }
            let mut elem_type: ort::ONNXTensorElementDataType = 0;
            let status = unsafe { get_elem_type(tensor_info, &mut elem_type) };
            if !status.is_null() {
                Self::check(status).ok();
                continue;
            }
            // Only copy int64 tensors (used for axes, shapes, etc.)
            if elem_type != ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 {
                continue;
            }
            let mut dims_count = 0usize;
            let status = unsafe { get_dims_count(tensor_info, &mut dims_count) };
            if !status.is_null() {
                Self::check(status).ok();
                continue;
            }
            // Only copy 1-D tensors with ≤ 64 elements (shape/axes metadata).
            if dims_count > 1 {
                continue;
            }

            // Get the initializer OrtValue.
            let mut ort_value: *const ort::OrtValue = ptr::null();
            let status = unsafe { get_init_val(info, &mut ort_value) };
            if !status.is_null() || ort_value.is_null() {
                Self::check(status).ok();
                continue;
            }

            // Get element count from the value's shape info.
            let mut val_shape: *mut ort::OrtTensorTypeAndShapeInfo = ptr::null_mut();
            let status = unsafe { get_tensor_shape(ort_value.cast_mut(), &mut val_shape) };
            if !status.is_null() || val_shape.is_null() {
                Self::check(status).ok();
                continue;
            }
            let mut elem_count = 0usize;
            let status = unsafe { get_shape_elem_count(val_shape, &mut elem_count) };
            // Release shape info.
            if let Some(rel) = unsafe { (*api).ReleaseTensorTypeAndShapeInfo } {
                unsafe { rel(val_shape) };
            }
            if !status.is_null() || elem_count == 0 || elem_count > 64 {
                Self::check(status).ok();
                continue;
            }

            // Read raw data.
            let mut data_ptr: *const std::ffi::c_void = ptr::null();
            let status = unsafe { get_data(ort_value, &mut data_ptr) };
            if !status.is_null() || data_ptr.is_null() {
                Self::check(status).ok();
                continue;
            }

            // Copy into owned Vec<i64>.
            let slice = unsafe { std::slice::from_raw_parts(data_ptr.cast::<i64>(), elem_count) };
            result.insert(name, slice.to_vec());
        }

        Ok(result)
    }

    /// Check an OrtStatus and convert to Result, extracting the error message.
    fn check(status: *mut ort::OrtStatus) -> Result<(), String> {
        if status.is_null() {
            return Ok(());
        }
        // Extract the real error message before releasing.
        let api = crate::status::host_api();
        let message = if !api.is_null() {
            // SAFETY: api is valid (set during CreateEpFactories).
            let msg_ptr = unsafe { (*api).GetErrorMessage.map(|f| f(status)) };
            let msg = msg_ptr
                .filter(|&p| !p.is_null())
                .map(|p| {
                    // SAFETY: ORT guarantees a null-terminated string.
                    unsafe { std::ffi::CStr::from_ptr(p) }
                        .to_string_lossy()
                        .into_owned()
                })
                .unwrap_or_else(|| "ORT API call failed".into());
            // Release the status only after we've copied the message.
            if let Some(release) = unsafe { (*api).ReleaseStatus } {
                unsafe { release(status) };
            }
            msg
        } else {
            "ORT API call failed (host API not available)".into()
        };
        Err(message)
    }
}
