use super::*;

/// Ops whose owned scratch is reserved by prepare-only planning (§736) so that
/// capacity refusal surfaces before request admission rather than as a late
/// device OOM. `BlockQuantizedMoE` (#747) reserves a session-persistent
/// workspace; `IndexShare` (#751) reserves a session-persistent workspace;
/// `com.microsoft::Attention` reserves a step-scoped Phase-2a scratch; the
/// default-domain `Attention` (#736) reserves one route-sized composite covering
/// its always-materialized f32 score matrix plus dense aliased K/V staging only
/// when used — step-scoped on the per-call prefill/batched route,
/// session-persistent on the capture-eligible single-token decode route (both
/// classes, fixed-capacity append and absent present-output staging charge zero);
/// `Conv` and the cuDNN `ReduceSum`/`ReduceMean` path reserve one
/// session-persistent cuDNN workspace when the current metadata makes its exact
/// byte size knowable; `com.microsoft::GroupQueryAttention` (#736) reserves one
/// session-persistent composite covering packed Q/K/V projection staging,
/// route-required BSH↔BNSH transpose scratch, and its f32 reference score
/// buffer; and the cuBLASLt GEMM family shares one session-persistent
/// heuristic-sized peak. Dynamic-axes reductions still use the same executor
/// workspace slot, but their first exact size is learned at execution time via
/// `workspace_requirement_for_execution` after the axes input is warmed.
pub(super) fn is_planned_workspace_node(node: &onnx_runtime_ir::Node) -> bool {
    (node.domain.is_empty()
        && matches!(
            node.op_type.as_str(),
            "MatMul" | "Gemm" | "Attention" | "Conv" | "ReduceSum" | "ReduceMean"
        ))
        || (node.domain == onnx_runtime_ir::RUNTIME_DOMAIN
            && matches!(node.op_type.as_str(), "BlockQuantizedMoE" | "IndexShare"))
        || (node.domain == "com.microsoft"
            && matches!(
                node.op_type.as_str(),
                "Attention"
                    | "GroupQueryAttention"
                    | "MatMulNBits"
                    | "FusedMatMulBias"
                    | "FusedGemm"
            ))
}

/// The justification for an upper bound applied to an otherwise-unresolved
/// dimension during prepare-only workspace planning. Both variants are a
/// *provable* ceiling on the dim's real extent, so reserving against them is
/// correct by construction for a *reservation* (which needs "enough", not
/// "exact"); a value that cannot be exceeded can never under-reserve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AxisBound {
    /// The symbol carries an explicit configured maximum (`max_seq_len`-style
    /// declared ceiling on the axis); the real extent can never exceed it.
    ConfiguredMax(usize),
    /// A recognized context/sequence axis bounded by the physically-allocated
    /// KV capacity established for this prepare call. Even a capacity-padded KV
    /// tensor cannot exceed its own allocation, so this is a hard ceiling.
    KvCapacity(usize),
}

impl AxisBound {
    pub(super) fn extent(self) -> usize {
        match self {
            AxisBound::ConfiguredMax(n) | AxisBound::KvCapacity(n) => n,
        }
    }
}

/// Outcome of resolving one planned-workspace node input shape for prepare-only
/// reservation. [`Self::Exact`] means every dim resolved by exact substitution
/// (the ONLY outcome for graphs that already resolved before this change — so
/// their reservations stay byte-identical). [`Self::Bounded`] means at least one
/// context/sequence-axis dim was unresolved and reserved against a provable
/// upper bound; the remaining outcome — unresolved *and* unbounded — is a hard
/// error, never a silent guess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum PlannedInputShape {
    Exact(Vec<usize>),
    Bounded {
        dims: Vec<usize>,
        /// `(axis, symbol, applied bound)` for each over-reserved dim.
        applied: Vec<(usize, SymbolId, AxisBound)>,
    },
}

impl PlannedInputShape {
    #[cfg(test)]
    pub(super) fn dims(&self) -> &[usize] {
        match self {
            PlannedInputShape::Exact(dims) | PlannedInputShape::Bounded { dims, .. } => dims,
        }
    }

    pub(super) fn into_dims(self) -> Vec<usize> {
        match self {
            PlannedInputShape::Exact(dims) | PlannedInputShape::Bounded { dims, .. } => dims,
        }
    }
}

/// Merge a per-node workspace requirement into the running peak for its
/// lifetime class, keeping the largest byte count and the strictest alignment.
fn merge_workspace_peak(peak: &mut WorkspaceRequirement, requirement: WorkspaceRequirement) {
    if requirement.bytes > peak.bytes {
        *peak = requirement;
    } else if requirement.bytes == peak.bytes {
        peak.alignment = peak.alignment.max(requirement.alignment);
    }
}

impl Executor {
    fn constant_i64_values(&self, value: ValueId) -> Option<Vec<i64>> {
        if let Some(WeightRef::Inline(tensor)) = self.graph.initializers.get(&value) {
            return match tensor.dtype {
                DataType::Int64 => onnx_runtime_ir::read_vec_le::<i64>(&tensor.data).ok(),
                DataType::Int32 => onnx_runtime_ir::read_vec_le::<i32>(&tensor.data)
                    .ok()
                    .map(|values| values.into_iter().map(i64::from).collect()),
                _ => None,
            };
        }
        let producer = self.graph.value(value).producer?;
        let node = self.graph.node(producer);
        if !node.domain.is_empty() || node.op_type != "Constant" {
            return None;
        }
        if let Some(values) = node.attr("value_ints").and_then(Attribute::as_ints) {
            return Some(values.to_vec());
        }
        match node.attr("value") {
            Some(Attribute::Tensor(tensor)) if tensor.dtype == DataType::Int64 => {
                onnx_runtime_ir::read_vec_le::<i64>(&tensor.data).ok()
            }
            Some(Attribute::Tensor(tensor)) if tensor.dtype == DataType::Int32 => {
                onnx_runtime_ir::read_vec_le::<i32>(&tensor.data)
                    .ok()
                    .map(|values| values.into_iter().map(i64::from).collect())
            }
            _ => None,
        }
    }

    fn normalize_i64_axis(axis: i64, rank: usize) -> Option<usize> {
        let rank = i64::try_from(rank).ok()?;
        let axis = if axis < 0 {
            axis.checked_add(rank)?
        } else {
            axis
        };
        if (0..rank).contains(&axis) {
            usize::try_from(axis).ok()
        } else {
            None
        }
    }

    fn normalize_i64_bound(bound: i64, len: usize) -> Option<i64> {
        let len = i64::try_from(len).ok()?;
        let bound = if bound < 0 {
            bound.checked_add(len)?
        } else {
            bound
        };
        Some(bound.clamp(0, len))
    }

    fn exact_runtime_i64_vector(
        &self,
        value: ValueId,
        symbols: &HashMap<SymbolId, usize>,
        depth: usize,
    ) -> Option<Vec<i64>> {
        if depth >= 16 {
            return None;
        }
        if let Some(values) = self.constant_i64_values(value) {
            return Some(values);
        }
        let producer = self.graph.value(value).producer?;
        let node = self.graph.node(producer);
        match (node.domain.as_str(), node.op_type.as_str()) {
            ("", "Shape") => {
                let input = node.inputs.first().copied().flatten()?;
                let shape = self.exact_runtime_value_shape(input, symbols, depth + 1)?;
                let rank = i64::try_from(shape.len()).ok()?;
                let start = node.attr("start").and_then(Attribute::as_int).unwrap_or(0);
                let end = node.attr("end").and_then(Attribute::as_int).unwrap_or(rank);
                let start = Self::normalize_i64_bound(start, shape.len())?;
                let end = Self::normalize_i64_bound(end, shape.len())?;
                if end < start {
                    return None;
                }
                shape
                    .get(usize::try_from(start).ok()?..usize::try_from(end).ok()?)?
                    .iter()
                    .map(|&dim| i64::try_from(dim).ok())
                    .collect()
            }
            ("", "Sub") => {
                let lhs = node.inputs.first().copied().flatten()?;
                let rhs = node.inputs.get(1).copied().flatten()?;
                let lhs = self.exact_runtime_i64_vector(lhs, symbols, depth + 1)?;
                let rhs = self.exact_runtime_i64_vector(rhs, symbols, depth + 1)?;
                let len = lhs.len().max(rhs.len());
                if !(lhs.len() == len || lhs.len() == 1) || !(rhs.len() == len || rhs.len() == 1) {
                    return None;
                }
                (0..len)
                    .map(|i| {
                        let a = lhs[if lhs.len() == 1 { 0 } else { i }];
                        let b = rhs[if rhs.len() == 1 { 0 } else { i }];
                        a.checked_sub(b)
                    })
                    .collect()
            }
            _ => None,
        }
    }

    fn broadcast_shapes(lhs: &[usize], rhs: &[usize]) -> Option<Vec<usize>> {
        let rank = lhs.len().max(rhs.len());
        let mut out = Vec::with_capacity(rank);
        for i in 0..rank {
            let a = lhs
                .len()
                .checked_sub(i + 1)
                .and_then(|axis| lhs.get(axis))
                .copied()
                .unwrap_or(1);
            let b = rhs
                .len()
                .checked_sub(i + 1)
                .and_then(|axis| rhs.get(axis))
                .copied()
                .unwrap_or(1);
            let dim = match (a, b) {
                (a, b) if a == b => a,
                (1, b) => b,
                (a, 1) => a,
                _ => return None,
            };
            out.push(dim);
        }
        out.reverse();
        Some(out)
    }

    fn exact_runtime_value_shape(
        &self,
        value: ValueId,
        symbols: &HashMap<SymbolId, usize>,
        depth: usize,
    ) -> Option<Vec<usize>> {
        if let Some(shape) = self
            .value_shapes
            .get(&value)
            .and_then(|shape| substitute(shape, symbols))
        {
            return Some(shape);
        }
        if depth >= 16 {
            return None;
        }
        let producer = self.graph.value(value).producer?;
        let node = self.graph.node(producer);
        match (node.domain.as_str(), node.op_type.as_str()) {
            ("", "Cast" | "CastLike" | "Identity" | "CumSum") => {
                let input = node.inputs.first().copied().flatten()?;
                self.exact_runtime_value_shape(input, symbols, depth + 1)
            }
            ("", "Unsqueeze") => {
                let input = node.inputs.first().copied().flatten()?;
                let mut shape = self.exact_runtime_value_shape(input, symbols, depth + 1)?;
                let axes = if let Some(axes) = node.attr("axes").and_then(Attribute::as_ints) {
                    axes.to_vec()
                } else {
                    let axes = node.inputs.get(1).copied().flatten()?;
                    self.constant_i64_values(axes)?
                };
                let output_rank = shape.len().checked_add(axes.len())?;
                let mut axes = axes
                    .into_iter()
                    .map(|axis| Self::normalize_i64_axis(axis, output_rank))
                    .collect::<Option<Vec<_>>>()?;
                axes.sort_unstable();
                axes.dedup();
                if axes.len() != output_rank - shape.len() {
                    return None;
                }
                for axis in axes {
                    shape.insert(axis, 1);
                }
                Some(shape)
            }
            ("", "Slice") => {
                let data = node.inputs.first().copied().flatten()?;
                let starts = node.inputs.get(1).copied().flatten()?;
                let ends = node.inputs.get(2).copied().flatten()?;
                let shape = self.exact_runtime_value_shape(data, symbols, depth + 1)?;
                let starts = self.exact_runtime_i64_vector(starts, symbols, depth + 1)?;
                let ends = self.exact_runtime_i64_vector(ends, symbols, depth + 1)?;
                let axes = match node.inputs.get(3).copied().flatten() {
                    Some(axes) => self.constant_i64_values(axes)?,
                    None => (0..i64::try_from(starts.len()).ok()?).collect(),
                };
                let steps = match node.inputs.get(4).copied().flatten() {
                    Some(steps) => self.constant_i64_values(steps)?,
                    None => vec![1; axes.len()],
                };
                if starts.len() != axes.len()
                    || ends.len() != axes.len()
                    || steps.len() != axes.len()
                {
                    return None;
                }
                let mut output = shape.clone();
                for ((&start, &end), (&axis, &step)) in
                    starts.iter().zip(&ends).zip(axes.iter().zip(&steps))
                {
                    if step <= 0 {
                        return None;
                    }
                    let axis = Self::normalize_i64_axis(axis, shape.len())?;
                    let start = Self::normalize_i64_bound(start, shape[axis])?;
                    let end = Self::normalize_i64_bound(end, shape[axis])?;
                    let len = if end <= start {
                        0
                    } else {
                        let span = usize::try_from(end.checked_sub(start)?).ok()?;
                        let step = usize::try_from(step).ok()?;
                        span.div_ceil(step)
                    };
                    output[axis] = len;
                }
                Some(output)
            }
            ("", "And" | "GreaterOrEqual") => {
                let lhs = node.inputs.first().copied().flatten()?;
                let rhs = node.inputs.get(1).copied().flatten()?;
                let lhs = self.exact_runtime_value_shape(lhs, symbols, depth + 1)?;
                let rhs = self.exact_runtime_value_shape(rhs, symbols, depth + 1)?;
                Self::broadcast_shapes(&lhs, &rhs)
            }
            ("", "Where") => {
                let condition = node.inputs.first().copied().flatten()?;
                let x = node.inputs.get(1).copied().flatten()?;
                let y = node.inputs.get(2).copied().flatten()?;
                let condition = self.exact_runtime_value_shape(condition, symbols, depth + 1)?;
                let x = self.exact_runtime_value_shape(x, symbols, depth + 1)?;
                let y = self.exact_runtime_value_shape(y, symbols, depth + 1)?;
                let xy = Self::broadcast_shapes(&x, &y)?;
                Self::broadcast_shapes(&condition, &xy)
            }
            ("", "Reshape") => {
                let data = node.inputs.first().copied().flatten()?;
                let target = node.inputs.get(1).copied().flatten()?;
                let data_shape = self.exact_runtime_value_shape(data, symbols, depth + 1)?;
                let target = self.constant_i64_values(target)?;
                let data_elements = data_shape
                    .iter()
                    .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))?;
                let mut output = Vec::with_capacity(target.len());
                let mut infer_axis = None;
                let mut known_elements = 1usize;
                for (axis, &dim) in target.iter().enumerate() {
                    let resolved = match dim {
                        -1 => {
                            if infer_axis.replace(axis).is_some() {
                                return None;
                            }
                            1
                        }
                        0 => *data_shape.get(axis)?,
                        n if n > 0 => usize::try_from(n).ok()?,
                        _ => return None,
                    };
                    known_elements = known_elements.checked_mul(resolved)?;
                    output.push(resolved);
                }
                if let Some(axis) = infer_axis {
                    if known_elements == 0 || !data_elements.is_multiple_of(known_elements) {
                        return None;
                    }
                    output[axis] = data_elements / known_elements;
                } else if known_elements != data_elements {
                    return None;
                }
                Some(output)
            }
            _ => None,
        }
    }

    pub(crate) fn prepare_mapped_growth(
        &self,
        bytes: u64,
        role: onnx_runtime_memory_governor::MemoryRole,
    ) -> Result<Option<onnx_runtime_memory_governor::MappedGrowthGrant>> {
        Ok(self.ep.prepare_mapped_growth(bytes, role)?)
    }

    pub(crate) fn release_mapped_growth(
        &self,
        bytes: u64,
        role: onnx_runtime_memory_governor::MemoryRole,
    ) {
        self.ep.release_mapped_growth(bytes, role);
    }

    pub(crate) fn workspace_node_locations(&self) -> Vec<String> {
        fn collect(graph: &Graph, scope: &str, out: &mut Vec<String>) {
            for (node_id, node) in graph.nodes.iter() {
                if is_planned_workspace_node(node) {
                    out.push(format!(
                        "{scope}node#{} '{}::{}'",
                        node_id.0, node.domain, node.op_type
                    ));
                }
            }
            for ((node_id, attribute), child) in &graph.subgraphs {
                collect(
                    child,
                    &format!("{scope}node#{}/{attribute}/", node_id.0),
                    out,
                );
            }
        }
        let mut locations = Vec::new();
        collect(&self.graph, "", &mut locations);
        locations
    }

    /// Derive a *provable* upper bound for an unresolved symbolic axis, or
    /// `None` when no justifiable ceiling exists (in which case the caller must
    /// fail rather than guess). A reservation needs "enough", not "exact": any
    /// value the real extent cannot exceed is correct by construction here.
    ///
    /// Two — and only two — ceilings are justifiable:
    /// 1. A recognized context/sequence axis
    ///    ([`Self::capture_growing_symbols`]) is bounded by the physically
    ///    allocated KV capacity established for this prepare call (the max
    ///    concrete extent bound to any growing symbol). This is the true ceiling
    ///    even when the KV tensor is capacity-padded beyond the model's declared
    ///    `max_seq_len`, so it is preferred; if the model *also* declares a
    ///    larger maximum, the larger of the two is taken to never under-reserve.
    /// 2. Any axis carrying its own configured maximum
    ///    ([`SymbolConstraints::max`], the `max_seq_len`-style declared ceiling)
    ///    is bounded by that maximum.
    ///
    /// Anything else — an unbound symbol that is neither a known sequence axis
    /// nor carries a declared maximum — returns `None`: its extent is genuinely
    /// unknown, and reserving against a guess would silently under-reserve
    /// (memory corruption surfacing far from here), which is exactly the class
    /// of bug this planner must refuse to introduce.
    fn planned_axis_upper_bound(
        &self,
        symbol: SymbolId,
        symbols: &HashMap<SymbolId, usize>,
    ) -> Option<AxisBound> {
        let declared_max = self
            .graph
            .symbol_constraints
            .get(&symbol)
            .and_then(|c| c.max);
        if self.capture_growing_symbols.contains(&symbol)
            && let Some(capacity) = self
                .capture_growing_symbols
                .iter()
                .filter_map(|growing| symbols.get(growing).copied())
                .max()
        {
            return Some(AxisBound::KvCapacity(
                capacity.max(declared_max.unwrap_or(0)),
            ));
        }
        declared_max.map(AxisBound::ConfiguredMax)
    }

    /// Resolve a planned-workspace node input shape for prepare-only reservation.
    /// Prefers exact substitution; where a dim is unresolved *because it is a
    /// context/sequence axis* (or otherwise carries a configured maximum), binds
    /// it to a provable upper bound and reports the over-reservation. A dim
    /// unresolved for any *other* reason is a hard error — there the extent is
    /// genuinely unknown and a guess would under-reserve.
    pub(super) fn resolve_planned_workspace_input_shape(
        &self,
        value: ValueId,
        symbols: &HashMap<SymbolId, usize>,
        node_id: NodeId,
        node: &Node,
        index: usize,
    ) -> Result<PlannedInputShape> {
        if let Some(dims) = self.exact_runtime_value_shape(value, symbols, 0) {
            return Ok(PlannedInputShape::Exact(dims));
        }
        let shape = self.value_shapes.get(&value).ok_or_else(|| {
            SessionError::Internal(format!(
                "prepare-only workspace planning has no loader shape for input {index} of \
                 node {} ('{}::{}')",
                node_id.0, node.domain, node.op_type
            ))
        })?;
        let mut dims = Vec::with_capacity(shape.len());
        let mut applied = Vec::new();
        for (axis, dim) in shape.iter().enumerate() {
            match dim {
                Dim::Static(n) => dims.push(*n),
                Dim::Symbolic(symbol) => {
                    if let Some(&bound) = symbols.get(symbol) {
                        dims.push(bound);
                    } else if let Some(axis_bound) = self.planned_axis_upper_bound(*symbol, symbols)
                    {
                        dims.push(axis_bound.extent());
                        applied.push((axis, *symbol, axis_bound));
                    } else {
                        let value_name = self
                            .graph
                            .value(value)
                            .name
                            .clone()
                            .unwrap_or_else(|| format!("value#{}", value.0));
                        let symbol_name = self
                            .symbol_name(*symbol)
                            .unwrap_or_else(|| format!("symbol#{}", symbol.0));
                        return Err(SessionError::Internal(format!(
                            "prepare-only workspace planning cannot resolve input {index} \
                             '{value_name}' for node {} ('{}::{}'): axis {axis} is symbolic \
                             ('{symbol_name}') and unresolved, and is neither a context/sequence \
                             axis bounded by max_seq_len/KV capacity nor an axis with a configured \
                             maximum — its extent is genuinely unknown, so reserving against a \
                             guess would under-reserve",
                            node_id.0, node.domain, node.op_type
                        )));
                    }
                }
            }
        }
        if applied.is_empty() {
            Ok(PlannedInputShape::Exact(dims))
        } else {
            Ok(PlannedInputShape::Bounded { dims, applied })
        }
    }

    /// Resolve concrete metadata and reserve kernel workspace without executing
    /// any graph node.
    pub(crate) fn prepare_with_device_bindings(
        &mut self,
        inputs: &[(&str, &Tensor)],
        bindings: &mut [DeviceIoBinding],
    ) -> Result<WorkspaceRequirement> {
        if self.heterogeneous.is_some() {
            return Err(heterogeneous_api_error(
                "workspace preparation with persistent device bindings",
            ));
        }
        self.workspace_preparation_required = true;
        let external = self.prepare_external_bindings_mode(bindings, true)?;
        let symbols = self.bind_symbols(inputs, &external)?;
        // Track one peak per lifetime class: session-persistent and step-scoped
        // scratch live in separate executor slots, so a single peak cannot stand
        // in for both when a graph mixes governed ops (e.g. QMoE + Attention).
        let mut peak_persistent = WorkspaceRequirement::NONE;
        let mut peak_step = WorkspaceRequirement::NONE;

        for pi in 0..self.plan.len() {
            let node_id = self.plan[pi].node_id;
            let node = self.graph.node(node_id);
            if !is_planned_workspace_node(node) {
                continue;
            }
            let planned_shapes = self.plan[pi]
                .inputs
                .iter()
                .enumerate()
                .map(|(index, input)| match input {
                    None => Ok(PlannedInputShape::Exact(Vec::new())),
                    Some(value) => self.resolve_planned_workspace_input_shape(
                        *value, &symbols, node_id, node, index,
                    ),
                })
                .collect::<Result<Vec<_>>>()?;
            if std::env::var("ONNX_GENAI_LOG_WORKSPACE_BOUND").is_ok() {
                for (index, planned) in planned_shapes.iter().enumerate() {
                    if let PlannedInputShape::Bounded { dims, applied } = planned {
                        eprintln!(
                            "[onnx-genai-workspace] node {} ('{}::{}') input {index} \
                             over-reserved against bounded axis: dims={dims:?} bounds={applied:?}",
                            node_id.0, node.domain, node.op_type
                        );
                    }
                }
            }
            let input_shapes = planned_shapes
                .into_iter()
                .map(PlannedInputShape::into_dims)
                .collect::<Vec<_>>();
            let constant_inputs = self.plan[pi]
                .inputs
                .iter()
                .map(|input| {
                    input.is_some_and(|value| self.graph.initializers.contains_key(&value))
                })
                .collect::<Vec<_>>();
            let constant_values = resolve_kernel_constant_inputs(
                &self.graph,
                &self.weights,
                &self.plan[pi].inputs,
                &input_shapes,
            )?;
            let opset = effective_opset(&self.graph, node);
            // Must match what dispatch computes for this node, or prepare-only
            // planning would key a different kernel than execution uses.
            let seq_independent =
                node_capture_seq_independent(&self.graph, node, &self.capture_growing_symbols);
            let graph_tokens = std::array::from_fn(|index| {
                let capture = &self.slot_capture[index];
                capture
                    .capture_schedule
                    .as_ref()
                    .is_some_and(|schedule| {
                        schedule.segments.iter().any(|segment| {
                            segment.captured && (segment.start..segment.end).contains(&pi)
                        })
                    })
                    .then_some(capture.device_graph_token)
                    .flatten()
            });
            let (kernel, key) = self.cache.get_or_create(
                node_id,
                node,
                &input_shapes,
                &self.plan[pi].input_dtypes,
                &constant_inputs,
                &constant_values,
                opset,
                seq_independent,
                self.ep.as_ref(),
                graph_tokens,
            )?;
            self.kernel_bindings[pi] = Some(key);
            let metadata = input_shapes
                .iter()
                .zip(&self.plan[pi].input_dtypes)
                .zip(&self.plan[pi].inputs)
                .map(|((shape, dtype), input)| TensorMetadata::new(*dtype, shape, input.is_some()))
                .collect::<Vec<_>>();
            let requirement = kernel.workspace_requirement(&metadata)?;
            if requirement.bytes == 0 {
                continue;
            }
            match requirement.role {
                onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped }
                    if step_scoped
                        == matches!(requirement.lifetime, WorkspaceLifetime::StepScoped) => {}
                _ => {
                    return Err(SessionError::Internal(format!(
                        "node {} ('{}::{}') returned an inconsistent workspace role/lifetime",
                        node_id.0, node.domain, node.op_type
                    )));
                }
            }
            let peak = match requirement.lifetime {
                WorkspaceLifetime::SessionPersistent => &mut peak_persistent,
                WorkspaceLifetime::StepScoped => &mut peak_step,
            };
            merge_workspace_peak(peak, requirement);
        }
        let child_graphs = self.graph.subgraphs.values().collect::<Vec<_>>();
        for child in child_graphs {
            self.collect_nested_workspace_requirement(
                child,
                &symbols,
                "control-flow/",
                &mut peak_persistent,
                &mut peak_step,
            )?;
        }

        Self::reserve_prepared_workspace(
            self.ep.as_ref(),
            &mut self.persistent_workspace,
            peak_persistent,
        )?;
        Self::reserve_prepared_workspace(self.ep.as_ref(), &mut self.step_workspace, peak_step)?;

        // Return the dominant requirement for callers that assert on a single
        // reserved workspace; the two slots are prepared independently above.
        Ok(if peak_persistent.bytes >= peak_step.bytes {
            peak_persistent
        } else {
            peak_step
        })
    }

    /// Reserve `peak` into `slot` against the device authority, reusing a
    /// large-enough existing preparation. A zero requirement is a no-op.
    fn reserve_prepared_workspace(
        ep: &dyn ExecutionProvider,
        slot: &mut Option<PreparedWorkspace>,
        peak: WorkspaceRequirement,
    ) -> Result<()> {
        if peak.bytes == 0 {
            return Ok(());
        }
        let bytes = usize::try_from(peak.bytes).map_err(|_| {
            SessionError::Internal(format!(
                "workspace requirement {} does not fit usize",
                peak.bytes
            ))
        })?;
        if slot
            .as_ref()
            .is_some_and(|prepared| prepared.bytes >= bytes && prepared.alignment >= peak.alignment)
        {
            return Ok(());
        };
        let old = slot.take().map(|workspace| workspace.buffer);
        let fresh = ep.replace_workspace(old, bytes, peak.alignment, peak.role)?;
        *slot = Some(PreparedWorkspace {
            buffer: fresh,
            bytes,
            alignment: peak.alignment,
        });
        Ok(())
    }

    fn collect_nested_workspace_requirement(
        &self,
        graph: &Graph,
        symbols: &HashMap<SymbolId, usize>,
        scope: &str,
        peak_persistent: &mut WorkspaceRequirement,
        peak_step: &mut WorkspaceRequirement,
    ) -> Result<()> {
        for (node_id, node) in graph.nodes.iter() {
            if !is_planned_workspace_node(node) {
                continue;
            }
            let input_shapes = node
                .inputs
                .iter()
                .enumerate()
                .map(|(index, input)| {
                    match input {
                    None => Ok(Vec::new()),
                    Some(value) => substitute(&graph.value(*value).shape, symbols).ok_or_else(|| {
                        let value_name = graph
                            .value(*value)
                            .name
                            .clone()
                            .unwrap_or_else(|| format!("value#{}", value.0));
                        SessionError::Internal(format!(
                            "prepare-only workspace planning cannot resolve nested input {index} \
                             '{value_name}' for {scope}node#{} ('{}::{}'); its formal/captured \
                             shape is runtime-dependent and no exact graph-metadata bound is \
                             available",
                            node_id.0, node.domain, node.op_type
                        ))
                    }),
                }
                })
                .collect::<Result<Vec<_>>>()?;
            let mut kernel =
                self.ep
                    .get_kernel(node, &input_shapes, effective_opset(&self.graph, node))?;
            let constant_inputs = node
                .inputs
                .iter()
                .map(|input| input.is_some_and(|value| graph.initializers.contains_key(&value)))
                .collect::<Vec<_>>();
            kernel.set_constant_inputs(&constant_inputs);
            let constant_values =
                resolve_kernel_constant_inputs(graph, &self.weights, &node.inputs, &input_shapes)?;
            kernel.prepare_constant_inputs(&constant_values, self.ep.as_ref())?;
            let metadata = input_shapes
                .iter()
                .zip(&node.inputs)
                .map(|(shape, input)| {
                    let dtype = input
                        .map(|value| graph.value(value).dtype)
                        .unwrap_or(DataType::Undefined);
                    TensorMetadata::new(dtype, shape, input.is_some())
                })
                .collect::<Vec<_>>();
            let requirement = kernel.workspace_requirement(&metadata)?;
            if requirement.bytes == 0 {
                continue;
            }
            match requirement.role {
                onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped }
                    if step_scoped
                        == matches!(requirement.lifetime, WorkspaceLifetime::StepScoped) => {}
                _ => {
                    return Err(SessionError::Internal(format!(
                        "{scope}node#{} ('{}::{}') returned an inconsistent workspace role/lifetime",
                        node_id.0, node.domain, node.op_type
                    )));
                }
            }
            let peak = match requirement.lifetime {
                WorkspaceLifetime::SessionPersistent => &mut *peak_persistent,
                WorkspaceLifetime::StepScoped => &mut *peak_step,
            };
            merge_workspace_peak(peak, requirement);
        }
        for ((node_id, attribute), child) in &graph.subgraphs {
            self.collect_nested_workspace_requirement(
                child,
                symbols,
                &format!("{scope}node#{}/{attribute}/", node_id.0),
                peak_persistent,
                peak_step,
            )?;
        }
        Ok(())
    }

    pub(super) fn release_step_workspace(&mut self) -> Result<()> {
        // When pinned (an installed fixed-shape verify graph baked this buffer's
        // address), keep the StepScoped scratch alive so a later replay reads a
        // valid pointer. `reserve_prepared_workspace` reuses it whenever the next
        // requirement fits, so the pointer stays stable across replays.
        if self.pin_step_workspace {
            return Ok(());
        }
        if let Some(workspace) = self.step_workspace.take() {
            self.ep.deallocate_workspace(workspace.buffer)?;
        }
        Ok(())
    }

    /// Whether this executor's StepScoped workspace is pinned across runs.
    pub(crate) fn step_workspace_pinned(&self) -> bool {
        self.pin_step_workspace
    }

    /// Pin (or unpin) this executor's StepScoped workspace across runs. Pinning
    /// keeps the reserved scratch buffer alive so a captured fixed-shape graph
    /// (the M=K speculative verify) replays against a stable pointer; unpinning
    /// frees it on the next release. See [`Self::pin_step_workspace`].
    pub(crate) fn set_pin_step_workspace(&mut self, pin: bool) {
        self.pin_step_workspace = pin;
    }

    /// Bind the graph's symbols to concrete sizes from the actual bound-input
    /// shapes, validating rank and static dims and detecting symbol conflicts.
    pub(super) fn bind_symbols(
        &self,
        inputs: &[(&str, &Tensor)],
        external: &ExternalBindings,
    ) -> Result<HashMap<SymbolId, usize>> {
        let mut bindings: HashMap<SymbolId, usize> = HashMap::new();
        for (name, tensor) in inputs {
            let vid = *self
                .input_index
                .get(*name)
                .ok_or_else(|| SessionError::InputNotFound {
                    name: (*name).to_string(),
                })?;
            self.bind_input_shape(name, vid, tensor.dtype, &tensor.shape, &mut bindings)?;
        }
        for (&vid, value) in &external.inputs {
            let name = self.graph.value(vid).name.as_deref().unwrap_or("<unnamed>");
            self.bind_input_shape(name, vid, value.dtype, &value.shape, &mut bindings)?;
        }
        Ok(bindings)
    }

    pub(super) fn bind_input_shape(
        &self,
        name: &str,
        vid: ValueId,
        dtype: DataType,
        shape: &[usize],
        bindings: &mut HashMap<SymbolId, usize>,
    ) -> Result<()> {
        let want_dtype = self.value_dtypes[&vid];
        if dtype != want_dtype {
            return Err(SessionError::DtypeMismatch {
                name: name.to_string(),
                expected: format!("{want_dtype:?}"),
                got: format!("{dtype:?}"),
            });
        }
        let decl = &self.value_shapes[&vid];
        if decl.len() != shape.len() {
            return Err(SessionError::RankMismatch {
                name: name.to_string(),
                expected: decl.len(),
                got: shape.len(),
            });
        }
        for (dim, &actual) in decl.iter().zip(shape) {
            match dim {
                Dim::Static(n) if *n != actual => {
                    return Err(SessionError::ShapeMismatch {
                        name: name.to_string(),
                        expected: as_static_shape(decl).unwrap_or_default(),
                        got: shape.to_vec(),
                    });
                }
                Dim::Static(_) => {}
                Dim::Symbolic(s) => {
                    if let Some(&prev) = bindings.get(s) {
                        if prev != actual {
                            let sym = self
                                .symbol_name(*s)
                                .unwrap_or_else(|| format!("symbol#{}", s.0));
                            return Err(SessionError::SymbolConflict {
                                symbol: sym,
                                first: prev,
                                second: actual,
                            });
                        }
                    } else {
                        bindings.insert(*s, actual);
                    }
                }
            }
        }
        Ok(())
    }

    /// Human-readable name of a symbol, if the graph recorded one.
    pub(super) fn symbol_name(&self, s: SymbolId) -> Option<String> {
        self.graph
            .symbol_constraints
            .get(&s)
            .and_then(|c| c.name.clone())
    }

    /// Sequential topological executor.
    pub(crate) fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Result<Vec<Tensor>> {
        self.run_outputs(inputs)?
            .into_iter()
            .map(|output| {
                match output {
                    SessionOutput::Tensor(tensor) => Ok(tensor),
                    SessionOutput::Sequence(_) => Err(SessionError::SequenceOp {
                        op: "<graph output>".to_string(),
                        reason: "the tensor-only run API received a Sequence graph output; use InferenceSession::run_outputs to preserve sequence values".to_string(),
                    }),
                }
            })
            .collect()
    }

    pub(crate) fn run_outputs(&mut self, inputs: &[(&str, &Tensor)]) -> Result<Vec<SessionOutput>> {
        if let Some(heterogeneous) = self.heterogeneous.as_mut() {
            return heterogeneous
                .run(inputs)
                .map(|outputs| outputs.into_iter().map(SessionOutput::Tensor).collect());
        }
        let result = self.run_scoped(inputs, &HashMap::new(), &ExternalBindings::default());
        self.release_step_workspace()?;
        result?
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    SessionError::Internal(
                        "ordinary run unexpectedly suppressed a bound graph output".into(),
                    )
                })
            })
            .collect()
    }

    pub(crate) fn run_with_device_bindings(
        &mut self,
        inputs: &[(&str, &Tensor)],
        bindings: &mut [DeviceIoBinding],
    ) -> Result<crate::DeviceBindingOutputs> {
        if self.heterogeneous.is_some() {
            return Err(heterogeneous_api_error(
                "execution with persistent device bindings/state",
            ));
        }
        let validation_submission =
            self.begin_device_validation_submission_for_bindings(bindings)?;
        let external = self.prepare_external_bindings(bindings)?;
        let result = self.run_scoped_mode(
            inputs,
            &HashMap::new(),
            &external,
            RunMode::Eager,
            Some(validation_submission),
        );
        self.scratch_external_bindings = external;
        self.release_step_workspace()?;
        let outputs = match result? {
            ScopedRunResult::Executed(outputs) => outputs,
            ScopedRunResult::NotCapturable(_) => unreachable!("eager runs are always executed"),
        };
        outputs
            .into_iter()
            .map(|output| match output {
                None => Ok(None),
                Some(SessionOutput::Tensor(tensor)) => Ok(Some(tensor)),
                Some(SessionOutput::Sequence(_)) => Err(SessionError::SequenceOp {
                    op: "<graph output>".to_string(),
                    reason: "run_with_device_bindings cannot return an unbound Sequence graph output; use run_outputs without tensor device bindings".to_string(),
                }),
            })
            .collect()
    }

    pub(crate) fn arm_block_quantized_moe_traffic(&mut self, request_id: u32) -> Result<usize> {
        self.reset_device_graph()?;
        self.cache.arm_block_quantized_moe_traffic(request_id)
    }

    pub(crate) fn reset_block_quantized_moe_traffic(&mut self) -> Result<()> {
        self.cache.reset_block_quantized_moe_traffic()
    }

    pub(crate) fn snapshot_block_quantized_moe_traffic(
        &self,
    ) -> Result<onnx_runtime_ep_api::BlockQuantizedMoeTraffic> {
        self.cache.snapshot_block_quantized_moe_traffic()
    }

    #[cfg(feature = "gpu-tests")]
    pub(crate) fn inject_block_quantized_moe_traffic_fault_for_test(
        &self,
        fault: onnx_runtime_ep_cuda::kernels::block_quantized_moe::BlockQuantizedMoeTrafficFaultForTest,
    ) -> Result<()> {
        self.cache
            .inject_block_quantized_moe_traffic_fault_for_test(fault)
    }

    pub(crate) fn disarm_block_quantized_moe_traffic(&mut self) -> Result<()> {
        self.reset_device_graph()?;
        self.cache.disarm_block_quantized_moe_traffic()
    }

    pub(crate) fn try_capture_with_device_bindings(
        &mut self,
        inputs: &[(&str, &Tensor)],
        bindings: &mut [DeviceIoBinding],
    ) -> Result<DeviceGraphCaptureResult> {
        if self.heterogeneous.is_some() {
            return Err(heterogeneous_api_error(
                "mixed-provider device-graph capture",
            ));
        }
        if self.cap().device_graph_token.is_some() {
            self.reset_device_graph()?;
        }
        let external = self.prepare_external_bindings(bindings)?;
        let result =
            self.run_scoped_mode(inputs, &HashMap::new(), &external, RunMode::Capture, None);
        self.scratch_external_bindings = external;
        self.release_step_workspace()?;
        match result? {
            ScopedRunResult::Executed(outputs) => {
                let mut tensors = Vec::with_capacity(outputs.len());
                for output in outputs {
                    match output {
                        None => tensors.push(None),
                        Some(SessionOutput::Tensor(tensor)) => tensors.push(Some(tensor)),
                        Some(SessionOutput::Sequence(_)) => {
                            self.reset_device_graph()?;
                            return Ok(DeviceGraphCaptureResult::NotCapturable(
                                CaptureDeclineReport::one(CaptureDecline::graph(
                                    "device graph capture cannot return a Sequence graph output",
                                )),
                            ));
                        }
                    }
                }
                let token = self.cap().device_graph_token.ok_or_else(|| {
                    SessionError::Internal(
                        "device graph capture completed without an installation token".into(),
                    )
                })?;
                for binding in bindings.iter_mut() {
                    binding.set_device_graph_token(token);
                }
                self.cap_mut().device_graph_signature = Some(Self::binding_signature(bindings));
                Ok(DeviceGraphCaptureResult::Captured(tensors))
            }
            ScopedRunResult::NotCapturable(reason) => {
                Ok(DeviceGraphCaptureResult::NotCapturable(reason))
            }
        }
    }

    /// Replay the installed device graph for one decode step. Returns `true` when
    /// the graph remains installed and valid for the next step, or `false` when a
    /// control-flow branch flip retired it mid-step (the token was still produced
    /// correctly via an eager fallback) and the caller must re-warm/re-capture.
    pub(crate) fn replay_device_graph(&mut self, bindings: &mut [DeviceIoBinding]) -> Result<bool> {
        if self.heterogeneous.is_some() {
            return Err(heterogeneous_api_error(
                "mixed-provider device-graph replay",
            ));
        }
        if !self.bindings_match_graph_signature(bindings) {
            self.reset_device_graph()?;
            return Err(SessionError::Internal(
                "device graph replay bindings changed logical/physical shape, logical-exposure \
                 policy, address, or I/O identity; graph was invalidated"
                    .into(),
            ));
        }
        let token = self.cap().device_graph_token.ok_or_else(|| {
            SessionError::Internal("device graph replay has no installation token".into())
        })?;
        // The installed graph for this slot can be reset out-of-band while our
        // host-side signature/schedule stays live: a kernel-variant eviction
        // retires kernels baked into a captured graph and resets BOTH the Primary
        // (M=1 decode) and Verify (M=K speculative) EP slots. That desync is
        // reachable only once both slots are populated (MTP: the M=1 base decode
        // and the M=K verify each install a graph, doubling per-node kernel
        // variants past the eviction bound). Replaying an emptied slot would
        // hard-error ("no executable is installed"); detect it and report an
        // invalidation so the caller re-warms and re-captures, exactly as it does
        // for a control-flow branch flip.
        if !self.ep.has_owned_device_graph(token)? {
            self.reset_device_graph()?;
            return Ok(false);
        }
        // Whole-subgraph capture (a single graph, no eager seams) keeps the
        // zero-host-work fast path: just relaunch the one installed graph.
        // Segmented capture must re-establish the run context and interleave
        // segment replays with eager seam-node execution, so it routes through
        // the scoped runner in replay mode.
        let single_graph = self
            .cap()
            .capture_schedule
            .as_ref()
            .is_none_or(CaptureSchedule::is_single_graph);
        if single_graph {
            let mut validation_submission =
                self.begin_device_validation_submission_for_bindings(bindings)?;
            if let Err(replay_error) = self.ep.replay_owned_device_graph(token) {
                let validation = self.finish_device_validation_boundary();
                validation_submission.disarm();
                return match validation {
                    Ok(()) => Err(replay_error.into()),
                    Err(validation_error) => Err(SessionError::Internal(format!(
                        "device graph replay failed: {replay_error}; deferred validation cleanup \
                         also failed: {validation_error}"
                    ))),
                };
            }
            validation_submission.disarm();
            return Ok(true);
        }
        let validation_submission =
            self.begin_device_validation_submission_for_bindings(bindings)?;
        let external = self.prepare_external_bindings(bindings)?;
        let result = self.run_scoped_mode(
            &[],
            &HashMap::new(),
            &external,
            RunMode::Replay,
            Some(validation_submission),
        );
        self.scratch_external_bindings = external;
        self.release_step_workspace()?;
        match result? {
            // `run_scoped_mode` clears `capture_schedule` when a branch flip
            // retired the graph this step; report that so the caller re-arms.
            ScopedRunResult::Executed(_) => Ok(self.cap().capture_schedule.is_some()),
            ScopedRunResult::NotCapturable(reason) => {
                self.reset_device_graph()?;
                Err(SessionError::Internal(format!(
                    "segmented device graph replay lost its schedule: {reason}"
                )))
            }
        }
    }

    pub(crate) fn reset_device_graph(&mut self) -> Result<bool> {
        if self.heterogeneous.is_some() {
            return Err(heterogeneous_api_error("mixed-provider device-graph reset"));
        }
        self.finish_device_validation_boundary()?;
        let token = self.cap().device_graph_token;
        let reset = match token {
            Some(token) => self.ep.reset_owned_device_graph(token)?,
            None => false,
        };
        if token.is_some() {
            let cap = self.cap_mut();
            cap.device_graph_token = None;
            cap.device_graph_signature = None;
            cap.capture_schedule = None;
            cap.capture_cf_shapes.clear();
            cap.capture_warm_seeded.clear();
        }
        Ok(reset)
    }

    /// Which of the EP's captured-graph slots this executor drives.
    pub(crate) fn graph_slot(&self) -> DeviceGraphSlot {
        self.graph_slot
    }

    /// Host-side captured-graph bookkeeping for the slot this executor is
    /// currently driving ([`Self::graph_slot`]). All capture/replay/seeding
    /// state reads and writes go through here so the `Primary` (M=1) and
    /// `Verify` (M=k+1) graphs never share a signature/schedule.
    #[inline]
    pub(super) fn cap(&self) -> &SlotCaptureState {
        &self.slot_capture[self.graph_slot.index()]
    }

    /// Mutable accessor mirroring [`Self::cap`].
    #[inline]
    pub(super) fn cap_mut(&mut self) -> &mut SlotCaptureState {
        let slot = self.graph_slot.index();
        &mut self.slot_capture[slot]
    }

    /// Retarget this executor's captured-graph slot. With per-slot host capture
    /// state ([`Self::slot_capture`]) each slot owns an independent
    /// signature/schedule/warm-shape set, so switching slots must NOT reset the
    /// other slot's installed graph — that is precisely what let the M=1 decode
    /// clobber the M=k+1 verify graph and pinned verify replays at 0. The switch
    /// is now a pure retarget: the EP keeps one [`CudaGraphLifecycle`] per slot,
    /// and this executor simply points its capture/replay calls at `slot`.
    /// Idempotent when `slot` already matches. Used to route the main executor's
    /// verify forward to [`DeviceGraphSlot::Verify`] while M=1 decode keeps
    /// [`DeviceGraphSlot::Primary`].
    pub(crate) fn set_graph_slot(&mut self, slot: DeviceGraphSlot) -> Result<()> {
        self.graph_slot = slot;
        Ok(())
    }

    /// Structured segment-boundary reasons from the most recent capture: one
    /// entry per non-capturable seam node the CUDA EP ran eagerly between
    /// captured segments. Empty for a whole-subgraph (single-graph) capture.
    pub(crate) fn capture_segmentation(&self) -> &[CaptureDecline] {
        &self.cap().capture_segmentation
    }

    /// Number of captured device-graph segments installed by the most recent
    /// capture (1 for a whole-subgraph capture, >=2 when seams split it).
    pub(crate) fn captured_segment_count(&self) -> usize {
        self.cap()
            .capture_schedule
            .as_ref()
            .map(CaptureSchedule::captured_segments)
            .unwrap_or(0)
    }

    pub(crate) fn check_device_capture_error(&self) -> Result<u32> {
        self.ep.sync()?;
        match self.pending_device_validation {
            Some(token) => self
                .ep
                .consume_device_validation_error(
                    self.validation_registration
                        .as_ref()
                        .expect("executor validation registration exists until Drop"),
                    token,
                )
                .map_err(SessionError::from),
            None => Ok(0),
        }
    }

    pub(crate) fn device_allocation_counts(&self) -> Option<DeviceAllocationCounts> {
        self.ep
            .device_allocation_counts()
            .map(|(allocations, frees)| DeviceAllocationCounts { allocations, frees })
    }

    pub(crate) fn raw_device_allocation_site_stats(
        &self,
    ) -> Vec<onnx_runtime_ep_api::RawDeviceAllocationSiteStats> {
        self.ep.raw_device_allocation_site_stats()
    }

    /// Place any long-lived device memory the provider holds under `governor`.
    /// Whether the memory this executor's provider hands out commits
    /// physically as it is used. See
    /// [`DeviceAllocator::commits_on_demand`](onnx_runtime_memory_governor::DeviceAllocator::commits_on_demand).
    pub(crate) fn commits_on_demand(&self) -> bool {
        self.ep.commits_on_demand()
    }

    pub(crate) fn adopt_memory_governor(
        &self,
        governor: &dyn onnx_runtime_memory_governor::MemoryGovernor,
        tier: onnx_runtime_memory_governor::Tier,
        holder: onnx_runtime_memory_governor::HolderId,
    ) -> onnx_runtime_ep_api::Result<u64> {
        self.ep.adopt_memory_governor(governor, tier, holder)
    }

    pub(crate) fn set_weight_residency_budget(
        &self,
        budget_bytes: u64,
    ) -> onnx_runtime_ep_api::Result<Option<u64>> {
        self.ep.set_weight_residency_budget(budget_bytes)
    }

    pub(crate) fn max_lazy_weight_working_set_bytes(&self) -> u64 {
        self.plan
            .iter()
            .map(|node| {
                node.lazy_weight_inputs
                    .iter()
                    .filter_map(|value_id| self.weight_handles.get(value_id))
                    .filter_map(|handle| handle.as_lazy())
                    .map(|weight| weight.region_bytes_len() as u64)
                    .sum()
            })
            .max()
            .unwrap_or(0)
    }

    pub(super) fn binding_signature(bindings: &[DeviceIoBinding]) -> Vec<DeviceBindingSignature> {
        bindings
            .iter()
            .map(|binding| DeviceBindingSignature {
                input_name: binding.input_name().to_string(),
                binds_input: binding.binds_input(),
                output_name: binding.output_name().map(str::to_string),
                dtype: binding.dtype,
                physical_shape: binding.physical_shape().to_vec(),
                logical_shape: binding.logical_shape().to_vec(),
                exposes_logical_input_shape: binding.exposes_logical_input_shape(),
                mask_decode_freeze_safe: binding.mask_decode_freeze_safe(),
                fixed_physical_strides: binding.fixed_physical_strides(),
                device_ptr: binding.device_ptr() as usize,
            })
            .collect()
    }

    fn bindings_match_graph_signature(&self, bindings: &[DeviceIoBinding]) -> bool {
        self.cap()
            .device_graph_signature
            .as_deref()
            .is_some_and(|signature| {
                signature.len() == bindings.len()
                    && signature.iter().zip(bindings).all(|(expected, binding)| {
                        expected.input_name == binding.input_name()
                            && expected.binds_input == binding.binds_input()
                            && expected.output_name.as_deref() == binding.output_name()
                            && expected.dtype == binding.dtype
                            && expected.physical_shape == binding.physical_shape()
                            && (expected.fixed_physical_strides
                                || self.segmented_binding_keeps_dynamic_input_eager(binding)
                                || expected.logical_shape == binding.logical_shape())
                            && expected.exposes_logical_input_shape
                                == binding.exposes_logical_input_shape()
                            && expected.mask_decode_freeze_safe == binding.mask_decode_freeze_safe()
                            && expected.fixed_physical_strides == binding.fixed_physical_strides()
                            && expected.device_ptr == binding.device_ptr() as usize
                    })
            })
    }

    fn segmented_binding_keeps_dynamic_input_eager(&self, binding: &DeviceIoBinding) -> bool {
        if !binding.has_dynamic_logical_input_shape() || !binding.binds_input() {
            return false;
        }
        let Some(schedule) = self.cap().capture_schedule.as_ref() else {
            return false;
        };
        if schedule.is_single_graph() {
            return false;
        }
        let Some(input) = self.input_index.get(binding.input_name()) else {
            return false;
        };
        !schedule.segments.iter().any(|segment| {
            segment.captured
                && (segment.start..segment.end).any(|pi| {
                    self.plan[pi]
                        .inputs
                        .iter()
                        .any(|planned_input| planned_input.as_ref() == Some(input))
                })
        })
    }

    pub(super) fn prepare_external_bindings(
        &mut self,
        bindings: &mut [DeviceIoBinding],
    ) -> Result<ExternalBindings> {
        let external = std::mem::take(&mut self.scratch_external_bindings);
        self.refill_external_bindings(external, bindings, false)
    }

    /// As [`Self::prepare_external_bindings`], but when `plan_capacity` is set the
    /// input value shapes bind a growing/logical-exposing input at its *physical
    /// capacity* rather than its current logical prefix.
    ///
    /// This is used only by prepare-only workspace planning
    /// ([`Self::prepare_with_device_bindings`]), which never executes a node: a
    /// governed kernel workspace must be reserved for the *maximum* extent an
    /// input reaches across the session, not the logical prefix bound at
    /// preparation time. A binding that exposes its logical prefix
    /// ([`DeviceIoBinding::exposes_logical_input_shape`]) — e.g. a growing-KV
    /// decode cache that cannot be frozen to capacity — otherwise binds its
    /// sequence symbol to the current (0/prefill) length, so the reserved
    /// SessionPersistent decode workspace is sized far below what steady-state
    /// decode consumes and later steps fault on the workspace invariant (#1179).
    /// Sizing against physical capacity over-reserves (reserve ≥ consume), which
    /// is exactly correct for a reservation; execution still binds the logical
    /// prefix and fits under it.
    pub(super) fn prepare_external_bindings_mode(
        &self,
        bindings: &mut [DeviceIoBinding],
        plan_capacity: bool,
    ) -> Result<ExternalBindings> {
        self.refill_external_bindings(ExternalBindings::default(), bindings, plan_capacity)
    }

    fn refill_external_bindings(
        &self,
        mut external: ExternalBindings,
        bindings: &mut [DeviceIoBinding],
        plan_capacity: bool,
    ) -> Result<ExternalBindings> {
        external.inputs.retain(|vid, _| {
            self.graph.value(*vid).name.as_deref().is_some_and(|name| {
                bindings
                    .iter()
                    .any(|binding| binding.binds_input() && binding.input_name() == name)
            })
        });
        external.outputs.retain(|vid, _| {
            self.graph.value(*vid).name.as_deref().is_some_and(|name| {
                bindings
                    .iter()
                    .any(|binding| binding.output_name() == Some(name))
            })
        });
        for index in 0..bindings.len() {
            for prior in &bindings[..index] {
                if bindings[index].binds_input()
                    && prior.binds_input()
                    && bindings[index].input_name() == prior.input_name()
                {
                    return Err(SessionError::Internal(format!(
                        "duplicate device input binding '{}'",
                        bindings[index].input_name()
                    )));
                }
                if let Some(output) = bindings[index].output_name()
                    && prior.output_name() == Some(output)
                {
                    return Err(SessionError::Internal(format!(
                        "duplicate device output binding '{output}'"
                    )));
                }
            }
        }
        for binding in bindings.iter_mut() {
            let ptr = binding.buffer_mut().as_mut_ptr();
            let input_name = binding.input_name();
            let bind_input = binding.binds_input();
            let output_name = binding.output_name();
            let dtype = binding.dtype;
            let len = binding.buffer().len();
            let alignment = binding.buffer().alignment();
            let device = binding.buffer().device();
            if device != self.ep.device_id() {
                return Err(SessionError::Internal(format!(
                    "device binding '{input_name}' is on {device:?}, session is on {:?}",
                    self.ep.device_id()
                )));
            }
            let physical_shape = binding.physical_shape();
            let fixed_physical_strides = binding.fixed_physical_strides();
            let required = required_binding_bytes(dtype, physical_shape, input_name)?;
            if required > len {
                return Err(SessionError::Internal(format!(
                    "device binding '{input_name}' needs {required} bytes for {physical_shape:?}, allocation has {len}"
                )));
            }
            if bind_input {
                let input_vid = *self.input_index.get(input_name).ok_or_else(|| {
                    SessionError::InputNotFound {
                        name: input_name.to_string(),
                    }
                })?;
                let value = external
                    .inputs
                    .entry(input_vid)
                    .or_insert_with(|| ExternalValue {
                        dtype,
                        shape: Vec::new(),
                        accepts_subshape: false,
                        strides: None,
                        fixed_stride_shape: None,
                        ptr: ptr as usize,
                        len,
                        alignment,
                        device,
                    });
                value.dtype = dtype;
                value.shape.clear();
                value.shape.extend_from_slice(if plan_capacity {
                    binding.physical_shape()
                } else {
                    binding.kernel_input_shape()
                });
                value.accepts_subshape = false;
                if fixed_physical_strides {
                    dispatch::refill_contiguous_strides(
                        value.strides.get_or_insert_default(),
                        physical_shape,
                    );
                    let stable_shape = value.fixed_stride_shape.get_or_insert_default();
                    stable_shape.clear();
                    stable_shape.extend_from_slice(physical_shape);
                } else {
                    value.strides = None;
                    value.fixed_stride_shape = None;
                }
                value.ptr = ptr as usize;
                value.len = len;
                value.alignment = alignment;
                value.device = device;
            }
            if let Some(output_name) = output_name {
                let output_vid = self
                    .graph
                    .outputs
                    .iter()
                    .copied()
                    .find(|&vid| self.graph.value(vid).name.as_deref() == Some(output_name))
                    .ok_or_else(|| {
                        SessionError::Internal(format!(
                            "device binding output not found: {output_name}"
                        ))
                    })?;
                if self.sequence_values.contains(&output_vid) {
                    return Err(SessionError::SequenceOp {
                        op: "<graph output binding>".to_string(),
                        reason: format!(
                            "graph output '{output_name}' is a Sequence value and cannot be bound to tensor device storage"
                        ),
                    });
                }
                if self.value_dtypes[&output_vid] != dtype {
                    return Err(SessionError::DtypeMismatch {
                        name: output_name.to_string(),
                        expected: format!("{:?}", self.value_dtypes[&output_vid]),
                        got: format!("{dtype:?}"),
                    });
                }
                let value = external
                    .outputs
                    .entry(output_vid)
                    .or_insert_with(|| ExternalValue {
                        dtype,
                        shape: Vec::new(),
                        accepts_subshape: false,
                        strides: None,
                        fixed_stride_shape: None,
                        ptr: ptr as usize,
                        len,
                        alignment,
                        device,
                    });
                value.dtype = dtype;
                value.shape.clear();
                value.shape.extend_from_slice(binding.physical_shape());
                value.accepts_subshape =
                    bind_input && binding.logical_shape() != binding.physical_shape();
                if fixed_physical_strides {
                    dispatch::refill_contiguous_strides(
                        value.strides.get_or_insert_default(),
                        physical_shape,
                    );
                    let stable_shape = value.fixed_stride_shape.get_or_insert_default();
                    stable_shape.clear();
                    stable_shape.extend_from_slice(physical_shape);
                } else {
                    value.strides = None;
                    value.fixed_stride_shape = None;
                }
                value.ptr = ptr as usize;
                value.len = len;
                value.alignment = alignment;
                value.device = device;
            }
        }
        Ok(external)
    }
}

pub(super) fn required_binding_bytes(
    dtype: DataType,
    physical_shape: &[usize],
    input_name: &str,
) -> Result<usize> {
    onnx_runtime_ir::checked_expected_bytes(dtype, physical_shape).ok_or_else(|| {
        SessionError::ShapeOverflow {
            value: format!("device binding '{input_name}'"),
            dims: physical_shape.to_vec(),
        }
    })
}

#[cfg(test)]
mod planned_workspace_node_tests {
    use super::*;
    use onnx_runtime_ir::{Node, NodeId};

    fn node(domain: &str, op_type: &str) -> Node {
        let mut node = Node::new(NodeId(0), op_type, Vec::new(), Vec::new());
        node.domain = domain.into();
        node
    }

    #[test]
    fn centralized_predicate_covers_the_governed_gemm_family() {
        for (domain, op_type) in [
            ("", "MatMul"),
            ("", "Gemm"),
            ("", "Attention"),
            ("", "Conv"),
            ("", "ReduceSum"),
            ("", "ReduceMean"),
            ("com.microsoft", "MatMulNBits"),
            ("com.microsoft", "FusedMatMulBias"),
            ("com.microsoft", "FusedGemm"),
            ("com.microsoft", "Attention"),
            ("com.microsoft", "GroupQueryAttention"),
        ] {
            assert!(
                is_planned_workspace_node(&node(domain, op_type)),
                "{domain}::{op_type} must be prepared before admission"
            );
        }
        assert!(!is_planned_workspace_node(&node("", "Add")));
    }

    #[test]
    fn same_lifetime_gemm_requirements_merge_as_one_peak_not_a_sum() {
        let requirement = WorkspaceRequirement {
            bytes: 32 * 1024 * 1024,
            alignment: 256,
            lifetime: WorkspaceLifetime::SessionPersistent,
            role: onnx_runtime_memory_governor::MemoryRole::Workspace { step_scoped: false },
        };
        let mut peak = WorkspaceRequirement::NONE;
        for _ in 0..4 {
            merge_workspace_peak(&mut peak, requirement);
        }
        assert_eq!(peak.bytes, requirement.bytes);
    }
}
