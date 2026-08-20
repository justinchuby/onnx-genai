use super::*;

impl ChildExecutor {
    /// Create the reusable wrapper for a loaded subgraph body.
    ///
    /// `body.inputs` and `body.outputs` are the loader-preserved ordered formal
    /// signature. Producer-less named values that are neither formals nor local
    /// initializers are lexical captures and are bound from `outer_scope`.
    pub(crate) fn new(
        name: impl Into<String>,
        body: Graph,
        inherited_opsets: HashMap<String, u64>,
        weights: Arc<WeightStore>,
        ep: Arc<dyn ExecutionProvider>,
    ) -> Result<Self> {
        let name = name.into();
        let formal_names = body
            .inputs
            .iter()
            .map(|&vid| {
                body.value(vid).name.clone().ok_or_else(|| {
                    SessionError::Internal(format!(
                        "subgraph '{name}' has an unnamed formal input value#{}",
                        vid.0
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let formal_set: HashSet<ValueId> = body.inputs.iter().copied().collect();
        let mut capture_names = body
            .values
            .iter()
            .filter_map(|(vid, value)| {
                (value.producer.is_none()
                    && !formal_set.contains(&vid)
                    && !body.initializers.contains_key(&vid))
                .then(|| value.name.clone())
                .flatten()
            })
            .collect::<Vec<_>>();
        capture_names.sort();
        let input_names = formal_names
            .iter()
            .chain(capture_names.iter())
            .cloned()
            .collect();

        Ok(Self {
            name,
            template: body,
            inherited_opsets,
            weights,
            ep,
            formal_names,
            capture_names,
            input_names,
            compiled: Vec::new(),
            builds: 0,
            runs: 0,
            trace: TraceContext::noop(),
            release_dead_values: false,
        })
    }

    pub(crate) fn stats(&self) -> ChildExecutorStats {
        ChildExecutorStats {
            builds: self.builds,
            runs: self.runs,
        }
    }

    /// Attach the shared trace context, propagating it to every already-compiled
    /// child plan and to plans compiled later.
    pub(crate) fn set_trace_context(&mut self, trace: TraceContext) {
        for plan in &mut self.compiled {
            plan.exec.set_trace_context(trace.clone());
        }
        self.trace = trace;
    }

    /// Mirror of [`Self::set_trace_context`] for dead-value release: applied to
    /// already-compiled child plans and remembered for plans compiled later.
    pub(crate) fn set_release_dead_values(&mut self, enabled: bool) {
        for plan in &mut self.compiled {
            plan.exec.set_release_dead_values(enabled);
        }
        self.release_dead_values = enabled;
    }

    pub(super) fn compile(&self, externals: &[&Tensor]) -> Result<CompiledChildPlan> {
        let mut graph = self.template.clone();
        // GraphProto has no opset table: nested graphs inherit the model-level
        // imports from their enclosing graph.
        graph.opset_imports = self.inherited_opsets.clone();

        let body_names = graph
            .values
            .iter()
            .filter_map(|(vid, value)| value.name.clone().map(|name| (name, vid)))
            .collect::<HashMap<_, _>>();

        // Direct captures become required graph inputs. Local inline
        // initializers stay in `graph.initializers`, preserving their scope.
        for name in &self.capture_names {
            let vid = *body_names.get(name).ok_or_else(|| {
                SessionError::Internal(format!(
                    "subgraph '{}' lost captured value '{name}'",
                    self.name
                ))
            })?;
            if !graph.inputs.contains(&vid) {
                graph.add_input(vid);
            }
        }

        for (name, tensor) in self.input_names.iter().zip(externals) {
            let vid = *body_names.get(name).ok_or_else(|| {
                SessionError::Internal(format!(
                    "subgraph '{}' is missing bound input '{name}'",
                    self.name
                ))
            })?;
            let value = graph.value_mut(vid);
            value.dtype = tensor.dtype;
            value.shape = tensor.shape.iter().map(|&dim| Dim::Static(dim)).collect();
        }

        // Seeded formal/capture shapes let inference resolve the body once.
        // Truly data-dependent outputs remain on Executor's JIT shape path.
        let registry = InferenceRegistry::default_registry();
        registry.infer_graph(&mut graph, &self.inherited_opsets, MergePolicy::Permissive)?;

        Ok(CompiledChildPlan {
            exec: {
                let mut exec = Executor::build(graph, self.weights.clone(), self.ep.clone())?;
                exec.set_trace_context(self.trace.clone());
                exec.set_release_dead_values(self.release_dead_values);
                exec
            },
            signature: externals
                .iter()
                .map(|tensor| ChildInputSignature {
                    dtype: tensor.dtype,
                    shape: tensor.shape.clone(),
                })
                .collect(),
        })
    }

    /// Execute the body with formal inputs in declared order and lexical values
    /// supplied by name. A cached plan is reused for matching dtype/shapes.
    #[cfg(test)]
    pub(crate) fn run(
        &mut self,
        formal_inputs: &[&Tensor],
        outer_scope: &HashMap<String, Tensor>,
    ) -> Result<Vec<Tensor>> {
        self.run_with_workspace(formal_inputs, outer_scope, None)
    }

    pub(crate) fn run_with_workspace(
        &mut self,
        formal_inputs: &[&Tensor],
        outer_scope: &HashMap<String, Tensor>,
        workspace: Option<WorkspaceView>,
    ) -> Result<Vec<Tensor>> {
        if self.formal_names.len() != formal_inputs.len() {
            return Err(SessionError::Internal(format!(
                "subgraph '{}' expects {} formal input(s) but {} were supplied",
                self.name,
                self.formal_names.len(),
                formal_inputs.len()
            )));
        }

        let mut externals = Vec::with_capacity(formal_inputs.len() + self.capture_names.len());
        externals.extend_from_slice(formal_inputs);
        for name in &self.capture_names {
            externals.push(
                outer_scope
                    .get(name)
                    .ok_or_else(|| missing_capture_error(&self.name, name))?,
            );
        }

        let signature = externals
            .iter()
            .map(|tensor| ChildInputSignature {
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
            })
            .collect::<Vec<_>>();
        let cache_index = if let Some(index) = self
            .compiled
            .iter()
            .position(|compiled| compiled.signature == signature)
        {
            let compiled = self.compiled.remove(index);
            self.compiled.push(compiled);
            self.compiled.len() - 1
        } else {
            let compiled = self.compile(&externals)?;
            if self.compiled.len() == CHILD_EXECUTOR_CACHE_CAPACITY {
                self.compiled.remove(0);
            }
            self.compiled.push(compiled);
            self.builds += 1;
            self.compiled.len() - 1
        };

        self.runs += 1;
        self.compiled[cache_index].exec.inherited_workspace =
            workspace.map(|view| (view.ptr().0 as usize, view.bytes()));
        self.compiled[cache_index]
            .exec
            .workspace_preparation_required = workspace.is_some();
        let inputs = self
            .input_names
            .iter()
            .map(String::as_str)
            .zip(externals)
            .collect::<Vec<_>>();
        self.compiled[cache_index]
            .exec
            .run_scoped(&inputs, outer_scope, &ExternalBindings::default())?
            .into_iter()
            .map(|output| {
                let output = output.ok_or_else(|| {
                    SessionError::Internal(format!(
                        "subgraph '{}' unexpectedly suppressed an output",
                        self.name
                    ))
                })?;
                match output {
                    SessionOutput::Tensor(tensor) => Ok(tensor),
                    SessionOutput::Sequence(_) => Err(SessionError::SequenceOp {
                        op: "<control-flow output>".to_string(),
                        reason: format!(
                            "subgraph '{}' produced a Sequence output where this control-flow path requires a tensor",
                            self.name
                        ),
                    }),
                }
            })
            .collect()
    }
}

// === Control-flow (subgraph-executing) ops: If / Loop / Scan ===
//
// These are handled at the executor level rather than as leaf kernels because
// they must recursively execute a nested ONNX [`Graph`] with the enclosing
// scope bound — something a `Kernel` (which sees only tensor views, never the
// session/graph context) cannot do. Each body is compiled to a child
// [`Executor`] once and **reused across iterations** (see [`ChildExecutor`]).
impl Executor {
    /// Materialize value `vid`'s current bytes into an owned host [`Tensor`],
    /// using its resolved concrete shape and recorded dtype.
    pub(super) fn value_tensor(
        &self,
        vid: ValueId,
        resolved: &HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> Result<Tensor> {
        let dtype = self.value_dtypes[&vid];
        let shape = resolved.get(&vid).cloned().ok_or_else(|| {
            let name = self
                .graph
                .try_value(vid)
                .and_then(|v| v.name.clone())
                .unwrap_or_else(|| format!("value#{}", vid.0));
            SessionError::UnresolvedShape {
                value: name,
                op: "<control-flow input>".to_string(),
            }
        })?;
        if let Some(value) = external
            .inputs
            .get(&vid)
            .or_else(|| external.outputs.get(&vid))
        {
            let n = checked_storage_bytes(
                dtype,
                checked_numel(&shape, || format!("value#{}", vid.0))?,
                || format!("value#{}", vid.0),
                &shape,
            )?;
            if value.dtype != dtype || value.len < n {
                return Err(SessionError::Internal(format!(
                    "external binding for value#{} cannot materialize {:?} {:?} (binding: {:?}, {} bytes)",
                    vid.0, dtype, shape, value.dtype, value.len
                )));
            }
            let buffer = value.readable_buffer()?;
            let mut bytes = vec![0_u8; n];
            self.ep.copy_to_host(&buffer, &mut bytes)?;
            return Tensor::from_raw(dtype, shape, &bytes);
        }
        // A view value owns no buffer; materialize its strided bytes contiguous.
        let bytes = self.contiguous_bytes(vid, &shape, dtype)?;
        Tensor::from_raw(dtype, shape, &bytes)
    }

    /// The buffer-owning (root) value backing `vid`: `vid` itself if it owns a
    /// buffer, or the `source` recorded in its view metadata (always a root,
    /// since views are flattened at creation).
    pub(super) fn root_of(&self, vid: ValueId) -> ValueId {
        match self.views.get(&vid) {
            Some(v) => v.source,
            None => vid,
        }
    }

    /// Zero-copy hand-off of a top-level graph output: move the produced host
    /// buffer straight into the returned tensor instead of copying it to host
    /// and re-allocating it in [`Tensor::from_raw`]. This eliminates two full
    /// per-output memcpys on the decode hot path (the growing KV-cache present
    /// outputs re-materialized every step) while being numerically identical —
    /// the tensor keeps the exact bytes the kernel wrote.
    ///
    /// Returns `None` (caller falls back to the copy path) unless every safety
    /// precondition holds: the value is an owned, host-resident, exactly-sized
    /// producer output that is not a view/sequence element, not pinned as a live
    /// view source, not shared, and not listed as a graph output more than once.
    /// Moving the buffer out forfeits this value's cross-run allocation reuse, so
    /// its `buffer_shapes` entry is cleared to force a fresh allocation next run.
    pub(super) fn try_move_host_output(
        &mut self,
        vid: ValueId,
        shape: &[usize],
        dtype: DataType,
    ) -> Result<Option<Tensor>> {
        // Values the copy path materializes specially (strided gather, shared
        // sequence element, in-place share, or a pinned live view source) must
        // not have their backing buffer stolen.
        if self.views.contains_key(&vid)
            || self.seq_elem_values.contains_key(&vid)
            || self.shared_buffers.contains_key(&vid)
            || self.pinned.contains(&vid)
        {
            return Ok(None);
        }
        // Only a produced value owns a writable buffer. A producer-less output
        // (initializer or graph-input passthrough) may alias read-only mmap or
        // foreign/borrowed memory that a tensor must never free.
        if self
            .graph
            .try_value(vid)
            .is_none_or(|value| value.producer.is_none())
        {
            return Ok(None);
        }
        // A value produced by a memoized loop-invariant `If` is served on later
        // steps directly from its resident buffer (the branch is skipped, see
        // `exec_if`). Moving that buffer out would leave the next memoized skip
        // handing back freed/reallocated memory, so fall back to the copy path
        // and keep the produced buffer resident for reuse.
        if let Some(producer) = self.graph.try_value(vid).and_then(|value| value.producer)
            && self.if_last_predicate.contains_key(&producer)
        {
            return Ok(None);
        }
        // A value listed as a graph output more than once would be taken twice.
        if self.graph.outputs.iter().filter(|&&o| o == vid).count() != 1 {
            return Ok(None);
        }
        let value_name = || format!("value#{}", vid.0);
        let numel = checked_numel(shape, value_name)?;
        let n = checked_storage_bytes(dtype, numel, value_name, shape)?;
        let movable = self.buffers.get(&vid).is_some_and(|buf| {
            buf.device().is_host_accessible() && !buf.is_borrowed() && buf.len() == n
        });
        if !movable {
            return Ok(None);
        }
        let buffer = self
            .buffers
            .remove(&vid)
            .expect("buffer presence checked above");
        // The buffer now belongs to the tensor; force a fresh allocation on the
        // next run instead of the reuse fast path (which assumes it is present).
        self.buffer_shapes.remove(&vid);
        Ok(Some(Tensor::from_owned_buffer(
            self.ep.clone(),
            dtype,
            shape.to_vec(),
            buffer,
        )))
    }

    /// Contiguous row-major bytes of `vid` for `shape`/`dtype`, materializing a
    /// view (strided gather over its source buffer) or truncating an owned
    /// buffer to its logical size. This is the single materialization seam used
    /// by the graph-output boundary and control-flow scope capture.
    /// Materialize graph output `vid` straight into an owned tensor.
    ///
    /// The general path builds the bytes in a `Vec` and hands them to
    /// `Tensor::from_raw`, which allocates the same bytes a second time and
    /// memcpys between them. For a **view** output on a host-accessible device
    /// -- a `Transpose`, `Reshape`, `Slice` or `Unsqueeze` whose result is a
    /// graph output -- that second allocate-and-copy is pure overhead on top of
    /// a gather that already writes every byte exactly once, so gather directly
    /// into the tensor's allocation instead.
    ///
    /// Returns the tensor and the host bytes it materialized, so the caller's
    /// attribution counter stays exact.
    pub(super) fn contiguous_output_tensor(
        &self,
        vid: ValueId,
        shape: &[usize],
        dtype: DataType,
    ) -> Result<(Tensor, usize)> {
        let esize = dtype.byte_size();
        if esize > 0
            && !self.seq_elem_values.contains_key(&vid)
            && let Some(view) = self.views.get(&vid)
            && let Some(buf) = self.buffers.get(&view.source)
            && buf.device().is_host_accessible()
        {
            let value_name = || format!("value#{}", vid.0);
            let numel = checked_numel(shape, value_name)?;
            let n = checked_storage_bytes(dtype, numel, value_name, shape)?;
            // SAFETY: `buf` is a live device buffer owned by this executor's EP
            // on a host-accessible device, so its `len()` bytes are readable
            // from `as_ptr()` (ep-api safety invariant #1), and `u8` has no
            // invalid bit patterns. The gather only reads, and the destination
            // is a freshly allocated tensor that cannot alias it.
            let src = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len()) };
            let tensor = Tensor::from_host_fill(dtype, shape.to_vec(), |dst| {
                gather_view_into(
                    dst,
                    src,
                    &view.shape,
                    &view.strides,
                    view.byte_offset,
                    esize,
                );
            })?;
            return Ok((tensor, n));
        }
        let bytes = self.contiguous_bytes(vid, shape, dtype)?;
        let n = bytes.len();
        Ok((Tensor::from_raw(dtype, shape.to_vec(), &bytes)?, n))
    }

    pub(super) fn contiguous_bytes(
        &self,
        vid: ValueId,
        shape: &[usize],
        dtype: DataType,
    ) -> Result<Vec<u8>> {
        let value_name = || {
            self.graph
                .try_value(vid)
                .and_then(|value| value.name.clone())
                .unwrap_or_else(|| format!("value#{}", vid.0))
        };
        let numel = checked_numel(shape, value_name)?;
        let n = checked_storage_bytes(dtype, numel, value_name, shape)?;
        // A tensor value backed by a shared sequence element (SequenceAt output)
        // owns no buffer; its bytes are the element's contiguous bytes. This is
        // the one materialization point where they are copied out (the boundary
        // back into owned tensors); the compute path reads them zero-copy.
        if let Some(elem) = self.seq_elem_values.get(&vid) {
            let bytes = elem.contiguous_bytes().map_err(SessionError::from)?;
            return Ok(bytes[..n.min(bytes.len())].to_vec());
        }
        if let Some(view) = self.views.get(&vid) {
            let buf = self.buffers.get(&view.source).ok_or_else(|| {
                SessionError::Internal(format!(
                    "view value#{} aliases missing source buffer value#{}",
                    vid.0, view.source.0
                ))
            })?;
            let esize = dtype.byte_size();
            if esize == 0 {
                // Sub-byte views are never created (Slice falls back to copy),
                // so reaching here is an internal invariant violation.
                return Err(SessionError::Internal(format!(
                    "cannot materialize sub-byte view value#{}",
                    vid.0
                )));
            }
            if buf.device().is_host_accessible() {
                // The source is already host memory, so staging a full copy of
                // it just to read it doubles the traffic of every view graph
                // output -- and the staging buffer is `vec![0u8; ..]`, so the
                // kernel zeroes pages that are overwritten immediately.
                //
                // SAFETY: `buf` is a live device buffer owned by this executor's
                // EP on a host-accessible device, so its `len()` bytes are
                // readable from `as_ptr()` (ep-api safety invariant #1), and
                // `u8` has no invalid bit patterns. `gather_view` only reads.
                let src =
                    unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len()) };
                return Ok(gather_view(
                    src,
                    &view.shape,
                    &view.strides,
                    view.byte_offset,
                    esize,
                ));
            }
            let mut host = vec![0u8; buf.len()];
            self.ep.copy_to_host(buf, &mut host)?;
            Ok(gather_view(
                &host,
                &view.shape,
                &view.strides,
                view.byte_offset,
                esize,
            ))
        } else {
            let buf = self
                .buffers
                .get(&vid)
                .ok_or_else(|| SessionError::Internal(format!("{} not produced", value_name())))?;
            let mut host = vec![0u8; n];
            self.ep.copy_to_host(buf, &mut host)?;
            Ok(host)
        }
    }

    /// Store a control-flow op's produced output `tensor` into this graph's
    /// output value `vid`: (re)size the backing buffer, copy the bytes, and
    /// record the runtime dtype/shape so the caller (and the final output
    /// collection) reads them back correctly. Control-flow output shapes are
    /// data-dependent (the loader never inferred inside the body), so they are
    /// resolved here, exactly as the JIT data-dependent path does for kernels.
    pub(super) fn store_output_tensor(
        &mut self,
        vid: ValueId,
        tensor: &Tensor,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> Result<()> {
        self.store_output_bytes(
            vid,
            tensor.dtype,
            tensor.shape.clone(),
            tensor.as_bytes(),
            resolved,
            external,
        )
    }

    pub(super) fn store_output_bytes(
        &mut self,
        vid: ValueId,
        dtype: DataType,
        dims: Vec<usize>,
        bytes: &[u8],
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> Result<()> {
        self.store_raw_tensor_output(vid, dtype, dims, bytes, resolved, external)
    }

    /// Prepare one selected control-flow subgraph and materialize only the free
    /// variables that body actually captures. This avoids copying every named
    /// value in the enclosing graph and, for Loop/Scan, keeps captures stable
    /// across all iterations.
    pub(super) fn prepare_subgraph(
        &self,
        node_id: NodeId,
        attr_key: &str,
        resolved: &HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
        external: &ExternalBindings,
    ) -> Result<PreparedSubgraph> {
        let key = (node_id, attr_key.to_string());
        let body = self.graph.subgraphs.get(&key).ok_or_else(|| {
            SessionError::Internal(format!(
                "control-flow node #{} references missing subgraph '{attr_key}'",
                node_id.0
            ))
        })?;

        let mut scope_names = required_outer_names(body).into_iter().collect::<Vec<_>>();
        scope_names.sort();
        let mut captures = HashMap::with_capacity(scope_names.len());
        for name in scope_names {
            let tensor = if let Some(&vid) = self.name_index.get(&name) {
                let materialized = self.buffers.contains_key(&vid)
                    || self.views.contains_key(&vid)
                    || self.seq_elem_values.contains_key(&vid)
                    || external.inputs.contains_key(&vid)
                    || external.outputs.contains_key(&vid);
                if resolved.contains_key(&vid) && materialized {
                    self.value_tensor(vid, resolved, external)?
                } else {
                    outer_scope
                        .get(&name)
                        .ok_or_else(|| missing_capture_error(attr_key, &name))?
                        .try_clone()?
                }
            } else {
                outer_scope
                    .get(&name)
                    .ok_or_else(|| missing_capture_error(attr_key, &name))?
                    .try_clone()?
            };
            captures.insert(name, tensor);
        }

        Ok(PreparedSubgraph { key, captures })
    }

    /// Run a prepared control-flow body with changing formal inputs. Captures and
    /// signature metadata are reused; only a concrete shape change rebuilds the
    /// child executor.
    pub(super) fn run_subgraph(
        &mut self,
        prepared: &PreparedSubgraph,
        formal_inputs: &[&Tensor],
    ) -> Result<Vec<Tensor>> {
        if !self.subgraph_execs.contains_key(&prepared.key) {
            let body = self
                .graph
                .subgraphs
                .get(&prepared.key)
                .cloned()
                .ok_or_else(|| {
                    SessionError::Internal(format!(
                        "control-flow node #{} has no registered subgraph '{}'",
                        prepared.key.0.0, prepared.key.1
                    ))
                })?;
            let mut child = ChildExecutor::new(
                format!("node#{}/{}", prepared.key.0.0, prepared.key.1),
                body,
                self.graph.opset_imports.clone(),
                self.weights.clone(),
                self.ep.clone(),
            )?;
            child.set_trace_context(self.trace.clone());
            child.set_release_dead_values(self.release_dead_values_enabled);
            self.subgraph_execs.insert(prepared.key.clone(), child);
        }

        let workspace = self
            .persistent_workspace
            .as_mut()
            .or(self.step_workspace.as_mut())
            .map(|prepared| {
                WorkspaceView::new(DevicePtrMut(prepared.buffer.as_mut_ptr()), prepared.bytes)
            })
            .or_else(|| {
                self.inherited_workspace
                    .map(|(ptr, bytes)| WorkspaceView::new(DevicePtrMut(ptr as *mut _), bytes))
            });
        let child = self
            .subgraph_execs
            .get_mut(&prepared.key)
            .expect("child present");
        let before = child.stats();
        let result = child.run_with_workspace(formal_inputs, &prepared.captures, workspace);
        let after = child.stats();
        self.control_flow_stats.subgraph_builds += after.builds - before.builds;
        self.control_flow_stats.subgraph_runs += after.runs - before.runs;
        result
    }

    /// Dispatch a control-flow plan node to its op-specific handler.
    pub(super) fn exec_control_flow(
        &mut self,
        pi: usize,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
        external: &ExternalBindings,
    ) -> Result<()> {
        let node = self.graph.node(self.plan[pi].node_id).clone();
        match node.op_type.as_str() {
            "If" => self.exec_if(&node, resolved, outer_scope, external),
            "Loop" => self.exec_loop(&node, resolved, outer_scope, external),
            "Scan" => self.exec_scan(&node, resolved, outer_scope, external),
            other => Err(SessionError::Internal(format!(
                "exec_control_flow reached non-control-flow op {other:?}"
            ))),
        }
    }

    /// ONNX `If`: read the scalar `cond`, execute exactly one branch subgraph
    /// (0 formal inputs), and route the branch's outputs to `If`'s outputs.
    pub(super) fn exec_if(
        &mut self,
        node: &Node,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
        external: &ExternalBindings,
    ) -> Result<()> {
        {
            let then_branch = self
                .graph
                .subgraphs
                .get(&(node.id, "then_branch".to_string()))
                .ok_or_else(|| SessionError::ControlFlow {
                    op: "If".to_string(),
                    reason: "missing required 'then_branch' subgraph".to_string(),
                })?;
            let else_branch = self
                .graph
                .subgraphs
                .get(&(node.id, "else_branch".to_string()))
                .ok_or_else(|| SessionError::ControlFlow {
                    op: "If".to_string(),
                    reason: "missing required 'else_branch' subgraph".to_string(),
                })?;

            if !then_branch.inputs.is_empty() || !else_branch.inputs.is_empty() {
                return Err(SessionError::ControlFlow {
                    op: "If".to_string(),
                    reason: format!(
                        "branch subgraphs must declare zero formal inputs, but then_branch has {} \
                         and else_branch has {}",
                        then_branch.inputs.len(),
                        else_branch.inputs.len()
                    ),
                });
            }
            validate_if_branch_outputs(&self.graph, node)?;
        }

        let cond_vid =
            node.inputs
                .first()
                .and_then(|s| *s)
                .ok_or_else(|| SessionError::ControlFlow {
                    op: "If".to_string(),
                    reason: "missing required 'cond' input".to_string(),
                })?;
        let cond_t = self.value_tensor(cond_vid, resolved, external)?;
        if cond_t.dtype != DataType::Bool {
            return Err(SessionError::DtypeMismatch {
                name: "If cond".to_string(),
                expected: format!("{:?}", DataType::Bool),
                got: format!("{:?}", cond_t.dtype),
            });
        }
        let cond = tensor_scalar_bool(&cond_t).ok_or_else(|| SessionError::ControlFlow {
            op: "If".to_string(),
            reason: format!(
                "'cond' must be a BOOL scalar or single-element tensor, got shape {:?}",
                cond_t.shape
            ),
        })?;

        // Capture-safe loop-invariant control-flow specialization. The predicate
        // is read every step (above) so a genuine branch flip is never missed.
        // When it matches the last observed value AND that value was recorded
        // only for a branch with *no outer captures* (so its outputs depend on
        // nothing but its own constants/initializers and are therefore invariant
        // across decode steps) AND those outputs are still resident in their
        // persistent buffers, re-running the branch is pure waste — skip it. The
        // downstream captured segment reads the unchanged buffers correctly. A
        // branch that reads loop-varying outer values is never memoized, so a
        // stale output is impossible.
        if self.if_last_predicate.get(&node.id) == Some(&cond)
            && node.outputs.iter().all(|v| resolved.contains_key(v))
        {
            return Ok(());
        }

        let attr_key = if cond { "then_branch" } else { "else_branch" };
        // A branch with outer captures may depend on values that change between
        // steps, so its output is not loop-invariant and must never be memoized.
        let taken_branch_is_invariant = self
            .graph
            .subgraphs
            .get(&(node.id, attr_key.to_string()))
            .map(|body| required_outer_names(body).is_empty())
            .unwrap_or(false);
        let prepared = {
            let _s = phase_span!("execif.prepare_subgraph");
            self.prepare_subgraph(node.id, attr_key, resolved, outer_scope, external)?
        };
        let outs = {
            let _s = phase_span!("execif.run_subgraph");
            self.run_subgraph(&prepared, &[])?
        };

        if outs.len() != node.outputs.len() {
            return Err(SessionError::OutputShapeCountMismatch {
                op: format!("If/{attr_key}"),
                expected: node.outputs.len(),
                got: outs.len(),
            });
        }
        {
            let _s = phase_span!("execif.store_output");
            for (vid, t) in node.outputs.iter().zip(outs.iter()) {
                self.store_output_tensor(*vid, t, resolved, external)?;
            }
        }
        // Only enable future skips when the taken branch is loop-invariant.
        // Otherwise drop any stale memo so this `If` always re-runs.
        if taken_branch_is_invariant {
            self.if_last_predicate.insert(node.id, cond);
        } else {
            self.if_last_predicate.remove(&node.id);
        }
        Ok(())
    }

    /// Validate a Loop body's positional contract before the first iteration and
    /// retain each scan output's element type/shape for the zero-iteration case.
    pub(super) fn loop_body_scan_specs(
        &self,
        node: &Node,
        carried: &[Tensor],
        num_scan: usize,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Result<OptionalTensorSpecs> {
        let body = self
            .graph
            .subgraphs
            .get(&(node.id, "body".to_string()))
            .ok_or_else(|| SessionError::ControlFlow {
                op: "Loop".to_string(),
                reason: "missing required 'body' subgraph".to_string(),
            })?;
        let expected_inputs = 2 + carried.len();
        if body.inputs.len() != expected_inputs {
            return Err(SessionError::ControlFlow {
                op: "Loop".to_string(),
                reason: format!(
                    "body declares {} formal input(s), expected {expected_inputs}",
                    body.inputs.len()
                ),
            });
        }
        let expected_outputs = 1 + carried.len() + num_scan;
        if body.outputs.len() != expected_outputs {
            return Err(SessionError::ControlFlow {
                op: "Loop".to_string(),
                reason: format!(
                    "body declares {} output(s), expected {expected_outputs}",
                    body.outputs.len()
                ),
            });
        }

        for (index, expected) in [(0, DataType::Int64), (1, DataType::Bool)] {
            let input = body.inputs[index];
            if body.value_type_is_known(input) && body.value(input).dtype != expected {
                return Err(SessionError::ControlFlow {
                    op: "Loop".to_string(),
                    reason: format!(
                        "body formal input {index} must be {expected:?}, got {:?}",
                        body.value(input).dtype
                    ),
                });
            }
        }
        let cond_out = body.outputs[0];
        if body.value_type_is_known(cond_out) && body.value(cond_out).dtype != DataType::Bool {
            return Err(SessionError::ControlFlow {
                op: "Loop".to_string(),
                reason: format!(
                    "body output 0 ('cond_out') must be Bool, got {:?}",
                    body.value(cond_out).dtype
                ),
            });
        }

        for (index, initial) in carried.iter().enumerate() {
            for (kind, value) in [
                ("formal input", body.inputs[2 + index]),
                ("output", body.outputs[1 + index]),
            ] {
                if body.value_type_is_known(value) && body.value(value).dtype != initial.dtype {
                    return Err(SessionError::ControlFlow {
                        op: "Loop".to_string(),
                        reason: format!(
                            "loop-carried {kind} {index} has dtype {:?}, but its initial value has \
                             dtype {:?}",
                            body.value(value).dtype,
                            initial.dtype
                        ),
                    });
                }
            }
        }

        body.outputs
            .iter()
            .skip(1 + carried.len())
            .zip(node.outputs.iter().skip(carried.len()))
            .enumerate()
            .map(|(index, (&body_output, &node_output))| {
                let body_value = body.value(body_output);
                let node_dtype = self.value_dtypes[&node_output];
                let dtype = if body.value_type_is_known(body_output) {
                    if self.graph.value_type_is_known(node_output)
                        && body_value.dtype != node_dtype
                    {
                        return Err(SessionError::ControlFlow {
                            op: "Loop".to_string(),
                            reason: format!(
                                "scan output {index} has body dtype {:?}, but the Loop node declares \
                                 {node_dtype:?}",
                                body_value.dtype
                            ),
                        });
                    }
                    body_value.dtype
                } else {
                    node_dtype
                };
                let elem_shape = body
                    .value_shape_is_known(body_output)
                    .then(|| as_static_shape(&body_value.shape))
                    .flatten()
                    .or_else(|| {
                        resolved
                            .get(&node_output)
                            .and_then(|shape| shape.get(1..).map(<[_]>::to_vec))
                    });
                Ok(elem_shape.map(|shape| (dtype, shape)))
            })
            .collect()
    }

    /// ONNX `Loop`: inputs `[M?, cond?, v_initial...]`, body signature
    /// `(iter_num, cond_in, carried...) -> (cond_out, carried..., scan_out...)`.
    /// Iterates while `cond` is true and `iter < M`, threading loop-carried
    /// values across iterations and stacking each scan output along a new
    /// leading iteration axis.
    pub(super) fn exec_loop(
        &mut self,
        node: &Node,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
        external: &ExternalBindings,
    ) -> Result<()> {
        // Inputs: [M, cond, v_initial...]. M and cond may be omitted (None slot)
        // or an empty-name optional; absence means "unbounded" / "true".
        let m: Option<i64> = match node.inputs.first().and_then(|s| *s) {
            Some(vid) => {
                let t = self.value_tensor(vid, resolved, external)?;
                if t.dtype != DataType::Int64 {
                    return Err(SessionError::DtypeMismatch {
                        name: "Loop M".to_string(),
                        expected: format!("{:?}", DataType::Int64),
                        got: format!("{:?}", t.dtype),
                    });
                }
                let m = tensor_scalar_i64(&t).ok_or_else(|| SessionError::ControlFlow {
                    op: "Loop".to_string(),
                    reason: format!(
                        "'M' must be an INT64 scalar or single-element tensor, got shape {:?}",
                        t.shape
                    ),
                })?;
                Some(m)
            }
            None => None,
        };
        let mut cond: Option<bool> =
            match node.inputs.get(1).and_then(|s| *s) {
                Some(vid) => {
                    let t = self.value_tensor(vid, resolved, external)?;
                    if t.dtype != DataType::Bool {
                        return Err(SessionError::DtypeMismatch {
                            name: "Loop cond".to_string(),
                            expected: format!("{:?}", DataType::Bool),
                            got: format!("{:?}", t.dtype),
                        });
                    }
                    Some(tensor_scalar_bool(&t).ok_or_else(|| SessionError::ControlFlow {
                    op: "Loop".to_string(),
                    reason: format!(
                        "'cond' must be a BOOL scalar or single-element tensor, got shape {:?}",
                        t.shape
                    ),
                })?)
                }
                None => None,
            };

        // Initial loop-carried dependencies (inputs after M and cond).
        let mut carried: Vec<Tensor> = Vec::new();
        for slot in node.inputs.iter().skip(2) {
            let vid = slot.ok_or_else(|| {
                SessionError::Internal(
                    "Loop: an interior loop-carried input is omitted (empty), which ONNX does not \
                 allow — every v_initial must be provided"
                        .to_string(),
                )
            })?;
            carried.push(self.value_tensor(vid, resolved, external)?);
        }
        let num_carried = carried.len();
        let carried_invariants: Vec<(DataType, Vec<usize>)> = carried
            .iter()
            .map(|tensor| (tensor.dtype, tensor.shape.clone()))
            .collect();
        // Loop outputs = carried finals ++ scan outputs. Scan-output count is
        // whatever remains after the carried finals.
        let num_outputs = node.outputs.len();
        if num_outputs < num_carried {
            return Err(SessionError::Internal(format!(
                "Loop: node declares {num_outputs} output(s) but has {num_carried} loop-carried \
                 dependency(ies); outputs must be carried-finals followed by scan-outputs"
            )));
        }
        let num_scan = num_outputs - num_carried;
        let empty_scan_specs = self.loop_body_scan_specs(node, &carried, num_scan, resolved)?;
        let mut scan_acc: Vec<TensorStackAccumulator> = (0..num_scan)
            .map(|_| TensorStackAccumulator::new())
            .collect();
        let prepared = self.prepare_subgraph(node.id, "body", resolved, outer_scope, external)?;
        let mut iter_tensor = scalar_i64_tensor(0)?;
        let mut cond_tensor = scalar_bool_tensor(cond.unwrap_or(true))?;

        let mut iter: i64 = 0;
        loop {
            if let Some(m) = m
                && iter >= m
            {
                break;
            }
            if cond == Some(false) {
                break;
            }

            iter_tensor.overwrite_bytes(&iter.to_le_bytes())?;
            cond_tensor.overwrite_bytes(&[u8::from(cond.unwrap_or(true))])?;
            let mut formal: Vec<&Tensor> = Vec::with_capacity(2 + num_carried);
            formal.push(&iter_tensor);
            formal.push(&cond_tensor);
            formal.extend(carried.iter());

            let outs = self.run_subgraph(&prepared, &formal)?;
            drop(formal);
            // Body outputs: cond_out, carried..., scan_out...
            let expected = 1 + num_carried + num_scan;
            if outs.len() != expected {
                return Err(SessionError::OutputShapeCountMismatch {
                    op: "Loop/body".to_string(),
                    expected,
                    got: outs.len(),
                });
            }
            let mut it = outs.into_iter();
            let cond_out = it.next().expect("cond_out present");
            cond = Some(tensor_scalar_bool(&cond_out).ok_or_else(|| {
                SessionError::Internal(format!(
                    "Loop: body's first output 'cond_out' must be a BOOL scalar, got dtype {:?}",
                    cond_out.dtype
                ))
            })?);
            let next_carried: Vec<Tensor> = (&mut it).take(num_carried).collect();
            for (index, (tensor, (expected_dtype, expected_shape))) in
                next_carried.iter().zip(&carried_invariants).enumerate()
            {
                if tensor.dtype != *expected_dtype {
                    return Err(SessionError::ControlFlow {
                        op: "Loop".to_string(),
                        reason: format!(
                            "loop-carried output {index} dtype mismatch: expected \
                             {expected_dtype:?}, got {:?}",
                            tensor.dtype
                        ),
                    });
                }
                if tensor.shape != *expected_shape {
                    return Err(SessionError::ControlFlow {
                        op: "Loop".to_string(),
                        reason: format!(
                            "loop-carried output {index} shape mismatch: expected \
                             {expected_shape:?}, got {:?}",
                            tensor.shape
                        ),
                    });
                }
            }
            carried = next_carried;
            for acc in scan_acc.iter_mut() {
                acc.push(it.next().expect("scan output present"))?;
            }

            iter = iter
                .checked_add(1)
                .ok_or_else(|| SessionError::ControlFlow {
                    op: "Loop".to_string(),
                    reason: "iteration counter overflowed INT64".to_string(),
                })?;
        }

        // Emit outputs: carried finals, then stacked scan outputs.
        for (i, t) in carried.iter().enumerate() {
            self.store_output_tensor(node.outputs[i], t, resolved, external)?;
        }
        for (s, (acc, empty_spec)) in scan_acc.into_iter().zip(empty_scan_specs).enumerate() {
            let (dtype, shape, bytes) = acc.finish_with_empty(empty_spec, s)?;
            self.store_output_bytes(
                node.outputs[num_carried + s],
                dtype,
                shape,
                &bytes,
                resolved,
                external,
            )?;
        }
        Ok(())
    }

    // The Scan operator's body specs derive from these independent ONNX
    // attributes/inputs; bundling them into a context struct belongs with the
    // control-flow decomposition (Dallas #6).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn scan_body_specs(
        &self,
        node: &Node,
        state: &[Tensor],
        scan_inputs: &[Tensor],
        input_axes: &[usize],
        num_scan_outputs: usize,
        output_axes: &[i64],
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Result<OptionalTensorSpecs> {
        let body = self
            .graph
            .subgraphs
            .get(&(node.id, "body".to_string()))
            .ok_or_else(|| SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: "missing required 'body' subgraph".to_string(),
            })?;
        let expected_inputs = state.len() + scan_inputs.len();
        if body.inputs.len() != expected_inputs {
            return Err(SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: format!(
                    "body declares {} formal input(s), expected {expected_inputs}",
                    body.inputs.len()
                ),
            });
        }
        let expected_outputs = state.len() + num_scan_outputs;
        if body.outputs.len() != expected_outputs {
            return Err(SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: format!(
                    "body declares {} output(s), expected {expected_outputs}",
                    body.outputs.len()
                ),
            });
        }

        for (index, initial) in state.iter().enumerate() {
            for (kind, value) in [
                ("formal input", body.inputs[index]),
                ("output", body.outputs[index]),
            ] {
                if body.value_type_is_known(value) && body.value(value).dtype != initial.dtype {
                    return Err(SessionError::ControlFlow {
                        op: "Scan".to_string(),
                        reason: format!(
                            "state {kind} {index} has dtype {:?}, but its initial value has dtype {:?}",
                            body.value(value).dtype,
                            initial.dtype
                        ),
                    });
                }
            }
        }
        for (index, ((input, &axis), &formal)) in scan_inputs
            .iter()
            .zip(input_axes)
            .zip(body.inputs.iter().skip(state.len()))
            .enumerate()
        {
            if body.value_type_is_known(formal) && body.value(formal).dtype != input.dtype {
                return Err(SessionError::ControlFlow {
                    op: "Scan".to_string(),
                    reason: format!(
                        "scan formal input {index} has dtype {:?}, but scan input {index} has dtype {:?}",
                        body.value(formal).dtype,
                        input.dtype
                    ),
                });
            }
            let mut slice_shape = input.shape.clone();
            slice_shape.remove(axis);
            if body.value_shape_is_known(formal)
                && let Some(shape) = as_static_shape(&body.value(formal).shape)
                && shape != slice_shape
            {
                return Err(SessionError::ControlFlow {
                    op: "Scan".to_string(),
                    reason: format!(
                        "scan formal input {index} has shape {shape:?}, but slicing input shape {:?} \
                         along axis {axis} produces {slice_shape:?}",
                        input.shape
                    ),
                });
            }
        }

        body.outputs
            .iter()
            .skip(state.len())
            .zip(node.outputs.iter().skip(state.len()))
            .zip(output_axes)
            .enumerate()
            .map(|(index, ((&body_output, &node_output), &axis))| {
                let body_value = body.value(body_output);
                let node_dtype = self.value_dtypes[&node_output];
                let dtype = if body.value_type_is_known(body_output) {
                    if self.graph.value_type_is_known(node_output)
                        && body_value.dtype != node_dtype
                    {
                        return Err(SessionError::ControlFlow {
                            op: "Scan".to_string(),
                            reason: format!(
                                "scan output {index} has body dtype {:?}, but the Scan node declares \
                                 {node_dtype:?}",
                                body_value.dtype
                            ),
                        });
                    }
                    body_value.dtype
                } else {
                    node_dtype
                };
                let elem_shape = body
                    .value_shape_is_known(body_output)
                    .then(|| as_static_shape(&body_value.shape))
                    .flatten()
                    .or_else(|| {
                        resolved.get(&node_output).and_then(|shape| {
                            normalize_axis(axis, shape.len()).map(|axis| {
                                let mut elem_shape = shape.clone();
                                elem_shape.remove(axis);
                                elem_shape
                            })
                        })
                    });
                if let Some(shape) = &elem_shape
                    && normalize_axis(axis, shape.len() + 1).is_none()
                {
                    return Err(SessionError::ControlFlow {
                        op: "Scan".to_string(),
                        reason: format!(
                            "scan_output_axes[{index}]={axis} is out of range for output rank {}",
                            shape.len() + 1
                        ),
                    });
                }
                Ok(elem_shape.map(|shape| (dtype, shape)))
            })
            .collect()
    }

    /// ONNX `Scan`: slice configured input axes/directions, thread invariant
    /// state through the body, and stack scan outputs on configured axes.
    pub(super) fn exec_scan(
        &mut self,
        node: &Node,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
        external: &ExternalBindings,
    ) -> Result<()> {
        let raw_num_scan_inputs = node
            .attr("num_scan_inputs")
            .and_then(|a| a.as_int())
            .ok_or_else(|| SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: "required attribute 'num_scan_inputs' is missing or not an INT".to_string(),
            })?;
        let num_scan_inputs = usize::try_from(raw_num_scan_inputs)
            .ok()
            .filter(|&count| count != 0)
            .ok_or_else(|| SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: format!(
                    "'num_scan_inputs' must be a positive INT, got {raw_num_scan_inputs}"
                ),
            })?;

        let total_inputs = node.inputs.len();
        if total_inputs < num_scan_inputs {
            return Err(SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: format!(
                    "node has {total_inputs} input(s) but num_scan_inputs={num_scan_inputs}"
                ),
            });
        }
        let num_state = total_inputs - num_scan_inputs;
        if node.outputs.len() < num_state {
            return Err(SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: format!(
                    "declares {} output(s) but has {num_state} state variable(s)",
                    node.outputs.len()
                ),
            });
        }
        let num_scan_outputs = node.outputs.len() - num_state;
        let input_axes_raw = scan_list_attr(node, "scan_input_axes", num_scan_inputs, 0)?;
        let input_directions = scan_list_attr(node, "scan_input_directions", num_scan_inputs, 0)?;
        let output_axes = scan_list_attr(node, "scan_output_axes", num_scan_outputs, 0)?;
        let output_directions =
            scan_list_attr(node, "scan_output_directions", num_scan_outputs, 0)?;
        for (name, values) in [
            ("scan_input_directions", &input_directions),
            ("scan_output_directions", &output_directions),
        ] {
            for (index, &value) in values.iter().enumerate() {
                if !matches!(value, 0 | 1) {
                    return Err(SessionError::ControlFlow {
                        op: "Scan".to_string(),
                        reason: format!(
                            "{name}[{index}] must be 0 (forward) or 1 (reverse), got {value}"
                        ),
                    });
                }
            }
        }

        let mut state: Vec<Tensor> = Vec::with_capacity(num_state);
        for slot in node.inputs.iter().take(num_state) {
            let vid = slot.ok_or_else(|| SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: "an initial-state input is omitted (empty), which ONNX does not allow"
                    .to_string(),
            })?;
            state.push(self.value_tensor(vid, resolved, external)?);
        }
        let mut scan_inputs: Vec<Tensor> = Vec::with_capacity(num_scan_inputs);
        for slot in node.inputs.iter().skip(num_state) {
            let vid = slot.ok_or_else(|| SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: "a scan input is omitted (empty), which ONNX does not allow".to_string(),
            })?;
            scan_inputs.push(self.value_tensor(vid, resolved, external)?);
        }

        let mut input_axes = Vec::with_capacity(num_scan_inputs);
        for (index, (input, &raw_axis)) in scan_inputs.iter().zip(&input_axes_raw).enumerate() {
            let axis = normalize_axis(raw_axis, input.shape.len()).ok_or_else(|| {
                SessionError::ControlFlow {
                    op: "Scan".to_string(),
                    reason: format!(
                        "scan_input_axes[{index}]={raw_axis} is out of range for input rank {}",
                        input.shape.len()
                    ),
                }
            })?;
            input_axes.push(axis);
        }
        let trip_count = scan_inputs[0].shape[input_axes[0]];
        for (index, (input, &axis)) in scan_inputs.iter().zip(&input_axes).enumerate() {
            let length = input.shape[axis];
            if length != trip_count {
                return Err(SessionError::ControlFlow {
                    op: "Scan".to_string(),
                    reason: format!(
                        "scan input {index} has scan-axis length {length}, but the first scan input \
                         has {trip_count}; all scan inputs must agree"
                    ),
                });
            }
        }

        let state_specs: Vec<(DataType, Vec<usize>)> = state
            .iter()
            .map(|tensor| (tensor.dtype, tensor.shape.clone()))
            .collect();
        let empty_specs = self.scan_body_specs(
            node,
            &state,
            &scan_inputs,
            &input_axes,
            num_scan_outputs,
            &output_axes,
            resolved,
        )?;
        let mut scan_acc: Vec<TensorStackAccumulator> = (0..num_scan_outputs)
            .map(|_| TensorStackAccumulator::new())
            .collect();
        let prepared = self.prepare_subgraph(node.id, "body", resolved, outer_scope, external)?;
        let mut scan_slices = Vec::with_capacity(num_scan_inputs);
        if trip_count != 0 {
            for (index, ((input, &axis), &direction)) in scan_inputs
                .iter()
                .zip(&input_axes)
                .zip(&input_directions)
                .enumerate()
            {
                let source_index = if direction == 0 { 0 } else { trip_count - 1 };
                let (shape, bytes) = scan_slice(input, axis, source_index, index)?;
                scan_slices.push(Tensor::from_raw(input.dtype, shape, &bytes)?);
            }
        }
        // Runtime dual-path (slice 1a). A single-trip Scan — trip_count == 1, the
        // per-token decode regime — runs its body ONCE straight-line under the
        // opt-in `scan_inline_single_trip_enabled` flag, instead of the generic
        // loop. The selection is keyed on the RUNTIME trip_count, never the graph:
        // prefill (trip_count = prompt_len > 1) and decode (trip_count == 1) share
        // this same executor/plan, so a static single-trip rewrite would corrupt
        // prefill. Flag OFF, or any trip_count other than 1, takes the unchanged
        // loop path below. Both regimes drive the body through the identical
        // `run_scan_body_step` and share the finishing code that follows, so the
        // inline path is byte-exact with a one-iteration loop by construction.
        if self.scan_inline_single_trip_enabled && trip_count == 1 {
            self.scan_inline_single_trip_count += 1;
            let (next_state, scan_outs) = self.run_scan_body_step(
                &prepared,
                &state,
                &scan_slices,
                num_state,
                num_scan_outputs,
                &state_specs,
            )?;
            state = next_state;
            for (acc, tensor) in scan_acc.iter_mut().zip(scan_outs) {
                acc.push(tensor)?;
            }
        } else {
            for step in 0..trip_count {
                if step != 0 {
                    for (index, (((input, &axis), &direction), slice)) in scan_inputs
                        .iter()
                        .zip(&input_axes)
                        .zip(&input_directions)
                        .zip(scan_slices.iter_mut())
                        .enumerate()
                    {
                        let source_index = if direction == 0 {
                            step
                        } else {
                            trip_count - 1 - step
                        };
                        let (_, bytes) = scan_slice(input, axis, source_index, index)?;
                        slice.overwrite_bytes(&bytes)?;
                    }
                }
                let (next_state, scan_outs) = self.run_scan_body_step(
                    &prepared,
                    &state,
                    &scan_slices,
                    num_state,
                    num_scan_outputs,
                    &state_specs,
                )?;
                state = next_state;
                for (acc, tensor) in scan_acc.iter_mut().zip(scan_outs) {
                    acc.push(tensor)?;
                }
            }
        }

        for (i, t) in state.iter().enumerate() {
            self.store_output_tensor(node.outputs[i], t, resolved, external)?;
        }
        for (s, ((acc, empty_spec), (&axis, &direction))) in scan_acc
            .into_iter()
            .zip(empty_specs)
            .zip(output_axes.iter().zip(&output_directions))
            .enumerate()
        {
            let (dtype, shape, bytes) = acc.finish_scan(axis, direction, empty_spec, s)?;
            self.store_output_bytes(
                node.outputs[num_state + s],
                dtype,
                shape,
                &bytes,
                resolved,
                external,
            )?;
        }
        Ok(())
    }

    /// Run the `Scan` body once for the current formal inputs (carried state
    /// followed by this step's scan-input slices), validate the body's output
    /// count and the carried-state dtype/shape invariants, and split the result
    /// into the next carried state and this step's scan outputs. Shared verbatim
    /// by the generic multi-trip loop and the single-trip inline dual-path so the
    /// two body-execution regimes can never diverge — the sole guarantee behind
    /// slice 1a's byte-exactness.
    fn run_scan_body_step(
        &mut self,
        prepared: &PreparedSubgraph,
        state: &[Tensor],
        scan_slices: &[Tensor],
        num_state: usize,
        num_scan_outputs: usize,
        state_specs: &[(DataType, Vec<usize>)],
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let mut formal: Vec<&Tensor> = Vec::with_capacity(state.len() + scan_slices.len());
        formal.extend(state.iter());
        formal.extend(scan_slices.iter());

        let outs = self.run_subgraph(prepared, &formal)?;
        drop(formal);
        let expected = num_state + num_scan_outputs;
        if outs.len() != expected {
            return Err(SessionError::OutputShapeCountMismatch {
                op: "Scan/body".to_string(),
                expected,
                got: outs.len(),
            });
        }
        let mut it = outs.into_iter();
        let next_state: Vec<Tensor> = (&mut it).take(num_state).collect();
        for (index, (tensor, (expected_dtype, expected_shape))) in
            next_state.iter().zip(state_specs).enumerate()
        {
            if tensor.dtype != *expected_dtype {
                return Err(SessionError::ControlFlow {
                    op: "Scan".to_string(),
                    reason: format!(
                        "state output {index} dtype mismatch: expected {expected_dtype:?}, got {:?}",
                        tensor.dtype
                    ),
                });
            }
            if tensor.shape != *expected_shape {
                return Err(SessionError::ControlFlow {
                    op: "Scan".to_string(),
                    reason: format!(
                        "state output {index} shape mismatch: expected {expected_shape:?}, got {:?}",
                        tensor.shape
                    ),
                });
            }
        }
        let scan_outs: Vec<Tensor> = it.collect();
        Ok((next_state, scan_outs))
    }
}

fn scan_slice(
    t: &Tensor,
    axis: usize,
    index: usize,
    input_index: usize,
) -> Result<(Vec<usize>, Vec<u8>)> {
    let axis_len = t.shape[axis];
    if index >= axis_len {
        return Err(SessionError::ControlFlow {
            op: "Scan".to_string(),
            reason: format!(
                "slice index {index} is out of range for scan input {input_index} axis {axis}"
            ),
        });
    }
    let esize = t.dtype.byte_size();
    if esize == 0 {
        return Err(SessionError::ControlFlow {
            op: "Scan".to_string(),
            reason: format!(
                "sub-byte dtype {:?} for scan input {input_index} is not supported",
                t.dtype
            ),
        });
    }
    let mut shape = t.shape.clone();
    shape.remove(axis);
    let outer = checked_numel(&t.shape[..axis], || format!("Scan input {input_index}"))?;
    let inner = checked_numel(&t.shape[axis + 1..], || format!("Scan input {input_index}"))?;
    let inner_bytes = checked_storage_bytes(
        t.dtype,
        inner,
        || format!("Scan input {input_index}"),
        &t.shape,
    )?;
    let total_bytes =
        outer
            .checked_mul(inner_bytes)
            .ok_or_else(|| SessionError::ShapeOverflow {
                value: format!("Scan input {input_index} slice"),
                dims: shape.clone(),
            })?;
    let source = t.as_bytes();
    let mut bytes = vec![0u8; total_bytes];
    for outer_index in 0..outer {
        let src = (outer_index * axis_len + index) * inner_bytes;
        let dst = outer_index * inner_bytes;
        bytes[dst..dst + inner_bytes].copy_from_slice(&source[src..src + inner_bytes]);
    }
    Ok((shape, bytes))
}

/// Incremental accumulator for Loop/Scan outputs. Iteration tensors are copied
/// into one byte buffer and dropped; non-leading Scan axes are rearranged once
/// when the final tensor is materialized.
struct TensorStackAccumulator {
    dtype: Option<DataType>,
    elem_shape: Vec<usize>,
    len: usize,
    bytes: Vec<u8>,
}

impl TensorStackAccumulator {
    pub(super) fn new() -> Self {
        Self {
            dtype: None,
            elem_shape: Vec::new(),
            len: 0,
            bytes: Vec::new(),
        }
    }

    pub(super) fn push(&mut self, tensor: Tensor) -> Result<()> {
        if let Some(dtype) = self.dtype {
            if tensor.shape != self.elem_shape || tensor.dtype != dtype {
                return Err(SessionError::Internal(format!(
                    "Loop/Scan: scan output slice {} has shape {:?} dtype {:?} but the first slice \
                     is shape {:?} dtype {:?}; every iteration's scan output must match",
                    self.len, tensor.shape, tensor.dtype, self.elem_shape, dtype
                )));
            }
        } else {
            if tensor.dtype.byte_size() == 0 {
                return Err(SessionError::Internal(format!(
                    "Loop/Scan: sub-byte dtype {:?} scan outputs are not supported",
                    tensor.dtype
                )));
            }
            self.dtype = Some(tensor.dtype);
            self.elem_shape = tensor.shape.clone();
        }
        self.bytes.extend_from_slice(tensor.as_bytes());
        self.len += 1;
        Ok(())
    }

    pub(super) fn finish(self) -> (DataType, Vec<usize>, Vec<u8>) {
        if self.len == 0 {
            return (DataType::Float32, vec![0], Vec::new());
        }
        let dtype = self.dtype.expect("non-empty accumulator has dtype");
        let mut shape = Vec::with_capacity(1 + self.elem_shape.len());
        shape.push(self.len);
        shape.extend(self.elem_shape);
        (dtype, shape, self.bytes)
    }

    pub(super) fn finish_with_empty(
        self,
        empty_spec: Option<(DataType, Vec<usize>)>,
        output_index: usize,
    ) -> Result<(DataType, Vec<usize>, Vec<u8>)> {
        if self.len != 0 {
            return Ok(self.finish());
        }
        let (dtype, elem_shape) = empty_spec.ok_or_else(|| SessionError::ControlFlow {
            op: "Loop".to_string(),
            reason: format!(
                "cannot determine the element shape of scan output {output_index} for a \
                 zero-iteration result"
            ),
        })?;
        let mut shape = Vec::with_capacity(1 + elem_shape.len());
        shape.push(0);
        shape.extend(elem_shape);
        Ok((dtype, shape, Vec::new()))
    }

    pub(super) fn finish_scan(
        self,
        axis: i64,
        direction: i64,
        empty_spec: Option<(DataType, Vec<usize>)>,
        output_index: usize,
    ) -> Result<(DataType, Vec<usize>, Vec<u8>)> {
        let (dtype, elem_shape) = match self.dtype {
            Some(dtype) => (dtype, self.elem_shape.clone()),
            None => empty_spec.ok_or_else(|| SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: format!(
                    "cannot determine the element shape of scan output {output_index} for a \
                     zero-iteration result"
                ),
            })?,
        };
        let output_rank = elem_shape.len() + 1;
        let axis = normalize_axis(axis, output_rank).ok_or_else(|| SessionError::ControlFlow {
            op: "Scan".to_string(),
            reason: format!(
                "scan_output_axes[{output_index}]={axis} is out of range for output rank \
                 {output_rank}"
            ),
        })?;
        if self.len == 0 {
            let mut shape = elem_shape;
            shape.insert(axis, 0);
            return Ok((dtype, shape, Vec::new()));
        }
        if axis == 0 && direction == 0 {
            let mut shape = Vec::with_capacity(output_rank);
            shape.push(self.len);
            shape.extend(elem_shape);
            return Ok((dtype, shape, self.bytes));
        }

        let elem_numel = checked_numel(&elem_shape, || {
            format!("Scan output {output_index} element")
        })?;
        let elem_bytes = checked_storage_bytes(
            dtype,
            elem_numel,
            || format!("Scan output {output_index} element"),
            &elem_shape,
        )?;
        let mut elements: Vec<&[u8]> = if elem_bytes == 0 {
            (0..self.len).map(|_| &self.bytes[..]).collect()
        } else {
            self.bytes.chunks_exact(elem_bytes).collect()
        };
        if direction == 1 {
            elements.reverse();
        }
        let (shape, bytes) = stack_new_axis(&elements, &elem_shape, axis, dtype.byte_size())?;
        Ok((dtype, shape, bytes))
    }
}
