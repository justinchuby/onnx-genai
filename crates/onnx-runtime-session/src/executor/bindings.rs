use super::*;

impl Executor {
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
        self.run_scoped(inputs, &HashMap::new(), &ExternalBindings::default())?
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
    ) -> Result<Vec<Option<Tensor>>> {
        let external = self.prepare_external_bindings(bindings)?;
        self.run_scoped(inputs, &HashMap::new(), &external)?
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

    pub(crate) fn try_capture_with_device_bindings(
        &mut self,
        inputs: &[(&str, &Tensor)],
        bindings: &mut [DeviceIoBinding],
    ) -> Result<DeviceGraphCaptureResult> {
        let external = self.prepare_external_bindings(bindings)?;
        match self.run_scoped_mode(inputs, &HashMap::new(), &external, RunMode::Capture)? {
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
                self.device_graph_signature = Some(Self::binding_signature(bindings));
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
        let external = self.prepare_external_bindings(bindings)?;
        let signature = Self::binding_signature(bindings);
        if self.device_graph_signature.as_ref() != Some(&signature) {
            self.reset_device_graph()?;
            return Err(SessionError::Internal(
                "device graph replay bindings changed shape, address, or I/O identity; graph was invalidated"
                    .into(),
            ));
        }
        // Whole-subgraph capture (a single graph, no eager seams) keeps the
        // zero-host-work fast path: just relaunch the one installed graph.
        // Segmented capture must re-establish the run context and interleave
        // segment replays with eager seam-node execution, so it routes through
        // the scoped runner in replay mode.
        let single_graph = self
            .capture_schedule
            .as_ref()
            .is_none_or(CaptureSchedule::is_single_graph);
        if single_graph {
            self.ep.replay_device_graph()?;
            return Ok(true);
        }
        match self.run_scoped_mode(&[], &HashMap::new(), &external, RunMode::Replay)? {
            // `run_scoped_mode` clears `capture_schedule` when a branch flip
            // retired the graph this step; report that so the caller re-arms.
            ScopedRunResult::Executed(_) => Ok(self.capture_schedule.is_some()),
            ScopedRunResult::NotCapturable(reason) => {
                self.reset_device_graph()?;
                Err(SessionError::Internal(format!(
                    "segmented device graph replay lost its schedule: {reason}"
                )))
            }
        }
    }

    pub(crate) fn reset_device_graph(&mut self) -> Result<bool> {
        self.device_graph_signature = None;
        self.capture_schedule = None;
        self.capture_cf_shapes.clear();
        self.capture_warm_seeded.clear();
        Ok(self.ep.reset_device_graph()?)
    }

    /// Structured segment-boundary reasons from the most recent capture: one
    /// entry per non-capturable seam node the CUDA EP ran eagerly between
    /// captured segments. Empty for a whole-subgraph (single-graph) capture.
    pub(crate) fn capture_segmentation(&self) -> &[CaptureDecline] {
        &self.capture_segmentation
    }

    /// Number of captured device-graph segments installed by the most recent
    /// capture (1 for a whole-subgraph capture, >=2 when seams split it).
    pub(crate) fn captured_segment_count(&self) -> usize {
        self.capture_schedule
            .as_ref()
            .map(CaptureSchedule::captured_segments)
            .unwrap_or(0)
    }

    pub(crate) fn check_device_capture_error(&self) -> Result<u32> {
        Ok(self.ep.check_device_capture_error()?)
    }

    pub(crate) fn device_allocation_counts(&self) -> Option<DeviceAllocationCounts> {
        self.ep
            .device_allocation_counts()
            .map(|(allocations, frees)| DeviceAllocationCounts { allocations, frees })
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
                device_ptr: binding.device_ptr() as usize,
            })
            .collect()
    }

    pub(super) fn prepare_external_bindings(
        &self,
        bindings: &mut [DeviceIoBinding],
    ) -> Result<ExternalBindings> {
        let mut external = ExternalBindings::default();
        for binding in bindings {
            let input_name = binding.input_name().to_string();
            let bind_input = binding.binds_input();
            let output_name = binding.output_name().map(str::to_string);
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
            let required = required_binding_bytes(dtype, physical_shape, &input_name)?;
            if required > len {
                return Err(SessionError::Internal(format!(
                    "device binding '{input_name}' needs {required} bytes for {physical_shape:?}, allocation has {len}"
                )));
            }
            let ptr = binding.buffer_mut().as_mut_ptr();
            if bind_input {
                let input_vid = *self.input_index.get(&input_name).ok_or_else(|| {
                    SessionError::InputNotFound {
                        name: input_name.clone(),
                    }
                })?;
                let value = ExternalValue {
                    dtype,
                    shape: binding.kernel_input_shape().to_vec(),
                    accepts_subshape: false,
                    ptr,
                    len,
                    alignment,
                    device,
                };
                if external.inputs.insert(input_vid, value).is_some() {
                    return Err(SessionError::Internal(format!(
                        "duplicate device input binding '{input_name}'"
                    )));
                }
            }
            if let Some(output_name) = output_name {
                let output_vid = self
                    .graph
                    .outputs
                    .iter()
                    .copied()
                    .find(|&vid| {
                        self.graph.value(vid).name.as_deref() == Some(output_name.as_str())
                    })
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
                        name: output_name.clone(),
                        expected: format!("{:?}", self.value_dtypes[&output_vid]),
                        got: format!("{dtype:?}"),
                    });
                }
                let value = ExternalValue {
                    dtype,
                    shape: binding.physical_shape().to_vec(),
                    accepts_subshape: bind_input
                        && binding.logical_shape() != binding.physical_shape(),
                    ptr,
                    len,
                    alignment,
                    device,
                };
                if external.outputs.insert(output_vid, value).is_some() {
                    return Err(SessionError::Internal(format!(
                        "duplicate device output binding '{output_name}'"
                    )));
                }
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
