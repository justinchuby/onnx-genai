use super::*;

impl Executor {
    pub(super) fn exec_state_node(
        &mut self,
        pi: usize,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> Result<()> {
        let node_id = self.plan[pi].node_id;
        let node = self.graph.node(node_id).clone();
        match node.op_type.as_str() {
            "AttentionWithCache" => self.exec_attention_with_cache(&node, resolved, external),
            "LinearAttentionWithState" => {
                self.exec_linear_attention_with_state(&node, resolved, external)
            }
            "CausalConvWithState" => self.exec_causal_conv_with_state(&node, resolved, external),
            _ => Err(self.state_error(
                &node.op_type,
                "unrecognized state operator in executor routing",
            )),
        }
    }

    fn state_error(&self, op: &str, reason: impl Into<String>) -> SessionError {
        SessionError::StateOp {
            op: op.to_string(),
            reason: reason.into(),
        }
    }

    fn tensor_f32(
        &self,
        input: Option<ValueId>,
        resolved: &HashMap<ValueId, Vec<usize>>,
        op: &str,
        name: &str,
    ) -> Result<Option<(Vec<usize>, Vec<f32>)>> {
        let Some(id) = input else { return Ok(None) };
        if self.opaque_value_ids.contains(&id) {
            return Err(
                self.state_error(op, format!("input {name:?} is opaque, expected a tensor"))
            );
        }
        let dtype = self.value_dtypes[&id];
        if dtype != DataType::Float32 {
            return Err(self.state_error(
                op,
                format!("input {name:?} has dtype {dtype:?}; this CPU prototype requires Float32"),
            ));
        }
        let shape = resolved.get(&id).cloned().ok_or_else(|| {
            self.state_error(op, format!("input {name:?} has no resolved runtime shape"))
        })?;
        let buffer = self.buffers.get(&id).ok_or_else(|| {
            self.state_error(op, format!("input {name:?} has no live tensor buffer"))
        })?;
        if !buffer.device().is_host_accessible() {
            return Err(self.state_error(
                op,
                format!(
                    "input {name:?} is on {:?}; prototype opaque-state operators currently require host-accessible CPU tensors",
                    buffer.device()
                ),
            ));
        }
        let values = onnx_runtime_ir::read_vec_le::<f32>(host_bytes(buffer)).map_err(|error| {
            self.state_error(
                op,
                format!("input {name:?} could not be decoded as Float32: {error}"),
            )
        })?;
        Ok(Some((shape, values)))
    }

    fn tensor_i64(
        &self,
        input: Option<ValueId>,
        resolved: &HashMap<ValueId, Vec<usize>>,
        op: &str,
        name: &str,
    ) -> Result<Option<(Vec<usize>, Vec<i64>)>> {
        let Some(id) = input else { return Ok(None) };
        let dtype = self.value_dtypes[&id];
        let shape = resolved.get(&id).cloned().ok_or_else(|| {
            self.state_error(op, format!("input {name:?} has no resolved runtime shape"))
        })?;
        let buffer = self.buffers.get(&id).ok_or_else(|| {
            self.state_error(op, format!("input {name:?} has no live tensor buffer"))
        })?;
        let bytes = host_bytes(buffer);
        let values = match dtype {
            DataType::Int64 => onnx_runtime_ir::read_vec_le::<i64>(bytes),
            DataType::Int32 => onnx_runtime_ir::read_vec_le::<i32>(bytes)
                .map(|v| v.into_iter().map(i64::from).collect()),
            _ => {
                return Err(self.state_error(
                    op,
                    format!("input {name:?} has dtype {dtype:?}; expected Int32 or Int64"),
                ));
            }
        }
        .map_err(|error| self.state_error(op, format!("input {name:?} is invalid: {error}")))?;
        Ok(Some((shape, values)))
    }

    fn required_tensor_f32(
        &self,
        input: Option<ValueId>,
        resolved: &HashMap<ValueId, Vec<usize>>,
        op: &str,
        name: &str,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        self.tensor_f32(input, resolved, op, name)?
            .ok_or_else(|| self.state_error(op, format!("required input {name:?} is missing")))
    }

    fn nonnegative_len(&self, value: i64, limit: usize, op: &str, name: &str) -> Result<usize> {
        let value = usize::try_from(value).map_err(|_| {
            self.state_error(
                op,
                format!("{name}={value} is negative; provide a value in 0..={limit}"),
            )
        })?;
        if value > limit {
            return Err(self.state_error(
                op,
                format!("{name}={value} exceeds the available length {limit}"),
            ));
        }
        Ok(value)
    }

    fn required_output(&self, node: &Node, slot: usize) -> Result<ValueId> {
        node.outputs.get(slot).copied().flatten().ok_or_else(|| {
            self.state_error(
                &node.op_type,
                format!("required output slot {slot} is omitted from the node"),
            )
        })
    }

    fn opaque_input(
        &self,
        input: Option<ValueId>,
        node: NodeId,
        kind: OpaqueStateKind,
        op: &str,
    ) -> Result<Option<OpaqueValue>> {
        let Some(id) = input else { return Ok(None) };
        let Some(value) = self.opaque_values.get(&id).cloned() else {
            return Ok(None);
        };
        if value.owner != self.instance_id.get() {
            return Err(self.state_error(
                op,
                format!(
                    "opaque {} belongs to executor {}, but this session is executor {}; pass state returned by this exact session",
                    kind.onnx_name(),
                    value.owner,
                    self.instance_id.get()
                ),
            ));
        }
        if value.node != node.0 {
            return Err(self.state_error(
                op,
                format!(
                    "opaque {} belongs to state slot node#{}, but input targets node#{}; connect each state output back to the same operator",
                    kind.onnx_name(),
                    value.node,
                    node.0
                ),
            ));
        }
        if value.kind != kind {
            return Err(self.state_error(
                op,
                format!(
                    "state slot expects {}, but received {}; connect the matching opaque state type",
                    kind.onnx_name(),
                    value.kind.onnx_name()
                ),
            ));
        }
        Ok(Some(value))
    }

    fn put_state(
        &mut self,
        output: ValueId,
        node: NodeId,
        kind: OpaqueStateKind,
        payload: StatePayload,
    ) {
        self.opaque_values.insert(
            output,
            OpaqueValue {
                owner: self.instance_id.get(),
                node: node.0,
                kind,
                payload: Arc::new(payload),
            },
        );
    }

    fn write_f32(
        &mut self,
        output: ValueId,
        shape: Vec<usize>,
        values: &[f32],
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> Result<()> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.store_raw_tensor_output(output, DataType::Float32, shape, &bytes, resolved, external)
    }

    fn kv_row_len(row: &KvRows) -> usize {
        match row {
            KvRows::Dense(records) => records.len(),
            KvRows::Paged { len, .. } => *len,
        }
    }

    fn kv_row_record(row: &KvRows, index: usize) -> Option<&KvRecord> {
        match row {
            KvRows::Dense(records) => records.get(index),
            KvRows::Paged {
                page_size, pages, ..
            } => pages.get(index / *page_size)?.get(index % *page_size),
        }
    }

    fn append_kv_row(&self, old: Option<&KvRows>, new: Vec<KvRecord>) -> KvRows {
        match self.state_config.attention {
            crate::AttentionCacheMode::Dense => {
                let mut records =
                    Vec::with_capacity(old.map(Self::kv_row_len).unwrap_or(0) + new.len());
                if let Some(old) = old {
                    for index in 0..Self::kv_row_len(old) {
                        records.push(Self::kv_row_record(old, index).unwrap().clone());
                    }
                }
                records.extend(new);
                KvRows::Dense(Arc::new(records))
            }
            crate::AttentionCacheMode::Paged { page_size } => {
                let appended = new.len();
                let mut pages = match old {
                    Some(KvRows::Paged { pages, .. }) => (**pages).clone(),
                    Some(old) => {
                        let mut pages = Vec::new();
                        let mut page = Vec::with_capacity(page_size);
                        for index in 0..Self::kv_row_len(old) {
                            page.push(Self::kv_row_record(old, index).unwrap().clone());
                            if page.len() == page_size {
                                pages.push(Arc::new(std::mem::take(&mut page)));
                                page = Vec::with_capacity(page_size);
                            }
                        }
                        if !page.is_empty() {
                            pages.push(Arc::new(page));
                        }
                        pages
                    }
                    None => Vec::new(),
                };
                let old_len = old.map(Self::kv_row_len).unwrap_or(0);
                let mut tail = if old_len % page_size != 0 {
                    Arc::make_mut(pages.last_mut().unwrap()).clone()
                } else {
                    Vec::with_capacity(page_size)
                };
                if old_len % page_size != 0 {
                    pages.pop();
                }
                for record in new {
                    tail.push(record);
                    if tail.len() == page_size {
                        pages.push(Arc::new(std::mem::take(&mut tail)));
                        tail = Vec::with_capacity(page_size);
                    }
                }
                if !tail.is_empty() {
                    pages.push(Arc::new(tail));
                }
                KvRows::Paged {
                    page_size,
                    pages: Arc::new(pages),
                    len: old_len + appended,
                }
            }
        }
    }

    fn exec_attention_with_cache(
        &mut self,
        node: &Node,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> Result<()> {
        let op = &node.op_type;
        if node.outputs.get(2).is_some_and(Option::is_some) {
            return Err(self.state_error(
                op,
                "optional qk_matmul_output is not implemented by the prototype; omit output slot 2",
            ));
        }
        if node
            .attr("allow_lossy_state")
            .and_then(Attribute::as_int)
            .unwrap_or(0)
            != 0
        {
            return Err(self.state_error(
                op,
                "allow_lossy_state=1 is not implemented; use lossless state storage",
            ));
        }
        if node.inputs.get(3).is_some_and(Option::is_some) {
            return Err(self.state_error(
                op,
                "attn_mask is not implemented by the prototype; omit input slot 3",
            ));
        }
        let (q_shape, q) = self
            .tensor_f32(node.inputs.first().copied().flatten(), resolved, op, "Q")?
            .ok_or_else(|| self.state_error(op, "required input Q is missing"))?;
        let (k_shape, k) = self
            .tensor_f32(node.inputs.get(1).copied().flatten(), resolved, op, "K")?
            .ok_or_else(|| self.state_error(op, "required input K is missing"))?;
        let (v_shape, v) = self
            .tensor_f32(node.inputs.get(2).copied().flatten(), resolved, op, "V")?
            .ok_or_else(|| self.state_error(op, "required input V is missing"))?;
        if q_shape.len() != 4 || k_shape.len() != 4 || v_shape != k_shape {
            return Err(self.state_error(
                op,
                format!(
                    "Q/K/V must be Float32 rank-4 [B,T,H,D] with matching K/V shapes; got {q_shape:?}, {k_shape:?}, {v_shape:?}"
                ),
            ));
        }
        let (batch, q_len, heads, dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
        if k_shape[0] != batch || k_shape[2] != heads || k_shape[3] != dim {
            return Err(self.state_error(
                op,
                "prototype AttentionWithCache requires equal Q/K/V head counts and head dimensions",
            ));
        }
        let kv_len = k_shape[1];
        let q_positions = self.tensor_i64(
            node.inputs.get(5).copied().flatten(),
            resolved,
            op,
            "q_positions",
        )?;
        let kv_positions = self.tensor_i64(
            node.inputs.get(6).copied().flatten(),
            resolved,
            op,
            "kv_positions",
        )?;
        for (name, value, len) in [
            ("q_positions", q_positions.as_ref(), q_len),
            ("kv_positions", kv_positions.as_ref(), kv_len),
        ] {
            if let Some((shape, _)) = value
                && shape.as_slice() != [batch, len]
            {
                return Err(self.state_error(
                    op,
                    format!("{name} must have shape [{batch}, {len}], got {shape:?}"),
                ));
            }
        }
        let valid = self.tensor_i64(
            node.inputs.get(7).copied().flatten(),
            resolved,
            op,
            "nonpad_kv_seqlen",
        )?;
        if let Some((shape, _)) = &valid
            && shape.as_slice() != [batch]
        {
            return Err(self.state_error(
                op,
                format!("nonpad_kv_seqlen must have shape [{batch}], got {shape:?}"),
            ));
        }
        let old = self.opaque_input(
            node.inputs.get(4).copied().flatten(),
            node.id,
            OpaqueStateKind::KVCache,
            op,
        )?;
        let old = match old.as_ref().map(|v| v.payload.as_ref()) {
            Some(StatePayload::Kv(state)) => Some(state),
            Some(_) => return Err(self.state_error(op, "opaque payload kind is inconsistent")),
            None => None,
        };
        if let Some(state) = old
            && (state.batch, state.heads, state.head_dim) != (batch, heads, dim)
        {
            return Err(self.state_error(
                op,
                format!(
                    "cache metadata is [B={},H={},D={}], current tensors require [B={batch},H={heads},D={dim}]",
                    state.batch, state.heads, state.head_dim
                ),
            ));
        }

        let mut rows = Vec::with_capacity(batch);
        let mut old_lengths = Vec::with_capacity(batch);
        for b in 0..batch {
            let base_position = old.map(|s| Self::kv_row_len(&s.rows[b])).unwrap_or(0);
            old_lengths.push(base_position);
            let count = match &valid {
                Some((_, values)) => {
                    self.nonnegative_len(values[b], kv_len, op, &format!("nonpad_kv_seqlen[{b}]"))?
                }
                None => kv_len,
            };
            let mut records = Vec::with_capacity(count);
            for t in 0..count {
                let mut key = Vec::with_capacity(heads * dim);
                let mut value = Vec::with_capacity(heads * dim);
                let offset = ((b * kv_len + t) * heads) * dim;
                key.extend_from_slice(&k[offset..offset + heads * dim]);
                value.extend_from_slice(&v[offset..offset + heads * dim]);
                let position = kv_positions
                    .as_ref()
                    .map(|(_, positions)| positions[b * kv_len + t])
                    .unwrap_or((base_position + t) as i64);
                records.push(KvRecord {
                    position,
                    key,
                    value,
                });
            }
            rows.push(self.append_kv_row(old.map(|s| &s.rows[b]), records));
        }
        // Correct the paged logical lengths after append without flattening.
        for b in 0..batch {
            if let KvRows::Paged { len, .. } = &mut rows[b] {
                *len = old_lengths[b]
                    + match &valid {
                        Some((_, values)) => self.nonnegative_len(
                            values[b],
                            kv_len,
                            op,
                            &format!("nonpad_kv_seqlen[{b}]"),
                        )?,
                        None => kv_len,
                    };
            }
        }
        let present = KvState {
            batch,
            heads,
            head_dim: dim,
            rows: Arc::new(rows),
        };
        let mut output = vec![0.0f32; q.len()];
        let scale = 1.0 / (dim as f32).sqrt();
        for b in 0..batch {
            let row = &present.rows[b];
            for qt in 0..q_len {
                let q_position = match &q_positions {
                    Some((_, positions)) => positions[b * q_len + qt],
                    None => i64::try_from(old_lengths[b] + qt).map_err(|_| {
                        self.state_error(
                            op,
                            "implicit query position exceeds Int64; provide explicit q_positions",
                        )
                    })?,
                };
                for h in 0..heads {
                    let qoff = ((b * q_len + qt) * heads + h) * dim;
                    let mut max_score = f32::NEG_INFINITY;
                    let mut scores = Vec::new();
                    for index in 0..Self::kv_row_len(row) {
                        let record = Self::kv_row_record(row, index).unwrap();
                        if record.position > q_position {
                            continue;
                        }
                        let koff = h * dim;
                        let score = q[qoff..qoff + dim]
                            .iter()
                            .zip(&record.key[koff..koff + dim])
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                            * scale;
                        max_score = max_score.max(score);
                        scores.push((record, score));
                    }
                    let denom = scores
                        .iter()
                        .map(|(_, score)| (*score - max_score).exp())
                        .sum::<f32>();
                    for (record, score) in scores {
                        let weight = (score - max_score).exp() / denom;
                        for d in 0..dim {
                            output[qoff + d] += weight * record.value[h * dim + d];
                        }
                    }
                }
            }
        }
        self.write_f32(
            self.required_output(node, 0)?,
            q_shape,
            &output,
            resolved,
            external,
        )?;
        self.put_state(
            self.required_output(node, 1)?,
            node.id,
            OpaqueStateKind::KVCache,
            StatePayload::Kv(present),
        );
        Ok(())
    }

    fn linear_values(state: &LinearState) -> Vec<f32> {
        match &state.storage {
            LinearStorage::Contiguous(values) => (**values).clone(),
            LinearStorage::Blocked(blocks) => blocks
                .iter()
                .flat_map(|block| block.iter().copied())
                .collect(),
        }
    }

    fn exec_linear_attention_with_state(
        &mut self,
        node: &Node,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> Result<()> {
        let op = &node.op_type;
        if node
            .attr("allow_lossy_state")
            .and_then(Attribute::as_int)
            .unwrap_or(0)
            != 0
        {
            return Err(self.state_error(op, "allow_lossy_state=1 is not implemented"));
        }
        let (qs, q) = self.required_tensor_f32(
            node.inputs.first().copied().flatten(),
            resolved,
            op,
            "query",
        )?;
        let (ks, k) =
            self.required_tensor_f32(node.inputs.get(1).copied().flatten(), resolved, op, "key")?;
        let (vs, v) =
            self.required_tensor_f32(node.inputs.get(2).copied().flatten(), resolved, op, "value")?;
        if qs.len() != 3
            || ks.len() != 3
            || vs.len() != 3
            || qs[..2] != ks[..2]
            || qs[..2] != vs[..2]
        {
            return Err(self.state_error(op, format!("query/key/value must be rank-3 [B,T,H*D] with equal B,T; got {qs:?}, {ks:?}, {vs:?}")));
        }
        let heads = node
            .attr("kv_num_heads")
            .and_then(Attribute::as_int)
            .and_then(|v| usize::try_from(v).ok())
            .filter(|v| *v > 0)
            .ok_or_else(|| self.state_error(op, "kv_num_heads must be a positive integer"))?;
        let q_heads = node
            .attr("q_num_heads")
            .and_then(Attribute::as_int)
            .and_then(|v| usize::try_from(v).ok())
            .filter(|v| *v > 0)
            .ok_or_else(|| self.state_error(op, "q_num_heads must be a positive integer"))?;
        let (batch, seq) = (qs[0], qs[1]);
        if !qs[2].is_multiple_of(q_heads)
            || !ks[2].is_multiple_of(heads)
            || !vs[2].is_multiple_of(heads)
        {
            return Err(self.state_error(
                op,
                "hidden sizes must be divisible by their declared head counts",
            ));
        }
        let dk = qs[2] / q_heads;
        if ks[2] / heads != dk || !q_heads.is_multiple_of(heads) {
            return Err(self.state_error(op, "prototype requires q_num_heads to be a multiple of kv_num_heads and equal key dimensions"));
        }
        let dv = vs[2] / heads;
        let old = self.opaque_input(
            node.inputs.get(3).copied().flatten(),
            node.id,
            OpaqueStateKind::LinearAttentionState,
            op,
        )?;
        let mut state = match old.as_ref().map(|v| v.payload.as_ref()) {
            Some(StatePayload::Linear(state)) => {
                if (state.batch, state.heads, state.key_dim, state.value_dim)
                    != (batch, heads, dk, dv)
                {
                    return Err(self.state_error(
                        op,
                        "linear-attention state metadata does not match current inputs",
                    ));
                }
                Self::linear_values(state)
            }
            Some(_) => return Err(self.state_error(op, "opaque payload kind is inconsistent")),
            None => vec![0.0; batch * heads * dk * dv],
        };
        let decay =
            self.tensor_f32(node.inputs.get(4).copied().flatten(), resolved, op, "decay")?;
        let beta = self.tensor_f32(node.inputs.get(5).copied().flatten(), resolved, op, "beta")?;
        let sequence_lens = self.tensor_i64(
            node.inputs.get(6).copied().flatten(),
            resolved,
            op,
            "sequence_lens",
        )?;
        if let Some((shape, _)) = &sequence_lens
            && shape.as_slice() != [batch]
        {
            return Err(self.state_error(
                op,
                format!("sequence_lens must have shape [{batch}], got {shape:?}"),
            ));
        }
        let update_rule = node
            .attr("update_rule")
            .and_then(Attribute::as_str)
            .unwrap_or("gated_delta");
        if !matches!(update_rule, "linear" | "gated" | "delta" | "gated_delta") {
            return Err(self.state_error(op, format!("unsupported update_rule {update_rule:?}")));
        }
        let scale = node
            .attr("scale")
            .and_then(Attribute::as_float)
            .filter(|v| *v != 0.0)
            .unwrap_or_else(|| 1.0 / (dk as f32).sqrt());
        let mut output = vec![0.0; batch * seq * q_heads * dv];
        for b in 0..batch {
            let valid = match &sequence_lens {
                Some((_, values)) => {
                    self.nonnegative_len(values[b], seq, op, &format!("sequence_lens[{b}]"))?
                }
                None => seq,
            };
            for t in 0..valid {
                for h in 0..heads {
                    let sbase = ((b * heads + h) * dk) * dv;
                    if matches!(update_rule, "gated" | "gated_delta") {
                        let gate = decay
                            .as_ref()
                            .ok_or_else(|| {
                                self.state_error(op, "decay is required by update_rule")
                            })?
                            .1[(b * seq + t) * heads + h]
                            .exp();
                        for x in &mut state[sbase..sbase + dk * dv] {
                            *x *= gate;
                        }
                    }
                    let kbase = (b * seq + t) * heads * dk + h * dk;
                    let vbase = (b * seq + t) * heads * dv + h * dv;
                    if matches!(update_rule, "delta" | "gated_delta") {
                        let gate = beta
                            .as_ref()
                            .ok_or_else(|| self.state_error(op, "beta is required by update_rule"))?
                            .1[(b * seq + t) * heads + h];
                        for j in 0..dv {
                            let retrieval = (0..dk)
                                .map(|i| state[sbase + i * dv + j] * k[kbase + i])
                                .sum::<f32>();
                            let delta = gate * (v[vbase + j] - retrieval);
                            for i in 0..dk {
                                state[sbase + i * dv + j] += k[kbase + i] * delta;
                            }
                        }
                    } else {
                        for i in 0..dk {
                            for j in 0..dv {
                                state[sbase + i * dv + j] += k[kbase + i] * v[vbase + j];
                            }
                        }
                    }
                }
                for qh in 0..q_heads {
                    let h = qh / (q_heads / heads);
                    let qbase = (b * seq + t) * q_heads * dk + qh * dk;
                    let obase = (b * seq + t) * q_heads * dv + qh * dv;
                    let sbase = ((b * heads + h) * dk) * dv;
                    for j in 0..dv {
                        output[obase + j] = scale
                            * (0..dk)
                                .map(|i| q[qbase + i] * state[sbase + i * dv + j])
                                .sum::<f32>();
                    }
                }
            }
        }
        let storage = match self.state_config.linear_attention {
            crate::LinearAttentionStateMode::Contiguous => {
                LinearStorage::Contiguous(Arc::new(state))
            }
            crate::LinearAttentionStateMode::Blocked => {
                let block = dk * dv;
                LinearStorage::Blocked(Arc::new(
                    state.chunks(block).map(|v| Arc::new(v.to_vec())).collect(),
                ))
            }
        };
        self.write_f32(
            self.required_output(node, 0)?,
            vec![batch, seq, q_heads * dv],
            &output,
            resolved,
            external,
        )?;
        self.put_state(
            self.required_output(node, 1)?,
            node.id,
            OpaqueStateKind::LinearAttentionState,
            StatePayload::Linear(LinearState {
                batch,
                heads,
                key_dim: dk,
                value_dim: dv,
                storage,
            }),
        );
        Ok(())
    }

    fn conv_values(state: &ConvState) -> Vec<f32> {
        match &state.storage {
            ConvStorage::Contiguous(values) => (**values).clone(),
            ConvStorage::Ring { data, start } => {
                let rows = state.batch * state.channels;
                let mut out = Vec::with_capacity(data.len());
                for row in 0..rows {
                    let base = row * state.history;
                    for i in 0..state.history {
                        out.push(data[base + (start + i) % state.history]);
                    }
                }
                out
            }
        }
    }

    fn exec_causal_conv_with_state(
        &mut self,
        node: &Node,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> Result<()> {
        let op = &node.op_type;
        let (xs, x) = self.required_tensor_f32(
            node.inputs.first().copied().flatten(),
            resolved,
            op,
            "input",
        )?;
        let (ws, weight) = self.required_tensor_f32(
            node.inputs.get(1).copied().flatten(),
            resolved,
            op,
            "weight",
        )?;
        let bias = self.tensor_f32(node.inputs.get(2).copied().flatten(), resolved, op, "bias")?;
        if xs.len() != 3 || ws.len() != 3 || ws[1] != 1 || ws[0] != xs[1] {
            return Err(self.state_error(
                op,
                format!("input must be [B,C,L] and weight [C,1,K], got {xs:?}, {ws:?}"),
            ));
        }
        let (batch, channels, len, kernel) = (xs[0], xs[1], xs[2], ws[2]);
        let dilation = node
            .attr("dilation")
            .and_then(Attribute::as_int)
            .unwrap_or(1);
        let dilation = usize::try_from(dilation)
            .ok()
            .filter(|v| *v > 0)
            .ok_or_else(|| self.state_error(op, "dilation must be a positive integer"))?;
        let history = kernel.saturating_sub(1) * dilation;
        if let Some((shape, _)) = &bias
            && shape.as_slice() != [channels]
        {
            return Err(self.state_error(
                op,
                format!("bias must have shape [{channels}], got {shape:?}"),
            ));
        }
        let sequence_lens = self.tensor_i64(
            node.inputs.get(4).copied().flatten(),
            resolved,
            op,
            "sequence_lens",
        )?;
        if let Some((shape, _)) = &sequence_lens
            && shape.as_slice() != [batch]
        {
            return Err(self.state_error(
                op,
                format!("sequence_lens must have shape [{batch}], got {shape:?}"),
            ));
        }
        let old = self.opaque_input(
            node.inputs.get(3).copied().flatten(),
            node.id,
            OpaqueStateKind::CausalConvState,
            op,
        )?;
        let old_values = match old.as_ref().map(|v| v.payload.as_ref()) {
            Some(StatePayload::Conv(state)) => {
                if (state.batch, state.channels, state.history) != (batch, channels, history) {
                    return Err(self.state_error(
                        op,
                        "causal-conv state metadata does not match current inputs",
                    ));
                }
                Self::conv_values(state)
            }
            Some(_) => return Err(self.state_error(op, "opaque payload kind is inconsistent")),
            None => vec![0.0; batch * channels * history],
        };
        let mut output = vec![0.0; x.len()];
        let mut present = vec![0.0; batch * channels * history];
        let activation = node
            .attr("activation")
            .and_then(Attribute::as_str)
            .unwrap_or("none");
        if !matches!(activation, "none" | "silu" | "swish") {
            return Err(self.state_error(
                op,
                format!("activation must be none, silu, or swish; got {activation:?}"),
            ));
        }
        for b in 0..batch {
            let valid = match &sequence_lens {
                Some((_, values)) => {
                    self.nonnegative_len(values[b], len, op, &format!("sequence_lens[{b}]"))?
                }
                None => len,
            };
            for c in 0..channels {
                let old_base = (b * channels + c) * history;
                let x_base = (b * channels + c) * len;
                for t in 0..valid {
                    let mut acc = bias.as_ref().map(|(_, v)| v[c]).unwrap_or(0.0);
                    for kidx in 0..kernel {
                        let logical = history + t;
                        let distance = (kernel - 1 - kidx) * dilation;
                        let value = if logical >= distance {
                            let pos = logical - distance;
                            if pos < history {
                                old_values[old_base + pos]
                            } else {
                                x[x_base + pos - history]
                            }
                        } else {
                            0.0
                        };
                        acc += weight[c * kernel + kidx] * value;
                    }
                    output[x_base + t] = if activation == "none" {
                        acc
                    } else {
                        acc / (1.0 + (-acc).exp())
                    };
                }
                for p in 0..history {
                    let source = valid + p;
                    present[old_base + p] = if source < history {
                        old_values[old_base + source]
                    } else {
                        x[x_base + source - history]
                    };
                }
            }
        }
        let storage = match self.state_config.causal_conv {
            crate::CausalConvStateMode::Contiguous => ConvStorage::Contiguous(Arc::new(present)),
            crate::CausalConvStateMode::Ring => ConvStorage::Ring {
                data: Arc::new(present),
                start: 0,
            },
        };
        self.write_f32(
            self.required_output(node, 0)?,
            xs,
            &output,
            resolved,
            external,
        )?;
        self.put_state(
            self.required_output(node, 1)?,
            node.id,
            OpaqueStateKind::CausalConvState,
            StatePayload::Conv(ConvState {
                batch,
                channels,
                history,
                storage,
            }),
        );
        Ok(())
    }
}
