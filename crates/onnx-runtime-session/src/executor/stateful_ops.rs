use super::*;

struct AttentionLayout {
    batch: usize,
    q_offsets: Vec<usize>,
    kv_offsets: Vec<usize>,
    q_capacity: Option<usize>,
    kv_capacity: Option<usize>,
    packed: bool,
}

impl AttentionLayout {
    fn q_len(&self, batch: usize) -> usize {
        self.q_offsets[batch + 1] - self.q_offsets[batch]
    }

    fn kv_len(&self, batch: usize) -> usize {
        self.kv_offsets[batch + 1] - self.kv_offsets[batch]
    }

    fn q_source(&self, batch: usize, token: usize) -> usize {
        self.q_capacity
            .map_or(self.q_offsets[batch] + token, |capacity| {
                batch * capacity + token
            })
    }

    fn kv_source(&self, batch: usize, token: usize) -> usize {
        self.kv_capacity
            .map_or(self.kv_offsets[batch] + token, |capacity| {
                batch * capacity + token
            })
    }
}

struct CompactVisibility {
    q_block_size: usize,
    k_block_size: usize,
    heads: usize,
    row_bases: Vec<usize>,
    row_offsets: Vec<usize>,
    key_blocks: Vec<usize>,
    partial_indices: Vec<usize>,
    partial_mask: Vec<u64>,
    mask_words: usize,
    bias: Option<Vec<f32>>,
}

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
        let elements = shape.iter().try_fold(1usize, |count, dim| {
            count.checked_mul(*dim).ok_or_else(|| {
                self.state_error(
                    op,
                    format!("input {name:?} shape {shape:?} overflows its element count"),
                )
            })
        })?;
        let byte_len = elements.checked_mul(4).ok_or_else(|| {
            self.state_error(
                op,
                format!("input {name:?} shape {shape:?} overflows its byte count"),
            )
        })?;
        let bytes = host_bytes(buffer);
        if bytes.len() < byte_len {
            return Err(self.state_error(
                op,
                format!(
                    "input {name:?} buffer has {} bytes, but shape {shape:?} requires {byte_len}",
                    bytes.len()
                ),
            ));
        }
        let values = onnx_runtime_ir::read_vec_le::<f32>(&bytes[..byte_len]).map_err(|error| {
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
        let elements = shape.iter().try_fold(1usize, |count, dim| {
            count.checked_mul(*dim).ok_or_else(|| {
                self.state_error(
                    op,
                    format!("input {name:?} shape {shape:?} overflows its element count"),
                )
            })
        })?;
        let element_width = match dtype {
            DataType::Int64 => 8,
            DataType::Int32 => 4,
            _ => 0,
        };
        let byte_len = elements.checked_mul(element_width).ok_or_else(|| {
            self.state_error(
                op,
                format!("input {name:?} shape {shape:?} overflows its byte count"),
            )
        })?;
        let buffer_bytes = host_bytes(buffer);
        if buffer_bytes.len() < byte_len {
            return Err(self.state_error(
                op,
                format!(
                    "input {name:?} buffer has {} bytes, but shape {shape:?} requires {byte_len}",
                    buffer_bytes.len()
                ),
            ));
        }
        let bytes = &buffer_bytes[..byte_len];
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

    fn tensor_u64(
        &self,
        input: Option<ValueId>,
        resolved: &HashMap<ValueId, Vec<usize>>,
        op: &str,
        name: &str,
    ) -> Result<Option<(Vec<usize>, Vec<u64>)>> {
        let Some(id) = input else { return Ok(None) };
        let dtype = self.value_dtypes[&id];
        if dtype != DataType::Uint64 {
            return Err(self.state_error(
                op,
                format!("input {name:?} has dtype {dtype:?}; expected Uint64"),
            ));
        }
        let shape = resolved.get(&id).cloned().ok_or_else(|| {
            self.state_error(op, format!("input {name:?} has no resolved runtime shape"))
        })?;
        let buffer = self.buffers.get(&id).ok_or_else(|| {
            self.state_error(op, format!("input {name:?} has no live tensor buffer"))
        })?;
        let elements = shape.iter().try_fold(1usize, |count, dim| {
            count.checked_mul(*dim).ok_or_else(|| {
                self.state_error(
                    op,
                    format!("input {name:?} shape {shape:?} overflows its element count"),
                )
            })
        })?;
        let byte_len = elements.checked_mul(8).ok_or_else(|| {
            self.state_error(
                op,
                format!("input {name:?} shape {shape:?} overflows its byte count"),
            )
        })?;
        let buffer_bytes = host_bytes(buffer);
        if buffer_bytes.len() < byte_len {
            return Err(self.state_error(
                op,
                format!(
                    "input {name:?} buffer has {} bytes, but shape {shape:?} requires {byte_len}",
                    buffer_bytes.len()
                ),
            ));
        }
        let values = buffer_bytes[..byte_len]
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
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

    fn attention_offsets(
        &self,
        input: Option<ValueId>,
        resolved: &HashMap<ValueId, Vec<usize>>,
        op: &str,
        name: &str,
        total: usize,
    ) -> Result<Option<(DataType, Vec<usize>)>> {
        let Some(id) = input else { return Ok(None) };
        let dtype = self.value_dtypes[&id];
        let (shape, values) = self.tensor_i64(Some(id), resolved, op, name)?.unwrap();
        if shape.len() != 1 || values.is_empty() {
            return Err(self.state_error(
                op,
                format!("{name} must be rank-1 [B+1] with at least one element, got {shape:?}"),
            ));
        }
        let mut offsets = Vec::with_capacity(values.len());
        for (index, value) in values.into_iter().enumerate() {
            let offset = usize::try_from(value).map_err(|_| {
                self.state_error(
                    op,
                    format!("{name}[{index}]={value} is negative; offsets must be nonnegative"),
                )
            })?;
            if let Some(previous) = offsets.last()
                && offset < *previous
            {
                return Err(self.state_error(
                    op,
                    format!(
                        "{name}[{index}]={offset} is less than the previous offset {previous}; offsets must be monotonic"
                    ),
                ));
            }
            offsets.push(offset);
        }
        if offsets[0] != 0 {
            return Err(self.state_error(
                op,
                format!("{name}[0]={}; offsets must start at zero", offsets[0]),
            ));
        }
        if *offsets.last().unwrap() != total {
            return Err(self.state_error(
                op,
                format!(
                    "{name} ends at {}, but the corresponding tensor has {total} tokens",
                    offsets.last().unwrap()
                ),
            ));
        }
        Ok(Some((dtype, offsets)))
    }

    fn positive_attr(&self, node: &Node, name: &str, default: usize) -> Result<usize> {
        let value = node
            .attr(name)
            .and_then(Attribute::as_int)
            .unwrap_or(default as i64);
        usize::try_from(value)
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                self.state_error(
                    &node.op_type,
                    format!("{name}={value} is invalid; provide a positive integer"),
                )
            })
    }

    fn compact_visibility(
        &self,
        node: &Node,
        resolved: &HashMap<ValueId, Vec<usize>>,
        layout: &AttentionLayout,
        query_heads: usize,
        key_lengths: &[usize],
    ) -> Result<Option<CompactVisibility>> {
        let op = &node.op_type;
        let row_input = node.inputs.get(10).copied().flatten();
        let block_input = node.inputs.get(11).copied().flatten();
        if row_input.is_some() != block_input.is_some() {
            return Err(self.state_error(
                op,
                "visibility_row_offsets (slot 10) and visibility_key_blocks (slot 11) must be provided together",
            ));
        }
        let Some(row_id) = row_input else {
            for (slot, name) in [
                (12, "visibility_partial_block_indices"),
                (13, "visibility_partial_block_mask"),
                (14, "attn_bias"),
            ] {
                if node.inputs.get(slot).is_some_and(Option::is_some) {
                    return Err(self.state_error(
                        op,
                        format!("{name} requires compact visibility inputs in slots 10 and 11"),
                    ));
                }
            }
            return Ok(None);
        };
        let block_id = block_input.unwrap();
        let row_dtype = self.value_dtypes[&row_id];
        let block_dtype = self.value_dtypes[&block_id];
        if row_dtype != block_dtype {
            return Err(self.state_error(
                op,
                format!(
                    "visibility_row_offsets and visibility_key_blocks must have the same Int32/Int64 dtype, got {row_dtype:?} and {block_dtype:?}"
                ),
            ));
        }
        let (row_shape, row_values) = self
            .tensor_i64(Some(row_id), resolved, op, "visibility_row_offsets")?
            .unwrap();
        let (block_shape, block_values) = self
            .tensor_i64(Some(block_id), resolved, op, "visibility_key_blocks")?
            .unwrap();
        if row_shape.len() != 1 || block_shape.len() != 1 {
            return Err(self.state_error(
                op,
                format!(
                    "compact visibility CSR inputs must be rank-1; got row offsets {row_shape:?} and key blocks {block_shape:?}"
                ),
            ));
        }
        let q_block_size = self.positive_attr(node, "visibility_q_block_size", 1)?;
        let k_block_size = self.positive_attr(node, "visibility_k_block_size", 64)?;
        let heads = self.positive_attr(node, "visibility_num_heads", 1)?;
        if heads != 1 && heads != query_heads {
            return Err(self.state_error(
                op,
                format!(
                    "visibility_num_heads must be 1 or the query head count {query_heads}, got {heads}"
                ),
            ));
        }
        let mut row_bases = Vec::with_capacity(layout.batch + 1);
        row_bases.push(0);
        for batch in 0..layout.batch {
            let blocks = layout.q_len(batch).div_ceil(q_block_size);
            row_bases.push(row_bases.last().unwrap() + heads * blocks);
        }
        let expected_rows = *row_bases.last().unwrap();
        if row_values.len() != expected_rows + 1 {
            return Err(self.state_error(
                op,
                format!(
                    "visibility_row_offsets must contain {expected_rows}+1 entries for batch -> visibility head -> local query block order, got {}",
                    row_values.len()
                ),
            ));
        }
        let mut row_offsets = Vec::with_capacity(row_values.len());
        for (index, value) in row_values.into_iter().enumerate() {
            let value = usize::try_from(value).map_err(|_| {
                self.state_error(op, format!("visibility_row_offsets[{index}] is negative"))
            })?;
            if let Some(previous) = row_offsets.last()
                && value < *previous
            {
                return Err(self.state_error(
                    op,
                    format!(
                        "visibility_row_offsets[{index}]={value} is less than the previous offset {previous}"
                    ),
                ));
            }
            row_offsets.push(value);
        }
        if row_offsets[0] != 0 || *row_offsets.last().unwrap() != block_values.len() {
            return Err(self.state_error(
                op,
                format!(
                    "visibility CSR must start at 0 and end at the {} active key blocks; got endpoints {} and {}",
                    block_values.len(),
                    row_offsets[0],
                    row_offsets.last().unwrap()
                ),
            ));
        }
        let mut key_blocks = Vec::with_capacity(block_values.len());
        for (index, value) in block_values.into_iter().enumerate() {
            key_blocks.push(usize::try_from(value).map_err(|_| {
                self.state_error(
                    op,
                    format!("visibility_key_blocks[{index}]={value} is negative"),
                )
            })?);
        }
        for batch in 0..layout.batch {
            let query_blocks = layout.q_len(batch).div_ceil(q_block_size);
            let key_block_limit = key_lengths[batch].div_ceil(k_block_size);
            for head in 0..heads {
                for query_block in 0..query_blocks {
                    let row = row_bases[batch] + head * query_blocks + query_block;
                    let entries = &key_blocks[row_offsets[row]..row_offsets[row + 1]];
                    if entries.iter().any(|block| *block >= key_block_limit) {
                        return Err(self.state_error(
                            op,
                            format!(
                                "visibility row {row} lists a key block outside 0..{key_block_limit} for batch {batch}"
                            ),
                        ));
                    }
                    if entries.windows(2).any(|pair| pair[0] >= pair[1]) {
                        return Err(self.state_error(
                            op,
                            format!("visibility row {row} key blocks must be sorted and unique"),
                        ));
                    }
                }
            }
        }
        let partial_index_input = node.inputs.get(12).copied().flatten();
        let partial_mask_input = node.inputs.get(13).copied().flatten();
        if partial_index_input.is_some() != partial_mask_input.is_some() {
            return Err(self.state_error(
                op,
                "visibility_partial_block_indices (slot 12) and visibility_partial_block_mask (slot 13) must be provided together",
            ));
        }
        let mask_words = k_block_size.div_ceil(64);
        let (partial_indices, partial_mask) = match (partial_index_input, partial_mask_input) {
            (Some(index_id), Some(mask_id)) => {
                if self.value_dtypes[&index_id] != row_dtype {
                    return Err(self.state_error(
                        op,
                        "visibility_partial_block_indices must use the same Int32/Int64 dtype as the visibility CSR",
                    ));
                }
                let (index_shape, index_values) = self
                    .tensor_i64(
                        Some(index_id),
                        resolved,
                        op,
                        "visibility_partial_block_indices",
                    )?
                    .unwrap();
                let (mask_shape, mask_values) = self
                    .tensor_u64(Some(mask_id), resolved, op, "visibility_partial_block_mask")?
                    .unwrap();
                if index_shape.len() != 1 {
                    return Err(self.state_error(
                        op,
                        format!(
                            "visibility_partial_block_indices must be rank-1 [P], got {index_shape:?}"
                        ),
                    ));
                }
                let mut indices = Vec::with_capacity(index_values.len());
                for (index, value) in index_values.into_iter().enumerate() {
                    let active = usize::try_from(value).map_err(|_| {
                        self.state_error(
                            op,
                            format!(
                                "visibility_partial_block_indices[{index}]={value} is negative"
                            ),
                        )
                    })?;
                    if active >= key_blocks.len() {
                        return Err(self.state_error(
                            op,
                            format!(
                                "visibility_partial_block_indices[{index}]={active} is outside the {} active entries",
                                key_blocks.len()
                            ),
                        ));
                    }
                    indices.push(active);
                }
                if indices.windows(2).any(|pair| pair[0] >= pair[1]) {
                    return Err(self.state_error(
                        op,
                        "visibility_partial_block_indices must be sorted and unique",
                    ));
                }
                let expected_shape = [indices.len(), q_block_size, mask_words];
                if mask_shape.as_slice() != expected_shape {
                    return Err(self.state_error(
                        op,
                        format!(
                            "visibility_partial_block_mask must have shape {expected_shape:?}, got {mask_shape:?}"
                        ),
                    ));
                }
                (indices, mask_values)
            }
            (None, None) => (Vec::new(), Vec::new()),
            _ => unreachable!(),
        };
        let bias = self.tensor_f32(
            node.inputs.get(14).copied().flatten(),
            resolved,
            op,
            "attn_bias",
        )?;
        let bias = match bias {
            Some((shape, values)) => {
                let expected = [key_blocks.len(), q_block_size, k_block_size];
                if shape.as_slice() != expected {
                    return Err(self.state_error(
                        op,
                        format!("attn_bias must have shape {expected:?}, got {shape:?}"),
                    ));
                }
                Some(values)
            }
            None => None,
        };
        Ok(Some(CompactVisibility {
            q_block_size,
            k_block_size,
            heads,
            row_bases,
            row_offsets,
            key_blocks,
            partial_indices,
            partial_mask,
            mask_words,
            bias,
        }))
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
                "legacy padded attn_mask is not implemented because the proposed ABI permits both boolean visibility and additive bias; use compact visibility slots 10-14 or omit input slot 3",
            ));
        }
        let (q_shape, q) =
            self.required_tensor_f32(node.inputs.first().copied().flatten(), resolved, op, "Q")?;
        let (k_shape, k) =
            self.required_tensor_f32(node.inputs.get(1).copied().flatten(), resolved, op, "K")?;
        let (v_shape, v) =
            self.required_tensor_f32(node.inputs.get(2).copied().flatten(), resolved, op, "V")?;
        let q_offset_input = node.inputs.get(8).copied().flatten();
        let kv_offset_input = node.inputs.get(9).copied().flatten();
        if q_offset_input.is_some() != kv_offset_input.is_some() {
            return Err(self.state_error(
                op,
                "q_sequence_offsets (slot 8) and kv_sequence_offsets (slot 9) must be provided together",
            ));
        }
        let packed = q_offset_input.is_some();
        let (layout, query_heads, kv_heads, key_dim, value_dim, output_shape) = if packed {
            if q_shape.len() != 2 || k_shape.len() != 2 || v_shape.len() != 2 {
                return Err(self.state_error(
                    op,
                    format!(
                        "packed Q/K/V must be rank-2 token-major tensors; got {q_shape:?}, {k_shape:?}, {v_shape:?}"
                    ),
                ));
            }
            if k_shape[0] != v_shape[0] {
                return Err(self.state_error(
                    op,
                    format!(
                        "packed K and V token totals must match, got {} and {}",
                        k_shape[0], v_shape[0]
                    ),
                ));
            }
            let query_heads = self.positive_attr(node, "q_num_heads", 0)?;
            let kv_heads = self.positive_attr(node, "kv_num_heads", 0)?;
            if !q_shape[1].is_multiple_of(query_heads)
                || !k_shape[1].is_multiple_of(kv_heads)
                || !v_shape[1].is_multiple_of(kv_heads)
            {
                return Err(self.state_error(
                    op,
                    format!(
                        "packed hidden widths must be divisible by q_num_heads={query_heads} and kv_num_heads={kv_heads}; got Q/K/V widths {}, {}, {}",
                        q_shape[1], k_shape[1], v_shape[1]
                    ),
                ));
            }
            let key_dim = q_shape[1] / query_heads;
            let value_dim = v_shape[1] / kv_heads;
            if key_dim == 0
                || value_dim == 0
                || k_shape[1] / kv_heads != key_dim
                || !query_heads.is_multiple_of(kv_heads)
            {
                return Err(self.state_error(
                    op,
                    "packed attention requires positive key/value head dimensions, equal Q/K head dimensions, and q_num_heads to be a multiple of kv_num_heads",
                ));
            }
            let (q_dtype, q_offsets) = self
                .attention_offsets(
                    q_offset_input,
                    resolved,
                    op,
                    "q_sequence_offsets",
                    q_shape[0],
                )?
                .unwrap();
            let (kv_dtype, kv_offsets) = self
                .attention_offsets(
                    kv_offset_input,
                    resolved,
                    op,
                    "kv_sequence_offsets",
                    k_shape[0],
                )?
                .unwrap();
            if q_dtype != kv_dtype {
                return Err(self.state_error(
                    op,
                    format!(
                        "q_sequence_offsets and kv_sequence_offsets must have the same Int32/Int64 dtype, got {q_dtype:?} and {kv_dtype:?}"
                    ),
                ));
            }
            if q_offsets.len() != kv_offsets.len() {
                return Err(self.state_error(
                    op,
                    format!(
                        "packed query and KV offsets must describe the same batch, got lengths {} and {}",
                        q_offsets.len(),
                        kv_offsets.len()
                    ),
                ));
            }
            let batch = q_offsets.len() - 1;
            (
                AttentionLayout {
                    batch,
                    q_offsets,
                    kv_offsets,
                    q_capacity: None,
                    kv_capacity: None,
                    packed: true,
                },
                query_heads,
                kv_heads,
                key_dim,
                value_dim,
                vec![q_shape[0], query_heads * value_dim],
            )
        } else {
            if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
                return Err(self.state_error(
                    op,
                    format!(
                        "padded Q/K/V must be rank-4 [B,T,H,D]; got {q_shape:?}, {k_shape:?}, {v_shape:?}"
                    ),
                ));
            }
            let (batch, q_capacity, query_heads, key_dim) =
                (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
            let (kv_capacity, kv_heads, value_dim) = (k_shape[1], k_shape[2], v_shape[3]);
            if k_shape[0] != batch
                || v_shape[0] != batch
                || v_shape[1] != kv_capacity
                || v_shape[2] != kv_heads
                || k_shape[3] != key_dim
                || key_dim == 0
                || value_dim == 0
                || !query_heads.is_multiple_of(kv_heads)
            {
                return Err(self.state_error(
                    op,
                    format!(
                        "padded attention requires matching batches/token capacities, equal Q/K head dimensions, matching K/V heads, and Q heads divisible by KV heads; got {q_shape:?}, {k_shape:?}, {v_shape:?}"
                    ),
                ));
            }
            for (name, declared, actual) in [
                (
                    "q_num_heads",
                    node.attr("q_num_heads").and_then(Attribute::as_int),
                    query_heads,
                ),
                (
                    "kv_num_heads",
                    node.attr("kv_num_heads").and_then(Attribute::as_int),
                    kv_heads,
                ),
            ] {
                if let Some(declared) = declared
                    && usize::try_from(declared).ok() != Some(actual)
                {
                    return Err(self.state_error(
                        op,
                        format!(
                            "{name}={declared} does not match the padded tensor head axis {actual}"
                        ),
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
            let q_offsets = (0..=batch)
                .map(|index| index * q_capacity)
                .collect::<Vec<_>>();
            let mut kv_offsets = Vec::with_capacity(batch + 1);
            kv_offsets.push(0);
            for b in 0..batch {
                let count = match &valid {
                    Some((_, values)) => self.nonnegative_len(
                        values[b],
                        kv_capacity,
                        op,
                        &format!("nonpad_kv_seqlen[{b}]"),
                    )?,
                    None => kv_capacity,
                };
                kv_offsets.push(kv_offsets.last().unwrap() + count);
            }
            (
                AttentionLayout {
                    batch,
                    q_offsets,
                    kv_offsets,
                    q_capacity: Some(q_capacity),
                    kv_capacity: Some(kv_capacity),
                    packed: false,
                },
                query_heads,
                kv_heads,
                key_dim,
                value_dim,
                vec![batch, q_capacity, query_heads, value_dim],
            )
        };
        if packed && node.inputs.get(7).is_some_and(Option::is_some) {
            return Err(self.state_error(
                op,
                "nonpad_kv_seqlen is padded-only; packed KV lengths come from kv_sequence_offsets",
            ));
        }
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
        for (name, value, total, padded_capacity) in [
            (
                "q_positions",
                q_positions.as_ref(),
                q_shape[0],
                layout.q_capacity,
            ),
            (
                "kv_positions",
                kv_positions.as_ref(),
                k_shape[0],
                layout.kv_capacity,
            ),
        ] {
            if let Some((shape, _)) = value {
                let expected = if layout.packed {
                    vec![total]
                } else {
                    vec![layout.batch, padded_capacity.unwrap()]
                };
                if *shape != expected {
                    return Err(self.state_error(
                        op,
                        format!("{name} must have shape {expected:?}, got {shape:?}"),
                    ));
                }
            }
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
            && (state.batch, state.heads, state.key_dim, state.value_dim)
                != (layout.batch, kv_heads, key_dim, value_dim)
        {
            return Err(self.state_error(
                op,
                format!(
                    "cache metadata is [B={},Hkv={},Dk={},Dv={}], current tensors require [B={},Hkv={kv_heads},Dk={key_dim},Dv={value_dim}]",
                    state.batch, state.heads, state.key_dim, state.value_dim, layout.batch
                ),
            ));
        }
        let key_lengths = (0..layout.batch)
            .map(|batch| {
                old.map(|state| Self::kv_row_len(&state.rows[batch]))
                    .unwrap_or(0)
                    + layout.kv_len(batch)
            })
            .collect::<Vec<_>>();
        let visibility =
            self.compact_visibility(node, resolved, &layout, query_heads, &key_lengths)?;

        let mut rows = Vec::with_capacity(layout.batch);
        let mut old_lengths = Vec::with_capacity(layout.batch);
        for b in 0..layout.batch {
            let base_position = old.map(|s| Self::kv_row_len(&s.rows[b])).unwrap_or(0);
            old_lengths.push(base_position);
            let count = layout.kv_len(b);
            let mut records = Vec::with_capacity(count);
            for t in 0..count {
                let source = layout.kv_source(b, t);
                let key_offset = source * kv_heads * key_dim;
                let value_offset = source * kv_heads * value_dim;
                let key = k[key_offset..key_offset + kv_heads * key_dim].to_vec();
                let value = v[value_offset..value_offset + kv_heads * value_dim].to_vec();
                let position = kv_positions
                    .as_ref()
                    .map(|(_, positions)| positions[source])
                    .unwrap_or((base_position + t) as i64);
                records.push(KvRecord {
                    position,
                    key,
                    value,
                });
            }
            rows.push(self.append_kv_row(old.map(|s| &s.rows[b]), records));
        }
        let present = KvState {
            batch: layout.batch,
            heads: kv_heads,
            key_dim,
            value_dim,
            rows: Arc::new(rows),
        };
        let output_len = layout.q_offsets[layout.batch] * query_heads * value_dim;
        let mut output = vec![0.0f32; output_len];
        let scale = 1.0 / (key_dim as f32).sqrt();
        let left_window = node
            .attr("left_window_size")
            .and_then(Attribute::as_int)
            .unwrap_or(-1);
        let right_window = node
            .attr("right_window_size")
            .and_then(Attribute::as_int)
            .unwrap_or(-1);
        if left_window < -1 || right_window < -1 {
            return Err(self.state_error(
                op,
                format!(
                    "left_window_size and right_window_size must be -1 or nonnegative, got {left_window} and {right_window}"
                ),
            ));
        }
        for b in 0..layout.batch {
            let row = &present.rows[b];
            for qt in 0..layout.q_len(b) {
                let q_source = layout.q_source(b, qt);
                let q_position = match &q_positions {
                    Some((_, positions)) => positions[q_source],
                    None => i64::try_from(old_lengths[b] + qt).map_err(|_| {
                        self.state_error(
                            op,
                            "implicit query position exceeds Int64; provide explicit q_positions",
                        )
                    })?,
                };
                for h in 0..query_heads {
                    self.attention_traversal_stats.logical_key_candidates +=
                        Self::kv_row_len(row) as u64;
                    let qoff = (q_source * query_heads + h) * key_dim;
                    let output_token = layout.q_offsets[b] + qt;
                    let output_offset = (output_token * query_heads + h) * value_dim;
                    let kv_head = h / (query_heads / kv_heads);
                    let mut max_score = f32::NEG_INFINITY;
                    let mut scores = Vec::new();
                    let mut evaluated = 0u64;
                    let mut visit = |index: usize, bias: f32| {
                        let record = Self::kv_row_record(row, index).unwrap();
                        let lower_ok = left_window < 0
                            || record.position >= q_position.saturating_sub(left_window);
                        let window_upper_ok = right_window < 0
                            || record.position <= q_position.saturating_add(right_window);
                        if record.position > q_position || !lower_ok || !window_upper_ok {
                            return;
                        }
                        evaluated += 1;
                        let koff = kv_head * key_dim;
                        let score = q[qoff..qoff + key_dim]
                            .iter()
                            .zip(&record.key[koff..koff + key_dim])
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                            * scale
                            + bias;
                        max_score = max_score.max(score);
                        scores.push((record, score));
                    };
                    if let Some(visibility) = &visibility {
                        let query_blocks = layout.q_len(b).div_ceil(visibility.q_block_size);
                        let visibility_head = if visibility.heads == 1 { 0 } else { h };
                        let query_block = qt / visibility.q_block_size;
                        let query_in_block = qt % visibility.q_block_size;
                        let csr_row =
                            visibility.row_bases[b] + visibility_head * query_blocks + query_block;
                        for active in
                            visibility.row_offsets[csr_row]..visibility.row_offsets[csr_row + 1]
                        {
                            let key_block = visibility.key_blocks[active];
                            let start = key_block * visibility.k_block_size;
                            let end = (start + visibility.k_block_size).min(Self::kv_row_len(row));
                            let partial = visibility.partial_indices.binary_search(&active).ok();
                            if partial.is_some() {
                                self.attention_traversal_stats.partial_block_visits += 1;
                            } else {
                                self.attention_traversal_stats.full_block_visits += 1;
                            }
                            for index in start..end {
                                let key_in_block = index - start;
                                if let Some(partial) = partial {
                                    let mask_index = (partial * visibility.q_block_size
                                        + query_in_block)
                                        * visibility.mask_words
                                        + key_in_block / 64;
                                    if (visibility.partial_mask[mask_index]
                                        & (1u64 << (key_in_block % 64)))
                                        == 0
                                    {
                                        continue;
                                    }
                                }
                                let bias = visibility.bias.as_ref().map_or(0.0, |bias| {
                                    bias[(active * visibility.q_block_size + query_in_block)
                                        * visibility.k_block_size
                                        + key_in_block]
                                });
                                visit(index, bias);
                            }
                        }
                    } else {
                        for index in 0..Self::kv_row_len(row) {
                            visit(index, 0.0);
                        }
                    }
                    drop(visit);
                    self.attention_traversal_stats.evaluated_key_candidates += evaluated;
                    if scores.is_empty() {
                        continue;
                    }
                    let denom = scores
                        .iter()
                        .map(|(_, score)| (*score - max_score).exp())
                        .sum::<f32>();
                    for (record, score) in scores {
                        let weight = (score - max_score).exp() / denom;
                        for d in 0..value_dim {
                            output[output_offset + d] +=
                                weight * record.value[kv_head * value_dim + d];
                        }
                    }
                }
            }
        }
        self.write_f32(
            self.required_output(node, 0)?,
            output_shape,
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
