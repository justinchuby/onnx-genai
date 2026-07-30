//! Native KV-backed static-cache continuous-batch decode session (design §J.10).
//!
//! [`NativeStaticCacheBatchedDecodeSession`] is the native, production
//! `BatchedDecodeSession` that drives a **static-cache `TensorScatter` export**
//! (e.g. the `tiny-llm-scatter` fixture) under the `ContinuousBatchManager` with
//! **independent per-row KV cursors**. Unlike the KV-free
//! [`super::NativeBatchedDecodeSession`], this session owns persistent
//! `[batch, MAX_LEN, KV_DIM]` key/value buffers and lets the graph's
//! `TensorScatter` node write each row's K/V at that row's own `write_indices`
//! cursor, so ragged continuous batching (rows at differing lengths in one batch)
//! runs natively.
//!
//! # KV binding — carry-forward (not IO-binding aliasing)
//!
//! The runtime owns one persistent full-capacity buffer per `key_cache.{i}` /
//! `value_cache.{i}`. Each step binds those buffers as the cache inputs; the
//! graph's `TensorScatter` emits the **full** updated buffer as
//! `updated_key_cache.{i}` / `updated_value_cache.{i}`, which the session moves
//! back into its buffer slot (no copy) to seed the next step. This is the ORT
//! `HandleSwap` semantics: the scatter output is the source of truth for the next
//! step, so no in-place input/output aliasing is required for correctness.
//! In-place aliasing (to elide the runtime's per-step full-buffer KV output
//! allocation) and active-row compaction (to skip idle-slot compute) are honest
//! efficiency follow-ups, not correctness gaps (design §J.10).
//!
//! # Grouped LoRA
//!
//! Identical to the KV-free session: when the graph declares a `lora.segments`
//! input, each step binds an `Int32 [rows]` route tensor from
//! [`Self::set_lora_routes`], reusing scratch buffers with no per-step heap
//! allocation. When there is no `lora.segments` input, routing is dormant and the
//! base fast path is preserved.

use super::*;
use onnx_genai_ort::decode::BatchedDecodeSession;
use onnx_genai_ort::{OrtError, Result as OrtResult, Value};

/// The `lora.segments` route id for a base-only row — the kernel's null route.
const BASE_LORA_ROUTE: i32 = -1;

/// One static-cache layer: the key/value cache inputs paired to their
/// `updated_*` outputs. All caches share axis 1 as the sequence axis.
struct StaticCacheLayer {
    key_input: String,
    value_input: String,
    key_output: String,
    value_output: String,
}

/// A native, production `BatchedDecodeSession` backed by a static-cache
/// `TensorScatter` graph. Borrows the session and owns per-row KV cursors plus
/// the persistent `[batch, MAX_LEN, KV_DIM]` cache buffers.
pub struct NativeStaticCacheBatchedDecodeSession<'session> {
    session: &'session mut InferenceSession,
    /// Declared token-ids graph input (`input_ids` / `decoder_input_ids`).
    token_input: String,
    /// Optional declared `position_ids` graph input.
    position_ids_input: Option<String>,
    /// The `write_indices` graph input (per-row start cursor).
    write_indices_input: String,
    /// The `nonpad_kv_seqlen` graph input (per-row valid length).
    nonpad_input: String,
    /// Declared logits graph output.
    logits_output: String,
    /// Static-cache layers (key/value cache pairs), sorted by layer index.
    layers: Vec<StaticCacheLayer>,
    /// Persistent full-capacity KV buffers keyed by cache input name, moved
    /// forward each step from the graph's `updated_*` outputs.
    kv_buffers: HashMap<String, Tensor>,
    /// The `lora.segments` graph input name, or `None` (routing dormant).
    lora_segments_input: Option<String>,
    /// Vocabulary width (last logits dimension), resolved on the first run.
    vocab: Option<usize>,
    batch_size: usize,
    max_len: usize,
    kv_dim: usize,
    dtype: DataType,
    /// Logical token length per physical row (the row's write cursor).
    row_lens: Vec<usize>,
    /// Whether each physical row is active.
    active: Vec<bool>,
    /// Per-row grouped-LoRA route for the NEXT step, in the ordering the upcoming
    /// step consumes. `None` when no routes were fed since the last step.
    pending_routes: Option<Vec<i32>>,
    /// Reused scratch buffers so the per-step index/route payloads are not
    /// reallocated on the hot path.
    token_scratch: Vec<i64>,
    position_scratch: Vec<i64>,
    write_indices_scratch: Vec<i64>,
    nonpad_scratch: Vec<i64>,
    segments_scratch: Vec<u8>,
    /// Reused physical-row-order route buffer (length `batch_size`) used to
    /// translate `step_active`'s active-row-ordered routes into the physical
    /// ordering `run_full_batch` actually runs, without a per-step allocation.
    physical_route_scratch: Vec<i32>,
}

impl<'session> NativeStaticCacheBatchedDecodeSession<'session> {
    /// Detect whether a native `InferenceSession` exposes the static-cache
    /// `TensorScatter` signature (a `write_indices` and `nonpad_kv_seqlen` input
    /// plus `key_cache.{i}` / `value_cache.{i}` inputs paired to
    /// `updated_key_cache.{i}` / `updated_value_cache.{i}` outputs). Returns
    /// `false` for a KV-free or decode-with-past graph, which routes to the
    /// KV-free [`super::NativeBatchedDecodeSession`] instead.
    pub fn is_static_cache(session: &InferenceSession) -> bool {
        let input_names = session
            .inputs()
            .iter()
            .map(|meta| meta.name.as_str())
            .collect::<Vec<_>>();
        input_names.contains(&"write_indices")
            && input_names.contains(&"nonpad_kv_seqlen")
            && input_names.iter().any(|name| static_cache_index(name, "key_cache.").is_some())
    }

    /// Build a native static-cache batched decode session over a grouped-capable
    /// `InferenceSession`. `batch_size` is the fixed number of physical rows; the
    /// KV capacity `MAX_LEN` is taken from the graph's `key_cache` signature (the
    /// requested `max_len` hint is ignored in favour of the graph's fixed
    /// buffer). Fails loud when the graph does not expose the static-cache
    /// signature or its layers are inconsistent.
    pub fn new(session: &'session mut InferenceSession, batch_size: usize) -> anyhow::Result<Self> {
        if batch_size == 0 {
            bail!("native static-cache batched decode requires batch_size > 0");
        }
        let input_names = session
            .inputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();
        let output_names = session
            .outputs()
            .iter()
            .map(|meta| meta.name.clone())
            .collect::<Vec<_>>();

        let token_input = input_names
            .iter()
            .find(|name| matches!(name.as_str(), "input_ids" | "decoder_input_ids"))
            .cloned()
            .context(
                "native static-cache batched decode requires an `input_ids` / `decoder_input_ids` graph input",
            )?;
        let position_ids_input = input_names
            .iter()
            .find(|name| name.as_str() == "position_ids")
            .cloned();
        let write_indices_input = input_names
            .iter()
            .find(|name| name.as_str() == "write_indices")
            .cloned()
            .context("native static-cache batched decode requires a `write_indices` graph input")?;
        let nonpad_input = input_names
            .iter()
            .find(|name| name.as_str() == "nonpad_kv_seqlen")
            .cloned()
            .context(
                "native static-cache batched decode requires a `nonpad_kv_seqlen` graph input",
            )?;
        let logits_output = output_names
            .iter()
            .find(|name| name.as_str() == "logits")
            .or_else(|| {
                // LoRA injection renames a graph output produced directly by the
                // shadowed base node (e.g. `lora.grouped.layers.0.q_proj.Y`), so
                // fall back to the single non-KV output when the canonical
                // `logits` name is absent.
                let mut non_kv = output_names
                    .iter()
                    .filter(|name| static_cache_index(name, "updated_key_cache.").is_none())
                    .filter(|name| static_cache_index(name, "updated_value_cache.").is_none());
                let first = non_kv.next();
                if non_kv.next().is_none() { first } else { None }
            })
            .cloned()
            .with_context(|| {
                format!(
                    "native static-cache batched decode could not resolve a `logits` output; outputs={output_names:?} inputs={input_names:?}"
                )
            })?;

        // Collect layer indices from `key_cache.{i}` inputs.
        let mut indices = input_names
            .iter()
            .filter_map(|name| static_cache_index(name, "key_cache."))
            .collect::<Vec<_>>();
        indices.sort_unstable();
        indices.dedup();
        if indices.is_empty() {
            bail!("native static-cache batched decode found no `key_cache.{{i}}` inputs");
        }

        let mut layers = Vec::with_capacity(indices.len());
        let mut max_len: Option<usize> = None;
        let mut kv_dim: Option<usize> = None;
        let mut dtype: Option<DataType> = None;
        for index in indices {
            let key_input = format!("key_cache.{index}");
            let value_input = format!("value_cache.{index}");
            let key_output = format!("updated_key_cache.{index}");
            let value_output = format!("updated_value_cache.{index}");
            let key_meta = session
                .inputs()
                .iter()
                .find(|meta| meta.name == key_input)
                .with_context(|| format!("missing static-cache input '{key_input}'"))?;
            let value_meta = session
                .inputs()
                .iter()
                .find(|meta| meta.name == value_input)
                .with_context(|| format!("missing static-cache input '{value_input}'"))?;
            if !output_names.contains(&key_output) {
                bail!("missing static-cache output '{key_output}'");
            }
            if !output_names.contains(&value_output) {
                bail!("missing static-cache output '{value_output}'");
            }
            let (layer_max_len, layer_kv_dim, layer_dtype) =
                static_cache_geometry(&key_input, key_meta)?;
            let (value_max_len, value_kv_dim, value_dtype) =
                static_cache_geometry(&value_input, value_meta)?;
            if (layer_max_len, layer_kv_dim, layer_dtype)
                != (value_max_len, value_kv_dim, value_dtype)
            {
                bail!(
                    "static-cache key/value geometry mismatch for layer {index}: key ({layer_max_len},{layer_kv_dim},{layer_dtype:?}) vs value ({value_max_len},{value_kv_dim},{value_dtype:?})"
                );
            }
            if *max_len.get_or_insert(layer_max_len) != layer_max_len {
                bail!("static-cache layers have inconsistent MAX_LEN");
            }
            if *kv_dim.get_or_insert(layer_kv_dim) != layer_kv_dim {
                bail!("static-cache layers have inconsistent KV_DIM");
            }
            if *dtype.get_or_insert(layer_dtype) != layer_dtype {
                bail!("static-cache layers have inconsistent dtypes");
            }
            layers.push(StaticCacheLayer {
                key_input,
                value_input,
                key_output,
                value_output,
            });
        }
        let max_len = max_len.context("static-cache signature has no layers")?;
        let kv_dim = kv_dim.context("static-cache signature has no layers")?;
        let dtype = dtype.context("static-cache signature has no layers")?;
        if !matches!(dtype, DataType::Float32 | DataType::Float16) {
            bail!(
                "native static-cache batched decode supports Float32/Float16 caches; got {dtype:?}"
            );
        }

        let lora_segments_input = session.lora_segments_input().map(str::to_string);

        // Allocate persistent zeroed KV buffers once.
        let mut kv_buffers = HashMap::with_capacity(layers.len() * 2);
        for layer in &layers {
            kv_buffers.insert(
                layer.key_input.clone(),
                zeroed_kv_buffer(dtype, batch_size, max_len, kv_dim)?,
            );
            kv_buffers.insert(
                layer.value_input.clone(),
                zeroed_kv_buffer(dtype, batch_size, max_len, kv_dim)?,
            );
        }

        Ok(Self {
            session,
            token_input,
            position_ids_input,
            write_indices_input,
            nonpad_input,
            logits_output,
            layers,
            kv_buffers,
            lora_segments_input,
            vocab: None,
            batch_size,
            max_len,
            kv_dim,
            dtype,
            row_lens: vec![0; batch_size],
            active: vec![true; batch_size],
            pending_routes: None,
            token_scratch: Vec::new(),
            position_scratch: Vec::new(),
            write_indices_scratch: Vec::new(),
            nonpad_scratch: Vec::new(),
            segments_scratch: Vec::new(),
            physical_route_scratch: Vec::new(),
        })
    }

    fn check_row(&self, row: usize) -> OrtResult<()> {
        if row >= self.batch_size {
            return Err(OrtError::InvalidArgument(format!(
                "row {row} out of range for native static-cache batch {}",
                self.batch_size
            )));
        }
        Ok(())
    }

    /// Zero one physical row's slice in every KV buffer, so a recycled slot does
    /// not leak the previous sequence's cache (mirrors ORT `zero_rank3_row`).
    fn zero_row(&mut self, row: usize) -> anyhow::Result<()> {
        let row_elements = self.max_len * self.kv_dim;
        let elem_size = self.dtype.storage_bytes(1);
        let start = row * row_elements * elem_size;
        let len = row_elements * elem_size;
        for buffer in self.kv_buffers.values_mut() {
            let mut bytes = buffer.as_bytes().to_vec();
            bytes[start..start + len].fill(0);
            *buffer = Tensor::from_raw(self.dtype, buffer.shape.clone(), &bytes)
                .context("rebuild zeroed native static-cache KV row")?;
        }
        Ok(())
    }

    /// Build the `lora.segments` tensor for `rows` activation rows, reusing the
    /// little-endian scratch buffer and the `pending_routes` Vec (swapped back so
    /// it is not reallocated per step). `Ok(None)` when routing is dormant.
    fn build_segments_tensor(&mut self, rows: usize) -> OrtResult<Option<Tensor>> {
        if self.lora_segments_input.is_none() {
            return Ok(None);
        }
        let mut routes = self.pending_routes.take().unwrap_or_default();
        if routes.is_empty() {
            // No routes fed since the last step: every row is base-only.
            routes.resize(rows, BASE_LORA_ROUTE);
        }
        if routes.len() != rows {
            let observed = routes.len();
            routes.clear();
            self.pending_routes = Some(routes);
            return Err(OrtError::InvalidArgument(format!(
                "native static-cache grouped-LoRA routes ({observed}) do not match the {rows} activation rows this step runs"
            )));
        }
        self.segments_scratch.clear();
        self.segments_scratch.reserve(rows * 4);
        for route in &routes {
            self.segments_scratch.extend_from_slice(&route.to_le_bytes());
        }
        let tensor = Tensor::from_raw(DataType::Int32, vec![rows], &self.segments_scratch)
            .map_err(|e| {
                OrtError::InvalidArgument(format!("build native lora.segments tensor: {e}"))
            });
        // Reuse the emptied Vec on the next step rather than dropping it.
        routes.clear();
        self.pending_routes = Some(routes);
        tensor.map(Some)
    }

    /// Translate `step_active`'s **active-row-ordered** pending routes (length
    /// `active_rows.len()`, fed by `ContinuousBatchManager::feed_active_lora_routes`)
    /// into a **physical-row-order** buffer of length `batch_size`, base-filling
    /// [`BASE_LORA_ROUTE`] for every physical row that is not active this step and
    /// placing each active row's route at its physical row index. The physical
    /// buffer is swapped into `pending_routes` so the subsequent
    /// `run_full_batch` -> `build_segments_tensor(batch_size)` consumes it in the
    /// ordering the full physical batch actually runs. Buffers are reused (the
    /// spare active buffer becomes the next `physical_route_scratch`), so there is
    /// no per-step heap allocation. A no-op when routing is dormant.
    fn expand_active_routes_to_physical(&mut self, active_rows: &[usize]) -> OrtResult<()> {
        if self.lora_segments_input.is_none() {
            return Ok(());
        }
        let mut active_routes = self.pending_routes.take().unwrap_or_default();
        let mut physical = std::mem::take(&mut self.physical_route_scratch);
        physical.clear();
        physical.resize(self.batch_size, BASE_LORA_ROUTE);
        if !active_routes.is_empty() {
            if active_routes.len() != active_rows.len() {
                let observed = active_routes.len();
                let expected = active_rows.len();
                active_routes.clear();
                self.pending_routes = Some(active_routes);
                self.physical_route_scratch = physical;
                return Err(OrtError::InvalidArgument(format!(
                    "native static-cache step_active grouped-LoRA routes ({observed}) do not match the {expected} active rows this step runs"
                )));
            }
            for (active_index, &row) in active_rows.iter().enumerate() {
                physical[row] = active_routes[active_index];
            }
        }
        // Hand the physical-order buffer to `build_segments_tensor` and keep the
        // now-spare active buffer around as reusable scratch (no per-step alloc).
        active_routes.clear();
        self.physical_route_scratch = active_routes;
        self.pending_routes = Some(physical);
        Ok(())
    }

    /// Run one decode step over the full physical batch. `tokens`/`positions` are
    /// physical-row indexed (length `batch_size`); `advances[row]` selects the
    /// rows whose cursor advances this step. Returns physical-row-indexed
    /// `[batch, 1, vocab]` logits and moves the updated KV buffers forward.
    fn run_full_batch(
        &mut self,
        tokens: &[i64],
        positions: &[i64],
        advances: &[bool],
    ) -> OrtResult<Value> {
        let batch = self.batch_size;
        if tokens.len() != batch || positions.len() != batch || advances.len() != batch {
            return Err(OrtError::InvalidArgument(format!(
                "native static-cache step expects batch_size {batch} entries (tokens {}, positions {}, advances {})",
                tokens.len(),
                positions.len(),
                advances.len()
            )));
        }
        for row in 0..batch {
            if advances[row] && self.row_lens[row] + 1 > self.max_len {
                return Err(OrtError::InvalidArgument(format!(
                    "row {row} static-cache write at {} exceeds capacity {}",
                    self.row_lens[row], self.max_len
                )));
            }
        }

        // Per-row cursors mirror ORT `try_run_batched_static_chunk`: write at the
        // row's current length; advertise +1 valid length only for advancing rows.
        self.write_indices_scratch.clear();
        self.nonpad_scratch.clear();
        for row in 0..batch {
            // Full-capacity edge (design §J.10): the always-full-batch run also
            // scatters INACTIVE rows, whose stale cursor can equal `max_len` (a
            // recycled slot left at capacity before the manager reassigns it).
            // `TensorScatter` bounds-checks `write_indices`, so `max_len` would
            // poison the whole step. An inactive row's KV is unused and re-zeroed
            // on `assign_row`, so clamp its write to the last in-range slot: a
            // harmless no-op into a discarded slice. Active rows are never clamped
            // (an advancing row at capacity is caught above; an active
            // non-advancing row at capacity is the deferred active-row-compaction
            // item, still §J.10).
            let write_index = if !self.active[row] && self.row_lens[row] >= self.max_len {
                self.max_len - 1
            } else {
                self.row_lens[row]
            };
            self.write_indices_scratch.push(write_index as i64);
            let nonpad = if advances[row] {
                self.row_lens[row] + 1
            } else {
                self.row_lens[row]
            };
            self.nonpad_scratch.push(nonpad as i64);
        }

        let segments = self.build_segments_tensor(batch)?;

        let token_tensor = Tensor::from_i64(&[batch, 1], tokens)
            .map_err(|e| OrtError::InvalidArgument(format!("native static-cache input_ids: {e}")))?;
        let write_indices_tensor = Tensor::from_i64(&[batch], &self.write_indices_scratch)
            .map_err(|e| {
                OrtError::InvalidArgument(format!("native static-cache write_indices: {e}"))
            })?;
        let nonpad_tensor = Tensor::from_i64(&[batch], &self.nonpad_scratch).map_err(|e| {
            OrtError::InvalidArgument(format!("native static-cache nonpad_kv_seqlen: {e}"))
        })?;
        let position_tensor = if self.position_ids_input.is_some() {
            Some(Tensor::from_i64(&[batch, 1], positions).map_err(|e| {
                OrtError::InvalidArgument(format!("native static-cache position_ids: {e}"))
            })?)
        } else {
            None
        };

        // Take the KV buffers out so they can be bound alongside the mutable run;
        // they are moved back from the `updated_*` outputs afterwards.
        let mut cache_tensors: Vec<(String, Tensor)> = Vec::with_capacity(self.layers.len() * 2);
        for layer in &self.layers {
            let key = self.kv_buffers.remove(&layer.key_input).ok_or_else(|| {
                OrtError::InvalidArgument(format!("missing KV buffer '{}'", layer.key_input))
            })?;
            let value = self.kv_buffers.remove(&layer.value_input).ok_or_else(|| {
                OrtError::InvalidArgument(format!("missing KV buffer '{}'", layer.value_input))
            })?;
            cache_tensors.push((layer.key_input.clone(), key));
            cache_tensors.push((layer.value_input.clone(), value));
        }

        let mut bindings: Vec<(&str, &Tensor)> = Vec::with_capacity(cache_tensors.len() + 5);
        bindings.push((self.token_input.as_str(), &token_tensor));
        bindings.push((self.write_indices_input.as_str(), &write_indices_tensor));
        bindings.push((self.nonpad_input.as_str(), &nonpad_tensor));
        if let (Some(name), Some(tensor)) =
            (self.position_ids_input.as_ref(), position_tensor.as_ref())
        {
            bindings.push((name.as_str(), tensor));
        }
        if let (Some(name), Some(tensor)) = (self.lora_segments_input.as_ref(), segments.as_ref()) {
            bindings.push((name.as_str(), tensor));
        }
        for (name, tensor) in &cache_tensors {
            bindings.push((name.as_str(), tensor));
        }

        let outputs = self.session.run(&bindings).map_err(|e| {
            OrtError::InvalidArgument(format!("native static-cache forward pass: {e}"))
        });
        // Whether the run succeeded or not, restore the cache buffers so the
        // session stays usable; on success they are replaced by `updated_*`.
        drop(bindings);
        for (name, tensor) in cache_tensors {
            self.kv_buffers.insert(name, tensor);
        }
        let outputs = outputs?;
        if outputs.len() != self.session.outputs().len() {
            return Err(OrtError::InvalidArgument(format!(
                "native static-cache decoder returned {} outputs, graph declares {}",
                outputs.len(),
                self.session.outputs().len()
            )));
        }

        let mut logits_tensor = None;
        let mut updated: HashMap<String, Tensor> = HashMap::with_capacity(self.layers.len() * 2);
        for (metadata, tensor) in self.session.outputs().iter().zip(outputs) {
            if metadata.name == self.logits_output {
                logits_tensor = Some(tensor);
            } else if let Some(input) = self.cache_input_for_output(&metadata.name) {
                updated.insert(input, tensor);
            }
        }
        // Carry the updated caches forward (ownership move, no copy).
        for layer in &self.layers {
            if let Some(tensor) = updated.remove(&layer.key_input) {
                self.kv_buffers.insert(layer.key_input.clone(), tensor);
            } else {
                return Err(OrtError::InvalidArgument(format!(
                    "native static-cache decoder omitted '{}'",
                    layer.key_output
                )));
            }
            if let Some(tensor) = updated.remove(&layer.value_input) {
                self.kv_buffers.insert(layer.value_input.clone(), tensor);
            } else {
                return Err(OrtError::InvalidArgument(format!(
                    "native static-cache decoder omitted '{}'",
                    layer.value_output
                )));
            }
        }

        let logits_tensor = logits_tensor.ok_or_else(|| {
            OrtError::InvalidArgument(format!(
                "native static-cache decoder omitted logits output '{}'",
                self.logits_output
            ))
        })?;

        // Advance the cursors only for advancing rows.
        for row in 0..batch {
            if advances[row] {
                self.row_lens[row] += 1;
            }
        }

        self.reshape_logits(logits_tensor, batch)
    }

    /// Map an `updated_key_cache.{i}` / `updated_value_cache.{i}` output name back
    /// to its `key_cache.{i}` / `value_cache.{i}` cache input name.
    fn cache_input_for_output(&self, output: &str) -> Option<String> {
        self.layers.iter().find_map(|layer| {
            if layer.key_output == output {
                Some(layer.key_input.clone())
            } else if layer.value_output == output {
                Some(layer.value_input.clone())
            } else {
                None
            }
        })
    }

    /// Reshape a raw logits tensor into a `[rows, 1, vocab]` `Value`.
    fn reshape_logits(&mut self, tensor: Tensor, rows: usize) -> OrtResult<Value> {
        let data = tensor.to_vec_f32();
        if data.is_empty() || !data.len().is_multiple_of(rows) {
            return Err(OrtError::InvalidArgument(format!(
                "native static-cache logits length {} is not a positive multiple of rows {rows}",
                data.len()
            )));
        }
        let vocab = data.len() / rows;
        match self.vocab {
            Some(previous) if previous != vocab => {
                return Err(OrtError::InvalidArgument(format!(
                    "native static-cache logits vocab changed from {previous} to {vocab}"
                )));
            }
            _ => self.vocab = Some(vocab),
        }
        Value::from_slice_f32(&data, &[rows as i64, 1, vocab as i64])
    }

    /// Gather the active rows' logits from physical-row-indexed `[batch,1,vocab]`
    /// into `[active,1,vocab]` in `active_rows` order.
    fn gather_active_logits(&self, physical: &Value, active_rows: &[usize]) -> OrtResult<Value> {
        let data = physical.to_vec_f32()?;
        let vocab = self.vocab.ok_or_else(|| {
            OrtError::InvalidArgument("native static-cache vocab unresolved".into())
        })?;
        let mut gathered = Vec::with_capacity(active_rows.len() * vocab);
        for &row in active_rows {
            let start = row * vocab;
            let end = start + vocab;
            if end > data.len() {
                return Err(OrtError::InvalidArgument(format!(
                    "native static-cache active row {row} out of logits range"
                )));
            }
            gathered.extend_from_slice(&data[start..end]);
        }
        Value::from_slice_f32(&gathered, &[active_rows.len() as i64, 1, vocab as i64])
    }
}

impl<'session> BatchedDecodeSession<'session>
    for NativeStaticCacheBatchedDecodeSession<'session>
{
    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn max_len(&self) -> usize {
        self.max_len
    }

    fn row_len(&self, row: usize) -> OrtResult<usize> {
        self.check_row(row)?;
        Ok(self.row_lens[row])
    }

    fn active_rows(&self) -> Vec<usize> {
        (0..self.batch_size)
            .filter(|&row| self.active[row])
            .collect()
    }

    fn deactivate_row(&mut self, row: usize) -> OrtResult<()> {
        self.check_row(row)?;
        self.active[row] = false;
        Ok(())
    }

    fn assign_row(&mut self, row: usize) -> OrtResult<()> {
        self.check_row(row)?;
        self.active[row] = true;
        self.row_lens[row] = 0;
        self.zero_row(row).map_err(|e| {
            OrtError::InvalidArgument(format!("native static-cache assign_row zero: {e}"))
        })?;
        Ok(())
    }

    fn step_select(
        &mut self,
        next_token_ids: &[i64],
        position_ids: &[i64],
        advance_rows: &[bool],
    ) -> OrtResult<Value> {
        if next_token_ids.len() != self.batch_size
            || position_ids.len() != self.batch_size
            || advance_rows.len() != self.batch_size
        {
            return Err(OrtError::InvalidArgument(format!(
                "native static-cache step_select expects batch_size {} entries (tokens {}, positions {}, advance {})",
                self.batch_size,
                next_token_ids.len(),
                position_ids.len(),
                advance_rows.len()
            )));
        }
        // Only advance active rows.
        let advances = (0..self.batch_size)
            .map(|row| self.active[row] && advance_rows[row])
            .collect::<Vec<_>>();
        self.run_full_batch(next_token_ids, position_ids, &advances)
    }

    fn step_active(&mut self, next_token_ids: &[i64], position_ids: &[i64]) -> OrtResult<Value> {
        let active_rows = self.active_rows();
        if active_rows.is_empty() {
            return Err(OrtError::InvalidArgument(
                "native static-cache step_active requires at least one active row".into(),
            ));
        }
        if next_token_ids.len() != active_rows.len() || position_ids.len() != active_rows.len() {
            return Err(OrtError::InvalidArgument(format!(
                "native static-cache step_active expects {} active-row entries (tokens {}, positions {})",
                active_rows.len(),
                next_token_ids.len(),
                position_ids.len()
            )));
        }
        // Expand active-row-ordered inputs to physical positions; run the full
        // batch, then gather the active rows' logits back in active order. The
        // grouped-LoRA routes fed for this step are active-row-ordered too, so
        // translate them into physical-row order (length batch_size, base-filled)
        // before run_full_batch's build_segments_tensor(batch_size) consumes them.
        self.expand_active_routes_to_physical(&active_rows)?;
        self.token_scratch.clear();
        self.token_scratch.resize(self.batch_size, 0);
        self.position_scratch.clear();
        self.position_scratch.resize(self.batch_size, 0);
        let mut advances = vec![false; self.batch_size];
        for (active_index, &row) in active_rows.iter().enumerate() {
            self.token_scratch[row] = next_token_ids[active_index];
            self.position_scratch[row] = position_ids[active_index];
            advances[row] = true;
        }
        let tokens = std::mem::take(&mut self.token_scratch);
        let positions = std::mem::take(&mut self.position_scratch);
        let physical = self.run_full_batch(&tokens, &positions, &advances);
        self.token_scratch = tokens;
        self.position_scratch = positions;
        let physical = physical?;
        self.gather_active_logits(&physical, &active_rows)
    }

    fn set_lora_routes(&mut self, routes: &[i32]) -> OrtResult<()> {
        if self.lora_segments_input.is_none() {
            return Ok(());
        }
        let mut buffer = self.pending_routes.take().unwrap_or_default();
        buffer.clear();
        buffer.extend_from_slice(routes);
        self.pending_routes = Some(buffer);
        Ok(())
    }
}

/// Parse `key_cache.{i}` / `value_cache.{i}` layer indices from an input name.
fn static_cache_index(name: &str, prefix: &str) -> Option<usize> {
    name.strip_prefix(prefix)?.parse().ok()
}

/// Extract `(MAX_LEN, KV_DIM, dtype)` from a rank-3 `[B, MAX_LEN, KV_DIM]` cache
/// input's static geometry, failing loud on a non-rank-3 or symbolic cache.
fn static_cache_geometry(
    name: &str,
    meta: &onnx_runtime_session::IoMeta,
) -> anyhow::Result<(usize, usize, DataType)> {
    if meta.shape.len() != 3 {
        bail!(
            "static-cache input '{name}' must be rank-3 [B, MAX_LEN, KV_DIM]; got rank {}",
            meta.shape.len()
        );
    }
    let max_len = meta.shape[1]
        .as_static()
        .with_context(|| format!("static-cache input '{name}' MAX_LEN dim must be static"))?;
    let kv_dim = meta.shape[2]
        .as_static()
        .with_context(|| format!("static-cache input '{name}' KV_DIM dim must be static"))?;
    if max_len == 0 || kv_dim == 0 {
        bail!("static-cache input '{name}' has a zero MAX_LEN or KV_DIM");
    }
    Ok((max_len, kv_dim, meta.dtype))
}

/// Allocate a zeroed `[batch, max_len, kv_dim]` KV buffer of `dtype`.
fn zeroed_kv_buffer(
    dtype: DataType,
    batch: usize,
    max_len: usize,
    kv_dim: usize,
) -> anyhow::Result<Tensor> {
    let numel = batch * max_len * kv_dim;
    let bytes = vec![0_u8; dtype.storage_bytes(numel).max(1)];
    Tensor::from_raw(dtype, vec![batch, max_len, kv_dim], &bytes)
        .context("allocate zeroed native static-cache KV buffer")
}
