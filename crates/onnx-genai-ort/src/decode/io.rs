use super::*;

#[derive(Debug, Clone)]
pub(super) struct KvPair {
    pub(super) past: String,
    pub(super) present: String,
    pub(super) input: TensorInfo,
    pub(super) seq_axis: usize,
}
pub(super) fn infer_kv_pairs(
    session: &Session,
    io: Option<&onnx_genai_metadata::ModelIoSpec>,
) -> Result<Vec<KvPair>> {
    if let Some(io) = io {
        return match (&io.kv_inputs, &io.kv_outputs) {
            (Some(inputs), Some(outputs)) if inputs.len() == outputs.len() => inputs
                .iter()
                .zip(outputs)
                .map(|(past, present)| kv_pair(session, past, present))
                .collect(),
            (Some(inputs), Some(outputs)) => Err(OrtError::InvalidArgument(format!(
                "io.kv_inputs ({}) and io.kv_outputs ({}) must have equal length",
                inputs.len(),
                outputs.len()
            ))),
            (None, None) => Ok(Vec::new()),
            _ => Err(OrtError::InvalidArgument(
                "io.kv_inputs and io.kv_outputs must be declared together".into(),
            )),
        };
    }
    Ok(Vec::new())
}

fn kv_pair(session: &Session, past_name: &str, present_name: &str) -> Result<KvPair> {
    let input = session
        .inputs()
        .iter()
        .find(|input| input.name == past_name)
        .cloned()
        .ok_or_else(|| {
            OrtError::InvalidArgument(format!("declared KV input '{past_name}' is not exposed"))
        })?;
    if !session
        .outputs()
        .iter()
        .any(|output| output.name == present_name)
    {
        return Err(OrtError::InvalidArgument(format!(
            "declared KV output '{present_name}' is not exposed"
        )));
    }
    if !matches!(
        input.dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        return Err(OrtError::InvalidArgument(format!(
            "KV input '{}' must be Float32, Float16, or BFloat16, got {:?}",
            input.name, input.dtype
        )));
    }
    if input.shape.len() < 3 {
        return Err(OrtError::InvalidArgument(format!(
            "KV input '{}' has unsupported shape {:?}",
            input.name, input.shape
        )));
    }
    let seq_axis = input.shape.len() - 2;
    Ok(KvPair {
        past: past_name.to_string(),
        present: present_name.to_string(),
        input,
        seq_axis,
    })
}

/// Resolved static-cache scatter-ABI control ports.
///
/// The `write_indices`/`kv_sequence_length` scatter control inputs are integer
/// vectors and therefore SHAPE-indistinguishable, so their names come only from
/// the package's declared scatter ABI. The token/position ports come from the
/// declared port roles. Both reach this driver through the resolved decode ABI
/// (`InferenceMetadata::decoder_io()`), which a workflow package derives from
/// its `state_service` group and its component port roles. Nothing is ever
/// interpreted from a graph port name: a static-cache graph whose package
/// declares no scatter ABI is rejected rather than name-guessed.
#[derive(Debug, Clone)]
pub(super) struct StaticCacheAbi {
    pub(super) token_input: String,
    pub(super) position_ids_input: Option<String>,
    pub(super) write_indices_input: String,
    pub(super) kv_sequence_length_input: String,
}

/// Role of a static-cache graph input, resolved name-agnostically from the ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StaticCacheInputRole {
    Token,
    Position,
    WriteIndices,
    KvSequenceLength,
}

impl StaticCacheAbi {
    /// Classify a control input by matching the resolved ABI port names. Returns
    /// `None` for a cache-buffer input, which the caller resolves against its
    /// per-layer buffers.
    pub(super) fn classify(&self, name: &str) -> Option<StaticCacheInputRole> {
        if name == self.token_input {
            Some(StaticCacheInputRole::Token)
        } else if Some(name) == self.position_ids_input.as_deref() {
            Some(StaticCacheInputRole::Position)
        } else if name == self.write_indices_input {
            Some(StaticCacheInputRole::WriteIndices)
        } else if name == self.kv_sequence_length_input {
            Some(StaticCacheInputRole::KvSequenceLength)
        } else {
            None
        }
    }
}

pub(super) fn detect_static_cache(
    session: &Session,
    io: Option<&onnx_genai_metadata::ModelIoSpec>,
) -> Result<Option<(StaticCacheSignature, Vec<StaticCachePair>, StaticCacheAbi)>> {
    // Explicit metadata is authoritative and fully name-agnostic: the graph's
    // scatter-ABI ports are exactly those declared, never inferred from names.
    if let Some(spec) = io.and_then(|io| io.static_cache.as_ref()) {
        return detect_static_cache_from_spec(session, io, spec).map(Some);
    }
    reject_undeclared_static_cache(session)
}

/// Resolve the static-cache ABI from the package's declared scatter discipline.
fn detect_static_cache_from_spec(
    session: &Session,
    io: Option<&onnx_genai_metadata::ModelIoSpec>,
    spec: &onnx_genai_metadata::StaticCacheIoSpec,
) -> Result<(StaticCacheSignature, Vec<StaticCachePair>, StaticCacheAbi)> {
    let layer_count = validate_static_cache_spec_lengths(spec)?;
    require_declared_input(
        session,
        &spec.write_indices_input,
        "static_cache.write_indices_input",
    )?;
    require_declared_input(
        session,
        &spec.kv_sequence_length_input,
        "static_cache.kv_sequence_length_input",
    )?;

    let mut pairs = Vec::with_capacity(layer_count);
    let mut geometry = None;
    for index in 0..layer_count {
        let key_input = declared_input(session, &spec.key_cache_inputs[index])?;
        let value_input = declared_input(session, &spec.value_cache_inputs[index])?;
        require_declared_output(session, &spec.key_cache_outputs[index])?;
        require_declared_output(session, &spec.value_cache_outputs[index])?;
        accumulate_static_cache_layer(&mut geometry, index, &key_input, &value_input)?;
        pairs.push(StaticCachePair {
            key_input,
            value_input,
            key_output: spec.key_cache_outputs[index].clone(),
            value_output: spec.value_cache_outputs[index].clone(),
        });
    }
    let (max_len, kv_dim, dtype) =
        geometry.ok_or_else(|| OrtError::InvalidArgument("non-empty static cache pairs".into()))?;
    let abi = StaticCacheAbi {
        token_input: token_input_name(session, io),
        position_ids_input: position_ids_input_name(session, io),
        write_indices_input: spec.write_indices_input.clone(),
        kv_sequence_length_input: spec.kv_sequence_length_input.clone(),
    };
    let signature = StaticCacheSignature {
        layers: pairs.len(),
        max_len,
        kv_dim,
        dtype,
        has_position_ids: abi.position_ids_input.is_some(),
    };
    Ok((signature, pairs, abi))
}

/// Fail-closed guard for the no-metadata path.
///
/// The TensorScatter static-cache scatter ABI is driven by SHAPE-indistinguish-
/// able integer control ports (`write_indices` / `kv_sequence_length`), so it can
/// only be bound from a declared scatter ABI — never guessed from graph port
/// names. When a graph exposes the historical scatter control ports but the
/// package declares no scatter discipline, we refuse to interpret those names and
/// instead return an actionable error naming the exact keys to declare. Graphs
/// that expose no static-cache scatter ABI return `Ok(None)` so the ordinary KV
/// path handles them.
fn reject_undeclared_static_cache(
    session: &Session,
) -> Result<Option<(StaticCacheSignature, Vec<StaticCachePair>, StaticCacheAbi)>> {
    let looks_like_static_cache = session
        .input_names()
        .iter()
        .any(|name| name == "write_indices" || name == "nonpad_kv_seqlen");
    if !looks_like_static_cache {
        return Ok(None);
    }
    Err(OrtError::InvalidArgument(
        "graph exposes a TensorScatter static-cache scatter ABI but the package \
         declares no fixed-capacity write discipline; its integer scatter control \
         ports (write_indices / kv_sequence_length) are shape-indistinguishable \
         and cannot be bound by port name. Declare the cache group at \
         pipeline.workflow.serving.state_service.groups.<group> with `update.kind: \
         indexed_scatter`, `update.write_indices_ports.<component>`, \
         `update.kv_length_ports.<component>`, and a `role: key` / `role: value` \
         (or `role: combined`) and `layer:` on each of the group's per-layer port \
         pairs."
            .into(),
    ))
}

/// Resolve the token-sequence input: the declared `token_input` role, else the
/// historical `input_ids` port.
fn token_input_name(_session: &Session, io: Option<&onnx_genai_metadata::ModelIoSpec>) -> String {
    io.and_then(|io| io.token_input.clone())
        .unwrap_or_else(|| "input_ids".to_string())
}

/// Resolve the position-ids input: the declared `position_ids_input` role, else
/// the historical `position_ids` port when the graph exposes one.
fn position_ids_input_name(
    session: &Session,
    io: Option<&onnx_genai_metadata::ModelIoSpec>,
) -> Option<String> {
    if let Some(declared) = io.and_then(|io| io.position_ids_input.clone()) {
        return Some(declared);
    }
    session
        .input_names()
        .iter()
        .any(|name| name == "position_ids")
        .then(|| "position_ids".to_string())
}

fn declared_input(session: &Session, name: &str) -> Result<TensorInfo> {
    session
        .inputs()
        .iter()
        .find(|input| input.name == name)
        .cloned()
        .ok_or_else(|| {
            OrtError::InvalidArgument(format!(
                "declared static-cache input '{name}' is not exposed"
            ))
        })
}

fn require_declared_input(session: &Session, name: &str, field: &str) -> Result<()> {
    if session.input_names().iter().any(|input| input == name) {
        Ok(())
    } else {
        Err(OrtError::InvalidArgument(format!(
            "the package's declared decode ABI names {field} '{name}', which the graph \
             does not expose"
        )))
    }
}

fn require_declared_output(session: &Session, name: &str) -> Result<()> {
    if session.output_names().iter().any(|output| output == name) {
        Ok(())
    } else {
        Err(OrtError::InvalidArgument(format!(
            "declared static-cache output '{name}' is not exposed"
        )))
    }
}

/// Validate one layer's key/value cache tensors and fold their geometry into the
/// running `(max_len, kv_dim, dtype)`, erroring on any cross-layer mismatch.
fn accumulate_static_cache_layer(
    geometry: &mut Option<(usize, usize, DataType)>,
    index: usize,
    key_input: &TensorInfo,
    value_input: &TensorInfo,
) -> Result<()> {
    validate_static_cache_tensor(key_input)?;
    validate_static_cache_tensor(value_input)?;
    if key_input.shape[1..] != value_input.shape[1..] {
        return Err(OrtError::InvalidArgument(format!(
            "key/value cache shape mismatch for layer {index}: {:?} vs {:?}",
            key_input.shape, value_input.shape
        )));
    }
    if key_input.dtype != value_input.dtype {
        return Err(OrtError::InvalidArgument(format!(
            "key/value cache dtype mismatch for layer {index}: {:?} vs {:?}",
            key_input.dtype, value_input.dtype
        )));
    }
    let layer_max_len = key_input.shape[1] as usize;
    let layer_kv_dim = key_input.shape[2] as usize;
    match geometry {
        None => *geometry = Some((layer_max_len, layer_kv_dim, key_input.dtype)),
        Some((max_len, kv_dim, dtype)) => {
            if *max_len != layer_max_len {
                return Err(OrtError::InvalidArgument(
                    "static-cache layers have inconsistent max lengths".into(),
                ));
            }
            if *kv_dim != layer_kv_dim {
                return Err(OrtError::InvalidArgument(
                    "static-cache layers have inconsistent KV dims".into(),
                ));
            }
            if *dtype != key_input.dtype {
                return Err(OrtError::InvalidArgument(
                    "static-cache layers have inconsistent dtypes".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Check that a port looks like a static-cache buffer.
///
/// The element type is deliberately not constrained. A static cache is a
/// fixed-capacity buffer the graph scatters into, and the runtime's job is to
/// allocate it, bind it, and hand back the handle — none of which depends on
/// what the elements mean. An FP8 cache is exactly as bindable as an fp16 one;
/// if the model's attention kernel has no FP8 implementation, the session fails
/// to load with the execution provider's own type error, which names the
/// operator. Rejecting the dtype here would replace that precise diagnosis with
/// a vaguer one from a layer that has no kernels to speak for.
fn validate_static_cache_tensor(info: &TensorInfo) -> Result<()> {
    if info.shape.len() != 3 || info.shape[1] <= 0 || info.shape[2] <= 0 {
        return Err(OrtError::InvalidArgument(format!(
            "static-cache tensor '{}' must have shape [B, MAX_LEN, KV_DIM], got {:?}",
            info.name, info.shape
        )));
    }
    Ok(())
}

/// Validate that an explicit `static_cache` spec declares a consistent,
/// positionally-paired set of per-layer cache ports, returning the layer count.
///
/// Pure (no [`Session`]): the four per-layer lists must be non-empty and of
/// equal length. Errors name the exact offending metadata key so a
/// misconfigured contract is actionable rather than a silent guess.
fn validate_static_cache_spec_lengths(
    spec: &onnx_genai_metadata::StaticCacheIoSpec,
) -> Result<usize> {
    let layer_count = spec.key_cache_inputs.len();
    if layer_count == 0 {
        return Err(OrtError::InvalidArgument(
            "the declared static-cache ABI has no key_cache_inputs; the cache group must bind \
             at least one per-layer port pair"
                .into(),
        ));
    }
    for (field, len) in [
        ("value_cache_inputs", spec.value_cache_inputs.len()),
        ("key_cache_outputs", spec.key_cache_outputs.len()),
        ("value_cache_outputs", spec.value_cache_outputs.len()),
    ] {
        if len != layer_count {
            return Err(OrtError::InvalidArgument(format!(
                "the declared static-cache ABI has {len} {field} but {layer_count} key cache \
                 inputs; every layer needs one port of each kind"
            )));
        }
    }
    Ok(layer_count)
}

#[cfg(test)]
mod static_cache_abi_tests {
    use super::*;
    use onnx_genai_metadata::StaticCacheIoSpec;

    fn spec(layers: usize) -> StaticCacheIoSpec {
        let name = |prefix: &str| (0..layers).map(|i| format!("{prefix}.{i}")).collect();
        StaticCacheIoSpec {
            write_indices_input: "scatter_positions".into(),
            kv_sequence_length_input: "valid_len".into(),
            key_cache_inputs: name("kc"),
            value_cache_inputs: name("vc"),
            key_cache_outputs: name("ukc"),
            value_cache_outputs: name("uvc"),
        }
    }

    #[test]
    fn abi_classify_is_name_agnostic() {
        // Non-standard control-port names resolve purely from the declared ABI;
        // graph port names are never interpreted.
        let abi = StaticCacheAbi {
            token_input: "opaque_tokens".into(),
            position_ids_input: Some("opaque_positions".into()),
            write_indices_input: "scatter_positions".into(),
            kv_sequence_length_input: "valid_len".into(),
        };
        assert_eq!(
            abi.classify("opaque_tokens"),
            Some(StaticCacheInputRole::Token)
        );
        assert_eq!(
            abi.classify("opaque_positions"),
            Some(StaticCacheInputRole::Position)
        );
        assert_eq!(
            abi.classify("scatter_positions"),
            Some(StaticCacheInputRole::WriteIndices)
        );
        assert_eq!(
            abi.classify("valid_len"),
            Some(StaticCacheInputRole::KvSequenceLength)
        );
        // A cache-buffer input is not a control port.
        assert_eq!(abi.classify("kc.0"), None);
        // The historical hardcoded names are NOT special once metadata renames
        // the ports: they classify as ordinary cache/other inputs.
        assert_eq!(abi.classify("write_indices"), None);
        assert_eq!(abi.classify("nonpad_kv_seqlen"), None);
    }

    #[test]
    fn abi_position_absent_never_classifies_position() {
        let abi = StaticCacheAbi {
            token_input: "input_ids".into(),
            position_ids_input: None,
            write_indices_input: "write_indices".into(),
            kv_sequence_length_input: "nonpad_kv_seqlen".into(),
        };
        assert_eq!(abi.classify("position_ids"), None);
    }

    #[test]
    fn spec_lengths_valid() {
        assert_eq!(validate_static_cache_spec_lengths(&spec(3)).unwrap(), 3);
        assert_eq!(validate_static_cache_spec_lengths(&spec(1)).unwrap(), 1);
    }

    #[test]
    fn spec_empty_errors_with_key() {
        let err = validate_static_cache_spec_lengths(&spec(0)).unwrap_err();
        assert!(
            err.to_string().contains("key_cache_inputs"),
            "error must name the missing key: {err}"
        );
    }

    #[test]
    fn spec_mismatched_lengths_error_with_offending_key() {
        for (mutate, field) in [
            (
                (|s: &mut StaticCacheIoSpec| s.value_cache_inputs.pop())
                    as fn(&mut StaticCacheIoSpec) -> Option<String>,
                "value_cache_inputs",
            ),
            (
                |s: &mut StaticCacheIoSpec| s.key_cache_outputs.pop(),
                "key_cache_outputs",
            ),
            (
                |s: &mut StaticCacheIoSpec| s.value_cache_outputs.pop(),
                "value_cache_outputs",
            ),
        ] {
            let mut broken = spec(2);
            mutate(&mut broken);
            let err = validate_static_cache_spec_lengths(&broken).unwrap_err();
            assert!(
                err.to_string().contains(field),
                "error must name the offending key '{field}': {err}"
            );
        }
    }
}
