//! CompressedSparseAttention (CSA/HCA) state-group threading for the native
//! decode runner.
//!
//! DeepSeek-V4 threads compressed KV attention state as role-typed `present_* ->
//! past_*` port pairs. This module validates a declared [`CsaStateGroupAbi`]
//! against the graph's real typed IO and lowers its edges into
//! `(past_input, present_output)` pairs. The runner folds those pairs into the
//! same recurrent-state machinery it already uses for fixed loop-carried state,
//! so present->past rebinding, stable-address scratch, and speculative
//! snapshot/rollback/fork all come from the existing authority rather than a new
//! coordinator.
//!
//! Every check runs *before* any device buffer is reserved: an unhonorable group
//! is a typed refusal, never a partial allocation.

use super::*;
use onnx_genai_metadata::{CsaCacheFormat, CsaStateGroupAbi, CsaStateRole};
use onnx_runtime_session::IoMeta;

/// The frozen v1 element dtype each CSA state role carries, given the group's
/// declared cache format.
///
/// The compression and index carries are always float32 accumulators, and the
/// learned index keys are always packed uint8 records. The compressed KV
/// records take the group's declared `cache_format`: an `f32` cache keeps them
/// float32, while a block-quantized cache (`fp8_e4m3_block64`) packs them into
/// uint8 byte records. This mirrors the CSA op kernel exactly, whose
/// `CacheFormat::dtype()` is `Float32` for `f32` and `Uint8` for the packed
/// formats; `fp4_e2m1_block32` is the learned-index format and is refused as a
/// KV cache format before this is ever reached.
fn expected_dtype(role: CsaStateRole, cache_format: CsaCacheFormat) -> DataType {
    match role {
        CsaStateRole::CompressedKv => match cache_format {
            CsaCacheFormat::F32 => DataType::Float32,
            CsaCacheFormat::Fp8E4m3Block64 | CsaCacheFormat::Fp4E2m1Block32 => DataType::Uint8,
        },
        CsaStateRole::IndexKey => DataType::Uint8,
        CsaStateRole::CompressionCarry | CsaStateRole::IndexCarry => DataType::Float32,
    }
}

/// Resolve a declared CSA state group against the graph's typed IO into
/// `(past_input, present_output)` edges, refusing before any allocation.
///
/// `occupied` names ports already claimed by KV or fixed-state pairs; a CSA edge
/// that reuses one is refused so the compressed state threads through its own
/// ports and never aliases a growing KV cache.
pub(super) fn resolve_csa_state_edges(
    inputs: &[IoMeta],
    outputs: &[IoMeta],
    group: &CsaStateGroupAbi,
    occupied: &HashSet<&str>,
) -> anyhow::Result<Vec<(String, String)>> {
    // Property-level validity first: ratio/format/recurrence/edge-set. This is
    // the typed refusal for an unknown MTP recurrence, an fp4 KV cache format,
    // a missing or duplicate edge, and an index edge on a ratio that carries no
    // index — all raised before a single graph port is touched.
    let edges = group.present_past_edges().map_err(|refusal| {
        anyhow::anyhow!("CSA state group refused before allocation: {refusal}")
    })?;

    let mut resolved = Vec::with_capacity(edges.len());
    for (role, past, present) in edges {
        if occupied.contains(past.as_str()) || occupied.contains(present.as_str()) {
            bail!(
                "CSA state edge '{role}' ('{past}'=>'{present}') overlaps a declared KV or fixed-state \
                 port; compressed state must thread through its own ports"
            );
        }
        let input = inputs.iter().find(|meta| meta.name == past).ok_or_else(|| {
            anyhow::anyhow!(
                "CSA state edge '{role}' declares past input '{past}' but the graph does not expose it"
            )
        })?;
        let output = outputs
            .iter()
            .find(|meta| meta.name == present)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "CSA state edge '{role}' declares present output '{present}' but the graph does not expose it"
                )
            })?;
        let expected = expected_dtype(role, group.cache_format);
        if input.dtype != expected {
            bail!(
                "CSA state edge '{role}' past input '{past}' has dtype {:?}, expected {expected:?} for this role",
                input.dtype
            );
        }
        if output.dtype != expected {
            bail!(
                "CSA state edge '{role}' present output '{present}' has dtype {:?}, expected {expected:?} for this role",
                output.dtype
            );
        }
        resolved.push((past, present));
    }
    Ok(resolved)
}

/// Resolve every declared CSA state group against the graph's typed IO,
/// accumulating claimed ports so two groups can never bind the same port.
///
/// Returns the flat `(past_input, present_output)` edge list in declaration
/// order — a schedule that alternates ratio-4 and ratio-128 layers yields the
/// ratio-4 layer's four edges and the ratio-128 layer's two edges, each threaded
/// through its own ports. `already_occupied` names ports already claimed by KV
/// or fixed-state pairs.
pub(super) fn resolve_csa_state_groups(
    inputs: &[IoMeta],
    outputs: &[IoMeta],
    groups: &[CsaStateGroupAbi],
    already_occupied: &HashSet<&str>,
) -> anyhow::Result<Vec<(String, String)>> {
    let mut occupied: HashSet<String> = already_occupied
        .iter()
        .map(|name| name.to_string())
        .collect();
    let mut all = Vec::new();
    for group in groups {
        let occupied_refs: HashSet<&str> = occupied.iter().map(String::as_str).collect();
        let edges = resolve_csa_state_edges(inputs, outputs, group, &occupied_refs)?;
        for (past, present) in edges {
            occupied.insert(past.clone());
            occupied.insert(present.clone());
            all.push((past, present));
        }
    }
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai_metadata::{CsaCacheFormat, CsaCompressionRatio, CsaRecurrence, CsaStateEdge};
    use onnx_runtime_ir::static_shape;

    fn meta(name: &str, dtype: DataType, shape: &[usize]) -> IoMeta {
        IoMeta {
            name: name.to_string(),
            dtype,
            shape: static_shape(shape.iter().copied()),
        }
    }

    fn edge(role: CsaStateRole, past: &str, present: &str) -> CsaStateEdge {
        CsaStateEdge {
            role,
            past_port: past.to_string(),
            present_port: present.to_string(),
        }
    }

    fn ratio128() -> CsaStateGroupAbi {
        CsaStateGroupAbi {
            ratio: CsaCompressionRatio::Ratio128,
            cache_format: CsaCacheFormat::Fp8E4m3Block64,
            recurrence: CsaRecurrence::Standard,
            edges: vec![
                edge(CsaStateRole::CompressedKv, "past_ckv", "present_ckv"),
                edge(CsaStateRole::CompressionCarry, "past_cc", "present_cc"),
            ],
        }
    }

    fn ratio4() -> CsaStateGroupAbi {
        CsaStateGroupAbi {
            ratio: CsaCompressionRatio::Ratio4,
            cache_format: CsaCacheFormat::Fp8E4m3Block64,
            recurrence: CsaRecurrence::Standard,
            edges: vec![
                edge(CsaStateRole::CompressedKv, "past_ckv", "present_ckv"),
                edge(CsaStateRole::CompressionCarry, "past_cc", "present_cc"),
                edge(CsaStateRole::IndexKey, "past_ik", "present_ik"),
                edge(CsaStateRole::IndexCarry, "past_ic", "present_ic"),
            ],
        }
    }

    fn ratio128_io() -> (Vec<IoMeta>, Vec<IoMeta>) {
        (
            vec![
                meta("past_ckv", DataType::Uint8, &[1, 0, 583]),
                meta("past_cc", DataType::Float32, &[1, 512]),
            ],
            vec![
                meta("present_ckv", DataType::Uint8, &[1, 1, 583]),
                meta("present_cc", DataType::Float32, &[1, 512]),
            ],
        )
    }

    fn ratio4_io() -> (Vec<IoMeta>, Vec<IoMeta>) {
        let (mut inp, mut out) = ratio128_io();
        inp.push(meta("past_ik", DataType::Uint8, &[1, 0, 68]));
        inp.push(meta("past_ic", DataType::Float32, &[1, 128]));
        out.push(meta("present_ik", DataType::Uint8, &[1, 1, 68]));
        out.push(meta("present_ic", DataType::Float32, &[1, 128]));
        (inp, out)
    }

    #[test]
    fn ratio128_resolves_two_edges() {
        let (inp, out) = ratio128_io();
        let edges = resolve_csa_state_edges(&inp, &out, &ratio128(), &HashSet::new()).unwrap();
        assert_eq!(
            edges,
            vec![
                ("past_ckv".to_string(), "present_ckv".to_string()),
                ("past_cc".to_string(), "present_cc".to_string()),
            ]
        );
    }

    #[test]
    fn ratio4_resolves_four_edges_in_role_order() {
        let (inp, out) = ratio4_io();
        let edges = resolve_csa_state_edges(&inp, &out, &ratio4(), &HashSet::new()).unwrap();
        assert_eq!(edges.len(), 4);
        assert_eq!(edges[2], ("past_ik".to_string(), "present_ik".to_string()));
        assert_eq!(edges[3], ("past_ic".to_string(), "present_ic".to_string()));
    }

    #[test]
    fn missing_past_input_is_refused() {
        let (mut inp, out) = ratio128_io();
        inp.retain(|m| m.name != "past_ckv");
        let err = resolve_csa_state_edges(&inp, &out, &ratio128(), &HashSet::new())
            .unwrap_err()
            .to_string();
        assert!(err.contains("past_ckv"), "{err}");
    }

    #[test]
    fn missing_present_output_is_refused() {
        let (inp, mut out) = ratio128_io();
        out.retain(|m| m.name != "present_cc");
        let err = resolve_csa_state_edges(&inp, &out, &ratio128(), &HashSet::new())
            .unwrap_err()
            .to_string();
        assert!(err.contains("present_cc"), "{err}");
    }

    #[test]
    fn wrong_role_dtype_is_refused() {
        let (mut inp, out) = ratio128_io();
        // Compressed KV must be a uint8 byte buffer, not float32.
        inp[0].dtype = DataType::Float32;
        let err = resolve_csa_state_edges(&inp, &out, &ratio128(), &HashSet::new())
            .unwrap_err()
            .to_string();
        assert!(err.contains("dtype"), "{err}");
    }

    #[test]
    fn overlap_with_kv_port_is_refused() {
        let (inp, out) = ratio128_io();
        let mut occupied = HashSet::new();
        occupied.insert("past_ckv");
        let err = resolve_csa_state_edges(&inp, &out, &ratio128(), &occupied)
            .unwrap_err()
            .to_string();
        assert!(err.contains("overlaps"), "{err}");
    }

    #[test]
    fn unknown_mtp_recurrence_is_refused_before_graph() {
        // No graph ports at all: the property-level refusal must fire first.
        let mut group = ratio128();
        group.recurrence = CsaRecurrence::MultiTokenPrediction;
        let err = resolve_csa_state_edges(&[], &[], &group, &HashSet::new())
            .unwrap_err()
            .to_string();
        assert!(err.contains("multi-token-prediction"), "{err}");
    }

    #[test]
    fn ratio128_with_index_edge_is_refused_before_graph() {
        let mut group = ratio128();
        group
            .edges
            .push(edge(CsaStateRole::IndexKey, "past_ik", "present_ik"));
        let err = resolve_csa_state_edges(&[], &[], &group, &HashSet::new())
            .unwrap_err()
            .to_string();
        assert!(err.contains("index"), "{err}");
    }

    /// A shape-faithful alternating ratio-4 / ratio-128 schedule: two layers with
    /// distinct ports thread all six edges (four for the ratio-4 layer, two for
    /// the ratio-128 layer) through the multi-group resolver.
    fn alternating_groups() -> (Vec<CsaStateGroupAbi>, Vec<IoMeta>, Vec<IoMeta>) {
        let layer0 = CsaStateGroupAbi {
            ratio: CsaCompressionRatio::Ratio4,
            cache_format: CsaCacheFormat::Fp8E4m3Block64,
            recurrence: CsaRecurrence::Standard,
            edges: vec![
                edge(CsaStateRole::CompressedKv, "l0_past_ckv", "l0_present_ckv"),
                edge(
                    CsaStateRole::CompressionCarry,
                    "l0_past_cc",
                    "l0_present_cc",
                ),
                edge(CsaStateRole::IndexKey, "l0_past_ik", "l0_present_ik"),
                edge(CsaStateRole::IndexCarry, "l0_past_ic", "l0_present_ic"),
            ],
        };
        let layer1 = CsaStateGroupAbi {
            ratio: CsaCompressionRatio::Ratio128,
            cache_format: CsaCacheFormat::F32,
            recurrence: CsaRecurrence::Standard,
            edges: vec![
                edge(CsaStateRole::CompressedKv, "l1_past_ckv", "l1_present_ckv"),
                edge(
                    CsaStateRole::CompressionCarry,
                    "l1_past_cc",
                    "l1_present_cc",
                ),
            ],
        };
        let inputs = vec![
            meta("l0_past_ckv", DataType::Uint8, &[1, 0, 583]),
            meta("l0_past_cc", DataType::Float32, &[1, 512]),
            meta("l0_past_ik", DataType::Uint8, &[1, 0, 68]),
            meta("l0_past_ic", DataType::Float32, &[1, 128]),
            meta("l1_past_ckv", DataType::Float32, &[1, 0, 512]),
            meta("l1_past_cc", DataType::Float32, &[1, 512]),
        ];
        let outputs = vec![
            meta("l0_present_ckv", DataType::Uint8, &[1, 1, 583]),
            meta("l0_present_cc", DataType::Float32, &[1, 512]),
            meta("l0_present_ik", DataType::Uint8, &[1, 1, 68]),
            meta("l0_present_ic", DataType::Float32, &[1, 128]),
            meta("l1_present_ckv", DataType::Float32, &[1, 1, 512]),
            meta("l1_present_cc", DataType::Float32, &[1, 512]),
        ];
        (vec![layer0, layer1], inputs, outputs)
    }

    #[test]
    fn alternating_ratio4_ratio128_layers_resolve_all_edges() {
        let (groups, inputs, outputs) = alternating_groups();
        let edges = resolve_csa_state_groups(&inputs, &outputs, &groups, &HashSet::new()).unwrap();
        assert_eq!(edges.len(), 6);
        // Ratio-4 layer contributes its four edges first, in role order.
        assert_eq!(edges[0].0, "l0_past_ckv");
        assert_eq!(edges[3].0, "l0_past_ic");
        // Ratio-128 layer contributes its two edges next.
        assert_eq!(edges[4].0, "l1_past_ckv");
        assert_eq!(edges[5].1, "l1_present_cc");
    }

    #[test]
    fn two_groups_sharing_a_port_are_refused() {
        let (mut groups, mut inputs, mut outputs) = alternating_groups();
        // Make the ratio-128 layer reuse the ratio-4 layer's compressed-KV past
        // input: a cross-group collision the accumulating resolver must reject.
        groups[1].edges[0].past_port = "l0_past_ckv".to_string();
        inputs.retain(|m| m.name != "l1_past_ckv");
        outputs.retain(|m| m.name != "l1_present_ckv"); // keep IO consistent
        outputs.push(meta("l1_present_ckv", DataType::Float32, &[1, 1, 512]));
        let err = resolve_csa_state_groups(&inputs, &outputs, &groups, &HashSet::new())
            .unwrap_err()
            .to_string();
        assert!(err.contains("overlaps"), "{err}");
    }

    /// The `f32` cache format threads its compressed-KV records as float32, not
    /// uint8: the ratio-128 HCA layer the official schedule builds keeps
    /// uncompressed float32 records, and the resolver must expect that dtype
    /// from the group's declared `cache_format` (mirroring the CSA op kernel's
    /// `CacheFormat::dtype()`), never a hard-coded uint8.
    #[test]
    fn f32_cache_expects_float32_compressed_kv_records() {
        let group = CsaStateGroupAbi {
            ratio: CsaCompressionRatio::Ratio128,
            cache_format: CsaCacheFormat::F32,
            recurrence: CsaRecurrence::Standard,
            edges: vec![
                edge(CsaStateRole::CompressedKv, "past_ckv", "present_ckv"),
                edge(CsaStateRole::CompressionCarry, "past_cc", "present_cc"),
            ],
        };
        let inp = vec![
            meta("past_ckv", DataType::Float32, &[1, 0, 128]),
            meta("past_cc", DataType::Float32, &[1, 512]),
        ];
        let out = vec![
            meta("present_ckv", DataType::Float32, &[1, 1, 128]),
            meta("present_cc", DataType::Float32, &[1, 512]),
        ];
        let edges = resolve_csa_state_edges(&inp, &out, &group, &HashSet::new()).unwrap();
        assert_eq!(
            edges,
            vec![
                ("past_ckv".to_string(), "present_ckv".to_string()),
                ("past_cc".to_string(), "present_cc".to_string()),
            ]
        );
    }

    /// The dtype the resolver expects is a function of the declared cache
    /// format: under an `f32` cache, uint8 compressed-KV records are the wrong
    /// dtype and must be refused before allocation.
    #[test]
    fn f32_cache_refuses_uint8_compressed_kv_records() {
        let group = CsaStateGroupAbi {
            ratio: CsaCompressionRatio::Ratio128,
            cache_format: CsaCacheFormat::F32,
            recurrence: CsaRecurrence::Standard,
            edges: vec![
                edge(CsaStateRole::CompressedKv, "past_ckv", "present_ckv"),
                edge(CsaStateRole::CompressionCarry, "past_cc", "present_cc"),
            ],
        };
        let inp = vec![
            meta("past_ckv", DataType::Uint8, &[1, 0, 512]),
            meta("past_cc", DataType::Float32, &[1, 512]),
        ];
        let out = vec![
            meta("present_ckv", DataType::Uint8, &[1, 1, 512]),
            meta("present_cc", DataType::Float32, &[1, 512]),
        ];
        let err = resolve_csa_state_edges(&inp, &out, &group, &HashSet::new())
            .unwrap_err()
            .to_string();
        assert!(err.contains("dtype"), "{err}");
    }
}
