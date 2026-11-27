//! Content identity fixed at model load for opt-in Native state files.
//! Selective prost messages skip inline payloads without copying them; binary
//! models are mapped while hashing, including every declared external region.
use super::*;
use prost::Message;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::io::{Read, Seek, SeekFrom};

pub(super) fn identity(
    directory: &onnx_genai_ort::ModelDirectory,
    metadata: &InferenceMetadata,
    tokenizer: &Tokenizer,
    precision: onnx_runtime_session::DecodePrecision,
    graph: &onnx_runtime_ir::Graph,
    keyed: &crate::keyed_weights::GraphWeightFiles,
) -> anyhow::Result<[u8; 32]> {
    let file = std::fs::File::open(&directory.model_path).with_context(|| {
        format!(
            "open Native state identity model '{}'",
            directory.model_path.display()
        )
    })?;
    let converted;
    let mapped;
    let bytes = if onnx_runtime_loader::is_textproto_path(&directory.model_path) {
        converted = onnx_runtime_loader::read_model_binary(&directory.model_path)?;
        converted.as_slice()
    } else {
        // SAFETY: As for runtime weight mappings, package artifacts are immutable
        // during loading. This read-only mapping is retained throughout hashing
        // and selective scanning; no pointer or borrowed payload escapes.
        mapped = unsafe { memmap2::MmapOptions::new().map(&file) }.with_context(|| {
            format!(
                "map Native state identity model '{}'",
                directory.model_path.display()
            )
        })?;
        &mapped[..]
    };
    let model = ScanModel::decode(bytes)
        .context("scan Native state identity external-weight declarations")?;
    let mut hash = Sha256::new();
    part(&mut hash, b"nxrt.native-state.model.v1");
    part(&mut hash, bytes);
    let mut regions = BTreeSet::new();
    if let Some(graph) = model.graph {
        graph.collect(
            &mut regions,
            keyed.for_graph(&directory.model_path)?.is_some(),
        )?;
    }
    for function in model.functions {
        for node in function.nodes {
            node.collect(&mut regions)?;
        }
        for attr in function.attributes {
            attr.collect(&mut regions)?;
        }
    }
    let base = directory
        .model_path
        .parent()
        .context("Native model path has no parent")?;
    hash_contexts(&mut hash, graph, base)?;
    let mut buffer = [0_u8; 64 * 1024];
    for (location, offset, length) in regions {
        anyhow::ensure!(
            !Path::new(&location).is_absolute()
                && Path::new(&location).components().all(|c| !matches!(
                    c,
                    std::path::Component::ParentDir
                        | std::path::Component::RootDir
                        | std::path::Component::Prefix(_)
                )),
            "Native state identity external location {location:?} escapes its model directory; use a model-relative weight path"
        );
        let path = base.join(&location);
        let mut file = std::fs::File::open(&path)
            .with_context(|| format!("open Native state identity weight '{}'", path.display()))?;
        let available = file.metadata()?.len().checked_sub(offset)
            .with_context(|| format!("external weight '{}' offset {offset} exceeds its file length; restore the original model package", path.display()))?;
        let length = length.unwrap_or(available);
        anyhow::ensure!(
            length <= available,
            "external weight '{}' region {offset}+{length} exceeds its file length; restore the original model package",
            path.display()
        );
        part(&mut hash, location.as_bytes());
        hash.update(offset.to_le_bytes());
        hash.update(length.to_le_bytes());
        file.seek(SeekFrom::Start(offset))?;
        let mut remaining = length;
        while remaining > 0 {
            let count = usize::try_from(remaining.min(buffer.len() as u64))?;
            file.read_exact(&mut buffer[..count])
                .with_context(|| format!("hash external weight '{}' at offset {}; restore a complete immutable model package", path.display(), offset + length - remaining))?;
            hash.update(&buffer[..count]);
            remaining -= count as u64;
        }
    }
    // A keyed initializer's bytes live in its declared container rather than
    // an ONNX location/offset region. Bind state files to complete immutable
    // containers, including their directory/index metadata and encoding.
    let sources = keyed.sources();
    if !sources.is_empty() {
        let root = directory
            .root
            .canonicalize()
            .context("resolve Native keyed-weight identity package root")?;
        let sources = sources
            .into_iter()
            .map(|path| {
                let label = path.strip_prefix(&root).with_context(|| {
                    format!(
                        "keyed-weight identity source '{}' escapes its admitted package",
                        path.display()
                    )
                })?;
                let label = label
                    .components()
                    .map(|part| {
                        part.as_os_str()
                            .to_str()
                            .map(str::to_owned)
                            .context("keyed-weight identity path must be UTF-8")
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?
                    .join("/");
                Ok((label, path))
            })
            .collect::<anyhow::Result<std::collections::BTreeMap<_, _>>>()?;
        part(&mut hash, b"nxrt.native-state.keyed-containers.v1");
        hash.update((sources.len() as u64).to_le_bytes());
        for (label, path) in sources {
            part(&mut hash, b"nxrt.native-state.keyed-container.v1");
            part(&mut hash, label.as_bytes());
            let mut file = std::fs::File::open(path).with_context(|| {
                format!(
                    "open Native keyed-weight identity source '{}'",
                    path.display()
                )
            })?;
            let length = file.metadata()?.len();
            hash.update(length.to_le_bytes());
            let mut remaining = length;
            while remaining > 0 {
                let count = usize::try_from(remaining.min(buffer.len() as u64))?;
                file.read_exact(&mut buffer[..count]).with_context(|| format!("hash keyed-weight identity source '{}' at byte {}; restore its complete immutable container", path.display(), length - remaining))?;
                hash.update(&buffer[..count]);
                remaining -= count as u64;
            }
        }
    }
    // Canonical object ordering prevents HashMap-backed tokenizer vocabulary
    // iteration from creating different identities on a server restart.
    for path in [
        directory.metadata_path.as_ref(),
        directory.genai_config_path.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("read Native state metadata identity '{}'", path.display()))?;
        part(
            &mut hash,
            onnx_genai_metadata::semantic_identity_of_str(&source)?.as_bytes(),
        );
    }
    canonical(&mut hash, &serde_json::to_value(metadata.decoder_io())?)?;
    canonical(
        &mut hash,
        &serde_json::from_str::<serde_json::Value>(
            &tokenizer.inner().to_string(false).map_err(|e| {
                anyhow::anyhow!("serialize tokenizer for Native state identity: {e}")
            })?,
        )?,
    )?;
    canonical(
        &mut hash,
        &serde_json::to_value(tokenizer.legacy_eos_token_ids())?,
    )?;
    part(&mut hash, format!("{precision:?}").as_bytes());
    Ok(hash.finalize().into())
}

fn hash_contexts(
    hash: &mut Sha256,
    graph: &onnx_runtime_ir::Graph,
    base: &Path,
) -> anyhow::Result<()> {
    // The loader owns external compiled-artifact semantics too. Hash typed
    // primary external payloads, not just the path bytes in their ONNX node.
    for context in onnx_runtime_loader::ep_context_nodes(graph)
        .filter(|c| c.main_context && c.embed_mode == onnx_runtime_loader::EmbedMode::ExternalFile)
    {
        let blob = onnx_runtime_loader::resolve_ep_context(base, &context)
            .context("resolve compiled artifact for Native state identity")?;
        part(hash, blob.bytes());
    }
    let mut children = graph.subgraphs.iter().collect::<Vec<_>>();
    children.sort_by_key(|(key, _)| *key);
    for (_, child) in children {
        hash_contexts(hash, child, base)?;
    }
    Ok(())
}

fn part(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}
fn canonical(hash: &mut Sha256, value: &serde_json::Value) -> anyhow::Result<()> {
    match value {
        serde_json::Value::Object(map) => {
            part(hash, b"object");
            hash.update((map.len() as u64).to_le_bytes());
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort();
            for key in keys {
                part(hash, key.as_bytes());
                canonical(hash, &map[key])?;
            }
        }
        serde_json::Value::Array(items) => {
            part(hash, b"array");
            hash.update((items.len() as u64).to_le_bytes());
            for item in items {
                canonical(hash, item)?;
            }
        }
        other => part(hash, &serde_json::to_vec(other)?),
    }
    Ok(())
}

type Regions = BTreeSet<(String, u64, Option<u64>)>;
#[derive(Clone, PartialEq, Message)]
struct ScanModel {
    #[prost(message, optional, tag = "7")]
    graph: Option<ScanGraph>,
    #[prost(message, repeated, tag = "25")]
    functions: Vec<ScanFunction>,
}
#[derive(Clone, PartialEq, Message)]
struct ScanGraph {
    #[prost(message, repeated, tag = "1")]
    nodes: Vec<ScanNode>,
    #[prost(message, repeated, tag = "5")]
    tensors: Vec<ScanTensor>,
    #[prost(message, repeated, tag = "15")]
    sparse: Vec<ScanSparse>,
}
#[derive(Clone, PartialEq, Message)]
struct ScanFunction {
    #[prost(message, repeated, tag = "7")]
    nodes: Vec<ScanNode>,
    #[prost(message, repeated, tag = "11")]
    attributes: Vec<ScanAttribute>,
}
#[derive(Clone, PartialEq, Message)]
struct ScanNode {
    #[prost(message, repeated, tag = "5")]
    attributes: Vec<ScanAttribute>,
}
#[derive(Clone, PartialEq, Message)]
struct ScanAttribute {
    #[prost(message, optional, tag = "5")]
    tensor: Option<ScanTensor>,
    #[prost(message, optional, boxed, tag = "6")]
    graph: Option<Box<ScanGraph>>,
    #[prost(message, repeated, tag = "10")]
    tensors: Vec<ScanTensor>,
    #[prost(message, repeated, tag = "11")]
    graphs: Vec<ScanGraph>,
    #[prost(message, optional, tag = "22")]
    sparse: Option<ScanSparse>,
    #[prost(message, repeated, tag = "23")]
    sparses: Vec<ScanSparse>,
}
#[derive(Clone, PartialEq, Message)]
struct ScanSparse {
    #[prost(message, optional, tag = "1")]
    values: Option<ScanTensor>,
    #[prost(message, optional, tag = "2")]
    indices: Option<ScanTensor>,
}
#[derive(Clone, PartialEq, Message)]
struct ScanTensor {
    #[prost(int64, repeated, tag = "1")]
    dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    dtype: i32,
    #[prost(message, repeated, tag = "13")]
    external: Vec<ScanEntry>,
    #[prost(int32, tag = "14")]
    location: i32,
}
#[derive(Clone, PartialEq, Message)]
struct ScanEntry {
    #[prost(string, tag = "1")]
    key: String,
    #[prost(string, tag = "2")]
    value: String,
}
impl ScanGraph {
    fn collect(self, regions: &mut Regions, keyed_initializers: bool) -> anyhow::Result<()> {
        for node in self.nodes {
            node.collect(regions)?;
        }
        for tensor in self.tensors {
            tensor.collect(regions, keyed_initializers)?;
        }
        for sparse in self.sparse {
            sparse.collect(regions)?;
        }
        Ok(())
    }
}
impl ScanNode {
    fn collect(self, regions: &mut Regions) -> anyhow::Result<()> {
        for attr in self.attributes {
            attr.collect(regions)?;
        }
        Ok(())
    }
}
impl ScanAttribute {
    fn collect(self, regions: &mut Regions) -> anyhow::Result<()> {
        if let Some(t) = self.tensor {
            t.collect(regions, false)?;
        }
        if let Some(g) = self.graph {
            g.collect(regions, false)?;
        }
        for t in self.tensors {
            t.collect(regions, false)?;
        }
        for g in self.graphs {
            g.collect(regions, false)?;
        }
        if let Some(s) = self.sparse {
            s.collect(regions)?;
        }
        for s in self.sparses {
            s.collect(regions)?;
        }
        Ok(())
    }
}
impl ScanSparse {
    fn collect(self, regions: &mut Regions) -> anyhow::Result<()> {
        if let Some(t) = self.values {
            t.collect(regions, false)?;
        }
        if let Some(t) = self.indices {
            t.collect(regions, false)?;
        }
        Ok(())
    }
}
impl ScanTensor {
    fn collect(self, regions: &mut Regions, keyed: bool) -> anyhow::Result<()> {
        if self.location != 1 {
            return Ok(());
        }
        let mut fields = std::collections::BTreeMap::new();
        for item in self.external {
            anyhow::ensure!(
                fields.insert(item.key.clone(), item.value).is_none(),
                "duplicate external weight descriptor {:?}; repair the model package before enabling Native state files",
                item.key
            );
        }
        if keyed && fields.get("location").is_none_or(String::is_empty) {
            // The loaded Native graph already proved these bindings through
            // the keyed loader. Container identity is hashed separately above.
            return Ok(());
        }
        let location = fields
            .get("location")
            .context("external weight is missing location; restore the original model package")?
            .clone();
        let parse = |key: &str| -> anyhow::Result<Option<u64>> {
            fields
                .get(key)
                .map(|v| {
                    v.parse::<u64>().with_context(|| {
                        format!("invalid external weight {key}={v:?}; use an unsigned byte count")
                    })
                })
                .transpose()
        };
        let inferred = || -> anyhow::Result<u64> {
            let kind = onnx_runtime_ir::DataType::from_onnx(self.dtype)
                .context("external weight has an invalid dtype; restore its original model")?;
            let shape = self
                .dims
                .iter()
                .map(|dim| {
                    usize::try_from(*dim)
                        .context("external weight has a negative or overflowing dimension")
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            u64::try_from(onnx_runtime_ir::checked_expected_bytes(kind, &shape)
                .context("external weight has unsupported or overflowing storage geometry; declare an explicit byte length")?).map_err(Into::into)
        };
        let length = match parse("length")? {
            Some(n) => n,
            None => inferred()?,
        };
        regions.insert((location, parse("offset")?.unwrap_or(0), Some(length)));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_loader::proto::onnx::{self, StringStringEntryProto};

    fn external(location: &str, offset: u64, length: Option<u64>) -> onnx::TensorProto {
        let mut external_data = vec![
            StringStringEntryProto {
                key: "location".into(),
                value: location.into(),
            },
            StringStringEntryProto {
                key: "offset".into(),
                value: offset.to_string(),
            },
        ];
        if let Some(n) = length {
            external_data.push(StringStringEntryProto {
                key: "length".into(),
                value: n.to_string(),
            });
        }
        onnx::TensorProto {
            name: location.into(),
            data_location: 1,
            data_type: 2,
            dims: vec![2],
            external_data,
            ..Default::default()
        }
    }
    fn scan(model: &onnx::ModelProto) -> anyhow::Result<Regions> {
        let scanned = ScanModel::decode(model.encode_to_vec().as_slice())?;
        let mut result = Regions::new();
        if let Some(g) = scanned.graph {
            g.collect(&mut result, false)?;
        }
        for f in scanned.functions {
            for node in f.nodes {
                node.collect(&mut result)?;
            }
            for attr in f.attributes {
                attr.collect(&mut result)?;
            }
        }
        Ok(result)
    }
    #[test]
    fn state_identity_scan_covers_dense_sparse_attributes_nested_graphs_and_functions()
    -> anyhow::Result<()> {
        let attr = onnx::AttributeProto {
            t: Some(external("attr", 0, Some(2))),
            tensors: vec![external("attrs", 1, None)],
            g: Some(onnx::GraphProto {
                initializer: vec![external("body", 2, None)],
                ..Default::default()
            }),
            graphs: vec![onnx::GraphProto {
                initializer: vec![external("bodies", 3, None)],
                ..Default::default()
            }],
            sparse_tensor: Some(onnx::SparseTensorProto {
                values: Some(external("attr-sparse-value", 4, None)),
                indices: Some(external("attr-sparse-index", 5, None)),
                ..Default::default()
            }),
            sparse_tensors: vec![onnx::SparseTensorProto {
                values: Some(external("attrs-sparse", 6, None)),
                ..Default::default()
            }],
            ..Default::default()
        };
        let model = onnx::ModelProto {
            graph: Some(onnx::GraphProto {
                initializer: vec![external("dense", 7, None)],
                sparse_initializer: vec![onnx::SparseTensorProto {
                    values: Some(external("sparse", 8, None)),
                    ..Default::default()
                }],
                node: vec![onnx::NodeProto {
                    attribute: vec![attr],
                    ..Default::default()
                }],
                ..Default::default()
            }),
            functions: vec![onnx::FunctionProto {
                node: vec![onnx::NodeProto {
                    attribute: vec![onnx::AttributeProto {
                        t: Some(external("function-node", 9, None)),
                        ..Default::default()
                    }],
                    ..Default::default()
                }],
                attribute_proto: vec![onnx::AttributeProto {
                    t: Some(external("function-default", 10, None)),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let found = scan(&model)?;
        assert_eq!(found.len(), 11);
        for (name, offset) in [
            ("attr", 0),
            ("attrs", 1),
            ("body", 2),
            ("bodies", 3),
            ("attr-sparse-value", 4),
            ("attr-sparse-index", 5),
            ("attrs-sparse", 6),
            ("dense", 7),
            ("sparse", 8),
            ("function-node", 9),
            ("function-default", 10),
        ] {
            assert!(
                found.contains(&(name.into(), offset, Some(2))),
                "missing {name}"
            );
        }
        Ok(())
    }

    #[test]
    fn state_identity_hashes_external_region_content_and_ignores_padding() -> anyhow::Result<()> {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-native-engine");
        let root = tempfile::tempdir()?;
        let model_path = root.path().join("model.onnx");
        let model = onnx::ModelProto {
            graph: Some(onnx::GraphProto {
                initializer: vec![external("weights", 1, None)],
                ..Default::default()
            }),
            ..Default::default()
        };
        std::fs::write(&model_path, model.encode_to_vec())?;
        let weights = root.path().join("weights");
        std::fs::write(&weights, [0, 1, 2, 3])?;
        let directory = onnx_genai_ort::ModelDirectory {
            root: root.path().into(),
            model_path,
            tokenizer_path: fixture.join("tokenizer.json"),
            metadata_path: Some(fixture.join("inference_metadata.yaml")),
            genai_config_path: None,
        };
        let metadata =
            onnx_genai_metadata::load_metadata(directory.metadata_path.as_ref().unwrap())?;
        let tokenizer = Tokenizer::from_file(&directory.tokenizer_path)?;
        let graph = onnx_runtime_ir::Graph::new();
        let hash = || {
            identity(
                &directory,
                &metadata,
                &tokenizer,
                onnx_runtime_session::DecodePrecision::Model,
                &graph,
                &crate::keyed_weights::GraphWeightFiles::default(),
            )
        };
        let original = hash()?;
        std::fs::write(&weights, [9, 1, 2, 8])?;
        assert_eq!(
            hash()?,
            original,
            "unreferenced prefix/trailer is not model weight state"
        );
        std::fs::write(&weights, [9, 1, 7, 8])?;
        assert_ne!(
            hash()?,
            original,
            "referenced bytes must be part of identity"
        );
        Ok(())
    }

    #[test]
    fn state_identity_canonicalizes_objects_and_hashes_external_compiled_payloads()
    -> anyhow::Result<()> {
        let mut a = Sha256::new();
        let mut b = Sha256::new();
        canonical(
            &mut a,
            &serde_json::from_str(r#"{"a":1,"b":{"x":2,"y":3}}"#)?,
        )?;
        canonical(
            &mut b,
            &serde_json::from_str(r#"{"b":{"y":3,"x":2},"a":1}"#)?,
        )?;
        assert_eq!(a.finalize(), b.finalize());
        let root = tempfile::tempdir()?;
        let file = root.path().join("compiled.bin");
        std::fs::write(&file, b"compiled-weights-a")?;
        let mut graph = onnx_runtime_ir::Graph::new();
        let mut node =
            onnx_runtime_ir::Node::new(onnx_runtime_ir::NodeId(0), "EPContext", vec![], vec![]);
        node.domain = "com.microsoft".into();
        node.attributes
            .insert("embed_mode".into(), onnx_runtime_ir::Attribute::Int(0));
        node.attributes.insert(
            "ep_cache_context".into(),
            onnx_runtime_ir::Attribute::String(b"compiled.bin".to_vec()),
        );
        graph.insert_node(node);
        let hash = || -> anyhow::Result<_> {
            let mut h = Sha256::new();
            hash_contexts(&mut h, &graph, root.path())?;
            Ok(h.finalize())
        };
        let original = hash()?;
        std::fs::write(&file, b"compiled-weights-b")?;
        assert_ne!(hash()?, original);
        Ok(())
    }
}
