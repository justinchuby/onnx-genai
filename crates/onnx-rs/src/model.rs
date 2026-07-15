//! Owned model type and ergonomic model I/O (ONNX_RS §3 / §4).
//!
//! `onnx-rs` deliberately does **not** own an IR: the graph, node, value, tensor
//! and weight types all come from the shared [`onnx_runtime_ir`] crate
//! (ONNX_RS §4.1 "Shared Crate"), and the protobuf parse/encode stack is reused
//! from [`onnx_runtime_loader`] (ONNX_RS §3.4 "Built-in: Protobuf Format").
//!
//! What this module adds is an *owned, self-contained* [`Model`] — the loader's
//! [`onnx_runtime_loader::Model`] only borrows a `Graph`, which is awkward to
//! pass around — plus the thin [`load_model`] / [`save_model`] entry points that
//! preserve model-level metadata (`ir_version`, producer, `metadata_props`, …)
//! across a round-trip. The loader's `load_model` drops that metadata, so we
//! decode it here from the same bytes.

use std::path::Path;
use std::sync::Arc;

use onnx_runtime_ir::Graph;
use onnx_runtime_loader::proto::{ModelProto, decode_model};
use onnx_runtime_loader::{
    Model as EncoderModel, ModelMetadata, WeightStore, load_model_bytes_with_weights, write_model,
};

use crate::check::{OnnxChecker, ValidationResult};
use crate::error::{Error, Result};
use crate::text::{self, PrintOptions};

/// An owned ONNX model: the shared-IR [`Graph`] plus the model-level metadata
/// and (optionally) the live weight store backing external initializers.
///
/// Unlike [`onnx_runtime_loader::Model`], which borrows a `&Graph`, this type
/// owns its graph so it can be returned from [`load_model`], mutated, dumped to
/// text, validated, and written back out.
pub struct Model {
    /// The computation graph, in the shared [`onnx_runtime_ir`] IR.
    pub graph: Graph,
    /// Model-level metadata not carried by the [`Graph`] itself.
    pub metadata: ModelMetadata,
    /// Live weight store backing any `External` initializers, kept alive so the
    /// model can be re-saved without re-mmapping. `None` for models built in
    /// memory with only inline weights.
    weights: Option<Arc<WeightStore>>,
}

impl Model {
    /// Wrap an in-memory [`Graph`] as a model with default metadata.
    ///
    /// The metadata's `opset_import` is still taken from `graph.opset_imports`
    /// at save time; only the header fields (`ir_version`, producer, …) default.
    pub fn new(graph: Graph) -> Self {
        Self {
            graph,
            metadata: ModelMetadata::default(),
            weights: None,
        }
    }

    /// Wrap a [`Graph`] together with explicit model-level metadata.
    pub fn with_metadata(graph: Graph, metadata: ModelMetadata) -> Self {
        Self {
            graph,
            metadata,
            weights: None,
        }
    }

    /// Attach the live weight store backing external initializers.
    pub fn set_weights(&mut self, weights: Arc<WeightStore>) {
        self.weights = Some(weights);
    }

    /// The live weight store, if this model carries one.
    pub fn weights(&self) -> Option<&Arc<WeightStore>> {
        self.weights.as_ref()
    }

    /// Render this model as a human-readable textual dump (ONNX_RS §5).
    pub fn to_text(&self) -> String {
        text::print(self)
    }

    /// Render this model as text with explicit [`PrintOptions`] (ONNX_RS §5.4).
    pub fn to_text_with(&self, opts: &PrintOptions) -> String {
        text::print_with(self, opts)
    }

    /// Parse a model previously rendered in the textual format (ONNX_RS §5.4).
    pub fn from_text(source: &str) -> Result<Self> {
        text::parse_model(source)
    }

    /// Validate this model with the default [`OnnxChecker`] (ONNX_RS §8).
    pub fn validate(&self) -> ValidationResult {
        OnnxChecker::new().check(self)
    }
}

/// Load an ONNX model from a `.onnx` protobuf file (ONNX_RS §3.4).
///
/// This reuses the runtime's loader pipeline (parse → build IR → resolve
/// weights → shape inference) and additionally decodes the model-level metadata
/// (`ir_version`, producer, `doc_string`, `metadata_props`, graph name) so that
/// a subsequent [`save_model`] reproduces the header faithfully. External
/// initializer data is resolved relative to the model file's directory and the
/// backing memory maps are kept alive on the returned [`Model`].
pub fn load_model(path: impl AsRef<Path>) -> Result<Model> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| Error::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let proto = decode_model(&bytes)?;
    let metadata = metadata_from_proto(&proto);

    let model_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let (graph, store) = load_model_bytes_with_weights(&bytes, model_dir)?;

    Ok(Model {
        graph,
        metadata,
        weights: Some(store),
    })
}

/// Save a [`Model`] to a `.onnx` protobuf file (ONNX_RS §3.4).
///
/// Delegates to the loader's encoder, which serialises the shared-IR graph plus
/// the model metadata. If the model carries external initializers it must also
/// carry the live [`WeightStore`] (as produced by [`load_model`]) so the encoder
/// can read their bytes.
pub fn save_model(model: &Model, path: impl AsRef<Path>) -> Result<()> {
    let mut encoder = EncoderModel::new(&model.graph).with_metadata(model.metadata.clone());
    if let Some(weights) = &model.weights {
        encoder = encoder.with_weights(weights);
    }
    write_model(&encoder, path)?;
    Ok(())
}

/// Project the header fields of a decoded [`ModelProto`] into [`ModelMetadata`].
///
/// The runtime loader keeps only `opset_imports` on the `Graph`; everything else
/// is recovered here so the round-trip is faithful.
fn metadata_from_proto(proto: &ModelProto) -> ModelMetadata {
    ModelMetadata {
        ir_version: proto.ir_version,
        producer_name: proto.producer_name.clone(),
        producer_version: proto.producer_version.clone(),
        domain: proto.domain.clone(),
        model_version: proto.model_version,
        doc_string: if proto.doc_string.is_empty() {
            None
        } else {
            Some(proto.doc_string.clone())
        },
        graph_name: proto
            .graph
            .as_ref()
            .map(|g| g.name.clone())
            .unwrap_or_default(),
        metadata_props: proto
            .metadata_props
            .iter()
            .map(|entry| (entry.key.clone(), entry.value.clone()))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{DataType, Node, NodeId, static_shape};

    /// Build a tiny `Z = Add(X, Y)` model and round-trip it through disk.
    fn add_graph() -> Graph {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 21);
        let x = g.create_named_value("X", DataType::Float32, static_shape([2, 3]));
        let y = g.create_named_value("Y", DataType::Float32, static_shape([2, 3]));
        let z = g.create_named_value("Z", DataType::Float32, static_shape([2, 3]));
        g.add_input(x);
        g.add_input(y);
        // `insert_node` overwrites the id, so the placeholder `NodeId(0)` is fine.
        let mut node = Node::new(NodeId(0), "Add", vec![Some(x), Some(y)], vec![z]);
        node.name = "add0".to_string();
        g.insert_node(node);
        g.add_output(z);
        g
    }

    #[test]
    fn new_wraps_graph_with_default_metadata() {
        let model = Model::new(add_graph());
        assert_eq!(model.metadata, ModelMetadata::default());
        assert_eq!(model.graph.num_nodes(), 1);
    }

    #[test]
    fn save_then_load_round_trips_structure_and_metadata() {
        let mut meta = ModelMetadata::default();
        meta.producer_name = "onnx-rs-test".to_string();
        meta.graph_name = "g".to_string();
        meta.metadata_props = vec![("author".to_string(), "deckard".to_string())];
        let model = Model::with_metadata(add_graph(), meta.clone());

        let dir = std::env::current_dir().unwrap().join("target");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("onnx_rs_roundtrip_test.onnx");
        save_model(&model, &path).unwrap();

        let loaded = load_model(&path).unwrap();
        assert_eq!(loaded.graph.num_nodes(), 1);
        assert_eq!(loaded.metadata.producer_name, "onnx-rs-test");
        assert_eq!(loaded.metadata.graph_name, "g");
        assert_eq!(
            loaded.metadata.metadata_props,
            vec![("author".to_string(), "deckard".to_string())]
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_missing_file_is_read_error() {
        let result = load_model("definitely-not-a-real-file.onnx");
        assert!(matches!(result, Err(Error::Read { .. })));
    }
}
