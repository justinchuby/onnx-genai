//! The model → text printer (ONNX_RS §5.4 "Implementation reference").
//!
//! Output shape, for a model `Z = Add(X, Y)`:
//!
//! ```text
//! <
//!   ir_version: 10,
//!   opset_import: ["" : 21]
//! >
//! main (float32[2, 3] X, float32[2, 3] Y) => (float32[2, 3] Z) {
//!   Z = Add(X, Y)
//! }
//! ```
//!
//! Design principles honoured here (§5.3): SSA-like syntax, `dtype[shape]`
//! types, compact `<attr = value>` attributes, `//` comments, weights as
//! references (never inlined), and nested `graph { ... }` blocks for control-flow
//! subgraphs.

use std::fmt::Write as _;

use onnx_runtime_ir::{Attribute, DataType, Dim, Graph, NodeId, Shape, ValueId, WeightRef};

use crate::model::Model;

/// Options controlling textual output (ONNX_RS §5.4 `PrintOptions`).
#[derive(Clone, Debug)]
pub struct PrintOptions {
    /// Indentation unit for one nesting level (default `"  "`).
    pub indent: String,
    /// Emit a `// initializers` reference block listing weight name + type
    /// (never the data). Default `true`.
    pub weight_shapes_only: bool,
    /// Emit `doc_string`s as trailing `//` comments. Default `false`.
    pub doc_strings: bool,
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self {
            indent: "  ".to_string(),
            weight_shapes_only: true,
            doc_strings: false,
        }
    }
}

/// Print `model` as text using default [`PrintOptions`].
pub fn print(model: &Model) -> String {
    print_with(model, &PrintOptions::default())
}

/// Print `model` as text with explicit options.
pub fn print_with(model: &Model, opts: &PrintOptions) -> String {
    let mut out = String::new();
    let meta = &model.metadata;

    // Model header block: ir_version + opset imports (sorted for determinism).
    out.push_str("<\n");
    let _ = writeln!(out, "{}ir_version: {},", opts.indent, meta.ir_version);
    let mut imports: Vec<(&String, &u64)> = model.graph.opset_imports.iter().collect();
    imports.sort_by(|a, b| a.0.cmp(b.0));
    let imports_str = imports
        .iter()
        .map(|(domain, version)| format!("{:?} : {}", domain, version))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(out, "{}opset_import: [{}]", opts.indent, imports_str);
    out.push_str(">\n");

    let graph_name = if meta.graph_name.is_empty() {
        "main"
    } else {
        &meta.graph_name
    };
    print_graph(&mut out, &model.graph, graph_name, 0, opts);
    out
}

/// Print a graph body (top-level or a nested control-flow subgraph).
fn print_graph(out: &mut String, graph: &Graph, name: &str, depth: usize, opts: &PrintOptions) {
    let pad = opts.indent.repeat(depth);
    let inner = opts.indent.repeat(depth + 1);

    let inputs = graph
        .inputs
        .iter()
        .map(|&v| typed_value(graph, v))
        .collect::<Vec<_>>()
        .join(", ");
    let outputs = graph
        .outputs
        .iter()
        .map(|&v| typed_value(graph, v))
        .collect::<Vec<_>>()
        .join(", ");

    let _ = writeln!(out, "{}{} ({}) => ({}) {{", pad, name, inputs, outputs);

    // Initializers as references (never inlined data) — §5.3.
    if opts.weight_shapes_only && !graph.initializers.is_empty() {
        let _ = writeln!(out, "{}// initializers", inner);
        let mut inits: Vec<(&ValueId, &WeightRef)> = graph.initializers.iter().collect();
        inits.sort_by_key(|(v, _)| v.0);
        for (vid, weight) in inits {
            let _ = writeln!(
                out,
                "{}// {} {} = <{} data omitted>",
                inner,
                weight_type(weight),
                value_name(graph, *vid),
                weight_kind(weight),
            );
        }
    }

    // Nodes in topological order (falls back to arena order on a cycle so a
    // malformed graph still dumps rather than panics).
    let order = graph
        .topological_order()
        .unwrap_or_else(|_| graph.nodes.keys().collect());
    for nid in order {
        print_node(out, graph, nid, depth + 1, opts);
    }

    let _ = writeln!(out, "{}}}", pad);
}

/// Print one node, including any nested subgraph attributes.
fn print_node(out: &mut String, graph: &Graph, nid: NodeId, depth: usize, opts: &PrintOptions) {
    let pad = opts.indent.repeat(depth);
    let node = graph.node(nid);

    let outputs = node
        .outputs
        .iter()
        .map(|&v| value_name(graph, v))
        .collect::<Vec<_>>()
        .join(", ");

    // Op name, qualified with a non-default domain.
    let op = if node.domain.is_empty() || node.domain == "ai.onnx" {
        node.op_type.clone()
    } else {
        format!("{}.{}", node.domain, node.op_type)
    };

    // Scalar / list attributes rendered inline; subgraph attributes deferred to
    // nested blocks after the node line.
    let mut inline_attrs: Vec<(&String, &Attribute)> = node
        .attributes
        .iter()
        .filter(|(_, a)| !is_subgraph_attr(a))
        .collect();
    inline_attrs.sort_by(|a, b| a.0.cmp(b.0));
    let attr_str = if inline_attrs.is_empty() {
        String::new()
    } else {
        let body = inline_attrs
            .iter()
            .map(|(k, v)| format!("{} = {}", k, attr_value(v)))
            .collect::<Vec<_>>()
            .join(", ");
        format!(" <{}>", body)
    };

    let inputs = node
        .inputs
        .iter()
        .map(|slot| match slot {
            Some(v) => value_name(graph, *v),
            None => "".to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ");

    let doc = if opts.doc_strings {
        match &node.doc_string {
            Some(d) if !d.is_empty() => format!("  // {}", d),
            _ => String::new(),
        }
    } else {
        String::new()
    };

    let lhs = if outputs.is_empty() {
        String::new()
    } else {
        format!("{} = ", outputs)
    };
    let _ = writeln!(out, "{}{}{}{}({}){}", pad, lhs, op, attr_str, inputs, doc);

    // Nested subgraph bodies (If/Loop/Scan) — §5.3.
    let mut subgraph_attrs: Vec<&String> = node
        .attributes
        .iter()
        .filter(|(_, a)| is_subgraph_attr(a))
        .map(|(k, _)| k)
        .collect();
    subgraph_attrs.sort();
    for attr_name in subgraph_attrs {
        if let Some(sub) = graph.subgraphs.get(&(nid, attr_name.clone())) {
            let _ = writeln!(out, "{}{} = graph", opts.indent.repeat(depth), attr_name);
            print_graph(out, sub, "", depth, opts);
        }
    }
}

fn is_subgraph_attr(attr: &Attribute) -> bool {
    matches!(attr, Attribute::Graph(_) | Attribute::Graphs(_))
}

/// A value rendered as `dtype[shape] name` (or just `dtype name` for a scalar).
fn typed_value(graph: &Graph, vid: ValueId) -> String {
    let value = graph.value(vid);
    let ty = type_string(value.dtype, &value.shape, graph);
    format!("{} {}", ty, value_name(graph, vid))
}

/// `dtype[d0, d1, ...]`, with symbolic dims shown by name.
fn type_string(dtype: DataType, shape: &Shape, graph: &Graph) -> String {
    let dims = shape
        .iter()
        .map(|d| dim_string(*d, graph))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{}[{}]", dtype_name(dtype), dims)
}

fn dim_string(dim: Dim, graph: &Graph) -> String {
    match dim {
        Dim::Static(n) => n.to_string(),
        Dim::Symbolic(sid) => graph
            .symbol_constraints
            .get(&sid)
            .and_then(|c| c.name.clone())
            .unwrap_or_else(|| format!("s{}", sid.0)),
    }
}

/// The name of a value, falling back to an SSA-style `%vN` for anonymous values.
fn value_name(graph: &Graph, vid: ValueId) -> String {
    match graph.try_value(vid).and_then(|v| v.name.as_deref()) {
        Some(name) if !name.is_empty() => name.to_string(),
        _ => format!("%v{}", vid.0),
    }
}

fn weight_type(weight: &WeightRef) -> String {
    let dims = weight
        .dims()
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("{}[{}]", dtype_name(weight.dtype()), dims)
}

fn weight_kind(weight: &WeightRef) -> &'static str {
    match weight {
        WeightRef::Inline(_) => "inline",
        WeightRef::External { .. } => "external",
    }
}

/// Render a scalar or list attribute value compactly.
fn attr_value(attr: &Attribute) -> String {
    match attr {
        Attribute::Int(v) => v.to_string(),
        Attribute::Float(v) => format!("{:?}", v),
        Attribute::String(bytes) => match std::str::from_utf8(bytes) {
            Ok(s) => format!("{:?}", s),
            Err(_) => format!("<{} bytes>", bytes.len()),
        },
        Attribute::Ints(v) => format!(
            "[{}]",
            v.iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Attribute::Floats(v) => format!(
            "[{}]",
            v.iter()
                .map(|f| format!("{:?}", f))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Attribute::Strings(v) => format!("<{} strings>", v.len()),
        Attribute::Tensor(t) => format!("<tensor {}[{:?}]>", dtype_name(t.dtype), t.dims),
        Attribute::SparseTensor(_) => "<sparse tensor>".to_string(),
        Attribute::TypeProto(_) => "<type>".to_string(),
        // Subgraph attributes are printed as nested blocks, not inline.
        Attribute::Graph(_) | Attribute::Graphs(_) => "<graph>".to_string(),
    }
}

/// ONNX textual dtype spellings.
fn dtype_name(dtype: DataType) -> &'static str {
    match dtype {
        DataType::Float32 => "float32",
        DataType::Uint8 => "uint8",
        DataType::Int8 => "int8",
        DataType::Uint16 => "uint16",
        DataType::Int16 => "int16",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::String => "string",
        DataType::Bool => "bool",
        DataType::Float16 => "float16",
        DataType::Float64 => "float64",
        DataType::Uint32 => "uint32",
        DataType::Uint64 => "uint64",
        DataType::BFloat16 => "bfloat16",
        DataType::Float8E4M3FN => "float8e4m3fn",
        DataType::Float8E4M3FNUZ => "float8e4m3fnuz",
        DataType::Float8E5M2 => "float8e5m2",
        DataType::Float8E5M2FNUZ => "float8e5m2fnuz",
        DataType::Uint4 => "uint4",
        DataType::Int4 => "int4",
        DataType::Float4E2M1 => "float4e2m1",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Node, TensorData, static_shape};
    use onnx_runtime_loader::ModelMetadata;

    fn add_model() -> Model {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 21);
        let x = g.create_named_value("X", DataType::Float32, static_shape([2, 3]));
        let y = g.create_named_value("Y", DataType::Float32, static_shape([2, 3]));
        let z = g.create_named_value("Z", DataType::Float32, static_shape([2, 3]));
        g.add_input(x);
        g.add_input(y);
        let mut node = Node::new(NodeId(0), "Add", vec![Some(x), Some(y)], vec![z]);
        node.name = "add0".to_string();
        g.insert_node(node);
        g.add_output(z);
        Model::with_metadata(g, ModelMetadata::default())
    }

    #[test]
    fn dumps_header_signature_and_node() {
        let text = print(&add_model());
        assert!(text.contains("ir_version: 10"), "header:\n{text}");
        assert!(text.contains("opset_import: [\"\" : 21]"), "opset:\n{text}");
        assert!(
            text.contains("main (float32[2, 3] X, float32[2, 3] Y) => (float32[2, 3] Z)"),
            "signature:\n{text}"
        );
        assert!(text.contains("Z = Add(X, Y)"), "node:\n{text}");
    }

    #[test]
    fn renders_inline_attribute() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 21);
        let x = g.create_named_value("X", DataType::Float32, static_shape([4]));
        let y = g.create_named_value("Y", DataType::Float32, static_shape([4]));
        g.add_input(x);
        let mut node = Node::new(NodeId(0), "LeakyRelu", vec![Some(x)], vec![y]);
        node.attributes
            .insert("alpha".to_string(), Attribute::Float(0.1));
        g.insert_node(node);
        g.add_output(y);
        let text = print(&Model::new(g));
        assert!(text.contains("Y = LeakyRelu <alpha = 0.1>(X)"), "{text}");
    }

    #[test]
    fn initializers_are_references_not_data() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 21);
        let w = g.create_named_value("W", DataType::Float32, static_shape([2]));
        let init = TensorData::from_raw(DataType::Float32, vec![2], vec![0u8; 8]);
        g.set_initializer(w, WeightRef::Inline(init));
        let text = print(&Model::new(g));
        assert!(text.contains("// initializers"), "{text}");
        assert!(
            text.contains("float32[2] W = <inline data omitted>"),
            "{text}"
        );
        // The raw bytes must never appear inline.
        assert!(!text.contains("\\x00"), "{text}");
    }
}
