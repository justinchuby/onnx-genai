//! Parser for the textual representation emitted by [`super::ser`].

use std::collections::HashMap;
use std::path::PathBuf;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use onnx_runtime_ir::{
    Attribute, DataType, Dim, Graph, Node, NodeId, Shape, SparseTensorData, TensorData, TypeProto,
    ValueId, WeightRef,
};
use onnx_runtime_loader::ModelMetadata;
use onnx_runtime_loader::proto::ModelProto;
use prost::Message;

use crate::error::{Error, Result};
use crate::model::Model;

#[derive(Clone, Copy)]
struct Line<'a> {
    number: usize,
    text: &'a str,
}

struct Parser<'a> {
    lines: Vec<Line<'a>>,
    pos: usize,
}

/// Parse a model rendered by [`crate::text::to_text`].
///
/// Binary initializer and tensor-attribute payloads are not present in the
/// textual format. They are reconstructed as zero-filled placeholders with the
/// printed dtype and shape; external initializers retain their external kind
/// but necessarily use an empty path and zero offset.
pub fn from_text(source: &str) -> Result<Model> {
    Parser::new(source).parse()
}

impl<'a> Parser<'a> {
    fn new(source: &'a str) -> Self {
        let lines = source
            .lines()
            .enumerate()
            .filter_map(|(index, text)| {
                let text = text.trim();
                (!text.is_empty()).then_some(Line {
                    number: index + 1,
                    text,
                })
            })
            .collect();
        Self { lines, pos: 0 }
    }

    fn parse(mut self) -> Result<Model> {
        let open = self.next()?;
        if open.text != "<" {
            return self.fail(open.number, "expected model header '<'");
        }

        let mut metadata = ModelMetadata::default();
        let mut imports = HashMap::new();
        let mut retained_proto = None;
        loop {
            let line = self.next()?;
            if line.text == ">" {
                break;
            }
            if let Some(value) = line.text.strip_prefix("ir_version:") {
                metadata.ir_version = trim_comma(value)
                    .parse()
                    .map_err(|_| self.error(line.number, "invalid ir_version"))?;
            } else if let Some(value) = line.text.strip_prefix("opset_import:") {
                imports = parse_opset_imports(value.trim(), line.number)?;
            } else if let Some(value) = line.text.strip_prefix("proto:") {
                let encoded: String = serde_yaml::from_str(trim_comma(value).trim())
                    .map_err(|error| self.error(line.number, format!("invalid proto: {error}")))?;
                let bytes = BASE64
                    .decode(encoded)
                    .map_err(|error| self.error(line.number, format!("invalid proto: {error}")))?;
                retained_proto =
                    Some(ModelProto::decode(bytes.as_slice()).map_err(|error| {
                        self.error(line.number, format!("invalid proto: {error}"))
                    })?);
            } else {
                return self.fail(line.number, "unknown model header field");
            }
        }

        if let Some(proto) = retained_proto {
            return Model::from_proto(proto);
        }

        let (graph_name, mut graph) = self.parse_graph()?;
        metadata.graph_name = graph_name;
        graph.opset_imports = imports;
        if self.pos != self.lines.len() {
            return self.fail(
                self.lines[self.pos].number,
                "unexpected content after graph",
            );
        }
        Ok(Model::with_metadata(graph, metadata))
    }

    fn parse_graph(&mut self) -> Result<(String, Graph)> {
        let signature = self.next()?;
        if !signature.text.ends_with('{') {
            let open = self.next()?;
            if open.text != "{" {
                return self.fail(open.number, "expected graph body '{'");
            }
        }
        let (name, inputs, outputs) = parse_signature(signature)?;
        let mut graph = Graph::new();
        let mut values = HashMap::new();

        for typed in inputs {
            let id = create_typed_value(&mut graph, &typed, signature.number)?;
            graph.add_input(id);
            values.insert(typed.name, id);
        }
        for typed in outputs {
            let id = match values.get(&typed.name) {
                Some(id) => *id,
                None => {
                    let id = create_typed_value(&mut graph, &typed, signature.number)?;
                    values.insert(typed.name, id);
                    id
                }
            };
            graph.add_output(id);
        }

        let mut last_node = None;
        loop {
            let line = self.peek()?;
            if line.text == "}" {
                self.pos += 1;
                break;
            }
            if line.text == "// initializers" {
                self.pos += 1;
                continue;
            }
            if let Some(init) = line.text.strip_prefix("// ")
                && init.contains(" data omitted>")
            {
                self.pos += 1;
                parse_initializer(init, line.number, &mut graph, &mut values)?;
                continue;
            }
            if let Some((attr_name, index)) = parse_subgraph_marker(line.text) {
                self.pos += 1;
                let (_, subgraph) = self.parse_graph()?;
                let node_id = last_node.ok_or_else(|| {
                    self.error(line.number, "subgraph attribute has no preceding node")
                })?;
                attach_subgraph(&mut graph, node_id, attr_name, index, subgraph);
                continue;
            }

            self.pos += 1;
            last_node = Some(parse_node(line, &mut graph, &mut values)?);
        }

        Ok((name, graph))
    }

    fn next(&mut self) -> Result<Line<'a>> {
        let line = self
            .lines
            .get(self.pos)
            .copied()
            .ok_or_else(|| self.error(self.last_line(), "unexpected end of input"))?;
        self.pos += 1;
        Ok(line)
    }

    fn peek(&self) -> Result<Line<'a>> {
        self.lines
            .get(self.pos)
            .copied()
            .ok_or_else(|| self.error(self.last_line(), "unexpected end of graph"))
    }

    fn last_line(&self) -> usize {
        self.lines.last().map_or(1, |line| line.number)
    }

    fn error(&self, line: usize, message: impl Into<String>) -> Error {
        Error::TextParse {
            line,
            message: message.into(),
        }
    }

    fn fail<T>(&self, line: usize, message: impl Into<String>) -> Result<T> {
        Err(self.error(line, message))
    }
}

#[derive(Debug)]
struct TypedValue {
    name: String,
    dtype: DataType,
    dims: Vec<String>,
}

fn parse_signature(line: Line<'_>) -> Result<(String, Vec<TypedValue>, Vec<TypedValue>)> {
    let arrow = line
        .text
        .find("=>")
        .ok_or_else(|| parse_error(line.number, "graph signature is missing '=>'"))?;
    let left = line.text[..arrow].trim();
    let right = line.text[arrow + 2..].trim();

    let left_open = left
        .find('(')
        .ok_or_else(|| parse_error(line.number, "graph inputs are missing"))?;
    let left_close = left
        .rfind(')')
        .ok_or_else(|| parse_error(line.number, "graph inputs are not closed"))?;
    let right = right.strip_suffix('{').unwrap_or(right).trim();
    let right_open = right
        .find('(')
        .ok_or_else(|| parse_error(line.number, "graph outputs are missing"))?;
    let right_close = right
        .rfind(')')
        .ok_or_else(|| parse_error(line.number, "graph outputs are not closed"))?;

    let name = left[..left_open].trim().to_string();
    let inputs = parse_typed_values(&left[left_open + 1..left_close], line.number)?;
    let outputs = parse_typed_values(&right[right_open + 1..right_close], line.number)?;
    Ok((name, inputs, outputs))
}

fn parse_typed_values(text: &str, line: usize) -> Result<Vec<TypedValue>> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    split_top_level(text, ',')
        .into_iter()
        .map(|item| parse_typed_value(item.trim(), line))
        .collect()
}

fn parse_typed_value(text: &str, line: usize) -> Result<TypedValue> {
    let open = text
        .find('[')
        .ok_or_else(|| parse_error(line, "typed value is missing '['"))?;
    let close = text
        .find(']')
        .ok_or_else(|| parse_error(line, "typed value is missing ']'"))?;
    let dtype = parse_dtype(text[..open].trim(), line)?;
    let name = text[close + 1..].trim();
    if name.is_empty() {
        return Err(parse_error(line, "typed value is missing a name"));
    }
    let dims = if text[open + 1..close].trim().is_empty() {
        Vec::new()
    } else {
        split_top_level(&text[open + 1..close], ',')
            .into_iter()
            .map(|dim| dim.trim().to_string())
            .collect()
    };
    Ok(TypedValue {
        name: name.to_string(),
        dtype,
        dims,
    })
}

fn create_typed_value(graph: &mut Graph, value: &TypedValue, line: usize) -> Result<ValueId> {
    let mut shape = Shape::new();
    for dim in &value.dims {
        let dim = unquote(dim);
        match dim.parse::<usize>() {
            Ok(size) => shape.push(Dim::Static(size)),
            Err(_) if !dim.is_empty() => {
                shape.push(Dim::Symbolic(graph.intern_symbol(&dim)));
            }
            Err(_) => return Err(parse_error(line, "empty dimension")),
        }
    }
    Ok(graph.create_named_value(&value.name, value.dtype, shape))
}

fn parse_node(
    line: Line<'_>,
    graph: &mut Graph,
    values: &mut HashMap<String, ValueId>,
) -> Result<NodeId> {
    let text = strip_trailing_comment(line.text);
    let open = find_top_level_char(text, '(')
        .ok_or_else(|| parse_error(line.number, "node is missing input list"))?;
    let close = text
        .rfind(')')
        .ok_or_else(|| parse_error(line.number, "node input list is not closed"))?;
    if !text[close + 1..].trim().is_empty() {
        return Err(parse_error(line.number, "unexpected text after node"));
    }

    let prefix = text[..open].trim();
    let (lhs, invocation) = match find_top_level_char(prefix, '=') {
        Some(eq) => (&prefix[..eq], prefix[eq + 1..].trim()),
        None => ("", prefix),
    };
    let outputs = split_names(lhs);
    let (op, attrs) = parse_invocation(invocation, line.number)?;
    let inputs = split_top_level(&text[open + 1..close], ',')
        .into_iter()
        .map(|name| {
            let name = name.trim();
            if name.is_empty() {
                Ok(None)
            } else {
                Ok(Some(value_id(graph, values, name)))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let output_ids = outputs
        .iter()
        .map(|name| value_id(graph, values, name))
        .collect();

    let (domain, op_type) = match op.rsplit_once('.') {
        Some((domain, op_type)) => (domain.to_string(), op_type.to_string()),
        None => (String::new(), op.to_string()),
    };
    if op_type.is_empty() {
        return Err(parse_error(line.number, "node is missing an op type"));
    }
    let mut node = Node::new(NodeId(0), op_type, inputs, output_ids);
    node.domain = domain;
    node.attributes = attrs;
    Ok(graph.insert_node(node))
}

fn parse_invocation(text: &str, line: usize) -> Result<(&str, HashMap<String, Attribute>)> {
    let Some(open) = find_top_level_char(text, '<') else {
        return Ok((text.trim(), HashMap::new()));
    };
    let close = matching_close(text, open, '<', '>')
        .ok_or_else(|| parse_error(line, "attribute block is not closed"))?;
    if !text[close + 1..].trim().is_empty() {
        return Err(parse_error(line, "unexpected text after attributes"));
    }
    let mut attrs = HashMap::new();
    for pair in split_top_level(&text[open + 1..close], ',') {
        let eq = find_top_level_char(pair, '=')
            .ok_or_else(|| parse_error(line, "attribute is missing '='"))?;
        let name = pair[..eq].trim();
        if name.is_empty() {
            return Err(parse_error(line, "attribute is missing a name"));
        }
        attrs.insert(
            name.to_string(),
            parse_attribute(pair[eq + 1..].trim(), line)?,
        );
    }
    Ok((text[..open].trim(), attrs))
}

fn parse_attribute(text: &str, line: usize) -> Result<Attribute> {
    match text {
        "[]:ints" => return Ok(Attribute::Ints(Vec::new())),
        "[]:floats" => return Ok(Attribute::Floats(Vec::new())),
        "[]:strings" => return Ok(Attribute::Strings(Vec::new())),
        "[]:graphs" => return Ok(Attribute::Graphs(Vec::new())),
        _ => {}
    }
    if text.starts_with('"') {
        let value: String = serde_yaml::from_str(text)
            .map_err(|error| parse_error(line, format!("invalid string: {error}")))?;
        return Ok(Attribute::String(value.into_bytes()));
    }
    if text.starts_with('[') {
        if text.contains('"') {
            let values: Vec<String> = serde_yaml::from_str(text)
                .map_err(|error| parse_error(line, format!("invalid string list: {error}")))?;
            return Ok(Attribute::Strings(
                values.into_iter().map(String::into_bytes).collect(),
            ));
        }
        let body = text
            .strip_prefix('[')
            .and_then(|v| v.strip_suffix(']'))
            .ok_or_else(|| parse_error(line, "attribute list is not closed"))?;
        if body.trim().is_empty() {
            return Ok(Attribute::Ints(Vec::new()));
        }
        let parts = split_top_level(body, ',');
        if parts.iter().any(|part| looks_float(part.trim())) {
            return parts
                .into_iter()
                .map(|part| parse_float(part.trim(), line))
                .collect::<Result<Vec<_>>>()
                .map(Attribute::Floats);
        }
        return parts
            .into_iter()
            .map(|part| {
                part.trim()
                    .parse::<i64>()
                    .map_err(|_| parse_error(line, "invalid integer list"))
            })
            .collect::<Result<Vec<_>>>()
            .map(Attribute::Ints);
    }
    if let Some(count) = text
        .strip_prefix('<')
        .and_then(|v| v.strip_suffix(" strings>"))
    {
        let count = count
            .parse::<usize>()
            .map_err(|_| parse_error(line, "invalid string-list reference"))?;
        if count > 1_000_000 {
            return Err(parse_error(line, "string-list reference is too large"));
        }
        return Ok(Attribute::Strings(vec![Vec::new(); count]));
    }
    if let Some(count) = text
        .strip_prefix('<')
        .and_then(|v| v.strip_suffix(" bytes>"))
    {
        let count = count
            .parse::<usize>()
            .map_err(|_| parse_error(line, "invalid byte-string reference"))?;
        if count > 64 * 1024 * 1024 {
            return Err(parse_error(line, "byte-string reference exceeds 64 MiB"));
        }
        return Ok(Attribute::String(vec![0; count]));
    }
    if let Some(body) = text
        .strip_prefix("<tensor ")
        .and_then(|v| v.strip_suffix('>'))
    {
        let typed = parse_tensor_reference(body, line)?;
        return Ok(Attribute::Tensor(typed));
    }
    if text == "<sparse tensor>" {
        let empty = TensorData::from_raw(DataType::Float32, vec![0], Vec::new());
        return Ok(Attribute::SparseTensor(SparseTensorData {
            values: empty.clone(),
            indices: TensorData::from_raw(DataType::Int64, vec![0], Vec::new()),
            dims: Vec::new(),
        }));
    }
    if text == "<type>" {
        return Ok(Attribute::TypeProto(TypeProto::Tensor {
            dtype: DataType::Float32,
            shape: Vec::new(),
        }));
    }
    if looks_float(text) {
        return parse_float(text, line).map(Attribute::Float);
    }
    text.parse::<i64>()
        .map(Attribute::Int)
        .map_err(|_| parse_error(line, "unrecognized attribute value"))
}

fn parse_tensor_reference(text: &str, line: usize) -> Result<TensorData> {
    let open = text
        .find('[')
        .ok_or_else(|| parse_error(line, "tensor reference is missing shape"))?;
    let dtype = parse_dtype(text[..open].trim(), line)?;
    let mut dims_text = text[open + 1..]
        .strip_suffix(']')
        .ok_or_else(|| parse_error(line, "tensor reference shape is not closed"))?
        .trim();
    if dims_text.starts_with('[') && dims_text.ends_with(']') {
        dims_text = &dims_text[1..dims_text.len() - 1];
    }
    let dims = parse_static_dims(dims_text, line)?;
    placeholder_tensor(dtype, dims, line)
}

fn parse_initializer(
    text: &str,
    line: usize,
    graph: &mut Graph,
    values: &mut HashMap<String, ValueId>,
) -> Result<()> {
    let eq = text
        .find('=')
        .ok_or_else(|| parse_error(line, "initializer reference is missing '='"))?;
    let typed = parse_typed_value(text[..eq].trim(), line)?;
    let kind = text[eq + 1..].trim();
    let dims = typed
        .dims
        .iter()
        .map(|dim| {
            dim.parse::<usize>()
                .map_err(|_| parse_error(line, "initializer shape must be static"))
        })
        .collect::<Result<Vec<_>>>()?;
    let id = match values.get(&typed.name) {
        Some(id) => *id,
        None => {
            let id = create_typed_value(graph, &typed, line)?;
            values.insert(typed.name, id);
            id
        }
    };
    let weight = if kind == "<inline data omitted>" {
        WeightRef::Inline(placeholder_tensor(typed.dtype, dims, line)?)
    } else if kind == "<external data omitted>" {
        let numel = checked_numel(&dims, line)?;
        let length = typed
            .dtype
            .checked_storage_bytes(numel)
            .ok_or_else(|| parse_error(line, "initializer byte size overflows"))?;
        WeightRef::External {
            path: PathBuf::new(),
            offset: 0,
            length,
            dtype: typed.dtype,
            dims,
        }
    } else {
        return Err(parse_error(line, "unknown initializer reference kind"));
    };
    graph.set_initializer(id, weight);
    Ok(())
}

fn placeholder_tensor(dtype: DataType, dims: Vec<usize>, line: usize) -> Result<TensorData> {
    let numel = checked_numel(&dims, line)?;
    let bytes = dtype
        .checked_storage_bytes(numel)
        .ok_or_else(|| parse_error(line, "tensor byte size overflows"))?;
    const MAX_PLACEHOLDER_BYTES: usize = 64 * 1024 * 1024;
    if bytes > MAX_PLACEHOLDER_BYTES {
        return Err(parse_error(
            line,
            "tensor placeholder exceeds 64 MiB safety limit",
        ));
    }
    Ok(TensorData::from_raw(dtype, dims, vec![0; bytes]))
}

fn checked_numel(dims: &[usize], line: usize) -> Result<usize> {
    dims.iter().try_fold(1usize, |count, dim| {
        count
            .checked_mul(*dim)
            .ok_or_else(|| parse_error(line, "tensor element count overflows"))
    })
}

fn attach_subgraph(
    graph: &mut Graph,
    node_id: NodeId,
    name: String,
    index: Option<usize>,
    subgraph: Graph,
) {
    let key = match index {
        Some(index) => format!("{name}[{index}]"),
        None => name.clone(),
    };
    graph.subgraphs.insert((node_id, key), subgraph.clone());
    let attrs = &mut graph.node_mut(node_id).attributes;
    match index {
        None => {
            attrs.insert(name, Attribute::Graph(Box::new(subgraph)));
        }
        Some(index) => {
            let attr = attrs
                .entry(name)
                .or_insert_with(|| Attribute::Graphs(Vec::new()));
            if let Attribute::Graphs(graphs) = attr {
                graphs.resize_with(index + 1, Graph::new);
                graphs[index] = subgraph;
            }
        }
    }
}

fn parse_subgraph_marker(text: &str) -> Option<(String, Option<usize>)> {
    let lhs = text.strip_suffix("= graph")?.trim();
    if let Some(open) = lhs.rfind('[')
        && lhs.ends_with(']')
        && let Ok(index) = lhs[open + 1..lhs.len() - 1].parse()
    {
        if index <= 65_535 {
            return Some((lhs[..open].to_string(), Some(index)));
        }
        return None;
    }
    (!lhs.is_empty()).then(|| (lhs.to_string(), None))
}

fn value_id(graph: &mut Graph, values: &mut HashMap<String, ValueId>, name: &str) -> ValueId {
    *values
        .entry(name.to_string())
        .or_insert_with(|| graph.create_named_value(name, DataType::Float32, Vec::new()))
}

fn split_names(text: &str) -> Vec<&str> {
    if text.trim().is_empty() {
        Vec::new()
    } else {
        text.split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .collect()
    }
}

fn parse_opset_imports(text: &str, line: usize) -> Result<HashMap<String, u64>> {
    let body = text
        .strip_prefix('[')
        .and_then(|v| v.strip_suffix(']'))
        .ok_or_else(|| parse_error(line, "opset_import must be enclosed in brackets"))?;
    let mut imports = HashMap::new();
    if body.trim().is_empty() {
        return Ok(imports);
    }
    for entry in split_top_level(body, ',') {
        let colon = find_top_level_char(entry, ':')
            .ok_or_else(|| parse_error(line, "opset import is missing ':'"))?;
        let domain: String = serde_yaml::from_str(entry[..colon].trim())
            .map_err(|error| parse_error(line, format!("invalid opset domain: {error}")))?;
        let version = entry[colon + 1..]
            .trim()
            .parse()
            .map_err(|_| parse_error(line, "invalid opset version"))?;
        imports.insert(domain, version);
    }
    Ok(imports)
}

fn parse_static_dims(text: &str, line: usize) -> Result<Vec<usize>> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    split_top_level(text, ',')
        .into_iter()
        .map(|dim| {
            dim.trim()
                .parse()
                .map_err(|_| parse_error(line, "invalid static dimension"))
        })
        .collect()
}

fn parse_dtype(text: &str, line: usize) -> Result<DataType> {
    let dtype = match text {
        "undefined" => DataType::Undefined,
        "float" | "float32" => DataType::Float32,
        "uint8" => DataType::Uint8,
        "int8" => DataType::Int8,
        "uint16" => DataType::Uint16,
        "int16" => DataType::Int16,
        "int32" => DataType::Int32,
        "int64" => DataType::Int64,
        "string" => DataType::String,
        "bool" => DataType::Bool,
        "float16" => DataType::Float16,
        "float64" => DataType::Float64,
        "uint32" => DataType::Uint32,
        "uint64" => DataType::Uint64,
        "complex64" => DataType::Complex64,
        "complex128" => DataType::Complex128,
        "bfloat16" => DataType::BFloat16,
        "float8e4m3fn" => DataType::Float8E4M3FN,
        "float8e4m3fnuz" => DataType::Float8E4M3FNUZ,
        "float8e5m2" => DataType::Float8E5M2,
        "float8e5m2fnuz" => DataType::Float8E5M2FNUZ,
        "uint4" => DataType::Uint4,
        "int4" => DataType::Int4,
        "float4e2m1" => DataType::Float4E2M1,
        _ => return Err(parse_error(line, format!("unknown dtype '{text}'"))),
    };
    Ok(dtype)
}

fn unquote(text: &str) -> String {
    text.strip_prefix('"')
        .and_then(|text| text.strip_suffix('"'))
        .unwrap_or(text)
        .to_string()
}

fn parse_float(text: &str, line: usize) -> Result<f32> {
    match text {
        "inf" => Ok(f32::INFINITY),
        "-inf" => Ok(f32::NEG_INFINITY),
        "NaN" => Ok(f32::NAN),
        _ => text
            .parse()
            .map_err(|_| parse_error(line, "invalid float attribute")),
    }
}

fn looks_float(text: &str) -> bool {
    text.contains(['.', 'e', 'E']) || matches!(text, "inf" | "-inf" | "NaN")
}

fn trim_comma(text: &str) -> &str {
    text.trim().strip_suffix(',').unwrap_or(text.trim()).trim()
}

fn strip_trailing_comment(text: &str) -> &str {
    let mut quoted = false;
    let mut escaped = false;
    for (index, ch) in text.char_indices() {
        if escaped {
            escaped = false;
        } else if ch == '\\' && quoted {
            escaped = true;
        } else if ch == '"' {
            quoted = !quoted;
        } else if ch == '/' && !quoted && text[index..].starts_with("//") {
            return text[..index].trim_end();
        }
    }
    text
}

fn split_top_level(text: &str, separator: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut stack = Vec::new();
    let mut quoted = false;
    let mut escaped = false;
    for (index, ch) in text.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && quoted {
            escaped = true;
            continue;
        }
        if ch == '"' {
            quoted = !quoted;
            continue;
        }
        if quoted {
            continue;
        }
        match ch {
            '(' | '[' | '<' => stack.push(ch),
            ')' | ']' | '>' => {
                stack.pop();
            }
            _ if ch == separator && stack.is_empty() => {
                parts.push(&text[start..index]);
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts
}

fn find_top_level_char(text: &str, needle: char) -> Option<usize> {
    let mut stack = Vec::new();
    let mut quoted = false;
    let mut escaped = false;
    for (index, ch) in text.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && quoted {
            escaped = true;
            continue;
        }
        if ch == '"' {
            quoted = !quoted;
            continue;
        }
        if quoted {
            continue;
        }
        if ch == needle && stack.is_empty() {
            return Some(index);
        }
        match ch {
            '(' | '[' | '<' => stack.push(ch),
            ')' | ']' | '>' => {
                stack.pop();
            }
            _ => {}
        }
    }
    None
}

fn matching_close(text: &str, open: usize, left: char, right: char) -> Option<usize> {
    let mut depth = 0;
    let mut quoted = false;
    let mut escaped = false;
    for (offset, ch) in text[open..].char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && quoted {
            escaped = true;
            continue;
        }
        if ch == '"' {
            quoted = !quoted;
        } else if !quoted && ch == left {
            depth += 1;
        } else if !quoted && ch == right {
            depth -= 1;
            if depth == 0 {
                return Some(open + offset);
            }
        }
    }
    None
}

fn parse_error(line: usize, message: impl Into<String>) -> Error {
    Error::TextParse {
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::to_text;
    use onnx_runtime_ir::{Node, static_shape};

    fn node_by_op<'a>(graph: &'a Graph, op: &str) -> &'a Node {
        graph
            .nodes
            .values()
            .find(|node| node.op_type == op)
            .unwrap()
    }

    #[test]
    fn round_trips_simple_graph_and_domain_imports() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        graph.opset_imports.insert("com.acme".into(), 3);
        let x = graph.create_named_value("X", DataType::Float32, static_shape([2, 3]));
        let y = graph.create_named_value("Y", DataType::Float32, static_shape([2, 3]));
        graph.add_input(x);
        let mut node = Node::new(NodeId(0), "CustomAdd", vec![Some(x)], vec![y]);
        node.domain = "com.acme".into();
        graph.insert_node(node);
        graph.add_output(y);
        let metadata = ModelMetadata {
            graph_name: "compute".into(),
            ir_version: 9,
            ..ModelMetadata::default()
        };
        let model = Model::with_metadata(graph, metadata);

        let parsed = from_text(&to_text(&model)).unwrap();
        assert_eq!(parsed.metadata.ir_version, 9);
        assert_eq!(parsed.metadata.graph_name, "compute");
        assert_eq!(parsed.graph.opset_imports, model.graph.opset_imports);
        assert_eq!(parsed.graph.inputs.len(), 1);
        assert_eq!(parsed.graph.outputs.len(), 1);
        let node = node_by_op(&parsed.graph, "CustomAdd");
        assert_eq!(node.domain, "com.acme");
        assert_eq!(node.inputs.len(), 1);
        assert_eq!(node.outputs.len(), 1);
    }

    #[test]
    fn round_trips_lists_tensor_attribute_and_initializer_reference() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let x = graph.create_named_value("X", DataType::Float32, static_shape([2]));
        let w = graph.create_named_value("W", DataType::Float32, static_shape([2]));
        let y = graph.create_named_value("Y", DataType::Float32, static_shape([2]));
        graph.add_input(x);
        graph.set_initializer(
            w,
            WeightRef::Inline(TensorData::from_raw(DataType::Float32, vec![2], vec![0; 8])),
        );
        let mut node = Node::new(NodeId(0), "Decorated", vec![Some(x), Some(w)], vec![y]);
        node.attributes.insert("axis".into(), Attribute::Int(-1));
        node.attributes
            .insert("alpha".into(), Attribute::Float(0.25));
        node.attributes
            .insert("label".into(), Attribute::String(b"hello".to_vec()));
        node.attributes
            .insert("axes".into(), Attribute::Ints(vec![1, 2]));
        node.attributes
            .insert("scales".into(), Attribute::Floats(vec![0.5, 2.0]));
        node.attributes.insert(
            "labels".into(),
            Attribute::Strings(vec![b"a".to_vec(), b"b".to_vec()]),
        );
        node.attributes.insert(
            "value".into(),
            Attribute::Tensor(TensorData::from_raw(DataType::Int64, vec![2], vec![0; 16])),
        );
        graph.insert_node(node);
        graph.add_output(y);

        let parsed = from_text(&to_text(&Model::new(graph))).unwrap();
        let node = node_by_op(&parsed.graph, "Decorated");
        assert!(matches!(node.attributes["axis"], Attribute::Int(-1)));
        assert!(matches!(node.attributes["alpha"], Attribute::Float(0.25)));
        assert_eq!(node.attributes["label"].as_str(), Some("hello"));
        assert!(matches!(&node.attributes["axes"], Attribute::Ints(v) if v == &[1, 2]));
        assert!(matches!(&node.attributes["scales"], Attribute::Floats(v) if v == &[0.5, 2.0]));
        assert!(matches!(
            &node.attributes["labels"],
            Attribute::Strings(v) if v == &[b"a".to_vec(), b"b".to_vec()]
        ));
        assert!(
            matches!(&node.attributes["value"], Attribute::Tensor(t) if t.dtype == DataType::Int64 && t.dims == [2])
        );
        let initializer = parsed.graph.initializers.values().next().unwrap();
        assert_eq!(initializer.dtype(), DataType::Float32);
        assert_eq!(initializer.dims(), &[2]);
    }

    #[test]
    fn round_trips_empty_typed_attribute_lists() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let x = graph.create_named_value("X", DataType::Float32, static_shape([2]));
        let y = graph.create_named_value("Y", DataType::Float32, static_shape([2]));
        graph.add_input(x);
        let mut node = Node::new(NodeId(0), "EmptyLists", vec![Some(x)], vec![y]);
        node.attributes
            .insert("ints".into(), Attribute::Ints(Vec::new()));
        node.attributes
            .insert("floats".into(), Attribute::Floats(Vec::new()));
        node.attributes
            .insert("strings".into(), Attribute::Strings(Vec::new()));
        node.attributes
            .insert("graphs".into(), Attribute::Graphs(Vec::new()));
        graph.insert_node(node);
        graph.add_output(y);

        let text = to_text(&Model::new(graph));
        assert!(text.contains("ints = []:ints"), "{text}");
        assert!(text.contains("floats = []:floats"), "{text}");
        assert!(text.contains("strings = []:strings"), "{text}");
        assert!(text.contains("graphs = []:graphs"), "{text}");

        let parsed = from_text(&text).unwrap();
        let node = node_by_op(&parsed.graph, "EmptyLists");
        assert!(matches!(&node.attributes["ints"], Attribute::Ints(v) if v.is_empty()));
        assert!(matches!(&node.attributes["floats"], Attribute::Floats(v) if v.is_empty()));
        assert!(matches!(&node.attributes["strings"], Attribute::Strings(v) if v.is_empty()));
        assert!(matches!(&node.attributes["graphs"], Attribute::Graphs(v) if v.is_empty()));
    }

    fn branch(input: &str, output: &str, op: &str) -> Graph {
        let mut graph = Graph::new();
        let x = graph.create_named_value(input, DataType::Float32, static_shape([2]));
        let y = graph.create_named_value(output, DataType::Float32, static_shape([2]));
        graph.add_input(x);
        graph.insert_node(Node::new(NodeId(0), op, vec![Some(x)], vec![y]));
        graph.add_output(y);
        graph
    }

    #[test]
    fn round_trips_if_subgraphs() {
        let then_branch = branch("then_in", "then_out", "Relu");
        let else_branch = branch("else_in", "else_out", "Neg");
        let case_zero = branch("case0_in", "case0_out", "Identity");
        let case_one = branch("case1_in", "case1_out", "Abs");
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let cond = graph.create_named_value("cond", DataType::Bool, static_shape([]));
        let out = graph.create_named_value("out", DataType::Float32, static_shape([2]));
        graph.add_input(cond);
        let mut node = Node::new(NodeId(0), "If", vec![Some(cond)], vec![out]);
        node.attributes.insert(
            "then_branch".into(),
            Attribute::Graph(Box::new(then_branch.clone())),
        );
        node.attributes.insert(
            "else_branch".into(),
            Attribute::Graph(Box::new(else_branch.clone())),
        );
        node.attributes.insert(
            "cases".into(),
            Attribute::Graphs(vec![case_zero.clone(), case_one.clone()]),
        );
        let node_id = graph.insert_node(node);
        graph
            .subgraphs
            .insert((node_id, "then_branch".into()), then_branch);
        graph
            .subgraphs
            .insert((node_id, "else_branch".into()), else_branch);
        graph
            .subgraphs
            .insert((node_id, "cases[0]".into()), case_zero);
        graph
            .subgraphs
            .insert((node_id, "cases[1]".into()), case_one);
        graph.add_output(out);

        let parsed = from_text(&to_text(&Model::new(graph))).unwrap();
        let if_node = node_by_op(&parsed.graph, "If");
        assert!(matches!(
            &if_node.attributes["then_branch"],
            Attribute::Graph(graph) if node_by_op(graph, "Relu").op_type == "Relu"
        ));
        assert!(matches!(
            &if_node.attributes["else_branch"],
            Attribute::Graph(graph) if node_by_op(graph, "Neg").op_type == "Neg"
        ));
        assert!(matches!(
            &if_node.attributes["cases"],
            Attribute::Graphs(graphs)
                if node_by_op(&graphs[0], "Identity").op_type == "Identity"
                    && node_by_op(&graphs[1], "Abs").op_type == "Abs"
        ));
        assert_eq!(parsed.graph.subgraphs.len(), 4);
    }

    #[test]
    fn malformed_text_returns_error() {
        let malformed = "<\n  ir_version: nope,\n>\nmain () => () {\n";
        assert!(matches!(from_text(malformed), Err(Error::TextParse { .. })));
    }
}
