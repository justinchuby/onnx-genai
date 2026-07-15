//! ONNX protobuf TextFormat (`.onnxtxt` / `.pbtxt`) interchange.
//!
//! This is protobuf's field-oriented text representation, not the human ONNX
//! DSL in [`crate::text`]. Conversion goes through the same generated ONNX proto
//! and shared-IR loader path as binary protobuf and JSON.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use onnx_runtime_loader::{
    Model as EncoderModel, encode_model_proto, load_model_bytes_with_weights,
};
use prost::Message;
use serde_json::{Map, Number, Value};

use crate::{Error, Model, Result};

/// Serialize a model as canonical protobuf TextFormat.
pub fn to_textproto(model: &Model) -> Result<String> {
    let mut encoder = EncoderModel::new(&model.graph).with_metadata(model.metadata.clone());
    if let Some(weights) = model.weights() {
        encoder = encoder.with_weights(weights);
    }
    let proto = encode_model_proto(&encoder)?;
    let value = crate::json::model_to_value(&proto);
    let mut output = String::new();
    print_message(&value, MessageKind::Model, 0, &mut output)?;
    Ok(output)
}

/// Parse protobuf TextFormat into the shared IR model.
pub fn from_textproto(source: &str) -> Result<Model> {
    let tokens = Lexer::new(source).tokenize()?;
    let mut parser = Parser::new(tokens);
    let value = parser.parse_root(MessageKind::Model)?;
    let proto = crate::json::parse_model(&value)
        .map_err(|error| textproto_error(error.to_string().replace("ONNX JSON error: ", "")))?;
    let metadata = onnx_runtime_loader::ModelMetadata {
        ir_version: proto.ir_version,
        producer_name: proto.producer_name.clone(),
        producer_version: proto.producer_version.clone(),
        domain: proto.domain.clone(),
        model_version: proto.model_version,
        doc_string: (!proto.doc_string.is_empty()).then(|| proto.doc_string.clone()),
        graph_name: proto
            .graph
            .as_ref()
            .map(|graph| graph.name.clone())
            .unwrap_or_default(),
        metadata_props: proto
            .metadata_props
            .iter()
            .map(|entry| (entry.key.clone(), entry.value.clone()))
            .collect(),
    };
    let bytes = proto.encode_to_vec();
    let (graph, weights) = load_model_bytes_with_weights(&bytes, ".")?;
    let mut model = Model::with_metadata(graph, metadata);
    model.set_weights(weights);
    Ok(model)
}

fn textproto_error(message: impl Into<String>) -> Error {
    Error::TextProto(message.into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MessageKind {
    Model,
    Opset,
    Entry,
    Graph,
    Node,
    Attribute,
    Tensor,
    ValueInfo,
    Type,
    TensorType,
    SequenceType,
    OptionalType,
    MapType,
    Shape,
    Dimension,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScalarKind {
    String,
    Bytes,
    Int,
    Uint,
    Float,
    Enum,
}

#[derive(Clone, Copy, Debug)]
enum FieldKind {
    Scalar(ScalarKind),
    Message(MessageKind),
    UnsupportedScalar(ScalarKind),
    UnsupportedMessage,
}

#[derive(Clone, Copy, Debug)]
struct Field {
    text_name: &'static str,
    json_name: &'static str,
    kind: FieldKind,
    repeated: bool,
}

macro_rules! scalar {
    ($text:literal, $json:literal, $kind:ident) => {
        Field {
            text_name: $text,
            json_name: $json,
            kind: FieldKind::Scalar(ScalarKind::$kind),
            repeated: false,
        }
    };
    ($text:literal, $json:literal, $kind:ident, repeated) => {
        Field {
            text_name: $text,
            json_name: $json,
            kind: FieldKind::Scalar(ScalarKind::$kind),
            repeated: true,
        }
    };
}

macro_rules! message {
    ($text:literal, $json:literal, $kind:ident) => {
        Field {
            text_name: $text,
            json_name: $json,
            kind: FieldKind::Message(MessageKind::$kind),
            repeated: false,
        }
    };
    ($text:literal, $json:literal, $kind:ident, repeated) => {
        Field {
            text_name: $text,
            json_name: $json,
            kind: FieldKind::Message(MessageKind::$kind),
            repeated: true,
        }
    };
}

const MODEL_FIELDS: &[Field] = &[
    scalar!("ir_version", "irVersion", Int),
    message!("opset_import", "opsetImport", Opset, repeated),
    scalar!("producer_name", "producerName", String),
    scalar!("producer_version", "producerVersion", String),
    scalar!("domain", "domain", String),
    scalar!("model_version", "modelVersion", Int),
    scalar!("doc_string", "docString", String),
    message!("graph", "graph", Graph),
    message!("metadata_props", "metadataProps", Entry, repeated),
    Field {
        text_name: "training_info",
        json_name: "trainingInfo",
        kind: FieldKind::UnsupportedMessage,
        repeated: true,
    },
    Field {
        text_name: "functions",
        json_name: "functions",
        kind: FieldKind::UnsupportedMessage,
        repeated: true,
    },
];
const OPSET_FIELDS: &[Field] = &[
    scalar!("domain", "domain", String),
    scalar!("version", "version", Int),
];
const ENTRY_FIELDS: &[Field] = &[
    scalar!("key", "key", String),
    scalar!("value", "value", String),
];
const GRAPH_FIELDS: &[Field] = &[
    message!("node", "node", Node, repeated),
    scalar!("name", "name", String),
    message!("initializer", "initializer", Tensor, repeated),
    Field {
        text_name: "sparse_initializer",
        json_name: "sparseInitializer",
        kind: FieldKind::UnsupportedMessage,
        repeated: true,
    },
    scalar!("doc_string", "docString", String),
    message!("input", "input", ValueInfo, repeated),
    message!("output", "output", ValueInfo, repeated),
    message!("value_info", "valueInfo", ValueInfo, repeated),
    Field {
        text_name: "quantization_annotation",
        json_name: "quantizationAnnotation",
        kind: FieldKind::UnsupportedMessage,
        repeated: true,
    },
    message!("metadata_props", "metadataProps", Entry, repeated),
];
const NODE_FIELDS: &[Field] = &[
    scalar!("input", "input", String, repeated),
    scalar!("output", "output", String, repeated),
    scalar!("name", "name", String),
    scalar!("op_type", "opType", String),
    scalar!("domain", "domain", String),
    message!("attribute", "attribute", Attribute, repeated),
    scalar!("doc_string", "docString", String),
    Field {
        text_name: "overload",
        json_name: "overload",
        kind: FieldKind::UnsupportedScalar(ScalarKind::String),
        repeated: false,
    },
    Field {
        text_name: "metadata_props",
        json_name: "metadataProps",
        kind: FieldKind::UnsupportedMessage,
        repeated: true,
    },
];
const ATTRIBUTE_FIELDS: &[Field] = &[
    scalar!("name", "name", String),
    Field {
        text_name: "ref_attr_name",
        json_name: "refAttrName",
        kind: FieldKind::UnsupportedScalar(ScalarKind::String),
        repeated: false,
    },
    Field {
        text_name: "doc_string",
        json_name: "docString",
        kind: FieldKind::UnsupportedScalar(ScalarKind::String),
        repeated: false,
    },
    scalar!("type", "type", Enum),
    scalar!("f", "f", Float),
    scalar!("i", "i", Int),
    scalar!("s", "s", Bytes),
    message!("t", "t", Tensor),
    message!("g", "g", Graph),
    Field {
        text_name: "sparse_tensor",
        json_name: "sparseTensor",
        kind: FieldKind::UnsupportedMessage,
        repeated: false,
    },
    message!("tp", "tp", Type),
    scalar!("floats", "floats", Float, repeated),
    scalar!("ints", "ints", Int, repeated),
    scalar!("strings", "strings", Bytes, repeated),
    Field {
        text_name: "tensors",
        json_name: "tensors",
        kind: FieldKind::UnsupportedMessage,
        repeated: true,
    },
    message!("graphs", "graphs", Graph, repeated),
    Field {
        text_name: "sparse_tensors",
        json_name: "sparseTensors",
        kind: FieldKind::UnsupportedMessage,
        repeated: true,
    },
    Field {
        text_name: "type_protos",
        json_name: "typeProtos",
        kind: FieldKind::UnsupportedMessage,
        repeated: true,
    },
];
const TENSOR_FIELDS: &[Field] = &[
    scalar!("dims", "dims", Int, repeated),
    scalar!("data_type", "dataType", Enum),
    Field {
        text_name: "segment",
        json_name: "segment",
        kind: FieldKind::UnsupportedMessage,
        repeated: false,
    },
    scalar!("float_data", "floatData", Float, repeated),
    scalar!("int32_data", "int32Data", Int, repeated),
    scalar!("string_data", "stringData", Bytes, repeated),
    scalar!("int64_data", "int64Data", Int, repeated),
    scalar!("name", "name", String),
    scalar!("doc_string", "docString", String),
    scalar!("raw_data", "rawData", Bytes),
    scalar!("double_data", "doubleData", Float, repeated),
    scalar!("uint64_data", "uint64Data", Uint, repeated),
    message!("external_data", "externalData", Entry, repeated),
    scalar!("data_location", "dataLocation", Enum),
    Field {
        text_name: "metadata_props",
        json_name: "metadataProps",
        kind: FieldKind::UnsupportedMessage,
        repeated: true,
    },
];
const VALUE_INFO_FIELDS: &[Field] = &[
    scalar!("name", "name", String),
    message!("type", "type", Type),
    Field {
        text_name: "doc_string",
        json_name: "docString",
        kind: FieldKind::UnsupportedScalar(ScalarKind::String),
        repeated: false,
    },
    Field {
        text_name: "metadata_props",
        json_name: "metadataProps",
        kind: FieldKind::UnsupportedMessage,
        repeated: true,
    },
];
const TYPE_FIELDS: &[Field] = &[
    message!("tensor_type", "tensorType", TensorType),
    message!("sequence_type", "sequenceType", SequenceType),
    message!("map_type", "mapType", MapType),
    message!("optional_type", "optionalType", OptionalType),
    message!("sparse_tensor_type", "sparseTensorType", TensorType),
    Field {
        text_name: "denotation",
        json_name: "denotation",
        kind: FieldKind::UnsupportedScalar(ScalarKind::String),
        repeated: false,
    },
];
const TENSOR_TYPE_FIELDS: &[Field] = &[
    scalar!("elem_type", "elemType", Enum),
    message!("shape", "shape", Shape),
];
const SEQUENCE_TYPE_FIELDS: &[Field] = &[message!("elem_type", "elemType", Type)];
const OPTIONAL_TYPE_FIELDS: &[Field] = &[message!("elem_type", "elemType", Type)];
const MAP_TYPE_FIELDS: &[Field] = &[
    scalar!("key_type", "keyType", Enum),
    message!("value_type", "valueType", Type),
];
const SHAPE_FIELDS: &[Field] = &[message!("dim", "dim", Dimension, repeated)];
const DIMENSION_FIELDS: &[Field] = &[
    scalar!("dim_value", "dimValue", Int),
    scalar!("dim_param", "dimParam", String),
    Field {
        text_name: "denotation",
        json_name: "denotation",
        kind: FieldKind::UnsupportedScalar(ScalarKind::String),
        repeated: false,
    },
];

fn fields(kind: MessageKind) -> &'static [Field] {
    match kind {
        MessageKind::Model => MODEL_FIELDS,
        MessageKind::Opset => OPSET_FIELDS,
        MessageKind::Entry => ENTRY_FIELDS,
        MessageKind::Graph => GRAPH_FIELDS,
        MessageKind::Node => NODE_FIELDS,
        MessageKind::Attribute => ATTRIBUTE_FIELDS,
        MessageKind::Tensor => TENSOR_FIELDS,
        MessageKind::ValueInfo => VALUE_INFO_FIELDS,
        MessageKind::Type => TYPE_FIELDS,
        MessageKind::TensorType => TENSOR_TYPE_FIELDS,
        MessageKind::SequenceType => SEQUENCE_TYPE_FIELDS,
        MessageKind::OptionalType => OPTIONAL_TYPE_FIELDS,
        MessageKind::MapType => MAP_TYPE_FIELDS,
        MessageKind::Shape => SHAPE_FIELDS,
        MessageKind::Dimension => DIMENSION_FIELDS,
    }
}

fn lookup_field(kind: MessageKind, name: &str) -> Result<&'static Field> {
    fields(kind)
        .iter()
        .find(|field| field.text_name == name || field.json_name == name)
        .ok_or_else(|| textproto_error(format!("unknown field {name:?} in {kind:?}")))
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Ident(String),
    Number(String),
    String(Vec<u8>),
    Colon,
    LBrace,
    RBrace,
    LAngle,
    RAngle,
    LBracket,
    RBracket,
    Comma,
    Semicolon,
}

struct Lexer<'a> {
    source: &'a [u8],
    offset: usize,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source: source.as_bytes(),
            offset: 0,
        }
    }

    fn tokenize(mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        while self.skip_trivia() {
            let byte = self.source[self.offset];
            let token = match byte {
                b':' => {
                    self.offset += 1;
                    Token::Colon
                }
                b'{' => {
                    self.offset += 1;
                    Token::LBrace
                }
                b'}' => {
                    self.offset += 1;
                    Token::RBrace
                }
                b'<' => {
                    self.offset += 1;
                    Token::LAngle
                }
                b'>' => {
                    self.offset += 1;
                    Token::RAngle
                }
                b'[' => {
                    self.offset += 1;
                    Token::LBracket
                }
                b']' => {
                    self.offset += 1;
                    Token::RBracket
                }
                b',' => {
                    self.offset += 1;
                    Token::Comma
                }
                b';' => {
                    self.offset += 1;
                    Token::Semicolon
                }
                b'"' | b'\'' => Token::String(self.string(byte)?),
                b'+' | b'-' | b'.' | b'0'..=b'9' => Token::Number(self.atom()),
                _ if is_ident_start(byte) => Token::Ident(self.atom()),
                _ => {
                    return Err(textproto_error(format!(
                        "unexpected byte 0x{byte:02x} at offset {}",
                        self.offset
                    )));
                }
            };
            tokens.push(token);
        }
        Ok(tokens)
    }

    fn skip_trivia(&mut self) -> bool {
        loop {
            while self.offset < self.source.len() && self.source[self.offset].is_ascii_whitespace()
            {
                self.offset += 1;
            }
            if self.offset >= self.source.len() {
                return false;
            }
            if self.source[self.offset] == b'#' {
                while self.offset < self.source.len() && self.source[self.offset] != b'\n' {
                    self.offset += 1;
                }
                continue;
            }
            return true;
        }
    }

    fn atom(&mut self) -> String {
        let start = self.offset;
        while self.offset < self.source.len()
            && !self.source[self.offset].is_ascii_whitespace()
            && !b":{}<>[],;\"'".contains(&self.source[self.offset])
        {
            self.offset += 1;
        }
        String::from_utf8_lossy(&self.source[start..self.offset]).into_owned()
    }

    fn string(&mut self, quote: u8) -> Result<Vec<u8>> {
        self.offset += 1;
        let mut output = Vec::new();
        while self.offset < self.source.len() {
            let byte = self.source[self.offset];
            self.offset += 1;
            if byte == quote {
                return Ok(output);
            }
            if byte != b'\\' {
                output.push(byte);
                continue;
            }
            let escaped = *self
                .source
                .get(self.offset)
                .ok_or_else(|| textproto_error("unterminated escape"))?;
            self.offset += 1;
            match escaped {
                b'a' => output.push(0x07),
                b'b' => output.push(0x08),
                b'f' => output.push(0x0c),
                b'n' => output.push(b'\n'),
                b'r' => output.push(b'\r'),
                b't' => output.push(b'\t'),
                b'v' => output.push(0x0b),
                b'\\' | b'\'' | b'"' => output.push(escaped),
                b'x' | b'X' => output.push(self.radix_escape(16, 2)?),
                b'0'..=b'7' => {
                    let mut value = escaped - b'0';
                    for _ in 0..2 {
                        match self.source.get(self.offset).copied() {
                            Some(next @ b'0'..=b'7') => {
                                value = value.wrapping_mul(8).wrapping_add(next - b'0');
                                self.offset += 1;
                            }
                            _ => break,
                        }
                    }
                    output.push(value);
                }
                _ => {
                    return Err(textproto_error(format!(
                        "unknown escape \\{}",
                        escaped as char
                    )));
                }
            }
        }
        Err(textproto_error("unterminated string"))
    }

    fn radix_escape(&mut self, radix: u32, count: usize) -> Result<u8> {
        let end = self.offset + count;
        let digits = self
            .source
            .get(self.offset..end)
            .ok_or_else(|| textproto_error("short hex escape"))?;
        let text =
            std::str::from_utf8(digits).map_err(|_| textproto_error("invalid hex escape"))?;
        self.offset = end;
        u8::from_str_radix(text, radix).map_err(|_| textproto_error("invalid hex escape"))
    }
}

fn is_ident_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

struct Parser {
    tokens: Vec<Token>,
    offset: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, offset: 0 }
    }

    fn parse_root(&mut self, kind: MessageKind) -> Result<Value> {
        let value = self.parse_message(kind, None)?;
        if self.offset != self.tokens.len() {
            return Err(textproto_error("trailing tokens"));
        }
        Ok(value)
    }

    fn parse_message(&mut self, kind: MessageKind, end: Option<Token>) -> Result<Value> {
        let mut map = Map::new();
        loop {
            if let Some(expected) = &end {
                if self.peek() == Some(expected) {
                    self.offset += 1;
                    break;
                }
            } else if self.peek().is_none() {
                break;
            }
            let name = match self.next() {
                Some(Token::Ident(name)) => name,
                other => {
                    return Err(textproto_error(format!(
                        "expected field name, found {other:?}"
                    )));
                }
            };
            let field = lookup_field(kind, &name)?;
            let value = match field.kind {
                FieldKind::Message(child) => {
                    self.eat(&Token::Colon);
                    let close = match self.next() {
                        Some(Token::LBrace) => Token::RBrace,
                        Some(Token::LAngle) => Token::RAngle,
                        other => {
                            return Err(textproto_error(format!(
                                "expected message body for {name}, found {other:?}"
                            )));
                        }
                    };
                    self.parse_message(child, Some(close))?
                }
                FieldKind::UnsupportedMessage => {
                    self.eat(&Token::Colon);
                    self.skip_message(&name)?;
                    Value::Object(Map::new())
                }
                FieldKind::Scalar(scalar) | FieldKind::UnsupportedScalar(scalar) => {
                    self.expect(Token::Colon)?;
                    self.parse_scalar(scalar, &name)?
                }
            };
            if matches!(
                field.kind,
                FieldKind::UnsupportedMessage | FieldKind::UnsupportedScalar(_)
            ) {
                return Err(textproto_error(format!(
                    "unsupported populated ONNX TextProto field: {}",
                    field.text_name
                )));
            }
            insert_field(&mut map, field, value)?;
            if matches!(self.peek(), Some(Token::Comma | Token::Semicolon)) {
                self.offset += 1;
            }
        }
        Ok(Value::Object(map))
    }

    fn parse_scalar(&mut self, kind: ScalarKind, name: &str) -> Result<Value> {
        if self.eat(&Token::LBracket) {
            return Err(textproto_error(format!(
                "list syntax for {name} is not supported; repeat the field"
            )));
        }
        let token = self
            .next()
            .ok_or_else(|| textproto_error(format!("missing value for {name}")))?;
        match (kind, token) {
            (ScalarKind::String, Token::String(bytes)) => String::from_utf8(bytes)
                .map(Value::String)
                .map_err(|_| textproto_error(format!("{name} is not valid UTF-8"))),
            (ScalarKind::Bytes, Token::String(bytes)) => Ok(Value::String(BASE64.encode(bytes))),
            (ScalarKind::Enum, Token::Ident(value)) => Ok(Value::String(value)),
            (ScalarKind::Enum, Token::Number(value)) => parse_integer(&value, name),
            (ScalarKind::Int, Token::Number(value)) => parse_integer(&value, name),
            (ScalarKind::Uint, Token::Number(value)) => parse_unsigned(&value, name),
            (ScalarKind::Float, Token::Number(value)) => parse_float(&value, name),
            (ScalarKind::Float, Token::Ident(value)) => parse_float(&value, name),
            (_, token) => Err(textproto_error(format!(
                "invalid value for {name}: {token:?}"
            ))),
        }
    }

    fn skip_message(&mut self, name: &str) -> Result<()> {
        let (open, close) = match self.next() {
            Some(Token::LBrace) => (Token::LBrace, Token::RBrace),
            Some(Token::LAngle) => (Token::LAngle, Token::RAngle),
            other => {
                return Err(textproto_error(format!(
                    "expected message body for {name}, found {other:?}"
                )));
            }
        };
        let mut depth = 1usize;
        while let Some(token) = self.next() {
            if token == open {
                depth += 1;
            } else if token == close {
                depth -= 1;
                if depth == 0 {
                    return Ok(());
                }
            }
        }
        Err(textproto_error(format!("unterminated message {name}")))
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.offset)
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.offset).cloned();
        self.offset += usize::from(token.is_some());
        token
    }

    fn eat(&mut self, token: &Token) -> bool {
        if self.peek() == Some(token) {
            self.offset += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, token: Token) -> Result<()> {
        if self.eat(&token) {
            Ok(())
        } else {
            Err(textproto_error(format!(
                "expected {token:?}, found {:?}",
                self.peek()
            )))
        }
    }
}

fn insert_field(map: &mut Map<String, Value>, field: &Field, value: Value) -> Result<()> {
    if field.repeated {
        match map
            .entry(field.json_name.to_string())
            .or_insert_with(|| Value::Array(Vec::new()))
        {
            Value::Array(values) => values.push(value),
            _ => unreachable!(),
        }
    } else if map.insert(field.json_name.to_string(), value).is_some() {
        return Err(textproto_error(format!(
            "field {:?} specified more than once",
            field.text_name
        )));
    }
    Ok(())
}

fn parse_integer(text: &str, name: &str) -> Result<Value> {
    let value = parse_i64_literal(text)
        .ok_or_else(|| textproto_error(format!("{name} must be an integer")))?;
    Ok(Value::String(value.to_string()))
}

fn parse_unsigned(text: &str, name: &str) -> Result<Value> {
    let value = text
        .parse::<u64>()
        .map_err(|_| textproto_error(format!("{name} must be an unsigned integer")))?;
    Ok(Value::String(value.to_string()))
}

fn parse_i64_literal(text: &str) -> Option<i64> {
    let (negative, value) = text.strip_prefix('-').map_or((false, text), |v| (true, v));
    let magnitude = if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).ok()?
    } else if value.len() > 1 && value.starts_with('0') {
        u64::from_str_radix(&value[1..], 8).ok()?
    } else {
        value.parse().ok()?
    };
    if negative {
        (magnitude <= (i64::MAX as u64) + 1).then(|| magnitude.wrapping_neg() as i64)
    } else {
        i64::try_from(magnitude).ok()
    }
}

fn parse_float(text: &str, name: &str) -> Result<Value> {
    let lower = text.to_ascii_lowercase();
    match lower.as_str() {
        "nan" | "+nan" | "-nan" => Ok(Value::String("NaN".into())),
        "inf" | "+inf" | "infinity" | "+infinity" => Ok(Value::String("Infinity".into())),
        "-inf" | "-infinity" => Ok(Value::String("-Infinity".into())),
        _ => {
            let number = text
                .parse::<f64>()
                .map_err(|_| textproto_error(format!("{name} must be a float")))?;
            Number::from_f64(number)
                .map(Value::Number)
                .ok_or_else(|| textproto_error(format!("invalid float for {name}")))
        }
    }
}

fn print_message(
    value: &Value,
    kind: MessageKind,
    indent: usize,
    output: &mut String,
) -> Result<()> {
    let map = value
        .as_object()
        .ok_or_else(|| textproto_error(format!("{kind:?} must be an object")))?;
    for field in fields(kind) {
        let Some(value) = map.get(field.json_name) else {
            continue;
        };
        if field.repeated {
            let values = value
                .as_array()
                .ok_or_else(|| textproto_error(format!("{} must be repeated", field.json_name)))?;
            for value in values {
                print_field(field, value, indent, output)?;
            }
        } else {
            print_field(field, value, indent, output)?;
        }
    }
    for key in map.keys() {
        if !fields(kind).iter().any(|field| field.json_name == key) {
            return Err(textproto_error(format!(
                "unsupported field {key:?} in {kind:?}"
            )));
        }
    }
    Ok(())
}

fn print_field(field: &Field, value: &Value, indent: usize, output: &mut String) -> Result<()> {
    output.push_str(&"  ".repeat(indent));
    output.push_str(field.text_name);
    match field.kind {
        FieldKind::Message(kind) => {
            output.push_str(" {\n");
            print_message(value, kind, indent + 1, output)?;
            output.push_str(&"  ".repeat(indent));
            output.push_str("}\n");
        }
        FieldKind::Scalar(kind) => {
            output.push_str(": ");
            print_scalar(value, kind, output)?;
            output.push('\n');
        }
        FieldKind::UnsupportedMessage | FieldKind::UnsupportedScalar(_) => {
            return Err(textproto_error(format!(
                "unsupported populated field {}",
                field.text_name
            )));
        }
    }
    Ok(())
}

fn print_scalar(value: &Value, kind: ScalarKind, output: &mut String) -> Result<()> {
    match kind {
        ScalarKind::String => {
            let value = value
                .as_str()
                .ok_or_else(|| textproto_error("expected string"))?;
            write_escaped(value.as_bytes(), output);
        }
        ScalarKind::Bytes => {
            let value = value
                .as_str()
                .ok_or_else(|| textproto_error("expected base64 bytes"))?;
            let bytes = BASE64
                .decode(value)
                .map_err(|error| textproto_error(format!("invalid bytes: {error}")))?;
            write_escaped(&bytes, output);
        }
        ScalarKind::Enum => match value {
            Value::String(value) => output.push_str(value),
            Value::Number(value) => output.push_str(&value.to_string()),
            _ => return Err(textproto_error("expected enum")),
        },
        ScalarKind::Int | ScalarKind::Uint => match value {
            Value::String(value) => output.push_str(value),
            Value::Number(value) => output.push_str(&value.to_string()),
            _ => return Err(textproto_error("expected integer")),
        },
        ScalarKind::Float => match value {
            Value::String(value) if value == "NaN" => output.push_str("nan"),
            Value::String(value) if value == "Infinity" => output.push_str("inf"),
            Value::String(value) if value == "-Infinity" => output.push_str("-inf"),
            Value::Number(value) => output.push_str(&value.to_string()),
            _ => return Err(textproto_error("expected float")),
        },
    }
    Ok(())
}

fn write_escaped(bytes: &[u8], output: &mut String) {
    output.push('"');
    for &byte in bytes {
        match byte {
            b'\n' => output.push_str("\\n"),
            b'\r' => output.push_str("\\r"),
            b'\t' => output.push_str("\\t"),
            b'\\' => output.push_str("\\\\"),
            b'"' => output.push_str("\\\""),
            0x20..=0x7e => output.push(byte as char),
            _ => output.push_str(&format!("\\{:03o}", byte)),
        }
    }
    output.push('"');
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use onnx_runtime_ir::{
        Attribute, DataType, Graph, Node, NodeId, TensorData, WeightRef, static_shape,
    };
    use onnx_runtime_loader::ModelMetadata;

    use super::*;

    fn branch_graph(name: &str, value: f32) -> Graph {
        let mut graph = Graph::new();
        let output = graph.create_named_value(name, DataType::Float32, static_shape([1]));
        graph.set_initializer(
            output,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![1],
                value.to_le_bytes().to_vec(),
            )),
        );
        graph.add_output(output);
        graph
    }

    #[test]
    fn model_round_trip_is_structurally_and_textually_stable() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let condition = graph.create_named_value("cond", DataType::Bool, static_shape([]));
        let weight = graph.create_named_value("raw_weight", DataType::Float32, static_shape([2]));
        let output = graph.create_named_value("result", DataType::Float32, static_shape([1]));
        graph.add_input(condition);
        graph.add_output(output);
        graph.set_initializer(
            weight,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![2],
                [1.25f32.to_le_bytes(), (-2.5f32).to_le_bytes()].concat(),
            )),
        );
        let mut node = Node::new(NodeId(0), "If", vec![Some(condition)], vec![output]);
        node.name = "choose".into();
        node.attributes = HashMap::from([
            (
                "then_branch".into(),
                Attribute::Graph(Box::new(branch_graph("then_value", 1.0))),
            ),
            (
                "else_branch".into(),
                Attribute::Graph(Box::new(branch_graph("else_value", -1.0))),
            ),
            (
                "labels".into(),
                Attribute::Strings(vec![b"a".to_vec(), vec![0, 255, b'"']]),
            ),
        ]);
        graph.insert_node(node);
        let mut metadata = ModelMetadata::default();
        metadata.producer_name = "onnx-rs".into();
        metadata.graph_name = "textproto_roundtrip".into();
        metadata.doc_string = Some("model documentation".into());
        metadata.metadata_props = vec![("purpose".into(), "interop".into())];
        let model = Model::with_metadata(graph, metadata.clone());

        let text = to_textproto(&model).expect("serialize");
        assert!(text.contains("raw_data: \"\\000\\000\\240?\\000\\000 \\300\""));
        assert!(text.contains("type: GRAPH"));
        let decoded = from_textproto(&text).expect("parse");
        assert_eq!(decoded.metadata, metadata);
        assert_eq!(decoded.graph.initializers.len(), 1);
        assert_eq!(decoded.graph.subgraphs.len(), 2);
        assert_eq!(to_textproto(&decoded).expect("re-serialize"), text);
    }

    #[test]
    fn handwritten_pbtxt_parses() {
        let source = r#"
            # Compatible with google.protobuf.text_format.
            ir_version: 10
            opset_import { version: 21 }
            producer_name: "golden"
            graph {
              name: "g"
              initializer {
                dims: 2
                data_type: FLOAT
                name: "W"
                raw_data: "\000\000\200?\000\000\000@"
              }
              output {
                name: "W"
                type { tensor_type { elem_type: FLOAT shape { dim { dim_value: 2 } } } }
              }
            }
        "#;
        let model = from_textproto(source).expect("golden pbtxt");
        assert_eq!(model.metadata.producer_name, "golden");
        assert_eq!(model.metadata.graph_name, "g");
        assert_eq!(model.graph.initializers.len(), 1);
        let printed = to_textproto(&model).expect("print golden");
        assert!(printed.contains("data_type: FLOAT"));
        assert!(printed.contains("raw_data: \"\\000\\000\\200?\\000\\000\\000@\""));
    }

    #[test]
    fn populated_unrepresentable_field_is_rejected() {
        let error = match from_textproto(
            r#"ir_version: 10
               opset_import { version: 21 }
               graph { doc_string: "not retained" }"#,
        ) {
            Ok(_) => panic!("graph doc_string must not be silently dropped"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("unsupported ONNX JSON field: docString")
        );
    }
}
