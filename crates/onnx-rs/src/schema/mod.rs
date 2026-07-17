//! Declarative operator schemas and an opset-aware registry (ONNX_RS §7).
//!
//! Schemas are authored as YAML and loaded into owned Rust values. The built-in
//! registry embeds high-value standard operators and can expand the YAML
//! catalogue without changing the registry API.

use std::collections::HashMap;

use onnx_runtime_ir::DataType;
use serde::{Deserialize, Serialize};

/// A complete operator definition for one opset interval.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpSchema {
    /// Operator domain (`""` and `"ai.onnx"` both mean the standard domain).
    #[serde(default)]
    pub domain: String,
    /// Operator type name.
    pub name: String,
    /// First opset version for which this schema is valid.
    pub since_version: u64,
    /// Last valid opset version, inclusive. `None` means no upper bound.
    #[serde(default)]
    pub until_version: Option<u64>,
    /// Human-readable operator documentation.
    #[serde(default)]
    pub doc: String,
    /// Positional input definitions.
    #[serde(default)]
    pub inputs: Vec<InputSpec>,
    /// Positional output definitions.
    #[serde(default)]
    pub outputs: Vec<OutputSpec>,
    /// Attribute definitions.
    #[serde(default)]
    pub attributes: Vec<AttributeSpec>,
    /// Type variables and the element types they admit.
    #[serde(default)]
    pub type_constraints: Vec<TypeConstraint>,
}

impl OpSchema {
    /// Whether this schema applies to `opset`.
    pub fn supports_opset(&self, opset: u64) -> bool {
        self.since_version <= opset && self.until_version.is_none_or(|until| opset <= until)
    }
}

/// One positional operator input.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputSpec {
    /// Schema-visible input name.
    pub name: String,
    /// Type variable or concrete type expression.
    pub type_str: String,
    /// Human-readable input documentation.
    #[serde(default)]
    pub doc: String,
    /// Whether this position may be omitted.
    #[serde(default)]
    pub optional: bool,
    /// Whether this position accepts a variable number of trailing values.
    #[serde(default)]
    pub variadic: bool,
    /// Minimum number of actual values consumed by a variadic position.
    #[serde(default = "default_min_arity")]
    pub min_arity: usize,
}

/// One positional operator output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputSpec {
    /// Schema-visible output name.
    pub name: String,
    /// Type variable or concrete type expression.
    pub type_str: String,
    /// Human-readable output documentation.
    #[serde(default)]
    pub doc: String,
    /// Whether this output may be omitted.
    #[serde(default)]
    pub optional: bool,
    /// Whether this position accepts a variable number of trailing values.
    #[serde(default)]
    pub variadic: bool,
    /// Minimum number of actual values produced by a variadic position.
    #[serde(default = "default_min_arity")]
    pub min_arity: usize,
}

const fn default_min_arity() -> usize {
    1
}

/// ONNX attribute kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributeType {
    /// Scalar 64-bit integer.
    Int,
    /// Scalar 32-bit float.
    Float,
    /// Raw byte string.
    String,
    /// Tensor value.
    Tensor,
    /// Graph value.
    Graph,
    /// Sparse tensor value.
    SparseTensor,
    /// Type-proto value.
    TypeProto,
    /// Integer list.
    Ints,
    /// Float list.
    Floats,
    /// String list.
    Strings,
    /// Graph list.
    Graphs,
    /// Tensor list.
    Tensors,
    /// Sparse tensor list.
    SparseTensors,
    /// Type-proto list.
    TypeProtos,
}

/// A typed YAML-compatible attribute default.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttributeDefault {
    /// Integer scalar.
    Int(i64),
    /// Floating-point scalar.
    Float(f64),
    /// String scalar.
    String(String),
    /// Integer list.
    Ints(Vec<i64>),
    /// Floating-point list.
    Floats(Vec<f64>),
    /// String list.
    Strings(Vec<String>),
}

/// One operator attribute definition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttributeSpec {
    /// Attribute name.
    pub name: String,
    /// Required ONNX attribute kind.
    #[serde(rename = "type")]
    pub attr_type: AttributeType,
    /// Whether callers must provide the attribute.
    #[serde(default)]
    pub required: bool,
    /// Default used when the attribute is omitted.
    #[serde(default)]
    pub default: Option<AttributeDefault>,
    /// Human-readable attribute documentation.
    #[serde(default)]
    pub doc: String,
}

/// Allowed element types for a type variable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeConstraint {
    /// Type variable name, such as `T`.
    pub type_param: String,
    /// Allowed shared-IR data types.
    #[serde(with = "data_types")]
    pub allowed: Vec<DataType>,
}

/// Failure while loading or registering schemas.
#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    /// YAML could not be decoded.
    #[error("invalid op-schema YAML: {0}")]
    Yaml(#[from] serde_yaml::Error),
    /// The schema declares an invalid or overlapping version interval.
    #[error("invalid schema {domain}::{name}: {message}")]
    Invalid {
        /// Operator domain.
        domain: String,
        /// Operator name.
        name: String,
        /// Explanation of the invalid schema.
        message: String,
    },
}

/// Owned registry resolving `(op_type, domain, opset)` to an operator schema.
#[derive(Clone, Debug, Default)]
pub struct SchemaRegistry {
    schemas: HashMap<(String, String), Vec<OpSchema>>,
}

impl SchemaRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load one schema from YAML and register it.
    pub fn load_yaml(&mut self, yaml: &str) -> Result<(), SchemaError> {
        self.register(serde_yaml::from_str(yaml)?)
    }

    /// Register a schema. A later `since_version` supersedes earlier versions.
    pub fn register(&mut self, mut schema: OpSchema) -> Result<(), SchemaError> {
        schema.domain = normalize_domain(&schema.domain).to_string();
        validate_schema(&schema)?;
        let key = (schema.domain.clone(), schema.name.clone());
        let versions = self.schemas.entry(key).or_default();
        if versions
            .iter()
            .any(|current| current.since_version == schema.since_version)
        {
            return Err(SchemaError::Invalid {
                domain: schema.domain.clone(),
                name: schema.name.clone(),
                message: format!(
                    "a schema already exists at since_version {}",
                    schema.since_version
                ),
            });
        }
        versions.push(schema);
        versions.sort_by_key(|schema| schema.since_version);
        Ok(())
    }

    /// Resolve the schema whose interval contains `opset`.
    pub fn lookup(&self, op_type: &str, domain: &str, opset: u64) -> Option<&OpSchema> {
        self.schemas
            .get(&(normalize_domain(domain).to_string(), op_type.to_string()))?
            .iter()
            .rev()
            .find(|schema| schema.supports_opset(opset))
    }

    /// Whether any version of this operator is registered.
    pub fn contains_operator(&self, op_type: &str, domain: &str) -> bool {
        self.schemas
            .contains_key(&(normalize_domain(domain).to_string(), op_type.to_string()))
    }

    /// Iterate over every registered schema.
    pub fn iter(&self) -> impl Iterator<Item = &OpSchema> {
        self.schemas.values().flatten()
    }

    /// Load the embedded standard-schema starter set.
    pub fn builtins() -> Self {
        let mut registry = Self::new();
        for yaml in BUILTIN_YAML {
            registry
                .load_yaml(yaml)
                .expect("embedded ONNX op schema must be valid");
        }
        registry
    }
}

fn normalize_domain(domain: &str) -> &str {
    if domain.is_empty() || domain == "ai.onnx" {
        "ai.onnx"
    } else {
        domain
    }
}

fn validate_schema(schema: &OpSchema) -> Result<(), SchemaError> {
    let invalid = |message: &str| SchemaError::Invalid {
        domain: schema.domain.clone(),
        name: schema.name.clone(),
        message: message.into(),
    };
    if schema.name.is_empty() {
        return Err(invalid("operator name must not be empty"));
    }
    if schema.since_version == 0 {
        return Err(invalid("since_version must be at least 1"));
    }
    if schema
        .until_version
        .is_some_and(|until| until < schema.since_version)
    {
        return Err(invalid("until_version precedes since_version"));
    }
    if schema.inputs.iter().filter(|input| input.variadic).count() > 1
        || schema
            .inputs
            .iter()
            .position(|input| input.variadic)
            .is_some_and(|index| index + 1 != schema.inputs.len())
    {
        return Err(invalid(
            "a variadic input must be the only trailing variadic",
        ));
    }
    if schema
        .outputs
        .iter()
        .filter(|output| output.variadic)
        .count()
        > 1
        || schema
            .outputs
            .iter()
            .position(|output| output.variadic)
            .is_some_and(|index| index + 1 != schema.outputs.len())
    {
        return Err(invalid(
            "a variadic output must be the only trailing variadic",
        ));
    }
    if schema.attributes.iter().any(|attribute| {
        attribute
            .default
            .as_ref()
            .is_some_and(|value| !default_matches(value, attribute.attr_type))
    }) {
        return Err(invalid(
            "an attribute default does not match its declared type",
        ));
    }
    Ok(())
}

fn default_matches(value: &AttributeDefault, attr_type: AttributeType) -> bool {
    matches!(
        (value, attr_type),
        (AttributeDefault::Int(_), AttributeType::Int)
            | (AttributeDefault::Float(_), AttributeType::Float)
            | (AttributeDefault::String(_), AttributeType::String)
            | (AttributeDefault::Ints(_), AttributeType::Ints)
            | (AttributeDefault::Floats(_), AttributeType::Floats)
            | (AttributeDefault::Strings(_), AttributeType::Strings)
    )
}

const BUILTIN_YAML: &[&str] = &[
    include_str!("../../schemas/standard/matmul.yaml"),
    include_str!("../../schemas/standard/gemm.yaml"),
    include_str!("../../schemas/standard/add.yaml"),
    include_str!("../../schemas/standard/sub.yaml"),
    include_str!("../../schemas/standard/div.yaml"),
    include_str!("../../schemas/standard/relu.yaml"),
    include_str!("../../schemas/standard/conv.yaml"),
    include_str!("../../schemas/standard/mul.yaml"),
    include_str!("../../schemas/standard/identity.yaml"),
    include_str!("../../schemas/standard/if.yaml"),
    include_str!("../../schemas/standard/softmax.yaml"),
    include_str!("../../schemas/standard/layer_normalization.yaml"),
    include_str!("../../schemas/standard/gather.yaml"),
    include_str!("../../schemas/standard/reshape_v14.yaml"),
    include_str!("../../schemas/standard/reshape_v19.yaml"),
    include_str!("../../schemas/standard/reshape_v21.yaml"),
    include_str!("../../schemas/standard/reshape_v23.yaml"),
    include_str!("../../schemas/standard/reshape.yaml"),
    include_str!("../../schemas/standard/transpose_v13.yaml"),
    include_str!("../../schemas/standard/transpose_v21.yaml"),
    include_str!("../../schemas/standard/transpose_v23.yaml"),
    include_str!("../../schemas/standard/transpose.yaml"),
    include_str!("../../schemas/standard/concat.yaml"),
    include_str!("../../schemas/standard/slice.yaml"),
    include_str!("../../schemas/standard/sigmoid.yaml"),
    include_str!("../../schemas/standard/tanh.yaml"),
    include_str!("../../schemas/standard/erf.yaml"),
    include_str!("../../schemas/standard/sqrt.yaml"),
    include_str!("../../schemas/standard/exp.yaml"),
    include_str!("../../schemas/standard/log.yaml"),
    include_str!("../../schemas/standard/pow.yaml"),
    include_str!("../../schemas/standard/clip.yaml"),
    include_str!("../../schemas/standard/expand.yaml"),
    include_str!("../../schemas/standard/where.yaml"),
    include_str!("../../schemas/standard/reduce_sum.yaml"),
    include_str!("../../schemas/standard/reduce_mean.yaml"),
    include_str!("../../schemas/standard/neg.yaml"),
    include_str!("../../schemas/standard/abs.yaml"),
    include_str!("../../schemas/standard/mod.yaml"),
    include_str!("../../schemas/standard/log_softmax.yaml"),
    include_str!("../../schemas/standard/rms_normalization.yaml"),
    include_str!("../../schemas/standard/reduce_max.yaml"),
    include_str!("../../schemas/standard/reduce_min.yaml"),
    include_str!("../../schemas/standard/reduce_prod.yaml"),
    include_str!("../../schemas/standard/reduce_l1.yaml"),
    include_str!("../../schemas/standard/reduce_l2.yaml"),
    include_str!("../../schemas/standard/reduce_log_sum.yaml"),
    include_str!("../../schemas/standard/reduce_log_sum_exp.yaml"),
    include_str!("../../schemas/standard/reduce_sum_square.yaml"),
    include_str!("../../schemas/standard/arg_max.yaml"),
    include_str!("../../schemas/standard/arg_min.yaml"),
    include_str!("../../schemas/standard/gather_elements.yaml"),
    include_str!("../../schemas/standard/gather_nd.yaml"),
    include_str!("../../schemas/standard/equal.yaml"),
    include_str!("../../schemas/standard/greater.yaml"),
    include_str!("../../schemas/standard/less.yaml"),
    include_str!("../../schemas/standard/and.yaml"),
    include_str!("../../schemas/standard/or.yaml"),
    include_str!("../../schemas/standard/not.yaml"),
    include_str!("../../schemas/standard/cast.yaml"),
    include_str!("../../schemas/standard/shape.yaml"),
    include_str!("../../schemas/standard/size.yaml"),
    include_str!("../../schemas/standard/non_zero.yaml"),
    include_str!("../../schemas/standard/range.yaml"),
    include_str!("../../schemas/standard/split.yaml"),
];

// FOLLOW-UP §7.4: complete the standard and ONNX-ML YAML catalogues.

mod data_types {
    use onnx_runtime_ir::DataType;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(types: &[DataType], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        types
            .iter()
            .map(|data_type| name(*data_type))
            .collect::<Vec<_>>()
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<DataType>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Vec::<String>::deserialize(deserializer)?
            .into_iter()
            .map(|value| {
                parse(&value)
                    .ok_or_else(|| serde::de::Error::custom(format!("unknown data type '{value}'")))
            })
            .collect()
    }

    fn parse(value: &str) -> Option<DataType> {
        Some(match value {
            "undefined" => DataType::Undefined,
            "float32" => DataType::Float32,
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
            "float8e8m0" => DataType::Float8E8M0,
            "uint2" => DataType::Uint2,
            "int2" => DataType::Int2,
            _ => return None,
        })
    }

    fn name(value: DataType) -> &'static str {
        match value {
            DataType::Undefined => "undefined",
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
            DataType::Complex64 => "complex64",
            DataType::Complex128 => "complex128",
            DataType::BFloat16 => "bfloat16",
            DataType::Float8E4M3FN => "float8e4m3fn",
            DataType::Float8E4M3FNUZ => "float8e4m3fnuz",
            DataType::Float8E5M2 => "float8e5m2",
            DataType::Float8E5M2FNUZ => "float8e5m2fnuz",
            DataType::Uint4 => "uint4",
            DataType::Int4 => "int4",
            DataType::Float4E2M1 => "float4e2m1",
            DataType::Float8E8M0 => "float8e8m0",
            DataType::Uint2 => "uint2",
            DataType::Int2 => "int2",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RELU_V6: &str = r#"
domain: ""
name: Relu
since_version: 6
inputs: [{ name: X, type_str: T }]
outputs: [{ name: Y, type_str: T }]
type_constraints:
  - type_param: T
    allowed: [float16, float32]
"#;

    #[test]
    fn yaml_schema_round_trips_every_public_field() {
        let schema: OpSchema = serde_yaml::from_str(
            r#"
domain: example
name: Variadic
since_version: 2
until_version: 4
doc: example op
inputs:
  - { name: X, type_str: T, doc: input, optional: true, variadic: true, min_arity: 2 }
outputs:
  - { name: Y, type_str: T, doc: output, optional: true, variadic: true }
attributes:
  - { name: axis, type: int, required: true, default: 1, doc: axis }
type_constraints:
  - { type_param: T, allowed: [float32, int64] }
"#,
        )
        .unwrap();
        assert_eq!(schema.domain, "example");
        assert!(schema.supports_opset(3));
        assert!(!schema.supports_opset(5));
        assert!(schema.inputs[0].optional && schema.inputs[0].variadic);
        assert!(schema.outputs[0].optional && schema.outputs[0].variadic);
        assert_eq!(schema.inputs[0].min_arity, 2);
        assert_eq!(schema.outputs[0].min_arity, 1);
        assert_eq!(schema.attributes[0].attr_type, AttributeType::Int);
        assert_eq!(schema.attributes[0].default, Some(AttributeDefault::Int(1)));
        assert_eq!(
            schema.type_constraints[0].allowed,
            vec![DataType::Float32, DataType::Int64]
        );
        let encoded = serde_yaml::to_string(&schema).unwrap();
        assert_eq!(serde_yaml::from_str::<OpSchema>(&encoded).unwrap(), schema);
    }

    #[test]
    fn registry_resolves_domains_and_opset_ranges() {
        let mut registry = SchemaRegistry::new();
        registry.load_yaml(RELU_V6).unwrap();
        let mut newer: OpSchema = serde_yaml::from_str(RELU_V6).unwrap();
        newer.since_version = 13;
        newer.until_version = None;
        registry.register(newer).unwrap();
        assert_eq!(registry.lookup("Relu", "", 10).unwrap().since_version, 6);
        assert_eq!(
            registry
                .lookup("Relu", "ai.onnx", 21)
                .unwrap()
                .since_version,
            13
        );
        assert!(registry.lookup("Relu", "", 5).is_none());
        assert!(registry.contains_operator("Relu", ""));
        assert_eq!(registry.iter().count(), 2);
    }

    #[test]
    fn registry_rejects_invalid_and_duplicate_version_schemas() {
        let mut registry = SchemaRegistry::new();
        registry.load_yaml(RELU_V6).unwrap();
        assert!(registry.load_yaml(RELU_V6).is_err());
        let invalid = RELU_V6.replace("since_version: 6", "since_version: 0");
        assert!(matches!(
            SchemaRegistry::new().load_yaml(&invalid),
            Err(SchemaError::Invalid { .. })
        ));
        assert!(matches!(
            SchemaRegistry::new().load_yaml("not: [valid"),
            Err(SchemaError::Yaml(_))
        ));
    }

    #[test]
    fn builtins_contain_expected_common_ops() {
        let registry = SchemaRegistry::builtins();
        for name in [
            "MatMul",
            "Gemm",
            "Add",
            "Sub",
            "Div",
            "Relu",
            "Conv",
            "Mul",
            "Identity",
            "If",
            "Softmax",
            "LayerNormalization",
            "Gather",
            "Reshape",
            "Transpose",
            "Concat",
            "Slice",
            "Sigmoid",
            "Tanh",
            "Erf",
            "Sqrt",
            "Exp",
            "Log",
            "Pow",
            "Clip",
            "Expand",
            "Where",
            "ReduceSum",
            "ReduceMean",
            "Neg",
            "Abs",
            "Mod",
            "LogSoftmax",
            "RMSNormalization",
            "ReduceMax",
            "ReduceMin",
            "ReduceProd",
            "ReduceL1",
            "ReduceL2",
            "ReduceLogSum",
            "ReduceLogSumExp",
            "ReduceSumSquare",
            "ArgMax",
            "ArgMin",
            "GatherElements",
            "GatherND",
            "Equal",
            "Greater",
            "Less",
            "And",
            "Or",
            "Not",
            "Cast",
            "Shape",
            "Size",
            "NonZero",
            "Range",
            "Split",
        ] {
            assert!(registry.lookup(name, "", 25).is_some(), "{name}");
        }
    }

    #[test]
    fn round_five_schemas_match_official_signatures() {
        let registry = SchemaRegistry::builtins();

        for (name, since_version) in [
            ("GatherElements", 13),
            ("GatherND", 13),
            ("Equal", 19),
            ("Greater", 13),
            ("Less", 13),
            ("And", 7),
            ("Or", 7),
            ("Not", 1),
            ("Cast", 25),
            ("Shape", 25),
            ("Size", 25),
            ("NonZero", 13),
            ("Range", 11),
            ("Split", 18),
        ] {
            assert_eq!(
                registry.lookup(name, "", 25).unwrap().since_version,
                since_version,
                "{name}"
            );
        }

        let gather_elements = registry.lookup("GatherElements", "", 25).unwrap();
        assert_eq!(
            gather_elements.attributes[0].default,
            Some(AttributeDefault::Int(0))
        );
        assert_eq!(
            gather_elements.type_constraints[1].allowed,
            [DataType::Int32, DataType::Int64]
        );

        let gather_nd = registry.lookup("GatherND", "", 25).unwrap();
        assert_eq!(gather_nd.inputs[1].type_str, "tensor(int64)");
        assert_eq!(
            gather_nd.attributes[0].default,
            Some(AttributeDefault::Int(0))
        );

        for name in ["Equal", "Greater", "Less", "And", "Or"] {
            let schema = registry.lookup(name, "", 25).unwrap();
            assert_eq!(schema.outputs[0].type_str, "T1");
            assert_eq!(
                schema.type_constraints.last().unwrap().allowed,
                [DataType::Bool],
                "{name}"
            );
        }
        assert_eq!(
            registry.lookup("Not", "", 25).unwrap().type_constraints[0].allowed,
            [DataType::Bool]
        );

        let cast = registry.lookup("Cast", "", 25).unwrap();
        assert_eq!(
            cast.attributes
                .iter()
                .find(|attribute| attribute.name == "round_mode")
                .unwrap()
                .default,
            Some(AttributeDefault::String("up".into()))
        );
        assert!(
            cast.attributes
                .iter()
                .find(|attribute| attribute.name == "to")
                .unwrap()
                .required
        );
        assert_eq!(cast.type_constraints[0].allowed.len(), 24);

        for name in ["Shape", "Size"] {
            let schema = registry.lookup(name, "", 25).unwrap();
            assert_eq!(schema.type_constraints[0].allowed.len(), 26);
            assert_eq!(schema.type_constraints[1].allowed, [DataType::Int64]);
        }

        let split = registry.lookup("Split", "", 25).unwrap();
        assert!(split.inputs[1].optional);
        assert!(split.outputs[0].variadic);
        assert_eq!(split.outputs[0].min_arity, 1);
        assert_eq!(
            registry.lookup("Range", "", 25).unwrap().type_constraints[0].allowed,
            [
                DataType::Float32,
                DataType::Float64,
                DataType::Int16,
                DataType::Int32,
                DataType::Int64
            ]
        );
    }

    #[test]
    fn round_four_schemas_match_official_signatures() {
        let registry = SchemaRegistry::builtins();
        let reductions = [
            ("ReduceMax", 20),
            ("ReduceMin", 20),
            ("ReduceProd", 18),
            ("ReduceL1", 18),
            ("ReduceL2", 18),
            ("ReduceLogSum", 18),
            ("ReduceLogSumExp", 18),
            ("ReduceSumSquare", 18),
        ];
        for (name, since_version) in reductions {
            let schema = registry.lookup(name, "", 24).unwrap();
            assert_eq!(schema.since_version, since_version);
            assert_eq!(schema.inputs.len(), 2);
            assert!(schema.inputs[1].optional);
            assert_eq!(schema.inputs[1].type_str, "tensor(int64)");
            assert_eq!(schema.outputs.len(), 1);
            assert_eq!(
                schema
                    .attributes
                    .iter()
                    .find(|attribute| attribute.name == "keepdims")
                    .unwrap()
                    .default,
                Some(AttributeDefault::Int(1))
            );
            assert_eq!(
                schema
                    .attributes
                    .iter()
                    .find(|attribute| attribute.name == "noop_with_empty_axes")
                    .unwrap()
                    .default,
                Some(AttributeDefault::Int(0))
            );
        }

        let rms = registry.lookup("RMSNormalization", "", 24).unwrap();
        assert_eq!(rms.since_version, 23);
        assert_eq!(
            rms.inputs
                .iter()
                .map(|input| input.type_str.as_str())
                .collect::<Vec<_>>(),
            ["T", "V"]
        );
        assert_eq!(rms.outputs[0].type_str, "V");
        assert_eq!(rms.type_constraints.len(), 2);

        for name in ["ArgMax", "ArgMin"] {
            let schema = registry.lookup(name, "", 24).unwrap();
            assert_eq!(schema.since_version, 13);
            assert_eq!(schema.outputs[0].type_str, "tensor(int64)");
            assert_eq!(schema.attributes.len(), 3);
        }
        let log_softmax = registry.lookup("LogSoftmax", "", 24).unwrap();
        assert_eq!(log_softmax.since_version, 13);
        assert_eq!(
            log_softmax.attributes[0].default,
            Some(AttributeDefault::Int(-1))
        );
    }

    #[test]
    fn softmax_schema_matches_opset_13() {
        let schema = SchemaRegistry::builtins()
            .lookup("Softmax", "", 24)
            .unwrap()
            .clone();
        assert_eq!(
            (
                schema.since_version,
                schema.inputs.len(),
                schema.outputs.len()
            ),
            (13, 1, 1)
        );
        assert_eq!(
            schema.attributes[0].default,
            Some(AttributeDefault::Int(-1))
        );
    }

    #[test]
    fn layer_normalization_schema_matches_opset_17() {
        let registry = SchemaRegistry::builtins();
        let schema = registry.lookup("LayerNormalization", "", 24).unwrap();
        assert_eq!(
            (
                schema.since_version,
                schema.inputs.len(),
                schema.outputs.len()
            ),
            (17, 3, 3)
        );
        assert!(schema.inputs[2].optional);
        assert!(schema.outputs[1].optional && schema.outputs[2].optional);
        assert_eq!(schema.type_constraints.len(), 2);
    }

    #[test]
    fn gather_schema_matches_opset_13() {
        let registry = SchemaRegistry::builtins();
        let schema = registry.lookup("Gather", "", 24).unwrap();
        assert_eq!(
            (
                schema.since_version,
                schema.inputs.len(),
                schema.outputs.len()
            ),
            (13, 2, 1)
        );
        assert_eq!(schema.inputs[1].type_str, "Tind");
        assert_eq!(
            schema.type_constraints[1].allowed,
            [DataType::Int32, DataType::Int64]
        );
    }

    #[test]
    fn reshape_schema_matches_opset_24() {
        let registry = SchemaRegistry::builtins();
        assert_eq!(
            registry.lookup("Reshape", "", 18).unwrap().since_version,
            14
        );
        assert_eq!(
            registry.lookup("Reshape", "", 19).unwrap().since_version,
            19
        );
        assert_eq!(
            registry.lookup("Reshape", "", 21).unwrap().since_version,
            21
        );
        assert_eq!(
            registry.lookup("Reshape", "", 23).unwrap().since_version,
            23
        );
        let schema = registry.lookup("Reshape", "", 24).unwrap();
        assert_eq!(
            (
                schema.since_version,
                schema.inputs.len(),
                schema.outputs.len()
            ),
            (24, 2, 1)
        );
        assert_eq!(schema.inputs[1].type_str, "tensor(int64)");
        assert_eq!(schema.attributes[0].name, "allowzero");
    }

    #[test]
    fn transpose_schema_matches_opset_24() {
        let registry = SchemaRegistry::builtins();
        assert_eq!(
            registry.lookup("Transpose", "", 20).unwrap().since_version,
            13
        );
        assert_eq!(
            registry.lookup("Transpose", "", 21).unwrap().since_version,
            21
        );
        assert_eq!(
            registry.lookup("Transpose", "", 23).unwrap().since_version,
            23
        );
        let schema = registry.lookup("Transpose", "", 24).unwrap();
        assert_eq!(
            (
                schema.since_version,
                schema.inputs.len(),
                schema.outputs.len()
            ),
            (24, 1, 1)
        );
        assert_eq!(schema.attributes[0].attr_type, AttributeType::Ints);
    }

    #[test]
    fn concat_schema_matches_opset_13() {
        let registry = SchemaRegistry::builtins();
        let schema = registry.lookup("Concat", "", 24).unwrap();
        assert_eq!(
            (
                schema.since_version,
                schema.inputs.len(),
                schema.outputs.len()
            ),
            (13, 1, 1)
        );
        assert!(schema.inputs[0].variadic);
        assert_eq!(schema.inputs[0].min_arity, 1);
        assert!(schema.attributes[0].required);
    }

    #[test]
    fn slice_schema_matches_opset_13() {
        let registry = SchemaRegistry::builtins();
        let schema = registry.lookup("Slice", "", 24).unwrap();
        assert_eq!(
            (
                schema.since_version,
                schema.inputs.len(),
                schema.outputs.len()
            ),
            (13, 5, 1)
        );
        assert!(!schema.inputs[2].optional);
        assert!(schema.inputs[3].optional && schema.inputs[4].optional);
    }

    macro_rules! common_schema_test {
        ($test:ident, $name:literal, $since:literal, $inputs:literal, $outputs:literal) => {
            #[test]
            fn $test() {
                let registry = SchemaRegistry::builtins();
                let schema = registry.lookup($name, "", 25).unwrap();
                assert_eq!(schema.since_version, $since);
                assert_eq!(schema.inputs.len(), $inputs);
                assert_eq!(schema.outputs.len(), $outputs);
            }
        };
    }

    common_schema_test!(sigmoid_schema_matches_opset_13, "Sigmoid", 13, 1, 1);
    common_schema_test!(tanh_schema_matches_opset_13, "Tanh", 13, 1, 1);
    common_schema_test!(erf_schema_matches_opset_13, "Erf", 13, 1, 1);
    common_schema_test!(sqrt_schema_matches_opset_13, "Sqrt", 13, 1, 1);
    common_schema_test!(exp_schema_matches_opset_13, "Exp", 13, 1, 1);
    common_schema_test!(log_schema_matches_opset_13, "Log", 13, 1, 1);
    common_schema_test!(pow_schema_matches_opset_15, "Pow", 15, 2, 1);
    common_schema_test!(clip_schema_matches_opset_13, "Clip", 13, 3, 1);
    common_schema_test!(expand_schema_matches_opset_13, "Expand", 13, 2, 1);
    common_schema_test!(where_schema_matches_opset_16, "Where", 16, 3, 1);
    common_schema_test!(reduce_sum_schema_matches_opset_13, "ReduceSum", 13, 2, 1);
    common_schema_test!(reduce_mean_schema_matches_opset_18, "ReduceMean", 18, 2, 1);
    common_schema_test!(sub_schema_matches_opset_14, "Sub", 14, 2, 1);
    common_schema_test!(div_schema_matches_opset_14, "Div", 14, 2, 1);
    common_schema_test!(neg_schema_matches_opset_13, "Neg", 13, 1, 1);
    common_schema_test!(abs_schema_matches_opset_13, "Abs", 13, 1, 1);
    common_schema_test!(mod_schema_matches_opset_13, "Mod", 13, 2, 1);

    #[test]
    fn added_schema_details_match_onnx_v1_20() {
        let registry = SchemaRegistry::builtins();

        let pow = registry.lookup("Pow", "", 25).unwrap();
        assert_eq!(pow.type_constraints[0].type_param, "T");
        assert_eq!(pow.type_constraints[1].type_param, "T1");
        assert_eq!(pow.type_constraints[0].allowed.len(), 6);
        assert_eq!(pow.type_constraints[1].allowed.len(), 12);

        let clip = registry.lookup("Clip", "", 25).unwrap();
        assert!(clip.inputs[1].optional && clip.inputs[2].optional);
        assert_eq!(clip.type_constraints[0].allowed.len(), 12);

        let expand = registry.lookup("Expand", "", 25).unwrap();
        assert_eq!(expand.inputs[1].type_str, "tensor(int64)");
        assert_eq!(expand.type_constraints[0].allowed.len(), 16);

        let where_op = registry.lookup("Where", "", 25).unwrap();
        assert_eq!(where_op.type_constraints[0].allowed, [DataType::Bool]);
        assert_eq!(where_op.type_constraints[1].allowed.len(), 16);

        for name in ["ReduceSum", "ReduceMean"] {
            let reduce = registry.lookup(name, "", 25).unwrap();
            assert!(reduce.inputs[1].optional);
            assert_eq!(reduce.attributes[0].default, Some(AttributeDefault::Int(1)));
            assert_eq!(reduce.attributes[1].default, Some(AttributeDefault::Int(0)));
            assert_eq!(reduce.type_constraints[0].allowed.len(), 8);
        }

        for name in ["Sub", "Div", "Abs", "Mod"] {
            assert_eq!(
                registry.lookup(name, "", 25).unwrap().type_constraints[0]
                    .allowed
                    .len(),
                12,
                "{name}"
            );
        }
        assert_eq!(
            registry.lookup("Neg", "", 25).unwrap().type_constraints[0]
                .allowed
                .len(),
            8
        );
        assert_eq!(
            registry.lookup("Mod", "", 25).unwrap().attributes[0].default,
            Some(AttributeDefault::Int(0))
        );
    }
}
