//! `ai.onnx.ml` tensor rules.
//!
//! These operators intentionally exclude `ZipMap`: its map-valued output
//! cannot be represented by the tensor-only [`TypeInfo`] inference model.
//!
//! `StringNormalizer` and `TfIdfVectorizer` are catalogued here alongside the
//! ML transforms, but they are **default-domain** (`ai.onnx`) string ops — not
//! `ai.onnx.ml` — so they are registered under the empty domain and gate
//! against the default opset.

use onnx_runtime_ir::{Attribute, DataType};

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::registry::InferenceRegistry;

/// Shape-preserving ML transforms whose output type is the input type.
fn same_type(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let Some(input) = ctx.input_type(0).cloned() {
        ctx.set_output_type(0, input);
    }
    Ok(())
}

/// `Normalizer` and `Scaler` preserve dimensions but always produce float32.
fn float_output(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let Some(shape) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) {
        ctx.set_output(0, DataType::Float32, shape);
    }
    Ok(())
}

/// `ArrayFeatureExtractor` selects the trailing features named by its rank-1
/// index tensor, replacing the final extent with the index count.
fn array_feature_extractor(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    if input.shape.is_empty() {
        return Err(ShapeInferError::InvalidRank {
            op: "ArrayFeatureExtractor".into(),
            index: 0,
            rank: 0,
            detail: "input must have rank at least 1".into(),
        });
    }
    let Some(indices) = ctx.input_shape(1) else {
        return Ok(());
    };
    if indices.len() != 1 {
        return Err(ShapeInferError::InvalidRank {
            op: "ArrayFeatureExtractor".into(),
            index: 1,
            rank: indices.len(),
            detail: "indices must be rank 1".into(),
        });
    }
    let mut shape = input.shape;
    shape.pop();
    shape.push(indices[0].clone());
    ctx.set_output(0, input.dtype, shape);
    Ok(())
}

/// Infer a `LabelEncoder` (v2+) output dtype from its value/default
/// attributes. The v2/v4 schemas use `values_*` (or tensors at v4).
fn label_dtype(ctx: &InferenceContext) -> Option<DataType> {
    for (attr, dtype) in [
        ("values_strings", DataType::String),
        ("values_int64s", DataType::Int64),
        ("values_floats", DataType::Float32),
    ] {
        if ctx.node.attr(attr).is_some() {
            return Some(dtype);
        }
    }
    if let Some(Attribute::Tensor(t)) = ctx.node.attr("values_tensor") {
        return Some(t.dtype);
    }
    for (attr, dtype) in [
        ("default_string", DataType::String),
        ("default_int64", DataType::Int64),
        ("default_float", DataType::Float32),
    ] {
        if ctx.node.attr(attr).is_some() {
            return Some(dtype);
        }
    }
    if let Some(Attribute::Tensor(t)) = ctx.node.attr("default_tensor") {
        return Some(t.dtype);
    }
    None
}

/// `LabelEncoder`-1 maps between int64 and string in either direction. Like
/// `CategoryMapper`, the direction — and therefore the output dtype — is chosen
/// by which `default_*` attribute is present: `default_int64` set converts
/// strings to int64 (int64 output); `default_string` set converts int64 to
/// strings (string output). `classes_strings` is present in *both* directions,
/// so it cannot select the output dtype.
fn label_encoder_v1(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(shape) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let dtype = match (
        ctx.node.attr("default_string").is_some(),
        ctx.node.attr("default_int64").is_some(),
    ) {
        (true, false) => DataType::String,
        (false, true) => DataType::Int64,
        _ => return Ok(()),
    };
    ctx.set_output(0, dtype, shape);
    Ok(())
}

fn label_encoder(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let (Some(shape), Some(dtype)) = (
        ctx.input_shape(0).map(<[DimExpr]>::to_vec),
        label_dtype(ctx),
    ) {
        ctx.set_output(0, dtype, shape);
    }
    Ok(())
}

/// `CategoryMapper` changes strings to int64 or int64 to strings according to
/// the selected default-value attribute, without changing tensor dimensions.
fn category_mapper(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(shape) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let dtype = match (
        ctx.node.attr("default_string").is_some(),
        ctx.node.attr("default_int64").is_some(),
    ) {
        (true, false) => DataType::String,
        (false, true) => DataType::Int64,
        _ => return Ok(()),
    };
    ctx.set_output(0, dtype, shape);
    Ok(())
}

/// `TfIdfVectorizer` accepts a sequence or a batch of sequences. Its vocabulary
/// extent is the largest `ngram_indexes` entry plus one.
fn tf_idf_vectorizer(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    if !(1..=2).contains(&input.len()) {
        return Err(ShapeInferError::InvalidRank {
            op: "TfIdfVectorizer".into(),
            index: 0,
            rank: input.len(),
            detail: "input must be rank 1 or 2".into(),
        });
    }
    let extent = ctx
        .node
        .attr("ngram_indexes")
        .and_then(Attribute::as_ints)
        .and_then(|indexes| indexes.iter().copied().max())
        .and_then(|index| index.checked_add(1))
        .filter(|&extent| extent >= 0)
        .map(DimExpr::constant)
        .unwrap_or_else(|| ctx.fresh_dim());
    let shape = if input.len() == 1 {
        vec![extent]
    } else {
        vec![input[0].clone(), extent]
    };
    ctx.set_output(0, DataType::Float32, shape);
    Ok(())
}

/// `StringNormalizer` removes stopwords from the sequence dimension. It
/// accepts `[C]` or `[1, C]`, so that trailing extent is data-dependent.
fn string_normalizer(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    if !(1..=2).contains(&input.len()) {
        return Err(ShapeInferError::InvalidRank {
            op: "StringNormalizer".into(),
            index: 0,
            rank: input.len(),
            detail: "input must be rank 1 or 2".into(),
        });
    }
    let mut shape = input;
    *shape.last_mut().expect("validated non-empty rank") = ctx.fresh_dim();
    ctx.set_output(0, DataType::String, shape);
    Ok(())
}

/// Register tensor-valued `ai.onnx.ml` operators, plus the default-domain
/// (`ai.onnx`) string ops `StringNormalizer` and `TfIdfVectorizer`.
pub fn register(reg: &mut InferenceRegistry) {
    const ML: &str = "ai.onnx.ml";
    const DEFAULT: &str = "";
    reg.register(ML, "ArrayFeatureExtractor", 1, array_feature_extractor);
    reg.register(ML, "Binarizer", 1, same_type);
    reg.register(ML, "CategoryMapper", 1, category_mapper);
    reg.register(ML, "Imputer", 1, same_type);
    reg.register(ML, "LabelEncoder", 1, label_encoder_v1);
    reg.register(ML, "LabelEncoder", 2, label_encoder);
    reg.register(ML, "LabelEncoder", 4, label_encoder);
    reg.register(ML, "Normalizer", 1, float_output);
    reg.register(ML, "Scaler", 1, float_output);
    // Default-domain (`ai.onnx`) string ops: StringNormalizer since v10,
    // TfIdfVectorizer since v9.
    reg.register(DEFAULT, "StringNormalizer", 10, string_normalizer);
    reg.register(DEFAULT, "TfIdfVectorizer", 9, tf_idf_vectorizer);
}
