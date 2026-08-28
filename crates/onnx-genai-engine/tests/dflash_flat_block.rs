//! Production DFlash v1 execution through the public engine surface.
//!
//! The two component artifacts are deliberately reduced, but are genuine ONNX
//! sessions.  The target produces the hidden conditioning and verifier logits;
//! the proposer receives the runtime-built anchor/mask block and produces every
//! candidate in one invocation.  No test replaces either component output.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use onnx_genai_engine::pipeline::speculative::DFlashGenerationCancelled;
use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, GenerationBoundary,
    GenerationControl, PackageExecutionError, PipelineGenerateRequest, SessionForkError,
    SessionPosition, package_execution_error,
};
use onnx_genai_ort::{DataType, Environment, Session, SessionOptions, Value};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};

const TARGET: &str = r#"
ir_version: 8
graph {
  node { input: "tokens" output: "shape" op_type: "Shape" }
  node {
    output: "hidden_width"
    op_type: "Constant"
    attribute { name: "value_ints" ints: 2 type: INTS }
  }
  node { input: "shape" input: "hidden_width" output: "hidden_shape" op_type: "Concat" attribute { name: "axis" i: 0 type: INT } }
  node { input: "token_embedding" input: "tokens" output: "token_hidden" op_type: "Gather" attribute { name: "axis" i: 0 type: INT } }
  node { input: "token_history" output: "history_f32" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
  node { output: "history_reduce_axis" op_type: "Constant" attribute { name: "value_ints" ints: 1 type: INTS } }
  node { input: "history_f32" input: "history_reduce_axis" output: "history_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 1 type: INT } }
  node { output: "hidden_axes" op_type: "Constant" attribute { name: "value_ints" ints: 2 type: INTS } }
  node { input: "history_sum" input: "hidden_axes" output: "history_hidden" op_type: "Unsqueeze" }
  node { output: "history_scale" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 0.01 } type: TENSOR } }
  node { input: "history_hidden" input: "history_scale" output: "scaled_history" op_type: "Mul" }
  node { input: "token_hidden" input: "scaled_history" output: "hidden" op_type: "Add" }
  node { output: "scale_5" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: {tap_5_scale} } type: TENSOR } }
  node { output: "scale_19" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: {tap_19_scale} } type: TENSOR } }
  node { output: "scale_33" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: {tap_33_scale} } type: TENSOR } }
  node { output: "scale_47" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: {tap_47_scale} } type: TENSOR } }
  node { output: "scale_61" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: {tap_61_scale} } type: TENSOR } }
  node { input: "hidden" input: "scale_5" output: "tap_5" op_type: "Mul" }
  node { input: "hidden" input: "scale_19" output: "tap_19" op_type: "Mul" }
  node { input: "hidden" input: "scale_33" output: "tap_33" op_type: "Mul" }
  node { input: "hidden" input: "scale_47" output: "tap_47" op_type: "Mul" }
  node { input: "hidden" input: "scale_61" output: "tap_61" op_type: "Mul" }
  node { input: "tap_5" input: "tap_19" input: "tap_33" input: "tap_47" input: "tap_61" output: "combined_features" op_type: "Concat" attribute { name: "axis" i: 2 type: INT } }
  node { input: "tap_5" input: "tap_19" input: "tap_33" output: "alternate_features" op_type: "Concat" attribute { name: "axis" i: 2 type: INT } }
  node { input: "hidden" output: "recurrent_prefixes" op_type: "Identity" }
  node { input: "token_history" input: "tokens" output: "token_history_out" op_type: "Concat" attribute { name: "axis" i: 1 type: INT } }
  node { input: "hidden" input: "lm_head" output: "logits" op_type: "MatMul" }
  name: "dflash_target"
  input { name: "tokens" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } } } } }
  input { name: "recurrent" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_value: 2 } } } } }
  input { name: "token_history" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "state_sequence" } } } } }
  output { name: "hidden" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  output { name: "tap_5" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  output { name: "tap_19" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  output { name: "tap_33" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  output { name: "tap_47" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  output { name: "tap_61" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  output { name: "combined_features" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 10 } } } } }
  output { name: "alternate_features" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 6 } } } } }
  output { name: "logits" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 4 } } } } }
  output { name: "recurrent_prefixes" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  output { name: "token_history_out" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "total_sequence" } } } } }
  initializer { name: "token_embedding" data_type: 1 dims: 4 dims: 2 {token_embedding_values} }
  initializer { name: "token_embedding_decoy" data_type: 1 dims: 4 dims: 2 {token_embedding_decoy_values} }
  initializer { name: "lm_head" data_type: 1 dims: 2 dims: 4 {lm_head_values} }
  initializer { name: "lm_head_decoy" data_type: 1 dims: 2 dims: 4 {lm_head_decoy_values} }
}
opset_import { domain: "" version: 18 }
"#;

const CONTEXTUAL_TARGET: &str = r#"
ir_version: 8
graph {
  node { input: "tokens" output: "shape" op_type: "Shape" }
  node {
    output: "hidden_width"
    op_type: "Constant"
    attribute { name: "value_ints" ints: 2 type: INTS }
  }
  node { input: "shape" input: "hidden_width" output: "hidden_shape" op_type: "Concat" attribute { name: "axis" i: 0 type: INT } }
  node { input: "token_embedding" input: "tokens" output: "token_hidden" op_type: "Gather" attribute { name: "axis" i: 0 type: INT } }
  node { input: "token_history" output: "history_f32" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
  node { output: "history_reduce_axis" op_type: "Constant" attribute { name: "value_ints" ints: 1 type: INTS } }
  node { input: "history_f32" input: "history_reduce_axis" output: "history_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 1 type: INT } }
  node { output: "hidden_axes" op_type: "Constant" attribute { name: "value_ints" ints: 2 type: INTS } }
  node { input: "history_sum" input: "hidden_axes" output: "history_hidden" op_type: "Unsqueeze" }
  node { output: "history_scale" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 0.01 } type: TENSOR } }
  node { input: "history_hidden" input: "history_scale" output: "scaled_history" op_type: "Mul" }
  node { input: "token_hidden" input: "scaled_history" output: "hidden" op_type: "Add" }
  node { output: "scale_5" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: {tap_5_scale} } type: TENSOR } }
  node { output: "scale_19" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: {tap_19_scale} } type: TENSOR } }
  node { output: "scale_33" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: {tap_33_scale} } type: TENSOR } }
  node { output: "scale_47" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: {tap_47_scale} } type: TENSOR } }
  node { output: "scale_61" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: {tap_61_scale} } type: TENSOR } }
  node { input: "hidden" input: "scale_5" output: "tap_5" op_type: "Mul" }
  node { input: "hidden" input: "scale_19" output: "tap_19" op_type: "Mul" }
  node { input: "hidden" input: "scale_33" output: "tap_33" op_type: "Mul" }
  node { input: "hidden" input: "scale_47" output: "tap_47" op_type: "Mul" }
  node { input: "hidden" input: "scale_61" output: "tap_61" op_type: "Mul" }
  node { input: "tap_5" input: "tap_19" input: "tap_33" input: "tap_47" input: "tap_61" output: "combined_features" op_type: "Concat" attribute { name: "axis" i: 2 type: INT } }
  node { input: "tap_5" input: "tap_19" input: "tap_33" output: "alternate_features" op_type: "Concat" attribute { name: "axis" i: 2 type: INT } }
  node { input: "hidden" output: "recurrent_prefixes" op_type: "Identity" }
  node { input: "token_history" input: "tokens" output: "token_history_out" op_type: "Concat" attribute { name: "axis" i: 1 type: INT } }
  node { input: "hidden" input: "lm_head" output: "logits" op_type: "MatMul" }
  name: "dflash_contextual_target"
  input { name: "tokens" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } } } } }
  input { name: "recurrent" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_value: 2 } } } } }
  input { name: "token_history" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "state_sequence" } } } } }
  output { name: "hidden" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  output { name: "tap_5" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  output { name: "tap_19" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  output { name: "tap_33" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  output { name: "tap_47" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  output { name: "tap_61" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  output { name: "combined_features" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 10 } } } } }
  output { name: "alternate_features" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 6 } } } } }
  output { name: "logits" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 4 } } } } }
  output { name: "recurrent_prefixes" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  output { name: "token_history_out" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "total_sequence" } } } } }
  initializer { name: "token_embedding" data_type: 1 dims: 4 dims: 2 {token_embedding_values} }
  initializer { name: "token_embedding_decoy" data_type: 1 dims: 4 dims: 2 {token_embedding_decoy_values} }
  initializer { name: "lm_head" data_type: 1 dims: 2 dims: 4 {lm_head_values} }
  initializer { name: "lm_head_decoy" data_type: 1 dims: 2 dims: 4 {lm_head_decoy_values} }
}
opset_import { domain: "" version: 18 }
"#;

const PROPOSER: &str = r#"
ir_version: 8
graph {
  node { output: "split_sizes" op_type: "Constant" attribute { name: "value" t { dims: 5 data_type: 7 int64_data: 2 int64_data: 2 int64_data: 2 int64_data: 2 int64_data: 2 } type: TENSOR } }
  node { input: "target_features" input: "split_sizes" output: "feature_5" output: "feature_19" output: "feature_33" output: "feature_47" output: "feature_61" op_type: "Split" attribute { name: "axis" i: 2 type: INT } }
  node { output: "reduce_axes" op_type: "Constant" attribute { name: "value" t { dims: 2 data_type: 7 int64_data: 1 int64_data: 2 } type: TENSOR } }
  node { input: "feature_5" input: "reduce_axes" output: "sum_5" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "feature_19" input: "reduce_axes" output: "sum_19" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "feature_33" input: "reduce_axes" output: "sum_33" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "feature_47" input: "reduce_axes" output: "sum_47" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "feature_61" input: "reduce_axes" output: "sum_61" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { output: "tap_weight_5" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 1.0 } type: TENSOR } }
  node { output: "tap_weight_19" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 3.0 } type: TENSOR } }
  node { output: "tap_weight_33" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 5.0 } type: TENSOR } }
  node { output: "tap_weight_47" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 7.0 } type: TENSOR } }
  node { output: "tap_weight_61" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 11.0 } type: TENSOR } }
  node { input: "sum_5" input: "tap_weight_5" output: "weighted_5" op_type: "Mul" }
  node { input: "sum_19" input: "tap_weight_19" output: "weighted_19" op_type: "Mul" }
  node { input: "sum_33" input: "tap_weight_33" output: "weighted_33" op_type: "Mul" }
  node { input: "sum_47" input: "tap_weight_47" output: "weighted_47" op_type: "Mul" }
  node { input: "sum_61" input: "tap_weight_61" output: "weighted_61" op_type: "Mul" }
  node { input: "weighted_5" input: "weighted_19" output: "drafter_layer_1" op_type: "Add" }
  node { input: "drafter_layer_1" input: "weighted_33" output: "drafter_layer_2" op_type: "Add" }
  node { input: "drafter_layer_2" input: "weighted_47" output: "drafter_layer_3" op_type: "Add" }
  node { input: "drafter_layer_3" input: "weighted_61" output: "drafter_layer_4" op_type: "Add" }
  node { input: "draft_history" output: "draft_history_f32" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
  node { output: "history_axis" op_type: "Constant" attribute { name: "value_ints" ints: 1 type: INTS } }
  node { input: "draft_history_f32" input: "history_axis" output: "draft_history_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "drafter_layer_4" input: "draft_history_sum" output: "drafter_layer_5" op_type: "Add" }
  node { input: "noise_embeddings" input: "reduce_axes" output: "noise_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "masked_positions" output: "masked_f32" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
  node { input: "masked_f32" input: "history_axis" output: "masked_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "position_ids" output: "positions_f32" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
  node { input: "positions_f32" input: "history_axis" output: "positions_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "attention_mask" output: "attention_f32" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
  node { input: "attention_f32" input: "history_axis" output: "attention_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { output: "projection_axes" op_type: "Constant" attribute { name: "value" t { dims: 2 data_type: 7 int64_data: 0 int64_data: 1 } type: TENSOR } }
  node { input: "output_projection" input: "projection_axes" output: "projection_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { output: "noise_weight" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 13.0 } type: TENSOR } }
  node { output: "mask_weight" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 17.0 } type: TENSOR } }
  node { output: "position_weight" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 19.0 } type: TENSOR } }
  node { output: "attention_weight" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 23.0 } type: TENSOR } }
  node { output: "projection_weight" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 29.0 } type: TENSOR } }
  node { input: "noise_sum" input: "noise_weight" output: "noise_signal" op_type: "Mul" }
  node { input: "masked_sum" input: "mask_weight" output: "mask_signal" op_type: "Mul" }
  node { input: "positions_sum" input: "position_weight" output: "position_signal" op_type: "Mul" }
  node { input: "attention_sum" input: "attention_weight" output: "attention_signal" op_type: "Mul" }
  node { input: "projection_sum" input: "projection_weight" output: "projection_signal" op_type: "Mul" }
  node { input: "drafter_layer_5" input: "noise_signal" output: "all_inputs_1" op_type: "Add" }
  node { input: "all_inputs_1" input: "mask_signal" output: "all_inputs_2" op_type: "Add" }
  node { input: "all_inputs_2" input: "position_signal" output: "all_inputs_3" op_type: "Add" }
  node { input: "all_inputs_3" input: "attention_signal" output: "all_inputs_4" op_type: "Add" }
  node { input: "all_inputs_4" input: "projection_signal" output: "all_inputs_signal" op_type: "Add" }
  node { input: "all_inputs_signal" output: "binding_signature" op_type: "Identity" }
  node { input: "all_inputs_signal" output: "candidate_i64" op_type: "Cast" attribute { name: "to" i: 7 type: INT } }
  node { output: "vocabulary" op_type: "Constant" attribute { name: "value" t { data_type: 7 int64_data: 4 } type: TENSOR } }
  node { input: "candidate_i64" input: "vocabulary" output: "candidate_mod" op_type: "Mod" }
  node { output: "candidate_axis" op_type: "Constant" attribute { name: "value_ints" ints: 1 type: INTS } }
  node { input: "candidate_mod" input: "candidate_axis" output: "candidate_row" op_type: "Unsqueeze" }
  node { output: "candidate_shape" op_type: "Constant" attribute { name: "value" t { dims: 2 data_type: 7 int64_data: 1 int64_data: 8 } type: TENSOR } }
  node { input: "candidate_row" input: "candidate_shape" output: "conditioning_offset" op_type: "Expand" }
  node {
    output: "candidate_base"
    op_type: "Constant"
    attribute {
      name: "value"
      t {
        dims: 1
        dims: 8
        data_type: 7
{candidate_values}
      }
      type: TENSOR
    }
  }
  node { input: "candidate_base" input: "conditioning_offset" output: "candidate_unbounded" op_type: "Add" }
  node { input: "candidate_unbounded" input: "vocabulary" output: "candidate_tokens" op_type: "Mod" }
  node { input: "draft_history" input: "candidate_tokens" output: "draft_history_out" op_type: "Concat" attribute { name: "axis" i: 1 type: INT } }
  node {
    output: "proposal_probabilities"
    op_type: "Constant"
    attribute {
      name: "value"
      t {
        dims: 1
        dims: 8
        dims: 4
        data_type: 1
{probability_values}
      }
      type: TENSOR
    }
  }
  name: "dflash_proposer"
  input { name: "target_features" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 10 } } } } }
  input { name: "noise_embeddings" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_value: 8 } dim { dim_value: 2 } } } } }
  input { name: "masked_positions" type { tensor_type { elem_type: 9 shape { dim { dim_param: "batch" } dim { dim_value: 8 } } } } }
  input { name: "position_ids" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "total" } } } } }
  input { name: "attention_mask" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "total" } } } } }
  input { name: "output_projection" type { tensor_type { elem_type: 1 shape { dim { dim_value: 2 } dim { dim_value: 4 } } } } }
  input { name: "draft_history" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "state_sequence" } } } } }
  output { name: "candidate_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_value: 8 } } } } }
  output { name: "proposal_probabilities" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_value: 8 } dim { dim_value: 4 } } } } }
  output { name: "draft_history_out" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "total_sequence" } } } } }
  output { name: "binding_signature" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } } } } }
}
opset_import { domain: "" version: 18 }
"#;

const ALTERNATE_PROPOSER: &str = r#"
ir_version: 8
graph {
  node { output: "split_sizes" op_type: "Constant" attribute { name: "value" t { dims: 3 data_type: 7 int64_data: 2 int64_data: 2 int64_data: 2 } type: TENSOR } }
  node { input: "target_features" input: "split_sizes" output: "early" output: "middle" output: "late" op_type: "Split" attribute { name: "axis" i: 2 type: INT } }
  node { output: "reduce_axes" op_type: "Constant" attribute { name: "value" t { dims: 2 data_type: 7 int64_data: 1 int64_data: 2 } type: TENSOR } }
  node { input: "early" input: "reduce_axes" output: "early_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "middle" input: "reduce_axes" output: "middle_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "late" input: "reduce_axes" output: "late_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { output: "early_weight" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 1.0 } type: TENSOR } }
  node { output: "middle_weight" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 3.0 } type: TENSOR } }
  node { output: "late_weight" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 5.0 } type: TENSOR } }
  node { input: "early_sum" input: "early_weight" output: "early_signal" op_type: "Mul" }
  node { input: "middle_sum" input: "middle_weight" output: "middle_signal" op_type: "Mul" }
  node { input: "late_sum" input: "late_weight" output: "late_signal" op_type: "Mul" }
  node { input: "early_signal" input: "middle_signal" output: "alternate_layer_1" op_type: "Add" }
  node { input: "alternate_layer_1" input: "late_signal" output: "alternate_layer_2" op_type: "Add" }
  node { input: "draft_history" output: "draft_history_f32" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
  node { output: "history_axis" op_type: "Constant" attribute { name: "value_ints" ints: 1 type: INTS } }
  node { input: "draft_history_f32" input: "history_axis" output: "history_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "noise_embeddings" input: "reduce_axes" output: "noise_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "masked_positions" output: "masked_f32" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
  node { input: "masked_f32" input: "history_axis" output: "masked_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "position_ids" output: "positions_f32" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
  node { input: "positions_f32" input: "history_axis" output: "positions_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "attention_mask" output: "attention_f32" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
  node { input: "attention_f32" input: "history_axis" output: "attention_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { output: "projection_axes" op_type: "Constant" attribute { name: "value" t { dims: 2 data_type: 7 int64_data: 0 int64_data: 1 } type: TENSOR } }
  node { input: "output_projection" input: "projection_axes" output: "projection_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { output: "history_weight" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 11.0 } type: TENSOR } }
  node { output: "noise_weight" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 13.0 } type: TENSOR } }
  node { output: "mask_weight" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 17.0 } type: TENSOR } }
  node { output: "position_weight" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 19.0 } type: TENSOR } }
  node { output: "attention_weight" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 23.0 } type: TENSOR } }
  node { output: "projection_weight" op_type: "Constant" attribute { name: "value" t { data_type: 1 float_data: 29.0 } type: TENSOR } }
  node { input: "history_sum" input: "history_weight" output: "history_signal" op_type: "Mul" }
  node { input: "noise_sum" input: "noise_weight" output: "noise_signal" op_type: "Mul" }
  node { input: "masked_sum" input: "mask_weight" output: "mask_signal" op_type: "Mul" }
  node { input: "positions_sum" input: "position_weight" output: "position_signal" op_type: "Mul" }
  node { input: "attention_sum" input: "attention_weight" output: "attention_signal" op_type: "Mul" }
  node { input: "projection_sum" input: "projection_weight" output: "projection_signal" op_type: "Mul" }
  node { input: "alternate_layer_2" input: "history_signal" output: "all_inputs_0" op_type: "Add" }
  node { input: "all_inputs_0" input: "noise_signal" output: "all_inputs_1" op_type: "Add" }
  node { input: "all_inputs_1" input: "mask_signal" output: "all_inputs_2" op_type: "Add" }
  node { input: "all_inputs_2" input: "position_signal" output: "all_inputs_3" op_type: "Add" }
  node { input: "all_inputs_3" input: "attention_signal" output: "all_inputs_4" op_type: "Add" }
  node { input: "all_inputs_4" input: "projection_signal" output: "all_inputs_signal" op_type: "Add" }
  node { input: "all_inputs_signal" output: "candidate_i64" op_type: "Cast" attribute { name: "to" i: 7 type: INT } }
  node { output: "vocabulary" op_type: "Constant" attribute { name: "value" t { data_type: 7 int64_data: 4 } type: TENSOR } }
  node { input: "candidate_i64" input: "vocabulary" output: "candidate_mod" op_type: "Mod" }
  node { output: "candidate_axis" op_type: "Constant" attribute { name: "value_ints" ints: 1 type: INTS } }
  node { input: "candidate_mod" input: "candidate_axis" output: "candidate_row" op_type: "Unsqueeze" }
  node { output: "candidate_shape" op_type: "Constant" attribute { name: "value" t { dims: 2 data_type: 7 int64_data: 1 int64_data: 5 } type: TENSOR } }
  node { input: "candidate_row" input: "candidate_shape" output: "conditioning_offset" op_type: "Expand" }
  node { output: "candidate_base" op_type: "Constant" attribute { name: "value" t { dims: 1 dims: 5 data_type: 7 int64_data: 1 int64_data: 1 int64_data: 1 int64_data: 1 int64_data: 1 } type: TENSOR } }
  node { input: "candidate_base" input: "conditioning_offset" output: "candidate_unbounded" op_type: "Add" }
  node { input: "candidate_unbounded" input: "vocabulary" output: "candidate_tokens" op_type: "Mod" }
  node { input: "draft_history" input: "candidate_tokens" output: "draft_history_out" op_type: "Concat" attribute { name: "axis" i: 1 type: INT } }
  node { output: "proposal_probabilities" op_type: "Constant" attribute { name: "value" t { dims: 1 dims: 5 dims: 4 data_type: 1
    float_data: 1.0 float_data: 0.0 float_data: 0.0 float_data: 0.0
    float_data: 1.0 float_data: 0.0 float_data: 0.0 float_data: 0.0
    float_data: 1.0 float_data: 0.0 float_data: 0.0 float_data: 0.0
    float_data: 1.0 float_data: 0.0 float_data: 0.0 float_data: 0.0
    float_data: 1.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 } type: TENSOR } }
  name: "alternate_dflash_proposer"
  input { name: "target_features" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 6 } } } } }
  input { name: "noise_embeddings" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_value: 5 } dim { dim_value: 2 } } } } }
  input { name: "masked_positions" type { tensor_type { elem_type: 9 shape { dim { dim_param: "batch" } dim { dim_value: 5 } } } } }
  input { name: "position_ids" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "total" } } } } }
  input { name: "attention_mask" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "total" } } } } }
  input { name: "output_projection" type { tensor_type { elem_type: 1 shape { dim { dim_value: 2 } dim { dim_value: 4 } } } } }
  input { name: "draft_history" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "state_sequence" } } } } }
  output { name: "candidate_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_value: 5 } } } } }
  output { name: "proposal_probabilities" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_value: 5 } dim { dim_value: 4 } } } } }
  output { name: "draft_history_out" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "total_sequence" } } } } }
}
opset_import { domain: "" version: 18 }
"#;

#[derive(Clone, Copy)]
enum FixtureGeometry {
    Qwen,
    Alternate,
}

fn metadata(version: &str, probabilities: bool) -> String {
    metadata_for_geometry(version, probabilities, FixtureGeometry::Qwen)
}

fn metadata_for_geometry(version: &str, probabilities: bool, geometry: FixtureGeometry) -> String {
    let (block, hidden_width, feature_binding, conditioning_sources) = match geometry {
        FixtureGeometry::Qwen => (
            8,
            10,
            "target.combined_features",
            "        - { component: target_component, output: tap_5 }\n        - { component: target_component, output: tap_19 }\n        - { component: target_component, output: tap_33 }\n        - { component: target_component, output: tap_47 }\n        - { component: target_component, output: tap_61 }",
        ),
        FixtureGeometry::Alternate => (
            5,
            6,
            "target.alternate_features",
            "        - { component: target_component, output: tap_5 }\n        - { component: target_component, output: tap_19 }\n        - { component: target_component, output: tap_33 }",
        ),
    };
    let max_proposal_width = block - 1;
    let probability_port = if probabilities {
        "      proposal_probabilities: proposal_probabilities\n"
    } else {
        ""
    };
    let probability_binding = if probabilities {
        "          proposal_probabilities: proposal.probabilities\n"
    } else {
        ""
    };
    let probability_output = if probabilities {
        format!(
            "            proposal_probabilities: {{ dtype: float32, shape: [batch, {block}, 4], batch_layout: {{ kind: request_aligned, axis: 0 }} }}\n"
        )
    } else {
        String::new()
    };
    let verification_probabilities = if probabilities {
        "    probabilities:\n      proposal: { component: proposer_component, output: proposal_probabilities }\n      target: { component: target_component, output: logits }\n"
    } else {
        ""
    };
    let (v2_outputs, v2_bindings, structure) = if version == "2" {
        (
            "            selector_candidates: { dtype: int64, shape: [batch, 3, 1], batch_layout: { kind: request_aligned, axis: 0 } }\n            selector_probabilities: { dtype: float32, shape: [batch, 3, 1], batch_layout: { kind: request_aligned, axis: 0 } }\n",
            "          selector_candidates: proposal.selector_candidates\n          selector_probabilities: proposal.selector_probabilities\n",
            "    structure:\n      kind: selector_convolution_v1\n      selector:\n        selected_tokens_output: candidate_tokens\n        candidate_ids_output: selector_candidates\n        conditional_probabilities_output: selector_probabilities\n        top_k: 1\n        rank: 1\n      convolution:\n        kernel_size: 2\n        group_size: 1\n        first_position_reads_anchor: true",
        )
    } else {
        ("", "", "    structure: { kind: base }")
    };
    format!(
        r#"
schema_version: v1.6
package:
  tokenizer:
    special_tokens:
      eos_token_id: [3]
pipeline:
  workflow:
    manifest: {{}}
    inputs:
      request.tokens:
        contract: {{ dtype: int64, shape: [batch, sequence], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: runtime, version: v1, role: prompt_tokens }}
        source: {{ kind: request }}
        required: true
      request.active:
        contract: {{ dtype: bool, shape: [batch], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: true
      request.done:
        contract: {{ dtype: bool, shape: [batch], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: false
      request.accepted_len:
        contract: {{ dtype: int64, shape: [batch], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: 0
      request.recurrent:
        contract: {{ dtype: float32, shape: [batch, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: 0.0
      request.history:
        contract: {{ dtype: int64, shape: [batch, sequence], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: 0
      package.noise:
        contract: {{ dtype: float32, shape: [batch, {block}, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: 0.0
      package.masked:
        contract: {{ dtype: bool, shape: [batch, {block}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: false
      package.positions:
        contract: {{ dtype: int64, shape: [batch, total], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: 0
      package.attention:
        contract: {{ dtype: int64, shape: [batch, total], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: 0
      package.output_projection:
        contract: {{ dtype: float32, shape: [2, 4], batch_layout: {{ kind: shared }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: 0.0
    outputs: {{}}
    components:
      termination_policy:
        implementation: {{ kind: binding }}
        ports: {{}}
        contract: {{ id: onnx-genai.token-policy, version: "1.0" }}
      target_component:
        implementation: {{ kind: onnx, artifact: target.onnx.textproto }}
        contract: {{ id: onnx-genai.dflash-target-state, version: "1.0" }}
        ports:
          inputs:
            tokens: {{ dtype: int64, shape: [batch, sequence], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            recurrent: {{ dtype: float32, shape: [batch, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            token_history: {{ dtype: int64, shape: [batch, state_sequence], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
          outputs:
            hidden: {{ dtype: float32, shape: [batch, sequence, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            tap_5: {{ dtype: float32, shape: [batch, sequence, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            tap_19: {{ dtype: float32, shape: [batch, sequence, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            tap_33: {{ dtype: float32, shape: [batch, sequence, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            tap_47: {{ dtype: float32, shape: [batch, sequence, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            tap_61: {{ dtype: float32, shape: [batch, sequence, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            combined_features: {{ dtype: float32, shape: [batch, sequence, 10], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            alternate_features: {{ dtype: float32, shape: [batch, sequence, 6], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            logits: {{ dtype: float32, shape: [batch, sequence, 4], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            recurrent_prefixes: {{ dtype: float32, shape: [batch, sequence, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            token_history_out: {{ dtype: int64, shape: [batch, total_sequence], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
          roles:
            hidden: hidden_states
            tap_5: hidden_states
            tap_19: hidden_states
            tap_33: hidden_states
            tap_47: hidden_states
            tap_61: hidden_states
            logits: logits
      proposer_component:
        implementation: {{ kind: onnx, artifact: proposer.onnx.textproto }}
        ports:
          inputs:
            target_features: {{ dtype: float32, shape: [batch, sequence, {hidden_width}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            noise_embeddings: {{ dtype: float32, shape: [batch, {block}, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            masked_positions: {{ dtype: bool, shape: [batch, {block}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            position_ids: {{ dtype: int64, shape: [batch, total], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            attention_mask: {{ dtype: int64, shape: [batch, total], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            output_projection: {{ dtype: float32, shape: [2, 4], batch_layout: {{ kind: shared }} }}
            draft_history: {{ dtype: int64, shape: [batch, state_sequence], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
          outputs:
            candidate_tokens: {{ dtype: int64, shape: [batch, {block}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            draft_history_out: {{ dtype: int64, shape: [batch, total_sequence], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
{v2_outputs}{probability_output}    steps:
      - kind: invoke
        component: target_component
        inputs: {{ tokens: request.tokens, recurrent: request.recurrent, token_history: request.history }}
        outputs:
          hidden: target.hidden
          tap_5: target.tap_5
          tap_19: target.tap_19
          tap_33: target.tap_33
          tap_47: target.tap_47
          tap_61: target.tap_61
          combined_features: target.combined_features
          alternate_features: target.alternate_features
          logits: target.logits
          recurrent_prefixes: target.recurrent_prefixes
          token_history_out: target.token_history
      - kind: invoke
        component: proposer_component
        inputs:
          target_features: {feature_binding}
          noise_embeddings: package.noise
          masked_positions: package.masked
          position_ids: package.positions
          attention_mask: package.attention
          output_projection: package.output_projection
          draft_history: request.history
        outputs:
          candidate_tokens: proposal.tokens
          draft_history_out: proposal.draft_history
{v2_bindings}{probability_binding}    state:
      recurrent:
        contract: {{ dtype: float32, shape: [batch, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        scope: invocation
        initializer: request.recurrent
        recurrence: {{ kind: invariant }}
        management: runtime
        release_boundary: invocation
        service_group: recurrent
      token_history:
        contract: {{ dtype: int64, shape: [batch, state_sequence], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        scope: session
        initializer: request.history
        recurrence: {{ kind: invariant }}
        management: runtime
        release_boundary: session
        service_group: token_history
      draft_history:
        contract: {{ dtype: int64, shape: [batch, state_sequence], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        scope: session
        initializer: request.history
        recurrence: {{ kind: invariant }}
        management: runtime
        release_boundary: session
        service_group: draft_history
    serving:
      active: request.active
      done: request.done
      accepted_len: request.accepted_len
      state_service:
        groups:
          recurrent:
            kind: recurrent
            layout: bf
            update: {{ kind: replace }}
            capabilities: {{ rollback_positions: {max_proposal_width}, snapshot: true, fork: true }}
            ports:
              target_component:
                recurrent: {{ input: recurrent, output: recurrent_prefixes }}
          token_history:
            kind: full_attention
            sequence_axis: 1
            layout: batch_sequence
            update: {{ kind: append }}
            capabilities: {{ rollback_positions: {max_proposal_width}, snapshot: true, fork: true, cascade: [draft_history, recurrent] }}
            ports:
              target_component:
                token_history: {{ input: token_history, output: token_history_out }}
          draft_history:
            kind: full_attention
            sequence_axis: 1
            layout: batch_sequence
            update: {{ kind: append }}
            capabilities: {{ rollback_positions: {max_proposal_width}, snapshot: true, fork: true, cascade: [token_history, recurrent] }}
            ports:
              proposer_component:
                draft_history: {{ input: draft_history, output: draft_history_out }}
speculative:
  identity: onnx-genai.speculative
  version: "1"
  proposer: proposer_component
  target: target_component
  proposal_execution:
    kind: dflash_flat_block
    version: "{version}"
    conditioning:
      sources:
{conditioning_sources}
      proposer_input: target_features
      combination: {{ kind: concatenate, axis: 2 }}
    block:
      noise_embeddings_input: noise_embeddings
      masked_positions_input: masked_positions
      position_ids_input: position_ids
      attention_mask_input: attention_mask
      anchor_position: 0
      first_candidate_position: 1
      mask_token_id: 0
    outputs:
      candidate_tokens: candidate_tokens
{probability_port}      verifier_logits: {{ component: target_component, output: logits }}
    shared_weights:
      input_embedding: {{ component: target_component, table: token_embedding }}
      output_projection:
        component: target_component
        initializer: lm_head
        proposer_input: output_projection
        layout: hidden_vocabulary
    accepted_prefix_state:
      recurrent: {{ kind: prefix_snapshots, source: {{ component: target_component, output: recurrent_prefixes }}, axis: 1 }}
      token_history: {{ kind: sequence, source: {{ component: target_component, output: token_history_out }} }}
      draft_history: {{ kind: sequence, source: {{ component: proposer_component, output: draft_history_out }} }}
    draft_private_state: [draft_history]
{structure}
  shared_weights:
    - {{ component: target_component, initializer: token_embedding }}
    - {{ component: target_component, initializer: lm_head }}
  vocabulary: {{ kind: identical }}
  max_proposal_width: {max_proposal_width}
  distribution_preserving: true
  verification:
    target_output: {{ component: target_component, output: logits }}
    accepted_path: {{ kind: runtime, binding: accepted_prefix }}
{verification_probabilities}  rollback_state: [recurrent, token_history, draft_history]
"#
    )
}

fn package(version: &str, probabilities: bool, candidates: &[i64]) -> anyhow::Result<PathBuf> {
    package_with_target(version, probabilities, candidates, TARGET)
}

fn package_with_target(
    version: &str,
    probabilities: bool,
    candidates: &[i64],
    target: &str,
) -> anyhow::Result<PathBuf> {
    package_with_target_scales(
        version,
        probabilities,
        candidates,
        target,
        &[1.0, 2.0, 3.0, 4.0, 4.0],
    )
}

fn fixture_root() -> anyhow::Result<PathBuf> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-fixtures/dflash-flat-block")
        .join(NEXT.fetch_add(1, Ordering::Relaxed).to_string());
    fs::create_dir_all(&root)?;
    Ok(root)
}

fn render_target(target: &str, tap_scales: &[f32; 5]) -> String {
    let lm_head = if target.contains("dflash_contextual_target") {
        [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0]
    } else {
        [0.0; 8]
    };
    render_target_with_weights(
        target,
        tap_scales,
        &[2.0, 0.0, 0.0, 2.0, -2.0, 1.0, 1.0, -2.0],
        &[3.0, 1.0, 1.0, 3.0, -3.0, 2.0, 2.0, -3.0],
        &lm_head,
        &[2.0, 1.0, -1.0, 1.0, 1.0, 2.0, 1.0, -1.0],
    )
}

fn render_target_with_weights(
    target: &str,
    tap_scales: &[f32; 5],
    token_embedding: &[f32; 8],
    token_embedding_decoy: &[f32; 8],
    lm_head: &[f32; 8],
    lm_head_decoy: &[f32; 8],
) -> String {
    let target = [
        "tap_5_scale",
        "tap_19_scale",
        "tap_33_scale",
        "tap_47_scale",
        "tap_61_scale",
    ]
    .into_iter()
    .zip(tap_scales)
    .fold(target.to_string(), |target, (name, scale)| {
        target.replace(&format!("{{{name}}}"), &scale.to_string())
    });
    [
        ("token_embedding_values", token_embedding.as_slice()),
        (
            "token_embedding_decoy_values",
            token_embedding_decoy.as_slice(),
        ),
        ("lm_head_values", lm_head.as_slice()),
        ("lm_head_decoy_values", lm_head_decoy.as_slice()),
    ]
    .into_iter()
    .fold(target, |target, (name, values)| {
        target.replace(
            &format!("{{{name}}}"),
            &values
                .iter()
                .map(|value| format!("float_data: {value}"))
                .collect::<Vec<_>>()
                .join(" "),
        )
    })
}

fn package_with_target_scales(
    version: &str,
    probabilities: bool,
    candidates: &[i64],
    target: &str,
    tap_scales: &[f32; 5],
) -> anyhow::Result<PathBuf> {
    let lm_head = if target.contains("dflash_contextual_target") {
        [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0]
    } else {
        [0.0; 8]
    };
    package_with_target_scales_and_weights(
        version,
        probabilities,
        candidates,
        target,
        tap_scales,
        &[2.0, 0.0, 0.0, 2.0, -2.0, 1.0, 1.0, -2.0],
        &lm_head,
    )
}

#[allow(clippy::too_many_arguments)]
fn package_with_target_scales_and_weights(
    version: &str,
    probabilities: bool,
    candidates: &[i64],
    target: &str,
    tap_scales: &[f32; 5],
    token_embedding: &[f32; 8],
    lm_head: &[f32; 8],
) -> anyhow::Result<PathBuf> {
    assert_eq!(candidates.len(), 8);
    let root = fixture_root()?;
    fs::write(
        root.join("inference_metadata.yaml"),
        metadata(version, probabilities),
    )?;
    fs::write(
        root.join("target.onnx.textproto"),
        render_target_with_weights(
            target,
            tap_scales,
            token_embedding,
            &[3.0, 1.0, 1.0, 3.0, -3.0, 2.0, 2.0, -3.0],
            lm_head,
            &[2.0, 1.0, -1.0, 1.0, 1.0, 2.0, 1.0, -1.0],
        ),
    )?;
    let candidate_values = candidates
        .iter()
        .map(|token| format!("        int64_data: {token}"))
        .collect::<Vec<_>>()
        .join("\n");
    let probability_values = (0..8)
        .flat_map(|_| [1.0_f32, 0.0, 0.0, 0.0])
        .map(|probability| format!("        float_data: {probability:.1}"))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(
        root.join("proposer.onnx.textproto"),
        PROPOSER
            .replace("{candidate_values}", &candidate_values)
            .replace("{probability_values}", &probability_values),
    )?;
    Ok(root)
}

fn run_proposer(
    session: &Session,
    target_features: &[f32],
    noise: &[f32],
    masked: &[u8],
    positions: &[i64],
    attention: &[i64],
    projection: &[f32],
) -> anyhow::Result<(Vec<i64>, f32)> {
    let target_features = Value::from_slice_f32(target_features, &[1, 2, 10])?;
    let noise = Value::from_slice_f32(noise, &[1, 8, 2])?;
    let masked = Value::from_raw_bytes(masked.to_vec(), &[1, 8], DataType::Bool)?;
    let positions = Value::from_slice_i64(positions, &[1, 10])?;
    let attention = Value::from_slice_i64(attention, &[1, 10])?;
    let projection = Value::from_slice_f32(projection, &[2, 4])?;
    let history = Value::from_slice_i64(&[0], &[1, 1])?;
    let outputs = session.run(&[
        ("target_features", &target_features),
        ("noise_embeddings", &noise),
        ("masked_positions", &masked),
        ("position_ids", &positions),
        ("attention_mask", &attention),
        ("output_projection", &projection),
        ("draft_history", &history),
    ])?;
    Ok((
        outputs
            .first()
            .expect("candidate_tokens is the first declared proposer output")
            .to_vec_i64()?,
        outputs
            .get(3)
            .expect("binding_signature is the fourth declared proposer output")
            .to_vec_f32()?[0],
    ))
}

fn alternate_geometry_package() -> anyhow::Result<PathBuf> {
    let root = fixture_root()?;
    fs::write(
        root.join("inference_metadata.yaml"),
        metadata_for_geometry("1", false, FixtureGeometry::Alternate),
    )?;
    fs::write(
        root.join("target.onnx.textproto"),
        render_target(CONTEXTUAL_TARGET, &[2.0, 1.0, 3.0, 4.0, 5.0]),
    )?;
    fs::write(root.join("proposer.onnx.textproto"), ALTERNATE_PROPOSER)?;
    Ok(root)
}

fn pipeline_request() -> PipelineGenerateRequest {
    PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![1, 2]),
        options: GenerateOptions {
            max_new_tokens: 4,
            greedy: true,
            stop_on_eos: false,
            ..GenerateOptions::default()
        },
    })
}

fn expect_error<T>(result: anyhow::Result<T>) -> anyhow::Error {
    match result {
        Ok(_) => panic!("DFlash raw workflow API unexpectedly executed"),
        Err(error) => error,
    }
}

#[test]
fn every_raw_workflow_api_typed_refuses_before_dflash_component_execution() -> anyhow::Result<()> {
    let root = package("1", false, &[0, 0, 0, 0, 0, 0, 0, 0])?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    for (operation, error) in [
        (
            "Engine::run_pipeline",
            expect_error(engine.run_pipeline(pipeline_request())),
        ),
        (
            "Engine::run_pipeline_outputs",
            expect_error(engine.run_pipeline_outputs(pipeline_request())),
        ),
        (
            "Engine::run_pipeline_retained",
            expect_error(engine.run_pipeline_retained(pipeline_request())),
        ),
    ] {
        assert!(matches!(
            package_execution_error(&error),
            Some(PackageExecutionError::DFlashRawWorkflowApi { operation: actual })
                if actual == operation
        ));
    }
    {
        let error = expect_error(engine.prepare_pipeline(pipeline_request()));
        assert!(matches!(
            package_execution_error(&error),
            Some(PackageExecutionError::DFlashRawWorkflowApi { operation })
                if operation == "Engine::prepare_pipeline"
        ));
    }
    {
        let error = expect_error(engine.models());
        assert!(matches!(
            package_execution_error(&error),
            Some(PackageExecutionError::DFlashRawWorkflowApi { operation })
                if operation == "Engine::models"
        ));
    }
    {
        let error = expect_error(engine.prepare_workflow_execution(pipeline_request()));
        assert!(matches!(
            package_execution_error(&error),
            Some(PackageExecutionError::DFlashRawWorkflowApi { operation })
                if operation == "Engine::prepare_workflow_execution"
        ));
    }
    assert!(
        engine.contract_executions().is_empty(),
        "raw API refusal must happen before target or proposer mutation"
    );
    let generated = engine.generate_with_pipeline_request(pipeline_request())?;
    assert_eq!(generated.token_ids.len(), 4);
    Ok(())
}

#[test]
fn consecutive_blocks_commit_anchor_and_never_condition_on_rejected_suffix() -> anyhow::Result<()> {
    let root = package_with_target("1", false, &[1, 1, 1, 1, 1, 1, 1, 1], CONTEXTUAL_TARGET)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let result = engine.generate(pipeline_request().request)?;
    assert_eq!(
        result.token_ids,
        vec![2, 2, 2, 2],
        "each block must commit its target anchor and correction; the rejected token 1 must \
         never become the next block's verifier anchor"
    );
    assert_eq!(
        engine
            .contract_executions()
            .get("onnx-genai.speculative-block")
            .copied(),
        Some(2),
        "zero-acceptance blocks commit anchor+correction, so four tokens require two blocks"
    );
    Ok(())
}

#[test]
fn dflash_batching_uses_isolated_generation_and_fork_declines_before_child_allocation()
-> anyhow::Result<()> {
    let root = package_with_target("1", false, &[1, 1, 1, 1, 1, 1, 1, 1], CONTEXTUAL_TARGET)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let expected = engine.generate(pipeline_request().request)?;
    let batched = engine
        .generate_batched_static(vec![pipeline_request().request, pipeline_request().request])?;
    assert_eq!(batched.len(), 2);
    assert!(
        batched
            .iter()
            .all(|result| result.token_ids == expected.token_ids),
        "unsupported shared DFlash batching must preserve the isolated generation path"
    );

    let source = engine.create_session()?;
    let error = match engine.prepare_session_fork(source, SessionPosition::new(0)) {
        Ok(_) => panic!("DFlash fork must decline before publishing a child"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        SessionForkError::UnsupportedParticipant { participant, reason, .. }
            if participant == "dflash.accepted_prefix"
                && reason.contains("before child allocation")
    ));
    Ok(())
}

#[test]
fn output_callback_failure_is_post_commit_and_retry_is_deterministic() -> anyhow::Result<()> {
    let root = package_with_target("1", false, &[1, 1, 1, 1, 1, 1, 1, 1], CONTEXTUAL_TARGET)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let mut delivered = Vec::new();
    let mut callback = |token: onnx_genai_engine::GenerateToken| -> anyhow::Result<()> {
        delivered.push(token.token_id);
        anyhow::bail!("injected delivery failure")
    };
    let error = engine
        .generate_with_callback(pipeline_request().request, Some(&mut callback))
        .expect_err("callback failure must be reported");
    let message = format!("{error:#}");
    assert!(message.contains("after semantic commit"), "{message}");
    assert!(message.contains("will not be replayed"), "{message}");
    assert_eq!(delivered, vec![2]);
    assert_eq!(
        engine
            .contract_executions()
            .get("onnx-genai.speculative-block")
            .copied(),
        Some(2),
        "all model execution must finish before the first callback"
    );

    let retry = engine.generate(pipeline_request().request)?;
    assert_eq!(retry.token_ids, vec![2, 2, 2, 2]);
    Ok(())
}

#[test]
fn alternate_block_geometry_uses_the_same_structural_dispatch() -> anyhow::Result<()> {
    let root = alternate_geometry_package()?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let diagnostic = engine.dflash_diagnostic().expect("DFlash diagnostic");
    assert_eq!(diagnostic.max_proposal_width, 4);
    assert_eq!(diagnostic.target_hidden_sources.len(), 3);
    let result = engine.generate(pipeline_request().request)?;
    assert_eq!(result.token_ids, vec![2, 2, 2, 2]);
    assert!(
        engine
            .contract_executions()
            .get("onnx-genai.speculative-block")
            .is_some_and(|blocks| *blocks > 0)
    );
    let traces = engine.take_dflash_block_traces();
    assert_eq!(traces[0].conditioning.len(), 3);
    assert!(
        (1..=4).contains(&traces[0].proposer_candidates.len()),
        "the alternate block may crop at EOS but must execute its declared width-5 path"
    );
    Ok(())
}

#[test]
fn proposer_failure_publishes_nothing_and_a_valid_retry_is_deterministic() -> anyhow::Result<()> {
    let invalid = package("1", false, &[1, 1, 1, 1, 1, 1, 1, 1])?;
    let proposer_path = invalid.join("proposer.onnx.textproto");
    let proposer = fs::read_to_string(&proposer_path)?.replacen(
        r#"node { input: "candidate_unbounded" input: "vocabulary" output: "candidate_tokens" op_type: "Mod" }"#,
        r#"node { input: "candidate_unbounded" input: "vocabulary" output: "bounded_candidates" op_type: "Mod" }
  node { output: "invalid_offset" op_type: "Constant" attribute { name: "value" t { data_type: 7 int64_data: 5 } type: TENSOR } }
  node { input: "bounded_candidates" input: "invalid_offset" output: "candidate_tokens" op_type: "Add" }"#,
        1,
    );
    fs::write(proposer_path, proposer)?;
    let mut failed = Engine::from_dir(&invalid, EngineConfig::default())?;
    let error = failed
        .generate(pipeline_request().request)
        .expect_err("out-of-vocabulary proposer output must fail before commit");
    assert!(
        format!("{error:#}").contains("outside shared target vocabulary"),
        "{error:#}"
    );
    assert!(
        failed.contract_executions().is_empty(),
        "a failed proposer block must not record a committed execution"
    );
    assert!(
        failed.take_committed_workflow_publications().is_empty(),
        "a failed proposer block must not expose a pre-commit publication"
    );

    let valid = package_with_target("1", false, &[1, 1, 1, 1, 1, 1, 1, 1], CONTEXTUAL_TARGET)?;
    let mut retry = Engine::from_dir(&valid, EngineConfig::default())?;
    assert_eq!(
        retry.generate(pipeline_request().request)?.token_ids,
        vec![2, 2, 2, 2]
    );
    Ok(())
}

#[test]
fn eos_anchor_commits_once_and_stops_before_any_suffix_is_exposed() -> anyhow::Result<()> {
    let root = package_with_target("1", false, &[1, 1, 1, 1, 1, 1, 1, 1], CONTEXTUAL_TARGET)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let mut request = pipeline_request().request;
    request.options.stop_on_eos = true;
    request.options.eos_token_ids = vec![2];
    let result = engine.generate(request)?;
    assert_eq!(result.token_ids, vec![2]);
    assert_eq!(
        result.finish_reason,
        onnx_genai_engine::FinishReason::EosToken
    );
    Ok(())
}

#[test]
fn engine_dispatches_dflash_v1_to_real_target_and_proposer_sessions() -> anyhow::Result<()> {
    for candidates in [
        &[0, 0, 0, 0, 0, 0, 0, 0][..],
        &[0, 1, 1, 1, 1, 1, 1, 1][..],
        &[1, 1, 1, 1, 1, 1, 1, 1][..],
    ] {
        let root = package("1", false, candidates)?;
        let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
        let result = engine.generate(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![1, 2]),
            options: GenerateOptions {
                max_new_tokens: 4,
                greedy: true,
                stop_on_eos: false,
                ..GenerateOptions::default()
            },
        })?;
        assert_eq!(result.token_ids, vec![0, 0, 0, 0]);
        assert!(
            engine
                .contract_executions()
                .get("onnx-genai.speculative-block")
                .is_some_and(|blocks| *blocks > 0),
            "candidate path {candidates:?} must reach the real target verifier, not a \
             test-provided acceptance decision"
        );
        assert_eq!(
            engine
                .dflash_diagnostic()
                .expect("DFlash diagnostic")
                .version,
            "1"
        );
    }
    Ok(())
}

#[test]
fn sampling_without_declared_proposal_probabilities_refuses_before_a_block_runs()
-> anyhow::Result<()> {
    let root = package("1", false, &[0, 0, 0, 0, 0, 0, 0, 0])?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let error = engine
        .generate(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![1]),
            options: GenerateOptions {
                greedy: false,
                temperature: 1.0,
                seed: Some(7),
                max_new_tokens: 1,
                ..GenerateOptions::default()
            },
        })
        .expect_err("sampling cannot use an undeclared proposal distribution");
    assert!(
        format!("{error:#}").contains("proposal probability distribution"),
        "{error:#}"
    );
    assert!(engine.contract_executions().is_empty());
    Ok(())
}

/// With a one-token output budget the verifier-owned anchor is the complete
/// committed path. The reference samples that target distribution directly,
/// independently of the DFlash implementation.
fn sampling_reference(seed: u64) -> u32 {
    let mut driver_rng = StdRng::seed_from_u64(seed);
    (driver_rng.random::<f32>() * 4.0) as u32
}

#[test]
fn dispatched_dflash_sampling_matches_the_declared_rejection_reference() -> anyhow::Result<()> {
    let root = package("1", true, &[0, 0, 0, 0, 0, 0, 0, 0])?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let mut observed = [0_usize; 4];
    for seed in 0..128_u64 {
        let result = engine.generate(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![1]),
            options: GenerateOptions {
                greedy: false,
                temperature: 1.0,
                seed: Some(seed),
                max_new_tokens: 1,
                stop_on_eos: false,
                ..GenerateOptions::default()
            },
        })?;
        assert_eq!(result.token_ids.len(), 1);
        assert_eq!(
            result.token_ids[0],
            sampling_reference(seed),
            "seed {seed} did not use the target/proposal rejection equation"
        );
        observed[result.token_ids[0] as usize] += 1;
    }
    assert!(
        observed.iter().all(|count| *count > 0),
        "the dispatched sampling path must retain all target outcomes: {observed:?}"
    );
    Ok(())
}

#[test]
fn context_exhaustion_finishes_before_the_first_dflash_component_run() -> anyhow::Result<()> {
    let root = package("1", false, &[0, 0, 0, 0, 0, 0, 0, 0])?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let result = engine.generate(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![1, 2]),
        options: GenerateOptions {
            max_new_tokens: 4,
            max_context: Some(2),
            greedy: true,
            ..GenerateOptions::default()
        },
    })?;
    assert!(result.token_ids.is_empty());
    assert_eq!(
        result.finish_reason,
        onnx_genai_engine::FinishReason::Length
    );
    assert!(
        engine.contract_executions().is_empty(),
        "context exhaustion must not begin a DFlash block"
    );
    Ok(())
}

fn controlled_request() -> GenerateRequest {
    GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![1, 2]),
        options: GenerateOptions {
            max_new_tokens: 4,
            greedy: false,
            temperature: 1.0,
            seed: Some(37),
            stop_on_eos: false,
            ..GenerateOptions::default()
        },
    }
}

fn cancellation_at(boundary: GenerationBoundary) -> GenerationControl {
    let control = GenerationControl::new();
    let cancellation = control.clone();
    control.with_checkpoint(move |observed| {
        if observed == boundary {
            cancellation.cancel();
        }
        Ok(())
    })
}

fn fault_at(boundary: GenerationBoundary) -> GenerationControl {
    let fired = AtomicUsize::new(0);
    GenerationControl::new().with_checkpoint(move |observed| {
        if observed == boundary && fired.fetch_add(1, Ordering::AcqRel) == 0 {
            anyhow::bail!("injected {boundary} failure");
        }
        Ok(())
    })
}

fn clean_second_turn(root: &Path, request: GenerateRequest) -> anyhow::Result<Vec<u32>> {
    let mut engine = Engine::from_dir(root, EngineConfig::default())?;
    let session = engine.create_session()?;
    engine.generate_in_session(session, request.clone())?;
    Ok(engine.generate_in_session(session, request)?.token_ids)
}

fn assert_cancelled_retry(boundary: GenerationBoundary) -> anyhow::Result<()> {
    let root = package("1", true, &[0, 0, 0, 0, 0, 0, 0, 0])?;
    let request = controlled_request();
    let expected = clean_second_turn(&root, request.clone())?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let session = engine.create_session()?;
    engine.generate_in_session(session, request.clone())?;
    let baseline_tokens = engine.session_token_count(session)?;
    let baseline_blocks = engine
        .contract_executions()
        .get("onnx-genai.speculative-block")
        .copied();
    engine.take_dflash_block_traces();
    let mut delivered = Vec::new();
    let mut callback = |token: onnx_genai_engine::GenerateToken| -> anyhow::Result<()> {
        delivered.push(token.token_id);
        Ok(())
    };
    let error = engine
        .generate_in_session_with_control_callbacks(
            session,
            request.clone(),
            cancellation_at(boundary),
            None,
            Some(&mut callback),
        )
        .expect_err("controlled DFlash generation must cancel");
    let cancelled = error
        .downcast_ref::<DFlashGenerationCancelled>()
        .expect("typed DFlash cancellation");
    assert_eq!(cancelled.boundary, boundary);
    assert!(matches!(
        cancelled.outcome,
        onnx_genai_engine::pipeline::TurnTransactionOutcome::AbortToBaseline {
            reason: onnx_genai_engine::pipeline::TurnAbortReason::Cancellation,
            ..
        }
    ));
    assert!(delivered.is_empty(), "cancelled output must remain staged");
    assert_eq!(engine.session_token_count(session)?, baseline_tokens);
    assert_eq!(
        engine
            .contract_executions()
            .get("onnx-genai.speculative-block")
            .copied(),
        baseline_blocks
    );
    assert!(engine.take_committed_workflow_publications().is_empty());
    assert!(engine.take_dflash_block_traces().is_empty());

    let retry = engine.generate_in_session(session, request)?;
    assert_eq!(retry.token_ids, expected);
    assert_eq!(
        engine.session_token_count(session)?,
        baseline_tokens + expected.len()
    );
    Ok(())
}

#[test]
fn cancellation_after_verifier_restores_same_session_and_retries_deterministically()
-> anyhow::Result<()> {
    assert_cancelled_retry(GenerationBoundary::AfterVerifier)
}

#[test]
fn cancellation_at_semantic_precommit_restores_same_session_and_retries_deterministically()
-> anyhow::Result<()> {
    assert_cancelled_retry(GenerationBoundary::BeforeSemanticCommit)
}

#[test]
fn cancellation_after_commit_is_delivery_only_and_cannot_roll_back() -> anyhow::Result<()> {
    let root = package_with_target("1", false, &[1, 1, 1, 1, 1, 1, 1, 1], CONTEXTUAL_TARGET)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let control = GenerationControl::new();
    let cancellation = control.clone();
    let won = Arc::new(AtomicBool::new(true));
    let observed = won.clone();
    let control = control.with_checkpoint(move |boundary| {
        if boundary == GenerationBoundary::BeforeOutputPublication {
            observed.store(cancellation.cancel(), Ordering::Release);
        }
        Ok(())
    });
    let mut delivered = Vec::new();
    let mut callback = |token: onnx_genai_engine::GenerateToken| -> anyhow::Result<()> {
        delivered.push(token.token_id);
        Ok(())
    };
    let result = engine.generate_with_control_callbacks(
        pipeline_request().request,
        control,
        None,
        Some(&mut callback),
    )?;
    assert!(
        !won.load(Ordering::Acquire),
        "semantic commit must win before output publication"
    );
    assert_eq!(delivered, result.token_ids);
    assert!(!engine.take_dflash_block_traces().is_empty());
    Ok(())
}

fn assert_fault_retry(boundary: GenerationBoundary) -> anyhow::Result<()> {
    let root = package("1", true, &[0, 0, 0, 0, 0, 0, 0, 0])?;
    let request = controlled_request();
    let expected = clean_second_turn(&root, request.clone())?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let session = engine.create_session()?;
    engine.generate_in_session(session, request.clone())?;
    let baseline_tokens = engine.session_token_count(session)?;
    let baseline_blocks = engine
        .contract_executions()
        .get("onnx-genai.speculative-block")
        .copied();
    engine.take_dflash_block_traces();
    let mut delivered = Vec::new();
    let mut callback = |token: onnx_genai_engine::GenerateToken| -> anyhow::Result<()> {
        delivered.push(token.token_id);
        Ok(())
    };
    let error = engine
        .generate_in_session_with_control_callbacks(
            session,
            request.clone(),
            fault_at(boundary),
            None,
            Some(&mut callback),
        )
        .expect_err("controlled checkpoint must fail");
    assert!(
        format!("{error:#}").contains(&format!("injected {boundary} failure")),
        "{error:#}"
    );
    assert!(delivered.is_empty());
    assert_eq!(engine.session_token_count(session)?, baseline_tokens);
    assert_eq!(
        engine
            .contract_executions()
            .get("onnx-genai.speculative-block")
            .copied(),
        baseline_blocks
    );
    assert!(engine.take_committed_workflow_publications().is_empty());
    assert!(engine.take_dflash_block_traces().is_empty());

    let retry = engine.generate_in_session(session, request)?;
    assert_eq!(retry.token_ids, expected);
    Ok(())
}

#[test]
fn verifier_output_failure_restores_same_session_rng_and_histories() -> anyhow::Result<()> {
    assert_fault_retry(GenerationBoundary::AfterVerifier)
}

#[test]
fn semantic_precommit_failure_restores_same_session_rng_and_histories() -> anyhow::Result<()> {
    assert_fault_retry(GenerationBoundary::BeforeSemanticCommit)
}

#[test]
fn qwen_geometry_binds_five_real_taps_and_each_changes_proposer_output() -> anyhow::Result<()> {
    let candidates = [0, 0, 0, 0, 0, 0, 0, 0];
    let base_scales = [1.0, 2.0, 3.0, 4.0, 4.0];
    let root =
        package_with_target_scales("1", false, &candidates, CONTEXTUAL_TARGET, &base_scales)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    assert_eq!(
        engine
            .dflash_diagnostic()
            .expect("DFlash diagnostic")
            .target_hidden_sources,
        [
            "target_component::tap_5",
            "target_component::tap_19",
            "target_component::tap_33",
            "target_component::tap_47",
            "target_component::tap_61"
        ]
    );
    engine.generate(pipeline_request().request)?;
    let base_trace = engine.take_dflash_block_traces();
    let base = base_trace.first().expect("committed DFlash block");
    assert_eq!(base.conditioning.len(), 5);
    assert!(
        base.conditioning
            .iter()
            .all(|source| source.shape == [1, 2, 2])
    );

    for tap in 0..base_scales.len() {
        let mut scales = base_scales;
        scales[tap] += 1.0;
        let root = package_with_target_scales("1", false, &candidates, CONTEXTUAL_TARGET, &scales)?;
        let mut changed = Engine::from_dir(&root, EngineConfig::default())?;
        changed.generate(pipeline_request().request)?;
        let changed_trace = changed.take_dflash_block_traces();
        assert_ne!(
            changed_trace[0].proposer_candidates, base.proposer_candidates,
            "tap {} was ignored by actual proposer execution",
            base.conditioning[tap].source
        );
    }
    Ok(())
}

#[test]
fn every_required_block_input_independently_changes_the_proposer_output() -> anyhow::Result<()> {
    let root = package("1", false, &[0, 0, 0, 0, 0, 0, 0, 0])?;
    let environment = Environment::new("dflash-input-sensitivity")?;
    let session = Session::new(
        &environment,
        &root.join("proposer.onnx.textproto"),
        SessionOptions::default(),
    )?;
    let base_features = vec![0.0; 20];
    let base_noise = vec![0.0; 16];
    let base_mask = vec![0; 8];
    let base_positions = vec![0; 10];
    let base_attention = vec![0; 10];
    let base_projection = vec![0.0; 8];
    let base = run_proposer(
        &session,
        &base_features,
        &base_noise,
        &base_mask,
        &base_positions,
        &base_attention,
        &base_projection,
    )?;
    assert_eq!(base.1, 0.0);

    for (tap, expected_signature) in [1.0, 3.0, 5.0, 7.0, 11.0].into_iter().enumerate() {
        let mut changed = base_features.clone();
        changed[tap * 2] = 1.0;
        let actual = run_proposer(
            &session,
            &changed,
            &base_noise,
            &base_mask,
            &base_positions,
            &base_attention,
            &base_projection,
        )?;
        assert_eq!(
            actual.1, expected_signature,
            "hidden tap {tap} is disconnected or swapped"
        );
    }
    let mut noise = base_noise.clone();
    noise[0] = 1.0;
    assert_eq!(
        run_proposer(
            &session,
            &base_features,
            &noise,
            &base_mask,
            &base_positions,
            &base_attention,
            &base_projection,
        )?
        .1,
        13.0,
        "noise_embeddings is ignored"
    );
    let mut mask = base_mask.clone();
    mask[0] = 1;
    assert_eq!(
        run_proposer(
            &session,
            &base_features,
            &base_noise,
            &mask,
            &base_positions,
            &base_attention,
            &base_projection,
        )?
        .1,
        17.0,
        "masked_positions is ignored"
    );
    let mut positions = base_positions.clone();
    positions[0] = 1;
    assert_eq!(
        run_proposer(
            &session,
            &base_features,
            &base_noise,
            &base_mask,
            &positions,
            &base_attention,
            &base_projection,
        )?
        .1,
        19.0,
        "position_ids is ignored"
    );
    let mut attention = base_attention.clone();
    attention[0] = 1;
    assert_eq!(
        run_proposer(
            &session,
            &base_features,
            &base_noise,
            &base_mask,
            &base_positions,
            &attention,
            &base_projection,
        )?
        .1,
        23.0,
        "attention_mask is ignored"
    );
    let mut projection = base_projection.clone();
    projection[0] = 1.0;
    assert_eq!(
        run_proposer(
            &session,
            &base_features,
            &base_noise,
            &base_mask,
            &base_positions,
            &base_attention,
            &projection,
        )?
        .1,
        29.0,
        "output_projection is ignored"
    );
    Ok(())
}

fn first_proposer_candidates(root: &Path) -> anyhow::Result<Vec<i64>> {
    let mut engine = Engine::from_dir(root, EngineConfig::default())?;
    engine.generate(pipeline_request().request)?;
    Ok(engine
        .take_dflash_block_traces()
        .into_iter()
        .next()
        .expect("one committed DFlash block")
        .proposer_candidates)
}

#[test]
fn runtime_consumes_exact_shared_embedding_and_lm_head_identity() -> anyhow::Result<()> {
    let candidates = [0, 0, 0, 0, 0, 0, 0, 0];
    let embedding = [2.0, 0.0, 0.0, 2.0, -2.0, 1.0, 1.0, -2.0];
    let lm_head = [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0];
    let base_root = package_with_target_scales_and_weights(
        "1",
        false,
        &candidates,
        CONTEXTUAL_TARGET,
        &[1.0, 2.0, 3.0, 4.0, 4.0],
        &embedding,
        &lm_head,
    )?;
    let base = first_proposer_candidates(&base_root)?;

    let mut changed_embedding = embedding;
    changed_embedding[4] += 1.0;
    let root = package_with_target_scales_and_weights(
        "1",
        false,
        &candidates,
        CONTEXTUAL_TARGET,
        &[1.0, 2.0, 3.0, 4.0, 4.0],
        &changed_embedding,
        &lm_head,
    )?;
    assert_ne!(
        first_proposer_candidates(&root)?,
        base,
        "runtime did not gather the declared target token_embedding"
    );

    let mut changed_lm_head = lm_head;
    changed_lm_head[0] += 1.0;
    let root = package_with_target_scales_and_weights(
        "1",
        false,
        &candidates,
        CONTEXTUAL_TARGET,
        &[1.0, 2.0, 3.0, 4.0, 4.0],
        &embedding,
        &changed_lm_head,
    )?;
    assert_ne!(
        first_proposer_candidates(&root)?,
        base,
        "runtime did not bind the declared target lm_head as output_projection"
    );

    for (declared, decoy) in [
        ("table: token_embedding", "table: token_embedding_decoy"),
        (
            "initializer: lm_head\n        proposer_input",
            "initializer: lm_head_decoy\n        proposer_input",
        ),
    ] {
        let root = package_with_target("1", false, &candidates, CONTEXTUAL_TARGET)?;
        let metadata_path = root.join("inference_metadata.yaml");
        let metadata = fs::read_to_string(&metadata_path)?.replace(declared, decoy);
        fs::write(metadata_path, metadata)?;
        let error = match Engine::from_dir(&root, EngineConfig::default()) {
            Ok(_) => panic!("compatible-shape shared-weight misbinding must fail admission"),
            Err(error) => error,
        };
        assert!(
            format!("{error:#}").contains("shared_weights"),
            "compatible-shape misbinding '{declared}' -> '{decoy}' was not rejected: {error:#}"
        );
    }
    Ok(())
}

#[test]
fn disconnected_required_dflash_input_fails_package_admission() -> anyhow::Result<()> {
    let root = package("1", false, &[0, 0, 0, 0, 0, 0, 0, 0])?;
    let proposer_path = root.join("proposer.onnx.textproto");
    let proposer = fs::read_to_string(&proposer_path)?.replace(
        r#"node { input: "attention_f32" input: "history_axis" output: "attention_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }"#,
        r#"node { input: "positions_f32" input: "history_axis" output: "attention_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }"#,
    );
    fs::write(proposer_path, proposer)?;
    let error = match Engine::from_dir(&root, EngineConfig::default()) {
        Ok(_) => panic!("disconnected attention_mask must fail before execution"),
        Err(error) => error,
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("block.attention_mask_input")
            && message.contains("attention_mask")
            && message.contains("does not connect")
            && message.contains("candidate output"),
        "{message}"
    );

    for (initializer, replacement, expected) in [
        (
            r#"input: "token_embedding" input: "tokens""#,
            r#"input: "token_embedding_decoy" input: "tokens""#,
            "input_embedding.table",
        ),
        (
            r#"input: "hidden" input: "lm_head""#,
            r#"input: "hidden" input: "lm_head_decoy""#,
            "output_projection.initializer",
        ),
    ] {
        let root = package("1", false, &[0, 0, 0, 0, 0, 0, 0, 0])?;
        let target_path = root.join("target.onnx.textproto");
        let target = fs::read_to_string(&target_path)?.replace(initializer, replacement);
        fs::write(target_path, target)?;
        let error = match Engine::from_dir(&root, EngineConfig::default()) {
            Ok(_) => panic!("disconnected shared initializer must fail admission"),
            Err(error) => error,
        };
        assert!(
            format!("{error:#}").contains(expected),
            "disconnected {initializer} did not report {expected}: {error:#}"
        );
    }
    Ok(())
}

#[test]
fn selector_convolution_v2_stays_a_typed_pre_mutation_refusal() -> anyhow::Result<()> {
    let root = package("2", false, &[0, 0, 0, 0, 0, 0, 0, 0])?;
    let error = match Engine::from_dir(&root, EngineConfig::default()) {
        Ok(_) => panic!("v2 selector/convolution semantics are not implemented"),
        Err(error) => error,
    };
    assert!(matches!(
        package_execution_error(&error),
        Some(PackageExecutionError::DFlashExecutionUnavailable { version, .. }) if version == "2"
    ));
    Ok(())
}
