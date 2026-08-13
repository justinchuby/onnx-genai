use std::fs;
use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};
use onnx_genai_ort::{DataType, Value};

fn package(name: &str, metadata: &str, models: &[(&str, &str)]) -> anyhow::Result<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-fixtures/workflow-policy")
        .join(name);
    fs::create_dir_all(&root)?;
    fs::write(root.join("inference_metadata.yaml"), metadata)?;
    for (name, model) in models {
        fs::write(root.join(name), model)?;
    }
    Ok(root)
}

const GREEDY: &str = r#"
ir_version: 8
graph {
  node {
    input: "logits"
    output: "token_ids"
    op_type: "ArgMax"
    attribute { name: "axis" i: -1 type: 2 }
    attribute { name: "keepdims" i: 0 type: 2 }
  }
  name: "greedy_sampler"
  input {
    name: "logits"
    type { tensor_type { elem_type: 1 shape {
      dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
    }}}
  }
  input { name: "temperature" type { tensor_type { elem_type: 1 shape {
    dim { dim_value: 1 }
  }}}}
  input { name: "top_k" type { tensor_type { elem_type: 7 shape {
    dim { dim_value: 1 }
  }}}}
  input { name: "top_p" type { tensor_type { elem_type: 1 shape {
    dim { dim_value: 1 }
  }}}}
  input { name: "grammar_mask" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
  output {
    name: "token_ids"
    type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } }}}
  }
}
opset_import { domain: "" version: 12 }
"#;

const ARGMIN_SAMPLER: &str = r#"
ir_version: 8
graph {
  node {
    input: "scores"
    output: "tokens"
    op_type: "ArgMin"
    attribute { name: "axis" i: -1 type: 2 }
    attribute { name: "keepdims" i: 0 type: 2 }
  }
  name: "application_sampler"
  input { name: "scores" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
  input { name: "temp" type { tensor_type { elem_type: 1 shape {
    dim { dim_value: 1 }
  }}}}
  input { name: "k" type { tensor_type { elem_type: 7 shape {
    dim { dim_value: 1 }
  }}}}
  input { name: "p" type { tensor_type { elem_type: 1 shape {
    dim { dim_value: 1 }
  }}}}
  input { name: "mask" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
  output { name: "tokens" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
}
opset_import { domain: "" version: 12 }
"#;

const SIMPLE_GREEDY: &str = r#"
ir_version: 8
graph {
  node {
    input: "logits" output: "token" op_type: "ArgMax"
    attribute { name: "axis" i: -1 type: INT }
    attribute { name: "keepdims" i: 0 type: INT }
  }
  name: "simple_greedy"
  input { name: "logits" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
  output { name: "token" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const EXTERNAL_BIAS: &str = r#"
ir_version: 8
graph {
  node { input: "logits" input: "bias" output: "adjusted" op_type: "Add" }
  name: "external_bias"
  initializer {
    dims: 4
    data_type: 1
    name: "bias"
    external_data { key: "location" value: "bias.bin" }
    external_data { key: "offset" value: "0" }
    external_data { key: "length" value: "16" }
    data_location: EXTERNAL
  }
  input { name: "logits" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_value: 4 }
  }}}}
  output { name: "adjusted" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_value: 4 }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const ADD_PAIR: &str = r#"
ir_version: 8
graph {
  node { input: "left" input: "right" output: "sum" op_type: "Add" }
  name: "add_pair"
  input { name: "left" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_value: 4 }
  }}}}
  input { name: "right" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_value: 4 }
  }}}}
  output { name: "sum" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_value: 4 }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const CACHE_DECODER: &str = r#"
ir_version: 8
graph {
  node {
    input: "past_key_values" input: "token_state"
    output: "present_key_values" op_type: "Concat"
    attribute { name: "axis" i: 1 type: INT }
  }
  name: "cache_decoder"
  input { name: "past_key_values" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "past_sequence" }
  }}}}
  input { name: "token_state" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_value: 1 }
  }}}}
  output { name: "present_key_values" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "present_sequence" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const MIN_P_FILTER: &str = r#"
ir_version: 8
graph {
  node {
    input: "logits" output: "max_logit" op_type: "ReduceMax"
    attribute { name: "axes" ints: 1 type: INTS }
    attribute { name: "keepdims" i: 1 type: INT }
  }
  node { input: "min_p" output: "log_min_p" op_type: "Log" }
  node { input: "max_logit" input: "log_min_p" output: "threshold" op_type: "Add" }
  node {
    input: "logits" input: "threshold" output: "keep" op_type: "GreaterOrEqual"
  }
  node {
    input: "keep" input: "logits" input: "negative"
    output: "filtered_logits" op_type: "Where"
  }
  name: "min_p_filter"
  initializer { data_type: 1 float_data: -1000000000 name: "negative" }
  input { name: "logits" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
  input { name: "min_p" type { tensor_type { elem_type: 1 shape {
    dim { dim_value: 1 }
  }}}}
  output { name: "filtered_logits" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const EOS_PREDICATE: &str = r#"
ir_version: 8
graph {
  node { input: "token" input: "eos" output: "done" op_type: "Equal" }
  name: "eos_predicate"
  input { name: "token" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "eos" type { tensor_type { elem_type: 7 shape {
    dim { dim_value: 1 }
  }}}}
  output { name: "done" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const EULER: &str = r#"
ir_version: 8
graph {
  node { input: "sample" input: "derivative" output: "next_state" op_type: "Sub" }
  name: "euler"
  input { name: "sample" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "width" }
  }}}}
  input { name: "derivative" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "width" }
  }}}}
  input { name: "step" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "schedule" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "schedule_length" }
  }}}}
  output { name: "next_state" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "width" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const MASKED_UPDATE: &str = r#"
ir_version: 8
graph {
  node { input: "masked" input: "proposed_tokens" input: "current_tokens"
         output: "next_state" op_type: "Where" }
  node { input: "masked" output: "next_mask" op_type: "Identity" }
  name: "masked_update"
  input { name: "current_tokens" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_param: "sequence" }
  }}}}
  input { name: "proposed_tokens" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_param: "sequence" }
  }}}}
  input { name: "masked" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" } dim { dim_param: "sequence" }
  }}}}
  input { name: "step" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  output { name: "next_state" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_param: "sequence" }
  }}}}
  output { name: "next_mask" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" } dim { dim_param: "sequence" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const SPECULATIVE_ACCEPTANCE: &str = r#"
ir_version: 8
graph {
  node { input: "target_scores" output: "target_tokens" op_type: "ArgMax"
         attribute { name: "axis" i: -1 type: 2 }
         attribute { name: "keepdims" i: 0 type: 2 } }
  node { input: "target_tokens" input: "proposed_tokens" output: "accepted" op_type: "Equal" }
  node { input: "accepted" output: "accepted_i64" op_type: "Cast"
         attribute { name: "to" i: 7 type: 2 } }
  node { input: "accepted_i64" input: "axes" output: "accepted_count" op_type: "ReduceSum"
         attribute { name: "keepdims" i: 0 type: 2 } }
  node { input: "proposed_tokens" output: "accepted_tokens" op_type: "Identity" }
  node { input: "accepted_count" input: "zero" output: "done" op_type: "Greater" }
  name: "speculative_acceptance"
  initializer { dims: 1 data_type: 7 name: "zero"
                raw_data: "\000\000\000\000\000\000\000\000" }
  initializer { dims: 1 data_type: 7 int64_data: -1 name: "axes" }
  input { name: "target_scores" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "draft_sequence" }
    dim { dim_param: "vocabulary" }
  }}}}
  input { name: "proposed_tokens" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_param: "draft_sequence" }
  }}}}
  output { name: "accepted_tokens" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_param: "draft_sequence" }
  }}}}
  output { name: "accepted_count" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  output { name: "done" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const ADD_STATE: &str = r#"
ir_version: 8
graph {
  node { input: "current" input: "update" output: "next" op_type: "Add" }
  name: "add_state"
  input { name: "current" type { tensor_type { elem_type: 7 shape {} }}}
  input { name: "update" type { tensor_type { elem_type: 7 shape {} }}}
  output { name: "next" type { tensor_type { elem_type: 7 shape {} }}}
}
opset_import { domain: "" version: 13 }
"#;

const LESS: &str = r#"
ir_version: 8
graph {
  node { input: "value" input: "limit" output: "continue" op_type: "Less" }
  name: "less"
  input { name: "value" type { tensor_type { elem_type: 7 shape {} }}}
  input { name: "limit" type { tensor_type { elem_type: 7 shape {} }}}
  output { name: "continue" type { tensor_type { elem_type: 9 shape {} }}}
}
opset_import { domain: "" version: 13 }
"#;

const SEEDED_SAMPLER: &str = r#"
ir_version: 8
graph {
  node {
    input: "logits" output: "token_ids" op_type: "ArgMax"
    attribute { name: "axis" i: -1 type: 2 }
    attribute { name: "keepdims" i: 0 type: 2 }
  }
  node { input: "offset" input: "one" output: "next_offset" op_type: "Add" }
  name: "seeded_sampler"
  initializer { dims: 1 data_type: 7 name: "one"
                raw_data: "\001\000\000\000\000\000\000\000" }
  input { name: "logits" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
  input { name: "seed" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "offset" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  output { name: "token_ids" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  output { name: "next_offset" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const EOS: &str = r#"
ir_version: 8
graph {
  node { input: "token_ids" input: "eos_token_ids" output: "terminated" op_type: "Equal" }
  name: "eos"
  input { name: "token_ids" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "eos_token_ids" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "iteration" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "max_iterations" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  output { name: "terminated" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const NOT: &str = r#"
ir_version: 8
graph {
  node { input: "done" output: "continue" op_type: "Not" }
  name: "not"
  input { name: "done" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" }
  }}}}
  output { name: "continue" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const INT64_MATRIX_IDENTITY: &str = r#"
ir_version: 8
graph {
  node { input: "input" output: "output" op_type: "Identity" }
  name: "int64_matrix_identity"
  input { name: "input" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_param: "sequence" }
  }}}}
  output { name: "output" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_param: "sequence" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const INT64_MATRIX_ADD: &str = r#"
ir_version: 8
graph {
  node { input: "left" input: "right" output: "output" op_type: "Add" }
  name: "int64_matrix_add"
  input { name: "left" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_param: "sequence" }
  }}}}
  input { name: "right" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_param: "sequence" }
  }}}}
  output { name: "output" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_param: "sequence" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const INT64_PREFIX: &str = r#"
ir_version: 8
graph {
  node {
    input: "state" input: "starts" input: "valid_length" input: "axes" input: "steps"
    output: "selected" op_type: "Slice"
  }
  name: "int64_prefix"
  initializer { dims: 1 data_type: 7 int64_data: 0 name: "starts" }
  initializer { dims: 1 data_type: 7 int64_data: 1 name: "axes" }
  initializer { dims: 1 data_type: 7 int64_data: 1 name: "steps" }
  input { name: "state" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_param: "sequence" }
  }}}}
  input { name: "valid_length" type { tensor_type { elem_type: 7 shape {
    dim { dim_value: 1 }
  }}}}
  output { name: "selected" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_param: "selected" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const MIN_LENGTH: &str = r#"
ir_version: 8
graph {
  node { input: "accepted" input: "grammar" output: "length" op_type: "Min" }
  name: "min_length"
  input { name: "accepted" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "grammar" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  output { name: "length" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const ADAPTIVE_PROPOSAL_BUDGET: &str = r#"
ir_version: 8
graph {
  node { input: "current_k" input: "one" output: "increased" op_type: "Add" }
  node { input: "draft_ms" input: "target_ms" output: "target_slower" op_type: "Less" }
  node {
    input: "filled_proposal_budget" input: "target_slower"
    output: "should_increase" op_type: "And"
  }
  node {
    input: "should_increase" input: "increased" input: "current_k"
    output: "next_k" op_type: "Where"
  }
  node { input: "estimates" output: "next_estimates" op_type: "Identity" }
  name: "adaptive_proposal_budget"
  initializer { dims: 1 data_type: 7 int64_data: 1 name: "one" }
  input { name: "current_k" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "accepted" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "evaluated" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "committed_tokens" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "filled_proposal_budget" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "draft_ms" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "target_ms" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "estimates" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "budget_slots" }
  }}}}
  output { name: "next_k" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  output { name: "next_estimates" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "budget_slots" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const GUIDED_SAMPLER: &str = r#"
ir_version: 8
graph {
  node {
    input: "logits_mask" input: "logits" input: "negative"
    output: "masked_logits" op_type: "Where"
  }
  node { input: "masked_logits" output: "sampled" op_type: "ArgMax"
    attribute { name: "axis" i: 1 type: INT }
    attribute { name: "keepdims" i: 1 type: INT }
  }
  node { input: "forced_length" input: "zero" output: "has_forced" op_type: "Greater" }
  node { input: "has_forced" input: "axes" output: "has_forced_2d" op_type: "Unsqueeze" }
  node {
    input: "has_forced_2d" input: "forced_tokens" input: "sampled"
    output: "token" op_type: "Where"
  }
  name: "guided_sampler"
  initializer { data_type: 1 float_data: -1000000000 name: "negative" }
  initializer { dims: 1 data_type: 7 int64_data: 1 name: "axes" }
  initializer { dims: 1 data_type: 7 int64_data: 0 name: "zero" }
  input { name: "logits" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
  input { name: "logits_mask" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
  input { name: "forced_tokens" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_value: 1 }
  }}}}
  input { name: "forced_length" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  output { name: "token" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_value: 1 }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const VISION_IDENTITY: &str = r#"
ir_version: 8
graph {
  node { input: "pixel_values" output: "image_features" op_type: "Identity" }
  name: "vision_encoder"
  input { name: "pixel_values" type { tensor_type { elem_type: 1 shape {
    dim { dim_value: 1 } dim { dim_value: 3 } dim { dim_value: 2 } dim { dim_value: 2 }
  }}}}
  input { name: "grid" type { tensor_type { elem_type: 7 shape {
    dim { dim_value: 1 } dim { dim_value: 2 }
  }}}}
  output { name: "image_features" type { tensor_type { elem_type: 1 shape {
    dim { dim_value: 1 } dim { dim_value: 3 } dim { dim_value: 2 } dim { dim_value: 2 }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const FEATURE_IDENTITY: &str = r#"
ir_version: 8
graph {
  node { input: "input" output: "output" op_type: "Identity" }
  name: "feature_identity"
  input { name: "input" type { tensor_type { elem_type: 1 shape {
    dim { dim_value: 1 } dim { dim_value: 3 } dim { dim_value: 2 } dim { dim_value: 2 }
  }}}}
  output { name: "output" type { tensor_type { elem_type: 1 shape {
    dim { dim_value: 1 } dim { dim_value: 3 } dim { dim_value: 2 } dim { dim_value: 2 }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

#[test]
fn workflow_materializes_image_adapter_outputs_as_ssa_values() -> anyhow::Result<()> {
    let metadata = r#"
preprocessing:
  image:
    transforms:
      - { op: decode, outputs: [decoded] }
      - { op: convert_rgb, inputs: [decoded], outputs: [rgb] }
      - { op: resize, inputs: [rgb], outputs: [pixels], size: 2, mode: stretch,
          interpolation: bilinear }
      - { op: emit_original_size, inputs: [rgb], outputs: [grid] }
    outputs:
      - source: pixels
        name: image.pixel_values
        content: pixels
        dtype: float32
        contract: { dtype: float32, rank: 4, shape: [batch, 3, 2, 2] }
      - source: grid
        name: image.grid
        content: original_size
        dtype: int64
        contract: { dtype: int64, rank: 2, shape: [1, 2] }
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: { onnx-genai.image-preprocess: "1" }
      custom_op_versions: {}
      capabilities: [workflow_ssa, typed_emit]
    inputs:
      request.batch_anchor:
        contract: { dtype: int64, rank: 2, shape: [batch, sequence] }
        role: { kind: opaque }
        source: { kind: application, name: batch_anchor }
        required: true
      request.image:
        contract: { dtype: uint8, rank: 1, shape: [encoded_bytes] }
        role: { kind: opaque }
        source: { kind: application, name: image }
        required: true
    outputs:
      result:
        contract: { dtype: float32, rank: 4, shape: [batch, 3, 2, 2] }
        role: image
        stage: post_adapter
    components:
      preprocess:
        implementation: { kind: adapter, abi: onnx-genai.image-preprocess, version: "1" }
        ports:
          inputs:
            encoded: { dtype: uint8, rank: 1, shape: [encoded_bytes] }
          outputs:
            pixel_values: { dtype: float32, rank: 4, shape: [batch, 3, 2, 2] }
            grid: { dtype: int64, rank: 2, shape: [1, 2] }
        effects: []
      vision:
        implementation: { kind: onnx, artifact: vision.onnx.textproto }
        ports:
          inputs:
            pixel_values: { dtype: float32, rank: 4, shape: [batch, 3, 2, 2] }
            grid: { dtype: int64, rank: 2, shape: [1, 2] }
          outputs:
            image_features: { dtype: float32, rank: 4, shape: [batch, 3, 2, 2] }
      embedding:
        implementation: { kind: onnx, artifact: embedding.onnx.textproto }
        ports:
          inputs:
            input: { dtype: float32, rank: 4, shape: [batch, 3, 2, 2] }
          outputs:
            output: { dtype: float32, rank: 4, shape: [batch, 3, 2, 2] }
      decoder:
        implementation: { kind: onnx, artifact: decoder.onnx.textproto }
        ports:
          inputs:
            input: { dtype: float32, rank: 4, shape: [batch, 3, 2, 2] }
          outputs:
            output: { dtype: float32, rank: 4, shape: [batch, 3, 2, 2] }
    steps:
        - kind: invoke
          component: preprocess
          inputs: { encoded: request.image }
          outputs: { pixel_values: image.pixel_values, grid: image.grid }
        - kind: invoke
          component: vision
          inputs: { pixel_values: image.pixel_values, grid: image.grid }
          outputs: { image_features: vision.features }
        - kind: invoke
          component: embedding
          inputs: { input: vision.features }
          outputs: { output: embedding.output }
        - kind: invoke
          component: decoder
          inputs: { input: embedding.output }
          outputs: { output: decoder.output }
        - kind: emit
          value: decoder.output
          output: result
          mode: replace
"#;
    let root = package(
        "image-adapter",
        metadata,
        &[
            ("vision.onnx.textproto", VISION_IDENTITY),
            ("embedding.onnx.textproto", FEATURE_IDENTITY),
            ("decoder.onnx.textproto", FEATURE_IDENTITY),
        ],
    )?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let png = vec![
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 2,
        0, 0, 0, 144, 119, 83, 222, 0, 0, 0, 12, 73, 68, 65, 84, 120, 156, 99, 248, 207, 192, 0, 0,
        3, 1, 1, 0, 201, 254, 146, 239, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ];
    let png_len = i64::try_from(png.len())?;
    let request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input("batch_anchor", Value::from_slice_i64(&[0, 0], &[1, 2])?)
            .with_input(
                "image",
                Value::from_raw_bytes(png, &[png_len], DataType::Uint8)?,
            );
    let output = engine.run_pipeline(request)?;
    assert_eq!(output["result"].shape(), [1, 3, 2, 2]);
    assert_eq!(output["result"].to_vec_f32()?.len(), 12);
    Ok(())
}

#[test]
fn workflow_executes_real_greedy_policy_artifact() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, typed_emit]
    inputs:
      logits:
        contract: { dtype: float32, rank: 2, shape: [batch, vocabulary] }
        role: { kind: opaque }
        source: { kind: application, name: logits }
        required: true
      temperature:
        contract: { dtype: float32, rank: 1, shape: [1] }
        role: { kind: runtime, version: "1", role: sampling_temperature }
        source: { kind: request }
        required: true
      top_k:
        contract: { dtype: int64, rank: 1, shape: [1] }
        role: { kind: runtime, version: "1", role: sampling_top_k }
        source: { kind: request }
        required: true
      top_p:
        contract: { dtype: float32, rank: 1, shape: [1] }
        role: { kind: runtime, version: "1", role: sampling_top_p }
        source: { kind: request }
        required: true
      grammar_mask:
        contract: { dtype: bool, rank: 2, shape: [batch, vocabulary] }
        role: { kind: opaque }
        source: { kind: application, name: grammar_mask }
        required: true
    outputs:
      token:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: tokens
        stage: pre_adapter
    components:
      sampler:
        implementation: { kind: onnx, artifact: sampler.onnx.textproto }
        application_overridable: true
        contract:
          id: onnx-genai.token-sampler
          version: "1"
          bindings:
            logits: logits
            temperature: temperature
            top_k: top_k
            top_p: top_p
            grammar_mask: grammar_mask
            token: token_ids
          parameters:
            mode: greedy
      application_sampler:
        implementation: { kind: onnx, artifact: application-sampler.onnx.textproto }
        contract:
          id: onnx-genai.token-sampler
          version: "1"
          bindings:
            logits: scores
            temperature: temp
            top_k: k
            top_p: p
            grammar_mask: mask
            token: tokens
          parameters:
            implementation: application
      incompatible_sampler:
        implementation: { kind: onnx, artifact: incompatible-sampler.onnx.textproto }
        contract:
          id: onnx-genai.token-sampler
          version: "2"
          bindings:
            logits: scores
            temperature: temp
            top_k: k
            top_p: p
            grammar_mask: mask
            token: tokens
    steps:
        - kind: invoke
          component: sampler
          inputs:
            logits: logits
            temperature: temperature
            top_k: top_k
            top_p: top_p
            grammar_mask: grammar_mask
          outputs: { token_ids: sampled }
        - kind: emit
          value: sampled
          output: token
          mode: replace
"#;
    let invalid = metadata.replace(
        "          outputs: { token_ids: sampled }",
        "          outputs: { typo: sampled }",
    );
    let invalid_root = package(
        "greedy-invalid-port",
        &invalid,
        &[
            ("sampler.onnx.textproto", GREEDY),
            ("application-sampler.onnx.textproto", ARGMIN_SAMPLER),
            ("incompatible-sampler.onnx.textproto", ARGMIN_SAMPLER),
        ],
    )?;
    let error = Engine::from_pipeline_dir(&invalid_root, EngineConfig::default())
        .err()
        .expect("unknown inferred ONNX port must fail at load");
    assert!(
        error
            .to_string()
            .contains("invocation port 'typo' is not covered by its semantic contract ABI"),
        "{error:#}"
    );

    let root = package(
        "greedy",
        metadata,
        &[
            ("sampler.onnx.textproto", GREEDY),
            ("application-sampler.onnx.textproto", ARGMIN_SAMPLER),
            ("incompatible-sampler.onnx.textproto", ARGMIN_SAMPLER),
        ],
    )?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let logits = || Value::from_slice_f32(&[0.1, 0.7, 0.2, 2.0, 1.0, 3.0], &[2, 3]);
    let mask = || Value::from_raw_bytes(vec![1; 6], &[2, 3], DataType::Bool);
    let mut generate = GenerateRequest::new(GeneratePrompt::TokenIds(vec![]));
    generate.options.temperature = 0.75;
    generate.options.top_k = 17;
    generate.options.top_p = 0.9;
    let request = PipelineGenerateRequest::new(generate.clone())
        .with_input("logits", logits()?)
        .with_input("grammar_mask", mask()?);
    let output = engine.run_pipeline(request)?;
    assert_eq!(output["token"].to_vec_i64()?, [1, 2]);

    let error = engine
        .run_pipeline(
            PipelineGenerateRequest::new(generate.clone())
                .with_input("logits", logits()?)
                .with_input("grammar_mask", mask()?)
                .with_component_override("sampler", "incompatible_sampler"),
        )
        .err()
        .expect("a replacement with a different ABI version must be rejected");
    assert!(
        error
            .to_string()
            .contains("has contract onnx-genai.token-sampler@2")
    );

    generate.options.temperature = 0.2;
    generate.options.top_k = 3;
    generate.options.top_p = 0.5;
    let output = engine.run_pipeline(
        PipelineGenerateRequest::new(generate)
            .with_input("logits", logits()?)
            .with_input("grammar_mask", mask()?)
            .with_component_override("sampler", "application_sampler"),
    )?;
    assert_eq!(output["token"].to_vec_i64()?, [0, 1]);
    Ok(())
}

#[test]
fn decoder_present_kv_is_direct_loop_carry() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, typed_emit, nested_control_flow]
    inputs:
      initial_cache:
        contract: { dtype: float32, rank: 2, shape: [batch, cache] }
        role: { kind: opaque }
        source: { kind: application, name: initial_cache }
        required: true
      token_state:
        contract: { dtype: float32, rank: 2, shape: [batch, 1] }
        role: { kind: opaque }
        source: { kind: application, name: token_state }
        required: true
      iterations:
        contract: { dtype: int64, rank: 1, shape: [1] }
        role: { kind: opaque }
        source: { kind: application, name: iterations }
        required: true
      continue:
        contract: { dtype: bool, rank: 0, shape: [] }
        role: { kind: opaque }
        source: { kind: application, name: continue }
        required: true
      one:
        contract: { dtype: int64, rank: 1, shape: [1] }
        role: { kind: opaque }
        source: { kind: application, name: one }
        required: true
      max_context:
        contract: { dtype: int64, rank: 1, shape: [1] }
        role: { kind: opaque }
        source: { kind: application, name: max_context }
        required: true
    outputs:
      final_cache:
        contract: { dtype: float32, rank: 2, shape: [batch, cache] }
        role: tensor
        stage: pre_adapter
    components:
      decoder:
        implementation: { kind: onnx, artifact: decoder.onnx.textproto }
    state:
      cache:
        contract: { dtype: float32, rank: 2, shape: [batch, cache] }
        scope: invocation
        initializer: initial_cache
        recurrence: { kind: growing, axis: 1, increment: one, max: max_context }
    steps:
      - kind: loop
        setup: []
        steps:
          - kind: invoke
            component: decoder
            inputs: { past_key_values: cache, token_state: token_state }
            outputs: { present_key_values: cache.next }
        continue_when: continue
        max_iterations: iterations
        carried: [{ cell: cache, initial: initial_cache, next: cache.next }]
      - kind: emit
        value: cache
        output: final_cache
        mode: replace
"#;
    let root = package(
        "decoder-direct-kv-carry",
        metadata,
        &[("decoder.onnx.textproto", CACHE_DECODER)],
    )?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let output = engine.run_pipeline(
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input("initial_cache", Value::from_slice_f32(&[1.0], &[1, 1])?)
            .with_input("token_state", Value::from_slice_f32(&[2.0], &[1, 1])?)
            .with_input("iterations", Value::from_slice_i64(&[2], &[1])?)
            .with_input(
                "continue",
                Value::from_raw_bytes(vec![1], &[], DataType::Bool)?,
            )
            .with_input("one", Value::from_slice_i64(&[1], &[1])?)
            .with_input("max_context", Value::from_slice_i64(&[3], &[1])?),
    )?;
    assert_eq!(output["final_cache"].to_vec_f32()?, [1.0, 2.0, 2.0]);
    let zero_trip = engine.run_pipeline(
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input("initial_cache", Value::from_slice_f32(&[1.0], &[1, 1])?)
            .with_input("token_state", Value::from_slice_f32(&[2.0], &[1, 1])?)
            .with_input("iterations", Value::from_slice_i64(&[2], &[1])?)
            .with_input(
                "continue",
                Value::from_raw_bytes(vec![0], &[], DataType::Bool)?,
            )
            .with_input("one", Value::from_slice_i64(&[1], &[1])?)
            .with_input("max_context", Value::from_slice_i64(&[3], &[1])?),
    )?;
    assert_eq!(zero_trip["final_cache"].to_vec_f32()?, [1.0]);
    Ok(())
}

#[test]
fn pure_policy_chain_lowers_to_one_execution_island() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, typed_emit]
    inputs:
      logits:
        contract: { dtype: float32, rank: 2, shape: [batch, vocabulary] }
        role: { kind: opaque }
        source: { kind: application, name: logits }
        required: true
      min_p:
        contract: { dtype: float32, rank: 1, shape: [1] }
        role: { kind: runtime, version: "1", role: sampling_min_p }
        source: { kind: request }
        required: true
      eos:
        contract: { dtype: int64, rank: 1, shape: [1] }
        role: { kind: opaque }
        source: { kind: application, name: eos }
        required: true
    outputs:
      token:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: tokens
        stage: pre_adapter
      done:
        contract: { dtype: bool, rank: 1, shape: [batch] }
        role: tensor
        stage: pre_adapter
    components:
      min_p_filter:
        implementation: { kind: onnx, artifact: min-p.onnx.textproto }
        contract:
          id: onnx-genai.logits-processor
          version: "1"
          bindings: { logits: logits, min_p: min_p, filtered_logits: filtered_logits }
          parameters: { operation: min_p }
      sampler:
        implementation: { kind: onnx, artifact: sampler.onnx.textproto }
        application_overridable: true
        contract:
          id: onnx-genai.token-sampler
          version: "1"
          bindings: { logits: logits, token: token }
          parameters: { mode: greedy }
      alternate_sampler:
        implementation: { kind: onnx, artifact: alternate-sampler.onnx.textproto }
        contract:
          id: onnx-genai.token-sampler
          version: "1"
          bindings: { logits: logits, token: token }
          parameters: { mode: alternate }
      termination:
        implementation: { kind: onnx, artifact: eos.onnx.textproto }
        contract:
          id: onnx-genai.termination-predicate
          version: "1"
          bindings: { tokens: token, eos_ids: eos, done: done }
    steps:
      - kind: invoke
        component: min_p_filter
        inputs: { logits: logits, min_p: min_p }
        outputs: { filtered_logits: filtered }
      - kind: invoke
        component: sampler
        inputs: { logits: filtered }
        outputs: { token: sampled }
      - kind: invoke
        component: termination
        inputs: { token: sampled, eos: eos }
        outputs: { done: is_done }
      - kind: emit
        value: sampled
        output: token
        mode: replace
      - kind: emit
        value: is_done
        output: done
        mode: replace
"#;
    let root = package(
        "execution-island-min-p",
        metadata,
        &[
            ("min-p.onnx.textproto", MIN_P_FILTER),
            ("sampler.onnx.textproto", SIMPLE_GREEDY),
            ("alternate-sampler.onnx.textproto", SIMPLE_GREEDY),
            ("eos.onnx.textproto", EOS_PREDICATE),
        ],
    )?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let run = |engine: &mut onnx_genai_engine::PipelineEngine, min_p: f32| {
        let mut generate = GenerateRequest::new(GeneratePrompt::TokenIds(vec![]));
        generate.options.min_p = min_p;
        engine.run_pipeline(
            PipelineGenerateRequest::new(generate)
                .with_input(
                    "logits",
                    Value::from_slice_f32(&[4.0, 3.0, 1.0, 0.0], &[1, 4])?,
                )
                .with_input("eos", Value::from_slice_i64(&[0], &[1])?),
        )
    };
    let first = run(&mut engine, 0.5)?;
    assert_eq!(first["token"].to_vec_i64()?, [0]);
    assert_eq!(first["done"].as_raw_bytes()?, [1]);
    let second = run(&mut engine, 0.1)?;
    assert_eq!(second["token"].to_vec_i64()?, [0]);

    let diagnostics = engine.execution_island_diagnostics();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(
        diagnostics[0].components,
        ["min_p_filter", "sampler", "termination"]
    );
    assert_eq!(diagnostics[0].runs, 2);
    assert_eq!(diagnostics[0].session_runs, 2);
    assert_eq!(diagnostics[0].eager_runs, 1);
    assert_eq!(diagnostics[0].stable_binding_runs, 1);
    assert_eq!(diagnostics[0].component_boundaries_elided, 2);
    assert_eq!(diagnostics[0].linked_node_count, 7);
    assert_eq!(diagnostics[0].host_to_device_copies, 0);
    assert_eq!(diagnostics[0].device_to_host_copies, 0);
    assert!(diagnostics[0].host_to_host_copies > 0);
    assert!(diagnostics[0].stable_binding_bytes > 0);
    assert!(diagnostics[0].total_run_ns > 0);
    assert_eq!(
        diagnostics[0].fallback_reason.as_deref(),
        Some("island is not placed on CUDA")
    );
    let performance = engine.workflow_performance_diagnostic();
    assert_eq!(performance.runs, 2);
    assert_eq!(performance.last_component_invocations, 3);
    assert_eq!(performance.last_emit_events, 2);
    assert_eq!(performance.last_emitted_elements, 2);
    assert!(performance.last_ttft_ns.is_some());
    assert!(performance.last_elements_per_second > 0.0);

    let mut generate = GenerateRequest::new(GeneratePrompt::TokenIds(vec![]));
    generate.options.min_p = 0.5;
    let output = engine.run_pipeline(
        PipelineGenerateRequest::new(generate)
            .with_input(
                "logits",
                Value::from_slice_f32(&[4.0, 3.0, 1.0, 0.0], &[1, 4])?,
            )
            .with_input("eos", Value::from_slice_i64(&[0], &[1])?)
            .with_component_override("sampler", "alternate_sampler"),
    )?;
    assert_eq!(output["token"].to_vec_i64()?, [0]);
    assert_eq!(
        engine.execution_island_diagnostics()[0].runs,
        2,
        "a selected replacement must execute the preserved unfused sequence"
    );
    Ok(())
}

#[test]
fn execution_island_references_external_weights_without_inlining() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, typed_emit]
    inputs:
      logits:
        contract: { dtype: float32, rank: 2, shape: [batch, 4] }
        role: { kind: opaque }
        source: { kind: application, name: logits }
        required: true
    outputs:
      token:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: tokens
        stage: pre_adapter
    components:
      bias:
        implementation: { kind: onnx, artifact: bias.onnx }
      sampler:
        implementation: { kind: onnx, artifact: sampler.onnx.textproto }
    steps:
      - kind: invoke
        component: bias
        inputs: { logits: logits }
        outputs: { adjusted: adjusted }
      - kind: invoke
        component: sampler
        inputs: { logits: adjusted }
        outputs: { token: sampled }
      - kind: emit
        value: sampled
        output: token
        mode: replace
"#;
    let root = package(
        "execution-island-external-data",
        metadata,
        &[
            ("bias.onnx.textproto", EXTERNAL_BIAS),
            ("sampler.onnx.textproto", SIMPLE_GREEDY),
        ],
    )?;
    let bias = [0.0_f32, 0.0, 10.0, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let bias_path = root.join("bias.bin");
    fs::write(&bias_path, bias)?;
    fs::write(
        root.join("bias.onnx"),
        onnx_runtime_loader::read_model_binary(root.join("bias.onnx.textproto"))?,
    )?;

    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let output = engine.run_pipeline(
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input(
                "logits",
                Value::from_slice_f32(&[4.0, 3.0, 1.0, 0.0], &[1, 4])?,
            ),
    )?;
    assert_eq!(output["token"].to_vec_i64()?, [2]);
    assert_eq!(engine.execution_island_diagnostics().len(), 1);
    Ok(())
}

#[test]
fn execution_island_disambiguates_sanitized_value_names() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, typed_emit]
    inputs:
      logits.raw:
        contract: { dtype: float32, rank: 2, shape: [batch, 4] }
        role: { kind: opaque }
        source: { kind: application, name: logits.raw }
        required: true
      logits_raw:
        contract: { dtype: float32, rank: 2, shape: [batch, 4] }
        role: { kind: opaque }
        source: { kind: application, name: logits_raw }
        required: true
    outputs:
      token:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: tokens
        stage: pre_adapter
    components:
      add:
        implementation: { kind: onnx, artifact: add.onnx.textproto }
      sampler:
        implementation: { kind: onnx, artifact: sampler.onnx.textproto }
    steps:
      - kind: invoke
        component: add
        inputs: { left: logits.raw, right: logits_raw }
        outputs: { sum: combined }
      - kind: invoke
        component: sampler
        inputs: { logits: combined }
        outputs: { token: sampled }
      - kind: emit
        value: sampled
        output: token
        mode: replace
"#;
    let root = package(
        "execution-island-name-collision",
        metadata,
        &[
            ("add.onnx.textproto", ADD_PAIR),
            ("sampler.onnx.textproto", SIMPLE_GREEDY),
        ],
    )?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let output = engine.run_pipeline(
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input(
                "logits.raw",
                Value::from_slice_f32(&[10.0, 0.0, 0.0, 0.0], &[1, 4])?,
            )
            .with_input(
                "logits_raw",
                Value::from_slice_f32(&[0.0, 0.0, 0.0, 5.0], &[1, 4])?,
            ),
    )?;
    assert_eq!(output["token"].to_vec_i64()?, [0]);
    assert_eq!(engine.execution_island_diagnostics().len(), 1);
    Ok(())
}

#[test]
fn workflow_scalar_literal_materializes_unbound_symbol_as_singleton() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, typed_emit]
    inputs:
      eos:
        contract: { dtype: int64, rank: 1, shape: [num_eos] }
        role: { kind: opaque }
        source: { kind: literal }
        required: true
        default: 99
    outputs:
      result:
        contract: { dtype: int64, rank: 1, shape: [num_eos] }
        role: tokens
        stage: pre_adapter
    components: {}
    steps:
      - kind: emit
        value: eos
        output: result
        mode: replace
"#;
    let root = package("symbolic-literal", metadata, &[])?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let output = engine.run_pipeline(PipelineGenerateRequest::new(GenerateRequest::new(
        GeneratePrompt::TokenIds(vec![]),
    )))?;
    assert_eq!(output["result"].shape(), [1]);
    assert_eq!(output["result"].to_vec_i64()?, [99]);
    Ok(())
}

#[test]
fn workflow_component_symbols_are_scoped_per_invocation() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, typed_emit]
    inputs:
      prompt:
        contract: { dtype: int64, rank: 2, shape: [batch, prompt_sequence] }
        role: { kind: opaque }
        source: { kind: application, name: prompt }
        required: true
      token:
        contract: { dtype: int64, rank: 2, shape: [batch, 1] }
        role: { kind: opaque }
        source: { kind: application, name: token }
        required: true
    outputs:
      result:
        contract: { dtype: int64, rank: 2, shape: [batch, 1] }
        role: tokens
        stage: pre_adapter
    components:
      identity:
        implementation: { kind: onnx, artifact: identity.onnx.textproto }
        ports:
          inputs:
            input: { dtype: int64, rank: 2, shape: [batch, sequence] }
          outputs:
            output: { dtype: int64, rank: 2, shape: [batch, sequence] }
    steps:
        - kind: invoke
          component: identity
          inputs: { input: prompt }
          outputs: { output: prompt.output }
        - kind: invoke
          component: identity
          inputs: { input: token }
          outputs: { output: token.output }
        - kind: emit
          value: token.output
          output: result
          mode: replace
"#;
    let root = package(
        "component-symbol-scope",
        metadata,
        &[("identity.onnx.textproto", INT64_MATRIX_IDENTITY)],
    )?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input("prompt", Value::from_slice_i64(&[1, 2], &[1, 2])?)
            .with_input("token", Value::from_slice_i64(&[3], &[1, 1])?);
    let output = engine.run_pipeline(request)?;
    assert_eq!(output["result"].to_vec_i64()?, [3]);
    Ok(())
}

#[test]
fn workflow_component_symbols_ignore_package_dynamic_symbol_names() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, typed_emit, bounded_state_growth]
    inputs:
      prompt:
        contract: { dtype: int64, rank: 2, shape: [batch, prompt_sequence] }
        role: { kind: opaque }
        source: { kind: application, name: prompt }
        required: true
      token:
        contract: { dtype: int64, rank: 2, shape: [batch, 1] }
        role: { kind: opaque }
        source: { kind: application, name: token }
        required: true
      one:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: { kind: opaque }
        source: { kind: literal }
        required: false
        default: 1
      maximum:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: { kind: opaque }
        source: { kind: literal }
        required: false
        default: 8
    outputs:
      result:
        contract: { dtype: int64, rank: 2, shape: [batch, prompt_sequence] }
        role: tokens
        stage: pre_adapter
    components:
      pair:
        implementation: { kind: onnx, artifact: pair.onnx.textproto }
        ports:
          inputs:
            left: { dtype: int64, rank: 2, shape: [batch, sequence] }
            right: { dtype: int64, rank: 2, shape: [batch, sequence] }
          outputs:
            output: { dtype: int64, rank: 2, shape: [batch, sequence] }
    state:
      dummy:
        contract: { dtype: int64, rank: 2, shape: [batch, sequence] }
        scope: invocation
        initializer: prompt
        recurrence: { kind: growing, axis: 1, increment: one, max: maximum }
    steps:
        - kind: invoke
          component: pair
          inputs: { left: prompt, right: token }
          outputs: { output: combined }
        - kind: emit
          value: combined
          output: result
          mode: replace
"#;
    let root = package(
        "component-dynamic-symbol-scope",
        metadata,
        &[("pair.onnx.textproto", INT64_MATRIX_ADD)],
    )?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input("prompt", Value::from_slice_i64(&[1, 2], &[1, 2])?)
            .with_input("token", Value::from_slice_i64(&[3], &[1, 1])?);
    let error = match engine.run_pipeline(request) {
        Ok(_) => anyhow::bail!("mismatched component symbols unexpectedly executed"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("binds symbol 'sequence'"), "{error}");
    Ok(())
}

#[test]
fn workflow_executes_autoregressive_rng_and_eos_loop() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, typed_emit, nested_control_flow]
    inputs:
      logits: { contract: { dtype: float32, rank: 2, shape: [batch, vocabulary] },
                role: { kind: opaque }, source: { kind: application, name: logits }, required: true }
      seed:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: { kind: runtime, version: v1, role: seed }
        source: { kind: request }
        required: true
      offset: { contract: { dtype: int64, rank: 1, shape: [batch] },
                role: { kind: opaque }, source: { kind: application, name: offset }, required: true }
      eos: { contract: { dtype: int64, rank: 1, shape: [batch] },
             role: { kind: opaque }, source: { kind: application, name: eos }, required: true }
      iteration: { contract: { dtype: int64, rank: 1, shape: [batch] },
                   role: { kind: opaque }, source: { kind: application, name: iteration }, required: true }
      max_iterations: { contract: { dtype: int64, rank: 1, shape: [batch] },
                        role: { kind: opaque }, source: { kind: application, name: max_iterations }, required: true }
      iterations:
        contract: { dtype: int64, rank: 0, shape: [] }
        role: { kind: runtime, version: v1, role: max_output_tokens }
        source: { kind: request }
        required: true
      initial_continue: { contract: { dtype: bool, rank: 1, shape: [1] },
                          role: { kind: opaque },
                          source: { kind: application, name: initial_continue }, required: true }
    outputs:
      tokens: { contract: { dtype: int64, rank: 1, shape: [generated] },
                role: tokens, stage: pre_adapter }
    components:
      binding:
        implementation: { kind: binding }
        ports:
          inputs: { value: { dtype: int64, rank: 1, shape: [batch] } }
          outputs: { value: { dtype: int64, rank: 1, shape: [batch] } }
      sampler:
        implementation: { kind: onnx, artifact: sampler.onnx.textproto }
        ports:
          inputs:
            logits: { dtype: float32, rank: 2, shape: [batch, vocabulary] }
            seed: { dtype: int64, rank: 1, shape: [batch] }
            offset: { dtype: int64, rank: 1, shape: [batch] }
          outputs:
            token_ids: { dtype: int64, rank: 1, shape: [batch] }
            next_offset: { dtype: int64, rank: 1, shape: [batch] }
        contract:
          id: onnx-genai.token-sampler
          version: "1"
          bindings:
            logits: logits
            token: token_ids
          parameters:
            mode: seeded_stochastic
      termination:
        implementation: { kind: onnx, artifact: eos.onnx.textproto }
        ports:
          inputs:
            token_ids: { dtype: int64, rank: 1, shape: [batch] }
            eos_token_ids: { dtype: int64, rank: 1, shape: [batch] }
            iteration: { dtype: int64, rank: 1, shape: [batch] }
            max_iterations: { dtype: int64, rank: 1, shape: [batch] }
          outputs:
            terminated: { dtype: bool, rank: 1, shape: [batch] }
        contract:
          id: onnx-genai.termination-predicate
          version: "1"
          bindings:
            tokens: token_ids
            eos_ids: eos_token_ids
            iteration: iteration
            max_iterations: max_iterations
            done: terminated
      invert:
        implementation: { kind: onnx, artifact: not.onnx.textproto }
        ports:
          inputs: { done: { dtype: bool, rank: 1, shape: [batch] } }
          outputs: { continue: { dtype: bool, rank: 1, shape: [1] } }
    state:
      rng:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        scope: invocation
        initializer: offset
        recurrence: { kind: invariant }
      active:
        contract: { dtype: bool, rank: 1, shape: [1] }
        scope: invocation
        initializer: initial_continue
        recurrence: { kind: invariant }
    steps:
      - kind: loop
        setup:
        - kind: invoke
          component: binding
          inputs: { value: offset }
          outputs: { value: rng.current }
        steps:
            - kind: invoke
              component: sampler
              inputs: { logits: logits, seed: seed, offset: rng }
              outputs: { token_ids: sampled, next_offset: rng.body_next }
            - kind: invoke
              component: termination
              inputs: { token_ids: sampled, eos_token_ids: eos,
                        iteration: iteration, max_iterations: max_iterations }
              outputs: { terminated: done }
            - kind: invoke
              component: invert
              inputs: { done: done }
              outputs: { continue: loop.continue }
            - kind: emit
              value: sampled
              output: tokens
              mode: append
        continue_when: active
        max_iterations: iterations
        carried:
          - cell: rng
            initial: rng.current
            next: rng.body_next
          - cell: active
            next: loop.continue
"#;
    let root = package(
        "autoregressive",
        metadata,
        &[
            ("sampler.onnx.textproto", SEEDED_SAMPLER),
            ("eos.onnx.textproto", EOS),
            ("not.onnx.textproto", NOT),
        ],
    )?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let options = onnx_genai_engine::GenerateOptions {
        max_new_tokens: 3,
        seed: Some(7),
        ..Default::default()
    };
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options,
    })
    .with_input("logits", Value::from_slice_f32(&[0.1, 0.9, 0.0], &[1, 3])?)
    .with_input("offset", Value::from_slice_i64(&[0], &[1])?)
    .with_input("eos", Value::from_slice_i64(&[2], &[1])?)
    .with_input(
        "initial_continue",
        Value::from_raw_bytes(vec![1], &[1], DataType::Bool)?,
    )
    .with_input("iteration", Value::from_slice_i64(&[0], &[1])?)
    .with_input("max_iterations", Value::from_slice_i64(&[10], &[1])?);
    let output = engine.run_pipeline(request)?;
    assert_eq!(output["tokens"].to_vec_i64()?, [1, 1, 1]);
    Ok(())
}

#[test]
fn workflow_executes_diffusion_solver_policy_artifact() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, typed_emit, nested_control_flow,
                     loop_induction_values]
    inputs:
      sample: { contract: { dtype: float32, rank: 2, shape: [batch, width] },
                role: { kind: opaque }, source: { kind: application, name: sample }, required: true }
      derivative: { contract: { dtype: float32, rank: 2, shape: [batch, width] },
                    role: { kind: opaque }, source: { kind: application, name: derivative }, required: true }
      schedule: { contract: { dtype: float32, rank: 1, shape: [schedule_length] },
                  role: { kind: opaque }, source: { kind: application, name: schedule }, required: true }
      iterations: { contract: { dtype: int64, rank: 0, shape: [] },
                    role: { kind: opaque }, source: { kind: application, name: iterations },
                    required: true }
      continue: { contract: { dtype: bool, rank: 0, shape: [] },
                  role: { kind: opaque }, source: { kind: application, name: continue },
                  required: true }
    outputs:
      latent: { contract: { dtype: float32, rank: 2, shape: [batch, width] },
                role: tensor, stage: pre_adapter }
      steps: { contract: { dtype: int64, rank: 1, shape: [generated] },
               role: event, stage: pre_adapter }
    components:
      sample_binding:
        implementation: { kind: binding }
        ports:
          inputs: { value: { dtype: float32, rank: 2, shape: [batch, width] } }
          outputs: { value: { dtype: float32, rank: 2, shape: [batch, width] } }
      solver:
        implementation: { kind: onnx, artifact: solver.onnx.textproto }
        ports:
          inputs:
            sample: { dtype: float32, rank: 2, shape: [batch, width] }
            derivative: { dtype: float32, rank: 2, shape: [batch, width] }
            step: { dtype: int64, rank: 1, shape: [batch] }
            schedule: { dtype: float32, rank: 1, shape: [schedule_length] }
          outputs:
            next_state: { dtype: float32, rank: 2, shape: [batch, width] }
        contract:
          id: onnx-genai.solver-step
          version: "1"
          bindings:
            state: sample
            estimate: derivative
            step: step
            schedule: schedule
            next_state: next_state
    state:
      latent:
        contract: { dtype: float32, rank: 2, shape: [batch, width] }
        scope: invocation
        initializer: latent.current
        recurrence: { kind: invariant }
    steps:
        - kind: loop
          setup:
            - kind: invoke
              component: sample_binding
              inputs: { value: sample }
              outputs: { value: latent.current }
          steps:
              - kind: invoke
                component: solver
                inputs: { sample: latent, derivative: derivative,
                          step: diffusion.step, schedule: schedule }
                outputs: { next_state: latent.next }
              - kind: emit
                value: diffusion.step
                output: steps
                mode: append
          continue_when: continue
          max_iterations: iterations
          iteration:
            value: diffusion.step
            contract: { dtype: int64, rank: 1, shape: [batch] }
          carried:
            - cell: latent
              initial: latent.current
              next: latent.next
        - kind: emit
          value: latent
          output: latent
          mode: replace
"#;
    let root = package("solver", metadata, &[("solver.onnx.textproto", EULER)])?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input("sample", Value::from_slice_f32(&[1.0, 2.0], &[1, 2])?)
            .with_input("derivative", Value::from_slice_f32(&[0.5, 0.25], &[1, 2])?)
            .with_input("schedule", Value::from_slice_f32(&[1.0, 0.0], &[2])?)
            .with_input("iterations", Value::from_slice_i64(&[3], &[])?)
            .with_input(
                "continue",
                Value::from_raw_bytes(vec![1], &[], DataType::Bool)?,
            );
    let output = engine.run_pipeline(request)?;
    assert_eq!(output["latent"].to_vec_f32()?, [-0.5, 1.25]);
    assert_eq!(output["steps"].to_vec_i64()?, [0, 1, 2]);
    let zero = PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
        .with_input("sample", Value::from_slice_f32(&[1.0, 2.0], &[1, 2])?)
        .with_input("derivative", Value::from_slice_f32(&[0.5, 0.25], &[1, 2])?)
        .with_input("schedule", Value::from_slice_f32(&[1.0, 0.0], &[2])?)
        .with_input("iterations", Value::from_slice_i64(&[0], &[])?)
        .with_input(
            "continue",
            Value::from_raw_bytes(vec![1], &[], DataType::Bool)?,
        );
    assert_eq!(
        engine.run_pipeline(zero)?["latent"].to_vec_f32()?,
        [1.0, 2.0]
    );
    Ok(())
}

#[test]
fn workflow_nested_loops_materialize_lexical_induction_values() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, typed_emit, nested_control_flow,
                     loop_induction_values]
    inputs:
      outer_count: { contract: { dtype: int64, rank: 0, shape: [] },
                     role: { kind: opaque }, source: { kind: application, name: outer_count },
                     required: true }
      inner_count: { contract: { dtype: int64, rank: 0, shape: [] },
                     role: { kind: opaque }, source: { kind: application, name: inner_count },
                     required: true }
      continue: { contract: { dtype: bool, rank: 0, shape: [] },
                  role: { kind: opaque }, source: { kind: application, name: continue },
                  required: true }
    outputs:
      outer_steps: { contract: { dtype: int64, rank: 1, shape: [outer_events] },
                     role: event, stage: pre_adapter }
      inner_steps: { contract: { dtype: int64, rank: 1, shape: [inner_events] },
                     role: event, stage: pre_adapter }
    components: {}
    steps:
      - kind: loop
        steps:
            - kind: emit
              value: outer.index
              output: outer_steps
              mode: append
            - kind: loop
              steps:
              - kind: emit
                value: inner.index
                output: inner_steps
                mode: append
              continue_when: continue
              max_iterations: inner_count
              iteration:
                value: inner.index
                contract: { dtype: int64, rank: 1, shape: [1] }
              carried: []
        continue_when: continue
        max_iterations: outer_count
        iteration:
          value: outer.index
          contract: { dtype: int64, rank: 1, shape: [1] }
        carried: []
"#;
    let root = package("nested-induction", metadata, &[])?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input("outer_count", Value::from_slice_i64(&[2], &[])?)
            .with_input("inner_count", Value::from_slice_i64(&[3], &[])?)
            .with_input(
                "continue",
                Value::from_raw_bytes(vec![1], &[], DataType::Bool)?,
            );
    let output = engine.run_pipeline(request)?;
    assert_eq!(output["outer_steps"].to_vec_i64()?, [0, 1]);
    assert_eq!(output["inner_steps"].to_vec_i64()?, [0, 1, 2, 0, 1, 2]);
    Ok(())
}

#[test]
fn workflow_executes_masked_and_speculative_policy_artifacts() -> anyhow::Result<()> {
    let masked_metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, typed_emit]
    inputs:
      current: { contract: { dtype: int64, rank: 2, shape: [batch, sequence] }, role: { kind: opaque },
                 source: { kind: application, name: current }, required: true }
      proposed: { contract: { dtype: int64, rank: 2, shape: [batch, sequence] }, role: { kind: opaque },
                  source: { kind: application, name: proposed }, required: true }
      mask: { contract: { dtype: bool, rank: 2, shape: [batch, sequence] }, role: { kind: opaque },
              source: { kind: application, name: mask }, required: true }
      step: { contract: { dtype: int64, rank: 1, shape: [batch] }, role: { kind: opaque },
              source: { kind: application, name: step }, required: true }
    outputs:
      tokens: { contract: { dtype: int64, rank: 2, shape: [batch, sequence] }, role: tokens, stage: pre_adapter }
    components:
      update:
        implementation: { kind: onnx, artifact: update.onnx.textproto }
        ports:
          inputs:
            current_tokens: { dtype: int64, rank: 2, shape: [batch, sequence] }
            proposed_tokens: { dtype: int64, rank: 2, shape: [batch, sequence] }
            masked: { dtype: bool, rank: 2, shape: [batch, sequence] }
            step: { dtype: int64, rank: 1, shape: [batch] }
          outputs:
            next_state: { dtype: int64, rank: 2, shape: [batch, sequence] }
            next_mask: { dtype: bool, rank: 2, shape: [batch, sequence] }
        contract:
          id: onnx-genai.masked-update
          version: "1"
          bindings:
            state: current_tokens
            proposal: proposed_tokens
            mask: masked
            step: step
            next_state: next_state
            next_mask: next_mask
    steps:
        - kind: invoke
          component: update
          inputs: { current_tokens: current, proposed_tokens: proposed, masked: mask, step: step }
          outputs: { next_state: updated, next_mask: remaining }
        - kind: emit
          value: updated
          output: tokens
          mode: replace
"#;
    let root = package(
        "masked",
        masked_metadata,
        &[("update.onnx.textproto", MASKED_UPDATE)],
    )?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input("current", Value::from_slice_i64(&[1, 99, 99], &[1, 3])?)
            .with_input("proposed", Value::from_slice_i64(&[4, 5, 6], &[1, 3])?)
            .with_input(
                "mask",
                Value::from_raw_bytes(vec![0, 1, 1], &[1, 3], onnx_genai_ort::DataType::Bool)?,
            )
            .with_input("step", Value::from_slice_i64(&[0], &[1])?);
    assert_eq!(
        engine.run_pipeline(request)?["tokens"].to_vec_i64()?,
        [1, 5, 6]
    );

    let speculative_metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, typed_emit, emit_valid_length]
    inputs:
      target: { contract: { dtype: float32, rank: 3, shape: [batch, draft, vocabulary] }, role: { kind: opaque },
                source: { kind: application, name: target }, required: true }
      proposed: { contract: { dtype: int64, rank: 2, shape: [batch, draft] }, role: { kind: opaque },
                  source: { kind: application, name: proposed }, required: true }
    outputs:
      accepted_len: { contract: { dtype: int64, rank: 1, shape: [batch] }, role: tensor, stage: pre_adapter }
      accepted_tokens: { contract: { dtype: int64, rank: 2, shape: [batch, accepted] },
                         role: tokens, stage: pre_adapter }
    components:
      verifier:
        implementation: { kind: onnx, artifact: verifier.onnx.textproto }
        ports:
          inputs:
            target_scores: { dtype: float32, rank: 3, shape: [batch, draft, vocabulary] }
            proposed_tokens: { dtype: int64, rank: 2, shape: [batch, draft] }
          outputs:
            accepted_tokens: { dtype: int64, rank: 2, shape: [batch, draft] }
            accepted_count: { dtype: int64, rank: 1, shape: [batch] }
            done: { dtype: bool, rank: 1, shape: [batch] }
        contract:
          id: onnx-genai.speculative-verifier
          version: "1"
          bindings:
            target_scores: target_scores
            proposed_tokens: proposed_tokens
            accepted_tokens: accepted_tokens
            accepted_len: accepted_count
            done: done
    steps:
        - kind: invoke
          component: verifier
          inputs: { target_scores: target, proposed_tokens: proposed }
          outputs: { accepted_tokens: accepted, accepted_count: count, done: done }
        - kind: emit
          value: accepted
          valid_length: count
          output: accepted_tokens
          mode: replace
        - kind: emit
          value: count
          output: accepted_len
          mode: replace
"#;
    let root = package(
        "speculative",
        speculative_metadata,
        &[("verifier.onnx.textproto", SPECULATIVE_ACCEPTANCE)],
    )?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input(
                "target",
                Value::from_slice_f32(&[0.1, 0.9, 0.8, 0.2, 0.7, 0.3], &[1, 3, 2])?,
            )
            .with_input("proposed", Value::from_slice_i64(&[1, 1, 0], &[1, 3])?);
    let outputs = engine.run_pipeline(request)?;
    assert_eq!(outputs["accepted_len"].to_vec_i64()?, [2]);
    assert_eq!(outputs["accepted_tokens.row.0"].shape(), [1, 2]);
    assert_eq!(outputs["accepted_tokens.row.0"].to_vec_i64()?, [1, 1]);

    let batched_request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input(
                "target",
                Value::from_slice_f32(
                    &[0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.1, 0.9, 0.8, 0.2, 0.7, 0.3],
                    &[2, 3, 2],
                )?,
            )
            .with_input(
                "proposed",
                Value::from_slice_i64(&[1, 1, 0, 1, 1, 0], &[2, 3])?,
            );
    let outputs = engine.run_pipeline(batched_request)?;
    assert_eq!(outputs["accepted_tokens.row.0"].to_vec_i64()?, [1, 1]);
    assert_eq!(outputs["accepted_tokens.row.1"].to_vec_i64()?, [1, 1]);
    Ok(())
}

#[test]
fn workflow_selects_bounded_state_prefix_through_branch_phi() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities:
        [workflow_ssa, typed_emit, nested_control_flow,
         bounded_state_recurrence]
    inputs:
      tentative: { contract: { dtype: int64, rank: 2, shape: [batch, state] },
                   role: { kind: opaque }, source: { kind: application, name: tentative },
                   required: true }
      correction: { contract: { dtype: int64, rank: 2, shape: [batch, state] },
                    role: { kind: opaque }, source: { kind: application, name: correction },
                    required: true }
      accepted_len: { contract: { dtype: int64, rank: 1, shape: [1] },
                      role: { kind: opaque }, source: { kind: application, name: accepted_len },
                      required: true }
      accept: { contract: { dtype: bool, rank: 0, shape: [] },
                role: { kind: opaque }, source: { kind: application, name: accept },
                required: true }
      continue: { contract: { dtype: bool, rank: 0, shape: [] },
                  role: { kind: opaque }, source: { kind: application, name: continue },
                  required: true }
      iterations: { contract: { dtype: int64, rank: 0, shape: [] },
                    role: { kind: opaque }, source: { kind: application, name: iterations },
                    required: true }
      max_context: { contract: { dtype: int64, rank: 0, shape: [] },
                     role: { kind: opaque }, source: { kind: application, name: max_context },
                     required: true }
    outputs:
      state: { contract: { dtype: int64, rank: 2, shape: [batch, state] },
               role: tensor, stage: pre_adapter }
    components:
      identity:
        implementation: { kind: onnx, artifact: identity.onnx.textproto }
        ports:
          inputs: { input: { dtype: int64, rank: 2, shape: [batch, state] } }
          outputs: { output: { dtype: int64, rank: 2, shape: [batch, state] } }
      accepted_prefix:
        implementation: { kind: onnx, artifact: prefix.onnx.textproto }
        ports:
          inputs:
            state: { dtype: int64, rank: 2, shape: [batch, tentative] }
            valid_length: { dtype: int64, rank: 1, shape: [1] }
          outputs:
            selected: { dtype: int64, rank: 2, shape: [batch, state] }
    state:
      rollback:
        contract: { dtype: int64, rank: 2, shape: [batch, state] }
        scope: invocation
        initializer: rollback.current
        recurrence: { kind: bounded, axis: 1, max: max_context }
    steps:
        - kind: loop
          setup:
            - kind: invoke
              component: identity
              inputs: { input: tentative }
              outputs: { output: rollback.current }
          steps:
              - kind: invoke
                component: accepted_prefix
                inputs: { state: rollback, valid_length: accepted_len }
                outputs: { selected: accepted.state }
              - kind: branch
                predicate: accept
                cases:
                  "true":
                    kind: invoke
                    component: identity
                    inputs: { input: accepted.state }
                    outputs: { output: branch.accepted }
                  "false":
                    kind: invoke
                    component: identity
                    inputs: { input: correction }
                    outputs: { output: branch.corrected }
                outputs:
                  selected:
                    cases: { "true": branch.accepted, "false": branch.corrected }
          continue_when: continue
          max_iterations: iterations
          carried:
            - cell: rollback
              initial: rollback.current
              next: selected
        - kind: invoke
          component: identity
          inputs: { input: rollback }
          outputs: { output: observed }
        - kind: emit
          value: observed
          output: state
          mode: replace
"#;
    let root = package(
        "bounded-state-prefix",
        metadata,
        &[
            ("identity.onnx.textproto", INT64_MATRIX_IDENTITY),
            ("prefix.onnx.textproto", INT64_PREFIX),
        ],
    )?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input(
                "tentative",
                Value::from_slice_i64(&[10, 11, 12, 13], &[1, 4])?,
            )
            .with_input("correction", Value::from_slice_i64(&[20, 21, 22], &[1, 3])?)
            .with_input("accepted_len", Value::from_slice_i64(&[2], &[1])?)
            .with_input(
                "accept",
                Value::from_raw_bytes(vec![1], &[], onnx_genai_ort::DataType::Bool)?,
            )
            .with_input(
                "continue",
                Value::from_raw_bytes(vec![1], &[], onnx_genai_ort::DataType::Bool)?,
            )
            .with_input("iterations", Value::from_slice_i64(&[1], &[])?)
            .with_input("max_context", Value::from_slice_i64(&[4], &[])?);
    let output = engine.run_pipeline(request)?;
    assert_eq!(output["state"].shape(), [1, 2]);
    assert_eq!(output["state"].to_vec_i64()?, [10, 11]);
    Ok(())
}

#[test]
fn workflow_combines_speculation_grammar_and_adaptive_budget() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: { onnx-genai.grammar-guidance: "1" }
      custom_op_versions: {}
      capabilities:
        [workflow_ssa, typed_emit, emit_valid_length,
         nested_control_flow, grammar_guidance_adapter, adaptive_proposal_budget,
         advisory_state]
    inputs:
      proposed: { contract: { dtype: int64, rank: 2, shape: [batch, proposal] },
                  role: { kind: opaque }, source: { kind: application, name: proposed },
                  required: true }
      target: { contract: { dtype: float32, rank: 3, shape: [batch, proposal, vocabulary] },
                role: { kind: opaque }, source: { kind: application, name: target },
                required: true }
      logits: { contract: { dtype: float32, rank: 2, shape: [batch, vocabulary] },
                role: { kind: opaque }, source: { kind: application, name: logits },
                required: true }
      grammar_state: { contract: { dtype: int64, rank: 1, shape: [batch] },
                       role: { kind: opaque },
                       source: { kind: application, name: grammar_state }, required: true }
      transition_table: { contract: { dtype: int64, rank: 2, shape: [grammar_states, vocabulary] },
                          role: { kind: opaque },
                          source: { kind: application, name: transition_table }, required: true }
      zero_length: { contract: { dtype: int64, rank: 1, shape: [batch] },
                     role: { kind: opaque }, source: { kind: application, name: zero_length },
                     required: true }
      evaluated: { contract: { dtype: int64, rank: 1, shape: [batch] },
                   role: { kind: opaque }, source: { kind: application, name: evaluated },
                   required: true }
      current_k: { contract: { dtype: int64, rank: 1, shape: [batch] },
                   role: { kind: opaque }, source: { kind: application, name: current_k },
                   required: true }
      estimates: { contract: { dtype: float32, rank: 2, shape: [batch, budget_slots] },
                   role: { kind: opaque }, source: { kind: application, name: estimates },
                   required: true }
      filled: { contract: { dtype: bool, rank: 1, shape: [batch] },
                role: { kind: opaque }, source: { kind: application, name: filled },
                required: true }
      draft_ms: { contract: { dtype: float32, rank: 1, shape: [batch] },
                  role: { kind: opaque }, source: { kind: application, name: draft_ms },
                  required: true }
      target_ms: { contract: { dtype: float32, rank: 1, shape: [batch] },
                   role: { kind: opaque }, source: { kind: application, name: target_ms },
                   required: true }
      continue: { contract: { dtype: bool, rank: 0, shape: [] },
                  role: { kind: opaque }, source: { kind: application, name: continue },
                  required: true }
      iterations: { contract: { dtype: int64, rank: 0, shape: [] },
                    role: { kind: opaque }, source: { kind: application, name: iterations },
                    required: true }
    outputs:
      tokens: { contract: { dtype: int64, rank: 2, shape: [batch, generated] },
                role: tokens, stage: pre_adapter }
      next_k: { contract: { dtype: int64, rank: 1, shape: [batch] },
                role: tensor, stage: pre_adapter }
      final_grammar_state: { contract: { dtype: int64, rank: 1, shape: [batch] },
                             role: tensor, stage: pre_adapter }
    components:
      bind_grammar:
        implementation: { kind: binding }
        ports:
          inputs: { value: { dtype: int64, rank: 1, shape: [batch] } }
          outputs: { value: { dtype: int64, rank: 1, shape: [batch] } }
        effects: []
      bind_k:
        implementation: { kind: binding }
        ports:
          inputs: { value: { dtype: int64, rank: 1, shape: [batch] } }
          outputs: { value: { dtype: int64, rank: 1, shape: [batch] } }
        effects: []
      bind_estimates:
        implementation: { kind: binding }
        ports:
          inputs: { value: { dtype: float32, rank: 2, shape: [batch, budget_slots] } }
          outputs: { value: { dtype: float32, rank: 2, shape: [batch, budget_slots] } }
        effects: []
      grammar_clone:
        implementation: { kind: adapter, abi: onnx-genai.grammar-guidance, version: "1" }
        contract:
          id: onnx-genai.grammar-guidance
          version: "1"
          bindings:
            state: state
            tokens: tokens
            valid_length: valid_length
            transition_table: transition_table
            next_state: next_state
            consumed_length: consumed_length
            logits_mask: logits_mask
            forced_tokens: forced_tokens
            forced_length: forced_length
          parameters:
            action: clone
        ports: &grammar_ports
          inputs:
            state: { dtype: int64, rank: 1, shape: [batch] }
            tokens: { dtype: int64, rank: 2, shape: [batch, proposal] }
            valid_length: { dtype: int64, rank: 1, shape: [batch] }
            transition_table: { dtype: int64, rank: 2, shape: [grammar_states, vocabulary] }
          outputs:
            next_state: { dtype: int64, rank: 1, shape: [batch] }
            consumed_length: { dtype: int64, rank: 1, shape: [batch] }
            logits_mask: { dtype: bool, rank: 2, shape: [batch, vocabulary] }
            forced_tokens: { dtype: int64, rank: 2, shape: [batch, 1] }
            forced_length: { dtype: int64, rank: 1, shape: [batch] }
        effects: [grammar]
      grammar_lookahead:
        implementation: { kind: adapter, abi: onnx-genai.grammar-guidance, version: "1" }
        contract:
          id: onnx-genai.grammar-guidance
          version: "1"
          bindings:
            state: state
            tokens: tokens
            valid_length: valid_length
            transition_table: transition_table
            next_state: next_state
            consumed_length: consumed_length
            logits_mask: logits_mask
            forced_tokens: forced_tokens
            forced_length: forced_length
          parameters:
            action: lookahead
        ports: *grammar_ports
        effects: [grammar]
      grammar_commit:
        implementation: { kind: adapter, abi: onnx-genai.grammar-guidance, version: "1" }
        contract:
          id: onnx-genai.grammar-guidance
          version: "1"
          bindings:
            state: state
            tokens: tokens
            valid_length: valid_length
            transition_table: transition_table
            next_state: next_state
            consumed_length: consumed_length
            logits_mask: logits_mask
            forced_tokens: forced_tokens
            forced_length: forced_length
          parameters:
            action: commit
        ports: *grammar_ports
        effects: [grammar]
      verifier:
        implementation: { kind: onnx, artifact: verifier.onnx.textproto }
        ports:
          inputs:
            target_scores: { dtype: float32, rank: 3, shape: [batch, proposal, vocabulary] }
            proposed_tokens: { dtype: int64, rank: 2, shape: [batch, proposal] }
          outputs:
            accepted_tokens: { dtype: int64, rank: 2, shape: [batch, proposal] }
            accepted_count: { dtype: int64, rank: 1, shape: [1] }
            done: { dtype: bool, rank: 1, shape: [batch] }
        contract:
          id: onnx-genai.speculative-verifier
          version: "1"
          bindings:
            target_scores: target_scores
            proposed_tokens: proposed_tokens
            accepted_tokens: accepted_tokens
            accepted_len: accepted_count
            done: done
      min_length:
        implementation: { kind: onnx, artifact: min.onnx.textproto }
        ports:
          inputs:
            accepted: { dtype: int64, rank: 1, shape: [batch] }
            grammar: { dtype: int64, rank: 1, shape: [batch] }
          outputs:
            length: { dtype: int64, rank: 1, shape: [1] }
      adaptive:
        implementation: { kind: onnx, artifact: adaptive.onnx.textproto }
        ports:
          inputs:
            current_k: { dtype: int64, rank: 1, shape: [batch] }
            accepted: { dtype: int64, rank: 1, shape: [batch] }
            evaluated: { dtype: int64, rank: 1, shape: [batch] }
            committed_tokens: { dtype: int64, rank: 1, shape: [batch] }
            filled_proposal_budget: { dtype: bool, rank: 1, shape: [batch] }
            draft_ms: { dtype: float32, rank: 1, shape: [batch] }
            target_ms: { dtype: float32, rank: 1, shape: [batch] }
            estimates: { dtype: float32, rank: 2, shape: [batch, budget_slots] }
          outputs:
            next_k: { dtype: int64, rank: 1, shape: [batch] }
            next_estimates: { dtype: float32, rank: 2, shape: [batch, budget_slots] }
        contract:
          id: onnx-genai.adaptive-proposal-budget
          version: "1"
          bindings:
            current_k: current_k
            accepted: accepted
            evaluated: evaluated
            committed_tokens: committed_tokens
            filled_proposal_budget: filled_proposal_budget
            draft_ms: draft_ms
            target_ms: target_ms
            estimates: estimates
            next_k: next_k
            next_estimates: next_estimates
      guided_sampler:
        implementation: { kind: onnx, artifact: guided.onnx.textproto }
        ports:
          inputs:
            logits: { dtype: float32, rank: 2, shape: [batch, vocabulary] }
            logits_mask: { dtype: bool, rank: 2, shape: [batch, vocabulary] }
            forced_tokens: { dtype: int64, rank: 2, shape: [batch, 1] }
            forced_length: { dtype: int64, rank: 1, shape: [batch] }
          outputs:
            token: { dtype: int64, rank: 2, shape: [batch, 1] }
    state:
      grammar:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        class: semantic
        scope: invocation
        initializer: grammar.current
        recurrence: { kind: invariant }
      adaptive:
        contract: { dtype: float32, rank: 2, shape: [batch, budget_slots] }
        class: advisory
        scope: invocation
        initializer: adaptive.current
        recurrence: { kind: invariant }
      proposal_k:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        class: advisory
        scope: invocation
        initializer: k.current
        recurrence: { kind: invariant }
    steps:
        - kind: loop
          setup:
            - kind: sequence
              steps:
              - kind: invoke
                component: bind_grammar
                inputs: { value: grammar_state }
                outputs: { value: grammar.current }
              - kind: invoke
                component: bind_estimates
                inputs: { value: estimates }
                outputs: { value: adaptive.current }
              - kind: invoke
                component: bind_k
                inputs: { value: current_k }
                outputs: { value: k.current }
          steps:
              - kind: invoke
                component: grammar_clone
                inputs: { state: grammar, tokens: proposed, valid_length: zero_length,
                          transition_table: transition_table }
                outputs: { next_state: grammar.clone, consumed_length: clone.consumed,
                           logits_mask: clone.mask, forced_tokens: clone.forced,
                           forced_length: clone.forced_length }
              - kind: invoke
                component: grammar_lookahead
                inputs: { state: grammar.clone, tokens: proposed, valid_length: evaluated,
                          transition_table: transition_table }
                outputs: { next_state: grammar.lookahead, consumed_length: grammar.valid,
                           logits_mask: lookahead.mask, forced_tokens: lookahead.forced,
                           forced_length: lookahead.forced_length }
              - kind: invoke
                component: verifier
                inputs: { target_scores: target, proposed_tokens: proposed }
                outputs: { accepted_tokens: accepted.tokens, accepted_count: verifier.accepted,
                           done: verifier.done }
              - kind: invoke
                component: min_length
                inputs: { accepted: verifier.accepted, grammar: grammar.valid }
                outputs: { length: committed.length }
              - kind: invoke
                component: grammar_commit
                inputs: { state: grammar, tokens: accepted.tokens,
                          valid_length: committed.length, transition_table: transition_table }
                outputs: { next_state: grammar.next, consumed_length: grammar.committed,
                           logits_mask: grammar.mask, forced_tokens: grammar.forced,
                           forced_length: grammar.forced_length }
              - kind: invoke
                component: adaptive
                inputs: { current_k: proposal_k, accepted: committed.length,
                          evaluated: evaluated, committed_tokens: committed.length,
                          filled_proposal_budget: filled, draft_ms: draft_ms,
                          target_ms: target_ms, estimates: adaptive }
                outputs: { next_k: k.next, next_estimates: adaptive.next }
              - kind: invoke
                component: guided_sampler
                inputs: { logits: logits, logits_mask: grammar.mask,
                          forced_tokens: grammar.forced,
                          forced_length: grammar.forced_length }
                outputs: { token: grammar.token }
              - kind: emit
                value: accepted.tokens
                valid_length: committed.length
                output: tokens
                mode: append
              - kind: emit
                value: grammar.token
                output: tokens
                mode: append
          continue_when: continue
          max_iterations: iterations
          carried:
            - cell: grammar
              initial: grammar.current
              next: grammar.next
            - cell: adaptive
              initial: adaptive.current
              next: adaptive.next
            - cell: proposal_k
              initial: k.current
              next: k.next
        - kind: emit
          value: proposal_k
          output: next_k
          mode: replace
        - kind: emit
          value: grammar
          output: final_grammar_state
          mode: replace
"#;
    let root = package(
        "grammar-adaptive-speculative",
        metadata,
        &[
            ("verifier.onnx.textproto", SPECULATIVE_ACCEPTANCE),
            ("min.onnx.textproto", MIN_LENGTH),
            ("adaptive.onnx.textproto", ADAPTIVE_PROPOSAL_BUDGET),
            ("guided.onnx.textproto", GUIDED_SAMPLER),
        ],
    )?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let target = [
        0.0, 1.0, 0.0, 0.0, 0.0, // token 1
        0.0, 0.0, 1.0, 0.0, 0.0, // token 2
        0.0, 0.0, 0.0, 0.0, 1.0, // token 4
    ];
    let transitions = [
        -1, 1, -1, -1, -1, // state 0: token 1
        -1, -1, 2, -1, -1, // state 1: token 2
        -1, -1, -1, 3, -1, // state 2: token 3
        -1, -1, -1, -1, 3, // state 3: token 4
    ];
    let request = |current_k: i64,
                   draft_ms: f32,
                   target_ms: f32|
     -> anyhow::Result<PipelineGenerateRequest> {
        Ok(
            PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
                .with_input("proposed", Value::from_slice_i64(&[1, 2, 4], &[1, 3])?)
                .with_input("target", Value::from_slice_f32(&target, &[1, 3, 5])?)
                .with_input(
                    "logits",
                    Value::from_slice_f32(&[9.0, 8.0, 7.0, 1.0, 6.0], &[1, 5])?,
                )
                .with_input("grammar_state", Value::from_slice_i64(&[0], &[1])?)
                .with_input(
                    "transition_table",
                    Value::from_slice_i64(&transitions, &[4, 5])?,
                )
                .with_input("zero_length", Value::from_slice_i64(&[0], &[1])?)
                .with_input("evaluated", Value::from_slice_i64(&[3], &[1])?)
                .with_input("current_k", Value::from_slice_i64(&[current_k], &[1])?)
                .with_input(
                    "estimates",
                    Value::from_slice_f32(&[0.0, 0.0, 0.0, 0.0], &[1, 4])?,
                )
                .with_input(
                    "filled",
                    Value::from_raw_bytes(vec![1], &[1], onnx_genai_ort::DataType::Bool)?,
                )
                .with_input("draft_ms", Value::from_slice_f32(&[draft_ms], &[1])?)
                .with_input("target_ms", Value::from_slice_f32(&[target_ms], &[1])?)
                .with_input(
                    "continue",
                    Value::from_raw_bytes(vec![1], &[], onnx_genai_ort::DataType::Bool)?,
                )
                .with_input("iterations", Value::from_slice_i64(&[1], &[])?),
        )
    };
    let output = engine.run_pipeline(request(2, 1.0, 2.0)?)?;
    let changed_k = engine.run_pipeline(request(4, 1.0, 2.0)?)?;
    let target_faster = engine.run_pipeline(request(4, 3.0, 2.0)?)?;
    assert_eq!(output["tokens.row.0"].shape(), [1, 3]);
    assert_eq!(output["tokens.row.0"].to_vec_i64()?, [1, 2, 3]);
    assert_eq!(output["next_k"].to_vec_i64()?, [3]);
    assert_eq!(output["final_grammar_state"].to_vec_i64()?, [2]);
    assert_eq!(changed_k["next_k"].to_vec_i64()?, [5]);
    assert_eq!(target_faster["next_k"].to_vec_i64()?, [4]);
    for changed in [&changed_k, &target_faster] {
        assert_eq!(
            changed["tokens.row.0"].to_vec_i64()?,
            output["tokens.row.0"].to_vec_i64()?,
            "advisory K and telemetry must not change semantic token distribution"
        );
        assert_eq!(
            changed["final_grammar_state"].to_vec_i64()?,
            output["final_grammar_state"].to_vec_i64()?
        );
    }
    let islands = engine.execution_island_diagnostics();
    assert!(
        islands.iter().any(|island| {
            island.components == ["verifier", "min_length"]
                && island.component_boundaries_elided == 1
        }),
        "{islands:?}"
    );
    assert!(islands.iter().any(|island| {
        island.components == ["adaptive", "guided_sampler"]
            && island.component_boundaries_elided == 1
    }));
    for island in islands.iter().filter(|island| island.capture_eligible) {
        assert_eq!(island.captures, 1);
        assert!(island.replays >= 1);
        assert_eq!(island.fallback_reason, None);
    }
    Ok(())
}

#[test]
fn workflow_executes_generic_telemetry_adapter() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: { onnx-genai.telemetry: "1" }
      custom_op_versions: {}
      capabilities: [workflow_ssa, typed_emit, telemetry_adapter]
    inputs: {}
    outputs:
      elapsed_ms: { contract: { dtype: float32, rank: 0, shape: [] },
                    role: tensor, stage: pre_adapter }
    components:
      clock_start:
        implementation: { kind: adapter, abi: onnx-genai.telemetry, version: "1" }
        contract:
          id: onnx-genai.telemetry
          version: "1"
          bindings:
            timestamp: timestamp
          parameters:
            action: start
        ports:
          inputs: {}
          outputs: { timestamp: { dtype: int64, rank: 0, shape: [] } }
        effects: [telemetry]
      clock_elapsed:
        implementation: { kind: adapter, abi: onnx-genai.telemetry, version: "1" }
        contract:
          id: onnx-genai.telemetry
          version: "1"
          bindings:
            timestamp: timestamp
            duration_ms: duration_ms
          parameters:
            action: elapsed
        ports:
          inputs: { timestamp: { dtype: int64, rank: 0, shape: [] } }
          outputs: { duration_ms: { dtype: float32, rank: 0, shape: [] } }
        effects: [telemetry]
    steps:
        - kind: invoke
          component: clock_start
          inputs: {}
          outputs: { timestamp: clock.started }
        - kind: invoke
          component: clock_elapsed
          inputs: { timestamp: clock.started }
          outputs: { duration_ms: clock.elapsed_ms }
        - kind: emit
          value: clock.elapsed_ms
          output: elapsed_ms
          mode: replace
"#;
    let root = package("telemetry-adapter", metadata, &[])?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])));
    let output = engine.run_pipeline(request)?;
    assert!(output["elapsed_ms"].to_vec_f32()?[0] >= 0.0);
    Ok(())
}

#[test]
fn workflow_threads_loop_branch_effects_and_session_state() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities:
        [workflow_ssa, typed_emit, streaming_emit,
         nested_control_flow, session_state_lease]
    inputs:
      initial: { contract: { dtype: int64, rank: 0, shape: [] }, role: { kind: opaque },
                 source: { kind: application, name: initial }, required: true }
      run_branch: { contract: { dtype: bool, rank: 0, shape: [] }, role: { kind: opaque },
                    source: { kind: application, name: run_branch }, required: true }
      increment: { contract: { dtype: int64, rank: 0, shape: [] }, role: { kind: opaque },
                   source: { kind: application, name: increment }, required: true }
      limit: { contract: { dtype: int64, rank: 0, shape: [] }, role: { kind: opaque },
               source: { kind: application, name: limit }, required: true }
      iterations:
        contract: { dtype: int64, rank: 0, shape: [] }
        role: { kind: runtime, version: v1, role: max_iterations }
        source: { kind: request }
        required: true
      initial_continue: { contract: { dtype: bool, rank: 0, shape: [] },
                          role: { kind: opaque },
                          source: { kind: application, name: initial_continue }, required: true }
    outputs:
      state: { contract: { dtype: int64, rank: 0, shape: [] }, role: tensor, stage: pre_adapter }
      events: { contract: { dtype: int64, rank: 0, shape: [] }, role: event, stage: pre_adapter }
    components:
      binding:
        implementation: { kind: binding }
        ports:
          inputs: { value: { dtype: int64, rank: 0, shape: [] } }
          outputs: { value: { dtype: int64, rank: 0, shape: [] } }
      update:
        implementation: { kind: onnx, artifact: update.onnx.textproto }
        ports:
          inputs:
            current: { dtype: int64, rank: 0, shape: [] }
            update: { dtype: int64, rank: 0, shape: [] }
          outputs:
            next: { dtype: int64, rank: 0, shape: [] }
        contract:
          id: onnx-genai.state-update
          version: "1"
          bindings:
            current: current
            update: update
            next: next
      predicate:
        implementation: { kind: onnx, artifact: less.onnx.textproto }
        ports:
          inputs:
            value: { dtype: int64, rank: 0, shape: [] }
            limit: { dtype: int64, rank: 0, shape: [] }
          outputs:
            continue: { dtype: bool, rank: 0, shape: [] }
    state:
      world:
        contract: { dtype: int64, rank: 0, shape: [] }
        scope: session
        initializer: initial
        recurrence: { kind: invariant }
        session: { policy: exclusive }
      active:
        contract: { dtype: bool, rank: 0, shape: [] }
        scope: invocation
        initializer: initial_continue
        recurrence: { kind: invariant }
    steps:
        - kind: branch
          predicate: run_branch
          cases:
            "true":
              kind: loop
              setup:
                - kind: invoke
                  component: binding
                  inputs: { value: initial }
                  outputs: { value: world.current }
              steps:
                  - kind: invoke
                    component: update
                    inputs: { current: world, update: increment }
                    outputs: { next: world.body_next }
                  - kind: invoke
                    component: predicate
                    inputs: { value: world.body_next, limit: limit }
                    outputs: { continue: loop.continue }
                  - kind: emit
                    value: world.body_next
                    output: events
                    mode: event
              continue_when: active
              max_iterations: iterations
              carried:
                - cell: world
                  initial: world.current
                  next: world.body_next
                - cell: active
                  next: loop.continue
          outputs:
            world.selected:
              cases: { "true": world }
        - kind: branch
          predicate: world.selected
          cases:
            "3":
              kind: emit
              value: world.selected
              output: state
              mode: replace
            "5":
              kind: emit
              value: world.selected
              output: state
              mode: replace
          default:
            kind: emit
            value: world.selected
            output: state
            mode: replace
"#;
    let invalid = metadata.replace(
        r#"          outputs:
            world.selected:
              cases: { "true": world }
"#,
        "          outputs: {}\n",
    );
    let invalid_root = package(
        "world-missing-state-phi",
        &invalid,
        &[
            ("update.onnx.textproto", ADD_STATE),
            ("less.onnx.textproto", LESS),
        ],
    )?;
    let error = Engine::from_pipeline_dir(&invalid_root, EngineConfig::default())
        .err()
        .expect("branch-local session state must escape through a phi output");
    assert!(
        error
            .to_string()
            .contains("updates session state 'world' to 'world'"),
        "{error}"
    );

    let root = package(
        "world",
        metadata,
        &[
            ("update.onnx.textproto", ADD_STATE),
            ("less.onnx.textproto", LESS),
        ],
    )?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let first_options = onnx_genai_engine::GenerateOptions {
        max_new_tokens: 4,
        ..Default::default()
    };
    let first = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: first_options,
    })
    .with_session_id("world-a")
    .with_input(
        "run_branch",
        Value::from_raw_bytes(vec![1], &[], onnx_genai_ort::DataType::Bool)?,
    )
    .with_input(
        "initial_continue",
        Value::from_raw_bytes(vec![1], &[], DataType::Bool)?,
    )
    .with_input("initial", Value::from_slice_i64(&[0], &[])?)
    .with_input("increment", Value::from_slice_i64(&[1], &[])?)
    .with_input("limit", Value::from_slice_i64(&[3], &[])?);
    assert_eq!(engine.run_pipeline(first)?["state"].to_vec_i64()?, [3]);

    let second_options = onnx_genai_engine::GenerateOptions {
        max_new_tokens: 1,
        ..Default::default()
    };
    let second = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: second_options,
    })
    .with_session_id("world-a")
    .with_input(
        "run_branch",
        Value::from_raw_bytes(vec![1], &[], onnx_genai_ort::DataType::Bool)?,
    )
    .with_input(
        "initial_continue",
        Value::from_raw_bytes(vec![1], &[], DataType::Bool)?,
    )
    .with_input("initial", Value::from_slice_i64(&[0], &[])?)
    .with_input("increment", Value::from_slice_i64(&[1], &[])?)
    .with_input("limit", Value::from_slice_i64(&[5], &[])?);
    assert_eq!(engine.run_pipeline(second)?["state"].to_vec_i64()?, [4]);

    let third_options = onnx_genai_engine::GenerateOptions {
        max_new_tokens: 1,
        ..Default::default()
    };
    let third = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: third_options,
    })
    .with_session_id("world-a")
    .with_input(
        "run_branch",
        Value::from_raw_bytes(vec![1], &[], onnx_genai_ort::DataType::Bool)?,
    )
    .with_input(
        "initial_continue",
        Value::from_raw_bytes(vec![1], &[], DataType::Bool)?,
    )
    .with_input("initial", Value::from_slice_i64(&[0], &[])?)
    .with_input("increment", Value::from_slice_i64(&[1], &[])?)
    .with_input("limit", Value::from_slice_i64(&[5], &[])?);
    assert_eq!(engine.run_pipeline(third)?["state"].to_vec_i64()?, [5]);
    Ok(())
}

#[test]
fn workflow_world_model_checkpoints_and_replays_semantic_state() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities:
        [workflow_ssa, typed_emit, streaming_emit,
         nested_control_flow, session_state_lease, advisory_state]
    inputs:
      initial: { contract: { dtype: int64, rank: 0, shape: [] },
                 role: { kind: opaque }, source: { kind: application, name: initial },
                 required: true }
      observation: { contract: { dtype: int64, rank: 0, shape: [] },
                     role: { kind: opaque },
                     source: { kind: application, name: observation }, required: true }
      action_threshold: { contract: { dtype: int64, rank: 0, shape: [] },
                          role: { kind: opaque },
                          source: { kind: application, name: action_threshold }, required: true }
      low_delta: { contract: { dtype: int64, rank: 0, shape: [] },
                   role: { kind: opaque }, source: { kind: application, name: low_delta },
                   required: true }
      high_delta: { contract: { dtype: int64, rank: 0, shape: [] },
                    role: { kind: opaque }, source: { kind: application, name: high_delta },
                    required: true }
      continue: { contract: { dtype: bool, rank: 0, shape: [] },
                  role: { kind: opaque }, source: { kind: application, name: continue },
                  required: true }
      iterations:
        contract: { dtype: int64, rank: 0, shape: [] }
        role: { kind: runtime, version: v1, role: max_iterations }
        source: { kind: request }
        required: true
    outputs:
      latent: { contract: { dtype: int64, rank: 0, shape: [] },
                role: tensor, stage: pre_adapter }
      advisory_count: { contract: { dtype: int64, rank: 0, shape: [] },
                        role: tensor, stage: pre_adapter }
      actions: { contract: { dtype: bool, rank: 0, shape: [] },
                 role: event, stage: pre_adapter }
    components:
      bind_state:
        implementation: { kind: binding }
        ports:
          inputs: { value: { dtype: int64, rank: 0, shape: [] } }
          outputs: { value: { dtype: int64, rank: 0, shape: [] } }
        effects: []
      observation_encoder:
        implementation: { kind: onnx, artifact: observation.onnx.textproto }
        ports:
          inputs:
            current: { dtype: int64, rank: 0, shape: [] }
            update: { dtype: int64, rank: 0, shape: [] }
          outputs:
            next: { dtype: int64, rank: 0, shape: [] }
      action_policy:
        implementation: { kind: onnx, artifact: action.onnx.textproto }
        ports:
          inputs:
            value: { dtype: int64, rank: 0, shape: [] }
            limit: { dtype: int64, rank: 0, shape: [] }
          outputs:
            continue: { dtype: bool, rank: 0, shape: [] }
      environment_low:
        implementation: { kind: onnx, artifact: environment-low.onnx.textproto }
        ports:
          inputs:
            current: { dtype: int64, rank: 0, shape: [] }
            update: { dtype: int64, rank: 0, shape: [] }
          outputs:
            next: { dtype: int64, rank: 0, shape: [] }
      environment_high:
        implementation: { kind: onnx, artifact: environment-high.onnx.textproto }
        ports:
          inputs:
            current: { dtype: int64, rank: 0, shape: [] }
            update: { dtype: int64, rank: 0, shape: [] }
          outputs:
            next: { dtype: int64, rank: 0, shape: [] }
      advisory_counter:
        implementation: { kind: onnx, artifact: advisory-counter.onnx.textproto }
        ports:
          inputs:
            current: { dtype: int64, rank: 0, shape: [] }
            update: { dtype: int64, rank: 0, shape: [] }
          outputs:
            next: { dtype: int64, rank: 0, shape: [] }
    state:
      latent:
        contract: { dtype: int64, rank: 0, shape: [] }
        class: semantic
        scope: session
        initializer: initial
        recurrence: { kind: invariant }
        session: { policy: exclusive }
      advisory_steps:
        contract: { dtype: int64, rank: 0, shape: [] }
        class: advisory
        scope: session
        initializer: initial
        recurrence: { kind: invariant }
        session: { policy: exclusive }
    steps:
        - kind: loop
          setup:
            - kind: sequence
              steps:
                - kind: invoke
                  component: bind_state
                  inputs: { value: initial }
                  outputs: { value: latent.current }
                - kind: invoke
                  component: bind_state
                  inputs: { value: initial }
                  outputs: { value: advisory.current }
          steps:
              - kind: invoke
                component: observation_encoder
                inputs: { current: latent, update: observation }
                outputs: { next: latent.observed }
              - kind: invoke
                component: action_policy
                inputs: { value: latent.observed, limit: action_threshold }
                outputs: { continue: action.selected }
              - kind: emit
                value: action.selected
                output: actions
                mode: event
              - kind: branch
                predicate: action.selected
                cases:
                  "true":
                    kind: invoke
                    component: environment_low
                    inputs: { current: latent.observed, update: low_delta }
                    outputs: { next: environment.low }
                  "false":
                    kind: invoke
                    component: environment_high
                    inputs: { current: latent.observed, update: high_delta }
                    outputs: { next: environment.high }
                outputs:
                  latent.next:
                    cases: { "true": environment.low, "false": environment.high }
              - kind: invoke
                component: advisory_counter
                inputs: { current: advisory_steps, update: low_delta }
                outputs: { next: advisory.next }
          continue_when: continue
          max_iterations: iterations
          carried:
            - cell: latent
              initial: latent.current
              next: latent.next
            - cell: advisory_steps
              initial: advisory.current
              next: advisory.next
        - kind: emit
          value: latent
          output: latent
          mode: replace
        - kind: emit
          value: advisory_steps
          output: advisory_count
          mode: replace
"#;
    let root = package(
        "world-model-checkpoint",
        metadata,
        &[
            ("observation.onnx.textproto", ADD_STATE),
            ("action.onnx.textproto", LESS),
            ("environment-low.onnx.textproto", ADD_STATE),
            ("environment-high.onnx.textproto", ADD_STATE),
            ("advisory-counter.onnx.textproto", ADD_STATE),
        ],
    )?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;

    let request = |iterations, observation| {
        let options = onnx_genai_engine::GenerateOptions {
            max_new_tokens: iterations,
            ..Default::default()
        };
        PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![]),
            options,
        })
        .with_session_id("world-checkpoint")
        .with_input("initial", Value::from_slice_i64(&[0], &[]).unwrap())
        .with_input(
            "observation",
            Value::from_slice_i64(&[observation], &[]).unwrap(),
        )
        .with_input(
            "action_threshold",
            Value::from_slice_i64(&[3], &[]).unwrap(),
        )
        .with_input("low_delta", Value::from_slice_i64(&[1], &[]).unwrap())
        .with_input("high_delta", Value::from_slice_i64(&[2], &[]).unwrap())
        .with_input(
            "continue",
            Value::from_raw_bytes(vec![1], &[], onnx_genai_ort::DataType::Bool).unwrap(),
        )
    };

    let first = engine.run_pipeline(request(2, 1))?;
    assert_eq!(first["latent"].to_vec_i64()?, [5]);
    assert_eq!(first["advisory_count"].to_vec_i64()?, [2]);
    assert_eq!(first["actions"].as_raw_bytes()?, [0]);
    let checkpoint = engine.checkpoint_session("world-checkpoint")?;

    let advanced = engine.run_pipeline(request(1, 2))?;
    assert_eq!(advanced["latent"].to_vec_i64()?, [9]);
    assert_eq!(advanced["advisory_count"].to_vec_i64()?, [3]);
    assert_eq!(advanced["actions"].as_raw_bytes()?, [0]);

    engine.restore_session_checkpoint("world-checkpoint", &checkpoint)?;
    let replayed = engine.run_pipeline(request(1, 2))?;
    assert_eq!(replayed["latent"].to_vec_i64()?, [9]);
    assert_eq!(
        replayed["advisory_count"].to_vec_i64()?,
        [4],
        "advisory session state is intentionally excluded from semantic checkpoints"
    );
    assert_eq!(
        replayed["actions"].as_raw_bytes()?,
        advanced["actions"].as_raw_bytes()?
    );
    Ok(())
}

#[test]
fn workflow_branch_joins_speculative_values_and_effects() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 24 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, typed_emit, streaming_emit, nested_control_flow]
    inputs:
      accept:
        contract: { dtype: bool, rank: 0, shape: [] }
        role: { kind: opaque }
        source: { kind: application, name: accept }
        required: true
      accepted_tokens:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: { kind: opaque }
        source: { kind: application, name: accepted_tokens }
        required: true
      corrected_tokens:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: { kind: opaque }
        source: { kind: application, name: corrected_tokens }
        required: true
      accepted_kv:
        contract: { dtype: float32, rank: 2, shape: [batch, cache] }
        role: { kind: opaque }
        source: { kind: application, name: accepted_kv }
        required: true
      corrected_kv:
        contract: { dtype: float32, rank: 2, shape: [batch, cache] }
        role: { kind: opaque }
        source: { kind: application, name: corrected_kv }
        required: true
      accepted_rng:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: { kind: opaque }
        source: { kind: application, name: accepted_rng }
        required: true
      corrected_rng:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: { kind: opaque }
        source: { kind: application, name: corrected_rng }
        required: true
    outputs:
      event:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: event
        stage: pre_adapter
      final:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: tokens
        stage: pre_adapter
    components:
      token_binding:
        implementation: { kind: binding }
        ports:
          inputs: { value: { dtype: int64, rank: 1, shape: [batch] } }
          outputs: { value: { dtype: int64, rank: 1, shape: [batch] } }
        effects: [speculative]
      plain_token_binding:
        implementation: { kind: binding }
        ports:
          inputs: { value: { dtype: int64, rank: 1, shape: [batch] } }
          outputs: { value: { dtype: int64, rank: 1, shape: [batch] } }
        effects: []
      kv_binding:
        implementation: { kind: binding }
        ports:
          inputs: { value: { dtype: float32, rank: 2, shape: [batch, cache] } }
          outputs: { value: { dtype: float32, rank: 2, shape: [batch, cache] } }
        effects: [kv]
      rng_binding:
        implementation: { kind: binding }
        ports:
          inputs: { value: { dtype: int64, rank: 1, shape: [batch] } }
          outputs: { value: { dtype: int64, rank: 1, shape: [batch] } }
        effects: [rng]
    steps:
        - kind: branch
          predicate: accept
          cases:
            "true":
              kind: sequence
              steps:
                - kind: invoke
                  component: token_binding
                  inputs: { value: accepted_tokens }
                  outputs: { value: accepted.tokens }
                - kind: invoke
                  component: kv_binding
                  inputs: { value: accepted_kv }
                  outputs: { value: accepted.kv }
                - kind: invoke
                  component: rng_binding
                  inputs: { value: accepted_rng }
                  outputs: { value: accepted.rng }
                - kind: emit
                  value: accepted.tokens
                  output: event
                  mode: event
                - kind: invoke
                  component: plain_token_binding
                  inputs: { value: accepted_tokens }
                  outputs: { value: event.secret }
            "false":
              kind: sequence
              steps:
                - kind: invoke
                  component: token_binding
                  inputs: { value: corrected_tokens }
                  outputs: { value: corrected.tokens }
                - kind: invoke
                  component: kv_binding
                  inputs: { value: corrected_kv }
                  outputs: { value: corrected.kv }
                - kind: invoke
                  component: rng_binding
                  inputs: { value: corrected_rng }
                  outputs: { value: corrected.rng }
                - kind: emit
                  value: corrected.tokens
                  output: event
                  mode: event
                - kind: invoke
                  component: plain_token_binding
                  inputs: { value: corrected_tokens }
                  outputs: { value: event.secret }
          outputs:
            selected.tokens:
              cases: { "true": accepted.tokens, "false": corrected.tokens }
            selected.kv:
              cases: { "true": accepted.kv, "false": corrected.kv }
            selected.rng:
              cases: { "true": accepted.rng, "false": corrected.rng }
        - kind: emit
          value: selected.tokens
          output: final
          mode: replace
"#;
    let root = package("speculative-branch", metadata, &[])?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;

    for (accept, tokens) in [(true, vec![11]), (false, vec![22])] {
        let request = PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![]),
            options: Default::default(),
        })
        .with_input(
            "accept",
            Value::from_raw_bytes(vec![u8::from(accept)], &[], onnx_genai_ort::DataType::Bool)?,
        )
        .with_input("accepted_tokens", Value::from_slice_i64(&[11], &[1])?)
        .with_input("corrected_tokens", Value::from_slice_i64(&[22], &[1])?)
        .with_input("accepted_kv", Value::from_slice_f32(&[1.0, 2.0], &[1, 2])?)
        .with_input("corrected_kv", Value::from_slice_f32(&[3.0, 4.0], &[1, 2])?)
        .with_input("accepted_rng", Value::from_slice_i64(&[4], &[1])?)
        .with_input("corrected_rng", Value::from_slice_i64(&[8], &[1])?);
        let outputs = engine.run_pipeline(request)?;

        assert_eq!(outputs["final"].to_vec_i64()?, tokens);
        assert_eq!(outputs["event.0"].to_vec_i64()?, tokens);
        assert!(!outputs.contains_key("selected.kv"));
        assert!(!outputs.contains_key("selected.rng"));
        assert!(!outputs.contains_key("event.secret"));
        assert!(
            !outputs.contains_key(if accept {
                "corrected.tokens"
            } else {
                "accepted.tokens"
            }),
            "case-local SSA must not escape"
        );
    }
    Ok(())
}
