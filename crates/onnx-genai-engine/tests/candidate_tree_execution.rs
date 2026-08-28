use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use onnx_genai_engine::pipeline::speculative::CandidateTreeGenerationCancelled;
use onnx_genai_engine::pipeline::{
    OutputFinality, TypedRevisionOperation, WorkflowOutputPublication,
};
use onnx_genai_engine::{
    Engine, EngineConfig, FinishReason, GenerateOptions, GeneratePrompt, GenerateRequest,
    GenerationBoundary, GenerationControl, PackageExecutionError, PipelineGenerateRequest,
    SessionForkError, SessionPosition, ToolCallPolicy, package_execution_error,
};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};

const PARENT_METADATA: &str = r#"
schema_version: v1.6
pipeline:
  workflow:
    manifest: {}
    inputs:
      request.tokens:
        contract: {dtype: int64, shape: [batch, sequence], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: runtime, version: "1.0", role: prompt_tokens}
        source: {kind: request}
      request.proposer_state:
        contract: {dtype: float32, shape: [batch, 2], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0.0
      request.target_state:
        contract: {dtype: float32, shape: [batch, 2], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0.0
      request.token_context:
        contract: {dtype: int64, shape: [batch, 4], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0
      runtime.ancestor_mask:
        contract: {dtype: bool, shape: [batch, 6, 6], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: false
      runtime.position_ids:
        contract: {dtype: int64, shape: [batch, 6], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0
      runtime.accepted_tokens:
        contract: {dtype: int64, shape: [batch, 0], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0
      runtime.active:
        contract: {dtype: bool, shape: [batch], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: true
      runtime.done:
        contract: {dtype: bool, shape: [batch], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: false
      runtime.accepted_len:
        contract: {dtype: int64, shape: [batch], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0
    outputs:
      tokens:
        contract: {dtype: int64, shape: [1]}
        role: tokens
        family: {kind: events}
        stage: pre_adapter
    components:
      proposer:
        implementation: {kind: onnx, artifact: proposer.onnx.textproto}
        ports:
          inputs:
            context_tokens: {dtype: int64, shape: [batch, sequence], batch_layout: {kind: request_aligned, axis: 0}}
            private_state: {dtype: float32, shape: [batch, 2], batch_layout: {kind: request_aligned, axis: 0}}
          outputs:
            candidate_tokens: {dtype: int64, shape: [batch, 6], batch_layout: {kind: request_aligned, axis: 0}}
            candidate_parents: {dtype: int64, shape: [batch, 6], batch_layout: {kind: request_aligned, axis: 0}}
            proposal_probabilities: {dtype: float32, shape: [batch, 7, 6], batch_layout: {kind: request_aligned, axis: 0}}
            private_state_out: {dtype: float32, shape: [batch, 2], batch_layout: {kind: request_aligned, axis: 0}}
        row_scope: {axis: 0, stateful: true}
      target:
        implementation: {kind: onnx, artifact: target.onnx.textproto}
        ports:
          inputs:
            context_tokens: {dtype: int64, shape: [batch, sequence], batch_layout: {kind: request_aligned, axis: 0}}
            candidate_tokens: {dtype: int64, shape: [batch, 6], batch_layout: {kind: request_aligned, axis: 0}}
            ancestor_mask: {dtype: bool, shape: [batch, 6, 6], batch_layout: {kind: request_aligned, axis: 0}}
            position_ids: {dtype: int64, shape: [batch, 6], batch_layout: {kind: request_aligned, axis: 0}}
            accepted_tokens: {dtype: int64, shape: [batch, accepted], batch_layout: {kind: request_aligned, axis: 0}}
            target_state: {dtype: float32, shape: [batch, 2], batch_layout: {kind: request_aligned, axis: 0}}
            token_context: {dtype: int64, shape: [batch, 4], batch_layout: {kind: request_aligned, axis: 0}}
          outputs:
            target_logits: {dtype: float32, shape: [batch, 7, 6], batch_layout: {kind: request_aligned, axis: 0}}
            target_probabilities: {dtype: float32, shape: [batch, 7, 6], batch_layout: {kind: request_aligned, axis: 0}}
            target_state_out: {dtype: float32, shape: [batch, 2], batch_layout: {kind: request_aligned, axis: 0}}
            token_context_out: {dtype: int64, shape: [batch, 4], batch_layout: {kind: request_aligned, axis: 0}}
          roles:
            target_logits: logits
        row_scope: {axis: 0, stateful: true}
    steps:
    - kind: invoke
      component: proposer
      inputs: {context_tokens: request.tokens, private_state: request.proposer_state}
      outputs:
        candidate_tokens: proposed.tokens
        candidate_parents: proposed.parents
        proposal_probabilities: proposed.probabilities
        private_state_out: proposed.state
    - kind: invoke
      component: target
      inputs:
        context_tokens: request.tokens
        candidate_tokens: proposed.tokens
        ancestor_mask: runtime.ancestor_mask
        position_ids: runtime.position_ids
        accepted_tokens: runtime.accepted_tokens
        target_state: request.target_state
        token_context: request.token_context
      outputs:
        target_logits: verified.logits
        target_probabilities: verified.probabilities
        target_state_out: verified.state
        token_context_out: verified.token_context
    state:
      proposer_private:
        contract: {dtype: float32, shape: [batch, 2], batch_layout: {kind: request_aligned, axis: 0}}
        class: semantic
        scope: session
        initializer: request.proposer_state
        recurrence: {kind: invariant}
        management: runtime
        release_boundary: session
        service_group: proposer_private
      target_recurrent:
        contract: {dtype: float32, shape: [batch, 2], batch_layout: {kind: request_aligned, axis: 0}}
        class: semantic
        scope: session
        initializer: request.target_state
        recurrence: {kind: invariant}
        management: runtime
        release_boundary: session
        service_group: target_recurrent
      token_context:
        contract: {dtype: int64, shape: [batch, 4], batch_layout: {kind: request_aligned, axis: 0}}
        class: semantic
        scope: session
        initializer: request.token_context
        recurrence: {kind: invariant}
        management: runtime
        release_boundary: session
        service_group: token_context
    serving:
      active: runtime.active
      done: runtime.done
      accepted_len: runtime.accepted_len
      state_service:
        groups:
          proposer_private:
            kind: recurrent
            layout: bf
            update: {kind: replace}
            capabilities: {rollback_positions: 6, snapshot: true, fork: false}
            ports:
              proposer:
                proposer_private: {input: private_state, output: private_state_out}
          target_recurrent:
            kind: recurrent
            layout: bf
            update: {kind: replace}
            capabilities: {rollback_positions: 6, snapshot: true, fork: false, cascade: [token_context]}
            ports:
              target:
                target_recurrent: {input: target_state, output: target_state_out}
          token_context:
            kind: recurrent
            layout: bf
            update: {kind: replace}
            capabilities: {rollback_positions: 6, snapshot: true, fork: false, cascade: [target_recurrent]}
            ports:
              target:
                token_context: {input: token_context, output: token_context_out}
speculative:
  identity: onnx-genai.speculative
  version: "1"
  proposer: proposer
  target: target
  proposal_execution:
    kind: candidate_tree
    candidate_tokens: candidate_tokens
    topology: {kind: parent_indices, output: candidate_parents}
  port_bindings: {context_tokens: context_tokens}
  target_port_bindings:
    context_tokens: context_tokens
    candidate_tokens: candidate_tokens
    ancestor_mask: ancestor_mask
    position_ids: position_ids
    accepted_tokens: accepted_tokens
  vocabulary: {kind: identical}
  max_proposal_width: 6
  distribution_preserving: true
  verification:
    target_output: {component: target, output: target_logits}
    accepted_path: {kind: runtime, binding: accepted_prefix}
    probabilities:
      proposal: {component: proposer, output: proposal_probabilities}
      target: {component: target, output: target_probabilities}
  rollback_state: [proposer_private, target_recurrent, token_context]
"#;

const PARENT_PROPOSER: &str = r#"
ir_version: 8
graph {
  node { input: "private_state" output: "state_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "state_sum" output: "state_offset" op_type: "Cast" attribute { name: "to" i: 7 type: INT } }
  node { output: "vocabulary" op_type: "Constant" attribute { name: "value" t { data_type: 7 int64_data: 6 } type: TENSOR } }
  node { input: "state_offset" input: "vocabulary" output: "offset" op_type: "Mod" }
  node { output: "candidate_base" op_type: "Constant" attribute { name: "value" t { dims: 1 dims: 6 data_type: 7 int64_data: 1 int64_data: 2 int64_data: 3 int64_data: 4 int64_data: 5 int64_data: 0 } type: TENSOR } }
  node { input: "candidate_base" input: "offset" output: "candidate_shifted" op_type: "Add" }
  node { input: "candidate_shifted" input: "vocabulary" output: "candidate_tokens" op_type: "Mod" }
  node { output: "candidate_parents" op_type: "Constant" attribute { name: "value" t { dims: 1 dims: 6 data_type: 7 {parent_values} } type: TENSOR } }
  node { output: "proposal_probabilities" op_type: "Constant" attribute { name: "value" t { dims: 1 dims: 7 dims: 6 data_type: 1 float_data: 0.0 float_data: 0.5 float_data: 0.5 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 1.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 1.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 1.0 float_data: 1.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 1.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 1.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 } type: TENSOR } }
  node { input: "context_tokens" output: "context_f32" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
  node { input: "context_f32" output: "context_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "private_state" input: "context_sum" output: "private_state_out" op_type: "Add" }
  name: "candidate_parent_proposer"
  input { name: "context_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_param: "sequence" } } } } }
  input { name: "private_state" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 2 } } } } }
  output { name: "candidate_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 6 } } } } }
  output { name: "candidate_parents" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 6 } } } } }
  output { name: "proposal_probabilities" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 7 } dim { dim_value: 6 } } } } }
  output { name: "private_state_out" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 2 } } } } }
}
opset_import { domain: "" version: 18 }
"#;

const PARENT_TARGET: &str = r#"
ir_version: 8
graph {
  node { output: "target_logits" op_type: "Constant" attribute { name: "value" t { dims: 1 dims: 7 dims: 6 data_type: 1 {logit_values} } type: TENSOR } }
  node { input: "target_logits" output: "target_probabilities" op_type: "Softmax" attribute { name: "axis" i: -1 type: INT } }
  node { input: "accepted_tokens" output: "accepted_f32" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
  node { input: "accepted_f32" output: "accepted_sum_f32" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "target_state" input: "accepted_sum_f32" output: "target_state_out" op_type: "Add" }
  node { input: "accepted_tokens" output: "accepted_sum_i64" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "token_context" input: "accepted_sum_i64" output: "token_context_out" op_type: "Add" }
  name: "candidate_parent_target"
  input { name: "context_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_param: "sequence" } } } } }
  input { name: "candidate_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 6 } } } } }
  input { name: "ancestor_mask" type { tensor_type { elem_type: 9 shape { dim { dim_value: 1 } dim { dim_value: 6 } dim { dim_value: 6 } } } } }
  input { name: "position_ids" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 6 } } } } }
  input { name: "accepted_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_param: "accepted" } } } } }
  input { name: "target_state" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 2 } } } } }
  input { name: "token_context" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 4 } } } } }
  output { name: "target_logits" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 7 } dim { dim_value: 6 } } } } }
  output { name: "target_probabilities" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 7 } dim { dim_value: 6 } } } } }
  output { name: "target_state_out" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 2 } } } } }
  output { name: "token_context_out" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 4 } } } } }
}
opset_import { domain: "" version: 18 }
"#;

const ANCESTOR_METADATA: &str = r#"
schema_version: v1.6
pipeline:
  workflow:
    manifest: {}
    inputs:
      prompt:
        contract: {dtype: int64, shape: [batch, sequence], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: runtime, version: "1.0", role: prompt_tokens}
        source: {kind: request}
      proposer_seed:
        contract: {dtype: float32, shape: [batch, 3], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0.0
      target_seed:
        contract: {dtype: float32, shape: [batch, 2, 2], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0.0
      history_seed:
        contract: {dtype: int64, shape: [batch, 5], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0
      mask_seed:
        contract: {dtype: bool, shape: [batch, 5, 5], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: false
      position_seed:
        contract: {dtype: int64, shape: [batch, 5], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0
      accepted_seed:
        contract: {dtype: int64, shape: [batch, 0], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0
      active:
        contract: {dtype: bool, shape: [batch], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: true
      done:
        contract: {dtype: bool, shape: [batch], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: false
      accepted_len:
        contract: {dtype: int64, shape: [batch], batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: opaque}
        source: {kind: literal}
        required: false
        default: 0
    outputs:
      tokens:
        contract: {dtype: int64, shape: [1]}
        role: tokens
        family: {kind: revisions, version: "1"}
        stage: pre_adapter
    components:
      branching_proposer:
        implementation: {kind: onnx, artifact: branching-proposer.onnx.textproto}
        ports:
          inputs:
            history_tokens: {dtype: int64, shape: [batch, sequence], batch_layout: {kind: request_aligned, axis: 0}}
            private_matrix: {dtype: float32, shape: [batch, 3], batch_layout: {kind: request_aligned, axis: 0}}
          outputs:
            branch_ids: {dtype: int64, shape: [batch, 5], batch_layout: {kind: request_aligned, axis: 0}}
            branch_ancestors: {dtype: bool, shape: [batch, 5, 5], batch_layout: {kind: request_aligned, axis: 0}}
            branch_probabilities: {dtype: float32, shape: [batch, 6, 5], batch_layout: {kind: request_aligned, axis: 0}}
            private_matrix_out: {dtype: float32, shape: [batch, 3], batch_layout: {kind: request_aligned, axis: 0}}
        row_scope: {axis: 0, stateful: true}
      branching_target:
        implementation: {kind: onnx, artifact: branching-target.onnx.textproto}
        ports:
          inputs:
            history_tokens: {dtype: int64, shape: [batch, sequence], batch_layout: {kind: request_aligned, axis: 0}}
            branch_ids: {dtype: int64, shape: [batch, 5], batch_layout: {kind: request_aligned, axis: 0}}
            branch_ancestors: {dtype: bool, shape: [batch, 5, 5], batch_layout: {kind: request_aligned, axis: 0}}
            branch_positions: {dtype: int64, shape: [batch, 5], batch_layout: {kind: request_aligned, axis: 0}}
            committed_tokens: {dtype: int64, shape: [batch, accepted], batch_layout: {kind: request_aligned, axis: 0}}
            recurrent_cube: {dtype: float32, shape: [batch, 2, 2], batch_layout: {kind: request_aligned, axis: 0}}
            token_window: {dtype: int64, shape: [batch, 5], batch_layout: {kind: request_aligned, axis: 0}}
          outputs:
            path_logits: {dtype: float32, shape: [batch, 6, 5], batch_layout: {kind: request_aligned, axis: 0}}
            path_probabilities: {dtype: float32, shape: [batch, 6, 5], batch_layout: {kind: request_aligned, axis: 0}}
            recurrent_cube_out: {dtype: float32, shape: [batch, 2, 2], batch_layout: {kind: request_aligned, axis: 0}}
            token_window_out: {dtype: int64, shape: [batch, 5], batch_layout: {kind: request_aligned, axis: 0}}
          roles:
            path_logits: logits
        row_scope: {axis: 0, stateful: true}
    steps:
    - kind: invoke
      component: branching_proposer
      inputs: {history_tokens: prompt, private_matrix: proposer_seed}
      outputs:
        branch_ids: tree.ids
        branch_ancestors: tree.ancestors
        branch_probabilities: tree.probabilities
        private_matrix_out: tree.private
    - kind: invoke
      component: branching_target
      inputs:
        history_tokens: prompt
        branch_ids: tree.ids
        branch_ancestors: tree.ancestors
        branch_positions: position_seed
        committed_tokens: accepted_seed
        recurrent_cube: target_seed
        token_window: history_seed
      outputs:
        path_logits: verification.logits
        path_probabilities: verification.probabilities
        recurrent_cube_out: verification.recurrent
        token_window_out: verification.history
    state:
      proposer_private:
        contract: {dtype: float32, shape: [batch, 3], batch_layout: {kind: request_aligned, axis: 0}}
        class: semantic
        scope: session
        initializer: proposer_seed
        recurrence: {kind: invariant}
        management: runtime
        release_boundary: session
        service_group: proposer_private
      target_recurrent:
        contract: {dtype: float32, shape: [batch, 2, 2], batch_layout: {kind: request_aligned, axis: 0}}
        class: semantic
        scope: session
        initializer: target_seed
        recurrence: {kind: invariant}
        management: runtime
        release_boundary: session
        service_group: target_recurrent
      token_context:
        contract: {dtype: int64, shape: [batch, 5], batch_layout: {kind: request_aligned, axis: 0}}
        class: semantic
        scope: session
        initializer: history_seed
        recurrence: {kind: invariant}
        management: runtime
        release_boundary: session
        service_group: token_context
    serving:
      active: active
      done: done
      accepted_len: accepted_len
      state_service:
        groups:
          proposer_private:
            kind: recurrent
            layout: bf
            update: {kind: replace}
            capabilities: {rollback_positions: 5, snapshot: true, fork: false}
            ports:
              branching_proposer:
                proposer_private: {input: private_matrix, output: private_matrix_out}
          target_recurrent:
            kind: recurrent
            layout: bhf
            update: {kind: replace}
            capabilities: {rollback_positions: 5, snapshot: true, fork: false, cascade: [token_context]}
            ports:
              branching_target:
                target_recurrent: {input: recurrent_cube, output: recurrent_cube_out}
          token_context:
            kind: recurrent
            layout: bf
            update: {kind: replace}
            capabilities: {rollback_positions: 5, snapshot: true, fork: false, cascade: [target_recurrent]}
            ports:
              branching_target:
                token_context: {input: token_window, output: token_window_out}
speculative:
  identity: onnx-genai.speculative
  version: "1"
  proposer: branching_proposer
  target: branching_target
  proposal_execution:
    kind: candidate_tree
    candidate_tokens: branch_ids
    topology: {kind: ancestor_mask, output: branch_ancestors}
  port_bindings: {context_tokens: history_tokens}
  target_port_bindings:
    context_tokens: history_tokens
    candidate_tokens: branch_ids
    ancestor_mask: branch_ancestors
    position_ids: branch_positions
    accepted_tokens: committed_tokens
  vocabulary: {kind: identical}
  max_proposal_width: 5
  distribution_preserving: true
  verification:
    target_output: {component: branching_target, output: path_logits}
    accepted_path: {kind: runtime, binding: selected_branch}
    probabilities:
      proposal: {component: branching_proposer, output: branch_probabilities}
      target: {component: branching_target, output: path_probabilities}
  rollback_state: [proposer_private, target_recurrent, token_context]
"#;

const ANCESTOR_PROPOSER: &str = r#"
ir_version: 8
graph {
  node { output: "branch_ids" op_type: "Constant" attribute { name: "value" t { dims: 1 dims: 5 data_type: 7 int64_data: 0 int64_data: 1 int64_data: 2 int64_data: 3 int64_data: 4 } type: TENSOR } }
  node { output: "branch_ancestors" op_type: "Constant" attribute { name: "value" t { dims: 1 dims: 5 dims: 5 data_type: 9 int32_data: 1 int32_data: 0 int32_data: 0 int32_data: 0 int32_data: 0 int32_data: 0 int32_data: 1 int32_data: 0 int32_data: 0 int32_data: 0 int32_data: 1 int32_data: 0 int32_data: 1 int32_data: 0 int32_data: 0 int32_data: 0 int32_data: 1 int32_data: 0 int32_data: 1 int32_data: 0 int32_data: 0 int32_data: 1 int32_data: 0 int32_data: 1 int32_data: 1 } type: TENSOR } }
  node { output: "branch_probabilities" op_type: "Constant" attribute { name: "value" t { dims: 1 dims: 6 dims: 5 data_type: 1 float_data: 0.25 float_data: 0.75 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 1.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 1.0 float_data: 0.0 float_data: 0.2 float_data: 0.2 float_data: 0.2 float_data: 0.2 float_data: 0.2 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 1.0 float_data: 0.1 float_data: 0.2 float_data: 0.3 float_data: 0.4 float_data: 0.0 } type: TENSOR } }
  node { input: "history_tokens" output: "history_f32" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
  node { input: "history_f32" output: "history_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "private_matrix" input: "history_sum" output: "private_matrix_out" op_type: "Add" }
  name: "candidate_ancestor_proposer"
  input { name: "history_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_param: "sequence" } } } } }
  input { name: "private_matrix" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 3 } } } } }
  output { name: "branch_ids" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 5 } } } } }
  output { name: "branch_ancestors" type { tensor_type { elem_type: 9 shape { dim { dim_value: 1 } dim { dim_value: 5 } dim { dim_value: 5 } } } } }
  output { name: "branch_probabilities" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 6 } dim { dim_value: 5 } } } } }
  output { name: "private_matrix_out" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 3 } } } } }
}
opset_import { domain: "" version: 18 }
"#;

const ANCESTOR_TARGET: &str = r#"
ir_version: 8
graph {
  node { output: "path_probabilities" op_type: "Constant" attribute { name: "value" t { dims: 1 dims: 6 dims: 5 data_type: 1 float_data: 0.75 float_data: 0.25 float_data: 0.0 float_data: 0.0 float_data: 0.0 float_data: 0.1 float_data: 0.1 float_data: 0.6 float_data: 0.1 float_data: 0.1 float_data: 0.1 float_data: 0.1 float_data: 0.1 float_data: 0.6 float_data: 0.1 float_data: 0.05 float_data: 0.15 float_data: 0.3 float_data: 0.4 float_data: 0.1 float_data: 0.1 float_data: 0.1 float_data: 0.1 float_data: 0.1 float_data: 0.6 float_data: 0.4 float_data: 0.3 float_data: 0.2 float_data: 0.1 float_data: 0.0 } type: TENSOR } }
  node { input: "path_probabilities" output: "path_logits" op_type: "Identity" }
  node { input: "committed_tokens" output: "accepted_f32" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
  node { input: "accepted_f32" output: "accepted_sum_f32" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "recurrent_cube" input: "accepted_sum_f32" output: "recurrent_cube_out" op_type: "Add" }
  node { input: "committed_tokens" output: "accepted_sum_i64" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "token_window" input: "accepted_sum_i64" output: "token_window_out" op_type: "Add" }
  name: "candidate_ancestor_target"
  input { name: "history_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_param: "sequence" } } } } }
  input { name: "branch_ids" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 5 } } } } }
  input { name: "branch_ancestors" type { tensor_type { elem_type: 9 shape { dim { dim_value: 1 } dim { dim_value: 5 } dim { dim_value: 5 } } } } }
  input { name: "branch_positions" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 5 } } } } }
  input { name: "committed_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_param: "accepted" } } } } }
  input { name: "recurrent_cube" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 2 } dim { dim_value: 2 } } } } }
  input { name: "token_window" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 5 } } } } }
  output { name: "path_logits" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 6 } dim { dim_value: 5 } } } } }
  output { name: "path_probabilities" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 6 } dim { dim_value: 5 } } } } }
  output { name: "recurrent_cube_out" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 2 } dim { dim_value: 2 } } } } }
  output { name: "token_window_out" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 5 } } } } }
}
opset_import { domain: "" version: 18 }
"#;

const COMPOSED_SETUP: &str = r#"
ir_version: 8
graph {
  node { input: "context_tokens" output: "prepared_context" op_type: "Identity" }
  node { input: "proposer_state" output: "proposer_state_out" op_type: "Identity" }
  node { input: "target_state" output: "target_state_out" op_type: "Identity" }
  node { input: "token_context" output: "token_context_out" op_type: "Identity" }
  node { input: "branch_state" output: "branch_state_out" op_type: "Identity" }
  node { output: "one" op_type: "Constant" attribute { name: "value" t { dims: 1 dims: 1 data_type: 7 int64_data: 1 } type: TENSOR } }
  node { input: "setup_counter" input: "one" output: "setup_counter_out" op_type: "Add" }
  node { input: "setup_counter_out" output: "setup_event" op_type: "Identity" }
  name: "candidate_composed_setup"
  input { name: "context_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_param: "sequence" } } } } }
  input { name: "proposer_state" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 2 } } } } }
  input { name: "target_state" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 2 } } } } }
  input { name: "token_context" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 4 } } } } }
  input { name: "branch_state" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 1 } } } } }
  input { name: "setup_counter" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 1 } } } } }
  output { name: "prepared_context" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_param: "sequence" } } } } }
  output { name: "proposer_state_out" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 2 } } } } }
  output { name: "target_state_out" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 2 } } } } }
  output { name: "token_context_out" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 4 } } } } }
  output { name: "branch_state_out" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 1 } } } } }
  output { name: "setup_counter_out" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 1 } } } } }
  output { name: "setup_event" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 1 } } } } }
}
opset_import { domain: "" version: 18 }
"#;

const COMPOSED_PRE_TRUE: &str = r#"
ir_version: 8
graph {
  node { input: "context_tokens" output: "prepared_context" op_type: "Identity" }
  name: "candidate_composed_pre_true"
  input { name: "context_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_param: "sequence" } } } } }
  output { name: "prepared_context" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_param: "sequence" } } } } }
}
opset_import { domain: "" version: 18 }
"#;

const COMPOSED_PRE_FALSE: &str = r#"
ir_version: 8
graph {
  node { output: "one" op_type: "Constant" attribute { name: "value" t { data_type: 7 int64_data: 1 } type: TENSOR } }
  node { input: "context_tokens" input: "one" output: "prepared_context" op_type: "Add" }
  name: "candidate_composed_pre_false"
  input { name: "context_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_param: "sequence" } } } } }
  output { name: "prepared_context" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_param: "sequence" } } } } }
}
opset_import { domain: "" version: 18 }
"#;

const COMPOSED_POST_TRUE: &str = r#"
ir_version: 8
graph {
  node { input: "accepted_tokens" output: "accepted_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 1 type: INT } }
  node { input: "branch_state" input: "accepted_sum" output: "branch_state_out" op_type: "Add" }
  node { input: "branch_state_out" output: "post_value" op_type: "Identity" }
  name: "candidate_composed_post_true"
  input { name: "accepted_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_param: "accepted" } } } } }
  input { name: "branch_state" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 1 } } } } }
  output { name: "post_value" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 1 } } } } }
  output { name: "branch_state_out" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 1 } } } } }
}
opset_import { domain: "" version: 18 }
"#;

const COMPOSED_POST_FALSE: &str = r#"
ir_version: 8
graph {
  node { output: "post_value" op_type: "Constant" attribute { name: "value" t { dims: 1 dims: 1 data_type: 7 int64_data: -1 } type: TENSOR } }
  node { input: "branch_state" output: "branch_state_out" op_type: "Identity" }
  name: "candidate_composed_post_false"
  input { name: "accepted_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_param: "accepted" } } } } }
  input { name: "branch_state" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 1 } } } } }
  output { name: "post_value" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 1 } } } } }
  output { name: "branch_state_out" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 1 } } } } }
}
opset_import { domain: "" version: 18 }
"#;

const COMPOSED_ROOT_POST: &str = r#"
ir_version: 8
graph {
  node { input: "accepted_tokens" output: "accepted_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 1 type: INT } }
  node { input: "root_state" input: "accepted_sum" output: "root_state_out" op_type: "Add" }
  node { input: "root_state_out" output: "post_value" op_type: "Identity" }
  name: "candidate_composed_root_post"
  input { name: "accepted_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_param: "accepted" } } } } }
  input { name: "root_state" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 1 } } } } }
  output { name: "post_value" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 1 } } } } }
  output { name: "root_state_out" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 1 } } } } }
}
opset_import { domain: "" version: 18 }
"#;

fn fixture_root(name: &str) -> anyhow::Result<PathBuf> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/candidate-tree-execution")
        .join(format!("{name}-{}", NEXT.fetch_add(1, Ordering::Relaxed)));
    fs::create_dir_all(&root)?;
    Ok(root)
}

fn parent_package(parents: &[i64; 6], decisions: &[usize; 7]) -> anyhow::Result<PathBuf> {
    let root = fixture_root("parents")?;
    fs::write(root.join("inference_metadata.yaml"), PARENT_METADATA)?;
    let parent_values = parents
        .iter()
        .map(|value| format!("int64_data: {value}"))
        .collect::<Vec<_>>()
        .join(" ");
    fs::write(
        root.join("proposer.onnx.textproto"),
        PARENT_PROPOSER.replace("{parent_values}", &parent_values),
    )?;
    let logits = decisions
        .iter()
        .flat_map(|decision| {
            (0..6).map(move |token| if token == *decision { 8.0_f32 } else { 0.0 })
        })
        .map(|value| format!("float_data: {value}"))
        .collect::<Vec<_>>()
        .join(" ");
    fs::write(
        root.join("target.onnx.textproto"),
        PARENT_TARGET.replace("{logit_values}", &logits),
    )?;
    Ok(root)
}

fn composed_parent_metadata(continue_loop: bool, reorder_components: bool) -> String {
    let controls = format!(
        r#"      accepted_prefix:
        contract: {{dtype: int64, shape: [batch, Any], batch_layout: {{kind: request_aligned, axis: 0}}}}
        role: {{kind: opaque}}
        source: {{kind: literal}}
        required: false
        default: 0
      control.continue:
        contract: {{dtype: bool, shape: [1]}}
        role: {{kind: opaque}}
        source: {{kind: literal}}
        required: false
        default: {continue_loop}
      control.max:
        contract: {{dtype: int64, shape: [1]}}
        role: {{kind: opaque}}
        source: {{kind: literal}}
        required: false
        default: 1
      control.branch:
        contract: {{dtype: bool, shape: [1]}}
        role: {{kind: opaque}}
        source: {{kind: literal}}
        required: false
        default: true
      setup.counter.seed:
        contract: {{dtype: int64, shape: [batch, 1], batch_layout: {{kind: request_aligned, axis: 0}}}}
        role: {{kind: opaque}}
        source: {{kind: literal}}
        required: false
        default: 0
      branch.state.seed:
        contract: {{dtype: int64, shape: [batch, 1], batch_layout: {{kind: request_aligned, axis: 0}}}}
        role: {{kind: opaque}}
        source: {{kind: literal}}
        required: false
        default: 0
      root.state.seed:
        contract: {{dtype: int64, shape: [batch, 1], batch_layout: {{kind: request_aligned, axis: 0}}}}
        role: {{kind: opaque}}
        source: {{kind: literal}}
        required: false
        default: 0
"#
    );
    let outputs = r#"    outputs:
      tokens:
        contract: {dtype: int64, shape: [batch, Any], batch_layout: {kind: request_aligned, axis: 0}}
        role: tokens
        family: {kind: events}
        stage: pre_adapter
      setup_events:
        contract: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
        role: tensor
        family: {kind: events}
        stage: pre_adapter
      setup_revision:
        contract: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
        role: tensor
        family: {kind: revisions, version: "1"}
        stage: pre_adapter
      root_events:
        contract: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
        role: tensor
        family: {kind: events}
        stage: pre_adapter
      post_value:
        contract: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
        role: tensor
        family: {kind: materialized}
        stage: pre_adapter
"#;
    let components = r#"      root_pre:
        implementation: {kind: onnx, artifact: pre-true.onnx.textproto}
        ports:
          inputs:
            context_tokens: {dtype: int64, shape: [batch, sequence], batch_layout: {kind: request_aligned, axis: 0}}
          outputs:
            prepared_context: {dtype: int64, shape: [batch, sequence], batch_layout: {kind: request_aligned, axis: 0}}
        row_scope: {axis: 0, stateful: false}
      root_post:
        implementation: {kind: onnx, artifact: root-post.onnx.textproto}
        ports:
          inputs:
            accepted_tokens: {dtype: int64, shape: [batch, accepted], batch_layout: {kind: request_aligned, axis: 0}}
            root_state: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
          outputs:
            post_value: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
            root_state_out: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
        row_scope: {axis: 0, stateful: true}
      setup:
        implementation: {kind: onnx, artifact: setup.onnx.textproto}
        effects: [audit]
        ports:
          inputs:
            context_tokens: {dtype: int64, shape: [batch, sequence], batch_layout: {kind: request_aligned, axis: 0}}
            proposer_state: {dtype: float32, shape: [batch, 2], batch_layout: {kind: request_aligned, axis: 0}}
            target_state: {dtype: float32, shape: [batch, 2], batch_layout: {kind: request_aligned, axis: 0}}
            token_context: {dtype: int64, shape: [batch, 4], batch_layout: {kind: request_aligned, axis: 0}}
            branch_state: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
            setup_counter: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
          outputs:
            prepared_context: {dtype: int64, shape: [batch, sequence], batch_layout: {kind: request_aligned, axis: 0}}
            proposer_state_out: {dtype: float32, shape: [batch, 2], batch_layout: {kind: request_aligned, axis: 0}}
            target_state_out: {dtype: float32, shape: [batch, 2], batch_layout: {kind: request_aligned, axis: 0}}
            token_context_out: {dtype: int64, shape: [batch, 4], batch_layout: {kind: request_aligned, axis: 0}}
            branch_state_out: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
            setup_counter_out: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
            setup_event: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
        row_scope: {axis: 0, stateful: true}
      pre_true:
        implementation: {kind: onnx, artifact: pre-true.onnx.textproto}
        ports:
          inputs:
            context_tokens: {dtype: int64, shape: [batch, sequence], batch_layout: {kind: request_aligned, axis: 0}}
          outputs:
            prepared_context: {dtype: int64, shape: [batch, sequence], batch_layout: {kind: request_aligned, axis: 0}}
        row_scope: {axis: 0, stateful: false}
      pre_false:
        implementation: {kind: onnx, artifact: pre-false.onnx.textproto}
        ports:
          inputs:
            context_tokens: {dtype: int64, shape: [batch, sequence], batch_layout: {kind: request_aligned, axis: 0}}
          outputs:
            prepared_context: {dtype: int64, shape: [batch, sequence], batch_layout: {kind: request_aligned, axis: 0}}
        row_scope: {axis: 0, stateful: false}
      post_true:
        implementation: {kind: onnx, artifact: post-true.onnx.textproto}
        ports:
          inputs:
            accepted_tokens: {dtype: int64, shape: [batch, accepted], batch_layout: {kind: request_aligned, axis: 0}}
            branch_state: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
          outputs:
            post_value: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
            branch_state_out: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
        row_scope: {axis: 0, stateful: true}
      post_false:
        implementation: {kind: onnx, artifact: post-false.onnx.textproto}
        ports:
          inputs:
            accepted_tokens: {dtype: int64, shape: [batch, accepted], batch_layout: {kind: request_aligned, axis: 0}}
            branch_state: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
          outputs:
            post_value: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
            branch_state_out: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
        row_scope: {axis: 0, stateful: true}
"#;
    let steps = r#"    steps:
    - kind: invoke
      component: root_pre
      inputs: {context_tokens: request.tokens}
      outputs: {prepared_context: root.context}
    - kind: loop
      setup:
      - kind: invoke
        component: setup
        inputs:
          context_tokens: root.context
          proposer_state: request.proposer_state
          target_state: request.target_state
          token_context: request.token_context
          branch_state: branch.state.seed
          setup_counter: setup.counter.seed
        outputs:
          prepared_context: setup.context
          proposer_state_out: setup.proposer_state
          target_state_out: setup.target_state
          token_context_out: setup.token_context
          branch_state_out: setup.branch_state
          setup_counter_out: setup.counter.next
          setup_event: setup.event
      - {kind: emit, value: setup.event, output: setup_events, mode: event}
      - {kind: emit, value: setup.counter.next, output: setup_revision, mode: append}
      steps:
      - kind: branch
        predicate: control.branch
        cases:
          "true":
            kind: invoke
            component: pre_true
            inputs: {context_tokens: setup.context}
            outputs: {prepared_context: branch.true.context}
        default:
          kind: invoke
          component: pre_false
          inputs: {context_tokens: setup.context}
          outputs: {prepared_context: branch.false.context}
        outputs:
          branch.context:
            cases: {"true": branch.true.context}
            default: branch.false.context
      - kind: invoke
        component: proposer
        inputs: {context_tokens: branch.context, private_state: proposer_private}
        outputs:
          candidate_tokens: proposed.tokens
          candidate_parents: proposed.parents
          proposal_probabilities: proposed.probabilities
          private_state_out: proposed.state
      - kind: invoke
        component: target
        inputs:
          context_tokens: branch.context
          candidate_tokens: proposed.tokens
          ancestor_mask: runtime.ancestor_mask
          position_ids: runtime.position_ids
          accepted_tokens: runtime.accepted_tokens
          target_state: target_recurrent
          token_context: token_context
        outputs:
          target_logits: verified.logits
          target_probabilities: verified.probabilities
          target_state_out: verified.state
          token_context_out: verified.token_context
      - kind: branch
        predicate: control.branch
        cases:
          "true":
            kind: invoke
            component: post_true
            inputs: {accepted_tokens: accepted_prefix, branch_state: branch_counter}
            outputs: {post_value: branch.true.post, branch_state_out: branch.true.state}
        default:
          kind: invoke
          component: post_false
          inputs: {accepted_tokens: accepted_prefix, branch_state: branch_counter}
          outputs: {post_value: branch.false.post, branch_state_out: branch.false.state}
        outputs:
          branch.post:
            cases: {"true": branch.true.post}
            default: branch.false.post
          branch.state:
            cases: {"true": branch.true.state}
            default: branch.false.state
      - {kind: emit, value: accepted_prefix, output: tokens, mode: event}
      - {kind: emit, value: branch.post, output: post_value, mode: replace}
      continue_when: control.continue
      max_iterations: control.max
      carried:
      - {cell: proposer_private, initial: setup.proposer_state, next: proposed.state}
      - {cell: target_recurrent, initial: setup.target_state, next: verified.state}
      - {cell: token_context, initial: setup.token_context, next: verified.token_context}
      - {cell: branch_counter, initial: setup.branch_state, next: branch.state}
    - kind: invoke
      component: root_post
      inputs: {accepted_tokens: setup.event, root_state: root.state.seed}
      outputs: {post_value: root.post, root_state_out: root.state.next}
    - {kind: emit, value: root.post, output: root_events, mode: event}
"#;
    let setup_state = r#"      setup_counter:
        contract: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
        class: semantic
        scope: session
        initializer: setup.counter.seed
        recurrence: {kind: invariant}
        management: runtime
        release_boundary: session
        service_group: setup_counter
      branch_counter:
        contract: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
        class: semantic
        scope: session
        initializer: branch.state.seed
        recurrence: {kind: invariant}
        management: runtime
        release_boundary: session
        service_group: branch_counter
      root_counter:
        contract: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}
        class: semantic
        scope: session
        initializer: root.state.seed
        recurrence: {kind: invariant}
        management: runtime
        release_boundary: session
        service_group: root_counter
"#;
    let setup_service = r#"          setup_counter:
            kind: recurrent
            layout: bf
            update: {kind: replace}
            capabilities: {rollback_positions: 1, snapshot: true, fork: false}
            ports:
              setup:
                setup_counter: {input: setup_counter, output: setup_counter_out}
          branch_counter:
            kind: recurrent
            layout: bf
            update: {kind: replace}
            capabilities: {rollback_positions: 6, snapshot: true, fork: false}
            ports:
              setup:
                branch_counter: {input: branch_state, output: branch_state_out}
              post_true:
                branch_counter: {input: branch_state, output: branch_state_out}
              post_false:
                branch_counter: {input: branch_state, output: branch_state_out}
          root_counter:
            kind: recurrent
            layout: bf
            update: {kind: replace}
            capabilities: {rollback_positions: 1, snapshot: true, fork: false}
            ports:
              root_post:
                root_counter: {input: root_state, output: root_state_out}
"#;

    let original_outputs = r#"    outputs:
      tokens:
        contract: {dtype: int64, shape: [1]}
        role: tokens
        family: {kind: events}
        stage: pre_adapter
    components:
"#;
    let mut metadata = PARENT_METADATA
        .replacen(
            "capabilities: [workflow_ssa, serving_service_contract, canonical_speculation, session_state_lease]",
            "capabilities: [workflow_ssa, nested_control_flow, linear_effects, typed_emit, serving_service_contract, canonical_speculation, session_state_lease]",
            1,
        )
        .replacen(
            original_outputs,
            &(controls + outputs + "    components:\n"),
            1,
        );
    let component_marker = if reorder_components {
        "    steps:\n"
    } else {
        "      proposer:\n"
    };
    metadata = metadata.replacen(
        component_marker,
        &(components.to_string() + component_marker),
        1,
    );
    let start = metadata.find("    steps:\n").expect("workflow steps");
    let end = metadata[start..]
        .find("    state:\n")
        .map(|offset| start + offset)
        .expect("workflow state");
    metadata.replace_range(start..end, steps);
    metadata = metadata
        .replacen("    state:\n", &("    state:\n".to_string() + setup_state), 1)
        .replacen(
            "    serving:\n",
            "    effects:\n      audit:\n        retry: transactional\n        speculation_safety: {kind: clonable}\n    serving:\n",
            1,
        )
        .replacen(
            "        groups:\n",
            &("        groups:\n".to_string() + setup_service),
            1,
        )
        .replacen(
            "          proposer_private:\n            kind: recurrent\n            layout: bf\n            update: {kind: replace}\n            capabilities: {rollback_positions: 6, snapshot: true, fork: false}\n            ports:\n              proposer:\n",
            "          proposer_private:\n            kind: recurrent\n            layout: bf\n            update: {kind: replace}\n            capabilities: {rollback_positions: 6, snapshot: true, fork: false}\n            ports:\n              setup:\n                proposer_private: {input: proposer_state, output: proposer_state_out}\n              proposer:\n",
            1,
        )
        .replacen(
            "          target_recurrent:\n            kind: recurrent\n            layout: bf\n            update: {kind: replace}\n            capabilities: {rollback_positions: 6, snapshot: true, fork: false, cascade: [token_context]}\n            ports:\n              target:\n",
            "          target_recurrent:\n            kind: recurrent\n            layout: bf\n            update: {kind: replace}\n            capabilities: {rollback_positions: 6, snapshot: true, fork: false, cascade: [token_context]}\n            ports:\n              setup:\n                target_recurrent: {input: target_state, output: target_state_out}\n              target:\n",
            1,
        )
        .replacen(
            "          token_context:\n            kind: recurrent\n            layout: bf\n            update: {kind: replace}\n            capabilities: {rollback_positions: 6, snapshot: true, fork: false, cascade: [target_recurrent]}\n            ports:\n              target:\n",
            "          token_context:\n            kind: recurrent\n            layout: bf\n            update: {kind: replace}\n            capabilities: {rollback_positions: 6, snapshot: true, fork: false, cascade: [target_recurrent]}\n            ports:\n              setup:\n                token_context: {input: token_context, output: token_context_out}\n              target:\n",
            1,
        )
        .replacen(
            "  rollback_state: [proposer_private, target_recurrent, token_context]",
            "  rollback_state: [proposer_private, target_recurrent, token_context, branch_counter]",
            1,
        );
    metadata
}

fn composed_parent_package(
    continue_loop: bool,
    reorder_components: bool,
) -> anyhow::Result<PathBuf> {
    let root = parent_package(&[-1, -1, 0, 1, 2, 4], &[2, 0, 4, 0, 0, 0, 0])?;
    fs::write(
        root.join("inference_metadata.yaml"),
        composed_parent_metadata(continue_loop, reorder_components),
    )?;
    for (name, model) in [
        ("setup.onnx.textproto", COMPOSED_SETUP),
        ("pre-true.onnx.textproto", COMPOSED_PRE_TRUE),
        ("pre-false.onnx.textproto", COMPOSED_PRE_FALSE),
        ("post-true.onnx.textproto", COMPOSED_POST_TRUE),
        ("post-false.onnx.textproto", COMPOSED_POST_FALSE),
        ("root-post.onnx.textproto", COMPOSED_ROOT_POST),
    ] {
        fs::write(root.join(name), model)?;
    }
    Ok(root)
}

fn retryable_composed_parent_package() -> anyhow::Result<PathBuf> {
    let root = composed_parent_package(true, false)?;
    replace_metadata(&root, |metadata| {
        metadata
            .replacen(
                "      setup_revision:\n        contract: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}\n        role: tensor\n        family: {kind: revisions, version: \"1\"}",
                "      setup_revision:\n        contract: {dtype: int64, shape: [batch, 1], batch_layout: {kind: request_aligned, axis: 0}}\n        role: tensor\n        family: {kind: events}",
                1,
            )
            .replacen(
                "{kind: emit, value: setup.counter.next, output: setup_revision, mode: append}",
                "{kind: emit, value: setup.counter.next, output: setup_revision, mode: event}",
                1,
            )
    })?;
    Ok(root)
}

fn repeated_composed_parent_package(iterations: usize) -> anyhow::Result<PathBuf> {
    let root = retryable_composed_parent_package()?;
    replace_metadata(&root, |metadata| {
        metadata.replacen(
            "      control.max:\n        contract: {dtype: int64, shape: [1]}\n        role: {kind: opaque}\n        source: {kind: literal}\n        required: false\n        default: 1",
            &format!(
                "      control.max:\n        contract: {{dtype: int64, shape: [1]}}\n        role: {{kind: opaque}}\n        source: {{kind: literal}}\n        required: false\n        default: {iterations}"
            ),
            1,
        )
    })?;
    Ok(root)
}

fn replace_metadata(root: &Path, update: impl FnOnce(String) -> String) -> anyhow::Result<()> {
    let path = root.join("inference_metadata.yaml");
    let metadata = fs::read_to_string(&path)?;
    fs::write(path, update(metadata))?;
    Ok(())
}

fn install_tool_protocol_tokenizer(root: &Path, token_texts: &[(u64, &str)]) -> anyhow::Result<()> {
    replace_metadata(root, |metadata| {
        metadata.replacen(
            "schema_version: v1.6\npipeline:",
            "schema_version: v1.6\npackage:\n  tool_protocol: {identity: tagged-json, version: v1}\npipeline:",
            1,
        )
    })?;
    let vocab = serde_json::Map::from_iter(
        (0..6).map(|token| (format!("token_{token}"), serde_json::Value::from(token))),
    );
    let mut vocab = vocab;
    vocab.insert("<unk>".to_string(), serde_json::Value::from(1));
    for (token, text) in token_texts {
        vocab.retain(|_, value| value.as_u64() != Some(*token));
        vocab.insert((*text).to_string(), serde_json::Value::from(*token));
    }
    let tokenizer = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    });
    fs::write(
        root.join("tokenizer.json"),
        serde_json::to_vec_pretty(&tokenizer)?,
    )?;
    Ok(())
}

fn assert_preload_candidate_refusal(
    root: &Path,
    expected: &[&str],
) -> anyhow::Result<anyhow::Error> {
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "textproto")
        {
            fs::write(path, "this artifact must not be parsed")?;
        }
    }
    let error = expect_error(
        Engine::from_dir(root, EngineConfig::default()),
        "candidate-tree composition must fail before component loading",
    );
    let reason = match package_execution_error(&error) {
        Some(PackageExecutionError::CandidateTreeExecutionUnavailable { reason, .. }) => reason,
        other => panic!("expected typed pre-load candidate-tree refusal, got {other:?}: {error:#}"),
    };
    for fragment in expected {
        assert!(
            reason.contains(fragment),
            "candidate-tree refusal does not contain {fragment:?}: {reason}"
        );
    }
    Ok(error)
}

fn with_control_flow_support(metadata: String) -> String {
    metadata
        .replacen(
            "capabilities: [workflow_ssa, serving_service_contract, canonical_speculation, session_state_lease]",
            "capabilities: [workflow_ssa, nested_control_flow, serving_service_contract, canonical_speculation, session_state_lease]",
            1,
        )
        .replacen(
            "      runtime.active:\n",
            "      control.continue:\n        contract: {dtype: bool, shape: [1]}\n        role: {kind: opaque}\n        source: {kind: literal}\n        required: false\n        default: false\n      control.max:\n        contract: {dtype: int64, shape: [1]}\n        role: {kind: opaque}\n        source: {kind: literal}\n        required: false\n        default: 1\n      runtime.active:\n",
            1,
        )
        .replacen(
            "      proposer:\n",
            "      observer:\n        implementation: {kind: onnx, artifact: observer.onnx.textproto}\n        ports: {inputs: {}, outputs: {}}\n        row_scope: {axis: 0, stateful: false}\n      proposer:\n",
            1,
        )
}

#[test]
fn composed_candidate_tree_runs_setup_branches_effects_and_all_outputs() -> anyhow::Result<()> {
    let root = composed_parent_package(true, false)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let session = engine.create_session()?;
    let result = engine.generate_in_session(session, greedy_request(3))?;
    assert_eq!(result.token_ids, vec![2, 4, 0]);
    assert_eq!(
        engine.workflow_session_effect_cursor(&session.to_string(), "audit"),
        Some(1),
        "setup effect must commit exactly once through the generic turn"
    );
    let invocations = engine.component_invocations();
    for (component, expected) in [
        ("root_pre", 1),
        ("setup", 1),
        ("pre_true", 1),
        ("proposer", 2),
        ("target", 2),
        ("post_true", 1),
        ("root_post", 1),
    ] {
        assert_eq!(
            invocations.get(component).copied(),
            Some(expected),
            "{component} must execute in authored order without duplication"
        );
    }
    assert!(!invocations.contains_key("pre_false"));
    assert!(!invocations.contains_key("post_false"));

    let publications = engine.take_committed_workflow_publications();
    let names = publications
        .iter()
        .map(|publication| match publication {
            WorkflowOutputPublication::Materialized { output, .. }
            | WorkflowOutputPublication::Event { output, .. } => output.as_str(),
            WorkflowOutputPublication::Revision(envelope) => envelope.output.as_str(),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        [
            "setup_events",
            "setup_revision",
            "tokens",
            "post_value",
            "root_events",
            "setup_revision",
        ]
    );
    let WorkflowOutputPublication::Event { payload, .. } = &publications[0] else {
        panic!("setup output must be an event")
    };
    assert_eq!(payload.to_vec_i64()?, vec![1]);
    let WorkflowOutputPublication::Event { payload, .. } = &publications[2] else {
        panic!("candidate accepted path must flow through the authored token emit")
    };
    assert_eq!(payload.to_vec_i64()?, vec![2, 4, 0]);
    let WorkflowOutputPublication::Materialized { payload, .. } = &publications[3] else {
        panic!("post-branch phi must publish a materialized value")
    };
    assert_eq!(payload.to_vec_i64()?, vec![6]);
    let error = engine
        .prepare_session_fork(session, SessionPosition::new(0))
        .expect_err("composed candidate fork must decline before child allocation");
    assert!(matches!(
        error,
        SessionForkError::UnsupportedParticipant { participant, reason, .. }
            if participant == "candidate_tree.accepted_path"
                && reason.contains("before child allocation")
    ));

    Ok(())
}

#[test]
fn candidate_token_binding_requires_exact_proposer_ssa_provenance_before_loading()
-> anyhow::Result<()> {
    let parents = [-1, -1, 0, 1, 2, 4];
    let decisions = [1, 4, 0, 0, 0, 0, 0];

    let literal = parent_package(&parents, &decisions)?;
    replace_metadata(&literal, |metadata| {
        metadata.replacen(
            "        candidate_tokens: proposed.tokens\n        ancestor_mask:",
            "        candidate_tokens: runtime.position_ids\n        ancestor_mask:",
            1,
        )
    })?;
    assert_preload_candidate_refusal(
        &literal,
        &[
            "pipeline.workflow.steps[1].inputs.candidate_tokens",
            "must consume proposer",
            "proposed.tokens",
            "workflow input 'runtime.position_ids'",
            "source Literal",
        ],
    )?;

    let transformed = parent_package(&parents, &decisions)?;
    replace_metadata(&transformed, |metadata| {
        metadata
            .replacen(
                "      proposer:\n",
                "      transformer:\n        implementation: {kind: onnx, artifact: transformer.onnx.textproto}\n        ports:\n          inputs: {}\n          outputs:\n            candidate_tokens: {dtype: int64, shape: [batch, 6], batch_layout: {kind: request_aligned, axis: 0}}\n        row_scope: {axis: 0, stateful: false}\n      proposer:\n",
                1,
            )
            .replacen(
                "    - kind: invoke\n      component: proposer\n",
                "    - kind: invoke\n      component: transformer\n      outputs: {candidate_tokens: transformed.tokens}\n    - kind: invoke\n      component: proposer\n",
                1,
            )
            .replacen(
                "        candidate_tokens: proposed.tokens\n        ancestor_mask:",
                "        candidate_tokens: transformed.tokens\n        ancestor_mask:",
                1,
            )
    })?;
    assert_preload_candidate_refusal(
        &transformed,
        &[
            "inputs.candidate_tokens",
            "proposed.tokens",
            "component 'transformer' output port 'candidate_tokens'",
            "unrelated component outputs do not prove candidate identity",
        ],
    )?;
    Ok(())
}

#[test]
fn topology_and_driver_owned_target_inputs_have_proved_sources_before_loading() -> anyhow::Result<()>
{
    let direct_topology = ancestor_package()?;
    replace_metadata(&direct_topology, |metadata| {
        metadata.replacen(
            "        branch_ancestors: tree.ancestors\n        branch_positions:",
            "        branch_ancestors: mask_seed\n        branch_positions:",
            1,
        )
    })?;
    assert_preload_candidate_refusal(
        &direct_topology,
        &[
            "inputs.branch_ancestors",
            "tree.ancestors",
            "workflow input 'mask_seed'",
            "candidate identity",
        ],
    )?;

    let derived = parent_package(&[-1, -1, 0, 1, 2, 4], &[1, 4, 0, 0, 0, 0, 0])?;
    replace_metadata(&derived, |metadata| {
        metadata.replacen(
            "      runtime.ancestor_mask:\n        contract: {dtype: bool, shape: [batch, 6, 6], batch_layout: {kind: request_aligned, axis: 0}}\n        role: {kind: opaque}\n        source: {kind: literal}",
            "      runtime.ancestor_mask:\n        contract: {dtype: bool, shape: [batch, 6, 6], batch_layout: {kind: request_aligned, axis: 0}}\n        role: {kind: opaque}\n        source: {kind: application, name: tree.mask}",
            1,
        )
    })?;
    assert_preload_candidate_refusal(
        &derived,
        &[
            "inputs.ancestor_mask",
            "Application { name: \"tree.mask\" }",
            "ancestor mask derived from the proved parent-index topology",
            "cannot be ignored",
        ],
    )?;
    Ok(())
}

#[test]
fn ambiguous_phi_candidate_binding_refuses_before_loading() -> anyhow::Result<()> {
    let root = parent_package(&[-1, -1, 0, 1, 2, 4], &[1, 4, 0, 0, 0, 0, 0])?;
    replace_metadata(&root, |metadata| {
        with_control_flow_support(metadata)
            .replacen(
                "    - kind: invoke\n      component: target\n",
                "    - kind: branch\n      predicate: control.continue\n      cases:\n        \"true\": {kind: invoke, component: observer}\n      default: {kind: invoke, component: observer}\n      outputs:\n        joined.tokens:\n          cases: {\"true\": proposed.tokens}\n          default: runtime.position_ids\n    - kind: invoke\n      component: target\n",
                1,
            )
            .replacen(
                "        candidate_tokens: proposed.tokens\n        ancestor_mask:",
                "        candidate_tokens: joined.tokens\n        ancestor_mask:",
                1,
            )
    })?;
    assert_preload_candidate_refusal(
        &root,
        &[
            "inputs.candidate_tokens",
            "joined.tokens",
            "no unique proved source",
            "proposed.tokens",
        ],
    )?;
    Ok(())
}

#[test]
fn intervening_work_at_the_exact_verification_seam_refuses_before_loading() -> anyhow::Result<()> {
    let root = parent_package(&[-1, -1, 0, 1, 2, 4], &[1, 4, 0, 0, 0, 0, 0])?;
    replace_metadata(&root, |metadata| {
        with_control_flow_support(metadata).replacen(
            "    - kind: invoke\n      component: target\n",
            "    - kind: invoke\n      component: observer\n    - kind: invoke\n      component: target\n",
            1,
        )
    })?;
    assert_preload_candidate_refusal(
        &root,
        &[
            "proposer",
            "not immediately followed",
            "observer",
            "Move unrelated work before or after",
        ],
    )?;
    Ok(())
}

#[test]
fn zero_trip_setup_and_component_map_reorder_preserve_semantics() -> anyhow::Result<()> {
    let zero_trip = composed_parent_package(false, false)?;
    let mut engine = Engine::from_dir(&zero_trip, EngineConfig::default())?;
    let result = engine.generate(greedy_request(3))?;
    assert!(result.token_ids.is_empty());
    let invocations = engine.component_invocations();
    assert_eq!(invocations.get("setup").copied(), Some(1));
    assert_eq!(invocations.get("root_pre").copied(), Some(1));
    assert_eq!(invocations.get("root_post").copied(), Some(1));
    assert!(!invocations.contains_key("proposer"));
    assert!(!invocations.contains_key("target"));
    let publications = engine.take_committed_workflow_publications();
    assert_eq!(publications.len(), 4);
    assert!(matches!(
        &publications[0],
        WorkflowOutputPublication::Event { output, payload, .. }
            if output == "setup_events" && payload.to_vec_i64()? == vec![1]
    ));

    let ordered = composed_parent_package(true, false)?;
    let reordered = composed_parent_package(true, true)?;
    let ordered_result =
        Engine::from_dir(&ordered, EngineConfig::default())?.generate(greedy_request(3))?;
    let reordered_result =
        Engine::from_dir(&reordered, EngineConfig::default())?.generate(greedy_request(3))?;
    assert_eq!(ordered_result.token_ids, reordered_result.token_ids);
    Ok(())
}

#[test]
fn zero_trip_candidate_workflow_finishes_required_policy_and_aborts_authored_mutations()
-> anyhow::Result<()> {
    let root = composed_parent_package(false, false)?;
    install_tool_protocol_tokenizer(&root, &[])?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let session = engine.create_session()?;
    let mut delivered = Vec::new();
    let mut callback = |token: onnx_genai_engine::GenerateToken| {
        delivered.push(token.token_id);
        Ok(())
    };
    let error = engine
        .generate_with_pipeline_tool_policy_callbacks(
            PipelineGenerateRequest::new(greedy_request(3))
                .with_session_id(session.to_string())
                .with_tool_call_policy(ToolCallPolicy::Required),
            None,
            Some(&mut callback),
        )
        .expect_err("zero-trip generation must still enforce required tool policy");
    let error = format!("{error:#}");
    assert!(
        error.contains("candidate-tree semantic commit")
            && error.contains("at least one was required"),
        "{error}"
    );
    assert!(delivered.is_empty());
    assert_eq!(engine.session_token_count(session)?, 0);
    assert_eq!(
        engine.workflow_session_effect_cursor(&session.to_string(), "audit"),
        None
    );
    assert!(engine.take_committed_workflow_publications().is_empty());
    assert!(engine.component_invocations().is_empty());
    Ok(())
}

#[test]
fn sampling_candidate_path_executes_inside_the_composed_workflow() -> anyhow::Result<()> {
    let root = composed_parent_package(true, false)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let result = engine.generate(sampling_request(23, 2))?;
    assert!(!result.token_ids.is_empty());
    let invocations = engine.component_invocations();
    assert_eq!(invocations.get("setup").copied(), Some(1));
    assert_eq!(invocations.get("root_pre").copied(), Some(1));
    assert_eq!(invocations.get("root_post").copied(), Some(1));
    assert_eq!(invocations.get("pre_true").copied(), Some(1));
    assert_eq!(invocations.get("post_true").copied(), Some(1));
    assert_eq!(
        engine.take_candidate_tree_block_traces().len(),
        1,
        "the authored loop owns one candidate-tree seam entry"
    );
    Ok(())
}

#[test]
fn token_output_identity_is_selected_by_role_not_by_name() -> anyhow::Result<()> {
    let root = parent_package(&[-1, -1, 0, 1, 2, 4], &[2, 0, 4, 0, 0, 0, 0])?;
    replace_metadata(&root, |metadata| {
        metadata.replacen("      tokens:\n", "      generated_ids:\n", 1)
    })?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let result = engine.generate(greedy_request(3))?;
    assert_eq!(result.token_ids, vec![2, 4, 0]);
    assert!(engine.take_committed_workflow_publications().iter().all(
        |publication| match publication {
            WorkflowOutputPublication::Materialized { output, .. }
            | WorkflowOutputPublication::Event { output, .. } => output == "generated_ids",
            WorkflowOutputPublication::Revision(envelope) => {
                envelope.output == "generated_ids"
            }
        }
    ));
    Ok(())
}

#[test]
fn authored_loop_repeats_only_its_body_and_never_duplicates_setup() -> anyhow::Result<()> {
    let root = repeated_composed_parent_package(2)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let result = engine.generate(greedy_request(6))?;
    assert!(!result.token_ids.is_empty());
    let invocations = engine.component_invocations();
    assert_eq!(engine.take_candidate_tree_block_traces().len(), 2);
    assert_eq!(invocations.get("setup").copied(), Some(1));
    assert_eq!(invocations.get("root_pre").copied(), Some(1));
    assert_eq!(invocations.get("root_post").copied(), Some(1));
    assert_eq!(invocations.get("pre_true").copied(), Some(2));
    assert_eq!(invocations.get("post_true").copied(), Some(2));
    assert_eq!(invocations.get("proposer").copied(), Some(4));
    assert_eq!(invocations.get("target").copied(), Some(4));
    Ok(())
}

#[test]
fn composed_loop_does_not_reenter_after_candidate_budget_or_eos_stop() -> anyhow::Result<()> {
    for (label, request, expected, finish) in [
        (
            "request budget",
            greedy_request(3),
            vec![2, 4, 0],
            FinishReason::MaxTokens,
        ),
        (
            "committed EOS",
            {
                let mut request = greedy_request(5);
                request.options.stop_on_eos = true;
                request.options.eos_token_ids = vec![2];
                request
            },
            vec![2],
            FinishReason::EosToken,
        ),
    ] {
        let root = repeated_composed_parent_package(3)?;
        let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
        let result = engine.generate(request)?;
        assert_eq!(
            result.token_ids, expected,
            "{label} must cap the committed path"
        );
        assert_eq!(result.finish_reason, finish, "{label} finish reason");
        let invocations = engine.component_invocations();
        assert_eq!(
            invocations.get("proposer"),
            Some(&2),
            "{label} must not enter the next authored proposer iteration"
        );
        assert_eq!(
            invocations.get("target"),
            Some(&2),
            "{label} must not enter the next authored target iteration"
        );
        assert_eq!(
            invocations.get("setup"),
            Some(&1),
            "{label} must preserve exactly-once loop setup"
        );
    }
    Ok(())
}

fn ancestor_package() -> anyhow::Result<PathBuf> {
    ancestor_package_with_sources(ANCESTOR_PROPOSER, ANCESTOR_TARGET)
}

fn ancestor_package_with_sources(proposer: &str, target: &str) -> anyhow::Result<PathBuf> {
    let root = fixture_root("ancestors")?;
    fs::write(root.join("inference_metadata.yaml"), ANCESTOR_METADATA)?;
    fs::write(root.join("branching-proposer.onnx.textproto"), proposer)?;
    fs::write(root.join("branching-target.onnx.textproto"), target)?;
    Ok(root)
}

fn greedy_request(max_new_tokens: usize) -> GenerateRequest {
    GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![1, 2]),
        options: GenerateOptions {
            max_new_tokens,
            greedy: true,
            stop_on_eos: false,
            ..GenerateOptions::default()
        },
    }
}

fn context_exhausted_tool_request(
    session_id: String,
    policy: ToolCallPolicy,
) -> PipelineGenerateRequest {
    let mut request = greedy_request(2);
    request.options.max_context = Some(2);
    PipelineGenerateRequest::new(request)
        .with_session_id(session_id)
        .with_tool_call_policy(policy)
}

fn expect_error<T>(result: anyhow::Result<T>, message: &str) -> anyhow::Error {
    match result {
        Ok(_) => panic!("{message}"),
        Err(error) => error,
    }
}

#[test]
fn parent_tree_engine_dispatches_nonzero_root_and_deep_paths() -> anyhow::Result<()> {
    let parents = [-1, -1, 0, 1, 2, 4];
    for (decisions, expected) in [
        ([0, 0, 0, 0, 0, 0, 0], vec![0]),
        ([1, 4, 0, 0, 0, 0, 0], vec![1, 4]),
        ([2, 0, 4, 0, 0, 0, 0], vec![2, 4, 0]),
        ([1, 3, 0, 5, 0, 0, 2], vec![1, 3, 5, 0, 2]),
    ] {
        let root = parent_package(&parents, &decisions)?;
        let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
        let result = engine.generate(greedy_request(expected.len()))?;
        assert_eq!(result.token_ids, expected);
        assert_eq!(
            engine.component_invocations(),
            std::collections::BTreeMap::from([
                ("proposer".to_string(), 2),
                ("target".to_string(), 2),
            ]),
            "one verification block plus accepted-path recomputation must execute both real \
             components through metadata-selected dispatch"
        );
        let trace = engine.take_candidate_tree_block_traces();
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].committed_tokens, expected);
    }

    Ok(())
}

#[test]
fn candidate_tree_waits_for_budget_boundary_across_complete_call_blocks() -> anyhow::Result<()> {
    let root = parent_package(&[-1, -1, 0, 1, 2, 4], &[2, 0, 4, 0, 0, 0, 0])?;
    install_tool_protocol_tokenizer(
        &root,
        &[
            (
                0,
                r#"<tool_call>{"name":"zero","arguments":{}}</tool_call>"#,
            ),
            (2, r#"<tool_call>{"name":"two","arguments":{}}</tool_call>"#),
            (
                4,
                r#"<tool_call>{"name":"four","arguments":{}}</tool_call>"#,
            ),
        ],
    )?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let result = engine.generate_with_pipeline_request(
        PipelineGenerateRequest::new(greedy_request(5)).with_tool_call_policy(ToolCallPolicy::Auto),
    )?;
    assert_eq!(result.token_ids, vec![2, 4, 0, 2, 4]);
    assert_eq!(result.finish_reason, FinishReason::ToolCalls);
    assert_eq!(
        result
            .tool_calls
            .iter()
            .map(|call| call.name.as_str())
            .collect::<Vec<_>>(),
        ["two", "four", "zero", "two", "four"]
    );
    Ok(())
}

#[test]
fn initial_context_exhaustion_enforces_tool_policy_without_session_mutation() -> anyhow::Result<()>
{
    let root = parent_package(&[-1, -1, 0, 1, 2, 4], &[2, 0, 4, 0, 0, 0, 0])?;
    install_tool_protocol_tokenizer(&root, &[])?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let session = engine.create_session()?;
    let session_id = session.to_string();
    let mut delivered = Vec::new();

    for (policy, required_text) in [
        (
            ToolCallPolicy::Required,
            "the model produced no tool call, but at least one was required",
        ),
        (
            ToolCallPolicy::Specific {
                function: "weather".to_string(),
            },
            "the model produced no tool call, but function \"weather\" was required",
        ),
    ] {
        let mut callback = |token: onnx_genai_engine::GenerateToken| {
            delivered.push(token.token_id);
            Ok(())
        };
        let error = engine
            .generate_with_pipeline_tool_policy_callbacks(
                context_exhausted_tool_request(session_id.clone(), policy),
                None,
                Some(&mut callback),
            )
            .expect_err("a required tool policy must fail at the no-generation boundary");
        let error = format!("{error:#}");
        assert!(error.contains(required_text), "{error}");
        assert!(
            error.contains("candidate-tree initial context-limit boundary"),
            "{error}"
        );
        assert!(delivered.is_empty());
        assert_eq!(engine.session_token_count(session)?, 0);
        assert!(engine.component_invocations().is_empty());
        assert!(engine.take_committed_workflow_publications().is_empty());
    }

    for policy in [ToolCallPolicy::Auto, ToolCallPolicy::Disabled] {
        let mut callback = |token: onnx_genai_engine::GenerateToken| {
            delivered.push(token.token_id);
            Ok(())
        };
        let result = engine.generate_with_pipeline_tool_policy_callbacks(
            context_exhausted_tool_request(session_id.clone(), policy),
            None,
            Some(&mut callback),
        )?;
        assert!(result.token_ids.is_empty());
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.finish_reason, FinishReason::Length);
        assert!(delivered.is_empty());
        assert_eq!(engine.session_token_count(session)?, 0);
        assert!(engine.component_invocations().is_empty());
        assert!(engine.take_committed_workflow_publications().is_empty());
    }
    Ok(())
}

#[test]
fn incomplete_candidate_tool_output_aborts_and_same_session_disabled_retry_is_clean()
-> anyhow::Result<()> {
    let root = parent_package(&[-1, -1, 0, 1, 2, 4], &[2, 0, 4, 0, 0, 0, 0])?;
    let incomplete = r#"<tool_call>{"name":"weather","arguments":{"city":"#;
    install_tool_protocol_tokenizer(&root, &[(2, incomplete)])?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let session = engine.create_session()?;
    let mut delivered = Vec::new();
    let mut callback = |token: onnx_genai_engine::GenerateToken| {
        delivered.push(token.token_id);
        Ok(())
    };
    let error = engine
        .generate_with_pipeline_tool_policy_callbacks(
            PipelineGenerateRequest::new(greedy_request(1))
                .with_session_id(session.to_string())
                .with_tool_call_policy(ToolCallPolicy::Auto),
            None,
            Some(&mut callback),
        )
        .expect_err("unfinished candidate tool output must fail at the budget boundary");
    let error = format!("{error:#}");
    assert!(
        error.contains("candidate-tree terminal generation boundary")
            && error.contains("produced incomplete staged output"),
        "{error}"
    );
    assert!(delivered.is_empty());
    assert_eq!(engine.session_token_count(session)?, 0);
    assert!(engine.take_committed_workflow_publications().is_empty());

    let retry = engine.generate_with_pipeline_request(
        PipelineGenerateRequest::new(greedy_request(1))
            .with_session_id(session.to_string())
            .with_tool_call_policy(ToolCallPolicy::Disabled),
    )?;
    assert_eq!(retry.token_ids, vec![2]);
    assert_eq!(retry.text, incomplete);
    assert_eq!(retry.finish_reason, FinishReason::MaxTokens);
    assert!(retry.tool_calls.is_empty());
    assert_eq!(engine.session_token_count(session)?, 0);
    assert_eq!(
        engine
            .contract_executions()
            .get("onnx-genai.speculative-block")
            .copied(),
        Some(1)
    );
    Ok(())
}

#[test]
fn ancestor_mask_tree_sampling_executes_second_real_fixture() -> anyhow::Result<()> {
    let root = ancestor_package()?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let session = engine.create_session()?;
    let result = engine.generate_in_session(
        session,
        GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![2, 1]),
            options: GenerateOptions {
                max_new_tokens: 4,
                greedy: false,
                temperature: 1.0,
                seed: Some(7),
                stop_on_eos: false,
                ..GenerateOptions::default()
            },
        },
    )?;
    assert!(!result.token_ids.is_empty());
    assert_eq!(
        engine
            .candidate_tree_diagnostic()
            .expect("candidate-tree diagnostic")
            .topology,
        "ancestor_mask"
    );
    assert!(!engine.take_candidate_tree_block_traces().is_empty());
    Ok(())
}

fn sampling_request(seed: u64, max_new_tokens: usize) -> GenerateRequest {
    GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![2, 1]),
        options: GenerateOptions {
            max_new_tokens,
            greedy: false,
            temperature: 1.0,
            seed: Some(seed),
            stop_on_eos: false,
            ..GenerateOptions::default()
        },
    }
}

const PROPOSAL_ROWS: [[f32; 5]; 6] = [
    [0.25, 0.75, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.1, 0.2, 0.3, 0.4, 0.0],
];

const TARGET_ROWS: [[f32; 5]; 6] = [
    [0.75, 0.25, 0.0, 0.0, 0.0],
    [0.1, 0.1, 0.6, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.6, 0.1],
    [0.05, 0.15, 0.3, 0.4, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.6],
    [0.4, 0.3, 0.2, 0.1, 0.0],
];

fn reference_sample(row: &[f32], random: f32) -> u32 {
    let mut cumulative = 0.0;
    for (token, probability) in row.iter().copied().enumerate() {
        cumulative += probability;
        if random < cumulative {
            return token as u32;
        }
    }
    row.iter().rposition(|value| *value > 0.0).unwrap() as u32
}

fn reference_residual(target: &[f32], proposal: &[f32], random: f32) -> u32 {
    let residual = target
        .iter()
        .zip(proposal)
        .map(|(target, proposal)| (target - proposal).max(0.0))
        .collect::<Vec<_>>();
    let total = residual.iter().sum::<f32>();
    let normalized = residual
        .iter()
        .map(|value| value / total)
        .collect::<Vec<_>>();
    reference_sample(&normalized, random)
}

/// Independent implementation of the fixture's sampling equation. It does not
/// use `SpecTree`, `verify_tree_sampling`, or any engine helper.
fn sampling_reference(seed: u64) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let root = reference_sample(&PROPOSAL_ROWS[0], rng.random());
    let path: Vec<(usize, u32)> = if root == 0 {
        let _ = reference_sample(&PROPOSAL_ROWS[1], rng.random());
        vec![(0, 0), (1, 2)]
    } else {
        let _ = reference_sample(&PROPOSAL_ROWS[2], rng.random());
        let _ = reference_sample(&PROPOSAL_ROWS[4], rng.random());
        vec![(0, 1), (2, 3), (4, 4)]
    };
    let random = (0..=path.len())
        .map(|_| (rng.random::<f32>(), rng.random::<f32>()))
        .collect::<Vec<_>>();
    let mut accepted = Vec::new();
    for (position, (row, token)) in path.iter().copied().enumerate() {
        let acceptance =
            (TARGET_ROWS[row][token as usize] / PROPOSAL_ROWS[row][token as usize]).min(1.0);
        if random[position].0 < acceptance {
            accepted.push(token);
        } else {
            accepted.push(reference_residual(
                &TARGET_ROWS[row],
                &PROPOSAL_ROWS[row],
                random[position].1,
            ));
            return accepted;
        }
    }
    let bonus_row = if root == 0 { 3 } else { 5 };
    accepted.push(reference_sample(
        &TARGET_ROWS[bonus_row],
        random[path.len()].0,
    ));
    accepted
}

#[test]
fn dispatched_sampling_matches_independent_reference_and_distribution_support() -> anyhow::Result<()>
{
    let root = ancestor_package()?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let mut observed = [0_usize; 5];
    for seed in 0..128 {
        let result = engine.generate(sampling_request(seed, 1))?;
        let expected = sampling_reference(seed);
        assert_eq!(result.token_ids, expected[..1], "seed {seed}");
        observed[result.token_ids[0] as usize] += 1;
    }
    assert_eq!(observed[2..], [0, 0, 0]);
    let token_zero = observed[0] as f32 / 128.0;
    assert!(
        (token_zero - 0.75).abs() < 0.1 && observed[1] > 0,
        "root distribution diverged from declared target [0.75, 0.25]: {observed:?}"
    );
    Ok(())
}

#[test]
fn sampling_without_both_declared_distributions_refuses_before_mutation() -> anyhow::Result<()> {
    let root = parent_package(&[-1, -1, 0, 1, 2, 4], &[1, 4, 0, 0, 0, 0, 0])?;
    replace_metadata(&root, |metadata| {
        metadata.replace(
            "    probabilities:\n      proposal: {component: proposer, output: proposal_probabilities}\n      target: {component: target, output: target_probabilities}\n",
            "",
        )
    })?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let error = engine
        .generate(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![1]),
            options: GenerateOptions {
                max_new_tokens: 1,
                greedy: false,
                temperature: 1.0,
                seed: Some(1),
                ..GenerateOptions::default()
            },
        })
        .expect_err("greedy-only candidate tree must reject sampling");
    assert!(
        format!("{error:#}").contains("probabilities is absent"),
        "{error:#}"
    );
    assert!(engine.contract_executions().is_empty());
    assert!(engine.take_candidate_tree_block_traces().is_empty());
    Ok(())
}

#[test]
fn malformed_parent_mask_and_probability_outputs_abort_before_commit() -> anyhow::Result<()> {
    let invalid_parent = parent_package(&[-1, 4, 0, 1, 2, 4], &[1, 4, 0, 0, 0, 0, 0])?;
    let mut engine = Engine::from_dir(&invalid_parent, EngineConfig::default())?;
    let error = engine
        .generate(greedy_request(1))
        .expect_err("forward parent must fail dynamically");
    assert!(
        format!("{error:#}").contains("not a preceding candidate"),
        "{error:#}"
    );
    assert!(engine.contract_executions().is_empty());

    let invalid_mask = ANCESTOR_PROPOSER.replacen(
        "data_type: 9 int32_data: 1",
        "data_type: 9 int32_data: 0",
        1,
    );
    let root = ancestor_package_with_sources(&invalid_mask, ANCESTOR_TARGET)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let error = engine
        .generate(sampling_request(1, 1))
        .expect_err("missing self edge must fail dynamically");
    assert!(format!("{error:#}").contains("self edge"), "{error:#}");
    assert!(engine.contract_executions().is_empty());

    let invalid_probability = ANCESTOR_PROPOSER.replacen("float_data: 0.25", "float_data: 0.35", 1);
    let root = ancestor_package_with_sources(&invalid_probability, ANCESTOR_TARGET)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let error = engine
        .generate(sampling_request(1, 1))
        .expect_err("unnormalized proposal row must fail dynamically");
    assert!(format!("{error:#}").contains("sums to"), "{error:#}");
    assert!(engine.contract_executions().is_empty());
    Ok(())
}

#[test]
fn raw_pipeline_apis_typed_refuse_before_candidate_components_run() -> anyhow::Result<()> {
    let root = parent_package(&[-1, -1, 0, 1, 2, 4], &[1, 4, 0, 0, 0, 0, 0])?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let request = || PipelineGenerateRequest::new(greedy_request(1));
    for (operation, error) in [
        (
            "Engine::run_pipeline",
            expect_error(engine.run_pipeline(request()), "raw run must fail"),
        ),
        (
            "Engine::run_pipeline_outputs",
            expect_error(
                engine.run_pipeline_outputs(request()),
                "raw output run must fail",
            ),
        ),
        (
            "Engine::run_pipeline_retained",
            expect_error(
                engine.run_pipeline_retained(request()),
                "raw retained run must fail",
            ),
        ),
    ] {
        assert!(matches!(
            package_execution_error(&error),
            Some(PackageExecutionError::CandidateTreeRawWorkflowApi {
                operation: actual
            }) if actual == operation
        ));
    }
    assert!(engine.contract_executions().is_empty());
    Ok(())
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

fn composed_control_at(
    boundary: GenerationBoundary,
    occurrence: usize,
    cancellation: bool,
) -> GenerationControl {
    let seen = AtomicUsize::new(0);
    let control = GenerationControl::new();
    let cancellation_control = control.clone();
    control.with_checkpoint(move |observed| {
        if observed == boundary && seen.fetch_add(1, Ordering::AcqRel) + 1 == occurrence {
            if cancellation {
                cancellation_control.cancel();
            } else {
                anyhow::bail!("injected composed {boundary} failure");
            }
        }
        Ok(())
    })
}

struct ComposedSecondTurn {
    tokens: Vec<u32>,
    candidates: Vec<u32>,
    setup_state: Vec<i64>,
    branch_state: Vec<i64>,
    root_state: Vec<i64>,
}

fn clean_composed_second_turn(root: &Path) -> anyhow::Result<ComposedSecondTurn> {
    let mut engine = Engine::from_dir(root, EngineConfig::default())?;
    let session = engine.create_session()?;
    engine.generate_in_session(session, greedy_request(1))?;
    engine.take_candidate_tree_block_traces();
    engine.take_committed_workflow_publications();
    let result = engine.generate_in_session(session, greedy_request(1))?;
    let candidates = engine.take_candidate_tree_block_traces()[0]
        .candidates
        .clone();
    let publications = engine.take_committed_workflow_publications();
    let WorkflowOutputPublication::Event { payload, .. } = &publications[0] else {
        panic!("composed setup must publish its state event first")
    };
    let setup_state = payload.to_vec_i64()?;
    let WorkflowOutputPublication::Materialized { payload, .. } = &publications[3] else {
        panic!("composed branch state must publish through post_value")
    };
    let branch_state = payload.to_vec_i64()?;
    let WorkflowOutputPublication::Event { payload, .. } = &publications[4] else {
        panic!("composed unrelated root state must publish through root_events")
    };
    Ok(ComposedSecondTurn {
        tokens: result.token_ids,
        candidates,
        setup_state,
        branch_state,
        root_state: payload.to_vec_i64()?,
    })
}

fn assert_composed_controlled_retry(
    boundary: GenerationBoundary,
    occurrence: usize,
    cancellation: bool,
) -> anyhow::Result<()> {
    let root = retryable_composed_parent_package()?;
    let expected = clean_composed_second_turn(&root)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let session = engine.create_session()?;
    engine.generate_in_session(session, greedy_request(1))?;
    engine.take_candidate_tree_block_traces();
    engine.take_committed_workflow_publications();
    let baseline = engine.session_token_count(session)?;
    let effect_baseline = engine
        .workflow_session_effect_cursor(&session.to_string(), "audit")
        .expect("first composed turn commits its audit effect");
    let control = composed_control_at(boundary, occurrence, cancellation);
    let mut delivered = Vec::new();
    let mut callback = |token: onnx_genai_engine::GenerateToken| -> anyhow::Result<()> {
        delivered.push(token.token_id);
        Ok(())
    };
    let error = engine
        .generate_in_session_with_control_callbacks(
            session,
            greedy_request(1),
            control,
            None,
            Some(&mut callback),
        )
        .expect_err("composed candidate-tree turn must abort");
    if cancellation {
        let cancelled = error
            .downcast_ref::<CandidateTreeGenerationCancelled>()
            .expect("typed composed candidate-tree cancellation");
        assert_eq!(cancelled.boundary, boundary);
    } else {
        assert!(
            format!("{error:#}").contains("injected composed"),
            "{error:#}"
        );
    }
    assert!(delivered.is_empty());
    assert_eq!(engine.session_token_count(session)?, baseline);
    assert_eq!(
        engine.workflow_session_effect_cursor(&session.to_string(), "audit"),
        Some(effect_baseline),
        "aborted setup effect must restore its committed cursor"
    );
    assert!(engine.take_candidate_tree_block_traces().is_empty());
    assert!(
        engine.take_committed_workflow_publications().is_empty(),
        "setup/candidate/branch publications must all remain provisional on abort"
    );

    let retry = engine.generate_in_session(session, greedy_request(1))?;
    let trace = engine.take_candidate_tree_block_traces();
    let publications = engine.take_committed_workflow_publications();
    let WorkflowOutputPublication::Event { payload, .. } = &publications[0] else {
        panic!("retry setup must publish its state event first")
    };
    assert_eq!(retry.token_ids, expected.tokens);
    assert_eq!(trace[0].candidates, expected.candidates);
    assert_eq!(
        payload.to_vec_i64()?,
        expected.setup_state,
        "aborted setup state must not advance before deterministic retry"
    );
    let WorkflowOutputPublication::Materialized { payload, .. } = &publications[3] else {
        panic!("retry branch state must publish through post_value")
    };
    assert_eq!(
        payload.to_vec_i64()?,
        expected.branch_state,
        "aborted branch-local state must not advance before deterministic retry"
    );
    let WorkflowOutputPublication::Event { payload, .. } = &publications[4] else {
        panic!("retry unrelated root state must publish through root_events")
    };
    assert_eq!(
        payload.to_vec_i64()?,
        expected.root_state,
        "aborted unrelated root state must not advance before deterministic retry"
    );
    assert_eq!(
        engine.workflow_session_effect_cursor(&session.to_string(), "audit"),
        Some(effect_baseline + 1),
        "retry must commit the setup effect once"
    );
    Ok(())
}

fn clean_second_turn(root: &Path) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    let mut engine = Engine::from_dir(root, EngineConfig::default())?;
    let session = engine.create_session()?;
    engine.generate_in_session(session, greedy_request(1))?;
    engine.take_candidate_tree_block_traces();
    let result = engine.generate_in_session(session, greedy_request(1))?;
    let candidates = engine.take_candidate_tree_block_traces()[0]
        .candidates
        .clone();
    Ok((result.token_ids, candidates))
}

fn assert_controlled_retry(boundary: GenerationBoundary, cancellation: bool) -> anyhow::Result<()> {
    let root = parent_package(&[-1, -1, 0, 1, 2, 4], &[1, 4, 0, 0, 0, 0, 0])?;
    let expected = clean_second_turn(&root)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let session = engine.create_session()?;
    engine.generate_in_session(session, greedy_request(1))?;
    engine.take_candidate_tree_block_traces();
    let baseline = engine.session_token_count(session)?;
    let control = if cancellation {
        cancellation_at(boundary)
    } else {
        fault_at(boundary)
    };
    let mut delivered = Vec::new();
    let mut callback = |token: onnx_genai_engine::GenerateToken| -> anyhow::Result<()> {
        delivered.push(token.token_id);
        Ok(())
    };
    let error = engine
        .generate_in_session_with_control_callbacks(
            session,
            greedy_request(1),
            control,
            None,
            Some(&mut callback),
        )
        .expect_err("controlled candidate-tree turn must abort");
    if cancellation {
        let cancelled = error
            .downcast_ref::<CandidateTreeGenerationCancelled>()
            .expect("typed candidate-tree cancellation");
        assert_eq!(cancelled.boundary, boundary);
    } else {
        assert!(format!("{error:#}").contains("injected"), "{error:#}");
    }
    assert!(delivered.is_empty());
    assert_eq!(engine.session_token_count(session)?, baseline);
    assert!(engine.take_candidate_tree_block_traces().is_empty());
    let retry = engine.generate_in_session(session, greedy_request(1))?;
    let trace = engine.take_candidate_tree_block_traces();
    assert_eq!(retry.token_ids, expected.0);
    assert_eq!(trace[0].candidates, expected.1);
    Ok(())
}

#[test]
fn fault_and_cancellation_restore_state_rng_and_staged_output_for_retry() -> anyhow::Result<()> {
    for boundary in [
        GenerationBoundary::AfterProposer,
        GenerationBoundary::AfterVerifier,
        GenerationBoundary::BeforeAcceptedPathCommit,
        GenerationBoundary::BeforeSemanticCommit,
    ] {
        assert_controlled_retry(boundary, false)?;
        assert_controlled_retry(boundary, true)?;
    }
    Ok(())
}

#[test]
fn composed_faults_and_cancellation_restore_setup_branch_state_effects_and_outputs()
-> anyhow::Result<()> {
    for (boundary, occurrence) in [
        (GenerationBoundary::AfterLoopSetup, 1),
        (GenerationBoundary::AfterVerifier, 1),
        (GenerationBoundary::AfterBranch, 2),
        (GenerationBoundary::BeforeSemanticCommit, 1),
    ] {
        assert_composed_controlled_retry(boundary, occurrence, false)?;
        assert_composed_controlled_retry(boundary, occurrence, true)?;
    }
    Ok(())
}

#[test]
fn committed_outputs_are_ordered_exactly_once_and_callback_failure_is_delivery_only()
-> anyhow::Result<()> {
    let root = parent_package(&[-1, -1, 0, 1, 2, 4], &[2, 0, 4, 0, 0, 0, 0])?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let result = engine.generate(greedy_request(3))?;
    assert_eq!(result.token_ids, vec![2, 4, 0]);
    let publications = engine.take_committed_workflow_publications();
    assert_eq!(publications.len(), 3);
    for (index, publication) in publications.iter().enumerate() {
        let WorkflowOutputPublication::Event {
            sequence,
            payload,
            finality,
            ..
        } = publication
        else {
            panic!("parent fixture must publish discrete token events")
        };
        assert_eq!(sequence.0, index as u64 + 1);
        assert_eq!(
            payload.to_vec_i64()?,
            vec![i64::from(result.token_ids[index])]
        );
        assert_eq!(*finality, OutputFinality::Final);
    }

    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let session = engine.create_session()?;
    let mut delivered = Vec::new();
    let mut callback = |token: onnx_genai_engine::GenerateToken| -> anyhow::Result<()> {
        delivered.push(token.token_id);
        anyhow::bail!("injected delivery failure")
    };
    let error = engine
        .generate_in_session_with_callback(session, greedy_request(3), Some(&mut callback))
        .expect_err("callback failure must be post-commit");
    assert!(format!("{error:#}").contains("after semantic commit"));
    assert_eq!(delivered, vec![2]);
    assert_eq!(engine.session_token_count(session)?, 3);
    assert_eq!(
        engine
            .contract_executions()
            .get("onnx-genai.speculative-block")
            .copied(),
        Some(1)
    );
    Ok(())
}

#[test]
fn revision_fixture_finalizes_after_committed_token_appends() -> anyhow::Result<()> {
    let root = ancestor_package()?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let result = engine.generate(sampling_request(17, 3))?;
    let publications = engine.take_committed_workflow_publications();
    assert_eq!(publications.len(), result.token_ids.len() + 1);
    for publication in &publications[..result.token_ids.len()] {
        let WorkflowOutputPublication::Revision(envelope) = publication else {
            panic!("sampling fixture must publish typed revisions")
        };
        assert_eq!(envelope.operation, TypedRevisionOperation::Append);
        assert_eq!(envelope.finality, OutputFinality::Final);
    }
    let WorkflowOutputPublication::Revision(finalize) = publications.last().unwrap() else {
        panic!("revision stream must finalize")
    };
    assert_eq!(finalize.operation, TypedRevisionOperation::Finalize);
    assert_eq!(finalize.finality, OutputFinality::Final);
    Ok(())
}

#[test]
fn eos_context_fork_and_batching_preserve_candidate_tree_boundaries() -> anyhow::Result<()> {
    let root = parent_package(&[-1, -1, 0, 1, 2, 4], &[2, 0, 4, 0, 0, 0, 0])?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let mut eos = greedy_request(5);
    eos.options.stop_on_eos = true;
    eos.options.eos_token_ids = vec![2];
    let result = engine.generate(eos)?;
    assert_eq!(result.token_ids, vec![2]);
    assert_eq!(result.finish_reason, FinishReason::EosToken);

    let mut exhausted = greedy_request(2);
    exhausted.options.max_context = Some(2);
    let result = engine.generate(exhausted)?;
    assert!(result.token_ids.is_empty());
    assert_eq!(result.finish_reason, FinishReason::Length);

    let session = engine.create_session()?;
    let error = engine
        .prepare_session_fork(session, SessionPosition::new(0))
        .expect_err("candidate-tree fork must decline before child allocation");
    assert!(matches!(
        error,
        SessionForkError::UnsupportedParticipant { participant, reason, .. }
            if participant == "candidate_tree.accepted_path"
                && reason.contains("before child allocation")
    ));

    let expected = engine.generate(greedy_request(2))?;
    let batch = engine.generate_batched_static(vec![greedy_request(2), greedy_request(2)])?;
    assert_eq!(batch.len(), 2);
    assert!(
        batch
            .iter()
            .all(|result| result.token_ids == expected.token_ids)
    );
    Ok(())
}
