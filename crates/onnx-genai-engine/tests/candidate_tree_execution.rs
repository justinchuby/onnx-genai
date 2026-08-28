use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use onnx_genai_engine::pipeline::speculative::CandidateTreeGenerationCancelled;
use onnx_genai_engine::pipeline::{
    OutputFinality, TypedRevisionOperation, WorkflowOutputPublication,
};
use onnx_genai_engine::{
    Engine, EngineConfig, FinishReason, GenerateOptions, GeneratePrompt, GenerateRequest,
    GenerationBoundary, GenerationControl, PackageCapabilityError, PipelineGenerateRequest,
    SessionForkError, SessionPosition, ToolCallPolicy, package_capability_error,
};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};

const PARENT_METADATA: &str = r#"
schema_version: v1.6
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, serving_service_contract, canonical_speculation, session_state_lease]
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
  distribution_preserving: false
  verification:
    target_output: {component: target, output: target_logits}
    accepted_path: {kind: runtime, binding: accepted_prefix}
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
  node { input: "context_tokens" output: "context_f32" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
  node { input: "context_f32" output: "context_sum" op_type: "ReduceSum" attribute { name: "keepdims" i: 0 type: INT } }
  node { input: "private_state" input: "context_sum" output: "private_state_out" op_type: "Add" }
  name: "candidate_parent_proposer"
  input { name: "context_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_param: "sequence" } } } } }
  input { name: "private_state" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 2 } } } } }
  output { name: "candidate_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 6 } } } } }
  output { name: "candidate_parents" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 6 } } } } }
  output { name: "private_state_out" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 2 } } } } }
}
opset_import { domain: "" version: 18 }
"#;

const PARENT_TARGET: &str = r#"
ir_version: 8
graph {
  node { output: "target_logits" op_type: "Constant" attribute { name: "value" t { dims: 1 dims: 7 dims: 6 data_type: 1 {logit_values} } type: TENSOR } }
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
  output { name: "target_state_out" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } dim { dim_value: 2 } } } } }
  output { name: "token_context_out" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } dim { dim_value: 4 } } } } }
}
opset_import { domain: "" version: 18 }
"#;

const ANCESTOR_METADATA: &str = r#"
schema_version: v1.6
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, serving_service_contract, canonical_speculation, session_state_lease]
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
    let reason = match package_capability_error(&error) {
        Some(PackageCapabilityError::CandidateTreeExecutionUnavailable { reason, .. }) => reason,
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

type MetadataUpdate = Box<dyn FnOnce(String) -> String>;
type RefusalCase = (&'static str, MetadataUpdate, Vec<&'static str>);

#[test]
fn candidate_tree_emit_site_is_refused_before_the_specialized_driver_can_skip_it()
-> anyhow::Result<()> {
    let root = parent_package(&[-1, -1, 0, 1, 2, 4], &[1, 4, 0, 0, 0, 0, 0])?;
    replace_metadata(&root, |metadata| {
        metadata
            .replacen(
                "capabilities: [workflow_ssa, serving_service_contract, canonical_speculation, session_state_lease]",
                "capabilities: [workflow_ssa, typed_emit, serving_service_contract, canonical_speculation, session_state_lease]",
                1,
            )
            .replacen(
                "    components:\n",
                "      side:\n        contract: {dtype: int64, shape: [batch, 6], batch_layout: {kind: request_aligned, axis: 0}}\n        role: tensor\n        family: {kind: events}\n        stage: pre_adapter\n    components:\n",
                1,
            )
            .replacen(
                "    state:\n",
                "    - kind: emit\n      value: proposed.tokens\n      output: side\n      mode: event\n    state:\n",
                1,
            )
    })?;
    assert_preload_candidate_refusal(
        &root,
        &[
            "pipeline.workflow.steps[2]",
            "cannot publish emit",
            "S4 family/site",
        ],
    )?;
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
fn unsupported_candidate_compositions_all_refuse_before_component_loading() -> anyhow::Result<()> {
    let parents = [-1, -1, 0, 1, 2, 4];
    let decisions = [1, 4, 0, 0, 0, 0, 0];
    let cases: Vec<RefusalCase> = vec![
        (
            "empty-loop",
            Box::new(|metadata| {
                with_control_flow_support(metadata).replacen(
                    "    - kind: invoke\n      component: proposer\n",
                    "    - kind: loop\n      setup: []\n      steps:\n      - {kind: invoke, component: observer}\n      continue_when: control.continue\n      max_iterations: control.max\n    - kind: invoke\n      component: proposer\n",
                    1,
                )
            }),
            vec![
                "pipeline.workflow.steps[0]",
                "loop",
                "condition/body ordering",
            ],
        ),
        (
            "setup-loop",
            Box::new(|metadata| {
                with_control_flow_support(metadata).replacen(
                    "    - kind: invoke\n      component: proposer\n",
                    "    - kind: loop\n      setup:\n      - {kind: invoke, component: observer}\n      steps:\n      - {kind: invoke, component: observer}\n      continue_when: control.continue\n      max_iterations: control.max\n    - kind: invoke\n      component: proposer\n",
                    1,
                )
            }),
            vec!["pipeline.workflow.steps[0]", "loop", "non-empty setup"],
        ),
        (
            "branch",
            Box::new(|metadata| {
                with_control_flow_support(metadata).replacen(
                    "    - kind: invoke\n      component: proposer\n",
                    "    - kind: branch\n      predicate: control.continue\n      cases:\n        \"true\": {kind: invoke, component: observer}\n      default: {kind: invoke, component: observer}\n    - kind: invoke\n      component: proposer\n",
                    1,
                )
            }),
            vec!["pipeline.workflow.steps[0]", "branch", "predicate/join"],
        ),
        (
            "unrelated",
            Box::new(|metadata| {
                metadata
                    .replacen(
                        "      proposer:\n",
                        "      observer:\n        implementation: {kind: onnx, artifact: observer.onnx.textproto}\n        ports: {inputs: {}, outputs: {}}\n        row_scope: {axis: 0, stateful: false}\n      proposer:\n",
                        1,
                    )
                    .replacen(
                        "    - kind: invoke\n      component: proposer\n",
                        "    - kind: invoke\n      component: observer\n    - kind: invoke\n      component: proposer\n",
                        1,
                    )
            }),
            vec![
                "pipeline.workflow.steps[0]",
                "unrelated component 'observer'",
            ],
        ),
        (
            "effect",
            Box::new(|metadata| {
                metadata
                    .replacen(
                        "capabilities: [workflow_ssa, serving_service_contract, canonical_speculation, session_state_lease]",
                        "capabilities: [workflow_ssa, linear_effects, serving_service_contract, canonical_speculation, session_state_lease]",
                        1,
                    )
                    .replacen(
                        "    serving:\n",
                        "    effects:\n      audit: {retry: pure}\n    serving:\n",
                        1,
                    )
            }),
            vec![
                "effectful candidate-tree regions",
                "accepted-path transaction",
            ],
        ),
        (
            "extra-output",
            Box::new(|metadata| {
                metadata.replacen(
                    "    components:\n",
                    "      side:\n        contract: {dtype: int64, shape: [batch, 6], batch_layout: {kind: request_aligned, axis: 0}}\n        role: tensor\n        family: {kind: events}\n        stage: pre_adapter\n    components:\n",
                    1,
                )
            }),
            vec![
                "declared output 'side'",
                "no candidate-tree output-publication participant",
            ],
        ),
    ];
    for (name, update, expected) in cases {
        let root = parent_package(&parents, &decisions)?;
        replace_metadata(&root, update)?;
        assert_preload_candidate_refusal(&root, &expected)
            .map_err(|error| anyhow::anyhow!("{name}: {error:#}"))?;
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
            package_capability_error(&error),
            Some(PackageCapabilityError::CandidateTreeRawWorkflowApi {
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
