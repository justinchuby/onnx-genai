# Runtime capability catalogue

This is the metadata crate's synchronized reader vocabulary, not a second
normative specification. Portable semantics live in
[`INFERENCE_METADATA_DECISIONS.md`](INFERENCE_METADATA_DECISIONS.md); this list
exists so releases and tests can detect when the implementation's built-in
admission identifiers drift. Extension identifiers remain open and namespaced.

<!-- capability-catalogue:start -->
| Identifier |
|---|
| `kv_cache` |
| `grouped_query_attention` |
| `multi_head_attention` |
| `prefix_cache` |
| `control_flow_loop` |
| `image_preprocessing_program` |
| `packed_image_outputs` |
| `position_program` |
| `multi_axis_positions` |
| `loop_carried_state` |
| `dual_sequence_inputs` |
| `workflow_ssa` |
| `linear_effects` |
| `serving_service_contract` |
| `parameter_adapters` |
| `heterogeneous_adapter_batching` |
| `session_state_lease` |
| `bounded_state_recurrence` |
| `advisory_state` |
| `adaptive_proposal_budget` |
| `grammar_guidance_adapter` |
| `telemetry_adapter` |
| `nested_control_flow` |
| `loop_induction_values` |
| `typed_emit` |
| `emit_valid_length` |
| `input_presence` |
| `explicit_transfer` |
| `token_context` |
<!-- capability-catalogue:end -->
