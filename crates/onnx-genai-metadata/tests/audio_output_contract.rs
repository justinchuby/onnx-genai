use onnx_genai_metadata::{InferenceMetadata, validate_metadata};

const AUDIO_WORKFLOW: &str = r#"
schema_version: v1
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, typed_emit]
    inputs: {}
    outputs:
      audio:
        contract: { dtype: uint8, rank: 1, shape: [wav_bytes] }
        role: audio
        stage: post_adapter
        media:
          container: wav
          encoding: pcm_s16_le
          sample_rate_hz: 32000
          channels: 2
          delivery: buffered
    components:
      synthesize:
        implementation: { kind: binding }
        ports:
          outputs:
            wav: { dtype: uint8, rank: 1, shape: [wav_bytes] }
    steps:
      - kind: invoke
        component: synthesize
        outputs: { wav: result.wav }
      - kind: emit
        value: result.wav
        output: audio
        mode: replace
"#;

#[test]
fn buffered_stereo_wav_contract_parses_and_validates() {
    let metadata: InferenceMetadata =
        serde_yaml::from_str(AUDIO_WORKFLOW).expect("audio workflow parses");
    validate_metadata(&metadata).expect("audio workflow validates");
}

#[test]
fn wav_contract_requires_encoded_bytes_and_physical_audio_properties() {
    for invalid in [
        AUDIO_WORKFLOW.replace("dtype: uint8", "dtype: float32"),
        AUDIO_WORKFLOW.replace("sample_rate_hz: 32000", "sample_rate_hz: 0"),
        AUDIO_WORKFLOW.replace("channels: 2", "channels: 0"),
        AUDIO_WORKFLOW.replace("role: audio", "role: tensor"),
    ] {
        let metadata: InferenceMetadata =
            serde_yaml::from_str(&invalid).expect("invalid audio workflow still parses");
        assert!(
            validate_metadata(&metadata).is_err(),
            "invalid media contract must fail admission"
        );
    }
}

#[test]
fn hierarchical_audio_workflow_admits_nested_frame_codebook_and_flow_loops() {
    let document = r#"
schema_version: v1
pipeline:
  workflow:
    manifest:
      capabilities:
        - workflow_ssa
        - nested_control_flow
        - loop_induction_values
        - loop_carried_state
        - typed_emit
    inputs:
      active:
        contract: { dtype: bool, rank: 0, shape: [] }
        role: { kind: opaque }
        source: { kind: application, name: active }
        required: true
      frame_limit:
        contract: { dtype: int64, rank: 0, shape: [] }
        role: { kind: opaque }
        source: { kind: application, name: frame_limit }
        required: true
      codebook_limit:
        contract: { dtype: int64, rank: 0, shape: [] }
        role: { kind: opaque }
        source: { kind: literal }
        required: true
      chunk_limit:
        contract: { dtype: int64, rank: 0, shape: [] }
        role: { kind: opaque }
        source: { kind: application, name: chunk_limit }
        required: true
      flow_step_limit:
        contract: { dtype: int64, rank: 0, shape: [] }
        role: { kind: opaque }
        source: { kind: literal }
        required: true
      global_cache.initial:
        contract: { dtype: float32, rank: 5, shape: [layers, 2, batch, heads, sequence] }
        role: { kind: opaque }
        source: { kind: application, name: global_cache }
        required: true
    outputs:
      audio:
        contract: { dtype: uint8, rank: 1, shape: [wav_bytes] }
        role: audio
        stage: post_adapter
        media:
          container: wav
          encoding: pcm_s16_le
          sample_rate_hz: 32000
          channels: 2
          delivery: buffered
    components:
      global_decoder:
        implementation: { kind: binding }
        ports:
          inputs:
            cache: { dtype: float32, rank: 5, shape: [layers, 2, batch, heads, sequence] }
          outputs:
            cache: { dtype: float32, rank: 5, shape: [layers, 2, batch, heads, sequence] }
            hidden: { dtype: float32, rank: 2, shape: [batch, hidden] }
            semantic_code: { dtype: int64, rank: 1, shape: [batch] }
      local_codebook_decoder:
        implementation: { kind: binding }
        ports:
          inputs:
            global_hidden: { dtype: float32, rank: 2, shape: [batch, hidden] }
            semantic_code: { dtype: int64, rank: 1, shape: [batch] }
            codebook_index: { dtype: int64, rank: 0, shape: [] }
          outputs:
            local_hidden: { dtype: float32, rank: 2, shape: [batch, hidden] }
            residual_code: { dtype: int64, rank: 1, shape: [batch] }
      flow_step:
        implementation: { kind: binding }
        ports:
          inputs:
            chunk_index: { dtype: int64, rank: 0, shape: [] }
            step_index: { dtype: int64, rank: 0, shape: [] }
          outputs:
            latent: { dtype: float32, rank: 3, shape: [batch, channels, latent_sequence] }
      buffered_wav:
        implementation: { kind: binding }
        ports:
          outputs:
            wav: { dtype: uint8, rank: 1, shape: [wav_bytes] }
    state:
      global_cache:
        contract: { dtype: float32, rank: 5, shape: [layers, 2, batch, heads, sequence] }
        scope: invocation
        initializer: global_cache.initial
        recurrence: { kind: invariant }
    steps:
      - kind: loop
        continue_when: active
        max_iterations: frame_limit
        iteration:
          value: frame_index
          contract: { dtype: int64, rank: 0, shape: [] }
        carried:
          - { cell: global_cache, next: global.cache.next }
        steps:
          - kind: invoke
            component: global_decoder
            inputs: { cache: global_cache }
            outputs:
              cache: global.cache.next
              hidden: frame.global_hidden
              semantic_code: frame.semantic_code
          - kind: loop
            continue_when: active
            max_iterations: codebook_limit
            iteration:
              value: codebook_index
              contract: { dtype: int64, rank: 0, shape: [] }
            steps:
              - kind: invoke
                component: local_codebook_decoder
                inputs:
                  global_hidden: frame.global_hidden
                  semantic_code: frame.semantic_code
                  codebook_index: codebook_index
                outputs:
                  local_hidden: frame.local_hidden
                  residual_code: frame.residual_code
      - kind: loop
        continue_when: active
        max_iterations: chunk_limit
        iteration:
          value: chunk_index
          contract: { dtype: int64, rank: 0, shape: [] }
        steps:
          - kind: loop
            continue_when: active
            max_iterations: flow_step_limit
            iteration:
              value: flow_step_index
              contract: { dtype: int64, rank: 0, shape: [] }
            steps:
              - kind: invoke
                component: flow_step
                inputs:
                  chunk_index: chunk_index
                  step_index: flow_step_index
                outputs: { latent: flow.latent }
      - kind: invoke
        component: buffered_wav
        outputs: { wav: result.wav }
      - kind: emit
        value: result.wav
        output: audio
        mode: replace
"#;

    let metadata: InferenceMetadata =
        serde_yaml::from_str(document).expect("hierarchical audio workflow parses");
    validate_metadata(&metadata).expect("hierarchical audio workflow validates");
    let capabilities = onnx_genai_metadata::derived_capabilities(&metadata);
    assert!(capabilities.contains("nested_control_flow"));
    assert!(capabilities.contains("loop_induction_values"));

    let workflow = &metadata.pipeline.expect("pipeline").workflow;
    let frame_loop = &workflow.steps[0];
    let serialized = serde_yaml::to_string(frame_loop).expect("frame loop serializes");
    assert_eq!(
        serialized.matches("cell: global_cache").count(),
        1,
        "only the global decoder owns persistent KV state"
    );
    assert!(
        !serialized.contains("local_cache"),
        "the local codebook decoder is stateless growing-length recomputation"
    );
}
