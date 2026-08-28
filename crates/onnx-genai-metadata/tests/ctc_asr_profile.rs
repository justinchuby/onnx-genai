//! End-to-end coverage for the architecture-neutral CTC ASR (speech-to-text)
//! contract: an audio preprocessing program that turns raw request bytes into
//! an encoder's input tensors, and a frame-synchronous CTC decoding contract
//! that turns the encoder's per-frame logits into a transcript (frame argmax
//! -> collapse repeats -> drop blank -> detokenize). Nothing here names a
//! model family; a runtime dispatches purely on the generic `preprocessing
//! .audio` transform vocabulary and `profiles.*.decoding` shape.

use onnx_genai_metadata::{InferenceMetadata, validate_metadata};

fn parse(document: &str) -> InferenceMetadata {
    serde_yaml::from_str(document).expect("metadata parses")
}

fn errors(document: &str) -> Vec<String> {
    validate_metadata(&parse(document)).expect_err("metadata must be rejected")
}

/// A wav2vec2-CTC-style package: one `onnx-genai.audio-preprocess` adapter,
/// one ONNX acoustic encoder invoked exactly once (no generation loop), and a
/// `transcription` profile that decodes the encoder's per-frame logits with
/// CTC. Model identity lives only in this comment and the artifact path; the
/// schema and validator never see a model name.
const CTC_ASR_DOCUMENT: &str = r#"
schema_version: v1.1
preprocessing:
  audio:
    transforms:
      - op: decode
        outputs: [audio.transform_0]
      - op: resample
        sample_rate: 16000
        inputs: [audio.transform_0]
        outputs: [audio.transform_1]
      - op: downmix
        channels: 1
        inputs: [audio.transform_1]
        outputs: [audio.transform_2]
      - op: zero_mean_unit_variance
        epsilon: 1.0e-7
        inputs: [audio.transform_2]
        outputs: [audio.transform_3]
      - op: pad
        mode: right
        pad_value: 0.0
        inputs: [audio.transform_3]
        outputs: [audio.transform_4]
      - op: emit_validity_mask
        inputs: [audio.transform_4]
        outputs: [audio.output_validity_mask]
    outputs:
      - name: input_values
        source: audio.transform_4
        content: waveform
        dtype: float32
        contract: { dtype: float32, shape: [batch, samples] }
      - name: attention_mask
        source: audio.output_validity_mask
        content: validity_mask
        dtype: int64
        contract: { dtype: int64, shape: [batch, samples] }
profiles:
  transcription:
    kind: transcription
    version: "1.0"
    outputs: { logits: logits, frame_lengths: frame_lengths }
    decoding:
      kind: ctc
      blank_id: 0
      collapse_repeats: true
      time_axis: 1
      class_axis: 2
      lengths: frame_lengths
      vocabulary:
        source: tokenizer
        size: 32
        word_delimiter: "|"
        ignored_tokens: ["<pad>", "<s>", "</s>", "<unk>"]
pipeline:
  workflow:
    manifest:
      adapter_abis: { onnx-genai.audio-preprocess: "1" }
      capabilities: [workflow_ssa, typed_emit]
    inputs:
      request.audio:
        contract: { dtype: uint8, shape: [encoded_bytes] }
        role: { kind: runtime, version: "1.0", role: media }
        source: { kind: request }
    outputs:
      logits:
        contract:
          dtype: float32
          shape: [batch, frames, vocab]
          batch_layout: { kind: request_aligned, axis: 0 }
          padding: [{ dimension: frames, valid_lengths: frame_lengths }]
        role: tensor
        stage: pre_adapter
      frame_lengths:
        contract:
          dtype: int64
          shape: [batch]
          batch_layout: { kind: shared }
        role: tensor
        stage: pre_adapter
    components:
      audio_preprocess:
        implementation:
          kind: adapter
          abi: onnx-genai.audio-preprocess
          version: "1"
        ports:
          inputs:
            encoded: { dtype: uint8, shape: [encoded_bytes] }
          outputs:
            input_values: { dtype: float32, shape: [batch, samples] }
            attention_mask: { dtype: int64, shape: [batch, samples] }
      encoder:
        implementation: { kind: onnx, artifact: encoder/model.onnx }
    steps:
      - kind: invoke
        component: audio_preprocess
        inputs: { encoded: request.audio }
        outputs: { input_values: input_values, attention_mask: attention_mask }
      - kind: invoke
        component: encoder
        inputs: { input_values: input_values, attention_mask: attention_mask }
        outputs: { logits: raw_logits, frame_lengths: raw_frame_lengths }
      - kind: emit
        value: raw_logits
        output: logits
        mode: replace
      - kind: emit
        value: raw_frame_lengths
        output: frame_lengths
        mode: replace
"#;

#[test]
fn ctc_asr_document_validates() {
    let metadata = parse(CTC_ASR_DOCUMENT);
    validate_metadata(&metadata).expect("a complete CTC ASR document validates");

    let profile = &metadata.profiles["transcription"];
    assert_eq!(profile.kind, "transcription");
    let decoding = profile.decoding.as_ref().expect("decoding contract");
    assert_eq!(decoding.kind, "ctc");
    assert_eq!(decoding.blank_id, Some(0));
    assert!(decoding.collapse_repeats);
    assert_eq!(decoding.lengths.as_deref(), Some("frame_lengths"));
    let logits_output = profile
        .outputs
        .get("logits")
        .expect("CTC profile exposes the canonical logits role");
    assert_eq!(logits_output, "logits");
    let lengths_role = decoding
        .lengths
        .as_deref()
        .expect("padded CTC lengths role");
    let lengths_output = &profile.outputs[lengths_role];
    let workflow = &metadata.pipeline.as_ref().expect("pipeline").workflow;
    let padding = &workflow.outputs[logits_output].contract.padding[0];
    assert_eq!(
        lengths_output, &padding.valid_lengths,
        "CTC decoding and the padded time axis use one length source"
    );
    let vocabulary = decoding.vocabulary.as_ref().expect("vocabulary");
    assert_eq!(vocabulary.source, "tokenizer");
    assert_eq!(vocabulary.size, Some(32));

    let audio = metadata
        .preprocessing
        .as_ref()
        .and_then(|preprocessing| preprocessing.audio.as_ref())
        .expect("audio preprocessing program");
    assert_eq!(audio.transforms.len(), 6);
    assert_eq!(audio.outputs.len(), 2);
}

#[test]
fn unknown_profile_kind_with_ignorable_requirement_still_validates() {
    // Proves the skip rule from `validate_profiles` still applies once
    // `transcription` becomes a known kind: an entirely different, unknown
    // kind is still ignorable, and its absent `decoding` block is never
    // interpreted by `validate_profile_decoding` either.
    let document = r#"
profiles:
  future_asr:
    kind: some.future.transcription
    version: "1"
    requirement: ignorable
"#;
    validate_metadata(&parse(document)).expect("an ignorable unknown profile kind is skippable");
}

#[test]
fn ctc_decoding_without_blank_id_is_rejected() {
    let document = r#"
profiles:
  transcription:
    kind: transcription
    version: "1.0"
    decoding:
      kind: ctc
      time_axis: 1
      class_axis: 2
"#;
    let reported = errors(document);
    assert!(
        reported.iter().any(|error| error.contains(
            "profiles.transcription.decoding requires blank_id \
                 because kind is 'ctc'"
        )),
        "{reported:?}"
    );
}

#[test]
fn decoding_time_axis_equal_to_class_axis_is_rejected() {
    let document = r#"
profiles:
  transcription:
    kind: transcription
    version: "1.0"
    decoding:
      kind: ctc
      blank_id: 0
      time_axis: 1
      class_axis: 1
"#;
    let reported = errors(document);
    assert!(
        reported.iter().any(|error| error.contains(
            "profiles.transcription.decoding.time_axis and class_axis must not both be axis 1"
        )),
        "{reported:?}"
    );
}

#[test]
fn decoding_lengths_role_must_be_a_declared_profile_output() {
    let document = r#"
profiles:
  transcription:
    kind: transcription
    version: "1.0"
    decoding:
      kind: ctc
      blank_id: 0
      time_axis: 1
      class_axis: 2
      lengths: frame_lengths
"#;
    let reported = errors(document);
    assert!(
        reported.iter().any(|error| error.contains(
            "profiles.transcription.decoding references output role 'frame_lengths' that the \
             profile does not declare"
        )),
        "{reported:?}"
    );
}

#[test]
fn transcription_profile_without_decoding_is_rejected() {
    let document = r#"
profiles:
  transcription:
    kind: transcription
    version: "1.0"
"#;
    let reported = errors(document);
    assert!(
        reported.iter().any(|error| error.contains(
            "profiles.transcription.decoding is required because profiles.transcription.kind \
             is 'transcription'"
        )),
        "{reported:?}"
    );
}

#[test]
fn inline_vocabulary_with_empty_tokens_is_rejected() {
    let document = r#"
profiles:
  transcription:
    kind: transcription
    version: "1.0"
    decoding:
      kind: ctc
      blank_id: 0
      time_axis: 1
      class_axis: 2
      vocabulary:
        source: inline
"#;
    let reported = errors(document);
    assert!(
        reported.iter().any(|error| error.contains(
            "profiles.transcription.decoding.vocabulary requires non-empty tokens because \
             source is 'inline'"
        )),
        "{reported:?}"
    );
}

#[test]
fn vocabulary_size_disagreeing_with_tokens_length_is_rejected() {
    let document = r#"
profiles:
  transcription:
    kind: transcription
    version: "1.0"
    decoding:
      kind: ctc
      blank_id: 0
      time_axis: 1
      class_axis: 2
      vocabulary:
        source: tokenizer
        size: 32
        tokens: ["a", "b", "c"]
"#;
    let reported = errors(document);
    assert!(
        reported.iter().any(|error| error.contains(
            "profiles.transcription.decoding.vocabulary size 32 disagrees with tokens length 3"
        )),
        "{reported:?}"
    );
}

#[test]
fn audio_adapter_without_preprocessing_audio_metadata_is_rejected() {
    let document = r#"
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa]
    inputs:
      request.audio:
        contract: { dtype: uint8, shape: [encoded_bytes] }
        role: { kind: runtime, version: "1.0", role: media }
        source: { kind: request }
    components:
      audio_preprocess:
        implementation: { kind: adapter, abi: onnx-genai.audio-preprocess, version: "1" }
        ports:
          inputs:
            encoded: { dtype: uint8, shape: [encoded_bytes] }
          outputs:
            input_values: { dtype: float32, shape: [batch, samples] }
    steps:
      - kind: invoke
        component: audio_preprocess
        inputs: { encoded: request.audio }
        outputs: { input_values: input_values }
"#;
    let reported = errors(document);
    assert!(
        reported.iter().any(|error| error.contains(
            "workflow adapter components using onnx-genai.audio-preprocess@1 require \
             preprocessing.audio metadata"
        )),
        "{reported:?}"
    );
}

#[test]
fn audio_output_not_produced_by_adapter_is_rejected() {
    let document = r#"
preprocessing:
  audio:
    outputs:
      - name: mystery_output
        source: audio.transform_0
        content: waveform
        dtype: float32
        contract: { dtype: float32, shape: [batch, samples] }
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa]
    inputs:
      request.audio:
        contract: { dtype: uint8, shape: [encoded_bytes] }
        role: { kind: runtime, version: "1.0", role: media }
        source: { kind: request }
    components:
      audio_preprocess:
        implementation: { kind: adapter, abi: onnx-genai.audio-preprocess, version: "1" }
        ports:
          inputs:
            encoded: { dtype: uint8, shape: [encoded_bytes] }
          outputs:
            input_values: { dtype: float32, shape: [batch, samples] }
    steps:
      - kind: invoke
        component: audio_preprocess
        inputs: { encoded: request.audio }
        outputs: { input_values: input_values }
"#;
    let reported = errors(document);
    assert!(
        reported.iter().any(|error| error.contains(
            "preprocessing.audio output 'mystery_output' must be a declared SSA output of \
             adapter invocation 'audio_preprocess'"
        )),
        "{reported:?}"
    );
}

#[test]
fn word_delimiter_absent_from_inline_tokens_is_rejected() {
    let document = r#"
profiles:
  transcription:
    kind: transcription
    version: "1.0"
    decoding:
      kind: ctc
      blank_id: 0
      time_axis: 1
      class_axis: 2
      vocabulary:
        source: inline
        word_delimiter: "|"
        tokens: ["<pad>", "a", "b"]
"#;
    let reported = errors(document);
    assert!(
        reported.iter().any(|error| error.contains(
            "profiles.transcription.decoding.vocabulary.word_delimiter '|' is not present in \
             tokens"
        )),
        "{reported:?}"
    );
}

#[test]
fn ignored_token_absent_from_inline_tokens_is_rejected() {
    let document = r#"
profiles:
  transcription:
    kind: transcription
    version: "1.0"
    decoding:
      kind: ctc
      blank_id: 0
      time_axis: 1
      class_axis: 2
      vocabulary:
        source: inline
        ignored_tokens: ["<s>"]
        tokens: ["<pad>", "a", "b"]
"#;
    let reported = errors(document);
    assert!(
        reported.iter().any(|error| error.contains(
            "profiles.transcription.decoding.vocabulary.ignored_tokens entry '<s>' is not \
             present in tokens"
        )),
        "{reported:?}"
    );
}

#[test]
fn blank_id_outside_inline_vocabulary_is_rejected() {
    let document = r#"
profiles:
  transcription:
    kind: transcription
    version: "1.0"
    decoding:
      kind: ctc
      blank_id: 7
      time_axis: 1
      class_axis: 2
      vocabulary:
        source: inline
        tokens: ["<pad>", "a", "b"]
"#;
    let reported = errors(document);
    assert!(
        reported.iter().any(|error| error.contains(
            "profiles.transcription.decoding.blank_id 7 is out of range for a vocabulary of 3 \
             tokens"
        )),
        "{reported:?}"
    );
}

#[test]
fn padded_ctc_without_lengths_binding_is_rejected() {
    let document = CTC_ASR_DOCUMENT.replace("      lengths: frame_lengths\n", "");
    let reported = errors(&document);
    assert!(
        reported.iter().any(|error| error.contains(
            "profiles.transcription.decoding.lengths is required because workflow output \
             'logits' pads decoded time axis 1 ('frames') with valid_lengths 'frame_lengths'"
        )),
        "{reported:?}"
    );
}

#[test]
fn ctc_cannot_alias_away_the_canonical_logits_role() {
    let document = CTC_ASR_DOCUMENT
        .replace(
            "    outputs: { logits: logits, frame_lengths: frame_lengths }",
            "    outputs: { emissions: logits, frame_lengths: frame_lengths }",
        )
        .replace("      lengths: frame_lengths", "      lengths: emissions");
    let reported = errors(&document);
    assert!(
        reported.iter().any(|error| error.contains(
            "profiles.transcription.outputs.logits is required because CTC decoding reads the \
             canonical 'logits' role; map that role to the workflow output containing the \
             frame-by-class logits tensor"
        )),
        "{reported:?}"
    );
}

#[test]
fn padded_ctc_cannot_bind_a_different_length_source() {
    let document =
        CTC_ASR_DOCUMENT.replace("      lengths: frame_lengths", "      lengths: logits");
    let reported = errors(&document);
    assert!(
        reported.iter().any(|error| error.contains(
            "decoding.lengths role 'logits' binds workflow output 'logits', but workflow output \
             'logits' pads decoded time axis 1 ('frames') with valid_lengths 'frame_lengths'"
        )),
        "{reported:?}"
    );
}

#[test]
fn padded_ctc_length_companion_must_be_int64_at_the_time_prefix_rank() {
    let wrong_dtype = CTC_ASR_DOCUMENT.replace(
        "      frame_lengths:\n        contract:\n          dtype: int64",
        "      frame_lengths:\n        contract:\n          dtype: float32",
    );
    let reported = errors(&wrong_dtype);
    assert!(
        reported.iter().any(|error| error.contains(
            "workflow output 'logits' valid_lengths 'frame_lengths' is float32 but must be int64"
        )),
        "{reported:?}"
    );

    let wrong_rank = CTC_ASR_DOCUMENT.replace(
        "          shape: [batch]\n          batch_layout: { kind: shared }",
        "          shape: [batch, extra]\n          batch_layout: { kind: shared }",
    );
    let reported = errors(&wrong_rank);
    assert!(
        reported.iter().any(|error| error.contains(
            "workflow output 'logits' valid_lengths 'frame_lengths' has rank 2 but dimension \
             'frames' is axis 1, so it must have rank 1"
        )),
        "{reported:?}"
    );
}

#[test]
fn unpadded_ctc_needs_no_lengths_binding() {
    let document = CTC_ASR_DOCUMENT
        .replace(
            "          padding: [{ dimension: frames, valid_lengths: frame_lengths }]\n",
            "",
        )
        .replace("      lengths: frame_lengths\n", "");
    validate_metadata(&parse(&document)).expect("an unpadded CTC time axis needs no lengths");
}

#[test]
fn padding_a_non_time_dimension_does_not_require_ctc_lengths() {
    let document = CTC_ASR_DOCUMENT
        .replace(
            "padding: [{ dimension: frames, valid_lengths: frame_lengths }]",
            "padding: [{ dimension: vocab, valid_lengths: vocab_lengths }]",
        )
        .replace("frame_lengths", "vocab_lengths")
        .replace(
            "          shape: [batch]\n          batch_layout: { kind: shared }",
            "          shape: [batch, frames]\n          batch_layout: { kind: shared }",
        )
        .replace("      lengths: vocab_lengths\n", "");
    validate_metadata(&parse(&document))
        .expect("padding outside the decoded time axis does not require frame lengths");
}
