use onnx_genai_metadata::{
    InferenceMetadata, WorkflowEmitMode, WorkflowOutputFamily, WorkflowStep, validate_metadata,
};

fn fixture(mode: WorkflowEmitMode) -> InferenceMetadata {
    let mut metadata: InferenceMetadata = serde_yaml::from_str(include_str!(
        "../../../examples/inference_metadata/catalogue/01-gemma4-text-decoder.yaml"
    ))
    .expect("catalogue fixture parses");
    metadata.schema_version = Some("v1.5".to_string());
    let workflow = &mut metadata.pipeline.as_mut().expect("pipeline").workflow;
    let output = workflow.outputs.keys().next().expect("output").clone();
    workflow.outputs.get_mut(&output).expect("output").family = WorkflowOutputFamily::Revisions {
        version: "1".to_string(),
    };
    workflow
        .outputs
        .get_mut(&output)
        .expect("output")
        .family_authored = true;
    workflow.steps.push(WorkflowStep::Emit {
        value: "payload_that_must_not_be_discarded".to_string(),
        when: None,
        valid_length: None,
        output,
        stream: Some("named".to_string()),
        mode,
        axis: None,
    });
    metadata
}

#[test]
fn payloadless_revision_operations_reject_values_with_site_and_stream_context() {
    for mode in [WorkflowEmitMode::Retract, WorkflowEmitMode::Finalize] {
        let errors =
            validate_metadata(&fixture(mode.clone())).expect_err("payloadless value must fail");
        assert!(
            errors.iter().any(|error| {
                error.contains("pipeline.workflow.steps")
                    && error.contains("payload_that_must_not_be_discarded")
                    && error.contains(&format!("{mode:?}"))
                    && error.contains("stream 'named'")
                    && error.contains("payloadless")
            }),
            "{mode:?}: {errors:#?}"
        );
    }
}
