use onnx_genai_metadata::{InferenceMetadata, SpeculativeProposalExecution, validate_metadata};

fn fixture(name: &str) -> InferenceMetadata {
    let path = format!("tests/fixtures/canonical_speculation/{name}.yaml");
    serde_yaml::from_str(
        &std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {path}: {error}")),
    )
    .unwrap_or_else(|error| panic!("{path} did not parse: {error}"))
}

#[test]
fn candidate_tree_fixtures_declare_distinct_greedy_and_sampling_contracts() {
    let greedy = fixture("greedy_tree");
    validate_metadata(&greedy).expect("greedy tree fixture must validate");
    let error = greedy
        .speculative
        .as_ref()
        .expect("speculative contract")
        .admit_sampling()
        .expect_err("greedy fixture intentionally has no probability contract");
    assert!(
        error.contains("probabilities is absent") && error.contains("greedy verification"),
        "{error}"
    );

    let sampling = fixture("sampling_tree");
    validate_metadata(&sampling).expect("sampling tree fixture must validate");
    sampling
        .speculative
        .as_ref()
        .expect("speculative contract")
        .admit_sampling()
        .expect("proposal and target probabilities admit exact sampling");
}

#[test]
fn unknown_canonical_contract_versions_fail_before_execution() {
    let mut metadata = fixture("greedy_tree");
    metadata
        .speculative
        .as_mut()
        .expect("speculative contract")
        .version = "2".to_string();
    let errors = validate_metadata(&metadata).expect_err("unknown version must fail closed");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("speculative.version '2'")),
        "{errors:#?}"
    );
}

#[test]
fn mtp_uses_declared_target_ports_and_immutable_weight_owners() {
    let document =
        std::fs::read_to_string("../../tests/fixtures/tiny-mtp-full/inference_metadata.yaml")
            .expect("read reduced canonical MTP fixture");
    let mut metadata: InferenceMetadata =
        serde_yaml::from_str(&document).expect("reduced canonical MTP fixture parses");
    validate_metadata(&metadata).expect("reduced MTP fixture validates");

    let speculative = metadata.speculative.as_mut().expect("MTP contract");
    let SpeculativeProposalExecution::Mtp { target_hidden, .. } =
        &mut speculative.proposal_execution
    else {
        panic!("fixture must use the generic MTP proposal form");
    };
    target_hidden.output = "inferred_hidden_output".to_string();
    let errors = validate_metadata(&metadata).expect_err("a hidden output cannot be guessed");
    assert!(
        errors.iter().any(|error| {
            error.contains("target_hidden") && error.contains("inferred_hidden_output")
        }),
        "{errors:#?}"
    );
}
