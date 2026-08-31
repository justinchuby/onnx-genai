use std::{fs, path::PathBuf};

use onnx_genai_metadata::{ComponentImplementation, InferenceMetadata, validate_metadata};

#[test]
fn every_catalogue_example_parses_and_validates() {
    let catalogue = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("examples/inference_metadata/catalogue");
    let mut examples = fs::read_dir(&catalogue)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", catalogue.display()))
        .map(|entry| entry.expect("catalogue entry").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "yaml")
        })
        .collect::<Vec<_>>();
    examples.sort();

    assert_eq!(
        examples.len(),
        25,
        "catalogue must cover all requested cases"
    );
    for path in examples {
        let document = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let metadata = serde_yaml::from_str::<InferenceMetadata>(&document)
            .unwrap_or_else(|error| panic!("{} did not parse: {error}", path.display()));
        validate_metadata(&metadata).unwrap_or_else(|errors| {
            panic!(
                "{} did not validate:\n{}",
                path.display(),
                errors.join("\n")
            )
        });

        let workflow = &metadata
            .pipeline
            .expect("catalogue example has a workflow")
            .workflow;
        if path
            .file_name()
            .is_some_and(|name| name == std::ffi::OsStr::new("01-gemma4-text-decoder.yaml"))
        {
            let full_key = workflow.state["full_key"].contract.shape.as_slice();
            let full_value = workflow.state["full_value"].contract.shape.as_slice();
            let sliding_key = workflow.state["sliding_key"].contract.shape.as_slice();
            assert_ne!(
                full_key[1], full_value[1],
                "K and V head geometry must remain independently representable"
            );
            assert_ne!(
                full_key[1], sliding_key[1],
                "different layers must not inherit one package-wide KV head count"
            );
        }
        if path
            .file_name()
            .is_some_and(|name| name == std::ffi::OsStr::new("23-gemma4-e2b-decoder.yaml"))
        {
            // This checkpoint is dense (no MoE); its real heterogeneity is head
            // WIDTH — the global head_dim is larger than the local head_dim.
            let full_key = workflow.state["full_key_0"].contract.shape.as_slice();
            let sliding_key = workflow.state["sliding_key_0"].contract.shape.as_slice();
            assert_ne!(
                full_key[3], sliding_key[3],
                "global and local layers must keep independent head widths"
            );
            assert!(
                metadata
                    .model
                    .as_ref()
                    .and_then(|model| model.mixture_of_experts.as_ref())
                    .is_none(),
                "the E2B target checkpoint is dense; no MoE metadata is invented"
            );
        }
        if path.file_name().is_some_and(|name| {
            name == std::ffi::OsStr::new("24-gemma4-e2b-assistant-speculative.yaml")
        }) {
            // The assistant reads the target's attention groups read-only and
            // never advances them; its resolved decode ABI therefore owns no KV.
            let assistant = onnx_genai_metadata::decoder_abi(workflow, "assistant")
                .expect("assistant ABI resolves");
            assert_eq!(
                assistant.kv_ownership,
                Some(onnx_genai_metadata::KvOwnership::Shared)
            );
            assert!(
                assistant.kv_inputs.is_none(),
                "read-only shares are not KV transitions"
            );
            assert!(matches!(
                metadata
                    .speculative
                    .as_ref()
                    .expect("speculative")
                    .proposal_execution,
                onnx_genai_metadata::SpeculativeProposalExecution::Chained { .. }
            ));
        }
        if path
            .file_name()
            .is_some_and(|name| name == std::ffi::OsStr::new("25-gemma4-26b-a4b-moe-decoder.yaml"))
        {
            // The 26B-A4B variant IS the MoE model — 128 routed + 1 shared
            // expert, 8 per token (verified from its pinned config).
            let moe = metadata
                .model
                .as_ref()
                .and_then(|model| model.mixture_of_experts.as_ref())
                .expect("26B-A4B declares a mixture-of-experts FFN");
            assert_eq!(moe.routed_expert_count, 128);
            assert_eq!(moe.experts_per_token, 8);
            assert_eq!(moe.shared_expert_count, 1);
            // Heterogeneous global/local geometry in BOTH axes: the global group
            // has fewer, wider KV heads than the local group.
            let full_key = workflow.state["full_key"].contract.shape.as_slice();
            let sliding_key = workflow.state["sliding_key"].contract.shape.as_slice();
            assert_ne!(
                full_key[1], sliding_key[1],
                "global vs local KV head count differs"
            );
            assert_ne!(
                full_key[3], sliding_key[3],
                "global vs local head width differs"
            );
        }
        for (name, component) in &workflow.components {
            if let ComponentImplementation::Onnx { artifact } = &component.implementation {
                assert!(
                    artifact.ends_with(".onnx.textproto"),
                    "{} component '{name}' must reference a textproto graph, got '{artifact}'",
                    path.display()
                );
            }
        }
    }
}
