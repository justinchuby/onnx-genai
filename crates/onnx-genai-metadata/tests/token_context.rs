//! Admission coverage for graph-internal, stateful token-context features.
//!
//! Executable numerical and lifecycle conformance lives in the engine's
//! `token_context_workflow` test; this file exercises only the portable
//! contract, semantic provenance, schema-version, and capability gates.

use std::collections::BTreeMap;

use onnx_genai_metadata::{
    BatchLayout, ComponentContract, ComponentImplementation, InferenceMetadata, PaddedDimension,
    PortRole, RuntimeInputRole, SemanticInputRole, ShapeRecurrence, StateGroupCapabilities,
    StateKind, StatePortAlias, StateSemanticRole, StateUpdate, TensorContract, TensorDimension,
    WorkflowBranchOutput, WorkflowComponent, WorkflowInput, WorkflowInputSource,
    WorkflowStateScope, WorkflowStep, classify_session_state, resolve_state_plan,
    validate_metadata,
};

const TOKEN_CONTEXT_CONTRACT: &str = "onnx-genai.token-context";

fn contract(dtype: &str, shape: &[&str]) -> TensorContract {
    TensorContract {
        dtype: dtype.to_string(),
        shape: shape
            .iter()
            .map(|dimension| TensorDimension::Symbol((*dimension).to_string()))
            .collect(),
        optional: false,
        batch_layout: BatchLayout::RequestAligned { axis: 0 },
        padding: Vec::new(),
    }
}

fn fixture() -> InferenceMetadata {
    let mut metadata: InferenceMetadata = serde_yaml::from_str(include_str!(
        "../../../examples/inference_metadata/catalogue/17-causal-convolution-recurrent.yaml"
    ))
    .expect("causal-convolution catalogue fixture parses");
    metadata.schema_version = Some("v1.4".to_string());
    let workflow = &mut metadata
        .pipeline
        .as_mut()
        .expect("catalogue fixture has pipeline")
        .workflow;
    workflow
        .inputs
        .get_mut("request.hidden_states")
        .expect("fixture has embedded sequence input")
        .contract = contract("float16", &["batch", "sequence", "hidden"]);

    workflow.inputs.insert(
        "request.token_ids".to_string(),
        WorkflowInput {
            contract: contract("int64", &["batch", "sequence"]),
            role: SemanticInputRole::Runtime {
                version: "1.0".to_string(),
                role: RuntimeInputRole::PromptTokens,
            },
            source: WorkflowInputSource::Request,
            required: true,
            default: None,
            present_as: None,
            externally_suppliable: false,
        },
    );

    let model = workflow
        .components
        .get_mut("model")
        .expect("catalogue fixture declares model");
    let mut embeds = model
        .ports
        .inputs
        .remove("hidden_states")
        .expect("model has hidden-state input");
    embeds.shape = vec![
        TensorDimension::Symbol("batch".to_string()),
        TensorDimension::Symbol("sequence".to_string()),
        TensorDimension::Symbol("hidden".to_string()),
    ];
    model
        .ports
        .inputs
        .insert("inputs_embeds".to_string(), embeds);
    model.ports.inputs.insert(
        "token_ids".to_string(),
        contract("int64", &["batch", "sequence"]),
    );
    model.ports.inputs.insert(
        "token_history".to_string(),
        contract("int64", &["batch", "token_history"]),
    );
    model.ports.outputs.insert(
        "next_token_history".to_string(),
        contract("int64", &["batch", "token_history"]),
    );
    model
        .ports
        .roles
        .insert("inputs_embeds".to_string(), PortRole::InputsEmbeds);
    model
        .ports
        .roles
        .insert("token_ids".to_string(), PortRole::TokenIds);
    model.contract = Some(ComponentContract {
        id: TOKEN_CONTEXT_CONTRACT.to_string(),
        version: "1".to_string(),
        equivalence: Default::default(),
        bindings: BTreeMap::new(),
        parameters: BTreeMap::new(),
    });

    let token_history = {
        let mut state = workflow
            .state
            .get("causal_conv_history")
            .expect("fixture has convolution history")
            .clone();
        state.contract = contract("int64", &["batch", "token_history"]);
        state.initializer = "request.token_history".to_string();
        state.scope = WorkflowStateScope::Session;
        state.release_boundary = Some(onnx_genai_metadata::StateReleaseBoundary::Session);
        state.service_group = Some("token_history".to_string());
        state.recurrence = ShapeRecurrence::Invariant;
        state
    };
    workflow
        .state
        .insert("token_history".to_string(), token_history);
    workflow.inputs.insert(
        "request.token_history".to_string(),
        WorkflowInput {
            contract: contract("int64", &["batch", "token_history"]),
            role: SemanticInputRole::Opaque,
            source: WorkflowInputSource::Application {
                name: "token_history".to_string(),
            },
            required: true,
            default: None,
            present_as: None,
            externally_suppliable: false,
        },
    );
    for state_name in ["causal_conv_history", "token_history"] {
        let state = workflow
            .state
            .get_mut(state_name)
            .expect("history state exists");
        state.scope = WorkflowStateScope::Session;
        state.release_boundary = Some(onnx_genai_metadata::StateReleaseBoundary::Session);
    }

    let mut token_group = workflow
        .serving
        .as_ref()
        .expect("fixture has serving contract")
        .state_service
        .groups
        .get("causal_conv_history")
        .expect("fixture has convolution state group")
        .clone();
    token_group.kind = StateKind::Recurrent;
    token_group.layout = "bt".to_string();
    token_group.update = Some(StateUpdate::Replace {});
    token_group.capabilities = StateGroupCapabilities {
        rollback_positions: None,
        snapshot: true,
        fork: true,
        cascade: Default::default(),
    };
    token_group.checkpoint = None;
    token_group.ports = BTreeMap::from([(
        "model".to_string(),
        BTreeMap::from([(
            "token_history".to_string(),
            StatePortAlias {
                input: "token_history".to_string(),
                output: Some("next_token_history".to_string()),
                access: Default::default(),
                role: None,
                layer: None,
            },
        )]),
    )]);
    let serving = workflow.serving.as_mut().expect("fixture has serving");
    serving
        .state_service
        .groups
        .get_mut("causal_conv_history")
        .expect("convolution state group exists")
        .capabilities = StateGroupCapabilities {
        rollback_positions: None,
        snapshot: true,
        fork: true,
        cascade: Default::default(),
    };
    serving
        .state_service
        .groups
        .get_mut("causal_conv_history")
        .expect("convolution state group exists")
        .checkpoint = None;
    serving
        .state_service
        .groups
        .insert("token_history".to_string(), token_group);

    let invoke = workflow
        .steps
        .iter_mut()
        .find_map(|step| match step {
            onnx_genai_metadata::WorkflowStep::Invoke {
                component,
                inputs,
                outputs,
            } if component == "model" => Some((inputs, outputs)),
            _ => None,
        })
        .expect("fixture invokes model");
    let (inputs, outputs) = invoke;
    let embeds = inputs
        .remove("hidden_states")
        .expect("model invocation feeds hidden states");
    inputs.insert("inputs_embeds".to_string(), embeds);
    inputs.insert("token_ids".to_string(), "request.token_ids".to_string());
    inputs.insert(
        "token_history".to_string(),
        "request.token_history".to_string(),
    );
    outputs.insert(
        "next_token_history".to_string(),
        "model.token_history".to_string(),
    );

    metadata
}

fn add_request_aligned_padding(metadata: &mut InferenceMetadata) {
    let workflow = &mut metadata.pipeline.as_mut().expect("pipeline").workflow;
    let padding = vec![PaddedDimension {
        dimension: "sequence".to_string(),
        valid_lengths: "request.token_lengths".to_string(),
    }];
    for input in ["request.hidden_states", "request.token_ids"] {
        workflow
            .inputs
            .get_mut(input)
            .expect("padded workflow input")
            .contract
            .padding = padding.clone();
    }
    workflow.inputs.insert(
        "request.token_lengths".to_string(),
        WorkflowInput {
            contract: contract("int64", &["batch"]),
            role: SemanticInputRole::Opaque,
            source: WorkflowInputSource::Application {
                name: "token_lengths".to_string(),
            },
            required: true,
            default: None,
            present_as: None,
            externally_suppliable: false,
        },
    );

    let model = workflow.components.get_mut("model").expect("model");
    let component_padding = vec![PaddedDimension {
        dimension: "sequence".to_string(),
        valid_lengths: "valid_lengths".to_string(),
    }];
    for input in ["inputs_embeds", "token_ids"] {
        model
            .ports
            .inputs
            .get_mut(input)
            .expect("padded component input")
            .padding = component_padding.clone();
    }
    model
        .ports
        .inputs
        .insert("valid_lengths".to_string(), contract("int64", &["batch"]));
    let inputs = workflow
        .steps
        .iter_mut()
        .find_map(|step| match step {
            WorkflowStep::Invoke {
                component, inputs, ..
            } if component == "model" => Some(inputs),
            _ => None,
        })
        .expect("model invoke");
    inputs.insert(
        "valid_lengths".to_string(),
        "request.token_lengths".to_string(),
    );
}

#[test]
fn padded_sequence_valid_lengths_follow_the_outer_request_axis() {
    let mut metadata = fixture();
    add_request_aligned_padding(&mut metadata);
    validate_metadata(&metadata).expect("request-aligned valid lengths are structurally valid");

    metadata
        .pipeline
        .as_mut()
        .expect("pipeline")
        .workflow
        .inputs
        .get_mut("request.token_lengths")
        .expect("valid lengths")
        .contract
        .batch_layout = BatchLayout::Shared;
    let errors =
        validate_metadata(&metadata).expect_err("shared valid lengths lose request-row ownership");
    assert!(
        errors.iter().any(|error| {
            error.contains("request.token_lengths")
                && error.contains("must declare request_aligned")
                && error.contains("preserve")
        }),
        "{errors:#?}"
    );
}

#[test]
fn token_context_requires_new_reader_and_runtime_admission() {
    let mut old = fixture();
    old.schema_version = Some("v1.3".to_string());
    let errors = validate_metadata(&old).expect_err("a v1.3 reader contract cannot carry S12");
    assert!(
        errors.iter().any(|error| {
            error.contains("onnx-genai.token-context")
                && error.contains("schema version v1.4")
                && error.contains("silently ignoring")
        }),
        "{errors:#?}"
    );
}

#[test]
fn token_context_invoke_rejects_shape_compatible_transformed_position_id_provenance() {
    let mut metadata = fixture();
    let workflow = &mut metadata.pipeline.as_mut().expect("pipeline").workflow;
    workflow.components.insert(
        "position_source".to_string(),
        WorkflowComponent {
            implementation: ComponentImplementation::Binding,
            ports: onnx_genai_metadata::ComponentPorts {
                inputs: BTreeMap::from([(
                    "value".to_string(),
                    contract("int64", &["batch", "sequence"]),
                )]),
                outputs: BTreeMap::from([(
                    "position_ids".to_string(),
                    contract("int64", &["batch", "sequence"]),
                )]),
                roles: BTreeMap::from([("position_ids".to_string(), PortRole::PositionIds)]),
            },
            contract: None,
            application_overridable: false,
            effects: Vec::new(),
            row_scope: None,
            cache_affects_state: Default::default(),
            batch_capacity: None,
        },
    );
    workflow.steps.insert(
        0,
        WorkflowStep::Invoke {
            component: "position_source".to_string(),
            inputs: BTreeMap::from([("value".to_string(), "request.token_ids".to_string())]),
            outputs: BTreeMap::from([(
                "position_ids".to_string(),
                "derived.position_ids".to_string(),
            )]),
        },
    );
    let model_invoke = workflow
        .steps
        .iter_mut()
        .find_map(|step| match step {
            WorkflowStep::Invoke {
                component, inputs, ..
            } if component == "model" => Some(inputs),
            _ => None,
        })
        .expect("model invoke");
    model_invoke.insert("token_ids".to_string(), "derived.position_ids".to_string());

    let errors = validate_metadata(&metadata)
        .expect_err("shape-compatible position IDs are not token identity");
    assert!(
        errors.iter().any(|error| {
            error.contains("token-context component 'model'")
                && error.contains("token_ids port 'token_ids'")
                && error.contains("derived.position_ids")
                && error.contains("position_ids")
                && error.contains("cannot distinguish token identity from position IDs")
        }),
        "{errors:#?}"
    );
}

#[test]
fn token_context_invoke_rejects_ambiguous_branch_provenance() {
    let mut metadata = fixture();
    let workflow = &mut metadata.pipeline.as_mut().expect("pipeline").workflow;
    workflow.inputs.insert(
        "choose_tokens".to_string(),
        WorkflowInput {
            contract: TensorContract {
                dtype: "bool".to_string(),
                shape: vec![TensorDimension::Fixed(1)],
                optional: false,
                batch_layout: BatchLayout::Shared,
                padding: Vec::new(),
            },
            role: SemanticInputRole::Opaque,
            source: WorkflowInputSource::Application {
                name: "choose_tokens".to_string(),
            },
            required: true,
            default: None,
            present_as: None,
            externally_suppliable: false,
        },
    );
    for (name, output_role) in [
        ("token_alias", PortRole::TokenIds),
        ("position_alias", PortRole::PositionIds),
    ] {
        workflow.components.insert(
            name.to_string(),
            WorkflowComponent {
                implementation: ComponentImplementation::Binding,
                ports: onnx_genai_metadata::ComponentPorts {
                    inputs: BTreeMap::from([(
                        "value".to_string(),
                        contract("int64", &["batch", "sequence"]),
                    )]),
                    outputs: BTreeMap::from([(
                        "value".to_string(),
                        contract("int64", &["batch", "sequence"]),
                    )]),
                    roles: BTreeMap::from([("value".to_string(), output_role)]),
                },
                contract: None,
                application_overridable: false,
                effects: Vec::new(),
                row_scope: None,
                cache_affects_state: Default::default(),
                batch_capacity: None,
            },
        );
    }
    workflow.steps.insert(
        0,
        WorkflowStep::Branch {
            predicate: "choose_tokens".to_string(),
            cases: BTreeMap::from([
                (
                    "true".to_string(),
                    WorkflowStep::Invoke {
                        component: "token_alias".to_string(),
                        inputs: BTreeMap::from([(
                            "value".to_string(),
                            "request.token_ids".to_string(),
                        )]),
                        outputs: BTreeMap::from([(
                            "value".to_string(),
                            "branch.token_ids".to_string(),
                        )]),
                    },
                ),
                (
                    "false".to_string(),
                    WorkflowStep::Invoke {
                        component: "position_alias".to_string(),
                        inputs: BTreeMap::from([(
                            "value".to_string(),
                            "request.token_ids".to_string(),
                        )]),
                        outputs: BTreeMap::from([(
                            "value".to_string(),
                            "branch.position_ids".to_string(),
                        )]),
                    },
                ),
            ]),
            default: None,
            outputs: BTreeMap::from([(
                "ambiguous.ids".to_string(),
                WorkflowBranchOutput {
                    cases: BTreeMap::from([
                        ("true".to_string(), "branch.token_ids".to_string()),
                        ("false".to_string(), "branch.position_ids".to_string()),
                    ]),
                    default: None,
                },
            )]),
        },
    );
    let model_inputs = workflow
        .steps
        .iter_mut()
        .find_map(|step| match step {
            WorkflowStep::Invoke {
                component, inputs, ..
            } if component == "model" => Some(inputs),
            _ => None,
        })
        .expect("model invoke");
    model_inputs.insert("token_ids".to_string(), "ambiguous.ids".to_string());

    let errors =
        validate_metadata(&metadata).expect_err("mixed token/position provenance is ambiguous");
    assert!(
        errors.iter().any(|error| {
            error.contains("ambiguous.ids")
                && error.contains("token_ids")
                && error.contains("position_ids")
                && error.contains("cannot distinguish token identity from position IDs")
        }),
        "{errors:#?}"
    );
}

#[test]
fn graph_internal_token_context_uses_generic_ports_and_state_groups() {
    let metadata = fixture();
    validate_metadata(&metadata).expect("token-context fixture validates");
    let workflow = &metadata.pipeline.expect("pipeline").workflow;
    let plan = resolve_state_plan(workflow);
    let carriers = classify_session_state(workflow);
    for history in ["token_history", "causal_conv_history"] {
        let state = plan.cell(history).expect("history appears in state plan");
        assert_eq!(state.lifecycle.scope, WorkflowStateScope::Session);
        assert_eq!(
            state.lifecycle.release,
            Some(onnx_genai_metadata::StateReleaseBoundary::Session)
        );
        assert!(state.snapshot.snapshot && state.snapshot.fork);
        assert!(state.transaction.required);
        assert_eq!(state.semantic_role, StateSemanticRole::TokenContextHistory);
        let service = state.service.as_ref().expect("history has a state service");
        assert!(service.snapshot && service.fork);
        assert_eq!(
            state.update,
            onnx_genai_metadata::StateUpdateRelation::Replace
        );
        assert!(
            state.final_writer.is_some(),
            "{history} has an unambiguous committed successor"
        );
        assert!(
            workflow
                .serving
                .as_ref()
                .expect("serving")
                .state_service
                .groups
                .get(history)
                .and_then(|group| group.checkpoint.as_ref())
                .is_none(),
            "{history} uses core runtime snapshot/fork semantics, not a portable checkpoint adapter"
        );
        assert_eq!(
            carriers.carrier(history),
            Some(onnx_genai_metadata::SessionStateCarrier::StateServiceGroup),
            "{history} uses the S10 state-service carrier for typed restore and commit"
        );
    }

    let schema = onnx_genai_metadata::inference_metadata_schema_json().expect("schema serializes");
    assert!(
        !schema.to_lowercase().contains("qwen"),
        "the conformance model name must never become schema vocabulary"
    );
}

#[test]
fn token_context_rejects_missing_or_mismatched_companions_before_execution() {
    let mut missing = fixture();
    let model = missing
        .pipeline
        .as_mut()
        .expect("pipeline")
        .workflow
        .components
        .get_mut("model")
        .expect("model");
    model.ports.roles.remove("token_ids");
    let errors = validate_metadata(&missing).expect_err("missing companion is rejected");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("token_ids companion") && error.contains("forbidden")),
        "{errors:#?}"
    );

    let mut mismatched = fixture();
    mismatched
        .pipeline
        .as_mut()
        .expect("pipeline")
        .workflow
        .components
        .get_mut("model")
        .expect("model")
        .ports
        .inputs
        .get_mut("token_ids")
        .expect("token ids")
        .shape = vec![
        TensorDimension::Symbol("batch".to_string()),
        TensorDimension::Symbol("different_sequence".to_string()),
    ];
    let errors = validate_metadata(&mismatched).expect_err("mismatched geometry is rejected");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("geometry") && error.contains("inputs_embeds")),
        "{errors:#?}"
    );

    let mut mismatched_padding = fixture();
    mismatched_padding
        .pipeline
        .as_mut()
        .expect("pipeline")
        .workflow
        .components
        .get_mut("model")
        .expect("model")
        .ports
        .inputs
        .get_mut("token_ids")
        .expect("token ids")
        .padding = vec![PaddedDimension {
        dimension: "sequence".to_string(),
        valid_lengths: "request.token_lengths".to_string(),
    }];
    let errors =
        validate_metadata(&mismatched_padding).expect_err("mismatched padding is rejected");
    assert!(
        errors.iter().any(|error| error.contains("padding")),
        "{errors:#?}"
    );

    let mut missing_state = fixture();
    missing_state
        .pipeline
        .as_mut()
        .expect("pipeline")
        .workflow
        .state
        .remove("token_history");
    let errors = validate_metadata(&missing_state).expect_err("missing history state is rejected");
    assert!(
        errors.iter().any(|error| error.contains("token_history")),
        "{errors:#?}"
    );

    let mut opaque_lookup = fixture();
    opaque_lookup
        .pipeline
        .as_mut()
        .expect("pipeline")
        .workflow
        .components
        .get_mut("model")
        .expect("model")
        .implementation = ComponentImplementation::Binding;
    let errors = validate_metadata(&opaque_lookup).expect_err("opaque lookup is rejected");
    assert!(
        errors
            .iter()
            .any(|error| error.contains("must be an ONNX component")),
        "{errors:#?}"
    );
}
