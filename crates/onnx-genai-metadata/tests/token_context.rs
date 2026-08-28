//! Conformance coverage for graph-internal, stateful token-context features.
//!
//! The first fixture preserves the public Qwen3.8 Flash Next geometry that is
//! relevant to this ABI: eight hashes for each of bigrams and trigrams, whose
//! values are injected after the first decoder layer. Its lookup tables and
//! projected width are intentionally tiny reference vectors, not model weights.
//! The second fixture changes every structural parameter that this test can
//! change without changing the generic state/port path.

use std::collections::BTreeMap;

use onnx_genai_metadata::{
    BatchLayout, ComponentContract, ComponentImplementation, InferenceMetadata, PaddedDimension,
    PortRole, RuntimeInputRole, SemanticInputRole, ShapeRecurrence, StateCheckpointContract,
    StateGroupCapabilities, StateKind, StatePortAlias, StateUpdate, TensorContract,
    TensorDimension, WorkflowInput, WorkflowInputSource, WorkflowStateScope,
    classify_session_state, resolve_state_plan, validate_metadata,
};

const TOKEN_CONTEXT_CONTRACT: &str = "onnx-genai.token-context";

fn contract(dtype: &str, shape: &[&str]) -> TensorContract {
    TensorContract {
        dtype: dtype.to_string(),
        rank: shape.len(),
        shape: Some(
            shape
                .iter()
                .map(|dimension| TensorDimension::Symbol((*dimension).to_string()))
                .collect(),
        ),
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
    let workflow = &mut metadata
        .pipeline
        .as_mut()
        .expect("catalogue fixture has pipeline")
        .workflow;
    workflow
        .manifest
        .capabilities
        .insert("session_state_lease".to_string());
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
    embeds.shape = Some(vec![
        TensorDimension::Symbol("batch".to_string()),
        TensorDimension::Symbol("sequence".to_string()),
        TensorDimension::Symbol("hidden".to_string()),
    ]);
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
    token_group.update = Some(StateUpdate::Replace);
    token_group.capabilities = StateGroupCapabilities {
        rollback_positions: Some(4),
        snapshot: true,
        fork: true,
        cascade: Default::default(),
    };
    token_group.checkpoint = Some(StateCheckpointContract {
        adapter: "onnx-genai.tensor-checkpoint".to_string(),
        version: "1".to_string(),
    });
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
        rollback_positions: Some(4),
        snapshot: true,
        fork: true,
        cascade: Default::default(),
    };
    serving
        .state_service
        .groups
        .get_mut("causal_conv_history")
        .expect("convolution state group exists")
        .checkpoint = Some(StateCheckpointContract {
        adapter: "onnx-genai.tensor-checkpoint".to_string(),
        version: "1".to_string(),
    });
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
                .is_some(),
            "{history} declares portable checkpoint handling"
        );
        assert_eq!(
            carriers.carrier(history),
            Some(onnx_genai_metadata::SessionStateCarrier::StateServiceGroup),
            "{history} uses the S10 state-service carrier for restore, row compaction, and \
             speculative rollback"
        );
        assert_eq!(
            workflow
                .serving
                .as_ref()
                .expect("serving")
                .state_service
                .groups
                .get(history)
                .expect("history group")
                .capabilities
                .rollback_positions,
            Some(4)
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
        .shape = Some(vec![
        TensorDimension::Symbol("batch".to_string()),
        TensorDimension::Symbol("different_sequence".to_string()),
    ]);
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

#[derive(Clone)]
struct ContextGeometry {
    orders: Vec<usize>,
    hash_heads: usize,
    table_sizes: Vec<u64>,
    feature_width: usize,
    convolution: Vec<f64>,
    dilation: usize,
    gated_injection: bool,
    eos: i64,
}

#[derive(Clone, Default)]
struct ContextState {
    token_history: Vec<i64>,
    convolution_history: Vec<Vec<f64>>,
}

fn hash_ngram(tokens: &[i64], head: usize, table_size: u64) -> u64 {
    tokens.iter().fold(
        0x9e37_79b9_u64 ^ (head as u64).wrapping_mul(0x85eb_ca6b),
        |hash, token| {
            hash.wrapping_mul(0x1000_0000_01b3)
                .wrapping_add(*token as u64)
        },
    ) % table_size
}

fn lookup(head: usize, index: u64, lane: usize) -> f64 {
    ((index
        .wrapping_mul(17)
        .wrapping_add((head as u64).wrapping_mul(31))
        .wrapping_add((lane as u64).wrapping_mul(13))
        % 101) as f64
        - 50.0)
        / 64.0
}

fn token_context_step(
    geometry: &ContextGeometry,
    state: &mut ContextState,
    token: i64,
    base: &[f64],
) -> Vec<f64> {
    assert_eq!(base.len(), geometry.feature_width);
    let history_len = geometry.orders.iter().copied().max().unwrap_or(1) - 1;
    let mut projected = vec![0.0; geometry.feature_width];
    for (order_index, &order) in geometry.orders.iter().enumerate() {
        let needed_history = order - 1;
        let mut ngram = vec![0; needed_history.saturating_sub(state.token_history.len())];
        ngram.extend(
            state.token_history[state.token_history.len().saturating_sub(needed_history)..]
                .iter()
                .copied(),
        );
        ngram.push(token);
        for head in 0..geometry.hash_heads {
            let table = geometry.table_sizes[(order_index + head) % geometry.table_sizes.len()];
            let index = hash_ngram(&ngram, head, table);
            for (lane, value) in projected.iter_mut().enumerate() {
                *value += lookup(head, index, lane) / geometry.hash_heads as f64;
            }
        }
    }
    let mut convolved = vec![0.0; geometry.feature_width];
    for (tap, weight) in geometry.convolution.iter().enumerate() {
        let distance = tap * geometry.dilation;
        let source = if distance == 0 {
            Some(&projected)
        } else {
            state.convolution_history.iter().rev().nth(distance - 1)
        };
        let Some(source) = source else {
            continue;
        };
        for (output, input) in convolved.iter_mut().zip(source) {
            *output += weight * input;
        }
    }
    state.token_history.push(token);
    state
        .token_history
        .drain(..state.token_history.len().saturating_sub(history_len));
    state.convolution_history.push(projected.clone());
    let convolution_history_len = (geometry.convolution.len() - 1) * geometry.dilation;
    state.convolution_history.drain(
        ..state
            .convolution_history
            .len()
            .saturating_sub(convolution_history_len),
    );
    if token == geometry.eos {
        state.token_history.clear();
        state.convolution_history.clear();
    }
    base.iter()
        .zip(convolved)
        .zip(projected)
        .map(|((base, convolution), projection)| {
            let gate = 1.0 / (1.0 + (-projection).exp());
            base + if geometry.gated_injection {
                gate * convolution
            } else {
                convolution
            }
        })
        .collect()
}

fn run(
    geometry: &ContextGeometry,
    chunks: &[&[i64]],
    bases: &[Vec<f64>],
) -> (Vec<Vec<f64>>, ContextState) {
    let mut state = ContextState::default();
    let mut output = Vec::new();
    let mut base = bases.iter();
    for chunk in chunks {
        for token in *chunk {
            output.push(token_context_step(
                geometry,
                &mut state,
                *token,
                base.next().expect("one base vector per token"),
            ));
        }
    }
    (output, state)
}

const QWEN_CONFIG_DERIVED_REFERENCE: [[f64; 3]; 5] = [
    [
        0.0326519011683161,
        0.1966631835289126,
        0.239_298_516_870_579,
    ],
    [0.2827337261570716, 0.3593935013823889, 0.5386143673779539],
    [0.5487833938102045, 0.6271187612564469, 0.6994257791766749],
    [
        1.037_654_537_684_801,
        1.1698281494872536,
        1.1100500962510207,
    ],
    [1.1524213454607968, 1.2547488046864723, 1.4019607543557262],
];

fn assert_features_eq(left: &[Vec<f64>], right: &[Vec<f64>]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right) {
            assert!(
                (left - right).abs() < 1e-12,
                "feature mismatch: {left} != {right}"
            );
        }
    }
}

#[test]
fn token_context_full_chunked_and_decode_boundaries_are_equivalent() {
    let qwen_config_derived = ContextGeometry {
        // Qwen3.8 Flash Next uses eight raw-token bigram and eight trigram
        // hashes. The small tables/width below are reference vectors only.
        orders: vec![2, 3],
        hash_heads: 8,
        table_sizes: vec![31, 37, 41],
        feature_width: 3,
        convolution: vec![0.5, -0.25, 0.125],
        dilation: 1,
        gated_injection: true,
        eos: 2,
    };
    let synthetic_alternate = ContextGeometry {
        orders: vec![3, 4],
        hash_heads: 3,
        table_sizes: vec![7, 11, 13, 17],
        feature_width: 4,
        convolution: vec![0.75, -0.5],
        dilation: 2,
        gated_injection: false,
        eos: 9,
    };

    for (fixture_index, (geometry, tokens, bases)) in [
        (
            &qwen_config_derived,
            vec![5, 7, 11, 2, 13],
            vec![
                vec![0.0, 0.1, 0.2],
                vec![0.3, 0.4, 0.5],
                vec![0.6, 0.7, 0.8],
                vec![0.9, 1.0, 1.1],
                vec![1.2, 1.3, 1.4],
            ],
        ),
        (
            &synthetic_alternate,
            vec![3, 5, 7, 9, 11],
            vec![
                vec![0.0, 0.1, 0.2, 0.3],
                vec![0.4, 0.5, 0.6, 0.7],
                vec![0.8, 0.9, 1.0, 1.1],
                vec![1.2, 1.3, 1.4, 1.5],
                vec![1.6, 1.7, 1.8, 1.9],
            ],
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let (full, full_state) = run(geometry, &[&tokens], &bases);
        let (chunked, chunked_state) = run(
            geometry,
            &[&tokens[..2], &tokens[2..4], &tokens[4..]],
            &bases,
        );
        let (decoded, decoded_state) = run(
            geometry,
            &tokens.iter().map(std::slice::from_ref).collect::<Vec<_>>(),
            &bases,
        );
        assert_features_eq(&full, &chunked);
        assert_features_eq(&full, &decoded);
        if fixture_index == 0 {
            let expected = QWEN_CONFIG_DERIVED_REFERENCE
                .iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();
            assert_features_eq(&full, &expected);
        }
        assert_eq!(full_state.token_history, chunked_state.token_history);
        assert_eq!(full_state.token_history, decoded_state.token_history);
        assert_eq!(
            full_state.convolution_history,
            chunked_state.convolution_history
        );
        assert_eq!(
            full_state.convolution_history,
            decoded_state.convolution_history
        );
    }
}
