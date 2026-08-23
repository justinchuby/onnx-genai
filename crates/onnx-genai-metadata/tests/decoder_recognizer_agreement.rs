//! The two decoder recognizers are one classification, and this pins it.
//!
//! Metadata validation, the CLI, the server and the engine loader all used to
//! decide "is this package a single decoder?" for themselves. Two of those
//! decisions were written out longhand — `is_single_decoder_workflow` scanned
//! port roles, `Engine::decode_core_covers` scanned contract ids — and both
//! opened by re-deriving the same structural fact: how many of the declared
//! components name a graph. They agreed on every package anyone had looked at,
//! which is not the same as being unable to disagree.
//!
//! [`classify_workflow`] is now the only place either question is answered, and
//! the contract layer is *defined* as the role layer plus the contract. This
//! file is the evidence: an exhaustive matrix over every maintained fixture and
//! every catalogue example, a coverage guard so a new fixture cannot skip the
//! matrix, and the adversarial shapes no fixture has — extra policy bindings,
//! two decoders, a decoder missing the roles that drive it, a composite whose
//! text head is decoder-shaped, and the 187-component published package that
//! caught the original defect.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use onnx_genai_metadata::{
    ComponentImplementation, GraphCardinality, WorkflowSpec, classify_workflow,
    is_single_decoder_workflow, load_metadata, sole_decoder_component,
};

/// One row of the classification matrix.
///
/// `single_decoder` is layer 1 (declared port roles) and `contracted` is layer
/// 2 (layer 1 plus `onnx-genai.autoregressive-decode`, which is what the engine
/// loader routes on). Spelling both out per package is the point: a row where
/// `contracted` is `Some` and `single_decoder` is `false` would be the loader
/// and the metadata layer disagreeing about one package.
struct Row {
    relative: &'static str,
    graph_components: usize,
    cardinality: GraphCardinality,
    decoder: Option<&'static str>,
    single_decoder: bool,
    contracted: Option<&'static str>,
}

const fn row(
    relative: &'static str,
    graph_components: usize,
    cardinality: GraphCardinality,
    decoder: Option<&'static str>,
    single_decoder: bool,
    contracted: Option<&'static str>,
) -> Row {
    Row {
        relative,
        graph_components,
        cardinality,
        decoder,
        single_decoder,
        contracted,
    }
}

use GraphCardinality::{Composite, SingleGraph};

/// Every workflow this repository maintains, and how it classifies.
const MATRIX: &[Row] = &[
    // ── catalogue: worked examples of the metadata format ────────────────────
    // These document graph ABIs. Several declare a lone decoder component
    // without the runtime's decode contract, which is exactly why the two
    // layers are reported separately rather than collapsed into one flag.
    row(
        "examples/inference_metadata/catalogue/01-gemma4-text-decoder.yaml",
        1,
        SingleGraph,
        Some("decoder"),
        true,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/02-cosmos3-edge-rollout.yaml",
        1,
        SingleGraph,
        None,
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/03-qwen3_5-vlm.yaml",
        3,
        Composite,
        Some("decoder"),
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/04-whisper-encoder-decoder.yaml",
        2,
        Composite,
        Some("decoder"),
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/05-wav2vec2-ctc.yaml",
        1,
        SingleGraph,
        None,
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/06-personaplex-full-duplex.yaml",
        3,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/07-stable-diffusion-text-to-image.yaml",
        3,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/08-qwen-image-edit.yaml",
        4,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/09-cogvideox-text-to-video.yaml",
        3,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/10-lora-adapter-selection.yaml",
        2,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/11-speculative-proposer-verifier.yaml",
        2,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/12-esm2-protein-encoder.yaml",
        1,
        SingleGraph,
        None,
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/13-protbert-protein-encoder.yaml",
        1,
        SingleGraph,
        None,
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/14-weathernext-rollout.yaml",
        1,
        SingleGraph,
        None,
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/15-windowed-attention.yaml",
        2,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/16-linear-attention-recurrent.yaml",
        1,
        SingleGraph,
        None,
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/17-causal-convolution-recurrent.yaml",
        1,
        SingleGraph,
        None,
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/18-static-cache-indexed-scatter.yaml",
        1,
        SingleGraph,
        Some("decoder"),
        true,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/19-operator-abi-comparison.yaml",
        5,
        Composite,
        None,
        false,
        None,
    ),
    // Two decoders: the target and its draft. "Which one is the decoder" has no
    // single answer, and these paths address their components by name.
    row(
        "examples/inference_metadata/catalogue/20-qwen3_5-hybrid-speculative-decoding.yaml",
        2,
        Composite,
        Some("target_decoder"),
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/21-shared-prefix-pixel-flow.yaml",
        23,
        Composite,
        Some("decoder"),
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/22-qwen3-chained-speculative-decoding.yaml",
        2,
        Composite,
        Some("verifier"),
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/23-gemma4-e2b-decoder.yaml",
        1,
        SingleGraph,
        Some("decoder"),
        true,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/24-gemma4-e2b-assistant-speculative.yaml",
        2,
        Composite,
        Some("target"),
        false,
        None,
    ),
    row(
        "examples/inference_metadata/catalogue/25-gemma4-26b-a4b-moe-decoder.yaml",
        1,
        SingleGraph,
        Some("decoder"),
        true,
        None,
    ),
    // ── in-repo package fixtures ─────────────────────────────────────────────
    row(
        "tests/fixtures/comfyui_workflows/txt2img_sd15/inference_metadata.yaml",
        12,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "tests/fixtures/onnx_genai_workflows/adapter/inference_metadata.yaml",
        2,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "tests/fixtures/onnx_genai_workflows/codec/inference_metadata.yaml",
        2,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "tests/fixtures/onnx_genai_workflows/decoder/inference_metadata.yaml",
        11,
        Composite,
        Some("model"),
        false,
        None,
    ),
    row(
        "tests/fixtures/onnx_genai_workflows/diffusion/inference_metadata.yaml",
        11,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "tests/fixtures/onnx_genai_workflows/diffusion_guided/inference_metadata.yaml",
        14,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "tests/fixtures/onnx_genai_workflows/gemma4_chained/inference_metadata.yaml",
        2,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "tests/fixtures/onnx_genai_workflows/gemma4_chained_mixed/inference_metadata.yaml",
        2,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "tests/fixtures/onnx_genai_workflows/masked/inference_metadata.yaml",
        2,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "tests/fixtures/onnx_genai_workflows/speculative/inference_metadata.yaml",
        12,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "tests/fixtures/onnx_genai_workflows/speech_wav/inference_metadata.yaml",
        2,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "tests/fixtures/onnx_genai_workflows/speech_wav_mixed_audio/inference_metadata.yaml",
        2,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "tests/fixtures/onnx_genai_workflows/speech_wav_two_adapters/inference_metadata.yaml",
        3,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "tests/fixtures/onnx_genai_workflows/speech_wav_two_audio/inference_metadata.yaml",
        2,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "tests/fixtures/onnx_genai_workflows/static_cache/inference_metadata.yaml",
        5,
        Composite,
        Some("model"),
        false,
        None,
    ),
    row(
        "tests/fixtures/onnx_genai_workflows/tts/inference_metadata.yaml",
        27,
        Composite,
        None,
        false,
        None,
    ),
    row(
        "tests/fixtures/onnx_genai_workflows/video/inference_metadata.yaml",
        15,
        Composite,
        None,
        false,
        None,
    ),
    // A vision encoder, a projector and a decoder. Its decoder is recognizable;
    // the package is not a decoder.
    row(
        "tests/fixtures/onnx_genai_workflows/vlm/inference_metadata.yaml",
        14,
        Composite,
        Some("decoder"),
        false,
        None,
    ),
    row(
        "tests/fixtures/tiny-deepseek-v2-qmoe-attention/inference_metadata.yaml",
        1,
        SingleGraph,
        Some("decoder"),
        true,
        Some("decoder"),
    ),
    row(
        "tests/fixtures/tiny-deepseek-v4-qmoe/inference_metadata.yaml",
        11,
        Composite,
        Some("model"),
        false,
        None,
    ),
    row(
        "tests/fixtures/tiny-gemma4-assistant/inference_metadata.yaml",
        1,
        SingleGraph,
        Some("decoder"),
        true,
        Some("decoder"),
    ),
    row(
        "tests/fixtures/tiny-gemma4-assistant-mixed/inference_metadata.yaml",
        1,
        SingleGraph,
        Some("decoder"),
        true,
        Some("decoder"),
    ),
    row(
        "tests/fixtures/tiny-glm52-full-attention/inference_metadata.yaml",
        11,
        Composite,
        Some("model"),
        false,
        None,
    ),
    row(
        "tests/fixtures/tiny-glm52-qmoe-indexshare/inference_metadata.yaml",
        1,
        SingleGraph,
        Some("decoder"),
        true,
        Some("decoder"),
    ),
    row(
        "tests/fixtures/tiny-llm/inference_metadata.yaml",
        1,
        SingleGraph,
        Some("decoder"),
        true,
        Some("decoder"),
    ),
    row(
        "tests/fixtures/tiny-llm-explicit-io/inference_metadata.yaml",
        1,
        SingleGraph,
        Some("decoder"),
        true,
        Some("decoder"),
    ),
    row(
        "tests/fixtures/tiny-llm-scatter/inference_metadata.yaml",
        1,
        SingleGraph,
        Some("decoder"),
        true,
        Some("decoder"),
    ),
    // A metadata-only fixture: one ONNX component, no runtime token policy and
    // no decode contract. Layer 1 recognizes it, layer 2 declines it.
    row(
        "tests/fixtures/tiny-llm-scatter-workflow/inference_metadata.yaml",
        1,
        SingleGraph,
        Some("model"),
        true,
        None,
    ),
    row(
        "tests/fixtures/tiny-llm-sharedbuffer/inference_metadata.yaml",
        1,
        SingleGraph,
        Some("decoder"),
        true,
        Some("decoder"),
    ),
    row(
        "tests/fixtures/tiny-mtp-full/inference_metadata.yaml",
        1,
        SingleGraph,
        Some("decoder"),
        true,
        Some("decoder"),
    ),
    row(
        "tests/fixtures/tiny-native-engine/inference_metadata.yaml",
        1,
        SingleGraph,
        Some("decoder"),
        true,
        Some("decoder"),
    ),
    row(
        "tests/fixtures/tiny-native-scalar-gqa/inference_metadata.yaml",
        1,
        SingleGraph,
        Some("decoder"),
        true,
        Some("decoder"),
    ),
    row(
        "tests/fixtures/tiny-native-sub4-engine/inference_metadata.yaml",
        1,
        SingleGraph,
        Some("decoder"),
        true,
        Some("decoder"),
    ),
    row(
        "tests/fixtures/tiny-reasoning/inference_metadata.yaml",
        1,
        SingleGraph,
        Some("decoder"),
        true,
        Some("decoder"),
    ),
];

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn workflow_at(relative: &str) -> WorkflowSpec {
    let path = repository_root().join(relative);
    let metadata =
        load_metadata(&path).unwrap_or_else(|error| panic!("{relative}: must load: {error}"));
    metadata
        .pipeline
        .unwrap_or_else(|| panic!("{relative}: must declare a pipeline.workflow"))
        .workflow
}

/// Every maintained workflow classifies exactly as the matrix records.
#[test]
fn the_matrix_holds_for_every_maintained_workflow() {
    for expected in MATRIX {
        let workflow = workflow_at(expected.relative);
        let classification = classify_workflow(&workflow);
        let context = expected.relative;

        assert_eq!(
            classification.graph_component_count(),
            expected.graph_components,
            "{context}: graph component count",
        );
        assert_eq!(
            classification.cardinality(),
            expected.cardinality,
            "{context}: graph cardinality"
        );
        assert_eq!(
            classification.decoder_component(),
            expected.decoder,
            "{context}: recognized decoder component",
        );
        assert_eq!(
            classification.is_single_decoder(),
            expected.single_decoder,
            "{context}: layer 1 (declared roles)",
        );
        assert_eq!(
            classification.contracted_single_decoder(),
            expected.contracted,
            "{context}: layer 2 (roles plus the decode contract)",
        );
    }
}

/// The free functions are views onto the classification, not second answers.
///
/// `sole_decoder_component` and `is_single_decoder_workflow` are the names four
/// crates already call. If either ever grew its own scan again, this fails.
#[test]
fn the_free_functions_are_the_classification() {
    for expected in MATRIX {
        let workflow = workflow_at(expected.relative);
        let classification = classify_workflow(&workflow);
        assert_eq!(
            sole_decoder_component(&workflow),
            classification.decoder_component(),
            "{}: sole_decoder_component diverged from the classification",
            expected.relative,
        );
        assert_eq!(
            is_single_decoder_workflow(&workflow),
            classification.is_single_decoder(),
            "{}: is_single_decoder_workflow diverged from the classification",
            expected.relative,
        );
    }
}

/// The matrix covers every maintained workflow, so a new one cannot skip it.
#[test]
fn the_matrix_covers_every_maintained_workflow() {
    let root = repository_root();
    let mut found = BTreeSet::new();
    for directory in ["tests/fixtures", "examples/inference_metadata/catalogue"] {
        collect_workflows(&root.join(directory), &root, &mut found);
    }
    let recorded = MATRIX
        .iter()
        .map(|row| row.relative.to_string())
        .collect::<BTreeSet<_>>();

    let missing = found.difference(&recorded).cloned().collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "these workflows declare a pipeline but are not in the classification matrix; add a row \
         stating how each one classifies: {missing:#?}",
    );
    let stale = recorded.difference(&found).cloned().collect::<Vec<_>>();
    assert!(
        stale.is_empty(),
        "these matrix rows name workflows that no longer exist: {stale:#?}",
    );
}

fn collect_workflows(directory: &Path, root: &Path, found: &mut BTreeSet<String>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_workflows(&path, root, found);
            continue;
        }
        let is_catalogue_example = directory.ends_with("catalogue");
        let is_package_metadata = path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("inference_metadata"));
        if path.extension().is_none_or(|extension| extension != "yaml")
            || !(is_catalogue_example || is_package_metadata)
        {
            continue;
        }
        let declares_workflow = load_metadata(&path)
            .ok()
            .is_some_and(|metadata| metadata.pipeline.is_some());
        if declares_workflow {
            let relative = path.strip_prefix(root).unwrap_or(&path);
            found.insert(relative.to_string_lossy().replace('\\', "/"));
        }
    }
}

// ── the layering invariant ───────────────────────────────────────────────────

/// Layer 2 never admits a package layer 1 refuses.
///
/// This is the property the two hand-rolled recognizers could only be *observed*
/// to have. It is asserted over the maintained corpus and over every adversarial
/// shape below, because the shapes that could break it are exactly the ones no
/// fixture has.
fn assert_layers_are_nested(context: &str, workflow: &WorkflowSpec) {
    let classification = classify_workflow(workflow);
    if let Some(component) = classification.contracted_single_decoder() {
        assert!(
            classification.is_single_decoder(),
            "{context}: layer 2 admitted '{component}' while layer 1 refused the package",
        );
        assert_eq!(
            classification.decoder_component(),
            Some(component),
            "{context}: layer 2 named a component layer 1 does not call the decoder",
        );
        assert_eq!(
            classification.cardinality(),
            SingleGraph,
            "{context}: layer 2 admitted a package that is not one graph",
        );
        assert_eq!(
            classification.sole_graph_component(),
            Some(component),
            "{context}: layer 2 named something other than the package's only graph",
        );
    }
    assert!(
        !classification.decoder_evidence().contradictory()
            || classification.contracted_single_decoder().is_none(),
        "{context}: a contradictory declaration was still routed to the decode core",
    );
}

#[test]
fn the_layers_are_nested_for_every_maintained_workflow() {
    for expected in MATRIX {
        assert_layers_are_nested(expected.relative, &workflow_at(expected.relative));
    }
}

// ── adversarial shapes no fixture has ────────────────────────────────────────

/// The smallest complete single decoder: one ONNX graph, one runtime policy.
fn single_decoder() -> WorkflowSpec {
    workflow_at("tests/fixtures/tiny-llm/inference_metadata.yaml")
}

fn component_named(workflow: &WorkflowSpec, name: &str) -> onnx_genai_metadata::WorkflowComponent {
    workflow.components[name].clone()
}

/// Assert one adversarial workflow against the same five facts as the matrix.
fn assert_classifies(
    context: &str,
    workflow: &WorkflowSpec,
    graph_components: usize,
    cardinality: GraphCardinality,
    decoder: Option<&str>,
    single_decoder: bool,
    contracted: Option<&str>,
) {
    let classification = classify_workflow(workflow);
    assert_eq!(
        classification.graph_component_count(),
        graph_components,
        "{context}: graph component count",
    );
    assert_eq!(
        classification.cardinality(),
        cardinality,
        "{context}: graph cardinality"
    );
    assert_eq!(
        classification.decoder_component(),
        decoder,
        "{context}: recognized decoder component",
    );
    assert_eq!(
        classification.is_single_decoder(),
        single_decoder,
        "{context}: layer 1 (declared roles)",
    );
    assert_eq!(
        classification.contracted_single_decoder(),
        contracted,
        "{context}: layer 2 (roles plus the decode contract)",
    );
    assert_layers_are_nested(context, workflow);
}

/// Steps the runtime implements are not graphs, however many a package declares.
///
/// A package may declare a token policy, a stop policy and a constraint policy;
/// none of them is a second graph, so the decode core still covers the package.
#[test]
fn extra_policy_components_do_not_make_a_package_composite() {
    let mut workflow = single_decoder();
    let policy = component_named(&workflow, "token_policy");
    for extra in ["stop_policy", "constraint_policy"] {
        assert!(matches!(
            policy.implementation,
            ComponentImplementation::Binding
        ));
        workflow
            .components
            .insert(extra.to_string(), policy.clone());
    }
    assert_classifies(
        "a single decoder with three runtime policies",
        &workflow,
        1,
        SingleGraph,
        Some("decoder"),
        true,
        Some("decoder"),
    );
}

/// Two recognizable decoders is not "a decoder".
///
/// A speculative pair declares a target and a draft. Neither is *the* decoder,
/// and the fused executor holds one session, so both layers decline.
#[test]
fn two_decoders_are_never_a_single_decoder() {
    let mut workflow = single_decoder();
    let decoder = component_named(&workflow, "decoder");
    workflow.components.insert("draft".to_string(), decoder);
    assert_classifies(
        "a target and a draft decoder",
        &workflow,
        2,
        Composite,
        None,
        false,
        None,
    );
}

/// A decode contract with no roles behind it is refused, not routed.
///
/// The fused executor is driven by the resolved `DecoderAbi`, which is derived
/// from the declared roles. A component naming the decode step without them
/// promises a step nothing can drive; layer 2 declines instead of handing it
/// over to fail on an empty ABI.
#[test]
fn a_decode_contract_without_roles_is_refused() {
    let mut workflow = single_decoder();
    workflow
        .components
        .get_mut("decoder")
        .expect("the fixture declares a decoder")
        .ports
        .roles = BTreeMap::new();
    assert_classifies(
        "a decoder declaring the contract but no roles",
        &workflow,
        1,
        SingleGraph,
        None,
        false,
        None,
    );
    assert!(
        classify_workflow(&workflow)
            .decoder_evidence()
            .contradictory(),
        "a contract with no roles is a contradictory declaration, and the evidence says so",
    );
}

/// A decoder that owns attention state is a decoder without a logits role.
///
/// Recognition asks whether the component consumes the autoregressive sequence
/// and *either* scores it or advances the KV cache. A package that declares its
/// sequence input and its cache, and leaves the output role off, is still the
/// graph that decodes.
#[test]
fn a_decoder_without_a_logits_role_is_recognized_by_its_kv_ownership() {
    let mut workflow = single_decoder();
    let roles = &mut workflow
        .components
        .get_mut("decoder")
        .expect("the fixture declares a decoder")
        .ports
        .roles;
    roles.retain(|_, role| *role == onnx_genai_metadata::PortRole::TokenIds);
    assert_classifies(
        "a decoder declaring only its token input",
        &workflow,
        1,
        SingleGraph,
        Some("decoder"),
        true,
        Some("decoder"),
    );
}

/// A graph that consumes a sequence but neither scores it nor caches it is not
/// a decoder, so a package built only from those has none.
#[test]
fn a_graph_with_no_decoder_evidence_is_not_a_decoder() {
    let mut workflow = single_decoder();
    let decoder = workflow
        .components
        .get_mut("decoder")
        .expect("the fixture declares a decoder");
    decoder.ports.roles = BTreeMap::new();
    decoder.contract = None;
    assert_classifies(
        "a lone graph declaring neither roles nor a contract",
        &workflow,
        1,
        SingleGraph,
        None,
        false,
        None,
    );
}

/// A composite package has a decoder; it is not one.
#[test]
fn a_composite_with_a_decoder_like_component_is_not_a_single_decoder() {
    let mut workflow = single_decoder();
    let mut vision = component_named(&workflow, "decoder");
    vision.ports.roles = BTreeMap::new();
    vision.contract = None;
    workflow.components.insert("vision".to_string(), vision);
    assert_classifies(
        "a vision encoder beside a decoder",
        &workflow,
        2,
        Composite,
        Some("decoder"),
        false,
        None,
    );
}

/// The published 187-component package that caught the original defect.
///
/// An any-to-any model with one text head: exactly one component carries
/// `token_ids`/`logits` roles, so "has a recognizable decoder" is true of it.
/// Had that stood in for "is only a decoder", the loader would have handed 186
/// other graphs to a single-graph executor that cannot run them.
#[test]
fn a_187_component_package_with_one_text_head_is_composite() {
    let mut workflow = single_decoder();
    let mut encoder = component_named(&workflow, "decoder");
    encoder.ports.roles = BTreeMap::new();
    encoder.contract = None;
    for index in 0..186 {
        workflow
            .components
            .insert(format!("component_{index:03}"), encoder.clone());
    }
    assert_classifies(
        "a 187-component any-to-any package",
        &workflow,
        187,
        Composite,
        Some("decoder"),
        false,
        None,
    );
}

/// A runtime-implemented step cannot masquerade as the decoder.
///
/// Recognition considers only components that name something to execute. A
/// `binding` declaring `token_ids` and `logits` is a policy describing what it
/// consumes, not a second decoder — counting it would make the real decoder
/// ambiguous and drop a perfectly ordinary package to the interpreter.
#[test]
fn a_binding_declaring_decoder_roles_is_not_a_decoder() {
    let mut workflow = single_decoder();
    let decoder_roles = workflow.components["decoder"].ports.roles.clone();
    workflow
        .components
        .get_mut("token_policy")
        .expect("the fixture declares a token policy")
        .ports
        .roles = decoder_roles;
    assert_classifies(
        "a token policy wearing the decoder's roles",
        &workflow,
        1,
        SingleGraph,
        Some("decoder"),
        true,
        Some("decoder"),
    );
}

/// An adapter is an artifact something has to execute, so it is a graph.
#[test]
fn an_adapter_is_a_graph_component() {
    let mut workflow = single_decoder();
    let mut adapter = component_named(&workflow, "decoder");
    adapter.implementation = ComponentImplementation::Adapter {
        abi: "onnx-genai.lora".to_string(),
        version: "1".to_string(),
        artifact: Some("adapter.safetensors".to_string()),
    };
    adapter.ports.roles = BTreeMap::new();
    adapter.contract = None;
    workflow.components.insert("lora".to_string(), adapter);
    assert_classifies(
        "a decoder with a LoRA adapter",
        &workflow,
        2,
        Composite,
        Some("decoder"),
        false,
        None,
    );
}

/// A workflow of nothing but runtime-implemented steps names no graph at all.
#[test]
fn a_workflow_with_no_graph_component_is_no_graph() {
    let mut workflow = single_decoder();
    workflow
        .components
        .get_mut("decoder")
        .expect("the fixture declares a decoder")
        .implementation = ComponentImplementation::Binding;
    assert_classifies(
        "a workflow of bindings only",
        &workflow,
        0,
        GraphCardinality::NoGraph,
        None,
        false,
        None,
    );
}
