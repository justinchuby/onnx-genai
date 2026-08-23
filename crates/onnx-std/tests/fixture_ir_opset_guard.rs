//! Repo-wide guard: every maintained checked-in ONNX model fixture must declare
//! ONNX IR version >= 11 and a single default (`ai.onnx`) opset >= 24, and must
//! use only opset-24 node schemas (no removed attribute forms), recursively
//! through nested subgraphs AND function bodies.
//!
//! Model fixtures are committed either as protobuf TextFormat (`*.onnx.textproto`
//! / `*.textproto`) or as binary `*.onnx`, and are loaded by the runtime through
//! `onnx_std` (`textproto::to_binary` for text, `decode_model` for binary).
//! Keeping every fixture on a modern IR/opset floor ensures the exported models
//! exercise the same schema surface the runtime targets in production, and
//! prevents new fixtures from silently regressing to legacy IR 8 / opset 13.
//!
//! There is deliberately **no allowlist**: a fixture below the floor, one that
//! declares no (or more than one) default `ai.onnx` opset, or one that still uses
//! a removed attribute form is a hard failure. Fixtures that genuinely need a
//! legacy-opset graph must build it in-memory inside the specific compatibility
//! test rather than checking in an old model.

use std::path::{Path, PathBuf};
use std::process::Command;

use onnx_runtime_loader::proto::onnx::{FunctionProto, GraphProto, ModelProto, NodeProto};
use onnx_runtime_loader::proto::{decode_model, onnx::OperatorSetIdProto};

/// Minimum ONNX IR version for maintained fixtures (IR 11, ONNX 1.18 / 2025-05).
const MIN_IR_VERSION: i64 = 11;
/// Minimum default-domain ONNX opset for maintained fixtures (opset 24).
const MIN_DEFAULT_OPSET: i64 = 24;
/// Robust lower bound on the fixture inventory. The current tree tracks 229
/// model fixtures; a scan that collapses well below this (e.g. a wrong repo root
/// or an unavailable `git`) is treated as a hard failure rather than a vacuous
/// pass. Chosen with generous headroom so ordinary fixture churn never trips it.
const MIN_FIXTURE_COUNT: usize = 200;

/// Standard-domain ops whose attribute selector(s) moved to tensor **inputs** by
/// opset 24. The named attributes are illegal under the opset-24 schema. Covers
/// the axes/split families plus the earlier attribute->input migrations
/// (Pad/Clip/Slice/TopK/Resize/Upsample) so a stale fixture using any of these
/// forms is caught rather than silently accepted.
fn removed_attributes(op_type: &str) -> &'static [&'static str] {
    match op_type {
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" | "ReduceProd" | "ReduceL1"
        | "ReduceL2" | "ReduceLogSum" | "ReduceLogSumExp" | "ReduceSumSquare" | "Squeeze"
        | "Unsqueeze" => &["axes"],
        "Split" => &["split"],
        "Pad" => &["pads", "value"],
        "Clip" => &["min", "max"],
        "Slice" => &["starts", "ends", "axes", "steps"],
        "TopK" => &["k"],
        "Resize" => &["scales"],
        "Upsample" => &["scales"],
        _ => &[],
    }
}

fn is_standard_domain(domain: &str) -> bool {
    domain.is_empty() || domain == "ai.onnx"
}

/// Recursively collect opset-24 schema violations across a list of nodes and
/// every nested subgraph reachable through node attributes (If/Loop/Scan bodies,
/// etc.). Shared by top-level graphs and function bodies.
fn collect_legacy_forms(nodes: &[NodeProto], trail: &str, out: &mut Vec<String>) {
    for node in nodes {
        if is_standard_domain(&node.domain) {
            let removed: Vec<&str> = removed_attributes(&node.op_type)
                .iter()
                .copied()
                .filter(|name| node.attribute.iter().any(|attr| attr.name == *name))
                .collect();
            if !removed.is_empty() {
                let name = if node.name.is_empty() {
                    "<unnamed>"
                } else {
                    node.name.as_str()
                };
                out.push(format!(
                    "{trail}{} {name} uses removed attribute(s) {removed:?} (opset-24 requires them as inputs)",
                    node.op_type
                ));
            }
        }
        for attr in &node.attribute {
            let child_trail = format!("{trail}{}[{}]/", node.op_type, attr.name);
            if let Some(sub) = &attr.g {
                collect_legacy_forms(&sub.node, &child_trail, out);
            }
            for sub in &attr.graphs {
                collect_legacy_forms(&sub.node, &child_trail, out);
            }
        }
    }
}

/// Whether any standard-domain op appears in `nodes` or any nested subgraph.
fn uses_standard_domain_op(nodes: &[NodeProto]) -> bool {
    nodes.iter().any(|node| {
        is_standard_domain(&node.domain)
            || node.attribute.iter().any(|attr| {
                attr.g
                    .as_ref()
                    .is_some_and(|sub| uses_standard_domain_op(&sub.node))
                    || attr
                        .graphs
                        .iter()
                        .any(|sub| uses_standard_domain_op(&sub.node))
            })
    })
}

/// Resolve the single default (`ai.onnx`) opset version from a set of imports.
/// Collecting *all* default-domain imports (both the empty domain and its
/// `ai.onnx` alias) and requiring exactly one rejects duplicate/aliased imports
/// used to smuggle a stale default past a naive first-match check. When
/// `require_present` is false an omitted default import is allowed (`Ok(None)`).
fn check_default_opset(
    imports: &[OperatorSetIdProto],
    require_present: bool,
) -> Result<Option<i64>, String> {
    let defaults: Vec<i64> = imports
        .iter()
        .filter(|import| is_standard_domain(&import.domain))
        .map(|import| import.version)
        .collect();
    match defaults.as_slice() {
        [] if require_present => Err("no default (ai.onnx) opset import".to_string()),
        [] => Ok(None),
        [version] => Ok(Some(*version)),
        many => Err(format!(
            "{} default (ai.onnx) opset imports (expected exactly one): {many:?}",
            many.len()
        )),
    }
}

/// Apply the opset floor to a resolved default import, pushing a problem if it is
/// present but below the floor.
fn check_opset_floor(resolved: Result<Option<i64>, String>, prefix: &str, out: &mut Vec<String>) {
    match resolved {
        Ok(Some(version)) if version < MIN_DEFAULT_OPSET => {
            out.push(format!(
                "{prefix}default opset {version} < {MIN_DEFAULT_OPSET}"
            ));
        }
        Ok(_) => {}
        Err(message) => out.push(format!("{prefix}{message}")),
    }
}

/// All floor/schema problems for one model, including its function bodies. Empty
/// => the fixture is compliant.
fn model_problems(proto: &ModelProto) -> Vec<String> {
    let mut problems = Vec::new();
    if proto.ir_version < MIN_IR_VERSION {
        problems.push(format!(
            "ir_version {} < {MIN_IR_VERSION}",
            proto.ir_version
        ));
    }
    // Top-level graph: a default (ai.onnx) opset import is mandatory.
    check_opset_floor(
        check_default_opset(&proto.opset_import, true),
        "",
        &mut problems,
    );
    if let Some(graph) = &proto.graph {
        collect_legacy_forms(&graph.node, "", &mut problems);
    }
    // Function bodies (local functions) carry their own opset imports and nodes.
    for func in &proto.functions {
        let prefix = format!("function `{}`: ", func.name);
        // A function that calls standard-domain ops must pin exactly one
        // ai.onnx opset >= floor; a function using only custom-domain ops may
        // omit the default import.
        let require_default = uses_standard_domain_op(&func.node);
        check_opset_floor(
            check_default_opset(&func.opset_import, require_default),
            &prefix,
            &mut problems,
        );
        collect_legacy_forms(
            &func.node,
            &format!("function `{}`/", func.name),
            &mut problems,
        );
    }
    problems
}

/// Fail if the enumerated fixture inventory collapsed below the robust lower
/// bound (wrong repo root, unavailable `git`, or a scoped scan). Extracted so the
/// negative test exercises exactly this production check.
fn enforce_inventory(count: usize) -> Result<(), String> {
    if count < MIN_FIXTURE_COUNT {
        return Err(format!(
            "fixture inventory collapsed to {count} model file(s) (expected at least \
             {MIN_FIXTURE_COUNT}). This usually means the repo root was resolved incorrectly \
             or `git ls-files` returned nothing — the guard refuses to pass vacuously."
        ));
    }
    Ok(())
}

fn repo_root() -> PathBuf {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .expect("git must be available to locate the repository root");
    assert!(
        output.status.success(),
        "git rev-parse --show-toplevel failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let root = String::from_utf8(output.stdout).expect("git path is UTF-8");
    PathBuf::from(root.trim())
}

/// Tracked ONNX model fixtures under `dir`: textproto (`*.textproto`, which also
/// covers `*.onnx.textproto`) and binary `*.onnx`. External weight blobs
/// (`*.onnx.data`) and non-model placeholders (`*.onnx.fixture`) do not match.
fn tracked_model_fixtures(dir: &Path) -> Vec<String> {
    let output = Command::new("git")
        .args(["ls-files", "*.textproto", "*.onnx"])
        .current_dir(dir)
        .output()
        .expect("git must be available to enumerate tracked fixtures");
    assert!(
        output.status.success(),
        "git ls-files failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("git paths are UTF-8")
        .lines()
        .map(str::to_owned)
        .collect()
}

/// Decode the full `ModelProto` from a committed fixture. Text fixtures go
/// through the loader-compatible textproto->binary path; binary `*.onnx` are
/// decoded directly. Neither resolves external weights nor builds the runtime
/// graph, so this works for every fixture (including external-weight and
/// custom-domain models) and preserves nested subgraphs / functions.
fn decode_fixture(path: &Path) -> ModelProto {
    let is_textproto = path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("textproto"));
    let bytes = if is_textproto {
        let text = std::fs::read_to_string(path).expect("textproto fixture is readable");
        onnx_std::textproto::to_binary(&text)
            .unwrap_or_else(|error| panic!("{}: textproto parse failed: {error}", path.display()))
    } else {
        std::fs::read(path).expect("binary .onnx fixture is readable")
    };
    decode_model(&bytes)
        .unwrap_or_else(|error| panic!("{}: ModelProto decode failed: {error}", path.display()))
}

#[test]
fn maintained_fixtures_meet_ir_opset_and_schema_floor() {
    let root = repo_root();
    let fixtures = tracked_model_fixtures(&root);

    enforce_inventory(fixtures.len())
        .unwrap_or_else(|error| panic!("{error} Root was {}.", root.display()));

    let mut violations = Vec::new();
    for rel in &fixtures {
        let proto = decode_fixture(&root.join(rel));
        for problem in model_problems(&proto) {
            violations.push(format!("  {rel}: {problem}"));
        }
    }

    assert!(
        violations.is_empty(),
        "maintained ONNX model fixtures (graphs AND function bodies) must declare IR >= \
         {MIN_IR_VERSION}, exactly one default (ai.onnx) opset >= {MIN_DEFAULT_OPSET}, and use \
         only opset-24 node schemas (move ReduceMean/ReduceSum/Unsqueeze/Squeeze `axes`, Split \
         `split`, Pad/Clip/Slice/TopK/Resize/Upsample attributes to tensor inputs, including in \
         nested subgraphs). There is no allowlist; regenerate the fixture via its generator, or \
         build a legacy-opset graph in-memory inside the specific test that needs it.\n{}",
        violations.join("\n")
    );
}

#[cfg(test)]
mod negative_self_tests {
    use super::*;
    use onnx_runtime_loader::proto::onnx::{AttributeProto, NodeProto};

    fn node(op_type: &str, domain: &str, attrs: &[&str]) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            domain: domain.to_string(),
            attribute: attrs
                .iter()
                .map(|name| AttributeProto {
                    name: (*name).to_string(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    fn opset(domain: &str, version: i64) -> OperatorSetIdProto {
        OperatorSetIdProto {
            domain: domain.to_string(),
            version,
        }
    }

    fn model(
        ir: i64,
        imports: Vec<OperatorSetIdProto>,
        nodes: Vec<NodeProto>,
        functions: Vec<FunctionProto>,
    ) -> ModelProto {
        ModelProto {
            ir_version: ir,
            opset_import: imports,
            graph: Some(GraphProto {
                node: nodes,
                ..Default::default()
            }),
            functions,
            ..Default::default()
        }
    }

    fn function(imports: Vec<OperatorSetIdProto>, nodes: Vec<NodeProto>) -> FunctionProto {
        FunctionProto {
            name: "F".to_string(),
            opset_import: imports,
            node: nodes,
            ..Default::default()
        }
    }

    #[test]
    fn production_inventory_check_rejects_a_scoped_scan() {
        // Run the REAL scanner over a narrow in-repo directory; the production
        // `enforce_inventory` must reject the collapsed count, while the full
        // repo scan passes.
        let root = repo_root();
        let full = tracked_model_fixtures(&root).len();
        assert!(
            enforce_inventory(full).is_ok(),
            "full scan ({full}) must pass"
        );

        let narrow = root.join("crates/onnx-model-package/tests/fixtures/valid-package");
        let scoped = tracked_model_fixtures(&narrow).len();
        assert!(scoped > 0, "narrow scan should still find some fixtures");
        assert!(
            scoped < MIN_FIXTURE_COUNT,
            "narrow scan must be below the floor"
        );
        assert!(
            enforce_inventory(scoped).is_err(),
            "production inventory enforcement must reject the scoped scan"
        );
    }

    #[test]
    fn duplicate_or_aliased_default_imports_are_rejected() {
        assert!(check_default_opset(&[opset("", 24), opset("", 24)], true).is_err());
        assert!(check_default_opset(&[opset("", 24), opset("ai.onnx", 24)], true).is_err());
        assert!(check_default_opset(&[opset("", 13), opset("", 24)], true).is_err());
        assert!(check_default_opset(&[opset("com.microsoft", 1)], true).is_err());
        assert_eq!(
            check_default_opset(&[opset("", 24), opset("com.microsoft", 1)], true),
            Ok(Some(24))
        );
        // Omission is allowed only when no default import is required.
        assert_eq!(
            check_default_opset(&[opset("com.microsoft", 1)], false),
            Ok(None)
        );
    }

    #[test]
    fn ir_below_floor_is_rejected() {
        let problems = model_problems(&model(8, vec![opset("", 24)], vec![], vec![]));
        assert!(
            problems.iter().any(|p| p.contains("ir_version 8 < 11")),
            "IR < 11 must be flagged: {problems:?}"
        );
    }

    #[test]
    fn missing_or_stale_default_opset_is_rejected() {
        assert!(
            model_problems(&model(11, vec![], vec![], vec![]))
                .iter()
                .any(|p| p.contains("no default")),
            "missing default opset must be flagged"
        );
        assert!(
            model_problems(&model(11, vec![opset("", 13)], vec![], vec![]))
                .iter()
                .any(|p| p.contains("default opset 13 < 24")),
            "stale default opset must be flagged"
        );
    }

    #[test]
    fn legacy_attribute_forms_are_rejected() {
        for (op, attr) in [
            ("Unsqueeze", "axes"),
            ("Squeeze", "axes"),
            ("ReduceMean", "axes"),
            ("ReduceL2", "axes"),
            ("Split", "split"),
            ("Pad", "pads"),
            ("Clip", "max"),
            ("Slice", "starts"),
            ("TopK", "k"),
            ("Resize", "scales"),
            ("Upsample", "scales"),
        ] {
            let mut out = Vec::new();
            collect_legacy_forms(&[node(op, "", &[attr])], "", &mut out);
            assert_eq!(out.len(), 1, "{op} `{attr}` attribute must be flagged");
        }
    }

    #[test]
    fn custom_domain_nodes_are_not_false_positives() {
        // A com.microsoft op named like a standard reduce, with an `axes`
        // attribute, is NOT a standard-domain op and must not be flagged.
        let mut out = Vec::new();
        collect_legacy_forms(
            &[node("ReduceMean", "com.microsoft", &["axes"])],
            "",
            &mut out,
        );
        assert!(
            out.is_empty(),
            "custom-domain node must not be flagged: {out:?}"
        );
        // Input-form standard op (no removed attribute) is compliant.
        let mut out2 = Vec::new();
        collect_legacy_forms(&[node("Unsqueeze", "", &[])], "", &mut out2);
        assert!(out2.is_empty(), "input-form Unsqueeze must pass");
    }

    #[test]
    fn nested_subgraph_legacy_forms_are_rejected() {
        let inner = GraphProto {
            node: vec![node("Unsqueeze", "", &["axes"])],
            ..Default::default()
        };
        let if_node = NodeProto {
            op_type: "If".to_string(),
            attribute: vec![AttributeProto {
                name: "then_branch".to_string(),
                g: Some(inner),
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut out = Vec::new();
        collect_legacy_forms(&[if_node], "", &mut out);
        assert_eq!(out.len(), 1, "nested legacy Unsqueeze must be flagged");
        assert!(
            out[0].contains("If[then_branch]"),
            "trail names the subgraph: {}",
            out[0]
        );
    }

    #[test]
    fn function_legacy_attribute_is_rejected() {
        // A local function whose body hides a legacy Unsqueeze.
        let func = function(vec![opset("", 24)], vec![node("Unsqueeze", "", &["axes"])]);
        let problems = model_problems(&model(11, vec![opset("", 24)], vec![], vec![func]));
        assert!(
            problems
                .iter()
                .any(|p| p.contains("function `F`/") && p.contains("Unsqueeze")),
            "function-body legacy attribute must be flagged: {problems:?}"
        );
    }

    #[test]
    fn function_stale_or_duplicate_opset_imports_are_rejected() {
        // Stale default opset on the function.
        let stale = function(vec![opset("", 13)], vec![node("Add", "", &[])]);
        assert!(
            model_problems(&model(11, vec![opset("", 24)], vec![], vec![stale]))
                .iter()
                .any(|p| p.contains("function `F`: default opset 13 < 24")),
            "function stale opset must be flagged"
        );
        // Duplicate default imports on the function.
        let dup = function(
            vec![opset("", 24), opset("ai.onnx", 24)],
            vec![node("Add", "", &[])],
        );
        assert!(
            model_problems(&model(11, vec![opset("", 24)], vec![], vec![dup]))
                .iter()
                .any(|p| p.contains("function `F`:") && p.contains("expected exactly one")),
            "function duplicate opset imports must be flagged"
        );
        // A function using standard-domain ops but omitting the default import.
        let omitted = function(vec![opset("com.microsoft", 1)], vec![node("Add", "", &[])]);
        assert!(
            model_problems(&model(11, vec![opset("", 24)], vec![], vec![omitted]))
                .iter()
                .any(|p| p.contains("function `F`:") && p.contains("no default")),
            "function omitting a required default import must be flagged"
        );
        // A function using ONLY custom-domain ops may omit the default import.
        let custom_only = function(
            vec![opset("com.microsoft", 1)],
            vec![node("Attention", "com.microsoft", &[])],
        );
        assert!(
            model_problems(&model(11, vec![opset("", 24)], vec![], vec![custom_only])).is_empty(),
            "custom-only function may omit the default import"
        );
    }
}
